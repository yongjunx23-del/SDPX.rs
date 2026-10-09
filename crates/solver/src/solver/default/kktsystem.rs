use super::*;
use crate::solver::{
    cones::{CompositeCone, Cone},
    core::{
        traits::{KKTSystem, Settings},
        StepDirection,
    },
    kkt::{direct::*, *},
};

use crate::algebra::*;
use crate::mpi::all_succeeded;

// We require Send/Sync here to allow pyo3 builds to share
// solver objects between threads.

type BoxedKKTSolver<T> = Box<dyn KKTSolver<T> + Send + Sync>;

/// Standard-form solver type implementing the [`KKTSystem`](crate::solver::core::traits::KKTSystem) trait
pub struct DefaultKKTSystem<T> {
    kktsolver: BoxedKKTSolver<T>,

    // work vector for the homogeneous scalar equation
    workx: Vec<T>,
    // Two RHS columns; after a batch, their conic slices also hold step scratch.
    batch_rhs: Vec<T>,
    // Constant (x2,z2), then affine/corrector (x1,z1), in RHS column order.
    batch_out: Vec<T>,
    affine_ready: Option<bool>,
    hs2: Vec<T>,
    // Fixed-τ phase: the constant column (−q, b) is not solved and the
    // prepared affine output is batch column 0 instead of 1.
    fixed_tau: bool,
}

impl<T> DefaultKKTSystem<T>
where
    T: FloatT,
{
    pub(crate) fn uses_condensed(
        data: &DefaultProblemData<T>,
        cones: &CompositeCone<T>,
        settings: &DefaultSettings<T>,
    ) -> bool {
        match settings.kkt_form.as_str() {
            "condensed" => data.n > 0,
            "augmented" => false,
            _ => CondensedKKTSolver::prefer_condensed(&data.P, &data.A, cones, settings.core()),
        }
    }
    /// Create a new KKT system solver
    pub fn new(
        data: &DefaultProblemData<T>,
        cones: &CompositeCone<T>,
        settings: &DefaultSettings<T>,
    ) -> Self {
        let (m, n) = (data.m, data.n);

        // Both formulations share the embedding and recovery.
        let use_condensed = Self::uses_condensed(data, cones, settings);
        let augmented = || -> BoxedKKTSolver<T> {
            Box::new(DirectLDLKKTSolver::<T>::new(
                &data.P,
                &data.A,
                cones,
                m,
                n,
                settings.core(),
            ))
        };
        let mut kktsolver: BoxedKKTSolver<T> = if use_condensed {
            let fully_sampled = data
                .sampled
                .as_ref()
                .is_some_and(|op| op.covers_all_psd(&data.cones));
            let build = if fully_sampled {
                CondensedKKTSolver::<T>::new_fully_sampled
            } else {
                CondensedKKTSolver::<T>::new
            };
            Box::new(build(&data.P, &data.A, &data.cones, cones, settings.core()))
        } else {
            augmented()
        };

        if let Some(operator) = &data.sampled {
            kktsolver.set_sampled_operator(std::sync::Arc::clone(operator));
        }

        Self {
            kktsolver,
            workx: vec![T::zero(); n],
            batch_rhs: vec![T::zero(); 2 * (n + m)],
            batch_out: vec![T::zero(); 2 * (n + m)],
            affine_ready: None,
            hs2: Vec::new(),
            fixed_tau: false,
        }
    }
}

impl<T> HasLinearSolverInfo for DefaultKKTSystem<T>
where
    T: FloatT,
{
    fn linear_solver_info(&self) -> LinearSolverInfo {
        self.kktsolver.linear_solver_info()
    }
}

impl<T: FloatT> DefaultKKTSystem<T> {
    /// Per-solve factorization/RHS accounting for execution receipts.
    pub fn counters(&self) -> SolveCounters {
        self.kktsolver.counters()
    }
}

impl<T> KKTSystem<T> for DefaultKKTSystem<T>
where
    T: FloatT,
{
    type D = DefaultProblemData<T>;
    type V = DefaultVariables<T>;
    type C = CompositeCone<T>;
    type SE = DefaultSettings<T>;

    fn reset_solve(&mut self) {
        self.kktsolver.reset_solve();
        self.affine_ready = None;
    }

    fn update(
        &mut self,
        data: &DefaultProblemData<T>,
        cones: &CompositeCone<T>,
        settings: &DefaultSettings<T>,
    ) -> bool {
        self.affine_ready = None;
        // Update the linear solver with new cones and solve for the constant
        // terms.  On failure escalate the static regularization and retry:
        // unpivoted elimination can overflow pivots on degenerate quasidef
        // systems even though the shifted matrix is benign, and iterative
        // refinement still runs against the unshifted matrix.
        loop {
            let updated = all_succeeded(self.kktsolver.update(cones, settings.core()));
            let is_success = updated
                && (self.fixed_tau
                    || all_succeeded(self.solve_constant_rhs(data, settings.core())));
            if is_success
                || !settings.core().static_regularization_enable
                || !all_succeeded(self.kktsolver.escalate_regularization())
            {
                return is_success;
            }
        }
    }

    fn update_affine(
        &mut self,
        data: &DefaultProblemData<T>,
        cones: &CompositeCone<T>,
        rhs: &DefaultVariables<T>,
        variables: &DefaultVariables<T>,
        settings: &DefaultSettings<T>,
    ) -> bool {
        self.affine_ready = None;
        let (n, m) = (data.n, data.m);
        let width = n + m;
        self.fixed_tau = variables.fixed_tau;
        if self.fixed_tau {
            // Δτ = 0: only the affine column, prepared as batch column 0.
            self.hs2.clear();
            loop {
                let updated = all_succeeded(self.kktsolver.update(cones, settings.core()));
                let mut affine_ok = false;
                if updated {
                    let (rhs_x, rhs_z) = self.batch_rhs[width..].split_at_mut(n);
                    rhs_x.copy_from_slice(&rhs.x);
                    rhs_z.waxpby(T::one(), &variables.s, -T::one(), &rhs.z);
                    let ok = self.kktsolver.solve_many(
                        n,
                        &self.batch_rhs[width..],
                        &mut self.batch_out[width..],
                        1,
                        settings.core(),
                    );
                    affine_ok = all_succeeded(ok[0]);
                    if affine_ok {
                        self.affine_ready = Some(true);
                    }
                }
                if affine_ok
                    || !settings.core().static_regularization_enable
                    || !all_succeeded(self.kktsolver.escalate_regularization())
                {
                    return affine_ok;
                }
            }
        }
        loop {
            let updated = all_succeeded(self.kktsolver.update(cones, settings.core()));
            let mut constant_ok = false;
            if updated {
                for (v, &q) in self.batch_rhs[..n].iter_mut().zip(&data.q) {
                    *v = -q;
                }
                self.batch_rhs[n..width].copy_from_slice(&data.b);
                self.batch_rhs[width..width + n].copy_from_slice(&rhs.x);
                self.batch_rhs[width + n..].waxpby(T::one(), &variables.s, -T::one(), &rhs.z);
                let ok = self.kktsolver.solve_many(
                    n,
                    &self.batch_rhs,
                    &mut self.batch_out,
                    2,
                    settings.core(),
                );
                constant_ok = all_succeeded(ok[0]);
                let affine_ok = all_succeeded(ok[1]);
                if constant_ok {
                    copy_scaled(&*self.kktsolver, 0, &mut self.hs2);
                    self.affine_ready = Some(affine_ok);
                }
            }
            // Match the existing retry policy: only factor/constant failure
            // escalates regularization; affine failure remains a failed step.
            if constant_ok
                || !settings.core().static_regularization_enable
                || !all_succeeded(self.kktsolver.escalate_regularization())
            {
                return constant_ok;
            }
        }
    }

    fn solve(
        &mut self,
        lhs: &mut DefaultVariables<T>,
        rhs: &DefaultVariables<T>,
        data: &DefaultProblemData<T>,
        variables: &DefaultVariables<T>,
        cones: &mut CompositeCone<T>,
        step_direction: StepDirection,
        settings: &DefaultSettings<T>,
    ) -> bool {
        let (constant, variable) = self.batch_out.split_at_mut(data.n + data.m);
        let (x2, z2) = constant.split_at(data.n);
        let (x1, z1) = variable.split_at_mut(data.n);
        let (constant_rhs, variable_rhs) = self.batch_rhs.split_at_mut(data.n + data.m);
        let workx = &mut self.workx;
        let workz = &mut variable_rhs[data.n..];
        let Δs_const_term = &mut constant_rhs[data.n..];

        // solve for (x1,z1)
        // -----------
        workx.copy_from(&rhs.x);

        // compute the vector c in the step equation HₛΔz + Δs = -c,
        // with shortcut in affine case
        match step_direction {
            StepDirection::Affine => {
                Δs_const_term.copy_from(&variables.s);
            }
            StepDirection::Combined => {
                cones.Δs_from_Δz_offset(Δs_const_term, &rhs.s, &mut lhs.z, &variables.z);
            }
        }

        match step_direction {
            StepDirection::Affine => {
                workz.waxpby(T::one(), Δs_const_term, -T::one(), &rhs.z);
            }
            StepDirection::Combined => {
                workz.copy_from_slice(&rhs.z);
                for (v, &c) in workz.iter_mut().zip(Δs_const_term.iter()) {
                    *v = c - *v;
                }
            }
        }

        // ---------------------------------------------------
        // this solves the variable part of reduced KKT system
        let cached = self.affine_ready.take();
        let prepared = match step_direction {
            StepDirection::Affine => cached,
            StepDirection::Combined => None,
        };
        let is_success = prepared.unwrap_or_else(|| {
            self.kktsolver.setrhs(workx, workz);
            self.kktsolver
                .solve(Some(&mut *x1), Some(&mut *z1), settings.core())
        });
        if !all_succeeded(is_success) {
            return false;
        }

        // One shared homogeneous scalar formula, independent of storage layout.
        if variables.fixed_tau {
            lhs.τ = T::zero();
            lhs.x.copy_from(x1);
            lhs.z.copy_from(z1);
        } else {
            let terms = hsd_terms(workx, &variables.x, variables.τ, data, x1, z1, x2, z2);
            trace_dtau(&terms, rhs.τ, rhs.κ, variables.τ, variables.κ);
            lhs.τ = hsd_tau(terms, rhs.τ, rhs.κ, variables.τ, variables.κ);
            lhs.x.waxpby(T::one(), x1, lhs.τ, x2);
            lhs.z.waxpby(T::one(), z1, lhs.τ, z2);
        }

        // solve for Δs
        // -------------
        //  compute the linear term HₛΔz, where Hs = WᵀW for symmetric
        //  cones and Hs = μH(z) for asymmetric cones
        // Cached affine output is batch column 1; a fresh solve is column 0.
        let column = usize::from(prepared.is_some() && !variables.fixed_tau);
        let hs1 = self
            .kktsolver
            .scaled_solution(column)
            .filter(|_| T::precision_bits() > 53)
            .unwrap_or(&[]);
        if hs1.len() == data.m && variables.fixed_tau {
            lhs.s.copy_from(hs1);
        } else if hs1.len() == data.m && self.hs2.len() == data.m {
            // Linearity: reuse original-operator products from the two accepted
            // KKT solutions. No product from a rejected refinement is reused.
            lhs.s.waxpby(T::one(), hs1, lhs.τ, &self.hs2);
        } else {
            cones.mul_Hs(&mut lhs.s, &lhs.z, workz);
        }
        lhs.s.axpby(-T::one(), Δs_const_term, -T::one()); // lhs.s = -(lhs.s+Δs_const_term);

        // Binary64 condensed rows: the solve recovers Δz = H⁻¹(AΔx − bΔτ − r)
        // there, so take Δs from the same linear row, AΔx + Δs − bΔτ = −r_z,
        // as SDPT3 takes ΔZ from dual feasibility. Recomputing HΔz instead
        // rounds H·H⁻¹ at cond(H)·eps, which near convergence put errors of
        // 1e-3 into the primal row (SDP_qap6 and the gpp cases ended
        // AlmostSolved with primal residuals 1e-6 to 1e-5).
        if T::precision_bits() <= 53 && data.sampled.is_none() {
            if let Some(retained) = self.kktsolver.retained_rows() {
                workz.waxpby(-T::one(), &rhs.z, lhs.τ, &data.b);
                data.A.gemv(workz, &lhs.x, -T::one(), T::one());
                for &row in retained {
                    workz[row] = lhs.s[row];
                }
                lhs.s.copy_from(workz);
            }
        }

        // solve for Δκ
        // --------------
        lhs.κ = if variables.fixed_tau {
            T::zero()
        } else {
            -(rhs.κ + variables.κ * lhs.τ) / variables.τ
        };

        // we don't check the validity of anything
        // after the KKT solve, so just return is_success
        // without further validation
        is_success
    }

    fn solve_initial_point(
        &mut self,
        variables: &mut DefaultVariables<T>,
        data: &DefaultProblemData<T>,
        settings: &DefaultSettings<T>,
    ) -> bool {
        let mut is_success;
        let workx = &mut self.workx;
        let workz = &mut self.batch_rhs[2 * data.n + data.m..];

        if data.P.nnz() == 0 {
            // LP initialization
            // solve with [0;b] as a RHS to get (x,-s) initializers
            // zero out any sparse cone variables at end
            workx.fill(T::zero());
            workz.copy_from(&data.b);
            self.kktsolver.setrhs(workx, workz);
            is_success = self.kktsolver.solve(
                Some(&mut variables.x),
                Some(&mut variables.s),
                settings.core(),
            );
            variables.s.negate();

            if !all_succeeded(is_success) {
                return false;
            }

            // solve with [-q;0] as a RHS to get z initializer
            // zero out any sparse cone variables at end
            workx.axpby(-T::one(), &data.q, T::zero());
            workz.fill(T::zero());

            self.kktsolver.setrhs(workx, workz);
            is_success = self
                .kktsolver
                .solve(None, Some(&mut variables.z), settings.core());
        } else {
            //QP initialization
            workx.scalarop_from(|q| -q, &data.q);
            workz.copy_from(&data.b);
            self.kktsolver.setrhs(workx, workz);
            is_success = self.kktsolver.solve(
                Some(&mut variables.x),
                Some(&mut variables.z),
                settings.core(),
            );
            variables.s.scalarop_from(|z| -z, &variables.z);
        }
        all_succeeded(is_success && !initial_point_degenerate(variables, data))
    }
}

/// Contributions to the HSD tau equation before global scalar reduction.
#[derive(Clone, Copy)]
pub(crate) struct HsdTerms<T> {
    pub q1: T,
    pub b1: T,
    pub quad1: T,
    pub q2: T,
    pub b2: T,
    pub delta: T,
    pub quad2: T,
}
impl<T: FloatT> HsdTerms<T> {
    pub fn add(&mut self, rhs: Self) {
        self.q1 += rhs.q1;
        self.b1 += rhs.b1;
        self.quad1 += rhs.quad1;
        self.q2 += rhs.q2;
        self.b2 += rhs.b2;
        self.delta += rhs.delta;
        self.quad2 += rhs.quad2;
    }
}
pub(crate) fn hsd_terms<T: FloatT>(
    xi: &mut [T],
    x: &[T],
    tau: T,
    data: &DefaultProblemData<T>,
    x1: &[T],
    z1: &[T],
    x2: &[T],
    z2: &[T],
) -> HsdTerms<T> {
    xi.axpby(T::recip(tau), x, T::zero());
    let q1 = data.q.dot(x1);
    let b1 = data.b.dot(z1);
    let quad1 = data.P.sym_up().quad_form(xi, x1);
    xi.axpby(-T::one(), x2, T::one());
    HsdTerms {
        q1,
        b1,
        quad1,
        q2: data.q.dot(x2),
        b2: data.b.dot(z2),
        delta: data.P.sym_up().quad_form(xi, xi),
        quad2: data.P.sym_up().quad_form(x2, x2),
    }
}
/// Diagnostic only (`SDPX_TRACE_TAU`): the Δτ numerator and denominator and
/// their cancellation (sum of absolute terms over the absolute sum).
fn trace_dtau<T: FloatT>(v: &HsdTerms<T>, rt: T, rk: T, tau: T, kappa: T) {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| std::env::var_os("SDPX_TRACE_TAU").is_some()) {
        return;
    }
    let two = T::from_f64(2.).unwrap();
    let num = [rt, -rk / tau, v.q1, v.b1, two * v.quad1];
    let den = [kappa / tau, -v.q2, -v.b2, v.delta, -v.quad2];
    let sum = |a: &[T]| a.iter().fold(T::zero(), |s, &x| s + x);
    let abs = |a: &[T]| a.iter().fold(T::zero(), |s, &x| s + T::abs(x));
    let (n, d) = (sum(&num), sum(&den));
    eprintln!(
        "dtau-trace num {:.3e} cancel {:.3e} den {:.3e} cancel {:.3e} q1 {:.3e} b1 {:.3e}",
        n,
        abs(&num) / T::abs(n),
        d,
        abs(&den) / T::abs(d),
        v.q1,
        v.b1
    );
}

pub(crate) fn hsd_tau<T: FloatT>(v: HsdTerms<T>, rt: T, rk: T, tau: T, kappa: T) -> T {
    let numerator = rt - rk / tau + v.q1 + v.b1 + T::from_f64(2.).unwrap() * v.quad1;
    let mut denominator = kappa / tau - v.q2 - v.b2;
    denominator += v.delta - v.quad2;
    numerator / denominator
}

/// The solver's `H·z` for a solution column, used for `Δs` in place of
/// `mul_Hs`. Only above binary64: the solver's `H·z` is rounded along a
/// different path than the cones' `Wᵀ(W·z)`, and near convergence (an
/// ill-conditioned `W`) the gap is amplified by cond(W)²: measured up to 1e-2
/// in binary64 with a squared `G = WᵀW`, enough to corrupt `Δs`. At MPFR
/// precision it is negligible, and the reuse saves a full-precision product
/// per block; in binary64 `mul_Hs` is cheap.
fn copy_scaled<T: FloatT>(solver: &dyn KKTSolver<T>, column: usize, out: &mut Vec<T>) {
    if let Some(value) = solver
        .scaled_solution(column)
        .filter(|_| T::precision_bits() > 53)
    {
        out.resize(value.len(), T::zero());
        out.copy_from_slice(value);
    } else {
        out.clear();
    }
}

/// True when the initial KKT solve returned a point whose magnitude
/// dwarfs the problem data by orders of magnitude.  Direct solves on
/// severely ill-conditioned systems can produce components along
/// near-null directions that make every relative residual measure
/// meaningless; such a point is not a usable initializer.
fn initial_point_degenerate<T: FloatT>(
    variables: &DefaultVariables<T>,
    data: &DefaultProblemData<T>,
) -> bool {
    let scale = data
        .b
        .norm_inf()
        .max(data.q.norm_inf())
        .max(data.constraint_norm_inf())
        .max(T::one());
    let bound = T::from_f64(1e12).unwrap() * scale;
    !(variables.x.norm_inf() <= bound && variables.z.norm_inf() <= bound)
}

impl<T> DefaultKKTSystem<T>
where
    T: FloatT,
{
    fn solve_constant_rhs(
        &mut self,
        data: &DefaultProblemData<T>,
        settings: &DefaultSettings<T>,
    ) -> bool {
        let workx = &mut self.workx;
        let workz = &mut self.batch_rhs[2 * data.n + data.m..];
        workx.axpby(-T::one(), &data.q, T::zero()); //workx .= -q
        workz.copy_from_slice(&data.b);
        self.kktsolver.setrhs(workx, workz);
        let (x2, z2) = self.batch_out[..data.n + data.m].split_at_mut(data.n);
        let is_success = self.kktsolver.solve(Some(x2), Some(z2), settings.core());

        copy_scaled(&*self.kktsolver, 0, &mut self.hs2);
        is_success
    }

    pub(crate) fn update_P(&mut self, P: &CscMatrix<T>) {
        self.affine_ready = None;
        self.kktsolver.update_P(P);
    }

    pub(crate) fn update_A(&mut self, A: &CscMatrix<T>) {
        self.affine_ready = None;
        self.kktsolver.update_A(A);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::{CoreSettings, SupportedConeT::NonnegativeConeT};

    #[derive(Default)]
    struct Calls {
        updates: usize,
        singles: usize,
        batches: usize,
        retries: usize,
    }
    struct BatchProbe {
        calls: std::sync::Arc<std::sync::Mutex<Calls>>,
        fail_constant_once: bool,
        affine_ok: bool,
    }
    impl HasLinearSolverInfo for BatchProbe {
        fn linear_solver_info(&self) -> LinearSolverInfo {
            LinearSolverInfo::default()
        }
    }
    impl KKTSolver<f64> for BatchProbe {
        fn update(&mut self, _: &CompositeCone<f64>, _: &CoreSettings<f64>) -> bool {
            self.calls.lock().unwrap().updates += 1;
            true
        }
        fn setrhs(&mut self, _: &[f64], _: &[f64]) {}
        fn solve(
            &mut self,
            _: Option<&mut [f64]>,
            _: Option<&mut [f64]>,
            _: &CoreSettings<f64>,
        ) -> bool {
            self.calls.lock().unwrap().singles += 1;
            false
        }
        fn solve_many(
            &mut self,
            n: usize,
            rhs: &[f64],
            out: &mut [f64],
            cols: usize,
            _: &CoreSettings<f64>,
        ) -> Vec<bool> {
            assert_eq!((n, cols), (1, 2));
            assert_eq!(rhs, &[-1., 2., 3., 3.]);
            self.calls.lock().unwrap().batches += 1;
            out.fill(0.);
            vec![
                !std::mem::take(&mut self.fail_constant_once),
                self.affine_ok,
            ]
        }
        fn escalate_regularization(&mut self) -> bool {
            self.calls.lock().unwrap().retries += 1;
            true
        }
        fn update_P(&mut self, _: &CscMatrix<f64>) {}
        fn update_A(&mut self, _: &CscMatrix<f64>) {}
    }
    #[test]
    fn affine_batch_retry_and_cache_lifetime() {
        let settings = DefaultSettings::<f64>::default();
        let kinds = vec![NonnegativeConeT(1)];
        let data = DefaultProblemData::new(
            &CscMatrix::zeros((1, 1)),
            &[1.],
            &CscMatrix::identity(1),
            &[2.],
            &kinds,
            &settings,
        );
        let mut cones = CompositeCone::new(&kinds);
        let mut v = DefaultVariables::new(1, 1);
        v.s[0] = 5.;
        v.τ = 1.;
        v.κ = 1.;
        let mut rhs = DefaultVariables::new(1, 1);
        rhs.x[0] = 3.;
        rhs.z[0] = 2.;
        let mut lhs = DefaultVariables::new(1, 1);
        for affine_ok in [false, true] {
            let calls = std::sync::Arc::new(std::sync::Mutex::new(Calls::default()));
            let mut kkt = DefaultKKTSystem::new(&data, &cones, &settings);
            kkt.kktsolver = Box::new(BatchProbe {
                calls: calls.clone(),
                fail_constant_once: affine_ok,
                affine_ok,
            });
            assert!(kkt.update_affine(&data, &cones, &rhs, &v, &settings));
            assert_eq!(
                kkt.solve(
                    &mut lhs,
                    &rhs,
                    &data,
                    &v,
                    &mut cones,
                    StepDirection::Affine,
                    &settings
                ),
                affine_ok
            );
            let count = calls.lock().unwrap();
            assert_eq!(count.singles, 0);
            assert_eq!(count.retries, usize::from(affine_ok));
            assert_eq!(count.batches, 1 + usize::from(affine_ok));
            assert_eq!(count.updates, count.batches);
            drop(count);
            // The cached affine solve is consumed once, including a failed one.
            assert!(!kkt.solve(
                &mut lhs,
                &rhs,
                &data,
                &v,
                &mut cones,
                StepDirection::Affine,
                &settings
            ));
            assert_eq!(calls.lock().unwrap().singles, 1);
            assert!(kkt.update_affine(&data, &cones, &rhs, &v, &settings));
            kkt.reset_solve();
            assert!(kkt.affine_ready.is_none());
        }
    }

    /// Run alone under a two-rank launcher with --ignored --test-threads=1.
    /// The mock uses a real, separate collective stream: a missing agreement
    /// before retry/solve either mismatches an event or hits the launch timeout.
    #[test]
    #[ignore]
    fn mpi_probe_two_rank_kkt_failure_retry() {
        use std::sync::{Arc, Mutex};

        struct FaultProbe {
            calls: Arc<Mutex<Vec<u32>>>,
            fail_update_once: bool,
            fail_constant_once: bool,
            retries: usize,
        }
        impl FaultProbe {
            fn event(&self, event: u32) {
                let world = crate::mpi::World::get().unwrap();
                let mut received = [0.0; 2];
                world.gather_slice(
                    crate::mpi::SITE_GRAM,
                    &[f64::from(event)],
                    &[(0, 1), (1, 1)],
                    &mut received,
                );
                if received != [f64::from(event); 2] {
                    world.abort("KKT fault probe observed different call sequences");
                }
                self.calls.lock().unwrap().push(event);
            }
        }
        impl HasLinearSolverInfo for FaultProbe {
            fn linear_solver_info(&self) -> LinearSolverInfo {
                LinearSolverInfo::default()
            }
        }
        impl KKTSolver<f64> for FaultProbe {
            fn update(&mut self, _: &CompositeCone<f64>, _: &CoreSettings<f64>) -> bool {
                self.event(1);
                !std::mem::take(&mut self.fail_update_once)
            }
            fn setrhs(&mut self, _: &[f64], _: &[f64]) {}
            fn solve(
                &mut self,
                x: Option<&mut [f64]>,
                z: Option<&mut [f64]>,
                _: &CoreSettings<f64>,
            ) -> bool {
                self.event(2);
                if let Some(x) = x {
                    x.fill(0.0);
                }
                if let Some(z) = z {
                    z.fill(0.0);
                }
                !std::mem::take(&mut self.fail_constant_once)
            }
            fn solve_many(
                &mut self,
                _: usize,
                _: &[f64],
                out: &mut [f64],
                cols: usize,
                _: &CoreSettings<f64>,
            ) -> Vec<bool> {
                self.event(3);
                assert_eq!(cols, 2);
                out.fill(0.0);
                vec![!std::mem::take(&mut self.fail_constant_once), true]
            }
            fn escalate_regularization(&mut self) -> bool {
                self.event(4);
                self.retries += 1;
                self.retries <= 2
            }
            fn update_P(&mut self, _: &CscMatrix<f64>) {}
            fn update_A(&mut self, _: &CscMatrix<f64>) {}
        }

        let mpi = crate::MpiContext::initialize();
        assert_eq!(mpi.size(), 2, "launch exactly this test on two ranks");
        let settings = DefaultSettings::<f64>::default();
        assert!(settings.static_regularization_enable);
        let kinds = vec![NonnegativeConeT(1)];
        let data = DefaultProblemData::new(
            &CscMatrix::zeros((1, 1)),
            &[1.],
            &CscMatrix::identity(1),
            &[2.],
            &kinds,
            &settings,
        );
        let cones = CompositeCone::new(&kinds);
        let variables = DefaultVariables::new(1, 1);
        let rhs = DefaultVariables::new(1, 1);
        let install = |fail_factor: bool, fail_constant: bool| {
            let calls = Arc::new(Mutex::new(Vec::new()));
            let mut kkt = DefaultKKTSystem::new(&data, &cones, &settings);
            kkt.kktsolver = Box::new(FaultProbe {
                calls: calls.clone(),
                fail_update_once: mpi.rank() == 0 && fail_factor,
                fail_constant_once: mpi.rank() == 0 && fail_constant,
                retries: 0,
            });
            (kkt, calls)
        };
        // Exercise both public update paths. Only rank zero fails; both ranks
        // must skip the same solve or retry the same constant RHS exactly once.
        for batched in [false, true] {
            for factor_failure in [true, false] {
                let (mut kkt, calls) = install(factor_failure, !factor_failure);
                let ok = if batched {
                    kkt.update_affine(&data, &cones, &rhs, &variables, &settings)
                } else {
                    kkt.update(&data, &cones, &settings)
                };
                let solve_event = if batched { 3 } else { 2 };
                let expected = if factor_failure {
                    vec![1, 4, 1, solve_event]
                } else {
                    vec![1, solve_event, 4, 1, solve_event]
                };
                let matches = ok
                    && *calls.lock().unwrap() == expected
                    && (!batched || kkt.affine_ready == Some(true));
                if !mpi.all_succeeded(matches) {
                    mpi.abort("KKT fault probe retry count or affine cache mismatch");
                }
            }
        }
        // LP initialization has two sequential solves. A first-solve failure
        // on rank zero must suppress the second solve on BOTH ranks.
        let (mut kkt, calls) = install(false, true);
        let mut point = DefaultVariables::new(1, 1);
        let ok = kkt.solve_initial_point(&mut point, &data, &settings);
        if !mpi.all_succeeded(!ok && *calls.lock().unwrap() == [2]) {
            mpi.abort("KKT fault probe entered second initializer after peer failure");
        }
        // Confirm that the non-failing initializer still performs both solves.
        let (mut kkt, calls) = install(false, false);
        let ok = kkt.solve_initial_point(&mut point, &data, &settings);
        if !mpi.all_succeeded(ok && *calls.lock().unwrap() == [2, 2]) {
            mpi.abort("KKT fault probe changed successful initialization");
        }
        mpi.finish();
    }

    #[test]
    fn degenerate_initial_point_detected() {
        let P = CscMatrix::<f64>::identity(2);
        let q = vec![1.0, 0.0];
        let A = CscMatrix::<f64>::identity(2);
        let b = vec![1.0, 1.0];
        let cones = vec![NonnegativeConeT(2)];
        let settings = DefaultSettings::default();
        let data = DefaultProblemData::new(&P, &q, &A, &b, &cones, &settings);

        let mut v = DefaultVariables::new(2, 2);
        v.x.fill(1.0);
        v.z.fill(1.0);
        assert!(!initial_point_degenerate(&v, &data));

        // a KKT initializer with components ~1e20 on unit-scale data
        // is numerically meaningless and must be rejected
        v.x[0] = 1e20;
        assert!(initial_point_degenerate(&v, &data));

        v.x[0] = 1.0;
        v.z[1] = -1e20;
        assert!(initial_point_degenerate(&v, &data));

        // NaN / non-finite initializers are equally unusable
        v.z[1] = 1.0;
        v.x[0] = f64::NAN;
        assert!(initial_point_degenerate(&v, &data));
    }
}
