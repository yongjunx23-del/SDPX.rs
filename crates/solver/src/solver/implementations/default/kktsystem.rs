use super::*;
use crate::solver::core::{
    cones::{CompositeCone, Cone},
    kktsolvers::{direct::*, *},
    traits::{KKTSystem, Settings},
    StepDirection,
};

use crate::algebra::*;

// We require Send/Sync here to allow pyo3 builds to share
// solver objects between threads.

type BoxedKKTSolver<T> = Box<dyn KKTSolver<T> + Send + Sync>;

/// Standard-form solver type implementing the [`KKTSystem`](crate::solver::core::traits::KKTSystem) trait
pub struct DefaultKKTSystem<T> {
    kktsolver: BoxedKKTSolver<T>,

    // solution vector for constant part of KKT solves
    x1: Vec<T>,
    z1: Vec<T>,

    // solution vector for general KKT solves
    x2: Vec<T>,
    z2: Vec<T>,

    // work vectors for assembling/dissambling vectors
    workx: Vec<T>,
    workz: Vec<T>,
    work_conic: Vec<T>,
}

impl<T> DefaultKKTSystem<T>
where
    T: FloatT,
{
    /// Create a new KKT system solver
    pub fn new(
        data: &DefaultProblemData<T>,
        cones: &CompositeCone<T>,
        settings: &DefaultSettings<T>,
    ) -> Self {
        let (m, n) = (data.m, data.n);

        // Both formulations share the embedding and recovery.
        #[cfg(feature = "sdp")]
        let use_condensed = match settings.kkt_form.as_str() {
            // A constant feasibility problem has no primal block to condense.
            "condensed" => data.n > 0,
            "augmented" => false,
            _ => CondensedKKTSolver::prefer_condensed(&data.P, &data.A, cones, settings.core()),
        };
        assert!(
            settings.direct_kkt_solver,
            "Indirect solve strategies are not supported."
        );
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
        #[cfg(feature = "sdp")]
        let mut kktsolver: BoxedKKTSolver<T> = if use_condensed {
            Box::new(CondensedKKTSolver::<T>::new(
                &data.P,
                &data.A,
                &data.cones,
                cones,
                settings.core(),
            ))
        } else {
            augmented()
        };
        #[cfg(not(feature = "sdp"))]
        let kktsolver = augmented();

        #[cfg(feature = "sdp")]
        if let Some(operator) = &data.sampled {
            kktsolver.set_sampled_operator(std::sync::Arc::clone(operator));
        }

        //the LHS constant part of the reduced solve
        let x1 = vec![T::zero(); n];
        let z1 = vec![T::zero(); m];

        //the LHS for other solves
        let x2 = vec![T::zero(); n];
        let z2 = vec![T::zero(); m];

        //workspace compatible with (x,z)
        let workx = vec![T::zero(); n];
        let workz = vec![T::zero(); m];

        //additional conic workspace vector compatible with s and z
        let work_conic = vec![T::zero(); m];

        Self {
            kktsolver,
            x1,
            z1,
            x2,
            z2,
            workx,
            workz,
            work_conic,
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

    fn update(
        &mut self,
        data: &DefaultProblemData<T>,
        cones: &CompositeCone<T>,
        settings: &DefaultSettings<T>,
    ) -> bool {
        // Update the linear solver with new cones and solve for the constant
        // terms.  On failure escalate the static regularization and retry:
        // unpivoted elimination can overflow pivots on degenerate quasidef
        // systems even though the shifted matrix is benign, and iterative
        // refinement still runs against the unshifted matrix.
        loop {
            let is_success = self.kktsolver.update(cones, settings.core())
                && self.solve_constant_rhs(data, settings.core());
            if is_success || !self.kktsolver.escalate_regularization() {
                return is_success;
            }
        }

        //PJG is_success should be a Result in rust
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
        let (x1, z1) = (&mut self.x1, &mut self.z1);
        let (x2, z2) = (&self.x2, &self.z2); //from constant solve, so not mut
        let (workx, workz) = (&mut self.workx, &mut self.workz);

        // solve for (x1,z1)
        // -----------
        workx.copy_from(&rhs.x);

        // compute the vector c in the step equation HₛΔz + Δs = -c,
        // with shortcut in affine case
        let Δs_const_term = &mut self.work_conic;

        match step_direction {
            StepDirection::Affine => {
                Δs_const_term.copy_from(&variables.s);
            }
            StepDirection::Combined => {
                cones.Δs_from_Δz_offset(Δs_const_term, &rhs.s, &mut lhs.z, &variables.z);
            }
        }

        workz.waxpby(T::one(), Δs_const_term, -T::one(), &rhs.z);

        // ---------------------------------------------------
        // this solves the variable part of reduced KKT system
        self.kktsolver.setrhs(workx, workz);
        let is_success = self.kktsolver.solve(Some(x1), Some(z1), settings.core());
        if !is_success {
            return false;
        }

        // solve for Δτ.
        // -----------
        // Numerator first
        let ξ = workx;
        ξ.axpby(T::recip(variables.τ), &variables.x, T::zero());

        let two: T = (2.).as_T();
        let tau_num = rhs.τ - rhs.κ / variables.τ
            + data.q.dot(x1)
            + data.b.dot(z1)
            + two * data.P.sym_up().quad_form(ξ, x1);

        // offset ξ for the quadratic form in the denominator
        let ξ_minus_x2 = ξ; //alias to ξ, same as workx
        ξ_minus_x2.axpby(-T::one(), x2, T::one());

        let mut tau_den = variables.κ / variables.τ - data.q.dot(x2) - data.b.dot(z2);
        tau_den +=
            data.P.sym_up().quad_form(ξ_minus_x2, ξ_minus_x2) - data.P.sym_up().quad_form(x2, x2);

        // solve for (Δx,Δz)
        // -----------
        lhs.τ = tau_num / tau_den;
        lhs.x.waxpby(T::one(), x1, lhs.τ, x2);
        lhs.z.waxpby(T::one(), z1, lhs.τ, z2);

        // solve for Δs
        // -------------
        //  compute the linear term HₛΔz, where Hs = WᵀW for symmetric
        //  cones and Hs = μH(z) for asymmetric cones
        cones.mul_Hs(&mut lhs.s, &lhs.z, workz);
        lhs.s.axpby(-T::one(), Δs_const_term, -T::one()); // lhs.s = -(lhs.s+Δs_const_term);

        // solve for Δκ
        // --------------
        lhs.κ = -(rhs.κ + variables.κ * lhs.τ) / variables.τ;

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

        if data.P.nnz() == 0 {
            // LP initialization
            // solve with [0;b] as a RHS to get (x,-s) initializers
            // zero out any sparse cone variables at end
            self.workx.fill(T::zero());
            self.workz.copy_from(&data.b);
            self.kktsolver.setrhs(&self.workx, &self.workz);
            is_success = self.kktsolver.solve(
                Some(&mut variables.x),
                Some(&mut variables.s),
                settings.core(),
            );
            variables.s.negate();

            if !is_success {
                return is_success;
            }

            // solve with [-q;0] as a RHS to get z initializer
            // zero out any sparse cone variables at end
            self.workx.axpby(-T::one(), &data.q, T::zero());
            self.workz.fill(T::zero());

            self.kktsolver.setrhs(&self.workx, &self.workz);
            is_success = self
                .kktsolver
                .solve(None, Some(&mut variables.z), settings.core());
        } else {
            //QP initialization
            self.workx.scalarop_from(|q| -q, &data.q);
            self.workz.copy_from(&data.b);
            self.kktsolver.setrhs(&self.workx, &self.workz);
            is_success = self.kktsolver.solve(
                Some(&mut variables.x),
                Some(&mut variables.z),
                settings.core(),
            );
            variables.s.scalarop_from(|z| -z, &variables.z);
        }
        is_success && !initial_point_degenerate(variables, data)
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
        .max(data.A.nzval.norm_inf())
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
        self.workx.axpby(-T::one(), &data.q, T::zero()); //workx .= -q
        self.kktsolver.setrhs(&self.workx, &data.b);
        let is_success =
            self.kktsolver
                .solve(Some(&mut self.x2), Some(&mut self.z2), settings.core());

        is_success
    }

    pub(crate) fn update_P(&mut self, P: &CscMatrix<T>) {
        self.kktsolver.update_P(P);
    }

    pub(crate) fn update_A(&mut self, A: &CscMatrix<T>) {
        self.kktsolver.update_A(A);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::SupportedConeT::NonnegativeConeT;

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
