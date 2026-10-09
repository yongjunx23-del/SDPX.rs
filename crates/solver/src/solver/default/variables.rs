use super::*;
use crate::algebra::*;
use crate::solver::{
    cones::{CompositeCone, Cone, PrimalOrDualCone},
    core::{
        traits::{Settings, Variables},
        ScalingStrategy, StepDirection,
    },
};
use rayon::prelude::*;

/// Shorten `α` by `shrink` (at most 50 times) until the homogeneous pair
/// stays in the neighborhood `τκ ≥ β·μ` at the new point, where
/// `sᵀz(α) = c₀ + α·c₁ + α²·c₂` and `tk = [τ, κ, Δτ, Δκ]`.
pub(crate) fn taukappa_backtrack<T: FloatT>(
    c: [T; 3],
    tk: [T; 4],
    degree: usize,
    mut α: T,
    beta: T,
    shrink: T,
) -> T {
    let central = T::from_usize(degree + 1).unwrap();
    for _ in 0..50 {
        let pair = (tk[0] + α * tk[2]) * (tk[1] + α * tk[3]);
        let mu = (c[0] + α * (c[1] + α * c[2]) + pair) / central;
        if !(pair < beta * mu) {
            break;
        }
        α *= shrink;
    }
    α
}

// ---------------
// Variables type for default problem format
// ---------------

/// Standard-form solver type implementing the [`Variables`](crate::solver::core::traits::Variables) trait
pub struct DefaultVariables<T> {
    /// scaled primal variables
    pub x: Vec<T>,
    /// slack variables
    pub s: Vec<T>,
    /// scaled dual variables
    pub z: Vec<T>,
    /// homogenization scalar τ
    pub τ: T,
    /// homogenization scalar κ
    pub κ: T,
    /// Fixed-τ end phase: directions keep `Δτ = Δκ = 0` and μ counts the
    /// cones only, so the iterate follows an infeasible-start path of the
    /// original problem at the current scale. Not copied by `copy_from`.
    pub fixed_tau: bool,
}

impl<T: std::fmt::Display + std::fmt::Debug> std::fmt::Debug for DefaultVariables<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "x: {:?}\ns: {:?}\nz: {:?}\nτ: {:?}\nκ: {:?}\n",
            self.x, self.s, self.z, self.τ, self.κ
        )
    }
}

impl<T> DefaultVariables<T>
where
    T: FloatT,
{
    /// Create a new `DefaultVariables` object
    pub fn new(n: usize, m: usize) -> Self {
        let x = vec![T::zero(); n];
        let s = vec![T::zero(); m];
        let z = vec![T::zero(); m];
        let τ = T::one();
        let κ = T::one();

        Self {
            x,
            s,
            z,
            τ,
            κ,
            fixed_tau: false,
        }
    }
}

impl<T> Variables<T> for DefaultVariables<T>
where
    T: FloatT,
{
    fn write_checkpoint(
        &self,
        data: &DefaultProblemData<T>,
        path: &std::path::Path,
        iter: u32,
    ) -> std::io::Result<()> {
        let eq = &data.equilibration;
        crate::solver::core::checkpoint::write(
            path,
            data.checkpoint_identity(),
            iter,
            [self.τ, self.κ, eq.c],
            [&self.x, &self.s, &self.z, &eq.d, &eq.e],
        )
    }

    fn read_checkpoint(
        &mut self,
        data: &DefaultProblemData<T>,
        path: &std::path::Path,
    ) -> std::io::Result<bool> {
        let id = data.checkpoint_identity();
        let (n, m) = (self.x.len(), self.s.len());
        let ck = crate::solver::core::checkpoint::read::<T>(path, id.structure, n, m)?;
        let eq = &data.equilibration;
        let exact = ck.values == id.values && ck.c == eq.c && ck.d == eq.d && ck.e == eq.e;
        if exact {
            self.x.copy_from_slice(&ck.x);
            self.s.copy_from_slice(&ck.s);
            self.z.copy_from_slice(&ck.z);
        } else {
            // Map through original coordinates: x̂ = d∘x, ŝ = s/e, ẑ = e∘z/c, κ̂ = κ/c.
            for i in 0..n {
                self.x[i] = (ck.d[i] * ck.x[i]) * eq.dinv[i];
            }
            for i in 0..m {
                self.s[i] = (ck.s[i] / ck.e[i]) * eq.e[i];
                self.z[i] = ((ck.e[i] * ck.z[i]) / ck.c) * (eq.c * eq.einv[i]);
            }
        }
        self.τ = ck.tau;
        self.κ = if exact {
            ck.kappa
        } else {
            (ck.kappa / ck.c) * eq.c
        };
        Ok(exact)
    }

    type D = DefaultProblemData<T>;
    type R = DefaultResiduals<T>;
    type C = CompositeCone<T>;
    type SE = DefaultSettings<T>;

    fn calc_mu(&mut self, residuals: &DefaultResiduals<T>, cones: &CompositeCone<T>) -> T {
        if self.fixed_tau {
            return residuals.products.sz / T::from_usize(cones.degree().max(1)).unwrap();
        }
        let denom = T::from_usize(cones.degree() + 1).unwrap();
        (residuals.products.sz + self.τ * self.κ) / denom
    }

    fn affine_step_rhs(
        &mut self,
        residuals: &DefaultResiduals<T>,
        variables: &Self,
        cones: &CompositeCone<T>,
    ) {
        self.x.copy_from(&residuals.rx);
        self.z.copy_from(&residuals.rz);
        cones.affine_ds(&mut self.s, &variables.s);
        self.τ = residuals.rτ;
        self.κ = variables.τ * variables.κ;
    }

    fn combined_step_rhs(
        &mut self,
        residuals: &DefaultResiduals<T>,
        variables: &Self,
        cones: &mut CompositeCone<T>,
        step: &mut Self,
        σ: T,
        μ: T,
        m: T,
    ) {
        self.combined_rhs_impl(residuals, variables, cones, step, σ, μ, m, false);
    }

    fn combined_step_rhs_prepared(
        &mut self,
        residuals: &DefaultResiduals<T>,
        variables: &Self,
        cones: &mut CompositeCone<T>,
        step: &mut Self,
        σ: T,
        μ: T,
    ) {
        self.combined_rhs_impl(residuals, variables, cones, step, σ, μ, T::one(), true);
    }

    fn centrality_correction(
        &mut self,
        step: &Self,
        variables: &Self,
        cones: &mut CompositeCone<T>,
        α: T,
        σμ: T,
    ) -> bool {
        let (lo, hi) = (σμ * (0.1).as_T(), σμ * (10.0).as_T());
        let mut changed = cones.centrality_correction(
            &mut self.s,
            &variables.s,
            &variables.z,
            &step.s,
            &step.z,
            α,
            lo,
            hi,
        );
        // κΔτ + τΔκ = −rhs.κ, as the orthant rows' zΔs + sΔz = −ds.
        // The fixed-τ phase holds the pair, so it has nothing to correct.
        let v = (variables.τ + α * step.τ) * (variables.κ + α * step.κ);
        if let (false, Some(t)) = (variables.fixed_tau, band_correction(v, lo, hi)) {
            self.κ -= t;
            changed = true;
        }
        changed
    }

    fn prepare_affine_step_length(
        &self,
        step: &mut Self,
        cones: &mut CompositeCone<T>,
        settings: &DefaultSettings<T>,
    ) -> T {
        let ατ = if step.τ < T::zero() {
            -self.τ / step.τ
        } else {
            T::max_value()
        };
        let ακ = if step.κ < T::zero() {
            -self.κ / step.κ
        } else {
            T::max_value()
        };
        let α = [ατ, ακ, T::one()].minimum();
        let (αz, αs) = cones.prepare_affine_bounds(
            &mut step.z,
            &mut step.s,
            &self.z,
            &self.s,
            settings.core(),
            α,
        );
        T::min(αz, αs)
    }

    fn calc_step_length(
        &self,
        step: &Self,
        cones: &mut CompositeCone<T>,
        settings: &DefaultSettings<T>,
        step_direction: StepDirection,
    ) -> T {
        let ατ = {
            if step.τ < T::zero() {
                -self.τ / step.τ
            } else {
                T::max_value()
            }
        };

        let ακ = {
            if step.κ < T::zero() {
                -self.κ / step.κ
            } else {
                T::max_value()
            }
        };

        let α = [ατ, ακ, T::one()].minimum();
        let (αz, αs) = cones.step_length(&step.z, &step.s, &self.z, &self.s, settings.core(), α);

        // itself only allows for a single maximum value.
        // To enable split lengths, we need to also pass a
        // tuple of limits to the step_length function of
        // every cone
        let mut α = T::min(αz, αs);

        if step_direction == StepDirection::Combined {
            α *= settings.core().max_step_fraction;
        }

        α
    }

    fn add_step(&mut self, step: &Self, α: T) {
        self.x.axpby(α, &step.x, T::one());
        self.s.axpby(α, &step.s, T::one());
        self.z.axpby(α, &step.z, T::one());
        self.τ += α * step.τ;
        self.κ += α * step.κ;
    }

    fn add_step_with_pool(
        &mut self,
        step: &Self,
        α: T,
        pool: Option<std::sync::Arc<rayon::ThreadPool>>,
    ) {
        let Some(pool) = pool else {
            self.add_step(step, α);
            return;
        };
        // Elementwise axpby on disjoint chunks — every element uses the
        // identical expression as the serial path, so results are bitwise
        // identical.  The scalar τ/κ updates stay sequential.
        const CHUNK: usize = 2048;
        fn par_axpby<T: FloatT>(y: &mut [T], a: T, x: &[T], b: T) {
            y.par_chunks_mut(CHUNK)
                .zip(x.par_chunks(CHUNK))
                .for_each(|(y, x)| {
                    for (y, x) in y.iter_mut().zip(x) {
                        *y = a * (*x) + b * (*y);
                    }
                });
        }
        let (x, s, z) = (&mut self.x, &mut self.s, &mut self.z);
        pool.install(|| {
            rayon::join(
                || par_axpby(x, α, &step.x, T::one()),
                || {
                    rayon::join(
                        || par_axpby(s, α, &step.s, T::one()),
                        || par_axpby(z, α, &step.z, T::one()),
                    )
                },
            );
        });
        self.τ += α * step.τ;
        self.κ += α * step.κ;
    }

    fn symmetric_initialization(&mut self, cones: &mut CompositeCone<T>) {
        _shift_to_cone_interior(&mut self.s, cones, PrimalOrDualCone::PrimalCone);
        _shift_to_cone_interior(&mut self.z, cones, PrimalOrDualCone::DualCone);

        self.τ = T::one();
        self.κ = T::one();
    }

    fn unit_initialization(&mut self, cones: &CompositeCone<T>) {
        cones.unit_initialization(&mut self.z, &mut self.s);

        self.x.set(T::zero());
        self.τ = T::one();
        self.κ = T::one();
    }

    fn taukappa_backtrack(
        &self,
        step: &Self,
        α: T,
        beta: T,
        shrink: T,
        cones: &CompositeCone<T>,
    ) -> T {
        if self.fixed_tau {
            return α;
        }
        let c = [
            self.s.dot(&self.z),
            self.s.dot(&step.z) + self.z.dot(&step.s),
            step.s.dot(&step.z),
        ];
        taukappa_backtrack(
            c,
            [self.τ, self.κ, step.τ, step.κ],
            cones.degree(),
            α,
            beta,
            shrink,
        )
    }

    fn set_initial_tau(&mut self, tau: T) {
        self.τ = tau;
        self.κ = T::recip(tau);
    }

    fn tau(&self) -> Option<T> {
        Some(self.τ)
    }

    fn freeze_tau(&mut self) -> bool {
        self.fixed_tau = true;
        true
    }

    fn tau_frozen(&self) -> bool {
        self.fixed_tau
    }

    /// Only with τ fixed and P = 0: r_z then depends on (x, s) alone and r_x
    /// on z alone, so each residual falls by its own step (SDPB's α_P, α_D).
    fn split_step(
        &self,
        step: &mut Self,
        α: T,
        data: &DefaultProblemData<T>,
        cones: &mut CompositeCone<T>,
        settings: &DefaultSettings<T>,
    ) -> T {
        if !self.fixed_tau || data.P.nnz() != 0 || !crate::solver::core::test_split_step() {
            return α;
        }
        // The composite step length returns one common bound: get each part
        // with the other part's direction zeroed (test switch only).
        let zero = vec![T::zero(); step.s.len()];
        let αz = T::min(cones.step_length(&step.z, &zero, &self.z, &self.s, settings.core(), T::one()).0, T::one());
        let αs = T::min(cones.step_length(&zero, &step.s, &self.z, &self.s, settings.core(), T::one()).1, T::one());
        let f = settings.core().max_step_fraction;
        let (αp, αd) = (αs * f, αz * f);
        let a = T::min(αp, αd);
        if !(a > T::zero()) {
            return α;
        }
        step.x.scale(αp / a);
        step.s.scale(αp / a);
        step.z.scale(αd / a);
        a
    }

    fn scale_stats(&self) -> String {
        let f = |v: T| v.to_f64().unwrap_or(f64::NAN);
        format!(
            "x_inf={:.3e} x_2={:.3e} s_inf={:.3e} s_2={:.3e} z_inf={:.3e} z_2={:.3e} sz={:.3e} tau={:.3e} kappa={:.3e}",
            f(self.x.norm_inf()),
            f(self.x.norm()),
            f(self.s.norm_inf()),
            f(self.s.norm()),
            f(self.z.norm_inf()),
            f(self.z.norm()),
            f(self.s.dot(&self.z)),
            f(self.τ),
            f(self.κ)
        )
    }

    fn new_like(&self) -> Self {
        Self::new(self.x.len(), self.s.len())
    }
    fn interpolate(&mut self, left: &Self, right: &Self, weight: T) {
        self.copy_from(left);
        let a = T::one() - weight;
        self.x.axpby(weight, &right.x, a);
        self.s.axpby(weight, &right.s, a);
        self.z.axpby(weight, &right.z, a);
        self.τ = a * left.τ + weight * right.τ;
        self.κ = a * left.κ + weight * right.κ;
    }

    fn copy_from(&mut self, src: &Self) {
        self.x.copy_from(&src.x);
        self.s.copy_from(&src.s);
        self.z.copy_from(&src.z);
        self.τ = src.τ;
        self.κ = src.κ;
    }

    fn scale_cones(
        &self,
        cones: &mut CompositeCone<T>,
        μ: T,
        scaling_strategy: ScalingStrategy,
    ) -> bool {
        cones.update_scaling(&self.s, &self.z, μ, scaling_strategy)
    }

    fn barrier(&self, step: &Self, α: T, cones: &mut CompositeCone<T>) -> T {
        let central_coef = (cones.degree() + 1).as_T();

        let cur_τ = self.τ + α * step.τ;
        let cur_κ = self.κ + α * step.κ;

        // compute current μ
        let sz = <[T] as VectorMath<T>>::dot_shifted(&self.z, &self.s, &step.z, &step.s, α);
        let μ = (sz + cur_τ * cur_κ) / central_coef;

        // barrier terms from gap and scalars
        let mut barrier = central_coef * μ.logsafe() - cur_τ.logsafe() - cur_κ.logsafe();

        // barriers from the cones
        let (z, s) = (&self.z, &self.s);
        let (dz, ds) = (&step.z, &step.s);

        barrier += cones.compute_barrier(z, s, dz, ds, α);

        barrier
    }
}

fn _shift_to_cone_interior<T>(z: &mut [T], cones: &mut CompositeCone<T>, pd: PrimalOrDualCone)
where
    T: FloatT,
{
    let (min_margin, pos_margin) = cones.margins(z, pd);
    let (first, second) = interior_shifts(min_margin, pos_margin, cones.degree());
    if std::env::var_os("SDPX_START_STATS").is_some() {
        let side = match pd {
            PrimalOrDualCone::PrimalCone => "primal",
            PrimalOrDualCone::DualCone => "dual",
        };
        let f = |v: T| v.to_f64().unwrap_or(f64::NAN);
        eprintln!(
            "start-stats interior side={side} degree={} min_margin={:.3e} pos_margin={:.3e} first_shift={:.3e} second_shift={:.3e}",
            cones.degree(),
            f(min_margin),
            f(pos_margin),
            f(first),
            f(second.unwrap_or(T::zero()))
        );
    }
    cones.scaled_unit_shift(z, first, pd);
    if let Some(second) = second {
        cones.scaled_unit_shift(z, second, pd);
    }
}

/// Shared scalar decision for serial and owned symmetric initialization.
pub(crate) fn interior_shifts<T: FloatT>(
    min_margin: T,
    pos_margin: T,
    degree: usize,
) -> (T, Option<T>) {
    let target = T::max(T::one(), (pos_margin * (0.1).as_T()) / degree.as_T());
    if min_margin <= T::zero() {
        // Two shifts avoid losing the positive target when -min is large.
        (-min_margin, Some(target))
    } else if min_margin < target {
        (target - min_margin, None)
    } else {
        (T::zero(), None)
    }
}

impl<T> DefaultVariables<T>
where
    T: FloatT,
{
    pub(crate) fn unscale(&mut self, data: &DefaultProblemData<T>, is_infeasible: bool) {
        // if we have an infeasible problem, normalize
        // using κ to get an infeasibility certificate.
        // Otherwise use τ to get an unscaled solution.
        let scaleinv = {
            if is_infeasible {
                T::recip(self.κ)
            } else {
                T::recip(self.τ)
            }
        };

        // also undo the equilibration
        let d = &data.equilibration.d;
        let (e, einv) = (&data.equilibration.e, &data.equilibration.einv);
        let cinv = T::recip(data.equilibration.c);

        self.x.hadamard(d).scale(scaleinv);
        self.z.hadamard(e).scale(scaleinv * cinv);
        self.s.hadamard(einv).scale(scaleinv);

        self.τ *= scaleinv;
        self.κ *= scaleinv;
    }

    pub(crate) fn dims(&self) -> (usize, usize) {
        (self.x.len(), self.s.len())
    }

    /// An independent copy of the iterate.
    pub(crate) fn copy_of(&self) -> Self {
        Self {
            x: self.x.clone(),
            s: self.s.clone(),
            z: self.z.clone(),
            τ: self.τ,
            κ: self.κ,
            fixed_tau: self.fixed_tau,
        }
    }
}

impl<T: FloatT> DefaultVariables<T> {
    fn combined_rhs_impl(
        &mut self,
        residuals: &DefaultResiduals<T>,
        variables: &Self,
        cones: &mut CompositeCone<T>,
        step: &mut Self,
        σ: T,
        μ: T,
        m: T,
        prepared: bool,
    ) {
        let dotσμ = σ * μ;

        self.x.axpby(T::one() - σ, &residuals.rx, T::zero()); //self.x  = (1 - σ)*rx
        self.τ = (T::one() - σ) * residuals.rτ;
        self.κ = -dotσμ + m * step.τ * step.κ + variables.τ * variables.κ;

        // ds is different for symmetric and asymmetric cones:
        // Symmetric cones: d.s = λ ◦ λ + W⁻¹Δs ∘ WΔz − σμe
        // Asymmetric cones: d.s = s + σμ*g(z)

        // we want to scale the Mehotra correction in the symmetric
        // case by M, so just scale step_z by M.  This is an unnecessary
        // vector operation (since it amounts to M*z'*s), but it
        // doesn't happen very often
        if m != T::one() {
            step.z.scale(m);
        }

        cones.combined_shift_impl(&mut self.z, &mut step.z, &mut step.s, dotσμ, prepared);

        //We are relying on d.s = affine_ds already here
        self.s.axpby(T::one(), &self.z, T::one());

        // now we copy the scaled res for rz and d.z is no longer work
        self.z.axpby(T::one() - σ, &residuals.rz, T::zero());
    }
}

#[cfg(test)]
#[path = "tests/affine_prepared.rs"]
mod affine_prepared_tests;

/// Gondzio's target change for a trial complementarity product `v`: back
/// to the band `[lo, hi]`, large products capped at `−hi` (Colombo &
/// Gondzio, 2008). `None` inside the band.
pub(crate) fn band_correction<T: FloatT>(v: T, lo: T, hi: T) -> Option<T> {
    if v < lo {
        Some(lo - v)
    } else if v > hi {
        Some(T::max(hi - v, -hi))
    } else {
        None
    }
}
