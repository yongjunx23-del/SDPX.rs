//! Variable stages for the existing generic HSD loop, on persistent owned slices.
use super::*;
use crate::solver::{
    cones::{Cone, PrimalOrDualCone, SupportedCone},
    core::{
        traits::{ConeCollection, Settings, Variables},
        ScalingStrategy, StepDirection,
    },
};
use rayon::prelude::*;
use std::{ops::Range, sync::Arc};

impl<T: FloatT> OwnedVariables<T> {
    pub fn new(data: &OwnedData<T>) -> Self {
        Self {
            blocks: data
                .blocks
                .iter()
                .map(|d| DefaultVariables::new(d.n, d.m))
                .collect(),
            layout: Arc::clone(&data.layout),
            owner_ids: data.owner_ids.clone(),
            all_owner_ids: data.all_owner_ids.clone(),
            collective: Arc::clone(&data.collective),
            tau: T::one(),
            kappa: T::one(),
            border_z: vec![T::zero(); data.layout.border_rows.len()],
            border_s: vec![T::zero(); data.layout.border_rows.len()],
        }
    }
    pub fn tau(&self) -> T {
        self.tau
    }
    pub fn kappa(&self) -> T {
        self.kappa
    }
    /// Equality duals and homogenization scalars have one canonical source;
    /// repeated equality slacks contribute only on owner zero.
    pub fn sync_border(&mut self) {
        for (local_owner, block) in self.blocks.iter_mut().enumerate() {
            block.τ = self.tau;
            block.κ = self.kappa;
            let ids = &self.layout.owners[local_owner];
            for (j, &row) in self.layout.border_rows.iter().enumerate() {
                if let Ok(target) = ids.rows.binary_search(&row) {
                    block.z[target] = self.border_z[j];
                    block.s[target] = self.border_s[j];
                }
            }
        }
    }
    fn scalar_bound(&self, step: &Self) -> T {
        let dt = step.tau();
        let dk = step.kappa();
        let at = if dt < T::zero() {
            -self.tau() / dt
        } else {
            T::max_value()
        };
        let ak = if dk < T::zero() {
            -self.kappa() / dk
        } else {
            T::max_value()
        };
        [at, ak, T::one()].minimum()
    }
    fn global_min_step(&self, alpha: T, site: usize) -> T {
        let finite = self
            .collective
            .all_true(site.saturating_sub(10), alpha.is_finite())
            .unwrap_or(false);
        let payload = if finite { -alpha } else { T::zero() };
        self.collective
            .reduce_max(site, &[payload])
            .ok()
            .and_then(|values| values.into_iter().next())
            .map_or(T::zero(), |value| if finite { -value } else { T::zero() })
    }

    fn refresh_border_from_owner_zero(&mut self) {
        let border = self.layout.border_rows.len();
        let mut local_s = vec![T::zero(); border];
        let mut local_z = vec![T::zero(); border];
        if let Some(local_owner) = self.owner_ids.iter().position(|&owner| owner == 0) {
            if let Some(block) = self.blocks.get(local_owner) {
                for (j, &row) in self.layout.border_rows.iter().enumerate() {
                    if let Ok(local) = self.layout.owners[local_owner].rows.binary_search(&row) {
                        local_s[j] = block.s[local];
                        local_z[j] = block.z[local];
                    }
                }
            }
        } else if self.collective.rank() == 0 {
            local_s.copy_from_slice(&self.border_s);
            local_z.copy_from_slice(&self.border_z);
        }
        let reduced_s = self.collective.reduce_sum(524, &local_s).ok();
        let reduced_z = self.collective.reduce_sum(525, &local_z).ok();
        if let Some(values) = reduced_s.filter(|values| values.len() == border) {
            self.border_s.copy_from_slice(&values);
        } else {
            self.border_s.fill(T::infinity());
        }
        if let Some(values) = reduced_z.filter(|values| values.len() == border) {
            self.border_z.copy_from_slice(&values);
        } else {
            self.border_z.fill(T::infinity());
        }
        self.sync_border();
    }
}
impl<T: FloatT> ConeCollection<T> for OwnedCones<T> {
    fn all_symmetric(&self) -> bool {
        let local = self.blocks.iter().all(|c| c.is_symmetric());
        self.collective.all_true(470, local).unwrap_or(false)
    }
    fn supports_primal_dual(&self) -> bool {
        let local = self.blocks.iter().all(|c| c.allows_primal_dual_scaling());
        self.collective.all_true(471, local).unwrap_or(false)
    }
    fn reset_scaling(&mut self) {
        if let Some(pool) = &self.pool {
            debug_assert!(owner_cone_pools_ok(&self.blocks));
            pool.install(|| {
                self.blocks
                    .par_iter_mut()
                    .for_each(|c| c.set_identity_scaling())
            });
        } else {
            for c in &mut self.blocks {
                c.set_identity_scaling();
            }
        }
    }
    fn worker_pool(&self) -> Option<Arc<rayon::ThreadPool>> {
        self.pool.clone()
    }
}
impl<T: FloatT> OwnedCones<T> {
    // Symmetric bounds only clip a cone-local bound by the current cap. As
    // in CompositeCone's cached symmetric path, evaluation at one common
    // finite positive cap preserves the final minimum. Mixed cones must keep
    // their original sequential cap-dependent evaluations.
    fn parallel_symmetric_bounds(
        &mut self,
        cap: T,
        evaluate: impl Fn(usize, &mut CompositeCone<T>, T) -> (T, T) + Sync + Send,
    ) -> Option<T> {
        if !cap.is_finite() || cap <= T::zero() || !self.all_symmetric() {
            return None;
        }
        let pool = self.pool.as_ref()?;
        let inner = pool.current_num_threads() > self.blocks.len();
        debug_assert!(owner_cone_pools_ok(&self.blocks));
        pool.install(|| {
            self.blocks
                .par_iter_mut()
                .zip(&mut self.bounds)
                .enumerate()
                .for_each(|(owner, (cone, bound))| {
                    let _inner = inner.then(sdpx_arithmetic::inner_parallel::Guard::enter);
                    *bound = evaluate(owner, cone, cap);
                })
        });
        Some(
            self.bounds
                .iter()
                .fold(cap, |cap, &(z, s)| T::min(cap, T::min(z, s))),
        )
    }
    fn fold_bounds(
        &mut self,
        mut alpha: T,
        mut evaluate: impl FnMut(usize, &mut SupportedCone<T>, Range<usize>, T) -> (T, T),
    ) -> T {
        let symmetric = self.all_symmetric();
        for sym in [true, false] {
            if !sym && !symmetric {
                alpha = T::min(alpha, T::one() - T::epsilon().sqrt());
            }
            for &(o, c) in &self.step_order {
                let rows = self.blocks[o].rng_cones[c].clone();
                let cone = self.blocks[o].iter_mut().nth(c).unwrap();
                if cone.is_symmetric() != sym {
                    continue;
                }
                let (z, s) = evaluate(o, cone, rows, alpha);
                alpha = T::min(alpha, T::min(z, s));
            }
        }
        alpha
    }
}
impl<T: FloatT> Variables<T> for OwnedVariables<T> {
    type D = OwnedData<T>;
    type R = OwnedResiduals<T>;
    type C = OwnedCones<T>;
    type SE = DefaultSettings<T>;
    fn calc_mu(&mut self, r: &Self::R, c: &Self::C) -> T {
        let sz = r
            .summary
            .as_ref()
            .expect("residuals must be updated before mu")
            .products
            .sz;
        (sz + self.tau() * self.kappa()) / T::from_usize(c.degree + 1).unwrap()
    }
    fn affine_step_rhs(&mut self, r: &Self::R, v: &Self, c: &Self::C) {
        if let Some(pool) = &c.pool {
            debug_assert!(owner_cone_pools_ok(&c.blocks));
            pool.install(|| {
                self.blocks
                    .par_iter_mut()
                    .zip(&r.blocks)
                    .zip(&v.blocks)
                    .zip(&c.blocks)
                    .for_each(|(((out, r), v), c)| out.affine_step_rhs(r, v, c));
            });
        } else {
            for (((out, r), v), c) in self
                .blocks
                .iter_mut()
                .zip(&r.blocks)
                .zip(&v.blocks)
                .zip(&c.blocks)
            {
                out.affine_step_rhs(r, v, c);
            }
        }
        self.tau = r.scalar.rτ;
        self.kappa = v.tau() * v.kappa();
        self.border_z.copy_from_slice(&r.border_residual);
        self.border_s.fill(T::zero());
        self.refresh_border_from_owner_zero();
    }
    fn combined_step_rhs(
        &mut self,
        r: &Self::R,
        v: &Self,
        c: &mut Self::C,
        step: &mut Self,
        sigma: T,
        mu: T,
        m: T,
    ) {
        if let Some(pool) = &c.pool {
            debug_assert!(owner_cone_pools_ok(&c.blocks));
            pool.install(|| {
                self.blocks
                    .par_iter_mut()
                    .zip(&r.blocks)
                    .zip(&v.blocks)
                    .zip(&mut c.blocks)
                    .zip(&mut step.blocks)
                    .for_each(|((((out, r), v), c), step)| {
                        out.combined_step_rhs(r, v, c, step, sigma, mu, m)
                    });
            });
        } else {
            for ((((out, r), v), c), step) in self
                .blocks
                .iter_mut()
                .zip(&r.blocks)
                .zip(&v.blocks)
                .zip(&mut c.blocks)
                .zip(&mut step.blocks)
            {
                out.combined_step_rhs(r, v, c, step, sigma, mu, m);
            }
        }
        self.tau = (T::one() - sigma) * r.scalar.rτ;
        self.kappa = -sigma * mu + m * step.tau() * step.kappa() + v.tau() * v.kappa();
        self.border_z
            .axpby(T::one() - sigma, &r.border_residual, T::zero());
        self.border_s.fill(T::zero());
        step.border_z.scale(m);
        self.refresh_border_from_owner_zero();
        step.refresh_border_from_owner_zero();
    }
    fn combined_step_rhs_prepared(
        &mut self,
        r: &Self::R,
        v: &Self,
        c: &mut Self::C,
        step: &mut Self,
        sigma: T,
        mu: T,
    ) {
        if let Some(pool) = &c.pool {
            debug_assert!(owner_cone_pools_ok(&c.blocks));
            pool.install(|| {
                self.blocks
                    .par_iter_mut()
                    .zip(&r.blocks)
                    .zip(&v.blocks)
                    .zip(&mut c.blocks)
                    .zip(&mut step.blocks)
                    .for_each(|((((out, r), v), c), step)| {
                        out.combined_step_rhs_prepared(r, v, c, step, sigma, mu)
                    });
            });
        } else {
            for ((((out, r), v), c), step) in self
                .blocks
                .iter_mut()
                .zip(&r.blocks)
                .zip(&v.blocks)
                .zip(&mut c.blocks)
                .zip(&mut step.blocks)
            {
                out.combined_step_rhs_prepared(r, v, c, step, sigma, mu);
            }
        }
        self.tau = (T::one() - sigma) * r.scalar.rτ;
        self.kappa = -sigma * mu + step.tau() * step.kappa() + v.tau() * v.kappa();
        self.border_z
            .axpby(T::one() - sigma, &r.border_residual, T::zero());
        self.border_s.fill(T::zero());
        self.refresh_border_from_owner_zero();
        step.refresh_border_from_owner_zero();
    }
    fn calc_step_length(
        &self,
        step: &Self,
        cones: &mut Self::C,
        settings: &Self::SE,
        direction: StepDirection,
    ) -> T {
        let cap = self.scalar_bound(step);
        if let Some(mut alpha) = cones.parallel_symmetric_bounds(cap, |o, cone, cap| {
            let v = &self.blocks[o];
            let d = &step.blocks[o];
            cone.step_length(&d.z, &d.s, &v.z, &v.s, settings.core(), cap)
        }) {
            if direction == StepDirection::Combined {
                alpha *= settings.core().max_step_fraction;
            }
            return self.global_min_step(alpha, 500);
        }
        let mut alpha = cones.fold_bounds(self.scalar_bound(step), |o, cone, r, cap| {
            let v = &self.blocks[o];
            let d = &step.blocks[o];
            cone.step_length(
                &d.z[r.clone()],
                &d.s[r.clone()],
                &v.z[r.clone()],
                &v.s[r],
                settings.core(),
                cap,
            )
        });
        if direction == StepDirection::Combined {
            alpha *= settings.core().max_step_fraction;
        }
        self.global_min_step(alpha, 501)
    }
    fn prepare_affine_step_length(
        &self,
        step: &mut Self,
        cones: &mut Self::C,
        settings: &Self::SE,
    ) -> T {
        let cap = self.scalar_bound(step);
        if cap.is_finite() && cap > T::zero() && cones.all_symmetric() {
            if let Some(pool) = &cones.pool {
                let inner = pool.current_num_threads() > cones.blocks.len();
                debug_assert!(owner_cone_pools_ok(&cones.blocks));
                pool.install(|| {
                    cones
                        .blocks
                        .par_iter_mut()
                        .zip(&mut step.blocks)
                        .zip(&self.blocks)
                        .zip(&mut cones.bounds)
                        .for_each(|(((cone, d), v), bound)| {
                            let _inner = inner.then(sdpx_arithmetic::inner_parallel::Guard::enter);
                            *bound = cone.prepare_affine_bounds(
                                &mut d.z,
                                &mut d.s,
                                &v.z,
                                &v.s,
                                settings.core(),
                                cap,
                            )
                        })
                });
                let alpha = cones
                    .bounds
                    .iter()
                    .fold(cap, |cap, &(z, s)| T::min(cap, T::min(z, s)));
                return self.global_min_step(alpha, 502);
            }
        }
        let alpha = cones.fold_bounds(self.scalar_bound(step), |o, cone, r, cap| {
            let v = &self.blocks[o];
            let d = &mut step.blocks[o];
            #[cfg(feature = "sdp")]
            if let SupportedCone::PSDTriangleCone(cone) = cone {
                return cone.prepare_affine_bounds(&mut d.z[r.clone()], &mut d.s[r], cap);
            }
            cone.step_length(
                &d.z[r.clone()],
                &d.s[r.clone()],
                &v.z[r.clone()],
                &v.s[r],
                settings.core(),
                cap,
            )
        });
        self.global_min_step(alpha, 503)
    }
    fn add_step(&mut self, step: &Self, alpha: T) {
        for (v, d) in self.blocks.iter_mut().zip(&step.blocks) {
            v.add_step(d, alpha);
        }
        self.tau += alpha * step.tau;
        self.kappa += alpha * step.kappa;
        for (value, &delta) in self.border_z.iter_mut().zip(&step.border_z) {
            *value += alpha * delta;
        }
        for (value, &delta) in self.border_s.iter_mut().zip(&step.border_s) {
            *value += alpha * delta;
        }
        self.sync_border();
    }
    fn symmetric_initialization(&mut self, cones: &mut Self::C) {
        for pd in [PrimalOrDualCone::PrimalCone, PrimalOrDualCone::DualCone] {
            let (mut minimum, mut positive) = (T::max_value(), T::zero());
            let mut original = None;
            let mut subtotal = T::zero();
            for &(o, c) in &cones.step_order {
                let rows = cones.blocks[o].rng_cones[c].clone();
                let values = match pd {
                    PrimalOrDualCone::PrimalCone => &mut self.blocks[o].s,
                    PrimalOrDualCone::DualCone => &mut self.blocks[o].z,
                };
                let (a, b) = cones.blocks[o]
                    .iter_mut()
                    .nth(c)
                    .unwrap()
                    .margins(&mut values[rows], pd);
                minimum = T::min(minimum, a);
                let id = self.layout.owners[o].cones[c].original;
                if original != Some(id) {
                    if original.is_some() {
                        positive += subtotal;
                    }
                    original = Some(id);
                    subtotal = b;
                } else {
                    subtotal += b;
                }
            }
            if original.is_some() {
                positive += subtotal;
            }
            let global_minimum = self
                .collective
                .reduce_max(520 + pd as usize, &[-minimum])
                .ok()
                .and_then(|values| values.into_iter().next())
                .map_or(T::zero(), |value| -value);
            let global_positive = self
                .collective
                .reduce_sum(522 + pd as usize, std::slice::from_ref(&positive))
                .ok()
                .and_then(|values| values.into_iter().next())
                .unwrap_or(T::infinity());
            let (first, second) = crate::solver::default::interior_shifts(
                global_minimum,
                global_positive,
                cones.degree,
            );
            for shift in [Some(first), second].into_iter().flatten() {
                for (v, c) in self.blocks.iter_mut().zip(&cones.blocks) {
                    let values = match pd {
                        PrimalOrDualCone::PrimalCone => &mut v.s,
                        PrimalOrDualCone::DualCone => &mut v.z,
                    };
                    c.scaled_unit_shift(values, shift, pd);
                }
            }
        }
        for v in &mut self.blocks {
            v.τ = T::one();
            v.κ = T::one();
        }
        self.tau = T::one();
        self.kappa = T::one();
        // Shared rows are ZeroCone rows, including a border-only owner.
        self.border_s.fill(T::zero());
        self.refresh_border_from_owner_zero();
    }
    fn unit_initialization(&mut self, cones: &Self::C) {
        for (v, c) in self.blocks.iter_mut().zip(&cones.blocks) {
            v.unit_initialization(c);
        }
        self.tau = T::one();
        self.kappa = T::one();
        self.border_s.fill(T::zero());
        self.border_z.fill(T::zero());
        self.refresh_border_from_owner_zero();
    }
    fn new_like(&self) -> Self {
        Self {
            blocks: self.blocks.iter().map(|v| v.new_like()).collect(),
            layout: Arc::clone(&self.layout),
            owner_ids: self.owner_ids.clone(),
            all_owner_ids: self.all_owner_ids.clone(),
            collective: Arc::clone(&self.collective),
            tau: self.tau,
            kappa: self.kappa,
            border_z: self.border_z.clone(),
            border_s: self.border_s.clone(),
        }
    }
    fn interpolate(&mut self, left: &Self, right: &Self, weight: T) {
        for ((v, a), b) in self.blocks.iter_mut().zip(&left.blocks).zip(&right.blocks) {
            v.interpolate(a, b, weight);
        }
        self.tau = left.tau + weight * (right.tau - left.tau);
        self.kappa = left.kappa + weight * (right.kappa - left.kappa);
        for ((out, &a), &b) in self
            .border_z
            .iter_mut()
            .zip(&left.border_z)
            .zip(&right.border_z)
        {
            *out = a + weight * (b - a);
        }
        for ((out, &a), &b) in self
            .border_s
            .iter_mut()
            .zip(&left.border_s)
            .zip(&right.border_s)
        {
            *out = a + weight * (b - a);
        }
        self.sync_border();
    }
    fn copy_from(&mut self, other: &Self) {
        assert_eq!(self.blocks.len(), other.blocks.len());
        for (v, s) in self.blocks.iter_mut().zip(&other.blocks) {
            v.copy_from(s);
        }
        self.tau = other.tau;
        self.kappa = other.kappa;
        self.border_z.copy_from_slice(&other.border_z);
        self.border_s.copy_from_slice(&other.border_s);
        self.sync_border();
    }
    fn scale_cones(&self, cones: &mut Self::C, mu: T, strategy: ScalingStrategy) -> bool {
        let local_ok = if let Some(pool) = &cones.pool {
            let inner = pool.current_num_threads() > cones.blocks.len();
            debug_assert!(owner_cone_pools_ok(&cones.blocks));
            // Complete every independent owner operation before reducing the
            // failure flag. No numerical sums change order, and failure still
            // returns false to the shared HSD strategy/retry path.
            pool.install(|| {
                cones
                    .blocks
                    .par_iter_mut()
                    .zip(&self.blocks)
                    .map(|(c, v)| {
                        let _inner = inner.then(sdpx_arithmetic::inner_parallel::Guard::enter);
                        v.scale_cones(c, mu, strategy)
                    })
                    .reduce(|| true, |a, b| a & b)
            })
        } else {
            let mut ok = true;
            for &(o, c) in &cones.step_order {
                let r = cones.blocks[o].rng_cones[c].clone();
                let v = &self.blocks[o];
                if !cones.blocks[o].iter_mut().nth(c).unwrap().update_scaling(
                    &v.s[r.clone()],
                    &v.z[r],
                    mu,
                    strategy,
                ) {
                    ok = false;
                }
            }
            ok
        };
        self.collective.all_true(480, local_ok).unwrap_or(false)
    }
    fn barrier(&self, step: &Self, alpha: T, cones: &mut Self::C) -> T {
        let central = T::from_usize(cones.degree + 1).unwrap();
        let tau = self.tau() + alpha * step.tau();
        let kappa = self.kappa() + alpha * step.kappa();
        let mut sz = T::zero();
        for &(o, c) in &cones.step_order {
            let v = &self.blocks[o];
            let d = &step.blocks[o];
            for i in cones.blocks[o].rng_cones[c].clone() {
                let s = d.s[i].mul_add(alpha, v.s[i]);
                let z = d.z[i].mul_add(alpha, v.z[i]);
                sz = s.mul_add(z, sz);
            }
        }
        let sz = self
            .collective
            .reduce_sum(510, std::slice::from_ref(&sz))
            .ok()
            .and_then(|values| values.into_iter().next())
            .unwrap_or(T::infinity());
        let mu = (sz + tau * kappa) / central;
        let mut barrier = central * mu.logsafe() - tau.logsafe() - kappa.logsafe();
        let mut total = T::zero();
        let mut original = None;
        let mut subtotal = T::zero();
        for &(o, c) in &cones.step_order {
            let r = cones.blocks[o].rng_cones[c].clone();
            let v = &self.blocks[o];
            let d = &step.blocks[o];
            let value = cones.blocks[o].iter_mut().nth(c).unwrap().compute_barrier(
                &v.z[r.clone()],
                &v.s[r.clone()],
                &d.z[r.clone()],
                &d.s[r],
                alpha,
            );
            let id = self.layout.owners[o].cones[c].original;
            if original != Some(id) {
                if original.is_some() {
                    total += subtotal;
                }
                original = Some(id);
                subtotal = value;
            } else {
                subtotal += value;
            }
        }
        if original.is_some() {
            total += subtotal;
        }
        let total = self
            .collective
            .reduce_sum(511, std::slice::from_ref(&total))
            .ok()
            .and_then(|values| values.into_iter().next())
            .unwrap_or(T::infinity());
        barrier += total;
        barrier
    }
    fn rescale(&mut self) {
        let scale = self
            .collective
            .reduce_max(512, &[T::max(self.tau, self.kappa)])
            .ok()
            .and_then(|values| values.into_iter().next())
            .unwrap_or(T::infinity());
        let invscale = scale.recip();
        for v in &mut self.blocks {
            v.x.scale(invscale);
            v.z.scale(invscale);
            v.s.scale(invscale);
        }
        self.tau *= invscale;
        self.kappa *= invscale;
        self.border_z.scale(invscale);
        self.border_s.scale(invscale);
        self.sync_border();
    }
}
#[cfg(test)]
#[path = "tests/hsd_steps.rs"]
mod tests;
