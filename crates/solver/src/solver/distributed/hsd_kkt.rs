//! Embedding direction assembly for local storage; scalar formulas and matrix
//! operations are shared with the default KKT implementation.
use super::*;
use crate::solver::default::{hsd_tau, hsd_terms};
use crate::solver::distributed::OwnedKkt;
use crate::solver::{
    core::{traits::KKTSystem, StepDirection},
    kkt::{HasLinearSolverInfo, LinearSolverInfo, SolveCounters},
};

pub(crate) struct OwnedKktSystem<T: FloatT> {
    pub(crate) kernel: OwnedKkt<T>,
    constant: OwnedVariables<T>,
    constant_work: OwnedVariables<T>,
    varying: OwnedVariables<T>,
    work: OwnedVariables<T>,
    hs_constant: Vec<Vec<T>>,
    hs_varying: Vec<Vec<T>>,
    affine_ready: Option<bool>,
    pool: Option<Arc<rayon::ThreadPool>>,
    collective: Arc<dyn crate::solver::distributed::collective::Collective<T>>,
}
impl<T: FloatT> OwnedKktSystem<T> {
    pub fn new_with_collective(
        data: &OwnedData<T>,
        cones: &OwnedCones<T>,
        settings: &DefaultSettings<T>,
        collective: Arc<dyn crate::solver::distributed::collective::Collective<T>>,
    ) -> Self {
        Self {
            kernel: OwnedKkt::new_with_pool_collective(
                &data.layout,
                &data.owner_ids,
                data.blocks.iter().zip(&cones.blocks),
                settings.core(),
                cones.pool.clone(),
                data.cost.record,
                Arc::clone(&collective),
            ),
            constant: OwnedVariables::new(data),
            constant_work: OwnedVariables::new(data),
            varying: OwnedVariables::new(data),
            work: OwnedVariables::new(data),
            hs_constant: data.blocks.iter().map(|_| Vec::new()).collect(),
            hs_varying: data.blocks.iter().map(|_| Vec::new()).collect(),
            affine_ready: None,
            pool: cones.pool.clone(),
            collective,
        }
    }
    pub fn counters(&self) -> SolveCounters {
        self.kernel.counters()
    }
    pub(crate) fn cost_samples(&self) -> Option<Vec<(f64, f64)>> {
        self.kernel.cost_samples()
    }
    fn copy_products(kernel: &OwnedKkt<T>, out: &mut [Vec<T>]) {
        for (i, out) in out.iter_mut().enumerate() {
            if let Some(h) = kernel.scaled_product(i) {
                out.resize(h.len(), T::zero());
                out.copy_from_slice(h);
            } else {
                out.clear();
            }
        }
    }
    fn constant_rhs(&mut self, data: &OwnedData<T>, settings: &DefaultSettings<T>) -> bool {
        let fill = |(out, d): (&mut DefaultVariables<T>, &DefaultProblemData<T>)| {
            out.x.scalarop_from(|q| -q, &d.q);
            out.z.copy_from_slice(&d.b);
        };
        for_blocks!(
            self.pool.as_ref(),
            (&mut self.work.blocks, &data.blocks),
            fill
        );
        self.work.border_z.copy_from_slice(&data.border_b);
        self.work.sync_border();
        let ok = self.kernel.solve_blocks_with_border(
            &self.work.blocks,
            &mut self.constant.blocks,
            Some(&self.work.border_z),
            settings.core(),
        );
        if ok {
            self.constant
                .border_z
                .copy_from_slice(self.kernel.last_border());
            self.constant.sync_border();
        }
        Self::copy_products(&self.kernel, &mut self.hs_constant);
        ok
    }
}
impl<T: FloatT> HasLinearSolverInfo for OwnedKktSystem<T> {
    fn linear_solver_info(&self) -> LinearSolverInfo {
        self.kernel.linear_info()
    }
}
impl<T: FloatT> KKTSystem<T> for OwnedKktSystem<T> {
    type D = OwnedData<T>;
    type V = OwnedVariables<T>;
    type C = OwnedCones<T>;
    type SE = DefaultSettings<T>;
    fn reset_solve(&mut self) {
        self.kernel.reset_solve();
        self.affine_ready = None;
    }
    fn update(
        &mut self,
        data: &OwnedData<T>,
        cones: &OwnedCones<T>,
        settings: &DefaultSettings<T>,
    ) -> bool {
        self.affine_ready = None;
        loop {
            let local_ok = self.kernel.update_local(&cones.blocks, settings.core())
                && self.constant_rhs(data, settings);
            // Keep retries collective: a local factor failure must not let a
            // successful rank skip the boost stream while peers enter it.
            let ok = self.collective.all_true(304, local_ok).unwrap_or(false);
            if ok {
                return true;
            }
            if !settings.static_regularization_enable
                || !self
                    .collective
                    .all_true(300, self.kernel.escalate_regularization())
                    .unwrap_or(false)
            {
                return false;
            }
        }
    }
    fn update_affine(
        &mut self,
        data: &OwnedData<T>,
        cones: &OwnedCones<T>,
        rhs: &OwnedVariables<T>,
        variables: &OwnedVariables<T>,
        settings: &DefaultSettings<T>,
    ) -> bool {
        let fill_constant = |(w, d): (&mut DefaultVariables<T>, &DefaultProblemData<T>)| {
            w.x.scalarop_from(|q| -q, &d.q);
            w.z.copy_from_slice(&d.b);
        };
        let fill_affine = |(w, r, v): (
            &mut DefaultVariables<T>,
            &DefaultVariables<T>,
            &DefaultVariables<T>,
        )| {
            w.x.copy_from_slice(&r.x);
            w.z.waxpby(T::one(), &v.s, -T::one(), &r.z);
        };
        loop {
            self.affine_ready = None;
            let updated_local = self.kernel.update_local(&cones.blocks, settings.core());
            let updated = self
                .collective
                .all_true(299, updated_local)
                .unwrap_or(false);
            let constant_ok;
            if updated {
                for_blocks!(
                    cones.pool.as_ref(),
                    (&mut self.constant_work.blocks, &data.blocks),
                    fill_constant
                );
                for_blocks!(
                    cones.pool.as_ref(),
                    (&mut self.work.blocks, &rhs.blocks, &variables.blocks),
                    fill_affine
                );
                self.constant_work.border_z.copy_from_slice(&data.border_b);
                for ((out, &s), &z) in self
                    .work
                    .border_z
                    .iter_mut()
                    .zip(&variables.border_s)
                    .zip(&rhs.border_z)
                {
                    *out = s - z;
                }
                self.constant_work.sync_border();
                self.work.sync_border();
                let ok = self.kernel.solve_blocks_pair_with_border(
                    [&self.constant_work.blocks, &self.work.blocks],
                    [&mut self.constant.blocks, &mut self.varying.blocks],
                    [&mut self.hs_constant, &mut self.hs_varying],
                    [
                        Some(&self.constant_work.border_z),
                        Some(&self.work.border_z),
                    ],
                    settings.core(),
                );
                if ok[0] {
                    self.constant
                        .border_z
                        .copy_from_slice(self.kernel.pair_border(0));
                    self.constant.sync_border();
                }
                if ok[1] {
                    self.varying
                        .border_z
                        .copy_from_slice(self.kernel.pair_border(1));
                    self.varying.sync_border();
                }
                constant_ok = self.collective.all_true(301, ok[0]).unwrap_or(false);
                self.affine_ready = Some(self.collective.all_true(302, ok[1]).unwrap_or(false));
            } else {
                // Preserve the same collective sequence after a rank-local
                // update failure; all ranks then take the common retry path.
                constant_ok = self.collective.all_true(301, false).unwrap_or(false);
                self.affine_ready = Some(self.collective.all_true(302, false).unwrap_or(false));
            }
            // Preserve production retry semantics: only a constant/factor
            // failure retries with a stronger shift; an affine failure remains
            // isolated and is consumed by solve() as a failed predictor.
            if constant_ok {
                return true;
            }
            if !settings.core().static_regularization_enable
                || !self
                    .collective
                    .all_true(303, self.kernel.escalate_regularization())
                    .unwrap_or(false)
            {
                return false;
            }
        }
    }
    fn solve(
        &mut self,
        lhs: &mut OwnedVariables<T>,
        rhs: &OwnedVariables<T>,
        data: &OwnedData<T>,
        variables: &OwnedVariables<T>,
        cones: &mut OwnedCones<T>,
        direction: StepDirection,
        settings: &DefaultSettings<T>,
    ) -> bool {
        let cached = self.affine_ready.take();
        // update_affine already prepared and solved this RHS. Only its offset
        // is needed for direction recovery; do not repack it into work again.
        let prepare_rhs = direction != StepDirection::Affine || cached.is_none();
        let prepare = |i: usize,
                       w: &mut DefaultVariables<T>,
                       constant_work: &mut DefaultVariables<T>,
                       cone: &mut CompositeCone<T>,
                       lhs: &mut DefaultVariables<T>| {
            let offset = &mut constant_work.s;
            let r = &rhs.blocks[i];
            let v = &variables.blocks[i];
            if prepare_rhs {
                w.x.copy_from_slice(&r.x);
            }
            match direction {
                StepDirection::Affine => {
                    offset.copy_from_slice(&v.s);
                    if prepare_rhs {
                        w.z.waxpby(T::one(), offset, -T::one(), &r.z);
                    }
                }
                StepDirection::Combined => {
                    cone.Δs_from_Δz_offset(offset, &r.s, &mut lhs.z, &v.z);
                    w.z.copy_from_slice(&r.z);
                }
            }
        };
        debug_assert!(cones.pool.is_none() || owner_cone_pools_ok(&cones.blocks));
        let owners = self.work.blocks.len();
        for_blocks!(
            cones.pool.as_ref(),
            (
                0..owners,
                &mut self.work.blocks,
                &mut self.constant_work.blocks,
                &mut cones.blocks,
                &mut lhs.blocks
            ),
            |(i, w, offset, cone, lhs)| prepare(i, w, offset, cone, lhs)
        );
        if prepare_rhs {
            if direction == StepDirection::Affine {
                for ((out, &s), &z) in self
                    .work
                    .border_z
                    .iter_mut()
                    .zip(&variables.border_s)
                    .zip(&rhs.border_z)
                {
                    *out = s - z;
                }
            } else {
                for (out, &value) in self.work.border_z.iter_mut().zip(&rhs.border_z) {
                    *out = -value;
                }
            }
            self.work.sync_border();
        }
        if direction == StepDirection::Combined {
            let subtract = |(w, c): (&mut DefaultVariables<T>, &DefaultVariables<T>)| {
                for (v, &c) in w.z.iter_mut().zip(&c.s) {
                    *v = c - *v;
                }
            };
            for_blocks!(
                cones.pool.as_ref(),
                (&mut self.work.blocks, &self.constant_work.blocks),
                subtract
            );
        }
        let prepared = if direction == StepDirection::Affine {
            cached
        } else {
            None
        };
        let used_kernel = prepared.is_none();
        let ok = prepared.unwrap_or_else(|| {
            let ok = self.kernel.solve_blocks_with_border(
                &self.work.blocks,
                &mut self.varying.blocks,
                Some(&self.work.border_z),
                settings.core(),
            );
            Self::copy_products(&self.kernel, &mut self.hs_varying);
            ok
        });
        if !ok {
            return self.collective.all_true(305, false).unwrap_or(false);
        }
        if used_kernel {
            self.varying
                .border_z
                .copy_from_slice(self.kernel.last_border());
            self.varying.sync_border();
        }
        let mut terms = None;
        for i in 0..data.blocks.len() {
            let first = &self.varying.blocks[i];
            let second = &self.constant.blocks[i];
            let local = hsd_terms(
                &mut self.work.blocks[i].x,
                &variables.blocks[i].x,
                variables.tau(),
                &data.blocks[i],
                &first.x,
                &first.z,
                &second.x,
                &second.z,
            );
            if let Some(ref mut total) = terms {
                crate::solver::default::HsdTerms::add(total, local);
            } else {
                terms = Some(local);
            }
        }
        let mut terms = terms.unwrap_or(crate::solver::default::HsdTerms {
            q1: T::zero(),
            b1: T::zero(),
            quad1: T::zero(),
            q2: T::zero(),
            b2: T::zero(),
            delta: T::zero(),
            quad2: T::zero(),
        });
        if self.collective.rank() == 0 && !data.owner_ids.contains(&0) {
            terms.b1 += data.border_b.dot(&self.varying.border_z);
            terms.b2 += data.border_b.dot(&self.constant.border_z);
        }
        let terms = self
            .collective
            .reduce_sum(
                350,
                &[
                    terms.q1,
                    terms.b1,
                    terms.quad1,
                    terms.q2,
                    terms.b2,
                    terms.delta,
                    terms.quad2,
                ],
            )
            .ok()
            .filter(|values| values.len() == 7)
            .map(|values| crate::solver::default::HsdTerms {
                q1: values[0],
                b1: values[1],
                quad1: values[2],
                q2: values[3],
                b2: values[4],
                delta: values[5],
                quad2: values[6],
            })
            .unwrap_or(crate::solver::default::HsdTerms {
                q1: T::nan(),
                b1: T::nan(),
                quad1: T::nan(),
                q2: T::nan(),
                b2: T::nan(),
                delta: T::nan(),
                quad2: T::nan(),
            });
        let dtau = hsd_tau(
            terms,
            rhs.tau,
            rhs.kappa,
            variables.tau(),
            variables.kappa(),
        );
        let dkappa = -(rhs.kappa + variables.kappa() * dtau) / variables.tau();
        lhs.tau = dtau;
        lhs.kappa = dkappa;
        let (varying, constant, hs_constant, hs_varying, offset) = (
            &self.varying,
            &self.constant,
            &self.hs_constant,
            &self.hs_varying,
            &self.constant_work.blocks,
        );
        let recover = |i: usize,
                       l: &mut DefaultVariables<T>,
                       work: &mut DefaultVariables<T>,
                       cone: &mut CompositeCone<T>| {
            let d = &data.blocks[i];
            let first = &varying.blocks[i];
            let second = &constant.blocks[i];
            l.τ = dtau;
            l.κ = dkappa;
            l.x.waxpby(T::one(), &first.x, dtau, &second.x);
            l.z.waxpby(T::one(), &first.z, dtau, &second.z);
            if hs_constant[i].len() == d.m && hs_varying[i].len() == d.m {
                l.s.waxpby(T::one(), &hs_varying[i], dtau, &hs_constant[i]);
            } else {
                cone.mul_Hs(&mut l.s, &l.z, &mut work.z);
            }
            l.s.axpby(-T::one(), &offset[i].s, -T::one());
        };
        let owners = lhs.blocks.len();
        for_blocks!(
            cones.pool.as_ref(),
            (
                0..owners,
                &mut lhs.blocks,
                &mut self.work.blocks,
                &mut cones.blocks
            ),
            |(i, l, work, cone)| recover(i, l, work, cone)
        );
        for ((out, &varying), &constant) in lhs
            .border_z
            .iter_mut()
            .zip(&self.varying.border_z)
            .zip(&self.constant.border_z)
        {
            *out = varying + dtau * constant;
        }
        lhs.sync_border();
        let local_finite = lhs.tau.is_finite()
            && lhs.kappa.is_finite()
            && lhs.border_z.is_finite()
            && lhs.border_s.is_finite()
            && lhs.blocks.iter().all(|block| {
                block.x.is_finite()
                    && block.s.is_finite()
                    && block.z.is_finite()
                    && block.τ.is_finite()
                    && block.κ.is_finite()
            });
        self.collective.all_true(305, local_finite).unwrap_or(false)
    }
    fn solve_initial_point(
        &mut self,
        v: &mut OwnedVariables<T>,
        data: &OwnedData<T>,
        settings: &DefaultSettings<T>,
    ) -> bool {
        let linear_local = data.blocks.iter().all(|d| d.P.nnz() == 0);
        let linear = self.collective.all_true(307, linear_local).unwrap_or(false);
        for (w, d) in self.work.blocks.iter_mut().zip(&data.blocks) {
            if linear {
                w.x.fill(T::zero());
            } else {
                w.x.scalarop_from(|q| -q, &d.q);
            }
            w.z.copy_from_slice(&d.b);
        }
        self.work.border_z.copy_from_slice(&data.border_b);
        self.work.sync_border();
        let initial_ok = self.kernel.solve_blocks_with_border(
            &self.work.blocks,
            &mut self.varying.blocks,
            Some(&self.work.border_z),
            settings.core(),
        );
        if !self.collective.all_true(306, initial_ok).unwrap_or(false) {
            return false;
        }
        v.border_z.copy_from_slice(self.kernel.last_border());
        v.sync_border();
        for (v, x) in v.blocks.iter_mut().zip(&self.varying.blocks) {
            v.x.copy_from_slice(&x.x);
            v.s.scalarop_from(|z| -z, &x.z);
            v.z.copy_from_slice(&x.z);
        }
        for (j, &row) in data.layout.border_rows.iter().enumerate() {
            let mut local_sum = T::zero();
            for (local_owner, block) in v.blocks.iter_mut().enumerate() {
                if let Ok(local) = data.layout.owners[local_owner].rows.binary_search(&row) {
                    local_sum += block.s[local];
                    block.s[local] = T::zero();
                }
            }
            v.border_s[j] = local_sum;
        }
        if !v.border_s.is_empty() {
            if self
                .collective
                .reduce_sum_in_place(370, &mut v.border_s)
                .is_err()
            {
                v.border_s.fill(T::infinity());
            }
        }
        if linear {
            for (w, d) in self.work.blocks.iter_mut().zip(&data.blocks) {
                w.x.scalarop_from(|q| -q, &d.q);
                w.z.fill(T::zero());
            }
            self.work.border_z.fill(T::zero());
            let second_ok = self.kernel.solve_blocks_with_border(
                &self.work.blocks,
                &mut self.varying.blocks,
                Some(&self.work.border_z),
                settings.core(),
            );
            if !self.collective.all_true(309, second_ok).unwrap_or(false) {
                return false;
            }
            v.border_z.copy_from_slice(self.kernel.last_border());
            v.sync_border();
            for (v, x) in v.blocks.iter_mut().zip(&self.varying.blocks) {
                v.z.copy_from_slice(&x.z);
            }
        }
        v.sync_border();
        let mut scale = T::one();
        let (mut nx, mut nz) = (T::zero(), T::zero());
        for (v, d) in v.blocks.iter().zip(&data.blocks) {
            scale = scale
                .max(d.b.norm_inf())
                .max(d.q.norm_inf())
                .max(d.constraint_norm_inf());
            nx = nx.max(v.x.norm_inf());
            nz = nz.max(v.z.norm_inf());
        }
        let scale = self
            .collective
            .reduce_max(308, std::slice::from_ref(&scale))
            .ok()
            .and_then(|values| values.into_iter().next())
            .unwrap_or(T::infinity());
        let bound = T::from_f64(1e12).unwrap() * scale;
        let finite = scale.is_finite()
            && v.tau.is_finite()
            && v.kappa.is_finite()
            && v.border_s.is_finite()
            && v.border_z.is_finite()
            && v.blocks
                .iter()
                .all(|block| block.x.is_finite() && block.s.is_finite() && block.z.is_finite());
        self.collective
            .all_true(306, finite && nx <= bound && nz <= bound)
            .unwrap_or(false)
    }
}
