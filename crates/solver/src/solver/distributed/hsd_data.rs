//! Data, residual, progress-information and solution adapters for the owned
//! HSD storage.  The numerical formulas remain in the default implementations;
//! this module only reduces owner-local state and delegates the scalar policy.

use super::*;
use crate::solver::core::{
    traits::{Info, InfoPrint, ProblemData, Residuals, Solution, Variables},
    SolverStatus,
};
use crate::timers::Timers;

impl<T: FloatT> ProblemData<T> for OwnedData<T> {
    type V = OwnedVariables<T>;
    type C = OwnedCones<T>;
    type SE = DefaultSettings<T>;

    /// `OwnedData` is constructed from the prepared default problem.  Its
    /// blocks already carry the equilibrated coefficients, so running Ruiz a
    /// second time would change the problem seen by the local KKT systems.
    fn equilibrate(&mut self, _cones: &OwnedCones<T>, _settings: &DefaultSettings<T>) {}
}

impl<T: FloatT> OwnedResiduals<T> {
    fn update_impl(
        &mut self,
        variables: &OwnedVariables<T>,
        data: &OwnedData<T>,
        pool: Option<std::sync::Arc<rayon::ThreadPool>>,
    ) {
        // Fewer owners than workers: lend each owner's sampled products the
        // rank pool (a nested install on the same pool runs inline).
        let lend = pool
            .as_ref()
            .filter(|p| self.blocks.len() < p.current_num_threads())
            .cloned();
        let update = |(residuals, variables, data, ids): (
            &mut DefaultResiduals<T>,
            &DefaultVariables<T>,
            &DefaultProblemData<T>,
            &OwnerIndices,
        )| {
            {
                residuals.sampled_pool = lend.clone();
            }
            residuals.update_counted(variables, data, Some(&ids.counted_rows));
            {
                residuals.sampled_pool = None;
            }
        };

        for_blocks!(
            pool.as_deref(),
            (
                &mut self.blocks,
                &variables.blocks,
                &data.blocks,
                &data.layout.owners
            ),
            update
        );

        // Equality rows are replicated in every owner.  Add their local
        // operator contributions once, then apply the affine b term from the
        // canonical owner.  The reduced values are copied back to all owners
        // so each local KKT residual observes the same border coordinate.
        // Pack all replicated equality coordinates into one vector reduction.
        // A nonfinite local coordinate is encoded as NaN, preserving the old
        // per-coordinate all_true/reduce_sum failure semantics while reusing
        // the persistent border_inf workspace as the payload.
        for (j, &row) in data.layout.border_rows.iter().enumerate() {
            let mut values = self
                .blocks
                .iter()
                .zip(&data.layout.owners)
                .map(|(residuals, ids)| {
                    let local = ids.rows.binary_search(&row).unwrap();
                    residuals.rz_inf[local]
                });
            let mut inf = values.next().unwrap_or_else(|| T::zero());
            for value in values {
                inf += value;
            }
            self.border_inf[j] = if inf.is_finite() { inf } else { T::nan() };
        }
        if !self.border_inf.is_empty() {
            if self
                .collective
                .reduce_sum_in_place(399, &mut self.border_inf)
                .is_err()
            {
                self.border_inf.fill(T::infinity());
            }
            for (j, value) in self.border_inf.iter().copied().enumerate() {
                self.border_residual[j] = value - variables.tau * data.border_b[j];
            }
        }

        for (residuals, ids) in self.blocks.iter_mut().zip(&data.layout.owners) {
            for (j, &row) in data.layout.border_rows.iter().enumerate() {
                let local = ids.rows.binary_search(&row).unwrap();
                residuals.rz_inf[local] = self.border_inf[j];
                residuals.rz[local] = self.border_residual[j];
            }
        }

        let mut summary = if self.blocks.is_empty() {
            ResidualSummary::empty(data.componentwise_enabled.then_some(T::zero()))
        } else {
            ResidualSummary::from_owners(
                self.blocks.iter().zip(&data.layout.owners).enumerate().map(
                    |(owner, (residuals, ids))| {
                        let variables = &variables.blocks[owner];
                        let data = &data.blocks[owner];
                        let eq = &data.equilibration;
                        let row = |values, scales| NormView {
                            values,
                            scales,
                            indices: Some(ids.counted_rows.as_slice()),
                        };
                        ResidualOwner {
                            products: residuals.products,
                            norms: [
                                NormView::dense(&variables.x, &eq.d),
                                row(&variables.z, &eq.e),
                                row(&variables.s, &eq.einv),
                                NormView::dense(&residuals.rx_inf, &eq.dinv),
                                NormView::dense(&residuals.Px, &eq.dinv),
                                row(&residuals.rz_inf, &eq.einv),
                                row(&residuals.rz, &eq.einv),
                                NormView::dense(&residuals.rx, &eq.dinv),
                            ],
                            dual_componentwise: residuals.dual_componentwise,
                        }
                    },
                ),
                pool.as_deref(),
            )
            .expect("owned residuals share one componentwise setting")
        };

        // Owner zero is the canonical source for replicated equality rows.
        // It can be a border-only owner, in which case the rank-local view
        // has no numeric block and the ordinary owner summary above has no
        // row products or norms to contribute.  Add that shared slice exactly
        // once before the rank reduction; all other ranks contribute the
        // identity for these coordinates.
        if self.blocks.is_empty() && self.collective.rank() == 0 {
            summary.add_border(
                ResidualProducts {
                    qx: T::zero(),
                    bz: data.border_b.dot(&variables.border_z),
                    sz: variables.border_s.dot(&variables.border_z),
                    xpx: T::zero(),
                },
                &variables.border_z,
                &variables.border_s,
                &self.border_inf,
                &self.border_residual,
                &data.border_e,
                &data.border_einv,
            );
        }

        let p = summary.products;
        let products_finite = self
            .collective
            .all_true(453, [p.qx, p.bz, p.sz, p.xpx].is_finite())
            .unwrap_or(false);
        let products = self
            .collective
            .reduce_sum(450, &[p.qx, p.bz, p.sz, p.xpx])
            .ok();
        let (local_scales, local_sumsq) = summary.norm_parts();
        let norms_finite = self
            .collective
            .all_true(454, local_scales.is_finite() && local_sumsq.is_finite())
            .unwrap_or(false);
        let scale_payload = if norms_finite {
            local_scales
        } else {
            [T::infinity(); 8]
        };
        let scales = self
            .collective
            .reduce_max(451, &scale_payload)
            .ok()
            .and_then(|values| values.try_into().ok())
            .unwrap_or([T::infinity(); 8]);
        let mut normalized = [T::zero(); 8];
        for i in 0..8 {
            if scales[i].is_zero() {
                normalized[i] = T::zero();
            } else if norms_finite && scales[i].is_finite() {
                let ratio = local_scales[i] / scales[i];
                normalized[i] = local_sumsq[i] * ratio * ratio;
            } else {
                normalized[i] = T::zero();
            }
        }
        let sumsq = self
            .collective
            .reduce_sum(455, &normalized)
            .ok()
            .and_then(|values| values.try_into().ok())
            .unwrap_or([T::infinity(); 8]);
        let norms = if norms_finite {
            std::array::from_fn(|i| scales[i] * sumsq[i].sqrt())
        } else {
            [T::infinity(); 8]
        };
        let dual_componentwise = if data.componentwise_enabled {
            let local = summary.dual_componentwise.unwrap_or(T::zero());
            let finite = self
                .collective
                .all_true(456, local.is_finite())
                .unwrap_or(false);
            let payload = if finite { local } else { T::infinity() };
            self.collective
                .reduce_max(452, &[payload])
                .ok()
                .and_then(|values| values.into_iter().next())
                .map(|value| if finite { value } else { T::infinity() })
        } else {
            None
        };
        let values = products
            .filter(|values| values.len() == 4)
            .unwrap_or_else(|| vec![T::nan(); 4]);
        summary.replace_global(
            ResidualProducts {
                qx: products_finite.then_some(values[0]).unwrap_or(T::nan()),
                bz: products_finite.then_some(values[1]).unwrap_or(T::nan()),
                sz: products_finite.then_some(values[2]).unwrap_or(T::nan()),
                xpx: products_finite.then_some(values[3]).unwrap_or(T::nan()),
            },
            norms,
            dual_componentwise,
        );

        let tau = variables.tau;
        let kappa = variables.kappa;
        let rτ = summary.products.qx + summary.products.bz + kappa + summary.products.xpx / tau;
        // Every local affine/combined RHS carries the same homogeneous
        // scalar residual.  The per-owner `bz` terms above intentionally
        // omit replicated border rows, so leave no owner-local value here.
        for residuals in &mut self.blocks {
            residuals.rτ = rτ;
        }
        self.scalar.products = summary.products;
        self.scalar.rτ = rτ;
        self.scalar.dual_componentwise = summary.dual_componentwise;
        self.summary = Some(summary);
    }
}

impl<T: FloatT> Residuals<T> for OwnedResiduals<T> {
    type D = OwnedData<T>;
    type V = OwnedVariables<T>;

    fn update(&mut self, variables: &OwnedVariables<T>, data: &OwnedData<T>) {
        self.update_impl(variables, data, None);
    }

    fn update_with_pool(
        &mut self,
        variables: &OwnedVariables<T>,
        data: &OwnedData<T>,
        pool: Option<std::sync::Arc<rayon::ThreadPool>>,
    ) {
        self.update_impl(variables, data, pool);
    }
}

impl<T: FloatT> InfoPrint<T> for OwnedInfo<T> {
    type D = OwnedData<T>;
    type C = OwnedCones<T>;
    type SE = DefaultSettings<T>;

    fn print_target(&mut self) -> &mut dyn std::io::Write {
        self.0.print_target()
    }

    fn print_configuration(
        &mut self,
        settings: &DefaultSettings<T>,
        data: &OwnedData<T>,
        cones: &OwnedCones<T>,
    ) -> std::io::Result<()> {
        if data.blocks.len() == 1 {
            self.0
                .print_configuration(settings, &data.blocks[0], &cones.blocks[0])
        } else if settings.verbose {
            writeln!(
                self.0.print_target(),
                "owned problem: {} variables, {} constraints, {} local blocks",
                data.layout.n,
                data.layout.m,
                data.blocks.len()
            )
        } else {
            Ok(())
        }
    }

    fn print_status_header(&mut self, settings: &DefaultSettings<T>) -> std::io::Result<()> {
        self.0.print_status_header(settings)
    }

    fn print_status(&mut self, settings: &DefaultSettings<T>) -> std::io::Result<()> {
        self.0.print_status(settings)
    }

    fn print_footer(&mut self, settings: &DefaultSettings<T>) -> std::io::Result<()> {
        self.0.print_footer(settings)
    }
}

impl<T: FloatT> Info<T> for OwnedInfo<T> {
    type V = OwnedVariables<T>;
    type R = OwnedResiduals<T>;

    fn reset(&mut self, timers: &mut Timers) {
        self.0.reset(timers);
    }

    fn set_linear_solver_info(&mut self, info: crate::solver::kkt::LinearSolverInfo) {
        self.0.set_linear_solver_info(info);
    }

    fn post_process(&mut self, residuals: &OwnedResiduals<T>, settings: &DefaultSettings<T>) {
        self.0.post_process(&residuals.scalar, settings);
    }

    fn finalize(&mut self, timers: &mut Timers) {
        self.0.finalize(timers);
    }

    fn update(
        &mut self,
        data: &mut OwnedData<T>,
        variables: &OwnedVariables<T>,
        residuals: &OwnedResiduals<T>,
        timers: &Timers,
    ) {
        let summary = residuals
            .summary
            .as_ref()
            .expect("residuals must be updated before info");
        self.0.update_from_summary(
            summary,
            variables.tau,
            variables.kappa,
            T::recip(data.c),
            data.normb,
            data.normq,
        );
        self.0.solve_time = timers.total_time().as_secs_f64();
        let local_time = T::from_f64(self.0.solve_time).unwrap_or(T::infinity());
        self.0.solve_time = data
            .collective
            .reduce_max(552, std::slice::from_ref(&local_time))
            .ok()
            .and_then(|values| values.into_iter().next())
            .and_then(|value| value.to_f64())
            .unwrap_or(f64::INFINITY);
    }

    fn update_with_pool(
        &mut self,
        data: &mut OwnedData<T>,
        variables: &OwnedVariables<T>,
        residuals: &OwnedResiduals<T>,
        timers: &Timers,
        _pool: Option<std::sync::Arc<rayon::ThreadPool>>,
    ) {
        self.update(data, variables, residuals, timers);
    }

    fn check_termination(
        &mut self,
        residuals: &OwnedResiduals<T>,
        settings: &DefaultSettings<T>,
        iter: u32,
    ) -> bool {
        let local = self.0.check_termination(&residuals.scalar, settings, iter);
        let done = match residuals.collective.all_true(550, local) {
            Ok(done) => done,
            Err(_) => {
                self.0.status = SolverStatus::NumericalError;
                return true;
            }
        };
        match residuals.collective.agree_u32(551, self.0.status as u32) {
            Ok(true) => done,
            Ok(false) | Err(_) => {
                self.0.status = SolverStatus::NumericalError;
                true
            }
        }
    }

    fn save_prev_iterate(
        &mut self,
        variables: &OwnedVariables<T>,
        prev_variables: &mut OwnedVariables<T>,
    ) {
        let empty = DefaultVariables::<T>::new(0, 0);
        let mut previous_empty = DefaultVariables::<T>::new(0, 0);
        self.0.save_prev_iterate(&empty, &mut previous_empty);
        for (current, previous) in variables
            .blocks
            .iter()
            .zip(prev_variables.blocks.iter_mut())
        {
            previous.copy_from(current);
        }
        prev_variables.tau = variables.tau;
        prev_variables.kappa = variables.kappa;
        prev_variables.border_z.copy_from_slice(&variables.border_z);
        prev_variables.border_s.copy_from_slice(&variables.border_s);
    }

    fn reset_to_prev_iterate(
        &mut self,
        variables: &mut OwnedVariables<T>,
        prev_variables: &OwnedVariables<T>,
    ) {
        let mut empty = DefaultVariables::<T>::new(0, 0);
        let previous_empty = DefaultVariables::<T>::new(0, 0);
        self.0.reset_to_prev_iterate(&mut empty, &previous_empty);
        for (current, previous) in variables.blocks.iter_mut().zip(&prev_variables.blocks) {
            current.copy_from(previous);
        }
        variables.tau = prev_variables.tau;
        variables.kappa = prev_variables.kappa;
        variables.border_z.copy_from_slice(&prev_variables.border_z);
        variables.border_s.copy_from_slice(&prev_variables.border_s);
    }

    fn save_scalars(&mut self, mu: T, alpha: T, sigma: T, iter: u32) {
        self.0.save_scalars(mu, alpha, sigma, iter);
    }

    fn get_status(&self) -> SolverStatus {
        self.0.get_status()
    }

    fn set_status(&mut self, status: SolverStatus) {
        self.0.set_status(status);
    }
}

impl<T: FloatT> Solution<T> for OwnedSolution<T> {
    type D = OwnedData<T>;
    type V = OwnedVariables<T>;
    type I = OwnedInfo<T>;
    type SE = DefaultSettings<T>;

    fn post_process(
        &mut self,
        data: &OwnedData<T>,
        variables: &mut OwnedVariables<T>,
        info: &OwnedInfo<T>,
        settings: &DefaultSettings<T>,
    ) {
        let infeasible = info.0.status.is_infeasible();
        self.0.status = info.0.status;
        self.0.obj_val = if infeasible {
            T::nan()
        } else {
            info.0.cost_primal
        };
        self.0.obj_val_dual = if infeasible {
            T::nan()
        } else {
            info.0.cost_dual
        };
        self.0.iterations = info.0.iterations;
        self.0.r_prim = info.0.res_primal;
        self.0.r_dual = info.0.res_dual;

        let is_root = variables.collective.rank() == 0;
        let (n, m) = if is_root {
            (data.layout.n, data.layout.m)
        } else {
            (0, 0)
        };
        let mut point = DefaultVariables::new(n, m);
        let scalar = if infeasible {
            variables.kappa.recip()
        } else {
            variables.tau.recip()
        };
        for ((variables, block), ids) in variables
            .blocks
            .iter_mut()
            .zip(&data.blocks)
            .zip(&data.layout.owners)
        {
            variables.unscale(block, infeasible);
            if !is_root {
                continue;
            }
            for (local, &global) in ids.columns.iter().enumerate() {
                point.x[global] = variables.x[local];
            }
            for &local in &ids.counted_rows {
                let global = ids.rows[local];
                point.s[global] = variables.s[local];
                point.z[global] = variables.z[local];
            }
        }
        variables.tau *= scalar;
        variables.kappa *= scalar;
        variables.border_s.hadamard(&data.border_einv).scale(scalar);
        variables
            .border_z
            .hadamard(&data.border_e)
            .scale(scalar * data.c.recip());
        point.τ = variables.tau;
        point.κ = variables.kappa;
        if is_root {
            for (j, &row) in data.global_layout.border_rows.iter().enumerate() {
                point.s[row] = variables.border_s[j];
                point.z[row] = variables.border_z[j];
            }
        }

        if variables.collective.size() > 1 {
            let full = &data.global_layout;
            let mut ranges = Vec::with_capacity(full.owners.len());
            let mut offset = 0usize;
            for ids in &full.owners {
                let length = ids.columns.len() + 2 * ids.counted_rows.len();
                ranges.push((offset, length));
                offset += length;
            }
            let mut payload = Vec::new();
            let payload_owner_ids: Vec<usize> = if variables.collective.size() > 1 {
                vec![variables.collective.rank()]
            } else {
                variables.all_owner_ids.clone()
            };
            for global_owner in payload_owner_ids {
                let ids = &full.owners[global_owner];
                let block = variables
                    .owner_ids
                    .iter()
                    .position(|&owner| owner == global_owner)
                    .and_then(|local_owner| variables.blocks.get(local_owner));
                for (local, _) in ids.columns.iter().enumerate() {
                    payload.push(block.map_or(T::zero(), |value| value.x[local]));
                }
                for &local in &ids.counted_rows {
                    let border = full
                        .border_rows
                        .iter()
                        .position(|&row| row == ids.rows[local]);
                    payload.push(block.map_or_else(
                        || border.map_or(T::zero(), |j| variables.border_s[j]),
                        |value| value.s[local],
                    ));
                }
                for &local in &ids.counted_rows {
                    let border = full
                        .border_rows
                        .iter()
                        .position(|&row| row == ids.rows[local]);
                    payload.push(block.map_or_else(
                        || border.map_or(T::zero(), |j| variables.border_z[j]),
                        |value| value.z[local],
                    ));
                }
            }
            let gathered = match variables.collective.gather_root(600, &payload, &ranges, 0) {
                Ok(values) => values,
                Err(_) => {
                    self.0.status = SolverStatus::NumericalError;
                    None
                }
            };
            if variables.collective.rank() == 0 {
                if let Some(values) = gathered {
                    point.x.fill(T::zero());
                    point.s.fill(T::zero());
                    point.z.fill(T::zero());
                    for (owner, ids) in full.owners.iter().enumerate() {
                        let (begin, length) = ranges[owner];
                        let end = begin + length;
                        if end > values.len() {
                            self.0.status = SolverStatus::NumericalError;
                            break;
                        }
                        let segment = &values[begin..end];
                        let mut cursor = ids.columns.len();
                        for (local, &global) in ids.columns.iter().enumerate() {
                            point.x[global] = segment[local];
                        }
                        for &local in &ids.counted_rows {
                            point.s[ids.rows[local]] = segment[cursor];
                            cursor += 1;
                        }
                        for &local in &ids.counted_rows {
                            point.z[ids.rows[local]] = segment[cursor];
                            cursor += 1;
                        }
                    }
                }
            } else {
                point.x.fill(T::zero());
                point.s.fill(T::zero());
                point.z.fill(T::zero());
            }
        }

        if !is_root {
            self.0.x.clear();
            self.0.s.clear();
            self.0.z.clear();
            return;
        }

        let reversed = data
            .chordal_info
            .as_ref()
            .map(|chordal| chordal.decomp_reverse(&point, &data.internal_cones, settings));
        let point = reversed.as_ref().unwrap_or(&point);

        if let Some(presolver) = &data.presolver {
            presolver.reverse_presolve(&mut self.0, point);
        } else {
            self.0.x.copy_from_slice(&point.x);
            self.0.s.copy_from_slice(&point.s);
            self.0.z.copy_from_slice(&point.z);
        }
    }

    fn finalize(&mut self, info: &OwnedInfo<T>) {
        self.0.solve_time = info.0.solve_time;
    }
}
