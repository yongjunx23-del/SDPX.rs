use super::super::refinement::{refine, Refinement};
use super::*;

impl<T: FloatT> HasLinearSolverInfo for CondensedKKTSolver<T> {
    fn linear_solver_info(&self) -> LinearSolverInfo {
        let mut info = self.reduced.linear_solver_info();
        info.name = if self.sampled.is_some() {
            format!("condensed_sampled_{}", info.name)
        } else {
            format!("condensed_{}", info.name)
        };
        info
    }
}

impl<T: FloatT> KKTSolver<T> for CondensedKKTSolver<T> {
    fn set_sampled_operator(&mut self, operator: Arc<SampledOperator<T>>) {
        debug_assert!(
            !self.local_only || operator.is_local_only(),
            "local KKT requires local sampled policy"
        );
        for (bi, sampled_block) in operator.blocks().iter().enumerate() {
            if let Some(block) = self
                .blocks
                .iter_mut()
                .find(|b| b.rows.start == sampled_block.row_start)
            {
                if let Scaling::Psd(p) = &mut block.scaling {
                    p.mat3c = Matrix::zeros(p.Rinv.size());
                    p.sampled = Some(SampledPsd {
                        work: SampledSchurWorkspace::new(sampled_block),
                        pair_lanes: Vec::new(),
                        adjoint: vec![T::zero(); sampled_block.column_count()],
                        operator: Arc::clone(&operator),
                        block: bi,
                    });
                }
            }
        }
        self.sampled = Some((Arc::clone(&operator), SampledWorkspace::new(&operator)));
        self.prepare_shared_pool();
        self.refresh_parallel_plan();
    }
    fn update(&mut self, cones: &CompositeCone<T>, settings: &CoreSettings<T>) -> bool {
        self.update_partition(cones, settings, true)
    }

    fn setrhs(&mut self, x: &[T], z: &[T]) {
        self.b[..self.n].copy_from_slice(x);
        self.b[self.n..].copy_from_slice(z);
    }

    fn counters(&self) -> crate::solver::core::kktsolvers::SolveCounters {
        let mut counters = self.reduced.counters();
        counters.rhs_applied = self.counters.rhs_applied;
        counters.batches = self.counters.batches;
        counters
    }

    fn reset_solve(&mut self) {
        self.scaled_valid.fill(false);
        self.counters = Default::default();
        self.reduced.reset_solve();
    }

    fn scaled_solution(&self, column: usize) -> Option<&[T]> {
        self.scaled_valid
            .get(column)
            .copied()
            .unwrap_or(false)
            .then(|| &self.scaled_solutions[column * self.A.m..(column + 1) * self.A.m])
    }

    fn solve_many(
        &mut self,
        n: usize,
        rhs: &[T],
        out: &mut [T],
        cols: usize,
        settings: &CoreSettings<T>,
    ) -> Vec<bool> {
        assert_eq!(n, self.n);
        let width = self.A.m + n;
        assert_eq!(rhs.len(), cols * width);
        assert_eq!(out.len(), rhs.len());
        if cols == 0 {
            return Vec::new();
        }
        self.scaled_valid.clear();
        self.scaled_valid.resize(cols, false);
        self.scaled_solutions.resize(cols * self.A.m, T::zero());
        self.counters.batches += 1;
        self.counters.rhs_applied += cols as u64;
        let reduced_width = n + self.retained_rows.len();
        let half_width: usize = if self.fused_sampled() {
            self.blocks
                .iter()
                .map(|b| match &b.scaling {
                    Scaling::Psd(p) if p.sampled.is_some() => p.mat3c.data().len(),
                    _ => 0,
                })
                .sum()
        } else {
            0
        };
        let mut b = std::mem::take(&mut self.batch_rhs);
        let mut reduced_out = std::mem::take(&mut self.batch_out);
        let mut halves = std::mem::take(&mut self.batch_halves);
        b.resize(cols * reduced_width, T::zero());
        reduced_out.resize(b.len(), T::zero());
        halves.resize(cols * half_width, T::zero());
        let valid: Vec<bool> = rhs
            .chunks(width)
            .map(|b| local_success(self.local_only, b.is_finite()))
            .collect();
        for c in 0..cols {
            let dest = &mut b[c * reduced_width..(c + 1) * reduced_width];
            if !valid[c] {
                dest.fill(T::zero());
                continue;
            }
            self.prepare_rhs(&rhs[c * width..(c + 1) * width]);
            dest[..n].copy_from_slice(&self.workx);
            dest[n..].copy_from_slice(&self.retained_rhs);
            let mut offset = c * half_width;
            if half_width != 0 {
                for block in &self.blocks {
                    if let Scaling::Psd(p) = &block.scaling {
                        if p.sampled.is_some() {
                            let values = p.mat3c.data();
                            halves[offset..offset + values.len()].copy_from_slice(values);
                            offset += values.len();
                        }
                    }
                }
            }
        }
        let mut flags = self
            .reduced
            .solve_many(n, &b, &mut reduced_out, cols, settings);
        let mut x = std::mem::take(&mut self.x);
        for c in 0..cols {
            flags[c] = local_success(self.local_only, flags[c]) && valid[c];
            if !flags[c] {
                continue;
            }
            let rhs = &rhs[c * width..(c + 1) * width];
            let r = &reduced_out[c * reduced_width..(c + 1) * reduced_width];
            x[..n].copy_from_slice(&r[..n]);
            self.retained_rhs.copy_from_slice(&r[n..]);
            let mut offset = c * half_width;
            if half_width != 0 {
                for block in &mut self.blocks {
                    if let Scaling::Psd(p) = &mut block.scaling {
                        if p.sampled.is_some() {
                            let values = p.mat3c.data_mut();
                            values.copy_from_slice(&halves[offset..offset + values.len()]);
                            offset += values.len();
                        }
                    }
                }
            }
            flags[c] = local_success(self.local_only, self.recover_rhs(&mut x, rhs))
                && self.refine_solution(&mut x, rhs, settings);
            if flags[c] {
                if settings.iterative_refinement_enable {
                    self.scaled_solutions[c * self.A.m..(c + 1) * self.A.m]
                        .copy_from_slice(&self.workh);
                    self.scaled_valid[c] = true;
                }
                out[c * width..(c + 1) * width].copy_from_slice(&x);
            }
        }
        self.x = x;
        self.batch_rhs = b;
        self.batch_out = reduced_out;
        self.batch_halves = halves;
        flags
    }

    fn solve(
        &mut self,
        lhsx: Option<&mut [T]>,
        lhsz: Option<&mut [T]>,
        settings: &CoreSettings<T>,
    ) -> bool {
        // Move reusable buffers to keep operator scratch and refinement storage
        // disjoint. Sampled operator caches initialize on their first pooled use;
        // subsequent calls reuse them without precision conversion.
        self.scaled_valid.clear();
        self.scaled_valid.push(false);
        self.counters.rhs_applied += 1;
        let b = std::mem::take(&mut self.b);
        let mut x = std::mem::take(&mut self.x);
        let success = (|| {
            if !local_success(self.local_only, b.is_finite())
                || !local_success(self.local_only, self.solve_raw(&mut x, &b, settings))
                || !self.refine_solution(&mut x, &b, settings)
            {
                return false;
            }
            if settings.iterative_refinement_enable {
                self.scaled_solutions.resize(self.A.m, T::zero());
                self.scaled_solutions.copy_from_slice(&self.workh);
                self.scaled_valid[0] = true;
            }
            // As in upstream DirectLDL, refinement may stop at a finite
            // stalled approximation; ordinary solver convergence is unchanged.
            if let Some(v) = lhsx {
                v.copy_from_slice(&x[..self.n]);
            }
            if let Some(v) = lhsz {
                v.copy_from_slice(&x[self.n..]);
            }
            true
        })();

        self.b = b;
        self.x = x;
        success
    }

    fn escalate_regularization(&mut self) -> bool {
        self.reduced.escalate_regularization()
    }

    fn update_P(&mut self, P: &CscMatrix<T>) {
        self.scaled_valid.fill(false);
        assert_eq!(P.size(), self.P.size());
        assert_eq!(P.colptr, self.P.colptr);
        assert_eq!(P.rowval, self.P.rowval);
        self.P.nzval.copy_from_slice(&P.nzval);
    }

    fn update_A(&mut self, A: &CscMatrix<T>) {
        self.scaled_valid.fill(false);
        assert_eq!(A.size(), self.A.size());
        assert_eq!(A.colptr, self.A.colptr);
        assert_eq!(A.rowval, self.A.rowval);
        self.A.nzval.copy_from_slice(&A.nzval);
        for block in &mut self.blocks {
            if let Scaling::Psd(psd) = &mut block.scaling {
                psd.coefficient_plan_valid = false;
            }
        }
        for (v, &p) in self
            .retained_A
            .nzval
            .iter_mut()
            .zip(&self.retained_positions)
        {
            *v = A.nzval[p];
        }
        self.reduced.update_A(&self.retained_A);
    }
}

impl<T: FloatT> CondensedKKTSolver<T> {
    fn refine_solution(&mut self, x: &mut Vec<T>, b: &[T], settings: &CoreSettings<T>) -> bool {
        let mut error = std::mem::take(&mut self.error);
        let mut candidate = std::mem::take(&mut self.candidate);
        let success = refine(
            &mut LocalRefinement {
                kernel: self,
                x,
                b,
                error: &mut error,
                candidate: &mut candidate,
            },
            settings,
        );
        self.error = error;
        self.candidate = candidate;
        success
    }
}

struct LocalRefinement<'a, T: FloatT> {
    kernel: &'a mut CondensedKKTSolver<T>,
    x: &'a mut Vec<T>,
    b: &'a [T],
    error: &'a mut Vec<T>,
    candidate: &'a mut Vec<T>,
}
impl<T: FloatT> Refinement<T> for LocalRefinement<'_, T> {
    fn all_succeeded(&self, value: bool) -> bool {
        local_success(self.kernel.local_only, value)
    }
    fn decision_agrees(&self, value: u32) -> bool {
        self.kernel.local_only || crate::mpi::decision_agrees(value)
    }
    fn rhs_norm(&self) -> T {
        self.b.norm_inf()
    }
    fn residual(&mut self, candidate: bool, reuse: bool) -> T {
        self.kernel.residual(
            self.error,
            self.b,
            if candidate { self.candidate } else { self.x },
            reuse,
        )
    }
    fn solve_correction(&mut self, settings: &CoreSettings<T>) -> bool {
        self.kernel.solve_raw(self.candidate, self.error, settings)
    }
    fn add_correction(&mut self) {
        for (c, &v) in self.candidate.iter_mut().zip(self.x.iter()) {
            *c += v;
        }
    }
    fn accept_candidate(&mut self) {
        std::mem::swap(self.x, self.candidate);
    }
    fn restore_product(&mut self) {
        self.kernel.restore_scaled_product(self.x);
    }
}

#[cfg(test)]
mod mpi_tests {
    use super::*;
    use crate::solver::core::ScalingStrategy;

    fn rank_local_rhs_failure<T: FloatT>(rank: usize) {
        let kinds = vec![SupportedConeT::PSDTriangleConeT(2); 2];
        let mut cones = CompositeCone::<T>::new(&kinds);
        let (mut slack, mut dual) = (vec![T::zero(); 6], vec![T::zero(); 6]);
        cones.unit_initialization(&mut dual, &mut slack);
        assert!(cones.update_scaling(&slack, &dual, T::one(), ScalingStrategy::PrimalDual));
        let p = CscMatrix::identity(6);
        let a = CscMatrix::identity(6);
        let settings = CoreSettings::<T>::default();
        let mut kkt = CondensedKKTSolver::new(&p, &a, &kinds, &cones, &settings);
        assert!(kkt.update(&cones, &settings));
        let mut rhs = vec![T::one(); 24];
        if rank == 0 {
            rhs[0] = T::nan();
        }
        let mut out = vec![T::zero(); rhs.len()];
        // A single-rank bad column must skip preparation/recovery on all
        // ranks, while the next independent column still solves normally.
        assert_eq!(
            kkt.solve_many(6, &rhs, &mut out, 2, &settings),
            [false, true]
        );
        assert!(out[12..].is_finite());
        kkt.setrhs(&rhs[..6], &rhs[6..12]);
        assert!(!kkt.solve(None, None, &settings));
        rhs.fill(T::one());
        assert_eq!(
            kkt.solve_many(6, &rhs, &mut out, 2, &settings),
            [true, true]
        );
        let mut error = vec![T::zero(); 12];
        for c in 0..2 {
            let norm = kkt.residual(
                &mut error,
                &rhs[c * 12..(c + 1) * 12],
                &out[c * 12..(c + 1) * 12],
                false,
            );
            assert!(norm < T::from_f64(1e-7).unwrap());
        }
    }

    #[test]
    #[ignore]
    fn mpi_probe_two_rank_condensed_rhs_failure() {
        let mpi = crate::MpiContext::initialize();
        assert_eq!(mpi.size(), 2);
        rank_local_rhs_failure::<f64>(mpi.rank());
        rank_local_rhs_failure::<sdpx_arithmetic::Bits256>(mpi.rank());
        mpi.finish();
    }
}

impl<T: FloatT> CondensedKKTSolver<T> {
    pub(crate) fn update_partition(
        &mut self,
        cones: &CompositeCone<T>,
        settings: &CoreSettings<T>,
        factor: bool,
    ) -> bool {
        self.update_partition_with_pool(cones, settings, factor, cones.thread_pool())
    }

    /// Update on an externally owned pool. None explicitly restores serial
    /// execution; no cone pool is created or implicitly substituted.
    pub(crate) fn update_partition_with_pool(
        &mut self,
        cones: &CompositeCone<T>,
        settings: &CoreSettings<T>,
        factor: bool,
        pool: Option<Arc<rayon::ThreadPool>>,
    ) -> bool {
        self.scaled_valid.fill(false);
        assert_eq!(self.blocks.len(), cones.len());
        // Follow this update's actual pool, including removal/replacement.
        self.pool = pool;
        self.reduced.set_residual_pool(self.pool.clone());
        self.reduced.set_factor_pool(self.pool.clone());
        self.refresh_parallel_plan();
        self.parallel_assembly = self.pool.is_some()
            && self.blocks.iter().all(|block| match &block.scaling {
                Scaling::Psd(p) => p.schur_values.len() == triangular_number(p.columns.len()),
                _ => true,
            });
        let inner_sampled = self.inner_sampled;
        let pool = &self.pool;
        let world = self.mpi_world();
        let sync = |(block, cone, owned): (&mut Block<T>, &SupportedCone<T>, bool)| {
            match (&mut block.scaling, cone) {
                (Scaling::Psd(p), SupportedCone::PSDTriangleCone(c)) => {
                    p.R.copy_from_slice(c.scaling_R().data());
                    p.Rinv.copy_from_slice(c.scaling_Rinv().data());
                    if let Some(sampled) = &mut p.sampled {
                        // One parallel level only. Fine inner lanes are reserved
                        // for the single dominant block; handing the pool to every
                        // block multiplies tiny GEMM/SYRK tiles and pair lanes by
                        // the block count, which measured slower than the outer
                        // block level it competes with. Under MPI the Gram
                        // update runs on the owning rank only; the gathered
                        // Gram below republishes it to every rank.
                        if owned {
                            sampled.work.update_with_pool(
                                &sampled.operator.blocks()[sampled.block],
                                &p.Rinv,
                                if inner_sampled == Some(block.rows.start) {
                                    pool.as_deref()
                                } else {
                                    None
                                },
                            );
                        }
                    }
                    p.G.syrk(&p.R, T::one(), T::zero(), MatrixTriangle::Triu);
                    p.Ginv
                        .syrk(&p.Rinv.t(), T::one(), T::zero(), MatrixTriangle::Triu);
                    for j in 0..c.n {
                        for i in j + 1..c.n {
                            p.G[(i, j)] = p.G[(j, i)];
                            p.Ginv[(i, j)] = p.Ginv[(j, i)];
                        }
                    }
                    if !p.R.data().is_finite()
                        || !p.Rinv.data().is_finite()
                        || !p.G.data().is_finite()
                        || !p.Ginv.data().is_finite()
                    {
                        return false;
                    }
                }
                (Scaling::Orthant { w, .. }, SupportedCone::NonnegativeCone(c)) => {
                    w.copy_from_slice(&c.w);
                    if !w.iter().all(|v| v.is_finite() && *v > T::zero()) {
                        return false;
                    }
                }
                (Scaling::Zero, SupportedCone::ZeroCone(_)) => {}
                (Scaling::Soc { w, eta }, SupportedCone::SecondOrderCone(c)) => {
                    w.copy_from_slice(&c.w);
                    *eta = c.η;
                }
                (Scaling::Dense3(h), SupportedCone::ExponentialCone(_))
                | (Scaling::Dense3(h), SupportedCone::PowerCone(_)) => cone.get_Hs(h),
                (
                    Scaling::GenPower {
                        p,
                        q,
                        r,
                        d1,
                        d2,
                        mu,
                    },
                    SupportedCone::GenPowerCone(c),
                ) => {
                    p.copy_from_slice(&c.data.p);
                    q.copy_from_slice(&c.data.q);
                    r.copy_from_slice(&c.data.r);
                    d1.copy_from_slice(&c.data.d1);
                    *d2 = c.data.d2;
                    *mu = c.data.μ;
                }
                _ => panic!("condensed KKT cone structure changed"),
            }
            true
        };
        let __ts = std::time::Instant::now();
        // Rank split follows block update cost (Gram entries for sampled
        // blocks, ~n^3 proxy otherwise), not block count — a count split
        // turns size skew into allgatherv wait on every collective.
        let block_costs: Vec<u64> = self
            .blocks
            .iter()
            .map(|b| {
                let rows = (b.rows.end - b.rows.start) as f64;
                match &b.scaling {
                    Scaling::Psd(p) => p
                        .sampled
                        .as_ref()
                        .map(|s| s.work.gram_slice().len().max(1) as u64)
                        .unwrap_or_else(|| rows.powf(1.5).max(1.0) as u64),
                    _ => rows.max(1.0) as u64,
                }
            })
            .collect();
        let owned_range = world
            .map(|w| {
                let (b0, len) = crate::mpi::cost_ranges(&block_costs, w.size())[w.rank()];
                b0..b0 + len
            })
            .unwrap_or(0..self.blocks.len());
        let owned = |i: usize| owned_range.contains(&i);
        let valid = if inner_sampled.is_some() {
            if let Some(pool) = &self.pool {
                // Block-level parallelism is safe here: only the dominant
                // block consumes inner pool lanes, the others update
                // serially inside their own outer task.
                pool.install(|| {
                    self.blocks
                        .par_iter_mut()
                        .zip(cones.iter().as_slice().par_iter())
                        .enumerate()
                        .map(|(i, (b, c))| sync((b, c, owned(i))))
                        .reduce(|| true, |a, b| a & b)
                })
            } else {
                self.blocks
                    .iter_mut()
                    .zip(cones.iter())
                    .enumerate()
                    .map(|(i, (b, c))| sync((b, c, owned(i))))
                    .fold(true, |a, b| a & b)
            }
        } else if let Some(pool) = &self.pool {
            pool.install(|| {
                self.blocks
                    .par_iter_mut()
                    .zip(cones.iter().as_slice().par_iter())
                    .enumerate()
                    .map(|(i, (b, c))| sync((b, c, owned(i))))
                    .reduce(|| true, |a, b| a & b)
            })
        } else {
            self.blocks
                .iter_mut()
                .zip(cones.iter())
                .enumerate()
                .map(|(i, (b, c))| sync((b, c, owned(i))))
                .fold(true, |a, b| a & b)
        };
        if let Some(world) = world {
            // Republish each block's Gram to every rank. Only the owning
            // rank ran `update_with_pool`; the concatenated block-ordered
            // exchange reproduces the serial Grams bitwise. Ownership and
            // layout both follow the `owned_range` block partition above.
            let gram_lens: Vec<usize> = self
                .blocks
                .iter()
                .map(|b| match &b.scaling {
                    Scaling::Psd(p) => p.sampled.as_ref().map_or(0, |s| s.work.gram_slice().len()),
                    _ => 0,
                })
                .collect();
            let mut offsets = Vec::with_capacity(self.blocks.len() + 1);
            offsets.push(0usize);
            for &n in &gram_lens {
                offsets.push(offsets.last().unwrap() + n);
            }
            let gather_ranges: Vec<(usize, usize)> =
                crate::mpi::cost_ranges(&block_costs, world.size())
                    .iter()
                    .map(|&(b0, len)| (offsets[b0], offsets[b0 + len] - offsets[b0]))
                    .collect();
            let (g0, g1) = (offsets[owned_range.start], offsets[owned_range.end]);
            let mut local = Vec::with_capacity(g1 - g0);
            for i in owned_range.clone() {
                if let Scaling::Psd(p) = &self.blocks[i].scaling {
                    if let Some(s) = &p.sampled {
                        local.extend_from_slice(s.work.gram_slice());
                    }
                }
            }
            debug_assert_eq!(local.len(), g1 - g0);
            let mut all = vec![T::zero(); *offsets.last().unwrap()];
            world.gather_slice(crate::mpi::SITE_GRAM, &local, &gather_ranges, &mut all);
            for (i, block) in self.blocks.iter_mut().enumerate() {
                let n = gram_lens[i];
                if n == 0 {
                    continue;
                }
                if let Scaling::Psd(p) = &mut block.scaling {
                    p.sampled
                        .as_mut()
                        .unwrap()
                        .work
                        .set_gram(&all[offsets[i]..offsets[i] + n]);
                }
            }
        }
        if crate::receipt::profile_requested() {
            eprintln!(
                "PHASE sync {:?} (inner_sampled={:?})",
                __ts.elapsed(),
                inner_sampled
            );
        }
        crate::receipt::phase_record("sync", __ts.elapsed());
        // `valid` folds this rank's owned blocks only; a failing owner
        // returning early while peers proceed would hang the next
        // collective. Merge the flag before any rank leaves the call.
        let valid = match world {
            // Every rank must enter the collective, including a failing owner.
            Some(w) => w.allreduce_max_f64(if valid { 0.0 } else { 1.0 }) == 0.0,
            None => valid,
        };
        if !valid {
            return false;
        }
        let __t0 = std::time::Instant::now();
        if !self.assemble() {
            return false;
        }
        crate::receipt::phase("assemble", __t0.elapsed());
        self.reduced.update_P(&self.schur);
        let retained = &self.retained_indices;
        let __t1 = std::time::Instant::now();
        let retained_cones = cones
            .iter()
            .enumerate()
            .filter(|(i, _)| retained.binary_search(i).is_ok())
            .map(|(_, c)| c);
        let __r = if factor {
            self.reduced.update_from_cones(retained_cones, settings)
        } else {
            self.reduced.assemble_from_cones(retained_cones);
            true
        };
        crate::receipt::phase("cones_schur", __t1.elapsed());
        __r
    }
}
