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
        for (bi, sampled_block) in operator.blocks().iter().enumerate() {
            if let Some(block) = self
                .blocks
                .iter_mut()
                .find(|b| b.rows.start == sampled_block.row_start)
            {
                if let Scaling::Psd(p) = &mut block.scaling {
                    p.sampled = Some(SampledPsd {
                        work: SampledSchurWorkspace::new(sampled_block),
                        pair_lanes: Vec::new(),
                        operator: Arc::clone(&operator),
                        block: bi,
                    });
                }
            }
        }
        self.sampled = Some((Arc::clone(&operator), SampledWorkspace::new(&operator)));
        // Reserve within the existing contribution-storage cap even if the
        // handle starts at one worker and is later reconfigured.
        let cells: u128 = self
            .blocks
            .iter()
            .map(|b| match &b.scaling {
                Scaling::Psd(p) => triangular_number(p.columns.len()) as u128,
                _ => 0,
            })
            .sum();
        if cells <= 2 * self.schur.nzval.len() as u128 {
            for block in &mut self.blocks {
                if let Scaling::Psd(p) = &mut block.scaling {
                    p.schur_values
                        .resize(triangular_number(p.columns.len()), T::zero());
                }
            }
        }
        self.plan_threads = 0;
        self.refresh_parallel_plan();
    }
    fn update(&mut self, cones: &CompositeCone<T>, settings: &CoreSettings<T>) -> bool {
        assert_eq!(self.blocks.len(), cones.len());
        // CompositeCone can be reconfigured between solves. Follow its actual
        // current pool; reconfiguration never silently retains an old budget.
        self.pool = cones.thread_pool();
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
        let world = crate::mpi::World::get();
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
                    } else {
                        p.Ginv
                            .syrk(&p.Rinv.t(), T::one(), T::zero(), MatrixTriangle::Triu);
                    }
                    if p.sampled.is_none() {
                        for j in 0..c.n {
                            for i in j + 1..c.n {
                                p.Ginv[(i, j)] = p.Ginv[(j, i)];
                            }
                        }
                    }
                    if !p.R.data().is_finite()
                        || !p.Rinv.data().is_finite()
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
                    Scaling::Psd(p) => {
                        p.sampled.as_ref().map_or(0, |s| s.work.gram_slice().len())
                    }
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
        if std::env::var_os("SDPX_PROFILE").is_some() {
            eprintln!(
                "PHASE sync {:?} (inner_sampled={:?})",
                __ts.elapsed(),
                inner_sampled
            );
        }
        crate::receipt::phase_record("sync", __ts.elapsed());
        if !valid {
            return false;
        }
        let __t0 = std::time::Instant::now();
        if !self.assemble() {
            return false;
        }
        crate::receipt::phase("assemble", __t0.elapsed());
        self.reduced.update_P(&self.schur);
        self.counters.factorizations += 1;
        let retained = &self.retained_indices;
        let __t1 = std::time::Instant::now();
        let __r = self.reduced.update_from_cones(
            cones
                .iter()
                .enumerate()
                .filter(|(i, _)| retained.binary_search(i).is_ok())
                .map(|(_, c)| c),
            settings,
        );
        crate::receipt::phase("cones_schur", __t1.elapsed());
        __r
    }

    fn setrhs(&mut self, x: &[T], z: &[T]) {
        self.b[..self.n].copy_from_slice(x);
        self.b[self.n..].copy_from_slice(z);
    }

    fn counters(&self) -> crate::solver::core::kktsolvers::SolveCounters {
        self.counters
    }

    /// One wave carrying `ncols` columns. The default column loop applies every
    /// column to the same factorization, so this adds only the wave accounting
    /// that distinguishes "three RHS in two waves" from "three waves".
    fn solve_many(
        &mut self,
        n: usize,
        rhs: &[T],
        out: &mut [T],
        ncols: usize,
        settings: &CoreSettings<T>,
    ) -> Vec<bool> {
        assert_eq!(
            n, self.n,
            "solve_many x-width must match the condensed system"
        );
        self.counters.batches += 1;
        crate::solver::core::kktsolvers::solve_many_by_columns(self, n, rhs, out, ncols, settings)
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
        self.counters.rhs_applied += 1;
        let b = std::mem::take(&mut self.b);
        let mut x = std::mem::take(&mut self.x);
        let mut error = std::mem::take(&mut self.error);
        let mut candidate = std::mem::take(&mut self.candidate);
        let success = (|| {
            if !b.is_finite() || !self.solve_raw(&mut x, &b, settings) {
                return false;
            }
            if settings.iterative_refinement_enable {
                let normb = b.norm_inf();
                let mut norme = self.residual(&mut error, &b, &x, true);
                if !norme.is_finite() {
                    return false;
                }
                for _ in 0..settings.iterative_refinement_max_iter {
                    if norme
                        <= settings.iterative_refinement_abstol
                            + settings.iterative_refinement_reltol * normb
                    {
                        break;
                    }
                    let previous = norme;
                    if !self.solve_raw(&mut candidate, &error, settings) {
                        return false;
                    }
                    for (c, &v) in candidate.iter_mut().zip(&x) {
                        *c += v;
                    }
                    norme = self.residual(&mut error, &b, &candidate, false);
                    if !norme.is_finite() {
                        return false;
                    }
                    let ratio = previous / norme;
                    if ratio < settings.iterative_refinement_stop_ratio {
                        if ratio > T::one() {
                            std::mem::swap(&mut x, &mut candidate);
                        }
                        break;
                    }
                    std::mem::swap(&mut x, &mut candidate);
                }
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
        self.error = error;
        self.candidate = candidate;
        success
    }

    fn update_P(&mut self, P: &CscMatrix<T>) {
        assert_eq!(P.size(), self.P.size());
        assert_eq!(P.colptr, self.P.colptr);
        assert_eq!(P.rowval, self.P.rowval);
        self.P.nzval.copy_from_slice(&P.nzval);
    }

    fn update_A(&mut self, A: &CscMatrix<T>) {
        assert_eq!(A.size(), self.A.size());
        assert_eq!(A.colptr, self.A.colptr);
        assert_eq!(A.rowval, self.A.rowval);
        self.A.nzval.copy_from_slice(&A.nzval);
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
