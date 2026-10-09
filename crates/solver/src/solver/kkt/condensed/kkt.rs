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
        // Fused sampled solves condense and recover through Rinv and apply H
        // through G; only the MPI path's inverse application reads Ginv.
        let fused = self.mpi_world().is_none();
        for (bi, sampled_block) in operator.blocks().iter().enumerate() {
            if let Some(block) = self
                .blocks
                .iter_mut()
                .find(|b| b.rows.start == sampled_block.row_start)
            {
                if let Scaling::Psd(p) = &mut block.scaling {
                    // Sampled Schur values come from the Gram workspace; the
                    // per-column A entry lists (one record per nonzero) are
                    // only read by the materialized paths.
                    for column in &mut p.columns {
                        column.entries = Vec::new();
                    }
                    // Also release generic plans for partially sampled problems.
                    p.axpy_plans = Vec::new();
                    p.column_groups = Vec::new();
                    p.dense_indices = Vec::new();
                    p.dense_representatives = Vec::new();
                    p.dense_column_map = Vec::new();
                    p.dense_row_first = Vec::new();
                    p.dense_row_offsets = Vec::new();
                    p.dense_vectors = Vec::new();
                    p.dense_acc = Vec::new();
                    p.transform_lanes = Vec::new();
                    p.coefficient_support = Vec::new();
                    if fused {
                        p.mat3c = Matrix::zeros(p.Rinv.size());
                        p.Ginv = Matrix::zeros((0, 0));
                        p.ginv_cache = ResidueCache::default();
                    }
                    p.sampled = Some(SampledPsd {
                        work: SampledSchurWorkspace::new(sampled_block),
                        pair_lanes: Vec::new(),
                        adjoint: Vec::new(),
                        operator: Arc::clone(&operator),
                        block: bi,
                    });
                }
            }
        }
        let mut work = SampledWorkspace::new(&operator);
        work.enable_basis_caches();
        self.sampled = Some((Arc::clone(&operator), work));
        // Every product now uses the factor operator, including its linear rows.
        self.a_panel = None;
        // With every PSD block sampled and only retained (zero-cone) rows
        // besides, products use the operator's factors and never read this
        // copy's values; keep its structure and release the values.
        let values_needed = self.blocks.iter().any(|b| match &b.scaling {
            Scaling::Psd(p) => p.sampled.is_none(),
            Scaling::Zero => false,
            _ => true,
        });
        if !values_needed {
            self.A.nzval = Vec::new();
            // Single-process products never read the pattern either; only
            // the MPI path shards products over it.
            if self.mpi_world().is_none() {
                self.A.colptr = vec![0; self.A.n + 1];
                self.A.rowval = Vec::new();
            }
        } else {
            assert!(
                !self.A.nzval.is_empty() || self.A.rowval.is_empty(),
                "condensed KKT built without A values needs every PSD block sampled"
            );
        }
        // Sampled A updates are rejected; the reduced KKT owns these values.
        self.retained_A = CscMatrix::zeros(self.retained_A.size());
        self.retained_positions = Vec::new();
        // Owner kernels reserve together after installing all factors.
        if !self.local_only {
            self.prepare_shared_pool(&mut parallel_assembly_budget_bytes());
        }
        self.refresh_parallel_plan();
    }
    fn update(&mut self, cones: &CompositeCone<T>, settings: &CoreSettings<T>) -> bool {
        self.update_partition(cones, settings, true)
    }

    fn setrhs(&mut self, x: &[T], z: &[T]) {
        self.b[..self.n].copy_from_slice(x);
        self.b[self.n..].copy_from_slice(z);
    }

    fn counters(&self) -> crate::solver::kkt::SolveCounters {
        let mut counters = self.reduced.counters();
        counters.rhs_applied = self.counters.rhs_applied;
        counters.batches = self.counters.batches;
        counters.outer_refinements = self.counters.outer_refinements;
        counters
    }

    fn reset_solve(&mut self) {
        self.gmres_continuation = false;
        self.scaled_valid.fill(false);
        self.counters = Default::default();
        self.reduced.reset_solve();
    }

    fn scaled_solution(&self, column: usize) -> Option<&[T]> {
        self.scaled_valid
            .get(column)
            .copied()
            .unwrap_or(false)
            .then(|| {
                // The final accepted product stays in refinement workspace;
                // earlier columns must survive the next right-hand side.
                if column + 1 == self.scaled_valid.len() {
                    &self.workh
                } else {
                    &self.scaled_solutions[column * self.A.m..(column + 1) * self.A.m]
                }
            })
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
        self.scaled_solutions
            .resize((cols - 1) * self.A.m, T::zero());
        self.counters.batches += 1;
        self.counters.rhs_applied += cols as u64;
        let reduced_width = n + self.retained_rows.len();
        let half_width: usize = if self.fused_sampled() {
            self.blocks
                .iter()
                .map(|b| match &b.scaling {
                    Scaling::Psd(p) if p.sampled.is_some() => triangular_number(p.mat3c.nrows()),
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
                            // Congruences mirror their upper triangle exactly;
                            // keep one copy of each value between RHS columns.
                            let side = p.mat3c.nrows();
                            for j in 0..side {
                                let values = &p.mat3c.data()[j * side..j * side + j + 1];
                                halves[offset..offset + values.len()].copy_from_slice(values);
                                offset += values.len();
                            }
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
                            let side = p.mat3c.nrows();
                            for j in 0..side {
                                let saved = &halves[offset..offset + j + 1];
                                p.mat3c.data_mut()[j * side..j * side + j + 1]
                                    .copy_from_slice(saved);
                                for (i, &value) in saved[..j].iter().enumerate() {
                                    p.mat3c[(j, i)] = value;
                                }
                                offset += j + 1;
                            }
                        }
                    }
                }
            }
            flags[c] = local_success(self.local_only, self.recover_rhs(&mut x, rhs))
                && self.refine_solution(&mut x, rhs, settings);
            if flags[c] {
                if settings.iterative_refinement_enable {
                    let m = self.A.m;
                    self.scaled_product();
                    if c + 1 != cols {
                        self.scaled_solutions[c * m..(c + 1) * m].copy_from_slice(&self.workh);
                    }
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
                self.scaled_product();
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

    fn retained_rows(&self) -> Option<&[usize]> {
        Some(&self.retained_rows)
    }

    fn refine_further(&mut self) {
        self.gmres_continuation = T::precision_bits() <= 53;
    }

    fn update_P(&mut self, P: &CscMatrix<T>) {
        self.scaled_valid.fill(false);
        // The solver built both patterns; only the values change.
        debug_assert!(
            P.size() == self.P.size() && P.colptr == self.P.colptr && P.rowval == self.P.rowval
        );
        self.P.nzval.copy_from_slice(&P.nzval);
    }

    fn update_A(&mut self, A: &CscMatrix<T>) {
        self.scaled_valid.fill(false);
        // The eliminated-row plan may cache constant A rows.
        self.eliminated = None;
        debug_assert_eq!(A.size(), self.A.size());
        // An empty copy means only the sampled factors define these rows.
        if !self.A.nzval.is_empty() {
            debug_assert!(A.colptr == self.A.colptr && A.rowval == self.A.rowval);
            self.A.nzval.copy_from_slice(&A.nzval);
            if self.a_panel.is_some() {
                self.a_panel = DenseColumns::new(&self.A);
            }
        }
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
                continued: false,
                basis: Vec::new(),
                directions: Vec::new(),
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
    /// GMRES-IR continues a stalled stationary refinement; the forward
    /// product of the raw solve no longer matches `x`.
    continued: bool,
    basis: Vec<Vec<T>>,
    directions: Vec<Vec<T>>,
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
            reuse && !self.continued,
        )
    }
    fn solve_correction(&mut self, settings: &CoreSettings<T>) -> bool {
        self.kernel.counters.outer_refinements += 1;
        self.kernel.solve_raw(self.candidate, self.error, settings)
    }
    fn add_correction(&mut self) {
        crate::algebra::add_assign(self.candidate, self.x);
    }
    fn accept_candidate(&mut self) {
        std::mem::swap(self.x, self.candidate);
    }
    fn restore_product(&mut self) {
        self.kernel.restore_scaled_product(self.x);
    }
    /// max |e_x|, max |e_z|, and the x-row floor eps·max(|Aᵀ||z| + |b_x|).
    fn trace_parts(&mut self, candidate: bool) -> Option<String> {
        let n = self.kernel.n;
        let point: &[T] = if candidate { self.candidate } else { self.x };
        let (ex, ez) = self.error.split_at(n);
        let mut abs = vec![T::zero(); n];
        if let Some((operator, work)) = &mut self.kernel.sampled {
            let z: Vec<T> = point[n..].iter().map(|v| v.abs()).collect();
            operator.add_adjoint_abs(&mut abs, &z, work, self.kernel.pool.as_ref());
        }
        let floor = abs
            .iter()
            .zip(&self.b[..n])
            .map(|(&a, &b)| a + b.abs())
            .fold(T::zero(), T::max)
            * T::epsilon();
        let retained = self
            .kernel
            .retained_rows
            .iter()
            .fold(T::zero(), |m, &r| T::max(m, ez[r].abs()));
        Some(format!(
            "ex {:.3e} ez {:.3e} floor_x {:.3e} ret {:.3e}",
            ex.norm_inf(),
            ez.norm_inf(),
            floor,
            retained
        ))
    }
    // Each correction here runs a complete refined reduced solve: GMRES-IR
    // at this level multiplied those solves (L35, 30 iterations: 180 -> 690 s)
    // while stationary steps converge in about two, so above binary64 only
    // the reduced DirectLDL level runs GMRES-IR. In binary64, once the driver
    // sees a residual norm grow (`refine_further`), a stalled stationary
    // refinement continues with GMRES-IR here: late in the solve
    // the static shift and the Schur rounding leave a few slowly contracting
    // directions (SDP_gpp250-1: dual residual 1e-8 to 1e-7 from iteration 20
    // with the constant column's x-row residual at 1e-5 of its right-hand
    // side; GMRES-IR reaches Solved in 21 iterations).
    fn gmres_continuation(&mut self) -> bool {
        self.continued = self.kernel.gmres_continuation;
        self.continued
    }
    fn gmres_supported(&self) -> bool {
        self.continued
    }
    fn gmres_reset(&mut self) {
        self.basis.clear();
        self.directions.clear();
    }
    fn gmres_push_basis(&mut self, scale: T) {
        let mut v = self.error.clone();
        v.scale(scale);
        self.basis.push(v);
    }
    fn gmres_precondition(&mut self, settings: &CoreSettings<T>) -> bool {
        self.kernel.counters.outer_refinements += 1;
        let v = self.basis.last().unwrap();
        let mut z = vec![T::zero(); v.len()];
        let ok = self.kernel.solve_raw(&mut z, v, settings);
        self.directions.push(z);
        ok
    }
    fn gmres_operator(&mut self) -> bool {
        let z = self.directions.last().unwrap();
        let zero = vec![T::zero(); z.len()];
        let norm = self.kernel.residual(self.error, &zero, z, false);
        self.error.negate();
        norm.is_finite()
    }
    fn gmres_dots(&mut self) -> Vec<T> {
        self.basis.iter().map(|v| self.error.dot(v)).collect()
    }
    fn gmres_subtract(&mut self, c: &[T]) {
        for (v, &a) in self.basis.iter().zip(c) {
            self.error.axpby(-a, v, T::one());
        }
    }
    fn gmres_norm2(&mut self) -> T {
        self.error.norm()
    }
    fn gmres_candidate(&mut self, y: &[T]) {
        self.candidate.copy_from_slice(self.x);
        for (z, &yi) in self.directions.iter().zip(y) {
            self.candidate.axpby(yi, z, T::one());
        }
    }
}

#[cfg(test)]
mod gmres_tests {
    use super::*;
    use crate::solver::core::ScalingStrategy;

    /// GMRES-IR on the condensed original system reaches the stationary
    /// refinement's accuracy on a small PSD KKT.
    fn condensed_gmres<T: FloatT>() {
        let kinds = vec![SupportedConeT::PSDTriangleConeT(3); 2];
        let mut cones = CompositeCone::<T>::new(&kinds);
        let (mut slack, mut dual) = (vec![T::zero(); 12], vec![T::zero(); 12]);
        cones.unit_initialization(&mut dual, &mut slack);
        for (i, v) in slack.iter_mut().enumerate() {
            *v += T::from_f64(0.01 * (i % 5) as f64).unwrap();
        }
        assert!(cones.update_scaling(&slack, &dual, T::one(), ScalingStrategy::PrimalDual));
        let p = CscMatrix::identity(12);
        let a = CscMatrix::identity(12);
        let rhs: Vec<T> = (0..24)
            .map(|i| T::from_f64(((i * 7) % 11) as f64 - 5.0).unwrap())
            .collect();
        let mut results = Vec::new();
        for gmres in [false, true] {
            let mut settings = CoreSettings::<T>::default();
            settings.iterative_refinement_gmres = gmres;
            let mut kkt = CondensedKKTSolver::new(&p, &a, &kinds, &cones, &settings);
            assert!(kkt.update(&cones, &settings));
            let mut out = vec![T::zero(); 24];
            assert_eq!(kkt.solve_many(12, &rhs, &mut out, 1, &settings), [true]);
            let mut error = vec![T::zero(); 24];
            let norm = kkt.residual(&mut error, &rhs, &out, false);
            results.push((norm, out));
        }
        let tol = T::from_f64(1e-30).unwrap();
        assert!(results[1].0 <= T::max(results[0].0 * T::from_f64(10.0).unwrap(), tol));
        for (a, b) in results[0].1.iter().zip(&results[1].1) {
            assert!(T::abs(*a - *b) <= tol * (T::one() + T::abs(*a)));
        }
    }

    #[test]
    fn condensed_gmres_matches_stationary_f64_tolerance_mpfr256() {
        condensed_gmres::<sdpx_arithmetic::Bits256>();
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

    #[test]
    #[ignore]
    fn mpi_probe_two_rank_sampled_rhs() {
        use crate::solver::SampledBlock;
        use num_traits::{FromPrimitive, One, Zero};
        type T = sdpx_arithmetic::Bits256;
        let mpi = crate::MpiContext::initialize();
        assert_eq!(mpi.size(), 2);
        // Serial fallback, aligned pooled blocks, mixed ordinary rows with
        // linear entries (full exchange) and without (aligned). All have an
        // empty arrow border, which must remain valid under MPI.
        for (h, threads, orthant, coupled) in [
            (2, 1, false, false),
            (16, 2, false, false),
            (16, 2, true, true),
            (16, 2, true, false),
        ] {
            let offset = usize::from(orthant);
            let rows = triangular_number(h);
            let (m, n) = (offset + 2 * rows, 2 * h);
            let mut kinds = Vec::new();
            if orthant {
                kinds.push(SupportedConeT::NonnegativeConeT(1));
            }
            kinds.extend(vec![SupportedConeT::PSDTriangleConeT(h); 2]);
            let linear = if coupled {
                CscMatrix::new(m, n, (0..=n).collect(), vec![0; n], vec![T::one(); n])
            } else {
                CscMatrix::zeros((m, n))
            };
            let blocks = (0..2)
                .map(|b| SampledBlock {
                    row_start: offset + b * rows,
                    column_start: b * h,
                    dim: 1,
                    basis_rows: h,
                    basis_cols: h,
                    basis: Matrix::<T>::identity(h).data().to_vec(),
                    weights: vec![T::one(); h],
                })
                .collect();
            let operator = Arc::new(SampledOperator::new(linear, blocks).unwrap());
            let mut cones = CompositeCone::<T>::new(&kinds);
            cones.configure_threads(threads).unwrap();
            let (mut slack, mut dual) = (vec![T::zero(); m], vec![T::zero(); m]);
            cones.unit_initialization(&mut dual, &mut slack);
            assert!(cones.update_scaling(&slack, &dual, T::one(), ScalingStrategy::PrimalDual));
            let settings = CoreSettings {
                max_threads: threads as u32,
                ..CoreSettings::default()
            };
            let mut kkt = CondensedKKTSolver::new(
                &CscMatrix::identity(n),
                &operator.materialize(),
                &kinds,
                &cones,
                &settings,
            );
            kkt.set_sampled_operator(operator);
            assert!(kkt.update(&cones, &settings));
            assert!(kkt.linear_solver_info().name.ends_with("arrow"));
            let point = vec![T::one(); n + m];
            let zero = vec![T::zero(); n + m];
            let mut rhs = zero.clone();
            kkt.original_residual(&mut rhs, &zero, &point);
            rhs.negate();
            kkt.setrhs(&rhs[..n], &rhs[n..]);
            let mut actual = zero.clone();
            let (x, z) = actual.split_at_mut(n);
            assert!(kkt.solve(Some(x), Some(z), &settings));
            actual.axpby(-T::one(), &point, T::one());
            assert!(actual.norm_inf() < T::from_f64(1e-30).unwrap());
        }
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
                    if !p.R.data().is_empty() {
                        p.R.copy_from_slice(c.scaling_R().data());
                    }
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
                    // Wide arithmetic reads this Gram; binary64 applies H
                    // through R. Cone scaling already formed the product.
                    let gram = c.scaling_gram().data();
                    if !p.G.data().is_empty() {
                        for j in 0..c.n {
                            let start = j * c.n;
                            p.G.data_mut()[start..start + j + 1]
                                .copy_from_slice(&gram[start..start + j + 1]);
                        }
                        for j in 0..c.n {
                            for i in j + 1..c.n {
                                p.G[(i, j)] = p.G[(j, i)];
                            }
                        }
                    }
                    // A released Ginv (fused sampled block) is never read.
                    if !p.Ginv.data().is_empty() {
                        p.Ginv
                            .syrk(&p.Rinv.t(), T::one(), T::zero(), MatrixTriangle::Triu);
                        for j in 0..c.n {
                            for i in j + 1..c.n {
                                p.Ginv[(i, j)] = p.Ginv[(j, i)];
                            }
                        }
                    }
                    if !p.R.data().is_finite()
                        || !p.Rinv.data().is_finite()
                        || !(0..c.n).all(|j| {
                            let start = j * c.n;
                            gram[start..start + j + 1].is_finite()
                        })
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
                (Scaling::Soc { w, eta }, SupportedCone::SecondOrderCone(c))
                | (Scaling::SocElim { w, eta, .. }, SupportedCone::SecondOrderCone(c)) => {
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
        let sync_timer = crate::receipt::start();
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
                        .map(|s| s.work.gram_len().max(1) as u64)
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
                    Scaling::Psd(p) => p.sampled.as_ref().map_or(0, |s| s.work.gram_len()),
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
                        s.work.pack_gram(&mut local);
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
        crate::receipt::finish("sync", sync_timer);
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
        let timer = crate::receipt::start();
        if !self.assemble() {
            return false;
        }
        crate::receipt::finish("assemble", timer);
        self.reduced.publish_P();
        let retained = &self.retained_indices;
        let timer = crate::receipt::start();
        let retained_cones = cones
            .iter()
            .enumerate()
            .filter(|(i, _)| retained.binary_search(i).is_ok())
            .map(|(_, c)| c);
        let result = if factor {
            self.reduced.update_from_cones(retained_cones, settings)
        } else {
            self.reduced.assemble_from_cones(retained_cones);
            true
        };
        crate::receipt::finish("cones_schur", timer);
        result
    }
}
