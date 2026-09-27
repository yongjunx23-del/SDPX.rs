#![allow(non_snake_case)]

use super::*;
use crate::algebra::sparse_parallel::SparseParallel;
use crate::solver::kkt::{HasLinearSolverInfo, KKTSolver, LinearSolverInfo};
use crate::solver::{cones::*, core::CoreSettings};
use std::iter::zip;

// -------------------------------------
// KKTSolver using direct LDL factorisation
// -------------------------------------

// We require Send/Sync here to allow pyo3 builds to share
// solver objects between threads.

pub(crate) type BoxedDirectLDLSolver<T> = Box<dyn DirectLDLSolver<T> + Send + Sync>;

/// Identity backend for an empty local KKT system.
///
/// QDLDL (and the other numerical providers) expect at least one factor
/// entry.  Owner-local condensed systems can nevertheless be legitimately
/// zero-dimensional, so keep that case explicit instead of introducing a
/// dummy variable that would leak into the Schur complement.
struct EmptyDirectLDLSolver;

impl DirectLDLSolverReqs for EmptyDirectLDLSolver {
    fn required_matrix_shape() -> MatrixTriangle {
        MatrixTriangle::Triu
    }
}

impl HasLinearSolverInfo for EmptyDirectLDLSolver {
    fn linear_solver_info(&self) -> LinearSolverInfo {
        LinearSolverInfo {
            name: "empty".to_string(),
            threads: 1,
            direct: true,
            nnzA: 0,
            nnzL: 0,
        }
    }
}

impl<T: FloatT> DirectLDLSolver<T> for EmptyDirectLDLSolver {
    fn update_values(&mut self, index: &[usize], values: &[T]) {
        debug_assert!(index.is_empty() && values.is_empty());
    }

    fn scale_values(&mut self, index: &[usize], _scale: T) {
        debug_assert!(index.is_empty());
    }

    fn offset_values(&mut self, index: &[usize], _offset: T, signs: &[i8]) {
        debug_assert!(index.is_empty() && signs.is_empty());
    }

    fn solve(&mut self, kkt: &CscMatrix<T>, x: &mut [T], b: &mut [T]) {
        debug_assert_eq!(kkt.n, 0);
        debug_assert!(x.is_empty() && b.is_empty());
    }

    fn refactor(&mut self, kkt: &CscMatrix<T>) -> bool {
        kkt.n == 0 && kkt.nzval.is_empty()
    }
}

pub struct DirectLDLKKTSolver<T> {
    // problem dimensions
    m: usize,
    n: usize,
    p: usize,

    // Left and right hand sides for solves
    x: Vec<T>,
    b: Vec<T>,
    batch_rhs: Vec<T>,
    batch_out: Vec<T>,

    // internal workspace for IR scheme
    // and static offsetting of KKT
    work1: Vec<T>,
    work2: Vec<T>,

    // KKT mapping from problem data to KKT
    map: LDLDataMap,

    // the expected signs of D in KKT = LDL^T
    dsigns: Vec<i8>,

    // a vector for storing the entries of Hs blocks
    // on the KKT matrix block diagonal
    Hsblocks: Vec<T>,

    // unpermuted KKT matrix
    KKT: CscMatrix<T>,

    // triangular storage shape for KKT
    KKTuplo: MatrixTriangle,

    // the direct linear LDL solver
    ldlsolver: BoxedDirectLDLSolver<T>,

    // the diagonal regularizer currently applied
    diagonal_regularizer: T,
    // static regularization escalation level, raised on refactor/solve
    // failure and retained for later factorizations
    reg_boost: usize,
    counters: crate::solver::kkt::SolveCounters,
    residual_plan: Option<SparseParallel>,
    #[cfg(feature = "sdp")]
    residual_dense: Option<Matrix<T>>,
    /// Full symmetric row lists of the KKT for exact residuals (MPFR only).
    exact_rows: Option<ExactRows>,
    residual_pool: Option<std::sync::Arc<rayon::ThreadPool>>,
}

impl<T> DirectLDLKKTSolver<T>
where
    T: FloatT,
{
    pub fn new(
        P: &CscMatrix<T>,
        A: &CscMatrix<T>,
        cones: &CompositeCone<T>,
        m: usize,
        n: usize,
        settings: &CoreSettings<T>,
    ) -> Self {
        // get a constructor for the LDL solver we should use,
        // and also the matrix shape it requires
        let (kktshape, ldl_ctor) = T::get_ldlsolver_config(settings);

        //construct a KKT matrix of the right shape
        let (KKT, map) = assemble_kkt_matrix(P, A, cones, kktshape);

        //Need this many extra variables for sparse cones
        let p = map.sparse_maps.pdim();

        // LHS/RHS/work for iterative refinement
        let x = vec![T::zero(); n + m + p];
        let b = vec![T::zero(); n + m + p];
        let work1 = vec![T::zero(); n + m + p];
        let work2 = vec![T::zero(); n + m + p];

        // the expected signs of D in LDL
        let mut dsigns = vec![1_i8; n + m + p];
        _fill_signs(&mut dsigns, m, n, &map);

        // updates to the diagonal of KKT will be
        // assigned here before updating matrix entries
        let Hsblocks = allocate_kkt_Hsblocks::<T, T>(cones);

        let diagonal_regularizer = T::zero();

        // now make the LDL linear solver engine
        // final argument is None, since it is only
        // used by the Auto solver type to pass on
        // AMD ordering vectors to its selected solver
        // If using a solver directly and no ordering is
        // provided, the solver finds one for itself
        let ldlsolver = if KKT.n == 0 {
            Box::new(EmptyDirectLDLSolver) as BoxedDirectLDLSolver<T>
        } else {
            ldl_ctor(&KKT, &dsigns, settings, None)
        };

        Self {
            m,
            n,
            p,
            x,
            b,
            batch_rhs: Vec::new(),
            batch_out: Vec::new(),
            work1,
            work2,
            map,
            dsigns,
            Hsblocks,
            KKT,
            KKTuplo: kktshape,
            ldlsolver,
            diagonal_regularizer,
            reg_boost: 0,
            counters: Default::default(),
            residual_plan: None,
            #[cfg(feature = "sdp")]
            residual_dense: None,
            exact_rows: None,
            residual_pool: None,
        }
    }
}

impl<T> HasLinearSolverInfo for DirectLDLKKTSolver<T>
where
    T: FloatT,
{
    fn linear_solver_info(&self) -> LinearSolverInfo {
        self.ldlsolver.linear_solver_info()
    }
}

impl<T> KKTSolver<T> for DirectLDLKKTSolver<T>
where
    T: FloatT,
{
    fn reset_solve(&mut self) {
        self.reg_boost = 0;
        self.counters = Default::default();
    }

    fn counters(&self) -> crate::solver::kkt::SolveCounters {
        self.counters
    }

    fn update(&mut self, cones: &CompositeCone<T>, settings: &CoreSettings<T>) -> bool {
        let pool = cones.thread_pool();
        self.set_residual_pool(pool.clone());
        self.set_factor_pool(pool);
        self.update_from_cones(cones.iter(), settings)
    }

    fn setrhs(&mut self, rhsx: &[T], rhsz: &[T]) {
        let (m, n, p) = (self.m, self.n, self.p);

        self.b[0..n].copy_from(rhsx);
        self.b[n..(n + m)].copy_from(rhsz);
        self.b[n + m..(n + m + p)].fill(T::zero());
    }

    fn solve(
        &mut self,
        lhsx: Option<&mut [T]>,
        lhsz: Option<&mut [T]>,
        settings: &CoreSettings<T>,
    ) -> bool {
        let __t0 = std::time::Instant::now();
        // Lossless snapshot capture (PR-02): the RHS is cloned before the
        // solve because a backend may overwrite it.
        #[cfg(feature = "snapshot")]
        let snap = if crate::snapshot::enabled() {
            use std::sync::atomic::{AtomicU64, Ordering};
            static SNAP_IDX: AtomicU64 = AtomicU64::new(0);
            Some((SNAP_IDX.fetch_add(1, Ordering::Relaxed), self.b.clone()))
        } else {
            None
        };
        self.counters.rhs_applied += 1;
        self.counters.linear_solves += 1;
        let __c0 = crate::receipt::cpu_start();
        self.ldlsolver.solve(&self.KKT, &mut self.x, &mut self.b);
        let __t_trsv = __t0.elapsed();
        crate::receipt::cpu_add("trsv", __c0);
        #[cfg(feature = "snapshot")]
        if let Some((idx, rhs)) = snap {
            let ok = self.x.is_finite();
            if crate::snapshot::wanted(idx, !ok) {
                if let Some(dir) = std::env::var_os("SDPX_SNAPSHOT") {
                    let backend = self.ldlsolver.linear_solver_info().name.clone();
                    let _ = crate::snapshot::write(
                        std::path::Path::new(&dir),
                        idx,
                        &self.KKT,
                        &self.dsigns,
                        &rhs,
                        &self.x,
                        ok,
                        &backend,
                        self.reg_boost,
                    );
                }
            }
        }
        let is_success = {
            if settings.iterative_refinement_enable {
                let __c_ir = crate::receipt::cpu_start();
                let r = self.iterative_refinement(settings);
                crate::receipt::cpu_add("ir", __c_ir);
                if crate::receipt::profile_requested() {
                    eprintln!("PHASE trsv {:?} ir {:?}", __t_trsv, __t0.elapsed());
                }
                crate::receipt::phase_record("trsv", __t_trsv);
                // The printed "ir" field is cumulative since __t0; record the
                // refinement-only share for per-phase accounting.
                crate::receipt::phase_record("ir", __t0.elapsed() - __t_trsv);
                r
            } else {
                crate::receipt::phase("trsv", __t_trsv);
                self.x.is_finite()
            }
        };

        if is_success {
            self.getlhs(lhsx, lhsz);
        }

        is_success
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
        let width = self.n + self.m;
        assert_eq!(rhs.len(), cols * width);
        assert_eq!(out.len(), rhs.len());
        if cols == 0 {
            return Vec::new();
        }
        #[cfg(feature = "snapshot")]
        if crate::snapshot::enabled() {
            return crate::solver::kkt::solve_many_by_columns(self, n, rhs, out, cols, settings);
        }
        self.counters.batches += 1;
        let full = width + self.p;
        let mut b = std::mem::take(&mut self.batch_rhs);
        let mut x = std::mem::take(&mut self.batch_out);
        b.resize(cols * full, T::zero());
        x.resize(cols * full, T::zero());
        for c in 0..cols {
            b[c * full..c * full + width].copy_from_slice(&rhs[c * width..(c + 1) * width]);
            b[c * full + width..(c + 1) * full].fill(T::zero());
        }
        let timer = crate::receipt::start();
        self.ldlsolver.solve_many(&self.KKT, &mut x, &mut b, cols);
        crate::receipt::finish("trsv", timer);
        self.counters.rhs_applied += cols as u64;
        self.counters.linear_solves += cols as u64;
        let mut flags = Vec::with_capacity(cols);
        for c in 0..cols {
            self.setrhs(
                &rhs[c * width..c * width + n],
                &rhs[c * width + n..(c + 1) * width],
            );
            self.x.copy_from_slice(&x[c * full..(c + 1) * full]);
            let timer = crate::receipt::start();
            let ok = if settings.iterative_refinement_enable {
                self.iterative_refinement(settings)
            } else {
                self.x.is_finite()
            };
            crate::receipt::finish("ir", timer);
            if ok {
                out[c * width..(c + 1) * width].copy_from_slice(&self.x[..width]);
            }
            flags.push(ok);
        }
        self.batch_rhs = b;
        self.batch_out = x;
        flags
    }

    fn escalate_regularization(&mut self) -> bool {
        // Each level multiplies the static shift by 100, so the budget of
        // three reaches ~1e-2·max(1,‖diag‖) for binary64 — beyond that the
        // factorized system is too perturbed to rescue the iteration.
        if self.reg_boost >= 3 {
            return false;
        }
        self.reg_boost += 1;
        true
    }

    fn update_P(&mut self, P: &CscMatrix<T>) {
        _update_values(&mut self.ldlsolver, &mut self.KKT, &self.map.P, &P.nzval);
    }

    fn update_A(&mut self, A: &CscMatrix<T>) {
        _update_values(&mut self.ldlsolver, &mut self.KKT, &self.map.A, &A.nzval);
    }
}

impl<T: FloatT> DirectLDLKKTSolver<T> {
    // A condensed system retains some original cones. Read their current
    // scaling directly; a second set of cone states would become stale.
    pub(crate) fn update_from_cones<'a, I>(&mut self, cones: I, settings: &CoreSettings<T>) -> bool
    where
        I: Iterator<Item = &'a SupportedCone<T>> + Clone,
    {
        self.assemble_from_cones(cones);
        self.regularize_and_refactor(settings)
    }

    /// Update the cone-dependent entries of the unshifted KKT matrix.
    ///
    /// This phase deliberately does not factor or apply static
    /// regularization.  Condensed owner-local callers can therefore assemble
    /// every local matrix first, agree on one shift, and factor afterwards.
    pub(crate) fn assemble_from_cones<'a, I>(&mut self, cones: I)
    where
        I: Iterator<Item = &'a SupportedCone<T>> + Clone,
    {
        let map = &self.map;

        // Set the elements the W^tW blocks in the KKT matrix.
        let mut offset = 0;
        for cone in cones.clone() {
            let len = if cone.Hs_is_diagonal() {
                cone.numel()
            } else {
                triangular_number(cone.numel())
            };
            cone.get_Hs(&mut self.Hsblocks[offset..offset + len]);
            offset += len;
        }
        assert_eq!(offset, self.Hsblocks.len());

        let (values, index) = (&mut self.Hsblocks, &map.Hsblocks);
        // change signs to get -W^TW
        values.negate();
        _update_values(&mut self.ldlsolver, &mut self.KKT, index, values);

        let mut sparse_map_iter = map.sparse_maps.iter();
        let ldl = &mut self.ldlsolver;
        let KKT = &mut self.KKT;

        for cone in cones {
            if cone.is_sparse_expandable() {
                let sc = cone.to_sparse_expansion().unwrap();
                let thismap = sparse_map_iter.next().unwrap();
                sc.csc_update_sparsecone(thismap, ldl, KKT, _update_values, _scale_values);
            }
        }
    }
}

impl<T> DirectLDLKKTSolver<T>
where
    T: FloatT,
{
    /// Infinity norm of the diagonal of the current, unshifted KKT.
    pub(crate) fn diagonal_norm(&self) -> T {
        self.map
            .diag_full
            .iter()
            .map(|&idx| self.KKT.nzval[idx].abs())
            .fold(T::zero(), T::max)
    }

    /// Dimension of the complete local KKT, including sparse-cone auxiliaries.
    pub(crate) fn factor_dimension(&self) -> usize {
        self.KKT.n
    }

    /// Factor the current KKT, optionally using a caller-agreed static shift.
    ///
    /// `Some(eps)` replaces the internally computed regularizer and its
    /// escalation boost.  A shift is ignored when static regularization is
    /// disabled.  As in the ordinary update path, the public KKT is restored
    /// to its unshifted values before this returns so residuals always use the
    /// original operator.
    pub(crate) fn factor_with_shift(
        &mut self,
        settings: &CoreSettings<T>,
        shift: Option<T>,
    ) -> bool {
        let map = &self.map;
        let KKT = &mut self.KKT;
        let dsigns = &self.dsigns;
        let diag_kkt = &mut self.work1;
        let diag_shifted = &mut self.work2;

        let applied_shift = if settings.static_regularization_enable {
            let eps = match shift {
                Some(eps) => eps,
                None => {
                    for (d, idx) in zip(&mut *diag_kkt, &map.diag_full) {
                        *d = KKT.nzval[*idx];
                    }

                    let mut eps = _compute_regularizer(diag_kkt, settings);
                    if self.reg_boost > 0 {
                        eps = eps * 100f64.powi(self.reg_boost as i32).as_T();
                    }
                    eps
                }
            };

            // Keep the true diagonal available both for residuals and for
            // restoring KKT after the factorization attempt.  For an
            // explicitly supplied shift the same copy is still required;
            // `diag_kkt` may contain data from an earlier phase.
            if shift.is_some() {
                for (d, idx) in zip(&mut *diag_kkt, &map.diag_full) {
                    *d = KKT.nzval[*idx];
                }
            }
            diag_shifted.copy_from(diag_kkt);

            zip(&mut *diag_shifted, dsigns).for_each(|(value, &sign)| {
                if sign == 1 {
                    *value += eps;
                } else {
                    *value -= eps;
                }
            });

            _update_values(&mut self.ldlsolver, KKT, &map.diag_full, diag_shifted);
            self.diagonal_regularizer = eps;
            true
        } else {
            false
        };

        // Refactor with the (possibly) shifted values.
        let __t0 = std::time::Instant::now();
        let __c0 = crate::receipt::cpu_start();
        self.counters.factor_attempts += 1;
        let is_success = self.ldlsolver.refactor(KKT);
        self.counters.factorizations += u64::from(is_success);
        crate::receipt::phase("refactor", __t0.elapsed());
        crate::receipt::cpu_add("refactor", __c0);

        if applied_shift {
            // The factor backend intentionally retains the shifted values,
            // while the authoritative sparse operator must remain unshifted.
            _update_values_KKT(KKT, &map.diag_full, diag_kkt);
        }

        // Dense binary64 systems spend more time traversing CSC during IR
        // than multiplying. Cache the authoritative, unshifted triangle once
        // per factor update; never use the regularized factor as an operator.
        #[cfg(feature = "sdp")]
        {
            let dim = KKT.n;
            let dense_entries = (dim as u128) * (dim as u128);
            if T::precision_bits() <= 53
                && dim >= 512
                && dense_entries * std::mem::size_of::<T>() as u128 <= (512u128 << 20)
                && (KKT.nzval.len() as u128) * 5 >= dense_entries
            {
                let dense = self
                    .residual_dense
                    .get_or_insert_with(|| Matrix::zeros((dim, dim)));
                for col in 0..dim {
                    for pos in KKT.colptr[col]..KKT.colptr[col + 1] {
                        dense[(KKT.rowval[pos], col)] = KKT.nzval[pos];
                    }
                }
            }
        }

        if let Some(dir) = kkt_dump_dir() {
            dump_kkt_f64(dir, KKT);
        }

        is_success
    }

    /// Solve a panel against the complete factorized local KKT.
    ///
    /// This is an internal application of the factors: it increments backend
    /// solve/batch counters but deliberately does not count caller RHS
    /// applications and performs no local iterative refinement.
    pub(crate) fn solve_factor_panel(&mut self, rhs: &[T], out: &mut [T], cols: usize) -> bool {
        let full = self.factor_dimension();
        assert_eq!(rhs.len(), cols * full);
        assert_eq!(out.len(), rhs.len());
        if cols == 0 {
            return true;
        }

        let mut batch_rhs = std::mem::take(&mut self.batch_rhs);
        let mut batch_out = std::mem::take(&mut self.batch_out);
        batch_rhs.resize(rhs.len(), T::zero());
        batch_out.resize(rhs.len(), T::zero());
        batch_rhs.copy_from_slice(rhs);

        let timer = crate::receipt::start();
        self.ldlsolver
            .solve_many(&self.KKT, &mut batch_out, &mut batch_rhs, cols);
        crate::receipt::finish("trsv", timer);
        self.counters.batches += 1;
        self.counters.linear_solves += cols as u64;

        let success = batch_out.iter().all(|v| v.is_finite());
        if success {
            out.copy_from_slice(&batch_out);
        }
        self.batch_rhs = batch_rhs;
        self.batch_out = batch_out;
        success
    }

    /// Residual of the complete, unshifted KKT for one point.
    pub(crate) fn residual_full(&self, out: &mut [T], rhs: &[T], point: &[T]) -> T {
        assert_eq!(rhs.len(), self.factor_dimension());
        assert_eq!(point.len(), rhs.len());
        assert_eq!(out.len(), rhs.len());
        let KKTsym = self.KKT.sym(self.KKTuplo);
        let plan = self.residual_plan.as_ref();
        if let Some(rows) = &self.exact_rows {
            return rows.residual(out, rhs, &self.KKT, point, self.residual_pool.as_deref());
        }
        #[cfg(feature = "sdp")]
        {
            let dense = self.residual_dense.as_ref();
            _get_refine_error_dense(out, rhs, &KKTsym, point, plan, dense)
        }
        #[cfg(not(feature = "sdp"))]
        {
            _get_refine_error(out, rhs, &KKTsym, point, plan)
        }
    }

    // extra helper functions, not required for KKTSolver trait
    fn getlhs(&self, lhsx: Option<&mut [T]>, lhsz: Option<&mut [T]>) {
        let x = &self.x;
        let (m, n) = (self.m, self.n);

        if let Some(v) = lhsx {
            v.copy_from(&x[0..n]);
        }
        if let Some(v) = lhsz {
            v.copy_from(&x[n..(n + m)]);
        }
    }

    fn regularize_and_refactor(&mut self, settings: &CoreSettings<T>) -> bool {
        self.factor_with_shift(settings, None)
    }

    /// Forward the solver thread pool to the LDL factorisation kernels that
    /// can use it (currently the elimination-tree parallel QDLDL path).
    pub(crate) fn set_factor_pool(&mut self, pool: Option<std::sync::Arc<rayon::ThreadPool>>) {
        self.ldlsolver.set_pool(pool);
    }

    pub(crate) fn set_residual_pool(&mut self, pool: Option<std::sync::Arc<rayon::ThreadPool>>) {
        let words = T::precision_bits().div_ceil(64) as u128;
        let work = self.KKT.nzval.len() as u128 * words * words;
        let pool = pool.filter(|p| {
            p.current_num_threads() > 1 && work >= 4096 * p.current_num_threads() as u128
        });
        self.residual_pool = pool.clone();
        if pool.is_some() && self.residual_plan.is_none() {
            self.residual_plan = Some(SparseParallel::new_symmetric(&self.KKT));
        }
        if let Some(plan) = &mut self.residual_plan {
            plan.configure(&self.KKT, pool);
        }
    }

    fn iterative_refinement(&mut self, settings: &CoreSettings<T>) -> bool {
        let (x, b) = (&mut self.x, &self.b);
        let (e, dx) = (&mut self.work1, &mut self.work2);

        // iterative refinement params
        let reltol = settings.iterative_refinement_reltol;
        let abstol = settings.iterative_refinement_abstol;
        let maxiter = settings.iterative_refinement_max_iter;
        let stopratio = settings.iterative_refinement_stop_ratio;

        let KKT = &self.KKT;
        let KKTsym = KKT.sym(self.KKTuplo);

        let plan = self.residual_plan.as_ref();
        let normb = b.norm_inf();

        if T::precision_bits() > 64
            && self
                .exact_rows
                .as_ref()
                .map_or(true, |r| r.entries.len() != r.expected(&self.KKT))
        {
            self.exact_rows = Some(ExactRows::new(KKT));
        }
        let exact = self.exact_rows.as_ref();
        let pool = self.residual_pool.as_deref();
        #[cfg(feature = "sdp")]
        let dense = self.residual_dense.as_ref();
        let error = |e: &mut [T], x: &mut [T]| -> T {
            if let Some(rows) = exact {
                return rows.residual(e, b, KKT, x, pool);
            }
            #[cfg(feature = "sdp")]
            {
                _get_refine_error_dense(e, b, &KKTsym, x, plan, dense)
            }
            #[cfg(not(feature = "sdp"))]
            {
                _get_refine_error(e, b, &KKTsym, x, plan)
            }
        };
        //compute the initial error
        let mut norme = error(e, x);

        if !norme.is_finite() {
            return false;
        }

        for pass in 0..maxiter {
            if norme <= (abstol + reltol * normb) {
                //within tolerance.  Exit
                if trace_ir() {
                    eprintln!("IRI converged pass={pass}");
                }
                break;
            }

            let lastnorme = norme;

            //make a refinement
            self.counters.linear_solves += 1;
            self.counters.refinements += 1;
            let timer = crate::receipt::start();
            self.ldlsolver.solve(KKT, dx, e);
            crate::receipt::finish("ir.solve", timer);

            //prospective solution is x + dx.  Use dx space to
            // hold it for a check before applying to x
            dx.axpby(T::one(), x, T::one());

            let timer = crate::receipt::start();
            norme = error(e, dx);
            crate::receipt::finish("ir.residual", timer);

            if !norme.is_finite() {
                return false;
            }

            let improved_ratio = lastnorme / norme;
            if trace_ir() {
                eprintln!(
                    "IRI pass={} before={:.3e} after={:.3e} ratio={:.3e} normb={:.3e}",
                    self.counters.refinements,
                    lastnorme.to_f64().unwrap_or(f64::NAN),
                    norme.to_f64().unwrap_or(f64::NAN),
                    improved_ratio.to_f64().unwrap_or(f64::NAN),
                    normb.to_f64().unwrap_or(f64::NAN),
                );
            }
            if improved_ratio < stopratio {
                //insufficient improvement.  Exit
                if improved_ratio > T::one() {
                    std::mem::swap(x, dx);
                }
                break;
            }
            std::mem::swap(x, dx);
        }
        //NB: "success" means only that we had a finite valued result
        true
    }
}

/// Diagnostic trace of inner refinement passes (`SDPX_TRACE_IR`).
fn trace_ir() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("SDPX_TRACE_IR").is_some())
}

fn _compute_regularizer<T: FloatT>(diag_kkt: &[T], settings: &CoreSettings<T>) -> T {
    let maxdiag = diag_kkt.norm_inf();

    // Compute a new regularizer
    settings.static_regularization_constant + settings.static_regularization_proportional * maxdiag
}

//  computes e = b - Kξ, overwriting the first argument
//  and returning its norm

/// Row lists of a symmetric KKT stored as one CSC triangle, for residuals
/// `b - K x` accumulated exactly and rounded once per row (Wilkinson
/// refinement with an extra-precise residual). A per-product rounding leaves
/// an error of eps·Σ|K_ij x_j|, which for badly scaled bootstrap systems
/// (|x| up to 1e80+) exceeds the residual itself and stalls refinement; the
/// exact residual keeps the error at eps·|e_i|.
struct ExactRows {
    ptr: Vec<usize>,
    /// (nz position, column) for every entry of each full symmetric row.
    entries: Vec<(usize, usize)>,
}

impl ExactRows {
    fn new<T>(k: &CscMatrix<T>) -> Self {
        let n = k.n;
        let mut count = vec![0usize; n + 1];
        for j in 0..n {
            for p in k.colptr[j]..k.colptr[j + 1] {
                let r = k.rowval[p];
                count[r + 1] += 1;
                if r != j {
                    count[j + 1] += 1;
                }
            }
        }
        for i in 0..n {
            count[i + 1] += count[i];
        }
        let ptr = count.clone();
        let mut fill = count;
        let mut entries = vec![(0usize, 0usize); ptr[n]];
        for j in 0..n {
            for p in k.colptr[j]..k.colptr[j + 1] {
                let r = k.rowval[p];
                entries[fill[r]] = (p, j);
                fill[r] += 1;
                if r != j {
                    entries[fill[j]] = (p, r);
                    fill[j] += 1;
                }
            }
        }
        Self { ptr, entries }
    }

    /// Entry count the current pattern implies (rebuild on mismatch).
    fn expected<T>(&self, k: &CscMatrix<T>) -> usize {
        let diag = (0..k.n)
            .filter(|&j| k.rowval[k.colptr[j]..k.colptr[j + 1]].contains(&j))
            .count();
        2 * k.nzval.len() - diag
    }

    fn residual<T: FloatT>(
        &self,
        e: &mut [T],
        b: &[T],
        k: &CscMatrix<T>,
        x: &[T],
        pool: Option<&rayon::ThreadPool>,
    ) -> T {
        use rayon::prelude::*;
        let neg: Vec<T> = x.iter().map(|&v| -v).collect();
        let one = T::one();
        let row = |i: usize| {
            let terms = &self.entries[self.ptr[i]..self.ptr[i + 1]];
            T::dot_fma(
                std::iter::once((&b[i], &one))
                    .chain(terms.iter().map(|&(p, j)| (&k.nzval[p], &neg[j]))),
            )
        };
        match pool {
            Some(pool) => {
                // Border rows hold ~10x the entries of leaf rows; split by
                // entry count (about four tasks per worker) so no task
                // collects a run of long rows. Every row is unchanged.
                let n = e.len();
                let parts = (4 * pool.current_num_threads()).clamp(1, n.max(1));
                let total = self.ptr[n];
                let mut chunks: Vec<(usize, &mut [T])> = Vec::with_capacity(parts);
                let (mut start, mut rest) = (0usize, &mut *e);
                for p in 1..=parts {
                    let end = if p == parts {
                        n
                    } else {
                        self.ptr.partition_point(|&v| v < total * p / parts).min(n)
                    };
                    if end > start {
                        let (chunk, tail) = std::mem::take(&mut rest).split_at_mut(end - start);
                        chunks.push((start, chunk));
                        rest = tail;
                        start = end;
                    }
                }
                pool.install(|| {
                    chunks.into_par_iter().for_each(|(first, chunk)| {
                        for (i, v) in chunk.iter_mut().enumerate() {
                            *v = row(first + i);
                        }
                    })
                })
            }
            None => e.iter_mut().enumerate().for_each(|(i, v)| *v = row(i)),
        }
        if e.is_finite() {
            e.norm_inf()
        } else {
            T::infinity()
        }
    }
}

fn _get_refine_error<T: FloatT>(
    e: &mut [T],
    b: &[T],
    KKTsym: &Symmetric<CscMatrix<T>>,
    ξ: &[T],
    plan: Option<&SparseParallel>,
) -> T {
    // Note that K is only triu data, so need to
    // be careful when computing the residual here

    e.copy_from(b);
    if let Some(plan) = plan {
        plan.symv(KKTsym.src, KKTsym.uplo, e, ξ, -T::one(), T::one());
    } else {
        KKTsym.symv(e, ξ, -T::one(), T::one());
    }

    e.norm_inf()
}

#[cfg(feature = "sdp")]
fn _get_refine_error_dense<T: FloatT>(
    e: &mut [T],
    b: &[T],
    KKTsym: &Symmetric<CscMatrix<T>>,
    ξ: &[T],
    plan: Option<&SparseParallel>,
    dense: Option<&Matrix<T>>,
) -> T {
    if let Some(dense) = dense {
        e.copy_from(b);
        dense.sym(KKTsym.uplo).symv(ξ, e, -T::one(), T::one());
        return e.norm_inf();
    }
    _get_refine_error(e, b, KKTsym, ξ, plan)
}

// update entries of the KKT matrix using the given index into its CSC representation.
// applied to both the unpermuted matrix of the kktsolver and also to the ldlsolver
fn _update_values<T: FloatT>(
    ldlsolver: &mut BoxedDirectLDLSolver<T>,
    KKT: &mut CscMatrix<T>,
    index: &[usize],
    values: &[T],
) {
    //Update values in the KKT matrix K
    _update_values_KKT(KKT, index, values);

    // give the LDL subsolver an opportunity to update the same
    // values if needed.   This latter is useful for QDLDL since
    // it stores its own permuted copy internally
    ldlsolver.update_values(index, values);
}

fn _update_values_KKT<T: FloatT>(KKT: &mut CscMatrix<T>, index: &[usize], values: &[T]) {
    for (idx, v) in zip(index, values) {
        KKT.nzval[*idx] = *v;
    }
}

fn _scale_values<T: FloatT>(
    ldlsolver: &mut BoxedDirectLDLSolver<T>,
    KKT: &mut CscMatrix<T>,
    index: &[usize],
    scale: T,
) {
    //Update values in the KKT matrix K
    _scale_values_KKT(KKT, index, scale);

    // ...and in the LDL subsolver if needed
    ldlsolver.scale_values(index, scale);
}

//scales KKT matrix values
fn _scale_values_KKT<T: FloatT>(KKT: &mut CscMatrix<T>, index: &[usize], scale: T) {
    for idx in index.iter() {
        KKT.nzval[*idx] *= scale;
    }
}

fn _fill_signs(signs: &mut [i8], m: usize, n: usize, map: &LDLDataMap) {
    signs.fill(1);

    //flip expected negative signs of D in LDL
    signs[n..(n + m)].iter_mut().for_each(|x| *x = -*x);

    let mut p = m + n;
    // assign D signs for sparse expansion cones
    for thismap in map.sparse_maps.iter() {
        let thisp = thismap.pdim();
        signs[p..(p + thisp)].copy_from_slice(thismap.Dsigns());
        p += thisp;
    }
}

/// Debug probe: dump the true (unregularized) KKT matrix as f64 CSC text.
/// Gated by SDPX_DUMP_KKT=<dir>; writes <dir>/kkt-NNNN.txt with the matrix
/// dimension, colptr, rowval and nzval converted elementwise to f64. Values
/// overflowing f64 clamp to ±f64::MAX, which already answers the conditioning
/// question the probe exists for.
/// `SDPX_DUMP_KKT` capture gate, read once per process. The gate is fixed at
/// launch; caching keeps getenv out of the per-factorization path.
fn kkt_dump_dir() -> Option<&'static std::ffi::OsStr> {
    static DIR: std::sync::OnceLock<Option<std::ffi::OsString>> = std::sync::OnceLock::new();
    DIR.get_or_init(|| std::env::var_os("SDPX_DUMP_KKT"))
        .as_deref()
}

fn dump_kkt_f64<T: FloatT>(dir: &std::ffi::OsStr, KKT: &CscMatrix<T>) {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static DUMP_IDX: AtomicUsize = AtomicUsize::new(0);
    let idx = DUMP_IDX.fetch_add(1, Ordering::Relaxed);
    let path = std::path::Path::new(dir).join(format!("kkt-{idx:04}.txt"));
    let mut out = String::with_capacity(KKT.nzval.len() * 24);
    out.push_str(&format!("{} {}\n", KKT.n, KKT.nzval.len()));
    for &p in KKT.colptr.iter() {
        out.push_str(&format!("{p}\n"));
    }
    for &r in KKT.rowval.iter() {
        out.push_str(&format!("{r}\n"));
    }
    for v in KKT.nzval.iter() {
        let f = v
            .to_f64()
            .unwrap_or_else(|| if *v < T::zero() { -f64::MAX } else { f64::MAX });
        out.push_str(&format!("{f:.17e}\n"));
    }
    let _ = std::fs::write(&path, out);
}

#[cfg(test)]
mod parallel_residual_tests {
    use super::*;
    use crate::solver::core::ScalingStrategy;

    #[test]
    fn phase_api_empty_owner_zero_dim() {
        let p = CscMatrix::<f64>::zeros((0, 0));
        let a = CscMatrix::<f64>::zeros((0, 0));
        let cones = CompositeCone::new(&[]);
        let settings = CoreSettings::<f64>::default();
        let mut solver = DirectLDLKKTSolver::new(&p, &a, &cones, 0, 0, &settings);

        assert_eq!(solver.factor_dimension(), 0);
        assert_eq!(solver.diagonal_norm(), 0.0);
        solver.assemble_from_cones(cones.iter());
        assert!(solver.factor_with_shift(&settings, Some(1e-3)));

        let rhs: [f64; 0] = [];
        let mut out: [f64; 0] = [];
        assert!(solver.solve_factor_panel(&rhs, &mut out, 2));
        let mut residual: [f64; 0] = [];
        assert_eq!(solver.residual_full(&mut residual, &rhs, &out), 0.0);

        let counters = solver.counters();
        assert_eq!(counters.factor_attempts, 1);
        assert_eq!(counters.factorizations, 1);
        assert_eq!(counters.linear_solves, 2);
        assert_eq!(counters.rhs_applied, 0);
    }

    #[test]
    fn phase_api_matches_default_panel_and_residual() {
        let p = CscMatrix::<f64>::identity(2);
        let a = CscMatrix::zeros((0, 2));
        let cones = CompositeCone::new(&[]);
        let mut settings = CoreSettings::<f64>::default();
        settings.direct_solve_method = "qdldl".to_string();
        settings.iterative_refinement_enable = false;
        settings.static_regularization_enable = false;

        let mut ordinary = DirectLDLKKTSolver::new(&p, &a, &cones, 0, 2, &settings);
        assert!(ordinary.update(&cones, &settings));
        let rhs = [1.0, -2.0];
        ordinary.setrhs(&rhs, &[]);
        let mut expected = [0.0; 2];
        assert!(ordinary.solve(Some(&mut expected), None, &settings));

        let mut phase = DirectLDLKKTSolver::new(&p, &a, &cones, 0, 2, &settings);
        phase.assemble_from_cones(cones.iter());
        assert!(phase.factor_with_shift(&settings, None));
        let panel_rhs = [rhs[0], rhs[1], 3.0, 4.0];
        let saved_rhs = panel_rhs;
        let mut actual = [0.0; 4];
        assert!(phase.solve_factor_panel(&panel_rhs, &mut actual, 2));
        assert_eq!(panel_rhs, saved_rhs);
        assert_eq!(&actual[..2], &expected);
        assert_eq!(&actual[2..], &[3.0, 4.0]);

        let mut residual = [0.0; 2];
        assert!(phase.residual_full(&mut residual, &panel_rhs[..2], &actual[..2]) < 1e-12);
        assert!(phase.residual_full(&mut residual, &panel_rhs[2..], &actual[2..]) < 1e-12);
    }

    #[test]
    fn phase_api_sparse_shift_restores_signed_diagonal() {
        let p = CscMatrix::<f64>::zeros((0, 0));
        let a = CscMatrix::zeros((5, 0));
        let kinds = [SupportedConeT::SecondOrderConeT(5)];
        let mut cones = CompositeCone::new(&kinds);
        let mut slack = vec![0.0; 5];
        let mut dual = vec![0.0; 5];
        cones.unit_initialization(&mut dual, &mut slack);
        assert!(cones.update_scaling(&slack, &dual, 1.0, ScalingStrategy::PrimalDual));
        let mut settings = CoreSettings::<f64>::default();
        settings.direct_solve_method = "qdldl".to_string();
        settings.iterative_refinement_enable = false;
        let mut solver = DirectLDLKKTSolver::new(&p, &a, &cones, 5, 0, &settings);
        solver.assemble_from_cones(cones.iter());
        let before: Vec<_> = solver
            .map
            .diag_full
            .iter()
            .map(|&idx| solver.KKT.nzval[idx])
            .collect();
        assert!(solver.dsigns[5..].contains(&-1));
        assert!(solver.dsigns[5..].contains(&1));
        let eps = 0.25;
        assert!(solver.factor_with_shift(&settings, Some(eps)));
        assert_eq!(solver.diagonal_regularizer, eps);
        let after: Vec<_> = solver
            .map
            .diag_full
            .iter()
            .map(|&idx| solver.KKT.nzval[idx])
            .collect();
        assert_eq!(after, before);

        // Reassembly must replace cone updates rather than compound them.
        solver.assemble_from_cones(cones.iter());
        assert!(solver.factor_with_shift(&settings, Some(eps)));
        let repeated: Vec<_> = solver
            .map
            .diag_full
            .iter()
            .map(|&idx| solver.KKT.nzval[idx])
            .collect();
        assert_eq!(repeated, before);

        // An explicit shift replaces, rather than multiplies, any retry
        // boost; the true KKT remains the residual operator after refactor.
        solver.reg_boost = 2;
        assert!(solver.factor_with_shift(&settings, Some(eps)));
        assert_eq!(solver.diagonal_regularizer, eps);
        let rhs = vec![1.0; solver.factor_dimension()];
        let mut out = vec![0.0; rhs.len()];
        assert!(solver.solve_factor_panel(&rhs, &mut out, 1));
        let mut residual = vec![0.0; rhs.len()];
        assert!(solver.residual_full(&mut residual, &rhs, &out).is_finite());
    }

    #[test]
    fn phase_api_panel_nonfinite_does_not_publish() {
        let p = CscMatrix::<f64>::identity(1);
        let a = CscMatrix::zeros((0, 1));
        let cones = CompositeCone::new(&[]);
        let mut settings = CoreSettings::<f64>::default();
        settings.direct_solve_method = "qdldl".to_string();
        settings.iterative_refinement_enable = false;
        let mut solver = DirectLDLKKTSolver::new(&p, &a, &cones, 0, 1, &settings);
        solver.assemble_from_cones(cones.iter());
        assert!(solver.factor_with_shift(&settings, None));

        let rhs = [f64::NAN];
        let mut out = [7.0];
        assert!(!solver.solve_factor_panel(&rhs, &mut out, 1));
        assert_eq!(out, [7.0]);
    }

    #[test]
    #[cfg(feature = "sdp")]
    fn dense_refinement_uses_original_updated_triangle() {
        let n = 512;
        let mut p = CscMatrix::zeros((n, n));
        p.colptr = vec![0];
        for col in 0..n {
            for row in 0..=col {
                p.rowval.push(row);
                p.nzval.push(if row == col { 4.0 } else { 0.001 });
            }
            p.colptr.push(p.nzval.len());
        }
        let a = CscMatrix::zeros((0, n));
        let cones = CompositeCone::new(&[]);
        let mut settings = CoreSettings::default();
        settings.static_regularization_constant = 1e-3;
        let mut solver = DirectLDLKKTSolver::new(&p, &a, &cones, 0, n, &settings);
        let expected: Vec<_> = (0..n).map(|i| (i % 7) as f64 - 3.0).collect();
        for repeat in 0..2 {
            if repeat == 1 {
                p.nzval.iter_mut().for_each(|v| *v *= 1.25);
                solver.update_P(&p);
            }
            assert!(solver.update(&cones, &settings));
            let dense = solver.residual_dense.as_ref().unwrap();
            for col in 0..n {
                for pos in p.colptr[col]..p.colptr[col + 1] {
                    assert_eq!(dense[(p.rowval[pos], col)], p.nzval[pos]);
                }
            }
            let mut rhs = vec![0.; n];
            p.sym(MatrixTriangle::Triu)
                .symv(&mut rhs, &expected, 1., 0.);
            solver.setrhs(&rhs, &[]);
            let mut actual = vec![0.; n];
            assert!(solver.solve(Some(&mut actual), None, &settings));
            assert!(actual
                .iter()
                .zip(&expected)
                .all(|(a, b)| (a - b).abs() < 1e-8));
            let mut residual = rhs.clone();
            p.sym(MatrixTriangle::Triu)
                .symv(&mut residual, &actual, -1., 1.);
            assert!(residual.norm_inf() < 1e-8);
        }
    }

    #[test]
    fn regularization_budget_and_solve_counters_reset() {
        let p = CscMatrix::<f64>::identity(1);
        let a = CscMatrix::zeros((0, 1));
        let cones = CompositeCone::new(&[]);
        let settings = CoreSettings::default();
        let mut solver = DirectLDLKKTSolver::new(&p, &a, &cones, 0, 1, &settings);
        for expected in 1..=3 {
            assert!(solver.escalate_regularization());
            assert_eq!(solver.reg_boost, expected);
        }
        for _ in 0..3 {
            assert!(!solver.escalate_regularization());
            assert_eq!(solver.reg_boost, 3);
        }
        assert!(solver.update(&cones, &settings));
        solver.setrhs(&[2.0], &[]);
        let mut x = [0.0];
        assert!(solver.solve(Some(&mut x), None, &settings));
        let counters = solver.counters();
        assert_eq!(counters.factor_attempts, 1);
        assert_eq!(counters.factorizations, 1);
        assert_eq!(counters.rhs_applied, 1);
        assert_eq!(counters.linear_solves, 1 + counters.refinements);
        solver.reset_solve();
        assert_eq!(solver.reg_boost, 0);
        assert_eq!(solver.counters(), Default::default());
        assert!(solver.update(&cones, &settings));
        solver.setrhs(&[3.0], &[]);
        assert!(solver.solve(Some(&mut x), None, &settings));
        assert!((x[0] - 3.0).abs() < 1e-8);
        assert_eq!(solver.counters().rhs_applied, 1);
    }

    fn activation<T: FloatT>() {
        let n = 192;
        let mut ptr = vec![0];
        let mut rows = Vec::new();
        let mut values = Vec::new();
        for j in 0..n {
            for i in 0..=j {
                rows.push(i);
                values.push(if i == j {
                    T::from_f64(3.).unwrap()
                } else {
                    T::from_f64(0.125).unwrap()
                });
            }
            ptr.push(rows.len());
        }
        let mut p = CscMatrix::new(n, n, ptr, rows, values);
        let a = CscMatrix::zeros((0, n));
        let cones = CompositeCone::<T>::new(&[]);
        let mut settings = CoreSettings::<T>::default();
        settings.direct_solve_method = "qdldl".into();
        let mut solver = DirectLDLKKTSolver::new(&p, &a, &cones, 0, n, &settings);
        assert!(solver.residual_plan.is_none());
        let rhs = vec![T::one(); n];
        let mut x: Vec<_> = (0..n)
            .map(|i| T::from_f64((i % 7) as f64 - 3.).unwrap())
            .collect();
        let mut storage = None;
        for workers in [4, 1, 2, 4] {
            let pool = (workers > 1).then(|| {
                std::sync::Arc::new(
                    rayon::ThreadPoolBuilder::new()
                        .num_threads(workers)
                        .build()
                        .unwrap(),
                )
            });
            solver.set_residual_pool(pool);
            let plan = solver.residual_plan.as_ref().unwrap();
            let (actual_workers, address) = plan.test_pool_and_storage();
            assert_eq!(actual_workers, workers);
            if let Some(previous) = storage {
                assert_eq!(previous, address);
            } else {
                storage = Some(address);
            }
            p.nzval[0] += T::from_f64(0.125).unwrap();
            solver.update_P(&p);
            let mut expected = vec![T::zero(); n];
            let mut actual = expected.clone();
            let k = solver.KKT.sym(solver.KKTuplo);
            let en = _get_refine_error(&mut expected, &rhs, &k, &mut x, None);
            let an =
                _get_refine_error(&mut actual, &rhs, &k, &mut x, solver.residual_plan.as_ref());
            assert_eq!(actual, expected);
            assert_eq!(an, en);
        }
    }
    #[test]
    fn activation_f64() {
        activation::<f64>();
    }
    #[test]
    fn activation_mpfr256() {
        activation::<sdpx_arithmetic::Bits256>();
    }
    #[test]
    fn activation_mpfr512() {
        activation::<sdpx_arithmetic::Bits512>();
    }
}
