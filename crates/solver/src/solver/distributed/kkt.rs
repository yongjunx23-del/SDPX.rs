//! Owner-local condensed KKT with one shared equality Schur complement.
//! Both refinement levels use the existing original-operator acceptance policy.
use super::*;
use crate::algebra::sparse_parallel::SparseParallel;
use crate::solver::kkt::SolveCounters;
use crate::solver::{
    core::CoreSettings,
    kkt::{
        direct::DirectLDLKKTSolver,
        refinement::{refine, Refinement},
        CondensedKKTSolver, KKTSolver,
    },
};
use rayon::prelude::*;
use std::sync::Arc;
use std::time::Instant;

struct LocalKkt<T: FloatT> {
    kernel: CondensedKKTSolver<T>,
    n: usize,
    width: usize,
    border_rows: Vec<usize>,
    coupling: CscMatrix<T>,
    /// Row-grouped view of `coupling` for the border assembly.
    coupling_rows: CouplingRows,
    /// Lanes for the border coupling products (54 x ~1200 dense on Λ19
    /// spins 0-50; 18 of 77 s serial on a 32-thread rank without them).
    coupling_plan: SparseParallel,
    coupling_rhs: Vec<T>,
    response: Vec<T>,
    // Two-column panel scratch for the constant/affine predictor pair.  The
    // values stay owner-local so MPFR storage is always owned by this solver.
    // Pair RHS input reuses the post-factor coupling scratch whenever it has
    // room for both columns.  The fallback stays lazy for border sizes 0/1.
    panel_rhs: Option<Vec<T>>,
    panel_out: Option<Vec<T>>,
    local_assemble_ns: f64,
    factor_response_ns: f64,
    // Fused sampled NT preparation leaves an RHS-specific mat3c in the
    // local kernel. Keep one snapshot per pair lane so recovery restores the
    // cache belonging to the column being recovered.
    rhs_cache: [Vec<T>; 2],
}

impl<T: FloatT> LocalKkt<T> {
    /// `y = alpha·op(coupling)·x + beta·y` over the owner pool's lanes; each
    /// output keeps the serial CSC gemv arithmetic, so the bits never change.
    fn coupling_product(
        &self,
        pool: Option<&Arc<rayon::ThreadPool>>,
        transpose: bool,
        y: &mut [T],
        x: &[T],
        alpha: T,
        beta: T,
    ) {
        match pool.filter(|_| self.coupling_plan.has_lanes()) {
            Some(pool) => pool.install(|| {
                self.coupling_plan
                    .product(&self.coupling, transpose, y, x, alpha, beta)
            }),
            None if transpose => self.coupling.t().gemv(y, x, alpha, beta),
            None => self.coupling.gemv(y, x, alpha, beta),
        }
    }

    fn panel_buffer(slot: &mut Option<Vec<T>>, len: usize) -> &mut [T] {
        let panel = slot.get_or_insert_with(|| Vec::with_capacity(len));
        if panel.len() < len {
            panel.resize(len, T::zero());
        }
        &mut panel[..len]
    }

    /// Apply one already-factorized local interior KKT to the constant and
    /// affine RHS together.  A bad column is excluded before the panel call;
    /// if a shared panel reports a nonfinite result, retrying its columns
    /// separately preserves failure isolation without changing the factors.
    fn solve_pair(&mut self, rhs0: &[T], rhs1: &[T], out0: &mut [T], out1: &mut [T]) -> [bool; 2] {
        let mut ok = [false; 2];
        let finite0 = rhs0.is_finite();
        let finite1 = rhs1.is_finite();
        if finite0 && finite1 {
            let panel_len = rhs0.len() + rhs1.len();
            let panel_rhs = if self.coupling_rhs.len() >= panel_len {
                &mut self.coupling_rhs[..panel_len]
            } else {
                Self::panel_buffer(&mut self.panel_rhs, panel_len)
            };
            panel_rhs[..rhs0.len()].copy_from_slice(rhs0);
            panel_rhs[rhs0.len()..].copy_from_slice(rhs1);
            let panel_out = Self::panel_buffer(&mut self.panel_out, panel_len);
            let kernel = &mut self.kernel;
            if kernel.solve_interior_panel(panel_rhs, panel_out, 2) {
                let split = out0.len();
                out0.copy_from_slice(&panel_out[..split]);
                out1.copy_from_slice(&panel_out[split..]);
                return [true, true];
            }
        }
        if finite0 && self.kernel.solve_interior_panel(rhs0, out0, 1) {
            ok[0] = true;
        }
        if finite1 && self.kernel.solve_interior_panel(rhs1, out1, 1) {
            ok[1] = true;
        }
        ok
    }
}

struct Point<T> {
    blocks: Vec<Vec<T>>,
    border: Vec<T>,
}
impl<T: FloatT> Point<T> {
    fn new(dimensions: &[usize], border: usize) -> Self {
        Self {
            blocks: dimensions.iter().map(|&n| vec![T::zero(); n]).collect(),
            border: vec![T::zero(); border],
        }
    }
    fn norm(&self) -> T {
        let mut norm = T::zero();
        for values in self
            .blocks
            .iter()
            .map(Vec::as_slice)
            .chain([self.border.as_slice()])
        {
            if !values.is_finite() {
                return T::infinity();
            }
            norm = T::max(norm, values.norm_inf());
        }
        norm
    }
    fn zeros_like(&self) -> Self {
        Self {
            blocks: self
                .blocks
                .iter()
                .map(|b| vec![T::zero(); b.len()])
                .collect(),
            border: vec![T::zero(); self.border.len()],
        }
    }
    fn clone_point(&self) -> Self {
        Self {
            blocks: self.blocks.clone(),
            border: self.border.clone(),
        }
    }
    fn copy_from_point(&mut self, other: &Self) {
        for (a, b) in self.blocks.iter_mut().zip(&other.blocks) {
            a.copy_from_slice(b);
        }
        self.border.copy_from_slice(&other.border);
    }
    fn fill(&mut self, c: T) {
        for v in self
            .blocks
            .iter_mut()
            .flatten()
            .chain(self.border.iter_mut())
        {
            *v = c;
        }
    }
    fn scale(&mut self, c: T) {
        for v in self
            .blocks
            .iter_mut()
            .flatten()
            .chain(self.border.iter_mut())
        {
            *v *= c;
        }
    }
    fn negate(&mut self) {
        for v in self
            .blocks
            .iter_mut()
            .flatten()
            .chain(self.border.iter_mut())
        {
            *v = -*v;
        }
    }
    fn axpy(&mut self, a: T, other: &Self) {
        for (x, y) in self.blocks.iter_mut().zip(&other.blocks) {
            x.axpby(a, y, T::one());
        }
        self.border.axpby(a, &other.border, T::one());
    }
    /// This rank's owned-block part of the inner product.
    fn block_dot(&self, other: &Self) -> T {
        let mut s = T::zero();
        for (x, y) in self.blocks.iter().zip(&other.blocks) {
            s += x.dot(y);
        }
        s
    }
    fn add(&mut self, other: &Self) {
        for (a, b) in self.blocks.iter_mut().zip(&other.blocks) {
            for (a, &b) in a.iter_mut().zip(b) {
                *a += b;
            }
        }
        for (a, &b) in self.border.iter_mut().zip(&other.border) {
            *a += b;
        }
    }
}
struct Work<T> {
    b: Point<T>,
    x: Point<T>,
    error: Point<T>,
    candidate: Point<T>,
}

/// Apply `row(i, &mut x[i])` to every interior row, split over `pool` by
/// rows when each task carries a grain of `border`-long dots. Rows are
/// independent, so the split never changes an entry.
fn border_rows<T: FloatT>(
    pool: Option<&Arc<rayon::ThreadPool>>,
    x: &mut [T],
    border: usize,
    row: impl Fn(usize, &mut T) + Sync + Send,
) {
    let d = x.len();
    let work = (d as u128 * border as u128)
        * sdpx_arithmetic::inner_parallel::weight(T::precision_bits() as usize);
    let tasks = sdpx_arithmetic::inner_parallel::tasks_if(pool.is_some(), work, d);
    match pool.filter(|_| tasks > 1) {
        Some(pool) => pool.install(|| {
            x.par_iter_mut()
                .enumerate()
                .with_min_len(d.div_ceil(tasks))
                .for_each(|(i, v)| row(i, v))
        }),
        None => x.iter_mut().enumerate().for_each(|(i, v)| row(i, v)),
    }
}

fn elapsed_ns(started: Instant) -> f64 {
    started.elapsed().as_nanos().max(1) as f64
}
impl<T: FloatT> Work<T> {
    fn new(dimensions: &[usize], border: usize) -> Self {
        Self {
            b: Point::new(dimensions, border),
            x: Point::new(dimensions, border),
            error: Point::new(dimensions, border),
            candidate: Point::new(dimensions, border),
        }
    }
}

pub(crate) struct OwnedKkt<T: FloatT> {
    locals: Vec<LocalKkt<T>>,
    border_matrix: CscMatrix<T>,
    border_factor: DirectLDLKKTSolver<T>,
    border_rhs: Vec<T>,
    inner: Option<Work<T>>,
    outer: Option<Work<T>>,
    // One extra lane; the scalar inner/outer workspaces provide the other
    // lane during a pair solve.
    batch_inner: Option<Work<T>>,
    batch_outer: Option<Work<T>>,
    border_panel_rhs: Vec<T>,
    reg_boost: usize,
    pub(crate) shift: T,
    scaled_valid: bool,
    pool: Option<Arc<rayon::ThreadPool>>,
    rhs_applied: u64,
    refinements: u64,
    /// Residual floor of the last reduced-level GMRES-IR solve.
    gmres_floor: Option<T>,
    record_costs: bool,
    collective: Arc<dyn crate::solver::distributed::collective::Collective<T>>,
    last_border: Vec<T>,
    pair_border: Vec<T>,
    /// Speculative reduced correction prepared with the last fused reduced
    /// residual: local interior solves and the border right-hand side.
    fused_interior: Vec<Vec<T>>,
    fused_border_rhs: Vec<T>,
    fused_ready: bool,
}

impl<T: FloatT> OwnedKkt<T> {
    /// Test-only convenience over [`Self::new_with_pool_collective`].
    #[cfg(test)]
    pub(crate) fn new_with_pool<'a>(
        layout: &OwnerLayout,
        parts: impl Iterator<Item = (&'a DefaultProblemData<T>, &'a CompositeCone<T>)>,
        settings: &CoreSettings<T>,
        pool: Option<Arc<rayon::ThreadPool>>,
        record_costs: bool,
    ) -> Self {
        let owner_ids: Vec<_> = (0..layout.owners.len()).collect();
        Self::new_with_pool_collective(
            layout,
            &owner_ids,
            parts,
            settings,
            pool,
            record_costs,
            Arc::new(crate::solver::distributed::collective::SerialCollective),
        )
    }

    pub(crate) fn new_with_pool_collective<'a>(
        layout: &OwnerLayout,
        owner_ids: &[usize],
        parts: impl Iterator<Item = (&'a DefaultProblemData<T>, &'a CompositeCone<T>)>,
        settings: &CoreSettings<T>,
        pool: Option<Arc<rayon::ThreadPool>>,
        record_costs: bool,
        collective: Arc<dyn crate::solver::distributed::collective::Collective<T>>,
    ) -> Self {
        // One pool owns the complete thread budget. Construct local kernels
        // without private pools; large matrix tasks borrow this same pool.
        let mut local_settings = settings.clone();
        local_settings.max_threads = 1;
        let settings = &local_settings;
        let border = layout.border_rows.len();
        let parts: Vec<_> = parts.collect();
        // Owners in one process run concurrently on the shared pool: each
        // plans its block phases for an equal share of it.
        let share = match &pool {
            Some(p) if parts.len() > 1 => (p.current_num_threads() / parts.len()).max(1),
            _ => 0,
        };
        let locals: Vec<_> = parts
            .into_iter()
            .zip(owner_ids.iter().enumerate())
            .map(|((data, cones), (owner, _global_owner))| {
                let ids = &layout.owners[owner];
                let heavy_owner = layout.owner_inner_admission(owner);
                let mut kernel = CondensedKKTSolver::new_local_partition(
                    &data.P,
                    &data.A,
                    &data.cones,
                    cones,
                    settings,
                    false,
                );
                // Keep one shared pool, but let a structurally or historically
                // dominant owner expose its existing inner matrix lanes.  The
                // condensed planner still requires >1 workers, 4096-unit
                // kernel work and a 75% local dominance threshold.
                kernel.set_owner_inner_admission(heavy_owner);
                if !heavy_owner {
                    kernel.set_worker_share(share);
                }
                if let Some(sampled) = &data.sampled {
                    kernel.set_sampled_operator(Arc::clone(sampled));
                }
                if pool.is_some() {
                    kernel.prepare_shared_pool();
                }
                let rows: Vec<_> = layout
                    .border_rows
                    .iter()
                    .map(|r| ids.rows.binary_search(r).unwrap())
                    .collect();
                // Only coupling entries are selected, not all data columns/rows.
                let columns: Vec<_> = (0..data.n).collect();
                let coupling = select_rows(&data.A, &rows, &columns);
                let coupling_rows = CouplingRows::new(&coupling);
                let mut coupling_plan = SparseParallel::new(&coupling);
                coupling_plan.configure(&coupling, pool.clone());
                let cells = kernel.interior_dimension() * border;
                LocalKkt {
                    kernel,
                    n: data.n,
                    width: data.n + data.m,
                    border_rows: rows,
                    coupling,
                    coupling_rows,
                    coupling_plan,
                    coupling_rhs: vec![T::zero(); cells],
                    response: vec![T::zero(); cells],
                    panel_rhs: None,
                    panel_out: None,
                    local_assemble_ns: 0.0,
                    factor_response_ns: 0.0,
                    rhs_cache: [Vec::new(), Vec::new()],
                }
            })
            .collect();
        let ptr: Vec<_> = (0..=border).map(|j| j * (j + 1) / 2).collect();
        let rowval = (0..border).flat_map(|j| 0..=j).collect();
        let matrix = CscMatrix::new(
            border,
            border,
            ptr,
            rowval,
            vec![T::zero(); border * (border + 1) / 2],
        );
        let cones = CompositeCone::new(&[]);
        let mut factor = DirectLDLKKTSolver::new(
            &matrix,
            &CscMatrix::zeros((0, border)),
            &cones,
            0,
            border,
            settings,
        );
        // The dense border is a single arrow leaf; without the pool its LDL
        // and sweeps ran serially (Λ27 4 nodes: 1.3 s of 48 s per rank).
        factor.set_factor_pool(pool.clone());
        let interior: Vec<_> = locals
            .iter()
            .map(|l| l.kernel.interior_dimension())
            .collect();
        let original: Vec<_> = locals.iter().map(|l| l.width).collect();
        Self {
            locals,
            border_matrix: matrix,
            border_factor: factor,
            border_rhs: vec![T::zero(); border],
            inner: Some(Work::new(&interior, border)),
            outer: Some(Work::new(&original, border)),
            // Pair lanes are allocated lazily; scalar-only callers retain
            // only the workspaces needed by the ordinary solve path.
            batch_inner: None,
            batch_outer: None,
            border_panel_rhs: vec![T::zero(); 2 * border],
            reg_boost: 0,
            shift: T::zero(),
            scaled_valid: false,
            pool,
            rhs_applied: 0,
            refinements: 0,
            gmres_floor: None,
            record_costs,
            collective,
            last_border: vec![T::zero(); border],
            pair_border: vec![T::zero(); 2 * border],
            fused_interior: Vec::new(),
            fused_border_rhs: Vec::new(),
            fused_ready: false,
        }
    }

    fn new_pair_work(&self, reduced: bool, border: usize) -> Work<T> {
        let dimensions: Vec<_> = self
            .locals
            .iter()
            .map(|local| {
                if reduced {
                    local.kernel.interior_dimension()
                } else {
                    local.width
                }
            })
            .collect();
        Work::new(&dimensions, border)
    }

    pub fn reset_solve(&mut self) {
        self.reg_boost = 0;
        self.scaled_valid = false;
        self.rhs_applied = 0;
        self.refinements = 0;
        for local in &mut self.locals {
            local.kernel.reset_solve();
            local.local_assemble_ns = 0.0;
            local.factor_response_ns = 0.0;
        }
        self.border_factor.reset_solve();
        self.last_border.fill(T::zero());
        self.pair_border.fill(T::zero());
    }

    pub(crate) fn last_border(&self) -> &[T] {
        &self.last_border
    }

    pub(crate) fn pair_border(&self, column: usize) -> &[T] {
        let border = self.border_matrix.n;
        let start = column.min(1) * border;
        &self.pair_border[start..start + border]
    }
    pub fn escalate_regularization(&mut self) -> bool {
        if self.reg_boost == 3 {
            return false;
        }
        self.reg_boost += 1;
        true
    }

    pub(crate) fn update_local(
        &mut self,
        cones: &[CompositeCone<T>],
        settings: &CoreSettings<T>,
    ) -> bool {
        let timer = crate::receipt::start();
        let shared_pool = self.pool.clone();
        let record_costs = self.record_costs;
        let assemble = |(local, cones): (&mut LocalKkt<T>, &CompositeCone<T>)| {
            let started = record_costs.then(Instant::now);
            let valid = local.kernel.update_partition_with_pool(
                cones,
                settings,
                false,
                shared_pool.clone(),
            );
            if let Some(started) = started {
                local.local_assemble_ns += elapsed_ns(started);
            }
            valid
        };
        let valid = all_blocks!(self.pool.as_ref(), (&mut self.locals, cones), assemble);
        crate::receipt::finish("owned.local_assemble", timer);
        self.finish_update(settings, valid)
    }
    fn finish_update(&mut self, settings: &CoreSettings<T>, valid: bool) -> bool {
        self.scaled_valid = false;
        let mut diagonal = T::zero();
        for local in &self.locals {
            diagonal = T::max(diagonal, local.kernel.interior_diagonal_norm());
        }
        let local_ok = valid && diagonal.is_finite();
        let global_ok = self.collective.all_true(100, local_ok).unwrap_or(false);
        let local_diag = if diagonal.is_finite() {
            diagonal
        } else {
            T::zero()
        };
        let diagonal = self
            .collective
            .reduce_max(101, std::slice::from_ref(&local_diag))
            .ok()
            .and_then(|values| values.into_iter().next())
            .unwrap_or(T::infinity());
        if !global_ok || !diagonal.is_finite() {
            return false;
        }
        // Shared equality diagonals are zero. Match the original complete
        // reduced KKT's shift, including the sparse-cone auxiliary diagonals.
        self.shift = if settings.static_regularization_enable {
            let base = settings.static_regularization_constant
                + settings.static_regularization_proportional * diagonal;
            if self.reg_boost == 0 {
                base
            } else {
                base * T::from_f64(100f64.powi(self.reg_boost as i32)).unwrap()
            }
        } else {
            T::zero()
        };
        self.border_matrix.nzval.fill(T::zero());
        // Preserve the serial accumulation order; the shared shift has one
        // canonical contributor when the matrix is reduced across ranks.
        if self.collective.rank() == 0 {
            for j in 0..self.border_matrix.n {
                self.border_matrix.nzval[self.border_matrix.colptr[j] + j] = self.shift;
            }
        }
        let border = self.border_matrix.n;
        let shift = self.shift;
        let timer = crate::receipt::start();
        let record_costs = self.record_costs;
        let factor = |(local,): (&mut LocalKkt<T>,)| {
            let started = record_costs.then(Instant::now);
            let valid = if !local.kernel.factor_interior(settings, shift) {
                false
            } else {
                let d = local.kernel.interior_dimension();
                local.coupling_rhs.fill(T::zero());
                for col in 0..local.n {
                    for p in local.coupling.colptr[col]..local.coupling.colptr[col + 1] {
                        local.coupling_rhs[local.coupling.rowval[p] * d + col] +=
                            local.coupling.nzval[p];
                    }
                }
                local
                    .kernel
                    .solve_interior_panel(&local.coupling_rhs, &mut local.response, border)
            };
            if let Some(started) = started {
                local.factor_response_ns += elapsed_ns(started);
            }
            valid
        };
        let valid = all_blocks!(self.pool.as_ref(), (&mut self.locals,), factor);
        crate::receipt::finish("owned.factor_response", timer);
        let valid = self.collective.all_true(102, valid).unwrap_or(false);
        if !valid {
            return false;
        }
        let timer = crate::receipt::start();
        if !self.assemble_border() {
            return false;
        }
        self.border_factor.update_P(&self.border_matrix);
        crate::receipt::finish("owned.border_assemble", timer);
        let mut border_settings = settings.clone();
        // Q = epsilon*I + sum Aeq * Kinterior^-1 * Aeq'. Its diagonal already
        // includes the one global equality shift. Dynamic pivot safeguards stay.
        border_settings.static_regularization_enable = false;
        let timer = crate::receipt::start();
        let result = self.border_factor.factor_with_shift(&border_settings, None);
        let result = self.collective.all_true(104, result).unwrap_or(false);
        crate::receipt::finish("owned.border_factor", timer);
        result
    }

    fn assemble_border(&mut self) -> bool {
        let border = self.border_matrix.n;
        let locals = &self.locals;
        let pointers = &self.border_matrix.colptr;
        // Column j of the upper triangle needs coupling rows 0..=j only. Each
        // entry is the value the full local GEMV gives that row (same terms,
        // order and rounding) added in owner order, so the matrix is
        // unchanged while about half the products are skipped. Columns are
        // independent; the row grouping is built once, not per GEMV.
        let assemble = |j: usize, column: &mut [T]| {
            for local in locals {
                let d = local.kernel.interior_dimension();
                let x = &local.response[j * d..j * d + local.n];
                for (i, entry) in column.iter_mut().enumerate() {
                    *entry += local.coupling_rows.value(&local.coupling, i, x);
                }
            }
        };
        let mut columns: Vec<(usize, &mut [T])> = Vec::with_capacity(border);
        let mut remaining = self.border_matrix.nzval.as_mut_slice();
        for j in 0..border {
            let (column, tail) = remaining.split_at_mut(pointers[j + 1] - pointers[j]);
            columns.push((j, column));
            remaining = tail;
        }
        match self.pool.as_ref().filter(|p| p.current_num_threads() > 1) {
            Some(pool) => {
                // Longest columns first; workers take one column at a time.
                columns.reverse();
                pool.install(|| {
                    columns
                        .into_par_iter()
                        .with_max_len(1)
                        .for_each(|(j, column)| assemble(j, column))
                });
            }
            None => columns
                .into_iter()
                .for_each(|(j, column)| assemble(j, column)),
        }
        self.collective
            .reduce_sum_in_place(103, &mut self.border_matrix.nzval)
            .is_ok()
    }

    fn fill_work_rhs(
        &self,
        work: &mut Work<T>,
        rhs: &[DefaultVariables<T>],
        rhs_border: Option<&[T]>,
    ) -> bool {
        for ((local, b), rhs) in self.locals.iter().zip(&mut work.b.blocks).zip(rhs) {
            b[..local.n].copy_from_slice(&rhs.x);
            b[local.n..].copy_from_slice(&rhs.z);
            for &r in &local.border_rows {
                b[local.n + r] = T::zero();
            }
        }
        if let Some(border) = rhs_border {
            if border.len() != work.b.border.len() {
                return false;
            }
            work.b.border.copy_from_slice(border);
        } else if let Some(first) = self.locals.first() {
            for (i, &r) in first.border_rows.iter().enumerate() {
                work.b.border[i] = rhs[0].z[r];
            }
        }
        work.b.norm().is_finite()
    }

    fn prepare_reduced_work(&mut self, source: &Point<T>, work: &mut Work<T>, lane: usize) {
        let prepare = |(local, b, rhs): (&mut LocalKkt<T>, &mut Vec<T>, &Vec<T>)| {
            local.kernel.prepare_interior_rhs(rhs, b);
            local
                .kernel
                .copy_sampled_rhs_cache(&mut local.rhs_cache[lane]);
        };
        for_blocks!(
            self.pool.as_ref(),
            (&mut self.locals, &mut work.b.blocks, &source.blocks),
            prepare
        );
        work.b.border.copy_from_slice(&source.border);
    }

    /// Linear (x,z) solve only. The single HSD layer computes tau/kappa and
    /// recovers delta-s; this function leaves those output fields untouched.
    ///
    /// Test-only entry point; production callers use
    /// [`Self::solve_blocks_with_border`].
    #[cfg(test)]
    pub(crate) fn solve_blocks(
        &mut self,
        rhs: &[DefaultVariables<T>],
        out: &mut [DefaultVariables<T>],
        settings: &CoreSettings<T>,
    ) -> bool {
        self.solve_blocks_with_border(rhs, out, None, settings)
    }

    pub(crate) fn solve_blocks_with_border(
        &mut self,
        rhs: &[DefaultVariables<T>],
        out: &mut [DefaultVariables<T>],
        rhs_border: Option<&[T]>,
        settings: &CoreSettings<T>,
    ) -> bool {
        self.rhs_applied += 1;
        self.scaled_valid = false;
        let mut work = self.outer.take().unwrap();
        let input_ok = self
            .collective
            .all_true(249, self.fill_work_rhs(&mut work, rhs, rhs_border))
            .unwrap_or(false);
        let success = input_ok
            && self.solve_original(&mut work.x, &work.b, settings)
            && refine(
                &mut OwnedRefinement {
                    solver: self,
                    work: &mut work,
                    reduced: false,
                    basis: Vec::new(),
                    directions: Vec::new(),
                },
                settings,
            );
        let success = self.collective.all_true(250, success).unwrap_or(false);
        if success {
            for ((local, x), out) in self.locals.iter().zip(&work.x.blocks).zip(out) {
                out.x.copy_from_slice(&x[..local.n]);
                out.z.copy_from_slice(&x[local.n..]);
            }
            self.last_border.copy_from_slice(&work.x.border);
            self.scaled_valid = settings.iterative_refinement_enable;
        }
        self.outer = Some(work);
        success
    }

    /// Solve the constant and affine owner-local RHS together after one
    /// factor update.  Interior panels share the existing local factors; the
    /// original residual/refinement policy still runs independently per
    /// column, including accepted-product recovery and failure isolation.
    /// Test-only entry point; production callers use
    /// [`Self::solve_blocks_pair_with_border`].
    #[cfg(test)]
    pub(crate) fn solve_blocks_pair(
        &mut self,
        rhs: [&[DefaultVariables<T>]; 2],
        out: [&mut [DefaultVariables<T>]; 2],
        products: [&mut [Vec<T>]; 2],
        settings: &CoreSettings<T>,
    ) -> [bool; 2] {
        self.solve_blocks_pair_with_border(rhs, out, products, [None, None], settings)
    }

    pub(crate) fn solve_blocks_pair_with_border(
        &mut self,
        rhs: [&[DefaultVariables<T>]; 2],
        out: [&mut [DefaultVariables<T>]; 2],
        mut products: [&mut [Vec<T>]; 2],
        rhs_border: [Option<&[T]>; 2],
        settings: &CoreSettings<T>,
    ) -> [bool; 2] {
        self.rhs_applied += 2;
        self.scaled_valid = false;
        for output in products.iter_mut() {
            (*output).iter_mut().for_each(Vec::clear);
        }

        let mut inner0 = self.inner.take().unwrap();
        let mut outer0 = self.outer.take().unwrap();
        let mut inner1 = self
            .batch_inner
            .take()
            .unwrap_or_else(|| self.new_pair_work(true, self.border_matrix.n));
        let mut outer1 = self
            .batch_outer
            .take()
            .unwrap_or_else(|| self.new_pair_work(false, self.border_matrix.n));

        let mut flags = [
            self.fill_work_rhs(&mut outer0, rhs[0], rhs_border[0]),
            self.fill_work_rhs(&mut outer1, rhs[1], rhs_border[1]),
        ];
        for c in 0..2 {
            if flags[c] {
                if c == 0 {
                    self.prepare_reduced_work(&outer0.b, &mut inner0, 0);
                } else {
                    self.prepare_reduced_work(&outer1.b, &mut inner1, 1);
                }
            }
        }

        let reduced = self.solve_reduced_panel([&mut inner0, &mut inner1], flags);
        flags = reduced;
        for c in 0..2 {
            if !flags[c] {
                continue;
            }
            flags[c] = refine(
                &mut OwnedRefinement {
                    solver: self,
                    work: if c == 0 { &mut inner0 } else { &mut inner1 },
                    reduced: true,
                    basis: Vec::new(),
                    directions: Vec::new(),
                },
                settings,
            );
            flags[c] = self.collective.all_true(260 + c, flags[c]).unwrap_or(false);
            if !flags[c] {
                continue;
            }
            flags[c] = if c == 0 {
                self.recover_original_work(&mut outer0.x, &outer0.b, &inner0.x, Some(0))
            } else {
                self.recover_original_work(&mut outer1.x, &outer1.b, &inner1.x, Some(1))
            };
            flags[c] = self.collective.all_true(270 + c, flags[c]).unwrap_or(false);
            if !flags[c] {
                continue;
            }
        }

        // The second lane is no longer needed for reduced recovery.  Reuse it
        // as the scalar correction workspace required by the shared original
        // refinement helper, leaving only one extra persistent Work value.
        self.inner = Some(inner1);
        for c in 0..2 {
            if !flags[c] {
                continue;
            }
            flags[c] = refine(
                &mut OwnedRefinement {
                    solver: self,
                    work: if c == 0 { &mut outer0 } else { &mut outer1 },
                    reduced: false,
                    basis: Vec::new(),
                    directions: Vec::new(),
                },
                settings,
            );
            flags[c] = self.collective.all_true(280 + c, flags[c]).unwrap_or(false);
            if flags[c] {
                for ((local, x), output) in self
                    .locals
                    .iter()
                    .zip(if c == 0 {
                        &outer0.x.blocks
                    } else {
                        &outer1.x.blocks
                    })
                    .zip(out[c].iter_mut())
                {
                    output.x.copy_from_slice(&x[..local.n]);
                    output.z.copy_from_slice(&x[local.n..]);
                }
                if settings.iterative_refinement_enable {
                    self.scaled_valid = true;
                    self.copy_products_to(products[c]);
                }
                let start = c * self.border_matrix.n;
                self.pair_border[start..start + self.border_matrix.n].copy_from_slice(if c == 0 {
                    &outer0.x.border
                } else {
                    &outer1.x.border
                });
            }
        }
        // Product buffers above retain each accepted column.  The kernel's
        // single-product view is intentionally invalidated because the pair
        // solve has no single authoritative current column.
        self.scaled_valid = false;
        inner1 = self.inner.take().unwrap();
        self.inner = Some(inner1);
        self.outer = Some(outer0);
        self.batch_inner = Some(inner0);
        self.batch_outer = Some(outer1);
        flags
    }

    fn copy_products_to(&self, out: &mut [Vec<T>]) {
        for (i, out) in out.iter_mut().enumerate() {
            if let Some(product) = self.scaled_product(i) {
                out.resize(product.len(), T::zero());
                out.copy_from_slice(product);
            } else {
                out.clear();
            }
        }
    }

    pub fn scaled_product(&self, owner: usize) -> Option<&[T]> {
        if self.scaled_valid {
            self.locals[owner].kernel.owned_scaled_product()
        } else {
            None
        }
    }

    pub(crate) fn linear_info(&self) -> crate::solver::kkt::LinearSolverInfo {
        use crate::solver::kkt::HasLinearSolverInfo;
        let mut info = self.border_factor.linear_solver_info();
        info.name = "owned_condensed".into();
        info.threads = self
            .pool
            .as_ref()
            .map_or(1, |pool| pool.current_num_threads());
        for local in &self.locals {
            let value = local.kernel.linear_solver_info();
            info.nnzA += value.nnzA;
            info.nnzL += value.nnzL;
            info.threads = info.threads.max(value.threads);
        }
        info
    }

    pub(crate) fn counters(&self) -> SolveCounters {
        let mut total = self.border_factor.counters();
        for local in &self.locals {
            let c = local.kernel.interior_counters();
            total.factor_attempts += c.factor_attempts;
            total.factorizations += c.factorizations;
            total.linear_solves += c.linear_solves;
            total.refinements += c.refinements;
            total.batches += c.batches;
        }
        total.refinements += self.refinements;
        total.outer_refinements = self.refinements;
        total.rhs_applied = self.rhs_applied;
        // Backend work includes local response panels and the shared boundary;
        // caller RHS counts are counted only once at the global solve boundary.
        total
    }

    fn solve_reduced_panel(&mut self, work: [&mut Work<T>; 2], input_ok: [bool; 2]) -> [bool; 2] {
        let [first, second] = work;
        let solve = |((((local, b0), b1), x0), x1): (
            (((&mut LocalKkt<T>, &Vec<T>), &Vec<T>), &mut Vec<T>),
            &mut Vec<T>,
        )| {
            if input_ok[0] && input_ok[1] {
                local.solve_pair(b0, b1, x0, x1)
            } else {
                [
                    input_ok[0] && local.kernel.solve_interior_panel(b0, x0, 1),
                    input_ok[1] && local.kernel.solve_interior_panel(b1, x1, 1),
                ]
            }
        };
        let mut valid = if let Some(pool) = &self.pool {
            pool.install(|| {
                self.locals
                    .par_iter_mut()
                    .zip(&first.b.blocks)
                    .zip(&second.b.blocks)
                    .zip(&mut first.x.blocks)
                    .zip(&mut second.x.blocks)
                    .map(solve)
                    .reduce(|| [true, true], |a, b| [a[0] & b[0], a[1] & b[1]])
            })
        } else {
            self.locals
                .iter_mut()
                .zip(&first.b.blocks)
                .zip(&second.b.blocks)
                .zip(&mut first.x.blocks)
                .zip(&mut second.x.blocks)
                .map(solve)
                .fold([true, true], |a, b| [a[0] & b[0], a[1] & b[1]])
        };
        valid[0] &= input_ok[0];
        valid[1] &= input_ok[1];

        let border = self.border_matrix.n;
        for c in 0..2 {
            let (source, target) = if c == 0 {
                (&first.b.border, &mut self.border_panel_rhs[..border])
            } else {
                (
                    &second.b.border,
                    &mut self.border_panel_rhs[border..2 * border],
                )
            };
            if valid[c] {
                for (dst, &value) in target.iter_mut().zip(source) {
                    // Shared boundary RHS is one global contribution.
                    *dst = if self.collective.rank() == 0 {
                        -value
                    } else {
                        T::zero()
                    };
                }
                for (local, x) in self.locals.iter().zip(if c == 0 {
                    &first.x.blocks
                } else {
                    &second.x.blocks
                }) {
                    local.coupling_product(
                        self.pool.as_ref(),
                        false,
                        target,
                        &x[..local.n],
                        T::one(),
                        T::one(),
                    );
                }
            } else {
                // All ranks still enter the same reduction.  A NaN payload
                // makes the resulting lane fail the finite consensus rather
                // than allowing the valid ranks to continue alone.
                target.fill(T::nan());
            }
            valid[c] &=
                self.collective.reduce_sum_in_place(200 + c, target).is_ok() && target.is_finite();
        }

        for (c, value) in valid.iter_mut().enumerate() {
            *value = self.collective.all_true(210 + c, *value).unwrap_or(false);
        }

        if valid[0] && valid[1] {
            let panel_ok = self.border_factor.solve_factor_panel(
                &self.border_panel_rhs,
                &mut self.pair_border,
                2,
            );
            if !panel_ok {
                valid[0] = self.border_factor.solve_factor_panel(
                    &self.border_panel_rhs[..border],
                    &mut self.pair_border[..border],
                    1,
                );
                valid[1] = self.border_factor.solve_factor_panel(
                    &self.border_panel_rhs[border..2 * border],
                    &mut self.pair_border[border..2 * border],
                    1,
                );
            }
        } else {
            for c in 0..2 {
                if valid[c] {
                    valid[c] = if c == 0 {
                        self.border_factor.solve_factor_panel(
                            &self.border_panel_rhs[..border],
                            &mut self.pair_border[..border],
                            1,
                        )
                    } else {
                        self.border_factor.solve_factor_panel(
                            &self.border_panel_rhs[border..2 * border],
                            &mut self.pair_border[border..2 * border],
                            1,
                        )
                    };
                }
            }
        }

        for (c, value) in valid.iter_mut().enumerate() {
            *value = self.collective.all_true(215 + c, *value).unwrap_or(false);
        }

        for c in 0..2 {
            if !valid[c] {
                continue;
            }
            let (point, border_point) = if c == 0 {
                (&mut first.x, &self.pair_border[..border])
            } else {
                (&mut second.x, &self.pair_border[border..2 * border])
            };
            point.border.copy_from_slice(border_point);
            for (local, x) in self.locals.iter().zip(&mut point.blocks) {
                let d = x.len();
                for (i, value) in x.iter_mut().enumerate() {
                    *value -= T::dot_fma(
                        (0..border).map(|j| (&local.response[j * d + i], &point.border[j])),
                    );
                }
            }
            valid[c] &= point.norm().is_finite();
        }
        for (c, value) in valid.iter_mut().enumerate() {
            *value = self.collective.all_true(220 + c, *value).unwrap_or(false);
        }
        valid
    }

    fn recover_original_work(
        &mut self,
        out: &mut Point<T>,
        rhs: &Point<T>,
        inner: &Point<T>,
        cache_lane: Option<usize>,
    ) -> bool {
        out.border.copy_from_slice(&inner.border);
        let border = out.border.as_slice();
        let recover =
            |(local, x, rhs, reduced): (&mut LocalKkt<T>, &mut Vec<T>, &Vec<T>, &Vec<T>)| {
                if let Some(lane) = cache_lane {
                    if !local
                        .kernel
                        .restore_sampled_rhs_cache(&local.rhs_cache[lane])
                    {
                        return false;
                    }
                }
                let success = local.kernel.recover_interior_rhs(x, rhs, reduced);
                for (j, &r) in local.border_rows.iter().enumerate() {
                    x[local.n + r] = border[j];
                }
                success
            };
        all_blocks!(
            self.pool.as_ref(),
            (
                &mut self.locals,
                &mut out.blocks,
                &rhs.blocks,
                &inner.blocks
            ),
            recover
        )
    }

    pub(crate) fn cost_samples(&self) -> Option<Vec<(f64, f64)>> {
        self.record_costs.then(|| {
            self.locals
                .iter()
                .map(|local| (local.local_assemble_ns, local.factor_response_ns))
                .collect()
        })
    }

    fn solve_reduced_raw(&mut self, out: &mut Point<T>, rhs: &Point<T>) -> bool {
        let solve = |(local, b, x): (&mut LocalKkt<T>, &Vec<T>, &mut Vec<T>)| {
            local.kernel.solve_interior_panel(b, x, 1)
        };
        let valid = all_blocks!(
            self.pool.as_ref(),
            (&mut self.locals, &rhs.blocks, &mut out.blocks),
            solve
        );
        let valid = self.collective.all_true(239, valid).unwrap_or(false);
        if !valid {
            return false;
        }
        for (out, &b) in self.border_rhs.iter_mut().zip(&rhs.border) {
            // Each local response contributes, but the shared RHS is counted once.
            *out = if self.collective.rank() == 0 {
                -b
            } else {
                T::zero()
            };
        }
        for (local, x) in self.locals.iter().zip(&out.blocks) {
            local.coupling_product(
                self.pool.as_ref(),
                false,
                &mut self.border_rhs,
                &x[..local.n],
                T::one(),
                T::one(),
            );
        }
        // The reduced border RHS is identical on every rank, so its
        // finiteness needs no agreement of its own.
        if self
            .collective
            .reduce_sum_in_place(240, &mut self.border_rhs)
            .is_err()
            || !self.border_rhs.is_finite()
        {
            return false;
        }
        if !self
            .border_factor
            .solve_factor_panel(&self.border_rhs, &mut out.border, 1)
        {
            return false;
        }
        let pool = self.pool.as_ref();
        let correct = |(local, x): (&LocalKkt<T>, &mut Vec<T>)| {
            let d = x.len();
            border_rows(pool, x, out.border.len(), |i, v| {
                *v -= T::dot_fma(
                    (0..out.border.len()).map(|j| (&local.response[j * d + i], &out.border[j])),
                );
            });
        };
        for_blocks!(self.pool.as_ref(), (&self.locals, &mut out.blocks), correct);
        self.collective
            .all_true(242, out.norm().is_finite())
            .unwrap_or(false)
    }

    fn solve_original(
        &mut self,
        out: &mut Point<T>,
        rhs: &Point<T>,
        settings: &CoreSettings<T>,
    ) -> bool {
        let mut work = self.inner.take().unwrap();
        let prepare = |(local, b, rhs): (&mut LocalKkt<T>, &mut Vec<T>, &Vec<T>)| {
            local.kernel.prepare_interior_rhs(rhs, b);
        };
        for_blocks!(
            self.pool.as_ref(),
            (&mut self.locals, &mut work.b.blocks, &rhs.blocks),
            prepare
        );
        work.b.border.copy_from_slice(&rhs.border);
        // Both outcomes are already agreed: every exit of the reduced solve
        // and of the refinement follows a collective or a replicated value.
        let mut success = self.solve_reduced_raw(&mut work.x, &work.b)
            && refine(
                &mut OwnedRefinement {
                    solver: self,
                    work: &mut work,
                    reduced: true,
                    basis: Vec::new(),
                    directions: Vec::new(),
                },
                settings,
            );
        if success {
            success = self.recover_original_work(out, rhs, &work.x, None);
            success = self.collective.all_true(291, success).unwrap_or(false);
        }
        self.inner = Some(work);
        success
    }

    /// Reduced residual fused with the next correction's local work: each
    /// owner also solves its interior with the new residual and forms its
    /// border contribution, and one all-gather carries the border residual,
    /// the border right-hand side and the residual norm. A refinement pass
    /// then waits once instead of at the residual and again inside the
    /// correction solve. The border residual and norm keep the reduction
    /// order of [`Self::residual`]; the correction right-hand side is
    /// `-e_border + Σ_rank Σ_owner C y` in rank order.
    fn residual_reduced_fused(
        &mut self,
        out: &mut Point<T>,
        rhs: &Point<T>,
        point: &Point<T>,
    ) -> T {
        let border = out.border.len();
        out.border.fill(T::zero());
        if self.fused_interior.len() != self.locals.len() {
            self.fused_interior = out
                .blocks
                .iter()
                .map(|b| vec![T::zero(); b.len()])
                .collect();
        }
        let pool = self.pool.as_ref();
        let local = |(local, e, b, x, y): (
            &mut LocalKkt<T>,
            &mut Vec<T>,
            &Vec<T>,
            &Vec<T>,
            &mut Vec<T>,
        )| {
            local.kernel.interior_residual(e, b, x);
            local.coupling_product(
                pool,
                true,
                &mut e[..local.n],
                &point.border,
                -T::one(),
                T::one(),
            );
            y.resize(e.len(), T::zero());
            e.is_finite() && local.kernel.solve_interior_panel(e, y, 1)
        };
        let solved = all_blocks!(
            self.pool.as_ref(),
            (
                &mut self.locals,
                &mut out.blocks,
                &rhs.blocks,
                &point.blocks,
                &mut self.fused_interior
            ),
            local
        );
        // [border residual part | border rhs contribution | local norm | solved]
        let mut message = vec![T::zero(); 2 * border + 2];
        let mut local_norm = T::zero();
        for ((local, e), (x, y)) in self
            .locals
            .iter()
            .zip(&out.blocks)
            .zip(point.blocks.iter().zip(&self.fused_interior))
        {
            local.coupling_product(
                self.pool.as_ref(),
                false,
                &mut message[..border],
                &x[..local.n],
                -T::one(),
                T::one(),
            );
            local.coupling_product(
                self.pool.as_ref(),
                false,
                &mut message[border..2 * border],
                &y[..local.n],
                T::one(),
                T::one(),
            );
            local_norm = if e.is_finite() {
                T::max(local_norm, e.norm_inf())
            } else {
                T::infinity()
            };
        }
        message[2 * border] = local_norm;
        message[2 * border + 1] = if solved { T::one() } else { T::zero() };
        let Ok(all) = self.collective.all_gather(323, &message) else {
            self.fused_ready = false;
            return T::infinity();
        };
        let width = message.len();
        let ranks = all.len() / width.max(1);
        // Border residual: rank-order sum, as `reduce_sum`, plus the RHS.
        out.border.copy_from_slice(&all[..border]);
        for r in 1..ranks {
            for (o, &v) in out
                .border
                .iter_mut()
                .zip(&all[r * width..r * width + border])
            {
                *o += v;
            }
        }
        for (e, &b) in out.border.iter_mut().zip(&rhs.border) {
            *e += b;
        }
        let mut norm = if out.border.is_finite() {
            out.border.norm_inf()
        } else {
            T::infinity()
        };
        let mut all_solved = true;
        for r in 0..ranks {
            let n = all[r * width + 2 * border];
            norm = if n.is_nan() || norm.is_nan() {
                T::infinity()
            } else {
                T::max(norm, n)
            };
            all_solved &= all[r * width + 2 * border + 1] == T::one();
        }
        self.fused_border_rhs.clear();
        self.fused_border_rhs.extend(out.border.iter().map(|&e| -e));
        for r in 0..ranks {
            for (o, &v) in self
                .fused_border_rhs
                .iter_mut()
                .zip(&all[r * width + border..r * width + 2 * border])
            {
                *o += v;
            }
        }
        self.fused_ready = all_solved && norm.is_finite() && self.fused_border_rhs.is_finite();
        norm
    }

    /// Finish the correction prepared by [`Self::residual_reduced_fused`].
    fn solve_reduced_fused(&mut self, out: &mut Point<T>) -> bool {
        if !std::mem::take(&mut self.fused_ready) {
            return false;
        }
        if !self
            .border_factor
            .solve_factor_panel(&self.fused_border_rhs, &mut out.border, 1)
        {
            return false;
        }
        let border = &out.border;
        let pool = self.pool.as_ref();
        let correct = |(local, x, y): (&LocalKkt<T>, &mut Vec<T>, &Vec<T>)| {
            let d = x.len();
            border_rows(pool, x, border.len(), |i, v| {
                *v = y[i]
                    - T::dot_fma(
                        (0..border.len()).map(|j| (&local.response[j * d + i], &border[j])),
                    );
            });
        };
        for_blocks!(
            self.pool.as_ref(),
            (&self.locals, &mut out.blocks, &self.fused_interior),
            correct
        );
        out.norm().is_finite()
    }

    fn residual(
        &mut self,
        out: &mut Point<T>,
        rhs: &Point<T>,
        point: &Point<T>,
        reduced: bool,
    ) -> T {
        out.border.fill(T::zero());
        let pool = self.pool.as_ref();
        let residual = |(local, e, b, x): (&mut LocalKkt<T>, &mut Vec<T>, &Vec<T>, &Vec<T>)| {
            if reduced {
                local.kernel.interior_residual(e, b, x);
                local.coupling_product(
                    pool,
                    true,
                    &mut e[..local.n],
                    &point.border,
                    -T::one(),
                    T::one(),
                );
            } else {
                // Original b contains zero in every local equality row. Add
                // the global affine term only after summing their operators.
                local.kernel.original_residual(e, b, x);
            }
        };
        for_blocks!(
            self.pool.as_ref(),
            (
                &mut self.locals,
                &mut out.blocks,
                &rhs.blocks,
                &point.blocks
            ),
            residual
        );
        for ((local, e), x) in self.locals.iter().zip(&mut out.blocks).zip(&point.blocks) {
            if reduced {
                local.coupling_product(
                    self.pool.as_ref(),
                    false,
                    &mut out.border,
                    &x[..local.n],
                    -T::one(),
                    T::one(),
                );
            } else {
                for (j, &r) in local.border_rows.iter().enumerate() {
                    out.border[j] += e[local.n + r];
                    e[local.n + r] = T::zero();
                }
            }
        }
        if self
            .collective
            .reduce_sum_in_place(320, &mut out.border)
            .is_err()
        {
            return T::infinity();
        }
        for (e, &b) in out.border.iter_mut().zip(&rhs.border) {
            *e += b;
        }
        // `Point::norm` is infinite on any non-finite entry, so the global
        // maximum also carries finiteness.
        self.collective
            .reduce_max(322, &[out.norm()])
            .ok()
            .and_then(|values| values.into_iter().next())
            .unwrap_or(T::infinity())
    }
}

/// Entries of a CSC matrix grouped by row, each row in column order: the
/// grouping `_csc_axpby_N` builds per call, kept so one output row of the
/// product can be formed alone with the same terms in the same order.
struct CouplingRows {
    ptr: Vec<usize>,
    /// `(position in nzval, column)` per entry.
    entries: Vec<(usize, usize)>,
    /// Whether the CSC product takes its exact wide-precision path.
    wide: bool,
}

impl CouplingRows {
    fn new<T: FloatT>(a: &CscMatrix<T>) -> Self {
        let mut ptr = vec![0usize; a.m + 1];
        for &r in &a.rowval {
            ptr[r + 1] += 1;
        }
        for i in 0..a.m {
            ptr[i + 1] += ptr[i];
        }
        let mut next = ptr.clone();
        let mut entries = vec![(0usize, 0usize); a.nnz()];
        for j in 0..a.n {
            for k in a.colptr[j]..a.colptr[j + 1] {
                let r = a.rowval[k];
                entries[next[r]] = (k, j);
                next[r] += 1;
            }
        }
        Self {
            ptr,
            entries,
            wide: T::precision_bits() > 64 && a.nnz() >= 4 * a.m,
        }
    }

    /// Row `i` of `a·x`, bitwise equal to `a.gemv(y, x, 1, 0)` at `y[i]`.
    fn value<T: FloatT>(&self, a: &CscMatrix<T>, i: usize, x: &[T]) -> T {
        let row = &self.entries[self.ptr[i]..self.ptr[i + 1]];
        if self.wide {
            let terms = row.iter().map(|&(k, j)| (&a.nzval[k], &x[j]));
            crate::algebra::wide_output(T::zero(), terms, T::one(), T::one())
        } else {
            let mut y = T::zero();
            for &(k, j) in row {
                y += a.nzval[k] * x[j];
            }
            y
        }
    }
}

fn select_rows<T: FloatT>(
    matrix: &CscMatrix<T>,
    rows: &[usize],
    columns: &[usize],
) -> CscMatrix<T> {
    let mut ptr = vec![0];
    let mut indices = Vec::new();
    let mut values = Vec::new();
    for &col in columns {
        for p in matrix.colptr[col]..matrix.colptr[col + 1] {
            if let Ok(row) = rows.binary_search(&matrix.rowval[p]) {
                indices.push(row);
                values.push(matrix.nzval[p]);
            }
        }
        ptr.push(indices.len());
    }
    CscMatrix::new(rows.len(), columns.len(), ptr, indices, values)
}

struct OwnedRefinement<'a, T: FloatT> {
    solver: &'a mut OwnedKkt<T>,
    work: &'a mut Work<T>,
    reduced: bool,
    /// GMRES bases V and Z = M⁻¹V (empty unless GMRES-IR runs).
    basis: Vec<Point<T>>,
    directions: Vec<Point<T>>,
}
impl<T: FloatT> Refinement<T> for OwnedRefinement<'_, T> {
    fn all_succeeded(&self, value: bool) -> bool {
        self.solver.collective.all_true(700, value).unwrap_or(false)
    }
    fn decision_agrees(&self, value: u32) -> bool {
        self.solver
            .collective
            .agree_u32(701, value)
            .unwrap_or(false)
    }
    fn agree_pair(&self, ok: bool, decision: u32) -> bool {
        self.solver
            .collective
            .agree_u32(702, decision.saturating_mul(2) | u32::from(!ok))
            .unwrap_or(false)
            && ok
    }
    fn rhs_norm(&self) -> T {
        self.solver
            .collective
            .reduce_max(330, &[self.work.b.norm()])
            .ok()
            .and_then(|values| values.into_iter().next())
            .unwrap_or(T::infinity())
    }
    fn residual(&mut self, candidate: bool, _reuse: bool) -> T {
        if self.reduced {
            return self.solver.residual_reduced_fused(
                &mut self.work.error,
                &self.work.b,
                if candidate {
                    &self.work.candidate
                } else {
                    &self.work.x
                },
            );
        }
        self.solver.residual(
            &mut self.work.error,
            &self.work.b,
            if candidate {
                &self.work.candidate
            } else {
                &self.work.x
            },
            self.reduced,
        )
    }
    fn solve_correction(&mut self, settings: &CoreSettings<T>) -> bool {
        self.solver.refinements += 1;
        let solved = if self.reduced {
            self.solver.solve_reduced_fused(&mut self.work.candidate)
        } else {
            self.solver
                .solve_original(&mut self.work.candidate, &self.work.error, settings)
        };
        if !solved {
            // The next residual (a global maximum) is then non-finite on
            // every rank, which fails refinement in the residual's agreement.
            self.work.candidate.fill(T::nan());
        }
        solved
    }
    fn defers_solve_agreement(&self) -> bool {
        true
    }
    fn add_correction(&mut self) {
        self.work.candidate.add(&self.work.x);
    }
    fn accept_candidate(&mut self) {
        std::mem::swap(&mut self.work.x, &mut self.work.candidate);
    }
    fn restore_product(&mut self) {
        if !self.reduced {
            for (local, x) in self.solver.locals.iter_mut().zip(&self.work.x.blocks) {
                local.kernel.restore_scaled_product(x);
            }
        }
    }
    // Blocks are rank-owned and summed over ranks; the border is replicated
    // (identical on every rank after each residual) and counted once.
    // Only the reduced level: an original-level correction is a complete
    // refined reduced solve (see the condensed kernel).
    fn gmres_supported(&self) -> bool {
        self.reduced
    }
    fn gmres_reset(&mut self) {
        self.basis.clear();
        self.directions.clear();
    }
    fn gmres_push_basis(&mut self, scale: T) {
        let mut v = self.work.error.clone_point();
        v.scale(scale);
        self.basis.push(v);
    }
    fn gmres_precondition(&mut self, _settings: &CoreSettings<T>) -> bool {
        debug_assert!(self.reduced);
        let v = self.basis.last().unwrap();
        let mut z = v.zeros_like();
        let ok = self.solver.solve_reduced_raw(&mut z, v);
        self.directions.push(z);
        ok
    }
    fn gmres_operator(&mut self) -> bool {
        let z = self.directions.last().unwrap();
        let zero = z.zeros_like();
        let norm = self.solver.residual(&mut self.work.error, &zero, z, true);
        self.work.error.negate();
        norm.is_finite()
    }
    fn gmres_dots(&mut self) -> Vec<T> {
        let e = &self.work.error;
        let mut local: Vec<T> = self.basis.iter().map(|v| e.block_dot(v)).collect();
        if self
            .solver
            .collective
            .reduce_sum_in_place(702, &mut local)
            .is_err()
        {
            return vec![T::nan(); self.basis.len()];
        }
        for (d, v) in local.iter_mut().zip(&self.basis) {
            *d += e.border.dot(&v.border);
        }
        local
    }
    fn gmres_subtract(&mut self, c: &[T]) {
        for (v, &a) in self.basis.iter().zip(c) {
            self.work.error.axpy(-a, v);
        }
    }
    fn gmres_norm2(&mut self) -> T {
        let e = &self.work.error;
        let mut local = [e.block_dot(e)];
        if self
            .solver
            .collective
            .reduce_sum_in_place(703, &mut local)
            .is_err()
        {
            return T::nan();
        }
        T::sqrt(local[0] + e.border.dot(&e.border))
    }
    fn gmres_floor(&self) -> Option<T> {
        self.solver.gmres_floor
    }
    fn set_gmres_floor(&mut self, floor: T) {
        self.solver.gmres_floor = Some(floor);
    }
    fn gmres_candidate(&mut self, y: &[T]) {
        let candidate = &mut self.work.candidate;
        candidate.copy_from_point(&self.work.x);
        for (z, &yi) in self.directions.iter().zip(y) {
            candidate.axpy(yi, z);
        }
    }
}

#[cfg(test)]
#[path = "tests/kkt.rs"]
mod tests;

#[cfg(test)]
#[path = "tests/border_assembly.rs"]
mod border_assembly_tests;
