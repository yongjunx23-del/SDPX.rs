//! Condensation of PSD/orthant rows beneath the existing embedding solver.
//!
//! Adapted from SDPX.jl's kktsolver_condensed.jl and condensed_schur.jl.
//! The retained system uses Clarabel's existing LDL shifts and refinement.
//! No equality rows are inverted or removed, and residual refinement below
//! uses the original augmented operator, not the regularized Schur matrix.
#![allow(non_snake_case)]

use super::{direct::DirectLDLKKTSolver, HasLinearSolverInfo, KKTSolver, LinearSolverInfo};
use crate::algebra::*;
use crate::solver::core::{cones::*, CoreSettings};
use crate::solver::{SampledOperator, SampledSchurWorkspace, SampledWorkspace};
use rayon::prelude::*;
use std::collections::{BTreeSet, HashMap};
use std::hash::{Hash, Hasher};
use std::ops::Range;
use std::sync::Arc;

/// Conservative storage selection, independent of numerical coefficients.
pub(crate) fn prefer_condensed<T: FloatT>(
    P: &CscMatrix<T>,
    A: &CscMatrix<T>,
    cones: &CompositeCone<T>,
    _settings: &CoreSettings<T>,
) -> bool {
    let mut psd = 0usize;
    let mut retained = 0usize;
    let mut retained_aux = 0usize;
    let mut augmented_cells = 0u128;
    let mut factor_cells = 0u128;
    for cone in cones.iter() {
        match cone {
            SupportedCone::PSDTriangleCone(c) => {
                psd = psd.saturating_add(c.numel());
                let p = c.numel() as u128;
                augmented_cells += p * (p + 1) / 2;
                factor_cells += 7 * (c.n as u128).pow(2);
            }
            SupportedCone::NonnegativeCone(_) => {}
            _ => {
                retained = retained.saturating_add(cone.numel());
                // Match the existing SOC/GenPower sparse expansion sizes.
                if cone.is_sparse_expandable() {
                    retained_aux = retained_aux.saturating_add(match cone {
                        SupportedCone::SecondOrderCone(_) => 2,
                        SupportedCone::GenPowerCone(_) => 3,
                        _ => 0,
                    });
                }
            }
        }
    }
    let reduced = A.n.saturating_add(retained);
    if A.n == 0 || psd < 256 || reduced > psd / 4 {
        return false;
    }

    // Bound the same structural union built by new(): P plus diagonals,
    // each PSD's participating-column clique, and each independent NN row's
    // clique. Count stored entries, including zeros. Distinct cone cliques
    // may overlap, so cap their sum by the dense upper-triangular count.
    // This scan is O(nnz(A)+nnz(P)+m); it never builds a dense trial Schur.
    let count = cones.len();
    let mut row_owner = vec![usize::MAX; A.m];
    let mut orthant_degrees = Vec::new();
    for (ci, (cone, rows)) in cones.iter().zip(&cones.rng_cones).enumerate() {
        match cone {
            SupportedCone::PSDTriangleCone(_) => row_owner[rows.clone()].fill(ci),
            SupportedCone::NonnegativeCone(_) => {
                let start = orthant_degrees.len();
                orthant_degrees.resize(start + rows.len(), 0usize);
                for (offset, row) in rows.clone().enumerate() {
                    row_owner[row] = count + start + offset;
                }
            }
            _ => {}
        }
    }
    let mut psd_degrees = vec![0usize; count];
    let mut last_column = vec![usize::MAX; count];
    for col in 0..A.n {
        for p in A.colptr[col]..A.colptr[col + 1] {
            let owner = row_owner[A.rowval[p]];
            if owner == usize::MAX {
                continue;
            } else if owner < count {
                if last_column[owner] != col {
                    psd_degrees[owner] += 1;
                    last_column[owner] = col;
                }
            } else {
                orthant_degrees[owner - count] += 1;
            }
        }
    }
    let n = A.n as u128;
    let dense_upper = n * (n + 1) / 2;
    let mut schur_upper = n;
    for col in 0..P.n {
        for p in P.colptr[col]..P.colptr[col + 1] {
            if P.rowval[p] <= col {
                schur_upper += 1;
            }
        }
    }
    for degree in psd_degrees.into_iter().chain(orthant_degrees) {
        let d = degree as u128;
        schur_upper = (schur_upper + d * (d + 1) / 2).min(dense_upper);
    }
    schur_upper = schur_upper.min(dense_upper);

    // Keep the existing six-copy factor/workspace allowance, now applied to
    // a symmetric structural storage estimate. Charge a fully dense retained
    // border and its factor block, including sparse-cone auxiliary variables.
    // At a dense primal union this reduces to the previous dense estimate.
    // This is a selection estimate, not a guarantee on symbolic factor fill.
    let r = retained as u128 + retained_aux as u128;
    let reduced_cells = 2 * schur_upper - n + 2 * n * r + r * r;
    6 * reduced_cells + factor_cells < 3 * augmented_cells
}

#[derive(Clone, Copy)]
struct Entry {
    position: usize,
    i: usize,
    j: usize,
}
struct Column {
    index: usize,
    entries: Vec<Entry>,
    sparse: bool,
    schur_positions: Vec<usize>,
}

struct PsdBlock<T> {
    // Fixed-precision immutable constant shared by serial and parallel assembly.
    sqrt2: T,
    R: Matrix<T>,
    Rinv: Matrix<T>,
    Ginv: Matrix<T>,
    mat1: Matrix<T>,
    mat2: Matrix<T>,
    mat3: Matrix<T>,
    mat2c: Matrix<T>,
    mat3c: Matrix<T>,
    vector: Vec<T>,
    columns: Vec<Column>,
    schur_values: Vec<T>,
    sampled: Option<SampledPsd<T>>,
    sparse_column_lanes: Vec<usize>,
    dense_indices: Vec<usize>,
    column_groups: Vec<Vec<usize>>,
    dense_representatives: Vec<usize>,
    dense_column_map: Vec<usize>,
    dense_vectors: Vec<T>,
    dense_row_first: Vec<usize>,
    dense_row_offsets: Vec<usize>,
    dense_acc: Vec<T>,
    // Per-column plan for W = Ginv*A: (output col q, source row p, entry)
    // sorted so each output column accumulates in ascending p, matching the
    // GEMM inner-product order bitwise.
    axpy_plans: Vec<Vec<(u32, u32, u32)>>,
}

struct SampledPsd<T> {
    operator: Arc<SampledOperator<T>>,
    block: usize,
    work: SampledSchurWorkspace<T>,
    pair_lanes: Vec<usize>,
}

// Immutable scaling snapshots with private arithmetic scratch. These are
// operator data, not another cone/solver state machine. Retained cone Hessians
// use the same formulas as their existing mul_Hs implementations.
enum Scaling<T> {
    Psd(PsdBlock<T>),
    Orthant {
        w: Vec<T>,
        rows: Vec<Vec<(usize, usize)>>,
        scaled_row: Vec<T>,
    },
    Zero,
    Soc {
        w: Vec<T>,
        eta: T,
    },
    Dense3([T; 6]),
    GenPower {
        p: Vec<T>,
        q: Vec<T>,
        r: Vec<T>,
        d1: Vec<T>,
        d2: T,
        mu: T,
    },
}
struct Block<T> {
    rows: Range<usize>,
    scaling: Scaling<T>,
}

pub(crate) struct CondensedKKTSolver<T: FloatT> {
    n: usize,
    P: CscMatrix<T>,
    A: CscMatrix<T>,
    blocks: Vec<Block<T>>,
    retained_indices: Vec<usize>,
    retained_rows: Vec<usize>,
    retained_positions: Vec<usize>,
    retained_A: CscMatrix<T>,
    schur: CscMatrix<T>,
    reduced: DirectLDLKKTSolver<T>,
    b: Vec<T>,
    x: Vec<T>,
    error: Vec<T>,
    candidate: Vec<T>,
    workx: Vec<T>,
    workz: Vec<T>,
    workh: Vec<T>,
    retained_rhs: Vec<T>,
    pool: Option<Arc<rayon::ThreadPool>>,
    parallel_assembly: bool,
    plan_threads: usize,
    scaling_lanes: Vec<usize>,
    scaling_tiles: usize,
    inner_schur: bool,
    inner_sampled: Option<usize>,
    sampled: Option<(Arc<SampledOperator<T>>, SampledWorkspace<T>)>,
    sparse_products: Option<crate::algebra::sparse_parallel::SparseParallel>,
    counters: crate::solver::core::kktsolvers::SolveCounters,
}

impl<T: FloatT> CondensedKKTSolver<T> {
    pub(crate) fn prefer_condensed(
        P: &CscMatrix<T>,
        A: &CscMatrix<T>,
        cones: &CompositeCone<T>,
        settings: &CoreSettings<T>,
    ) -> bool {
        prefer_condensed(P, A, cones, settings)
    }

    pub(crate) fn new(
        P: &CscMatrix<T>,
        A: &CscMatrix<T>,
        types: &[SupportedConeT<T>],
        cones: &CompositeCone<T>,
        settings: &CoreSettings<T>,
    ) -> Self {
        let (n, m) = (A.n, A.m);
        assert!(n > 0, "condensed KKT requires primal variables");
        assert_eq!(P.size(), (n, n));
        assert_eq!(cones.numel(), m);
        assert_eq!(types.len(), cones.len());
        let mut retained_indices = Vec::new();
        let mut retained_rows = Vec::new();
        let mut retained_types = Vec::new();
        let mut rowmap = vec![usize::MAX; m];
        let mut blocks = Vec::with_capacity(cones.len());
        // Route every row to its owning eliminated block once, then fill all
        // block-local index structures from a single pass over A. Scanning A
        // per block is O(blocks * nnz), which dominates setup on the
        // many-block shapes this backend exists for (bound-heavy models and
        // chordal/compact rewrites with hundreds of small cliques).
        let mut row_psd = vec![u32::MAX; m];
        let mut row_orthant = vec![u32::MAX; m];
        let mut psd_starts: Vec<usize> = Vec::new();
        let mut psd_numels: Vec<usize> = Vec::new();
        let mut psd_coordinates: Vec<Vec<(usize, usize)>> = Vec::new();
        let mut psd_columns: Vec<Vec<Column>> = Vec::new();
        let mut orthant_starts: Vec<usize> = Vec::new();
        let mut orthant_entries: Vec<Vec<Vec<(usize, usize)>>> = Vec::new();
        // Pass 1: classify cones and publish row ownership.
        for (ci, (cone, rows)) in cones.iter().zip(&cones.rng_cones).enumerate() {
            match cone {
                SupportedCone::PSDTriangleCone(c) => {
                    let psd = psd_starts.len();
                    row_psd[rows.clone()].fill(psd as u32);
                    psd_starts.push(rows.start);
                    psd_numels.push(rows.len());
                    let mut coordinates = Vec::with_capacity(rows.len());
                    for j in 0..c.n {
                        for i in 0..=j {
                            coordinates.push((i, j));
                        }
                    }
                    psd_coordinates.push(coordinates);
                    psd_columns.push(Vec::new());
                }
                SupportedCone::NonnegativeCone(_) => {
                    let orthant = orthant_starts.len();
                    row_orthant[rows.clone()].fill(orthant as u32);
                    orthant_starts.push(rows.start);
                    orthant_entries.push(vec![Vec::new(); rows.len()]);
                }
                _ => {
                    retained_indices.push(ci);
                    retained_types.push(types[ci].clone());
                    for row in rows.clone() {
                        rowmap[row] = retained_rows.len();
                        retained_rows.push(row);
                    }
                }
            }
        }
        // Pass 2: one scan of A fills every eliminated block's local indices.
        let mut psd_last_column = vec![usize::MAX; psd_columns.len()];
        for col in 0..n {
            for p in A.colptr[col]..A.colptr[col + 1] {
                let row = A.rowval[p];
                let psd = row_psd[row];
                if psd != u32::MAX {
                    let psd = psd as usize;
                    if psd_last_column[psd] != col {
                        psd_columns[psd].push(Column {
                            index: col,
                            entries: Vec::new(),
                            sparse: false,
                            schur_positions: Vec::new(),
                        });
                        psd_last_column[psd] = col;
                    }
                    let (i, j) = psd_coordinates[psd][row - psd_starts[psd]];
                    psd_columns[psd]
                        .last_mut()
                        .expect("column pushed above")
                        .entries
                        .push(Entry { position: p, i, j });
                    continue;
                }
                let orthant = row_orthant[row];
                if orthant != u32::MAX {
                    let orthant = orthant as usize;
                    orthant_entries[orthant][row - orthant_starts[orthant]].push((col, p));
                }
            }
        }
        // Pass 3: materialize the per-cone scaling structures.
        let mut psd_taken = psd_columns.into_iter().zip(psd_numels);
        let mut orthant_taken = orthant_entries.into_iter();
        for (ci, cone) in cones.iter().enumerate() {
            let scaling = match cone {
                SupportedCone::PSDTriangleCone(c) => {
                    let (columns, numel) = psd_taken.next().expect("one column set per PSD cone");
                    Scaling::Psd(PsdBlock::from_columns(c.n, numel, columns))
                }
                SupportedCone::NonnegativeCone(c) => {
                    let entries = orthant_taken.next().expect("one entry set per orthant");
                    Scaling::Orthant {
                        w: c.w.clone(),
                        scaled_row: vec![
                            T::zero();
                            if T::precision_bits() > 53 {
                                entries.iter().map(Vec::len).max().unwrap_or(0)
                            } else {
                                0
                            }
                        ],
                        rows: entries,
                    }
                }
                SupportedCone::ZeroCone(_) => Scaling::Zero,
                SupportedCone::SecondOrderCone(c) => Scaling::Soc {
                    w: c.w.clone(),
                    eta: c.η,
                },
                SupportedCone::ExponentialCone(_) | SupportedCone::PowerCone(_) => {
                    Scaling::Dense3([T::zero(); 6])
                }
                SupportedCone::GenPowerCone(c) => Scaling::GenPower {
                    p: c.data.p.clone(),
                    q: c.data.q.clone(),
                    r: c.data.r.clone(),
                    d1: c.data.d1.clone(),
                    d2: c.data.d2,
                    mu: c.data.μ,
                },
            };
            blocks.push(Block {
                rows: cones.rng_cones[ci].clone(),
                scaling,
            });
        }
        let mut colptr = vec![0];
        let mut rowval = Vec::new();
        let mut nzval = Vec::new();
        let mut retained_positions = Vec::new();
        for col in 0..n {
            for p in A.colptr[col]..A.colptr[col + 1] {
                let row = rowmap[A.rowval[p]];
                if row != usize::MAX {
                    rowval.push(row);
                    nzval.push(A.nzval[p]);
                    retained_positions.push(p);
                }
            }
            colptr.push(nzval.len());
        }
        let nr = retained_rows.len();
        let retained_A = CscMatrix::new(nr, n, colptr, rowval, nzval);
        // Exact structural union. Each PSD block couples its active columns;
        // each orthant coordinate couples only columns touching that row.
        // Global equality rows do not turn independent primal blocks dense.
        let mut pattern: Vec<BTreeSet<usize>> = (0..n).map(|j| BTreeSet::from([j])).collect();
        for j in 0..n {
            for p in P.colptr[j]..P.colptr[j + 1] {
                if P.rowval[p] <= j {
                    pattern[j].insert(P.rowval[p]);
                }
            }
        }
        for block in &blocks {
            match &block.scaling {
                Scaling::Psd(p) => {
                    for (b, right) in p.columns.iter().enumerate() {
                        for left in &p.columns[..=b] {
                            pattern[right.index.max(left.index)]
                                .insert(right.index.min(left.index));
                        }
                    }
                }
                Scaling::Orthant { rows, .. } => {
                    for row in rows {
                        for (b, &(j, _)) in row.iter().enumerate() {
                            for &(i, _) in &row[..=b] {
                                pattern[j].insert(i);
                            }
                        }
                    }
                }
                _ => {}
            }
        }
        let count: usize = pattern.iter().map(BTreeSet::len).sum();
        let mut colptr = Vec::with_capacity(n + 1);
        let mut rowval = Vec::with_capacity(count);
        colptr.push(0);
        for column in pattern {
            rowval.extend(column);
            colptr.push(rowval.len());
        }
        // Retain structural zeros: numerical updates never alter this plan.
        let schur = CscMatrix::new(n, n, colptr, rowval, vec![T::zero(); count]);
        let pool = cones.thread_pool();
        let contribution_cells: u128 = blocks
            .iter()
            .map(|block| match &block.scaling {
                Scaling::Psd(p) => triangular_number(p.columns.len()) as u128,
                _ => 0,
            })
            .sum();
        // Bound extra contribution storage by two copies of the sparse Schur
        // values. Strongly overlapping cliques keep allocation-free serial
        // assembly; their scaling/operator phases can still run in parallel.
        let parallel_assembly = pool.is_some() && contribution_cells <= 2 * count as u128;
        for block in &mut blocks {
            if let Scaling::Psd(p) = &mut block.scaling {
                if parallel_assembly {
                    p.schur_values
                        .resize(triangular_number(p.columns.len()), T::zero());
                }
                let columns: Vec<usize> = p.columns.iter().map(|c| c.index).collect();
                for (b, right) in p.columns.iter_mut().enumerate() {
                    right.schur_positions = columns[..=b]
                        .iter()
                        .map(|&i| schur_position(&schur, i.min(right.index), i.max(right.index)))
                        .collect();
                }
            }
        }
        let retained_cones = CompositeCone::new(&retained_types);
        let reduced =
            DirectLDLKKTSolver::new(&schur, &retained_A, &retained_cones, nr, n, settings);
        let mut solver = Self {
            n,
            P: P.clone(),
            A: A.clone(),
            blocks,
            retained_indices,
            retained_rows,
            retained_positions,
            retained_A,
            schur,
            reduced,
            b: vec![T::zero(); n + m],
            x: vec![T::zero(); n + m],
            error: vec![T::zero(); n + m],
            candidate: vec![T::zero(); n + m],
            workx: vec![T::zero(); n],
            workz: vec![T::zero(); m],
            workh: vec![T::zero(); m],
            retained_rhs: vec![T::zero(); nr],
            pool,
            parallel_assembly,
            plan_threads: 0,
            scaling_lanes: Vec::new(),
            scaling_tiles: 1,
            inner_schur: false,
            inner_sampled: None,
            sampled: None,
            sparse_products: (crate::mpi::World::get().is_some()
                && crate::algebra::sparse_parallel::worthwhile(A))
            .then(|| crate::algebra::sparse_parallel::SparseParallel::new(A)),
            counters: Default::default(),
        };
        if let Some(plan) = &mut solver.sparse_products {
            plan.configure(A, solver.pool.clone());
        }
        solver.refresh_parallel_plan();
        solver
    }

    /// `y = alpha * op(A) * x + beta * y`, rank-sharded when an MPI world
    /// exists; identical arithmetic to the CSC gemv on every output.
    fn sparse_gemv(
        plan: Option<&crate::algebra::sparse_parallel::SparseParallel>,
        a: &CscMatrix<T>,
        transpose: bool,
        y: &mut [T],
        x: &[T],
        alpha: T,
        beta: T,
    ) {
        if let (Some(world), Some(plan)) = (crate::mpi::World::get(), plan) {
            plan.product_sharded(
                a,
                transpose,
                y,
                x,
                alpha,
                beta,
                world,
                if transpose {
                    crate::mpi::SITE_RX
                } else {
                    crate::mpi::SITE_RZ
                },
            );
        } else if transpose {
            a.t().gemv(y, x, alpha, beta);
        } else {
            a.gemv(y, x, alpha, beta);
        }
    }

    fn refresh_parallel_plan(&mut self) {
        let workers = self.pool.as_ref().map_or(1, |p| p.current_num_threads());
        if workers == self.plan_threads {
            return;
        }
        self.plan_threads = workers;
        let costs: Vec<_> = self
            .blocks
            .iter()
            .map(|block| match &block.scaling {
                Scaling::Psd(p) => 4 * (p.R.size().0 as u128).pow(3),
                _ => block.rows.len() as u128,
            })
            .collect();
        let (lanes, tiles) = scaling_dispatch(&costs, workers);
        self.scaling_lanes = lanes;
        self.scaling_tiles = tiles;
        let active_psd = self
            .blocks
            .iter()
            .filter(|b| matches!(&b.scaling, Scaling::Psd(p) if !p.columns.is_empty()))
            .count();
        let spare_workers = workers > active_psd;
        let (mut largest_sparse, mut total_work) = (0u128, 0u128);
        let mut dominant_sampled = None;
        let words = T::precision_bits().div_ceil(64) as u128;
        for block in &mut self.blocks {
            if let Scaling::Psd(p) = &mut block.scaling {
                if let Some(sampled) = &mut p.sampled {
                    let dense =
                        sampled
                            .work
                            .configure_parallel(if spare_workers { workers } else { 1 });
                    let costs: Vec<_> = (1..=p.columns.len())
                        .map(|n| 8 * n as u128 * words * words)
                        .collect();
                    let pairs: u128 = costs.iter().sum();
                    let lanes = if spare_workers {
                        workers
                            .min((pairs / 4096).min(usize::MAX as u128) as usize)
                            .max(1)
                    } else {
                        1
                    };
                    sampled.pair_lanes = weighted_lanes(&costs, lanes);
                    let work = dense + pairs;
                    total_work += work;
                    let splittable = sampled.work.has_parallel_columns()
                        || (sampled.pair_lanes.len() > 1 && !p.schur_values.is_empty());
                    if splittable && dominant_sampled.map_or(true, |(_, largest)| work > largest) {
                        dominant_sampled = Some((block.rows.start, work));
                    }
                } else {
                    let (sparse, total) =
                        p.configure_sparse_columns(if spare_workers { workers } else { 1 });
                    total_work += total;
                    if p.sparse_column_lanes.len() > 1 {
                        largest_sparse = largest_sparse.max(sparse);
                    }
                }
            } else if self.sampled.is_some() {
                total_work += block.rows.len() as u128 * words * words;
            }
        }
        // Use one parallel level. A dominant sampled block can occupy spare
        // workers internally; many small blocks retain outer scheduling.
        self.inner_sampled = dominant_sampled
            .filter(|&(_, work)| spare_workers && work >= 8192 && work * 4 >= total_work * 3)
            .map(|(row, _)| row);
        self.inner_schur = self.sampled.is_none()
            && spare_workers
            && largest_sparse > 0
            && largest_sparse * 4 >= total_work * 3;
    }

    fn assemble(&mut self) -> bool {
        // Each PSD block owns its contribution buffer. Publish in cone order
        // after joining, preserving the serial sum even for overlapping cliques.
        if self.parallel_assembly {
            let inner_schur = self.inner_schur;
            let inner_sampled = self.inner_sampled;
            let matrix_values = &self.A.nzval;
            self.pool.as_ref().unwrap().install(|| {
                let compute = |block: &mut Block<T>| {
                    if let Scaling::Psd(psd) = &mut block.scaling {
                        let mut output = std::mem::take(&mut psd.schur_values);
                        if inner_sampled == Some(block.rows.start) {
                            if let Some(sampled) = &psd.sampled {
                                if sampled.pair_lanes.len() > 1 {
                                    split_sampled_columns(
                                        &psd.columns,
                                        sampled,
                                        &mut output,
                                        &sampled.pair_lanes,
                                        psd.columns.len(),
                                    );
                                    psd.schur_values = output;
                                    return;
                                }
                            }
                        }
                        let split_columns = inner_schur
                            && psd.sampled.is_none()
                            && psd.sparse_column_lanes.len() > 1;
                        if split_columns {
                            split_sparse_columns(
                                &psd.columns,
                                &psd.Ginv,
                                matrix_values,
                                &mut output,
                                &psd.sparse_column_lanes,
                                psd.columns.len(),
                                psd.sqrt2,
                            );
                        }
                        psd.compute_schur_selected(matrix_values, split_columns, |b, a, _, v| {
                            output[triangular_number(b) + a] = v;
                        });
                        psd.schur_values = output;
                    }
                };
                // Choose one parallel level: spare workers run independent
                // columns, otherwise each outer task owns a complete block.
                if inner_schur || inner_sampled.is_some() {
                    self.blocks.iter_mut().for_each(compute);
                } else {
                    self.blocks.par_iter_mut().for_each(compute);
                }
            });
        }
        self.schur.nzval.fill(T::zero());
        for j in 0..self.n {
            for p in self.P.colptr[j]..self.P.colptr[j + 1] {
                let i = self.P.rowval[p];
                if i <= j {
                    let q = schur_position(&self.schur, i, j);
                    self.schur.nzval[q] += self.P.nzval[p];
                }
            }
        }
        for block in &mut self.blocks {
            match &mut block.scaling {
                Scaling::Psd(psd) => {
                    if self.parallel_assembly {
                        psd.scatter_schur(&mut self.schur);
                    } else {
                        psd.compute_schur(&self.A.nzval, |_, _, position, v| {
                            self.schur.nzval[position] += v;
                        });
                    }
                }
                Scaling::Orthant {
                    w,
                    rows,
                    scaled_row,
                } => {
                    if T::precision_bits() <= 53 {
                        for (row, entries) in rows.iter().enumerate() {
                            for (b, &(j, q)) in entries.iter().enumerate() {
                                let aj = self.A.nzval[q] / w[row];
                                for &(i, p) in &entries[..=b] {
                                    let ai = self.A.nzval[p] / w[row];
                                    let p = schur_position(&self.schur, i, j);
                                    let v = &mut self.schur.nzval[p];
                                    *v = ai.mul_add(aj, *v);
                                }
                            }
                        }
                        continue;
                    }
                    for (row, entries) in rows.iter().enumerate() {
                        // Recompute after every scaling/data update. Each quotient and
                        // the order of all Schur FMA updates match the uncached path.
                        for (value, &(_, position)) in scaled_row.iter_mut().zip(entries) {
                            *value = self.A.nzval[position] / w[row];
                        }
                        for (b, &(j, _)) in entries.iter().enumerate() {
                            let aj = scaled_row[b];
                            for (&(i, _), &ai) in entries[..=b].iter().zip(&scaled_row[..=b]) {
                                let p = schur_position(&self.schur, i, j);
                                let v = &mut self.schur.nzval[p];
                                *v = ai.mul_add(aj, *v);
                            }
                        }
                    }
                }
                _ => {}
            }
        }
        self.schur.nzval.is_finite()
    }

    fn solve_raw(&mut self, out: &mut [T], rhs: &[T], settings: &CoreSettings<T>) -> bool {
        let (bx, bz) = rhs.split_at(self.n);
        let (x, z) = out.split_at_mut(self.n);
        apply_scaling_pool(
            &self.pool,
            &self.scaling_lanes,
            self.scaling_tiles,
            &mut self.blocks,
            &mut self.workz,
            bz,
            true,
        );
        self.workx.copy_from_slice(bx);
        if let Some((operator, work)) = &mut self.sampled {
            operator.apply_transpose_with_pool(
                &mut self.workx,
                &self.workz,
                T::one(),
                T::one(),
                work,
                self.pool.as_ref(),
            );
        } else {
            Self::sparse_gemv(
                self.sparse_products.as_ref(),
                &self.A,
                true,
                &mut self.workx,
                &self.workz,
                T::one(),
                T::one(),
            );
        }
        for (v, &row) in self.retained_rhs.iter_mut().zip(&self.retained_rows) {
            *v = bz[row];
        }
        self.reduced.setrhs(&self.workx, &self.retained_rhs);
        if !self
            .reduced
            .solve(Some(&mut *x), Some(&mut self.retained_rhs), settings)
        {
            return false;
        }
        if let Some((operator, work)) = &mut self.sampled {
            operator.apply_with_pool(
                &mut self.workz,
                x,
                T::one(),
                T::zero(),
                work,
                self.pool.as_ref(),
            );
        } else {
            Self::sparse_gemv(
                self.sparse_products.as_ref(),
                &self.A,
                false,
                &mut self.workz,
                x,
                T::one(),
                T::zero(),
            );
        }
        for (v, &b) in self.workz.iter_mut().zip(bz) {
            *v -= b;
        }
        apply_scaling_pool(
            &self.pool,
            &self.scaling_lanes,
            self.scaling_tiles,
            &mut self.blocks,
            z,
            &self.workz,
            true,
        );
        for (&v, &row) in self.retained_rhs.iter().zip(&self.retained_rows) {
            z[row] = v;
        }
        out.is_finite()
    }

    fn residual(&mut self, out: &mut [T], rhs: &[T], solution: &[T], reuse_forward: bool) -> T {
        let __t0 = std::time::Instant::now();
        let __r = self.residual_inner(out, rhs, solution, reuse_forward);
        if std::env::var_os("SDPX_PROFILE").is_some() {
            eprintln!("PHASE residual {:?}", __t0.elapsed());
        }
        __r
    }

    fn residual_inner(
        &mut self,
        out: &mut [T],
        rhs: &[T],
        solution: &[T],
        reuse_forward: bool,
    ) -> T {
        let (x, z) = solution.split_at(self.n);
        let (ex, ez) = out.split_at_mut(self.n);
        ex.copy_from_slice(&rhs[..self.n]);
        ez.copy_from_slice(&rhs[self.n..]);
        // These stages own disjoint output/scratch buffers. Sharing the existing
        // pool can fill block-tail idle time without changing either arithmetic
        // order or allocating another set of workers.
        let Self {
            P,
            A,
            sampled,
            pool,
            workz,
            workh,
            blocks,
            scaling_lanes,
            scaling_tiles,
            sparse_products,
            ..
        } = self;
        let mut products = || {
            P.sym_up().symv(ex, x, -T::one(), T::one());
            if let Some((operator, work)) = sampled {
                let __t = std::time::Instant::now();
                operator.apply_transpose_with_pool(ex, z, -T::one(), T::one(), work, pool.as_ref());
                if std::env::var_os("SDPX_PROFILE").is_some() {
                    eprintln!("PHASE residual.adj {:?}", __t.elapsed());
                }
                if !reuse_forward {
                    let __t = std::time::Instant::now();
                    operator.apply_with_pool(ez, x, -T::one(), T::one(), work, pool.as_ref());
                    if std::env::var_os("SDPX_PROFILE").is_some() {
                        eprintln!("PHASE residual.fwd {:?}", __t.elapsed());
                    }
                }
            } else if let (Some(world), Some(plan)) =
                (crate::mpi::World::get(), sparse_products.as_ref())
            {
                plan.product_sharded(
                    A,
                    true,
                    ex,
                    z,
                    -T::one(),
                    T::one(),
                    world,
                    crate::mpi::SITE_RX,
                );
                if !reuse_forward {
                    plan.product_sharded(
                        A,
                        false,
                        ez,
                        x,
                        -T::one(),
                        T::one(),
                        world,
                        crate::mpi::SITE_RZ,
                    );
                }
            } else {
                A.t().gemv(ex, z, -T::one(), T::one());
                if !reuse_forward {
                    A.gemv(ez, x, -T::one(), T::one());
                }
            }
            if reuse_forward {
                // solve_raw just evaluated A*x-bz at this exact returned x.
                // Reuse that original-operator product only for the initial point;
                // adding a refinement correction invalidates it.
                for (e, &a_minus_b) in ez.iter_mut().zip(workz.iter()) {
                    *e = -a_minus_b;
                }
            }
        };
        let mut scaling = || {
            let __t = std::time::Instant::now();
            apply_scaling_pool(pool, scaling_lanes, *scaling_tiles, blocks, workh, z, false);
            if std::env::var_os("SDPX_PROFILE").is_some() {
                eprintln!("PHASE residual.scale {:?}", __t.elapsed());
            }
        };
        if let Some(pool) = pool.as_ref().filter(|_| scaling_lanes.len() > 1) {
            pool.install(|| rayon::join(products, scaling));
        } else {
            products();
            scaling();
        }
        for (e, &h) in ez.iter_mut().zip(&self.workh) {
            *e += h;
        }
        if out.is_finite() {
            out.norm_inf()
        } else {
            T::infinity()
        }
    }
}

#[path = "condensed_scaling.rs"]
mod scaling;
use scaling::*;

#[path = "condensed_psd.rs"]
mod psd_impl;
#[cfg(test)]
pub(crate) use psd_impl::POOLED_CONGRUENCE_TILES;

#[path = "condensed_kkt.rs"]
mod kkt_impl;

#[cfg(test)]
#[path = "condensed_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "condensed_parallel_tests.rs"]
mod parallel_tests;

#[cfg(test)]
#[path = "condensed_graded_tests.rs"]
mod graded_action_tests;
