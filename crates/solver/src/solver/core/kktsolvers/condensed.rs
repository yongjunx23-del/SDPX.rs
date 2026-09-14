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
use std::collections::BTreeSet;
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
    R: Matrix<T>,
    Rinv: Matrix<T>,
    G: Matrix<T>,
    Ginv: Matrix<T>,
    mat1: Matrix<T>,
    mat2: Matrix<T>,
    mat3: Matrix<T>,
    vector: Vec<T>,
    columns: Vec<Column>,
    schur_values: Vec<T>,
    sampled: Option<SampledPsd<T>>,
    sparse_column_lanes: Vec<usize>,
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
    inner_schur: bool,
    inner_sampled: Option<usize>,
    sampled: Option<(Arc<SampledOperator<T>>, SampledWorkspace<T>)>,
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
        for (ci, (cone, rows)) in cones.iter().zip(&cones.rng_cones).enumerate() {
            let scaling = match cone {
                SupportedCone::PSDTriangleCone(c) => Scaling::Psd(PsdBlock::new(c.n, A, rows)),
                SupportedCone::NonnegativeCone(c) => {
                    let mut entries = vec![Vec::new(); rows.len()];
                    for col in 0..n {
                        for p in A.colptr[col]..A.colptr[col + 1] {
                            let row = A.rowval[p];
                            if rows.contains(&row) {
                                entries[row - rows.start].push((col, p));
                            }
                        }
                    }
                    Scaling::Orthant {
                        w: c.w.clone(),
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
            if !matches!(scaling, Scaling::Psd(_) | Scaling::Orthant { .. }) {
                retained_indices.push(ci);
                retained_types.push(types[ci].clone());
                for row in rows.clone() {
                    rowmap[row] = retained_rows.len();
                    retained_rows.push(row);
                }
            }
            blocks.push(Block {
                rows: rows.clone(),
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
                            pattern[right.index].insert(left.index);
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
                        .map(|&i| schur_position(&schur, i, right.index))
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
            inner_schur: false,
            inner_sampled: None,
            sampled: None,
        };
        solver.refresh_parallel_plan();
        solver
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
        self.scaling_lanes = weighted_lanes(&costs, workers);
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
                Scaling::Orthant { w, rows } => {
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
                self.pool.as_deref(),
            );
        } else {
            self.A
                .t()
                .gemv(&mut self.workx, &self.workz, T::one(), T::one());
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
                self.pool.as_deref(),
            );
        } else {
            self.A.gemv(&mut self.workz, x, T::one(), T::zero());
        }
        for (v, &b) in self.workz.iter_mut().zip(bz) {
            *v -= b;
        }
        apply_scaling_pool(
            &self.pool,
            &self.scaling_lanes,
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

    fn residual(&mut self, out: &mut [T], rhs: &[T], solution: &[T]) -> T {
        let (x, z) = solution.split_at(self.n);
        let (ex, ez) = out.split_at_mut(self.n);
        ex.copy_from_slice(&rhs[..self.n]);
        ez.copy_from_slice(&rhs[self.n..]);
        self.P.sym_up().symv(ex, x, -T::one(), T::one());
        if let Some((operator, work)) = &mut self.sampled {
            operator.apply_transpose_with_pool(
                ex,
                z,
                -T::one(),
                T::one(),
                work,
                self.pool.as_deref(),
            );
            operator.apply_with_pool(ez, x, -T::one(), T::one(), work, self.pool.as_deref());
        } else {
            self.A.t().gemv(ex, z, -T::one(), T::one());
            self.A.gemv(ez, x, -T::one(), T::one());
        }
        apply_scaling_pool(
            &self.pool,
            &self.scaling_lanes,
            &mut self.blocks,
            &mut self.workh,
            z,
            false,
        );
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
        self.refresh_parallel_plan();
        self.parallel_assembly = self.pool.is_some()
            && self.blocks.iter().all(|block| match &block.scaling {
                Scaling::Psd(p) => p.schur_values.len() == triangular_number(p.columns.len()),
                _ => true,
            });
        let inner_sampled = self.inner_sampled;
        let pool = &self.pool;
        let sync = |(block, cone): (&mut Block<T>, &SupportedCone<T>)| {
            match (&mut block.scaling, cone) {
                (Scaling::Psd(p), SupportedCone::PSDTriangleCone(c)) => {
                    p.R.copy_from_slice(c.scaling_R().data());
                    p.Rinv.copy_from_slice(c.scaling_Rinv().data());
                    let fast_apply = T::precision_bits() > 53;
                    if let Some(sampled) = &mut p.sampled {
                        sampled.work.update_with_pool(
                            &sampled.operator.blocks()[sampled.block],
                            &p.Rinv,
                            if inner_sampled == Some(block.rows.start) {
                                pool.as_deref()
                            } else {
                                None
                            },
                        );
                        if fast_apply {
                            p.Ginv.syrk(
                                &p.Rinv.t(),
                                T::one(),
                                T::zero(),
                                MatrixTriangle::Triu,
                            );
                        }
                    } else {
                        p.Ginv
                            .syrk(&p.Rinv.t(), T::one(), T::zero(), MatrixTriangle::Triu);
                    }
                    if fast_apply {
                        p.G.syrk(&p.R, T::one(), T::zero(), MatrixTriangle::Triu);
                        for j in 0..c.n {
                            for i in j + 1..c.n {
                                p.G[(i, j)] = p.G[(j, i)];
                                p.Ginv[(i, j)] = p.Ginv[(j, i)];
                            }
                        }
                    } else if p.sampled.is_none() {
                        for j in 0..c.n {
                            for i in j + 1..c.n {
                                p.Ginv[(i, j)] = p.Ginv[(j, i)];
                            }
                        }
                    }
                    if !p.R.data().is_finite()
                        || !p.Rinv.data().is_finite()
                        || !p.Ginv.data().is_finite()
                        || (fast_apply && !p.G.data().is_finite())
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
        let valid = if inner_sampled.is_some() {
            self.blocks
                .iter_mut()
                .zip(cones.iter())
                .map(sync)
                .fold(true, |a, b| a & b)
        } else if let Some(pool) = &self.pool {
            pool.install(|| {
                self.blocks
                    .par_iter_mut()
                    .zip(cones.iter().as_slice().par_iter())
                    .map(sync)
                    .reduce(|| true, |a, b| a & b)
            })
        } else {
            self.blocks
                .iter_mut()
                .zip(cones.iter())
                .map(sync)
                .fold(true, |a, b| a & b)
        };
        if !valid {
            return false;
        }
        if !self.assemble() {
            return false;
        }
        self.reduced.update_P(&self.schur);
        let retained = &self.retained_indices;
        self.reduced.update_from_cones(
            cones
                .iter()
                .enumerate()
                .filter(|(i, _)| retained.binary_search(i).is_ok())
                .map(|(_, c)| c),
            settings,
        )
    }

    fn setrhs(&mut self, x: &[T], z: &[T]) {
        self.b[..self.n].copy_from_slice(x);
        self.b[self.n..].copy_from_slice(z);
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
                let mut norme = self.residual(&mut error, &b, &x);
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
                    norme = self.residual(&mut error, &b, &candidate);
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

impl<T: FloatT> PsdBlock<T> {
    fn new(n: usize, A: &CscMatrix<T>, rows: &Range<usize>) -> Self {
        let mut coordinates = Vec::with_capacity(rows.len());
        for j in 0..n {
            for i in 0..=j {
                coordinates.push((i, j));
            }
        }
        let mut columns = Vec::new();
        let mut prefix_entries = 0u128;
        for col in 0..A.n {
            let mut entries = Vec::new();
            for position in A.colptr[col]..A.colptr[col + 1] {
                let row = A.rowval[position];
                if rows.contains(&row) {
                    let (i, j) = coordinates[row - rows.start];
                    entries.push(Entry { position, i, j });
                }
            }
            if !entries.is_empty() {
                prefix_entries += entries.len() as u128;
                // Exact sparse coefficient pairs versus two dense congruence
                // products. Select per column so one dense coefficient does
                // not force all sparse columns through dense GEMMs.
                let sparse = 8 * entries.len() as u128 * prefix_entries
                    <= 4 * (n as u128).pow(3) + prefix_entries;
                columns.push(Column {
                    index: col,
                    entries,
                    sparse,
                    schur_positions: Vec::new(),
                });
            }
        }
        Self {
            R: Matrix::zeros((n, n)),
            Rinv: Matrix::zeros((n, n)),
            G: Matrix::zeros((n, n)),
            Ginv: Matrix::zeros((n, n)),
            mat1: Matrix::zeros((n, n)),
            mat2: Matrix::zeros((n, n)),
            mat3: Matrix::zeros((n, n)),
            vector: vec![T::zero(); rows.len()],
            columns,
            schur_values: Vec::new(),
            sampled: None,
            sparse_column_lanes: Vec::new(),
        }
    }

    fn configure_sparse_columns(&mut self, workers: usize) -> (u128, u128) {
        let words = T::precision_bits().div_ceil(64) as u128;
        let mut prefix_entries = 0u128;
        let mut dense_work = 0u128;
        let costs: Vec<_> = self
            .columns
            .iter()
            .map(|column| {
                prefix_entries += column.entries.len() as u128;
                if column.sparse {
                    8 * prefix_entries * column.entries.len() as u128 * words * words
                } else {
                    dense_work +=
                        (4 * (self.R.size().0 as u128).pow(3) + prefix_entries) * words * words;
                    0
                }
            })
            .collect();
        // This is a structural work estimate, not a measured timing.
        let lanes =
            workers.min((costs.iter().sum::<u128>() / 4096).min(usize::MAX as u128) as usize);
        self.sparse_column_lanes = weighted_lanes(&costs, lanes.max(1));
        let sparse_work = costs.iter().sum::<u128>();
        (sparse_work, sparse_work + dense_work)
    }

    fn compute_schur(&mut self, values: &[T], mut store: impl FnMut(usize, usize, usize, T)) {
        self.compute_schur_selected(values, false, &mut store);
    }

    fn compute_schur_selected(
        &mut self,
        values: &[T],
        skip_sparse: bool,
        mut store: impl FnMut(usize, usize, usize, T),
    ) {
        if let Some(sampled) = &self.sampled {
            let block = &sampled.operator.blocks()[sampled.block];
            for (b, right) in self.columns.iter().enumerate() {
                for (a, left) in self.columns[..=b].iter().enumerate() {
                    let value = sampled.work.entry(
                        block,
                        right.index - block.column_start,
                        left.index - block.column_start,
                    );
                    store(b, a, right.schur_positions[a], value);
                }
            }
            return;
        }
        for (b, right) in self.columns.iter().enumerate() {
            if skip_sparse && right.sparse {
                continue;
            }
            if !right.sparse {
                self.mat1.data_mut().fill(T::zero());
                for e in &right.entries {
                    let v = values[e.position];
                    if e.i == e.j {
                        self.mat1[(e.i, e.j)] = v;
                    } else {
                        let v = v * T::FRAC_1_SQRT_2();
                        self.mat1[(e.i, e.j)] = v;
                        self.mat1[(e.j, e.i)] = v;
                    }
                }
                self.mat2.mul(&self.Ginv, &self.mat1, T::one(), T::zero());
                self.mat3.mul(&self.mat2, &self.Ginv, T::one(), T::zero());
                mat_to_svec(&mut self.vector, &self.mat3);
            }
            for (a, left) in self.columns[..=b].iter().enumerate() {
                let mut v = T::zero();
                if right.sparse {
                    v = sparse_schur_value(left, right, &self.Ginv, values);
                } else {
                    for e in &left.entries {
                        v = values[e.position]
                            .mul_add(self.vector[triangular_number(e.j) + e.i], v);
                    }
                }
                store(b, a, right.schur_positions[a], v);
            }
        }
    }

    fn scatter_schur(&self, S: &mut CscMatrix<T>) {
        for (b, column) in self.columns.iter().enumerate() {
            for (a, &position) in column.schur_positions.iter().enumerate() {
                S.nzval[position] += self.schur_values[triangular_number(b) + a];
            }
        }
    }

    fn apply(&mut self, y: &mut [T], x: &[T], inverse: bool) {
        svec_to_mat(&mut self.mat1, x);
        if T::precision_bits() > 53 {
            // Reassociated congruence: (RinvᵀRinv)·X·(RinvᵀRinv) equals the
            // factorized four-product form up to the declared precision.
            let m: &Matrix<T> = if inverse { &self.Ginv } else { &self.G };
            self.mat2.mul(&self.mat1, m, T::one(), T::zero());
            self.mat3.mul(m, &self.mat2, T::one(), T::zero());
            mat_to_svec(y, &self.mat3);
            return;
        }
        if inverse {
            // H^-1 = W^-1 W^-T; retain the inverse factors in solve-time
            // applications rather than squaring them into Ginv.
            self.mat2
                .mul(&self.mat1, &self.Rinv.t(), T::one(), T::zero());
            self.mat3.mul(&self.Rinv, &self.mat2, T::one(), T::zero());
            mat_to_svec(&mut self.vector, &self.mat3);
            svec_to_mat(&mut self.mat1, &self.vector);
            self.mat2
                .mul(&self.Rinv.t(), &self.mat1, T::one(), T::zero());
            self.mat3.mul(&self.mat2, &self.Rinv, T::one(), T::zero());
        } else {
            self.mat2.mul(&self.R.t(), &self.mat1, T::one(), T::zero());
            self.mat3.mul(&self.mat2, &self.R, T::one(), T::zero());
            mat_to_svec(&mut self.vector, &self.mat3);
            svec_to_mat(&mut self.mat1, &self.vector);
            self.mat2.mul(&self.mat1, &self.R.t(), T::one(), T::zero());
            self.mat3.mul(&self.R, &self.mat2, T::one(), T::zero());
        }
        mat_to_svec(y, &self.mat3);
    }
}

fn psd_entry<T: FloatT>(G: &Matrix<T>, a: Entry, b: Entry) -> T {
    match (a.i == a.j, b.i == b.j) {
        (true, true) => G[(a.i, b.i)] * G[(a.i, b.i)],
        (true, false) => T::SQRT_2() * G[(a.i, b.i)] * G[(a.i, b.j)],
        (false, true) => T::SQRT_2() * G[(a.i, b.i)] * G[(a.j, b.i)],
        (false, false) => G[(a.i, b.i)] * G[(a.j, b.j)] + G[(a.i, b.j)] * G[(a.j, b.i)],
    }
}

fn apply_scaling<T: FloatT>(blocks: &mut [Block<T>], y: &mut [T], x: &[T], inverse: bool) {
    y.fill(T::zero());
    let offset = blocks.first().map_or(0, |b| b.rows.start);
    for block in blocks {
        let rows = block.rows.start - offset..block.rows.end - offset;
        let (y, x) = (&mut y[rows.clone()], &x[rows]);
        match &mut block.scaling {
            Scaling::Psd(p) => p.apply(y, x, inverse),
            Scaling::Orthant { w, .. } => {
                for ((y, &x), &w) in y.iter_mut().zip(x).zip(w.iter()) {
                    *y = if inverse { (x / w) / w } else { w * (w * x) };
                }
            }
            Scaling::Zero => {}
            _ if inverse => {} // Retained rows are never inverted.
            Scaling::Soc { w, eta } => {
                let two: T = (2.).as_T();
                let c = two * w.dot(x);
                y.copy_from_slice(x);
                if !y.is_empty() {
                    y[0] = -x[0];
                }
                y.axpby(c, w, T::one());
                y.scale(*eta * *eta);
            }
            Scaling::Dense3(h) => {
                let mut p = 0;
                for j in 0..3 {
                    for i in 0..=j {
                        y[i] = h[p].mul_add(x[j], y[i]);
                        if i != j {
                            y[j] = h[p].mul_add(x[i], y[j]);
                        }
                        p += 1;
                    }
                }
            }
            Scaling::GenPower {
                p,
                q,
                r,
                d1,
                d2,
                mu,
            } => {
                let dim1 = q.len();
                let cp = p.dot(x);
                let cq = q.dot(&x[..dim1]);
                let cr = r.dot(&x[dim1..]);
                for i in 0..dim1 {
                    y[i] = d1[i] * x[i] - cq * q[i];
                }
                for i in dim1..x.len() {
                    y[i] = *d2 * x[i] - cr * r[i - dim1];
                }
                y.axpby(cp, p, T::one());
                y.scale(*mu);
            }
        }
    }
}

// Cached contiguous partitions use structural work, never runtime timings.
fn weighted_lanes(costs: &[u128], workers: usize) -> Vec<usize> {
    if costs.is_empty() {
        return Vec::new();
    }
    let workers = workers.max(1).min(costs.len());
    let mut prefix = Vec::with_capacity(costs.len() + 1);
    prefix.push(0u128);
    for &cost in costs {
        prefix.push(prefix.last().unwrap().saturating_add(cost));
    }
    let mut lanes = Vec::with_capacity(workers);
    let mut begin = 0;
    for lane in 0..workers {
        lanes.push(begin);
        let remaining = workers - lane;
        if remaining == 1 {
            break;
        }
        let target = (prefix[costs.len()] - prefix[begin]) / remaining as u128;
        let last = costs.len() - (remaining - 1);
        let mut end = begin + 1;
        while end < last && prefix[end] - prefix[begin] < target {
            end += 1;
        }
        if end > begin + 1
            && target.abs_diff(prefix[end - 1] - prefix[begin])
                <= target.abs_diff(prefix[end] - prefix[begin])
        {
            end -= 1;
        }
        begin = end;
    }
    lanes
}

fn apply_scaling_pool<T: FloatT>(
    pool: &Option<Arc<rayon::ThreadPool>>,
    lanes: &[usize],
    blocks: &mut [Block<T>],
    y: &mut [T],
    x: &[T],
    inverse: bool,
) {
    if let Some(pool) = pool {
        if lanes.len() > 1 {
            pool.install(|| split_scaling(blocks, y, x, inverse, lanes));
            return;
        }
    }
    apply_scaling(blocks, y, x, inverse);
}

fn split_scaling<T: FloatT>(
    blocks: &mut [Block<T>],
    y: &mut [T],
    x: &[T],
    inverse: bool,
    lanes: &[usize],
) {
    if lanes.len() <= 1 {
        apply_scaling(blocks, y, x, inverse);
        return;
    }
    let mid = lanes.len() / 2;
    let block = lanes[mid] - lanes[0];
    let row = blocks[block].rows.start - blocks[0].rows.start;
    let (left, right) = blocks.split_at_mut(block);
    let (yl, yr) = y.split_at_mut(row);
    let (xl, xr) = x.split_at(row);
    rayon::join(
        || split_scaling(left, yl, xl, inverse, &lanes[..mid]),
        || split_scaling(right, yr, xr, inverse, &lanes[mid..]),
    );
}

fn sparse_schur_value<T: FloatT>(
    left: &Column,
    right: &Column,
    ginv: &Matrix<T>,
    values: &[T],
) -> T {
    let mut v = T::zero();
    for a in &left.entries {
        for b in &right.entries {
            v += values[a.position] * values[b.position] * psd_entry(ginv, *a, *b);
        }
    }
    v
}

fn split_sparse_columns<T: FloatT>(
    columns: &[Column],
    ginv: &Matrix<T>,
    values: &[T],
    output: &mut [T],
    lanes: &[usize],
    end: usize,
) {
    let begin = lanes[0];
    if lanes.len() == 1 {
        let offset = triangular_number(begin);
        for b in begin..end {
            let right = &columns[b];
            if !right.sparse {
                continue;
            }
            for (a, left) in columns[..=b].iter().enumerate() {
                output[triangular_number(b) + a - offset] =
                    sparse_schur_value(left, right, ginv, values);
            }
        }
    } else {
        let mid = lanes.len() / 2;
        let cut = lanes[mid];
        let (left, right) = output.split_at_mut(triangular_number(cut) - triangular_number(begin));
        rayon::join(
            || split_sparse_columns(columns, ginv, values, left, &lanes[..mid], cut),
            || split_sparse_columns(columns, ginv, values, right, &lanes[mid..], end),
        );
    }
}

fn split_sampled_columns<T: FloatT>(
    columns: &[Column],
    sampled: &SampledPsd<T>,
    output: &mut [T],
    lanes: &[usize],
    end: usize,
) {
    let begin = lanes[0];
    if lanes.len() == 1 {
        let block = &sampled.operator.blocks()[sampled.block];
        let offset = triangular_number(begin);
        for b in begin..end {
            for a in 0..=b {
                output[triangular_number(b) + a - offset] = sampled.work.entry(
                    block,
                    columns[b].index - block.column_start,
                    columns[a].index - block.column_start,
                );
            }
        }
    } else {
        let mid = lanes.len() / 2;
        let cut = lanes[mid];
        let (left, right) = output.split_at_mut(triangular_number(cut) - triangular_number(begin));
        rayon::join(
            || split_sampled_columns(columns, sampled, left, &lanes[..mid], cut),
            || split_sampled_columns(columns, sampled, right, &lanes[mid..], end),
        );
    }
}

fn schur_position<T>(S: &CscMatrix<T>, i: usize, j: usize) -> usize {
    let start = S.colptr[j];
    start
        + S.rowval[start..S.colptr[j + 1]]
            .binary_search(&i)
            .expect("missing structural Schur entry")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::core::ScalingStrategy;
    use sdpx_arithmetic::Bits128;

    // Repeated small PSD blocks share local column groups, while equality
    // rows touch every variable. Dimensions are deliberately generic and
    // chosen so charging a dense primal factor would reject this structure.
    fn selector_local_problem(
        independent_orthant: Option<bool>,
    ) -> (CscMatrix<f64>, CscMatrix<f64>, CompositeCone<f64>) {
        let (groups, columns_per_group, side, equalities) = (10, 18, 10, 8);
        let n = groups * columns_per_group;
        let psd_rows = triangular_number(side);
        let all_psd_rows = 2 * groups * psd_rows;
        let nn_rows = match independent_orthant {
            None => 0,
            Some(true) => n,
            Some(false) => 1,
        };
        let mut kinds = vec![SupportedConeT::PSDTriangleConeT(side); 2 * groups];
        kinds.push(SupportedConeT::ZeroConeT(equalities));
        if nn_rows != 0 {
            kinds.push(SupportedConeT::NonnegativeConeT(nn_rows));
        }
        let mut colptr = vec![0];
        let mut rowval = Vec::new();
        for col in 0..n {
            let group = col / columns_per_group;
            let coordinate = col % columns_per_group;
            rowval.push(2 * group * psd_rows + coordinate);
            rowval.push((2 * group + 1) * psd_rows + coordinate);
            rowval.extend(all_psd_rows..all_psd_rows + equalities);
            if let Some(independent) = independent_orthant {
                rowval.push(all_psd_rows + equalities + if independent { col } else { 0 });
            }
            colptr.push(rowval.len());
        }
        let nzval = vec![1.; rowval.len()];
        let A = CscMatrix::new(
            all_psd_rows + equalities + nn_rows,
            n,
            colptr,
            rowval,
            nzval,
        );
        (CscMatrix::identity(n), A, CompositeCone::new(&kinds))
    }

    #[test]
    fn selector_prefers_local_psd_groups_with_global_equality_border() {
        let (mut P, mut A, cones) = selector_local_problem(None);
        let settings = CoreSettings::default();
        assert!(prefer_condensed(&P, &A, &cones, &settings));
        // Selection is invariant to coefficients, including all stored zeros.
        P.nzval.fill(0.);
        A.nzval.fill(0.);
        assert!(prefer_condensed(&P, &A, &cones, &settings));
    }

    #[test]
    fn selector_counts_stored_edges_and_independent_orthant_rows() {
        let (P, mut A, cones) = selector_local_problem(None);
        let settings = CoreSettings::default();
        // One PSD block touching every column forces a dense primal clique.
        // The stored zero coefficients must still contribute to this graph.
        A.nzval.fill(0.);
        for col in 0..A.n {
            A.rowval[A.colptr[col]] = 0;
        }
        assert!(!prefer_condensed(&P, &A, &cones, &settings));

        let (_, A, cones) = selector_local_problem(None);
        let mut colptr = vec![0];
        let mut rowval = Vec::new();
        for col in 0..A.n {
            rowval.extend(0..=col);
            colptr.push(rowval.len());
        }
        let nzval = vec![0.; rowval.len()];
        let dense_P = CscMatrix::new(A.n, A.n, colptr, rowval, nzval);
        assert!(!prefer_condensed(&dense_P, &A, &cones, &settings));

        let (P, A, cones) = selector_local_problem(Some(true));
        assert!(prefer_condensed(&P, &A, &cones, &settings));
        let (P, A, cones) = selector_local_problem(Some(false));
        assert!(!prefer_condensed(&P, &A, &cones, &settings));
    }

    #[test]
    fn selector_preserves_coarse_guards_and_dense_storage_comparison() {
        let (_, A, cones) = selector_local_problem(None);
        let settings = CoreSettings::default();
        assert!(!prefer_condensed(
            &CscMatrix::zeros((0, 0)),
            &CscMatrix::zeros((A.m, 0)),
            &cones,
            &settings
        ));
        let wide = CscMatrix::zeros((A.m, 300));
        assert!(!prefer_condensed(
            &CscMatrix::identity(wide.n),
            &wide,
            &cones,
            &settings
        ));
        let small = CompositeCone::new(&[SupportedConeT::PSDTriangleConeT(10)]);
        assert!(!prefer_condensed(
            &CscMatrix::identity(1),
            &CscMatrix::zeros((55, 1)),
            &small,
            &settings
        ));
        // A dense Schur is still worthwhile when its PSD block is much larger.
        let large = CompositeCone::new(&[SupportedConeT::PSDTriangleConeT(40)]);
        let A = CscMatrix::new(820, 20, (0..=20).collect(), (0..20).collect(), vec![1.; 20]);
        assert!(prefer_condensed(
            &CscMatrix::identity(20),
            &A,
            &large,
            &settings
        ));
    }

    fn mixed_operator<T: FloatT>() {
        let kinds = [
            SupportedConeT::PSDTriangleConeT(2),
            SupportedConeT::NonnegativeConeT(2),
            SupportedConeT::ZeroConeT(1),
            SupportedConeT::SecondOrderConeT(3),
        ];
        let mut cones = CompositeCone::new(&kinds);
        let mut P = CscMatrix::from(&[
            [T::from_f64(3.).unwrap(), T::from_f64(0.25).unwrap()],
            [T::zero(), T::from_f64(2.).unwrap()],
        ]);
        let mut A = CscMatrix::from(&[
            [T::one(), T::zero()],
            [T::from_f64(0.3).unwrap(), T::one()],
            [T::zero(), T::one()],
            [T::one(), T::from_f64(-0.5).unwrap()],
            [T::from_f64(0.25).unwrap(), T::one()],
            [T::one(), T::one()],
            [T::one(), T::from_f64(-0.25).unwrap()],
            [T::from_f64(0.5).unwrap(), T::one()],
            [T::one(), T::from_f64(0.5).unwrap()],
        ]);
        let conv = |v: &[f64]| {
            v.iter()
                .map(|&x| T::from_f64(x).unwrap())
                .collect::<Vec<_>>()
        };
        let mut s = conv(&[3., 0.2, 2., 1.3, 2.1, 0., 2., 0.2, 0.1]);
        let z = conv(&[1.5, -0.1, 2.5, 0.7, 1.2, 1., 3., -0.2, 0.1]);
        let mut settings = CoreSettings::<T>::default();
        settings.iterative_refinement_abstol = T::epsilon() * (1024.).as_T();
        settings.iterative_refinement_reltol = settings.iterative_refinement_abstol;
        let mut solver = CondensedKKTSolver::new(&P, &A, &kinds, &cones, &settings);
        let exact_x = conv(&[0.25, -0.5]);
        let exact_z = conv(&[0.5, -0.2, 0.25, 0.125, -0.25, 0.5, -0.25, 0.2, 0.1]);
        for update in 0..2 {
            if update != 0 {
                P.nzval[0] += T::from_f64(0.125).unwrap();
                A.nzval[0] *= T::from_f64(1.25).unwrap();
                s[0] += T::from_f64(0.25).unwrap();
                solver.update_P(&P);
                solver.update_A(&A);
            }
            assert!(cones.update_scaling(&s, &z, T::one(), ScalingStrategy::PrimalDual));
            assert!(solver.update(&cones, &settings));
            let mut bx = vec![T::zero(); 2];
            let mut bz = vec![T::zero(); 9];
            let mut hz = vec![T::zero(); 9];
            let mut scratch = vec![T::zero(); 9];
            P.sym_up().symv(&mut bx, &exact_x, T::one(), T::zero());
            A.t().gemv(&mut bx, &exact_z, T::one(), T::one());
            A.gemv(&mut bz, &exact_x, T::one(), T::zero());
            cones.mul_Hs(&mut hz, &exact_z, &mut scratch);
            for (b, h) in bz.iter_mut().zip(hz) {
                *b -= h;
            }
            solver.setrhs(&bx, &bz);
            let (mut x, mut z) = (vec![T::zero(); 2], vec![T::zero(); 9]);
            assert!(solver.solve(Some(&mut x), Some(&mut z), &settings));
            // Independent original cone operator, not the cached inverse or S.
            let mut ex = bx.clone();
            let mut ez = bz.clone();
            P.sym_up().symv(&mut ex, &x, -T::one(), T::one());
            A.t().gemv(&mut ex, &z, -T::one(), T::one());
            A.gemv(&mut ez, &x, -T::one(), T::one());
            cones.mul_Hs(&mut scratch, &z, &mut solver.workh);
            for (e, h) in ez.iter_mut().zip(&scratch) {
                *e += *h;
            }
            let tolerance = settings.iterative_refinement_abstol
                + settings.iterative_refinement_reltol * bx.norm_inf().max(bz.norm_inf());
            assert!(ex.norm_inf().max(ez.norm_inf()) <= tolerance);

            for block in &mut solver.blocks {
                if let Scaling::Psd(p) = &mut block.scaling {
                    for c in &mut p.columns {
                        c.sparse = true;
                    }
                }
            }
            assert!(solver.assemble());
            let sparse = solver.schur.nzval.clone();
            for block in &mut solver.blocks {
                if let Scaling::Psd(p) = &mut block.scaling {
                    for c in &mut p.columns {
                        c.sparse = false;
                    }
                }
            }
            assert!(solver.assemble());
            assert!(
                solver.schur.nzval.norm_inf_diff(&sparse)
                    <= settings.iterative_refinement_abstol * sparse.norm_inf()
            );
        }
    }

    #[test]
    fn condensed_mixed_original_operator_f64() {
        mixed_operator::<f64>();
    }
    #[test]
    fn condensed_mixed_original_operator_mpfr128() {
        mixed_operator::<Bits128>();
    }

    fn nonsymmetric_operator<T: FloatT>() {
        let kinds = [
            SupportedConeT::PSDTriangleConeT(2),
            SupportedConeT::ExponentialConeT(),
            SupportedConeT::PowerConeT(T::from_f64(0.4).unwrap()),
            SupportedConeT::GenPowerConeT(
                vec![T::from_f64(0.25).unwrap(), T::from_f64(0.75).unwrap()],
                2,
            ),
        ];
        let mut cones = CompositeCone::new(&kinds);
        let m = cones.numel();
        let (mut s, mut z) = (vec![T::zero(); m], vec![T::zero(); m]);
        cones.unit_initialization(&mut z, &mut s);
        assert!(cones.update_scaling(&s, &z, T::one(), ScalingStrategy::Dual));
        let rows: Vec<[T; 2]> = (0..m)
            .map(|i| {
                [
                    T::from_usize(i % 5 + 1).unwrap() / T::from_usize(5).unwrap(),
                    T::from_usize(i % 7 + 1).unwrap() / T::from_usize(7).unwrap(),
                ]
            })
            .collect();
        let A = CscMatrix::from(&rows);
        let P = CscMatrix::identity(2);
        let mut settings = CoreSettings::<T>::default();
        settings.iterative_refinement_abstol = T::epsilon() * (1024.).as_T();
        settings.iterative_refinement_reltol = settings.iterative_refinement_abstol;
        let mut kkt = CondensedKKTSolver::new(&P, &A, &kinds, &cones, &settings);
        assert!(kkt.update(&cones, &settings));
        let exact_x = [T::from_f64(0.25).unwrap(), T::from_f64(-0.5).unwrap()];
        let exact_z = vec![T::from_f64(0.125).unwrap(); m];
        let mut bx = exact_x.to_vec();
        A.t().gemv(&mut bx, &exact_z, T::one(), T::one());
        let mut bz = vec![T::zero(); m];
        let mut hs = vec![T::zero(); m];
        let mut scratch = vec![T::zero(); m];
        A.gemv(&mut bz, &exact_x, T::one(), T::zero());
        cones.mul_Hs(&mut hs, &exact_z, &mut scratch);
        for (b, &h) in bz.iter_mut().zip(&hs) {
            *b -= h;
        }
        kkt.setrhs(&bx, &bz);
        let (mut x, mut z) = (vec![T::zero(); 2], vec![T::zero(); m]);
        assert!(kkt.solve(Some(&mut x), Some(&mut z), &settings));
        let mut ex = bx.clone();
        let mut ez = bz.clone();
        P.sym_up().symv(&mut ex, &x, -T::one(), T::one());
        A.t().gemv(&mut ex, &z, -T::one(), T::one());
        A.gemv(&mut ez, &x, -T::one(), T::one());
        cones.mul_Hs(&mut hs, &z, &mut scratch);
        for (e, h) in ez.iter_mut().zip(hs) {
            *e += h;
        }
        let tolerance = settings.iterative_refinement_abstol
            + settings.iterative_refinement_reltol * bx.norm_inf().max(bz.norm_inf());
        assert!(ex.norm_inf().max(ez.norm_inf()) <= tolerance);
    }

    #[test]
    fn condensed_retained_nonsymmetric_original_operator_f64() {
        nonsymmetric_operator::<f64>();
    }
    #[test]
    fn condensed_retained_nonsymmetric_original_operator_mpfr128() {
        nonsymmetric_operator::<Bits128>();
    }

    #[test]
    fn structural_psd_blocks_remain_sparse_with_global_equalities() {
        let types = [
            SupportedConeT::PSDTriangleConeT(2),
            SupportedConeT::PSDTriangleConeT(2),
            SupportedConeT::ZeroConeT(1),
        ];
        let cones = CompositeCone::new(&types);
        let P = CscMatrix::new(
            4,
            4,
            vec![0, 1, 2, 4, 5],
            vec![0, 1, 0, 2, 3],
            vec![1., 1., 0., 1., 1.],
        );
        let A = CscMatrix::from(&[
            [1., 0., 0., 0.],
            [1., 1., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 1., 1.],
            [0., 0., 0., 1.],
            [1., 1., 1., 1.],
        ]);
        let solver = CondensedKKTSolver::new(&P, &A, &types, &cones, &CoreSettings::default());
        // Two 2x2 primal cliques plus one explicitly stored P edge, even zero.
        assert_eq!(solver.schur.nnz(), 7);
        assert_eq!(solver.retained_rows, vec![6]);
        assert_eq!(solver.retained_A.nnz(), 4);
        assert_eq!(
            solver.schur.rowval[solver.schur.colptr[3]..solver.schur.colptr[4]],
            [2, 3]
        );
    }

    #[test]
    fn condensed_dependent_equalities_keep_bordered_refinement() {
        use crate::solver::{DefaultSolver, IPSolver, SolverStatus};
        for scale in [0.01, 1., 100.] {
            let P = CscMatrix::identity(3);
            let A = CscMatrix::from(&[
                [0., scale, scale],
                [0., scale, -scale],
                [scale, 2. * scale, -scale],
                [2. * scale, -scale, 3. * scale],
                [0., 0., 0.],
                [0., 0., 0.],
                [0., 0., 0.],
                [0., 0., 0.],
            ]);
            let b = [scale, scale, scale, scale, 1., 0., 1., 1.];
            let kinds = [
                SupportedConeT::ZeroConeT(4),
                SupportedConeT::PSDTriangleConeT(2),
                SupportedConeT::NonnegativeConeT(1),
            ];
            let settings = CoreSettings {
                verbose: false,
                kkt_form: "condensed".to_string(),
                ..CoreSettings::default()
            };
            let mut solver = DefaultSolver::new(&P, &[0.; 3], &A, &b, &kinds, settings).unwrap();
            solver.solve();
            assert_eq!(
                solver.solution.status,
                SolverStatus::PrimalInfeasible,
                "scale {scale}"
            );
            let mut atz = vec![0.; 3];
            A.t().gemv(&mut atz, &solver.solution.z, 1., 0.);
            assert!(atz.norm_inf() <= 1e-8 * solver.solution.z.norm_inf());
            assert!(b.dot(&solver.solution.z) < 0.);
        }
    }

    #[test]
    fn forced_condensed_constant_program_keeps_public_contract() {
        use crate::solver::{DefaultSolver, IPSolver, SolverStatus};
        let kinds = [
            SupportedConeT::ZeroConeT(1),
            SupportedConeT::NonnegativeConeT(1),
            SupportedConeT::PSDTriangleConeT(2),
        ];
        let settings = CoreSettings {
            verbose: false,
            kkt_form: "condensed".to_string(),
            ..CoreSettings::default()
        };
        let mut solver = DefaultSolver::new(
            &CscMatrix::<f64>::zeros((0, 0)),
            &[],
            &CscMatrix::zeros((5, 0)),
            &[0., 1., 1., 0., 1.],
            &kinds,
            settings,
        )
        .unwrap();
        solver.solve();
        assert_eq!(solver.solution.status, SolverStatus::Solved);
        assert!(solver.solution.x.is_empty());
    }
}

#[cfg(test)]
#[path = "condensed_parallel_tests.rs"]
mod parallel_tests;
