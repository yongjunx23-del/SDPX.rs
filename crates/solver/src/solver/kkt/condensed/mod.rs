//! Condensation of PSD/orthant rows beneath the existing embedding solver.
//!
//! Adapted from SDPX.jl's kktsolver_condensed.jl and condensed_schur.jl.
//! The retained system uses Clarabel's existing LDL shifts and refinement.
//! No equality rows are inverted or removed, and residual refinement below
//! uses the original augmented operator, not the regularized Schur matrix.
#![allow(non_snake_case)]

use super::{direct::DirectLDLKKTSolver, HasLinearSolverInfo, KKTSolver, LinearSolverInfo};
use crate::algebra::*;
use crate::solver::{cones::*, core::CoreSettings};
use crate::solver::{SampledOperator, SampledSchurWorkspace, SampledWorkspace};
use rayon::prelude::*;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::ops::Range;
use std::sync::Arc;

/// Minimum extra per-block contribution storage allowed for the ordered
/// parallel assembly, in bytes. The publish is bitwise identical to the
/// serial scatter because blocks are summed in cone order.
const PARALLEL_ASSEMBLY_BUDGET_BYTES: usize = 256 << 20;

/// Physical memory capped by an exposed cgroup allocation.
fn effective_memory_bytes() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let info = std::fs::read_to_string("/proc/meminfo").ok()?;
        let line = info.lines().find(|l| l.starts_with("MemTotal:"))?;
        let kib: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
        let physical = kib.saturating_mul(1024);
        Some(cgroup_memory_limit_bytes().map_or(physical, |limit| physical.min(limit)))
    }
    #[cfg(target_os = "macos")]
    {
        let out = std::process::Command::new("sysctl")
            .args(["-n", "hw.memsize"])
            .output()
            .ok()?;
        String::from_utf8(out.stdout).ok()?.trim().parse().ok()
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        None
    }
}

#[cfg(target_os = "linux")]
fn cgroup_memory_limit_bytes() -> Option<u64> {
    use std::path::Path;
    let groups = std::fs::read_to_string("/proc/self/cgroup").ok()?;
    let mounts = std::fs::read_to_string("/proc/self/mountinfo").ok()?;
    let decode = |s: &str| {
        s.replace("\\040", " ")
            .replace("\\011", "\t")
            .replace("\\012", "\n")
            .replace("\\134", "\\")
    };
    let mut limit: Option<u64> = None;
    for group in groups.lines() {
        let mut fields = group.splitn(3, ':');
        fields.next();
        let (controllers, group) = (fields.next()?, fields.next()?);
        let (kind, file) = if controllers.is_empty() {
            ("cgroup2", "memory.max")
        } else if controllers.split(',').any(|c| c == "memory") {
            ("cgroup", "memory.limit_in_bytes")
        } else {
            continue;
        };
        for mount in mounts.lines() {
            let Some((fields, options)) = mount.split_once(" - ") else {
                continue;
            };
            let mut options = options.split_whitespace();
            if options.next() != Some(kind) {
                continue;
            }
            options.next();
            if kind == "cgroup" && !options.next()?.split(',').any(|c| c == "memory") {
                continue;
            }
            let mut fields = fields.split_whitespace();
            let root = decode(fields.nth(3)?);
            let mount = decode(fields.next()?);
            let Ok(relative) = Path::new(group).strip_prefix(&root) else {
                continue;
            };
            let mount = Path::new(&mount);
            let mut path = mount.join(relative);
            loop {
                if let Some(bytes) = std::fs::read_to_string(path.join(file))
                    .ok()
                    .and_then(|s| s.trim().parse::<u64>().ok())
                {
                    limit = Some(limit.map_or(bytes, |old| old.min(bytes)));
                }
                if path == mount || !path.pop() {
                    break;
                }
            }
        }
    }
    limit
}

/// Assembly buffer budget: one eighth of effective memory, never below the
/// fixed floor. Cached after the first query.
pub(crate) fn parallel_assembly_budget_bytes() -> usize {
    static BUDGET: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *BUDGET.get_or_init(|| {
        let share = effective_memory_bytes().map_or(0, |b| (b / 8).min(usize::MAX as u64) as usize);
        share.max(PARALLEL_ASSEMBLY_BUDGET_BYTES)
    })
}

/// Use per-block assembly buffers when a cone pool exists and the extra
/// storage is bounded either by the original two-copy rule or by the byte
/// budget.
pub(crate) fn parallel_assembly_allowed<T: FloatT>(
    pool_present: bool,
    contribution_cells: u128,
    schur_stored: u128,
) -> bool {
    let budget_cells = (parallel_assembly_budget_bytes() / std::mem::size_of::<T>()).max(1) as u128;
    pool_present && (contribution_cells <= 2 * schur_stored || contribution_cells <= budget_cells)
}

/// Per-iteration flop estimates for a PSD formulation, from block sizes and
/// column counts only (no coefficients, no thread count).
///
/// `blocks` holds `(p, m)` per PSD cone: side `p` and the number of
/// variables whose columns touch the cone. `n` variables, `retained` rows
/// kept in the reduced system, `schur_fill` the structural fraction of the
/// dense Schur upper triangle.
/// - condensed: per cone `m p^3` (the congruences `W A_j W`) plus
///   `m^2 p^2 / 2` (their inner products), then the reduced factor
///   `(n + retained)^3 / 3` scaled by the Schur fill;
/// - augmented: per cone the dense scaling block `P^3 / 3` with
///   `P = p (p + 1) / 2`, plus its coupling `m P^2` to the variables;
/// - scaling: `10 p^3` per cone (eigen/SVD), common to both forms.
pub(crate) fn psd_form_costs(
    blocks: &[(usize, usize)],
    n: usize,
    retained: usize,
    schur_fill: f64,
) -> (f64, f64, f64) {
    let (mut condensed, mut augmented, mut scaling) = (0f64, 0f64, 0f64);
    for &(p, m) in blocks {
        let (p, m) = (p as f64, m as f64);
        let big = p * (p + 1.0) / 2.0;
        condensed += m * p * p * p + m * m * p * p / 2.0;
        augmented += big * big * big / 3.0 + m * big * big;
        scaling += 10.0 * p * p * p;
    }
    let reduced = (n + retained) as f64;
    condensed += reduced * reduced * reduced / 3.0 * schur_fill.clamp(0.0, 1.0);
    (condensed, augmented, scaling)
}

/// Number of distinct columns of `A` with a nonzero in each row range.
pub(crate) fn columns_touching<T: FloatT>(A: &CscMatrix<T>, ranges: &[Range<usize>]) -> Vec<usize> {
    let mut owner = vec![usize::MAX; A.m];
    for (k, rows) in ranges.iter().enumerate() {
        owner[rows.clone()].fill(k);
    }
    let mut count = vec![0usize; ranges.len()];
    let mut last = vec![usize::MAX; ranges.len()];
    for col in 0..A.n {
        for p in A.colptr[col]..A.colptr[col + 1] {
            let k = owner[A.rowval[p]];
            if k != usize::MAX && last[k] != col {
                last[k] = col;
                count[k] += 1;
            }
        }
    }
    count
}

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
    if A.n == 0 || psd < 256 {
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
    if 6 * reduced_cells + factor_cells >= 3 * augmented_cells {
        return false;
    }
    // Storage allows both: take the cheaper per-iteration flops. This
    // replaces the former `reduced > psd / 4` gate, which sent SDP_qap5 and
    // SDP_qap6 to the augmented form at 2-3x the time.
    let mut ranges = Vec::new();
    let mut sides = Vec::new();
    for (cone, rows) in cones.iter().zip(&cones.rng_cones) {
        if let SupportedCone::PSDTriangleCone(c) = cone {
            ranges.push(rows.clone());
            sides.push(c.n);
        }
    }
    let counts = columns_touching(A, &ranges);
    let blocks: Vec<(usize, usize)> = sides.into_iter().zip(counts).collect();
    let fill = schur_upper as f64 / dense_upper.max(1) as f64;
    let (condensed, augmented, _) = psd_form_costs(&blocks, A.n, reduced - A.n, fill);
    if crate::receipt::profile_requested() {
        eprintln!("KKT_FORM condensed_flops={condensed:.3e} augmented_flops={augmented:.3e}");
    }
    condensed <= augmented
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
    // Global Schur value index of each pair (b, a <= b); u32 halves the
    // largest per-iteration index stream (checked when the pattern is built).
    // Empty when the Schur is a full upper triangle, indexed from `index`.
    schur_positions: Vec<u32>,
}

impl Column {
    #[inline(always)]
    fn schur_position(&self, left: &Self, a: usize) -> usize {
        if self.schur_positions.is_empty() {
            triangular_number(self.index.max(left.index)) + self.index.min(left.index)
        } else {
            self.schur_positions[a] as usize
        }
    }
}

struct PsdBlock<T> {
    // Fixed-precision immutable constant shared by serial and parallel assembly.
    sqrt2: T,
    R: Matrix<T>,
    Rinv: Matrix<T>,
    G: Matrix<T>,
    Ginv: Matrix<T>,
    // Residues of Rinv / G / Ginv reused by every congruence of an iteration.
    rinv_cache: ResidueCache,
    g_cache: ResidueCache,
    ginv_cache: ResidueCache,
    // Fused sampled recovery workspace; empty for every other block.
    mat3c: Matrix<T>,
    columns: Vec<Column>,
    schur_values: Vec<T>,
    sampled: Option<SampledPsd<T>>,
    sparse_column_lanes: Vec<usize>,
    dense_indices: Vec<usize>,
    column_groups: Vec<Vec<usize>>,
    dense_representatives: Vec<usize>,
    coefficient_plan_valid: bool,
    dense_column_map: Vec<usize>,
    dense_vectors: Vec<T>,
    dense_row_first: Vec<usize>,
    dense_row_offsets: Vec<usize>,
    dense_acc: Vec<T>,
    coefficient_support: Vec<Vec<usize>>,
    // One reusable panel set per dense-transform lane (see `transform_chunk`).
    transform_lanes: Vec<TransformPanels<T>>,
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
    adjoint: Vec<T>,
}

// Immutable scaling snapshots with private arithmetic scratch. These are
// operator data, not another cone/solver state machine. Retained cone Hessians
// use the same formulas as their existing mul_Hs implementations.
/// Second-order cones up to this dimension are eliminated into the Schur
/// complement through their explicit `W⁻¹` instead of being retained.
fn eliminated_soc(dim: usize) -> bool {
    dim <= 16
}

/// Eliminated rows `B` of orthants (`A_r/w_r`) and small second-order cones
/// (`W⁻¹A_k`), whose Schur contribution is `BᵀB`. Columns present in at least
/// half of the rows form a dense panel handled by one SYRK; every other
/// product goes through a precomputed per-position plan.
struct EliminatedRows<T> {
    /// Per source block: orthant rows or one SOC; entries index `values`.
    sources: Vec<Source>,
    values: Vec<T>,
    /// Virtual rows as `values` ranges, with their columns.
    rows: Vec<(usize, usize)>,
    dense: Vec<usize>,
    /// Panel cells `(row + slot·rows, value index)` of the dense entries of
    /// rows that also have sparse entries (their values feed the plan too).
    cells: Vec<(usize, usize)>,
    /// Panel cell of each value index (`usize::MAX` off the panel), and
    /// whether a virtual row lies entirely in the panel (written directly).
    cell_of: Vec<usize>,
    panel_only: Vec<bool>,
    panel: Matrix<T>,
    gram: Matrix<T>,
    out: Vec<usize>,
    plan: OrthantPlan,
    /// Wide precision: panel-only orthant rows keep their constant A rows and
    /// enter as `Aᵀdiag(1/w²)A` with cached residues.
    fixed: Option<FixedRows<T>>,
}

struct FixedRows<T> {
    /// `(block, orthant row)` of each fixed row, and whether a virtual row
    /// is fixed.
    w_index: Vec<(usize, usize)>,
    member: Vec<bool>,
    /// Column-major `fixed rows × panel columns` copy of A (filled on first
    /// assembly; the solver rebuilds this plan when A changes).
    a: Vec<T>,
    weights: Vec<T>,
    gram: Matrix<T>,
    cache: ResidueCache,
    filled: bool,
}

enum Source {
    /// Orthant block `block`: row `r` uses `positions[r]` (A positions).
    Orthant {
        block: usize,
        first: usize,
        positions: Vec<Vec<usize>>,
    },
    /// SOC block `block` with `dim` rows over the union of its columns;
    /// `positions[r·width + c]` is the A position or `usize::MAX`.
    Soc {
        block: usize,
        first: usize,
        width: usize,
        positions: Vec<usize>,
    },
}

impl<T: FloatT> EliminatedRows<T> {
    fn new(blocks: &[Block<T>], position: impl Fn(usize, usize) -> usize) -> Self {
        let (mut sources, mut rows, mut cols) = (Vec::new(), Vec::new(), Vec::new());
        let mut orthant_of: Vec<Option<(usize, usize)>> = Vec::new();
        for (bi, block) in blocks.iter().enumerate() {
            match &block.scaling {
                Scaling::Orthant { rows: entries, .. } => {
                    let first = rows.len();
                    for (r, row) in entries.iter().enumerate() {
                        rows.push((cols.len(), cols.len() + row.len()));
                        cols.extend(row.iter().map(|&(j, _)| j));
                        orthant_of.push(Some((bi, r)));
                    }
                    let positions = entries
                        .iter()
                        .map(|row| row.iter().map(|&(_, q)| q).collect())
                        .collect();
                    sources.push(Source::Orthant {
                        block: bi,
                        first,
                        positions,
                    });
                }
                Scaling::SocElim { rows: entries, .. } => {
                    let mut union: Vec<usize> = entries.iter().flatten().map(|&(j, _)| j).collect();
                    union.sort_unstable();
                    union.dedup();
                    let width = union.len();
                    let mut positions = vec![usize::MAX; entries.len() * width];
                    for (r, row) in entries.iter().enumerate() {
                        for &(j, q) in row {
                            positions[r * width + union.binary_search(&j).unwrap()] = q;
                        }
                    }
                    let first = rows.len();
                    for _ in 0..entries.len() {
                        rows.push((cols.len(), cols.len() + width));
                        cols.extend(&union);
                        orthant_of.push(None);
                    }
                    sources.push(Source::Soc {
                        block: bi,
                        first,
                        width,
                        positions,
                    });
                }
                _ => {}
            }
        }
        let n = cols.iter().max().map_or(0, |&j| j + 1);
        let mut count = vec![0usize; n];
        for &j in &cols {
            count[j] += 1;
        }
        let threshold = (rows.len() / 2).max(16);
        let dense: Vec<usize> = (0..n).filter(|&j| count[j] >= threshold).collect();
        let mut slot = vec![usize::MAX; n];
        for (d, &j) in dense.iter().enumerate() {
            slot[j] = d;
        }
        let out = (0..dense.len())
            .flat_map(|c| (0..=c).map(move |i| (i, c)))
            .map(|(i, c)| position(dense[i], dense[c]))
            .collect();
        // Every product with at least one non-panel column.
        let mut triples = Vec::new();
        for &(start, end) in &rows {
            for b in start..end {
                for a in start..=b {
                    let (i, j) = (cols[a], cols[b]);
                    if slot[i] != usize::MAX && slot[j] != usize::MAX {
                        continue;
                    }
                    triples.push((position(i.min(j), i.max(j)), a as u32, b as u32));
                }
            }
        }
        let in_panel = |e: usize| slot[cols[e]] != usize::MAX;
        let panel_only: Vec<bool> = rows
            .iter()
            .map(|&(start, end)| !dense.is_empty() && (start..end).all(in_panel))
            .collect();
        let fixed_row: Vec<bool> = (0..rows.len())
            .map(|i| T::precision_bits() > 64 && panel_only[i] && orthant_of[i].is_some())
            .collect();
        // Panel rows (rebuilt every assembly) and fixed rows are numbered apart.
        let mut index = vec![usize::MAX; rows.len()];
        let (mut k, mut k_fixed) = (0, 0);
        for i in 0..rows.len() {
            if dense.is_empty() {
                continue;
            }
            if fixed_row[i] {
                index[i] = k_fixed;
                k_fixed += 1;
            } else {
                index[i] = k;
                k += 1;
            }
        }
        let mut cell_of = vec![usize::MAX; cols.len()];
        for (i, &(start, end)) in rows.iter().enumerate() {
            let height = if fixed_row[i] { k_fixed } else { k };
            for e in start..end {
                if in_panel(e) {
                    cell_of[e] = index[i] + slot[cols[e]] * height;
                }
            }
        }
        let fixed = (k_fixed > 0).then(|| {
            let members: Vec<usize> = (0..rows.len()).filter(|&i| fixed_row[i]).collect();
            FixedRows {
                w_index: members.iter().map(|&i| orthant_of[i].unwrap()).collect(),
                member: fixed_row.clone(),
                a: vec![T::zero(); k_fixed * dense.len()],
                weights: vec![T::zero(); k_fixed],
                gram: Matrix::zeros((dense.len(), dense.len())),
                cache: ResidueCache::default(),
                filled: false,
            }
        });
        let cells = rows
            .iter()
            .enumerate()
            .filter(|&(i, _)| !panel_only[i] && !fixed_row[i])
            .flat_map(|(_, &(start, end))| start..end)
            .filter(|&e| cell_of[e] != usize::MAX)
            .map(|e| (cell_of[e], e))
            .collect();
        Self {
            sources,
            values: vec![T::zero(); cols.len()],
            rows,
            panel: Matrix::zeros((k, dense.len())),
            gram: Matrix::zeros((dense.len(), dense.len())),
            dense,
            cells,
            cell_of,
            panel_only,
            out,
            plan: OrthantPlan::from_triples(triples),
            fixed,
        }
    }

    /// `schur += BᵀB` for the current scalings.
    fn accumulate(
        &mut self,
        schur: &mut [T],
        blocks: &[Block<T>],
        a: &[T],
        pool: Option<&rayon::ThreadPool>,
    ) {
        for source in &self.sources {
            match source {
                Source::Orthant {
                    block,
                    first,
                    positions,
                } => {
                    let Scaling::Orthant { w, .. } = &blocks[*block].scaling else {
                        unreachable!()
                    };
                    let panel = self.panel.data_mut();
                    for (r, row) in positions.iter().enumerate() {
                        let start = self.rows[first + r].0;
                        if let Some(fixed) = self.fixed.as_mut().filter(|f| f.member[first + r]) {
                            if !fixed.filled {
                                for (&cell, &q) in self.cell_of[start..].iter().zip(row) {
                                    fixed.a[cell] = a[q];
                                }
                            }
                            continue;
                        }
                        let inv = w[r].recip();
                        if self.panel_only[first + r] {
                            for (&cell, &q) in self.cell_of[start..].iter().zip(row) {
                                panel[cell] = a[q] * inv;
                            }
                        } else {
                            for (v, &q) in self.values[start..].iter_mut().zip(row) {
                                *v = a[q] * inv;
                            }
                        }
                    }
                }
                Source::Soc {
                    block,
                    first,
                    width,
                    positions,
                } => {
                    let Scaling::SocElim { w, eta, .. } = &blocks[*block].scaling else {
                        unreachable!()
                    };
                    // W⁻¹ = η⁻¹[w0, -w1ᵀ; -w1, I + w1w1ᵀ/(1+w0)], column by column.
                    let dim = w.len();
                    let inv = eta.recip();
                    let tail = (T::one() + w[0]).recip();
                    let start = self.rows[*first].0;
                    let entry = |r: usize, c: usize| match positions[r * width + c] {
                        usize::MAX => T::zero(),
                        q => a[q],
                    };
                    for c in 0..*width {
                        let a0 = entry(0, c);
                        let mut t = T::zero();
                        for r in 1..dim {
                            t = w[r].mul_add(entry(r, c), t);
                        }
                        self.values[start + c] = (w[0] * a0 - t) * inv;
                        let shift = t * tail - a0;
                        for r in 1..dim {
                            self.values[start + r * width + c] =
                                w[r].mul_add(shift, entry(r, c)) * inv;
                        }
                    }
                }
            }
        }
        if let Some(fixed) = self.fixed.as_mut() {
            fixed.filled = true;
            for (v, &(block, r)) in fixed.weights.iter_mut().zip(&fixed.w_index) {
                let Scaling::Orthant { w, .. } = &blocks[block].scaling else {
                    unreachable!()
                };
                let inv = w[r].recip();
                *v = inv * inv;
            }
            let d = self.dense.len();
            let k = fixed.weights.len();
            if !T::diag_congruence_upper_exact(
                d,
                k,
                &fixed.a,
                &fixed.weights,
                fixed.gram.data_mut(),
                pool,
                &mut fixed.cache,
            ) {
                // Declined: scale the rows and use the exact SYRK.
                let mut b = Matrix::zeros((k, d));
                for c in 0..d {
                    for i in 0..k {
                        let (block, r) = fixed.w_index[i];
                        let Scaling::Orthant { w, .. } = &blocks[block].scaling else {
                            unreachable!()
                        };
                        b[(i, c)] = fixed.a[i + c * k] * w[r].recip();
                    }
                }
                fixed
                    .gram
                    .syrk(&b.t(), T::one(), T::zero(), MatrixTriangle::Triu);
            }
            let mut p = 0;
            for c in 0..d {
                for i in 0..=c {
                    schur[self.out[p]] += fixed.gram[(i, c)];
                    p += 1;
                }
            }
        }
        if self.panel.nrows() > 0 {
            let d = self.dense.len();
            // Written cells are structurally fixed and fully rewritten; all
            // others stay zero from construction.
            let panel = self.panel.data_mut();
            for &(cell, e) in &self.cells {
                panel[cell] = self.values[e];
            }
            self.gram
                .syrk(&self.panel.t(), T::one(), T::zero(), MatrixTriangle::Triu);
            let mut p = 0;
            for c in 0..d {
                for i in 0..=c {
                    schur[self.out[p]] += self.gram[(i, c)];
                    p += 1;
                }
            }
        }
        if T::precision_bits() <= 53 {
            self.plan.accumulate_rounded(schur, &self.values);
        } else {
            self.plan.accumulate(schur, &self.values, pool);
        }
    }
}

/// Orthant Schur terms grouped by destination: group `g` adds the products
/// `scaled[a]·scaled[b]` for `(a, b)` in `terms[starts[g].1..starts[g + 1].1]`
/// to Schur value `starts[g].0`.
#[derive(Default)]
struct OrthantPlan {
    starts: Vec<(usize, usize)>,
    terms: Vec<(u32, u32)>,
}

impl OrthantPlan {
    fn from_triples(mut triples: Vec<(usize, u32, u32)>) -> Self {
        triples.sort_unstable();
        let mut starts = Vec::new();
        for (k, &(p, _, _)) in triples.iter().enumerate() {
            if starts.last().is_none_or(|&(q, _)| q != p) {
                starts.push((p, k));
            }
        }
        starts.push((usize::MAX, triples.len()));
        let terms = triples.into_iter().map(|(_, a, b)| (a, b)).collect();
        Self { starts, terms }
    }

    /// Binary64: `values[p] += Σ scaled[a]·scaled[b]` as an FMA chain.
    fn accumulate_rounded<T: FloatT>(&self, values: &mut [T], scaled: &[T]) {
        for g in 0..self.starts.len().saturating_sub(1) {
            let p = self.starts[g].0;
            let mut v = values[p];
            for &(a, b) in &self.terms[self.starts[g].1..self.starts[g + 1].1] {
                v = scaled[a as usize].mul_add(scaled[b as usize], v);
            }
            values[p] = v;
        }
    }

    /// `values[p] += Σ scaled[a]·scaled[b]`, one exact accumulation per entry.
    fn accumulate<T: FloatT>(
        &self,
        values: &mut [T],
        scaled: &[T],
        pool: Option<&rayon::ThreadPool>,
    ) {
        let one = T::one();
        let groups = self.starts.len() - 1;
        let entry = |g: usize, current: T| {
            let range = self.starts[g].1..self.starts[g + 1].1;
            T::dot_fma(
                std::iter::once((&current, &one)).chain(
                    self.terms[range]
                        .iter()
                        .map(|&(a, b)| (&scaled[a as usize], &scaled[b as usize])),
                ),
            )
        };
        match pool.filter(|p| p.current_num_threads() > 1) {
            Some(pool) => {
                let sums: Vec<T> = pool.install(|| {
                    (0..groups)
                        .into_par_iter()
                        .map(|g| entry(g, values[self.starts[g].0]))
                        .collect()
                });
                for (g, v) in sums.into_iter().enumerate() {
                    values[self.starts[g].0] = v;
                }
            }
            None => {
                for g in 0..groups {
                    let p = self.starts[g].0;
                    values[p] = entry(g, values[p]);
                }
            }
        }
    }
}

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
    /// An eliminated second-order cone: its rows enter the Schur complement
    /// as `W⁻¹A_k` (entries `(column, A position)` per row).
    SocElim {
        w: Vec<T>,
        eta: T,
        rows: Vec<Vec<(usize, usize)>>,
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
    /// Last measured seconds per scaling action (condense, recover, apply),
    /// used to give expensive blocks extra workers (SDPB-style measured
    /// load balancing); 0 until the action has run once.
    cost: [f64; 3],
    /// Workers granted to this block for the next call of each action.
    ways: [usize; 3],
}

pub(crate) struct CondensedKKTSolver<T: FloatT> {
    n: usize,
    P: CscMatrix<T>,
    A: CscMatrix<T>,
    blocks: Vec<Block<T>>,
    /// Schur plan of the orthant and eliminated SOC rows (first assembly).
    eliminated: Option<EliminatedRows<T>>,
    /// Binary64 products with A through a dense panel of its dense columns.
    a_panel: Option<DenseColumns<T>>,
    retained_indices: Vec<usize>,
    retained_rows: Vec<usize>,
    retained_positions: Vec<usize>,
    retained_A: CscMatrix<T>,
    schur_nnz: usize,
    reduced: DirectLDLKKTSolver<T>,
    b: Vec<T>,
    x: Vec<T>,
    error: Vec<T>,
    candidate: Vec<T>,
    /// Continue stalled outer refinements with GMRES-IR (`refine_further`).
    gmres_continuation: bool,
    batch_rhs: Vec<T>,
    batch_out: Vec<T>,
    batch_halves: Vec<T>,
    scaled_solutions: Vec<T>,
    scaled_valid: Vec<bool>,
    workx: Vec<T>,
    workz: Vec<T>,
    workh: Vec<T>,
    // The last aligned MPI residual left only this rank's rows of workh.
    workh_partial: bool,
    // Rows holding sampled linear entries outside zero blocks (cached):
    // the aligned prepare republishes their scaled values.
    linear_scaled_rows: Option<Vec<usize>>,
    retained_rhs: Vec<T>,
    local_only: bool,
    pool: Option<Arc<rayon::ThreadPool>>,
    parallel_assembly: bool,
    plan_threads: usize,
    scaling_lanes: Vec<usize>,
    scaling_tiles: usize,
    /// Workers the block phases plan for: the pool width, or this owner's
    /// share of a pool shared by several in-process owners.
    scaling_workers: usize,
    worker_share: usize,
    inner_schur: bool,
    inner_sampled: Option<usize>,
    // Owner-local kernels normally leave inner lanes disabled while every
    // owner task occupies the shared pool.  A dominant owner may opt in to
    // the same heavy-tail thresholds used by the monolithic planner.
    owner_inner_admission: bool,
    sampled: Option<(Arc<SampledOperator<T>>, SampledWorkspace<T>)>,
    sparse_products: Option<crate::algebra::sparse_parallel::SparseParallel>,
    counters: crate::solver::kkt::SolveCounters,
}

fn local_world(local_only: bool) -> Option<crate::mpi::World> {
    if local_only {
        None
    } else {
        crate::mpi::World::get()
    }
}
fn local_success(local_only: bool, value: bool) -> bool {
    if local_only {
        value
    } else {
        crate::mpi::all_succeeded(value)
    }
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
        Self::new_partition_policy(P, A, types, cones, settings, true, false, true)
    }

    /// [`Self::new`] for a problem whose every PSD cone is a sampled block
    /// and whose other cones are zero or nonnegative cones: products then use
    /// the sampled factors, so the solver keeps values for non-PSD rows only.
    pub(crate) fn new_fully_sampled(
        P: &CscMatrix<T>,
        A: &CscMatrix<T>,
        types: &[SupportedConeT<T>],
        cones: &CompositeCone<T>,
        settings: &CoreSettings<T>,
    ) -> Self {
        Self::new_partition_policy(P, A, types, cones, settings, true, false, false)
    }
    pub(crate) fn new_local_partition(
        P: &CscMatrix<T>,
        A: &CscMatrix<T>,
        types: &[SupportedConeT<T>],
        cones: &CompositeCone<T>,
        settings: &CoreSettings<T>,
        keep_equalities: bool,
    ) -> Self {
        Self::new_partition_policy(P, A, types, cones, settings, keep_equalities, true, true)
    }
    fn mpi_world(&self) -> Option<crate::mpi::World> {
        local_world(self.local_only)
    }
    fn new_partition_policy(
        P: &CscMatrix<T>,
        A: &CscMatrix<T>,
        types: &[SupportedConeT<T>],
        cones: &CompositeCone<T>,
        settings: &CoreSettings<T>,
        keep_equalities: bool,
        local_only: bool,
        keep_values: bool,
    ) -> Self {
        debug_assert!(
            !local_only || cones.is_local_only(),
            "local KKT requires local cone policy"
        );
        let (n, m) = (A.n, A.m);
        assert!(
            n > 0 || !keep_equalities,
            "condensed KKT requires primal variables"
        );
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
        // Small second-order cones are eliminated like orthants, through W⁻¹.
        let mut row_soc = vec![u32::MAX; m];
        let mut soc_starts: Vec<usize> = Vec::new();
        let mut soc_entries: Vec<Vec<Vec<(usize, usize)>>> = Vec::new();
        // Pass 1: classify cones and publish row ownership.
        for (ci, (cone, rows)) in cones.iter().zip(&cones.rng_cones).enumerate() {
            match cone {
                SupportedCone::PSDTriangleCone(c) => {
                    let psd = psd_starts.len();
                    row_psd[rows.clone()].fill(psd as u32);
                    psd_starts.push(rows.start);
                    psd_numels.push(rows.len());
                    if keep_values {
                        let mut coordinates = Vec::with_capacity(rows.len());
                        for j in 0..c.n {
                            for i in 0..=j {
                                coordinates.push((i, j));
                            }
                        }
                        psd_coordinates.push(coordinates);
                    }
                    psd_columns.push(Vec::new());
                }
                SupportedCone::NonnegativeCone(_) => {
                    let orthant = orthant_starts.len();
                    row_orthant[rows.clone()].fill(orthant as u32);
                    orthant_starts.push(rows.start);
                    orthant_entries.push(vec![Vec::new(); rows.len()]);
                }
                SupportedCone::SecondOrderCone(c) if eliminated_soc(c.dim) => {
                    let soc = soc_starts.len();
                    row_soc[rows.clone()].fill(soc as u32);
                    soc_starts.push(rows.start);
                    soc_entries.push(vec![Vec::new(); rows.len()]);
                }
                SupportedCone::ZeroCone(_) if !keep_equalities => {}
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
        // A structure-only build keeps values just for non-PSD rows, in a
        // compact copy that orthant entries address.
        let mut compact_colptr = vec![0];
        let mut compact_rowval = Vec::new();
        let mut compact_nzval = Vec::new();
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
                    // Entries address A's values; a structure-only (fully
                    // sampled) build keeps none, and the sampled operator
                    // replaces them, so record only the column pattern.
                    if keep_values {
                        let (i, j) = psd_coordinates[psd][row - psd_starts[psd]];
                        psd_columns[psd]
                            .last_mut()
                            .expect("column pushed above")
                            .entries
                            .push(Entry { position: p, i, j });
                    }
                    continue;
                }
                let position = if keep_values {
                    p
                } else {
                    compact_rowval.push(row);
                    compact_nzval.push(A.nzval[p]);
                    compact_rowval.len() - 1
                };
                let orthant = row_orthant[row];
                if orthant != u32::MAX {
                    let orthant = orthant as usize;
                    orthant_entries[orthant][row - orthant_starts[orthant]].push((col, position));
                }
                let soc = row_soc[row];
                if soc != u32::MAX {
                    let soc = soc as usize;
                    soc_entries[soc][row - soc_starts[soc]].push((col, position));
                }
            }
            compact_colptr.push(compact_rowval.len());
        }
        // Pass 3: materialize the per-cone scaling structures.
        let mut psd_taken = psd_columns.into_iter().zip(psd_numels);
        let mut orthant_taken = orthant_entries.into_iter();
        let mut soc_taken = soc_entries.into_iter();
        for (ci, cone) in cones.iter().enumerate() {
            let scaling = match cone {
                SupportedCone::PSDTriangleCone(c) => {
                    let (columns, numel) = psd_taken.next().expect("one column set per PSD cone");
                    Scaling::Psd(PsdBlock::from_columns(c.n, numel, columns, keep_values))
                }
                SupportedCone::NonnegativeCone(c) => {
                    let entries = orthant_taken.next().expect("one entry set per orthant");
                    Scaling::Orthant {
                        w: c.w.clone(),
                        rows: entries,
                    }
                }
                SupportedCone::ZeroCone(_) => Scaling::Zero,
                SupportedCone::SecondOrderCone(c) if eliminated_soc(c.dim) => Scaling::SocElim {
                    w: c.w.clone(),
                    eta: c.η,
                    rows: soc_taken.next().expect("one entry set per eliminated SOC"),
                },
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
                cost: [0.0; 3],
                ways: [1; 3],
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
        // Every PSD block's column set and every orthant row is a clique; P adds
        // its own lower pattern. Column j gathers rows i <= j of the cliques it
        // belongs to, stamped for dedup and sorted.
        let mut cliques: Vec<Vec<usize>> = Vec::new();
        for block in &blocks {
            match &block.scaling {
                Scaling::Psd(p) => cliques.push(p.columns.iter().map(|c| c.index).collect()),
                Scaling::Orthant { rows, .. } => {
                    cliques.extend(rows.iter().map(|row| row.iter().map(|&(j, _)| j).collect()))
                }
                Scaling::SocElim { rows, .. } => {
                    let mut union: Vec<usize> = rows.iter().flatten().map(|&(j, _)| j).collect();
                    union.sort_unstable();
                    union.dedup();
                    cliques.push(union);
                }
                _ => {}
            }
        }
        let mut column_cliques: Vec<Vec<u32>> = vec![Vec::new(); n];
        for (c, clique) in cliques.iter_mut().enumerate() {
            clique.sort_unstable();
            for &j in clique.iter() {
                column_cliques[j].push(c as u32);
            }
        }
        // Two passes: count, then fill an exactly sized `rowval` (the Schur
        // pattern is the largest setup array; no doubling slack at the peak).
        let mut stamp = vec![usize::MAX; n];
        let mut colptr = vec![0; n + 1];
        let mut rowval = Vec::new();
        for fill in [false, true] {
            if fill {
                rowval.reserve_exact(colptr[n]);
                stamp.fill(usize::MAX);
            }
            for j in 0..n {
                let start = rowval.len();
                let mut len = 0;
                let mut add = |i: usize| {
                    if stamp[i] != j {
                        stamp[i] = j;
                        len += 1;
                        if fill {
                            rowval.push(i);
                        }
                    }
                };
                add(j);
                for p in P.colptr[j]..P.colptr[j + 1] {
                    if P.rowval[p] <= j {
                        add(P.rowval[p]);
                    }
                }
                for &c in &column_cliques[j] {
                    for &i in cliques[c as usize].iter().take_while(|&&i| i <= j) {
                        add(i);
                    }
                }
                if fill {
                    rowval[start..].sort_unstable();
                } else {
                    colptr[j + 1] = colptr[j] + len;
                }
            }
        }
        drop((cliques, column_cliques, stamp));
        let count = rowval.len();
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
        // Extra per-block contribution storage. The buffers let independent
        // cones assemble on the pool while the publish stays in cone order
        // (bitwise identical to the serial scatter), so they are used when
        // they fit either the original two-copy rule or the byte budget.
        // Overlapping cliques on few dense blocks previously always took the
        // allocation-free serial path; the budget lets those multi-cone models
        // reach the cone pool without unbounded scratch.
        let parallel_assembly =
            parallel_assembly_allowed::<T>(pool.is_some(), contribution_cells, count as u128);
        assert!(
            count <= u32::MAX as usize,
            "Schur pattern exceeds u32 positions"
        );
        let dense_schur = count == triangular_number(n);
        let mut position_of = vec![0u32; if dense_schur { 0 } else { n }];
        for block in &mut blocks {
            if let Scaling::Psd(p) = &mut block.scaling {
                if parallel_assembly {
                    p.schur_values
                        .resize(triangular_number(p.columns.len()), T::zero());
                }
                if dense_schur {
                    continue;
                }
                // Pair (b, a <= b) lives in Schur column max(index) at row
                // min(index). Visit columns by index and scatter each Schur
                // column once instead of binary searching every pair.
                let index: Vec<usize> = p.columns.iter().map(|c| c.index).collect();
                let mut positions: Vec<Vec<u32>> =
                    (0..index.len()).map(|b| vec![0; b + 1]).collect();
                let mut order: Vec<usize> = (0..index.len()).collect();
                order.sort_unstable_by_key(|&c| index[c]);
                for (t, &c) in order.iter().enumerate() {
                    let j = index[c];
                    for q in schur.colptr[j]..schur.colptr[j + 1] {
                        position_of[schur.rowval[q]] = q as u32;
                    }
                    for &m in &order[..=t] {
                        positions[c.max(m)][c.min(m)] = position_of[index[m]];
                    }
                }
                for (column, positions) in p.columns.iter_mut().zip(positions) {
                    column.schur_positions = positions;
                }
            }
        }
        let retained_cones = CompositeCone::new(&retained_types);
        let mut reduced =
            DirectLDLKKTSolver::new(&schur, &retained_A, &retained_cones, nr, n, settings);
        // The reduced system is replicated; a structured backend shares its
        // factorization work over the ranks.
        reduced.set_world(local_world(local_only));
        reduced.set_factor_pool(pool.clone());
        reduced.set_residual_pool(pool.clone());
        let kkt = reduced.kkt_matrix_mut();
        debug_assert_eq!(kkt.colptr[..n + 1], schur.colptr);
        debug_assert_eq!(kkt.rowval[..count], schur.rowval);
        drop(schur);
        let mut solver = Self {
            local_only,
            n,
            P: P.clone(),
            A: if keep_values {
                A.clone()
            } else {
                // Structure only (see `new_fully_sampled`): the sampled
                // operator supplies PSD rows, so only other rows keep values.
                CscMatrix::new(A.m, A.n, compact_colptr, compact_rowval, compact_nzval)
            },
            blocks,
            eliminated: None,
            a_panel: if keep_values {
                DenseColumns::new(A)
            } else {
                None
            },
            retained_indices,
            retained_rows,
            retained_positions,
            retained_A,
            schur_nnz: count,
            reduced,
            b: vec![T::zero(); n + m],
            x: vec![T::zero(); n + m],
            error: vec![T::zero(); n + m],
            candidate: vec![T::zero(); n + m],
            gmres_continuation: false,
            batch_rhs: Vec::new(),
            batch_out: Vec::new(),
            batch_halves: Vec::new(),
            scaled_solutions: Vec::new(),
            scaled_valid: Vec::new(),
            workx: vec![T::zero(); n],
            workz: vec![T::zero(); m],
            workh: vec![T::zero(); m],
            workh_partial: false,
            linear_scaled_rows: None,
            retained_rhs: vec![T::zero(); nr],
            pool,
            parallel_assembly,
            plan_threads: 0,
            scaling_lanes: Vec::new(),
            scaling_tiles: 1,
            scaling_workers: 1,
            worker_share: 0,
            inner_schur: false,
            inner_sampled: None,
            owner_inner_admission: false,
            sampled: None,
            sparse_products: (local_world(local_only).is_some()
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
    #[allow(clippy::too_many_arguments)]
    fn sparse_gemv(
        world: Option<crate::mpi::World>,
        plan: Option<&crate::algebra::sparse_parallel::SparseParallel>,
        a: &CscMatrix<T>,
        panel: Option<&DenseColumns<T>>,
        transpose: bool,
        y: &mut [T],
        x: &[T],
        alpha: T,
        beta: T,
    ) {
        if let (None, Some(panel)) = (world, panel) {
            panel.gemv(transpose, y, x, alpha, beta);
        } else if let (Some(world), Some(plan)) = (world, plan) {
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

    /// Setup-only reservation for later use of an external shared pool. The
    /// same contribution allowance as construction applies. Existing
    /// allocations survive pool removal.
    pub(crate) fn prepare_shared_pool(&mut self) {
        let cells: u128 = self
            .blocks
            .iter()
            .map(|b| match &b.scaling {
                Scaling::Psd(p) => triangular_number(p.columns.len()) as u128,
                _ => 0,
            })
            .sum();
        if parallel_assembly_allowed::<T>(true, cells, self.schur_nnz as u128) {
            for block in &mut self.blocks {
                if let Scaling::Psd(p) = &mut block.scaling {
                    p.schur_values
                        .resize(triangular_number(p.columns.len()), T::zero());
                }
            }
        }
        self.plan_threads = 0;
    }

    /// Permit this owner-local kernel to use the existing shared pool for a
    /// dominant inner block.  The planner still applies the worker-count,
    /// minimum-work and 75% dominance thresholds; this flag only removes the
    /// outer-task guard that assumes all PSD blocks are equally costly.
    pub(crate) fn set_owner_inner_admission(&mut self, admitted: bool) {
        self.owner_inner_admission = admitted;
        self.plan_threads = 0;
    }

    /// Plan this kernel's block phases for `share` workers of the pool it is
    /// given (0: the whole pool). In-process owners share one pool; each
    /// planning for its full width floods it with tiles and residue ways.
    pub(crate) fn set_worker_share(&mut self, share: usize) {
        self.worker_share = share;
        self.plan_threads = 0;
    }

    fn refresh_parallel_plan(&mut self) {
        let mut workers = self.pool.as_ref().map_or(1, |p| p.current_num_threads());
        if self.worker_share > 0 {
            workers = workers.min(self.worker_share);
        }
        if workers == self.plan_threads {
            return;
        }
        self.plan_threads = workers;
        self.scaling_workers = workers;
        let costs: Vec<_> = self
            .blocks
            .iter()
            .map(|block| match &block.scaling {
                Scaling::Psd(p) => 4 * (p.Rinv.size().0 as u128).pow(3),
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
        // A heavy owner is the sole exception to the one-lane-per-block
        // admission rule.  Keep one worker serial even when the owner is
        // marked heavy; the inner thresholds below cannot create useful lanes
        // on a one-thread pool.
        let allow_inner = workers > 1 && (spare_workers || self.owner_inner_admission);
        let (mut largest_sparse, mut total_work) = (0u128, 0u128);
        let mut dominant_sampled = None;
        let words = T::precision_bits().div_ceil(64) as u128;
        for block in &mut self.blocks {
            if let Scaling::Psd(p) = &mut block.scaling {
                if let Some(sampled) = &mut p.sampled {
                    let dense =
                        sampled
                            .work
                            .configure_parallel(if allow_inner { workers } else { 1 });
                    let costs: Vec<_> = (1..=p.columns.len())
                        .map(|n| 8 * n as u128 * words * words)
                        .collect();
                    let pairs: u128 = costs.iter().sum();
                    let lanes = if allow_inner {
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
                        p.configure_sparse_columns(if allow_inner { workers } else { 1 });
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
            .filter(|&(_, work)| allow_inner && work >= 8192 && work * 4 >= total_work * 3)
            .map(|(row, _)| row);
        self.inner_schur = self.sampled.is_none()
            && allow_inner
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
            // Cone-local dense transforms split their chunks across the same
            // pool; chunk windows are disjoint, so the published values are
            // bitwise identical to the serial assembly.
            let assembly_pool = self.pool.as_deref();
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
                        psd.compute_schur_packed(
                            matrix_values,
                            split_columns,
                            &mut output,
                            assembly_pool,
                        );
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
        let pool = self.pool.clone();
        let schur = self.reduced.kkt_matrix_mut();
        match pool.as_deref().filter(|p| p.current_num_threads() > 1) {
            Some(pool) => pool.install(|| {
                schur.nzval[..self.schur_nnz]
                    .par_chunks_mut(8192)
                    .for_each(|chunk| chunk.fill(T::zero()))
            }),
            None => schur.nzval[..self.schur_nnz].fill(T::zero()),
        }
        // A complete upper Schur stores rows 0..=j in column j: address
        // entries directly instead of searching the column.
        let dense = self.schur_nnz == triangular_number(self.n);
        let position = |schur: &CscMatrix<T>, i: usize, j: usize| {
            if dense {
                schur.colptr[j] + i
            } else {
                schur_position(schur, i, j)
            }
        };
        for j in 0..self.n {
            for p in self.P.colptr[j]..self.P.colptr[j + 1] {
                let i = self.P.rowval[p];
                if i <= j {
                    let q = position(schur, i, j);
                    schur.nzval[q] += self.P.nzval[p];
                }
            }
        }
        for block in &mut self.blocks {
            match &mut block.scaling {
                Scaling::Psd(psd) => {
                    if self.parallel_assembly {
                        psd.scatter_schur(schur);
                    } else {
                        psd.compute_schur(&self.A.nzval, |_, _, position, v| {
                            schur.nzval[position] += v;
                        });
                    }
                }
                _ => {}
            }
        }
        // Orthant and eliminated SOC rows form one Gram BᵀB.
        let elim = self
            .eliminated
            .get_or_insert_with(|| EliminatedRows::new(&self.blocks, |i, j| position(schur, i, j)));
        elim.accumulate(
            &mut schur.nzval,
            &self.blocks,
            &self.A.nzval,
            pool.as_deref(),
        );
        schur.nzval[..self.schur_nnz].is_finite()
    }

    fn fused_sampled(&self) -> bool {
        self.sampled.is_some() && self.mpi_world().is_none()
    }

    /// Under MPI, align the sampled products' rank partition with the
    /// scaling partition, so each rank's sampled block rows are exactly the
    /// rows it scales and intermediate vectors need no exchange. Returns the
    /// world when the partitions align (sampled blocks contiguous per rank).
    fn align_sampled_parts(&mut self) -> Option<crate::mpi::World> {
        let world = self.mpi_world()?;
        let (operator, work) = self.sampled.as_ref()?;
        // Partial rows require the sharded adjoint. The aligned prepare keeps
        // scaled rows on their owner, but the column-sharded linear adjoint
        // reads every row holding a linear entry: rows outside zero blocks
        // (whose scaled value is nonzero) are republished by their owners.
        if !work.parallel_eligible(operator, self.pool.as_ref()) {
            return None;
        }
        let scaled_rows = self.linear_scaled_rows.get_or_insert_with(|| {
            let a = operator.linear();
            let mut touched = vec![false; a.m];
            for &r in &a.rowval {
                touched[r] = true;
            }
            for block in &self.blocks {
                if matches!(block.scaling, Scaling::Zero) {
                    touched[block.rows.clone()].fill(false);
                }
            }
            (0..a.m).filter(|&r| touched[r]).collect()
        });
        if scaled_rows.len() * 8 > operator.linear().m
            || self
                .blocks
                .iter()
                .any(|b| matches!(&b.scaling, Scaling::Psd(p) if p.sampled.is_none()))
        {
            return None;
        }
        let nsampled = operator.blocks().len();
        let mut parts = Vec::with_capacity(world.size());
        let mut next = 0usize;
        for (b0, len) in scaling_parts(&self.blocks, world.size()) {
            let first = next;
            for block in &self.blocks[b0..b0 + len] {
                if let Scaling::Psd(p) = &block.scaling {
                    match &p.sampled {
                        Some(s) if s.block == next => next += 1,
                        Some(_) => return None,
                        None => {}
                    }
                }
            }
            parts.push((first, next - first));
        }
        if next != nsampled {
            return None;
        }
        self.sampled.as_mut().unwrap().1.set_rank_parts(Some(parts));
        Some(world)
    }

    pub(crate) fn interior_dimension(&self) -> usize {
        self.reduced.factor_dimension()
    }

    pub(crate) fn interior_counters(&self) -> super::SolveCounters {
        self.reduced.counters()
    }

    pub(crate) fn interior_diagonal_norm(&self) -> T {
        self.reduced.diagonal_norm()
    }

    pub(crate) fn factor_interior(&mut self, settings: &CoreSettings<T>, shift: T) -> bool {
        self.reduced.factor_with_shift(settings, Some(shift))
    }

    pub(crate) fn solve_interior_panel(&mut self, rhs: &[T], out: &mut [T], cols: usize) -> bool {
        self.reduced.solve_factor_panel(rhs, out, cols)
    }

    pub(crate) fn interior_residual(&self, out: &mut [T], rhs: &[T], point: &[T]) -> T {
        self.reduced.residual_full(out, rhs, point)
    }

    pub(crate) fn prepare_interior_rhs(&mut self, rhs: &[T], out: &mut [T]) {
        self.prepare_rhs(rhs);
        out.fill(T::zero());
        out[..self.n].copy_from_slice(&self.workx);
        out[self.n..self.n + self.retained_rhs.len()].copy_from_slice(&self.retained_rhs);
    }

    /// Save the sampled NT RHS workspace generated by the most recent
    /// `prepare_interior_rhs` call. Fused sampled recovery reads mat3c back;
    /// pair solves therefore keep one immutable snapshot per RHS column.
    pub(crate) fn copy_sampled_rhs_cache(&self, out: &mut Vec<T>) {
        out.clear();
        if !self.fused_sampled() {
            return;
        }
        for block in &self.blocks {
            if let Scaling::Psd(p) = &block.scaling {
                if p.sampled.is_some() {
                    out.extend_from_slice(p.mat3c.data());
                }
            }
        }
    }

    /// Restore a workspace snapshot captured by `copy_sampled_rhs_cache`.
    /// Return false on a structural mismatch so a stale cache can never be
    /// used as a successful recovery result.
    pub(crate) fn restore_sampled_rhs_cache(&mut self, saved: &[T]) -> bool {
        if !self.fused_sampled() {
            return saved.is_empty();
        }
        let expected: usize = self
            .blocks
            .iter()
            .filter_map(|block| match &block.scaling {
                Scaling::Psd(p) if p.sampled.is_some() => Some(p.mat3c.data().len()),
                _ => None,
            })
            .sum();
        if expected != saved.len() {
            return false;
        }
        let mut offset = 0;
        for block in &mut self.blocks {
            if let Scaling::Psd(p) = &mut block.scaling {
                if p.sampled.is_some() {
                    let values = p.mat3c.data_mut();
                    let len = values.len();
                    values.copy_from_slice(&saved[offset..offset + len]);
                    offset += len;
                }
            }
        }
        true
    }

    pub(crate) fn recover_interior_rhs(
        &mut self,
        out: &mut [T],
        rhs: &[T],
        interior: &[T],
    ) -> bool {
        out[..self.n].copy_from_slice(&interior[..self.n]);
        let nr = self.retained_rhs.len();
        self.retained_rhs
            .copy_from_slice(&interior[self.n..self.n + nr]);
        self.recover_rhs(out, rhs)
    }

    pub(crate) fn original_residual(&mut self, out: &mut [T], rhs: &[T], point: &[T]) -> T {
        self.residual(out, rhs, point, false)
    }

    pub(crate) fn restore_scaled_product(&mut self, point: &[T]) {
        self.workh_partial = false;
        apply_scaling_pool_with_world(
            self.mpi_world(),
            &self.pool,
            &self.scaling_lanes,
            self.scaling_tiles,
            self.scaling_workers,
            &mut self.blocks,
            &mut self.workh,
            &point[self.n..],
            false,
        );
    }

    /// After an aligned prepare, give every rank the scaled values of the
    /// linear-touched rows outside zero blocks (owners hold them).
    fn republish_linear_rows(&mut self, world: crate::mpi::World) {
        let rows = self.linear_scaled_rows.as_deref().unwrap_or(&[]);
        if rows.is_empty() {
            return;
        }
        let (_, spans) = rank_rows(&self.blocks, world);
        let mut ranges = Vec::with_capacity(spans.len());
        let mut k = 0;
        for &(start, len) in &spans {
            let first = k;
            while k < rows.len() && rows[k] >= start && rows[k] < start + len {
                k += 1;
            }
            ranges.push((first, k - first));
        }
        debug_assert_eq!(k, rows.len());
        let (k0, kl) = ranges[world.rank()];
        let local: Vec<T> = rows[k0..k0 + kl].iter().map(|&r| self.workz[r]).collect();
        let mut all = vec![T::zero(); rows.len()];
        world.gather_slice(crate::mpi::SITE_SCALING, &local, &ranges, &mut all);
        for (&r, v) in rows.iter().zip(all) {
            self.workz[r] = v;
        }
    }

    /// `H·z` of the last residual on every rank (gathered once if the
    /// aligned residual left it rank-partial).
    pub(crate) fn scaled_product(&mut self) -> &[T] {
        if std::mem::take(&mut self.workh_partial) {
            let world = self.mpi_world().unwrap();
            let ((y0, y1), ranges) = rank_rows(&self.blocks, world);
            let local = self.workh[y0..y1].to_vec();
            world.gather_slice(crate::mpi::SITE_SCALING, &local, &ranges, &mut self.workh);
        }
        &self.workh
    }

    pub(crate) fn owned_scaled_product(&self) -> Option<&[T]> {
        Some(&self.workh)
    }

    fn prepare_rhs(&mut self, rhs: &[T]) {
        let phase_timer = crate::receipt::start();
        let (bx, bz) = rhs.split_at(self.n);
        let fused = self.fused_sampled();
        let block_timer = crate::receipt::start();
        let aligned = if fused {
            None
        } else {
            self.align_sampled_parts()
        };
        if let Some(world) = aligned {
            // The sampled adjoint below reads only this rank's block rows.
            let ((y0, y1), _) = rank_rows(&self.blocks, world);
            let local = scale_owned(
                world,
                &self.pool,
                self.scaling_tiles,
                &mut self.blocks,
                &bz[y0..y1],
                ScalingAction::Apply(true),
            );
            self.workz.fill(T::zero());
            self.workz[y0..y1].copy_from_slice(&local);
            self.republish_linear_rows(world);
        } else {
            apply_block_pool_with_world(
                self.mpi_world(),
                &self.pool,
                &self.scaling_lanes,
                self.scaling_tiles,
                self.scaling_workers,
                &mut self.blocks,
                &mut self.workz,
                bz,
                if fused {
                    ScalingAction::Condense
                } else {
                    ScalingAction::Apply(true)
                },
            );
        }
        crate::receipt::finish("prepare_rhs.scaling", block_timer);
        self.workx.copy_from_slice(bx);
        if fused {
            let (operator, work) = self.sampled.as_mut().unwrap();
            work.linear_product_in_pool(
                operator,
                true,
                &mut self.workx,
                &self.workz,
                T::one(),
                T::one(),
                self.pool.as_ref(),
                "sampled.linear.prepare",
            );
            for block in &self.blocks {
                if let Scaling::Psd(p) = &block.scaling {
                    if let Some(s) = &p.sampled {
                        let start = s.operator.blocks()[s.block].column_start;
                        for (dst, &v) in self.workx[start..].iter_mut().zip(&s.adjoint) {
                            *dst += v;
                        }
                    }
                }
            }
        } else if let Some((operator, work)) = &mut self.sampled {
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
                self.mpi_world(),
                self.sparse_products.as_ref(),
                &self.A,
                self.a_panel.as_ref(),
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
        crate::receipt::finish("prepare_rhs", phase_timer);
    }

    fn recover_rhs(&mut self, out: &mut [T], rhs: &[T]) -> bool {
        let phase_timer = crate::receipt::start();
        let bz = &rhs[self.n..];
        let (x, z) = out.split_at_mut(self.n);
        let fused = self.fused_sampled();
        let aligned = !fused && self.align_sampled_parts().is_some();
        if fused {
            // `L x - bz` feeds only blocks recovered from `workz`; sampled PSD,
            // zero and retained rows never read it (and the fused residual
            // recomputes its own forward product), so skip it without them.
            if self.blocks.iter().any(|b| match &b.scaling {
                Scaling::Psd(p) => p.sampled.is_none(),
                Scaling::Orthant { .. } | Scaling::SocElim { .. } => true,
                _ => false,
            }) {
                self.recover_linear(x, bz);
            }
        } else if let Some((operator, work)) = &mut self.sampled {
            // Aligned: only this rank's rows are needed by its scaling.
            if aligned {
                operator.apply_owned_with_pool(
                    &mut self.workz,
                    x,
                    T::one(),
                    T::zero(),
                    work,
                    self.pool.as_ref(),
                );
            } else {
                operator.apply_with_pool(
                    &mut self.workz,
                    x,
                    T::one(),
                    T::zero(),
                    work,
                    self.pool.as_ref(),
                );
            }
        } else {
            Self::sparse_gemv(
                self.mpi_world(),
                self.sparse_products.as_ref(),
                &self.A,
                self.a_panel.as_ref(),
                false,
                &mut self.workz,
                x,
                T::one(),
                T::zero(),
            );
        }
        if !fused {
            for (v, &b) in self.workz.iter_mut().zip(bz) {
                *v -= b;
            }
        }
        let block_timer = crate::receipt::start();
        apply_block_pool_with_world(
            self.mpi_world(),
            &self.pool,
            &self.scaling_lanes,
            self.scaling_tiles,
            self.scaling_workers,
            &mut self.blocks,
            z,
            &self.workz,
            if fused {
                ScalingAction::Recover(x)
            } else {
                ScalingAction::Apply(true)
            },
        );
        for (&v, &row) in self.retained_rhs.iter().zip(&self.retained_rows) {
            z[row] = v;
        }
        crate::receipt::finish("recover_rhs.scaling", block_timer);
        let finite = out.is_finite();
        crate::receipt::finish("recover_rhs", phase_timer);
        finite
    }

    fn solve_raw(&mut self, out: &mut [T], rhs: &[T], settings: &CoreSettings<T>) -> bool {
        self.solve_reduced(out, rhs, settings)
            && local_success(self.local_only, self.recover_rhs(out, rhs))
    }

    /// The reduced half of `solve_raw`: condense `rhs` and solve for
    /// `out[..n]` and the retained rows; `recover_rhs` finishes `out`.
    fn solve_reduced(&mut self, out: &mut [T], rhs: &[T], settings: &CoreSettings<T>) -> bool {
        self.prepare_rhs(rhs);
        self.reduced.setrhs(&self.workx, &self.retained_rhs);
        let reduced_ok = self.reduced.solve(
            Some(&mut out[..self.n]),
            Some(&mut self.retained_rhs),
            settings,
        );
        local_success(self.local_only, reduced_ok)
    }

    fn residual(&mut self, out: &mut [T], rhs: &[T], solution: &[T], reuse_forward: bool) -> T {
        let timer = crate::receipt::start();
        let r = self.residual_inner(out, rhs, solution, reuse_forward);
        crate::receipt::finish("residual", timer);
        r
    }

    fn residual_inner(
        &mut self,
        out: &mut [T],
        rhs: &[T],
        solution: &[T],
        reuse_forward: bool,
    ) -> T {
        let reuse_forward = reuse_forward && !self.fused_sampled();
        // Aligned partitions: forward and scaling fill only this rank's rows
        // of ez, exchanged once at the end instead of twice.
        let aligned = if self.fused_sampled() {
            None
        } else {
            self.align_sampled_parts()
        };
        let rows = aligned.map(|w| rank_rows(&self.blocks, w));
        let (x, z) = solution.split_at(self.n);
        let (ex, ez) = out.split_at_mut(self.n);
        ex.copy_from_slice(&rhs[..self.n]);
        ez.copy_from_slice(&rhs[self.n..]);
        // These stages own disjoint output/scratch buffers. Sharing the existing
        // pool can fill block-tail idle time without changing either arithmetic
        // order or allocating another set of workers.
        let world = self.mpi_world();
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
            scaling_workers,
            sparse_products,
            a_panel,
            ..
        } = self;
        // Single process: the linear halves of both sampled products run
        // first, on the whole pool. Inside the join below the scaling lanes
        // hold every other worker, so the linear lanes ran on the calling
        // worker alone (Λ27: the 272k-term adjoint took 11.7 ms median).
        // Each output keeps its operation order (P, linear, then blocks).
        let split = !reuse_forward
            && world.is_none()
            && pool.as_ref().is_some_and(|p| p.current_num_threads() > 1)
            && sampled
                .as_ref()
                .is_some_and(|(operator, work)| operator.pooled_halves_apply(work, pool.as_ref()));
        if split {
            let (operator, work) = sampled.as_mut().unwrap();
            P.sym_up().symv(ex, x, -T::one(), T::one());
            work.linear_product_in_pool(
                operator,
                true,
                ex,
                z,
                -T::one(),
                T::one(),
                pool.as_ref(),
                "sampled.linear.adj",
            );
            work.linear_product_in_pool(
                operator,
                false,
                ez,
                x,
                -T::one(),
                T::one(),
                pool.as_ref(),
                "sampled.linear.fwd",
            );
        }
        let mut products = || {
            if split {
                let (operator, work) = sampled.as_mut().unwrap();
                let __t = std::time::Instant::now();
                operator.adjoint_blocks_with_pool(ex, z, -T::one(), work, pool.as_ref());
                crate::receipt::phase("residual.adj", __t.elapsed());
                let __t = std::time::Instant::now();
                operator.forward_blocks_with_pool(ez, x, -T::one(), work, pool.as_ref());
                crate::receipt::phase("residual.fwd", __t.elapsed());
                return;
            }
            P.sym_up().symv(ex, x, -T::one(), T::one());
            if let Some((operator, work)) = sampled {
                let __t = std::time::Instant::now();
                operator.apply_transpose_with_pool(ex, z, -T::one(), T::one(), work, pool.as_ref());
                crate::receipt::phase("residual.adj", __t.elapsed());
                if !reuse_forward {
                    let __t = std::time::Instant::now();
                    if aligned.is_some() {
                        operator.apply_owned_with_pool(
                            ez,
                            x,
                            -T::one(),
                            T::one(),
                            work,
                            pool.as_ref(),
                        );
                    } else {
                        operator.apply_with_pool(ez, x, -T::one(), T::one(), work, pool.as_ref());
                    }
                    crate::receipt::phase("residual.fwd", __t.elapsed());
                }
            } else if let (Some(world), Some(plan)) = (world, sparse_products.as_ref()) {
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
                match a_panel {
                    Some(panel) => panel.gemv(true, ex, z, -T::one(), T::one()),
                    None => A.t().gemv(ex, z, -T::one(), T::one()),
                }
                if !reuse_forward {
                    match a_panel {
                        Some(panel) => panel.gemv(false, ez, x, -T::one(), T::one()),
                        None => A.gemv(ez, x, -T::one(), T::one()),
                    }
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
            if let (Some(w), Some(((y0, y1), _))) = (aligned, &rows) {
                let local = scale_owned(
                    w,
                    pool,
                    *scaling_tiles,
                    blocks,
                    &z[*y0..*y1],
                    ScalingAction::Apply(false),
                );
                workh.fill(T::zero());
                workh[*y0..*y1].copy_from_slice(&local);
            } else {
                apply_scaling_pool_with_world(
                    world,
                    pool,
                    scaling_lanes,
                    *scaling_tiles,
                    *scaling_workers,
                    blocks,
                    workh,
                    z,
                    false,
                );
            }
            crate::receipt::phase("residual.scale", __t.elapsed());
        };
        if let Some(pool) = pool.as_ref().filter(|_| scaling_lanes.len() > 1) {
            pool.install(|| rayon::join(products, scaling));
        } else {
            products();
            scaling();
        }
        crate::algebra::add_assign(ez, &self.workh);
        if let (Some(w), Some(((y0, y1), ranges))) = (aligned, rows) {
            let local = ez[y0..y1].to_vec();
            w.gather_slice(crate::mpi::SITE_SCALING, &local, &ranges, ez);
        }
        self.workh_partial = aligned.is_some();
        if out.is_finite() {
            out.norm_inf()
        } else {
            T::infinity()
        }
    }
}

impl<T: FloatT> CondensedKKTSolver<T> {
    /// `workz <- L*x - bz` for the unsampled part: the first half of the
    /// fused sampled `recover_rhs`.
    fn recover_linear(&mut self, x: &[T], bz: &[T]) {
        let (operator, work) = self.sampled.as_mut().unwrap();
        work.linear_product_in_pool(
            operator,
            false,
            &mut self.workz,
            x,
            T::one(),
            T::zero(),
            self.pool.as_ref(),
            "sampled.linear.recover",
        );
        let subtract = |(v, &b): (&mut T, &T)| *v -= b;
        match &self.pool {
            Some(pool) if pool.current_num_threads() > 1 => pool.install(|| {
                self.workz
                    .par_iter_mut()
                    .zip(bz)
                    .with_min_len(4096)
                    .for_each(subtract)
            }),
            _ => self.workz.iter_mut().zip(bz).for_each(subtract),
        }
    }
}

#[path = "scaling.rs"]
mod scaling;
use scaling::*;

#[path = "psd.rs"]
mod psd_impl;
#[cfg(all(test, target_arch = "x86_64"))]
use psd_impl::SchurSink;
use psd_impl::TransformPanels;
#[cfg(test)]
pub(crate) use psd_impl::{PARALLEL_DOT_LANES, PARALLEL_TRANSFORM_LANES, POOLED_CONGRUENCE_TILES};

#[path = "kkt.rs"]
mod kkt_impl;

#[cfg(test)]
#[path = "tests/basic.rs"]
mod tests;

#[cfg(test)]
#[path = "tests/parallel.rs"]
mod parallel_tests;

#[cfg(test)]
#[path = "tests/graded.rs"]
mod graded_action_tests;

#[cfg(test)]
#[path = "tests/local_policy.rs"]
mod local_policy_tests;
