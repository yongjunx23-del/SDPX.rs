//! Factor-authoritative sampled PSD coefficients. Materialization is a rounded
//! view of these factors, not the definition of their operator.
use crate::algebra::sparse_parallel::SparseParallel;
use crate::algebra::*;
use rayon::prelude::*;
use sdpx_arithmetic::{EncodeSide, Residues, RnsPlan};
use std::sync::Arc;

/// A sampled PSD row block with canonical columns `(s, r, k)`, r <= s.
/// Each column is weight * svec(sym(e_r e_s') ⊗ q_k q_k').
/// `sym` averages its two arguments, so off-diagonal blocks carry one half.
#[derive(Clone, Debug)]
pub struct SampledBlock<T> {
    /// Zero-based start of this block in the conic row vector.
    pub row_start: usize,
    /// Zero-based start of canonical primitive columns in the variable vector.
    pub column_start: usize,
    /// Number of polynomial matrix block rows and columns.
    pub dim: usize,
    /// Number of rows in each basis vector.
    pub basis_rows: usize,
    /// Number of sample basis vectors; zero is permitted.
    pub basis_cols: usize,
    /// Column-major basis matrix Q.
    pub basis: Vec<T>,
    /// Weights in nested `(s, r, k)` order, with `r <= s` and `k` varying fastest.
    pub weights: Vec<T>,
}

fn checked_triangle(n: usize) -> Option<usize> {
    let next = n.checked_add(1)?;
    if n % 2 == 0 {
        (n / 2).checked_mul(next)
    } else {
        n.checked_mul(next / 2)
    }
}
fn tri(n: usize) -> usize {
    n * (n + 1) / 2
}
fn tri_work(n: usize) -> u128 {
    (n as u128) * (n as u128 + 1) / 2
}

/// `SDPX_RNS_OPS` selects exact residue-domain dots for sampled products.
/// Reconstruction rounds each output once, so results may differ from the
/// `dot_fma` chain in the last ulp; every dot remains the correctly rounded
/// exact sum.
fn rns_ops_enabled() -> bool {
    std::env::var_os("SDPX_RNS_OPS").is_some()
}

/// Cached constant-side encoding: residues of an operand that never changes
/// across calls. Encoded once at the full prime-table width so it stays
/// valid while the per-call plan's prime count moves with the varying
/// side's exponent window (see [`RnsPlan::encode_wide`]).
struct RnsConst {
    shift: i64,
    primes: usize,
    res: Residues,
}

/// Shared context passed to split leaves: the constant and per-call variable
/// residues plus the plan they were encoded against. Immutable and `Sync`, so
/// parallel subtasks evaluate disjoint outputs against the same operands.
#[derive(Clone, Copy)]
struct RnsSide<'a> {
    plan: &'a RnsPlan,
    cns: &'a Residues,
    var: &'a Residues,
}

/// Refresh `cache` to the plan's a-side encoding of `const_vals`.
fn rns_const<T: FloatT>(
    cache: &mut Option<RnsConst>,
    plan: &RnsPlan,
    const_vals: &[T],
) -> Option<()> {
    let hit = matches!(cache, Some(c) if c.shift == plan.shift_a() && c.primes >= plan.primes());
    if !hit {
        let res = T::rns_encode_wide(plan, EncodeSide::A, const_vals)?;
        *cache = Some(RnsConst {
            shift: plan.shift_a(),
            primes: res.k,
            res,
        });
    }
    Some(())
}

/// Build this call's plan and encode the varying side; the constant side is
/// scanned once (`range`) and encoded once per `shift_a` (`cache`).
/// Returns `None` — keeping the MPFR dot path — when the gate is off, the
/// backend has no residue support, an operand is non-finite, the exponent
/// window overflows, or the cost model says the swap is not profitable.
fn rns_call<T: FloatT>(
    cache: &mut Option<RnsConst>,
    range: &mut Option<Option<(i64, i64)>>,
    const_vals: &[T],
    var: &[T],
    terms: usize,
    outputs: usize,
) -> Option<(RnsPlan, Residues)> {
    if !rns_ops_enabled() {
        return None;
    }
    // Measured on Λ=15 (jobs 214136/214153): reconstruction is per-output,
    // so products with more outputs than terms pay more Garner+MPFR work
    // than the residue dots save — the forward product (76 terms, 741
    // outputs) regressed ~40% while the adjoint (741 terms, 76 outputs)
    // improved ~20%. Restrict RNS to term-dominated products.
    if outputs > terms {
        return None;
    }
    let a_range = match range {
        Some(r) => *r,
        slot @ None => {
            let r = T::rns_exponent_range(const_vals);
            *slot = Some(r);
            r
        }
    }?;
    let b_range = T::rns_exponent_range(var)?;
    let plan = T::rns_plan_ranges(a_range, b_range, terms)?;
    // The constant side is encoded once; the recurring cost is the varying
    // side plus a small amortized share of the constant encode.
    let encode_elems = var.len() + const_vals.len() / 64;
    if !plan.profitable(terms, outputs, encode_elems, T::precision_bits() / 64) {
        return None;
    }
    rns_const(cache, &plan, const_vals)?;
    let var = T::rns_encode(&plan, EncodeSide::B, var)?;
    Some((plan, var))
}

impl<T: FloatT> SampledBlock<T> {
    /// Structural scalar work in the limb-squared units already used by the
    /// cone and sampled schedulers. It depends only on dimensions and on the
    /// arithmetic precision, never on problem identity or runtime timing.
    pub(crate) fn scheduled_work(&self) -> u128 {
        if self.dim == 0 || self.basis_cols == 0 {
            return 0;
        }
        let words = T::precision_bits().div_ceil(64) as u128;
        tri_work(self.dim)
            .saturating_mul((self.basis_rows as u128).pow(2))
            .saturating_mul(self.basis_cols as u128)
            .saturating_mul(words * words)
    }
}

impl<T> SampledBlock<T> {
    /// PSD matrix side length: `dim * basis_rows`.
    pub fn side(&self) -> usize {
        self.dim * self.basis_rows
    }
    /// Number of packed upper-triangle conic rows.
    pub fn row_count(&self) -> usize {
        tri(self.side())
    }
    /// Number of weighted canonical primitive columns.
    pub fn column_count(&self) -> usize {
        self.weights.len()
    }
}

/// Block-concatenated term offsets, len `blocks.len() + 1` with a trailing
/// total. Deterministic on every rank since it depends only on structure.
pub(crate) fn term_offsets<T>(
    blocks: &[SampledBlock<T>],
    count: impl Fn(&SampledBlock<T>) -> usize,
) -> Vec<usize> {
    let mut offsets = Vec::with_capacity(blocks.len() + 1);
    offsets.push(0);
    for b in blocks {
        offsets.push(offsets.last().unwrap() + count(b));
    }
    offsets
}

/// `(min row_start, max row_end)` over a block slice; `(0, 0)` when empty.
/// PSD row ranges are disjoint by operator construction.
pub(crate) fn block_row_span<T>(blocks: &[SampledBlock<T>]) -> (usize, usize) {
    let mut span = (usize::MAX, 0usize);
    for b in blocks {
        span.0 = span.0.min(b.row_start);
        span.1 = span.1.max(b.row_start + b.row_count());
    }
    if span.0 > span.1 {
        (0, 0)
    } else {
        span
    }
}

/// Conic coefficient operator combining ordinary CSC entries and sampled factors.
#[derive(Clone, Debug)]
pub struct SampledOperator<T> {
    linear: CscMatrix<T>,
    blocks: Vec<SampledBlock<T>>,
    ordered_rows: bool,
}

impl<T: FloatT> SampledOperator<T> {
    /// PSD row ranges must be disjoint and contain no nonzero linear entries.
    /// Column ranges may overlap, including overlaps between different blocks.
    pub fn new(mut linear: CscMatrix<T>, blocks: Vec<SampledBlock<T>>) -> Result<Self, String> {
        linear
            .check_format()
            .map_err(|e| format!("sampled linear CSC: {e:?}"))?;
        if linear.colptr.first() != Some(&0) {
            return Err("sampled linear CSC must start at offset zero".into());
        }
        if !linear.nzval.iter().all(|v| v.is_finite()) {
            return Err("sampled linear CSC contains nonfinite values".into());
        }
        let mut ranges = Vec::with_capacity(blocks.len());
        for block in &blocks {
            if !block
                .basis
                .iter()
                .chain(&block.weights)
                .all(|v| v.is_finite())
            {
                return Err("sampled basis or weights contain nonfinite values".into());
            }
            let bad = || "invalid sampled block dimensions or range".to_string();
            if block.dim == 0 || block.basis_rows == 0 {
                return Err(bad());
            }
            let side = block.dim.checked_mul(block.basis_rows).ok_or_else(bad)?;
            let rows = checked_triangle(side).ok_or_else(bad)?;
            let cols = checked_triangle(block.dim)
                .and_then(|v| v.checked_mul(block.basis_cols))
                .ok_or_else(bad)?;
            let basis_len = block
                .basis_rows
                .checked_mul(block.basis_cols)
                .ok_or_else(bad)?;
            let end = block.row_start.checked_add(rows).ok_or_else(bad)?;
            let col_end = block.column_start.checked_add(cols).ok_or_else(bad)?;
            if end > linear.m
                || col_end > linear.n
                || block.basis.len() != basis_len
                || block.weights.len() != cols
            {
                return Err(bad());
            }
            // Dense provider dimensions use BLAS integers; also validate the
            // sizes of persistent NT and pairing arrays before allocation.
            let rank = block.dim.checked_mul(block.basis_cols).ok_or_else(bad)?;
            for size in [side, rank, block.basis_rows, block.basis_cols] {
                if size > i32::MAX as usize {
                    return Err(bad());
                }
            }
            side.checked_mul(rank).ok_or_else(bad)?;
            rank.checked_mul(rank).ok_or_else(bad)?;
            block
                .basis_rows
                .checked_mul(block.basis_rows)
                .ok_or_else(bad)?;
            ranges.push(block.row_start..end);
        }
        ranges.sort_by_key(|r| r.start);
        if ranges.windows(2).any(|w| w[0].end > w[1].start) {
            return Err("overlapping sampled PSD rows".into());
        }
        // Publish sampled-row membership once. Probing the block ranges per
        // stored entry instead is O(nnz * blocks), which is quadratic on the
        // many-block factorizations this route exists for.
        let mut sampled_row = vec![false; linear.m];
        for range in &ranges {
            sampled_row[range.clone()].fill(true);
        }
        if linear
            .rowval
            .iter()
            .zip(&linear.nzval)
            .any(|(&r, &v)| v != T::zero() && sampled_row[r])
        {
            return Err("linear CSC must be zero on sampled PSD rows".into());
        }
        // The sampled factors exclusively define these rows. Strip even
        // explicit zeros, which otherwise invent active columns outside a
        // block's canonical primitive range during condensed setup. Ordinary
        // row storage, including structural zeros, remains unchanged.
        let mut write = 0;
        let mut start = 0;
        for col in 0..linear.n {
            let end = linear.colptr[col + 1];
            linear.colptr[col] = write;
            for idx in start..end {
                let row = linear.rowval[idx];
                if !sampled_row[row] {
                    linear.rowval[write] = row;
                    linear.nzval[write] = linear.nzval[idx];
                    write += 1;
                }
            }
            start = end;
        }
        linear.colptr[linear.n] = write;
        linear.rowval.truncate(write);
        linear.nzval.truncate(write);
        let ordered_rows = blocks.windows(2).all(|w| w[0].row_start < w[1].row_start);
        Ok(Self {
            linear,
            blocks,
            ordered_rows,
        })
    }
    /// Ordinary CSC coefficients outside sampled PSD row blocks.
    pub fn linear(&self) -> &CscMatrix<T> {
        &self.linear
    }
    /// Sampled block descriptors in construction order.
    pub fn blocks(&self) -> &[SampledBlock<T>] {
        &self.blocks
    }
    /// Operator dimensions `(conic rows, variables)`.
    pub fn dims(&self) -> (usize, usize) {
        (self.linear.m, self.linear.n)
    }

    /// Apply E*A*D. The caller must provide uniform E on each sampled PSD block.
    pub fn scale(&mut self, d: &[T], e: &[T]) {
        assert_eq!(d.len(), self.linear.n);
        assert_eq!(e.len(), self.linear.m);
        self.linear.lrscale(e, d);
        for block in &mut self.blocks {
            let row_scale = e[block.row_start];
            for (p, w) in block.weights.iter_mut().enumerate() {
                *w = row_scale * d[block.column_start + p] * *w;
            }
        }
    }

    /// Materialize at T's working precision without changing factor semantics.
    pub fn materialize(&self) -> CscMatrix<T> {
        let mut rows = self.linear.rowval.clone();
        let mut vals = self.linear.nzval.clone();
        let mut cols = Vec::with_capacity(vals.len());
        for j in 0..self.linear.n {
            cols.extend(std::iter::repeat_n(
                j,
                self.linear.colptr[j + 1] - self.linear.colptr[j],
            ));
        }
        for b in &self.blocks {
            let h = b.basis_rows;
            let mut p = 0;
            for s in 0..b.dim {
                for r in 0..=s {
                    for k in 0..b.basis_cols {
                        for j in 0..h {
                            for i in 0..if r == s { j + 1 } else { h } {
                                let row = b.row_start + tri(s * h + j) + r * h + i;
                                let scale = if r != s {
                                    T::FRAC_1_SQRT_2()
                                } else if i != j {
                                    T::SQRT_2()
                                } else {
                                    T::one()
                                };
                                let value =
                                    b.weights[p] * b.basis[i + k * h] * b.basis[j + k * h] * scale;
                                if value != T::zero() {
                                    rows.push(row);
                                    cols.push(b.column_start + p);
                                    vals.push(value);
                                }
                            }
                        }
                        p += 1;
                    }
                }
            }
        }
        CscMatrix::new_from_triplets(self.linear.m, self.linear.n, rows, cols, vals)
    }

    /// y <- alpha*A*x + beta*y, using persistent per-block scratch.
    pub fn apply(&self, y: &mut [T], x: &[T], alpha: T, beta: T, work: &mut SampledWorkspace<T>) {
        assert_eq!(y.len(), self.linear.m);
        assert_eq!(x.len(), self.linear.n);
        assert_eq!(work.blocks.len(), self.blocks.len());
        self.linear.gemv(y, x, alpha, beta);
        if alpha == T::zero() {
            return;
        }
        for (b, w) in self.blocks.iter().zip(&mut work.blocks) {
            w.forward_terms(b, x, alpha, |i, term| y[b.row_start + i] += term);
        }
    }

    /// y <- alpha*A'*x + beta*y. Shared columns accumulate in block order.
    pub fn apply_transpose(
        &self,
        y: &mut [T],
        x: &[T],
        alpha: T,
        beta: T,
        work: &mut SampledWorkspace<T>,
    ) {
        assert_eq!(y.len(), self.linear.n);
        assert_eq!(x.len(), self.linear.m);
        assert_eq!(work.blocks.len(), self.blocks.len());
        self.linear.t().gemv(y, x, alpha, beta);
        if alpha == T::zero() {
            return;
        }
        for (b, w) in self.blocks.iter().zip(&mut work.blocks) {
            w.adjoint_terms(b, x, alpha, |i, term| y[b.column_start + i] += term);
        }
    }
    pub(crate) fn apply_with_pool(
        &self,
        y: &mut [T],
        x: &[T],
        alpha: T,
        beta: T,
        work: &mut SampledWorkspace<T>,
        pool: Option<&Arc<rayon::ThreadPool>>,
    ) {
        if !work.parallel_eligible(self, pool) || alpha == T::zero() {
            return self.apply(y, x, alpha, beta, work);
        }
        assert_eq!(y.len(), self.linear.m);
        assert_eq!(x.len(), self.linear.n);
        assert_eq!(work.blocks.len(), self.blocks.len());
        work.linear_product(self, false, y, x, alpha, beta, pool);
        if self.ordered_rows {
            let chunks = block_chunks(&self.blocks, pool);
            if let Some(world) = crate::mpi::World::get() {
                // Rank-sharded forward pass. Each rank evaluates a contiguous
                // block range on its own row segment; gathered segments
                // reproduce the serial result bitwise.
                let block_range = world.range(self.blocks.len());
                let (row_begin, row_end) = block_row_span(&self.blocks[block_range.clone()]);
                let gather_ranges: Vec<(usize, usize)> =
                    crate::mpi::ranges(self.blocks.len(), world.size())
                        .iter()
                        .map(|&(b0, len)| {
                            if len == 0 {
                                (0, 0)
                            } else {
                                let (begin, end) =
                                    block_row_span(&self.blocks[b0..b0 + len]);
                                (begin, end - begin)
                            }
                        })
                        .collect();
                let mut local = y[row_begin..row_end].to_vec();
                pool.unwrap().install(|| {
                    forward_disjoint(
                        &self.blocks[block_range.clone()],
                        &mut work.blocks[block_range],
                        &mut local,
                        row_begin,
                        x,
                        alpha,
                        chunks,
                    );
                });
                world.gather_slice(crate::mpi::SITE_FORWARD, &local, &gather_ranges, y);
                return;
            }
            pool.unwrap().install(|| {
                forward_disjoint(&self.blocks, &mut work.blocks, y, 0, x, alpha, chunks);
            });
            return;
        }
        let chunks = block_chunks(&self.blocks, pool);
        let world = crate::mpi::World::get();
        let active = world
            .map(|w| w.range(self.blocks.len()))
            .unwrap_or(0..self.blocks.len());
        pool.unwrap().install(|| {
            self.blocks[active.clone()]
                .par_iter()
                .zip(work.blocks[active.clone()].par_iter_mut())
                .for_each(|(b, w)| {
                    if b.basis_cols == 0 {
                        return;
                    }
                    let mut terms = std::mem::take(&mut w.forward);
                    terms.resize(b.row_count(), T::zero());
                    if chunks > 1 {
                        // The split writer accumulates, so the reused workspace
                        // buffer must start from zero on every call.
                        terms.fill(T::zero());
                        let h = b.basis_rows;
                        let kmax = b.basis_cols;
                        let sqrt2 = if h > 1 { T::SQRT_2() } else { T::zero() };
                        let inv_sqrt2 = if h > 0 && b.dim > 1 {
                            T::FRAC_1_SQRT_2()
                        } else {
                            T::zero()
                        };
                        let q = BorrowedMatrix {
                            size: (h, kmax),
                            data: b.basis.as_slice(),
                            phantom: std::marker::PhantomData,
                        };
                        let call = w.prepare_fwd_rns(b, x);
                        let rns = call.as_ref().map(|(p, v)| RnsSide {
                            plan: p,
                            cns: &w.rns_fwd.as_ref().unwrap().res,
                            var: v,
                        });
                        forward_split_chunks(
                            b,
                            &q,
                            x,
                            alpha,
                            0..b.dim,
                            &mut terms[..],
                            sqrt2,
                            inv_sqrt2,
                            chunks,
                            &mut w.square,
                            &mut w.panel,
                            rns,
                        );
                    } else {
                        w.forward_terms(b, x, alpha, |i, term| terms[i] = term);
                    }
                    w.forward = terms;
                })
        });
        if let Some(world) = world {
            // Gather per-block terms in block order; column ranges may
            // overlap, so the exchange layout is a per-block concatenation.
            let offsets = term_offsets(&self.blocks, SampledBlock::row_count);
            let gather_ranges: Vec<(usize, usize)> =
                crate::mpi::ranges(self.blocks.len(), world.size())
                    .iter()
                    .map(|&(b0, len)| (offsets[b0], offsets[b0 + len] - offsets[b0]))
                    .collect();
            let (t0, t1) = (offsets[active.start], offsets[active.end]);
            let mut local = Vec::with_capacity(t1 - t0);
            for (b, w) in self.blocks[active.clone()].iter().zip(&work.blocks[active]) {
                let n = b.row_count();
                if b.basis_cols == 0 {
                    local.resize(local.len() + n, T::zero());
                } else {
                    local.extend_from_slice(&w.forward[..n]);
                }
            }
            debug_assert_eq!(local.len(), t1 - t0);
            let mut all = vec![T::zero(); *offsets.last().unwrap()];
            world.gather_slice(crate::mpi::SITE_FORWARD, &local, &gather_ranges, &mut all);
            for (b, off) in self.blocks.iter().zip(&offsets) {
                if b.basis_cols == 0 {
                    continue;
                }
                for (i, &term) in all[*off..*off + b.row_count()].iter().enumerate() {
                    y[b.row_start + i] += term;
                }
            }
            return;
        }
        for (b, w) in self.blocks.iter().zip(&work.blocks) {
            if b.basis_cols == 0 {
                continue;
            }
            for (i, term) in w.forward.iter().enumerate() {
                y[b.row_start + i] += *term;
            }
        }
    }

    pub(crate) fn apply_transpose_with_pool(
        &self,
        y: &mut [T],
        x: &[T],
        alpha: T,
        beta: T,
        work: &mut SampledWorkspace<T>,
        pool: Option<&Arc<rayon::ThreadPool>>,
    ) {
        if !work.parallel_eligible(self, pool) || alpha == T::zero() {
            return self.apply_transpose(y, x, alpha, beta, work);
        }
        assert_eq!(y.len(), self.linear.n);
        assert_eq!(x.len(), self.linear.m);
        assert_eq!(work.blocks.len(), self.blocks.len());
        work.linear_product(self, true, y, x, alpha, beta, pool);
        let chunks = block_chunks(&self.blocks, pool);
        let world = crate::mpi::World::get();
        let active = world
            .map(|w| w.range(self.blocks.len()))
            .unwrap_or(0..self.blocks.len());
        pool.unwrap().install(|| {
            self.blocks[active.clone()]
                .par_iter()
                .zip(work.blocks[active.clone()].par_iter_mut())
                .for_each(|(b, w)| {
                    if b.basis_cols == 0 {
                        return;
                    }
                    let mut terms = std::mem::take(&mut w.adjoint);
                    terms.resize(b.column_count(), T::zero());
                    if chunks > 1 {
                        // The split writer assigns into a reused workspace
                        // buffer, so clear the previous call's contents first.
                        terms.fill(T::zero());
                        let h = b.basis_rows;
                        let kmax = b.basis_cols;
                        let inv_sqrt2 = if h > 0 && (h > 1 || b.dim > 1) {
                            T::FRAC_1_SQRT_2()
                        } else {
                            T::zero()
                        };
                        let q = BorrowedMatrix {
                            size: (h, kmax),
                            data: b.basis.as_slice(),
                            phantom: std::marker::PhantomData,
                        };
                        let call = w.prepare_adj_rns(b, x);
                        let rns = call.as_ref().map(|(p, v)| RnsSide {
                            plan: p,
                            cns: &w.rns_adj.as_ref().unwrap().res,
                            var: v,
                        });
                        adjoint_split_chunks(
                            b,
                            &q,
                            &w.wdiag,
                            x,
                            alpha,
                            0..b.dim,
                            &mut terms[..],
                            inv_sqrt2,
                            chunks,
                            &mut w.square,
                            &mut w.panel,
                            rns,
                        );
                    } else {
                        w.adjoint_terms(b, x, alpha, |i, term| terms[i] = term);
                    }
                    w.adjoint = terms;
                })
        });
        if let Some(world) = world {
            // Gather per-block adjoint terms in block order; column ranges
            // may overlap, so the exchange layout is a concatenation.
            let offsets = term_offsets(&self.blocks, SampledBlock::column_count);
            let gather_ranges: Vec<(usize, usize)> =
                crate::mpi::ranges(self.blocks.len(), world.size())
                    .iter()
                    .map(|&(b0, len)| (offsets[b0], offsets[b0 + len] - offsets[b0]))
                    .collect();
            let (t0, t1) = (offsets[active.start], offsets[active.end]);
            let mut local = Vec::with_capacity(t1 - t0);
            for (b, w) in self.blocks[active.clone()].iter().zip(&work.blocks[active]) {
                let n = b.column_count();
                if b.basis_cols == 0 {
                    local.resize(local.len() + n, T::zero());
                } else {
                    local.extend_from_slice(&w.adjoint[..n]);
                }
            }
            debug_assert_eq!(local.len(), t1 - t0);
            let mut all = vec![T::zero(); *offsets.last().unwrap()];
            world.gather_slice(crate::mpi::SITE_ADJOINT, &local, &gather_ranges, &mut all);
            for (b, off) in self.blocks.iter().zip(&offsets) {
                if b.basis_cols == 0 {
                    continue;
                }
                for (i, &term) in all[*off..*off + b.column_count()].iter().enumerate() {
                    y[b.column_start + i] += term;
                }
            }
            return;
        }
        // Descriptor order is authoritative even for unsorted rows and shared
        // variable ranges. Assigning terms avoids an extra rounded addition.
        for (b, w) in self.blocks.iter().zip(&work.blocks) {
            if b.basis_cols == 0 {
                continue;
            }
            for (i, term) in w.adjoint.iter().enumerate() {
                y[b.column_start + i] += *term;
            }
        }
    }
}

#[path = "sampled_split.rs"]
mod split;
use split::*;
struct SampledBlockWorkspace<T> {
    // Lazy contribution storage: validated disjoint row ranges bound all
    // forward slices by m; adjoint slices total the primitive column counts.
    forward: Vec<T>,
    adjoint: Vec<T>,
    // Per-block constant `W[p, k] = c_p·q[i,k]·q[j,k]` over the packed
    // `p = tri(j)+i` diagonal triangle, with `c = 1` on `i == j` and `√2`
    // off-diagonal. It turns the `q'·S·q` quadratic forms of every `r == s`
    // adjoint pair into flat dot products over `x`'s svec entries — half the
    // `panel.mul` work for the common single-row blocks.
    wdiag: Vec<T>,
    panel: Matrix<T>,
    square: Matrix<T>,
    /// Transposed `wdiag` (`qq[tri(j)+i][k] = wdiag[k][tri(j)+i]`): turns the
    /// forward `q'·D·q` triangle into flat `qq·d` dots on the residue path.
    /// Lazily built for `dim == 1` blocks only.
    qq: Vec<T>,
    /// `d[k] = weights[k]·x[column_start+k]` scratch for the residue forward.
    dvec: Vec<T>,
    /// Constant-side residue caches: `adj` encodes `wdiag`, `fwd` encodes `qq`.
    rns_adj: Option<RnsConst>,
    rns_fwd: Option<RnsConst>,
    /// Shared `wdiag`/`qq` exponent range (same values, permuted layout);
    /// `Some(None)` records a permanent non-finite scan failure.
    rns_range: Option<Option<(i64, i64)>>,
}
impl<T: FloatT> SampledBlockWorkspace<T> {
    /// `dim == 1` adjoint residue context: `wdiagᵀ·x_sub` over the block's
    /// contiguous svec row range. Constant residues land in `self.rns_adj`.
    fn prepare_adj_rns(
        &mut self,
        b: &SampledBlock<T>,
        x: &[T],
    ) -> Option<(RnsPlan, Residues)> {
        if b.dim != 1 || b.basis_cols == 0 {
            return None;
        }
        let terms = tri(b.basis_rows);
        let x_sub = &x[b.row_start..b.row_start + terms];
        rns_call(
            &mut self.rns_adj,
            &mut self.rns_range,
            &self.wdiag,
            x_sub,
            terms,
            b.basis_cols,
        )
    }

    /// `dim == 1` forward residue context: `qq·d` where `d_k` is the
    /// weight-scaled primitive slice and `qq` is `wdiag` transposed so the
    /// svec scale factors ride inside the residues.
    fn prepare_fwd_rns(
        &mut self,
        b: &SampledBlock<T>,
        x: &[T],
    ) -> Option<(RnsPlan, Residues)> {
        if b.dim != 1 || b.basis_cols == 0 {
            return None;
        }
        let h = b.basis_rows;
        let kmax = b.basis_cols;
        let trih = tri(h);
        // Skip building `qq` entirely for output-dominated shapes that
        // `rns_call` rejects anyway.
        if trih > kmax {
            return None;
        }
        if self.qq.is_empty() {
            self.qq = vec![T::zero(); trih * kmax];
            for k in 0..kmax {
                for t in 0..trih {
                    self.qq[t * kmax + k] = self.wdiag[k * trih + t];
                }
            }
        }
        self.dvec.clear();
        self.dvec.resize(kmax, T::zero());
        for k in 0..kmax {
            self.dvec[k] = b.weights[k] * x[b.column_start + k];
        }
        rns_call(
            &mut self.rns_fwd,
            &mut self.rns_range,
            &self.qq,
            &self.dvec,
            kmax,
            trih,
        )
    }

    fn forward_terms(
        &mut self,
        b: &SampledBlock<T>,
        x: &[T],
        alpha: T,
        mut store: impl FnMut(usize, T),
    ) {
        let h = b.basis_rows;
        let kmax = b.basis_cols;
        self.check(b);
        if kmax == 0 {
            return;
        }
        if b.dim == 1 {
            if let Some((plan, var)) = self.prepare_fwd_rns(b, x) {
                // qq folds the svec √2 scale into the constant residues, so
                // each output row is a single residue dot against `d`.
                let cns = &self.rns_fwd.as_ref().unwrap().res;
                for t in 0..tri(h) {
                    let v = T::rns_dot(&plan, cns, t * kmax, 1, &var, 0, 1, kmax);
                    store(t, alpha * v);
                }
                return;
            }
        }
        let sqrt2 = if h > 1 { T::SQRT_2() } else { T::zero() };
        let inv_sqrt2 = if h > 0 && b.dim > 1 {
            T::FRAC_1_SQRT_2()
        } else {
            T::zero()
        };
        let q = BorrowedMatrix {
            size: (h, kmax),
            data: b.basis.as_slice(),
            phantom: std::marker::PhantomData,
        };
        let mut p = 0;
        for s in 0..b.dim {
            for r in 0..=s {
                for k in 0..kmax {
                    let weight = b.weights[p + k] * x[b.column_start + p + k];
                    let q_col = &q.data()[k * h..(k + 1) * h];
                    let p_col = &mut self.panel.data_mut()[k * h..(k + 1) * h];
                    for i in 0..h {
                        p_col[i] = q_col[i] * weight;
                    }
                }
                let q_data = q.data();
                let p_data = self.panel.data();
                for j in 0..h {
                    for i in 0..=j {
                        let v = T::dot_fma(
                            (0..kmax).map(|k| (&p_data[i + k * h], &q_data[j + k * h])),
                        );
                        self.square[(i, j)] = v;
                        if r != s {
                            self.square[(j, i)] = v;
                        }
                    }
                }
                for j in 0..h {
                    for i in 0..if r == s { j + 1 } else { h } {
                        let scale = if r != s {
                            inv_sqrt2
                        } else if i != j {
                            sqrt2
                        } else {
                            T::one()
                        };
                        store(
                            tri(s * h + j) + r * h + i,
                            alpha * scale * self.square[(i, j)],
                        );
                    }
                }
                p += kmax;
            }
        }
    }
    fn adjoint_terms(
        &mut self,
        b: &SampledBlock<T>,
        x: &[T],
        alpha: T,
        mut store: impl FnMut(usize, T),
    ) {
        let h = b.basis_rows;
        let kmax = b.basis_cols;
        self.check(b);
        if kmax == 0 {
            return;
        }
        let inv_sqrt2 = if h > 0 && (h > 1 || b.dim > 1) {
            T::FRAC_1_SQRT_2()
        } else {
            T::zero()
        };
        let q = BorrowedMatrix {
            size: (h, kmax),
            data: b.basis.as_slice(),
            phantom: std::marker::PhantomData,
        };
        let mut p = 0;
        let trih = tri(h);
        for s in 0..b.dim {
            for r in 0..=s {
                if r == s {
                    if b.dim == 1 {
                        if let Some((plan, var)) = self.prepare_adj_rns(b, x) {
                            let cns = &self.rns_adj.as_ref().unwrap().res;
                            for k in 0..kmax {
                                let v =
                                    T::rns_dot(&plan, cns, k * trih, 1, &var, 0, 1, trih);
                                store(p + k, alpha * b.weights[p + k] * v);
                            }
                            p += kmax;
                            continue;
                        }
                    }
                    // Diagonal pair: `v_k = q'·S·q` with symmetric `S` reduces
                    // to a flat dot over the block's svec triangle against the
                    // precomputed `wdiag` — half the panel product's work.
                    for k in 0..kmax {
                        let w = &self.wdiag[k * trih..(k + 1) * trih];
                        let v = T::dot_fma((0..h).flat_map(|j| {
                            (0..=j).map(move |i| {
                                (
                                    &x[b.row_start + tri(r * h + j) + r * h + i],
                                    &w[tri(j) + i],
                                )
                            })
                        }));
                        store(p + k, alpha * b.weights[p + k] * v);
                    }
                } else {
                    for j in 0..h {
                        for i in 0..h {
                            let (a, c) = (r * h + i, s * h + j);
                            let scale = if a == c { T::one() } else { inv_sqrt2 };
                            self.square[(i, j)] = x[b.row_start + tri(c) + a] * scale;
                        }
                    }
                    self.panel.mul(&self.square, &q, T::one(), T::zero());
                    for k in 0..kmax {
                        let q_col = &q.data()[k * h..(k + 1) * h];
                        let p_col = &self.panel.data()[k * h..(k + 1) * h];
                        let v = T::dot_fma(q_col.iter().zip(p_col.iter()));
                        store(p + k, alpha * b.weights[p + k] * v);
                    }
                }
                p += kmax;
            }
        }
    }
    fn check(&self, b: &SampledBlock<T>) {
        assert_eq!(self.panel.size(), (b.basis_rows, b.basis_cols));
        assert_eq!(self.square.size(), (b.basis_rows, b.basis_rows));
    }
}
/// Mutable scratch is owned by one caller, independently of shared metadata.
pub struct SampledWorkspace<T> {
    blocks: Vec<SampledBlockWorkspace<T>>,
    // Row/column lanes for the ordinary CSC part. The plan is built once from
    // the immutable pattern and reconfigured only when the pool width changes.
    linear_plan: SparseParallel,
    linear_plan_workers: usize,
}
impl<T: FloatT> SampledWorkspace<T> {
    /// Configure the ordinary-product plan for the current pool width. Called
    /// on every pooled product; the lane plan is rebuilt only on width changes.
    fn linear_product(
        &mut self,
        operator: &SampledOperator<T>,
        transpose: bool,
        y: &mut [T],
        x: &[T],
        alpha: T,
        beta: T,
        pool: Option<&Arc<rayon::ThreadPool>>,
    ) {
        let workers = pool.map_or(1, |p| p.current_num_threads());
        if workers > self.linear_plan_workers {
            self.linear_plan
                .configure(&operator.linear, pool.map(Arc::clone));
            self.linear_plan_workers = workers;
        }
        if self.linear_plan_workers > 1 && self.linear_plan.has_lanes() {
            self.linear_plan
                .product(&operator.linear, transpose, y, x, alpha, beta);
            return;
        }
        if transpose {
            operator.linear.t().gemv(y, x, alpha, beta);
        } else {
            operator.linear.gemv(y, x, alpha, beta);
        }
    }

    fn parallel_eligible(
        &self,
        operator: &SampledOperator<T>,
        pool: Option<&Arc<rayon::ThreadPool>>,
    ) -> bool {
        if pool.map_or(1, |p| p.current_num_threads()) < 2 {
            return false;
        }
        let mut active = 0;
        let mut work = 0u128;
        for b in &operator.blocks {
            if b.basis_cols > 0 {
                active += 1;
                work += b.scheduled_work();
            }
        }
        active > 1 && work >= 8192
    }
    /// Allocate reusable product scratch for the supplied operator.
    pub fn new(operator: &SampledOperator<T>) -> Self {
        Self {
            blocks: operator
                .blocks
                .iter()
                .map(|b| {
                    let h = b.basis_rows;
                    let trih = tri(h);
                    let kmax = b.basis_cols;
                    let sqrt2 = if h > 1 { T::SQRT_2() } else { T::zero() };
                    let mut wdiag = vec![T::zero(); kmax * trih];
                    for k in 0..kmax {
                        for j in 0..h {
                            for i in 0..=j {
                                let c = if i == j { T::one() } else { sqrt2 };
                                wdiag[k * trih + tri(j) + i] =
                                    c * b.basis[i + k * h] * b.basis[j + k * h];
                            }
                        }
                    }
                    SampledBlockWorkspace {
                        forward: Vec::new(),
                        adjoint: Vec::new(),
                        wdiag,
                        panel: Matrix::zeros((h, kmax)),
                        square: Matrix::zeros((h, h)),
                        qq: Vec::new(),
                        dvec: Vec::new(),
                        rns_adj: None,
                        rns_fwd: None,
                        rns_range: None,
                    }
                })
                .collect(),
            linear_plan: SparseParallel::new(&operator.linear),
            linear_plan_workers: 0,
        }
    }
}

/// NT Gram workspace. Its columns retain canonical primitive indices even
/// when basis values or weights are zero.
///
/// The structured update exploits the block-diagonal layout of
/// `I_dim ⊗ Qbasis`: the old code materialized a `side × (dim·count)`
/// matrix `u` whose r-th column block is nonzero only in rows
/// `[r·h, (r+1)·h)` (h = basis_rows) and ran a dense `side²·dim·count`
/// GEMM. The workspace instead stores the packed `h × count` unique basis
/// and computes each output block as `Rinv[:, r·h..] · ub`, a
/// `side·count·h` product — `dim×` less work. Dropped terms are exactly
/// `Rinv·(+0)`; ordered-FMA accumulation makes nonzero results bitwise
/// identical. A `±0` output can differ in sign because skipped zero terms
/// may carry the opposite sign, so zero results are recomputed with the
/// full-side accumulation to preserve bitwise parity.
pub struct SampledSchurWorkspace<T> {
    ub: Matrix<T>,
    v: Matrix<T>,
    gram: Matrix<T>,
    pairs: Vec<(usize, usize)>,
    dim: usize,
    side: usize,
    count: usize,
    plan_threads: usize,
    gemm_tile: usize,
    syrk_tile: usize,
}
impl<T: FloatT> SampledSchurWorkspace<T> {
    /// Allocate NT Gram storage for a validated sampled block.
    pub fn new(b: &SampledBlock<T>) -> Self {
        // Compare complete basis vectors at the working precision. Reuse only
        // exact duplicates; weights and canonical primitive indices stay intact.
        let basis = |k: usize| &b.basis[k * b.basis_rows..(k + 1) * b.basis_rows];
        let mut order: Vec<usize> = (0..b.basis_cols).collect();
        order.sort_by(|&i, &j| basis(i).partial_cmp(basis(j)).unwrap());
        let mut unique = Vec::new();
        let mut map = vec![0; b.basis_cols];
        for k in order {
            if unique.last().map_or(true, |&j| basis(k) != basis(j)) {
                unique.push(k);
            }
            map[k] = unique.len() - 1;
        }
        // Preserve the existing layout on the usual all-distinct input.
        if unique.len() == b.basis_cols {
            for k in 0..b.basis_cols {
                unique[k] = k;
                map[k] = k;
            }
        }
        let count = unique.len();
        let rank = b.dim * count;
        let mut ub = Matrix::zeros((b.basis_rows, count));
        for (k, &original) in unique.iter().enumerate() {
            for i in 0..b.basis_rows {
                ub[(i, k)] = b.basis[i + original * b.basis_rows];
            }
        }
        let mut pairs = Vec::with_capacity(b.weights.len());
        for s in 0..b.dim {
            for r in 0..=s {
                for &k in &map {
                    pairs.push((r * count + k, s * count + k));
                }
            }
        }
        Self {
            ub,
            v: Matrix::zeros((b.side(), rank)),
            gram: Matrix::zeros((rank, rank)),
            pairs,
            dim: b.dim,
            side: b.side(),
            count,
            plan_threads: 0,
            gemm_tile: 0,
            syrk_tile: 0,
        }
    }
    /// The Gram contribution buffer for the rank-sharded exchange.
    pub(crate) fn gram_slice(&self) -> &[T] {
        self.gram.data()
    }
    /// Republish a gathered Gram into this block's workspace.
    pub(crate) fn set_gram(&mut self, data: &[T]) {
        self.gram.data_mut().copy_from_slice(data);
    }
    // Dimensions are fixed by construction; only the current pool width changes.
    pub(crate) fn configure_parallel(&mut self, workers: usize) -> u128 {
        let (side, rank) = (self.side, self.dim * self.count);
        let h = side / self.dim.max(1);
        let words = T::precision_bits().div_ceil(64) as u128;
        // The structured update costs side·count·h per block, dim blocks total.
        let gemm = side as u128 * rank as u128 * h as u128 * words * words;
        let syrk = side as u128 * rank as u128 * (rank + 1) as u128 / 2 * words * words;
        if self.plan_threads != workers {
            self.plan_threads = workers;
            let block_work = gemm / self.dim.max(1) as u128;
            let lanes = workers
                .min(self.count)
                .min((block_work / 4096).min(usize::MAX as u128) as usize);
            self.gemm_tile = if lanes > 1 {
                self.count.div_ceil(lanes)
            } else {
                0
            };
            let lanes = workers
                .min(rank)
                .min((syrk / 4096).min(usize::MAX as u128) as usize);
            self.syrk_tile = if lanes > 1 { rank.div_ceil(lanes) } else { 0 };
        }
        gemm + syrk
    }
    pub(crate) fn has_parallel_columns(&self) -> bool {
        T::precision_bits() > 64 && (self.gemm_tile > 0 || self.syrk_tile > 0)
    }
    /// Update using the NT inverse factor for the block supplied at construction.
    pub(crate) fn update_with_pool(
        &mut self,
        b: &SampledBlock<T>,
        rinv: &Matrix<T>,
        pool: Option<&rayon::ThreadPool>,
    ) {
        assert_eq!(rinv.size(), (b.side(), b.side()));
        assert_eq!(self.side, b.side());
        assert_eq!(self.pairs.len(), b.column_count());
        if b.basis_cols == 0 {
            return;
        }
        let (side, h, count) = (self.side, b.basis_rows, self.count);
        let (side32, h32, count32) = (
            i32::try_from(side).unwrap(),
            i32::try_from(h).unwrap(),
            i32::try_from(count).unwrap(),
        );
        if let Some(pool) = pool {
            self.configure_parallel(pool.current_num_threads());
        }
        // Block-diagonal structure: V[:, r·count..) = Rinv[:, r·h..) · ub.
        let vcols = self.v.data_mut();
        for r in 0..self.dim {
            let a = &rinv.data()[r * h * side..(r + 1) * h * side];
            let ccols = &mut vcols[r * count * side..(r + 1) * count * side];
            match pool {
                Some(pool) => T::xgemm_pool(
                    b'N',
                    b'N',
                    side32,
                    count32,
                    h32,
                    T::one(),
                    a,
                    side32,
                    self.ub.data(),
                    h32,
                    T::zero(),
                    ccols,
                    side32,
                    pool,
                    self.gemm_tile,
                ),
                None => T::xgemm(
                    b'N',
                    b'N',
                    side32,
                    count32,
                    h32,
                    T::one(),
                    a,
                    side32,
                    self.ub.data(),
                    h32,
                    T::zero(),
                    ccols,
                    side32,
                ),
            }
        }
        // Signed-zero parity: a ±0 output may carry the wrong sign because
        // skipped out-of-block terms were ±0 too. Recompute those elements
        // with the full-side ordered accumulation. dim == 1 skips nothing.
        if self.dim > 1 {
            let zero = T::zero();
            for c in 0..self.dim * count {
                let (r, k) = (c / count, c % count);
                for i in 0..side {
                    let slot = &mut vcols[i + c * side];
                    if slot.is_zero() {
                        *slot = T::dot_fma((0..side).map(|j| {
                            let rj = &rinv.data()[i + j * side];
                            if j / h == r {
                                (rj, &self.ub.data()[j - r * h + k * h])
                            } else {
                                (rj, &zero)
                            }
                        }));
                    }
                }
            }
        }
        let rank32 = i32::try_from(self.dim * count).unwrap();
        if let Some(pool) = pool {
            T::xsyrk_pool(
                b'U',
                b'T',
                rank32,
                side32,
                T::one(),
                self.v.data(),
                side32,
                T::zero(),
                self.gram.data_mut(),
                rank32,
                pool,
                self.syrk_tile,
            );
        } else {
            self.gram
                .syrk(&self.v.t(), T::one(), T::zero(), MatrixTriangle::Triu);
        }
    }
    /// Schur entry for zero-based canonical primitive indices in this block.
    pub fn entry(&self, b: &SampledBlock<T>, p: usize, q: usize) -> T {
        let (a, c) = self.pairs[p];
        let (d, e) = self.pairs[q];
        let k = |i: usize, j: usize| self.gram[(i.min(j), i.max(j))];
        let half: T = (0.5).as_T();
        b.weights[p] * b.weights[q] * half * (k(a, d) * k(c, e) + k(a, e) * k(c, d))
    }
}

#[cfg(test)]
#[path = "sampled_tests.rs"]
mod tests;
