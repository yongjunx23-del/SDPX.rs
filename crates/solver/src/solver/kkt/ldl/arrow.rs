#![allow(non_snake_case)]
//! Multi-leaf arrow LDLᵀ solver for quasidefinite KKT systems whose
//! positive-sign variables decompose into several structurally disconnected
//! components.  Each component factors as an independent dense leaf while the
//! negative-sign variables form a small border Schur complement:
//!
//! K = [diag(H_i)  B_i;  B_iᵀ  C],   H_i ≻ 0,   C ≺ 0.
//!
//! Compared to the general sparse QDLDL this trades fill-reducing orderings
//! for dense leaf factorizations that parallelize trivially and reuse the
//! same coupling transforms across refactorizations.  Eligibility is decided
//! once from the structure (stored zeros count as edges); refactor failures
//! delegate to a lazily constructed QDLDL fallback. Local SOC and scalar-bound
//! leaves also admit mixed signs, with explicit signs for the retained border.
use crate::algebra::*;
use crate::solver::core::CoreSettings;
use crate::solver::kkt::direct::{BoxedDirectLDLSolver, DirectLDLSolver};
use crate::solver::kkt::{HasLinearSolverInfo, LinearSolverInfo};
use rayon::prelude::*;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

mod local_bounds;
mod local_cones;
mod local_soc;
mod shared_soc;

#[derive(Clone, Copy)]
enum LocalStructure {
    Soc,
    SharedSoc,
    Bounds,
    // Any mix of small orthant/SOC leaves; couplings stored per leaf over
    // its own border columns (see `local_cones`).
    Cones,
}

/// Dense working-set cap of the local arrow structures, and the size up to
/// which generic leaves are always admitted. Beyond it a generic arrow must
/// stay within `ARROW_KKT_FACTOR` times the stored KKT, so sparse structure
/// is never densified without bound; a dense border coupling (as in sampled
/// bootstrap problems) fills QDLDL's factor just as much.
const ARROW_MAX_BYTES: u128 = 512 * 1024 * 1024;
const ARROW_KKT_FACTOR: u128 = 8;
/// Smallest single dense leaf admitted to the arrow backend.
const SINGLE_LEAF_MIN: usize = 16;
/// Trailing columns below which a split factor step runs serially.
const SPLIT_FACTOR_MIN: usize = 32;
/// Per-step factor tasks are small: split a leaf factor only with at least
/// this many threads per leaf (at ~2 per leaf it measured slower), and the
/// border factor only from this dimension.
const SPLIT_FACTOR_THREADS_PER_LEAF: usize = 4;
const SPLIT_BORDER_FACTOR_MIN: usize = 128;

fn find(parent: &mut [usize], mut i: usize) -> usize {
    while parent[i] != i {
        parent[i] = parent[parent[i]];
        i = parent[i];
    }
    i
}

/// Unpivoted dense LDLᵀ with the same per-pivot dynamic regularization
/// semantics as QDLDL: a pivot that violates its expected sign by more than
/// `eps` is clamped to `delta * sign` and counted.
struct DenseLeaf<T> {
    n: usize,
    l: Vec<T>,
    dinv: Vec<T>,
}

impl<T: FloatT> DenseLeaf<T> {
    fn new(n: usize) -> Self {
        Self {
            n,
            l: vec![T::zero(); n * n],
            dinv: vec![T::zero(); n],
        }
    }

    fn pivot(mut d: T, sign: T, reg: Option<(T, T)>, count: &mut usize) -> Result<T, &'static str> {
        if !d.is_finite() {
            return Err("nonfinite_pivot");
        }
        if let Some((eps, delta)) = reg {
            if d * sign < eps {
                d = delta * sign;
                *count += 1;
            }
        }
        if d.is_zero() {
            return Err("zero_pivot");
        }
        Ok(d)
    }

    // Unrolled signed LDL for one/two scalar bounds. Keep the generic
    // factor's pivot order and fused updates, including structural zeros.
    fn factor_bounds(
        &mut self,
        a: &[T],
        reg: Option<(T, T)>,
        count: &mut usize,
    ) -> Result<(), &'static str> {
        let g = self.n;
        debug_assert!(g == 2 || g == 3);
        self.l.copy_from_slice(a);
        let d0 = Self::pivot(self.l[0], -T::one(), reg, count)?;
        self.dinv[0] = T::one() / d0;
        self.l[0] = T::one();
        self.l[1] *= self.dinv[0];
        if g == 3 {
            self.l[2] *= self.dinv[0];
        }
        let v = self.l[1] * d0;
        self.l[g + 1] = (-self.l[1]).mul_add(v, self.l[g + 1]);
        if g == 3 {
            self.l[g + 2] = (-self.l[2]).mul_add(v, self.l[g + 2]);
            self.l[2 * g + 2] = (-self.l[2]).mul_add(self.l[2] * d0, self.l[2 * g + 2]);
            let d1 = Self::pivot(self.l[g + 1], -T::one(), reg, count)?;
            self.dinv[1] = T::one() / d1;
            self.l[g + 1] = T::one();
            self.l[g + 2] *= self.dinv[1];
            self.l[2 * g + 2] = (-self.l[g + 2]).mul_add(self.l[g + 2] * d1, self.l[2 * g + 2]);
        }
        let last = g - 1;
        let d = Self::pivot(self.l[last + last * g], T::one(), reg, count)?;
        self.dinv[last] = T::one() / d;
        self.l[last + last * g] = T::one();
        Ok(())
    }

    /// `split`: run each step's trailing update over columns in parallel
    /// (inside the caller's pool). Every entry receives the same sequence of
    /// fused updates, so the factor is bitwise identical.
    #[cfg(test)]
    fn factor(
        &mut self,
        a: &[T],
        sign: i8,
        reg: Option<(T, T)>,
        regularize_count: &mut usize,
        split: bool,
    ) -> Result<(), &'static str> {
        self.factor_signed(a, &[sign], reg, regularize_count, split)
    }

    fn factor_signed(
        &mut self,
        a: &[T],
        signs: &[i8],
        reg: Option<(T, T)>,
        regularize_count: &mut usize,
        split: bool,
    ) -> Result<(), &'static str> {
        self.l.copy_from_slice(a);
        let n = self.n;
        for k in 0..n {
            let s = T::from_i8(signs[if signs.len() == 1 { 0 } else { k }]).unwrap();
            let d = Self::pivot(self.l[k + k * n], s, reg, regularize_count)?;
            self.dinv[k] = T::one() / d;
            self.l[k + k * n] = T::one();
            for i in k + 1..n {
                self.l[i + k * n] *= self.dinv[k];
            }
            if split && n - k > SPLIT_FACTOR_MIN {
                let (head, tail) = self.l.split_at_mut((k + 1) * n);
                let pivot = &head[k * n..];
                tail.par_chunks_mut(n)
                    .enumerate()
                    .for_each(|(offset, column)| {
                        let j = k + 1 + offset;
                        let v = pivot[j] * d;
                        for i in j..n {
                            column[i] = (-pivot[i]).mul_add(v, column[i]);
                        }
                    });
                continue;
            }
            for j in k + 1..n {
                let v = self.l[j + k * n] * d;
                for i in j..n {
                    self.l[i + j * n] = (-self.l[i + k * n]).mul_add(v, self.l[i + j * n]);
                }
            }
        }
        Ok(())
    }

    fn forward(&self, x: &mut [T]) {
        self.forward_suffix(x, 0);
    }

    // Omitted leading RHS coordinates are structurally zero. Forward solve
    // leaves them zero, so compact coupling panels need only the suffix.
    fn forward_suffix(&self, x: &mut [T], start: usize) {
        for i in 0..x.len() {
            for k in 0..i {
                x[i] = (-self.l[i + start + (k + start) * self.n]).mul_add(x[k], x[i]);
            }
        }
    }

    fn backward(&self, x: &mut [T]) {
        for i in (0..self.n).rev() {
            for k in i + 1..self.n {
                x[i] = (-self.l[k + i * self.n]).mul_add(x[k], x[i]);
            }
        }
    }

    // Row-major RHS lanes keep each factor coefficient live across columns.
    // Every column retains the single-RHS accumulation order.
    fn forward_many(&self, x: &mut [T], cols: usize) {
        for i in 0..self.n {
            for k in 0..i {
                let l = -self.l[i + k * self.n];
                for c in 0..cols {
                    x[i * cols + c] = l.mul_add(x[k * cols + c], x[i * cols + c]);
                }
            }
        }
    }
    fn backward_many(&self, x: &mut [T], cols: usize) {
        for i in (0..self.n).rev() {
            for k in i + 1..self.n {
                let l = -self.l[k + i * self.n];
                for c in 0..cols {
                    x[i * cols + c] = l.mul_add(x[k * cols + c], x[i * cols + c]);
                }
            }
        }
    }

    /// Apply a batch kernel to `chunks` column ranges of the row-interleaved
    /// panel `x` (entry (i, c) at `i * cols + c`) in parallel. Columns are
    /// independent, so the result equals one call over all columns.
    fn chunked(
        &self,
        x: &mut [T],
        cols: usize,
        chunks: usize,
        kernel: impl Fn(&Self, &mut [T], usize) + Sync,
    ) {
        let n = self.n;
        let source: &[T] = x;
        let parts: Vec<Vec<T>> = (0..chunks)
            .into_par_iter()
            .map(|p| {
                let (c0, c1) = (cols * p / chunks, cols * (p + 1) / chunks);
                let mut part = Vec::with_capacity(n * (c1 - c0));
                for i in 0..n {
                    part.extend_from_slice(&source[i * cols + c0..i * cols + c1]);
                }
                kernel(self, &mut part, c1 - c0);
                part
            })
            .collect();
        for (p, part) in parts.iter().enumerate() {
            let (c0, c1) = (cols * p / chunks, cols * (p + 1) / chunks);
            let width = c1 - c0;
            for i in 0..n {
                x[i * cols + c0..i * cols + c1].copy_from_slice(&part[i * width..(i + 1) * width]);
            }
        }
    }

    fn solve(&self, x: &mut [T]) {
        self.forward(x);
        for (x, d) in x.iter_mut().zip(&self.dinv) {
            *x *= *d;
        }
        self.backward(x);
    }

    /// `solve` with the forward sweep split over `pool`. Rows are processed
    /// in blocks: once a block is final, every later row applies that
    /// block's columns in ascending order (rows are independent), then the
    /// next block finishes serially. Each entry sees the serial FMA order,
    /// so results are bitwise identical. The backward sweep stays serial.
    fn solve_pooled(&self, x: &mut [T], pool: Option<&rayon::ThreadPool>) {
        const BLOCK: usize = 32;
        let n = self.n;
        match pool.filter(|p| p.current_num_threads() > 1 && n >= 8 * BLOCK) {
            Some(pool) => pool.install(|| {
                let mut r0 = 0;
                while r0 < n {
                    let r1 = (r0 + BLOCK).min(n);
                    for i in r0..r1 {
                        for k in r0..i {
                            x[i] = (-self.l[i + k * n]).mul_add(x[k], x[i]);
                        }
                    }
                    let (head, tail) = x.split_at_mut(r1);
                    let done = &head[r0..r1];
                    tail.par_chunks_mut(16).enumerate().for_each(|(c, rows)| {
                        for (j, v) in rows.iter_mut().enumerate() {
                            let i = r1 + c * 16 + j;
                            for (k, &xk) in (r0..r1).zip(done) {
                                *v = (-self.l[i + k * n]).mul_add(xk, *v);
                            }
                        }
                    });
                    r0 = r1;
                }
            }),
            None => self.forward(x),
        }
        for (x, d) in x.iter_mut().zip(&self.dinv) {
            *x *= *d;
        }
        self.backward(x);
    }
}

struct Leaf<T> {
    ids: Vec<usize>,
    signs: Vec<i8>,
    // Structurally zero prefix of the border coupling (zero for shared SOCs).
    coupling_start: usize,
    // Border columns with a structural B entry (all for unpacked local leaves).
    // Packed bounds use their panel maps instead.
    // Other Y columns stay exactly zero, so they are skipped everywhere.
    coupled: Vec<usize>,
    couples: Vec<bool>,
    // Local leaves store B/Y/Z over `coupled` only (column c of the suffix
    // at `c * width`); generic leaves keep all t border columns.
    packed: bool,
    h: Vec<T>,
    b: Vec<T>,
    factor: DenseLeaf<T>,
    y: Vec<T>,
    z: Vec<T>,
    w: Vec<T>,
    // Only the coupled suffix needs a diagonally scaled forward value.
    v: Vec<T>,
    batch_w: Vec<T>,
    batch_v: Vec<T>,
}

impl<T: FloatT> Leaf<T> {
    fn new(ids: Vec<usize>, t: usize) -> Self {
        let g = ids.len();
        Self {
            ids,
            signs: Vec::new(),
            coupling_start: 0,
            coupled: (0..t).collect(),
            couples: vec![true; t],
            packed: false,
            h: vec![T::zero(); g * g],
            b: vec![T::zero(); g * t],
            factor: DenseLeaf::new(g),
            y: vec![T::zero(); g * t],
            // Generic leaves form Z = D⁻¹Y one column at a time while the
            // border Schur is assembled; only local structures store it.
            z: Vec::new(),
            w: Vec::new(),
            v: Vec::new(),
            batch_w: Vec::new(),
            batch_v: Vec::new(),
        }
    }

    fn coupling_index(&self, row: usize, col: usize) -> usize {
        row - self.coupling_start + self.column(col) * (self.ids.len() - self.coupling_start)
    }

    /// Storage column of border column `col`.
    fn column(&self, col: usize) -> usize {
        if self.packed {
            self.coupled.binary_search(&col).unwrap()
        } else {
            col
        }
    }

    /// Raw coupling entry: with `b` shorter than `y`, the suffix's first row
    /// (L = 1) stays raw in Y and `b` holds the remaining rows.
    fn raw_coupling(&mut self, row: usize, col: usize) -> &mut T {
        if self.b.len() < self.y.len() {
            let width = self.ids.len() - self.coupling_start;
            let (r, c) = (row - self.coupling_start, self.column(col));
            if r == 0 {
                &mut self.y[c * width]
            } else {
                &mut self.b[r - 1 + c * (width - 1)]
            }
        } else {
            let index = self.coupling_index(row, col);
            &mut self.b[index]
        }
    }

    fn coupling_values(&self) -> &[T] {
        if self.y.is_empty() {
            &self.b
        } else {
            &self.y
        }
    }

    /// H = LDLᵀ and Y = L⁻¹B; the border Schur S = C - ΣYᵀD⁻¹Y is formed by
    /// `subtract_contribution`. `split`: the pool has more threads than
    /// leaves, so the border columns of Y also run in parallel (each column's
    /// arithmetic is unchanged, so results are bitwise identical).
    fn refactor(
        &mut self,
        t: usize,
        reg: Option<(T, T)>,
        regularize_count: &mut usize,
        split: bool,
        split_factor: bool,
    ) -> Result<(), &'static str> {
        let g = self.ids.len();
        // A scalar primal suffix has L=1: Y=B. Packed bound panels own Z,
        // so these leaves need neither a duplicate Y nor a duplicate Z (nor
        // stored signs).
        if self.signs.is_empty() {
            return self.factor.factor_bounds(&self.h, reg, regularize_count);
        }
        self.factor
            .factor_signed(&self.h, &self.signs, reg, regularize_count, split_factor)?;
        if self.y.is_empty() {
            return Ok(());
        }
        let start = self.coupling_start;
        let width = g - start;
        if self.b.len() < self.y.len() {
            // The suffix's first row has L=1 and stays raw in Y.
            if width > 1 {
                for (column, raw) in self.y.chunks_mut(width).zip(self.b.chunks(width - 1)) {
                    column[1..].copy_from_slice(raw);
                }
            }
        } else {
            self.y.copy_from_slice(&self.b);
        }
        let (factor, couples, packed) = (&self.factor, &self.couples, self.packed);
        if split && width > 0 {
            self.y
                .par_chunks_mut(width)
                .enumerate()
                .filter(|(j, _)| packed || couples[*j])
                .for_each(|(_, column)| factor.forward_suffix(column, start));
        } else if packed {
            for column in self.y.chunks_mut(width) {
                factor.forward_suffix(column, start);
            }
        } else {
            for &j in &self.coupled {
                factor.forward_suffix(&mut self.y[j * width..(j + 1) * width], start);
            }
        }
        if !self.z.is_empty() {
            let columns = if packed { self.coupled.len() } else { t };
            for j in 0..columns {
                for i in 0..width {
                    self.z[i + j * width] = self.y[i + j * width] * self.factor.dinv[i + start];
                }
            }
        }
        Ok(())
    }

    /// `s -= YᵀD⁻¹Y` for a factored generic leaf (`t × t`, both triangles).
    /// Each entry is the dot of a Y column with the rounded column
    /// `z_j = y_j ⊙ dinv`, as one exact accumulation; entry (r, c) is owned by
    /// column max(r, c), so columns update disjoint entries in parallel.
    fn first_many(&mut self, rhs: &[T], n: usize, cols: usize, chunks: usize) {
        let g = self.ids.len();
        self.batch_w.resize(g * cols, T::zero());
        if self.batch_v.is_empty() {
            self.batch_v.reserve_exact((g - self.coupling_start) * cols);
        }
        self.batch_v
            .resize((g - self.coupling_start) * cols, T::zero());
        for (i, &id) in self.ids.iter().enumerate() {
            for c in 0..cols {
                self.batch_w[i * cols + c] = rhs[c * n + id];
            }
        }
        if chunks > 1 {
            self.factor
                .chunked(&mut self.batch_w, cols, chunks, DenseLeaf::forward_many);
        } else {
            self.factor.forward_many(&mut self.batch_w, cols);
        }
        for i in self.coupling_start..g {
            for c in 0..cols {
                self.batch_v[(i - self.coupling_start) * cols + c] =
                    self.batch_w[i * cols + c] * self.factor.dinv[i];
            }
        }
    }
    fn second_many(&mut self, xt: &[T], cols: usize, chunks: usize) {
        let g = self.ids.len();
        let start = self.coupling_start;
        let width = g - start;
        let values = if self.y.is_empty() { &self.b } else { &self.y };
        let (coupled, dinv, packed) = (&self.coupled, &self.factor.dinv, self.packed);
        let update = |(i, row): (usize, &mut [T])| {
            for (c, v) in row.iter_mut().enumerate() {
                let s = if i < start {
                    T::zero()
                } else {
                    T::dot_fma(coupled.iter().enumerate().map(|(p, &j)| {
                        let column = if packed { p } else { j };
                        (&values[i - start + column * width], &xt[j * cols + c])
                    }))
                };
                *v = (*v - s) * dinv[i];
            }
        };
        if rayon::current_thread_index().is_some() && g * coupled.len() * cols >= 1 << 14 {
            self.batch_w
                .par_chunks_mut(cols)
                .enumerate()
                .with_min_len(8)
                .for_each(update);
        } else {
            self.batch_w.chunks_mut(cols).enumerate().for_each(update);
        }
        if chunks > 1 {
            self.factor
                .chunked(&mut self.batch_w, cols, chunks, DenseLeaf::backward_many);
        } else {
            self.factor.backward_many(&mut self.batch_w, cols);
        }
    }

    fn first_solve(&mut self, rhs: &[T]) {
        if self.w.is_empty() {
            let g = self.ids.len();
            self.w.resize(g, T::zero());
            self.v = vec![T::zero(); g - self.coupling_start];
        }
        for (w, &id) in self.w.iter_mut().zip(&self.ids) {
            *w = rhs[id];
        }
        self.factor.forward(&mut self.w);
        for i in self.coupling_start..self.ids.len() {
            self.v[i - self.coupling_start] = self.w[i] * self.factor.dinv[i];
        }
    }

    fn second_solve(&mut self, xt: &[T]) {
        let g = self.ids.len();
        let start = self.coupling_start;
        let width = g - start;
        let values = if self.y.is_empty() { &self.b } else { &self.y };
        let (coupled, dinv, packed) = (&self.coupled, &self.factor.dinv, self.packed);
        let update = |(i, v): (usize, &mut T)| {
            let s = if i < start {
                T::zero()
            } else {
                T::dot_fma(coupled.iter().enumerate().map(|(p, &j)| {
                    let column = if packed { p } else { j };
                    (&values[i - start + column * width], &xt[j])
                }))
            };
            *v = (*v - s) * dinv[i];
        };
        // Only on a pool worker: a serial caller must not reach the global pool.
        if rayon::current_thread_index().is_some() && g * coupled.len() >= 1 << 14 {
            self.w
                .par_iter_mut()
                .enumerate()
                .with_min_len(8)
                .for_each(update);
        } else {
            self.w.iter_mut().enumerate().for_each(update);
        }
        self.factor.backward(&mut self.w);
    }

    // Same arithmetic as second_solve/second_many, but the coupling column
    // comes from the packed panels: y[index + j*n] == coupling_index(i, j)
    // for a width-1 bound leaf, so every output is bitwise identical.
    fn second_solve_bounds(&mut self, xt: &[T], y: &[T], index: usize, n: usize) {
        let g = self.ids.len();
        for i in 0..g {
            let s = if i < self.coupling_start {
                T::zero()
            } else {
                T::dot_fma((0..xt.len()).map(|j| (&y[index + j * n], &xt[j])))
            };
            self.w[i] = (self.w[i] - s) * self.factor.dinv[i];
        }
        self.factor.backward(&mut self.w);
    }

    fn second_many_bounds(
        &mut self,
        xt: &[T],
        y: &[T],
        index: usize,
        n: usize,
        cols: usize,
        chunks: usize,
    ) {
        let g = self.ids.len();
        for i in 0..g {
            for c in 0..cols {
                let s = if i < self.coupling_start {
                    T::zero()
                } else {
                    T::dot_fma((0..xt.len() / cols).map(|j| (&y[index + j * n], &xt[j * cols + c])))
                };
                self.batch_w[i * cols + c] = (self.batch_w[i * cols + c] - s) * self.factor.dinv[i];
            }
        }
        if chunks > 1 {
            self.factor
                .chunked(&mut self.batch_w, cols, chunks, DenseLeaf::backward_many);
        } else {
            self.factor.backward_many(&mut self.batch_w, cols);
        }
    }
}

/// `s -= Σ_leaves YᵀD⁻¹Y` on the upper triangle of the `t × t` border Schur
/// (the caller mirrors it). One task per border column walks the leaves in
/// order, so every entry receives the leaf contributions in leaf order (the
/// serial arithmetic) with no per-leaf barrier. Each term is the dot of a Y
/// column with the rounded `z_j = y_j ⊙ dinv`, one exact accumulation.
/// Columns run longest-first: later columns carry more pairs.
fn subtract_contributions<T: FloatT>(
    leaves: &[Leaf<T>],
    s: &mut [T],
    t: usize,
    pool: Option<&rayon::ThreadPool>,
) {
    if t == 0 {
        return;
    }
    // Above the inline-dot precisions, batch substantial products only
    // when result storage is proportional to Y and outer work fills the pool.
    let width = pool.map_or(1, |p| p.current_num_threads());
    let eligible = |leaf: &Leaf<T>| {
        let (g, q) = (leaf.ids.len(), leaf.coupled.len());
        T::precision_bits() > 256
            && T::residue_blas_applies(q, q, g)
            && g * q * q >= 1 << 21
            && q * q <= 8 * g * t
    };
    if leaves.len() >= width && leaves.iter().any(eligible) {
        let build = |leaf: &Leaf<T>| {
            let (g, q) = (leaf.ids.len(), leaf.coupled.len());
            let mut packed = Vec::new();
            let y = if q == t {
                &leaf.y[..]
            } else {
                packed.reserve_exact(g * q);
                for &j in &leaf.coupled {
                    packed.extend_from_slice(&leaf.y[j * g..(j + 1) * g]);
                }
                &packed[..]
            };
            let mut z = Vec::with_capacity(g * q);
            for source in y.chunks(g) {
                z.extend(source.iter().zip(&leaf.factor.dinv).map(|(&a, &d)| a * d));
            }
            let mut product = vec![T::zero(); triangular_number(q)];
            let timer = crate::receipt::start();
            let exact =
                T::xgemm_upper_exact(b'T', b'N', q, q, g, y, g, &z, g, &mut product, None, None);
            crate::receipt::finish("arrow.residue_contribution", timer);
            if !exact {
                for c in 0..q {
                    for r in 0..=c {
                        product[r + triangular_number(c)] =
                            T::dot_fma(y[r * g..(r + 1) * g].iter().zip(&z[c * g..(c + 1) * g]));
                    }
                }
            }
            product
        };
        for group in leaves.chunk_by(|a, b| eligible(a) == eligible(b)) {
            if !eligible(&group[0]) || group.len() < width {
                // Keep short leaves column-parallel and avoid square result
                // storage, preserving their position in every entry's sum.
                subtract_contributions(group, s, t, pool);
                continue;
            }
            for batch in group.chunks(width) {
                if batch.len() < width {
                    subtract_contributions(batch, s, t, pool);
                    continue;
                }
                let products: Vec<Vec<T>> = match pool {
                    Some(p) => p.install(|| batch.par_iter().map(build).collect()),
                    None => batch.iter().map(build).collect(),
                };
                // Columns are independent; each entry still subtracts leaves
                // in the original order before the next batch begins.
                let scatter = |(j, col): (usize, &mut [T])| {
                    for (leaf, product) in batch.iter().zip(&products) {
                        if !leaf.couples[j] {
                            continue;
                        }
                        let c = leaf.coupled.binary_search(&j).unwrap();
                        for (r, &i) in leaf.coupled[..=c].iter().enumerate() {
                            col[i] -= product[r + triangular_number(c)];
                        }
                    }
                };
                match pool {
                    Some(p) if t > 1 => p.install(|| {
                        s.par_chunks_mut(t)
                            .enumerate()
                            .with_max_len(1)
                            .for_each(scatter)
                    }),
                    _ => s.chunks_mut(t).enumerate().for_each(scatter),
                }
            }
        }
        return;
    }
    let column = |j: usize, col: &mut [T]| {
        for leaf in leaves {
            if !leaf.couples[j] {
                continue;
            }
            let g = leaf.ids.len();
            let (y, dinv) = (&leaf.y, &leaf.factor.dinv);
            let c = leaf.coupled.binary_search(&j).unwrap();
            let z: Vec<T> = (0..g).map(|k| y[k + j * g] * dinv[k]).collect();
            for &i in &leaf.coupled[..=c] {
                col[i] -= T::dot_fma(y[i * g..(i + 1) * g].iter().zip(&z));
            }
        }
    };
    let mut cols: Vec<(usize, &mut [T])> = s.chunks_mut(t).enumerate().collect();
    match pool {
        Some(pool) if t > 1 => {
            cols.reverse();
            pool.install(|| {
                cols.par_iter_mut()
                    .with_max_len(1)
                    .for_each(|(j, col)| column(*j, col))
            });
        }
        _ => cols.into_iter().for_each(|(j, col)| column(j, col)),
    }
}

/// Generic leaves shared over MPI ranks. Each rank factors and solves one
/// contiguous, cost-balanced range of leaves; border partial sums are added
/// in rank order, so one rank reproduces the serial arithmetic exactly and a
/// fixed rank count is deterministic.
struct LeafRanks {
    world: crate::mpi::World,
    /// `(first leaf, count)` per rank.
    parts: Vec<(usize, usize)>,
    /// Prefix sums of leaf sizes, for gathering leaf solution values.
    offsets: Vec<usize>,
}

impl LeafRanks {
    fn owned(&self) -> std::ops::Range<usize> {
        let (first, len) = self.parts[self.world.rank()];
        first..first + len
    }

    fn is_root(&self) -> bool {
        self.world.rank() == 0
    }

    /// `out = Σ_ranks local` (equal lengths), added in rank order.
    fn sum<T: FloatT>(&self, local: &[T], out: &mut [T]) {
        let len = local.len();
        if len == 0 {
            return;
        }
        let size = self.world.size();
        let ranges: Vec<(usize, usize)> = (0..size).map(|r| (r * len, len)).collect();
        let mut all = vec![T::zero(); size * len];
        self.world
            .gather_slice(crate::mpi::SITE_ARROW, local, &ranges, &mut all);
        out.copy_from_slice(&all[..len]);
        for part in all.chunks(len).skip(1) {
            for (o, &v) in out.iter_mut().zip(part) {
                *o += v;
            }
        }
    }

    /// Gather every rank's leaf values (`cols` per leaf entry, in leaf order).
    fn gather_leaves<T: FloatT>(&self, local: &[T], cols: usize, out: &mut [T]) {
        let ranges: Vec<(usize, usize)> = self
            .parts
            .iter()
            .map(|&(first, len)| {
                let begin = self.offsets[first];
                (begin * cols, (self.offsets[first + len] - begin) * cols)
            })
            .collect();
        self.world
            .gather_slice(crate::mpi::SITE_ARROW, local, &ranges, out);
    }
}

/// Coupling terms of border column `j` over local leaves: each leaf's
/// suffix rows against `value(leaf, row)`, class by class in leaf order.
fn class_terms<'a, T: FloatT>(
    classes: &'a [CouplingClass],
    leaves: &'a [Leaf<T>],
    j: usize,
    value: impl Fn(&'a Leaf<T>, usize) -> &'a T + Copy + 'a,
) -> impl Iterator<Item = (&'a T, &'a T)> + 'a {
    classes
        .iter()
        .filter(move |class| class.position[j] != usize::MAX)
        .flat_map(move |class| {
            let p = class.position[j];
            class.leaves.iter().flat_map(move |&l| {
                let leaf = &leaves[l];
                let width = leaf.ids.len() - leaf.coupling_start;
                (0..width).map(move |i| (&leaf.coupling_values()[i + p * width], value(leaf, i)))
            })
        })
}

/// Local leaves sharing one list of border columns.
struct CouplingClass {
    /// Storage column of each border column (`usize::MAX` when absent).
    position: Vec<usize>,
    leaves: Vec<usize>,
}

pub struct ArrowLDLSolver<T: FloatT> {
    // Only the fixed sparse pattern is global. Numerical entries live in
    // their leaf H/B or border C; the parent retains the original KKT for IR.
    colptr: Vec<usize>,
    rowval: Vec<usize>,
    signs: Vec<i8>,
    settings: CoreSettings<T>,
    n: usize,
    trunk: Vec<usize>,
    owner: Vec<usize>,
    local: Vec<usize>,
    leaves: Vec<Leaf<T>>,
    c: Vec<T>,
    s: Vec<T>,
    tf: DenseLeaf<T>,
    tx: Vec<T>,
    batch_tx: Vec<T>,
    pool: Option<Arc<rayon::ThreadPool>>,
    regularize_count: usize,
    fallback: Option<BoxedDirectLDLSolver<T>>,
    use_arrow: bool,
    // Shared-SOC leaf primal columns: eliminated after their SPD cone block,
    // their pivot P_jj + aᵀH⁻¹a is positive without the static shift.
    unshifted: Vec<usize>,
    shift_exempt: bool,
    local_structure: Option<LocalStructure>,
    // Local leaves grouped by equal border columns, in leaf order.
    classes: Vec<CouplingClass>,
    border_signs: Vec<i8>,
    exact_bound_panels: Option<local_bounds::ExactBoundPanels<T>>,
    #[cfg(feature = "faer-sparse")]
    bound_panels: Option<local_bounds::BoundPanels>,
    ranks: Option<LeafRanks>,
}

impl<T: FloatT> ArrowLDLSolver<T> {
    /// Eligibility is structural only: connected components of the
    /// positive-sign subgraph become leaves, negative-sign variables become
    /// the border.  Returns `None` when the shape is not exploitable, so the
    /// caller transparently keeps QDLDL.
    pub fn try_new(k: &CscMatrix<T>, signs: &[i8], settings: &CoreSettings<T>) -> Option<Self> {
        let n = k.n;
        debug_assert!(k.m == n && signs.len() == n && signs.iter().all(|&s| s == 1 || s == -1));
        if n == 0 {
            return None;
        }
        let mut parent: Vec<usize> = (0..n).collect();
        // Stored positive-positive entries (upper triangle with diagonal).
        let mut positive_entries = 0usize;
        for j in 0..n {
            for q in k.colptr[j]..k.colptr[j + 1] {
                let i = k.rowval[q];
                if signs[i] > 0 && signs[j] > 0 {
                    positive_entries += 1;
                    let a = find(&mut parent, i);
                    let b = find(&mut parent, j);
                    parent[a] = b;
                }
            }
        }
        let mut components: std::collections::BTreeMap<usize, Vec<usize>> =
            std::collections::BTreeMap::new();
        let mut trunk = Vec::new();
        for (i, &s) in signs.iter().enumerate() {
            if s < 0 {
                trunk.push(i);
            } else {
                let r = find(&mut parent, i);
                components.entry(r).or_default().push(i);
            }
        }
        let mut groups: Vec<Vec<usize>> = components.into_values().collect();
        groups.sort_by_key(|g| g[0]);
        let t = trunk.len();
        let n_pos = (n - t) as u128;
        let cells: u128 = groups
            .iter()
            .map(|g| {
                let g = g.len() as u128;
                // h and L; b and y; solve vectors.
                2 * g * g + 2 * g * t as u128 + 8 * g
            })
            .sum::<u128>()
            + 4 * (t as u128).pow(2)
            + 8 * n as u128;
        let max_bytes = ARROW_MAX_BYTES
            .max(ARROW_KKT_FACTOR * k.nnz() as u128 * std::mem::size_of::<T>() as u128);
        // Dense work estimates: leaf factorizations ~ sum(g^3)/3, coupling
        // transforms ~ n_pos*t^2/2, trunk factor ~ t^3/3.  Arrow only pays
        // when real dense leaf blocks exist to amortize that dense
        // coupling/trunk work; singleton-leaf structures (e.g. a diagonal
        // positive block in an LP) degenerate to a dense Schur assembly and
        // lose to sparse QDLDL by an order of magnitude.
        let leaf_work: u128 = groups.iter().map(|g| (g.len() as u128).pow(3)).sum();
        let border_work = n_pos * (t as u128).pow(2) + (t as u128).pow(3);
        let work_ok = border_work <= 24 * leaf_work.max(1);
        // A single leaf pays only when it is a large dense block (an owner
        // rank holding one sampled PSD block): the dense, pool-parallel leaf
        // factor then replaces scalar sparse LDL. Sparse single components
        // (e.g. a connected LP) stay on the sparse backend.
        let leaves_ok = groups.len() >= 2
            || groups.first().is_some_and(|g| {
                g.len() >= SINGLE_LEAF_MIN && 2 * positive_entries >= g.len() * (g.len() + 1) / 2
            });
        if crate::receipt::profile_requested() {
            // Observation-only grouping stats (plan PR-06): positive-sign
            // components, leaf size spread, border size, leaf-border coupling
            // edges and the dense working-set estimate.
            let mut sizes: Vec<usize> = groups.iter().map(|g| g.len()).collect();
            sizes.sort_unstable();
            let coupling = (0..n)
                .flat_map(|j| (k.colptr[j]..k.colptr[j + 1]).map(move |q| (k.rowval[q], j)))
                .filter(|&(i, j)| signs[i] != signs[j])
                .count();
            eprintln!(
                "GROUP_STATS n={n} components={} leaf_min={} leaf_med={} leaf_max={} border={t} coupling_nnz={coupling} dense_mib={:.1} leaf_work={leaf_work} border_work={border_work} eligible={}",
                groups.len(),
                sizes.first().copied().unwrap_or(0),
                sizes.get(sizes.len() / 2).copied().unwrap_or(0),
                sizes.last().copied().unwrap_or(0),
                cells as f64 * std::mem::size_of::<T>() as f64 / 1048576.0,
                leaves_ok && work_ok && cells * std::mem::size_of::<T>() as u128 <= max_bytes,
            );
        }
        if !leaves_ok {
            return None;
        }
        if !work_ok {
            return None;
        }
        if cells * std::mem::size_of::<T>() as u128 > max_bytes {
            return None;
        }
        Some(Self::from_groups(k, signs, settings, groups, trunk, None))
    }

    /// Leaves made of an SOC block followed by one primal column with a
    /// nonzero coupling to it (upper-triangle entries in the cone columns).
    fn single_primal_leaf_columns(
        k: &CscMatrix<T>,
        signs: &[i8],
        leaves: &[Leaf<T>],
    ) -> Vec<usize> {
        leaves
            .iter()
            .filter_map(|leaf| {
                let (&p, cone) = leaf.ids.split_last()?;
                if signs[p] != 1 || cone.is_empty() || cone.iter().any(|&c| signs[c] != -1 || c < p)
                {
                    return None;
                }
                cone.iter()
                    .any(|&c| {
                        (k.colptr[c]..k.colptr[c + 1])
                            .any(|q| k.rowval[q] == p && k.nzval[q] != T::zero())
                    })
                    .then_some(p)
            })
            .collect()
    }

    fn from_groups(
        k: &CscMatrix<T>,
        signs: &[i8],
        settings: &CoreSettings<T>,
        groups: Vec<Vec<usize>>,
        trunk: Vec<usize>,
        local_structure: Option<LocalStructure>,
    ) -> Self {
        let (n, t) = (k.n, trunk.len());
        let mut owner = vec![usize::MAX; n];
        let mut local = vec![0usize; n];
        for (gi, g) in groups.iter().enumerate() {
            for (li, &id) in g.iter().enumerate() {
                owner[id] = gi;
                local[id] = li;
            }
        }
        for (i, &id) in trunk.iter().enumerate() {
            local[id] = i;
        }
        let bound_count = groups.len();
        // Local cone leaves: their sorted border columns, and whether a cone
        // row couples to the border (then the whole leaf is the suffix).
        let mut links = if matches!(local_structure, Some(LocalStructure::Cones)) {
            let mut links = vec![(Vec::new(), false); groups.len()];
            for j in 0..n {
                for q in k.colptr[j]..k.colptr[j + 1] {
                    let i = k.rowval[q];
                    let (inner, outer) = match (owner[i], owner[j]) {
                        (x, usize::MAX) if x != usize::MAX => (i, j),
                        (usize::MAX, y) if y != usize::MAX => (j, i),
                        _ => continue,
                    };
                    let link = &mut links[owner[inner]];
                    link.0.push(local[outer]);
                    link.1 |= signs[inner] < 0;
                }
            }
            for link in &mut links {
                link.0.sort_unstable();
                link.0.dedup();
            }
            links
        } else {
            Vec::new()
        };
        let leaves: Vec<Leaf<T>> = groups
            .into_iter()
            .enumerate()
            .map(|(gi, g)| {
                // Local constraints may have a zero prefix in their border coupling.
                let mut leaf = Leaf::new(g, if local_structure.is_some() { 0 } else { t });
                if let Some(kind) = local_structure {
                    let width = match kind {
                        LocalStructure::Soc => 2,
                        LocalStructure::SharedSoc => leaf.ids.len(),
                        LocalStructure::Bounds => 1,
                        LocalStructure::Cones if links[gi].1 => leaf.ids.len(),
                        LocalStructure::Cones => {
                            leaf.ids.iter().filter(|&&i| signs[i] > 0).count()
                        }
                    };
                    leaf.coupling_start = leaf.ids.len() - width;
                    let panels = matches!(kind, LocalStructure::Bounds)
                        && ((T::precision_bits() > 64)
                            || (cfg!(feature = "faer-sparse")
                                && std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>()));
                    if !panels {
                        leaf.signs = leaf.ids.iter().map(|&i| signs[i]).collect();
                        leaf.packed = true;
                        if matches!(kind, LocalStructure::Cones) {
                            // The suffix's first row stays raw in Y; B keeps the rest.
                            let columns = std::mem::take(&mut links[gi].0);
                            let q = if width == 0 { 0 } else { columns.len() };
                            leaf.b.resize(width.saturating_sub(1) * q, T::zero());
                            leaf.y.resize(width * q, T::zero());
                            leaf.z.resize(width * q, T::zero());
                            leaf.couples = vec![false; t];
                            for &c in &columns[..q] {
                                leaf.couples[c] = true;
                            }
                            leaf.coupled = columns;
                            leaf.coupled.truncate(q);
                        } else {
                            let b_size = if matches!(kind, LocalStructure::Soc) {
                                t
                            } else {
                                width * t
                            };
                            leaf.b.resize(b_size, T::zero());
                            leaf.y.resize(width * t, T::zero());
                            leaf.z.resize(width * t, T::zero());
                            leaf.coupled = (0..t).collect();
                            leaf.couples = vec![true; t];
                        }
                    }
                } else {
                    leaf.signs = vec![1];
                }
                leaf
            })
            .collect();
        // Local leaves with equal border columns form one coupling class.
        let mut classes: Vec<CouplingClass> = Vec::new();
        if local_structure.is_some() {
            let mut index: std::collections::HashMap<&[usize], usize> =
                std::collections::HashMap::new();
            for (l, leaf) in leaves.iter().enumerate() {
                if !leaf.packed || leaf.coupled.is_empty() {
                    continue;
                }
                let class = *index.entry(&leaf.coupled[..]).or_insert_with(|| {
                    let mut position = vec![usize::MAX; t];
                    for (p, &j) in leaf.coupled.iter().enumerate() {
                        position[j] = p;
                    }
                    classes.push(CouplingClass {
                        position,
                        leaves: Vec::new(),
                    });
                    classes.len() - 1
                });
                classes[class].leaves.push(l);
            }
        }
        let unshifted = if matches!(local_structure, Some(LocalStructure::SharedSoc)) {
            Self::single_primal_leaf_columns(k, signs, &leaves)
        } else {
            Vec::new()
        };
        let mut solver = Self {
            colptr: k.colptr.clone(),
            rowval: k.rowval.clone(),
            signs: signs.to_vec(),
            settings: settings.clone(),
            n,
            border_signs: trunk.iter().map(|&i| signs[i]).collect(),
            trunk,
            owner,
            local,
            leaves,
            c: vec![T::zero(); t * t],
            s: vec![T::zero(); t * t],
            tf: DenseLeaf::new(t),
            tx: vec![T::zero(); t],
            batch_tx: Vec::new(),
            pool: None,
            regularize_count: 0,
            fallback: None,
            use_arrow: false,
            unshifted,
            shift_exempt: false,
            local_structure,
            classes,
            exact_bound_panels: if matches!(local_structure, Some(LocalStructure::Bounds))
                && T::precision_bits() > 64
            {
                Some(local_bounds::ExactBoundPanels::new(bound_count, t))
            } else {
                None
            },
            #[cfg(feature = "faer-sparse")]
            bound_panels: if matches!(local_structure, Some(LocalStructure::Bounds))
                && std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>()
            {
                Some(local_bounds::BoundPanels::new(bound_count, t))
            } else {
                None
            },
            ranks: None,
        };
        #[cfg(feature = "faer-sparse")]
        if let Some(panels) = solver.bound_panels.as_mut() {
            let start = solver.leaves[0].ids[solver.leaves[0].coupling_start];
            panels.primal_start = solver
                .leaves
                .iter()
                .enumerate()
                .all(|(i, leaf)| leaf.ids[leaf.coupling_start] == start + i)
                .then_some(start);
            for j in 0..k.n {
                for q in k.colptr[j]..k.colptr[j + 1] {
                    let i = k.rowval[q];
                    if (solver.owner[i] == usize::MAX) == (solver.owner[j] == usize::MAX) {
                        panels.residual_entries.push((q, i, j));
                    }
                }
            }
        }
        if local_structure.is_none() {
            for leaf in &mut solver.leaves {
                leaf.couples.fill(false);
            }
            for j in 0..k.n {
                for q in k.colptr[j]..k.colptr[j + 1] {
                    let i = k.rowval[q];
                    match (solver.owner[i], solver.owner[j]) {
                        (x, usize::MAX) if x != usize::MAX => {
                            solver.leaves[x].couples[solver.local[j]] = true
                        }
                        (usize::MAX, y) if y != usize::MAX => {
                            solver.leaves[y].couples[solver.local[i]] = true
                        }
                        _ => {}
                    }
                }
            }
            for leaf in &mut solver.leaves {
                leaf.coupled = (0..t).filter(|&j| leaf.couples[j]).collect();
            }
        }
        solver.update_entries(0..k.nzval.len(), |q, value| *value = k.nzval[q]);
        solver
    }

    /// Route sparse-position updates directly to the owned numeric blocks.
    /// Sorted runs reuse their current column; arbitrary/repeated positions
    /// retain the update order and locate a new column only when needed.
    fn update_entries<I, F>(&mut self, positions: I, mut update: F)
    where
        I: IntoIterator<Item = usize>,
        F: FnMut(usize, &mut T),
    {
        let mut j = 0;
        let t = self.trunk.len();
        let mut bounds_y_dirty = false;
        for (q, position) in positions.into_iter().enumerate() {
            // Index maps are mostly increasing: step forward a few columns
            // before falling back to a binary search.
            if position < self.colptr[j] || position >= self.colptr[j + 1] {
                let mut steps = 0;
                while position >= self.colptr[j + 1] && steps < 4 && j + 2 < self.colptr.len() {
                    j += 1;
                    steps += 1;
                }
                if position < self.colptr[j] || position >= self.colptr[j + 1] {
                    j = self.colptr.partition_point(|&p| p <= position) - 1;
                }
            }
            let i = self.rowval[position];
            let (li, lj) = (self.local[i], self.local[j]);
            match (self.owner[i], self.owner[j]) {
                (x, y) if x == usize::MAX && y == usize::MAX => {
                    update(q, &mut self.c[li + lj * t]);
                    self.c[lj + li * t] = self.c[li + lj * t];
                }
                (x, y) if x != usize::MAX && y != usize::MAX => {
                    debug_assert_eq!(x, y);
                    let leaf = &mut self.leaves[x];
                    let g = leaf.ids.len();
                    update(q, &mut leaf.h[li + lj * g]);
                    leaf.h[lj + li * g] = leaf.h[li + lj * g];
                }
                (x, y) => {
                    let (id, row, col) = if x != usize::MAX {
                        (x, li, lj)
                    } else {
                        (y, lj, li)
                    };
                    if let Some(panels) = self.exact_bound_panels.as_mut() {
                        update(q, &mut panels.y[id + col * self.leaves.len()]);
                        bounds_y_dirty = true;
                        continue;
                    }
                    #[cfg(feature = "faer-sparse")]
                    if let Some(panels) = self.bound_panels.as_mut() {
                        let index = id + col * self.leaves.len();
                        let mut value = T::from_f64(panels.y[index]).unwrap();
                        update(q, &mut value);
                        panels.y[index] = value.to_f64().unwrap();
                        continue;
                    }
                    update(q, self.leaves[id].raw_coupling(row, col));
                }
            }
        }
        if bounds_y_dirty {
            if let Some(panels) = self.exact_bound_panels.as_mut() {
                panels.y_residues = ResidueCache::default();
            }
        }
    }

    /// The existing QDLDL fallback needs a CSC input. Reconstruct it only
    /// when Arrow fails, including the caller's current static shifts.
    fn materialize(&self) -> CscMatrix<T> {
        let t = self.trunk.len();
        let coupling = |id: usize, row: usize, col: usize| {
            if let Some(panels) = &self.exact_bound_panels {
                return panels.y[id + col * self.leaves.len()];
            }
            #[cfg(feature = "faer-sparse")]
            if let Some(panels) = &self.bound_panels {
                return T::from_f64(panels.y[id + col * self.leaves.len()]).unwrap();
            }
            let leaf = &self.leaves[id];
            if leaf.b.len() < leaf.y.len() {
                let width = leaf.ids.len() - leaf.coupling_start;
                let (r, c) = (row - leaf.coupling_start, leaf.column(col));
                if r == 0 {
                    leaf.y[c * width]
                } else {
                    leaf.b[r - 1 + c * (width - 1)]
                }
            } else {
                leaf.b[leaf.coupling_index(row, col)]
            }
        };
        let mut values = Vec::with_capacity(self.rowval.len());
        for j in 0..self.n {
            for q in self.colptr[j]..self.colptr[j + 1] {
                let i = self.rowval[q];
                let (li, lj) = (self.local[i], self.local[j]);
                values.push(match (self.owner[i], self.owner[j]) {
                    (x, y) if x == usize::MAX && y == usize::MAX => self.c[li + lj * t],
                    (x, y) if x != usize::MAX && y != usize::MAX => {
                        debug_assert_eq!(x, y);
                        let leaf = &self.leaves[x];
                        leaf.h[li + lj * leaf.ids.len()]
                    }
                    (x, y) => {
                        let (id, row, col) = if x != usize::MAX {
                            (x, li, lj)
                        } else {
                            (y, lj, li)
                        };
                        coupling(id, row, col)
                    }
                });
            }
        }
        CscMatrix::new(
            self.n,
            self.n,
            self.colptr.clone(),
            self.rowval.clone(),
            values,
        )
    }

    fn owned_leaves(&self) -> std::ops::Range<usize> {
        self.ranks
            .as_ref()
            .map_or(0..self.leaves.len(), LeafRanks::owned)
    }

    fn factor_arrow(&mut self) -> bool {
        if !self.c.is_finite()
            || self.leaves.iter().any(|leaf| {
                !leaf.h.is_finite()
                    || !leaf.b.is_finite()
                    || (leaf.b.len() < leaf.y.len()
                        && leaf
                            .y
                            .iter()
                            .step_by(leaf.ids.len() - leaf.coupling_start)
                            .any(|v| !v.is_finite()))
            })
        {
            return false;
        }
        let reg = self.settings.dynamic_regularization_enable.then_some((
            self.settings.dynamic_regularization_eps,
            self.settings.dynamic_regularization_delta,
        ));
        let t = self.trunk.len();
        let owned = self.owned_leaves();
        let timer = crate::receipt::start();
        let leaves_ok = if let Some(pool) = &self.pool {
            let threads = pool.current_num_threads();
            // Y columns are independent; rayon steals them from large leaves
            // even when leaves outnumber threads (per-column arithmetic is
            // unchanged).
            let split = threads > 1;
            // Split a leaf's factor when it alone outweighs an even share of
            // the rank's factor work (it would otherwise be the straggler), or
            // when threads amply outnumber leaves.
            let factor_work: u128 = self.leaves[owned.clone()]
                .iter()
                .map(|l| (l.ids.len() as u128).pow(3))
                .sum();
            let ample = threads >= SPLIT_FACTOR_THREADS_PER_LEAF * owned.len();
            let split_factor =
                |g: usize| ample || (g as u128).pow(3) * threads as u128 > factor_work;
            // Counts are diagnostic only; no per-leaf counter buffer is needed.
            let count = AtomicUsize::new(0);
            let ok = pool.install(|| {
                self.leaves[owned.clone()]
                    .par_iter_mut()
                    .try_for_each(|leaf| {
                        let mut n = 0;
                        let result =
                            leaf.refactor(t, reg, &mut n, split, split_factor(leaf.ids.len()));
                        if n != 0 {
                            count.fetch_add(n, Ordering::Relaxed);
                        }
                        result
                    })
            });
            self.regularize_count += count.into_inner();
            ok.is_ok()
        } else {
            let count = &mut self.regularize_count;
            self.leaves[owned.clone()]
                .iter_mut()
                .try_for_each(|leaf| leaf.refactor(t, reg, count, false, false))
                .is_ok()
        };
        crate::receipt::finish("arrow.leaves", timer);
        let leaves_ok = match &self.ranks {
            Some(ranks) => ranks.world.all_true(leaves_ok),
            None => leaves_ok,
        };
        if !leaves_ok {
            return false;
        }
        if self.ranks.as_ref().is_some_and(|r| !r.is_root()) {
            self.s.fill(T::zero());
        } else {
            self.s.copy_from_slice(&self.c);
        }
        if self.local_structure.is_some() {
            self.assemble_local_schur();
        } else {
            let timer = crate::receipt::start();
            subtract_contributions(&self.leaves[owned], &mut self.s, t, self.pool.as_deref());
            for j in 0..t {
                for i in 0..j {
                    self.s[j + i * t] = self.s[i + j * t];
                }
            }
            crate::receipt::finish("arrow.contribution", timer);
        }
        if let Some(ranks) = &self.ranks {
            let timer = crate::receipt::start();
            let mut upper: Vec<T> = (0..t)
                .flat_map(|j| (0..=j).map(move |i| (i, j)))
                .map(|(i, j)| self.s[i + j * t])
                .collect();
            let local = upper.clone();
            ranks.sum(&local, &mut upper);
            let mut p = 0;
            for j in 0..t {
                for i in 0..=j {
                    self.s[i + j * t] = upper[p];
                    self.s[j + i * t] = upper[p];
                    p += 1;
                }
            }
            crate::receipt::finish("arrow.border_sum", timer);
        }
        self.factor_border()
    }

    fn factor_border(&mut self) -> bool {
        let reg = self.settings.dynamic_regularization_enable.then_some((
            self.settings.dynamic_regularization_eps,
            self.settings.dynamic_regularization_delta,
        ));
        let mut border_count = 0usize;
        let timer = crate::receipt::start();
        // Every pool thread is idle during the border factor.
        let (tf, s) = (&mut self.tf, &self.s);
        let ok = match &self.pool {
            Some(pool) if tf.n >= SPLIT_BORDER_FACTOR_MIN => pool
                .install(|| tf.factor_signed(s, &self.border_signs, reg, &mut border_count, true)),
            _ => tf.factor_signed(s, &self.border_signs, reg, &mut border_count, false),
        }
        .is_ok();
        crate::receipt::finish("arrow.border_factor", timer);
        self.regularize_count += border_count;
        ok
    }

    fn factor_fallback(&mut self) -> bool {
        let matrix = self.materialize();
        if self.fallback.is_none() {
            let solver: BoxedDirectLDLSolver<T> = Box::new(
                super::qdldl::QDLDLDirectLDLSolver::new(&matrix, &self.signs, &self.settings, None),
            );
            self.fallback = Some(solver);
        }
        let solver = self.fallback.as_mut().unwrap();
        // The fallback's internal permuted copy must observe the same values
        // the caller has written into the owned blocks, including its
        // temporary regularization shifts.
        let indices: Vec<usize> = (0..self.rowval.len()).collect();
        solver.update_values(&indices, &matrix.nzval);
        solver.set_pool(self.pool.clone());
        solver.refactor(&matrix)
    }

    fn solve_arrow(&mut self, x: &mut [T], b: &[T]) {
        #[cfg(feature = "faer-sparse")]
        if self.solve_bounds_faer(x, b, 1) {
            return;
        }
        let t = self.trunk.len();
        let owned = self.owned_leaves();
        let timer = crate::receipt::start();
        if let Some(pool) = &self.pool {
            pool.install(|| {
                self.leaves[owned.clone()]
                    .par_iter_mut()
                    .for_each(|l| l.first_solve(b))
            });
        } else {
            for leaf in &mut self.leaves[owned.clone()] {
                leaf.first_solve(b);
            }
        }
        crate::receipt::finish("arrow.leaf_forward", timer);
        let timer = crate::receipt::start();
        // Local constraint coupling uses one exact MPFR accumulation per output,
        // avoiding thousands of rounded two-term dots. Each output retains
        // the same leaf order at every thread count.
        let leaves = &self.leaves;
        let nl = leaves.len();
        if let Some(panels) = self.exact_bound_panels.as_mut() {
            for (i, leaf) in leaves.iter().enumerate() {
                panels.vs[i] = leaf.v[0];
            }
        }
        let panels = self.exact_bound_panels.as_ref();
        let partial = self.ranks.as_ref().is_some_and(|r| !r.is_root());
        let couple = |(j, v): (usize, &mut T)| {
            *v = if partial { T::zero() } else { b[self.trunk[j]] };
            if let Some(panels) = panels {
                *v -= T::dot_fma(panels.y[j * nl..(j + 1) * nl].iter().zip(&panels.vs));
            } else if self.local_structure.is_some() {
                *v -= T::dot_fma(class_terms(&self.classes, leaves, j, |leaf, i| &leaf.v[i]));
            } else {
                for leaf in leaves[owned.clone()].iter().filter(|l| l.couples[j]) {
                    let g = leaf.ids.len();
                    *v -= T::dot_fma(
                        (0..g).map(|i| (&leaf.coupling_values()[i + j * g], &leaf.v[i])),
                    );
                }
            }
        };
        match &self.pool {
            Some(pool) if t > 1 => {
                pool.install(|| self.tx.par_iter_mut().enumerate().for_each(couple))
            }
            _ => self.tx.iter_mut().enumerate().for_each(couple),
        }
        if let Some(ranks) = &self.ranks {
            let local = self.tx.clone();
            ranks.sum(&local, &mut self.tx);
        }
        crate::receipt::finish("arrow.couple", timer);
        let timer = crate::receipt::start();
        self.tf.solve_pooled(&mut self.tx, self.pool.as_deref());
        crate::receipt::finish("arrow.trunk", timer);
        let timer = crate::receipt::start();
        let tx = &self.tx;
        let panels = self.exact_bound_panels.as_ref();
        let backward = |(i, leaf): (usize, &mut Leaf<T>)| match panels {
            Some(panels) => leaf.second_solve_bounds(tx, &panels.y, i, nl),
            None => leaf.second_solve(tx),
        };
        let first = owned.start;
        if let Some(pool) = &self.pool {
            pool.install(|| {
                self.leaves[owned.clone()]
                    .par_iter_mut()
                    .enumerate()
                    .for_each(|(i, leaf)| backward((first + i, leaf)))
            });
        } else {
            self.leaves[owned.clone()]
                .iter_mut()
                .enumerate()
                .for_each(|(i, leaf)| backward((first + i, leaf)));
        }
        if let Some(ranks) = &self.ranks {
            let local: Vec<T> = self.leaves[owned]
                .iter()
                .flat_map(|leaf| leaf.w.iter().copied())
                .collect();
            let mut all = vec![T::zero(); *ranks.offsets.last().unwrap()];
            ranks.gather_leaves(&local, 1, &mut all);
            for (leaf, &start) in self.leaves.iter().zip(&ranks.offsets) {
                for (i, &id) in leaf.ids.iter().enumerate() {
                    x[id] = all[start + i];
                }
            }
        } else {
            for leaf in &self.leaves {
                for (&id, &value) in leaf.ids.iter().zip(&leaf.w) {
                    x[id] = value;
                }
            }
        }
        for (&id, &value) in self.trunk.iter().zip(&self.tx) {
            x[id] = value;
        }
        crate::receipt::finish("arrow.leaf_backward", timer);
    }
}

impl<T: FloatT> HasLinearSolverInfo for ArrowLDLSolver<T> {
    fn linear_solver_info(&self) -> LinearSolverInfo {
        if !self.use_arrow {
            if let Some(solver) = &self.fallback {
                return solver.linear_solver_info();
            }
        }
        LinearSolverInfo {
            name: match self.local_structure {
                Some(LocalStructure::Soc) => "local_soc_arrow",
                Some(LocalStructure::Cones) => "local_cone_arrow",
                Some(LocalStructure::SharedSoc) => "shared_soc_arrow",
                #[cfg(feature = "faer-sparse")]
                Some(LocalStructure::Bounds) if self.bound_panels.is_some() => "local_bounds_faer",
                Some(LocalStructure::Bounds) => "local_bounds_arrow",
                None => "arrow",
            }
            .to_string(),
            threads: self.pool.as_ref().map_or(1, |p| p.current_num_threads()),
            direct: true,
            nnzA: self.rowval.len(),
            nnzL: self
                .leaves
                .iter()
                .map(|l| l.ids.len() * (l.ids.len() + 1) / 2)
                .sum::<usize>()
                + self.trunk.len() * (self.trunk.len() + 1) / 2,
        }
    }
}

impl<T: FloatT> DirectLDLSolver<T> for ArrowLDLSolver<T> {
    fn update_values(&mut self, index: &[usize], values: &[T]) {
        self.update_entries(index.iter().copied().take(values.len()), |q, value| {
            *value = values[q];
        });
    }

    fn scale_values(&mut self, index: &[usize], scale: T) {
        self.update_entries(index.iter().copied(), |_, value| *value *= scale);
    }

    fn residual(&self, kkt: &CscMatrix<T>, out: &mut [T], rhs: &[T], point: &[T]) -> Option<T> {
        #[cfg(feature = "faer-sparse")]
        if self.use_arrow {
            return self.residual_bounds_faer(kkt, out, rhs, point);
        }
        let _ = (kkt, out, rhs, point);
        None
    }

    fn refactor(&mut self, _kkt: &CscMatrix<T>) -> bool {
        self.use_arrow = self.factor_arrow();
        // A general fallback ordering needs every static shift: report the
        // failure so the caller escalates, which also ends the exemption.
        if !self.use_arrow && self.shift_exempt && !self.unshifted.is_empty() {
            return false;
        }
        self.use_arrow || self.factor_fallback()
    }

    fn unshifted_columns(&self) -> &[usize] {
        &self.unshifted
    }

    fn set_shift_exemption(&mut self, active: bool) {
        self.shift_exempt = active;
    }

    fn solve_many(&mut self, kkt: &CscMatrix<T>, x: &mut [T], b: &mut [T], cols: usize) {
        if !self.use_arrow {
            self.fallback.as_mut().unwrap().solve_many(kkt, x, b, cols);
            return;
        }
        let (n, t) = (self.n, self.trunk.len());
        assert_eq!(x.len(), n * cols);
        assert_eq!(b.len(), x.len());
        if cols == 0 {
            return;
        }
        #[cfg(feature = "faer-sparse")]
        if self.solve_bounds_faer(x, b, cols) {
            return;
        }
        self.batch_tx.resize(t * cols, T::zero());
        let owned = self.owned_leaves();
        // Spare threads (more than leaves) split each leaf's columns; keep
        // at least four columns per chunk.
        let chunks = self.pool.as_ref().map_or(1, |pool| {
            pool.current_num_threads()
                .div_ceil(owned.len().max(1))
                .min(cols / 4)
                .max(1)
        });
        if let Some(pool) = &self.pool {
            pool.install(|| {
                self.leaves[owned.clone()]
                    .par_iter_mut()
                    .for_each(|l| l.first_many(b, n, cols, chunks))
            });
        } else {
            self.leaves[owned.clone()]
                .iter_mut()
                .for_each(|l| l.first_many(b, n, cols, 1));
        }
        // Use the same coupling accumulation as the single-RHS solve.
        let tx = &mut self.batch_tx;
        let (leaves, trunk) = (&self.leaves, &self.trunk);
        let nl = leaves.len();
        if let Some(panels) = self.exact_bound_panels.as_mut() {
            panels.vs.resize(nl * cols, T::zero());
            for (i, leaf) in leaves.iter().enumerate() {
                for c in 0..cols {
                    panels.vs[i * cols + c] = leaf.batch_v[c];
                }
            }
        }
        let panels = self.exact_bound_panels.as_ref();
        let rhs: &[T] = b;
        let partial = self.ranks.as_ref().is_some_and(|r| !r.is_root());
        let couple = |(j, row): (usize, &mut [T])| {
            for (c, v) in row.iter_mut().enumerate() {
                *v = if partial {
                    T::zero()
                } else {
                    rhs[c * n + trunk[j]]
                };
            }
            if let Some(panels) = panels {
                for (c, v) in row.iter_mut().enumerate() {
                    *v -= T::dot_fma(
                        (0..nl).map(|i| (&panels.y[i + j * nl], &panels.vs[i * cols + c])),
                    );
                }
            } else if self.local_structure.is_some() {
                for (c, v) in row.iter_mut().enumerate() {
                    *v -= T::dot_fma(class_terms(&self.classes, leaves, j, |leaf, i| {
                        &leaf.batch_v[i * cols + c]
                    }));
                }
            } else {
                for leaf in leaves[owned.clone()].iter().filter(|l| l.couples[j]) {
                    let g = leaf.ids.len();
                    for (c, v) in row.iter_mut().enumerate() {
                        *v -= T::dot_fma((0..g).map(|i| {
                            (
                                &leaf.coupling_values()[i + j * g],
                                &leaf.batch_v[i * cols + c],
                            )
                        }));
                    }
                }
            }
        };
        match &self.pool {
            Some(pool) if t > 1 => {
                pool.install(|| tx.par_chunks_mut(cols).enumerate().for_each(couple))
            }
            _ => tx.chunks_mut(cols).enumerate().for_each(couple),
        }
        if let Some(ranks) = &self.ranks {
            let local = tx.clone();
            ranks.sum(&local, tx);
        }
        self.tf.forward_many(tx, cols);
        for j in 0..t {
            for c in 0..cols {
                tx[j * cols + c] *= self.tf.dinv[j];
            }
        }
        self.tf.backward_many(tx, cols);
        let backward = |(i, leaf): (usize, &mut Leaf<T>)| match panels {
            Some(panels) => leaf.second_many_bounds(tx, &panels.y, i, nl, cols, chunks),
            None => leaf.second_many(tx, cols, chunks),
        };
        let first = owned.start;
        if let Some(pool) = &self.pool {
            pool.install(|| {
                self.leaves[owned.clone()]
                    .par_iter_mut()
                    .enumerate()
                    .for_each(|(i, leaf)| backward((first + i, leaf)))
            });
        } else {
            self.leaves[owned.clone()]
                .iter_mut()
                .enumerate()
                .for_each(|(i, leaf)| backward((first + i, leaf)));
        }
        if let Some(ranks) = &self.ranks {
            let local: Vec<T> = self.leaves[owned]
                .iter()
                .flat_map(|leaf| leaf.batch_w.iter().copied())
                .collect();
            let mut all = vec![T::zero(); *ranks.offsets.last().unwrap() * cols];
            ranks.gather_leaves(&local, cols, &mut all);
            for (leaf, &start) in self.leaves.iter().zip(&ranks.offsets) {
                for (i, &id) in leaf.ids.iter().enumerate() {
                    for c in 0..cols {
                        x[c * n + id] = all[(start + i) * cols + c];
                    }
                }
            }
        } else {
            for leaf in &self.leaves {
                for (i, &id) in leaf.ids.iter().enumerate() {
                    for c in 0..cols {
                        x[c * n + id] = leaf.batch_w[i * cols + c];
                    }
                }
            }
        }
        for (i, &id) in self.trunk.iter().enumerate() {
            for c in 0..cols {
                x[c * n + id] = tx[i * cols + c];
            }
        }
    }

    fn solve(&mut self, kkt: &CscMatrix<T>, x: &mut [T], b: &mut [T]) {
        if !self.use_arrow {
            self.fallback.as_mut().unwrap().solve(kkt, x, b);
            return;
        }
        self.solve_arrow(x, b);
    }

    fn set_pool(&mut self, pool: Option<Arc<rayon::ThreadPool>>) {
        self.pool = pool;
    }

    fn set_world(&mut self, world: crate::MpiContext) {
        // Local structures keep their replicated assembly.
        self.ranks = world
            .world()
            .filter(|w| w.size() > 1 && self.local_structure.is_none())
            .map(|world| {
                let t = self.trunk.len();
                // Leaf work: contribution g·t²/2, Y = L⁻¹B g²·t/2, factor
                // g³/3. Measured on Λ27, the full border width balances ranks
                // better than shape-fitted models (exact-dot cost per term
                // also varies with the values).
                let tw = t as u64;
                let costs: Vec<u64> = self
                    .leaves
                    .iter()
                    .map(|l| {
                        let g = l.ids.len() as u64;
                        g.saturating_mul(3 * tw * tw + 3 * g * tw + 2 * g * g)
                            .max(1)
                    })
                    .collect();
                if world.rank() == 0 && crate::receipt::profile_requested() {
                    // Observation only: leaf sizes and coupled border widths.
                    let shape: Vec<(usize, usize)> = self
                        .leaves
                        .iter()
                        .map(|l| (l.ids.len(), l.coupled.len()))
                        .collect();
                    eprintln!("ARROW_LEAVES t={t} shape={shape:?}");
                }
                let mut offsets = vec![0];
                for leaf in &self.leaves {
                    offsets.push(offsets.last().unwrap() + leaf.ids.len());
                }
                LeafRanks {
                    world,
                    parts: crate::mpi::cost_ranges(&costs, world.size()),
                    offsets,
                }
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::{FromPrimitive, ToPrimitive, Zero};
    use sdpx_arithmetic::MpFloat;

    // The pooled forward sweep must reproduce the serial solve bitwise.
    fn pooled_solve_parity<T: FloatT>() {
        let n = 300;
        let mut leaf = DenseLeaf::<T>::new(n);
        for j in 0..n {
            for i in j + 1..n {
                leaf.l[i + j * n] =
                    T::from_usize((i * 7 + j * 13) % 17 + 1).unwrap() / T::from_usize(97).unwrap();
            }
            leaf.dinv[j] = T::from_usize(j % 5 + 2).unwrap().recip();
        }
        let rhs: Vec<T> = (0..n)
            .map(|i| T::from_usize(i % 11 + 1).unwrap() / T::from_usize(3).unwrap())
            .collect();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        let (mut serial, mut pooled) = (rhs.clone(), rhs);
        leaf.solve(&mut serial);
        leaf.solve_pooled(&mut pooled, Some(&pool));
        assert_eq!(serial, pooled);
    }
    #[test]
    fn pooled_solve_parity_f64() {
        pooled_solve_parity::<f64>();
    }
    #[test]
    fn pooled_solve_parity_mpfr256() {
        pooled_solve_parity::<sdpx_arithmetic::Bits256>();
    }

    fn batch_parity<T: FloatT>() {
        let (base, signs) = arrow_kkt();
        let k = CscMatrix::new(
            base.m,
            base.n,
            base.colptr,
            base.rowval,
            base.nzval
                .iter()
                .map(|&v| T::from_f64(v).unwrap())
                .collect(),
        );
        let settings = CoreSettings::<T>::default();
        let mut solver = ArrowLDLSolver::try_new(&k, &signs, &settings).unwrap();
        assert!(solver.refactor(&k));
        // 16 workers over 3 leaves split the refactor's border columns and
        // chunk the 8/16-column panels; every result must stay bitwise equal.
        let mut reference: Vec<Vec<T>> = Vec::new();
        for (round, workers) in [1, 4, 16, 1].into_iter().enumerate() {
            solver.set_pool((workers > 1).then(|| {
                Arc::new(
                    rayon::ThreadPoolBuilder::new()
                        .num_threads(workers)
                        .build()
                        .unwrap(),
                )
            }));
            assert!(solver.refactor(&k));
            for (case, cols) in [1, 3, 2, 8, 16].into_iter().enumerate() {
                let rhs: Vec<T> = (0..cols * k.n)
                    .map(|i| T::from_f64((i as f64 - 5.) / 8.).unwrap())
                    .collect();
                let mut out = vec![T::zero(); rhs.len()];
                solver.solve_many(&k, &mut out, &mut rhs.clone(), cols);
                for c in 0..cols {
                    let mut single = vec![T::zero(); k.n];
                    solver.solve(&k, &mut single, &mut rhs[c * k.n..(c + 1) * k.n].to_vec());
                    assert_eq!(single, out[c * k.n..(c + 1) * k.n]);
                }
                if round == 0 {
                    reference.push(out);
                } else {
                    assert_eq!(reference[case], out);
                }
            }
        }
        // Force the existing fallback, and verify the multi-column call follows it.
        let diag = k.colptr[0];
        solver.settings.dynamic_regularization_enable = false;
        solver.update_values(&[diag], &[T::zero()]);
        assert!(solver.refactor(&k));
        assert!(!solver.use_arrow);
        let rhs = vec![T::one(); k.n * 2];
        let mut out = vec![T::zero(); rhs.len()];
        solver.solve_many(&k, &mut out, &mut rhs.clone(), 2);
        let mut single = vec![T::zero(); k.n];
        solver.solve(&k, &mut single, &mut rhs[..k.n].to_vec());
        assert_eq!(single, out[..k.n]);
        assert_eq!(single, out[k.n..]);
    }
    fn split_factor_parity<T: FloatT>() {
        let n = 80;
        let mut a = vec![T::zero(); n * n];
        for j in 0..n {
            for i in j..n {
                let v = if i == j {
                    n as f64 + (i % 7) as f64
                } else {
                    ((i * 31 + j * 17) % 13) as f64 / 13.0 - 0.5
                };
                a[i + j * n] = T::from_f64(v).unwrap();
            }
        }
        let mut serial = DenseLeaf::<T>::new(n);
        let mut count = 0;
        serial.factor(&a, 1, None, &mut count, false).unwrap();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(8)
            .build()
            .unwrap();
        let mut split = DenseLeaf::<T>::new(n);
        pool.install(|| split.factor(&a, 1, None, &mut count, true))
            .unwrap();
        assert_eq!(serial.l, split.l);
        assert_eq!(serial.dinv, split.dinv);
    }
    #[test]
    fn arrow_single_dense_leaf() {
        // One dense 20-row leaf plus a two-row border: admitted and exact.
        let (g, t) = (20usize, 2usize);
        let n = g + t;
        let mut colptr = vec![0usize];
        let (mut rowval, mut nzval) = (Vec::new(), Vec::new());
        for j in 0..n {
            for i in 0..=j {
                let v = if j < g {
                    if i == j {
                        2.0 * g as f64
                    } else {
                        ((i * 7 + j * 3) % 5) as f64 / 5.0 - 0.4
                    }
                } else if i < g {
                    ((i + j) % 3) as f64 / 3.0
                } else if i == j {
                    -3.0
                } else {
                    0.1
                };
                rowval.push(i);
                nzval.push(v);
            }
            colptr.push(rowval.len());
        }
        let k = CscMatrix::new(n, n, colptr, rowval, nzval);
        let signs: Vec<i8> = (0..n).map(|i| if i < g { 1 } else { -1 }).collect();
        let settings = CoreSettings::<f64>::default();
        let mut solver = ArrowLDLSolver::try_new(&k, &signs, &settings).unwrap();
        assert!(solver.refactor(&k) && solver.use_arrow);
        let b: Vec<f64> = (0..n).map(|i| (i as f64 - 7.0) / 5.0).collect();
        let mut x = vec![0.0; n];
        solver.solve(&k, &mut x, &mut b.clone());
        let expect = reference_solve(&k, &b);
        for i in 0..n {
            assert!((x[i] - expect[i]).abs() < 1e-10, "{x:?} != {expect:?}");
        }
    }
    #[test]
    fn arrow_split_factor_f64() {
        split_factor_parity::<f64>();
    }
    #[test]
    fn arrow_split_factor_512() {
        split_factor_parity::<MpFloat<8>>();
    }
    #[test]
    fn arrow_batch_f64() {
        batch_parity::<f64>();
    }
    #[test]
    fn arrow_batch_256() {
        batch_parity::<sdpx_arithmetic::Bits256>();
    }
    #[test]
    fn arrow_batch_512() {
        batch_parity::<sdpx_arithmetic::Bits512>();
    }

    /// K = [H1 . . B1; . H2 . B2; . . H3 B3; B1' B2' B3' C] with three
    /// positive leaves and a negative border, stored as upper-triangular CSC.
    fn arrow_kkt() -> (CscMatrix<f64>, Vec<i8>) {
        arrow_kkt_merged(false)
    }

    fn arrow_kkt_merged(merge: bool) -> (CscMatrix<f64>, Vec<i8>) {
        // leaves: {0,1}, {2,3}, {4} ; border: {5,6}
        let n = 7;
        let mut dense = vec![vec![0f64; n]; n];
        let h = |i: usize, j: usize, v: f64, d: &mut Vec<Vec<f64>>| {
            d[i][j] = v;
            d[j][i] = v;
        };
        h(0, 0, 4.0, &mut dense);
        h(0, 1, 1.0, &mut dense);
        h(1, 1, 3.0, &mut dense);
        h(2, 2, 5.0, &mut dense);
        h(2, 3, -1.0, &mut dense);
        h(3, 3, 2.0, &mut dense);
        h(4, 4, 6.0, &mut dense);
        dense[5][5] = -2.0;
        dense[6][6] = -3.0;
        dense[0][5] = 0.5;
        dense[2][5] = -0.25;
        dense[4][6] = 0.75;
        dense[1][6] = 0.1;
        dense[5][6] = 0.2;
        if merge {
            dense[1][2] = 0.5; // positive-positive edges merge all leaves
            dense[3][4] = -0.4;
        }
        // Sparse CSC: only actual structural entries are stored.  Positive
        // components {0,1},{2,3},{4} connect only through border {5,6}.
        let mut colptr = vec![0usize];
        let mut rowval = Vec::new();
        let mut nzval = Vec::new();
        for j in 0..n {
            for i in 0..=j {
                if dense[i][j] != 0.0 || i == j {
                    rowval.push(i);
                    nzval.push(dense[i][j]);
                }
            }
            colptr.push(rowval.len());
        }
        (
            CscMatrix::new(n, n, colptr, rowval, nzval),
            vec![1, 1, 1, 1, 1, -1, -1],
        )
    }

    fn reference_solve(k: &CscMatrix<f64>, b: &[f64]) -> Vec<f64> {
        let mut x = vec![0f64; k.n];
        // dense Gaussian elimination for reference
        let mut a = vec![vec![0f64; k.n]; k.n];
        for j in 0..k.n {
            for q in k.colptr[j]..k.colptr[j + 1] {
                a[k.rowval[q]][j] = k.nzval[q];
                a[j][k.rowval[q]] = k.nzval[q];
            }
        }
        let mut rhs = b.to_vec();
        for c in 0..k.n {
            let p = (c..k.n)
                .max_by(|&i, &j| a[i][c].abs().total_cmp(&a[j][c].abs()))
                .unwrap();
            a.swap(c, p);
            rhs.swap(c, p);
            for r in c + 1..k.n {
                let f = a[r][c] / a[c][c];
                for k2 in c..k.n {
                    a[r][k2] -= f * a[c][k2];
                }
                rhs[r] -= f * rhs[c];
            }
        }
        for c in (0..k.n).rev() {
            let mut s = rhs[c];
            for k2 in c + 1..k.n {
                s -= a[c][k2] * x[k2];
            }
            x[c] = s / a[c][c];
        }
        x
    }

    #[test]
    fn arrow_solves_quasidefinite_and_falls_back() {
        let (k, signs) = arrow_kkt();
        let settings = CoreSettings::<f64>::default();
        let mut solver = ArrowLDLSolver::try_new(&k, &signs, &settings).unwrap();
        assert_eq!(solver.leaves.len(), 3);
        assert_eq!(solver.trunk, vec![5, 6]);
        assert!(solver.refactor(&k));
        assert!(solver.use_arrow);
        let b = vec![1.0, -2.0, 0.5, 3.0, -1.0, 2.0, 0.25];
        let mut rhs = b.clone();
        let mut x = vec![0.0; k.n];
        solver.solve(&k, &mut x, &mut rhs);
        let expect = reference_solve(&k, &b);
        for i in 0..k.n {
            assert!((x[i] - expect[i]).abs() < 1e-10, "{x:?} != {expect:?}");
        }
        // single connected positive component: not eligible
        let (k1, s1) = arrow_kkt_merged(true);
        assert!(ArrowLDLSolver::try_new(&k1, &s1, &settings).is_none());
        // QDLDL parity: a wrong-sign leaf pivot is regularized, not rejected.
        // Solver-side updates go through update_values into owned blocks.
        let (k2, s2) = arrow_kkt();
        let diag0 = k2.colptr[1] - 1; // (0,0) is the last entry of column 0
        let mut solver = ArrowLDLSolver::try_new(&k2, &s2, &settings).unwrap();
        solver.update_values(&[diag0], &[-100.0]);
        assert!(solver.refactor(&k2));
        assert!(solver.use_arrow);
        assert!(solver.regularize_count > 0);
        // Both backends must reject a zero singleton pivot when dynamic
        // regularization is disabled, then recover after restoring it.
        let (k3, s3) = arrow_kkt();
        let mut settings3 = CoreSettings::<f64>::default();
        settings3.dynamic_regularization_enable = false;
        let diag4 = k3.colptr[5] - 1; // (4,4)
        let mut solver = ArrowLDLSolver::try_new(&k3, &s3, &settings3).unwrap();
        solver.update_values(&[diag4], &[0.0]);
        assert!(!solver.refactor(&k3));
        assert!(!solver.use_arrow);
        assert!(solver.fallback.is_some());
        solver.update_values(&[diag4], &[k3.nzval[diag4]]);
        assert!(solver.refactor(&k3));
        assert!(solver.use_arrow);
        let mut rhs = b.clone();
        let mut x = vec![0.0; k3.n];
        solver.solve(&k3, &mut x, &mut rhs);
        let expect = reference_solve(&k3, &b);
        for i in 0..k3.n {
            assert!((x[i] - expect[i]).abs() < 1e-10, "{x:?} != {expect:?}");
        }
    }

    /// Same fixture under arbitrary precision: the production MPFR path must
    /// factor and solve identically.
    #[test]
    fn arrow_solves_quasidefinite_mpfr() {
        let (k64, signs) = arrow_kkt();
        let b64 = vec![1.0, -2.0, 0.5, 3.0, -1.0, 2.0, 0.25];
        let n = k64.n;
        let k = CscMatrix::<MpFloat<2>>::new(
            n,
            n,
            k64.colptr.clone(),
            k64.rowval.clone(),
            k64.nzval
                .iter()
                .map(|&v| MpFloat::<2>::from_f64(v).unwrap())
                .collect(),
        );
        let b: Vec<MpFloat<2>> = b64
            .iter()
            .map(|&v| MpFloat::<2>::from_f64(v).unwrap())
            .collect();
        let expect = reference_solve(&k64, &b64);
        let settings = CoreSettings::<MpFloat<2>>::default();
        let mut solver = ArrowLDLSolver::try_new(&k, &signs, &settings).unwrap();
        assert!(solver.refactor(&k));
        assert!(solver.use_arrow);
        let mut rhs = b.clone();
        let mut x = vec![MpFloat::<2>::zero(); n];
        solver.solve(&k, &mut x, &mut rhs);
        for i in 0..n {
            assert!(
                (x[i].to_f64().unwrap() - expect[i]).abs() < 1e-10,
                "{x:?} != {expect:?}"
            );
        }
    }

    /// A diagonal positive block (the LP shape) yields only singleton
    /// leaves: the dense trunk/coupling work is not amortized by any real
    /// leaf factorization, so the arrow representation must decline and
    /// leave the system to sparse QDLDL.
    #[test]
    fn arrow_rejects_singleton_leaves() {
        let n = 40;
        let t = 8;
        // Column j stores rows i <= j; positive->border edges sit in the
        // border columns.
        let mut colptr = vec![0usize];
        let mut rowval = Vec::new();
        let mut nzval = Vec::new();
        for j in 0..n + t {
            if j < n {
                rowval.push(j);
                nzval.push(2.0);
            } else {
                for i in (0..n).filter(|i| i % t == j - n) {
                    rowval.push(i);
                    nzval.push(0.25);
                }
                rowval.push(j);
                nzval.push(-1.0);
            }
            colptr.push(rowval.len());
        }
        let k = CscMatrix::new(n + t, n + t, colptr, rowval, nzval);
        let signs = vec![1i8; n]
            .into_iter()
            .chain(vec![-1i8; t])
            .collect::<Vec<_>>();
        assert!(ArrowLDLSolver::try_new(&k, &signs, &CoreSettings::<f64>::default()).is_none());
    }
    include!("arrow_storage_tests.rs");
}
