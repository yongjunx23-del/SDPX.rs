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
use crate::solver::kkt::direct::{BoxedDirectLDLSolver, DirectLDLSolver, DirectLDLSolverReqs};
use crate::solver::kkt::{HasLinearSolverInfo, LinearSolverInfo};
use rayon::prelude::*;
use std::sync::Arc;

mod local_bounds;
mod local_soc;

#[derive(Clone, Copy)]
enum LocalStructure {
    Soc,
    Bounds,
}

/// Hard cap on the dense working set of the arrow representation.  Larger
/// systems keep using QDLDL rather than densifying without bound.
const ARROW_MAX_BYTES: u128 = 512 * 1024 * 1024;
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
            let mut d = self.l[k + k * n];
            if !d.is_finite() {
                return Err("nonfinite_pivot");
            }
            // Same dynamic regularization as QDLDL: a pivot violating its
            // expected sign past eps is clamped to delta*sign and counted.
            if let Some((eps, delta)) = reg {
                if d * s < eps {
                    d = delta * s;
                    *regularize_count += 1;
                }
            }
            if d.is_zero() {
                return Err("zero_pivot");
            }
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
}

struct Leaf<T> {
    ids: Vec<usize>,
    signs: Vec<i8>,
    // SOC dual coordinates precede the two local primal coordinates. Their
    // coupling to the equality border is structurally zero after forward solve.
    coupling_start: usize,
    h: Vec<T>,
    b: Vec<T>,
    factor: DenseLeaf<T>,
    y: Vec<T>,
    z: Vec<T>,
    contribution: Vec<T>,
    w: Vec<T>,
    v: Vec<T>,
    batch_w: Vec<T>,
    batch_v: Vec<T>,
}

impl<T: FloatT> Leaf<T> {
    fn new(ids: Vec<usize>, t: usize) -> Self {
        let g = ids.len();
        Self {
            ids,
            signs: vec![1],
            coupling_start: 0,
            h: vec![T::zero(); g * g],
            b: vec![T::zero(); g * t],
            factor: DenseLeaf::new(g),
            y: vec![T::zero(); g * t],
            z: vec![T::zero(); g * t],
            contribution: vec![T::zero(); t * t],
            w: vec![T::zero(); g],
            v: vec![T::zero(); g],
            batch_w: Vec::new(),
            batch_v: Vec::new(),
        }
    }

    fn coupling_index(&self, row: usize, col: usize) -> usize {
        row - self.coupling_start + col * (self.ids.len() - self.coupling_start)
    }

    fn coupling_values(&self) -> &[T] {
        if self.y.is_empty() {
            &self.b
        } else {
            &self.y
        }
    }

    /// H = LDLᵀ, Y = L⁻¹B, contribution = YᵀD⁻¹Y accumulated into S = C - ΣYᵀZ.
    /// `split`: the pool has more threads than leaves, so the border
    /// columns of Y and of the contribution also run in parallel (each
    /// column's arithmetic is unchanged, so results are bitwise identical).
    fn refactor(
        &mut self,
        t: usize,
        reg: Option<(T, T)>,
        regularize_count: &mut usize,
        split: bool,
        split_factor: bool,
    ) -> Result<(), &'static str> {
        let g = self.ids.len();
        self.factor
            .factor_signed(&self.h, &self.signs, reg, regularize_count, split_factor)?;
        // A scalar primal suffix has L=1: Y=B. Packed bound panels own Z,
        // so these leaves need neither a duplicate Y nor a duplicate Z.
        if self.y.is_empty() {
            return Ok(());
        }
        let start = self.coupling_start;
        let width = g - start;
        self.y.copy_from_slice(&self.b);
        if split && width > 0 {
            let factor = &self.factor;
            self.y
                .par_chunks_mut(width)
                .for_each(|column| factor.forward_suffix(column, start));
        } else {
            for j in 0..t {
                self.factor
                    .forward_suffix(&mut self.y[j * width..(j + 1) * width], start);
            }
        }
        for j in 0..t {
            for i in 0..width {
                self.z[i + j * width] = self.y[i + j * width] * self.factor.dinv[i + start];
            }
        }
        if self.contribution.is_empty() {
            return Ok(());
        }
        let (y, z) = (&self.y, &self.z);
        let entry = |i: usize, j: usize| T::dot_fma((0..g).map(|k| (&y[k + i * g], &z[k + j * g])));
        if split {
            let columns: Vec<Vec<T>> = (0..t)
                .into_par_iter()
                .map(|j| (0..=j).map(|i| entry(i, j)).collect())
                .collect();
            for (j, column) in columns.into_iter().enumerate() {
                for (i, s) in column.into_iter().enumerate() {
                    self.contribution[i + j * t] = s;
                    self.contribution[j + i * t] = s;
                }
            }
        } else {
            for j in 0..t {
                for i in 0..=j {
                    let s = entry(i, j);
                    self.contribution[i + j * t] = s;
                    self.contribution[j + i * t] = s;
                }
            }
        }
        Ok(())
    }

    fn first_many(&mut self, rhs: &[T], n: usize, cols: usize, chunks: usize) {
        let g = self.ids.len();
        self.batch_w.resize(g * cols, T::zero());
        self.batch_v.resize(g * cols, T::zero());
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
        for i in 0..g {
            for c in 0..cols {
                self.batch_v[i * cols + c] = self.batch_w[i * cols + c] * self.factor.dinv[i];
            }
        }
    }
    fn second_many(&mut self, xt: &[T], cols: usize, chunks: usize) {
        let g = self.ids.len();
        for i in 0..g {
            for c in 0..cols {
                let s = if i < self.coupling_start {
                    T::zero()
                } else {
                    T::dot_fma((0..xt.len() / cols).map(|j| {
                        (
                            &self.coupling_values()[self.coupling_index(i, j)],
                            &xt[j * cols + c],
                        )
                    }))
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

    fn first_solve(&mut self, rhs: &[T]) {
        for (w, &id) in self.w.iter_mut().zip(&self.ids) {
            *w = rhs[id];
        }
        self.factor.forward(&mut self.w);
        for i in 0..self.ids.len() {
            self.v[i] = self.w[i] * self.factor.dinv[i];
        }
    }

    fn second_solve(&mut self, xt: &[T]) {
        let g = self.ids.len();
        for i in 0..g {
            let s = if i < self.coupling_start {
                T::zero()
            } else {
                T::dot_fma(
                    (0..xt.len())
                        .map(|j| (&self.coupling_values()[self.coupling_index(i, j)], &xt[j])),
                )
            };
            self.w[i] = (self.w[i] - s) * self.factor.dinv[i];
        }
        self.factor.backward(&mut self.w);
    }
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
    local_structure: Option<LocalStructure>,
    border_signs: Vec<i8>,
    exact_bound_panels: Option<local_bounds::ExactBoundPanels<T>>,
    #[cfg(feature = "faer-sparse")]
    bound_panels: Option<local_bounds::BoundPanels>,
}

impl<T: FloatT> ArrowLDLSolver<T> {
    /// Eligibility is structural only: connected components of the
    /// positive-sign subgraph become leaves, negative-sign variables become
    /// the border.  Returns `None` when the shape is not exploitable, so the
    /// caller transparently keeps QDLDL.
    pub fn try_new(k: &CscMatrix<T>, signs: &[i8], settings: &CoreSettings<T>) -> Option<Self> {
        let n = k.n;
        if k.m != n || signs.len() != n || signs.iter().any(|&s| s != 1 && s != -1) || n == 0 {
            return None;
        }
        let mut parent: Vec<usize> = (0..n).collect();
        // Stored positive-positive entries (upper triangle with diagonal).
        let mut positive_entries = 0usize;
        for j in 0..n {
            for q in k.colptr[j]..k.colptr[j + 1] {
                let i = k.rowval[q];
                // One owned slot per coordinate requires canonical CSC, as
                // does the public problem-data contract. Keep malformed or
                // duplicate coordinates on the existing general backend.
                if i > j || (q > k.colptr[j] && k.rowval[q - 1] >= i) {
                    return None;
                }
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
                4 * g * g + 4 * g * t as u128 + (t as u128).pow(2) + 8 * g
            })
            .sum::<u128>()
            + 6 * (t as u128).pow(2)
            + 8 * n as u128;
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
                leaves_ok
                    && work_ok
                    && cells * std::mem::size_of::<T>() as u128 <= ARROW_MAX_BYTES,
            );
        }
        if !leaves_ok {
            return None;
        }
        if !work_ok {
            return None;
        }
        if cells * std::mem::size_of::<T>() as u128 > ARROW_MAX_BYTES {
            return None;
        }
        Some(Self::from_groups(k, signs, settings, groups, trunk, None))
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
        let leaves = groups
            .into_iter()
            .map(|g| {
                // Local constraints couple through their primal coordinates only.
                let mut leaf = Leaf::new(g, if local_structure.is_some() { 0 } else { t });
                if let Some(kind) = local_structure {
                    let width = match kind {
                        LocalStructure::Soc => 2,
                        LocalStructure::Bounds => 1,
                    };
                    leaf.signs = leaf.ids.iter().map(|&i| signs[i]).collect();
                    leaf.coupling_start = leaf.ids.len() - width;
                    leaf.b.resize(width * t, T::zero());
                    let packed = matches!(kind, LocalStructure::Bounds)
                        && ((T::precision_bits() > 64)
                            || (cfg!(feature = "faer-sparse")
                                && std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>()));
                    if !packed {
                        leaf.y.resize(width * t, T::zero());
                        leaf.z.resize(width * t, T::zero());
                    }
                }
                leaf
            })
            .collect();
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
            local_structure,
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
        };
        #[cfg(feature = "faer-sparse")]
        if let Some(panels) = solver.bound_panels.as_mut() {
            for j in 0..k.n {
                for q in k.colptr[j]..k.colptr[j + 1] {
                    let i = k.rowval[q];
                    if (solver.owner[i] == usize::MAX) == (solver.owner[j] == usize::MAX) {
                        panels.residual_entries.push((q, i, j));
                    }
                }
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
        for (q, position) in positions.into_iter().enumerate() {
            if position < self.colptr[j] || position >= self.colptr[j + 1] {
                j = self.colptr.partition_point(|&p| p <= position) - 1;
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
                (x, _) if x != usize::MAX => {
                    let leaf = &mut self.leaves[x];
                    let index = leaf.coupling_index(li, lj);
                    update(q, &mut leaf.b[index]);
                }
                (_, y) => {
                    let leaf = &mut self.leaves[y];
                    let index = leaf.coupling_index(lj, li);
                    update(q, &mut leaf.b[index]);
                }
            }
        }
    }

    /// The existing QDLDL fallback needs a CSC input. Reconstruct it only
    /// when Arrow fails, including the caller's current static shifts.
    fn materialize(&self) -> CscMatrix<T> {
        let t = self.trunk.len();
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
                    (x, _) if x != usize::MAX => {
                        let leaf = &self.leaves[x];
                        leaf.b[leaf.coupling_index(li, lj)]
                    }
                    (_, y) => {
                        let leaf = &self.leaves[y];
                        leaf.b[leaf.coupling_index(lj, li)]
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

    fn factor_arrow(&mut self) -> bool {
        if !self.c.is_finite()
            || self
                .leaves
                .iter()
                .any(|leaf| !leaf.h.is_finite() || !leaf.b.is_finite())
        {
            return false;
        }
        let reg = self.settings.dynamic_regularization_enable.then_some((
            self.settings.dynamic_regularization_eps,
            self.settings.dynamic_regularization_delta,
        ));
        let t = self.trunk.len();
        let leaves_ok = if let Some(pool) = &self.pool {
            let threads = pool.current_num_threads();
            let split = threads > self.leaves.len();
            let split_factor = threads >= SPLIT_FACTOR_THREADS_PER_LEAF * self.leaves.len();
            // Regularization counts are diagnostic only; accumulate per-leaf
            // counts after the parallel section to stay deterministic.
            let mut counts = vec![0usize; self.leaves.len()];
            let ok = pool.install(|| {
                self.leaves
                    .par_iter_mut()
                    .zip(counts.par_iter_mut())
                    .map(|(leaf, c)| leaf.refactor(t, reg, c, split, split_factor))
                    .collect::<Result<Vec<_>, _>>()
            });
            self.regularize_count += counts.iter().sum::<usize>();
            ok.is_ok()
        } else {
            let count = &mut self.regularize_count;
            self.leaves
                .iter_mut()
                .try_for_each(|leaf| leaf.refactor(t, reg, count, false, false))
                .is_ok()
        };
        if !leaves_ok {
            return false;
        }
        self.s.copy_from_slice(&self.c);
        if self.local_structure.is_some() {
            self.assemble_local_schur();
        } else {
            // Deterministic merge order regardless of leaf scheduling.
            for leaf in &self.leaves {
                for (s, a) in self.s.iter_mut().zip(&leaf.contribution) {
                    *s -= *a;
                }
            }
        }
        let mut border_count = 0usize;
        // Every pool thread is idle during the border factor.
        let (tf, s) = (&mut self.tf, &self.s);
        let ok = match &self.pool {
            Some(pool) if tf.n >= SPLIT_BORDER_FACTOR_MIN => pool
                .install(|| tf.factor_signed(s, &self.border_signs, reg, &mut border_count, true)),
            _ => tf.factor_signed(s, &self.border_signs, reg, &mut border_count, false),
        }
        .is_ok();
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
        let timer = crate::receipt::start();
        if let Some(pool) = &self.pool {
            pool.install(|| self.leaves.par_iter_mut().for_each(|l| l.first_solve(b)));
        } else {
            for leaf in &mut self.leaves {
                leaf.first_solve(b);
            }
        }
        crate::receipt::finish("arrow.leaf_forward", timer);
        let timer = crate::receipt::start();
        // Local constraint coupling uses one exact MPFR accumulation per output,
        // avoiding thousands of rounded two-term dots. Each output retains
        // the same leaf order at every thread count.
        let leaves = &self.leaves;
        let couple = |(j, v): (usize, &mut T)| {
            *v = b[self.trunk[j]];
            if self.local_structure.is_some() {
                *v -= T::dot_fma(leaves.iter().flat_map(|leaf| {
                    let width = leaf.ids.len() - leaf.coupling_start;
                    (0..width).map(move |i| {
                        (
                            &leaf.coupling_values()[i + j * width],
                            &leaf.v[i + leaf.coupling_start],
                        )
                    })
                }));
            } else {
                for leaf in leaves {
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
        crate::receipt::finish("arrow.couple", timer);
        let timer = crate::receipt::start();
        self.tf.solve(&mut self.tx);
        crate::receipt::finish("arrow.trunk", timer);
        let timer = crate::receipt::start();
        let tx = &self.tx;
        if let Some(pool) = &self.pool {
            pool.install(|| self.leaves.par_iter_mut().for_each(|l| l.second_solve(tx)));
        } else {
            for leaf in &mut self.leaves {
                leaf.second_solve(tx);
            }
        }
        for leaf in &self.leaves {
            for (&id, &value) in leaf.ids.iter().zip(&leaf.w) {
                x[id] = value;
            }
        }
        for (&id, &value) in self.trunk.iter().zip(&self.tx) {
            x[id] = value;
        }
        crate::receipt::finish("arrow.leaf_backward", timer);
    }
}

impl<T: FloatT> DirectLDLSolverReqs for ArrowLDLSolver<T> {
    fn required_matrix_shape() -> MatrixTriangle {
        MatrixTriangle::Triu
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

    fn offset_values(&mut self, index: &[usize], offset: T, signs: &[i8]) {
        self.update_entries(index.iter().copied().take(signs.len()), |q, value| {
            *value += offset * T::from_i8(signs[q]).unwrap();
        });
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
        self.use_arrow || self.factor_fallback()
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
        // Spare threads (more than leaves) split each leaf's columns; keep
        // at least four columns per chunk.
        let chunks = self.pool.as_ref().map_or(1, |pool| {
            pool.current_num_threads()
                .div_ceil(self.leaves.len().max(1))
                .min(cols / 4)
                .max(1)
        });
        if let Some(pool) = &self.pool {
            pool.install(|| {
                self.leaves
                    .par_iter_mut()
                    .for_each(|l| l.first_many(b, n, cols, chunks))
            });
        } else {
            self.leaves
                .iter_mut()
                .for_each(|l| l.first_many(b, n, cols, 1));
        }
        // Use the same coupling accumulation as the single-RHS solve.
        let tx = &mut self.batch_tx;
        let (leaves, trunk) = (&self.leaves, &self.trunk);
        let rhs: &[T] = b;
        let couple = |(j, row): (usize, &mut [T])| {
            for (c, v) in row.iter_mut().enumerate() {
                *v = rhs[c * n + trunk[j]];
            }
            if self.local_structure.is_some() {
                for (c, v) in row.iter_mut().enumerate() {
                    *v -= T::dot_fma(leaves.iter().flat_map(|leaf| {
                        let width = leaf.ids.len() - leaf.coupling_start;
                        (0..width).map(move |i| {
                            (
                                &leaf.coupling_values()[i + j * width],
                                &leaf.batch_v[(i + leaf.coupling_start) * cols + c],
                            )
                        })
                    }));
                }
            } else {
                for leaf in leaves {
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
        self.tf.forward_many(tx, cols);
        for j in 0..t {
            for c in 0..cols {
                tx[j * cols + c] *= self.tf.dinv[j];
            }
        }
        self.tf.backward_many(tx, cols);
        if let Some(pool) = &self.pool {
            pool.install(|| {
                self.leaves
                    .par_iter_mut()
                    .for_each(|l| l.second_many(tx, cols, chunks))
            });
        } else {
            self.leaves
                .iter_mut()
                .for_each(|l| l.second_many(tx, cols, 1));
        }
        for leaf in &self.leaves {
            for (i, &id) in leaf.ids.iter().enumerate() {
                for c in 0..cols {
                    x[c * n + id] = leaf.batch_w[i * cols + c];
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::{FromPrimitive, ToPrimitive, Zero};
    use sdpx_arithmetic::MpFloat;

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
