//! Eliminate scalar variables with one or two local bound rows.
//!
//! Negative bound pivots are eliminated first, leaving one positive scalar
//! pivot per variable: P_ii + shift + sum(a_bound^2 / (Hs_bound + shift)).
//! This is the ordinary signed LDL operation on a 2x2 or 3x3 leaf. Equality
//! multipliers and unbounded variables form the border. The parent KKT solver
//! still regularizes and refines against the complete original operator.
use super::*;
use crate::algebra::ResidueCache;
use crate::solver::cones::{CompositeCone, SupportedCone};

impl<T: FloatT> ArrowLDLSolver<T> {
    pub(crate) fn try_local_bounds(
        k: &CscMatrix<T>,
        signs: &[i8],
        a: &CscMatrix<T>,
        cones: &CompositeCone<T>,
        settings: &CoreSettings<T>,
        amd: &std::cell::OnceCell<crate::solver::kkt::ldl::AmdOrdering>,
    ) -> Option<Self> {
        // Binary64 requires the packed faer Schur kernel.
        if T::precision_bits() <= 53 && !cfg!(feature = "faer-sparse") {
            return None;
        }
        let n = a.n;
        let mut bound_rows = vec![false; a.m];
        let mut trunk = Vec::new();
        for (cone, rows) in cones.iter().zip(&cones.rng_cones) {
            match cone {
                SupportedCone::ZeroCone(_) => trunk.extend(rows.clone().map(|r| n + r)),
                SupportedCone::NonnegativeCone(_) => bound_rows[rows.clone()].fill(true),
                _ => return None,
            }
        }
        debug_assert!(k.n == n + a.m && k.m == k.n && signs.len() == k.n);
        // Every nonnegative row must be a bound on exactly one variable.
        let mut seen = vec![false; a.m];
        let mut groups = Vec::new();
        for col in 0..n {
            let mut ids = Vec::new();
            for p in a.colptr[col]..a.colptr[col + 1] {
                let row = a.rowval[p];
                if bound_rows[row] {
                    if seen[row] {
                        return None;
                    }
                    seen[row] = true;
                    ids.push(n + row);
                }
            }
            if ids.is_empty() {
                // Eliminate the negative equality border before these positive
                // pivots, avoiding division by a free variable's tiny shift.
                trunk.push(col);
            } else {
                if ids.len() > 2 {
                    return None;
                }
                ids.push(col);
                groups.push(ids);
            }
        }
        if bound_rows
            .iter()
            .zip(&seen)
            .any(|(&bound, &used)| bound && !used)
            || groups.len() < 64
            || trunk.is_empty()
        {
            return None;
        }
        let t = trunk.len() as u128;
        let packed = T::precision_bits() > 64
            || (cfg!(feature = "faer-sparse")
                && std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>());
        let cells = groups
            .iter()
            .map(|g| {
                let g = g.len() as u128;
                2 * g * g + 4 * g + if packed { 0 } else { 3 * t }
            })
            .sum::<u128>()
            + 6 * t * t
            + 8 * k.n as u128;
        let packed_bytes = if cfg!(feature = "faer-sparse")
            && std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>()
        {
            2 * groups.len() as u128 * t * 8
        } else if T::precision_bits() > 64 {
            2 * groups.len() as u128 * t * std::mem::size_of::<T>() as u128
        } else {
            0
        };
        // Tall couplings already occupy the input CSC. Bound the additional
        // dense storage, rather than rejecting a large, already dense LP.
        if cells * std::mem::size_of::<T>() as u128 + packed_bytes
            > ARROW_MAX_BYTES + k.nzval.len() as u128 * std::mem::size_of::<T>() as u128
        {
            return None;
        }
        let mut owner = vec![usize::MAX; k.n];
        for (group, ids) in groups.iter().enumerate() {
            for &id in ids {
                owner[id] = group;
            }
        }
        if signs
            .iter()
            .enumerate()
            .any(|(i, &s)| s != if i < n { 1 } else { -1 })
        {
            return None;
        }
        // Diagonal P only. Reject cross-leaf edges before installing the
        // persistent block map; stored zeros remain edges.
        for j in 0..k.n {
            for p in k.colptr[j]..k.colptr[j + 1] {
                let i = k.rowval[p];
                if (j < n && i != j)
                    || (owner[i] != usize::MAX && owner[j] != usize::MAX && owner[i] != owner[j])
                {
                    return None;
                }
                if owner[i] != owner[j]
                    && ((owner[i] != usize::MAX && i >= n) || (owner[j] != usize::MAX && j >= n))
                {
                    return None;
                }
            }
        }
        if T::precision_bits() <= 64 && !dense_border_pays(k, groups.len(), trunk.len(), amd) {
            return None;
        }
        Some(Self::from_groups(
            k,
            signs,
            settings,
            groups,
            trunk,
            Some(LocalStructure::Bounds),
        ))
    }
}

/// Dense BLAS-3 flops per sparse scalar LDL flop at equal time (binary64):
/// LP_bnl1 runs its dense border at 4.2 GF/s and QDLDL at about 0.4 GF/s.
const DENSE_FLOP_RATIO: f64 = 10.0;

/// The bound elimination leaves a dense `t x t` border: forming it costs
/// `groups * t^2` and factoring it `t^3 / 3` per refactor, against the sparse
/// LDL flops of the whole KKT under AMD. Keep the dense border only when it
/// is cheaper after crediting dense kernels with [`DENSE_FLOP_RATIO`].
/// Measured on the regression LPs, sparse LDL was 4-80x faster (LP_bnl1
/// 8.09 -> 0.098 s, LP_agg 2.55 -> 0.032 s).
/// The AMD ordering is kept in `amd` for the sparse factor that follows.
fn dense_border_pays<T: FloatT>(
    k: &CscMatrix<T>,
    groups: usize,
    t: usize,
    amd: &std::cell::OnceCell<crate::solver::kkt::ldl::AmdOrdering>,
) -> bool {
    let t = t as f64;
    let dense = groups as f64 * t * t + t * t * t / 3.0;
    let info = &amd.get_or_init(|| crate::solver::kkt::ldl::amd_order(k)).2;
    let sparse = (info.n_div + info.n_mult_subs_ldl) as f64;
    if crate::receipt::profile_requested() {
        eprintln!(
            "LOCAL_BOUNDS dense_flops={dense:.3e} sparse_flops={sparse:.3e} choice={}",
            if dense <= DENSE_FLOP_RATIO * sparse {
                "dense"
            } else {
                "sparse"
            }
        );
    }
    dense <= DENSE_FLOP_RATIO * sparse
}

#[cfg(feature = "faer-sparse")]
const BOUND_BLAS_COLS: usize = 16;

// Persistent column-major panels avoid gathering individual leaves for every
// Schur entry. Only actual binary64 problems enter this path; MPFR never casts.
#[cfg(feature = "faer-sparse")]
/// `out = op(Y)·x` for the column-major `n × t` panel `Y` and `cols`
/// right-hand sides (`op` = `N`: n × cols out, `T`: t × cols out), through
/// the linked BLAS like the bound Gram.
fn panel_product(
    op: u8,
    y: &[f64],
    n: usize,
    t: usize,
    x: &[f64],
    cols: usize,
    out: &mut [f64],
    pool: Option<&rayon::ThreadPool>,
) {
    let (rows, inner, tile) = if op == b'N' {
        (n, t, 128 * BOUND_BLAS_COLS)
    } else {
        (t, n, BOUND_BLAS_COLS)
    };
    if pool.is_none() || n * t * cols < 262144 {
        // SAFETY: y is n×t, x is inner×cols and out is rows×cols.
        unsafe {
            if cols == 1 {
                blas::dgemv(op, n as i32, t as i32, 1.0, y, n as i32, x, 1, 0.0, out, 1);
                return;
            }
            blas::dgemm(
                op,
                b'N',
                rows as i32,
                cols as i32,
                inner as i32,
                1.0,
                y,
                n as i32,
                x,
                inner as i32,
                0.0,
                out,
                rows as i32,
            );
        }
        return;
    }
    let product = |(j, dst): (usize, &mut [f64]), rhs: &[f64]| {
        let first = j * tile;
        let offset = if op == b'N' { first } else { first * n };
        // SAFETY: each tile owns its output rows. The panel retains its
        // original leading dimension; RHS columns are processed separately.
        unsafe {
            if cols == 1 {
                let (m, columns) = if op == b'N' {
                    (dst.len(), t)
                } else {
                    (n, dst.len())
                };
                blas::dgemv(
                    op,
                    m as i32,
                    columns as i32,
                    1.0,
                    &y[offset..],
                    n as i32,
                    rhs,
                    1,
                    0.0,
                    dst,
                    1,
                );
                return;
            }
            blas::dgemm(
                op,
                b'N',
                dst.len() as i32,
                1,
                inner as i32,
                1.0,
                &y[offset..],
                n as i32,
                rhs,
                inner as i32,
                0.0,
                dst,
                dst.len() as i32,
            );
        }
    };
    // Fixed tiles retain the same accumulation across parallel thread counts.
    pool.unwrap().install(|| {
        out.par_chunks_mut(rows)
            .zip(x.par_chunks(inner))
            .for_each(|(dst, rhs)| {
                dst.par_chunks_mut(tile)
                    .enumerate()
                    .for_each(|tile| product(tile, rhs));
            });
    });
}

#[cfg(feature = "faer-sparse")]
fn bound_gram(y: &[f64], z: &[f64], n: usize, t: usize, out: &mut [f64], parallel: bool) {
    let tile = BOUND_BLAS_COLS;
    let gram = |(j, dst): (usize, &mut [f64])| {
        let first = j * tile;
        let columns = dst.len() / t;
        let end = first + columns;
        let diagonal = |col: usize, dst: &mut [f64]| {
            let start = first + col;
            // Z contains individually rounded Y*d products. A symmetric
            // rank-k update would change that operator, so keep YᵀZ dots.
            unsafe {
                blas::dgemm(
                    b'T',
                    b'N',
                    (end - start) as i32,
                    1,
                    n as i32,
                    1.0,
                    &y[start * n..],
                    n as i32,
                    &z[start * n..],
                    n as i32,
                    0.0,
                    &mut dst[start..],
                    t as i32,
                );
            }
        };
        // Long inner products amortize square output tasks. Fixed column
        // panels otherwise leave the first task carrying the whole tall
        // rectangle (1496 of 5151 dots for a 101-column border).
        if parallel && n * tile * tile >= 262144 && end < t {
            let mut tails: Vec<_> = dst
                .chunks_mut(t)
                .enumerate()
                .map(|(col, column)| {
                    let (head, tail) = column.split_at_mut(end);
                    diagonal(col, head);
                    tail
                })
                .collect();
            rayon::scope(|scope| {
                for row in (end..t).step_by(tile) {
                    let height = tile.min(t - row);
                    let outputs: Vec<_> = tails
                        .iter_mut()
                        .map(|tail| {
                            let (output, rest) = std::mem::take(tail).split_at_mut(height);
                            *tail = rest;
                            output
                        })
                        .collect();
                    scope.spawn(move |_| {
                        let mut product = [0.0; BOUND_BLAS_COLS * BOUND_BLAS_COLS];
                        // Outputs own disjoint row slices of every column.
                        // The compact tile avoids a strided mutable alias.
                        unsafe {
                            blas::dgemm(
                                b'T',
                                b'N',
                                height as i32,
                                columns as i32,
                                n as i32,
                                1.0,
                                &y[row * n..],
                                n as i32,
                                &z[first * n..],
                                n as i32,
                                0.0,
                                &mut product,
                                height as i32,
                            );
                        }
                        for (output, column) in outputs.into_iter().zip(product.chunks(height)) {
                            output.copy_from_slice(column);
                        }
                    });
                }
            });
            return;
        }
        // SAFETY: each task owns complete output columns. The off-diagonal
        // rectangle starts below its diagonal tile, with leading dimension t.
        unsafe {
            if end < t {
                blas::dgemm(
                    b'T',
                    b'N',
                    (t - end) as i32,
                    columns as i32,
                    n as i32,
                    1.0,
                    &y[end * n..],
                    n as i32,
                    &z[first * n..],
                    n as i32,
                    0.0,
                    &mut dst[end..],
                    t as i32,
                );
            }
        }
        for (col, column) in dst.chunks_mut(t).enumerate() {
            diagonal(col, column);
        }
    };
    if parallel {
        out.par_chunks_mut(t * tile).enumerate().for_each(gram);
    } else {
        out.chunks_mut(t * tile).enumerate().for_each(gram);
    }
}

#[cfg(feature = "faer-sparse")]
pub(super) struct BoundPanels {
    pub(super) y: Vec<f64>,
    z: Vec<f64>,
    rhs: Vec<f64>,
    border: Vec<f64>,
    pub(super) residual_entries: Vec<(usize, usize, usize)>,
    pub(super) primal_start: Option<usize>,
    // Leaf factors copied into flat arrays after each refactor, so the batched
    // bound solve streams them instead of chasing per-leaf allocations.
    flat: FlatLeaves,
}

/// Per-leaf ids, strict-lower L entries and dinv, concatenated. The bound
/// coupling is the last row; leaf i spans off[i]..off[i+1] (loff for L).
#[cfg(feature = "faer-sparse")]
#[derive(Default)]
struct FlatLeaves {
    stale: bool,
    off: Vec<usize>,
    loff: Vec<usize>,
    ids: Vec<usize>,
    l: Vec<f64>,
    dinv: Vec<f64>,
}

#[cfg(feature = "faer-sparse")]
impl FlatLeaves {
    fn rebuild<T: FloatT>(&mut self, leaves: &[Leaf<T>]) {
        if self.off.is_empty() {
            self.off.push(0);
            self.loff.push(0);
            for leaf in leaves {
                let g = leaf.ids.len();
                self.ids.extend_from_slice(&leaf.ids);
                self.off.push(self.ids.len());
                self.loff.push(self.loff.last().unwrap() + g * (g - 1) / 2);
            }
        }
        self.l.clear();
        self.dinv.clear();
        for leaf in leaves {
            let g = leaf.ids.len();
            self.l.push(leaf.factor.l[1].to_f64().unwrap());
            if g == 3 {
                self.l.push(leaf.factor.l[2].to_f64().unwrap());
                self.l.push(leaf.factor.l[5].to_f64().unwrap());
            }
            self.dinv
                .extend(leaf.factor.dinv.iter().map(|v| v.to_f64().unwrap()));
        }
        self.stale = false;
    }
}

// try_local_bounds gives each leaf distinct ids; rhs slots i+c*n are also
// distinct across leaves. Only those disjoint slots use this helper;
// synchronous pool installs join before the borrowed slices are used again.
#[cfg(feature = "faer-sparse")]
struct BoundSlots<'a, T>(*mut T, std::marker::PhantomData<&'a mut [T]>);
#[cfg(feature = "faer-sparse")]
unsafe impl<T: Send> Sync for BoundSlots<'_, T> {}
#[cfg(feature = "faer-sparse")]
impl<'a, T: Copy> BoundSlots<'a, T> {
    fn new(values: &'a mut [T]) -> Self {
        Self(values.as_mut_ptr(), std::marker::PhantomData)
    }
    unsafe fn read(&self, i: usize) -> T {
        *self.0.add(i)
    }
    unsafe fn write(&self, i: usize, value: T) {
        *self.0.add(i) = value;
    }
}

#[cfg(feature = "faer-sparse")]
impl BoundPanels {
    pub(super) fn new(capacity: usize, border: usize) -> Self {
        Self {
            y: vec![0.0; capacity * border],
            z: Vec::with_capacity(capacity * border),
            flat: FlatLeaves::default(),
            rhs: Vec::new(),
            border: Vec::new(),
            residual_entries: Vec::new(),
            primal_start: None,
        }
    }
}

#[cfg(feature = "faer-sparse")]
impl<T: FloatT> ArrowLDLSolver<T> {
    pub(super) fn assemble_bound_schur_faer(&mut self) -> bool {
        let Some(panels) = self.bound_panels.as_mut() else {
            return false;
        };
        let n = self.leaves.len();
        let t = self.trunk.len();
        panels.flat.stale = true;
        panels.y.resize(n * t, 0.0);
        panels.z.resize(n * t, 0.0);
        // Z = Y D^-1: the multiplies are bitwise the same products the old
        // per-leaf reads formed, just computed column-contiguously.
        // Solve scratch is refilled before its next read.
        panels.rhs.clear();
        panels.rhs.extend(
            self.leaves
                .iter()
                .map(|leaf| leaf.factor.dinv[leaf.coupling_start].to_f64().unwrap()),
        );
        let d = &panels.rhs;
        let scale = |(z, y): (&mut [f64], &[f64])| {
            for ((z, y), d) in z.iter_mut().zip(y).zip(d) {
                *z = y * d;
            }
        };
        // Scale first, then form fixed column tiles on the existing pool.
        // Every lower-triangle entry is formed once; tile shapes are fixed.
        {
            // BoundPanels exists only for T=f64. Every lower Gram entry
            // overwrites S before the shifted C subtraction below.
            let gram = unsafe {
                std::slice::from_raw_parts_mut(self.s.as_mut_ptr().cast::<f64>(), self.s.len())
            };
            match &self.pool {
                Some(pool) if n * t * t >= 262144 => pool.install(|| {
                    panels
                        .z
                        .par_chunks_mut(n)
                        .zip(panels.y.par_chunks(n))
                        .for_each(scale);
                    bound_gram(&panels.y, &panels.z, n, t, gram, true);
                }),
                _ => {
                    panels
                        .z
                        .chunks_mut(n)
                        .zip(panels.y.chunks(n))
                        .for_each(scale);
                    bound_gram(&panels.y, &panels.z, n, t, gram, false);
                }
            }
        }
        for j in 0..t {
            for i in j..t {
                let value = self.c[i + j * t] - self.s[i + j * t];
                self.s[i + j * t] = value;
                self.s[j + i * t] = value;
            }
        }
        true
    }
}

#[cfg(feature = "faer-sparse")]
impl<T: FloatT> ArrowLDLSolver<T> {
    // Share single and batched RHS kernels, including refinement solves.
    pub(super) fn solve_bounds_faer(&mut self, x: &mut [T], b: &[T], cols: usize) -> bool {
        let Some(panels) = self.bound_panels.as_mut() else {
            return false;
        };
        let (n, t, dim) = (self.leaves.len(), self.trunk.len(), self.n);
        if panels.flat.stale || panels.flat.off.len() != n + 1 {
            panels.flat.rebuild(&self.leaves);
        }
        let f = &panels.flat;
        panels.rhs.resize(n * cols, 0.0);
        panels.border.resize(t * cols, 0.0);
        // Leaf output slots hold forward intermediates until the backward
        // solve; the intervening border solve writes only disjoint trunk ids.
        // Forward leaf solves, as `Leaf::first_many` and
        // `DenseLeaf::forward_many`: same fused updates in the same order.
        let grain = 1024;
        let ranges = n.div_ceil(grain);
        {
            let (x, rhs) = (BoundSlots::new(x), BoundSlots::new(&mut panels.rhs));
            let forward = |range: std::ops::Range<usize>| unsafe {
                for i in range {
                    let (o, g) = (f.off[i], f.off[i + 1] - f.off[i]);
                    let l = &f.l[f.loff[i]..f.loff[i + 1]];
                    debug_assert!(g == 2 || g == 3);
                    for c in 0..cols {
                        let w0 = b[c * dim + f.ids[o]].to_f64().unwrap();
                        let w1 = (-l[0]).mul_add(w0, b[c * dim + f.ids[o + 1]].to_f64().unwrap());
                        x.write(f.ids[o] + c * dim, T::from_f64(w0).unwrap());
                        x.write(f.ids[o + 1] + c * dim, T::from_f64(w1).unwrap());
                        let last = if g == 3 {
                            let w2 =
                                (-l[1]).mul_add(w0, b[c * dim + f.ids[o + 2]].to_f64().unwrap());
                            let w2 = (-l[2]).mul_add(w1, w2);
                            x.write(f.ids[o + 2] + c * dim, T::from_f64(w2).unwrap());
                            w2
                        } else {
                            w1
                        };
                        rhs.write(i + c * n, last * f.dinv[o + g - 1]);
                    }
                }
            };
            match &self.pool {
                Some(pool) if ranges > 1 => pool.install(|| {
                    (0..ranges).into_par_iter().for_each(|j| {
                        forward(j * grain..((j + 1) * grain).min(n));
                    });
                }),
                _ => forward(0..n),
            }
        }
        panel_product(
            b'T',
            &panels.y,
            n,
            t,
            &panels.rhs,
            cols,
            &mut panels.border,
            self.pool.as_deref(),
        );
        for c in 0..cols {
            for (j, &id) in self.trunk.iter().enumerate() {
                self.tx[j] = b[id + c * self.n] - T::from_f64(panels.border[j + c * t]).unwrap();
            }
            self.tf.solve(&mut self.tx);
            for (j, &id) in self.trunk.iter().enumerate() {
                x[id + c * self.n] = self.tx[j];
                panels.border[j + c * t] = self.tx[j].to_f64().unwrap();
            }
        }
        panel_product(
            b'N',
            &panels.y,
            n,
            t,
            &panels.border,
            cols,
            &mut panels.rhs,
            self.pool.as_deref(),
        );
        // Backward leaf solves, as `DenseLeaf::backward_many`; rows other than
        // the coupling row subtract zero, which leaves them unchanged.
        {
            let x = BoundSlots::new(x);
            let backward = |range: std::ops::Range<usize>| unsafe {
                for i in range {
                    let (o, g) = (f.off[i], f.off[i + 1] - f.off[i]);
                    let l = &f.l[f.loff[i]..f.loff[i + 1]];
                    for c in 0..cols {
                        let w0 = (x.read(f.ids[o] + c * dim).to_f64().unwrap() - 0.0) * f.dinv[o];
                        let value = panels.rhs[i + c * n];
                        let w1 = (x.read(f.ids[o + 1] + c * dim).to_f64().unwrap()
                            - if g == 2 { value } else { 0.0 })
                            * f.dinv[o + 1];
                        let (w0, w1) = if g == 3 {
                            let w2 = (x.read(f.ids[o + 2] + c * dim).to_f64().unwrap() - value)
                                * f.dinv[o + 2];
                            let w1 = (-l[2]).mul_add(w2, w1);
                            let w0 = (-l[0]).mul_add(w1, w0);
                            x.write(f.ids[o + 2] + c * dim, T::from_f64(w2).unwrap());
                            ((-l[1]).mul_add(w2, w0), w1)
                        } else {
                            ((-l[0]).mul_add(w1, w0), w1)
                        };
                        x.write(f.ids[o] + c * dim, T::from_f64(w0).unwrap());
                        x.write(f.ids[o + 1] + c * dim, T::from_f64(w1).unwrap());
                    }
                }
            };
            match &self.pool {
                Some(pool) if ranges > 1 => pool.install(|| {
                    (0..ranges).into_par_iter().for_each(|j| {
                        backward(j * grain..((j + 1) * grain).min(n));
                    });
                }),
                _ => backward(0..n),
            }
        }
        true
    }
}

#[cfg(feature = "faer-sparse")]
impl<T: FloatT> ArrowLDLSolver<T> {
    pub(super) fn residual_bounds_faer(
        &self,
        k: &CscMatrix<T>,
        out: &mut [T],
        rhs: &[T],
        point: &[T],
    ) -> Option<T> {
        let panels = self.bound_panels.as_ref()?;
        let (n, t) = (self.leaves.len(), self.trunk.len());
        // BoundPanels exists only for T=f64. A contiguous leaf range can read
        // the point directly and use its residual output as the forward product.
        let point64 =
            unsafe { std::slice::from_raw_parts(point.as_ptr().cast::<f64>(), point.len()) };
        let mut gathered = Vec::new();
        let primal = if let Some(start) = panels.primal_start {
            &point64[start..start + n]
        } else {
            gathered.extend(self.leaves.iter().map(|l| point64[l.ids[l.coupling_start]]));
            &gathered
        };
        let border: Vec<f64> = self.trunk.iter().map(|&i| point64[i]).collect();
        let mut product = vec![
            0.0;
            if panels.primal_start.is_some() {
                t
            } else {
                n + t
            }
        ];
        {
            let (leaf_product, border_product) = if let Some(start) = panels.primal_start {
                // BLAS beta=0 overwrites every entry without reading old out.
                let out64 = unsafe {
                    std::slice::from_raw_parts_mut(out.as_mut_ptr().cast::<f64>(), out.len())
                };
                (&mut out64[start..start + n], &mut product[..])
            } else {
                product.split_at_mut(n)
            };
            panel_product(
                b'N',
                &panels.y,
                n,
                t,
                &border,
                1,
                leaf_product,
                self.pool.as_deref(),
            );
            panel_product(
                b'T',
                &panels.y,
                n,
                t,
                primal,
                1,
                border_product,
                self.pool.as_deref(),
            );
        }
        let border_product = if let Some(start) = panels.primal_start {
            out[..start].copy_from_slice(&rhs[..start]);
            for i in start..start + n {
                out[i] = rhs[i] - out[i];
            }
            out[start + n..].copy_from_slice(&rhs[start + n..]);
            &product[..]
        } else {
            out.copy_from_slice(rhs);
            for (i, l) in self.leaves.iter().enumerate() {
                out[l.ids[l.coupling_start]] -= T::from_f64(product[i]).unwrap();
            }
            &product[n..]
        };
        for (i, &id) in self.trunk.iter().enumerate() {
            out[id] -= T::from_f64(border_product[i]).unwrap();
        }
        // Read current unshifted values from the parent's KKT. Factor storage
        // contains regularized diagonals and must never define this residual.
        for &(q, i, j) in &panels.residual_entries {
            out[i] = (-k.nzval[q]).mul_add(point[j], out[i]);
            if i != j {
                out[j] = (-k.nzval[q]).mul_add(point[i], out[j]);
            }
        }
        let norm = out.norm_inf();
        Some(if norm.is_finite() {
            norm
        } else {
            T::infinity()
        })
    }
}

pub(super) struct ExactBoundPanels<T> {
    pub(super) y: Vec<T>,
    z: Vec<T>,
    // Packed leaf `v`/`batch_v` rows at `coupling_start`, so the couple dot
    // reads one contiguous column of `y` against one contiguous `vs` column.
    pub(super) vs: Vec<T>,
    // Residues of the constant `y` for the exact `Yᵀ·diag(d)·Y` kernel.
    pub(super) y_residues: ResidueCache,
}
impl<T: FloatT> ExactBoundPanels<T> {
    pub(super) fn new(n: usize, t: usize) -> Self {
        Self {
            y: vec![T::zero(); n * t],
            z: Vec::new(),
            y_residues: ResidueCache::default(),
            vs: vec![T::zero(); n],
        }
    }
}
impl<T: FloatT> ArrowLDLSolver<T> {
    pub(super) fn assemble_bound_schur_exact(&mut self) -> bool {
        let Some(panels) = self.exact_bound_panels.as_mut() else {
            return false;
        };
        let (n, t) = (self.leaves.len(), self.trunk.len());
        // S holds the upper Gram until the shifted C subtraction below.
        // Y's residues remain cached across iterations. A declined kernel
        // reaches the rounded-Z fallback, which overwrites every upper entry.
        // Solve scratch is refilled before its next read.
        for (d, leaf) in panels.vs[..n].iter_mut().zip(&self.leaves) {
            *d = leaf.factor.dinv[leaf.coupling_start];
        }
        if T::diag_congruence_upper_exact(
            t,
            n,
            &panels.y,
            &panels.vs[..n],
            &mut self.s,
            self.pool.as_deref(),
            &mut panels.y_residues,
        ) {
            for j in 0..t {
                for i in 0..=j {
                    let v = self.c[j + i * t] - self.s[i + j * t];
                    self.s[j + i * t] = v;
                    self.s[i + j * t] = v;
                }
            }
            return true;
        }
        // Z = Y diag(d): independent products, split by trunk column.
        panels.z.resize(n * t, T::zero());
        let leaves = &self.leaves;
        let fill = |(j, column): (usize, &mut [T])| {
            for (i, (z, leaf)) in column.iter_mut().zip(leaves).enumerate() {
                *z = panels.y[i + j * n] * leaf.factor.dinv[leaf.coupling_start];
            }
        };
        match &self.pool {
            Some(pool) => pool.install(|| {
                panels.z[..n * t]
                    .par_chunks_mut(n)
                    .enumerate()
                    .for_each(fill)
            }),
            None => panels.z[..n * t].chunks_mut(n).enumerate().for_each(fill),
        }
        // Upper Z^T Y is the transpose of the old lower Y^T Z. This keeps
        // exactly the same chosen entries even though rounding Z can make
        // the unmirrored product differ in the last bit across the diagonal.
        // The exact residue kernel was measured here and reverted: it
        // assembled faster but raised peak RSS several-fold (see journal).
        {
            let (y, z) = (&panels.y, &panels.z);
            let entry =
                |i: usize, j: usize| T::dot_slices(&z[i * n..(i + 1) * n], &y[j * n..(j + 1) * n]);
            // Pair column j with column t-1-j so every task owns about t+1
            // upper entries; per-column parallelism is triangular-imbalanced.
            // Each entry's dot terms and order are unchanged.
            let pairs = t.div_ceil(2);
            let pair = move |k: usize| {
                let (a, b) = (k, t - 1 - k);
                let mut cols = Vec::with_capacity(2);
                cols.push((a, (0..=a).map(|i| entry(i, a)).collect::<Vec<T>>()));
                if b != a {
                    cols.push((b, (0..=b).map(|i| entry(i, b)).collect()));
                }
                cols
            };
            let mut store = |cols: Vec<(usize, Vec<T>)>| {
                for (j, col) in cols {
                    for (i, v) in col.into_iter().enumerate() {
                        self.s[i + j * t] = v;
                    }
                }
            };
            if let Some(pool) = &self.pool {
                let all = pool.install(|| {
                    (0..pairs)
                        .into_par_iter()
                        .map(pair)
                        .collect::<Vec<Vec<(usize, Vec<T>)>>>()
                });
                for cols in all {
                    store(cols);
                }
            } else {
                for k in 0..pairs {
                    store(pair(k));
                }
            }
        }
        for j in 0..t {
            for i in 0..=j {
                let v = self.c[j + i * t] - self.s[i + j * t];
                self.s[j + i * t] = v;
                self.s[i + j * t] = v;
            }
        }
        true
    }
}
