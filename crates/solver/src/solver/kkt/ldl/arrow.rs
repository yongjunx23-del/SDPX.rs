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
    atomic::{AtomicBool, AtomicUsize, Ordering},
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
/// Pivots per panel of the split leaf factor (one join per panel).
const FACTOR_PANEL: usize = 8;
/// Wide-type leaves from this order factor recursively
/// ([`DenseLeaf::factor_rec`]); blocks up to `REC_BASE` take exact dots.
/// Dense leaves, faer residue products (Mac): from 256 the recursion is
/// 15–28% faster than exact dots serially and level at 4 threads; at 1806
/// rows (128 bits) 4.2 → 1.5 s serial, 1.2 → 0.6 s at 4 threads. It also
/// beat the earlier blocked exact form (48-pivot panels) at every order and
/// precision tried (256–512 bits, up to 2.6× at 4 threads).
const REC_FACTOR_MIN: usize = 256;
const REC_BASE: usize = 128;
/// Estimated workspace of one batch of exact leaf contributions.
const BATCH_WORK_BYTES: usize = 4 << 30;
/// Per-step factor tasks are small: split a leaf factor only with at least
/// this many threads per leaf (at ~2 per leaf it measured slower), and the
/// border factor only from this dimension.
const SPLIT_FACTOR_THREADS_PER_LEAF: usize = 4;
const SPLIT_BORDER_FACTOR_MIN: usize = 64;
/// Row block of the split trunk sweeps.
const FORWARD_BLOCK: usize = 32;
/// Leaf sweeps split with smaller blocks: Λ27 leaves of order 88-113 on
/// ranks with two leaves and 16 workers ran their sweeps serially.
const LEAF_BLOCK: usize = 16;
// Backward sweeps use a per-factor row block (`DenseLeaf::back`): row `i`
// applies its terms `k > i` block by block from the bottom (`[n-B, n)`,
// `[n-2B, n-B)`, ...), ascending within a block. That is the only order a
// pooled sweep can follow, so a factor's serial, batched and pooled sweeps
// give the same bits. Leaves use `LEAF_BLOCK`; the trunk keeps
// `FORWARD_BLOCK`, halving its pooled phases (Λ27 at 64 threads: 16-row
// trunk blocks cost +1.5% per iteration).
const LEAF_SPLIT_MIN: usize = 4 * LEAF_BLOCK;

/// The split thresholds above were measured on MPFR leaves. A binary64 FMA
/// costs about two orders less while a pooled join still costs tens of
/// microseconds, so binary64 splits sweeps (and row passes) from `scale`
/// times the order: 16 for sweeps, 8 for factor panels. theta1's single
/// 104-row leaf swept 6x and factored 2x slower split at 8 threads.
fn f64_scaled<T: FloatT>(order: usize, scale: usize) -> usize {
    if T::precision_bits() <= 64 {
        order * scale
    } else {
        order
    }
}

fn find(parent: &mut [usize], mut i: usize) -> usize {
    while parent[i] != i {
        parent[i] = parent[parent[i]];
        i = parent[i];
    }
    i
}

/// Wide [`DenseLeaf`] LDLᵀ of the order-`n` block at `l` (leading dimension
/// `ld`) in place, left-looking: entry (i, j) is one exact dot product of
/// rows of L and W = L·D (rounded entrywise), rounded once, instead of j
/// fused updates. Rows are packed row-major so both operands are contiguous,
/// and dots start at the row envelope (`first`, local to the block; the
/// skipped products are exact zeros, so values are unchanged). `split`:
/// after each panel of `FACTOR_PANEL` pivots, the later rows take the
/// panel's columns in parallel, one join per panel; every entry is the same
/// dot, so the factor is bitwise the serial one. On success the lower
/// triangle holds the unit lower L and `dinv` its D⁻¹; the strict upper
/// triangle is untouched.
#[allow(clippy::too_many_arguments)]
fn ldl_dots<T: FloatT>(
    l: &mut [T],
    ld: usize,
    n: usize,
    first: &[usize],
    sign: &(dyn Fn(usize) -> T + Sync),
    reg: Option<(T, T)>,
    regularize_count: &mut usize,
    split: bool,
    dinv: &mut [T],
) -> Result<(), &'static str> {
    use sdpx_arithmetic::inner_parallel as ip;
    let row = |i: usize| i * (i + 1) / 2;
    let mut r = vec![T::zero(); row(n)];
    let mut w = vec![T::zero(); row(n)];
    let mut d = vec![T::zero(); n];
    let a = &*l;
    // entry (i, j) from row i's first j entries and row j of W
    let entry = |i: usize, j: usize, ri: &[T], wj: &[T], dinv: T| {
        if first[i] > j {
            return T::zero();
        }
        let lo = first[i].max(first[j]);
        (a[i + j * ld] - T::dot_slices(&ri[lo..j], &wj[lo..j])) * dinv
    };
    let mut p = 0;
    while p < n {
        let pe = if split { (p + FACTOR_PANEL).min(n) } else { n };
        for j in p..pe {
            let fj = first[j];
            let (rj, wj) = (&r[row(j) + fj..row(j) + j], &w[row(j) + fj..row(j) + j]);
            let dj = a[j + j * ld] - T::dot_slices(rj, wj);
            let dj = DenseLeaf::<T>::pivot(dj, sign(j), reg, regularize_count)?;
            d[j] = dj;
            dinv[j] = T::one() / dj;
            let (wh, wt) = w.split_at_mut(row(j + 1));
            let wj = &wh[row(j)..row(j) + j];
            for i in j + 1..pe {
                let v = entry(i, j, &r[row(i)..row(i) + i], wj, dinv[j]);
                r[row(i) + j] = v;
                wt[row(i) - row(j + 1) + j] = v * dj;
            }
        }
        if pe < n {
            let (_, mut rt) = r.split_at_mut(row(pe));
            let (wh, mut wt) = w.split_at_mut(row(pe));
            let mut rows = Vec::with_capacity(n - pe);
            for i in pe..n {
                let (ri, rrest) = rt.split_at_mut(i + 1);
                let (wi, wrest) = wt.split_at_mut(i + 1);
                rows.push((i, ri, wi));
                (rt, wt) = (rrest, wrest);
            }
            let (dinv, d) = (&*dinv, &d);
            let panel = |(i, ri, wi): &mut (usize, &mut [T], &mut [T])| {
                for j in p..pe {
                    let v = entry(*i, j, ri, &wh[row(j)..row(j) + j], dinv[j]);
                    ri[j] = v;
                    wi[j] = v * d[j];
                }
            };
            // rows per task: one grain of dot work
            let work = ip::weight(T::precision_bits() as usize) * ((pe - p) * pe) as u128;
            let tile = (ip::grain() / work.max(1)).max(1) as usize;
            rows.par_iter_mut().with_min_len(tile).for_each(panel);
        }
        p = pe;
    }
    for j in 0..n {
        l[j + j * ld] = T::one();
        for i in j + 1..n {
            l[i + j * ld] = r[row(i) + j];
        }
    }
    Ok(())
}

/// Recursive [`ldl_dots`] of the order-`n` block at `l` (leading dimension
/// `ld`, row envelope `first` local to the block). A cut of the envelope
/// (no row from `c` on reaches a column before `c`) leaves the coupling
/// block an exact zero, so the two sides factor independently (concurrently
/// when `ways > 1`); otherwise the block halves: W21 = A21·L11⁻ᵀ
/// ([`solve_unit_lower_t`]), L21 = W21·D1⁻¹, and A22 −= L21·W21ᵀ as one
/// exact product (each entry the exact sum rounded once, then subtracted).
/// Envelope zeros stay exact zeros, so `first` remains the factor's
/// envelope. Blocks up to `REC_BASE` take [`ldl_dots`]. `ways`: spare workers
/// granted to the products and base factors.
#[allow(clippy::too_many_arguments)]
fn ldl_rec<T: FloatT>(
    l: &mut [T],
    ld: usize,
    n: usize,
    first: &[usize],
    sign: &(dyn Fn(usize) -> T + Sync),
    reg: Option<(T, T)>,
    regularize_count: &mut usize,
    ways: usize,
    dinv: &mut [T],
) -> Result<(), &'static str> {
    if n <= REC_BASE {
        return ldl_dots(l, ld, n, first, sign, reg, regularize_count, ways > 1, dinv);
    }
    // the cut nearest the middle, if any
    let mut low = n;
    let mut cut = None;
    for c in (1..n).rev() {
        low = low.min(first[c]);
        if low >= c && cut.is_none_or(|k: usize| c.abs_diff(n / 2) < k.abs_diff(n / 2)) {
            cut = Some(c);
        }
    }
    if let Some(c) = cut {
        let (left, right) = l.split_at_mut(c * ld);
        let (d1, d2) = dinv.split_at_mut(c);
        let tail: Vec<usize> = first[c..].iter().map(|&f| f - c).collect();
        let sign2 = |k: usize| sign(c + k);
        let mut count2 = 0;
        let first_side = || {
            ldl_rec(
                left,
                ld,
                c,
                &first[..c],
                sign,
                reg,
                regularize_count,
                ways,
                d1,
            )
        };
        let second_side = || {
            ldl_rec(
                &mut right[c..],
                ld,
                n - c,
                &tail,
                &sign2,
                reg,
                &mut count2,
                ways,
                d2,
            )
        };
        // The larger side stays on this worker: a worker waiting for a
        // stolen larger side would take unrelated work meanwhile (BFSS
        // L11u: the 2m side of a 3m ORE leaf, after the m side).
        let (r1, r2) = if 2 * c >= n {
            join_if(ways > 1, first_side, second_side)
        } else {
            let (r2, r1) = join_if(ways > 1, second_side, first_side);
            (r1, r2)
        };
        *regularize_count += count2;
        return r1.and(r2);
    }
    let (n1, n2) = (n / 2, n - n / 2);
    let (left, right) = l.split_at_mut(n1 * ld);
    let (d1, d2) = dinv.split_at_mut(n1);
    ldl_rec(
        left,
        ld,
        n1,
        &first[..n1],
        sign,
        reg,
        regularize_count,
        ways,
        d1,
    )?;
    let l11: Vec<T> = (0..n1 * n1).map(|e| left[e % n1 + e / n1 * ld]).collect();
    let low: Vec<usize> = first[n1..].iter().map(|&f| f.min(n1)).collect();
    solve_unit_lower_t(n2, n1, &l11, n1, &mut left[n1..], ld, &low, ways);
    // W21 before scaling; L21 = W21·D1⁻¹ in place
    let mut w = vec![T::zero(); n2 * n1];
    for j in 0..n1 {
        for i in 0..n2 {
            let v = &mut left[n1 + i + j * ld];
            w[i + j * n2] = *v;
            *v *= d1[j];
        }
    }
    let l21 = &left[n1..];
    // packed (a, b), a ≤ b, is the Schur term (b, a)
    let mut packed = vec![T::zero(); n2 * (n2 + 1) / 2];
    let upper = crate::algebra::with_split_hint(ways, || {
        T::xgemm_upper_exact(
            b'N',
            b'T',
            n2,
            n2,
            n1,
            &w,
            n2,
            l21,
            ld,
            &mut packed,
            None,
            None,
        )
    });
    if upper {
        for a in 0..n2 {
            for b in a..n2 {
                right[n1 + b + a * ld] -= packed[b * (b + 1) / 2 + a];
            }
        }
    } else {
        let mut s = vec![T::zero(); n2 * n2];
        exact_product(
            (b'N', b'T'),
            (n2, n2, n1),
            (&w, n2),
            (l21, ld),
            &mut s,
            ways,
        );
        for a in 0..n2 {
            for b in a..n2 {
                right[n1 + b + a * ld] -= s[a + b * n2];
            }
        }
    }
    let tail: Vec<usize> = first[n1..].iter().map(|&f| f.saturating_sub(n1)).collect();
    let sign2 = |k: usize| sign(n1 + k);
    ldl_rec(
        &mut right[n1..],
        ld,
        n2,
        &tail,
        &sign2,
        reg,
        regularize_count,
        ways,
        d2,
    )
}

/// X·Lᵀ = B for X (`m × n`, overwriting B with leading dimension `ldb`), L
/// unit lower of order `n` (its diagonal is not read): recursive halves of L
/// with one exact product update between them; up to `REC_BASE` columns,
/// rows of B solve independently by exact dots from their first nonzero
/// column (`low`, at most `n`: the skipped entries stay exact zeros).
#[allow(clippy::too_many_arguments)]
fn solve_unit_lower_t<T: FloatT>(
    m: usize,
    n: usize,
    l: &[T],
    ldl: usize,
    b: &mut [T],
    ldb: usize,
    low: &[usize],
    ways: usize,
) {
    use sdpx_arithmetic::inner_parallel as ip;
    if n <= REC_BASE {
        let lr: Vec<T> = (0..n * n).map(|e| l[e / n + e % n * ldl]).collect();
        let mut rows: Vec<T> = (0..m * n).map(|e| b[e / n + e % n * ldb]).collect();
        let solve = |(i, x): (usize, &mut [T])| {
            for j in low[i]..n {
                let lo = low[i];
                x[j] -= T::dot_slices(&x[lo..j], &lr[j * n + lo..j * n + j]);
            }
        };
        if ways > 1 {
            // rows per task: one grain of dot work
            let work = ip::weight(T::precision_bits() as usize) * (n * n / 2).max(1) as u128;
            let tile = (ip::grain() / work).max(1) as usize;
            rows.par_chunks_mut(n)
                .enumerate()
                .with_min_len(tile)
                .for_each(solve);
        } else {
            rows.chunks_mut(n).enumerate().for_each(solve);
        }
        for (e, v) in rows.into_iter().enumerate() {
            b[e / n + e % n * ldb] = v;
        }
        return;
    }
    let (n1, n2) = (n / 2, n - n / 2);
    solve_unit_lower_t(
        m,
        n1,
        l,
        ldl,
        b,
        ldb,
        &low.iter().map(|&f| f.min(n1)).collect::<Vec<_>>(),
        ways,
    );
    let (b1, b2) = b.split_at_mut(n1 * ldb);
    let mut p = vec![T::zero(); m * n2];
    exact_product(
        (b'N', b'T'),
        (m, n2, n1),
        (b1, ldb),
        (&l[n1..], ldl),
        &mut p,
        ways,
    );
    for j in 0..n2 {
        for i in 0..m {
            b2[i + j * ldb] -= p[i + j * m];
        }
    }
    let low2: Vec<usize> = low.iter().map(|&f| f.saturating_sub(n1)).collect();
    solve_unit_lower_t(m, n2, &l[n1 + n1 * ldl..], ldl, b2, ldb, &low2, ways);
}

/// `c = op(a)·op(b)` (`m × n`, inner `k`): one exact sum rounded once per
/// entry, with `ways` offered to the residue product.
fn exact_product<T: FloatT>(
    (ta, tb): (u8, u8),
    (m, n, k): (usize, usize, usize),
    (a, lda): (&[T], usize),
    (b, ldb): (&[T], usize),
    c: &mut [T],
    ways: usize,
) {
    crate::algebra::with_split_hint(ways, || {
        let (mi, ni, ki) = (m as i32, n as i32, k as i32);
        T::xgemm(
            ta,
            tb,
            mi,
            ni,
            ki,
            T::one(),
            a,
            lda as i32,
            b,
            ldb as i32,
            T::zero(),
            c,
            mi,
        )
    });
}

/// `rayon::join` when spare workers may take `b`; under a board offer `b`
/// is posted instead, so this thread never waits in a join.
fn join_if<RA: Send, RB: Send>(
    paired: bool,
    a: impl FnOnce() -> RA + Send,
    b: impl FnOnce() -> RB + Send,
) -> (RA, RB) {
    use sdpx_arithmetic::inner_parallel::board;
    use std::sync::Mutex;
    if !paired {
        return (a(), b());
    }
    if !board::offered() {
        return rayon::join(a, b);
    }
    let (a, b) = (Mutex::new(Some(a)), Mutex::new(Some(b)));
    let (ra, rb) = (Mutex::new(None), Mutex::new(None));
    board::run(2, &|i| {
        if i == 0 {
            *ra.lock().unwrap() = a.lock().unwrap().take().map(|f| f());
        } else {
            *rb.lock().unwrap() = b.lock().unwrap().take().map(|f| f());
        }
    });
    (ra.into_inner().unwrap().unwrap(), rb.into_inner().unwrap().unwrap())
}

/// Unpivoted dense LDLᵀ with the same per-pivot dynamic regularization
/// semantics as QDLDL: a pivot that violates its expected sign by more than
/// `eps` is clamped to `delta * sign` and counted.
struct DenseLeaf<T> {
    n: usize,
    l: Vec<T>,
    dinv: Vec<T>,
    /// Row block of every backward sweep of this factor.
    back: usize,
    /// First nonzero column of each row of a factored leaf (empty: none
    /// recorded). L keeps this envelope exactly: the products that would
    /// fill it in are exact zeros.
    first: Vec<usize>,
}

impl<T: FloatT> DenseLeaf<T> {
    fn new(n: usize, back: usize) -> Self {
        Self {
            n,
            l: vec![T::zero(); n * n],
            dinv: vec![T::zero(); n],
            back,
            first: Vec::new(),
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
        if split && n > f64_scaled::<T>(SPLIT_FACTOR_MIN, 8) {
            return self.factor_panels(signs, reg, regularize_count);
        }
        for k in 0..n {
            let s = T::from_i8(signs[if signs.len() == 1 { 0 } else { k }]).unwrap();
            let d = Self::pivot(self.l[k + k * n], s, reg, regularize_count)?;
            self.dinv[k] = T::one() / d;
            self.l[k + k * n] = T::one();
            for i in k + 1..n {
                self.l[i + k * n] *= self.dinv[k];
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

    /// [`Self::factor_signed`] in panels of `FACTOR_PANEL` pivots: each
    /// panel's columns are factored serially, then every later column
    /// receives the panel's updates in ascending pivot order in one parallel
    /// pass (one join per panel instead of per pivot). Entry (i, j) still
    /// sees the fused updates k = 0, 1, ..., j-1 in order with the same
    /// operands, so the factor is bitwise identical to the serial loop.
    fn factor_panels(
        &mut self,
        signs: &[i8],
        reg: Option<(T, T)>,
        regularize_count: &mut usize,
    ) -> Result<(), &'static str> {
        let n = self.n;
        let mut d = vec![T::zero(); n];
        let mut p = 0;
        while p < n {
            let pe = (p + FACTOR_PANEL).min(n);
            for k in p..pe {
                let s = T::from_i8(signs[if signs.len() == 1 { 0 } else { k }]).unwrap();
                let dk = Self::pivot(self.l[k + k * n], s, reg, regularize_count)?;
                d[k] = dk;
                self.dinv[k] = T::one() / dk;
                self.l[k + k * n] = T::one();
                for i in k + 1..n {
                    self.l[i + k * n] *= self.dinv[k];
                }
                for j in k + 1..pe {
                    let v = self.l[j + k * n] * dk;
                    for i in j..n {
                        self.l[i + j * n] = (-self.l[i + k * n]).mul_add(v, self.l[i + j * n]);
                    }
                }
            }
            if pe < n {
                let (head, tail) = self.l.split_at_mut(pe * n);
                let panel = &head[p * n..];
                let d = &d[p..pe];
                tail.par_chunks_mut(n)
                    .enumerate()
                    .for_each(|(offset, column)| {
                        let j = pe + offset;
                        for (q, &dk) in d.iter().enumerate() {
                            let pivot = &panel[q * n..(q + 1) * n];
                            let v = pivot[j] * dk;
                            for i in j..n {
                                column[i] = (-pivot[i]).mul_add(v, column[i]);
                            }
                        }
                    });
            }
            p = pe;
        }
        Ok(())
    }

    /// A leaf's [`Self::factor_signed`]: large wide-type leaves factor
    /// recursively, the others by exact dots. The border takes exact dots
    /// (mixed Λ27/1024: a blocked exact form took 2.0 s instead of 0.58 s per
    /// iteration for the scalar panels).
    fn factor_leaf(
        &mut self,
        a: &[T],
        signs: &[i8],
        reg: Option<(T, T)>,
        regularize_count: &mut usize,
        split: bool,
    ) -> Result<(), &'static str> {
        self.envelope(a);
        if T::precision_bits() > 64 {
            if self.n >= REC_FACTOR_MIN {
                return self.factor_rec(a, signs, reg, regularize_count, split);
            }
            return self.factor_dots(a, signs, reg, regularize_count, split);
        }
        self.factor_signed(a, signs, reg, regularize_count, split)
    }

    /// Each row's first nonzero column of `a` (`first`).
    fn envelope(&mut self, a: &[T]) {
        let n = self.n;
        self.first.clear();
        self.first
            .extend((0..n).map(|i| (0..i).find(|&k| !a[i + k * n].is_zero()).unwrap_or(i)));
    }

    /// Left-looking [`Self::factor_signed`] for wide types: entry (i, j) is
    /// one exact dot product of rows of L and W = L·D (rounded entrywise),
    /// rounded once, instead of j fused updates ([`ldl_dots`]).
    fn factor_dots(
        &mut self,
        a: &[T],
        signs: &[i8],
        reg: Option<(T, T)>,
        regularize_count: &mut usize,
        split: bool,
    ) -> Result<(), &'static str> {
        let n = self.n;
        self.l.copy_from_slice(a);
        let sign = |k: usize| T::from_i8(signs[if signs.len() == 1 { 0 } else { k }]).unwrap();
        ldl_dots(
            &mut self.l,
            n,
            n,
            &self.first,
            &sign,
            reg,
            regularize_count,
            split,
            &mut self.dinv,
        )
    }

    /// Recursive [`Self::factor_dots`] for wide leaves from `REC_FACTOR_MIN`:
    /// [`ldl_rec`] makes the panel solve and the Schur update large exact
    /// products, so residue operands are encoded once per large product
    /// instead of every entry taking its own exact dot. Thread-invariant but
    /// not bitwise equal to [`Self::factor_dots`] (one rounding per level).
    /// `split`: the leaf may use the pool's spare workers.
    fn factor_rec(
        &mut self,
        a: &[T],
        signs: &[i8],
        reg: Option<(T, T)>,
        regularize_count: &mut usize,
        split: bool,
    ) -> Result<(), &'static str> {
        let n = self.n;
        self.l.copy_from_slice(a);
        let sign = |k: usize| T::from_i8(signs[if signs.len() == 1 { 0 } else { k }]).unwrap();
        let ways = if split {
            rayon::current_num_threads()
        } else {
            1
        };
        let mut count = 0;
        let ok = ldl_rec(
            &mut self.l,
            n,
            n,
            &self.first,
            &sign,
            reg,
            &mut count,
            ways,
            &mut self.dinv,
        );
        *regularize_count += count;
        ok
    }

    fn forward(&self, x: &mut [T]) {
        self.forward_suffix(x, 0);
    }

    /// [`Self::forward`] split over the ambient pool: once a row block is
    /// final, every later row applies that block's columns in ascending order
    /// (rows are independent), then the next block finishes serially. Each
    /// entry sees the serial FMA order, so results are bitwise identical.
    fn forward_blocked(&self, x: &mut [T], block: usize) {
        let n = self.n;
        let mut r0 = 0;
        while r0 < n {
            let r1 = (r0 + block).min(n);
            for i in r0..r1 {
                for k in r0..i {
                    x[i] = (-self.l[i + k * n]).mul_add(x[k], x[i]);
                }
            }
            let (head, tail) = x.split_at_mut(r1);
            let done = &head[r0..r1];
            let rows_per = (block / 2).max(1);
            tail.par_chunks_mut(rows_per)
                .enumerate()
                .for_each(|(c, rows)| {
                    for (j, v) in rows.iter_mut().enumerate() {
                        let i = r1 + c * rows_per + j;
                        for (k, &xk) in (r0..r1).zip(done) {
                            *v = (-self.l[i + k * n]).mul_add(xk, *v);
                        }
                    }
                });
            r0 = r1;
        }
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

    /// Copy the negated factor rows from `start` into the strict upper
    /// triangle, which no sweep reads after factoring (the next factor
    /// recopies its input): row `r` lies at `l[r * n + start..=r * n + r]`,
    /// ending in the unit diagonal.
    fn stage_rows(&mut self, start: usize) {
        let n = self.n;
        for r in start..n {
            for k in start..r {
                self.l[k + r * n] = -self.l[r + k * n];
            }
            self.l[r + r * n] = T::one();
        }
    }

    /// [`Self::forward_suffix`] after [`Self::stage_rows`]: each entry is
    /// `x[i] - Σ L x` as one exact dot product rounded once. Never less
    /// accurate than the FMA chain, but not bitwise equal to it.
    fn forward_suffix_rows(&self, x: &mut [T], start: usize) {
        let n = self.n;
        for i in 0..x.len() {
            let r = i + start;
            let lo = self.first.get(r).map_or(start, |&f| f.max(start));
            let v = T::dot_slices(&self.l[r * n + lo..=r * n + r], &x[lo - start..=i]);
            x[i] = v;
        }
    }

    /// [`Self::backward`] after [`Self::stage_rows`]`(0)`: each entry is
    /// `x[i] - Σ_{k>i} L_ki x_k` as one exact dot rounded once. Column `i`
    /// from its staged unit diagonal is contiguous, so solved entries are
    /// kept negated (exactly) and negated back at the end. Never less
    /// accurate than the FMA chain, but not bitwise equal to it.
    fn backward_staged(&self, x: &mut [T]) {
        let n = self.n;
        for i in (0..n).rev() {
            let v = T::dot_slices(&self.l[i * n + i..(i + 1) * n], &x[i..]);
            x[i] = -v;
        }
        for v in x.iter_mut() {
            *v = -*v;
        }
    }

    /// [`Self::backward`] split over the ambient pool: the bottom row block
    /// finishes serially, then every earlier row applies that block's columns
    /// in parallel. A row sums its terms block by block from the bottom,
    /// ascending within a block: the `back` order of the serial sweep, so
    /// the bits are identical.
    fn backward_blocked(&self, x: &mut [T]) {
        let n = self.n;
        let block = self.back;
        let mut r1 = n;
        while r1 > 0 {
            let r0 = r1.saturating_sub(block);
            for i in (r0..r1).rev() {
                for k in i + 1..r1 {
                    x[i] = (-self.l[k + i * n]).mul_add(x[k], x[i]);
                }
            }
            let (head, tail) = x.split_at_mut(r0);
            let done = &tail[..r1 - r0];
            let rows_per = (block / 2).max(1);
            head.par_chunks_mut(rows_per)
                .enumerate()
                .for_each(|(c, rows)| {
                    for (j, v) in rows.iter_mut().enumerate() {
                        let i = c * rows_per + j;
                        let column = &self.l[r0 + i * n..r1 + i * n];
                        for (&lk, &xk) in column.iter().zip(done) {
                            *v = (-lk).mul_add(xk, *v);
                        }
                    }
                });
            r1 = r0;
        }
    }

    /// Backward sweep in the `back` order.
    fn backward(&self, x: &mut [T]) {
        let n = self.n;
        for i in (0..n).rev() {
            let mut r1 = n;
            while r1 > i + 1 {
                let r0 = r1.saturating_sub(self.back);
                for k in r0.max(i + 1)..r1 {
                    x[i] = (-self.l[k + i * n]).mul_add(x[k], x[i]);
                }
                r1 = r0;
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
        let n = self.n;
        for i in (0..n).rev() {
            let mut r1 = n;
            while r1 > i + 1 {
                let r0 = r1.saturating_sub(self.back);
                for k in r0.max(i + 1)..r1 {
                    let l = -self.l[k + i * n];
                    for c in 0..cols {
                        x[i * cols + c] = l.mul_add(x[k * cols + c], x[i * cols + c]);
                    }
                }
                r1 = r0;
            }
        }
    }

    /// `L D Lᵀ x = b` for a row-interleaved panel (entry (i, c) at
    /// `i * cols + c`); every column keeps the single-RHS order.
    fn solve_many(&self, x: &mut [T], cols: usize) {
        self.forward_many(x, cols);
        for j in 0..self.n {
            for c in 0..cols {
                x[j * cols + c] *= self.dinv[j];
            }
        }
        self.backward_many(x, cols);
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
    /// next block finishes serially; the backward sweep is blocked the same
    /// way from the bottom ([`Self::backward_blocked`]).
    fn solve_pooled(&self, x: &mut [T], pool: Option<&rayon::ThreadPool>) {
        match pool.filter(|p| {
            p.current_num_threads() > 1 && self.n >= f64_scaled::<T>(2 * FORWARD_BLOCK, 16)
        }) {
            Some(pool) => pool.install(|| {
                self.forward_blocked(x, FORWARD_BLOCK);
                for (x, d) in x.iter_mut().zip(&self.dinv) {
                    *x *= *d;
                }
                self.backward_blocked(x);
            }),
            None => {
                self.forward(x);
                for (x, d) in x.iter_mut().zip(&self.dinv) {
                    *x *= *d;
                }
                self.backward(x);
            }
        }
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
    // `contribution_product`, formed in the leaf phase (empty otherwise).
    product: Vec<T>,
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
            factor: DenseLeaf::new(g, LEAF_BLOCK),
            y: vec![T::zero(); g * t],
            // Generic leaves form Z = D⁻¹Y one column at a time while the
            // border Schur is assembled; only local structures store it.
            z: Vec::new(),
            w: Vec::new(),
            v: Vec::new(),
            batch_w: Vec::new(),
            batch_v: Vec::new(),
            product: Vec::new(),
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
            .factor_leaf(&self.h, &self.signs, reg, regularize_count, split_factor)?;
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
        // Exactly accumulated types solve Y by rows: one exact dot per entry.
        let rows = T::precision_bits() > 64;
        if rows {
            self.factor.stage_rows(start);
        }
        let (factor, couples, packed) = (&self.factor, &self.couples, self.packed);
        let solve = |column: &mut [T]| {
            if rows {
                factor.forward_suffix_rows(column, start)
            } else {
                factor.forward_suffix(column, start)
            }
        };
        // Posting pays from one grain of work per extra item: the board is
        // one lock, which thousands of small leaves would contend for (csdr3,
        // 4200 SOC leaves at 4 threads: +2.3%).
        let posted = sdpx_arithmetic::inner_parallel::board::offered() && {
            use sdpx_arithmetic::inner_parallel as ip;
            let columns = self.coupled.len();
            let work = ip::weight(T::precision_bits() as usize) * (width * width / 2 * columns) as u128;
            ip::tasks_if(true, work, columns) > 1
        };
        if split && width > 0 && posted {
            // One posted item per column, so idle workers of the leaf phase
            // help the longest Y first and no worker waits in a join.
            let columns: Vec<std::sync::Mutex<&mut [T]>> = self
                .y
                .chunks_mut(width)
                .enumerate()
                .filter(|(j, _)| packed || couples[*j])
                .map(|(_, column)| std::sync::Mutex::new(column))
                .collect();
            sdpx_arithmetic::inner_parallel::board::run(columns.len(), &|i| {
                solve(&mut columns[i].lock().unwrap())
            });
        } else if split && width > 0 {
            self.y
                .par_chunks_mut(width)
                .enumerate()
                .filter(|(j, _)| packed || couples[*j])
                .for_each(|(_, column)| solve(column));
        } else if packed {
            for column in self.y.chunks_mut(width) {
                solve(column);
            }
        } else {
            for &j in &self.coupled {
                solve(&mut self.y[j * width..(j + 1) * width]);
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
    fn first_many(&mut self, rhs: &[T], n: usize, cols: usize, chunks: usize, split: bool) {
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
        } else if cols == 1 && split && self.factor.n >= f64_scaled::<T>(LEAF_SPLIT_MIN, 16) {
            // One column has the single-RHS layout and order.
            self.factor.forward_blocked(&mut self.batch_w, LEAF_BLOCK);
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
    fn second_many(&mut self, xt: &[T], cols: usize, chunks: usize, split: bool) {
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
        if rayon::current_thread_index().is_some()
            && g * coupled.len() * cols >= f64_scaled::<T>(1 << 14, 16)
        {
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
        } else if cols == 1 && split && self.factor.n >= f64_scaled::<T>(LEAF_SPLIT_MIN, 16) {
            self.factor.backward_blocked(&mut self.batch_w);
        } else {
            self.factor.backward_many(&mut self.batch_w, cols);
        }
    }

    /// Whether a single-RHS solve of this leaf can use pool workers.
    fn solve_splits(&self) -> bool {
        self.factor.n >= f64_scaled::<T>(LEAF_SPLIT_MIN, 16)
            || self.ids.len() * self.coupled.len() >= f64_scaled::<T>(1 << 14, 16)
    }

    /// `split`: the pool has spare workers beyond the leaves, so a large
    /// leaf's forward sweep runs blocked over them (bitwise identical).
    fn first_solve(&mut self, rhs: &[T], split: bool) {
        if self.w.is_empty() {
            let g = self.ids.len();
            self.w.resize(g, T::zero());
            self.v = vec![T::zero(); g - self.coupling_start];
        }
        for (w, &id) in self.w.iter_mut().zip(&self.ids) {
            *w = rhs[id];
        }
        if split && self.factor.n >= f64_scaled::<T>(LEAF_SPLIT_MIN, 16) {
            self.factor.forward_blocked(&mut self.w, LEAF_BLOCK);
        } else {
            self.factor.forward(&mut self.w);
        }
        for i in self.coupling_start..self.ids.len() {
            self.v[i - self.coupling_start] = self.w[i] * self.factor.dinv[i];
        }
    }

    fn second_solve(&mut self, xt: &[T], split: bool) {
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
        if rayon::current_thread_index().is_some()
            && g * coupled.len() >= f64_scaled::<T>(1 << 14, 16)
        {
            self.w
                .par_iter_mut()
                .enumerate()
                .with_min_len(8)
                .for_each(update);
        } else {
            self.w.iter_mut().enumerate().for_each(update);
        }
        if split && self.factor.n >= f64_scaled::<T>(LEAF_SPLIT_MIN, 16) {
            self.factor.backward_blocked(&mut self.w);
        } else {
            self.factor.backward(&mut self.w);
        }
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
        split: bool,
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
        } else if cols == 1 && split && self.factor.n >= f64_scaled::<T>(LEAF_SPLIT_MIN, 16) {
            self.factor.backward_blocked(&mut self.batch_w);
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
/// Column chunks per leaf for a batched solve of `cols` columns on
/// `workers`: one wave of pieces, each extra chunk going to the leaf whose
/// piece (`g·(g + coupled)` work per column) is largest, at least four
/// columns per chunk. The even `⌈workers/leaves⌉` left an owner's 3 leaves
/// on 16 workers as 18 pieces, a second wave, and ignored leaf sizes.
/// Columns are independent, so the chunks change timing only.
fn leaf_chunks<T: FloatT>(leaves: &[Leaf<T>], workers: usize, cols: usize) -> Vec<usize> {
    let work: Vec<f64> = leaves
        .iter()
        .map(|l| (l.ids.len() * (l.ids.len() + l.coupled.len())) as f64)
        .collect();
    one_wave_chunks(&work, workers, (cols / 4).max(1))
}

/// Chunks per item, at most `cap` each and `workers` in all (at least one
/// each): every extra chunk goes to the item with the largest piece
/// `work / chunks`, ties to the earlier item.
fn one_wave_chunks(work: &[f64], workers: usize, cap: usize) -> Vec<usize> {
    let mut chunks = vec![1; work.len()];
    for _ in work.len()..workers {
        let next = (0..work.len())
            .filter(|&i| chunks[i] < cap)
            .max_by(|&a, &b| {
                let piece = |i: usize| work[i] / chunks[i] as f64;
                piece(a).total_cmp(&piece(b)).then(b.cmp(&a))
            });
        let Some(i) = next else {
            break;
        };
        chunks[i] += 1;
    }
    chunks
}

/// Leaves whose `YᵀD⁻¹Y` is one residue product ([`contribution_product`]):
/// substantial products whose result storage is proportional to Y.
fn residue_contribution<T: FloatT>(leaf: &Leaf<T>, t: usize) -> bool {
    let (g, q) = (leaf.ids.len(), leaf.coupled.len());
    T::residue_product_applies(q, q, g) && g * q * q >= 1 << 21 && q * q <= 8 * g * t
}

/// Working-precision storage of one [`contribution_product`]: Y, Z and the
/// product (its residues are transient).
fn contribution_bytes<T: FloatT>(leaf: &Leaf<T>) -> usize {
    let (g, q) = (leaf.ids.len(), leaf.coupled.len());
    (2 * g * q + q * q) * std::mem::size_of::<T>()
}

/// Upper triangle of a factored leaf's `YᵀD⁻¹Y` over its coupled columns,
/// each entry one exact sum rounded once (Z = D⁻¹Y rounded entrywise).
fn contribution_product<T: FloatT>(leaf: &Leaf<T>, t: usize) -> Vec<T> {
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
    let exact = T::xgemm_upper_exact(b'T', b'N', q, q, g, y, g, &z, g, &mut product, None, None);
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
}

fn subtract_contributions<T: FloatT>(
    leaves: &mut [Leaf<T>],
    s: &mut [T],
    t: usize,
    pool: Option<&rayon::ThreadPool>,
) {
    if t == 0 {
        return;
    }
    // Batch substantial residue products when result storage is
    // proportional to Y. A batch holds at most one leaf
    // per worker; a short batch grants its products the spare workers. Both
    // paths round every entry's exact sum once, so batching never changes
    // the bits. Short groups batch too: on Λ35 (36 leaves of g ≈ 114,
    // q = 170, 64 threads) residue products cut contribution CPU 2.3× and
    // iterations 3.8%, for 1.5 GB more peak RSS (5.6 → 7.2 GB; residue
    // workspace of the concurrent products). Products the leaf phase
    // already formed (`Leaf::product`) are only subtracted here.
    let width = pool.map_or(1, |p| p.current_num_threads());
    let eligible = |leaf: &Leaf<T>| residue_contribution(leaf, t);
    if leaves.iter().any(eligible) {
        for group in leaves.chunk_by_mut(|a, b| eligible(a) == eligible(b)) {
            if !eligible(&group[0]) {
                // Keep these leaves column-parallel and avoid square result
                // storage, preserving their position in every entry's sum.
                subtract_contributions_columns(group, s, t, pool);
                continue;
            }
            // At most one leaf per worker and about `BATCH_WORK_BYTES` of
            // workspace per batch, in batches of equal leaf counts: Y, Z and
            // the product in working precision (the residues are transient).
            // Mixed Λ27/1024 (border 524): batching all 117 leaves raised
            // 128-thread peak RSS from 23 to 36 GiB, about this estimate;
            // also counting residues split it into batches of 2–3 leaves
            // and slowed contributions 6.2 → 10.7 s at 64 threads.
            let bytes: usize = group.iter().map(contribution_bytes).sum();
            let count = group
                .len()
                .div_ceil(width)
                .max(bytes.div_ceil(BATCH_WORK_BYTES))
                .min(group.len());
            let size = group.len().div_ceil(count);
            for batch in group.chunks_mut(size) {
                let ways = width.div_ceil(batch.len()).clamp(1, 8);
                let build = |leaf: &mut Leaf<T>| {
                    if leaf.product.is_empty() {
                        crate::algebra::with_split_hint(ways, || contribution_product(leaf, t))
                    } else {
                        std::mem::take(&mut leaf.product)
                    }
                };
                let products: Vec<Vec<T>> = match pool {
                    Some(p) => p.install(|| batch.par_iter_mut().map(build).collect()),
                    None => batch.iter_mut().map(build).collect(),
                };
                // Columns are independent; each entry still subtracts leaves
                // in the original order before the next batch begins.
                let batch = &*batch;
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
    subtract_contributions_columns(leaves, s, t, pool);
}

/// Column-parallel `s -= Σ YᵀD⁻¹Y`, one exact dot per entry.
fn subtract_contributions_columns<T: FloatT>(
    leaves: &[Leaf<T>],
    s: &mut [T],
    t: usize,
    pool: Option<&rayon::ThreadPool>,
) {
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
    if let Some(pool) = pool.filter(|p| t > 1 && p.current_num_threads() > 1) {
        return queued_contribution_columns(leaves, s, t, pool);
    }
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

/// [`subtract_contributions_columns`] as largest-first queue jobs: each
/// column's exact dot count is known, so the queue packs them in LPT order,
/// and a column above an eighth of a worker's share splits into row ranges.
/// Every entry still subtracts its leaves in order, so the bits are those of
/// the column path.
fn queued_contribution_columns<T: FloatT>(
    leaves: &[Leaf<T>],
    s: &mut [T],
    t: usize,
    pool: &rayon::ThreadPool,
) {
    let workers = pool.current_num_threads();
    // Dot terms of column j: Σ over coupling leaves of (rows ≤ j)·g.
    let work: Vec<usize> = (0..t)
        .map(|j| {
            leaves
                .iter()
                .filter(|l| l.couples[j])
                .map(|l| (l.coupled.partition_point(|&i| i <= j) + 1) * l.ids.len())
                .sum()
        })
        .collect();
    let grain = (work.iter().sum::<usize>() / (8 * workers)).max(1);
    let mut jobs: Vec<crate::algebra::Job<'_>> = Vec::new();
    for (j, col) in s.chunks_mut(t).enumerate() {
        if work[j] == 0 {
            continue;
        }
        let pieces = work[j].div_ceil(grain).clamp(1, j + 1);
        let rows = (j + 1).div_ceil(pieces);
        for (p, part) in col[..=j].chunks_mut(rows).enumerate() {
            let (r0, len) = (p * rows, part.len());
            let r1 = r0 + len;
            let est = (work[j] * len) as f64 / (j + 1) as f64;
            let job = move || {
                for leaf in leaves.iter().filter(|l| l.couples[j]) {
                    let coupled = &leaf.coupled;
                    let c = coupled.binary_search(&j).unwrap();
                    let lo = coupled.partition_point(|&i| i < r0);
                    let hi = coupled[..=c].partition_point(|&i| i < r1);
                    if lo >= hi {
                        continue;
                    }
                    let g = leaf.ids.len();
                    let (y, dinv) = (&leaf.y, &leaf.factor.dinv);
                    let z: Vec<T> = (0..g).map(|k| y[k + j * g] * dinv[k]).collect();
                    for &i in &coupled[lo..hi] {
                        part[i - r0] -= T::dot_fma(y[i * g..(i + 1) * g].iter().zip(&z));
                    }
                }
            };
            jobs.push((est, j, Box::new(job)));
        }
    }
    pool.install(|| crate::algebra::run_queue(jobs, workers));
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
    // the caller's ordering for the fallback (None: QDLDL's own AMD)
    fallback_perm: Option<Vec<usize>>,
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
    cone_panels: Option<local_cones::ConePanels<T>>,
    ranks: Option<LeafRanks>,
    /// One plan per index map: an update writes the cone blocks and then
    /// each operator relative entropy cone's rows, so a single cached plan
    /// was rebuilt serially for every map (BFSS L11u: about 17M entries per
    /// iteration).
    value_plans: Vec<ValuePlan>,
}

/// Entries from which `update_values` routes through a [`ValuePlan`].
const VALUE_PLAN_MIN: usize = 1 << 15;
/// Cached plans beyond which the cache restarts (the KKT has a few fixed maps).
const VALUE_PLANS_MAX: usize = 64;

/// The routing of one `update_values` index map. The KKT maps are fixed, so
/// the column search and owner lookup run once; leaf entries are then written
/// per leaf in parallel, and trunk and bound-panel entries (`rest`, positions
/// in `index`) keep the serial route. Every entry gets the same value as in
/// the serial route.
struct ValuePlan {
    index: Vec<usize>,
    panels: bool,
    leaves: Vec<Vec<(u32, LeafSlot)>>,
    rest: Vec<usize>,
}

#[derive(Clone, Copy)]
enum LeafSlot {
    /// `h[.0]`, mirrored to `h[.1]`.
    H(u32, u32),
    /// `raw_coupling(row, col)`.
    Coupling(u32, u32),
}

/// The column of CSC `position`, searched from column `j`. Index maps are
/// mostly increasing: step forward a few columns before falling back to a
/// binary search.
fn column_of(colptr: &[usize], mut j: usize, position: usize) -> usize {
    if position < colptr[j] || position >= colptr[j + 1] {
        let mut steps = 0;
        while position >= colptr[j + 1] && steps < 4 && j + 2 < colptr.len() {
            j += 1;
            steps += 1;
        }
        if position < colptr[j] || position >= colptr[j + 1] {
            j = colptr.partition_point(|&p| p <= position) - 1;
        }
    }
    j
}

impl<T: FloatT> ArrowLDLSolver<T> {
    /// Eligibility is structural only: connected components of the
    /// positive-sign subgraph become leaves, negative-sign variables become
    /// the border.  Returns `None` when the shape is not exploitable, so the
    /// caller transparently keeps QDLDL.
    pub fn try_new(k: &CscMatrix<T>, signs: &[i8], settings: &CoreSettings<T>) -> Option<Self> {
        let (groups, trunk, _) = Self::plan(k, signs)?;
        Some(Self::from_groups(k, signs, settings, groups, trunk, None))
    }

    /// Dense work estimate of the arrow of `signs`; `None` when ineligible.
    pub fn work(k: &CscMatrix<T>, signs: &[i8]) -> Option<u128> {
        Self::plan(k, signs).map(|(_, _, work)| work)
    }

    /// [`Self::try_new`]'s leaves, border and dense work estimate.
    fn plan(k: &CscMatrix<T>, signs: &[i8]) -> Option<(Vec<Vec<usize>>, Vec<usize>, u128)> {
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
        Some((groups, trunk, leaf_work + border_work))
    }

    /// The arrow of −K: dense negative-sign blocks (cone Hessians retained
    /// in the KKT, such as operator relative entropy rows) become leaves and
    /// the primal variables the border. LDLᵀ of −K is that of K with D
    /// negated, so values and right-hand sides change sign exactly. A failed
    /// factor falls back to QDLDL in `perm` (the caller's ordering of K).
    pub fn try_negated(
        k: &CscMatrix<T>,
        signs: &[i8],
        settings: &CoreSettings<T>,
        perm: Option<Vec<usize>>,
    ) -> Option<Negated<T>> {
        let mut neg = k.clone();
        neg.nzval.iter_mut().for_each(|v| *v = -*v);
        let signs: Vec<i8> = signs.iter().map(|&s| -s).collect();
        let mut inner = Self::try_new(&neg, &signs, settings)?;
        inner.fallback_perm = perm;
        Some(Negated { inner })
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
        // Local cone leaves: their sorted border columns, and the first leaf
        // position with a border coupling (the suffix starts there).
        let mut links = if matches!(local_structure, Some(LocalStructure::Cones)) {
            let mut links: Vec<(Vec<usize>, usize)> =
                groups.iter().map(|g| (Vec::new(), g.len())).collect();
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
                    link.1 = link.1.min(local[inner]);
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
                        LocalStructure::Cones => leaf.ids.len() - links[gi].1,
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
                            // MPFR forms Z = D⁻¹Y inside the exact Schur
                            // product, binary64 in its class panels.
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
                            if T::precision_bits() <= 64 {
                                leaf.z.resize(width * t, T::zero());
                            }
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
        let cone_panels = (matches!(local_structure, Some(LocalStructure::Cones))
            && T::precision_bits() <= 64)
            .then(|| local_cones::ConePanels::new(&classes, &leaves));
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
            tf: DenseLeaf::new(t, FORWARD_BLOCK),
            tx: vec![T::zero(); t],
            batch_tx: Vec::new(),
            pool: None,
            regularize_count: 0,
            fallback: None,
            fallback_perm: None,
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
            cone_panels,
            ranks: None,
            value_plans: Vec::new(),
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
            j = column_of(&self.colptr, j, position);
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

    /// Route a fixed index map once (see [`ValuePlan`]); `None` when the
    /// positions do not fit the compact plan.
    fn value_plan(&self, index: &[usize]) -> Option<ValuePlan> {
        if index.len() > u32::MAX as usize
            || self.leaves.iter().any(|l| l.ids.len().pow(2) > u32::MAX as usize)
        {
            return None;
        }
        let mut panels = self.exact_bound_panels.is_some();
        #[cfg(feature = "faer-sparse")]
        {
            panels |= self.bound_panels.is_some();
        }
        let mut leaves = vec![Vec::new(); self.leaves.len()];
        let mut rest = Vec::new();
        let mut j = 0;
        for (q, &position) in index.iter().enumerate() {
            j = column_of(&self.colptr, j, position);
            let i = self.rowval[position];
            let (li, lj) = (self.local[i], self.local[j]);
            let slot = match (self.owner[i], self.owner[j]) {
                (x, y) if x == usize::MAX && y == usize::MAX => None,
                (x, y) if x != usize::MAX && y != usize::MAX => {
                    let g = self.leaves[x].ids.len();
                    Some((x, LeafSlot::H((li + lj * g) as u32, (lj + li * g) as u32)))
                }
                _ if panels => None,
                (x, _) if x != usize::MAX => Some((x, LeafSlot::Coupling(li as u32, lj as u32))),
                (_, y) => Some((y, LeafSlot::Coupling(lj as u32, li as u32))),
            };
            match slot {
                Some((leaf, slot)) => leaves[leaf].push((q as u32, slot)),
                None => rest.push(q),
            }
        }
        Some(ValuePlan {
            index: index.to_vec(),
            panels,
            leaves,
            rest,
        })
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

    /// `x = K⁻¹b` for a trunk-free arrow (`K` block diagonal over its
    /// leaves), column by column with exact-dot rows: every forward and
    /// backward entry is one exact dot rounded once. Never less accurate than
    /// the lane panel's FMA chains, but not bitwise equal to them. Columns of
    /// every leaf run in parallel, each in the same order for any pool, so
    /// the bits do not depend on the thread count. Each column reads the
    /// whole factor, yet on the Λ35 owners' 170-column border panel at 1024
    /// bits this took 1.5–1.7× less time than the lane sweeps.
    fn solve_rows_panel(&mut self, x: &mut [T], b: &[T], cols: usize) {
        let n = self.n;
        assert_eq!(x.len(), n * cols);
        assert_eq!(b.len(), x.len());
        let parallel = self.pool.is_some();
        let solve = |leaf: &mut Leaf<T>| {
            let g = leaf.ids.len();
            if g == 0 {
                return;
            }
            leaf.factor.stage_rows(0);
            leaf.batch_w.resize(g * cols, T::zero());
            for (c, column) in leaf.batch_w.chunks_mut(g).enumerate() {
                for (v, &id) in column.iter_mut().zip(&leaf.ids) {
                    *v = b[c * n + id];
                }
            }
            let factor = &leaf.factor;
            let column = |w: &mut [T]| {
                factor.forward_suffix_rows(w, 0);
                for (v, &d) in w.iter_mut().zip(&factor.dinv) {
                    *v *= d;
                }
                factor.backward_staged(w);
            };
            if parallel {
                leaf.batch_w.par_chunks_mut(g).for_each(column);
            } else {
                leaf.batch_w.chunks_mut(g).for_each(column);
            }
        };
        match &self.pool {
            Some(pool) => pool.install(|| self.leaves.par_iter_mut().for_each(solve)),
            None => self.leaves.iter_mut().for_each(solve),
        }
        for leaf in self.leaves.iter().filter(|leaf| !leaf.ids.is_empty()) {
            let g = leaf.ids.len();
            for (c, column) in leaf.batch_w.chunks(g).enumerate().take(cols) {
                for (&v, &id) in column.iter().zip(&leaf.ids) {
                    x[c * n + id] = v;
                }
            }
        }
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
            // A large leaf forms its residue contribution as soon as it is
            // factored, so products fill the phase's tail instead of a
            // separate one (BFSS L11u: 2.7 s per iteration at 9.7 of 16
            // workers). Each product depends on its leaf alone, and
            // `subtract_contributions` keeps the order of the subtractions.
            // Products are admitted largest first while their workspace fits
            // one contribution batch (`BATCH_WORK_BYTES`); the others form in
            // `subtract_contributions`, batched as before.
            // A product above a worker's share of all products' work splits
            // into that many ways (the largest would otherwise end the phase).
            let product_work = |l: &Leaf<T>| (l.ids.len() * l.coupled.len().pow(2)) as f64;
            let mut admitted = vec![false; owned.len()];
            if self.local_structure.is_none() {
                let leaves = &self.leaves[owned.clone()];
                let mut eligible: Vec<usize> =
                    (0..leaves.len()).filter(|&k| residue_contribution(&leaves[k], t)).collect();
                eligible.sort_by(|&a, &b| product_work(&leaves[b]).total_cmp(&product_work(&leaves[a])));
                let mut bytes = 0;
                for k in eligible {
                    bytes += contribution_bytes(&leaves[k]);
                    if bytes > BATCH_WORK_BYTES {
                        break;
                    }
                    admitted[k] = true;
                }
            }
            let products: f64 = self.leaves[owned.clone()]
                .iter()
                .zip(&admitted)
                .filter(|(_, &admit)| admit)
                .map(|(l, _)| product_work(l))
                .sum();
            let share = products / threads as f64;
            let ways = |l: &Leaf<T>| crate::algebra::measured_ways(product_work(l), share);
            // Counts are diagnostic only; no per-leaf counter buffer is needed.
            let count = AtomicUsize::new(0);
            // Lanes serve a leaf that splits its factor (a straggler, whose
            // joins would let waiting workers steal whole leaves) and the
            // in-phase products; balanced leaves keep the parallel iterator
            // (binary64 nb, 797 leaves of g ≤ 6 on two workers: lanes cost
            // +1.3%). Lanes take every worker, so a factor running inside the
            // pool (owner blocks of an in-process partitioned solve) keeps the
            // parallel iterator too: two broadcasts there could wait on each
            // other through the board.
            let lanes = pool.current_thread_index().is_none()
                && (products > 0.0
                    || self.leaves[owned.clone()]
                        .iter()
                        .any(|l| split_factor(l.ids.len())));
            if !lanes {
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
                // One lane per worker claims leaves, largest first. Leaves are
                // not pool tasks, so a worker waiting inside one leaf's split
                // work steals other leaves' inner pieces, never a whole leaf
                // (BFSS L11u: over a parallel iterator, g = 10 leaves spent 4 s
                // in their Y solves and 1785 leaves' factors 3–9 s). Leaves post
                // their split work on the board, where lanes out of leaves help
                // until every leaf is done: a poster never waits in a join.
                let cost = |l: &Leaf<T>| (l.ids.len() as f64).powi(3) + product_work(l);
                let mut order: Vec<(&mut Leaf<T>, bool)> =
                    self.leaves[owned.clone()].iter_mut().zip(admitted).collect();
                order.sort_by(|a, b| cost(b.0).total_cmp(&cost(a.0)));
                let order: Vec<(std::sync::Mutex<&mut Leaf<T>>, bool)> = order
                    .into_iter()
                    .map(|(leaf, admit)| (std::sync::Mutex::new(leaf), admit))
                    .collect();
                let (next, failed) = (AtomicUsize::new(0), AtomicBool::new(false));
                let finished = AtomicUsize::new(0);
                pool.broadcast(|_| {
                    let _offer = sdpx_arithmetic::inner_parallel::board::Offer::enter();
                    while !failed.load(Ordering::Relaxed) {
                        let Some((slot, admit)) = order.get(next.fetch_add(1, Ordering::Relaxed))
                        else {
                            break;
                        };
                        let leaf: &mut Leaf<T> = &mut slot.lock().unwrap();
                        let mut n = 0;
                        let result =
                            leaf.refactor(t, reg, &mut n, split, split_factor(leaf.ids.len()));
                        if n != 0 {
                            count.fetch_add(n, Ordering::Relaxed);
                        }
                        if result.is_ok() && *admit {
                            leaf.product = crate::algebra::with_split_hint(ways(leaf), || {
                                contribution_product(leaf, t)
                            });
                        }
                        if result.is_err() {
                            failed.store(true, Ordering::Relaxed);
                        }
                        finished.fetch_add(1, Ordering::Release);
                    }
                    while finished.load(Ordering::Acquire) < order.len() && !failed.load(Ordering::Relaxed) {
                        if sdpx_arithmetic::inner_parallel::board::help() {
                            continue;
                        }
                        if !matches!(rayon::yield_now(), Some(rayon::Yield::Executed)) {
                            std::thread::yield_now();
                        }
                    }
                });
                self.regularize_count += count.into_inner();
                !failed.into_inner()
            }
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
            for leaf in &mut self.leaves {
                leaf.product = Vec::new();
            }
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
            subtract_contributions(&mut self.leaves[owned], &mut self.s, t, self.pool.as_deref());
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
        // Wide types take exact dots, as leaves do.
        let signs = &self.border_signs;
        let factor = |tf: &mut DenseLeaf<T>, count: &mut usize, split: bool| {
            if T::precision_bits() > 64 {
                tf.envelope(s);
                tf.factor_dots(s, signs, reg, count, split)
            } else {
                tf.factor_signed(s, signs, reg, count, split)
            }
        };
        let ok = match &self.pool {
            Some(pool) if tf.n >= f64_scaled::<T>(SPLIT_BORDER_FACTOR_MIN, 8) => {
                pool.install(|| factor(tf, &mut border_count, true))
            }
            _ => factor(tf, &mut border_count, false),
        }
        .is_ok();
        crate::receipt::finish("arrow.border_factor", timer);
        self.regularize_count += border_count;
        ok
    }

    fn factor_fallback(&mut self) -> bool {
        let matrix = self.materialize();
        if self.fallback.is_none() {
            let solver: BoxedDirectLDLSolver<T> =
                Box::new(super::qdldl::QDLDLDirectLDLSolver::new(
                    &matrix,
                    &self.signs,
                    &self.settings,
                    self.fallback_perm.clone(),
                ));
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
        if self.solve_cones(x, b, 1) {
            return;
        }
        let t = self.trunk.len();
        let owned = self.owned_leaves();
        // One leaf too small to split its sweeps gains nothing from a round
        // trip through the pool.
        let pool = self.pool.clone().filter(|_| {
            owned.len() > 1 || self.leaves[owned.clone()].iter().any(Leaf::solve_splits)
        });
        let timer = crate::receipt::start();
        if let Some(pool) = &pool {
            // Large leaf sweeps split; idle workers steal their row chunks, so
            // the largest leaves stop setting the phase's tail.
            let split = pool.current_num_threads() > 1;
            pool.install(|| {
                self.leaves[owned.clone()]
                    .par_iter_mut()
                    .for_each(|l| l.first_solve(b, split))
            });
        } else {
            for leaf in &mut self.leaves[owned.clone()] {
                leaf.first_solve(b, false);
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
        let split = pool.as_ref().is_some_and(|p| p.current_num_threads() > 1);
        let backward = |(i, leaf): (usize, &mut Leaf<T>)| match panels {
            Some(panels) => leaf.second_solve_bounds(tx, &panels.y, i, nl),
            None => leaf.second_solve(tx, split),
        };
        let first = owned.start;
        if let Some(pool) = &pool {
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

/// [`ArrowLDLSolver::try_negated`]: an arrow factor of −K solving K x = b.
pub struct Negated<T: FloatT> {
    inner: ArrowLDLSolver<T>,
}

impl<T: FloatT> Negated<T> {
    fn negated_solve(&mut self, b: &mut [T], solve: impl FnOnce(&mut ArrowLDLSolver<T>, &mut [T])) {
        b.iter_mut().for_each(|v| *v = -*v);
        solve(&mut self.inner, b);
        b.iter_mut().for_each(|v| *v = -*v);
    }
}

impl<T: FloatT> HasLinearSolverInfo for Negated<T> {
    fn linear_solver_info(&self) -> LinearSolverInfo {
        let mut info = self.inner.linear_solver_info();
        info.name = format!("negated_{}", info.name);
        info
    }
}

impl<T: FloatT> DirectLDLSolver<T> for Negated<T> {
    fn update_values(&mut self, index: &[usize], values: &[T]) {
        self.inner.update_values_mapped(index, values, |v| -v);
    }

    fn scale_values(&mut self, index: &[usize], scale: T) {
        self.inner.scale_values(index, scale);
    }

    fn solve(&mut self, kkt: &CscMatrix<T>, x: &mut [T], b: &mut [T]) {
        self.negated_solve(b, |s, b| s.solve(kkt, x, b));
    }

    fn solve_many(&mut self, kkt: &CscMatrix<T>, x: &mut [T], b: &mut [T], cols: usize) {
        self.negated_solve(b, |s, b| s.solve_many(kkt, x, b, cols));
    }

    fn solve_coupling_panel(&mut self, kkt: &CscMatrix<T>, x: &mut [T], b: &mut [T], cols: usize) {
        self.negated_solve(b, |s, b| s.solve_coupling_panel(kkt, x, b, cols));
    }

    fn refactor(&mut self, kkt: &CscMatrix<T>) -> bool {
        self.inner.refactor(kkt)
    }

    fn unshifted_columns(&self) -> &[usize] {
        self.inner.unshifted_columns()
    }

    fn set_shift_exemption(&mut self, active: bool) {
        self.inner.set_shift_exemption(active);
    }

    fn set_pool(&mut self, pool: Option<Arc<rayon::ThreadPool>>) {
        self.inner.set_pool(pool);
    }

    fn set_world(&mut self, world: crate::MpiContext) {
        self.inner.set_world(world);
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

impl<T: FloatT> ArrowLDLSolver<T> {
    /// `update_values` writing `map(value)` (exact maps only, such as
    /// negation, so every entry is the value a caller-side map would give).
    fn update_values_mapped(&mut self, index: &[usize], values: &[T], map: impl Fn(T) -> T + Sync) {
        let index = &index[..index.len().min(values.len())];
        let pool = self.pool.clone().filter(|p| {
            p.current_num_threads() > 1 && self.leaves.len() > 1 && index.len() >= VALUE_PLAN_MIN
        });
        let Some(pool) = pool else {
            self.update_entries(index.iter().copied(), |q, value| *value = map(values[q]));
            return;
        };
        let mut panels = self.exact_bound_panels.is_some();
        #[cfg(feature = "faer-sparse")]
        {
            panels |= self.bound_panels.is_some();
        }
        let cached = self.value_plans.iter().position(|plan| {
            plan.panels == panels
                && plan.index.len() == index.len()
                && plan.index.first() == index.first()
                && pool.install(|| {
                    plan.index
                        .par_chunks(VALUE_PLAN_MIN)
                        .zip(index.par_chunks(VALUE_PLAN_MIN))
                        .all(|(a, b)| a == b)
                })
        });
        let plan = match cached {
            Some(k) => self.value_plans.swap_remove(k),
            None => match self.value_plan(index) {
                Some(plan) => plan,
                None => {
                    self.update_entries(index.iter().copied(), |q, value| *value = map(values[q]));
                    return;
                }
            },
        };
        // Each leaf takes its entries in index order, so a repeated position
        // keeps its last value as in the serial route.
        pool.install(|| {
            self.leaves
                .par_iter_mut()
                .zip(&plan.leaves)
                .for_each(|(leaf, slots)| {
                    for &(q, slot) in slots {
                        let value = map(values[q as usize]);
                        match slot {
                            LeafSlot::H(at, mirror) => {
                                leaf.h[at as usize] = value;
                                leaf.h[mirror as usize] = value;
                            }
                            LeafSlot::Coupling(row, col) => {
                                *leaf.raw_coupling(row as usize, col as usize) = value
                            }
                        }
                    }
                })
        });
        let rest = &plan.rest;
        self.update_entries(rest.iter().map(|&q| index[q]), |k, value| {
            *value = map(values[rest[k]]);
        });
        if self.value_plans.len() >= VALUE_PLANS_MAX {
            self.value_plans.clear();
        }
        self.value_plans.push(plan);
    }
}

impl<T: FloatT> DirectLDLSolver<T> for ArrowLDLSolver<T> {
    fn update_values(&mut self, index: &[usize], values: &[T]) {
        self.update_values_mapped(index, values, |v| v);
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
        if self.solve_cones(x, b, cols) {
            return;
        }
        self.batch_tx.resize(t * cols, T::zero());
        let owned = self.owned_leaves();
        // Spare threads (more than leaves) split leaves' columns.
        let chunks = match &self.pool {
            Some(pool) => leaf_chunks(
                &self.leaves[owned.clone()],
                pool.current_num_threads(),
                cols,
            ),
            None => vec![1; owned.len()],
        };
        let prof = crate::receipt::start();
        if let Some(pool) = &self.pool {
            let split = pool.current_num_threads() > 1;
            pool.install(|| {
                self.leaves[owned.clone()]
                    .par_iter_mut()
                    .zip(&chunks)
                    .for_each(|(l, &chunks)| l.first_many(b, n, cols, chunks, split))
            });
        } else {
            self.leaves[owned.clone()]
                .iter_mut()
                .for_each(|l| l.first_many(b, n, cols, 1, false));
        }
        crate::receipt::finish("many.first", prof);
        let prof = crate::receipt::start();
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
        crate::receipt::finish("many.couple", prof);
        let prof = crate::receipt::start();
        // A single trunk column splits its sweeps over the pool.
        match self
            .pool
            .as_ref()
            .filter(|_| cols == 1 && t >= f64_scaled::<T>(2 * FORWARD_BLOCK, 16))
        {
            Some(pool) => {
                let tf = &self.tf;
                pool.install(|| {
                    tf.forward_blocked(tx, FORWARD_BLOCK);
                    for (x, d) in tx.iter_mut().zip(&tf.dinv) {
                        *x *= *d;
                    }
                    tf.backward_blocked(tx);
                });
            }
            None => {
                // Several trunk columns: independent column chunks share the
                // pool, each with the serial per-column order.
                match self.pool.as_ref().filter(|p| {
                    cols > 1
                        && p.current_num_threads() > 1
                        && t >= f64_scaled::<T>(2 * FORWARD_BLOCK, 16)
                }) {
                    Some(pool) => {
                        let chunks = pool.current_num_threads().min(cols);
                        let tf = &self.tf;
                        pool.install(|| tf.chunked(tx, cols, chunks, DenseLeaf::solve_many));
                    }
                    None => self.tf.solve_many(tx, cols),
                }
            }
        }
        crate::receipt::finish("many.trunk", prof);
        let prof = crate::receipt::start();
        let split = self
            .pool
            .as_ref()
            .is_some_and(|p| p.current_num_threads() > 1);
        let backward = |(i, (leaf, &chunks)): (usize, (&mut Leaf<T>, &usize))| match panels {
            Some(panels) => leaf.second_many_bounds(tx, &panels.y, i, nl, cols, chunks, split),
            None => leaf.second_many(tx, cols, chunks, split),
        };
        let first = owned.start;
        if let Some(pool) = &self.pool {
            pool.install(|| {
                self.leaves[owned.clone()]
                    .par_iter_mut()
                    .zip(&chunks)
                    .enumerate()
                    .for_each(|(i, leaf)| backward((first + i, leaf)))
            });
        } else {
            self.leaves[owned.clone()]
                .iter_mut()
                .zip(&chunks)
                .enumerate()
                .for_each(|(i, leaf)| backward((first + i, leaf)));
        }
        crate::receipt::finish("many.second", prof);
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

    fn solve_coupling_panel(&mut self, kkt: &CscMatrix<T>, x: &mut [T], b: &mut [T], cols: usize) {
        #[cfg(feature = "faer-sparse")]
        let bounds = self.bound_panels.is_some();
        #[cfg(not(feature = "faer-sparse"))]
        let bounds = false;
        // Trunk-free generic leaves at exactly accumulated precisions only.
        if !self.use_arrow
            || bounds
            || !self.trunk.is_empty()
            || self.ranks.is_some()
            || self.local_structure.is_some()
            || self.exact_bound_panels.is_some()
            || T::precision_bits() <= 64
            || self.leaves.iter().any(|leaf| leaf.signs.is_empty())
        {
            self.solve_many(kkt, x, b, cols);
            return;
        }
        let prof = crate::receipt::start();
        self.solve_rows_panel(x, b, cols);
        crate::receipt::finish("many.rows", prof);
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

    #[test]
    fn blocked_sweeps_match_serial_sweeps() {
        type T = sdpx_arithmetic::Bits256;
        let n = 200;
        let mut a = vec![T::zero(); n * n];
        for j in 0..n {
            for i in 0..n {
                let v =
                    1.0 / (1.0 + (i as f64 - j as f64).abs()) + if i == j { n as f64 } else { 0.0 };
                a[i + j * n] = T::from_f64(v).unwrap();
            }
        }
        for back in [LEAF_BLOCK, FORWARD_BLOCK] {
            let mut leaf = DenseLeaf::<T>::new(n, back);
            let mut count = 0;
            leaf.factor_signed(&a, &[1], None, &mut count, false)
                .unwrap();
            let rhs: Vec<T> = (0..n)
                .map(|i| T::from_f64((i % 7) as f64 - 3.0).unwrap())
                .collect();
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(4)
                .build()
                .unwrap();
            // Forward: bitwise identical.
            let (mut s, mut b) = (rhs.clone(), rhs.clone());
            leaf.forward(&mut s);
            pool.install(|| leaf.forward_blocked(&mut b, LEAF_BLOCK));
            assert_eq!(s, b);
            // Backward: the shared block order, bitwise identical, and equal to
            // the batched sweep column by column.
            let (mut s, mut b) = (rhs.clone(), rhs.clone());
            leaf.backward(&mut s);
            pool.install(|| leaf.backward_blocked(&mut b));
            assert_eq!(s, b);
            let mut many: Vec<T> = rhs.iter().flat_map(|&v| [v, -v]).collect();
            leaf.backward_many(&mut many, 2);
            for (i, &v) in s.iter().enumerate() {
                assert_eq!(many[2 * i], v);
                assert_eq!(many[2 * i + 1], -v);
            }
        }
    }
    use sdpx_arithmetic::MpFloat;

    // The pooled solve keeps the serial forward order and reorders only the
    // backward sums by row block: equal to working precision.
    fn pooled_solve_parity<T: FloatT>() {
        let n = 300;
        let mut leaf = DenseLeaf::<T>::new(n, LEAF_BLOCK);
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
        let scale = serial
            .iter()
            .map(|v| v.to_f64().unwrap().abs())
            .fold(1.0, f64::max);
        let eps = T::epsilon().to_f64().unwrap();
        for (x, y) in serial.iter().zip(&pooled) {
            let d = (*x - *y).to_f64().unwrap().abs();
            assert!(d <= 1e4 * eps * scale, "{d} vs scale {scale}");
        }
    }
    fn chunked_trunk_parity<T: FloatT>() {
        let (n, cols) = (70, 5);
        let mut leaf = DenseLeaf::<T>::new(n, LEAF_BLOCK);
        for j in 0..n {
            for i in j + 1..n {
                leaf.l[i + j * n] =
                    T::from_usize((i * 7 + j * 13) % 17 + 1).unwrap() / T::from_usize(97).unwrap();
            }
            leaf.dinv[j] = T::from_usize(j % 5 + 2).unwrap().recip();
        }
        let rhs: Vec<T> = (0..n * cols)
            .map(|i| T::from_usize(i % 11 + 1).unwrap() / T::from_usize(3).unwrap())
            .collect();
        let mut serial = rhs.clone();
        leaf.solve_many(&mut serial, cols);
        for workers in [2, 4, 8] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .build()
                .unwrap();
            let mut pooled = rhs.clone();
            let chunks = workers.min(cols);
            pool.install(|| leaf.chunked(&mut pooled, cols, chunks, DenseLeaf::solve_many));
            assert_eq!(serial, pooled);
        }
    }
    /// One wave: 3 equal leaves on 16 workers take 16 chunks (not 18),
    /// larger leaves take more, and the column cap and fewer workers than
    /// leaves keep the old counts.
    #[test]
    fn one_wave_chunks_plan() {
        assert_eq!(one_wave_chunks(&[1.0; 3], 16, 42), vec![6, 5, 5]);
        assert_eq!(one_wave_chunks(&[1.0; 2], 16, 42), vec![8, 8]);
        assert_eq!(one_wave_chunks(&[4.0, 1.0], 10, 42), vec![8, 2]);
        assert_eq!(one_wave_chunks(&[1.0; 4], 3, 42), vec![1; 4]);
        assert_eq!(one_wave_chunks(&[1.0], 64, 42), vec![42]);
        assert_eq!(one_wave_chunks(&[1.0; 2], 16, 3), vec![3, 3]);
    }
    #[test]
    fn chunked_trunk_parity_f64() {
        chunked_trunk_parity::<f64>();
    }
    #[test]
    fn chunked_trunk_parity_mpfr256() {
        chunked_trunk_parity::<sdpx_arithmetic::Bits256>();
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
    /// The recursive wide leaf factor matches `factor_dots` to rounding, keeps
    /// the row envelope (a cut and ragged rows past it), and gives the same
    /// bits serially and on a pool.
    fn recursive_leaf_factor<T: FloatT>() {
        let n = 300;
        let signs: Vec<i8> = (0..n).map(|k| if k % 5 == 4 { -1 } else { 1 }).collect();
        let first = |i: usize| match i {
            0..110 => 0,
            110..230 => 110,
            _ => 110 + i % 13,
        };
        let mut a = vec![T::zero(); n * n];
        for j in 0..n {
            for i in (j..n).filter(|&i| j >= first(i)) {
                let v = if i == j {
                    f64::from(signs[i])
                        * if i % 37 == 3 {
                            1e-40
                        } else {
                            n as f64 + (i % 7) as f64
                        }
                } else {
                    ((i * 29 + j * 13) % 17) as f64 / 17.0 - 0.5
                };
                a[i + j * n] = T::from_f64(v).unwrap();
            }
        }
        let reg = Some((T::from_f64(1e-20).unwrap(), T::one()));
        let (mut c0, mut c1, mut c2) = (0, 0, 0);
        let mut dots = DenseLeaf::<T>::new(n, LEAF_BLOCK);
        dots.envelope(&a);
        dots.factor_dots(&a, &signs, reg, &mut c0, false).unwrap();
        let mut serial = DenseLeaf::<T>::new(n, LEAF_BLOCK);
        serial.factor_leaf(&a, &signs, reg, &mut c1, false).unwrap();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(8)
            .build()
            .unwrap();
        let mut split = DenseLeaf::<T>::new(n, LEAF_BLOCK);
        pool.install(|| split.factor_leaf(&a, &signs, reg, &mut c2, true))
            .unwrap();
        assert!(c0 > 0);
        assert_eq!((c0, c0), (c1, c2));
        assert_eq!(serial.l, split.l);
        assert_eq!(serial.dinv, split.dinv);
        assert_ne!(serial.l, dots.l);
        let close = |x: T, y: T| {
            (x - y).abs().to_f64().unwrap() <= 1e-25 * y.abs().to_f64().unwrap().max(1.0)
        };
        for j in 0..n {
            assert!(close(serial.dinv[j], dots.dinv[j]));
            for i in j + 1..n {
                let v = serial.l[i + j * n];
                assert!(j >= first(i) || v.is_zero());
                assert!(close(v, dots.l[i + j * n]), "({i}, {j})");
            }
        }
    }
    #[test]
    fn recursive_leaf_factor_mpfr128() {
        recursive_leaf_factor::<sdpx_arithmetic::Bits128>();
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
        let mut serial = DenseLeaf::<T>::new(n, LEAF_BLOCK);
        let mut count = 0;
        serial.factor(&a, 1, None, &mut count, false).unwrap();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(8)
            .build()
            .unwrap();
        let mut split = DenseLeaf::<T>::new(n, LEAF_BLOCK);
        pool.install(|| split.factor(&a, 1, None, &mut count, true))
            .unwrap();
        assert_eq!(serial.l, split.l);
        assert_eq!(serial.dinv, split.dinv);
        // Panel edges off the panel width, mixed signs and clamped pivots.
        let n = 83;
        let signs: Vec<i8> = (0..n).map(|k| if k % 5 == 4 { -1 } else { 1 }).collect();
        let mut a = vec![T::zero(); n * n];
        for j in 0..n {
            for i in j..n {
                let v = if i == j {
                    f64::from(signs[i])
                        * if i % 11 == 3 {
                            1e-40
                        } else {
                            n as f64 + (i % 7) as f64
                        }
                } else {
                    ((i * 29 + j * 13) % 17) as f64 / 17.0 - 0.5
                };
                a[i + j * n] = T::from_f64(v).unwrap();
            }
        }
        let reg = Some((T::from_f64(1e-20).unwrap(), T::one()));
        let (mut c1, mut c2) = (0, 0);
        let mut serial = DenseLeaf::<T>::new(n, LEAF_BLOCK);
        serial
            .factor_signed(&a, &signs, reg, &mut c1, false)
            .unwrap();
        let mut split = DenseLeaf::<T>::new(n, LEAF_BLOCK);
        pool.install(|| split.factor_signed(&a, &signs, reg, &mut c2, true))
            .unwrap();
        assert!(c1 > 0);
        assert_eq!(c1, c2);
        assert_eq!(serial.l, split.l);
        assert_eq!(serial.dinv, split.dinv);
        // The leaf form (exact dots above 64 bits), with a row envelope.
        for i in 40..n {
            for j in 0..i % 9 {
                a[i + j * n] = T::zero();
            }
        }
        let (mut c1, mut c2) = (0, 0);
        serial.factor_leaf(&a, &signs, reg, &mut c1, false).unwrap();
        pool.install(|| split.factor_leaf(&a, &signs, reg, &mut c2, true))
            .unwrap();
        assert!(c1 > 0);
        assert_eq!(c1, c2);
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

    /// Trunk-free arrow: the coupling panel matches the lane panel to working
    /// accuracy and is bitwise the same for every pool (f64: the lane panel).
    fn coupling_panel<T: FloatT>(cols: usize) {
        let sizes = [30usize, 41, 25];
        let n: usize = sizes.iter().sum();
        let (mut colptr, mut rowval, mut nzval) = (vec![0usize], Vec::new(), Vec::new());
        let mut start = 0;
        for &g in &sizes {
            for j in start..start + g {
                for i in start..=j {
                    rowval.push(i);
                    nzval.push(if i == j {
                        T::from_f64(2.0 * g as f64 + (i % 5) as f64).unwrap()
                    } else {
                        T::from_f64(((i * 7 + j * 3) % 11) as f64 / 11.0 - 0.45).unwrap()
                    });
                }
                colptr.push(rowval.len());
            }
            start += g;
        }
        let k = CscMatrix::new(n, n, colptr, rowval, nzval);
        let signs = vec![1i8; n];
        let mut solver =
            ArrowLDLSolver::try_new(&k, &signs, &CoreSettings::<T>::default()).unwrap();
        assert!(solver.refactor(&k) && solver.use_arrow && solver.trunk.is_empty());
        let b: Vec<T> = (0..n * cols)
            .map(|e| T::from_f64(((e * 13) % 17) as f64 / 17.0 - 0.3).unwrap())
            .collect();
        let mut lanes = vec![T::zero(); b.len()];
        solver.solve_many(&k, &mut lanes, &mut b.clone(), cols);
        let mut serial = vec![T::zero(); b.len()];
        solver.solve_coupling_panel(&k, &mut serial, &mut b.clone(), cols);
        let exact = T::precision_bits() <= 64;
        let tol = T::from_f64(if exact { 0.0 } else { 1e-60 }).unwrap();
        for (a, l) in serial.iter().zip(&lanes) {
            assert!((*a - *l).abs() <= tol);
        }
        for threads in [2, 4] {
            solver.set_pool(Some(Arc::new(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap(),
            )));
            assert!(solver.refactor(&k));
            let mut pooled = vec![T::zero(); b.len()];
            solver.solve_coupling_panel(&k, &mut pooled, &mut b.clone(), cols);
            assert_eq!(pooled, serial);
        }
    }
    #[test]
    fn arrow_coupling_panel() {
        for cols in [1, 7, 26] {
            coupling_panel::<f64>(cols);
            coupling_panel::<sdpx_arithmetic::Bits256>(cols);
            coupling_panel::<sdpx_arithmetic::Bits512>(cols);
        }
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
