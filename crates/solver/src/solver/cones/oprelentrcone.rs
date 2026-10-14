#![allow(non_snake_case)]
//! Operator relative entropy cone
//!
//! K = cl{(T, X, Y) : X, Y ≻ 0, T ⪰ D(X, Y)},  D(X, Y) = X^½ g(X^-½ Y X^-½) X^½,
//! g = −log, for n×n symmetric T, X, Y stacked in SDPX's scaled triangle
//! form (the PSD cone's svec). D(X, Y) = −X^½ log(X^-½ Y X^-½) X^½ is picos'
//! `oprelentr(X, Y)` and QICS' `OpPerspecEpi(n, "log")`.
//!
//! Barrier f = −logdet(T − D) − logdet X − logdet Y, ν = 3n. Only the primal
//! barrier is known, so the cone uses it as the scaling: Hs = (μ∇²f(s))⁻¹ and
//! the centrality condition z = −μ∇f(s) linearizes to
//! Δs + HsΔz = −(Hs z − σs) for the target σμ (`combined_ds_shift`).
//!
//! Everything is evaluated in the generalized eigenbasis of (X, Y):
//! X = LLᵀ, L⁻¹YL⁻ᵀ = UΛUᵀ, Q = L⁻ᵀU (QᵀXQ = I, QᵀYQ = Λ), P = XQ = LU = Q⁻ᵀ.
//! Primal directions map as M ↦ QᵀMQ, covectors as M ↦ PᵀMP and back as
//! M ↦ QMQᵀ. There Ẑ = QᵀTQ − g(Λ) = Qᵀ(T − D)Q and W = Ẑ⁻¹, the derivative
//! of D is entrywise, DD[H, K]ᵢⱼ = cᵢⱼHᵢⱼ + g⁽¹⁾ᵢⱼKᵢⱼ, and
//! ∇²f = Lᵀ·diag(W⊗W, S₀)·L with L = [I −DD; 0 I] and S₀ the (X, Y) Schur
//! block (`schur_apply`). So (∇²f)⁻¹ needs only S₀⁻¹ (size 2m, m = n(n+1)/2)
//! and the explicit Ẑ⊗Ẑ, which keeps W's conditioning from being squared.
//!
//! Dual iterates are kept in K* by a Dikin test at the trial primal point:
//! z ∈ int K* when ⟨z, s⟩² > (ν − 1)⟨z, (∇²f)⁻¹z⟩. Bounding ∇²f below by
//! dropping S₀'s (positive semidefinite) curvature of D gives an upper bound
//! of the quadratic form in O(n³) (`dual_form`); it is exact on the central
//! path, where z ∝ −∇f(s).

use super::*;
use crate::algebra::*;
use rayon::prelude::*;

/// Tasks for `mul_adds` exact multiply-adds over `items` independent
/// pieces: more than one only on a solver-pool lane with spare workers (the
/// inner-parallel gate). Entries stay one dot each, so results do not depend
/// on the split.
fn tasks<T: FloatT>(mul_adds: usize, items: usize) -> usize {
    use sdpx_arithmetic::inner_parallel as ip;
    ip::tasks(mul_adds as u128 * ip::weight(T::precision_bits() as usize), items)
}

/// `f(k, piece)` over the `len`-element pieces of `out` (the last may be
/// shorter) when [`tasks`] grants a split of `mul_adds`: posted on the
/// scaling queue's board, whose idle workers only help posts, else on the
/// pool. Who runs a piece never changes it.
fn pieces<T: FloatT>(
    out: &mut [T],
    len: usize,
    mul_adds: usize,
    f: impl Fn(usize, &mut [T]) + Sync,
) {
    use sdpx_arithmetic::inner_parallel::board;
    let items = out.len().div_ceil(len);
    if tasks::<T>(mul_adds, items) <= 1 {
        out.chunks_mut(len).enumerate().for_each(|(k, p)| f(k, p));
    } else if board::offered() {
        struct Out<T>(*mut T);
        // SAFETY: items write disjoint pieces of `out`.
        unsafe impl<T> Sync for Out<T> {}
        let (base, total) = (Out(out.as_mut_ptr()), out.len());
        board::run(items, &|k| {
            let (start, base) = (k * len, &base);
            let end = (start + len).min(total);
            // SAFETY: piece k is [start, end), owned by item k alone.
            f(k, unsafe { std::slice::from_raw_parts_mut(base.0.add(start), end - start) });
        });
    } else {
        out.par_chunks_mut(len).enumerate().for_each(|(k, p)| f(k, p));
    }
}

/// Scalar central point (t, x, y) of the relative entropy cone, s = −∇f(s);
/// refined to the working precision by `central_point`.
const CENTRAL: [f64; 3] = [0.8278383990656786, 0.8051020015847954, 1.290927709856958];

/// Order up to which the recursive wide kernels take exact dots.
const BASE: usize = 128;

/// `c = op(a)·op(b)` (`m × n`, inner `k`): one exact sum rounded once per
/// entry, with the spare workers [`tasks`] grants offered to the residue
/// product.
fn product<T: FloatT>(
    (ta, tb): (u8, u8),
    (m, n, k): (usize, usize, usize),
    (a, lda): (&[T], usize),
    (b, ldb): (&[T], usize),
    c: &mut [T],
) {
    let ways = tasks::<T>(m * n * k, usize::MAX);
    crate::algebra::with_split_hint(ways, || {
        let (mi, ni, ki) = (m as i32, n as i32, k as i32);
        T::xgemm(ta, tb, mi, ni, ki, T::one(), a, lda as i32, b, ldb as i32, T::zero(), c, mi)
    });
}

#[inline]
fn tri(i: usize, j: usize) -> usize {
    // svec index of (i, j), i ≤ j (upper triangle by column)
    j * (j + 1) / 2 + i
}

fn gemm<T: FloatT>(ta: bool, tb: bool, n: usize, a: &[T], b: &[T], c: &mut [T]) {
    let ni = n as i32;
    let t = |x: bool| if x { b'T' } else { b'N' };
    T::xgemm(t(ta), t(tb), ni, ni, ni, T::one(), a, ni, b, ni, T::zero(), c, ni);
}

/// Lower Cholesky factor in place (upper triangle zeroed); false unless ≻ 0.
/// Wide types go left-looking on rows, each entry one exact dot product
/// rounded once (3× faster than potrf's FMA chains at order 72); binary64
/// keeps LAPACK.
fn cholesky<T: FloatT>(n: usize, a: &mut [T]) -> bool {
    if T::precision_bits() <= 53 {
        let mut info = 0;
        T::xpotrf(b'L', n as i32, a, n as i32, &mut info);
        if info != 0 {
            return false;
        }
        for j in 0..n {
            for i in 0..j {
                a[i + j * n] = T::zero();
            }
        }
        return (0..n).all(|i| a[i + i * n].is_finite() && a[i + i * n] > T::zero());
    }
    if n > BASE {
        if !cholesky_rec(n, a, n) {
            return false;
        }
        for j in 1..n {
            a[j * n..j * n + j].fill(T::zero());
        }
        return true;
    }
    cholesky_dots(n, a)
}

/// Wide [`cholesky`] of order `n`, left-looking on rows: each entry one
/// exact dot product rounded once (3× faster than potrf's FMA chains at
/// order 72).
fn cholesky_dots<T: FloatT>(n: usize, a: &mut [T]) -> bool {
    use sdpx_arithmetic::inner_parallel as ip;
    let mut r = vec![T::zero(); n * n];
    let mut col = Vec::with_capacity(n);
    // rows per piece: one grain of dot work
    let grain = ip::grain() / ip::weight(T::precision_bits() as usize);
    for j in 0..n {
        let rj = j * n;
        let d = a[j + j * n] - T::dot_slices(&r[rj..rj + j], &r[rj..rj + j]);
        if !(d.is_finite() && d > T::zero()) {
            return false;
        }
        let d = d.sqrt();
        r[rj + j] = d;
        let rr = &r;
        let entry =
            |i: usize| (a[i + j * n] - T::dot_slices(&rr[i * n..i * n + j], &rr[rj..rj + j])) / d;
        let rows = n - j - 1;
        let tile = (grain / j.max(1) as u128).max(1) as usize;
        col.resize(rows, T::zero());
        pieces::<T>(&mut col, tile, rows * j, |k, p| {
            for (q, v) in p.iter_mut().enumerate() {
                *v = entry(j + 1 + k * tile + q);
            }
        });
        for (i, &v) in (j + 1..n).zip(&col) {
            r[i * n + j] = v;
        }
    }
    for j in 0..n {
        for i in 0..n {
            a[i + j * n] = if i < j { T::zero() } else { r[j + i * n] };
        }
    }
    true
}

/// Lower Cholesky factor of the order-`n` block at `a` (leading dimension
/// `ld`) in place, upper triangle untouched. Halving recursively makes the
/// panel solve and the Schur update large exact products (each entry the
/// exact sum rounded once, then subtracted), so residue operands are encoded
/// once per large product; blocks up to [`BASE`] take [`cholesky_dots`].
fn cholesky_rec<T: FloatT>(n: usize, a: &mut [T], ld: usize) -> bool {
    if n <= BASE {
        let mut b = vec![T::zero(); n * n];
        for j in 0..n {
            b[j + j * n..(j + 1) * n].copy_from_slice(&a[j + j * ld..n + j * ld]);
        }
        if !cholesky_dots(n, &mut b) {
            return false;
        }
        for j in 0..n {
            a[j + j * ld..n + j * ld].copy_from_slice(&b[j + j * n..(j + 1) * n]);
        }
        return true;
    }
    let (n1, n2) = (n / 2, n - n / 2);
    if !cholesky_rec(n1, a, ld) {
        return false;
    }
    let l11: Vec<T> = (0..n1 * n1).map(|e| a[e % n1 + e / n1 * ld]).collect();
    solve_lower_t(n2, n1, &l11, n1, &mut a[n1..], ld);
    let (left, right) = a.split_at_mut(n1 * ld);
    let l21 = &left[n1..];
    // packed (j, i), j ≤ i, is the Schur term (i, j)
    let mut packed = vec![T::zero(); n2 * (n2 + 1) / 2];
    let ways = tasks::<T>(n2 * n2 * n1 / 2, usize::MAX);
    let upper = crate::algebra::with_split_hint(ways, || {
        T::xgemm_upper_exact(b'N', b'T', n2, n2, n1, l21, ld, l21, ld, &mut packed, None, None)
    });
    if upper {
        for j in 0..n2 {
            for i in j..n2 {
                right[n1 + i + j * ld] -= packed[i * (i + 1) / 2 + j];
            }
        }
    } else {
        let mut s = vec![T::zero(); n2 * n2];
        product((b'N', b'T'), (n2, n2, n1), (l21, ld), (l21, ld), &mut s);
        for j in 0..n2 {
            for i in j..n2 {
                right[n1 + i + j * ld] -= s[i + j * n2];
            }
        }
    }
    cholesky_rec(n2, &mut right[n1..], ld)
}

/// X·Lᵀ = B for X (`m × n`, overwriting B with leading dimension `ldb`),
/// L lower of order `n`: recursive halves of L with one exact product
/// update between them; up to [`BASE`] columns, rows of B solve
/// independently by exact dots.
fn solve_lower_t<T: FloatT>(m: usize, n: usize, l: &[T], ldl: usize, b: &mut [T], ldb: usize) {
    if n <= BASE {
        let lr: Vec<T> = (0..n * n).map(|e| l[e / n + e % n * ldl]).collect();
        let mut rows: Vec<T> = (0..m * n).map(|e| b[e / n + e % n * ldb]).collect();
        pieces::<T>(&mut rows, n, m * n * n / 2, |_, x| {
            for j in 0..n {
                let v = x[j] - T::dot_slices(&x[..j], &lr[j * n..j * n + j]);
                x[j] = v / lr[j + j * n];
            }
        });
        for (e, v) in rows.into_iter().enumerate() {
            b[e / n + e % n * ldb] = v;
        }
        return;
    }
    let (n1, n2) = (n / 2, n - n / 2);
    solve_lower_t(m, n1, l, ldl, b, ldb);
    let (b1, b2) = b.split_at_mut(n1 * ldb);
    let mut p = vec![T::zero(); m * n2];
    product((b'N', b'T'), (m, n2, n1), (b1, ldb), (&l[n1..], ldl), &mut p);
    for j in 0..n2 {
        for i in 0..m {
            b2[i + j * ldb] -= p[i + j * m];
        }
    }
    solve_lower_t(m, n2, &l[n1 + n1 * ldl..], ldl, b2, ldb);
}

/// L⁻¹ of a lower triangular L by forward substitution (lower triangle),
/// each entry one exact dot product; binary64 solves against the identity
/// with TRSM.
fn lower_inverse<T: FloatT>(n: usize, l: &[T], li: &mut [T]) {
    if T::precision_bits() <= 53 {
        li.fill(T::zero());
        (0..n).for_each(|i| li[i + i * n] = T::one());
        T::xtrsm_lower(n, l, li, false);
        return;
    }
    if n > BASE {
        return lower_inverse_rec(n, l, n, li, n);
    }
    lower_inverse_dots(n, l, li);
}

fn lower_inverse_dots<T: FloatT>(n: usize, l: &[T], li: &mut [T]) {
    let mut rows = vec![T::zero(); n * n];
    for j in 0..n {
        for i in j..n {
            rows[j + i * n] = l[i + j * n];
        }
    }
    // columns are independent
    pieces::<T>(li, n, n * n * n / 6, |col, lc| {
        lc[col] = T::one() / l[col + col * n];
        for i in col + 1..n {
            let v = T::dot_slices(&rows[i * n + col..i * n + i], &lc[col..i]);
            lc[i] = -v / l[i + i * n];
        }
    });
}

/// L⁻¹ (`x`, leading dimension `ldx`, strict upper triangle left zero) of
/// the order-`n` lower L at `l`: X21 = −X22·(L21·X11) from the inverses of
/// the halves, two exact products per level; up to [`BASE`],
/// [`lower_inverse_dots`].
fn lower_inverse_rec<T: FloatT>(n: usize, l: &[T], ldl: usize, x: &mut [T], ldx: usize) {
    if n <= BASE {
        let lb: Vec<T> = (0..n * n).map(|e| l[e % n + e / n * ldl]).collect();
        let mut xb = vec![T::zero(); n * n];
        lower_inverse_dots(n, &lb, &mut xb);
        for j in 0..n {
            x[j + j * ldx..n + j * ldx].copy_from_slice(&xb[j + j * n..(j + 1) * n]);
        }
        return;
    }
    let (n1, n2) = (n / 2, n - n / 2);
    lower_inverse_rec(n1, l, ldl, x, ldx);
    lower_inverse_rec(n2, &l[n1 + n1 * ldl..], ldl, &mut x[n1 + n1 * ldx..], ldx);
    let mut t = vec![T::zero(); n2 * n1];
    product((b'N', b'N'), (n2, n1, n1), (&l[n1..], ldl), (x, ldx), &mut t);
    let mut p = vec![T::zero(); n2 * n1];
    product((b'N', b'N'), (n2, n1, n2), (&x[n1 + n1 * ldx..], ldx), (&t, n2), &mut p);
    for j in 0..n1 {
        for i in 0..n2 {
            x[n1 + i + j * ldx] = -p[i + j * n2];
        }
    }
}

/// Inverse of a symmetric positive definite matrix from its lower factor:
/// L⁻ᵀL⁻¹ (LAPACK's potri: a third of the products of solving against the
/// identity), each entry one exact dot product of two columns of L⁻¹.
fn spd_inverse<T: FloatT>(n: usize, l: &[T], out: &mut [T]) -> bool {
    let mut li = vec![T::zero(); n * n];
    lower_inverse(n, l, &mut li);
    if T::precision_bits() <= 53 {
        let ni = n as i32;
        T::xsyrk(b'L', b'T', ni, ni, T::one(), &li, ni, T::zero(), out, ni);
        for j in 0..n {
            for i in j + 1..n {
                out[j + i * n] = out[i + j * n];
            }
        }
        return out.iter().all(|v| v.is_finite());
    }
    // One exact symmetric product (residue operands encoded once), else
    // exact dots: either way each entry is one exact sum rounded once.
    let mut packed = vec![T::zero(); n * (n + 1) / 2];
    let ways = tasks::<T>(n * n * n / 2, usize::MAX);
    let exact = crate::algebra::with_split_hint(ways, || {
        T::xgemm_upper_exact(b'T', b'N', n, n, n, &li, n, &li, n, &mut packed, None, None)
    });
    if exact {
        for j in 0..n {
            for i in 0..=j {
                let v = packed[j * (j + 1) / 2 + i];
                out[i + j * n] = v;
                out[j + i * n] = v;
            }
        }
        return out.iter().all(|v| v.is_finite());
    }
    pieces::<T>(out, n, n * n * n / 6, |j, oc| {
        for i in j..n {
            oc[i] = T::dot_slices(&li[i + i * n..(i + 1) * n], &li[i + j * n..(j + 1) * n]);
        }
    });
    for j in 0..n {
        for i in j + 1..n {
            out[j + i * n] = out[i + j * n];
        }
    }
    out.iter().all(|v| v.is_finite())
}

fn symmetrize<T: FloatT>(n: usize, a: &mut [T]) {
    let half: T = (0.5).as_T();
    for j in 0..n {
        for i in 0..j {
            let v = (a[i + j * n] + a[j + i * n]) * half;
            a[i + j * n] = v;
            a[j + i * n] = v;
        }
    }
}

/// Symmetric eigendecomposition (ascending), vectors in `u`.
fn sym_eig<T: FloatT>(n: usize, a: &mut [T], w: &mut [T], u: &mut [T]) -> bool {
    let ni = n as i32;
    let (mut m, mut info) = (0, 0);
    let mut isuppz = vec![0i32; 2 * n.max(1)];
    let mut work = vec![T::zero(); 1];
    let mut iwork = vec![0i32; 1];
    let (zero, tol) = (T::zero(), -T::one());
    T::xsyevr(
        b'V', b'A', b'U', ni, a, ni, zero, zero, 0, 0, tol, &mut m, w, u, ni, &mut isuppz,
        &mut work, -1, &mut iwork, -1, &mut info,
    );
    if info != 0 {
        return false;
    }
    let lwork = work[0].to_usize().unwrap_or(1).max(1);
    let liwork = iwork[0].max(1) as usize;
    work.resize(lwork, T::zero());
    iwork.resize(liwork, 0);
    T::xsyevr(
        b'V', b'A', b'U', ni, a, ni, zero, zero, 0, 0, tol, &mut m, w, u, ni, &mut isuppz,
        &mut work, lwork as i32, &mut iwork, liwork as i32, &mut info,
    );
    info == 0 && m == ni
}

fn smat<T: FloatT>(n: usize, x: &[T], out: &mut [T]) {
    let r = T::FRAC_1_SQRT_2();
    for j in 0..n {
        for i in 0..j {
            let v = x[tri(i, j)] * r;
            out[i + j * n] = v;
            out[j + i * n] = v;
        }
        out[j + j * n] = x[tri(j, j)];
    }
}

fn svec<T: FloatT>(n: usize, a: &[T], out: &mut [T]) {
    let r = T::FRAC_1_SQRT_2();
    for j in 0..n {
        for i in 0..j {
            out[tri(i, j)] = (a[i + j * n] + a[j + i * n]) * r;
        }
        out[tri(j, j)] = a[j + j * n];
    }
}

/// `out = Bᵀ·M·B` (`left_t`) or `B·M·Bᵀ`, through `tmp`.
fn congruence<T: FloatT>(n: usize, b: &[T], m: &[T], left_t: bool, tmp: &mut [T], out: &mut [T]) {
    if left_t {
        gemm(false, false, n, m, b, tmp); // M B
        gemm(true, false, n, b, tmp, out); // Bᵀ M B
    } else {
        gemm(false, true, n, m, b, tmp); // M Bᵀ
        gemm(false, false, n, b, tmp, out); // B M Bᵀ
    }
    symmetrize(n, out);
}

/// `out = X·S` (`right`) or `S·X` for a symmetric S whose nonzeros lie at
/// (p, q) and (q, p): one product per output entry, the dense product's
/// value (its other terms are exact zeros).
fn mul_unit<T: FloatT>(n: usize, x: &[T], s: &[T], (p, q): (usize, usize), right: bool, out: &mut [T]) {
    out.fill(T::zero());
    for (r, c) in [(p, q), (q, p)].into_iter().take(if p == q { 1 } else { 2 }) {
        let v = s[r + c * n];
        if v == T::zero() {
            continue;
        }
        for a in 0..n {
            if right {
                out[a + c * n] = x[a + r * n] * v; // (XS)[a, c] = X[a, r]·S[r, c]
            } else {
                out[r + a * n] = v * x[c + a * n]; // (SX)[r, a] = S[r, c]·X[c, a]
            }
        }
    }
}

/// Sum of rᵏ/(2k+1), k ≥ 0, i.e. atanh(r)/r, for |r| ≤ 1/8.
fn atanh_ratio<T: FloatT>(r: T) -> T {
    let r2 = r * r;
    let (mut term, mut sum) = (T::one(), T::one());
    let eps = T::epsilon();
    let mut k = 1usize;
    loop {
        term *= r2;
        let t = term / (2 * k + 1).as_T();
        sum += t;
        if t <= eps * sum {
            return sum;
        }
        k += 1;
    }
}

/// First divided difference of g = −log at (a, b), with ln a, ln b given.
fn divided1<T: FloatT>(a: T, b: T, la: T, lb: T) -> T {
    let s = a + b;
    let r = (a - b) / s;
    if r.abs() <= (0.125).as_T() {
        // ln(a/b) = 2·atanh(r), a − b = r·s: no cancellation for close a, b.
        -(T::one() + T::one()) / s * atanh_ratio(r)
    } else {
        -(la - lb) / (a - b)
    }
}

/// Second divided difference of g = −log at (a, b, c), given g⁽¹⁾ values.
fn divided2<T: FloatT>(v: [T; 3], d1: impl Fn(usize, usize) -> T) -> T {
    // Order the nodes; (lo, hi) carries the widest gap.
    let mut idx = [0usize, 1, 2];
    idx.sort_by(|&p, &q| v[p].partial_cmp(&v[q]).unwrap_or(std::cmp::Ordering::Equal));
    let (lo, mid, hi) = (idx[0], idx[1], idx[2]);
    if (v[hi] - v[lo]) > (v[hi] + v[lo]) / (16.0).as_T() {
        return (d1(mid, hi) - d1(mid, lo)) / (v[hi] - v[lo]);
    }
    // Taylor series at the mean m: Σⱼ (−1)ʲ hⱼ(u)/(j+2) / m², u = (v − m)/m,
    // hⱼ the complete homogeneous symmetric polynomial (|u| ≤ 1/8 here).
    let m = (v[0] + v[1] + v[2]) / (3.0).as_T();
    let u = [(v[0] - m) / m, (v[1] - m) / m, (v[2] - m) / m];
    let umax = u[0].abs().max(u[1].abs()).max(u[2].abs());
    let (mut e1, mut e2, mut e3) = (T::one(), T::one(), T::one());
    let mut sum: T = (0.5).as_T();
    let mut bound = T::one();
    let eps = T::epsilon();
    let mut j = 1usize;
    loop {
        e1 *= u[0];
        e2 = u[1] * e2 + e1;
        e3 = u[2] * e3 + e2;
        let t = e3 / (j + 2).as_T();
        if j % 2 == 1 {
            sum -= t;
        } else {
            sum += t;
        }
        bound *= umax;
        // |hⱼ| ≤ (j+1)(j+2)/2·umaxʲ
        if bound * ((j + 1) as f64 / 2.0).as_T() <= eps * sum.abs() || j > 4096 {
            break;
        }
        j += 1;
    }
    sum / (m * m)
}

/// Barrier oracle at an interior point, in the generalized eigenbasis.
pub(crate) struct Oracle<T> {
    n: usize,
    q: Vec<T>,
    p: Vec<T>,
    lam: Vec<T>,
    g: Vec<T>,
    /// g⁽¹⁾ (n×n) and c = (gᵢ+gⱼ)/2 − g⁽¹⁾ᵢⱼ(λᵢ+λⱼ)/2, the X-part of DD.
    d1: Vec<T>,
    c: Vec<T>,
    /// g⁽²⁾ (n×n×n), built on demand for Hessian products, with the
    /// point's constant Schur operands V = g⁽¹⁾∘W, A = ΛV + VΛ and
    /// A2 = GW + WG (n×n each).
    d2: Vec<T>,
    vaa: Vec<T>,
    zh: Vec<T>,
    lz: Vec<T>,
    w: Vec<T>,
    /// Lower Cholesky factor of S₀ (built on demand; None if S₀ is not ≻ 0).
    s0l: Option<Option<Vec<T>>>,
    /// f(s) (the tests check the gradient against it)
    #[cfg_attr(not(test), allow(dead_code))]
    pub barrier: T,
}

impl<T: FloatT> Oracle<T> {
    /// None unless s is interior.
    pub(crate) fn new(n: usize, s: &[T]) -> Option<Self> {
        let m = n * (n + 1) / 2;
        let nn = n * n;
        let (st, rest) = s.split_at(m);
        let (sx, sy) = rest.split_at(m);
        let mut lx = vec![T::zero(); nn];
        smat(n, sx, &mut lx);
        if !cholesky(n, &mut lx) {
            return None;
        }
        let mut li = vec![T::zero(); nn];
        lower_inverse(n, &lx, &mut li);
        let mut y = vec![T::zero(); nn];
        smat(n, sy, &mut y);
        let mut tmp = vec![T::zero(); nn];
        let mut psi = vec![T::zero(); nn];
        congruence(n, &li, &y, false, &mut tmp, &mut psi);
        let mut lam = vec![T::zero(); n];
        let mut u = vec![T::zero(); nn];
        if !sym_eig(n, &mut psi, &mut lam, &mut u) {
            return None;
        }
        if !lam.iter().all(|&l| l.is_finite() && l > T::zero()) {
            return None;
        }
        let mut q = vec![T::zero(); nn];
        gemm(true, false, n, &li, &u, &mut q);
        let mut p = vec![T::zero(); nn];
        gemm(false, false, n, &lx, &u, &mut p);
        let ln: Vec<T> = lam.iter().map(|&l| l.ln()).collect();
        let g: Vec<T> = ln.iter().map(|&l| -l).collect();
        let mut zh = vec![T::zero(); nn];
        let mut tm = vec![T::zero(); nn];
        smat(n, st, &mut tm);
        congruence(n, &q, &tm, true, &mut tmp, &mut zh);
        for i in 0..n {
            zh[i + i * n] -= g[i];
        }
        let mut lz = zh.clone();
        if !cholesky(n, &mut lz) {
            return None;
        }
        let mut w = vec![T::zero(); nn];
        if !spd_inverse(n, &lz, &mut w) {
            return None;
        }
        let half: T = (0.5).as_T();
        let mut d1 = vec![T::zero(); nn];
        let mut c = vec![T::zero(); nn];
        for j in 0..n {
            for i in 0..n {
                let v = if i == j {
                    -lam[i].recip()
                } else {
                    divided1(lam[i], lam[j], ln[i], ln[j])
                };
                d1[i + j * n] = v;
                c[i + j * n] = half * (g[i] + g[j]) - half * v * (lam[i] + lam[j]);
            }
        }
        let two = T::one() + T::one();
        let mut logdet_z = T::zero();
        let mut logdet_x = T::zero();
        for i in 0..n {
            logdet_z += two * lz[i + i * n].ln();
            logdet_x += two * lx[i + i * n].ln();
        }
        let three: T = (3.0).as_T();
        let barrier = -logdet_z - three * logdet_x + g.iter().fold(T::zero(), |a, &v| a + v);
        Some(Self {
            n,
            q,
            p,
            lam,
            g,
            d1,
            c,
            d2: Vec::new(),
            vaa: Vec::new(),
            s0l: None,
            zh,
            lz,
            w,
            barrier,
        })
    }

    fn ensure_d2(&mut self) {
        let n = self.n;
        if !self.d2.is_empty() {
            return;
        }
        let mut d2 = vec![T::zero(); n * n * n];
        let d1 = &self.d1;
        for a in 0..n {
            for b in a..n {
                for c in b..n {
                    let ids = [a, b, c];
                    let v = [self.lam[a], self.lam[b], self.lam[c]];
                    let x = divided2(v, |p, q| d1[ids[p] + ids[q] * n]);
                    for (i, j, k) in [(a, b, c), (a, c, b), (b, a, c), (b, c, a), (c, a, b), (c, b, a)] {
                        d2[i + n * (j + n * k)] = x;
                    }
                }
            }
        }
        self.d2 = d2;
        let nn = n * n;
        let mut vaa = vec![T::zero(); 3 * nn];
        let (v, rest) = vaa.split_at_mut(nn);
        let (a, a2) = rest.split_at_mut(nn);
        let (lam, g, w) = (&self.lam, &self.g, &self.w);
        for j in 0..n {
            for i in 0..n {
                let ij = i + j * n;
                v[ij] = self.d1[ij] * w[ij];
                a[ij] = (lam[i] + lam[j]) * v[ij];
                a2[ij] = (g[i] + g[j]) * w[ij];
            }
        }
        self.vaa = vaa;
    }

    /// Primal gradient ∇f(s) (svec, stacked T, X, Y).
    #[cfg(test)]
    pub(crate) fn gradient(&self, out: &mut [T]) {
        let n = self.n;
        let m = n * (n + 1) / 2;
        let nn = n * n;
        let mut e = vec![T::zero(); nn];
        let mut tmp = vec![T::zero(); nn];
        let mut b = vec![T::zero(); nn];
        for part in 0..3 {
            for j in 0..n {
                for i in 0..n {
                    let w = self.w[i + j * n];
                    e[i + j * n] = match part {
                        0 => -w,
                        1 => self.c[i + j * n] * w,
                        _ => self.d1[i + j * n] * w,
                    };
                }
                match part {
                    1 => e[j + j * n] -= T::one(),
                    2 => e[j + j * n] -= self.lam[j].recip(),
                    _ => {}
                }
            }
            congruence(n, &self.q, &e, false, &mut tmp, &mut b);
            svec(n, &b, &mut out[part * m..(part + 1) * m]);
        }
    }

    /// Schur block S₀[H, K] in eigen-coordinates: the (X, Y) Hessian with the
    /// coupling through T eliminated. Writes (out_h, out_k). `unit` = (p, q)
    /// when H and K vanish outside (p, q), (q, p) (S₀'s columns): every
    /// term then vanishes outside rows and columns p, q, so only those
    /// entries are formed, O(n) instead of O(n³).
    #[allow(clippy::too_many_arguments)]
    fn schur_apply(
        &self,
        h: &[T],
        k: &[T],
        out_h: &mut [T],
        out_k: &mut [T],
        ws: &mut Vec<T>,
        unit: Option<(usize, usize)>,
    ) {
        let n = self.n;
        let nn = n * n;
        ws.resize(10 * nn, T::zero());
        let (e1, rest) = ws.split_at_mut(nn);
        let (gam, rest) = rest.split_at_mut(nn);
        let (f1, rest) = rest.split_at_mut(nn);
        let (dj, rest) = rest.split_at_mut(nn);
        let (m1, rest) = rest.split_at_mut(nn);
        let (m2, rest) = rest.split_at_mut(nn);
        let (m3, rest) = rest.split_at_mut(nn);
        let (m4, rest) = rest.split_at_mut(nn);
        let (m5, rest) = rest.split_at_mut(nn);
        let (m6, _) = rest.split_at_mut(nn);
        let (lam, g, d1, w) = (&self.lam, &self.g, &self.d1, &self.w);
        let (v, rest) = self.vaa.split_at(nn);
        let (aw, a2) = rest.split_at(nn);
        // entries (i, j) that can be nonzero
        let support: Vec<(usize, usize)> = (0..n)
            .flat_map(|j| (0..n).map(move |i| (i, j)))
            .filter(|&(i, j)| unit.is_none_or(|(p, q)| i == p || i == q || j == p || j == q))
            .collect();
        let half: T = (0.5).as_T();
        for &(i, j) in &support {
            let ij = i + j * n;
            e1[ij] = k[ij] - half * (lam[i] + lam[j]) * h[ij];
            f1[ij] = d1[ij] * e1[ij];
        }
        // X·S (`right`) or S·X
        let prod = |x: &[T], s: &[T], right: bool, out: &mut [T]| match unit {
            Some(pq) => mul_unit(n, x, s, pq, right, out),
            None if right => gemm(false, false, n, x, s, out),
            None => gemm(false, false, n, s, x, out),
        };
        // Γ_ab = Σ_c g⁽²⁾[a,b,c](W_ac E1_cb + E1_ac W_cb); E1 is zero outside
        // rows and columns p, q for a unit
        let cs: Vec<usize> = match unit {
            Some((p, q)) if p == q => vec![p],
            Some((p, q)) => vec![p.min(q), p.max(q)],
            None => (0..n).collect(),
        };
        let d2 = &self.d2;
        for &(a, b) in &support {
            let mut acc = T::zero();
            for &c in &cs {
                let t = w[a + c * n] * e1[c + b * n] + e1[a + c * n] * w[c + b * n];
                acc += d2[a + n * (b + n * c)] * t;
            }
            gam[a + b * n] = acc;
        }
        prod(v, h, true, m1); // V H
        prod(w, h, true, m2); // W H
        prod(v, k, false, m3); // K V
        for &(i, j) in &support {
            let (ij, ji) = (i + j * n, j + i * n);
            dj[ij] = d1[ij] * half * (m2[ij] + m2[ji]); // g⁽¹⁾∘J, J = (WH + HW)/2
        }
        prod(aw, h, true, m4); // A H, A = ΛV + VΛ
        prod(w, f1, true, m5); // W F1
        prod(a2, h, true, m6); // A2 H, A2 = GW + WG
        let (q38, q14, q18): (T, T, T) = ((0.375).as_T(), (0.25).as_T(), (0.125).as_T());
        if unit.is_some() {
            out_h.fill(T::zero());
            out_k.fill(T::zero());
        }
        for &(i, j) in &support {
            let (ij, ji) = (i + j * n, j + i * n);
            let (li, lj, gi, gj) = (lam[i], lam[j], g[i], g[j]);
            let dhq = -half * (m3[ij] + m3[ji]) + q38 * (m4[ij] + m4[ji])
                + q14 * (lj * m1[ij] + li * m1[ji])
                - half * (li + lj) * gam[ij]
                + half * (m5[ij] + m5[ji])
                - half * (li + lj) * dj[ij]
                + q14 * (gj * m2[ij] + gi * m2[ji])
                - q18 * (m6[ij] + m6[ji]);
            let dkq = -half * (m1[ij] + m1[ji]) + gam[ij] + dj[ij];
            out_h[ij] = h[ij] + dhq;
            out_k[ij] = k[ij] / (li * lj) + dkq;
        }
        symmetrize(n, out_h);
        symmetrize(n, out_k);
    }

    /// Hessian product ∇²f(s)[d] (svec in, svec out).
    pub(crate) fn hess_prod(&mut self, d: &[T], out: &mut [T]) {
        self.ensure_d2();
        let n = self.n;
        let m = n * (n + 1) / 2;
        let nn = n * n;
        let mut mats = vec![T::zero(); 3 * nn];
        let mut tmp = vec![T::zero(); nn];
        let mut x = vec![T::zero(); nn];
        for part in 0..3 {
            smat(n, &d[part * m..(part + 1) * m], &mut x);
            congruence(n, &self.q, &x, true, &mut tmp, &mut mats[part * nn..(part + 1) * nn]);
        }
        let (s, hk) = mats.split_at(nn);
        let (h, k) = hk.split_at(nn);
        let mut oh = vec![T::zero(); nn];
        let mut ok = vec![T::zero(); nn];
        let mut ws = Vec::new();
        self.schur_apply(h, k, &mut oh, &mut ok, &mut ws, None);
        // R = W(S − DD[H,K])W; outputs (R, S₀h − DDᵀR).
        let mut dd = vec![T::zero(); nn];
        for ij in 0..nn {
            dd[ij] = s[ij] - (self.c[ij] * h[ij] + self.d1[ij] * k[ij]);
        }
        let mut r = vec![T::zero(); nn];
        congruence(n, &self.w, &dd, false, &mut tmp, &mut r);
        for ij in 0..nn {
            oh[ij] -= self.c[ij] * r[ij];
            ok[ij] -= self.d1[ij] * r[ij];
        }
        for (part, mat) in [(0usize, &r), (1, &oh), (2, &ok)] {
            congruence(n, &self.q, mat, false, &mut tmp, &mut x);
            svec(n, &x, &mut out[part * m..(part + 1) * m]);
        }
    }

    /// S₀⁻¹ (column-major, size 2m) in eigen svec coordinates.
    pub(crate) fn schur_inverse(&mut self) -> Option<Vec<T>> {
        let m2 = self.n * (self.n + 1);
        let l = self.schur_factor()?;
        let mut s0i = vec![T::zero(); m2 * m2];
        let timer = crate::receipt::start();
        let ok = spd_inverse(m2, &l, &mut s0i) && s0i.iter().all(|v| v.is_finite());
        crate::receipt::finish("ore.schur_inverse", timer);
        ok.then_some(s0i)
    }

    /// Lower Cholesky factor of S₀ (column-major, size 2m).
    fn schur_factor(&mut self) -> Option<Vec<T>> {
        self.ensure_d2();
        let n = self.n;
        let m = n * (n + 1) / 2;
        let nn = n * n;
        let m2 = 2 * m;
        let mut s0 = vec![T::zero(); m2 * m2];
        let mut h = vec![T::zero(); nn];
        let mut k = vec![T::zero(); nn];
        let mut oh = vec![T::zero(); nn];
        let mut ok = vec![T::zero(); nn];
        let mut ws = Vec::new();
        let mut e = vec![T::zero(); m];
        let pairs: Vec<(usize, usize)> = (0..n).flat_map(|j| (0..=j).map(move |i| (i, j))).collect();
        for col in 0..m2 {
            e.fill(T::zero());
            e[col % m] = T::one();
            if col < m {
                smat(n, &e, &mut h);
                k.fill(T::zero());
            } else {
                h.fill(T::zero());
                smat(n, &e, &mut k);
            }
            self.schur_apply(&h, &k, &mut oh, &mut ok, &mut ws, Some(pairs[col % m]));
            let column = &mut s0[col * m2..(col + 1) * m2];
            svec(n, &oh, &mut column[..m]);
            svec(n, &ok, &mut column[m..]);
        }
        symmetrize(m2, &mut s0);
        let timer = crate::receipt::start();
        let ok = cholesky(m2, &mut s0);
        crate::receipt::finish("ore.schur_cholesky", timer);
        ok.then_some(s0)
    }

    /// ⟨z, (∇²f)⁻¹z⟩ = t + vᵀS₀⁻¹v with t = tr(ẐR̂ẐR̂) = ‖LzᵀR̂Lz‖²,
    /// R̂ = PᵀZ_TP and v = svec(Û + c∘R̂, V̂ + g⁽¹⁾∘R̂). Returns (t, v) and
    /// the diagonal of B = diag(1, λᵢλⱼ) ⪰ S₀⁻¹ (exact on the central path).
    fn dual_parts(&self, z: &[T]) -> (T, Vec<T>, Vec<T>) {
        let n = self.n;
        let m = n * (n + 1) / 2;
        let nn = n * n;
        let mut x = vec![T::zero(); nn];
        let mut tmp = vec![T::zero(); nn];
        let mut hat = vec![T::zero(); 3 * nn];
        for part in 0..3 {
            smat(n, &z[part * m..(part + 1) * m], &mut x);
            congruence(n, &self.p, &x, true, &mut tmp, &mut hat[part * nn..(part + 1) * nn]);
        }
        let (r, rest) = hat.split_at_mut(nn);
        let (u, v) = rest.split_at_mut(nn);
        congruence(n, &self.lz, r, true, &mut tmp, &mut x);
        let t = x.iter().fold(T::zero(), |a, &t| a + t * t);
        for ij in 0..nn {
            u[ij] += self.c[ij] * r[ij];
            v[ij] += self.d1[ij] * r[ij];
        }
        let mut w = vec![T::zero(); 2 * m];
        svec(n, u, &mut w[..m]);
        svec(n, v, &mut w[m..]);
        let mut b = vec![T::one(); 2 * m];
        for j in 0..n {
            for i in 0..=j {
                b[m + tri(i, j)] = self.lam[i] * self.lam[j];
            }
        }
        (t, w, b)
    }

    /// ⟨z, (∇²f)⁻¹z⟩ (`exact`, through S₀'s factor) or its upper bound with
    /// B in place of S₀⁻¹ (`dual_parts`).
    pub(crate) fn dual_form(&mut self, z: &[T], exact: bool) -> Option<T> {
        let (t, w, b) = self.dual_parts(z);
        if exact {
            Some(t + self.schur_norm(w)?)
        } else {
            Some(w.iter().zip(&b).fold(t, |a, (&w, &b)| a + b * w * w))
        }
    }

    /// wᵀS₀⁻¹w = ‖L₀⁻¹w‖² by forward substitution (S₀'s factor built once
    /// per oracle).
    fn schur_norm(&mut self, mut w: Vec<T>) -> Option<T> {
        if self.s0l.is_none() {
            self.s0l = Some(self.schur_factor());
        }
        let l = self.s0l.as_ref().unwrap().as_ref()?;
        let m2 = w.len();
        let mut q = T::zero();
        for j in 0..m2 {
            let wj = w[j] / l[j + j * m2];
            q += wj * wj;
            for i in j + 1..m2 {
                w[i] -= l[i + j * m2] * wj;
            }
        }
        Some(q)
    }

    /// `ok(⟨z, (∇²f)⁻¹z⟩)` for a monotone `ok` (true below a limit), decided
    /// as the exact form would be. For any x and r = v − S₀x,
    /// vᵀS₀⁻¹v = 2vᵀx − xᵀS₀x + rᵀS₀⁻¹r with 0 ≤ rᵀS₀⁻¹r ≤ rᵀBr. Conjugate
    /// gradients on S₀ preconditioned by B narrow these bounds, two O(n³)
    /// products per step (one recomputes the residual for the bounds); S₀'s O(n⁶)
    /// factor decides only what about as many products as it costs leave
    /// open. That budget is halved `cg` times: the caller counts tests CG
    /// left open since it last decided one (`cg` += 1, or 0 on a decision).
    pub(crate) fn dual_ok(&mut self, z: &[T], ok: impl Fn(T) -> bool, cg: &mut u32) -> bool {
        let (t, w, b) = self.dual_parts(z);
        let upper = w.iter().zip(&b).fold(t, |a, (&w, &b)| a + b * w * w);
        if ok(upper) {
            return true;
        }
        let n = self.n;
        let m = n * (n + 1) / 2;
        let nn = n * n;
        let m2 = w.len();
        // the factor's (2m)³/6 multiply-adds against ≈ 18n³ per CG step,
        // above order `BASE` with the factor's residue products ≈ 4.6×
        // cheaper per multiply-add (one_matrix_L12f at 128 bits, n = 42:
        // factor 1.4 s, step 8.8 ms); binary64 factors with LAPACK, far
        // cheaper per multiply-add than CG's products (one_matrix_L12: 18.6 s
        // without CG, 24.1 s with)
        let budget = if T::precision_bits() <= 53 {
            0
        } else {
            let per = if m2 > BASE { 500 } else { 108 };
            (m2.pow(3) / (per * n.pow(3))).max(1).checked_shr(*cg).unwrap_or(0)
        };
        // the exact form is t + q with q ≥ 0
        if !ok(t) {
            return false;
        }
        if budget == 0 || self.s0l.is_some() {
            return self.schur_norm(w).is_some_and(|q| ok(t + q));
        }
        self.ensure_d2();
        let mut x = vec![T::zero(); m2];
        let mut r = w.clone();
        let mut br: Vec<T> = r.iter().zip(&b).map(|(&r, &b)| r * b).collect();
        let mut p = br.clone();
        let mut rho = r.dot(&br);
        let mut sp = vec![T::zero(); m2];
        let (mut h, mut k) = (vec![T::zero(); nn], vec![T::zero(); nn]);
        let (mut oh, mut ok_) = (vec![T::zero(); nn], vec![T::zero(); nn]);
        let mut ws = Vec::new();
        for _ in 0..budget {
            smat(n, &p[..m], &mut h);
            smat(n, &p[m..], &mut k);
            self.schur_apply(&h, &k, &mut oh, &mut ok_, &mut ws, None);
            svec(n, &oh, &mut sp[..m]);
            svec(n, &ok_, &mut sp[m..]);
            let psp = p.dot(&sp);
            if !(psp > T::zero()) {
                break;
            }
            let alpha = rho / psp;
            x.axpby(alpha, &p, T::one());
            // Bounds need r = w − S₀x, not CG's recursively rounded residual.
            smat(n, &x[..m], &mut h);
            smat(n, &x[m..], &mut k);
            self.schur_apply(&h, &k, &mut oh, &mut ok_, &mut ws, None);
            svec(n, &oh, &mut sp[..m]);
            svec(n, &ok_, &mut sp[m..]);
            r.waxpby(T::one(), &w, -T::one(), &sp);
            let wx = w.dot(&x);
            let low = t + wx + wx - x.dot(&sp);
            let upper = r.iter().zip(&b).fold(low, |a, (&r, &b)| a + b * r * r);
            if ok(upper) {
                *cg = 0;
                return true;
            }
            if !ok(low) {
                *cg = 0;
                return false;
            }
            for i in 0..m2 {
                br[i] = r[i] * b[i];
            }
            let rho1 = r.dot(&br);
            let beta = rho1 / rho;
            rho = rho1;
            for i in 0..m2 {
                p[i] = br[i] + beta * p[i];
            }
        }
        *cg += 1;
        self.schur_norm(w).is_some_and(|q| ok(t + q))
    }
}

/// Largest α ≤ αmax, backtracking by `step`, with `ok(α)`; 0 below αmin.
fn search<T: FloatT>(αmax: T, αmin: T, step: T, mut ok: impl FnMut(T) -> bool) -> T {
    let mut α = αmax;
    while α > αmin && !ok(α) {
        α *= step;
    }
    if α > αmin {
        α
    } else {
        T::zero()
    }
}

/// Scalar central point refined by Newton's method at the working precision.
fn central_point<T: FloatT>() -> [T; 3] {
    let mut p: [T; 3] = [CENTRAL[0].as_T(), CENTRAL[1].as_T(), CENTRAL[2].as_T()];
    for _ in 0..12 {
        let [t, x, y] = p;
        let r = (x / y).ln();
        let zeta = t - x * r;
        let one = T::one();
        let f = [
            t - zeta.recip(),
            x - x.recip() + (r + one) / zeta,
            y - y.recip() - x / (y * zeta),
        ];
        // dζ = dt − (r+1)dx + (x/y)dy, dr = dx/x − dy/y
        let dz = [one, -(r + one), x / y];
        let dr = [T::zero(), x.recip(), -y.recip()];
        let z2 = zeta * zeta;
        let mut jac = [[T::zero(); 3]; 3];
        for c in 0..3 {
            let id = |row: usize| if row == c { one } else { T::zero() };
            jac[0][c] = id(0) + dz[c] / z2;
            jac[1][c] = id(1) * (one + (x * x).recip()) + dr[c] / zeta - (r + one) * dz[c] / z2;
            let dx = id(1);
            let dy = id(2);
            jac[2][c] = dy * (one + (y * y).recip())
                - (dx / (y * zeta) - x * dy / (y * y * zeta) - x * dz[c] / (y * z2));
        }
        let step = solve3(jac, f);
        let mut size = T::zero();
        for i in 0..3 {
            p[i] -= step[i];
            size = size.max(step[i].abs());
        }
        if size <= T::epsilon() {
            break;
        }
    }
    p
}

fn solve3<T: FloatT>(mut a: [[T; 3]; 3], mut b: [T; 3]) -> [T; 3] {
    for c in 0..3 {
        let piv = (c..3)
            .max_by(|&i, &j| a[i][c].abs().partial_cmp(&a[j][c].abs()).unwrap())
            .unwrap();
        a.swap(c, piv);
        b.swap(c, piv);
        for r in c + 1..3 {
            let f = a[r][c] / a[c][c];
            for k in c..3 {
                let v = a[c][k];
                a[r][k] -= f * v;
            }
            let v = b[c];
            b[r] -= f * v;
        }
    }
    let mut x = [T::zero(); 3];
    for r in (0..3).rev() {
        let mut v = b[r];
        for k in r + 1..3 {
            v -= a[r][k] * x[k];
        }
        x[r] = v / a[r][r];
    }
    x
}

/// The cone's change of KKT row coordinates. With 𝒫[M] = PMPᵀ,
/// (μ∇²f)⁻¹ = N·D·Nᵀ, N = 𝒫·[I DD; 0 I], D = diag(Ẑ⊗Ẑ, S₀⁻¹)/μ. Forming
/// N·D·Nᵀ explicitly adds D's T-block (O(μ²) on the boundary) to O(1)
/// terms, so it is lost once μ² falls below the rounding unit. The augmented
/// KKT instead takes the cone's rows as N⁻¹A with −D on its diagonal and
/// maps the solution back by N⁻ᵀ.
#[derive(Clone, Default)]
pub(crate) struct OpRelEntropyRows<T> {
    n: usize,
    q: Vec<T>,
    /// DD in svec form: the X (c) and Y (g⁽¹⁾) coefficients
    dc: Vec<T>,
    dk: Vec<T>,
}

impl<T: FloatT> OpRelEntropyRows<T> {
    fn parts(&self, b: &[T], v: &mut [T], left_t: bool) {
        congruence_parts(self.n, b, v, left_t);
    }

    /// v ← N⁻¹v
    pub(crate) fn to_eigen(&self, v: &mut [T]) {
        self.parts(&self.q, v, true);
        let m = self.dc.len();
        let (t, xy) = v.split_at_mut(m);
        let (x, y) = xy.split_at(m);
        for i in 0..m {
            t[i] -= self.dc[i] * x[i] + self.dk[i] * y[i];
        }
    }

    /// v ← N⁻ᵀv
    pub(crate) fn from_eigen(&self, v: &mut [T]) {
        let m = self.dc.len();
        let (t, xy) = v.split_at_mut(m);
        let (x, y) = xy.split_at_mut(m);
        for i in 0..m {
            x[i] -= self.dc[i] * t[i];
            y[i] -= self.dk[i] * t[i];
        }
        self.parts(&self.q, v, false);
    }
}

/// Each of the three svec parts of v ← BᵀMB (`left_t`) or BMBᵀ.
fn congruence_parts<T: FloatT>(n: usize, b: &[T], v: &mut [T], left_t: bool) {
    let m = n * (n + 1) / 2;
    let nn = n * n;
    let mut x = vec![T::zero(); nn];
    let mut tmp = vec![T::zero(); nn];
    let mut out = vec![T::zero(); nn];
    for part in v.chunks_exact_mut(m) {
        smat(n, part, &mut x);
        congruence(n, b, &x, left_t, &mut tmp, &mut out);
        svec(n, &out, part);
    }
}

#[derive(Clone)]
pub struct OpRelEntropyConeData<T> {
    μ: T,
    s: Vec<T>,
    z: Vec<T>,
    /// Hs = N·D·Nᵀ (`OpRelEntropyRows`): N's parts, Ẑ and S₀⁻¹
    rows: OpRelEntropyRows<T>,
    p: Vec<T>,
    zh: Vec<T>,
    s0i: Vec<T>,
    work: Vec<T>,
    /// Dual tests CG left open since it last decided one (`Oracle::dual_ok`).
    cg: u32,
}

#[derive(Clone)]
pub struct OpRelEntropyCone<T> {
    pub n: usize,
    data: Box<OpRelEntropyConeData<T>>,
}

impl<T: FloatT> OpRelEntropyCone<T> {
    pub fn new(n: usize) -> Self {
        let d = 3 * triangular_number(n);
        Self {
            n,
            data: Box::new(OpRelEntropyConeData {
                μ: T::one(),
                s: vec![T::zero(); d],
                z: vec![T::zero(); d],
                rows: OpRelEntropyRows::default(),
                p: Vec::new(),
                zh: Vec::new(),
                s0i: Vec::new(),
                work: vec![T::zero(); d],
                cg: 0,
            }),
        }
    }

    fn dim(&self) -> usize {
        3 * triangular_number(self.n)
    }

    fn nu(&self) -> T {
        (3 * self.n).as_T()
    }

    /// ∇³f(s)[h, h] by central differences of the gradient along h/‖h‖ₛ
    /// with step ε_mach^¼ (relative error ≈ ε_mach^½): the third-order
    /// adjustment of the combined step needs it only as a corrector.
    #[cfg(test)]
    fn third_order(&self, h: &[T]) -> Option<Vec<T>> {
        let s = &self.data.s;
        let mut o = Oracle::new(self.n, s)?;
        let mut hh = vec![T::zero(); h.len()];
        o.hess_prod(h, &mut hh);
        let norm2 = hh.dot(h);
        if !(norm2 > T::zero()) || !norm2.is_finite() {
            return None;
        }
        let eps = T::epsilon().sqrt().sqrt().min((0.25).as_T());
        let step = eps / norm2.sqrt();
        let mut g0 = vec![T::zero(); h.len()];
        o.gradient(&mut g0);
        let mut out = vec![T::zero(); h.len()];
        let mut w = vec![T::zero(); h.len()];
        for sign in [T::one(), -T::one()] {
            w.waxpby(T::one(), s, sign * step, h);
            let p = Oracle::new(self.n, &w)?;
            p.gradient(&mut hh);
            out.iter_mut().zip(&hh).for_each(|(o, &g)| *o += g);
        }
        // (g₊ − 2g₀ + g₋)/ε² · ‖h‖ₛ²
        let scale = norm2 / (eps * eps);
        out.iter_mut().zip(&g0).for_each(|(o, &g)| *o = (*o - g - g) * scale);
        out.iter().all(|v| v.is_finite()).then_some(out)
    }

    /// ∇³f(s)[a, b] = d/dt ∇²f(s + ta)b by central differences with
    /// ‖ta‖ₛ = ε_mach^¼ (inside the Dikin ellipsoid, so both points are
    /// interior). Ẑ = T − D(X, Y) carries absolute rounding ≈ ε_mach, so the
    /// quotient's rounding error grows as λ_min(Ẑ) shrinks: at ε_mach^⅓ it
    /// exceeded the corrector itself near the end of two_L4 at 128 bits
    /// (λ_min(Ẑ) ≤ 4e-23), while the truncation term ε_mach^½ stays small.
    fn third_mixed(&self, a: &[T], b: &[T]) -> Option<Vec<T>> {
        let s = &self.data.s;
        let mut o = Oracle::new(self.n, s)?;
        let mut ha = vec![T::zero(); a.len()];
        o.hess_prod(a, &mut ha);
        let norm2 = ha.dot(a);
        if !(norm2 > T::zero()) || !norm2.is_finite() {
            return None;
        }
        let eps = T::epsilon().sqrt().sqrt().min((0.1).as_T());
        let t = eps / norm2.sqrt();
        let mut out = vec![T::zero(); a.len()];
        let mut w = vec![T::zero(); a.len()];
        for sign in [T::one(), -T::one()] {
            w.waxpby(T::one(), s, sign * t, a);
            let mut p = Oracle::new(self.n, &w)?;
            p.hess_prod(b, &mut ha);
            out.iter_mut().zip(&ha).for_each(|(o, &h)| *o += sign * h);
        }
        let scale = (t + t).recip();
        out.iter_mut().for_each(|o| *o *= scale);
        out.iter().all(|v| v.is_finite()).then_some(out)
    }

    pub(crate) fn rows(&self) -> &OpRelEntropyRows<T> {
        &self.data.rows
    }

    /// z passes the Dikin test at the primal point s of `oracle`:
    /// r = (ν−1)⟨z, ∇²f(s)⁻¹z⟩/⟨s,z⟩² < 1 certifies z ∈ int K* (r₀ =
    /// (ν−1)/ν on the central path). The O(n³) bound first, then the exact
    /// form (QICS's proximity test), bracketed by conjugate gradients before
    /// S₀'s factor: the bound can be far above it off the central path
    /// (two_matrix_L6: r 0.913 by the bound, 0.8333 exact).
    fn dual_ok(&self, oracle: &mut Oracle<T>, s: &[T], z: &[T], cg: &mut u32) -> bool {
        let sz = s.dot(z);
        if !(sz > T::zero()) || !z.iter().all(|v| v.is_finite()) {
            return false;
        }
        let nu1 = self.nu() - T::one();
        oracle.dual_ok(z, |q| nu1 * q < sz * sz, cg)
    }
}

impl<T> Cone<T> for OpRelEntropyCone<T>
where
    T: FloatT,
{
    fn degree(&self) -> usize {
        3 * self.n
    }

    fn numel(&self) -> usize {
        self.dim()
    }

    fn is_symmetric(&self) -> bool {
        false
    }

    fn is_sparse_expandable(&self) -> bool {
        false
    }

    fn allows_primal_dual_scaling(&self) -> bool {
        false
    }

    fn rectify_equilibration(&self, δ: &mut [T], e: &[T]) -> bool {
        // one scalar for the whole cone: (T, X, Y) ↦ c(T, X, Y) keeps K
        δ.copy_from(e).recip().scale(e.mean());
        true
    }

    fn margins(&mut self, _z: &mut [T], _pd: PrimalOrDualCone) -> (T, T) {
        // nonsymmetric problems always use unit_initialization
        unreachable!();
    }

    fn scaled_unit_shift(&self, _z: &mut [T], _α: T, _pd: PrimalOrDualCone) {
        unreachable!();
    }

    fn unit_initialization(&self, z: &mut [T], s: &mut [T]) {
        let n = self.n;
        let m = triangular_number(n);
        let [t, x, y] = central_point::<T>();
        s.set(T::zero());
        for i in 0..n {
            let k = tri(i, i);
            s[k] = t;
            s[m + k] = x;
            s[2 * m + k] = y;
        }
        z.copy_from(s);
    }

    fn set_identity_scaling(&mut self) {
        unreachable!();
    }

    fn update_scaling(
        &mut self,
        s: &[T],
        z: &[T],
        μ: T,
        _scaling_strategy: ScalingStrategy,
    ) -> bool {
        let Some(mut oracle) = Oracle::new(self.n, s) else {
            return false;
        };
        let Some(s0i) = oracle.schur_inverse() else {
            return false;
        };
        let n = self.n;
        let m = triangular_number(n);
        let data = &mut self.data;
        let rows = &mut data.rows;
        rows.n = n;
        rows.dc.resize(m, T::zero());
        rows.dk.resize(m, T::zero());
        for j in 0..n {
            for i in 0..=j {
                rows.dc[tri(i, j)] = oracle.c[i + j * n];
                rows.dk[tri(i, j)] = oracle.d1[i + j * n];
            }
        }
        rows.q = std::mem::take(&mut oracle.q);
        data.p = std::mem::take(&mut oracle.p);
        data.zh = std::mem::take(&mut oracle.zh);
        data.s0i = s0i;
        data.μ = μ;
        data.s.copy_from(s);
        data.z.copy_from(z);
        true
    }

    fn Hs_is_diagonal(&self) -> bool {
        false
    }

    fn get_Hs(&self, Hsblock: &mut [T]) {
        // D in the KKT's eigen row coordinates (`OpRelEntropyRows`)
        let n = self.n;
        let m = triangular_number(n);
        let (zh, s0i) = (&self.data.zh, &self.data.s0i);
        let inv = self.data.μ.recip();
        let sqrt2 = T::SQRT_2();
        let z = |i: usize, j: usize| zh[i + j * n];
        let mut idx = 0;
        for l in 0..n {
            for k in 0..=l {
                for j in 0..=l {
                    for i in 0..=j {
                        if tri(i, j) > tri(k, l) {
                            break;
                        }
                        Hsblock[idx] = inv
                            * match (i == j, k == l) {
                                (false, false) => z(i, k) * z(j, l) + z(i, l) * z(j, k),
                                (true, false) => sqrt2 * z(j, l) * z(j, k),
                                (false, true) => sqrt2 * z(i, l) * z(j, k),
                                (true, true) => z(j, l) * z(j, l),
                            };
                        idx += 1;
                    }
                }
            }
        }
        let m2 = 2 * m;
        for j in 0..m2 {
            Hsblock[idx..idx + m].fill(T::zero());
            idx += m;
            for i in 0..=j {
                Hsblock[idx] = inv * s0i[i + j * m2];
                idx += 1;
            }
        }
    }

    fn mul_Hs(&mut self, y: &mut [T], x: &[T], _work: &mut [T]) {
        // y = N·D·Nᵀx
        let n = self.n;
        let m = triangular_number(n);
        let data = &self.data;
        let rows = &data.rows;
        let mut v = x[..3 * m].to_vec();
        congruence_parts(n, &data.p, &mut v, true);
        for i in 0..m {
            let t = v[i];
            v[m + i] += rows.dc[i] * t;
            v[2 * m + i] += rows.dk[i] * t;
        }
        let nn = n * n;
        let (mut a, mut tmp, mut b) = (vec![T::zero(); nn], vec![T::zero(); nn], vec![T::zero(); nn]);
        smat(n, &v[..m], &mut a);
        congruence(n, &data.zh, &a, false, &mut tmp, &mut b);
        let inv = data.μ.recip();
        svec(n, &b, &mut y[..m]);
        let m2 = 2 * m;
        for i in 0..m2 {
            y[m + i] = T::zero();
        }
        for j in 0..m2 {
            let vj = v[m + j];
            let col = &data.s0i[j * m2..(j + 1) * m2];
            for i in 0..m2 {
                y[m + i] += col[i] * vj;
            }
        }
        y[..3 * m].iter_mut().for_each(|t| *t *= inv);
        for i in 0..m {
            y[i] += rows.dc[i] * y[m + i] + rows.dk[i] * y[2 * m + i];
        }
        congruence_parts(n, &data.p, &mut y[..3 * m], false);
    }

    fn affine_ds(&self, ds: &mut [T], s: &[T]) {
        ds.copy_from(s);
    }

    fn combined_ds_shift(
        &mut self, shift: &mut [T], _step_z: &mut [T], _step_s: &mut [T], σμ: T
    ) {
        // ds = Hs z − (σμ/μ)s; the caller adds affine_ds = s.
        let z = std::mem::take(&mut self.data.z);
        let mut hz = std::mem::take(&mut self.data.work);
        self.mul_Hs(&mut hz, &z, &mut []);
        let c = T::one() + σμ / self.data.μ;
        for ((o, &h), &s) in shift.iter_mut().zip(&hz).zip(&self.data.s) {
            *o = h - c * s;
        }
        self.data.z = z;
        self.data.work = hz;
        {
            // Mehrotra term of z + σμ∇f(s) = 0 (Dahl–Andersen, primal side):
            // ds −= ½ Hs ∇³f(s)[Δs, H⁻¹Δz], H⁻¹ = μHs; for the orthant
            // this is (ΔsΔz)/z, the symmetric-cone corrector.
            let mut w = vec![T::zero(); shift.len()];
            self.mul_Hs(&mut w, _step_z, &mut []);
            let mu = self.data.μ;
            w.iter_mut().for_each(|v| *v *= mu);
            if let Some(v) = self.third_mixed(_step_s, &w) {
                let mut u = vec![T::zero(); v.len()];
                self.mul_Hs(&mut u, &v, &mut []);
                // weighted by α_aff (σ = (1 − α_aff)³): the full term
                // overshoots after short affine steps
                let σ = (σμ / mu).max(T::zero()).min(T::one());
                let half: T = (0.5).as_T();
                let w = half * (T::one() - σ.cbrt());
                shift.iter_mut().zip(&u).for_each(|(o, &u)| *o -= w * u);
            }
        }
    }

    fn Δs_from_Δz_offset(&mut self, out: &mut [T], ds: &[T], _work: &mut [T], _z: &[T]) {
        out.copy_from(ds);
    }

    fn step_length(
        &mut self,
        dz: &[T],
        ds: &[T],
        z: &[T],
        s: &[T],
        settings: &CoreSettings<T>,
        αmax: T,
    ) -> (T, T) {
        let step = settings.linesearch_backtrack_step;
        let αmin = settings.min_terminate_step_length;
        let n = self.n;
        let mut work = std::mem::take(&mut self.data.work);
        let mut wz = vec![T::zero(); work.len()];
        // primal: s + α ds interior; the oracle at αs serves the dual test
        let mut at = None;
        let αs = search(αmax, αmin, step, |α| {
            work.waxpby(T::one(), s, α, ds);
            at = Oracle::new(n, &work).map(|o| (α, o));
            at.is_some()
        });
        // dual: z + α dz certified at the trial primal point s + min(α, αs) ds
        // (`work` holds the point of `at`). A certificate at the current s
        // admits pairs far off the central path after the joint step:
        // two_matrix_L4 at 256 bits reached r = 1.22 at the new s, where no
        // dual step certifies (AlmostSolved at 1e-42). Neighbouring trials
        // are alike, and so are later searches: each test CG leaves open
        // halves the next one's budget (one_matrix_L12f: from iteration 40
        // every search's first trial ran the whole budget, then factored).
        // Trials below αs each have their own primal point and oracle; idle
        // pool workers take the next ones of the chain at once and the first
        // that passes wins, as in order (one_matrix_L12f: about seven such
        // trials per late affine search, each a factor of S₀).
        let mut cg = self.data.cg;
        // outside a pool (one solver thread) rayon would use its global pool
        let ways = rayon::current_thread_index().map_or(1, |_| rayon::current_num_threads());
        let this = &*self;
        let mut α = αmax;
        let mut αz = T::zero();
        while α > αmin {
            if α >= αs || ways == 1 {
                wz.waxpby(T::one(), z, α, dz);
                let a = α.min(αs);
                if at.as_ref().is_none_or(|(b, _)| *b != a) {
                    work.waxpby(T::one(), s, a, ds);
                    at = Oracle::new(n, &work).map(|o| (a, o));
                }
                if at.as_mut().is_some_and(|(_, o)| this.dual_ok(o, &work, &wz, &mut cg)) {
                    αz = α;
                    break;
                }
                α *= step;
                continue;
            }
            let mut batch = Vec::with_capacity(ways);
            while batch.len() < ways && α > αmin {
                batch.push(α);
                α *= step;
            }
            let start = cg;
            let trials: Vec<(bool, u32)> = batch
                .par_iter()
                .map(|&a| {
                    let mut p = vec![T::zero(); s.len()];
                    p.waxpby(T::one(), s, a, ds);
                    let mut q = vec![T::zero(); z.len()];
                    q.waxpby(T::one(), z, a, dz);
                    let mut cg = start;
                    let ok = Oracle::new(n, &p)
                        .is_some_and(|mut o| this.dual_ok(&mut o, &p, &q, &mut cg));
                    (ok, cg)
                })
                .collect();
            if let Some(i) = trials.iter().position(|t| t.0) {
                (αz, cg) = (batch[i], trials[i].1);
                break;
            }
            cg = trials.iter().map(|t| t.1).max().unwrap_or(cg);
        }
        self.data.work = work;
        self.data.cg = cg;
        (αz, αs)
    }

    fn compute_barrier(&mut self, z: &[T], s: &[T], dz: &[T], ds: &[T], α: T) -> T {
        // f(s) + f*(z) ≈ −ν(1 + ln μ̃) + ½(B(z)/μ̃² − ν), μ̃ = ⟨s, z⟩/ν: the
        // conjugate's second-order expansion about −μ̃∇f(s), with the O(n³)
        // upper bound B of the inverse-Hessian norm (`dual_form`). Used only
        // beside exponential or power cones (`barrier_backtrack`); the dual
        // step is certified in `step_length`.
        let mut ws = std::mem::take(&mut self.data.work);
        ws.waxpby(T::one(), s, α, ds);
        let mut wz = vec![T::zero(); ws.len()];
        wz.waxpby(T::one(), z, α, dz);
        let out = match Oracle::new(self.n, &ws) {
            None => T::infinity(),
            Some(mut o) => {
                let nu = self.nu();
                let mu = ws.dot(&wz) / nu;
                if !(mu > T::zero()) {
                    T::infinity()
                } else {
                    let half: T = (0.5).as_T();
                    -nu * (T::one() + mu.ln()) + half * (o.dual_form(&wz, false).unwrap() / (mu * mu) - nu)
                }
            }
        };
        self.data.work = ws;
        out
    }
}

#[cfg(test)]
#[path = "tests/oprelentrcone.rs"]
mod tests;
