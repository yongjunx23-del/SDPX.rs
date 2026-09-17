//! Precision-preserving dense kernels for the inline MPFR scalar.
//!
//! Column-major/triangle conventions follow the Clarabel BLAS provider and
//! BFLA's generic Cholesky/triangular kernels. SVD uses Householder reduction
//! and bidiagonal QR (never an eigendecomposition of a Gram matrix). Symmetric
//! eigendecomposition uses Householder tridiagonalization and implicit QL
//! iteration with Wilkinson shifts (EISPACK tql2 structure). All arithmetic,
//! scaling, and stopping criteria remain at the scalar's declared precision.
#![allow(clippy::too_many_arguments)]
use super::*;
use num_traits::{One, ToPrimitive, Zero};
use rayon::prelude::*;
use sdpx_arithmetic::{MpFloat, Scalar};
type F<const N: usize> = MpFloat<N>;

fn upper(c: u8) -> u8 {
    c.to_ascii_uppercase()
}
fn tri(c: u8) -> bool {
    matches!(upper(c), b'U' | b'L')
}
fn trans(c: u8) -> bool {
    matches!(upper(c), b'N' | b'T' | b'C')
}
fn at<const N: usize>(a: &[F<N>], ld: usize, i: usize, j: usize, t: u8) -> F<N> {
    if upper(t) == b'N' {
        a[i + j * ld]
    } else {
        a[j + i * ld]
    }
}
fn sym<const N: usize>(a: &[F<N>], ld: usize, i: usize, j: usize, u: u8) -> F<N> {
    if (upper(u) == b'U' && i <= j) || (upper(u) == b'L' && i >= j) {
        a[i + j * ld]
    } else {
        a[j + i * ld]
    }
}
fn vi(i: usize, n: usize, inc: i32) -> usize {
    if inc > 0 {
        i * inc as usize
    } else {
        (n - 1 - i) * (-inc as i64) as usize
    }
}
fn valid(a: usize, rows: i32, cols: i32, ld: i32) -> bool {
    rows >= 0
        && cols >= 0
        && ld >= rows.max(1)
        && (rows == 0 || cols == 0 || a >= (cols as usize - 1) * ld as usize + rows as usize)
}
fn axpby<const N: usize>(alpha: F<N>, x: F<N>, beta: F<N>, y: F<N>) -> F<N> {
    let p = if alpha == F::zero() {
        F::zero()
    } else {
        alpha * x
    };
    if beta == F::zero() {
        p
    } else {
        p + beta * y
    }
}
// Each tile contains complete output columns. Padding and the unused SYRK
// triangle remain untouched; each output owns its original scalar p-loop.
fn output_columns<const N: usize>(
    c: &mut [F<N>],
    rows: usize,
    columns: usize,
    ld: usize,
    parallel: Option<(&rayon::ThreadPool, usize)>,
    column: impl Fn(usize, &mut [F<N>]) + Sync,
) {
    let used = (columns - 1) * ld + rows;
    if let Some((pool, tile)) =
        parallel.filter(|(pool, tile)| pool.current_num_threads() > 1 && *tile > 0)
    {
        let tile = tile.min(columns);
        pool.install(|| {
            c[..used]
                .par_chunks_mut(tile * ld)
                .enumerate()
                .for_each(|(t, values)| {
                    for (j, values) in values.chunks_mut(ld).enumerate() {
                        column(t * tile + j, &mut values[..rows]);
                    }
                })
        });
    } else {
        for j in 0..columns {
            column(j, &mut c[j * ld..j * ld + rows]);
        }
    }
}
// Prefix sums count touched entries, so the common inner dimension cancels
// from SYRK's balancing ratio. u128 covers all validated BLAS dimensions.
fn syrk_prefix(n: usize, end: usize, upper: bool) -> u128 {
    let end = end as u128;
    if upper {
        end * (end + 1) / 2
    } else {
        end * (2 * n as u128 - end + 1) / 2
    }
}
fn syrk_cut(n: usize, begin: usize, end: usize, lanes: usize, upper: bool) -> usize {
    let left_lanes = lanes / 2;
    let base = syrk_prefix(n, begin, upper);
    let target = (syrk_prefix(n, end, upper) - base) * left_lanes as u128;
    // Reserve at least one complete column for every remaining leaf.
    let first = begin + left_lanes;
    let mut lo = first;
    let mut hi = end - (lanes - left_lanes);
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if (syrk_prefix(n, mid, upper) - base) * (lanes as u128) < target {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    let error = |cut| ((syrk_prefix(n, cut, upper) - base) * lanes as u128).abs_diff(target);
    if lo > first && error(lo - 1) <= error(lo) {
        lo - 1
    } else {
        lo
    }
}
fn split_syrk_columns<const N: usize>(
    c: &mut [F<N>],
    n: usize,
    ld: usize,
    begin: usize,
    end: usize,
    lanes: usize,
    upper: bool,
    column: &(impl Fn(usize, &mut [F<N>]) + Sync),
) {
    if lanes == 1 {
        for (offset, values) in c.chunks_mut(ld).enumerate() {
            column(begin + offset, &mut values[..n]);
        }
    } else {
        let cut = syrk_cut(n, begin, end, lanes, upper);
        let (left, right) = c.split_at_mut((cut - begin) * ld);
        rayon::join(
            || split_syrk_columns(left, n, ld, begin, cut, lanes / 2, upper, column),
            || split_syrk_columns(right, n, ld, cut, end, lanes - lanes / 2, upper, column),
        );
    }
}
fn syrk_output_columns<const N: usize>(
    c: &mut [F<N>],
    n: usize,
    ld: usize,
    upper: bool,
    parallel: Option<(&rayon::ThreadPool, usize)>,
    column: impl Fn(usize, &mut [F<N>]) + Sync,
) {
    if let Some((pool, tile)) = parallel.filter(|(p, t)| p.current_num_threads() > 1 && *t > 0) {
        // Retain the existing configured number of column tasks, changing
        // only their boundaries. Splits borrow disjoint slices; no scratch.
        let lanes = n.div_ceil(tile.min(n));
        pool.install(|| {
            split_syrk_columns(
                &mut c[..(n - 1) * ld + n],
                n,
                ld,
                0,
                n,
                lanes,
                upper,
                &column,
            )
        });
    } else {
        output_columns(c, n, n, ld, None, column);
    }
}

impl<const N: usize> private::BlasFloatSealed for F<N> {}
impl<const N: usize> BlasFloatT for F<N> {}
fn gemm<const N: usize>(
    ta: u8,
    tb: u8,
    m: i32,
    n: i32,
    k: i32,
    alpha: F<N>,
    a: &[F<N>],
    lda: i32,
    b: &[F<N>],
    ldb: i32,
    beta: F<N>,
    c: &mut [F<N>],
    ldc: i32,
    parallel: Option<(&rayon::ThreadPool, usize)>,
) {
    assert!(trans(ta) && trans(tb) && m >= 0 && n >= 0 && k >= 0);
    assert!(valid(
        a.len(),
        if upper(ta) == b'N' { m } else { k },
        if upper(ta) == b'N' { k } else { m },
        lda
    ));
    assert!(
        valid(
            b.len(),
            if upper(tb) == b'N' { k } else { n },
            if upper(tb) == b'N' { n } else { k },
            ldb
        ) && valid(c.len(), m, n, ldc)
    );
    if m == 0 || n == 0 {
        return;
    }
    let column = |j: usize, column: &mut [F<N>]| {
        for i in 0..m as usize {
            let mut v = F::<N>::zero();
            if alpha != F::<N>::zero() {
                // Fused accumulation at every precision; the descriptor form is
                // not specific to any limb count.
                let (a0, da) = if upper(ta) == b'N' {
                    (i, lda as usize)
                } else {
                    (i * lda as usize, 1)
                };
                let (b0, db) = if upper(tb) == b'N' {
                    (j * ldb as usize, 1)
                } else {
                    (j, ldb as usize)
                };
                v = F::dot_fma((0..k as usize).map(|p| (&a[a0 + p * da], &b[b0 + p * db])));
            }
            column[i] = axpby(alpha, v, beta, column[i]);
        }
    };
    output_columns(c, m as usize, n as usize, ldc as usize, parallel, column);
}
impl<const N: usize> XgemmScalar for F<N> {
    fn xgemm(
        ta: u8,
        tb: u8,
        m: i32,
        n: i32,
        k: i32,
        alpha: Self,
        a: &[Self],
        lda: i32,
        b: &[Self],
        ldb: i32,
        beta: Self,
        c: &mut [Self],
        ldc: i32,
    ) {
        gemm(ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, None);
    }
    fn xgemm_pool(
        ta: u8,
        tb: u8,
        m: i32,
        n: i32,
        k: i32,
        alpha: Self,
        a: &[Self],
        lda: i32,
        b: &[Self],
        ldb: i32,
        beta: Self,
        c: &mut [Self],
        ldc: i32,
        pool: &rayon::ThreadPool,
        column_tile: usize,
    ) {
        gemm(
            ta,
            tb,
            m,
            n,
            k,
            alpha,
            a,
            lda,
            b,
            ldb,
            beta,
            c,
            ldc,
            Some((pool, column_tile)),
        );
    }
}
impl<const N: usize> XgemvScalar for F<N> {
    fn xgemv(
        t: u8,
        m: i32,
        n: i32,
        alpha: Self,
        a: &[Self],
        lda: i32,
        x: &[Self],
        incx: i32,
        beta: Self,
        y: &mut [Self],
        incy: i32,
    ) {
        assert!(trans(t) && valid(a.len(), m, n, lda) && incx != 0 && incy != 0);
        let (r, k) = if upper(t) == b'N' {
            (m as usize, n as usize)
        } else {
            (n as usize, m as usize)
        };
        if m == 0 || n == 0 {
            return;
        }
        assert!(
            x.len() > vi(0, k, incx).max(vi(k - 1, k, incx))
                && y.len() > vi(0, r, incy).max(vi(r - 1, r, incy))
        );
        for i in 0..r {
            let mut v = Self::zero();
            if alpha != Self::zero() {
                let (a0, da) = if upper(t) == b'N' {
                    (i, lda as usize)
                } else {
                    (i * lda as usize, 1)
                };
                v = F::dot_fma((0..k).map(|p| (&a[a0 + p * da], &x[vi(p, k, incx)])));
            }
            let q = vi(i, r, incy);
            y[q] = axpby(alpha, v, beta, y[q]);
        }
    }
}
impl<const N: usize> XsymvScalar for F<N> {
    fn xsymv(
        u: u8,
        n: i32,
        alpha: Self,
        a: &[Self],
        lda: i32,
        x: &[Self],
        incx: i32,
        beta: Self,
        y: &mut [Self],
        incy: i32,
    ) {
        assert!(tri(u) && valid(a.len(), n, n, lda) && incx != 0 && incy != 0);
        let n = n as usize;
        for i in 0..n {
            let mut v = Self::zero();
            if alpha != Self::zero() {
                for p in 0..n {
                    v = sym(a, lda as usize, i, p, u).mul_add(x[vi(p, n, incx)], v);
                }
            }
            let q = vi(i, n, incy);
            y[q] = axpby(alpha, v, beta, y[q]);
        }
    }
}
fn syrk<const N: usize>(
    u: u8,
    t: u8,
    n: i32,
    k: i32,
    alpha: F<N>,
    a: &[F<N>],
    lda: i32,
    beta: F<N>,
    c: &mut [F<N>],
    ldc: i32,
    parallel: Option<(&rayon::ThreadPool, usize)>,
) {
    assert!(tri(u) && trans(t) && n >= 0 && k >= 0 && valid(c.len(), n, n, ldc));
    assert!(valid(
        a.len(),
        if upper(t) == b'N' { n } else { k },
        if upper(t) == b'N' { k } else { n },
        lda
    ));
    if n == 0 {
        return;
    }
    let column = |j: usize, column: &mut [F<N>]| {
        for i in 0..n as usize {
            if (upper(u) == b'U' && i > j) || (upper(u) == b'L' && i < j) {
                continue;
            }
            let mut v = F::<N>::zero();
            if alpha != F::<N>::zero() {
                let (a0, da) = if upper(t) == b'N' {
                    (i, lda as usize)
                } else {
                    (i * lda as usize, 1)
                };
                let (b0, db) = if upper(t) == b'N' {
                    (j, lda as usize)
                } else {
                    (j * lda as usize, 1)
                };
                v = F::dot_fma((0..k as usize).map(|p| (&a[a0 + p * da], &a[b0 + p * db])));
            }
            column[i] = axpby(alpha, v, beta, column[i]);
        }
    };
    syrk_output_columns(
        c,
        n as usize,
        ldc as usize,
        upper(u) == b'U',
        parallel,
        column,
    );
}
impl<const N: usize> XsyrkScalar for F<N> {
    fn xsyrk(
        u: u8,
        t: u8,
        n: i32,
        k: i32,
        alpha: Self,
        a: &[Self],
        lda: i32,
        beta: Self,
        c: &mut [Self],
        ldc: i32,
    ) {
        syrk(u, t, n, k, alpha, a, lda, beta, c, ldc, None);
    }
    fn xsyrk_pool(
        u: u8,
        t: u8,
        n: i32,
        k: i32,
        alpha: Self,
        a: &[Self],
        lda: i32,
        beta: Self,
        c: &mut [Self],
        ldc: i32,
        pool: &rayon::ThreadPool,
        column_tile: usize,
    ) {
        syrk(
            u,
            t,
            n,
            k,
            alpha,
            a,
            lda,
            beta,
            c,
            ldc,
            Some((pool, column_tile)),
        );
    }
}
impl<const N: usize> Xsyr2kScalar for F<N> {
    fn xsyr2k(
        u: u8,
        t: u8,
        n: i32,
        k: i32,
        alpha: Self,
        a: &[Self],
        lda: i32,
        b: &[Self],
        ldb: i32,
        beta: Self,
        c: &mut [Self],
        ldc: i32,
    ) {
        assert!(tri(u) && trans(t) && n >= 0 && k >= 0 && valid(c.len(), n, n, ldc));
        let (r, s) = if upper(t) == b'N' { (n, k) } else { (k, n) };
        assert!(valid(a.len(), r, s, lda) && valid(b.len(), r, s, ldb));
        for j in 0..n as usize {
            for i in 0..n as usize {
                if (upper(u) == b'U' && i > j) || (upper(u) == b'L' && i < j) {
                    continue;
                }
                let mut v = Self::zero();
                if alpha != Self::zero() {
                    for p in 0..k as usize {
                        v += at(a, lda as usize, i, p, t) * at(b, ldb as usize, j, p, t)
                            + at(b, ldb as usize, i, p, t) * at(a, lda as usize, j, p, t);
                    }
                }
                let q = i + j * ldc as usize;
                c[q] = axpby(alpha, v, beta, c[q]);
            }
        }
    }
}
impl<const N: usize> XpotrfScalar for F<N> {
    fn xpotrf(u: u8, n: i32, a: &mut [Self], lda: i32, info: &mut i32) {
        *info = if !tri(u) {
            -1
        } else if n < 0 {
            -2
        } else if !valid(a.len(), n, n, lda) {
            -4
        } else {
            0
        };
        if *info != 0 {
            return;
        }
        let n = n as usize;
        let ld = lda as usize;
        for j in 0..n {
            let mut d = a[j + j * ld];
            for k in 0..j {
                let v = if upper(u) == b'L' {
                    a[j + k * ld]
                } else {
                    a[k + j * ld]
                };
                d = (-v).mul_add(v, d);
            }
            a[j + j * ld] = d;
            if !d.is_finite() || d <= Self::zero() {
                *info = (j + 1) as i32;
                return;
            }
            let d = d.sqrt();
            a[j + j * ld] = d;
            for i in j + 1..n {
                let q = if upper(u) == b'L' {
                    i + j * ld
                } else {
                    j + i * ld
                };
                let mut v = a[q];
                for k in 0..j {
                    let (v1, v2) = if upper(u) == b'L' {
                        (a[i + k * ld], a[j + k * ld])
                    } else {
                        (a[k + i * ld], a[k + j * ld])
                    };
                    v = (-v1).mul_add(v2, v);
                }
                a[q] = v / d;
            }
        }
    }
}
impl<const N: usize> XpotrsScalar for F<N> {
    fn xpotrs(
        u: u8,
        n: i32,
        nrhs: i32,
        a: &[Self],
        lda: i32,
        b: &mut [Self],
        ldb: i32,
        info: &mut i32,
    ) {
        *info = if !tri(u) {
            -1
        } else if n < 0 {
            -2
        } else if nrhs < 0 {
            -3
        } else if !valid(a.len(), n, n, lda) {
            -5
        } else if !valid(b.len(), n, nrhs, ldb) {
            -7
        } else {
            0
        };
        if *info != 0 {
            return;
        }
        let n = n as usize;
        let ld = lda as usize;
        let lb = ldb as usize;
        for r in 0..nrhs as usize {
            for i in 0..n {
                let mut v = b[i + r * lb];
                for k in 0..i {
                    let a_val = if upper(u) == b'L' {
                        a[i + k * ld]
                    } else {
                        a[k + i * ld]
                    };
                    let b_val = b[k + r * lb];
                    v = (-a_val).mul_add(b_val, v);
                }
                b[i + r * lb] = v / a[i + i * ld];
            }
            for i in (0..n).rev() {
                let mut v = b[i + r * lb];
                for k in i + 1..n {
                    let a_val = if upper(u) == b'L' {
                        a[k + i * ld]
                    } else {
                        a[i + k * ld]
                    };
                    let b_val = b[k + r * lb];
                    v = (-a_val).mul_add(b_val, v);
                }
                b[i + r * lb] = v / a[i + i * ld];
            }
        }
    }
}
impl<const N: usize> XgesvScalar for F<N> {
    fn xgesv(
        n: i32,
        nrhs: i32,
        a: &mut [Self],
        lda: i32,
        ipiv: &mut [i32],
        b: &mut [Self],
        ldb: i32,
        info: &mut i32,
    ) {
        *info = if n < 0 {
            -1
        } else if nrhs < 0 {
            -2
        } else if !valid(a.len(), n, n, lda) {
            -4
        } else if ipiv.len() < n as usize {
            -5
        } else if !valid(b.len(), n, nrhs, ldb) {
            -7
        } else {
            0
        };
        if *info != 0 {
            return;
        }
        let n = n as usize;
        let ld = lda as usize;
        let lb = ldb as usize;
        for j in 0..n {
            let mut p = j;
            for i in j + 1..n {
                if a[i + j * ld].abs() > a[p + j * ld].abs() {
                    p = i;
                }
            }
            ipiv[j] = (p + 1) as i32;
            if a[p + j * ld] == Self::zero() || !a[p + j * ld].is_finite() {
                *info = (j + 1) as i32;
                return;
            }
            for k in 0..n {
                a.swap(j + k * ld, p + k * ld);
            }
            for k in 0..nrhs as usize {
                b.swap(j + k * lb, p + k * lb);
            }
            for i in j + 1..n {
                let v = a[i + j * ld] / a[j + j * ld];
                a[i + j * ld] = v;
                for k in j + 1..n {
                    let d = a[j + k * ld];
                    a[i + k * ld] -= v * d;
                }
                for k in 0..nrhs as usize {
                    let d = b[j + k * lb];
                    b[i + k * lb] -= v * d;
                }
            }
        }
        for r in 0..nrhs as usize {
            for i in (0..n).rev() {
                let mut v = b[i + r * lb];
                for k in i + 1..n {
                    v -= a[i + k * ld] * b[k + r * lb];
                }
                b[i + r * lb] = v / a[i + i * ld];
            }
        }
    }
}

fn num<const N: usize>(v: usize) -> F<N> {
    <F<N> as num_traits::FromPrimitive>::from_usize(v).unwrap()
}
fn hypot<const N: usize>(x: F<N>, y: F<N>) -> F<N> {
    let x = x.abs();
    let y = y.abs();
    let h = if x > y { x } else { y };
    if h == F::zero() {
        h
    } else {
        let x = x / h;
        let y = y / h;
        h * (x * x + y * y).sqrt()
    }
}
fn norm<const N: usize>(a: &[F<N>]) -> F<N> {
    a.iter().fold(F::zero(), |v, &x| hypot(v, x))
}
fn rotate<const N: usize>(a: &mut [F<N>], rows: usize, p: usize, q: usize, c: F<N>, s: F<N>) {
    for i in 0..rows {
        let x = a[i + p * rows];
        let y = a[i + q * rows];
        a[i + p * rows] = c * x - s * y;
        a[i + q * rows] = s * x + c * y;
    }
}
// Work is owned and retained by the decomposition engine, as in COSMO's
// PsdBlasWorkspace lifecycle: query once, resize, then reuse across factors.
fn take_work<'a, const N: usize>(work: &mut &'a mut [F<N>], len: usize) -> &'a mut [F<N>] {
    let (head, tail) = std::mem::take(work).split_at_mut(len);
    *work = tail;
    head
}
fn identity<const N: usize>(v: &mut [F<N>], n: usize) {
    v.fill(F::zero());
    for i in 0..n {
        v[i + i * n] = F::one();
    }
}
// Bidiagonal reduction, Demmel--Kahan iteration, shifted QR, and reflector
// reconstruction adapted from GenericLinearAlgebra.jl v0.4.0 src/svd.jl and
// SDPX.jl/src/core/utils/bigfloat_svd.jl (the latter maps v0.4.1 upstream).
//
// The MIT License (MIT)
// Copyright (c) 2014-2018 Andreas Noack
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// Real Givens: [c s; -s c] [x;y] = [r;0]. Stable hypot avoids squares of
// unscaled entries; all signs, norms, and divisions use the declared precision.
fn givens<const N: usize>(x: F<N>, y: F<N>) -> (F<N>, F<N>, F<N>) {
    let r = hypot(x, y);
    if r == F::zero() {
        (F::one(), F::zero(), r)
    } else {
        (x / r, y / r, r)
    }
}
fn svd_rotate<const N: usize>(u: &mut [F<N>], rows: usize, p: usize, c: F<N>, s: F<N>) {
    if !u.is_empty() {
        rotate(u, rows, p, p + 1, c, -s);
    }
}
// Construct H=I-tau*v*v', with v[0]=1 and H*x=beta*e_1.
fn reflector<const N: usize>(x: &mut [F<N>]) -> (F<N>, F<N>) {
    let magnitude = norm(x);
    if magnitude == F::zero() {
        return (F::zero(), F::zero());
    }
    let beta = if x[0] >= F::zero() {
        -magnitude
    } else {
        magnitude
    };
    let denominator = x[0] - beta;
    let tau = (beta - x[0]) / beta;
    for value in &mut x[1..] {
        *value /= denominator;
    }
    x[0] = F::one();
    (beta, tau)
}
fn reduce_bidiagonal<const N: usize>(
    a: &mut [F<N>],
    m: usize,
    n: usize,
    d: &mut [F<N>],
    e: &mut [F<N>],
    left: &mut [F<N>],
    right: &mut [F<N>],
    scratch: &mut [F<N>],
) {
    for k in 0..n {
        let x = &mut scratch[..m - k];
        x.copy_from_slice(&a[k + k * m..(k + 1) * m]);
        let (beta, tau) = reflector(x);
        d[k] = beta;
        left[k] = tau;
        for j in k + 1..n {
            let mut dot = F::zero();
            for i in k..m {
                dot += x[i - k] * a[i + j * m];
            }
            dot *= tau;
            for i in k..m {
                a[i + j * m] -= x[i - k] * dot;
            }
        }
        a[k + k * m] = beta;
        for i in k + 1..m {
            a[i + k * m] = x[i - k];
        }
        if k + 1 < n {
            let x = &mut scratch[..n - k - 1];
            for j in k + 1..n {
                x[j - k - 1] = a[k + j * m];
            }
            let (beta, tau) = reflector(x);
            e[k] = beta;
            right[k] = tau;
            for i in k + 1..m {
                let mut dot = F::zero();
                for j in k + 1..n {
                    dot += a[i + j * m] * x[j - k - 1];
                }
                dot *= tau;
                for j in k + 1..n {
                    a[i + j * m] -= dot * x[j - k - 1];
                }
            }
            a[k + (k + 1) * m] = beta;
            for j in k + 2..n {
                a[k + j * m] = x[j - k - 1];
            }
        }
    }
}
fn demmel_kahan<const N: usize>(
    d: &mut [F<N>],
    e: &mut [F<N>],
    lo: usize,
    hi: usize,
    u: &mut [F<N>],
    m: usize,
    v: &mut [F<N>],
    n: usize,
) {
    let (mut c, mut oldc, mut olds) = (F::one(), F::one(), F::zero());
    for i in lo..hi {
        let (nc, s, r) = givens(d[i] * c, e[i]);
        c = nc;
        svd_rotate(v, n, i, c, s);
        if i > lo {
            e[i - 1] = olds * r;
        }
        let (nc, ns, r) = givens(oldc * r, d[i + 1] * s);
        oldc = nc;
        olds = ns;
        d[i] = r;
        svd_rotate(u, m, i, oldc, olds);
    }
    let h = d[hi] * c;
    e[hi - 1] = h * olds;
    d[hi] = h * oldc;
}
fn shifted_qr<const N: usize>(
    d: &mut [F<N>],
    e: &mut [F<N>],
    lo: usize,
    hi: usize,
    shift: F<N>,
    u: &mut [F<N>],
    m: usize,
    v: &mut [F<N>],
    n: usize,
) {
    let (c, s, _) = givens(d[lo] - shift * (shift / d[lo]), e[lo]);
    svd_rotate(v, n, lo, c, s);
    let mut di = d[lo] * c + e[lo] * s;
    let mut ei1 = -d[lo] * s + e[lo] * c;
    let mut di1 = d[lo + 1] * c;
    let mut bulge = d[lo + 1] * s;
    for i in lo..hi - 1 {
        let (c, s, _) = givens(di, bulge);
        svd_rotate(u, m, i, c, s);
        d[i] = c * di + s * bulge;
        let ei = c * ei1 + s * di1;
        di1 = -s * ei1 + c * di1;
        ei1 = e[i + 1] * c;
        bulge = s * e[i + 1];
        let (c, s, _) = givens(ei, bulge);
        svd_rotate(v, n, i + 1, c, s);
        e[i] = ei * c + bulge * s;
        di = di1 * c + ei1 * s;
        ei1 = -di1 * s + ei1 * c;
        bulge = d[i + 2] * s;
        di1 = d[i + 2] * c;
    }
    let (c, s, _) = givens(di, bulge);
    svd_rotate(u, m, hi - 1, c, s);
    d[hi - 1] = c * di + s * bulge;
    e[hi - 1] = c * ei1 + s * di1;
    d[hi] = -s * ei1 + c * di1;
}
// LAWN 3 Lemma 1 recurrences. Deflate off-diagonals relative to the estimated
// smallest singular value, never by an absolute cutoff on diagonal values.
fn estimate_smallest<const N: usize>(
    d: &[F<N>],
    e: &mut [F<N>],
    lo: usize,
    hi: usize,
    tol: F<N>,
) -> F<N> {
    let mut mu = d[lo].abs();
    let mut b1 = mu;
    for j in lo..hi {
        let denominator = mu + e[j].abs();
        mu = if denominator == F::zero() {
            F::zero()
        } else {
            d[j + 1].abs() * (mu / denominator)
        };
        if e[j].abs() < tol * mu {
            e[j] = F::zero();
        }
        if mu < b1 {
            b1 = mu;
        }
    }
    let mut lambda = d[hi].abs();
    let mut binf = lambda;
    for j in (lo..hi).rev() {
        let denominator = lambda + e[j].abs();
        lambda = if denominator == F::zero() {
            F::zero()
        } else {
            d[j].abs() * (lambda / denominator)
        };
        if e[j].abs() < tol * lambda {
            e[j] = F::zero();
        }
        if lambda < binf {
            binf = lambda;
        }
    }
    if binf < b1 {
        binf
    } else {
        b1
    }
}
// Smaller singular value of a 2x2 upper bidiagonal. Scaling and the product
// formula avoid squaring a tiny singular value when constructing the shift.
fn small_shift<const N: usize>(d1: F<N>, d2: F<N>, e: F<N>) -> F<N> {
    let mut h = d1.abs();
    if d2.abs() > h {
        h = d2.abs();
    }
    if e.abs() > h {
        h = e.abs();
    }
    if h == F::zero() {
        return h;
    }
    let a = d1 / h;
    let b = d2 / h;
    let c = e / h;
    let aa = a * a;
    let bb = b * b;
    let cc = c * c;
    let discriminant = hypot(
        (a + b) * (a - b),
        c * hypot(num::<N>(2).sqrt() * hypot(a, b), c),
    );
    let largest = ((aa + bb + cc + discriminant) / num::<N>(2)).sqrt();
    if d1.abs() < d2.abs() {
        (d1.abs() / largest) * (d2.abs() / h)
    } else {
        (d2.abs() / largest) * (d1.abs() / h)
    }
}
fn bidiagonal_svd<const N: usize>(
    d: &mut [F<N>],
    e: &mut [F<N>],
    u: &mut [F<N>],
    m: usize,
    v: &mut [F<N>],
    n: usize,
) -> Result<(), i32> {
    if n < 2 {
        return Ok(());
    }
    // Upstream GenericLinearAlgebra `__svd!` default tol = 100*eps(T); LAPACK
    // dbdsqr TOLMUL is also 100 at every MPFR precision >= 128 bits.
    let tol = F::epsilon() * num::<N>(100);
    let mut hi = n - 1;
    // Unlike upstream's unbounded controller, exhaustion is an explicit error.
    for _ in 0..(64 + 128 * N) * n {
        while hi > 0 && e[hi - 1] == F::zero() {
            hi -= 1;
        }
        if hi == 0 {
            return Ok(());
        }
        let mut lo = hi - 1;
        while lo > 0 && e[lo - 1] != F::zero() {
            lo -= 1;
        }
        if d[lo..=hi].iter().any(|&x| x == F::zero()) {
            demmel_kahan(d, e, lo, hi, u, m, v, n);
            continue;
        }
        let smallest = estimate_smallest(d, e, lo, hi, tol);
        let threshold = tol * smallest;
        let mut split = false;
        for x in &mut e[lo..hi] {
            if x.abs() <= threshold {
                *x = F::zero();
                split = true;
            }
        }
        if split {
            continue;
        }
        let mut largest = F::zero();
        for x in d[lo..=hi].iter().chain(e[lo..hi].iter()) {
            if x.abs() > largest {
                largest = x.abs();
            }
        }
        // Upstream `__svd!` shift guard at its default tolerance:
        // fudge * tol * sigma^- <= eps * sigma^+.
        if num::<N>(hi - lo + 1) * tol * smallest <= F::epsilon() * largest {
            demmel_kahan(d, e, lo, hi, u, m, v, n);
        } else {
            let shift = small_shift(d[hi - 1], d[hi], e[hi - 1]);
            if (shift / d[lo]).abs() < F::epsilon().sqrt() {
                demmel_kahan(d, e, lo, hi, u, m, v, n);
            } else {
                shifted_qr(d, e, lo, hi, shift, u, m, v, n);
            }
        }
        if d.iter().chain(e.iter()).any(|x| !x.is_finite()) {
            return Err(1);
        }
    }
    Err(1)
}
// Packed Householder reconstruction, applied backwards as in SDPX's upstream
// reflector routines. Initial identity columns also supply all null vectors.
fn apply_reflectors<const N: usize>(
    a: &[F<N>],
    m: usize,
    n: usize,
    left: &[F<N>],
    right: &[F<N>],
    u: &mut [F<N>],
    uc: usize,
    v: &mut [F<N>],
) {
    for k in (0..n).rev() {
        for j in 0..uc {
            let mut dot = u[k + j * m];
            for i in k + 1..m {
                dot += a[i + k * m] * u[i + j * m];
            }
            dot *= left[k];
            u[k + j * m] -= dot;
            for i in k + 1..m {
                u[i + j * m] -= a[i + k * m] * dot;
            }
        }
    }
    if !v.is_empty() {
        for k in (0..n.saturating_sub(1)).rev() {
            for j in 0..n {
                let mut dot = v[k + 1 + j * n];
                for i in k + 2..n {
                    dot += a[k + i * m] * v[i + j * n];
                }
                dot *= right[k];
                v[k + 1 + j * n] -= dot;
                for i in k + 2..n {
                    v[i + j * n] -= a[k + i * m] * dot;
                }
            }
        }
    }
}

// Scale only if every nonzero input remains representable. Returning a
// bounded failure is preferable to silently replacing a small singular value
// by zero at MPFR's exponent boundary.
fn scale_checked<const N: usize>(a: &mut [F<N>], scale: F<N>) -> Result<(), i32> {
    if scale == F::zero() {
        return Ok(());
    }
    for x in a {
        let old = *x;
        *x /= scale;
        if !x.is_finite() || (old != F::zero() && *x == F::zero()) {
            return Err(1);
        }
    }
    Ok(())
}
// Work layout for the tall factorization; wide matrices transpose both the
// input and requested vector counts. GESVD has no integer work argument, so
// exact integer indices occupy scalar work cells during output ordering.
fn svd_work_len(m: usize, n: usize, uc: usize, vr: usize) -> Option<usize> {
    let (m, n, uc, vr) = if m < n {
        (n, m, vr, uc)
    } else {
        (m, n, uc, vr)
    };
    let mut size = 0usize;
    for cells in [
        m.checked_mul(n)?,
        n,
        n.saturating_sub(1),
        n,
        n.saturating_sub(1),
        m,
        m.checked_mul(uc)?,
        if vr > 0 { n.checked_mul(n)? } else { 0 },
        n,
        m.checked_mul(uc)?,
        vr.checked_mul(n)?,
        n,
    ] {
        size = size.checked_add(cells)?;
    }
    Some(size.max(1))
}
// The returned vectors are in the tall orientation. The caller transposes
// their indexing for a wide input, without allocating intermediate factors.
fn svd<'a, const N: usize>(
    a: &[F<N>],
    lda: usize,
    m: usize,
    n: usize,
    uc: usize,
    vr: usize,
    work: &'a mut [F<N>],
) -> Result<(&'a [F<N>], &'a [F<N>], &'a [F<N>]), i32> {
    let wide = m < n;
    let (m, n, uc, vr) = if wide { (n, m, vr, uc) } else { (m, n, uc, vr) };
    work.fill(F::zero());
    let mut work = work;
    let b = take_work(&mut work, m * n);
    let d = take_work(&mut work, n);
    let e = take_work(&mut work, n.saturating_sub(1));
    let left = take_work(&mut work, n);
    let right = take_work(&mut work, n.saturating_sub(1));
    let scratch = take_work(&mut work, m);
    let u = take_work(&mut work, m * uc);
    let v = take_work(&mut work, if vr > 0 { n * n } else { 0 });
    let order = take_work(&mut work, n);
    let sorted_u = take_work(&mut work, m * uc);
    let vt = take_work(&mut work, vr * n);
    let ss = take_work(&mut work, n);
    for j in 0..n {
        for i in 0..m {
            b[i + j * m] = if wide { a[j + i * lda] } else { a[i + j * lda] };
        }
    }
    let mut scale = F::zero();
    for &x in b.iter() {
        if !x.is_finite() {
            return Err(1);
        }
        if x.abs() > scale {
            scale = x.abs();
        }
    }
    scale_checked(b, scale)?;
    reduce_bidiagonal(b, m, n, d, e, left, right, scratch);
    if b.iter()
        .chain(d.iter())
        .chain(e.iter())
        .any(|x| !x.is_finite())
    {
        return Err(1);
    }
    for j in 0..uc {
        u[j + j * m] = F::one();
    }
    if vr > 0 {
        identity(v, n);
    }
    bidiagonal_svd(d, e, u, m, v, n)?;
    apply_reflectors(b, m, n, left, right, u, uc, v);
    for (i, cell) in order.iter_mut().enumerate() {
        *cell = num::<N>(i);
    }
    // Explicit index ties reproduce the original stable ordering without
    // sort_by's temporary heap buffer.
    order.sort_unstable_by(|i, j| {
        let (i, j) = (i.to_usize().unwrap(), j.to_usize().unwrap());
        d[j].abs().partial_cmp(&d[i].abs()).unwrap().then(i.cmp(&j))
    });
    for j in 0..n {
        let p = order[j].to_usize().unwrap();
        ss[j] = d[p].abs() * scale;
        if !ss[j].is_finite() || (d[p] != F::zero() && ss[j] == F::zero()) {
            return Err(1);
        }
        if j < uc {
            let sign = if d[p] < F::zero() {
                -F::one()
            } else {
                F::one()
            };
            for i in 0..m {
                sorted_u[i + j * m] = sign * u[i + p * m];
            }
        }
        if j < vr {
            for i in 0..n {
                vt[j + i * vr] = v[i + p * n];
            }
        }
    }
    for j in n..uc {
        for i in 0..m {
            sorted_u[i + j * m] = u[i + j * m];
        }
    }
    Ok((ss, sorted_u, vt))
}

impl<const N: usize> XgesvdScalar for F<N> {
    fn xgesvd(
        ju: u8,
        jv: u8,
        m: i32,
        n: i32,
        a: &mut [Self],
        lda: i32,
        s: &mut [Self],
        u: &mut [Self],
        ldu: i32,
        vt: &mut [Self],
        ldvt: i32,
        work: &mut [Self],
        lwork: i32,
        info: &mut i32,
    ) {
        let ju = upper(ju);
        let jv = upper(jv);
        let r = m.min(n).max(0);
        let uc = if ju == b'A' {
            m
        } else if ju == b'S' {
            r
        } else {
            0
        };
        let vr = if jv == b'A' {
            n
        } else if jv == b'S' {
            r
        } else {
            0
        };
        *info = if !matches!(ju, b'A' | b'S' | b'O' | b'N') {
            -1
        } else if !matches!(jv, b'A' | b'S' | b'O' | b'N') || (ju == b'O' && jv == b'O') {
            -2
        } else if m < 0 {
            -3
        } else if n < 0 {
            -4
        } else if lda < m.max(1) {
            -6
        } else if ldu < 1 || (uc > 0 && ldu < m) {
            -9
        } else if ldvt < vr.max(1) {
            -11
        } else if work.is_empty() || lwork < -1 || (lwork != -1 && lwork < 1) {
            -13
        } else {
            0
        };
        if *info != 0 {
            return;
        }
        let out_uc = if ju == b'O' { r } else { uc } as usize;
        let out_vr = if jv == b'O' { r } else { vr } as usize;
        let Some(required) = svd_work_len(m as usize, n as usize, out_uc, out_vr)
            .filter(|&size| size <= i32::MAX as usize)
        else {
            *info = -13;
            return;
        };
        work[0] = num::<N>(required);
        if lwork == -1 {
            return;
        }
        if lwork < required as i32 || work.len() < lwork as usize {
            *info = -13;
            return;
        }
        if !valid(a.len(), m, n, lda) {
            *info = -5;
            return;
        }
        if s.len() < r as usize {
            *info = -7;
            return;
        }
        if uc > 0 && !valid(u.len(), m, uc, ldu) {
            *info = -8;
            return;
        }
        if vr > 0 && !valid(vt.len(), vr, n, ldvt) {
            *info = -10;
            return;
        }
        let mm = m as usize;
        let nn = n as usize;
        match svd(
            a,
            lda as usize,
            mm,
            nn,
            out_uc,
            out_vr,
            &mut work[..required],
        ) {
            Err(e) => *info = e,
            Ok((ss, uu, vv)) => {
                s[..r as usize].copy_from_slice(ss);
                let get_u = |i, j| {
                    if mm < nn {
                        vv[j + i * out_uc]
                    } else {
                        uu[i + j * mm]
                    }
                };
                let get_vt = |i, j| {
                    if mm < nn {
                        uu[j + i * nn]
                    } else {
                        vv[i + j * out_vr]
                    }
                };
                for j in 0..uc as usize {
                    for i in 0..mm {
                        u[i + j * ldu as usize] = get_u(i, j);
                    }
                }
                for j in 0..nn {
                    for i in 0..vr as usize {
                        vt[i + j * ldvt as usize] = get_vt(i, j);
                    }
                }
                if ju == b'O' {
                    for j in 0..r as usize {
                        for i in 0..mm {
                            a[i + j * lda as usize] = get_u(i, j);
                        }
                    }
                }
                if jv == b'O' {
                    for j in 0..nn {
                        for i in 0..r as usize {
                            a[i + j * lda as usize] = get_vt(i, j);
                        }
                    }
                }
            }
        }
        work[0] = num::<N>(required);
    }
}
impl<const N: usize> XgesddScalar for F<N> {
    fn xgesdd(
        j: u8,
        m: i32,
        n: i32,
        a: &mut [Self],
        lda: i32,
        s: &mut [Self],
        u: &mut [Self],
        ldu: i32,
        vt: &mut [Self],
        ldvt: i32,
        work: &mut [Self],
        lwork: i32,
        _iwork: &mut [i32],
        info: &mut i32,
    ) {
        let j = upper(j);
        if !matches!(j, b'A' | b'S' | b'O' | b'N') {
            *info = -1;
            return;
        }
        let (ju, jv) = if j == b'O' {
            if m >= n {
                (b'O', b'A')
            } else {
                (b'A', b'O')
            }
        } else {
            (j, j)
        };
        Self::xgesvd(ju, jv, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, info);
        // GESDD has one fewer JOB argument; otherwise positions are identical.
        if *info < -2 {
            *info += 1;
        } else if *info == -2 {
            *info = -1;
        }
    }
}

// Householder tridiagonalization of the scaled symmetric matrix in `b`
// (n×n, column-major, both triangles filled). Produces the diagonal `d`,
// the subdiagonal `e` (e[k] = T[k+1,k]), and packs each reflector's v[1..]
// into b[k+2..n, k] with its tau in `taus` for backward accumulation.
fn tridiagonalize<const N: usize>(
    b: &mut [F<N>],
    n: usize,
    d: &mut [F<N>],
    e: &mut [F<N>],
    taus: &mut [F<N>],
    scratch: &mut [F<N>],
) {
    for k in 0..n.saturating_sub(2) {
        let len = n - k - 1;
        let (v, p) = scratch.split_at_mut(n);
        let v = &mut v[..len];
        for i in 0..len {
            v[i] = b[k + 1 + i + k * n];
        }
        let (beta, tau) = reflector(v);
        e[k] = beta;
        taus[k] = tau;
        if tau != F::zero() {
            // Two-sided update of the trailing block B = b[k+1..n, k+1..n]:
            // p = tau * B v; w = p - (tau/2)(p·v) v; B -= v wᵀ + w vᵀ.
            let p = &mut p[..len];
            for i in 0..len {
                let mut acc = F::zero();
                for j in 0..len {
                    acc += b[k + 1 + i + (k + 1 + j) * n] * v[j];
                }
                p[i] = tau * acc;
            }
            let mut dot = F::zero();
            for i in 0..len {
                dot += p[i] * v[i];
            }
            let alpha = (tau / num::<N>(2)) * dot;
            for i in 0..len {
                p[i] -= alpha * v[i];
            }
            for j in 0..len {
                for i in 0..len {
                    b[k + 1 + i + (k + 1 + j) * n] -= v[i] * p[j] + p[i] * v[j];
                }
            }
        }
        // v[0] == 1 is implicit; only the tail is stored.
        for i in 1..len {
            b[k + 1 + i + k * n] = v[i];
        }
    }
    if n >= 2 {
        e[n - 2] = b[n - 1 + (n - 2) * n];
    }
    for i in 0..n {
        d[i] = b[i + i * n];
    }
}
// Accumulate the packed reflectors backwards into the identity, matching the
// apply_reflectors pattern: Q = H_0 ... H_{n-3} with H_k acting on k+1..n.
fn form_q<const N: usize>(b: &[F<N>], n: usize, taus: &[F<N>], q: &mut [F<N>]) {
    identity(q, n);
    for k in (0..n.saturating_sub(2)).rev() {
        for j in 0..n {
            let mut dot = q[k + 1 + j * n];
            for i in k + 2..n {
                dot += b[i + k * n] * q[i + j * n];
            }
            dot *= taus[k];
            q[k + 1 + j * n] -= dot;
            for i in k + 2..n {
                q[i + j * n] -= b[i + k * n] * dot;
            }
        }
    }
}
// Implicit QL iteration with Wilkinson shift on the tridiagonal (d, e),
// following EISPACK tql2 / Numerical Recipes tqli. When `q` is nonempty its
// columns accumulate the rotations; the scalar sequence on (d, e) is
// identical either way, so 'N' and 'V' eigenvalues agree bit-for-bit.
fn tridiagonal_ql<const N: usize>(
    d: &mut [F<N>],
    e: &mut [F<N>],
    q: &mut [F<N>],
    n: usize,
) -> Result<(), i32> {
    let eps = F::epsilon();
    if n == 0 {
        return Ok(());
    }
    e[n - 1] = F::zero();
    let mut budget = (64 + 2 * N * 64) * n;
    for l in 0..n {
        loop {
            // Locate a small subdiagonal element.
            let mut m = l;
            while m < n - 1 {
                let dd = d[m].abs() + d[m + 1].abs();
                if e[m].abs() <= eps * dd {
                    break;
                }
                m += 1;
            }
            if m == l {
                break;
            }
            budget -= 1;
            if budget == 0 {
                return Err(1);
            }
            let mut g = (d[l + 1] - d[l]) / (num::<N>(2) * e[l]);
            let r = hypot(g, F::one());
            g = d[m] - d[l] + e[l] / (g + if g >= F::zero() { r } else { -r });
            let (mut s, mut c, mut p) = (F::one(), F::one(), F::zero());
            let mut underflow = false;
            for i in (l..m).rev() {
                let f = s * e[i];
                let bb = c * e[i];
                let r = hypot(f, g);
                e[i + 1] = r;
                if r == F::zero() {
                    // Recover from underflow.
                    d[i + 1] -= p;
                    e[m] = F::zero();
                    underflow = true;
                    break;
                }
                s = f / r;
                c = g / r;
                g = d[i + 1] - p;
                let r = (d[i] - g) * s + num::<N>(2) * c * bb;
                p = s * r;
                d[i + 1] = g + p;
                g = c * r - bb;
                if !q.is_empty() {
                    for k in 0..n {
                        let f = q[k + (i + 1) * n];
                        q[k + (i + 1) * n] = s * q[k + i * n] + c * f;
                        q[k + i * n] = c * q[k + i * n] - s * f;
                    }
                }
            }
            if underflow {
                continue;
            }
            d[l] -= p;
            e[l] = g;
            e[m] = F::zero();
        }
    }
    Ok(())
}

// Number of eigenvalues strictly below x on the tridiagonal (d, e), via
// the Sturm sequence of T - xI. Use the limit from below in x: an exactly
// zero pivot is not counted and is continued as a tiny positive pivot.
// Scale that continuation with the matrix; disconnected blocks require no
// division and hence no continuation of the preceding block's zero pivot.
fn sturm_count<const N: usize>(d: &[F<N>], e: &[F<N>], n: usize, x: F<N>) -> usize {
    let mut scale = F::zero();
    for value in d[..n].iter().chain(e[..n.saturating_sub(1)].iter()) {
        if value.abs() > scale {
            scale = value.abs();
        }
    }
    let tiny = (scale * F::epsilon()).max(F::min_positive_value());
    let mut count = 0usize;
    let mut q = d[0] - x;
    if q < F::zero() {
        count += 1;
    }
    for i in 1..n {
        q = if e[i - 1] == F::zero() {
            d[i] - x
        } else {
            if q == F::zero() {
                q = tiny;
            }
            d[i] - x - (e[i - 1] * e[i - 1]) / q
        };
        if q < F::zero() {
            count += 1;
        }
    }
    count
}

// Solve (T - sI) x = b on the tridiagonal (d, e) by Gaussian elimination
// with partial pivoting (LAPACK dgtsv/dgttrs), fusing factor and solve.
// `dd`,`du`,`du2`,`dl` are n-sized scratch.  Returns false on an exactly
// singular pivot — at a nearly exact eigenvalue shift the caller treats
// that as convergence, not failure.
#[allow(clippy::too_many_arguments)]
fn tridiag_solve<const N: usize>(
    d: &[F<N>],
    e: &[F<N>],
    n: usize,
    s: F<N>,
    b: &mut [F<N>],
    dl: &mut [F<N>],
    dd: &mut [F<N>],
    du: &mut [F<N>],
    du2: &mut [F<N>],
) -> bool {
    for i in 0..n {
        dd[i] = d[i] - s;
    }
    for i in 0..n - 1 {
        dl[i] = e[i];
        du[i] = e[i];
    }
    for i in 0..n.saturating_sub(2) {
        du2[i] = F::zero();
    }
    for i in 0..n - 1 {
        if dd[i].abs() >= dl[i].abs() {
            if dd[i] != F::zero() {
                let fact = dl[i] / dd[i];
                dd[i + 1] -= fact * du[i];
                if i + 1 < n - 1 {
                    // For tridiagonal input du2[i] is zero on entry to this
                    // step; retain the general banded elimination identity.
                    du[i + 1] -= fact * du2[i];
                }
                b[i + 1] -= fact * b[i];
                dl[i] = fact;
            } else {
                dl[i] = F::zero();
            }
        } else {
            let fact = dd[i] / dl[i];
            dd[i] = dl[i];
            let temp = dd[i + 1];
            dd[i + 1] = du[i] - fact * temp;
            if i + 1 < n - 1 {
                let next = du[i + 1];
                du[i + 1] = du2[i] - fact * next;
                du2[i] = next;
            } else {
                du2[i] = F::zero();
            }
            du[i] = temp;
            dl[i] = fact;
            b.swap(i, i + 1);
            b[i + 1] -= dl[i] * b[i];
        }
    }
    if (0..n).any(|i| dd[i] == F::zero()) {
        return false;
    }
    b[n - 1] /= dd[n - 1];
    if n >= 2 {
        b[n - 2] = (b[n - 2] - du[n - 2] * b[n - 1]) / dd[n - 2];
        for i in (0..n - 2).rev() {
            b[i] = (b[i] - du[i] * b[i + 1] - du2[i] * b[i + 2]) / dd[i];
        }
    }
    true
}

#[cfg(test)]
std::thread_local! {
    static EIGVAL_STEPS: std::cell::Cell<(usize, usize)> = const { std::cell::Cell::new((0, 0)) };
}

// The k-th smallest eigenvalue (0-based) of the tridiagonal (d, e).
// Sturm-sequence bisection first isolates a bracket containing only that
// eigenvalue, then Rayleigh quotient iteration on the same tridiagonal
// polishes it.  Sturm counts verify the result, so failure is detected
// and the caller can fall back to the full QL sweep.  `scratch` needs
// at least 6n cells and is left clobbered.
fn tridiag_eigval_at<const N: usize>(
    d: &[F<N>],
    e: &[F<N>],
    n: usize,
    k: usize,
    scratch: &mut [F<N>],
) -> Option<F<N>> {
    #[cfg(test)]
    EIGVAL_STEPS.with(|v| v.set((0, 0)));
    if n == 1 {
        return (k == 0).then_some(d[0]);
    }
    // Gershgorin radius of the tridiagonal.
    let mut ger = d[0].abs() + e[0].abs();
    for i in 1..n {
        let mut r = d[i].abs() + e[i - 1].abs();
        if i + 1 < n {
            r += e[i].abs();
        }
        if r > ger {
            ger = r;
        }
    }
    if ger == F::zero() {
        return Some(F::zero());
    }
    let (mut lo, mut hi) = (-ger, ger);
    // Isolate: keep count(lo) <= k and count(hi) >= k+1 until the bracket
    // holds exactly one eigenvalue.
    // This is a fast path: spend at most 32 halvings on isolation (a
    // 2^-32 fraction of the initial bracket). More tightly clustered or
    // repeated roots use the full QL sweep, at the original precision.
    let max_bisect = 32;
    for _ in 0..max_bisect {
        if sturm_count(d, e, n, hi) - sturm_count(d, e, n, lo) <= 1 {
            break;
        }
        #[cfg(test)]
        EIGVAL_STEPS.with(|v| {
            let (b, r) = v.get();
            v.set((b + 1, r));
        });
        let mid = (lo + hi) / num::<N>(2);
        if mid == lo || mid == hi {
            break;
        }
        if sturm_count(d, e, n, mid) > k {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    if sturm_count(d, e, n, hi) - sturm_count(d, e, n, lo) != 1 {
        return None;
    }
    // Rayleigh quotient iteration: solve (T - μI)ŷ = y, μ += ŷᵀy/ŷᵀŷ.
    let (y, rest) = scratch.split_at_mut(n);
    let (b, rest) = rest.split_at_mut(n);
    let (dl, rest) = rest.split_at_mut(n);
    let (dd, rest) = rest.split_at_mut(n);
    let (du, du2) = rest.split_at_mut(n);
    let mut mu = (lo + hi) / num::<N>(2);
    let inv_n = F::one() / num::<N>(n).sqrt();
    y.fill(inv_n);
    let eps = F::epsilon();
    for _ in 0..32 {
        #[cfg(test)]
        EIGVAL_STEPS.with(|v| {
            let (b, r) = v.get();
            v.set((b, r + 1));
        });
        b.copy_from_slice(y);
        if !tridiag_solve(d, e, n, mu, b, dl, dd, du, du2) {
            break; // exact singular pivot: μ is already an eigenvalue
        }
        let mut numer = F::zero();
        let mut denom = F::zero();
        for i in 0..n {
            numer += b[i] * y[i];
            denom += b[i] * b[i];
        }
        if denom == F::zero() {
            return None;
        }
        // Keep the isolating bracket: unconstrained RQI can converge to
        // a different eigenvalue. An escaping trial uses bisection instead.
        if sturm_count(d, e, n, mu) > k {
            hi = hi.min(mu);
        } else {
            lo = lo.max(mu);
        }
        let step = numer / denom;
        let trial = mu + step;
        let safeguarded = trial < lo || trial > hi;
        mu = if safeguarded {
            (lo + hi) / num::<N>(2)
        } else {
            trial
        };
        let inv = F::one() / denom.sqrt();
        for i in 0..n {
            y[i] = b[i] * inv;
        }
        if !safeguarded && step.abs() <= eps * (mu.abs() + ger) {
            break;
        }
    }
    // Verify: exactly k eigenvalues below and k+1 at-or-below the answer
    // within a few ulps of the Gershgorin scale.
    let delta = eps * ger * num::<N>(8);
    if sturm_count(d, e, n, mu - delta) == k && sturm_count(d, e, n, mu + delta) == k + 1 {
        Some(mu)
    } else {
        None
    }
}

impl<const N: usize> XsyevrScalar for F<N> {
    fn xsyevr(
        job: u8,
        range: u8,
        uplo: u8,
        n: i32,
        a: &mut [Self],
        lda: i32,
        vl: Self,
        vu: Self,
        il: i32,
        iu: i32,
        abstol: Self,
        m: &mut i32,
        w: &mut [Self],
        z: &mut [Self],
        ldz: i32,
        isuppz: &mut [i32],
        work: &mut [Self],
        lwork: i32,
        iwork: &mut [i32],
        liwork: i32,
        info: &mut i32,
    ) {
        let job = upper(job);
        let range = upper(range);
        *m = 0;
        *info = if !matches!(job, b'N' | b'V') {
            -1
        } else if !matches!(range, b'A' | b'V' | b'I') {
            -2
        } else if !tri(uplo) {
            -3
        } else if n < 0 {
            -4
        } else if lda < n.max(1) {
            -6
        } else if range == b'V' && n > 0 && vu <= vl {
            -8
        } else if range == b'I' && (il < 1 || il > n.max(1)) {
            -9
        } else if range == b'I' && (iu < n.min(il) || iu > n) {
            -10
        } else if ldz < 1 || (job == b'V' && ldz < n) {
            -15
        } else if work.is_empty() || lwork < -1 || (lwork != -1 && liwork != -1 && lwork < 1) {
            -18
        } else if iwork.is_empty() || liwork < -1 || (lwork != -1 && liwork != -1 && liwork < 1) {
            -20
        } else {
            0
        };
        if *info != 0 {
            return;
        }
        let nn = n as usize;
        // Owned scratch: b (n²) + d (n) + e (n) + v/p (2n) + taus (n),
        // plus Q (n²) when vectors are requested and a 7n block for the
        // single-index eigenvalue path (Sturm + RQI).
        let indexed = job == b'N' && range == b'I' && il == iu;
        let Some(required) = nn
            .checked_mul(nn)
            .and_then(|cells| cells.checked_mul(if job == b'V' { 2 } else { 1 }))
            .and_then(|cells| cells.checked_add(nn.checked_mul(if indexed { 12 } else { 5 })?))
            .map(|cells| cells.max(1))
            .filter(|&cells| cells <= i32::MAX as usize)
        else {
            *info = -18;
            return;
        };
        let integer_required = nn.max(1);
        work[0] = num::<N>(required);
        iwork[0] = integer_required as i32;
        if lwork == -1 || liwork == -1 {
            return;
        }
        if lwork < required as i32 || work.len() < lwork as usize {
            *info = -18;
            return;
        }
        if liwork < integer_required as i32 || iwork.len() < liwork as usize {
            *info = -20;
            return;
        }
        if !valid(a.len(), n, n, lda) {
            *info = -5;
            return;
        }
        if w.len() < n as usize {
            *info = -13;
            return;
        }
        let mut tail = &mut work[..required];
        let b = take_work(&mut tail, nn * nn);
        let d = take_work(&mut tail, nn);
        let e = take_work(&mut tail, nn);
        let scratch = take_work(&mut tail, 2 * nn);
        let taus = take_work(&mut tail, nn);
        let idx = take_work(&mut tail, if indexed { 7 * nn } else { 0 });
        let q = if job == b'V' {
            take_work(&mut tail, nn * nn)
        } else {
            &mut tail[..0]
        };
        let mut scale = Self::zero();
        for j in 0..nn {
            for i in 0..nn {
                let x = sym(a, lda as usize, i, j, uplo);
                if !x.is_finite() {
                    *info = 1;
                    return;
                }
                b[i + j * nn] = x;
                if x.abs() > scale {
                    scale = x.abs();
                }
            }
        }
        if scale_checked(b, scale).is_err() {
            *info = 1;
            return;
        }
        tridiagonalize(b, nn, d, e, taus, scratch);
        if job == b'V' {
            form_q(b, nn, taus, q);
        }
        if indexed {
            // Single eigenvalue by index: Sturm isolation plus RQI polish,
            // verified by Sturm counts.  (d, e) are read-only here, so a
            // failure can fall through to the full QL sweep below.
            match tridiag_eigval_at(d, e, nn, (il - 1) as usize, idx) {
                Some(v) => {
                    w[0] = v * scale;
                    *m = 1;
                    work[0] = num::<N>(required);
                    iwork[0] = integer_required as i32;
                    return;
                }
                None => {}
            }
        }
        // ABSTOL is accepted for LAPACK signature compatibility but must not
        // loosen the eps-based deflation criterion inside the QL iteration.
        let _ = abstol;
        if tridiagonal_ql(d, e, q, nn).is_err() {
            *info = 1;
            return;
        }
        let order = &mut iwork[..nn];
        for (i, p) in order.iter_mut().enumerate() {
            *p = i as i32;
        }
        order.sort_unstable_by(|&i, &j| {
            let (i, j) = (i as usize, j as usize);
            d[i].partial_cmp(&d[j])
                .unwrap()
                .then(i.cmp(&j))
        });
        let selected = |k: usize, p: usize| {
            range == b'A'
                || (range == b'I' && k + 1 >= il as usize && k + 1 <= iu as usize)
                || (range == b'V' && d[p] * scale > vl && d[p] * scale <= vu)
        };
        let count = order
            .iter()
            .enumerate()
            .filter(|&(k, &p)| selected(k, p as usize))
            .count();
        if job == b'V' && (!valid(z.len(), n, count as i32, ldz) || isuppz.len() < 2 * count) {
            *info = -14;
            return;
        }
        let mut j = 0;
        for (k, &p) in order.iter().enumerate() {
            let p = p as usize;
            if !selected(k, p) {
                continue;
            }
            w[j] = d[p] * scale;
            if job == b'V' {
                for i in 0..nn {
                    z[i + j * ldz as usize] = q[i + p * nn];
                }
                isuppz[2 * j] = 1;
                isuppz[2 * j + 1] = n;
            }
            j += 1;
        }
        *m = count as i32;
        work[0] = num::<N>(required);
        iwork[0] = integer_required as i32;
    }
}

#[cfg(test)]
mod syrk_partition_tests {
    use super::*;

    fn leaves(
        n: usize,
        begin: usize,
        end: usize,
        lanes: usize,
        upper: bool,
    ) -> Vec<(usize, usize)> {
        if lanes == 1 {
            return vec![(begin, end)];
        }
        let cut = syrk_cut(n, begin, end, lanes, upper);
        let left = lanes / 2;
        assert!((begin + left..=end - (lanes - left)).contains(&cut));
        // Check the selected boundary against the actual touched-entry sum,
        // independently of the prefix formula and binary search.
        let weight = |a: usize, b: usize| {
            (a..b)
                .map(|j| if upper { j + 1 } else { n - j })
                .sum::<usize>()
        };
        let error = |at: usize| (weight(begin, at) * lanes).abs_diff(weight(begin, end) * left);
        assert_eq!(
            error(cut),
            (begin + left..=end - (lanes - left))
                .map(error)
                .min()
                .unwrap()
        );
        let mut out = leaves(n, begin, cut, left, upper);
        out.extend(leaves(n, cut, end, lanes - left, upper));
        out
    }
    #[test]
    fn triangular_partitions_preserve_budget_and_balance() {
        for n in 1usize..=65 {
            for upper in [false, true] {
                for lanes in 1..=n.min(16) {
                    let spans = leaves(n, 0, n, lanes, upper);
                    assert_eq!(spans.len(), lanes);
                    assert_eq!(spans[0].0, 0);
                    assert_eq!(spans.last().unwrap().1, n);
                    assert!(spans.iter().all(|&(a, b)| a < b));
                    assert!(spans.windows(2).all(|w| w[0].1 == w[1].0));
                }
                // Representative eight-task triangle: equal columns have a
                // heaviest lane of 69 entries; weighted splits cap it at 45.
                if n == 24 {
                    let spans = leaves(n, 0, n, 8, upper);
                    let heaviest = spans
                        .iter()
                        .map(|&(a, b)| {
                            (a..b)
                                .map(|j| if upper { j + 1 } else { n - j })
                                .sum::<usize>()
                        })
                        .max()
                        .unwrap();
                    assert!(heaviest <= 45);
                }
            }
        }
    }
}

#[cfg(test)]
mod tridiagonal_tests {
    use super::*;
    type R = F<4>; // 256-bit MPFR throughout, including the references.
    fn r(x: i32) -> R {
        if x < 0 {
            -num::<4>((-x) as usize)
        } else {
            num::<4>(x as usize)
        }
    }
    fn solve(d: &[R], e: &[R], s: R, b: &mut [R]) -> bool {
        let n = d.len();
        tridiag_solve(
            d,
            e,
            n,
            s,
            b,
            &mut vec![r(0); n],
            &mut vec![r(0); n],
            &mut vec![r(0); n],
            &mut vec![r(0); n],
        )
    }
    #[test]
    fn pivot_reproducer() {
        let mut b = vec![r(1), r(0), r(0)];
        assert!(solve(&[r(0), r(3), r(4)], &[r(2), r(1)], r(0), &mut b));
        println!("pivot first entry: {}", b[0]);
        assert!((b[0] + r(11) / r(16)).abs() <= R::epsilon());
    }
    #[test]
    fn sturm_zero_reproducer() {
        let count = sturm_count(&[r(0), r(0)], &[r(1)], 2, r(0));
        println!("zero-pivot count: {count}");
        assert_eq!(count, 1);
    }

    fn dense_solve(d: &[R], e: &[R], shift: R, rhs: &[R]) -> Option<Vec<R>> {
        let n = d.len();
        let mut a = vec![vec![r(0); n]; n];
        let mut b = rhs.to_vec();
        for i in 0..n {
            a[i][i] = d[i] - shift;
            if i + 1 < n {
                a[i][i + 1] = e[i];
                a[i + 1][i] = e[i];
            }
        }
        for k in 0..n {
            let pivot = (k..n)
                .max_by(|&i, &j| a[i][k].abs().partial_cmp(&a[j][k].abs()).unwrap())
                .unwrap();
            if a[pivot][k] == r(0) {
                return None;
            }
            a.swap(k, pivot);
            b.swap(k, pivot);
            for i in k + 1..n {
                let m = a[i][k] / a[k][k];
                for j in k + 1..n {
                    let v = m * a[k][j];
                    a[i][j] -= v;
                }
                let v = m * b[k];
                b[i] -= v;
            }
        }
        for i in (0..n).rev() {
            for j in i + 1..n {
                let v = a[i][j] * b[j];
                b[i] -= v;
            }
            b[i] /= a[i][i];
        }
        Some(b)
    }
    fn spectrum(d: &[R], e: &[R]) -> Vec<R> {
        let mut w = d.to_vec();
        let mut off = e.to_vec();
        off.resize(d.len(), r(0));
        tridiagonal_ql(&mut w, &mut off, &mut [], d.len()).unwrap();
        w.sort_by(|a, b| a.partial_cmp(b).unwrap());
        w
    }
    fn scale(d: &[R], e: &[R], shift: R) -> R {
        (0..d.len())
            .map(|i| {
                (d[i] - shift).abs()
                    + if i > 0 { e[i - 1].abs() } else { r(0) }
                    + if i + 1 < d.len() { e[i].abs() } else { r(0) }
            })
            .fold(r(0), |a, b| a.max(b))
    }
    fn sample(seed: &mut u64) -> R {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        r(((*seed >> 32) % 65) as i32 - 32) / r(16)
    }
    fn check_solve(d: &[R], e: &[R], shift: R, rhs: &[R]) {
        let reference = dense_solve(d, e, shift, rhs);
        let mut x = rhs.to_vec();
        assert_eq!(solve(d, e, shift, &mut x), reference.is_some());
        if let Some(reference) = reference {
            let norm = |v: &[R]| v.iter().fold(r(0), |a, b| a.max(b.abs()));
            let tol = R::epsilon() * r(16);
            let mut residual = r(0);
            for i in 0..d.len() {
                let mut v = (d[i] - shift) * x[i] - rhs[i];
                if i > 0 {
                    v += e[i - 1] * x[i - 1];
                }
                if i + 1 < d.len() {
                    v += e[i] * x[i + 1];
                }
                residual = residual.max(v.abs());
                assert!(
                    (x[i] - reference[i]).abs() <= tol * norm(&reference),
                    "dense disagreement n={} i={i}",
                    d.len()
                );
            }
            assert!(
                residual <= tol * (norm(rhs) + scale(d, e, shift) * norm(&x)),
                "residual n={}",
                d.len()
            );
        }
    }
    #[test]
    fn tridiagonal_solve_dense_reference() {
        // First two elimination steps both interchange; their multipliers
        // are nonzero, exercising propagation into the next superdiagonal.
        check_solve(
            &[r(1) / r(4), r(1), r(3), r(4)],
            &[r(2), r(3), r(1)],
            r(0),
            &[r(1), r(2), r(3), r(4)],
        );
        check_solve(&[r(0), r(0)], &[r(1)], r(1), &[r(1), r(0)]);
        check_solve(&[r(0)], &[], r(0), &[r(1)]);
        let mut seed = 4711;
        for n in 1..=12 {
            for trial in 0..12 {
                let d: Vec<_> = (0..n).map(|_| sample(&mut seed) / r(4)).collect();
                let e: Vec<_> = (0..n - 1)
                    .map(|i| {
                        if (i + trial) % 7 == 0 {
                            r(0)
                        } else {
                            sample(&mut seed)
                        }
                    })
                    .collect();
                let b: Vec<_> = (0..n).map(|_| sample(&mut seed)).collect();
                check_solve(&d, &e, r(1) / r(8), &b);
            }
        }
    }
    #[test]
    fn sturm_strict_counts() {
        for n in 1..=8 {
            for (x, want) in [(r(0), 0), (r(1), n), (-r(1), 0)] {
                assert_eq!(sturm_count(&vec![r(0); n], &vec![r(0); n - 1], n, x), want);
            }
        }
        // Exactly representable eigenvalues and recurrence zeros, including
        // disconnected blocks, avoid confusing QL rounding with strictness.
        for (d, e, queries) in [
            (vec![r(0), r(0)], vec![r(1)], vec![-r(1), r(0), r(1)]),
            (
                vec![r(1), r(1), r(2), r(2)],
                vec![r(0), r(0), r(0)],
                vec![r(1), r(2), r(3) / r(2)],
            ),
            (
                vec![r(0), r(0), r(0)],
                vec![r(3), r(4)],
                vec![-r(5), r(0), r(5)],
            ),
        ] {
            let w = spectrum(&d, &e);
            // QL may round exact roots; the integer spectra here are known.
            let exact: Vec<_> = if d.len() == 3 {
                vec![-r(5), r(0), r(5)]
            } else if d.len() == 2 {
                vec![-r(1), r(1)]
            } else {
                d.clone()
            };
            for (a, b) in w.iter().zip(&exact) {
                assert!((*a - *b).abs() <= R::epsilon() * r(32));
            }
            for x in queries {
                assert_eq!(
                    sturm_count(&d, &e, d.len(), x),
                    exact.iter().filter(|&&v| v < x).count()
                );
            }
        }
        for factor in [R::epsilon(), r(1), r(1) / R::epsilon()] {
            assert_eq!(sturm_count(&[r(0), r(0)], &[factor], 2, r(0)), 1);
        }
        let mut seed = 932;
        for n in 2..=12 {
            let d: Vec<_> = (0..n).map(|_| sample(&mut seed)).collect();
            let e: Vec<_> = (0..n - 1).map(|_| sample(&mut seed)).collect();
            let w = spectrum(&d, &e);
            for x in [r(0), -scale(&d, &e, r(0)), scale(&d, &e, r(0))]
                .into_iter()
                .chain(
                    w.windows(2)
                        .filter(|v| v[0] != v[1])
                        .map(|v| (v[0] + v[1]) / r(2)),
                )
            {
                assert_eq!(
                    sturm_count(&d, &e, n, x),
                    w.iter().filter(|&&v| v < x).count()
                );
            }
        }
    }
    fn check_eigenvalues(d: &[R], e: &[R]) {
        let reference = spectrum(d, e);
        for (k, &want) in reference.iter().enumerate() {
            let got = tridiag_eigval_at(d, e, d.len(), k, &mut vec![r(0); 6 * d.len()]);
            // Unconstrained RQI can converge to a different root. None is
            // the expected safe outcome, and the public caller must recover.
            let got = got.unwrap_or_else(|| caller_eigenvalue(d, e, k));
            assert!(
                (got - want).abs() <= R::epsilon() * scale(d, e, r(0)) * r(16),
                "eigenvalue n={} k={k}",
                d.len()
            );
        }
    }
    #[test]
    fn indexed_eigenvalues_and_safe_fallback_ql_reference() {
        let mut seed = 2718;
        for n in 1..=12 {
            let d: Vec<_> = (0..n).map(|_| sample(&mut seed)).collect();
            let e: Vec<_> = (0..n - 1).map(|_| sample(&mut seed)).collect();
            check_eigenvalues(&d, &e);
        }
        check_eigenvalues(&[r(1), r(1), r(2), r(2)], &[r(0), r(1) / r(2), r(0)]);
        let gap = r(1) / r(1024);
        check_eigenvalues(
            &[r(1), r(1) + gap, r(1) + gap * r(2)],
            &[gap / r(8), gap / r(4)],
        );
        let d = [r(1), r(1), r(2), r(2)];
        let e = [r(0); 3];
        for k in 0..4 {
            assert!(tridiag_eigval_at(&d, &e, 4, k, &mut vec![r(0); 24]).is_none());
            let steps = EIGVAL_STEPS.with(|v| v.get());
            assert!(steps.0 <= 32);
            assert_eq!(steps.1, 0);
            if k == 0 {
                assert_eq!(steps.0, 32);
            }
            assert_eq!(caller_eigenvalue(&d, &e, k), d[k]);
        }
        assert_eq!(spectrum(&d, &e), d);
    }

    fn caller_eigenvalue(d: &[R], e: &[R], k: usize) -> R {
        let n = d.len();
        let mut a = vec![r(0); n * n];
        for i in 0..n {
            a[i + i * n] = d[i];
            if i + 1 < n {
                a[i + (i + 1) * n] = e[i];
                a[i + 1 + i * n] = e[i];
            }
        }
        let mut w = vec![r(0); n];
        let lwork = n * n + 12 * n;
        let mut work = vec![r(0); lwork];
        let mut iw = vec![0; n];
        let (mut count, mut info) = (0, 0);
        R::xsyevr(
            b'N',
            b'I',
            b'U',
            n as i32,
            &mut a,
            n as i32,
            r(0),
            r(0),
            (k + 1) as i32,
            (k + 1) as i32,
            r(0),
            &mut count,
            &mut w,
            &mut [],
            1,
            &mut [],
            &mut work,
            lwork as i32,
            &mut iw,
            n as i32,
            &mut info,
        );
        assert_eq!(info, 0);
        assert_eq!(count, 1);
        w[0]
    }
    #[test]
    fn rqi_wrong_root_safeguard() {
        let d = [
            -r(21) / r(16),
            r(3) / r(2),
            -r(3) / r(8),
            r(5) / r(16),
            -r(9) / r(16),
        ];
        let e = [r(5) / r(8), -r(17) / r(16), r(5) / r(4), -r(9) / r(16)];
        let got = tridiag_eigval_at(&d, &e, 5, 0, &mut vec![r(0); 30]);
        println!("n=5 Some={}", got.is_some());
        let steps = EIGVAL_STEPS.with(|v| v.get());
        println!("n=5 RQI: bisections={}, iterations={}", steps.0, steps.1);
        let got = got.expect("the safeguarded RQI must recover this root");
        assert!((got - spectrum(&d, &e)[0]).abs() <= R::epsilon() * scale(&d, &e, r(0)) * r(16));
        assert!(
            (caller_eigenvalue(&d, &e, 0) - spectrum(&d, &e)[0]).abs()
                <= R::epsilon() * scale(&d, &e, r(0)) * r(16)
        );
    }

    #[test]
    fn indexed_fixed_random_some_rate() {
        let mut seed = 928371;
        let (mut some, mut total) = (0, 0);
        for case in 0..64 {
            let n = 2 + case % 11;
            let d: Vec<_> = (0..n).map(|_| sample(&mut seed)).collect();
            let e: Vec<_> = (0..n - 1).map(|_| sample(&mut seed)).collect();
            let w = spectrum(&d, &e);
            for k in 0..n {
                total += 1;
                let got = tridiag_eigval_at(&d, &e, n, k, &mut vec![r(0); 6 * n]);
                let got = if let Some(got) = got {
                    some += 1;
                    got
                } else {
                    caller_eigenvalue(&d, &e, k)
                };
                assert!(
                    (got - w[k]).abs() <= R::epsilon() * scale(&d, &e, r(0)) * r(16),
                    "case={case}, k={k}"
                );
            }
        }
        println!("indexed fixed random Some-rate: {some}/{total}");
        // The unsafeguarded driver returned only 370/439 on this fixed set.
        assert!(some >= 438, "unexpected loss of verified fast-path results");
    }
}
