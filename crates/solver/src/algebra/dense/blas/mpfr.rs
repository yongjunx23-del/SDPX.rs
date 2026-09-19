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
use sdpx_arithmetic::{EncodeSide, MpFloat, RnsPlan, Scalar};
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
    // Exact residue accumulation replaces the FMA chain when the operand
    // window admits a plan; reconstruction rounds once at the destination.
    let rns = (alpha != F::<N>::zero() && k > 0)
        .then(|| RnsPlan::for_pair(a, b, k as usize))
        .flatten()
        .filter(|plan| {
            plan.profitable(
                k as usize,
                m as usize * n as usize,
                a.len() + b.len(),
                N,
            )
        })
        .and_then(|plan| {
            plan.encode(a, EncodeSide::A)
                .zip(plan.encode(b, EncodeSide::B))
                .map(|(ra, rb)| (plan, ra, rb))
        });
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
                v = match &rns {
                    Some((plan, ra, rb)) => {
                        plan.dot(ra, a0, da, rb, b0, db, k as usize)
                    }
                    None => {
                        F::dot_fma((0..k as usize).map(|p| (&a[a0 + p * da], &b[b0 + p * db])))
                    }
                };
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
    // Same operand on both sides: one encode serves both residue columns.
    let rns = (alpha != F::<N>::zero() && k > 0)
        .then(|| RnsPlan::for_pair(a, a, k as usize))
        .flatten()
        .filter(|plan| {
            plan.profitable(
                k as usize,
                n as usize * (n as usize + 1) / 2,
                a.len(),
                N,
            )
        })
        .and_then(|plan| plan.encode(a, EncodeSide::A).map(|ra| (plan, ra)));
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
                v = match &rns {
                    Some((plan, ra)) => plan.dot(ra, a0, da, ra, b0, db, k as usize),
                    None => {
                        F::dot_fma((0..k as usize).map(|p| (&a[a0 + p * da], &a[b0 + p * db])))
                    }
                };
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


#[path = "mpfr_svd.rs"]
mod svd;
#[path = "mpfr_eigen.rs"]
mod eigen;
#[cfg(test)]
use eigen::*;
use svd::*;

#[cfg(test)]
#[path = "mpfr_syrk_partition_tests.rs"]
mod syrk_partition_tests;

#[cfg(test)]
#[path = "mpfr_tridiagonal_tests.rs"]
mod tridiagonal_tests;
