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

// Minimum trailing column/element count before a kernel offers independent
// column work to the ambient solver pool while `inner_parallel` is active;
// smaller tails stay serial to avoid dispatch overhead.
const PAR_COLS: usize = 4;

// True when this kernel may re-offer independent column work to the ambient
// Rayon pool. Only solver-pool lanes set the TLS gate, so par iterators here
// join the solver pool rather than the global pool.
fn inner_par() -> bool {
    sdpx_arithmetic::inner_parallel::active()
}

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
    } else if alpha == F::one() {
        x
    } else if alpha == -F::one() {
        -x
    } else {
        alpha * x
    };
    if beta == F::zero() {
        p
    } else if beta == F::one() {
        p + y
    } else if beta == -F::one() {
        p - y
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
    } else if inner_par() && columns >= PAR_COLS {
        // Called on a solver-pool lane without an explicit pool: offer the
        // same complete-column tiles to the ambient pool.
        let tile = columns.div_ceil(4 * rayon::current_num_threads()).max(1);
        c[..used]
            .par_chunks_mut(tile * ld)
            .enumerate()
            .for_each(|(t, values)| {
                for (j, values) in values.chunks_mut(ld).enumerate() {
                    column(t * tile + j, &mut values[..rows]);
                }
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
    packed: bool,
    column: &(impl Fn(usize, &mut [F<N>]) + Sync),
) {
    if lanes == 1 {
        for j in begin..end {
            let start = if packed {
                j * (j + 1) / 2 - begin * (begin + 1) / 2
            } else {
                (j - begin) * ld
            };
            let len = if packed { j + 1 } else { n };
            column(j, &mut c[start..start + len]);
        }
    } else {
        let cut = syrk_cut(n, begin, end, lanes, upper);
        let offset = if packed {
            cut * (cut + 1) / 2 - begin * (begin + 1) / 2
        } else {
            (cut - begin) * ld
        };
        let (left, right) = c.split_at_mut(offset);
        rayon::join(
            || split_syrk_columns(left, n, ld, begin, cut, lanes / 2, upper, packed, column),
            || split_syrk_columns(right, n, ld, cut, end, lanes - lanes / 2, upper, packed, column),
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
    let packed = upper && ld == n && c.len() == n * (n + 1) / 2;
    let used = if packed { c.len() } else { (n - 1) * ld + n };
    if let Some((pool, tile)) = parallel.filter(|(p, t)| p.current_num_threads() > 1 && *t > 0) {
        // Retain the existing configured number of column tasks, changing
        // only their boundaries. Splits borrow disjoint slices; no scratch.
        let lanes = n.div_ceil(tile.min(n));
        pool.install(|| {
            split_syrk_columns(
                &mut c[..used],
                n,
                ld,
                0,
                n,
                lanes,
                upper,
                packed,
                &column,
            )
        });
    } else if inner_par() && n >= PAR_COLS {
        // Ambient-pool fallback on a solver-pool lane; split_syrk_columns
        // keeps the triangular load balanced across lanes.
        let lanes = (4 * rayon::current_num_threads()).min(n).max(1);
        split_syrk_columns(
            &mut c[..used],
            n,
            ld,
            0,
            n,
            lanes,
            upper,
            packed,
            &column,
        );
    } else {
        split_syrk_columns(&mut c[..used], n, ld, 0, n, 1, upper, packed, &column);
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
    if alpha != F::<N>::zero()
        && (residue_blas_profitable::<N>(m as usize, n as usize, k as usize)
            || (N >= 16 && m == n && n == k && m >= 12))
    {
        let (mu, nu) = (m as usize, n as usize);
        let exact_product = |out: &mut [F<N>]| {
            super::rns_blas::gemm(
                ta,
                tb,
                mu,
                nu,
                k as usize,
                a,
                lda as usize,
                b,
                ldb as usize,
                false,
                parallel.map(|(pool, _)| pool),
                out,
                None,
            )
        };
        if alpha == F::<N>::one() && beta == F::<N>::zero() && ldc == m {
            // Most scaling products overwrite a compact matrix. Reconstruct
            // directly into it, avoiding a full MPFR temporary and copy.
            if exact_product(&mut c[..mu * nu]) {
                return;
            }
        } else {
            let mut product = vec![F::<N>::zero(); mu * nu];
            if exact_product(&mut product) {
                for j in 0..nu {
                    for i in 0..mu {
                        let dst = &mut c[i + j * ldc as usize];
                        *dst = axpby(alpha, product[i + j * mu], beta, *dst);
                    }
                }
                return;
            }
        }
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
                // Exact accumulation, rounded once at the destination.
                v = F::dot_fma((0..k as usize).map(|p| (&a[a0 + p * da], &b[b0 + p * db])));
            }
            column[i] = axpby(alpha, v, beta, column[i]);
        }
    };
    output_columns(c, m as usize, n as usize, ldc as usize, parallel, column);
}
/// Structural gate for the exact residue-BLAS product: enough multiply work
/// per output to amortize encode and CRT reconstruction. Both paths return the
/// same correctly rounded values. Measured serial square GEMM break-even on
/// AMD EPYC 7742 (2026-09-30): the residue kernel wins from 24³ (1.14× at 512
/// bits, 1.20× at 768, 1.50× at 1024) and is 1.7–2.4× faster at 40³.
fn residue_blas_profitable<const N: usize>(m: usize, n: usize, k: usize) -> bool {
    N >= 4 && k >= 24 && m * n >= 576
}

impl<const N: usize> XgemmScalar for F<N> {
    fn residue_blas_applies(m: usize, n: usize, k: usize) -> bool {
        residue_blas_profitable::<N>(m, n, k)
    }
    fn xsvec_quadratic_exact(
        h: usize,
        kmax: usize,
        q: &[Self],
        abs_q: bool,
        x: &[Self],
        sqrt2: Self,
        pool: Option<&rayon::ThreadPool>,
        out: &mut [Self],
        cache_q: Option<&mut super::ResidueCache>,
    ) -> bool {
        // A cached dense quadratic form reconstructs only kmax outputs,
        // so it amortizes residue work at smaller shapes than a full GEMM.
        let cached_dense = N >= 4
            && h >= 12
            && kmax >= 16
            && sqrt2 == num_traits::FromPrimitive::from_u8(2).unwrap()
            && cache_q.is_some();
        (residue_blas_profitable::<N>(h, kmax, h) || cached_dense)
            && super::rns_blas::svec_quadratic(h, kmax, q, abs_q, x, sqrt2, pool, out, cache_q)
    }
    fn xsymmetric_bilinear_exact(
        h: usize,
        columns: usize,
        q: &[Self],
        x: &[Self],
        pairs: &[(usize, usize)],
        out: &mut [Self],
        cache_q: &mut super::ResidueCache,
    ) -> bool {
        N >= 4
            && h >= 12
            && columns >= 16
            && super::rns_blas::symmetric_bilinear(h, columns, q, x, pairs, out, cache_q)
    }
    fn xcongruence_exact(
        ta: u8,
        m: usize,
        k: usize,
        a: &[Self],
        lda: usize,
        x: &[Self],
        ldx: usize,
        c: &mut [Self],
        upper_only: bool,
        pool: Option<&rayon::ThreadPool>,
        cache_a: Option<&mut super::ResidueCache>,
    ) -> bool {
        residue_blas_profitable::<N>(m, m, k)
            && super::rns_blas::congruence(ta, m, k, a, lda, x, ldx, upper_only, pool, c, cache_a)
    }
    fn diag_congruence_upper_exact(
        m: usize,
        k: usize,
        a: &[Self],
        d: &[Self],
        c: &mut [Self],
        pool: Option<&rayon::ThreadPool>,
        cache_a: &mut super::ResidueCache,
    ) -> bool {
        super::rns_blas::diag_congruence(m, k, a, d, true, pool, c, cache_a)
    }
    fn xgemm_upper_exact(
        ta: u8,
        tb: u8,
        m: usize,
        n: usize,
        k: usize,
        a: &[Self],
        lda: usize,
        b: &[Self],
        ldb: usize,
        c: &mut [Self],
        pool: Option<&rayon::ThreadPool>,
        cache_b: Option<&mut super::ResidueCache>,
    ) -> bool {
        residue_blas_profitable::<N>(m, n, k)
            && super::rns_blas::gemm(ta, tb, m, n, k, a, lda, b, ldb, true, pool, c, cache_b)
    }
    fn xgemm_blocks_upper_exact(
        m: usize,
        blocks: &[(&[Self], &[Self], &[usize])],
        c: &mut [Self],
        pool: Option<&rayon::ThreadPool>,
    ) -> bool {
        let rows: usize = blocks.iter().map(|b| b.0.len() / b.2.len().max(1)).sum();
        // Every block row is one rank-one term of many outputs, so exact dots
        // cost far more per term than residue products already at 128 bits.
        N >= 2 && rows >= 24 && m * m >= 576
            && super::rns_blas::gemm_blocks_upper(m, blocks, c, pool)
    }
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
    assert!(tri(u) && trans(t) && n >= 0 && k >= 0);
    let packed = upper(u) == b'U' && ldc == n && c.len() == (n as usize) * (n as usize + 1) / 2;
    assert!(packed || valid(c.len(), n, n, ldc));
    assert!(valid(
        a.len(),
        if upper(t) == b'N' { n } else { k },
        if upper(t) == b'N' { k } else { n },
        lda
    ));
    if n == 0 {
        return;
    }
    // The common Gram update writes a compact upper triangle. Reuse the
    // exact residue-BLAS product, which shares this operand's encoding and
    // rounds each dot once, just like the per-entry exact path below.
    if upper(u) == b'U'
        && ldc == n
        && alpha == F::<N>::one()
        && beta == F::<N>::zero()
        && residue_blas_profitable::<N>(n as usize, n as usize, k as usize)
        && super::rns_blas::gemm(
            t,
            if upper(t) == b'N' { b'T' } else { b'N' },
            n as usize,
            n as usize,
            k as usize,
            a,
            lda as usize,
            a,
            lda as usize,
            true,
            parallel.map(|(pool, _)| pool),
            c,
            None,
        )
    {
        return;
    }
    let column = |j: usize, column: &mut [F<N>]| {
        for i in 0..n as usize {
            if (upper(u) == b'U' && i > j) || (upper(u) == b'L' && i < j) {
                continue;
            }
            let mut v = F::<N>::zero();
            if alpha != F::<N>::zero() && k > 0 {
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
                v = if da == 1 {
                    F::dot_slices(&a[a0..a0 + k as usize], &a[b0..b0 + k as usize])
                } else {
                    F::dot_fma((0..k as usize).map(|p| (&a[a0 + p * da], &a[b0 + p * db])))
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
        let (n, k) = (n as usize, k as usize);
        let (lda, ldb, ldc) = (lda as usize, ldb as usize, ldc as usize);
        // Output columns are disjoint and read only a/b, so they can join
        // the ambient pool while the solver's inner-parallel gate is active.
        let column = |j: usize, cj: &mut [Self]| {
            for i in 0..n {
                if (upper(u) == b'U' && i > j) || (upper(u) == b'L' && i < j) {
                    continue;
                }
                let mut v = Self::zero();
                if alpha != Self::zero() {
                    for p in 0..k {
                        v += at(a, lda, i, p, t) * at(b, ldb, j, p, t)
                            + at(b, ldb, i, p, t) * at(a, lda, j, p, t);
                    }
                }
                cj[i] = axpby(alpha, v, beta, cj[i]);
            }
        };
        if inner_par() && n >= PAR_COLS && ldc == n {
            let tile = n.div_ceil(4 * rayon::current_num_threads()).max(1);
            c[..n * ldc]
                .par_chunks_mut(tile * ldc)
                .enumerate()
                .for_each(|(ti, chunk)| {
                    for (jc, cj) in chunk.chunks_mut(ldc).enumerate() {
                        column(ti * tile + jc, cj);
                    }
                });
        } else {
            for j in 0..n {
                column(j, &mut c[j * ldc..(j + 1) * ldc]);
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
            let tail = n - j - 1;
            // Each trailing update reads only completed columns and writes a
            // disjoint element, so it can join the ambient pool under the
            // solver's inner-parallel gate.
            if inner_par() && tail >= PAR_COLS && upper(u) == b'L' {
                let (done, rest) = a.split_at_mut(j * ld);
                rest[j + 1..n]
                    .par_iter_mut()
                    .enumerate()
                    .for_each(|(t, q)| {
                        let i = j + 1 + t;
                        let mut v = *q;
                        for k in 0..j {
                            v = (-done[i + k * ld]).mul_add(done[j + k * ld], v);
                        }
                        *q = v / d;
                    });
            } else if inner_par() && tail >= PAR_COLS {
                let (head, right) = a.split_at_mut((j + 1) * ld);
                right.par_chunks_mut(ld).for_each(|col| {
                    let mut v = col[j];
                    for k in 0..j {
                        v = (-col[k]).mul_add(head[k + j * ld], v);
                    }
                    col[j] = v / d;
                });
            } else {
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
        let nrhs = nrhs as usize;
        if n == 0 || nrhs == 0 {
            return;
        }
        let solve_column = |b: &mut [Self]| {
            for i in 0..n {
                let mut v = b[i];
                for k in 0..i {
                    let a_val = if upper(u) == b'L' {
                        a[i + k * ld]
                    } else {
                        a[k + i * ld]
                    };
                    let b_val = b[k];
                    v = (-a_val).mul_add(b_val, v);
                }
                b[i] = v / a[i + i * ld];
            }
            for i in (0..n).rev() {
                let mut v = b[i];
                for k in i + 1..n {
                    let a_val = if upper(u) == b'L' {
                        a[k + i * ld]
                    } else {
                        a[i + k * ld]
                    };
                    let b_val = b[k];
                    v = (-a_val).mul_add(b_val, v);
                }
                b[i] = v / a[i + i * ld];
            }
        };
        // One task owns each complete RHS; the dependent triangular sweeps
        // inside a column retain exactly the serial arithmetic order.
        let parallel = inner_par()
            && rayon::current_num_threads() > 1
            && nrhs >= PAR_COLS
            && (n as u128) * (n as u128) * (nrhs as u128) >= 4096;
        if parallel {
            #[cfg(test)]
            POOLED_POTRS_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let tile = nrhs.div_ceil(4 * rayon::current_num_threads()).max(1);
            b[..(nrhs - 1) * lb + n]
                .par_chunks_mut(tile * lb)
                .for_each(|tile| {
                    for column in tile.chunks_mut(lb) {
                        solve_column(&mut column[..n]);
                    }
                });
        } else {
            for r in 0..nrhs {
                solve_column(&mut b[r * lb..r * lb + n]);
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

#[path = "mpfr_eigen.rs"]
mod eigen;
#[path = "mpfr_svd.rs"]
mod svd;
#[cfg(test)]
use eigen::*;
use svd::*;

#[cfg(test)]
#[path = "mpfr_syrk_partition_tests.rs"]
mod syrk_partition_tests;

#[cfg(test)]
#[path = "mpfr_tridiagonal_tests.rs"]
mod tridiagonal_tests;

#[cfg(test)]
static POOLED_POTRS_CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
#[cfg(test)]
#[path = "mpfr_potrs_tests.rs"]
mod potrs_tests;
