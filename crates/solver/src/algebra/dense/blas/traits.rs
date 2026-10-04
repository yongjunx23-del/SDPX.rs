#![cfg_attr(rustfmt, rustfmt_skip)]
#![allow(clippy::too_many_arguments)]

// Standard imports via blas-lapack-rs crates. The solver crate has no Python
// runtime dependency; the public language boundary is the separate native ABI.
extern crate blas_src;
extern crate lapack_src;
use lapack::*;
use blas::*;

pub trait BlasFloatT:
    private::BlasFloatSealed
    + XsyevrScalar
    + XpotrfScalar
    + XpotrsScalar
    + XgesddScalar
    + XgesvdScalar
    + XgemmScalar
    + XsymvScalar
    + XsyrkScalar
    + Xsyr2kScalar
    + XgesvScalar
{}

	impl BlasFloatT for f32 {}
impl BlasFloatT for f64 {}

mod private {
  pub trait BlasFloatSealed {}
  impl BlasFloatSealed for f32 {}
  impl BlasFloatSealed for f64 {}
}


// --------------------------------------
// ?syevr : Symmetric eigen decomposition
// --------------------------------------

pub trait XsyevrScalar: Sized {
    fn xsyevr(
        jobz: u8, range: u8, uplo: u8, n: i32, a: &mut [Self], lda: i32, vl: Self, vu: Self, il: i32, iu: i32,
        abstol: Self, m: &mut i32, w: &mut [Self], z: &mut [Self], ldz: i32, isuppz: &mut [i32],
        work: &mut [Self], lwork: i32, iwork: &mut [i32], liwork: i32, info: &mut i32,
    );
}

macro_rules! impl_blas_xsyevr {
    ($T:ty, $XSYEVR:path) => {
        impl XsyevrScalar for $T {
            fn xsyevr(
                jobz: u8, range: u8, uplo: u8, n: i32, a: &mut [Self], lda: i32, vl: Self, vu: Self, il: i32, iu: i32,
                abstol: Self, m: &mut i32, w: &mut [Self], z: &mut [Self], ldz: i32, isuppz: &mut [i32],
                work: &mut [$T], lwork: i32, iwork: &mut [i32], liwork: i32, info: &mut i32,
            ) {
                unsafe{
                    $XSYEVR(
                        jobz, range, uplo, n, a, lda, vl, vu, il, iu, abstol, m,
                        w, z, ldz, isuppz, work, lwork, iwork, liwork, info,
                    );
                }
            }
        }
    };
}

impl_blas_xsyevr!(f32, ssyevr);
impl_blas_xsyevr!(f64, dsyevr);

// --------------------------------------
// ?potrf : Cholesky decomposition
// --------------------------------------

pub trait XpotrfScalar: Sized {
    fn xpotrf(
        uplo: u8, n: i32, a: &mut [Self], lda: i32, info: &mut i32
    );
}

pub trait XpotrsScalar: Sized {
    fn xpotrs(
        uplo: u8, n: i32, nrhs: i32, a: &[Self], lda: i32,  b: &mut [Self], ldb: i32, info: &mut i32
    );
    /// Solve L*X=B (or L'*X=B), with a square lower-triangular factor.
    /// MPFR uses the shared fixed-precision arithmetic; native types use TRSM.
    fn xtrsm_lower(n: usize, a: &[Self], b: &mut [Self], transpose: bool)
    where Self: crate::algebra::FloatT {
        use rayon::prelude::*;
        if n == 0 { return; }
        assert_eq!(a.len(), n*n);
        assert_eq!(b.len() % n, 0);
        let solve = |column: &mut [Self]| {
            for step in 0..n {
                let i = if transpose { n-1-step } else { step };
                let mut v = column[i];
                let range = if transpose { i+1..n } else { 0..i };
                for k in range {
                    let av = if transpose { a[k+i*n] } else { a[i+k*n] };
                    v = (-av).mul_add(column[k], v);
                }
                column[i] = v / a[i+i*n];
            }
        };
        if sdpx_arithmetic::inner_parallel::active() && n*b.len() >= 16384 {
            b.par_chunks_mut(n).for_each(solve);
        } else {
            b.chunks_mut(n).for_each(solve);
        }
    }
}

macro_rules! impl_blas_xpotrfs{
    ($T:ty, $XPOTRF:path, $XPOTRS:path, $XTRSM:path) => {
        impl XpotrfScalar for $T {
            fn xpotrf(
                uplo: u8, n: i32, a: &mut [Self], lda: i32, info: &mut i32
            ) {
                unsafe{
                    $XPOTRF(
                        uplo, n, a, lda, info
                    );
                }
            }
        }
        impl XpotrsScalar for $T {
            fn xtrsm_lower(n: usize, a: &[Self], b: &mut [Self], transpose: bool) {
                if n == 0 { return; }
                assert_eq!(a.len(), n*n);
                assert_eq!(b.len() % n, 0);
                let rows = i32::try_from(n).unwrap();
                let cols = i32::try_from(b.len()/n).unwrap();
                unsafe { $XTRSM(b'L', b'L', if transpose { b'T' } else { b'N' },
                    b'N', rows, cols, 1.0, a, rows, b, rows); }
            }
            fn xpotrs(
                uplo: u8, n: i32, nrhs: i32, a: &[Self], lda: i32, b: &mut [Self], ldb: i32, info: &mut i32
            ) {
                unsafe{
                    $XPOTRS(
                        uplo, n, nrhs, a, lda, b, ldb, info
                    );
                }
            }
        }
    };
}

impl_blas_xpotrfs!(f32, spotrf, spotrs, strsm);
impl_blas_xpotrfs!(f64, dpotrf, dpotrs, dtrsm);


// --------------------------------------
// ?gesdd : SVD (divide and conquer method)
// --------------------------------------

pub trait XgesddScalar: Sized {
    fn xgesdd(
        jobz: u8, m: i32, n: i32, a: &mut [Self], lda: i32,
        s: &mut [Self], u: &mut [Self], ldu: i32, vt: &mut [Self], ldvt: i32,
        work: &mut [Self], lwork: i32, iwork: &mut [i32], info: &mut i32
    );
}

macro_rules! impl_blas_xgesdd{
    ($T:ty, $XGESDD:path) => {
        impl XgesddScalar for $T {
            fn xgesdd(
                jobz: u8, m: i32, n: i32, a: &mut [Self], lda: i32,
                s: &mut [Self], u: &mut [Self], ldu: i32, vt: &mut [Self], ldvt: i32,
                work: &mut [Self], lwork: i32, iwork: &mut [i32], info: &mut i32
            ) {
                unsafe{
                    $XGESDD(
                        jobz, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, iwork, info
                    );
                }
            }
        }
    };
}

impl_blas_xgesdd!(f32, sgesdd);
impl_blas_xgesdd!(f64, dgesdd);


// --------------------------------------
// ?gesvd : SVD (QR method)
// --------------------------------------

pub trait XgesvdScalar: Sized {
    fn xgesvd(
        jobu: u8, jobvt: u8,m: i32, n: i32, a: &mut [Self], lda: i32,
        s: &mut [Self], u: &mut [Self], ldu: i32, vt: &mut [Self], ldvt: i32,
        work: &mut [Self], lwork: i32, info: &mut i32
    );
}

macro_rules! impl_blas_xgesvd{
    ($T:ty, $XGESVD:path) => {
        impl XgesvdScalar for $T {
            fn xgesvd(
                jobu: u8, jobvt: u8,m: i32, n: i32, a: &mut [Self], lda: i32,
                s: &mut [Self], u: &mut [Self], ldu: i32, vt: &mut [Self], ldvt: i32,
                work: &mut [Self], lwork: i32, info: &mut i32
            ) {
                unsafe{
                    $XGESVD(
                        jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, info
                    );
                }
            }
        }
    };
}

impl_blas_xgesvd!(f32, sgesvd);
impl_blas_xgesvd!(f64, dgesvd);


// --------------------------------------
// ?gemm : matrix matrix multiply
// --------------------------------------

pub trait XgemmScalar: Sized {
    fn xgemm(
        transa: u8, transb: u8, m: i32, n: i32, k: i32, alpha: Self, a: &[Self],
        lda: i32, b: &[Self], ldb: i32, beta: Self, c: &mut [Self], ldc: i32
    );
    // Native BLAS retains one opaque call; MPFR overrides with disjoint output
    // tiles in this caller-owned pool, without changing scalar accumulation.
    fn xgemm_pool(
        transa: u8, transb: u8, m: i32, n: i32, k: i32, alpha: Self, a: &[Self],
        lda: i32, b: &[Self], ldb: i32, beta: Self, c: &mut [Self], ldc: i32,
        _pool: &rayon::ThreadPool, _column_tile: usize
    ) { Self::xgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc); }
    // Upper triangle (`i <= j`) of `op(a)·op(b)` into `c` (ldc = m) when the
    // exact product is known to be symmetric. Only the exact residue-BLAS
    // kernel implements it; `false` means nothing was written.
    // `cache_b` optionally keeps `b`'s residues when it is a constant operand.
    fn xgemm_upper_exact(
        _transa: u8, _transb: u8, _m: usize, _n: usize, _k: usize, _a: &[Self], _lda: usize,
        _b: &[Self], _ldb: usize, _c: &mut [Self], _pool: Option<&rayon::ThreadPool>,
        _cache_b: Option<&mut ResidueCache>
    ) -> bool { false }
    // Upper triangle of `aᵀ·diag(d)·a` for a constant column-major `k × m`
    // operand `a` (its residues kept in `cache_a`), each entry the exact sum
    // rounded once. Reset `cache_a` whenever `a` changes.
    // Only the exact residue kernel implements it; `false`
    // means nothing was written.
    fn diag_congruence_upper_exact(
        _m: usize, _k: usize, _a: &[Self], _d: &[Self], _c: &mut [Self],
        _pool: Option<&rayon::ThreadPool>, _cache_a: &mut ResidueCache
    ) -> bool { false }
    // Whether `xgemm_upper_exact` (and the matching `xgemm` path) uses the
    // exact residue-BLAS kernel for this shape.
    fn residue_blas_applies(_m: usize, _n: usize, _k: usize) -> bool { false }
    // `v[k] = Σ_{i≤j} c_ij·x_t·q_ik·q_jk` (svec `x`, `c_ij = sqrt2` off the
    // diagonal) for `q` of size h × kmax, rounded once from the exact value.
    // Passing scale 2 instead of sqrt2 accepts an unscaled packed symmetric X.
    // `abs_q` evaluates the form on |q| while reading (and caching) q itself.
    fn xsvec_quadratic_exact(
        _h: usize, _kmax: usize, _q: &[Self], _abs_q: bool, _x: &[Self], _sqrt2: Self,
        _pool: Option<&rayon::ThreadPool>, _out: &mut [Self], _cache_q: Option<&mut ResidueCache>
    ) -> bool where Self: Sized { false }
    // Selected q_aᵀ X q_b values for symmetric, unscaled packed X, rounded
    // once from each exact bilinear form. Use the outputs only on success.
    fn xsymmetric_bilinear_exact(
        _h: usize, _columns: usize, _q: &[Self], _x: &[Self],
        _pairs: &[(usize, usize)], _out: &mut [Self], _cache_q: &mut ResidueCache
    ) -> bool { false }
    // `op(a)·x·op(a)ᵀ` (m × m, x is k × k) rounded once from the exact value
    // into `c` (ldc = m); upper triangle only when `upper_only`.
    fn xcongruence_exact(
        _transa: u8, _m: usize, _k: usize, _a: &[Self], _lda: usize, _x: &[Self], _ldx: usize,
        _c: &mut [Self], _upper_only: bool, _pool: Option<&rayon::ThreadPool>,
        _cache_a: Option<&mut ResidueCache>
    ) -> bool { false }

}

macro_rules! impl_blas_gemm {
    ($T:ty, $XGEMM:path) => {
        impl XgemmScalar for $T {
            fn xgemm(
                transa: u8, transb: u8, m: i32, n: i32, k: i32, alpha: Self, a: &[Self],
                lda: i32, b: &[Self], ldb: i32, beta: Self, c: &mut [Self], ldc: i32
            ) {
                unsafe{
                    $XGEMM(
                        transa, transb, m, n, k, alpha, a,
                        lda, b, ldb, beta, c, ldc
                    );
                }
            }
        }
    };
}

impl_blas_gemm!(f32, sgemm);
impl_blas_gemm!(f64, dgemm);

// --------------------------------------
// ?symv : matrix vector multiply (symmetric)
// --------------------------------------

pub trait XsymvScalar: Sized {
    fn xsymv(
        uplo: u8, n: i32, alpha: Self, a: &[Self], lda: i32,
        x: &[Self], incx: i32, beta: Self, y: &mut [Self], incy: i32
    );
}


macro_rules! impl_blas_gsymv {
    ($T:ty, $XSYMV:path) => {
        impl XsymvScalar for $T {
            fn xsymv(
                uplo: u8, n: i32, alpha: Self, a: &[Self], lda: i32,
                x: &[Self], incx: i32, beta: Self, y: &mut [Self], incy: i32
            ) {
                unsafe{
                    $XSYMV(
                        uplo, n, alpha, a, lda, x, incx, beta, y, incy
                    );
                }
            }
        }
    };
}

impl_blas_gsymv!(f32, ssymv);
impl_blas_gsymv!(f64, dsymv);


// --------------------------------------
// ?syrk : symmetric rank k update
// --------------------------------------

pub trait XsyrkScalar: Sized {
    fn xsyrk(
        uplo: u8, trans: u8, n: i32, k: i32, alpha: Self,
        a: &[Self], lda: i32, beta: Self, c: &mut [Self], ldc: i32
    );
    fn xsyrk_pool(
        uplo: u8, trans: u8, n: i32, k: i32, alpha: Self,
        a: &[Self], lda: i32, beta: Self, c: &mut [Self], ldc: i32,
        _pool: &rayon::ThreadPool, _column_tile: usize
    ) { Self::xsyrk(uplo, trans, n, k, alpha, a, lda, beta, c, ldc); }

}


macro_rules! impl_blas_gsyrk {
    ($T:ty, $XSYRK:path) => {
        impl XsyrkScalar for $T {
            fn xsyrk(
                uplo: u8, trans: u8, n: i32, k: i32, alpha: Self,
                a: &[Self], lda: i32, beta: Self, c: &mut [Self], ldc: i32
            ) {
                unsafe{
                    $XSYRK(
                        uplo, trans, n, k, alpha, a, lda, beta, c, ldc
                    );
                }
            }
        }
    };
}

impl_blas_gsyrk!(f32, ssyrk);
impl_blas_gsyrk!(f64, dsyrk);

// --------------------------------------
// ?syrk : symmetric rank 2k update
// --------------------------------------

pub trait Xsyr2kScalar: Sized {
    fn xsyr2k(
        uplo: u8, trans: u8, n: i32, k: i32, alpha: Self, a: &[Self], lda: i32,
        b: &[Self], ldb: i32, beta: Self, c: &mut [Self], ldc: i32
    );
}


macro_rules! impl_blas_gsyr2k {
    ($T:ty, $XSYR2K:path) => {
        impl Xsyr2kScalar for $T {
            fn xsyr2k(
                uplo: u8, trans: u8, n: i32, k: i32, alpha: Self, a: &[Self], lda: i32,
                b: &[Self], ldb: i32, beta: Self, c: &mut [Self], ldc: i32
            ) {
                unsafe{
                    $XSYR2K(
                        uplo, trans, n, k, alpha, a, lda,
                        b, ldb, beta, c, ldc
                    );
                }
            }
        }
    };
}

impl_blas_gsyr2k!(f32, ssyr2k);
impl_blas_gsyr2k!(f64, dsyr2k);


// --------------------------------------
// ?gesv : Generalized (LU) linear solve, multiple right hand side
// --------------------------------------

pub trait XgesvScalar: Sized {
    fn xgesv(
        n: i32, nrhs: i32, a: &mut [Self], lda: i32, ipiv: &mut [i32],
                b: &mut [Self], ldb: i32, info:&mut i32,
    );
}

macro_rules! impl_blas_xgesv{
    ($T:ty, $XGESV:path) => {
        impl XgesvScalar for $T {
            fn xgesv(
                n: i32, nrhs: i32, a: &mut [Self], lda: i32, ipiv: &mut [i32],
                b: &mut [Self], ldb: i32, info:&mut i32,
            ) {
                unsafe{
                    $XGESV(
                        n, nrhs, a, lda, ipiv, b, ldb, info,
                    );
                }
            }
        }
    };
}

impl_blas_xgesv!(f32, sgesv);
impl_blas_xgesv!(f64, dgesv);
// Inline MPFR precision modes share the dense provider boundary.
#[path = "mpfr.rs"]
mod mpfr;
#[path = "rns_blas.rs"]
mod rns_blas;
// Integration tests include this file by `#[path]` and use only part of it.
#[allow(unused_imports)]
pub(crate) use rns_blas::{measured_ways, with_split_hint, ResidueCache};
