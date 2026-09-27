//! All-precision GEMM/SYRK checks with independent accumulation/indexing at 8192 bits.
//! The reference shares the production MpFloat wrapper and decimal conversion;
//! it is not a binding-independent native MPFR oracle.
#![cfg(feature = "sdp")]
// This test imports the whole provider but exercises only selected operations.
#[allow(dead_code)]
#[path = "../src/algebra/dense/blas/traits.rs"]
mod provider;
use num_traits::{FromPrimitive, One, Zero};
use provider::{XgemmScalar, XsyrkScalar};
use sdpx_arithmetic::{MpFloat, Scalar};
use sdpx_solver::algebra;
type F<const N: usize> = MpFloat<N>;
type Oracle = F<128>;

fn f<const N: usize>(v: i32) -> F<N> {
    F::from_i32(v).unwrap()
}

fn lift<const N: usize>(v: F<N>) -> Oracle {
    // These bounded dyadic fixtures need fewer than 4p fractional decimal
    // digits. Unlike a shortest round-trip string, this preserves the exact
    // working-precision input/output when promoted to the higher-precision
    // reference, using the shared production decimal conversion.
    v.to_decimal(Some(4 * N * 64 + 64)).parse().unwrap()
}

fn pack<const N: usize>(rows: &[Vec<F<N>>], transpose: bool) -> (Vec<F<N>>, i32) {
    let (m, n) = (rows.len(), rows[0].len());
    let (stored_rows, stored_cols) = if transpose { (n, m) } else { (m, n) };
    let ld = stored_rows + 2;
    let mut out = vec![f(987); ld * stored_cols];
    for i in 0..m {
        for j in 0..n {
            out[if transpose { j + i * ld } else { i + j * ld }] = rows[i][j];
        }
    }
    (out, ld as i32)
}

fn check<const N: usize>(actual: F<N>, expected: Oracle, work: Oracle) {
    let bound = f::<128>(64) * lift(F::<N>::epsilon()) * work;
    assert!(actual.is_finite());
    assert!(
        (lift(actual) - expected).abs() <= bound,
        "p={} actual={actual}, oracle={expected}, bound={bound}",
        N * 64
    );
}

fn run<const N: usize>() {
    let d = f::<N>(2).powi(-(32 * N as i32 + 1));
    let a = vec![
        vec![-F::one(), F::one() + d, F::zero()],
        vec![f(2), f(-3), f(4)],
    ];
    let b = vec![
        vec![F::one(), f(2), f(3)],
        vec![F::one() - d, f(4), f(-2)],
        vec![F::zero(), f(5), f(6)],
    ];
    let d_high = f::<128>(2).powi(-(32 * N as i32 + 1));
    let exact_cancel = -d_high * d_high;
    // All modes retain the same forward-error bounds below. Only the fused
    // modes also require relative accuracy for the tiny cancellation result;
    // baseline multiply/add legitimately rounds that product before adding.
    let parameters = [
        (F::one(), F::zero()),
        (-F::one(), F::one()),
        (F::one(), -F::one()),
        (-f(3) / f(2), F::one() / f(4)),
        (F::zero(), f(-2)),
        (F::zero(), F::zero()),
    ];
    for ta in [b'N', b'T', b'C'] {
        for tb in [b'N', b'T', b'C'] {
            let (aa, lda) = pack(&a, ta != b'N');
            let (bb, ldb) = pack(&b, tb != b'N');
            let saved = (aa.clone(), bb.clone());
            for (alpha, beta) in parameters {
                let ldc = 4;
                let mut c = vec![f(987); ldc * 3];
                for j in 0..3 {
                    for i in 0..2 {
                        c[i + j * ldc] = f((i + j + 1) as i32);
                    }
                }
                let before = c.clone();
                F::<N>::xgemm(
                    ta, tb, 2, 3, 3, alpha, &aa, lda, &bb, ldb, beta, &mut c, ldc as i32,
                );
                for j in 0..3 {
                    for i in 0..2 {
                        // Separate multiply/add at 8192 bits, no provider or
                        // production indexing helpers in the reference.
                        let mut sum = Oracle::zero();
                        let mut work = Oracle::zero();
                        for k in 0..3 {
                            let term = lift(a[i][k]) * lift(b[k][j]);
                            sum += term;
                            work += term.abs();
                        }
                        let prior = lift(beta) * lift(before[i + j * ldc]);
                        check(
                            c[i + j * ldc],
                            lift(alpha) * sum + prior,
                            lift(alpha).abs() * work + prior.abs(),
                        );
                    }
                    assert_eq!(
                        &c[2 + j * ldc..4 + j * ldc],
                        &before[2 + j * ldc..4 + j * ldc]
                    );
                }
                if matches!(N, 4 | 8 | 12 | 16) && alpha == F::one() && beta == F::zero() {
                    // -1 + (1+d)(1-d) loses d² with separate product
                    // rounding; a fused final accumulation preserves it.
                    check(c[0], exact_cancel, exact_cancel.abs());
                }
                assert_eq!((&aa, &bb), (&saved.0, &saved.1));
            }
        }
    }
    let a = vec![
        a[0].clone(),
        vec![F::one(), F::one() - d, F::zero()],
        a[1].clone(),
    ];
    for t in [b'N', b'T', b'C'] {
        let (aa, lda) = pack(&a, t != b'N');
        let saved = aa.clone();
        for u in [b'U', b'L'] {
            for (alpha, beta) in parameters {
                let ldc = 5;
                let mut c = vec![f(987); ldc * 3];
                for j in 0..3 {
                    for i in 0..3 {
                        c[i + j * ldc] = f((i + j + 1) as i32);
                    }
                }
                let before = c.clone();
                F::<N>::xsyrk(u, t, 3, 3, alpha, &aa, lda, beta, &mut c, ldc as i32);
                for j in 0..3 {
                    for i in 0..ldc {
                        if i >= 3 || (u == b'U' && i > j) || (u == b'L' && i < j) {
                            assert_eq!(c[i + j * ldc], before[i + j * ldc]);
                            continue;
                        }
                        let mut sum = Oracle::zero();
                        let mut work = Oracle::zero();
                        for k in 0..3 {
                            let term = lift(a[i][k]) * lift(a[j][k]);
                            sum += term;
                            work += term.abs();
                        }
                        let prior = lift(beta) * lift(before[i + j * ldc]);
                        check(
                            c[i + j * ldc],
                            lift(alpha) * sum + prior,
                            lift(alpha).abs() * work + prior.abs(),
                        );
                    }
                }
                if matches!(N, 4 | 8 | 12 | 16) && alpha == F::one() && beta == F::zero() {
                    check(
                        c[if u == b'U' { ldc } else { 1 }],
                        exact_cancel,
                        exact_cancel.abs(),
                    );
                }
                assert_eq!(aa, saved);
            }
        }
    }
}

#[test]
fn gemm_syrk_oracle_128() {
    run::<2>();
}
#[test]
fn gemm_syrk_oracle_256() {
    run::<4>();
}
#[test]
fn gemm_syrk_oracle_512() {
    run::<8>();
}
#[test]
fn gemm_syrk_oracle_768() {
    run::<12>();
}
#[test]
fn gemm_syrk_oracle_1024() {
    run::<16>();
}
#[test]
fn gemm_syrk_oracle_2048() {
    run::<32>();
}
