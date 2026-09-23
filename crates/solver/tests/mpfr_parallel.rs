//! Pooled dense kernels retain every output's serial arithmetic and padding.
#![cfg(feature = "sdp")]
// This test imports the whole provider but exercises only selected operations.
#[allow(dead_code)]
#[path = "../src/algebra/dense/blas/traits.rs"]
mod provider;
use sdpx_solver::algebra;
use num_traits::{FromPrimitive, Zero};
use provider::{XgemmScalar, XsyrkScalar};
use sdpx_arithmetic::{MpFloat, Scalar};
type F<const N: usize> = MpFloat<N>;
fn f<const N: usize>(x: i32) -> F<N> {
    F::from_i32(x).unwrap()
}
fn same<const N: usize>(a: &[F<N>], b: &[F<N>]) {
    assert_eq!(a, b);
    // Finite MPFR values have unique normalized representations apart from
    // signed zero, whose sign is checked separately.
    for (a, b) in a.iter().zip(b) {
        assert_eq!(a.is_sign_negative(), b.is_sign_negative());
    }
}
fn run<const N: usize>() {
    for workers in [1, 2, 4, 8] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .unwrap();
        for ta in [b'N', b'T', b'C', b'n'] {
            for tb in [b'N', b'T', b'C'] {
                for (m, n, k) in [(7usize, 9usize, 11usize), (0, 3, 4), (3, 0, 4), (3, 5, 0)] {
                    let (ar, ac) = if ta.to_ascii_uppercase() == b'N' {
                        (m, k)
                    } else {
                        (k, m)
                    };
                    let (br, bc) = if tb == b'N' { (k, n) } else { (n, k) };
                    let (lda, ldb, ldc) = (ar + 2, br + 3, m + 4);
                    let a: Vec<_> = (0..lda * ac)
                        .map(|i| f::<N>((i % 13) as i32 - 6) / f(7))
                        .collect();
                    let b: Vec<_> = (0..ldb * bc)
                        .map(|i| f::<N>((i % 17) as i32 - 8) / f(11))
                        .collect();
                    for (alpha, beta) in [(f(1), f(0)), (f(-3) / f(2), f(2) / f(3)), (f(0), f(-1))]
                    {
                        // Minimal valid last-column storage, with interior padding.
                        let len = if n == 0 { 0 } else { (n - 1) * ldc + m };
                        let initial: Vec<_> = (0..len)
                            .map(|i| f::<N>((i % 19) as i32 - 9) / f(5))
                            .collect();
                        let mut serial = initial.clone();
                        F::xgemm(
                            ta,
                            tb,
                            m as i32,
                            n as i32,
                            k as i32,
                            alpha,
                            &a,
                            lda as i32,
                            &b,
                            ldb as i32,
                            beta,
                            &mut serial,
                            ldc as i32,
                        );
                        for tile in [0, 1, 3, 20] {
                            let mut pooled = initial.clone();
                            F::xgemm_pool(
                                ta,
                                tb,
                                m as i32,
                                n as i32,
                                k as i32,
                                alpha,
                                &a,
                                lda as i32,
                                &b,
                                ldb as i32,
                                beta,
                                &mut pooled,
                                ldc as i32,
                                &pool,
                                tile,
                            );
                            same(&pooled, &serial);
                            for j in 0..n.saturating_sub(1) {
                                same(
                                    &pooled[j * ldc + m..(j + 1) * ldc],
                                    &initial[j * ldc + m..(j + 1) * ldc],
                                );
                            }
                        }
                    }
                }
            }
        }
        for u in [b'U', b'L', b'u'] {
            for t in [b'N', b'T', b'C'] {
                for (n, k) in [
                    (9usize, 11usize),
                    (0, 3),
                    (1, 0),
                    (2, 1),
                    (3, 2),
                    (5, 0),
                    (24, 5),
                    (25, 7),
                ] {
                    let (ar, ac) = if t == b'N' { (n, k) } else { (k, n) };
                    let (lda, ldc) = (ar + 2, n + 3);
                    let a: Vec<_> = (0..lda * ac)
                        .map(|i| f::<N>((i % 17) as i32 - 8) / f(7))
                        .collect();
                    for (alpha, beta) in [
                        (f(1), f(0)),
                        (f(-3) / f(2), f(2) / f(3)),
                        (F::zero(), f(-1)),
                    ] {
                        let len = if n == 0 { 0 } else { (n - 1) * ldc + n };
                        let initial = vec![f::<N>(-7); len];
                        let mut serial = initial.clone();
                        F::xsyrk(
                            u,
                            t,
                            n as i32,
                            k as i32,
                            alpha,
                            &a,
                            lda as i32,
                            beta,
                            &mut serial,
                            ldc as i32,
                        );
                        for tile in [0, 1, 3, 20] {
                            let mut pooled = initial.clone();
                            F::xsyrk_pool(
                                u,
                                t,
                                n as i32,
                                k as i32,
                                alpha,
                                &a,
                                lda as i32,
                                beta,
                                &mut pooled,
                                ldc as i32,
                                &pool,
                                tile,
                            );
                            same(&pooled, &serial);
                            for j in 0..n {
                                for i in 0..n {
                                    if (u.to_ascii_uppercase() == b'U' && i > j)
                                        || (u == b'L' && i < j)
                                    {
                                        assert_eq!(pooled[i + j * ldc], initial[i + j * ldc]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
#[test]
fn mpfr128() {
    run::<2>();
}
#[test]
fn mpfr256() {
    run::<4>();
}
#[test]
fn mpfr512() {
    run::<8>();
}
#[test]
fn mpfr768() {
    run::<12>();
}
#[test]
fn mpfr1024() {
    run::<16>();
}
#[test]
fn mpfr2048() {
    run::<32>();
}
