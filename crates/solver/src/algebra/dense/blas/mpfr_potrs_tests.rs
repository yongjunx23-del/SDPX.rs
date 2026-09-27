use super::*;
use num_traits::FromPrimitive;
fn number<const N: usize>(value: i32) -> F<N> {
    F::from_i32(value).unwrap()
}
use std::sync::atomic::Ordering;

// Literal pre-change sweep, including its original padded-column indexing.
fn frozen<const N: usize>(
    u: u8,
    n: usize,
    nrhs: usize,
    a: &[F<N>],
    ld: usize,
    b: &mut [F<N>],
    lb: usize,
) {
    for r in 0..nrhs {
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
fn exact<const N: usize>(a: &[F<N>], b: &[F<N>]) {
    for (a, b) in a.iter().zip(b) {
        assert_eq!(a.is_nan(), b.is_nan());
        if !b.is_nan() {
            assert_eq!(a, b);
            assert_eq!(a.is_sign_negative(), b.is_sign_negative());
        }
    }
    assert_eq!(a.len(), b.len());
}
fn cases<const N: usize>() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    for (n, nrhs) in [(0usize, 4usize), (3, 0), (3, 7), (32, 1), (32, 4), (32, 9)] {
        for u in [b'L', b'u'] {
            let ld = n + 2;
            let lb = n + 3;
            let mut a = vec![F::<N>::nan(); ld * n];
            for j in 0..n {
                for i in j..n {
                    let v = if i == j {
                        number::<N>(2)
                    } else {
                        number::<N>(((i + j) % 7) as i32 - 3) / number::<N>(32)
                    };
                    if upper(u) == b'L' {
                        a[i + j * ld] = v;
                    } else {
                        a[j + i * ld] = v;
                    }
                }
            }
            let input: Vec<_> = (0..lb * nrhs)
                .map(|i| number::<N>((i % 17) as i32 - 8) / number::<N>(16))
                .collect();
            // Test full padding and minimal legal final-column storage.
            for short in [false, true] {
                let len = if short && nrhs > 0 {
                    (nrhs - 1) * lb + n
                } else {
                    input.len()
                };
                let mut expected = input[..len].to_vec();
                frozen(u, n, nrhs, &a, ld, &mut expected, lb);
                for enabled in [false, true] {
                    let mut actual = input[..len].to_vec();
                    let before = POOLED_POTRS_CALLS.load(Ordering::Relaxed);
                    let mut info = 99;
                    pool.install(|| {
                        let _guard =
                            sdpx_arithmetic::inner_parallel::Guard::enter_levels(enabled, false);
                        F::<N>::xpotrs(
                            u,
                            n as i32,
                            nrhs as i32,
                            &a,
                            ld as i32,
                            &mut actual,
                            lb as i32,
                            &mut info,
                        );
                    });
                    assert_eq!(info, 0);
                    exact(&actual, &expected);
                    if enabled && n == 32 && nrhs >= 4 {
                        assert!(POOLED_POTRS_CALLS.load(Ordering::Relaxed) > before);
                    }
                }
            }
            // Special RHS classes use the same arithmetic, without sanitizing.
            if n == 32 && nrhs == 4 {
                let mut input = input.clone();
                input[0] = F::infinity();
                input[lb] = F::nan();
                input[2 * lb] = -F::zero();
                let mut expected = input.clone();
                frozen(u, n, nrhs, &a, ld, &mut expected, lb);
                pool.install(|| {
                    let _guard = sdpx_arithmetic::inner_parallel::Guard::enter();
                    let mut info = 99;
                    F::<N>::xpotrs(
                        u,
                        n as i32,
                        nrhs as i32,
                        &a,
                        ld as i32,
                        &mut input,
                        lb as i32,
                        &mut info,
                    );
                    assert_eq!(info, 0);
                });
                exact(&input, &expected);
            }
        }
    }
    let a = vec![F::<N>::one(); 16];
    for (u, n, r, lda, ldb, expected) in [
        (b'?', 4, 4, 4, 4, -1),
        (b'L', -1, 4, 4, 4, -2),
        (b'L', 4, -1, 4, 4, -3),
        (b'L', 4, 4, 3, 4, -5),
        (b'L', 4, 4, 4, 3, -7),
    ] {
        let mut b = vec![number::<N>(7); 16];
        let saved = b.clone();
        let mut info = 99;
        pool.install(|| {
            let _guard = sdpx_arithmetic::inner_parallel::Guard::enter();
            F::<N>::xpotrs(u, n, r, &a, lda, &mut b, ldb, &mut info);
        });
        assert_eq!(info, expected);
        exact(&b, &saved);
    }
    for short_a in [true, false] {
        let mut b = vec![F::<N>::one(); if short_a { 16 } else { 15 }];
        let saved = b.clone();
        let mut info = 99;
        F::<N>::xpotrs(
            b'L',
            4,
            4,
            &a[..if short_a { 15 } else { 16 }],
            4,
            &mut b,
            4,
            &mut info,
        );
        assert_eq!(info, if short_a { -5 } else { -7 });
        exact(&b, &saved);
    }
}

fn cancellation_and_extreme_panel<const N: usize>() {
    let n = 8usize;
    let nrhs = 3usize;
    let ld = n;
    let lb = n + 2;
    let mut a = vec![F::<N>::nan(); ld * n];
    for j in 0..n {
        for i in j..n {
            a[i + j * ld] = if i == j {
                number::<N>(2)
            } else {
                number::<N>(((i + 3 * j) % 5) as i32 - 2) / number::<N>(4)
            };
        }
    }
    let tiny = F::<N>::min_positive_value();
    let huge = F::<N>::from_u64(2).unwrap().powi(1024);
    let mut input = vec![F::<N>::zero(); lb * nrhs];
    for r in 0..nrhs {
        for i in 0..n {
            input[i + r * lb] = if r == 0 && i == 0 {
                huge
            } else if r == 0 && i == 1 {
                -huge
            } else if r == 1 && i == 2 {
                tiny
            } else {
                number::<N>((i as i32) - 3) / number::<N>(8)
            };
        }
    }
    input[lb + 4] = -F::<N>::zero();
    input[2 * lb + 5] = F::<N>::nan();
    let mut expected = input.clone();
    frozen(b'L', n, nrhs, &a, ld, &mut expected, lb);
    let mut actual = input;
    let mut info = 99;
    F::<N>::xpotrs(
        b'L',
        n as i32,
        nrhs as i32,
        &a,
        ld as i32,
        &mut actual,
        lb as i32,
        &mut info,
    );
    assert_eq!(info, 0);
    exact(&actual, &expected);
}

#[test]
fn pooled_potrs_mpfr256() {
    cases::<4>();
    cancellation_and_extreme_panel::<4>();
}
#[test]
fn pooled_potrs_mpfr512() {
    cases::<8>();
    cancellation_and_extreme_panel::<8>();
}
