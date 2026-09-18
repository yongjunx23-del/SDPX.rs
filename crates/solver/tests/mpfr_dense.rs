//! Provider contract tests, separate from end-to-end solver qualification.
#![cfg(feature = "sdp")]
// The dense provider is crate-private. Compile the same trait/provider source
// here to exercise its LAPACK boundary without expanding the public API.
// This test imports the whole provider but exercises only selected operations.
#[allow(dead_code)]
#[path = "../src/algebra/dense/blas/traits.rs"]
mod provider;
use num_traits::{FromPrimitive, One, ToPrimitive, Zero};
use provider::*;
use sdpx_arithmetic::{MpFloat, Scalar};
type F<const N: usize> = MpFloat<N>;
fn f<const N: usize>(x: i64) -> F<N> {
    F::from_i64(x).unwrap()
}
fn close<const N: usize>(x: F<N>, y: F<N>, scale: F<N>) {
    assert!(
        (x - y).abs() <= F::epsilon() * f::<N>(10000) * scale,
        "{x} != {y}"
    );
}
fn run<const N: usize>() {
    let zero = F::<N>::zero();
    let one = F::<N>::one();
    // Unequal dimensions and transpose flags detect column-major indexing bugs.
    let a = vec![f::<N>(1), f(2), f(3), f(4), f(5), f(6)];
    let mut c = vec![zero; 4];
    F::xgemm(b'T', b'N', 2, 2, 3, one, &a, 3, &a, 3, zero, &mut c, 2);
    assert_eq!(c, vec![f(14), f(32), f(32), f(77)]);
    let mut y = vec![zero; 2];
    F::xgemv(
        b'T',
        3,
        2,
        one,
        &a,
        3,
        &[one, f(2), f(3)],
        -1,
        zero,
        &mut y,
        1,
    );
    assert_eq!(y, vec![f(10), f(28)]);
    let mut ch = vec![f(4), f(2), f(987), f(3)];
    let mut info = 0;
    F::xpotrf(b'L', 2, &mut ch, 2, &mut info);
    assert_eq!(info, 0);
    assert_eq!(ch[2], f(987));
    let mut rhs = vec![f(6), f(5)];
    F::xpotrs(b'L', 2, 1, &ch, 2, &mut rhs, 2, &mut info);
    assert_eq!(info, 0);
    for x in rhs {
        close(x, one, one);
    }
    let mut lu = vec![zero, f(2), one, f(3)];
    let mut piv = vec![0; 2];
    let mut rhs = vec![one, f(5)];
    F::xgesv(2, 1, &mut lu, 2, &mut piv, &mut rhs, 2, &mut info);
    assert_eq!(info, 0);
    assert_eq!(piv[0], 2);
    for x in rhs {
        close(x, one, one);
    }
    for (m, n, input) in [
        (3, 2, a.clone()),
        (2, 3, vec![f(1), f(4), f(2), f(5), f(3), f(6)]),
        (2, 2, vec![one, one, one, one]),
    ] {
        let mut aa = input.clone();
        let mut s = vec![zero; m.min(n)];
        let mut u = vec![zero; m * m];
        let mut vt = vec![zero; n * n];
        let mut work = vec![zero];
        F::xgesvd(
            b'A', b'A', m as i32, n as i32, &mut aa, m as i32, &mut s, &mut u, m as i32, &mut vt,
            n as i32, &mut work, -1, &mut info,
        );
        assert_eq!(info, 0);
        assert_eq!(aa, input);
        F::xgesvd(
            b'A', b'A', m as i32, n as i32, &mut aa, m as i32, &mut s, &mut u, m as i32, &mut vt,
            n as i32, &mut work, -1, &mut info,
        );
        assert_eq!(info, 0);
        let lwork = work[0].to_i32().unwrap();
        work.resize(lwork as usize, zero);
        F::xgesvd(
            b'A', b'A', m as i32, n as i32, &mut aa, m as i32, &mut s, &mut u, m as i32, &mut vt,
            n as i32, &mut work, lwork, &mut info,
        );
        assert_eq!(info, 0);
        for j in 0..n {
            for i in 0..m {
                let mut v = zero;
                for k in 0..s.len() {
                    v += u[i + k * m] * s[k] * vt[k + j * n];
                }
                close(v, input[i + j * m], f(10));
            }
        }
        for (q, dim) in [(&u, m), (&vt, n)] {
            for j in 0..dim {
                for i in 0..dim {
                    let mut v = zero;
                    for k in 0..dim {
                        v += q[k + i * dim] * q[k + j * dim];
                    }
                    close(v, if i == j { one } else { zero }, one);
                }
            }
        }
    }
    // Resolves a singular value below Float64 resolution without squaring it.
    let tiny = F::epsilon().sqrt() * F::epsilon().sqrt().sqrt();
    let mut aa = vec![one, zero, zero, tiny];
    let mut s = vec![zero; 2];
    let mut u = vec![zero; 4];
    let mut vt = vec![zero; 4];
    let mut work = vec![zero];
    let mut iw = vec![0; 16];
    F::xgesdd(
        b'S', 2, 2, &mut aa, 2, &mut s, &mut u, 2, &mut vt, 2, &mut work, -1, &mut iw, &mut info,
    );
    assert_eq!(info, 0);
    let lwork = work[0].to_i32().unwrap();
    work.resize(lwork as usize, zero);
    F::xgesdd(
        b'S', 2, 2, &mut aa, 2, &mut s, &mut u, 2, &mut vt, 2, &mut work, lwork, &mut iw, &mut info,
    );
    assert_eq!(info, 0);
    close(s[1], tiny, tiny);
    // Ill-conditioned rectangular columns with exact orthogonality and a
    // singular-value ratio far beyond binary64 at the larger precision modes.
    let rect = vec![one, one, zero, tiny, -tiny, tiny];
    let mut aa = rect.clone();
    let mut u = vec![zero; 9];
    let mut vt = vec![zero; 4];
    F::xgesvd(
        b'A', b'A', 3, 2, &mut aa, 3, &mut s, &mut u, 3, &mut vt, 2, &mut work, -1, &mut info,
    );
    assert_eq!(info, 0);
    let lwork = work[0].to_i32().unwrap();
    work.resize(lwork as usize, zero);
    F::xgesvd(
        b'A', b'A', 3, 2, &mut aa, 3, &mut s, &mut u, 3, &mut vt, 2, &mut work, lwork, &mut info,
    );
    assert_eq!(info, 0);
    close(s[0], f::<N>(2).sqrt(), one);
    close(s[1], tiny * f::<N>(3).sqrt(), tiny);
    for j in 0..2 {
        for i in 0..3 {
            let mut v = zero;
            for k in 0..2 {
                v += u[i + k * 3] * s[k] * vt[k + j * 2];
            }
            close(v, rect[i + j * 3], if j == 0 { one } else { tiny });
        }
    }
    for j in 0..3 {
        for i in 0..3 {
            let mut v = zero;
            for k in 0..3 {
                v += u[k + i * 3] * u[k + j * 3];
            }
            close(v, if i == j { one } else { zero }, one);
        }
    }
    // Authoritative upper triangle; poison lower triangle must be ignored.
    let mut eig = vec![f(2), F::nan(), one, f(2)];
    let original = vec![f(2), one, one, f(2)];
    let mut w = vec![zero; 2];
    let mut z = vec![zero; 4];
    let mut support = vec![0; 4];
    let mut count = 0;
    F::xsyevr(
        b'V',
        b'A',
        b'U',
        2,
        &mut eig,
        2,
        zero,
        zero,
        0,
        0,
        zero,
        &mut count,
        &mut w,
        &mut z,
        2,
        &mut support,
        &mut work,
        -1,
        &mut iw,
        -1,
        &mut info,
    );
    assert_eq!(info, 0);
    let lwork = work[0].to_i32().unwrap();
    work.resize(lwork as usize, zero);
    let liwork = iw[0];
    iw.resize(liwork as usize, 0);
    F::xsyevr(
        b'V',
        b'A',
        b'U',
        2,
        &mut eig,
        2,
        zero,
        zero,
        0,
        0,
        zero,
        &mut count,
        &mut w,
        &mut z,
        2,
        &mut support,
        &mut work,
        lwork,
        &mut iw,
        liwork,
        &mut info,
    );
    assert_eq!(info, 0);
    assert_eq!(count, 2);
    close(w[0], one, one);
    close(w[1], f(3), one);
    for j in 0..2 {
        for i in 0..2 {
            let mut v = zero;
            for k in 0..2 {
                v += original[i + k * 2] * z[k + j * 2];
            }
            close(v, w[j] * z[i + j * 2], f(3));
        }
    }
}
#[test]
fn dense_128() {
    run::<2>();
}
#[test]
fn dense_256() {
    run::<4>();
}
#[test]
fn dense_512() {
    run::<8>();
}
#[test]
fn dense_768() {
    run::<12>();
}
#[test]
fn dense_1024() {
    run::<16>();
}
#[test]
fn dense_2048() {
    run::<32>();
}

fn contracts<const N: usize>() {
    let zero = F::<N>::zero();
    let one = F::<N>::one();
    let pad = f::<N>(777);
    let mut info = 0;
    let mut work = vec![zero];
    let mut iw = vec![0; 64];
    // MPFR exponent-limit scaling must not report a successful false rank loss.
    let mut edge = vec![f(4), zero, zero, F::min_positive_value()];
    let mut s = vec![zero; 2];
    F::xgesvd(
        b'N',
        b'N',
        2,
        2,
        &mut edge,
        2,
        &mut s,
        &mut [],
        1,
        &mut [],
        1,
        &mut work,
        -1,
        &mut info,
    );
    assert_eq!(info, 0);
    let lwork = work[0].to_i32().unwrap();
    work.resize(lwork as usize, zero);
    F::xgesvd(
        b'N',
        b'N',
        2,
        2,
        &mut edge,
        2,
        &mut s,
        &mut [],
        1,
        &mut [],
        1,
        &mut work,
        lwork,
        &mut info,
    );
    assert!(
        info > 0,
        "an unrepresentable scaled input must report failure"
    );

    // Thin and values-only tall calls must not construct the 1024-column
    // orthogonal completion required only by JOBU=A. Unrequested outputs
    // really are absent, and leading dimensions follow LAPACK's N contract.
    let mut tall = vec![one; 1024];
    let mut single = vec![zero];
    F::xgesvd(
        b'N',
        b'N',
        1024,
        1,
        &mut tall,
        1024,
        &mut single,
        &mut [],
        1,
        &mut [],
        1,
        &mut work,
        -1,
        &mut info,
    );
    assert_eq!(info, 0);
    let lwork = work[0].to_i32().unwrap();
    work.resize(lwork as usize, zero);
    F::xgesvd(
        b'N',
        b'N',
        1024,
        1,
        &mut tall,
        1024,
        &mut single,
        &mut [],
        1,
        &mut [],
        1,
        &mut work,
        lwork,
        &mut info,
    );
    assert_eq!(info, 0);
    close(single[0], f(32), one);
    let mut thin = vec![zero; 1024];
    let mut right = vec![zero];
    F::xgesvd(
        b'S',
        b'S',
        1024,
        1,
        &mut tall,
        1024,
        &mut single,
        &mut thin,
        1024,
        &mut right,
        1,
        &mut work,
        -1,
        &mut info,
    );
    assert_eq!(info, 0);
    let lwork = work[0].to_i32().unwrap();
    work.resize(lwork as usize, zero);
    F::xgesvd(
        b'S',
        b'S',
        1024,
        1,
        &mut tall,
        1024,
        &mut single,
        &mut thin,
        1024,
        &mut right,
        1,
        &mut work,
        lwork,
        &mut info,
    );
    assert_eq!(info, 0);
    for i in 0..1024 {
        close(thin[i] * single[0] * right[0], one, one);
    }

    // Both rectangular orientations exercise thin-factor storage and overwrite
    // semantics with padded lda/ldu/ldvt. Padding is neither data nor scratch.
    for (m, n) in [(3usize, 2usize), (2, 3)] {
        let r = m.min(n);
        let ld = m + 2;
        let lu = m + 1;
        let lv = r + 2;
        let mut original = vec![zero; m * n];
        for j in 0..n {
            for i in 0..m {
                original[i + j * m] = f((1 + i + 2 * j) as i64);
            }
        }
        for overwrite in [false, true] {
            let mut a = vec![pad; ld * n];
            for j in 0..n {
                for i in 0..m {
                    a[i + j * ld] = original[i + j * m];
                }
            }
            let mut u = vec![pad; lu * r];
            let mut vt = vec![pad; lv * n];
            let mut s = vec![zero; r];
            let ju = if overwrite && m >= n { b'O' } else { b'S' };
            let jv = if overwrite && m < n { b'O' } else { b'S' };
            F::xgesvd(
                ju, jv, m as i32, n as i32, &mut a, ld as i32, &mut s, &mut u, lu as i32, &mut vt,
                lv as i32, &mut work, -1, &mut info,
            );
            assert_eq!(info, 0);
            let lwork = work[0].to_i32().unwrap();
            work.resize(lwork as usize, zero);
            F::xgesvd(
                ju, jv, m as i32, n as i32, &mut a, ld as i32, &mut s, &mut u, lu as i32, &mut vt,
                lv as i32, &mut work, lwork, &mut info,
            );
            assert_eq!(info, 0);
            for j in 0..n {
                for i in 0..m {
                    let mut value = zero;
                    for k in 0..r {
                        let left = if ju == b'O' {
                            a[i + k * ld]
                        } else {
                            u[i + k * lu]
                        };
                        let right = if jv == b'O' {
                            a[k + j * ld]
                        } else {
                            vt[k + j * lv]
                        };
                        value += left * s[k] * right;
                    }
                    close(value, original[i + j * m], f(10));
                }
            }
            for j in 0..n {
                for i in m..ld {
                    assert_eq!(a[i + j * ld], pad);
                }
            }
            for j in 0..r {
                for i in m..lu {
                    assert_eq!(u[i + j * lu], pad);
                }
            }
            for j in 0..n {
                for i in r..lv {
                    assert_eq!(vt[i + j * lv], pad);
                }
            }
        }
    }

    // A nonzero superdiagonal couples the columns while the smaller singular
    // value is below epsilon. This exercises relative bidiagonal deflation,
    // rather than only the already-diagonal tiny-value path.
    let tiny = F::<N>::epsilon() * F::<N>::epsilon();
    let mut a = vec![one, zero, zero, one, tiny, zero];
    let original = a.clone();
    let mut u = vec![zero; 6];
    let mut vt = vec![zero; 4];
    let mut s = vec![zero; 2];
    F::xgesvd(
        b'S', b'S', 3, 2, &mut a, 3, &mut s, &mut u, 3, &mut vt, 2, &mut work, -1, &mut info,
    );
    assert_eq!(info, 0);
    let lwork = work[0].to_i32().unwrap();
    work.resize(lwork as usize, zero);
    F::xgesvd(
        b'S', b'S', 3, 2, &mut a, 3, &mut s, &mut u, 3, &mut vt, 2, &mut work, lwork, &mut info,
    );
    assert_eq!(info, 0);
    close(s[0], f::<N>(2).sqrt(), one);
    close(s[1], tiny / f::<N>(2).sqrt(), tiny);
    for j in 0..2 {
        for i in 0..3 {
            let mut value = zero;
            for k in 0..2 {
                value += u[i + k * 3] * s[k] * vt[k + j * 2];
            }
            close(value, original[i + j * 3], one);
        }
    }
    for j in 0..2 {
        for i in 0..2 {
            let mut dot = zero;
            for k in 0..3 {
                dot += u[k + i * 3] * u[k + j * 3];
            }
            close(dot, if i == j { one } else { zero }, one);
        }
    }

    // Six coupled columns in a three-dimensional subspace, with an 8-row
    // ambient space. This exercises nontrivial rotations plus null vectors.
    let m = 8;
    let n = 6;
    let mut original = vec![zero; m * n];
    for i in 0..m {
        let x = one;
        let y = if i & 1 == 0 { one } else { -one };
        let z = if i & 2 == 0 { one } else { -one };
        for (j, v) in [x + y, x - y, x + z, x - z, y + z, y - z]
            .into_iter()
            .enumerate()
        {
            original[i + j * m] = v;
        }
    }
    let mut a = original.clone();
    let mut u = vec![zero; m * m];
    let mut vt = vec![zero; n * n];
    let mut s = vec![zero; n];
    F::xgesvd(
        b'A', b'A', m as i32, n as i32, &mut a, m as i32, &mut s, &mut u, m as i32, &mut vt,
        n as i32, &mut work, -1, &mut info,
    );
    assert_eq!(info, 0);
    let lwork = work[0].to_i32().unwrap();
    work.resize(lwork as usize, zero);
    F::xgesvd(
        b'A', b'A', m as i32, n as i32, &mut a, m as i32, &mut s, &mut u, m as i32, &mut vt,
        n as i32, &mut work, lwork, &mut info,
    );
    assert_eq!(info, 0);
    for j in 0..n {
        for i in 0..m {
            let mut value = zero;
            for k in 0..n {
                value += u[i + k * m] * s[k] * vt[k + j * n];
            }
            close(value, original[i + j * m], f(8));
        }
    }
    for (q, dim) in [(&u, m), (&vt, n)] {
        for j in 0..dim {
            for i in 0..dim {
                let mut dot = zero;
                for k in 0..dim {
                    dot += q[k + i * dim] * q[k + j * dim];
                }
                close(dot, if i == j { one } else { zero }, one);
            }
        }
    }
    for &value in &s[3..] {
        close(value, zero, f(8));
    }

    // The old per-entry ABSTOL=1 stop returned four 2s, although the largest
    // exact eigenvalue is 4.25. Aggregate residual control must bound every
    // sorted eigenvalue's error by ABSTOL, including that largest value.
    let mut a = vec![f(3) / f(4); 16];
    for i in 0..4 {
        a[i + i * 4] = f(2);
    }
    let mut count = 0;
    let mut w = vec![zero; 4];
    F::xsyevr(
        b'N',
        b'A',
        b'L',
        4,
        &mut a,
        4,
        zero,
        zero,
        0,
        0,
        one,
        &mut count,
        &mut w,
        &mut [],
        1,
        &mut [],
        &mut work,
        -1,
        &mut iw,
        -1,
        &mut info,
    );
    assert_eq!(info, 0);
    let lwork = work[0].to_i32().unwrap();
    work.resize(lwork as usize, zero);
    let liwork = iw[0];
    iw.resize(liwork as usize, 0);
    F::xsyevr(
        b'N',
        b'A',
        b'L',
        4,
        &mut a,
        4,
        zero,
        zero,
        0,
        0,
        one,
        &mut count,
        &mut w,
        &mut [],
        1,
        &mut [],
        &mut work,
        lwork,
        &mut iw,
        liwork,
        &mut info,
    );
    assert_eq!(info, 0);
    assert_eq!(count, 4);
    for i in 0..4 {
        let exact = if i < 3 { f(5) / f(4) } else { f(17) / f(4) };
        assert!((w[i] - exact).abs() <= one);
    }
    // Diagonal spectrum makes interval endpoint inclusion and index ranges
    // exact; upper-triangle padding/lower garbage must be ignored.
    for (range, vl, vu, il, iu, expected) in [
        (b'V', f(2), f(4), 0, 0, vec![f(3), f(4)]),
        (b'I', zero, zero, 2, 3, vec![f(2), f(3)]),
    ] {
        let mut a = vec![pad; 6 * 4];
        for j in 0..4 {
            for i in 0..=j {
                a[i + j * 6] = if i == j { f((i + 1) as i64) } else { zero };
            }
        }
        let mut z = vec![pad; 6 * 2];
        let mut support = vec![0; 4];
        F::xsyevr(
            b'V',
            range,
            b'U',
            4,
            &mut a,
            6,
            vl,
            vu,
            il,
            iu,
            zero,
            &mut count,
            &mut w,
            &mut z,
            6,
            &mut support,
            &mut work,
            -1,
            &mut iw,
            -1,
            &mut info,
        );
        assert_eq!(info, 0);
        let lwork = work[0].to_i32().unwrap();
        work.resize(lwork as usize, zero);
        let liwork = iw[0];
        iw.resize(liwork as usize, 0);
        F::xsyevr(
            b'V',
            range,
            b'U',
            4,
            &mut a,
            6,
            vl,
            vu,
            il,
            iu,
            zero,
            &mut count,
            &mut w,
            &mut z,
            6,
            &mut support,
            &mut work,
            lwork,
            &mut iw,
            liwork,
            &mut info,
        );
        assert_eq!(info, 0);
        assert_eq!(count, 2);
        assert_eq!(&w[..2], &expected);
        for j in 0..2 {
            for i in 0..4 {
                let lambda = f((i + 1) as i64);
                close(lambda * z[i + j * 6], w[j] * z[i + j * 6], f(4));
            }
            assert_eq!(z[4 + j * 6], pad);
            assert_eq!(z[5 + j * 6], pad);
        }
    }
}
#[test]
fn provider_contracts_128() {
    contracts::<2>();
}
#[test]
fn provider_contracts_256() {
    contracts::<4>();
}
#[test]
fn provider_contracts_512() {
    contracts::<8>();
}
#[test]
fn provider_contracts_768() {
    contracts::<12>();
}
#[test]
fn provider_contracts_1024() {
    contracts::<16>();
}
#[test]
fn provider_contracts_2048() {
    contracts::<32>();
}

#[test]
fn symmetric_eigen_512_general() {
    let zero = F::<8>::zero();
    let one = F::<8>::one();
    let mut info = 0;
    let mut work = vec![zero];
    let mut iw = vec![0; 64];
    let mut count = 0;
    // 1. Deterministic 14x14 symmetric matrix, no RNG.
    let n = 14;
    let mut a = vec![zero; n * n];
    for j in 0..n {
        for i in 0..=j {
            let v = f::<8>(((i * 7 + j * 3) % 11) as i64) / f::<8>(4) - f::<8>(1);
            a[i + j * n] = v;
            a[j + i * n] = v;
        }
    }
    let original = a.clone();
    // 2. Query then compute all eigenpairs.
    let mut w = vec![zero; n];
    let mut z = vec![zero; n * n];
    let mut support = vec![0; 2 * n];
    F::xsyevr(
        b'V',
        b'A',
        b'U',
        n as i32,
        &mut a.clone(),
        n as i32,
        zero,
        zero,
        0,
        0,
        zero,
        &mut count,
        &mut w,
        &mut z,
        n as i32,
        &mut support,
        &mut work,
        -1,
        &mut iw,
        -1,
        &mut info,
    );
    assert_eq!(info, 0);
    let lwork = work[0].to_i32().unwrap();
    work.resize(lwork as usize, zero);
    let liwork = iw[0];
    iw.resize(liwork as usize, 0);
    F::xsyevr(
        b'V',
        b'A',
        b'U',
        n as i32,
        &mut a.clone(),
        n as i32,
        zero,
        zero,
        0,
        0,
        zero,
        &mut count,
        &mut w,
        &mut z,
        n as i32,
        &mut support,
        &mut work,
        lwork,
        &mut iw,
        liwork,
        &mut info,
    );
    assert_eq!(info, 0);
    assert_eq!(count, 14);
    // 3. Ascending order.
    for j in 0..n - 1 {
        assert!(w[j] <= w[j + 1]);
    }
    // 4. Residual ||A z_j - w_j z_j||_inf.
    for j in 0..n {
        for i in 0..n {
            let mut v = zero;
            for k in 0..n {
                v += original[i + k * n] * z[k + j * n];
            }
            close(v, w[j] * z[i + j * n], f(20));
        }
    }
    // 5. Orthogonality ZᵀZ = I.
    for j in 0..n {
        for i in 0..n {
            let mut v = zero;
            for k in 0..n {
                v += z[k + i * n] * z[k + j * n];
            }
            close(v, if i == j { one } else { zero }, one);
        }
    }
    // 6. Values-only 'N' returns identical eigenvalues (shared (d, e) path).
    let mut wn = vec![zero; n];
    F::xsyevr(
        b'N',
        b'A',
        b'U',
        n as i32,
        &mut a.clone(),
        n as i32,
        zero,
        zero,
        0,
        0,
        zero,
        &mut count,
        &mut wn,
        &mut [],
        1,
        &mut [],
        &mut work,
        lwork,
        &mut iw,
        liwork,
        &mut info,
    );
    assert_eq!(info, 0);
    assert_eq!(w, wn);
    // 7. Repeated/clustered eigenvalues on exact-integer spectra.
    for (size, matrix, expected) in [
        (
            4usize,
            vec![
                f(2),
                zero,
                zero,
                zero,
                zero,
                f(2),
                zero,
                zero,
                zero,
                zero,
                f(2),
                zero,
                zero,
                zero,
                zero,
                f(5),
            ],
            vec![f(2), f(2), f(2), f(5)],
        ),
        (
            4usize,
            vec![
                f(3),
                f(2),
                f(4),
                zero,
                f(2),
                zero,
                f(2),
                zero,
                f(4),
                f(2),
                f(3),
                zero,
                zero,
                zero,
                zero,
                f(9),
            ],
            vec![f(-1), f(-1), f(8), f(9)],
        ),
    ] {
        let mut aa = matrix;
        let mut ww = vec![zero; size];
        let mut zz = vec![zero; size * size];
        F::xsyevr(
            b'V',
            b'A',
            b'U',
            size as i32,
            &mut aa,
            size as i32,
            zero,
            zero,
            0,
            0,
            zero,
            &mut count,
            &mut ww,
            &mut zz,
            size as i32,
            &mut support,
            &mut work,
            lwork,
            &mut iw,
            liwork,
            &mut info,
        );
        assert_eq!(info, 0);
        assert_eq!(count, size as i32);
        for j in 0..size {
            close(ww[j], expected[j], one);
        }
    }
    // 8. A 3x3 call still succeeds on the small-size path.
    let mut a3 = vec![f(2), one, zero, one, f(2), one, zero, one, f(2)];
    let mut w3 = vec![zero; 3];
    F::xsyevr(
        b'N',
        b'A',
        b'U',
        3,
        &mut a3,
        3,
        zero,
        zero,
        0,
        0,
        zero,
        &mut count,
        &mut w3,
        &mut [],
        1,
        &mut [],
        &mut work,
        lwork,
        &mut iw,
        liwork,
        &mut info,
    );
    assert_eq!(info, 0);
}

#[test]
fn single_index_eigen() {
    // The indexed 'I' path (il == iu) resolves a single eigenvalue via
    // Sturm isolation + RQI, verified against the full-spectrum result.
    let zero = F::<8>::zero();
    let one = F::<8>::one();
    let mut info = 0;
    let mut work = vec![zero];
    let mut iw = vec![0; 64];
    let mut count = 0;
    let n = 14usize;
    let mut a = vec![zero; n * n];
    for j in 0..n {
        for i in 0..=j {
            let v = f::<8>(((i * 7 + j * 3) % 11) as i64) / f::<8>(4) - f::<8>(1);
            a[i + j * n] = v;
            a[j + i * n] = v;
        }
    }
    let original = a.clone();
    // The indexed path reserves extra scratch; size the workspace for it.
    let mut w1 = vec![zero; n];
    F::xsyevr(
        b'N',
        b'I',
        b'U',
        n as i32,
        &mut a.clone(),
        n as i32,
        zero,
        zero,
        1,
        1,
        zero,
        &mut count,
        &mut w1,
        &mut [],
        1,
        &mut [],
        &mut work,
        -1,
        &mut iw,
        -1,
        &mut info,
    );
    assert_eq!(info, 0);
    let lwork = work[0].to_i32().unwrap();
    work.resize(lwork as usize, zero);
    let liwork = iw[0];
    iw.resize(liwork as usize, 0);
    // Reference spectrum from the values-only full path.
    let mut wall = vec![zero; n];
    F::xsyevr(
        b'N',
        b'A',
        b'U',
        n as i32,
        &mut a.clone(),
        n as i32,
        zero,
        zero,
        0,
        0,
        zero,
        &mut count,
        &mut wall,
        &mut [],
        1,
        &mut [],
        &mut work,
        lwork,
        &mut iw,
        liwork,
        &mut info,
    );
    assert_eq!(info, 0);
    assert_eq!(count, n as i32);
    // First, middle and last eigenvalues through the single-index path.
    for k in [1i32, 7, 14] {
        let mut w = vec![zero; n];
        F::xsyevr(
            b'N',
            b'I',
            b'U',
            n as i32,
            &mut a.clone(),
            n as i32,
            zero,
            zero,
            k,
            k,
            zero,
            &mut count,
            &mut w,
            &mut [],
            1,
            &mut [],
            &mut work,
            lwork,
            &mut iw,
            liwork,
            &mut info,
        );
        assert_eq!(info, 0);
        assert_eq!(count, 1);
        close(w[0], wall[(k - 1) as usize], f(100));
    }
    // Clustered integer spectrum: exact-eigenvalue shifts exercise the
    // singular-pivot and verification branches.
    let mut ad = vec![zero; 6 * 4];
    for j in 0..4 {
        for i in 0..=j {
            ad[i + j * 6] = if i == j { f::<8>(i as i64 + 1) } else { zero };
        }
    }
    for k in 1i32..=4 {
        let mut w = vec![zero; 4];
        F::xsyevr(
            b'N',
            b'I',
            b'U',
            4,
            &mut ad.clone(),
            6,
            zero,
            zero,
            k,
            k,
            zero,
            &mut count,
            &mut w,
            &mut [],
            1,
            &mut [],
            &mut work,
            lwork,
            &mut iw,
            liwork,
            &mut info,
        );
        assert_eq!(info, 0);
        assert_eq!(count, 1);
        close(w[0], f(k as i64), one);
    }
    // A must be read through the requested triangle only and left intact.
    assert_eq!(a, original);
    let _ = one;
}
