use super::*;
type R = F<4>; // 256-bit MPFR

fn mk(i: usize, j: usize) -> R {
    // Deterministic full-rank matrix with a spread spectrum.
    let x = (i * 7 + j * 13 + (i * j) % 5) as usize;
    let mut v = num::<4>(x % 9 + 1) / num::<4>(3);
    if i == j {
        v += num::<4>(4);
    }
    if (i + 2 * j) % 11 == 0 {
        v = v / num::<4>(17);
    }
    v
}

#[test]
fn jacobi_matches_qr_path() {
    let n = 20;
    let m = 20;
    let a: Vec<R> = (0..m * n).map(|e| mk(e % m, e / m)).collect();
    let len = svd_work_len(m, n, n, n).unwrap();

    // Reference: the serial bidiagonal-QR path (no inner-parallel guard).
    let mut work = vec![R::zero(); len];
    let s_qr: Vec<R> = svd(&a, m, m, n, n, n, &mut work).unwrap().0.to_vec();

    // Jacobi path (guard enters ambient-parallel eligibility).
    let _g = sdpx_arithmetic::inner_parallel::Guard::enter();
    work.fill(R::zero());
    let (s_j, u, vt) = svd(&a, m, m, n, n, n, &mut work).unwrap();

    // Singular values agree between the two methods.
    let tol = R::epsilon().sqrt();
    for j in 0..n {
        let denom = s_qr[j].abs().max(R::one());
        assert!(
            (s_j[j] - s_qr[j]).abs() <= tol * denom,
            "singular value {j}: jacobi={} qr={}",
            s_j[j],
            s_qr[j]
        );
    }

    // Reconstruction residual ||A - U S Vt|| and column orthonormality.
    let mut res = R::zero();
    for j in 0..n {
        for i in 0..m {
            let mut acc = -a[i + j * m];
            for k in 0..n {
                acc += u[i + k * m] * s_j[k] * vt[k + j * n];
            }
            res += acc * acc;
        }
    }
    assert!(res.sqrt() <= tol * num::<4>(m * n));

    let mut worst = R::zero();
    for p in 0..n {
        for q in 0..n {
            let mut dot = R::zero();
            for i in 0..m {
                dot += u[i + p * m] * u[i + q * m];
            }
            if p == q {
                dot -= R::one();
            }
            worst = worst.max(dot.abs());
        }
    }
    assert!(worst <= tol * num::<4>(n));
}

#[test]
fn jacobi_zero_column_falls_back() {
    let n = 20;
    let m = 20;
    let mut a: Vec<R> = (0..m * n).map(|e| mk(e % m, e / m)).collect();
    for i in 0..m {
        a[i + 3 * m] = R::zero(); // rank-deficient column
    }
    let len = svd_work_len(m, n, n, n).unwrap();
    let mut work = vec![R::zero(); len];
    let (s, _, _) = {
        let _g = sdpx_arithmetic::inner_parallel::Guard::enter();
        svd(&a, m, m, n, n, n, &mut work).unwrap()
    };
    // QR fallback still returns the (nearly) zero singular value last.
    assert!(s[n - 1].abs() <= R::epsilon().sqrt() * num::<4>(m));
}
