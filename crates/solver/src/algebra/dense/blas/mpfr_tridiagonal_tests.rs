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
