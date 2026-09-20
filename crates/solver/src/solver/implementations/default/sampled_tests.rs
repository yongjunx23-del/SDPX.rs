use super::*;
use num_traits::FromPrimitive;
use sdpx_arithmetic::{Bits256, Bits512, MpFloat, Scalar};
type Oracle = MpFloat<128>;
fn lift<T: FloatT>(v: T) -> Oracle {
    // Fixtures have bounded exponents; these digits retain the complete dyadic
    // value of the actual stored working-precision input.
    format!("{v:.digits$e}", digits = 4 * T::precision_bits() + 64)
        .parse()
        .unwrap()
}
fn oracle_close<T: FloatT>(a: T, b: Oracle) {
    let one = Oracle::from_i32(1).unwrap();
    let bound = Oracle::from_i32(4096).unwrap() * lift(T::epsilon()) * b.abs().max(one);
    assert!(
        (lift(a) - b).abs() <= bound,
        "actual={a}, oracle={b}, bound={bound}"
    );
}

fn c<T: FloatT>(v: i32) -> T {
    T::from_i32(v).unwrap()
}
fn close<T: FloatT>(a: T, b: T) {
    let bound = c::<T>(4096) * T::epsilon() * b.abs().max(T::one());
    assert!(
        (a - b).abs() <= bound,
        "actual={a}, expected={b}, bound={bound}"
    );
}
fn block<T: FloatT>(row_start: usize, column_start: usize, dim: usize) -> SampledBlock<T> {
    let k = 3;
    SampledBlock {
        row_start,
        column_start,
        dim,
        basis_rows: 2,
        basis_cols: k,
        basis: vec![c::<T>(1) / c(3), c::<T>(2) / c(7), c(-1), c(1), c(0), c(0)],
        weights: (0..tri(dim) * k)
            .map(|p| {
                if p % 4 == 0 {
                    T::zero()
                } else {
                    c::<T>(p as i32 + 1) * if p % 2 == 0 { T::one() } else { -T::one() } / c(4)
                }
            })
            .collect(),
    }
}
fn parity<T: FloatT>() {
    for dim in [1, 2, 3] {
        let first = block::<T>(1, 1, dim);
        let second = block::<T>(1 + first.row_count(), 1, 1); // shared column range
        let m = 2 + first.row_count() + second.row_count();
        let n = 2 + first.column_count();
        let linear =
            CscMatrix::new_from_triplets(m, n, vec![0, m - 1], vec![0, n - 1], vec![c(2), c(-3)]);
        let mut op = SampledOperator::new(linear, vec![first, second]).unwrap();
        let mut workspace = SampledWorkspace::new(&op);
        for scaled in [false, true] {
            if scaled {
                let d: Vec<T> = (0..n).map(|i| c::<T>(i as i32 + 1) / c(2)).collect();
                let mut e = vec![c::<T>(3); m];
                e[0] = c(2);
                e[m - 1] = c(4);
                op.scale(&d, &e);
            }
            let a = op.materialize();
            assert!(a.check_format().is_ok());
            let x: Vec<T> = (0..n).map(|i| c::<T>(i as i32 - 2) / c(4)).collect();
            let z: Vec<T> = (0..m).map(|i| c::<T>(i as i32 - 3) / c(8)).collect();
            for (alpha, beta) in [
                (T::one(), T::zero()),
                (-c::<T>(3) / c(2), T::one() / c(4)),
                (T::zero(), -c::<T>(2)),
            ] {
                let mut y = z.clone();
                let mut expected = y.clone();
                op.apply(&mut y, &x, alpha, beta, &mut workspace);
                a.gemv(&mut expected, &x, alpha, beta);
                for (&v, &e) in y.iter().zip(&expected) {
                    close(v, e);
                }
                let mut yt = x.clone();
                let mut et = yt.clone();
                op.apply_transpose(&mut yt, &z, alpha, beta, &mut workspace);
                a.t().gemv(&mut et, &z, alpha, beta);
                for (&v, &e) in yt.iter().zip(&et) {
                    close(v, e);
                }
            }
            let mut ax = vec![T::zero(); m];
            let mut atz = vec![T::zero(); n];
            op.apply(&mut ax, &x, T::one(), T::zero(), &mut workspace);
            op.apply_transpose(&mut atz, &z, T::one(), T::zero(), &mut workspace);
            close(ax.dot(&z), x.dot(&atz));
        }
    }
}
#[test]
fn sampled_parity_f64() {
    parity::<f64>();
}
#[test]
fn sampled_parity_256() {
    parity::<Bits256>();
}
#[test]
fn sampled_parity_512() {
    parity::<Bits512>();
}

#[test]
fn zero_basis_columns_and_input_structure() {
    let empty = SampledBlock::<f64> {
        row_start: 1,
        column_start: 2,
        dim: 2,
        basis_rows: 1,
        basis_cols: 0,
        basis: vec![],
        weights: vec![],
    };
    let linear = CscMatrix::new_from_triplets(5, 2, vec![0, 4], vec![0, 1], vec![2., 3.]);
    let op = SampledOperator::new(linear.clone(), vec![empty.clone()]).unwrap();
    assert_eq!(op.materialize(), linear);
    let mut work = SampledWorkspace::new(&op);
    let mut y = vec![1.; 5];
    op.apply(&mut y, &[2., 3.], 1., 2., &mut work);
    assert_eq!(y, vec![6., 2., 2., 2., 11.]);
    let mut yt = vec![1.; 2];
    op.apply_transpose(&mut yt, &[1.; 5], 1., 2., &mut work);
    assert_eq!(yt, vec![4., 5.]);
    let mut bad = empty.clone();
    bad.weights.push(1.);
    assert!(SampledOperator::new(linear.clone(), vec![bad]).is_err());
    assert!(SampledOperator::new(linear.clone(), vec![empty.clone(), empty.clone()]).is_err());
    let mut bad = empty.clone();
    bad.row_start = 3;
    assert!(SampledOperator::new(linear, vec![bad]).is_err());
    let nonzero = CscMatrix::new_from_triplets(5, 2, vec![1], vec![0], vec![1.]);
    assert!(SampledOperator::new(nonzero, vec![empty]).is_err());
}

// Independent 8192-bit accumulation of each symmetric coefficient from the
// actual stored inputs (using the separately tested scalar wrapper), followed by
// explicit Rinv*M*Rinv' products and Frobenius inner products. No Gram identity
// or production svec conversion is used in this reference.
fn schur_parity<T: FloatT>() {
    schur_parity_block(block::<T>(0, 0, 2));
}
fn schur_parity_block<T: FloatT>(b: SampledBlock<T>) {
    let side = b.side();
    let mut r = Matrix::identity(side);
    for j in 0..side {
        for i in 0..j {
            r[(i, j)] = c::<T>((i + j + 1) as i32) / c(8);
        }
    }
    let mut work = SampledSchurWorkspace::new(&b);
    work.update_with_pool(&b, &r, None);
    let mut transformed = Vec::new();
    let mut p = 0;
    for s in 0..b.dim {
        for rr in 0..=s {
            for k in 0..b.basis_cols {
                let mut coefficient = vec![Oracle::from_i32(0).unwrap(); side * side];
                for j in 0..b.basis_rows {
                    for i in 0..b.basis_rows {
                        let value = lift(b.weights[p])
                            * lift(b.basis[i + k * b.basis_rows])
                            * lift(b.basis[j + k * b.basis_rows]);
                        if rr == s {
                            coefficient[(rr * b.basis_rows + i) + (s * b.basis_rows + j) * side] =
                                value;
                        } else {
                            coefficient[(rr * b.basis_rows + i) + (s * b.basis_rows + j) * side] =
                                value / Oracle::from_i32(2).unwrap();
                            coefficient[(s * b.basis_rows + j) + (rr * b.basis_rows + i) * side] =
                                value / Oracle::from_i32(2).unwrap();
                        }
                    }
                }
                let mut result = vec![Oracle::from_i32(0).unwrap(); side * side];
                for j in 0..side {
                    for i in 0..side {
                        for v in 0..side {
                            for u in 0..side {
                                result[i + j * side] +=
                                    lift(r[(i, u)]) * coefficient[u + v * side] * lift(r[(j, v)]);
                            }
                        }
                    }
                }
                transformed.push(result);
                p += 1;
            }
        }
    }
    for p in 0..b.column_count() {
        for q in 0..b.column_count() {
            let expected = transformed[p]
                .iter()
                .zip(&transformed[q])
                .fold(Oracle::from_i32(0).unwrap(), |sum, (&a, &b)| sum + a * b);
            oracle_close(work.entry(&b, p, q), expected);
            assert_eq!(work.entry(&b, p, q), work.entry(&b, q, p));
        }
    }
}
#[test]
fn sampled_schur_f64() {
    schur_parity::<f64>();
}
#[test]
fn sampled_schur_256() {
    schur_parity::<Bits256>();
}
#[test]
fn sampled_schur_512() {
    schur_parity::<Bits512>();
}

fn analytic_schur<T: FloatT>() {
    // Orthogonal signed basis: q0'q1 is exactly zero even though its two
    // summands are nonzero. Canonical zero-weight/basis columns stay present.
    let b = SampledBlock {
        row_start: 0,
        column_start: 0,
        dim: 1,
        basis_rows: 2,
        basis_cols: 2,
        basis: vec![T::one(), T::one(), T::one(), -T::one()],
        weights: vec![c::<T>(2), c(-3)],
    };
    let mut work = SampledSchurWorkspace::new(&b);
    work.update_with_pool(&b, &Matrix::identity(2), None);
    assert_eq!(work.entry(&b, 0, 0), c(16));
    assert_eq!(work.entry(&b, 1, 1), c(36));
    assert_eq!(work.entry(&b, 0, 1), T::zero());
    // An exactly represented small positive quadratic pairing must survive;
    // no absolute-tolerance floor is used for this cancellation-sensitive cell.
    let delta = c::<T>(2).powi(-(T::precision_bits() as i32 / 4));
    let r = Matrix::from(&[[T::one(), T::zero()], [T::zero(), T::one() + delta]]);
    work.update_with_pool(&b, &r, None);
    let cross = -(c::<T>(2) * delta + delta * delta);
    let expected = -c::<T>(6) * cross * cross;
    let actual = work.entry(&b, 0, 1);
    assert!((actual - expected).abs() <= c::<T>(4096) * T::epsilon() * expected.abs());
    assert!(actual < T::zero());
}
#[test]
fn sampled_schur_analytic_f64() {
    analytic_schur::<f64>();
}
#[test]
fn sampled_schur_analytic_256() {
    analytic_schur::<Bits256>();
}
#[test]
fn sampled_schur_analytic_512() {
    analytic_schur::<Bits512>();
}

#[test]
fn primitive_offdiagonal_half_and_scaling() {
    let block = SampledBlock {
        row_start: 0,
        column_start: 0,
        dim: 2,
        basis_rows: 1,
        basis_cols: 1,
        basis: vec![2.],
        weights: vec![3., -4., 5.],
    };
    let mut op = SampledOperator::new(CscMatrix::zeros((3, 3)), vec![block]).unwrap();
    let a = op.materialize();
    assert_eq!(a.colptr, vec![0, 1, 2, 3]);
    assert_eq!(a.rowval, vec![0, 1, 2]);
    close(a.nzval[0], 12.);
    close(a.nzval[1], -16. * std::f64::consts::FRAC_1_SQRT_2);
    close(a.nzval[2], 20.);
    let mut expected = a.clone();
    let d = [2., 3., 4.];
    let e = [0.5; 3];
    expected.lrscale(&e, &d);
    op.scale(&d, &e);
    for (&actual, &expected) in op.materialize().nzval.iter().zip(&expected.nzval) {
        close(actual, expected);
    }
}

#[test]
fn stored_psd_zeros_do_not_invent_primitive_columns() {
    let block = SampledBlock {
        row_start: 1,
        column_start: 1,
        dim: 1,
        basis_rows: 1,
        basis_cols: 1,
        basis: vec![2.],
        weights: vec![3.],
    };
    // PSD row 1 has zeros before, within, and after the primitive range.
    // Ordinary row 0's explicit zero must survive constructor normalization.
    let linear = CscMatrix::new(
        3,
        3,
        vec![0, 2, 3, 5],
        vec![0, 1, 1, 1, 2],
        vec![0., 0., -0., 0., 4.],
    );
    let op = SampledOperator::new(linear, vec![block]).unwrap();
    assert_eq!(op.linear().colptr, vec![0, 1, 1, 2]);
    assert_eq!(op.linear().rowval, vec![0, 2]);
    assert_eq!(op.linear().nzval, vec![0., 4.]);
    let materialized = op.materialize();
    assert!(materialized.check_format().is_ok());
    for col in 0..3 {
        let rows = &materialized.rowval[materialized.colptr[col]..materialized.colptr[col + 1]];
        assert_eq!(rows.contains(&1), col == 1);
    }
    let mut work = SampledWorkspace::new(&op);
    let mut result = vec![0.; 3];
    op.apply(&mut result, &[7., 2., 3.], 1., 0., &mut work);
    assert_eq!(result, vec![0., 24., 12.]);
}

#[test]
fn sampled_constructor_rejects_nonfinite_inputs() {
    let block = SampledBlock {
        row_start: 1,
        column_start: 0,
        dim: 1,
        basis_rows: 1,
        basis_cols: 1,
        basis: vec![2.],
        weights: vec![3.],
    };
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let linear = CscMatrix::new(2, 1, vec![0, 1], vec![0], vec![bad]);
        assert!(SampledOperator::new(linear, vec![block.clone()]).is_err());
        let mut invalid = block.clone();
        invalid.basis[0] = bad;
        assert!(SampledOperator::new(CscMatrix::zeros((2, 1)), vec![invalid]).is_err());
        let mut invalid = block.clone();
        invalid.weights[0] = bad;
        assert!(SampledOperator::new(CscMatrix::zeros((2, 1)), vec![invalid]).is_err());
    }
}

fn pooled_gram<T: FloatT>() {
    let mut b = block::<T>(0, 0, 3);
    b.basis_rows = 24;
    b.basis_cols = 8;
    b.basis = (0..192)
        .map(|i| c::<T>((i % 11) as i32 - 5) / c(8))
        .collect();
    b.weights = (0..48).map(|i| c::<T>((i % 7) as i32 - 3) / c(4)).collect();
    let mut serial = SampledSchurWorkspace::new(&b);
    let mut pooled = SampledSchurWorkspace::new(&b);
    let mut rinv = Matrix::zeros((b.side(), b.side()));
    for i in 0..b.side() {
        rinv[(i, i)] = c(1);
    }
    for width in [1, 2, 4, 8, 1, 4] {
        let pool = std::sync::Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(width)
                .build()
                .unwrap(),
        );
        rinv[(0, 0)] += c::<T>(1) / c(16);
        serial.update_with_pool(&b, &rinv, None);
        pooled.update_with_pool(&b, &rinv, Some(&pool));
        assert_eq!(pooled.plan_threads, width);
        assert_eq!(pooled.v.data(), serial.v.data());
        assert_eq!(pooled.gram.data(), serial.gram.data());
        let pointers = (
            pooled.u.data().as_ptr(),
            pooled.v.data().as_ptr(),
            pooled.gram.data().as_ptr(),
        );
        let tiles = (pooled.gemm_tile, pooled.syrk_tile);
        pooled.update_with_pool(&b, &rinv, Some(&pool));
        assert_eq!(
            pointers,
            (
                pooled.u.data().as_ptr(),
                pooled.v.data().as_ptr(),
                pooled.gram.data().as_ptr()
            )
        );
        assert_eq!(tiles, (pooled.gemm_tile, pooled.syrk_tile));
        for p in 0..b.column_count() {
            for q in 0..=p {
                assert_eq!(pooled.entry(&b, p, q), serial.entry(&b, p, q));
            }
        }
    }
}
#[test]
fn pooled_gram_f64() {
    pooled_gram::<f64>();
}
#[test]
fn pooled_gram_mpfr256() {
    pooled_gram::<Bits256>();
}
#[test]
fn pooled_gram_mpfr512() {
    pooled_gram::<Bits512>();
}

fn pooled_operators<T: FloatT>() {
    let make = |row_start, column_start| SampledBlock {
        row_start,
        column_start,
        dim: 2,
        basis_rows: 24,
        basis_cols: 4,
        basis: (0..96)
            .map(|i| c::<T>((i % 11) as i32 - 5) / c(7))
            .collect(),
        weights: (0..12).map(|i| c::<T>((i % 5) as i32 - 2) / c(3)).collect(),
    };
    let first = make(1, 1);
    let second = make(1 + first.row_count(), 3);
    let empty = SampledBlock {
        row_start: second.row_start + second.row_count(),
        column_start: 2,
        dim: 1,
        basis_rows: 3,
        basis_cols: 0,
        basis: vec![],
        weights: vec![],
    };
    let (m, n) = (empty.row_start + empty.row_count() + 1, 16);
    let linear = CscMatrix::new_from_triplets(
        m,
        n,
        vec![0, m - 1, 0],
        vec![1, 3, n - 1],
        vec![c(2), c(-3), c(1)],
    );
    // Deliberately unsorted rows and overlapping primitive column ranges.
    let mut operator = SampledOperator::new(linear, vec![second, empty, first]).unwrap();
    let mut serial = SampledWorkspace::new(&operator);
    let mut pooled = SampledWorkspace::new(&operator);
    let mut x: Vec<_> = (0..n).map(|i| c::<T>(i as i32 - 8) / c(5)).collect();
    let mut z: Vec<_> = (0..m).map(|i| c::<T>((i % 17) as i32 - 8) / c(9)).collect();
    let same = |a: &[T], b: &[T]| {
        assert_eq!(a, b);
        for (a, b) in a.iter().zip(b) {
            assert_eq!(a.is_sign_negative(), b.is_sign_negative());
        }
    };
    let sizes = |w: &SampledWorkspace<T>| {
        w.blocks
            .iter()
            .map(|b| (b.forward.len(), b.adjoint.len()))
            .collect::<Vec<_>>()
    };
    let pointers = |w: &SampledWorkspace<T>| {
        w.blocks
            .iter()
            .map(|b| {
                (
                    b.forward.as_ptr(),
                    b.adjoint.as_ptr(),
                    b.panel.data().as_ptr(),
                    b.square.data().as_ptr(),
                )
            })
            .collect::<Vec<_>>()
    };
    let mut saved = None;
    for width in [1, 2, 4, 8, 1, 4] {
        let pool = std::sync::Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(width)
                .build()
                .unwrap(),
        );
        for (alpha, beta) in [
            (T::zero(), -c::<T>(2)),
            (T::one(), T::zero()),
            (-c::<T>(3) / c(2), c::<T>(2) / c(7)),
            (T::one(), T::one()),
        ] {
            let mut ys = z.clone();
            let mut yp = z.clone();
            let mut ts = x.clone();
            let mut tp = x.clone();
            let before = sizes(&pooled);
            operator.apply(&mut ys, &x, alpha, beta, &mut serial);
            operator.apply_with_pool(&mut yp, &x, alpha, beta, &mut pooled, Some(&pool));
            operator.apply_transpose(&mut ts, &z, alpha, beta, &mut serial);
            operator.apply_transpose_with_pool(&mut tp, &z, alpha, beta, &mut pooled, Some(&pool));
            same(&ys, &yp);
            same(&ts, &tp);
            if width == 1 || alpha == T::zero() {
                assert_eq!(sizes(&pooled), before);
            }
            if width > 1 && alpha != T::zero() {
                let sizes = sizes(&pooled);
                assert!(sizes.iter().map(|s| s.0).sum::<usize>() <= m);
                assert_eq!(sizes.iter().map(|s| s.1).sum::<usize>(), 24);
                assert_eq!(sizes[1], (0, 0));
                if let Some(saved) = &saved {
                    assert_eq!(&pointers(&pooled), saved);
                } else {
                    saved = Some(pointers(&pooled));
                }
            }
            if let Some(saved) = &saved {
                assert_eq!(&pointers(&pooled), saved);
            }
        }
        // Reuse the same storage with changed inputs and factor weights.
        for v in &mut x {
            *v += c::<T>(1) / c(32);
        }
        for v in &mut z {
            *v -= c::<T>(1) / c(64);
        }
        operator.scale(&vec![c::<T>(3) / c(4); n], &vec![T::one(); m]);
    }
    assert!(serial
        .blocks
        .iter()
        .all(|b| b.forward.is_empty() && b.adjoint.is_empty()));
    // A live wide pool does not allocate caches for tiny work or one block.
    let tiny = SampledOperator::new(
        CscMatrix::zeros((3, 1)),
        vec![SampledBlock {
            row_start: 0,
            column_start: 0,
            dim: 1,
            basis_rows: 2,
            basis_cols: 1,
            basis: vec![T::one(); 2],
            weights: vec![T::one()],
        }],
    )
    .unwrap();
    let mut tiny_work = SampledWorkspace::new(&tiny);
    let pool = std::sync::Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(8)
            .build()
            .unwrap(),
    );
    tiny.apply_with_pool(
        &mut vec![T::zero(); 3],
        &[T::one()],
        T::one(),
        T::zero(),
        &mut tiny_work,
        Some(&pool),
    );
    assert!(tiny_work.blocks[0].forward.is_empty());
}
/// `dim = 1` blocks cannot split at the `s` level; the leaf subdivision by
/// basis column (adjoint) or output band (forward) must stay bitwise equal to
/// the serial evaluation.
fn pooled_dim1_operators<T: FloatT>() {
    let make = |row_start, column_start| SampledBlock {
        row_start,
        column_start,
        dim: 1,
        basis_rows: 12,
        basis_cols: 8,
        basis: (0..96)
            .map(|i| c::<T>((i % 9) as i32 - 4) / c(6))
            .collect(),
        weights: (0..8).map(|i| c::<T>((i % 4) as i32 - 1) / c(3)).collect(),
    };
    let first = make(0, 0);
    let second = make(first.row_count(), 3);
    let (m, n) = (second.row_start + second.row_count(), 16);
    // Every row is sampled, so the linear part is empty.
    let linear = CscMatrix::zeros((m, n));
    let operator = SampledOperator::new(linear, vec![first, second]).unwrap();
    let mut serial = SampledWorkspace::new(&operator);
    let mut pooled = SampledWorkspace::new(&operator);
    let x: Vec<_> = (0..n).map(|i| c::<T>(i as i32 - 6) / c(7)).collect();
    let z: Vec<_> = (0..m).map(|i| c::<T>((i % 13) as i32 - 5) / c(4)).collect();
    for width in [2, 4, 8] {
        let pool = std::sync::Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(width)
                .build()
                .unwrap(),
        );
        let (mut ys, mut yp) = (z.clone(), z.clone());
        let (mut ts, mut tp) = (x.clone(), x.clone());
        operator.apply(&mut ys, &x, T::one(), -c::<T>(2), &mut serial);
        operator.apply_with_pool(&mut yp, &x, T::one(), -c::<T>(2), &mut pooled, Some(&pool));
        operator.apply_transpose(&mut ts, &z, -c::<T>(3) / c(2), c::<T>(2), &mut serial);
        operator.apply_transpose_with_pool(
            &mut tp,
            &z,
            -c::<T>(3) / c(2),
            c::<T>(2),
            &mut pooled,
            Some(&pool),
        );
        assert_eq!(ys, yp);
        assert_eq!(ts, tp);
    }
}
#[test]
fn pooled_dim1_operators_f64() {
    pooled_dim1_operators::<f64>();
}
#[test]
fn pooled_dim1_operators_mpfr256() {
    pooled_dim1_operators::<Bits256>();
}
#[test]
fn pooled_dim1_operators_mpfr512() {
    pooled_dim1_operators::<Bits512>();
}
#[test]
fn pooled_operators_f64() {
    pooled_operators::<f64>();
}
#[test]
fn pooled_operators_mpfr256() {
    pooled_operators::<Bits256>();
}
#[test]
fn pooled_operators_mpfr512() {
    pooled_operators::<Bits512>();
}

fn duplicate_basis<T: FloatT>() {
    let b = SampledBlock {
        row_start: 0,
        column_start: 0,
        dim: 2,
        basis_rows: 2,
        basis_cols: 6,
        basis: vec![
            c(1),
            c(2),
            c(-1),
            c(3),
            c(1),
            c(2),
            c(0),
            c(0),
            c::<T>(1) + T::epsilon() * c(16),
            c(2),
            c(0),
            c(0),
        ],
        weights: (0..18).map(|i| c::<T>(i % 7 - 3) / c(4)).collect(),
    };
    let mut work = SampledSchurWorkspace::new(&b);
    assert_eq!(work.u.ncols(), 8); // Four exact vectors, including a near duplicate.
    assert_eq!(work.gram.size(), (8, 8));
    assert_eq!(work.pairs.len(), b.weights.len());
    let mut pooled = SampledSchurWorkspace::new(&b);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    let r = Matrix::identity(b.side());
    work.update_with_pool(&b, &r, None);
    pooled.update_with_pool(&b, &r, Some(&pool));
    for p in 0..b.column_count() {
        for q in 0..b.column_count() {
            assert_eq!(work.entry(&b, p, q), pooled.entry(&b, p, q));
        }
    }
    schur_parity_block(b);
}
#[test]
fn sampled_duplicate_basis_f64() {
    duplicate_basis::<f64>();
}
#[test]
fn sampled_duplicate_basis_256() {
    duplicate_basis::<Bits256>();
}
#[test]
fn sampled_duplicate_basis_512() {
    duplicate_basis::<Bits512>();
}

// The ordinary CSC block is the operator's only unbounded serial section. A
// row/column plan must reproduce the serial product exactly, including the
// zero-beta clearing and the alpha branches, once the pool engages.
fn pooled_linear_products<T: FloatT>() {
    // The two sampled blocks own the rows above the ordinary CSC part, which is
    // the operator's only unbounded serial section.
    let (ordinary_rows, cols_per_block) = (600usize, 80usize);
    let block = SampledBlock {
        row_start: 0,
        column_start: 0,
        dim: 4,
        basis_rows: 12,
        basis_cols: 8,
        basis: (0..96).map(|i| c::<T>((i % 7) as i32 - 3) / c(4)).collect(),
        weights: (0..80).map(|i| c::<T>((i % 5) as i32 - 2) / c(3)).collect(),
    };
    let rows_per_block = block.row_count();
    let first = SampledBlock {
        row_start: ordinary_rows,
        ..block.clone()
    };
    let second = SampledBlock {
        row_start: ordinary_rows + rows_per_block,
        column_start: cols_per_block,
        ..block.clone()
    };
    let (m, n) = (second.row_start + rows_per_block, 2 * cols_per_block);
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let (mut r, mut cidx, mut v) = (Vec::new(), Vec::new(), Vec::new());
    for column in 0..n {
        for _ in 0..400 {
            r.push((next() % ordinary_rows as u64) as usize);
            cidx.push(column);
            v.push(c::<T>((next() % 19) as i32 - 9) / c(7));
        }
    }
    let linear = CscMatrix::new_from_triplets(m, n, r, cidx, v);
    assert!(linear.nnz() >= 32768);
    let operator = SampledOperator::new(linear, vec![first, second]).unwrap();
    let mut serial = SampledWorkspace::new(&operator);
    let mut pooled = SampledWorkspace::new(&operator);
    let x: Vec<_> = (0..n).map(|i| c::<T>(i as i32 % 11 - 5) / c(9)).collect();
    let z: Vec<_> = (0..m).map(|i| c::<T>(i as i32 % 13 - 6) / c(5)).collect();
    for width in [1, 2, 4, 8] {
        let pool = std::sync::Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(width)
                .build()
                .unwrap(),
        );
        for (alpha, beta) in [
            (T::zero(), T::zero()),
            (T::zero(), -c::<T>(3)),
            (T::one(), T::one()),
            (-c::<T>(5) / c(2), c::<T>(2) / c(11)),
        ] {
            let mut ys = z.clone();
            let mut yp = z.clone();
            let mut ts = x.clone();
            let mut tp = x.clone();
            operator.apply(&mut ys, &x, alpha, beta, &mut serial);
            operator.apply_with_pool(&mut yp, &x, alpha, beta, &mut pooled, Some(&pool));
            operator.apply_transpose(&mut ts, &z, alpha, beta, &mut serial);
            operator.apply_transpose_with_pool(&mut tp, &z, alpha, beta, &mut pooled, Some(&pool));
            assert_eq!(ys, yp);
            assert_eq!(ts, tp);
        }
        if width > 1 {
            // The pooled path must own real lanes, not a serial fallback.
            assert!(pooled.linear_plan_workers > 1 && pooled.linear_plan.has_lanes());
        }
    }
}
#[test]
fn pooled_linear_products_f64() {
    pooled_linear_products::<f64>();
}
#[test]
fn pooled_linear_products_mpfr256() {
    pooled_linear_products::<Bits256>();
}
#[test]
fn pooled_linear_products_mpfr512() {
    pooled_linear_products::<Bits512>();
}
