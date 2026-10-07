use super::*;
use num_traits::{FromPrimitive, One, Zero};
use sdpx_arithmetic::{Bits256, Bits512, Bits768, MpFloat, Scalar};
use std::sync::Arc;
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
        weights: (0..triangular_number(dim) * k)
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

fn exact_with_zero_sign<T: FloatT>(a: &[T], b: &[T]) {
    assert_eq!(a, b);
    for (&x, &y) in a.iter().zip(b) {
        assert_eq!(x.is_zero(), y.is_zero());
        if x.is_zero() {
            assert_eq!(x.is_sign_negative(), y.is_sign_negative());
        }
    }
}

fn cache_block<T: FloatT>(row_start: usize, column_start: usize, h: usize) -> SampledBlock<T> {
    let k = 21;
    let basis = (0..h * k)
        .map(|i| {
            let value = (i as i32 % 11) - 5;
            c::<T>(value) / c(7)
        })
        .collect();
    let weights = (0..k)
        .map(|i| {
            if i % 5 == 0 {
                T::zero()
            } else {
                c::<T>((i as i32 % 7) - 3)
            }
        })
        .collect();
    SampledBlock {
        row_start,
        column_start,
        dim: 1,
        basis_rows: h,
        basis_cols: k,
        basis,
        weights,
    }
}

fn sampled_constant_cache<T: FloatT>() {
    let first = cache_block::<T>(0, 0, 3);
    let second = cache_block::<T>(first.row_count(), 21, 20); // triangular_number(20) == 210 > 21.
    let m = first.row_count() + second.row_count();
    let n = 42;
    let operator = SampledOperator::new(CscMatrix::zeros((m, n)), vec![first, second]).unwrap();
    let cloned = operator.clone();
    let mut serial = SampledWorkspace::new(&operator);
    let mut pooled = SampledWorkspace::new(&operator);
    let cloned_work = SampledWorkspace::new(&cloned);

    assert_eq!(serial.blocks.len(), 2);
    assert_eq!(
        serial.blocks[0].constants.wdiag.get().unwrap().len(),
        21 * triangular_number(3)
    );
    assert_eq!(
        serial.blocks[1].constants.wdiag.get().unwrap().len(),
        21 * triangular_number(20)
    );
    let constant_elements = 21 * (triangular_number(3) + triangular_number(20));
    let shared_payload = constant_elements * std::mem::size_of::<T>();
    let old_three_workspace_payload = 3 * shared_payload;
    assert_eq!(
        serial
            .blocks
            .iter()
            .map(|b| b.constants.wdiag.get().unwrap().len())
            .sum::<usize>()
            * std::mem::size_of::<T>(),
        shared_payload
    );
    assert_eq!(
        old_three_workspace_payload - shared_payload,
        2 * shared_payload
    );
    assert_eq!(
        serial.blocks[0].constants.wdiag.get().unwrap().len() * std::mem::size_of::<T>(),
        21 * triangular_number(3) * std::mem::size_of::<T>()
    );
    assert!(Arc::ptr_eq(
        &serial.blocks[0].constants,
        &pooled.blocks[0].constants
    ));
    assert!(Arc::ptr_eq(
        &serial.blocks[0].constants,
        &cloned_work.blocks[0].constants
    ));
    assert!(!Arc::ptr_eq(
        &serial.blocks[0].constants,
        &serial.blocks[1].constants
    ));

    let x: Vec<_> = (0..n).map(|i| c::<T>((i as i32 % 9) - 4) / c(5)).collect();
    let z: Vec<_> = (0..m)
        .map(|i| c::<T>((i as i32 % 13) - 6) / c(11))
        .collect();
    let mut ys = vec![T::zero(); m];
    let mut yp = vec![T::zero(); m];
    operator.apply(&mut ys, &x, T::one(), T::zero(), &mut serial);
    let pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap(),
    );
    operator.apply_with_pool(&mut yp, &x, T::one(), T::zero(), &mut pooled, Some(&pool));
    exact_with_zero_sign(&ys, &yp);

    let mut ts = vec![T::zero(); n];
    let mut tp = vec![T::zero(); n];
    operator.apply_transpose(&mut ts, &z, T::one(), T::zero(), &mut serial);
    operator.apply_transpose_with_pool(&mut tp, &z, T::one(), T::zero(), &mut pooled, Some(&pool));
    exact_with_zero_sign(&ts, &tp);

    let materialized = operator.materialize();
    let mut expected = vec![T::zero(); m];
    materialized.gemv(&mut expected, &x, T::one(), T::zero());
    for (&actual, &reference) in ys.iter().zip(&expected) {
        close(actual, reference);
    }

    let wdiag_before = serial.blocks[0].constants.wdiag.get().unwrap().clone();
    let wdiag_ptr = serial.blocks[0].constants.wdiag.get().unwrap().as_ptr();
    let d = vec![c::<T>(2); n];
    let e = vec![c::<T>(3); m];
    let mut scaled = operator;
    scaled.scale(&d, &e);
    assert_eq!(
        serial.blocks[0].constants.wdiag.get().unwrap().as_ptr(),
        wdiag_ptr
    );
    assert_eq!(
        serial.blocks[0].constants.wdiag.get().unwrap(),
        &wdiag_before
    );
    let mut scaled_y = vec![T::zero(); m];
    scaled.apply(&mut scaled_y, &x, T::one(), T::zero(), &mut serial);
    let scaled_materialized = scaled.materialize();
    let mut scaled_expected = vec![T::zero(); m];
    scaled_materialized.gemv(&mut scaled_expected, &x, T::one(), T::zero());
    for (&actual, &reference) in scaled_y.iter().zip(&scaled_expected) {
        close(actual, reference);
    }

    let mut different_basis = cache_block::<T>(0, 0, 3);
    different_basis.basis[0] = c(19);
    let different = SampledOperator::new(
        CscMatrix::zeros((different_basis.row_count(), n)),
        vec![different_basis],
    )
    .unwrap();
    let different_work = SampledWorkspace::new(&different);
    assert!(!Arc::ptr_eq(
        &serial.blocks[0].constants,
        &different_work.blocks[0].constants
    ));
    assert_ne!(
        serial.blocks[0].constants.wdiag.get().unwrap(),
        different_work.blocks[0].constants.wdiag.get().unwrap()
    );
}

#[test]
fn sampled_constant_cache_f64() {
    sampled_constant_cache::<f64>();
}
#[test]
fn sampled_constant_cache_256() {
    sampled_constant_cache::<Bits256>();
}
#[test]
fn sampled_constant_cache_512() {
    sampled_constant_cache::<Bits512>();
}
#[test]
fn sampled_constant_cache_768() {
    sampled_constant_cache::<Bits768>();
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
                .enumerate()
                .fold(Oracle::from_i32(0).unwrap(), |sum, (_, (&a, &b))| {
                    sum + a * b
                });
            oracle_close(work.entry(&b, p, q), expected);
            close(work.entry(&b, p, q), work.entry(&b, q, p));
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

fn sampled_adjoint_abs_matches_materialized<T: FloatT>() {
    // The materialized view is used only as an independent small oracle here;
    // production accuracy work walks the factors directly. The two basis
    // columns cancel in the signed adjoint while their individual terms stay
    // nonzero, which is the failure mode this denominator must expose.
    let block = SampledBlock {
        row_start: 0,
        column_start: 0,
        dim: 1,
        basis_rows: 2,
        basis_cols: 2,
        basis: vec![c::<T>(1), c(1), c(1), c(-1)],
        weights: vec![c(1), c(1)],
    };
    let op = SampledOperator::new(
        CscMatrix::zeros((block.row_count(), block.column_count())),
        vec![block],
    )
    .unwrap();
    let z = vec![c::<T>(1), T::zero(), c(-1)];
    let mut got = vec![T::zero(); op.dims().1];
    let mut work = SampledWorkspace::new(&op);
    op.add_adjoint_abs(&mut got, &z, &mut work, None);

    let mut signed = vec![T::zero(); op.dims().1];
    op.apply_transpose(&mut signed, &z, T::one(), T::zero(), &mut work);
    assert!(signed.iter().all(|&v| v == T::zero()));
    assert!(got.iter().all(|&v| v > T::one()));

    let materialized = op.materialize();
    let mut expected = vec![T::zero(); op.dims().1];
    for col in 0..materialized.n {
        for idx in materialized.colptr[col]..materialized.colptr[col + 1] {
            expected[col] += T::abs(materialized.nzval[idx] * z[materialized.rowval[idx]]);
        }
    }
    for (&actual, &want) in got.iter().zip(&expected) {
        close(actual, want);
    }
}

#[test]
fn sampled_adjoint_abs_f64() {
    sampled_adjoint_abs_matches_materialized::<f64>();
}

#[test]
fn sampled_adjoint_abs_256() {
    sampled_adjoint_abs_matches_materialized::<Bits256>();
}

#[test]
fn sampled_adjoint_abs_512() {
    sampled_adjoint_abs_matches_materialized::<Bits512>();
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
        assert_eq!(pooled.gram, serial.gram);
        let pointers = (pooled.v.data().as_ptr(), pooled.gram.as_ptr());
        let tiles = (pooled.gemm_tile, pooled.syrk_tile);
        pooled.update_with_pool(&b, &rinv, Some(&pool));
        assert_eq!(pointers, (pooled.v.data().as_ptr(), pooled.gram.as_ptr()));
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
            .map(|b| (b.forward.as_ptr(), b.adjoint.as_ptr()))
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
        basis: (0..96).map(|i| c::<T>((i % 9) as i32 - 4) / c(6)).collect(),
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
    assert_eq!(work.count, 4); // Four exact vectors, including a near duplicate.
    assert_eq!(
        work.gram.len(),
        if T::precision_bits() > 64 { 36 } else { 64 }
    );
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
// PR-05: the block-diagonal update must reproduce the old dense
// Rinv·(I_dim ⊗ Qbasis) product bit-for-bit, including stored zeros.
fn structured_v_parity<T: FloatT>() {
    for dim in [1usize, 2, 3] {
        let (h, cols) = (4usize, 5usize);
        let mut basis: Vec<T> = (0..h * cols)
            .map(|i| c::<T>((i % 9) as i32 - 4) / c(7))
            .collect();
        // An all-zero basis column exercises the ±0 fixup path.
        for i in 0..h {
            basis[i + (cols - 1) * h] = T::zero();
        }
        let b = SampledBlock {
            row_start: 0,
            column_start: 0,
            dim,
            basis_rows: h,
            basis_cols: cols,
            basis,
            weights: (0..triangular_number(dim) * cols)
                .map(|i| c::<T>((i % 6) as i32 - 2) / c(5))
                .collect(),
        };
        let side = b.side();
        let mut rinv = Matrix::identity(side);
        for j in 0..side {
            for i in 0..j {
                rinv[(i, j)] = c::<T>(((i * 3 + j) % 11) as i32 - 5) / c(9);
            }
        }
        let mut work = SampledSchurWorkspace::new(&b);
        work.update_with_pool(&b, &rinv, None);
        // Reference: the explicit dense block-diagonal u, dense GEMM.
        let count = work.count;
        let ub = work.ub.as_deref().unwrap_or(&b.basis);
        let rank = dim * count;
        let mut u = Matrix::zeros((side, rank));
        for r in 0..dim {
            for k in 0..count {
                for i in 0..h {
                    u[(r * h + i, r * count + k)] = ub[i + k * h];
                }
            }
        }
        let mut vref = Matrix::zeros((side, rank));
        vref.mul(&rinv, &u, T::one(), T::zero());
        // MPFR uses the ordered dot_fma kernel, so the structured product is
        // bitwise identical. f64 routes to BLAS whose accumulation order is
        // shape-dependent — compare within rounding there.
        if T::precision_bits() > 64 {
            assert_eq!(work.v.data(), vref.data(), "dim={dim}");
        } else {
            for (&a, &b) in work.v.data().iter().zip(vref.data()) {
                close(a, b);
            }
        }
        let mut gref = Matrix::zeros((rank, rank));
        gref.syrk(&vref.t(), T::one(), T::zero(), MatrixTriangle::Triu);
        if T::precision_bits() > 64 {
            let packed: Vec<T> = (0..rank)
                .flat_map(|j| gref.data()[j * rank..j * rank + j + 1].iter().copied())
                .collect();
            assert_eq!(work.gram, packed, "dim={dim}");
        } else {
            for (&a, &b) in work.gram.iter().zip(gref.data()) {
                close(a, b);
            }
        }
    }
}
#[test]
fn structured_v_parity_f64() {
    structured_v_parity::<f64>();
}
#[test]
fn structured_v_parity_256() {
    structured_v_parity::<Bits256>();
}
#[test]
fn structured_v_parity_512() {
    structured_v_parity::<Bits512>();
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
#[ignore = "extended: MPFR pool/threading sweep; default f64 covers the equivalence logic"]
fn pooled_linear_products_mpfr256() {
    pooled_linear_products::<Bits256>();
}
#[test]
#[ignore = "extended: MPFR pool/threading sweep; default f64 covers the equivalence logic"]
fn pooled_linear_products_mpfr512() {
    pooled_linear_products::<Bits512>();
}

// Independent high-precision dense compositions exercise the fused sampled
// operators, signed/zero weights, repeated bases and non-diagonal graded factors.
fn fused_oracle<T: FloatT>() {
    for dim in [1, 2, 3] {
        let mut b = block::<T>(0, 0, dim);
        b.basis[4] = b.basis[0];
        b.basis[5] = b.basis[1];
        let n = b.side();
        let mut r = Matrix::<T>::zeros((n, n));
        for j in 0..n {
            for i in 0..n {
                r[(i, j)] = if i == j {
                    c::<T>(2).powi(i as i32 * 3 - 5)
                } else {
                    c::<T>((i as i32 - j as i32) % 3) / c::<T>(32)
                };
            }
        }
        let mut schur = SampledSchurWorkspace::new(&b);
        schur.update_with_pool(&b, &r, None);
        let bo = SampledBlock {
            row_start: 0,
            column_start: 0,
            dim,
            basis_rows: b.basis_rows,
            basis_cols: b.basis_cols,
            basis: b.basis.iter().map(|&v| lift(v)).collect(),
            weights: b.weights.iter().map(|&v| lift(v)).collect(),
        };
        let op = SampledOperator::new(
            CscMatrix::zeros((b.row_count(), b.column_count())),
            vec![bo],
        )
        .unwrap();
        let mut work = SampledWorkspace::new(&op);
        let mut ro = Matrix::<Oracle>::zeros((n, n));
        for j in 0..n {
            for i in 0..n {
                ro[(i, j)] = lift(r[(i, j)]);
            }
        }
        let x: Vec<T> = (0..b.column_count())
            .map(|i| c::<T>(i as i32 - 3) / c(8))
            .collect();
        let xo: Vec<_> = x.iter().map(|&v| lift(v)).collect();
        let mut ax = vec![Oracle::zero(); b.row_count()];
        op.apply(&mut ax, &xo, Oracle::one(), Oracle::zero(), &mut work);
        let mut a = Matrix::<Oracle>::zeros((n, n));
        svec_to_mat(&mut a, &ax);
        let mut temp = Matrix::<Oracle>::zeros((n, n));
        let mut expected = Matrix::<Oracle>::zeros((n, n));
        temp.mul(&ro, &a, Oracle::one(), Oracle::zero());
        expected.mul(&temp, &ro.t(), Oracle::one(), Oracle::zero());
        let mut actual = Matrix::<T>::zeros((n, n));
        schur.inverse_forward(&b, &x, &mut actual);
        for (&a, &e) in actual.data().iter().zip(expected.data()) {
            oracle_close(a, e);
        }
        // Adjoint accepts any symmetric half-scaled RHS U.
        let mut u = Matrix::<T>::zeros((n, n));
        for j in 0..n {
            for i in 0..=j {
                u[(i, j)] = c::<T>(i as i32 - j as i32 + 2) / c(16);
                u[(j, i)] = u[(i, j)];
            }
        }
        for j in 0..n {
            for i in 0..n {
                a[(i, j)] = lift(u[(i, j)]);
            }
        }
        temp.mul(&ro.t(), &a, Oracle::one(), Oracle::zero());
        expected.mul(&temp, &ro, Oracle::one(), Oracle::zero());
        mat_to_svec(&mut ax, &expected);
        let mut adj = vec![Oracle::zero(); b.column_count()];
        op.apply_transpose(&mut adj, &ax, Oracle::one(), Oracle::zero(), &mut work);
        let mut actual = vec![T::zero(); adj.len()];
        schur.inverse_adjoint(&b, &u, &mut actual);
        for (&a, &e) in actual.iter().zip(&adj) {
            oracle_close(a, e);
        }
    }
}
#[test]
fn fused_sampled_f64() {
    fused_oracle::<f64>();
}
#[test]
fn fused_sampled_256() {
    fused_oracle::<Bits256>();
}
#[test]
fn fused_sampled_512() {
    fused_oracle::<Bits512>();
}
