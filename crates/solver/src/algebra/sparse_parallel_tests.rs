use super::*;
use crate::solver::{core::cones::CompositeCone, core::traits::Residuals, *};

fn num<T: FloatT>(n: usize) -> T {
    T::from_usize(n).unwrap()
}
fn same<T: FloatT>(a: &[T], b: &[T]) {
    assert_eq!(a.len(), b.len());
    for (&a, &b) in a.iter().zip(b) {
        if a.is_nan() {
            assert!(b.is_nan());
        } else {
            assert_eq!(a, b);
            if a == T::zero() {
                assert_eq!(a.is_sign_negative(), b.is_sign_negative());
            }
        }
    }
}

fn kernel_equivalence<T: FloatT>() {
    let big = num::<T>(2).powi(T::precision_bits() as i32 + 8);
    let mut a = CscMatrix::new(
        9,
        9,
        vec![0, 3, 5, 8, 8, 10, 12, 14, 16, 16],
        vec![0, 1, 3, 0, 2, 0, 2, 5, 1, 4, 2, 5, 1, 3, 0, 4],
        vec![
            big,
            T::one(),
            -T::one(),
            T::one(),
            T::one(),
            -big,
            T::zero(),
            T::one(),
            -T::one(),
            T::one(),
            T::one(),
            -T::one(),
            T::one(),
            T::one(),
            T::one(),
            -T::one(),
        ],
    );
    // Exercise the kernel on a deliberately small, cancellation-rich matrix;
    // production's launch threshold is tested separately below.
    let mut plan = SparseParallel::new(&a);
    let entry_pointer = plan.entries.as_ptr();
    let row_pointer = plan.rowptr.as_ptr();
    let mut cones = CompositeCone::<T>::new(&[NonnegativeConeT(32768)]);
    let scales = [
        T::zero(),
        T::one(),
        -T::one(),
        num::<T>(3) / num::<T>(4),
        -T::one() / num::<T>(8),
    ];
    for threads in [1, 2, 4, 8, 1, 4] {
        cones.configure_threads(threads).unwrap();
        plan.configure(&a, cones.thread_pool());
        assert_eq!(plan.workers, threads);
        let lanes = plan.row_lanes.as_ptr();
        plan.configure(&a, cones.thread_pool());
        assert_eq!(plan.row_lanes.as_ptr(), lanes);
        assert_eq!(plan.entries.as_ptr(), entry_pointer);
        assert_eq!(plan.rowptr.as_ptr(), row_pointer);
        for transpose in [false, true] {
            for alpha in scales {
                for beta in scales {
                    for nan_seed in [false, true] {
                        let x = vec![T::one(); 9];
                        let mut serial: Vec<T> =
                            (0..9).map(|i| num::<T>(i) / num::<T>(8)).collect();
                        serial[6] = -T::zero(); // empty row/column signed-zero semantics
                        if nan_seed {
                            serial[7] = T::nan();
                        }
                        let mut parallel = serial.clone();
                        if transpose {
                            a.t().gemv(&mut serial, &x, alpha, beta);
                        } else {
                            a.gemv(&mut serial, &x, alpha, beta);
                        }
                        if let Some(pool) = &plan.pool {
                            pool.install(|| {
                                plan.apply_in_pool(&a, transpose, &mut parallel, &x, alpha, beta)
                            });
                        } else {
                            plan.apply_in_pool(&a, transpose, &mut parallel, &x, alpha, beta);
                        }
                        same(&serial, &parallel);
                    }
                }
            }
        }
        // Numerical updates must be read through the original CSC positions.
        a.nzval[1] += T::one() / num::<T>(16);
        a.nzval[6] = -a.nzval[6]; // preserve explicit zero in the pattern
    }
}

#[test]
fn sparse_products_f64() {
    kernel_equivalence::<f64>();
}
#[test]
fn sparse_products_mpfr256() {
    kernel_equivalence::<sdpx_arithmetic::Bits256>();
}
#[test]
fn sparse_products_mpfr512() {
    kernel_equivalence::<sdpx_arithmetic::Bits512>();
}

#[test]
fn sparse_threshold_empty_and_zero_patterns() {
    let mut cones = CompositeCone::<f64>::new(&[NonnegativeConeT(32768)]);
    cones.configure_threads(8).unwrap();
    for (m, n) in [(0, 0), (0, 9), (9, 0), (9, 9)] {
        let a = CscMatrix::<f64>::zeros((m, n));
        assert!(!worthwhile(&a));
        let mut plan = SparseParallel::new(&a);
        plan.configure(&a, cones.thread_pool());
        for transpose in [false, true] {
            let (rows, cols) = if transpose { (n, m) } else { (m, n) };
            let x = vec![f64::NAN; cols];
            let mut y = vec![f64::NAN; rows];
            cones
                .thread_pool()
                .unwrap()
                .install(|| plan.apply_in_pool(&a, transpose, &mut y, &x, 0., 0.));
            same(&y, &vec![0.; rows]);
        }
    }
    let a = CscMatrix::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![0., 0.]);
    assert!(!worthwhile(&a));
    let mut plan = SparseParallel::new(&a);
    plan.configure(&a, cones.thread_pool());
    let mut y = vec![1.; 2];
    cones
        .thread_pool()
        .unwrap()
        .install(|| plan.apply_in_pool(&a, false, &mut y, &[f64::NAN; 2], 1., 0.));
    assert!(y.iter().all(|v| v.is_nan())); // stored zeros are productive 0*NaN terms
}

fn live_residual_updates<T: FloatT>(soc: bool) {
    let m = 32768;
    let n = 4;
    let mut ptr = vec![0];
    let mut rows = Vec::new();
    for col in 0..n {
        rows.extend((col..m).step_by(n));
        ptr.push(rows.len());
    }
    let mut a = CscMatrix::new(m, n, ptr, rows, vec![T::one(); m]);
    assert!(worthwhile(&a));
    let kinds = if soc {
        vec![SecondOrderConeT(4096); 8]
    } else {
        vec![NonnegativeConeT(m)]
    };
    let settings = DefaultSettings {
        verbose: false,
        max_threads: 1,
        presolve_enable: false,
        input_sparse_dropzeros: false,
        equilibrate_enable: false,
        direct_solve_method: "qdldl".into(),
        kkt_form: "augmented".into(),
        ..DefaultSettings::<T>::default()
    };
    let mut solver = DefaultSolver::new(
        &CscMatrix::identity(n),
        &vec![T::one(); n],
        &a,
        &vec![T::one(); m],
        &kinds,
        settings,
    )
    .unwrap();
    for (i, v) in solver.variables.x.iter_mut().enumerate() {
        *v = num::<T>(i + 1) / num::<T>(8);
    }
    for (i, v) in solver.variables.z.iter_mut().enumerate() {
        *v = num::<T>(i % 7 + 1) / num::<T>(16);
    }
    for (i, v) in solver.variables.s.iter_mut().enumerate() {
        *v = num::<T>(i % 5 + 1) / num::<T>(32);
    }
    solver.variables.τ = T::one();
    solver.variables.κ = T::one();
    let mut serial = DefaultResiduals::new(n, m);
    for threads in [1, 2, 4, 8, 1, 4] {
        solver.cones.configure_threads(threads).unwrap();
        for (i, value) in a.nzval.iter_mut().enumerate() {
            *value += num::<T>(i % 3 + 1) / num::<T>(128);
        }
        solver.update_A(&a).unwrap();
        serial.update(&solver.variables, &solver.data);
        solver.residuals.update_with_pool(
            &solver.variables,
            &solver.data,
            solver.cones.thread_pool(),
        );
        same(&serial.rx, &solver.residuals.rx);
        same(&serial.rz, &solver.residuals.rz);
        same(&serial.rx_inf, &solver.residuals.rx_inf);
        same(&serial.rz_inf, &solver.residuals.rz_inf);
        same(
            &[
                serial.rτ,
                serial.dot_xPx,
                serial.dot_qx,
                serial.dot_bz,
                serial.dot_sz,
            ],
            &[
                solver.residuals.rτ,
                solver.residuals.dot_xPx,
                solver.residuals.dot_qx,
                solver.residuals.dot_bz,
                solver.residuals.dot_sz,
            ],
        );
        if let Some(plan) = &solver.residuals.sparse_parallel {
            assert_eq!(plan.workers, threads);
            if threads == 1 {
                assert!(plan.pool.is_none());
            } else {
                assert!(Arc::ptr_eq(
                    plan.pool.as_ref().unwrap(),
                    &solver.cones.thread_pool().unwrap()
                ));
            }
        } else {
            assert_eq!(threads, 1);
        }
    }
}

#[test]
fn sparse_live_lp_soc_f64() {
    live_residual_updates::<f64>(false);
    live_residual_updates::<f64>(true);
}
#[test]
fn sparse_live_lp_soc_mpfr256() {
    live_residual_updates::<sdpx_arithmetic::Bits256>(false);
    live_residual_updates::<sdpx_arithmetic::Bits256>(true);
}
#[test]
fn sparse_live_lp_soc_mpfr512() {
    live_residual_updates::<sdpx_arithmetic::Bits512>(false);
    live_residual_updates::<sdpx_arithmetic::Bits512>(true);
}

#[cfg(feature = "sdp")]
#[test]
fn sampled_factor_residuals_do_not_build_sparse_row_plan() {
    let block = SampledBlock {
        row_start: 0,
        column_start: 0,
        dim: 1,
        basis_rows: 32,
        basis_cols: 64,
        basis: vec![1.; 32 * 64],
        weights: vec![1.; 64],
    };
    let linear = CscMatrix::zeros((528, 64));
    let mut settings = DefaultSettings::default();
    settings.verbose = false;
    settings.max_threads = 8;
    settings.presolve_enable = false;
    settings.chordal_decomposition_enable = false;
    settings.kkt_form = "condensed".into();
    settings.direct_solve_method = "qdldl".into();
    let solver = DefaultSolver::new_sampled(
        &CscMatrix::identity(64),
        &[0.; 64],
        &linear,
        &[1.; 528],
        &[PSDTriangleConeT(32)],
        vec![block],
        settings,
    )
    .unwrap();
    assert!(worthwhile(&solver.data.A));
    assert!(solver.data.sampled.is_some());
    assert!(solver.residuals.sparse_parallel.is_none());
}
#[cfg(test)]
fn row_gather_parity<T: FloatT>() {
    let cv = |x: f64| T::from_f64(x).unwrap();
    // Upper triangle with duplicate entries, gaps, cancellation and zero.
    let mut a = CscMatrix::new(
        4,
        4,
        vec![0, 1, 4, 6, 9],
        vec![0, 0, 0, 1, 0, 2, 1, 2, 3],
        [1., 1e15, -1e15, 0., -0.125, 3., 2., -2., 0.25]
            .map(cv)
            .to_vec(),
    );
    let x = [0.125, -2., 3., -0.].map(cv);
    let mut k: CscMatrix<T> = a.t().into();
    // Both upper and lower storage must follow their respective CSC order.
    for (matrix, uplo) in [
        (&mut a, MatrixTriangle::Triu),
        (&mut k, MatrixTriangle::Tril),
    ] {
        let mut plan = SparseParallel::new_symmetric(matrix);
        for width in [1, 2, 4] {
            let pool = Arc::new(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(width)
                    .build()
                    .unwrap(),
            );
            plan.configure(matrix, Some(pool));
            for (alpha, beta) in [(1., 0.), (-1., 1.), (0.25, -2.), (0., 1.)] {
                let mut expected = [-0., 1., -2., 3.].map(cv).to_vec();
                let mut actual = expected.clone();
                matrix
                    .sym(uplo)
                    .symv(&mut expected, &x, cv(alpha), cv(beta));
                plan.symv(matrix, uplo, &mut actual, &x, cv(alpha), cv(beta));
                assert_eq!(actual, expected);
                for (a, b) in actual.iter().zip(&expected) {
                    assert_eq!(a.is_sign_negative(), b.is_sign_negative());
                }
            }
            // The same symbolic map must read changed values, not a copy.
            matrix.nzval[0] += cv(0.125);
        }
    }
}
#[test]
fn row_gather_f64() {
    row_gather_parity::<f64>();
}
#[test]
fn row_gather_mpfr256() {
    row_gather_parity::<sdpx_arithmetic::Bits256>();
}
#[test]
fn row_gather_mpfr512() {
    row_gather_parity::<sdpx_arithmetic::Bits512>();
}
