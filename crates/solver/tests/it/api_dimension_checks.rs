#![allow(non_snake_case)]
#![allow(clippy::type_complexity)]
use sdpx_solver::{algebra::*, solver::*};

// a collection of tests to ensure that data of
// incompatible dimension won't be accepted

fn api_dim_check_data() -> (
    CscMatrix<f64>,
    Vec<f64>,
    CscMatrix<f64>,
    Vec<f64>,
    Vec<SupportedConeT<f64>>,
) {
    let P = CscMatrix::<f64>::zeros((4, 4));
    let q = vec![0.; 4];
    let A = CscMatrix::<f64>::zeros((6, 4));
    let b = vec![0.; 6];
    let cones = vec![ZeroConeT(1), NonnegativeConeT(2), NonnegativeConeT(3)];
    (P, q, A, b, cones)
}

#[test]
fn api_dim_check_working() {
    // This example should work because dimensions are
    // all compatible.  All following checks vary one
    // of these sizes to test dimension checks

    let (P, q, A, b, cones) = api_dim_check_data();

    let settings = DefaultSettings::default();
    let _solver = DefaultSolver::new(&P, &q, &A, &b, &cones, settings);
}

#[test]
fn api_dim_check_bad_P() {
    let (_P, q, A, b, cones) = api_dim_check_data();
    let P = CscMatrix::<f64>::zeros((3, 3));

    let settings = DefaultSettings::default();
    assert!(DefaultSolver::new(&P, &q, &A, &b, &cones, settings).is_err());
}

#[test]
fn api_dim_check_bad_A_rows() {
    let (P, q, _A, b, cones) = api_dim_check_data();
    let A = CscMatrix::<f64>::zeros((5, 4));

    let settings = DefaultSettings::default();
    assert!(DefaultSolver::new(&P, &q, &A, &b, &cones, settings).is_err());
}

#[test]
fn api_dim_check_bad_A_cols() {
    let (P, q, _A, b, cones) = api_dim_check_data();
    let A = CscMatrix::<f64>::zeros((6, 3));

    let settings = DefaultSettings::default();
    assert!(DefaultSolver::new(&P, &q, &A, &b, &cones, settings).is_err());
}

#[test]
fn api_dim_check_P_not_square() {
    let (_P, q, A, b, cones) = api_dim_check_data();
    let P = CscMatrix::<f64>::zeros((4, 3));

    let settings = DefaultSettings::default();
    assert!(DefaultSolver::new(&P, &q, &A, &b, &cones, settings).is_err());
}

#[test]
fn api_dim_check_bad_cones() {
    let (P, q, A, b, _cones) = api_dim_check_data();
    let cones = vec![ZeroConeT(1), NonnegativeConeT(2), NonnegativeConeT(4)];

    let settings = DefaultSettings::default();
    assert!(DefaultSolver::new(&P, &q, &A, &b, &cones, settings).is_err());
}

#[test]
fn api_noncanonical_A_duplicate_rejected() {
    let (P, q, _A, b, cones) = api_dim_check_data();
    // Two entries in column 0 share row 0: a duplicate coordinate.
    let A = CscMatrix::new(6, 4, vec![0, 2, 2, 2, 2], vec![0, 0], vec![1., 1.]);

    let settings = DefaultSettings::default();
    assert!(matches!(
        DefaultSolver::new(&P, &q, &A, &b, &cones, settings),
        Err(SolverError::BadInputData(_))
    ));
}

#[test]
fn api_noncanonical_A_unsorted_rejected() {
    let (P, q, _A, b, cones) = api_dim_check_data();
    // Rows [1, 0] inside column 0 are not strictly increasing.
    let A = CscMatrix::new(6, 4, vec![0, 2, 2, 2, 2], vec![1, 0], vec![1., 1.]);

    let settings = DefaultSettings::default();
    assert!(matches!(
        DefaultSolver::new(&P, &q, &A, &b, &cones, settings),
        Err(SolverError::BadInputData(_))
    ));
}
