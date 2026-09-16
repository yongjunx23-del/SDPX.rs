#![allow(non_snake_case)]

use sdpx_solver::{algebra::*, solver::*};
use sdpx_solver::{default_infinity, get_infinity, set_infinity};

#[allow(clippy::type_complexity)]
fn presolve_test_data() -> (
    CscMatrix<f64>,
    Vec<f64>,
    CscMatrix<f64>,
    Vec<f64>,
    Vec<SupportedConeT<f64>>,
) {
    let n = 3;
    let P = CscMatrix::identity(n);
    let mut A2 = CscMatrix::identity(n);
    A2.negate();
    let mut A = CscMatrix::vcat(&P, &A2).unwrap();
    A.scale(2.);

    let c = vec![3., -2., 1.];
    let b = vec![1.; 2 * n];

    let cones = vec![NonnegativeConeT(3), NonnegativeConeT(3)];

    (P, c, A, b, cones)
}

#[test]
fn test_presolve_single_unbounded() {
    let (P, c, A, mut b, cones) = presolve_test_data();

    b[3] = 1e30_f64;

    let settings = DefaultSettings::default();

    let mut solver = DefaultSolver::new(&P, &c, &A, &b, &cones, settings).unwrap();

    solver.solve();

    assert_eq!(solver.solution.status, SolverStatus::Solved);
    assert_eq!(solver.variables.z.len(), 5);
    assert_eq!(solver.solution.z[3], 0.);
    assert_eq!(solver.solution.s[3], get_infinity());
}

#[test]
fn test_presolve_single_unbounded_2() {
    // tests against https://github.com/oxfordcontrol/Clarabel.rs/issues/127
    let (P, c, A, mut b, _) = presolve_test_data();

    b[4] = 1e30_f64;

    let cones = vec![ZeroConeT(2), NonnegativeConeT(4)];

    let settings = DefaultSettings::default();

    let mut solver = DefaultSolver::new(&P, &c, &A, &b, &cones, settings).unwrap();

    solver.solve();

    assert_eq!(solver.solution.status, SolverStatus::Solved);
    assert_eq!(solver.variables.z.len(), 5);
}

#[test]
fn test_presolve_completely_redundant_cone() {
    let (P, c, A, mut b, cones) = presolve_test_data();

    b[0] = 1e30_f64;
    b[1] = 1e30_f64;
    b[2] = 1e30_f64;

    let settings = DefaultSettings::default();

    let mut solver = DefaultSolver::new(&P, &c, &A, &b, &cones, settings).unwrap();

    solver.solve();

    assert_eq!(solver.solution.status, SolverStatus::Solved);
    assert_eq!(solver.variables.z.len(), 3);
    assert_eq!(solver.solution.z[0..3], vec![0., 0., 0.]);
    let inf = get_infinity();
    assert_eq!(solver.solution.s[0..3], vec![inf, inf, inf]);
    let refsol = vec![-0.5, 2., -0.5];
    assert!(solver.solution.x.dist(&refsol) <= 1e-6);
}

#[test]
fn test_presolve_every_constraint_redundant() {
    let (P, mut c, A, mut b, cones) = presolve_test_data();

    b.fill(1e30_f64);

    let settings = DefaultSettings::default();

    let mut solver = DefaultSolver::new(&P, &c, &A, &b, &cones, settings).unwrap();

    solver.solve();

    assert_eq!(solver.solution.status, SolverStatus::Solved);
    assert_eq!(solver.variables.z.len(), 0);
    assert!(solver.solution.x.dist(c.negate()) <= 1e-6);
}

#[test]
fn test_presolve_settable_bound() {
    default_infinity();
    let default_bound = get_infinity();
    set_infinity(1e21_f64);
    assert_eq!(get_infinity(), 1e21_f64);
    default_infinity();
    assert_eq!(get_infinity(), default_bound);
}

#[test]
fn exact_equalities_restore_original_dual_and_slack() {
    let a = CscMatrix::<f64>::new(
        4,
        2,
        vec![0, 3, 5],
        vec![0, 1, 3, 2, 3],
        vec![1., 2., 1., 1., 1.],
    );
    let b = vec![1., 2., 2., 3.];
    let p = CscMatrix::identity(2);
    let q = vec![0.; 2];
    let cones = vec![ZeroConeT(4)];
    let mut settings = DefaultSettings::default();
    settings.verbose = false;
    let mut solver = DefaultSolver::new(&p, &q, &a, &b, &cones, settings.clone()).unwrap();
    assert_eq!(solver.variables.z.len(), 2);
    solver.solve();
    assert_eq!(solver.solution.status, SolverStatus::Solved);
    assert!(solver.solution.x.dist(&[1., 2.]) < 1e-8);
    assert_eq!(solver.solution.z[1], 0.);
    assert_eq!(solver.solution.z[3], 0.);
    assert_eq!(solver.solution.s, vec![0.; 4]);
    let z = &solver.solution.z;
    assert!((solver.solution.x[0] + z[0] + 2. * z[1] + z[3]).abs() < 1e-8);
    assert!((solver.solution.x[1] + z[2] + z[3]).abs() < 1e-8);
    settings.presolve_enable = false;
    let untouched = DefaultSolver::new(&p, &q, &a, &b, &cones, settings).unwrap();
    assert_eq!(untouched.variables.z.len(), 4);
}

#[test]
fn inconsistent_equalities_preserve_infeasibility_ray() {
    let a = CscMatrix::<f64>::new(3, 1, vec![0, 3], vec![0, 1, 2], vec![1., 2., 1.]);
    let b = vec![1., 2., 3.];
    let mut settings = DefaultSettings::default();
    settings.verbose = false;
    let mut solver = DefaultSolver::new(
        &CscMatrix::zeros((1, 1)),
        &[0.],
        &a,
        &b,
        &[ZeroConeT(3)],
        settings,
    )
    .unwrap();
    assert_eq!(solver.variables.z.len(), 2);
    solver.solve();
    assert_eq!(solver.solution.status, SolverStatus::PrimalInfeasible);
    let z = &solver.solution.z;
    assert_eq!(z[1], 0.);
    assert!((z[0] + 2. * z[1] + z[2]).abs() < 1e-8);
    assert!(z[0] + 2. * z[1] + 3. * z[2] < -0.9);
}
