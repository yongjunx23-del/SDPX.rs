#![allow(non_snake_case)]

use sdpx_solver::{algebra::*, solver::*};

#[allow(clippy::type_complexity)]
fn basic_lp_data() -> (
    CscMatrix<f64>,
    Vec<f64>,
    CscMatrix<f64>,
    Vec<f64>,
    Vec<SupportedConeT<f64>>,
) {
    let P = CscMatrix::<f64>::zeros((3, 3));

    let I1 = CscMatrix::<f64>::identity(3);
    let mut I2 = CscMatrix::<f64>::identity(3);
    I2.negate();
    let mut A = CscMatrix::vcat(&I1, &I2).unwrap();
    A.scale(2.);

    let c = vec![3., -2., 1.];
    let b = vec![1.; 6];

    let cones = vec![NonnegativeConeT(3), NonnegativeConeT(3)];

    (P, c, A, b, cones)
}

#[test]
fn test_lp_feasible() {
    let (P, c, A, b, cones) = basic_lp_data();

    // A 1e-8 objective check needs a tighter gap than the 1e-8 default.
    let settings = DefaultSettings {
        tol_gap_abs: 1e-9,
        tol_gap_rel: 1e-9,
        ..DefaultSettings::default()
    };

    let mut solver = DefaultSolver::new(&P, &c, &A, &b, &cones, settings).unwrap();

    solver.solve();

    assert_eq!(solver.solution.status, SolverStatus::Solved);

    let refsol = vec![-0.5, 0.5, -0.5];
    assert!(solver.solution.x.dist(&refsol) <= 1e-8);

    let refobj = -3.;
    assert!(f64::abs(solver.solution.obj_val - refobj) <= 1e-8);
    assert!(f64::abs(solver.solution.obj_val_dual - refobj) <= 1e-8);
}

#[test]
fn test_lp_primal_infeasible() {
    let (P, c, A, mut b, cones) = basic_lp_data();

    b[0] = -1.;
    b[3] = -1.;

    let settings = DefaultSettings::default();

    let mut solver = DefaultSolver::new(&P, &c, &A, &b, &cones, settings).unwrap();

    solver.solve();

    assert_eq!(solver.solution.status, SolverStatus::PrimalInfeasible);
    assert!(solver.solution.obj_val.is_nan());
    assert!(solver.solution.obj_val_dual.is_nan());
}

#[test]
fn test_lp_dual_infeasible() {
    let (P, _c, mut A, b, cones) = basic_lp_data();

    A.nzval[1] = 1.; //swap lower bound on first variable to redundant upper bound
    let c = vec![1., 0., 0.];

    let settings = DefaultSettings::default();

    let mut solver = DefaultSolver::new(&P, &c, &A, &b, &cones, settings).unwrap();

    solver.solve();

    assert_eq!(solver.solution.status, SolverStatus::DualInfeasible);
    assert!(solver.solution.obj_val.is_nan());
    assert!(solver.solution.obj_val_dual.is_nan());
}

#[test]
fn test_lp_dual_infeasible_ill_cond() {
    let (P, _c, mut A, b, cones) = basic_lp_data();

    A.nzval[0] = f64::EPSILON;
    A.nzval[1] = 0.0;
    let c = vec![1., 0., 0.];

    let settings = DefaultSettings::default();

    let mut solver = DefaultSolver::new(&P, &c, &A, &b, &cones, settings).unwrap();

    solver.solve();

    assert_eq!(solver.solution.status, SolverStatus::DualInfeasible);
    assert!(solver.solution.obj_val.is_nan());
    assert!(solver.solution.obj_val_dual.is_nan());
}

#[test]
fn bounded_lp_with_wide_equality_border_uses_sparse_ldl() {
    // 200 bounded variables, 100 sparse equality rows: the bound elimination
    // would leave a dense 100 x 100 border (groups * t^2 + t^3 / 3 flops)
    // against a few thousand sparse LDL flops, so the sparse backend wins.
    let (n, m) = (200usize, 100usize);
    let (mut rows, mut cols, mut vals) = (Vec::new(), Vec::new(), Vec::new());
    for j in 0..n {
        for (r, v) in [(j % m, 1.0), ((j * 7 + 3) % m, -0.5)] {
            rows.push(r);
            cols.push(j);
            vals.push(v);
        }
        rows.push(m + j);
        cols.push(j);
        vals.push(-1.0);
    }
    let A = CscMatrix::new_from_triplets(m + n, n, rows, cols, vals);
    let mut b = vec![0.0; m + n];
    b[..m]
        .iter_mut()
        .enumerate()
        .for_each(|(i, v)| *v = 1.0 + (i % 3) as f64);
    let q: Vec<f64> = (0..n).map(|j| 1.0 + (j % 5) as f64).collect();
    let cones = vec![ZeroConeT(m), NonnegativeConeT(n)];
    let P = CscMatrix::<f64>::zeros((n, n));
    let settings = DefaultSettings {
        presolve_enable: false,
        ..DefaultSettings::default()
    };
    let mut solver = DefaultSolver::new(&P, &q, &A, &b, &cones, settings).unwrap();
    solver.solve();
    assert_eq!(solver.solution.status, SolverStatus::Solved);
    assert!(
        !solver.info.linsolver.name.starts_with("local_bounds"),
        "{}",
        solver.info.linsolver.name
    );
}
