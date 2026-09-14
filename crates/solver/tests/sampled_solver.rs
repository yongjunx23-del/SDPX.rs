#![cfg(feature = "sdp")]
use sdpx_solver::algebra::{CscMatrix, FloatT};
use sdpx_solver::solver::*;

fn number<T: FloatT>(x: f64) -> T {
    T::from_f64(x).unwrap()
}

fn compare<T: FloatT>() {
    let q = vec![T::one(); 2];
    let b = vec![-T::one(), T::zero(), -T::one()];
    let p = CscMatrix::zeros((2, 2));
    let linear = CscMatrix::zeros((3, 2));
    let block = SampledBlock {
        row_start: 0,
        column_start: 0,
        dim: 1,
        basis_rows: 2,
        basis_cols: 2,
        basis: vec![T::one(), number(0.25), number(0.25), T::one()],
        weights: vec![-T::one(); 2],
    };
    let operator = SampledOperator::new(linear.clone(), vec![block.clone()]).unwrap();
    let a = operator.materialize();
    let cones = vec![PSDTriangleConeT(2)];
    let mut settings = DefaultSettings::<T>::default();
    settings.verbose = false;
    settings.direct_solve_method = "qdldl".into();
    settings.kkt_form = "condensed".into();
    let tol = number::<T>(if T::precision_bits() > 53 {
        1e-28
    } else {
        1e-9
    });
    settings.tol_feas = tol;
    settings.tol_gap_abs = tol;
    settings.tol_gap_rel = tol;
    let mut generic = DefaultSolver::new(&p, &q, &a, &b, &cones, settings.clone()).unwrap();
    let mut sampled =
        DefaultSolver::new_sampled(&p, &q, &linear, &b, &cones, vec![block], settings).unwrap();
    assert_eq!(sampled.info.linsolver.name, "condensed_sampled_qdldl");
    // The factor route retains Ruiz and chooses a genuinely scalar PSD map.
    let e = &sampled.data.equilibration.e;
    assert!(e.iter().all(|v| *v == e[0]));
    generic.solve();
    sampled.solve();
    assert_eq!(generic.solution.status, SolverStatus::Solved);
    assert_eq!(sampled.solution.status, SolverStatus::Solved);
    // Symmetry makes both primal variables 1/(1-1/4)^2 = 16/9.
    let expected = number::<T>(16.) / number(9.);
    let bound = number::<T>(128.) * tol;
    for x in &sampled.solution.x {
        assert!((*x - expected).abs() <= bound);
    }
    assert!((sampled.solution.obj_val - generic.solution.obj_val).abs() <= bound);
    // Evaluate original factors outside the solver, independently of its CSC.
    let mut residual = b.iter().map(|v| -*v).collect::<Vec<_>>();
    operator.apply(
        &mut residual,
        &sampled.solution.x,
        T::one(),
        T::one(),
        &mut SampledWorkspace::new(&operator),
    );
    for (r, s) in residual.iter().zip(&sampled.solution.s) {
        assert!((*r + *s).abs() <= bound);
    }
    // Updating affine terms retains factors. Arbitrary A updates cannot leave
    // a stale operator behind; preparing a fresh factor input is explicit.
    assert!(matches!(
        sampled.update_A(&a),
        Err(DataUpdateError::SampledMatrixUpdate)
    ));
    let twice_b: Vec<_> = b.iter().map(|v| number::<T>(2.) * *v).collect();
    sampled.update_b(&twice_b).unwrap();
    sampled.update_q(&q).unwrap();
    sampled.solve();
    assert_eq!(sampled.solution.status, SolverStatus::Solved);
    for x in &sampled.solution.x {
        assert!((*x - number::<T>(2.) * expected).abs() <= number::<T>(4.) * bound);
    }
    // Empty matrix updates preserve the standard batched q/b update API.
    sampled.update_data(&[], &q, &[], &b).unwrap();
    let saved_q = sampled.data.q.clone();
    let saved_b = sampled.data.b.clone();
    let replacement_q = vec![number::<T>(3.); 2];
    assert!(matches!(
        sampled.update_data(&[], &replacement_q, &a, &twice_b),
        Err(DataUpdateError::SampledMatrixUpdate)
    ));
    assert_eq!(sampled.data.q, saved_q);
    assert_eq!(sampled.data.b, saved_b);
}

#[test]
fn sampled_solver_float64() {
    compare::<f64>();
}
#[test]
fn sampled_solver_mpfr256() {
    compare::<sdpx_arithmetic::Bits256>();
}
#[test]
fn sampled_solver_mpfr512() {
    compare::<sdpx_arithmetic::Bits512>();
}

#[test]
fn sampled_singleton_uses_upstream_cone_collapse() {
    let linear = CscMatrix::<f64>::zeros((1, 1));
    let block = SampledBlock {
        row_start: 0,
        column_start: 0,
        dim: 1,
        basis_rows: 1,
        basis_cols: 1,
        basis: vec![1.],
        weights: vec![-1.],
    };
    let mut settings = DefaultSettings::default();
    settings.verbose = false;
    let mut solver = DefaultSolver::new_sampled(
        &CscMatrix::zeros((1, 1)),
        &[1.],
        &linear,
        &[-1.],
        &[PSDTriangleConeT(1)],
        vec![block],
        settings,
    )
    .unwrap();
    assert!(!solver.info.linsolver.name.contains("sampled"));
    assert!(matches!(
        solver.update_A(&CscMatrix::identity(1)),
        Err(DataUpdateError::SampledMatrixUpdate)
    ));
    solver.solve();
    assert_eq!(solver.solution.status, SolverStatus::Solved);
    assert!((solver.solution.x[0] - 1.).abs() < 1e-7);
}

#[test]
fn sampled_blocks_must_cover_whole_psd_cones() {
    let block = SampledBlock {
        row_start: 0,
        column_start: 0,
        dim: 1,
        basis_rows: 2,
        basis_cols: 1,
        basis: vec![1., 1.],
        weights: vec![-1.],
    };
    let result = DefaultSolver::new_sampled(
        &CscMatrix::zeros((1, 1)),
        &[1.],
        &CscMatrix::zeros((3, 1)),
        &[0.; 3],
        &[NonnegativeConeT(3)],
        vec![block],
        DefaultSettings::default(),
    );
    assert!(result.is_err());
}
