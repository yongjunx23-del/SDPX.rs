use sdpx_solver::{algebra::*, solver::*};

fn lp() -> (
    CscMatrix<f64>,
    Vec<f64>,
    CscMatrix<f64>,
    Vec<f64>,
    Vec<SupportedConeT<f64>>,
) {
    let p = CscMatrix::<f64>::zeros((3, 3));
    let mut a = CscMatrix::vcat(&CscMatrix::identity(3), &{
        let mut m = CscMatrix::<f64>::identity(3);
        m.negate();
        m
    })
    .unwrap();
    a.scale(2.);
    (
        p,
        vec![3., -2., 1.],
        a,
        vec![1.; 6],
        vec![NonnegativeConeT(3), NonnegativeConeT(3)],
    )
}

fn checkpointed(
    q: &[f64],
    file: &std::path::Path,
    every: u32,
    max_iter: u32,
) -> DefaultSolver<f64> {
    let (p, _, a, b, cones) = lp();
    let settings = DefaultSettings {
        max_iter,
        ..DefaultSettings::default()
    };
    let mut solver = DefaultSolver::new(&p, q, &a, &b, &cones, settings).unwrap();
    solver.set_checkpoint(file, every);
    solver.solve();
    solver
}

#[test]
fn restart_from_checkpoint_solves() {
    let (p, q, a, b, cones) = lp();
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("iterate.ckpt");
    let first = checkpointed(&q, &file, 1, 200);
    assert_eq!(first.solution.status, SolverStatus::Solved);
    assert!(file.exists());

    let mut second =
        DefaultSolver::new(&p, &q, &a, &b, &cones, DefaultSettings::default()).unwrap();
    second.set_restart(&file);
    second.check_restart().unwrap();
    second.solve();
    assert_eq!(second.solution.status, SolverStatus::Solved);
    assert!((second.solution.obj_val + 3.).abs() <= 1e-8);
    assert!(second.solution.iterations <= first.solution.iterations);
}

#[test]
fn hot_start_from_nearby_problem() {
    let (p, q, a, b, cones) = lp();
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("iterate.ckpt");
    // an interrupted solve of a nearby objective
    checkpointed(&[3.1, -2., 1.], &file, 1, 4);

    let mut solver =
        DefaultSolver::new(&p, &q, &a, &b, &cones, DefaultSettings::default()).unwrap();
    solver.set_restart(&file);
    solver.check_restart().unwrap();
    solver.solve();
    assert_eq!(solver.solution.status, SolverStatus::Solved);
    assert!((solver.solution.obj_val + 3.).abs() <= 1e-8);
}

#[test]
fn restart_rejects_different_structure() {
    let (p, q, a, b, _) = lp();
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("iterate.ckpt");
    checkpointed(&q, &file, 1, 200);

    // NN(3)+NN(3) merges to NN(6); a second-order cone is a different structure
    let other = vec![SecondOrderConeT(3), NonnegativeConeT(3)];
    let mut solver =
        DefaultSolver::new(&p, &q, &a, &b, &other, DefaultSettings::default()).unwrap();
    solver.set_restart(&file);
    assert!(solver.check_restart().is_err());
    std::fs::write(&file, b"junk").unwrap();
    assert!(solver.check_restart().is_err());
}
