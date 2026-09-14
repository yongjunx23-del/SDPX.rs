//! End-to-end returned-point check exercising affine and combined directions.
use sdpx_solver::{algebra::*, solver::*};

#[test]
fn quadratic_equality_direction_reconstruction() {
    // min x'Px/2 + q'x, x1+x2=1, x>=0; optimum (1/4,3/4).
    let p = CscMatrix::from(&[[2., 1.], [0., 2.]]);
    let q = [0., -0.5];
    let a = CscMatrix::from(&[[1., 1.], [-1., 0.], [0., -1.]]);
    let b = [1., 0., 0.];
    let cones = [ZeroConeT(1), NonnegativeConeT(2)];
    let settings = DefaultSettings {
        verbose: false,
        equilibrate_enable: false,
        ..DefaultSettings::default()
    };
    let mut solver = DefaultSolver::new(&p, &q, &a, &b, &cones, settings).unwrap();
    solver.solve();
    let sol = &solver.solution;
    assert_eq!(sol.status, SolverStatus::Solved);
    assert!((sol.x[0] - 0.25_f64).abs() < 1e-7);
    assert!((sol.x[1] - 0.75_f64).abs() < 1e-7);
    assert!((sol.x[0] + sol.x[1] - 1.).abs() < 1e-8);
    assert!(sol.s[0].abs() < 1e-12);
    for j in 0..2 {
        let dual = 2. * sol.x[j] + sol.x[1 - j] + q[j] + sol.z[0] - sol.z[j + 1];
        assert!(dual.abs() < 1e-7);
        assert!((sol.s[j + 1] - sol.x[j]).abs() < 1e-8);
        assert!(sol.z[j + 1] >= -1e-9);
        assert!((sol.z[j + 1] * sol.s[j + 1]).abs() < 1e-7);
    }
}

#[test]
fn dependent_equalities_retain_primal_infeasibility_under_scaling() {
    // The first four equations are inconsistent. An optional duplicated last
    // equation adds another null(A') mode without changing feasibility.
    let base = [
        [0., 1., 1.],
        [0., 1., -1.],
        [1., 2., -1.],
        [2., -1., 3.],
        [4., -2., 6.],
    ];
    for m in [4, 5] {
        for global_scale in [0.01_f64, 1., 100.] {
            let row_scale = [0.125, 2., 8., 0.5, 4.];
            let mut rows = vec![[0.; 3]; m];
            let mut b = vec![0.; m];
            for i in 0..m {
                let scale = global_scale * row_scale[i];
                for j in 0..3 {
                    rows[i][j] = scale * base[i][j];
                }
                b[i] = scale * if i == 4 { 2. } else { 1. };
            }
            let mut a = CscMatrix::zeros((m, 3));
            for j in 0..3 {
                for i in 0..m {
                    if rows[i][j] != 0. {
                        a.rowval.push(i);
                        a.nzval.push(rows[i][j]);
                    }
                }
                a.colptr[j + 1] = a.nzval.len();
            }
            let settings = DefaultSettings {
                verbose: false,
                equilibrate_enable: false,
                presolve_enable: false,
                ..DefaultSettings::default()
            };
            let mut solver = DefaultSolver::new(
                &CscMatrix::identity(3),
                &[0.; 3],
                &a,
                &b,
                &[ZeroConeT(m)],
                settings,
            )
            .unwrap();
            solver.solve();
            let z = &solver.solution.z;
            assert_eq!(
                solver.solution.status,
                SolverStatus::PrimalInfeasible,
                "m={m}, scale={global_scale}"
            );
            assert!(z.iter().all(|v| v.is_finite()));
            let btz: f64 = b.iter().zip(z).map(|(b, z)| b * z).sum();
            assert!(btz < 0.);
            // Check the returned ray externally in the original coordinates.
            for j in 0..3 {
                let residual: f64 = (0..m).map(|i| rows[i][j] * z[i]).sum();
                assert!(
                    residual.abs() / (-btz) <= 1e-7,
                    "m={m}, scale={global_scale}, column={j}, residual={residual}, btz={btz}"
                );
            }
        }
    }
}
