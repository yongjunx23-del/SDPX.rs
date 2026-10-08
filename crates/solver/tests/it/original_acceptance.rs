#![allow(non_snake_case)]
//! `Solved` implies the original-coordinate audit at `tol_original`.

use sdpx_solver::{algebra::*, solver::*};

/// min x0 + x1 s.t. x0 + x1 = 1 (zero cone), x >= 0, and a 3x3 PSD block
/// [[1, x0, 0], [x0, 1, 0], [0, 0, 1]]: the block is sparse, so chordal
/// decomposition applies.
fn data() -> (
    CscMatrix<f64>,
    Vec<f64>,
    CscMatrix<f64>,
    Vec<f64>,
    Vec<SupportedConeT<f64>>,
) {
    let r2 = std::f64::consts::SQRT_2;
    // rows: 0 eq; 1,2 orthant; 3..9 svec of the PSD block (upper, by columns):
    // (0,0) (0,1) (1,1) (0,2) (1,2) (2,2)
    let A = CscMatrix::new_from_triplets(
        9,
        2,
        vec![0, 1, 4, 0, 2],
        vec![0, 0, 0, 1, 1],
        vec![1.0, -1.0, -r2, 1.0, -1.0],
    );
    let b = vec![1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0];
    let cones = vec![ZeroConeT(1), NonnegativeConeT(2), PSDTriangleConeT(3)];
    (CscMatrix::zeros((2, 2)), vec![1.0, 1.0], A, b, cones)
}

/// Audit residuals as `benchmark/research/native_oracle.jl` computes them,
/// with `s := b - Ax`, for the zero, orthant and PSD cones used here.
fn audit(A: &CscMatrix<f64>, b: &[f64], q: &[f64], x: &[f64], z: &[f64]) -> (f64, f64, f64) {
    // r = b - A x and rd = q + A'z, column by column.
    let (mut r, mut rd) = (b.to_vec(), q.to_vec());
    for j in 0..A.n {
        for p in A.colptr[j]..A.colptr[j + 1] {
            let (i, v) = (A.rowval[p], A.nzval[p]);
            r[i] -= v * x[j];
            rd[j] += v * z[i];
        }
    }
    let primal = r[0].abs().max(r[1..3].iter().fold(0f64, |a, &v| a.max(-v)));
    let dual = rd
        .iter()
        .fold(0f64, |a, &v| a.max(v.abs()))
        .max(z[1..3].iter().fold(0f64, |a, &v| a.max(-v)));
    let obj: f64 = q.iter().zip(x).map(|(a, b)| a * b).sum();
    let bz: f64 = b.iter().zip(z).map(|(a, b)| a * b).sum();
    let normb = b.iter().fold(0f64, |a, &v| a.max(v.abs()));
    let normq = q.iter().fold(0f64, |a, &v| a.max(v.abs()));
    (
        primal / (1.0 + normb),
        dual / (1.0 + normq),
        (obj + bz).abs() / (1.0 + obj.abs()),
    )
}

#[test]
fn solved_points_pass_the_original_audit() {
    let (P, q, A, b, cones) = data();
    for chordal in [false, true] {
        let settings = DefaultSettings {
            chordal_decomposition_enable: chordal,
            ..DefaultSettings::default()
        };
        assert_eq!(settings.tol_original, Some(1e-6));
        let mut solver = DefaultSolver::new(&P, &q, &A, &b, &cones, settings).unwrap();
        solver.solve();
        assert_eq!(solver.solution.status, SolverStatus::Solved);
        let (rp, rd, gap) = audit(&A, &b, &q, &solver.solution.x, &solver.solution.z);
        assert!(
            rp <= 1e-6 && rd <= 1e-6 && gap <= 1e-6,
            "{rp:e} {rd:e} {gap:e}"
        );
    }
}

#[test]
fn unattainable_original_tolerance_is_not_solved() {
    // The internal test still passes at its own tolerance; an original-
    // coordinate tolerance below rounding must not be reported as `Solved`.
    let (P, q, A, b, cones) = data();
    let settings = DefaultSettings {
        tol_original: Some(1e-30),
        ..DefaultSettings::default()
    };
    let mut solver = DefaultSolver::new(&P, &q, &A, &b, &cones, settings).unwrap();
    solver.solve();
    assert_ne!(solver.solution.status, SolverStatus::Solved);
    // Without the original-coordinate test the same solve is `Solved`.
    let settings = DefaultSettings {
        tol_original: None,
        ..DefaultSettings::default()
    };
    let mut solver = DefaultSolver::new(&P, &q, &A, &b, &cones, settings).unwrap();
    solver.solve();
    assert_eq!(solver.solution.status, SolverStatus::Solved);
}
