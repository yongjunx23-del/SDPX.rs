//! End to end: PMP JSON -> sampled SDP directory -> solve. Maximize y subject
//! to 1 - y + x^2 >= 0 for x >= 0, whose optimum is y = 1.
use sdpx_arithmetic::Bits128;
use sdpx_pmp::PolynomialMatrixProgram;
use sdpx_solver::solver::*;

#[test]
fn readme_pmp_solves_to_one() {
    let pmp: PolynomialMatrixProgram = serde_json::from_str(
        r#"{"objective":["0","1"],"PositiveMatrixWithPrefactorArray":[{
            "prefactor":{"constant":"1","base":"0.5","poles":[]},
            "polynomials":[[[["1","0","1"],["-1"]]]]}]}"#,
    )
    .unwrap();
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("sdp");
    pmp.write_sdp::<Bits128>(&out).unwrap();
    let sdp = read_sdpb_sampled::<Bits128>(&out).unwrap();
    let mut problem = sdp.problem;
    let tol: Bits128 = "1e-20".parse().unwrap();
    problem.settings.tol_gap_abs = tol;
    problem.settings.tol_gap_rel = tol;
    problem.settings.tol_feas = tol;
    problem.settings.verbose = false;
    let mut solver = problem.into_solver().unwrap();
    solver.solve();
    assert_eq!(solver.solution.status, SolverStatus::Solved);
    let y = -solver.solution.z[0];
    let one: Bits128 = "1".parse().unwrap();
    let tol: Bits128 = "1e-15".parse().unwrap();
    let error = y - one;
    assert!(error < tol && -error < tol, "y = {y:?}");
}
