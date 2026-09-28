//! Solve an SDPB `pmp2sdp --outputFormat=json` bootstrap problem at 768 bits.
use sdpx_arithmetic::Bits768;
use sdpx_solver::solver::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Directory written by `pmp2sdp --outputFormat=json`.
    let dir = std::env::args()
        .nth(1)
        .expect("usage: bootstrap <pmp2sdp-json-dir>");
    let sdp = read_sdpb_sampled::<Bits768>(dir)?;
    let mut problem = sdp.problem;

    let tol: Bits768 = "1e-42".parse().unwrap();
    problem.settings.tol_gap_abs = tol;
    problem.settings.tol_gap_rel = tol;
    problem.settings.tol_feas = tol;
    problem.settings.max_threads = 32;
    problem.settings.verbose = true;

    let mut solver = problem.into_solver()?;
    solver.solve();

    // Add the objective constant to recover SDPB's objective.
    let objective = solver.solution.obj_val + sdp.objective_constant;
    println!(
        "{:?} after {} iterations",
        solver.solution.status, solver.solution.iterations
    );
    println!("objective = {}", objective.to_decimal(Some(50)));
    Ok(())
}
