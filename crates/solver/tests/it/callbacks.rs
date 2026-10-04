#![allow(non_snake_case)]

#[cfg(test)]
mod callback_test {

    use sdpx_solver::solver::DefaultInfo;
    use sdpx_solver::{algebra::*, solver::*};

    // setup a custom termination function
    fn callback_r(info: &DefaultInfo<f64>) -> bool {
        if info.iterations < 3 {
            false //continue
        } else {
            true // stop
        }
    }

    #[test]
    fn test_callbacks() {
        let P = CscMatrix::identity(1);
        let c = [0.];
        let A = CscMatrix::identity(1);
        let b = [1.];
        let cones = [NonnegativeConeT(1)];

        let settings = DefaultSettings::default();
        let mut solver = DefaultSolver::new(&P, &c, &A, &b, &cones, settings).unwrap();

        solver.set_termination_callback(callback_r);
        solver.solve();
        assert_eq!(solver.solution.status, SolverStatus::CallbackTerminated);
        assert_eq!(solver.solution.iterations, 3);

        // turn it off and run again
        solver.unset_termination_callback();
        solver.solve();
        assert_eq!(solver.solution.status, SolverStatus::Solved);

        // and back on for a fresh solve
        solver.set_termination_callback(callback_r);
        solver.solve();
        assert_eq!(solver.solution.status, SolverStatus::CallbackTerminated);
        assert_eq!(solver.solution.iterations, 3);
    }
}
