#![allow(non_snake_case)]

use sdpx_solver::{algebra::*, io::ConfigurablePrintTarget, solver::*};

fn num<T: FloatT>(x: f64) -> T {
    T::from_f64(x).unwrap()
}

/// min x0 + x1 + x2 with x ≥ (1, 2, 3) written as `-x + s = -l`, s ≥ 0,
/// plus a 2×2 PSD block [[x0, 1], [1, x1]] ⪰ 0 (optimum 6).
fn solve<T: FloatT>(settings: DefaultSettings<T>) -> (SolverStatus, T, String) {
    let n = 3;
    let s2 = T::from_f64(std::f64::consts::SQRT_2).unwrap();
    let mut A = CscMatrix::<T>::zeros((6, n));
    // rows 0..3: -x_i; rows 3..6: svec([[x0,1],[1,x1]]) as -A x + s = b
    A.colptr = vec![0, 2, 4, 5];
    A.rowval = vec![0, 3, 1, 5, 2];
    A.nzval = vec![-T::one(), -T::one(), -T::one(), -T::one(), -T::one()];
    let b = vec![
        T::from_f64(-1.).unwrap(),
        T::from_f64(-2.).unwrap(),
        T::from_f64(-3.).unwrap(),
        T::zero(),
        s2,
        T::zero(),
    ];
    let c = vec![T::one(); n];
    let P = CscMatrix::<T>::zeros((n, n));
    let cones = [NonnegativeConeT(3), PSDTriangleConeT(2)];
    let mut solver = DefaultSolver::new(&P, &c, &A, &b, &cones, settings).unwrap();
    solver.print_to_buffer();
    solver.solve();
    let log = solver.get_print_buffer().unwrap();
    (solver.solution.status, solver.solution.obj_val, log)
}

fn close<T: FloatT>(a: T, b: f64) {
    let b = T::from_f64(b).unwrap();
    assert!(T::abs(a - b) <= num::<T>(1e-20), "{a} vs {b}");
}

#[test]
fn large_start_matches_unit_start() {
    type T = sdpx_arithmetic::Bits256;
    let defaults = DefaultSettings::<T>::default();
    assert_eq!(defaults.initial_tau, num::<T>(1.0));
    let (status, obj, _) = solve::<T>(defaults);
    assert_eq!(status, SolverStatus::Solved);
    close(obj, 6.0);
    let large = DefaultSettings::<T> {
        initial_tau: num::<T>(1e-20),
        ..DefaultSettings::default()
    };
    let (status, obj, _) = solve::<T>(large);
    assert_eq!(status, SolverStatus::Solved);
    close(obj, 6.0);
}

#[test]
fn overshooting_start_still_solves() {
    type T = sdpx_arithmetic::Bits256;
    // A start far beyond the solution scale costs iterations (the scale
    // comes down about two decades per step) but not the answer.
    let settings = DefaultSettings::<T> {
        initial_tau: num::<T>(1e-70),
        ..DefaultSettings::default()
    };
    let (status, obj, log) = solve::<T>(settings);
    assert_eq!(status, SolverStatus::Solved, "{log}");
    close(obj, 6.0);
}

fn rejected(settings: DefaultSettings<f64>) -> bool {
    let P = CscMatrix::<f64>::identity(1);
    let A = CscMatrix::<f64>::identity(1);
    DefaultSolver::new(&P, &[0.], &A, &[1.], &[NonnegativeConeT(1)], settings).is_err()
}

#[test]
fn start_scale_settings_are_validated() {
    assert!(!rejected(DefaultSettings::default()));
    for tau in [0.0, -1.0, 2.0, f64::NAN, f64::INFINITY] {
        let settings = DefaultSettings::<f64> {
            initial_tau: tau,
            ..DefaultSettings::default()
        };
        assert!(rejected(settings), "initial_tau {tau}");
    }
    for beta in [-0.1, 1.0] {
        let settings = DefaultSettings::<f64> {
            taukappa_proximity: beta,
            ..DefaultSettings::default()
        };
        assert!(rejected(settings), "taukappa_proximity {beta}");
    }
    for floor in [-0.1, 1.0] {
        let settings = DefaultSettings::<f64> {
            centering_floor: floor,
            ..DefaultSettings::default()
        };
        assert!(rejected(settings), "centering_floor {floor}");
    }
}

#[test]
fn taukappa_proximity_keeps_solving() {
    let settings = DefaultSettings::<f64> {
        taukappa_proximity: 0.01,
        ..DefaultSettings::default()
    };
    let (status, obj, _) = solve::<f64>(settings);
    assert_eq!(status, SolverStatus::Solved);
    assert!((obj - 6.0).abs() < 1e-6);
}
