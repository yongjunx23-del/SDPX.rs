//! End to end at 128 bits: the operator relative entropy limit cone
//! K(2, 1) through both KKT forms. With a = 2, c = 1 fixed, N(2, 1) =
//! 2 ln 2 − 1 and G(b) = [Nb², b(1 + N); b(1 + N), 1 + N], so min tr T − b
//! is 1 + N − 1/(4N) at b = 1/(2N).
#![allow(non_snake_case)]
use num_traits::{FromPrimitive, One, Zero};
use sdpx_arithmetic::{Bits128, Scalar};
use sdpx_solver::{algebra::*, solver::*};

#[test]
fn limit_cone_solves_in_both_kkt_forms() {
    type T = Bits128;
    let f = |v: i32| T::from_i32(v).unwrap();
    // x = svec T (t₀₀, √2·t₀₁, t₁₁) | b | a | c
    let q = vec![f(1), f(0), f(1), f(-1), f(0), f(0)];
    let P = CscMatrix::<T>::zeros((6, 6));
    // a = 2, c = 1; then −x + s = 0 with s in the cone
    let mut rows = vec![vec![T::zero(); 6]; 8];
    rows[0][4] = T::one();
    rows[1][5] = T::one();
    for (i, row) in rows[2..].iter_mut().enumerate() {
        row[i] = -T::one();
    }
    let A = CscMatrix::from(&rows);
    let mut b = vec![T::zero(); 8];
    b[0] = f(2);
    b[1] = f(1);
    let cones = [ZeroConeT(2), OpRelEntropyLimitConeT(2, 1)];
    let n = f(2) * f(2).ln() - T::one();
    let (want, want_b) = (T::one() + n - (f(4) * n).recip(), (f(2) * n).recip());
    let tol: T = "1e-25".parse().unwrap();
    for form in ["augmented", "condensed"] {
        let settings = DefaultSettings::<T> {
            kkt_form: form.into(),
            verbose: false,
            tol_gap_abs: tol,
            tol_gap_rel: tol,
            tol_feas: tol,
            ..DefaultSettings::default()
        };
        let mut solver = DefaultSolver::new(&P, &q, &A, &b, &cones, settings).unwrap();
        solver.solve();
        assert_eq!(solver.solution.status, SolverStatus::Solved, "{form}");
        let close: T = "1e-20".parse().unwrap();
        let cost = solver.info.cost_primal;
        assert!((cost - want).abs() < close, "{form}: {cost} vs {want}");
        let x3 = solver.solution.x[3];
        assert!(
            (x3 - want_b).abs() < close.sqrt(),
            "{form}: b = {x3} vs {want_b}"
        );
    }
}
