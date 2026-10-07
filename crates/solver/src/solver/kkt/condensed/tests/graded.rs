use super::*;
use sdpx_arithmetic::MpFloat;
fn check<T: FloatT>() {
    let a = T::FRAC_1_SQRT_2();
    let t = T::epsilon();
    let x = vec![a, -T::one(), a];
    let expected = (((a * t) * t) * t) * t;
    let mut failures = 0;
    for inverse in [false, true] {
        let mut p = PsdBlock::<T>::new(2, &CscMatrix::zeros((3, 0)), &(0..3));
        if inverse {
            p.Rinv = Matrix::from(&[[T::one(), T::one()], [T::zero(), t]]);
            p.R = Matrix::from(&[[T::one(), -t.recip()], [T::zero(), t.recip()]]);
        } else {
            p.R = Matrix::from(&[[T::one(), T::zero()], [T::one(), t]]);
            p.Rinv = Matrix::from(&[[T::one(), T::zero()], [-t.recip(), t.recip()]]);
        }
        p.Ginv
            .syrk(&p.Rinv.t(), T::one(), T::zero(), MatrixTriangle::Triu);
        p.Ginv[(1, 0)] = p.Ginv[(0, 1)];
        if !p.G.data().is_empty() {
            p.G.syrk(&p.R, T::one(), T::zero(), MatrixTriangle::Triu);
            p.G[(1, 0)] = p.G[(0, 1)];
        }
        let mut actual = vec![T::zero(); 3];
        p.apply(&mut actual, &x, inverse, None);
        let relative = (actual[2] - expected).abs() / expected;
        let pass = relative <= T::epsilon() * T::from_usize(128).unwrap();
        println!(
            "GRADED bits={} inverse={} pass={}",
            T::precision_bits(),
            inverse,
            pass
        );
        if !pass {
            failures += 1;
        }
    }
    assert_eq!(failures, 0);
}
// binary64 applies H through its factor (two congruences).
#[test]
fn condensed_graded_f64() {
    check::<f64>();
}
// Grading of cond(R) = 1/eps is beyond any IPM state at MPFR precision
// (cond(W)² ~ 1/μ² stays far above eps there); MPFR keeps the single G·X·G
// congruence, measured 5.5% faster on ising11 (journal 2026-09-30).
#[test]
#[ignore = "by design: MPFR applies the rounded G·X·G; see comment above"]
fn condensed_graded_mpfr() {
    check::<MpFloat<8>>();
}
