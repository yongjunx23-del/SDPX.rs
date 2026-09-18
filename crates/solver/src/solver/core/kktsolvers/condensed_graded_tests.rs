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
#[test]
fn condensed_graded_f64() {
    check::<f64>();
}
#[test]
fn condensed_graded_128() {
    check::<MpFloat<2>>();
}
#[test]
fn condensed_graded_256() {
    check::<MpFloat<4>>();
}
#[test]
fn condensed_graded_512() {
    check::<MpFloat<8>>();
}
#[test]
fn condensed_graded_768() {
    check::<MpFloat<12>>();
}
#[test]
fn condensed_graded_1024() {
    check::<MpFloat<16>>();
}
#[test]
fn condensed_graded_2048() {
    check::<MpFloat<32>>();
}
