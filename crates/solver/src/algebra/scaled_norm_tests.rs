use super::*;
use num_traits::{One, Zero};
use sdpx_arithmetic::{MpFloat, Scalar};

// Frozen pre-refactor implementation, including its nonfinite behavior.
fn frozen<T: FloatT>(x: impl Iterator<Item = T>) -> T {
    let (scale, sumsq) =
        x.filter(|b| *b != T::zero())
            .fold((T::zero(), T::one()), |(scale, sumsq), xi| {
                let absxi = xi.abs();
                if scale < absxi {
                    let r = scale / absxi;
                    (absxi, T::one() + sumsq * r * r)
                } else {
                    let r = absxi / scale;
                    (scale, sumsq + r * r)
                }
            });
    scale * sumsq.sqrt()
}

fn same<T: FloatT>(a: T, b: T) {
    if b.is_nan() {
        assert!(a.is_nan());
    } else {
        assert_eq!(a, b);
        assert_eq!(a.is_sign_negative(), b.is_sign_negative());
    }
}

fn check<T: FloatT>() {
    let z = T::zero();
    let o = T::one();
    let two = o + o;
    let cases = [
        vec![],
        vec![z, -z],
        vec![o, -o, two, -two],
        vec![T::max_value() / two, o, T::min_positive_value()],
        vec![T::min_positive_value() / two, -T::min_positive_value()],
        vec![T::nan()],
        vec![T::infinity()],
        vec![-T::infinity(), o],
        vec![T::infinity(), T::infinity()],
        vec![T::nan(), T::infinity()],
        vec![T::infinity(), T::nan()],
    ];
    for values in &cases {
        same(values.norm(), frozen(values.iter().copied()));
        same(ScaledNorm::from_iter(values.iter()).norm(), values.norm());
        let scales: Vec<_> = (0..values.len())
            .map(|i| if i % 2 == 0 { -two } else { z })
            .collect();
        same(
            values.norm_scaled(&scales),
            frozen(values.iter().zip(&scales).map(|(&x, &s)| x * s)),
        );
        same(
            values.norm_shifted(&scales, -two),
            frozen(values.iter().zip(&scales).map(|(&x, &s)| x + -two * s)),
        );
        let empty = ScaledNorm::<T>::from_iter(std::iter::empty::<T>());
        let state = ScaledNorm::from_iter(values.iter());
        same(empty.merge(state).norm(), state.norm());
        same(state.merge(empty).norm(), state.norm());
    }
    let inf = ScaledNorm::from_iter([T::infinity()].into_iter());
    assert!(inf.merge(inf).norm().is_nan());
    let nan = ScaledNorm::from_iter([T::nan()].into_iter());
    assert!(nan.merge(inf).norm().is_nan());
    assert!(inf.merge(nan).norm().is_nan());

    // Independently accumulated 2048-bit reference. Exact dyadic inputs include
    // products with opposing exponents and a shifted cancellation at one ulp.
    type H = MpFloat<32>;
    let htwo = H::one() + H::one();
    let exponent = if T::precision_bits() == 53 {
        500
    } else {
        100_000
    };
    let p = T::precision_bits() as i32;
    let products = vec![
        two.powi(exponent) * two.powi(-exponent),
        two.powi(exponent - 1) * two,
        -two.powi(exponent - 2),
        two.powi(-exponent),
        (o + two.powi(1 - p)) - o,
        z,
    ];
    let high = [
        H::one(),
        htwo.powi(exponent),
        -htwo.powi(exponent - 2),
        htwo.powi(-exponent),
        htwo.powi(1 - p),
        H::zero(),
    ];
    let reference = high.iter().fold(H::zero(), |s, &x| s + x * x).sqrt();
    // Decimal roundtrip has ample guard digits at every supported precision.
    let to_high = |v: T| -> H { format!("{:.650e}", v).parse().unwrap() };
    let tolerance = htwo.powi(1 - p) * htwo.powi(7);
    let bound = |v: T| {
        assert!(v.is_finite());
        let rel = ((to_high(v) - reference) / reference).abs();
        assert!(rel <= tolerance, "relative error {rel} exceeds {tolerance}");
    };
    bound(products.norm());
    for cut in 0..=products.len() {
        let a = ScaledNorm::from_iter(products[..cut].iter());
        let b = ScaledNorm::from_iter(products[cut..].iter());
        let empty = ScaledNorm::<T>::from_iter(std::iter::empty::<T>());
        bound(a.merge(empty).merge(b).norm());
        bound(b.merge(a).merge(empty).norm());
    }
    // Test underflow-scale and cancellation results on their own, so a large
    // coordinate cannot mask an incorrect small-coordinate result.
    for e in [-exponent, 0, exponent - 4] {
        let unit = two.powi(e);
        let values = [unit * (two + o), -unit * (two + two)];
        let href = htwo.powi(e) * (htwo + htwo + H::one());
        let merged = ScaledNorm::from_iter(values[..1].iter())
            .merge(ScaledNorm::from_iter(values[1..].iter()))
            .norm();
        for result in [values.norm(), merged] {
            assert!(((to_high(result) - href) / href).abs() <= tolerance);
        }
    }
    let cancelled = [o + two.powi(1 - p), -o - two.powi(1 - p)];
    let directions = [-o, o];
    let expected = htwo.powi(1 - p) * htwo.sqrt();
    let shifted = cancelled.norm_shifted(&directions, o);
    assert!(((to_high(shifted) - expected) / expected).abs() <= tolerance);
    let a = ScaledNorm::from_iter(products[..2].iter());
    let b = ScaledNorm::from_iter(products[2..4].iter());
    let c = ScaledNorm::from_iter(products[4..].iter());
    bound(a.merge(b).merge(c).norm());
    bound(a.merge(b.merge(c)).norm());
}

#[test]
fn scaled_norm_f64() {
    check::<f64>();
}
#[test]
fn scaled_norm_mpfr256() {
    check::<MpFloat<4>>();
}
#[test]
fn scaled_norm_mpfr512() {
    check::<MpFloat<8>>();
}
#[test]
fn scaled_norm_mpfr768() {
    check::<MpFloat<12>>();
}
