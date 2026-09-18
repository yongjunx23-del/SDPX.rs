use super::*;
fn oracle<const N: usize>() {
    assert_eq!(MpFloat::<N>::precision_bits(), N * 64);
    type Op = unsafe extern "C" fn(
        *mut mpfr::mpfr_t,
        *const mpfr::mpfr_t,
        *const mpfr::mpfr_t,
        mpfr::rnd_t,
    ) -> i32;
    let a: MpFloat<N> = "1.2345678901234567890123456789012345678901234567890123456789"
        .parse()
        .unwrap();
    let b: MpFloat<N> = "-0.98765432109876543210987654321098765432109876543210987654321"
        .parse()
        .unwrap();
    // Heap-owned MPFR operands independently parsed from the same decimals.
    let mut x = Temp::new(N * 64);
    let mut y = Temp::new(N * 64);
    let mut z = Temp::new(N * 64);
    unsafe {
        mpfr::set_str(
            &mut x.0,
            c"1.2345678901234567890123456789012345678901234567890123456789".as_ptr(),
            10,
            ROUND,
        );
        mpfr::set_str(
            &mut y.0,
            c"-0.98765432109876543210987654321098765432109876543210987654321".as_ptr(),
            10,
            ROUND,
        );
        for (actual, op) in [
            (a + b, mpfr::add as Op),
            (a - b, mpfr::sub as Op),
            (a * b, mpfr::mul as Op),
            (a / b, mpfr::div as Op),
            (a % b, mpfr::fmod as Op),
            (a.atan2(b), mpfr::atan2 as Op),
        ] {
            op(&mut z.0, &x.0, &y.0, ROUND);
            assert_eq!(mpfr::cmp(&actual.descriptor(), &z.0), 0);
            assert_eq!(actual.to_string().parse::<MpFloat<N>>().unwrap(), actual);
        }
        mpfr::sqrt(&mut z.0, &x.0, ROUND);
        assert_eq!(mpfr::cmp(&a.sqrt().descriptor(), &z.0), 0);
        mpfr::cbrt(&mut z.0, &y.0, ROUND);
        assert_eq!(mpfr::cmp(&b.cbrt().descriptor(), &z.0), 0);
        mpfr::fma(&mut z.0, &x.0, &y.0, &x.0, ROUND);
        assert_eq!(mpfr::cmp(&a.mul_add(b, a).descriptor(), &z.0), 0);
    }
    let one = MpFloat::<N>::one();
    let eps = MpFloat::<N>::epsilon();
    assert_eq!((one + eps) - one, eps);
    assert_eq!(one + eps / MpFloat::from_u64(2).unwrap(), one); // ties to even
                                                                // A fused operation retains the product tail cancelled by the addend.
    assert_eq!((one + eps).mul_add(one - eps, -one), -(eps * eps));
    assert_eq!((one + eps) * (one - eps) - one, MpFloat::zero());
    assert!(eps < MpFloat::from_f64(f64::EPSILON).unwrap());
    let original = a;
    let mut copy = a;
    copy += one;
    assert_eq!(a, original);
    assert_ne!(a, copy);
    let mut v = vec![a; 2];
    for _ in 0..1000 {
        v.push(b);
    }
    v[0] += one;
    assert_eq!(v[1], a);
    let moved = std::thread::spawn(move || a + one).join().unwrap();
    assert_eq!(moved, copy);
}
#[test]
fn all_precisions_oracle_and_ownership() {
    oracle::<2>();
    oracle::<4>();
    oracle::<8>();
    oracle::<12>();
    oracle::<16>();
    oracle::<32>();
}
fn assignment_oracle<const N: usize>() {
    fn check<const N: usize>(actual: MpFloat<N>, expected: &mpfr::mpfr_t) {
        unsafe {
            assert_eq!(actual.is_nan(), mpfr::nan_p(expected) != 0);
            assert_eq!(actual.is_infinite(), mpfr::inf_p(expected) != 0);
            assert_eq!(actual.is_zero(), mpfr::zero_p(expected) != 0);
            if !actual.is_nan() {
                assert_eq!(actual.is_sign_negative(), mpfr::signbit(expected) != 0);
                assert_eq!(mpfr::cmp(&actual.descriptor(), expected), 0);
            }
        }
    }
    type Op = unsafe extern "C" fn(
        *mut mpfr::mpfr_t,
        *const mpfr::mpfr_t,
        *const mpfr::mpfr_t,
        mpfr::rnd_t,
    ) -> i32;
    let ops: [(fn(&mut MpFloat<N>, MpFloat<N>), Op); 5] = [
        (MpFloat::<N>::add_assign, mpfr::add),
        (MpFloat::<N>::sub_assign, mpfr::sub),
        (MpFloat::<N>::mul_assign, mpfr::mul),
        (MpFloat::<N>::div_assign, mpfr::div),
        (MpFloat::<N>::rem_assign, mpfr::fmod),
    ];
    let one = MpFloat::<N>::one();
    let values = [
        MpFloat::zero(),
        -MpFloat::zero(),
        one,
        -one,
        one + MpFloat::epsilon(),
        MpFloat::epsilon() / MpFloat::from_u64(2).unwrap(),
        "1.23456789012345678901234567890123456789".parse().unwrap(),
        "-0.987654321098765432109876543210987654321"
            .parse()
            .unwrap(),
        MpFloat::min_positive_value(),
        MpFloat::max_value(),
        MpFloat::infinity(),
        MpFloat::neg_infinity(),
        MpFloat::nan(),
    ];
    let mut x = Temp::new(N * 64);
    let mut y = Temp::new(N * 64);
    let mut z = Temp::new(N * 64);
    for a in values {
        for b in values {
            for (assign, op) in ops {
                let mut actual = a;
                unsafe {
                    // Independent heap storage: the oracle deliberately has
                    // distinct output/input limbs, unlike the assignment.
                    mpfr::set(&mut x.0, &a.descriptor(), ROUND);
                    mpfr::set(&mut y.0, &b.descriptor(), ROUND);
                    op(&mut z.0, &x.0, &y.0, ROUND);
                    assign(&mut actual, b);
                    check(actual, &z.0);
                    // The copied operands must retain their owned values,
                    // including signed zero and nonfinite classifications.
                    for (original, heap) in [(a, &x.0), (b, &y.0)] {
                        check(original, heap);
                    }
                }
            }
        }
    }
}
#[test]
fn all_precisions_in_place_assignment_oracle() {
    assignment_oracle::<2>();
    assignment_oracle::<4>();
    assignment_oracle::<8>();
    assignment_oracle::<12>();
    assignment_oracle::<16>();
    assignment_oracle::<32>();
}
#[test]
fn special_values() {
    type T = Bits256;
    let z: T = "-0".parse().unwrap();
    assert!(z.is_zero() && z.is_sign_negative());
    assert_eq!(z.to_string(), "-0");
    assert!(z.recip().is_infinite() && z.recip().is_sign_negative());
    assert!(T::nan() != T::nan());
    assert_eq!(T::nan().partial_cmp(&T::one()), None);
    assert!(T::infinity() > T::max_value());
    assert!(T::max_value().is_finite());
    assert!(T::min_positive_value() > T::zero());
    assert_eq!(T::one().min(T::nan()), T::one());
    assert!(T::zero().min(z).is_sign_negative());
    assert!(!T::zero().max(z).is_sign_negative());
    assert_eq!("bad".parse::<T>().unwrap_err().precision_bits, 256);
    assert_eq!(T::from_u64(u64::MAX).unwrap().to_u64(), Some(u64::MAX));
    assert_eq!(T::infinity().to_i64(), None);
    let pz = T::zero();
    for x in [pz, T::one()] {
        let above = pz.atan2(x);
        let below = z.atan2(x);
        assert!(above.is_zero() && !above.is_sign_negative());
        assert!(below.is_zero() && below.is_sign_negative());
    }
    for x in [z, -T::one()] {
        assert_eq!(pz.atan2(x), T::PI());
        assert_eq!(z.atan2(x), -T::PI());
    }
    assert_eq!(T::one().atan2(pz), T::FRAC_PI_2());
    assert_eq!((-T::one()).atan2(z), -T::FRAC_PI_2());
}
#[test]
fn constants_and_decimal_precision() {
    type T = Bits2048;
    let pi = T::PI();
    assert_eq!(pi.to_string().parse::<T>().unwrap(), pi);
    assert!(pi.to_string().len() > 610);
    assert_eq!(T::SQRT_2(), T::from_u64(2).unwrap().sqrt());
    assert_eq!(T::LN_2(), T::from_u64(2).unwrap().ln());
    assert!(T::FRAC_2_SQRT_PI() > T::one());
    let a: T = "1.000000000000000000000000000000000000000000000000000000001"
        .parse()
        .unwrap();
    assert_ne!(a, T::one());
    assert_eq!(a.to_f64(), Some(1.0));
}
fn ordered_dot_fma<const N: usize>() {
    type F<const N: usize> = MpFloat<N>;
    let one = F::<N>::one();
    let zero = F::<N>::zero();
    let eps = F::<N>::epsilon();
    let big = (one + one).powi((N * 64) as i32);
    let a = [big, one, -big];
    let b = [one; 3];
    // A once-rounded exact dot gives 1; the required ordered FMAs give 0.
    assert_eq!(F::dot_fma(a.iter().zip(&b)), zero);
    assert_eq!(F::dot_fma(a[..0].iter().zip(&b[..0])).kind, mpfr::ZERO_KIND);
    let a = [-one, one + eps];
    let b = [one, one - eps];
    // Separately rounded multiplication would lose this cancellation term.
    assert_eq!(F::dot_fma(a.iter().zip(&b)), -(eps * eps));
    let mut a: Vec<_> = (0..37).map(|i| F::<N>::from_i32(i - 18).unwrap() / F::from_i32(7).unwrap()).collect();
    let b: Vec<_> = (0..37).map(|i| F::<N>::from_i32(i % 11 - 5).unwrap() / F::from_i32(13).unwrap()).collect();
    let before = (a.clone(), b.clone());
    let expected = a.iter().zip(&b).fold(zero, |v, (&x, &y)| x.mul_add(y, v));
    let actual = F::dot_fma(a.iter().zip(&b));
    assert_eq!(actual, expected);
    assert_eq!((a.clone(), b.clone()), before);
    a.fill(zero);
    assert_eq!(actual, expected);
    for (a, b) in [
        (vec![-zero, zero], vec![one, one]),
        (vec![F::infinity(), one], vec![one, one]),
        (vec![F::infinity(), F::neg_infinity()], vec![one, one]),
        (vec![F::nan(), one], vec![one, one]),
    ] {
        let expected = a.iter().zip(&b).fold(zero, |v, (&x, &y)| x.mul_add(y, v));
        let actual = F::dot_fma(a.iter().zip(&b));
        if expected.is_nan() {
            assert!(actual.is_nan());
        } else {
            assert_eq!(actual, expected);
            assert_eq!(actual.kind, expected.kind);
        }
    }
}
#[test]
fn ordered_dot_fma_all_precisions() {
    ordered_dot_fma::<2>();
    ordered_dot_fma::<4>();
    ordered_dot_fma::<8>();
    ordered_dot_fma::<12>();
    ordered_dot_fma::<16>();
    ordered_dot_fma::<32>();
}
