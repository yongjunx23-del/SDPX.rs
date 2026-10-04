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
            (a.hypot(b), mpfr::hypot as Op),
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
    unsafe {
        assert_eq!(mpfr::cmp_ui(&one.descriptor(), 1), 0);
    }
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

#[test]
fn comparisons_match_mpfr_all_precisions() {
    fn check<const N: usize>() {
        type F<const N: usize> = MpFloat<N>;
        let mut values = vec![
            F::<N>::zero(),
            -F::zero(),
            F::nan(),
            F::infinity(),
            F::neg_infinity(),
            F::one(),
            -F::one(),
            F::min_positive_value(),
            F::max_value(),
        ];
        // Include adjacent significands and values produced by arithmetic;
        // special results retain arbitrary old limb contents in MPFR.
        values.extend([
            F::<N>::one() + F::epsilon(),
            F::one() - F::epsilon(),
            F::one() - F::one(),
            F::one() / F::zero(),
        ]);
        let mut state = 0xabcddcba12344321_u64;
        for e in [-4096, -1, 0, 1, 4096] {
            for _ in 0..8 {
                let limbs = std::array::from_fn(|i| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    if i == N - 1 {
                        state | (1 << 63)
                    } else {
                        state
                    }
                });
                let value = F::<N>::exact_decode(mpfr::REGULAR_KIND, e, limbs);
                values.extend([value, -value]);
            }
        }
        for a in &values {
            for b in &values {
                let expected = if a.is_nan() || b.is_nan() {
                    None
                } else {
                    Some(unsafe { mpfr::cmp(&a.descriptor(), &b.descriptor()) }.cmp(&0))
                };
                assert_eq!(a.partial_cmp(b), expected);
                assert_eq!(a == b, expected == Some(Ordering::Equal));
            }
        }
    }
    check::<2>();
    check::<4>();
    check::<8>();
    check::<12>();
    check::<16>();
    check::<32>();
}

#[test]
fn hypot_extreme_exponents_and_special_values() {
    fn check<const N: usize>() {
        type F<const N: usize> = MpFloat<N>;
        for exponent in [-4096, 0, 4096] {
            let scale = F::<N>::from_u64(2).unwrap().powi(exponent);
            let x = F::<N>::from_u64(3).unwrap() * scale;
            let y = F::<N>::from_u64(4).unwrap() * scale;
            assert_eq!(x.hypot(-y), F::<N>::from_u64(5).unwrap() * scale);
            assert_eq!(x.hypot(F::zero()), x);
        }
        assert_eq!(F::<N>::infinity().hypot(F::nan()), F::infinity());
        assert!(F::<N>::one().hypot(F::nan()).is_nan());
        assert!(!(-F::<N>::zero()).hypot(F::zero()).is_sign_negative());
    }
    check::<2>();
    check::<4>();
    check::<8>();
    check::<12>();
    check::<16>();
    check::<32>();
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

#[test]
fn constant_cache_precision_ownership_and_threads() {
    fn check<const N: usize>() {
        for constant in [
            Constant::Pi,
            Constant::Sqrt2,
            Constant::Frac1Sqrt2,
            Constant::Ln2,
        ] {
            let reference = MpFloat::<N>::constant_uncached(constant);
            let mut value = MpFloat::<N>::constant(constant);
            assert_eq!(value, reference);
            value += MpFloat::one();
            assert_ne!(value, reference);
            assert_eq!(MpFloat::<N>::constant(constant), reference);
            // Returned limbs have independent ownership, including across threads.
            assert_eq!(
                std::thread::spawn(move || MpFloat::<N>::constant(constant))
                    .join()
                    .unwrap(),
                reference
            );
        }
    }
    check::<8>();
    check::<2>();
    check::<12>();
    check::<4>();
    check::<16>();
    check::<32>();
    check::<2>();
}
fn ordered_dot_fma<const N: usize>() {
    type F<const N: usize> = MpFloat<N>;
    let one = F::<N>::one();
    let zero = F::<N>::zero();
    let eps = F::<N>::epsilon();
    let big = (one + one).powi((N * 64) as i32);
    let a = [big, one, -big];
    let b = [one; 3];
    // The ordered FMA chain gives 0; the exact, once-rounded dot gives 1.
    assert_eq!(F::dot_fma_chain(a.iter().zip(&b)), zero);
    assert_eq!(F::dot_fma(a.iter().zip(&b)), one);
    assert_eq!(F::dot_fma(a[..0].iter().zip(&b[..0])).kind, mpfr::ZERO_KIND);
    let a = [-one, one + eps];
    let b = [one, one - eps];
    // Separately rounded multiplication would lose this cancellation term.
    assert_eq!(F::dot_fma_chain(a.iter().zip(&b)), -(eps * eps));
    assert_eq!(F::dot_fma(a.iter().zip(&b)), -(eps * eps));
    let mut a: Vec<_> = (0..37)
        .map(|i| F::<N>::from_i32(i - 18).unwrap() / F::from_i32(7).unwrap())
        .collect();
    let b: Vec<_> = (0..37)
        .map(|i| F::<N>::from_i32(i % 11 - 5).unwrap() / F::from_i32(13).unwrap())
        .collect();
    let before = (a.clone(), b.clone());
    let expected = a.iter().zip(&b).fold(zero, |v, (&x, &y)| x.mul_add(y, v));
    let actual = F::dot_fma_chain(a.iter().zip(&b));
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

// The inline product must equal mpfr_mul bit for bit (nearest-even has one
// answer): random full mantissas, exponent spread, signs, ties and carries.
fn inline_mul_matches_mpfr<const N: usize>() {
    let mut s = 0x2545f4914f6cdd1du64 ^ N as u64;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let random = |next: &mut dyn FnMut() -> u64, bits: Option<u32>| {
        let mut limbs = [0u64; N];
        for l in limbs.iter_mut() {
            *l = next();
        }
        if let Some(bits) = bits {
            // Few leading bits: products then land on exact ties often.
            limbs = [0; N];
            limbs[N - 1] = (next() | 1 << 63) & !(u64::MAX >> bits);
        }
        limbs[N - 1] |= 1 << 63;
        let e = (next() % 400) as i64 - 200;
        let kind = if next() & 1 == 0 {
            mpfr::REGULAR_KIND
        } else {
            -mpfr::REGULAR_KIND
        };
        MpFloat::<N> {
            limbs,
            kind,
            exponent: e as mpfr::exp_t,
        }
    };
    let top = MpFloat::<N> {
        limbs: [u64::MAX; N],
        kind: mpfr::REGULAR_KIND,
        exponent: 0,
    };
    let mut cases = vec![(top, top)];
    for i in 0..20_000 {
        let bits = match i % 4 {
            0 => None,
            1 => Some(2),
            2 => Some(5),
            _ => Some(17),
        };
        let a = random(&mut next, None);
        let b = random(&mut next, bits);
        cases.push((a, b));
        cases.push((b, a));
    }
    for (a, b) in cases {
        let fast = a.mul_regular(&b).expect("regular operands in range");
        let reference = a.binary(b, mpfr::mul);
        assert_eq!(
            (fast.kind, fast.exponent, fast.limbs),
            (reference.kind, reference.exponent, reference.limbs),
            "N={N} {a:?} * {b:?}"
        );
    }
    // Specials and wide exponents defer to MPFR.
    let one = MpFloat::<N>::one();
    assert!(one.mul_regular(&MpFloat::<N>::zero()).is_none());
    let huge = MpFloat::<N> {
        exponent: (1 << 29) as mpfr::exp_t,
        ..one
    };
    assert!(huge.mul_regular(&huge).is_none());
    assert_eq!((huge * huge).kind, (huge.binary(huge, mpfr::mul)).kind);
}

#[test]
fn inline_multiply_matches_mpfr_mul() {
    inline_mul_matches_mpfr::<1>();
    inline_mul_matches_mpfr::<2>();
    inline_mul_matches_mpfr::<3>();
    inline_mul_matches_mpfr::<4>();
    inline_mul_matches_mpfr::<8>();
    inline_mul_matches_mpfr::<12>();
    inline_mul_matches_mpfr::<16>();
}

// Inline add/sub must equal mpfr_add/mpfr_sub bit for bit.
fn inline_add_matches_mpfr<const N: usize>() {
    let mut s = 0x9e3779b97f4a7c15u64 ^ (N as u64 * 977);
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let p = (N * 64) as i64;
    let make = |next: &mut dyn FnMut() -> u64, e: i64, style: u64| {
        let mut limbs = [0u64; N];
        match style % 5 {
            0 | 1 => limbs.iter_mut().for_each(|l| *l = next()),
            2 => limbs[N - 1] = next() & !(u64::MAX >> 3), // few bits: ties
            3 => {}                                        // power of two
            _ => limbs = [u64::MAX; N],                    // all ones: carry-out
        }
        limbs[N - 1] |= 1 << 63;
        let kind = if next() & 1 == 0 {
            mpfr::REGULAR_KIND
        } else {
            -mpfr::REGULAR_KIND
        };
        MpFloat::<N> {
            limbs,
            kind,
            exponent: e as mpfr::exp_t,
        }
    };
    let gaps = [
        0,
        1,
        2,
        3,
        63,
        64,
        65,
        p - 2,
        p - 1,
        p,
        p + 1,
        p + 2,
        p + 3,
        2 * p,
        1000,
    ];
    let mut cases = Vec::new();
    for _ in 0..3000 {
        for &g in &gaps {
            let e = (next() % 200) as i64 - 100;
            let style = next();
            let a = make(&mut next, e, style);
            let style = next();
            let b = make(&mut next, e - g, style);
            cases.push((a, b));
            cases.push((b, a));
        }
        // Near-total cancellation: b = -a perturbed in the last limb.
        let a = make(&mut next, 7, 0);
        let mut b = MpFloat::<N> { kind: -a.kind, ..a };
        b.limbs[0] ^= next() & 0xff;
        b.limbs[N - 1] |= 1 << 63;
        cases.push((a, b));
        cases.push((a, MpFloat::<N> { kind: -a.kind, ..a }));
    }
    for (a, b) in cases {
        for negate in [false, true] {
            let fast = a.add_regular(&b, negate).expect("in range");
            let reference = a.binary(b, if negate { mpfr::sub } else { mpfr::add });
            let norm = |v: MpFloat<N>| {
                if v.kind.abs() == mpfr::ZERO_KIND {
                    (0, 0, [0; N])
                } else {
                    (v.kind, v.exponent as i64, v.limbs)
                }
            };
            assert_eq!(
                norm(fast),
                norm(reference),
                "N={N} {a:?} {} {b:?}",
                if negate { '-' } else { '+' }
            );
            if fast.kind.abs() == mpfr::ZERO_KIND {
                assert_eq!(fast.kind, reference.kind, "zero sign");
            }
        }
    }
}

#[test]
fn inline_add_sub_matches_mpfr() {
    inline_add_matches_mpfr::<1>();
    inline_add_matches_mpfr::<2>();
    inline_add_matches_mpfr::<3>();
    inline_add_matches_mpfr::<4>();
    inline_add_matches_mpfr::<8>();
}

// Timing only (cargo test --release -- --ignored --nocapture): inline vs
// MPFR per width, to keep each fast path only where it wins.
fn time_ops<const N: usize>() {
    let mut s = 0x1234_5678_9abc_def1u64;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let vals: Vec<MpFloat<N>> = (0..4096)
        .map(|_| {
            let mut limbs = [0u64; N];
            limbs.iter_mut().for_each(|l| *l = next());
            limbs[N - 1] |= 1 << 63;
            let kind = if next() & 1 == 0 {
                mpfr::REGULAR_KIND
            } else {
                -mpfr::REGULAR_KIND
            };
            MpFloat::<N> {
                limbs,
                kind,
                exponent: ((next() % 9) as i64 - 4) as mpfr::exp_t,
            }
        })
        .collect();
    let reps = 200;
    let time = |f: &dyn Fn(&MpFloat<N>, &MpFloat<N>) -> MpFloat<N>| {
        let t = std::time::Instant::now();
        let mut acc = 0i64;
        for _ in 0..reps {
            for w in vals.windows(2) {
                let r = std::hint::black_box(f(
                    std::hint::black_box(&w[0]),
                    std::hint::black_box(&w[1]),
                ));
                acc = acc.wrapping_add(r.exponent as i64 ^ r.limbs[0] as i64);
            }
        }
        (
            t.elapsed().as_nanos() as f64 / (reps * (vals.len() - 1)) as f64,
            acc,
        )
    };
    let (mi, _) = time(&|a, b| a.mul_regular(b).unwrap_or(*a));
    let (mm, _) = time(&|a, b| a.binary(*b, mpfr::mul));
    let (ai, _) = time(&|a, b| a.add_regular(b, false).unwrap_or(*a));
    let (am, _) = time(&|a, b| a.binary(*b, mpfr::add));
    println!("N={N:2}: mul inline {mi:5.1} mpfr {mm:5.1} | add inline {ai:5.1} mpfr {am:5.1} ns");
}

#[test]
#[ignore]
fn time_inline_vs_mpfr() {
    time_ops::<2>();
    time_ops::<4>();
    time_ops::<8>();
    time_ops::<12>();
    time_ops::<16>();
    time_ops::<19>();
    time_ops::<32>();
}

fn time_fmma<const N: usize>() {
    let mut s = 0xabcdef12345u64 ^ N as u64;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let vals: Vec<MpFloat<N>> = (0..4096)
        .map(|_| {
            let mut limbs = [0u64; N];
            limbs.iter_mut().for_each(|l| *l = next());
            limbs[N - 1] |= 1 << 63;
            let kind = if next() & 1 == 0 {
                mpfr::REGULAR_KIND
            } else {
                -mpfr::REGULAR_KIND
            };
            MpFloat::<N> {
                limbs,
                kind,
                exponent: ((next() % 9) as i64 - 4) as mpfr::exp_t,
            }
        })
        .collect();
    let reps = 100;
    let t = std::time::Instant::now();
    let mut acc = 0i64;
    for _ in 0..reps {
        for w in vals.windows(4) {
            let r = std::hint::black_box(MpFloat::<N>::dot_fma2(&w[0], &w[1], &w[2], &w[3]));
            acc ^= r.limbs[0] as i64;
        }
    }
    let a = t.elapsed().as_nanos() as f64 / (reps * 4093) as f64;
    let t = std::time::Instant::now();
    for _ in 0..reps {
        for w in vals.windows(4) {
            let r = std::hint::black_box(MpFloat::<N>::dot_fma([(&w[0], &w[1]), (&w[2], &w[3])]));
            acc ^= r.limbs[0] as i64;
        }
    }
    let b = t.elapsed().as_nanos() as f64 / (reps * 4093) as f64;
    for w in vals.windows(4).take(2000) {
        assert_eq!(
            MpFloat::<N>::dot_fma2(&w[0], &w[1], &w[2], &w[3]),
            MpFloat::<N>::dot_fma([(&w[0], &w[1]), (&w[2], &w[3])])
        );
    }
    println!("N={N:2}: fmma {a:6.1} ns, exact dot of 2 {b:6.1} ns ({acc})");
}

#[test]
#[ignore]
fn time_fmma_vs_exact_dot() {
    time_fmma::<4>();
    time_fmma::<8>();
    time_fmma::<12>();
}

// Narrow dot_fma2 (exact dot) must equal mpfr_fmma bit for bit.
fn narrow_fmma_matches_mpfr<const N: usize>() {
    let mut s = 0x51ed2701f3a4b5c7u64 ^ N as u64;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let value = |next: &mut dyn FnMut() -> u64| {
        let mut limbs = [0u64; N];
        limbs.iter_mut().for_each(|l| *l = next());
        limbs[N - 1] |= 1 << 63;
        let kind = if next() & 1 == 0 {
            mpfr::REGULAR_KIND
        } else {
            -mpfr::REGULAR_KIND
        };
        let e = (next() % 300) as i64 - 150;
        MpFloat::<N> {
            limbs,
            kind,
            exponent: e as mpfr::exp_t,
        }
    };
    for i in 0..20_000 {
        let (a, b, c) = (value(&mut next), value(&mut next), value(&mut next));
        // Every fourth case cancels the first product exactly.
        let d = if i % 4 == 0 {
            -(a * b) / c
        } else {
            value(&mut next)
        };
        let reference = {
            let (a, b, c, d) = (
                a.descriptor(),
                b.descriptor(),
                c.descriptor(),
                d.descriptor(),
            );
            MpFloat::<N>::output(|r| unsafe {
                mpfr::fmma(r, &a, &b, &c, &d, ROUND);
            })
        };
        let fast = MpFloat::<N>::dot_fma2(&a, &b, &c, &d);
        let norm = |v: MpFloat<N>| {
            if v.kind.abs() == mpfr::ZERO_KIND {
                (0, 0, [0; N])
            } else {
                (v.kind, v.exponent as i64, v.limbs)
            }
        };
        assert_eq!(norm(fast), norm(reference), "N={N} case {i}");
    }
}

#[test]
fn narrow_fmma_matches_mpfr_fmma() {
    narrow_fmma_matches_mpfr::<1>();
    narrow_fmma_matches_mpfr::<2>();
    narrow_fmma_matches_mpfr::<3>();
    narrow_fmma_matches_mpfr::<4>();
}
