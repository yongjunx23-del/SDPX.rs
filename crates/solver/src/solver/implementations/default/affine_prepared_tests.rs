use super::*;
use crate::solver::core::{cones::*, traits::Variables, ScalingStrategy, StepDirection};
use crate::solver::SupportedConeT::*;

fn equal<T: FloatT>(a: &[T], b: &[T]) {
    assert_eq!(a.len(), b.len());
    for (a, b) in a.iter().zip(b) {
        assert!(*a == *b || (a.is_nan() && b.is_nan()));
    }
}
fn fixture<T: FloatT>(
    threads: usize,
    asymmetric: bool,
) -> (CompositeCone<T>, DefaultVariables<T>, DefaultVariables<T>) {
    let mut kinds = vec![
        ZeroConeT(1),
        NonnegativeConeT(2),
        SecondOrderConeT(3),
        PSDTriangleConeT(16),
        PSDTriangleConeT(16),
        PSDTriangleConeT(16),
        PSDTriangleConeT(16),
    ];
    if asymmetric {
        kinds.insert(3, ExponentialConeT());
    }
    let mut cones = CompositeCone::new(&kinds);
    cones.configure_threads(threads).unwrap();
    assert_eq!(cones.cone_threads(), threads);
    let mut v = DefaultVariables::new(2, cones.numel());
    cones.unit_initialization(&mut v.z, &mut v.s);
    for (i, (s, z)) in v.s.iter_mut().zip(&mut v.z).enumerate() {
        let a = T::from_usize(i % 7 + 1).unwrap() / T::from_usize(32).unwrap();
        *s *= T::one() + a;
        *z *= T::one() + a + a;
    }
    // Noncommuting SPD blocks: positive diagonal plus distinct rank-one terms.
    let mut offset = if asymmetric { 9 } else { 6 };
    for _ in 0..4 {
        let mut row = offset;
        for j in 0..16 {
            for i in 0..=j {
                let u = T::from_usize((i + 1) * (j + 1)).unwrap() / T::from_usize(1024).unwrap();
                let w = T::from_usize((16 - i) * (16 - j)).unwrap() / T::from_usize(1024).unwrap();
                let scale = if i == j { T::one() } else { T::SQRT_2() };
                v.s[row] += u * scale;
                v.z[row] += w * scale;
                row += 1;
            }
        }
        offset = row;
    }
    assert!(cones.update_scaling(&v.s, &v.z, T::one(), ScalingStrategy::PrimalDual));
    let mut step = DefaultVariables::new(2, cones.numel());
    for (i, (s, z)) in step.s.iter_mut().zip(&mut step.z).enumerate() {
        *s = T::from_usize(i % 5 + 1).unwrap() / T::from_usize(16).unwrap();
        *z = -T::from_usize(i % 7 + 1).unwrap() / T::from_usize(8).unwrap();
    }
    // An early non-PSD bound must not cause later PSD consumption to be skipped.
    step.z[1] = -T::from_usize(100).unwrap();
    (cones, v, step)
}
fn check<T: FloatT>() {
    let settings = DefaultSettings::<T>::default();
    for threads in [1, 4] {
        for cap in [T::zero(), T::one(), -T::one(), T::infinity(), T::nan()] {
            let (mut raw, v, mut dr) = fixture::<T>(threads, false);
            let (mut prep, _, mut dp) = fixture::<T>(threads, false);
            let saved_z = dr.z.clone();
            let saved_s = dr.s.clone();
            let a = raw.step_length(&dr.z, &dr.s, &v.z, &v.s, settings.core(), cap);
            let again = raw.step_length(&dr.z, &dr.s, &v.z, &v.s, settings.core(), cap);
            equal(&[a.0, a.1], &[again.0, again.1]);
            equal(&dr.z, &saved_z);
            equal(&dr.s, &saved_s);
            let b =
                prep.prepare_affine_bounds(&mut dp.z, &mut dp.s, &v.z, &v.s, settings.core(), cap);
            equal(&[a.0, a.1], &[b.0, b.1]);
            let mut sr = vec![T::zero(); v.s.len()];
            let mut sp = sr.clone();
            raw.combined_ds_shift(&mut sr, &mut dr.z, &mut dr.s, T::one());
            prep.combined_shift_impl(&mut sp, &mut dp.z, &mut dp.s, T::one(), true);
            equal(&sr, &sp);
            equal(&dr.z, &dp.z);
            equal(&dr.s, &dp.s);
        }
        let (mut raw, v, mut dr) = fixture::<T>(threads, true);
        let (mut prep, _, mut dp) = fixture::<T>(threads, true);
        let residuals = DefaultResiduals::new(v.x.len(), v.s.len());
        let mut rr = DefaultVariables::new(v.x.len(), v.s.len());
        let mut rp = DefaultVariables::new(v.x.len(), v.s.len());
        rr.affine_step_rhs(&residuals, &v, &raw);
        rp.affine_step_rhs(&residuals, &v, &prep);
        let a = v.calc_step_length(&dr, &mut raw, &settings, StepDirection::Affine);
        let b = v.prepare_affine_step_length(&mut dp, &mut prep, &settings);
        equal(&[a], &[b]);
        rr.combined_step_rhs(
            &residuals,
            &v,
            &mut raw,
            &mut dr,
            T::one(),
            T::one(),
            T::one(),
        );
        rp.combined_step_rhs_prepared(&residuals, &v, &mut prep, &mut dp, T::one(), T::one());
        equal(&rr.x, &rp.x);
        equal(&rr.z, &rp.z);
        equal(&rr.s, &rp.s);
        equal(&[rr.τ, rr.κ], &[rp.τ, rp.κ]);
        equal(&dr.z, &dp.z);
        equal(&dr.s, &dp.s);
    }
}
#[test]
fn consuming_affine_f64() {
    check::<f64>();
}
#[test]
fn consuming_affine_mpfr256() {
    check::<sdpx_arithmetic::Bits256>();
}
#[test]
fn consuming_affine_mpfr512() {
    check::<sdpx_arithmetic::Bits512>();
}
