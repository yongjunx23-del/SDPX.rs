use super::tests::{runtime, scatter};
use super::*;
use crate::solver::core::{
    traits::{ProblemData, Solution},
    SolverStatus,
};
use crate::solver::implementations::default::owned_hsd::*;
use sdpx_arithmetic::MpFloat;

fn num<T: FloatT>(v: i32) -> T {
    T::from_i32(v).unwrap()
}
fn exact<T: FloatT>(a: T, b: T) {
    if b.is_nan() {
        assert!(a.is_nan());
    } else {
        assert_eq!(a, b);
        assert_eq!(a.is_sign_negative(), b.is_sign_negative());
    }
}
fn settings<T: FloatT>() -> DefaultSettings<T> {
    DefaultSettings {
        verbose: false,
        presolve_enable: false,
        equilibrate_enable: true,
        #[cfg(feature = "sdp")]
        chordal_decomposition_enable: false,
        ..DefaultSettings::default()
    }
}
fn ordinary<T: FloatT>(settings: &DefaultSettings<T>) -> DefaultProblemData<T> {
    let a = CscMatrix::from(&[
        [num::<T>(8), num(2)],
        [num(8), num(2)],
        [num(0), num(32)],
        [num(4), num(0)],
    ]);
    let b = if settings.presolve_enable {
        vec![num(6), num(6), T::infinity(), num(9)]
    } else {
        vec![num(6), num(6), num(5), num(9)]
    };
    let mut d = DefaultProblemData::new(
        &CscMatrix::from(&[[num(16), num(0)], [num(0), num(64)]]),
        &[num(2), num(-8)],
        &a,
        &b,
        &[
            SupportedConeT::ZeroConeT(2),
            SupportedConeT::NonnegativeConeT(2),
        ],
        settings,
    );
    d.equilibrate(&CompositeCone::new(&d.cones), settings);
    d
}

// These are deliberately arbitrary internal points, not certified solver outputs.
fn compare<T: FloatT>(
    build: impl Fn() -> DefaultProblemData<T>,
    settings: &DefaultSettings<T>,
    original: (usize, usize),
    presolve: bool,
    chordal: bool,
) {
    for status in [
        SolverStatus::Solved,
        SolverStatus::PrimalInfeasible,
        SolverStatus::DualInfeasible,
    ] {
        let reference = build();
        let second = build();
        assert_eq!((reference.n, reference.m), (second.n, second.m));
        assert_eq!(reference.presolver.is_some(), presolve);
        let keep = reference
            .presolver
            .as_ref()
            .map(|p| p.reduce_map.as_ref().unwrap().keep_logical.clone());
        let internal_cones = reference.cones.clone();
        let mut state = runtime(second, 4, settings.clone(), original);
        assert_eq!(state.data.internal_cones, internal_cones);
        assert_eq!(
            (state.data.layout.n, state.data.layout.m),
            (reference.n, reference.m)
        );
        assert_eq!(state.data.presolver.is_some(), presolve);
        if let Some(p) = &state.data.presolver {
            assert_eq!(p.mfull, original.1);
            assert_eq!(p.mreduced, reference.m);
            assert_eq!(
                p.reduce_map.as_ref().unwrap().keep_logical,
                keep.clone().unwrap()
            );
            assert_eq!(keep.as_ref().unwrap(), &vec![true, false, false, true]);
        }
        #[cfg(feature = "sdp")]
        {
            assert_eq!(state.data.chordal_info.is_some(), chordal);
            if let Some(c) = &state.data.chordal_info {
                let r = reference.chordal_info.as_ref().unwrap();
                assert_eq!(c.init_dims, r.init_dims);
                assert_eq!(c.init_cones, r.init_cones);
                assert_eq!(c.spatterns.len(), r.spatterns.len());
                assert!(!c.spatterns.is_empty());
                assert_ne!((reference.n, reference.m), original);
            }
        }
        #[cfg(not(feature = "sdp"))]
        assert!(!chordal);
        // Recovery metadata is central; shards must not carry redundant copies.
        for (shard, variables) in state.data.blocks.iter().zip(&state.variables.blocks) {
            assert!(shard.presolver.is_none());
            #[cfg(feature = "sdp")]
            assert!(shard.chordal_info.is_none());
            assert_eq!(shard.n, variables.x.len());
            assert_eq!(shard.m, variables.s.len());
        }
        let mut point = DefaultVariables::new(reference.n, reference.m);
        for (i, x) in point.x.iter_mut().enumerate() {
            *x = num::<T>(i as i32 + 1) / num(8);
        }
        for (i, s) in point.s.iter_mut().enumerate() {
            *s = num::<T>(i as i32 + 2) / num(4);
        }
        for (i, z) in point.z.iter_mut().enumerate() {
            *z = -num::<T>(i as i32 + 3) / num(16);
        }
        point.τ = num(2);
        point.κ = num(5);
        scatter(&mut state.variables, &point);
        for &row in &state.data.layout.border_rows {
            let mut counted = 0;
            for (rank, (ids, variables)) in state
                .data
                .layout
                .owners
                .iter()
                .zip(&state.variables.blocks)
                .enumerate()
            {
                let local = ids.rows.binary_search(&row).unwrap();
                exact(
                    variables.s[local],
                    if rank == 0 { point.s[row] } else { T::zero() },
                );
                exact(variables.z[local], point.z[row]);
                counted += usize::from(ids.counted_rows.contains(&local));
            }
            assert_eq!(counted, 1);
        }
        let mut info = DefaultInfo::new();
        info.status = status;
        info.iterations = 17;
        info.solve_time = 1.25;
        info.cost_primal = num(7);
        info.cost_dual = num(-3);
        info.res_primal = num::<T>(1) / num(32);
        info.res_dual = num::<T>(1) / num(64);
        let normalization = if status.is_infeasible() {
            num::<T>(5).recip()
        } else {
            num::<T>(2).recip()
        };
        let first_x = point.x[0] * reference.equilibration.d[0] * normalization;
        let mut expected = DefaultSolution::new(original.0, original.1);
        expected.post_process(&reference, &mut point, &info, settings);
        expected.finalize(&info);
        let info = OwnedInfo(info);
        state
            .solution
            .post_process(&state.data, &mut state.variables, &info, settings);
        state.solution.finalize(&info);
        let actual = &state.solution.0;
        if !chordal {
            exact(actual.x[0], first_x);
        }
        assert_eq!(actual.status, expected.status);
        assert_eq!(actual.iterations, expected.iterations);
        assert_eq!(actual.solve_time, expected.solve_time);
        exact(actual.obj_val, expected.obj_val);
        exact(actual.obj_val_dual, expected.obj_val_dual);
        exact(actual.r_prim, expected.r_prim);
        exact(actual.r_dual, expected.r_dual);
        for (a, b) in [
            (&actual.x, &expected.x),
            (&actual.s, &expected.s),
            (&actual.z, &expected.z),
        ] {
            assert_eq!(a.len(), b.len());
            for (&a, &b) in a.iter().zip(b) {
                exact(a, b);
            }
        }
        if presolve {
            exact(actual.s[1], T::zero());
            exact(actual.z[1], T::zero());
            exact(actual.z[2], T::zero());
        }
    }
}
fn ordinary_and_presolve<T: FloatT>() {
    let mut s = settings::<T>();
    let d = ordinary(&s);
    assert!(d.equilibration.d.iter().any(|&x| x != T::one()));
    compare(|| ordinary(&s), &s, (2, 4), false, false);
    s.presolve_enable = true;
    assert_eq!(ordinary(&s).m, 2);
    compare(|| ordinary(&s), &s, (2, 4), true, false);
}
#[test]
fn owned_recovery_f64() {
    ordinary_and_presolve::<f64>();
}
#[test]
fn owned_recovery_mpfr256() {
    ordinary_and_presolve::<MpFloat<4>>();
}
#[test]
fn owned_recovery_mpfr512() {
    ordinary_and_presolve::<MpFloat<8>>();
}

#[cfg(feature = "sdp")]
fn chordal<T: FloatT>() {
    let mut s = settings::<T>();
    s.chordal_decomposition_enable = true;
    s.chordal_decomposition_merge_method = "none".into();
    // Arbitrary points need not be PSD; completion is a separate numerical
    // operation. This test exercises both existing structural reverse maps.
    s.chordal_decomposition_complete_dual = false;
    for compact in [false, true] {
        s.chordal_decomposition_compact = compact;
        let build = || {
            // Path graph 0--1--2--3: three overlapping 2x2 cliques.
            // Existing preprocessing skips PSD dimensions through three.
            let a = CscMatrix::new(
                10,
                4,
                vec![0, 2, 4, 6, 7],
                vec![0, 1, 2, 4, 5, 8, 9],
                vec![
                    num::<T>(8),
                    num(2),
                    num(4),
                    num(2),
                    num(16),
                    num(2),
                    num(32),
                ],
            );
            let mut d = DefaultProblemData::new(
                &CscMatrix::identity(4),
                &[num(1), num(2), num(3), num(4)],
                &a,
                &[
                    num(1),
                    num(0),
                    num(1),
                    num(0),
                    num(0),
                    num(1),
                    num(0),
                    num(0),
                    num(0),
                    num(1),
                ],
                &[SupportedConeT::PSDTriangleConeT(4)],
                &s,
            );
            d.equilibrate(&CompositeCone::new(&d.cones), &s);
            d
        };
        compare(build, &s, (4, 10), false, true);
    }
}
#[cfg(feature = "sdp")]
#[test]
fn owned_recovery_chordal_f64() {
    chordal::<f64>();
}
#[cfg(feature = "sdp")]
#[test]
fn owned_recovery_chordal_mpfr256() {
    chordal::<MpFloat<4>>();
}
#[cfg(feature = "sdp")]
#[test]
fn owned_recovery_chordal_mpfr512() {
    chordal::<MpFloat<8>>();
}
