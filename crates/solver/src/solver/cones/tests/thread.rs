//! Kernel equivalence uses identical arithmetic inside each cone and leaves
//! reductions/line-search ordering untouched. Run native PSD tests with BLAS=1.
use crate::{
    algebra::FloatT,
    solver::{
        cones::{CompositeCone, Cone, SupportedConeT},
        core::ScalingStrategy,
    },
};
use sdpx_arithmetic::Bits256;

fn check_kernels<T: FloatT>(kinds: &[SupportedConeT<T>]) {
    check_kernels_threads(kinds, 4);
}

fn check_kernels_threads<T: FloatT>(kinds: &[SupportedConeT<T>], threads: usize) {
    let mut serial = CompositeCone::new(kinds);
    let mut parallel = CompositeCone::new(kinds);
    serial.configure_threads(1).unwrap();
    parallel.configure_threads(threads).unwrap();
    assert_eq!(serial.cone_threads(), 1);
    assert!(
        parallel.cone_threads() > 1,
        "fixture must exercise the parallel path"
    );
    let m = serial.numel();
    let mut s = vec![T::zero(); m];
    let mut z = vec![T::zero(); m];
    serial.unit_initialization(&mut z, &mut s);
    let denominator = T::from_usize(64 * m).unwrap();
    for i in 0..m {
        let perturbation = T::from_usize(i + 1).unwrap() / denominator;
        s[i] *= T::one() + perturbation;
        z[i] *= T::one() + perturbation + perturbation;
    }
    {
        let mut offset = 0;
        for (cone, kind) in serial.iter().zip(kinds) {
            if let SupportedConeT::PSDTriangleConeT(n) = kind {
                // S=D+uu' and Z=E+vv' have positive diagonal D,E, hence
                // are SPD. Distinct rank-one terms make them noncommuting.
                // Construct every coefficient in T, including svec's sqrt(2).
                let denominator = T::from_usize(n + 1).unwrap();
                let two = T::one() + T::one();
                let root_two = two.sqrt();
                let mut index = offset;
                for j in 0..*n {
                    let uj = T::from_usize(j + 1).unwrap() / denominator;
                    let vj = (if j % 2 == 0 { T::one() } else { -T::one() })
                        / T::from_usize(j + 1).unwrap();
                    for i in 0..=j {
                        let ui = T::from_usize(i + 1).unwrap() / denominator;
                        let vi = (if i % 2 == 0 { T::one() } else { -T::one() })
                            / T::from_usize(i + 1).unwrap();
                        s[index] = ui * uj;
                        z[index] = vi * vj;
                        if i == j {
                            s[index] += T::one() + ui;
                            z[index] += two + T::from_usize(n - i).unwrap() / denominator;
                        } else {
                            s[index] *= root_two;
                            z[index] *= root_two;
                        }
                        index += 1;
                    }
                }
            }
            offset += cone.numel();
        }
    }
    let saved_s = s.clone();
    let saved_z = z.clone();
    let x: Vec<T> = (0..m)
        .map(|i| T::from_usize(i % 13).unwrap() / T::from_usize(64).unwrap())
        .collect();
    let saved_x = x.clone();
    for iteration in 0..3 {
        let mu = T::one() / T::from_usize(iteration + 1).unwrap();
        assert!(serial.update_scaling(&s, &z, mu, ScalingStrategy::Dual));
        assert!(parallel.update_scaling(&s, &z, mu, ScalingStrategy::Dual));
        if matches!(kinds, [SupportedConeT::NonnegativeConeT(_)]) {
            let mut a = vec![T::zero(); m];
            let mut b = a.clone();
            serial.affine_ds(&mut a, &s);
            parallel.affine_ds(&mut b, &s);
            assert_eq!(a, b); // also checks lambda, not just w
            serial.get_Hs(&mut a);
            parallel.get_Hs(&mut b);
            assert_eq!(a, b);
            let mut zs = z.clone();
            let mut zp = z.clone();
            assert_eq!(
                serial.margins(&mut zs, super::PrimalOrDualCone::DualCone),
                parallel.margins(&mut zp, super::PrimalOrDualCone::DualCone),
            );
            let settings = crate::solver::CoreSettings::default();
            assert_eq!(
                serial.step_length(&x, &x, &z, &s, &settings, T::one()),
                parallel.step_length(&x, &x, &z, &s, &settings, T::one()),
            );
            assert_eq!(
                serial.compute_barrier(&mut zs, &mut s.clone(), &x, &x, T::one()),
                parallel.compute_barrier(&mut zp, &mut s.clone(), &x, &x, T::one()),
            );
        }
        let mut ys = vec![T::zero(); m];
        let mut yp = ys.clone();
        let mut ws = ys.clone();
        let mut wp = ys.clone();
        serial.mul_Hs(&mut ys, &x, &mut ws);
        parallel.mul_Hs(&mut yp, &x, &mut wp);
        assert_eq!(ys, yp);
        assert_eq!(ws, wp);
        assert!(yp.iter().all(|v| v.is_finite()));

        let mut shift_s = vec![T::zero(); m];
        let mut shift_p = shift_s.clone();
        let mut dz_s = x.clone();
        let mut dz_p = x.clone();
        let mut ds_s: Vec<T> = x.iter().map(|x| -*x).collect();
        let mut ds_p = ds_s.clone();
        serial.combined_ds_shift(&mut shift_s, &mut dz_s, &mut ds_s, mu);
        parallel.combined_ds_shift(&mut shift_p, &mut dz_p, &mut ds_p, mu);
        assert_eq!(shift_s, shift_p);
        assert_eq!(dz_s, dz_p);
        assert_eq!(ds_s, ds_p);

        let mut out_s = vec![T::zero(); m];
        let mut out_p = out_s.clone();
        ws.fill(T::zero());
        wp.fill(T::zero());
        serial.Δs_from_Δz_offset(&mut out_s, &shift_s, &mut ws, &z);
        parallel.Δs_from_Δz_offset(&mut out_p, &shift_p, &mut wp, &z);
        assert_eq!(out_s, out_p);
        assert_eq!(ws, wp);
        assert!(out_p.iter().all(|v| v.is_finite()));
        assert_eq!(s, saved_s);
        assert_eq!(z, saved_z);
        assert_eq!(x, saved_x);
        assert!(parallel.cone_threads() > 1);
    }
    // Reconfiguration replaces a pool only outside an active kernel phase.
    parallel.configure_threads(1).unwrap();
    assert_eq!(parallel.cone_threads(), 1);
}

fn mixed<T: FloatT>() -> Vec<SupportedConeT<T>> {
    let mut kinds = vec![SupportedConeT::SecondOrderConeT(2048); 8];
    let half = T::one() / T::from_usize(2).unwrap();
    kinds.extend([
        SupportedConeT::NonnegativeConeT(64),
        SupportedConeT::ZeroConeT(3),
        SupportedConeT::ExponentialConeT(),
        SupportedConeT::PowerConeT(half),
        SupportedConeT::GenPowerConeT(vec![half, half], 2),
    ]);
    kinds
}

#[test]
fn mixed_f64_parallel_equivalence() {
    check_kernels(&mixed::<f64>());
}
#[test]
fn mixed_mpfr256_parallel_equivalence() {
    check_kernels(&mixed::<Bits256>());
}

#[test]
fn psd_f64_parallel_equivalence() {
    check_kernels::<f64>(&vec![SupportedConeT::PSDTriangleConeT(16); 4]);
}
#[test]
fn psd_mpfr256_parallel_equivalence() {
    check_kernels::<Bits256>(&vec![SupportedConeT::PSDTriangleConeT(16); 4]);
}

#[test]
fn small_and_indivisible_cone_workloads_remain_serial() {
    for kinds in [
        vec![SupportedConeT::SecondOrderConeT(3); 8],
        vec![SupportedConeT::NonnegativeConeT(4096)],
        vec![SupportedConeT::SecondOrderConeT(32768)],
    ] {
        let mut cones = CompositeCone::<f64>::new(&kinds);
        cones.configure_threads(8).unwrap();
        assert_eq!(cones.cone_threads(), 1);
    }
}

#[test]
fn single_orthant_f64_parallel_equivalence() {
    // Nonmultiple size exercises the final short chunk at both budgets.
    for threads in [2, 4] {
        check_kernels_threads::<f64>(&[SupportedConeT::NonnegativeConeT(16387)], threads);
    }
}

#[test]
fn single_orthant_mpfr256_parallel_equivalence() {
    for threads in [2, 4] {
        check_kernels_threads::<Bits256>(&[SupportedConeT::NonnegativeConeT(1027)], threads);
    }
}

#[test]
fn single_orthant_mpfr512_parallel_equivalence() {
    for threads in [2, 4] {
        check_kernels_threads::<sdpx_arithmetic::Bits512>(
            &[SupportedConeT::NonnegativeConeT(259)],
            threads,
        );
    }
}

#[test]
fn single_orthant_pool_has_one_outer_lane_and_bounded_chunks() {
    let mut cones = CompositeCone::<f64>::new(&[SupportedConeT::NonnegativeConeT(16387)]);
    cones.configure_threads(4).unwrap();
    let threading = cones.threading.as_ref().unwrap();
    assert_eq!(threading.pool.current_num_threads(), 4);
    assert_eq!(threading.lanes.len(), 1);
    assert_eq!(threading.orthant_chunk, Some(4097));
    cones.configure_threads(1).unwrap();
    assert!(cones.threading.is_none());
    cones.configure_threads(2).unwrap();
    let threading = cones.threading.as_ref().unwrap();
    assert_eq!(threading.pool.current_num_threads(), 2);
    assert_eq!(threading.lanes.len(), 1);
    assert_eq!(threading.orthant_chunk, Some(8194));
    // Exercise the re-enabled instance against a fresh serial instance.
    let mut serial = CompositeCone::<f64>::new(&[SupportedConeT::NonnegativeConeT(16387)]);
    let s = vec![2.; 16387];
    let z = vec![3.; 16387];
    assert!(serial.update_scaling(&s, &z, 1., ScalingStrategy::Dual));
    assert!(cones.update_scaling(&s, &z, 1., ScalingStrategy::Dual));
    let mut a = vec![0.; 16387];
    let mut b = a.clone();
    serial.affine_ds(&mut a, &s);
    cones.affine_ds(&mut b, &s);
    assert_eq!(a, b);
    serial.mul_Hs(&mut a, &s, &mut vec![0.; 16387]);
    cones.mul_Hs(&mut b, &s, &mut vec![0.; 16387]);
    assert_eq!(a, b);
}

fn single_orthant_complete_lp<T: FloatT>(m: usize) {
    use crate::{
        algebra::CscMatrix,
        solver::{DefaultSettings, DefaultSolution, DefaultSolver, IPSolver, SolverStatus},
    };
    // min x subject to x >= (i % 7 + 1)/7. The optimum is exactly one.
    // One column keeps the factorization small while exercising a large cone.
    let a = CscMatrix::new(m, 1, vec![0, m], (0..m).collect(), vec![-T::one(); m]);
    let b: Vec<T> = (0..m)
        .map(|i| -T::from_usize(i % 7 + 1).unwrap() / T::from_usize(7).unwrap())
        .collect();
    let mut baseline: Option<DefaultSolution<T>> = None;
    for threads in [1, 2, 4] {
        let settings = DefaultSettings::<T> {
            verbose: false,
            max_threads: threads,
            // Keep the large cone and isolate cone parallelism from LDL backend
            // parallelism. Precision and convergence tolerances are unchanged.
            presolve_enable: false,
            direct_solve_method: "qdldl".into(),
            kkt_form: "augmented".into(),
            ..DefaultSettings::default()
        };
        let tol_feas = settings.tol_feas;
        let tol_gap_abs = settings.tol_gap_abs;
        let tol_gap_rel = settings.tol_gap_rel;
        let mut solver = DefaultSolver::new(
            &CscMatrix::zeros((1, 1)),
            &[T::one()],
            &a,
            &b,
            &[SupportedConeT::NonnegativeConeT(m)],
            settings,
        )
        .unwrap();
        assert_eq!(solver.cones.cone_threads(), threads as usize);
        solver.solve(); // Returned equations and cone feasibility are checked below.
        let sol = solver.solution;
        assert_eq!(sol.status, SolverStatus::Solved);
        assert!(sol.iterations > 0);
        let mut sum_z = T::zero();
        for i in 0..m {
            assert!(sol.s[i] >= T::zero());
            assert!(sol.z[i] >= T::zero());
            let error = (-sol.x[0] + sol.s[i] - b[i]).abs();
            let work = sol.x[0].abs() + sol.s[i].abs() + b[i].abs();
            assert!(error / work < tol_feas);
            sum_z += sol.z[i];
        }
        assert!((T::one() - sum_z).abs() / (T::one() + sum_z) < tol_feas);
        let gap = (sol.obj_val - sol.obj_val_dual).abs();
        assert!(
            gap < tol_gap_abs
                || gap / T::one().max(sol.obj_val.abs().min(sol.obj_val_dual.abs())) < tol_gap_rel
        );
        if let Some(ref serial) = baseline {
            assert_eq!(sol.iterations, serial.iterations);
            assert_eq!(sol.x, serial.x);
            assert_eq!(sol.s, serial.s);
            assert_eq!(sol.z, serial.z);
            assert_eq!(sol.obj_val, serial.obj_val);
            assert_eq!(sol.obj_val_dual, serial.obj_val_dual);
            assert_eq!(sol.r_prim, serial.r_prim);
            assert_eq!(sol.r_dual, serial.r_dual);
        } else {
            baseline = Some(sol);
        }
    }
}

#[test]
fn single_orthant_complete_lp_f64() {
    single_orthant_complete_lp::<f64>(16387);
}

#[test]
#[ignore = "extended: MPFR pool/threading sweep; default f64 covers the equivalence logic"]
fn single_orthant_complete_lp_mpfr256() {
    single_orthant_complete_lp::<Bits256>(1027);
}

#[test]
#[ignore = "extended: MPFR pool/threading sweep; default f64 covers the equivalence logic"]
fn single_orthant_complete_lp_mpfr512() {
    single_orthant_complete_lp::<sdpx_arithmetic::Bits512>(259);
}

#[test]
fn single_psd_shares_wider_pool_with_condensed_columns() {
    let mut cones = CompositeCone::<f64>::new(&[SupportedConeT::PSDTriangleConeT(32)]);
    for workers in [2, 4, 8, 1, 4] {
        cones.configure_threads(workers).unwrap();
        assert_eq!(cones.cone_threads(), workers);
        if let Some(threading) = &cones.threading {
            assert_eq!(threading.lanes.len(), 1);
            assert!(threading.orthant_chunk.is_none());
        }
    }
}

fn psd_steps<T: FloatT>() {
    let num = |n: i32| T::from_i32(n).unwrap();
    let half = T::one() / num(2);
    // Nonsymmetric descriptors occur before, between and after symmetric
    // cones; their original second-pass order must remain authoritative.
    let kinds = vec![
        SupportedConeT::ExponentialConeT(),
        SupportedConeT::PSDTriangleConeT(16),
        SupportedConeT::NonnegativeConeT(2),
        SupportedConeT::ZeroConeT(0),
        SupportedConeT::PowerConeT(half),
        SupportedConeT::PSDTriangleConeT(32),
        SupportedConeT::SecondOrderConeT(3),
        SupportedConeT::PSDTriangleConeT(0),
        SupportedConeT::ZeroConeT(1),
        SupportedConeT::ExponentialConeT(),
    ];
    let mut serial = CompositeCone::new(&kinds);
    let mut pooled = CompositeCone::new(&kinds);
    let m = serial.numel();
    let (mut z, mut s) = (vec![T::zero(); m], vec![T::zero(); m]);
    serial.unit_initialization(&mut z, &mut s);
    assert!(serial.update_scaling(&s, &z, T::one(), ScalingStrategy::Dual));
    assert!(pooled.update_scaling(&s, &z, T::one(), ScalingStrategy::Dual));
    let settings = crate::solver::CoreSettings::default();
    let mut saved_pointer = None;
    for width in [1, 8, 2, 4, 1, 8] {
        pooled.configure_threads(width).unwrap();
        assert_eq!(pooled.cone_threads(), width);
        if width > 1 {
            assert!(pooled.threading.as_ref().unwrap().sym_step_lanes.len() >= 2);
        }
        for case in 0..3 {
            let (mut dz, mut ds) = (vec![T::zero(); m], vec![T::zero(); m]);
            for (index, (kind, range)) in kinds.iter().zip(&serial.rng_cones).enumerate() {
                let rate = match kind {
                    SupportedConeT::PSDTriangleConeT(_) => {
                        if case == 2 {
                            num(1)
                        } else if (index == 1) == (case == 0) {
                            -num(4)
                        } else {
                            -half
                        }
                    }
                    SupportedConeT::NonnegativeConeT(_) => {
                        if case == 1 {
                            -num(16)
                        } else if case == 2 {
                            num(1)
                        } else {
                            -num(2)
                        }
                    }
                    SupportedConeT::ExponentialConeT() | SupportedConeT::PowerConeT(_) => {
                        if case == 2 {
                            -num(3)
                        } else {
                            -half
                        }
                    }
                    _ => -half,
                };
                for i in range.clone() {
                    dz[i] = rate * z[i];
                    ds[i] = rate * s[i] / num(2);
                }
                if let SupportedConeT::PSDTriangleConeT(n) = kind {
                    if *n > 1 {
                        // Non-diagonal direction distinguishes PSD eigensolves
                        // from merely scanning diagonal coordinates.
                        dz[range.start + 1] = num(1) / num(32);
                        ds[range.start + 1] = -num(1) / num(64);
                    }
                }
            }
            for cap in [T::one(), half, num(1) / num(128), T::zero(), -T::zero()] {
                let a = serial.step_length(&dz, &ds, &z, &s, &settings, cap);
                let b = pooled.step_length(&dz, &ds, &z, &s, &settings, cap);
                assert_eq!(a, b, "width={width}, case={case}, cap={cap}");
                assert_eq!(a.0.is_sign_negative(), b.0.is_sign_negative());
                assert_eq!(a.1.is_sign_negative(), b.1.is_sign_negative());
                if width > 1 {
                    assert_eq!(pooled.sym_step_bounds.len(), kinds.len());
                    if let Some(pointer) = saved_pointer {
                        assert_eq!(pooled.sym_step_bounds.as_ptr(), pointer);
                    } else {
                        saved_pointer = Some(pooled.sym_step_bounds.as_ptr());
                    }
                }
            }
        }
        if let Some(pointer) = saved_pointer {
            assert_eq!(pooled.sym_step_bounds.as_ptr(), pointer);
        }
    }
    assert!(serial.sym_step_bounds.is_empty());
    // Outside the finite-positive cap domain, preserve the complete serial
    // path, including native min/NaN behavior and absence of cache writes.
    let kinds = [
        SupportedConeT::PSDTriangleConeT(16),
        SupportedConeT::PSDTriangleConeT(32),
    ];
    let mut serial = CompositeCone::<T>::new(&kinds);
    let mut pooled = CompositeCone::<T>::new(&kinds);
    pooled.configure_threads(8).unwrap();
    let m = serial.numel();
    let (mut z, mut s) = (vec![T::zero(); m], vec![T::zero(); m]);
    serial.unit_initialization(&mut z, &mut s);
    assert!(serial.update_scaling(&s, &z, T::one(), ScalingStrategy::Dual));
    assert!(pooled.update_scaling(&s, &z, T::one(), ScalingStrategy::Dual));
    let dz: Vec<_> = z.iter().map(|v| -*v).collect();
    let ds: Vec<_> = s.iter().map(|v| -*v).collect();
    for cap in [T::zero(), -T::zero(), -T::one(), T::infinity(), T::nan()] {
        let a = serial.step_length(&dz, &ds, &z, &s, &settings, cap);
        let b = pooled.step_length(&dz, &ds, &z, &s, &settings, cap);
        for (a, b) in [(a.0, b.0), (a.1, b.1)] {
            if a.is_nan() {
                assert!(b.is_nan());
            } else {
                assert_eq!(a, b);
                assert_eq!(a.is_sign_negative(), b.is_sign_negative());
            }
        }
        assert!(pooled.sym_step_bounds.is_empty());
    }
}
#[test]
fn psd_step_lengths_f64() {
    psd_steps::<f64>();
}
#[test]
#[ignore = "extended: MPFR pool/threading sweep; default f64 covers the equivalence logic"]
fn psd_step_lengths_mpfr256() {
    psd_steps::<Bits256>();
}
#[test]
#[ignore = "extended: MPFR pool/threading sweep; default f64 covers the equivalence logic"]
fn psd_step_lengths_mpfr512() {
    psd_steps::<sdpx_arithmetic::Bits512>();
}

/// A single large orthant takes the elementwise-chunk path; every other
/// cone mix takes the symmetric-lane bounds pass. Both must reproduce the
/// serial step lengths exactly, including signed-zero caps.
fn single_orthant_steps<T: FloatT>() {
    let num = |n: i32| T::from_i32(n).unwrap();
    let half = T::one() / num(2);
    // Sizes clear the structural MIN_LANE_WORK threshold even at f64
    // (one word per element), so both the chunk and the lane paths run.
    for kinds in [
        vec![SupportedConeT::NonnegativeConeT(32768)],
        vec![
            SupportedConeT::NonnegativeConeT(8192),
            SupportedConeT::SecondOrderConeT(64),
            SupportedConeT::ZeroConeT(8),
        ],
    ] {
        let mut serial = CompositeCone::new(&kinds);
        let mut pooled = CompositeCone::new(&kinds);
        pooled.configure_threads(4).unwrap();
        assert!(pooled.cone_threads() > 1);
        let m = serial.numel();
        let (mut z, mut s) = (vec![T::zero(); m], vec![T::zero(); m]);
        serial.unit_initialization(&mut z, &mut s);
        assert!(serial.update_scaling(&s, &z, T::one(), ScalingStrategy::Dual));
        assert!(pooled.update_scaling(&s, &z, T::one(), ScalingStrategy::Dual));
        let settings = crate::solver::CoreSettings::default();
        for case in 0..3 {
            let rate = [num(-2), -half, num(-16)][case];
            let dz: Vec<_> = z.iter().map(|v| rate * *v).collect();
            let ds: Vec<_> = s.iter().map(|v| rate * *v / num(2)).collect();
            for cap in [T::one(), half, num(1) / num(128), T::zero(), -T::zero()] {
                let a = serial.step_length(&dz, &ds, &z, &s, &settings, cap);
                let b = pooled.step_length(&dz, &ds, &z, &s, &settings, cap);
                assert_eq!(a, b, "case={case}, cap={cap}");
                assert_eq!(a.0.is_sign_negative(), b.0.is_sign_negative());
                assert_eq!(a.1.is_sign_negative(), b.1.is_sign_negative());
            }
        }
    }
}

#[test]
fn orthant_step_lengths_f64() {
    single_orthant_steps::<f64>();
}
#[test]
#[ignore = "extended: MPFR pool/threading sweep; default f64 covers the equivalence logic"]
fn orthant_step_lengths_mpfr256() {
    single_orthant_steps::<Bits256>();
}
#[test]
#[ignore = "extended: MPFR pool/threading sweep; default f64 covers the equivalence logic"]
fn orthant_step_lengths_mpfr512() {
    single_orthant_steps::<sdpx_arithmetic::Bits512>();
}
