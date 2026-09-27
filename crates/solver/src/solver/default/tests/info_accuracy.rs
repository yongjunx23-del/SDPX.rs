use super::*;
use crate::solver::{
    cones::{CompositeCone, SupportedConeT},
    core::traits::{ProblemData, Residuals},
};
use num_traits::FromPrimitive;
use sdpx_arithmetic::{Bits256, Scalar};

fn settings<T: FloatT>() -> DefaultSettings<T> {
    DefaultSettings {
        verbose: false,
        presolve_enable: false,
        input_sparse_dropzeros: false,
        equilibrate_enable: false,
        ..DefaultSettings::default()
    }
}
fn close<T: FloatT>(actual: T, expected: T) {
    let tolerance = T::from_usize(4096).unwrap() * T::epsilon() * expected.abs().max(T::one());
    assert!(
        (actual - expected).abs() <= tolerance,
        "actual={actual}, expected={expected}"
    );
}

fn large_iterate_upstream_normalization<T: FloatT>() {
    let mut settings = settings::<T>();
    let two = T::from_usize(2).unwrap();
    settings.tol_feas = two.powi(-40);
    let kinds = [SupportedConeT::NonnegativeConeT(1)];
    let mut data = DefaultProblemData::new(
        &CscMatrix::zeros((2, 2)),
        &[T::zero(); 2],
        &CscMatrix::from(&[[T::one(), -T::one()]]),
        &[T::zero()],
        &kinds,
        &settings,
    );
    for primal_fault in [true, false] {
        let variables = DefaultVariables {
            // Large enough for upstream normwise convergence, while keeping
            // the unit affine error representable in Float64 accumulation.
            x: vec![two.powi(44); 2],
            s: vec![if primal_fault { T::one() } else { T::zero() }],
            z: vec![if primal_fault { T::zero() } else { T::one() }],
            τ: T::one(),
            κ: T::zero(),
        };
        let mut residuals = DefaultResiduals::new(2, 1);
        residuals.update(&variables, &data);
        let mut info = DefaultInfo::new();
        info.update(&mut data, &variables, &residuals, &Timers::default());
        close(info.gap_abs, T::zero());
        let expected_primal =
            residuals.rz.norm() / T::one().max(variables.x.norm() + variables.s.norm());
        let expected_dual =
            residuals.rx.norm() / T::one().max(variables.x.norm() + variables.z.norm());
        close(info.res_primal, expected_primal);
        close(info.res_dual, expected_dual);
        // Upstream normalization accepts this synthetic point despite a unit
        // original equation error. External point audits remain independent.
        assert_eq!(
            if primal_fault {
                residuals.rz.norm_inf()
            } else {
                residuals.rx.norm_inf()
            },
            T::one()
        );
        info.check_convergence_full(&residuals, &settings);
        assert_eq!(info.status, SolverStatus::Solved);
    }
}

#[test]
fn large_iterate_accuracy_f64() {
    large_iterate_upstream_normalization::<f64>();
}
#[test]
fn large_iterate_accuracy_mpfr256() {
    large_iterate_upstream_normalization::<Bits256>();
}

fn ruiz_qp_original_coordinates<T: FloatT>() {
    let c = |n| T::from_i32(n).unwrap();
    let mut settings = settings::<T>();
    settings.equilibrate_enable = true;
    let p = CscMatrix::from(&[[c(16), c(2)], [c(0), c(8)]]);
    let a = CscMatrix::from(&[[c(4), c(0)], [c(0), T::one() / c(4)]]);
    let q = [c(64), c(128)];
    let b = [c(-2), c(3)];
    let kinds = [SupportedConeT::NonnegativeConeT(2)];
    let mut data = DefaultProblemData::new(&p, &q, &a, &b, &kinds, &settings);
    data.equilibrate(&CompositeCone::new(&kinds), &settings);
    assert_ne!(data.equilibration.c, T::one());
    assert!(data.equilibration.d.iter().any(|d| *d != T::one()));
    assert!(data.equilibration.e.iter().any(|e| *e != T::one()));
    let x = vec![T::one() / c(2), -T::one() / c(4)];
    let s = vec![T::one(), c(2)];
    let z = vec![c(3) / c(4), -c(5) / c(4)];
    let tau = c(2);
    let eq = &data.equilibration;
    let variables = DefaultVariables {
        x: x.iter()
            .zip(&eq.dinv)
            .map(|(&x, &di)| tau * di * x)
            .collect(),
        s: s.iter().zip(&eq.e).map(|(&s, &e)| tau * e * s).collect(),
        z: z.iter()
            .zip(&eq.einv)
            .map(|(&z, &ei)| tau * eq.c * ei * z)
            .collect(),
        τ: tau,
        κ: T::one() / c(4),
    };
    let saved = (
        variables.x.clone(),
        variables.s.clone(),
        variables.z.clone(),
    );
    let mut residuals = DefaultResiduals::new(2, 2);
    residuals.update(&variables, &data);
    let mut info = DefaultInfo::new();
    info.update(&mut data, &variables, &residuals, &Timers::default());
    // Directly compute original QP equations, independently of D/E/c/τ.
    let mut rp = s.clone();
    a.gemv(&mut rp, &x, T::one(), T::one());
    for i in 0..2 {
        rp[i] -= b[i];
    }
    let mut rd = q.to_vec();
    a.t().gemv(&mut rd, &z, T::one(), T::one());
    p.sym_up().symv(&mut rd, &x, T::one(), T::one());
    let expected_primal = rp.norm() / T::one().max(b.norm_inf() + x.norm() + s.norm());
    let expected_dual = rd.norm() / T::one().max(q.norm_inf() + x.norm() + z.norm());
    close(info.res_primal, expected_primal);
    close(info.res_dual, expected_dual);
    close(info.cost_primal, c(2));
    close(info.cost_dual, c(13) / c(4));
    assert_eq!((variables.x, variables.s, variables.z), saved);
}
#[test]
fn ruiz_qp_accuracy_f64() {
    ruiz_qp_original_coordinates::<f64>();
}
#[test]
fn ruiz_qp_accuracy_mpfr256() {
    ruiz_qp_original_coordinates::<Bits256>();
}

#[test]
fn updated_q_b_refresh_original_normalizers() {
    let mut settings = settings::<f64>();
    settings.equilibrate_enable = true;
    let mut solver = DefaultSolver::new(
        &CscMatrix::identity(2),
        &[32., 64.],
        &CscMatrix::from(&[[4., 0.], [0., 0.25]]),
        &[2., 3.],
        &[SupportedConeT::NonnegativeConeT(2)],
        settings,
    )
    .unwrap();
    close(solver.data.get_normq(), 64.);
    close(solver.data.get_normb(), 3.);
    solver.update_q(&vec![5., -7.]).unwrap();
    solver.update_b(&vec![-11., 13.]).unwrap();
    close(solver.data.get_normq(), 7.);
    close(solver.data.get_normb(), 13.);
    let eq = &solver.data.equilibration;
    let variables = DefaultVariables {
        x: vec![0.; 2],
        s: vec![0.; 2],
        z: vec![0.; 2],
        τ: 2.,
        κ: 0.,
    };
    let mut residuals = DefaultResiduals::new(2, 2);
    residuals.update(&variables, &solver.data);
    let old_primal = residuals.rz.norm_scaled(&eq.einv) / 2. / 13.;
    let old_dual = residuals.rx.norm_scaled(&eq.dinv) / 2. / eq.c / 7.;
    let mut info = DefaultInfo::new();
    info.update(&mut solver.data, &variables, &residuals, &Timers::default());
    close(info.res_primal, old_primal);
    close(info.res_dual, old_dual);
    // Updated zero affine terms refresh the ordinary residuals and data norms.
    solver.update_q(&vec![0.; 2]).unwrap();
    solver.update_b(&vec![0.; 2]).unwrap();
    residuals.update(&variables, &solver.data);
    info.update(&mut solver.data, &variables, &residuals, &Timers::default());
    assert_eq!(info.res_primal, 0.);
    assert_eq!(info.res_dual, 0.);
}

fn affine_scale_accuracy<T: FloatT>(tiny: T) {
    let kinds = [SupportedConeT::NonnegativeConeT(2)];
    let settings = settings::<T>();
    // The first equation is exact and dominates the global data norm; only
    // the tiny second equation is wrong, on both sides of the KKT system.
    let mut data = DefaultProblemData::new(
        &CscMatrix::zeros((2, 2)),
        &[T::one(), tiny],
        &CscMatrix::identity(2),
        &[T::one(), tiny],
        &kinds,
        &settings,
    );
    let variables = DefaultVariables {
        x: vec![T::one(), T::zero()],
        s: vec![T::zero(); 2],
        z: vec![-T::one(), T::zero()],
        τ: T::one(),
        κ: T::zero(),
    };
    let mut residuals = DefaultResiduals::new(2, 2);
    residuals.update(&variables, &data);
    let mut info = DefaultInfo::new();
    info.update(&mut data, &variables, &residuals, &Timers::default());
    assert!(info.res_primal < settings.tol_feas);
    assert!(info.res_dual < settings.tol_feas);
    // Retain the tiny-row discrepancy as external evidence: upstream ordinary
    // convergence uses the global norms and does not impose a per-row gate.
    assert_eq!(residuals.rz[1], -tiny);
    assert_eq!(residuals.rx[1], -tiny);
    info.check_convergence_full(&residuals, &settings);
    assert_eq!(info.status, SolverStatus::Solved);
    info.status = SolverStatus::MaxIterations;
    info.post_process(&residuals, &settings);
    assert_eq!(info.status, SolverStatus::AlmostSolved);
}

#[test]
fn affine_scale_accuracy_f64() {
    affine_scale_accuracy::<f64>(1e-14);
}

#[test]
fn affine_scale_accuracy_mpfr512() {
    // Far below f64's exponent range: no precision-losing conversion is viable.
    affine_scale_accuracy::<sdpx_arithmetic::Bits512>(
        sdpx_arithmetic::Bits512::from_u32(2).unwrap().powi(-1500),
    );
}

#[test]
fn homogeneous_boundary_equations_still_solve() {
    use crate::solver::IPSolver;
    for a in [0., -1.] {
        let mut solver = DefaultSolver::new(
            &CscMatrix::zeros((1, 1)),
            &[0.],
            &CscMatrix::from(&[[a]]),
            &[0.],
            &[SupportedConeT::NonnegativeConeT(1)],
            settings::<f64>(),
        )
        .unwrap();
        solver.solve();
        assert_eq!(solver.solution.status, SolverStatus::Solved);
        assert!(solver.info.res_primal < solver.settings.tol_feas);
        assert!(solver.info.res_dual < solver.settings.tol_feas);
    }
}

#[test]
fn ordinary_convergence_uses_live_non_nested_tolerances() {
    let residuals = DefaultResiduals::<f64>::new(1, 1);
    let mut info = DefaultInfo::new();
    info.res_primal = 1e-5;
    info.res_dual = 1e-5;
    info.gap_abs = 1e-5;
    info.gap_rel = 1e-5;
    info.ktratio = 0.;
    let mut settings = settings();
    info.check_convergence_full(&residuals, &settings);
    assert_eq!(info.status, SolverStatus::Unsolved);
    info.status = SolverStatus::MaxIterations;
    info.post_process(&residuals, &settings);
    assert_eq!(info.status, SolverStatus::AlmostSolved);

    // Swap the two sets: full-only convergence must use current settings,
    // without requiring an Info or residual refresh.
    std::mem::swap(&mut settings.tol_gap_abs, &mut settings.reduced_tol_gap_abs);
    std::mem::swap(&mut settings.tol_gap_rel, &mut settings.reduced_tol_gap_rel);
    std::mem::swap(&mut settings.tol_feas, &mut settings.reduced_tol_feas);
    info.status = SolverStatus::Unsolved;
    info.check_convergence_full(&residuals, &settings);
    assert_eq!(info.status, SolverStatus::Solved);
    info.status = SolverStatus::MaxIterations;
    info.post_process(&residuals, &settings);
    assert_eq!(info.status, SolverStatus::MaxIterations);
}

#[test]
fn ordinary_recovery_restores_accepted_metrics_and_variables() {
    let mut info = DefaultInfo::<f64>::new();
    let mut variables = DefaultVariables::new(1, 1);
    variables.x[0] = 2.;
    let mut previous = DefaultVariables::new(1, 1);
    info.cost_primal = 3.;
    info.cost_dual = 3.;
    info.res_primal = 1e-12;
    info.res_dual = 2e-12;
    info.res_dual_componentwise = Some(3e-12);
    info.gap_abs = 0.;
    info.gap_rel = 0.;
    info.save_prev_iterate(&variables, &mut previous);
    variables.x[0] = 7.;
    info.cost_primal = 5.;
    info.cost_dual = 6.;
    info.res_primal = 1.;
    info.res_dual = 1.;
    info.res_dual_componentwise = None;
    info.gap_abs = 1.;
    info.gap_rel = 1.;
    info.reset_to_prev_iterate(&mut variables, &previous);
    assert_eq!(variables.x, vec![2.]);
    assert_eq!((info.cost_primal, info.cost_dual), (3., 3.));
    assert_eq!((info.res_primal, info.res_dual), (1e-12, 2e-12));
    assert_eq!(info.res_dual_componentwise, Some(3e-12));
    assert_eq!((info.gap_abs, info.gap_rel), (0., 0.));
    // Residuals can still belong to a rejected iterate after rollback;
    // ordinary solved checks consume the restored Info metrics.
    let mut stale = DefaultResiduals::new(1, 1);
    stale.rx[0] = 1.;
    stale.rz[0] = 1.;
    info.status = SolverStatus::InsufficientProgress;
    info.post_process(&stale, &settings());
    assert_eq!(info.status, SolverStatus::AlmostSolved);
    info.reset(&mut Timers::default());
    assert_eq!(info.status, SolverStatus::Unsolved);
}

#[test]
fn ordinary_convergence_requires_gap_and_both_residuals() {
    let residuals = DefaultResiduals::<f64>::new(1, 1);
    let settings = settings();
    for fault in 0..3 {
        let mut info = DefaultInfo::new();
        match fault {
            0 => {
                info.gap_abs = 1.;
                info.gap_rel = 1.;
            }
            1 => info.res_primal = 1.,
            _ => info.res_dual = 1.,
        }
        info.check_convergence_full(&residuals, &settings);
        assert_eq!(info.status, SolverStatus::Unsolved);
        info.status = SolverStatus::MaxIterations;
        info.post_process(&residuals, &settings);
        assert_eq!(info.status, SolverStatus::MaxIterations);
    }
}

#[test]
fn componentwise_dual_gate_is_opt_in_and_full_only() {
    let mut residuals = DefaultResiduals::<f64>::new(1, 1);
    residuals.dual_componentwise = Some(1e-12);
    let mut info = DefaultInfo::new();
    info.gap_abs = 0.;
    info.gap_rel = 0.;
    info.res_primal = 0.;
    info.res_dual = 0.;
    info.res_dual_componentwise = residuals.dual_componentwise;
    info.ktratio = 0.;

    // Existing global convergence remains unchanged when the option is off.
    let mut settings = settings::<f64>();
    info.check_convergence_full(&residuals, &settings);
    assert_eq!(info.status, SolverStatus::Solved);

    // Opting in rejects the same globally-converged point when the
    // componentwise residual exceeds the requested tolerance.
    info.status = SolverStatus::Unsolved;
    settings.tol_feas_componentwise = Some(1e-30);
    info.check_convergence_full(&residuals, &settings);
    assert_eq!(info.status, SolverStatus::Unsolved);

    // Reduced AlmostSolved remains a reduced global status and does not gain
    // full-credit componentwise semantics.
    info.status = SolverStatus::MaxIterations;
    info.post_process(&residuals, &settings);
    assert_eq!(info.status, SolverStatus::AlmostSolved);
}

fn componentwise_metric_tracks_hsd_and_ruiz<T: FloatT>() {
    let two = T::from_i32(2).unwrap();
    let q = two.powi(4);
    let a_value = T::one() / q;
    let p = CscMatrix::zeros((1, 1));
    let a = CscMatrix::from(&[[a_value]]);
    let b = [T::zero()];
    let cones = [SupportedConeT::NonnegativeConeT(1)];
    let physical_x = two.powi(80);
    let physical_s = -a_value * physical_x;
    // Keep the cancellation remainder well above the f64 test helper's
    // absolute floor while the large x norm still drives the global metric
    // below 1e-30.
    let physical_z = -(q / a_value) + two.powi(-20);

    let mut no_ruiz = settings::<T>();
    no_ruiz.equilibrate_enable = false;
    let mut data = DefaultProblemData::new(&p, &[q], &a, &b, &cones, &no_ruiz);
    data.componentwise_enabled = true;
    let variables = DefaultVariables {
        x: vec![physical_x],
        s: vec![physical_s],
        z: vec![physical_z],
        τ: T::one(),
        κ: T::zero(),
    };
    let mut residuals = DefaultResiduals::new(1, 1);
    residuals.update(&variables, &data);
    let metric = residuals.dual_componentwise.unwrap();
    assert!(metric > T::from_f64(1e-30).unwrap());
    let mut info = DefaultInfo::new();
    info.update(&mut data, &variables, &residuals, &Timers::default());
    assert!(info.res_dual < T::from_f64(1e-30).unwrap());
    assert!(info.res_dual_componentwise.unwrap() > T::from_f64(1e-30).unwrap());

    // Positive HSD scaling changes every dual-equation term by the same tau
    // factor; the work-relative metric must therefore remain unchanged.
    let doubled = DefaultVariables {
        x: vec![two * physical_x],
        s: vec![two * physical_s],
        z: vec![two * physical_z],
        τ: two,
        κ: T::zero(),
    };
    let mut doubled_residuals = DefaultResiduals::new(1, 1);
    doubled_residuals.update(&doubled, &data);
    close(doubled_residuals.dual_componentwise.unwrap(), metric);

    // Ruiz data coordinates use the same positive diagonal map as solution
    // recovery. The ratio is invariant when the physical point is mapped into
    // those coordinates.
    let mut ruiz = settings::<T>();
    ruiz.equilibrate_enable = true;
    let mut scaled = DefaultProblemData::new(&p, &[q], &a, &b, &cones, &ruiz);
    scaled.equilibrate(&CompositeCone::new(&cones), &ruiz);
    scaled.componentwise_enabled = true;
    let eq = &scaled.equilibration;
    let scaled_variables = DefaultVariables {
        x: vec![physical_x * eq.dinv[0]],
        s: vec![physical_s * eq.e[0]],
        z: vec![physical_z * eq.c * eq.einv[0]],
        τ: T::one(),
        κ: T::zero(),
    };
    let mut scaled_residuals = DefaultResiduals::new(1, 1);
    scaled_residuals.update(&scaled_variables, &scaled);
    close(scaled_residuals.dual_componentwise.unwrap(), metric);
}

#[test]
fn componentwise_metric_hsd_ruiz_f64() {
    componentwise_metric_tracks_hsd_and_ruiz::<f64>();
}

#[test]
fn componentwise_metric_hsd_ruiz_mpfr256() {
    componentwise_metric_tracks_hsd_and_ruiz::<Bits256>();
}

#[test]
fn componentwise_metric_zero_and_nonfinite_rows_are_conservative() {
    let settings = settings::<f64>();
    let cones = [SupportedConeT::NonnegativeConeT(1)];
    let p = CscMatrix::zeros((1, 1));
    // Keep a structural zero so the nonfinite z path exercises an actual
    // operator product (0 * infinity -> NaN), which the metric must reject.
    let a = CscMatrix::new(1, 1, vec![0, 1], vec![0], vec![0.]);
    let mut data = DefaultProblemData::new(&p, &[0.], &a, &[0.], &cones, &settings);
    data.componentwise_enabled = true;

    let finite = DefaultVariables {
        x: vec![0.],
        s: vec![0.],
        z: vec![0.],
        τ: 1.,
        κ: 0.,
    };
    let mut residuals = DefaultResiduals::new(1, 1);
    residuals.update(&finite, &data);
    assert_eq!(residuals.dual_componentwise, Some(0.));

    let nonfinite = DefaultVariables {
        z: vec![f64::INFINITY],
        ..finite
    };
    residuals.update(&nonfinite, &data);
    assert!(residuals.dual_componentwise.unwrap().is_infinite());
}
