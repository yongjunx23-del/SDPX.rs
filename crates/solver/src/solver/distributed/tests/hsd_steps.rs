use super::*;
use crate::solver::core::traits::Residuals;
use sdpx_arithmetic::MpFloat;
fn n<T: FloatT>(i: i32) -> T {
    T::from_i32(i).unwrap()
}
fn near<T: FloatT>(a: T, b: T) {
    if b.is_nan() {
        assert!(a.is_nan());
    } else {
        assert!(
            (a - b).abs() <= n::<T>(4096) * T::epsilon() * b.abs().max(T::one()),
            "{a} != {b}"
        );
    }
}
fn from_full<T: FloatT>(
    layout: &Arc<OwnerLayout>,
    full: &DefaultVariables<T>,
) -> OwnedVariables<T> {
    let mut out = OwnedVariables {
        layout: layout.clone(),
        owner_ids: (0..layout.owners.len()).collect(),
        all_owner_ids: (0..layout.owners.len()).collect(),
        collective: Arc::new(crate::solver::distributed::collective::SerialCollective),
        tau: full.τ,
        kappa: full.κ,
        border_z: vec![T::zero(); layout.border_rows.len()],
        border_s: vec![T::zero(); layout.border_rows.len()],
        blocks: layout
            .owners
            .iter()
            .map(|ids| DefaultVariables::new(ids.columns.len(), ids.rows.len()))
            .collect(),
    };
    for (v, ids) in out.blocks.iter_mut().zip(&layout.owners) {
        for (x, &i) in v.x.iter_mut().zip(&ids.columns) {
            *x = full.x[i];
        }
        for (j, &i) in ids.rows.iter().enumerate() {
            v.z[j] = full.z[i];
            v.s[j] = if ids.counted_rows.contains(&j) {
                full.s[i]
            } else {
                T::zero()
            };
        }
        v.τ = full.τ;
        v.κ = full.κ;
    }
    // Keep the shared scalar/equality coordinates alongside the local block
    // scatter.  Do not call sync_border here: the fixture deliberately keeps
    // replicated equality slacks zero on non-counting owners.
    for (j, &row) in layout.border_rows.iter().enumerate() {
        out.border_s[j] = full.s[row];
        out.border_z[j] = full.z[row];
    }
    out
}
fn compare<T: FloatT>(owned: &OwnedVariables<T>, full: &DefaultVariables<T>) {
    for (v, ids) in owned.blocks.iter().zip(&owned.layout.owners) {
        for (&x, &i) in v.x.iter().zip(&ids.columns) {
            near(x, full.x[i]);
        }
        for (j, &i) in ids.rows.iter().enumerate() {
            near(v.z[j], full.z[i]);
            near(
                v.s[j],
                if ids.counted_rows.contains(&j) {
                    full.s[i]
                } else {
                    T::zero()
                },
            );
        }
        near(v.τ, full.τ);
        near(v.κ, full.κ);
    }
}
fn stages<T: FloatT>(mixed: bool) {
    let mut kinds = vec![
        SupportedConeT::ZeroConeT(1),
        SupportedConeT::NonnegativeConeT(2),
        SupportedConeT::SecondOrderConeT(3),
    ];
    if mixed {
        kinds.push(SupportedConeT::ExponentialConeT());
    }
    #[cfg(feature = "sdp")]
    kinds.push(SupportedConeT::PSDTriangleConeT(2));
    let m = kinds.iter().map(|c| c.nvars()).sum();
    let settings = DefaultSettings {
        max_threads: 1,
        presolve_enable: false,
        equilibrate_enable: false,
        #[cfg(feature = "sdp")]
        chordal_decomposition_enable: false,
        ..DefaultSettings::default()
    };
    let data = DefaultProblemData::new(
        &CscMatrix::identity(2),
        &[n::<T>(1), n(2)],
        &CscMatrix::new(m, 2, vec![0, 2, 4], vec![0, 1, 0, 2], vec![n(1); 4]),
        &vec![n(1); m],
        &kinds,
        &settings,
    );
    let prepared = DefaultSolver::new(
        &data.P,
        &data.q,
        &data.A,
        &data.b,
        &data.cones,
        settings.clone(),
    )
    .unwrap();
    let runtime = OwnedSolver::from_default(prepared, 12).unwrap();
    let owned_data = runtime.data;
    let mut owned_residuals = runtime.residuals;
    let mut cones = runtime.cones;
    let layout = Arc::clone(&owned_data.layout);
    let mut original = CompositeCone::new(&data.cones);
    let allocated = OwnedVariables::new(&owned_data);
    assert_eq!(allocated.blocks.len(), 12);
    assert!(allocated
        .blocks
        .iter()
        .any(|v| v.x.is_empty() && v.s.len() == 1));
    assert_eq!(cones.all_symmetric(), original.is_symmetric());
    assert_eq!(
        cones.supports_primal_dual(),
        original.allows_primal_dual_scaling()
    );
    assert!(cones.worker_pool().is_none());
    if !mixed {
        cones.reset_scaling();
        original.set_identity_scaling();
    }
    let mut full = DefaultVariables::new(2, m);
    let mut owned = from_full(&layout, &full);
    if !mixed {
        for (i, x) in full.s.iter_mut().enumerate() {
            *x = -n::<T>(i as i32 + 1);
        }
        full.z.fill(n(2));
        owned = from_full(&layout, &full);
        full.symmetric_initialization(&mut original);
        owned.symmetric_initialization(&mut cones);
        compare(&owned, &full);
    }
    full.unit_initialization(&original);
    owned.unit_initialization(&cones);
    compare(&owned, &full);
    #[cfg(feature = "sdp")]
    {
        full.s[m - 3] = n(2);
        full.s[m - 2] = T::one() / n(8);
        full.s[m - 1] = n(3);
        full.z[m - 3] = n(4);
        full.z[m - 2] = -T::one() / n(16);
        full.z[m - 1] = n(2);
    }
    full.z[0] = T::one() / n(8);
    full.x[0] = T::one() / n(4);
    full.τ = n(2);
    full.κ = n(3);
    owned = from_full(&layout, &full);
    let mut residuals = DefaultResiduals::new(2, m);
    residuals.update(&full, &data);
    owned_residuals.update(&owned, &owned_data);
    let mu = full.calc_mu(&residuals, &original);
    near(owned.calc_mu(&owned_residuals, &cones), mu);
    assert!(full.scale_cones(&mut original, mu, ScalingStrategy::PrimalDual));
    assert!(owned.scale_cones(&mut cones, mu, ScalingStrategy::PrimalDual));
    let mut rhs = full.new_like();
    rhs.affine_step_rhs(&residuals, &full, &original);
    let mut local_rhs = owned.new_like();
    local_rhs.affine_step_rhs(&owned_residuals, &owned, &cones);
    compare(&local_rhs, &rhs);
    let mut step = full.new_like();
    step.τ = -T::one() / n(8);
    step.κ = T::one() / n(16);
    step.x[0] = n(1);
    step.x[1] = -n::<T>(2);
    for i in 0..m {
        step.s[i] = -full.s[i] / n(32);
        step.z[i] = full.z[i] / n(64);
    }
    let mut direction = from_full(&layout, &step);
    let mut ordinary_rhs = rhs.new_like();
    ordinary_rhs.copy_from(&rhs);
    let mut ordinary_step = step.new_like();
    ordinary_step.copy_from(&step);
    let mut ordinary_local_rhs = local_rhs.new_like();
    ordinary_local_rhs.copy_from(&local_rhs);
    let mut ordinary_local_step = direction.new_like();
    ordinary_local_step.copy_from(&direction);
    ordinary_rhs.combined_step_rhs(
        &residuals,
        &full,
        &mut original,
        &mut ordinary_step,
        T::one() / n(4),
        mu,
        T::one() / n(2),
    );
    ordinary_local_rhs.combined_step_rhs(
        &owned_residuals,
        &owned,
        &mut cones,
        &mut ordinary_local_step,
        T::one() / n(4),
        mu,
        T::one() / n(2),
    );
    compare(&ordinary_local_rhs, &ordinary_rhs);

    near(
        owned.calc_step_length(&direction, &mut cones, &settings, StepDirection::Combined),
        full.calc_step_length(&step, &mut original, &settings, StepDirection::Combined),
    );
    near(
        owned.barrier(&direction, T::one() / n(8), &mut cones),
        full.barrier(&step, T::one() / n(8), &mut original),
    );
    let mut backup = owned.new_like();
    backup.copy_from(&owned);
    let mut full_backup = full.new_like();
    full_backup.copy_from(&full);
    for _ in 0..2 {
        owned.add_step(&direction, T::one() / n(8));
        full.add_step(&step, T::one() / n(8));
        compare(&owned, &full);
        compare(&backup, &full_backup);
    }
    let mut interpolated = owned.new_like();
    let mut full_interpolated = full.new_like();
    interpolated.interpolate(&backup, &owned, T::one() / n(4));
    full_interpolated.interpolate(&full_backup, &full, T::one() / n(4));
    compare(&interpolated, &full_interpolated);
    interpolated.rescale();
    full_interpolated.rescale();
    compare(&interpolated, &full_interpolated);
    owned.copy_from(&backup);
    full.copy_from(&full_backup);
    compare(&owned, &full);
    near(
        owned.prepare_affine_step_length(&mut direction, &mut cones, &settings),
        full.prepare_affine_step_length(&mut step, &mut original, &settings),
    );
    compare(&direction, &step);
    rhs.combined_step_rhs_prepared(
        &residuals,
        &full,
        &mut original,
        &mut step,
        T::one() / n(4),
        mu,
    );
    local_rhs.combined_step_rhs_prepared(
        &owned_residuals,
        &owned,
        &mut cones,
        &mut direction,
        T::one() / n(4),
        mu,
    );
    compare(&local_rhs, &rhs);
}
fn all<T: FloatT>() {
    for mixed in [false, true] {
        stages::<T>(mixed);
    }
}
#[test]
fn owned_hsd_steps_f64() {
    all::<f64>();
}
#[test]
fn owned_hsd_steps_mpfr256() {
    all::<MpFloat<4>>();
}
#[test]
fn owned_hsd_steps_mpfr512() {
    all::<MpFloat<8>>();
}
