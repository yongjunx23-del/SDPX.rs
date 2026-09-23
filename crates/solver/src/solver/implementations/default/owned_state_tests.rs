use super::*;
use crate::solver::core::{
    traits::{Info, Residuals},
    SolverStatus,
};
use crate::solver::implementations::default::owned_hsd::*;
use crate::timers::Timers;
use sdpx_arithmetic::MpFloat;

// Prepare already-equilibrated fixtures without repeating preprocessing or Ruiz.
pub(super) fn runtime<T: FloatT>(
    data: DefaultProblemData<T>,
    count: usize,
    mut settings: DefaultSettings<T>,
    original: (usize, usize),
) -> OwnedSolver<T> {
    settings.max_threads = 1;
    let cones = CompositeCone::new(&data.cones);
    OwnedSolver::from_prepared(
        PreparedProblem {
            data,
            cones,
            settings,
            solution: DefaultSolution::new(original.0, original.1),
            timers: Timers::default(),
            cost_input_fingerprint: None,
        },
        count,
    )
    .unwrap()
}
// Test-only scatter of a manufactured internal point, with shared slack counted once.
pub(super) fn scatter<T: FloatT>(variables: &mut OwnedVariables<T>, point: &DefaultVariables<T>) {
    let layout = &variables.layout;
    assert_eq!(
        (point.x.len(), point.s.len(), point.z.len()),
        (layout.n, layout.m, layout.m)
    );
    for (rank, (block, ids)) in variables.blocks.iter_mut().zip(&layout.owners).enumerate() {
        for (v, &i) in block.x.iter_mut().zip(&ids.columns) {
            *v = point.x[i];
        }
        for (j, &i) in ids.rows.iter().enumerate() {
            block.z[j] = point.z[i];
            block.s[j] = if rank != 0 && layout.border_rows.binary_search(&i).is_ok() {
                T::zero()
            } else {
                point.s[i]
            };
        }
        block.τ = point.τ;
        block.κ = point.κ;
    }
    // The owned representation keeps homogeneous scalars and replicated
    // equality coordinates in canonical fields.  Test points are intentionally
    // scattered without synchronising local replicas so the existing counted
    // row assertions continue to exercise the one-owner accounting rule.
    variables.tau = point.τ;
    variables.kappa = point.κ;
    for (j, &row) in layout.border_rows.iter().enumerate() {
        variables.border_s[j] = point.s[row];
        variables.border_z[j] = point.z[row];
    }
}

fn n<T: FloatT>(x: i32) -> T {
    T::from_i32(x).unwrap()
}
fn close<T: FloatT>(a: T, b: T) {
    if b.is_nan() {
        assert!(a.is_nan());
    } else if b.is_infinite() {
        assert_eq!(a, b);
    } else {
        assert!(
            (a - b).abs() <= n::<T>(4096) * T::epsilon() * b.abs().max(T::one()),
            "{a} != {b}"
        );
    }
}
fn settings<T: FloatT>() -> DefaultSettings<T> {
    DefaultSettings {
        verbose: false,
        presolve_enable: false,
        #[cfg(feature = "sdp")]
        chordal_decomposition_enable: false,
        input_sparse_dropzeros: false,
        equilibrate_enable: true,
        tol_feas_componentwise: Some(T::epsilon().sqrt()),
        ..DefaultSettings::default()
    }
}
pub(super) fn ordinary<T: FloatT>() -> DefaultProblemData<T> {
    let a = [
        [1, 1, 1, 1, 1, 1],
        [1, 0, 2, 0, 0, 0],
        [0, 0, -1, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, -1, 0, 1, 0, 0],
        [2, -1, 3, -2, 1, 4],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0],
    ]
    .map(|r| r.map(n::<T>));
    let mut p = [[T::zero(); 6]; 6];
    for i in 0..6 {
        p[i][i] = n(i as i32 + 2);
    }
    p[0][2] = T::one() / n(4);
    p[1][3] = -T::one() / n(8);
    let kinds = vec![
        SupportedConeT::ZeroConeT(1),
        SupportedConeT::NonnegativeConeT(2),
        SupportedConeT::SecondOrderConeT(3),
        SupportedConeT::ZeroConeT(1),
        SupportedConeT::NonnegativeConeT(3),
    ];
    let settings = settings();
    let mut d = DefaultProblemData::new(
        &CscMatrix::from(&p),
        &[1, -2, 3, -4, 5, -6].map(n),
        &CscMatrix::from(&a),
        &[1, 0, 2, 4, -1, 2, 3, 6, 7, 8].map(n),
        &kinds,
        &settings,
    );
    d.equilibrate(&CompositeCone::new(&d.cones), &settings);
    d
}

#[cfg(feature = "sdp")]
pub(super) fn sampled<T: FloatT>() -> DefaultProblemData<T> {
    let mut ptr = vec![0];
    let mut rows = Vec::new();
    let mut values = Vec::new();
    for col in 0..6 {
        rows.extend([0, 13]);
        values.extend([n::<T>(col + 1), -n::<T>(col + 2)]);
        ptr.push(rows.len());
    }
    let linear = CscMatrix::new(14, 6, ptr, rows, values);
    let blocks = [0, 0, 2, 4]
        .into_iter()
        .enumerate()
        .map(|(i, column)| SampledBlock {
            row_start: 1 + i * 3,
            column_start: column,
            dim: 1,
            basis_rows: 2,
            basis_cols: 2,
            basis: vec![T::one(), T::one() / n(2), T::one() / n(4), T::one()],
            weights: vec![n::<T>(i as i32 + 1) / n(4), -T::one() / n(2)],
        })
        .collect();
    let operator = SampledOperator::new(linear, blocks).unwrap();
    let kinds = vec![
        SupportedConeT::ZeroConeT(1),
        SupportedConeT::PSDTriangleConeT(2),
        SupportedConeT::PSDTriangleConeT(2),
        SupportedConeT::PSDTriangleConeT(2),
        SupportedConeT::PSDTriangleConeT(2),
        SupportedConeT::ZeroConeT(1),
    ];
    let mut p = [[T::zero(); 6]; 6];
    for i in 0..6 {
        p[i][i] = n(i as i32 + 1);
    }
    p[2][4] = T::one() / n(8); // Cross-factor P edge must merge their owners.
    let settings = settings();
    let mut data = DefaultProblemData::new(
        &CscMatrix::from(&p),
        &[1, -2, 3, -4, 5, -6].map(n),
        &operator.materialize(),
        &vec![T::one(); 14],
        &kinds,
        &settings,
    );
    data.equilibrate(&CompositeCone::new(&data.cones), &settings);
    data.install_sampled(operator);
    assert!(data.sampled.is_some());
    data
}

fn compare_info<T: FloatT>(a: &DefaultInfo<T>, b: &DefaultInfo<T>) {
    for (a, b) in [
        a.cost_primal,
        a.cost_dual,
        a.res_primal,
        a.res_dual,
        a.res_primal_inf,
        a.res_dual_inf,
        a.gap_abs,
        a.gap_rel,
        a.ktratio,
    ]
    .into_iter()
    .zip([
        b.cost_primal,
        b.cost_dual,
        b.res_primal,
        b.res_dual,
        b.res_primal_inf,
        b.res_dual_inf,
        b.gap_abs,
        b.gap_rel,
        b.ktratio,
    ]) {
        close(a, b);
    }
    match (a.res_dual_componentwise, b.res_dual_componentwise) {
        (Some(a), Some(b)) => close(a, b),
        (None, None) => (),
        _ => panic!("componentwise setting changed"),
    }
}

fn check<T: FloatT>(make: impl Fn() -> DefaultProblemData<T>) {
    for count in [1, 2, 8] {
        let mut reference = make();
        let input = make();
        #[cfg(feature = "sdp")]
        let weak = input.sampled.as_ref().map(Arc::downgrade);
        let mut owned = runtime(input, count, settings(), (reference.n, reference.m));
        #[cfg(feature = "sdp")]
        if let Some(weak) = weak {
            assert!(weak.upgrade().is_none(), "retained global sampled operator");
        }
        assert_eq!(
            owned.data.blocks.iter().map(|o| o.P.nnz()).sum::<usize>(),
            reference.P.nnz()
        );
        assert_eq!(
            owned.data.blocks.iter().map(|o| o.A.nnz()).sum::<usize>(),
            reference.A.nnz()
        );
        assert_eq!(
            owned
                .variables
                .blocks
                .iter()
                .map(|o| o.x.len())
                .sum::<usize>(),
            reference.n
        );
        assert_eq!(
            owned
                .variables
                .blocks
                .iter()
                .map(|o| o.z.len())
                .sum::<usize>(),
            reference.m + (count - 1) * owned.data.layout.border_rows.len()
        );
        if count > 1 {
            assert!(owned
                .variables
                .blocks
                .iter()
                .all(|o| o.x.len() < reference.n));
            assert!(owned
                .variables
                .blocks
                .iter()
                .all(|o| o.s.len() < reference.m));
        }
        if count == 8 {
            assert!(owned.variables.blocks.iter().any(|o| o.x.is_empty()));
        }
        #[cfg(feature = "sdp")]
        if let Some(operator) = &reference.sampled {
            assert_eq!(
                owned
                    .data
                    .blocks
                    .iter()
                    .flat_map(|o| o.sampled.as_ref().unwrap().blocks())
                    .map(|b| b.basis.len() + b.weights.len())
                    .sum::<usize>(),
                operator
                    .blocks()
                    .iter()
                    .map(|b| b.basis.len() + b.weights.len())
                    .sum::<usize>()
            );
        }
        let mut point = DefaultVariables::new(reference.n, reference.m);
        let mut residual = DefaultResiduals::new(reference.n, reference.m);
        let mut reference_info = DefaultInfo::new();
        let pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(2)
                .build()
                .unwrap(),
        );
        let pointers: Vec<_> = owned
            .variables
            .blocks
            .iter()
            .zip(&owned.residuals.blocks)
            .map(|(v, r)| {
                (
                    v.x.as_ptr(),
                    r.Px.as_ptr(),
                    r.rx_inf.as_ptr(),
                    r.rz_inf.as_ptr(),
                )
            })
            .collect();
        for turn in 0..3 {
            for (i, x) in point.x.iter_mut().enumerate() {
                *x = n::<T>((i + 1 + turn) as i32) / n(8);
            }
            for (i, x) in point.s.iter_mut().enumerate() {
                *x = n::<T>((i + 2 + turn) as i32) / n(16);
            }
            for (i, x) in point.z.iter_mut().enumerate() {
                *x = -n::<T>((i + 3 + turn) as i32) / n(32);
            }
            point.τ = n(2);
            point.κ = T::one() / n(4);
            scatter(&mut owned.variables, &point);
            residual.update(&point, &reference);
            reference_info.update(&mut reference, &point, &residual, &Timers::default());
            owned.residuals.update_with_pool(
                &owned.variables,
                &owned.data,
                if turn == 1 {
                    Some(Arc::clone(&pool))
                } else {
                    None
                },
            );
            let summary = owned.residuals.summary.as_ref().unwrap();
            let mut info = OwnedInfo(DefaultInfo::new());
            info.update(
                &mut owned.data,
                &owned.variables,
                &owned.residuals,
                &Timers::default(),
            );
            compare_info(&info.0, &reference_info);
            for (a, b) in [
                summary.products.qx,
                summary.products.bz,
                summary.products.sz,
                summary.products.xpx,
            ]
            .into_iter()
            .zip([
                residual.products.qx,
                residual.products.bz,
                residual.products.sz,
                residual.products.xpx,
            ]) {
                close(a, b);
            }
            close(owned.residuals.scalar.rτ, residual.rτ);
            for (i, ids) in owned.data.layout.owners.iter().enumerate() {
                let owner = &owned.residuals.blocks[i];
                assert_eq!(owned.cones.blocks[i].numel(), owned.data.blocks[i].m);
                for (local, &global) in ids.columns.iter().enumerate() {
                    for (a, b) in [
                        (owner.rx[local], residual.rx[global]),
                        (owner.rx_inf[local], residual.rx_inf[global]),
                        (owner.Px[local], residual.Px[global]),
                    ] {
                        close(a, b);
                        if count == 1 {
                            assert_eq!(a, b);
                        }
                    }
                }
                for (local, &global) in ids.rows.iter().enumerate() {
                    close(owner.rz_inf[local], residual.rz_inf[global]);
                    close(owner.rz[local], residual.rz[global]);
                }
            }
            info.check_termination(&owned.residuals, &settings(), 0);
            reference_info.status = SolverStatus::Unsolved;
            reference_info.check_termination(&residual, &settings(), 0);
            assert_eq!(info.0.status, reference_info.status);
        }
        let after: Vec<_> = owned
            .variables
            .blocks
            .iter()
            .zip(&owned.residuals.blocks)
            .map(|(v, r)| {
                (
                    v.x.as_ptr(),
                    r.Px.as_ptr(),
                    r.rx_inf.as_ptr(),
                    r.rz_inf.as_ptr(),
                )
            })
            .collect();
        assert_eq!(pointers, after);
    }
}

#[test]
fn owned_state_f64() {
    check(ordinary::<f64>);
}
#[test]
fn owned_state_mpfr256() {
    check(ordinary::<MpFloat<4>>);
}
#[test]
fn owned_state_mpfr512() {
    check(ordinary::<MpFloat<8>>);
}
#[test]
fn owned_state_mpfr768() {
    check(ordinary::<MpFloat<12>>);
}
#[cfg(feature = "sdp")]
#[test]
fn owned_sampled_state_f64() {
    check(sampled::<f64>);
}
#[cfg(feature = "sdp")]
#[test]
fn owned_sampled_state_mpfr256() {
    check(sampled::<MpFloat<4>>);
}
#[cfg(feature = "sdp")]
#[test]
fn owned_sampled_state_mpfr512() {
    check(sampled::<MpFloat<8>>);
}
#[cfg(feature = "sdp")]
#[test]
fn owned_sampled_state_mpfr768() {
    check(sampled::<MpFloat<12>>);
}

#[test]
fn owned_border_applies_affine_term_after_sum() {
    // Splitting (large + 1) - large by first subtracting b loses the one.
    let p = CscMatrix::<f64>::zeros((2, 2));
    let a = CscMatrix::from(&[[1., 1.], [1., 0.], [0., 1.]]);
    let kinds = [
        SupportedConeT::ZeroConeT(1),
        SupportedConeT::NonnegativeConeT(2),
    ];
    let mut se = settings();
    se.equilibrate_enable = false;
    let data = DefaultProblemData::new(&p, &[0., 0.], &a, &[1e16, 0., 0.], &kinds, &se);
    let point = DefaultVariables {
        x: vec![1e16, 1.],
        s: vec![0.; 3],
        z: vec![0.; 3],
        τ: 1.,
        κ: 0.,
    };
    let mut owned = runtime(data, 2, settings(), (point.x.len(), point.s.len()));
    scatter(&mut owned.variables, &point);
    owned.residuals.update(&owned.variables, &owned.data);
    let summary = owned.residuals.summary.as_ref().unwrap();
    for (r, ids) in owned.residuals.blocks.iter().zip(&owned.data.layout.owners) {
        let row = ids.rows.binary_search(&0).unwrap();
        // Same rounding as the original: (1e16 + 1) rounds, then subtract b.
        assert_eq!(r.rz[row], 0.);
    }
    assert!(summary.norms().iter().all(|v| v.is_finite()));
}

#[test]
fn owned_counted_products_skip_nonfinite_replicated_rows() {
    let mut data = ordinary::<f64>();
    // Disable the optional work scan here to isolate counted products.
    data.componentwise_enabled = false;
    let dims = (data.n, data.m);
    let mut owned = runtime(data, 2, settings(), dims);
    let mut point = DefaultVariables::new(owned.data.layout.n, owned.data.layout.m);
    point.z.fill(2.);
    point.s.fill(3.);
    point.x.fill(1.);
    point.z[owned.data.layout.border_rows[0]] = f64::INFINITY;
    scatter(&mut owned.variables, &point);
    owned.residuals.update(&owned.variables, &owned.data);
    // Owner 0 counts the nonfinite global boundary. Owner 1 must not multiply
    // its zero b/s boundary replicas by infinity while forming local products.
    let ids = &owned.data.layout.owners[1];
    assert!(!ids.counted_rows.is_empty());
    let data = &owned.data.blocks[1];
    let variables = &owned.variables.blocks[1];
    let owner = &owned.residuals.blocks[1];
    let bz = ids
        .counted_rows
        .iter()
        .fold(0.0, |sum, &i| data.b[i].mul_add(variables.z[i], sum));
    let sz = ids
        .counted_rows
        .iter()
        .fold(0.0, |sum, &i| variables.s[i].mul_add(variables.z[i], sum));
    assert!(bz.is_finite() && sz.is_finite());
    assert_eq!(owner.products.bz, bz);
    assert_eq!(owner.products.sz, sz);
}
