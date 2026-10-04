use super::*;
use crate::solver::cones::SupportedCone;
use crate::solver::core::traits::KKTSystem;
use crate::solver::{IPSolver, SolverStatus};
use sdpx_arithmetic::MpFloat;

fn n<T: FloatT>(value: i32) -> T {
    T::from_i32(value).unwrap()
}

fn settings<T: FloatT>(workers: usize) -> DefaultSettings<T> {
    let tolerance = if T::precision_bits() <= 53 {
        T::from_f64(1e-9).unwrap()
    } else {
        T::from_f64(1e-35).unwrap()
    };
    DefaultSettings {
        verbose: false,
        max_iter: 100,
        max_threads: workers as u32,
        presolve_enable: false,
        chordal_decomposition_enable: false,
        tol_feas: tolerance,
        tol_gap_abs: tolerance,
        tol_gap_rel: tolerance,
        tol_feas_componentwise: Some(tolerance),
        ..DefaultSettings::default()
    }
}

/// A bounded, asymmetric product-cone problem.  A positive definite P makes
/// the x block invertible even though the constraint operator is deliberately
/// sparse (and includes a shared zero-cone border row).
fn mixed_inputs<T: FloatT>(
    workers: usize,
) -> (
    CscMatrix<T>,
    Vec<T>,
    CscMatrix<T>,
    Vec<T>,
    Vec<SupportedConeT<T>>,
    DefaultSettings<T>,
) {
    let cones = vec![
        SupportedConeT::ZeroConeT(1),
        SupportedConeT::NonnegativeConeT(2),
        SupportedConeT::ExponentialConeT(),
        SupportedConeT::SecondOrderConeT(3),
        SupportedConeT::PSDTriangleConeT(2),
    ];
    let b = vec![
        T::zero(),
        n(1),
        n(2),
        T::zero(),
        T::one(),
        n(2),
        n(3),
        T::zero(),
        T::zero(),
        T::one(),
        T::zero(),
        T::one(),
    ];
    let p = CscMatrix::identity(2);
    let q = vec![n(1), n(-2)];
    let a = CscMatrix::zeros((b.len(), 2));
    (p, q, a, b, cones, settings(workers))
}

fn mixed_prepared<T: FloatT>(workers: usize) -> PreparedProblem<T> {
    let (p, q, a, b, cones, set) = mixed_inputs(workers);
    PreparedProblem::new(&p, &q, &a, &b, &cones, set).unwrap()
}

fn mixed_problem<T: FloatT>(workers: usize) -> DefaultSolver<T> {
    DefaultSolver::from_prepared(mixed_prepared(workers)).unwrap()
}

/// The equality row couples the owner-local interiors through the border
/// Schur complement.  The second owner has no columns, exercising the empty
/// owner path while the border itself remains globally invertible.
fn boundary_prepared<T: FloatT>(workers: usize) -> PreparedProblem<T> {
    let mut set = settings::<T>(workers);
    set.static_regularization_enable = false;
    let p = CscMatrix::identity(1);
    let q = [T::zero()];
    let a = CscMatrix::from(&[[T::one()], [T::zero()]]);
    let b = [T::zero(), T::one()];
    let cones = [
        SupportedConeT::ZeroConeT(1),
        SupportedConeT::NonnegativeConeT(1),
    ];
    PreparedProblem::new(&p, &q, &a, &b, &cones, set).unwrap()
}

fn boundary_problem<T: FloatT>(workers: usize) -> DefaultSolver<T> {
    DefaultSolver::from_prepared(boundary_prepared(workers)).unwrap()
}

fn value_close<T: FloatT>(actual: T, expected: T) {
    if !actual.is_finite() || !expected.is_finite() {
        assert_eq!(actual.is_nan(), expected.is_nan(), "{actual} != {expected}");
        assert_eq!(
            actual.is_sign_negative(),
            expected.is_sign_negative(),
            "{actual} != {expected}"
        );
        return;
    }
    let base = if T::precision_bits() <= 53 {
        T::from_f64(1e-7).unwrap()
    } else {
        T::from_f64(1e-30).unwrap()
    };
    let scale = T::one() + actual.abs().max(expected.abs());
    assert!(
        (actual - expected).abs() <= base * scale,
        "{actual} != {expected}"
    );
}

fn solution_close<T: FloatT>(actual: &DefaultSolution<T>, expected: &DefaultSolution<T>) {
    assert_eq!(actual.status, expected.status);
    assert_eq!(actual.x.len(), expected.x.len());
    assert_eq!(actual.s.len(), expected.s.len());
    assert_eq!(actual.z.len(), expected.z.len());
    for (&a, &b) in actual.x.iter().zip(&expected.x) {
        value_close(a, b);
    }
    for (&a, &b) in actual.s.iter().zip(&expected.s) {
        value_close(a, b);
    }
    for (&a, &b) in actual.z.iter().zip(&expected.z) {
        value_close(a, b);
    }
    value_close(actual.obj_val, expected.obj_val);
    value_close(actual.obj_val_dual, expected.obj_val_dual);
    value_close(actual.r_prim, expected.r_prim);
    value_close(actual.r_dual, expected.r_dual);
}

fn pool_widths<T: FloatT>() {
    let mut reference = mixed_problem::<T>(1);
    reference.solve();
    assert_eq!(reference.solution.status, SolverStatus::Solved);

    for workers in [1, 2, 4] {
        let mut owned = OwnedSolver::from_prepared(mixed_prepared::<T>(workers), 6).unwrap();
        assert_eq!(
            owned
                .cones
                .pool
                .as_ref()
                .map_or(1, |pool| pool.current_num_threads()),
            workers
        );
        assert!(owned
            .cones
            .blocks
            .iter()
            .all(|block| block.thread_pool().is_none()));
        assert!(owned.data.blocks.iter().any(|block| block.n == 0));
        owned.solve();
        assert_eq!(owned.solution.0.status, SolverStatus::Solved);
        solution_close(&owned.solution.0, &reference.solution);
    }
}

#[test]
fn owned_pool_mixed_f64() {
    pool_widths::<f64>();
}

#[test]
fn owned_pool_mixed_mpfr256() {
    pool_widths::<MpFloat<4>>();
}

#[test]
fn owned_pool_mixed_mpfr512() {
    pool_widths::<MpFloat<8>>();
}

#[test]
fn owned_pool_callback_stops_hsd() {
    let mut owned = OwnedSolver::from_prepared(mixed_prepared::<f64>(2), 4).unwrap();
    owned.set_termination_callback(|_| true);
    owned.solve();
    assert_eq!(owned.solution.0.status, SolverStatus::CallbackTerminated);
}

#[test]
fn owned_pool_static_regularization_off_boundary() {
    let mut reference = boundary_problem::<f64>(1);
    reference.solve();
    assert_eq!(reference.solution.status, SolverStatus::Solved);

    let mut owned = OwnedSolver::from_prepared(boundary_prepared::<f64>(2), 2).unwrap();
    assert_eq!(
        owned
            .cones
            .pool
            .as_ref()
            .map_or(1, |p| p.current_num_threads()),
        2
    );
    assert!(owned.data.blocks.iter().any(|block| block.n == 0));
    owned.solve();
    assert_eq!(owned.solution.0.status, SolverStatus::Solved);
    solution_close(&owned.solution.0, &reference.solution);
}

#[test]
fn owned_pool_repeat_resets_counters() {
    let mut owned = OwnedSolver::from_prepared(boundary_prepared::<f64>(2), 2).unwrap();
    assert!(owned
        .cones
        .blocks
        .iter()
        .all(|block| block.thread_pool().is_none()));
    owned.solve();
    assert_eq!(owned.solution.0.status, SolverStatus::Solved);
    let first = (
        owned.solution.0.x.clone(),
        owned.solution.0.s.clone(),
        owned.solution.0.z.clone(),
        owned.solution.0.obj_val,
        owned.solution.0.obj_val_dual,
        owned.solution.0.r_prim,
        owned.solution.0.r_dual,
    );
    let first_counters = owned.kktsystem.counters();
    assert!(first_counters.factor_attempts > 0);

    owned.solve();
    assert_eq!(owned.solution.0.status, SolverStatus::Solved);
    for (&actual, &expected) in owned.solution.0.x.iter().zip(&first.0) {
        value_close(actual, expected);
    }
    for (&actual, &expected) in owned.solution.0.s.iter().zip(&first.1) {
        value_close(actual, expected);
    }
    for (&actual, &expected) in owned.solution.0.z.iter().zip(&first.2) {
        value_close(actual, expected);
    }
    for (&actual, expected) in [
        owned.solution.0.obj_val,
        owned.solution.0.obj_val_dual,
        owned.solution.0.r_prim,
        owned.solution.0.r_dual,
    ]
    .iter()
    .zip([first.3, first.4, first.5, first.6])
    {
        value_close(actual, expected);
    }
    assert_eq!(owned.kktsystem.counters(), first_counters);

    owned.kktsystem.reset_solve();
    assert_eq!(
        owned.kktsystem.counters(),
        crate::solver::kkt::SolveCounters::default()
    );
}

fn presolved_prepared<T: FloatT>(workers: usize) -> PreparedProblem<T> {
    let mut set = settings::<T>(workers);
    set.presolve_enable = true;
    let p = CscMatrix::identity(1);
    let q = [T::zero()];
    let a = CscMatrix::from(&[[T::one()], [T::zero()]]);
    let b = [T::zero(), T::from_f64(crate::get_infinity()).unwrap()];
    let cones = [
        SupportedConeT::ZeroConeT(1),
        SupportedConeT::NonnegativeConeT(1),
    ];
    PreparedProblem::new(&p, &q, &a, &b, &cones, set).unwrap()
}

fn presolved_problem<T: FloatT>(workers: usize) -> DefaultSolver<T> {
    DefaultSolver::from_prepared(presolved_prepared(workers)).unwrap()
}

#[test]
fn owned_pool_presolve_recovery_matches_default() {
    let mut reference = presolved_problem::<f64>(1);
    assert!(reference.data.presolver.is_some());
    reference.solve();

    let mut owned = OwnedSolver::from_prepared(presolved_prepared::<f64>(2), 3).unwrap();
    assert!(owned.data.presolver.is_some());
    owned.solve();
    assert_eq!(owned.solution.0.status, reference.solution.status);
    solution_close(&owned.solution.0, &reference.solution);
    let infinity = crate::get_infinity();
    assert_eq!(owned.solution.0.s[1], infinity);
    assert_eq!(owned.solution.0.z[1], 0.0);
}

fn chordal_inputs<T: FloatT>(
    workers: usize,
) -> (
    CscMatrix<T>,
    Vec<T>,
    CscMatrix<T>,
    Vec<T>,
    Vec<SupportedConeT<T>>,
    DefaultSettings<T>,
) {
    let mut set = settings::<T>(workers);
    set.chordal_decomposition_enable = true;
    set.chordal_decomposition_merge_method = "none".into();
    set.chordal_decomposition_compact = false;
    set.chordal_decomposition_complete_dual = false;
    // Diagonal b and a zero operator give the analytic solution x=0, s=b.
    // The four diagonal entries still produce four singleton cliques, so the
    // existing PSD4 chordal threshold and original-coordinate recovery run.
    let a = CscMatrix::zeros((10, 4));
    let b = [
        n::<T>(1),
        T::zero(),
        n(1),
        T::zero(),
        T::zero(),
        n(1),
        T::zero(),
        T::zero(),
        T::zero(),
        n(1),
    ];
    let cones = [SupportedConeT::PSDTriangleConeT(4)];
    let p = CscMatrix::identity(4);
    let q = vec![T::zero(); 4];
    (p, q, a, b.to_vec(), cones.to_vec(), set)
}

fn chordal_prepared<T: FloatT>(workers: usize) -> PreparedProblem<T> {
    let (p, q, a, b, cones, mut set) = chordal_inputs(workers);
    // Exercise the ordinary default stopping criterion. In the decomposed
    // zero-objective fixture an auxiliary stationarity row has only vanishing
    // same-sign terms: its relative residual need not tend to zero. The
    // optional componentwise case is checked separately below, without
    // promoting AlmostSolved or changing any production setting.
    set.tol_feas_componentwise = None;
    PreparedProblem::new(&p, &q, &a, &b, &cones, set).unwrap()
}

fn chordal_problem<T: FloatT>(workers: usize) -> DefaultSolver<T> {
    DefaultSolver::from_prepared(chordal_prepared(workers)).unwrap()
}

#[test]
fn owned_pool_chordal_psd4_recovery_matches_default() {
    let mut reference = chordal_problem::<f64>(1);
    assert!(reference.data.chordal_info.is_some());
    reference.solve();
    assert_eq!(reference.solution.status, SolverStatus::Solved);

    let mut owned = OwnedSolver::from_prepared(chordal_prepared::<f64>(2), 4).unwrap();
    assert!(owned.data.chordal_info.is_some());
    owned.solve();
    assert_eq!(owned.solution.0.status, SolverStatus::Solved);
    solution_close(&owned.solution.0, &reference.solution);
}

#[test]
fn owned_pool_chordal_componentwise_preserves_almost_status() {
    let make = |workers| {
        let (p, q, a, b, cones, set) = chordal_inputs::<f64>(workers);
        PreparedProblem::new(&p, &q, &a, &b, &cones, set).unwrap()
    };
    let mut reference = DefaultSolver::from_prepared(make(1)).unwrap();
    reference.solve();
    assert_eq!(reference.solution.status, SolverStatus::AlmostSolved);
    let mut owned = OwnedSolver::from_prepared(make(2), 4).unwrap();
    owned.solve();
    assert_eq!(owned.solution.0.status, SolverStatus::AlmostSolved);
    solution_close(&owned.solution.0, &reference.solution);
}

fn exact_bound_value<T: FloatT>(a: T, b: T) {
    if b.is_nan() {
        assert!(a.is_nan());
    } else {
        assert_eq!(a, b);
        assert_eq!(a.is_sign_negative(), b.is_sign_negative());
    }
}

fn bounds_fixture<T: FloatT>(workers: usize, mixed: bool) -> OwnedSolver<T> {
    if mixed {
        return OwnedSolver::from_prepared(mixed_prepared(workers), 12).unwrap();
    }
    let kinds = vec![
        SupportedConeT::ZeroConeT(1),
        SupportedConeT::NonnegativeConeT(2),
        SupportedConeT::SecondOrderConeT(3),
        SupportedConeT::PSDTriangleConeT(2),
        SupportedConeT::PSDTriangleConeT(2),
        SupportedConeT::PSDTriangleConeT(2),
        SupportedConeT::PSDTriangleConeT(2),
    ];
    let m = kinds.iter().map(|c| c.nvars()).sum();
    let prepared = PreparedProblem::new(
        &CscMatrix::identity(2),
        &[n(1), n(2)],
        &CscMatrix::zeros((m, 2)),
        &vec![n(1); m],
        &kinds,
        settings(workers),
    )
    .unwrap();
    OwnedSolver::from_prepared(prepared, 12).unwrap()
}

/// Worker 1 uses the retained global sequential fold. Workers 2/4 use the
/// common-cap symmetric path, except for mixed cones and nonpositive caps.
fn exact_pooled_bounds<T: FloatT>() {
    use crate::solver::core::{ScalingStrategy, StepDirection};
    for mixed in [false, true] {
        for cap in [
            T::one(),
            T::one() / n(8),
            T::zero(),
            -T::zero(),
            -T::one() / n(4),
        ] {
            let mut reference: Option<(T, T, Vec<T>)> = None;
            for workers in [1, 2, 4] {
                let mut solver = bounds_fixture::<T>(workers, mixed);
                let v = &mut solver.variables;
                let cones = &mut solver.cones;
                let storage = cones.bounds.as_ptr();
                v.unit_initialization(cones);
                for (owner, block) in cones.blocks.iter().enumerate() {
                    for (cone, rows) in block.iter().zip(&block.rng_cones) {
                        if matches!(cone, SupportedCone::PSDTriangleCone(_)) {
                            let r = rows.start;
                            v.blocks[owner].s[r] = n(2);
                            v.blocks[owner].s[r + 1] = T::one() / n(8);
                            v.blocks[owner].s[r + 2] = n(3);
                            v.blocks[owner].z[r] = n(4);
                            v.blocks[owner].z[r + 1] = -T::one() / n(16);
                            v.blocks[owner].z[r + 2] = n(2);
                        }
                    }
                }
                for b in &mut v.blocks {
                    b.τ = cap;
                    b.κ = T::one();
                }
                assert!(v.scale_cones(cones, T::one(), ScalingStrategy::PrimalDual));
                let mut original = v.new_like();
                for (owner, (d, x)) in original.blocks.iter_mut().zip(&v.blocks).enumerate() {
                    for (local, rows) in cones.blocks[owner].rng_cones.iter().enumerate() {
                        let id = v.layout.owners[owner].cones[local].original;
                        let ratio = n::<T>((id % 3 + 1) as i32) / n(2);
                        for r in rows.clone() {
                            d.s[r] = -x.s[r] * ratio;
                            d.z[r] = x.z[r] / n(8);
                        }
                    }
                    d.τ = -T::one();
                    d.κ = T::zero();
                }
                original.sync_border();
                let alpha =
                    v.calc_step_length(&original, cones, &solver.settings, StepDirection::Combined);
                let mut transformed = original.new_like();
                transformed.copy_from(&original);
                let affine =
                    v.prepare_affine_step_length(&mut transformed, cones, &solver.settings);
                let values: Vec<T> = transformed
                    .blocks
                    .iter()
                    .flat_map(|b| {
                        b.x.iter()
                            .chain(&b.s)
                            .chain(&b.z)
                            .copied()
                            .chain([b.τ, b.κ])
                    })
                    .collect();
                if let Some((a, b, point)) = &reference {
                    exact_bound_value(alpha, *a);
                    exact_bound_value(affine, *b);
                    assert_eq!(values.len(), point.len());
                    for (&x, &y) in values.iter().zip(point) {
                        exact_bound_value(x, y);
                    }
                } else {
                    reference = Some((alpha, affine, values.clone()));
                }
                // Reuse the same cache and reset only direction values;
                // prepared transforms must not leak across repeated calls.
                transformed.copy_from(&original);
                exact_bound_value(
                    v.prepare_affine_step_length(&mut transformed, cones, &solver.settings),
                    affine,
                );
                let repeated: Vec<T> = transformed
                    .blocks
                    .iter()
                    .flat_map(|b| {
                        b.x.iter()
                            .chain(&b.s)
                            .chain(&b.z)
                            .copied()
                            .chain([b.τ, b.κ])
                    })
                    .collect();
                for (&x, &y) in repeated.iter().zip(&values) {
                    exact_bound_value(x, y);
                }
                assert_eq!(cones.bounds.as_ptr(), storage);
            }
        }
    }
}
#[test]
fn owned_pool_bounds_exact_f64() {
    exact_pooled_bounds::<f64>();
}
#[test]
fn owned_pool_bounds_exact_mpfr256() {
    exact_pooled_bounds::<MpFloat<4>>();
}
#[test]
fn owned_pool_bounds_exact_mpfr512() {
    exact_pooled_bounds::<MpFloat<8>>();
}

fn automatic_solver<T: FloatT>() {
    let mut reference = mixed_problem::<T>(1);
    reference.solve();
    assert_eq!(reference.solution.status, SolverStatus::Solved);
    for width in [1, 2, 4, 8] {
        let prepared = mixed_prepared::<T>(width);
        let expected = OwnerLayout::new_auto(&prepared.data, width).unwrap();
        let mut solver = OwnedSolver::from_prepared_auto(prepared).unwrap();
        assert_eq!(*solver.data.layout, expected);
        let workers = solver
            .cones
            .pool
            .as_ref()
            .map_or(1, |p| p.current_num_threads());
        assert_eq!(workers, width);
        assert!(expected
            .owners
            .iter()
            .all(|o| !o.columns.is_empty() || !o.counted_rows.is_empty()));
        solver.solve();
        solution_close(&solver.solution.0, &reference.solution);
    }
    let mut reference = boundary_problem::<T>(1);
    reference.solve();
    assert_eq!(reference.solution.status, SolverStatus::Solved);
    let mut solver = OwnedSolver::from_prepared_auto(boundary_prepared::<T>(2)).unwrap();
    solver.solve();
    solution_close(&solver.solution.0, &reference.solution);
}
#[test]
fn owned_auto_solver_f64() {
    automatic_solver::<f64>();
}
#[test]
fn owned_auto_solver_mpfr256() {
    automatic_solver::<MpFloat<4>>();
}
#[test]
fn owned_auto_solver_mpfr512() {
    automatic_solver::<MpFloat<8>>();
}

// One PSD owner must still borrow the entire solver pool for matrix work.
fn heavy_owner<T: FloatT>() {
    let order = 16;
    let rows = order * (order + 1) / 2;
    let mut diagonal = Vec::new();
    let mut b = vec![T::zero(); rows];
    for j in 0..order {
        let row = j * (j + 1) / 2 + j;
        diagonal.push(row);
        b[row] = n(4);
    }
    let a = CscMatrix::new(rows, 1, vec![0, order], diagonal, vec![-T::one(); order]);
    let mut reference = None;
    for width in [1, 4] {
        let prepared = PreparedProblem::new(
            &CscMatrix::identity(1),
            &[-T::one()],
            &a,
            &b,
            &[SupportedConeT::PSDTriangleConeT(order)],
            settings(width),
        )
        .unwrap();
        let mut solver = OwnedSolver::from_prepared(prepared, 1).unwrap();
        assert_eq!(
            solver
                .cones
                .pool
                .as_ref()
                .map_or(1, |p| p.current_num_threads()),
            width
        );
        solver.solve();
        assert_eq!(solver.solution.0.status, SolverStatus::Solved);
        value_close(solver.solution.0.x[0], T::one());
        if let Some(expected) = &reference {
            solution_close(&solver.solution.0, expected);
        } else {
            reference = Some(solver.solution.0);
        }
    }
}
#[test]
fn owned_heavy_owner_f64() {
    heavy_owner::<f64>();
}
#[test]
fn owned_heavy_owner_mpfr256() {
    heavy_owner::<MpFloat<4>>();
}
#[test]
fn owned_heavy_owner_mpfr512() {
    heavy_owner::<MpFloat<8>>();
}
