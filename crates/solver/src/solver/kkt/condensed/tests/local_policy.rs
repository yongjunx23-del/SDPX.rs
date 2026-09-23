use super::*;
use crate::solver::core::ScalingStrategy;
use crate::solver::SampledBlock;
use sdpx_arithmetic::Bits256;

// Every operation below is local, including direct solve/refinement, batched
// solve, cone scaling and sampled products. A live MPI world must be irrelevant.
fn exercise_mode<T: FloatT>(sampled: bool) {
    let kinds = vec![SupportedConeT::PSDTriangleConeT(32); 2];
    let mut cones = CompositeCone::new_local(&kinds);
    assert!(cones.is_local_only());
    cones.configure_threads(4).unwrap();
    let block = SampledBlock {
        row_start: 0,
        column_start: 0,
        dim: 1,
        basis_rows: 32,
        basis_cols: 4,
        basis: (0..128)
            .map(|i| {
                if i % 32 == i / 32 {
                    T::one()
                } else {
                    T::zero()
                }
            })
            .collect(),
        weights: vec![T::one(); 4],
    };
    let mut second = block.clone();
    second.row_start = block.row_count();
    let m = 2 * block.row_count();
    let op = Arc::new(
        SampledOperator::new_local(CscMatrix::zeros((m, 4)), vec![block, second]).unwrap(),
    );
    assert!(op.is_local_only());
    let a = op.materialize();
    let p = CscMatrix::identity(4);
    let settings = CoreSettings {
        max_threads: 1,
        direct_solve_method: "qdldl".into(),
        // Keep regularization/refinement active, but ask the fixture's solve
        // for the precision-level accuracy checked by the original residual.
        iterative_refinement_abstol: T::from_usize(64).unwrap() * T::epsilon(),
        iterative_refinement_reltol: T::from_usize(64).unwrap() * T::epsilon(),
        ..CoreSettings::default()
    };
    let mut s = vec![T::zero(); m];
    let mut z = s.clone();
    cones.unit_initialization(&mut z, &mut s);
    assert!(cones.update_scaling(&s, &z, T::one(), ScalingStrategy::PrimalDual));
    let mut kkt = CondensedKKTSolver::new_local_partition(&p, &a, &kinds, &cones, &settings, true);
    assert!(kkt.mpi_world().is_none());
    assert!(kkt.sparse_products.is_none());
    if sampled {
        kkt.set_sampled_operator(Arc::clone(&op));
    }
    assert!(kkt.update(&cones, &settings));
    let point = vec![T::one(); 4 + m];
    let zero = vec![T::zero(); point.len()];
    let mut rhs = zero.clone();
    kkt.original_residual(&mut rhs, &zero, &point);
    rhs.negate();
    kkt.setrhs(&rhs[..4], &rhs[4..]);
    let mut actual = zero.clone();
    let (x, z) = actual.split_at_mut(4);
    assert!(kkt.solve(Some(x), Some(z), &settings));
    let mut error = zero.clone();
    assert!(
        kkt.original_residual(&mut error, &rhs, &actual)
            <= T::from_usize(4096).unwrap() * T::epsilon() * (T::one() + rhs.norm_inf())
    );
    let mut batch_rhs = rhs.clone();
    batch_rhs.extend_from_slice(&rhs);
    let mut batch = vec![T::zero(); batch_rhs.len()];
    assert_eq!(
        kkt.solve_many(4, &batch_rhs, &mut batch, 2, &settings),
        vec![true, true]
    );
    assert_eq!(&batch[..actual.len()], &actual);
    let mut work = SampledWorkspace::new(&op);
    let mut fwd = vec![T::zero(); m];
    let mut serial = fwd.clone();
    op.apply(&mut serial, &point[..4], T::one(), T::zero(), &mut work);
    op.apply_with_pool(
        &mut fwd,
        &point[..4],
        T::one(),
        T::zero(),
        &mut work,
        cones.thread_pool().as_ref(),
    );
    assert_eq!(fwd, serial);
    let mut adj = vec![T::zero(); 4];
    let mut serial = adj.clone();
    op.apply_transpose(&mut serial, &point[4..], T::one(), T::zero(), &mut work);
    op.apply_transpose_with_pool(
        &mut adj,
        &point[4..],
        T::one(),
        T::zero(),
        &mut work,
        cones.thread_pool().as_ref(),
    );
    assert_eq!(adj, serial);
}
fn exercise<T: FloatT>() {
    exercise_mode::<T>(false);
    exercise_mode::<T>(true);
}
#[test]
fn local_policy_f64() {
    exercise::<f64>();
}
#[test]
fn local_policy_mpfr256() {
    exercise::<Bits256>();
}
#[test]
#[ignore = "requires bounded two-rank local launcher, single test thread, and timeout"]
fn local_policy_unequal_rank_work() {
    let mpi = crate::MpiContext::initialize();
    assert_eq!(mpi.size(), 2);
    // Different numbers of numerical calls expose any accidental consensus or
    // global shard operation. No numerical collective belongs inside this loop.
    for _ in 0..=mpi.rank() {
        exercise::<Bits256>();
    }
    assert_eq!(crate::mpi::World::get().unwrap().allreduce_max_f64(1.), 1.);
    mpi.finish();
}
