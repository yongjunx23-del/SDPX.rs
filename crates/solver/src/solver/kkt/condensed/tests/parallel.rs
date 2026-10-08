use super::*;
use crate::solver::core::ScalingStrategy;
use sdpx_arithmetic::Bits256;

fn num<T: FloatT>(n: usize) -> T {
    T::from_usize(n).unwrap()
}
fn wide_pool_tiles() -> usize {
    use std::sync::atomic::Ordering;
    crate::solver::kkt::condensed::POOLED_CONGRUENCE_TILES.load(Ordering::Relaxed)
}

fn close<T: FloatT>(actual: &[T], expected: &[T]) {
    let tolerance = num::<T>(65536) * T::epsilon() * (T::one() + expected.norm_inf());
    assert!(
        actual.norm_inf_diff(expected) <= tolerance,
        "residual={}, tolerance={}",
        actual.norm_inf_diff(expected),
        tolerance
    );
}

fn kinds<T: FloatT>(overlap_fallback: bool) -> Vec<SupportedConeT<T>> {
    let half = T::one() / num::<T>(2);
    let mut kinds = vec![
        SupportedConeT::ZeroConeT(0),
        SupportedConeT::PSDTriangleConeT(32),
        SupportedConeT::NonnegativeConeT(2),
        SupportedConeT::PSDTriangleConeT(0),
        SupportedConeT::SecondOrderConeT(3),
        SupportedConeT::PSDTriangleConeT(16),
        SupportedConeT::NonnegativeConeT(0),
        SupportedConeT::ZeroConeT(1),
        SupportedConeT::ExponentialConeT(),
        SupportedConeT::PowerConeT(half),
        SupportedConeT::GenPowerConeT(vec![half, half], 2),
        SupportedConeT::PSDTriangleConeT(0),
    ];
    if overlap_fallback {
        kinds.push(SupportedConeT::PSDTriangleConeT(2));
    }
    kinds
}

fn data<T: FloatT>(cones: &CompositeCone<T>) -> (CscMatrix<T>, CscMatrix<T>, Vec<T>, Vec<T>) {
    let m = cones.numel();
    let (mut s, mut z) = (vec![T::zero(); m], vec![T::zero(); m]);
    cones.unit_initialization(&mut z, &mut s);
    let mut rows = vec![[T::zero(); 3]; m];
    for (cone, range) in cones.iter().zip(&cones.rng_cones) {
        for row in range.clone() {
            for col in 0..3 {
                rows[row][col] = num::<T>((row + 2 * col) % 7 + 1) / num::<T>(16);
            }
        }
        match cone {
            SupportedCone::PSDTriangleCone(k) => {
                // Two different SPD rank-one updates, constructed directly in T.
                let mut row = range.start;
                for j in 0..k.n {
                    for i in 0..=j {
                        let ui = num::<T>(i + 1) / num::<T>(k.n + 1);
                        let uj = num::<T>(j + 1) / num::<T>(k.n + 1);
                        let vi = (if i % 2 == 0 { T::one() } else { -T::one() }) / num::<T>(i + 1);
                        let vj = (if j % 2 == 0 { T::one() } else { -T::one() }) / num::<T>(j + 1);
                        if i == j {
                            s[row] = T::one() + ui + ui * uj;
                            z[row] = num::<T>(2) + num::<T>(k.n - i) / num::<T>(k.n + 1) + vi * vj;
                        } else {
                            s[row] = T::SQRT_2() * ui * uj;
                            z[row] = T::SQRT_2() * vi * vj;
                        }
                        // Each PSD contributes to all three columns, so cliques
                        // overlap. One dense coefficient and two sparse ones
                        // exercise both per-column Schur computation routes.
                        rows[row][0] = if row == range.start {
                            T::one()
                        } else {
                            T::zero()
                        };
                        rows[row][2] = if row == range.start + 1 {
                            T::one() / num::<T>(4)
                        } else {
                            T::zero()
                        };
                        row += 1;
                    }
                }
            }
            SupportedCone::ZeroCone(_) => {
                for row in range.clone() {
                    rows[row] = [T::one(), T::one() / num::<T>(4), -T::one() / num::<T>(2)];
                }
            }
            _ => {}
        }
    }
    let p = CscMatrix::from(&[
        [
            num::<T>(4),
            T::one() / num::<T>(8),
            -T::one() / num::<T>(16),
        ],
        [T::zero(), num::<T>(5), T::one() / num::<T>(8)],
        [T::zero(), T::zero(), num::<T>(6)],
    ]);
    (p, CscMatrix::from(&rows), s, z)
}

#[test]
fn assembly_buffer_budget_bounds_extra_storage() {
    // Original rule: inside two copies of the stored Schur values.
    assert!(parallel_assembly_allowed::<f64>(true, 100, 100));
    // Overlapping cliques fit the byte budget until it is exceeded.
    let budget_cells = (parallel_assembly_budget_bytes() / std::mem::size_of::<f64>()) as u128;
    assert!(parallel_assembly_allowed::<f64>(true, budget_cells, 10));
    assert!(!parallel_assembly_allowed::<f64>(
        true,
        budget_cells + 1,
        10
    ));
    // Without a cone pool the allocation-free serial path is kept.
    assert!(!parallel_assembly_allowed::<f64>(false, 1, 1));
}

fn pooled_equivalence<T: FloatT>(overlap_fallback: bool) {
    let kinds = kinds::<T>(overlap_fallback);
    let mut serial_cones = CompositeCone::new(&kinds);
    let mut pooled_cones = CompositeCone::new(&kinds);
    pooled_cones.configure_threads(4).unwrap();
    assert_eq!(pooled_cones.cone_threads(), 4);
    let (mut p, mut a, mut s, mut z) = data(&serial_cones);
    let m = a.m;
    let mut settings = CoreSettings::<T>::default();
    settings.max_threads = 1;
    settings.direct_solve_method = "qdldl".into();
    settings.iterative_refinement_abstol = T::epsilon() * num::<T>(1024);
    settings.iterative_refinement_reltol = settings.iterative_refinement_abstol;
    let mut serial = CondensedKKTSolver::new(&p, &a, &kinds, &serial_cones, &settings);
    let mut pooled = CondensedKKTSolver::new(&p, &a, &kinds, &pooled_cones, &settings);
    assert!(!serial.parallel_assembly);
    // This fixture's contribution buffers fit the assembly budget, so
    // overlapping cliques now use the ordered parallel path as well; the
    // value-equality assertions below still pin the bitwise-identical publish.
    assert!(pooled.parallel_assembly);
    assert!(pooled
        .blocks
        .iter()
        .any(|b| matches!(&b.scaling,Scaling::Psd(p) if p.columns.iter().any(|c| c.sparse))));
    assert!(pooled
        .blocks
        .iter()
        .any(|b| matches!(&b.scaling,Scaling::Psd(p) if p.columns.iter().any(|c| !c.sparse))));
    let buffer_cells = |kkt: &CondensedKKTSolver<T>| {
        kkt.blocks
            .iter()
            .map(|b| match &b.scaling {
                Scaling::Psd(p) => p.schur_values.len(),
                _ => 0,
            })
            .sum::<usize>()
    };
    assert_eq!(buffer_cells(&serial), 0);
    // Buffers are bounded by whichever limit admitted them: the two-copy
    // rule or the byte budget.
    let budget_cells = (parallel_assembly_budget_bytes() / std::mem::size_of::<T>()) as u128;
    let allowed = (2 * pooled.schur_nnz as u128).max(budget_cells);
    assert!(buffer_cells(&pooled) as u128 <= allowed);
    assert!(buffer_cells(&pooled) > 0);
    let saved_buffers = buffer_cells(&pooled);
    let probe: Vec<T> = (0..m)
        .map(|i| num::<T>(i % 11 + 1) / num::<T>(32))
        .collect();
    let exact_x = [
        T::one() / num::<T>(4),
        -T::one() / num::<T>(2),
        T::one() / num::<T>(8),
    ];
    for (iteration, width) in [4, 1, 2, 8].into_iter().enumerate() {
        pooled_cones.configure_threads(width).unwrap();
        if iteration > 0 {
            for (i, value) in a.nzval.iter_mut().enumerate() {
                *value *= T::one() + num::<T>(i % 5 + 1) / num::<T>(128);
            }
            p.nzval[0] += T::one() / num::<T>(16);
            for value in &mut s {
                *value *= T::one() + T::one() / num::<T>(16);
            }
            for value in &mut z {
                *value *= T::one() + T::one() / num::<T>(32);
            }
            serial.update_A(&a);
            pooled.update_A(&a);
            serial.update_P(&p);
            pooled.update_P(&p);
        }
        let mu = T::one() / num::<T>(iteration + 1);
        assert!(serial_cones.update_scaling(&s, &z, mu, ScalingStrategy::Dual));
        assert!(pooled_cones.update_scaling(&s, &z, mu, ScalingStrategy::Dual));
        assert!(serial.update(&serial_cones, &settings));
        assert!(pooled.update(&pooled_cones, &settings));
        assert!(serial.pool.is_none());
        assert_eq!(
            pooled.pool.as_ref().map_or(1, |p| p.current_num_threads()),
            width
        );
        if let Some(pool) = pooled_cones.thread_pool() {
            assert!(Arc::ptr_eq(&pool, pooled.pool.as_ref().unwrap()));
        }
        assert_eq!(pooled.parallel_assembly, width > 1);
        assert_eq!(buffer_cells(&pooled), saved_buffers);
        assert_eq!(
            serial.reduced.kkt_matrix().colptr[..serial.n + 1],
            pooled.reduced.kkt_matrix().colptr[..pooled.n + 1]
        );
        assert_eq!(
            serial.reduced.kkt_matrix().rowval[..serial.schur_nnz],
            pooled.reduced.kkt_matrix().rowval[..pooled.schur_nnz]
        );
        assert_eq!(
            serial.reduced.kkt_matrix().nzval[..serial.schur_nnz],
            pooled.reduced.kkt_matrix().nzval[..pooled.schur_nnz]
        );
        assert_eq!(serial.retained_rows, pooled.retained_rows);
        assert_eq!(serial.retained_A.nzval, pooled.retained_A.nzval);
        assert!(!pooled.retained_rows.is_empty());
        for (a, b) in serial.blocks.iter().zip(&pooled.blocks) {
            if let (Scaling::Psd(a), Scaling::Psd(b)) = (&a.scaling, &b.scaling) {
                assert_eq!(a.R.data(), b.R.data());
                assert_eq!(a.Rinv.data(), b.Rinv.data());
                assert_eq!(a.Ginv.data(), b.Ginv.data());
            }
        }
        for inverse in [false, true] {
            let mut ys = vec![T::nan(); m];
            let mut yp = ys.clone();
            apply_scaling_pool(
                &serial.pool,
                &serial.scaling_lanes,
                serial.scaling_tiles,
                &mut serial.blocks,
                &mut ys,
                &probe,
                inverse,
            );
            apply_scaling_pool(
                &pooled.pool,
                &pooled.scaling_lanes,
                pooled.scaling_tiles,
                &mut pooled.blocks,
                &mut yp,
                &probe,
                inverse,
            );
            assert_eq!(ys, yp);
            assert!(yp.is_finite());

            let mut oracle = vec![T::zero(); m];
            let mut scratch = oracle.clone();
            if inverse {
                serial_cones.mul_Hs(&mut oracle, &yp, &mut scratch);
                let mut expected = probe.clone();
                for block in &pooled.blocks {
                    if !matches!(
                        block.scaling,
                        Scaling::Psd(_) | Scaling::Orthant { .. } | Scaling::SocElim { .. }
                    ) {
                        expected[block.rows.clone()].fill(T::zero());
                        assert!(yp[block.rows.clone()].iter().all(|x| *x == T::zero()));
                    }
                }
                close(&oracle, &expected);
            } else {
                serial_cones.mul_Hs(&mut oracle, &probe, &mut scratch);
                close(&yp, &oracle);
            }
        }
        // Generate a full augmented RHS from the original cone operator.
        let mut bx = vec![T::zero(); 3];
        let mut bz = vec![T::zero(); m];
        let mut hs = bz.clone();
        let mut scratch = bz.clone();
        p.sym_up().symv(&mut bx, &exact_x, T::one(), T::zero());
        a.t().gemv(&mut bx, &probe, T::one(), T::one());
        a.gemv(&mut bz, &exact_x, T::one(), T::zero());
        serial_cones.mul_Hs(&mut hs, &probe, &mut scratch);
        for (b, h) in bz.iter_mut().zip(&hs) {
            *b -= *h;
        }
        let (mut xs, mut xp) = (vec![T::zero(); 3], vec![T::zero(); 3]);
        let (mut zs, mut zp) = (vec![T::zero(); m], vec![T::zero(); m]);
        serial.setrhs(&bx, &bz);
        pooled.setrhs(&bx, &bz);
        assert!(serial.solve(Some(&mut xs), Some(&mut zs), &settings));
        assert!(pooled.solve(Some(&mut xp), Some(&mut zp), &settings));
        assert_eq!(xs, xp);
        assert_eq!(zs, zp);
        // Verify both reused initial products and fresh post-correction products.
        let full_rhs = [bx.as_slice(), bz.as_slice()].concat();
        let full_solution = [xp.as_slice(), zp.as_slice()].concat();
        a.gemv(&mut serial.workz, &xp, T::one(), T::zero());
        for (v, b) in serial.workz.iter_mut().zip(&bz) {
            *v -= *b;
        }
        pooled.workz.copy_from_slice(&serial.workz);
        for reuse in [false, true] {
            let mut rs = vec![T::nan(); full_rhs.len()];
            let mut rp = rs.clone();
            let ns = serial.residual(&mut rs, &full_rhs, &full_solution, reuse);
            let np = pooled.residual(&mut rp, &full_rhs, &full_solution, reuse);
            assert_eq!(rs, rp);
            assert_eq!(ns, np);
        }
        let mut ex = bx.clone();
        let mut ez = bz.clone();
        p.sym_up().symv(&mut ex, &xp, -T::one(), T::one());
        a.t().gemv(&mut ex, &zp, -T::one(), T::one());
        a.gemv(&mut ez, &xp, -T::one(), T::one());
        serial_cones.mul_Hs(&mut hs, &zp, &mut scratch);
        for (e, h) in ez.iter_mut().zip(&hs) {
            *e += *h;
        }
        let tolerance =
            num::<T>(65536) * T::epsilon() * (T::one() + bx.norm_inf().max(bz.norm_inf()));
        assert!(
            ex.norm_inf().max(ez.norm_inf()) <= tolerance,
            "original KKT residual={}, tolerance={}",
            ex.norm_inf().max(ez.norm_inf()),
            tolerance
        );
    }
    // A solver constructed without contribution caches can still enable
    // pooled operators later, without allocating another Schur-sized copy.
    serial_cones.configure_threads(2).unwrap();
    assert!(serial.update(&serial_cones, &settings));
    assert_eq!(serial.pool.as_ref().unwrap().current_num_threads(), 2);
    assert!(!serial.parallel_assembly);
    assert_eq!(buffer_cells(&serial), 0);
    assert_eq!(
        serial.reduced.kkt_matrix().nzval[..serial.schur_nnz],
        pooled.reduced.kkt_matrix().nzval[..pooled.schur_nnz]
    );
}

#[test]
fn pooled_condensed_f64() {
    pooled_equivalence::<f64>(false);
}
#[test]
#[ignore = "extended: MPFR pool/threading sweep; default f64 covers the equivalence logic"]
fn pooled_condensed_mpfr256() {
    pooled_equivalence::<Bits256>(false);
}
#[test]
#[ignore = "extended: MPFR pool/threading sweep; default f64 covers the equivalence logic"]
fn pooled_condensed_mpfr512() {
    pooled_equivalence::<sdpx_arithmetic::Bits512>(false);
}
#[test]
#[ignore = "extended: MPFR pool/threading sweep; default f64 covers the equivalence logic"]
fn overlapping_memory_fallback_mpfr512() {
    pooled_equivalence::<sdpx_arithmetic::Bits512>(true);
}
#[test]
fn overlapping_memory_fallback_f64() {
    pooled_equivalence::<f64>(true);
}
#[test]
#[ignore = "extended: MPFR pool/threading sweep; default f64 covers the equivalence logic"]
fn overlapping_memory_fallback_mpfr256() {
    pooled_equivalence::<Bits256>(true);
}

fn orthant_fma<T: FloatT>() {
    let kinds = [
        SupportedConeT::ZeroConeT(0),
        SupportedConeT::NonnegativeConeT(1),
        SupportedConeT::PSDTriangleConeT(0),
    ];
    let mut cones = CompositeCone::new(&kinds);
    cones.set_identity_scaling();
    let delta = num::<T>(2).powi(-((T::precision_bits() / 2 + 1) as i32));
    let ai = T::one() + delta;
    let aj = T::one() - delta;
    assert_eq!(ai * aj - T::one(), T::zero());
    let p = CscMatrix::from(&[[num::<T>(2), -T::one()], [T::zero(), num::<T>(2)]]);
    let a = CscMatrix::from(&[[ai, aj]]);
    let settings = CoreSettings::default();
    let mut solver = CondensedKKTSolver::new(&p, &a, &kinds, &cones, &settings);
    // Explicit test pool for the tiny zero-length edge fixture. Assembly's
    // orthant path is intentionally serial, even when block operators pool.
    solver.pool = Some(Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap(),
    ));
    for block in &mut solver.blocks {
        if let Scaling::Orthant { w, .. } = &mut block.scaling {
            w.fill(T::one());
        }
    }
    assert!(solver.assemble());
    let position = schur_position(solver.reduced.kkt_matrix(), 0, 1);
    assert_eq!(
        solver.reduced.kkt_matrix().nzval[..solver.schur_nnz][position],
        -delta * delta
    );
    for inverse in [false, true] {
        let mut y = [T::nan()];
        apply_scaling_pool(
            &solver.pool,
            &solver.scaling_lanes,
            solver.scaling_tiles,
            &mut solver.blocks,
            &mut y,
            &[ai],
            inverse,
        );
        assert_eq!(y, [ai]);
        apply_scaling_pool::<T>(&solver.pool, &[], 1, &mut [], &mut [], &[], inverse);
    }
}

// The production case this exists for: a pool wider than the block count, where
// the lane level alone cannot fill it. The fixture's cone sizing deliberately
// caps its own pool, so this installs an explicit wide pool to reach the tiled
// branch, then requires the widened schedule to reproduce the serial result
// exactly and to actually take that branch.

fn wide_pool_splits_congruence_tiles<T: FloatT>() {
    let kinds = kinds::<T>(false);
    let mut cones = CompositeCone::new(&kinds);
    let (p, a, s, z) = data(&cones);
    let m = a.m;
    let mut settings = CoreSettings::<T>::default();
    settings.max_threads = 1;

    settings.direct_solve_method = "qdldl".into();
    settings.iterative_refinement_abstol = T::epsilon() * num::<T>(1024);
    settings.iterative_refinement_reltol = settings.iterative_refinement_abstol;
    assert!(cones.update_scaling(&s, &z, T::one(), ScalingStrategy::Dual));
    let mut solver = CondensedKKTSolver::new(&p, &a, &kinds, &cones, &settings);
    assert!(solver.update(&cones, &settings));
    let psd_blocks = solver
        .blocks
        .iter()
        .filter(|b| matches!(&b.scaling, Scaling::Psd(_)))
        .count();
    let workers = 64;
    assert!(
        psd_blocks < workers,
        "fixture must have fewer PSD blocks than the pool to exercise tiling"
    );
    let probe: Vec<T> = (0..m)
        .map(|i| num::<T>(i % 11 + 1) / num::<T>(32))
        .collect();
    let mut serial = vec![vec![T::nan(); m]; 2];
    let mut wide = serial.clone();
    for (slot, inverse) in serial.iter_mut().zip([false, true]) {
        apply_scaling_pool(
            &None,
            &solver.scaling_lanes,
            solver.scaling_tiles,
            &mut solver.blocks,
            slot,
            &probe,
            inverse,
        );
    }
    solver.pool = Some(Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .unwrap(),
    ));
    // The plan follows the pool that exists when it is refreshed, so install
    // the wider pool first and then recompute the dispatch.
    solver.refresh_parallel_plan();
    let before = wide_pool_tiles();
    for (slot, inverse) in wide.iter_mut().zip([false, true]) {
        apply_scaling_pool(
            &solver.pool,
            &solver.scaling_lanes,
            solver.scaling_tiles,
            &mut solver.blocks,
            slot,
            &probe,
            inverse,
        );
    }
    assert_eq!(serial, wide);
    assert!(
        wide_pool_tiles() > before,
        "a pool wider than the block count must split congruence GEMMs into column tiles"
    );
}

#[test]
fn wide_pool_splits_congruence_tiles_f64() {
    wide_pool_splits_congruence_tiles::<f64>();
}

#[test]
#[ignore = "extended: MPFR pool/threading sweep; default f64 covers the equivalence logic"]
fn wide_pool_splits_congruence_tiles_mpfr256() {
    wide_pool_splits_congruence_tiles::<Bits256>();
}

#[test]
#[ignore = "extended: MPFR pool/threading sweep; default f64 covers the equivalence logic"]
fn wide_pool_splits_congruence_tiles_mpfr512() {
    wide_pool_splits_congruence_tiles::<sdpx_arithmetic::Bits512>();
}

#[test]
fn pooled_orthant_preserves_fma_f64() {
    orthant_fma::<f64>();
}
#[test]
fn pooled_orthant_preserves_fma_mpfr256() {
    orthant_fma::<Bits256>();
}

#[test]
fn scaling_dispatch_decouples_lanes_from_workers() {
    // The 3D Ising reduced system: 22 near-uniform PSD blocks at 512 bits.
    let dims = [
        12usize, 12, 13, 12, 14, 13, 15, 14, 16, 15, 16, 15, 16, 15, 16, 15, 16, 15, 16, 15, 16, 15,
    ];
    let costs: Vec<u128> = dims.iter().map(|d| 4 * (*d as u128).pow(3)).collect();
    let total: u128 = costs.iter().sum();

    // The chooser may not use more than four lanes per worker, but within that
    // bound it must never be worse than the worker-count partition it replaces,
    // and at the widths that matter it uses every block as its own lane.
    for workers in [2usize, 4, 8, 16] {
        let (lanes, _) = scaling_dispatch(&costs, workers);
        assert!(
            lanes.len() <= costs.len().min(workers * 4),
            "workers={workers}"
        );
        assert!(lanes.len() >= workers.min(costs.len()), "workers={workers}");
        let chosen = lpt_makespan(&lane_loads(&costs, &lanes), workers);
        let worker_partition = weighted_lanes(&costs, workers);
        let previous = lpt_makespan(&lane_loads(&costs, &worker_partition), workers);
        assert!(
            chosen <= previous,
            "workers={workers}: chosen={chosen} previous={previous}"
        );
        // The whole point of the finer partition: at eight workers the
        // worker-count schedule loses 1.31x while one lane per block reaches
        // 1.09x, and no lane may exceed a single block's cost.
        assert!(chosen <= total / workers as u128 * 5 / 2);
        if workers >= 8 {
            assert_eq!(lanes.len(), costs.len(), "workers={workers}");
        }
        if workers >= 8 {
            let loads = lane_loads(&costs, &lanes);
            assert_eq!(loads.iter().copied().max().unwrap(), 4 * 16u128.pow(3));
        }
    }
}

#[test]
fn scaling_dispatch_tiles_the_dominant_block() {
    // A dominant block that outweighs an equal share of the pool must be split
    // into column tiles even when it already owns a lane, which is the case the
    // old lane-count gate left unsplit.
    let costs = [
        4 * 64u128.pow(3),
        4 * 8u128.pow(3),
        4 * 8u128.pow(3),
        4 * 8u128.pow(3),
    ];
    let (lanes, tiles) = scaling_dispatch(&costs, 8);
    assert_eq!(lanes.len(), costs.len());
    assert!(tiles >= 2, "tiles={tiles}");
    // Balanced blocks never tile.
    let balanced = [1000u128, 1000, 1000, 1000];
    let (_, tiles) = scaling_dispatch(&balanced, 4);
    assert_eq!(tiles, 1);
    // A serial plan has a single lane and no tiles.
    let (lanes, tiles) = scaling_dispatch(&balanced, 1);
    assert_eq!(lanes, vec![0]);
    assert_eq!(tiles, 1);
    assert_eq!(scaling_dispatch(&[], 8), (Vec::new(), 1));
}

#[test]
fn scaling_partitions_follow_cost_and_cover_zero_work() {
    assert_eq!(weighted_lanes(&[900, 100, 100, 100], 2), vec![0, 1]);
    // Equal-count splitting would put the first two blocks in the same lane.
    for costs in [vec![0, 0, 0], vec![0, 900, 0, 100, 0], vec![1, 7, 31, 2, 0]] {
        for workers in [1, 2, 4, 8] {
            let lanes = weighted_lanes(&costs, workers);
            assert_eq!(lanes[0], 0);
            assert_eq!(lanes.len(), workers.min(costs.len()));
            assert!(lanes.windows(2).all(|pair| pair[0] < pair[1]));
            assert!(*lanes.last().unwrap() < costs.len());
        }
    }
    assert!(weighted_lanes(&[], 8).is_empty());
}

fn dominant_sparse_psd<T: FloatT>() {
    // The block dimension must clear the precision-scaled sparse-column
    // threshold so the sparse pair path stays exercised at Float64 too.
    let kinds = [SupportedConeT::PSDTriangleConeT(64)];
    let mut serial_cones = CompositeCone::new(&kinds);
    let mut pooled_cones = CompositeCone::new(&kinds);
    pooled_cones.configure_threads(8).unwrap();
    assert_eq!(pooled_cones.cone_threads(), 8);
    serial_cones.set_identity_scaling();
    pooled_cones.set_identity_scaling();
    let m = serial_cones.numel();
    let n = 64;
    let mut colptr = vec![0];
    let mut rowval = Vec::new();
    let mut nzval = Vec::new();
    for j in 0..n {
        for i in 0..4 {
            // Repeated, overlapping supports ensure nonzero multi-term sums.
            rowval.push(i * 16 + j % 13);
            nzval.push(num::<T>((j + i) % 11 + 1) / num::<T>(16));
        }
        colptr.push(nzval.len());
    }
    let mut a = CscMatrix::new(m, n, colptr, rowval, nzval);
    let p = CscMatrix::identity(n);
    let settings = CoreSettings::<T> {
        max_threads: 1,
        direct_solve_method: "qdldl".into(),
        ..CoreSettings::default()
    };
    let mut serial = CondensedKKTSolver::new(&p, &a, &kinds, &serial_cones, &settings);
    let mut pooled = CondensedKKTSolver::new(&p, &a, &kinds, &pooled_cones, &settings);
    // Exercise the owner override explicitly.  The planner must still keep
    // inner admission off for a one-worker pool and enable it only after the
    // shared pool grows, preserving the existing single-level thresholds.
    pooled.set_owner_inner_admission(true);
    for workers in [2, 4, 8, 1, 4] {
        pooled_cones.configure_threads(workers).unwrap();
        for (i, v) in a.nzval.iter_mut().enumerate() {
            *v *= T::one() + num::<T>(i % 3 + 1) / num::<T>(128);
        }
        serial.update_A(&a);
        pooled.update_A(&a);
        assert!(serial.update(&serial_cones, &settings));
        assert!(pooled.update(&pooled_cones, &settings));
        assert_eq!(pooled.plan_threads, workers);
        assert_eq!(pooled.inner_schur, workers > 1);
        assert_eq!(pooled.parallel_assembly, workers > 1);
        assert_eq!(pooled.scaling_lanes, vec![0]);
        assert_eq!(
            serial.reduced.kkt_matrix().nzval[..serial.schur_nnz],
            pooled.reduced.kkt_matrix().nzval[..pooled.schur_nnz]
        );
        let Scaling::Psd(block) = &pooled.blocks[0].scaling else {
            unreachable!()
        };
        assert!(block.columns.iter().all(|c| c.sparse));
        assert_eq!(block.sparse_column_lanes.len(), workers);
        let plan_pointer = block.sparse_column_lanes.as_ptr();
        let output_pointer = block.schur_values.as_ptr();
        let scaling_pointer = pooled.scaling_lanes.as_ptr();
        assert!(pooled.update(&pooled_cones, &settings));
        let Scaling::Psd(block) = &pooled.blocks[0].scaling else {
            unreachable!()
        };
        assert_eq!(block.sparse_column_lanes.as_ptr(), plan_pointer);
        assert_eq!(block.schur_values.as_ptr(), output_pointer);
        assert_eq!(pooled.scaling_lanes.as_ptr(), scaling_pointer);
        assert_eq!(
            serial.reduced.kkt_matrix().nzval[..serial.schur_nnz],
            pooled.reduced.kkt_matrix().nzval[..pooled.schur_nnz]
        );
    }
}

#[test]
fn dominant_sparse_psd_f64() {
    dominant_sparse_psd::<f64>();
}

#[test]
#[ignore = "extended: MPFR pool/threading sweep; default f64 covers the equivalence logic"]
fn dominant_sparse_psd_mpfr256() {
    dominant_sparse_psd::<Bits256>();
}

fn dominant_sampled_psd<T: FloatT>() {
    dominant_sampled_pool_mode::<T>(false, 1);
}
fn dominant_sampled_pool_mode<T: FloatT>(external: bool, blocks: usize) {
    use crate::solver::SampledBlock;
    let block = SampledBlock {
        row_start: 0,
        column_start: 0,
        dim: 2,
        basis_rows: 24,
        basis_cols: 16,
        basis: (0..384)
            .map(|i| num::<T>(i % 13 + 1) / num::<T>(32))
            .collect(),
        weights: vec![T::one(); 48],
    };
    let (rows, n) = (block.row_count(), block.column_count());
    let m = rows * blocks;
    let operator = Arc::new(
        SampledOperator::new(
            CscMatrix::zeros((m, n)),
            (0..blocks)
                .map(|i| {
                    let mut b = block.clone();
                    b.row_start = i * rows;
                    b
                })
                .collect(),
        )
        .unwrap(),
    );
    let a = operator.materialize();
    let p = CscMatrix::identity(n);
    let kinds = vec![SupportedConeT::PSDTriangleConeT(48); blocks];
    let mut serial_cones = CompositeCone::new(&kinds);
    let mut pooled_cones = CompositeCone::new(&kinds);
    let mut reference_cones = CompositeCone::new(&kinds);
    reference_cones.set_identity_scaling();
    serial_cones.set_identity_scaling();
    pooled_cones.set_identity_scaling();
    let settings = CoreSettings::<T> {
        max_threads: 1,
        direct_solve_method: "qdldl".into(),
        iterative_refinement_abstol: T::epsilon() * num::<T>(1024),
        iterative_refinement_reltol: T::epsilon() * num::<T>(1024),
        ..CoreSettings::default()
    };
    let mut serial = CondensedKKTSolver::new(&p, &a, &kinds, &serial_cones, &settings);
    let mut pooled = CondensedKKTSolver::new(&p, &a, &kinds, &pooled_cones, &settings);
    serial.set_sampled_operator(Arc::clone(&operator));
    pooled.set_sampled_operator(Arc::clone(&operator));
    let mut operator_work = SampledWorkspace::new(&operator);
    let exact_x: Vec<_> = (0..n).map(|i| num::<T>(i % 7 + 1) / num::<T>(64)).collect();
    let exact_z: Vec<_> = (0..m)
        .map(|i| num::<T>(i % 11 + 1) / num::<T>(128))
        .collect();
    let mut bx = exact_x.clone();
    operator.apply_transpose(&mut bx, &exact_z, T::one(), T::one(), &mut operator_work);
    let mut bz = exact_z.clone();
    let mut hs_work = vec![T::zero(); m];
    reference_cones.mul_Hs(&mut bz, &exact_z, &mut hs_work);
    operator.apply(&mut bz, &exact_x, T::one(), -T::one(), &mut operator_work);
    // The input and serial scaling are invariant across pool transitions.
    // Build the independent reference once; each candidate still refactors,
    // checks reuse, solves both RHS columns, and checks original residuals.
    assert!(serial.update(&serial_cones, &settings));
    let (mut xs, mut zs) = (vec![T::zero(); n], vec![T::zero(); m]);
    serial.setrhs(&bx, &bz);
    assert!(serial.solve(Some(&mut xs), Some(&mut zs), &settings));
    for workers in [1, 2, 4, 8, 1, 4] {
        let shared = if external {
            test_shared_pool(workers)
        } else {
            pooled_cones.configure_threads(workers).unwrap();
            pooled_cones.thread_pool()
        };
        assert!(if external {
            pooled.update_partition_with_pool(&pooled_cones, &settings, true, shared.clone())
        } else {
            pooled.update(&pooled_cones, &settings)
        });
        if external {
            assert!(pooled_cones.thread_pool().is_none());
            assert_shared_pool(&pooled, &shared);
        }
        assert_eq!(pooled.plan_threads, workers);
        assert_eq!(
            pooled.inner_sampled,
            if workers > 1 && blocks == 1 {
                Some(0)
            } else {
                None
            }
        );
        assert_eq!(pooled.parallel_assembly, workers > 1);
        assert_eq!(
            serial.reduced.kkt_matrix().nzval[..serial.schur_nnz],
            pooled.reduced.kkt_matrix().nzval[..pooled.schur_nnz]
        );
        let Scaling::Psd(psd) = &pooled.blocks[0].scaling else {
            unreachable!()
        };
        let sampled = psd.sampled.as_ref().unwrap();
        if workers > blocks {
            assert!(sampled.pair_lanes.len() > 1);
        }
        let pointers = (sampled.pair_lanes.as_ptr(), psd.schur_values.as_ptr());
        assert!(if external {
            pooled.update_partition_with_pool(&pooled_cones, &settings, true, shared.clone())
        } else {
            pooled.update(&pooled_cones, &settings)
        });
        if external {
            assert!(pooled_cones.thread_pool().is_none());
            assert_shared_pool(&pooled, &shared);
        }
        let Scaling::Psd(psd) = &pooled.blocks[0].scaling else {
            unreachable!()
        };
        assert_eq!(
            pointers,
            (
                psd.sampled.as_ref().unwrap().pair_lanes.as_ptr(),
                psd.schur_values.as_ptr()
            )
        );
        let (mut xp, mut zp) = (vec![T::zero(); n], vec![T::zero(); m]);
        pooled.setrhs(&bx, &bz);
        assert!(pooled.solve(Some(&mut xp), Some(&mut zp), &settings));
        assert_eq!(xs, xp);
        assert_eq!(zs, zp);
        // Each batched column has its own half-scaled RHS and refinement state.
        let mut rhs = bx.clone();
        rhs.extend_from_slice(&bz);
        let first = rhs.clone();
        rhs.extend(first.iter().map(|&v| -v));
        let mut batch = vec![T::zero(); rhs.len()];
        assert_eq!(
            pooled.solve_many(n, &rhs, &mut batch, 2, &settings),
            vec![true, true]
        );
        assert_eq!(&batch[..n], &xp);
        assert_eq!(&batch[n..n + m], &zp);
        pooled.setrhs(&rhs[n + m..2 * n + m], &rhs[2 * n + m..]);
        let mut second = vec![T::zero(); n + m];
        let (xx, zz) = second.split_at_mut(n);
        assert!(pooled.solve(Some(xx), Some(zz), &settings));
        assert_eq!(&batch[n + m..], &second);
        // External full augmented equations, using factor-authoritative A.
        let mut ex = bx.clone();
        for (e, x) in ex.iter_mut().zip(&xp) {
            *e -= *x;
        }
        operator.apply_transpose(&mut ex, &zp, -T::one(), T::one(), &mut operator_work);
        let mut ez = bz.clone();
        let mut hz = vec![T::zero(); m];
        reference_cones.mul_Hs(&mut hz, &zp, &mut hs_work);
        for (e, z) in ez.iter_mut().zip(&hz) {
            *e += *z;
        }
        operator.apply(&mut ez, &xp, -T::one(), T::one(), &mut operator_work);
        let tolerance =
            num::<T>(65536) * T::epsilon() * (T::one() + bx.norm_inf().max(bz.norm_inf()));
        let residual = ex.norm_inf().max(ez.norm_inf());
        assert!(
            residual <= tolerance,
            "original KKT residual={residual}, tolerance={tolerance}, workers={workers}"
        );
        close(&xp, &exact_x);
        close(&zp, &exact_z);
    }
}
#[test]
fn dominant_sampled_psd_f64() {
    dominant_sampled_psd::<f64>();
}
#[test]
#[ignore = "extended: MPFR pool/threading sweep; default f64 covers the equivalence logic"]
fn dominant_sampled_psd_mpfr256() {
    dominant_sampled_psd::<Bits256>();
}
#[test]
#[ignore = "extended: MPFR pool/threading sweep; default f64 covers the equivalence logic"]
fn dominant_sampled_psd_mpfr512() {
    dominant_sampled_psd::<sdpx_arithmetic::Bits512>();
}

#[test]
fn many_small_sampled_blocks_keep_outer_plan() {
    use crate::solver::SampledBlock;
    let mut rows = 0;
    let blocks: Vec<_> = (0..22)
        .map(|_| {
            let b = SampledBlock {
                row_start: rows,
                column_start: 0,
                dim: 2,
                basis_rows: 8,
                basis_cols: 1,
                basis: vec![num::<Bits256>(1); 8],
                weights: vec![num::<Bits256>(1); 3],
            };
            rows += b.row_count();
            b
        })
        .collect();
    let operator = Arc::new(SampledOperator::new(CscMatrix::zeros((rows, 3)), blocks).unwrap());
    let kinds = vec![SupportedConeT::PSDTriangleConeT(16); 22];
    let mut cones = CompositeCone::new(&kinds);
    cones.configure_threads(8).unwrap();
    let settings = CoreSettings::<Bits256> {
        max_threads: 8,
        direct_solve_method: "qdldl".into(),
        ..CoreSettings::default()
    };
    let mut kkt = CondensedKKTSolver::new(
        &CscMatrix::identity(3),
        &operator.materialize(),
        &kinds,
        &cones,
        &settings,
    );
    kkt.set_sampled_operator(operator);
    assert_eq!(kkt.plan_threads, 8);
    assert_eq!(kkt.inner_sampled, None);
    for block in &kkt.blocks {
        if let Scaling::Psd(p) = &block.scaling {
            let sampled = p.sampled.as_ref().unwrap();
            assert_eq!(sampled.pair_lanes.len(), 1);
            assert!(!sampled.work.has_parallel_columns());
        }
    }
}

// Compare against the original assembly after changing both scaling and A.
// Include non-dyadic quotients and an empty row to exercise cache reuse.
fn cached_orthant_division<T: FloatT>() {
    let kinds = [SupportedConeT::NonnegativeConeT(3)];
    let mut cones = CompositeCone::new(&kinds);
    cones.set_identity_scaling();
    let p = CscMatrix::zeros((3, 3));
    let a = CscMatrix::from(&[
        [
            num::<T>(1) / num::<T>(7),
            num::<T>(2) / num::<T>(3),
            -num::<T>(4),
        ],
        [T::zero(), num::<T>(3), num::<T>(5) / num::<T>(11)],
        [T::zero(), T::zero(), T::zero()],
    ]);
    let settings = CoreSettings::default();
    let mut solver = CondensedKKTSolver::new(&p, &a, &kinds, &cones, &settings);
    for turn in 1..=3 {
        if turn == 3 {
            let mut changed = a.clone();
            for v in &mut changed.nzval {
                *v *= num::<T>(2);
            }
            solver.update_A(&changed);
        }
        for block in &mut solver.blocks {
            if let Scaling::Orthant { w, .. } = &mut block.scaling {
                w.fill(num::<T>(7) / num::<T>(turn + 1));
            }
        }
        // Quotients use one reciprocal per row; binary64 chains FMAs in row
        // order, wide precision rounds each entry's exact sum once.
        let mut terms = vec![Vec::new(); solver.schur_nnz];
        for block in &solver.blocks {
            if let Scaling::Orthant { w, rows, .. } = &block.scaling {
                for (r, entries) in rows.iter().enumerate() {
                    let inv = w[r].recip();
                    for (b, &(j, q)) in entries.iter().enumerate() {
                        let aj = solver.A.nzval[q] * inv;
                        for &(i, k) in &entries[..=b] {
                            let ai = solver.A.nzval[k] * inv;
                            let pos = schur_position(solver.reduced.kkt_matrix(), i, j);
                            terms[pos].push((ai, aj));
                        }
                    }
                }
            }
        }
        let expected: Vec<T> = terms
            .iter()
            .map(|t| {
                if T::precision_bits() <= 53 {
                    t.iter().fold(T::zero(), |v, &(a, b)| a.mul_add(b, v))
                } else {
                    T::dot_fma(t.iter().map(|(a, b)| (a, b)))
                }
            })
            .collect();
        assert!(solver.assemble());
        assert_eq!(
            solver.reduced.kkt_matrix().nzval[..solver.schur_nnz],
            expected
        );
    }
}
#[test]
fn cached_orthant_division_f64() {
    cached_orthant_division::<f64>();
}
#[test]
fn cached_orthant_division_256() {
    cached_orthant_division::<Bits256>();
}
#[test]
fn cached_orthant_division_512() {
    cached_orthant_division::<sdpx_arithmetic::Bits512>();
}

// A20/A21: a batched wave must apply every column to the one shared
// factorization, must reproduce each single-column solve exactly, must isolate a
// bad column instead of masking it, and must keep the accounting honest.
fn solve_many_accounting<T: FloatT>() {
    use crate::solver::kkt::SolveCounters;
    let kinds = kinds::<T>(false);
    let mut cones = CompositeCone::new(&kinds);
    let (p, a, s, z) = data(&cones);
    let mut settings = CoreSettings::<T>::default();
    settings.max_threads = 1;
    settings.iterative_refinement_enable = true;
    settings.iterative_refinement_abstol = T::epsilon() * num::<T>(1024);
    settings.iterative_refinement_reltol = settings.iterative_refinement_abstol;
    assert!(cones.update_scaling(&s, &z, T::one(), ScalingStrategy::Dual));
    let mut solver = CondensedKKTSolver::new(&p, &a, &kinds, &cones, &settings);
    assert!(solver.update(&cones, &settings));

    let n = solver.n;
    let width = a.m + n;
    let counters = crate::solver::kkt::KKTSolver::counters(&solver);
    assert_eq!(
        counters,
        SolveCounters {
            factor_attempts: 1,
            factorizations: 1,
            ..Default::default()
        },
        "one KKT update must produce exactly one factorization"
    );

    // Three independent right-hand sides.
    let ncols = 3;
    let rhs: Vec<T> = (0..ncols * width)
        .map(|i| num::<T>(i % 13 + 1) / num::<T>(16))
        .collect();
    let mut batch = vec![T::zero(); ncols * width];
    let ok = KKTSolver::solve_many(&mut solver, n, &rhs, &mut batch, ncols, &settings);
    assert_eq!(ok, vec![true; ncols], "all three columns must solve");

    let after = KKTSolver::counters(&solver);
    assert_eq!(after.factorizations, 1, "batching must not refactorize");
    assert_eq!(after.rhs_applied, ncols as u64);
    assert_eq!(after.batches, 1, "the wave is one batch, not {ncols}");

    let mut hs = vec![T::zero(); a.m];
    let mut scratch = hs.clone();
    for c in 0..ncols {
        cones.mul_Hs(
            &mut hs,
            &batch[c * width + n..(c + 1) * width],
            &mut scratch,
        );
        close(
            solver
                .scaled_solution(c)
                .expect("accepted original residual product"),
            &hs,
        );
    }

    // Every column must match the single-RHS path bit for bit.
    for c in 0..ncols {
        let base = c * width;
        let mut single = vec![T::zero(); width];
        let (xl, zl) = single.split_at_mut(n);
        KKTSolver::setrhs(
            &mut solver,
            &rhs[base..base + n],
            &rhs[base + n..base + width],
        );
        assert!(KKTSolver::solve(&mut solver, Some(xl), Some(zl), &settings));
        assert_eq!(
            &single[..],
            &batch[base..base + width],
            "column {c} differs between batched and single solve"
        );
    }

    // A non-finite column must fail alone, leaving the good columns usable.
    let mut mixed = rhs.clone();
    for v in mixed[n..width].iter_mut() {
        *v = T::nan();
    }
    let mut mixed_out = vec![T::zero(); ncols * width];
    let flags = KKTSolver::solve_many(&mut solver, n, &mixed, &mut mixed_out, ncols, &settings);
    assert_eq!(
        flags,
        vec![false, true, true],
        "failure must stay local to its column"
    );
    assert!(solver.scaled_solution(0).is_none());
    assert!(solver.scaled_solution(1).is_some());
    settings.iterative_refinement_enable = false;
    solver.setrhs(&rhs[..n], &rhs[n..width]);
    let (x, z) = mixed_out[..width].split_at_mut(n);
    assert!(solver.solve(Some(x), Some(z), &settings));
    assert!(
        solver.scaled_solution(0).is_none(),
        "no original residual was computed"
    );
    assert!(solver.update(&cones, &settings));
    assert!(
        solver.scaled_solution(0).is_none(),
        "factor updates invalidate products"
    );
}

#[test]
fn solve_many_accounting_f64() {
    solve_many_accounting::<f64>();
}

#[test]
#[ignore = "extended: MPFR pool/threading sweep; default f64 covers the equivalence logic"]
fn solve_many_accounting_mpfr256() {
    solve_many_accounting::<Bits256>();
}

#[test]
#[ignore = "extended: MPFR pool/threading sweep; default f64 covers the equivalence logic"]
fn solve_many_accounting_mpfr512() {
    solve_many_accounting::<sdpx_arithmetic::Bits512>();
}

fn test_shared_pool(workers: usize) -> Option<Arc<rayon::ThreadPool>> {
    (workers > 1).then(|| {
        Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .build()
                .unwrap(),
        )
    })
}
fn assert_shared_pool<T: FloatT>(
    solver: &CondensedKKTSolver<T>,
    expected: &Option<Arc<rayon::ThreadPool>>,
) {
    match (&solver.pool, expected) {
        (Some(actual), Some(expected)) => assert!(Arc::ptr_eq(actual, expected)),
        (None, None) => (),
        _ => panic!("explicit pool was substituted"),
    }
}
fn external_ordinary<T: FloatT>(blocks: usize) {
    let kinds = vec![SupportedConeT::PSDTriangleConeT(32); blocks];
    let mut cones = CompositeCone::new(&kinds);
    let (p, a, s, z) = data(&cones);
    assert!(cones.update_scaling(&s, &z, T::one(), ScalingStrategy::PrimalDual));
    let settings = CoreSettings {
        max_threads: 1,
        direct_solve_method: "qdldl".into(),
        iterative_refinement_abstol: T::epsilon() * num::<T>(1024),
        iterative_refinement_reltol: T::epsilon() * num::<T>(1024),
        ..CoreSettings::default()
    };
    let mut serial = CondensedKKTSolver::new(&p, &a, &kinds, &cones, &settings);
    let mut pooled = CondensedKKTSolver::new(&p, &a, &kinds, &cones, &settings);
    pooled.prepare_shared_pool();
    let storage: Vec<_> = pooled
        .blocks
        .iter()
        .filter_map(|b| match &b.scaling {
            Scaling::Psd(p) => Some(p.schur_values.as_ptr()),
            _ => None,
        })
        .collect();
    assert!(serial.update(&cones, &settings));
    let known: Vec<T> = (0..a.n + a.m)
        .map(|i| num::<T>(i % 13 + 1) / num::<T>(64))
        .collect();
    let zeros = vec![T::zero(); known.len()];
    let mut rhs = zeros.clone();
    serial.original_residual(&mut rhs, &zeros, &known);
    rhs.negate();
    let mut xs = vec![T::zero(); a.n];
    let mut zs = vec![T::zero(); a.m];
    serial.setrhs(&rhs[..a.n], &rhs[a.n..]);
    assert!(serial.solve(Some(&mut xs), Some(&mut zs), &settings));
    // Repeated width 4 replaces a pool at unchanged capacity; retain it once.
    for width in [1, 2, 4, 8, 4, 4, 1] {
        let pool = test_shared_pool(width);
        assert!(pooled.update_partition_with_pool(&cones, &settings, true, pool.clone()));
        assert_shared_pool(&pooled, &pool);
        assert!(cones.thread_pool().is_none());
        assert_eq!(pooled.plan_threads, width);
        assert_eq!(
            serial.reduced.kkt_matrix().nzval[..serial.schur_nnz],
            pooled.reduced.kkt_matrix().nzval[..pooled.schur_nnz]
        );
        // Two fully overlapping cliques can exceed the original contribution
        // cap; operator tiling still applies when assembly stays serial.
        let before = wide_pool_tiles();
        let mut scaled = vec![T::zero(); a.m];
        let mut expected = scaled.clone();
        apply_scaling_pool(
            &pooled.pool,
            &pooled.scaling_lanes,
            pooled.scaling_tiles,
            &mut pooled.blocks,
            &mut scaled,
            &known[a.n..],
            false,
        );
        apply_scaling_pool(
            &serial.pool,
            &serial.scaling_lanes,
            serial.scaling_tiles,
            &mut serial.blocks,
            &mut expected,
            &known[a.n..],
            false,
        );
        // Native POTRS may choose a different blocked TRSM accumulation for
        // the tiled RHS width. MPFR retains its exact per-column sweep.
        assert_eq!(scaled, expected);
        if width > blocks {
            assert!(wide_pool_tiles() > before);
        }
        let mut xp = xs.clone();
        let mut zp = zs.clone();
        pooled.setrhs(&rhs[..a.n], &rhs[a.n..]);
        assert!(pooled.solve(Some(&mut xp), Some(&mut zp), &settings));
        assert_eq!(xs, xp);
        assert_eq!(zs, zp);
        xp.extend(zp);
        let mut residual = zeros.clone();
        assert!(
            pooled.original_residual(&mut residual, &rhs, &xp)
                <= num::<T>(65536) * T::epsilon() * (T::one() + rhs.norm_inf())
        );
        let current: Vec<_> = pooled
            .blocks
            .iter()
            .filter_map(|b| match &b.scaling {
                Scaling::Psd(p) => Some(p.schur_values.as_ptr()),
                _ => None,
            })
            .collect();
        assert_eq!(storage, current);
    }
    // Failed numerical update must not prevent replacing/removing the pool
    // and refactoring valid values on the next update.
    let pool = test_shared_pool(4);
    let mut bad = p.clone();
    bad.nzval[0] = T::nan();
    pooled.update_P(&bad);
    assert!(!pooled.update_partition_with_pool(&cones, &settings, true, pool.clone()));
    assert_shared_pool(&pooled, &pool);
    pooled.update_P(&p);
    assert!(pooled.update_partition_with_pool(&cones, &settings, true, None));
    assert_shared_pool(&pooled, &None);
    assert_eq!(
        pooled.reduced.kkt_matrix().nzval[..pooled.schur_nnz],
        serial.reduced.kkt_matrix().nzval[..serial.schur_nnz]
    );
    assert!(pooled.update_partition_with_pool(&cones, &settings, false, pool));
    // The existing/default entry must still follow the cone configuration,
    // rather than retaining a pool supplied by an earlier explicit call.
    assert!(pooled.update(&cones, &settings));
    assert!(pooled.pool.is_none());
}
fn external_pool_cases<T: FloatT>() {
    for blocks in [1, 2] {
        external_ordinary::<T>(blocks);
    }
    for blocks in [1, 2] {
        dominant_sampled_pool_mode::<T>(true, blocks);
    }
}
#[test]
fn external_shared_pool_f64() {
    external_pool_cases::<f64>();
}
#[test]
#[ignore = "extended: MPFR pool/threading sweep; default f64 covers the equivalence logic"]
fn external_shared_pool_mpfr256() {
    external_pool_cases::<Bits256>();
}
#[test]
#[ignore = "extended: MPFR pool/threading sweep; default f64 covers the equivalence logic"]
fn external_shared_pool_mpfr512() {
    external_pool_cases::<sdpx_arithmetic::MpFloat<8>>();
}
