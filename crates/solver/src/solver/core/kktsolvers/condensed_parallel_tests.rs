use super::*;
use crate::solver::core::ScalingStrategy;
use sdpx_arithmetic::Bits256;

fn num<T: FloatT>(n: usize) -> T {
    T::from_usize(n).unwrap()
}
fn wide_pool_tiles() -> usize {
    use std::sync::atomic::Ordering;
    crate::solver::core::kktsolvers::condensed::POOLED_CONGRUENCE_TILES.load(Ordering::Relaxed)
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
    assert_eq!(pooled.parallel_assembly, !overlap_fallback);
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
    assert!(buffer_cells(&pooled) <= 2 * pooled.schur.nnz());
    assert_eq!(buffer_cells(&pooled) == 0, overlap_fallback);
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
        assert_eq!(pooled.parallel_assembly, width > 1 && !overlap_fallback);
        assert_eq!(buffer_cells(&pooled), saved_buffers);
        assert_eq!(serial.schur.colptr, pooled.schur.colptr);
        assert_eq!(serial.schur.rowval, pooled.schur.rowval);
        assert_eq!(serial.schur.nzval, pooled.schur.nzval);
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
                    if !matches!(block.scaling, Scaling::Psd(_) | Scaling::Orthant { .. }) {
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
    assert_eq!(serial.schur.nzval, pooled.schur.nzval);
}

#[test]
fn pooled_condensed_f64() {
    pooled_equivalence::<f64>(false);
}
#[test]
fn pooled_condensed_mpfr256() {
    pooled_equivalence::<Bits256>(false);
}
#[test]
fn pooled_condensed_mpfr512() {
    pooled_equivalence::<sdpx_arithmetic::Bits512>(false);
}
#[test]
fn overlapping_memory_fallback_mpfr512() {
    pooled_equivalence::<sdpx_arithmetic::Bits512>(true);
}
#[test]
fn overlapping_memory_fallback_f64() {
    pooled_equivalence::<f64>(true);
}
#[test]
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
    let position = schur_position(&solver.schur, 0, 1);
    assert_eq!(solver.schur.nzval[position], -delta * delta);
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
fn wide_pool_splits_congruence_tiles_mpfr256() {
    wide_pool_splits_congruence_tiles::<Bits256>();
}

#[test]
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
        12usize, 12, 13, 12, 14, 13, 15, 14, 16, 15, 16, 15, 16, 15, 16, 15, 16, 15, 16, 15, 16,
        15,
    ];
    let costs: Vec<u128> = dims.iter().map(|d| 4 * (*d as u128).pow(3)).collect();
    let total: u128 = costs.iter().sum();

    // The chooser may not use more than four lanes per worker, but within that
    // bound it must never be worse than the worker-count partition it replaces,
    // and at the widths that matter it uses every block as its own lane.
    for workers in [2usize, 4, 8, 16] {
        let (lanes, _) = scaling_dispatch(&costs, workers);
        assert!(lanes.len() <= costs.len().min(workers * 4), "workers={workers}");
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
    let costs = [4 * 64u128.pow(3), 4 * 8u128.pow(3), 4 * 8u128.pow(3), 4 * 8u128.pow(3)];
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
        assert_eq!(serial.schur.nzval, pooled.schur.nzval);
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
        assert_eq!(serial.schur.nzval, pooled.schur.nzval);
    }
}

#[test]
fn dominant_sparse_psd_f64() {
    dominant_sparse_psd::<f64>();
}

#[test]
fn dominant_sparse_psd_mpfr256() {
    dominant_sparse_psd::<Bits256>();
}

fn dominant_sampled_psd<T: FloatT>() {
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
    let (m, n) = (block.row_count(), block.column_count());
    let operator = Arc::new(SampledOperator::new(CscMatrix::zeros((m, n)), vec![block]).unwrap());
    let a = operator.materialize();
    let p = CscMatrix::identity(n);
    let kinds = vec![SupportedConeT::PSDTriangleConeT(48)];
    let mut serial_cones = CompositeCone::new(&kinds);
    let mut pooled_cones = CompositeCone::new(&kinds);
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
    operator.apply(&mut bz, &exact_x, T::one(), -T::one(), &mut operator_work);
    for workers in [1, 2, 4, 8, 1, 4] {
        pooled_cones.configure_threads(workers).unwrap();
        assert!(serial.update(&serial_cones, &settings));
        assert!(pooled.update(&pooled_cones, &settings));
        assert_eq!(pooled.plan_threads, workers);
        assert_eq!(
            pooled.inner_sampled,
            if workers > 1 { Some(0) } else { None }
        );
        assert_eq!(pooled.parallel_assembly, workers > 1);
        assert_eq!(serial.schur.nzval, pooled.schur.nzval);
        let Scaling::Psd(psd) = &pooled.blocks[0].scaling else {
            unreachable!()
        };
        let sampled = psd.sampled.as_ref().unwrap();
        if workers > 1 {
            assert!(sampled.pair_lanes.len() > 1);
        }
        let pointers = (sampled.pair_lanes.as_ptr(), psd.schur_values.as_ptr());
        assert!(pooled.update(&pooled_cones, &settings));
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
        let (mut xs, mut xp) = (vec![T::zero(); n], vec![T::zero(); n]);
        let (mut zs, mut zp) = (vec![T::zero(); m], vec![T::zero(); m]);
        serial.setrhs(&bx, &bz);
        pooled.setrhs(&bx, &bz);
        assert!(serial.solve(Some(&mut xs), Some(&mut zs), &settings));
        assert!(pooled.solve(Some(&mut xp), Some(&mut zp), &settings));
        assert_eq!(xs, xp);
        assert_eq!(zs, zp);
        // External full augmented equations, using factor-authoritative A.
        let mut ex = bx.clone();
        for (e, x) in ex.iter_mut().zip(&xp) {
            *e -= *x;
        }
        operator.apply_transpose(&mut ex, &zp, -T::one(), T::one(), &mut operator_work);
        let mut ez = bz.clone();
        for (e, z) in ez.iter_mut().zip(&zp) {
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
fn dominant_sampled_psd_mpfr256() {
    dominant_sampled_psd::<Bits256>();
}
#[test]
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
        let mut expected = vec![T::zero(); solver.schur.nzval.len()];
        for block in &solver.blocks {
            if let Scaling::Orthant { w, rows, .. } = &block.scaling {
                for (r, entries) in rows.iter().enumerate() {
                    for (b, &(j, q)) in entries.iter().enumerate() {
                        let aj = solver.A.nzval[q] / w[r];
                        for &(i, k) in &entries[..=b] {
                            let ai = solver.A.nzval[k] / w[r];
                            let pos = schur_position(&solver.schur, i, j);
                            expected[pos] = ai.mul_add(aj, expected[pos]);
                        }
                    }
                }
            }
        }
        assert!(solver.assemble());
        assert_eq!(solver.schur.nzval, expected);
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
    use crate::solver::core::kktsolvers::SolveCounters;
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
    let counters = crate::solver::core::kktsolvers::KKTSolver::counters(&solver);
    assert_eq!(
        counters,
        SolveCounters {
            factorizations: 1,
            rhs_applied: 0,
            batches: 0
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

    // Every column must match the single-RHS path bit for bit.
    for c in 0..ncols {
        let base = c * width;
        let mut single = vec![T::zero(); width];
        let (xl, zl) = single.split_at_mut(n);
        KKTSolver::setrhs(&mut solver, &rhs[base..base + n], &rhs[base + n..base + width]);
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
    assert_eq!(flags, vec![false, true, true], "failure must stay local to its column");
}

#[test]
fn solve_many_accounting_f64() {
    solve_many_accounting::<f64>();
}

#[test]
fn solve_many_accounting_mpfr256() {
    solve_many_accounting::<Bits256>();
}

#[test]
fn solve_many_accounting_mpfr512() {
    solve_many_accounting::<sdpx_arithmetic::Bits512>();
}

