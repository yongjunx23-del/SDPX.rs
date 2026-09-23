use super::super::tests::{runtime, scatter};
use super::*;
use crate::solver::core::traits::Variables;
use crate::solver::core::ScalingStrategy;
use crate::solver::implementations::default::owned_hsd::*;
use sdpx_arithmetic::MpFloat;

fn n<T: FloatT>(v: i32) -> T {
    T::from_i32(v).unwrap()
}
fn close<T: FloatT>(a: T, b: T) {
    let tolerance = n::<T>(4096) * T::epsilon().sqrt();
    assert!(
        (a - b).abs() <= tolerance * T::max(T::one(), b.abs()),
        "{a} != {b}"
    );
}
fn split<T: FloatT>(state: &OwnedSolver<T>, flat: &[T]) -> OwnedVariables<T> {
    let mut result = state.variables.new_like();
    for (rhs, ids) in result.blocks.iter_mut().zip(&state.data.layout.owners) {
        for (v, &col) in rhs.x.iter_mut().zip(&ids.columns) {
            *v = flat[col];
        }
        for (v, &row) in rhs.z.iter_mut().zip(&ids.rows) {
            *v = flat[state.data.layout.n + row];
        }
    }
    result
}

// Exercise the same pair path with a real owner-local pool.  This is kept to
// one mixed-cone f64 fixture so the generic serial checks above remain cheap
// while all supported shared-pool widths still execute the panel code.
fn check_pair_pool_widths<T: FloatT>(make: impl Fn() -> DefaultProblemData<T>) {
    let settings = CoreSettings {
        iterative_refinement_reltol: T::epsilon().sqrt() / n(1000),
        iterative_refinement_abstol: T::epsilon().sqrt() / n(1000),
        ..CoreSettings::<T>::default()
    };
    for workers in [1, 2, 4, 8] {
        let data = make();
        let mut cones = CompositeCone::new(&data.cones);
        let mut state = runtime(
            make(),
            4,
            DefaultSettings {
                max_threads: 1,
                ..DefaultSettings::default()
            },
            (data.n, data.m),
        );
        let mut mono = CondensedKKTSolver::new(&data.P, &data.A, &data.cones, &cones, &settings);
        let pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .build()
                .unwrap(),
        );
        let mut owned = OwnedKkt::new_with_pool(
            &state.data.layout,
            state.data.blocks.iter().zip(&state.cones.blocks),
            &settings,
            Some(pool),
            false,
        );

        let mut point = DefaultVariables::new(data.n, data.m);
        cones.unit_initialization(&mut point.z, &mut point.s);
        scatter(&mut state.variables, &point);
        assert!(cones.update_scaling(&point.s, &point.z, T::one(), ScalingStrategy::PrimalDual));
        assert!(state.variables.scale_cones(
            &mut state.cones,
            T::one(),
            ScalingStrategy::PrimalDual
        ));
        assert!(mono.update(&cones, &settings));
        assert!(owned.update_local(&state.cones.blocks, &settings));

        let known: Vec<T> = (0..data.n + data.m)
            .map(|i| n::<T>(i as i32 + 1) / n(16))
            .collect();
        let zero = vec![T::zero(); known.len()];
        let mut rhs = zero.clone();
        mono.original_residual(&mut rhs, &zero, &known);
        rhs.negate();
        let input = split(&state, &rhs);

        let mut reference = state.variables.new_like();
        assert!(owned.solve_blocks(&input.blocks, &mut reference.blocks, &settings));
        let mut pair0 = state.variables.new_like();
        let mut pair1 = state.variables.new_like();
        let mut product0: Vec<Vec<T>> = (0..owned.locals.len()).map(|_| Vec::new()).collect();
        let mut product1: Vec<Vec<T>> = (0..owned.locals.len()).map(|_| Vec::new()).collect();
        assert_eq!(
            owned.solve_blocks_pair(
                [&input.blocks, &input.blocks],
                [&mut pair0.blocks, &mut pair1.blocks],
                [&mut product0, &mut product1],
                &settings,
            ),
            [true, true]
        );
        for paired in [&pair0, &pair1] {
            for (paired, reference) in paired.blocks.iter().zip(&reference.blocks) {
                for (a, b) in paired.x.iter().zip(&reference.x) {
                    close(*a, *b);
                }
                for (a, b) in paired.z.iter().zip(&reference.z) {
                    close(*a, *b);
                }
            }
        }
        assert!(product0.iter().any(|product| !product.is_empty()));
        assert!(product1.iter().any(|product| !product.is_empty()));

        // A failed affine column must leave the valid constant column and its
        // accepted products usable, while clearing any stale failed-column
        // product from the preceding successful pair.
        let mut bad_second = split(&state, &rhs);
        bad_second
            .blocks
            .iter_mut()
            .find(|block| !block.x.is_empty())
            .unwrap()
            .x[0] = T::nan();
        let mut good_out = state.variables.new_like();
        let mut bad_out = state.variables.new_like();
        let mut good_product = product0.clone();
        let mut bad_product = product1.clone();
        assert_eq!(
            owned.solve_blocks_pair(
                [&input.blocks, &bad_second.blocks],
                [&mut good_out.blocks, &mut bad_out.blocks],
                [&mut good_product, &mut bad_product],
                &settings,
            ),
            [true, false]
        );
        for (good, reference) in good_out.blocks.iter().zip(&reference.blocks) {
            for (a, b) in good.x.iter().zip(&reference.x) {
                close(*a, *b);
            }
            for (a, b) in good.z.iter().zip(&reference.z) {
                close(*a, *b);
            }
        }
        for (before, after) in product0.iter().zip(&good_product) {
            assert_eq!(before.len(), after.len());
            for (a, b) in before.iter().zip(after) {
                close(*a, *b);
            }
        }
        assert!(good_product.iter().any(|product| !product.is_empty()));
        assert!(bad_product.iter().all(Vec::is_empty));

        // A changed valid RHS after the failed affine column must solve from
        // current scratch, rather than reusing the prior successful pair.
        let mut changed = split(&state, &rhs);
        changed
            .blocks
            .iter_mut()
            .find(|block| !block.x.is_empty())
            .unwrap()
            .x[0] += T::one();
        let mut changed_reference = state.variables.new_like();
        assert!(owned.solve_blocks(&changed.blocks, &mut changed_reference.blocks, &settings));
        let mut changed_pair = state.variables.new_like();
        let mut unchanged_pair = state.variables.new_like();
        let mut changed_product: Vec<Vec<T>> =
            (0..owned.locals.len()).map(|_| Vec::new()).collect();
        let mut unchanged_product: Vec<Vec<T>> =
            (0..owned.locals.len()).map(|_| Vec::new()).collect();
        assert_eq!(
            owned.solve_blocks_pair(
                [&changed.blocks, &input.blocks],
                [&mut changed_pair.blocks, &mut unchanged_pair.blocks],
                [&mut changed_product, &mut unchanged_product],
                &settings,
            ),
            [true, true]
        );
        for (paired, reference) in changed_pair.blocks.iter().zip(&changed_reference.blocks) {
            for (a, b) in paired.x.iter().zip(&reference.x) {
                close(*a, *b);
            }
            for (a, b) in paired.z.iter().zip(&reference.z) {
                close(*a, *b);
            }
        }
        assert!(changed_product.iter().any(|product| !product.is_empty()));

        if workers == 1 {
            // Force overflow only in the first lane after the Schur border
            // response correction.  The second lane has a zero RHS and must
            // remain successful with iterative refinement both on and off.
            let row = *state
                .data
                .layout
                .border_rows
                .first()
                .expect("ordinary owner fixture has a shared border");
            let local_row = state.data.layout.owners[0]
                .rows
                .binary_search(&row)
                .unwrap();
            let mut overflow_rhs = state.variables.new_like();
            overflow_rhs.blocks[0].z[local_row] = T::from_f64(1e8).unwrap();
            let zero_rhs = state.variables.new_like();
            {
                let response = owned
                    .locals
                    .iter_mut()
                    .find_map(|local| local.response.first_mut())
                    .expect("ordinary owner fixture has a border response");
                *response = T::from_f64(1e308).unwrap();
            }
            for iterative in [true, false] {
                let mut overflow_settings = settings.clone();
                overflow_settings.iterative_refinement_enable = iterative;
                let mut good_out = state.variables.new_like();
                let mut bad_out = state.variables.new_like();
                for block in &mut bad_out.blocks {
                    block.x.fill(n(77));
                    block.z.fill(n(77));
                }
                let mut good_product: Vec<Vec<T>> =
                    (0..owned.locals.len()).map(|_| Vec::new()).collect();
                let mut bad_product: Vec<Vec<T>> =
                    (0..owned.locals.len()).map(|_| Vec::new()).collect();
                assert_eq!(
                    owned.solve_blocks_pair(
                        [&overflow_rhs.blocks, &zero_rhs.blocks],
                        [&mut bad_out.blocks, &mut good_out.blocks],
                        [&mut bad_product, &mut good_product],
                        &overflow_settings,
                    ),
                    [false, true]
                );
                assert!(bad_out
                    .blocks
                    .iter()
                    .all(|block| block.x.iter().all(|&v| v == n(77))));
                assert!(bad_out
                    .blocks
                    .iter()
                    .all(|block| block.z.iter().all(|&v| v == n(77))));
                assert!(good_out
                    .blocks
                    .iter()
                    .all(|block| block.x.is_finite() && block.z.is_finite()));
                assert!(bad_product.iter().all(Vec::is_empty));
            }
        }
    }
}

fn check<T: FloatT>(make: impl Fn() -> DefaultProblemData<T>) {
    let settings = CoreSettings {
        iterative_refinement_reltol: T::epsilon().sqrt() / n(1000),
        iterative_refinement_abstol: T::epsilon().sqrt() / n(1000),
        ..CoreSettings::<T>::default()
    };
    for count in [1, 2, 8] {
        let data = make();
        let mut cones = CompositeCone::new(&data.cones);
        let mut state = runtime(
            make(),
            count,
            DefaultSettings {
                max_threads: 1,
                ..DefaultSettings::default()
            },
            (data.n, data.m),
        );
        let mut mono = CondensedKKTSolver::new(&data.P, &data.A, &data.cones, &cones, &settings);
        if let Some(sampled) = &data.sampled {
            mono.set_sampled_operator(Arc::clone(sampled));
        }
        let mut owned = OwnedKkt::new_with_pool(
            &state.data.layout,
            state.data.blocks.iter().zip(&state.cones.blocks),
            &settings,
            None,
            false,
        );
        let pointers: Vec<_> = owned.locals.iter().map(|l| l.response.as_ptr()).collect();
        let mut point = DefaultVariables::new(data.n, data.m);
        cones.unit_initialization(&mut point.z, &mut point.s);
        for turn in [1, 2] {
            point.s.scale(n::<T>(turn + 1));
            point.z.scale(n::<T>(turn + 2));
            scatter(&mut state.variables, &point);
            assert!(cones.update_scaling(
                &point.s,
                &point.z,
                T::one(),
                ScalingStrategy::PrimalDual
            ));
            assert!(state.variables.scale_cones(
                &mut state.cones,
                T::one(),
                ScalingStrategy::PrimalDual
            ));
            assert!(mono.update(&cones, &settings));
            assert!(owned.update_local(&state.cones.blocks, &settings));
            // A manufactured solution is checked through the independent
            // monolithic original operator, not the assembled Schur.
            let known: Vec<T> = (0..data.n + data.m)
                .map(|i| n::<T>(i as i32 + 1) / n(16))
                .collect();
            let zero = vec![T::zero(); known.len()];
            let mut rhs = zero.clone();
            mono.original_residual(&mut rhs, &zero, &known);
            rhs.negate();
            let input = split(&state, &rhs);
            let mut output = state.variables.new_like();
            for o in &mut output.blocks {
                o.s.fill(n(71));
                o.τ = n(7);
                o.κ = n(9);
            }
            assert!(owned.solve_blocks(&input.blocks, &mut output.blocks, &settings));
            for (out, ids) in output.blocks.iter().zip(&state.data.layout.owners) {
                assert!(out.s.iter().all(|&v| v == n(71)));
                assert_eq!(out.τ, n(7));
                assert_eq!(out.κ, n(9));
                for (&v, &i) in out.x.iter().zip(&ids.columns) {
                    close(v, known[i]);
                }
                for (&v, &i) in out.z.iter().zip(&ids.rows) {
                    close(v, known[data.n + i]);
                }
            }
            let mut gathered = vec![T::zero(); known.len()];
            for (o, ids) in output.blocks.iter().zip(&state.data.layout.owners) {
                for (&v, &i) in o.x.iter().zip(&ids.columns) {
                    gathered[i] = v;
                }
                for &local in &ids.counted_rows {
                    gathered[data.n + ids.rows[local]] = o.z[local];
                }
            }
            let mut error = zero.clone();
            let residual = mono.original_residual(&mut error, &rhs, &gathered);
            assert!(
                residual <= n::<T>(4096) * T::epsilon().sqrt() * rhs.norm_inf().max(T::one()),
                "residual {residual}"
            );
            // The owner-local pair path must share the already-built
            // factors while retaining the same accepted original
            // residual for each column.
            let mut pair0 = state.variables.new_like();
            let mut pair1 = state.variables.new_like();
            let mut product0: Vec<Vec<T>> = (0..owned.locals.len()).map(|_| Vec::new()).collect();
            let mut product1: Vec<Vec<T>> = (0..owned.locals.len()).map(|_| Vec::new()).collect();
            let pair_ok = owned.solve_blocks_pair(
                [&input.blocks, &input.blocks],
                [&mut pair0.blocks, &mut pair1.blocks],
                [&mut product0, &mut product1],
                &settings,
            );
            assert_eq!(pair_ok, [true, true]);
            for paired in [&pair0, &pair1] {
                for (paired, reference) in paired.blocks.iter().zip(&output.blocks) {
                    for (a, b) in paired.x.iter().zip(&reference.x) {
                        close(*a, *b);
                    }
                    for (a, b) in paired.z.iter().zip(&reference.z) {
                        close(*a, *b);
                    }
                }
            }
            // A malformed constant column must not suppress the affine
            // column or leak a stale accepted-product cache into it.
            let mut bad_pair = split(&state, &rhs);
            bad_pair
                .blocks
                .iter_mut()
                .find(|b| !b.x.is_empty())
                .unwrap()
                .x[0] = T::nan();
            let mut bad_out = state.variables.new_like();
            let mut good_out = state.variables.new_like();
            let mut bad_product: Vec<Vec<T>> =
                (0..owned.locals.len()).map(|_| Vec::new()).collect();
            let mut good_product: Vec<Vec<T>> =
                (0..owned.locals.len()).map(|_| Vec::new()).collect();
            let isolated = owned.solve_blocks_pair(
                [&bad_pair.blocks, &input.blocks],
                [&mut bad_out.blocks, &mut good_out.blocks],
                [&mut bad_product, &mut good_product],
                &settings,
            );
            assert_eq!(isolated, [false, true]);
            for (good, reference) in good_out.blocks.iter().zip(&output.blocks) {
                for (a, b) in good.x.iter().zip(&reference.x) {
                    close(*a, *b);
                }
                for (a, b) in good.z.iter().zip(&reference.z) {
                    close(*a, *b);
                }
            }
            // Malformed numerical input cannot publish successful outputs;
            // the next independent solve reuses the same factors/buffers.
            let mut bad = split(&state, &rhs);
            bad.blocks.iter_mut().find(|b| !b.x.is_empty()).unwrap().x[0] = T::nan();
            assert!(!owned.solve_blocks(&bad.blocks, &mut output.blocks, &settings));
            assert!(owned.scaled_product(0).is_none());
            assert!(owned.solve_blocks(&input.blocks, &mut output.blocks, &settings));
        }
        assert_eq!(
            pointers,
            owned
                .locals
                .iter()
                .map(|l| l.response.as_ptr())
                .collect::<Vec<_>>()
        );
        for _ in 0..3 {
            assert!(owned.escalate_regularization());
        }
        assert!(!owned.escalate_regularization());
        owned.reset_solve();
        assert_eq!(owned.reg_boost, 0);
    }
}
fn no_border<T: FloatT>() -> DefaultProblemData<T> {
    let settings = DefaultSettings {
        presolve_enable: false,
        equilibrate_enable: false,
        chordal_decomposition_enable: false,
        ..DefaultSettings::default()
    };
    DefaultProblemData::new(
        &CscMatrix::identity(2),
        &[T::one(); 2],
        &CscMatrix::identity(2),
        &[T::one(); 2],
        &[SupportedConeT::NonnegativeConeT(2)],
        &settings,
    )
}

fn gather<T: FloatT>(state: &OwnedSolver<T>, values: &OwnedVariables<T>) -> Vec<T> {
    let mut gathered = vec![T::zero(); state.data.layout.n + state.data.layout.m];
    for (block, ids) in values.blocks.iter().zip(&state.data.layout.owners) {
        for (&value, &column) in block.x.iter().zip(&ids.columns) {
            gathered[column] = value;
        }
        for &local in &ids.counted_rows {
            gathered[state.data.layout.n + ids.rows[local]] = block.z[local];
        }
    }
    gathered
}

// Distinct sampled PSD RHS columns exercise the per-column NT workspace
// snapshots. Run both refinement modes at one shared-pool width for each
// scalar family; the original operator residual remains the audit oracle.
fn sampled_pair_cache_case<T: FloatT>() {
    for iterative in [false, true] {
        let refinement_tol = if iterative {
            T::epsilon() * T::epsilon()
        } else {
            T::epsilon().sqrt() / n(1000)
        };
        let settings = CoreSettings {
            iterative_refinement_enable: iterative,
            iterative_refinement_reltol: refinement_tol,
            iterative_refinement_abstol: refinement_tol,
            static_regularization_constant: T::epsilon(),
            static_regularization_proportional: T::zero(),
            iterative_refinement_max_iter: 10,
            ..CoreSettings::<T>::default()
        };
        let data = super::super::tests::sampled::<T>();
        let mut cones = CompositeCone::new(&data.cones);
        let mut state = runtime(
            super::super::tests::sampled::<T>(),
            4,
            DefaultSettings {
                max_threads: 1,
                ..DefaultSettings::default()
            },
            (data.n, data.m),
        );
        let mut mono = CondensedKKTSolver::new(&data.P, &data.A, &data.cones, &cones, &settings);
        if let Some(sampled) = &data.sampled {
            mono.set_sampled_operator(Arc::clone(sampled));
        }
        let pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(2)
                .build()
                .unwrap(),
        );
        let mut pair_solver = OwnedKkt::new_with_pool(
            &state.data.layout,
            state.data.blocks.iter().zip(&state.cones.blocks),
            &settings,
            Some(pool),
            false,
        );
        let mut scalar_solver = OwnedKkt::new_with_pool(
            &state.data.layout,
            state.data.blocks.iter().zip(&state.cones.blocks),
            &settings,
            None,
            false,
        );
        let mut point = DefaultVariables::new(data.n, data.m);
        cones.unit_initialization(&mut point.z, &mut point.s);
        scatter(&mut state.variables, &point);
        assert!(cones.update_scaling(&point.s, &point.z, T::one(), ScalingStrategy::PrimalDual));
        assert!(state.variables.scale_cones(
            &mut state.cones,
            T::one(),
            ScalingStrategy::PrimalDual
        ));
        assert!(mono.update(&cones, &settings));
        assert!(pair_solver.update_local(&state.cones.blocks, &settings));
        assert!(scalar_solver.update_local(&state.cones.blocks, &settings));

        let zero = vec![T::zero(); data.n + data.m];
        let known0: Vec<T> = (0..data.n + data.m)
            .map(|i| n::<T>(i as i32 + 1) / n(16))
            .collect();
        let known1: Vec<T> = (0..data.n + data.m)
            .map(|i| n::<T>(3 * i as i32 + 5) / n(11))
            .collect();
        let mut rhs0 = zero.clone();
        let mut rhs1 = zero.clone();
        mono.original_residual(&mut rhs0, &zero, &known0);
        mono.original_residual(&mut rhs1, &zero, &known1);
        rhs0.negate();
        rhs1.negate();
        let input0 = split(&state, &rhs0);
        let input1 = split(&state, &rhs1);

        let mut pair0 = state.variables.new_like();
        let mut pair1 = state.variables.new_like();
        let mut pair_product0: Vec<Vec<T>> =
            (0..pair_solver.locals.len()).map(|_| Vec::new()).collect();
        let mut pair_product1: Vec<Vec<T>> =
            (0..pair_solver.locals.len()).map(|_| Vec::new()).collect();
        assert_eq!(
            pair_solver.solve_blocks_pair(
                [&input0.blocks, &input1.blocks],
                [&mut pair0.blocks, &mut pair1.blocks],
                [&mut pair_product0, &mut pair_product1],
                &settings,
            ),
            [true, true]
        );
        if iterative {
            // The nonzero static shift and very tight tolerance force the
            // reduced refinement stage to perform a correction.  Snapshots
            // remain distinct after that stage, before either outer lane's
            // nested scalar correction can overwrite the live mat3c.
            assert!(pair_solver.shift > T::zero());
            assert!(pair_solver.refinements >= 4);
            let mut distinct = false;
            for local in &pair_solver.locals {
                if local.rhs_cache[0].len() == local.rhs_cache[1].len()
                    && local.rhs_cache[0]
                        .iter()
                        .zip(&local.rhs_cache[1])
                        .any(|(a, b)| a != b)
                {
                    distinct = true;
                    break;
                }
            }
            assert!(distinct, "sampled RHS snapshots unexpectedly equal");
        }

        let mut scalar0 = state.variables.new_like();
        let mut scalar1 = state.variables.new_like();
        assert!(scalar_solver.solve_blocks(&input0.blocks, &mut scalar0.blocks, &settings));
        assert!(scalar_solver.solve_blocks(&input1.blocks, &mut scalar1.blocks, &settings));
        for (paired, reference) in pair0.blocks.iter().zip(&scalar0.blocks) {
            for (a, b) in paired.x.iter().zip(&reference.x) {
                close(*a, *b);
            }
            for (a, b) in paired.z.iter().zip(&reference.z) {
                close(*a, *b);
            }
        }
        for (paired, reference) in pair1.blocks.iter().zip(&scalar1.blocks) {
            for (a, b) in paired.x.iter().zip(&reference.x) {
                close(*a, *b);
            }
            for (a, b) in paired.z.iter().zip(&reference.z) {
                close(*a, *b);
            }
        }
        if iterative {
            assert!(pair_product0.iter().any(|product| !product.is_empty()));
            assert!(pair_product1.iter().any(|product| !product.is_empty()));
        } else {
            assert!(pair_product0.iter().all(Vec::is_empty));
            assert!(pair_product1.iter().all(Vec::is_empty));
        }

        let mut error = vec![T::zero(); data.n + data.m];
        for (rhs, output) in [(&rhs0, &pair0), (&rhs1, &pair1)] {
            let gathered = gather(&state, output);
            let residual = mono.original_residual(&mut error, rhs, &gathered);
            assert!(
                residual <= n::<T>(16384) * T::epsilon().sqrt() * rhs.norm_inf().max(T::one()),
                "sampled pair residual {residual}"
            );
        }
    }
}

#[test]
fn owned_kkt_f64() {
    check(super::super::tests::ordinary::<f64>);
    check(super::super::tests::sampled::<f64>);
    check(no_border::<f64>);
    check_pair_pool_widths(super::super::tests::ordinary::<f64>);
    sampled_pair_cache_case::<f64>();
}
#[test]
fn owned_kkt_mpfr256() {
    check(super::super::tests::ordinary::<MpFloat<4>>);
    check(super::super::tests::sampled::<MpFloat<4>>);
    check(no_border::<MpFloat<4>>);
    sampled_pair_cache_case::<MpFloat<4>>();
}
#[test]
fn owned_kkt_mpfr512() {
    check(super::super::tests::ordinary::<MpFloat<8>>);
    check(super::super::tests::sampled::<MpFloat<8>>);
    check(no_border::<MpFloat<8>>);
    sampled_pair_cache_case::<MpFloat<8>>();
}

#[test]
#[ignore = "requires frozen accepted Ising input/point; linear-direction diagnostic, no HSD solve"]
fn owned_ising_linear_direction() {
    use crate::solver::core::traits::{Residuals, Variables};
    use num_traits::{FromPrimitive, One, Zero};
    use sdpx_arithmetic::Scalar;
    use serde_json::{json, Value};
    use std::path::PathBuf;
    type T = MpFloat<8>;
    let input = PathBuf::from(std::env::var_os("SDPX_OWNER_TEST_INPUT").unwrap());
    let raw: Value = serde_json::from_slice(
        &std::fs::read(std::env::var_os("SDPX_OWNER_TEST_POINT").unwrap()).unwrap(),
    )
    .unwrap();
    let out = PathBuf::from(std::env::var_os("SDPX_OWNER_TEST_OUTPUT").unwrap());
    std::fs::create_dir_all(&out).unwrap();
    assert_eq!(raw["precision_bits"], 512);
    assert_eq!(raw["status"], "Solved");
    let settings: DefaultSettings<T> = serde_json::from_value(raw["settings"].clone()).unwrap();
    let make = || {
        let mut problem = read_sdpb_sampled::<T>(&input).unwrap().problem;
        problem.settings = settings.clone();
        problem.into_solver().unwrap()
    };
    let mut full = make();
    assert!(full.data.presolver.is_none() && full.data.chordal_info.is_none());
    let x: Vec<T> = serde_json::from_value(raw["x"].clone()).unwrap();
    let s: Vec<T> = serde_json::from_value(raw["s"].clone()).unwrap();
    let z: Vec<T> = serde_json::from_value(raw["z"].clone()).unwrap();
    let eq = &full.data.equilibration;
    let point = DefaultVariables {
        x: x.iter().zip(&eq.dinv).map(|(&x, &d)| x * d).collect(),
        s: s.iter().zip(&eq.e).map(|(&s, &e)| s * e).collect(),
        z: z.iter()
            .zip(&eq.einv)
            .map(|(&z, &e)| z * eq.c * e)
            .collect(),
        τ: T::one(),
        κ: T::zero(),
    };
    let mu = (point.z.dot(&point.s) + point.τ * point.κ)
        / T::from_usize(full.cones.degree() + 1).unwrap();
    assert!(point.scale_cones(&mut full.cones, mu, ScalingStrategy::PrimalDual));
    let d = &full.data;
    let mut reference = CondensedKKTSolver::new(&d.P, &d.A, &d.cones, &full.cones, &settings);
    reference.set_sampled_operator(Arc::clone(d.sampled.as_ref().unwrap()));
    assert!(reference.update(&full.cones, &settings));
    let mut residual = DefaultResiduals::new(d.n, d.m);
    residual.update(&point, d);
    let mut affine = DefaultVariables::new(d.n, d.m);
    affine.affine_step_rhs(&residual, &point, &full.cones);
    let mut rhs = affine.x.clone();
    rhs.extend(point.s.iter().zip(&affine.z).map(|(&s, &z)| s - z));
    let raw_rhs = rhs.clone();
    let mut reference_point = vec![T::zero(); d.n + d.m];
    reference.setrhs(&rhs[..d.n], &rhs[d.n..]);
    let (rx, rz) = reference_point.split_at_mut(d.n);
    assert!(reference.solve(Some(rx), Some(rz), &settings));
    let mut error = vec![T::zero(); d.n + d.m];
    let reference_residual = reference.original_residual(&mut error, &rhs, &reference_point)
        / rhs.norm_inf().max(T::one());
    let gate: T = "1e-30".parse().unwrap();
    let mut rows = vec![json!({"kind":"monolithic","residual_relative":reference_residual})];
    std::fs::write(
        out.join("linear.json"),
        serde_json::to_vec_pretty(&rows).unwrap(),
    )
    .unwrap();
    assert!(reference_residual <= gate);
    for count in [1, 2, 4, 8] {
        let mut state = runtime(make().data, count, settings.clone(), (d.n, d.m));
        scatter(&mut state.variables, &point);
        assert!(state
            .variables
            .scale_cones(&mut state.cones, mu, ScalingStrategy::PrimalDual));
        let mut kernel = OwnedKkt::new_with_pool(
            &state.data.layout,
            state.data.blocks.iter().zip(&state.cones.blocks),
            &settings,
            None,
            false,
        );
        assert!(kernel.update_local(&state.cones.blocks, &settings));
        let b = split(&state, &raw_rhs);
        for (local, ids) in b.blocks.iter().zip(&state.data.layout.owners) {
            for (&value, &row) in local.z.iter().zip(&ids.rows) {
                assert_eq!(value, rhs[d.n + row]);
            }
        }
        let mut sol = state.variables.new_like();
        assert!(kernel.solve_blocks(&b.blocks, &mut sol.blocks, &settings));
        let mut gathered = vec![T::zero(); d.n + d.m];
        for (o, ids) in sol.blocks.iter().zip(&state.data.layout.owners) {
            for (&v, &i) in o.x.iter().zip(&ids.columns) {
                gathered[i] = v;
            }
            for &j in &ids.counted_rows {
                gathered[d.n + ids.rows[j]] = o.z[j];
            }
        }
        let norm =
            reference.original_residual(&mut error, &rhs, &gathered) / rhs.norm_inf().max(T::one());
        rows.push(json!({"kind":"owned","owners":count,"residual_relative":norm,
            "interior_dimensions":kernel.locals.iter().map(|l|l.kernel.interior_dimension()).collect::<Vec<_>>(),
            "border":kernel.border_matrix.n,"shift":kernel.shift,
            "scope":"affine linear direction at fixed accepted point; not an owned HSD solve"}));
        std::fs::write(
            out.join("linear.json"),
            serde_json::to_vec_pretty(&rows).unwrap(),
        )
        .unwrap();
        assert!(
            norm <= gate,
            "owners {count}: relative original KKT residual {norm}"
        );
    }
}
