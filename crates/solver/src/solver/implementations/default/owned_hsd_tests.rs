use super::*;
use crate::collective::MockCollective;
use crate::solver::core::cones::{Cone, PrimalOrDualCone};
use crate::solver::{IPSolver, SolverStatus};
use sdpx_arithmetic::MpFloat;
use std::sync::Arc;
use std::thread;

fn n<T: FloatT>(v: i32) -> T {
    T::from_i32(v).unwrap()
}
fn settings<T: FloatT>() -> DefaultSettings<T> {
    let tol = if T::precision_bits() <= 53 {
        T::from_f64(1e-9).unwrap()
    } else {
        T::from_f64(1e-35).unwrap()
    };
    DefaultSettings {
        verbose: false,
        max_iter: 100,
        presolve_enable: false,
        chordal_decomposition_enable: false,
        tol_feas: tol,
        tol_gap_abs: tol,
        tol_gap_rel: tol,
        tol_feas_componentwise: Some(tol),
        ..DefaultSettings::default()
    }
}
struct ProblemFixture<T: FloatT> {
    p: CscMatrix<T>,
    q: Vec<T>,
    a: CscMatrix<T>,
    b: Vec<T>,
    cones: Vec<SupportedConeT<T>>,
    solver: DefaultSolver<T>,
}

fn problem<T: FloatT>(kind: usize) -> ProblemFixture<T> {
    let (a, b, cones) = if kind < 2 {
        (
            CscMatrix::from(&[[n(1), n(1)], [n(-1), n(0)], [n(0), n(-1)]]),
            vec![n(1), n(0), n(0)],
            vec![
                SupportedConeT::ZeroConeT(1),
                SupportedConeT::NonnegativeConeT(2),
            ],
        )
    } else {
        let a = CscMatrix::from(&[
            [n(1), n(1)],
            [n(-1), n(0)],
            [n(0), n(0)],
            [n(0), n(0)],
            [n(0), n(-1)],
            [n(0), n(0)],
            [n(0), n(0)],
        ]);
        let cone = if kind == 2 {
            SupportedConeT::SecondOrderConeT(3)
        } else {
            SupportedConeT::PSDTriangleConeT(2)
        };
        let b = if kind == 2 {
            vec![n(3), n(0), n(1), n(0), n(0), n(1), n(0)]
        } else {
            vec![n(1), n(0), n(0), n(1), n(0), n(0), n(1)]
        };
        (a, b, vec![SupportedConeT::ZeroConeT(1), cone.clone(), cone])
    };
    let p = if kind == 1 {
        CscMatrix::identity(2)
    } else {
        CscMatrix::zeros((2, 2))
    };
    let q = if kind == 1 {
        vec![T::zero(); 2]
    } else {
        vec![n(1), n(2)]
    };
    let solver = DefaultSolver::new(&p, &q, &a, &b, &cones, settings()).unwrap();
    ProblemFixture {
        p,
        q,
        a,
        b,
        cones,
        solver,
    }
}

fn cone_margin<T: FloatT>(cones: &[SupportedConeT<T>], values: &[T], pd: PrimalOrDualCone) -> T {
    let mut composite = CompositeCone::new(cones);
    let mut values = values.to_vec();
    composite.margins(&mut values, pd).0
}

fn assert_original_solution<T: FloatT>(
    p: &CscMatrix<T>,
    q: &[T],
    a: &CscMatrix<T>,
    b: &[T],
    cones: &[SupportedConeT<T>],
    x: &[T],
    s: &[T],
    z: &[T],
    gate: T,
    context: &str,
) {
    assert_eq!(x.len(), p.n, "{context}: x length");
    assert_eq!(s.len(), a.m, "{context}: s length");
    assert_eq!(z.len(), a.m, "{context}: z length");

    let mut primal = s.to_vec();
    a.gemv(&mut primal, x, T::one(), T::one());
    for (value, &rhs) in primal.iter_mut().zip(b) {
        *value -= rhs;
    }
    let mut dual = q.to_vec();
    a.t().gemv(&mut dual, z, T::one(), T::one());
    p.sym_up().symv(&mut dual, x, T::one(), T::one());
    let primal_gate = gate * T::max(T::one(), b.norm_inf());
    let dual_gate = gate * T::max(T::one(), q.norm_inf());
    assert!(
        primal.norm_inf() <= primal_gate,
        "{context}: A*x+s-b={} > {primal_gate}",
        primal.norm_inf()
    );
    assert!(
        dual.norm_inf() <= dual_gate,
        "{context}: P*x+q+A^T*z={} > {dual_gate}",
        dual.norm_inf()
    );

    let primal_margin = cone_margin(cones, s, PrimalOrDualCone::PrimalCone);
    let dual_margin = cone_margin(cones, z, PrimalOrDualCone::DualCone);
    let slack_gate = gate * T::max(T::one(), s.norm_inf());
    let dual_cone_gate = gate * T::max(T::one(), z.norm_inf());
    assert!(
        primal_margin >= -slack_gate,
        "{context}: primal cone margin={primal_margin}"
    );
    assert!(
        dual_margin >= -dual_cone_gate,
        "{context}: dual cone margin={dual_margin}"
    );

    let quadratic = p.sym_up().quad_form(x, x);
    let primal_objective = quadratic / n(2) + q.dot(x);
    let dual_objective = -quadratic / n(2) - b.dot(z);
    let gap = (primal_objective - dual_objective).abs();
    let gap_gate = gate
        * T::max(
            T::one(),
            T::min(primal_objective.abs(), dual_objective.abs()),
        );
    assert!(
        gap <= gap_gate,
        "{context}: objective gap={gap} > {gap_gate} (p={primal_objective}, d={dual_objective})"
    );
}

fn equality_only<T: FloatT>() -> DefaultSolver<T> {
    let p = CscMatrix::zeros((0, 0));
    let a = CscMatrix::zeros((1, 0));
    let b = vec![T::zero()];
    DefaultSolver::new(&p, &[], &a, &b, &[SupportedConeT::ZeroConeT(1)], settings()).unwrap()
}
fn full_solve<T: FloatT>() {
    let gate = if T::precision_bits() <= 53 {
        T::from_f64(1e-7).unwrap()
    } else {
        T::from_f64(1e-30).unwrap()
    };
    for kind in 0..4 {
        let mut reference = problem::<T>(kind).solver;
        reference.solve();
        assert_eq!(reference.solution.status, SolverStatus::Solved);
        for count in [1, 2, 6] {
            let mut owned = OwnedSolver::from_default(problem::<T>(kind).solver, count).unwrap();
            owned.solve();
            assert_eq!(
                owned.solution.0.status,
                SolverStatus::Solved,
                "kind {kind}, owners {count}"
            );
            let out = &owned.solution.0;
            for (&x, &r) in out.x.iter().zip(&reference.solution.x) {
                assert!((x - r).abs() <= gate, "x {x} vs {r}");
            }
            assert!((out.obj_val - reference.solution.obj_val).abs() <= gate);
            assert!(out.r_prim <= gate && out.r_dual <= gate);
            let expected = if kind == 1 {
                T::one() / n(4)
            } else if kind == 2 {
                n(4)
            } else {
                T::one()
            };
            assert!((out.obj_val - expected).abs() <= gate);
        }
    }
    for infeasible in [false, true] {
        let make = || {
            let a = if infeasible {
                CscMatrix::from(&[[n::<T>(-1)], [n(1)]])
            } else {
                CscMatrix::from(&[[n(-1)]])
            };
            let b = if infeasible {
                vec![n(-1), T::zero()]
            } else {
                vec![T::zero()]
            };
            DefaultSolver::new(
                &CscMatrix::zeros((1, 1)),
                &[n(-1)],
                &a,
                &b,
                &[SupportedConeT::NonnegativeConeT(b.len())],
                settings(),
            )
            .unwrap()
        };
        let mut reference = make();
        reference.solve();
        for count in [1, 4] {
            let mut owned = OwnedSolver::from_default(make(), count).unwrap();
            owned.solve();
            assert_eq!(owned.solution.0.status, reference.solution.status);
            assert_eq!(
                owned.solution.0.status,
                if infeasible {
                    SolverStatus::PrimalInfeasible
                } else {
                    SolverStatus::DualInfeasible
                }
            );
        }
    }
}

fn rank_local_mock_matches_serial<T: FloatT>(kind: usize) {
    let gate = if T::precision_bits() <= 53 {
        T::from_f64(1e-7).unwrap()
    } else {
        T::from_f64(1e-30).unwrap()
    };
    let ProblemFixture {
        p,
        q,
        a,
        b,
        cones,
        solver: mut reference,
    } = problem::<T>(kind);
    reference.solve();
    assert_eq!(reference.solution.status, SolverStatus::Solved);
    let expected = reference.solution.x.clone();
    let objective = reference.solution.obj_val;

    // Three ranks for this two-variable/equality problem leave at least one
    // owner with no local numeric block. Every rank still participates in the
    // same HSD/KKT collectives and only rank zero gathers the final point.
    let handles = MockCollective::<T>::group(3).unwrap();
    let mut joins = Vec::with_capacity(handles.len());
    for comm in handles {
        joins.push(thread::spawn(move || {
            let prepared = problem::<T>(kind).solver;
            let DefaultSolver {
                data,
                cones,
                settings,
                solution,
                timers,
                ..
            } = prepared;
            let prepared = PreparedProblem {
                data,
                cones,
                settings,
                solution,
                timers: timers.unwrap(),
                cost_input_fingerprint: None,
            };
            let rank = comm.rank();
            let mut solver = OwnedSolver::from_prepared_rank_local(prepared, Arc::new(comm))
                .expect("rank-local construction");
            let local_blocks = solver.data.blocks.len();
            solver.solve();
            (
                rank,
                local_blocks,
                solver.solution.0.status,
                solver.solution.0.x,
                solver.solution.0.s,
                solver.solution.0.z,
                solver.solution.0.obj_val,
                solver.solution.0.r_prim,
                solver.solution.0.r_dual,
            )
        }));
    }
    let mut results = Vec::new();
    for join in joins {
        results.push(join.join().expect("rank-local mock solve panicked"));
    }
    results.sort_by_key(|result| result.0);

    assert!(results.iter().any(|result| result.1 == 0));
    for (rank, _, status, x, s, z, obj, r_prim, r_dual) in results {
        assert_eq!(status, SolverStatus::Solved, "kind {kind}, rank {rank}");
        assert!(
            r_prim <= gate,
            "kind {kind}, rank {rank} primal residual {r_prim}"
        );
        assert!(
            r_dual <= gate,
            "kind {kind}, rank {rank} dual residual {r_dual}"
        );
        if rank == 0 {
            assert_original_solution(
                &p,
                &q,
                &a,
                &b,
                &cones,
                &x,
                &s,
                &z,
                gate,
                &format!("kind {kind}, rank {rank}"),
            );
            for (&actual, &expected) in x.iter().zip(&expected) {
                assert!((actual - expected).abs() <= gate);
            }
            assert!((obj - objective).abs() <= gate);
        } else {
            // Non-root ranks intentionally do not receive the gathered point
            // or retain any output allocation.
            assert!(x.is_empty() && s.is_empty() && z.is_empty());
            assert_eq!(x.capacity(), 0);
            assert_eq!(s.capacity(), 0);
            assert_eq!(z.capacity(), 0);
        }
    }
}

#[test]
fn owned_hsd_rank_local_mock_matches_serial_with_empty_owner() {
    for kind in 0..4 {
        rank_local_mock_matches_serial::<f64>(kind);
    }
}

#[test]
fn owned_hsd_rank_local_mock_mpfr256_matches_serial_with_empty_owner() {
    for kind in 0..4 {
        rank_local_mock_matches_serial::<MpFloat<4>>(kind);
    }
}

#[test]
fn owned_hsd_rank_local_mock_equality_only_empty_numeric_ranks() {
    let mut reference = equality_only::<f64>();
    reference.solve();
    assert_eq!(reference.solution.status, SolverStatus::Solved);
    let handles = MockCollective::<f64>::group(2).unwrap();
    let mut joins = Vec::with_capacity(handles.len());
    for comm in handles {
        joins.push(thread::spawn(move || {
            let prepared = equality_only::<f64>();
            let DefaultSolver {
                data,
                cones,
                settings,
                solution,
                timers,
                ..
            } = prepared;
            let prepared = PreparedProblem {
                data,
                cones,
                settings,
                solution,
                timers: timers.unwrap(),
                cost_input_fingerprint: None,
            };
            let mut solver = OwnedSolver::from_prepared_rank_local(prepared, Arc::new(comm))
                .expect("rank-local equality-only construction");
            let local_blocks = solver.data.blocks.len();
            let rank = solver.variables.collective.rank();
            solver.solve();
            (
                rank,
                local_blocks,
                solver.solution.0.status,
                solver.solution.0.x,
                solver.solution.0.s,
                solver.solution.0.z,
            )
        }));
    }
    let results: Vec<_> = joins
        .into_iter()
        .map(|join| {
            join.join()
                .expect("rank-local equality-only solve panicked")
        })
        .collect();
    assert!(results.iter().all(|result| result.1 == 0));
    for (rank, _, status, x, s, z) in results {
        assert_eq!(status, SolverStatus::Solved);
        assert!(x.is_empty());
        if rank == 0 {
            assert_eq!(s.len(), 1);
            assert_eq!(z.len(), 1);
            assert!(s[0].is_finite() && z[0].is_finite());
        } else {
            assert!(s.is_empty() && z.is_empty());
        }
    }
}

#[test]
fn owned_hsd_rank_local_mock_single_rank_equality_infeasible_ray() {
    let handles = MockCollective::<f64>::group(1).unwrap();
    let mut joins = Vec::with_capacity(handles.len());
    for comm in handles {
        joins.push(thread::spawn(move || {
            let p = CscMatrix::zeros((0, 0));
            let q = Vec::<f64>::new();
            let a = CscMatrix::zeros((1, 0));
            let b = vec![1.0];
            let cones = vec![SupportedConeT::ZeroConeT(1)];
            let prepared = DefaultSolver::new(&p, &q, &a, &b, &cones, settings()).unwrap();
            let DefaultSolver {
                data,
                cones,
                settings,
                solution,
                timers,
                ..
            } = prepared;
            let prepared = PreparedProblem {
                data,
                cones,
                settings,
                solution,
                timers: timers.unwrap(),
                cost_input_fingerprint: None,
            };
            let mut solver = OwnedSolver::from_prepared_rank_local(prepared, Arc::new(comm))
                .expect("rank-local infeasible equality construction");
            solver.solve();
            (
                solver.solution.0.status,
                solver.solution.0.x,
                solver.solution.0.s,
                solver.solution.0.z,
            )
        }));
    }
    let results: Vec<_> = joins
        .into_iter()
        .map(|join| {
            join.join()
                .expect("rank-local infeasible equality solve panicked")
        })
        .collect();
    assert_eq!(results.len(), 1);
    let (status, x, s, z) = results.into_iter().next().unwrap();
    assert_eq!(status, SolverStatus::PrimalInfeasible);
    assert!(x.is_empty());
    assert_eq!(s.len(), 1);
    assert_eq!(z.len(), 1);
    assert!(z[0].is_finite() && z[0] != 0.0);
    let a = CscMatrix::zeros((1, 0));
    let mut atz = vec![0.0; a.n];
    a.t().gemv(&mut atz, &z, 1.0, 0.0);
    assert_eq!(atz.len(), 0);
    assert_eq!(atz.norm_inf(), 0.0);
    assert!(z[0] < 0.0, "expected b^T z < 0, got {}", z[0]);
}

#[cfg(feature = "serde")]
#[test]
fn rank_local_cost_history_gathers_and_reimports_assignments() {
    const RANKS: usize = 3;
    const FINGERPRINT: [u8; 32] = [0x5a; 32];

    fn prepared() -> PreparedProblem<f64> {
        let ProblemFixture { solver, .. } = problem::<f64>(3);
        let DefaultSolver {
            data,
            cones,
            settings,
            solution,
            timers,
            ..
        } = solver;
        PreparedProblem {
            data,
            cones,
            settings,
            solution,
            timers: timers.unwrap(),
            cost_input_fingerprint: Some(FINGERPRINT),
        }
    }

    let handles = MockCollective::<f64>::group(RANKS).unwrap();
    let joins: Vec<_> = handles
        .into_iter()
        .map(|comm| {
            thread::spawn(move || {
                let rank = comm.rank();
                let mut solver = OwnedSolver::from_prepared_rank_local_with_cost_history(
                    prepared(),
                    Arc::new(comm),
                    CostHistoryOptions::training(),
                )
                .expect("rank-local training construction");
                solver.solve();
                let owner_ids = solver.data.owner_ids.clone();
                let history = solver.cost_history().expect("rank-local history gather");
                (rank, owner_ids, history)
            })
        })
        .collect();
    let results: Vec<_> = joins
        .into_iter()
        .map(|join| join.join().expect("rank-local history solve panicked"))
        .collect();
    let root_history = results
        .iter()
        .find(|(rank, _, _)| *rank == 0)
        .and_then(|(_, _, history)| history.clone())
        .expect("root must receive gathered owner timings");
    assert!(results
        .iter()
        .filter(|(rank, _, _)| *rank != 0)
        .all(|(_, _, history)| history.is_none()));
    assert!(results
        .iter()
        .any(|(_, owner_ids, history)| owner_ids.is_empty() && history.is_none()));
    assert_eq!(root_history.owner_samples.len(), 2);
    assert!(root_history
        .owner_samples
        .iter()
        .all(|sample| sample.local_assemble_ns.is_finite()
            && sample.factor_response_ns.is_finite()
            && sample.local_assemble_ns + sample.factor_response_ns > 0.0));
    assert!(root_history
        .components
        .iter()
        .all(|component| component.cost.is_finite() && component.cost > 0.0));

    let mut expected = std::collections::BTreeMap::new();
    for sample in &root_history.owner_samples {
        for &identity in &sample.components {
            assert!(expected.insert(identity, sample.owner).is_none());
        }
    }
    assert_eq!(expected.len(), root_history.components.len());
    assert_eq!(expected.len(), 2);
    let handles = MockCollective::<f64>::group(RANKS).unwrap();
    let joins: Vec<_> = handles
        .into_iter()
        .map(|comm| {
            let history = root_history.clone();
            thread::spawn(move || {
                let solver = OwnedSolver::from_prepared_rank_local_with_cost_history(
                    prepared(),
                    Arc::new(comm),
                    CostHistoryOptions::with_history(history),
                )
                .expect("rank-local history reimport");
                solver
                    .data
                    .layout
                    .components
                    .iter()
                    .map(|component| (component.identity, component.owner))
                    .collect::<Vec<_>>()
            })
        })
        .collect();
    let mut reference_assignments = None;
    for join in joins {
        let assignments = join.join().expect("rank-local reimport panicked");
        if let Some(reference) = &reference_assignments {
            assert_eq!(reference, &assignments);
        } else {
            reference_assignments = Some(assignments.clone());
        }
        // Measured costs can reorder owners on reimport. Require a common,
        // complete valid assignment, not the old structural LPT placement.
        assert_eq!(assignments.len(), expected.len());
        let mut seen = std::collections::BTreeSet::new();
        for (identity, owner) in assignments {
            assert!(expected.contains_key(&identity));
            assert!(seen.insert(identity));
            assert!(owner < RANKS);
        }
    }
}

#[test]
fn owned_hsd_full_f64() {
    full_solve::<f64>();
}
#[test]
fn owned_hsd_full_mpfr256() {
    full_solve::<MpFloat<4>>();
}
#[test]
fn owned_hsd_full_mpfr512() {
    full_solve::<MpFloat<8>>();
}

#[test]
#[ignore = "requires frozen small Ising input, settings, and output path; actual owned HSD solve"]
fn owned_hsd_ising_solve() {
    use serde_json::{json, Value};
    type T = MpFloat<8>;
    let input = std::path::PathBuf::from(std::env::var_os("SDPX_OWNER_TEST_INPUT").unwrap());
    let template: Value = serde_json::from_slice(
        &std::fs::read(std::env::var_os("SDPX_OWNER_TEST_POINT").unwrap()).unwrap(),
    )
    .unwrap();
    let settings: DefaultSettings<T> =
        serde_json::from_value(template["settings"].clone()).unwrap();
    assert_eq!(
        settings.max_threads, 1,
        "single-core integration diagnostic"
    );
    let out = std::path::PathBuf::from(std::env::var_os("SDPX_OWNER_TEST_OUTPUT").unwrap());
    std::fs::create_dir_all(&out).unwrap();

    // Optional rank-local diagnostic. Every rank reads the same frozen
    // sampled input and prepared settings, then runs one owner-local HSD
    // instance through MockCollective. Only rank zero emits the established
    // raw.json payload; non-root vectors remain empty by contract.
    let ranks = std::env::var("SDPX_OWNER_TEST_RANKS")
        .ok()
        .map(|value| {
            value
                .parse::<usize>()
                .expect("SDPX_OWNER_TEST_RANKS must be a positive integer")
        })
        .unwrap_or(0);
    if ranks > 0 {
        let handles = MockCollective::<T>::group(ranks).unwrap();
        let mut joins = Vec::with_capacity(handles.len());
        for comm in handles {
            let input = input.clone();
            let settings = settings.clone();
            joins.push(thread::spawn(move || {
                let mut problem = read_sdpb_sampled::<T>(&input).unwrap().problem;
                problem.settings = settings;
                let DefaultSolver {
                    data,
                    cones,
                    settings,
                    solution,
                    timers,
                    ..
                } = problem.into_solver().unwrap();
                let prepared = PreparedProblem {
                    data,
                    cones,
                    settings,
                    solution,
                    timers: timers.unwrap(),
                    cost_input_fingerprint: None,
                };
                let mut solver = OwnedSolver::from_prepared_rank_local(prepared, Arc::new(comm))
                    .expect("rank-local sampled construction");
                let rank = solver.variables.collective.rank();
                let local_blocks = solver.data.blocks.len();
                let sizes: Vec<_> = solver
                    .data
                    .blocks
                    .iter()
                    .map(|d| json!({"n":d.n,"m":d.m}))
                    .collect();
                let before = std::time::Instant::now();
                solver.solve();
                let seconds = before.elapsed().as_secs_f64();
                let solution = &solver.solution.0;
                let payload = if rank == 0 {
                    Some(json!({"version":crate::VERSION,"precision_bits":512,"status":format!("{:?}",solution.status),
                        "x":&solution.x,"s":&solution.s,"z":&solution.z,"iterations":solution.iterations,
                        "objective":solution.obj_val,"dual_objective":solution.obj_val_dual,
                        "primal_residual":solution.r_prim,"dual_residual":solution.r_dual,
                        "dual_componentwise_residual":solver.info.0.res_dual_componentwise,
                        "native_seconds":solution.solve_time,"api_seconds":seconds,"settings":solver.settings(),
                        "owned_diagnostic":{"owners":ranks,"sizes":sizes,"scope":"mock rank-local, one thread per rank; no MPI/performance qualification"}}))
                } else {
                    None
                };
                (
                    rank,
                    local_blocks,
                    solution.status,
                    solution.iterations,
                    solution.x.clone(),
                    solution.s.clone(),
                    solution.z.clone(),
                    payload,
                )
            }));
        }
        let mut results = Vec::with_capacity(joins.len());
        for join in joins {
            results.push(join.join().expect("rank-local sampled solve panicked"));
        }
        results.sort_by_key(|result| result.0);
        let expected_status = results[0].2;
        let expected_iterations = results[0].3;
        let mut root_payload = None;
        for (rank, _, status, iterations, x, s, z, payload) in results {
            assert_eq!(status, SolverStatus::Solved, "rank {rank}");
            assert_eq!(status, expected_status, "rank {rank} status mismatch");
            assert_eq!(
                iterations, expected_iterations,
                "rank {rank} iteration mismatch"
            );
            if rank == 0 {
                root_payload = payload;
                assert!(!x.is_empty() || !s.is_empty() || !z.is_empty());
            } else {
                assert!(x.is_empty() && s.is_empty() && z.is_empty());
                assert_eq!(x.capacity(), 0);
                assert_eq!(s.capacity(), 0);
                assert_eq!(z.capacity(), 0);
                assert!(payload.is_none());
            }
        }
        std::fs::write(
            out.join("raw.json"),
            serde_json::to_vec_pretty(&root_payload.expect("rank zero payload missing")).unwrap(),
        )
        .unwrap();
        return;
    }

    let count: usize = std::env::var("SDPX_OWNER_COUNT").unwrap().parse().unwrap();
    let mut problem = read_sdpb_sampled::<T>(&input).unwrap().problem;
    problem.settings = settings.clone();
    let mut solver = OwnedSolver::from_default(problem.into_solver().unwrap(), count).unwrap();
    let sizes: Vec<_> = solver
        .data
        .blocks
        .iter()
        .map(|d| json!({"n":d.n,"m":d.m}))
        .collect();
    let before = std::time::Instant::now();
    solver.solve();
    let seconds = before.elapsed().as_secs_f64();
    let solution = &solver.solution.0;
    let result = json!({"version":crate::VERSION,"precision_bits":512,"status":format!("{:?}",solution.status),
        "x":solution.x,"s":solution.s,"z":solution.z,"iterations":solution.iterations,
        "objective":solution.obj_val,"dual_objective":solution.obj_val_dual,
        "primal_residual":solution.r_prim,"dual_residual":solution.r_dual,
        "dual_componentwise_residual":solver.info.0.res_dual_componentwise,
        "native_seconds":solution.solve_time,"api_seconds":seconds,"settings":settings,
        "owned_diagnostic":{"owners":count,"sizes":sizes,"scope":"single process, actual existing HSD loop; no MPI/performance qualification"}});
    std::fs::write(
        out.join("raw.json"),
        serde_json::to_vec_pretty(&result).unwrap(),
    )
    .unwrap();
    assert_eq!(solution.status, SolverStatus::Solved);
}
