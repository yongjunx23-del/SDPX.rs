//! Opt-in fixed accepted-point diagnostic. No solve or performance claim.
use super::tests::{runtime, scatter};
use super::*;
use crate::solver::core::{
    traits::{Info, Residuals, Solution},
    SolverStatus,
};
use crate::timers::Timers;
use num_traits::{One, Zero};
use sdpx_arithmetic::MpFloat;
use serde_json::{json, Value};
use std::path::PathBuf;

#[test]
#[ignore = "requires frozen Ising input, accepted baseline point and output directory"]
fn owned_ising_accepted_point() {
    type T = MpFloat<8>;
    let input = PathBuf::from(std::env::var_os("SDPX_OWNER_TEST_INPUT").unwrap());
    let raw_path = PathBuf::from(std::env::var_os("SDPX_OWNER_TEST_POINT").unwrap());
    let out = PathBuf::from(std::env::var_os("SDPX_OWNER_TEST_OUTPUT").unwrap());
    std::fs::create_dir_all(&out).unwrap();
    let raw: Value = serde_json::from_slice(&std::fs::read(raw_path).unwrap()).unwrap();
    assert_eq!(raw["precision_bits"], 512);
    assert_eq!(raw["status"], "Solved");
    let settings: DefaultSettings<T> = serde_json::from_value(raw["settings"].clone()).unwrap();
    let make = || {
        let mut problem = read_sdpb_sampled::<T>(&input).unwrap().problem;
        problem.settings = settings.clone();
        problem.into_prepared().unwrap().data
    };
    let mut data = make();
    assert!(
        !data.is_chordal_decomposed(),
        "this frozen point is not chordal mapped"
    );
    let x: Vec<T> = serde_json::from_value(raw["x"].clone()).unwrap();
    let mut s: Vec<T> = serde_json::from_value(raw["s"].clone()).unwrap();
    let mut z: Vec<T> = serde_json::from_value(raw["z"].clone()).unwrap();
    if let Some(presolve) = &data.presolver {
        let keep = &presolve.reduce_map.as_ref().unwrap().keep_logical;
        s = s
            .iter()
            .zip(keep)
            .filter_map(|(&v, &k)| k.then_some(v))
            .collect();
        z = z
            .iter()
            .zip(keep)
            .filter_map(|(&v, &k)| k.then_some(v))
            .collect();
    }
    assert_eq!((x.len(), s.len(), z.len()), (data.n, data.m, data.m));
    // Test the recovered physical point at tau=1,kappa=0; do not pretend these
    // are the saved solver's unreported homogeneous scalars or an owned solve.
    let eq = &data.equilibration;
    let point = DefaultVariables {
        x: x.iter().zip(&eq.dinv).map(|(&v, &d)| v * d).collect(),
        s: s.iter().zip(&eq.e).map(|(&v, &e)| v * e).collect(),
        z: z.iter()
            .zip(&eq.einv)
            .map(|(&v, &ei)| v * eq.c * ei)
            .collect(),
        τ: T::one(),
        κ: T::zero(),
    };
    let mut residual = DefaultResiduals::new(data.n, data.m);
    residual.update(&point, &data);
    let mut reference = DefaultInfo::new();
    reference.update(&mut data, &point, &residual, &Timers::default());
    reference.check_termination(&residual, &settings, 0);
    assert_eq!(reference.status, SolverStatus::Solved);
    let mut rows = Vec::new();
    for count in [1, 2, 4, 8] {
        let mut owned = runtime(
            make(),
            count,
            settings.clone(),
            (x.len(), raw["s"].as_array().unwrap().len()),
        );
        scatter(&mut owned.variables, &point);
        let pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(count)
                .build()
                .unwrap(),
        );
        owned
            .residuals
            .update_with_pool(&owned.variables, &owned.data, Some(pool));
        let mut owned_info = OwnedInfo(DefaultInfo::new());
        owned_info.update(
            &mut owned.data,
            &owned.variables,
            &owned.residuals,
            &Timers::default(),
        );
        owned_info.check_termination(&owned.residuals, &settings, 0);
        let info = &owned_info.0;
        assert_eq!(info.status, SolverStatus::Solved);
        assert!(info.res_primal <= settings.tol_feas && info.res_dual <= settings.tol_feas);
        assert!(info.res_dual_componentwise.unwrap() <= settings.tol_feas_componentwise.unwrap());
        let columns: Vec<_> = owned.data.blocks.iter().map(|o| o.n).collect();
        let conic_rows: Vec<_> = owned.data.blocks.iter().map(|o| o.m).collect();
        let nnz: Vec<_> = owned
            .data
            .blocks
            .iter()
            .map(|o| o.constraint_nnz())
            .collect();
        let sampled: Vec<_> = owned
            .data
            .blocks
            .iter()
            .map(|o| o.sampled.as_ref().unwrap().blocks().len())
            .collect();
        assert_eq!(columns.iter().sum::<usize>(), data.n);
        assert_eq!(nnz.iter().sum::<usize>(), data.constraint_nnz());
        assert_eq!(
            conic_rows.iter().sum::<usize>(),
            data.m + (count - 1) * owned.data.layout.border_rows.len()
        );
        if count > 1 {
            assert!(columns.iter().all(|&n| n < data.n));
        }
        owned
            .solution
            .post_process(&owned.data, &mut owned.variables, &owned_info, &settings);
        owned.solution.finalize(&owned_info);
        let recovered = &owned.solution.0;
        let point_path = out.join(format!("owners-{count}-point.json"));
        let record = json!({"precision_bits":512,"status":"Solved","iterations":raw["iterations"],
            "x":recovered.x,"s":recovered.s,"z":recovered.z,
            "scope":"fixed baseline point, partitioned residual and recovery only; iterations belong to baseline solve"});
        std::fs::write(&point_path, serde_json::to_vec_pretty(&record).unwrap()).unwrap();
        rows.push(json!({"owners":count,"local_columns":columns,"local_rows":conic_rows,
            "local_A_nnz":nnz,"local_sampled_blocks":sampled,"shared_equalities":owned.data.layout.border_rows.len(),
            "res_primal":info.res_primal,"res_dual":info.res_dual,"res_componentwise":info.res_dual_componentwise,
            "gap_abs":info.gap_abs,"point":point_path}));
    }
    std::fs::write(
        out.join("rows.json"),
        serde_json::to_vec_pretty(&rows).unwrap(),
    )
    .unwrap();
}
