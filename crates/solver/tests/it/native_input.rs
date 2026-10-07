#![cfg(feature = "serde")]
use sdpx_arithmetic::{MpFloat, Scalar};
use sdpx_solver::solver::*;
use std::{fs, process::Command};

const LP: &str = r#"{
  "P":{"m":1,"n":1,"colptr":[0,0],"rowval":[],"nzval":[]},
  "q":[1],"A":{"m":1,"n":1,"colptr":[0,1],"rowval":[0],"nzval":[-1]},
  "b":[-1],"cones":[{"NonnegativeConeT":1}],"settings":{"verbose":false}
}"#;

#[test]
fn mpfr_json_retains_digits_and_rejects_f64_transport() {
    type T = MpFloat<4>;
    let exact = "1.0000000000000000000000000000000000000000000000000000000001";
    let input = LP.replace("\"q\":[1]", &format!("\"q\":[\"{exact}\"]"));
    let data = JsonProblem::<T>::read(input.as_bytes()).unwrap();
    assert_eq!(data.q[0], exact.parse().unwrap());
    assert_ne!(data.q[0], "1".parse().unwrap());
    let json = serde_json::to_string(&data.q).unwrap();
    let recovered: Vec<T> = serde_json::from_str(&json).unwrap();
    assert_eq!(data.q, recovered);
    let mut bytes = Vec::new();
    data.write(&mut bytes).unwrap();
    let reread = JsonProblem::<T>::read(bytes.as_slice()).unwrap();
    assert_eq!(reread.q, data.q);
    assert_eq!(reread.settings.time_limit, f64::INFINITY);
    for bad in ["1.25", "\"NaN\"", "\"inf\"", "1e100"] {
        let input = LP.replace("\"q\":[1]", &format!("\"q\":[{bad}]"));
        assert!(JsonProblem::<T>::read(input.as_bytes()).is_err(), "{bad}");
    }
    let mut solver = data.into_solver().unwrap();
    solver.solve();
    assert_eq!(solver.solution.status, SolverStatus::Solved);
    assert!((solver.solution.x[0] - "1".parse::<T>().unwrap()).abs() < "1e-20".parse().unwrap());
}

#[test]
fn native_cli_status_settings_and_precision() {
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("input.json");
    fs::write(&input, LP).unwrap();
    let run = |args: &[&str]| {
        Command::new(env!("CARGO_BIN_EXE_sdpx"))
            .arg(&input)
            .args(args)
            .output()
            .unwrap()
    };
    let widest = cfg!(feature = "all-precisions").then_some("2048");
    for precision in ["53", "128", "256", "512", "768", "1024"]
        .into_iter()
        .chain(widest)
    {
        let output = run(&["--precision", precision, "--threads", "1"]);
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let point: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(point["status"], "Solved");
        assert_eq!(point["precision_bits"], precision.parse::<usize>().unwrap());
        assert!(point.get("psd_direction").is_none());
        assert_eq!(point["mpi_world_size"], 1);
        assert_eq!(point["output_rank"], 0);
        assert_eq!(point["settings"]["equilibrate_enable"], true);
        assert_eq!(point["settings"]["presolve_enable"], true);
        assert_eq!(point["settings"]["chordal_decomposition_enable"], true);
        assert!(point["api_seconds"].as_f64().unwrap() > 0.0);
        assert_eq!(point["x"][0].is_string(), precision != "53");
        if precision == "53" || precision == "256" {
            let settings_path = temp.path().join("roundtrip-settings.json");
            fs::write(
                &settings_path,
                serde_json::to_vec(&point["settings"]).unwrap(),
            )
            .unwrap();
            let rerun = run(&[
                "--precision",
                precision,
                "--settings",
                settings_path.to_str().unwrap(),
            ]);
            assert!(
                rerun.status.success(),
                "{}",
                String::from_utf8_lossy(&rerun.stderr)
            );
            let repeated: serde_json::Value = serde_json::from_slice(&rerun.stdout).unwrap();
            assert_eq!(repeated["x"], point["x"]);
        }
    }
    let settings = temp.path().join("settings.json");
    fs::write(
        &settings,
        r#"{"max_iter":0,"verbose":false,"presolve_enable":false}"#,
    )
    .unwrap();
    let output = run(&["--settings", settings.to_str().unwrap()]);
    assert_eq!(output.status.code(), Some(2));
    let point: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(point["status"], "MaxIterations");
    assert_eq!(run(&["--precision", "106"]).status.code(), Some(1));
    assert_eq!(run(&["--direction", "typo"]).status.code(), Some(1));
    assert_eq!(run(&["--unknown"]).status.code(), Some(1));
    let malformed = LP.replace("\"colptr\":[0,1]", "\"colptr\":[1,1]");
    fs::write(&input, malformed).unwrap();
    let output = run(&[]);
    assert_eq!(output.status.code(), Some(1));
    assert!(!String::from_utf8_lossy(&output.stderr).contains("panicked"));
}

#[test]
fn native_cli_rejects_bad_cones_and_reports_infeasibility() {
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("input.json");
    let run = |bits: &str| {
        Command::new(env!("CARGO_BIN_EXE_sdpx"))
            .arg(&input)
            .args(["--precision", bits, "--quiet"])
            .output()
            .unwrap()
    };
    for cone in [
        r#"{"GenPowerConeT":[[],1]}"#,
        r#"{"GenPowerConeT":[[1,1],0]}"#,
        r#"{"PowerConeT":0}"#,
    ] {
        fs::write(&input, LP.replace(r#"{"NonnegativeConeT":1}"#, cone)).unwrap();
        let output = run("53");
        assert_eq!(output.status.code(), Some(1));
        assert!(!String::from_utf8_lossy(&output.stderr).contains("panicked"));
    }
    // A one-dimensional SOC is deliberately supported and collapses to NN.
    fs::write(&input, LP.replace("NonnegativeConeT", "SecondOrderConeT")).unwrap();
    assert!(run("53").status.success());
    fs::write(
        &input,
        r#"{
      "P":{"m":1,"n":1,"colptr":[0,0],"rowval":[],"nzval":[]},
      "q":[-1],"A":{"m":0,"n":1,"colptr":[0,0],"rowval":[],"nzval":[]},
      "b":[],"cones":[],"settings":{"verbose":false}
    }"#,
    )
    .unwrap();
    for bits in ["53", "256"] {
        let output = run(bits);
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let point: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(point["status"], "DualInfeasible");
        assert!(point["objective"].is_null());
        assert!(point["dual_objective"].is_null());
    }
}

fn sampled_fixture(path: &std::path::Path) {
    for (name, text) in [
        ("control.json", r#"{"num_blocks":1,"command":"pmp2sdp"}"#),
        ("objectives.json", r#"{"constant":"3","b":["1"]}"#),
        ("block_info_0.json", r#"{"dim":2,"num_points":1}"#),
        (
            "block_data_0.json",
            r#"{"bilinear_bases_even":[["2"]],"bilinear_bases_odd":[],"c":["1","0","1"],"B":[["1"],["0"],["1"]]}"#,
        ),
    ] {
        fs::write(path.join(name), text).unwrap();
    }
}

#[test]
fn sdpb_reader_matches_trace_map_and_native_results() {
    let temp = tempfile::tempdir().unwrap();
    sampled_fixture(temp.path());
    let data = read_sdpb_sampled::<f64>(temp.path()).unwrap();
    assert_eq!(data.objective_constant, 3.0);
    assert_eq!(data.num_equalities, 1);
    assert_eq!((data.grams[0].row_start, data.grams[0].side), (1, 2));
    let operator =
        SampledOperator::new(data.problem.A.clone(), data.problem.sampled.clone()).unwrap();
    let matrix = operator.materialize();
    let mut ax = vec![0.; 4];
    for (column, x) in [1., 2., 3.].iter().enumerate() {
        for i in matrix.colptr[column]..matrix.colptr[column + 1] {
            ax[matrix.rowval[i]] += matrix.nzval[i] * x;
        }
    }
    assert_eq!(ax[0], 4.);
    assert_eq!(ax[1], -4.);
    assert!((ax[2] + 4. * 2f64.sqrt()).abs() < 1e-14);
    assert_eq!(ax[3], -12.);
    for precision in ["53", "256"] {
        let output = Command::new(env!("CARGO_BIN_EXE_sdpx"))
            .arg(temp.path())
            .args(["--precision", precision, "--threads", "1", "--quiet"])
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let point: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(point["status"], "Solved");
        let objective = &point["original_objective"];
        let value = objective
            .as_f64()
            .unwrap_or_else(|| objective.as_str().unwrap().parse().unwrap());
        assert!((value - 4.).abs() < 1e-8);
        assert_eq!(point["sampled_y"].as_array().unwrap().len(), 1);
    }
    let file = temp.path().join("block_data_0.json");
    let bad = fs::read_to_string(&file)
        .unwrap()
        .replace("[[\"2\"]]", "[[2]]");
    fs::write(file, bad).unwrap();
    assert!(read_sdpb_sampled::<MpFloat<4>>(temp.path()).is_err());
}

#[test]
fn sampled_input_boundary_checks_and_optional_command() {
    let temp = tempfile::tempdir().unwrap();
    sampled_fixture(temp.path());
    fs::write(temp.path().join("control.json"), r#"{"num_blocks":1}"#).unwrap();
    assert!(read_sdpb_sampled::<f64>(temp.path()).is_ok());
    fs::write(temp.path().join("unexpected.json"), "{}").unwrap();
    assert!(read_sdpb_sampled::<f64>(temp.path()).is_err());
    fs::remove_file(temp.path().join("unexpected.json")).unwrap();
    let file = temp.path().join("block_data_0.json");
    let source = fs::read_to_string(&file).unwrap();
    fs::write(&file, source.replacen('{', "{\"extra\":0,", 1)).unwrap();
    assert!(read_sdpb_sampled::<f64>(temp.path()).is_err());
    for value in ["1e200", "1e-200"] {
        fs::write(
            &file,
            source.replace("[[\"2\"]]", &format!("[[\"{value}\"]]")),
        )
        .unwrap();
        let data = read_sdpb_sampled::<f64>(temp.path()).unwrap();
        let result = data.problem.into_solver();
        assert!(matches!(result, Err(SolverError::SampledInput(_))));
    }
}

#[test]
fn sampled_reader_keeps_matrix_sample_and_parity_order() {
    let temp = tempfile::tempdir().unwrap();
    sampled_fixture(temp.path());
    fs::write(
        temp.path().join("block_info_0.json"),
        r#"{"dim":2,"num_points":2}"#,
    )
    .unwrap();
    fs::write(
        temp.path().join("block_data_0.json"),
        r#"{
      "bilinear_bases_even":[["1","2"],["3","4"]],"bilinear_bases_odd":[["5","6"]],
      "c":["1","2","3","4","5","6"],"B":[["1"],["2"],["3"],["4"],["5"],["6"]]
    }"#,
    )
    .unwrap();
    let data = read_sdpb_sampled::<f64>(temp.path()).unwrap();
    assert_eq!(
        data.grams
            .iter()
            .map(|g| (g.parity, g.row_start, g.side))
            .collect::<Vec<_>>(),
        vec![(0, 1, 4), (1, 11, 2)]
    );
    let matrix = SampledOperator::new(data.problem.A, data.problem.sampled)
        .unwrap()
        .materialize();
    let column = |j: usize| {
        let mut values = vec![0.; matrix.m];
        for k in matrix.colptr[j]..matrix.colptr[j + 1] {
            values[matrix.rowval[k]] = matrix.nzval[k];
        }
        values
    };
    // The third primitive is the first sample of the off-diagonal polynomial
    // entry: sym(e0*e1') carries one half before svec multiplies by sqrt(2).
    let cross = column(2);
    for (row, value) in [
        (0, 3.),
        (4, -1. / 2f64.sqrt()),
        (5, -3. / 2f64.sqrt()),
        (7, -3. / 2f64.sqrt()),
        (8, -9. / 2f64.sqrt()),
        (12, -25. / 2f64.sqrt()),
    ] {
        assert!((cross[row] - value).abs() < 1e-13);
    }
    assert_eq!(cross.iter().filter(|x| **x != 0.).count(), 6);
    let last = column(5);
    for (row, value) in [
        (0, 6.),
        (6, -4.),
        (9, -8. * 2f64.sqrt()),
        (10, -16.),
        (13, -36.),
    ] {
        assert!((last[row] - value).abs() < 1e-13);
    }
    assert_eq!(last.iter().filter(|x| **x != 0.).count(), 5);
}
