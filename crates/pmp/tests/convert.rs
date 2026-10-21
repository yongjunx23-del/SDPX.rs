//! Golden checks on the README example: maximize y subject to 1 - y + x^2 >= 0 for x >= 0.
use sdpx_arithmetic::Bits128;
use sdpx_pmp::PolynomialMatrixProgram;
use std::{collections::BTreeMap, fs, path::Path};

const PMP: &str = r#"{
  "objective": ["0", "1"],
  "PositiveMatrixWithPrefactorArray": [
    {
      "prefactor": { "constant": "1", "base": "0.5", "poles": [] },
      "polynomials": [[[ ["1", "0", "1"], ["-1"] ]]]
    }
  ]
}"#;

fn read_dir(dir: &Path) -> BTreeMap<String, Vec<u8>> {
    fs::read_dir(dir)
        .unwrap()
        .map(|e| {
            let e = e.unwrap();
            (
                e.file_name().to_string_lossy().into_owned(),
                fs::read(e.path()).unwrap(),
            )
        })
        .collect()
}

fn convert(name: &str, threads: usize) -> BTreeMap<String, Vec<u8>> {
    let root = std::env::temp_dir().join(format!("sdpx-pmp-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    let pmp: PolynomialMatrixProgram = serde_json::from_str(PMP).unwrap();
    pmp.write_sdp_with_threads::<Bits128>(&root, threads)
        .unwrap();
    let files = read_dir(&root);
    fs::remove_dir_all(&root).unwrap();
    files
}

#[test]
fn readme_example_writes_expected_layout() {
    let files = convert("layout", 1);
    assert!(files.contains_key("control.json"), "{:?}", files.keys());
    assert!(
        files.keys().any(|k| k.starts_with("block_data")),
        "{:?}",
        files.keys()
    );
    let control: serde_json::Value = serde_json::from_slice(&files["control.json"]).unwrap();
    assert!(control.is_object());
}

#[test]
fn output_is_identical_across_thread_counts() {
    assert_eq!(convert("t1", 1), convert("t2", 2));
}

#[test]
fn existing_output_path_is_refused() {
    let root = std::env::temp_dir().join(format!("sdpx-pmp-{}-exists", std::process::id()));
    fs::create_dir_all(&root).unwrap();
    let pmp: PolynomialMatrixProgram = serde_json::from_str(PMP).unwrap();
    assert!(pmp.write_sdp::<Bits128>(&root).is_err());
    fs::remove_dir_all(&root).unwrap();
}

#[test]
fn explicit_samples_survive_default_resampling_and_still_validate() {
    let root = std::env::temp_dir().join(format!("sdpx-pmp-{}-samples", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir(&root).unwrap();
    let mut pmp: serde_json::Value = serde_json::from_str(PMP).unwrap();
    let matrix = &mut pmp["PositiveMatrixWithPrefactorArray"][0];
    matrix.as_object_mut().unwrap().remove("prefactor");
    matrix["samplePoints"] = serde_json::json!(["0", "1", "2"]);
    matrix["sampleScalings"] = serde_json::json!(["1", "1", "1"]);
    let input = root.join("input.json");
    fs::write(&input, serde_json::to_vec(&pmp).unwrap()).unwrap();
    let preserved = root.join("preserved");
    let default = root.join("default");
    sdpx_pmp::convert_file::<Bits128>(&input, &preserved, 1, false).unwrap();
    sdpx_pmp::convert_file::<Bits128>(&input, &default, 1, true).unwrap();
    assert_eq!(read_dir(&preserved), read_dir(&default));
    pmp["PositiveMatrixWithPrefactorArray"][0]
        .as_object_mut()
        .unwrap()
        .remove("sampleScalings");
    fs::write(&input, serde_json::to_vec(&pmp).unwrap()).unwrap();
    sdpx_pmp::convert_file::<Bits128>(&input, root.join("implicit_scalings"), 1, true).unwrap();
    pmp["PositiveMatrixWithPrefactorArray"][0]["sampleScalings"] =
        serde_json::json!(["1", "0", "1"]);
    fs::write(&input, serde_json::to_vec(&pmp).unwrap()).unwrap();
    let invalid = root.join("invalid");
    assert!(sdpx_pmp::convert_file::<Bits128>(&input, &invalid, 1, true).is_err());
    assert!(!invalid.exists());
    fs::remove_dir_all(root).unwrap();
}
