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
