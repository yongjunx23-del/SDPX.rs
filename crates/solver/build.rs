//! Default BLAS/LAPACK linkage when no `sdp-*` provider feature is chosen:
//! Accelerate on macOS, the system's dynamic OpenBLAS elsewhere on Unix.
//! `SDPX_BLAS_LINK=none` leaves linkage to the caller (e.g. RUSTFLAGS);
//! `OPENBLAS_LIB_DIR` adds a library search path.
fn main() {
    println!("cargo:rerun-if-env-changed=SDPX_BLAS_LINK");
    println!("cargo:rerun-if-env-changed=OPENBLAS_LIB_DIR");
    let provider = ["ACCELERATE", "NETLIB", "OPENBLAS", "MKL"]
        .iter()
        .any(|p| std::env::var_os(format!("CARGO_FEATURE_SDP_{p}")).is_some());
    if provider || std::env::var("SDPX_BLAS_LINK").is_ok_and(|v| v == "none") {
        return;
    }
    match std::env::var("CARGO_CFG_TARGET_OS").as_deref() {
        Ok("macos") => println!("cargo:rustc-link-lib=framework=Accelerate"),
        Ok("windows") => {}
        _ => {
            if let Ok(dir) = std::env::var("OPENBLAS_LIB_DIR") {
                println!("cargo:rustc-link-search=native={dir}");
            }
            println!("cargo:rustc-link-lib=dylib=openblas");
        }
    }
}
