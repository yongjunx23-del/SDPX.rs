//! SDPX Rust solver core, adapted from Clarabel.rs (Apache-2.0).
//! A single generic predictor/corrector engine serves Float64 and owned MPFR
//! arithmetic. See the workspace README for Julia usage and numerical backends.

//Rust hates greek characters
#![allow(confusable_idents)]
#![warn(missing_docs)]

const VERSION: &str = env!("CARGO_PKG_VERSION");

pub mod algebra;
pub mod io;
pub mod qdldl;
pub mod solver;
pub mod timers;

pub(crate) mod utils;
pub use crate::utils::infbounds::*;

#[allow(unused_macros)]
macro_rules! printbuildenv {
    ($tag:expr) => {
        if let Some(opt) = option_env!(concat!("VERGEN_", $tag)) {
            writeln!(crate::io::stdout(), "{}: {}", $tag, opt).unwrap();
        }
    };
}

/// print detailed build configuration info to stdout
#[allow(clippy::explicit_write)]
pub fn buildinfo() {
    use std::io::Write;

    #[cfg(feature = "buildinfo")]
    {
        printbuildenv!("BUILD_TIMESTAMP");
        printbuildenv!("CARGO_DEBUG");
        printbuildenv!("CARGO_FEATURES");
        printbuildenv!("CARGO_OPT_LEVEL");
        printbuildenv!("CARGO_TARGET_TRIPLE");
        printbuildenv!("RUSTC_CHANNEL");
        printbuildenv!("RUSTC_COMMIT_DATE");
        printbuildenv!("RUSTC_COMMIT_HASH");
        printbuildenv!("RUSTC_HOST_TRIPLE");
        printbuildenv!("RUSTC_LLVM_VERSION");
        printbuildenv!("RUSTC_SEMVER");
        printbuildenv!("SYSINFO_NAME");
        printbuildenv!("SYSINFO_OS_VERSION");
        printbuildenv!("SYSINFO_TOTAL_MEMORY");
        printbuildenv!("SYSINFO_CPU_VENDOR");
        printbuildenv!("SYSINFO_CPU_CORE_COUNT");
        printbuildenv!("SYSINFO_CPU_BRAND");
        printbuildenv!("SYSINFO_CPU_FREQUENCY");
    }
    #[cfg(not(feature = "buildinfo"))]
    writeln!(crate::io::stdout(), "no build info available").unwrap();
}

pub(crate) const _INFINITY_DEFAULT: f64 = 1e20;

#[test]
fn test_buildinfo() {
    buildinfo();
}
