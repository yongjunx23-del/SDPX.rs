//! SDPX Rust solver core, adapted from Clarabel.rs (Apache-2.0).
//! A single generic predictor/corrector engine serves Float64 and owned MPFR
//! arithmetic. See the workspace README for Julia usage and numerical backends.

//Rust hates greek characters
#![allow(confusable_idents)]
#![warn(missing_docs)]

const VERSION: &str = env!("CARGO_PKG_VERSION");

pub mod algebra;
pub mod io;
pub(crate) mod mpi;
pub mod qdldl;
pub mod receipt;
pub mod solver;
pub mod timers;

pub(crate) mod utils;
pub use crate::utils::infbounds::*;

pub(crate) const _INFINITY_DEFAULT: f64 = 1e20;
