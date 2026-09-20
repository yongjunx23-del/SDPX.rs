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
pub mod snapshot;
pub mod solver;
pub mod timers;

pub(crate) mod utils;
pub use crate::utils::infbounds::*;

pub(crate) const _INFINITY_DEFAULT: f64 = 1e20;

/// Number of MPI ranks the solver actually engaged, or 1 when running
/// without MPI. Lets callers distinguish a real distributed run from a
/// silent serial fallback under `mpiexec` (a launcher failure would
/// otherwise produce identical-looking serial results on every rank).
pub fn mpi_world_size() -> i32 {
    crate::mpi::World::get().map_or(1, |w| w.size() as i32)
}
