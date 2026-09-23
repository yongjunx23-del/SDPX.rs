//! SDPX Rust solver core, adapted from Clarabel.rs (Apache-2.0).
//! A single generic predictor/corrector engine serves Float64 and owned MPFR
//! arithmetic. See the workspace README for the native API/CLI and numerical backends.

//Rust hates greek characters
#![allow(confusable_idents)]
#![warn(missing_docs)]

const VERSION: &str = env!("CARGO_PKG_VERSION");

pub mod algebra;
pub(crate) mod collective;
pub mod io;
pub(crate) mod mpi;
pub use mpi::MpiContext;
pub mod qdldl;
pub mod receipt;
#[cfg(feature = "snapshot")]
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

/// Finalize MPI only when SDPX initialized it. This does not initialize MPI
/// and leaves a host-owned MPI runtime untouched.
///
/// Call on the thread that first activated SDPX's MPI world, after every
/// solver call and worker using that world has finished. All ranks must call
/// this together; MPI cannot be restarted afterward. The CLI initializes on
/// its main thread and also has an exit hook. Embedded callers initializing
/// on another thread must call this before that thread exits.
pub fn mpi_finalize() {
    crate::mpi::finalize_owned();
}
