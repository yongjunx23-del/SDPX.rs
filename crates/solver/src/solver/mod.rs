//! SDPX solver: one generic HSD predictor/corrector loop (`core`) with the
//! standard-format implementation (`default`), sampled operators, and the
//! owner-partitioned MPI implementation (`distributed`).

pub(crate) mod cones;
pub(crate) mod core;
pub mod default;
#[cfg(feature = "sdp")]
pub(crate) mod distributed;
pub(crate) mod kkt;
#[cfg(feature = "sdp")]
pub mod sampled;

#[cfg(feature = "sdp")]
pub(crate) mod chordal;

pub use crate::solver::cones::{SupportedConeT, SupportedConeT::*};
pub use crate::solver::core::traits;
pub use crate::solver::core::CoreSettings;
#[cfg(feature = "serde")]
pub use crate::solver::core::SolverJSONReadWrite;
pub use crate::solver::core::{IPSolver, SolverStatus};
pub use crate::solver::kkt::LinearSolverInfo;

pub use crate::solver::default::*;
#[cfg(feature = "sdp")]
pub use crate::solver::distributed::{
    CostComponent, CostHistory, CostHistoryOptions, CostOwnerSample, PartitionedSolver,
};
#[cfg(feature = "sdp")]
pub use crate::solver::sampled::*;
