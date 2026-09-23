//! Owner-partitioned HSD implementation for MPI: block state stays on its
//! owning rank and ranks share one equality Schur complement. It plugs into the
//! core predictor/corrector loop through the same traits as `default`.

#[allow(unused_imports)]
use crate::algebra::*;
#[cfg(feature = "sdp")]
#[allow(unused_imports)]
use crate::solver::chordal::ChordalInfo;
#[allow(unused_imports)]
use crate::solver::cones::{CompositeCone, Cone};
#[allow(unused_imports)]
use crate::solver::core::traits::ProblemData;
#[allow(unused_imports)]
use crate::solver::default::*;
#[cfg(feature = "sdp")]
#[allow(unused_imports)]
use crate::solver::sampled::*;
#[allow(unused_imports)]
use crate::solver::SupportedConeT;

pub(crate) mod collective;
mod costs;
#[cfg(feature = "sdp")]
mod hsd;
#[cfg(feature = "sdp")]
mod kkt;
mod layout;
#[cfg(feature = "sdp")]
mod partitioned;
mod state;

#[cfg(all(feature = "serde", feature = "sdp"))]
pub(crate) use costs::input_fingerprint;
pub(crate) use costs::CostRuntimeConfig;
#[cfg(feature = "sdp")]
pub use costs::{CostComponent, CostHistory, CostHistoryOptions, CostOwnerSample};
#[cfg(feature = "sdp")]
#[allow(unused_imports)]
pub(crate) use hsd::*;
#[cfg(feature = "sdp")]
pub(crate) use kkt::*;
pub(crate) use layout::*;
#[cfg(feature = "sdp")]
pub use partitioned::PartitionedSolver;
#[allow(unused_imports)]
pub(crate) use state::*;
