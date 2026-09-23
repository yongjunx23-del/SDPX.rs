//! Implementation of core types for the standard problem format
//! described in the documentation [main page](crate).

#![allow(non_snake_case)]

mod data_updating;
mod equilibration;
mod info;
mod info_print;
mod kktsystem;
mod owned_costs;
#[cfg(feature = "sdp")]
mod owned_hsd;
mod owned_layout;
#[cfg(feature = "sdp")]
mod partitioned;
mod presolver;
mod problemdata;
mod residuals;
#[cfg(feature = "sdp")]
mod sampled;
mod settings;
mod solution;
mod solver;
mod statistics;
mod variables;

// export flattened
pub use data_updating::*;
pub use equilibration::*;
pub use info::*;
pub use kktsystem::*;
pub(crate) use owned_costs::CostRuntimeConfig;
#[cfg(feature = "sdp")]
pub use owned_costs::{CostComponent, CostHistory, CostHistoryOptions, CostOwnerSample};
pub(crate) use owned_layout::*;
#[cfg(feature = "sdp")]
pub use partitioned::PartitionedSolver;
pub(crate) use presolver::*;
pub use problemdata::*;
pub use residuals::*;
#[cfg(feature = "sdp")]
pub use sampled::*;
pub use settings::*;
pub use solution::*;
pub use solver::*;
pub(crate) use statistics::*;
pub use variables::*;

#[cfg(feature = "serde")]
mod json;
#[cfg(feature = "serde")]
pub use json::JsonProblem;

#[cfg(all(feature = "serde", feature = "sdp"))]
mod sampled_json;
#[cfg(all(feature = "serde", feature = "sdp"))]
pub use sampled_json::{read_sdpb_sampled, SampledGramLayout, SampledJsonProblem};
