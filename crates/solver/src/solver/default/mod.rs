//! Implementation of core types for the standard problem format
//! described in the documentation [main page](crate).

#![allow(non_snake_case)]

mod data_updating;
mod equilibration;
mod info;
mod info_print;
mod kktsystem;
mod presolver;
mod problemdata;
mod residuals;
mod settings;
mod solution;
mod solver;
mod statistics;
mod variables;

// shared by the modules below through `use super::*`
#[allow(unused_imports)]
use crate::solver::distributed::*;
#[cfg(feature = "sdp")]
#[allow(unused_imports)]
use crate::solver::sampled::*;

// export flattened
pub use data_updating::*;
pub use equilibration::*;
pub use info::*;
pub use kktsystem::*;
pub(crate) use presolver::*;
pub use problemdata::*;
pub use residuals::*;
pub use settings::*;
pub use solution::*;
pub use solver::*;
pub(crate) use statistics::*;
pub use variables::*;

#[cfg(feature = "serde")]
mod json;
#[cfg(feature = "serde")]
pub use json::JsonProblem;
