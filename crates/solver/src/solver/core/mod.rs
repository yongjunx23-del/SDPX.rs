//! Generic predictor/corrector loop, its component traits and core settings.

pub mod callbacks;
pub(crate) mod checkpoint;
pub mod traits;

mod settings;
mod solver;
pub use settings::*;
pub use solver::*;
