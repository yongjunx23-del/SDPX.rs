//! Generic predictor/corrector loop, its component traits and core settings.

pub mod callbacks;
pub mod traits;

mod settings;
mod solver;
pub use settings::*;
pub use solver::*;
