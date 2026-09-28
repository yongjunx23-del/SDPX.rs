use crate::solver::default::DefaultSettings;
use thiserror::Error;

/// Solver general core settings are the same as in the default solver.
///
/// Go [here](crate::solver::default::DefaultSettings)
/// to view the complete list.
///
pub type CoreSettings<T> = DefaultSettings<T>;

/// One automatic worker budget for setup, cones and factorization. Respect
/// the process CPU allocation and an optional Rayon environment limit.
pub(crate) fn worker_budget(requested: usize) -> usize {
    if requested != 0 {
        return requested;
    }
    let available = std::thread::available_parallelism().map_or(1, usize::from);
    std::env::var("RAYON_NUM_THREADS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
        .map_or(available, |limit| available.min(limit))
}

#[derive(Error, Debug)]
/// Error type returned by settings validation
pub enum SettingsError {
    /// An error attributable to one of the fields
    #[error("Bad value for field \"{0}\"")]
    BadFieldValue(&'static str),
    /// An error thrown when immutable settings are modified within the solver
    #[error("Attempt to modify immutable setting \"{0}\"")]
    ImmutableSetting(&'static str),
    /// a subsolver error of some kind (e.g. not found, no license)
    #[error("Problem with {solver} solver ({problem})")]
    LinearSolverProblem {
        solver: &'static str,
        problem: &'static str,
    },
}
