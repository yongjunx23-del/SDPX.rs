//! Generic predictor/corrector loop, its component traits and core settings.

pub mod callbacks;
pub(crate) mod checkpoint;
pub mod traits;

mod settings;
mod solver;
pub use settings::*;
pub use solver::*;

/// Test only: `SDPX_TEST_SPLIT_STEP` set gives the fixed-τ phase separate
/// primal and dual step lengths. It diverges on Λ27 (journal 2026-10-11).
pub(crate) fn test_split_step() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| std::env::var_os("SDPX_TEST_SPLIT_STEP").is_some())
}

/// Diagnostic only: `SDPX_DBG_STEPCONE` set prints each iteration's affine
/// step and σ, and the PSD cones that limit the combined step with their
/// NT centrality spread (λ²/μ).
pub(crate) fn dbg_stepcone() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| std::env::var_os("SDPX_DBG_STEPCONE").is_some())
}

/// Diagnostic only: `SDPX_TEST_NO_PROGRESS_STOP` set keeps iterating where
/// the insufficient-progress rules would stop (and restart) the solve, to
/// observe whether a stalled run escapes on its own. Never set in production.
pub(crate) fn test_no_progress_stop() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| std::env::var_os("SDPX_TEST_NO_PROGRESS_STOP").is_some())
}
