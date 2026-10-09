//! Required traits for types providing a Clarabel solver implementation.
//!
//! This module defines the core traits that must be implemented by a collection
//! of mutually associated data types to make a solver for a particular problem
//! format.
//!
//! In nearly all cases there is no need for a user to implement these traits.
//! Instead, users should use the collection of types that are provided
//! in the [Default solver implementation](crate::solver::default),
//!  which collectively implement support for the problem format described in the top
//! level crate documentation.

use super::{CoreSettings, ScalingStrategy, SettingsError};
use super::{SolverStatus, StepDirection};
use crate::algebra::*;
use crate::solver::cones::Cone;
use crate::timers::*;

/// Cone-level control needed by the common HSD loop, independent of vector storage.
pub trait ConeCollection<T: FloatT> {
    /// Whether every member cone is symmetric.
    fn all_symmetric(&self) -> bool;
    /// Whether any member has rows the centrality correctors act on
    /// (nonnegative orthant, second-order cone, or binary64 PSD cone).
    fn has_correctable(&self) -> bool {
        false
    }
    /// Whether primal-dual scaling is supported by all members.
    fn supports_primal_dual(&self) -> bool;
    /// Set identity scaling before the symmetric initializer.
    fn reset_scaling(&mut self);
    /// Shared worker pool for independent numerical stages.
    fn worker_pool(&self) -> Option<std::sync::Arc<rayon::ThreadPool>>;
}
impl<T: FloatT> ConeCollection<T> for crate::solver::cones::CompositeCone<T> {
    fn has_correctable(&self) -> bool {
        use crate::solver::cones::SupportedCone;
        self.iter().any(|c| {
            matches!(
                c,
                SupportedCone::NonnegativeCone(_) | SupportedCone::SecondOrderCone(_)
            ) || (matches!(c, SupportedCone::PSDTriangleCone(_)) && T::precision_bits() <= 53)
        })
    }
    fn all_symmetric(&self) -> bool {
        self.is_symmetric()
    }
    fn supports_primal_dual(&self) -> bool {
        self.allows_primal_dual_scaling()
    }
    fn reset_scaling(&mut self) {
        self.set_identity_scaling()
    }
    fn worker_pool(&self) -> Option<std::sync::Arc<rayon::ThreadPool>> {
        Cone::thread_pool(self)
    }
}

/// Data for a conic optimization problem.
pub trait ProblemData<T: FloatT> {
    /// associated variable type
    type V: Variables<T>;
    /// associated cone type
    type C: ConeCollection<T>;
    /// associated settings type
    type SE: Settings<T>;

    /// Equilibrate internal data before solver starts.
    fn equilibrate(&mut self, cones: &Self::C, settings: &Self::SE);

    /// Scale statistics of the equilibrated data (diagnostics).
    fn scale_stats(&self) -> String {
        String::new()
    }

    /// Starting τ for the unit fallback start, when the data imply one
    /// (see `DefaultSettings::auto_initial_tau`).
    fn unit_start_tau(&self) -> Option<T> {
        None
    }
}

/// Variables for a conic optimization problem.
pub trait Variables<T: FloatT> {
    /// associated problem data type
    type D: ProblemData<T>;
    /// associated problem residuals type
    type R: Residuals<T>;
    /// associated cone type
    type C: ConeCollection<T>;
    /// associated settings type
    type SE: Settings<T>;

    /// Compute the scaled duality gap.
    fn calc_mu(&mut self, residuals: &Self::R, cones: &Self::C) -> T;

    /// Write the iterate to a checkpoint file. Unsupported by default.
    fn write_checkpoint(
        &self,
        _data: &Self::D,
        _path: &std::path::Path,
        _iter: u32,
    ) -> std::io::Result<()> {
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "this solver backend does not support checkpoints",
        ))
    }

    /// Replace the iterate from a checkpoint file of a problem with the same
    /// structure; returns whether the problem data was identical (exact
    /// continuation) rather than nearby (hot start). Unsupported by default.
    fn read_checkpoint(
        &mut self,
        _data: &Self::D,
        _path: &std::path::Path,
    ) -> std::io::Result<bool> {
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "this solver backend does not support checkpoints",
        ))
    }

    /// Compute the KKT RHS for a pure Newton step.
    fn affine_step_rhs(&mut self, residuals: &Self::R, variables: &Self, cones: &Self::C);

    /// Compute the KKT RHS for an interior point centering step.
    #[allow(clippy::too_many_arguments)]
    fn combined_step_rhs(
        &mut self,
        residuals: &Self::R,
        variables: &Self,
        cones: &mut Self::C,
        step: &mut Self, //mut allows step to double as working space
        σ: T,
        μ: T,
        m: T,
    );

    /// Compute the maximum step length possible in the given
    /// step direction without violating a cone boundary.
    fn calc_step_length(
        &self,
        step_lhs: &Self,
        cones: &mut Self::C,
        settings: &Self::SE,
        step_direction: StepDirection,
    ) -> T;

    /// Prepare an affine direction for the immediately following prepared RHS.
    fn prepare_affine_step_length(
        &self,
        step: &mut Self,
        cones: &mut Self::C,
        settings: &Self::SE,
    ) -> T {
        self.calc_step_length(step, cones, settings, StepDirection::Affine)
    }

    /// Add Gondzio centrality corrections to this combined right-hand side for
    /// a trial step `α` along `step`: complementarity products outside
    /// `[0.1σμ, 10σμ]` are pushed back into the band. `false` when none is.
    fn centrality_correction(
        &mut self,
        _step: &Self,
        _variables: &Self,
        _cones: &mut Self::C,
        _α: T,
        _σμ: T,
    ) -> bool {
        false
    }

    /// Consume the direction from `prepare_affine_step_length`, with correction m=1.
    fn combined_step_rhs_prepared(
        &mut self,
        residuals: &Self::R,
        variables: &Self,
        cones: &mut Self::C,
        step: &mut Self,
        σ: T,
        μ: T,
    ) {
        self.combined_step_rhs(residuals, variables, cones, step, σ, μ, T::one());
    }

    /// Update the variables in the given step direction, scaled by `α`.
    fn add_step(&mut self, step_lhs: &Self, α: T);

    /// Elementwise step update on the current cone worker pool when
    /// supported.  Every element is computed identically to
    /// [`Variables::add_step`], so results are bitwise identical.
    fn add_step_with_pool(
        &mut self,
        step_lhs: &Self,
        α: T,
        _pool: Option<std::sync::Arc<rayon::ThreadPool>>,
    ) {
        self.add_step(step_lhs, α);
    }

    /// Bring the variables into the interior of the cone constraints.
    fn symmetric_initialization(&mut self, cones: &mut Self::C);

    /// Initialize all conic variables to unit values.
    fn unit_initialization(&mut self, cones: &Self::C);

    /// Start the embedding at `τ = tau`, `κ = 1/tau` (τκ unchanged at 1).
    fn set_initial_tau(&mut self, tau: T);

    /// Scale statistics of the iterate (diagnostics).
    fn scale_stats(&self) -> String {
        String::new()
    }

    /// `‖s‖∞·‖z‖∞` of the iterate (the scale of a KKT initializer).
    fn slack_dual_scale(&self) -> T;

    /// Current homogeneous τ, when the variables carry one.
    fn tau(&self) -> Option<T> {
        None
    }

    /// Enter the fixed-τ end phase (`Δτ = Δκ = 0`, μ over the cones).
    /// Returns false when the variables do not support it.
    fn freeze_tau(&mut self) -> bool {
        false
    }

    /// Whether the fixed-τ end phase is active.
    fn tau_frozen(&self) -> bool {
        false
    }

    /// Separate primal (x, s) and dual (z) step lengths: rescale `step` so
    /// that the returned common `α` applies each part's own longest step.
    /// The default keeps the common step.
    fn split_step(&self, _step: &mut Self, α: T, _data: &Self::D, _cones: &mut Self::C, _settings: &Self::SE) -> T {
        α
    }

    /// Shorten `α` by `shrink` until `τκ ≥ beta·μ` at the new point (at
    /// most 50 times). The default leaves `α` unchanged.
    fn taukappa_backtrack(&self, _step: &Self, α: T, _beta: T, _shrink: T, _cones: &Self::C) -> T {
        α
    }

    /// Independent scratch for a curve direction.
    fn new_like(&self) -> Self;
    /// Form (1-weight)*left + weight*right without changing either source.
    fn interpolate(&mut self, left: &Self, right: &Self, weight: T);

    /// Overwrite values with those from another object
    fn copy_from(&mut self, src: &Self);

    /// Apply NT scaling to a collection of cones
    fn scale_cones(&self, cones: &mut Self::C, μ: T, scaling_strategy: ScalingStrategy) -> bool;

    /// Compute the barrier function
    fn barrier(&self, step: &Self, α: T, cones: &mut Self::C) -> T;
}

/// Residuals for a conic optimization problem.
pub trait Residuals<T: FloatT> {
    /// associated problem data type
    type D: ProblemData<T>;
    /// associated variable type
    type V: Variables<T>;

    /// Compute residuals for the given variables.
    ///
    fn update(&mut self, variables: &Self::V, data: &Self::D);

    /// Update using the current cone worker pool when supported.
    fn update_with_pool(
        &mut self,
        variables: &Self::V,
        data: &Self::D,
        _pool: Option<std::sync::Arc<rayon::ThreadPool>>,
    ) {
        self.update(variables, data);
    }
}

/// KKT linear solver object.
pub trait KKTSystem<T: FloatT>: crate::solver::kkt::HasLinearSolverInfo {
    /// associated problem data type
    type D: ProblemData<T>;
    /// associated variable type
    type V: Variables<T>;
    /// associated cone type
    type C: ConeCollection<T>;
    /// associated settings type
    type SE: Settings<T>;

    /// Reset per-solve accounting and numerical retry state.
    fn reset_solve(&mut self) {}

    /// Binary64: from now on, continue stalled refinements with GMRES-IR
    /// (see [`crate::solver::kkt::KKTSolver::refine_further`]).
    fn refine_further(&mut self) {}

    /// Update the KKT system.   In particular, update KKT
    /// matrix entries with new variable and refactor.
    fn update(&mut self, data: &Self::D, cones: &Self::C, settings: &Self::SE) -> bool;

    /// Update with the independent affine RHS available for a batched solve.
    fn update_affine(
        &mut self,
        data: &Self::D,
        cones: &Self::C,
        _rhs: &Self::V,
        _variables: &Self::V,
        settings: &Self::SE,
    ) -> bool {
        self.update(data, cones, settings)
    }

    /// Solve the KKT system for the given RHS.
    #[allow(clippy::too_many_arguments)]
    fn solve(
        &mut self,
        step_lhs: &mut Self::V,
        step_rhs: &Self::V,
        data: &Self::D,
        variables: &Self::V,
        cones: &mut Self::C,
        step_direction: StepDirection,
        settings: &Self::SE,
    ) -> bool;

    /// Find an IP starting condition
    fn solve_initial_point(
        &mut self,
        variables: &mut Self::V,
        data: &Self::D,
        settings: &Self::SE,
    ) -> bool;
}

/// Printing functions for the solver's Info
pub trait InfoPrint<T>
where
    T: FloatT,
{
    /// associated problem data type
    type D: ProblemData<T>;
    /// associated cone type
    type C: ConeCollection<T>;
    /// associated settings type
    type SE: Settings<T>;

    /// Return the print target for the solver.
    fn print_target(&mut self) -> &mut dyn std::io::Write;

    /// Print the solver configuration, e.g. settings etc.
    /// This function is called once at the start of the solve.
    fn print_configuration(
        &mut self,
        settings: &Self::SE,
        data: &Self::D,
        cones: &Self::C,
    ) -> std::io::Result<()>;

    /// Print a header to appear at the top of progress information.
    fn print_status_header(&mut self, settings: &Self::SE) -> std::io::Result<()>;

    /// Print solver progress information.   Called once per iteration.
    fn print_status(&mut self, settings: &Self::SE) -> std::io::Result<()>;

    /// Print solver final status and other exit information.   Called at
    /// solver termination.
    fn print_footer(&mut self, settings: &Self::SE) -> std::io::Result<()>;
}

/// Internal information for the solver to monitor progress and check for termination.
pub trait Info<T>: InfoPrint<T> + Sized + Clone
where
    T: FloatT,
{
    /// associated variables type
    type V: Variables<T>;
    /// associated problem residuals type
    type R: Residuals<T>;

    /// Reset internal data and start the solve clock.
    fn reset(&mut self, timers: &mut Timers);

    /// Refresh metadata after lazy backend selection, pool changes or fallback.
    fn set_linear_solver_info(&mut self, _info: crate::solver::kkt::LinearSolverInfo) {}

    /// Final convergence checks, e.g. for "almost" convergence cases
    fn post_process(&mut self, residuals: &Self::R, settings: &Self::SE);

    /// Compute final values before solver termination
    fn finalize(&mut self, timers: &mut Timers);

    /// Update solver progress information
    fn update(
        &mut self,
        data: &mut Self::D,
        variables: &Self::V,
        residuals: &Self::R,
        timers: &Timers,
    );

    /// Update using the current cone worker pool when supported.
    /// Implementations run independent residual-norm scans concurrently;
    /// every scan keeps its serial reduction order, so results are
    /// bitwise identical to [`Info::update`].
    fn update_with_pool(
        &mut self,
        data: &mut Self::D,
        variables: &Self::V,
        residuals: &Self::R,
        timers: &Timers,
        _pool: Option<std::sync::Arc<rayon::ThreadPool>>,
    ) {
        self.update(data, variables, residuals, timers);
    }

    /// Return `true` if termination conditions have been reached.
    fn check_termination(&mut self, residuals: &Self::R, settings: &Self::SE, iter: u32) -> bool;

    /// save a prior iterate
    fn save_prev_iterate(&mut self, variables: &Self::V, prev_variables: &mut Self::V);
    /// restore a prior iterate
    fn reset_to_prev_iterate(&mut self, variables: &mut Self::V, prev_variables: &Self::V);

    /// Record some of the top level solver's choice of various
    /// scalars. `μ = ` normalized gap.  `α = ` computed step length.
    /// `σ = ` multiplier for the updated centering parameter.
    fn save_scalars(&mut self, μ: T, α: T, σ: T, iter: u32);

    /// Report the termination status
    fn get_status(&self) -> SolverStatus;
    /// Relative duality gap of the last evaluated iterate, if recorded.
    fn gap_rel(&self) -> Option<T> {
        None
    }
    /// Larger of the primal and dual feasibility residuals, if recorded.
    fn residual_max(&self) -> Option<T> {
        None
    }
    /// Unnormalized primal and dual residual norms `‖Ax+s−bτ‖`, `‖Px+A'z+qτ‖`
    /// of the scaled problem, if recorded. Exact Newton steps never increase
    /// them (each step scales both by `1 − α(1−σ)`).
    fn residual_norms(&self) -> Option<(T, T)> {
        None
    }
    /// Set the termination status
    fn set_status(&mut self, status: SolverStatus);
    /// Forget the status and progress history before a restart.
    fn restart(&mut self) {
        self.set_status(SolverStatus::Unsolved);
    }
}

/// Solution for a conic optimization problem.
pub trait Solution<T: FloatT> {
    /// Associated problem data type
    type D: ProblemData<T>;
    /// Associated problem variable type
    type V: Variables<T>;
    /// Associated progress information type
    type I: Info<T>;
    /// Associated solver settings settings
    type SE: Settings<T>;

    /// Compute solution from the Variables at solver termination
    fn post_process(
        &mut self,
        data: &Self::D,
        variables: &mut Self::V,
        info: &Self::I,
        settings: &Self::SE,
    );

    /// finalize the solution, e.g. extract final timing from info
    fn finalize(&mut self, info: &Self::I);

    /// Original-coordinate acceptance of the current iterate: the largest
    /// multiple of the `Solved` (`reduced`: `AlmostSolved`) tolerances over
    /// the residuals of the point that would be returned. At most one
    /// passes. `None` when the test does not apply.
    fn original_ratio(
        &mut self,
        _data: &Self::D,
        _variables: &Self::V,
        _settings: &Self::SE,
        _reduced: bool,
    ) -> Option<T> {
        None
    }
}

/// Settings for a conic optimization problem.
///
/// Implementers of this trait can define any internal or problem
/// specific settings they wish.   They must, however, also maintain
/// a settings object of type [`CoreSettings`](crate::solver::core::CoreSettings)
/// and return this to the solver internally.
pub trait Settings<T: FloatT>: Sized + Clone {
    /// Return the core settings.
    fn core(&self) -> &CoreSettings<T>;

    /// sanity check the settings
    fn validate(&self) -> Result<(), SettingsError>;

    /// check that the settings are a valid update to some previous
    /// version.   Used to ensure that settings used only at solver
    /// initialization are not changed during the solve.
    fn validate_as_update(&self, prev: &Self) -> Result<(), SettingsError>;
}
