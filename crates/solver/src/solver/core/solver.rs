use self::internal::*;
use super::callbacks::{Callback, TerminationCallback};
use super::{traits::*, SettingsError};
use crate::algebra::*;
use crate::solver::core::callbacks::SolverCallbacks;
#[cfg(feature = "serde")]
use crate::solver::SolverError;
use crate::timers::*;
use std::io::Write;

// ---------------------------------
// Solver status type
// ---------------------------------

/// Status of solver at termination
#[repr(C)]
#[derive(PartialEq, Eq, Clone, Debug, Copy, Default)]
pub enum SolverStatus {
    /// Problem is not solved (solver hasn't run).
    #[default]
    Unsolved = 0,
    /// Solver terminated with a solution.
    Solved,
    /// Problem is primal infeasible.  Solution returned is a certificate of primal infeasibility.
    PrimalInfeasible,
    /// Problem is dual infeasible.  Solution returned is a certificate of dual infeasibility.
    DualInfeasible,
    /// Solver terminated with a solution (reduced accuracy)
    AlmostSolved,
    /// Problem is primal infeasible.  Solution returned is a certificate of primal infeasibility (reduced accuracy).
    AlmostPrimalInfeasible,
    /// Problem is dual infeasible.  Solution returned is a certificate of dual infeasibility (reduced accuracy).
    AlmostDualInfeasible,
    /// Iteration limit reached before solution or infeasibility certificate found.
    MaxIterations,
    /// Time limit reached before solution or infeasibility certificate found.
    MaxTime,
    /// Solver terminated with a numerical error
    NumericalError,
    /// Solver terminated due to lack of progress.
    InsufficientProgress,
    /// Solver terminated by user callback
    CallbackTerminated,
}

impl SolverStatus {
    pub(crate) fn is_infeasible(&self) -> bool {
        matches!(
            *self,
            |SolverStatus::PrimalInfeasible| SolverStatus::DualInfeasible
                | SolverStatus::AlmostPrimalInfeasible
                | SolverStatus::AlmostDualInfeasible
        )
    }

    pub(crate) fn is_errored(&self) -> bool {
        // status is any of the error codes
        matches!(
            *self,
            SolverStatus::NumericalError | SolverStatus::InsufficientProgress
        )
    }
}

#[repr(u32)]
#[derive(PartialEq, Eq, Clone, Debug, Copy)]
pub enum StepDirection {
    Affine,
    Combined,
}

/// Scaling strategy used by the solver when
/// linearizing centrality conditions.
#[repr(u32)]
#[derive(PartialEq, Eq, Clone, Debug, Copy)]
pub enum ScalingStrategy {
    PrimalDual,
    Dual,
}

/// An enum for reporting strategy checkpointing
#[repr(u32)]
#[derive(PartialEq, Eq, Clone, Debug, Copy)]
enum StrategyCheckpoint {
    Update(ScalingStrategy), // Checkpoint is suggesting a new ScalingStrategy
    NoUpdate,                // Checkpoint recommends no change to ScalingStrategy
    Fail,                    // Checkpoint found a problem but no more ScalingStrategies to try
}

impl StrategyCheckpoint {
    fn synchronized(self) -> Self {
        let action = match self {
            Self::NoUpdate => 0,
            Self::Fail => 1,
            Self::Update(scaling) => 2 + scaling as u32,
        };
        crate::mpi::assert_agree(action, "inconsistent replicated solver strategy decision");
        self
    }
}

impl std::fmt::Display for SolverStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

/// JSON file read/write trait for solver data.
/// Only available with the "serde" feature enabled.
#[cfg(feature = "serde")]
pub trait SolverJSONReadWrite<T>: Sized
where
    T: FloatT,
{
    /// Write internal problem data to a JSON file. Sampled solvers return
    /// `Unsupported`; retain and write the original `JsonProblem` to preserve
    /// authoritative factors.
    fn save_to_file(&self, file: &mut std::fs::File) -> Result<(), std::io::Error>;
    /// load problem data from a JSON file previously saved using [`save_to_file`](self::SolverJSONReadWrite::save_to_file)
    fn load_from_file(
        file: &mut std::fs::File,
        settings: Option<crate::solver::DefaultSettings<T>>,
    ) -> Result<Self, SolverError>;
}

// ---------------------------------
// top level solver container type
// ---------------------------------

// The top-level solver.

// This trait is defined with a collection of mutually interacting associated types.
// See the [`DefaultSolver`](crate::solver::default) for an example.

pub struct Solver<T, D, V, R, K, C, I, SO, SE>
where
    SE: Settings<T>,
    T: FloatT,
{
    pub data: D,
    pub variables: V,
    pub residuals: R,
    pub kktsystem: K,
    pub cones: C,
    pub step_lhs: V,
    pub step_rhs: V,
    pub prev_vars: V,
    pub info: I,
    pub solution: SO,
    pub(crate) settings: SE, // not public to avoid unchecked modifications
    pub timers: Option<Timers>,
    pub(crate) callbacks: SolverCallbacks<I>,
    /// Starting τ of the current attempt (`initial_tau` until a restart).
    pub(crate) start_tau: Option<T>,
    pub(crate) phantom: std::marker::PhantomData<T>,
}

fn _print_banner(out: &mut dyn Write, is_verbose: bool) -> std::io::Result<()> {
    if !is_verbose {
        return std::io::Result::Ok(());
    }
    let rule = "-".repeat(61);
    writeln!(out, "{rule}")?;
    writeln!(
        out,
        "SDPX v{} - homogeneous interior-point conic solver{}",
        crate::VERSION,
        if cfg!(debug_assertions) {
            " (debug build)"
        } else {
            ""
        }
    )?;
    writeln!(
        out,
        "Float64 and MPFR arithmetic; IPM core derived from Clarabel.rs"
    )?;
    writeln!(out, "(Apache-2.0, see provenance/upstream.json)")?;
    writeln!(out, "{rule}")?;
    std::io::Result::Ok(())
}

impl<T, D, V, R, K, C, I, SO, SE> Solver<T, D, V, R, K, C, I, SO, SE>
where
    I: Info<T, D = D, V = V, R = R, C = C, SE = SE>,
    SE: Settings<T>,
    T: FloatT,
{
    /// Create a new solver object
    pub fn set_termination_callback(&mut self, callback: impl TerminationCallback<I> + 'static) {
        self.callbacks.termination_callback = Callback::Rust(Box::new(callback));
    }

    pub fn unset_termination_callback(&mut self) {
        self.callbacks.termination_callback = Callback::None;
    }

    /// Write the accepted iterate to `path` every `every` iterations
    /// (replicated solvers only). Call [`Self::check_restart`] before
    /// restarting from such a file.
    pub fn set_checkpoint(&mut self, path: impl Into<std::path::PathBuf>, every: u32) {
        self.callbacks.checkpoint.path = Some(path.into());
        self.callbacks.checkpoint.every = every.max(1);
    }

    /// Start the next `solve` from a checkpoint instead of the default start.
    pub fn set_restart(&mut self, path: impl Into<std::path::PathBuf>) {
        self.callbacks.checkpoint.restart = Some(path.into());
    }

    pub fn settings(&self) -> &SE {
        &self.settings
    }

    pub fn update_settings(&mut self, settings: SE) -> Result<(), SettingsError> {
        settings.validate_as_update(&self.settings)?;
        self.settings = settings;
        Ok(())
    }
}

impl<T, D, V, R, K, C, I, SO, SE> Solver<T, D, V, R, K, C, I, SO, SE>
where
    D: ProblemData<T>,
    V: Variables<T, D = D>,
    SE: Settings<T>,
    T: FloatT,
{
    /// Check that the restart file (if any) matches this problem and precision.
    pub fn check_restart(&self) -> std::io::Result<()> {
        match &self.callbacks.checkpoint.restart {
            None => Ok(()),
            Some(path) => self
                .variables
                .new_like()
                .read_checkpoint(&self.data, path)
                .map(|_| ()),
        }
    }
}

// ---------------------------------
// IPSolver trait and its standard implementation.
// ---------------------------------

/// An interior point solver implementing a predictor-corrector scheme
//
// Only the main solver function lives in IPSolver, since this is the
// only publicly facing trait we want to give the solver.   Additional
// internal functionality for the top level solver object is implemented
// for the IPSolverUtilities trait below, upon which IPSolver depends
pub trait IPSolver<T, D, V, R, K, C, I, SO, SE> {
    /// Run the solver
    fn solve(&mut self);
}

impl<T, D, V, R, K, C, I, SO, SE> IPSolver<T, D, V, R, K, C, I, SO, SE>
    for Solver<T, D, V, R, K, C, I, SO, SE>
where
    T: FloatT,
    D: ProblemData<T, V = V>,
    V: Variables<T, D = D, R = R, C = C, SE = SE>,
    R: Residuals<T, D = D, V = V>,
    K: KKTSystem<T, D = D, V = V, C = C, SE = SE>,
    C: ConeCollection<T>,
    I: Info<T, D = D, V = V, R = R, C = C, SE = SE>,
    SO: Solution<T, D = D, V = V, I = I, SE = SE>,
    SE: Settings<T>,
{
    fn solve(&mut self) {
        let _receipt_scope = crate::receipt::Scope::begin();
        // Long vector operations on this thread use the cone worker pool.
        let _vector_pool = crate::algebra::VectorPoolGuard::install(self.cones.worker_pool());
        self.kktsystem.reset_solve();

        // Timers live in an Option so the solver's fields can be borrowed
        // freely while they run.
        let mut timers = self.timers.take().unwrap();
        _print_banner(self.info.print_target(), self.settings.core().verbose).unwrap();
        self.info
            .print_configuration(&self.settings, &self.data, &self.cones)
            .unwrap();
        self.info.print_status_header(&self.settings).unwrap();
        self.info.reset(&mut timers);

        let mut state = IterationState::new(self.cones.supports_primal_dual());
        let mut curve = CurveSearch::new(&self.variables, &self.cones);

        timeit! {"solve"; {
        self.start_tau = None;
        timeit! {"default start"; {
            self.start_point();
        }}
        timeit! {"IP iteration"; {
        'attempts: loop {
        let mut chase = TauChase::default();
        let mut chase_tau = None;
        loop {
            self.evaluate(&mut state, &timers);
            if self.terminate(&mut state).stop() {
                break;
            }
            if self.start_tau.is_none()
                && self.settings.core().auto_start_scale
                && self.callbacks.checkpoint.restart.is_none()
            {
                let tau0 = self.settings.core().initial_tau;
                if let (Some(tau), Some(gap)) = (self.variables.tau(), self.info.gap_rel()) {
                    if chase.observe(gap, state.μ, tau, tau0) {
                        chase_tau = Some(TauChase::restart_tau(tau));
                        break;
                    }
                }
            }
            self.write_checkpoint(state.iter);
            if !self.scale(&state) {
                break;
            }
            // Only iterations that update the KKT system are counted.
            state.iter += 1;
            if state.iter <= 2 {
                crate::receipt::memory_mark(if state.iter == 1 { "iteration 1 start" } else { "iteration 2 start" });
            }
            match self.direction(&mut state, &mut curve) {
                Flow::Proceed => {}
                Flow::Retry => continue,
                Flow::Stop => break,
            }
            match self.step_length(&mut state, &mut curve) {
                Flow::Proceed => {}
                Flow::Retry => continue,
                Flow::Stop => break,
            }
            // Keep the previous iterate in case the next one is a dud.
            self.info.save_prev_iterate(&self.variables, &mut self.prev_vars);
            timeit! {"iterate update"; {
            self.variables
                .add_step_with_pool(&self.step_lhs, state.α, self.cones.worker_pool());
            }}
        }
        // A large starting scale overshoots on some problems: restart with
        // a 1e10 times larger τ₀ (up to the unit start) within the same
        // iteration budget, as SDPA's guidance to retune lambdaStar.
        let tau = self.start_tau.unwrap_or(self.settings.core().initial_tau);
        let failed = matches!(
            self.info.get_status(),
            SolverStatus::InsufficientProgress | SolverStatus::NumericalError
        );
        if chase_tau.is_none()
            && !(failed
                && tau < T::one()
                && state.iter < self.settings.core().max_iter
                && self.callbacks.checkpoint.restart.is_none())
        {
            break 'attempts;
        }
        let next = match chase_tau {
            Some(found) => found,
            None => T::min(T::one(), tau * (1e10).as_T()),
        };
        if self.settings.core().verbose {
            let status = self.info.get_status();
            let _ = writeln!(
                self.info.print_target(),
                "restart: {} at iteration {}, initial tau {:.1e} -> {:.1e}",
                if chase_tau.is_some() { "tau chase".to_string() } else { format!("{status:?}") },
                state.iter,
                tau,
                next
            );
        }
        self.start_tau = Some(next);
        self.info.restart();
        self.kktsystem.reset_solve();
        let iter = state.iter;
        state = IterationState::new(self.cones.supports_primal_dual());
        state.iter = iter;
        self.default_start();
        }
        }}
        }}

        // Without a final step the last line reports the stopping iterate.
        if state.α.is_zero() {
            self.info
                .save_scalars(state.μ, state.α, state.σ, state.iter);
            self.info.print_status(&self.settings).unwrap();
        }
        timeit! {"post-process"; {
            self.finish(&timers);
        }}
        timers.stop_solve();
        self.info.finalize(&mut timers);
        crate::receipt::memory_mark("solved");
        self.solution.finalize(&self.info);
        self.info.print_footer(&self.settings).unwrap();
        self.timers.replace(timers);
    }
}

/// Scalars carried from one interior-point iteration to the next.
struct IterationState<T> {
    iter: u32,
    μ: T,
    σ: T,
    α: T,
    scaling: ScalingStrategy,
}

impl<T: FloatT> IterationState<T> {
    fn new(primal_dual: bool) -> Self {
        Self {
            iter: 0,
            μ: T::zero(),
            σ: T::one(),
            α: T::zero(),
            scaling: if primal_dual {
                ScalingStrategy::PrimalDual
            } else {
                ScalingStrategy::Dual
            },
        }
    }
}

/// What the driver does after a stage of an iteration.
enum Flow {
    /// Continue with the next stage.
    Proceed,
    /// Start a new iteration (the scaling strategy changed).
    Retry,
    /// Leave the main loop.
    Stop,
}

impl Flow {
    fn stop(&self) -> bool {
        matches!(self, Flow::Stop)
    }
}

/// Quadratic predictor-corrector curve `t·affine + t²·(combined − affine)`,
/// after Hypatia's curve search, from the two existing directions (no extra
/// KKT solves). Binary64 with symmetric cones only: its extra PSD step
/// searches regress the 512-bit Ising workload.
struct CurveSearch<V> {
    affine: Option<V>,
    trial: Option<V>,
    /// Corrected direction of the Gondzio centrality correctors (symmetric
    /// cones with orthant or second-order rows, any precision).
    corrected: Option<V>,
}

impl<V> CurveSearch<V> {
    fn new<T: FloatT, C: ConeCollection<T>>(like: &V, cones: &C) -> Self
    where
        V: Variables<T>,
    {
        let enabled = T::precision_bits() <= 53 && cones.all_symmetric();
        Self {
            affine: enabled.then(|| like.new_like()),
            trial: enabled.then(|| like.new_like()),
            corrected: (cones.all_symmetric() && cones.has_correctable()).then(|| like.new_like()),
        }
    }
}

// Encapsulate the internal helpers trait in a private module
// so it doesn't get exported
mod internal {
    use super::*;

    pub(super) trait IPSolverInternals<T, D, V, R, K, C, I, SO, SE> {
        /// Initial point: the default start, or a checkpoint to restart from.
        fn start_point(&mut self);

        /// Residuals, μ and convergence measures of the current iterate.
        fn evaluate(&mut self, state: &mut IterationState<T>, timers: &Timers);

        /// Callback and internal termination tests (with strategy fallback).
        fn terminate(&mut self, state: &mut IterationState<T>) -> Flow;

        /// Write the accepted iterate when a checkpoint is due.
        fn write_checkpoint(&mut self, iter: u32);

        /// Update the cone scalings; `false` ends the solve.
        fn scale(&mut self, state: &IterationState<T>) -> bool;

        /// Predictor and corrector directions.
        fn direction(&mut self, state: &mut IterationState<T>, curve: &mut CurveSearch<V>) -> Flow;

        /// Final step length, optionally along the predictor-corrector curve.
        fn step_length(
            &mut self,
            state: &mut IterationState<T>,
            curve: &mut CurveSearch<V>,
        ) -> Flow;

        /// Gondzio multiple centrality correctors on the combined direction.
        fn centrality_correctors(&mut self, state: &mut IterationState<T>, corrected: &mut V);

        /// Status refinement and solution extraction after the main loop.
        fn finish(&mut self, timers: &Timers);

        /// Find an initial condition
        fn default_start(&mut self);

        /// Compute a centering parameter
        fn centering_parameter(&self, α: T) -> T;

        /// Compute the current step length
        fn get_step_length(&mut self, step_direction: StepDirection, scaling: ScalingStrategy)
            -> T;

        /// backtrack a step direction to the barrier
        fn backtrack_step_to_barrier(&mut self, αinit: T) -> T;

        /// Scaling strategy checkpointing functions
        fn strategy_checkpoint_insufficient_progress(
            &mut self,
            scaling: ScalingStrategy,
        ) -> StrategyCheckpoint;

        fn strategy_checkpoint_numerical_error(
            &mut self,
            is_kkt_solve_success: bool,
            scaling: ScalingStrategy,
        ) -> StrategyCheckpoint;

        fn strategy_checkpoint_small_step(
            &mut self,
            α: T,
            scaling: ScalingStrategy,
        ) -> StrategyCheckpoint;

        fn strategy_checkpoint_is_scaling_success(
            &mut self,
            is_scaling_success: bool,
            scaling: ScalingStrategy,
        ) -> StrategyCheckpoint;
    }

    impl<T, D, V, R, K, C, I, SO, SE> IPSolverInternals<T, D, V, R, K, C, I, SO, SE>
        for Solver<T, D, V, R, K, C, I, SO, SE>
    where
        T: FloatT,
        D: ProblemData<T, V = V>,
        V: Variables<T, D = D, R = R, C = C, SE = SE>,
        R: Residuals<T, D = D, V = V>,
        K: KKTSystem<T, D = D, V = V, C = C, SE = SE>,
        C: ConeCollection<T>,
        I: Info<T, D = D, V = V, R = R, C = C, SE = SE>,
        SO: Solution<T, D = D, V = V, I = I, SE = SE>,
        SE: Settings<T>,
    {
        fn start_point(&mut self) {
            self.default_start();
            if let Some(path) = self.callbacks.checkpoint.restart.clone() {
                // validated by `check_restart`; the same file is read on every rank
                self.variables
                    .read_checkpoint(&self.data, &path)
                    .unwrap_or_else(|e| panic!("restart from {}: {e}", path.display()));
            }
        }

        fn evaluate(&mut self, state: &mut IterationState<T>, timers: &Timers) {
            timeit! {"residual update"; {
            self.residuals.update_with_pool(&self.variables, &self.data, self.cones.worker_pool());
            }}
            timeit! {"mu+info"; {
            state.μ = self.variables.calc_mu(&self.residuals, &self.cones);
            // Record the scalars of the latest iteration (μ at iteration zero).
            self.info.save_scalars(state.μ, state.α, state.σ, state.iter);
            self.info.update_with_pool(
                &mut self.data,
                &self.variables,
                &self.residuals,
                timers,
                self.cones.worker_pool(),
            );
            }}
            self.info.print_status(&self.settings).unwrap();
        }

        fn terminate(&mut self, state: &mut IterationState<T>) -> Flow {
            let callback_stop = self.callbacks.check_termination(&self.info);
            if crate::mpi::any_true(callback_stop) {
                self.info.set_status(SolverStatus::CallbackTerminated);
                return Flow::Stop;
            }
            let is_done = self
                .info
                .check_termination(&self.residuals, &self.settings, state.iter);
            crate::mpi::assert_agree(
                self.info.get_status() as u32,
                "inconsistent replicated solver termination status",
            );
            if !is_done {
                return Flow::Proceed;
            }
            // Slow progress may continue under another scaling strategy.
            match self
                .strategy_checkpoint_insufficient_progress(state.scaling)
                .synchronized()
            {
                StrategyCheckpoint::NoUpdate | StrategyCheckpoint::Fail => Flow::Stop,
                StrategyCheckpoint::Update(s) => {
                    state.scaling = s;
                    Flow::Retry
                }
            }
        }

        fn write_checkpoint(&mut self, iter: u32) {
            // The iterate has passed the progress checks: it is accepted.
            let Some(path) = self.callbacks.checkpoint.path.as_deref() else {
                return;
            };
            if iter > 0
                && iter % self.callbacks.checkpoint.every.max(1) == 0
                && crate::mpi::is_root()
            {
                if let Err(e) = self.variables.write_checkpoint(&self.data, path, iter) {
                    eprintln!("checkpoint {}: {e}", path.display());
                }
            }
        }

        fn scale(&mut self, state: &IterationState<T>) -> bool {
            let is_scaling_success;
            timeit! {"scale cones"; {
                is_scaling_success = self.variables.scale_cones(&mut self.cones, state.μ, state.scaling);
            }}
            let is_scaling_success = crate::mpi::all_succeeded(is_scaling_success);
            match self
                .strategy_checkpoint_is_scaling_success(is_scaling_success, state.scaling)
                .synchronized()
            {
                StrategyCheckpoint::Fail => false,
                StrategyCheckpoint::NoUpdate => true,
                StrategyCheckpoint::Update(_) => unreachable!(),
            }
        }

        fn direction(&mut self, state: &mut IterationState<T>, curve: &mut CurveSearch<V>) -> Flow {
            // The affine RHS sits beside the constant RHS so the KKT update
            // reuses one factorization and multi-RHS solve.
            timeit! {"affine rhs"; {
            self.step_rhs
                .affine_step_rhs(&self.residuals, &self.variables, &self.cones);
            }}
            let mut ok: bool;
            timeit! {"kkt update"; {
                ok = self.kktsystem.update_affine(&self.data, &self.cones, &self.step_rhs, &self.variables, &self.settings);
            }}
            ok = crate::mpi::all_succeeded(ok);
            timeit! {"kkt solve"; {
                ok = ok && self.kktsystem.solve(
                    &mut self.step_lhs,
                    &self.step_rhs,
                    &self.data,
                    &self.variables,
                    &mut self.cones,
                    StepDirection::Affine,
                    &self.settings,
                );
            }}
            ok = crate::mpi::all_succeeded(ok);

            // The corrector runs only after a successful predictor.
            if ok {
                // Keep the raw direction before prepared cone operations consume it.
                if let Some(affine) = &mut curve.affine {
                    affine.copy_from(&self.step_lhs);
                }
                let iter = state.iter;
                timeit! {"affine step len"; {
                state.α = if iter > 1 {
                    self.variables.prepare_affine_step_length(&mut self.step_lhs, &mut self.cones, &self.settings)
                } else {
                    self.get_step_length(StepDirection::Affine, state.scaling)
                };
                }}
                state.σ = self.centering_parameter(state.α);
                timeit! {"combined rhs"; {
                if iter > 1 {
                    self.step_rhs.combined_step_rhs_prepared(&self.residuals, &self.variables,
                        &mut self.cones, &mut self.step_lhs, state.σ, state.μ);
                } else {
                    // A reduced Mehrotra correction in the first iteration
                    // accommodates badly centred starting points.
                    self.step_rhs.combined_step_rhs(
                        &self.residuals,
                        &self.variables,
                        &mut self.cones,
                        &mut self.step_lhs,
                        state.σ,
                        state.μ,
                        state.α,
                    );
                }
                }}
                timeit! {"kkt solve" ; {
                    ok = self.kktsystem.solve(
                        &mut self.step_lhs,
                        &self.step_rhs,
                        &self.data,
                        &self.variables,
                        &mut self.cones,
                        StepDirection::Combined,
                        &self.settings,
                    );
                }}
            }
            ok = crate::mpi::all_succeeded(ok);
            match self
                .strategy_checkpoint_numerical_error(ok, state.scaling)
                .synchronized()
            {
                StrategyCheckpoint::NoUpdate => Flow::Proceed,
                StrategyCheckpoint::Update(s) => {
                    state.α = T::zero();
                    state.scaling = s;
                    Flow::Retry
                }
                StrategyCheckpoint::Fail => {
                    state.α = T::zero();
                    Flow::Stop
                }
            }
        }

        fn step_length(
            &mut self,
            state: &mut IterationState<T>,
            curve: &mut CurveSearch<V>,
        ) -> Flow {
            let flow;
            timeit! {"final step len"; {
            state.α = self.get_step_length(StepDirection::Combined, state.scaling);
            if let Some(corrected) = &mut curve.corrected {
                self.centrality_correctors(state, corrected);
            }
            let (α, σ) = (state.α, state.σ);
            if crate::mpi::agreed_branch(curve.affine.is_some() && α > T::zero() && α < (0.9).as_T()) {
                let affine = curve.affine.as_ref().unwrap();
                let trial = curve.trial.as_mut().unwrap();
                for fraction in [0.5, 0.25] {
                    let t = α + (T::one() - α) * T::from_f64(fraction).unwrap();
                    // Only a predicted gain of more than 1% is worth a trial.
                    if crate::mpi::agreed_branch(t * (T::one() - σ * t) <= T::from_f64(1.01).unwrap() * α * (T::one() - σ)) {
                        continue;
                    }
                    trial.interpolate(affine, &self.step_lhs, t);
                    let bound = self.variables.calc_step_length(trial, &mut self.cones,
                        &self.settings, StepDirection::Combined);
                    if crate::mpi::agreed_branch(t <= bound) {
                        self.step_lhs.copy_from(trial);
                        state.α = t;
                        break;
                    }
                }
            }
            let beta = self.settings.core().taukappa_proximity;
            if beta > T::zero() && state.α > T::zero() {
                let shrink = self.settings.core().linesearch_backtrack_step;
                state.α = self.variables.taukappa_backtrack(&self.step_lhs, state.α, beta, shrink, &self.cones);
            }
            flow = match self.strategy_checkpoint_small_step(state.α, state.scaling).synchronized() {
                StrategyCheckpoint::NoUpdate => Flow::Proceed,
                StrategyCheckpoint::Update(s) => {
                    state.α = T::zero();
                    state.scaling = s;
                    Flow::Retry
                }
                StrategyCheckpoint::Fail => {
                    state.α = T::zero();
                    Flow::Stop
                }
            };
            }}
            flow
        }

        fn centrality_correctors(&mut self, state: &mut IterationState<T>, corrected: &mut V) {
            // Colombo & Gondzio (2008): aim for a longer step α̃, push the
            // trial complementarity products back into the centrality band
            // and keep the corrected direction only if the step grows by 1%.
            // Each corrector reuses the factorization (one more solve).
            // A step of at least 0.9 already gains little from another solve.
            const MAX_CORRECTORS: usize = 2;
            for _ in 0..MAX_CORRECTORS {
                let α = state.α;
                if !crate::mpi::agreed_branch(α > T::zero() && α < (0.9).as_T()) {
                    return;
                }
                let target = T::min(T::one(), α * (1.5).as_T() + (0.3).as_T());
                let changed;
                timeit! {"corrector rhs"; {
                    changed = self.step_rhs.centrality_correction(
                        &self.step_lhs,
                        &self.variables,
                        &mut self.cones,
                        target,
                        state.σ * state.μ,
                    );
                }}
                if !crate::mpi::agreed_branch(changed) {
                    return;
                }
                let ok;
                timeit! {"corrector solve"; {
                    ok = self.kktsystem.solve(
                        corrected,
                        &self.step_rhs,
                        &self.data,
                        &self.variables,
                        &mut self.cones,
                        StepDirection::Combined,
                        &self.settings,
                    );
                }}
                if !crate::mpi::all_succeeded(ok) {
                    return;
                }
                let α_new;
                timeit! {"corrector step len"; {
                    α_new = self.variables.calc_step_length(
                        corrected,
                        &mut self.cones,
                        &self.settings,
                        StepDirection::Combined,
                    );
                }}
                if !crate::mpi::agreed_branch(α_new >= α * (1.01).as_T()) {
                    return;
                }
                self.step_lhs.copy_from(corrected);
                state.α = α_new;
                crate::receipt::phase_record("corrector accepted", std::time::Duration::ZERO);
            }
        }

        fn finish(&mut self, timers: &Timers) {
            self.info
                .set_linear_solver_info(self.kktsystem.linear_solver_info());
            if self.info.get_status() == SolverStatus::InsufficientProgress {
                // Rollback restored the iterate; recompute its residuals and
                // infeasibility products before the reduced convergence test.
                self.residuals.update_with_pool(
                    &self.variables,
                    &self.data,
                    self.cones.worker_pool(),
                );
                self.info.update_with_pool(
                    &mut self.data,
                    &self.variables,
                    &self.residuals,
                    timers,
                    self.cones.worker_pool(),
                );
            }
            // "Almost" convergence check, then solution extraction.
            self.info.post_process(&self.residuals, &self.settings);
            self.solution
                .post_process(&self.data, &mut self.variables, &self.info, &self.settings);
        }

        fn default_start(&mut self) {
            if self.cones.all_symmetric() {
                // set all scalings to identity (or zero for the zero cone)
                self.cones.reset_scaling();
                // Refactor
                let timer = crate::receipt::start();
                self.kktsystem
                    .update(&self.data, &self.cones, &self.settings);
                crate::receipt::finish("start.update", timer);
                // solve for primal/dual initial points via KKT
                let timer = crate::receipt::start();
                let ok = self.kktsystem.solve_initial_point(
                    &mut self.variables,
                    &self.data,
                    &self.settings,
                );
                crate::receipt::finish("start.solve", timer);
                // fix up (z,s) so that they are in the cone
                let timer = crate::receipt::start();
                self.variables.symmetric_initialization(&mut self.cones);
                crate::receipt::finish("start.shift", timer);
                // a failed or degenerate KKT initializer is not a valid
                // starting point; fall back to the unit interior point
                if !ok {
                    self.variables.unit_initialization(&self.cones);
                }
            } else {
                // Assigns unit (z,s) and zeros the primal variables
                self.variables.unit_initialization(&self.cones);
            }
            let tau = self.start_tau.unwrap_or(self.settings.core().initial_tau);
            if tau != T::one() {
                self.variables.set_initial_tau(tau);
            }
        }

        fn centering_parameter(&self, α: T) -> T {
            T::max(
                T::powi(T::one() - α, 3),
                self.settings.core().centering_floor,
            )
        }

        fn get_step_length(
            &mut self,
            step_direction: StepDirection,
            scaling: ScalingStrategy,
        ) -> T {
            //step length to stay within the cones
            let mut α = self.variables.calc_step_length(
                &self.step_lhs,
                &mut self.cones,
                &self.settings,
                step_direction,
            );

            // additional barrier function limits for asymmetric cones
            if !self.cones.all_symmetric()
                && step_direction == StepDirection::Combined
                && scaling == ScalingStrategy::Dual
            {
                let αinit = α;
                α = self.backtrack_step_to_barrier(αinit);
            }
            α
        }

        fn backtrack_step_to_barrier(&mut self, αinit: T) -> T {
            let step = self.settings.core().linesearch_backtrack_step;
            let mut α = αinit;

            for _ in 0..50 {
                let barrier = self.variables.barrier(&self.step_lhs, α, &mut self.cones);
                if barrier < T::one() {
                    return α;
                } else {
                    α = step * α;
                }
            }
            α
        }

        fn strategy_checkpoint_insufficient_progress(
            &mut self,
            scaling: ScalingStrategy,
        ) -> StrategyCheckpoint {
            let output;
            if self.info.get_status() != SolverStatus::InsufficientProgress {
                // there is no problem, so nothing to do
                output = StrategyCheckpoint::NoUpdate;
            } else {
                // recover old iterate since "insufficient progress" often
                // involves actual degradation of results
                self.info
                    .reset_to_prev_iterate(&mut self.variables, &self.prev_vars);

                // If problem is asymmetric, we can try to continue with the dual-only strategy
                if !self.cones.all_symmetric() && (scaling == ScalingStrategy::PrimalDual) {
                    self.info.set_status(SolverStatus::Unsolved);
                    output = StrategyCheckpoint::Update(ScalingStrategy::Dual);
                } else {
                    output = StrategyCheckpoint::Fail;
                }
            }
            output
        }

        fn strategy_checkpoint_numerical_error(
            &mut self,
            is_kkt_solve_success: bool,
            scaling: ScalingStrategy,
        ) -> StrategyCheckpoint {
            let output;
            // No update if kkt updates successfully
            if is_kkt_solve_success {
                output = StrategyCheckpoint::NoUpdate;
            }
            // If problem is asymmetric, we can try to continue with the dual-only strategy
            else if !self.cones.all_symmetric() && (scaling == ScalingStrategy::PrimalDual) {
                output = StrategyCheckpoint::Update(ScalingStrategy::Dual);
            } else {
                // out of tricks.  Bail out with an error
                self.info.set_status(SolverStatus::NumericalError);
                output = StrategyCheckpoint::Fail;
            }
            output
        }

        fn strategy_checkpoint_small_step(
            &mut self,
            α: T,
            scaling: ScalingStrategy,
        ) -> StrategyCheckpoint {
            let output;

            if !self.cones.all_symmetric()
                && scaling == ScalingStrategy::PrimalDual
                && α < self.settings.core().min_switch_step_length
            {
                output = StrategyCheckpoint::Update(ScalingStrategy::Dual);
            } else if α <= T::max(T::zero(), self.settings.core().min_terminate_step_length) {
                self.info.set_status(SolverStatus::InsufficientProgress);
                output = StrategyCheckpoint::Fail;
            } else {
                output = StrategyCheckpoint::NoUpdate;
            }

            output
        }

        fn strategy_checkpoint_is_scaling_success(
            &mut self,
            is_scaling_success: bool,
            _scaling: ScalingStrategy,
        ) -> StrategyCheckpoint {
            if is_scaling_success {
                StrategyCheckpoint::NoUpdate
            } else {
                self.info.set_status(SolverStatus::NumericalError);
                StrategyCheckpoint::Fail
            }
        }
    } // end trait impl
} //end internals module

/// Detects a τ chase on a unit-scale first attempt: since the relative gap
/// last improved tenfold, μ fell by 1e10 while τ fell below `1e-4·τ₀`. On
/// Λ27 the gap stalls near 1e-13 while μ keeps falling (451 iterations from
/// τ₀ = 1, 181 from 1e-30).
#[derive(Default)]
struct TauChase<T> {
    anchor: Option<(T, T)>,
}

impl<T: FloatT> TauChase<T> {
    fn observe(&mut self, gap: T, mu: T, tau: T, tau0: T) -> bool {
        if !(gap.is_finite() && mu.is_finite() && tau.is_finite() && tau > T::zero())
            || tau0 < (1e-10).as_T()
        {
            return false;
        }
        match self.anchor {
            Some((best, _)) if gap >= best * (0.1).as_T() => {}
            _ => {
                self.anchor = Some((gap, mu));
                return false;
            }
        }
        let (_, mu0) = self.anchor.unwrap();
        mu <= mu0 * (1e-10).as_T() && tau < tau0 * (1e-4).as_T()
    }

    /// Restart scale: the current τ, at most `eps^(1/8)` (about 1e-29 at
    /// 768 bits, 0.01 in binary64).
    fn restart_tau(tau: T) -> T {
        T::min(tau, T::epsilon().sqrt().sqrt().sqrt())
    }
}

#[cfg(test)]
mod tau_chase_tests {
    use super::TauChase;

    #[test]
    fn fires_on_gap_stall_with_falling_mu_only() {
        // Healthy run: gap and μ fall together.
        let mut c = TauChase::<f64>::default();
        for k in 0..60 {
            let v = 10f64.powi(-k / 2);
            assert!(!c.observe(v, v, v, 1.0));
        }
        // Stall: gap flat at 1e-13 while μ falls; fires once μ drops 1e10.
        let mut c = TauChase::<f64>::default();
        let mut fired = None;
        for k in 0..40 {
            let mu = 1e-20 * 10f64.powi(-k);
            if c.observe(1e-13, mu, 1e-11, 1.0) {
                fired = Some(k);
                break;
            }
        }
        assert_eq!(fired, Some(10));
        // A small user start disables the rule.
        let mut c = TauChase::<f64>::default();
        assert!((0..40).all(|k| !c.observe(1e-13, 10f64.powi(-k), 1e-40, 1e-30)));
        assert_eq!(TauChase::<f64>::restart_tau(1e-11), 1e-11);
    }
}
