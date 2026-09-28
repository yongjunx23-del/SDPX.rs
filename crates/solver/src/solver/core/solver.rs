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
        if let Some(world) = crate::mpi::World::get() {
            let action = match self {
                Self::NoUpdate => 0,
                Self::Fail => 1,
                Self::Update(scaling) => 2 + scaling as u32,
            };
            if !world.agree_u32(action) {
                world.abort("inconsistent replicated solver strategy decision");
            }
        }
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
    pub(crate) phantom: std::marker::PhantomData<T>,
}

fn _print_banner(out: &mut dyn Write, is_verbose: bool) -> std::io::Result<()> {
    if !is_verbose {
        return std::io::Result::Ok(());
    }

    writeln!(
        out,
        "-------------------------------------------------------------"
    )?;
    writeln!(
        out,
        "           SDPX v{}                                         ",
        crate::VERSION
    )?;
    #[cfg(debug_assertions)]
    writeln!(
        out,
        "                  *** debug build ***                        ",
    )?;
    #[cfg(not(debug_assertions))]
    writeln!(out)?;
    writeln!(
        out,
        "                   (c) Paul Goulart                          "
    )?;
    writeln!(
        out,
        "                University of Oxford, 2022                   "
    )?;
    writeln!(
        out,
        "-------------------------------------------------------------"
    )?;
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

    pub fn settings(&self) -> &SE {
        &self.settings
    }

    pub fn update_settings(&mut self, settings: SE) -> Result<(), SettingsError> {
        settings.validate_as_update(&self.settings)?;
        self.settings = settings;
        Ok(())
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
        // various initializations
        // The generic curve is validated at all precisions, but its extra PSD
        // step searches regress the 512-bit Ising workload; retain the measured policy.
        let use_curve = T::precision_bits() <= 53 && self.cones.all_symmetric();
        let mut affine_direction = use_curve.then(|| self.variables.new_like());
        let mut curve_direction = use_curve.then(|| self.variables.new_like());
        let mut iter: u32 = 0;
        let mut σ = T::one();
        let mut α = T::zero();
        let mut μ;

        //timers is stored as an option so that
        //we can swap it out here and avoid
        //borrow conflicts with other fields.
        let mut timers = self.timers.take().unwrap();

        // solver release info, solver config
        // problem dimensions, cone types etc
        notimeit! {timers; {
            _print_banner(self.info.print_target(), self.settings.core().verbose).unwrap();
            self.info.print_configuration(&self.settings, &self.data, &self.cones).unwrap();
            self.info.print_status_header(&self.settings).unwrap();
        }}

        self.info.reset(&mut timers);

        timeit! {timers => "solve"; {

        // initialize variables to some reasonable starting point
        timeit!{timers => "default start"; {
            self.default_start();
        }}

        timeit!{timers => "IP iteration"; {

        // ----------
        // main loop
        // ----------

        let mut scaling = {
            if self.cones.supports_primal_dual() {ScalingStrategy::PrimalDual}
            else {ScalingStrategy::Dual}
        };

        loop {

            //update the residuals
            //--------------
            timeit!{timers => "residual update"; {
            self.residuals.update_with_pool(&self.variables, &self.data, self.cones.worker_pool());
            }}

            //calculate duality gap (scaled)
            //--------------
            timeit!{timers => "mu+info"; {
            μ = self.variables.calc_mu(&self.residuals, &self.cones);

            // record scalar values from most recent iteration.
            // This captures μ at iteration zero.
            self.info.save_scalars(μ, α, σ, iter);

            // convergence check and printing
            // --------------
            self.info.update_with_pool(
                &mut self.data,
                &self.variables,
                &self.residuals,
                &timers,
                self.cones.worker_pool());
            }}

            notimeit!{timers; {
                self.info.print_status(&self.settings).unwrap();
            }}

            // termination checks
            // --------------

            // user defined termination checks
            let callback_stop = self.callbacks.check_termination(&self.info);
            let callback_stop = crate::mpi::World::get()
                .map_or(callback_stop, |w| !w.all_true(!callback_stop));
            if callback_stop {
                self.info.set_status(SolverStatus::CallbackTerminated);
                break;
            }
            // internal termination checks
            let is_done = self.info.check_termination(&self.residuals, &self.settings, iter);
            if let Some(world) = crate::mpi::World::get() {
                if !world.agree_u32(self.info.get_status() as u32) {
                    world.abort("inconsistent replicated solver termination status");
                }
            }

            // check for termination due to slow progress and update strategy
            if is_done{
                    match self.strategy_checkpoint_insufficient_progress(scaling).synchronized(){
                        StrategyCheckpoint::NoUpdate | StrategyCheckpoint::Fail => {break}
                        StrategyCheckpoint::Update(s) => {scaling = s; continue}
                    }
            }  // allows continuation if new strategy provided


            // update the scalings
            // --------------
            let is_scaling_success;
            timeit!{timers => "scale cones"; {
                is_scaling_success = self.variables.scale_cones(&mut self.cones,μ,scaling);
            }}
            let is_scaling_success = crate::mpi::World::get()
                .map_or(is_scaling_success, |w| w.all_true(is_scaling_success));
            // check whether variables are interior points
            match self.strategy_checkpoint_is_scaling_success(is_scaling_success,scaling).synchronized(){
                StrategyCheckpoint::Fail => {break}
                StrategyCheckpoint::NoUpdate => {} // we only expect NoUpdate or Fail here
                StrategyCheckpoint::Update(_) => {unreachable!()}
            }

            //increment counter here because we only count
            //iterations that produce a KKT update
            iter += 1;
            if iter <= 2 {
                crate::receipt::memory_mark(if iter == 1 { "iteration 1 start" } else { "iteration 2 start" });
            }

            // Keep the affine RHS beside the constant RHS so the KKT update
            // can reuse the factorization and multi-RHS solve.
            timeit!{timers => "affine rhs"; {
            self.step_rhs
                .affine_step_rhs(&self.residuals, &self.variables, &self.cones);
            }}

            // Update the KKT system and the constant parts of its solution.
            // Keep track of the success of each step that calls KKT
            // --------------
            //PJG: This should be a Result in Rust, but needs changes down
            //into the KKT solvers to do that.
            let mut is_kkt_solve_success : bool;
            timeit!{timers => "kkt update"; {
                is_kkt_solve_success = self.kktsystem.update_affine(&self.data, &self.cones, &self.step_rhs, &self.variables, &self.settings);
            }} // end "kkt update" timer
            is_kkt_solve_success = crate::mpi::World::get()
                .map_or(is_kkt_solve_success, |w| w.all_true(is_kkt_solve_success));

            timeit!{timers => "kkt solve"; {
                is_kkt_solve_success = is_kkt_solve_success &&
                self.kktsystem.solve(
                    &mut self.step_lhs,
                    &self.step_rhs,
                    &self.data,
                    &self.variables,
                    &mut self.cones,
                    StepDirection::Affine,
                    &self.settings,
                );
            }}  //end "kkt solve affine" timer
            is_kkt_solve_success = crate::mpi::World::get()
                .map_or(is_kkt_solve_success, |w| w.all_true(is_kkt_solve_success));

            // combined step only on affine step success
            if is_kkt_solve_success {

                // Preserve the raw direction before prepared cone operations consume it.
                if let Some(affine) = &mut affine_direction { affine.copy_from(&self.step_lhs); }

                //calculate step length and centering parameter
                // --------------
                timeit!{timers => "affine step len"; {
                α = if iter > 1 {
                    self.variables.prepare_affine_step_length(&mut self.step_lhs, &mut self.cones, &self.settings)
                } else {
                    self.get_step_length(StepDirection::Affine, scaling)
                };
                }}
                σ = self.centering_parameter(α);

                // make a reduced Mehrotra correction in the first iteration
                // to accommodate badly centred starting points
                let m = if iter > 1 {T::one()} else {α};

                // calculate the combined step and length
                // --------------
                timeit!{timers => "combined rhs"; {
                if iter > 1 {
                    self.step_rhs.combined_step_rhs_prepared(&self.residuals, &self.variables,
                        &mut self.cones, &mut self.step_lhs, σ, μ);
                } else {
                self.step_rhs.combined_step_rhs(
                    &self.residuals,
                    &self.variables,
                    &mut self.cones,
                    &mut self.step_lhs,
                    σ,
                    μ,
                    m
                );
                }
                }}

                timeit!{timers => "kkt solve" ; {
                    is_kkt_solve_success =
                    self.kktsystem.solve(
                        &mut self.step_lhs,
                        &self.step_rhs,
                        &self.data,
                        &self.variables,
                        &mut self.cones,
                        StepDirection::Combined,
                        &self.settings,
                    );
                }} //end "kkt solve"
            }

            // check for numerical failure and update strategy
            is_kkt_solve_success = crate::mpi::World::get()
                .map_or(is_kkt_solve_success, |w| w.all_true(is_kkt_solve_success));
            match self.strategy_checkpoint_numerical_error(is_kkt_solve_success,scaling).synchronized() {
                StrategyCheckpoint::NoUpdate => {}
                StrategyCheckpoint::Update(s) => {α = T::zero(); scaling = s; continue}
                StrategyCheckpoint::Fail => {α = T::zero(); break}
            }


            // compute final step length and update the current iterate
            // --------------
            timeit!{timers => "final step len"; {
            α = self.get_step_length(StepDirection::Combined,scaling);

            // Inspired by Hypatia curve search, using the two existing NT directions.
            // Quadratic predictor-corrector curve: t*affine + t^2*(combined-affine).
            // No new KKT solves. Trial bounds use the existing cone-interior margin.
            if crate::mpi::agreed_branch(use_curve && α > T::zero() && α < (0.9).as_T()) {
                let original_alpha = α;
                let affine = affine_direction.as_ref().unwrap();
                let curve = curve_direction.as_mut().unwrap();
                for fraction in [0.5, 0.25] {
                    let t = original_alpha + (T::one()-original_alpha)*T::from_f64(fraction).unwrap();
                    if crate::mpi::agreed_branch(t*(T::one()-σ*t) <= T::from_f64(1.01).unwrap()*original_alpha*(T::one()-σ)) {
                        continue;
                    }
                    curve.interpolate(affine, &self.step_lhs, t);
                    let bound = self.variables.calc_step_length(curve, &mut self.cones,
                        &self.settings, StepDirection::Combined);
                    if crate::mpi::agreed_branch(t <= bound) {
                        self.step_lhs.copy_from(curve);
                        α = t;
                        break;
                    }
                }
            }

            // check for undersized step and update strategy
            match self.strategy_checkpoint_small_step(α, scaling).synchronized() {
                StrategyCheckpoint::NoUpdate => {}
                StrategyCheckpoint::Update(s) => {α = T::zero(); scaling = s; continue}
                StrategyCheckpoint::Fail => {α = T::zero(); break}
            }
            }} // end "final step len" timer

            // Copy previous iterate in case the next one is a dud
            self.info.save_prev_iterate(&self.variables,&mut self.prev_vars);

            timeit!{timers => "iterate update"; {
            self.variables
                .add_step_with_pool(&self.step_lhs, α, self.cones.worker_pool());
            }}

        } //end loop
        // ----------
        // ----------

        }} //end "IP iteration" timer

        }} // end "solve" timer

        // Check we if actually took a final step.  If not, we need
        // to recapture the scalars and print one last line
        if α.is_zero() {
            self.info.save_scalars(μ, α, σ, iter);
            notimeit! {timers; {self.info.print_status(&self.settings).unwrap();}}
        }

        timeit! {timers => "post-process"; {
            self.info.set_linear_solver_info(self.kktsystem.linear_solver_info());
            //check for "almost" convergence case and then extract solution
            self.info.post_process(&self.residuals, &self.settings);
            self.solution
                .post_process(&self.data, &mut self.variables, &self.info, &self.settings);
        }}

        //halt timers
        self.info.finalize(&mut timers);
        crate::receipt::memory_mark("solved");
        self.solution.finalize(&self.info);

        if crate::receipt::profile_requested()
            && crate::mpi::World::get().is_none_or(|w| w.rank() == 0)
        {
            timers.print();
        }

        self.info.print_footer(&self.settings).unwrap();

        //stow the timers back into Option in the solver struct
        self.timers.replace(timers);
    }
}

// Encapsulate the internal helpers trait in a private module
// so it doesn't get exported
mod internal {
    use super::*;

    pub(super) trait IPSolverInternals<T, D, V, R, K, C, I, SO, SE> {
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
        SO: Solution<T, D = D, V = V, I = I>,
        SE: Settings<T>,
    {
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
        }

        fn centering_parameter(&self, α: T) -> T {
            T::powi(T::one() - α, 3)
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
