use super::*;
use crate::solver::core::callbacks::SolverCallbacks;
use crate::solver::traits::Settings;
use crate::{
    io::ConfigurablePrintTarget,
    solver::core::{
        cones::{CompositeCone, SupportedConeT},
        kktsolvers::HasLinearSolverInfo,
        traits::ProblemData,
        SettingsError, Solver,
    },
};
use thiserror::Error;

use crate::algebra::*;
use crate::timers::*;

/// Solver for problems in standard conic program form
pub type DefaultSolver<T = f64> = Solver<
    T,
    DefaultProblemData<T>,
    DefaultVariables<T>,
    DefaultResiduals<T>,
    DefaultKKTSystem<T>,
    CompositeCone<T>,
    DefaultInfo<T>,
    DefaultSolution<T>,
    DefaultSettings<T>,
>;

/// Error types returned by the DefaultSolver

#[derive(Error, Debug)]
/// Error type returned by settings validation
pub enum SolverError {
    /// An error attributable to one of the fields
    #[error("Bad input data: {0}")]
    BadInputData(&'static str),

    /// Invalid dimensions or coefficients in a factor-defined sampled input.
    #[cfg(feature = "sdp")]
    #[error("Bad sampled input: {0}")]
    SampledInput(String),

    /// Error from settings validation with details
    #[error("Bad settings: {0}")]
    SettingsError(#[from] SettingsError),

    /// Error from I/O operations
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    /// Error from JSON parsing/serialization
    #[cfg(feature = "serde")]
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),
}

impl<T> DefaultSolver<T>
where
    T: FloatT,
{
    pub fn new(
        P: &CscMatrix<T>,
        q: &[T],
        A: &CscMatrix<T>,
        b: &[T],
        cones: &[SupportedConeT<T>],
        settings: DefaultSettings<T>,
    ) -> Result<Self, SolverError> {
        Self::new_with_setup(P, q, A, b, cones, settings, |_| {})
    }

    /// Construct a problem whose sampled PSD coefficients are defined by the
    /// supplied factors. `A_linear` is zero on every described PSD row range.
    #[cfg(feature = "sdp")]
    pub fn new_sampled(
        P: &CscMatrix<T>,
        q: &[T],
        A_linear: &CscMatrix<T>,
        b: &[T],
        cones: &[SupportedConeT<T>],
        blocks: Vec<SampledBlock<T>>,
        settings: DefaultSettings<T>,
    ) -> Result<Self, SolverError> {
        let mut sampled_timers = Timers::default();
        sampled_timers.start_as_current("setup");
        let operator =
            SampledOperator::new(A_linear.clone(), blocks).map_err(SolverError::SampledInput)?;
        let mut row = 0;
        let ranges: Vec<_> = cones
            .iter()
            .map(|cone| {
                let start = row;
                row += cone.nvars();
                (start, row, cone)
            })
            .collect();
        for block in operator.blocks() {
            if !ranges.iter().any(|(start, end, cone)| {
                *start == block.row_start
                    && *end == block.row_start + block.row_count()
                    && matches!(cone, SupportedConeT::PSDTriangleConeT(n) if *n == block.side())
            }) {
                return Err(SolverError::BadInputData(
                    "sampled rows must cover a complete PSD cone",
                ));
            }
        }
        let A = operator.materialize();
        let mut solver = Self::new_with_setup(P, q, &A, b, cones, settings, move |data| {
            data.install_sampled(operator)
        })?;
        // Include factor-input assembly in native setup time as well as the
        // ordinary setup performed by the shared constructor.
        sampled_timers.stop_current();
        solver.timers = Some(sampled_timers);
        Ok(solver)
    }

    fn new_with_setup(
        P: &CscMatrix<T>,
        q: &[T],
        A: &CscMatrix<T>,
        b: &[T],
        cones: &[SupportedConeT<T>],
        settings: DefaultSettings<T>,
        prepare_data: impl FnOnce(&mut DefaultProblemData<T>),
    ) -> Result<Self, SolverError> {
        //sanity check problem dimensions
        check_dimensions(P, q, A, b, cones)?;
        //sanity check settings
        settings.validate()?;

        let mut timers = Timers::default();
        let mut output;
        let mut info = DefaultInfo::<T>::new();

        timeit! {timers => "setup"; {

        // user facing results go here.
        let solution = DefaultSolution::<T>::new(A.n, A.m);

        // presolve / chordal decomposition if needed,
        // then take an internal copy of the problem data
        let mut data;
        timeit!{timers => "presolve"; {
            data = DefaultProblemData::<T>::new(P,q,A,b,cones,&settings);
        }}

        let mut cones = CompositeCone::<T>::new(&data.cones);
        cones
            .configure_threads(settings.max_threads as usize)
            .map_err(|_| SettingsError::LinearSolverProblem {
                solver: "cone workers",
                problem: "failed to create worker pool",
            })?;
        assert_eq!(cones.numel, data.m);
        let variables = DefaultVariables::<T>::new(data.n,data.m);
        let mut residuals = DefaultResiduals::<T>::new(data.n,data.m);

        // equilibrate problem data immediately on setup.
        // this prevents multiple equlibrations if solve!
        // is called more than once.
        timeit!{timers => "equilibration"; {
            data.equilibrate(&cones,&settings);
            prepare_data(&mut data);
        }}

        #[cfg(feature = "sdp")]
        if let Some(operator) = &data.sampled {
            residuals.sampled_workspace = Some(SampledWorkspace::new(operator));
        }

        residuals.prepare_sparse(&data, cones.thread_pool());

        let kktsystem;
        timeit!{timers => "kktinit"; {
            kktsystem = DefaultKKTSystem::<T>::new(&data,&cones,&settings);
        }}
        info.linsolver = kktsystem.linear_solver_info();

        // work variables for assembling step direction LHS/RHS
        let step_rhs  = DefaultVariables::<T>::new(data.n,data.m);
        let step_lhs  = DefaultVariables::<T>::new(data.n,data.m);
        let prev_vars = DefaultVariables::<T>::new(data.n,data.m);

        // configure empty user callbacks

        output = Self{
            data,variables,residuals,kktsystem,
            step_lhs,step_rhs,prev_vars,info,
            solution,cones,settings,
            timers: None,
            callbacks: SolverCallbacks::default(),
            phantom: std::marker::PhantomData };

        }} //end "setup" timer.

        //now that the timer is finished we can swap our
        //timer object into the solver structure
        output.timers.replace(timers);

        Ok(output)
    }
}

fn check_dimensions<T: FloatT>(
    P: &CscMatrix<T>,
    q: &[T],
    A: &CscMatrix<T>,
    b: &[T],
    cone_types: &[SupportedConeT<T>],
) -> Result<(), SolverError> {
    let m = b.len();
    let n = q.len();
    let p = cone_types.iter().fold(0, |acc, cone| acc + cone.nvars());

    if m != A.nrows() {
        return Err(SolverError::BadInputData("A and b incompatible dimensions"));
    }
    if p != m {
        return Err(SolverError::BadInputData(
            "Constraint dimensions inconsistent with size of cones",
        ));
    }
    if n != A.ncols() {
        return Err(SolverError::BadInputData("A and q incompatible dimensions"));
    }
    if n != P.ncols() {
        return Err(SolverError::BadInputData("P and q incompatible dimensions"));
    }
    if !P.is_square() {
        return Err(SolverError::BadInputData("P not square"));
    }

    Ok(())
}

impl<T> ConfigurablePrintTarget for DefaultSolver<T>
where
    T: FloatT,
{
    fn print_to_stdout(&mut self) {
        self.info.print_to_stdout();
    }
    fn print_to_file(&mut self, file: std::fs::File) {
        self.info.print_to_file(file)
    }
    fn print_to_stream(&mut self, stream: Box<dyn std::io::Write + Send + Sync>) {
        self.info.print_to_stream(stream)
    }
    fn print_to_sink(&mut self) {
        self.info.print_to_sink()
    }
    fn print_to_buffer(&mut self) {
        self.info.print_to_buffer();
    }
    fn get_print_buffer(&mut self) -> std::io::Result<String> {
        self.info.get_print_buffer()
    }
}
