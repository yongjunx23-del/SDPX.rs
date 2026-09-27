use super::*;
use crate::solver::core::callbacks::SolverCallbacks;
use crate::solver::traits::Settings;
use crate::{
    io::ConfigurablePrintTarget,
    solver::{
        cones::{CompositeCone, SupportedConeT},
        core::{traits::ProblemData, SettingsError, Solver},
        kkt::HasLinearSolverInfo,
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

/// Shared validated, preprocessed and equilibrated input. No iteration vectors,
/// residual workspaces or KKT system have been allocated. Consumers resume the
/// stopped `setup` timer while constructing their own persistent state.
pub(crate) struct PreparedProblem<T: FloatT> {
    pub data: DefaultProblemData<T>,
    pub cones: CompositeCone<T>,
    pub settings: DefaultSettings<T>,
    pub solution: DefaultSolution<T>,
    pub timers: Timers,
    /// Exact raw-input identity for optional owner-cost histories.
    pub cost_input_fingerprint: Option<[u8; 32]>,
}

impl<T: FloatT> PreparedProblem<T> {
    pub(crate) fn new(
        P: &CscMatrix<T>,
        q: &[T],
        A: &CscMatrix<T>,
        b: &[T],
        cones: &[SupportedConeT<T>],
        settings: DefaultSettings<T>,
    ) -> Result<Self, SolverError> {
        Self::new_with_cost_identity_impl(P, q, A, b, cones, settings, false)
    }

    #[cfg(feature = "serde")]
    pub(crate) fn new_with_cost_identity(
        P: &CscMatrix<T>,
        q: &[T],
        A: &CscMatrix<T>,
        b: &[T],
        cones: &[SupportedConeT<T>],
        settings: DefaultSettings<T>,
    ) -> Result<Self, SolverError> {
        Self::new_with_cost_identity_impl(P, q, A, b, cones, settings, true)
    }

    fn new_with_cost_identity_impl(
        P: &CscMatrix<T>,
        q: &[T],
        A: &CscMatrix<T>,
        b: &[T],
        cones: &[SupportedConeT<T>],
        settings: DefaultSettings<T>,
        with_identity: bool,
    ) -> Result<Self, SolverError> {
        #[cfg(all(feature = "sdp", feature = "serde"))]
        let input_fingerprint = with_identity
            .then(|| crate::solver::distributed::input_fingerprint(P, q, A, b, cones, None));
        #[cfg(not(all(feature = "sdp", feature = "serde")))]
        let input_fingerprint = None;
        Self::new_with_setup(
            P,
            q,
            std::borrow::Cow::Borrowed(A),
            b,
            cones,
            settings,
            input_fingerprint,
            |_, _| {},
        )
    }

    #[cfg(feature = "sdp")]
    pub(crate) fn new_sampled(
        P: &CscMatrix<T>,
        q: &[T],
        A_linear: &CscMatrix<T>,
        b: &[T],
        cones: &[SupportedConeT<T>],
        blocks: Vec<SampledBlock<T>>,
        settings: DefaultSettings<T>,
    ) -> Result<Self, SolverError> {
        Self::new_sampled_with_cost_identity_impl(P, q, A_linear, b, cones, blocks, settings, false)
    }

    #[cfg(all(feature = "serde", feature = "sdp"))]
    pub(crate) fn new_sampled_with_cost_identity(
        P: &CscMatrix<T>,
        q: &[T],
        A_linear: &CscMatrix<T>,
        b: &[T],
        cones: &[SupportedConeT<T>],
        blocks: Vec<SampledBlock<T>>,
        settings: DefaultSettings<T>,
        with_identity: bool,
    ) -> Result<Self, SolverError> {
        Self::new_sampled_with_cost_identity_impl(
            P,
            q,
            A_linear,
            b,
            cones,
            blocks,
            settings,
            with_identity,
        )
    }

    #[cfg(feature = "sdp")]
    fn new_sampled_with_cost_identity_impl(
        P: &CscMatrix<T>,
        q: &[T],
        A_linear: &CscMatrix<T>,
        b: &[T],
        cones: &[SupportedConeT<T>],
        blocks: Vec<SampledBlock<T>>,
        settings: DefaultSettings<T>,
        with_identity: bool,
    ) -> Result<Self, SolverError> {
        #[cfg(feature = "serde")]
        let input_fingerprint = with_identity.then(|| {
            crate::solver::distributed::input_fingerprint(P, q, A_linear, b, cones, Some(&blocks))
        });
        #[cfg(not(feature = "serde"))]
        let input_fingerprint = None;
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
        // The solver's pool does not exist yet; expand the blocks on a
        // setup-only pool of the configured width (0 = available CPUs).
        let setup_pool = (settings.max_threads != 1)
            .then(|| {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(settings.max_threads as usize)
                    .build()
                    .ok()
            })
            .flatten();
        let A = operator
            .materialize_checked_pooled(setup_pool.as_ref())
            .map_err(SolverError::SampledInput)?;
        drop(setup_pool);
        crate::receipt::memory_mark("sampled materialized");
        let mut prepared = Self::new_with_setup(
            P,
            q,
            std::borrow::Cow::Owned(A),
            b,
            cones,
            settings,
            input_fingerprint,
            move |data, pool| data.install_sampled(operator, pool),
        )?;
        // Include factor-input assembly in native setup time as well as the
        // ordinary preparation performed by the shared constructor.
        sampled_timers.stop_current();
        prepared.timers = sampled_timers;
        Ok(prepared)
    }

    fn new_with_setup(
        P: &CscMatrix<T>,
        q: &[T],
        A: std::borrow::Cow<'_, CscMatrix<T>>,
        b: &[T],
        cones: &[SupportedConeT<T>],
        settings: DefaultSettings<T>,
        input_fingerprint: Option<[u8; 32]>,
        prepare_data: impl FnOnce(&mut DefaultProblemData<T>, Option<&rayon::ThreadPool>),
    ) -> Result<Self, SolverError> {
        check_dimensions(P, q, &A, b, cones)?;
        settings.validate()?;
        let mut timers = Timers::default();
        timers.start_as_current("setup");
        let solution = DefaultSolution::<T>::new(A.n, A.m);
        let mut data;
        timeit! {timers => "presolve"; {
            data = DefaultProblemData::<T>::new_cow(P,q,A,b,cones,&settings);
        }}
        crate::receipt::memory_mark("problem data");
        let mut cones = CompositeCone::<T>::new(&data.cones);
        cones
            .configure_threads(settings.max_threads as usize)
            .map_err(|_| SettingsError::LinearSolverProblem {
                solver: "cone workers",
                problem: "failed to create worker pool",
            })?;
        if cones.numel != data.m {
            return Err(SolverError::BadInputData(
                "cone dimensions do not match the reduced problem",
            ));
        }
        timeit! {timers => "equilibration"; {
            data.equilibrate(&cones,&settings);
            prepare_data(&mut data, cones.thread_pool().as_deref());
        }}
        crate::receipt::memory_mark("equilibrated");
        timers.stop_current();
        Ok(Self {
            data,
            cones,
            settings,
            solution,
            timers,
            cost_input_fingerprint: input_fingerprint,
        })
    }
}

impl<T: FloatT> DefaultSolver<T> {
    pub fn new(
        P: &CscMatrix<T>,
        q: &[T],
        A: &CscMatrix<T>,
        b: &[T],
        cones: &[SupportedConeT<T>],
        settings: DefaultSettings<T>,
    ) -> Result<Self, SolverError> {
        Self::from_prepared(PreparedProblem::new(P, q, A, b, cones, settings)?)
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
        Self::from_prepared(PreparedProblem::new_sampled(
            P, q, A_linear, b, cones, blocks, settings,
        )?)
    }

    pub(crate) fn from_prepared(prepared: PreparedProblem<T>) -> Result<Self, SolverError> {
        let PreparedProblem {
            data,
            cones,
            settings,
            solution,
            mut timers,
            cost_input_fingerprint: _,
        } = prepared;
        timers.start_as_current("setup");
        let variables = DefaultVariables::<T>::new(data.n, data.m);
        let mut residuals = DefaultResiduals::<T>::new(data.n, data.m);
        #[cfg(feature = "sdp")]
        if let Some(operator) = &data.sampled {
            operator.prepare_constants(cones.thread_pool().as_deref());
            // Applied twice per iteration for the IPM residuals: keep the
            // basis residues across calls as the KKT workspace does.
            let mut work = SampledWorkspace::new(operator);
            work.enable_basis_caches();
            residuals.sampled_workspace = Some(work);
        }
        residuals.prepare_sparse(&data, cones.thread_pool());
        crate::receipt::memory_mark("residual workspace");
        let kktsystem;
        timeit! {timers => "kktinit"; {
            kktsystem = DefaultKKTSystem::<T>::new(&data,&cones,&settings);
        }}
        crate::receipt::memory_mark("kkt system");
        let mut info = DefaultInfo::<T>::new();
        info.linsolver = kktsystem.linear_solver_info();
        let step_rhs = DefaultVariables::<T>::new(data.n, data.m);
        let step_lhs = DefaultVariables::<T>::new(data.n, data.m);
        let prev_vars = DefaultVariables::<T>::new(data.n, data.m);
        let mut output = Self {
            data,
            variables,
            residuals,
            kktsystem,
            step_lhs,
            step_rhs,
            prev_vars,
            info,
            solution,
            cones,
            settings,
            timers: None,
            callbacks: SolverCallbacks::default(),
            phantom: std::marker::PhantomData,
        };
        timers.stop_current();
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
