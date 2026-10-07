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
        Self::new_cow(
            std::borrow::Cow::Borrowed(P),
            std::borrow::Cow::Borrowed(q),
            std::borrow::Cow::Borrowed(A),
            std::borrow::Cow::Borrowed(b),
            cones,
            settings,
            None,
        )
    }

    pub(super) fn new_cow(
        P: std::borrow::Cow<'_, CscMatrix<T>>,
        q: std::borrow::Cow<'_, [T]>,
        A: std::borrow::Cow<'_, CscMatrix<T>>,
        b: std::borrow::Cow<'_, [T]>,
        cones: &[SupportedConeT<T>],
        settings: DefaultSettings<T>,
        input_fingerprint: Option<[u8; 32]>,
    ) -> Result<Self, SolverError> {
        check_dimensions(&P, &q, &A, &b, cones)?;
        P.check_format().map_err(|_| {
            SolverError::BadInputData("P must be canonical CSC (sorted, unique, in-range rows)")
        })?;
        A.check_format().map_err(|_| {
            SolverError::BadInputData("A must be canonical CSC (sorted, unique, in-range rows)")
        })?;
        settings.validate()?;
        Self::new_with_setup(P, q, A, b, cones, settings, input_fingerprint, |_, _| {})
    }

    pub(crate) fn new_sampled(
        P: &CscMatrix<T>,
        q: &[T],
        A_linear: &CscMatrix<T>,
        b: &[T],
        cones: &[SupportedConeT<T>],
        blocks: Vec<SampledBlock<T>>,
        settings: DefaultSettings<T>,
    ) -> Result<Self, SolverError> {
        Self::new_sampled_cow(
            std::borrow::Cow::Borrowed(P),
            std::borrow::Cow::Borrowed(q),
            std::borrow::Cow::Borrowed(A_linear),
            std::borrow::Cow::Borrowed(b),
            cones,
            blocks,
            settings,
            None,
        )
    }

    pub(super) fn new_sampled_cow(
        P: std::borrow::Cow<'_, CscMatrix<T>>,
        q: std::borrow::Cow<'_, [T]>,
        A_linear: std::borrow::Cow<'_, CscMatrix<T>>,
        b: std::borrow::Cow<'_, [T]>,
        cones: &[SupportedConeT<T>],
        blocks: Vec<SampledBlock<T>>,
        settings: DefaultSettings<T>,
        input_fingerprint: Option<[u8; 32]>,
    ) -> Result<Self, SolverError> {
        let mut sampled_timers = Timers::default();
        sampled_timers.start_setup();
        check_dimensions(&P, &q, &A_linear, &b, cones)?;
        P.check_format().map_err(|_| {
            SolverError::BadInputData("P must be canonical CSC (sorted, unique, in-range rows)")
        })?;
        settings.validate()?;
        let operator = SampledOperator::new(A_linear.into_owned(), blocks)
            .map_err(SolverError::SampledInput)?;
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
        // setup-only pool with the same budget as the runtime phases.
        let workers = crate::solver::core::worker_budget(settings.max_threads as usize);
        let setup_pool = (workers > 1)
            .then(|| {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(workers)
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
        sampled_timers.stop_setup();
        prepared.timers = sampled_timers;
        Ok(prepared)
    }

    fn new_with_setup(
        P: std::borrow::Cow<'_, CscMatrix<T>>,
        q: std::borrow::Cow<'_, [T]>,
        A: std::borrow::Cow<'_, CscMatrix<T>>,
        b: std::borrow::Cow<'_, [T]>,
        cones: &[SupportedConeT<T>],
        settings: DefaultSettings<T>,
        input_fingerprint: Option<[u8; 32]>,
        prepare_data: impl FnOnce(&mut DefaultProblemData<T>, Option<&rayon::ThreadPool>),
    ) -> Result<Self, SolverError> {
        let mut timers = Timers::default();
        timers.start_setup();
        let solution = DefaultSolution::<T>::new(A.n, A.m);
        let mut data;
        timeit! {"presolve"; {
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
        timeit! {"equilibration"; {
            data.equilibrate(&cones,&settings);
            prepare_data(&mut data, cones.thread_pool().as_deref());
        }}
        crate::receipt::memory_mark("equilibrated");
        timers.stop_setup();
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
        let mut data = data;
        timers.start_setup();
        let kktsystem;
        timeit! {"kktinit"; {
            kktsystem = DefaultKKTSystem::<T>::new(&data,&cones,&settings);
        }}
        data.compact_sampled_matrix();
        crate::receipt::memory_mark("kkt system");
        let variables = DefaultVariables::<T>::new(data.n, data.m);
        let mut residuals = DefaultResiduals::<T>::new(data.n, data.m);
        residuals.prepare_sparse(&data, cones.thread_pool());
        crate::receipt::memory_mark("residual workspace");
        let mut info = DefaultInfo::<T>::new();
        info.linsolver = kktsystem.linear_solver_info();
        info.objective_offset = data
            .presolver
            .as_ref()
            .map_or(T::zero(), |p| p.objective_offset);
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
        timers.stop_setup();
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
    let mut p = 0usize;
    for cone in cone_types {
        let invalid = || SolverError::BadInputData("invalid cone dimensions or parameters");
        let size = match cone {
            SupportedConeT::ZeroConeT(n)
            | SupportedConeT::NonnegativeConeT(n)
            | SupportedConeT::SecondOrderConeT(n) => *n,
            SupportedConeT::ExponentialConeT() => 3,
            SupportedConeT::PowerConeT(alpha) => {
                if !(*alpha > T::zero() && *alpha < T::one()) {
                    return Err(invalid());
                }
                3
            }
            SupportedConeT::GenPowerConeT(alpha, dim2) => {
                // Match the constructor's strict sum tolerance before setup.
                if !alpha.iter().all(|a| *a > T::zero())
                    || !((T::one() - alpha.as_slice().sum()).abs()
                        < T::epsilon() * alpha.len().as_T() * (0.5).as_T())
                {
                    return Err(invalid());
                }
                alpha.len().checked_add(*dim2).ok_or_else(invalid)?
            }
            SupportedConeT::PSDTriangleConeT(n) => n
                .checked_add(1)
                .and_then(|next| n.checked_mul(next))
                .map(|twice| twice / 2)
                .ok_or_else(invalid)?,
        };
        p = p.checked_add(size).ok_or_else(invalid)?;
    }

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
