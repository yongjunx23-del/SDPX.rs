//! Public facade for the owner-partitioned HSD backend.
//!
//! The numerical loop and all storage remain private.  This wrapper exposes
//! the same small solve/configuration surface as [`DefaultSolver`] while
//! keeping the owner-local implementation behind the default module.

use super::hsd::OwnedSolver;
use super::{
    CostHistory, CostHistoryOptions, DefaultInfo, DefaultSettings, DefaultSolution,
    PreparedProblem, SolverError,
};
use crate::algebra::FloatT;
use crate::io::ConfigurablePrintTarget;
use crate::solver::distributed::collective::WorldCollective;
use crate::solver::kkt::SolveCounters;
use std::io::Write;
use std::sync::Arc;

/// Owner-partitioned HSD solver.
///
/// In an active multi-rank MPI world, the same facade selects one global owner
/// per rank and exchanges only the reductions required by the shared loop.
pub struct PartitionedSolver<T: FloatT> {
    pub(crate) inner: OwnedSolver<T>,
}

impl<T: FloatT> PartitionedSolver<T> {
    /// Choose bounded structural tasks independently of the worker count.
    pub(crate) fn from_prepared_auto(prepared: PreparedProblem<T>) -> Result<Self, SolverError> {
        if let Some(world) = crate::mpi::World::get() {
            let collective = Arc::new(WorldCollective(world));
            return OwnedSolver::from_prepared_rank_local(prepared, collective)
                .map(|inner| Self { inner })
                .map_err(SolverError::SampledInput);
        }
        OwnedSolver::from_prepared_auto(prepared)
            .map(|inner| Self { inner })
            .map_err(SolverError::SampledInput)
    }

    #[cfg(feature = "serde")]
    pub(crate) fn from_prepared_auto_with_cost_history(
        prepared: PreparedProblem<T>,
        options: CostHistoryOptions,
    ) -> Result<Self, SolverError> {
        if let Some(world) = crate::mpi::World::get() {
            let collective = Arc::new(WorldCollective(world));
            return OwnedSolver::from_prepared_rank_local_with_cost_history(
                prepared, collective, options,
            )
            .map(|inner| Self { inner })
            .map_err(SolverError::SampledInput);
        }
        OwnedSolver::from_prepared_auto_with_cost_history(prepared, options)
            .map(|inner| Self { inner })
            .map_err(SolverError::SampledInput)
    }

    /// Build a partitioned solver from shared, prepared problem data.
    ///
    /// Preparation performs validation, preprocessing and equilibration once;
    /// owner conversion then allocates only the local HSD state.
    pub(crate) fn from_prepared(
        prepared: PreparedProblem<T>,
        partitions: usize,
    ) -> Result<Self, SolverError> {
        if partitions == 0 {
            return Err(SolverError::BadInputData(
                "partition count must be positive",
            ));
        }
        if let Some(world) = crate::mpi::World::get() {
            if partitions != world.size() {
                return Err(SolverError::BadInputData(
                    "MPI partition count must equal the world size",
                ));
            }
            let collective = Arc::new(WorldCollective(world));
            return OwnedSolver::from_prepared_rank_local(prepared, collective)
                .map(|inner| Self { inner })
                .map_err(SolverError::SampledInput);
        }
        OwnedSolver::from_prepared(prepared, partitions)
            .map(|inner| Self { inner })
            .map_err(SolverError::SampledInput)
    }

    #[cfg(feature = "serde")]
    pub(crate) fn from_prepared_with_cost_history(
        prepared: PreparedProblem<T>,
        partitions: usize,
        options: CostHistoryOptions,
    ) -> Result<Self, SolverError> {
        if partitions == 0 {
            return Err(SolverError::BadInputData(
                "partition count must be positive",
            ));
        }
        if let Some(world) = crate::mpi::World::get() {
            if partitions != world.size() {
                return Err(SolverError::BadInputData(
                    "MPI partition count must equal the world size",
                ));
            }
            let collective = Arc::new(WorldCollective(world));
            return OwnedSolver::from_prepared_rank_local_with_cost_history(
                prepared, collective, options,
            )
            .map(|inner| Self { inner })
            .map_err(SolverError::SampledInput);
        }
        OwnedSolver::from_prepared_with_cost_history(prepared, partitions, options)
            .map(|inner| Self { inner })
            .map_err(SolverError::SampledInput)
    }

    /// Run the shared predictor/corrector loop.
    pub fn solve(&mut self) {
        use crate::solver::IPSolver;
        self.inner.solve();
    }

    /// Return the recovered original-coordinate solution.
    pub fn solution(&self) -> &DefaultSolution<T> {
        &self.inner.solution.0
    }

    /// Return solver progress information and termination status.
    pub fn info(&self) -> &DefaultInfo<T> {
        &self.inner.info.0
    }

    /// Return the effective solver settings.
    pub fn settings(&self) -> &DefaultSettings<T> {
        self.inner.settings()
    }

    /// Number of workers in the owner/cone pool, or one for serial execution.
    pub fn cone_threads(&self) -> usize {
        self.inner
            .cones
            .pool
            .as_ref()
            .map_or(1, |pool| pool.current_num_threads())
    }

    /// Number of owner partitions.
    pub fn partitions(&self) -> usize {
        self.inner.data.global_layout.owners.len()
    }

    /// Export opt-in owner-local training timings as a validated history.
    /// In MPI, every rank must call this method in the same order. Timings are
    /// gathered to rank zero; other ranks return `None`.
    #[cfg(feature = "serde")]
    pub fn cost_history(&self) -> Result<Option<CostHistory>, String> {
        self.inner.cost_history()
    }

    /// Per-solve factorization and RHS accounting for execution receipts.
    pub(crate) fn counters(&self) -> SolveCounters {
        self.inner.kktsystem.counters()
    }

    /// Redirect progress output to a stream.
    pub fn print_to_stream(&mut self, stream: Box<dyn Write + Send + Sync>) {
        self.inner.info.0.print_to_stream(stream);
    }

    /// Suppress progress output.
    pub fn print_to_sink(&mut self) {
        self.inner.info.0.print_to_sink();
    }
}

impl<T: FloatT> ConfigurablePrintTarget for PartitionedSolver<T> {
    fn print_to_stdout(&mut self) {
        self.inner.info.0.print_to_stdout();
    }

    fn print_to_file(&mut self, file: std::fs::File) {
        self.inner.info.0.print_to_file(file);
    }

    fn print_to_stream(&mut self, stream: Box<dyn Write + Send + Sync>) {
        self.inner.info.0.print_to_stream(stream);
    }

    fn print_to_sink(&mut self) {
        self.inner.info.0.print_to_sink();
    }

    fn print_to_buffer(&mut self) {
        self.inner.info.0.print_to_buffer();
    }

    fn get_print_buffer(&mut self) -> std::io::Result<String> {
        self.inner.info.0.get_print_buffer()
    }
}
