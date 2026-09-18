#![allow(non_snake_case)]
use super::{cones::CompositeCone, CoreSettings};
use crate::algebra::*;

#[cfg(feature = "sdp")]
mod condensed;
pub mod direct;
#[cfg(feature = "sdp")]
pub(crate) use condensed::CondensedKKTSolver;

/// Per-solve accounting. Diagnoses how many factorizations and right-hand-side
/// applications a solver actually performs, so an iteration's cost can be
/// attributed instead of assumed.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct SolveCounters {
    /// Successful numeric factorizations of the current KKT structure.
    pub factorizations: u64,
    /// Right-hand-side applications across all columns and waves.
    pub rhs_applied: u64,
    /// Batched submissions; one wave may carry several columns.
    pub batches: u64,
}

pub trait KKTSolver<T: FloatT>: HasLinearSolverInfo {
    fn update(&mut self, cones: &CompositeCone<T>, settings: &CoreSettings<T>) -> bool;
    fn setrhs(&mut self, x: &[T], z: &[T]);
    fn solve(
        &mut self,
        x: Option<&mut [T]>,
        z: Option<&mut [T]>,
        settings: &CoreSettings<T>,
    ) -> bool;

    /// Accounting for the current solver instance. Backends that do not track it
    /// report zeros rather than a guess.
    ///
    /// Inert until the main loop batches right-hand sides (plan item A22): only
    /// tests call it today, which the workspace build reports as dead code.
    #[allow(dead_code)]
    fn counters(&self) -> SolveCounters {
        SolveCounters::default()
    }

    /// Solve several right-hand sides on the current factorization in one wave.
    ///
    /// Each column is `[x (n); z (m)]`, and `rhs`/`out` hold `ncols` such
    /// columns contiguously. Returns one flag per column: a failure in one column
    /// must never be reported as another column's success.
    ///
    /// The default implementation loops the single-RHS path column by column, so
    /// every existing backend stays correct. Only a backend that genuinely
    /// exploits several columns at once should override it.
    ///
    /// Inert until the main loop batches right-hand sides (plan item A22).
    #[allow(dead_code)]
    fn solve_many(
        &mut self,
        n: usize,
        rhs: &[T],
        out: &mut [T],
        ncols: usize,
        settings: &CoreSettings<T>,
    ) -> Vec<bool> {
        solve_many_by_columns(self, n, rhs, out, ncols, settings)
    }

    fn update_P(&mut self, P: &CscMatrix<T>);
    fn update_A(&mut self, A: &CscMatrix<T>);
    #[cfg(feature = "sdp")]
    fn set_sampled_operator(
        &mut self,
        _operator: std::sync::Arc<crate::solver::SampledOperator<T>>,
    ) {
    }
}

/// The portable column loop behind the default `solve_many`.
#[allow(dead_code)]
pub fn solve_many_by_columns<T, K>(
    solver: &mut K,
    n: usize,
    rhs: &[T],
    out: &mut [T],
    ncols: usize,
    settings: &CoreSettings<T>,
) -> Vec<bool>
where
    T: FloatT,
    K: KKTSolver<T> + ?Sized,
{
    if ncols == 0 {
        return Vec::new();
    }
    assert_eq!(rhs.len(), out.len(), "rhs and out must have equal length");
    assert_eq!(
        rhs.len() % ncols,
        0,
        "rhs/out must hold whole columns: len={} ncols={ncols}",
        rhs.len()
    );
    let width = rhs.len() / ncols;
    assert!(width >= n, "column width must cover the x block");
    let mut results = Vec::with_capacity(ncols);
    for c in 0..ncols {
        let base = c * width;
        solver.setrhs(&rhs[base..base + n], &rhs[base + n..base + width]);
        let (xl, zl) = out[base..base + width].split_at_mut(n);
        results.push(solver.solve(Some(xl), Some(zl), settings));
    }
    results
}

pub trait HasLinearSolverInfo {
    fn linear_solver_info(&self) -> LinearSolverInfo;
}
#[repr(C)]
#[derive(Debug, Default, Clone)]
/// Linear subsolver information.
///
pub struct LinearSolverInfo {
    /// Name of the linear solver that was used
    pub name: String,
    /// Number of threads used by solver
    pub threads: usize,
    /// Whether the solver used a direct factorisation method
    pub direct: bool,
    /// Number of nonzeros in the linear system
    pub nnzA: usize, // nnz in A for A = LDL^T
    /// Number of nonzeros in the factored system
    pub nnzL: usize, // nnz in L for A = LDL^T
}

