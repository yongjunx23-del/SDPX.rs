#![allow(non_snake_case)]
use super::{cones::CompositeCone, CoreSettings};
use crate::algebra::*;

mod condensed;
pub mod direct;
pub mod ldl;
pub(crate) mod refinement;
pub(crate) use condensed::CondensedKKTSolver;
pub(crate) use condensed::{columns_touching, parallel_assembly_budget_bytes, psd_form_costs};

/// Per-solve accounting. Diagnoses how many factorizations and right-hand-side
/// applications a solver actually performs, so an iteration's cost can be
/// attributed instead of assumed.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct SolveCounters {
    /// Numeric factorization attempts, including failed attempts.
    pub factor_attempts: u64,
    /// Backend linear solves, including iterative refinement corrections.
    pub linear_solves: u64,
    /// Iterative refinement corrections applied by the direct solver.
    pub refinements: u64,
    /// Successful numeric factorizations of the current KKT structure.
    pub factorizations: u64,
    /// Caller-supplied RHS columns, excluding refinement corrections.
    pub rhs_applied: u64,
    /// Batched submissions; one wave may carry several columns.
    pub batches: u64,
    /// Corrections applied by the outer (full-KKT) refinement of a condensed
    /// or distributed solve; each one is a complete reduced solve.
    pub outer_refinements: u64,
}

pub trait KKTSolver<T: FloatT>: HasLinearSolverInfo {
    /// Begin an independent solve, resetting counters and retry state.
    fn reset_solve(&mut self) {}

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

    fn counters(&self) -> SolveCounters {
        SolveCounters::default()
    }

    /// Solve several right-hand sides on the current factorization in one wave.
    ///
    /// Each column is `[x (n); z (m)]`, and `rhs`/`out` hold `ncols` such
    /// columns contiguously. Returns one flag per column: a failure in one column
    /// must never be reported as another column's success.
    fn solve_many(
        &mut self,
        n: usize,
        rhs: &[T],
        out: &mut [T],
        ncols: usize,
        settings: &CoreSettings<T>,
    ) -> Vec<bool>;

    /// Exact working-precision Hs*z computed by the original-operator residual
    /// for an accepted single solution (column 0) or the last batched wave.
    /// None when refinement did not evaluate it, or that column failed.
    fn scaled_solution(&self, _column: usize) -> Option<&[T]> {
        None
    }

    /// W·z of an accepted solution column on the blocks (in cone order)
    /// whose recovery formed it, from the recovery's pre-images. None when
    /// the solver did not form them or a refinement correction changed z.
    fn scaled_preimage(&self, _column: usize) -> Option<(&[T], &[bool])> {
        None
    }

    /// Move `scaled_solution(column)` into `out` (empty when None). The
    /// solver may give up the column, so call it after the column's last use.
    fn take_scaled_solution(&mut self, column: usize, out: &mut Vec<T>) {
        out.clear();
        if let Some(values) = self.scaled_solution(column) {
            out.extend_from_slice(values);
        }
    }

    /// Move `scaled_preimage(column)` into `out` (empty when None), as
    /// [`KKTSolver::take_scaled_solution`].
    fn take_scaled_preimage(&mut self, column: usize, out: &mut (Vec<T>, Vec<bool>)) {
        out.0.clear();
        out.1.clear();
        if let Some((values, blocks)) = self.scaled_preimage(column) {
            out.0.extend_from_slice(values);
            out.1.extend_from_slice(blocks);
        }
    }

    /// Leave column `column` of the next `solve_many` unrefined (the affine
    /// predictor). Solvers that ignore it refine every column. `prepared`:
    /// its step bounds take PSD rows of Δs from W·Δz, not from Hs·Δz.
    fn skip_refinement(&mut self, _column: usize, _prepared: bool) {}

    /// Binary64: continue a stationary refinement that stalls above its
    /// tolerance with GMRES-IR for the rest of the solve. The driver asks
    /// for it once a feasibility residual grows; until then the cheaper
    /// stationary result stands, as before.
    fn refine_further(&mut self) {}

    /// Condensed solvers: the rows kept in the reduced system. The `z` of
    /// every other row is recovered as `H⁻¹(A x − r_z)`.
    fn retained_rows(&self) -> Option<&[usize]> {
        None
    }

    fn update_P(&mut self, P: &CscMatrix<T>);
    fn update_A(&mut self, A: &CscMatrix<T>);

    /// Raise the static regularization level used by the next factorization.
    /// Returns false once the escalation budget is exhausted; solvers that
    /// cannot escalate report false immediately. Escalation only perturbs the
    /// factorized matrix — iterative refinement still runs against the true
    /// one, so convergence semantics are unchanged.
    fn escalate_regularization(&mut self) -> bool {
        false
    }
    fn set_sampled_operator(
        &mut self,
        _operator: std::sync::Arc<crate::solver::SampledOperator<T>>,
    ) {
    }
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
