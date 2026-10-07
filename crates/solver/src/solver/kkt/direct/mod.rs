use crate::{algebra::*, solver::kkt::HasLinearSolverInfo};

mod datamaps;
mod kkt_assembly;
mod solver;
use datamaps::*;
use kkt_assembly::*;
pub use solver::*;

pub trait DirectLDLSolver<T: FloatT>: HasLinearSolverInfo {
    fn update_values(&mut self, index: &[usize], values: &[T]);
    fn scale_values(&mut self, index: &[usize], scale: T);
    fn solve(&mut self, kkt: &CscMatrix<T>, x: &mut [T], b: &mut [T]);
    /// Column-major RHS panel, sharing the current factors. Backends may fuse
    /// traversal; the portable implementation preserves each scalar solve.
    fn solve_many(&mut self, kkt: &CscMatrix<T>, x: &mut [T], b: &mut [T], ncols: usize) {
        assert_eq!(x.len(), kkt.n * ncols);
        assert_eq!(b.len(), x.len());
        if ncols == 0 || kkt.n == 0 {
            return;
        }
        for (x, b) in x.chunks_mut(kkt.n).zip(b.chunks_mut(kkt.n)) {
            self.solve(kkt, x, b);
        }
    }
    /// Optional structure-aware residual against the complete unshifted KKT.
    fn residual(&self, _kkt: &CscMatrix<T>, _out: &mut [T], _rhs: &[T], _point: &[T]) -> Option<T> {
        None
    }
    fn refactor(&mut self, kkt: &CscMatrix<T>) -> bool;
    /// KKT columns whose static diagonal shift this factorization does not
    /// need: its fixed elimination order gives them a positive pivot.
    fn unshifted_columns(&self) -> &[usize] {
        &[]
    }
    /// Whether the next refactor sees `unshifted_columns` without the shift.
    fn set_shift_exemption(&mut self, _active: bool) {}
    /// Share the solver thread pool with factorisation/solve kernels that
    /// support it.  Solvers without a parallel path ignore the pool.
    fn set_pool(&mut self, _pool: Option<std::sync::Arc<rayon::ThreadPool>>) {}

    /// Share the factorization over the ranks of `world` where the backend
    /// supports it; every rank must make the same call.
    fn set_world(&mut self, _world: crate::MpiContext) {}
}
