use crate::{algebra::*, solver::kkt::HasLinearSolverInfo};

mod datamaps;
mod kkt_assembly;
mod solver;
use datamaps::*;
use kkt_assembly::*;
pub use solver::*;

pub trait DirectLDLSolverReqs {
    fn required_matrix_shape() -> MatrixTriangle
    where
        Self: Sized;
}
pub trait DirectLDLSolver<T: FloatT>: DirectLDLSolverReqs + HasLinearSolverInfo {
    fn update_values(&mut self, index: &[usize], values: &[T]);
    fn scale_values(&mut self, index: &[usize], scale: T);
    #[allow(dead_code)] //PJG: could be removed.
    fn offset_values(&mut self, index: &[usize], offset: T, signs: &[i8]);
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
    fn refactor(&mut self, kkt: &CscMatrix<T>) -> bool;
    /// Share the solver thread pool with factorisation/solve kernels that
    /// support it.  Solvers without a parallel path ignore the pool.
    fn set_pool(&mut self, _pool: Option<std::sync::Arc<rayon::ThreadPool>>) {}
}
