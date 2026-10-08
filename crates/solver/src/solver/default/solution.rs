use super::*;
use crate::{
    algebra::*,
    solver::core::{traits::Solution, SolverStatus},
};

/// Standard-form solver type implementing the [`Solution`](crate::solver::core::traits::Solution) trait
#[derive(Debug)]
pub struct DefaultSolution<T> {
    /// primal solution
    pub x: Vec<T>,
    /// dual solution (in dual cone)
    pub z: Vec<T>,
    /// vector of slacks (in primal cone)
    pub s: Vec<T>,
    /// final solver status
    pub status: SolverStatus,
    /// primal objective value
    pub obj_val: T,
    /// dual objective value
    pub obj_val_dual: T,
    /// solve time in seconds
    pub solve_time: f64,
    /// number of iterations
    pub iterations: u32,
    /// primal residual
    pub r_prim: T,
    /// dual residual
    pub r_dual: T,
}

impl<T> DefaultSolution<T>
where
    T: FloatT,
{
    /// Create a new `DefaultSolution` object
    pub fn new(n: usize, m: usize) -> Self {
        let x = vec![T::zero(); n];
        let z = vec![T::zero(); m];
        let s = vec![T::zero(); m];

        Self {
            x,
            z,
            s,
            status: SolverStatus::Unsolved,
            obj_val: T::nan(),
            obj_val_dual: T::nan(),
            solve_time: 0f64,
            iterations: 0,
            r_prim: T::nan(),
            r_dual: T::nan(),
        }
    }
}

impl<T> DefaultSolution<T>
where
    T: FloatT,
{
    /// Map internal variables to the original problem: undo equilibration
    /// and τ (or κ for certificates), then chordal decomposition, then
    /// presolve. `variables` is left unscaled.
    fn reconstruct(
        &mut self,
        data: &DefaultProblemData<T>,
        variables: &mut DefaultVariables<T>,
        settings: &DefaultSettings<T>,
        is_infeasible: bool,
    ) {
        // unscale the variables to get a solution
        // to the internal problem as we solved it
        variables.unscale(data, is_infeasible);

        // unwind the chordal decomp and presolve, in the
        // reverse of the order in which they were applied
        let tmp = data
            .chordal_info
            .as_ref()
            .map(|chordal_info| chordal_info.decomp_reverse(variables, &data.cones, settings));
        let variables = tmp.as_ref().unwrap_or(variables);

        if let Some(ref presolver) = data.presolver {
            presolver.reverse_presolve(self, variables);
        } else {
            self.x.copy_from(&variables.x);
            self.z.copy_from(&variables.z);
            self.s.copy_from(&variables.s);
        }
    }
}

impl<T> Solution<T> for DefaultSolution<T>
where
    T: FloatT,
{
    type D = DefaultProblemData<T>;
    type V = DefaultVariables<T>;
    type I = DefaultInfo<T>;
    type SE = DefaultSettings<T>;

    fn post_process(
        &mut self,
        data: &DefaultProblemData<T>,
        variables: &mut DefaultVariables<T>,
        info: &DefaultInfo<T>,
        settings: &DefaultSettings<T>,
    ) {
        self.status = info.status;
        let is_infeasible = info.status.is_infeasible();

        if is_infeasible {
            self.obj_val = T::nan();
            self.obj_val_dual = T::nan();
        } else {
            self.obj_val = info.cost_primal;
            self.obj_val_dual = info.cost_dual;
        }

        self.iterations = info.iterations;
        self.r_prim = info.res_primal;
        self.r_dual = info.res_dual;

        self.reconstruct(data, variables, settings, is_infeasible);
    }

    fn original_ratio(
        &mut self,
        data: &DefaultProblemData<T>,
        variables: &DefaultVariables<T>,
        settings: &DefaultSettings<T>,
        reduced: bool,
    ) -> Option<T> {
        let tol = settings.tol_original?;
        if data.sampled.is_some() {
            return None;
        }
        let mut scratch = variables.copy_of();
        let status = self.status;
        self.status = SolverStatus::Solved;
        self.reconstruct(data, &mut scratch, settings, false);
        self.status = status;
        let original = data.original_data();
        let psd_dual = data.chordal_info.is_none() || settings.chordal_decomposition_complete_dual;
        let residuals = original.residuals(&self.x, &self.s, &self.z, psd_dual);
        Some(if reduced {
            residuals.ratio(
                T::max(tol, settings.reduced_tol_feas),
                T::max(tol, settings.reduced_tol_gap_abs),
                T::max(tol, settings.reduced_tol_gap_rel),
            )
        } else {
            residuals.ratio(tol, tol, tol)
        })
    }

    fn finalize(&mut self, info: &DefaultInfo<T>) {
        self.solve_time = info.solve_time;
    }
}
