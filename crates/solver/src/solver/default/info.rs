use super::*;
use crate::algebra::*;
use crate::io::PrintTarget;
use crate::solver::core::{traits::Info, SolverStatus};
use crate::solver::kkt::LinearSolverInfo;
use crate::solver::traits::Variables;
use crate::timers::*;

/// Standard-form solver type implementing the [`Info`](crate::solver::core::traits::Info) and [`InfoPrint`](crate::solver::core::traits::InfoPrint) traits
#[repr(C)]
#[derive(Default, Debug, Clone)]
pub struct DefaultInfo<T> {
    /// interior point path parameter μ
    pub mu: T,
    /// interior point path parameter reduction ratio σ
    pub sigma: T,
    /// step length for the current iteration
    pub step_length: T,
    /// number of iterations
    pub iterations: u32,
    /// primal objective value
    pub cost_primal: T,
    /// dual objective value
    pub cost_dual: T,
    /// primal residual
    pub res_primal: T,
    /// dual residual
    pub res_dual: T,
    /// Optional operator-aware componentwise dual residual. It is `None`
    /// unless `tol_feas_componentwise` is enabled.
    pub res_dual_componentwise: Option<T>,
    /// Original-coordinate dual Euclidean norm divided by `1 + ‖q‖∞`;
    /// gated by `tol_dual_qnorm`.
    pub res_dual_qnorm: T,
    /// primal infeasibility residual
    pub res_primal_inf: T,
    /// dual infeasibility residual
    pub res_dual_inf: T,
    /// absolute duality gap
    pub gap_abs: T,
    /// relative duality gap
    pub gap_rel: T,
    /// κ/τ ratio
    pub ktratio: T,
    /// Unnormalized primal and dual residual norms (scaled problem).
    pub(crate) residual_norms: (T, T),

    // previous iterate
    /// primal object value from previous iteration
    pub(crate) prev_cost_primal: T,
    /// dual objective value from previous iteration
    pub(crate) prev_cost_dual: T,
    /// primal residual from previous iteration
    pub(crate) prev_res_primal: T,
    /// dual residual from previous iteration
    pub(crate) prev_res_dual: T,
    /// componentwise dual residual from previous iteration
    pub(crate) prev_res_dual_componentwise: Option<T>,
    /// audit-normalized dual residual from previous iteration
    pub(crate) prev_res_dual_qnorm: T,
    /// absolute duality gap from previous iteration
    pub(crate) prev_gap_abs: T,
    /// relative duality gap from previous iteration
    pub(crate) prev_gap_rel: T,
    /// solve time
    pub solve_time: f64,
    /// solver status
    pub status: SolverStatus,

    /// linear solver information
    pub linsolver: LinearSolverInfo,

    // target stream for printing
    pub(crate) stream: PrintTarget,

    /// Objective constant of presolve-fixed variables (original units).
    pub(crate) objective_offset: T,
}

impl<T> DefaultInfo<T>
where
    T: FloatT,
{
    /// creates a new `DefaultInfo` object
    pub fn new() -> Self {
        Self::default()
    }

    fn update_impl(
        &mut self,
        data: &mut DefaultProblemData<T>,
        variables: &DefaultVariables<T>,
        residuals: &DefaultResiduals<T>,
        timers: &Timers,
        pool: Option<std::sync::Arc<rayon::ThreadPool>>,
    ) {
        let normb = data.get_normb();
        let normq = data.get_normq();
        let eq = &data.equilibration;
        let owner = ResidualOwner {
            products: residuals.products,
            norms: [
                NormView::dense(&variables.x, &eq.d),
                NormView::dense(&variables.z, &eq.e),
                NormView::dense(&variables.s, &eq.einv),
                NormView::dense(&residuals.rx_inf, &eq.dinv),
                NormView::dense(&residuals.Px, &eq.dinv),
                NormView::dense(&residuals.rz_inf, &eq.einv),
                NormView::dense(&residuals.rz, &eq.einv),
                NormView::dense(&residuals.rx, &eq.dinv),
            ],
            dual_componentwise: residuals.dual_componentwise,
        };
        // Current local/replicated solves have exactly one counted owner.
        // A sharded caller must first sum coupled operator contributions and
        // supply each residual coordinate once, then use the same summary API.
        let summary = ResidualSummary::from_owners(std::iter::once(owner), pool.as_deref())
            .expect("one residual owner has consistent settings");
        self.update_from_summary(
            &summary,
            variables.τ,
            variables.κ,
            T::recip(eq.c),
            normb,
            normq,
        );

        // solve time so far (includes setup)
        self.solve_time = timers.total_time().as_secs_f64();
        if let Some(world) = crate::mpi::World::get() {
            // A local deadline must not let one rank leave while its peers
            // enter the next numerical collective. All ranks observe the
            // slowest elapsed clock before applying the same stopping rule.
            self.solve_time = world.allreduce_max_f64(self.solve_time);
        }
    }
    /// Consume global residual statistics without accessing full iterate arrays.
    pub(crate) fn update_from_summary(
        &mut self,
        summary: &ResidualSummary<T>,
        tau: T,
        kappa: T,
        cinv: T,
        normb: T,
        normq: T,
    ) {
        let τinv = T::recip(tau);
        // primal and dual costs. dot products are invariant w.r.t
        // equilibration, but we still need to back out the overall
        // objective scaling term c

        let xPx_τinvsq_over2 = summary.products.xpx * τinv * τinv / (2.).as_T();
        self.cost_primal = (summary.products.qx * τinv + xPx_τinvsq_over2) * cinv;
        self.cost_dual = (-summary.products.bz * τinv - xPx_τinvsq_over2) * cinv;
        self.cost_primal += self.objective_offset;
        self.cost_dual += self.objective_offset;

        let [mut normx, mut normz, mut norms, rx_inf_ns, px_ns, rz_inf_ns, rz_ns, rx_ns] =
            summary.norms();
        normz *= cinv;

        // variables norms, undoing the equilibration.  Do not unscale
        // by τ yet because the infeasibility residuals are ratios of
        // terms that have no affine parts anyway

        // primal and dual infeasibility residuals.
        self.res_primal_inf = (rx_inf_ns * cinv) / T::max(T::one(), normz);
        self.res_dual_inf = T::max(
            px_ns / T::max(T::one(), normx),
            rz_inf_ns / T::max(T::one(), normx + norms),
        );

        // now back out the τ scaling so we can normalize the unscaled primal / dual errors
        normx *= τinv;
        normz *= τinv;
        norms *= τinv;

        self.residual_norms = (rz_ns, rx_ns);

        // primal and dual relative residuals.
        self.res_primal = rz_ns * τinv / T::max(T::one(), normb + normx + norms);
        self.res_dual = rx_ns * τinv * cinv / T::max(T::one(), normq + normx + normz);
        self.res_dual_componentwise = summary.dual_componentwise;
        self.res_dual_qnorm = rx_ns * τinv * cinv / (T::one() + normq);

        // absolute and relative gaps
        self.gap_abs = T::abs(self.cost_primal - self.cost_dual);
        self.gap_rel = self.gap_abs
            / T::max(
                T::one(),
                T::min(T::abs(self.cost_primal), T::abs(self.cost_dual)),
            );

        // κ/τ ratio (scaled)
        self.ktratio = kappa * τinv;
    }
}

impl<T> Info<T> for DefaultInfo<T>
where
    T: FloatT,
{
    type V = DefaultVariables<T>;
    type R = DefaultResiduals<T>;

    fn set_linear_solver_info(&mut self, info: LinearSolverInfo) {
        self.linsolver = info;
    }

    fn reset(&mut self, timers: &mut Timers) {
        self.status = SolverStatus::Unsolved;
        self.iterations = 0;
        self.solve_time = 0f64;
        self.res_dual_componentwise = None;
        self.prev_res_dual_componentwise = None;
        self.res_dual_qnorm = T::zero();
        self.prev_res_dual_qnorm = T::zero();

        timers.start_solve();
    }

    fn post_process(&mut self, residuals: &DefaultResiduals<T>, settings: &DefaultSettings<T>) {
        // if there was an error or we ran out of time
        // or iterations, check for partial convergence

        if self.status.is_errored()
            || matches!(self.status, SolverStatus::MaxIterations)
            || matches!(self.status, SolverStatus::MaxTime)
        {
            self.check_convergence_almost(residuals, settings);
        }
    }

    fn finalize(&mut self, timers: &mut Timers) {
        //final check of timers
        self.solve_time = timers.total_time().as_secs_f64();
        if let Some(world) = crate::mpi::World::get() {
            self.solve_time = world.allreduce_max_f64(self.solve_time);
        }
    }

    fn update(
        &mut self,
        data: &mut DefaultProblemData<T>,
        variables: &DefaultVariables<T>,
        residuals: &DefaultResiduals<T>,
        timers: &Timers,
    ) {
        self.update_impl(data, variables, residuals, timers, None);
    }

    fn update_with_pool(
        &mut self,
        data: &mut DefaultProblemData<T>,
        variables: &DefaultVariables<T>,
        residuals: &DefaultResiduals<T>,
        timers: &Timers,
        pool: Option<std::sync::Arc<rayon::ThreadPool>>,
    ) {
        self.update_impl(data, variables, residuals, timers, pool);
    }

    fn check_termination(
        &mut self,
        residuals: &DefaultResiduals<T>,
        settings: &DefaultSettings<T>,
        iter: u32,
    ) -> bool {
        //  optimality or infeasibility
        // ---------------------
        self.check_convergence_full(residuals, settings);

        //  poor progress
        // ----------------------
        if self.status == SolverStatus::Unsolved
            && iter > 1u32
            && !crate::solver::core::test_no_progress_stop()
            && (self.res_dual > self.prev_res_dual || self.res_primal > self.prev_res_primal)
        {
            // Poor progress at high tolerance.
            if self.ktratio < T::epsilon() * (100.).as_T()
                && (self.prev_gap_abs < settings.tol_gap_abs
                    || self.prev_gap_rel < settings.tol_gap_rel)
            {
                self.status = SolverStatus::InsufficientProgress;
            }

            // Going backwards. Stop immediately if residuals diverge out of feasibility tolerance.
            #[allow(clippy::collapsible_if)] // nested if for readability
            if self.ktratio < T::one() {
                if (self.res_dual > settings.tol_feas * (100.).as_T()
                    && self.res_dual > self.prev_res_dual * (100.).as_T())
                    || (self.res_primal > settings.tol_feas * (100.).as_T()
                        && self.res_primal > self.prev_res_primal * (100.).as_T())
                {
                    self.status = SolverStatus::InsufficientProgress;
                }
            }
        }

        // time or iteration limits
        // ----------------------
        if self.status == SolverStatus::Unsolved {
            if settings.max_iter == self.iterations {
                self.status = SolverStatus::MaxIterations;
            } else if self.solve_time > settings.time_limit {
                self.status = SolverStatus::MaxTime;
            }
        }

        // return TRUE if we settled on a final status
        self.status != SolverStatus::Unsolved
    }

    fn restart(&mut self) {
        self.status = SolverStatus::Unsolved;
        // A restarted iterate is compared with its own history only.
        self.prev_res_primal = T::infinity();
        self.prev_res_dual = T::infinity();
        self.prev_gap_abs = T::infinity();
        self.prev_gap_rel = T::infinity();
        self.prev_res_dual_componentwise = None;
        self.prev_res_dual_qnorm = T::infinity();
    }

    fn save_prev_iterate(&mut self, variables: &Self::V, prev_variables: &mut Self::V) {
        self.prev_cost_primal = self.cost_primal;
        self.prev_cost_dual = self.cost_dual;
        self.prev_res_primal = self.res_primal;
        self.prev_res_dual = self.res_dual;
        self.prev_res_dual_componentwise = self.res_dual_componentwise;
        self.prev_res_dual_qnorm = self.res_dual_qnorm;
        self.prev_gap_abs = self.gap_abs;
        self.prev_gap_rel = self.gap_rel;

        prev_variables.copy_from(variables);
    }

    fn reset_to_prev_iterate(&mut self, variables: &mut Self::V, prev_variables: &Self::V) {
        self.cost_primal = self.prev_cost_primal;
        self.cost_dual = self.prev_cost_dual;
        self.res_primal = self.prev_res_primal;
        self.res_dual = self.prev_res_dual;
        self.res_dual_componentwise = self.prev_res_dual_componentwise;
        self.res_dual_qnorm = self.prev_res_dual_qnorm;
        self.gap_abs = self.prev_gap_abs;
        self.gap_rel = self.prev_gap_rel;

        variables.copy_from(prev_variables);
        self.ktratio = variables.κ * T::recip(variables.τ);
    }

    fn save_scalars(&mut self, μ: T, α: T, σ: T, iter: u32) {
        self.mu = μ;
        self.step_length = α;
        self.sigma = σ;
        self.iterations = iter;
    }

    fn get_status(&self) -> SolverStatus {
        self.status
    }

    fn gap_rel(&self) -> Option<T> {
        Some(self.gap_rel)
    }

    fn residual_max(&self) -> Option<T> {
        Some(T::max(self.res_primal, self.res_dual))
    }

    fn residual_norms(&self) -> Option<(T, T)> {
        Some(self.residual_norms)
    }

    fn set_status(&mut self, status: SolverStatus) {
        self.status = status;
    }
}

// Utility functions for convergence checkiing

impl<T> DefaultInfo<T>
where
    T: FloatT,
{
    fn check_convergence_full(
        &mut self,
        residuals: &DefaultResiduals<T>,
        settings: &DefaultSettings<T>,
    ) {
        // "full" tolerances
        let tol_gap_abs = settings.tol_gap_abs;
        let tol_gap_rel = settings.tol_gap_rel;
        let tol_feas = settings.tol_feas;
        let tol_infeas_abs = settings.tol_infeas_abs;
        let tol_infeas_rel = settings.tol_infeas_rel;
        let tol_ktratio = settings.tol_ktratio;

        let solved_status = SolverStatus::Solved;
        let pinf_status = SolverStatus::PrimalInfeasible;
        let dinf_status = SolverStatus::DualInfeasible;

        self.check_convergence(
            residuals,
            tol_gap_abs,
            tol_gap_rel,
            tol_feas,
            tol_infeas_abs,
            tol_infeas_rel,
            tol_ktratio,
            settings.tol_feas_componentwise,
            settings.tol_dual_qnorm,
            solved_status,
            pinf_status,
            dinf_status,
        );
    }

    fn check_convergence_almost(
        &mut self,
        residuals: &DefaultResiduals<T>,
        settings: &DefaultSettings<T>,
    ) {
        // "almost" tolerances
        let tol_gap_abs = settings.reduced_tol_gap_abs;
        let tol_gap_rel = settings.reduced_tol_gap_rel;
        let tol_feas = settings.reduced_tol_feas;
        let tol_infeas_abs = settings.reduced_tol_infeas_abs;
        let tol_infeas_rel = settings.reduced_tol_infeas_rel;
        let tol_ktratio = settings.reduced_tol_ktratio;

        let solved_status = SolverStatus::AlmostSolved;
        let pinf_status = SolverStatus::AlmostPrimalInfeasible;
        let dinf_status = SolverStatus::AlmostDualInfeasible;

        self.check_convergence(
            residuals,
            tol_gap_abs,
            tol_gap_rel,
            tol_feas,
            tol_infeas_abs,
            tol_infeas_rel,
            tol_ktratio,
            None,
            None,
            solved_status,
            pinf_status,
            dinf_status,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn check_convergence(
        &mut self,
        residuals: &DefaultResiduals<T>,
        tol_gap_abs: T,
        tol_gap_rel: T,
        tol_feas: T,
        tol_infeas_abs: T,
        tol_infeas_rel: T,
        tol_ktratio: T,
        tol_componentwise: Option<T>,
        tol_qnorm: Option<T>,
        solved_status: SolverStatus,
        pinf_status: SolverStatus,
        dinf_status: SolverStatus,
    ) {
        if self.ktratio <= T::one()
            && self.is_solved(
                tol_gap_abs,
                tol_gap_rel,
                tol_feas,
                tol_componentwise,
                tol_qnorm,
            )
        {
            self.status = solved_status;
        // hardcoded factor 1000 here should be fixed
        } else if self.ktratio > tol_ktratio.recip() * (1000.0).as_T() {
            if self.is_primal_infeasible(residuals, tol_infeas_abs, tol_infeas_rel) {
                self.status = pinf_status;
            } else if self.is_dual_infeasible(residuals, tol_infeas_abs, tol_infeas_rel) {
                self.status = dinf_status;
            }
        }
    }

    fn is_solved(
        &self,
        tol_gap_abs: T,
        tol_gap_rel: T,
        tol_feas: T,
        tol_componentwise: Option<T>,
        tol_qnorm: Option<T>,
    ) -> bool {
        ((self.gap_abs < tol_gap_abs) || (self.gap_rel < tol_gap_rel))
            && (self.res_primal < tol_feas)
            && (self.res_dual < tol_feas)
            && tol_qnorm.map_or(true, |tol| self.res_dual_qnorm < tol)
            && tol_componentwise.map_or(true, |tol| {
                self.res_dual_componentwise
                    .is_some_and(|residual| residual < tol)
            })
    }

    fn is_primal_infeasible(
        &self,
        residuals: &DefaultResiduals<T>,
        tol_infeas_abs: T,
        tol_infeas_rel: T,
    ) -> bool {
        (residuals.products.bz < -tol_infeas_abs)
            && (self.res_primal_inf < -tol_infeas_rel * residuals.products.bz)
    }

    fn is_dual_infeasible(
        &self,
        residuals: &DefaultResiduals<T>,
        tol_infeas_abs: T,
        tol_infeas_rel: T,
    ) -> bool {
        (residuals.products.qx < -tol_infeas_abs)
            && (self.res_dual_inf < -tol_infeas_rel * residuals.products.qx)
    }
}

#[cfg(test)]
#[path = "tests/info_accuracy.rs"]
mod accuracy_tests;
