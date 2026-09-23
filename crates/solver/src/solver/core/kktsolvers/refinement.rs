//! Shared acceptance policy for the original-operator refinement stages.
use crate::algebra::FloatT;
use crate::mpi::{all_succeeded, decision_agrees};
use crate::solver::core::CoreSettings;

pub(crate) trait Refinement<T: FloatT> {
    fn all_succeeded(&self, value: bool) -> bool {
        all_succeeded(value)
    }
    fn decision_agrees(&self, value: u32) -> bool {
        decision_agrees(value)
    }
    fn rhs_norm(&self) -> T;
    fn residual(&mut self, candidate: bool, reuse_forward: bool) -> T;
    fn solve_correction(&mut self, settings: &CoreSettings<T>) -> bool;
    fn add_correction(&mut self);
    fn accept_candidate(&mut self);
    fn restore_product(&mut self);
}

pub(crate) fn refine<T: FloatT>(work: &mut impl Refinement<T>, settings: &CoreSettings<T>) -> bool {
    if !settings.iterative_refinement_enable {
        return true;
    }
    let normb = work.rhs_norm();
    let mut norme = work.residual(false, true);
    if !work.all_succeeded(norme.is_finite()) {
        return false;
    }
    for _ in 0..settings.iterative_refinement_max_iter {
        let converged = norme
            <= settings.iterative_refinement_abstol + settings.iterative_refinement_reltol * normb;
        if !work.decision_agrees(u32::from(converged)) {
            return false;
        }
        if converged {
            break;
        }
        let previous = norme;
        let solved = work.solve_correction(settings);
        if !work.all_succeeded(solved) {
            return false;
        }
        work.add_correction();
        norme = work.residual(true, false);
        if !work.all_succeeded(norme.is_finite()) {
            return false;
        }
        let ratio = previous / norme;
        let stop = ratio < settings.iterative_refinement_stop_ratio;
        let accept = ratio > T::one();
        let action = if stop {
            if accept {
                1
            } else {
                2
            }
        } else {
            0
        };
        if !work.decision_agrees(action) {
            return false;
        }
        if stop {
            if accept {
                work.accept_candidate();
            } else {
                work.restore_product();
            }
            break;
        }
        work.accept_candidate();
    }
    true
}
