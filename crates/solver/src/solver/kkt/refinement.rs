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
    /// Whether a correction with the current factorization is known to gain
    /// less than `stop_ratio` (measured on an earlier right-hand side).
    fn correction_expected_to_stall(&self, _stop_ratio: T) -> bool {
        false
    }
    fn record_correction_ratio(&mut self, _ratio: T) {}
    /// Relative residual (‖e‖/‖b‖) a stalled correction last reached; kept
    /// across factorizations. `None` when unknown.
    fn stall_floor(&self) -> Option<T> {
        None
    }
    fn set_stall_floor(&mut self, _floor: T) {}
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
    // Predicting a stall from an earlier right-hand side (or factorization)
    // saves full-precision residuals and corrections. In binary64 they are
    // cheap, and near convergence a skipped correction leaves an unrefined
    // direction from an ill-conditioned factorization that breaks primal
    // feasibility. binary64 keeps upstream refinement: each right-hand side
    // stops only on its own convergence or measured stall.
    let predict = T::precision_bits() > 53;
    let stop_ratio = settings.iterative_refinement_stop_ratio;
    // A correction with this factorization is known to stall, so no
    // outcome of the residual would lead to one: skip evaluating it and only
    // restore the scaled product H·z that callers reuse.
    if predict && work.correction_expected_to_stall(stop_ratio) {
        work.restore_product();
        return true;
    }
    let normb = work.rhs_norm();
    let mut norme = work.residual(false, true);
    if !work.all_succeeded(norme.is_finite()) {
        return false;
    }
    for pass in 0..settings.iterative_refinement_max_iter {
        let converged = norme
            <= settings.iterative_refinement_abstol + settings.iterative_refinement_reltol * normb;
        if !work.decision_agrees(u32::from(converged)) {
            return false;
        }
        if converged {
            break;
        }
        // Refinement contracts at a rate set by the factorization (about
        // κ·eps of the correction solve). Once a correction with this
        // factorization gained less than the stop ratio, later right-hand
        // sides would pay a full pass for the same stall: stop here instead.
        if predict && work.correction_expected_to_stall(stop_ratio) {
            break;
        }
        // A residual already within a stop-ratio factor of the level a
        // stalled correction last reached cannot gain the stop ratio: the
        // floor is set by the problem's conditioning, not the right-hand
        // side. Residuals far above it (useful corrections) still correct.
        if pass == 0 {
            let at_floor = predict
                && work.stall_floor().is_some_and(|floor| {
                    norme <= settings.iterative_refinement_stop_ratio * floor * normb
                });
            if !work.decision_agrees(u32::from(at_floor)) {
                return false;
            }
            if at_floor {
                break;
            }
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
        if pass == 0 {
            // Only a right-hand side's first correction predicts the next
            // one's; later ones may stall at the floor after a large gain.
            work.record_correction_ratio(ratio);
        }
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
            if normb > T::zero() {
                work.set_stall_floor(T::min(previous, norme) / normb);
            }
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
