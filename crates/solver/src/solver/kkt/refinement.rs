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
    /// `all_succeeded(ok) && decision_agrees(decision)` in one agreement:
    /// the code `2·decision + !ok` agrees on every rank only when every rank
    /// holds the same decision and the same `ok`, and `ok` then is global.
    /// Implementations whose `decision_agrees` is one collective therefore
    /// pay one round instead of two.
    fn agree_pair(&self, ok: bool, decision: u32) -> bool {
        self.decision_agrees(decision.saturating_mul(2) | u32::from(!ok)) && ok
    }
    /// True when a failed `solve_correction` leaves a candidate whose next
    /// residual is non-finite on every rank, so its failure is agreed with
    /// that residual instead of in a round of its own.
    fn defers_solve_agreement(&self) -> bool {
        false
    }
    fn rhs_norm(&self) -> T;
    fn residual(&mut self, candidate: bool, reuse_forward: bool) -> T;
    fn solve_correction(&mut self, settings: &CoreSettings<T>) -> bool;
    fn add_correction(&mut self);
    fn accept_candidate(&mut self);
    fn restore_product(&mut self);
    /// Diagnostic only (`SDPX_TRACE_TAU`): residual parts of the last
    /// evaluated point (implementer-defined), printed after each residual.
    fn trace_parts(&mut self, _candidate: bool) -> Option<String> {
        None
    }

    // GMRES-IR support (see [`refine_gmres`]). `error` doubles as the Arnoldi
    // work vector; implementers keep the bases V (orthonormal) and Z = M⁻¹V.
    fn gmres_supported(&self) -> bool {
        false
    }
    /// Continue a stationary refinement that stalled above its tolerance
    /// with GMRES-IR from the refined point; the implementer then reports
    /// `gmres_supported` and treats the stored forward product as stale.
    fn gmres_continuation(&mut self) -> bool {
        false
    }
    /// Drop the bases of the previous cycle.
    fn gmres_reset(&mut self) {}
    /// Append `V_k = scale · error`.
    fn gmres_push_basis(&mut self, _scale: T) {}
    /// Append `Z_k = M⁻¹ V_k` for the last basis vector.
    fn gmres_precondition(&mut self, _settings: &CoreSettings<T>) -> bool {
        false
    }
    /// `error = K Z_k` for the last preconditioned vector.
    fn gmres_operator(&mut self) -> bool {
        false
    }
    /// Global `⟨error, V_i⟩` for every stored basis vector (one reduction).
    fn gmres_dots(&mut self) -> Vec<T> {
        Vec::new()
    }
    /// `error -= Σ c_i · V_i`.
    fn gmres_subtract(&mut self, _c: &[T]) {}
    /// Global `‖error‖₂`.
    fn gmres_norm2(&mut self) -> T {
        T::nan()
    }
    /// `candidate = x + Σ y_i Z_i`.
    fn gmres_candidate(&mut self, _y: &[T]) {}
    /// Relative residual floor met by the last GMRES-IR solve, if stored.
    fn gmres_floor(&self) -> Option<T> {
        None
    }
    fn set_gmres_floor(&mut self, _floor: T) {}
}

pub(crate) fn refine<T: FloatT>(work: &mut impl Refinement<T>, settings: &CoreSettings<T>) -> bool {
    if !settings.iterative_refinement_enable {
        return true;
    }
    if settings.iterative_refinement_gmres && work.gmres_supported() {
        return refine_gmres(work, settings);
    }
    static TRACE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let mut stats = [T::zero(); 4];
    let ok = refine_stationary(work, settings, &mut stats);
    if *TRACE.get_or_init(|| std::env::var_os("SDPX_TRACE_TAU").is_some()) {
        // Diagnostic only: rhs norm, first and last residual, corrections.
        eprintln!(
            "refine-trace {:.3e} {:.3e} {:.3e} {}",
            stats[0], stats[1], stats[2], stats[3]
        );
    }
    if T::precision_bits() <= 53 {
        let tol =
            settings.iterative_refinement_abstol + settings.iterative_refinement_reltol * stats[0];
        // `ok` and the residual norms are already globally agreed, so no
        // further agreement round is needed.
        let stalled = ok && stats[2] > tol;
        if stalled && work.gmres_continuation() {
            return refine_gmres(work, settings);
        }
    }
    ok
}

fn refine_stationary<T: FloatT>(
    work: &mut impl Refinement<T>,
    settings: &CoreSettings<T>,
    stats: &mut [T; 4],
) -> bool {
    static PARTS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let parts = *PARTS.get_or_init(|| std::env::var_os("SDPX_TRACE_TAU").is_some());
    let normb = work.rhs_norm();
    let tol = settings.iterative_refinement_abstol + settings.iterative_refinement_reltol * normb;
    let mut norme = work.residual(false, true);
    if parts {
        if let Some(s) = work.trace_parts(false) {
            eprintln!("refine-parts 0 {:.3e} {:.3e} {s}", normb, norme);
        }
    }
    *stats = [normb, norme, norme, T::zero()];
    if settings.iterative_refinement_max_iter == 0 {
        return work.all_succeeded(norme.is_finite());
    }
    // Each residual's finiteness, its acceptance action and the next
    // convergence test share one agreement (see `Refinement::agree_pair`).
    let mut converged = norme <= tol;
    if !work.agree_pair(norme.is_finite(), u32::from(converged)) {
        return false;
    }
    for _ in 0..settings.iterative_refinement_max_iter {
        if converged {
            break;
        }
        let previous = norme;
        stats[3] += T::one();
        let solved = work.solve_correction(settings);
        if !work.defers_solve_agreement() && !work.all_succeeded(solved) {
            return false;
        }
        work.add_correction();
        norme = work.residual(true, false);
        if parts {
            if let Some(s) = work.trace_parts(true) {
                eprintln!("refine-parts {} {:.3e} {:.3e} {s}", stats[3], normb, norme);
            }
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
        // Only a continuing step reads the next convergence test.
        let next = !stop && norme <= tol;
        if !work.agree_pair(norme.is_finite(), action + 3 * u32::from(next)) {
            return false;
        }
        if stop {
            if accept {
                work.accept_candidate();
                stats[2] = norme;
            } else {
                work.restore_product();
            }
            break;
        }
        work.accept_candidate();
        stats[2] = norme;
        converged = next;
    }
    true
}

/// GMRES-IR under the same acceptance policy: each cycle runs restarted GMRES
/// on `K d = e` right-preconditioned by the correction solve, forms the
/// candidate `x + d` and keeps it only if its true residual is lower. Inner
/// steps (one correction solve and one operator product each) count against
/// `iterative_refinement_max_iter`. Every branch follows globally reduced
/// values and is confirmed with `decision_agrees`.
pub(crate) fn refine_gmres<T: FloatT>(
    work: &mut impl Refinement<T>,
    settings: &CoreSettings<T>,
) -> bool {
    let normb = work.rhs_norm();
    let tol = settings.iterative_refinement_abstol + settings.iterative_refinement_reltol * normb;
    let mut norme = work.residual(false, true);
    if !work.all_succeeded(norme.is_finite()) {
        return false;
    }
    let maxiter = settings.iterative_refinement_max_iter as usize;
    // Aim at the floor met last time; retry at `tol` when the true residual
    // still follows the Krylov estimate (see the DirectLDL GMRES-IR).
    let mut target = match work.gmres_floor() {
        Some(f) => T::max(tol, f * normb),
        None => tol,
    };
    let mut cap = usize::MAX;
    let mut steps = 0;
    while steps < maxiter {
        let converged = norme <= tol;
        if !work.decision_agrees(u32::from(converged)) {
            return false;
        }
        if converged {
            break;
        }
        work.gmres_reset();
        let beta = work.gmres_norm2();
        if !work.all_succeeded(beta.is_finite() && beta > T::zero()) {
            return false;
        }
        work.gmres_push_basis(T::recip(beta));
        let mut g = vec![beta];
        let mut h: Vec<Vec<T>> = Vec::new();
        let mut rot: Vec<(T, T)> = Vec::new();
        while steps < maxiter {
            steps += 1;
            let ok = work.gmres_precondition(settings);
            if !work.all_succeeded(ok) {
                return false;
            }
            let ok = work.gmres_operator();
            if !work.all_succeeded(ok) {
                return false;
            }
            // Classical Gram–Schmidt applied twice: one global reduction
            // per pass, orthogonal to working precision.
            let j = h.len();
            let mut col = vec![T::zero(); j + 2];
            for _ in 0..2 {
                let c = work.gmres_dots();
                if !work.all_succeeded(c.len() == j + 1 && c.iter().all(|v| v.is_finite())) {
                    return false;
                }
                work.gmres_subtract(&c);
                for (dst, v) in col.iter_mut().zip(c) {
                    *dst += v;
                }
            }
            let hnext = work.gmres_norm2();
            col[j + 1] = hnext;
            for (i, &(cs, sn)) in rot.iter().enumerate() {
                let (a, b) = (col[i], col[i + 1]);
                col[i] = cs * a + sn * b;
                col[i + 1] = -sn * a + cs * b;
            }
            let (a, b) = (col[j], col[j + 1]);
            let rho = T::sqrt(a * a + b * b);
            let (cs, sn) = if rho == T::zero() {
                (T::one(), T::zero())
            } else {
                (a / rho, b / rho)
            };
            col[j] = rho;
            col[j + 1] = T::zero();
            rot.push((cs, sn));
            let gj = g[j];
            g[j] = cs * gj;
            g.push(-sn * gj);
            h.push(col);
            let estimate = T::abs(g[j + 1]);
            let done =
                !estimate.is_finite() || hnext == T::zero() || estimate <= target || h.len() >= cap;
            if !work.decision_agrees(u32::from(done)) {
                return false;
            }
            if done {
                break;
            }
            work.gmres_push_basis(T::recip(hnext));
        }
        let k = h.len();
        let estimate = T::abs(g[k]);
        let mut y = vec![T::zero(); k];
        for i in (0..k).rev() {
            let mut s = g[i];
            for l in i + 1..k {
                s -= h[l][i] * y[l];
            }
            y[i] = s / h[i][i];
        }
        if !work.all_succeeded(y.iter().all(|v| v.is_finite())) {
            return false;
        }
        work.gmres_candidate(&y);
        let trial = work.residual(true, false);
        let ratio = norme / trial;
        let stopratio = settings.iterative_refinement_stop_ratio;
        let floor = trial > stopratio * estimate;
        let action = if ratio <= T::one() {
            2
        } else if trial <= tol || ratio < stopratio {
            1
        } else {
            0
        };
        // A non-finite trial fails the pair whatever its action reads.
        if !work.agree_pair(trial.is_finite(), action) {
            return false;
        }
        if action == 2 {
            work.restore_product();
            break;
        }
        work.accept_candidate();
        norme = trial;
        if floor && cap == usize::MAX {
            work.set_gmres_floor(trial / T::max(normb, T::min_positive_value()));
        }
        if action == 1 {
            break;
        }
        // Later cycles take one step each, as stationary refinement does.
        cap = 1;
        target = tol;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    /// Scalar refinement whose residual after `k` corrections is
    /// `factors[0] * ... * factors[k-1]`; counts agreement rounds.
    struct Mock {
        factors: Vec<f64>,
        accepted: usize,
        candidate: usize,
        restored: bool,
        rounds: Cell<usize>,
        /// Defer the solve flag (a failed solve poisons the candidate).
        defer: bool,
        /// Correction number whose solve fails.
        fail_at: Option<usize>,
        poisoned: bool,
    }
    impl Refinement<f64> for Mock {
        fn all_succeeded(&self, value: bool) -> bool {
            self.rounds.set(self.rounds.get() + 1);
            value
        }
        fn decision_agrees(&self, _value: u32) -> bool {
            self.rounds.set(self.rounds.get() + 1);
            true
        }
        fn rhs_norm(&self) -> f64 {
            1.0
        }
        fn residual(&mut self, candidate: bool, _reuse: bool) -> f64 {
            if candidate && self.poisoned {
                return f64::NAN;
            }
            let k = if candidate {
                self.candidate
            } else {
                self.accepted
            };
            self.factors[..k].iter().product()
        }
        fn solve_correction(&mut self, _settings: &CoreSettings<f64>) -> bool {
            let ok = self.fail_at != Some(self.accepted + 1);
            self.poisoned = !ok && self.defer;
            ok
        }
        fn defers_solve_agreement(&self) -> bool {
            self.defer
        }
        fn add_correction(&mut self) {
            self.candidate = self.accepted + 1;
        }
        fn accept_candidate(&mut self) {
            self.accepted = self.candidate;
        }
        fn restore_product(&mut self) {
            self.restored = true;
        }
    }

    fn run(factors: Vec<f64>, tol: f64, defer: bool, fail_at: Option<usize>, ok: bool) -> Mock {
        let mut settings = CoreSettings::<f64>::default();
        settings.iterative_refinement_abstol = tol;
        settings.iterative_refinement_reltol = 0.0;
        settings.iterative_refinement_max_iter = 10;
        let mut mock = Mock {
            factors,
            accepted: 0,
            candidate: 0,
            restored: false,
            rounds: Cell::new(0),
            defer,
            fail_at,
            poisoned: false,
        };
        assert_eq!(refine(&mut mock, &settings), ok);
        mock
    }

    #[test]
    fn one_agreement_per_residual() {
        // Contracting tenfold: six corrections reach 2e-6. One round for the
        // first residual, then the solve flag and one pair per correction
        // (formerly 1 + 7 convergence + 3 per correction = 26 rounds).
        let m = run(vec![0.1; 10], 2e-6, false, None, true);
        assert_eq!((m.accepted, m.restored), (6, false));
        assert_eq!(m.rounds.get(), 1 + 2 * 6);
        // A stalled step (ratio < 5) that still improves is accepted, then
        // refinement stops; a step that worsens is rejected and restored.
        let m = run(vec![0.1, 0.5, 0.1], 1e-9, false, None, true);
        assert_eq!((m.accepted, m.restored), (2, false));
        let m = run(vec![0.1, 2.0, 0.1], 1e-9, false, None, true);
        assert_eq!((m.accepted, m.restored), (1, true));
        // Already converged: one round, no correction.
        let m = run(vec![], 1.0, false, None, true);
        assert_eq!((m.accepted, m.rounds.get()), (0, 1));
        // A failed correction fails refinement in its own round...
        let m = run(vec![0.1; 10], 2e-6, false, Some(3), false);
        assert_eq!((m.accepted, m.rounds.get()), (2, 1 + 2 * 2 + 1));
    }

    #[test]
    fn deferred_solve_flags_keep_outcomes() {
        // One round per correction, same accepted iterate.
        let m = run(vec![0.1; 10], 2e-6, true, None, true);
        assert_eq!((m.accepted, m.rounds.get()), (6, 1 + 6));
        // A failed correction still fails, in the residual's round.
        let m = run(vec![0.1; 10], 2e-6, true, Some(3), false);
        assert_eq!((m.accepted, m.rounds.get()), (2, 1 + 3));
    }
}
