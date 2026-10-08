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

    // GMRES-IR support (see [`refine_gmres`]). `error` doubles as the Arnoldi
    // work vector; implementers keep the bases V (orthonormal) and Z = M⁻¹V.
    fn gmres_supported(&self) -> bool {
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
        if !work.all_succeeded(trial.is_finite()) {
            return false;
        }
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
        if !work.decision_agrees(action) {
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
