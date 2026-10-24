//! Experimental feasible-primal barrier continuation for an existing linear
//! cone program. This probe does not change the production solver or its gates.
use super::oprelentrcone::{cholesky, smat, spd_inverse, svec, Oracle as Ore};
use super::oprelentrlimitcone::Oracle as Limit;
use super::*;
use crate::algebra::*;
use crate::solver::core::traits::{KKTSystem, Residuals};
use crate::solver::core::{ScalingStrategy, StepDirection};
use crate::solver::{DefaultSolver, JsonProblem};
use rayon::prelude::*;
use serde::{de::DeserializeOwned, Serialize};

fn barrier<T: FloatT>(cone: &SupportedConeT<T>, s: &[T]) -> Option<(T, Vec<T>)> {
    let mut g = vec![T::zero(); s.len()];
    let value = match *cone {
        SupportedConeT::PSDTriangleConeT(n) => {
            let mut l = vec![T::zero(); n * n];
            smat(n, s, &mut l);
            if !cholesky(n, &mut l) {
                return None;
            }
            let two = T::one() + T::one();
            let value = -(0..n).fold(T::zero(), |v, i| v + two * l[i + i * n].ln());
            let mut inv = vec![T::zero(); n * n];
            if !spd_inverse(n, &l, &mut inv) {
                return None;
            }
            svec(n, &inv, &mut g);
            g.iter_mut().for_each(|v| *v = -*v);
            value
        }
        SupportedConeT::NonnegativeConeT(_) => {
            if s.iter().any(|v| *v <= T::zero()) {
                return None;
            }
            for (v, &x) in g.iter_mut().zip(s) {
                *v = -x.recip();
            }
            -s.iter().fold(T::zero(), |v, x| v + x.ln())
        }
        SupportedConeT::OpRelEntropyConeT(n) => {
            let o = Ore::new(n, s)?;
            o.gradient(&mut g);
            o.barrier
        }
        SupportedConeT::OpRelEntropyLimitConeT(n, m) => {
            let o = Limit::new(n, m, s)?;
            o.gradient(&mut g);
            o.barrier
        }
        _ => return None,
    };
    Some((value, g))
}

fn all_barriers<T: FloatT>(cones: &[SupportedConeT<T>], s: &[T]) -> Result<(T, Vec<T>), String> {
    let mut offset = 0;
    let pieces: Vec<_> = cones
        .iter()
        .map(|c| {
            let lo = offset;
            offset += c.nvars();
            (c, lo..offset)
        })
        .collect();
    let values: Vec<_> = pieces
        .par_iter()
        .enumerate()
        .map(|(i, (c, r))| {
            barrier(c, &s[r.clone()])
                .ok_or_else(|| format!("primal cone {i} is not strictly feasible"))
        })
        .collect();
    let mut value = T::zero();
    let mut gradient = Vec::with_capacity(s.len());
    for result in values {
        let (v, g) = result?;
        value += v;
        gradient.extend(g);
    }
    Ok((value, gradient))
}

fn slack<T: FloatT>(solver: &DefaultSolver<T>, x: &[T]) -> Vec<T> {
    let mut s = solver.data.b.clone();
    let a = &solver.data.A;
    for j in 0..a.n {
        for p in a.colptr[j]..a.colptr[j + 1] {
            s[a.rowval[p]] -= a.nzval[p] * x[j];
        }
    }
    s
}

/// Start at a supplied strictly feasible point of this same-precision problem.
/// Newton solves use SDPX's existing KKT factorization. Near a center, a
/// local-norm residual contraction avoids subtracting nearly equal barrier
/// values; other steps use Armijo decrease. Dual points are -mu*grad F(s).
pub fn run<T: FloatT + Serialize + DeserializeOwned>(
    mut problem: JsonProblem<T>,
    start_x: Vec<T>,
    mu_original: T,
    max_iter: usize,
    output: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    if !mu_original.is_finite()
        || mu_original <= T::zero()
        || start_x.iter().any(|x| !x.is_finite())
    {
        return Err("probe requires a positive finite mu and finite starting coordinates".into());
    }
    if problem.P.nzval.iter().any(|v| *v != T::zero()) || !problem.sampled.is_empty() {
        return Err("probe requires a linear, explicitly assembled cone program".into());
    }
    let q_original = problem.q.clone();
    let b_original = problem.b.clone();
    problem.settings.presolve_enable = false;
    problem.settings.chordal_decomposition_enable = false;
    let mut solver = problem.into_solver()?;
    if start_x.len() != solver.data.n {
        return Err("start dimension mismatch".into());
    }
    solver.variables.fixed_tau = true;
    solver.variables.τ = T::one();
    solver.variables.κ = T::one();
    for ((x, &v), &d) in solver
        .variables
        .x
        .iter_mut()
        .zip(&start_x)
        .zip(&solver.data.equilibration.dinv)
    {
        *x = v * d;
    }
    let degree: usize = solver.cones.iter().map(|c| c.degree()).sum();
    eprintln!(
        "primal probe: barrier degree {degree}, precision {} bits",
        T::precision_bits()
    );
    let mut mu = mu_original * solver.data.equilibration.c;
    let gap_abs_tol = solver.settings.tol_gap_abs;
    let gap_rel_tol = solver.settings.tol_gap_rel;
    let residual_tol = solver.settings.tol_feas;
    let four: T = (4.).as_T();
    let mu_floor = gap_abs_tol * solver.data.equilibration.c / (four * degree.as_T());
    let armijo: T = (0.01).as_T();
    let mut alpha_hint = T::one();
    let mut reason = "MaxIterations";
    let started = std::time::Instant::now();
    for iter in 0..=max_iter {
        solver.variables.s = slack(&solver, &solver.variables.x);
        let (f, gradient) = all_barriers(&solver.data.cones, &solver.variables.s)?;
        for (z, &g) in solver.variables.z.iter_mut().zip(&gradient) {
            *z = -mu * g;
        }
        solver.residuals.update(&solver.variables, &solver.data);
        let eq = &solver.data.equilibration;
        let x: Vec<_> = solver
            .variables
            .x
            .iter()
            .zip(&eq.d)
            .map(|(&x, &d)| x * d)
            .collect();
        let s: Vec<_> = solver
            .variables
            .s
            .iter()
            .zip(&eq.einv)
            .map(|(&s, &e)| s * e)
            .collect();
        let z: Vec<_> = solver
            .variables
            .z
            .iter()
            .zip(&eq.e)
            .map(|(&z, &e)| z * e / eq.c)
            .collect();
        let p = q_original.dot(&x);
        let d = -b_original.dot(&z);
        let rd = solver
            .residuals
            .rx
            .iter()
            .zip(&eq.dinv)
            .fold(T::zero(), |a, (&v, &di)| a.max((v * di / eq.c).abs()));
        let weighted = solver
            .variables
            .x
            .iter()
            .zip(&solver.residuals.rx)
            .fold(T::zero(), |a, (&x, &r)| a + (x * r / eq.c).abs());
        let gap = (p - d).abs();
        let size = T::one().max(p.abs());
        eprintln!("primal {iter} p {p:.15e} d {d:.15e} gap {gap:.3e} rd {rd:.3e} abs_xrd {weighted:.3e} mu {:.3e} sec {:.1}", mu/eq.c, started.elapsed().as_secs_f64());
        let gap_tol = gap_abs_tol.max(gap_rel_tol * size);
        if gap < gap_tol && weighted < gap_tol && rd < residual_tol {
            reason = "CandidateConverged";
        }
        let save = |status: &str| -> Result<(), Box<dyn std::error::Error>> {
            let result = serde_json::json!({
                "status": status, "precision_bits": T::precision_bits(), "iterations": iter,
                "objective": p, "dual_objective": d, "gap_abs": gap, "dual_residual_abs": rd,
                "absolute_weighted_dual_residual": weighted, "mu": mu/eq.c,
                "complementarity": s.dot(&z), "x": x, "s": s, "z": z,
                "seconds": started.elapsed().as_secs_f64(), "method": "feasible_primal_probe"
            });
            let part = output.with_extension("json.part");
            serde_json::to_writer(
                std::io::BufWriter::new(std::fs::File::create(&part)?),
                &result,
            )?;
            std::fs::rename(part, output)?;
            Ok(())
        };
        if iter == max_iter || reason == "CandidateConverged" || iter % 5 == 0 {
            save(reason)?;
        }
        if iter == max_iter || reason == "CandidateConverged" {
            break;
        }
        if !solver.cones.update_scaling(
            &solver.variables.s,
            &solver.variables.z,
            mu,
            ScalingStrategy::Dual,
        ) {
            return Err("cone scaling failed".into());
        }
        if !solver
            .kktsystem
            .update(&solver.data, &solver.cones, &solver.settings)
        {
            return Err("KKT factorization failed".into());
        }
        solver.step_rhs.x.copy_from_slice(&solver.residuals.rx);
        solver.step_rhs.z.fill(T::zero());
        solver.step_rhs.s.fill(T::zero());
        solver.step_rhs.τ = T::zero();
        solver.step_rhs.κ = T::zero();
        if !solver.kktsystem.solve(
            &mut solver.step_lhs,
            &solver.step_rhs,
            &solver.data,
            &solver.variables,
            &mut solver.cones,
            StepDirection::Combined,
            &solver.settings,
        ) {
            return Err("Newton solve failed".into());
        }
        let descent = solver.step_rhs.x.dot(&solver.step_lhs.x);
        let decrement2 = descent / mu;
        if decrement2 < T::zero() {
            return Err("Newton direction is not descent".into());
        }
        if decrement2 < (0.01).as_T() && mu > mu_floor {
            let reduction: T = (0.2).as_T();
            let next_mu = (mu * reduction).max(mu_floor);
            // With H(new_mu) = beta H(mu), beta times the new Newton
            // direction is the tangent predictor from the old central point.
            // Starting there avoids overshooting the path by 1/beta.
            alpha_hint = next_mu / mu;
            mu = next_mu;
            eprintln!("  centered: decrement2 {decrement2:.3e}; reduce mu");
            continue;
        }
        // Adapted from QICS's projected inverse-Hessian proximity test:
        // He, Saunderson and Fawzi, arXiv:2410.17803, Section 3.5, and
        // qics/_stepper/nonsym.py at be18e5e. This is a feasible-primal
        // centering safeguard, not a port of QICS's full HSD stepper.
        // Here r = -grad(q'x + mu F(b-Ax)), and r'H^-1 r / mu is the
        // squared projected Newton decrement. Reuse the current factor:
        // for delta = ||dx||_F < 1, self-concordance gives
        // H(trial)^-1 <= H(current)^-1 / (1-delta)^2. Thus this bound
        // certifies a contraction in the *trial* local norm without another
        // factorization. Strict cone membership and final gates still apply.
        if alpha_hint == T::one() && decrement2 > T::zero() && decrement2 < (0.0625).as_T() {
            let dx = solver.step_lhs.x.clone();
            let trial: Vec<_> = solver
                .variables
                .x
                .iter()
                .zip(&dx)
                .map(|(&x, &d)| x + d)
                .collect();
            if let Ok((_, trial_gradient)) =
                all_barriers(&solver.data.cones, &slack(&solver, &trial))
            {
                let rhs = solver.step_rhs.x.clone();
                solver.step_rhs.x.copy_from(&solver.data.q);
                solver.step_rhs.x.scale(-T::one());
                solver
                    .data
                    .A
                    .t()
                    .gemv(&mut solver.step_rhs.x, &trial_gradient, mu, T::one());
                let solved = solver.kktsystem.solve(
                    &mut solver.step_lhs,
                    &solver.step_rhs,
                    &solver.data,
                    &solver.variables,
                    &mut solver.cones,
                    StepDirection::Combined,
                    &solver.settings,
                );
                let trial_decrement2 = solver.step_rhs.x.dot(&solver.step_lhs.x) / mu;
                let margin = T::one() - decrement2.sqrt();
                let bound = trial_decrement2 / (margin * margin);
                solver.step_rhs.x = rhs;
                solver.step_lhs.x = dx;
                if solved && bound >= T::zero() && bound < decrement2 / (T::one() + T::one()) {
                    eprintln!(
                        "  proximity center: decrement2 {decrement2:.3e} trial_bound {bound:.3e}"
                    );
                    solver.variables.x = trial;
                    continue;
                }
            }
        }
        let old_value = solver.data.q.dot(&solver.variables.x) + mu * f;
        let mut alpha = alpha_hint;
        alpha_hint = T::one();
        let mut accepted = false;
        for _ in 0..80 {
            let trial: Vec<_> = solver
                .variables
                .x
                .iter()
                .zip(&solver.step_lhs.x)
                .map(|(&x, &dx)| x + alpha * dx)
                .collect();
            let ss = slack(&solver, &trial);
            if let Ok((ff, _)) = all_barriers(&solver.data.cones, &ss) {
                let value = solver.data.q.dot(&trial) + mu * ff;
                if value <= old_value - armijo * alpha * descent {
                    solver.variables.x = trial;
                    accepted = true;
                    break;
                }
            }
            alpha *= (0.5).as_T();
        }
        eprintln!("  decrement2 {decrement2:.3e} alpha {alpha:.3e}");
        if !accepted {
            save("LineSearchFailed")?;
            return Err("primal Armijo search failed".into());
        }
    }
    Ok(())
}
