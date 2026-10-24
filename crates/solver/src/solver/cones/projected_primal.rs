//! Experimental dense feasible-primal solve for projected entropy epigraphs.
//! The JSON wrapper distinguishes these programs from ordinary cone inputs.
//! Output contains numerical candidates, never a rigorous certificate.
use super::oprelentrcone::{cholesky, congruence, smat, spd_inverse, svec, Oracle as Ore};
use super::oprelentrlimitcone::{Oracle as Limit, ProjectedOracle};
use super::*;
use crate::algebra::*;
use crate::solver::JsonProblem;
use rayon::prelude::*;
use serde::{de::DeserializeOwned, Deserialize, Serialize};

/// Column-major full-rank map P from the output subspace into the original
/// entropy output. The constraint is P' (T-G) P >= 0.
#[derive(Deserialize)]
pub struct Projection<T> {
    /// Number of independent columns.
    pub rank: usize,
    /// Column-major entries of P in the original output coordinates.
    pub columns: Vec<T>,
}

/// This wrapper is intentionally rejected by the ordinary `sdpx` CLI.
#[derive(Deserialize)]
#[serde(bound = "T: Serialize + DeserializeOwned", deny_unknown_fields)]
pub struct ProjectedProblem<T: FloatT> {
    /// Affine data with the original entropy slot dimensions.
    pub problem: JsonProblem<T>,
    /// One optional output projection per cone; PSD and orthant entries are null.
    pub projections: Vec<Option<Projection<T>>>,
}

enum Oracle<T> {
    Psd(usize, Vec<T>, T),
    Nonnegative(Vec<T>, T),
    Ore(Ore<T>),
    Limit(Limit<T>),
    Projected(ProjectedOracle<T>),
}

impl<T: FloatT> Oracle<T> {
    fn new(c: &SupportedConeT<T>, p: Option<&Projection<T>>, s: &[T]) -> Option<Self> {
        if let Some(p) = p {
            let (n, m) = match *c {
                SupportedConeT::OpRelEntropyConeT(n) => (n, 0),
                SupportedConeT::OpRelEntropyLimitConeT(n, m) => (n, m),
                _ => return None,
            };
            return ProjectedOracle::new(n, m, s, p.rank, &p.columns).map(Self::Projected);
        }
        match *c {
            SupportedConeT::PSDTriangleConeT(n) => {
                let mut l = vec![T::zero(); n * n];
                smat(n, s, &mut l);
                if !cholesky(n, &mut l) {
                    return None;
                }
                let value = -(0..n).fold(T::zero(), |v, i| {
                    v + (T::one() + T::one()) * l[i + i * n].ln()
                });
                let mut inv = vec![T::zero(); n * n];
                spd_inverse(n, &l, &mut inv).then_some(Self::Psd(n, inv, value))
            }
            SupportedConeT::NonnegativeConeT(_) => {
                if s.iter().any(|v| *v <= T::zero()) {
                    return None;
                }
                Some(Self::Nonnegative(
                    s.iter().map(|v| v.recip()).collect(),
                    -s.iter().fold(T::zero(), |v, x| v + x.ln()),
                ))
            }
            SupportedConeT::OpRelEntropyConeT(n) => Ore::new(n, s).map(Self::Ore),
            SupportedConeT::OpRelEntropyLimitConeT(n, m) => Limit::new(n, m, s).map(Self::Limit),
            _ => None,
        }
    }

    fn value_gradient(&self, out: &mut [T]) -> T {
        match self {
            Self::Psd(n, w, v) => {
                svec(*n, w, out);
                out.iter_mut().for_each(|x| *x = -*x);
                *v
            }
            Self::Nonnegative(w, v) => {
                for (o, x) in out.iter_mut().zip(w) {
                    *o = -*x;
                }
                *v
            }
            Self::Ore(o) => {
                o.gradient(out);
                o.barrier
            }
            Self::Limit(o) => {
                o.gradient(out);
                o.barrier
            }
            Self::Projected(o) => {
                o.gradient(out);
                o.barrier()
            }
        }
    }

    fn product(&mut self, d: &[T], out: &mut [T]) {
        match self {
            Self::Psd(n, w, _) => {
                let mut h = vec![T::zero(); *n * *n];
                let mut tmp = h.clone();
                let mut result = h.clone();
                smat(*n, d, &mut h);
                congruence(*n, w, &h, false, &mut tmp, &mut result);
                svec(*n, &result, out);
            }
            Self::Nonnegative(w, _) => {
                for ((o, d), w) in out.iter_mut().zip(d).zip(w) {
                    *o = *d * *w * *w;
                }
            }
            Self::Ore(o) => o.hess_prod(d, out),
            Self::Limit(o) => o.hess_prod(d, out),
            Self::Projected(o) => o.hess_prod(d, out),
        }
    }
}

struct Block<T> {
    range: std::ops::Range<usize>,
    columns: Vec<(usize, Vec<T>)>,
}

fn evaluate<T: FloatT>(
    p: &ProjectedProblem<T>,
    blocks: &[Block<T>],
    s: &[T],
    hessian: bool,
) -> Result<(T, Vec<T>, Vec<T>), String> {
    let n = p.problem.q.len();
    let parts: Vec<_> = blocks
        .par_iter()
        .enumerate()
        .map(|(ci, block)| {
            let mut o = Oracle::new(
                &p.problem.cones[ci],
                p.projections[ci].as_ref(),
                &s[block.range.clone()],
            )
            .ok_or_else(|| format!("projected primal cone {ci} is not interior"))?;
            let mut g = vec![T::zero(); block.range.len()];
            let value = o.value_gradient(&mut g);
            let nc = block.columns.len();
            let mut h = if hessian {
                vec![T::zero(); nc * nc]
            } else {
                Vec::new()
            };
            if hessian {
                if let Oracle::Nonnegative(w, _) = &o {
                    // Initialization bounds have one coefficient per row.
                    // Assemble the diagonal-cone product from its actual
                    // sparsity instead of multiplying dense zero columns.
                    for (row, &weight) in w.iter().enumerate() {
                        let entries: Vec<_> = block
                            .columns
                            .iter()
                            .enumerate()
                            .filter_map(|(j, (_, c))| {
                                (c[row] != T::zero()).then_some((j, c[row] * weight))
                            })
                            .collect();
                        for &(j, a) in &entries {
                            for &(i, b) in &entries {
                                h[i + j * nc] += a * b;
                            }
                        }
                    }
                    return Ok::<_, String>((value, g, h));
                }
                let mut hd = vec![T::zero(); g.len()];
                for (j, (_, col)) in block.columns.iter().enumerate() {
                    o.product(col, &mut hd);
                    for i in 0..=j {
                        let v = T::dot_slices(&block.columns[i].1, &hd);
                        h[i + j * nc] = v;
                        h[j + i * nc] = v;
                    }
                }
            }
            Ok::<_, String>((value, g, h))
        })
        .collect();
    let mut value = T::zero();
    let mut gradient = Vec::with_capacity(s.len());
    let mut h = if hessian {
        vec![T::zero(); n * n]
    } else {
        Vec::new()
    };
    for (block, part) in blocks.iter().zip(parts) {
        let (v, g, hb) = part?;
        value += v;
        gradient.extend(g);
        if hessian {
            let nc = block.columns.len();
            for (j, (cj, _)) in block.columns.iter().enumerate() {
                for (i, (ci, _)) in block.columns.iter().enumerate() {
                    h[ci + cj * n] += hb[i + j * nc];
                }
            }
        }
    }
    Ok((value, gradient, h))
}

fn slack<T: FloatT>(p: &JsonProblem<T>, x: &[T]) -> Vec<T> {
    let mut s = p.b.clone();
    p.A.gemv(&mut s, x, -T::one(), T::one());
    s
}

fn residual<T: FloatT>(p: &JsonProblem<T>, g: &[T], mu: T) -> Vec<T> {
    let mut r: Vec<_> = p.q.iter().map(|q| -*q / mu).collect();
    p.A.t().gemv(&mut r, g, T::one(), T::one());
    r
}

fn solve<T: FloatT>(l: &[T], rhs: &[T]) -> Option<Vec<T>> {
    let n = rhs.len() as i32;
    let mut x = rhs.to_vec();
    let mut info = 0;
    T::xpotrs(b'L', n, 1, l, n, &mut x, n, &mut info);
    (info == 0 && x.iter().all(|v| v.is_finite())).then_some(x)
}

/// Dense Newton continuation with native-precision assembly and factorization.
pub fn run<T: FloatT + Serialize + DeserializeOwned>(
    p: ProjectedProblem<T>,
    mut x: Vec<T>,
    mut mu: T,
    max_iter: usize,
    output: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let data = &p.problem;
    let n = data.q.len();
    data.A.check_format()?;
    if n == 0
        || x.len() != n
        || data.A.n != n
        || data.A.m != data.b.len()
        || data.P.nzval.iter().any(|v| *v != T::zero())
        || !data.sampled.is_empty()
        || p.projections.len() != data.cones.len()
        || mu <= T::zero()
        || !mu.is_finite()
        || data.settings.time_limit.is_nan()
        || data.settings.time_limit <= 0.0
        || data
            .q
            .iter()
            .chain(&data.b)
            .chain(&data.A.nzval)
            .chain(&x)
            .any(|v| !v.is_finite())
    {
        return Err("invalid projected linear program or starting point".into());
    }
    let mut degree = 0usize;
    let mut offset = 0;
    let mut blocks = Vec::new();
    for (ci, cone) in data.cones.iter().enumerate() {
        match *cone {
            SupportedConeT::PSDTriangleConeT(0)
            | SupportedConeT::NonnegativeConeT(0)
            | SupportedConeT::OpRelEntropyConeT(0) => {
                return Err("cone dimensions must be positive".into());
            }
            SupportedConeT::OpRelEntropyLimitConeT(nn, m) if m >= nn => {
                return Err("limit cone requires output dimension greater than U dimension".into());
            }
            _ => {}
        }
        let d = match *cone {
            SupportedConeT::PSDTriangleConeT(n) | SupportedConeT::NonnegativeConeT(n) => n,
            SupportedConeT::OpRelEntropyConeT(n) => 3 * n,
            SupportedConeT::OpRelEntropyLimitConeT(n, m) => 3 * n - 2 * m,
            _ => return Err("projected probe supports PSD, orthant, and entropy cones".into()),
        };
        if let Some(pr) = &p.projections[ci] {
            let (nn, k) = match *cone {
                SupportedConeT::OpRelEntropyConeT(nn) => (nn, nn),
                SupportedConeT::OpRelEntropyLimitConeT(nn, m) => (nn, nn - m),
                _ => return Err("only entropy outputs may be projected".into()),
            };
            if pr.rank > nn
                || pr.columns.len() != nn * pr.rank
                || pr.columns.iter().any(|v| !v.is_finite())
            {
                return Err("invalid output projection".into());
            }
            let mut gram = vec![T::zero(); pr.rank * pr.rank];
            for j in 0..pr.rank {
                for i in 0..pr.rank {
                    gram[i + j * pr.rank] = T::dot_slices(
                        &pr.columns[i * nn..(i + 1) * nn],
                        &pr.columns[j * nn..(j + 1) * nn],
                    );
                }
            }
            if pr.rank > 0 && !cholesky(pr.rank, &mut gram) {
                return Err("projection is not full rank".into());
            }
            degree += pr.rank + 2 * k;
        } else {
            degree += d;
        }
        let range = offset..offset + cone.nvars();
        offset = range.end;
        let mut columns = Vec::new();
        for j in 0..n {
            let entries: Vec<_> = (data.A.colptr[j]..data.A.colptr[j + 1])
                .filter(|&v| range.contains(&data.A.rowval[v]))
                .collect();
            if !entries.is_empty() {
                let mut col = vec![T::zero(); range.len()];
                for v in entries {
                    col[data.A.rowval[v] - range.start] = data.A.nzval[v];
                }
                columns.push((j, col));
            }
        }
        blocks.push(Block { range, columns });
    }
    if offset != data.b.len() {
        return Err("cone dimensions do not match affine rows".into());
    }
    let floor = data.settings.tol_gap_abs / ((4 * degree).as_T());
    let started = std::time::Instant::now();
    let mut hint = T::one();
    for iter in 0..=max_iter {
        let s = slack(data, &x);
        let (f, g, h) = evaluate(&p, &blocks, &s, true)?;
        let r = residual(data, &g, mu);
        let z: Vec<_> = g.iter().map(|v| -mu * *v).collect();
        let primal = data.q.dot(&x);
        let dual = -data.b.dot(&z);
        let gap = (primal - dual).abs();
        let rd = r.iter().fold(T::zero(), |a, v| a.max((mu * *v).abs()));
        let weighted = x
            .iter()
            .zip(&r)
            .fold(T::zero(), |a, (x, r)| a + (mu * *x * *r).abs());
        let tol = data
            .settings
            .tol_gap_abs
            .max(data.settings.tol_gap_rel * T::one().max(primal.abs()));
        let converged = gap < tol && weighted < tol && rd < data.settings.tol_feas;
        let timed_out = started.elapsed().as_secs_f64() >= data.settings.time_limit;
        let status = if converged {
            "CandidateConverged"
        } else if timed_out {
            "MaxTime"
        } else {
            "MaxIterations"
        };
        eprintln!("projected {iter} p {primal:.15e} d {dual:.15e} gap {gap:.3e} rd {rd:.3e} abs_xrd {weighted:.3e} mu {mu:.3e} sec {:.1}",started.elapsed().as_secs_f64());
        let save = |status: &str, x: &[T]| -> Result<(), Box<dyn std::error::Error>> {
            let result = serde_json::json!({"status":status,"precision_bits":T::precision_bits(),"iterations":iter,
                "objective":primal,"dual_objective":dual,"gap_abs":gap,"dual_residual_abs":rd,
                "absolute_weighted_dual_residual":weighted,"mu":mu,"barrier_degree":degree,
                "x":x,"s":s,"z":z,"seconds":started.elapsed().as_secs_f64(),"method":"projected_entropy_primal_probe"});
            let part = output.with_extension("json.part");
            serde_json::to_writer(
                std::io::BufWriter::new(std::fs::File::create(&part)?),
                &result,
            )?;
            std::fs::rename(part, output)?;
            Ok(())
        };
        if converged || timed_out || iter == max_iter || iter % 5 == 0 {
            save(status, &x)?;
        }
        if converged || timed_out || iter == max_iter {
            break;
        }
        let mut l = h.clone();
        if !cholesky(n, &mut l) {
            save("HessianFailed", &x)?;
            return Err(
                "projected Hessian is not positive definite; eliminate affine lineality first"
                    .into(),
            );
        }
        let mut dx = solve(&l, &r).ok_or("projected Newton solve failed")?;
        // Refine the same-precision factor against the assembled Hessian.
        for _ in 0..4 {
            let residual: Vec<_> = (0..n)
                .map(|i| r[i] - T::dot_slices(&h[i * n..(i + 1) * n], &dx))
                .collect();
            let correction = solve(&l, &residual).ok_or("projected refinement failed")?;
            for (x, d) in dx.iter_mut().zip(correction) {
                *x += d;
            }
        }
        let decrement = r.dot(&dx);
        if decrement < T::zero() {
            save("NonDescent", &x)?;
            return Err("projected Newton direction is not descent".into());
        }
        if decrement < (0.01).as_T() && mu > floor {
            let next = (mu * (0.2).as_T()).max(floor);
            hint = next / mu;
            mu = next;
            continue;
        }
        if hint == T::one() && decrement > T::zero() && decrement < (0.0625).as_T() {
            let trial: Vec<_> = x.iter().zip(&dx).map(|(x, d)| *x + *d).collect();
            if let Ok((_, tg, _)) = evaluate(&p, &blocks, &slack(data, &trial), false) {
                let tr = residual(data, &tg, mu);
                let td = solve(&l, &tr).ok_or("projected proximity solve failed")?;
                let margin = T::one() - decrement.sqrt();
                let bound = tr.dot(&td) / (margin * margin);
                if bound >= T::zero() && bound < decrement / (T::one() + T::one()) {
                    x = trial;
                    continue;
                }
            }
        }
        let old = primal + mu * f;
        let mut alpha = hint;
        hint = T::one();
        let mut accepted = false;
        let armijo: T = (0.01).as_T();
        for _ in 0..80 {
            let trial: Vec<_> = x.iter().zip(&dx).map(|(x, d)| *x + alpha * *d).collect();
            if let Ok((ff, _, _)) = evaluate(&p, &blocks, &slack(data, &trial), false) {
                if data.q.dot(&trial) + mu * ff <= old - armijo * alpha * mu * decrement {
                    x = trial;
                    accepted = true;
                    break;
                }
            }
            alpha *= (0.5).as_T();
        }
        eprintln!("  decrement2 {decrement:.3e} alpha {alpha:.3e}");
        if !accepted {
            save("LineSearchFailed", &x)?;
            return Err("projected Armijo search failed".into());
        }
    }
    Ok(())
}
