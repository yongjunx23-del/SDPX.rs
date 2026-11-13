//! Bootstrap post-processing of an SDPX solution of an SDPB sampled SDP:
//! SDPB-layout output files, extremal-functional spectrum extraction and
//! first-order objective estimates for nearby SDPs.
//!
//! Everything works from the sampled directory and the solver's JSON result.
//! Each block's functional is a polynomial matrix of degree `num_points - 1`
//! known at the sample points, so it is evaluated by barycentric
//! interpolation without the original PMP.
use crate::{decimal, require, Result};
use sdpx_arithmetic::Scalar;
use serde_json::{json, Value};
use std::{fs, io::Write, path::Path, str::FromStr};

fn read_json(path: &Path) -> Result<Value> {
    let file = fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(serde_json::from_reader(std::io::BufReader::new(file))?)
}
fn number<T: Scalar + FromStr>(v: &Value) -> Result<T> {
    match v {
        Value::String(s) => decimal(s),
        Value::Number(n) => decimal(&n.to_string()),
        _ => Err("expected a decimal string".into()),
    }
}
fn vector<T: Scalar + FromStr>(v: &Value, name: &str) -> Result<Vec<T>> {
    v.as_array()
        .ok_or_else(|| format!("missing array {name}"))?
        .iter()
        .map(number)
        .collect()
}
fn dot<T: Scalar>(a: &[T], b: &[T]) -> T {
    a.iter().zip(b).fold(T::zero(), |s, (&x, &y)| s + x * y)
}
fn norm<T: Scalar>(a: &[T]) -> T {
    dot(a, a).sqrt()
}
fn s<T: Scalar>(v: T) -> String {
    v.decimal_string()
}

/// One sampled block: c and B rows in (entry, sample) order, entries
/// (r, c) with r <= c in column order, as written by pmp2sdp and SDPB.
struct Block<T> {
    dim: usize,
    points: Option<Vec<T>>,
    scalings: Option<Vec<T>>,
    c: Vec<T>,
    b: Vec<Vec<T>>,
    num_points: usize,
}
/// The sampled SDP: objective constant, b, and blocks.
struct Sdp<T> {
    constant: T,
    b: Vec<T>,
    blocks: Vec<Block<T>>,
}
fn read_sdp<T: Scalar + FromStr>(dir: &Path) -> Result<Sdp<T>> {
    let control = read_json(&dir.join("control.json"))?;
    let count = control["num_blocks"]
        .as_u64()
        .ok_or("control.json: num_blocks")? as usize;
    let objectives = read_json(&dir.join("objectives.json"))?;
    let info = dir
        .join("pmp_info.json")
        .exists()
        .then(|| read_json(&dir.join("pmp_info.json")))
        .transpose()?;
    let b = vector::<T>(&objectives["b"], "b")?;
    let mut blocks = Vec::with_capacity(count);
    for j in 0..count {
        let bi = read_json(&dir.join(format!("block_info_{j}.json")))?;
        let data = read_json(&dir.join(format!("block_data_{j}.json")))?;
        let dim = bi["dim"].as_u64().ok_or("block_info: dim")? as usize;
        let num_points = bi["num_points"].as_u64().ok_or("block_info: num_points")? as usize;
        let c = vector::<T>(&data["c"], "c")?;
        let rows = data["B"].as_array().ok_or("block_data: B")?;
        let b_rows = rows
            .iter()
            .map(|r| vector::<T>(r, "B row"))
            .collect::<Result<Vec<_>>>()?;
        require(
            c.len() == dim * (dim + 1) / 2 * num_points
                && b_rows.len() == c.len()
                && b_rows.iter().all(|r| r.len() == b.len()),
            "inconsistent block dimensions (B must be dense rows)",
        )?;
        let meta = info.as_ref().and_then(|i| i.get(j));
        let field = |name: &str| -> Result<Option<Vec<T>>> {
            meta.and_then(|m| m.get(name))
                .map(|v| vector::<T>(v, name))
                .transpose()
        };
        let (points, scalings) = (field("samplePoints")?, field("sampleScalings")?);
        require(
            points.as_ref().map_or(true, |p| p.len() == num_points)
                && scalings.as_ref().map_or(true, |p| p.len() == num_points),
            "pmp_info sample counts differ from block_info",
        )?;
        blocks.push(Block {
            dim,
            points,
            scalings,
            c,
            b: b_rows,
            num_points,
        });
    }
    Ok(Sdp {
        constant: number(&objectives["constant"])?,
        b,
        blocks,
    })
}

/// The parts of an SDPX result used here: SDPB x (= conic x), y and the
/// conic s/z with the PSD layout of every Gram matrix.
struct Solution<T> {
    raw: Value,
    x: Vec<T>,
    y: Vec<T>,
    s: Vec<T>,
    z: Vec<T>,
}
fn read_solution<T: Scalar + FromStr>(path: &Path) -> Result<Solution<T>> {
    let raw = read_json(path)?;
    require(
        raw["sampled"].is_object() && raw["sampled_y"].is_array(),
        "solution must come from an SDPX solve of a sampled directory",
    )?;
    Ok(Solution {
        x: vector(&raw["x"], "x")?,
        y: vector(&raw["sampled_y"], "sampled_y")?,
        s: vector(&raw["s"], "s")?,
        z: vector(&raw["z"], "z")?,
        raw,
    })
}

/// Lagrangian c.x + b.y - y'Bx + constant at (x, y). At an optimum it is the
/// objective; for a nearby SDP it is the first-order objective estimate.
fn lagrangian<T: Scalar>(sdp: &Sdp<T>, x: &[T], y: &[T]) -> Result<T> {
    let mut value = sdp.constant + dot(&sdp.b, y);
    let mut k = 0;
    for block in &sdp.blocks {
        for (c, row) in block.c.iter().zip(&block.b) {
            require(k < x.len(), "solution x is shorter than the SDP")?;
            value += x[k] * (*c - dot(row, y));
            k += 1;
        }
    }
    require(k == x.len(), "solution x does not match the SDP")?;
    Ok(value)
}

/// `sdpx-bootstrap approx-objective`: first-order objective of NEW at the
/// solution of the base SDP (same block structure), as in SDPB's
/// approx_objective linear term.
pub fn approx_objective<T: Scalar + FromStr>(
    base: &Path,
    new: &Path,
    solution: &Path,
) -> Result<Value> {
    let sol = read_solution::<T>(solution)?;
    let (old, new_sdp) = (read_sdp::<T>(base)?, read_sdp::<T>(new)?);
    require(
        old.b.len() == new_sdp.b.len()
            && old.blocks.len() == new_sdp.blocks.len()
            && old
                .blocks
                .iter()
                .zip(&new_sdp.blocks)
                .all(|(a, b)| a.c.len() == b.c.len()),
        "base and new SDPs differ in structure",
    )?;
    let base_value = lagrangian(&old, &sol.x, &sol.y)?;
    let new_value = lagrangian(&new_sdp, &sol.x, &sol.y)?;
    Ok(json!({
        "base_objective": s(base_value),
        "approx_objective": s(new_value),
        "d_objective": s(new_value - base_value),
    }))
}

fn terminate_reason(status: &str) -> &str {
    match status {
        "Solved" => "found primal-dual optimal solution",
        "DualFeasible" => "found dual feasible solution",
        "PrimalFeasible" => "found primal feasible solution",
        "DualInfeasible" => "dual infeasible (primal ray found)",
        "PrimalInfeasible" => "primal infeasible (dual ray found)",
        "MaxIterations" => "maxIterations exceeded",
        "MaxTime" => "maxRuntime exceeded",
        other => other,
    }
}

fn write_vector<T: Scalar>(path: &Path, rows: usize, cols: usize, values: &[T]) -> Result<()> {
    let mut out = std::io::BufWriter::new(fs::File::create(path)?);
    writeln!(out, "{rows} {cols}")?;
    for v in values {
        writeln!(out, "{}", s(*v))?;
    }
    out.flush()?;
    Ok(())
}

/// Full symmetric matrix from an svec slice (upper triangle, column order,
/// off-diagonal entries scaled by sqrt(2)).
fn unpack<T: Scalar>(svec: &[T], side: usize) -> Vec<T> {
    let r2 = T::from_usize(2).unwrap().sqrt();
    let mut m = vec![T::zero(); side * side];
    let mut k = 0;
    for c in 0..side {
        for r in 0..=c {
            let v = if r == c { svec[k] } else { svec[k] / r2 };
            m[r * side + c] = v;
            m[c * side + r] = v;
            k += 1;
        }
    }
    m
}

/// `sdpx-bootstrap sdpb-out`: SDPB's output directory layout (out.txt,
/// y.txt, x_j.txt, X_matrix_j_parity.txt, Y_matrix_j_parity.txt). Text
/// files start with a "rows cols" line followed by one value per line.
pub fn sdpb_out<T: Scalar + FromStr>(sdp_dir: &Path, solution: &Path, out: &Path) -> Result<Value> {
    let sol = read_solution::<T>(solution)?;
    let sdp = read_sdp::<T>(sdp_dir)?;
    require(!out.exists(), "output path already exists")?;
    fs::create_dir(out)?;
    let raw = &sol.raw;
    let field = |name: &str| {
        raw[name]
            .as_str()
            .map(str::to_owned)
            .or_else(|| raw[name].as_f64().map(|v| format!("{v:e}")))
    };
    let primal = field("original_objective").unwrap_or_else(|| "nan".into());
    let dual = field("original_dual_objective").unwrap_or_else(|| "nan".into());
    let gap = match (decimal::<T>(&primal), decimal::<T>(&dual)) {
        (Ok(p), Ok(d)) => {
            let scale = T::one().max((p.abs() + d.abs()) / T::from_usize(2).unwrap());
            s((p - d).abs() / scale)
        }
        _ => "nan".into(),
    };
    let status = raw["status"].as_str().unwrap_or("Unknown");
    let mut text = format!(
        "terminateReason = \"{}\";\nprimalObjective = {primal};\ndualObjective   = {dual};\ndualityGap      = {gap};\nprimalError     = {};\ndualError       = {};\nSolver runtime  = {};\n",
        terminate_reason(status),
        field("primal_residual").unwrap_or_else(|| "nan".into()),
        field("dual_residual").unwrap_or_else(|| "nan".into()),
        raw["native_seconds"].as_f64().unwrap_or(f64::NAN),
    );
    text.push_str(&format!(
        "iterations      = {};\nsdpxStatus      = \"{status}\";\n",
        raw["iterations"]
    ));
    fs::write(out.join("out.txt"), text)?;
    write_vector(&out.join("y.txt"), sol.y.len(), 1, &sol.y)?;
    let mut k = 0;
    for (j, block) in sdp.blocks.iter().enumerate() {
        let n = block.c.len();
        require(k + n <= sol.x.len(), "solution x does not match the SDP")?;
        write_vector(&out.join(format!("x_{j}.txt")), n, 1, &sol.x[k..k + n])?;
        k += n;
    }
    require(k == sol.x.len(), "solution x does not match the SDP")?;
    for gram in raw["sampled"]["grams"]
        .as_array()
        .ok_or("missing gram layout")?
    {
        let get = |name: &str| {
            gram[name]
                .as_u64()
                .map(|v| v as usize)
                .ok_or("bad gram layout")
        };
        let (j, parity, start, side) = (
            get("block_index")?,
            get("parity")?,
            get("row_start")?,
            get("side")?,
        );
        let len = side * (side + 1) / 2;
        require(start + len <= sol.s.len(), "gram layout outside s/z")?;
        for (name, v) in [("X", &sol.s), ("Y", &sol.z)] {
            let m = unpack(&v[start..start + len], side);
            write_vector(
                &out.join(format!("{name}_matrix_{j}_{parity}.txt")),
                side,
                side,
                &m,
            )?;
        }
    }
    Ok(json!({"directory": out.display().to_string(), "blocks": sdp.blocks.len()}))
}

/// Cholesky of a symmetric positive definite matrix (row-major, in place);
/// returns the smallest pivot relative to the largest diagonal, or None
/// at the first nonpositive pivot.
fn cholesky<T: Scalar>(a: &mut [T], n: usize) -> Option<T> {
    let largest = (0..n).fold(T::zero(), |m, i| m.max(a[i * n + i].abs()));
    let mut smallest = T::infinity();
    for j in 0..n {
        let mut d = a[j * n + j];
        for k in 0..j {
            d -= a[j * n + k] * a[j * n + k];
        }
        if !(d > T::zero()) {
            return None;
        }
        smallest = smallest.min(d / largest.max(T::min_positive_value()));
        let d = d.sqrt();
        a[j * n + j] = d;
        for i in j + 1..n {
            let mut v = a[i * n + j];
            for k in 0..j {
                v -= a[i * n + k] * a[j * n + k];
            }
            a[i * n + j] = v / d;
        }
    }
    Some(smallest)
}
fn solve_spd<T: Scalar>(mut g: Vec<T>, n: usize, mut r: Vec<T>) -> Option<Vec<T>> {
    cholesky(&mut g, n)?;
    for i in 0..n {
        for k in 0..i {
            r[i] = r[i] - g[i * n + k] * r[k];
        }
        r[i] /= g[i * n + i];
    }
    for i in (0..n).rev() {
        for k in i + 1..n {
            r[i] = r[i] - g[k * n + i] * r[k];
        }
        r[i] /= g[i * n + i];
    }
    Some(r)
}

/// `sdpx-bootstrap certify`: SDPB-style dual "jump". Keep the functional y,
/// add to each Gram matrix Y the smallest (Frobenius) correction that makes
/// <Q_p, Y> = c_p - B_p y hold for every sample column, and test the
/// corrected Y by Cholesky. Success certifies c - B y as a sum of squares at
/// working precision: the functional is positive and the point is excluded.
/// Column p = (s, r, k) pairs with q_k'((Y_rs + Y_sr)/2)q_k in each parity's
/// basis (the solver's sampled operator, `sym` averaging off-diagonals).
pub fn certify<T: Scalar + FromStr>(sdp_dir: &Path, solution: &Path) -> Result<Value> {
    let sol = read_solution::<T>(solution)?;
    let sdp = read_sdp::<T>(sdp_dir)?;
    let half = T::one() / T::from_usize(2).unwrap();
    // Gram layout per (block, parity).
    let mut grams = std::collections::BTreeMap::new();
    for g in sol.raw["sampled"]["grams"]
        .as_array()
        .ok_or("missing gram layout")?
    {
        let get = |name: &str| {
            g[name]
                .as_u64()
                .map(|v| v as usize)
                .ok_or("bad gram layout")
        };
        grams.insert(
            (get("block_index")?, get("parity")?),
            (get("row_start")?, get("side")?),
        );
    }
    let (mut certified, mut worst_pivot, mut worst_before, mut worst_after) =
        (true, T::infinity(), T::zero(), T::zero());
    let mut report = Vec::new();
    for (j, block) in sdp.blocks.iter().enumerate() {
        let data = read_json(&sdp_dir.join(format!("block_data_{j}.json")))?;
        let (dim, np) = (block.dim, block.num_points);
        // Per parity: basis q (h x np) and the Gram matrix Y (side x side).
        let mut parts = Vec::new();
        for (parity, name) in ["bilinear_bases_even", "bilinear_bases_odd"]
            .iter()
            .enumerate()
        {
            let rows = data[*name].as_array().ok_or("missing bilinear basis")?;
            if rows.is_empty() {
                continue;
            }
            let q: Vec<Vec<T>> = rows
                .iter()
                .map(|r| vector(r, name))
                .collect::<Result<_>>()?;
            let &(start, side) = grams
                .get(&(j, parity))
                .ok_or("gram layout missing a block")?;
            require(side == dim * q.len(), "gram side does not match the basis")?;
            parts.push((
                q,
                unpack(&sol.z[start..start + side * (side + 1) / 2], side),
                side,
            ));
        }
        let qkq =
            |q: &Vec<Vec<T>>, y: &[T], side: usize, r: usize, s: usize, k: usize, l: usize| {
                // q_k' Y[r, s] q_l with Y's (r, s) block of size h.
                let h = q.len();
                let mut total = T::zero();
                for a in 0..h {
                    for b in 0..h {
                        total += q[a][k] * y[(r * h + a) * side + s * h + b] * q[b][l];
                    }
                }
                total
            };
        let (mut before, mut after) = (T::zero(), T::zero());
        let mut e = 0;
        for s in 0..dim {
            for r in 0..=s {
                let factor = if r == s { T::one() } else { half };
                let target: Vec<T> = (0..np)
                    .map(|k| block.c[e * np + k] - dot(&block.b[e * np + k], &sol.y))
                    .collect();
                let value = |parts: &Vec<(Vec<Vec<T>>, Vec<T>, usize)>, k: usize| {
                    parts.iter().fold(T::zero(), |acc, (q, y, side)| {
                        acc + (qkq(q, y, *side, r, s, k, k) + qkq(q, y, *side, s, r, k, k)) * half
                    })
                };
                let residual: Vec<T> = (0..np).map(|k| target[k] - value(&parts, k)).collect();
                let scale = norm(&target).max(T::min_positive_value());
                before = before.max(norm(&residual) / scale);
                // G[k, l] = sum over parities of factor (q_k . q_l)^2.
                let mut g = vec![T::zero(); np * np];
                for (q, _, _) in &parts {
                    for k in 0..np {
                        for l in 0..np {
                            let d = q.iter().fold(T::zero(), |acc, row| acc + row[k] * row[l]);
                            g[k * np + l] += factor * d * d;
                        }
                    }
                }
                let w = solve_spd(g, np, residual).ok_or("sample Gram system is singular")?;
                for (q, y, side) in parts.iter_mut() {
                    let h = q.len();
                    for a in 0..h {
                        for b in 0..h {
                            let d = (0..np)
                                .fold(T::zero(), |acc, k| acc + w[k] * q[a][k] * q[b][k])
                                * factor;
                            y[(r * h + a) * *side + s * h + b] += d;
                            if r != s {
                                y[(s * h + a) * *side + r * h + b] += d;
                            }
                        }
                    }
                }
                let residual: Vec<T> = (0..np).map(|k| target[k] - value(&parts, k)).collect();
                after = after.max(norm(&residual) / scale);
                e += 1;
            }
        }
        let mut pivots = Vec::new();
        for (_, y, side) in parts.iter_mut() {
            match cholesky(y, *side) {
                Some(p) => {
                    worst_pivot = worst_pivot.min(p);
                    pivots.push(s(p));
                }
                None => {
                    certified = false;
                    worst_pivot = T::zero();
                    pivots.push("not positive definite".into());
                }
            }
        }
        worst_before = worst_before.max(before);
        worst_after = worst_after.max(after);
        report.push(json!({"block": j, "residual_before": s(before), "residual_after": s(after), "min_pivot": pivots}));
    }
    Ok(json!({
        "certified": certified,
        "status": sol.raw["status"], "iterations": sol.raw["iterations"],
        "min_pivot": s(worst_pivot),
        "max_residual_before": s(worst_before), "max_residual_after": s(worst_after),
        "blocks": report,
    }))
}

/// Barycentric weights, scaled by a constant per factor to stay near unit
/// magnitude (only ratios matter).
fn weights<T: Scalar>(x: &[T]) -> Vec<T> {
    let n = x.len();
    let span = (x[n - 1] - x[0]).abs().max(T::one()) / T::from_usize(4).unwrap();
    (0..n)
        .map(|k| {
            let mut w = T::one();
            for m in 0..n {
                if m != k {
                    w *= span / (x[k] - x[m]);
                }
            }
            w
        })
        .collect()
}
/// Barycentric interpolation coefficients l_k(t), sum_k l_k(t) f_k = f(t).
fn lagrange<T: Scalar>(x: &[T], w: &[T], t: T) -> Vec<T> {
    if let Some(k) = x.iter().position(|&v| v == t) {
        let mut l = vec![T::zero(); x.len()];
        l[k] = T::one();
        return l;
    }
    let mut l: Vec<T> = x.iter().zip(w).map(|(&xk, &wk)| wk / (t - xk)).collect();
    let total = l.iter().fold(T::zero(), |a, &b| a + b);
    l.iter_mut().for_each(|v| *v /= total);
    l
}

/// Eigen-decomposition of a small symmetric matrix (row-major) by cyclic
/// Jacobi at working precision: smallest eigenvalue and its unit vector.
fn min_eigen<T: Scalar>(mut a: Vec<T>, n: usize) -> (T, Vec<T>) {
    let mut v = vec![T::zero(); n * n];
    (0..n).for_each(|i| v[i * n + i] = T::one());
    let two = T::from_usize(2).unwrap();
    for _sweep in 0..100 {
        let off: T = (0..n)
            .flat_map(|i| (0..n).filter(move |&j| j != i).map(move |j| (i, j)))
            .fold(T::zero(), |s, (i, j)| s + a[i * n + j] * a[i * n + j]);
        let diag = (0..n).fold(T::zero(), |s, i| s + a[i * n + i] * a[i * n + i]);
        if off <= T::epsilon() * T::epsilon() * diag || off == T::zero() {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                let apq = a[p * n + q];
                if apq == T::zero() {
                    continue;
                }
                let theta = (a[q * n + q] - a[p * n + p]) / (two * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + T::one()).sqrt());
                let t = if theta == T::zero() { T::one() } else { t };
                let c = T::one() / (t * t + T::one()).sqrt();
                let sn = t * c;
                for k in 0..n {
                    let (akp, akq) = (a[k * n + p], a[k * n + q]);
                    a[k * n + p] = c * akp - sn * akq;
                    a[k * n + q] = sn * akp + c * akq;
                }
                for k in 0..n {
                    let (apk, aqk) = (a[p * n + k], a[q * n + k]);
                    a[p * n + k] = c * apk - sn * aqk;
                    a[q * n + k] = sn * apk + c * aqk;
                }
                for k in 0..n {
                    let (vkp, vkq) = (v[k * n + p], v[k * n + q]);
                    v[k * n + p] = c * vkp - sn * vkq;
                    v[k * n + q] = sn * vkp + c * vkq;
                }
            }
        }
    }
    let i = (0..n)
        .min_by(|&i, &j| a[i * n + i].partial_cmp(&a[j * n + j]).unwrap())
        .unwrap();
    (a[i * n + i], (0..n).map(|k| v[k * n + i]).collect())
}

/// Matrix of a block's sampled values (entry-major) contracted with the
/// interpolation coefficients `l`: entries (r, c) in column order.
fn matrix_at<T: Scalar>(values: &[T], dim: usize, np: usize, l: &[T]) -> Vec<T> {
    let mut m = vec![T::zero(); dim * dim];
    let mut e = 0;
    for c in 0..dim {
        for r in 0..=c {
            let v = dot(&values[e * np..(e + 1) * np], l);
            m[r * dim + c] = v;
            m[c * dim + r] = v;
            e += 1;
        }
    }
    m
}

/// Contract entry-major per-entry rows with v v' (weight 2 off-diagonal, the
/// pairing of the primal x with the Gram entries).
fn contract<T: Scalar>(per_entry: &[T], dim: usize, v: &[T]) -> T {
    let two = T::from_usize(2).unwrap();
    let mut total = T::zero();
    let mut e = 0;
    for c in 0..dim {
        for r in 0..=c {
            let w = if r == c {
                v[r] * v[c]
            } else {
                two * v[r] * v[c]
            };
            total += w * per_entry[e];
            e += 1;
        }
    }
    total
}

/// Least squares min |G c - b| by Householder QR (columns of G are zeros).
fn least_squares<T: Scalar>(mut g: Vec<Vec<T>>, mut b: Vec<T>) -> Vec<T> {
    let (m, n) = (b.len(), g.len());
    let mut diag = vec![T::zero(); n];
    for k in 0..n.min(m) {
        let alpha = norm(&g[k][k..]);
        if alpha == T::zero() {
            continue;
        }
        let alpha = if g[k][k] > T::zero() { -alpha } else { alpha };
        let mut u: Vec<T> = g[k][k..].to_vec();
        u[0] -= alpha;
        let un = dot(&u, &u);
        diag[k] = alpha;
        for col in g.iter_mut().skip(k + 1).chain(std::iter::once(&mut b)) {
            let f = T::from_usize(2).unwrap() * dot(&u, &col[k..]) / un;
            col[k..]
                .iter_mut()
                .zip(&u)
                .for_each(|(c, &ui)| *c -= f * ui);
        }
    }
    let mut c = vec![T::zero(); n];
    for k in (0..n.min(m)).rev() {
        if diag[k] == T::zero() {
            continue;
        }
        let mut v = b[k];
        for j in k + 1..n {
            v -= g[j][k] * c[j];
        }
        c[k] = v / diag[k];
    }
    c
}

/// Golden-section minimization of f on [lo, hi] to working precision.
fn golden<T: Scalar>(mut lo: T, mut hi: T, f: impl Fn(T) -> T) -> T {
    let r = (T::from_usize(5).unwrap().sqrt() - T::one()) / T::from_usize(2).unwrap();
    let (mut a, mut b) = (hi - r * (hi - lo), lo + r * (hi - lo));
    let (mut fa, mut fb) = (f(a), f(b));
    let tol = T::epsilon().sqrt() * (lo.abs() + hi.abs()).max(T::one());
    let mut steps = 0;
    while (hi - lo).abs() > tol && steps < 2000 {
        if fa < fb {
            hi = b;
            b = a;
            fb = fa;
            a = hi - r * (hi - lo);
            fa = f(a);
        } else {
            lo = a;
            a = b;
            fa = fb;
            b = lo + r * (hi - lo);
            fb = f(b);
        }
        steps += 1;
    }
    if fa < fb {
        a
    } else {
        b
    }
}

/// `sdpx-bootstrap spectrum`: zeros of the extremal functional and their
/// weights. For each block with sample metadata, the functional matrix
/// F(x) = (c - B y)/scaling is interpolated through the samples; local
/// minima of lambda_min(F)/(|c| + sum_i |y_i B_i|) on a refined grid over [0, last sample] are
/// refined by golden section; those below `threshold` are zeros. Weights
/// solve the primal equations sum_z w_z v_z' B(x_z) v_z = b in least squares.
pub fn spectrum<T: Scalar + FromStr>(
    sdp_dir: &Path,
    solution: &Path,
    threshold: &str,
    subdivisions: usize,
) -> Result<Value> {
    let sol = read_solution::<T>(solution)?;
    let sdp = read_sdp::<T>(sdp_dir)?;
    let threshold: T = decimal(threshold)?;
    let mut columns = Vec::new();
    let mut zeros = Vec::new();
    let mut blocks = Vec::new();
    for (j, block) in sdp.blocks.iter().enumerate() {
        let (Some(points), Some(scalings)) = (&block.points, &block.scalings) else {
            blocks.push(
                json!({"block": j, "skipped": "no samplePoints/sampleScalings in pmp_info.json"}),
            );
            continue;
        };
        let (dim, np) = (block.dim, block.num_points);
        if np < 2 {
            blocks.push(json!({"block": j, "skipped": "single sample"}));
            continue;
        }
        // Functional values and B rows divided by the scalings, entry-major.
        let mut f = Vec::with_capacity(block.c.len());
        let mut cs = Vec::with_capacity(block.c.len());
        let mut brow = Vec::with_capacity(block.c.len());
        for (k, (c, row)) in block.c.iter().zip(&block.b).enumerate() {
            let sk = scalings[k % np];
            f.push((*c - dot(row, &sol.y)) / sk);
            cs.push(*c / sk);
            brow.push(row.iter().map(|&v| v / sk).collect::<Vec<_>>());
        }
        // Per-equality sampled values y_i B_i / scaling, entry-major, for the
        // cancellation scale |c(t)| + sum_i |y_i B_i(t)|.
        let terms: Vec<Vec<T>> = (0..sdp.b.len())
            .map(|i| brow.iter().map(|r| r[i] * sol.y[i]).collect())
            .collect();
        let w = weights(points);
        let measure = |t: T| -> (T, Vec<T>) {
            let l = lagrange(points, &w, t);
            let m = matrix_at(&f, dim, np, &l);
            let scale = terms
                .iter()
                .fold(norm(&matrix_at(&cs, dim, np, &l)), |acc, v| {
                    acc + norm(&matrix_at(v, dim, np, &l))
                })
                .max(T::min_positive_value());
            let (lam, v) = min_eigen(m, dim);
            (lam / scale, v)
        };
        let lo = T::zero().min(points[0]);
        let mut grid = vec![lo];
        let sub = T::from_usize(subdivisions).unwrap();
        let mut prev = lo;
        for &p in points.iter() {
            if p > prev {
                for i in 1..subdivisions {
                    grid.push(prev + (p - prev) * T::from_usize(i).unwrap() / sub);
                }
                grid.push(p);
            }
            prev = p;
        }
        let values: Vec<T> = grid.iter().map(|&t| measure(t).0).collect();
        let mut found = Vec::new();
        for i in 0..grid.len() {
            let left = i == 0 || values[i] <= values[i - 1];
            let right = i + 1 == grid.len() || values[i] <= values[i + 1];
            // Refine every local minimum: near a double zero the grid value is
            // O(h^2), far above the threshold, until refined.
            if !(left && right) {
                continue;
            }
            let (a, b) = (grid[i.saturating_sub(1)], grid[(i + 1).min(grid.len() - 1)]);
            let t = if a < b {
                golden(a, b, |t| measure(t).0)
            } else {
                grid[i]
            };
            let (value, v) = measure(t);
            let zero = value <= threshold;
            if zero {
                let l = lagrange(points, &w, t);
                // Column: v' B_i(t) v for every equality i.
                let col: Vec<T> = (0..sdp.b.len())
                    .map(|i| {
                        let per_entry: Vec<T> = (0..dim * (dim + 1) / 2)
                            .map(|e| {
                                (0..np).fold(T::zero(), |acc, k| acc + l[k] * brow[e * np + k][i])
                            })
                            .collect();
                        contract(&per_entry, dim, &v)
                    })
                    .collect();
                columns.push(col);
                zeros.push((j, t, v.clone()));
            }
            found.push(json!({"x": s(t), "lambda_min_rel": s(value), "zero": zero,
                "vector": v.iter().map(|&e| s(e)).collect::<Vec<_>>()}));
        }
        blocks.push(
            json!({"block": j, "dim": dim, "range": [s(lo), s(points[np - 1])], "minima": found}),
        );
    }
    let weights_out = if columns.is_empty() {
        json!(null)
    } else {
        // B x = b with x a nonnegative measure on the zeros; solve then report the residual.
        let c = least_squares(columns.clone(), sdp.b.clone());
        let residual: Vec<T> = (0..sdp.b.len())
            .map(|i| {
                columns
                    .iter()
                    .zip(&c)
                    .fold(T::zero(), |acc, (col, &ck)| acc + col[i] * ck)
                    - sdp.b[i]
            })
            .collect();
        json!({
            "relative_residual": s(norm(&residual) / norm(&sdp.b).max(T::min_positive_value())),
            "zeros": zeros.iter().zip(&c).map(|((j, t, _), w)| json!({"block": j, "x": s(*t), "weight": s(*w)})).collect::<Vec<_>>(),
        })
    };
    Ok(json!({"threshold": s(threshold), "blocks": blocks, "spectrum": weights_out}))
}
