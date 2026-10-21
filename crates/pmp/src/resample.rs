//! Recover a block's damped-rational prefactor from explicit sample points
//! and scalings, so the SDPB 3.1 rule can choose new points (the CLI default).
//!
//! PyCFTBoot and SDPB.m XML give s_k = c·r^x_k / Π_i (x_k − q_i) with
//! r = 3 − 2√2 and no prefactor. Then v(x) = r^x / s is a polynomial whose
//! degree is the pole count and whose leading coefficient is 1/c. The fit
//! runs at 2048 bits; the recovered prefactor must reproduce every given
//! scaling, otherwise the supplied samples are retained. Scalings only weight
//! the sampled constraints, so fitting never changes the program's solution.
use crate::{decimal, require, Prefactor, Result};
use sdpx_arithmetic::{MpFloat, Scalar};
use std::str::FromStr;

pub(crate) fn prefactor(points: &[String], scalings: &[String]) -> Result<Option<Prefactor>> {
    fit::<MpFloat<32>>(points, scalings)
}

fn fit<W: Scalar + FromStr>(points: &[String], scalings: &[String]) -> Result<Option<Prefactor>> {
    let number = |n: f64| W::from_f64(n).unwrap();
    let n = points.len();
    require(
        n >= 2 && scalings.len() == n,
        "resampling needs matching points and scalings",
    )?;
    let x: Vec<W> = points.iter().map(|p| decimal(p)).collect::<Result<_>>()?;
    let s: Vec<W> = scalings.iter().map(|s| decimal(s)).collect::<Result<_>>()?;
    require(
        x.iter().all(|&x| x >= W::zero()) && x.windows(2).all(|w| w[0] < w[1]),
        "sample points must be nonnegative and increasing",
    )?;
    require(
        s.iter().all(|&s| s > W::zero()),
        "sample scalings must be positive",
    )?;
    let base = number(3.0) - number(8.0).sqrt();
    // Newton divided differences of v_k = r^x_k / s_k.
    let mut dd = Vec::with_capacity(n);
    for (&xk, &sk) in x.iter().zip(&s) {
        dd.push(base.powf(xk) / sk);
    }
    for j in 1..n {
        for i in (j..n).rev() {
            dd[i] = (dd[i] - dd[i - 1]) / (x[i] - x[i - j]);
        }
    }
    if dd.iter().any(|v| !v.is_finite()) {
        return Ok(None);
    }
    // Degree: the last coefficient that is not negligible next to the
    // products it multiplies at the largest point.
    let mut scale = Vec::with_capacity(n);
    let mut product = W::one();
    for j in 0..n {
        scale.push(dd[j].abs() * product);
        product *= (x[n - 1] - x[j]).abs() + W::one();
    }
    let big = scale.iter().fold(W::zero(), |a, &b| a.max(b));
    let tiny = big * decimal::<W>("1e-150")?;
    let m = (0..n).rev().find(|&j| scale[j] > tiny).unwrap_or(0);
    if m + 1 >= n {
        return Ok(None); // No unused sample remains to determine the pole count.
    }
    // Convert the Newton interpolant to monomial coefficients. Derivative
    // roots split a real-rooted polynomial into monotone intervals; a root
    // shared with its derivative retains its full multiplicity.
    let mut coeff = vec![dd[m]];
    for j in (0..m).rev() {
        coeff.push(W::zero());
        for k in (1..coeff.len()).rev() {
            coeff[k] = coeff[k - 1] - x[j] * coeff[k];
        }
        coeff[0] = dd[j] - x[j] * coeff[0];
    }
    let Some(mut poles) = real_roots(&coeff, decimal::<W>("1e-150")?) else {
        return Ok(None);
    };
    poles.reverse();
    // A pole within rounding of 0 is the unitarity-bound pole at 0.
    let zero = decimal::<W>("1e-30")?;
    if poles.iter().any(|q| !q.is_finite() || *q > zero) {
        return Ok(None);
    }
    let constant = dd[m].recip();
    let tolerance = decimal::<W>("1e-60")?;
    for (&xk, &sk) in x.iter().zip(&s) {
        let model = poles
            .iter()
            .fold(constant * base.powf(xk), |a, &q| a / (xk - q));
        if !model.is_finite() || (model / sk - W::one()).abs() > tolerance {
            return Ok(None);
        }
    }
    // The given scalings are reproduced with the recovered poles; the new
    // prefactor writes a rounded-up pole at 0. Scalings only weight the
    // sampled constraints, so this never changes the program's solution.
    Ok(Some(Prefactor {
        constant: constant.decimal_string(),
        base: base.decimal_string(),
        poles: poles
            .iter()
            .map(|&q| q.min(W::zero()).decimal_string())
            .collect(),
    }))
}

// Real roots, including multiplicities, in increasing order. A numerical
// multiple root is accepted only if the final prefactor reproduces all data.
fn real_roots<W: Scalar>(coeff: &[W], tolerance: W) -> Option<Vec<W>> {
    let degree = coeff.len() - 1;
    if degree == 0 {
        return Some(Vec::new());
    }
    if degree == 1 {
        let root = -coeff[0] / coeff[1];
        return root.is_finite().then(|| vec![root]);
    }
    let derivative: Vec<_> = (1..=degree)
        .map(|k| coeff[k] * W::from_usize(k).unwrap())
        .collect();
    let critical = real_roots(&derivative, tolerance)?;
    let value = |x: W| {
        let (mut p, mut scale) = (W::zero(), W::zero());
        for &c in coeff.iter().rev() {
            p = p * x + c;
            scale = scale * x.abs() + c.abs();
        }
        (p, scale)
    };
    let bound = W::one()
        + coeff[..degree]
            .iter()
            .fold(W::zero(), |a, &c| a.max((c / coeff[degree]).abs()));
    if !bound.is_finite() {
        return None;
    }
    let mut stops = vec![-bound];
    let mut roots = Vec::with_capacity(degree);
    for &x in &critical {
        if stops.last() == Some(&x) {
            if roots.last() == Some(&x) {
                roots.push(x);
            }
            continue;
        }
        stops.push(x);
        let (p, scale) = value(x);
        if p.abs() <= scale * tolerance {
            roots.extend([x, x]);
        }
    }
    stops.push(bound);
    for interval in stops.windows(2) {
        let (mut lo, mut hi) = (interval[0], interval[1]);
        let ((flo, slo), (fhi, shi)) = (value(lo), value(hi));
        if flo.abs() <= slo * tolerance
            || fhi.abs() <= shi * tolerance
            || (flo < W::zero()) == (fhi < W::zero())
        {
            continue;
        }
        let two = W::from_usize(2).unwrap();
        let mut x = (lo + hi) / two;
        for _ in 0..W::precision_bits() * 2 {
            let (p, scale) = value(x);
            if p.abs() <= scale * tolerance * tolerance || x == lo || x == hi {
                break;
            }
            if (p < W::zero()) == (flo < W::zero()) {
                lo = x;
            } else {
                hi = x;
            }
            let dp = crate::sampling::evaluate(&derivative, x);
            let next = x - p / dp;
            x = if next.is_finite() && next > lo && next < hi {
                next
            } else {
                (lo + hi) / two
            };
        }
        roots.push(x);
    }
    roots.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (roots.len() == degree).then_some(roots)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Cover multiplicity, close roots and the rounded-up Ising unitarity pole.
    #[test]
    fn recovers_repeated_close_and_rounded_poles() {
        type W = MpFloat<32>;
        fn convert<W: Scalar>(n: f64) -> W {
            W::from_f64(n).unwrap()
        }
        let number = convert::<W>;
        let poles = [
            number(2f64.powi(-124)),
            number(-1.01),
            number(-1.02),
            number(-1.05),
            number(-1.05),
            number(-1000.0),
        ];
        let base = number(3.0) - number(8.0).sqrt();
        let x: Vec<W> = (0..10).map(|k| number(0.01 + 0.7 * k as f64)).collect();
        let scalings: Vec<String> = x
            .iter()
            .map(|&xk| {
                let s = poles
                    .iter()
                    .fold(number(2.5) * base.powf(xk), |a, &q| a / (xk - q));
                s.decimal_string()
            })
            .collect();
        let points: Vec<String> = x.iter().map(Scalar::decimal_string).collect();
        let fit = prefactor(&points, &scalings).unwrap().unwrap();
        assert_eq!(fit.poles.len(), poles.len());
        let recovered: Vec<f64> = fit.poles.iter().map(|p| p.parse().unwrap()).collect();
        assert_eq!(recovered[0], 0.0);
        for (r, e) in recovered[1..]
            .iter()
            .zip([-1.01, -1.02, -1.05, -1.05, -1000.0])
        {
            assert!((r - e).abs() < 1e-12, "{r} vs {e}");
        }
    }
}
