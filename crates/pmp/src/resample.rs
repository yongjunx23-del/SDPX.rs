//! Recover a block's damped-rational prefactor from explicit sample points
//! and scalings, so the SDPB 3.1 rule can choose new points (`--resample`).
//!
//! PyCFTBoot and SDPB.m XML give s_k = c·r^x_k / Π_i (x_k − q_i) with
//! r = 3 − 2√2 and no prefactor. Then v(x) = r^x / s is a polynomial whose
//! degree is the pole count and whose leading coefficient is 1/c. The fit
//! runs at 2048 bits; the recovered prefactor must reproduce every given
//! scaling, otherwise conversion stops. Scalings only weight the sampled
//! constraints, so the fit's rounding never changes the program's solution.
use crate::{decimal, require, Prefactor, Result};
use sdpx_arithmetic::{MpFloat, Scalar};
use std::str::FromStr;

pub(crate) fn prefactor(points: &[String], scalings: &[String]) -> Result<Prefactor> {
    fit::<MpFloat<32>>(points, scalings)
}

fn fit<W: Scalar + FromStr>(points: &[String], scalings: &[String]) -> Result<Prefactor> {
    let number = |n: f64| W::from_f64(n).unwrap();
    let n = points.len();
    require(
        n >= 2 && scalings.len() == n,
        "resampling needs matching points and scalings",
    )?;
    let x: Vec<W> = points.iter().map(|p| decimal(p)).collect::<Result<_>>()?;
    let base = number(3.0) - number(8.0).sqrt();
    // Newton divided differences of v_k = r^x_k / s_k.
    let mut dd = Vec::with_capacity(n);
    for (&xk, s) in x.iter().zip(scalings) {
        dd.push(base.powf(xk) / decimal::<W>(s)?);
    }
    for j in 1..n {
        for i in (j..n).rev() {
            dd[i] = (dd[i] - dd[i - 1]) / (x[i] - x[i - j]);
        }
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
    require(
        m + 1 < n,
        "resampling cannot determine the pole count: too few sample points",
    )?;
    let value = |z: W| {
        let (mut p, mut dp) = (dd[m], W::zero());
        for j in (0..m).rev() {
            dp = dp * (z - x[j]) + p;
            p = p * (z - x[j]) + dd[j];
        }
        (p, dp)
    };
    // Poles are real and nonpositive: bracket sign changes on a 1/8 grid
    // downwards from below the first sample point, then bisect and polish
    // with Newton steps. Starting above 0 also brackets a pole at 0 that the
    // generator rounded upwards (2^-124 in mixed Ising PyCFTBoot output).
    let step = number(0.125);
    let lowest = -(x[n - 1] + number(4.0 * n as f64 + 200.0));
    let mut poles = Vec::with_capacity(m);
    let start = x[0] / number(2.0);
    let (mut hi, mut fhi) = (start, value(start).0);
    while poles.len() < m && hi > lowest {
        let lo = hi - step;
        let flo = value(lo).0;
        if fhi == W::zero() {
            poles.push(hi);
        } else if (flo < W::zero()) != (fhi < W::zero()) && flo != W::zero() {
            let (mut a, mut b, fb) = (lo, hi, fhi);
            for _ in 0..64 {
                let c = (a + b) / number(2.0);
                if (value(c).0 < W::zero()) == (fb < W::zero()) {
                    b = c;
                } else {
                    a = c;
                }
            }
            let mut q = (a + b) / number(2.0);
            for _ in 0..8 {
                let (p, dp) = value(q);
                q -= p / dp;
            }
            poles.push(q);
        }
        (hi, fhi) = (lo, flo);
    }
    // A pole within rounding of 0 is the unitarity-bound pole at 0.
    let zero = decimal::<W>("1e-30")?;
    require(
        poles.len() == m && poles.iter().all(|q| q.is_finite() && *q <= zero),
        "resampling found no damped-rational prefactor with base 3-2*sqrt(2)",
    )?;
    let constant = dd[m].recip();
    let tolerance = decimal::<W>("1e-60")?;
    for (&xk, s) in x.iter().zip(scalings) {
        let model = poles
            .iter()
            .fold(constant * base.powf(xk), |a, &q| a / (xk - q));
        require(
            (model / decimal::<W>(s)? - W::one()).abs() <= tolerance,
            "recovered prefactor does not reproduce the sample scalings",
        )?;
    }
    // The given scalings are reproduced with the recovered poles; the new
    // prefactor writes a rounded-up pole at 0. Scalings only weight the
    // sampled constraints, so this never changes the program's solution.
    Ok(Prefactor {
        constant: constant.decimal_string(),
        base: base.decimal_string(),
        poles: poles
            .iter()
            .map(|&q| q.min(W::zero()).decimal_string())
            .collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mixed Ising PyCFTBoot output writes the unitarity pole at 0 as 2^-124.
    #[test]
    fn recovers_pole_rounded_above_zero() {
        type W = MpFloat<32>;
        fn convert<W: Scalar>(n: f64) -> W {
            W::from_f64(n).unwrap()
        }
        let number = convert::<W>;
        let poles = [number(2f64.powi(-124)), number(-1.0), number(-1.5), number(-2.0)];
        let base = number(3.0) - number(8.0).sqrt();
        let x: Vec<W> = (0..8).map(|k| number(0.01 + 0.7 * k as f64)).collect();
        let scalings: Vec<String> = x
            .iter()
            .map(|&xk| {
                let s = poles.iter().fold(number(2.5) * base.powf(xk), |a, &q| a / (xk - q));
                s.decimal_string()
            })
            .collect();
        let points: Vec<String> = x.iter().map(Scalar::decimal_string).collect();
        let fit = prefactor(&points, &scalings).unwrap();
        assert_eq!(fit.poles.len(), 4);
        let recovered: Vec<f64> = fit.poles.iter().map(|p| p.parse().unwrap()).collect();
        assert_eq!(recovered[0], 0.0);
        for (r, e) in recovered[1..].iter().zip([-1.0, -1.5, -2.0]) {
            assert!((r - e).abs() < 1e-12, "{r} vs {e}");
        }
    }
}
