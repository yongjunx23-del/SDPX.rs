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
    // from 0 downwards, then bisect and polish with Newton steps.
    let step = number(0.125);
    let lowest = -(x[n - 1] + number(4.0 * n as f64 + 200.0));
    let mut poles = Vec::with_capacity(m);
    let (mut hi, mut fhi) = (W::zero(), value(W::zero()).0);
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
    require(
        poles.len() == m && poles.iter().all(|q| q.is_finite() && *q <= W::zero()),
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
    Ok(Prefactor {
        constant: constant.decimal_string(),
        base: base.decimal_string(),
        poles: poles.iter().map(Scalar::decimal_string).collect(),
    })
}
