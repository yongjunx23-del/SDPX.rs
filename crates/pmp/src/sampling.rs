// Adapted from SDPB 3.1.0, src/pmp/convert. See provenance/SDPB-LICENSE.
use crate::{decimal, numbers, require, Result};
use sdpx_arithmetic::Scalar;
use serde::{Deserialize, Serialize};
use std::str::FromStr;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Prefactor {
    pub constant: String,
    pub base: String,
    pub poles: Vec<String>,
}

pub(crate) struct Damped<T> {
    pub constant: T,
    pub base: T,
    pub poles: Vec<T>,
}
impl<T: Scalar + FromStr> Damped<T> {
    pub fn read(input: Option<&Prefactor>, degree: usize) -> Result<Self> {
        let value = match input {
            Some(p) => Self {
                constant: decimal(&p.constant)?,
                base: decimal(&p.base)?,
                poles: numbers(&p.poles)?,
            },
            None => Self {
                constant: T::one(),
                base: if degree == 0 {
                    T::one()
                } else {
                    (-T::one()).exp()
                },
                poles: vec![],
            },
        };
        require(
            value.constant > T::zero() && value.base > T::zero(),
            "prefactor constant and base must be positive",
        )?;
        require(
            value.poles.iter().all(|&p| p <= T::zero()),
            "prefactor poles must be nonpositive",
        )?;
        Ok(value)
    }
    pub fn scalings(&self, points: &[T]) -> Result<Vec<T>> {
        let floor = decimal::<T>("1e-16")?;
        let mut out = Vec::with_capacity(points.len());
        for &x in points {
            let denominator = self
                .poles
                .iter()
                .fold(T::one(), |d, &p| d * (x - p).max(floor));
            let value = self.constant * self.base.powf(x) / denominator;
            require(
                value.is_finite() && value > T::zero(),
                "prefactor scaling overflow or underflow",
            )?;
            out.push(value);
        }
        Ok(out)
    }
    pub fn metadata(&self) -> Prefactor {
        Prefactor {
            constant: self.constant.to_string(),
            base: self.base.to_string(),
            poles: self.poles.iter().map(ToString::to_string).collect(),
        }
    }
}
fn number<T: Scalar>(n: usize) -> T {
    T::from_usize(n).unwrap()
}
fn acos<T: Scalar>(x: T) -> T {
    let x = x.max(-T::one()).min(T::one());
    ((T::one() - x) * (T::one() + x)).sqrt().atan2(x)
}
// Safeguarded Newton iteration followed by representable-neighbor bisection.
// No relative-tolerance exit: keep the original full-precision stopping rule.
fn root<T: Scalar>(mut lo: T, mut hi: T, f: impl Fn(T) -> (T, T)) -> Result<T> {
    let (left, right) = (f(lo).0, f(hi).0);
    require(
        left.is_finite() && right.is_finite() && left <= T::zero() && right >= T::zero(),
        "sampling root is not bracketed",
    )?;
    let two = number::<T>(2);
    let mut x = (lo + hi) / two;
    let mut previous_step = hi - lo;
    let mut polish = false;
    for _ in 0..(T::precision_bits() * 4 + 64) {
        if x == lo || x == hi {
            return Ok(x);
        }
        let (value, derivative) = f(x);
        require(value.is_finite(), "non-finite sampling density")?;
        if value == T::zero() {
            return Ok(x);
        }
        if value < T::zero() {
            lo = x;
        } else {
            hi = x;
        }
        if !polish && derivative.is_finite() && derivative > T::zero() {
            let step = value / derivative;
            let next = x - step;
            // Once Newton reaches rounding noise, certify a small sign bracket
            // and bisect it to neighboring representable values. If this local
            // bracket fails, retain the original bracket and bisect that instead.
            let radius = number::<T>(8) * T::epsilon() * x.abs();
            if next == x || step.abs() <= radius {
                let a = lo.max(x - radius);
                let b = hi.min(x + radius);
                let (fa, fb) = (f(a).0, f(b).0);
                if fa.is_finite() && fb.is_finite() && fa <= T::zero() && fb >= T::zero() {
                    lo = a;
                    hi = b;
                }
                polish = true;
            } else if next.is_finite()
                && next > lo
                && next < hi
                && step.abs() < previous_step * number::<T>(3) / number::<T>(4)
            {
                previous_step = step.abs();
                x = next;
                continue;
            }
        }
        previous_step = (hi - lo) / two;
        x = (lo + hi) / two;
    }
    Err("sampling root did not converge at the requested precision".into())
}
pub(crate) fn points<T: Scalar + FromStr>(n: usize, p: &Damped<T>) -> Result<Vec<T>> {
    require(n > 0, "sample count must be positive")?;
    if n == 1 {
        return Ok(vec![T::zero()]);
    }
    require(
        p.base < T::one(),
        "automatic sampling requires 0 < prefactor base < 1",
    )?;
    let threshold = decimal::<T>("1e-10")?;
    let two = number::<T>(2);
    let sample_count = number::<T>(n);
    let log = p.base.ln();
    // A small outward rounding margin keeps the analytic upper bound
    // bracketed even when multiplication by log(base) rounds downward.
    let upper = -(two * sample_count / log) * (T::one() + number::<T>(16) * T::epsilon());
    let b = root(threshold, upper, |b| {
        let mut value = -b * log / two - sample_count;
        let mut derivative = -log / two;
        for &pole in &p.poles {
            let distance = b - pole;
            let ratio = (-pole / distance).sqrt();
            value = value + T::one() - ratio;
            derivative += ratio / (two * distance);
        }
        (value, derivative)
    })?;
    let small = p
        .poles
        .iter()
        .filter(|&&v| v.abs() <= threshold)
        .count()
        .min(n);
    let mut out = vec![T::zero(); n];
    let mut lower = threshold;
    // These full-precision constants are identical for every density
    // evaluation. Preserve the original operation and summation order.
    let pi = T::PI();
    let density_scale = -log / pi;
    let half_b = b / two;
    let half = T::one() / two;
    let pole_terms: Vec<_> = p
        .poles
        .iter()
        .map(|&pole| (pole, b - pole, (-pole / (b - pole)).sqrt()))
        .collect();
    for (i, slot) in out.iter_mut().enumerate().skip(small) {
        let index = number::<T>(i);
        *slot = root(lower, b, |z| {
            let angle = acos(T::one() - two * z / b);
            let mut density = density_scale * (((b - z) * z).sqrt() + half_b * angle);
            let mut slope = -log;
            for &(pole, distance, ratio) in &pole_terms {
                density +=
                    (acos(T::one() - two * z * distance / (b * (z - pole))) - ratio * angle) / pi;
                slope += ratio / (z - pole);
            }
            let derivative = ((b - z) / z).sqrt() / pi * slope;
            (density - index - half, derivative)
        })?;
        lower = *slot;
    }
    let end = if small == n { b } else { out[small] };
    for (i, slot) in out.iter_mut().enumerate().take(small) {
        *slot = end * number::<T>(i) / number::<T>(small);
    }
    Ok(out)
}

pub(crate) fn evaluate<T: Scalar>(coeff: &[T], x: T) -> T {
    // Separate multiplication/addition matches upstream Horner evaluation.
    coeff
        .iter()
        .rev()
        .copied()
        .reduce(|y, c| y * x + c)
        .unwrap_or_else(T::zero)
}
pub(crate) fn basis<T: Scalar>(points: &[T], scales: &[T]) -> Result<[Vec<Vec<T>>; 2]> {
    let degree = points.len() - 1;
    if degree == 0 {
        return Ok([vec![vec![T::one()]], vec![]]);
    }
    let mut moments = vec![T::zero(); degree + 1];
    for (&x, &s) in points.iter().zip(scales) {
        let mut power = T::one();
        for moment in &mut moments {
            *moment += power * s;
            power *= x;
        }
    }
    let mut result = [Vec::new(), Vec::new()];
    for parity in 0..2 {
        let n = if parity == 0 {
            degree / 2 + 1
        } else {
            (degree + 1) / 2
        };
        // Upper Cholesky of the Hankel moment matrix, then U^{-1}.
        let mut u = vec![vec![T::zero(); n]; n];
        for j in 0..n {
            for i in 0..=j {
                let mut value = moments[i + j + parity];
                for row in u.iter().take(i) {
                    value -= row[i] * row[j];
                }
                if i == j {
                    require(value > T::zero() && value.is_finite(), "bilinear moment matrix is not positive definite; increase precision or provide a basis")?;
                    u[i][j] = value.sqrt();
                } else {
                    u[i][j] = value / u[i][i];
                }
            }
        }
        for j in 0..n {
            let mut poly = vec![T::zero(); j + 1];
            for i in (0..=j).rev() {
                let mut v = if i == j { T::one() } else { T::zero() };
                for k in i + 1..=j {
                    v -= u[i][k] * poly[k];
                }
                poly[i] = v / u[i][i];
            }
            result[parity].push(poly);
        }
    }
    Ok(result)
}
