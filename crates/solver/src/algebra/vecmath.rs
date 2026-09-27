use super::{FloatT, ScalarMath, VectorMath};
use itertools::izip;
use rayon::prelude::*;
use std::borrow::Borrow;
use std::cell::RefCell;
use std::iter::zip;
use std::sync::Arc;

// Long vector operations on the solver's main thread run in its worker pool
// (a longer threshold for native floats). Only elementwise operations and order-free reductions
// (max, all) are split, so results are bitwise identical to the serial
// loops. Calls from pool workers stay serial.
thread_local! {
    static VECTOR_POOL: RefCell<Option<Arc<rayon::ThreadPool>>> = const { RefCell::new(None) };
}
const VECTOR_CHUNK: usize = 2048;
/// Minimum length worth a pool dispatch: MPFR elementwise operations cost
/// tens of ns each, binary64/32 about one, so native floats need ~16x more.
const VECTOR_MIN_LEN: usize = 4096;
const VECTOR_MIN_LEN_NATIVE: usize = 1 << 16;

/// Registers a pool for this thread's long vector operations until dropped.
pub(crate) struct VectorPoolGuard(Option<Arc<rayon::ThreadPool>>);

impl VectorPoolGuard {
    pub(crate) fn install(pool: Option<Arc<rayon::ThreadPool>>) -> Self {
        let pool = pool.filter(|p| p.current_num_threads() > 1);
        Self(VECTOR_POOL.with(|slot| std::mem::replace(&mut *slot.borrow_mut(), pool)))
    }
}

impl Drop for VectorPoolGuard {
    fn drop(&mut self) {
        let previous = self.0.take();
        VECTOR_POOL.with(|slot| *slot.borrow_mut() = previous);
    }
}

fn vector_pool<T: FloatT>(len: usize) -> Option<Arc<rayon::ThreadPool>> {
    let min = if T::precision_bits() > 64 {
        VECTOR_MIN_LEN
    } else {
        VECTOR_MIN_LEN_NATIVE
    };
    if len < min || rayon::current_thread_index().is_some() {
        return None;
    }
    VECTOR_POOL.with(|slot| slot.borrow().clone())
}

fn par_update<T: FloatT>(v: &mut [T], f: impl Fn(&mut T) + Sync + Send) -> bool {
    let Some(pool) = vector_pool::<T>(v.len()) else {
        return false;
    };
    pool.install(|| {
        v.par_chunks_mut(VECTOR_CHUNK)
            .for_each(|chunk| chunk.iter_mut().for_each(&f))
    });
    true
}

fn par_update_with<T: FloatT>(v: &mut [T], x: &[T], f: impl Fn(&mut T, T) + Sync + Send) -> bool {
    let Some(pool) = vector_pool::<T>(v.len()) else {
        return false;
    };
    pool.install(|| {
        v.par_chunks_mut(VECTOR_CHUNK)
            .zip(x.par_chunks(VECTOR_CHUNK))
            .for_each(|(chunk, x)| zip(chunk, x).for_each(|(v, &x)| f(v, x)))
    });
    true
}

/// `y[i] += x[i]`, pooled like the elementwise `VectorMath` operations.
pub(crate) fn add_assign<T: FloatT>(y: &mut [T], x: &[T]) {
    assert_eq!(y.len(), x.len());
    if !par_update_with(y, x, |y, x| *y += x) {
        zip(y, x).for_each(|(y, &x)| *y += x);
    }
}

impl<T: FloatT> VectorMath<T> for [T] {
    fn copy_from(&mut self, src: &[T]) -> &mut Self {
        self.copy_from_slice(src);
        self
    }

    fn select(&self, index: &[bool]) -> Vec<T> {
        assert_eq!(self.len(), index.len());
        zip(self, index)
            .filter(|(_x, &b)| b)
            .map(|(&x, _b)| x)
            .collect()
    }

    fn scalarop(&mut self, op: impl Fn(T) -> T) -> &mut Self {
        for x in &mut *self {
            *x = op(*x);
        }
        self
    }

    fn scalarop_from(&mut self, op: impl Fn(T) -> T, v: &[T]) -> &mut Self {
        for (x, v) in zip(&mut *self, v) {
            *x = op(*v);
        }
        self
    }

    fn translate(&mut self, c: T) -> &mut Self {
        //NB: translate is a scalar shift of all variables and is
        //used only in the NN cone to force vectors into R^n_+
        if par_update(self, |x| *x = *x + c) {
            return self;
        }
        self.scalarop(|x| x + c)
    }

    fn set(&mut self, c: T) -> &mut Self {
        self.fill(c);
        self
    }

    fn scale(&mut self, c: T) -> &mut Self {
        if par_update(self, |x| *x = *x * c) {
            return self;
        }
        self.scalarop(|x| x * c)
    }

    fn recip(&mut self) -> &mut Self {
        if par_update(self, |x| *x = T::recip(*x)) {
            return self;
        }
        self.scalarop(T::recip)
    }

    fn sqrt(&mut self) -> &mut Self {
        if par_update(self, |x| *x = T::sqrt(*x)) {
            return self;
        }
        self.scalarop(T::sqrt)
    }

    fn rsqrt(&mut self) -> &mut Self {
        if par_update(self, |x| *x = T::recip(T::sqrt(*x))) {
            return self;
        }
        self.scalarop(|x| T::recip(T::sqrt(x)))
    }

    fn negate(&mut self) -> &mut Self {
        if par_update(self, |x| *x = -*x) {
            return self;
        }
        self.scalarop(|x| -x)
    }

    fn hadamard(&mut self, y: &[T]) -> &mut Self {
        assert_eq!(self.len(), y.len());
        if par_update_with(self, y, |x, y| *x *= y) {
            return self;
        }
        zip(&mut *self, y).for_each(|(x, y)| *x *= *y);
        self
    }

    fn clip(&mut self, min_thresh: T, max_thresh: T) -> &mut Self {
        if par_update(self, |x| *x = x.clip(min_thresh, max_thresh)) {
            return self;
        }
        self.scalarop(|x| x.clip(min_thresh, max_thresh))
    }

    fn normalize(&mut self) -> T {
        let norm = self.norm();
        if norm.is_zero() {
            return T::zero();
        }
        self.scale(norm.recip());
        norm
    }

    fn dot(&self, y: &[T]) -> T {
        zip(self, y).fold(T::zero(), |acc, (&x, &y)| x.mul_add(y, acc))
    }

    fn dot_shifted(z: &[T], s: &[T], dz: &[T], ds: &[T], α: T) -> T {
        assert_eq!(z.len(), s.len());
        assert_eq!(z.len(), dz.len());
        assert_eq!(s.len(), ds.len());

        let mut out = T::zero();
        for (&s, &ds, &z, &dz) in izip!(s, ds, z, dz) {
            let si = ds.mul_add(α, s);
            let zi = dz.mul_add(α, z);
            out = si.mul_add(zi, out);
        }
        out
    }

    fn dist(&self, y: &Self) -> T {
        let dist2 = zip(self, y).fold(T::zero(), |acc, (&x, &y)| acc + T::powi(x - y, 2));
        T::sqrt(dist2)
    }

    fn sum(&self) -> T {
        self.iter().fold(T::zero(), |acc, &x| acc + x)
    }

    fn sumsq(&self) -> T {
        self.dot(self)
    }

    // 2-norm
    fn norm(&self) -> T {
        // T::sqrt(self.sumsq()) // not robust
        stable_norm(self.iter())
    }

    //2-norm of elementwise product self.*v
    fn norm_scaled(&self, v: &[T]) -> T {
        assert_eq!(self.len(), v.len());
        stable_norm(izip!(self, v).map(|(&yi, &vi)| yi * vi))
    }

    // 2-norm of (self + α.dz)
    fn norm_shifted(&self, dz: &[T], α: T) -> T {
        stable_norm(izip!(self, dz).map(|(&zi, &dzi)| zi + α * dzi))
    }

    // Returns infinity norm
    fn norm_inf(&self) -> T {
        fn serial<T: FloatT>(v: &[T]) -> T {
            let mut out = T::zero();
            for &v in v {
                if v.is_nan() {
                    return T::nan();
                }
                out = T::max(out, v.abs());
            }
            out
        }
        // A max is exact, so chunk maxima combine to the serial result; any
        // NaN chunk makes the whole result NaN, as in the serial scan.
        if let Some(pool) = vector_pool::<T>(self.len()) {
            let parts: Vec<T> =
                pool.install(|| self.par_chunks(VECTOR_CHUNK).map(serial).collect());
            if parts.iter().any(|v| v.is_nan()) {
                return T::nan();
            }
            return parts.into_iter().fold(T::zero(), T::max);
        }
        serial(self)
    }

    // Returns one norm
    fn norm_one(&self) -> T {
        self.iter().fold(T::zero(), |acc, v| acc + v.abs())
    }

    //inf-norm of elementwise product self.*v
    fn norm_inf_scaled(&self, v: &Self) -> T {
        assert_eq!(self.len(), v.len());
        zip(self, v).fold(T::zero(), |acc, (&x, &y)| T::max(acc, T::abs(x * y)))
    }

    //
    fn norm_one_scaled(&self, v: &Self) -> T {
        zip(self, v).fold(T::zero(), |acc, (&x, &y)| acc + T::abs(x * y))
    }

    // max absolute difference (used for unit testing)
    fn norm_inf_diff(&self, b: &[T]) -> T {
        zip(self, b).fold(T::zero(), |acc, (x, y)| T::max(acc, T::abs(*x - *y)))
    }

    fn minimum(&self) -> T {
        self.iter().fold(T::infinity(), |r, &s| T::min(r, s))
    }

    fn maximum(&self) -> T {
        self.iter().fold(-T::infinity(), |r, &s| T::max(r, s))
    }

    fn mean(&self) -> T {
        let mean = if self.is_empty() {
            T::zero()
        } else {
            let num = self.iter().fold(T::zero(), |r, &s| r + s);
            let den = T::from_usize(self.len()).unwrap();
            num / den
        };
        mean
    }

    fn is_finite(&self) -> bool {
        if let Some(pool) = vector_pool::<T>(self.len()) {
            return pool.install(|| {
                self.par_chunks(VECTOR_CHUNK)
                    .all(|chunk| chunk.iter().all(|&x| T::is_finite(x)))
            });
        }
        self.iter().all(|&x| T::is_finite(x))
    }

    fn axpby(&mut self, a: T, x: &[T], b: T) -> &mut Self {
        assert_eq!(self.len(), x.len());
        if par_update_with(self, x, |y, x| *y = a * x + b * (*y)) {
            return self;
        }

        zip(&mut *self, x).for_each(|(y, x)| *y = a * (*x) + b * (*y));
        self
    }

    fn waxpby(&mut self, a: T, x: &[T], b: T, y: &[T]) -> &mut Self {
        assert_eq!(self.len(), x.len());
        assert_eq!(self.len(), y.len());
        if let Some(pool) = vector_pool::<T>(self.len()) {
            pool.install(|| {
                self.par_chunks_mut(VECTOR_CHUNK)
                    .zip(x.par_chunks(VECTOR_CHUNK))
                    .zip(y.par_chunks(VECTOR_CHUNK))
                    .for_each(|((w, x), y)| {
                        for (w, (x, y)) in zip(w, zip(x, y)) {
                            *w = a * (*x) + b * (*y);
                        }
                    })
            });
            return self;
        }

        for (w, (x, y)) in zip(&mut *self, zip(x, y)) {
            *w = a * (*x) + b * (*y);
        }
        self
    }
}

/// Owned scaled sum of squares. Serial construction preserves the vector norm's
/// historical operation order; merging partitions may round differently.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ScaledNorm<T> {
    scale: T,
    sumsq: T,
}

impl<T: FloatT> ScaledNorm<T> {
    pub(crate) fn scale(&self) -> T {
        self.scale
    }

    pub(crate) fn sumsq(&self) -> T {
        self.sumsq
    }

    pub(crate) fn from_norm(norm: T) -> Self {
        Self {
            scale: norm,
            sumsq: T::one(),
        }
    }

    /// Norm from the exactly accumulated sum of squares, rounded once, then
    /// one square root. For high-precision types only: their exponent range
    /// makes the overflow-guarding scaled recurrence (a division per entry)
    /// unnecessary, and this is at least as accurate.
    pub(crate) fn from_exact_squares(x: impl Iterator<Item = T>) -> Self {
        let values: Vec<T> = x.collect();
        let sumsq = T::dot_fma(values.iter().map(|v| (v, v)));
        if sumsq.is_nan() {
            // Same poison form as the scaled recurrence.
            return Self {
                scale: T::zero(),
                sumsq: T::nan(),
            };
        }
        Self::from_norm(sumsq.sqrt())
    }

    pub(crate) fn from_iter<I, B>(x: I) -> Self
    where
        I: Iterator<Item = B>,
        B: Borrow<T>,
    {
        let (scale, sumsq) = x.filter(|b| *b.borrow() != T::zero()).fold(
            (T::zero(), T::one()),
            |(scale, sumsq), b| {
                let xi = *b.borrow();
                let absxi = xi.abs();
                if scale < absxi {
                    let r = scale / absxi;
                    (absxi, T::one() + sumsq * r * r)
                } else {
                    let r = absxi / scale;
                    (scale, sumsq + r * r)
                }
            },
        );
        Self { scale, sumsq }
    }

    pub(crate) fn norm(&self) -> T {
        self.scale * self.sumsq.sqrt()
    }

    pub(crate) fn merge(self, other: Self) -> Self {
        // A NaN input can leave scale at zero: test poison before identity.
        if self.sumsq.is_nan() || other.sumsq.is_nan() {
            return Self {
                scale: T::zero(),
                sumsq: T::nan(),
            };
        }
        if self.scale == T::zero() {
            return other;
        }
        if other.scale == T::zero() {
            return self;
        }
        if self.scale < other.scale {
            let r = self.scale / other.scale;
            Self {
                scale: other.scale,
                sumsq: other.sumsq + self.sumsq * r * r,
            }
        } else {
            let r = other.scale / self.scale;
            Self {
                scale: self.scale,
                sumsq: self.sumsq + other.sumsq * r * r,
            }
        }
    }
}

// Numerically stable 2-norm without changing serial evaluation order.
fn stable_norm<T, I, B>(x: I) -> T
where
    T: FloatT,
    I: Iterator<Item = B>,
    B: Borrow<T>,
{
    ScaledNorm::from_iter(x).norm()
}

#[cfg(test)]
#[path = "scaled_norm_tests.rs"]
mod scaled_norm_tests;
