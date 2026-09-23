//! Statistics from uniquely owned coordinates, after coupled operator
//! contributions have been reduced. Shared/border coordinates must be counted
//! by one owner only; these summaries do not perform sparse operator exchange.
use crate::algebra::*;
use rayon::prelude::*;

#[derive(Clone, Copy, Debug)]
pub(crate) struct ResidualProducts<T> {
    pub qx: T,
    pub bz: T,
    pub sz: T,
    pub xpx: T,
}

impl<T: FloatT> ResidualProducts<T> {
    pub fn zero() -> Self {
        Self {
            qx: T::zero(),
            bz: T::zero(),
            sz: T::zero(),
            xpx: T::zero(),
        }
    }
}

/// Borrowed counted coordinates. Indexed views skip replicated boundary rows
/// entirely, rather than multiplying their possibly nonfinite values by zero.
#[derive(Clone, Copy)]
pub(crate) struct NormView<'a, T> {
    pub values: &'a [T],
    pub scales: &'a [T],
    pub indices: Option<&'a [usize]>,
}
impl<'a, T: FloatT> NormView<'a, T> {
    pub fn dense(values: &'a [T], scales: &'a [T]) -> Self {
        Self {
            values,
            scales,
            indices: None,
        }
    }
    fn len(&self) -> usize {
        self.indices.map_or(self.values.len(), |ids| ids.len())
    }
    fn scan(&self) -> ScaledNorm<T> {
        assert_eq!(self.values.len(), self.scales.len());
        if let Some(ids) = self.indices {
            ScaledNorm::from_iter(ids.iter().map(|&i| self.values[i] * self.scales[i]))
        } else {
            ScaledNorm::from_iter(self.values.iter().zip(self.scales).map(|(&v, &d)| v * d))
        }
    }
}

/// Norm order: x, z, s, rx_inf, Px, rz_inf, rz, rx. Each view holds only the
/// coordinates counted by this owner and their matching equilibration scales.
pub(crate) struct ResidualOwner<'a, T> {
    pub products: ResidualProducts<T>,
    pub norms: [NormView<'a, T>; 8],
    pub dual_componentwise: Option<T>,
}

/// A compact reduction payload; no global or per-owner numeric vectors.
/// The identity one-owner case performs exactly the original scalar operations.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ResidualSummary<T> {
    pub products: ResidualProducts<T>,
    norms: [ScaledNorm<T>; 8],
    pub dual_componentwise: Option<T>,
    primal_count: usize,
    row_count: usize,
}

impl<T: FloatT> ResidualSummary<T> {
    /// Identity summary used by a rank with no local numeric owner.  Shared
    /// products and norms are supplied by the enclosing collective reduction.
    pub(crate) fn empty(dual_componentwise: Option<T>) -> Self {
        let empty = ScaledNorm::from_iter(std::iter::empty::<T>());
        Self {
            products: ResidualProducts::zero(),
            norms: [empty; 8],
            dual_componentwise,
            primal_count: 0,
            row_count: 0,
        }
    }

    pub(crate) fn replace_global(
        &mut self,
        products: ResidualProducts<T>,
        norms: [T; 8],
        dual_componentwise: Option<T>,
    ) {
        self.products = products;
        self.norms = norms.map(ScaledNorm::from_norm);
        self.dual_componentwise = dual_componentwise;
    }

    /// Add the replicated equality/border coordinates when the canonical
    /// owner has no local numeric block.  A rank with no block still owns the
    /// shared boundary scalars, and those coordinates must participate in the
    /// same products and norms as a regular owner.  The caller invokes this
    /// only on the canonical rank so the subsequent collective does not count
    /// the replicated values once per rank.
    pub(crate) fn add_border(
        &mut self,
        products: ResidualProducts<T>,
        border_z: &[T],
        border_s: &[T],
        border_inf: &[T],
        border_residual: &[T],
        border_e: &[T],
        border_einv: &[T],
    ) {
        let empty: &[T] = &[];
        let owner = ResidualOwner {
            products,
            norms: [
                NormView::dense(empty, empty),
                NormView::dense(border_z, border_e),
                NormView::dense(border_s, border_einv),
                NormView::dense(empty, empty),
                NormView::dense(empty, empty),
                NormView::dense(border_inf, border_einv),
                NormView::dense(border_residual, border_einv),
                NormView::dense(empty, empty),
            ],
            dual_componentwise: self.dual_componentwise,
        };
        self.merge(Self::local(owner, None))
            .expect("canonical border summary has consistent dimensions");
    }

    fn local(owner: ResidualOwner<'_, T>, pool: Option<&rayon::ThreadPool>) -> Self {
        let primal_count = owner.norms[0].len();
        let row_count = owner.norms[1].len();
        let empty = ScaledNorm::from_iter(std::iter::empty::<T>());
        let mut norms = [empty; 8];
        let scan = |(out, view): (&mut ScaledNorm<T>, NormView<'_, T>)| {
            *out = view.scan();
        };
        // Parallelize independent scans, not their internal accumulation order.
        if let Some(pool) = pool {
            pool.install(|| {
                norms
                    .par_iter_mut()
                    .zip(owner.norms.into_par_iter())
                    .for_each(scan)
            });
        } else {
            norms.iter_mut().zip(owner.norms).for_each(scan);
        }
        Self {
            products: owner.products,
            norms,
            dual_componentwise: owner.dual_componentwise,
            primal_count,
            row_count,
        }
    }

    /// Merge owner summaries in the supplied deterministic order. Partitioned
    /// reductions may round differently from one serial scan; no extra accuracy
    /// tolerance or status promotion is introduced here.
    pub fn from_owners<'a>(
        owners: impl IntoIterator<Item = ResidualOwner<'a, T>>,
        pool: Option<&rayon::ThreadPool>,
    ) -> Result<Self, &'static str> {
        let mut owners = owners.into_iter();
        let first = owners
            .next()
            .ok_or("statistics require an owner (which may be empty)")?;
        let mut out = Self::local(first, pool);
        for owner in owners {
            out.merge(Self::local(owner, pool))?;
        }
        Ok(out)
    }

    fn merge(&mut self, other: Self) -> Result<(), &'static str> {
        let componentwise = match (self.dual_componentwise, other.dual_componentwise) {
            (None, None) => None,
            (Some(a), Some(b)) => Some(if a.is_nan() || b.is_nan() {
                T::nan()
            } else {
                a.max(b)
            }),
            _ => return Err("statistics owners disagree on componentwise feasibility"),
        };
        // An empty owner is an identity, including for signed underflow zero.
        let add = |a, b, left, right| {
            if right == 0 {
                a
            } else if left == 0 {
                b
            } else {
                a + b
            }
        };
        self.products.qx = add(
            self.products.qx,
            other.products.qx,
            self.primal_count,
            other.primal_count,
        );
        self.products.xpx = add(
            self.products.xpx,
            other.products.xpx,
            self.primal_count,
            other.primal_count,
        );
        self.products.bz = add(
            self.products.bz,
            other.products.bz,
            self.row_count,
            other.row_count,
        );
        self.products.sz = add(
            self.products.sz,
            other.products.sz,
            self.row_count,
            other.row_count,
        );
        for (state, rhs) in self.norms.iter_mut().zip(other.norms) {
            *state = state.merge(rhs);
        }
        self.dual_componentwise = componentwise;
        self.primal_count += other.primal_count;
        self.row_count += other.row_count;
        Ok(())
    }

    pub fn norms(&self) -> [T; 8] {
        self.norms.map(|state| state.norm())
    }

    pub(crate) fn norm_parts(&self) -> ([T; 8], [T; 8]) {
        (
            self.norms.map(|state| state.scale()),
            self.norms.map(|state| state.sumsq()),
        )
    }
}

#[cfg(test)]
#[path = "tests/statistics.rs"]
mod tests;
