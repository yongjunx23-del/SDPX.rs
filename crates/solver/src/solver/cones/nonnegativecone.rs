use super::*;
use crate::algebra::*;
use itertools::izip;
use rayon::prelude::*;
use std::iter::zip;

// -------------------------------------
// Nonnegative Cone
// -------------------------------------

pub struct NonnegativeCone<T> {
    dim: usize,
    pub(crate) w: Vec<T>,
    λ: Vec<T>,
}

impl<T> NonnegativeCone<T>
where
    T: FloatT,
{
    // Called only for a single large orthant inside the solver-owned pool.
    // Each chunk owns its destinations and retains the serial scalar ordering.
    pub(super) fn update_scaling_parallel(&mut self, s: &[T], z: &[T], chunk: usize) {
        self.λ
            .par_chunks_mut(chunk)
            .zip(self.w.par_chunks_mut(chunk))
            .zip(s.par_chunks(chunk))
            .zip(z.par_chunks(chunk))
            .for_each(|(((lambda, w), s), z)| {
                for (lambda, w, s, z) in izip!(lambda, w, s, z) {
                    *lambda = T::sqrt(*s * *z);
                    *w = T::sqrt(*s / *z);
                }
            });
    }

    pub(super) fn mul_Hs_parallel(&self, y: &mut [T], x: &[T], chunk: usize) {
        y.par_chunks_mut(chunk)
            .zip(self.w.par_chunks(chunk))
            .zip(x.par_chunks(chunk))
            .for_each(|((y, w), x)| {
                for (y, w, x) in izip!(y, w, x) {
                    *y = *w * (*w * *x);
                }
            });
    }

    pub(super) fn combined_ds_shift_parallel(
        &self,
        shift: &mut [T],
        step_z: &mut [T],
        step_s: &mut [T],
        sigma_mu: T,
        chunk: usize,
    ) {
        shift
            .par_chunks_mut(chunk)
            .zip(step_z.par_chunks_mut(chunk))
            .zip(step_s.par_chunks_mut(chunk))
            .zip(self.w.par_chunks(chunk))
            .for_each(|(((shift, step_z), step_s), w)| {
                for (shift, step_z, step_s, w) in izip!(shift, step_z, step_s, w) {
                    // Match mul_W/mul_Winv, including their alpha/beta terms,
                    // followed by circ_op and scaled_unit_shift. No reduction.
                    *step_z = T::one() * (*step_z * *w) + T::zero() * *step_z;
                    *step_s = T::one() * (*step_s / *w) + T::zero() * *step_s;
                    *shift = *step_s * *step_z;
                    *shift += -sigma_mu;
                }
            });
    }

    pub(super) fn offset_parallel(&self, out: &mut [T], ds: &[T], z: &[T], chunk: usize) {
        out.par_chunks_mut(chunk)
            .zip(ds.par_chunks(chunk))
            .zip(z.par_chunks(chunk))
            .for_each(|((out, ds), z)| {
                for (out, ds, z) in izip!(out, ds, z) {
                    *out = *ds / *z;
                }
            });
    }

    // Chunked step-to-boundary for the single-large-orthant path. Each
    // chunk returns its own (αz, αs) minimum; folding chunk partials in
    // index order is bitwise identical to the serial scan because min is
    // associative and commutative.
    pub(super) fn step_length_parallel(
        &self,
        dz: &[T],
        ds: &[T],
        z: &[T],
        s: &[T],
        chunk: usize,
        αmax: T,
    ) -> (T, T) {
        let bounds: Vec<(T, T)> = dz
            .par_chunks(chunk)
            .zip(ds.par_chunks(chunk))
            .zip(z.par_chunks(chunk))
            .zip(s.par_chunks(chunk))
            .map(|(((dz, ds), z), s)| {
                let mut αz = αmax;
                let mut αs = αmax;
                for (dz, ds, z, s) in izip!(dz, ds, z, s) {
                    if *dz < T::zero() {
                        αz = T::min(αz, -*z / *dz);
                    }
                    if *ds < T::zero() {
                        αs = T::min(αs, -*s / *ds);
                    }
                }
                (αz, αs)
            })
            .collect();
        let mut αz = αmax;
        let mut αs = αmax;
        for (bz, bs) in bounds {
            αz = T::min(αz, bz);
            αs = T::min(αs, bs);
        }
        // The caller's fold returns one shared α = min(αz, αs) for both
        // components; match that contract exactly.
        let α = T::min(αz, αs);
        (α, α)
    }

    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            w: vec![T::zero(); dim],
            λ: vec![T::zero(); dim],
        }
    }
}

impl<T> Cone<T> for NonnegativeCone<T>
where
    T: FloatT,
{
    fn degree(&self) -> usize {
        self.dim
    }

    fn numel(&self) -> usize {
        self.dim
    }

    fn is_symmetric(&self) -> bool {
        true
    }

    fn is_sparse_expandable(&self) -> bool {
        false
    }

    fn allows_primal_dual_scaling(&self) -> bool {
        true
    }

    fn rectify_equilibration(&self, δ: &mut [T], _e: &[T]) -> bool {
        δ.set(T::one());
        false
    }

    fn margins(&mut self, z: &mut [T], _pd: PrimalOrDualCone) -> (T, T) {
        let α = z.minimum();
        let β = z.iter().fold(T::zero(), |β, &zi| β + T::max(zi, T::zero()));
        (α, β)
    }

    fn scaled_unit_shift(&self, z: &mut [T], α: T, _pd: PrimalOrDualCone) {
        z.translate(α);
    }

    fn unit_initialization(&self, z: &mut [T], s: &mut [T]) {
        z.fill(T::one());
        s.fill(T::one());
    }

    fn set_identity_scaling(&mut self) {
        self.w.fill(T::one());
    }

    fn update_scaling(
        &mut self,
        s: &[T],
        z: &[T],
        _μ: T,
        _scaling_strategy: ScalingStrategy,
    ) -> bool {
        for (λ, w, s, z) in izip!(&mut self.λ, &mut self.w, s, z) {
            *λ = T::sqrt((*s) * (*z));
            *w = T::sqrt((*s) / (*z));
        }

        true
    }

    fn Hs_is_diagonal(&self) -> bool {
        true
    }

    fn get_Hs(&self, Hsblock: &mut [T]) {
        assert_eq!(self.w.len(), Hsblock.len());
        for (blki, &wi) in zip(Hsblock, &self.w) {
            *blki = wi * wi;
        }
    }

    fn mul_Hs(&mut self, y: &mut [T], x: &[T], _work: &mut [T]) {
        //NB : seemingly sensitive to order of multiplication
        for (yi, (&wi, &xi)) in y.iter_mut().zip(self.w.iter().zip(x)) {
            *yi = wi * (wi * xi)
        }
    }

    fn affine_ds(&self, ds: &mut [T], _s: &[T]) {
        assert_eq!(self.λ.len(), ds.len());
        for (dsi, &λi) in zip(ds, &self.λ) {
            *dsi = λi * λi;
        }
    }

    fn combined_ds_shift(&mut self, dz: &mut [T], step_z: &mut [T], step_s: &mut [T], σμ: T) {
        //PJG: could be done faster for nonnegatives?
        self._combined_ds_shift_symmetric(dz, step_z, step_s, σμ);
    }

    fn Δs_from_Δz_offset(&mut self, out: &mut [T], ds: &[T], _work: &mut [T], z: &[T]) {
        for (outi, (&dsi, &zi)) in zip(out, zip(ds, z)) {
            *outi = dsi / zi;
        }
    }

    fn step_length(
        &mut self,
        dz: &[T],
        ds: &[T],
        z: &[T],
        s: &[T],
        _settings: &CoreSettings<T>,
        αmax: T,
    ) -> (T, T) {
        assert_eq!(z.len(), s.len());
        assert_eq!(dz.len(), z.len());
        assert_eq!(ds.len(), s.len());

        let mut αz = αmax;
        let mut αs = αmax;

        for i in 0..z.len() {
            if dz[i] < T::zero() {
                αz = T::min(αz, -z[i] / dz[i]);
            }
            if ds[i] < T::zero() {
                αs = T::min(αs, -s[i] / ds[i]);
            }
        }
        (αz, αs)
    }

    fn compute_barrier(&mut self, z: &[T], s: &[T], dz: &[T], ds: &[T], α: T) -> T {
        assert_eq!(z.len(), s.len());
        assert_eq!(dz.len(), z.len());
        assert_eq!(ds.len(), s.len());
        let mut barrier = T::zero();
        for (&s, &ds, &z, &dz) in izip!(s, ds, z, dz) {
            let si = s + α * ds;
            let zi = z + α * dz;
            barrier -= (si * zi).logsafe();
        }
        barrier
    }
}

// ---------------------------------------------
// operations supported by symmetric cones only
// ---------------------------------------------

impl<T> SymmetricCone<T> for NonnegativeCone<T>
where
    T: FloatT,
{
    fn λ_inv_circ_op(&mut self, x: &mut [T], z: &[T]) {
        _inv_circ_op(x, &self.λ, z);
    }

    fn mul_W(&mut self, _is_transpose: MatrixShape, y: &mut [T], x: &[T], α: T, β: T) {
        assert_eq!(y.len(), x.len());
        assert_eq!(y.len(), self.w.len());
        for i in 0..y.len() {
            y[i] = α * (x[i] * self.w[i]) + β * y[i];
        }
    }

    fn mul_Winv(&mut self, _is_transpose: MatrixShape, y: &mut [T], x: &[T], α: T, β: T) {
        assert_eq!(y.len(), x.len());
        assert_eq!(y.len(), self.w.len());
        for i in 0..y.len() {
            y[i] = α * (x[i] / self.w[i]) + β * y[i];
        }
    }
}

// ---------------------------------------------
// Jordan algebra operations for symmetric cones
// ---------------------------------------------

impl<T> JordanAlgebra<T> for NonnegativeCone<T>
where
    T: FloatT,
{
    fn circ_op(&mut self, x: &mut [T], y: &[T], z: &[T]) {
        _circ_op(x, y, z);
    }

}

// circ ops don't use self for this cone, so put the actual
// implementations outside so that they can be called by
// other functions with entering borrow check hell
fn _circ_op<T>(x: &mut [T], y: &[T], z: &[T])
where
    T: FloatT,
{
    for (x, (y, z)) in zip(x, zip(y, z)) {
        *x = (*y) * (*z);
    }
}

fn _inv_circ_op<T>(x: &mut [T], y: &[T], z: &[T])
where
    T: FloatT,
{
    for (x, (y, z)) in zip(x, zip(y, z)) {
        *x = (*z) / (*y);
    }
}
