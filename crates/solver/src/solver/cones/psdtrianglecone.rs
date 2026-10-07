use super::*;
use crate::algebra::*;

// ------------------------------------
// Positive Semidefinite Cone (Scaled triangular form)
// ------------------------------------

pub struct PSDConeData<T> {
    λ: Vec<T>,
    Λisqrt: Vec<T>,
    pub(crate) R: Matrix<T>,
    Rinv: Matrix<T>,
    // G = R R^T, with an authoritative upper triangle. The p-by-p
    // Hessian is generated only into an augmented KKT caller's packed block.
    G: Matrix<T>,
}

impl<T> PSDConeData<T>
where
    T: FloatT,
{
    pub fn new(n: usize) -> Self {
        Self {
            λ: vec![T::zero(); n],
            Λisqrt: vec![T::zero(); n],
            R: Matrix::zeros((n, n)),
            Rinv: Matrix::zeros((n, n)),
            G: Matrix::zeros((n, n)),
        }
    }
}

// Scratch of one cone call. Every buffer is written before it is read within
// a call, and only R, Rinv, G and λ persist between calls, so each worker
// thread keeps one set for all the cones it serves instead of one per cone.
struct PsdWork<T> {
    n: usize,
    chol1: CholeskyEngine<T>,
    // Larger second factors use mat2 until L2' * L1 consumes them.
    chol2: CholeskyEngine<T>,
    // Scaling consumes Vt before other cone calls reuse it as their third
    // matrix; the next factorization overwrites every entry.
    svd: SVDEngine<T>,
    eig: EigEngine<T>,
    mat1: Matrix<T>,
    mat2: Matrix<T>,
    vector: Vec<T>,
    // Allocated only when dz/ds bounds actually run on two workers.
    second: Option<Box<StepWorkspace<T>>>,
}

struct StepWorkspace<T> {
    eig: EigEngine<T>,
    mat1: Matrix<T>,
    mat2: Matrix<T>,
    mat3: Matrix<T>,
    vector: Vec<T>,
}
impl<T: FloatT> StepWorkspace<T> {
    fn new(n: usize) -> Self {
        Self {
            eig: EigEngine::new(n),
            mat1: Matrix::zeros((n, n)),
            mat2: Matrix::zeros((n, n)),
            mat3: Matrix::zeros((n, n)),
            vector: vec![T::zero(); triangular_number(n)],
        }
    }
}

impl<T: FloatT> PsdWork<T> {
    fn new(n: usize) -> Self {
        Self {
            n,
            chol1: CholeskyEngine::new(n),
            chol2: CholeskyEngine::new(if n <= 3 { n } else { 0 }),
            svd: SVDEngine::new_right((n, n)),
            eig: EigEngine::new(n),
            mat1: Matrix::zeros((n, n)),
            mat2: Matrix::zeros((n, n)),
            vector: vec![T::zero(); triangular_number(n)],
            second: None,
        }
    }
}

/// Run `f` with this thread's scratch for a cone of order `n`. A set built for
/// another order is replaced by a fresh one, so the Cholesky factors' upper
/// triangles start at zero exactly as in a newly constructed cone.
fn with_work<T: FloatT, R>(n: usize, f: impl FnOnce(&mut PsdWork<T>) -> R) -> R {
    crate::algebra::scratch::with_scratch(|slot: &mut Option<PsdWork<T>>| {
        if slot.as_ref().map_or(true, |w| w.n != n) {
            *slot = Some(PsdWork::new(n));
        }
        f(slot.as_mut().unwrap())
    })
}

pub struct PSDTriangleCone<T> {
    pub(crate) n: usize,                  // matrix dimension, i.e. matrix is n × n
    numel: usize, // total number of elements in (lower triangle of) the matrix
    pub(crate) data: Box<PSDConeData<T>>, // Boxed so that the PSDCone enum_dispatch variant isn't huge
}

impl<T> PSDTriangleCone<T>
where
    T: FloatT,
{
    pub fn new(n: usize) -> Self {
        // n >= 0 guaranteed by type limit
        Self {
            n,
            numel: triangular_number(n),
            data: Box::new(PSDConeData::<T>::new(n)),
        }
    }

    pub(crate) fn scaling_R(&self) -> &Matrix<T> {
        &self.data.R
    }

    pub(crate) fn scaling_Rinv(&self) -> &Matrix<T> {
        &self.data.Rinv
    }

    /// Number of `T` values the scaling state exchange carries per cone:
    /// R and Rinv in full (n² each) plus λ. Λisqrt and G are derived from
    /// them locally on receipt, bitwise identical to the owner's values.
    pub(crate) fn scaling_state_len(&self) -> usize {
        2 * self.n * self.n + self.n
    }

    /// Append this cone's scaling state to `out` in `[R, Rinv, λ]` order.
    /// Called on the owning rank only, after `update_scaling`.
    /// Cached R·Rᵀ; only the upper triangle is authoritative.
    pub(crate) fn scaling_gram(&self) -> &Matrix<T> {
        &self.data.G
    }

    pub(crate) fn pack_scaling_state(&self, out: &mut Vec<T>) {
        out.extend_from_slice(self.data.R.data());
        out.extend_from_slice(self.data.Rinv.data());
        out.extend_from_slice(&self.data.λ);
    }

    /// Install a scaling state produced by another rank's `update_scaling`,
    /// then recompute the derived caches exactly as `update_scaling` does:
    /// Λisqrt = sqrt(λ)⁻¹ and G = R·Rᵀ (authoritative upper triangle) are
    /// deterministic functions of the packed values, so the result is
    /// bitwise identical to running the update locally.
    pub(crate) fn unpack_scaling_state(&mut self, src: &[T]) {
        let n2 = self.n * self.n;
        debug_assert_eq!(src.len(), self.scaling_state_len());
        let f = &mut self.data;
        f.R.data_mut().copy_from_slice(&src[..n2]);
        f.Rinv.data_mut().copy_from_slice(&src[n2..2 * n2]);
        f.λ.copy_from_slice(&src[2 * n2..]);
        f.Λisqrt.copy_from(&f.λ).sqrt().recip();
        f.G.data_mut().set(T::zero());
        f.G.syrk(&f.R, T::one(), T::zero(), MatrixTriangle::Triu);
    }
}

impl<T> Cone<T> for PSDTriangleCone<T>
where
    T: FloatT,
{
    fn degree(&self) -> usize {
        self.n
    }

    fn numel(&self) -> usize {
        self.numel
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

    fn rectify_equilibration(&self, δ: &mut [T], e: &[T]) -> bool {
        δ.copy_from(e).recip().scale(e.mean());
        true // scalar equilibration
    }

    // functions relating to unit vectors and cone initialization
    fn margins(&mut self, z: &mut [T], _pd: PrimalOrDualCone) -> (T, T) {
        if z.is_empty() {
            return (T::max_value(), T::zero());
        }
        with_work(self.n, |w: &mut PsdWork<T>| {
            svec_to_mat(&mut w.mat1, z);
            let result = w.eig.eigvals(&mut w.mat1, &mut w.mat2.data);
            w.mat2.data.resize(self.n * self.n, T::zero());
            result.expect("Eigval error");
            let e = &w.eig.λ;
            let α = e.minimum();
            let β = e.iter().fold(T::zero(), |s, x| s + T::max(*x, T::zero())); //= sum(e[e.>0])
            (α, β)
        })
    }

    fn scaled_unit_shift(&self, z: &mut [T], α: T, _pd: PrimalOrDualCone) {
        //adds αI to the vectorized triangle,
        //at elements [1,3,6....n(n+1)/2]
        for k in 0..self.n {
            z[triangular_index(k)] += α
        }
    }

    fn unit_initialization(&self, z: &mut [T], s: &mut [T]) {
        s.fill(T::zero());
        z.fill(T::zero());
        self.scaled_unit_shift(s, T::one(), PrimalOrDualCone::PrimalCone);
        self.scaled_unit_shift(z, T::one(), PrimalOrDualCone::DualCone);
    }

    fn set_identity_scaling(&mut self) {
        self.data.R.set_identity();
        self.data.Rinv.set_identity();
        self.data.G.set_identity();
    }

    fn update_scaling(
        &mut self,
        s: &[T],
        z: &[T],
        _μ: T,
        _scaling_strategy: ScalingStrategy,
    ) -> bool {
        if s.is_empty() {
            //bail early on zero length cone
            return true;
        }

        let n = self.n;
        let f = &mut *self.data;
        with_work(n, |w: &mut PsdWork<T>| {
            let (S, Z) = (&mut w.mat1, &mut w.mat2);
            svec_to_mat(S, s);
            if n <= 3 {
                svec_to_mat(Z, z);
            } else {
                let scale = T::FRAC_1_SQRT_2();
                let mut idx = 0;
                for j in 0..n {
                    Z.col_slice_mut(j)[..j].fill(T::zero());
                    for i in 0..=j {
                        Z[(j, i)] = if i == j { z[idx] } else { z[idx] * scale };
                        idx += 1;
                    }
                }
            }

            //compute Cholesky factors. The S and Z factorizations are
            //independent, so offer the second to an idle ambient worker.
            let (ch1, ch2) = (&mut w.chol1, &mut w.chol2);
            let mut factor_z = || {
                if n <= 3 {
                    ch2.factor(Z)
                } else {
                    let n = n.try_into().unwrap();
                    let mut info = 0;
                    T::xpotrf(b'L', n, Z.data_mut(), n, &mut info);
                    if info != 0 {
                        Err(DenseFactorizationError::Cholesky(info))
                    } else {
                        Ok(())
                    }
                }
            };
            let (c1, c2) = if sdpx_arithmetic::inner_parallel::paired() {
                rayon::join(|| ch1.factor(S), factor_z)
            } else {
                (ch1.factor(S), factor_z())
            };

            // bail if the cholesky factorization fails
            // PJG: Need proper Result return type here
            if c1.is_err() || c2.is_err() {
                return false;
            }

            let L1 = &w.chol1.L;
            let L2 = if n <= 3 { &w.chol2.L } else { &w.mat2 };

            // SVD of L2'*L1,
            let tmp = &mut w.mat1;
            tmp.mul(&L2.t(), L1, T::one(), T::zero());

            // Direct SVD avoids squaring the condition number of L2' * L1.
            // Only V is needed: with L2'L1 = UΛV', R = L1 V Λ^{-1/2} and its
            // exact inverse Rinv = Λ^{-1/2} U' L2' = Λ^{1/2} V' L1^{-1}. Recovering
            // Rinv by one triangular solve replaces the accumulation of U.
            let __ts = std::time::Instant::now();
            // L2 is consumed; its storage can serve as SVD workspace until
            // the inverse construction overwrites it with V.
            let svd_ok = w.svd.factor_right(tmp, &mut w.mat2.data).is_ok();
            w.mat2.data.resize(n * n, T::zero());
            crate::receipt::phase("cone_svd", __ts.elapsed());
            // non-finite or non-converged SVD: report a scaling failure so the
            // solver ends with NumericalError instead of panicking
            if !svd_ok {
                return false;
            }

            // assemble λ (diagonal), R and Rinv.
            f.λ.copy_from(&w.svd.s);
            // Singular values persist in λ; reuse their scratch for both rounded roots.
            w.svd.s.sqrt();
            f.Λisqrt.copy_from(&w.svd.s).recip();

            //f.R = L1*V*Λ^{-1/2} and f.Rinv = Λ^{1/2}*(L1^{-T}*V)' are
            //independent; pair them when inner workers are idle.
            {
                let (R, Rinv, svd, Λi, λ) = (&mut f.R, &mut f.Rinv, &w.svd, &f.Λisqrt, &f.λ);
                let X = &mut w.mat2;
                let n = λ.len();
                let mut build_r = || {
                    R.mul(L1, &svd.Vt.t(), T::one(), T::zero());
                    R.rscale(Λi);
                };
                let mut build_rinv = || {
                    for j in 0..n {
                        for i in 0..n {
                            X[(i, j)] = svd.Vt[(j, i)];
                        }
                    }
                    T::xtrsm_lower(n, L1.data(), X.data_mut(), true);
                    for i in 0..n {
                        let root = svd.s[i];
                        for j in 0..n {
                            Rinv[(i, j)] = X[(j, i)] * root;
                        }
                    }
                };
                if sdpx_arithmetic::inner_parallel::paired() {
                    rayon::join(build_r, build_rinv);
                } else {
                    build_r();
                    build_rinv();
                }
            }

            // Cache only the matrix defining the congruence X -> G X G.
            // Keeping its upper triangle authoritative preserves the original
            // syrk rounding and skron indexing, without a p-by-p cone allocation.
            f.G.data_mut().set(T::zero());
            f.G.syrk(&f.R, T::one(), T::zero(), MatrixTriangle::Triu);

            true //PJG: Should return result, with "?" operators above
        })
    }

    fn Hs_is_diagonal(&self) -> bool {
        false
    }

    fn get_Hs(&self, Hsblock: &mut [T]) {
        skron_packed(Hsblock, &self.data.G.sym_up());
    }

    fn mul_Hs(&mut self, y: &mut [T], x: &[T], work: &mut [T]) {
        // Preserve the factored action; this route needs no packed Hessian.
        self.mul_W(MatrixShape::N, work, x, T::one(), T::zero()); // work = Wx
        self.mul_W(MatrixShape::T, y, work, T::one(), T::zero()); // y = c Wᵀwork = W^TWx
    }

    fn affine_ds(&self, ds: &mut [T], _s: &[T]) {
        ds.set(T::zero());
        for k in 0..self.n {
            ds[triangular_index(k)] = self.data.λ[k] * self.data.λ[k];
        }
    }

    fn combined_ds_shift(&mut self, shift: &mut [T], step_z: &mut [T], step_s: &mut [T], σμ: T) {
        self._combined_ds_shift_symmetric(shift, step_z, step_s, σμ);
    }

    fn Δs_from_Δz_offset(&mut self, out: &mut [T], ds: &[T], work: &mut [T], _z: &[T]) {
        self._Δs_from_Δz_offset_symmetric(out, ds, work);
    }

    fn step_length(
        &mut self,
        dz: &[T],
        ds: &[T],
        _z: &[T],
        _s: &[T],
        _settings: &CoreSettings<T>,
        αmax: T,
    ) -> (T, T) {
        if sdpx_arithmetic::inner_parallel::paired() {
            let (n, f) = (self.n, &*self.data);
            with_work(n, |w: &mut PsdWork<T>| {
                let PsdWork {
                    vector,
                    mat1,
                    mat2,
                    svd,
                    eig,
                    second,
                    ..
                } = w;
                let StepWorkspace {
                    eig: eig2,
                    mat1: mat4,
                    mat2: mat5,
                    mat3: mat6,
                    vector: vector2,
                } = second
                    .get_or_insert_with(|| Box::new(StepWorkspace::new(n)))
                    .as_mut();
                rayon::join(
                    || {
                        step_component_inner(
                            dz,
                            &f.R,
                            &f.Λisqrt,
                            false,
                            αmax,
                            vector,
                            mat1,
                            mat2,
                            &mut svd.Vt,
                            eig,
                        )
                    },
                    || {
                        step_component_inner(
                            ds, &f.Rinv, &f.Λisqrt, true, αmax, vector2, mat4, mat5, mat6, eig2,
                        )
                    },
                )
            })
        } else {
            let αz = self.step_component(dz, false, αmax);
            let αs = self.step_component(ds, true, αmax);
            (αz, αs)
        }
    }

    fn compute_barrier(&mut self, z: &[T], s: &[T], dz: &[T], ds: &[T], α: T) -> T {
        let mut barrier = T::zero();
        barrier -= self.logdet_barrier(z, dz, α);
        barrier -= self.logdet_barrier(s, ds, α);
        barrier
    }
}

impl<T> PSDTriangleCone<T>
where
    T: FloatT,
{
    fn step_component(&mut self, direction: &[T], primal: bool, αmax: T) -> T {
        let f = &*self.data;
        with_work(self.n, |w: &mut PsdWork<T>| {
            step_component_inner(
                direction,
                if primal { &f.Rinv } else { &f.R },
                &f.Λisqrt,
                primal,
                αmax,
                &mut w.vector,
                &mut w.mat1,
                &mut w.mat2,
                &mut w.svd.Vt,
                &mut w.eig,
            )
        })
    }

    pub(crate) fn prepare_affine_bounds(&mut self, dz: &mut [T], ds: &mut [T], αmax: T) -> (T, T) {
        // The dz (dual, R) and ds (primal, Rinv) bounds are independent.
        // With a second scratch set they can run on two ambient workers;
        // without one they fold serially, bitwise identical either way.
        if sdpx_arithmetic::inner_parallel::paired() {
            let (n, f) = (self.n, &*self.data);
            with_work(n, |w: &mut PsdWork<T>| {
                let PsdWork {
                    vector,
                    mat1,
                    mat2,
                    svd,
                    eig,
                    second,
                    ..
                } = w;
                let StepWorkspace {
                    eig: eig2,
                    mat1: mat4,
                    mat2: mat5,
                    mat3: mat6,
                    vector: vector2,
                } = second
                    .get_or_insert_with(|| Box::new(StepWorkspace::new(n)))
                    .as_mut();
                rayon::join(
                    || {
                        let α = step_component_inner(
                            dz,
                            &f.R,
                            &f.Λisqrt,
                            false,
                            αmax,
                            vector,
                            mat1,
                            mat2,
                            &mut svd.Vt,
                            eig,
                        );
                        dz.copy_from_slice(vector);
                        α
                    },
                    || {
                        let α = step_component_inner(
                            ds, &f.Rinv, &f.Λisqrt, true, αmax, vector2, mat4, mat5, mat6, eig2,
                        );
                        ds.copy_from_slice(vector2);
                        α
                    },
                )
            })
        } else {
            let f = &*self.data;
            with_work(self.n, |w: &mut PsdWork<T>| {
                let αz = step_component_inner(
                    dz,
                    &f.R,
                    &f.Λisqrt,
                    false,
                    αmax,
                    &mut w.vector,
                    &mut w.mat1,
                    &mut w.mat2,
                    &mut w.svd.Vt,
                    &mut w.eig,
                );
                dz.copy_from_slice(&w.vector);
                let αs = step_component_inner(
                    ds,
                    &f.Rinv,
                    &f.Λisqrt,
                    true,
                    αmax,
                    &mut w.vector,
                    &mut w.mat1,
                    &mut w.mat2,
                    &mut w.svd.Vt,
                    &mut w.eig,
                );
                ds.copy_from_slice(&w.vector);
                (αz, αs)
            })
        }
    }

    pub(super) fn combined_shift_prepared(&mut self, shift: &mut [T], dz: &[T], ds: &[T], σμ: T) {
        self.circ_op(shift, ds, dz);
        self.scaled_unit_shift(shift, -σμ, PrimalOrDualCone::PrimalCone);
    }

    fn logdet_barrier(&mut self, x: &[T], dx: &[T], α: T) -> T
    where
        T: FloatT,
    {
        with_work(self.n, |w: &mut PsdWork<T>| {
            w.vector.waxpby(T::one(), x, α, dx);
            svec_to_mat(&mut w.mat1, &w.vector);

            match w.chol1.factor(&mut w.mat1) {
                Ok(_) => w.chol1.logdet(),
                Err(_) => T::infinity(),
            }
        })
    }
}

// ---------------------------------------------
// operations supported by symmetric cones only
// ---------------------------------------------

impl<T> SymmetricCone<T> for PSDTriangleCone<T>
where
    T: FloatT,
{
    // implements x = λ \ z for the SDP cone
    fn λ_inv_circ_op(&mut self, x: &mut [T], z: &[T]) {
        let (n, λ) = (self.n, &self.data.λ);
        with_work(n, |w: &mut PsdWork<T>| {
            let X = &mut w.mat1;
            let Z = &mut w.mat2;

            svec_to_mat(Z, z);

            let two: T = (2.).as_T();
            // Z is symmetric and the Jordan inverse uses the same denominator
            // for (i,j) and (j,i). Evaluate each pair once, then mirror it before
            // mat_to_svec performs its usual symmetric packing.
            for j in 0..n {
                for i in 0..=j {
                    let value = (two * Z[(i, j)]) / (λ[i] + λ[j]);
                    X[(i, j)] = value;
                    if i != j {
                        X[(j, i)] = value;
                    }
                }
            }
            mat_to_svec(x, X);
        })
    }

    fn mul_W(&mut self, is_transpose: MatrixShape, y: &mut [T], x: &[T], α: T, β: T) {
        mul_Wx_inner(is_transpose, y, x, α, β, &self.data.R)
    }

    fn mul_Winv(&mut self, is_transpose: MatrixShape, y: &mut [T], x: &[T], α: T, β: T) {
        mul_Wx_inner(is_transpose, y, x, α, β, &self.data.Rinv)
    }
}

fn mul_Wx_inner<T>(is_transpose: MatrixShape, y: &mut [T], x: &[T], α: T, β: T, Rx: &Matrix<T>)
where
    T: FloatT,
{
    with_work(Rx.nrows(), |w: &mut PsdWork<T>| {
        mul_Wx_scratch(
            is_transpose,
            y,
            x,
            α,
            β,
            Rx,
            &mut w.mat1,
            &mut w.mat2,
            &mut w.svd.Vt,
        )
    })
}

#[allow(clippy::too_many_arguments)]
fn mul_Wx_scratch<T>(
    is_transpose: MatrixShape,
    y: &mut [T],
    x: &[T],
    α: T,
    β: T,
    Rx: &Matrix<T>,
    workmat1: &mut Matrix<T>,
    workmat2: &mut Matrix<T>,
    workmat3: &mut Matrix<T>,
) where
    T: FloatT,
{
    let (X, Y, tmp) = (workmat1, workmat2, workmat3);
    svec_to_mat(X, x);

    let __ts = std::time::Instant::now();
    match is_transpose {
        MatrixShape::T => {
            // Y .= α*(R*X*R') + βY        #W^T*x,   or....
            // Y .= α*(Rinv*X*Rinv') + βY  #W^{-T}*x
            if !congruence_exact_sym(Y, Rx, false, X, None, None) {
                tmp.mul(X, &Rx.t(), T::one(), T::zero());
                pooled_gemm_sym(Y, Rx, tmp, None);
            }
        }
        MatrixShape::N => {
            // Y .= α*(R'*X*R) + βY         #W*x
            // Y .= α*(Rinv'*X*Rinv) + βY   #W^{-1}*x
            if !congruence_exact_sym(Y, Rx, true, X, None, None) {
                tmp.mul(&Rx.t(), X, T::one(), T::zero());
                pooled_gemm_sym(Y, tmp, Rx, None);
            }
        }
    }
    crate::receipt::phase("cone_wprod", __ts.elapsed());
    if α == T::one() && β == T::zero() {
        mat_to_svec(y, Y);
    } else {
        let scale = if Y.ncols() > 1 {
            T::FRAC_1_SQRT_2()
        } else {
            T::zero()
        };
        let mut idx = 0;
        for col in 0..Y.ncols() {
            for row in 0..=col {
                let entry = if row == col {
                    Y[(row, col)]
                } else {
                    (Y[(row, col)] + Y[(col, row)]) * scale
                };
                y[idx] = α * entry + β * y[idx];
                idx += 1;
            }
        }
    }
}

// ---------------------------------------------
// Jordan algebra operations for symmetric cones
// ---------------------------------------------

impl<T> JordanAlgebra<T> for PSDTriangleCone<T>
where
    T: FloatT,
{
    fn circ_op(&mut self, x: &mut [T], y: &[T], z: &[T]) {
        with_work(self.n, |w: &mut PsdWork<T>| {
            let (Y, Z, X) = (&mut w.mat1, &mut w.mat2, &mut w.svd.Vt);
            svec_to_mat(Y, y);
            svec_to_mat(Z, z);

            // X .= (Y*Z + Z*Y)/2
            // NB: works b/c Y and Z are both symmetric
            X.data_mut().set(T::zero()); //X.sym_up() will assert is_triu
            X.syr2k(Y, Z, (0.5).as_T(), T::zero());
            mat_to_svec(x, &X.sym_up());
        })
    }
}

//-----------------------------------------
// internal operations for SDP cones
// ----------------------------------------

fn step_component_inner<T>(
    direction: &[T],
    rx: &Matrix<T>,
    Λisqrt: &[T],
    primal: bool,
    αmax: T,
    wv: &mut Vec<T>,
    wm1: &mut Matrix<T>,
    wm2: &mut Matrix<T>,
    wm3: &mut Matrix<T>,
    eig: &mut EigEngine<T>,
) -> T
where
    T: FloatT,
{
    mul_Wx_scratch(
        if primal {
            MatrixShape::T
        } else {
            MatrixShape::N
        },
        wv,
        direction,
        T::one(),
        T::zero(),
        rx,
        wm1,
        wm2,
        wm3,
    );
    step_length_psd_component(wm1, eig, wv, Λisqrt, αmax, &mut wm2.data)
}

fn step_length_psd_component<T>(
    workΔ: &mut Matrix<T>,
    engine: &mut EigEngine<T>,
    d: &[T],
    Λisqrt: &[T],
    αmax: T,
    work: &mut Vec<T>,
) -> T
where
    T: FloatT,
{
    let γ = {
        if d.is_empty() {
            T::max_value()
        } else {
            svec_to_mat(workΔ, d);
            lrscale_symmetric(workΔ, Λisqrt);
            let __ts = std::time::Instant::now();
            let fast = if T::precision_bits() > 64 && workΔ.nrows() > 3 {
                eigval_min_f64(workΔ, αmax)
            } else {
                None
            };
            let v = match fast {
                Some(v) => Some(v),
                None => {
                    let result = engine.eigval_min(workΔ, work);
                    work.resize(workΔ.nrows() * workΔ.ncols(), T::zero());
                    result.ok()
                }
            };
            crate::receipt::phase("cone_eigmin", __ts.elapsed());
            // a failed eigensolve means an unusable direction: zero step
            match v {
                Some(v) => v,
                None => return T::zero(),
            }
        }
    };

    if γ < T::zero() {
        T::min(-γ.recip(), αmax)
    } else {
        αmax
    }
}

/// Smallest eigenvalue of the full-precision scaled step matrix `Δ` from a
/// Float64 copy, when the step length it implies is certified to 1e-10
/// relative accuracy; `None` falls back to the full-precision solver.
///
/// Householder tridiagonalization and Sturm bisection are backward stable:
/// with the copy's rounding, the true λmin lies within
/// `B = 2(n+1)²ε‖Δ‖_F` of the computed one (a conservative bound). The step
/// `min(-1/λ, αmax)` then changes by at most `B / max(|λ|, 1/αmax)`
/// relatively. Step lengths are a free IPM parameter (the step is scaled by
/// the step fraction and kept interior), so this accuracy is ample.
/// Pure Rust: no shared BLAS/LAPACK state across the pool's threads.
fn eigval_min_f64<T: FloatT>(delta: &Matrix<T>, αmax: T) -> Option<T> {
    let n = delta.nrows();
    let amax = αmax.to_f64()?;
    let mut a = Vec::with_capacity(n * n);
    let mut sumsq = 0f64;
    for src in delta.data() {
        let v = src.to_f64()?;
        if !v.is_finite() || v.abs() > 1e150 {
            return None;
        }
        sumsq += v * v;
        a.push(v);
    }
    let gamma = min_eigenvalue_f64(&mut a, n)?;
    let bound = 2.0 * ((n + 1) as f64).powi(2) * f64::EPSILON * sumsq.sqrt();
    let floor = if amax > 0.0 && amax.is_finite() {
        1.0 / amax
    } else {
        0.0
    };
    if bound <= 1e-10 * gamma.abs().max(floor) {
        T::from_f64(gamma)
    } else {
        None
    }
}

/// Smallest eigenvalue of a symmetric matrix (`a` column-major, both
/// triangles stored; overwritten) by Householder tridiagonalization without
/// vectors and Sturm-count bisection.
fn min_eigenvalue_f64(a: &mut [f64], n: usize) -> Option<f64> {
    let at = |i: usize, j: usize| i + j * n;
    let mut d = vec![0f64; n];
    let mut e = vec![0f64; n];
    for i in (1..n).rev() {
        let l = i - 1;
        let mut h = 0f64;
        if l > 0 {
            let scale: f64 = (0..=l).map(|k| a[at(i, k)].abs()).sum();
            if scale == 0.0 {
                e[i] = a[at(i, l)];
            } else {
                for k in 0..=l {
                    a[at(i, k)] /= scale;
                    h += a[at(i, k)] * a[at(i, k)];
                }
                let f = a[at(i, l)];
                let g = if f >= 0.0 { -h.sqrt() } else { h.sqrt() };
                e[i] = scale * g;
                h -= f * g;
                a[at(i, l)] = f - g;
                let mut f = 0f64;
                for j in 0..=l {
                    let mut g = 0f64;
                    for k in 0..=j {
                        g += a[at(j, k)] * a[at(i, k)];
                    }
                    for k in j + 1..=l {
                        g += a[at(k, j)] * a[at(i, k)];
                    }
                    e[j] = g / h;
                    f += e[j] * a[at(i, j)];
                }
                let hh = f / (h + h);
                for j in 0..=l {
                    let f = a[at(i, j)];
                    let g = e[j] - hh * f;
                    e[j] = g;
                    for k in 0..=j {
                        a[at(j, k)] -= f * e[k] + g * a[at(i, k)];
                    }
                }
            }
        } else {
            e[i] = a[at(i, l)];
        }
        d[i] = h;
    }
    // e[0] served as scratch above; it is not an off-diagonal entry.
    e[0] = 0.0;
    for i in 0..n {
        d[i] = a[at(i, i)];
    }
    // Eigenvalues below x (Sturm sequence of the tridiagonal d, e[1..]).
    let tiny = f64::MIN_POSITIVE / f64::EPSILON;
    let below = |x: f64| {
        let mut count = 0;
        let mut q = d[0] - x;
        for i in 0..n {
            if i > 0 {
                q = d[i] - x - e[i] * e[i] / q;
            }
            if q == 0.0 {
                q = -tiny;
            }
            if q < 0.0 {
                count += 1;
            }
        }
        count
    };
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for i in 0..n {
        let r = e[i].abs() + if i + 1 < n { e[i + 1].abs() } else { 0.0 };
        lo = lo.min(d[i] - r);
        hi = hi.max(d[i] + r);
    }
    if !lo.is_finite() || !hi.is_finite() {
        return None;
    }
    let width = (hi - lo).max(hi.abs().max(lo.abs()) * f64::EPSILON);
    lo -= width * f64::EPSILON;
    hi += width * f64::EPSILON;
    for _ in 0..128 {
        let mid = 0.5 * (lo + hi);
        if mid <= lo || mid >= hi {
            break;
        }
        if below(mid) >= 1 {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    Some(0.5 * (lo + hi))
}

// `svec_to_mat` already mirrors this matrix, and congruence uses the same
// diagonal vector on both sides. Scale each symmetric pair once to avoid
// duplicate high-precision multiplications.
fn lrscale_symmetric<T>(matrix: &mut Matrix<T>, d: &[T])
where
    T: FloatT,
{
    let n = matrix.nrows();
    debug_assert_eq!(matrix.ncols(), n);
    debug_assert_eq!(d.len(), n);
    for col in 0..n {
        for row in 0..=col {
            matrix[(row, col)] *= d[row] * d[col];
            if row != col {
                matrix[(col, row)] = matrix[(row, col)];
            }
        }
    }
}

// Direct packed form of the existing symmetric Kronecker formula. Columns
// follow svec's upper-triangular order; each operator column writes only rows
// 0..=col, exactly matching Symmetric::pack_triu. No Hessian scratch is needed.
fn skron_packed<T>(out: &mut [T], A: &Symmetric<Matrix<T>>)
where
    T: FloatT,
{
    assert!(A.is_triu_src());
    let n = A.nrows();
    assert_eq!(out.len(), triangular_number(triangular_number(n)));
    let sqrt2 = T::SQRT_2();
    let mut col = 0;
    let mut index = 0;
    for l in 0..n {
        for k in 0..=l {
            let mut row = 0;
            for j in 0..=l {
                let Ajl = A[(j, l)];
                let Ajk = A[(j, k)];
                for i in 0..=j {
                    if row > col {
                        break;
                    }
                    out[index] = match (i == j, k == l) {
                        (false, false) => A[(i, k)] * Ajl + A[(i, l)] * Ajk,
                        (true, false) => sqrt2 * Ajl * Ajk,
                        (false, true) => sqrt2 * A[(i, l)] * Ajk,
                        (true, true) => Ajl * Ajl,
                    };
                    index += 1;
                    row += 1;
                }
            }
            col += 1;
        }
    }
}

// Retain the upstream full-matrix implementation only as a test reference.
#[cfg(test)]
// produce the upper triangular part of the Symmetric Kronecker product of
// a symmtric matrix A with itself, i.e. triu(A ⊗_s A)
fn skron<T>(out: &mut Matrix<T>, A: &Symmetric<Matrix<T>>)
where
    T: FloatT,
{
    // A is symmetric, so we can use the triu() method
    assert!(A.is_triu_src());

    let sqrt2 = T::SQRT_2();
    let n = A.nrows();

    let mut col = 0;
    for l in 0..n {
        for k in 0..=l {
            let mut row = 0;
            let kl_eq = k == l;

            for j in 0..n {
                let Ajl = A[(j, l)];
                let Ajk = A[(j, k)];

                for i in 0..=j {
                    if row > col {
                        break;
                    }

                    let ij_eq = i == j;

                    out[(row, col)] = {
                        match (ij_eq, kl_eq) {
                            (false, false) => A[(i, k)] * Ajl + A[(i, l)] * Ajk,
                            (true, false) => sqrt2 * Ajl * Ajk,
                            (false, true) => sqrt2 * A[(i, l)] * Ajk,
                            (true, true) => Ajl * Ajl,
                        }
                    };

                    row += 1;
                } //end i
            } //end j
            col += 1;
        } //end k
    } //end l
}

#[cfg(test)]
#[path = "tests/psd_hessian.rs"]
mod hessian_tests;

#[cfg(test)]
mod f64_eigmin_tests {
    use super::*;
    use sdpx_arithmetic::MpFloat;

    fn check<T: FloatT>() {
        for (n, seed) in [(4usize, 1u64), (7, 2), (20, 3), (45, 4)] {
            let mut state = seed;
            let mut next = || {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 11) as f64 / (1u64 << 53) as f64) - 0.5
            };
            let mut m = Matrix::<T>::zeros((n, n));
            for j in 0..n {
                for i in 0..=j {
                    let v = T::from_f64(next() * 10.0).unwrap();
                    m[(i, j)] = v;
                    m[(j, i)] = v;
                }
            }
            let mut work = Matrix::<T>::zeros((n, n));
            work.data_mut().copy_from_slice(m.data());
            let exact = EigEngine::<T>::new(n)
                .eigval_min(&mut work, &mut vec![T::zero()])
                .unwrap();
            let exact64 = exact.to_f64().unwrap();
            let mut a: Vec<f64> = m.data().iter().map(|v| v.to_f64().unwrap()).collect();
            let fast = min_eigenvalue_f64(&mut a, n).unwrap();
            assert!(
                (fast - exact64).abs() <= 1e-10 * n as f64,
                "n={n}: {fast} vs {exact64}"
            );
            // The certified path, when it answers, agrees with full precision.
            if let Some(v) = eigval_min_f64(&m, T::one()) {
                assert!((v.to_f64().unwrap() - exact64).abs() <= 1e-10);
            }
        }
    }

    #[test]
    fn f64_min_eigenvalue_matches_mpfr() {
        check::<MpFloat<8>>();
    }
}
