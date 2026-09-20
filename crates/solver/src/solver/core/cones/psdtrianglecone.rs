use super::*;
use crate::algebra::*;

// ------------------------------------
// Positive Semidefinite Cone (Scaled triangular form)
// ------------------------------------

pub struct PSDConeData<T> {
    chol1: CholeskyEngine<T>,
    chol2: CholeskyEngine<T>,
    SVD: SVDEngine<T>,
    Eig: EigEngine<T>,
    λ: Vec<T>,
    Λisqrt: Vec<T>,
    pub(crate) R: Matrix<T>,
    Rinv: Matrix<T>,
    // G = R R^T, with an authoritative upper triangle. The p-by-p
    // Hessian is generated only into an augmented KKT caller's packed block.
    G: Matrix<T>,

    //workspace for various internal uses
    workmat1: Matrix<T>,
    workmat2: Matrix<T>,
    workmat3: Matrix<T>,
    workvec: Vec<T>,
    // Second scratch set so the independent dz and ds step-bound
    // evaluations can run on two workers at once.
    eig2: EigEngine<T>,
    workmat4: Matrix<T>,
    workmat5: Matrix<T>,
    workmat6: Matrix<T>,
    workvec2: Vec<T>,
}

impl<T> PSDConeData<T>
where
    T: FloatT,
{
    pub fn new(n: usize) -> Self {
        Self {
            chol1: CholeskyEngine::<T>::new(n),
            chol2: CholeskyEngine::<T>::new(n),
            SVD: SVDEngine::<T>::new((n, n)),
            Eig: EigEngine::<T>::new(n),

            λ: vec![T::zero(); n],
            Λisqrt: vec![T::zero(); n],
            R: Matrix::zeros((n, n)),
            Rinv: Matrix::zeros((n, n)),
            G: Matrix::zeros((n, n)),

            //workspace for various internal uses
            workmat1: Matrix::zeros((n, n)),
            workmat2: Matrix::zeros((n, n)),
            workmat3: Matrix::zeros((n, n)),
            workvec: vec![T::zero(); triangular_number(n)],
            eig2: EigEngine::<T>::new(n),
            workmat4: Matrix::zeros((n, n)),
            workmat5: Matrix::zeros((n, n)),
            workmat6: Matrix::zeros((n, n)),
            workvec2: vec![T::zero(); triangular_number(n)],
        }
    }
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
        let α: T;
        let β: T;

        if z.is_empty() {
            α = T::max_value();
            β = T::zero();
        } else {
            let Z = &mut self.data.workmat1;
            svec_to_mat(Z, z);
            self.data.Eig.eigvals(Z).expect("Eigval error");
            let e = &self.data.Eig.λ;
            α = e.minimum();
            β = e.iter().fold(T::zero(), |s, x| s + T::max(*x, T::zero())); //= sum(e[e.>0])
        }

        (α, β)
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

        let f = &mut self.data;
        let (S, Z) = (&mut f.workmat1, &mut f.workmat2);
        svec_to_mat(S, s);
        svec_to_mat(Z, z);

        //compute Cholesky factors. The S and Z factorizations are
        //independent, so offer the second to an idle ambient worker.
        let (ch1, ch2) = (&mut f.chol1, &mut f.chol2);
        let (c1, c2) = if sdpx_arithmetic::inner_parallel::paired() {
            rayon::join(|| ch1.factor(S), || ch2.factor(Z))
        } else {
            (ch1.factor(S), ch2.factor(Z))
        };

        // bail if the cholesky factorization fails
        // PJG: Need proper Result return type here
        if c1.is_err() || c2.is_err() {
            return false;
        }

        let (L1, L2) = (&f.chol1.L, &f.chol2.L);

        // SVD of L2'*L1,
        let tmp = &mut f.workmat1;
        tmp.mul(&L2.t(), L1, T::one(), T::zero());
        if let Some(dir) = std::env::var_os("SDPX_DUMP_CONE") {
            dump_cone_f64(&dir, tmp);
        }

        // Direct SVD avoids squaring the condition number of L2' * L1.
        let __ts = std::time::Instant::now();
        f.SVD.factor(tmp).expect("SVD error");
        crate::receipt::phase("cone_svd", __ts.elapsed());

        // assemble λ (diagonal), R and Rinv.
        f.λ.copy_from(&f.SVD.s);
        f.Λisqrt.copy_from(&f.λ).sqrt().recip();

        //f.R = L1*(f.SVD.V)*f.Λisqrt and f.Rinv = f.Λisqrt*(f.SVD.U)'*L2'
        //are independent products; pair them when inner workers are idle.
        {
            let (R, Rinv, svd, Λi) = (&mut f.R, &mut f.Rinv, &f.SVD, &f.Λisqrt);
            let mut build_r = || {
                R.mul(L1, &svd.Vt.t(), T::one(), T::zero());
                R.rscale(Λi);
            };
            let mut build_rinv = || {
                Rinv.mul(&svd.U.t(), &L2.t(), T::one(), T::zero());
                Rinv.lscale(Λi);
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
            let f = &mut *self.data;
            let PSDConeData {
                R,
                Rinv,
                Λisqrt,
                workvec,
                workmat1,
                workmat2,
                workmat3,
                Eig,
                workvec2,
                workmat4,
                workmat5,
                workmat6,
                eig2,
                ..
            } = f;
            rayon::join(
                || {
                    step_component_inner(
                        dz, R, Λisqrt, false, αmax, workvec, workmat1, workmat2, workmat3, Eig,
                    )
                },
                || {
                    step_component_inner(
                        ds, Rinv, Λisqrt, true, αmax, workvec2, workmat4, workmat5, workmat6, eig2,
                    )
                },
            )
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
        let f = &mut self.data;
        step_component_inner(
            direction,
            if primal { &f.Rinv } else { &f.R },
            &f.Λisqrt,
            primal,
            αmax,
            &mut f.workvec,
            &mut f.workmat1,
            &mut f.workmat2,
            &mut f.workmat3,
            &mut f.Eig,
        )
    }

    pub(super) fn prepare_affine_bounds(&mut self, dz: &mut [T], ds: &mut [T], αmax: T) -> (T, T) {
        // The dz (dual, R) and ds (primal, Rinv) bounds are independent.
        // With a second scratch set they can run on two ambient workers;
        // without one they fold serially, bitwise identical either way.
        if sdpx_arithmetic::inner_parallel::paired() {
            let f = &mut *self.data;
            let PSDConeData {
                R,
                Rinv,
                Λisqrt,
                workvec,
                workmat1,
                workmat2,
                workmat3,
                Eig,
                workvec2,
                workmat4,
                workmat5,
                workmat6,
                eig2,
                ..
            } = f;
            let (αz, αs) = rayon::join(
                || {
                    let α = step_component_inner(
                        dz, R, Λisqrt, false, αmax, workvec, workmat1, workmat2, workmat3, Eig,
                    );
                    dz.copy_from_slice(workvec);
                    α
                },
                || {
                    let α = step_component_inner(
                        ds, Rinv, Λisqrt, true, αmax, workvec2, workmat4, workmat5, workmat6, eig2,
                    );
                    ds.copy_from_slice(workvec2);
                    α
                },
            );
            (αz, αs)
        } else {
            let αz = self.step_component(dz, false, αmax);
            dz.copy_from_slice(&self.data.workvec);
            let αs = self.step_component(ds, true, αmax);
            ds.copy_from_slice(&self.data.workvec);
            (αz, αs)
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
        let (Q, q) = (&mut self.data.workmat1, &mut self.data.workvec);
        q.waxpby(T::one(), x, α, dx);
        svec_to_mat(Q, q);

        match self.data.chol1.factor(Q) {
            Ok(_) => self.data.chol1.logdet(),
            Err(_) => T::infinity(),
        }
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
        let X = &mut self.data.workmat1;
        let Z = &mut self.data.workmat2;

        svec_to_mat(X, x);
        svec_to_mat(Z, z);

        let λ = &self.data.λ;
        let two: T = (2.).as_T();
        for i in 0..self.n {
            for j in 0..self.n {
                X[(i, j)] = (two * Z[(i, j)]) / (λ[i] + λ[j]);
            }
        }
        mat_to_svec(x, X);
    }

    fn mul_W(&mut self, is_transpose: MatrixShape, y: &mut [T], x: &[T], α: T, β: T) {
        mul_Wx_inner(
            is_transpose,
            y,
            x,
            α,
            β,
            &self.data.R,
            &mut self.data.workmat1,
            &mut self.data.workmat2,
            &mut self.data.workmat3,
        )
    }

    fn mul_Winv(&mut self, is_transpose: MatrixShape, y: &mut [T], x: &[T], α: T, β: T) {
        mul_Wx_inner(
            is_transpose,
            y,
            x,
            α,
            β,
            &self.data.Rinv,
            &mut self.data.workmat1,
            &mut self.data.workmat2,
            &mut self.data.workmat3,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn mul_Wx_inner<T>(
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
            tmp.mul(X, &Rx.t(), T::one(), T::zero());
            pooled_gemm_sym(Y, Rx, tmp, None);
        }
        MatrixShape::N => {
            // Y .= α*(R'*X*R) + βY         #W*x
            // Y .= α*(Rinv'*X*Rinv) + βY   #W^{-1}*x
            tmp.mul(&Rx.t(), X, T::one(), T::zero());
            pooled_gemm_sym(Y, tmp, Rx, None);
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
        let (Y, Z, X) = (
            &mut self.data.workmat1,
            &mut self.data.workmat2,
            &mut self.data.workmat3,
        );
        svec_to_mat(Y, y);
        svec_to_mat(Z, z);

        // X .= (Y*Z + Z*Y)/2
        // NB: works b/c Y and Z are both symmetric
        X.data_mut().set(T::zero()); //X.sym_up() will assert is_triu
        X.syr2k(Y, Z, (0.5).as_T(), T::zero());
        mat_to_svec(x, &X.sym_up());
    }

    fn inv_circ_op(&mut self, _x: &mut [T], _y: &[T], _z: &[T]) {
        // X should be the solution to (YX + XY)/2 = Z

        //  For general arguments this requires solution to a symmetric
        // Sylvester equation.  Throwing an error here since I do not think
        // the inverse of the ∘ operator is ever required for general arguments,
        // and solving this equation is best avoided.
        unreachable!();
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
    mul_Wx_inner(
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
    step_length_psd_component(wm1, eig, wv, Λisqrt, αmax)
}

fn step_length_psd_component<T>(
    workΔ: &mut Matrix<T>,
    engine: &mut EigEngine<T>,
    d: &[T],
    Λisqrt: &[T],
    αmax: T,
) -> T
where
    T: FloatT,
{
    let γ = {
        if d.is_empty() {
            T::max_value()
        } else {
            svec_to_mat(workΔ, d);
            workΔ.lrscale(Λisqrt, Λisqrt);
            let __ts = std::time::Instant::now();
            let v = engine.eigval_min(workΔ).expect("Eigval error");
            crate::receipt::phase("cone_eigmin", __ts.elapsed());
            v
        }
    };

    if γ < T::zero() {
        T::min(-γ.recip(), αmax)
    } else {
        αmax
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

/// Debug probe: dump the SVD input `L2'·L1` as f64 text, keyed by the
/// workspace's stable data pointer so consecutive files per cone form the
/// iteration sequence. Gated by SDPX_DUMP_CONE=<dir>; measures how much the
/// right singular factor drifts between IPM iterations.
fn dump_cone_f64<T: FloatT>(dir: &std::ffi::OsStr, m: &Matrix<T>) {
    use std::collections::HashMap;
    use std::sync::Mutex;
    static SEQ: Mutex<Option<HashMap<usize, usize>>> = Mutex::new(None);
    let key = m.data().as_ptr() as usize;
    let iter = {
        let mut guard = SEQ.lock().unwrap();
        *guard.get_or_insert_with(HashMap::new).entry(key).or_insert(0)
    };
    *SEQ.lock().unwrap().as_mut().unwrap().get_mut(&key).unwrap() += 1;
    let path = std::path::Path::new(dir).join(format!("cone-{key:x}-iter{iter:04}.txt"));
    let (rows, cols) = m.size();
    let mut out = String::with_capacity(m.data().len() * 24);
    out.push_str(&format!("{rows} {cols}\n"));
    for v in m.data().iter() {
        let f = v.to_f64().unwrap_or_else(|| {
            if *v < T::zero() {
                -f64::MAX
            } else {
                f64::MAX
            }
        });
        out.push_str(&format!("{f:.17e}\n"));
    }
    let _ = std::fs::write(&path, out);
}

#[cfg(test)]
#[path = "psd_hessian_tests.rs"]
mod hessian_tests;
