#![allow(non_snake_case)]
use super::*;
use crate::algebra::*;
use crate::solver::core::traits::Residuals;
use rayon::prelude::*;

// ---------------
// Residuals type for default problem format
// ---------------

/// Standard-form solver type implementing the [`Residuals`](crate::solver::core::traits::Residuals) trait
pub struct DefaultResiduals<T> {
    // the main KKT residuals
    pub(crate) rx: Vec<T>,
    pub(crate) rz: Vec<T>,
    pub(crate) rτ: T,

    // partial residuals for infeasibility checks
    pub(crate) rx_inf: Vec<T>,
    pub(crate) rz_inf: Vec<T>,

    // various inner products.
    // NB: these are invariant w.r.t equilibration
    pub(crate) products: ResidualProducts<T>,

    // the product Px by itself. Required for infeasibilty checks
    pub(crate) Px: Vec<T>,
    /// Optional operator-aware maximum componentwise dual residual. It is
    /// populated only when the corresponding setting is enabled, so the
    /// default global residual path has no extra scan or allocation.
    pub(crate) dual_componentwise: Option<T>,
    pub(crate) sparse_parallel: Option<sparse_parallel::SparseParallel>,
    /// Whether an installed sparse plan belongs to an owner-local block.
    /// Local blocks must bypass the implicit MPI world and use only their
    /// shared owner worker pool when evaluating residual products.
    pub(crate) sparse_parallel_local: bool,
    pub(crate) sampled_workspace: Option<SampledWorkspace<T>>,
    pub(crate) sampled_pool: Option<std::sync::Arc<rayon::ThreadPool>>,
    /// Binary64 dense-column view of A for serial products; built on first
    /// use and dropped when A's values change.
    pub(crate) a_panel: Option<Option<DenseColumns<T>>>,
}

impl<T> DefaultResiduals<T>
where
    T: FloatT,
{
    /// Create a new `DefaultResiduals` object
    pub fn new(n: usize, m: usize) -> Self {
        let rx = vec![T::zero(); n];
        let rz = vec![T::zero(); m];
        let rτ = T::one();

        let rx_inf = vec![T::zero(); n];
        let rz_inf = vec![T::zero(); m];

        let Px = vec![T::zero(); n];

        Self {
            rx,
            rz,
            rτ,
            rx_inf,
            rz_inf,
            Px,
            dual_componentwise: None,
            sparse_parallel: None,
            sparse_parallel_local: false,
            products: ResidualProducts::zero(),
            sampled_workspace: None,
            sampled_pool: None,
            a_panel: None,
        }
    }
    pub(super) fn prepare_sparse(
        &mut self,
        data: &DefaultProblemData<T>,
        pool: Option<std::sync::Arc<rayon::ThreadPool>>,
    ) {
        self.prepare_sparse_impl(data, pool, false);
    }

    /// Prepare the sparse row plan for an already owner-local problem block.
    /// Unlike the production route this intentionally does not inspect the
    /// implicit MPI world: rank-local blocks must not enter rank collectives or
    /// get sharded a second time while running inside the shared owner pool.
    pub(crate) fn prepare_sparse_local(
        &mut self,
        data: &DefaultProblemData<T>,
        pool: Option<std::sync::Arc<rayon::ThreadPool>>,
    ) {
        self.prepare_sparse_impl(data, pool, true);
    }

    fn prepare_sparse_impl(
        &mut self,
        data: &DefaultProblemData<T>,
        pool: Option<std::sync::Arc<rayon::ThreadPool>>,
        local_only: bool,
    ) {
        self.sparse_parallel_local = local_only;
        let is_sampled = data.sampled.is_some();
        if self.sampled_workspace.is_none() {
            if let Some(operator) = &data.sampled {
                operator.prepare_constants(pool.as_deref());
                let mut work = SampledWorkspace::new(operator);
                work.enable_basis_caches();
                self.sampled_workspace = Some(work);
            }
        }
        if !is_sampled {
            self.ensure_panel(data);
        }
        if !is_sampled
            && self.sparse_parallel.is_none()
            && !matches!(self.a_panel, Some(Some(_)))
            && (pool.as_ref().is_some_and(|p| p.current_num_threads() > 1)
                || (!local_only && crate::mpi::World::get().is_some())
                || T::precision_bits() > 64)
            && sparse_parallel::worthwhile(&data.A)
        {
            self.sparse_parallel = Some(sparse_parallel::SparseParallel::new(&data.A));
        }
        if let Some(plan) = &mut self.sparse_parallel {
            plan.configure(&data.A, if is_sampled { None } else { pool });
        }
    }

    /// A binary64 dense-column panel serves every thread count, so results
    /// do not depend on it; replicated MPI products keep the row plan.
    /// `update_A` drops the panel.
    fn ensure_panel(&mut self, data: &DefaultProblemData<T>) {
        if self.a_panel.is_none()
            && (self.sparse_parallel_local || crate::mpi::World::get().is_none())
        {
            self.a_panel = Some(DenseColumns::new(&data.A));
        }
    }

    fn ordinary_products(&mut self, variables: &DefaultVariables<T>, data: &DefaultProblemData<T>) {
        self.ensure_panel(data);
        if let Some(Some(panel)) = &self.a_panel {
            panel.gemv(true, &mut self.rx_inf, &variables.z, -T::one(), T::zero());
            panel.gemv(false, &mut self.rz_inf, &variables.x, T::one(), T::one());
        } else if let Some(plan) = &self.sparse_parallel {
            if self.sparse_parallel_local {
                plan.residual_products_local(
                    &data.A,
                    &variables.x,
                    &variables.z,
                    &mut self.rx_inf,
                    &mut self.rz_inf,
                );
            } else {
                plan.residual_products(
                    &data.A,
                    &variables.x,
                    &variables.z,
                    &mut self.rx_inf,
                    &mut self.rz_inf,
                );
            }
        } else {
            data.A
                .t()
                .gemv(&mut self.rx_inf, &variables.z, -T::one(), T::zero());
            data.A
                .gemv(&mut self.rz_inf, &variables.x, T::one(), T::one());
        }
    }

    /// Compute the operator-aware dual backward error in the current
    /// (equilibrated, homogeneous) coordinates. Multiplying every coordinate
    /// by its positive unscaling factor would multiply both numerator and
    /// denominator equally, so this ratio is identical in original
    /// coordinates after `tau` recovery.
    fn componentwise_dual_error(
        &mut self,
        variables: &DefaultVariables<T>,
        data: &DefaultProblemData<T>,
    ) -> T {
        let mut work = vec![T::zero(); data.n];

        // The affine objective contribution is |tau*q_i|.
        for (w, &q) in work.iter_mut().zip(&data.q) {
            *w = T::abs(variables.τ * q);
        }

        // P is stored in its upper triangle and acts symmetrically. Preserve
        // each stored coefficient as one individual contribution in both
        // affected dual coordinates.
        for col in 0..data.P.n {
            for idx in data.P.colptr[col]..data.P.colptr[col + 1] {
                let row = data.P.rowval[idx];
                let value = data.P.nzval[idx];
                work[row] += T::abs(value * variables.x[col]);
                if row != col {
                    work[col] += T::abs(value * variables.x[row]);
                }
            }
        }

        // Ordinary CSC entries are authoritative for non-sampled rows. The
        // sampled materialized view contains rounded PSD rows, so use the
        // operator's stripped linear view whenever factors are installed.
        // Each column accumulates only its own entries, so a column-parallel
        // pass keeps every value bitwise equal to the serial loop.
        let pool = self.sampled_pool.clone();
        let accumulate_linear = |work: &mut [T], a: &CscMatrix<T>| {
            let column = |(col, w): (usize, &mut T)| {
                for idx in a.colptr[col]..a.colptr[col + 1] {
                    let row = a.rowval[idx];
                    *w += T::abs(a.nzval[idx] * variables.z[row]);
                }
            };
            match &pool {
                Some(pool) if pool.current_num_threads() > 1 && a.nnz() >= 4096 => {
                    pool.install(|| {
                        work[..a.n]
                            .par_iter_mut()
                            .enumerate()
                            .with_min_len(8)
                            .for_each(column)
                    })
                }
                _ => work[..a.n].iter_mut().enumerate().for_each(column),
            }
        };
        if let Some(operator) = &data.sampled {
            accumulate_linear(&mut work, operator.linear());
        } else {
            accumulate_linear(&mut work, &data.A);
        }

        if let Some(operator) = &data.sampled {
            let timer = crate::receipt::start();
            let sampled_workspace = self
                .sampled_workspace
                .get_or_insert_with(|| SampledWorkspace::new(operator));
            operator.add_adjoint_abs(
                &mut work,
                &variables.z,
                sampled_workspace,
                self.sampled_pool.as_ref(),
            );
            crate::receipt::finish("ipm_residual.adjoint_abs", timer);
        }

        // A zero denominator with a nonzero residual is a genuine failure;
        // represent it as +infinity rather than silently normalizing by one.
        work.into_iter()
            .zip(&self.rx)
            .map(|(denom, &residual)| {
                let numerator = T::abs(residual);
                if !denom.is_finite() || !numerator.is_finite() {
                    T::infinity()
                } else if denom == T::zero() {
                    if numerator == T::zero() {
                        T::zero()
                    } else {
                        T::infinity()
                    }
                } else {
                    let ratio = numerator / denom;
                    if ratio.is_finite() {
                        ratio
                    } else {
                        T::infinity()
                    }
                }
            })
            .fold(T::zero(), T::max)
    }
}

impl<T> Residuals<T> for DefaultResiduals<T>
where
    T: FloatT,
{
    type D = DefaultProblemData<T>;
    type V = DefaultVariables<T>;

    fn update_with_pool(
        &mut self,
        variables: &DefaultVariables<T>,
        data: &DefaultProblemData<T>,
        pool: Option<std::sync::Arc<rayon::ThreadPool>>,
    ) {
        {
            self.sampled_pool = pool.clone();
        }
        self.prepare_sparse(data, pool);
        self.update(variables, data);
        {
            self.sampled_pool = None;
        }
    }

    fn update(&mut self, variables: &DefaultVariables<T>, data: &DefaultProblemData<T>) {
        self.update_counted(variables, data, None);
    }
}

impl<T: FloatT> DefaultResiduals<T> {
    /// Local residual update with uniquely counted row products. The default
    /// path passes None and retains its original dot-product evaluation.
    pub(crate) fn update_counted(
        &mut self,
        variables: &DefaultVariables<T>,
        data: &DefaultProblemData<T>,
        counted_rows: Option<&[usize]>,
    ) {
        // various products used multiple times
        let qx = data.q.dot(&variables.x);
        let row_dot = |left: &[T]| {
            if let Some(ids) = counted_rows {
                ids.iter()
                    .fold(T::zero(), |sum, &i| left[i].mul_add(variables.z[i], sum))
            } else {
                left.dot(&variables.z)
            }
        };
        let bz = row_dot(&data.b);
        let sz = row_dot(&variables.s);

        //Px = P*x, P treated as symmetric
        let symP = data.P.sym_up();
        symP.symv(&mut self.Px, &variables.x, T::one(), T::zero());

        let xPx = variables.x.dot(&self.Px);

        //partial residual calc so we can check primal/dual
        //infeasibility conditions

        //Same as:
        //rx_inf .= -data.A'* variables.z
        //Same as:  residuals.rz_inf .=  data.A * variables.x + variables.s
        self.rz_inf.copy_from(&variables.s);
        if let Some(operator) = &data.sampled {
            let timer = crate::receipt::start();
            let work = self
                .sampled_workspace
                .get_or_insert_with(|| SampledWorkspace::new(operator));
            operator.apply_transpose_with_pool(
                &mut self.rx_inf,
                &variables.z,
                -T::one(),
                T::zero(),
                work,
                self.sampled_pool.as_ref(),
            );
            operator.apply_with_pool(
                &mut self.rz_inf,
                &variables.x,
                T::one(),
                T::one(),
                work,
                self.sampled_pool.as_ref(),
            );
            crate::receipt::finish("ipm_residual.products", timer);
        } else {
            self.ordinary_products(variables, data);
        }

        //complete the residuals
        //rx = rx_inf - Px - qτ
        self.rx.waxpby(-T::one(), &self.Px, -variables.τ, &data.q);
        self.rx.axpby(T::one(), &self.rx_inf, T::one());

        self.dual_componentwise = data
            .componentwise_enabled
            .then(|| self.componentwise_dual_error(variables, data));

        // rz = rz_inf - bτ
        self.rz
            .waxpby(T::one(), &self.rz_inf, -variables.τ, &data.b);

        // τ = qz + bz + κ + xPx/τ;
        self.rτ = qx + bz + variables.κ + xPx / variables.τ;

        if counted_rows.is_none() {
            trace_gap(variables, data, &self.rx, &self.rz, qx, bz, sz);
        }

        //save local versions
        self.products = ResidualProducts {
            qx,
            bz,
            sz,
            xpx: xPx,
        };
    }
}

/// Diagnostic only (`SDPX_TRACE_TAU`): with P = 0, τ²(p − d) = sᵀz − xᵀr_x − zᵀr_z
/// (unscaled by c). Prints the three terms over τ² and both costs.
fn trace_gap<T: FloatT>(
    v: &DefaultVariables<T>,
    data: &DefaultProblemData<T>,
    rx: &[T],
    rz: &[T],
    qx: T,
    bz: T,
    sz: T,
) {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| std::env::var_os("SDPX_TRACE_TAU").is_some()) {
        return;
    }
    let cinv = T::recip(data.equilibration.c);
    let t2 = T::recip(v.τ * v.τ) * cinv;
    eprintln!(
        "gap-trace sz {:.3e} xrx {:.3e} zrz {:.3e} rx {:.3e} rz {:.3e} p {:.45e} d {:.45e}",
        sz * t2,
        -v.x.dot(rx) * t2,
        -v.z.dot(rz) * t2,
        rx.norm_inf() * cinv / v.τ,
        rz.norm_inf() / v.τ,
        qx * cinv / v.τ,
        -bz * cinv / v.τ
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::SupportedConeT::NonnegativeConeT;
    use std::sync::Arc;

    fn dense_operator<T: FloatT>(m: usize, n: usize) -> CscMatrix<T> {
        let mut colptr = Vec::with_capacity(n + 1);
        let mut rowval = Vec::with_capacity(m * n);
        let mut nzval = Vec::with_capacity(m * n);
        colptr.push(0);
        let denom = T::from_usize(17).unwrap();
        for col in 0..n {
            for row in 0..m {
                rowval.push(row);
                let value = T::from_usize((row + 3 * col) % 17 + 1).unwrap() / denom;
                nzval.push(value);
            }
            colptr.push(rowval.len());
        }
        CscMatrix::new(m, n, colptr, rowval, nzval)
    }

    fn local_sparse_products_match_serial<T: FloatT>() {
        // The f64 launch threshold is 32768 stored entries.  Keeping exactly
        // that many dense entries also leaves a useful MPFR exercise while
        // preserving a compact, deterministic fixture.
        let (m, n) = (128, 256);
        let a = dense_operator::<T>(m, n);
        let p = CscMatrix::zeros((n, n));
        let q = vec![T::zero(); n];
        let b = vec![T::zero(); m];
        let settings = DefaultSettings {
            verbose: false,
            max_threads: 2,
            presolve_enable: false,
            input_sparse_dropzeros: false,
            equilibrate_enable: false,
            ..DefaultSettings::<T>::default()
        };
        let data = DefaultProblemData::new(&p, &q, &a, &b, &[NonnegativeConeT(m)], &settings);
        let mut variables = DefaultVariables::new(n, m);
        let denom_x = T::from_usize(19).unwrap();
        let denom_z = T::from_usize(23).unwrap();
        for (i, value) in variables.x.iter_mut().enumerate() {
            *value = T::from_usize(i % 19 + 1).unwrap() / denom_x;
        }
        for (i, value) in variables.z.iter_mut().enumerate() {
            *value = T::from_usize(i % 23 + 1).unwrap() / denom_z;
        }
        for (i, value) in variables.s.iter_mut().enumerate() {
            *value = T::from_usize(i % 11 + 1).unwrap() / T::from_usize(11).unwrap();
        }

        let mut serial = DefaultResiduals::new(n, m);
        serial.prepare_sparse(&data, None);
        serial.update(&variables, &data);

        let pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(4)
                .build()
                .unwrap(),
        );
        let mut local = DefaultResiduals::new(n, m);
        local.prepare_sparse_local(&data, Some(Arc::clone(&pool)));
        assert!(local.sparse_parallel_local);
        if T::precision_bits() <= 53 {
            // Binary64 dense columns use the panel at every thread count.
            assert!(matches!(local.a_panel, Some(Some(_))));
            assert!(local.sparse_parallel.is_none());
        } else {
            let plan = local.sparse_parallel.as_ref().unwrap();
            assert!(plan.has_lanes());
            assert_eq!(plan.test_pool_and_storage().0, 4);
        }
        local.update(&variables, &data);

        // Every row/column lane scans the original CSC positions in order;
        // equality therefore checks the product and the complete residual,
        // rather than only a norm.
        assert_eq!(serial.rx_inf, local.rx_inf);
        assert_eq!(serial.rz_inf, local.rz_inf);
        assert_eq!(serial.rx, local.rx);
        assert_eq!(serial.rz, local.rz);
        assert_eq!(serial.Px, local.Px);
        assert_eq!(serial.rτ, local.rτ);
        assert_eq!(serial.products.qx, local.products.qx);
        assert_eq!(serial.products.bz, local.products.bz);
        assert_eq!(serial.products.sz, local.products.sz);
        assert_eq!(serial.products.xpx, local.products.xpx);
    }

    #[test]
    fn owner_local_sparse_products_f64() {
        local_sparse_products_match_serial::<f64>();
    }

    #[test]
    fn owner_local_sparse_products_mpfr256() {
        local_sparse_products_match_serial::<sdpx_arithmetic::Bits256>();
    }

    #[test]
    fn owner_local_sparse_products_mpfr512() {
        local_sparse_products_match_serial::<sdpx_arithmetic::Bits512>();
    }
}
