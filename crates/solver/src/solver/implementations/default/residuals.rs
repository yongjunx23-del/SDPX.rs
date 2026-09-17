#![allow(non_snake_case)]
use super::*;
use crate::algebra::*;
use crate::solver::core::traits::Residuals;

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
    pub(crate) dot_qx: T,
    pub(crate) dot_bz: T,
    pub(crate) dot_sz: T,
    pub(crate) dot_xPx: T,

    // the product Px by itself. Required for infeasibilty checks
    pub(crate) Px: Vec<T>,
    pub(crate) sparse_parallel: Option<sparse_parallel::SparseParallel>,
    #[cfg(feature = "sdp")]
    pub(crate) sampled_workspace: Option<SampledWorkspace<T>>,
    #[cfg(feature = "sdp")]
    sampled_pool: Option<std::sync::Arc<rayon::ThreadPool>>,
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
            sparse_parallel: None,
            dot_qx: T::zero(),
            dot_bz: T::zero(),
            dot_sz: T::zero(),
            dot_xPx: T::zero(),
            #[cfg(feature = "sdp")]
            sampled_workspace: None,
            #[cfg(feature = "sdp")]
            sampled_pool: None,
        }
    }
    pub(super) fn prepare_sparse(
        &mut self,
        data: &DefaultProblemData<T>,
        pool: Option<std::sync::Arc<rayon::ThreadPool>>,
    ) {
        #[cfg(feature = "sdp")]
        let is_sampled = data.sampled.is_some();
        #[cfg(not(feature = "sdp"))]
        let is_sampled = false;
        if !is_sampled
            && self.sparse_parallel.is_none()
            && pool.as_ref().is_some_and(|p| p.current_num_threads() > 1)
            && sparse_parallel::worthwhile(&data.A)
        {
            self.sparse_parallel = Some(sparse_parallel::SparseParallel::new(&data.A));
        }
        if let Some(plan) = &mut self.sparse_parallel {
            plan.configure(&data.A, if is_sampled { None } else { pool });
        }
    }

    fn ordinary_products(&mut self, variables: &DefaultVariables<T>, data: &DefaultProblemData<T>) {
        if let Some(plan) = &self.sparse_parallel {
            plan.residual_products(
                &data.A,
                &variables.x,
                &variables.z,
                &mut self.rx_inf,
                &mut self.rz_inf,
            );
        } else {
            data.A
                .t()
                .gemv(&mut self.rx_inf, &variables.z, -T::one(), T::zero());
            data.A
                .gemv(&mut self.rz_inf, &variables.x, T::one(), T::one());
        }
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
        #[cfg(feature = "sdp")]
        {
            self.sampled_pool = pool.clone();
        }
        self.prepare_sparse(data, pool);
        self.update(variables, data);
        #[cfg(feature = "sdp")]
        {
            self.sampled_pool = None;
        }
    }

    fn update(&mut self, variables: &DefaultVariables<T>, data: &DefaultProblemData<T>) {
        // various products used multiple times
        let qx = data.q.dot(&variables.x);
        let bz = data.b.dot(&variables.z);
        let sz = variables.s.dot(&variables.z);

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
        #[cfg(feature = "sdp")]
        if let Some(operator) = &data.sampled {
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
        } else {
            self.ordinary_products(variables, data);
        }
        #[cfg(not(feature = "sdp"))]
        {
            self.ordinary_products(variables, data);
        }

        //complete the residuals
        //rx = rx_inf - Px - qτ
        self.rx.waxpby(-T::one(), &self.Px, -variables.τ, &data.q);
        self.rx.axpby(T::one(), &self.rx_inf, T::one());

        // rz = rz_inf - bτ
        self.rz
            .waxpby(T::one(), &self.rz_inf, -variables.τ, &data.b);

        // τ = qz + bz + κ + xPx/τ;
        self.rτ = qx + bz + variables.κ + xPx / variables.τ;

        //save local versions
        self.dot_qx = qx;
        self.dot_bz = bz;
        self.dot_sz = sz;
        self.dot_xPx = xPx;
    }
}
