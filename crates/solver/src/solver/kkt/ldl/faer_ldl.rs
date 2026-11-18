#![allow(non_snake_case)]

use faer::dyn_stack::{MemBuffer, MemStack, StackReq};

use faer::{
    linalg::cholesky::ldlt::factor::{LdltParams, LdltRegularization},
    perm::PermRef,
    sparse::{
        linalg::cholesky::{
            factorize_symbolic_cholesky, CholeskySymbolicParams, LdltRef, SymbolicCholesky,
            SymmetricOrdering,
        },
        SparseColMatRef, SymbolicSparseColMatRef,
    },
    Conj, MatMut, Par, Side, Spec,
};

use crate::algebra::*;
use crate::solver::{
    core::CoreSettings,
    kkt::{direct::DirectLDLSolver, HasLinearSolverInfo, LinearSolverInfo},
};

#[derive(Debug)]

struct FaerLDLRegularizerParams<T: FloatT + faer_traits::RealField> {
    regularize: bool,
    eps: T,
    delta: T,
}

impl<'a, T> FaerLDLRegularizerParams<T>
where
    T: FloatT + faer_traits::RealField,
{
    fn to_faer(&self, Dsigns: &'a [i8]) -> LdltRegularization<'a, T> {
        let option = if self.regularize { Some(Dsigns) } else { None };
        LdltRegularization {
            dynamic_regularization_signs: option,
            dynamic_regularization_delta: self.delta,
            dynamic_regularization_epsilon: self.eps,
        }
    }
}

pub struct FaerDirectLDLSolver<T: FloatT + faer_traits::RealField> {
    nnz: usize,
    signs: Vec<i8>,

    // regularizer parameters captured from settings
    regularizer_params: FaerLDLRegularizerParams<T>,
    ldlt_params: Spec<LdltParams, T>,

    // symbolic + numeric cholesky data
    symbolic_cholesky: SymbolicCholesky<usize>,
    ld_vals: Vec<T>,

    // workspace for faer factor/solve calls
    work: MemBuffer,

    parallelism: Par,
    pool: Option<std::sync::Arc<rayon::ThreadPool>>,
}

impl<T> FaerDirectLDLSolver<T>
where
    T: FloatT + faer_traits::RealField,
{
    pub fn new(
        KKT: &CscMatrix<T>,
        Dsigns: &[i8],
        settings: &CoreSettings<T>,
        perm: Option<Vec<usize>>,
    ) -> Self {
        assert!(KKT.is_square(), "KKT matrix is not square");

        // -----------------------------

        // The shared solver pool is supplied after construction. Until then
        // stay serial rather than initializing a separate global Rayon pool.
        let parallelism = Par::Seq;

        // perm has possibly been passed by LDL auto selector.
        // If not, find an ordering now
        let (perm, iperm) = {
            if let Some(perm) = perm {
                let iperm = invperm(&perm);
                (perm, iperm)
            } else {
                let (perm, iperm, _) = super::amd_order(KKT);
                (perm, iperm)
            }
        };

        let symbolic = SymbolicSparseColMatRef::new_unsorted_checked(
            KKT.n,
            KKT.n,
            &KKT.colptr,
            None,
            &KKT.rowval,
        );
        let symbolic_cholesky = factorize_symbolic_cholesky(
            symbolic,
            Side::Upper,
            SymmetricOrdering::Custom(PermRef::new_checked(&perm, &iperm, KKT.n)),
            CholeskySymbolicParams::default(),
        )
        .unwrap();

        let ld_vals = vec![T::zero(); symbolic_cholesky.len_val()];

        let regularizer_params = FaerLDLRegularizerParams {
            regularize: settings.dynamic_regularization_enable,
            delta: settings.dynamic_regularization_delta,
            eps: settings.dynamic_regularization_eps,
        };

        let ldlt_params: Spec<LdltParams, T> = Default::default();

        // Required workspace for faer factor and solve
        let req_factor =
            symbolic_cholesky.factorize_numeric_ldlt_scratch::<T>(parallelism, ldlt_params);
        let req_solve = symbolic_cholesky.solve_in_place_scratch::<T>(1, parallelism); // 1 is the number of RHS
        let req = StackReq::any_of(&[req_factor, req_solve]);
        let work = MemBuffer::new(req);

        Self {
            nnz: KKT.nnz(),
            signs: Dsigns.to_vec(),
            regularizer_params,
            ldlt_params,
            symbolic_cholesky,
            ld_vals,
            work,
            parallelism,
            pool: None,
        }
    }
}

impl<T> HasLinearSolverInfo for FaerDirectLDLSolver<T>
where
    T: FloatT + faer_traits::RealField,
{
    fn linear_solver_info(&self) -> LinearSolverInfo {
        LinearSolverInfo {
            name: "faer".to_string(),
            threads: self.parallelism.degree(),
            direct: true,
            nnzA: self.nnz,
            nnzL: self.ld_vals.len() - self.symbolic_cholesky.nrows(),
        }
    }
}

impl<T> DirectLDLSolver<T> for FaerDirectLDLSolver<T>
where
    T: FloatT + faer_traits::RealField,
{
    // The owner updates the CSC before notifying the backend. Faer reads
    // that matrix while the static shift is applied and retains its factors.
    fn update_values(&mut self, _: &[usize], _: &[T]) {}
    fn update_prefix(&mut self, _: &[T]) {}
    fn scale_values(&mut self, _: &[usize], _: T) {}

    fn solve(&mut self, _kkt: &CscMatrix<T>, x: &mut [T], b: &mut [T]) {
        x.copy_from_slice(b);
        let rhs = MatMut::from_column_major_slice_mut(x, b.len(), 1);
        let ldlt = LdltRef::new(&self.symbolic_cholesky, self.ld_vals.as_slice());

        let solve = || {
            ldlt.solve_in_place_with_conj(
                Conj::No,
                rhs,
                self.parallelism,
                MemStack::new(&mut self.work),
            );
        };
        match &self.pool {
            Some(pool) => pool.install(solve),
            None => solve(),
        }
    }

    fn refactor(&mut self, kkt: &CscMatrix<T>) -> bool {
        // SAFETY: constructor checked this fixed CSC structure. Numeric
        // updates change only coefficients and never the input pattern.
        let symbolic = unsafe {
            SymbolicSparseColMatRef::new_unchecked(kkt.n, kkt.n, &kkt.colptr, None, &kkt.rowval)
        };
        let a = SparseColMatRef::new(symbolic, &kkt.nzval);
        let regularizer = self.regularizer_params.to_faer(&self.signs);

        let mut factorize = || {
            self.symbolic_cholesky
                .factorize_numeric_ldlt(
                    self.ld_vals.as_mut_slice(),
                    a,
                    Side::Upper,
                    regularizer,
                    self.parallelism,
                    MemStack::new(&mut self.work),
                    self.ldlt_params,
                )
                .is_ok() // bool, as the other LDL backends report
        };
        match &self.pool {
            Some(pool) => pool.install(factorize),
            None => factorize(),
        }
    }

    fn set_pool(&mut self, pool: Option<std::sync::Arc<rayon::ThreadPool>>) {
        let parallelism = match pool.as_ref().map_or(1, |pool| pool.current_num_threads()) {
            1 => Par::Seq,
            workers => Par::rayon(workers),
        };
        if parallelism != self.parallelism {
            let req_factor = self
                .symbolic_cholesky
                .factorize_numeric_ldlt_scratch::<T>(parallelism, self.ldlt_params);
            let req_solve = self
                .symbolic_cholesky
                .solve_in_place_scratch::<T>(1, parallelism);
            let req = StackReq::any_of(&[req_factor, req_solve]);
            if !MemStack::new(&mut self.work).can_hold(req) {
                // Release the old factor workspace before acquiring the larger one.
                self.work = MemBuffer::new(StackReq::empty());
                self.work = MemBuffer::new(req);
            }
            self.parallelism = parallelism;
        }
        self.pool = pool;
    }
}

// ---------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------

#[test]
fn test_faer_ldl() {
    let mut KKT = CscMatrix {
        m: 6,
        n: 6,
        colptr: vec![0, 1, 2, 4, 6, 8, 10],
        rowval: vec![0, 1, 0, 2, 1, 3, 0, 4, 1, 5],
        nzval: vec![1.0, 2.0, 1.0, -1.0, 1.0, -2.0, -1.0, -3.0, -1.0, -4.0],
    };

    let Dsigns = vec![1, 1, -1, -1, -1, -1];

    let mut solver = FaerDirectLDLSolver::new(&KKT, &Dsigns, &CoreSettings::default(), None);

    let mut x = vec![0.0; 6];
    let mut b = vec![1.0, 2.0, 3.0, 4., 5., 6.];

    solver.refactor(&KKT);
    solver.solve(&KKT, &mut x, &mut b);

    let xsol = vec![
        1.0,
        0.9090909090909091,
        -2.0,
        -1.5454545454545454,
        -2.0,
        -1.7272727272727275,
    ];
    assert!(x.norm_inf_diff(&xsol) < 1e-10);

    // assign a new entry at the end of the KKT matrix and resolve
    KKT.nzval[9] = -10.;

    solver.refactor(&KKT);
    solver.solve(&KKT, &mut x, &mut b);

    let xsol = vec![
        1.0,
        1.3076923076923077,
        -2.0,
        -1.346153846153846,
        -2.0,
        -0.7307692307692306,
    ];
    assert!(x.norm_inf_diff(&xsol) < 1e-10);
}

#[test]
fn test_faer_qp() {
    use crate::solver::IPSolver;
    use crate::solver::SolverStatus;

    let P = CscMatrix {
        m: 1,
        n: 1,
        colptr: vec![0, 1],
        rowval: vec![0],
        nzval: vec![2.0],
    };
    let q = [1.0];
    let A = CscMatrix {
        m: 1,
        n: 1,
        colptr: vec![0, 1],
        rowval: vec![0],
        nzval: vec![-1.0],
    };
    let b = [-2.0];
    let cones = vec![crate::solver::SupportedConeT::NonnegativeConeT(1)];

    let settings = crate::solver::DefaultSettingsBuilder::default()
        .direct_solve_method("faer".to_owned())
        .build()
        .unwrap();

    let mut solver = crate::solver::DefaultSolver::new(&P, &q, &A, &b, &cones, settings).unwrap();

    solver.solve();

    assert_eq!(solver.solution.status, SolverStatus::Solved);
}
