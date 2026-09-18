#![allow(non_snake_case)]
#[cfg(feature = "faer-sparse")]
use crate::solver::core::kktsolvers::direct::ldlsolvers::qdldl::QDLDLDirectLDLSolver;
use crate::{
    algebra::*,
    solver::core::{
        kktsolvers::direct::{BoxedDirectLDLSolver, DirectLDLSolverReqs},
        CoreSettings,
    },
};

pub struct AutoDirectLDLSolver<T> {
    T: std::marker::PhantomData<T>,
}

impl<T> DirectLDLSolverReqs for AutoDirectLDLSolver<T>
where
    T: FloatT,
{
    fn required_matrix_shape() -> MatrixTriangle {
        MatrixTriangle::Triu
    }
}

impl<T> AutoDirectLDLSolver<T>
where
    T: FloatT,
{
    #[allow(clippy::new_ret_no_self)]
    pub fn new(
        KKT: &CscMatrix<T>,
        Dsigns: &[i8],
        settings: &CoreSettings<T>,
        _perm: Option<Vec<usize>>,
    ) -> BoxedDirectLDLSolver<T> {
        T::auto_ldlsolver(KKT, Dsigns, settings)
    }
}

#[cfg(feature = "faer-sparse")]
pub(super) fn ldl_auto_select<T>(
    KKT: &CscMatrix<T>,
    Dsigns: &[i8],
    settings: &CoreSettings<T>,
) -> BoxedDirectLDLSolver<T>
where
    T: FloatT + faer_traits::RealField,
{
    use crate::solver::core::kktsolvers::direct::ldlsolvers::faer_ldl::FaerDirectLDLSolver;

    assert!(KKT.is_square(), "KKT matrix is not square");

    // Compute an AMD ordering for the KKT matrix,
    // and use it to determine whether we want to
    // use the QDLDL solver or the faer.  Slight
    // inefficiency here because we will end up computing
    // the AMD ordering twice.   Switch rule is the same
    // as the one internal to faer.   Done this way because
    // QDLDL appears to be faster than faer's simplicial method.

    // manually compute an AMD ordering for the KKT matrix
    let (perm, _iperm, info) = super::amd_order(KKT);

    // estimate flops and then use the faer switching rule
    let flops = (info.n_div + info.n_mult_subs_ldl) as f64;
    let Lnnz = info.lnz as f64;

    // threshold for switching to QDLDL
    // let thresh = faer::sparse::linalg::CHOLESKY_SUPERNODAL_RATIO_FACTOR;
    let thresh = 40.0;

    if (flops / Lnnz) < thresh {
        // use QDLDL
        let solver = QDLDLDirectLDLSolver::<T>::new(KKT, Dsigns, settings, Some(perm));
        Box::new(solver)
    } else {
        // use faer
        let solver = FaerDirectLDLSolver::<T>::new(KKT, Dsigns, settings, Some(perm));
        Box::new(solver)
    }
}
