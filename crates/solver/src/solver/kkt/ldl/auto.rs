#![allow(non_snake_case)]
use crate::solver::kkt::ldl::qdldl::QDLDLDirectLDLSolver;
use crate::{
    algebra::*,
    solver::{core::CoreSettings, kkt::direct::BoxedDirectLDLSolver},
};

pub(super) fn ldl_auto_select<T>(
    KKT: &CscMatrix<T>,
    Dsigns: &[i8],
    settings: &CoreSettings<T>,
) -> BoxedDirectLDLSolver<T>
where
    T: FloatT + faer_traits::RealField,
{
    use crate::solver::kkt::ldl::faer_ldl::FaerDirectLDLSolver;

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

    // The supernodal, multithreaded factor pays only for large factors; small
    // ones (e.g. many tiny cones) are faster with QDLDL.
    const MIN_SUPERNODAL_FLOPS: f64 = 1e8;
    if crate::receipt::profile_requested() {
        eprintln!(
            "LDL_AUTO n={} flops={flops:.3e} lnz={Lnnz:.3e} ratio={:.1} choice={}",
            KKT.n,
            flops / Lnnz,
            if flops / Lnnz < thresh || flops < MIN_SUPERNODAL_FLOPS {
                "qdldl"
            } else {
                "faer"
            }
        );
    }
    if (flops / Lnnz) < thresh || flops < MIN_SUPERNODAL_FLOPS {
        // use QDLDL
        let solver = QDLDLDirectLDLSolver::<T>::new(KKT, Dsigns, settings, Some(perm));
        Box::new(solver)
    } else {
        // use faer
        let solver = FaerDirectLDLSolver::<T>::new(KKT, Dsigns, settings, Some(perm));
        Box::new(solver)
    }
}
