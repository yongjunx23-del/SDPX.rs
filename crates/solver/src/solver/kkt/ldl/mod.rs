pub mod arrow;
#[cfg(feature = "faer-sparse")]
pub mod auto;
pub mod config;
mod dense_block;
pub mod qdldl;
mod supernodal;

#[cfg(feature = "faer-sparse")]
pub mod faer_ldl;

/// AMD permutation, its inverse and the symbolic statistics of one KKT pattern.
pub(crate) type AmdOrdering = (Vec<usize>, Vec<usize>, amd::Info);

/// A sparse factor that still needs half of the dense elimination's
/// arithmetic saves too little once several workers are available: the
/// pooled dense kernels use the whole pool, the sparse factors far less of it
/// (Ising 41/85-site condensed systems: faer ran 1.3 of 8 and 7.6 of 64
/// workers). A `serial` sparse factor (QDLDL at MPFR) loses once its
/// arithmetic exceeds the dense one spread over the pool. The ordering stays
/// with the caller for the sparse solver.
fn dense_factor<T: crate::algebra::FloatT>(
    k: &crate::algebra::CscMatrix<T>,
    settings: &crate::solver::CoreSettings<T>,
    amd: &mut Option<AmdOrdering>,
    serial: bool,
) -> bool {
    let workers = crate::solver::core::worker_budget(settings.max_threads as usize);
    let fill = k.colptr[k.n] as f64 / ((k.n as f64) * (k.n as f64 + 1.) / 2.);
    if workers < 2 || (!serial && fill < 0.1) {
        return false;
    }
    let info = &amd
        .get_or_insert_with(|| crate::qdldl::get_amd_ordering(k, 1.5))
        .2;
    // Multiply-subtract pairs, as the AMD statistics count them.
    let sparse = (info.n_div + info.n_mult_subs_ldl) as f64;
    let dense = (k.n as f64).powi(3) / 6.;
    if serial {
        sparse * workers as f64 >= dense
    } else {
        sparse >= 0.5 * dense
    }
}

#[cfg(feature = "faer-sparse")]
pub(crate) fn amd_order<T>(KKT: &crate::algebra::CscMatrix<T>) -> AmdOrdering
where
    T: crate::algebra::FloatT,
{
    // manually compute an AMD ordering for the KKT matrix
    let amd_dense_scale = 1.5; // magic number from QDLDL
    let (perm, iperm, info) = crate::qdldl::get_amd_ordering(KKT, amd_dense_scale);
    (perm, iperm, info)
}
