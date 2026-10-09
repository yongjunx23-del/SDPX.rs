pub mod arrow;
#[cfg(feature = "faer-sparse")]
pub mod auto;
pub mod config;
mod dense_block;
pub mod qdldl;

#[cfg(feature = "faer-sparse")]
pub mod faer_ldl;

/// AMD permutation, its inverse and the symbolic statistics of one KKT pattern.
pub(crate) type AmdOrdering = (Vec<usize>, Vec<usize>, amd::Info);

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
