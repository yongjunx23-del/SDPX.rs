use super::qdldl::QDLDLDirectLDLSolver;
use crate::{
    algebra::{CscMatrix, FloatT, MatrixTriangle},
    solver::{kkt::direct::BoxedDirectLDLSolver, CoreSettings},
};
use sdpx_arithmetic::MpFloat;

/// The last argument is an AMD ordering of the same pattern computed earlier
/// in backend selection (`None` when none was needed).
type LDLConstructor<T> =
    fn(&CscMatrix<T>, &[i8], &CoreSettings<T>, Option<super::AmdOrdering>) -> BoxedDirectLDLSolver<T>;

/// Sparse numerical-provider selection, separate from scalar arithmetic.
/// Implementations select only backends that support their scalar type.
pub trait LDLConfiguration: Sized {
    fn get_ldlsolver_config(
        settings: &CoreSettings<Self>,
    ) -> (MatrixTriangle, LDLConstructor<Self>)
    where
        Self: FloatT;

    #[doc(hidden)]
    fn auto_ldlsolver(
        matrix: &CscMatrix<Self>,
        signs: &[i8],
        settings: &CoreSettings<Self>,
        amd: Option<super::AmdOrdering>,
    ) -> BoxedDirectLDLSolver<Self>
    where
        Self: FloatT;
}

macro_rules! primitive_configuration {
    ($t:ty, $dense:expr) => {
        impl LDLConfiguration for $t {
            fn get_ldlsolver_config(
                settings: &CoreSettings<Self>,
            ) -> (MatrixTriangle, LDLConstructor<Self>) {
                match settings.direct_solve_method.as_str() {
                    "auto" => (MatrixTriangle::Triu, |m, d, s, amd| {
                        Self::auto_ldlsolver(m, d, s, amd)
                    }),
                    "qdldl" => (MatrixTriangle::Triu, |m, d, s, amd| {
                        Box::new(QDLDLDirectLDLSolver::new(m, d, s, amd.map(|o| o.0)))
                    }),
                    #[cfg(feature = "faer-sparse")]
                    "faer" => (MatrixTriangle::Triu, |m, d, s, amd| {
                        Box::new(super::faer_ldl::FaerDirectLDLSolver::new(m, d, s, amd.map(|o| o.0)))
                    }),
                    method => panic!(
                        "LDL backend {method:?} is unavailable for {}",
                        stringify!($t)
                    ),
                }
            }

            fn auto_ldlsolver(
                matrix: &CscMatrix<Self>,
                signs: &[i8],
                settings: &CoreSettings<Self>,
                amd: Option<super::AmdOrdering>,
            ) -> BoxedDirectLDLSolver<Self> {
                let dense: fn(
                    &CscMatrix<Self>,
                    &[i8],
                    &CoreSettings<Self>,
                ) -> Option<BoxedDirectLDLSolver<Self>> = $dense;
                if let Some(solver) = dense(matrix, signs, settings) {
                    return solver;
                }
                #[cfg(feature = "faer-sparse")]
                {
                    super::auto::ldl_auto_select(matrix, signs, settings, amd)
                }
                #[cfg(not(feature = "faer-sparse"))]
                {
                    Box::new(QDLDLDirectLDLSolver::new(matrix, signs, settings, amd.map(|o| o.0)))
                }
            }
        }
    };
}

// Single precision is configured only so the dense BLAS suites can exercise
// the `s`-prefixed kernels; the solver itself runs at f64 and MPFR.
primitive_configuration!(f32, |_, _, _| None);
primitive_configuration!(f64, |matrix, signs, settings| {
    super::dense_block::DenseBlockSolver::try_new(matrix, signs, settings)
        .map(|s| Box::new(s) as BoxedDirectLDLSolver<f64>)
        .or_else(|| {
            super::arrow::ArrowLDLSolver::try_new(matrix, signs, settings)
                .map(|a| Box::new(a) as BoxedDirectLDLSolver<f64>)
        })
});

impl<const N: usize> LDLConfiguration for MpFloat<N> {
    fn get_ldlsolver_config(
        settings: &CoreSettings<Self>,
    ) -> (MatrixTriangle, LDLConstructor<Self>) {
        match settings.direct_solve_method.as_str() {
            // "auto" promotes eligible quasidefinite systems to the dense
            // multi-leaf arrow factorization; "qdldl" pins the baseline.
            "auto" => (MatrixTriangle::Triu, |m, d, s, amd| {
                Self::auto_ldlsolver(m, d, s, amd)
            }),
            "qdldl" => (MatrixTriangle::Triu, |m, d, s, amd| {
                Box::new(QDLDLDirectLDLSolver::new(m, d, s, amd.map(|o| o.0)))
            }),
            method => panic!(
                "LDL backend {method:?} does not support {}-bit arithmetic",
                Self::PRECISION_BITS
            ),
        }
    }

    fn auto_ldlsolver(
        matrix: &CscMatrix<Self>,
        signs: &[i8],
        settings: &CoreSettings<Self>,
        amd: Option<super::AmdOrdering>,
    ) -> BoxedDirectLDLSolver<Self> {
        super::arrow::ArrowLDLSolver::try_new(matrix, signs, settings)
            .map(|a| Box::new(a) as BoxedDirectLDLSolver<Self>)
            .unwrap_or_else(|| {
                Box::new(QDLDLDirectLDLSolver::new(matrix, signs, settings, amd.map(|o| o.0)))
            })
    }
}
