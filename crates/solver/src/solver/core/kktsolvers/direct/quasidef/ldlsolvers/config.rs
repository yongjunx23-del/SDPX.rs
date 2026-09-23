use super::{auto::AutoDirectLDLSolver, qdldl::QDLDLDirectLDLSolver};
use crate::{
    algebra::{CscMatrix, FloatT, MatrixTriangle},
    solver::{core::kktsolvers::direct::BoxedDirectLDLSolver, CoreSettings},
};
use sdpx_arithmetic::MpFloat;

type LDLConstructor<T> =
    fn(&CscMatrix<T>, &[i8], &CoreSettings<T>, Option<Vec<usize>>) -> BoxedDirectLDLSolver<T>;

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
    ) -> BoxedDirectLDLSolver<Self>
    where
        Self: FloatT;
}

macro_rules! primitive_configuration {
    ($t:ty) => {
        impl LDLConfiguration for $t {
            fn get_ldlsolver_config(
                settings: &CoreSettings<Self>,
            ) -> (MatrixTriangle, LDLConstructor<Self>) {
                match settings.direct_solve_method.as_str() {
                    "auto" => (MatrixTriangle::Triu, AutoDirectLDLSolver::new),
                    "qdldl" => (MatrixTriangle::Triu, |m, d, s, p| {
                        Box::new(QDLDLDirectLDLSolver::new(m, d, s, p))
                    }),
                    #[cfg(feature = "faer-sparse")]
                    "faer" => (MatrixTriangle::Triu, |m, d, s, p| {
                        Box::new(super::faer_ldl::FaerDirectLDLSolver::new(m, d, s, p))
                    }),
                    method => Self::specialized_ldl(method),
                }
            }

            fn auto_ldlsolver(
                matrix: &CscMatrix<Self>,
                signs: &[i8],
                settings: &CoreSettings<Self>,
            ) -> BoxedDirectLDLSolver<Self> {
                if let Some(solver) = Self::dense_ldl(matrix, signs, settings) {
                    return solver;
                }
                #[cfg(feature = "faer-sparse")]
                {
                    super::auto::ldl_auto_select(matrix, signs, settings)
                }
                #[cfg(not(feature = "faer-sparse"))]
                {
                    Box::new(QDLDLDirectLDLSolver::new(matrix, signs, settings, None))
                }
            }
        }
    };
}

trait SpecializedLDL: FloatT {
    fn specialized_ldl(method: &str) -> (MatrixTriangle, LDLConstructor<Self>);
    fn dense_ldl(
        _matrix: &CscMatrix<Self>,
        _signs: &[i8],
        _settings: &CoreSettings<Self>,
    ) -> Option<BoxedDirectLDLSolver<Self>> {
        None
    }
}
/// Single-precision stays configured because the dense BLAS suites exercise
/// both `s`- and `d`-prefixed kernels; the solver itself is instantiated only
/// at f64 and the fixed-precision MPFR types.
impl SpecializedLDL for f32 {
    fn specialized_ldl(method: &str) -> (MatrixTriangle, LDLConstructor<Self>) {
        panic!("LDL backend {method:?} is unavailable for Float32")
    }
}
impl SpecializedLDL for f64 {
    fn dense_ldl(
        matrix: &CscMatrix<Self>,
        signs: &[i8],
        settings: &CoreSettings<Self>,
    ) -> Option<BoxedDirectLDLSolver<Self>> {
        #[cfg(feature = "sdp")]
        {
            super::dense_block::DenseBlockSolver::try_new(matrix, signs, settings)
                .map(|solver| Box::new(solver) as BoxedDirectLDLSolver<Self>)
                .or_else(|| {
                    super::arrow::ArrowLDLSolver::try_new(matrix, signs, settings)
                        .map(|a| Box::new(a) as BoxedDirectLDLSolver<Self>)
                })
        }
        #[cfg(not(feature = "sdp"))]
        {
            let _ = (matrix, signs, settings);
            None
        }
    }
    fn specialized_ldl(method: &str) -> (MatrixTriangle, LDLConstructor<Self>) {
        panic!("LDL backend {method:?} is unavailable for Float64")
    }
}
primitive_configuration!(f32);
primitive_configuration!(f64);

impl<const N: usize> LDLConfiguration for MpFloat<N> {
    fn get_ldlsolver_config(
        settings: &CoreSettings<Self>,
    ) -> (MatrixTriangle, LDLConstructor<Self>) {
        match settings.direct_solve_method.as_str() {
            // "auto" promotes eligible quasidefinite systems to the dense
            // multi-leaf arrow factorization; "qdldl" pins the baseline.
            "auto" => (MatrixTriangle::Triu, |m, d, s, _p| Self::auto_ldlsolver(m, d, s)),
            "qdldl" => (MatrixTriangle::Triu, |m, d, s, p| {
                Box::new(QDLDLDirectLDLSolver::new(m, d, s, p))
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
    ) -> BoxedDirectLDLSolver<Self> {
        super::arrow::ArrowLDLSolver::try_new(matrix, signs, settings)
            .map(|a| Box::new(a) as BoxedDirectLDLSolver<Self>)
            .unwrap_or_else(|| Box::new(QDLDLDirectLDLSolver::new(matrix, signs, settings, None)))
    }
}
