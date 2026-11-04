use super::qdldl::QDLDLDirectLDLSolver;
use crate::{
    algebra::{CscMatrix, FloatT, MatrixTriangle},
    solver::{kkt::direct::BoxedDirectLDLSolver, CoreSettings},
};
use sdpx_arithmetic::MpFloat;

/// The last argument is an AMD ordering of the same pattern computed earlier
/// in backend selection (`None` when none was needed).
type LDLConstructor<T> = fn(
    &CscMatrix<T>,
    &[i8],
    &CoreSettings<T>,
    Option<super::AmdOrdering>,
) -> BoxedDirectLDLSolver<T>;

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
                        Box::new(super::faer_ldl::FaerDirectLDLSolver::new(
                            m,
                            d,
                            s,
                            amd.map(|o| o.0),
                        ))
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
                mut amd: Option<super::AmdOrdering>,
            ) -> BoxedDirectLDLSolver<Self> {
                // The dense test may order the pattern; the sparse solver
                // then reuses that ordering.
                let dense: fn(
                    &CscMatrix<Self>,
                    &[i8],
                    &CoreSettings<Self>,
                    &mut Option<super::AmdOrdering>,
                ) -> Option<BoxedDirectLDLSolver<Self>> = $dense;
                if let Some(solver) = dense(matrix, signs, settings, &mut amd) {
                    return solver;
                }
                #[cfg(feature = "faer-sparse")]
                {
                    super::auto::ldl_auto_select(matrix, signs, settings, amd)
                }
                #[cfg(not(feature = "faer-sparse"))]
                {
                    Box::new(QDLDLDirectLDLSolver::new(
                        matrix,
                        signs,
                        settings,
                        amd.map(|o| o.0),
                    ))
                }
            }
        }
    };
}

// Single precision is configured only so the dense BLAS suites can exercise
// the `s`-prefixed kernels; the solver itself runs at f64 and MPFR.
primitive_configuration!(f32, |_, _, _, _| None);
primitive_configuration!(f64, |matrix, signs, settings, amd| {
    super::dense_block::DenseBlockSolver::try_new(matrix, signs, settings, amd)
        .map(|s| Box::new(s) as BoxedDirectLDLSolver<f64>)
        .or_else(|| {
            super::arrow::ArrowLDLSolver::try_new(matrix, signs, settings, 0)
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
        use super::arrow::ArrowLDLSolver as Arrow;
        let flipped: Vec<i8> = signs.iter().map(|&s| -s).collect();
        // `ldl_nnz`: QDLDL's factor size when an ordering is known, else 0.
        let arrow = |ldl_nnz: usize, perm: Option<&Vec<usize>>| {
            let negated = || {
                Arrow::try_negated(matrix, signs, settings, perm.cloned(), ldl_nnz)
                    .map(|a| Box::new(a) as BoxedDirectLDLSolver<Self>)
            };
            // When both arrows are eligible, the one with less dense work
            // (two_L6: K's border of 186 ORE rows against −K's 81 primal columns).
            let negated_first = Arrow::work(matrix, &flipped, ldl_nnz)
                .is_some_and(|w| Arrow::work(matrix, signs, ldl_nnz).is_none_or(|p| w < p));
            if negated_first {
                negated()
            } else {
                Arrow::try_new(matrix, signs, settings, ldl_nnz)
                    .map(|a| Box::new(a) as BoxedDirectLDLSolver<Self>)
                    .or_else(negated)
            }
        };
        let ordered = amd.is_some();
        let known = amd.as_ref().map_or(0, |(_, _, info)| info.lnz);
        if let Some(solver) = arrow(known, amd.as_ref().map(|(perm, _, _)| perm)) {
            return solver;
        }
        // No arrow fits the stored KKT's storage bound: QDLDL's own AMD
        // ordering, which the QDLDL fallback needs anyway, gives its factor
        // size for a second storage test.
        let ordering = amd.unwrap_or_else(|| crate::qdldl::get_amd_ordering(matrix, 1.5));
        let info = &ordering.2;
        if crate::receipt::profile_requested() {
            eprintln!(
                "MP_LDL n={} lnz={} flops={:.3e}",
                matrix.n,
                info.lnz,
                (info.n_div + info.n_mult_subs_ldl) as f64
            );
        }
        if !ordered {
            if let Some(solver) = arrow(info.lnz, Some(&ordering.0)) {
                return solver;
            }
        }
        // No arrow structure: one pooled dense leaf when serial QDLDL would
        // cost more.
        let mut amd = Some(ordering);
        if let Some(solver) = Arrow::try_dense(matrix, signs, settings, &mut amd) {
            return Box::new(solver);
        }
        Box::new(QDLDLDirectLDLSolver::new(
            matrix,
            signs,
            settings,
            amd.map(|o| o.0),
        ))
    }
}
