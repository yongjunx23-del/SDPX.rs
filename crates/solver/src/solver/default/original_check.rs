//! Original-coordinate acceptance test.
//!
//! The internal convergence test divides residuals by `‖x‖ + ‖s‖` (primal)
//! and `‖x‖ + ‖z‖` (dual) of the transformed problem. Presolve, chordal
//! overlap variables and large solutions inflate those norms: on SDP_control1
//! the decomposed `‖x‖` is 1.2e5 against 18 for the original variables, and
//! the returned point had `‖A'z + q‖∞ = 0.036` while the internal test passed.
//! This module evaluates the point the solver would return, in original
//! coordinates, with the audit normalization `1 + ‖b‖∞`, `1 + ‖q‖∞` and
//! `1 + |primal objective|`, so that a `Solved` status implies the audit.
#![allow(non_snake_case)]

use crate::algebra::*;
use crate::solver::SupportedConeT;

/// Copy of the problem data before presolve, chordal decomposition and
/// equilibration. Kept only when presolve or chordal decomposition changed
/// the problem; otherwise the original data are recovered by unscaling.
#[derive(Clone)]
pub(crate) struct OriginalData<T> {
    pub P: CscMatrix<T>,
    pub q: Vec<T>,
    pub A: CscMatrix<T>,
    pub b: Vec<T>,
    pub cones: Vec<SupportedConeT<T>>,
    pub sampled: Option<std::sync::Arc<crate::solver::sampled::SampledOperator<T>>>,
}

/// Original-coordinate residuals, already normalized.
#[derive(Debug, Clone, Copy)]
pub(crate) struct OriginalResiduals<T> {
    /// `max(‖b-Ax-s‖∞, dist_K(s), dist_K(b-Ax)) / (1 + ‖b‖∞)`
    pub primal: T,
    /// `max(‖Px+A'z+q‖∞, dist_K*(z)) / (1 + ‖q‖∞)`
    pub dual: T,
    /// `|primal objective - dual objective|`
    pub gap_abs: T,
    /// `gap_abs / (1 + |primal objective|)`
    pub gap_rel: T,
}

impl<T: FloatT> OriginalResiduals<T> {
    /// Largest tolerance multiple; at most one means the tolerances hold.
    /// NaN residuals give infinity.
    pub(crate) fn ratio(&self, tol_feas: T, tol_gap_abs: T, tol_gap_rel: T) -> T {
        let div = |v: T, tol: T| {
            if v.is_nan() {
                T::infinity()
            } else if tol > T::zero() {
                v / tol
            } else if v > T::zero() {
                T::infinity()
            } else {
                T::zero()
            }
        };
        let gap = T::min(
            div(self.gap_abs, tol_gap_abs),
            div(self.gap_rel, tol_gap_rel),
        );
        T::max(
            T::max(div(self.primal, tol_feas), div(self.dual, tol_feas)),
            gap,
        )
    }
}

impl<T: FloatT> OriginalData<T> {
    /// Residuals of the point `(x, s, z)` against these data. Rows whose `b`
    /// is at or above the solver's infinity are unbounded and carry no
    /// primal residual. Without `psd_dual`, PSD duals are not checked for
    /// cone membership: chordal decomposition without dual completion
    /// returns only the entries on the sparsity pattern.
    pub(crate) fn residuals(
        &self,
        x: &[T],
        s: &[T],
        z: &[T],
        psd_dual: bool,
    ) -> OriginalResiduals<T> {
        self.products(x, z).residuals(s, z, &self.cones, psd_dual)
    }

    fn products(&self, x: &[T], z: &[T]) -> OriginalProducts<T> {
        let (m, n) = (self.A.m, self.A.n);
        debug_assert!(x.len() == n && z.len() == m);
        let mut r = self.b.clone();
        let mut px = vec![T::zero(); n];
        if self.P.nnz() > 0 {
            self.P
                .sym(MatrixTriangle::Triu)
                .symv(&mut px, x, T::one(), T::zero());
        }
        let xpx = x.dot(&px);
        let mut rd = px;
        if let Some(op) = &self.sampled {
            let mut work = crate::solver::sampled::SampledWorkspace::new(op);
            op.apply(&mut r, x, -T::one(), T::one(), &mut work);
            op.apply_transpose(&mut rd, z, T::one(), T::one(), &mut work);
        } else {
            self.A.gemv(&mut r, x, -T::one(), T::one());
            self.A.t().gemv(&mut rd, z, T::one(), T::one());
        }
        rd.axpby(T::one(), &self.q, T::one());
        OriginalProducts::new(self.b.clone(), r, rd, &self.q, x, z, xpx)
    }
}

/// Products in original coordinates, without retaining another coefficient matrix.
pub(crate) struct OriginalProducts<T> {
    pub b: Vec<T>,
    pub r: Vec<T>,
    pub rd: Vec<T>,
    pub qx: T,
    pub xpx: T,
    pub bz: T,
    pub normb: T,
    pub normq: T,
}

impl<T: FloatT> OriginalProducts<T> {
    pub(crate) fn new(b: Vec<T>, r: Vec<T>, rd: Vec<T>, q: &[T], x: &[T], z: &[T], xpx: T) -> Self {
        let infbound: T = crate::get_infinity().as_T();
        let finite = |v: T| T::abs(v) < infbound;
        let bz = b
            .iter()
            .zip(z)
            .filter(|(&bi, _)| finite(bi))
            .fold(T::zero(), |acc, (&bi, &zi)| acc + bi * zi);
        let normb = b
            .iter()
            .filter(|&&v| finite(v))
            .fold(T::zero(), |acc, &v| T::max(acc, T::abs(v)));
        Self {
            b,
            r,
            rd,
            qx: q.dot(x),
            xpx,
            bz,
            normb,
            normq: q.norm_inf(),
        }
    }

    pub(crate) fn residuals(
        &self,
        s: &[T],
        z: &[T],
        cones: &[SupportedConeT<T>],
        psd_dual: bool,
    ) -> OriginalResiduals<T> {
        debug_assert!(s.len() == self.b.len() && z.len() == self.b.len());
        let infbound: T = crate::get_infinity().as_T();
        let finite = |v: T| T::abs(v) < infbound;
        let r = &self.r;

        let mut primal = T::zero();
        for (i, (&ri, &si)) in r.iter().zip(s).enumerate() {
            if finite(self.b[i]) {
                primal = T::max(primal, T::abs(ri - si));
            }
        }
        let mut offset = 0;
        let mut dual_cone = T::zero();
        for cone in cones {
            let len = cone.nvars();
            let rows = offset..offset + len;
            offset += len;
            if rows.clone().any(|i| !finite(self.b[i])) {
                continue;
            }
            primal = T::max(primal, cone_distance(cone, &s[rows.clone()], false));
            primal = T::max(primal, cone_distance(cone, &r[rows.clone()], false));
            if psd_dual || !matches!(cone, SupportedConeT::PSDTriangleConeT(_)) {
                dual_cone = T::max(dual_cone, cone_distance(cone, &z[rows], true));
            }
        }

        let objective = self.qx + self.xpx / (2.).as_T();
        let dual_objective = -self.bz - self.xpx / (2.).as_T();
        let gap_abs = T::abs(objective - dual_objective);
        OriginalResiduals {
            primal: primal / (T::one() + self.normb),
            dual: T::max(self.rd.norm_inf(), dual_cone) / (T::one() + self.normq),
            gap_abs,
            gap_rel: gap_abs / (T::one() + T::abs(objective)),
        }
    }
}

impl<T: FloatT> OriginalData<T> {
    /// Replace `s` by the projection of `b - A x` onto the cone, cone by
    /// cone, so the returned slack is consistent with `x`: `‖b-Ax-s‖` is the
    /// distance of `b - Ax` from the cone. Zero, orthant and second-order
    /// cones project in closed form; PSD cones through an eigendecomposition
    /// in binary64 (higher precisions keep the iterate unless `b - Ax` is
    /// already PSD). Other cones and rows with an infinite bound keep `s`.
    pub(crate) fn project_slack(&self, x: &[T], s: &mut [T]) {
        let mut r = self.b.clone();
        if let Some(op) = &self.sampled {
            op.apply(
                &mut r,
                x,
                -T::one(),
                T::one(),
                &mut crate::solver::sampled::SampledWorkspace::new(op),
            );
        } else {
            self.A.gemv(&mut r, x, -T::one(), T::one());
        }
        project_slack(&r, &self.b, &self.cones, s);
    }
}

impl<T: FloatT> OriginalProducts<T> {
    pub(crate) fn project_slack(&self, cones: &[SupportedConeT<T>], s: &mut [T]) {
        project_slack(&self.r, &self.b, cones, s);
    }
}

fn project_slack<T: FloatT>(r: &[T], b: &[T], cones: &[SupportedConeT<T>], s: &mut [T]) {
    let infbound: T = crate::get_infinity().as_T();
    let mut offset = 0;
    for cone in cones {
        let len = cone.nvars();
        let rows = offset..offset + len;
        offset += len;
        if rows.clone().any(|i| T::abs(b[i]) >= infbound) {
            continue;
        }
        let (r, s) = (&r[rows.clone()], &mut s[rows]);
        match cone {
            SupportedConeT::ZeroConeT(_) => s.fill(T::zero()),
            SupportedConeT::NonnegativeConeT(_) => {
                for (si, &ri) in s.iter_mut().zip(r) {
                    *si = T::max(ri, T::zero());
                }
            }
            SupportedConeT::SecondOrderConeT(_) if !r.is_empty() => {
                let (t, norm) = (r[0], r[1..].norm());
                if norm <= t {
                    s.copy_from_slice(r);
                } else if norm <= -t {
                    s.fill(T::zero());
                } else {
                    let half = (t + norm) / (2.).as_T();
                    s[0] = half;
                    for (si, &ri) in s[1..].iter_mut().zip(&r[1..]) {
                        *si = half * ri / norm;
                    }
                }
            }
            SupportedConeT::PSDTriangleConeT(dim) => {
                if cone_distance(cone, r, false) == T::zero() {
                    s.copy_from_slice(r);
                } else if T::precision_bits() <= 64 {
                    project_psd_f64(*dim, r, s);
                }
            }
            _ => {}
        }
    }
}

/// `s = svec(V max(Λ, 0) V')` for `mat(r) = V Λ V'`, in binary64.
fn project_psd_f64<T: FloatT>(dim: usize, r: &[T], s: &mut [T]) {
    use crate::solver::cones::{svec_eigen_f64, svec_spectral_f64};
    let Some((a, w)) = svec_eigen_f64(r, dim) else {
        return;
    };
    let t: Vec<f64> = w.iter().map(|&l| if l > 0. { l } else { 0. }).collect();
    for (s, v) in s.iter_mut().zip(svec_spectral_f64(&a, dim, &t)) {
        *s = T::from_f64(v).unwrap();
    }
}

/// Distance of `v` from the cone (`dual`: from its dual cone), measured as
/// in the audit: `|v|∞` for the zero cone, the most negative entry for the
/// orthant, `‖v̄‖ - v₀` for second-order cones, `-λ_min` for PSD triangles.
/// Other cones are not transformed by presolve or chordal decomposition and
/// keep the interior iterate, so they report zero.
fn cone_distance<T: FloatT>(cone: &SupportedConeT<T>, v: &[T], dual: bool) -> T {
    match cone {
        SupportedConeT::ZeroConeT(_) => {
            if dual {
                T::zero()
            } else {
                v.norm_inf()
            }
        }
        SupportedConeT::NonnegativeConeT(_) => T::max(
            T::zero(),
            -v.iter().fold(T::infinity(), |a, &b| T::min(a, b)),
        ),
        SupportedConeT::SecondOrderConeT(_) => {
            if v.is_empty() {
                T::zero()
            } else {
                T::max(T::zero(), v[1..].norm() - v[0])
            }
        }
        SupportedConeT::PSDTriangleConeT(dim) => {
            let dim = *dim;
            if dim == 0 {
                return T::zero();
            }
            if !v.iter().all(|x| x.is_finite()) {
                return T::infinity();
            }
            let mut mat = Matrix::<T>::zeros((dim, dim));
            svec_to_mat(&mut mat, v);
            let mut engine = EigEngine::<T>::new(dim);
            // The workspace query writes its size into work[0].
            let mut work = vec![T::zero()];
            match engine.eigval_min(&mut mat, &mut work) {
                Ok(lmin) => T::max(T::zero(), -lmin),
                Err(_) => T::infinity(),
            }
        }
        _ => T::zero(),
    }
}
