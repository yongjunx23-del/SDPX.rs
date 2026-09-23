//! Dense quasi-definite block elimination using the configured BLAS/LAPACK.
//! K = [H B; B' -C], H = L L', Y = L^-1 B, S = C + Y'Y.
use crate::algebra::{CscMatrix, MatrixTriangle};
use crate::solver::{
    core::CoreSettings,
    kkt::{
        direct::{BoxedDirectLDLSolver, DirectLDLSolver, DirectLDLSolverReqs},
        HasLinearSolverInfo, LinearSolverInfo,
    },
};

pub(super) struct DenseBlockSolver {
    matrix: CscMatrix<f64>,
    signs: Vec<i8>,
    settings: CoreSettings<f64>,
    n: usize,
    m: usize,
    h: Vec<f64>,
    y: Vec<f64>,
    s: Vec<f64>,
    fallback: Option<BoxedDirectLDLSolver<f64>>,
    use_dense: bool,
}

impl DenseBlockSolver {
    pub(super) fn try_new(
        k: &CscMatrix<f64>,
        signs: &[i8],
        settings: &CoreSettings<f64>,
    ) -> Option<Self> {
        if k.m != k.n || signs.len() != k.n {
            return None;
        }
        let n = signs.iter().take_while(|&&s| s == 1).count();
        let m = k.n - n;
        if n < 128 || signs[n..].iter().any(|&s| s != -1) {
            return None;
        }
        let elements = n
            .checked_mul(n)?
            .checked_add(n.checked_mul(m)?)?
            .checked_add(m.checked_mul(m)?)?;
        if elements > 256 * 1024 * 1024 || k.n > i32::MAX as usize {
            return None;
        }
        // Structural admission only: the values are populated after
        // construction, and a failed dense factorization falls back to the
        // sparse solver, so over-admission costs memory, not correctness.
        // Dense block elimination wins when the matrix is mostly dense
        // (scalar sparse elimination pays index overhead on every entry) or
        // the tail is small and the leading block is dense (classic QP form).
        let leading_fill = k.colptr[n] as f64 / ((n * (n + 1) / 2) as f64);
        let total_fill = k.colptr[k.n] as f64 / ((k.n as f64) * (k.n as f64 + 1.) / 2.);
        if !(total_fill >= 0.4 || (m <= 256 && m <= n / 4 && leading_fill >= 0.8)) {
            return None;
        }
        Some(Self::new(k, signs, settings, n))
    }

    fn new(k: &CscMatrix<f64>, signs: &[i8], settings: &CoreSettings<f64>, n: usize) -> Self {
        let m = k.n - n;
        Self {
            matrix: k.clone(),
            signs: signs.to_vec(),
            settings: settings.clone(),
            n,
            m,
            h: vec![0.; n * n],
            y: vec![0.; n * m],
            s: vec![0.; m * m],
            fallback: None,
            use_dense: false,
        }
    }

    fn factor_dense(&mut self) -> bool {
        let (n, m) = (self.n, self.m);
        self.h.fill(0.);
        self.y.fill(0.);
        self.s.fill(0.);
        for j in 0..self.matrix.n {
            for p in self.matrix.colptr[j]..self.matrix.colptr[j + 1] {
                let i = self.matrix.rowval[p];
                let v = self.matrix.nzval[p];
                if !v.is_finite() {
                    return false;
                }
                if j < n {
                    self.h[i + j * n] = v;
                } else if i < n {
                    self.y[i + (j - n) * n] = v;
                } else {
                    self.s[i - n + (j - n) * m] = -v;
                }
            }
        }
        let mut info = 0;
        unsafe {
            lapack::dpotrf(b'U', n as i32, &mut self.h, n as i32, &mut info);
        }
        if info != 0 || !self.valid_factor(&self.h, n) {
            return false;
        }
        if m == 0 {
            return true;
        }
        unsafe {
            blas::dtrsm(
                b'L',
                b'U',
                b'T',
                b'N',
                n as i32,
                m as i32,
                1.,
                &self.h,
                n as i32,
                &mut self.y,
                n as i32,
            );
            blas::dsyrk(
                b'U',
                b'T',
                m as i32,
                n as i32,
                1.,
                &self.y,
                n as i32,
                1.,
                &mut self.s,
                m as i32,
            );
            lapack::dpotrf(b'U', m as i32, &mut self.s, m as i32, &mut info);
        }
        info == 0 && self.y.iter().all(|v| v.is_finite()) && self.valid_factor(&self.s, m)
    }

    fn valid_factor(&self, a: &[f64], n: usize) -> bool {
        a.iter().all(|v| v.is_finite())
            && (0..n).all(|i| {
                let d = a[i + i * n];
                d > 0.
                    && (!self.settings.dynamic_regularization_enable
                        || d * d > self.settings.dynamic_regularization_eps)
            })
    }

    fn factor_fallback(&mut self) -> bool {
        if self.fallback.is_none() {
            #[cfg(feature = "faer-sparse")]
            let solver = super::auto::ldl_auto_select(&self.matrix, &self.signs, &self.settings);
            #[cfg(not(feature = "faer-sparse"))]
            let solver: BoxedDirectLDLSolver<f64> =
                Box::new(super::qdldl::QDLDLDirectLDLSolver::new(
                    &self.matrix,
                    &self.signs,
                    &self.settings,
                    None,
                ));
            self.fallback = Some(solver);
        }
        let solver = self.fallback.as_mut().unwrap();
        // Fallback can be dormant across many dense updates. Synchronize all
        // current values, including the caller's temporary static shift.
        let indices: Vec<_> = (0..self.matrix.nzval.len()).collect();
        solver.update_values(&indices, &self.matrix.nzval);
        solver.refactor(&self.matrix)
    }
}

impl DirectLDLSolverReqs for DenseBlockSolver {
    fn required_matrix_shape() -> MatrixTriangle {
        MatrixTriangle::Triu
    }
}
impl HasLinearSolverInfo for DenseBlockSolver {
    fn linear_solver_info(&self) -> LinearSolverInfo {
        if !self.use_dense {
            if let Some(solver) = &self.fallback {
                return solver.linear_solver_info();
            }
        }
        LinearSolverInfo {
            name: "dense_block".into(),
            threads: 1,
            direct: true,
            nnzA: self.matrix.nzval.len(),
            nnzL: self.n * (self.n + 1) / 2 + self.n * self.m + self.m * (self.m + 1) / 2,
        }
    }
}
impl DirectLDLSolver<f64> for DenseBlockSolver {
    fn update_values(&mut self, indices: &[usize], values: &[f64]) {
        for (&i, &v) in indices.iter().zip(values) {
            self.matrix.nzval[i] = v;
        }
    }
    fn scale_values(&mut self, indices: &[usize], scale: f64) {
        for &i in indices {
            self.matrix.nzval[i] *= scale;
        }
    }
    fn offset_values(&mut self, indices: &[usize], offset: f64, signs: &[i8]) {
        for (&i, &s) in indices.iter().zip(signs) {
            self.matrix.nzval[i] += offset * s as f64;
        }
    }
    fn refactor(&mut self, _k: &CscMatrix<f64>) -> bool {
        self.use_dense = self.factor_dense();
        self.use_dense || self.factor_fallback()
    }
    fn solve(&mut self, k: &CscMatrix<f64>, x: &mut [f64], b: &mut [f64]) {
        if !self.use_dense {
            self.fallback.as_mut().unwrap().solve(k, x, b);
            return;
        }
        x.copy_from_slice(b);
        let (u, v) = x.split_at_mut(self.n);
        let (n, m) = (self.n as i32, self.m as i32);
        unsafe {
            blas::dtrsv(b'U', b'T', b'N', n, &self.h, n, u, 1);
            if m > 0 {
                blas::dgemv(b'T', n, m, 1., &self.y, n, u, 1, -1., v, 1);
                blas::dtrsv(b'U', b'T', b'N', m, &self.s, m, v, 1);
                blas::dtrsv(b'U', b'N', b'N', m, &self.s, m, v, 1);
                blas::dgemv(b'N', n, m, -1., &self.y, n, v, 1, 1., u, 1);
            }
            blas::dtrsv(b'U', b'N', b'N', n, &self.h, n, u, 1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn check(s: &mut DenseBlockSolver) {
        let k = s.matrix.clone();
        assert!(s.refactor(&k));
        let rhs = vec![1., -2., 3., 4.];
        let mut b = rhs.clone();
        let mut x = vec![0.; 4];
        // Caller restores original diagonals after refactor; factors must
        // still represent the shifted matrix held inside the backend.
        let mut restored = k.clone();
        restored.nzval[0] -= 0.01;
        s.solve(&restored, &mut x, &mut b);
        assert_eq!(rhs, b);
        let mut ax = vec![0.; 4];
        for j in 0..4 {
            for p in k.colptr[j]..k.colptr[j + 1] {
                let i = k.rowval[p];
                let a = k.nzval[p];
                ax[i] += a * x[j];
                if i != j {
                    ax[j] += a * x[i];
                }
            }
        }
        for i in 0..4 {
            assert!((ax[i] - rhs[i]).abs() < 1e-10, "{ax:?} != {rhs:?}");
        }
    }
    #[test]
    fn dense_block_updates_and_fallback() {
        let k = CscMatrix::new(
            4,
            4,
            vec![0, 1, 3, 6, 10],
            vec![0, 0, 1, 0, 1, 2, 0, 1, 2, 3],
            vec![4., 1., 3., 1., 2., -3., 2., -1., -0.5, -2.],
        );
        let mut s = DenseBlockSolver::new(&k, &[1, 1, -1, -1], &CoreSettings::default(), 2);
        check(&mut s);
        assert!(s.use_dense);
        s.scale_values(&[0], 2.);
        s.offset_values(&[2], 0.5, &[1]);
        check(&mut s);
        // H indefinite but full K invertible: sparse fallback and later recovery.
        s.update_values(&[0], &[-4.]);
        assert!(!s.factor_dense());
        assert!(s.refactor(&k));
        assert!(!s.use_dense);
        s.update_values(&[0], &[5.]);
        check(&mut s);
        assert!(s.use_dense);
        s.update_values(&[0], &[-5.]);
        s.scale_values(&[3], 0.5);
        s.offset_values(&[9], 1., &[-1]);
        assert!(s.refactor(&k));
        assert!(!s.use_dense);
        let mut fresh = DenseBlockSolver::new(&s.matrix, &s.signs, &s.settings, 2);
        assert!(fresh.refactor(&k));
        let mut b = vec![1., 2., 3., 4.];
        let mut b2 = b.clone();
        let mut x = vec![0.; 4];
        let mut x2 = x.clone();
        s.solve(&k, &mut x, &mut b);
        fresh.solve(&k, &mut x2, &mut b2);
        assert_eq!(x, x2);
    }
    #[test]
    fn dense_block_rejects_tiny_pivot() {
        let k = CscMatrix::new(1, 1, vec![0, 1], vec![0], vec![1e-30]);
        let mut s = DenseBlockSolver::new(&k, &[1], &CoreSettings::default(), 1);
        assert!(!s.factor_dense());
    }
}
