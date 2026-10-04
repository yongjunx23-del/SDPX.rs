//! Dense quasi-definite block elimination using the configured BLAS/LAPACK.
//! K = [H B; B' -C], H = L L', Y = L^-1 B, S = C + Y'Y.
use crate::algebra::{CscMatrix, VectorMath};
use crate::solver::{
    core::CoreSettings,
    kkt::{
        direct::{BoxedDirectLDLSolver, DirectLDLSolver},
        HasLinearSolverInfo, LinearSolverInfo,
    },
};
use rayon::prelude::*;
use std::sync::Arc;

pub(super) struct DenseBlockSolver {
    nnz: usize,
    signs: Vec<i8>,
    settings: CoreSettings<f64>,
    n: usize,
    m: usize,
    h: Vec<f64>,
    y: Vec<f64>,
    s: Vec<f64>,
    fallback: Option<BoxedDirectLDLSolver<f64>>,
    use_dense: bool,
    packed_leading: bool,
    pool: Option<Arc<rayon::ThreadPool>>,
    // Pooled factorization tiles, kept across refactors (fully overwritten).
    tiles: Vec<Vec<Vec<f64>>>,
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
            nnz: k.nzval.len(),
            signs: signs.to_vec(),
            settings: settings.clone(),
            n,
            m,
            h: vec![0.; n * n],
            y: vec![0.; n * m],
            s: vec![0.; m * m],
            fallback: None,
            use_dense: false,
            packed_leading: k.colptr[n] == n * (n + 1) / 2,
            pool: None,
            tiles: Vec::new(),
        }
    }

    fn factor_dense(&mut self, k: &CscMatrix<f64>) -> bool {
        let (n, m) = (self.n, self.m);
        // Only the upper triangle of H and S is read (`U` factorizations and
        // solves), so clear just the columns' upper parts as they are filled.
        for j in 0..n {
            self.h[j * n..j * n + j + 1].fill(0.);
        }
        self.y.fill(0.);
        for j in 0..m {
            self.s[j * m..j * m + j + 1].fill(0.);
        }
        for j in 0..k.n {
            for p in k.colptr[j]..k.colptr[j + 1] {
                let i = k.rowval[p];
                let v = k.nzval[p];
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
        let pool = self
            .pool
            .as_deref()
            .filter(|pool| pool.current_num_threads() > 1 && n >= 2 * TILE);
        let mut info = 0;
        match pool {
            Some(pool) => {
                if !tiled_potrf(&mut self.h, n, pool, &mut self.tiles) {
                    return false;
                }
            }
            None => unsafe {
                lapack::dpotrf(b'U', n as i32, &mut self.h, n as i32, &mut info);
            },
        }
        if info != 0 || !self.valid_factor(&self.h, n) {
            return false;
        }
        if m == 0 {
            return true;
        }
        // Columns of Y are independent right-hand sides: split them across
        // the pool. Each column's triangular solve is unchanged.
        let h = &self.h;
        let trsm = |y: &mut [f64]| unsafe {
            blas::dtrsm(
                b'L',
                b'U',
                b'T',
                b'N',
                n as i32,
                (y.len() / n) as i32,
                1.,
                h,
                n as i32,
                y,
                n as i32,
            );
        };
        match pool {
            Some(pool) => {
                let per = m.div_ceil(pool.current_num_threads()).max(1);
                pool.install(|| self.y.par_chunks_mut(per * n).for_each(trsm));
            }
            None => trsm(&mut self.y),
        }
        unsafe {
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

    fn factor_fallback(&mut self, k: &CscMatrix<f64>) -> bool {
        if self.fallback.is_none() {
            #[cfg(feature = "faer-sparse")]
            let solver = super::auto::ldl_auto_select(k, &self.signs, &self.settings);
            #[cfg(not(feature = "faer-sparse"))]
            let solver: BoxedDirectLDLSolver<f64> = Box::new(
                super::qdldl::QDLDLDirectLDLSolver::new(k, &self.signs, &self.settings, None),
            );
            self.fallback = Some(solver);
        }
        let solver = self.fallback.as_mut().unwrap();
        solver.set_pool(self.pool.clone());
        // Fallback can be dormant across many dense updates. Synchronize all
        // current values, including the caller's temporary static shift.
        let indices: Vec<_> = (0..k.nzval.len()).collect();
        solver.update_values(&indices, &k.nzval);
        solver.refactor(k)
    }
}

/// Tile width of the pooled Cholesky. Tile boundaries depend only on `n`, and
/// every tile operation is one BLAS call, so factors do not depend on the
/// number of workers.
const TILE: usize = 128;

/// Right-looking tiled Cholesky `A = U'U` of the upper triangle of the
/// column-major `a` (n x n), run on `pool`. Tiles are packed into separate
/// buffers so lanes borrow disjoint storage. Returns false on a failed pivot.
fn tiled_potrf(
    a: &mut [f64],
    n: usize,
    pool: &rayon::ThreadPool,
    tiles: &mut Vec<Vec<Vec<f64>>>,
) -> bool {
    let p = n.div_ceil(TILE);
    let size = |b: usize| TILE.min(n - b * TILE);
    // tiles[j][i], i <= j: rows of block i, columns of block j, ld = size(i).
    // Buffers are reused across calls; packing overwrites every element.
    tiles.resize_with(p, Vec::new);
    for (j, column) in tiles.iter_mut().enumerate() {
        column.resize_with(j + 1, Vec::new);
        for (i, t) in column.iter_mut().enumerate() {
            let (ri, cj) = (size(i), size(j));
            t.resize(ri * cj, 0.);
            for c in 0..cj {
                let src = (j * TILE + c) * n + i * TILE;
                t[c * ri..][..ri].copy_from_slice(&a[src..src + ri]);
            }
        }
    }
    let ok = pool.install(|| {
        for k in 0..p {
            let kk = size(k) as i32;
            let mut info = 0;
            unsafe { lapack::dpotrf(b'U', kk, &mut tiles[k][k], kk, &mut info) };
            if info != 0 {
                return false;
            }
            // Panel row k: U_kk' X = A_kj for every later block column.
            let (done, rest) = tiles.split_at_mut(k + 1);
            let diag = &done[k][k];
            rest.par_iter_mut().enumerate().for_each(|(r, column)| {
                let cj = size(k + 1 + r) as i32;
                unsafe {
                    blas::dtrsm(
                        b'L',
                        b'U',
                        b'T',
                        b'N',
                        kk,
                        cj,
                        1.,
                        diag,
                        kk,
                        &mut column[k],
                        kk,
                    )
                };
            });
            // Trailing update A_ij -= A_ki' A_kj, k < i <= j.
            let panel: Vec<Vec<f64>> = rest.iter_mut().map(|c| std::mem::take(&mut c[k])).collect();
            rest.par_iter_mut().enumerate().for_each(|(r, column)| {
                let j = k + 1 + r;
                let cj = size(j) as i32;
                let akj = &panel[r];
                column[k + 1..=j]
                    .par_iter_mut()
                    .enumerate()
                    .for_each(|(s, tile)| {
                        let i = k + 1 + s;
                        let ri = size(i) as i32;
                        unsafe {
                            if i == j {
                                blas::dsyrk(b'U', b'T', cj, kk, -1., akj, kk, 1., tile, cj);
                            } else {
                                blas::dgemm(
                                    b'T', b'N', ri, cj, kk, -1., &panel[s], kk, akj, kk, 1., tile,
                                    ri,
                                );
                            }
                        }
                    });
            });
            for (column, tile) in rest.iter_mut().zip(panel) {
                column[k] = tile;
            }
        }
        true
    });
    if !ok {
        return false;
    }
    for (j, column) in tiles.iter().enumerate() {
        for (i, t) in column.iter().enumerate() {
            let (ri, cj) = (size(i), size(j));
            for c in 0..cj {
                let dst = (j * TILE + c) * n + i * TILE;
                a[dst..dst + ri].copy_from_slice(&t[c * ri..][..ri]);
            }
        }
    }
    true
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
            threads: self.pool.as_ref().map_or(1, |pool| {
                if self.n >= 2 * TILE {
                    pool.current_num_threads()
                } else {
                    1
                }
            }),
            direct: true,
            nnzA: self.nnz,
            nnzL: self.n * (self.n + 1) / 2 + self.n * self.m + self.m * (self.m + 1) / 2,
        }
    }
}
impl DirectLDLSolver<f64> for DenseBlockSolver {
    // Refactor reads the caller's complete KKT, including temporary shifts.
    // A dormant sparse fallback is synchronized there before factorization.
    fn update_values(&mut self, _indices: &[usize], _values: &[f64]) {}
    fn scale_values(&mut self, _indices: &[usize], _scale: f64) {}
    fn set_pool(&mut self, pool: Option<Arc<rayon::ThreadPool>>) {
        if let Some(solver) = &mut self.fallback {
            solver.set_pool(pool.clone());
        }
        self.pool = pool;
    }
    fn residual(
        &self,
        k: &CscMatrix<f64>,
        out: &mut [f64],
        rhs: &[f64],
        point: &[f64],
    ) -> Option<f64> {
        if !self.packed_leading {
            return None;
        }
        out.copy_from_slice(rhs);
        let n = self.n;
        unsafe {
            blas::dspmv(
                b'U',
                n as i32,
                -1.,
                &k.nzval[..k.colptr[n]],
                &point[..n],
                1,
                1.,
                &mut out[..n],
                1,
            );
        }
        for col in n..k.n {
            for p in k.colptr[col]..k.colptr[col + 1] {
                let row = k.rowval[p];
                let value = -k.nzval[p];
                out[row] = value.mul_add(point[col], out[row]);
                if row != col {
                    out[col] = value.mul_add(point[row], out[col]);
                }
            }
        }
        Some(out.norm_inf())
    }

    fn refactor(&mut self, k: &CscMatrix<f64>) -> bool {
        self.use_dense = self.factor_dense(k);
        self.use_dense || self.factor_fallback(k)
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
    fn check(s: &mut DenseBlockSolver, k: &CscMatrix<f64>) {
        assert!(s.refactor(k));
        let rhs = vec![1., -2., 3., 4.];
        let mut b = rhs.clone();
        let mut x = vec![0.; 4];
        // Caller restores original diagonals after refactor; factors must
        // still represent the shifted matrix passed to refactor.
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
        let mut k = CscMatrix::new(
            4,
            4,
            vec![0, 1, 3, 6, 10],
            vec![0, 0, 1, 0, 1, 2, 0, 1, 2, 3],
            vec![4., 1., 3., 1., 2., -3., 2., -1., -0.5, -2.],
        );
        let mut s = DenseBlockSolver::new(&k, &[1, 1, -1, -1], &CoreSettings::default(), 2);
        check(&mut s, &k);
        assert!(s.use_dense);
        k.nzval[0] *= 2.;
        s.scale_values(&[0], 2.);
        k.nzval[2] += 0.5;
        s.update_values(&[2], &[k.nzval[2]]);
        check(&mut s, &k);
        // H indefinite but full K invertible: sparse fallback and later recovery.
        k.nzval[0] = -4.;
        s.update_values(&[0], &[-4.]);
        assert!(!s.factor_dense(&k));
        assert!(s.refactor(&k));
        assert!(!s.use_dense);
        k.nzval[0] = 5.;
        s.update_values(&[0], &[5.]);
        check(&mut s, &k);
        assert!(s.use_dense);
        k.nzval[0] = -5.;
        s.update_values(&[0], &[-5.]);
        k.nzval[3] *= 0.5;
        s.scale_values(&[3], 0.5);
        k.nzval[9] -= 1.;
        s.update_values(&[9], &[k.nzval[9]]);
        assert!(s.refactor(&k));
        assert!(!s.use_dense);
        let mut fresh = DenseBlockSolver::new(&k, &s.signs, &s.settings, 2);
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
    fn tiled_potrf_matches_lapack_and_is_thread_invariant() {
        let n = 2 * TILE + 45;
        let mut seed = 0x9e3779b97f4a7c15u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        let b: Vec<f64> = (0..n * n).map(|_| next()).collect();
        let mut a = vec![0.; n * n];
        for j in 0..n {
            for i in 0..=j {
                a[i + j * n] = (0..n).map(|k| b[k + i * n] * b[k + j * n]).sum::<f64>()
                    + if i == j { n as f64 } else { 0. };
            }
        }
        let mut reference = a.clone();
        let mut info = 0;
        unsafe { lapack::dpotrf(b'U', n as i32, &mut reference, n as i32, &mut info) };
        assert_eq!(info, 0);
        let factor = |threads| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            let mut u = a.clone();
            assert!(tiled_potrf(&mut u, n, &pool, &mut Vec::new()));
            u
        };
        let (two, five) = (factor(2), factor(5));
        assert!(two
            .iter()
            .zip(&five)
            .all(|(x, y)| x.to_bits() == y.to_bits()));
        for j in 0..n {
            for i in 0..=j {
                let (x, y) = (two[i + j * n], reference[i + j * n]);
                assert!(
                    (x - y).abs() <= 1e-12 * y.abs().max(1.),
                    "({i},{j}) {x} vs {y}"
                );
            }
        }
        let mut indefinite = a.clone();
        indefinite[(n - 1) * (n + 1)] = -1e6;
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(3)
            .build()
            .unwrap();
        assert!(!tiled_potrf(&mut indefinite, n, &pool, &mut Vec::new()));
    }
    #[test]
    fn dense_block_rejects_tiny_pivot() {
        let k = CscMatrix::new(1, 1, vec![0, 1], vec![0], vec![1e-30]);
        let mut s = DenseBlockSolver::new(&k, &[1], &CoreSettings::default(), 1);
        assert!(!s.factor_dense(&k));
    }
}
