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
        debug_assert!(k.m == k.n && signs.len() == k.n);
        let n = signs.iter().take_while(|&&s| s == 1).count();
        let m = k.n - n;
        if n < 128 || signs[n..].iter().any(|&s| s != -1) {
            return None;
        }
        let elements = (n * n + n * m + m * m) as u128;
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
        // Every blocked kernel runs the same tiles whatever the worker count
        // (a single-worker pool when there is no pool), so the factor does
        // not depend on the number of threads.
        let pool = self
            .pool
            .as_deref()
            .filter(|pool| pool.current_num_threads() > 1);
        let blocked = pool.unwrap_or(serial_pool());
        let rule = self.pivot_rule();
        if !factor_block(&mut self.h, n, blocked, &mut self.tiles, rule)
            || !self.valid_factor(&self.h, n)
        {
            return false;
        }
        if m == 0 {
            return true;
        }
        // Columns of Y are independent right-hand sides, solved in fixed
        // TILE-column chunks (split across the pool when there is one).
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
            Some(pool) => pool.install(|| self.y.par_chunks_mut(TILE * n).for_each(trsm)),
            None => self.y.chunks_mut(TILE * n).for_each(trsm),
        }
        // S += Y'Y: upper column blocks of S are independent.
        if m >= 2 * TILE {
            let y = &self.y;
            let block = |(b, cols): (usize, &mut [f64])| {
                let (c0, w) = (b * TILE, cols.len() / m);
                // Rows 0..c0 of this block, then its diagonal tile.
                tile_gemm(c0, w, n, &y[..c0 * n], n, &y[c0 * n..], n, cols, m);
                tile_syrk(w, n, &y[c0 * n..], n, &mut cols[c0..], m);
            };
            match pool {
                Some(pool) => {
                    pool.install(|| self.s.par_chunks_mut(TILE * m).enumerate().for_each(block))
                }
                None => self.s.chunks_mut(TILE * m).enumerate().for_each(block),
            }
        } else {
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
            }
        }
        let mut tiles = Vec::new();
        if !factor_block(&mut self.s, m, blocked, &mut tiles, rule) {
            return false;
        }
        self.y.iter().all(|v| v.is_finite()) && self.valid_factor(&self.s, m)
    }

    /// `(eps, delta)` of the dynamic pivot rule, when enabled: a Cholesky
    /// pivot at or below `eps` is replaced by `delta`, as in the sparse LDL.
    fn pivot_rule(&self) -> Option<(f64, f64)> {
        self.settings.dynamic_regularization_enable.then_some((
            self.settings.dynamic_regularization_eps,
            self.settings.dynamic_regularization_delta,
        ))
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

/// Cholesky of the upper triangle of `a` (n x n): tiled on `pool` from
/// `2 * TILE`, where the tile boundaries alone fix the arithmetic, so any
/// worker count gives the same factor; LAPACK below that. A pivot rejected by
/// LAPACK reruns tiled with the dynamic pivot rule.
fn factor_block(
    a: &mut [f64],
    n: usize,
    pool: &rayon::ThreadPool,
    tiles: &mut Vec<Vec<Vec<f64>>>,
    rule: Option<(f64, f64)>,
) -> bool {
    if n >= 2 * TILE {
        return tiled_potrf(a, n, pool, tiles, rule);
    }
    let saved = rule.map(|_| a.to_vec());
    let mut info = 0;
    unsafe { lapack::dpotrf(b'U', n as i32, a, n as i32, &mut info) };
    let rejected = info != 0
        || !a.iter().all(|v| v.is_finite())
        || rule.is_some_and(|(eps, _)| (0..n).any(|i| a[i + i * n] * a[i + i * n] <= eps));
    match saved {
        Some(saved) if rejected => {
            a.copy_from_slice(&saved);
            tiled_potrf(a, n, serial_pool(), tiles, rule)
        }
        _ => info == 0,
    }
}

/// Single-worker pool for the blocked kernels when no pool is set.
fn serial_pool() -> &'static rayon::ThreadPool {
    static POOL: std::sync::OnceLock<rayon::ThreadPool> = std::sync::OnceLock::new();
    POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("serial pool")
    })
}

/// Unblocked upper Cholesky of the `n x n` leading part of `a` (leading
/// dimension `lda`) with the dynamic pivot rule: a pivot at or below `eps`
/// becomes `delta` when it is within rounding of zero relative to the
/// unfactored diagonal `diag`. A clearly negative pivot (indefinite block)
/// or a non-finite one fails, leaving the sparse fallback in charge.
fn potrf_dynamic(
    a: &mut [f64],
    n: usize,
    lda: usize,
    (eps, delta): (f64, f64),
    diag: &[f64],
) -> bool {
    for k in 0..n {
        let mut d = a[k + k * lda];
        if !d.is_finite() {
            return false;
        }
        if d <= eps {
            if d < -64. * f64::EPSILON * (n as f64) * diag[k].abs() {
                return false;
            }
            d = delta;
        }
        let r = d.sqrt();
        a[k + k * lda] = r;
        for j in k + 1..n {
            a[k + j * lda] /= r;
        }
        for j in k + 1..n {
            let akj = a[k + j * lda];
            for i in k + 1..=j {
                a[i + j * lda] -= a[k + i * lda] * akj;
            }
        }
    }
    true
}

/// Factor one diagonal tile: LAPACK, then the dynamic rule on a rejected
/// pivot. Without the rule a failed or nonpositive pivot fails the tile.
fn potrf_tile(t: &mut [f64], n: usize, rule: Option<(f64, f64)>, diag: &[f64]) -> bool {
    let saved = rule.map(|_| t.to_vec());
    let mut info = 0;
    unsafe { lapack::dpotrf(b'U', n as i32, t, n as i32, &mut info) };
    let Some((eps, delta)) = rule else {
        return info == 0;
    };
    if info == 0 && (0..n).all(|i| t[i + i * n] * t[i + i * n] > eps) {
        return true;
    }
    t.copy_from_slice(&saved.unwrap());
    potrf_dynamic(t, n, n, (eps, delta), diag)
}

/// `C(m x n) -= A' B` with `A` (k x m) and `B` (k x n) column-major.
#[allow(clippy::too_many_arguments)]
fn tile_gemm_sub(
    m: usize,
    n: usize,
    k: usize,
    a: &[f64],
    lda: usize,
    b: &[f64],
    ldb: usize,
    c: &mut [f64],
    ldc: usize,
) {
    tile_product(m, n, k, a, lda, b, ldb, c, ldc, -1.)
}

/// `C(m x n) += A' B` (border update `S += Y'Y`, off-diagonal part).
#[allow(clippy::too_many_arguments)]
fn tile_gemm(
    m: usize,
    n: usize,
    k: usize,
    a: &[f64],
    lda: usize,
    b: &[f64],
    ldb: usize,
    c: &mut [f64],
    ldc: usize,
) {
    tile_product(m, n, k, a, lda, b, ldb, c, ldc, 1.)
}

/// Upper `C(n x n) += A' A` with `A` (k x n).
fn tile_syrk(n: usize, k: usize, a: &[f64], lda: usize, c: &mut [f64], ldc: usize) {
    if n == 0 || k == 0 {
        return;
    }
    unsafe {
        blas::dsyrk(
            b'U', b'T', n as i32, k as i32, 1., a, lda as i32, 1., c, ldc as i32,
        )
    };
}

/// Tile products through the linked BLAS. faer tile products (tried for
/// OpenBLAS contention at 64 workers) changed the rounding enough that the
/// medium Float64 SDP stalled at `AlmostSolved`/28 instead of `Solved`/21;
/// with BLAS the tiled factor matches LAPACK's `dpotrf` result.
#[allow(clippy::too_many_arguments)]
fn tile_product(
    m: usize,
    n: usize,
    k: usize,
    a: &[f64],
    lda: usize,
    b: &[f64],
    ldb: usize,
    c: &mut [f64],
    ldc: usize,
    alpha: f64,
) {
    if m == 0 || n == 0 || k == 0 {
        return;
    }
    assert!((k - 1) + (m - 1) * lda < a.len() && (k - 1) + (n - 1) * ldb < b.len());
    assert!((m - 1) + (n - 1) * ldc < c.len());
    unsafe {
        blas::dgemm(
            b'T', b'N', m as i32, n as i32, k as i32, alpha, a, lda as i32, b, ldb as i32, 1., c,
            ldc as i32,
        );
    }
}

/// Right-looking tiled Cholesky `A = U'U` of the upper triangle of the
/// column-major `a` (n x n), run on `pool`. Tiles are packed into separate
/// buffers so lanes borrow disjoint storage. Returns false on a failed pivot.
fn tiled_potrf(
    a: &mut [f64],
    n: usize,
    pool: &rayon::ThreadPool,
    tiles: &mut Vec<Vec<Vec<f64>>>,
    rule: Option<(f64, f64)>,
) -> bool {
    let p = n.div_ceil(TILE);
    let size = |b: usize| TILE.min(n - b * TILE);
    let diag: Vec<f64> = (0..n).map(|i| a[i + i * n]).collect();
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
            if !potrf_tile(&mut tiles[k][k], kk as usize, rule, &diag[k * TILE..]) {
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
                        if i == j {
                            unsafe { blas::dsyrk(b'U', b'T', cj, kk, -1., akj, kk, 1., tile, cj) };
                        } else {
                            let (ri, cj, kk) = (ri as usize, cj as usize, kk as usize);
                            tile_gemm_sub(ri, cj, kk, &panel[s], kk, akj, kk, tile, ri);
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
    fn dense_block_factor_is_thread_invariant() {
        // Leading block above the tiling threshold and a border above it too:
        // with and without a pool the factor runs the same tiles.
        let (n, m) = (2 * TILE + 9, 2 * TILE + 3);
        let mut seed = 0x2545f4914f6cdd1du64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        let total = n + m;
        let (mut rows, mut cols, mut vals) = (Vec::new(), Vec::new(), Vec::new());
        for j in 0..total {
            for i in 0..=j {
                let v = if i == j {
                    if j < n {
                        n as f64
                    } else {
                        -(m as f64)
                    }
                } else if (i < n) == (j < n) || i < n {
                    next() * 0.1
                } else {
                    0.
                };
                rows.push(i);
                cols.push(j);
                vals.push(v);
            }
        }
        let k = CscMatrix::new_from_triplets(total, total, rows, cols, vals);
        let signs: Vec<i8> = (0..total).map(|i| if i < n { 1 } else { -1 }).collect();
        let settings = CoreSettings::default();
        let run = |threads: usize| {
            let mut s = DenseBlockSolver::new(&k, &signs, &settings, n);
            if threads > 1 {
                s.set_pool(Some(Arc::new(
                    rayon::ThreadPoolBuilder::new()
                        .num_threads(threads)
                        .build()
                        .unwrap(),
                )));
            }
            assert!(s.factor_dense(&k));
            (s.h, s.y, s.s)
        };
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        let (h1, y1, s1) = run(1);
        for threads in [2, 5] {
            let (h, y, s) = run(threads);
            assert_eq!(bits(&h1), bits(&h));
            assert_eq!(bits(&y1), bits(&y));
            assert_eq!(bits(&s1), bits(&s));
        }
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
            assert!(tiled_potrf(&mut u, n, &pool, &mut Vec::new(), None));
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
        assert!(!tiled_potrf(
            &mut indefinite,
            n,
            &pool,
            &mut Vec::new(),
            None
        ));
    }
    #[test]
    fn dense_block_replaces_tiny_pivot() {
        // A tiny pivot takes the dynamic rule inside the dense factor
        // (as in the sparse LDL) instead of forcing the sparse fallback.
        let k = CscMatrix::new(1, 1, vec![0, 1], vec![0], vec![1e-30]);
        let settings = CoreSettings::default();
        let mut s = DenseBlockSolver::new(&k, &[1], &settings, 1);
        assert!(s.factor_dense(&k));
        assert_eq!(s.h[0], settings.dynamic_regularization_delta.sqrt());
        // Without the rule it is still rejected.
        let mut off = settings.clone();
        off.dynamic_regularization_enable = false;
        let k = CscMatrix::new(1, 1, vec![0, 1], vec![0], vec![-1e-30]);
        let mut s = DenseBlockSolver::new(&k, &[1], &off, 1);
        assert!(!s.factor_dense(&k));
    }

    #[test]
    fn tiled_dynamic_pivot_matches_serial_rerun() {
        // Rank-deficient PSD H: rounding-level pivots get the rule in both
        // the pooled tiles and the serial rerun, with the same factor.
        let n = 2 * TILE + 7;
        let r = 40;
        let g: Vec<f64> = (0..n * r)
            .map(|i| ((i * 37 % 101) as f64 - 50.) / 50.)
            .collect();
        let mut h = vec![0.; n * n];
        for j in 0..n {
            for i in 0..=j {
                h[i + j * n] = (0..r).map(|t| g[i * r + t] * g[j * r + t]).sum();
            }
        }
        let rule = Some((1e-13, 2e-7));
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        let mut a = h.clone();
        assert!(tiled_potrf(&mut a, n, &pool, &mut Vec::new(), rule));
        let mut b = h.clone();
        assert!(tiled_potrf(&mut b, n, serial_pool(), &mut Vec::new(), rule));
        assert_eq!(a, b);
        assert!((0..n).all(|i| a[i + i * n] > 0.));
    }
}
