#![allow(non_snake_case)]
use crate::algebra::*;
use rayon::prelude::*;

// MatrixMath<T> should be implemented for a more
// general type, e.g. the DenseStorageMatrix<S,T> type
// or similar.   That would provide math functionality
// for more types, e.g. statically size matrices or
// the ones on borrowed data.

impl<T: FloatT> MatrixMath<T> for Matrix<T> {
    fn col_sums(&self, sums: &mut [T]) {
        assert_eq!(self.ncols(), sums.len());
        for (col, sum) in sums.iter_mut().enumerate() {
            *sum = self.col_slice(col).sum();
        }
    }

    fn row_sums(&self, sums: &mut [T]) {
        assert_eq!(self.nrows(), sums.len());
        sums.fill(T::zero());
        for col in 0..self.ncols() {
            let slice = self.col_slice(col);
            for (row, &v) in slice.iter().enumerate() {
                sums[row] += v;
            }
        }
    }

    fn col_norms(&self, norms: &mut [T]) {
        norms.fill(T::zero());
        self.col_norms_no_reset(norms);
    }

    fn col_norms_no_reset(&self, norms: &mut [T]) {
        for (i, norm) in norms.iter_mut().enumerate() {
            let colnorm = self.col_slice(i).norm_inf();
            *norm = T::max(*norm, colnorm);
        }
    }

    fn col_norms_sym(&self, norms: &mut [T]) {
        norms.fill(T::zero());
        self.col_norms_sym_no_reset(norms);
    }

    fn col_norms_sym_no_reset(&self, norms: &mut [T]) {
        for c in 0..self.ncols() {
            for r in 0..=c {
                let tmp = self[(r, c)];
                norms[r] = T::max(norms[r], tmp);
                norms[c] = T::max(norms[c], tmp);
            }
        }
    }

    fn row_norms(&self, norms: &mut [T]) {
        norms.fill(T::zero());
        self.row_norms_no_reset(norms);
    }

    fn row_norms_no_reset(&self, norms: &mut [T]) {
        for r in 0..self.nrows() {
            for c in 0..self.ncols() {
                norms[r] = T::max(norms[r], T::abs(self[(r, c)]))
            }
        }
    }
}

impl<T: FloatT> MatrixMathMut<T> for Matrix<T> {
    //scalar mut operations
    fn scale(&mut self, c: T) {
        self.data.scale(c);
    }

    fn negate(&mut self) {
        self.data.negate();
    }

    fn lscale(&mut self, l: &[T]) {
        for col in 0..self.ncols() {
            self.col_slice_mut(col).hadamard(l);
        }
    }

    fn rscale(&mut self, r: &[T]) {
        for (col, val) in r.iter().enumerate() {
            self.col_slice_mut(col).scale(*val);
        }
    }

    fn lrscale(&mut self, l: &[T], r: &[T]) {
        for i in 0..self.nrows() {
            for j in 0..self.ncols() {
                self[(i, j)] *= l[i] * r[j];
            }
        }
    }
}

impl<T> Matrix<T>
where
    T: FloatT,
{
    #[cfg(test)]
    pub(crate) fn kron<MATA, MATB>(&mut self, A: &MATA, B: &MATB)
    where
        MATA: DenseMatrix<T>,
        MATB: DenseMatrix<T>,
    {
        let (pp, qq) = A.size();
        let (rr, ss) = B.size();
        assert!(self.nrows() == pp * rr);
        assert!(self.ncols() == qq * ss);

        let mut i = 0;
        for q in 0..qq {
            for s in 0..ss {
                for p in 0..pp {
                    let Apq = A[(p, q)];
                    for r in 0..rr {
                        self.data_mut()[i] = (Apq) * B[(r, s)];
                        i += 1;
                    }
                }
            }
        }
    }
}

// additional functions that require floating point operations

#[cfg(test)]
impl<S, T> DenseStorageMatrix<S, T>
where
    T: FloatT,
    S: AsRef<[T]> + AsMut<[T]>,
{
    /// Set A = (A + A') / 2.  Assumes A is real
    pub fn symmetric_part(&mut self) -> &mut Self {
        assert!(self.is_square());
        let half: T = (0.5_f64).as_T();

        for r in 0..self.nrows() {
            for c in 0..r {
                let val = half * (self[(r, c)] + self[(c, r)]);
                self[(c, r)] = val;
                self[(r, c)] = val;
            }
        }
        self
    }
}

pub(crate) fn svec_to_mat<S, T>(M: &mut DenseStorageMatrix<S, T>, x: &[T])
where
    T: FloatT,
    S: AsRef<[T]> + AsMut<[T]>,
{
    let scale = if M.ncols() > 1 {
        T::FRAC_1_SQRT_2()
    } else {
        T::zero()
    };
    let mut idx = 0;
    for col in 0..M.ncols() {
        for row in 0..=col {
            if row == col {
                M[(row, col)] = x[idx];
            } else {
                M[(row, col)] = x[idx] * scale;
                M[(col, row)] = M[(row, col)];
            }
            idx += 1;
        }
    }
}

// Perhaps implementation for Symmetric type would be faster
pub(crate) fn mat_to_svec<T, MATM>(x: &mut [T], M: &MATM)
where
    MATM: DenseMatrix<T>,
    T: FloatT,
{
    let scale = if M.ncols() > 1 {
        T::FRAC_1_SQRT_2()
    } else {
        T::zero()
    };
    let mut idx = 0;
    for col in 0..M.ncols() {
        for row in 0..=col {
            x[idx] = {
                if row == col {
                    M[(row, col)]
                } else {
                    (M[(row, col)] + M[(col, row)]) * scale
                }
            };
            idx += 1;
        }
    }
}

/// Exact symmetric congruence `c = a·x·aᵀ` (`aᵀ·x·a` with `transpose_a`),
/// rounded once, when the residue kernel applies; the upper triangle is
/// mirrored. `false` leaves `c` untouched for the caller's two-product path.
pub(crate) fn congruence_exact_sym<T: FloatT>(
    c: &mut Matrix<T>,
    a: &Matrix<T>,
    transpose_a: bool,
    x: &Matrix<T>,
    pool: Option<&rayon::ThreadPool>,
    cache_a: Option<&mut ResidueCache>,
) -> bool {
    let (m, k) = if transpose_a {
        (a.ncols(), a.nrows())
    } else {
        (a.nrows(), a.ncols())
    };
    let ta = if transpose_a { b'T' } else { b'N' };
    if !T::xcongruence_exact(
        ta,
        m,
        k,
        a.data(),
        a.nrows(),
        x.data(),
        x.nrows(),
        c.data_mut(),
        true,
        pool,
        cache_a,
    ) {
        return false;
    }
    for j in 0..m {
        for i in j + 1..m {
            c[(i, j)] = c[(j, i)];
        }
    }
    true
}

/// `a·b` when the exact-arithmetic product is symmetric. Only the upper
/// triangle is evaluated and then mirrored — the same ascending-k accumulation
/// order as a dense `mul`, so upper entries stay bitwise identical while the
/// lower half is an exact copy instead of independently rounded products.
pub(crate) fn pooled_gemm_sym<T: FloatT, MATA, MATB>(
    c: &mut Matrix<T>,
    a: &MATA,
    b: &MATB,
    gemm: Option<(&rayon::ThreadPool, usize)>,
) where
    MATA: DenseMatrix<T>,
    MATB: DenseMatrix<T>,
{
    pooled_gemm_sym_cached(c, a, b, gemm, None)
}

/// [`pooled_gemm_sym`] with `b` a constant operand whose residues may be kept
/// in `cache_b` across calls (the cache checks the operand's exact bits).
pub(crate) fn pooled_gemm_sym_cached<T: FloatT, MATA, MATB>(
    c: &mut Matrix<T>,
    a: &MATA,
    b: &MATB,
    gemm: Option<(&rayon::ThreadPool, usize)>,
    cache_b: Option<&mut ResidueCache>,
) where
    MATA: DenseMatrix<T>,
    MATB: DenseMatrix<T>,
{
    let (m, n, k) = (a.nrows(), b.ncols(), a.ncols());
    debug_assert_eq!(m, n);
    if T::precision_bits() <= 64 {
        // Primitive floats: exact-arithmetic-symmetric product via BLAS gemm,
        // then mirror the upper triangle so the result is exactly symmetric.
        c.mul(a, b, T::one(), T::zero());
        for j in 0..n {
            for i in j + 1..n {
                c[(i, j)] = c[(j, i)];
            }
        }
        return;
    }
    let (ta, tb) = (a.shape().as_blas_char(), b.shape().as_blas_char());
    let lda = if a.shape() == MatrixShape::N { m } else { k };
    let ldb = if b.shape() == MatrixShape::N { k } else { n };
    let (adata, bdata) = (a.data(), b.data());
    // Exact residue-BLAS product: same correctly rounded values as the
    // per-entry exact dots below.
    if T::xgemm_upper_exact(
        ta,
        tb,
        m,
        n,
        k,
        adata,
        lda,
        bdata,
        ldb,
        c.data_mut(),
        gemm.map(|(p, _)| p),
        cache_b,
    ) {
        for j in 0..n {
            for i in j + 1..n {
                c[(i, j)] = c[(j, i)];
            }
        }
        return;
    }
    let ae =
        |i: usize, p: usize| -> &T { &adata[if ta == b'N' { i + p * lda } else { p + i * lda }] };
    let be =
        |p: usize, j: usize| -> &T { &bdata[if tb == b'N' { p + j * ldb } else { j + p * ldb }] };
    let column = |j: usize, col: &mut [T]| {
        for i in 0..=j {
            col[i] = T::dot_fma((0..k).map(|p| (ae(i, p), be(p, j))));
        }
    };
    if let Some((pool, tiles)) = gemm.filter(|(p, t)| *t > 1 && p.current_num_threads() > 1) {
        if n > 1 {
            let tile = n.div_ceil(tiles.min(n));
            pool.install(|| {
                c.data_mut()
                    .par_chunks_mut(tile * m)
                    .enumerate()
                    .for_each(|(t, chunk)| {
                        let j0 = t * tile;
                        let j1 = (j0 + tile).min(n);
                        for (jc, j) in (j0..j1).enumerate() {
                            column(j, &mut chunk[jc * m..(jc + 1) * m]);
                        }
                    });
            });
        } else {
            for j in 0..n {
                column(j, &mut c.data_mut()[j * m..(j + 1) * m]);
            }
        }
    } else if let tasks @ 2.. = sdpx_arithmetic::inner_parallel::tasks(
        (n * (n + 1) / 2 * k.max(1)) as u128
            * sdpx_arithmetic::inner_parallel::weight(T::precision_bits()),
        n,
    ) {
        // No explicit pool, but a caller marked this as inner work: offer the
        // triangular columns to the ambient pool's idle workers when each task
        // carries a grain of work. Per-element accumulation order is
        // unchanged, so results stay bitwise identical.
        let tile = n.div_ceil(tasks);
        c.data_mut()
            .par_chunks_mut(tile * m)
            .enumerate()
            .for_each(|(t, chunk)| {
                let j0 = t * tile;
                let j1 = (j0 + tile).min(n);
                for (jc, j) in (j0..j1).enumerate() {
                    column(j, &mut chunk[jc * m..(jc + 1) * m]);
                }
            });
    } else {
        for j in 0..n {
            column(j, &mut c.data_mut()[j * m..(j + 1) * m]);
        }
    }
    for j in 0..n {
        for i in j + 1..n {
            c[(i, j)] = c[(j, i)];
        }
    }
}

#[test]
fn test_row_col_sums_and_norms() {
    #[rustfmt::skip]
    let A = Matrix::from(&[
        [-1.,  4.,  6.],
        [ 3., -8.,  7.],
        [ 0.,  4.,  9.],
    ]);

    let mut rsums = vec![0.0; 3];
    let mut csums = vec![0.0; 3];

    A.row_sums(&mut rsums);
    assert_eq!(rsums, [9.0, 2.0, 13.0]);
    A.col_sums(&mut csums);
    assert_eq!(csums, [2.0, 0.0, 22.0]);

    let mut rnorms = vec![0.0; 3];
    let mut cnorms = vec![0.0; 3];

    A.row_norms(&mut rnorms);
    assert!(rnorms == [6.0, 8.0, 9.0]);
    A.col_norms(&mut cnorms);
    assert!(cnorms == [3.0, 8.0, 9.0]);

    //no reset versions
    let mut rnorms = vec![0.0; 3];
    let mut cnorms = vec![0.0; 3];
    rnorms[2] = 100.;
    cnorms[2] = 100.;

    A.row_norms_no_reset(&mut rnorms);
    assert!(rnorms == [6.0, 8.0, 100.0]);
    A.col_norms_no_reset(&mut cnorms);
    assert!(cnorms == [3.0, 8.0, 100.0]);
}

#[test]
#[rustfmt::skip]
fn test_l_r_scalings() {

    let A = Matrix::from(&[
        [-1.,  4.,  6.],
        [ 3., -8.,  7.],
        [ 0.,  4.,  9.],
    ]);

    let lscale = vec![1., -2., 3.];
    let rscale = vec![-2., 1., -3.];

    //right scale
    let mut B = A.clone();
    B.rscale(&rscale);
    let Btest = Matrix::from(&[
        [ 2.,  4.,  -18.],
        [-6., -8.,  -21.],
        [ 0.,  4.,  -27.],
    ]);
    assert_eq!(B,Btest);

    //left scale
    let mut B = A.clone();
    B.lscale(&lscale);
    let Btest = Matrix::from(&[
        [-1.,  4.,   6.],
        [-6., 16., -14.],
        [ 0., 12.,  27.],
    ]);
    assert_eq!(B,Btest);

    //left-right scale
    let mut B = A;
    B.lrscale(&lscale, &rscale);
    let Btest = Matrix::from(&[
        [ 2.,  4., -18.],
        [12., 16.,  42.],
        [ 0., 12., -81.],
    ]);
    assert_eq!(B,Btest);
}

#[test]
fn test_symmetric_part() {
    let mut A = Matrix::from(&[[-1., 4., 6.], [2., -8., 8.], [0., 4., 9.]]);

    let B = Matrix::from(&[[-1., 3., 3.], [3., -8., 6.], [3., 6., 9.]]);

    A.symmetric_part();
    assert_eq!(B, A);
}

#[test]
fn test_col_norms_sym() {
    let A = Matrix::from(&[[-1., 4., 6.], [2., -8., 8.], [0., 4., 9.]]);

    let mut v = vec![0.0; 3];
    A.col_norms_sym(&mut v);
    assert_eq!(v, [6.0, 8.0, 9.0]);
}

#[test]
#[rustfmt::skip]
fn test_kron() {

    let A = Matrix::from(
        &[[ 1.,  2.],
          [ 4.,  5.]]);

    let B = Matrix::from(
        &[[ 1.,  2.]]);


    // A ⊗ B
    let (k1, m1) = A.size();
    let (k2, m2) = B.size();
    let mut K = Matrix::<f64>::zeros((k1 * k2, m1 * m2));
    K.kron(&A, &B);

    let Ktest = Matrix::from(
        &[[ 1.,  2.,  2.,  4.],
          [ 4.,  8.,  5., 10.]]);

    assert_eq!(K,Ktest);

    // A' ⊗ B
    let (k1, m1) = A.t().size();
    let (k2, m2) = B.size();
    let mut K = Matrix::<f64>::zeros((k1 * k2, m1 * m2));
    K.kron(&A.t(), &B);

    let Ktest = Matrix::from(
        &[[ 1.,  2.,  4.,  8.],
          [ 2.,  4.,  5., 10.]]);

    assert_eq!(K,Ktest);

    // A ⊗ B'
    let (k1, m1) = A.size();
    let (k2, m2) = B.t().size();
    let mut K = Matrix::<f64>::zeros((k1 * k2, m1 * m2));
    K.kron(&A, &B.t());

    let Ktest = Matrix::from(
        &[[1., 2. ],
          [2., 4. ],
          [4., 5. ],
          [8., 10.]]);

    assert_eq!(K,Ktest);

    // A' ⊗ B'
    let (k1, m1) = A.t().size();
    let (k2, m2) = B.t().size();
    let mut K = Matrix::<f64>::zeros((k1 * k2, m1 * m2));
    K.kron(&A.t(), &B.t());

    let Ktest = Matrix::from(
        &[[1., 4. ],
          [2., 8. ],
          [2., 5. ],
          [4., 10.]]);

    assert_eq!(K,Ktest);
}

#[test]
fn test_svec_conversions() {
    let n = 3;

    let X = Matrix::from(&[
        [1., 3., -2.], //
        [3., -4., 7.], //
        [-2., 7., 5.], //
    ]);

    let Y = Matrix::from(&[
        [2., 5., -4.],  //
        [5., 6., 2.],   //
        [-4., 2., -3.], //
    ]);

    let mut Z = Matrix::zeros((3, 3));

    let mut x = vec![0.; triangular_number(n)];
    let mut y = vec![0.; triangular_number(n)];

    // check inner product identity
    mat_to_svec(&mut x, &X);
    mat_to_svec(&mut y, &Y);

    assert!(f64::abs(x.dot(&y) - X.data().dot(Y.data())) < 1e-12);

    // check round trip
    mat_to_svec(&mut x, &X);
    svec_to_mat(&mut Z, &x);
    assert!(X.data().norm_inf_diff(Z.data()) < 1e-12);
}
