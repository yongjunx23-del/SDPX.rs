use crate::algebra::*;

/// Binary64 `A` split into a column-major dense panel of its dense columns
/// (at least a quarter full) and a sparse remainder, so products run the
/// dense part through BLAS. The authoritative CSC keeps all values.
pub(crate) struct DenseColumns<T> {
    cols: Vec<usize>,
    panel: Vec<T>,
    rest: CscMatrix<T>,
}

impl<T: FloatT> DenseColumns<T> {
    pub(crate) fn new(a: &CscMatrix<T>) -> Option<Self> {
        if T::precision_bits() > 53 || a.m == 0 {
            return None;
        }
        let dense = |j: usize| 4 * (a.colptr[j + 1] - a.colptr[j]) >= a.m;
        let cols: Vec<usize> = (0..a.n).filter(|&j| dense(j)).collect();
        if cols.len() < 4 || cols.len() * a.m > 1 << 24 {
            return None;
        }
        let mut panel = vec![T::zero(); a.m * cols.len()];
        for (d, &j) in cols.iter().enumerate() {
            for q in a.colptr[j]..a.colptr[j + 1] {
                panel[d * a.m + a.rowval[q]] = a.nzval[q];
            }
        }
        let (mut colptr, mut rowval, mut nzval) = (vec![0], Vec::new(), Vec::new());
        for j in 0..a.n {
            if !dense(j) {
                rowval.extend_from_slice(&a.rowval[a.colptr[j]..a.colptr[j + 1]]);
                nzval.extend_from_slice(&a.nzval[a.colptr[j]..a.colptr[j + 1]]);
            }
            colptr.push(rowval.len());
        }
        let rest = CscMatrix::new(a.m, a.n, colptr, rowval, nzval);
        Some(Self { cols, panel, rest })
    }

    /// `y = α·op(A)·x + β·y`.
    pub(crate) fn gemv(&self, transpose: bool, y: &mut [T], x: &[T], alpha: T, beta: T) {
        let (m, d) = (self.rest.m, self.cols.len());
        if transpose {
            self.rest.t().gemv(y, x, alpha, beta);
            let mut t = vec![T::zero(); d];
            T::xgemm(
                b'T',
                b'N',
                d as i32,
                1,
                m as i32,
                T::one(),
                &self.panel,
                m as i32,
                x,
                m as i32,
                T::zero(),
                &mut t,
                d as i32,
            );
            for (&j, &v) in self.cols.iter().zip(&t) {
                y[j] = alpha.mul_add(v, y[j]);
            }
        } else {
            self.rest.gemv(y, x, alpha, beta);
            let xd: Vec<T> = self.cols.iter().map(|&j| x[j]).collect();
            T::xgemm(
                b'N',
                b'N',
                m as i32,
                1,
                d as i32,
                alpha,
                &self.panel,
                m as i32,
                &xd,
                d as i32,
                T::one(),
                y,
                m as i32,
            );
        }
    }
}
