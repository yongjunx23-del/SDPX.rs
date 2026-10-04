#![allow(non_snake_case)]

// 2z2 symmetric eigendecomposition using Jacobi method.
// This uses Jacobi rotations from the 3x3 method, but
// only one rotation is required
use crate::algebra::dense::fixed::dense3x3::eigen::compute_jacobi_rotation;
use crate::algebra::FloatT;
use crate::algebra::{DenseMatrix2, DenseMatrixSym2};

impl<T> DenseMatrixSym2<T>
where
    T: FloatT,
{
    pub(crate) fn eigvals(&mut self) -> [T; 2] {
        eig2(self, None)
    }
}

fn eig2<T>(A: &DenseMatrixSym2<T>, V: Option<&mut DenseMatrix2<T>>) -> [T; 2]
where
    T: FloatT,
{
    // assume 2x2 input here
    let App = A.data[0];
    let Apq = A.data[1];
    let Aqq = A.data[2];
    let (c, s, t) = compute_jacobi_rotation(Apq, App, Aqq);
    // one step Givens rotation
    let tApq = t * Apq;
    let e1 = App - tApq;
    let e2 = Aqq + tApq;
    let noswap = e1 < e2;
    let e = if noswap { [e1, e2] } else { [e2, e1] };

    // compute eigenvectors if needed
    if let Some(Vp) = V {
        if noswap {
            Vp[(0, 0)] = c;
            Vp[(1, 0)] = -s;
            Vp[(0, 1)] = s;
            Vp[(1, 1)] = c;
        } else {
            Vp[(0, 0)] = s;
            Vp[(1, 0)] = c;
            Vp[(0, 1)] = c;
            Vp[(1, 1)] = -s;
        }
    }
    e
}

// ---- unit testing ----
