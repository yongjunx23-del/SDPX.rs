#![allow(non_snake_case)]

use num_traits::Num;
use std::iter::zip;

/// Sparse vector type (internal use only at present)

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SparseVector<T = f64> {
    /// vector dimension
    pub n: usize,
    /// vector of entry indices
    pub nzind: Vec<usize>,
    /// vector of non-zero vector elements
    pub nzval: Vec<T>,
}

/// Creates a `SparseVector` from a dense slice.
impl<T> SparseVector<T>
where
    T: Num + Copy,
{
    pub fn new(values: &[T]) -> Self {
        let mut nzind = Vec::new();
        let mut nzval = Vec::new();

        let mut n = 0;
        for (i, &v) in values.iter().enumerate() {
            if v != T::zero() {
                nzind.push(i);
                nzval.push(v);
            }
            n += 1;
        }
        SparseVector { n, nzind, nzval }
    }
}

impl<T> From<SparseVector<T>> for Vec<T>
where
    T: Num + Copy,
{
    fn from(sv: SparseVector<T>) -> Vec<T> {
        let mut v = vec![T::zero(); sv.n];
        for (i, nz) in zip(sv.nzind, sv.nzval) {
            v[i] = nz;
        }
        v
    }
}

#[test]
fn test_sparsevector_new() {
    let v = vec![0.1, 0.3, 0.0, 0.0, 0.4, 0.0];

    let vs = SparseVector::new(&v);

    assert_eq!(vs.n, v.len());
    assert_eq!(vs.nzind, vec![0, 1, 4]);
    assert_eq!(vs.nzval, vec![0.1, 0.3, 0.4]);

    let vback: Vec<f64> = vs.into();
    assert_eq!(v, vback);
}
