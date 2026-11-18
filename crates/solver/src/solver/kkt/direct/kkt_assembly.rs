#![allow(non_snake_case)]

use super::datamaps::*;
use crate::algebra::*;
use crate::solver::cones::*;
use num_traits::Zero;

pub(crate) fn allocate_kkt_Hsblocks<T, Z>(cones: &CompositeCone<T>) -> Vec<Z>
where
    T: FloatT,
    Z: Zero + Clone,
{
    let mut nnz = 0;
    if let Some(rng_last) = cones.rng_blocks.last() {
        nnz = rng_last.end;
    }
    vec![Z::zero(); nnz]
}

/// A condensed primal block starts at zero. Sparse rows already in final
/// column order can be moved into the KKT; a dense triangle needs only n.
pub(crate) enum KktPrimal<'a, T> {
    Matrix(&'a CscMatrix<T>),
    DenseSchur,
    SparseSchur {
        colptr: Vec<usize>,
        rowval: Vec<usize>,
    },
}

pub(crate) fn assemble_kkt_matrix<T: FloatT>(
    mut primal: KktPrimal<'_, T>,
    A: &CscMatrix<T>,
    cones: &CompositeCone<T>,
    shape: MatrixTriangle,
    pool: Option<&rayon::ThreadPool>,
) -> (CscMatrix<T>, LDLDataMap) {
    let (m, n) = A.size();
    let (plen, diag, indices) = match &primal {
        KktPrimal::Matrix(p) => (
            p.nnz(),
            p.count_diagonal_entries(MatrixTriangle::Triu),
            PrimalMap::Indexed(vec![0; p.nnz()]),
        ),
        KktPrimal::DenseSchur => {
            let len = triangular_number(n);
            (len, n, PrimalMap::Prefix(len))
        }
        KktPrimal::SparseSchur { rowval, .. } => (rowval.len(), n, PrimalMap::Prefix(rowval.len())),
    };
    let mut map = LDLDataMap::new(indices, A, cones);
    let p = map.sparse_maps.pdim();

    // total entries in the Hs blocks
    let nnz_Hsblocks = map.Hsblocks.len();

    let nnzKKT = plen +      // Number of elements in P
    n -                         // Number of elements in diagonal top left block
    diag +                 // remove double count on the diagonal if P has entries
    A.nnz() +                   // Number of nonzeros in A
    nnz_Hsblocks +              // Number of elements in diagonal below A'
    map.sparse_maps.nnz_vec() +  // Number of elements in sparse cone off diagonals
    p; //Number of elements in diagonal of sparse cones

    let dim = m + n + p;
    let mut colptr = vec![0; dim + 1];
    colptr[dim] = nnzKKT;
    let rowval = match &mut primal {
        KktPrimal::SparseSchur { rowval, .. } => {
            let mut rows = std::mem::take(rowval);
            rows.reserve_exact(nnzKKT - rows.len());
            rows.resize(nnzKKT, 0);
            rows
        }
        _ => vec![0; nnzKKT],
    };
    let nzval = pool
        .and_then(|p| crate::algebra::scratch::spread_vec(nnzKKT, |_| T::zero(), p))
        .unwrap_or_else(|| vec![T::zero(); nnzKKT]);
    let mut K = CscMatrix::new(dim, dim, colptr, rowval, nzval);

    _kkt_assemble_colcounts(&mut K, &primal, A, cones, &map, shape);
    _kkt_assemble_fill(&mut K, &primal, A, cones, &mut map, shape);

    (K, map)
}
fn _kkt_assemble_colcounts<T: FloatT>(
    K: &mut CscMatrix<T>,
    primal: &KktPrimal<'_, T>,
    A: &CscMatrix<T>,
    cones: &CompositeCone<T>,
    map: &LDLDataMap,
    shape: MatrixTriangle,
) {
    let (m, n) = A.size();

    // use K.p to hold nnz entries in each
    // column of the KKT matrix
    K.colptr.fill(0);

    match (primal, shape) {
        (KktPrimal::Matrix(p), MatrixTriangle::Triu) => {
            K.colcount_block(p, 0, MatrixShape::N);
            K.colcount_missing_diag(p, 0);
        }
        (KktPrimal::Matrix(p), MatrixTriangle::Tril) => {
            K.colcount_missing_diag(p, 0);
            K.colcount_block(p, 0, MatrixShape::T);
        }
        (KktPrimal::DenseSchur, MatrixTriangle::Triu) => {
            for (j, count) in K.colptr[..n].iter_mut().enumerate() {
                *count = j + 1;
            }
        }
        (KktPrimal::SparseSchur { colptr, .. }, MatrixTriangle::Triu) => {
            for (count, span) in K.colptr[..n].iter_mut().zip(colptr.windows(2)) {
                *count = span[1] - span[0];
            }
        }
        _ => unreachable!("condensed assembly uses an upper KKT"),
    }
    match shape {
        MatrixTriangle::Triu => K.colcount_block(A, n, MatrixShape::T),
        MatrixTriangle::Tril => K.colcount_block(A, 0, MatrixShape::N),
    }

    // track the next sparse column to fill (assuming triu fill)
    let mut pcol = m + n; //next sparse column to fill
    let mut sparse_map_iter = map.sparse_maps.iter();

    for (i, cone) in cones.iter().enumerate() {
        let row = cones.rng_cones[i].start + n;

        // add the Hs blocks in the lower right
        let blockdim = cone.numel();
        if cone.Hs_is_diagonal() {
            K.colcount_diag(row, blockdim);
        } else {
            K.colcount_dense_triangle(row, blockdim, shape);
        }

        //add sparse expansions columns for sparse cones
        if cone.is_sparse_expandable() {
            let sc = cone.to_sparse_expansion().unwrap();
            let thismap = sparse_map_iter.next().unwrap();
            sc.csc_colcount_sparsecone(thismap, K, row, pcol, shape);
            pcol += thismap.pdim();
        }
    }
}

fn _kkt_assemble_fill<T: FloatT>(
    K: &mut CscMatrix<T>,
    primal: &KktPrimal<'_, T>,
    A: &CscMatrix<T>,
    cones: &CompositeCone<T>,
    map: &mut LDLDataMap,
    shape: MatrixTriangle,
) {
    let (m, n) = A.size();

    // cumsum total entries to convert to K.p
    K.colcount_to_colptr();

    match (primal, &mut map.P) {
        (KktPrimal::Matrix(p), PrimalMap::Indexed(indices)) => match shape {
            MatrixTriangle::Triu => {
                K.fill_block(p, indices, 0, 0, MatrixShape::N);
                K.fill_missing_diag(p, 0);
            }
            MatrixTriangle::Tril => {
                K.fill_missing_diag(p, 0);
                K.fill_block(p, indices, 0, 0, MatrixShape::T);
            }
        },
        (_, PrimalMap::Prefix(_)) => {
            if matches!(primal, KktPrimal::DenseSchur) {
                for span in K.colptr[..=n].windows(2) {
                    for (i, row) in K.rowval[span[0]..span[1]].iter_mut().enumerate() {
                        *row = i;
                    }
                }
            }
            // Sparse rows are already in place. Advance the fill cursors
            // just as fill_block would before appending the retained rows.
            K.colptr.copy_within(1..n + 1, 0);
        }
        _ => unreachable!(),
    }
    match shape {
        MatrixTriangle::Triu => K.fill_block(A, &mut map.A, 0, n, MatrixShape::T),
        MatrixTriangle::Tril => K.fill_block(A, &mut map.A, n, 0, MatrixShape::N),
    }

    // track the next sparse column to fill (assuming triu fill)
    let mut pcol = m + n; //next sparse column to fill
    let mut sparse_map_iter = map.sparse_maps.iter_mut();

    for (i, cone) in cones.iter().enumerate() {
        let row = cones.rng_cones[i].start + n;

        // add the Hs blocks in the lower right
        let blockdim = cone.numel();
        let block = &mut map.Hsblocks[cones.rng_blocks[i].clone()];

        if cone.Hs_is_diagonal() {
            K.fill_diag(block, row, blockdim);
        } else {
            K.fill_dense_triangle(block, row, blockdim, shape);
        }

        //add sparse expansions columns for sparse cones
        if cone.is_sparse_expandable() {
            let sc = cone.to_sparse_expansion().unwrap();
            let thismap = sparse_map_iter.next().unwrap();
            sc.csc_fill_sparsecone(thismap, K, row, pcol, shape);
            pcol += thismap.pdim();
        }
    }

    // backshift the colptrs to recover K.p again
    K.backshift_colptrs();

    // Now we can populate the index of the full diagonal.
    // We have filled in structural zeros on it everywhere.

    match shape {
        MatrixTriangle::Triu => {
            // matrix is triu, so diagonal is last in each column
            map.diag_full.copy_from_slice(&K.colptr[1..]);
            map.diag_full.iter_mut().for_each(|x| *x -= 1);
            // and the diagonal of just the upper left
            map.diagP.copy_from_slice(&K.colptr[1..=n]);
            map.diagP.iter_mut().for_each(|x| *x -= 1);
        }

        MatrixTriangle::Tril => {
            // matrix is tril, so diagonal is first in each column
            map.diag_full
                .copy_from_slice(&K.colptr[0..K.colptr.len() - 1]);
            // and the diagonal of just the upper left
            map.diagP.copy_from_slice(&K.colptr[0..n]);
        }
    }
}

#[test]
fn condensed_prefix_matches_explicit_primal() {
    let a = CscMatrix::from(&[[2., 0., -3.], [0., 1., 4.]]);
    let cones = CompositeCone::new(&[SupportedConeT::ZeroConeT(2)]);
    for (colptr, rows) in [
        (vec![0, 1, 3, 6], vec![0, 0, 1, 0, 1, 2]),
        (vec![0, 1, 2, 4], vec![0, 1, 0, 2]),
    ] {
        let len = rows.len();
        let p = CscMatrix::new(3, 3, colptr.clone(), rows.clone(), vec![0.; len]);
        let assemble =
            |pattern| assemble_kkt_matrix(pattern, &a, &cones, MatrixTriangle::Triu, None);
        let (expected, old) = assemble(KktPrimal::Matrix(&p));
        let pattern = if len == triangular_number(a.n) {
            KktPrimal::DenseSchur
        } else {
            KktPrimal::SparseSchur {
                colptr,
                rowval: rows,
            }
        };
        let (actual, new) = assemble(pattern);
        assert_eq!(actual, expected);
        assert!(matches!(old.P, PrimalMap::Indexed(v) if v.iter().copied().eq(0..len)));
        assert!(matches!(new.P, PrimalMap::Prefix(n) if n == len));
        assert_eq!(new.A, old.A);
        assert_eq!(new.Hsblocks, old.Hsblocks);
        assert_eq!(new.diagP, old.diagP);
        assert_eq!(new.diag_full, old.diag_full);
    }
}

#[test]
fn test_kkt_assembly_upper_lower() {
    let P = CscMatrix::from(&[
        [1., 2., 4.], //
        [0., 3., 5.], //
        [0., 0., 6.], //
    ]);
    let A = CscMatrix::from(&[
        [7., 0., 8.],  //
        [0., 9., 10.], //
        [1., 2., 3.],
        [7., 0., 8.],  //
        [0., 9., 10.], //
        [1., 2., 3.],
    ]);

    let Ku_true_nncone = CscMatrix::from(&[
        [1., 2., 4., 7., 0., 1., 7., 0., 1.],   //
        [0., 3., 5., 0., 9., 2., 0., 9., 2.],   //
        [0., 0., 6., 8., 10., 3., 8., 10., 3.], //
        [0., 0., 0., -1., 0., 0., 0., 0., 0.],  //
        [0., 0., 0., 0., -1., 0., 0., 0., 0.],  //
        [0., 0., 0., 0., 0., -1., 0., 0., 0.],  //
        [0., 0., 0., 0., 0., 0., -1., 0., 0.],  //
        [0., 0., 0., 0., 0., 0., 0., -1., 0.],  //
        [0., 0., 0., 0., 0., 0., 0., 0., -1.],  //
    ]);

    let Kl_true_nncone = CscMatrix::from(&[
        [1., 0., 0., 0., 0., 0., 0., 0., 0.],   //
        [2., 3., 0., 0., 0., 0., 0., 0., 0.],   //
        [4., 5., 6., 0., 0., 0., 0., 0., 0.],   //
        [7., 0., 8., -1., 0., 0., 0., 0., 0.],  //
        [0., 9., 10., 0., -1., 0., 0., 0., 0.], //
        [1., 2., 3., 0., 0., -1., 0., 0., 0.],  //
        [7., 0., 8., 0., 0., 0., -1., 0., 0.],  //
        [0., 9., 10., 0., 0., 0., 0., -1., 0.], //
        [1., 2., 3., 0., 0., 0., 0., 0., -1.],  //
    ]);

    let Ku_true_expcones = CscMatrix::from(&[
        [1., 2., 4., 7., 0., 1., 7., 0., 1.],    //
        [0., 3., 5., 0., 9., 2., 0., 9., 2.],    //
        [0., 0., 6., 8., 10., 3., 8., 10., 3.],  //
        [0., 0., 0., -1., -1., -1., 0., 0., 0.], //
        [0., 0., 0., 0., -1., -1., 0., 0., 0.],  //
        [0., 0., 0., 0., 0., -1., 0., 0., 0.],   //
        [0., 0., 0., 0., 0., 0., -1., -1., -1.], //
        [0., 0., 0., 0., 0., 0., 0., -1., -1.],  //
        [0., 0., 0., 0., 0., 0., 0., 0., -1.],   //
    ]);

    let Kl_true_expcones = CscMatrix::from(&[
        [1., 0., 0., 0., 0., 0., 0., 0., 0.],    //
        [2., 3., 0., 0., 0., 0., 0., 0., 0.],    //
        [4., 5., 6., 0., 0., 0., 0., 0., 0.],    //
        [7., 0., 8., -1., 0., 0., 0., 0., 0.],   //
        [0., 9., 10., -1., -1., 0., 0., 0., 0.], //
        [1., 2., 3., -1., -1., -1., 0., 0., 0.], //
        [7., 0., 8., 0., 0., 0., -1., 0., 0.],   //
        [0., 9., 10., 0., 0., 0., -1., -1., 0.], //
        [1., 2., 3., 0., 0., 0., -1., -1., -1.], //
    ]);

    let Ku_true_socone = CscMatrix::from(&[
        [1., 2., 4., 7., 0., 1., 7., 0., 1., 0., 0.],   //
        [0., 3., 5., 0., 9., 2., 0., 9., 2., 0., 0.],   //
        [0., 0., 6., 8., 10., 3., 8., 10., 3., 0., 0.], //
        [0., 0., 0., -1., 0., 0., 0., 0., 0., 2., 3.],  //
        [0., 0., 0., 0., -1., 0., 0., 0., 0., 2., 3.],  //
        [0., 0., 0., 0., 0., -1., 0., 0., 0., 2., 3.],  //
        [0., 0., 0., 0., 0., 0., -1., 0., 0., 2., 3.],  //
        [0., 0., 0., 0., 0., 0., 0., -1., 0., 2., 3.],  //
        [0., 0., 0., 0., 0., 0., 0., 0., -1., 2., 3.],  //
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 4., 0.],   //
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 4.],   //
    ]);

    let Kl_true_socone = CscMatrix::from(&[
        [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],   //
        [2., 3., 0., 0., 0., 0., 0., 0., 0., 0., 0.],   //
        [4., 5., 6., 0., 0., 0., 0., 0., 0., 0., 0.],   //
        [7., 0., 8., -1., 0., 0., 0., 0., 0., 0., 0.],  //
        [0., 9., 10., 0., -1., 0., 0., 0., 0., 0., 0.], //
        [1., 2., 3., 0., 0., -1., 0., 0., 0., 0., 0.],  //
        [7., 0., 8., 0., 0., 0., -1., 0., 0., 0., 0.],  //
        [0., 9., 10., 0., 0., 0., 0., -1., 0., 0., 0.], //
        [1., 2., 3., 0., 0., 0., 0., 0., -1., 0., 0.],  //
        [0., 0., 0., 2., 2., 2., 2., 2., 2., 4., 0.],   //
        [0., 0., 0., 3., 3., 3., 3., 3., 3., 0., 4.],   //
    ]);

    // diagonal lower right block tests
    // --------------------------------
    let K = SupportedConeT::NonnegativeConeT(6);
    let cones = CompositeCone::new(&[K]);

    let (mut Ku, mapu) = assemble_kkt_matrix(
        KktPrimal::Matrix(&P),
        &A,
        &cones,
        MatrixTriangle::Triu,
        None,
    );
    for i in mapu.Hsblocks {
        Ku.nzval[i] = -1.;
    }
    assert_eq!(Ku, Ku_true_nncone);

    let (mut Kl, mapl) = assemble_kkt_matrix(
        KktPrimal::Matrix(&P),
        &A,
        &cones,
        MatrixTriangle::Tril,
        None,
    );
    for i in mapl.Hsblocks {
        Kl.nzval[i] = -1.;
    }
    assert_eq!(Kl, Kl_true_nncone);

    // dense blocks lower right block tests
    // --------------------------------
    let K = SupportedConeT::ExponentialConeT();
    let cones = CompositeCone::new(&[K.clone(), K.clone()]);

    let (mut Ku, mapu) = assemble_kkt_matrix(
        KktPrimal::Matrix(&P),
        &A,
        &cones,
        MatrixTriangle::Triu,
        None,
    );
    for i in mapu.Hsblocks {
        Ku.nzval[i] = -1.;
    }
    assert_eq!(Ku, Ku_true_expcones);

    let (mut Kl, mapl) = assemble_kkt_matrix(
        KktPrimal::Matrix(&P),
        &A,
        &cones,
        MatrixTriangle::Tril,
        None,
    );
    for i in mapl.Hsblocks {
        Kl.nzval[i] = -1.;
    }
    assert_eq!(Kl, Kl_true_expcones);

    // sparsified-SOC lower right block tests
    // --------------------------------
    let K = SupportedConeT::SecondOrderConeT(6);
    let cones = CompositeCone::new(&[K]);

    fn fill_sparse_cone(K: &mut CscMatrix, map: &LDLDataMap) {
        //fill in the sparse expansion terms for the sparse cones
        for sparsemap in map.sparse_maps.iter() {
            match sparsemap {
                SparseExpansionMap::SOCExpansionMap(socmap) => {
                    for &i in socmap.v.iter() {
                        K.nzval[i] = 2.;
                    }
                    for &i in socmap.u.iter() {
                        K.nzval[i] = 3.;
                    }
                    for &i in socmap.D.iter() {
                        K.nzval[i] = 4.;
                    }
                }
                _ => {
                    panic!("unexpected sparse map");
                }
            }
        }
    }

    let (mut Ku, mapu) = assemble_kkt_matrix(
        KktPrimal::Matrix(&P),
        &A,
        &cones,
        MatrixTriangle::Triu,
        None,
    );

    fill_sparse_cone(&mut Ku, &mapu);

    for i in mapu.Hsblocks {
        Ku.nzval[i] = -1.;
    }
    assert_eq!(Ku, Ku_true_socone);

    let (mut Kl, mapl) = assemble_kkt_matrix(
        KktPrimal::Matrix(&P),
        &A,
        &cones,
        MatrixTriangle::Tril,
        None,
    );

    fill_sparse_cone(&mut Kl, &mapl);

    for i in mapl.Hsblocks {
        Kl.nzval[i] = -1.;
    }
    assert_eq!(Kl, Kl_true_socone);
}
