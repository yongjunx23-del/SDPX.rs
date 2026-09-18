use super::*;

#[test]
#[rustfmt::skip]
fn test_matrix_istriu_istril() {

    let A = CscMatrix::from(&[
        [1., 2., 3.],
        [0., 2., 0.],
        [0., 0., 1.]]);

    assert!(A.is_triu());
    assert!(!A.is_tril());
    assert!(A.sym_up().is_triu_src());
    assert!(!A.sym_up().is_tril_src());

    let A = CscMatrix::from(&[
        [1., 2., 3.],
        [0., 2., 0.],
        [1., 0., 1.]]);

    assert!(!A.is_triu());
    assert!(!A.is_tril());

    let A = CscMatrix::from(&[
        [1., 0., 0.],
        [0., 2., 0.],
        [1., 1., 1.]]);

    assert!(!A.is_triu());
    assert!(A.is_tril());
    assert!(!A.sym_lo().is_triu_src());
    assert!(A.sym_lo().is_tril_src());
}

#[test]
fn test_csc_from_slice_of_arrays() {
    let A = CscMatrix::new(
        3,                    // m
        2,                    // n
        vec![0, 2, 4],        // colptr
        vec![0, 1, 0, 2],     // rowval
        vec![1., 3., 2., 4.], // nzval
    );

    let B = CscMatrix::from(&[
        [1., 2.], //
        [3., 0.], //
        [0., 4.],
    ]); //

    let C: CscMatrix = (&[
        [1., 2.], //
        [3., 0.], //
        [0., 4.],
    ])
        .into();

    assert_eq!(A, B);
    assert_eq!(A, C);
}

#[test]
fn test_csc_get_entry() {
    let A = CscMatrix::from(&[
        [0.0, 4.0, 0.0, 0.0, 12.0],
        [1.0, 5.0, 0.0, 0.0, 0.0],
        [0.0, 6.0, 0.0, 0.0, 13.0],
        [2.0, 7.0, 10.0, 0.0, 0.0],
        [0.0, 8.0, 11.0, 0.0, 14.0],
        [3.0, 9.0, 0.0, 0.0, 0.0],
    ]);

    assert_eq!(A.get_entry((1, 0)), Some(1.));
    assert_eq!(A.get_entry((5, 0)), Some(3.));
    assert_eq!(A.get_entry((0, 1)), Some(4.));
    assert_eq!(A.get_entry((3, 1)), Some(7.));
    assert_eq!(A.get_entry((5, 1)), Some(9.));
    assert_eq!(A.get_entry((3, 2)), Some(10.));
    assert_eq!(A.get_entry((4, 2)), Some(11.));
    assert_eq!(A.get_entry((4, 4)), Some(14.));

    assert_eq!(A.get_entry((0, 0)), None);
    assert_eq!(A.get_entry((4, 0)), None);
    assert_eq!(A.get_entry((2, 2)), None);
    assert_eq!(A.get_entry((1, 3)), None);
    assert_eq!(A.get_entry((2, 3)), None);
    assert_eq!(A.get_entry((4, 3)), None);
    assert_eq!(A.get_entry((3, 4)), None);
}

#[test]
fn test_csc_set_entry() {
    let mut A = CscMatrix::from(&[
        [0.0, 3.0, 6.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 4.0, 7.0, 8.0],
        [2.0, 5.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ]);

    let B = CscMatrix::from(&[
        [0.0, 3.0, -6.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 4.0, 7.0, -8.0],
        [2.0, 5.0, 10.0, 0.0],
        [0.0, 0.0, 0.0, 11.0],
    ]);

    // overwrite existing entries
    A.set_entry((0, 2), -6.0);
    A.set_entry((2, 3), -8.0);

    // add new entries
    A.set_entry((3, 2), 10.0);
    A.set_entry((4, 3), 11.0);

    assert_eq!(A, B);
}

#[test]
fn test_csc_index_to_coord() {
    let A = CscMatrix::from(&[
        [0.0, 4.0, 0.0, 0.0, 12.0],
        [1.0, 5.0, 0.0, 0.0, 0.0],
        [0.0, 6.0, 0.0, 0.0, 13.0],
        [2.0, 7.0, 10.0, 0.0, 0.0],
        [0.0, 8.0, 11.0, 0.0, 14.0],
        [3.0, 9.0, 0.0, 0.0, 0.0],
    ]);

    assert_eq!(A.index_to_coord(0), (1, 0));
    assert_eq!(A.index_to_coord(1), (3, 0));
    assert_eq!(A.index_to_coord(2), (5, 0));
    assert_eq!(A.index_to_coord(3), (0, 1));
    assert_eq!(A.index_to_coord(4), (1, 1));
    assert_eq!(A.index_to_coord(5), (2, 1));
    assert_eq!(A.index_to_coord(6), (3, 1));
    assert_eq!(A.index_to_coord(7), (4, 1));
    assert_eq!(A.index_to_coord(8), (5, 1));
    assert_eq!(A.index_to_coord(9), (3, 2));
    assert_eq!(A.index_to_coord(10), (4, 2));
    assert_eq!(A.index_to_coord(11), (0, 4));
    assert_eq!(A.index_to_coord(12), (2, 4));
    assert_eq!(A.index_to_coord(13), (4, 4));
}

#[test]
fn test_adjoint_into() {
    let A: CscMatrix = (&[
        [1., 0., 0.], //
        [2., 4., 0.], //
        [3., 5., 6.],
    ])
        .into();

    let T: CscMatrix = (&[
        [1., 2., 3.], //
        [0., 4., 5.], //
        [0., 0., 6.],
    ])
        .into();

    let B: CscMatrix = A.t().into(); //Concrete form.  Allocates and copies.

    assert_eq!(B, T);
}

#[test]
fn test_triplets() {
    let A: CscMatrix = (&[
        [1., 0., 0., 5.], //
        [0., 0., 3., 0.], //
        [2., 0., 4., 0.],
    ])
        .into();

    let cols = vec![0, 0, 2, 2, 3];
    let rows = vec![0, 2, 1, 2, 0];
    let vals = vec![1., 2., 3., 4., 5.];

    // extract triplet format data and compare
    let (I, J, V) = A.findnz();
    assert_eq!(I, rows);
    assert_eq!(J, cols);
    assert_eq!(V, vals);

    // construct from triplets and compare
    let B: CscMatrix = CscMatrix::new_from_triplets(3, 4, rows, cols, vals);
    assert_eq!(A, B);

    // same thing, but with data in the wrong order
    let cols = vec![2, 0, 2, 0, 3];
    let rows = vec![2, 2, 1, 0, 0];
    let vals = vec![4., 2., 3., 1., 5.];

    let B: CscMatrix = CscMatrix::new_from_triplets(3, 4, rows, cols, vals);

    assert_eq!(A, B);

    // case with repeated entries, unsorted

    let A: CscMatrix<isize> = (&[
        [0, 0, 0],   //
        [-20, 0, 0], //
        [-20, -20, 0],
    ])
        .into();

    let rows = vec![1, 2, 2, 1, 2, 2];
    let cols = vec![0, 0, 1, 0, 0, 1];
    let vals = vec![-10, -10, -10, -10, -10, -10];

    let B = CscMatrix::new_from_triplets(3, 3, rows, cols, vals);
    assert_eq!(A, B);
}

#[test]
fn test_drop_zeros() {
    let mut A = CscMatrix::from(&[
        [0.0, 3.0, 6.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 4.0, 7.0, 8.0],
        [2.0, 5.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ]);

    // same, but with 2,6,7,8 set to zero
    let mut B = CscMatrix::from(&[
        [0.0, 3.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 4.0, 0.0, 0.0],
        [0.0, 5.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ]);

    // overwrite existing entries
    let dropped = [2, 6, 7, 8];
    for idx in dropped {
        A.nzval[idx - 1] = 0.0;
    }

    //squeeze out the zeros
    let count = A.dropzeros();

    assert_eq!(count, 4);
    assert_eq!(A, B);

    // nothing to drop
    let count = B.dropzeros();
    assert_eq!(count, 0);
}

#[test]
fn test_sort_indices() {
    let mut A = CscMatrix {
        m: 4,
        n: 3,
        colptr: vec![0, 2, 4, 5],
        rowval: vec![3, 1, 4, 2, 2],
        nzval: vec![2.0, 3.0, 1.0, 4.0, 5.0],
    };

    A.sort_indices().unwrap();
    assert_eq!(A.rowval, vec![1, 3, 2, 4, 2]);
    assert_eq!(A.nzval, vec![3.0, 2.0, 4.0, 1.0, 5.0]);

    //nothing to sort
    A.sort_indices().unwrap();
    assert_eq!(A.rowval, vec![1, 3, 2, 4, 2]);
    assert_eq!(A.nzval, vec![3.0, 2.0, 4.0, 1.0, 5.0]);
}

#[test]
fn test_sort_indices_with_duplicates() {
    let mut A = CscMatrix {
        m: 4,
        n: 2,
        colptr: vec![0, 3, 5],
        rowval: vec![3, 3, 1, 2, 4],
        nzval: vec![2.0, 3.0, 1.0, 1.0, 4.0],
    };

    A.sort_indices().unwrap();
    assert_eq!(A.rowval, vec![1, 3, 3, 2, 4]);
    assert_eq!(A.nzval, vec![1.0, 2.0, 3.0, 1.0, 4.0]);
}

#[test]
fn test_deduplicate() {
    let mut A = CscMatrix {
        m: 4,
        n: 2,
        colptr: vec![0, 2, 4],
        rowval: vec![1, 1, 2, 4],
        nzval: vec![3.0, 2.0, 1.0, 4.0],
    };

    A.deduplicate().unwrap();
    assert_eq!(A.colptr, vec![0, 1, 3]);
    assert_eq!(A.rowval, vec![1, 2, 4]);
    assert_eq!(A.nzval, vec![5.0, 1.0, 4.0]);

    // nothing to deduplicate
    A.deduplicate().unwrap();
    assert_eq!(A.colptr, vec![0, 1, 3]);
    assert_eq!(A.rowval, vec![1, 2, 4]);
    assert_eq!(A.nzval, vec![5.0, 1.0, 4.0]);
}

#[test]
fn test_deduplicate_multiple_columns() {
    let mut A = CscMatrix {
        m: 4,
        n: 3,
        colptr: vec![0, 2, 4, 6],
        rowval: vec![1, 1, 2, 4, 3, 3],
        nzval: vec![3.0, 2.0, 1.0, 4.0, 5.0, 6.0],
    };

    A.deduplicate().unwrap();
    assert_eq!(A.colptr, vec![0, 1, 3, 4]);
    assert_eq!(A.rowval, vec![1, 2, 4, 3]);
    assert_eq!(A.nzval, vec![5.0, 1.0, 4.0, 11.0]);
}

#[test]
fn test_deduplicate_1col() {
    let mut A = CscMatrix {
        m: 4,
        n: 1,
        colptr: vec![0, 3],
        rowval: vec![1, 1, 4],
        nzval: vec![2.0, 3.0, 4.0],
    };

    A.deduplicate().unwrap();
    assert_eq!(A.colptr, vec![0, 2]);
    assert_eq!(A.rowval, vec![1, 4]);
    assert_eq!(A.nzval, vec![5.0, 4.0]);
}

#[test]
fn test_canonicalize() {
    let mut A = CscMatrix {
        m: 4,
        n: 3,
        colptr: vec![0, 3, 4, 7],
        rowval: vec![2, 1, 1, 4, 3, 4, 3],
        nzval: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
    };

    A.canonicalize().unwrap();
    assert_eq!(A.colptr, vec![0, 2, 3, 5]);
    assert_eq!(A.rowval, vec![1, 2, 4, 3, 4]);
    assert_eq!(A.nzval, vec![5.0, 1.0, 4.0, 12.0, 6.0]);
}

#[test]
fn test_canonicalize_structural_zeros() {
    let mut A = CscMatrix {
        m: 4,
        n: 3,
        colptr: vec![0, 3, 4, 7],
        rowval: vec![2, 1, 1, 4, 3, 4, 3],
        nzval: vec![1.0, 2.0, 3.0, 0.0, 5.0, 6.0, -5.0],
    };

    A.canonicalize().unwrap();
    assert_eq!(A.colptr, vec![0, 2, 3, 5]);
    assert_eq!(A.rowval, vec![1, 2, 4, 3, 4]);
    assert_eq!(A.nzval, vec![5.0, 1.0, 0.0, 0.0, 6.0]);
}

#[test]
fn test_canonicalize_empty() {
    let mut A: CscMatrix<f64> = CscMatrix {
        m: 0,
        n: 0,
        colptr: vec![0],
        rowval: vec![],
        nzval: vec![],
    };

    A.canonicalize().unwrap();
    assert!(A.rowval.is_empty());
    assert!(A.nzval.is_empty());
}

#[test]
fn test_canonicalize_singleton() {
    let mut A = CscMatrix {
        m: 4,
        n: 1,
        colptr: vec![0, 1],
        rowval: vec![2],
        nzval: vec![5.0],
    };

    A.sort_indices().unwrap();
    assert_eq!(A.rowval, vec![2]);
    assert_eq!(A.nzval, vec![5.0]);
}
