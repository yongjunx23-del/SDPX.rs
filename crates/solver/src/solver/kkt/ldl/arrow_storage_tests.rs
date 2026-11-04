// Included in arrow.rs::tests: uses the existing Arrow fixture and kernels.
/// Values of `indices` after adding `offset * sign` in order (repeats accumulate).
fn shifted<T: FloatT>(m: &CscMatrix<T>, indices: &[usize], offset: T, signs: &[i8]) -> Vec<T> {
    let mut v = m.nzval.clone();
    for (&p, &s) in indices.iter().zip(signs) {
        v[p] += offset * T::from_i8(s).unwrap();
    }
    indices.iter().map(|&p| v[p]).collect()
}

fn storage_number<T: FloatT>(x: f64) -> T {
    T::from_f64(x).unwrap()
}

fn storage_fixture<T: FloatT>() -> (CscMatrix<T>, Vec<i8>) {
    let (base, old_signs) = arrow_kkt();
    // Both signs and each nonsingleton positive component have noncontiguous
    // IDs. Coupling entries occur on both sides of the global border IDs.
    let permutation = [4, 0, 6, 2, 5, 1, 3];
    let mut entries = std::collections::BTreeMap::new();
    for col in 0..base.n {
        for p in base.colptr[col]..base.colptr[col + 1] {
            let (a, b) = (permutation[base.rowval[p]], permutation[col]);
            entries.insert((a.max(b), a.min(b)), storage_number::<T>(base.nzval[p]));
        }
    }
    // Stored signed zero, not a missing entry. This does not merge leaves.
    assert!(entries.insert((1, 0), -T::zero()).is_none());
    let mut signs = vec![0; base.n];
    for i in 0..base.n {
        signs[permutation[i]] = old_signs[i];
    }
    let mut colptr = vec![0];
    let mut rowval = Vec::new();
    let mut nzval = Vec::new();
    for col in 0..base.n {
        for (&(j, i), &v) in &entries {
            if j == col {
                rowval.push(i);
                nzval.push(v);
            }
        }
        colptr.push(nzval.len());
    }
    (CscMatrix::new(base.n, base.n, colptr, rowval, nzval), signs)
}

fn storage_same<T: FloatT>(a: &[T], b: &[T]) {
    assert_eq!(a.len(), b.len());
    for (&a, &b) in a.iter().zip(b) {
        if a.is_nan() {
            assert!(b.is_nan());
        } else {
            assert_eq!(a, b);
            if a.is_zero() {
                assert_eq!(a.is_sign_negative(), b.is_sign_negative());
            }
        }
    }
}

fn storage_matrix_same<T: FloatT>(a: &CscMatrix<T>, b: &CscMatrix<T>) {
    assert_eq!(a.size(), b.size());
    assert_eq!(a.colptr, b.colptr);
    assert_eq!(a.rowval, b.rowval);
    storage_same(&a.nzval, &b.nzval);
}

fn storage_position<T>(k: &CscMatrix<T>, i: usize, j: usize) -> usize {
    (k.colptr[j]..k.colptr[j + 1])
        .find(|&p| k.rowval[p] == i)
        .unwrap()
}

fn storage_check_solve<T: FloatT>(
    solver: &mut ArrowLDLSolver<T>,
    k: &CscMatrix<T>,
    signs: &[i8],
    check_residual: bool,
) {
    let mut fresh = ArrowLDLSolver::try_new(k, signs, &solver.settings).unwrap();
    assert!(solver.refactor(k));
    assert!(fresh.refactor(k));
    assert_eq!(solver.use_arrow, fresh.use_arrow);
    for cols in [1, 3] {
        let rhs: Vec<T> = (0..k.n * cols)
            .map(|i| storage_number::<T>((i as f64 - 3.) / 8.))
            .collect();
        let mut actual = vec![T::zero(); rhs.len()];
        let mut expected = actual.clone();
        solver.solve_many(k, &mut actual, &mut rhs.clone(), cols);
        fresh.solve_many(k, &mut expected, &mut rhs.clone(), cols);
        storage_same(&actual, &expected);
        for c in 0..cols {
            let span = c * k.n..(c + 1) * k.n;
            let mut single = vec![T::zero(); k.n];
            solver.solve(k, &mut single, &mut rhs[span.clone()].to_vec());
            storage_same(&single, &actual[span.clone()]);
            // Independent original-input CSC multiply, never factor products.
            let mut error = rhs[span.clone()].to_vec();
            k.sym_up()
                .symv(&mut error, &actual[span.clone()], -T::one(), T::one());
            assert!(error.is_finite());
            if check_residual {
                assert!(
                    error.norm_inf()
                        <= T::epsilon()
                            * storage_number::<T>(65536.)
                            * rhs[span].norm_inf().max(T::one())
                );
            }
        }
    }
}

/// The caller's update: the KKT changes, and the backend is told (a generic
/// arrow ignores it and reads the KKT at refactor).
fn storage_update<T: FloatT>(
    solver: &mut ArrowLDLSolver<T>,
    k: &mut CscMatrix<T>,
    indices: &[usize],
    values: &[T],
) {
    for (&p, &v) in indices.iter().zip(values) {
        k.nzval[p] = v;
    }
    solver.update_values(indices, values);
}

fn storage_update_parity<T: FloatT>() {
    let (input, signs) = storage_fixture::<T>();
    let untouched = input.clone();
    let mut expected = input.clone();
    let settings = CoreSettings::<T>::default();
    let mut solver = ArrowLDLSolver::try_new(&input, &signs, &settings).unwrap();
    assert_eq!(solver.trunk, vec![1, 3]);
    assert_eq!(
        solver
            .leaves
            .iter()
            .map(|l| l.ids.clone())
            .collect::<Vec<_>>(),
        vec![vec![0, 4], vec![2, 6], vec![5]]
    );
    // A generic arrow keeps no copy of the KKT values.
    assert!(solver.c.is_empty());
    assert!(solver.leaves.iter().all(|l| l.h.is_empty() && l.b.is_empty()));
    let pointers: Vec<_> = solver
        .leaves
        .iter()
        .map(|l| (l.y.as_ptr(), l.factor.l.as_ptr()))
        .collect();
    // Sparse arbitrary/reversed updates and repeated positions: last write wins.
    let last = expected.nzval.len() - 1;
    let indices = [last, 0, last / 2, 1, last, 0];
    let values: Vec<T> = indices
        .iter()
        .enumerate()
        .map(|(j, &p)| expected.nzval[p] + storage_number::<T>((j + 1) as f64 / 128.))
        .collect();
    storage_update(&mut solver, &mut expected, &indices, &values);
    storage_check_solve(&mut solver, &expected, &signs, true);

    let indices = [last, 1, last / 2, 1];
    let scale = storage_number::<T>(0.5);
    solver.scale_values(&indices, scale);
    for &p in &indices {
        expected.nzval[p] *= scale;
    }
    let offset_signs = [-1, 1, 1, -1];
    let offset = storage_number::<T>(0.03125);
    let values = shifted(&expected, &indices, offset, &offset_signs);
    storage_update(&mut solver, &mut expected, &indices, &values);
    storage_check_solve(&mut solver, &expected, &signs, true);

    let zero = storage_position(&expected, 0, 1);
    storage_update(&mut solver, &mut expected, &[zero, zero], &[T::zero(), -T::zero()]);
    solver.scale_values(&[zero], -T::one());
    expected.nzval[zero] *= -T::one();

    // Model DirectLDL's signed static shift and explicit restoration. The
    // caller's original unshifted KKT is never used as mutable backend storage.
    let diagonal: Vec<_> = (0..expected.n)
        .map(|i| storage_position(&expected, i, i))
        .collect();
    let old_diag: Vec<_> = diagonal.iter().map(|&p| expected.nzval[p]).collect();
    let shift = storage_number::<T>(0.0625);
    let values = shifted(&expected, &diagonal, shift, &signs);
    storage_update(&mut solver, &mut expected, &diagonal, &values);
    storage_check_solve(&mut solver, &expected, &signs, true);
    storage_update(&mut solver, &mut expected, &diagonal, &old_diag);
    storage_check_solve(&mut solver, &expected, &signs, true);

    // A zero singleton pivot fails in both backends when regularization is
    // disabled. The lazy fallback factors the KKT it is given, and restoring
    // the pivot must recover a valid solve.
    let restore_diag: Vec<_> = diagonal.iter().map(|&p| expected.nzval[p]).collect();
    let values = shifted(&expected, &diagonal, shift, &signs);
    storage_update(&mut solver, &mut expected, &diagonal, &values);
    solver.settings.dynamic_regularization_enable = false;
    let singleton = storage_position(&expected, 5, 5);
    let original = expected.nzval[singleton];
    storage_update(&mut solver, &mut expected, &[singleton], &[T::zero()]);
    assert!(!solver.refactor(&expected));
    assert!(!solver.use_arrow);
    assert!(solver.fallback.is_some());
    let mut fresh = ArrowLDLSolver::try_new(&expected, &signs, &solver.settings).unwrap();
    assert!(!fresh.refactor(&expected));
    storage_update(&mut solver, &mut expected, &[singleton], &[original]);
    storage_check_solve(&mut solver, &expected, &signs, true);
    assert!(solver.use_arrow);
    storage_update(&mut solver, &mut expected, &diagonal, &restore_diag);

    // Wrong-sign pivot follows the same dynamic regularization as a fresh
    // same-valued solver.
    solver.settings.dynamic_regularization_enable = true;
    storage_update(&mut solver, &mut expected, &[singleton], &[-T::one()]);
    storage_check_solve(&mut solver, &expected, &signs, false);
    assert!(solver.use_arrow);
    assert!(solver.regularize_count > 0);
    storage_matrix_same(&input, &untouched);
    // Any nonfinite H, B or C entry rejects the arrow. Exercise the arrow
    // directly so this check does not depend on QDLDL's NaN policy.
    for p in [
        storage_position(&expected, 0, 0),
        storage_position(&expected, 0, 1),
        storage_position(&expected, 1, 1),
    ] {
        for bad in [T::nan(), T::infinity(), -T::infinity()] {
            let mut input = expected.clone();
            input.nzval[p] = bad;
            assert!(!solver.factor_arrow(&input));
            assert!(solver.factor_arrow(&expected));
        }
    }
    for (leaf, &ptrs) in solver.leaves.iter().zip(&pointers) {
        assert_eq!((leaf.y.as_ptr(), leaf.factor.l.as_ptr()), ptrs);
    }
}

#[test]
fn arrow_storage_updates_f64() {
    storage_update_parity::<f64>();
}
#[test]
fn arrow_storage_updates_mpfr512() {
    storage_update_parity::<sdpx_arithmetic::Bits512>();
}
#[test]
fn arrow_storage_updates_mpfr768() {
    storage_update_parity::<sdpx_arithmetic::Bits768>();
}
