use super::*;
use sdpx_arithmetic::MpFloat;

fn assert_close<T: FloatT>(actual: &[T], expected: &[T], n: usize) {
    let scale = expected.norm_inf().max(T::one());
    let tolerance = T::epsilon() * T::from_usize(128 * n.max(1) * n.max(1)).unwrap() * scale;
    assert!(
        actual.norm_inf_diff(expected) <= tolerance,
        "PSD Hessian action differs: residual {}, tolerance {}",
        actual.norm_inf_diff(expected),
        tolerance
    );
}

fn packed_action<T: FloatT>(packed: &[T], x: &[T]) -> Vec<T> {
    let mut y = vec![T::zero(); x.len()];
    let mut index = 0;
    for j in 0..x.len() {
        for i in 0..=j {
            y[i] += packed[index] * x[j];
            if i != j {
                y[j] += packed[index] * x[i];
            }
            index += 1;
        }
    }
    y
}

fn check_operator<T: FloatT>(cone: &mut PSDTriangleCone<T>) {
    let n = cone.n;
    let p = cone.numel();
    let mut reference = Matrix::zeros((p, p));
    skron(&mut reference, &cone.data.G.sym_up());
    let mut expected = vec![T::zero(); triangular_number(p)];
    reference.sym_up().pack_triu(&mut expected);
    let mut packed = vec![T::nan(); expected.len()];
    cone.get_Hs(&mut packed);
    // The formula and operation order are unchanged, so packing is exact.
    assert_eq!(packed, expected);
    if p == 0 {
        return;
    }

    let mut G = Matrix::zeros((n, n));
    G.mul(cone.scaling_R(), &cone.scaling_R().t(), T::one(), T::zero());
    for basis in 0..p {
        // Testing every svec basis includes all off-diagonal sqrt(2) factors.
        let mut x = vec![T::zero(); p];
        x[basis] = T::one();
        let packed_y = packed_action(&packed, &x);
        let mut y = vec![T::zero(); p];
        let mut work = vec![T::zero(); p];
        cone.mul_Hs(&mut y, &x, &mut work);
        assert_close(&packed_y, &y, n);

        // Independent matrix-coordinate oracle: svec(G * smat(x) * G).
        let mut X = Matrix::zeros((n, n));
        let mut tmp = Matrix::zeros((n, n));
        let mut Y = Matrix::zeros((n, n));
        svec_to_mat(&mut X, &x);
        tmp.mul(&G, &X, T::one(), T::zero());
        Y.mul(&tmp, &G, T::one(), T::zero());
        let mut direct = vec![T::zero(); p];
        mat_to_svec(&mut direct, &Y);
        assert_close(&packed_y, &direct, n);
    }
}

fn hessian_equivalence<T: FloatT>() {
    for n in 0usize..=4 {
        let mut cone = PSDTriangleCone::<T>::new(n);
        cone.set_identity_scaling();
        check_operator(&mut cone);
        if n == 0 {
            continue;
        }

        // Invertible nonsymmetric R with positive and negative entries makes
        // G SPD and exercises all four diagonal/off-diagonal packing cases.
        for j in 0..n {
            for i in 0..n {
                cone.data.R[(i, j)] = if i == j {
                    T::from_usize(i + 2).unwrap()
                } else if i < j {
                    -T::from_usize(i + j + 1).unwrap() / T::from_usize(8).unwrap()
                } else {
                    T::from_usize(i + 2 * j + 1).unwrap() / T::from_usize(7).unwrap()
                };
            }
        }
        cone.data.G.data_mut().fill(T::zero());
        cone.data
            .G
            .syrk(&cone.data.R, T::one(), T::zero(), MatrixTriangle::Triu);
        check_operator(&mut cone);

        // Also exercise productive writes in update_scaling, rather than just
        // seeding the cached matrix. S and Z are noncommuting SPD matrices.
        let mut S = Matrix::zeros((n, n));
        S.mul(&cone.data.R, &cone.data.R.t(), T::one(), T::zero());
        let mut Z = Matrix::zeros((n, n));
        Z.mul(&cone.data.R.t(), &cone.data.R, T::one(), T::zero());
        for i in 0..n {
            S[(i, i)] += T::one();
            Z[(i, i)] += T::from_usize(2).unwrap();
        }
        let mut s = vec![T::zero(); cone.numel()];
        let mut z = vec![T::zero(); cone.numel()];
        mat_to_svec(&mut s, &S);
        mat_to_svec(&mut z, &Z);
        assert!(cone.update_scaling(&s, &z, T::one(), ScalingStrategy::PrimalDual));
        check_operator(&mut cone);

        // Reset after nontrivial scaling must also reset the cached Hessian.
        cone.set_identity_scaling();
        check_operator(&mut cone);
    }
}

#[test]
fn psd_hessian_packed_matches_factored_f64() {
    hessian_equivalence::<f64>();
}

#[test]
fn psd_hessian_packed_matches_factored_mpfr512() {
    hessian_equivalence::<MpFloat<8>>();
}

#[test]
fn psd_hessian_constructor_cache_is_matrix_sized() {
    // Storage-contract check only; process RSS is measured separately by the
    // benchmark. The packed p(p+1)/2 output belongs to the selected KKT route.
    let cone = PSDTriangleCone::<f64>::new(64);
    assert_eq!(cone.data.G.size(), (64, 64));
    assert_eq!(cone.data.G.data().len(), 64 * 64);
}
