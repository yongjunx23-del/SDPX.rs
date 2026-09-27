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

// Forming M^T M loses the small singular direction for these SPD inputs.
// Check against the 2x2 determinant identity, at the declared precision.
fn graded_nt_scaling<T: FloatT>() {
    let one = T::one();
    let two = one + one;
    let bits = T::precision_bits() as i32;
    let shift = 5 * bits / 8;
    let delta = two.powi(-shift);
    let t = two.powi(-shift / 2);
    let mut S = Matrix::<T>::zeros((2, 2));
    let mut Z = Matrix::<T>::zeros((2, 2));
    S[(0, 0)] = delta;
    S[(0, 1)] = t;
    S[(1, 0)] = t;
    S[(1, 1)] = two;
    Z[(0, 0)] = one;
    Z[(1, 1)] = one;
    Z[(0, 1)] = one - delta;
    Z[(1, 0)] = one - delta;
    let mut s = vec![T::zero(); 3];
    let mut z = s.clone();
    mat_to_svec(&mut s, &S);
    mat_to_svec(&mut z, &Z);
    svec_to_mat(&mut S, &s);
    svec_to_mat(&mut Z, &z);
    let det_s = S[(0, 0)] * S[(1, 1)] - S[(0, 1)] * S[(1, 0)];
    let det_z = Z[(0, 0)] * Z[(1, 1)] - Z[(0, 1)] * Z[(1, 0)];
    assert!(det_s > T::zero() && det_z > T::zero());
    let expected = (det_s * det_z).sqrt();
    let mut cone = PSDTriangleCone::<T>::new(2);
    assert!(cone.update_scaling(&s, &z, one, ScalingStrategy::PrimalDual));
    assert!(cone
        .data
        .R
        .data()
        .iter()
        .chain(cone.data.Rinv.data())
        .all(|x| x.is_finite()));
    let relative = ((cone.data.λ[0] * cone.data.λ[1] - expected) / expected).abs();
    // The matrices consume about 5/8 of the precision through conditioning.
    let tolerance = two.powi(-bits / 4);
    assert!(
        relative.is_finite() && relative < tolerance,
        "singular product: {relative}"
    );
    let mut product = Matrix::<T>::zeros((2, 2));
    product.mul(&cone.data.R, &cone.data.Rinv, one, T::zero());
    for j in 0..2 {
        for i in 0..2 {
            let expected = if i == j { one } else { T::zero() };
            assert!((product[(i, j)] - expected).abs() < tolerance, "R * Rinv");
        }
    }
}
#[test]
fn graded_nt_scaling_mpfr() {
    graded_nt_scaling::<MpFloat<2>>();
    graded_nt_scaling::<MpFloat<4>>();
    graded_nt_scaling::<MpFloat<8>>();
    graded_nt_scaling::<MpFloat<12>>();
    graded_nt_scaling::<MpFloat<16>>();
    graded_nt_scaling::<MpFloat<32>>();
}

fn graded_hessian_action<T: FloatT>() {
    let mut cone = PSDTriangleCone::<T>::new(2);
    cone.set_identity_scaling();
    let t = T::epsilon();
    cone.data.R[(0, 0)] = T::one();
    cone.data.R[(0, 1)] = T::zero();
    cone.data.R[(1, 0)] = T::one();
    cone.data.R[(1, 1)] = t;
    cone.data.G.data_mut().fill(T::zero());
    cone.data
        .G
        .syrk(&cone.data.R, T::one(), T::zero(), MatrixTriangle::Triu);
    // Authoritative smat(x) is exactly a*[1 -1; -1 1]. The tiny
    // Hessian response must survive even when rounded R*R' has rank one.
    let a = T::FRAC_1_SQRT_2();
    let x = vec![a, -T::one(), a];
    let mut ref_work = vec![T::zero(); 3];
    let mut expected = ref_work.clone();
    cone.mul_W(MatrixShape::N, &mut ref_work, &x, T::one(), T::zero());
    cone.mul_W(
        MatrixShape::T,
        &mut expected,
        &ref_work,
        T::one(),
        T::zero(),
    );
    assert!(
        expected[2] > T::zero(),
        "factored action lost tiny response"
    );
    let analytic = ((a * t) * t) * t * t;
    assert!(
        (expected[2] - analytic).abs() / analytic <= T::epsilon() * T::from_usize(128).unwrap()
    );
    let mut actual = vec![T::zero(); 3];
    let mut work = actual.clone();
    cone.mul_Hs(&mut actual, &x, &mut work);
    let relative = (actual[2] - expected[2]).abs() / expected[2];
    assert!(
        relative <= T::epsilon() * T::from_usize(128).unwrap(),
        "bits={} tiny response relative error={}",
        T::precision_bits(),
        relative
    );
}
#[test]
fn graded_hessian_f64() {
    graded_hessian_action::<f64>();
}
#[test]
fn graded_hessian_128() {
    graded_hessian_action::<MpFloat<2>>();
}
#[test]
fn graded_hessian_256() {
    graded_hessian_action::<MpFloat<4>>();
}
#[test]
fn graded_hessian_512() {
    graded_hessian_action::<MpFloat<8>>();
}
#[test]
fn graded_hessian_768() {
    graded_hessian_action::<MpFloat<12>>();
}
#[test]
fn graded_hessian_1024() {
    graded_hessian_action::<MpFloat<16>>();
}
#[test]
fn graded_hessian_2048() {
    graded_hessian_action::<MpFloat<32>>();
}
