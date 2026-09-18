use super::*;
use crate::solver::core::ScalingStrategy;
use sdpx_arithmetic::Bits128;

// Repeated small PSD blocks share local column groups, while equality
// rows touch every variable. Dimensions are deliberately generic and
// chosen so charging a dense primal factor would reject this structure.
fn selector_local_problem(
    independent_orthant: Option<bool>,
) -> (CscMatrix<f64>, CscMatrix<f64>, CompositeCone<f64>) {
    let (groups, columns_per_group, side, equalities) = (10, 18, 10, 8);
    let n = groups * columns_per_group;
    let psd_rows = triangular_number(side);
    let all_psd_rows = 2 * groups * psd_rows;
    let nn_rows = match independent_orthant {
        None => 0,
        Some(true) => n,
        Some(false) => 1,
    };
    let mut kinds = vec![SupportedConeT::PSDTriangleConeT(side); 2 * groups];
    kinds.push(SupportedConeT::ZeroConeT(equalities));
    if nn_rows != 0 {
        kinds.push(SupportedConeT::NonnegativeConeT(nn_rows));
    }
    let mut colptr = vec![0];
    let mut rowval = Vec::new();
    for col in 0..n {
        let group = col / columns_per_group;
        let coordinate = col % columns_per_group;
        rowval.push(2 * group * psd_rows + coordinate);
        rowval.push((2 * group + 1) * psd_rows + coordinate);
        rowval.extend(all_psd_rows..all_psd_rows + equalities);
        if let Some(independent) = independent_orthant {
            rowval.push(all_psd_rows + equalities + if independent { col } else { 0 });
        }
        colptr.push(rowval.len());
    }
    let nzval = vec![1.; rowval.len()];
    let A = CscMatrix::new(
        all_psd_rows + equalities + nn_rows,
        n,
        colptr,
        rowval,
        nzval,
    );
    (CscMatrix::identity(n), A, CompositeCone::new(&kinds))
}

fn reordered_original_coordinates<T: FloatT>() {
    let (side, cols) = (4, 5);
    let kinds = [SupportedConeT::PSDTriangleConeT(side)];
    let mut cones = CompositeCone::new(&kinds);
    let rows = triangular_number(side);
    let lengths = [10, 1, 6, 2, 3];
    let mut data = vec![vec![T::zero(); cols]; rows];
    for j in 0..cols {
        for i in 0..lengths[j] {
            data[i][j] = ((i + j + 1) as f64 / 16.).as_T();
        }
    }
    let mut a = CscMatrix::from(&data);
    let p = CscMatrix::identity(cols);
    let settings = CoreSettings::<T>::default();
    let mut solver = CondensedKKTSolver::new(&p, &a, &kinds, &cones, &settings);
    let psd = match &solver.blocks[0].scaling {
        Scaling::Psd(p) => p,
        _ => unreachable!(),
    };
    assert_eq!(
        psd.columns.iter().map(|c| c.index).collect::<Vec<_>>(),
        [1, 3, 4, 2, 0]
    );
    assert!(solver.schur.check_format().is_ok());
    let (mut z, mut slack) = (vec![T::zero(); rows], vec![T::zero(); rows]);
    cones.unit_initialization(&mut z, &mut slack);
    for update in 0..2 {
        if update == 1 {
            a.nzval[0] = T::zero();
            data[0][0] = T::zero();
            solver.update_A(&a);
        }
        assert!(cones.update_scaling(&slack, &z, T::one(), ScalingStrategy::PrimalDual));
        assert!(solver.update(&cones, &settings));
        for j in 0..cols {
            for i in 0..=j {
                let expected = data
                    .iter()
                    .fold(if i == j { T::one() } else { T::zero() }, |sum, row| {
                        row[i].mul_add(row[j], sum)
                    });
                let got = solver.schur.nzval[schur_position(&solver.schur, i, j)];
                assert!(
                    (got - expected).abs()
                        <= T::from_f64(4096.).unwrap() * T::epsilon() * (T::one() + expected.abs())
                );
            }
        }
    }
}
#[test]
fn reordered_coordinates_f64() {
    reordered_original_coordinates::<f64>();
}
#[test]
fn reordered_coordinates_256() {
    reordered_original_coordinates::<sdpx_arithmetic::Bits256>();
}
#[test]
fn reordered_coordinates_512() {
    reordered_original_coordinates::<sdpx_arithmetic::Bits512>();
}

#[test]
fn compact_panel_late_coordinates_and_changing_width() {
    let (n, cols) = (8, 150);
    let rows = triangular_number(n);
    let mut data = vec![vec![0.; cols]; rows];
    for j in 0..cols {
        let positions = if j < 64 {
            vec![0]
        } else if j < 128 {
            vec![1, 2]
        } else {
            vec![0, 3 + (j - 128) % 30, 35]
        };
        for i in positions {
            data[i][j] = (j + i + 1) as f64 / 17.;
        }
    }
    let mut a = CscMatrix::from(&data);
    let mut p = PsdBlock::new(n, &a, &(0..rows));
    for j in 0..n {
        for i in 0..n {
            p.Ginv[(i, j)] = if i == j {
                2.
            } else {
                0.1 / (1 + i.abs_diff(j)) as f64
            };
        }
    }
    for (b, c) in p.columns.iter_mut().enumerate() {
        c.schur_positions = (0..=b).map(|a| triangular_number(b) + a).collect();
        c.sparse = false;
    }
    let mut previous_width = usize::MAX;
    for update in 0..2 {
        if update == 1 {
            for (j, c) in p.columns.iter_mut().enumerate() {
                if c.entries.len() == 1 {
                    a.nzval[c.entries[0].position] = 1.;
                }
                c.sparse = j % 7 == 0;
            }
        }
        p.dense_vectors.fill(f64::NAN);
        let mut got = vec![f64::NAN; triangular_number(cols)];
        p.compute_schur(&a.nzval, |_, _, pos, v| got[pos] = v);
        assert!(p.dense_indices.len() < previous_width);
        previous_width = p.dense_indices.len();
        assert!(p.dense_vectors.len() < rows * previous_width / 2);
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            for skip in [false, true] {
                let mut generic = vec![f64::NAN; triangular_number(cols)];
                let mut accelerated = generic.clone();
                p.compute_schur_dense_impl(&a.nzval, skip, |_, _, pos, v| generic[pos] = v);
                // SAFETY: both CPU features were checked above.
                unsafe {
                    p.compute_schur_dense_fma(&a.nzval, skip, |_, _, pos, v| accelerated[pos] = v);
                }
                for (plain, fast) in generic.iter().zip(&accelerated) {
                    assert_eq!(plain.to_bits(), fast.to_bits());
                }
            }
        }
        for b in 0..cols {
            let mut coeff = vec![0.; rows];
            for e in &p.columns[b].entries {
                coeff[triangular_number(e.j) + e.i] = a.nzval[e.position];
            }
            let mut m = Matrix::zeros((n, n));
            let mut tmp = Matrix::zeros((n, n));
            let mut out = Matrix::zeros((n, n));
            svec_to_mat(&mut m, &coeff);
            tmp.mul(&p.Ginv, &m, 1., 0.);
            out.mul(&tmp, &p.Ginv, 1., 0.);
            mat_to_svec(&mut coeff, &out);
            for left in 0..=b {
                let expected = p.columns[left].entries.iter().fold(0., |sum, e| {
                    a.nzval[e.position].mul_add(coeff[triangular_number(e.j) + e.i], sum)
                });
                assert!(
                    (got[triangular_number(b) + left] - expected).abs()
                        <= 1e-12 * (1. + expected.abs())
                );
            }
        }
    }
}

#[test]
fn equal_columns_reuse_matches_explicit_congruence_and_updates() {
    let n = 8;
    let cols = 139;
    let rows = triangular_number(n);
    let mut data = vec![vec![0.; cols]; rows];
    for j in 0..cols {
        for i in 0..rows {
            if (i + 3 * (j % 11)) % 7 < 3 {
                data[i][j] = ((i * 13 + (j % 11) * 3) % 17) as f64 / 19. - 0.4;
            }
        }
    }
    let mut A = CscMatrix::from(&data);
    let mut p = PsdBlock::new(n, &A, &(0..rows));
    for j in 0..n {
        for i in 0..n {
            p.Ginv[(i, j)] = if i == j {
                2.
            } else {
                0.1 / (1 + i.abs_diff(j)) as f64
            };
        }
    }
    for (b, c) in p.columns.iter_mut().enumerate() {
        c.schur_positions = (0..=b).map(|a| triangular_number(b) + a).collect();
        c.sparse = b % 5 == 0;
    }
    for update in 0..2 {
        if update == 1 {
            for (i, v) in A.nzval.iter_mut().enumerate() {
                *v *= if i % 3 == 0 { 0. } else { 1.125 };
            }
            for (b, c) in p.columns.iter_mut().enumerate() {
                c.sparse = b % 11 == 0;
            }
        }
        let mut serial = vec![0.; triangular_number(cols)];
        let groups = std::mem::take(&mut p.column_groups);
        p.compute_schur_selected(&A.nzval, false, |_, _, pos, v| serial[pos] = v);
        p.column_groups = groups;
        let mut reused = serial.clone();
        reused.fill(f64::NAN);
        // Unwritten panel cells must never be read, even across A updates.
        p.dense_vectors.fill(f64::NAN);
        p.compute_schur_selected(&A.nzval, false, |_, _, pos, v| reused[pos] = v);
        if update == 0 {
            assert!(p.dense_indices.len() < cols / 2);
        }
        assert_eq!(serial, reused);
        // Independent full matrix products, including symmetric packing.
        for b in 0..cols {
            let mut coeff = vec![0.; rows];
            for e in &p.columns[b].entries {
                coeff[triangular_number(e.j) + e.i] = A.nzval[e.position];
            }
            let mut mat = Matrix::zeros((n, n));
            svec_to_mat(&mut mat, &coeff);
            let mut tmp = Matrix::zeros((n, n));
            let mut out = Matrix::zeros((n, n));
            tmp.mul(&p.Ginv, &mat, 1., 0.);
            out.mul(&tmp, &p.Ginv, 1., 0.);
            mat_to_svec(&mut coeff, &out);
            for a in 0..=b {
                let expected: f64 = p.columns[a]
                    .entries
                    .iter()
                    .map(|e| A.nzval[e.position] * coeff[triangular_number(e.j) + e.i])
                    .sum();
                let got = reused[triangular_number(b) + a];
                assert!((got - expected).abs() <= 1e-12 * (1. + expected.abs()));
            }
        }
    }
}

fn streamed_exact_reuse<T: FloatT>() {
    let n = 3;
    let cols = 12;
    let rows = triangular_number(n);
    let mut data = vec![vec![T::one(); cols]; rows];
    // These collide in the f64 bucket at high precision but are not equal.
    data[0][1] += T::epsilon();
    let mut a = CscMatrix::from(&data);
    let mut p = PsdBlock::new(n, &a, &(0..rows));
    for i in 0..n {
        p.Ginv[(i, i)] = T::one();
    }
    for (b, c) in p.columns.iter_mut().enumerate() {
        c.sparse = false;
        c.schur_positions = (0..=b).map(|a| triangular_number(b) + a).collect();
    }
    for update in 0..2 {
        if update == 1 {
            a.nzval[0] += T::one();
        }
        let groups = std::mem::take(&mut p.column_groups);
        let mut expected = vec![T::zero(); triangular_number(cols)];
        p.compute_schur_selected(&a.nzval, false, |_, _, pos, v| expected[pos] = v);
        p.column_groups = groups;
        let mut got = vec![T::nan(); expected.len()];
        p.compute_schur_selected(&a.nzval, false, |_, _, pos, v| got[pos] = v);
        assert_eq!(got, expected);
        assert!(p.dense_indices.len() < cols / 2);
    }
}
#[test]
fn streamed_reuse_256() {
    streamed_exact_reuse::<sdpx_arithmetic::Bits256>();
}
#[test]
fn streamed_reuse_512() {
    streamed_exact_reuse::<sdpx_arithmetic::Bits512>();
}

#[test]
fn selector_prefers_local_psd_groups_with_global_equality_border() {
    let (mut P, mut A, cones) = selector_local_problem(None);
    let settings = CoreSettings::default();
    assert!(prefer_condensed(&P, &A, &cones, &settings));
    // Selection is invariant to coefficients, including all stored zeros.
    P.nzval.fill(0.);
    A.nzval.fill(0.);
    assert!(prefer_condensed(&P, &A, &cones, &settings));
}

#[test]
fn selector_counts_stored_edges_and_independent_orthant_rows() {
    let (P, mut A, cones) = selector_local_problem(None);
    let settings = CoreSettings::default();
    // One PSD block touching every column forces a dense primal clique.
    // The stored zero coefficients must still contribute to this graph.
    A.nzval.fill(0.);
    for col in 0..A.n {
        A.rowval[A.colptr[col]] = 0;
    }
    assert!(!prefer_condensed(&P, &A, &cones, &settings));

    let (_, A, cones) = selector_local_problem(None);
    let mut colptr = vec![0];
    let mut rowval = Vec::new();
    for col in 0..A.n {
        rowval.extend(0..=col);
        colptr.push(rowval.len());
    }
    let nzval = vec![0.; rowval.len()];
    let dense_P = CscMatrix::new(A.n, A.n, colptr, rowval, nzval);
    assert!(!prefer_condensed(&dense_P, &A, &cones, &settings));

    let (P, A, cones) = selector_local_problem(Some(true));
    assert!(prefer_condensed(&P, &A, &cones, &settings));
    let (P, A, cones) = selector_local_problem(Some(false));
    assert!(!prefer_condensed(&P, &A, &cones, &settings));
}

#[test]
fn selector_preserves_coarse_guards_and_dense_storage_comparison() {
    let (_, A, cones) = selector_local_problem(None);
    let settings = CoreSettings::default();
    assert!(!prefer_condensed(
        &CscMatrix::zeros((0, 0)),
        &CscMatrix::zeros((A.m, 0)),
        &cones,
        &settings
    ));
    let wide = CscMatrix::zeros((A.m, 300));
    assert!(!prefer_condensed(
        &CscMatrix::identity(wide.n),
        &wide,
        &cones,
        &settings
    ));
    let small = CompositeCone::new(&[SupportedConeT::PSDTriangleConeT(10)]);
    assert!(!prefer_condensed(
        &CscMatrix::identity(1),
        &CscMatrix::zeros((55, 1)),
        &small,
        &settings
    ));
    // A dense Schur is still worthwhile when its PSD block is much larger.
    let large = CompositeCone::new(&[SupportedConeT::PSDTriangleConeT(40)]);
    let A = CscMatrix::new(820, 20, (0..=20).collect(), (0..20).collect(), vec![1.; 20]);
    assert!(prefer_condensed(
        &CscMatrix::identity(20),
        &A,
        &large,
        &settings
    ));
}

fn mixed_operator<T: FloatT>() {
    let kinds = [
        SupportedConeT::PSDTriangleConeT(2),
        SupportedConeT::NonnegativeConeT(2),
        SupportedConeT::ZeroConeT(1),
        SupportedConeT::SecondOrderConeT(3),
    ];
    let mut cones = CompositeCone::new(&kinds);
    let mut P = CscMatrix::from(&[
        [T::from_f64(3.).unwrap(), T::from_f64(0.25).unwrap()],
        [T::zero(), T::from_f64(2.).unwrap()],
    ]);
    let mut A = CscMatrix::from(&[
        [T::one(), T::zero()],
        [T::from_f64(0.3).unwrap(), T::one()],
        [T::zero(), T::one()],
        [T::one(), T::from_f64(-0.5).unwrap()],
        [T::from_f64(0.25).unwrap(), T::one()],
        [T::one(), T::one()],
        [T::one(), T::from_f64(-0.25).unwrap()],
        [T::from_f64(0.5).unwrap(), T::one()],
        [T::one(), T::from_f64(0.5).unwrap()],
    ]);
    let conv = |v: &[f64]| {
        v.iter()
            .map(|&x| T::from_f64(x).unwrap())
            .collect::<Vec<_>>()
    };
    let mut s = conv(&[3., 0.2, 2., 1.3, 2.1, 0., 2., 0.2, 0.1]);
    let z = conv(&[1.5, -0.1, 2.5, 0.7, 1.2, 1., 3., -0.2, 0.1]);
    let mut settings = CoreSettings::<T>::default();
    settings.iterative_refinement_abstol = T::epsilon() * (1024.).as_T();
    settings.iterative_refinement_reltol = settings.iterative_refinement_abstol;
    let mut solver = CondensedKKTSolver::new(&P, &A, &kinds, &cones, &settings);
    let exact_x = conv(&[0.25, -0.5]);
    let exact_z = conv(&[0.5, -0.2, 0.25, 0.125, -0.25, 0.5, -0.25, 0.2, 0.1]);
    for update in 0..2 {
        if update != 0 {
            P.nzval[0] += T::from_f64(0.125).unwrap();
            A.nzval[0] *= T::from_f64(1.25).unwrap();
            s[0] += T::from_f64(0.25).unwrap();
            solver.update_P(&P);
            solver.update_A(&A);
        }
        assert!(cones.update_scaling(&s, &z, T::one(), ScalingStrategy::PrimalDual));
        assert!(solver.update(&cones, &settings));
        let mut bx = vec![T::zero(); 2];
        let mut bz = vec![T::zero(); 9];
        let mut hz = vec![T::zero(); 9];
        let mut scratch = vec![T::zero(); 9];
        P.sym_up().symv(&mut bx, &exact_x, T::one(), T::zero());
        A.t().gemv(&mut bx, &exact_z, T::one(), T::one());
        A.gemv(&mut bz, &exact_x, T::one(), T::zero());
        cones.mul_Hs(&mut hz, &exact_z, &mut scratch);
        for (b, h) in bz.iter_mut().zip(hz) {
            *b -= h;
        }
        let rhs: Vec<T> = bx.iter().chain(&bz).copied().collect();
        let mut point = vec![T::zero(); rhs.len()];
        assert!(solver.solve_raw(&mut point, &rhs, &settings));
        let mut reused = vec![T::zero(); rhs.len()];
        let mut fresh = vec![T::zero(); rhs.len()];
        assert!(solver.residual(&mut reused, &rhs, &point, true).is_finite());
        assert!(solver.residual(&mut fresh, &rhs, &point, false).is_finite());
        let rounding_bound = T::epsilon() * (256.).as_T() * rhs.norm_inf().max(T::one());
        for (&a, &b) in reused.iter().zip(&fresh) {
            assert!((a - b).abs() <= rounding_bound);
        }
        // A changed point must not read the previous raw forward product.
        point[0] += (0.125).as_T();
        solver.workz.fill(T::infinity());
        assert!(solver.residual(&mut fresh, &rhs, &point, false).is_finite());
        solver.setrhs(&bx, &bz);
        let (mut x, mut z) = (vec![T::zero(); 2], vec![T::zero(); 9]);
        assert!(solver.solve(Some(&mut x), Some(&mut z), &settings));
        // Independent original cone operator, not the cached inverse or S.
        let mut ex = bx.clone();
        let mut ez = bz.clone();
        P.sym_up().symv(&mut ex, &x, -T::one(), T::one());
        A.t().gemv(&mut ex, &z, -T::one(), T::one());
        A.gemv(&mut ez, &x, -T::one(), T::one());
        cones.mul_Hs(&mut scratch, &z, &mut solver.workh);
        for (e, h) in ez.iter_mut().zip(&scratch) {
            *e += *h;
        }
        let tolerance = settings.iterative_refinement_abstol
            + settings.iterative_refinement_reltol * bx.norm_inf().max(bz.norm_inf());
        assert!(ex.norm_inf().max(ez.norm_inf()) <= tolerance);

        for block in &mut solver.blocks {
            if let Scaling::Psd(p) = &mut block.scaling {
                for c in &mut p.columns {
                    c.sparse = true;
                }
            }
        }
        assert!(solver.assemble());
        let sparse = solver.schur.nzval.clone();
        for block in &mut solver.blocks {
            if let Scaling::Psd(p) = &mut block.scaling {
                for c in &mut p.columns {
                    c.sparse = false;
                }
            }
        }
        assert!(solver.assemble());
        assert!(
            solver.schur.nzval.norm_inf_diff(&sparse)
                <= settings.iterative_refinement_abstol * sparse.norm_inf()
        );
    }
}

#[test]
fn condensed_mixed_original_operator_f64() {
    mixed_operator::<f64>();
}
#[test]
fn condensed_mixed_original_operator_mpfr128() {
    mixed_operator::<Bits128>();
}

fn nonsymmetric_operator<T: FloatT>() {
    let kinds = [
        SupportedConeT::PSDTriangleConeT(2),
        SupportedConeT::ExponentialConeT(),
        SupportedConeT::PowerConeT(T::from_f64(0.4).unwrap()),
        SupportedConeT::GenPowerConeT(
            vec![T::from_f64(0.25).unwrap(), T::from_f64(0.75).unwrap()],
            2,
        ),
    ];
    let mut cones = CompositeCone::new(&kinds);
    let m = cones.numel();
    let (mut s, mut z) = (vec![T::zero(); m], vec![T::zero(); m]);
    cones.unit_initialization(&mut z, &mut s);
    assert!(cones.update_scaling(&s, &z, T::one(), ScalingStrategy::Dual));
    let rows: Vec<[T; 2]> = (0..m)
        .map(|i| {
            [
                T::from_usize(i % 5 + 1).unwrap() / T::from_usize(5).unwrap(),
                T::from_usize(i % 7 + 1).unwrap() / T::from_usize(7).unwrap(),
            ]
        })
        .collect();
    let A = CscMatrix::from(&rows);
    let P = CscMatrix::identity(2);
    let mut settings = CoreSettings::<T>::default();
    settings.iterative_refinement_abstol = T::epsilon() * (1024.).as_T();
    settings.iterative_refinement_reltol = settings.iterative_refinement_abstol;
    let mut kkt = CondensedKKTSolver::new(&P, &A, &kinds, &cones, &settings);
    assert!(kkt.update(&cones, &settings));
    let exact_x = [T::from_f64(0.25).unwrap(), T::from_f64(-0.5).unwrap()];
    let exact_z = vec![T::from_f64(0.125).unwrap(); m];
    let mut bx = exact_x.to_vec();
    A.t().gemv(&mut bx, &exact_z, T::one(), T::one());
    let mut bz = vec![T::zero(); m];
    let mut hs = vec![T::zero(); m];
    let mut scratch = vec![T::zero(); m];
    A.gemv(&mut bz, &exact_x, T::one(), T::zero());
    cones.mul_Hs(&mut hs, &exact_z, &mut scratch);
    for (b, &h) in bz.iter_mut().zip(&hs) {
        *b -= h;
    }
    kkt.setrhs(&bx, &bz);
    let (mut x, mut z) = (vec![T::zero(); 2], vec![T::zero(); m]);
    assert!(kkt.solve(Some(&mut x), Some(&mut z), &settings));
    let mut ex = bx.clone();
    let mut ez = bz.clone();
    P.sym_up().symv(&mut ex, &x, -T::one(), T::one());
    A.t().gemv(&mut ex, &z, -T::one(), T::one());
    A.gemv(&mut ez, &x, -T::one(), T::one());
    cones.mul_Hs(&mut hs, &z, &mut scratch);
    for (e, h) in ez.iter_mut().zip(hs) {
        *e += h;
    }
    let tolerance = settings.iterative_refinement_abstol
        + settings.iterative_refinement_reltol * bx.norm_inf().max(bz.norm_inf());
    assert!(ex.norm_inf().max(ez.norm_inf()) <= tolerance);
}

#[test]
fn condensed_retained_nonsymmetric_original_operator_f64() {
    nonsymmetric_operator::<f64>();
}
#[test]
fn condensed_retained_nonsymmetric_original_operator_mpfr128() {
    nonsymmetric_operator::<Bits128>();
}

#[test]
fn structural_psd_blocks_remain_sparse_with_global_equalities() {
    let types = [
        SupportedConeT::PSDTriangleConeT(2),
        SupportedConeT::PSDTriangleConeT(2),
        SupportedConeT::ZeroConeT(1),
    ];
    let cones = CompositeCone::new(&types);
    let P = CscMatrix::new(
        4,
        4,
        vec![0, 1, 2, 4, 5],
        vec![0, 1, 0, 2, 3],
        vec![1., 1., 0., 1., 1.],
    );
    let A = CscMatrix::from(&[
        [1., 0., 0., 0.],
        [1., 1., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 1., 0.],
        [0., 0., 1., 1.],
        [0., 0., 0., 1.],
        [1., 1., 1., 1.],
    ]);
    let solver = CondensedKKTSolver::new(&P, &A, &types, &cones, &CoreSettings::default());
    // Two 2x2 primal cliques plus one explicitly stored P edge, even zero.
    assert_eq!(solver.schur.nnz(), 7);
    assert_eq!(solver.retained_rows, vec![6]);
    assert_eq!(solver.retained_A.nnz(), 4);
    assert_eq!(
        solver.schur.rowval[solver.schur.colptr[3]..solver.schur.colptr[4]],
        [2, 3]
    );
}

#[test]
fn condensed_dependent_equalities_keep_bordered_refinement() {
    use crate::solver::{DefaultSolver, IPSolver, SolverStatus};
    for scale in [0.01, 1., 100.] {
        let P = CscMatrix::identity(3);
        let A = CscMatrix::from(&[
            [0., scale, scale],
            [0., scale, -scale],
            [scale, 2. * scale, -scale],
            [2. * scale, -scale, 3. * scale],
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
        ]);
        let b = [scale, scale, scale, scale, 1., 0., 1., 1.];
        let kinds = [
            SupportedConeT::ZeroConeT(4),
            SupportedConeT::PSDTriangleConeT(2),
            SupportedConeT::NonnegativeConeT(1),
        ];
        let settings = CoreSettings {
            verbose: false,
            kkt_form: "condensed".to_string(),
            ..CoreSettings::default()
        };
        let mut solver = DefaultSolver::new(&P, &[0.; 3], &A, &b, &kinds, settings).unwrap();
        solver.solve();
        assert_eq!(
            solver.solution.status,
            SolverStatus::PrimalInfeasible,
            "scale {scale}"
        );
        let mut atz = vec![0.; 3];
        A.t().gemv(&mut atz, &solver.solution.z, 1., 0.);
        assert!(atz.norm_inf() <= 1e-8 * solver.solution.z.norm_inf());
        assert!(b.dot(&solver.solution.z) < 0.);
    }
}

#[test]
fn forced_condensed_constant_program_keeps_public_contract() {
    use crate::solver::{DefaultSolver, IPSolver, SolverStatus};
    let kinds = [
        SupportedConeT::ZeroConeT(1),
        SupportedConeT::NonnegativeConeT(1),
        SupportedConeT::PSDTriangleConeT(2),
    ];
    let settings = CoreSettings {
        verbose: false,
        kkt_form: "condensed".to_string(),
        ..CoreSettings::default()
    };
    let mut solver = DefaultSolver::new(
        &CscMatrix::<f64>::zeros((0, 0)),
        &[],
        &CscMatrix::zeros((5, 0)),
        &[0., 1., 1., 0., 1.],
        &kinds,
        settings,
    )
    .unwrap();
    solver.solve();
    assert_eq!(solver.solution.status, SolverStatus::Solved);
    assert!(solver.solution.x.is_empty());
}
