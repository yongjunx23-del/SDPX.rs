#![cfg(feature = "sdp")]

use sdpx_solver::{algebra::*, solver::*};

fn num<T: FloatT>(value: f64) -> T {
    T::from_f64(value).unwrap()
}

fn settings<T: FloatT>(form: &str) -> DefaultSettings<T> {
    let tol = num(if T::precision_bits() > 53 {
        1e-28
    } else {
        1e-9
    });
    DefaultSettings {
        verbose: false,
        max_threads: 1,
        direct_solve_method: "qdldl".into(),
        kkt_form: form.into(),
        equilibrate_enable: true,
        presolve_enable: false,
        chordal_decomposition_enable: false,
        input_sparse_dropzeros: false,
        tol_feas: tol,
        tol_gap_abs: tol,
        tol_gap_rel: tol,
        ..DefaultSettings::default()
    }
}

// Same fixed external allowance as sampled_solver.rs. Neither settings nor
// this gate adapt to solver status or to the formulation under test.
fn bound<T: FloatT>() -> T {
    num::<T>(128.) * settings::<T>("augmented").tol_feas
}

// Independent explicit coefficients in original coordinates, without using
// SampledOperator::materialize/apply or any solver workspace/scaling.
fn explicit_matrix<T: FloatT>(linear: &CscMatrix<T>, blocks: &[SampledBlock<T>]) -> CscMatrix<T> {
    let mut columns = vec![vec![T::zero(); linear.m]; linear.n];
    for j in 0..linear.n {
        for idx in linear.colptr[j]..linear.colptr[j + 1] {
            columns[j][linear.rowval[idx]] += linear.nzval[idx];
        }
    }
    for block in blocks {
        let h = block.basis_rows;
        let mut column = block.column_start;
        for s in 0..block.dim {
            for r in 0..=s {
                for k in 0..block.basis_cols {
                    let weight = block.weights[column - block.column_start];
                    for j in 0..h {
                        for i in 0..if r == s { j + 1 } else { h } {
                            let row = r * h + i;
                            let col = s * h + j;
                            let factor = if r != s {
                                T::FRAC_1_SQRT_2()
                            } else if i != j {
                                T::SQRT_2()
                            } else {
                                T::one()
                            };
                            columns[column][block.row_start + col * (col + 1) / 2 + row] +=
                                weight * block.basis[i + h * k] * block.basis[j + h * k] * factor;
                        }
                    }
                    column += 1;
                }
            }
        }
    }
    let mut ptr = vec![0];
    let mut rows = Vec::new();
    let mut values = Vec::new();
    for column in columns {
        for (i, value) in column.into_iter().enumerate() {
            if value != T::zero() {
                rows.push(i);
                values.push(value);
            }
        }
        ptr.push(values.len());
    }
    CscMatrix::new(linear.m, linear.n, ptr, rows, values)
}

fn assert_psd<T: FloatT>(svec: &[T], n: usize) {
    let mut matrix = vec![vec![T::zero(); n]; n];
    let mut index = 0;
    let mut scale = T::one();
    for j in 0..n {
        for i in 0..=j {
            let value = if i == j {
                svec[index]
            } else {
                svec[index] * T::FRAC_1_SQRT_2()
            };
            matrix[i][j] = value;
            matrix[j][i] = value;
            scale = scale.max(value.abs());
            index += 1;
        }
    }
    // Positive definiteness after a fixed acceptance-sized diagonal shift
    // checks the returned semidefinite point without a Float64 eigensolve.
    for i in 0..n {
        matrix[i][i] += bound::<T>() * scale;
    }
    for k in 0..n {
        let pivot = matrix[k][k];
        assert!(pivot.is_finite() && pivot > T::zero());
        for i in k + 1..n {
            for j in i..n {
                let updated = matrix[j][i] - matrix[j][k] * matrix[i][k] / pivot;
                matrix[j][i] = updated;
                matrix[i][j] = updated;
            }
        }
    }
}

fn audit<T: FloatT>(
    p: &CscMatrix<T>,
    q: &[T],
    a: &CscMatrix<T>,
    b: &[T],
    cones: &[SupportedConeT<T>],
    solution: &DefaultSolution<T>,
) {
    assert_eq!(solution.status, SolverStatus::Solved);
    let mut primal: Vec<_> = solution.s.iter().zip(b).map(|(&s, &b)| s - b).collect();
    let mut primal_work: Vec<_> = solution
        .s
        .iter()
        .zip(b)
        .map(|(&s, &b)| s.abs() + b.abs())
        .collect();
    let mut dual = q.to_vec();
    let mut dual_work: Vec<_> = q.iter().map(|q| q.abs()).collect();
    for j in 0..a.n {
        for idx in a.colptr[j]..a.colptr[j + 1] {
            let i = a.rowval[idx];
            primal[i] += a.nzval[idx] * solution.x[j];
            primal_work[i] += (a.nzval[idx] * solution.x[j]).abs();
            dual[j] += a.nzval[idx] * solution.z[i];
            dual_work[j] += (a.nzval[idx] * solution.z[i]).abs();
        }
        for idx in p.colptr[j]..p.colptr[j + 1] {
            let i = p.rowval[idx];
            if i > j {
                continue;
            }
            dual[i] += p.nzval[idx] * solution.x[j];
            dual_work[i] += (p.nzval[idx] * solution.x[j]).abs();
            if i != j {
                dual[j] += p.nzval[idx] * solution.x[i];
                dual_work[j] += (p.nzval[idx] * solution.x[i]).abs();
            }
        }
    }
    for (error, work) in primal
        .iter()
        .zip(&primal_work)
        .chain(dual.iter().zip(&dual_work))
    {
        assert!(
            error.is_finite() && error.abs() <= bound::<T>() * T::one().max(*work),
            "original equation error={error}, work={work}"
        );
    }
    let mut row = 0;
    for cone in cones {
        match cone {
            PSDTriangleConeT(n) => {
                let count = n * (n + 1) / 2;
                assert_psd(&solution.s[row..row + count], *n);
                assert_psd(&solution.z[row..row + count], *n);
                row += count;
            }
            NonnegativeConeT(n) => {
                for &value in solution.s[row..row + n]
                    .iter()
                    .chain(&solution.z[row..row + n])
                {
                    assert!(value >= -bound::<T>());
                }
                row += n;
            }
            _ => unreachable!(),
        }
    }
    let gap = (solution.obj_val - solution.obj_val_dual).abs();
    assert!(gap <= bound::<T>() * T::one().max(solution.obj_val.abs()));
}

fn parity_fixture<T: FloatT>(
    with_zeros: bool,
) -> (
    CscMatrix<T>,
    Vec<T>,
    CscMatrix<T>,
    Vec<T>,
    Vec<SupportedConeT<T>>,
    Vec<SampledBlock<T>>,
) {
    let (m, n) = (22, 8);
    let mut p = CscMatrix::identity(n);
    for (i, value) in p.nzval.iter_mut().enumerate() {
        *value = num::<T>(2.).powi(i as i32);
    }
    let target = [0.5, 1.5, -0.75, 0.5, -1.25, 1.75, -0.5, 0.75];
    let q = target
        .iter()
        .enumerate()
        .map(|(i, t)| -p.nzval[i] * num(*t))
        .collect();
    // Includes a preserved ordinary-row stored zero, and PSD-row stored zeros
    // in both columns outside the canonical range [1,7).
    let linear = CscMatrix::new(
        m,
        n,
        vec![0, 2, 2, 3, 3, 3, 3, 3, 5],
        vec![0, 2, 0, 1, 12],
        vec![-T::one(), T::zero(), T::zero(), -T::one(), T::zero()],
    );
    let mut b = vec![T::zero(); m];
    b[0] = num(2.);
    b[1] = num(2.);
    for start in [2, 12] {
        for j in 0..4 {
            b[start + j * (j + 1) / 2 + j] = num(1. + j as f64 / 4.);
        }
        b[start + 1] = num::<T>(0.125) * T::SQRT_2();
        b[start + 8] = num::<T>(-0.0625) * T::SQRT_2();
    }
    let mut blocks = vec![
        SampledBlock {
            row_start: 2,
            column_start: 1,
            dim: 2,
            basis_rows: 2,
            basis_cols: 2,
            basis: vec![num(1.), num(0.25), num(-0.125), num(2.)],
            weights: vec![num(1.), num(-2.), num(-1.), num(0.5), num(1.5), num(-0.75)],
        },
        SampledBlock {
            row_start: 12,
            column_start: 1,
            dim: 2,
            basis_rows: 2,
            basis_cols: 2,
            basis: vec![num(0.5), num(-0.25), num(0.125), num(1.5)],
            weights: vec![
                num(-0.5),
                num(1.25),
                num(0.75),
                num(-1.5),
                num(-1.),
                num(2.),
            ],
        },
    ];
    if with_zeros {
        blocks[0].weights[2] = T::zero();
        blocks[1].basis[2] = T::zero();
        blocks[1].basis[3] = T::zero();
    }
    (
        p,
        q,
        linear,
        b,
        vec![
            NonnegativeConeT(2),
            PSDTriangleConeT(4),
            PSDTriangleConeT(4),
        ],
        blocks,
    )
}

fn parity_comparison<T: FloatT>() {
    for with_zeros in [false, true] {
        let (p, q, linear, b, cones, blocks) = parity_fixture::<T>(with_zeros);
        let original = explicit_matrix(&linear, &blocks);
        let operator = SampledOperator::new(linear.clone(), blocks.clone()).unwrap();
        assert!(operator.linear().rowval.iter().all(|&r| r < 2));
        assert!(operator.linear().nzval.iter().any(|&v| v == T::zero()));
        assert_eq!(operator.blocks()[0].column_count(), 6);
        for form in ["condensed", "augmented"] {
            let mut generic =
                DefaultSolver::new(&p, &q, &original, &b, &cones, settings(form)).unwrap();
            let mut sampled = DefaultSolver::new_sampled(
                &p,
                &q,
                &linear,
                &b,
                &cones,
                blocks.clone(),
                settings(form),
            )
            .unwrap();
            assert_eq!(
                sampled.info.linsolver.name,
                if form == "condensed" {
                    "condensed_sampled_qdldl"
                } else {
                    "qdldl"
                }
            );
            let eq = &sampled.data.equilibration;
            assert!(eq.d.iter().any(|&v| v != T::one()));
            for start in [2, 12] {
                assert!(eq.e[start..start + 10].iter().all(|&v| v == eq.e[start]));
            }
            generic.solve();
            sampled.solve();
            audit(&p, &q, &original, &b, &cones, &generic.solution);
            audit(&p, &q, &original, &b, &cones, &sampled.solution);
            let scale = T::one().max(generic.solution.obj_val.abs());
            assert!(
                (sampled.solution.obj_val - generic.solution.obj_val).abs() <= bound::<T>() * scale
            );
        }
    }
}

#[test]
fn dim2_signed_parities_ruiz_f64() {
    parity_comparison::<f64>();
}
#[test]
fn dim2_signed_parities_ruiz_mpfr256() {
    parity_comparison::<sdpx_arithmetic::Bits256>();
}
#[test]
fn dim2_signed_parities_ruiz_mpfr512() {
    parity_comparison::<sdpx_arithmetic::Bits512>();
}

fn structural_fallback<T: FloatT>(presolve: bool) {
    let h = if presolve { 2 } else { 4 };
    let offset = usize::from(presolve);
    let m = offset + h * (h + 1) / 2;
    let mut basis = vec![T::zero(); h * h];
    for i in 0..h {
        basis[i + i * h] = T::one();
    }
    let block = SampledBlock {
        row_start: offset,
        column_start: 0,
        dim: 1,
        basis_rows: h,
        basis_cols: h,
        basis,
        weights: vec![T::one(); h],
    };
    let linear = CscMatrix::zeros((m, h));
    let original = explicit_matrix(&linear, &[block.clone()]);
    let p = CscMatrix::identity(h);
    let q = vec![num::<T>(-2.); h];
    let mut b = vec![T::zero(); m];
    if presolve {
        b[0] = num(sdpx_solver::get_infinity());
    }
    for j in 0..h {
        b[offset + j * (j + 1) / 2 + j] = T::one();
    }
    let mut cones = Vec::new();
    if presolve {
        cones.push(NonnegativeConeT(1));
    }
    cones.push(PSDTriangleConeT(h));
    let mut settings = settings::<T>("condensed");
    settings.presolve_enable = presolve;
    settings.chordal_decomposition_enable = !presolve;
    settings.chordal_decomposition_merge_method = "none".into();
    let mut generic = DefaultSolver::new(&p, &q, &original, &b, &cones, settings.clone()).unwrap();
    let mut sampled =
        DefaultSolver::new_sampled(&p, &q, &linear, &b, &cones, vec![block], settings).unwrap();
    // Prove preprocessing actually changed the matrix, rather than merely
    // enabling a setting on a fixture that never takes the fallback path.
    if presolve {
        assert_eq!(sampled.data.m, m - 1);
    } else {
        assert!(sampled.data.m != m || sampled.data.n != h);
    }
    assert_eq!(
        sampled.info.linsolver.name,
        if presolve {
            "condensed_sampled_qdldl"
        } else {
            "condensed_qdldl"
        }
    );
    generic.solve();
    sampled.solve();
    audit(&p, &q, &original, &b, &cones, &generic.solution);
    audit(&p, &q, &original, &b, &cones, &sampled.solution);
    for &x in &sampled.solution.x {
        assert!((x - T::one()).abs() <= bound::<T>());
    }
    assert!(
        (sampled.solution.obj_val - generic.solution.obj_val).abs()
            <= bound::<T>() * T::one().max(generic.solution.obj_val.abs())
    );
}

#[test]
fn actual_presolve_retains_factors_f64() {
    structural_fallback::<f64>(true);
}
#[test]
fn actual_presolve_retains_factors_mpfr256() {
    structural_fallback::<sdpx_arithmetic::Bits256>(true);
}
#[test]
fn actual_chordal_fallback_f64() {
    structural_fallback::<f64>(false);
}
#[test]
fn actual_chordal_fallback_mpfr256() {
    structural_fallback::<sdpx_arithmetic::Bits256>(false);
}
