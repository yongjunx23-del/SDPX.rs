#![allow(non_snake_case)]
#![allow(clippy::type_complexity)]
#![cfg(feature = "sdp")]
use sdpx_solver::{algebra::*, solver::*};

fn basic_sdp_data() -> (
    CscMatrix<f64>,
    Vec<f64>,
    CscMatrix<f64>,
    Vec<f64>,
    Vec<SupportedConeT<f64>>,
) {
    //  problem will be 3x3, so upper triangle
    //  of problem data has 6 entries

    let P = CscMatrix::identity(6);

    // A = [1. 1;1 0; 0 1]; A = [-A;A]
    let A = CscMatrix::identity(6);

    let c = vec![0.0; 6];
    let b = vec![-3., 1., 4., 1., 2., 5.];

    let cones = vec![PSDTriangleConeT(3)];

    (P, c, A, b, cones)
}

fn basic_sdp_solution() -> (Vec<f64>, f64) {
    let refsol = vec![
        -3.0729833267361095,
        0.3696004167288786,
        -0.022226685581313674,
        0.31441213129613066,
        -0.026739700851545107,
        -0.016084530571308823,
    ];
    let refobj = 4.840076866013861;

    (refsol, refobj)
}

#[test]
fn test_sdp_feasible() {
    let (P, c, A, b, cones) = basic_sdp_data();
    let (refsol, refobj) = basic_sdp_solution();

    let settings = DefaultSettings::default();

    let mut solver = DefaultSolver::new(&P, &c, &A, &b, &cones, settings).unwrap();

    solver.solve();

    assert_eq!(solver.solution.status, SolverStatus::Solved);
    assert!(solver.solution.x.dist(&refsol) <= 1e-6);
    assert!(f64::abs(solver.info.cost_primal - refobj) <= 1e-6);
}

#[test]
fn test_sdp_empty_cone() {
    let (P, c, A, b, mut cones) = basic_sdp_data();
    let (refsol, refobj) = basic_sdp_solution();

    cones.append(&mut vec![PSDTriangleConeT(0)]);

    let settings = DefaultSettings::default();

    let mut solver = DefaultSolver::new(&P, &c, &A, &b, &cones, settings).unwrap();

    solver.solve();

    assert_eq!(solver.solution.status, SolverStatus::Solved);
    assert!(solver.solution.x.dist(&refsol) <= 1e-6);
    assert!(f64::abs(solver.info.cost_primal - refobj) <= 1e-6);
}

#[test]
fn test_sdp_primal_infeasible() {
    let (P, c, A, mut b, mut cones) = basic_sdp_data();

    // this adds a negative definiteness constraint to x
    let mut A2 = A.clone();
    A2.negate();
    let A = CscMatrix::vcat(&A, &A2).unwrap();
    b.extend(vec![0.0; b.len()]);
    cones.extend([cones[0].clone()]);

    let settings = DefaultSettings::default();

    let mut solver = DefaultSolver::new(&P, &c, &A, &b, &cones, settings).unwrap();

    solver.solve();

    assert_eq!(solver.solution.status, SolverStatus::PrimalInfeasible);
}

#[test]
fn projection_matches_both_kkt_forms() {
    let (p, q, a, b, mut cones) = basic_sdp_data();
    cones.push(PSDTriangleConeT(0));
    let (expected, objective) = basic_sdp_solution();
    for form in ["augmented", "condensed"] {
        for threads in [1, 4] {
            let settings = DefaultSettings {
                kkt_form: form.into(),
                max_threads: threads,
                verbose: false,
                ..DefaultSettings::default()
            };
            let mut solver = DefaultSolver::new(&p, &q, &a, &b, &cones, settings).unwrap();
            solver.solve();
            assert_eq!(solver.solution.status, SolverStatus::Solved);
            assert!(solver.solution.x.dist(&expected) < 1e-6);
            assert!((solver.info.cost_primal - objective).abs() < 1e-6);
            let mut rp: Vec<_> = b
                .iter()
                .zip(&solver.solution.s)
                .map(|(b, s)| b - s)
                .collect();
            let mut rd = q.clone();
            for col in 0..a.n {
                for k in a.colptr[col]..a.colptr[col + 1] {
                    rp[a.rowval[k]] -= a.nzval[k] * solver.solution.x[col];
                    rd[col] += a.nzval[k] * solver.solution.z[a.rowval[k]];
                }
                for k in p.colptr[col]..p.colptr[col + 1] {
                    rd[p.rowval[k]] += p.nzval[k] * solver.solution.x[col];
                    if p.rowval[k] != col {
                        rd[col] += p.nzval[k] * solver.solution.x[p.rowval[k]];
                    }
                }
            }
            assert!(rp.norm_inf() < 1e-7 && rd.norm_inf() < 1e-7);
        }
    }
}
