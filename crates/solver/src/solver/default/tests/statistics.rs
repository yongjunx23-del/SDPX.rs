use super::*;
use crate::solver::core::{traits::Info, SolverStatus};
use crate::solver::default::{DefaultInfo, DefaultResiduals, DefaultSettings};
use sdpx_arithmetic::MpFloat;

fn n<T: FloatT>(x: i32) -> T {
    T::from_i32(x).unwrap()
}
fn same<T: FloatT>(a: T, b: T) {
    if b.is_nan() {
        assert!(a.is_nan());
    } else {
        assert_eq!(a, b);
        assert_eq!(a.is_sign_negative(), b.is_sign_negative());
    }
}
/// High-precision norms use exact squares (rounded once), so they may
/// differ from the scaled recurrence in the last bits; f64 is unchanged.
fn same_norm<T: FloatT>(a: T, b: T) {
    if T::precision_bits() > 64 {
        close(a, b);
    } else {
        same(a, b);
    }
}
fn close<T: FloatT>(a: T, b: T) {
    assert!(
        (a - b).abs() <= n::<T>(256) * T::epsilon() * b.abs().max(T::one()),
        "{a} != {b}"
    );
}

struct SliceOwner<T> {
    values: [Vec<T>; 8],
    scales: [Vec<T>; 8],
    products: ResidualProducts<T>,
    componentwise: Option<T>,
}
impl<T: FloatT> SliceOwner<T> {
    fn view(&self) -> ResidualOwner<'_, T> {
        ResidualOwner {
            products: self.products,
            norms: std::array::from_fn(|k| {
                NormView::dense(self.values[k].as_slice(), self.scales[k].as_slice())
            }),
            dual_componentwise: self.componentwise,
        }
    }
}

fn ownership<T: FloatT>() {
    // Primal IDs 0..5 and row IDs 0..3 are distinct spaces. rx_inf has
    // already received two operator contributions at shared primal ID 2.
    let large = n::<T>(2).powi(30);
    let coupled = large + (-large + n::<T>(3));
    let values: [Vec<T>; 8] = [
        vec![1, 2, 3, 4, 5],
        vec![2, 3, 4],
        vec![5, 6, 7],
        vec![2, 4, 3, 8, 10],
        vec![1, 4, 9, 16, 25],
        vec![7, 8, 9],
        vec![1, 2, 3],
        vec![2, 3, 4, 5, 6],
    ]
    .map(|v| v.into_iter().map(n::<T>).collect());
    same(values[3][2], coupled);
    let scales: [Vec<T>; 8] = std::array::from_fn(|k| {
        (0..values[k].len())
            .map(|j| T::one() / n::<T>(1 + (j % 3) as i32))
            .collect()
    });
    let primal = [true, false, false, true, true, false, false, true];
    let make = |cols: &[usize], rows: &[usize], componentwise: Option<T>| {
        let gather = |source: &[Vec<T>; 8]| {
            std::array::from_fn(|k| {
                let ids = if primal[k] { cols } else { rows };
                ids.iter().map(|&j| source[k][j]).collect()
            })
        };
        let qx = cols
            .iter()
            .fold(T::zero(), |s, &j| s + n::<T>(j as i32 + 2) * values[0][j]);
        let xpx = cols
            .iter()
            .fold(T::zero(), |s, &j| s + values[0][j] * values[4][j]);
        let bz = rows
            .iter()
            .fold(T::zero(), |s, &j| s + n::<T>(j as i32 + 1) * values[1][j]);
        let sz = rows
            .iter()
            .fold(T::zero(), |s, &j| s + values[2][j] * values[1][j]);
        SliceOwner {
            values: gather(&values),
            scales: gather(&scales),
            products: ResidualProducts { qx, bz, sz, xpx },
            componentwise,
        }
    };
    let all = make(&[0, 1, 2, 3, 4], &[0, 1, 2], Some(n(2)));
    let single = ResidualSummary::from_owners([all.view()], None).unwrap();
    let original: [T; 8] = std::array::from_fn(|k| values[k].norm_scaled(&scales[k]));
    for (a, b) in single.norms().into_iter().zip(original) {
        same_norm(a, b);
    }
    same(single.products.qx, n(70));
    same(single.products.xpx, n(225));
    same(single.products.bz, n(20));
    same(single.products.sz, n(56));
    for width in [1, 2, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(width)
            .build()
            .unwrap();
        let parallel = ResidualSummary::from_owners([all.view()], Some(&pool)).unwrap();
        for (a, b) in parallel.norms().into_iter().zip(single.norms()) {
            same(a, b);
        }
    }

    // Two noncontiguous owners, plus no-row, no-primal and empty owners.
    let a = make(&[4, 0], &[1], Some(n(1)));
    let b = make(&[2], &[2, 0], Some(n(2)));
    let c = make(&[3, 1], &[], Some(T::zero()));
    let empty = make(&[], &[], Some(T::zero()));
    let split =
        ResidualSummary::from_owners([a.view(), empty.view(), b.view(), c.view()], None).unwrap();
    let row_only = make(&[], &[0, 1, 2], Some(n(2)));
    let col_only = make(&[4, 2, 0, 3, 1], &[], Some(T::zero()));
    let disjoint = ResidualSummary::from_owners([row_only.view(), col_only.view()], None).unwrap();
    for summary in [&split, &disjoint] {
        for (a, b) in summary.norms().into_iter().zip(original) {
            close(a, b);
        }
        same(summary.products.qx, n(70));
        same(summary.products.xpx, n(225));
        same(summary.products.bz, n(20));
        same(summary.products.sz, n(56));
        same(summary.dual_componentwise.unwrap(), n(2));
    }
    // The homogeneous complementarity term and degree are global: tau*kappa
    // is added once, not once per owner (including the empty owner).
    let tau = n::<T>(2);
    let kappa = n::<T>(3);
    let degree = n::<T>(3);
    same(
        (split.products.sz + tau * kappa) / (degree + T::one()),
        n::<T>(31) / n(2),
    );
    // A cross-owner quadratic edge contributes to both rows of symmetric P*x.
    // Reduce complete P*x coordinates before taking each owner's x dot P*x.
    // P = [2 3 0; 3 5 7; 0 7 11], x = [1 2 3].
    let px = [
        n::<T>(2) + n(6),
        n::<T>(3) + n(10) + n(21),
        n::<T>(14) + n(33),
    ];
    let a_xpx = n::<T>(1) * px[0] + n::<T>(3) * px[2];
    let b_xpx = n::<T>(2) * px[1];
    same(a_xpx + b_xpx, n(217));
    let mut pa = make(&[4, 0], &[], Some(T::zero()));
    let mut pb = make(&[2], &[], Some(T::zero()));
    pa.products.xpx = a_xpx;
    pb.products.xpx = b_xpx;
    let quadratic =
        ResidualSummary::from_owners([pa.view(), empty.view(), pb.view()], None).unwrap();
    same(quadratic.products.xpx, n(217));

    // Summarizing raw shared-coordinate contributions is observably wrong.
    let wrong = ScaledNorm::from_iter([large].into_iter())
        .merge(ScaledNorm::from_iter([-large + n::<T>(3)].into_iter()))
        .norm();
    assert!(wrong > coupled.abs() * n(100));

    let mut reference = DefaultInfo::<T>::new();
    reference.update_from_summary(&single, n(2), n(1), n(3), n(5), n(7));
    let ti = T::one() / n::<T>(2);
    let quadratic = n::<T>(225) * ti * ti / n::<T>(2);
    same(reference.cost_primal, (n::<T>(70) * ti + quadratic) * n(3));
    same(reference.cost_dual, (-n::<T>(20) * ti - quadratic) * n(3));
    // The formulas are checked on the summary's own norms (compared with
    // the scaled recurrence above).
    let norms = single.norms();
    let nx = norms[0];
    let nz = norms[1] * n(3);
    let ns = norms[2];
    same(reference.res_primal_inf, norms[3] * n(3) / nz.max(T::one()));
    same(
        reference.res_dual_inf,
        (norms[4] / nx.max(T::one())).max(norms[5] / (nx + ns).max(T::one())),
    );
    same(
        reference.res_primal,
        norms[6] * ti / (n::<T>(5) + nx * ti + ns * ti).max(T::one()),
    );
    same(
        reference.res_dual,
        norms[7] * ti * n(3) / (n::<T>(7) + nx * ti + nz * ti).max(T::one()),
    );
    same(
        reference.gap_abs,
        (reference.cost_primal - reference.cost_dual).abs(),
    );
    same(reference.ktratio, ti);
    let mut residuals = DefaultResiduals::<T>::new(0, 0);
    residuals.products = single.products;
    let settings = DefaultSettings::<T>::default();
    reference.check_termination(&residuals, &settings, 0);
    assert_eq!(reference.status, SolverStatus::Unsolved);
    for summary in [&split, &disjoint] {
        let mut info = DefaultInfo::<T>::new();
        info.update_from_summary(summary, n(2), n(1), n(3), n(5), n(7));
        for (a, b) in [
            info.cost_primal,
            info.cost_dual,
            info.res_primal,
            info.res_dual,
            info.res_primal_inf,
            info.res_dual_inf,
            info.gap_abs,
            info.gap_rel,
            info.ktratio,
        ]
        .into_iter()
        .zip([
            reference.cost_primal,
            reference.cost_dual,
            reference.res_primal,
            reference.res_dual,
            reference.res_primal_inf,
            reference.res_dual_inf,
            reference.gap_abs,
            reference.gap_rel,
            reference.ktratio,
        ]) {
            close(a, b);
        }
        info.check_termination(&residuals, &settings, 0);
        assert_eq!(info.status, reference.status);
    }

    // Exercise the real convergence gate at its normal configured tolerances.
    let mut solved_owner = make(&[0, 1, 2, 3, 4], &[0, 1, 2], Some(T::zero()));
    solved_owner.products = ResidualProducts::zero();
    for values in &mut solved_owner.values[3..] {
        values.fill(T::zero());
    }
    let mut gated_settings = DefaultSettings::<T>::default();
    gated_settings.tol_feas_componentwise = Some(gated_settings.tol_feas);
    for componentwise in [T::zero(), T::nan()] {
        solved_owner.componentwise = Some(componentwise);
        let summary =
            ResidualSummary::from_owners([solved_owner.view(), empty.view()], None).unwrap();
        let mut info = DefaultInfo::<T>::new();
        info.update_from_summary(&summary, T::one(), T::zero(), T::one(), T::one(), T::one());
        info.check_termination(&residuals, &gated_settings, 0);
        assert_eq!(
            info.status,
            if componentwise.is_nan() {
                SolverStatus::Unsolved
            } else {
                SolverStatus::Solved
            }
        );
    }

    let mismatch = make(&[], &[], None);
    assert!(ResidualSummary::from_owners([all.view(), mismatch.view()], None).is_err());
    assert!(ResidualSummary::<T>::from_owners(std::iter::empty(), None).is_err());
    let poisoned = make(&[], &[], Some(T::nan()));
    assert!(
        ResidualSummary::from_owners([all.view(), poisoned.view()], None)
            .unwrap()
            .dual_componentwise
            .unwrap()
            .is_nan()
    );
    let mut norm_nan = make(&[2], &[], Some(T::zero()));
    norm_nan.values[7][0] = T::nan();
    let bad = ResidualSummary::from_owners([norm_nan.view(), empty.view()], None).unwrap();
    assert!(bad.norms()[7].is_nan());
    let mut info = DefaultInfo::<T>::new();
    info.update_from_summary(&bad, n(1), T::zero(), n(1), n(1), n(1));
    assert!(info.res_dual.is_nan());
    info.check_termination(&residuals, &settings, 0);
    assert_ne!(info.status, SolverStatus::Solved);
}

#[test]
fn owned_statistics_f64() {
    ownership::<f64>();
}
#[test]
fn owned_statistics_mpfr256() {
    ownership::<MpFloat<4>>();
}
#[test]
fn owned_statistics_mpfr512() {
    ownership::<MpFloat<8>>();
}
#[test]
fn owned_statistics_mpfr768() {
    ownership::<MpFloat<12>>();
}
