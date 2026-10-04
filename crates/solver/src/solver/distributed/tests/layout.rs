use super::super::costs::CostComponent;
use super::*;
use crate::algebra::CscMatrix;
use crate::solver::default::DefaultSettings;
use sdpx_arithmetic::MpFloat;

fn data<T: FloatT>(
    a: CscMatrix<T>,
    p: CscMatrix<T>,
    cones: Vec<SupportedConeT<T>>,
) -> DefaultProblemData<T> {
    let settings = DefaultSettings {
        presolve_enable: false,
        chordal_decomposition_enable: false,
        equilibrate_enable: false,
        input_sparse_dropzeros: false,
        ..DefaultSettings::default()
    };
    DefaultProblemData::new(
        &p,
        &vec![T::zero(); a.n],
        &a,
        &vec![T::zero(); a.m],
        &cones,
        &settings,
    )
}
fn owner(layout: &OwnerLayout, column: usize) -> usize {
    layout
        .owners
        .iter()
        .position(|o| o.columns.contains(&column))
        .unwrap()
}
fn invariant(layout: &OwnerLayout) {
    let mut columns = vec![0; layout.n];
    let mut rows = vec![0; layout.m];
    for (id, o) in layout.owners.iter().enumerate() {
        assert!(o.columns.windows(2).all(|w| w[0] < w[1]));
        assert!(o.rows.windows(2).all(|w| w[0] < w[1]));
        assert!(o
            .cones
            .windows(2)
            .all(|w| w[0].rows.start <= w[1].rows.start));
        for &c in &o.columns {
            columns[c] += 1;
        }
        for &r in &layout.border_rows {
            assert!(o.rows.contains(&r));
        }
        for &local in &o.counted_rows {
            rows[o.rows[local]] += 1;
        }
        if id > 0 {
            assert!(o
                .counted_rows
                .iter()
                .all(|&r| !layout.border_rows.contains(&o.rows[r])));
        }
        assert_eq!(
            o.rows,
            o.cones
                .iter()
                .flat_map(|c| c.rows.clone())
                .collect::<Vec<_>>()
        );
    }
    assert_eq!(columns, vec![1; layout.n]);
    assert_eq!(rows, vec![1; layout.m]);
}
fn generic<T: FloatT>() {
    let o = T::one();
    let z = T::zero();
    // Column 2 is equality-only. Columns 0/3 merge through a stored-zero P
    // edge; 1/4/6 merge through the whole SOC; the EXP has no columns.
    let a = CscMatrix::new(
        12,
        8,
        vec![0, 1, 2, 3, 4, 5, 6, 7, 8],
        vec![2, 6, 0, 3, 7, 4, 8, 5],
        vec![o, o, o, o, z, o, o, o],
    );
    let p = CscMatrix::new(8, 8, vec![0, 0, 0, 0, 1, 1, 1, 1, 1], vec![0], vec![z]);
    let d = data(
        a,
        p,
        vec![
            SupportedConeT::ZeroConeT(2),
            SupportedConeT::NonnegativeConeT(4),
            SupportedConeT::SecondOrderConeT(3),
            SupportedConeT::ExponentialConeT(),
        ],
    );
    assert_eq!(d.P.nzval.len(), 1);
    assert!(OwnerLayout::new(&d, 0).is_err());
    let layout = OwnerLayout::new(&d, 20).unwrap();
    invariant(&layout);
    assert_eq!(layout, OwnerLayout::new(&d, 20).unwrap());
    assert_eq!(layout.border_rows, vec![0, 1]);
    assert_eq!(owner(&layout, 0), owner(&layout, 3));
    assert_eq!(owner(&layout, 1), owner(&layout, 4));
    assert_eq!(owner(&layout, 1), owner(&layout, 6));
    assert_ne!(owner(&layout, 5), owner(&layout, 7));
    assert_ne!(owner(&layout, 2), owner(&layout, 0));
    let exp = layout
        .owners
        .iter()
        .find(|o| o.cones.iter().any(|c| c.original == 3))
        .unwrap();
    assert!(exp.columns.is_empty());
    assert_eq!(
        exp.cones.iter().find(|c| c.original == 3).unwrap().rows,
        9..12
    );
    assert!(layout
        .owners
        .iter()
        .any(|o| o.columns.is_empty() && o.rows == layout.border_rows));
    let nn: Vec<_> = layout
        .owners
        .iter()
        .flat_map(|o| &o.cones)
        .filter(|c| c.original == 1)
        .collect();
    assert_eq!(nn.len(), 4);
    assert!(nn.iter().all(|c| c.rows.len() == 1));
    invariant(&OwnerLayout::new(&d, 1).unwrap());

    // Empty variable space still schedules atomic power/general-power cones.
    let d = data(
        CscMatrix::<T>::zeros((6, 0)),
        CscMatrix::zeros((0, 0)),
        vec![
            SupportedConeT::PowerConeT(o / (o + o)),
            SupportedConeT::GenPowerConeT(vec![o / (o + o); 2], 1),
        ],
    );
    let layout = OwnerLayout::new(&d, 4).unwrap();
    invariant(&layout);
    assert_eq!(
        layout.owners.iter().map(|o| o.cones.len()).sum::<usize>(),
        2
    );
    assert!(layout.owners.iter().all(|o| o.columns.is_empty()));
}
#[test]
fn owner_layout_generic_f64() {
    generic::<f64>();
}
#[test]
fn owner_layout_generic_mpfr512() {
    generic::<MpFloat<8>>();
}

#[test]
fn owner_layout_shared_sampled_ranges() {
    use crate::solver::{SampledBlock, SampledOperator};
    let mut d = data(
        CscMatrix::<f64>::zeros((7, 8)),
        CscMatrix::zeros((8, 8)),
        vec![
            SupportedConeT::PSDTriangleConeT(2),
            SupportedConeT::PSDTriangleConeT(2),
            SupportedConeT::ZeroConeT(1),
        ],
    );
    let blocks = vec![
        SampledBlock {
            row_start: 0,
            column_start: 1,
            dim: 1,
            basis_rows: 2,
            basis_cols: 3,
            basis: vec![0.; 6],
            weights: vec![0.; 3],
        },
        SampledBlock {
            row_start: 3,
            column_start: 3,
            dim: 1,
            basis_rows: 2,
            basis_cols: 3,
            basis: vec![0.; 6],
            weights: vec![0.; 3],
        },
    ];
    d.sampled = Some(std::sync::Arc::new(
        SampledOperator::new(CscMatrix::zeros((7, 8)), blocks).unwrap(),
    ));
    let layout = OwnerLayout::new(&d, 5).unwrap();
    invariant(&layout);
    let id = owner(&layout, 1);
    for col in 1..6 {
        assert_eq!(owner(&layout, col), id);
    }
    let local = &layout.owners[id];
    assert!(local.rows.starts_with(&[0, 1, 2, 3, 4, 5]));
    for range in [1..4, 3..6] {
        let positions: Vec<_> = range
            .map(|c| local.columns.binary_search(&c).unwrap())
            .collect();
        assert!(positions.windows(2).all(|w| w[1] == w[0] + 1));
    }
    assert_ne!(owner(&layout, 0), id);
}

fn automatic<T: FloatT>() {
    let d = data(
        CscMatrix::<T>::identity(11),
        CscMatrix::identity(11),
        vec![SupportedConeT::NonnegativeConeT(11)],
    );
    for (width, count) in [(0, 2), (1, 2), (2, 4), (4, 8), (8, 11), (usize::MAX, 11)] {
        let auto = OwnerLayout::new_auto(&d, width).unwrap();
        assert_eq!(auto.owners.len(), count);
        invariant(&auto);
        assert!(auto.owners.iter().all(|o| !o.columns.is_empty()));
        // Auto selection only selects a count; explicit LPT is unchanged.
        assert_eq!(auto, OwnerLayout::new(&d, count).unwrap());
    }
    let explicit = OwnerLayout::new(&d, 15).unwrap();
    assert_eq!(explicit.owners.len(), 15);
    assert!(explicit.owners.iter().any(|o| o.columns.is_empty()));
    assert_eq!(explicit.owners[0].columns, vec![0]);
    assert_eq!(explicit.owners[10].columns, vec![10]);

    // Every stored P edge counts, including exact zeros.
    let p = CscMatrix::new(
        11,
        11,
        (0..=11).collect(),
        (0usize..11).map(|i| i.saturating_sub(1)).collect(),
        vec![T::zero(); 11],
    );
    let coupled = data(
        CscMatrix::<T>::identity(11),
        p,
        vec![SupportedConeT::NonnegativeConeT(11)],
    );
    assert_eq!(OwnerLayout::new_auto(&coupled, 8).unwrap().owners.len(), 1);
    // A shared ordinary row couples all columns, even if its values are zero.
    let coupled = data(
        CscMatrix::<T>::new(1, 11, (0..=11).collect(), vec![0; 11], vec![T::zero(); 11]),
        CscMatrix::identity(11),
        vec![SupportedConeT::NonnegativeConeT(1)],
    );
    assert_eq!(OwnerLayout::new_auto(&coupled, 8).unwrap().owners.len(), 1);

    // Eleven isolated columns and four shared rows: at most two owners.
    let border = data(
        CscMatrix::<T>::zeros((4, 11)),
        CscMatrix::identity(11),
        vec![SupportedConeT::ZeroConeT(4)],
    );
    let auto = OwnerLayout::new_auto(&border, 8).unwrap();
    assert_eq!(auto.owners.len(), 2);
    assert!((auto.owners.len() - 1) * auto.border_rows.len() <= auto.m);
    invariant(&auto);
    let empty = data(
        CscMatrix::<T>::zeros((0, 0)),
        CscMatrix::zeros((0, 0)),
        vec![],
    );
    assert_eq!(
        OwnerLayout::new_auto(&empty, usize::MAX)
            .unwrap()
            .owners
            .len(),
        1
    );
    let rows = data(
        CscMatrix::<T>::zeros((3, 0)),
        CscMatrix::zeros((0, 0)),
        vec![SupportedConeT::NonnegativeConeT(3)],
    );
    let auto = OwnerLayout::new_auto(&rows, 4).unwrap();
    assert_eq!(auto.owners.len(), 3);
    assert!(auto.owners.iter().all(|o| !o.counted_rows.is_empty()));
    invariant(&auto);
}
#[test]
fn owner_layout_auto_f64() {
    automatic::<f64>();
}
#[test]
fn owner_layout_auto_mpfr512() {
    automatic::<MpFloat<8>>();
}

#[test]
fn owner_inner_admission_is_dominant_and_deterministic() {
    // One SOC component owns eight columns/rows; the final orthant coordinate
    // is a light independent component.  The structural LPT plan therefore
    // has a single owner whose work exceeds the existing 75% inner-lane gate.
    let a = CscMatrix::new(9, 9, (0..=9).collect(), (0..9).collect(), vec![1.0; 9]);
    let d = data(
        a,
        CscMatrix::identity(9),
        vec![
            SupportedConeT::SecondOrderConeT(8),
            SupportedConeT::NonnegativeConeT(1),
        ],
    );
    let layout = OwnerLayout::new(&d, 2).unwrap();
    let heavy = owner(&layout, 0);
    let light = owner(&layout, 8);
    assert_ne!(heavy, light);
    assert_eq!(layout.dominant_owner, Some(heavy));
    assert!(layout.owner_inner_admission(heavy));
    assert!(!layout.owner_inner_admission(light));
    assert_eq!(layout, OwnerLayout::new(&d, 2).unwrap());

    // Equal isolated components have no dominant owner, so the ordinary
    // outer owner tasks retain their existing scheduling policy.
    let equal = data::<f64>(
        CscMatrix::identity(8),
        CscMatrix::identity(8),
        vec![SupportedConeT::NonnegativeConeT(8)],
    );
    let equal = OwnerLayout::new(&equal, 2).unwrap();
    assert_eq!(equal.dominant_owner, None);
    assert!(!equal.owner_inner_admission(0));
    assert!(!equal.owner_inner_admission(1));
}

#[test]
fn owner_layout_auto_canonical_sampled_coupling() {
    use crate::solver::{SampledBlock, SampledOperator};
    let mut d = data(
        CscMatrix::<f64>::zeros((3, 11)),
        CscMatrix::identity(11),
        vec![SupportedConeT::PSDTriangleConeT(2)],
    );
    d.sampled = Some(std::sync::Arc::new(
        SampledOperator::new(
            CscMatrix::zeros((3, 11)),
            vec![SampledBlock {
                row_start: 0,
                column_start: 0,
                dim: 1,
                basis_rows: 2,
                basis_cols: 11,
                basis: vec![0.; 22],
                weights: vec![0.; 11],
            }],
        )
        .unwrap(),
    ));
    let auto = OwnerLayout::new_auto(&d, 8).unwrap();
    assert_eq!(auto.owners.len(), 1);
    invariant(&auto);
}

#[cfg(feature = "serde")]
#[test]
fn historical_costs_validate_identity_and_change_only_lpt_order() {
    let p = CscMatrix::identity(4);
    let q = vec![0.0; 4];
    let a = CscMatrix::identity(4);
    let b = vec![0.0; 4];
    let cones = vec![SupportedConeT::NonnegativeConeT(4)];
    let d = data(a.clone(), p.clone(), cones.clone());
    let baseline = OwnerLayout::new(&d, 2).unwrap();
    let fingerprint =
        crate::solver::distributed::costs::input_fingerprint(&p, &q, &a, &b, &cones, None);
    let components = baseline
        .components
        .iter()
        .enumerate()
        .map(|(index, component)| CostComponent {
            identity: component.identity,
            structural_weight: component.structural_weight,
            cost: if index == 0 { 100.0 } else { 1.0 },
        })
        .collect();
    let history = CostHistory {
        schema_version: crate::solver::distributed::costs::COST_HISTORY_SCHEMA_VERSION,
        input_fingerprint: fingerprint,
        precision_bits: 53,
        thread_budget: 2,
        n: 4,
        m: 4,
        components,
        owner_samples: Vec::new(),
        apportionment: "owner_elapsed_apportioned_by_structural_weight".into(),
        direct_solve_method: "auto".into(),
        kkt_form: "auto".into(),
        provider: crate::solver::distributed::costs::provider_tag(),
    };
    let weighted = OwnerLayout::new_with_history(&d, 2, 2, Some(fingerprint), &history).unwrap();
    invariant(&weighted);
    assert_eq!(weighted.components.len(), baseline.components.len());
    let heavy = weighted
        .components
        .iter()
        .find(|component| component.identity == history.components[0].identity)
        .unwrap()
        .owner;
    assert!(weighted.owner_inner_admission(heavy));
    assert_eq!(weighted.dominant_owner, Some(heavy));
    let mut mismatched = history.clone();
    mismatched.schema_version = 1;
    assert!(OwnerLayout::new_with_history(&d, 2, 2, Some(fingerprint), &mismatched).is_err());
    mismatched = history.clone();
    mismatched.thread_budget = 1;
    assert!(OwnerLayout::new_with_history(&d, 2, 2, Some(fingerprint), &mismatched).is_err());
    mismatched = history;
    mismatched.input_fingerprint[0] ^= 1;
    assert!(OwnerLayout::new_with_history(&d, 2, 2, Some(fingerprint), &mismatched).is_err());
}
