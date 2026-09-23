//! One-time split of prepared data into persistent owner-local storage.
//! Iteration and recovery are implemented by the shared HSD consumers.
use super::*;
#[cfg(feature = "sdp")]
use std::sync::Arc;

pub(crate) struct OwnerState<T: FloatT> {
    pub data: DefaultProblemData<T>,
    pub variables: DefaultVariables<T>,
    pub residuals: DefaultResiduals<T>,
    pub cones: CompositeCone<T>,
}

pub(crate) struct OwnedState<T: FloatT> {
    pub layout: OwnerLayout,
    /// Global owner id for each entry in `owners`.  The serial constructor
    /// keeps every owner; rank-local construction keeps only the requested
    /// owner while retaining the full structural layout above.
    pub owner_ids: Vec<usize>,
    pub owners: Vec<OwnerState<T>>,
    // Global cone order, with shared zero cones counted on owner zero only.
    pub(crate) step_order: Vec<(usize, usize)>,
    pub(crate) normq: T,
    pub(crate) normb: T,
    pub(crate) c: T,
    /// Shared equality metadata is retained even when this rank has no
    /// numeric owner block (for example an equality-only rank).
    pub border_b: Vec<T>,
    pub border_e: Vec<T>,
    pub border_einv: Vec<T>,
    pub componentwise_enabled: bool,
    // Keep the existing recovery metadata, not the full original matrices.
    pub internal_cones: Vec<SupportedConeT<T>>,
    pub presolver: Option<Presolver<T>>,
    pub cost: CostRuntimeConfig,
    #[cfg(feature = "sdp")]
    pub chordal_info: Option<ChordalInfo<T>>,
}

fn selected<T: Copy>(values: &[T], indices: &[usize]) -> Vec<T> {
    indices.iter().map(|&i| values[i]).collect()
}

// Columns have a unique owner. Every stored row in one of these columns must
// belong to that owner or to the common equality border. Stored zeros count.
fn local_matrix<T: FloatT>(
    matrix: &CscMatrix<T>,
    rows: &[usize],
    columns: &[usize],
) -> Result<CscMatrix<T>, String> {
    let mut ptr = Vec::with_capacity(columns.len() + 1);
    let mut ids = Vec::new();
    let mut values = Vec::new();
    ptr.push(0);
    for &column in columns {
        for k in matrix.colptr[column]..matrix.colptr[column + 1] {
            let local = rows
                .binary_search(&matrix.rowval[k])
                .map_err(|_| "owner layout omitted a stored matrix edge")?;
            ids.push(local);
            values.push(matrix.nzval[k]);
        }
        ptr.push(ids.len());
    }
    Ok(CscMatrix::new(rows.len(), columns.len(), ptr, ids, values))
}

impl<T: FloatT> OwnedState<T> {
    /// Consume a validated, already preprocessed/equilibrated problem. Splitting
    /// does not run presolve, clip b, materialize sampled factors, or scale again.
    /// Initial construction still sees the whole input; streaming input is a
    /// separate outstanding part of the distributed backend.
    ///
    /// Test-only convenience: the production entry points take the explicit
    /// history variant (`new_with_history` / `new_rank_local_with_history`).
    #[cfg(test)]
    pub fn new(data: DefaultProblemData<T>, count: usize) -> Result<Self, String> {
        let layout = OwnerLayout::new(&data, count)?;
        Self::from_layout(data, layout, CostRuntimeConfig::disabled(), None)
    }

    #[cfg(feature = "sdp")]
    pub(crate) fn new_with_history(
        data: DefaultProblemData<T>,
        count: usize,
        cost: CostRuntimeConfig,
        options: CostHistoryOptions,
    ) -> Result<Self, String> {
        let layout = if let Some(history) = options.history.as_ref() {
            OwnerLayout::new_with_history(
                &data,
                count,
                cost.thread_budget,
                cost.input_fingerprint,
                history,
            )?
        } else {
            OwnerLayout::new(&data, count)?
        };
        let mut cost = cost;
        cost.record = options.record;
        Self::from_layout(data, layout, cost, None)
    }

    #[cfg(feature = "sdp")]
    pub(crate) fn new_auto_with_history(
        data: DefaultProblemData<T>,
        cost: CostRuntimeConfig,
        options: CostHistoryOptions,
    ) -> Result<Self, String> {
        let layout = if let Some(history) = options.history.as_ref() {
            OwnerLayout::new_auto_with_history(
                &data,
                cost.thread_budget,
                cost.input_fingerprint,
                history,
            )?
        } else {
            OwnerLayout::new_auto(&data, cost.thread_budget)?
        };
        let mut cost = cost;
        cost.record = options.record;
        Self::from_layout(data, layout, cost, None)
    }

    /// Build only one global owner's numeric state for a rank-local solver.
    /// The full structural layout and shared equality metadata remain
    /// available to the caller, but unowned matrix/cone workspaces are never
    /// allocated in `owners`.
    #[cfg(feature = "sdp")]
    pub(crate) fn new_rank_local_with_history(
        data: DefaultProblemData<T>,
        count: usize,
        owner: usize,
        cost: CostRuntimeConfig,
        options: CostHistoryOptions,
    ) -> Result<Self, String> {
        if owner >= count {
            return Err("rank-local owner is outside the global owner count".into());
        }
        let layout = if let Some(history) = options.history.as_ref() {
            OwnerLayout::new_with_history(
                &data,
                count,
                cost.thread_budget,
                cost.input_fingerprint,
                history,
            )?
        } else {
            OwnerLayout::new(&data, count)?
        };
        let mut cost = cost;
        cost.record = options.record;
        Self::from_layout(data, layout, cost, Some(std::slice::from_ref(&owner)))
    }

    fn from_layout(
        mut data: DefaultProblemData<T>,
        layout: OwnerLayout,
        cost: CostRuntimeConfig,
        owner_filter: Option<&[usize]>,
    ) -> Result<Self, String> {
        let count = layout.owners.len();
        let normq = data.get_normq();
        let normb = data.get_normb();
        let c = data.equilibration.c;
        let border_b = layout.border_rows.iter().map(|&row| data.b[row]).collect();
        let border_e = layout
            .border_rows
            .iter()
            .map(|&row| data.equilibration.e[row])
            .collect();
        let border_einv = layout
            .border_rows
            .iter()
            .map(|&row| data.equilibration.einv[row])
            .collect();
        let componentwise_enabled = data.componentwise_enabled;
        let mut owners = Vec::with_capacity(owner_filter.map_or(count, |selected| selected.len()));
        let mut owner_ids =
            Vec::with_capacity(owner_filter.map_or(count, |selected| selected.len()));
        for (rank, ids) in layout.owners.iter().enumerate() {
            if owner_filter.is_some_and(|selected| !selected.contains(&rank)) {
                continue;
            }
            let p = local_matrix(&data.P, &ids.columns, &ids.columns)?;
            let a = local_matrix(&data.A, &ids.rows, &ids.columns)?;
            let q = selected(&data.q, &ids.columns);
            let mut b = selected(&data.b, &ids.rows);
            if rank != 0 {
                for &row in &layout.border_rows {
                    b[ids.rows.binary_search(&row).unwrap()] = T::zero();
                }
            }
            let cones: Vec<_> = ids
                .cones
                .iter()
                .map(|piece| match &data.cones[piece.original] {
                    SupportedConeT::ZeroConeT(_) => SupportedConeT::ZeroConeT(piece.rows.len()),
                    SupportedConeT::NonnegativeConeT(_) => {
                        SupportedConeT::NonnegativeConeT(piece.rows.len())
                    }
                    cone => cone.clone(),
                })
                .collect();
            let eq = &data.equilibration;
            let equilibration = DefaultEquilibrationData {
                d: selected(&eq.d, &ids.columns),
                dinv: selected(&eq.dinv, &ids.columns),
                e: selected(&eq.e, &ids.rows),
                einv: selected(&eq.einv, &ids.rows),
                c,
            };
            #[cfg(feature = "sdp")]
            let sampled = data
                .sampled
                .as_ref()
                .map(|operator| {
                    // A factor belongs to one owner in its entirety. Sorted local
                    // columns preserve every canonical interval contiguously.
                    let mut blocks = Vec::new();
                    for block in operator.blocks() {
                        let Ok(row) = ids.rows.binary_search(&block.row_start) else {
                            continue;
                        };
                        let mut local = block.clone();
                        local.row_start = row;
                        local.column_start = if block.column_count() == 0 {
                            ids.columns.partition_point(|&i| i < block.column_start)
                        } else {
                            ids.columns
                                .binary_search(&block.column_start)
                                .map_err(|_| "sampled factor column has no local owner")?
                        };
                        for k in 0..block.column_count() {
                            if ids.columns.get(local.column_start + k)
                                != Some(&(block.column_start + k))
                            {
                                return Err(
                                    "sampled canonical columns are not contiguous locally".into()
                                );
                            }
                        }
                        blocks.push(local);
                    }
                    // Keep the original linear CSC order and authoritative factors;
                    // the partition never rebuilds the rounded coefficients.
                    let linear = local_matrix(operator.linear(), &ids.rows, &ids.columns)?;
                    SampledOperator::new_local(linear, blocks).map(Arc::new)
                })
                .transpose()?;
            let local = DefaultProblemData {
                P: p,
                q,
                A: a,
                b,
                cones,
                n: ids.columns.len(),
                m: ids.rows.len(),
                componentwise_enabled: data.componentwise_enabled,
                equilibration,
                normq: None,
                normb: None,
                presolver: None,
                dropped_zeros: 0,
                #[cfg(feature = "sdp")]
                sampled,
                #[cfg(feature = "sdp")]
                sampled_input: data.sampled_input,
                #[cfg(feature = "sdp")]
                chordal_info: None,
            };
            let variables = DefaultVariables::new(local.n, local.m);
            let residuals = DefaultResiduals::new(local.n, local.m);
            let cones = CompositeCone::new_local(&local.cones);
            owners.push(OwnerState {
                data: local,
                variables,
                residuals,
                cones,
            });
            owner_ids.push(rank);
        }
        let border_rows = &layout.border_rows;
        let mut step_order: Vec<_> = layout
            .owners
            .iter()
            .enumerate()
            .flat_map(|(rank, ids)| {
                ids.cones.iter().enumerate().filter_map(move |(ci, piece)| {
                    (rank == 0 || !border_rows.contains(&piece.rows.start)).then_some((rank, ci))
                })
            })
            .collect();
        step_order.sort_by_key(|&(rank, ci)| layout.owners[rank].cones[ci].rows.start);
        Ok(Self {
            layout,
            owner_ids,
            owners,
            step_order,
            normq,
            normb,
            c,
            border_b,
            border_e,
            border_einv,
            componentwise_enabled,
            internal_cones: std::mem::take(&mut data.cones),
            presolver: data.presolver.take(),
            cost,
            #[cfg(feature = "sdp")]
            chordal_info: data.chordal_info.take(),
        })
    }
}

#[cfg(feature = "sdp")]
#[path = "owned_kkt.rs"]
pub(crate) mod owned_kkt;

#[cfg(all(test, feature = "sdp"))]
#[path = "owned_state_tests.rs"]
mod tests;

#[cfg(all(test, feature = "sdp"))]
#[path = "owned_recovery_tests.rs"]
mod recovery_tests;

#[cfg(all(test, feature = "sdp", feature = "serde"))]
#[path = "owned_ising_tests.rs"]
mod ising_tests;

#[cfg(test)]
mod construction_tests {
    use super::*;
    #[test]
    fn split_construction_has_unique_columns_and_counted_rows() {
        let settings = DefaultSettings::<f64> {
            presolve_enable: false,
            equilibrate_enable: false,
            ..DefaultSettings::default()
        };
        let data = DefaultProblemData::new(
            &CscMatrix::identity(2),
            &[1., 2.],
            &CscMatrix::from(&[[1., 1.], [1., 0.], [0., 1.]]),
            &[0., 1., 1.],
            &[
                SupportedConeT::ZeroConeT(1),
                SupportedConeT::NonnegativeConeT(2),
            ],
            &settings,
        );
        let state = OwnedState::new(data, 4).unwrap();
        assert_eq!(state.owners.len(), 4);
        assert_eq!(state.owners.iter().map(|o| o.data.n).sum::<usize>(), 2);
        assert_eq!(
            state
                .layout
                .owners
                .iter()
                .map(|o| o.counted_rows.len())
                .sum::<usize>(),
            3
        );
        assert!(state.owners.iter().any(|o| o.data.n == 0));
        for (owner, ids) in state.owners.iter().zip(&state.layout.owners) {
            assert_eq!(owner.variables.x.len(), ids.columns.len());
            assert_eq!(owner.residuals.rz.len(), ids.rows.len());
            assert_eq!(owner.cones.numel(), ids.rows.len());
            assert!(ids.rows.contains(&0));
        }
    }

    #[test]
    fn rank_local_split_materializes_only_selected_owner() {
        let settings = DefaultSettings::<f64> {
            presolve_enable: false,
            equilibrate_enable: false,
            ..DefaultSettings::default()
        };
        let data = DefaultProblemData::new(
            &CscMatrix::identity(2),
            &[1., 2.],
            &CscMatrix::from(&[[1., 1.], [1., 0.], [0., 1.]]),
            &[0., 1., 1.],
            &[
                SupportedConeT::ZeroConeT(1),
                SupportedConeT::NonnegativeConeT(2),
            ],
            &settings,
        );
        let state = OwnedState::new_rank_local_with_history(
            data,
            4,
            2,
            CostRuntimeConfig::disabled(),
            CostHistoryOptions::default(),
        )
        .unwrap();
        assert_eq!(state.layout.owners.len(), 4);
        assert_eq!(state.owner_ids, vec![2]);
        assert_eq!(state.owners.len(), 1);
        let owner = state.owner_ids[0];
        assert_eq!(
            state.owners[0].data.n,
            state.layout.owners[owner].columns.len()
        );
        assert_eq!(
            state.owners[0].data.m,
            state.layout.owners[owner].rows.len()
        );
        assert_eq!(state.border_b.len(), state.layout.border_rows.len());
    }
}
