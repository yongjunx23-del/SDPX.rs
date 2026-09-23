//! Persistent block storage plugged into the existing generic HSD loop.
use super::*;
use crate::solver::chordal::ChordalInfo;
use crate::solver::distributed::collective::{Collective, SerialCollective};
use crate::solver::SupportedConeT;
use crate::solver::{
    cones::{CompositeCone, Cone},
    core::{
        traits::{ConeCollection, Settings, Variables},
        Solver,
    },
};
use std::sync::Arc;

pub(crate) type CollectiveHandle<T> = Arc<dyn Collective<T>>;

pub(crate) struct OwnedData<T: FloatT> {
    pub layout: Arc<OwnerLayout>,
    pub global_layout: Arc<OwnerLayout>,
    /// Canonical global owner id for each local numeric block.  The serial
    /// path stores one entry per owner; a rank-local path stores only this
    /// rank's entries and retains the global layout as metadata.
    pub owner_ids: Vec<usize>,
    pub all_owner_ids: Vec<usize>,
    pub collective: CollectiveHandle<T>,
    pub blocks: Vec<DefaultProblemData<T>>,
    pub normq: T,
    pub normb: T,
    pub c: T,
    pub border_b: Vec<T>,
    pub border_e: Vec<T>,
    pub border_einv: Vec<T>,
    pub componentwise_enabled: bool,
    pub internal_cones: Vec<SupportedConeT<T>>,
    pub presolver: Option<Presolver<T>>,
    pub chordal_info: Option<ChordalInfo<T>>,
    pub cost: CostRuntimeConfig,
}
pub(crate) struct OwnedVariables<T: FloatT> {
    pub blocks: Vec<DefaultVariables<T>>,
    pub layout: Arc<OwnerLayout>,
    pub owner_ids: Vec<usize>,
    pub all_owner_ids: Vec<usize>,
    pub collective: CollectiveHandle<T>,
    /// Homogeneous scalars are shared explicitly so empty ranks never need a
    /// block zero as a scalar source.
    pub tau: T,
    pub kappa: T,
    pub border_z: Vec<T>,
    pub border_s: Vec<T>,
}
pub(crate) struct OwnedResiduals<T: FloatT> {
    pub blocks: Vec<DefaultResiduals<T>>,
    pub collective: CollectiveHandle<T>,
    pub summary: Option<ResidualSummary<T>>,
    pub border_inf: Vec<T>,
    pub border_residual: Vec<T>,
    pub scalar: DefaultResiduals<T>,
}
pub(crate) struct OwnedCones<T: FloatT> {
    pub blocks: Vec<CompositeCone<T>>,
    pub collective: CollectiveHandle<T>,
    pub step_order: Vec<(usize, usize)>,
    pub degree: usize,
    pub bounds: Vec<(T, T)>,
    pub pool: Option<Arc<rayon::ThreadPool>>,
}
#[derive(Clone)]
pub(crate) struct OwnedInfo<T: FloatT>(pub DefaultInfo<T>);
pub(crate) struct OwnedSolution<T: FloatT>(pub DefaultSolution<T>);

#[path = "hsd_data.rs"]
mod data;
#[path = "hsd_kkt.rs"]
mod kkt;
#[path = "hsd_steps.rs"]
mod steps;
pub(crate) use kkt::OwnedKktSystem;

pub(crate) type OwnedSolver<T> = Solver<
    T,
    OwnedData<T>,
    OwnedVariables<T>,
    OwnedResiduals<T>,
    OwnedKktSystem<T>,
    OwnedCones<T>,
    OwnedInfo<T>,
    OwnedSolution<T>,
    DefaultSettings<T>,
>;

impl<T: FloatT> OwnedSolver<T> {
    /// Adapter retained for internal comparison fixtures. Native callers prepare
    /// the input directly and never allocate the global KKT/iteration state.
    #[cfg(test)]
    pub(crate) fn from_default(prepared: DefaultSolver<T>, count: usize) -> Result<Self, String> {
        let DefaultSolver {
            data,
            cones,
            settings,
            timers,
            solution,
            ..
        } = prepared;
        Self::from_prepared(
            PreparedProblem {
                data,
                cones,
                settings,
                solution,
                timers: timers.unwrap(),
                cost_input_fingerprint: None,
            },
            count,
        )
    }

    pub(crate) fn from_prepared(
        prepared: PreparedProblem<T>,
        count: usize,
    ) -> Result<Self, String> {
        Self::from_prepared_tasks(
            prepared,
            Some(count),
            CostHistoryOptions::default(),
            None,
            None,
        )
    }

    pub(crate) fn from_prepared_auto(prepared: PreparedProblem<T>) -> Result<Self, String> {
        Self::from_prepared_tasks(prepared, None, CostHistoryOptions::default(), None, None)
    }

    pub(crate) fn from_prepared_with_cost_history(
        prepared: PreparedProblem<T>,
        count: usize,
        options: CostHistoryOptions,
    ) -> Result<Self, String> {
        Self::from_prepared_tasks(prepared, Some(count), options, None, None)
    }

    pub(crate) fn from_prepared_auto_with_cost_history(
        prepared: PreparedProblem<T>,
        options: CostHistoryOptions,
    ) -> Result<Self, String> {
        Self::from_prepared_tasks(prepared, None, options, None, None)
    }

    /// Build one rank-local owner view over the shared HSD loop.  Prepared
    /// data is consumed once to derive the deterministic global layout; all
    /// non-local numeric blocks are then dropped before the solver is
    /// returned.  The transport is control-thread-only and supplies the
    /// reductions needed by the shared border KKT.
    pub(crate) fn from_prepared_rank_local(
        prepared: PreparedProblem<T>,
        collective: CollectiveHandle<T>,
    ) -> Result<Self, String> {
        Self::from_prepared_rank_local_with_cost_history(
            prepared,
            collective,
            CostHistoryOptions::default(),
        )
    }

    pub(crate) fn from_prepared_rank_local_with_cost_history(
        prepared: PreparedProblem<T>,
        collective: CollectiveHandle<T>,
        options: CostHistoryOptions,
    ) -> Result<Self, String> {
        if !prepared.cones.all_symmetric() {
            return Err(
                "rank-local transport currently supports symmetric cones only (LP/SOCP/PSD)".into(),
            );
        }
        let size = collective.size();
        let rank = collective.rank();
        if size == 0 || rank >= size {
            return Err("invalid rank-local collective dimensions".into());
        }
        Self::from_prepared_tasks(prepared, Some(size), options, Some(collective), Some(rank))
    }

    fn from_prepared_tasks(
        prepared: PreparedProblem<T>,
        count: Option<usize>,
        options: CostHistoryOptions,
        collective: Option<CollectiveHandle<T>>,
        local_rank: Option<usize>,
    ) -> Result<Self, String> {
        if local_rank.is_none() && crate::mpi::World::get().is_some() {
            return Err("partitioned storage does not yet support MPI transport".into());
        }
        if prepared.settings.kkt_form == "augmented" {
            return Err("partitioned storage requires kkt_form auto or condensed".into());
        }
        let PreparedProblem {
            data,
            cones: global_cones,
            settings,
            mut solution,
            mut timers,
            cost_input_fingerprint,
        } = prepared;
        timers.start_as_current("setup");
        // Ruiz is complete. Release the global cone workspaces and their pool
        // before allocating persistent local state and the one shared pool.
        drop(global_cones);
        let budget = if settings.max_threads == 0 {
            std::thread::available_parallelism().map_or(1, usize::from)
        } else {
            settings.max_threads as usize
        };
        let provider = super::costs::provider_tag();
        if let Some(history) = options.history.as_ref() {
            history.validate_runtime(
                &settings.direct_solve_method,
                &settings.kkt_form,
                &provider,
            )?;
        }
        let cost = CostRuntimeConfig {
            input_fingerprint: cost_input_fingerprint,
            thread_budget: budget,
            record: options.record,
            direct_solve_method: settings.direct_solve_method.clone(),
            kkt_form: settings.kkt_form.clone(),
            provider,
        };
        // Rank-local setup only materializes the owner selected by this rank.
        // The layout and shared metadata remain global, while the temporary
        // prepared input is consumed exactly once by the filtered splitter.
        let state = match (count, local_rank) {
            (Some(count), Some(rank)) => {
                OwnedState::new_rank_local_with_history(data, count, rank, cost.clone(), options)?
            }
            (Some(count), None) => {
                OwnedState::new_with_history(data, count, cost.clone(), options)?
            }
            (None, None) => OwnedState::new_auto_with_history(data, cost, options)?,
            (None, Some(_)) => {
                return Err("rank-local setup requires an explicit global owner count".into())
            }
        };
        let count = state.layout.owners.len();
        let all_owner_ids: Vec<usize> = (0..count).collect();
        let full_layout = state.layout.clone();
        let border_b = state.border_b;
        let border_e = state.border_e;
        let border_einv = state.border_einv;
        let componentwise_enabled = state.componentwise_enabled;
        let full_step_order = state.step_order.clone();
        let owner_ids: Vec<usize> = local_rank
            .map(|rank| vec![rank])
            .unwrap_or_else(|| (0..count).collect());
        let numeric_owner_ids: Vec<usize> = if local_rank.is_some() {
            owner_ids
                .iter()
                .copied()
                .filter(|&owner| {
                    let ids = &full_layout.owners[owner];
                    !ids.columns.is_empty()
                        || ids
                            .cones
                            .iter()
                            .any(|cone| !full_layout.border_rows.contains(&cone.rows.start))
                })
                .collect()
        } else {
            owner_ids.clone()
        };
        let layout_value = if local_rank.is_some() {
            OwnerLayout {
                owners: owner_ids
                    .iter()
                    .map(|&owner| full_layout.owners[owner].clone())
                    .collect(),
                border_rows: full_layout.border_rows.clone(),
                n: full_layout.n,
                m: full_layout.m,
                components: full_layout.components.clone(),
                dominant_owner: full_layout
                    .dominant_owner
                    .and_then(|global| owner_ids.iter().position(|&id| id == global)),
            }
        } else {
            full_layout.clone()
        };
        let global_layout = Arc::new(full_layout.clone());
        let layout = Arc::new(layout_value);
        let collective: CollectiveHandle<T> =
            collective.unwrap_or_else(|| Arc::new(SerialCollective));
        let local_step_order: Vec<(usize, usize)> = if let Some(rank) = local_rank {
            full_step_order
                .into_iter()
                .filter_map(|(owner, cone)| {
                    (owner == rank && numeric_owner_ids.contains(&owner)).then_some((0, cone))
                })
                .collect()
        } else {
            full_step_order
        };
        // Heavy owners can expose matrix subtasks on the same pool, even
        // when the number of independent components is below the budget.
        let workers = budget.max(1);
        let pool = if workers > 1 {
            Some(Arc::new(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(workers)
                    .build()
                    .map_err(|e| format!("owner worker pool: {e}"))?,
            ))
        } else {
            None
        };
        // Keep one handle for owner-local sparse residual plans before the
        // shared pool is moved into `OwnedCones` below.
        let residual_pool = pool.clone();
        let mut data = OwnedData {
            layout: Arc::clone(&layout),
            global_layout,
            owner_ids: numeric_owner_ids.clone(),
            all_owner_ids: all_owner_ids.clone(),
            collective: Arc::clone(&collective),
            blocks: Vec::new(),
            normq: state.normq,
            normb: state.normb,
            c: state.c,
            border_b,
            border_e,
            border_einv,
            componentwise_enabled,
            internal_cones: state.internal_cones,
            presolver: state.presolver,
            chordal_info: state.chordal_info,
            cost: state.cost,
        };
        let mut variables = OwnedVariables {
            blocks: Vec::new(),
            layout: Arc::clone(&layout),
            owner_ids: numeric_owner_ids.clone(),
            all_owner_ids: all_owner_ids.clone(),
            collective: Arc::clone(&collective),
            tau: T::one(),
            kappa: T::one(),
            border_z: vec![T::zero(); layout.border_rows.len()],
            border_s: vec![T::zero(); layout.border_rows.len()],
        };
        let mut residuals = OwnedResiduals {
            blocks: Vec::new(),
            collective: Arc::clone(&collective),
            summary: None,
            border_inf: vec![T::zero(); layout.border_rows.len()],
            border_residual: vec![T::zero(); layout.border_rows.len()],
            scalar: DefaultResiduals::new(0, 0),
        };
        let mut cones = OwnedCones {
            blocks: Vec::new(),
            collective: Arc::clone(&collective),
            step_order: local_step_order,
            degree: 0,
            bounds: vec![(T::zero(), T::zero()); numeric_owner_ids.len()],
            pool,
        };
        for (global_owner, owner) in state.owner_ids.into_iter().zip(state.owners) {
            if local_rank == Some(global_owner) {
                variables.tau = owner.variables.τ;
                variables.kappa = owner.variables.κ;
                for (j, &row) in layout.border_rows.iter().enumerate() {
                    if let Ok(local) = full_layout.owners[global_owner].rows.binary_search(&row) {
                        variables.border_z[j] = owner.variables.z[local];
                        variables.border_s[j] = owner.variables.s[local];
                    }
                }
            }
            if !numeric_owner_ids.contains(&global_owner) {
                continue;
            }
            data.blocks.push(owner.data);
            variables.blocks.push(owner.variables);
            residuals.blocks.push(owner.residuals);
            cones.blocks.push(owner.cones);
        }
        // The owned update path already delegates the numerical residual
        // formulas to `DefaultResiduals`, but it intentionally skips the
        // ordinary solver's pool preparation hook.  Prepare each local block
        // once here with the same shared owner pool so large LP/SOCP blocks
        // retain the production sparse products without MPI sharding.
        for (residuals, data) in residuals.blocks.iter_mut().zip(&data.blocks) {
            residuals.prepare_sparse_local(data, residual_pool.clone());
        }
        if let Some(first) = variables.blocks.first() {
            variables.tau = first.τ;
            variables.kappa = first.κ;
            for (j, &row) in layout.border_rows.iter().enumerate() {
                if let Ok(local) = layout.owners[0].rows.binary_search(&row) {
                    variables.border_z[j] = first.z[local];
                    variables.border_s[j] = first.s[local];
                }
            }
        }
        // Reverse presolve/chordal reconstruction is a root-only output
        // phase.  Empty and non-root ranks retain only the local numeric
        // state plus the shared structural metadata needed by the HSD loop.
        if collective.rank() != 0 {
            data.presolver = None;
            data.chordal_info = None;
        }
        cones.degree = cones
            .step_order
            .iter()
            .map(|&(o, c)| cones.blocks[o].iter().nth(c).unwrap().degree())
            .sum();
        let global_degree = collective
            .reduce_sum(472, &[T::from_usize(cones.degree).unwrap_or(T::zero())])
            .map_err(|_| "rank-local cone-degree reduction failed".to_string())?
            .into_iter()
            .next()
            .and_then(|value| value.to_usize())
            .ok_or_else(|| "rank-local cone-degree reduction returned invalid data".to_string())?;
        cones.degree = global_degree;
        let kktsystem =
            OwnedKktSystem::new_with_collective(&data, &cones, &settings, Arc::clone(&collective));
        let mut info = DefaultInfo::new();
        use crate::solver::kkt::HasLinearSolverInfo;
        info.linsolver = kktsystem.linear_solver_info();
        if collective.rank() != 0 {
            solution.x = Vec::new();
            solution.s = Vec::new();
            solution.z = Vec::new();
        }
        let mut output = Self {
            step_lhs: variables.new_like(),
            step_rhs: variables.new_like(),
            prev_vars: variables.new_like(),
            data,
            variables,
            residuals,
            cones,
            kktsystem,
            info: OwnedInfo(info),
            solution: OwnedSolution(solution),
            settings,
            timers: None,
            callbacks: crate::solver::core::callbacks::SolverCallbacks::default(),
            phantom: std::marker::PhantomData,
        };
        timers.stop_current();
        output.timers.replace(timers);
        Ok(output)
    }

    #[cfg(feature = "serde")]
    pub(crate) fn cost_history(&self) -> Result<Option<CostHistory>, String> {
        let distributed = self.data.collective.size() > 1;
        let samples = self.kktsystem.cost_samples();
        if distributed {
            let all_recorded = self
                .data
                .collective
                .all_true(602, samples.is_some())
                .map_err(|_| "rank-local cost-history availability check failed")?;
            if !all_recorded {
                return Ok(None);
            }
        }
        let Some(samples) = samples else {
            return Ok(None);
        };
        let owner_count = self.data.global_layout.owners.len();
        let mut component_ids = vec![Vec::new(); owner_count];
        let components: Vec<_> = self
            .data
            .global_layout
            .components
            .iter()
            .map(|component| {
                component_ids[component.owner].push(component.identity);
                (
                    component.identity,
                    component.structural_weight,
                    component.owner,
                )
            })
            .collect();
        let owner_samples = if distributed {
            // Every rank contributes one fixed-width timing pair for its
            // global owner.  This gathers only measurements, never the local
            // matrices or KKT state, and leaves non-root ranks without an
            // assembled history.
            let size = self.data.collective.size();
            let owner_count_ok = self
                .data
                .collective
                .all_true(603, owner_count == size)
                .map_err(|_| "rank-local cost-history owner count check failed")?;
            if !owner_count_ok {
                return Err("rank-local cost-history owner count mismatch".into());
            }
            let rank = self.data.collective.rank();
            let mut timing = (Some(T::zero()), Some(T::zero()));
            for (&owner, &(local_assemble_ns, factor_response_ns)) in
                self.data.owner_ids.iter().zip(&samples)
            {
                if owner == rank {
                    timing = (
                        T::from_f64(local_assemble_ns),
                        T::from_f64(factor_response_ns),
                    );
                    break;
                }
            }
            let timing_ok = self
                .data
                .collective
                .all_true(604, timing.0.is_some() && timing.1.is_some())
                .map_err(|_| "rank-local cost-history timing check failed")?;
            if !timing_ok {
                return Err("cost-history timing cannot be represented".into());
            }
            let payload = vec![
                timing.0.expect("timing representation was agreed"),
                timing.1.expect("timing representation was agreed"),
            ];
            let ranges: Vec<_> = (0..size).map(|owner| (owner * 2, 2)).collect();
            let gathered = self
                .data
                .collective
                .gather_root(601, &payload, &ranges, 0)
                .map_err(|_| "rank-local cost-history gather failed")?;
            if rank != 0 {
                return Ok(None);
            }
            let values = gathered.ok_or("rank-local cost-history root gather was empty")?;
            let mut owner_samples = Vec::new();
            for owner in 0..owner_count {
                if component_ids[owner].is_empty() {
                    continue;
                }
                let begin = owner * 2;
                let local_assemble_ns = values
                    .get(begin)
                    .and_then(|value| value.to_f64())
                    .ok_or("invalid gathered cost-history timing")?;
                let factor_response_ns = values
                    .get(begin + 1)
                    .and_then(|value| value.to_f64())
                    .ok_or("invalid gathered cost-history timing")?;
                owner_samples.push(CostOwnerSample {
                    owner,
                    local_assemble_ns,
                    factor_response_ns,
                    components: component_ids[owner].clone(),
                });
            }
            owner_samples
        } else {
            samples
                .into_iter()
                .enumerate()
                .filter_map(|(owner, (local_assemble_ns, factor_response_ns))| {
                    if component_ids[owner].is_empty() {
                        return None;
                    }
                    Some(CostOwnerSample {
                        owner,
                        local_assemble_ns,
                        factor_response_ns,
                        components: component_ids[owner].clone(),
                    })
                })
                .collect()
        };
        CostHistory::from_measurements(
            self.data.cost.input_fingerprint,
            T::precision_bits(),
            self.data.cost.thread_budget,
            self.data.layout.n,
            self.data.layout.m,
            &components,
            owner_samples,
            self.data.cost.direct_solve_method.clone(),
            self.data.cost.kkt_form.clone(),
            self.data.cost.provider.clone(),
        )
        .map(Some)
    }
}

#[cfg(test)]
#[path = "tests/hsd.rs"]
mod tests;

#[cfg(test)]
#[path = "tests/pool.rs"]
mod pool_tests;
