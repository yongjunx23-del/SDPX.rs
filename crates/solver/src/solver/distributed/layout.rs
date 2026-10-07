//! Structural ownership in current problem coordinates; contains no scalar data.
use super::costs::CostHistory;
use super::DefaultProblemData;
use crate::algebra::FloatT;
use crate::solver::cones::SupportedConeT;
use std::{cmp::Reverse, collections::BinaryHeap, ops::Range};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct OwnerCone {
    pub original: usize,
    pub rows: Range<usize>,
}
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct OwnerIndices {
    pub columns: Vec<usize>,
    pub rows: Vec<usize>,
    pub cones: Vec<OwnerCone>,
    /// Local row positions; repeated border rows are counted only on owner 0.
    pub counted_rows: Vec<usize>,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct LayoutComponent {
    pub identity: u64,
    pub structural_weight: u64,
    pub owner: usize,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct OwnerLayout {
    pub owners: Vec<OwnerIndices>,
    pub border_rows: Vec<usize>,
    pub n: usize,
    pub m: usize,
    pub components: Vec<LayoutComponent>,
    /// At most one owner may admit inner matrix lanes.  The winner is selected
    /// from structural LPT weights or validated historical costs during layout
    /// construction and then remains fixed for every KKT update.
    pub dominant_owner: Option<usize>,
}

struct Components {
    parent: Vec<usize>,
    size: Vec<usize>,
}
impl Components {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            size: vec![1; n],
        }
    }
    fn root(&mut self, mut i: usize) -> usize {
        while i != self.parent[i] {
            self.parent[i] = self.parent[self.parent[i]];
            i = self.parent[i];
        }
        i
    }
    fn join(&mut self, a: usize, b: usize) {
        let (mut a, mut b) = (self.root(a), self.root(b));
        if a == b {
            return;
        }
        if self.size[a] < self.size[b] || (self.size[a] == self.size[b] && a > b) {
            std::mem::swap(&mut a, &mut b);
        }
        self.parent[b] = a;
        self.size[a] += self.size[b];
    }
}

fn structural_mix(mut hash: u64, tag: u64, value: u64) -> u64 {
    hash ^= tag;
    hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    hash ^= value;
    hash.wrapping_mul(0x0000_0100_0000_01b3)
}

fn structural_finish(mut hash: u64) -> u64 {
    hash ^= hash >> 29;
    hash = hash.wrapping_mul(0x9e37_79b9_7f4a_7c15);
    hash ^= hash >> 31;
    if hash == 0 {
        1
    } else {
        hash
    }
}

/// Tell the user when a fixed owner count (the MPI world size) cannot be
/// balanced: owners beyond the component count stay idle, and an uneven
/// count per owner leaves the busiest owner up to twice the lightest one's
/// work while every collective waits for it (Λ19 spins 0–50, 26 components
/// on 16 ranks: per-rank SVD time 6 versus 12 s, journal 2026-10-08).
fn warn_uneven(roots: &[usize], assigned: &[usize], owners: usize) {
    let mut counts = vec![0usize; owners];
    for &root in roots {
        counts[assigned[root]] += 1;
    }
    let (lo, hi) = (
        counts.iter().copied().min().unwrap_or(0),
        counts.iter().copied().max().unwrap_or(0),
    );
    let first = crate::mpi::World::get().is_none_or(|world| world.rank() == 0);
    if lo == hi || !first {
        return;
    }
    let components = roots.len();
    let even: Vec<String> = (1..=components)
        .filter(|d| components % d == 0 && *d <= 2 * owners)
        .map(|d| d.to_string())
        .collect();
    eprintln!(
        "SDPX: {components} independent components over {owners} ranks gives {lo} to {hi} per rank; \
         rank counts dividing {components} ({}) balance them",
        even.join(", ")
    );
}

/// Components above this count keep the plain LPT plan (refinement is
/// quadratic in the busiest owner's members).
const REFINE_COMPONENTS: usize = 4096;

/// Improve an LPT plan by moving or swapping one component between the
/// busiest owner and another while that strictly lowers the pair's maximum.
/// Each step strictly lowers the sum of squared loads, so it terminates;
/// the scan order is fixed, so the plan is deterministic.
fn refine_assignment(roots: &[usize], work: &[u128], owners: usize, assigned: &mut [usize]) {
    if owners < 2 {
        return;
    }
    let mut members = vec![Vec::new(); owners];
    let mut loads = vec![0u128; owners];
    for &root in roots {
        members[assigned[root]].push(root);
        loads[assigned[root]] += work[root];
    }
    for _ in 0..roots.len().saturating_mul(owners) {
        let hi = (0..owners)
            .max_by(|&a, &b| loads[a].cmp(&loads[b]).then_with(|| b.cmp(&a)))
            .unwrap();
        // (new pair maximum, lo, index in hi, index in lo or none)
        let mut best: Option<(u128, usize, usize, Option<usize>)> = None;
        for lo in (0..owners).filter(|&lo| lo != hi) {
            for (ia, &a) in members[hi].iter().enumerate() {
                let candidates = std::iter::once(None).chain((0..members[lo].len()).map(Some));
                for ib in candidates {
                    let back = ib.map_or(0, |ib| work[members[lo][ib]]);
                    if work[a] <= back {
                        continue;
                    }
                    let delta = work[a] - back;
                    let peak = (loads[hi] - delta).max(loads[lo] + delta);
                    if peak < loads[hi] && best.is_none_or(|(value, ..)| peak < value) {
                        best = Some((peak, lo, ia, ib));
                    }
                }
            }
        }
        let Some((_, lo, ia, ib)) = best else {
            break;
        };
        let a = members[hi].swap_remove(ia);
        let back = ib.map(|ib| members[lo].swap_remove(ib));
        let delta = work[a] - back.map_or(0, |b| work[b]);
        loads[hi] -= delta;
        loads[lo] += delta;
        assigned[a] = lo;
        members[lo].push(a);
        if let Some(b) = back {
            assigned[b] = hi;
            members[hi].push(b);
        }
    }
}

impl OwnerLayout {
    #[cfg(test)]
    pub(crate) fn new<T: FloatT>(
        data: &DefaultProblemData<T>,
        owners: usize,
    ) -> Result<Self, String> {
        if owners == 0 {
            return Err("owner count must be positive".into());
        }
        Self::build(data, Some(owners), 1, None, None)
    }

    /// Budget is the resolved worker budget, not the desired number of tasks.
    #[cfg(test)]
    pub(crate) fn new_auto<T: FloatT>(
        data: &DefaultProblemData<T>,
        budget: usize,
    ) -> Result<Self, String> {
        Self::build(data, None, budget, None, None)
    }

    #[cfg(test)]
    pub(crate) fn new_with_history<T: FloatT>(
        data: &DefaultProblemData<T>,
        owners: usize,
        budget: usize,
        input_fingerprint: Option<[u8; 32]>,
        history: &CostHistory,
    ) -> Result<Self, String> {
        if owners == 0 {
            return Err("owner count must be positive".into());
        }
        Self::build(data, Some(owners), budget, Some(history), input_fingerprint)
    }

    pub(super) fn build<T: FloatT>(
        data: &DefaultProblemData<T>,
        explicit: Option<usize>,
        budget: usize,
        history: Option<&CostHistory>,
        input_fingerprint: Option<[u8; 32]>,
    ) -> Result<Self, String> {
        let a = data.sampled.as_ref().map_or(&data.A, |s| s.linear());
        let (n, m) = (a.n, a.m);
        let mut units = Vec::new();
        let mut borders = Vec::new();
        let mut row_unit = vec![usize::MAX; m];
        let mut offset = 0usize;
        for (original, cone) in data.cones.iter().enumerate() {
            let end = offset
                .checked_add(cone.nvars())
                .filter(|&end| end <= m)
                .ok_or("cone rows exceed operator rows")?;
            match cone {
                SupportedConeT::ZeroConeT(_) => borders.push(OwnerCone {
                    original,
                    rows: offset..end,
                }),
                SupportedConeT::NonnegativeConeT(_) if offset < end => {
                    for row in offset..end {
                        row_unit[row] = units.len();
                        units.push(OwnerCone {
                            original,
                            rows: row..row + 1,
                        });
                    }
                }
                _ => {
                    row_unit[offset..end].fill(units.len());
                    units.push(OwnerCone {
                        original,
                        rows: offset..end,
                    });
                }
            }
            offset = end;
        }
        if offset != m {
            return Err("cone rows do not cover operator rows".into());
        }
        let count = n
            .checked_add(units.len())
            .ok_or("owner structure size overflow")?;
        let mut sets = Components::new(count);
        for col in 0..n {
            for &row in &a.rowval[a.colptr[col]..a.colptr[col + 1]] {
                if row_unit[row] != usize::MAX {
                    sets.join(col, n + row_unit[row]);
                }
            }
        }
        if let Some(sampled) = &data.sampled {
            for block in sampled.blocks() {
                let end = block.column_start + block.column_count();
                // Include stored-zero weights and every canonical factor column.
                // Valid installed sampled blocks occupy one complete PSD cone.
                let unit = row_unit[block.row_start];
                if unit == usize::MAX
                    || units[unit].rows != (block.row_start..block.row_start + block.row_count())
                {
                    return Err("sampled block must occupy a complete nonzero cone".into());
                }
                for col in block.column_start..end {
                    sets.join(col, n + unit);
                }
            }
        }
        for col in 0..n {
            for &row in &data.P.rowval[data.P.colptr[col]..data.P.colptr[col + 1]] {
                sets.join(col, row);
            }
        }
        // Structural work proxy: columns, cone rows and stored A/P entries.
        // Descending components, then stable root ID; ties choose lowest owner.
        let roots: Vec<_> = (0..count).map(|i| sets.root(i)).collect();
        let mut cost = vec![0u128; count];
        for col in 0..n {
            cost[roots[col]] += 1
                + (a.colptr[col + 1] - a.colptr[col]) as u128
                + (data.P.colptr[col + 1] - data.P.colptr[col]) as u128;
        }
        for (i, unit) in units.iter().enumerate() {
            cost[roots[n + i]] += 1 + unit.rows.len() as u128;
        }

        // Component identities are structural: member columns/units and all
        // incident stored matrix edges are fed in canonical index order.  The
        // authoritative numeric input fingerprint is validated separately by
        // `CostHistory`; this hash only prevents relabeling components when a
        // historical plan is imported.
        let mut component_hash = vec![0xcbf2_9ce4_8422_2325u64; count];
        for col in 0..n {
            let root = roots[col];
            component_hash[root] = structural_mix(component_hash[root], 0, col as u64);
        }
        for (i, _) in units.iter().enumerate() {
            let root = roots[n + i];
            component_hash[root] = structural_mix(component_hash[root], 1, i as u64);
        }
        for col in 0..n {
            for &row in &a.rowval[a.colptr[col]..a.colptr[col + 1]] {
                let root = roots[col];
                component_hash[root] = structural_mix(component_hash[root], 2, row as u64);
            }
            for &row in &data.P.rowval[data.P.colptr[col]..data.P.colptr[col + 1]] {
                let root = roots[col];
                component_hash[root] = structural_mix(component_hash[root], 3, row as u64);
            }
        }
        if let Some(sampled) = &data.sampled {
            for block in sampled.blocks() {
                let unit = row_unit[block.row_start];
                let root = roots[n + unit];
                for value in [
                    block.row_start as u64,
                    block.column_start as u64,
                    block.row_count() as u64,
                    block.column_count() as u64,
                    block.side() as u64,
                ] {
                    component_hash[root] = structural_mix(component_hash[root], 4, value);
                }
            }
        }
        let component_roots: Vec<_> = (0..count).filter(|&i| cost[i] > 0).collect();
        let component_ids: Vec<_> = component_roots
            .iter()
            .map(|&root| structural_finish(component_hash[root]))
            .collect();
        {
            let mut seen = component_ids.clone();
            seen.sort_unstable();
            if seen.windows(2).any(|pair| pair[0] == pair[1]) {
                return Err("structural component identity collision".into());
            }
        }
        let structural_weights: Vec<_> = component_roots
            .iter()
            .map(|&root| {
                u64::try_from(cost[root]).map_err(|_| "component structural weight overflow")
            })
            .collect::<Result<Vec<_>, _>>()?;
        let historical = history
            .map(|history| {
                history.validate(
                    input_fingerprint,
                    T::precision_bits(),
                    budget,
                    n,
                    m,
                    &component_ids,
                    &structural_weights,
                )
            })
            .transpose()?;
        let mut historical_by_root = vec![0.0; count];
        if let Some(costs) = &historical {
            for ((&root, &identity), &value) in
                component_roots.iter().zip(&component_ids).zip(costs)
            {
                let _ = identity;
                historical_by_root[root] = value;
            }
        }
        let owners = explicit.unwrap_or_else(|| {
            // Zero-dimensional cone units do not create useful tasks. Keep the
            // original LPT costs/order, including those units, for both routes.
            let mut nonempty = vec![false; count];
            for &root in &roots[..n] {
                nonempty[root] = true;
            }
            for (i, unit) in units.iter().enumerate() {
                if !unit.rows.is_empty() {
                    nonempty[roots[n + i]] = true;
                }
            }
            let component_count = nonempty.into_iter().filter(|&v| v).count();
            let border = borders.iter().map(|b| b.rows.len()).sum::<usize>();
            let memory_bound = if border == 0 {
                usize::MAX
            } else {
                1usize.saturating_add(m / border)
            };
            component_count
                .min(budget.max(1).saturating_mul(2))
                .min(memory_bound)
                .max(1)
        });
        let mut assigned = vec![0; count];
        if historical.is_some() {
            let mut order = component_roots.clone();
            order.sort_by(|&a, &b| {
                historical_by_root[b]
                    .total_cmp(&historical_by_root[a])
                    .then_with(|| component_hash[a].cmp(&component_hash[b]))
                    .then_with(|| a.cmp(&b))
            });
            let mut loads = vec![0.0f64; owners];
            for root in order {
                let owner = (0..owners)
                    .min_by(|&a, &b| loads[a].total_cmp(&loads[b]).then_with(|| a.cmp(&b)))
                    .unwrap();
                assigned[root] = owner;
                loads[owner] += historical_by_root[root];
            }
        } else {
            // Per-iteration dense work grows cubically with a component's
            // size: the leaf LDLᵀ over its columns and the SVD/eig of its PSD
            // cones. The linear structural weight undercounts large blocks
            // (Λ19 spins 0–50 on 8 owners: 1.13× the mean cubic work on the
            // busiest owner), so balance a cubic estimate instead.
            let mut work = vec![0u128; count];
            let mut columns = vec![0u128; count];
            for &root in &roots[..n] {
                columns[root] += 1;
            }
            for &root in &component_roots {
                work[root] = columns[root].pow(3) + cost[root];
            }
            for (i, unit) in units.iter().enumerate() {
                if let SupportedConeT::PSDTriangleConeT(side) = &data.cones[unit.original] {
                    work[roots[n + i]] += 8 * (*side as u128).pow(3);
                }
            }
            let mut order: Vec<_> = component_roots.clone();
            order.sort_by_key(|&i| (Reverse(work[i]), i));
            let mut queue: BinaryHeap<_> = (0..owners).map(|i| Reverse((0u128, i))).collect();
            for root in order {
                let Reverse((load, owner)) = queue.pop().unwrap();
                assigned[root] = owner;
                queue.push(Reverse((load + work[root], owner)));
            }
            if component_roots.len() <= REFINE_COMPONENTS {
                refine_assignment(&component_roots, &work, owners, &mut assigned);
            }
        }
        if explicit.is_some() {
            let mut with_cone = vec![false; count];
            for (i, unit) in units.iter().enumerate() {
                if !unit.rows.is_empty() {
                    with_cone[roots[n + i]] = true;
                }
            }
            let coned: Vec<usize> = component_roots
                .iter()
                .copied()
                .filter(|&root| with_cone[root])
                .collect();
            warn_uneven(&coned, &assigned, owners);
        }
        let mut owner_structural_weights = vec![0u128; owners];
        for &root in &component_roots {
            let owner = assigned[root];
            owner_structural_weights[owner] = owner_structural_weights[owner]
                .checked_add(cost[root])
                .ok_or("owner structural weight overflow")?;
        }
        let dominant_owner = if let Some(costs) = historical {
            let mut sums = vec![0.0f64; owners];
            for (&root, value) in component_roots.iter().zip(costs) {
                let owner = assigned[root];
                sums[owner] += value;
            }
            if sums.iter().any(|value| !value.is_finite()) {
                return Err("owner historical cost overflow".into());
            }
            let total = sums.iter().sum::<f64>();
            let (owner, cost) = sums.iter().enumerate().fold(
                (None, 0.0f64),
                |(best_owner, best_cost), (owner, &cost)| {
                    if cost > best_cost {
                        (Some(owner), cost)
                    } else {
                        (best_owner, best_cost)
                    }
                },
            );
            (total.is_finite() && total > 0.0 && cost / total >= 0.75)
                .then_some(owner)
                .flatten()
        } else {
            let total = owner_structural_weights
                .iter()
                .try_fold(0u128, |total, &value| total.checked_add(value))
                .ok_or("owner structural weight overflow")?;
            let (owner, weight) = owner_structural_weights.iter().enumerate().fold(
                (None, 0u128),
                |(best_owner, best_weight), (owner, &weight)| {
                    if weight > best_weight {
                        (Some(owner), weight)
                    } else {
                        (best_owner, best_weight)
                    }
                },
            );
            (weight > 0 && weight >= total.saturating_sub(total / 4))
                .then_some(owner)
                .flatten()
        };
        let mut out = Self {
            owners: vec![OwnerIndices::default(); owners],
            border_rows: borders.iter().flat_map(|b| b.rows.clone()).collect(),
            n,
            m,
            components: component_roots
                .iter()
                .zip(&component_ids)
                .zip(&structural_weights)
                .map(|((&root, &identity), &structural_weight)| LayoutComponent {
                    identity,
                    structural_weight,
                    owner: assigned[root],
                })
                .collect(),
            dominant_owner,
        };
        for col in 0..n {
            out.owners[assigned[roots[col]]].columns.push(col);
        }
        for (i, unit) in units.into_iter().enumerate() {
            out.owners[assigned[roots[n + i]]].cones.push(unit);
        }
        for (owner, indices) in out.owners.iter_mut().enumerate() {
            indices.cones.extend(borders.iter().cloned());
            indices.cones.sort_by_key(|c| (c.rows.start, c.original));
            indices.rows = indices.cones.iter().flat_map(|c| c.rows.clone()).collect();
            indices.counted_rows = indices
                .rows
                .iter()
                .enumerate()
                .filter_map(|(local, &global)| {
                    (owner == 0 || row_unit[global] != usize::MAX).then_some(local)
                })
                .collect();
        }
        Ok(out)
    }

    /// Whether this owner is the structurally or historically dominant owner
    /// selected at construction.  Worker count and kernel work thresholds are
    /// checked by the condensed planner itself.
    pub(crate) fn owner_inner_admission(&self, owner: usize) -> bool {
        self.dominant_owner == Some(owner)
    }
}

#[cfg(test)]
#[path = "tests/layout.rs"]
mod tests;
