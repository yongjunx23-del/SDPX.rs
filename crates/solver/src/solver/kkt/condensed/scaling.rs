use super::*;

// G*A in ascending source-row order, shared by scalar and batched providers.
// The cached plan retains the last occurrence of each coordinate, matching
// dense coefficient materialization even when the input contains duplicates.
#[inline(always)]
pub(super) fn coefficient_product<T: FloatT>(
    output: &mut [T],
    ld: usize,
    offset: usize,
    g: &Matrix<T>,
    column: &Column,
    plan: &[(u32, u32, u32)],
    values: &[T],
) {
    let n = g.nrows();
    let inv_sqrt2 = if T::precision_bits() <= 53
        || plan.iter().any(|&(_, _, eidx)| {
            let e = &column.entries[eidx as usize];
            e.i != e.j
        }) {
        T::FRAC_1_SQRT_2()
    } else {
        T::zero()
    };
    let coefficient = |&(_, p, eidx): &(u32, u32, u32)| {
        let e = &column.entries[eidx as usize];
        let v = if e.i == e.j {
            values[e.position]
        } else {
            values[e.position] * inv_sqrt2
        };
        (v, &g.data()[p as usize * n..(p as usize + 1) * n])
    };
    // The plan is sorted by output column. Each output element folds its
    // column's terms in plan order from zero, as AXPYs into a zeroed column
    // do, but in registers: one store per element instead of one per term.
    // A block shorter than R folds into its zeroed column; otherwise the last
    // register block overlaps the previous one and rewrites equal values.
    const R: usize = 16;
    let mut next = 0;
    for terms in plan.chunk_by(|a, b| a.0 == b.0) {
        let q = terms[0].0 as usize;
        for z in next..q {
            output[z * ld + offset..z * ld + offset + n].fill(T::zero());
        }
        let dst = &mut output[q * ld + offset..q * ld + offset + n];
        let fold = |acc: &mut [T], r0: usize| {
            for t in terms {
                let (v, src) = coefficient(t);
                for (a, &y) in acc.iter_mut().zip(&src[r0..]) {
                    *a = v.mul_add(y, *a);
                }
            }
        };
        if n < R {
            dst.fill(T::zero());
            fold(dst, 0);
        } else {
            for r0 in (0..n).step_by(R) {
                let r0 = r0.min(n - R);
                let mut acc = [T::zero(); R];
                fold(&mut acc, r0);
                dst[r0..r0 + R].copy_from_slice(&acc);
            }
        }
        next = q + 1;
    }
    for z in next..n {
        output[z * ld + offset..z * ld + offset + n].fill(T::zero());
    }
}

pub(super) fn psd_entry<T: FloatT>(G: &Matrix<T>, a: Entry, b: Entry, sqrt2: T) -> T {
    match (a.i == a.j, b.i == b.j) {
        (true, true) => G[(a.i, b.i)] * G[(a.i, b.i)],
        (true, false) => sqrt2 * G[(a.i, b.i)] * G[(a.i, b.j)],
        (false, true) => sqrt2 * G[(a.i, b.i)] * G[(a.j, b.i)],
        (false, false) => G[(a.i, b.i)] * G[(a.j, b.j)] + G[(a.i, b.j)] * G[(a.j, b.i)],
    }
}

#[derive(Clone, Copy)]
pub(super) enum ScalingAction<'a, T> {
    Apply(bool),
    Condense,
    Recover(&'a [T]),
}

fn apply_blocks<T: FloatT>(
    blocks: &mut [Block<T>],
    y: &mut [T],
    x: &[T],
    action: ScalingAction<'_, T>,
    gemm: Option<(&rayon::ThreadPool, usize)>,
) {
    y.fill(T::zero());
    let offset = blocks.first().map_or(0, |b| b.rows.start);
    for block in blocks {
        let rows = block.rows.start - offset..block.rows.end - offset;
        apply_block(block, &mut y[rows.clone()], &x[rows], action, gemm);
    }
}

/// One block of [`apply_blocks`]; `y` (the block's rows) must start zeroed.
pub(super) fn apply_block<T: FloatT>(
    block: &mut Block<T>,
    y: &mut [T],
    x: &[T],
    action: ScalingAction<'_, T>,
    gemm: Option<(&rayon::ThreadPool, usize)>,
) {
    let inverse = !matches!(action, ScalingAction::Apply(false));
    {
        match &mut block.scaling {
            Scaling::Psd(p) => {
                let a = action_index(&action);
                let start = std::time::Instant::now();
                with_split_hint(block.ways[a], || match action {
                    ScalingAction::Condense if p.sampled.is_some() => p.condense_rhs(x, gemm),
                    ScalingAction::Recover(primal) if p.sampled.is_some() => {
                        p.recover_rhs(y, primal, gemm)
                    }
                    _ => p.apply(y, x, inverse, gemm),
                });
                // Work, not wall time: a split call finishes sooner, and
                // using its wall time would oscillate the next allocation.
                block.cost[a] = start.elapsed().as_secs_f64() * block.ways[a] as f64;
            }
            Scaling::Orthant { w, .. } => {
                for ((y, &x), &w) in y.iter_mut().zip(x).zip(w.iter()) {
                    *y = if inverse { (x / w) / w } else { w * (w * x) };
                }
            }
            Scaling::Zero => {}
            Scaling::SocElim { w, eta, .. } => {
                // H = η²(2wwᵀ - J) and H⁻¹ = η⁻²(2(Jw)(Jw)ᵀ - J), wᵀJw = 1.
                let two: T = (2.).as_T();
                let c = if inverse {
                    two * (w[0] * x[0] - w[1..].dot(&x[1..]))
                } else {
                    two * w.dot(x)
                };
                y.copy_from_slice(x);
                y[0] = -x[0];
                if inverse {
                    y[0] += c * w[0];
                    for (y, &w) in y[1..].iter_mut().zip(&w[1..]) {
                        *y -= c * w;
                    }
                    y.scale((*eta * *eta).recip());
                } else {
                    y.axpby(c, w, T::one());
                    y.scale(*eta * *eta);
                }
            }
            _ if inverse => {} // Retained rows are never inverted.
            Scaling::Soc { w, eta } => {
                let two: T = (2.).as_T();
                let c = two * w.dot(x);
                y.copy_from_slice(x);
                if !y.is_empty() {
                    y[0] = -x[0];
                }
                y.axpby(c, w, T::one());
                y.scale(*eta * *eta);
            }
            Scaling::Ore(c) => {
                // As a PSD block: measured work sets the next call's ways,
                // which split S₀⁻¹'s rows without changing any entry.
                let a = action_index(&action);
                let start = std::time::Instant::now();
                with_split_hint(block.ways[a], || c.mul_Hs(y, x, &mut []));
                block.cost[a] = start.elapsed().as_secs_f64() * block.ways[a] as f64;
            }
            Scaling::OreLimit(c) => {
                let a = action_index(&action);
                let start = std::time::Instant::now();
                with_split_hint(block.ways[a], || c.mul_Hs(y, x, &mut []));
                block.cost[a] = start.elapsed().as_secs_f64() * block.ways[a] as f64;
            }
            Scaling::Dense3(h) => {
                let mut p = 0;
                for j in 0..3 {
                    for i in 0..=j {
                        y[i] = h[p].mul_add(x[j], y[i]);
                        if i != j {
                            y[j] = h[p].mul_add(x[i], y[j]);
                        }
                        p += 1;
                    }
                }
            }
            Scaling::GenPower {
                p,
                q,
                r,
                d1,
                d2,
                mu,
            } => {
                let dim1 = q.len();
                let cp = p.dot(x);
                let cq = q.dot(&x[..dim1]);
                let cr = r.dot(&x[dim1..]);
                for i in 0..dim1 {
                    y[i] = d1[i] * x[i] - cq * q[i];
                }
                for i in dim1..x.len() {
                    y[i] = *d2 * x[i] - cr * r[i - dim1];
                }
                y.axpby(cp, p, T::one());
                y.scale(*mu);
            }
        }
    }
}

pub(super) fn action_index<T>(action: &ScalingAction<'_, T>) -> usize {
    match action {
        ScalingAction::Condense => 0,
        ScalingAction::Recover(_) => 1,
        ScalingAction::Apply(_) => 2,
    }
}

/// Measured load balancing (SDPB 2.0 §2.2.2, worst-fit): a block whose last
/// measured cost for this action exceeds a worker's fair share `total/W`
/// gets `floor(cost/share)` ways for its residue products; the rest stay
/// serial. Costs are data dependent (prime counts follow exponent spreads),
/// so they are measured rather than modelled; the first call keeps 1 way.
/// Granting ways to fill a wave's idle tail (`planned_ways`) measured slower
/// here (Λ35-rs at 64 threads: prepare/recover/residual +5–10%, 2026-10-10),
/// and SDPB-style worst-fit within the worker count (spare workers to the
/// largest cost per way) was neutral (Λ27-rs/Λ19-rs ±0.7%, 2026-10-11).
pub(super) fn assign_ways<T>(blocks: &mut [Block<T>], action: usize, workers: usize) {
    let total: f64 = blocks.iter().map(|b| b.cost[action]).sum();
    if workers <= 1 || total <= 0.0 {
        return;
    }
    let share = total / workers as f64;
    for block in blocks.iter_mut() {
        block.ways[action] = measured_ways(block.cost[action], share);
    }
}

/// Choose the dispatch granularity of the lane-parallel block phases.
///
/// Lanes are dispatch units, not workers. `split_scaling` walks them as a
/// `rayon::join` tree whose leaves are work-stolen, so a finer partition lets a
/// wide pool approach the LPT bound of the same cost array: on a 22-block
/// uniform input the worker-count partition leaves a 1.31x lane imbalance at
/// eight workers, while one lane per block reaches 1.09x. Candidate lane counts
/// are scored by the LPT makespan of their contiguous cost partition, and the
/// winner decides whether the longest block also needs intra-block column
/// tiling so that the remaining workers can share it.
///
/// This is a structural cost model, not a runtime measurement: the plan stays a
/// function of the matrix sizes and the worker count alone.
pub(super) fn scaling_dispatch(costs: &[u128], workers: usize) -> (Vec<usize>, usize) {
    let workers = workers.max(1);
    if costs.is_empty() || workers == 1 {
        return (weighted_lanes(costs, 1), 1);
    }
    let fineness_cap = workers.saturating_mul(4).min(costs.len());
    let mut best = (u128::MAX, 1usize);
    for target in [workers, workers * 2, workers * 4, costs.len()] {
        let target = target.clamp(1, fineness_cap);
        let lanes = weighted_lanes(costs, target);
        let makespan = lpt_makespan(&lane_loads(costs, &lanes), workers);
        if makespan < best.0 || (makespan == best.0 && target < best.1) {
            best = (makespan, target);
        }
    }
    let lanes = weighted_lanes(costs, best.1);
    let loads = lane_loads(costs, &lanes);
    let total: u128 = costs.iter().sum();
    // One task would decide the makespan whenever the longest lane outweighs an
    // equal share of the pool; split its congruence GEMM into that many column
    // tiles. Lanes already shorter than a share keep tiles == 1.
    let share = (total / workers as u128).max(1);
    let tiles = loads
        .iter()
        .copied()
        .max()
        .unwrap_or(1)
        .div_ceil(share)
        .clamp(1, 16) as usize;
    (lanes, tiles)
}

pub(super) fn lane_loads(costs: &[u128], lanes: &[usize]) -> Vec<u128> {
    let mut loads = Vec::with_capacity(lanes.len());
    for (i, &begin) in lanes.iter().enumerate() {
        let end = lanes.get(i + 1).copied().unwrap_or(costs.len());
        loads.push(costs[begin..end].iter().sum());
    }
    loads
}

/// Longest-processing-time makespan of `loads` on `workers` identical workers.
pub(super) fn lpt_makespan(loads: &[u128], workers: usize) -> u128 {
    let mut busy = vec![0u128; workers.max(1)];
    let mut order: Vec<u128> = loads.to_vec();
    order.sort_unstable_by(|a, b| b.cmp(a));
    for load in order {
        let (_, slot) = busy
            .iter_mut()
            .enumerate()
            .min_by_key(|(_, busy)| **busy)
            .expect("nonempty worker list");
        *slot += load;
    }
    busy.into_iter().max().unwrap_or(0)
}

// Cached contiguous partitions use structural work, never runtime timings.
pub(super) fn weighted_lanes(costs: &[u128], workers: usize) -> Vec<usize> {
    if costs.is_empty() {
        return Vec::new();
    }
    let mut prefix = Vec::with_capacity(costs.len() + 1);
    prefix.push(0u128);
    for &cost in costs {
        prefix.push(prefix.last().unwrap().saturating_add(cost));
    }
    crate::utils::partition::contiguous_lanes(&prefix, workers.max(1).min(costs.len()))
}

#[cfg(test)]
pub(super) fn apply_scaling_pool<T: FloatT>(
    pool: &Option<Arc<rayon::ThreadPool>>,
    lanes: &[usize],
    tiles: usize,
    blocks: &mut [Block<T>],
    y: &mut [T],
    x: &[T],
    inverse: bool,
) {
    apply_scaling_pool_with_world(
        crate::mpi::World::get(),
        pool,
        lanes,
        tiles,
        pool.as_ref().map_or(1, |p| p.current_num_threads()),
        blocks,
        y,
        x,
        inverse,
    );
}

pub(super) fn apply_scaling_pool_with_world<T: FloatT>(
    world: Option<crate::mpi::World>,
    pool: &Option<Arc<rayon::ThreadPool>>,
    lanes: &[usize],
    tiles: usize,
    workers: usize,
    blocks: &mut [Block<T>],
    y: &mut [T],
    x: &[T],
    inverse: bool,
) {
    apply_block_pool_with_world(
        world,
        pool,
        lanes,
        tiles,
        workers,
        blocks,
        y,
        x,
        ScalingAction::Apply(inverse),
    );
}

pub(super) fn apply_block_pool_with_world<T: FloatT>(
    world: Option<crate::mpi::World>,
    pool: &Option<Arc<rayon::ThreadPool>>,
    lanes: &[usize],
    tiles: usize,
    workers: usize,
    blocks: &mut [Block<T>],
    y: &mut [T],
    x: &[T],
    action: ScalingAction<'_, T>,
) {
    // A pool wider than the lane count cannot be filled by the lane level
    // alone: one congruence GEMM per block is a single task. `tiles` is derived
    // from the cost model in `scaling_dispatch`, so a dominant block is split
    // into disjoint column tiles even when the lane count already covers every
    // worker. Tiles stay 1 whenever no lane outweighs an equal share of the
    // pool, so narrow pools and balanced inputs keep the previous schedule.
    let gemm = pool.as_ref().map(|pool| (pool.as_ref(), tiles.max(1)));
    if let Some(world) = world {
        // Rank-sharded scaling products. Block row ranges are disjoint, so
        // each rank fills its own y segment; gathered segments reproduce the
        // serial result bitwise.
        let ((y0, y1), gather_ranges) = rank_rows(blocks, world);
        let local = scale_owned(world, pool, tiles, blocks, &x[y0..y1], action);
        y.fill(T::zero());
        world.gather_slice(crate::mpi::SITE_SCALING, &local, &gather_ranges, &mut y[..]);
        return;
    }
    if let Some(pool) = pool {
        if lanes.len() > 1 {
            assign_ways(
                blocks,
                action_index(&action),
                workers.min(pool.current_num_threads()),
            );
            // Queued: the longest blocks (last measured) start first and
            // the shortest fill the tail.
            let workers = workers.min(pool.current_num_threads());
            pool.install(|| {
                let jobs = scaling_jobs(blocks, lanes, y, x, action, gemm);
                crate::algebra::run_queue(jobs, workers)
            });
            return;
        }
    }
    apply_blocks(blocks, y, x, action, gemm);
}

/// Per-rank contiguous block ranges of the sharded scaling. Ranks split by
/// cost, not count: a PSD block's congruence product is ~n^3 while orthant
/// rows are ~n.
pub(super) fn scaling_parts<T>(blocks: &[Block<T>], ranks: usize) -> Vec<(usize, usize)> {
    let costs: Vec<u64> = blocks
        .iter()
        .map(|b| {
            let n = (b.rows.end - b.rows.start) as f64;
            match &b.scaling {
                Scaling::Psd(_) => n.powf(1.5).max(1.0) as u64,
                _ => n.max(1.0) as u64,
            }
        })
        .collect();
    crate::mpi::cost_ranges(&costs, ranks)
}

/// This rank's row span and every rank's `(offset, len)` row range under
/// [`scaling_parts`].
pub(super) fn rank_rows<T>(
    blocks: &[Block<T>],
    world: crate::mpi::World,
) -> ((usize, usize), Vec<(usize, usize)>) {
    let span = |slice: &[Block<T>]| {
        let (s, e) = slice
            .iter()
            .map(|b| (b.rows.start, b.rows.end))
            .fold((usize::MAX, 0usize), |(a0, a1), (s, e)| {
                (a0.min(s), a1.max(e))
            });
        if s > e {
            (0, 0)
        } else {
            (s, e)
        }
    };
    let parts = scaling_parts(blocks, world.size());
    let (b0, len) = parts[world.rank()];
    let ranges = parts
        .iter()
        .map(|&(b0, len)| {
            if len == 0 {
                (0, 0)
            } else {
                let (s, e) = span(&blocks[b0..b0 + len]);
                (s, e - s)
            }
        })
        .collect();
    (span(&blocks[b0..b0 + len]), ranges)
}

/// Scale this rank's blocks of `x` (its row span) without exchanging:
/// returns the rank's output segment.
pub(super) fn scale_owned<T: FloatT>(
    world: crate::mpi::World,
    pool: &Option<Arc<rayon::ThreadPool>>,
    tiles: usize,
    blocks: &mut [Block<T>],
    x: &[T],
    action: ScalingAction<'_, T>,
) -> Vec<T> {
    let gemm = pool.as_ref().map(|pool| (pool.as_ref(), tiles.max(1)));
    let (b0, len) = scaling_parts(blocks, world.size())[world.rank()];
    let owned = &mut blocks[b0..b0 + len];
    let mut local = vec![T::zero(); x.len()];
    // One lane per owned block keeps block-level parallelism inside the
    // rank; a serial block walk would leave the pool idle between the
    // per-block GEMM tiles.
    let timer = crate::receipt::start();
    if let Some(pool) = pool.as_ref().filter(|_| owned.len() > 1) {
        let lanes: Vec<usize> = (0..owned.len()).collect();
        pool.install(|| split_scaling(owned, &mut local, x, action, &lanes, gemm));
    } else {
        apply_blocks(owned, &mut local, x, action, gemm);
    }
    crate::receipt::finish("scale.local", timer);
    local
}

pub(super) fn split_scaling<T: FloatT>(
    blocks: &mut [Block<T>],
    y: &mut [T],
    x: &[T],
    action: ScalingAction<'_, T>,
    lanes: &[usize],
    gemm: Option<(&rayon::ThreadPool, usize)>,
) {
    if lanes.len() <= 1 {
        apply_blocks(blocks, y, x, action, gemm);
        return;
    }
    let mid = lanes.len() / 2;
    let block = lanes[mid] - lanes[0];
    let row = blocks[block].rows.start - blocks[0].rows.start;
    let (left, right) = blocks.split_at_mut(block);
    let (yl, yr) = y.split_at_mut(row);
    let (xl, xr) = x.split_at(row);
    rayon::join(
        || split_scaling(left, yl, xl, action, &lanes[..mid], gemm),
        || split_scaling(right, yr, xr, action, &lanes[mid..], gemm),
    );
}

/// One pooled scaling pass as jobs for [`run_queue`], one per block, each
/// estimated by its last measured wall time. Each job zeroes and fills its
/// own rows (rows between blocks are zeroed here), so no serial pass touches
/// the whole vector; the per-block arithmetic is that of [`apply_blocks`].
/// One queue job per PSD or ORE block. Consecutive blocks of linear-cost
/// cones within one of `lanes` share a job: one-row orthants as separate
/// jobs serialize on the queue's lock (fix-a: 20200 per call).
pub(super) fn scaling_jobs<'a, T: FloatT>(
    blocks: &'a mut [Block<T>],
    lanes: &[usize],
    y: &'a mut [T],
    x: &'a [T],
    action: ScalingAction<'a, T>,
    gemm: Option<(&'a rayon::ThreadPool, usize)>,
) -> Vec<crate::algebra::Job<'a>> {
    type Part<'a, T> = (&'a mut Block<T>, &'a mut [T], &'a [T]);
    let a = action_index(&action);
    let offset = blocks.first().map_or(0, |b| b.rows.start);
    let mut jobs = Vec::with_capacity(blocks.len().min(lanes.len().max(1) * 2));
    let mut group: Vec<Part<'a, T>> = Vec::new();
    let flush = |group: &mut Vec<Part<'a, T>>, jobs: &mut Vec<crate::algebra::Job<'a>>| {
        if group.is_empty() {
            return;
        }
        let group = std::mem::take(group);
        let seconds = group
            .iter()
            .map(|(b, ..)| b.cost[a] / b.ways[a].max(1) as f64)
            .sum();
        let len = group.iter().map(|(_, rows, _)| rows.len()).sum();
        let job = move || {
            for (block, rows, xs) in group {
                rows.fill(T::zero());
                apply_block(block, rows, xs, action, gemm);
            }
        };
        jobs.push((seconds, len, Box::new(job) as Box<dyn FnOnce() + Send + 'a>));
    };
    let mut rest = y;
    let mut at = offset;
    let mut lane = 0;
    for (k, block) in blocks.iter_mut().enumerate() {
        while lane < lanes.len() && lanes[lane] <= k {
            flush(&mut group, &mut jobs);
            lane += 1;
        }
        let (gap, tail) = std::mem::take(&mut rest).split_at_mut(block.rows.start - at);
        gap.fill(T::zero());
        let (rows, tail) = tail.split_at_mut(block.rows.len());
        rest = tail;
        at = block.rows.end;
        let xs = &x[block.rows.start - offset..block.rows.end - offset];
        let alone = matches!(
            block.scaling,
            Scaling::Psd(_) | Scaling::Ore(_) | Scaling::OreLimit(_)
        );
        if alone {
            flush(&mut group, &mut jobs);
        }
        group.push((block, rows, xs));
        if alone {
            flush(&mut group, &mut jobs);
        }
    }
    flush(&mut group, &mut jobs);
    rest.fill(T::zero());
    jobs
}

pub(super) fn sparse_schur_value<T: FloatT>(
    left: &Column,
    right: &Column,
    ginv: &Matrix<T>,
    values: &[T],
    sqrt2: T,
) -> T {
    let mut v = T::zero();
    for a in &left.entries {
        for b in &right.entries {
            v += values[a.position] * values[b.position] * psd_entry(ginv, *a, *b, sqrt2);
        }
    }
    v
}

pub(super) fn split_sparse_columns<T: FloatT>(
    columns: &[Column],
    ginv: &Matrix<T>,
    values: &[T],
    output: &mut [T],
    lanes: &[usize],
    end: usize,
    sqrt2: T,
) {
    let begin = lanes[0];
    if lanes.len() == 1 {
        let offset = triangular_number(begin);
        for b in begin..end {
            let right = &columns[b];
            if !right.sparse {
                continue;
            }
            for (a, left) in columns[..=b].iter().enumerate() {
                output[triangular_number(b) + a - offset] =
                    sparse_schur_value(left, right, ginv, values, sqrt2);
            }
        }
    } else {
        let mid = lanes.len() / 2;
        let cut = lanes[mid];
        let (left, right) = output.split_at_mut(triangular_number(cut) - triangular_number(begin));
        rayon::join(
            || split_sparse_columns(columns, ginv, values, left, &lanes[..mid], cut, sqrt2),
            || split_sparse_columns(columns, ginv, values, right, &lanes[mid..], end, sqrt2),
        );
    }
}

pub(super) fn split_sampled_columns<T: FloatT>(
    columns: &[Column],
    sampled: &SampledPsd<T>,
    output: &mut [T],
    lanes: &[usize],
    end: usize,
) {
    let begin = lanes[0];
    if lanes.len() == 1 {
        let block = &sampled.operator.blocks()[sampled.block];
        let offset = triangular_number(begin);
        for b in begin..end {
            for a in 0..=b {
                output[triangular_number(b) + a - offset] = sampled.work.entry(
                    block,
                    columns[b].index - block.column_start,
                    columns[a].index - block.column_start,
                );
            }
        }
    } else {
        let mid = lanes.len() / 2;
        let cut = lanes[mid];
        let (left, right) = output.split_at_mut(triangular_number(cut) - triangular_number(begin));
        rayon::join(
            || split_sampled_columns(columns, sampled, left, &lanes[..mid], cut),
            || split_sampled_columns(columns, sampled, right, &lanes[mid..], end),
        );
    }
}

pub(super) fn schur_position<T>(S: &CscMatrix<T>, i: usize, j: usize) -> usize {
    let start = S.colptr[j];
    start
        + S.rowval[start..S.colptr[j + 1]]
            .binary_search(&i)
            .expect("missing structural Schur entry")
}
