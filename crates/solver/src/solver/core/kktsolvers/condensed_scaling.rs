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
    for q in 0..n {
        output[q * ld + offset..q * ld + offset + n].fill(T::zero());
    }
    for &(q, p, eidx) in plan {
        let e = &column.entries[eidx as usize];
        let v = if e.i == e.j {
            values[e.position]
        } else {
            values[e.position] * inv_sqrt2
        };
        let dst = &mut output[q as usize * ld + offset..q as usize * ld + offset + n];
        let src = &g.data()[p as usize * n..(p as usize + 1) * n];
        for (x, &y) in dst.iter_mut().zip(src) {
            *x = v.mul_add(y, *x);
        }
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

pub(super) fn apply_scaling<T: FloatT>(
    blocks: &mut [Block<T>],
    y: &mut [T],
    x: &[T],
    inverse: bool,
    gemm: Option<(&rayon::ThreadPool, usize)>,
) {
    y.fill(T::zero());
    let offset = blocks.first().map_or(0, |b| b.rows.start);
    for block in blocks {
        let rows = block.rows.start - offset..block.rows.end - offset;
        let (y, x) = (&mut y[rows.clone()], &x[rows]);
        match &mut block.scaling {
            Scaling::Psd(p) => p.apply(y, x, inverse, gemm),
            Scaling::Orthant { w, .. } => {
                for ((y, &x), &w) in y.iter_mut().zip(x).zip(w.iter()) {
                    *y = if inverse { (x / w) / w } else { w * (w * x) };
                }
            }
            Scaling::Zero => {}
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
    let workers = workers.max(1).min(costs.len());
    let mut prefix = Vec::with_capacity(costs.len() + 1);
    prefix.push(0u128);
    for &cost in costs {
        prefix.push(prefix.last().unwrap().saturating_add(cost));
    }
    let mut lanes = Vec::with_capacity(workers);
    let mut begin = 0;
    for lane in 0..workers {
        lanes.push(begin);
        let remaining = workers - lane;
        if remaining == 1 {
            break;
        }
        let target = (prefix[costs.len()] - prefix[begin]) / remaining as u128;
        let last = costs.len() - (remaining - 1);
        let mut end = begin + 1;
        while end < last && prefix[end] - prefix[begin] < target {
            end += 1;
        }
        if end > begin + 1
            && target.abs_diff(prefix[end - 1] - prefix[begin])
                <= target.abs_diff(prefix[end] - prefix[begin])
        {
            end -= 1;
        }
        begin = end;
    }
    lanes
}

pub(super) fn apply_scaling_pool<T: FloatT>(
    pool: &Option<Arc<rayon::ThreadPool>>,
    lanes: &[usize],
    tiles: usize,
    blocks: &mut [Block<T>],
    y: &mut [T],
    x: &[T],
    inverse: bool,
) {
    // A pool wider than the lane count cannot be filled by the lane level
    // alone: one congruence GEMM per block is a single task. `tiles` is derived
    // from the cost model in `scaling_dispatch`, so a dominant block is split
    // into disjoint column tiles even when the lane count already covers every
    // worker. Tiles stay 1 whenever no lane outweighs an equal share of the
    // pool, so narrow pools and balanced inputs keep the previous schedule.
    let gemm = pool.as_ref().map(|pool| (pool.as_ref(), tiles.max(1)));
    if let Some(pool) = pool {
        if lanes.len() > 1 {
            pool.install(|| split_scaling(blocks, y, x, inverse, lanes, gemm));
            return;
        }
    }
    apply_scaling(blocks, y, x, inverse, gemm);
}

pub(super) fn split_scaling<T: FloatT>(
    blocks: &mut [Block<T>],
    y: &mut [T],
    x: &[T],
    inverse: bool,
    lanes: &[usize],
    gemm: Option<(&rayon::ThreadPool, usize)>,
) {
    if lanes.len() <= 1 {
        apply_scaling(blocks, y, x, inverse, gemm);
        return;
    }
    let mid = lanes.len() / 2;
    let block = lanes[mid] - lanes[0];
    let row = blocks[block].rows.start - blocks[0].rows.start;
    let (left, right) = blocks.split_at_mut(block);
    let (yl, yr) = y.split_at_mut(row);
    let (xl, xr) = x.split_at(row);
    rayon::join(
        || split_scaling(left, yl, xl, inverse, &lanes[..mid], gemm),
        || split_scaling(right, yr, xr, inverse, &lanes[mid..], gemm),
    );
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
