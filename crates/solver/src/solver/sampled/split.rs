use super::*;

pub(super) fn block_chunks<T: FloatT>(
    blocks: &[SampledBlock<T>],
    pool: Option<&Arc<rayon::ThreadPool>>,
) -> usize {
    let Some(pool) = pool else {
        return 1;
    };
    let active = blocks.iter().filter(|b| b.basis_cols > 0).count().max(1);
    pool.current_num_threads().div_ceil(active).max(1)
}

/// Work-balanced interior index for a contiguous block range. Cumulative
/// structural work replaces the block count so that a tree of joins finishes
/// near the true half-work point rather than near the half-count point.
pub(super) fn balanced_block_split<T: FloatT>(blocks: &[SampledBlock<T>]) -> usize {
    debug_assert!(blocks.len() >= 2);
    let total: u128 = blocks.iter().map(|b| b.scheduled_work()).sum();
    if total == 0 {
        return blocks.len() / 2;
    }
    let (mut acc, mut best, mut best_gap) = (0u128, 1, u128::MAX);
    for (i, b) in blocks[..blocks.len() - 1].iter().enumerate() {
        acc += b.scheduled_work();
        let gap = acc.saturating_mul(2).abs_diff(total);
        if gap < best_gap {
            best_gap = gap;
            best = i + 1;
        }
    }
    best
}

/// Balanced interior split of an `s` level range. Level `s` owns `s + 1`
/// `(s, r)` pairs, so the work of `[start, end)` is `triangular_number(end) - triangular_number(start)`.
/// Pick the level that hands `left` of `total` shares to the left child.
pub(super) fn balanced_level_split(start: usize, end: usize, left: usize, total: usize) -> usize {
    debug_assert!(end >= start + 2 && left > 0 && left < total);
    let begin = tri_work(start);
    let target = begin + (tri_work(end) - begin) * left as u128 / total as u128;
    let mid = (((8 * target + 1).isqrt() - 1) / 2) as usize;
    mid.clamp(start + 1, end - 1)
}

pub(super) fn forward_split_chunks<T: FloatT>(
    b: &SampledBlock<T>,
    q: &BorrowedMatrix<'_, T>,
    x: &[T],
    alpha: T,
    s_range: std::ops::Range<usize>,
    out: &mut [T],
    sqrt2: T,
    inv_sqrt2: T,
    chunks: usize,
    square: &mut Matrix<T>,
    panel: &mut Matrix<T>,
    mut cache_q: Option<&mut ResidueCache>,
) {
    let kernel = T::residue_blas_applies(b.basis_rows, b.basis_rows, b.basis_cols);
    if s_range.len() == 1 && chunks > 1 && !kernel {
        // A lone `s` level owns a contiguous svec row band; split it into
        // `j` bands so a pool wider than the block count still fills. Each
        // task rebuilds the pair panels (h*kmax) and evaluates every
        // position's dot in the serial order, keeping writes disjoint and
        // bitwise identical.
        forward_band_chunks(
            b,
            q,
            x,
            alpha,
            s_range.start,
            0..b.basis_rows,
            out,
            sqrt2,
            inv_sqrt2,
            chunks,
            panel,
        );
        return;
    }
    if chunks <= 1 || s_range.len() < 2 || kernel {
        let s_offset = triangular_number(s_range.start * b.basis_rows);
        for s in s_range {
            for r in 0..=s {
                forward_pair(
                    b,
                    q,
                    x,
                    alpha,
                    s,
                    r,
                    sqrt2,
                    inv_sqrt2,
                    square,
                    panel,
                    cache_q.as_deref_mut(),
                    |idx, v| out[idx - s_offset] += v,
                );
            }
        }
        return;
    }
    let h = b.basis_rows;
    let kmax = b.basis_cols;
    let mid = balanced_level_split(s_range.start, s_range.end, chunks / 2, chunks);
    let split_idx = triangular_number(mid * h) - triangular_number(s_range.start * h);
    let (left, right) = out.split_at_mut(split_idx);
    let left_chunks = chunks / 2;
    let right_chunks = chunks - left_chunks;
    let mut local_square = Matrix::<T>::zeros((h, h));
    let mut local_panel = Matrix::<T>::zeros((h, kmax));
    rayon::join(
        || {
            forward_split_chunks(
                b,
                q,
                x,
                alpha,
                s_range.start..mid,
                left,
                sqrt2,
                inv_sqrt2,
                left_chunks,
                square,
                panel,
                cache_q,
            )
        },
        || {
            forward_split_chunks(
                b,
                q,
                x,
                alpha,
                mid..s_range.end,
                right,
                sqrt2,
                inv_sqrt2,
                right_chunks,
                &mut local_square,
                &mut local_panel,
                None,
            )
        },
    );
}

/// Forward pair `(s, r)`: `alpha·Σ_k w_k x_k q_k q_kᵀ` as one `panel·Qᵀ`
/// product, passed to `store(svec index, value)` for the pair's svec entries.
#[allow(clippy::too_many_arguments)]
pub(super) fn forward_pair<T: FloatT>(
    b: &SampledBlock<T>,
    q: &BorrowedMatrix<'_, T>,
    x: &[T],
    alpha: T,
    s: usize,
    r: usize,
    sqrt2: T,
    inv_sqrt2: T,
    square: &mut Matrix<T>,
    panel: &mut Matrix<T>,
    cache_q: Option<&mut ResidueCache>,
    mut store: impl FnMut(usize, T),
) {
    let h = b.basis_rows;
    let kmax = b.basis_cols;
    let p = (triangular_number(s) + r) * kmax;
    for k in 0..kmax {
        let weight = b.weights[p + k] * x[b.column_start + p + k];
        let q_col = &q.data()[k * h..(k + 1) * h];
        let p_col = &mut panel.data_mut()[k * h..(k + 1) * h];
        for i in 0..h {
            p_col[i] = q_col[i] * weight;
        }
    }
    // `panel·Qᵀ` is exactly symmetric: correctly rounded upper entries, mirrored.
    pooled_gemm_sym_cached(square, &*panel, &q.t(), None, cache_q);
    for j in 0..h {
        for i in 0..if r == s { j + 1 } else { h } {
            let scale = if r != s {
                inv_sqrt2
            } else if i != j {
                sqrt2
            } else {
                T::one()
            };
            store(
                triangular_number(s * h + j) + r * h + i,
                alpha * scale * square[(i, j)],
            );
        }
    }
}

/// Diagonal pair `(s, s)`, basis column `k`: `q_k'·S·q_k` for symmetric `S`
/// as one flat dot over the svec triangle against the constant `wdiag`
/// weights, which fix the term order (half the panel product's work).
#[inline]
pub(super) fn diag_pair_dot<T: FloatT>(x: &[T], wdiag: &[T], h: usize, s: usize, k: usize) -> T {
    let trih = triangular_number(h);
    let w = &wdiag[k * trih..(k + 1) * trih];
    T::dot_fma((0..h).flat_map(|j| {
        (0..=j).map(move |i| {
            (
                &x[triangular_number(s * h + j) + s * h + i],
                &w[triangular_number(j) + i],
            )
        })
    }))
}

/// Off-diagonal pair `(s, r)`: `q_k'·S_rs·q_k` for every basis column `k`,
/// through one `square·Q` panel product, passed to `store(k, value)`.
#[allow(clippy::too_many_arguments)]
pub(super) fn offdiag_pair_dots<T: FloatT>(
    q: &BorrowedMatrix<'_, T>,
    x: &[T],
    h: usize,
    s: usize,
    r: usize,
    inv_sqrt2: T,
    square: &mut Matrix<T>,
    panel: &mut Matrix<T>,
    mut store: impl FnMut(usize, T),
) {
    for j in 0..h {
        for i in 0..h {
            let (a, c) = (r * h + i, s * h + j);
            let scale = if a == c { T::one() } else { inv_sqrt2 };
            square[(i, j)] = x[triangular_number(c) + a] * scale;
        }
    }
    panel.mul(&*square, q, T::one(), T::zero());
    for k in 0..q.ncols() {
        let q_col = &q.data()[k * h..(k + 1) * h];
        let p_col = &panel.data()[k * h..(k + 1) * h];
        store(k, T::dot_fma(q_col.iter().zip(p_col.iter())));
    }
}

/// Serial adjoint terms of level `s`, pairs `r_range`: `out[i]` receives the
/// term of global column `base + i`.
#[allow(clippy::too_many_arguments)]
pub(super) fn adjoint_pairs<T: FloatT>(
    b: &SampledBlock<T>,
    q: &BorrowedMatrix<'_, T>,
    wdiag: &[T],
    x: &[T],
    alpha: T,
    s: usize,
    r_range: std::ops::RangeInclusive<usize>,
    inv_sqrt2: T,
    square: &mut Matrix<T>,
    panel: &mut Matrix<T>,
    mut store: impl FnMut(usize, T),
) {
    let h = b.basis_rows;
    let kmax = b.basis_cols;
    for r in r_range {
        let p = (triangular_number(s) + r) * kmax;
        if r == s {
            for k in 0..kmax {
                store(
                    p + k,
                    alpha * b.weights[p + k] * diag_pair_dot(x, wdiag, h, s, k),
                );
            }
        } else {
            offdiag_pair_dots(q, x, h, s, r, inv_sqrt2, square, panel, |k, v| {
                store(p + k, alpha * b.weights[p + k] * v)
            });
        }
    }
}

pub(super) fn adjoint_split_chunks<T: FloatT>(
    b: &SampledBlock<T>,
    q: &BorrowedMatrix<'_, T>,
    wdiag: &[T],
    x: &[T],
    alpha: T,
    s_range: std::ops::Range<usize>,
    out: &mut [T],
    inv_sqrt2: T,
    chunks: usize,
    square: &mut Matrix<T>,
    panel: &mut Matrix<T>,
) {
    if s_range.len() == 1 && chunks > 1 {
        // A lone `s` level still owns `s + 1` pairs (or, for a diagonal pair,
        // `kmax` independent column dots) — keep subdividing so a pool wider
        // than the block count is not left idle.
        adjoint_level_chunks(
            b,
            q,
            wdiag,
            x,
            alpha,
            s_range.start,
            0..=s_range.start,
            out,
            inv_sqrt2,
            chunks,
            square,
            panel,
        );
        return;
    }
    if chunks <= 1 || s_range.len() < 2 {
        let base = triangular_number(s_range.start) * b.basis_cols;
        for s in s_range {
            adjoint_pairs(
                b,
                q,
                wdiag,
                x,
                alpha,
                s,
                0..=s,
                inv_sqrt2,
                square,
                panel,
                |i, v| out[i - base] = v,
            );
        }
        return;
    }
    let h = b.basis_rows;
    let kmax = b.basis_cols;
    let mid = balanced_level_split(s_range.start, s_range.end, chunks / 2, chunks);
    let split_idx = (triangular_number(mid) - triangular_number(s_range.start)) * kmax;
    let (left, right) = out.split_at_mut(split_idx);
    let left_chunks = chunks / 2;
    let right_chunks = chunks - left_chunks;
    let mut local_square = Matrix::<T>::zeros((h, h));
    let mut local_panel = Matrix::<T>::zeros((h, kmax));
    rayon::join(
        || {
            adjoint_split_chunks(
                b,
                q,
                wdiag,
                x,
                alpha,
                s_range.start..mid,
                left,
                inv_sqrt2,
                left_chunks,
                square,
                panel,
            )
        },
        || {
            adjoint_split_chunks(
                b,
                q,
                wdiag,
                x,
                alpha,
                mid..s_range.end,
                right,
                inv_sqrt2,
                right_chunks,
                &mut local_square,
                &mut local_panel,
            )
        },
    );
}

/// Diagonal-pair dots are independent per basis column: each `out[k]` is one
/// `wdiag` flat dot evaluated in the same order as the serial leaf, so a
/// column-range split reproduces it bitwise.
fn adjoint_diag_chunks<T: FloatT>(
    b: &SampledBlock<T>,
    wdiag: &[T],
    x: &[T],
    alpha: T,
    s: usize,
    k_range: std::ops::Range<usize>,
    out: &mut [T],
    chunks: usize,
) {
    let h = b.basis_rows;
    let kmax = b.basis_cols;
    if chunks > 1 && k_range.len() > 1 {
        let mid = k_range.start + k_range.len() / 2;
        let (left, right) = out.split_at_mut(mid - k_range.start);
        let left_chunks = chunks / 2;
        let right_chunks = chunks - left_chunks;
        rayon::join(
            || adjoint_diag_chunks(b, wdiag, x, alpha, s, k_range.start..mid, left, left_chunks),
            || adjoint_diag_chunks(b, wdiag, x, alpha, s, mid..k_range.end, right, right_chunks),
        );
        return;
    }
    let p = (triangular_number(s) + s) * kmax;
    let k0 = k_range.start;
    for k in k_range {
        out[k - k0] = alpha * b.weights[p + k] * diag_pair_dot(x, wdiag, h, s, k);
    }
}

/// Split one `s` level's `(s, r)` pairs. Pair outputs are contiguous
/// (`kmax` entries each), so an `r` split keeps disjoint writes; a remaining
/// lone pair falls back to a column split whose sub-tasks each evaluate the
/// same per-`k` dots as the serial body.
fn adjoint_level_chunks<T: FloatT>(
    b: &SampledBlock<T>,
    q: &BorrowedMatrix<'_, T>,
    wdiag: &[T],
    x: &[T],
    alpha: T,
    s: usize,
    r_range: std::ops::RangeInclusive<usize>,
    out: &mut [T],
    inv_sqrt2: T,
    chunks: usize,
    square: &mut Matrix<T>,
    panel: &mut Matrix<T>,
) {
    let h = b.basis_rows;
    let kmax = b.basis_cols;
    let r0 = *r_range.start();
    let r1 = *r_range.end();
    if chunks > 1 && r0 < r1 {
        let mid = r0 + (r1 - r0 + 1) / 2;
        let (left, right) = out.split_at_mut((mid - r0) * kmax);
        let left_chunks = chunks / 2;
        let right_chunks = chunks - left_chunks;
        let mut local_square = Matrix::<T>::zeros((h, h));
        let mut local_panel = Matrix::<T>::zeros((h, kmax));
        rayon::join(
            || {
                adjoint_level_chunks(
                    b,
                    q,
                    wdiag,
                    x,
                    alpha,
                    s,
                    r0..=mid - 1,
                    left,
                    inv_sqrt2,
                    left_chunks,
                    square,
                    panel,
                )
            },
            || {
                adjoint_level_chunks(
                    b,
                    q,
                    wdiag,
                    x,
                    alpha,
                    s,
                    mid..=r1,
                    right,
                    inv_sqrt2,
                    right_chunks,
                    &mut local_square,
                    &mut local_panel,
                )
            },
        );
        return;
    }
    if chunks > 1 && r0 == s && kmax > 1 {
        adjoint_diag_chunks(b, wdiag, x, alpha, s, 0..kmax, out, chunks);
        return;
    }
    let base = (triangular_number(s) + r0) * kmax;
    adjoint_pairs(
        b,
        q,
        wdiag,
        x,
        alpha,
        s,
        r0..=r1,
        inv_sqrt2,
        square,
        panel,
        |i, v| out[i - base] = v,
    );
}

// Row ranges were checked disjoint at construction. Preserve each row's
// arithmetic order while allowing independent blocks to update in parallel.
/// `y_slice += alpha·A_b·x` over block `b`'s own rows: the pooled forward leaf.
fn forward_leaf<T: FloatT>(
    b: &SampledBlock<T>,
    w: &mut SampledBlockWorkspace<T>,
    y_slice: &mut [T],
    x: &[T],
    alpha: T,
    chunks: usize,
) {
    if b.basis_cols == 0 {
        return;
    }
    let h = b.basis_rows;
    let kmax = b.basis_cols;
    let sqrt2 = if h > 1 { T::SQRT_2() } else { T::zero() };
    let inv_sqrt2 = if h > 0 && b.dim > 1 {
        T::FRAC_1_SQRT_2()
    } else {
        T::zero()
    };
    let q = BorrowedMatrix {
        size: (h, kmax),
        data: b.basis.as_slice(),
        phantom: std::marker::PhantomData,
    };
    if chunks > 1 {
        with_panels(h, kmax, |square, panel| {
            forward_split_chunks(
                b,
                &q,
                x,
                alpha,
                0..b.dim,
                y_slice,
                sqrt2,
                inv_sqrt2,
                chunks,
                square,
                panel,
                w.q_fwd_cache.as_mut(),
            )
        });
    } else {
        w.forward_terms(b, x, alpha, |i, term| y_slice[i] += term);
    }
}

pub(super) fn forward_disjoint<T: FloatT>(
    blocks: &[SampledBlock<T>],
    work: &mut [SampledBlockWorkspace<T>],
    y: &mut [T],
    row_start: usize,
    x: &[T],
    alpha: T,
    chunks: usize,
) {
    if blocks.len() <= 1 {
        if let Some(b) = blocks.first() {
            let offset = b.row_start - row_start;
            forward_leaf(
                b,
                &mut work[0],
                &mut y[offset..offset + b.row_count()],
                x,
                alpha,
                chunks,
            );
        }
        return;
    }
    let mid = balanced_block_split(blocks);
    let split = blocks[mid].row_start;
    let (left_y, right_y) = y.split_at_mut(split - row_start);
    let (left_work, right_work) = work.split_at_mut(mid);
    rayon::join(
        || {
            forward_disjoint(
                &blocks[..mid],
                left_work,
                left_y,
                row_start,
                x,
                alpha,
                chunks,
            )
        },
        || forward_disjoint(&blocks[mid..], right_work, right_y, split, x, alpha, chunks),
    );
}

/// One `s` level's svec rows are contiguous (`triangular_number(s*h)..triangular_number((s+1)*h)`), so a
/// `j`-band split keeps disjoint writes while covering every `(s, r)` pair.
/// Position `(j, i)` stores `dot(panel_min(i,j), q_max(i,j))` — the same value
/// the serial path reads from its mirrored `square`, evaluated in the same
/// operand order. Each task refills each pair's panel; that `h*kmax` redo is
/// the only redundant work.
fn forward_band_chunks<T: FloatT>(
    b: &SampledBlock<T>,
    q: &BorrowedMatrix<'_, T>,
    x: &[T],
    alpha: T,
    s: usize,
    j_range: std::ops::Range<usize>,
    out: &mut [T],
    sqrt2: T,
    inv_sqrt2: T,
    chunks: usize,
    panel: &mut Matrix<T>,
) {
    let h = b.basis_rows;
    let kmax = b.basis_cols;
    if chunks > 1 && j_range.len() > 1 {
        let mid = balanced_level_split(j_range.start, j_range.end, chunks / 2, chunks);
        let split_idx = triangular_number(s * h + mid) - triangular_number(s * h + j_range.start);
        let (left, right) = out.split_at_mut(split_idx);
        let left_chunks = chunks / 2;
        let right_chunks = chunks - left_chunks;
        let mut local_panel = Matrix::<T>::zeros((h, kmax));
        rayon::join(
            || {
                forward_band_chunks(
                    b,
                    q,
                    x,
                    alpha,
                    s,
                    j_range.start..mid,
                    left,
                    sqrt2,
                    inv_sqrt2,
                    left_chunks,
                    panel,
                )
            },
            || {
                forward_band_chunks(
                    b,
                    q,
                    x,
                    alpha,
                    s,
                    mid..j_range.end,
                    right,
                    sqrt2,
                    inv_sqrt2,
                    right_chunks,
                    &mut local_panel,
                )
            },
        );
        return;
    }
    let base = triangular_number(s * h + j_range.start);
    for r in 0..=s {
        let p = (triangular_number(s) + r) * kmax;
        for k in 0..kmax {
            let weight = b.weights[p + k] * x[b.column_start + p + k];
            let q_col = &q.data()[k * h..(k + 1) * h];
            let p_col = &mut panel.data_mut()[k * h..(k + 1) * h];
            for i in 0..h {
                p_col[i] = q_col[i] * weight;
            }
        }
        let p_data = panel.data();
        let q_data = q.data();
        for j in j_range.clone() {
            for i in 0..if r == s { j + 1 } else { h } {
                let (a, c) = (i.min(j), i.max(j));
                let v = T::dot_fma((0..kmax).map(|k| (&p_data[a + k * h], &q_data[c + k * h])));
                let scale = if r != s {
                    inv_sqrt2
                } else if i != j {
                    sqrt2
                } else {
                    T::one()
                };
                out[triangular_number(s * h + j) - base + r * h + i] += alpha * scale * v;
            }
        }
    }
}
