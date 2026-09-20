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
/// `(s, r)` pairs, so the work of `[start, end)` is `tri(end) - tri(start)`.
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
) {
    if s_range.len() == 1 && chunks > 1 {
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
    if chunks <= 1 || s_range.len() < 2 {
        let h = b.basis_rows;
        let kmax = b.basis_cols;
        let s_offset = tri(s_range.start * h);
        for s in s_range {
            for r in 0..=s {
                let p = (tri(s) + r) * kmax;
                for k in 0..kmax {
                    let weight = b.weights[p + k] * x[b.column_start + p + k];
                    let q_col = &q.data()[k * h..(k + 1) * h];
                    let p_col = &mut panel.data_mut()[k * h..(k + 1) * h];
                    for i in 0..h {
                        p_col[i] = q_col[i] * weight;
                    }
                }
                let q_data = q.data();
                let p_data = panel.data();
                for j in 0..h {
                    for i in 0..=j {
                        let v = T::dot_fma(
                            (0..kmax).map(|k| (&p_data[i + k * h], &q_data[j + k * h])),
                        );
                        square[(i, j)] = v;
                        if r != s {
                            square[(j, i)] = v;
                        }
                    }
                }
                for j in 0..h {
                    for i in 0..if r == s { j + 1 } else { h } {
                        let scale = if r != s {
                            inv_sqrt2
                        } else if i != j {
                            sqrt2
                        } else {
                            T::one()
                        };
                        let idx = tri(s * h + j) + r * h + i - s_offset;
                        out[idx] += alpha * scale * square[(i, j)];
                    }
                }
            }
        }
        return;
    }
    let h = b.basis_rows;
    let kmax = b.basis_cols;
    let mid = balanced_level_split(s_range.start, s_range.end, chunks / 2, chunks);
    let split_idx = tri(mid * h) - tri(s_range.start * h);
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
            )
        },
    );
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
        let h = b.basis_rows;
        let kmax = b.basis_cols;
        let trih = tri(h);
        let s_offset = tri(s_range.start) * kmax;
        for s in s_range {
            for r in 0..=s {
                let p = (tri(s) + r) * kmax;
                if r == s {
                    // Same `q'·S·q` flat dots as `adjoint_terms`: bitwise
                    // identical because the `wdiag` weights fix the order.
                    for k in 0..kmax {
                        let w = &wdiag[k * trih..(k + 1) * trih];
                        let v = T::dot_fma((0..h).flat_map(|j| {
                            (0..=j).map(move |i| {
                                (
                                    &x[b.row_start + tri(r * h + j) + r * h + i],
                                    &w[tri(j) + i],
                                )
                            })
                        }));
                        let idx = p + k - s_offset;
                        out[idx] = alpha * b.weights[p + k] * v;
                    }
                    continue;
                }
                for j in 0..h {
                    for i in 0..h {
                        let (a, c) = (r * h + i, s * h + j);
                        let scale = if a == c { T::one() } else { inv_sqrt2 };
                        square[(i, j)] = x[b.row_start + tri(c) + a] * scale;
                    }
                }
                panel.mul(square, q, T::one(), T::zero());
                for k in 0..kmax {
                    let q_col = &q.data()[k * h..(k + 1) * h];
                    let p_col = &panel.data()[k * h..(k + 1) * h];
                    let v = T::dot_fma(q_col.iter().zip(p_col.iter()));
                    let idx = p + k - s_offset;
                    out[idx] = alpha * b.weights[p + k] * v;
                }
            }
        }
        return;
    }
    let h = b.basis_rows;
    let kmax = b.basis_cols;
    let mid = balanced_level_split(s_range.start, s_range.end, chunks / 2, chunks);
    let split_idx = (tri(mid) - tri(s_range.start)) * kmax;
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
    let trih = tri(h);
    if chunks > 1 && k_range.len() > 1 {
        let mid = k_range.start + k_range.len() / 2;
        let (left, right) = out.split_at_mut(mid - k_range.start);
        let left_chunks = chunks / 2;
        let right_chunks = chunks - left_chunks;
        rayon::join(
            || {
                adjoint_diag_chunks(
                    b,
                    wdiag,
                    x,
                    alpha,
                    s,
                    k_range.start..mid,
                    left,
                    left_chunks,
                )
            },
            || {
                adjoint_diag_chunks(
                    b,
                    wdiag,
                    x,
                    alpha,
                    s,
                    mid..k_range.end,
                    right,
                    right_chunks,
                )
            },
        );
        return;
    }
    let p = (tri(s) + s) * kmax;
    let k0 = k_range.start;
    for k in k_range {
        let w = &wdiag[k * trih..(k + 1) * trih];
        let v = T::dot_fma((0..h).flat_map(|j| {
            (0..=j).map(move |i| {
                (
                    &x[b.row_start + tri(s * h + j) + s * h + i],
                    &w[tri(j) + i],
                )
            })
        }));
        out[k - k0] = alpha * b.weights[p + k] * v;
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
    let trih = tri(h);
    for r in r0..=r1 {
        let p = (tri(s) + r) * kmax;
        if r == s {
            for k in 0..kmax {
                let w = &wdiag[k * trih..(k + 1) * trih];
                let v = T::dot_fma((0..h).flat_map(|j| {
                    (0..=j).map(move |i| {
                        (
                            &x[b.row_start + tri(r * h + j) + r * h + i],
                            &w[tri(j) + i],
                        )
                    })
                }));
                out[(r - r0) * kmax + k] = alpha * b.weights[p + k] * v;
            }
            continue;
        }
        for j in 0..h {
            for i in 0..h {
                let (a, c) = (r * h + i, s * h + j);
                let scale = if a == c { T::one() } else { inv_sqrt2 };
                square[(i, j)] = x[b.row_start + tri(c) + a] * scale;
            }
        }
        panel.mul(square, q, T::one(), T::zero());
        for k in 0..kmax {
            let q_col = &q.data()[k * h..(k + 1) * h];
            let p_col = &panel.data()[k * h..(k + 1) * h];
            let v = T::dot_fma(q_col.iter().zip(p_col.iter()));
            out[(r - r0) * kmax + k] = alpha * b.weights[p + k] * v;
        }
    }
}

// Row ranges were checked disjoint at construction. Preserve each row's
// arithmetic order while allowing independent blocks to update in parallel.
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
            if b.basis_cols > 0 {
                let offset = b.row_start - row_start;
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
                let y_slice = &mut y[offset..offset + b.row_count()];
                if chunks > 1 {
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
                        &mut work[0].square,
                        &mut work[0].panel,
                    );
                } else {
                    work[0].forward_terms(b, x, alpha, |i, term| y_slice[i] += term);
                }
            }
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

/// One `s` level's svec rows are contiguous (`tri(s*h)..tri((s+1)*h)`), so a
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
        let split_idx = tri(s * h + mid) - tri(s * h + j_range.start);
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
    let base = tri(s * h + j_range.start);
    for r in 0..=s {
        let p = (tri(s) + r) * kmax;
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
                let v = T::dot_fma(
                    (0..kmax).map(|k| (&p_data[a + k * h], &q_data[c + k * h])),
                );
                let scale = if r != s {
                    inv_sqrt2
                } else if i != j {
                    sqrt2
                } else {
                    T::one()
                };
                out[tri(s * h + j) - base + r * h + i] += alpha * scale * v;
            }
        }
    }
}
