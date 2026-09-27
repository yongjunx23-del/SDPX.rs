use super::*;

// One block of the PSD congruence through the shared pool. `gemm.1` is the
// desired number of column tiles; MPFR splits the output into disjoint tiles so
// every entry keeps its scalar accumulation order and results are unchanged.
// A single block cannot fill a wide pool on its own, so the outer lane level and
// this inner level share one pool instead of leaving most workers idle.
// Test-only proof that the tiled branch below actually runs: the equivalence
// assertions are worthless if the wide-pool path is silently skipped.
#[cfg(test)]
pub(crate) static POOLED_CONGRUENCE_TILES: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

// Test-only proof that the chunk-parallel dense transform actually ran. The
// serial and pooled assemblies are compared bitwise; that comparison is
// worthless if the pooled call silently kept one lane.
#[cfg(test)]
pub(crate) static PARALLEL_TRANSFORM_LANES: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

// Test-only proof that the tiled dense dot ran, not just the serial loop.
#[cfg(test)]
pub(crate) static PARALLEL_DOT_LANES: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

fn pooled_gemm<T: FloatT, MATA, MATB>(
    c: &mut Matrix<T>,
    a: &MATA,
    b: &MATB,
    gemm: Option<(&rayon::ThreadPool, usize)>,
) where
    MATA: DenseMatrix<T>,
    MATB: DenseMatrix<T>,
{
    let (m, n, k) = (a.nrows(), b.ncols(), a.ncols());
    if let Some((pool, tiles)) = gemm.filter(|(p, t)| *t > 1 && p.current_num_threads() > 1) {
        if n > 1 {
            #[cfg(test)]
            POOLED_CONGRUENCE_TILES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let tile = n.div_ceil(tiles.min(n));
            let (ta, tb) = (a.shape().as_blas_char(), b.shape().as_blas_char());
            let lda = if a.shape() == MatrixShape::N { m } else { k };
            let ldb = if b.shape() == MatrixShape::N { k } else { n };
            T::xgemm_pool(
                ta,
                tb,
                i32::try_from(m).unwrap(),
                i32::try_from(n).unwrap(),
                i32::try_from(k).unwrap(),
                T::one(),
                a.data(),
                i32::try_from(lda).unwrap(),
                b.data(),
                i32::try_from(ldb).unwrap(),
                T::zero(),
                c.data_mut(),
                i32::try_from(m).unwrap(),
                pool,
                tile,
            );
            return;
        }
    }
    c.mul(a, b, T::one(), T::zero());
}

/// Symmetric congruence `c = a·x·aᵀ` (or `aᵀ·x·a` with `transpose_a`). The
/// exact residue kernel keeps the intermediate exact and rounds once; other
/// precisions and small blocks keep the two pooled products through `work`.
fn congruence_sym<T: FloatT>(
    c: &mut Matrix<T>,
    a: &Matrix<T>,
    transpose_a: bool,
    x: &Matrix<T>,
    work: &mut Matrix<T>,
    gemm: Option<(&rayon::ThreadPool, usize)>,
    cache_a: &mut ResidueCache,
) {
    if congruence_exact_sym(c, a, transpose_a, x, gemm.map(|(p, _)| p), Some(cache_a)) {
        return;
    }
    if transpose_a {
        pooled_gemm(work, &a.t(), x, gemm);
        pooled_gemm_sym(c, work, a, gemm);
    } else {
        pooled_gemm(work, x, &a.t(), gemm);
        pooled_gemm_sym(c, a, work, gemm);
    }
}

/// Lane-private panels for the chunk-parallel dense transform. A lane keeps
/// its matrices across every chunk it owns, so the number of live panel
/// allocations is bounded by the lane count rather than by the chunk count.
pub(super) struct TransformPanels<T> {
    pub(super) mat2c: Matrix<T>,
    pub(super) mat3c: Matrix<T>,
    pub(super) support_product: Vec<T>,
}

impl<T: FloatT> Default for TransformPanels<T> {
    fn default() -> Self {
        Self {
            mat2c: Matrix::zeros((0, 0)),
            mat3c: Matrix::zeros((0, 0)),
            support_product: Vec::new(),
        }
    }
}

/// Write pointer to the packed dense vectors, shared by the transform lanes.
///
/// SAFETY: `transform_chunk` writes only the dense-position window of the
/// chunk it was handed, and chunk windows are disjoint, so concurrent lanes
/// never touch the same element. The buffer is not reallocated while lanes
/// run, and every lane stays inside its own row slice.
struct DenseVectorsPtr<T>(*mut T);
unsafe impl<T: Send> Send for DenseVectorsPtr<T> {}
unsafe impl<T: Send> Sync for DenseVectorsPtr<T> {}

/// Publish one tile of accumulators: every `(b, a)` pair with `a` in the tile
/// and `b >= a` is written once with the accumulator `acc[t * width + d]`.
/// Shared by the mapped and packed publish paths so both write identical pairs.
#[allow(clippy::too_many_arguments)]
#[inline]
fn publish_tile<T: FloatT, F: FnMut(usize, usize, usize, T)>(
    acc: &[T],
    columns: &[Column],
    dense_column_map: &[usize],
    width: usize,
    a0: usize,
    a1: usize,
    mut write: F,
) {
    let tmax_max = a1 - a0;
    for b in a0..columns.len() {
        if columns[b].sparse {
            continue;
        }
        let d = dense_column_map[b];
        let tmax = (b + 1 - a0).min(tmax_max);
        for t in 0..tmax {
            write(b, a0 + t, d, acc[t * width + d]);
        }
    }
}

/// Packed publish target for one block's Schur values: `triangular(b) + a` in
/// the block's own buffer, the layout the parallel assembly scatters from.
///
/// SAFETY: dense tiles write disjoint `(b, a)` sets (each pair has one `a`, so
/// one tile) and the sparse loop writes only pairs with sparse `b`; the buffer
/// outlives every lane.
struct PackedSchur<T> {
    out: *mut T,
    len: usize,
}

impl<T: FloatT> PackedSchur<T> {
    fn new(out: &mut [T]) -> Self {
        Self {
            out: out.as_mut_ptr(),
            len: out.len(),
        }
    }

    #[inline(always)]
    fn write(&self, b: usize, a: usize, value: T) {
        let index = triangular_number(b) + a;
        debug_assert!(index < self.len);
        // SAFETY: see the type comment; callers only ever write their own
        // disjoint pairs.
        unsafe { *self.out.add(index) = value };
    }
}

unsafe impl<T: Send> Send for PackedSchur<T> {}
unsafe impl<T: Send> Sync for PackedSchur<T> {}

/// Where a dense-tile assembly publishes a Schur entry: a caller closure over
/// precomputed global positions, or this block's packed buffer. Both arms are
/// monomorphized, so the store loop keeps a direct call.
enum SchurSink<'a, F, T> {
    Mapped(&'a mut F, std::marker::PhantomData<T>),
    Packed(PackedSchur<T>),
}

impl<T: FloatT, F: FnMut(usize, usize, usize, T)> SchurSink<'_, F, T> {
    #[inline(always)]
    fn write(&mut self, b: usize, a: usize, position: usize, value: T) {
        match self {
            SchurSink::Mapped(store, _) => store(b, a, position, value),
            SchurSink::Packed(packed) => packed.write(b, a, value),
        }
    }
}

/// Accumulate one left column's coefficient dot into row `t` of a tile buffer.
///
/// Returns true when the column reaches the end of the dense axis, which means
/// no later column can produce a store either. Shared by the serial and tiled
/// paths so both fold every accumulator in exactly the same order.
#[allow(clippy::too_many_arguments)]
#[inline]
fn accumulate_column<T: FloatT>(
    acc: &mut [T],
    columns: &[Column],
    dense_indices: &[usize],
    dense_representatives: &[usize],
    dense_row_offsets: &[usize],
    dense_vectors: &[T],
    values: &[T],
    width: usize,
    a0: usize,
    a: usize,
) -> bool {
    let t = a - a0;
    let left = &columns[a];
    let dmin = dense_indices.partition_point(|&c| c < a);
    if dmin == width {
        return true;
    }
    // Exact coefficient aliases share their left dot products too. Earlier rows
    // have a superset of this row's live suffix, so reuse within the existing
    // tile without another large cache.
    let representative = dense_representatives[a];
    if let Some(previous) = (a0..a).find(|&p| dense_representatives[p] == representative) {
        let source = (previous - a0) * width + dmin;
        acc.copy_within(source..source + width - dmin, t * width + dmin);
        return false;
    }
    // All live slices share one provable length so the inner loops carry no
    // per-element bounds checks. The FMA order per accumulator is unchanged,
    // keeping Schur values bitwise identical.
    let len = width - dmin;
    let row = &mut acc[t * width + dmin..][..len];
    row.iter_mut().for_each(|x| *x = T::zero());
    // Four independent source streams share one accumulator load/store,
    // retaining the entry-wise FMA order exactly.
    let mut wide_groups = left.entries.chunks_exact(8);
    for group in &mut wide_groups {
        let rows: [&[T]; 8] = std::array::from_fn(|k| {
            let e = &group[k];
            let pos = triangular_number(e.j) + e.i;
            let end = dense_row_offsets[pos + 1];
            &dense_vectors[end - len..][..len]
        });
        let v: [T; 8] = std::array::from_fn(|k| values[group[k].position]);
        // Keep each output's FMA order, but interleave four independent
        // outputs so the processor can overlap the dependent accumulator
        // chains. The source slices are contiguous along the dense axis.
        let mut i = 0;
        while i + 4 <= len {
            let mut x0 = row[i];
            let mut x1 = row[i + 1];
            let mut x2 = row[i + 2];
            let mut x3 = row[i + 3];
            for k in 0..8 {
                x0 = v[k].mul_add(rows[k][i], x0);
                x1 = v[k].mul_add(rows[k][i + 1], x1);
                x2 = v[k].mul_add(rows[k][i + 2], x2);
                x3 = v[k].mul_add(rows[k][i + 3], x3);
            }
            row[i] = x0;
            row[i + 1] = x1;
            row[i + 2] = x2;
            row[i + 3] = x3;
            i += 4;
        }
        for i in i..len {
            let mut value = row[i];
            for k in 0..8 {
                value = v[k].mul_add(rows[k][i], value);
            }
            row[i] = value;
        }
    }
    let mut groups = wide_groups.remainder().chunks_exact(4);
    for group in &mut groups {
        let rows: [&[T]; 4] = std::array::from_fn(|k| {
            let e = &group[k];
            let pos = triangular_number(e.j) + e.i;
            let end = dense_row_offsets[pos + 1];
            &dense_vectors[end - len..][..len]
        });
        let v: [T; 4] = std::array::from_fn(|k| values[group[k].position]);
        let mut i = 0;
        while i + 4 <= len {
            let mut x00 = row[i];
            let mut x01 = row[i + 1];
            let mut x02 = row[i + 2];
            let mut x03 = row[i + 3];
            for k in 0..4 {
                x00 = v[k].mul_add(rows[k][i], x00);
                x01 = v[k].mul_add(rows[k][i + 1], x01);
                x02 = v[k].mul_add(rows[k][i + 2], x02);
                x03 = v[k].mul_add(rows[k][i + 3], x03);
            }
            row[i] = x00;
            row[i + 1] = x01;
            row[i + 2] = x02;
            row[i + 3] = x03;
            i += 4;
        }
        for i in i..len {
            let x0 = v[0].mul_add(rows[0][i], row[i]);
            let x1 = v[1].mul_add(rows[1][i], x0);
            let x2 = v[2].mul_add(rows[2][i], x1);
            row[i] = v[3].mul_add(rows[3][i], x2);
        }
    }
    for e in groups.remainder() {
        let v = values[e.position];
        let pos = triangular_number(e.j) + e.i;
        let end = dense_row_offsets[pos + 1];
        let source = &dense_vectors[end - len..][..len];
        for i in 0..len {
            row[i] = v.mul_add(source[i], row[i]);
        }
    }
    false
}

/// Accumulate and publish one tile of left columns `[a0, a1)`.
#[allow(clippy::too_many_arguments)]
#[inline]
fn dot_tile<T: FloatT>(
    acc: &mut [T],
    columns: &[Column],
    dense_indices: &[usize],
    dense_representatives: &[usize],
    dense_column_map: &[usize],
    dense_row_offsets: &[usize],
    dense_vectors: &[T],
    values: &[T],
    width: usize,
    a0: usize,
    a1: usize,
    packed: &PackedSchur<T>,
) {
    for a in a0..a1 {
        if accumulate_column(
            acc,
            columns,
            dense_indices,
            dense_representatives,
            dense_row_offsets,
            dense_vectors,
            values,
            width,
            a0,
            a,
        ) {
            break;
        }
    }
    // An alias b uses a representative r >= b. Every required (a, b) therefore
    // has a computed (a, r); publish only a <= b.
    // One bounds check per tile keeps the packed publish honest in release
    // builds without paying for it on every store.
    assert!(
        triangular_number(columns.len() - 1) + a1 - 1 < packed.len,
        "packed Schur buffer too small for this block"
    );
    publish_tile(
        acc,
        columns,
        dense_column_map,
        width,
        a0,
        a1,
        |b, a, _d, v| packed.write(b, a, v),
    );
}

/// Tile the dense columns over the pool: SDPB-style within-block splitting, so
/// a single dominant cone can still use every worker. Tiles are dealt out
/// round robin because their work falls off with the suffix length.
#[allow(clippy::too_many_arguments)]
fn dot_tiles_parallel<T: FloatT>(
    columns: &[Column],
    dense_indices: &[usize],
    dense_representatives: &[usize],
    dense_column_map: &[usize],
    dense_row_offsets: &[usize],
    dense_vectors: &[T],
    values: &[T],
    width: usize,
    tile: usize,
    packed: &PackedSchur<T>,
    pool: &rayon::ThreadPool,
) {
    let column_count = columns.len();
    // The store work of a tile falls off with its suffix start, so one tile per
    // lane would leave the first lane with most of the work. Keep several tiles
    // per lane (dealt out round robin below) once the pool is wide.
    let tile = tile
        .min(
            column_count
                .div_ceil(pool.current_num_threads() * 3)
                .max(32),
        )
        .max(1);
    let tiles = column_count.div_ceil(tile.max(1)).max(1);
    let lanes = pool.current_num_threads().min(tiles).max(1);
    #[cfg(test)]
    PARALLEL_DOT_LANES.fetch_add(lanes, std::sync::atomic::Ordering::Relaxed);
    pool.install(|| {
        (0..lanes).into_par_iter().for_each(|lane| {
            let mut acc = vec![T::zero(); tile * width];
            let mut index = lane;
            while index < tiles {
                let a0 = index * tile;
                let a1 = (a0 + tile).min(column_count);
                dot_tile(
                    &mut acc,
                    columns,
                    dense_indices,
                    dense_representatives,
                    dense_column_map,
                    dense_row_offsets,
                    dense_vectors,
                    values,
                    width,
                    a0,
                    a1,
                    packed,
                );
                index += lanes;
            }
        });
    });
}

/// Transform one chunk of dense columns into the packed dense vectors: pack
/// `svec(Ginv A Ginv)` for every column of `chunk`.
///
/// This is the body of the original serial chunk loop, extracted so that
/// disjoint chunks can run on concurrent lanes. `dense_vectors` is written
/// through a raw pointer because each lane owns a different dense-position
/// window of it: chunk `c0` writes `[c0 * chunk_width, +chunk.len())` of every
/// packed row. Every entry is computed with exactly the arithmetic of the
/// serial loop, so the packed values are bitwise identical no matter how
/// chunks are distributed across lanes.
/// Thin dispatcher: the body is `#[inline(always)]`, so it lands in whichever
/// CPU context calls this.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn transform_chunk<T: FloatT>(
    ginv: &Matrix<T>,
    columns: &[Column],
    axpy_plans: &[Vec<(u32, u32, u32)>],
    coefficient_support: &[Vec<usize>],
    dense_row_offsets: &[usize],
    values: &[T],
    dense_vectors: *mut T,
    dense_len: usize,
    width: usize,
    chunk_width: usize,
    c0: usize,
    chunk: &[usize],
    panels: &mut TransformPanels<T>,
) {
    transform_chunk_body(
        ginv,
        columns,
        axpy_plans,
        coefficient_support,
        dense_row_offsets,
        values,
        dense_vectors,
        dense_len,
        width,
        chunk_width,
        c0,
        chunk,
        panels,
    );
}

/// Same body under the x86_64 features LLVM needs to select hardware FMA for
/// the `mul_add` loops. Lane bodies run inside a rayon closure, which cannot
/// inherit the caller's target features, so the lanes call this wrapper
/// explicitly after runtime detection. On aarch64 FMA is baseline and the
/// generic path is used.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn transform_chunk_avx2<T: FloatT>(
    ginv: &Matrix<T>,
    columns: &[Column],
    axpy_plans: &[Vec<(u32, u32, u32)>],
    coefficient_support: &[Vec<usize>],
    dense_row_offsets: &[usize],
    values: &[T],
    dense_vectors: *mut T,
    dense_len: usize,
    width: usize,
    chunk_width: usize,
    c0: usize,
    chunk: &[usize],
    panels: &mut TransformPanels<T>,
) {
    transform_chunk_body(
        ginv,
        columns,
        axpy_plans,
        coefficient_support,
        dense_row_offsets,
        values,
        dense_vectors,
        dense_len,
        width,
        chunk_width,
        c0,
        chunk,
        panels,
    );
}

/// Body of `transform_chunk`, split so the x86_64 wrapper can inline it under
/// the x86_64 feature set without duplicating any arithmetic.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn transform_chunk_body<T: FloatT>(
    ginv: &Matrix<T>,
    columns: &[Column],
    axpy_plans: &[Vec<(u32, u32, u32)>],
    coefficient_support: &[Vec<usize>],
    dense_row_offsets: &[usize],
    values: &[T],
    dense_vectors: *mut T,
    dense_len: usize,
    width: usize,
    chunk_width: usize,
    c0: usize,
    chunk: &[usize],
    panels: &mut TransformPanels<T>,
) {
    let n = ginv.nrows();
    // mat2 = Ginv * A via column AXPYs in GEMM's ascending-p order:
    // bitwise-identical sums at ~e*n flops instead of 2n^3. The W blocks are
    // stacked so the second product runs as one GEMM. Use only live rows in
    // the tail GEMM. resize retains capacity; every active input entry is
    // overwritten before multiplication.
    let ld = chunk.len() * n;
    panels.mat2c.resize((ld, n));
    panels.mat3c.resize((ld, n));
    {
        let m2c = panels.mat2c.data_mut();
        for (t, &ci) in chunk.iter().enumerate() {
            coefficient_product(m2c, ld, t * n, ginv, &columns[ci], &axpy_plans[ci], values);
        }
    }
    // Omit structurally zero intermediate columns. The coefficient support is
    // exact and survives value updates; this is not a rank approximation.
    // Require a large work reduction to amortize the extra packing and
    // per-column BLAS calls.
    if n >= 64
        && chunk
            .iter()
            .all(|&ci| 3 * coefficient_support[ci].len() <= n)
    {
        for (t, &ci) in chunk.iter().enumerate() {
            let support = &coefficient_support[ci];
            let k = support.len();
            panels.support_product.resize(2 * n * k, T::zero());
            let (left, right) = panels.support_product.split_at_mut(n * k);
            for (j, &q) in support.iter().enumerate() {
                left[j * n..(j + 1) * n]
                    .copy_from_slice(&panels.mat2c.data()[q * ld + t * n..q * ld + (t + 1) * n]);
            }
            for j in 0..n {
                for (i, &q) in support.iter().enumerate() {
                    right[j * k + i] = ginv[(q, j)];
                }
            }
            T::xgemm(
                b'N',
                b'N',
                n as i32,
                n as i32,
                k as i32,
                T::one(),
                left,
                n as i32,
                right,
                k as i32,
                T::zero(),
                &mut panels.mat3c.data_mut()[t * n..],
                ld as i32,
            );
        }
    } else {
        panels.mat3c.mul(&panels.mat2c, ginv, T::one(), T::zero());
    }
    // Pack only used row suffixes; writes remain contiguous across the chunk's
    // live columns. Unused coordinates have empty rows.
    let b3 = panels.mat3c.data();
    let d0 = c0 * chunk_width;
    let cw = chunk.len();
    let mut pos = 0;
    for col in 0..n {
        for row in 0..=col {
            let base = dense_row_offsets[pos];
            let start = width - (dense_row_offsets[pos + 1] - base);
            if start >= d0 + cw {
                pos += 1;
                continue;
            }
            let begin = start.saturating_sub(d0);
            let offset = base + d0 + begin - start;
            debug_assert!(offset + (cw - begin) <= dense_len);
            // SAFETY: [d0, d0 + cw) is this chunk's dense-position window; the
            // slice is that window clipped to the row's live columns, so it
            // stays inside the packed row, and no other chunk (hence no other
            // lane) writes any part of it.
            let out =
                unsafe { std::slice::from_raw_parts_mut(dense_vectors.add(offset), cw - begin) };
            if row == col {
                for (o, t) in out.iter_mut().zip(begin..) {
                    *o = b3[col * ld + t * n + row];
                }
            } else {
                for (o, t) in out.iter_mut().zip(begin..) {
                    *o = (b3[col * ld + t * n + row] + b3[row * ld + t * n + col])
                        * T::FRAC_1_SQRT_2();
                }
            }
            pos += 1;
        }
    }
}

impl<T: FloatT> PsdBlock<T> {
    /// Single-block constructor used by tests. The solver routes one shared
    /// pass over A through `from_columns` instead.
    #[cfg(test)]
    pub(super) fn new(n: usize, A: &CscMatrix<T>, rows: &Range<usize>) -> Self {
        let mut coordinates = Vec::with_capacity(rows.len());
        for j in 0..n {
            for i in 0..=j {
                coordinates.push((i, j));
            }
        }
        let mut columns = Vec::new();
        for col in 0..A.n {
            let mut entries = Vec::new();
            for position in A.colptr[col]..A.colptr[col + 1] {
                let row = A.rowval[position];
                if rows.contains(&row) {
                    let (i, j) = coordinates[row - rows.start];
                    entries.push(Entry { position, i, j });
                }
            }
            if !entries.is_empty() {
                columns.push(Column {
                    index: col,
                    entries,
                    sparse: false,
                    schur_positions: Vec::new(),
                });
            }
        }
        Self::from_columns(n, rows.len(), columns)
    }

    /// Build from pre-collected columns in increasing column order: exactly the
    /// non-empty columns of this block's rows, each entry in CSC position
    /// order. The shared A scan in `CondensedKKTSolver::new` produces this.
    pub(super) fn from_columns(n: usize, rows_len: usize, mut columns: Vec<Column>) -> Self {
        // Only the local assembly order changes; indices and CSC destinations
        // stay in the original coordinates. Sparse left coefficients occur in
        // more triangular dot products, reducing total dot work (MOSEK ISMP 2012).
        columns.sort_by_key(|c| c.entries.len());
        let mut prefix_entries = 0u128;
        for column in &mut columns {
            prefix_entries += column.entries.len() as u128;
            let rate: u128 = if T::precision_bits() <= 53 { 64 } else { 1 };
            let expanded = column
                .entries
                .iter()
                .map(|e| if e.i == e.j { 1u128 } else { 2 })
                .sum::<u128>();
            let dense = if T::precision_bits() <= 53 {
                // Preserve the qualified batched-BLAS crossover until its
                // replacement also wins end-to-end measurements.
                4 * (n as u128).pow(3) + prefix_entries
            } else {
                2 * (n as u128).pow(3) + 2 * n as u128 * expanded + prefix_entries
            };
            column.sparse = 8 * column.entries.len() as u128 * prefix_entries * rate <= dense;
        }
        // Structural grouping is independent of coefficients and survives A
        // updates. Numerical equality is checked again before every assembly.
        let mut groups = HashMap::<Vec<(usize, usize)>, Vec<usize>>::new();
        for (ci, column) in columns.iter().enumerate() {
            let key = column.entries.iter().map(|e| (e.i, e.j)).collect();
            groups.entry(key).or_default().push(ci);
        }
        let column_groups = groups.into_values().filter(|g| g.len() > 1).collect();
        let dense_representatives = vec![usize::MAX; columns.len()];
        let dense_column_map = vec![usize::MAX; columns.len()];
        let dense_indices = columns
            .iter()
            .enumerate()
            .filter(|(_, column)| !column.sparse)
            .map(|(index, _)| index)
            .collect();
        // A packed coordinate is read only by left columns containing it.
        // Cache its first local use, including stored zeros and sparse columns.
        let mut dense_row_first = Vec::new();
        if T::precision_bits() <= 53 {
            dense_row_first.resize(rows_len, columns.len());
            for (ci, column) in columns.iter().enumerate() {
                for e in &column.entries {
                    let pos = triangular_number(e.j) + e.i;
                    dense_row_first[pos] = dense_row_first[pos].min(ci);
                }
            }
        }
        let axpy_plans = columns
            .iter()
            .map(|column| {
                let mut plan = Vec::with_capacity(2 * column.entries.len());
                for (eidx, e) in column.entries.iter().enumerate() {
                    plan.push((e.j as u32, e.i as u32, eidx as u32));
                    if e.i != e.j {
                        plan.push((e.i as u32, e.j as u32, eidx as u32));
                    }
                }
                plan.sort_by_key(|t| (t.0, t.1));
                // Duplicate (q, p) pairs keep the last entry's value, matching
                // the `=` scatter semantics of the dense fill they replace.
                plan.dedup_by(|next, prev| {
                    if (next.0, next.1) == (prev.0, prev.1) {
                        *prev = *next;
                        true
                    } else {
                        false
                    }
                });
                plan
            })
            .collect();
        let coefficient_support = if T::precision_bits() <= 53 {
            columns
                .iter()
                .map(|column| {
                    let mut indices: Vec<_> =
                        column.entries.iter().flat_map(|e| [e.i, e.j]).collect();
                    indices.sort_unstable();
                    indices.dedup();
                    indices
                })
                .collect()
        } else {
            Vec::new()
        };
        Self {
            sqrt2: if n > 1 { T::SQRT_2() } else { T::zero() },
            R: Matrix::zeros((n, n)),
            Rinv: Matrix::zeros((n, n)),
            G: Matrix::zeros((n, n)),
            Ginv: Matrix::zeros((n, n)),
            rinv_cache: ResidueCache::default(),
            g_cache: ResidueCache::default(),
            ginv_cache: ResidueCache::default(),
            mat1: Matrix::zeros((n, n)),
            mat2: Matrix::zeros((n, n)),
            mat3: Matrix::zeros((n, n)),
            // Only fused sampled recovery needs this workspace. Every other
            // block must not reserve 128*n² unused scalars.
            mat3c: Matrix::zeros((0, 0)),
            vector: vec![T::zero(); rows_len],
            columns,
            schur_values: Vec::new(),
            sampled: None,
            sparse_column_lanes: Vec::new(),
            dense_indices,
            column_groups,
            dense_representatives,
            coefficient_plan_valid: false,
            dense_column_map,
            dense_vectors: Vec::new(),
            dense_row_first,
            dense_row_offsets: Vec::new(),
            dense_acc: Vec::new(),
            coefficient_support,
            transform_lanes: Vec::new(),
            axpy_plans,
        }
    }

    pub(super) fn configure_sparse_columns(&mut self, workers: usize) -> (u128, u128) {
        let words = T::precision_bits().div_ceil(64) as u128;
        let mut prefix_entries = 0u128;
        let mut dense_work = 0u128;
        let costs: Vec<_> = self
            .columns
            .iter()
            .map(|column| {
                prefix_entries += column.entries.len() as u128;
                if column.sparse {
                    8 * prefix_entries * column.entries.len() as u128 * words * words
                } else {
                    let n = self.R.size().0 as u128;
                    let expanded = column
                        .entries
                        .iter()
                        .map(|e| if e.i == e.j { 1u128 } else { 2 })
                        .sum::<u128>();
                    dense_work +=
                        (2 * n.pow(3) + 2 * n * expanded + prefix_entries) * words * words;
                    0
                }
            })
            .collect();
        // This is a structural work estimate, not a measured timing.
        let lanes =
            workers.min((costs.iter().sum::<u128>() / 4096).min(usize::MAX as u128) as usize);
        self.sparse_column_lanes = weighted_lanes(&costs, lanes.max(1));
        let sparse_work = costs.iter().sum::<u128>();
        (sparse_work, sparse_work + dense_work)
    }

    pub(super) fn compute_schur(
        &mut self,
        values: &[T],
        mut store: impl FnMut(usize, usize, usize, T),
    ) {
        self.compute_schur_selected(values, false, &mut store, None);
    }

    // A and column classifications are immutable between explicit updates.
    // q/b and cone scaling changes do not invalidate this exact equality proof.
    fn prepare_coefficients(&mut self, values: &[T]) {
        // Sampled blocks read their Gram, not `values`; no plan is needed.
        if self.coefficient_plan_valid || self.sampled.is_some() {
            return;
        }
        for (ci, rep) in self.dense_representatives.iter_mut().enumerate() {
            *rep = ci;
        }
        {
            let mut buckets = HashMap::<u64, Vec<usize>>::new();
            for group in &self.column_groups {
                buckets.clear();
                // Last matching dense column preserves triangular dot bounds.
                for &ci in group.iter().rev() {
                    if self.columns[ci].sparse {
                        continue;
                    }
                    // f64 is only a bucket key; full T equality below proves reuse,
                    // including values that collide after precision conversion.
                    let mut hash = std::collections::hash_map::DefaultHasher::new();
                    for e in &self.columns[ci].entries {
                        values[e.position]
                            .to_f64()
                            .unwrap()
                            .to_bits()
                            .hash(&mut hash);
                    }
                    let candidates = buckets.entry(hash.finish()).or_default();
                    if let Some(&cj) = candidates.iter().find(|&&cj| {
                        self.columns[ci]
                            .entries
                            .iter()
                            .zip(&self.columns[cj].entries)
                            .all(|(a, b)| values[a.position] == values[b.position])
                    }) {
                        self.dense_representatives[ci] = cj;
                    } else {
                        candidates.push(ci);
                    }
                }
            }
        }

        self.dense_indices.clear();
        self.dense_indices.extend(
            self.columns
                .iter()
                .enumerate()
                .filter(|(ci, c)| !c.sparse && self.dense_representatives[*ci] == *ci)
                .map(|(ci, _)| ci),
        );
        for (d, &ci) in self.dense_indices.iter().enumerate() {
            self.dense_column_map[ci] = d;
        }
        for ci in 0..self.columns.len() {
            if !self.columns[ci].sparse {
                self.dense_column_map[ci] = self.dense_column_map[self.dense_representatives[ci]];
            }
        }
        let width = self.dense_indices.len();
        // Rows retain only their used suffix in the dense-column axis. Exact
        // column reuse can change the width after A updates, so rebuild offsets.
        self.dense_row_offsets.clear();
        self.dense_row_offsets.push(0);
        for (pos, &first) in self.dense_row_first.iter().enumerate() {
            let start = self.dense_indices.partition_point(|&ci| ci < first);
            self.dense_row_offsets
                .push(self.dense_row_offsets[pos] + width - start);
        }
        if crate::receipt::profile_requested() {
            let entries: usize = self.columns.iter().map(|c| c.entries.len()).sum();
            eprintln!(
                "PSD_PLAN n={} columns={} entries={} dense={} sparse={}",
                self.Ginv.nrows(),
                self.columns.len(),
                entries,
                width,
                self.columns.iter().filter(|c| c.sparse).count()
            );
        }
        self.coefficient_plan_valid = true;
    }

    pub(super) fn compute_schur_selected(
        &mut self,
        values: &[T],
        skip_sparse: bool,
        mut store: impl FnMut(usize, usize, usize, T),
        // Assembly pool; `None` keeps the serial chunk loop.
        pool: Option<&rayon::ThreadPool>,
    ) {
        if let Some(sampled) = &self.sampled {
            let block = &sampled.operator.blocks()[sampled.block];
            for (b, right) in self.columns.iter().enumerate() {
                for (a, left) in self.columns[..=b].iter().enumerate() {
                    let value = sampled.work.entry(
                        block,
                        right.index - block.column_start,
                        left.index - block.column_start,
                    );
                    store(b, a, right.schur_positions[a], value);
                }
            }
            return;
        }
        // Dense columns pack svec(Ginv·A·Ginv) per column and then dot it
        // against every earlier coefficient vector. At Float64 the per-pair
        // dot is scalar and dominates, so batch it: keep one packed vector
        // per dense column in a workspace whose used row suffixes are
        // contiguous across columns so the accumulation
        // vectorizes over the dense-column axis. The entry order per
        // (b, a) pair is unchanged, so values are bitwise identical to the
        // per-column path. High precision keeps that path: its products are
        // scalar anyway and the svec_n × dense_count workspace is far more
        // expensive in wide arithmetic.
        self.prepare_coefficients(values);
        let dense_count = self.dense_indices.len();
        let batched = dense_count > 0
            && T::precision_bits() <= 53
            && (self.vector.len() as u128) * (dense_count as u128) <= (1u128 << 26);
        if batched {
            self.compute_schur_dense_batched(values, skip_sparse, &mut store, pool);
            return;
        }
        for (b, right) in self.columns.iter().enumerate() {
            if self.dense_representatives[b] != b {
                continue;
            }
            if skip_sparse && right.sparse {
                continue;
            }
            if !right.sparse {
                coefficient_product(
                    self.mat2.data_mut(),
                    self.Ginv.nrows(),
                    0,
                    &self.Ginv,
                    right,
                    &self.axpy_plans[b],
                    values,
                );
                pooled_gemm_sym(&mut self.mat3, &self.mat2, &self.Ginv, None);
                mat_to_svec(&mut self.vector, &self.mat3);
            }
            // Stream one transform per exact representative at every precision.
            // Float64 may batch these transforms; wide arithmetic keeps O(n²) scratch.
            for (target, column) in self.columns[..=b].iter().enumerate() {
                if self.dense_representatives[target] != b {
                    continue;
                }
                for (a, left) in self.columns[..=target].iter().enumerate() {
                    let mut v = T::zero();
                    if right.sparse {
                        v = sparse_schur_value(left, right, &self.Ginv, values, self.sqrt2);
                    } else {
                        for e in &left.entries {
                            v = values[e.position]
                                .mul_add(self.vector[triangular_number(e.j) + e.i], v);
                        }
                    }
                    store(target, a, column.schur_positions[a], v);
                }
            }
        }
    }

    pub(super) fn compute_schur_dense_batched(
        &mut self,
        values: &[T],
        skip_sparse: bool,
        store: impl FnMut(usize, usize, usize, T),
        pool: Option<&rayon::ThreadPool>,
    ) {
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            // SAFETY: both required CPU features were checked above.
            return unsafe { self.compute_schur_dense_fma(values, skip_sparse, store, pool) };
        }
        self.compute_schur_dense_impl(values, skip_sparse, store, pool);
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn compute_schur_dense_fma(
        &mut self,
        values: &[T],
        skip_sparse: bool,
        store: impl FnMut(usize, usize, usize, T),
        pool: Option<&rayon::ThreadPool>,
    ) {
        self.compute_schur_dense_impl(values, skip_sparse, store, pool);
    }

    // One arithmetic implementation, inlined into either CPU context. The
    // per-accumulator FMA order is unchanged; there is no fast-math reduction.
    #[inline(always)]
    pub(super) fn compute_schur_dense_impl(
        &mut self,
        values: &[T],
        skip_sparse: bool,
        mut store: impl FnMut(usize, usize, usize, T),
        pool: Option<&rayon::ThreadPool>,
    ) {
        let mut sink = SchurSink::Mapped(&mut store, std::marker::PhantomData);
        self.compute_schur_dense_sink(values, skip_sparse, &mut sink, pool);
    }

    /// Packed publish used by the parallel assembly: entries land in this
    /// block's own buffer at `triangular(b) + a`, which is what
    /// `scatter_schur` reads back in cone order.
    pub(super) fn compute_schur_packed(
        &mut self,
        values: &[T],
        skip_sparse: bool,
        out: &mut [T],
        pool: Option<&rayon::ThreadPool>,
    ) {
        self.prepare_coefficients(values);
        let dense_count = self.dense_indices.len();
        let batched = self.sampled.is_none()
            && dense_count > 0
            && T::precision_bits() <= 53
            && (self.vector.len() as u128) * (dense_count as u128) <= (1u128 << 26);
        if batched {
            let mut sink: SchurSink<'_, fn(usize, usize, usize, T), T> =
                SchurSink::Packed(PackedSchur::new(out));
            self.compute_schur_dense_sink(values, skip_sparse, &mut sink, pool);
            return;
        }
        // Sampled and wide-precision blocks keep the closure path; only the
        // f64 batched dense path tiles over the pool.
        let packed = PackedSchur::new(out);
        self.compute_schur_selected(
            values,
            skip_sparse,
            |b, a, _, v| packed.write(b, a, v),
            pool,
        );
    }

    /// The one dense-tile implementation, parameterized by its publish target.
    #[inline(always)]
    fn compute_schur_dense_sink<F: FnMut(usize, usize, usize, T)>(
        &mut self,
        values: &[T],
        skip_sparse: bool,
        sink: &mut SchurSink<'_, F, T>,
        pool: Option<&rayon::ThreadPool>,
    ) {
        let svec_n = self.vector.len();
        let width = self.dense_indices.len();
        self.dense_vectors
            .resize(self.dense_row_offsets[svec_n], T::zero());
        const CHUNK: usize = 64;
        let n = self.Ginv.nrows();
        // Bound each Float64 congruence panel to about 2 MiB. Large PSD
        // blocks otherwise turn a fixed 64-column tile into a cache spill.
        let pool_threads = pool.map_or(1, |pool| pool.current_num_threads());
        let max_panel = CHUNK.min((262144 / (n * n)).max(1));
        // Dense-position windows are disjoint across chunks, so lanes can
        // transform different chunks concurrently and publish without a merge
        // or a lock. Each lane keeps its own panels and reuses them across the
        // chunks it owns; the serial path is lane zero of the same code.
        let chunk_width = if pool_threads > 1 {
            // Split at least once per worker so no lane is idle; panels only
            // shrink, so the cache bound above still holds.
            max_panel.min(width.div_ceil(pool_threads)).max(1)
        } else {
            max_panel
        };
        let chunk_count = width.div_ceil(chunk_width).max(1);
        let lanes = pool_threads.min(chunk_count);
        let transform_timer = crate::receipt::start();
        // Larger left tiles amortize Schur-position walks and expose more
        // exact aliases. Bound the accumulator to 4 MiB at Float64 (or one
        // row for very wide blocks) rather than growing it with both axes.
        let tile = 256.min((524288 / width).max(1));
        self.dense_acc.clear();
        self.dense_acc.resize(tile * width, T::zero());
        let per_lane = chunk_count.div_ceil(lanes);
        {
            let ginv = &self.Ginv;
            let columns = &self.columns;
            let axpy_plans = &self.axpy_plans;
            let coefficient_support = &self.coefficient_support;
            let dense_row_offsets = &self.dense_row_offsets;
            let dense_indices = &self.dense_indices;
            let transform_lanes = &mut self.transform_lanes;
            if transform_lanes.len() < lanes {
                transform_lanes.resize_with(lanes, TransformPanels::default);
            }
            // Bind the wrapper by reference: a closure that reaches `dv.0`
            // directly would capture the raw pointer, which is not `Sync`.
            let dv = &DenseVectorsPtr(self.dense_vectors.as_mut_ptr());
            let dense_len = self.dense_vectors.len();
            // A rayon lane runs in a separate function body and cannot inherit
            // the caller's CPU features. Detect once and call the explicit
            // x86_64 wrapper so `mul_add` still selects hardware FMA there.
            #[cfg(target_arch = "x86_64")]
            let lanes_avx2 =
                std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma");
            // A lane's share of the chunks: one contiguous run, so its panels
            // are reused by every chunk it owns.
            let run_lane = |lane: usize, panels: &mut TransformPanels<T>| {
                let first = lane * per_lane;
                let last = ((lane + 1) * per_lane).min(chunk_count);
                for c0 in first..last {
                    let start = c0 * chunk_width;
                    let end = (start + chunk_width).min(width);
                    let chunk = &dense_indices[start..end];
                    #[cfg(target_arch = "x86_64")]
                    if lanes_avx2 {
                        // SAFETY: both features were detected just above.
                        unsafe {
                            transform_chunk_avx2(
                                ginv,
                                columns,
                                axpy_plans,
                                coefficient_support,
                                dense_row_offsets,
                                values,
                                dv.0,
                                dense_len,
                                width,
                                chunk_width,
                                c0,
                                chunk,
                                panels,
                            )
                        };
                        continue;
                    }
                    transform_chunk(
                        ginv,
                        columns,
                        axpy_plans,
                        coefficient_support,
                        dense_row_offsets,
                        values,
                        dv.0,
                        dense_len,
                        width,
                        chunk_width,
                        c0,
                        chunk,
                        panels,
                    );
                }
            };
            match pool {
                Some(pool) if lanes > 1 => {
                    #[cfg(test)]
                    PARALLEL_TRANSFORM_LANES.fetch_add(lanes, std::sync::atomic::Ordering::Relaxed);
                    let run_lane = &run_lane;
                    pool.install(|| {
                        transform_lanes
                            .par_iter_mut()
                            .enumerate()
                            .for_each(|(lane, panels)| run_lane(lane, panels));
                    });
                }
                _ => {
                    for (lane, panels) in transform_lanes.iter_mut().enumerate().take(lanes) {
                        run_lane(lane, panels);
                    }
                }
            }
        }
        crate::receipt::finish("schur.transform", transform_timer);
        let dot_timer = crate::receipt::start();
        // Tile left columns so the store pass reads each column's
        // schur_positions once per tile instead of once per (a, d) pair.
        // Row t of the tile buffer accumulates left column a0 + t over the
        // same entries in the same FMA order as the untiled loop.
        let column_count = self.columns.len();
        // SDPB-style within-block split: with few, large cones the block-level
        // plan alone leaves most workers idle, so when the caller publishes
        // into this block's packed buffer the dense tiles also run on the
        // pool. Tiles write disjoint (b, a) pairs, so values are unchanged.
        if let (Some(pool), SchurSink::Packed(packed)) = (pool, &*sink) {
            if pool.current_num_threads() > 1 && column_count > tile {
                dot_tiles_parallel(
                    &self.columns,
                    &self.dense_indices,
                    &self.dense_representatives,
                    &self.dense_column_map,
                    &self.dense_row_offsets,
                    &self.dense_vectors,
                    values,
                    width,
                    tile,
                    packed,
                    pool,
                );
                crate::receipt::finish("schur.dot_scatter", dot_timer);
                self.publish_sparse(values, skip_sparse, sink);
                return;
            }
        }
        let mut a0 = 0;
        while a0 < column_count {
            let a1 = (a0 + tile).min(column_count);
            let mut exhausted = false;
            let fma_timer = crate::receipt::start();
            for a in a0..a1 {
                if accumulate_column(
                    &mut self.dense_acc,
                    &self.columns,
                    &self.dense_indices,
                    &self.dense_representatives,
                    &self.dense_row_offsets,
                    &self.dense_vectors,
                    values,
                    width,
                    a0,
                    a,
                ) {
                    // The column reaches the end of the dense axis, so no
                    // later column produces a store either.
                    exhausted = true;
                    break;
                }
            }
            crate::receipt::finish("schur.dot_fma", fma_timer);
            let store_timer = crate::receipt::start();
            {
                let columns = &self.columns;
                publish_tile(
                    &self.dense_acc,
                    columns,
                    &self.dense_column_map,
                    width,
                    a0,
                    a1,
                    |b, a, _d, v| {
                        let position = columns[b].schur_positions[a];
                        sink.write(b, a, position, v);
                    },
                );
            }
            crate::receipt::finish("schur.dot_store", store_timer);
            if exhausted {
                break;
            }
            a0 = a1;
        }
        crate::receipt::finish("schur.dot_scatter", dot_timer);
        self.publish_sparse(values, skip_sparse, sink);
    }

    /// Sparse left columns keep their exact scalar path, published through the
    /// same sink.
    fn publish_sparse<F: FnMut(usize, usize, usize, T)>(
        &self,
        values: &[T],
        skip_sparse: bool,
        sink: &mut SchurSink<'_, F, T>,
    ) {
        let sparse_timer = crate::receipt::start();
        for (b, right) in self.columns.iter().enumerate() {
            if !right.sparse || skip_sparse {
                continue;
            }
            for (a, left) in self.columns[..=b].iter().enumerate() {
                let v = sparse_schur_value(left, right, &self.Ginv, values, self.sqrt2);
                sink.write(b, a, right.schur_positions[a], v);
            }
        }
        crate::receipt::finish("schur.sparse", sparse_timer);
    }

    pub(super) fn scatter_schur(&self, S: &mut CscMatrix<T>) {
        for (b, column) in self.columns.iter().enumerate() {
            for (a, &position) in column.schur_positions.iter().enumerate() {
                S.nzval[position] += self.schur_values[triangular_number(b) + a];
            }
        }
    }

    // Half of H^-1 stays factored, so neither fused application squares Rinv.
    pub(super) fn condense_rhs(&mut self, rhs: &[T], gemm: Option<(&rayon::ThreadPool, usize)>) {
        svec_to_mat(&mut self.mat1, rhs);
        congruence_sym(
            &mut self.mat3c,
            &self.Rinv,
            false,
            &self.mat1,
            &mut self.mat2,
            gemm,
            &mut self.rinv_cache,
        );
        let sampled = self.sampled.as_mut().unwrap();
        sampled.work.inverse_adjoint(
            &sampled.operator.blocks()[sampled.block],
            &self.mat3c,
            &mut sampled.adjoint,
        );
    }

    pub(super) fn recover_rhs(
        &mut self,
        y: &mut [T],
        x: &[T],
        gemm: Option<(&rayon::ThreadPool, usize)>,
    ) {
        let sampled = self.sampled.as_mut().unwrap();
        sampled
            .work
            .inverse_forward(&sampled.operator.blocks()[sampled.block], x, &mut self.mat1);
        for (v, &b) in self.mat1.data_mut().iter_mut().zip(self.mat3c.data()) {
            *v -= b;
        }
        congruence_sym(
            &mut self.mat3,
            &self.Rinv,
            true,
            &self.mat1,
            &mut self.mat2,
            gemm,
            &mut self.rinv_cache,
        );
        mat_to_svec(y, &self.mat3);
    }

    pub(super) fn apply(
        &mut self,
        y: &mut [T],
        x: &[T],
        inverse: bool,
        gemm: Option<(&rayon::ThreadPool, usize)>,
    ) {
        svec_to_mat(&mut self.mat1, x);
        if T::precision_bits() <= 53 {
            // binary64: apply H = R·Rᵀ (H⁻¹ = Rinvᵀ·Rinv) through its factor
            // in two congruences. The squared G/Ginv lose cond(R)² accuracy
            // near convergence, so refinement would converge to a different
            // operator than the cones' W-based `Δs` and the primal residual
            // grows.
            let (r, first, second) = if inverse {
                (&self.Rinv, false, true)
            } else {
                (&self.R, true, false)
            };
            let cache = &mut self.rinv_cache;
            congruence_sym(
                &mut self.mat3,
                r,
                first,
                &self.mat1,
                &mut self.mat2,
                gemm,
                cache,
            );
            mat_to_svec(&mut self.vector, &self.mat3);
            svec_to_mat(&mut self.mat1, &self.vector);
            congruence_sym(
                &mut self.mat3,
                r,
                second,
                &self.mat1,
                &mut self.mat2,
                gemm,
                cache,
            );
            mat_to_svec(y, &self.mat3);
            return;
        }
        // `g` is stored exactly symmetric, so `x·gᵀ` equals the former `x·g`.
        let (g, cache) = if inverse {
            (&self.Ginv, &mut self.ginv_cache)
        } else {
            (&self.G, &mut self.g_cache)
        };
        congruence_sym(
            &mut self.mat3,
            g,
            false,
            &self.mat1,
            &mut self.mat2,
            gemm,
            cache,
        );
        mat_to_svec(y, &self.mat3);
    }
}
