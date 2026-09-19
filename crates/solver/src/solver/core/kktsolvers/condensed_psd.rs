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

/// `a·b` when the exact-arithmetic product is symmetric. Only the upper
/// triangle is evaluated and then mirrored — the same ascending-k accumulation
/// order as `pooled_gemm`, so upper entries stay bitwise identical while the
/// lower half is an exact copy instead of independently rounded products.
fn pooled_gemm_sym<T: FloatT, MATA, MATB>(
    c: &mut Matrix<T>,
    a: &MATA,
    b: &MATB,
    gemm: Option<(&rayon::ThreadPool, usize)>,
) where
    MATA: DenseMatrix<T>,
    MATB: DenseMatrix<T>,
{
    let (m, n, k) = (a.nrows(), b.ncols(), a.ncols());
    debug_assert_eq!(m, n);
    let (ta, tb) = (a.shape().as_blas_char(), b.shape().as_blas_char());
    let lda = if a.shape() == MatrixShape::N { m } else { k };
    let ldb = if b.shape() == MatrixShape::N { k } else { n };
    let (adata, bdata) = (a.data(), b.data());
    let ae = |i: usize, p: usize| -> &T {
        &adata[if ta == b'N' { i + p * lda } else { p + i * lda }]
    };
    let be = |p: usize, j: usize| -> &T {
        &bdata[if tb == b'N' { p + j * ldb } else { j + p * ldb }]
    };
    let column = |j: usize, col: &mut [T]| {
        for i in 0..=j {
            col[i] = T::dot_fma((0..k).map(|p| (ae(i, p), be(p, j))));
        }
    };
    if let Some((pool, tiles)) = gemm.filter(|(p, t)| *t > 1 && p.current_num_threads() > 1) {
        if n > 1 {
            let tile = n.div_ceil(tiles.min(n));
            pool.install(|| {
                c.data_mut()
                    .par_chunks_mut(tile * m)
                    .enumerate()
                    .for_each(|(t, chunk)| {
                        let j0 = t * tile;
                        let j1 = (j0 + tile).min(n);
                        for (jc, j) in (j0..j1).enumerate() {
                            column(j, &mut chunk[jc * m..(jc + 1) * m]);
                        }
                    });
            });
        } else {
            for j in 0..n {
                column(j, &mut c.data_mut()[j * m..(j + 1) * m]);
            }
        }
    } else {
        for j in 0..n {
            column(j, &mut c.data_mut()[j * m..(j + 1) * m]);
        }
    }
    for j in 0..n {
        for i in j + 1..n {
            c[(i, j)] = c[(j, i)];
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
        Self {
            sqrt2: if n > 1 { T::SQRT_2() } else { T::zero() },
            R: Matrix::zeros((n, n)),
            Rinv: Matrix::zeros((n, n)),
            Ginv: Matrix::zeros((n, n)),
            mat1: Matrix::zeros((n, n)),
            mat2: Matrix::zeros((n, n)),
            mat3: Matrix::zeros((n, n)),
            mat2c: Matrix::zeros((64 * n, n)),
            mat3c: Matrix::zeros((64 * n, n)),
            vector: vec![T::zero(); rows_len],
            columns,
            schur_values: Vec::new(),
            sampled: None,
            sparse_column_lanes: Vec::new(),
            dense_indices,
            column_groups,
            dense_representatives,
            dense_column_map,
            dense_vectors: Vec::new(),
            dense_row_first,
            dense_row_offsets: Vec::new(),
            dense_acc: Vec::new(),
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
        self.compute_schur_selected(values, false, &mut store);
    }

    pub(super) fn compute_schur_selected(
        &mut self,
        values: &[T],
        skip_sparse: bool,
        mut store: impl FnMut(usize, usize, usize, T),
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
        // The sparse flags are authoritative and may change between
        // assemblies, so rebuild the dense-column list every pass.
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
        let dense_count = self.dense_indices.len();
        let batched = dense_count > 0
            && T::precision_bits() <= 53
            && (self.vector.len() as u128) * (dense_count as u128) <= (1u128 << 26);
        if batched {
            self.compute_schur_dense_batched(values, skip_sparse, &mut store);
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
    ) {
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            // SAFETY: both required CPU features were checked above.
            return unsafe { self.compute_schur_dense_fma(values, skip_sparse, store) };
        }
        self.compute_schur_dense_impl(values, skip_sparse, store);
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn compute_schur_dense_fma(
        &mut self,
        values: &[T],
        skip_sparse: bool,
        store: impl FnMut(usize, usize, usize, T),
    ) {
        self.compute_schur_dense_impl(values, skip_sparse, store);
    }

    // One arithmetic implementation, inlined into either CPU context. The
    // per-accumulator FMA order is unchanged; there is no fast-math reduction.
    #[inline(always)]
    pub(super) fn compute_schur_dense_impl(
        &mut self,
        values: &[T],
        skip_sparse: bool,
        mut store: impl FnMut(usize, usize, usize, T),
    ) {
        let svec_n = self.vector.len();
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
        self.dense_vectors
            .resize(self.dense_row_offsets[svec_n], T::zero());
        const TILE: usize = 64;
        self.dense_acc.clear();
        self.dense_acc.resize(TILE * width, T::zero());
        const CHUNK: usize = 64;
        let n = self.Ginv.nrows();
        for (c0, chunk) in self.dense_indices.chunks(CHUNK).enumerate() {
            // mat2 = Ginv * A via column AXPYs in GEMM's ascending-p order:
            // bitwise-identical sums at ~e*n flops instead of 2n^3. The W
            // blocks are stacked so the second product runs as one GEMM.
            // mat2c/mat3c are (CHUNK*n) x n column-major: block t's column
            // q lives at data[q*ld + t*n .. +n] with ld = CHUNK*n. Tail rows
            // beyond chunk.len() blocks hold stale data that is never read.
            let ld = CHUNK * n;
            {
                let m2c = self.mat2c.data_mut();
                for (t, &ci) in chunk.iter().enumerate() {
                    coefficient_product(
                        m2c,
                        ld,
                        t * n,
                        &self.Ginv,
                        &self.columns[ci],
                        &self.axpy_plans[ci],
                        values,
                    );
                }
            }
            self.mat3c.mul(&self.mat2c, &self.Ginv, T::one(), T::zero());
            // Pack only used row suffixes; writes remain contiguous across
            // the chunk's live columns. Unused coordinates have empty rows.
            {
                let b3 = self.mat3c.data();
                let d0 = c0 * CHUNK;
                let cw = chunk.len();
                let mut pos = 0;
                for col in 0..n {
                    for row in 0..=col {
                        let base = self.dense_row_offsets[pos];
                        let start = width - (self.dense_row_offsets[pos + 1] - base);
                        if start >= d0 + cw {
                            pos += 1;
                            continue;
                        }
                        let begin = start.saturating_sub(d0);
                        let out = &mut self.dense_vectors
                            [base + d0 + begin - start..base + d0 + cw - start];
                        if row == col {
                            for (offset, o) in out.iter_mut().enumerate() {
                                let t = begin + offset;
                                *o = b3[col * ld + t * n + row];
                            }
                        } else {
                            for (offset, o) in out.iter_mut().enumerate() {
                                let t = begin + offset;
                                *o = (b3[col * ld + t * n + row] + b3[row * ld + t * n + col])
                                    * T::FRAC_1_SQRT_2();
                            }
                        }
                        pos += 1;
                    }
                }
            }
        }
        // Tile left columns so the store pass reads each column's
        // schur_positions once per tile instead of once per (a, d) pair.
        // Row t of the tile buffer accumulates left column a0 + t over the
        // same entries in the same FMA order as the untiled loop.
        let column_count = self.columns.len();
        let mut a0 = 0;
        while a0 < column_count {
            let a1 = (a0 + TILE).min(column_count);
            let mut exhausted = false;
            for a in a0..a1 {
                let t = a - a0;
                let left = &self.columns[a];
                let dmin = self.dense_indices.partition_point(|&c| c < a);
                if dmin == width {
                    // dmin is nondecreasing in a, so no later column
                    // produces a store either.
                    exhausted = true;
                    break;
                }
                self.dense_acc[t * width + dmin..(t + 1) * width]
                    .iter_mut()
                    .for_each(|x| *x = T::zero());
                // Four independent source streams share one accumulator
                // load/store, retaining the entry-wise FMA order exactly.
                let mut wide_groups = left.entries.chunks_exact(8);
                for group in &mut wide_groups {
                    let rows: [&[T]; 8] = std::array::from_fn(|k| {
                        let e = &group[k];
                        let pos = triangular_number(e.j) + e.i;
                        &self.dense_vectors[self.dense_row_offsets[pos + 1] - (width - dmin)
                            ..self.dense_row_offsets[pos + 1]]
                    });
                    let v: [T; 8] = std::array::from_fn(|k| values[group[k].position]);
                    let acc = &mut self.dense_acc[t * width + dmin..(t + 1) * width];
                    for (i, x) in acc.iter_mut().enumerate() {
                        let mut value = *x;
                        for k in 0..8 {
                            value = v[k].mul_add(rows[k][i], value);
                        }
                        *x = value;
                    }
                }
                let mut groups = wide_groups.remainder().chunks_exact(4);
                for group in &mut groups {
                    let rows: [&[T]; 4] = std::array::from_fn(|k| {
                        let e = &group[k];
                        let pos = triangular_number(e.j) + e.i;
                        &self.dense_vectors[self.dense_row_offsets[pos + 1] - (width - dmin)
                            ..self.dense_row_offsets[pos + 1]]
                    });
                    let v: [T; 4] = std::array::from_fn(|k| values[group[k].position]);
                    let acc = &mut self.dense_acc[t * width + dmin..(t + 1) * width];
                    for (i, x) in acc.iter_mut().enumerate() {
                        let x0 = v[0].mul_add(rows[0][i], *x);
                        let x1 = v[1].mul_add(rows[1][i], x0);
                        let x2 = v[2].mul_add(rows[2][i], x1);
                        *x = v[3].mul_add(rows[3][i], x2);
                    }
                }
                for e in groups.remainder() {
                    let v = values[e.position];
                    let pos = triangular_number(e.j) + e.i;
                    let row = &self.dense_vectors[self.dense_row_offsets[pos + 1] - (width - dmin)
                        ..self.dense_row_offsets[pos + 1]];
                    let acc = &mut self.dense_acc[t * width + dmin..(t + 1) * width];
                    for (i, x) in acc.iter_mut().enumerate() {
                        *x = v.mul_add(row[i], *x);
                    }
                }
            }
            // An alias b uses a representative r >= b. Every required
            // (a, b) therefore has a computed (a, r); publish only a <= b.
            for b in a0..column_count {
                if self.columns[b].sparse {
                    continue;
                }
                let d = self.dense_column_map[b];
                let positions = &self.columns[b].schur_positions;
                let tmax = (b + 1 - a0).min(a1 - a0);
                for t in 0..tmax {
                    let a = a0 + t;
                    store(b, a, positions[a], self.dense_acc[t * width + d]);
                }
            }
            if exhausted {
                break;
            }
            a0 = a1;
        }
        for (b, right) in self.columns.iter().enumerate() {
            if !right.sparse || skip_sparse {
                continue;
            }
            for (a, left) in self.columns[..=b].iter().enumerate() {
                let v = sparse_schur_value(left, right, &self.Ginv, values, self.sqrt2);
                store(b, a, right.schur_positions[a], v);
            }
        }
    }

    pub(super) fn scatter_schur(&self, S: &mut CscMatrix<T>) {
        for (b, column) in self.columns.iter().enumerate() {
            for (a, &position) in column.schur_positions.iter().enumerate() {
                S.nzval[position] += self.schur_values[triangular_number(b) + a];
            }
        }
    }

    pub(super) fn apply(
        &mut self,
        y: &mut [T],
        x: &[T],
        inverse: bool,
        gemm: Option<(&rayon::ThreadPool, usize)>,
    ) {
        svec_to_mat(&mut self.mat1, x);
        if inverse {
            // H^-1 = W^-1 W^-T; retain the inverse factors in solve-time
            // applications rather than squaring them into Ginv.
            pooled_gemm(&mut self.mat2, &self.mat1, &self.Rinv.t(), gemm);
            pooled_gemm_sym(&mut self.mat3, &self.Rinv, &self.mat2, gemm);
            mat_to_svec(&mut self.vector, &self.mat3);
            svec_to_mat(&mut self.mat1, &self.vector);
            pooled_gemm(&mut self.mat2, &self.Rinv.t(), &self.mat1, gemm);
            pooled_gemm_sym(&mut self.mat3, &self.mat2, &self.Rinv, gemm);
        } else {
            pooled_gemm(&mut self.mat2, &self.R.t(), &self.mat1, gemm);
            pooled_gemm_sym(&mut self.mat3, &self.mat2, &self.R, gemm);
            mat_to_svec(&mut self.vector, &self.mat3);
            svec_to_mat(&mut self.mat1, &self.vector);
            pooled_gemm(&mut self.mat2, &self.mat1, &self.R.t(), gemm);
            pooled_gemm_sym(&mut self.mat3, &self.R, &self.mat2, gemm);
        }
        mat_to_svec(y, &self.mat3);
    }
}
