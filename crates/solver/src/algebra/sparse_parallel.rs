//! Deterministic sparse residual products in the solver's existing worker pool.
use crate::algebra::*;
use std::sync::Arc;

const MIN_PARALLEL_WORK: u128 = 32768;
/// Entries from which a row of a wide exact forward product is read column
/// by column (`forward_long`). Apple M4, 99 rows at 256 bits, 1–8 threads:
/// rows of 4096 entries and more run 1.04–2.6× faster; rows of 1024–2048,
/// still cache-resident row by row, run up to 1.4× slower.
const LONG_ROW_ENTRIES: usize = 4096;
/// Column chunks per lane of `forward_long`, so faster cores take more of
/// them (M4 at 8–10 threads: −9…−10%; 2–4 threads unchanged).
const LONG_CHUNKS_PER_LANE: usize = 4;

pub(crate) fn worthwhile<T: FloatT>(a: &CscMatrix<T>) -> bool {
    let words = T::precision_bits().div_ceil(64) as u128;
    (a.m > 1 || a.n > 1) && a.nnz() as u128 * words * words >= MIN_PARALLEL_WORK
}

#[derive(Clone, Copy)]
struct Entry {
    column: usize,
    position: usize,
}

pub(crate) struct SparseParallel {
    rowptr: Vec<usize>,
    entries: Vec<Entry>,
    row_lanes: Vec<usize>,
    column_lanes: Vec<usize>,
    // Rows holding entries, when some rows hold none, with their prefix
    // entry counts and lanes (`forward_active`).
    active: Vec<usize>,
    active_ptr: Vec<usize>,
    active_lanes: Vec<usize>,
    // Long rows of a wide exact forward product, with each row's index among
    // them (usize::MAX elsewhere) and column lanes balancing their entries.
    // A row's entries lie one per CSC column across the whole matrix, so a
    // row-by-row pass strides through memory (fix-a 256: 22 ns per term,
    // against 9.5 for the contiguous adjoint); `forward_long` reads them
    // column by column into per-row exact accumulators instead.
    long_rows: Vec<usize>,
    long_index: Vec<usize>,
    long_lanes: Vec<usize>,
    // Exponent range (`product_exponent`) of each long row's nonzero entries
    // when planned, so `forward_long` bounds its exact sums without a scan of
    // the terms; empty when some entry has none. Values changed since only
    // cost that scan again.
    long_exponents: Vec<(i64, i64)>,
    symmetric: bool,
    workers: usize,
    pool: Option<Arc<rayon::ThreadPool>>,
}

impl SparseParallel {
    #[cfg(test)]
    pub(crate) fn test_pool_and_storage(&self) -> (usize, usize) {
        (
            self.pool.as_ref().map_or(1, |p| p.current_num_threads()),
            self.entries.as_ptr() as usize,
        )
    }

    pub(crate) fn new<T>(a: &CscMatrix<T>) -> Self {
        Self::build(a, false)
    }

    pub(crate) fn new_symmetric<T>(a: &CscMatrix<T>) -> Self {
        assert_eq!(a.n, a.m);
        Self::build(a, true)
    }

    fn build<T>(a: &CscMatrix<T>, symmetric: bool) -> Self {
        let mut rowptr = vec![0; a.m + 1];
        for &row in &a.rowval {
            rowptr[row + 1] += 1;
        }
        if symmetric {
            for col in 0..a.n {
                for p in a.colptr[col]..a.colptr[col + 1] {
                    if a.rowval[p] != col {
                        rowptr[col + 1] += 1;
                    }
                }
            }
        }
        for i in 0..a.m {
            rowptr[i + 1] += rowptr[i];
        }
        let mut entries = vec![
            Entry {
                column: 0,
                position: 0
            };
            rowptr[a.m]
        ];
        // Scanning CSC in exactly its original order preserves the order of
        // every row's additions, including explicitly stored zeros.
        for column in 0..a.n {
            for position in a.colptr[column]..a.colptr[column + 1] {
                let row = a.rowval[position];
                entries[rowptr[row]] = Entry { column, position };
                rowptr[row] += 1;
                if symmetric && row != column {
                    entries[rowptr[column]] = Entry {
                        column: row,
                        position,
                    };
                    rowptr[column] += 1;
                }
            }
        }
        // Filled cursors are row ends; shift them back into row starts.
        rowptr.copy_within(..a.m, 1);
        rowptr[0] = 0;
        let mut active = Vec::new();
        let mut active_ptr = vec![0];
        if !symmetric {
            for r in 0..a.m {
                let len = rowptr[r + 1] - rowptr[r];
                if len > 0 {
                    active.push(r);
                    active_ptr.push(active_ptr.last().unwrap() + len);
                }
            }
            if active.len() == a.m {
                // Every row holds entries: the ordinary row lanes apply.
                active = Vec::new();
                active_ptr = vec![0];
            }
        }
        Self {
            rowptr,
            entries,
            row_lanes: Vec::new(),
            column_lanes: Vec::new(),
            active,
            active_ptr,
            active_lanes: Vec::new(),
            long_rows: Vec::new(),
            long_index: Vec::new(),
            long_lanes: Vec::new(),
            long_exponents: Vec::new(),
            symmetric,
            workers: 0,
            pool: None,
        }
    }

    pub(crate) fn configure<T: FloatT>(
        &mut self,
        a: &CscMatrix<T>,
        pool: Option<Arc<rayon::ThreadPool>>,
    ) {
        let workers = pool.as_ref().map_or(1, |p| p.current_num_threads());
        self.pool = pool;
        if self.workers != workers {
            self.workers = workers;
            // Every lane carries at least one grain of multiply-adds; extra
            // lanes on a small operator only add fork-join latency.
            let grain = sdpx_arithmetic::inner_parallel::grain().max(1);
            let work = self.entries.len() as u128
                * sdpx_arithmetic::inner_parallel::weight(T::precision_bits());
            let lanes = workers.min((work / grain).max(1).min(usize::MAX as u128) as usize);
            self.row_lanes = partitions(&self.rowptr, lanes);
            self.column_lanes = partitions(&a.colptr, lanes);
            self.active_lanes = if self.active.is_empty() {
                Vec::new()
            } else {
                partitions(&self.active_ptr, lanes)
            };
            self.plan_long_rows(a, lanes);
        }
    }

    /// Rows with at least [`LONG_ROW_ENTRIES`] entries, when the forward
    /// product is exact and they hold at least half of its entries (the
    /// column pass reads every entry to find theirs).
    fn plan_long_rows<T: FloatT>(&mut self, a: &CscMatrix<T>, lanes: usize) {
        self.long_rows.clear();
        self.long_index = Vec::new();
        self.long_lanes = Vec::new();
        self.long_exponents = Vec::new();
        let exact = T::precision_bits() > 64 && a.nnz() >= 4 * a.m;
        if !exact || self.symmetric {
            return;
        }
        let length = |r: usize| self.rowptr[r + 1] - self.rowptr[r];
        let rows: Vec<usize> = (0..a.m)
            .filter(|&r| length(r) >= LONG_ROW_ENTRIES)
            .collect();
        if 2 * rows.iter().map(|&r| length(r)).sum::<usize>() < a.nnz() {
            return;
        }
        let mut index = vec![usize::MAX; a.m];
        for (k, &r) in rows.iter().enumerate() {
            index[r] = k;
        }
        let mut ptr = Vec::with_capacity(a.n + 1);
        ptr.push(0);
        for j in 0..a.n {
            let long = a.rowval[a.colptr[j]..a.colptr[j + 1]]
                .iter()
                .filter(|&&r| index[r] != usize::MAX)
                .count();
            ptr.push(ptr[j] + long);
        }
        self.long_lanes = partitions(&ptr, LONG_CHUNKS_PER_LANE * lanes.max(1));
        let mut exponents = vec![(i64::MAX, i64::MIN); rows.len()];
        let known = a.rowval.iter().zip(&a.nzval).all(|(&r, v)| {
            let k = index[r];
            if k == usize::MAX || v.is_zero() {
                return true;
            }
            v.product_exponent()
                .map(|e| {
                    let range = &mut exponents[k];
                    *range = (range.0.min(e), range.1.max(e));
                })
                .is_some()
        });
        self.long_exponents = if known { exponents } else { Vec::new() };
        self.long_rows = rows;
        self.long_index = index;
    }

    /// Ranges holding the exponent sums of every finite nonzero term of
    /// `forward_long`'s outputs: the planned entry ranges plus x's, widened
    /// by each output's head. `None` when an x or an entry has no exponent.
    fn long_bounds<T: FloatT>(&self, x: &[T], heads: &[T], one: &T) -> Option<Vec<(i64, i64)>> {
        if self.long_exponents.len() != self.long_rows.len() {
            return None;
        }
        let (mut lo, mut hi) = (i64::MAX, i64::MIN);
        for v in x {
            match v.product_exponent() {
                Some(e) => (lo, hi) = (lo.min(e), hi.max(e)),
                None if v.is_zero() => {}
                None => return None,
            }
        }
        let unit = one.product_exponent()?;
        let bounds = self
            .long_exponents
            .iter()
            .enumerate()
            .map(|(k, &(a0, a1))| {
                let mut range = if a0 <= a1 && lo <= hi {
                    (a0 + lo, a1 + hi)
                } else {
                    (i64::MAX, i64::MIN)
                };
                if let Some(e) = heads.get(k).and_then(|h| h.product_exponent()) {
                    range = (range.0.min(e + unit), range.1.max(e + unit));
                }
                range
            });
        Some(bounds.collect())
    }

    /// Whether a forward product with this `alpha` takes `forward_long`.
    fn streams_long<T: FloatT>(&self, alpha: T) -> bool {
        !self.long_rows.is_empty() && alpha != T::zero()
    }

    fn is_long(&self, row: usize) -> bool {
        self.long_index.get(row).is_some_and(|&k| k != usize::MAX)
    }

    /// The long rows of `y = alpha·A·x + beta·y`, read column by column on
    /// the long lanes into per-row exact accumulators. Each output is the
    /// exact sum `output` forms (with the incoming y for alpha = ±1),
    /// rounded once; a row whose exact sum declines runs `output` itself.
    fn forward_long<T: FloatT>(&self, a: &CscMatrix<T>, y: &mut [T], x: &[T], alpha: T, beta: T) {
        use rayon::prelude::*;
        let one = T::one();
        let signed = alpha == one || alpha == -one;
        let base: Vec<T> = self
            .long_rows
            .iter()
            .map(|&r| {
                let mut v = y[r];
                scale_output(&mut v, beta);
                v
            })
            .collect();
        let heads: Vec<T> = if alpha == -one {
            base.iter().map(|&v| -v).collect()
        } else {
            base.clone()
        };
        let lanes = &self.long_lanes;
        let terms = |c: usize| {
            let (j0, j1) = (lanes[c], lanes.get(c + 1).copied().unwrap_or(a.n));
            LongTerms {
                a,
                x,
                index: &self.long_index,
                heads: if c == 0 && signed { &heads } else { &[] },
                one: &one,
                head: 0,
                column: j0,
                end_column: j1,
                position: a.colptr[j0],
            }
        };
        let par_for = |n: usize, f: &(dyn Fn(usize) + Sync)| (0..n).into_par_iter().for_each(f);
        let bounds = self.long_bounds(x, if signed { &heads[..] } else { &[] }, &one);
        let sums = T::dot_outputs_chunked(
            self.long_rows.len(),
            lanes.len(),
            bounds.as_deref(),
            &terms,
            &par_for,
        );
        for ((k, &r), sum) in self.long_rows.iter().enumerate().zip(sums) {
            match sum {
                Some(s) if alpha == one => y[r] = s,
                Some(s) if alpha == -one => y[r] = -s,
                Some(s) => y[r] = base[k] + alpha * s,
                None => self.output(a, false, r, &mut y[r], x, alpha, beta),
            }
        }
    }

    pub(crate) fn symv<T: FloatT>(
        &self,
        a: &CscMatrix<T>,
        uplo: MatrixTriangle,
        y: &mut [T],
        x: &[T],
        alpha: T,
        beta: T,
    ) {
        if let Some(pool) = &self.pool {
            if self.row_lanes.len() > 1 {
                y.scale(beta);
                pool.install(|| {
                    split_outputs(y, &self.row_lanes, &|row, out| {
                        for e in &self.entries[self.rowptr[row]..self.rowptr[row + 1]] {
                            *out += alpha * a.nzval[e.position] * x[e.column];
                        }
                    })
                });
                return;
            }
        }
        a.sym(uplo).symv(y, x, alpha, beta);
    }

    /// Two independent `yᵢ = β·yᵢ + α·K·xᵢ` products sharing one pass over
    /// the row plan; each output keeps the single-product entry order.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn symv_pair<T: FloatT>(
        &self,
        a: &CscMatrix<T>,
        uplo: MatrixTriangle,
        y: [&mut [T]; 2],
        x: [&[T]; 2],
        alpha: T,
        beta: T,
    ) {
        let [y0, y1] = y;
        if let Some(pool) = &self.pool {
            if self.row_lanes.len() > 1 {
                y0.scale(beta);
                y1.scale(beta);
                pool.install(|| {
                    split_output_pair(y0, y1, &self.row_lanes, &|row, o0, o1| {
                        for e in &self.entries[self.rowptr[row]..self.rowptr[row + 1]] {
                            let v = a.nzval[e.position];
                            *o0 += alpha * v * x[0][e.column];
                            *o1 += alpha * v * x[1][e.column];
                        }
                    })
                });
                return;
            }
        }
        a.sym(uplo).symv(y0, x[0], alpha, beta);
        a.sym(uplo).symv(y1, x[1], alpha, beta);
    }

    pub(crate) fn residual_products<T: FloatT>(
        &self,
        a: &CscMatrix<T>,
        x: &[T],
        z: &[T],
        rx: &mut [T],
        rz: &mut [T],
    ) {
        self.residual_products_impl(a, x, z, rx, rz, crate::mpi::World::get());
    }

    /// Evaluate both ordinary residual products using only this plan's local
    /// worker pool.  Owned rank-local solvers must use this route: consulting
    /// the implicit MPI world from a worker would shard an already-local block
    /// a second time and can also introduce a collective into the owner pool.
    pub(crate) fn residual_products_local<T: FloatT>(
        &self,
        a: &CscMatrix<T>,
        x: &[T],
        z: &[T],
        rx: &mut [T],
        rz: &mut [T],
    ) {
        self.residual_products_impl(a, x, z, rx, rz, None);
    }

    fn residual_products_impl<T: FloatT>(
        &self,
        a: &CscMatrix<T>,
        x: &[T],
        z: &[T],
        rx: &mut [T],
        rz: &mut [T],
        world: Option<crate::mpi::World>,
    ) {
        if let Some(world) = world {
            self.product_sharded(
                a,
                true,
                rx,
                z,
                -T::one(),
                T::zero(),
                world,
                crate::mpi::SITE_RX,
            );
            self.product_sharded(
                a,
                false,
                rz,
                x,
                T::one(),
                T::one(),
                world,
                crate::mpi::SITE_RZ,
            );
            return;
        }
        // Pooled and serial products share the CSC gemv arithmetic, so every
        // thread count produces the same residual bits.
        if let Some(pool) = &self.pool {
            // One entry into the pool for both products. Phases are joined;
            // no outer tasks compete with a nested full-budget product.
            pool.install(|| {
                self.apply_in_pool(a, true, rx, z, -T::one(), T::zero());
                self.apply_in_pool(a, false, rz, x, T::one(), T::one());
            });
        } else {
            a.t().gemv(rx, z, -T::one(), T::zero());
            a.gemv(rz, x, T::one(), T::one());
        }
    }

    /// True when the cached plan owns more than one output lane.
    pub(crate) fn has_lanes(&self) -> bool {
        self.row_lanes.len() > 1 || self.column_lanes.len() > 1
    }

    /// `y = alpha * op(A) * x + beta * y` on disjoint output lanes. Reuses the
    /// CSC gemv scalar branches, so results match the serial product exactly.
    pub(crate) fn product<T: FloatT>(
        &self,
        a: &CscMatrix<T>,
        transpose: bool,
        y: &mut [T],
        x: &[T],
        alpha: T,
        beta: T,
    ) {
        self.apply_in_pool(a, transpose, y, x, alpha, beta);
    }

    /// [`Self::product`] entered through this plan's own pool (serial lanes
    /// without one), for callers outside it.
    pub(crate) fn product_local<T: FloatT>(
        &self,
        a: &CscMatrix<T>,
        transpose: bool,
        y: &mut [T],
        x: &[T],
        alpha: T,
        beta: T,
    ) {
        match &self.pool {
            Some(pool) => pool.install(|| self.apply_in_pool(a, transpose, y, x, alpha, beta)),
            None => self.apply_in_pool(a, transpose, y, x, alpha, beta),
        }
    }

    /// Forward `y = alpha·A·x + beta·y` on the sorted `rows` only, through
    /// this plan's pool; every other output is left unchanged. Each listed
    /// output equals the full product's.
    pub(crate) fn forward_rows_local<T: FloatT>(
        &self,
        a: &CscMatrix<T>,
        rows: &[usize],
        y: &mut [T],
        x: &[T],
        alpha: T,
        beta: T,
    ) {
        if rows.is_empty() {
            return;
        }
        let mut ptr = Vec::with_capacity(rows.len() + 1);
        ptr.push(0);
        for &r in rows {
            ptr.push(ptr.last().unwrap() + self.rowptr[r + 1] - self.rowptr[r]);
        }
        let lanes = partitions(&ptr, self.workers.max(1));
        let mut values: Vec<T> = rows.iter().map(|&r| y[r]).collect();
        let mut run = || {
            split_outputs(&mut values, &lanes, &|k, value| {
                self.output(a, false, rows[k], value, x, alpha, beta)
            })
        };
        match &self.pool {
            Some(pool) => pool.install(run),
            None => run(),
        }
        for (&r, value) in rows.iter().zip(values) {
            y[r] = value;
        }
    }

    /// Rank-sharded product. Every output is evaluated on exactly one rank in
    /// its original accumulation order; the gathered outputs reproduce the
    /// serial product bitwise.
    pub(crate) fn product_sharded<T: FloatT>(
        &self,
        a: &CscMatrix<T>,
        transpose: bool,
        y: &mut [T],
        x: &[T],
        alpha: T,
        beta: T,
        world: crate::mpi::World,
        site: usize,
    ) {
        let count = if transpose { a.n } else { a.m };
        let rank_lanes = partitions(
            if transpose { &a.colptr } else { &self.rowptr },
            world.size(),
        );
        // Lane starts with a trailing sentinel turn into per-rank ranges.
        let gather_ranges: Vec<(usize, usize)> = (0..world.size())
            .map(|r| {
                let begin = rank_lanes.get(r).copied().unwrap_or(count);
                let end = rank_lanes.get(r + 1).copied().unwrap_or(count);
                (begin, end - begin)
            })
            .collect();
        let (o0, len) = gather_ranges[world.rank()];
        let timer = crate::receipt::start();
        // The local segment inherits y's current values so `beta` applies to
        // the same base as the serial product.
        let mut local = y[o0..o0 + len].to_vec();
        let ptr = if transpose { &a.colptr } else { &self.rowptr };
        let compute =
            |output: usize, value: &mut T| self.output(a, transpose, output, value, x, alpha, beta);
        // Two-level parallelism: the rank's output span also splits across
        // the thread pool; every output keeps its original order either way.
        let workers = self.pool.as_ref().map_or(1, |p| p.current_num_threads());
        if workers > 1 && len > 0 {
            let lanes = partitions(&ptr[o0..o0 + len + 1], workers.min(len));
            if let Some(pool) = &self.pool {
                let base = o0;
                pool.install(|| {
                    split_outputs(&mut local, &lanes, &|offset, value| {
                        compute(base + offset, value)
                    })
                });
            }
        } else {
            for (i, value) in local.iter_mut().enumerate() {
                compute(o0 + i, value);
            }
        }
        crate::receipt::finish("sharded.compute", timer);
        y.fill(T::zero());
        world.gather_slice(site, &local, &gather_ranges, y);
    }

    /// Rows of `a` holding entries, for [`Self::forward_sharded_active`].
    pub(crate) fn active_rows(&self) -> Vec<usize> {
        (0..self.rowptr.len() - 1)
            .filter(|&r| self.rowptr[r + 1] > self.rowptr[r])
            .collect()
    }

    /// Rank-sharded forward product exchanging only the rows that hold
    /// entries (`active`, from [`Self::active_rows`]);
    /// every other output takes the `beta` scaling alone, as gemv does.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_sharded_active<T: FloatT>(
        &self,
        a: &CscMatrix<T>,
        y: &mut [T],
        x: &[T],
        alpha: T,
        beta: T,
        world: crate::mpi::World,
        site: usize,
        active: &[usize],
    ) {
        // Deal rows round-robin before the contiguous split: exact-dot cost
        // varies with row magnitudes, which correlate with row order.
        let size = world.size();
        let order: Vec<usize> = (0..size)
            .flat_map(|r| (r..active.len()).step_by(size))
            .map(|k| active[k])
            .collect();
        let active = order.as_slice();
        let mut ptr = Vec::with_capacity(active.len() + 1);
        ptr.push(0);
        for &r in active {
            ptr.push(ptr.last().unwrap() + self.rowptr[r + 1] - self.rowptr[r] + 1);
        }
        let count = active.len();
        let lanes = partitions(&ptr, size);
        let ranges: Vec<(usize, usize)> = (0..world.size())
            .map(|r| {
                let begin = lanes.get(r).copied().unwrap_or(count);
                let end = lanes.get(r + 1).copied().unwrap_or(count);
                (begin, end - begin)
            })
            .collect();
        let (o0, len) = ranges[world.rank()];
        let timer = crate::receipt::start();
        let mut local: Vec<T> = active[o0..o0 + len].iter().map(|&r| y[r]).collect();
        let compute =
            |k: usize, value: &mut T| self.output(a, false, active[o0 + k], value, x, alpha, beta);
        match &self.pool {
            // Active rows are few and long (one per equality): one task each.
            Some(pool) if pool.current_num_threads() > 1 && len > 1 => {
                use rayon::prelude::*;
                pool.install(|| {
                    local
                        .par_iter_mut()
                        .enumerate()
                        .for_each(|(k, v)| compute(k, v))
                });
            }
            _ => local
                .iter_mut()
                .enumerate()
                .for_each(|(k, v)| compute(k, v)),
        }
        crate::receipt::finish("sharded.active_compute", timer);
        let timer = crate::receipt::start();
        for value in y.iter_mut() {
            scale_output(value, beta);
        }
        crate::receipt::finish("sharded.scale_rest", timer);
        let mut all = vec![T::zero(); count];
        world.gather_slice(site, &local, &ranges, &mut all);
        for (&r, v) in active.iter().zip(all) {
            y[r] = v;
        }
    }

    fn apply_in_pool<T: FloatT>(
        &self,
        a: &CscMatrix<T>,
        transpose: bool,
        y: &mut [T],
        x: &[T],
        alpha: T,
        beta: T,
    ) {
        if !transpose && beta == T::one() && self.active_lanes.len() > 1 {
            self.forward_active(a, y, x, alpha, beta);
            return;
        }
        if transpose
            && beta == T::one()
            && !self.active.is_empty()
            && self.active.iter().all(|&r| x[r].is_zero())
        {
            self.adjoint_of_zero(a, y, x, alpha, beta);
            return;
        }
        let lanes = if transpose {
            &self.column_lanes
        } else {
            &self.row_lanes
        };
        let long = !transpose && self.streams_long(alpha);
        if lanes.len() <= 1 && !long {
            if transpose {
                a.t().gemv(y, x, alpha, beta);
            } else {
                a.gemv(y, x, alpha, beta);
            }
            return;
        }
        if long {
            self.forward_long(a, y, x, alpha, beta);
        }
        split_outputs(y, lanes, &|output, y| {
            if !(long && self.is_long(output)) {
                self.output(a, transpose, output, y, x, alpha, beta)
            }
        });
    }

    /// Forward product with `beta = 1` when some rows hold no entries: the
    /// rows holding entries run on lanes balanced by their entry counts, and
    /// an empty row is evaluated only where the evaluation can change it.
    /// With `beta = 1` an empty row's output is `y` itself, except in the
    /// exact wide branch, which rounds `y` alone: a zero takes the sign that
    /// branch gives it, and a non-finite value goes through the chain. Those
    /// rows still run `output`, so every output equals the full product's.
    /// (Λ27's sampled linear part: 104 of 67431 rows hold all entries, and
    /// evaluating every empty row through the exact branch set the wall.)
    fn forward_active<T: FloatT>(&self, a: &CscMatrix<T>, y: &mut [T], x: &[T], alpha: T, beta: T) {
        use rayon::prelude::*;
        debug_assert!(beta == T::one());
        let long = self.streams_long(alpha);
        if long {
            self.forward_long(a, y, x, alpha, beta);
        }
        let mut values: Vec<T> = self.active.iter().map(|&r| y[r]).collect();
        split_outputs(&mut values, &self.active_lanes, &|k, value| {
            let row = self.active[k];
            if !(long && self.is_long(row)) {
                self.output(a, false, row, value, x, alpha, beta)
            }
        });
        for (&r, value) in self.active.iter().zip(values) {
            y[r] = value;
        }
        let exact = T::precision_bits() > 64 && a.nnz() >= 4 * a.m;
        if !exact || alpha == T::zero() {
            return;
        }
        const CHUNK: usize = 4096;
        let rowptr = &self.rowptr;
        y.par_chunks_mut(CHUNK).enumerate().for_each(|(c, chunk)| {
            for (i, value) in chunk.iter_mut().enumerate() {
                let r = c * CHUNK + i;
                if rowptr[r + 1] == rowptr[r] && (value.is_zero() || !value.is_finite()) {
                    self.output(a, false, r, value, x, alpha, beta);
                }
            }
        });
    }

    /// Adjoint product with `beta = 1` of an `x` that is zero on every row
    /// holding entries (the fused prepare's `L'·0`): every product is zero,
    /// so an output whose `y` is finite and nonzero is `y` itself in each
    /// branch (exact or rounded). Zero and non-finite outputs still run
    /// `output`, which fixes the sign of a zero sum as the full product does.
    fn adjoint_of_zero<T: FloatT>(
        &self,
        a: &CscMatrix<T>,
        y: &mut [T],
        x: &[T],
        alpha: T,
        beta: T,
    ) {
        use rayon::prelude::*;
        debug_assert!(beta == T::one());
        const CHUNK: usize = 4096;
        y.par_chunks_mut(CHUNK).enumerate().for_each(|(c, chunk)| {
            for (i, value) in chunk.iter_mut().enumerate() {
                if value.is_zero() || !value.is_finite() {
                    self.output(a, true, c * CHUNK + i, value, x, alpha, beta);
                }
            }
        });
    }

    /// One output of `y = alpha·op(A)·x + beta·y` with exactly the CSC gemv
    /// arithmetic, so serial, pooled and rank-sharded products agree bitwise.
    /// Wide precision: a column with at least four entries (and, when
    /// nnz ≥ 4m, every row) is one exact accumulation including the incoming
    /// y for alpha = ±1, rounded once; other outputs use the rounded chain.
    #[allow(clippy::too_many_arguments)]
    fn output<T: FloatT>(
        &self,
        a: &CscMatrix<T>,
        transpose: bool,
        output: usize,
        y: &mut T,
        x: &[T],
        alpha: T,
        beta: T,
    ) {
        scale_output(y, beta);
        if alpha == T::zero() {
            return;
        }
        let wide = T::precision_bits() > 64;
        if transpose {
            let (s, e) = (a.colptr[output], a.colptr[output + 1]);
            let terms = (s..e).map(|p| (&a.nzval[p], &x[a.rowval[p]]));
            if wide && e - s >= 4 {
                exact_output(y, terms, alpha);
            } else {
                terms.for_each(|(&v, &xv)| accumulate(y, v, xv, alpha));
            }
        } else {
            let row = &self.entries[self.rowptr[output]..self.rowptr[output + 1]];
            let terms = row.iter().map(|e| (&a.nzval[e.position], &x[e.column]));
            if wide && a.nnz() >= 4 * a.m {
                exact_output(y, terms, alpha);
            } else {
                terms.for_each(|(&v, &xv)| accumulate(y, v, xv, alpha));
            }
        }
    }
}

fn exact_output<'a, T: FloatT + 'a>(
    y: &mut T,
    terms: impl Iterator<Item = (&'a T, &'a T)>,
    alpha: T,
) {
    *y = crate::algebra::csc::wide_output(*y, terms, alpha, T::one());
}

// Preserve CSC gemv's scalar branches, including zero-beta clearing NaNs and
// signed alpha=-1 subtraction. Do not turn the general product into an FMA.
fn scale_output<T: FloatT>(y: &mut T, beta: T) {
    if beta == T::zero() {
        *y = T::zero();
    } else if beta == -T::one() {
        *y = -*y;
    } else if beta != T::one() {
        *y *= beta;
    }
}
fn accumulate<T: FloatT>(y: &mut T, a: T, x: T, alpha: T) {
    if alpha == T::one() {
        *y += a * x;
    } else if alpha == -T::one() {
        *y -= a * x;
    } else {
        *y += alpha * a * x;
    }
}

// Prefix nonzero counts plus one unit per output also distribute empty rows.
// Plans change only with worker count; matrix-value updates keep all indices.
fn partitions(ptr: &[usize], workers: usize) -> Vec<usize> {
    let count = ptr.len() - 1;
    let workers = workers.min(count);
    let mut lanes = Vec::with_capacity(workers);
    let mut begin = 0;
    for lane in 0..workers {
        lanes.push(begin);
        let remaining = workers - lane;
        if remaining == 1 {
            break;
        }
        let cost = |end: usize| (ptr[end] - ptr[begin]) as u128 + (end - begin) as u128;
        let target = cost(count) / remaining as u128;
        let last = count - (remaining - 1);
        let mut end = begin + 1;
        while end < last && cost(end) < target {
            end += 1;
        }
        if end > begin + 1 && target.abs_diff(cost(end - 1)) <= target.abs_diff(cost(end)) {
            end -= 1;
        }
        begin = end;
    }
    lanes
}

fn split_outputs<T: FloatT, F: Fn(usize, &mut T) + Sync>(y: &mut [T], lanes: &[usize], f: &F) {
    if lanes.len() == 1 {
        for (i, value) in y.iter_mut().enumerate() {
            f(lanes[0] + i, value);
        }
    } else {
        let mid = lanes.len() / 2;
        let (left, right) = y.split_at_mut(lanes[mid] - lanes[0]);
        rayon::join(
            || split_outputs(left, &lanes[..mid], f),
            || split_outputs(right, &lanes[mid..], f),
        );
    }
}

fn split_output_pair<T: FloatT, F: Fn(usize, &mut T, &mut T) + Sync>(
    y0: &mut [T],
    y1: &mut [T],
    lanes: &[usize],
    f: &F,
) {
    if lanes.len() == 1 {
        for (i, (a, b)) in y0.iter_mut().zip(y1.iter_mut()).enumerate() {
            f(lanes[0] + i, a, b);
        }
    } else {
        let mid = lanes.len() / 2;
        let cut = lanes[mid] - lanes[0];
        let (l0, r0) = y0.split_at_mut(cut);
        let (l1, r1) = y1.split_at_mut(cut);
        rayon::join(
            || split_output_pair(l0, l1, &lanes[..mid], f),
            || split_output_pair(r0, r1, &lanes[mid..], f),
        );
    }
}

/// Terms of the long rows in columns `column..end_column`, in CSC order, as
/// `(long row index, a, x)`; chunk 0 leads with the incoming heads.
struct LongTerms<'a, T> {
    a: &'a CscMatrix<T>,
    x: &'a [T],
    index: &'a [usize],
    heads: &'a [T],
    one: &'a T,
    head: usize,
    column: usize,
    end_column: usize,
    position: usize,
}

impl<'a, T> Iterator for LongTerms<'a, T> {
    type Item = (usize, &'a T, &'a T);

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.head < self.heads.len() {
            self.head += 1;
            return Some((self.head - 1, &self.heads[self.head - 1], self.one));
        }
        while self.column < self.end_column {
            let end = self.a.colptr[self.column + 1];
            while self.position < end {
                let p = self.position;
                self.position += 1;
                let k = self.index[self.a.rowval[p]];
                if k != usize::MAX {
                    return Some((k, &self.a.nzval[p], &self.x[self.column]));
                }
            }
            self.column += 1;
        }
        None
    }
}

#[cfg(test)]
#[path = "sparse_parallel_tests.rs"]
mod tests;
