//! Deterministic sparse residual products in the solver's existing worker pool.
use crate::algebra::*;
use std::sync::Arc;

const MIN_PARALLEL_WORK: u128 = 32768;

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
        let mut cursor = rowptr[..a.m].to_vec();
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
                entries[cursor[row]] = Entry { column, position };
                cursor[row] += 1;
                if symmetric && row != column {
                    entries[cursor[column]] = Entry {
                        column: row,
                        position,
                    };
                    cursor[column] += 1;
                }
            }
        }
        Self {
            rowptr,
            entries,
            row_lanes: Vec::new(),
            column_lanes: Vec::new(),
            workers: 0,
            pool: None,
        }
    }

    pub(crate) fn configure<T>(&mut self, a: &CscMatrix<T>, pool: Option<Arc<rayon::ThreadPool>>) {
        let workers = pool.as_ref().map_or(1, |p| p.current_num_threads());
        self.pool = pool;
        if self.workers != workers {
            self.workers = workers;
            self.row_lanes = partitions(&self.rowptr, workers);
            self.column_lanes = partitions(&a.colptr, workers);
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

    /// Rows of `a` holding entries, and their prefix work (entries plus one
    /// per row) for [`Self::forward_sharded_active`].
    pub(crate) fn active_rows(&self) -> (Vec<usize>, Vec<usize>) {
        let rows: Vec<usize> = (0..self.rowptr.len() - 1)
            .filter(|&r| self.rowptr[r + 1] > self.rowptr[r])
            .collect();
        let mut ptr = Vec::with_capacity(rows.len() + 1);
        ptr.push(0);
        for &r in &rows {
            ptr.push(ptr.last().unwrap() + self.rowptr[r + 1] - self.rowptr[r] + 1);
        }
        (rows, ptr)
    }

    /// Rank-sharded forward product exchanging only the rows that hold
    /// entries (`active`, with work prefix `ptr` from [`Self::active_rows`]);
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
        (active, ptr): (&[usize], &[usize]),
    ) {
        let count = active.len();
        let lanes = partitions(ptr, world.size());
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
        let lanes = if transpose {
            &self.column_lanes
        } else {
            &self.row_lanes
        };
        if lanes.len() <= 1 {
            if transpose {
                a.t().gemv(y, x, alpha, beta);
            } else {
                a.gemv(y, x, alpha, beta);
            }
            return;
        }
        split_outputs(y, lanes, &|output, y| {
            self.output(a, transpose, output, y, x, alpha, beta)
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

#[cfg(test)]
#[path = "sparse_parallel_tests.rs"]
mod tests;
