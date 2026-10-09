//! Cached, structurally weighted cone scheduling. Every task owns disjoint
//! cone state and row slices; joining a phase is its only synchronization.
use super::*;
use crate::algebra::CscMatrix;

pub(super) struct ConeThreading {
    pub(super) pool: std::sync::Arc<rayon::ThreadPool>,
    // KKT may widen the shared pool; cone planning keeps its original width.
    pub(super) cone_workers: usize,
    pub(super) cone_chunk: usize,
    pub(super) lanes: Vec<Lane>,
    pub(super) sym_step_lanes: Vec<Lane>,
    // A single large orthant, optionally with equality rows, uses disjoint
    // elementwise chunks on the same worker pool.
    pub(super) orthant_chunk: Option<usize>,
    // Inner (within-cone) work re-offered to the ambient pool only pays
    // when the cone budget has spare workers beyond its lanes.
    pub(super) inner_parallel: bool,
    // Workers per lane for split-capable kernels inside a cone (residue
    // congruences): SDPB-style, a cone budget wider than its lanes shares each
    // lane's heavy calls among ways = workers / lanes.
    pub(super) inner_ways: usize,
    // Paired joins of two independent operations (e.g. the two cone
    // Choleskys) need only one stealer, so they activate with any spare
    // cone worker — a weaker condition than `inner_parallel`.
    pub(super) paired: bool,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Lane {
    cone_start: usize,
    row_start: usize,
}

// Conservative initial launch threshold, expressed in scalar work units.
// Costs depend only on dimensions and precision, never on problem identities.
const MIN_LANE_WORK: u128 = 4096;
// Initial KKT admission cutoff in precision-weighted neighbor products.
// Much more work is needed to amortize extra workers for short KKT phases.
const MIN_KKT_WORK: u128 = 8 << 20;

fn cone_cost<T: FloatT>(cone: &SupportedCone<T>) -> u128 {
    let dimension_cost = match cone {
        SupportedCone::PSDTriangleCone(k) => (k.n as u128).saturating_pow(3),
        _ => cone.numel() as u128,
    };
    let words = ((T::precision_bits() + 63) / 64) as u128;
    dimension_cost.saturating_mul(words.saturating_mul(words))
}

impl ConeThreading {
    pub(super) fn new<T: FloatT>(
        cones: &[SupportedCone<T>],
        requested: usize,
        kkt: Option<(&CscMatrix<T>, &CscMatrix<T>)>,
    ) -> Result<Option<Self>, rayon::ThreadPoolBuildError> {
        Self::build(cones, requested, kkt, None)
    }

    /// Plan lanes for `cones` on an existing pool (its width is the budget).
    pub(super) fn with_pool<T: FloatT>(
        cones: &[SupportedCone<T>],
        pool: std::sync::Arc<rayon::ThreadPool>,
    ) -> Option<Self> {
        let width = pool.current_num_threads();
        Self::build(cones, width, None, Some(pool)).ok().flatten()
    }

    // Scale heavy PSD cones first. Whole cones remain independent; workers
    // pull the next largest job instead of waiting for a fixed lane's tail.
    // Workers left without a job help the running cones (see below).
    pub(super) fn update_scaling<T: FloatT>(
        &self,
        cones: &mut [SupportedCone<T>],
        s: &[T],
        z: &[T],
        mu: T,
        strategy: ScalingStrategy,
    ) -> Option<bool> {
        let workers = self.pool.current_num_threads();
        if T::precision_bits() <= 64
            || cones.len() < 2
            || !cones
                .iter()
                .any(|c| matches!(c, SupportedCone::PSDTriangleCone(_)))
        {
            return None;
        }
        let mut row = 0;
        let mut jobs: Vec<_> = cones
            .iter_mut()
            .map(|cone| {
                let rows = row..row + cone.numel();
                row = rows.end;
                (cone_cost(cone), cone, rows)
            })
            .collect();
        jobs.sort_unstable_by_key(|job| job.0);
        let total = jobs.len();
        let jobs = std::sync::Mutex::new(jobs);
        let done = std::sync::atomic::AtomicUsize::new(0);
        // A worker without a queued cone helps the running cones' posted
        // inner items (SVD rotation replay rows) until every cone is scaled:
        // the slowest cone otherwise sets the phase alone.
        Some(self.pool.install(|| {
            (0..workers)
                .into_par_iter()
                .map(|_| {
                    let mut ok = true;
                    loop {
                        let Some((_, cone, rows)) = jobs.lock().unwrap().pop() else {
                            break;
                        };
                        let _offer = sdpx_arithmetic::inner_parallel::board::Offer::enter();
                        let _inner = (self.inner_parallel || self.paired).then(|| {
                            sdpx_arithmetic::inner_parallel::Guard::enter_levels(
                                self.inner_parallel,
                                self.paired,
                            )
                        });
                        ok &= crate::algebra::with_split_hint(self.inner_ways, || {
                            cone.update_scaling(&s[rows.clone()], &z[rows], mu, strategy)
                        });
                        done.fetch_add(1, std::sync::atomic::Ordering::Release);
                    }
                    let _helper = sdpx_arithmetic::inner_parallel::board::Helper::enter();
                    while done.load(std::sync::atomic::Ordering::Acquire) < total {
                        // Posted rows first, then any stealable Rayon job
                        // (paired joins, split kernels) of the running cones.
                        if !sdpx_arithmetic::inner_parallel::board::help()
                            && rayon::yield_now() != Some(rayon::Yield::Executed)
                        {
                            std::hint::spin_loop();
                        }
                    }
                    ok
                })
                .reduce(|| true, |a, b| a && b)
        }))
    }

    fn build<T: FloatT>(
        cones: &[SupportedCone<T>],
        requested: usize,
        kkt: Option<(&CscMatrix<T>, &CscMatrix<T>)>,
        existing: Option<std::sync::Arc<rayon::ThreadPool>>,
    ) -> Result<Option<Self>, rayon::ThreadPoolBuildError> {
        let budget = crate::solver::core::worker_budget(requested);
        let single_orthant = matches!(cones, [SupportedCone::NonnegativeCone(_)]);
        let single_zero = matches!(cones, [SupportedCone::ZeroCone(_)]);
        let orthant_size = cones
            .iter()
            .find(|c| matches!(c, SupportedCone::NonnegativeCone(_)))
            .filter(|_| {
                cones.iter().all(|c| {
                    matches!(
                        c,
                        SupportedCone::ZeroCone(_) | SupportedCone::NonnegativeCone(_)
                    )
                }) && cones
                    .iter()
                    .filter(|c| matches!(c, SupportedCone::NonnegativeCone(_)))
                    .count()
                    == 1
            })
            .filter(|c| c.numel() > 0)
            .filter(|_| single_orthant || T::precision_bits() > 53)
            .map(|c| c.numel());
        let has_psd = cones.iter().any(|cone| {
            if matches!(cone, SupportedCone::PSDTriangleCone(_)) {
                return true;
            }
            let _ = cone;
            false
        });
        if budget <= 1
            || cones.is_empty()
            || (kkt.is_none() && cones.len() < 2 && !single_orthant && !single_zero && !has_psd)
        {
            return Ok(None);
        }
        let mut prefix = Vec::with_capacity(cones.len() + 1);
        prefix.push(0_u128);
        for cone in cones {
            prefix.push(prefix.last().unwrap().saturating_add(cone_cost(cone)));
        }
        let total = *prefix.last().unwrap();
        // Preserve the measured PSD admission. Materialized PSD A columns
        // do not describe the condensed factor's work.
        let cap = if existing.is_none() {
            useful_width(cones)
        } else {
            usize::MAX
        };
        let cone_workers = budget
            .min(cap)
            .min(if single_orthant {
                cones[0].numel()
            } else {
                budget
            })
            .min((total / MIN_LANE_WORK).min(usize::MAX as u128) as usize);
        let cone_workers = if cones.len() < 2 && !single_orthant && !single_zero && !has_psd {
            1
        } else {
            cone_workers.max(1)
        };
        // Squared column-neighbor counts estimate Gram/initial factor work
        // from the validated, preprocessed CSC pattern without another AMD
        // ordering or a dense allocation. This is a proxy, not a fill count.
        let kkt_workers = if has_psd {
            0
        } else {
            let words = T::precision_bits().div_ceil(64) as u128;
            let work = kkt.map_or(0, |(p, a)| {
                a.colptr
                    .windows(2)
                    .zip(p.colptr.windows(2))
                    .fold(0u128, |work, (a, p)| {
                        let neighbors = (a[1] - a[0]) as u128 + (p[1] - p[0]) as u128;
                        work.saturating_add(neighbors.saturating_pow(2))
                    })
            });
            (work.saturating_mul(words * words) / MIN_KKT_WORK).min(budget as u128) as usize
        };
        let workers = cone_workers.max(kkt_workers).min(cap);
        if workers <= 1 {
            return Ok(None);
        }
        // Contiguous lanes preserve original cone/data ordering and permit
        // safe slice splitting, without raw pointers or per-call job vectors.
        let orthant_chunk = orthant_size
            .filter(|_| cone_workers > 1)
            .map(|n| n.div_ceil(cone_workers));
        // Deliberate over-splitting: balanced lanes follow a structural cost
        // model, so spare tasks are what lets work stealing absorb model error.
        // Measured worse when reduced to one lane per worker (w8 13.35 -> 14.37,
        // w16 12.47 -> 14.11), so the task budget stays as it was.
        let task_budget = cone_workers
            .saturating_mul(4)
            .min((total / MIN_LANE_WORK).min(usize::MAX as u128) as usize)
            .max(1);
        let lanes = balanced_lanes(cones, &prefix, task_budget.min(cones.len()));
        // Any spare worker may steal a heavy cone's inner work (SVD rotation
        // replay rows, reflector columns); the slowest cone otherwise sets
        // the phase (Λ19 spins 0–50: 52 lanes on 64 workers, SVD 200 ms
        // median, 406 ms max). Inner splits are bitwise neutral.
        let inner_parallel = cone_workers > lanes.len().max(1);
        // Any spare worker can steal the second half of a paired join.
        let paired = cone_workers > lanes.len().max(1);
        let sym_step_lanes = {
            // Step-to-boundary bounds are cap-insensitive for symmetric
            // cones (Nonnegative/SOC/PSD/Zero): every cone is evaluated at
            // the same initial cap and the results fold into the running
            // minimum, bitwise identical to the serial pass.  Nonsymmetric
            // cones stay serial — their iterative searches use the cap.
            let mut prefix = Vec::with_capacity(cones.len() + 1);
            prefix.push(0u128);
            let mut active = 0usize;
            for cone in cones {
                let cost = if cone.is_symmetric() && cone.numel() > 0 {
                    active += 1;
                    cone_cost(cone)
                } else {
                    0
                };
                prefix.push(prefix.last().unwrap().saturating_add(cost));
            }
            let lanes = cone_workers
                .saturating_mul(4)
                .min(active)
                .min((*prefix.last().unwrap() / MIN_LANE_WORK).min(usize::MAX as u128) as usize);
            if lanes > 1 {
                balanced_lanes(cones, &prefix, lanes)
            } else {
                Vec::new()
            }
        };
        let pool = match existing {
            Some(pool) => pool,
            None => std::sync::Arc::new(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(workers)
                    .start_handler(pin_worker(budget, workers))
                    .build()?,
            ),
        };
        let inner_ways = (cone_workers / lanes.len().max(1)).max(1);
        let cone_chunk = if workers > cone_workers {
            cones.len().div_ceil(cone_workers)
        } else {
            1
        };
        Ok(Some(Self {
            pool,
            cone_workers,
            cone_chunk,
            lanes,
            sym_step_lanes,
            orthant_chunk,
            inner_parallel,
            inner_ways,
            paired,
        }))
    }
}

/// Workers an MPFR problem with PSD cones can keep busy: one per PSD cone
/// up to `psd_unit` of cost, one more per further unit, and the other cones'
/// cost in the same units, times `WIDTH_FACTOR`. Idle workers beyond
/// this only spin between the many short parallel phases, and the busy-wait
/// lowers the clock of the cores doing the work: ising11 (22 PSD cones of
/// order 12–16) ran at 3.12 GHz / IPC 2.69 on 8 threads but 2.35 GHz / IPC
/// 0.83 on 64, with useful instructions unchanged (journal 2026-10-07).
/// Problems without PSD cones and binary64 keep the requested width.
fn useful_width<T: FloatT>(cones: &[SupportedCone<T>]) -> usize {
    if T::precision_bits() <= 64 {
        return usize::MAX;
    }
    let unit = psd_unit::<T>();
    let (mut units, mut other, mut psd) = (0u128, 0u128, false);
    for cone in cones {
        if matches!(cone, SupportedCone::PSDTriangleCone(_)) {
            psd = true;
            units += cone_cost(cone).div_ceil(unit).max(1);
        } else {
            other += cone_cost(cone);
        }
    }
    if !psd {
        return usize::MAX;
    }
    units += other.div_ceil(unit);
    let width = (units as f64 * width_factor()).ceil();
    if width >= usize::MAX as f64 {
        usize::MAX
    } else {
        (width as usize).max(2)
    }
}

/// Cost (`cone_cost`) one worker absorbs per PSD cone: a cone of order 64.
fn psd_unit<T: FloatT>() -> u128 {
    let words = T::precision_bits().div_ceil(64) as u128;
    64u128.pow(3) * words * words
}

/// Width per useful unit.
const WIDTH_FACTOR: f64 = 1.25;

fn width_factor() -> f64 {
    WIDTH_FACTOR
}

/// Keep serial solve work on the first allocated CPU while existing worker
/// pools retain the full allocation. Restore the caller's mask on return.
pub(crate) struct SolveAffinityGuard {
    #[cfg(target_os = "linux")]
    original: Option<libc::cpu_set_t>,
}

impl SolveAffinityGuard {
    pub(crate) fn enter(budget: usize) -> Self {
        #[cfg(target_os = "linux")]
        unsafe {
            let mut original: libc::cpu_set_t = std::mem::zeroed();
            let size = std::mem::size_of::<libc::cpu_set_t>();
            if budget > 1
                && libc::sched_getaffinity(0, size, &mut original) == 0
                && libc::CPU_COUNT(&original) as usize == budget
            {
                let first = (0..libc::CPU_SETSIZE as usize)
                    .find(|&cpu| libc::CPU_ISSET(cpu, &original))
                    .unwrap();
                let mut pinned: libc::cpu_set_t = std::mem::zeroed();
                libc::CPU_SET(first, &mut pinned);
                if libc::sched_setaffinity(0, size, &pinned) == 0 {
                    return Self {
                        original: Some(original),
                    };
                }
            }
            Self { original: None }
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = budget;
            Self {}
        }
    }

    /// Callback-created workers must inherit the full allocation. Repin the
    /// calling thread when this returned guard drops, including on unwind.
    pub(crate) fn suspend(&self) -> Self {
        #[cfg(target_os = "linux")]
        unsafe {
            if let Some(allocation) = &self.original {
                let mut pinned: libc::cpu_set_t = std::mem::zeroed();
                let size = std::mem::size_of::<libc::cpu_set_t>();
                if libc::sched_getaffinity(0, size, &mut pinned) == 0
                    && libc::sched_setaffinity(0, size, allocation) == 0
                {
                    return Self {
                        original: Some(pinned),
                    };
                }
            }
            Self { original: None }
        }
        #[cfg(not(target_os = "linux"))]
        Self {}
    }
}

impl Drop for SolveAffinityGuard {
    fn drop(&mut self) {
        #[cfg(target_os = "linux")]
        if let Some(original) = &self.original {
            // This guard stays on Solver::solve's calling thread.
            unsafe {
                libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), original);
            }
        }
    }
}

/// SDPB-style static placement: when the process is bound to exactly as
/// many CPUs as the requested thread budget (numactl/taskset/MPI binding
/// gave it its own cores), worker `i` is bound to the i-th of them for its
/// lifetime. A wider mask is left alone, so unbound processes sharing a node
/// never collide on the same cores. Measured −4% at 52 threads (Λ19 spins 0–50). Unpinned workers migrate between cores and lose
/// their caches, and at high thread counts every phase (serial ones too)
/// measured 20–40% slower per call. A pool narrower than the budget uses
/// its first `workers` CPUs. The caller keeps its mask, so later pools and
/// repeated handle calls inherit the original allocation. Linux only;
/// elsewhere a no-op.
pub(crate) fn pin_worker(budget: usize, workers: usize) -> impl Fn(usize) + Send + Sync + 'static {
    #[cfg(target_os = "linux")]
    let cpus: Vec<usize> = unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        if libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut set) == 0 {
            (0..libc::CPU_SETSIZE as usize)
                .filter(|&c| libc::CPU_ISSET(c, &set))
                .collect()
        } else {
            Vec::new()
        }
    };
    #[cfg(not(target_os = "linux"))]
    let cpus: Vec<usize> = Vec::new();
    let pin = cpus.len() == budget;
    move |i: usize| {
        debug_assert!(i < workers);
        #[cfg(target_os = "linux")]
        if pin {
            unsafe {
                let mut set: libc::cpu_set_t = std::mem::zeroed();
                libc::CPU_SET(cpus[i], &mut set);
                libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
            }
        }
        #[cfg(not(target_os = "linux"))]
        let _ = (i, pin, &cpus);
    }
}

fn balanced_lanes<T: FloatT>(
    cones: &[SupportedCone<T>],
    prefix: &[u128],
    workers: usize,
) -> Vec<Lane> {
    let (mut next, mut row) = (0, 0);
    crate::utils::partition::contiguous_lanes(prefix, workers)
        .into_iter()
        .map(|cone_start| {
            row += cones[next..cone_start]
                .iter()
                .map(|c| c.numel())
                .sum::<usize>();
            next = cone_start;
            Lane {
                cone_start,
                row_start: row,
            }
        })
        .collect()
}

// Splitting consumes the borrowed bundle and returns nonoverlapping bundles.
// The type system prevents cross-lane aliasing, including MPFR-owned values.
pub(super) trait RowBuffers: Send + Sized {
    fn split(self, at: usize) -> (Self, Self);
}
impl RowBuffers for () {
    fn split(self, _at: usize) -> (Self, Self) {
        ((), ())
    }
}
impl<'a, T: Send> RowBuffers for &'a mut [T] {
    fn split(self, at: usize) -> (Self, Self) {
        self.split_at_mut(at)
    }
}
impl<T: Send> RowBuffers for (&mut [T], &mut [T]) {
    fn split(self, at: usize) -> (Self, Self) {
        let (a, b) = self.0.split_at_mut(at);
        let (c, d) = self.1.split_at_mut(at);
        ((a, c), (b, d))
    }
}
impl<T: Send> RowBuffers for (&mut [T], &mut [T], &mut [T]) {
    fn split(self, at: usize) -> (Self, Self) {
        let (a, b) = self.0.split_at_mut(at);
        let (c, d) = self.1.split_at_mut(at);
        let (e, f) = self.2.split_at_mut(at);
        ((a, c, e), (b, d, f))
    }
}

pub(super) fn apply<T, B, F>(
    cones: &mut [SupportedCone<T>],
    lanes: &[Lane],
    inner: bool,
    paired: bool,
    ways: usize,
    buffers: B,
    kernel: &F,
) -> bool
where
    T: FloatT,
    B: RowBuffers,
    F: Fn(&mut SupportedCone<T>, Range<usize>, B) -> bool + Sync,
{
    if lanes.len() == 1 {
        // This leaf runs on a solver-pool worker (apply is only ever invoked
        // under `pool.install`), so heavy per-cone kernels may re-offer
        // independent inner work to the ambient pool when spare workers exist.
        let _inner = (inner || paired)
            .then(|| sdpx_arithmetic::inner_parallel::Guard::enter_levels(inner, paired));
        let mut buffers = buffers;
        let mut row = lanes[0].row_start;
        let mut success = true;
        for cone in cones {
            let end = row + cone.numel();
            let (current, remaining) = buffers.split(end - row);
            // Do not short circuit: all block tasks finish before reporting
            // failure; the caller then discards the failed scaling phase.
            success &= crate::algebra::with_split_hint(ways, || kernel(cone, row..end, current));
            buffers = remaining;
            row = end;
        }
        success
    } else {
        let mid = lanes.len() / 2;
        let (left, right) = cones.split_at_mut(lanes[mid].cone_start - lanes[0].cone_start);
        let (left_buffers, right_buffers) =
            buffers.split(lanes[mid].row_start - lanes[0].row_start);
        let (a, b) = rayon::join(
            || {
                apply(
                    left,
                    &lanes[..mid],
                    inner,
                    paired,
                    ways,
                    left_buffers,
                    kernel,
                )
            },
            || {
                apply(
                    right,
                    &lanes[mid..],
                    inner,
                    paired,
                    ways,
                    right_buffers,
                    kernel,
                )
            },
        );
        a && b
    }
}

// The only shared inputs are immutable direction/state slices. Every cone
// writes one disjoint cone-indexed bound pair; symmetric cones evaluate
// `step_length` at the common cap, which the ordered fold then reduces —
// bitwise identical to the serial running-cap fold because their step
// lengths are cap-insensitive.
pub(super) fn sym_step_bounds<T: FloatT>(
    cones: &mut [SupportedCone<T>],
    bounds: &mut [(T, T)],
    lanes: &[Lane],
    inner: bool,
    paired: bool,
    ways: usize,
    dz: &[T],
    ds: &[T],
    z: &[T],
    s: &[T],
    settings: &CoreSettings<T>,
    alpha: T,
) {
    if lanes.len() == 1 {
        // Leaf task on a pool worker: per-cone step kernels may re-offer
        // independent inner work (paired dz/ds bounds, dense kernels) to
        // the ambient pool when spare workers exist.
        let _inner = (inner || paired)
            .then(|| sdpx_arithmetic::inner_parallel::Guard::enter_levels(inner, paired));
        crate::algebra::with_split_hint(ways, || {
            let mut row = lanes[0].row_start;
            for (cone, bound) in cones.iter_mut().zip(bounds) {
                let end = row + cone.numel();
                if cone.is_symmetric() {
                    *bound = cone.step_length(
                        &dz[row..end],
                        &ds[row..end],
                        &z[row..end],
                        &s[row..end],
                        settings,
                        alpha,
                    );
                }
                row = end;
            }
        })
    } else {
        let mid = lanes.len() / 2;
        let cut = lanes[mid].cone_start - lanes[0].cone_start;
        let (left, right) = cones.split_at_mut(cut);
        let (left_bounds, right_bounds) = bounds.split_at_mut(cut);
        rayon::join(
            || {
                sym_step_bounds(
                    left,
                    left_bounds,
                    &lanes[..mid],
                    inner,
                    paired,
                    ways,
                    dz,
                    ds,
                    z,
                    s,
                    settings,
                    alpha,
                )
            },
            || {
                sym_step_bounds(
                    right,
                    right_bounds,
                    &lanes[mid..],
                    inner,
                    paired,
                    ways,
                    dz,
                    ds,
                    z,
                    s,
                    settings,
                    alpha,
                )
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn balanced_partition_and_joined_failure_own_each_row_once() {
        let kinds = vec![SupportedConeT::<f64>::NonnegativeConeT(4096); 4];
        let mut cones: Vec<_> = kinds.iter().map(make_cone).collect();
        let threading = ConeThreading::new(&cones, 2, None).unwrap().unwrap();
        assert_eq!(threading.pool.current_num_threads(), 2);
        assert_eq!(threading.lanes.len(), 4);
        for (i, lane) in threading.lanes.iter().enumerate() {
            assert_eq!(lane.cone_start, i);
            assert_eq!(lane.row_start, 4096 * i);
        }
        let mut a = vec![0.; 16384];
        let mut b = a.clone();
        let ok = threading.pool.install(|| {
            apply(
                &mut cones,
                &threading.lanes,
                false,
                false,
                1,
                (&mut a[..], &mut b[..]),
                &|_cone, rows, (a, b)| {
                    assert!(rayon::current_thread_index().is_some());
                    a.fill((rows.start + 1) as f64);
                    b.fill((rows.end + 1) as f64);
                    rows.start != 0
                },
            )
        });
        assert!(!ok);
        // Even the failed first lane and every later lane completed before
        // the caller sees false; no shared reduction/writeback was required.
        for i in 0..16384 {
            assert_eq!(a[i], ((i / 4096) * 4096 + 1) as f64);
            assert_eq!(b[i], ((i / 4096 + 1) * 4096 + 1) as f64);
        }
    }
}

pub(super) fn prepare_affine_bounds<T: FloatT>(
    cones: &mut [SupportedCone<T>],
    bounds: &mut [(T, T)],
    lanes: &[Lane],
    inner: bool,
    paired: bool,
    ways: usize,
    dz: &mut [T],
    ds: &mut [T],
    z: &[T],
    s: &[T],
    settings: &CoreSettings<T>,
    alpha: T,
) {
    if lanes.len() <= 1 {
        let _inner = (inner || paired)
            .then(|| sdpx_arithmetic::inner_parallel::Guard::enter_levels(inner, paired));
        crate::algebra::with_split_hint(ways, || {
            let mut row = 0;
            for (cone, bound) in cones.iter_mut().zip(bounds) {
                let end = row + cone.numel();
                if let SupportedCone::PSDTriangleCone(cone) = cone {
                    *bound =
                        cone.prepare_affine_bounds(&mut dz[row..end], &mut ds[row..end], alpha);
                } else if cone.is_symmetric() {
                    *bound = cone.step_length(
                        &dz[row..end],
                        &ds[row..end],
                        &z[row..end],
                        &s[row..end],
                        settings,
                        alpha,
                    );
                }
                row = end;
            }
        })
    } else {
        let mid = lanes.len() / 2;
        let cut = lanes[mid].cone_start - lanes[0].cone_start;
        let row = lanes[mid].row_start - lanes[0].row_start;
        let (lc, rc) = cones.split_at_mut(cut);
        let (lb, rb) = bounds.split_at_mut(cut);
        let (lz, rz) = dz.split_at_mut(row);
        let (ls, rs) = ds.split_at_mut(row);
        let (lv, rv) = z.split_at(row);
        let (lw, rw) = s.split_at(row);
        rayon::join(
            || {
                prepare_affine_bounds(
                    lc,
                    lb,
                    &lanes[..mid],
                    inner,
                    paired,
                    ways,
                    lz,
                    ls,
                    lv,
                    lw,
                    settings,
                    alpha,
                )
            },
            || {
                prepare_affine_bounds(
                    rc,
                    rb,
                    &lanes[mid..],
                    inner,
                    paired,
                    ways,
                    rz,
                    rs,
                    rv,
                    rw,
                    settings,
                    alpha,
                )
            },
        );
    }
}
