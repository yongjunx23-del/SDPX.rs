//! Per-solve phase aggregation for execution receipts (plan PR-01).
//!
//! `phase()` replaces the raw `SDPX_PROFILE` eprintln sites: it keeps the
//! stderr lines byte-compatible and, when `SDPX_RECEIPT` names an output
//! path, also accumulates count/total/samples per phase name for the JSON
//! receipt emitted by the FFI layer at the end of each solve.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::sync::{Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant};

/// Aggregated timing for one named phase within a solve.
#[derive(Clone, Debug, Default)]
pub struct PhaseStat {
    /// Number of recorded observations.
    pub count: u64,
    /// Sum of all observations.
    pub total: Duration,
    /// Largest single observation.
    pub max: Duration,
    samples: Vec<Duration>,
}

impl PhaseStat {
    /// Median observed duration (`Duration::ZERO` when empty).
    pub fn median(&self) -> Duration {
        if self.samples.is_empty() {
            return Duration::ZERO;
        }
        let mut s = self.samples.clone();
        s.sort_unstable();
        s[s.len() / 2]
    }
}

static PHASES: Mutex<BTreeMap<&'static str, PhaseStat>> = Mutex::new(BTreeMap::new());
/// Process CPU time per phase for timers that measured it.
static CPU: Mutex<BTreeMap<&'static str, Duration>> = Mutex::new(BTreeMap::new());
/// Calling-thread CPU time per phase (serial time for main-thread timers).
static SERIAL: Mutex<BTreeMap<&'static str, Duration>> = Mutex::new(BTreeMap::new());
/// Time spent entering each numbered MPI collective site (count, total).
/// The first agreement round absorbs waiting for the slowest rank, so this
/// locates load imbalance by call site.
static SITES: Mutex<BTreeMap<usize, (u64, Duration)>> = Mutex::new(BTreeMap::new());

/// Record the entry time of one collective at `site`.
pub(crate) fn site_record(site: usize, dur: Duration) {
    if receipts_requested() {
        let mut g = SITES.lock().unwrap();
        let s = g.entry(site).or_default();
        s.0 += 1;
        s.1 += dur;
    }
}

fn receipts_requested() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("SDPX_RECEIPT").is_some())
}

/// A detailed timer: wall start plus, under receipts, process CPU time.
pub struct Mark {
    wall: Instant,
    cpu: Option<Duration>,
    serial: Option<Duration>,
}

/// Start a detailed timer only when instrumentation is requested.
pub fn start() -> Option<Mark> {
    (receipts_requested() || profile_requested()).then(|| Mark {
        wall: Instant::now(),
        cpu: receipts_requested().then(process_cpu).flatten(),
        serial: receipts_requested().then(thread_cpu).flatten(),
    })
}

/// Finish an optional detailed timer. For phases timed on the calling
/// thread around a whole parallel section, `cpu_s / total_s` in the receipt
/// is the section's average busy threads.
pub fn finish(name: &'static str, timer: Option<Mark>) {
    if let Some(timer) = timer {
        phase(name, timer.wall.elapsed());
        if let (Some(cpu), Some(now)) = (timer.cpu, process_cpu()) {
            *CPU.lock().unwrap().entry(name).or_default() += now.saturating_sub(cpu);
        }
        serial_add(name, timer.serial);
    }
}

/// Record one phase observation: echo the legacy `PHASE` line under
/// `SDPX_PROFILE` and aggregate when a receipt was requested.
pub fn phase(name: &'static str, dur: Duration) {
    if profile_requested() {
        eprintln!("PHASE {name} {dur:?}");
    }
    phase_record(name, dur);
}

/// Record without printing — for sites whose stderr line carries extra
/// fields beyond `PHASE <name> <dur>` and keeps its own eprintln.
pub(crate) fn phase_record(name: &'static str, dur: Duration) {
    if receipts_requested() {
        let mut g = PHASES.lock().unwrap();
        let s = g.entry(name).or_default();
        s.count += 1;
        s.total += dur;
        s.max = s.max.max(dur);
        s.samples.push(dur);
    }
}

// Only instrumented solves serialize. Rayon workers join before Scope drops;
// completed observations belong to the calling thread even if another handle
// starts solving before the FFI writes its receipt.
static RECORDING: Mutex<()> = Mutex::new(());
thread_local! {
    static COMPLETED: RefCell<BTreeMap<&'static str, PhaseStat>> = RefCell::default();
}

/// Process CPU time (user + system, all threads) when receipts are on.
fn process_cpu() -> Option<Duration> {
    clock_cpu(if cfg!(target_os = "macos") { 12 } else { 2 })
}

/// CPU time of the calling thread. Timed phases run on the solver's main thread,
/// which sleeps inside `pool.install`, so this is the phase's serial time.
fn thread_cpu() -> Option<Duration> {
    // CLOCK_THREAD_CPUTIME_ID: 3 on Linux, 16 on macOS.
    clock_cpu(if cfg!(target_os = "macos") { 16 } else { 3 })
}

fn clock_cpu(clock: i32) -> Option<Duration> {
    #[repr(C)]
    struct Timespec {
        sec: i64,
        nsec: i64,
    }
    extern "C" {
        fn clock_gettime(clock: i32, ts: *mut Timespec) -> i32;
    }
    let mut ts = Timespec { sec: 0, nsec: 0 };
    // SAFETY: the caller-owned timespec matches the 64-bit C layout.
    (unsafe { clock_gettime(clock, &mut ts) } == 0)
        .then(|| Duration::new(ts.sec.max(0) as u64, ts.nsec.clamp(0, 999_999_999) as u32))
}

/// Start a CPU-time mark for a top-level solver phase (receipts only).
pub fn cpu_start() -> Option<CpuMark> {
    receipts_requested()
        .then(process_cpu)
        .flatten()
        .map(|cpu| CpuMark {
            wall: Instant::now(),
            cpu,
            serial: thread_cpu(),
        })
}

/// Wall, process-CPU and calling-thread-CPU start of a phase.
pub struct CpuMark {
    wall: Instant,
    cpu: Duration,
    serial: Option<Duration>,
}

/// Add calling-thread CPU time since `start` to phase `name`'s serial time.
fn serial_add(name: &'static str, start: Option<Duration>) {
    if let (Some(start), Some(now)) = (start, thread_cpu()) {
        *SERIAL.lock().unwrap().entry(name).or_default() += now.saturating_sub(start);
    }
}

/// Record the process CPU time spent in a phase as `name` (`cpu.<phase>`);
/// divided by the phase's wall time it gives the average busy threads
/// (idle rayon workers spin briefly, so this is an upper bound).
pub fn cpu_finish(name: &'static str, wall: &'static str, start: Option<CpuMark>) {
    if let (Some(mark), Some(now)) = (start, process_cpu()) {
        phase_record(name, now.saturating_sub(mark.cpu));
        phase_record(wall, mark.wall.elapsed());
        serial_add(wall, mark.serial);
    }
}

/// Process peak resident set size in bytes (`getrusage` high-water mark).
pub fn peak_rss_bytes() -> Option<u64> {
    #[repr(C)]
    struct Rusage {
        utime: [i64; 2],
        stime: [i64; 2],
        maxrss: i64,
        rest: [i64; 13],
    }
    extern "C" {
        fn getrusage(who: i32, usage: *mut Rusage) -> i32;
    }
    let mut usage = std::mem::MaybeUninit::<Rusage>::zeroed();
    // SAFETY: RUSAGE_SELF (0) fills the caller-owned struct; the layout
    // matches Linux x86_64/aarch64 and macOS (two 16-byte timevals, then longs).
    if unsafe { getrusage(0, usage.as_mut_ptr()) } != 0 {
        return None;
    }
    let maxrss = unsafe { usage.assume_init() }.maxrss.max(0) as u64;
    // Linux reports kilobytes, macOS bytes.
    Some(if cfg!(target_os = "macos") {
        maxrss
    } else {
        maxrss * 1024
    })
}

/// Under `SDPX_PROFILE`, print the peak RSS reached by `label`.
pub fn memory_mark(label: &str) {
    if profile_requested() {
        if let Some(bytes) = peak_rss_bytes() {
            eprintln!(
                "MEMORY {label} peak_rss={:.0} MiB",
                bytes as f64 / (1u64 << 20) as f64
            );
        }
    }
}

/// Cached profile configuration; disabled instrumentation does no environment lookup.
pub fn profile_requested() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("SDPX_PROFILE").is_some())
}

pub(crate) struct Scope(Option<MutexGuard<'static, ()>>);
impl Scope {
    pub(crate) fn begin() -> Self {
        Self::configured(receipts_requested())
    }

    fn configured(enabled: bool) -> Self {
        if !enabled {
            return Self(None);
        }
        let guard = RECORDING.lock().unwrap_or_else(|e| e.into_inner());
        PHASES.lock().unwrap().clear();
        SITES.lock().unwrap().clear();
        CPU.lock().unwrap().clear();
        SERIAL.lock().unwrap().clear();
        COMPLETED.with(|c| c.borrow_mut().clear());
        Self(Some(guard))
    }
}
impl Drop for Scope {
    fn drop(&mut self) {
        if self.0.is_some() {
            let stats = std::mem::take(&mut *PHASES.lock().unwrap());
            COMPLETED.with(|c| *c.borrow_mut() = stats);
        }
    }
}

/// Take this thread's most recently completed solve observations.
pub fn drain() -> BTreeMap<&'static str, PhaseStat> {
    COMPLETED.with(|c| std::mem::take(&mut *c.borrow_mut()))
}

/// Write the optional execution receipt shared by native and C ABI callers.
/// File errors are diagnostic and never change a numerical solve's status.
/// `peak_rss` is an optional caller-supplied process high-water mark in bytes.
#[cfg(feature = "serde")]
pub fn write<T: crate::algebra::FloatT>(
    solver: &crate::solver::DefaultSolver<T>,
    peak_rss: Option<u64>,
) {
    if let Err(error) = try_write(solver, peak_rss) {
        eprintln!("SDPX_RECEIPT write failed: {error}");
    }
}

/// Write a receipt with an observable I/O result. Native MPI applications use
/// this on rank zero and synchronize the result before shutting down.
#[cfg(feature = "serde")]
pub fn try_write<T: crate::algebra::FloatT>(
    solver: &crate::solver::DefaultSolver<T>,
    peak_rss: Option<u64>,
) -> std::io::Result<()> {
    try_write_parts(
        &solver.solution,
        &solver.info,
        solver.kktsystem.counters(),
        solver.cones.cone_threads(),
        1,
        peak_rss,
        solver
            .timers
            .as_ref()
            .map(|t| BTreeMap::from([("setup".to_string(), t.setup_time().as_secs_f64())])),
    )
}

/// Write a receipt for the owner-partitioned backend.
#[cfg(feature = "serde")]
pub fn try_write_partitioned<T: crate::algebra::FloatT>(
    solver: &crate::solver::PartitionedSolver<T>,
    peak_rss: Option<u64>,
) -> std::io::Result<()> {
    try_write_parts(
        solver.solution(),
        solver.info(),
        solver.counters(),
        solver.cone_threads(),
        solver.partitions(),
        peak_rss,
        None,
    )
}

#[cfg(feature = "serde")]
fn try_write_parts<T: crate::algebra::FloatT>(
    solution: &crate::solver::DefaultSolution<T>,
    info: &crate::solver::DefaultInfo<T>,
    ctr: crate::solver::kkt::SolveCounters,
    cone_threads: usize,
    partitions: usize,
    peak_rss: Option<u64>,
    setup: Option<BTreeMap<String, f64>>,
) -> std::io::Result<()> {
    let Some(mut path) = std::env::var_os("SDPX_RECEIPT") else {
        return Ok(());
    };
    // Non-root ranks (written only under SDPX_RECEIPT_ALL_RANKS) get a
    // `.rank<r>` suffix so per-rank phase totals expose load imbalance.
    let rank = crate::MpiContext::initialize().rank();
    if rank != 0 {
        path.push(format!(".rank{rank}"));
    }
    let phases = drain();
    let cpu = CPU.lock().unwrap().clone();
    let serial = SERIAL.lock().unwrap().clone();
    let mut phase_map = serde_json::Map::new();
    for (name, s) in &phases {
        let mut entry = serde_json::json!({
            "count": s.count,
            "total_s": s.total.as_secs_f64(),
            "median_s": s.median().as_secs_f64(),
            "max_s": s.max.as_secs_f64(),
        });
        if let Some(cpu) = cpu.get(name) {
            entry["cpu_s"] = serde_json::json!(cpu.as_secs_f64());
        }
        if let Some(serial) = serial.get(name) {
            entry["serial_s"] = serde_json::json!(serial.as_secs_f64());
        }
        phase_map.insert((*name).to_string(), entry);
    }
    let mut site_map = serde_json::Map::new();
    for (site, (count, total)) in SITES.lock().unwrap().iter() {
        site_map.insert(
            site.to_string(),
            serde_json::json!({"count": count, "total_s": total.as_secs_f64()}),
        );
    }
    let env = |k: &str| match std::env::var(k) {
        Ok(v) => serde_json::json!(v),
        Err(_) => serde_json::Value::Null,
    };
    let v = serde_json::json!({
        "schema_version": 1,
        "crate_version": env!("CARGO_PKG_VERSION"),
        "git_hash": option_env!("SDPX_GIT_HASH"),
        "precision_bits": T::precision_bits(),
        "mpi": {
            "world_size": crate::mpi_world_size(),
            "rank": rank,
            "phase_scope": "calling rank only",
            "site_entry": site_map,
        },
        "backend": info.linsolver.name,
        "threads": {
            "backend": info.linsolver.threads,
            "cones": cone_threads,
        },
        "partitions": partitions,
        "dimensions": {"n": solution.x.len(), "m": solution.z.len()},
        "iterations": solution.iterations,
        "status": format!("{:?}", solution.status),
        "solve_time_s": solution.solve_time,
        "counters": {
            "factor_attempts": ctr.factor_attempts,
            "linear_solves": ctr.linear_solves,
            "refinements": ctr.refinements,
            "factorizations": ctr.factorizations,
            "rhs_applied": ctr.rhs_applied,
            "batches": ctr.batches,
        },
        "phases": phase_map,
        "setup_seconds_inclusive": setup,
        "memory": {"peak_rss_bytes": peak_rss},
        "env": {
            "SDPX_DIRECT_SOLVE": env("SDPX_DIRECT_SOLVE"),
            "SDPX_INPUT_ID": env("SDPX_INPUT_ID"),
        },
    });
    // Multiple handles may finish before the previous receipt is written.
    // A shared output path must still contain one complete JSON document.
    static WRITER: Mutex<()> = Mutex::new(());
    let _writer = WRITER.lock().unwrap_or_else(|e| e.into_inner());
    std::fs::write(
        &path,
        serde_json::to_string_pretty(&v).map_err(std::io::Error::other)?,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn concurrent_scopes_and_repeated_solves_are_isolated() {
        let handles: Vec<_> = (1..=2)
            .map(|id| {
                std::thread::spawn(move || {
                    for _ in 0..3 {
                        {
                            let _scope = Scope::configured(true);
                            std::thread::scope(|s| {
                                s.spawn(|| {
                                    let mut phases = PHASES.lock().unwrap();
                                    let stat = phases.entry("worker").or_default();
                                    stat.count += id;
                                });
                            });
                        }
                        std::thread::yield_now();
                        assert_eq!(drain()["worker"].count, id);
                        assert!(drain().is_empty());
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
    }
}
