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

fn receipts_requested() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("SDPX_RECEIPT").is_some())
}

/// Start a detailed timer only when instrumentation is requested.
pub fn start() -> Option<Instant> {
    (receipts_requested() || profile_requested()).then(Instant::now)
}

/// Finish an optional detailed timer.
pub fn finish(name: &'static str, timer: Option<Instant>) {
    if let Some(timer) = timer {
        phase(name, timer.elapsed());
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
pub fn phase_record(name: &'static str, dur: Duration) {
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
    )
}

/// Write a receipt for the owner-partitioned backend.
#[cfg(all(feature = "serde", feature = "sdp"))]
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
) -> std::io::Result<()> {
    let Some(path) = std::env::var_os("SDPX_RECEIPT") else {
        return Ok(());
    };
    let phases = drain();
    let mut phase_map = serde_json::Map::new();
    for (name, s) in &phases {
        phase_map.insert(
            (*name).to_string(),
            serde_json::json!({
                "count": s.count,
                "total_s": s.total.as_secs_f64(),
                "median_s": s.median().as_secs_f64(),
                "max_s": s.max.as_secs_f64(),
            }),
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
            "rank": crate::MpiContext::initialize().rank(),
            "phase_scope": "calling rank only",
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
        "memory": {"peak_rss_bytes": peak_rss},
        "env": {
            "SDPX_DIRECT_SOLVE": env("SDPX_DIRECT_SOLVE"),
            "SDPX_RNS_OPS": env("SDPX_RNS_OPS"),
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
