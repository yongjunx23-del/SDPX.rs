//! Per-solve phase aggregation for execution receipts (plan PR-01).
//!
//! `phase()` replaces the raw `SDPX_PROFILE` eprintln sites: it keeps the
//! stderr lines byte-compatible and, when `SDPX_RECEIPT` names an output
//! path, also accumulates count/total/samples per phase name for the JSON
//! receipt emitted by the FFI layer at the end of each solve.

use std::collections::BTreeMap;
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

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

/// Record one phase observation: echo the legacy `PHASE` line under
/// `SDPX_PROFILE` and aggregate when a receipt was requested.
pub fn phase(name: &'static str, dur: Duration) {
    if std::env::var_os("SDPX_PROFILE").is_some() {
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

/// Take the accumulated phase stats, leaving the registry empty. Call at
/// solve boundaries so each receipt describes one solve.
pub fn drain() -> BTreeMap<&'static str, PhaseStat> {
    std::mem::take(&mut *PHASES.lock().unwrap())
}
