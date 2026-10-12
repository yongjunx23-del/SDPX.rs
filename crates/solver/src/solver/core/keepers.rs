//! Idle-priority core keepers.
//!
//! A frequency governor that raises a core's clock only under sustained load
//! (`conservative`: one 5% step per 20 ms sample) keeps cores that serve
//! short parallel regions at the lowest P-state. On EPYC 7742 nodes, the
//! split congruence bench ran 30–35% faster at 4–8 threads with every core
//! held busy (1.5 → 2.25 GHz). A keeper is one `SCHED_IDLE` thread pinned to
//! each CPU of the process's affinity mask: a normal thread waking on that
//! CPU preempts it at once, so it only fills idle time.
//!
//! Owner MPI ranks wait on each other and on short regions, and gain: the
//! 4-node owner Λ35 solve ran 8.0% faster. One process whose pool keeps its
//! cores busy loses: plain Λ35 on 64 threads ran 2.0% slower (journal
//! 2026-10-12, L35). Keepers therefore start by default only for multi-rank
//! solves under a load-following governor; `SDPX_KEEP_CORES=0` disables them
//! and any other value forces them. Linux only.

/// Keepers stop and join when dropped.
pub struct CoreKeepers {
    #[cfg(target_os = "linux")]
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    #[cfg(target_os = "linux")]
    threads: Vec<std::thread::JoinHandle<()>>,
}

impl CoreKeepers {
    /// Keepers for a process of an MPI world with `ranks` processes: one per
    /// CPU of the calling thread's affinity mask. Call it before any
    /// affinity narrowing of the calling thread.
    pub fn for_world(ranks: usize) -> Self {
        match std::env::var("SDPX_KEEP_CORES") {
            Ok(value) if value == "0" => Self::none(),
            Ok(_) => Self::start(false),
            Err(_) if ranks > 1 => Self::start(true),
            Err(_) => Self::none(),
        }
    }

    fn none() -> Self {
        Self {
            #[cfg(target_os = "linux")]
            stop: Default::default(),
            #[cfg(target_os = "linux")]
            threads: Vec::new(),
        }
    }

    /// Number of running keepers.
    pub fn len(&self) -> usize {
        #[cfg(target_os = "linux")]
        {
            self.threads.len()
        }
        #[cfg(not(target_os = "linux"))]
        0
    }

    /// True when no keeper runs.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// With `auto`, start only when the first CPU's governor follows load.
    #[cfg(target_os = "linux")]
    fn start(auto: bool) -> Self {
        use std::sync::atomic::Ordering;
        let mut keepers = Self::none();
        // SAFETY: plain libc calls on zeroed, correctly sized cpu_set_t values.
        let cpus: Vec<usize> = unsafe {
            let mut mask: libc::cpu_set_t = std::mem::zeroed();
            if libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut mask) != 0 {
                return keepers;
            }
            (0..libc::CPU_SETSIZE as usize)
                .filter(|&cpu| libc::CPU_ISSET(cpu, &mask))
                .collect()
        };
        if auto && !cpus.first().is_some_and(|&cpu| load_governed(cpu)) {
            return keepers;
        }
        for cpu in cpus {
            let stop = keepers.stop.clone();
            let spawned = std::thread::Builder::new()
                .name(format!("sdpx-keeper-{cpu}"))
                .stack_size(64 << 10)
                .spawn(move || {
                    // SAFETY: as above; both calls act on this thread only.
                    let idle = unsafe {
                        let mut pinned: libc::cpu_set_t = std::mem::zeroed();
                        libc::CPU_SET(cpu, &mut pinned);
                        let param = libc::sched_param { sched_priority: 0 };
                        libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &pinned)
                            == 0
                            && libc::sched_setscheduler(0, libc::SCHED_IDLE, &param) == 0
                    };
                    // Never spin at normal priority.
                    if !idle {
                        return;
                    }
                    while !stop.load(Ordering::Relaxed) {
                        for _ in 0..1024 {
                            std::hint::spin_loop();
                        }
                    }
                });
            match spawned {
                Ok(handle) => keepers.threads.push(handle),
                Err(_) => break,
            }
        }
        crate::receipt::set_keepers(!keepers.threads.is_empty());
        keepers
    }

    #[cfg(not(target_os = "linux"))]
    fn start(_auto: bool) -> Self {
        Self::none()
    }
}

/// Whether `cpu`'s clock follows sampled load (the governors that leave a
/// briefly busy core at a low P-state).
#[cfg(target_os = "linux")]
fn load_governed(cpu: usize) -> bool {
    std::fs::read_to_string(format!(
        "/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor"
    ))
    .is_ok_and(|governor| matches!(governor.trim(), "conservative" | "ondemand" | "schedutil"))
}

impl Drop for CoreKeepers {
    fn drop(&mut self) {
        #[cfg(target_os = "linux")]
        {
            if self.threads.is_empty() {
                return;
            }
            self.stop.store(true, std::sync::atomic::Ordering::Relaxed);
            for handle in self.threads.drain(..) {
                let _ = handle.join();
            }
            crate::receipt::set_keepers(false);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_process_starts_no_keepers_by_default() {
        if std::env::var_os("SDPX_KEEP_CORES").is_none() {
            assert!(CoreKeepers::for_world(1).is_empty());
        }
    }

    #[test]
    fn forced_keepers_stop_on_drop() {
        // Linux may refuse SCHED_IDLE and start none; dropping joins the rest.
        let keepers = CoreKeepers::start(false);
        if !cfg!(target_os = "linux") {
            assert!(keepers.is_empty());
        }
        drop(keepers);
    }
}
