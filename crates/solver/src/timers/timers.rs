use std::time::{Duration, Instant};

/// Wall clock of one solver: construction time plus the current solve. Phase
/// timings (wall and CPU) are recorded by [`timeit!`] in [`crate::receipt`].
#[derive(Debug, Default, Clone)]
pub struct Timers {
    setup: Duration,
    solve: Duration,
    setup_start: Option<Instant>,
    solve_start: Option<Instant>,
}

impl Timers {
    /// Begin (or resume) timing solver construction.
    pub fn start_setup(&mut self) {
        self.setup_start = Some(Instant::now());
    }

    /// Stop timing construction, adding the elapsed time to the setup total.
    pub fn stop_setup(&mut self) {
        if let Some(start) = self.setup_start.take() {
            self.setup += start.elapsed();
        }
    }

    /// Start the solve clock; a repeated solve restarts it.
    pub fn start_solve(&mut self) {
        self.solve = Duration::ZERO;
        self.solve_start = Some(Instant::now());
    }

    /// Freeze the completed solve's elapsed time.
    pub fn stop_solve(&mut self) {
        if let Some(start) = self.solve_start.take() {
            self.solve = start.elapsed();
        }
    }

    /// Construction time so far.
    pub fn setup_time(&self) -> Duration {
        self.setup
    }

    /// Construction time plus the current or completed solve.
    pub fn total_time(&self) -> Duration {
        self.setup + self.solve_start.map_or(self.solve, |start| start.elapsed())
    }
}

/// Run a block as a named receipt phase (wall and calling-thread CPU time).
macro_rules! timeit {
    ($key:literal; $($tt:tt)+) => {
        let __cpu = $crate::receipt::cpu_start();
        $(
            $tt
        )+
        $crate::receipt::cpu_finish(concat!("cpu.", $key), concat!("wall.", $key), __cpu);
    }
}
pub(crate) use timeit;
