# Float64 conic10 comparison

This adapter reuses the retained `/tmp/sdpx-performance-goal-20260913/final-selection`
corpus, reference runners and source/provider fingerprints, with a sequential
process-group watchdog. It does not copy input data or historical results into this project.
`sdpx_runner.jl` is adapted from that campaign's Julia runner: only package-specific
settings, result receipts and the direct `solve_conic` call change. Its original-
coordinate external oracle and timing scopes are preserved. Upstream benchmark
attribution and licenses remain in the stable sibling `SDPX.jl/benchmark`.

Freeze project sources, native library, Julia environments and reference harness
before running. The Julia environments need JSON and AppleAccelerate in addition
to their respective SDPX package. Use Julia 1.12.6 and the workspace depot recipe.
Run all commands sequentially, with distinct empty output directories. The fixed
protocol is ten cases, one thread, 200 iterations, internal full tolerances 1e-8,
external gate 1e-6, first call plus three warm fresh solves, 900 seconds/4096 MiB
per process group. Clarabel.rs and new SDPX use their default Ruiz equilibration,
presolve and chordal decomposition. MOSEK retains product defaults
except for one thread, including its default tolerances. These settings differ:
report actual settings and the common external accuracy gate alongside timings.

```sh
export SDPX_BENCH_BLAS=accelerate
export SDPX_LIBRARY=/path/to/frozen/libsdpx.dylib
export SDPX_EXPECTED_SOURCE=/path/to/frozen/new/julia/SDPX.jl
python3 run.py --reference /tmp/sdpx-performance-goal-20260913/final-selection \
  --engine new --project-source /path/to/frozen/new \
  --sdpx-env /path/to/new-env --rust-bin /path/to/frozen/reference-binary \
  --rust-source /path/to/frozen/reference-harness --output /tmp/new-results
```

For phase35 use `--engine old`, its frozen environment and `SDPX_EXPECTED_SOURCE`
pointing to `/tmp/sdpx-performance-goal-20260913/frozen/phase35/source/SDPX.jl`.
For reference legs use `--engine rust_clarabel` or `--engine mosek`; retain the
same required environment/binary arguments and specify `--mosek-python` if needed.
The retained two-arm binary is under `final-selection/rust-harness/builds/` in
directory `9e074c9a83405675bde4fa5a2ee9c5a69bafa798bafa340d0d633450430f2a90`.
Use its `target/release/sdpx-bench-harness` and adjacent `build_receipt.json`.
The wrapper selects `reference_default` for Clarabel.rs even if the shell retains
`SDPX_RUST_ARM=controlled` from an older experiment. Use
`--engine rust_clarabel --reference-arm controlled` only for an explicit diagnostic
with all three preprocessing features disabled. Keep its outputs separate from
default results. A disabled-preprocessing failure does not establish failure of
the default solver; report solver status separately from the external accuracy gate.
New SDPX likewise defaults to `--sdpx-arm default`; use `--sdpx-arm controlled`
only for a separately labeled diagnostic. Its direct-solve runner checks the
effective preprocessing settings returned by the frontend.

Parsing and external checks are excluded from solve timing. Each repetition builds
fresh cone/settings/input objects and invokes a fresh direct solve. The solve scope
includes native setup, solve, result retrieval and handle cleanup. First-call and
warmed end-to-end timings are separate; setup/solve component medians preserve the
historical all-repetition convention. No prepared-handle reuse occurs. Only full
optimal status plus all external residual gates passes. Preserve all raw failures,
timeouts and resource-limit receipts; they receive no speed credit.

The wrapper owns process supervision. A timeout, RSS limit or watchdog error
remains the primary outcome even if cleanup fails; cleanup errors are recorded
separately. Unconfirmed cleanup stops the campaign before another launch and
writes `supervision_abort.json`. Custom controllers that import this module
must install `suite.run_owned = owned_supervisor(suite.group_rss_kib)` before
optionally wrapping it with `with_native_highwater`.

The retained reference harness supports exact one-thread Accelerate execution only.
For candidate-only measurements, `--engine new --threads 1|2|4|8` selects the native
cone/KKT budget while Julia remains single threaded and native BLAS stays at one
thread. The wrapper sets `RAYON_NUM_THREADS` to the candidate budget for Faer.
The factorization's configured width must equal the requested budget for Faer or
one for QDLDL. `backend_threads` reports this width, not busy CPU cores or measured
utilization. The actual cone pool size may be smaller than requested for small
workloads or a single cone. Requested budgets and factorization/cone/BLAS receipts
accompany the raw results; these are not total process thread counts. Timing and
CPU measurements are needed to establish scaling efficiency.

Add `--native-highwater` for reliable OS high-water RSS on macOS/Linux. A small
owned probe waits for the exact solver child with `wait4`, records its native
high-water RSS in `logs/*.resource.json`, and preserves the child's stdout, stderr,
exit code and signal exit. Each aggregate row includes this as `native_highwater`.
The helper is fingerprinted with the working driver. macOS reports bytes; Linux
reports KiB, converted explicitly to bytes/MiB. This is the single solver process
high-water mark across startup, parsing, all repetitions and external checks, not
solver-only memory or a simultaneous process-group maximum. These runners do not
launch solver subprocesses; native threads belong to the measured child.

The existing `peak_rss_mib` remains the sampled process-group maximum and includes
the small probe process when enabled. Its 0.2-second polling can miss short-lived
reference peaks; cached values with this limitation must not be relabeled as OS
high-water measurements. Watchdog interruption can kill the probe before `wait4`
returns: then `native_highwater.complete=false`, and the original timeout/resource
failure remains intact. Missing native receipts are incomplete memory evidence,
regardless of a solver's separate accuracy result. Solver-internal timing scopes
remain unchanged; process wall time includes the probe's startup overhead.

Run the full one-thread candidate first. Threaded timing comparisons require the
same accuracy outcomes; failures receive no speed credit. Reuse historical
reference results with their original dates, source/input/provider identities and
resource outcomes rather than calling them fresh measurements. The working driver
can target a frozen production project through `--project-source` and `SDPX_LIBRARY`;
its own files are fingerprinted separately from the frozen project.

For a direct Rust-core comparison, `build_kernel.py` (Python 3.11+) adapts the
retained Rust harness in a disposable output directory. Pass `--reference` for
the harness source, `--source` for the frozen project, `--identity` for its file
hash manifest, and `--output` for a new build directory; then build its Cargo
manifest with `--release --locked --offline`. The script checks dependency
versions against the frozen lockfile. Inputs, external oracle and fresh
setup/solve timing are reused; source and actual backend receipts identify SDPX.
Its release profile matches the retained Rust harness. Optional
`SDPX_BENCH_PHASE_TIMERS=1` prints existing phase timers after timed solves.
The generated SDPX harness defaults to the `reference_default` preprocessing arm
(the upstream-derived core defaults); pass `--arm=reference_default` explicitly
when running from a shell that may retain an old `SDPX_RUST_ARM` value.

Report direct Rust memory separately from Julia-interface memory: the latter
includes Julia, package loading and JIT compilation. Also compare actual
factorization names; a change from QDLDL to Faer is a backend change, not evidence
of frontend overhead or pure thread scaling.
