# SDPX development plan

Updated 2026-09-29. Rules: [AGENTS.md](AGENTS.md). Full experiment history,
job IDs and evidence paths: [docs/JOURNAL.md](docs/JOURNAL.md). This file keeps
only current state, open work and closed directions; update it in place.

## Goals

- Float64: audited medium/large solves within 1.3×/2× MOSEK time.
- MPFR: faster than SDPB at matched precision, tolerances and hardware
  (time to reach 1e-10/1e-20/1e-30 original-coordinate residuals).
- Memory: peak PSS/RSS parity with SDPB on matched thread counts (shared RNS buffers and replay compaction).
- Architecture: unified MPFR supernodal LDLᵀ replacing fragmented arrow variants.
- Reproducible serial, threaded and MPI results.
- 20–30k production lines, reached by deleting duplication.
- Fast development loop: focused tests, per-crate rebuilds in seconds.
- Ecosystem: official lightweight Python/Julia bindings and cluster-scale checkpoint/restart.

## Current status

Timings are release medians from one A–B–B–A batch unless noted. Cluster
(node9, AMD EPYC 7742) and Mac M4 timings are not comparable.

| Case | Latest result | Notes |
|---|---|---|
| Gravity LP, Mac | Float64 0.093 s; MPFR128 4.19 s (1.92 s at 4 threads); MPFR256 5.80 s | `local_bounds` backend + packed kernels. Float64 dual-residual audit fails (below) |
| Medium, Float64 | `Solved`/18 default; `Solved`/19 with `tol_dual_qnorm=1e-6` | Default: external dual residual 1.92e-6 > 1.75e-6, failure. With the opt-in: 6.1e-7, audit passes (cluster PBS 221978, audited locally) |
| Large, Float64 | Historical 129 s / 26 it (MOSEK 25.4 s) | Needs a fresh baseline |
| Ising11, MPFR512, node9 | 24.90 s (1 thread), 7.47 s (4) | `Solved`/52, audited; identical points across threads |
| CSDR3, MPFR256, node9 | 56.27 s (1 thread), 17.82 s (4) | `Solved`/57, audited; `local_soc_arrow` |
| Direct-support SOC, MPFR512, Mac M4 | 327.57 → 101.15 s on 8 threads (3.24×) | `Solved`/81, all four ABBA audits pass; `shared_soc_arrow`; RSS +19.7% |
| Lambda19, MPFR768, node9 | 422.1 s on 4 cores; peak PSS 912 MiB | `Solved`/119, audited. No new matched SDPB run |
| Lambda19 spins 0–50 | Historical 237.6 s (1 node) vs 238.7 s (2 nodes, TCP) | No two-node advantage shown |
| Lambda43, MPFR1216 | Pending | Corrected input generation held; 4/16/64/256-core SDPX vs SDPB 3.1.0 not run |

Lambda19 profile (4 cores): KKT update 40% (includes batched constant/affine
RHS; factorization alone ≈8%), PSD scaling 34%, KKT solve 12%. SVD rotation
replay is 62% of summed SVD time.

PMP converter (Ising11 XML, MPFR1216, whole command): 3.21 s serial and
0.63 s with 8 workers, vs SDPB 4.69 s / 2.52 s; peak PSS 6.8 / 40 MiB vs
126 / 160 MiB. Output matches SDPB below 1e-100 scaled difference.

Code size (2026-09-29 review, non-blank non-comment lines, tests excluded):
≈46.9k production lines vs the 41.5k recorded 2026-09-23 and the 20–30k goal.
Memory vs SDPB (Λ19, 4 cores, 2026-09-27 pilot): 1524 vs 445 MiB, with 119 vs
243 iterations and 2.4× less wall time; the 912 MiB above is the newer figure.

## Known failures (keep visible)

- **Gravity Float64:** `Solved`/17, original dual residual 4.01e-6 > 2e-6 gate.
  MPFR128/256 pass.
- **Medium Float64:** default settings still miss the audit (1.916e-6 > 1.75e-6);
  with `tol_dual_qnorm=1e-6` the solve is `Solved`/19 and the audit passes
  (r_d 6.14e-7, gap 2.41e-7). Confirms the normalization explanation below.
- **Float64 audit gate (review 2026-09-29, observed + hypothesis):** the audit
  tests `‖r_d‖∞ ≤ tol·(1+‖q‖∞)` (`benchmark/research/native.py`); the solver
  stops on `‖r_x‖/max(1,‖q‖+‖x‖+‖z‖)` (`default/info.rs`), looser whenever
  ‖x‖,‖z‖ ≫ ‖q‖. Hypatia misses the same gate (2.15e-6 vs 1.75e-6). Likely a
  normalization mismatch, not a solver defect; both stay listed as failures
  the opt-in criterion (F5, implemented) is applied to them. Gravity has not
  been rerun with it.
- **SDP_control3:** 29 iterations, primal residual ≈4.98e-4. Deferred.
- **MPFR `condensed_graded`:** one ignored regression (`condensed_graded_mpfr`);
  explicit G·X·G loses the graded tail. Float64 passes.
- **Normalized 2×2 PMP diagnostic:** `AlmostSolved`/24 at MPFR512, 1e-42.

## Build and test layout

- Frontends (CLI, C ABI, converter) compile Float64 + MPFR 128/256/512/768/1024
  by default. `all-precisions` enables every multiple of 64 up to 2048
  (needed for Lambda43 at 1216 bits; cluster builds must pass it).
- Input CSC format is validated once in `PreparedProblem` setup (and by the
  JSON reader and C ABI). Internal backends no longer re-check it.
- Solver integration tests are one binary: `--test it FILTER`.
- Dev profile: line-table debug info; dependencies at opt-level 2.
  Rebuild after touching the solver: 29 s / 12.5 s → 6.8 s / 5.7 s.
- `fast` profile for local end-to-end runs, `release` for quoted timings.

## Review findings, 2026-09-29

Source: whole-project review against Clarabel.rs 0.11.1, Hypatia and COSMO.
"Observed" = read in code or journal; "hypothesis" = needs a cluster A–B.
Line numbers drift; search the named function.

| # | Finding | Evidence | Work / done when |
|---|---|---|---|
| F1 | PSD scaling panics instead of failing cleanly | Observed: `psdtrianglecone.rs` `factor_right(..).expect("SVD error")`, `eigvals(..).expect("Eigval error")` in `margins` and `step_length_psd_component`; Cholesky failure just above returns `false`; `mpfr_svd.rs` returns `Err(1)` on non-finite input or non-convergence; scaling `false` already maps to `NumericalError` with accepted-iterate recovery | Return `false` from scaling (zero step in step length); one unit test with a forced SVD failure |
| F2 | Working tree not safely committable | Observed: 38 modified files, 27 staged renames; `kkt/ldl/arrow/shared_soc.rs` and `tests/it/main.rs` untracked but required; stash "svd+eigmin phase timers" from `main` | Checkpoint commit when the user asks; then drop or apply the stash |
| F3 | `MpFloat::exact_decode` is safe but unvalidated | Observed: `arithmetic/src/lib.rs`; `scalar_exact_decode` → `snapshot.rs` skips `wire::valid_regular` (only `read_wire` validates); corrupt snapshot = UB (feature `snapshot` only) | Make it `unsafe` or validate through the wire check |
| F4 | Presolve exact elimination has no work budget | Observed: `presolver.rs` `redundant_equalities` runs to completion, only `Exact::bounded` declines; only the ≤256-row proof has a budget. csdr3 setup 6.15 s, presolve 5.99 s (journal 2026-09-27) before that fast path | Operation budget with inconclusive fallback, receipt timer for presolve; unmeasured whether csdr3-like inputs still pay |
| F5 | Float64 audit gate vs solver normalization | See Known failures | Opt-in audit-aligned dual criterion (like `tol_feas_componentwise`, off by default). Needs approval: touches the convergence contract. Do not relax the gate |
| F6 | Curve search is Float64-only | Observed: `use_curve = precision_bits() <= 53` in `core/solver.rs`; the "regresses 512-bit Ising" comment has no journal entry and predates the certified binary64 λmin. Hypothesis: WΔWᵀ is linear in Δ, so M(t)=t·M_aff+t²(M_comb−M_aff); caching the two scaled directions makes a trial cost one λmin instead of two congruences per PSD block | Cluster A–B on ising11 (52 it) and Λ19 (119 it); keep only with ≥2% end-to-end and identical audits. Record the result either way |
| F7 | No checkpoint/restart | Observed: no save/load of iterates. Λ43 at 1216 bits on PBS will hit walltime; SDPB has it; `exact_encode` already round-trips MPFR values | Save (x,s,z,τ,κ,iteration) every N iterations plus `--restart`; reliability feature, expect no warm-start speedup |
| F8 | Three structure-matching arrow backends | Observed: `local_soc` 140, `local_bounds` 468, `shared_soc` 258 lines (SOC3-only, ≥8 groups, border ≤128) beside `arrow.rs` 1.5k | Decide before adding a fourth: general MPFR supernodal LDLᵀ with exact-dot kernels (what faer gives Float64) vs keeping special cases. High effort |
| F9 | Diagnostics inside numerics | Observed: `SDPX_DUMP_KKT` (`direct/solver.rs`) and `SDPX_DUMP_CONE` (`psdtrianglecone.rs`) duplicate the `snapshot` feature; `SDPX_TRACE_IR`; 32 `Instant::now()`; `cfg(test)` counters in `condensed/psd.rs` and `blas/mpfr.rs`; 61 `allow(dead_code\|unused…)` | Delete the two dumps; drop the allows and let the compiler list dead code; pinned points identical |
| F10 | Behaviour-changing env switches without docs | Observed: `SDPX_RNS_OPS` (`sampled/mod.rs`; off by default; last-ulp differences; absent from README and journal), `SDPX_FUSED_REDUCED`, `SDPX_SERIAL_QDLDL`, `SDPX_DIRECT_SOLVE` | For each: make default or delete; document any that remain |
| F11 | MPI branches inside the generic loop | Observed: 8 `World::get()` in `core/solver.rs`; `distributed/` 7k lines, `mpi.rs` 1.6k (45 unsafe, OpenMPI-only via dlopen; four identical `resolve_handle … None` arms) | A `Comm` trait with a no-op serial impl; folds into item 4. Chordal (3.1k lines) is upstream-identical: track via `provenance/upstream.json`, do not refactor |
| F12 | `sdpx-pmp` has no tests | Observed: 1.3k lines, zero `#[test]`, no dependent test; validated only by gitignored SDPB comparisons | Golden test on the README example (1−y+x², y*=1) at 128 bits, comparing the output directory and the solved value |
| F13 | CI coverage | Observed: ubuntu only; no macOS/Accelerate, no MPI job (3 `#[ignore]`d MPI tests), no `all-precisions` or `snapshot` build; clippy informational; floating `stable`, no `rust-toolchain.toml`; MSRV 1.85 untested | Pin the toolchain; add a macOS job and an `all-precisions` build check; MPI stays a cluster gate |
| F14 | Doc drift | Observed: workspace `AGENTS.md` says v0.7.0, crates say 0.8.0; `CHANGELOG.md` lacks `local_bounds`, `shared_soc` and the precision-feature change | Update both when committing |

Implementation status (2026-09-29; built and tested on the cluster, PBS
221970 and 221978, `~/projects/sdpx-review-20260929/`; lib 446, `it` 195,
ffi 13, pmp 3, arithmetic 36 passed with the pinned OpenBLAS provider):

- **Done:** F1 (SVD failure → scaling `false`; eigensolve failure → zero step;
  no forced-failure test, the SVD error path cannot be triggered from valid
  cone data), F3 (`scalar_exact_decode` validates), F4 (operation budgets and a
  `presolve_rank` receipt timer; no measurement yet), F5 (`tol_dual_qnorm`,
  default off), F7 (`--checkpoint FILE [--checkpoint-every N]`, `--restart FILE`
  and `Solver::{set_checkpoint,set_restart,check_restart}`; ordinary solver
  only; ising11 restart from the iteration-50 checkpoint solved in 2 iterations
  to the same objective), F9 (the two text dumps deleted), F10
  (`SDPX_FUSED_REDUCED` deleted; the others documented in the README),
  F11 (core loop no longer touches `mpi::World`; helpers in `mpi.rs`; a `Comm`
  trait was not needed), F12 (`crates/pmp/tests/convert.rs`,
  `tests/it/pmp_solve.rs`), F13 (macOS job, `all-precisions` and `snapshot`
  build checks; toolchain not pinned because the local toolchain is offline),
  F14 (changelog, workspace note).
- **Rejected by measurement — F6:** enabling the curve search at MPFR512 on
  ising11 (t1, ABBA, 2 runs each): 51 vs 52 iterations, medians 28.10 s vs
  27.16 s (+3.5%). One iteration saved cannot repay the trial step lengths, so
  the linearity/caching idea is capped below the 2% threshold. Keep Float64-only.
- **F4 measurement:** csdr3 (MPFR256, 4 threads, `Solved`/57, 17.1 s) now spends
  0.0035 s of its 0.47 s setup in presolve, so the 6 s seen on 2026-09-27 no
  longer occurs there; the budget only bounds worst cases.
- **F9 finished (2026-09-29, second pass):** removed `MatrixShape::t`,
  `SparseVector::dropzeros`, `sortperm` (and the tests that pinned them);
  test-only helpers are `#[cfg(test)]`; blanket `allow(dead_code|unused)` on
  modules and imports replaced by `cfg_attr(not(feature = "sdp"), …)` on the
  sdp-only items, as elsewhere in the crate. Unused per-file imports in the
  merged `it` binary removed. Cluster `cargo check`: sdp lib+tests, the
  workspace with tests, and the no-default-features lib all build without
  warnings (the no-sdp lib had 22 before). `cfg(test)` path counters stay:
  they prove parallel branches ran in the equivalence tests.
- **F13 finished:** CI pins `dtolnay/rust-toolchain@1.98.1` (the cluster and
  local toolchain). No `rust-toolchain.toml`: the local offline rustup would
  try to download a toolchain named `1.98.1`.
- **F2 (commit) not done:** moving `slim-repo` onto the other agent's
  `2456c01` (so the commit would sit on `codex/shared-soc-workspace-20260929`)
  was refused by the permission layer. The working tree already contains
  `2456c01`'s code (`shared_soc_max_bytes`); its journal/plan text does not.
  The user decides how to commit.
- **Environment finding (observed):** the static `openblas-src` build
  (`sdp-openblas`, `cache`) is not safe for concurrent BLAS calls from the solver
  pool: with it, 9 Float64 pooled-equivalence tests and 3 residue-BLAS tests
  fail (also on the committed HEAD), and a mutex around the kernel's `dgemm`
  makes the residue tests pass. The pinned provider
  (`~/projects/sdpx-medium-20260921-dense01/providers`, dynamic `libopenblas`,
  `RUSTFLAGS="-L native=$P -l dylib=openblas"`, features
  `sdp,blas-src,lapack-src,faer-sparse`) passes everything. Cluster builds and
  tests must use that provider, not `sdp-openblas`. Solver results from a static
  `sdp-openblas` build with more than one thread were not audited here.

Deferred: Gondzio-style multiple correctors. Each costs one KKT solve (~12% of
an iteration) plus a full step-length evaluation, so it must save >20% of
iterations; revisit only if F6 succeeds.

## Active g0 cluster campaign (2026-09-29)

Deploy the tested shared-SOC source snapshot `03dd5ae2c096d1593a1e92d59935dbf9692939bb`
in a new immutable cluster release. Build/MPFR512 original-coordinate validation:
PBS `221952.node220`, 8 cores (retry after the original node rebooted).
Follow-on PBS `221954.node220` reserves 64 cores/96 GiB for
at most four hours, four workers with 16-thread solves. Fifth-order upper and
rational third-order lower searches use 32/64/128 positive midpoint nodes,
then one additional tail Bernstein subdivision at 128 nodes; change one axis
at a time. Settings: 512 bits, feasibility 1e-40, gaps 1e-30, external conic
checks 1e-30, outward integration 1e-4 per spin. Two successive node changes
below 1e-3 are a finite quadrature gate, not global hierarchy optimality.
Rank only exact-tail/full-domain certified candidates. Also integrate disjoint
tail-spin savings for the existing lower candidate. Frozen input/source hashes
and all rejected rows live in the analytical project's
`numerical/cluster_soc_20260929/`; existing best certificates remain in force.

## Next work, in order

Fix F1–F4 first (small, no numerical-contract change); F5–F7 need a cluster
run or user approval.

| # | Work | Done when |
|---|---|---|
| 1 | Lambda43 cluster comparison | Corrected input frozen at 1216 bits; SDPX vs SDPB 3.1.0 at 4/16/64/256 cores with memory and audits |
| 2 | Hotspot optimization: SVD rotation replay & sampled RHS | ≥2% end-to-end on Lambda19 with identical audited points; loop vectorization and compact replay scheduling |
| 3 | Peak memory reduction & RNS buffer pooling | Thread-local RNS CRT accumulators and replay scratch dynamically pooled; close the 912 vs 445 MiB PSS gap on Lambda19 |
| 4 | Distributed scaling, MPI communication hiding & multi-node restart | Non-blocking collective overlap for sampled operators; extend F7 checkpoint/restart to MPI ranks |
| 5 | Code de-bloating & unified MPFR supernodal LDLᵀ (F8–F11) | Converge `local_soc`/`local_bounds`/`shared_soc` into a general supernodal elimination tree with exact-dot kernels; production lines back to 20–30k |
| 6 | SIMD vectorization for RNS prime fields | AVX-512 / ARM Neon intrinsics for batch modular multiply-accumulate across 64-bit primes |
| 7 | Ecosystem bindings & Dual Certificate tool | Python (PyO3) and Julia (MOI) SDKs over C ABI; automated continuous dual functional certificate verifier |
| 8 | Float64 performance (medium/large vs MOSEK) | Resume on request; unify dual residual stopping criteria with audit gates |

## Strategic Improvement Areas

### 1. Core Numerics & Memory Footprint
- **Dynamic RNS Buffer Pool:** Transition from independent per-thread allocations to a thread-safe shared/dynamic pool for CRT accumulators and encodings, preventing linear memory inflation on 64/128-core runs.
- **SVD Rotation Replay Optimization:** SVD replay accounts for 62% of SVD time (and 34% of overall solve time). Optimize memory layout for rotation logs, vectorize inner loops, and eliminate redundant intermediate scratch buffers.
- **Unified MPFR Supernodal LDLᵀ:** Retire ad-hoc backends (`local_soc`, `local_bounds`, `shared_soc`) in favor of a general-purpose MPFR supernodal LDLᵀ factorizer equipped with exact-dot RNS kernels (analogous to Faer for Float64).

### 2. Parallelism & Distributed Scaling
- **Asynchronous Communication-Computation Overlap:** Pipeline sampled PSD operator updates with MPI collective reductions (`MPI_Iallreduce`) to hide network latency on multi-node runs.
- **Cluster Distributed Checkpoint/Restart:** Extend the single-node checkpoint mechanism (F7) to cluster MPI runs with partitioned rank states, ensuring fault-tolerance for multi-day jobs on supercomputers.
- **Vectorized Modular Arithmetic:** Implement AVX-512 and ARM Neon intrinsics for RNS modular integer multiplication and Barrett/Montgomery reduction.

### 3. Codebase Architecture & Maintainability
- **Code De-bloat (Target: 20–30k lines):** Strip redundant abstractions, obsolete benchmark harnesses, and duplicate linear-algebra helpers.
- **Clean Separation of Distributed Logic:** Replace inline `mpi::World` checks throughout solver core routines with a unified zero-overhead `Comm` abstraction.
- **Robust Error Propagation:** Replace all remaining `.expect()` / `.unwrap()` panics in deep numerical routines (e.g. SVD/eigenvalue failures in PSD cones) with clean error signals and graceful fallback.

### 4. Ecosystem & Toolchain
- **Python & Julia First-Class Bindings:** Build `sdpx-py` (via PyO3) and `SDPX.jl` (via MathOptInterface) wrapping `sdpx-ffi`, simplifying integration into physics workflows (e.g., `Blocks.jl`, `PyCFTBoot`).
- **Bootstrap Dual Certificate Verification:** Provide automated tools to extract, interpolate, and certify continuous non-negativity of dual functionals over $x \in [0, \infty)$.

## Closed directions

Do not reopen without new evidence; details are in the journal.

- Lower MPFR precision, Float64 factorization of MPFR systems, relaxed
  refinement tolerances, NaN pivot clamping, eigenanalysis of MᵀM instead of SVD.
- SVD/eigenvector warm starts; scalar SVD/eigenvalue substitutions without a
  complete-solve gain; forcing small sparse systems into dense factorization.
- Float64 step tuning, panel fusion, column reorder/alias schemes, fused-recovery
  GEMV removal, four-GEMM PSD application.
- Uniform MPI layouts, cross-rank splitting of one block, the ordinary per-site
  MPI path, excessive task splitting.
- Refinement-pass fusion, allocator tuning, intra-cone parallelism based only
  on microbenchmarks.
- Zero-pair SVD replay shortcut; PMP zero-term trimming; Serde `collect_str`
  output wrappers; residue-BLAS Schur assembly on tall LP panels (memory).

Cluster notes: OpenMPI over TCP (`--mca btl self,vader,tcp`); `openib` hangs.
Avoid node70.
