# Changelog

All notable changes to SDPX are recorded here. Versions follow
[Semantic Versioning](https://semver.org/). Measurements quote audited full
solves; the experiment log is [docs/JOURNAL.md](docs/JOURNAL.md).

## [Unreleased]

g0-min reproduction SOCP (106,175 SOC3 cones, fixed multipliers): Float64
solve 2.02 s/50 it (objective error 4.5e-7) → 9 ms/0 it (6.6e-14), whole CLI
0.06 s against MOSEK 11.1.3 0.12–0.14 s internal; MPFR256 26.7 s/33 it →
0.09 s, exact to 3e-77 against a 600-bit reference.

### Added

- Presolve merges second-order-cone tail coordinates whose rows of `A` are
  zero: `(s1, s_V, s_C) ∈ K` iff `(s1, s_V, ‖b_C‖) ∈ K`; a fully constant tail
  becomes the orthant row `s1 − ‖b_C‖ ≥ 0`. Postsolve lifts `s`, `z` (and
  certificates) back to the original cone.
- Presolve fixes a variable without quadratic terms whose only nonzero lies
  in a single-entry orthant or equality row when its multiplier `−q_j/a` is
  admissible; row and column leave, `q_j x_j` is an objective constant
  reported in both costs. Problems that reduce completely return in 0
  iterations.

### Performance

- Gondzio multiple centrality correctors (approved 2026-10-07) on orthant
  rows, second-order cones (spectral) and τκ, for symmetric problems with
  orthant or SOC rows: parity set iterations 993 → 849; larger gravity
  37 → 22 (−16% native); csdr3 MPFR256 57 → 36; sched SOCPs now all Solved.
  Pure PSD problems are unchanged.
- Iterative refinement of the constant and affine right-hand sides shares
  one residual pass and one two-column correction solve (−3–4% on the free-λ
  g0 SOCP and gravity).

- Binary64 local/shared SOC arrow Schur assembly streams each leaf once with
  fixed 1024-leaf partials summed in chunk order (thread-count invariant):
  84 → 4.4 ms per factorization on 106k leaves, border 16.

### Changed (regularization contract; approved 2026-10-07)

- The shared-SOC arrow eliminates each leaf's cone block before its single
  primal column, whose pivot `P_jj + aᵀH⁻¹a` is positive; that column keeps
  its true diagonal instead of the static shift. Any factorization failure
  escalates as before and restores the shift. Synthetic free-multiplier g0
  SOCP: 60 → 15 iterations, wrong objective (6e-5) → optimal.

### Code structure (bitwise-identical solves)

- The IPM driver is SDPX's own staged loop (`evaluate`, `terminate`,
  `scale`, `direction`, `step_length`, `finish`) with an explicit
  `IterationState` and control-flow `Flow`; the Hypatia-style curve search is
  a `CurveSearch` value instead of loose options. All 49 parity cases (46
  Float64 benchmark problems, ising11 MPFR512, csdr3 MPFR256, g0 cases) give
  bitwise-identical x, s, z, status and iterations.
- SDPX banner and presolve report (constraints removed, variables fixed,
  cones reduced); crate authors are "SDPX contributors"; upstream attribution
  moves to `NOTICE` and `provenance/`; upstream personal TODO notes removed.

### Removed

- `shared_soc_max_bytes` setting: the shared-SOC arrow uses the common
  512 MiB arrow workspace cap (the former default). Settings files that set
  it are rejected as unknown fields.
- Runtime re-validation of solver-built KKT shapes and pivot signs in the
  arrow and dense-block backend constructors (now `debug_assert!`).

### Build

- Without an `sdp-*` provider feature, `build.rs` links Accelerate on macOS
  and the system dynamic OpenBLAS on other Unix targets (`SDPX_BLAS_LINK=none`
  defers to RUSTFLAGS, `OPENBLAS_LIB_DIR` adds a search path); `faer-sparse`
  is a default feature, so a plain `cargo build` produces the tuned backend.

## [0.9.1] - 2026-10-06

Mixed Λ27 at 1024 bits, 32 threads per node: full solve on 4 nodes 1369 s
(0.9.0: 1608 s; SDPB 2592 s), 1 node 2106 s (SDPB 4014 s). Same-allocation
3-iteration A/B of the release head against the pre-repair head: 4 nodes
on par, 1 node +2.4% (removed refinement prediction, partly offset by
residue contributions).

### Fixed

- MPFR sparse products (pooled lanes, MPI-sharded products and residual
  products) now use exactly the CSC gemv arithmetic. In 0.9.0 a solve's
  point depended on the thread count for sampled MPFR problems.
- The aligned MPI exchange republishes the scaled values of rows that hold
  sampled linear entries outside zero blocks; before, non-owner ranks read
  zero there and refinement had to correct the first solve.
- Rollback to the accepted iterate refreshes residuals before the reduced
  convergence test; checkpoint readers reject nonfinite or nonpositive
  homogenization/scaling values; cone dimensions and parameters are
  validated once for all frontends (the C ABI keeps its last result when an
  update is rejected); empty arrow borders are handled under MPI.
- Refinement no longer predicts stalls across right-hand sides or
  factorizations (each right-hand side refines on its own residuals, as in
  Clarabel). This costs some MPFR solve time on one node.
- Sampled block files parse within `--threads` instead of every core.

### Performance

- Exact residue products for generic arrow leaf contributions when a rank
  holds at least one leaf per worker (Λ27 1 node: contributions 35 → 22 s).

### Performance

- Generic arrow leaves skip border columns they are not coupled to (Y
  columns, Schur contributions and solve dots), cutting refactor work about
  10% on mixed Λ27; points are unchanged.
- Sampled JSON block files are parsed in parallel (Λ11 load 0.30 → 0.07 s).
- MPI: condensed scaling and the sampled products share one block
  partition; prepare/recover/residual keep rank-local rows and exchange once,
  and the sampled linear part is sharded. Mixed Λ27 (1024 bits, 4 nodes ×
  32 threads, same allocation): 3 iterations 124.8 → 117 s (−6%).
- MPI cone partition balances measured per-cone scaling cost (SVD CPU per
  rank 148–202 s → 168–182 s), and sampled Grams are exchanged as packed
  upper triangles (`sync` 7.7 → 4.9 s); together about −3% more.
- Arrow border solve splits its forward sweep over the pool and leaf
  back-substitution splits its row couplings (both bitwise identical):
  Λ27 4 nodes 113–115 → 109 s, 8 nodes 103–104 → 97–98 s.
- Arrow contributions update the border Schur in parallel inside one pass
  over all owned leaves (no serial per-leaf apply), and a leaf whose factor
  outweighs an even per-thread share splits it over the pool. Bitwise
  identical; Λ27 1 node 192 → 181 s, 4 nodes −3%.

## [0.9.0] - 2026-10-04

### Performance: SOC elimination, dense panels, distributed arrow (2026-10-04)

- The condensed form eliminates second-order cones of dimension ≤ 16 through
  the explicit NT factor W⁻¹, so orthant and small SOC rows form one Gram
  BᵀB (dense-column SYRK plus a sparse pair plan; exact per entry at MPFR).
  The condensed shared-SOC arrow is no longer selected there; the augmented
  shared-SOC arrow is unchanged. g0 medium: Float64 0.20 → 0.024 s (MOSEK
  0.029 s), 256-bit 6.65 → 2.55 s, 256-bit audit still accepted.
- Binary64 products with A use a dense BLAS panel for columns at least a
  quarter full, in residuals and condensed solves, at every thread count.
- MPFR CSC products accumulate each long output row/column exactly, rounded
  once; MPFR panel-only orthant rows use a cached exact residue congruence.
- The generic arrow computes leaf contributions on the fly instead of storing
  them, parallelizes leaf columns, sizes its cap by KKT storage, and under
  MPI distributes leaves across ranks (border summed in rank order; exact
  refinement residual rows split by work). Λ27 1024-bit: ~670 → ~50 s per
  iteration on one node.
- Sampled operators fold unmatched one-dimensional blocks lying in orthant
  rows into the linear rows instead of dropping the factored operator.

### Simplification (2026-10-04)

- Removed the diagnostic `SDPX_TRACE_IR` and `SDPX_SERIAL_QDLDL` environment
  switches. Receipt timers use one `receipt::start/finish` form throughout.
- Sampled forward/adjoint pair kernels are shared between the serial and
  split paths instead of being written out per path; points are identical.
- `docs/ARCHITECTURE.md` now describes the design only; measurements live in
  the plan and journal.
- Removed the opt-in `snapshot` feature (KKT capture/replay, the `kkt_replay`
  example and `Scalar::scalar_exact_{encode,decode}`) and the `bench` feature
  with its test-only micro-benchmarks. Neither was part of any workflow.
- Removed dead internals: `DirectLDLSolver::offset_values` and the QDLDL
  method behind it, `Variables::rescale`, the unused `MultiplyGEMV`/`xgemv`
  chain, the unused `SVDEngineAlgorithm` selector, the default
  `KKTSolver::solve_many` column loop, the `AutoDirectLDLSolver` and
  `SpecializedLDL` wrappers, and the `cfg-if` and ffi `serde_json`
  dependencies. LDL backend selection now lives in `kkt/ldl/config.rs`.
- Removed leftover `#[allow(unused...)]` attributes and redundant nested
  blocks in the CLI; the build is warning-free with and without `faer-sparse`.
- Primitive `Scalar` math (`mul_add`, `sqrt`, ...) is `#[inline]` across the
  crate boundary. Release (thin LTO) already inlined it; non-LTO builds (the
  `fast` profile, downstream crates) ran the Float64 Schur kernels through a
  call per FMA. Results are bitwise identical.
- Consumed conic and sampled JSON transfer owned inputs into setup; CSC
  validation is shared, and exports stream through a buffered writer.
- KKT storage has one authoritative CSC. Dense-block refactors read it,
  condensed assembly writes its primal prefix, and dense Schur destinations
  are computed instead of stored. Exact residual rows retain incoming mirrors
  only. Sampled RHS snapshots and generic arrow contributions store one triangle.
- MPFR SVD reuses existing input/output storage, including caller Vt for
  compact tall/square factors. HSD solutions use their batch output directly,
  with RHS conic slices reused as step scratch.
- SVD reuses the shifted-QR quotient, and Ruiz skips norm work on an empty
  quadratic matrix. Both remove redundant computation with identical points.
- MPFR empty quadratic forms skip zero arithmetic for finite operands;
  nonfinite behavior and Float64 arithmetic are unchanged.
- Equality plus one MPFR orthant reuses the existing parallel cone chunks.
  Matched release gravity256/four-thread solves improve 5.70% small and 4.25%
  larger, with identical points and original-coordinate audits passing.
- Float64 local-bound leaf solves use disjoint solution slots for forward
  intermediates, removing their separate RHS work buffer.
- Parallel arrow refactors accumulate diagnostic regularization counts
  without a per-leaf counter vector and use `try_for_each` without collecting
  unit results.
- Leaf coupling scratch stores only its coupled suffix. Exact refinement
  residuals borrow and restore the mutable point instead of copying it.
- Higher-order sampled inverse adjoints write directly to output, removing
  their persistent quadratic buffer. Scalar projections and fallback writes
  are preserved; a768-bit matrix SDP matches its full accepted point/audit.
- CRT prime workers borrow one immutable output-index slice, removing their
  cloned index arrays. Ising512 complete points and original audit match.
- CRT accumulation compacts dead product residues in place, removing its
  separate residue buffer. Full Lambda19/768 and gravity256 points remain
  identical, with bound original-coordinate audits passing.
- Single-product diagonal scratch and balanced residue multiplication improve
  larger gravity256 by3.41%; small improves0.92%, Ising512 regresses0.66%.
  All12 matched full points and bound original audits pass.
- Declined slice-dot accumulations follow the generic dot's filtered FMA
  fallback, preserving its nonfinite and underflow zero behavior.
- Exact-dot classification handles regular pairs first. Matched gravity256
  release solves improve2.87% small and regress0.41% larger; full points and
  original-coordinate audits are preserved.
- Input validation stays at construction boundaries; sampled setup trusts its
  internally built CSC. Exact norms use the slice accumulator without pointer
  staging. PSD scaling reuses its second input for the lower factor, and leaf
  single-RHS buffers allocate only when called.
- Sampled scalar adjoints skip unused matrix panels. PSD scaling reuses dead
  SVD singular-value scratch for rounded roots shared by R and Rinv.
- Sampled setup releases numerical A before allocating initial residual and
  variable work, reducing their overlap without changing persistent storage.
- Sampled condensed setup also releases duplicate retained coefficient values
  after installing the factored operator; its CSC pattern remains available.
- Narrow MPFR scalar FMA uses exact stack arithmetic with one rounding;
  matched release gravity256 runs improve by 2.24% on the smaller case.
- Narrow two-product FMMA improves matched release Ising256 by 2.87%, with
  identical points. PSD synchronization copies only its authoritative triangle.
- MPFR eigenvalues-only requests omit unused reflector storage. PMP generated
  bases store only the Cholesky triangle; converted output stays identical.
- Exact diagonal congruences pack upper partial residues and bound CRT groups,
  reducing active scratch while retaining exact accumulation and one rounding.
- Diagonal congruences use the actual scaling spread instead of a duplicate
  fixed reserve. Matched gravity256 solves improve3.48% small/4.22% larger,
  with full audited point identity. Row blocks use one exact product and
  tile-sized scratch, removing the unused second buffer.
- Owner MPI steps reuse constant-RHS slack scratch for offsets; the frozen
  two-rank solve preserves complete points and original-coordinate audits.
- PMP conversion writes checked decimal coefficients directly and reuses one
  output row buffer per block, reducing allocations with identical output.
  Unchanged reduced prefactors and sample scalings borrow the original values,
  removing repeated parsing and duplicate vectors.
- Bound Schur diagonals reuse solve scratch, removing a duplicate MPFR
  vector and recurring Float64 temporary allocations.
- Float64 bound residuals borrow contiguous primal ranges, and one-RHS panel
  products use GEMV. Matched release gravity solves improve at unchanged
  convergence rules and accepted original-coordinate audits; see the journal.
- New [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md); README, AGENTS.md and the
  plan point to it. Performance decisions and pending trials stay in the plan
  and journal.

### Earlier unreleased changes

- New KKT backends `local_bounds` (LPs with bound rows; packed faer kernels in
  binary64) and `shared_soc_arrow` (SOC3 leaves around a small shared border).
- Frontends compile Float64 plus MPFR 128/256/512/768/1024 by default;
  `all-precisions` restores every 64-bit width up to 2048.
- Integration tests are one binary (`--test it`).
- PSD scaling reports a non-converged SVD as `NumericalError` instead of
  panicking; a failed eigensolve in the step length gives a zero step.
- Presolve's exact redundant-equality elimination has an operation budget.
- Removed the `SDPX_DUMP_KKT`, `SDPX_DUMP_CONE`, `SDPX_FUSED_REDUCED` and
  `SDPX_RNS_OPS` switches (the last was slower where measured); the remaining
  diagnostics use `SDPX_RECEIPT` and `SDPX_PROFILE`.
- Checkpoint/restart: `--checkpoint FILE [--checkpoint-every N]`,
  `--restart FILE`. A file restarts the same problem exactly, or hot-starts a
  nearby problem of the same structure (e.g. the next point of a scan).
- Opt-in `tol_dual_qnorm`: audit-aligned dual residual `‖r_d‖∞/(1+‖q‖∞)`.
- MPI: the MPICH ABI (MPICH, Intel MPI, MVAPICH) is supported alongside
  OpenMPI.
- `sdp-openblas` builds OpenBLAS with `USE_LOCKING=1`; the default
  single-threaded build was not safe for concurrent calls from the solver
  pool and could give wrong multithreaded results.

## [0.8.0] — 2026-09-26

A performance release for large high-precision SDPs (conformal bootstrap),
multithreaded and multi-node runs, and for nonsymmetric cones and binary64.
Unless noted, changes keep results bitwise identical to the previous build.
Every non-bitwise change was accepted only after independent audits.

### Performance: high-precision SDP

- **Exact residue-number-system BLAS kernel** (`rns_blas`). It serves
  congruences, Schur products and sampled-operator products. Products
  accumulate exactly across primes and round once. Features:
  - thread-local `gemm` packing;
  - streamed CRT;
  - prime-range splitting for heavy blocks.
- **Iterative refinement:**
  - Corrections predicted to stall are skipped, measured per factorization.
  - Residual evaluation is skipped when a stall is predicted.
  - A persistent stall floor carries across factorizations.
- **Accuracy fix for large bootstrap SDPs:** equilibration bounds now scale
  with precision, and the inner refinement uses an exact residual. Λ19 spins
  0–50 now solves at 768 bits; it failed before.
- **SDPB-style per-block workers** when threads outnumber blocks. Pool
  workers are pinned when the affinity mask matches.
- **Arrow LDLᵀ:**
  - When the pool has spare threads, the leaf refactor, the L⁻¹B panels and
    the dense leaf factor run over columns in parallel.
  - A single large dense leaf is admitted to the arrow backend; it no longer
    falls back to scalar sparse LDL.
- **Less serial work on the control thread** (it was 20–23% of wall time;
  now roughly halved). Now on the solver pool:
  - long vector operations;
  - linear-part products;
  - componentwise error accumulation;
  - initialization cone margins.
  The exact inner residual is partitioned by entry count.
- **Certified binary64 step-length eigenvalue** (not bitwise; audited). The
  PSD step length needs only λmin. It is computed in binary64 with a
  rigorous backward-error bound and falls back to full precision when not
  certified. This removes the 768-bit eigensolve from the step phases.
- The SVD rotation replay, the largest cone-scaling kernel, is offered to
  idle workers. Its rotation log and replay buffer are reused per thread.
- **Allocation-free kernel reuse.** The residue kernel reuses per-thread
  buffers (encodings, scratch, CRT accumulators), and a split call encodes
  operand rows in parallel. On glibc Linux the CLI raises malloc's `mmap`
  and trim thresholds, so large buffers are not remapped on every call:
  Λ19 at 32 threads −4.8%.
- **Exact info norms** (not bitwise in reported norms): for MPFR they are
  computed from exactly accumulated squares.

### Performance: nonsymmetric cones and binary64

- **Exponential and power cones.** Every backtracking trial is screened in
  binary64 with a rigorous error bound. Only near-boundary trials evaluate
  the high-precision `log`/`exp`. With 64 or more nonsymmetric cones, step
  lengths run over cones in parallel, then are confirmed. Barrier
  evaluations run in parallel and are summed in the original order. At 256
  bits: power-cone problem 7.4 → 3.8 s, exponential 2.25 → 1.92 s.
- **binary64 KKT.** With the `faer-sparse` feature, high-fill factorizations
  (≥1e8 flops) use faer's supernodal multithreaded LDLᵀ; small ones keep
  QDLDL. On a sparse LP: 24.2 → 3.7–4.8 s.
- The vector pool also serves native floats (vectors of 65,536 entries or
  more).

### Fixes: binary64 accuracy

Three binary64 paths had inherited choices made for MPFR speed; each lost
accuracy near convergence. MPFR results are unchanged (bitwise).
- The condensed PSD scaling applies `H` and `H⁻¹` through the NT factor, in
  two congruences, instead of the squared `G = WᵀW`. Squaring costs cond(W)²
  of accuracy.
- Iterative refinement no longer predicts stalls in binary64. As upstream,
  each right-hand side stops on its own convergence or measured stall. A
  skipped correction left an unrefined direction from an ill-conditioned
  factorization, which broke primal feasibility.
- `Δs` uses the cones' `Hs·Δz` in binary64, not the KKT solver's product.

Results:
- The `dim2_signed_parities_ruiz_f64` integration test passes again.
- The Float64 `medium` benchmark now ends `Solved` in 18 iterations (was
  `AlmostSolved`, 19). Its external dual residual improves from 2.85e-6 to
  1.92e-6, but still exceeds the audit's 1.75e-6 (see Known limitations).

### MPI (owner-partitioned path)

- The layout balances a cubic work estimate: LPT, then move/swap
  refinement.
- Fused reduced-refinement passes: one all-gather per inner pass instead of
  two synchronizing rounds.
- Each collective's site / operation / length agreement is now one vector
  allreduce; allreduce count −85%.
- New `all_gather` collective.
- Collectives called from a non-control thread abort with a backtrace.
- Transport note: on clusters where OpenMPI's `openib` BTL loses or
  corrupts messages (observed with OpenMPI 4.1.4 without UCX), use
  `--mca btl self,vader,tcp`. SDPX's wire checks report corrupted payloads.

### Diagnostics

- Receipts record per-phase process CPU and control-thread CPU (`cpu_s`,
  `serial_s`) and per-site MPI entry time.
- `SDPX_RECEIPT_ALL_RANKS=1` writes one receipt per rank.
- `SDPX_TRACE_IR` traces inner refinement passes.
- Default-start stages and SVD stages are timed.

### Structure

- The repository publishes only the solver: benchmark inputs, harnesses and
  archived plans are no longer tracked (the tracked tree drops from 11.6 MB
  to 2.8 MB). The README is rewritten around Rust usage. A new
  `bootstrap` example solves a `pmp2sdp` directory at 768 bits.
- Solver modules were reorganized into a flat, responsibility-based layout,
  with dead code removed. The owner-partitioned MPI implementation is gated
  on the `sdp` feature.

### Results (768 bits, audited)

| Problem | 0.7 | 0.8 | SDPB |
|---|---|---|---|
| Λ19, 32 threads | 318 s | 102–106 s | 204 s |
| Λ19, 64 threads | — | 102 s | 153 s |
| Λ19 spins 0–50, 1 node × 52 threads | fails | 238 s | 328 s (64 ranks) |
| Λ19 spins 0–50, 2 nodes, 16 ranks × 8 threads | — | 239 s | — |

### Known limitations

- High-precision LP/SOCP are limited by serial sparse LDLᵀ: faer does not
  support MPFR.
- On the bootstrap SDPs above, two nodes are not faster than one well-used
  node. Per-block phases stop scaling beyond about two threads per block.
- The Float64 `medium` benchmark fails its external 1e-6 audit narrowly
  (dual residual 1.92e-6 > 1.75e-6), although the solver reports `Solved`.

## [0.7.0]

Initial Rust release of SDPX: native CLI, Rust API, versioned C ABI, MPFR
arithmetic, sampled SDPB input and the owner-partitioned MPI path.
