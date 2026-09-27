# Changelog

All notable changes to SDPX are recorded here. Versions follow
[Semantic Versioning](https://semver.org/). Measurements quote audited full
solves; the experiment log is [docs/JOURNAL.md](docs/JOURNAL.md).

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
