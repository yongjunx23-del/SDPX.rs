# SDPX review and plan

Consolidated plan file (replaces `PERFORMANCE_PLAN.md` and
`PLAN_LEDGER_20260918.md`, both retired 2026-09-19). Contracts: [AGENTS.md](AGENTS.md).
Benchmark protocol: [benchmark/research](benchmark/research/README.md).

## Architecture

One Julia frontend (`julia/SDPX.jl`), one Rust solver core (`crates/solver`),
one native FFI library (`crates/ffi`), arbitrary-precision arithmetic
(`crates/arithmetic`). The retired standalone `SDPX.jl` solver is out of scope.

## Done (2026-09-19)

### Cleanup and file simplification

- Split oversized sources: `condensed.rs` 2860→855 (`condensed_kkt.rs`,
  `condensed_psd.rs`, `condensed_scaling.rs`, test files), `mpfr.rs` 2422→692
  (`mpfr_decomp.rs`→`mpfr_svd.rs`+`mpfr_eigen.rs`, test files), `ffi/lib.rs`,
  `sampled.rs`→`sampled_split.rs`, `csc/core.rs` (bare `#[test]` fns into
  `#[cfg(test)]` module — they were compiled into the production lib).
- Removed unreachable code: `pardiso-*` features + `pardiso-wrapper` +
  `pardiso.rs` (ABI never selects `"mkl"`/`"panua"`), `buildinfo`/`vergen`
  (sole consumer was its own test), `sdp-r` (no R consumer), `lazy_static`
  (→ `const static`), `mp512_probe.rs` (undeclared, unbuildable).
- Kept `sdp-netlib`: it is the frozen Netlib-PIC build recipe used by
  `benchmark/research/comparison.pbs` on the cluster (Linux only).

### QDLDL elimination-tree parallelism

The MPFR condensed Schur solve is the dominant high-precision cost (measured
24% serial tail: refactor 30ms + trsv 3.1ms per call on the Ising Λ=11 512-bit
reduced matrix). Its elimination tree is an arrow: 11 leaf chains (the PSD
blocks) converging on a 20-column trunk (equalities). Implemented
`qdldl/parallel.rs`:

- Symbolic plan built after symbolic factorization: junction nodes and their
  ancestors form the trunk; other columns group by leaf subtree. The `L`
  pattern is validated (leaf columns only reference own-group leaf rows or
  trunk rows; trunk columns only trunk rows) — unsupported patterns fall back
  to the serial kernels.
- Phase A: leaf groups factor independently in parallel (private `ColStore`
  + workspaces, serial order within a group → bit-identical).
- Phase B: trunk rows run sequentially; each row's leaf-column segment is
  split per group (private `y_vals`, bucketed trunk deltas, D deltas with
  sequence labels) and replayed in the serial `y_idx` order → bit-identical.
- Forward solve: leaf groups in parallel, trunk contributions replayed in
  column order, trunk tail serial. Backward solve: trunk serial first, leaf
  groups in parallel.
- Pool wiring: `cones.thread_pool()` → `DirectLDLKKTSolver::set_factor_pool`
  → `ldlsolver.set_pool` → `QDLDLFactorisation` (default no-op `set_pool` on
  the `DirectLDLSolver` trait; other backends unaffected).
- Memory guard: `groups × n > 500_000` cells or `<2` groups → serial plan is
  not built (general sparse KKTs can produce thousands of tiny subtrees;
  per-group `O(n)` scratch would dominate memory — this caused an 86GB OOM
  during development and is now bounded).
- Escape hatch: `SDPX_SERIAL_QDLDL=1` skips plan construction (A/B timing).

Measured (Ising Λ=11, 512-bit, 8 threads, this machine):

- refactor 30ms→13ms, trsv 3.1ms→1.1ms, solver 12.35s→11.01s (~11%),
  `status=optimal`, iters=50, solution **bitwise identical** to serial
  (verified on x/s/y vectors).

Tests: `parallel_factor_solve_bitwise_identical`,
`parallel_fallback_and_fork` (qdldl::parallel::tests) — bitwise L/D/Dinv and
solve equality vs serial, arrow and fork structures, serial fallback.

## Pending

- Cluster: deploy release build, run Ising Λ=11 512-bit acceptance
  (`benchmark/ising/run.jl`, 1e-30 audits) at 1/2/4/8 threads; record
  factorization/trsv phase timings and thread scaling.
- Unrelated pre-existing warnings (`cached_psd`, `prepared`, `has_lanes`,
  `product` dead code) — not in scope, flag for follow-up.
- `julia/SDPX.jl/deps/build.log` untracked artifact — hygiene.

## Evidence base

- Local 322×322 Schur / 50-iteration profile: refactor 1530ms, residual
  1446ms (already parallel), IR+trsv 1020ms, assemble 63ms of 12.5s.
- Elimination tree dump (now a one-line `QDLDLSTRUCT` summary under
  `SDPX_PROFILE`): 11 chains → trunk cols 322–341.
- SDPB reference: `bigint_syrk` RNS batching (MPFR→fmpz residues→double
  GEMM→CRT) for its Gram/Schur products; Elemental distributed Cholesky/Trsm
  for its Schur solve. SDPX's single-node analogues are the existing pooled
  assembly and the new etree-parallel QDLDL kernels.
