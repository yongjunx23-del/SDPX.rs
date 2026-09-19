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
  sequence labels) and replayed in the **exact serial `y_idx` order** —
  per-position merge lists, trunk columns eliminated inline at their path
  positions (bit-identical even for interleaved leaf/trunk trees, commit
  69a5f40).
- Forward solve: leaf groups in parallel, trunk contributions replayed in
  exact serial column order with trunk columns scattering inline at their
  positions. Backward solve: trunk serial first, leaf groups in parallel.
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

### Cluster acceptance (UCAS HIAS, verified 2026-09-19)

Release `69a5f40` deployed under `~/projects/SDPX.jl/releases/`, `current`
promoted after all gates passed (jobs 213600 validation + 213601 scaling):

- **Validation gates** (PBS compute node, 8 physical cores pinned):
  Julia package tests 46s ✓, analytic 2×2 PSD ✓ (fixed stale driver —
  Clarabel `Ax+s=b` convention + unscaled MOI triangle entries),
  sampled Ising Λ=11 512-bit ✓.
- **Ising thread scaling** (solve time, fresh process per width,
  `OPENBLAS_NUM_THREADS=1`):

  | threads | solve_s | speedup |
  |---|---|---|
  | 1 | 60.97 | 1.00× |
  | 2 | 33.11 | 1.84× |
  | 4 | 18.96 | 3.22× |
  | 8 | 11.36 | 5.37× |

- **Audits**: `accepted=true`, `optimal=true` at every width; gap 3.35e-43,
  reference-objective agreement 4.6e-35 (gate 1e-30), 50 iterations.
- **Cross-width determinism**: solution vectors bitwise identical at
  1/2/4/8 threads (sha256 `251f8938ff9296d0` on x/y/s).
- Receipt: `releases/69a5f40…/metadata/acceptance.json`; raw logs in
  `results/213600.node220/` and `results/213601.node220-scaling/`.

### Cone-pool extension to LP/SOCP + iteration-loop serial phases

Following the full `timeit!` phase decomposition (IP iteration: kkt solve
4.21s, kkt update 2.14s, scale cones 1.94s, step lengths 1.39s, mu+info
0.28s, residual/combined/iterate ≈0.5s on Ising 512-bit):

- `step_length` / `prepare_affine_bounds` now parallelise **all symmetric
  cones** (PSD, SOC, Nonnegative, Zero), not only PSD: `psd_step_lanes` →
  `sym_step_lanes`, bounds evaluated at the common cap and folded in cone
  order — bitwise identical because symmetric-cone step lengths only use
  the cap for clipping. Nonsymmetric cones (Exp/Pow/GenPow, cap-sensitive
  iterative searches) stay serial in the fold's second pass. This was the
  missing path for LP/SOCP — `step_length` was 100% serial for problems
  without PSD cones.
- Single large orthant (pure LP): `NonnegativeCone::step_length_parallel`
  uses the existing `orthant_chunk` elementwise split; chunk partial minima
  fold in index order (min is associative — bitwise identical). The fold
  contract is one shared α = min(αz, αs) for both components.
- `Info::update_with_pool`: the eight independent residual-norm scans
  (`x/z/s/rx_inf/Px/rz_inf/rz/rx norm_scaled`) run concurrently on the cone
  pool — each scan keeps its serial reduction order → bitwise identical.
- `Variables::add_step_with_pool`: x/s/z axpby on disjoint elementwise
  chunks — per-element identical expressions → bitwise identical.
- `timeit!` sub-timers now cover the whole IP loop (residual update,
  mu+info, affine/combined rhs, step lengths, iterate update) and
  `SDPX_PROFILE=1` dumps the timer tree at solve end.

Measured (this machine, 8 threads):

- LP 512-bit, single orthant m=8000 (augmented KKT, QDLDL stays serial —
  chain-like etree): 4.37s → 2.35s; mu+info 651→159ms, iterate update
  129→23ms; w1/w8 solution vectors bitwise identical.
- Ising Λ=11 512-bit: 11.05s → 10.77s; mu+info 275→68ms, iterate
  57→24ms; solution bitwise identical.
- float64 LP end-to-end identical at w1/w8 — all changes are `T`-generic;
  the structural cost model (MIN_LANE_WORK scaled by limb count) limits
  f64 parallelism to appropriately larger problems.

Correctness fixes made during this round:

- **Committed bug (69a5f40)**: `trunk_row` cleared per-group `trunk_delta`/
  `d_delta` *after* the empty-`row_cols` early return — a leaf group absent
  from a trunk row's path replayed stale sequence tags from earlier rows
  (out-of-bounds panic or silent wrong-position accumulation). Buffers are
  now reset for every group before path partitioning; detected by
  `pooled_condensed`/`overlapping_memory_fallback` tests.
- `cone_parallel::prepare_affine_bounds`: immutable `z`/`s` slices now
  split alongside `dz`/`ds` on lane recursion (right-half lanes read wrong
  rows otherwise; f64 masked it, MPFR exposed it).
- `step_length_parallel` returns the shared `(α, α)` pair the caller's
  fold contract requires.

New tests: `orthant_step_lengths_{f64,mpfr256,mpfr512}` (chunk path +
mixed symmetric lanes, bitwise vs serial incl. signed-zero caps); the
renamed `psd_step_lengths_*` (now `sym_*` internals) still pass.

### MPI rank-sharded cross-node parallelism (2800be0, 21b4cd1, 03b236d)

Optional cross-node execution via dynamically loaded MPI (`mpi.rs`,
`dlopen`/`dlsym` — no Cargo dependency, no lock change). A `World` is
detected only when `mpiexec` advertises >1 rank or `SDPX_MPI` is set;
otherwise every path falls back to the identical local code. MPI is
initialised with `MPI_THREAD_MULTIPLE` and each collective site uses a
dedicated duplicated communicator (`SITE_FORWARD/ADJOINT/SCALING/GRAM/
RX/RZ`), because `residual_inner`'s `rayon::join(products, scaling)`
can run collectives concurrently.

Sharded work is partitioned by disjoint block/output ownership; each
output is computed wholly on one rank, then republished by
`MPI_Allgatherv` in deterministic block/row order — every value keeps
the serial arithmetic, so results are bitwise identical across rank
counts:

- sampled operator forward/adjoint products (`sampled.rs`) — forward by
  disjoint row spans; adjoint terms gathered per block in block order
  (column ranges may overlap), empty blocks emit zero segments;
- NT scaling products (`condensed_scaling.rs`) — block row spans;
- sampled Gram updates (`condensed_kkt.rs`) — only the owning rank runs
  `update_with_pool`, Grams republished in full-block-index order;
- condensed-KKT sparse gemvs in `solve_raw`/`residual_inner`
  (`condensed.rs` + `SparseParallel::product_sharded`) — the LP/SOCP
  IR-residual cost, output-partitioned with rank-local pool splitting;
- non-dominant Gram updates now also parallelise at block level locally
  (03b236d), removing the inner_sampled serial tail.

Verified on the cluster (OpenMPI 4.1.4, job 213606/213612):

- Λ=11 768-bit: np=1 vs np=2 (2 nodes × 8 threads) — solution vectors
  **bitwise identical** (sha256 of x/s/z), rank0 == rank1 identical.
- Λ=15 768-bit: np=1/np=2/np=4 (4 nodes × 8 threads) — solution vectors
  **bitwise identical** (sha256 `0aa72c2e…`).
- Note: these SDPB Ising inputs report `optimal` at iteration 0, but
  the independent audit **rejects** the returned point (Λ=11:
  primal 0.352 / dual 0.874; Λ=15: 0.366 / 0.945; equality rows are
  exact, PSD rows carry ~46 abs residual). So the runs verify MPI
  bitwise determinism only — they are **not** acceptance evidence.
  np=2 costs ~+9s of redundant setup/default_start + gather latency.
  Rank sharding pays off only when iterated residuals/Gram products
  dominate.

### Λ=11/15 iter-0 false convergence — verified root cause

External audits falsify the iter-0 "Solved" result. Verified chain:

- The returned point **is** the internal iterate (equilibration off →
  unscale is identity → same objective -22.59, same 0.335 residual).
- `res_primal = ‖r‖/(‖b‖+‖x‖+‖s‖)` (Clarabel formula): ‖r‖₂ ≈ 3e3 but
  ‖x‖₂ ≈ 2e78 — the initial-point KKT solve piles ~1e78 components
  onto near-degenerate columns (input coefficients span ~1e-156…1e1),
  so the relative residual collapses to ~1.5e-75 < any tolerance.
  gap_abs ~1e-103 and res_dual collapse the same way; ktratio=1 →
  `Solved` at iteration 0.
- Ruled out by direct experiment: presolve OFF, equilibration OFF,
  chordal OFF — all identical; materialized CSC route ≡ sampled-factor
  route (same -20.2169 objective, same residuals); pre-change 69a5f40
  and the prep01 audit show the same failure — **not a regression**
  from the parallelisation/MPI work; the input was never truly
  accepted (prep01 audit: accepted=False).
- MPFR-768 `static_regularization_constant ≈ ε^(3/4) ≈ 1e-173` is far
  below the degenerate data scale, so regularization does not bound
  the initial point's ~1e78 components.
- Local synthetic repro confirms both ends of the mechanism
  (1e-80-col → iterates to dual_infeasible with x~1e80; ≤1e-120 →
  regularization bounds x and it converges properly).
- This is upstream-Clarabel-inherited normalization behaviour on
  pathological conditioning, not an SDPX logic bug — do not "fix" by
  loosening the termination formula (contract). The Λ=11/15/19
  pmp2sdp conversions are outside the solver's current conditioning
  envelope; acceptance must use inputs that iterate (587-class or
  better-conditioned conversions).

## Pending

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
