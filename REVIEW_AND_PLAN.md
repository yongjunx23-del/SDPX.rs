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

Further verified findings (cluster jobs 213629/213668/213669):

- Forcing 60 real iterations (tol=1e-300) on Λ=11 converges to the
  independently audited objective **-18.1686186450146** with audit
  primal 5.2e-37 / dual 1.7e-35 — the IPM solves the problem fine;
  only the premature acceptance was wrong.
- The converged solution itself has ‖x‖ up to ~8e78 (p50 ~4e11) —
  the huge scale is **intrinsic to the true solution**, not an
  initial-point artefact; unit column scaling only reduces it to
  ~1e57 (near-null direction survives coordinate changes).
- The iter-0 pass needs gap + primal + dual to hold *simultaneously*;
  the degenerate LSQ initial point coincidentally satisfies the gap
  check (qᵀx̃ ≈ -bᵀz̃ structurally). A unit interior start makes the
  gap check honest → forces real iteration.

### Fix: degenerate initial-point guard (0373c68) — VERIFIED

`solve_initial_point` now returns failure when the KKT initializer's
‖x‖/‖z‖ exceeds `1e12 × max(1, ‖b‖∞, ‖q‖∞, ‖A‖∞)` or is non-finite —
a catastrophic degeneracy signature (ising: ‖x‖~1e78 vs bound ~1e14).
`default_start` falls back to `unit_initialization` on any failed or
degenerate initializer (previously the solve result was ignored).
No termination criteria, tolerances or convergence checks change —
the guard only discards provably-broken starting points.

**Verified on cluster, Λ=11 768-bit, default settings (job 213671):**
`status=optimal, iters=69` (vs the false `iters=0`), returned point
audit **primal=3.2e-44 / dual=1.0e-41 / map=7.3e-11**, objective
`-18.1686186450146` matching the independent reference exactly. The
unit-interior start keeps the gap check honest, so iteration
proceeds to real convergence. 328 lib tests pass (incl. detector
unit test); Julia suite green.

**Sampled-factor route verified (job 213673):** `iters=69` on the
production `sampled_program` path — same convergence, 423s vs 1223s
materialized. Λ=15/Λ=19 sampled-route validation running (213674).

**Solution-scale column scaling control (job 213672):** scaling
columns by `max(|x̂_j|,1)` from a first-pass estimate makes the solver
see an `x′~O(1)` problem — `status=optimal, iters=96`, audit
**primal=4.8e-74 / gap=4.3e-76 / affine-map=6e-205**, ~30 orders
tighter than the unscaled run's 3.2e-44 (honest feas criterion forces
true absolute-residual convergence). Same objective to all digits.
This is the benchmark-side high-precision route if tighter residuals
are needed; the solver fix alone already passes the 1e-30 protocol.

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
- `bigint_syrk` port — **measured negative, reverted**: a full RNS pipeline
  (dyadic images → per-prime Barrett residues → exact f64 `dsyrk` →
  coefficient-form CRT, bit-exact vs `dot_fma`) was built and verified, then
  benchmarked pool-for-pool (8t): n=39/k=76 skipped on gate, n=80/k=150
  skipped, n=150/k=300 engaged at 641ms vs scalar 298ms (2.2× slower). The
  fused MPFR FMA already parallelizes perfectly; the residue phase's n·k·P
  folds plus the serial image/plan prefix never amortize below n≈300, beyond
  the solver's block sizes (n≈39–80). No production benefit → removed.

### Residual-family optimization (f67e3db, 3c2c651, 29eccc6) — VERIFIED

Cluster Λ=11 profile (403.5s solve) identified the residual product family as
the dominant cost: `residual.scale` ~69.9s + `residual.adj` ~66.4s timed,
with another ~30s+~70s inside `solve_raw`. Three changes, all preserving the
required factorized arithmetic:

- **Sampled adjoint diagonal pairs (f67e3db)**: for `r == s` pairs,
  `v_k = q'·S·q` with symmetric `S` reduces to a flat dot over the block's
  svec triangle against a precomputed constant `W[p,k] = c_p·q_i·q_j`
  (`c = 1` diagonal, `√2` off-diagonal). Replaces the `h²·kmax` panel product
  with `kmax·tri(h)` dots — 1.74× kernel-level, covering all `dim=1` blocks
  (every ising block). Serial and pooled paths share `wdiag`, bitwise parity.
- **Symmetric scaling products (3c2c651)**: in `PsdBlock::apply`, two of the
  four factorized congruence products produce symmetric outputs
  (`Rᵀ·X·R`, `R·S·Rᵀ` and the `Rinv` analogues). `pooled_gemm_sym` evaluates
  only the upper triangle (same ascending-k accumulation → bitwise identical
  entries) and mirrors the lower — `4h³ → 3h³` per apply. The factorized
  form stays (the earlier `G = R·Rᵀ` squaring failed all 7 graded tests and
  was reverted).
- **Schur per-column congruence (29eccc6)**: same `pooled_gemm_sym` on the
  streamed `Ginv·A·Ginv` product.

Λ=11 (np=1×8t, 768-bit): 403.5s → 374.5s solve (**7.2%**), `status=optimal
iters=69` identical. Phase deltas: `residual.scale` 266→195ms (−27%),
`residual.adj` 252→177ms (−30%), `residual` 278→205ms. All 328 lib tests
pass including the 7 `condensed_graded_*` extremes and parallel-equivalence
bitwise checks.

### Cone W-products + operator inner loops (8295aec VERIFIED, 6abfde8 pending)

`SDPX_PROFILE` timers on `update_scaling` (`PHASE cone_svd`) and
`step_length_psd_component` (`PHASE cone_eigmin`) exposed the true serial
budget (Λ=11, 768-bit): `cone_svd` 428s serial ≈ 53s wall (~16% of solve,
1932 calls × 221ms) and `cone_eigmin` 117s ≈ 15s wall (7728 × 15ms). The
solve-path cone products (`mul_Hs`, `step_component`, `Δs_offset`,
`combined_ds_shift` — ~11 `mul_Wx_inner` calls per block-iteration) were
previously uninstrumented (~48s wall estimated).

- **Symmetric cone W-products (8295aec)**: `mul_Wx_inner`'s second product
  (`Rx·tmp` / `tmp·Rx`) always produces a symmetric output consumed only via
  `mat_to_svec`. Now uses `pooled_gemm_sym` (moved to `matrix_math.rs` for
  sharing); the α/β-accumulate general path folds `α·entry + β·y` at the
  svec write. Λ=11: 374.5s → **352.3s** (`status=optimal iters=69`), −24s.
- **Operator inner loops via `dot_fma` (6abfde8)**: adjoint `wdiag` dots,
  forward `square` dots and r<s panel dots accumulated a fresh `MpFloat`
  per op via `mul_add`; `T::dot_fma` folds the same pairs in the same order
  through the in-place MPFR FMA accumulator — bitwise identical, less
  per-op overhead. Serial and split-chunk pooled paths both updated.

### SVD → symmetric-eig NT scaling — measured negative, reverted

`update_scaling`'s SVD of `L2ᵀ·L1` is the largest single kernel. A
single-eigendecomposition variant was implemented: `X = M·Mᵀ` (syrk) →
symmetric `eigen()` for U and σ², then `R = L2⁻ᵀ·U·Λ^{1/2}` by triangular
back-substitution (avoids the unstable `V = Mᵀ·U·σ⁻¹` recovery and the
second eig). All 15 graded tests passed at 128–2048-bit MPFR, but **all
f64 `basic_sdp` tests failed with `NumericalError`**: `X = M·Mᵀ` squares
the condition number (κ(X) = κ(M)²), which is exactly the pathology the
upstream comment warns about — at 53-bit the lost digits are fatal. The
SVD's κ-preserving bidiagonal QR is load-bearing; no safe eig-based
replacement exists at fixed conditioning. Reverted to direct SVD.

Remaining measured opportunities (Λ=11): `cone_svd` ~53s wall has no
known safe algorithmic substitute (eig variants square κ or need two
decompositions ≈ slower); `cone_eigmin` ~15s wall uses the proven
Sturm+RQI minimum-eigenvalue path; residual/operator products continue
to benefit from `dot_fma` accumulation.

### SVD inner-loop FMA fusion (14f44ee) — neutral, retained

`rotate`/`apply_reflectors`/`reduce_bidiagonal` element loops fused to
single-rounding `mpfr_fma` (rotate 6→4 ops, rank-1 update mul+sub→fma).
Λ=11: ~1s effect (≈0.3%) — per-op cost is dominated by `mpfr_fma`
internals, not op count. Retained for the strictly better rounding.

### Wide fixed-point dot (cedd1e9) — measured negative, reverted (2f15769)

Single-rounding accumulation of each product's 2N-limb significand into
a 4N+2-limb two's-complement window, falling back to ordered FMA on
ambiguity. Local microbenchmarks showed 6–25× — **but only because every
test operand came `from_f64` with ~2 nonzero limbs**; the inner loop's
`ai == 0` skips hid the N² cost. With dense 768-bit significands the
same code costs ~530–710ns/term vs `mpfr_fma`'s ~170–290ns (0.3–0.4×).
Cluster A/B on one node, identical 69-iteration trajectory:
cedd1e9 **1182.9s** vs 14f44ee control **345.6s** (~3.4× slower,
degrading with iterate conditioning). `mpfr_fma`'s limb-level inner
multiply-accumulate is already optimal for dense significands;
a same-model accumulator cannot beat it. Reverted; benchmark rule:
microbench operands must use full-precision dense significands, never
`from_f64` values.

**Next viable kernel lever is a different arithmetic model** — SDPB's
RNS path (encode significands into residues mod ~N·64/62 machine primes,
dot in u64, CRT-reconstruct once). Residue width needed ≈ window span
(~50 primes at 768-bit) caps the gain at ~2–4× on dot kernels; operand
encoding amortizes only when values are reused across many dots (gemm).
High implementation complexity for a bounded win — revisit only if the
dot kernels remain the bottleneck after profiling Λ=15.

### Thread scaling and pool-bound SVD inner parallelism (a77acd8) — retained

Λ=11 same-trajectory A/B (69 iters, one node, no code change between runs):
8 threads 345.6s → 16 threads 219.7s → **32 threads 169.7s** → 64 threads
172.2s. Scaling saturates near 32 threads (≈28 PSD blocks is the
block-level parallelism unit). A thread-local gate
(`sdpx_arithmetic::inner_parallel`) set inside `cone_parallel::apply`
leaves lets `reduce_bidiagonal`/`apply_reflectors` re-offer independent
column updates to the ambient solver pool (`par_chunks_mut`, ≥4-column
threshold, serial fallback). Scratch extended m→2m for the right-reflector
`ndot` buffer; 330 lib + all integration tests pass. Λ=15 t32 single-node:
251.0s → **241.8s** (75 iters, `status=optimal`).

### MPI dynamic loading — three bugs fixed, verified active; cross-node
### optimization stopped at the transport/duplication wall

`mpi.rs` runtime `dlopen` path had three latent bugs that silently disabled
the MPI world in every prior run:

1. `dlsym` probed `MpiComm_rank`-style names — real exports are
   `MPI_Comm_rank` etc.; `load()` always failed.
2. `resolve_handle` dereferenced `ompi_mpi_comm_world`/`ompi_mpi_byte`
   globals one level too deep — OpenMPI handles are the globals' addresses.
3. **`RTLD_LOCAL` was 4, which is `RTLD_NOLOAD`** — `dlopen` returned null
   for not-yet-loaded `libmpi` without setting `dlerror` ("unknown"
   diagnostic). RTLD_LOCAL is 0 on Linux. This was the decisive bug.

With all three fixed (580506a, d979c6c), mpiexec ranks report `size=2`,
`MPI_THREAD_MULTIPLE` granted, and per-site `mpi.gather*` timings appear.

### Cone-state sharding, cost-balanced partitioning and RNS gating
(42f1c6a, b79bfe8) — retained

Following activation, the remaining ~65% duplicated per-iteration work was
sharded: `CompositeCone` eval loops (`update_scaling`, `combined_shift`,
`margins`, `compute_barrier`, `step_length` bounds, `prepare_affine_bounds`,
`mul_Hs`) run each cone on its owning rank, then `allgatherv` the exchange
surface (PSD `R`/`Rinv`/`λ`; derived `Λisqrt`/`G` are rebuilt locally and
deterministically). Rank ranges come from `mpi::cost_ranges` — contiguous
blocks split at prefix sums nearest equal cost (PSD ~numel^1.5, cheap
cones unit cost), identical on every rank; the same partition drives
scaling products and the condensed-KKT Gram exchange so computation and
gather layout never disagree. Two fixes landed here:

- `step_z`/`step_s` are *inputs* to `combined_shift` in both prepared and
  unprepared modes — the sharded pack must seed them, not treat the buffer
  as write-only scratch (fixed; trajectory returned to bitwise match).
- `split_scaling` lanes are begin indices with `blocks.len()` as implicit
  end; a `0..=n` boundary list indexed past the last owned block (job
  213803 panic; fixed).

RNS batch products (`crates/arithmetic/rns.rs`): pseudo-Mersenne 60-bit
residue dots + Garner CRT, exact for finite fixed-precision values, MPFR
fallback otherwise. Microbenchmarks showed the raw residue dot beating
MPFR but plan construction (~407µs, Miller–Rabin per prime) and encoding
(~16ns/limb/prime) dominating; fixed by caching the prime table in a
`OnceLock` and a profitability gate that charges plan+encode+CRT against
the MPFR baseline — RNS now engages only where reuse amortizes it.

Λ=15 np=2 (2 nodes × 32 threads, `--bind-to none`, job 213804):
**optimal, 75 iters, 315.0s**, both ranks' `x` bitwise identical — vs
759.9s before cone sharding and 240.7s single-node. MPI remains a net
loss at Λ=15 (TCP/IPoIB transport, no verbs path engaged); the loader and
sharding are correct and retained for larger inputs where per-rank
compute dominates, but no speedup is claimed. `mpi/drive.jl` now calls
`MPI_Barrier`+`MPI_Finalize` (via `ccall` on the already-loaded libmpi)
before exit so no rank is killed mid-write. Cross-node transport
optimization and distributed KKT factorization were stopped per direction.

### Cluster state at task close

`SDPX.jl/current` → `releases/b79bfe8…/source` (MPI loader fixes, cone
sharding, cost-balanced partitioning, RNS gating, SVD inner parallelism);
Julia loads and links the rebuilt `libsdpx.so` (smoke test passed).
Release workspace `sdpx-releases/580506a` carries the same source and
built library; Λ=15 validated single-node (240.7s) and np=2 (315.0s).
Best measured: Λ=11 169.7s (t32), Λ=15 241.8s (t32, np=1).

### Single-node inner parallelism beyond the 28-lane cap (2026-09-20)

Λ=15's 28 PSD cones cap lane-level parallelism at 28; t64/t128 were flat
(243/240s vs 240.7 t32). Two gate levels now let spare workers absorb
intra-cone work (`inner_parallel` TLS pair in `arithmetic::inner_parallel`,
set by `ConeThreading::{inner_parallel,paired}` and applied at
`cone_parallel` leaves):

- **Heavy inner splits** (`workers ≥ 2×lanes`): column ranges of
  `xgemm`/`xsyrk`/`xsyr2k`/`pooled_gemm_sym`/`xpotrf` tail updates,
  `tridiagonalize` symv+rank-2, `form_q` columns re-offered to the pool.
- **Paired joins** (`workers > lanes`, one stealer suffices): the two
  cone Choleskys, `R`/`Rinv` products, and `dz`/`ds` step-bound
  `eigval_min` evaluations run as `rayon::join` pairs (second per-cone
  scratch set: `workvec2`/`workmat4-6`/`eig2`).

All splits preserve per-element accumulation order → bitwise identical.

Measured Λ=15 768-bit (jobs 213853–213869, `secs=`):

| build | t32 | t64 | t128 |
|---|---|---|---|
| a77acd8 baseline | 240.7 | 243.1 | 240.5 |
| +inner splits+pairing (ungated) | 248.6* | 228.2 | 192.4 |
| +gate (workers≥2·lanes) | — | 231.6 | 196.6 |
| +paired gate (df86cbe) | pending | pending | pending |

*The apparent t32 regression was a measurement artefact:
`prof_l15_t32.pbs` hard-pinned release `a77acd8`, so all "regressed"
t32 numbers (248.6/251.4/250.3) re-measured the baseline library on
sugon-queue nodes — the ~4% spread is node variance, not code. New
`sdpx_l15_t32.pbs` points at the live release.

**t128 OOM in job 213869** was a job-script bug, not solver: ppn=64 +
mem=96gb while running `-t 128` oversubscribed 2:1 and the 231MB setup
alloc hit the cgroup ceiling. Script now requests mem=160gb.

**One-sided Jacobi SVD — measured negative, reverted (c4a51c5).** To
break the serial QR bulge-chase inside `cone_svd` (~576s agg at t64),
a deterministic tournament-scheduled one-sided Jacobi path was built
(parallel Gram + column-pair rotations, QR fallback on non-convergence).
Local probe at 44×44/768-bit: Jacobi **5.92s vs QR 1.27s** (4.6× slower
before scheduling overhead); cluster `cone_svd` rose to 570–770ms.
High-precision orthogonality thresholds need too many sweeps at too
fine a task granularity. Reverted; `cone_svd` stays on the QR path.

**RNS weight-table encode (e7d87d5) — retained.** `encode` folds limbs
as `Σ limb_l·2^(64l) mod p` via a per-call weight table with u128
accumulation reduced every 8 limbs (~2× encode). Gate unchanged in
effect: `profitable` still rejects RNS at 44×44 (encode+CRT exceeds the
MPFR dot at this size) — verified by direct cost-model evaluation.
RNS pays only at larger blocks; the remaining real lever is an
iteration-scoped operand cache so encodes amortize across the ~6
products per cone-iteration — deferred pending Λ=19 sizing.

**SDPB reference curve measured** (same Λ=15 input, 768-bit):
np32=170s, np64=125s (1.36×), np128 (2 nodes) diverged reproducibly
(`maxIterations exceeded`, objective diverging — a deterministic
numerical failure, not transport noise). SDPB's single-node edge comes
from residual-domain `dsyrk` machine-word arithmetic on the *Schur-side*
matrices, not from more parallel units at the cone layer.

## Route re-evaluation (2026-09-20, post-probe)

Deep probes against the measured wall-clock model (Λ=15, t128,
189.7s/75it ≈ 2.53s/iter: kkt-update 820ms + kkt-solve 800ms +
scale-cones 410ms + step 152ms + residual 60ms) closed two
previously-plausible directions and reshaped the remaining plan:

**Probe A — KKT conditioning kills the f64-factor route.** Debug hook
`SDPX_DUMP_KKT=<dir>` (in `regularize_and_refactor`, after the diagonal
restore so the dumped matrix is the *true* one) writes the factored
CSC as f64. Measured cond: iter2 = 5.8e18, iter30 = 8.1e22,
iter50 = 2.3e34, iter74 = 3.0e54 (smin 1.7e-57 — at the f64 subnormal
floor). κ·ε_f64 ≫ 1 from iteration 2 onward: an f64 factorization of
this matrix carries zero correct digits. This is intrinsic to the
768-bit barrier at tol=1e-42, not a structural choice — it is also why
SDPB keeps its factorization in MPFR. **Closed.**

**Probe B — residue Gram breaks even at real dimensions.** Actual
sampled-block dims: `dim=1, basis_rows≈38 → side≈38, rank≤76`
(14 input blocks × 2 parities = 28 cones). Per-block per-iter Gram
work ≈ 25ms MPFR vs ≈23–29ms residue (encode Rinv ~4ms + residue
gemm/syrk ~13ms + CRT ~6–12ms). RNS does not reduce operation count —
it only relocates it; without SIMD residue kernels (~4× u64 lanes) the
constant-factor gap at n≈38–76 cannot amortize encode+CRT. The same
arithmetic dooms a residue-domain dense LDLᵀ: n³/6·30 primes ≈
130–260ms vs QDLDL's 196ms. **Both closed pending SIMD kernels.**

**Structural accounting corrections.** `kktsystem.update` hides a
*third* full solve per iteration (`solve_constant_rhs`); per-iteration
solve work = 3 solves × ~2.7 IR sub-solves, each with trsv (~13ms) and
a full-residual eval. The 66ms residual eval is the **sampled operator
A/Aᵀ products (2.07M nnz)**, not the 88k KKT matvec — already pooled,
≈30–50% headroom at best.

### Remaining levers (ranked)

1. **Warm-start MPFR Jacobi SVD** (probe before committing): the
   reverted cold-start Jacobi needed too many sweeps; restarting from
   the *previous iteration's* V (iterates move smoothly) or an f64
   skeleton of `M/‖M‖` should converge quadratically in ~4–6 sweeps.
   Output contract unchanged — Jacobi's convergence test *is* the
   MPFR verification; fall back to QR per-block when it fails.
   Targets `cone_svd` 289ms serial floor. Honest expectation:
   0–30% of that phase — sweep cost is the unknown.
2. **eigmin warm-start** — same pattern on the tridiagonal QR eig
   (~25ms/call → target ~5–10ms).
3. **Residual-path audit** — confirm the 2M-nnz sampled products are
   actually engaging the pool inside `residual()`.
4. **Λ=19 acceptance** — profile first; per-block sizes only grow to
   ~46–47, so no residue windfall is expected.

Closed permanently: cross-node transport (np2 = −31% measured, SDPB
np128 diverges), cold Jacobi, eig-for-SVD (κ²), f64 preconditioner
(κ≥1e18), residue Gram/Schur/LDLᵀ without SIMD.

Cluster validation status: t128 = 193.6s clean on an exclusive node
(job 213892); t32 = 251.9s on sugon queue (same as old library → node
variance, no regression); t64 pending resubmit (co-tenant memory
pressure caused repeated setup-stage OOMs — `mem=` requests set
RLIMIT_DATA/-m AND scheduler does not memory-pack, so co-tenancy is
unavoidable; mitigated by `naccesspolicy=singlejob` + mem=180gb).
