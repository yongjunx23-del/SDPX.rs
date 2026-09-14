# SDPX performance plan

## Goal and acceptance

Optimize the Julia frontend / Rust core / native numerical-library implementation
without reducing precision or numerical reliability. Aim for Float64 performance
near or above Clarabel.rs and MOSEK, and high-precision SDP performance near or
above SDPB. These are measured targets, not promised speedups. Keep one simple
engine, reuse mature implementations, and preserve upstream attribution.

Preserve declared arithmetic precision, original-coordinate outputs, accepted-
iterate recovery and Clarabel.rs convergence, infeasibility, regularization and
linear-solve refinement. Per the latest user instruction, remove SDPX-only
five-equation runtime verification/correction and extra convergence gates.
Keep essential FFI memory/ABI and supported-precision validation. Numerical
accuracy remains qualified by external tests, outside solver timing. Do not use mixed precision, precision-ladder warm
starts, approximate rank reduction, benchmark-name branches, or looser accuracy
gates. Legitimate arithmetic reordering need not be bitwise identical, but must
pass the same numerical contract. Do not restore a production certificate stage;
benchmarks validate returned points and rays outside solve timing.

Direct solves default to Ruiz, presolve and chordal decomposition. Reusable
handles retain Ruiz and disable structural preprocessing for q/b updates. Keep
the user-authorized upstream reduced tolerances; AlmostSolved is not a full
optimal result and earns no speed credit in the full-accuracy comparison.

## Autonomous research loop (2026-09-15)

The user asked for this work to run as an autonomous research goal with no
wall-clock limit: iterate continuously — hypothesis, implementation in an
external candidate copy, focused tests, measurement, keep or discard, ledger
entry — until the user stops, rather than stopping at the first milestone. The
loop contract and its data/evaluator identity live in `benchmark/research/`
(`README.md`, `program.md`, `catalog.json`, `run.py`, `evaluate.py`).

Development speed is part of the contract, so the default iteration is short
(target at most about ten minutes):

1. Only the test targets affected by the change, for example
   `cargo test -p sdpx-solver --features sdp-accelerate --lib` with an optional
   name filter plus the single affected integration test file. The full Rust
   workspace suite, both Julia runtime legs, and the regression/holdout/MPFR
   and Ising protocols are milestone acceptance, not per-iteration checks.
2. `run.py pair --profile screen`: a predeclared three-case development subset
   (`LP_afiro`, `SOCP_sambal`, `SDP_truss1`), one cold plus one warmed fresh
   solve, a single AB block, one thread, and clamped 120 s process and 600 s
   campaign caps. Screen verdicts are `screen_pass` / `screen_fail` and can
   never be `keep`.
3. The full protocol (four repetitions, AB and BA blocks, 1/2/4/8 threads,
   regression, holdout, MPFR, Ising) keeps every existing gate and remains the
   only source of speed credit, exactly as specified in the sections below.

Screen measurements guide the search only; no performance claim in this file
may rest on them.

### Open work list from the 2026-09-15 review

The independent review in `/tmp/sdpx-review-20260915/REVIEW.md` (review ID
`5564232504c3b01632c545209d98873bcc6b4d07ff782d4735c4ffd92e068a70`, snapshot
matching production frozen27 `77e07fc7…`) lists four P1 numerical items. They
are the current work list and take precedence over further performance work:

1. `mpfr.rs` pivoted tridiagonal solve stored the wrong second superdiagonal
   (`T = [[0,2,0],[2,3,1],[0,1,4]]`, `rhs = [1,0,0]` returned a first entry of
   `-5/8` instead of `-11/16`). Under repair.
2. `mpfr.rs` `sturm_count` mishandled a zero pivot, so `d = [0,0]`, `e = [1]`,
   `x = 0` counted 0 instead of 1; the same count drives both interval
   isolation and the RQI acceptance test, so a later QL fallback does not make
   the path correct. Under repair.
3. `psdtrianglecone.rs` used `eig(MᵀM)` for NT scaling at every precision above
   53 bits, truncating negative eigenvalues to zero and producing non-finite
   inverse square roots on an ill-conditioned packed SPD input family. The
   resolution (restore the direct SVD, or keep a fast path with an explicit
   condition/error acceptance test and direct fallback) is under oracle review.
4. `condensed.rs` applied the PSD Hessian through cached `G = RRᵀ` and
   `Ginv = RinvᵀRinv` for high precision instead of the factorized form; the
   two-product congruence drops an `O(t²)` term that the four-product
   evaluation keeps, and the refinement-residual path shares the same apply.
   Resolution under oracle review.

Items 3 and 4 were part of the frozen27 speedup, so any restoration of the
factor-defined arithmetic must be re-measured, not assumed to be free; the
plan's existing retention rule (a stable ≥2% median gain under comparable
conditions, with no accuracy regression) still governs.

## Current evidence

- Frozen27 adds three high-precision-only (`T::precision_bits() > 53`)
  per-iteration optimizations on top of frozen26; Float64 keeps every
  original route. (a) Condensed PSD Hessian apply uses cached
  `Ginv = RinvᵀRinv` and `G = RRᵀ` as a two-product congruence instead of
  the factorized four-product form; this is the precision-gated version of
  the variant previously rejected under Float64 (`AlmostSolved` regression),
  and that rejection still stands for unrestricted use. (b) PSD NT scaling
  replaces the SVD of `L2ᵀL1` with an eigendecomposition of `MᵀM`,
  reconstructing `U = MVΣ⁻¹`; this squares the condition number, which is
  affordable at ≥256-bit precision and is gated off for Float64 for that
  reason. (c) PSD step length requests only eigenvalue index 1 through
  `xsyevr` range `I` (Sturm isolation and inverse-iteration polish in the
  MPFR tridiagonal stage); Float64 keeps full-spectrum `syevr` because
  ulp-level λ_min changes can move borderline step lengths. Source SHA256
  `77e07fc7…`, library `a09f2399…` (receipts in
  `/tmp/sdpx-rust-acceptance/source-27.json` and
  `validation-27-state.json`). Rust workspace qualification: 31 suites,
  0 failures (rust-27.log); Julia 1.12.6 and 1.13.0 legs pass
  (julia-27, julia113-27 state files). Ising512 paired A/B in both
  execution orders, all 32 points accepted and optimal at unchanged
  512-bit/1e-42/1e-30 settings and 50 iterations, objective agreement
  4.623e-35 bitwise across arms. Order base→candidate: w1 51.62→44.29 s
  (1.165x), w8 12.29→11.37 s (1.081x). Order candidate→base: w1
  52.91→46.30 s (1.143x), w8 12.41→11.50 s (1.079x). Single-arm
  diagnostics during development: apply-only 1.15x/1.21x (w1/w8),
  NT-eig-only ~1.17x w1, λ_min-only 1.09x w1. Per-iteration profile after
  the frozen26 eigensolver shows block-level H⁻¹ applies (affine,
  corrector, refinement residuals and the constant-RHS solve) account for
  roughly half of the 0.93 s/iter; the three changes attack that block.
  Receipts: `/tmp/sdpx-combined-candidate/ab-results{,-rev}/`,
  `/tmp/sdpx-rust-acceptance/frozen-27/`. Iteration trace, scaling,
  line-search and convergence semantics unchanged.
- Frozen26 replaces the MPFR symmetric eigendecomposition's cyclic Jacobi
  iteration with Householder tridiagonalization and implicit QL iteration
  (Wilkinson shifts, packed reflector storage, explicit iteration budget with
  failure return). Only `mpfr.rs` and `tests/mpfr_dense.rs` differ from
  frozen24; the condensed G/Ginv two-GEMM variant was rejected after a Float64
  `AlmostSolved` regression and reverted byte-for-byte. Source SHA256
  `c6e4321d…`, library `ea2adcbe…` (full values in
  `/tmp/sdpx-rust-acceptance/source-26.json` and `validation-26-state.json`).
  377 Rust checks and Julia 1.12.6/1.13.0 qualifications pass. Microbenchmark:
  eigvals 3.7–9.2x faster at n=12–40 for 512/768 bits; SVD constant-factor
  lower at n=14–16, unchanged at n=40 (GEMM-dominated). Ising512 paired A/B
  in both execution orders: all 32 points accepted and optimal at unchanged
  512-bit/1e-42/1e-30 settings and 50 iterations; objective agreement 4.6e-35.
  Native medians improve from 70.09/66.97 s to 50.33/50.37 s at one worker
  (1.33–1.39x) and from 16.92/16.92 s to 12.96/12.91 s at eight workers
  (1.30–1.31x). Peak process-group RSS ~850 MB. Receipts:
  `/tmp/sdpx-eig-candidate-20260914/results-{24-26,26-24}/` and
  `/tmp/sdpx-rust-acceptance/frozen-26/`. This changes no convergence,
  line-search or scaling semantics; the iteration trace differs only in
  rounding order.
- Qualified integration: source SHA256
  `a52c22f3366d92ab2d69e2dcfe56487b364e4f34390ba727691326c02451ae79`.
  376 Rust checks; 508 each on Julia 1.12.6 and 1.13.0; optional PMP2SDP
  callback 8 and sampled extension 32. Receipts and frozen source:
  `/tmp/sdpx-rust-acceptance/`, candidate 24. The no-SDP configuration also
  compiles. This includes pooled sampled A/Aᵀ operations, balanced triangular
  MPFR SYRK tasks and parallel PSD step bounds.
- PSD step-length computations were still serial, including two eigenanalyses
  per PSD block per affine/corrector call. The current frozen candidate, SHA256
  `a52c22f3366d92ab2d69e2dcfe56487b364e4f34390ba727691326c02451ae79`,
  computes independent PSD bounds in the existing pool and preserves the
  original ordered cap reduction. Independent numerical-design/code reviews and
  full tests passed. On macOS, eight-thread native medians improve from36.55
  to17.07s; reverse order gives36.38/17.05s, a repeated2.13–2.14x gain.
  Single-thread medians remain about68s; configured eight-thread scaling rises
  from1.87x to3.97x. All24 forward/reverse points pass unchanged512/1e-42/1e-30
  settings and return identical decimal x/s/z vectors and50iterations. This is
  a configured-budget measurement, not physical-core cluster scaling.
  Receipts: `/tmp/sdpx-local-ising-step-length-20260914/repeated-comparison.json`.
  Final same-allocation cluster qualification job212627 completed on node7:
  all28 points pass, with native1/2/4/8-core medians135.00/74.17/42.46/30.24s
  versus fresh SDPB127/69/36/25s. Eight-core scaling is4.46x versus5.08x,
  leaving SDPX20.97% slower at8cores. Whole-invocation SDPX RSS615–676MiB;
  SDPB sampled aggregate group peak521MiB at8ranks, with different frontend
  scopes. All213 source files and the library remain pinned before/after.
  Receipts: `/tmp/sdpx-rust-ising-20260914-final24/complete-results/` and
  `/tmp/sdpx-final24-summary-20260914/summary.json`.
  This changes neither scaling nor line-search acceptance criteria.
- Implemented and qualified: Clarabel runtime checks, removal of SDPX-only
  direction/convergence gates, owned sampled factors with paired operators and
  NT Gram Schur, scalar PSD Ruiz maps, preprocessing fallback, prepared q/b
  updates, MPFR decomposition scratch reuse and selected FMA, cached weighted
  block partitions and dominant sparse-PSD Schur column parallelism.
- Generic sparse A/Aᵀ residual parallelism is qualified (including seven
  no-SDP-feature tests). It reuses the live pool and indices into current CSC
  values, preserving each output's accumulation order. Repeated A–B–B–A
  orthant campaigns show no stable 2% end-to-end gain; no speed credit is claimed.
- A subsequent few-cone LP experiment uncapped the existing pool beyond cone
  count. Candidate25 passed377 Rust tests,508 Julia checks per runtime and40
  PMP checks, but its32-point Float64/256-bit diagnostic found no qualifying
  benefit: eight-thread native changes were about+1.57%/-0.37% time. Since
  `max_threads` is an upper bound, this is not a configuration defect. The
  change was rejected and source24 plus its qualified library restored exactly.
  Do not increase resource use solely to report more active workers. Receipts:
  `/tmp/sdpx-few-cone-parallel-20260914/results-first/` and
  `/tmp/sdpx-rust-acceptance/candidate25-retention-decision.json`.
- Dominant sampled-block inner GEMM/SYRK and pair-entry parallelism passes
  numerical tests. Initial n=24 MPFR four/eight-worker slowdowns of6.36%/9.90%
  were not reproduced in reverse order (+0.81%/+0.68%); no stable end-to-end
  gain is established. Admission was therefore not tuned to that one result.
  Raw paired receipts: `/tmp/sdpx-dominant-sampled-20260914/`.
- A separately measured structural fix balances SYRK by cumulative triangular
  work, retaining the existing task budget, scalar accumulation order and pool.
  At n=64/128 and256/512bits, it improves the SYRK kernel by1.3–1.7x across
  two/four/eight workers; eight-worker scaling reaches4.5–4.9x. Both72-case
  diagnostics pass exact comparisons with matching input hashes and seven
  paired repetitions. This is kernel evidence, not an assumed solver speedup.
  Receipts: `/tmp/sdpx-sampled-inner-diagnostic-20260914{,-23}/`.
- Candidate 22 pools sampled A/Aᵀ across independent blocks with bounded lazy
  contribution caches and deterministic descriptor-order scatter. Review found
  no blocker; this change is retained in frozen24 and covered by its qualification.
- Frozen23 Float64 passes4/10 conic10 and12/15 holdout optimal-point gates;
  fresh MOSEK passes7/10 and12/15. SDPX/MOSEK's warmed end-to-end geometric
  mean is1.54 across14 common accepted cases. MOSEK's internal defaults differ;
  external gates are unchanged. Its unvalidated LP_agg certificate is excluded.
  Fresh Clarabel holdout passes11/15; SDPX/Clarabel's common11-point ratio is0.91,
  strongly influenced by arch0 (4.51/18.36s). Several small LPs remain4–18%
  slower. Two Clarabel points have post-launch supervision/cleanup errors,
  no complete memory receipt or solver status; preserve them as incomplete.
  This single campaign does not establish a stable optimization gain or broad
  superiority. Raw results: `/tmp/sdpx-float64-final23-prep-20260914/`,
  `/tmp/sdpx-float64-final22-prep-20260914/` (fresh MOSEK), and
  `/tmp/sdpx-float64-clarabel-holdout8-20260914/`.
  Benchmark supervision now preserves primary timeout/RSS causes and prevents
  another launch after unconfirmed cleanup; seven supervisor checks pass.
- The pre-removal Ising baseline, job 212584, passed all original-coordinate
  audits with unchanged source and input. At 512 bits, 50 iterations and
  1/2/4/8 physical cores, native medians are
  178.58/127.97/100.81/89.64 s (eight-core speedup 1.99x). Retained same-node
  SDPB medians are 117/64/33/23 s (5.09x). Factor-input job 212604 returned
  46 iterations / 123.45 s at one core, but failed the external per-equation
  relative gate (1.72e-28 > 1e-30); it earns no speed credit. Sampled and CSC
  residuals agree to 6.1e-149. The global stopping criterion does not imply the
  stricter componentwise criterion: the saved point has D/min(w)=1.576e9.
  A separate matched 1e-42 public-tolerance experiment ran for both SDPX
  and fresh SDPB runs, preserving 512 bits and all external 1e-30 gates.
  Job212615, frozen22, completed with all28 SDPX/SDPB points accepted
  and unchanged source. SDPX native medians at1/2/4/8cores are
  132.68/97.23/78.46/72.00s, all50iterations (1.84x eight-core scaling).
  Fresh SDPB medians are126/70/36/25s, with final logged iteration201.
  Eight-core total-time ratio is2.88; dividing by iteration counts gives11.58,
  but the algorithms perform different work per iteration. All28 points pass;
  objective spread is2.327e-41. SDPX invocation RSS616–661MiB includes Julia
  startup/audits; SDPB sampled aggregate peak reaches525MiB at8ranks.
  Complete receipts: `/tmp/sdpx-rust-ising-20260914-strict10/complete-results/`.
  The job requested node7; the actual compute hostname was not retained.
  Both solvers' affinity and physical-core topology were recorded within the
  same allocation. The job suffix identifies the PBS server, not the compute host.
  Receipts: `/tmp/sdpx-rust-ising-20260914-{accuracy07a,sampled08}/`.
- Earlier phase attribution found Schur assembly 63.11 s, cone scaling
  20.53 s, RHS/recovery 26.80 s and sparse factor update only 2.09 s; roughly
  75 s remained to subdivide. Nested phase timers overlap. Reprofile the
  structurally changed candidate before selecting further large kernels.
- Current source24 small-Ising diagnostics pass both external audits at1/8
  threads. Eight-thread native total17.52s comprises KKT solve5.49s,
  affine/combined step bounds5.55s, KKT update2.86s, scaling2.17s and
  combined RHS0.573s. These are nested diagnostic timers, not speed evidence.
  The affine PSD transforms are recomputed in combined RHS; a bounded candidate
  will explicitly consume the disposable affine direction when m=1 and reuse
  its packed transforms. Public cone methods and the first iteration retain
  their original semantics. Its small phase share demands measured benefit;
  do not assume eliminating four GEMMs yields a large solver speedup.
  Receipts: `/tmp/sdpx-ising-profile24-20260914/results/phase-summary.json`.
- The synthetic orthant LP campaigns show no useful Float64 scaling and only
  about 1.11x at 256/512 bits. Their one-nonzero-per-row matrix does not establish
  scaling for substantial sparse products. Do not attribute changed convergence
  checks to threading. macOS budgets do not establish homogeneous physical-core
  scaling.
- Larger Ising selection is fixed from metadata before new solve outcomes:
  Lambda11, n=1099, m=22196, 28 PSD blocks of orders36–43. Reuse the existing
  512-bit common input; sealed plan SHA256
  `48de57b1d52821c290aaf98766514f7bac5e0e9b0f730a729edab1c128ba1774`.
  Input SHA256: `bb1fa49da0d461ebba2b9539412222e5dc134553dfad8f7bc128b23acf61ed1d`.
  The qualified frozen23 pilot used8cores/48GB/4h, with a fresh independently
  audited SDPB anchor followed by one cold and one warm SDPX solve.
  Job212619 ran on node100. Its512-bit SDPB reference failed Cholesky
  positive-definiteness after logged iteration230, before reaching tolerance;
  SDPX was not started and no cross-solver speed is credited.
  Its submission receipt's node7 description was corrected separately against
  the actual PBS request and allocation. Do not compare timings across these
  different hosts as a source-only change. A separate fixed768-bit retry for
  both solvers, job212626, retains the exact512-bit-converted decimal input and
  tolerances, with no checkpoint or warmstart. Its SDPB point passes, but SDPX
  returns iteration-zero optimal status with primal0.352, dual0.874 and mapping
  residual near1. It earns no speed credit. The unchanged upstream stopping
  denominators include ||x||≈1.95e78, permitting this false acceptance relative
  to the external contract. Source review found no convergence/Ruiz divergence;
  independent saved-point reconstruction confirms sampled/CSC products agree
  to relative8.59e-176/1.64e-230. Residual numerators1136.73/527.70 divided by
  about2.03e78 explain the upstream acceptance without an operator defect.
  SDPB uses absolute maximum residuals, so equal numeric internal tolerances
  are not comparable. The actual sampled-audit denominator gives a pointwise
  sufficient feasibility tolerance1.33e-133. A separately declared pilot fixes
  SDPX feasibility at1e-140 with gap1e-42; SDPB keeps absolute1e-42 tolerances.
  Both retain768-bit arithmetic, the same512-bit-converted input bytes and
  external1e-30 gates. Run SDPX cold and warm against the retained accepted
  reference first, then a fresh SDPB run only if both points pass. Reuse the
  qualified frozen24 Linux library only after exact provider verification on
  the compute node. Job212628 passed provider qualification and the analytic
  gate on node100, then stopped after the first SDPX point failed external
  sampled dual consistency:2.1634e-22 exceeds1e-30. The94-iteration point
  has primal2.11e-66,dual6.91e-65,gap7.33e-67 and reference-objective
  agreement1.31e-42; those gates pass. Its native1287.85s earns no speed
  credit, and neither a warm solve nor another SDPB run was started.
  Source/provider identities match before/after. Saved-point replay shows Dd
  grew4.03x while minimum sampled work fell from2.69e-25 to9.34e-68, about43
  orders of magnitude. The worst component has absolute error2.02e-89 and
  relative error2.16e-22. Sampled/CSC products still agree; the initial bound
  was not uniform. The new pointwise bound1.14e-176 is diagnostic, not a
  recommendation for another blind tolerance retry. Receipts:
  `/tmp/sdpx-lambda11-target-point-diagnostic-20260914/`. This pilot is not a
  stable-median speed comparison.
  A bound at the saved point is not a guarantee along
  later iterates; every returned point still needs external qualification.
  This retains the user's single upstream convergence stage. Use original-coordinate audits;
  do not fabricate a larger benchmark by duplicating blocks.
- Selective MPFR GEMM/SYRK FMA showed repeated isolated 1.11–1.26x gains at
  256–1024 bits; 128/2048 retain ordinary multiply/add. In-place assignment
  was rejected for inconsistent gains. The nonsymmetric trace Schur shortcut
  was rejected after failing a fixed 4096-epsilon cancellation gate. Neither
  experiment supplies an assumed end-to-end speedup.

## Execution order

The selected runtime cleanup, MPFR workspace/FMA changes, cached block scheduling,
sampled operators, sparse residual products and balanced dense parallel kernels
are implemented and qualified in frozen24, including PSD step-length parallelism.
The small same-node Ising comparison is complete; current work diagnoses the
remaining large-instance componentwise accuracy failure and tests only
optimizations supported by the updated phase profile.
Job212619's512-bit SDPB failure motivates the separate fixed768-bit Lambda11
pilot; neither failed runs nor changed precision protocols are speed evidence.
Review and freeze
any subsequent algorithm change before numerical testing. Removing validation is the user-selected behavior
change, not evidence that previously rejected points became more accurate.
Existing external benchmark gates and tolerances remain unchanged.

| Stage | Implementation and decision | Acceptance |
| --- | --- | --- |
| P0 — baseline | Complete the ten-case default Float64 comparison and a restored-default 512-bit Ising baseline before crediting kernel changes; declare independent holdout inputs before tuning. Fill missing high-precision phase attribution. Record actual preprocessing, arithmetic, factor backend and thread settings. | Immutable source, dependencies and inputs; full original-coordinate gates; first-call and warmed timings separate. |
| P1 — structure | Optimize exact Schur assembly, RHS reduction, recovery and direction/residual operators using measured shapes. Reuse persistent workspaces and remove redundant traversals. Add sampled-basis descriptors only where the frontend can preserve an exact representation. | Generic and structured routes agree within fixed precision-specific tolerances; sparse, dense, rank-deficient and mixed-cone regressions; no model-name dispatch. |
| P2 — MPFR kernels | Measure borrowed/in-place arithmetic, fused operations, packed/block kernels and allocation removal in actual hot kernels. Prioritize Schur/congruence work over sparse factorization on the measured Ising input. | Scalar ownership/aliasing and all supported precision modes checked; numerical regressions after rounding changes; microbench and end-to-end gains reported separately. |
| P3 — parallelism | Use one coherent thread budget; avoid nested oversubscription. Improve block scheduling using measured costs, deterministic reductions and bounded scratch storage. Parallelize sufficiently large dense kernels only when outer block concurrency cannot use the budget. | Sequential 1/2/4/8-core single-solve campaigns, actual affinity and backend receipts, memory and utilization; same accuracy at every width. Batch throughput reported separately. |
| P4 — Float64 | Profile default LP_bnl1 and representative LP/SOCP/SDP cases. Compare existing clique merge strategies. Optimize factor/solve, allocations and direction work before algorithm changes. Trial extra correctors only with a derivation compatible with the existing embedding. | Cross-family and holdout success rates do not regress; total solve time includes extra RHS, checks and line search. No assumed iteration-count speedup. |
| P5 — final comparison | Freeze the integrated candidate; qualify affected production contracts, then run Clarabel.rs, MOSEK and SDPX on fixed and holdout Float64 suites and SDPX/SDPB on same-node 512-bit Ising and a larger predeclared instance. | Bind prepared q/b updates, all precision modes, MPFR ownership, recovery/status and numerical regressions to the same frozen identity as benchmarks. Report all failures/timeouts, residuals, objective agreement, time and memory. Update a concise README summary with measured gaps and source identity. |

Implementation may proceed in disjoint files using Astra workers while baseline
tests use immutable snapshots. Integrate dependent changes sequentially and use a
separate reviewer. Never run competing numerical benchmarks on the same machine.
Keep the stable sibling SDPX.jl and reference checkouts untouched; no commits or
pushes are part of this goal.

## Parallel implementation: single node and multiple nodes

The user's September 14 follow-up makes node-local and distributed execution an
explicit workstream. Reuse the block/collective separation in the
[SDPB scaling paper, sections 2.2.1–2.2.4](https://arxiv.org/pdf/1909.09745):
measure steady-iteration block costs, keep block work local where possible,
balance expensive blocks, and account for global matrix reduction/factorization
and its temporary memory. Its distributed matrix Q is not automatically the
same matrix as SDPX's condensed system; derive the communication boundary from
SDPX's equations. Do not copy historical performance ratios as predictions.

### Concrete source reuse and first deliverables

Source review confirms the following order. SDPB reference paths below are
relative to `src/` in the retained, file-hashed checkout
`/tmp/sdpx-performance-goal-20260913/sdpb-source-review/`; local Hypatia and COSMO
paths are relative to their untouched workspace repositories.

| Priority | Reference implementation | SDPX deliverable and scope |
| --- | --- | --- |
| 1: sampled SDP work reduction | SDPB `sdp_solve/SDP_Solver/run/compute_bilinear_pairings/compute_A_X_inv.cxx` and `step/initialize_schur_complement_solver/compute_schur_complement.cxx` under the same `run/` directory | Add one owned, factor-authoritative sampled operator alongside CSC; transform shared basis columns with existing NT factors and reuse their Gram entries in `condensed.rs`. Keep the existing predictor/corrector, LDL and refinement. Apply the same operator to residuals, RHS/recovery and external direction tests. |
| 2: complete parallel coverage | SDPB `sdpb/main.cxx`, `sdpb_util/block_mapping/compute_block_grid_mapping.hxx`, and `sdp_solve/Block_Info/allocate_blocks.cxx` | Replace block-count splitting in condensed scaling with cached cost-aware partitions; measure phase costs after atypical initial iterations. Extend the shared budget to sparse products for LP/SOCP and output-column/RHS tiles for dominant PSD blocks. Keep each output's accumulation order and deterministic global reductions. |
| 3: reuse decomposition storage | COSMO `src/convexset.jl` eigensolver workspace ownership; SDPB cached bigint context and shape-dependent job schedules | Remove repeated MPFR SVD/reflector scratch allocation in `algebra/dense/blas/mpfr.rs`, using owned dimension/precision-specific buffers. Existing PSD scratch, Schur positions, factor and predictor/corrector reuse already exist; preserve them. |
| 4: batch generic dense cone work | Hypatia `src/Solvers/systemsolvers/qrchol.jl` and `src/Cones/possemideftri.jl` square-root Hessian panels | Trial bounded panels of NT-scaled coefficient matrices and Gram assembly for generic dense PSD blocks. Preserve sparse-column routing; qualify cancellation and memory before adoption. Hypatia's assembly loop itself is serial, with possible BLAS threading. |
| 5: large-product/distributed backends | SDPB `sdp_solve/SDP_Solver/run/step/initialize_schur_complement_solver/compute_Q.cxx` and `run/bigint_syrk/` | Qualify residue/CRT BLAS with reconstruction/error bounds, then distribute both block work and the reduced global system when profiles justify it. MPI alone does not reduce generic constraint work. |

The sampled representation is a numerical contract, not a metadata hint.
`../PMP2SDP.jl/src/sdpb_native/primal_conic.jl` already exposes basis metadata but
explicitly makes rounded CSC coefficients authoritative. Independently rounded
outer products, svec conversion and duplicate accumulation generally destroy
exact low-rank identities. Existing CSC inputs therefore keep their current
semantics. New factor-defined inputs must own basis values, grouping, weights,
coordinate maps and precision; their stored factors define the operator.
Ruiz and structural preprocessing must propagate that representation faithfully.
Do not substitute approximate factors into the old CSC contract.

For a primitive sampled coefficient, the NT-compatible construction is
`U = I_m ⊗ Q`, `V = Rinv U`, `K = Vᵀ V`, followed by the appropriate symmetric
pair products of K. This borrows SDPB's shared-basis organization while retaining
SDPX's search direction. It avoids the rejected nonsymmetric trace shortcut;
off-diagonal cancellation and ill-conditioned transforms still need fixed
precision oracle checks against the actual stored factors.

COSMO's composite projection is serial; Hypatia has no general parallel cone
scheduler to transplant. Their useful contributions here are workspaces and
operator/factor interfaces. SDPX already has a Float64 Pardiso hook, so qualify
that existing provider before adding another. Defer a matrix-free Krylov route
until factor fill dominates and a preconditioner is demonstrated; do not copy
COSMO's ADMM tolerances or Hypatia's untuned MINRES stopping into the IPM.

SDPB's CRT code uses double BLAS on bounded residues, not a direct Float64
conversion of high-precision solver data. Its half-working-bit normalized-Q
diagonal sanity check is not SDPX's accuracy contract and does not establish
that SDPB computes only half-precision results. Reused kernels need independent
qualification at SDPX's unchanged precision and tolerances.

1. **One resource budget.** Keep a persistent explicitly sized
   [Rayon pool](https://docs.rs/rayon/latest/rayon/struct.ThreadPoolBuilder.html)
   for CPU kernels. Partition physical cores across outer block work, large
   dense kernels and factorization; never independently give each level the
   entire budget. Measure idle/barrier time and NUMA placement as well as CPU
   utilization. Scheduling decisions must use structure and measured costs,
   never instance names.
2. **General conic work units.** PSD blocks are useful tasks; tiny SOC/orthant
   blocks need batching. Large sparse LP/SOCP work must also partition sparse
   products, assembly and eligible factorization tasks. A cone-only pool does
   not establish general LP/SOCP scaling. Preserve deterministic ownership and
   bounded scratch storage; recalibrate only when measured imbalance justifies
   its overhead.
3. **MPI process layer.** Use mature MPI through Rust bindings rather than a
   custom network runtime. Evaluate [rsmpi communicators and node-local
   splitting](https://docs.rs/mpi/latest/mpi/topology/trait.Communicator.html).
   Begin with a communication owner on the main thread and verify the MPI
   implementation's provided threading level. Workers operate on disjoint local
   blocks; synchronization carries the mathematically required reductions,
   direction data and termination decisions. Do not call MPI from arbitrary
   Rayon workers without a deliberately qualified threading contract.
4. **Precision and data movement.** Transport high-precision values without
   Float64 conversion or raw pointers. Define versioned owned serialization,
   precision/range checks and bounded communication buffers. Preserve the
   complete residual/recovery logic and consistent decisions across ranks.
   Start with a correct deterministic reduction path; optimize reduction order,
   nonblocking overlap and distributed factorization against numerical and
   communication profiles. Avoid full global-matrix copies per worker.
5. **Acceptance.** First prove single-node 1/2/4/8-core correctness and speed.
   Separately compare pure MPI and MPI+threads at equal physical-core budgets,
   then test a predeclared larger problem across nodes. Report strong scaling
   `T1/Tp`, efficiency `T1/(p*Tp)`, per-stage compute/communication/wait time,
   aggregate and per-rank memory, process/core affinity and actual thread levels.
   Float64 and MPFR share the scheduling architecture and retain distinct
   provider measurements. Small problems may saturate; concurrency alone is
   not evidence of speedup. Run distributed qualification after the current
   default Ising accuracy failure is resolved.

## Conditional work

These are gated experiments, not mandatory architectural additions:

- Replacing `(1-alpha_aff)^3` with `(mu_aff/mu)^3` is mathematically equivalent
  for the exact affine direction of the current LP/SOCP/SDP homogeneous
  embedding, including tau/kappa. It supplies no justified iteration reduction.
  Hypatia's combined stepper additionally requires curvature/proximity machinery;
  do not transplant it as a small sigma patch. A same-factor corrector experiment
  needs measured short-step frequency and a compatible derivation first.

- Structured Gram/CRT requires exact encoding, reconstruction/error bounds and
  an applicable frontend descriptor. Do not claim a generic dense-kernel speedup.
- QR+Cholesky requires suitable rank/nullspace structure and a safe direct
  fallback. Sparse KKT factorization is currently too small a fraction of Ising
  runtime to justify this as the first optimization.
- TwoFloat is an optional new approximately 106-bit LP/SOCP/SDP mode, with f64
  exponent range. It cannot replace 128-bit or 512-bit modes. Unsupported cones
  must reject that mode explicitly; dependency availability and the precision
  contract must be settled before implementation.
- Krylov methods need a useful preconditioner, true-residual control and direct
  fallback. Do not transplant an ADMM tolerance schedule into the IPM.
- MPI implementation and qualification follow the explicit parallel workstream
  above. Advanced distributed factorization/collective algorithms are selected
  only when measured global-stage cost justifies them.
- An optional GMP backend must disclose its distinct precision/rounding contract
  and pass separate qualification; equal requested bits do not mean identical
  MPFR semantics. Preserve MPFR as the default high-precision implementation.

## Measurement and retention rules

Reuse the retained conic10 manifest
`a8f757a02188c281c531932f208471233f5a082c2321ea0a81833fc3881fa151`.
Use one first call plus three fresh warmed solves, 200 iterations, internal full
tolerances 1e-8 and external gate 1e-6 for the existing Float64 protocol. MOSEK
product-default tolerances differ and must be disclosed alongside the common
external gate. Extend reference thread support before claiming a 1/2/4/8-core
reference comparison. Preserve watchdog bounds and record incomplete runs.

Reserve the existing `../SDPX.jl/benchmark/data/holdout` fifteen-case suite for
this cycle's final cross-instance check, without tuning on its new results.
Its manifest SHA256 is
`0b55e6b31b05530042b7a5058cf5a498da15923f20388812902b3f1deb3d34c2`;
all fifteen JSON hashes have been verified and are disjoint from conic10.
This is a reused, instance-disjoint suite, not a claim of historically unseen
data or disjoint problem families. Retain its declared eight repetitions and
resource bounds; do not compare its medians against a different protocol without
disclosure. Any subsequently needed genuinely unseen suite must be selected from
metadata and frozen before observing candidate outcomes.

For the existing Ising input, preserve SHA256
`e4484eb8895e504a8f5c83651a5b964172e24c7060db24ddf09b737dcc78fddf`,
512 bits and external tolerance 1e-30. Preserve the existing 1e-34-internal
campaigns and their failures unchanged. The new strict10 protocol requests
1e-42 public feasibility/gap tolerances from both SDPX and fresh SDPB runs;
old SDPB points anchor input/objective, not matched timing. This is tighter
stopping at the same arithmetic precision, not precision reduction or a restored
custom runtime gate. Audit every returned point; the saved-point bound is not
a universal guarantee. Record the SDPB arithmetic provider as well as requested
precision. Preprocessing-default runs remain separate from historical disabled-
preprocessing runs.

Retain speed changes only with a reproducible median improvement of at least 2%
under comparable conditions, or a separately justified correctness/memory benefit.
Broaden repetitions when timing noise prevents a conclusion. Report paired-case
ratios and aggregate only jointly qualified cases; also report total success
counts to avoid hiding difficult inputs. No speed credit for failed accuracy,
AlmostSolved, iteration limits, resource limits or incomplete memory evidence.
All ten development cases and the fixed Ising case are primary cases. A repeatable
slowdown above 5% on a jointly qualified primary case blocks unconditional default
adoption, even if another case improves; investigate and remove it or explicitly
document and justify the measured tradeoff. Do not drop or replace slow cases.

Record source hashes including dirty changes, dependencies, input hashes,
precision, settings, actual provider/thread configuration and raw repetitions.
Separate native solve, end-to-end frontend time, setup, first call, allocations,
and process high-water memory. Raw logs and frozen copies stay outside the repo.
This file is the single active plan; the README retains only usage, architecture
and a concise measured performance summary. Unmet targets remain explicit.
