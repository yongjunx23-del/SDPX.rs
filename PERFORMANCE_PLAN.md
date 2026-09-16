# SDPX performance plan

Target: Float64 near or above Clarabel.rs/MOSEK and high-precision SDP near or
above SDPB, without losing accuracy. These are measured targets, not promises.
Contracts: [AGENTS.md](AGENTS.md). Evaluation: [benchmark/research](benchmark/research/README.md).

## Continuous optimization objective

Continue evidence-driven optimization without a preset end date. Each cycle:
profile the current frozen baseline; formulate one testable bottleneck hypothesis;
implement the smallest shared-precision change; run focused numerical checks and
bounded paired benchmarks; retain repeatable gains or revert. Short development
screens have explicit timeouts; broader acceptance runs only for promising changes.
Failures remain in coverage reports. Preserve original-coordinate accuracy,
fixed precision, default preprocessing and the single-engine architecture.

After several unsuccessful hypotheses, or when the dominant cost changes, review
recent relevant papers and mature source implementations, then resume with a new
measurable hypothesis. Do not repeatedly benchmark the same failed idea without
new evidence. Reserve a new independent family before making a new generalization
claim: the current fresh group is already exposed.

Immediate order: residual-stage overlap (212949) and runtime AVX2/FMA dispatch
with shared-kernel inlining (213041) have passed full acceptance and are integrated.
The combined candidate passed 291 Rust core tests, 508 Julia assertions across 32
testsets, all 56 cluster numerical receipts, 666 source and 6 binary identity checks.
The actual x86 generic/accelerated bitwise-equivalence fixture also passed.
Matched Ising512 comparison (212965) completed all 21 point audits. Refresh profiling
and matched Float64 references on the accepted build next. Sync/assembly fusion
(213040) was rejected for measured regressions. Preserve four unexposed holdouts
for final generalization checks.
Resolve the remaining medium/hinf3 and larger Lambda11 accuracy issues; continue
high-precision iteration-cost optimization and real 1/4/16/64/256-core,
multi-node single-solve qualification. Small Ising timing does not close those
larger-scale requirements.
Publish verified code and matching cluster snapshots under the user's existing
authorization. Default to working without subagents. No speed or parity claim
without matched evidence; no claim of automatic background research while Goal
is paused. The old Goal's five-equation requirement and no-push restriction are
superseded by current project instructions and explicit publication authorization.

## Current evidence — September 16, 2026

Accepted runtime dispatch preserves the arithmetic implementation and checks AVX2
and FMA before entering the specialized kernel, retaining the portable fallback.
Against the accepted residual-overlap baseline on node49, medium native medians
at 1/4/16 cores improve 11.344→9.017 / 11.288→8.898 / 11.380→8.950 seconds
(20.5–21.4%). LP_bore3d/SOCP_axis_wide/SDP_truss2 are 0.38/1.58/1.48% slower;
SOCP_nb16 is 1.08% faster. Ordinary 256/512-bit SDP improves 1.88/0.54%, below
the performance-credit threshold. Sampled Ising bypasses the modified helper;
its observed 1.01/0.87/4.20% changes do not establish a causal speed benefit.
Precision and numerical gates are unchanged. These Linux timings are not a
matched MOSEK comparison. Production source SHA 94114e53, macOS FFI 948f6aa0.
Evidence: `/tmp/sdpx-runtime-fma-combined-20260916/acceptance/final-decision.json`,
`acceptance/build-logs/`, `frontend-verification.json`, `integration-review.json`.
The publication scope retains only accepted changes. The isolated profile includes all 13 stage
counters, explicitly fixing the previous diagnostic's missing scaling-sync span. PBS 213042.node220
completed medium at 1/16 and Ising512 at 16 physical cores in isolation.
All numerical/profile gates, 363 source and 3 library checks pass. Instrumented
cargo check passed. Medium assembly remains 5.079/5.348 seconds at 1/16 cores,
with refactor 1.300/1.300 and reduced refinement 1.103/0.580 seconds. Refresh
assembly's transform/dot/store breakdown before the next kernel hypothesis;
previous unsuccessful parallel assembly variants remain rejected. Ising16
assembly is only 0.059 seconds: prioritize refactor (1.763), outer residual
(1.246), and raw input/output scaling (0.732/0.891), not Schur assembly for this
sampled instance. Reduced and refinement/backsolve counters nest; never sum
parent and child times. These observations are not a paired speed comparison.
Evidence: `/tmp/sdpx-post-fma-profile-20260916/final-decision.json` and `build-logs/`.
Fine assembly diagnostic 213043 completed with all numerical/counter gates,
363 frozen-source and 3 binary checks passing. On medium1, worker-span totals:
coefficient product 0.408s, GEMM 2.748s, suffix packing 0.411s, dense dot 0.215s,
store 1.088s, sparse pairs 0.195s. Transform total 3.581s and selected total
5.111s include children; do not add them together. Medium16 remains slower in
these serial-dominated stages. No production instrumentation.
Evidence: `/tmp/sdpx-post-fma-detail-20260916/final-decision.json` and `build-logs/`.

Store-transpose screen 213044 completed all eight numerical receipts and
653 source/four binary checks. Reject it: medium1 improves only 0.11%
(9.0570→9.0468s), while Ising512-16 regresses 10.57% (12.9608→14.3306s).
No integration and no unchanged repeat. Its local 291 core tests passed, but
numerical correctness alone does not justify the extra workspace.
Evidence: `/tmp/sdpx-store-transpose-20260916/final-decision.json` and `build-logs/`.

Independent MPFR candidate uses a shared mutable-descriptor helper for output
construction and in-place assignment. The existing +=/-=/*=/ /=/%= operators
call the same MPFR operations with unchanged fixed precision and nearest-even
rounding; destination storage is exclusively borrowed and no descriptor escapes.
[MPFR variable conventions](https://www.mpfr.org/mpfr-current/mpfr.html#MPFR-Variable-Conventions)
permit input/output aliasing. The candidate preserves copy ownership and tests
128/256/512/768/1024/2048 bits, signed zeros, infinities, NaNs, cancellation,
large exponents and reused destinations. All 6 arithmetic and 291 solver core
tests pass. Warm local add/sub ABBA microtests reduce elapsed time by
19.86/31.16/38.39% at 256/512/1024 bits; this is not whole-solve evidence.
Isolated screen 213045.node220 completed after 213044;
medium1/Ising512-16 ABBA, 653 frozen files, release arithmetic tests, rebuilt
baseline/candidate FFI, fixed accuracy gates and final identities. It starts
from the accepted runtime-FMA source, excluding the store-transpose candidate.
16 cores/32 GiB/45 minutes. No production arithmetic change or adoption claim.
Evidence: `/tmp/sdpx-mpfr-assign-20260916/`.
All eight numerical receipts and 653 source/four binary checks pass, but reject
the candidate for insufficient whole-solve benefit: medium1 9.0068→9.0260s
(0.21% slower), Ising512-16 13.2363→13.2115s (0.19% faster). The microbenchmark
improvement did not translate into a qualifying solver improvement. No integration.
The stable runtime-FMA/residual-overlap version remains the publication candidate.






Latest matched high-precision evidence: at 512 bits and fixed 1e-42 internal /
1e-30 external gates, Ising SDPX warm-native medians at 1/4/16 physical cores are
71.19/22.81/13.26 s, versus SDPB per-iteration timer sums of 127.06/35.85/15.66 s.
These clocks have different scopes. SDPX takes 50 iterations, SDPB 201; SDPX's
iteration cost and parallel efficiency remain worse. This qualifies one finite
Ising problem, not general SDPB parity. Detailed receipts and scope appear below.
Runtime FMA and inlining passed short screens, but remain outside production
until the combined broad campaign finishes.

Latest local single-core medium refresh is SDPX **3.055060 s / 18 iterations**
versus MOSEK 11.1.3 **1.921669 s / 16 iterations**: native reported medians give
**1.59x slower**. Two fresh-process samples each, interleaved SDPX/MOSEK/MOSEK/SDPX;
328 frozen files unchanged, identical original raw coefficients verified after
MOSEK export. SDPX passes its additional independent dual audit; the retained
MOSEK checker covers original equalities, PSD feasibility and nonmissing gap,
without an identical dual-stationarity oracle. The frozen MOSEK driver explicitly uses CVXPY `eps=1e-6` (not MOSEK
default tolerances); SDPX primary internal tolerances are also 1e-6, with
solver-specific stopping rules. Expanded MOSEK parameters are recorded in
`effective-settings.json`. This remains a scoped comparison, not parity evidence.
The prior 3.159440/1.945962 pair is historical, not a controlled speedup baseline.
Evidence: `/tmp/sdpx-local-refresh-20260916/summary.json` and `results/`.
A supplemental MOSEK solve now exports its original x/equality/PSD dual point
and passes the exact shared 256-bit-accumulation Julia dual auditor used for
SDPX. Equality dual signs are converted from CVXPY; nine analytic sign/trace
checks pass. Upstream-normalized stationarity 1.6242e-8, gap 3.7197e-7, dual PSD
violation zero, original equality residual 1.0383e-9 and primal minimum eigenvalue
-1.7982e-14. Original raw inputs and shared helper match byte-for-byte.
This validates the supplemental point, not retrospectively the timed points
which were not exported. Evidence: `/tmp/sdpx-medium-common-audit-20260916/`.

This is native solve time at the original 1e-6 external gate, not broad parity.
Input `../大规模矩阵/models/medium/SU2path465var1887.json` has SHA256
`b1d973ac564e206fbb29aaad0a9b947c112f1130956aab31e79d899023c30048`,
1887 variables, 47 equalities and PSD orders 60, 57, 59, 57, 116.
Exact presolve removes 14 equalities. The 1e-8 full-accuracy failure remains.
A newly independent original-model dual audit also finds stationarity/max(1,
norm(q,Inf)) = 1.91649e-6 at the stable returned point, above a 1e-6 gate.
The historical timing pair passed its original primal/gap checks; it does not
prove that stronger target-only stationarity acceptance. Independently
reconstructed Clarabel normalization is 2.51278e-7 and passes the established
1e-6 criterion. See the normalization clarification below.

A fresh group was fixed before running: six previously reserved public inputs
from ClarabelBenchmarks revision `3679912c6bbd3f64c5c962f9d1c09d524561c412`
and three deterministic planted LP/SOCP/SDP-interface diagnostics (two SOCP,
one SDP). All nine are now exposed regression cases. Four holdouts are now reserved without solver execution: LP_ship04s,
SDP_copo14, SDP_filter48_socp and pure SOCP_strictmin_2D_43_dual. Source provenance,
conversion checks and 180-second/4096-MiB execution caps are recorded below.
These remain unavailable for optimization tuning; passing final holdout acceptance
is still unproven. The previously downloaded ss30 is another truss-topology instance; db_shear_wall appears in a historical exclusion list. Neither proves a new independent family. Both were format-converted without solver execution, and are excluded from the new holdout pending genuinely independent sources. Synthetic planted optima do not establish representative performance.
MOSEK passes 9/9, SDPX passes 8/9. SDP_hinf3 returns AlmostOptimal: original
primal residual 3.12e-4 exceeds 1.32e-5, and dual residual 2.64e-6 exceeds 2e-6.
Explicit condensed formulation also fails and is not adopted.

Warmed API medians in milliseconds, one thread, unchanged original-coordinate
1e-6 gate, default preprocessing. Entries below summarize the first matched run;
reverse-order and additional samples are retained externally.

| Case | SDPX baseline | Exact-presolve candidate | MOSEK | SDPX gate |
|---|---:|---:|---:|---|
| LP_bore3d | 7.047 | 3.534 | 0.808 | pass |
| LP_e226 | 5.491 | 5.449 | 2.928 | pass |
| LP_grow7 | 13.477 | 3.705 | 3.035 | pass |
| LP_sc50a | 0.271 | 0.265 | 0.329 | pass |
| SDP_hinf3 | 1.338 | 1.302 | 3.812 | fail; no speed credit |
| SDP_truss2 | 3.029 | 3.023 | 6.394 | pass |
| SOCP_axis_many (synthetic) | 7.043 | 7.109 | 3.645 | pass |
| SOCP_axis_wide (synthetic) | 7.300 | 13.324 | 6.227 | pass |
| SDP_planted80 (synthetic) | 59.237 | 59.470 | 19.584 | pass |

SOCP_axis_wide's apparent regression did not reproduce: retaining all 18 warm
samples per arm gives 7.266 versus 7.217 ms. The planted SDP takes MOSEK one
iteration and SDPX seven; it cannot support a general SDP ranking.
Process high-water RSS in the first paired run is 659/65 MiB (SDPX/MOSEK)
for bore3d and 674/83 MiB for planted80. These include Julia/Python and the
external oracle, not solver-only memory; do not attribute the whole gap to
native matrix storage.

The exact-presolve candidate peels structurally independent singleton-column
rows before bounded GMP elimination. It never deletes a row merely because it
was peeled. LP_grow7 previously spent 10.9 ms in presolve; all 140 equality rows
are structurally independent. Repeated local LP gains are about 50% on bore3d
and 73% on grow7. The candidate is frozen externally; **rejected for now after an Ising regression**.

Later local medium timings changed to 6.82/11.28/12.40/6.72 s (baseline/candidate/
candidate/baseline). Numerical outputs match, but these measurements do not
qualify adoption. Local Ising baseline also shifted from about 39.6 to 85–86 s;
no causal kernel conclusion is drawn from this changed timing environment.

### Fixed-core pilot observations

PBS 212917 on node48 has completed the Float64 ABBA cells: bore3d warm API
medians are 14.430 -> 6.819 ms, grow7 28.833 -> 7.252 ms, and axis_wide
26.595 -> 26.572 ms. All pass. Medium native medians are 61.045098 ->
60.788815 s, with 18 iterations and original gates passing; its prior local
regression does not reproduce here. Ising paired acceptance subsequently completed; its regression decision is below.
These Linux numbers use Netlib and must not be compared directly to macOS
Accelerate numbers or treated as a best-provider comparison with MOSEK.

A raw-medium sparsity screen of HDSDP-style intermediate contractions predicts
no winning columns under the current flop cost model, at either Float64 or
512 bits. This is a structural estimate, not a kernel timing or a proof that
M4 is useless elsewhere. Avoid adding a route solely to match HDSDP's checklist.
PBS **212918.node220** completed on node49. This experiment
compared the same candidate source and LP64
arithmetic using static Netlib versus installed OpenBLAS 0.3.29 on one fixed
physical core. Both libraries and input/operator audits are preserved; optimized
BLAS may explain much of Linux medium's cost. The build and runtime preflight
confirm OpenBLAS 0.3.29, the Zen kernel, LP64 integers, one thread, and loading
from the immutable campaign native-library copy. Completed paired timings and their scope are recorded below.
Experiment snapshots: `/tmp/sdpx-fresh-20260916/pilot-observation-01/` and
`hpc:~/projects/sdpx-blas-20260916-pilot01`.

## Retained implementation

- Dense positive-block Cholesky with the correctly signed bordered complement,
  existing refinement and sparse fallback; packed Schur dot fusion.
- Exact coefficient reuse and block-local sparsity ordering share Float64/MPFR
  logic. Equality is checked after scaling and updates; hashes never suffice.
  Compact triangular Float64 panels reduce memory; MPFR uses streamed storage.
- Factor-authoritative sampled operators, exact sampled-basis reuse, structured
  direction/residual products and bounded threaded contributions.
- Bounded two-direction curve search for Float64 symmetric cones: no extra KKT
  solve. Medium improved 3.339455 -> 3.148746 s, 19 -> 18 iterations. MPFR keeps
  its original step policy because the same heuristic regressed Ising.
- One exact GMP equality presolver for Float64/MPFR, with original-coordinate
  recovery and sampled-factor remapping. Proof limits: 512 rows, one million
  elimination entries, 2048-bit rational components. Inconsistent RHS and
  unproven dependencies are retained. Medium row reduction itself is not a
  measured speedup.
- MPFR allocation test now counts only its calling thread, avoiding false failures
  caused by concurrently running provider tests. This changes test instrumentation.

Rejected prototypes are outside the repository: lagged-factor GMRES (all attempts
failed its residual gate), additional recentering solves, sign-opposite reuse,
COSMO eigensolver workspace/minimum-eigenvalue caching, short-dot specialization,
active-row tail GEMM, and alternative Float64 crossover. They did not deliver
repeatable qualifying full-solve improvements. Do not reintroduce them without
new evidence. Prior receipts: `/tmp/sdpx-{curve,unified,cosmo,signed}-20260916/`.

## Immediate acceptance and accuracy work

1. Qualify the residual-stage overlap candidate with **212949.node220**:
   56 fixed receipts covering LP/SOCP/SDP, medium and Ising at1/4/16 workers,
   plus ordinary256/512-bit single-worker regression. The initial paired screen
   passed all gates and reduced Ising512/16 time7.74%; medium16 improved1.81%.
   Do not integrate before broader acceptance. Independently, **212948.node220**
   tests byte-identical stable source with AVX2/FMA enabled after disassembly
   proved scalar indirect FMA calls in a hot generic-x86 kernel. Separate build
   gains from algorithm gains and qualify CPU compatibility before deployment.
   Both candidates are independent of rejected fusion/layout/batch experiments.
2. Preserve the integrated residual-reuse and parallel row-gather baseline.
   PBS **212929** passed all 48 numerical receipts; gains were 7.38% on
   medium-16, 5.27% on Ising-512/1 and 28.27% on Ising-512/16. Local release
   verification passed 508 assertions in 32 Julia testsets. Stable library SHA256:
   `8e4017e2fcbaf220950ddc2a5a2239a4244ccb16ae518fa19298a1f7fddf664f`.
   Evidence: `/tmp/sdpx-combined-20260916/final-decision.json` and build/test logs.
   These are matched SDPX comparisons, not new MOSEK or SDPB parity results.
3. Diagnose medium's 1e-8 and SDP_hinf3 full-accuracy failures before interpreting
   full-accuracy speed. Preserve existing residual normalization and distinguish
   the stronger target-only diagnostic. No tolerance/status promotion. Failed
   regularization and precision-preserving residual experiments need a new
   hypothesis before another sweep.
4. Larger Ising/Lambda11 remains unqualified: the previous fixed 768-bit run had
   sampled dual consistency 2.16e-22 versus the 1e-30 gate. The saved-point replay rules out a material CSC/factor mapping mismatch;
   diagnose conditioning and near-null solution growth before large-scale comparisons. The unsolved LP, SDP and pure-SOCP reservations now cover separate
   source families; exposed development problems cannot establish generalization.
5. After numerical qualification, compare matching providers, precision and
   tolerances against Clarabel.rs/MOSEK/SDPB, then extend real single-solve
   scaling beyond 16 cores. Ising currently reaches about 4.99x at 16 cores;
   medium has no demonstrated net multicore speedup. MPI and 64/256-core
   multi-node acceptance remain unfinished.

Keep rejected prototypes and detailed receipts outside production. Historical
experiments and failure diagnoses below remain evidence, not active work orders.

## Multi-node work after numerical qualification

The user authorizes **1, 4, 16, 64, 256 physical cores**, cluster execution and
GitHub/cluster publication of the verified result. SDPX currently has a shared
thread pool and serial MPFR QDLDL, **no distributed MPI solver**. MPI launch
receipts now use global ranks and explicit per-host CPU maps (five binding tests
pass); the existing controller still measures one node. Running multiple
independent full solves measures throughput, not single-solve scaling.

Implement in this order, keeping one solver algorithm and optional typed backends:

1. Measure per-iteration cone scaling, structured assembly, factorization, backsolve,
   reductions and idle time; record block costs and memory. Reuse scheduling plans
   instead of rebuilding them during every cone operation. Demonstrate 1/4/16/64
   single-node behavior before attributing any limit to networking.
2. Exploit exact block-arrow structure in the condensed Newton system. Local SPD
   factors and triangular solves form a smaller global complement. Preserve signs,
   regularization, full residual refinement and direct fallback. Share arithmetic
   logic across Float64/MPFR; do not force unrelated sparse LP/SOCP into SDP blocks.
3. Distribute independent blocks across MPI process groups with measured-cost
   allocation. Own block matrices and scratch locally; communicate the required
   coupling data. Within a large block, use a mature distributed dense provider
   through a narrow native interface instead of writing another matrix library.
4. Qualify the provider's arbitrary-precision storage, arithmetic and serialization.
   The SDPB Elemental fork's GMP arithmetic is not automatically equivalent to
   our MPFR rounding contract. No conversion through Float64, raw MPFR-pointer
   transfer or reduced precision in communication/reductions.
5. On a qualified large frozen problem, compare both solvers sequentially at each
   width on matched nodes. Verify physical cores versus SMT, MPI rank placement,
   node count, BLAS budgets and memory. Report T1/Tp, T1/(p*Tp), iteration counts,
   per-iteration phase time, communication time and aggregate/per-rank RSS.
   Include cross-node 256-core execution and failures; a tiny 22-block Ising case
   alone cannot establish useful 256-core scaling.
6. Adopt only validated improvements, remove unused experiment plumbing, then
   commit/push the reviewed source and publish an immutable matching cluster
   snapshot with source, dependency, library and input hashes. Publication remains
   pending; do not label unfinished MPI work as released.

## Source guidance

[MOSEK's GAMS documentation](https://www.gams.com/53/docs/S_MOSEK.html) exposes
solve-form choice, multiple correctors, ordering/dense-column handling and bounded
presolve work. These suggest experiments, not knowledge of proprietary internals.
Approximate folding and bound perturbations conflict with the present contract.
[QOCO](https://arxiv.org/html/2503.12658v4) suggests precomputing fixed structural
work; its specialized solver generation and different infeasibility machinery
are not replacements for SDPX's general homogeneous solver.
[Proximal stabilized SDP](https://doi.org/10.1007/s10589-024-00614-3) is a possible
conditioning direction, requiring algorithm-level accuracy and cost validation.

[SDPB's scaling paper](https://arxiv.org/html/1909.09745v1) and
[implementation](https://github.com/davidsd/sdpb) motivate hierarchical block
ownership, measured load balancing, distributed matrix kernels and reducing
shared global operations. Inspect `allocate_blocks.cxx` and `compute_Q.cxx`.
COSMO's clique merging is already represented; ADMM projection and Anderson
acceleration cannot be copied unchanged into an NT interior-point step.

### HDSDP source review

Reviewed COPT-Public/HDSDP at `8842fedd13cddcd021f047dd9a9f289aa2fc90b2`.
Its `interface/hdsdp_conic_sdp.c` chooses M2–M5 per coefficient using rank,
nonzeros and remaining-column work. M1 eigen-decomposition is explicitly disabled.
Prioritize a measured comparison of the missing intermediate-product strategy
against SDPX's existing sparse-pair and packed-dense routes; reuse one typed
planner and precision-specific cost estimates. Existing sampled rank-one and
sparsity paths are not new opportunities merely because HDSDP has them too.

Do not copy `1e-10` rank-one extraction from `dense_opts.c`/`sparse_opts.c`:
use authoritative factors or exact structure. Its restarted PCG with reused
Cholesky applies to an SPD Schur system, not the full indefinite SDPX KKT;
our rejected lagged-factor experiment is still relevant counter-evidence.
The dual-scaling algorithm could reduce scaling work but changes iterates,
primal recovery and convergence behavior; it is a separate research direction,
not a replacement NT formula. The inspected public C kernels use double and
threaded numerical providers, not SDPB-style distributed MPFR/MPI matrices.
No HDSDP-derived numerical implementation or speedup is claimed from this review.
Source: [HDSDP](https://github.com/COPT-Public/HDSDP/tree/8842fedd13cddcd021f047dd9a9f289aa2fc90b2),
[paper sections 5 and 7](https://arxiv.org/html/2207.13862v2).

### Completed first cluster acceptance

PBS 212917 completed. All 12 Ising512 solves pass the original 1e-42 internal /
1e-30 external protocol at 50 iterations. Four native observations per arm give
baseline 73.673786 s, exact-presolve candidate 76.203102 s (+3.43%), and combined
MPFR candidate 73.013663 s (-0.90%). The presolve candidate's x/s/z are exactly
identical to baseline. The regression is real in this paired campaign, but its
mechanism is unresolved; do not attribute it to changed iterates. Restore the
production presolver and keep the LP improvement as evidence for a revised trial.
The combined MPFR prototype does not meet the 2% full-solve adoption threshold.
Raw receipts: `/tmp/sdpx-fresh-20260916/pilot-final/`.

A generic HDSDP M4 prototype now passes 284 solver library tests outside the
workspace. Its contraction uses G*A and evaluates only requested left entries;
no numerical rank extraction or arithmetic conversion. A 12-case kernel screen
covers Float64/256/512 bits and four sparsity configurations. In two moderate
sparsity MPFR cases M4 beats both existing kernels by about 2–3x; Float64 favors
batched/dense products. These are local microbenchmarks, not solver speedups.
The external prototype selects M4 with a structural cost model for streamed wide
arithmetic and updates scheduling cost estimates; full-solve acceptance remains.
Source/tests: `/tmp/sdpx-hdsdp-m4-20260916/`. No production M4 route is adopted.

### Completed Linux BLAS paired result

The Float64 medium ABBA cells in PBS 212918 pass at 18 iterations:
Netlib 60.863673 / 61.014321 s, OpenBLAS 11.389320 / 11.382507 s.
Paired native medians are 60.938997 -> 11.385913 s (5.35x speed ratio).
This uses the identical frozen presolve-candidate source, one physical core,
and unchanged original-coordinate gates. It isolates provider configuration;
it is neither an algorithmic speedup nor a MOSEK comparison. Confirm the provider
on restored stable source before publishing. The high-precision cross-provider
audit completed successfully; it has only one process per provider and does not
establish a repeatable Ising speedup. Keep OpenBLAS as the leading Linux provider candidate.
The restored production presolver passes all seven exact-dependency tests.

### M4 whole-solve development screen

A fixed ordinary CSC SDP (PSD order 32, 12 variables, 24 coefficient entries
per sparse column, deterministic signed coefficients, known optimum zero) now
passes eight independent original-coordinate audits. Default Ruiz/presolve/chordal
remain on; this diagnostic explicitly selects condensed KKT. At 256 bits,
internal 1e-24 and external 1e-20 gates give 24 iterations, baseline/candidate
native medians 5.773249 / 5.634718 s (2.40% lower). At 512 bits, internal 1e-42
and external 1e-30 give 41 iterations, 14.413975 / 14.092439 s (2.23% lower).
ABBA order is fixed and every primal/dual residual, gap, PSD violation and
known-optimum check passes. Local exploratory results do not establish broad
performance or Ising speedup. Keep the prototype external until independent
ordinary/sampled and fixed-core acceptance. Receipts and frozen executables:
`/tmp/sdpx-hdsdp-m4-20260916/pairs/`, `decision.json`, `m4-solve-*`.
PBS 212918 has completed with successful Ising output audits and unchanged
native-library hashes; no OpenBLAS-related accuracy failure was observed in
that campaign. Stable-source provider qualification remains the next Linux step.

### Completed stable-source M4 acceptance

PBS **212920.node220** completed on node49 with exit 0, staged at
`hpc:~/projects/sdpx-m4-20260916-accept02`. The prior attempt 212919 exited
101 while building the example: its feature list omitted `sdpx-solver/sdp`.
No numerical cells ran. Failure records remain in accept01. The corrected
feature combination passes a local example compilation check; source algorithms,
inputs and tolerance settings are unchanged. Both frozen arms use the restored
stable presolver and the same copied OpenBLAS LP64 provider. Their only numerical
source difference is the generic M4 assembly route and its work estimates.
The script builds both libraries/executables before timing, then runs ABBA
LP/SOCP/SDP and medium, ordinary SDP at 256/512 bits, and Ising512. Every timed
solve binds one physical core; audits remain outside timing. Both builds have
completed. All 36 receipts and original numerical gates pass. Native median
seconds (baseline/candidate): medium 11.387008/11.298859, ordinary256
11.503297/11.494823, ordinary512 26.901631/26.373859, Ising512
73.822490/75.962958. M4 is **rejected**: Ising regresses 2.90%; other gains
remain below 2%. Final native/library identity checks pass. The stable-source
OpenBLAS arm is numerically qualified on these inputs, not MOSEK/SDPB parity.
Receipts: `/tmp/sdpx-hdsdp-m4-20260916/acceptance-final/`; decision:
`/tmp/sdpx-hdsdp-m4-20260916/acceptance-decision.json`.
Two additional local tests pass: M4 selection and original Schur entries after
coefficient changes/stored-zero updates with repeated-column reuse, at 256 and
512 bits. These test-only additions are outside the frozen cluster source;
the numerical implementation is unchanged. No candidate is promoted yet.

The external campaign assessor requires all 36 expected receipts, checks each
individual numerical result, and withholds acceptance on missing/failed cells.
Tampering tests reject NaN, infinity, negative and out-of-tolerance residuals even
when the stored accepted flag is true. `assess_cluster.py` remains outside the
repository; these are experiment checks, not new runtime solver validation.

### Accuracy isolation and acceptance evaluator repair

A frozen hinf3 diagnostic disables only the Float64 curve step outside production.
Both arms still return AlmostSolved at 22 iterations, with identical printed
original primal residual 3.118910e-4, dual residual 2.637993e-6 and gap 3.358287e-9.
This does not support the curve as the cause; do not remove the qualified medium
step optimization on this evidence. Observation-only instrumentation isolates the stop to KKT at iteration 22:
the factorizer returns success but the initial refinement residual is NaN.
Earlier constant-RHS solve residuals deteriorate to about 0.05 for RHS norm 19.1.
PSD scaling and the small-step gate do not fail first. An isolated Faer substitution also fails (iteration 26, initial refinement
residual NaN; original primal residual 6.983969e-4 and dual 2.771077e-6).
Backend substitution alone is insufficient. Next inspect PSD-derived KKT
numerical ranges and regularization/refinement stability; retain upstream checks. Logs and candidate are in
`/tmp/sdpx-accuracy-20260916/`; the stable production library hash is unchanged.

The M4 campaign's Float64 ABBA cells pass: medium medians 11.387008 / 11.298859 s
(baseline/candidate, only 0.77% lower). Ordinary 256-bit SDP also passes, at
11.503297 / 11.494823 s (0.07% lower). These gains do not qualify adoption.
The 512-bit ordinary pair also passes at 26.901631 / 26.373859 s (1.96% lower,
still below threshold). Ising acceptance has now completed and rejects M4, as recorded above. The external assessor's
exact-decimal tolerance comparison falsely rejected the normal MPFR round-trip
representation of 1e-20. It now compares the tolerance at the declared binary
precision; original residual gates still use the exact decimal limits. Tests
reject real tolerance changes and invalid/out-of-bound metrics. Solver and
frozen campaign evaluators are unchanged. Snapshot: `observation-02` under the
external M4 experiment directory.

The finer hinf3 trace identifies factor growth: at iteration 21, KKT max is
4.97e13 and L max is 3.87e8. At iteration 22, KKT remains finite (max 7.11e14),
but L contains infinity after 10 dynamic pivot regularizations. D and Dinv
remain finite, so the existing Dinv-only refactor check reports success; the
subsequent solve/refinement correctly rejects NaN. No invalid result is promoted.
Prioritize scaling/regularization against factor growth, not extra refinement
of already nonfinite factors. Trace: `/tmp/sdpx-accuracy-20260916/factor-overflow.json`.

A bounded diagnostic sweep of the existing static regularization constant
(1e-8, 1e-6, 1e-4, 1e-2; unchanged full tolerances, one core) returns
AlmostSolved at 22, 19, 62, 200 iterations respectively. None qualifies an
accuracy fix. Do not raise the global default on this evidence. The external
`accuracy-probe` executable permits further targeted settings tests without
recompiling the production library; artifacts remain outside the repository.

A four-sweep symmetric KKT equilibration prototype applies D*K*D, scales RHS
and recovered solutions consistently, and retains refinement in the original
KKT coordinates. It avoids the observed hinf3 overflow but returns AlmostSolved
at 57 iterations: original primal residual worsens from 3.11891e-4 to 1.12629e-3,
although dual residual and gap improve. Reject this simple scaling route; neither
finite factors nor a small gap establish full convergence. Prototype and paired
raw points: `/tmp/sdpx-accuracy-20260916/scaled*`. No production numerical edit.

### Research after stalled accuracy experiments

Reviewed local Hypatia `systemsolvers/qrchol.jl`: it eliminates equality directions
with QR, then builds a reduced SPD matrix from Hessian products; square-root
oracles form Gram products when available. This is not a drop-in stable solver
for the full SDPX augmented system, and forming a Gram matrix can still square
conditioning. Reuse operators where applicable rather than adding a second engine.

[HiPO (2025), sections 4.5–4.6](https://arxiv.org/html/2508.04370v1) addresses
pivot growth with local pivot selection and scale-aware regularization. A first
small diagnostic retains the magnitude of a wrong-sign pivot when restoring its
expected sign, instead of replacing a large pivot by the tiny fixed shift. This
is an isolated hypothesis, not an implementation of HiPO's full pivot rule.
Original KKT refinement and accuracy gates remain authoritative.

[Well-conditioned SDP reformulation](https://arxiv.org/html/2407.14013v2)
derives a reformulation/preconditioner with assumptions on the spectral splitting,
centering and strengthened uniqueness. It is relevant to large low-rank SDPs,
but does not establish generic conditioning or speed for our medium/Ising inputs.
Any future spectral partition must preserve the complete Newton operator; no
approximate rank deletion or mixed precision is authorized.

The magnitude-preserving pivot diagnostic also fails: AlmostSolved after 27
iterations. It is rejected; no global pivot policy changes. Evidence:
`/tmp/sdpx-accuracy-20260916/pivot-decision.json`. A subsequent diagnostic should
compare a mature pivoting factorization on the same small failing KKT before
committing to a new regularization or reduced-system design.

A dense partial-pivot LU diagnostic reuses the existing LAPACK `gesv` provider
on the same statically regularized KKT, retaining original-operator refinement.
It reaches internal Solved in 46 iterations but violates the external primal
gate (1.46883e-4 > 1.31853e-5). Same-T compensated residual accumulation with
FMA product-error recovery reduces this to 20 iterations, but primal 8.70847e-5
and dual 2.37563e-6 still fail. Internal tolerances 1e-9 and 1e-10 both stall at
103 iterations (diagnostics only; defaults unchanged). Reject adoption: stable
pivoting and more accurate residuals alone do not repair this ill-conditioned
case. This prototype refactors each RHS and is not a production performance
implementation. Evidence: `/tmp/sdpx-accuracy-20260916/lu-compensation-decision.json`.

### Stable Ising scaling pilot

PBS **212922.node220** submitted to node49 with 16 physical-core slots, 32 GiB,
2 hours; initially queued. Campaign: `hpc:~/projects/sdpx-scaling-20260916-pilot01`.
Reuse the qualified stable OpenBLAS library from completed 212920; no numerical
source rebuild or candidate changes. Width order is 1/4/16/16/4/1, one process at
a time, each first plus one warmed fresh solve. Internal 1e-42, external 1e-30,
512 bits and one BLAS thread remain unchanged. Pin disjoint physical CPUs within
the allocation and verify effective cone/factor widths. The external driver only
extends its width whitelist to 16. GNU time RSS includes Julia and audit work.
The same Ising input has 322 variables, 20 equalities and 22 PSD blocks of orders
12–16; it is a scaling diagnostic, not evidence for large/multinode performance.

Code inspection: cone scaling and selected Schur work use the reusable worker
pool, while the MPFR condensed QDLDL factor remains serial. Existing raw receipts
report thread counts but no phase profile; infer no measured bottleneck shares
from counts alone. Use the scaling results to choose a phase profile and next
parallel implementation; avoid further hinf3 parameter sweeps without a new
structural hypothesis. Script receipt: `/tmp/sdpx-scaling-20260916/`.

The first scaling cell passes all original metrics and effective-plan checks:
1-core native first/warm times 74.964629/74.936771 s, whole-process peak RSS
669972 KiB. Other widths/reverse samples are pending; no speedup yet. External
assessor `/tmp/sdpx-scaling-20260916/assess.py` checks all six expected cells,
input identities, every point and audit, actual widths, affinity and resource
receipts. Tests reject nonfinite/out-of-bound metrics and mismatched widths.

An isolated scheduling candidate creates up to four tasks per pool worker,
bounded by cone count and existing minimum work per task. Pool thread count,
contiguous disjoint ownership, single-orthant chunking and precision are
unchanged. It targets static-lane stragglers without a new scheduler or raw
pointers. Source: `/tmp/sdpx-scaling-20260916/fine-lanes/`. The focused ownership/join test passes with four lanes on two workers,
including a failed lane whose siblings still complete. Paired full-solve
acceptance is still required before adoption; no measured gain.

Forward scaling cells all pass: native warmed times 74.936771 s (1 core),
24.660085 s (4), 21.164499 s (16). Whole-process RSS is respectively
669972/689800/731060 KiB. Reverse samples are still pending; these preliminary
values show diminishing returns beyond four cores, not a completed scaling claim.

The fine-lane candidate passes pooled condensed equivalence tests at Float64 and
256 bits, in addition to its ownership/join check. PBS **212923.node220** is
submitted with `afterok:212922.node220`, initially dependency-held. Frozen campaign:
`hpc:~/projects/sdpx-lanes-20260916-pair01`. Source comparison confirms only
`cone_parallel.rs` differs between arms; both share observation-only printing
of existing phase timers after `solve()`. Build both libraries before any timing.
Run ABBA at each width 1/4/16, first plus one warmed solve per process, 512 bits,
unchanged internal/external gates, fixed physical affinity, BLAS one thread and
process RSS. Allocation 16 cores/32 GiB/2 hours; no numerical concurrency with
212922. Results and phase shares are pending; no scheduler change is adopted.

### Completed 1/4/16-core Ising baseline

PBS 212922 completed; all six cells / twelve solves pass original-coordinate
512-bit gates and effective-thread checks. Frozen file checks pass. Warm native
medians: **74.929828 s (1), 24.698106 s (4), 21.460200 s (16)**. Relative speedups
are **3.03x** at four cores and **3.49x** at sixteen (parallel efficiencies about
76% and 22%). Peak process RSS ranges 654–662 MiB, 674–680 MiB and 713–714 MiB
respectively, including Julia/audits. This qualifies only the small 22-block
Ising input; large/multinode scaling remains unqualified. Final receipts:
`/tmp/sdpx-scaling-20260916/final/decision.json`.

Dependency 212923 has released and is running its builds. Its existing timer
output distinguishes cone phases from the overall KKT update/solve; it cannot
separate Schur assembly from factorization inside KKT. Add finer instrumentation
in a subsequent frozen experiment only if the measured phase requires it.
The external pair assessor also preserves timer hierarchy/units and refuses
missing or malformed profiles, avoiding double-counted parent/child times.

### Structural factorization opportunity

A no-solve analysis of the exact frozen Ising operator finds 11 positive-block
support components of orders 24,25,27,29,31,31,31,31,31,31,31; P has no entries.
All 20 original equalities connect all components. Dense component storage would
use 9498 scalars versus 103684 for a full 322-square positive block (excluding
border, factors and workspace). This is a storage count, not measured memory or
speed. Input/mapping identities match the qualified campaign. Evidence:
`/tmp/sdpx-scaling-20260916/structure.json`.

The existing DenseBlockSolver handles one dense positive block in Float64 only;
its density threshold excludes this sparse collection of dense components, and
MPFR uses QDLDL. Investigate generalizing that existing elimination to typed,
independent connected positive components and a retained equality border. Reuse
provider Cholesky/triangular/Gram kernels and original-operator refinement; retain
sparse fallback for failed SPD factors or unsuitable structure. Discover structure
once from symbolic support (including P), preserve updates and all equalities,
and do not introduce problem-name special cases or a separate Ising solver.
Profile evidence must still justify adoption and parallel work granularity.

### First measured phase profile (212923, partial)

The first completed baseline cell passes both original-coordinate audits at
512 bits and 50 iterations. Warm native solve is 74.498816 s: KKT direction
solve 34.092678 s (45.8%), KKT update 14.101460 s (18.9%), cone scaling
8.290972 s (11.1%). These are aggregate phase timings from one single-core
cell, not candidate gains or a parallel phase breakdown. Remaining ABBA cells
are still running; no scheduler change is adopted.

Prioritize decomposing the direction-solve phase before implementing the
component-factor proposal. `CondensedKKTSolver::solve_raw` includes two scaling
applications, sampled forward/transpose operations and a reduced solve; outer
refinement repeats this work and evaluates the original operator. The reduced
DirectLDL solve also retains its own refinement. Counts alone do not prove
redundancy: preserve both accuracy contracts. The next isolated profile should
separate reduced backsolve/refinement, sampled operations, scaling applications,
and original-operator residual work, while counting correction calls. Likewise
split KKT assembly from refactor before assigning a factorization speedup budget.
Evidence: `/tmp/sdpx-scaling-20260916/first-phase-decision.json`.


### Fine KKT diagnostic prepared

An external observation-only copy instruments twelve aggregate counters on the
solve thread, flushing after native timers finalize. Raw solve stages are scaling
in, transpose, reduced solve, forward, scaling out; additional counters cover
original residuals, assembly, regularize/refactor, reduced refinement, initial
and correction backsolves, and reduced residuals. Counters are nested where stated;
do not sum a parent with its children. No numerical operations or gates change.
The Float64 and MPFR-256 pooled condensed equivalence tests pass (2 tests).
Profile parser checks reject negative, missing and duplicate counters.

PBS **212924.node220**, initially dependency-held after successful 212923, runs
one first/warm diagnostic at 1 and 16 physical cores, sequentially, 512 bits,
unchanged tolerances, BLAS one thread. Allocation: 16 cores, 32 GiB, 45 minutes;
build timeout 900 s, each process timeout 400 s. Frozen 335 files under
`hpc:~/projects/sdpx-kkt-profile-20260916-pilot01`. Local scripts and isolated
source: `/tmp/sdpx-kkt-profile-20260916/`. Instrumented timings locate costs;
they do not establish a performance improvement. Production source is unchanged.


### Direct parallel sampled forward candidate

An isolated candidate avoids temporary forward contributions and their serial
scatter when sampled descriptors are ordered by their validated disjoint row
ranges. Safe recursive slice splitting writes each block directly into its output
rows. Unsorted descriptors retain the existing path; shared-column transpose
accumulation keeps descriptor order. One typed implementation covers Float64 and
MPFR; no input-name branch or raw pointer. Single-thread behavior is unchanged.

Six exact operator tests pass for ordered/unsorted Float64, 256-bit and 512-bit
inputs, including changing alpha/beta, gaps, empty basis blocks, overlapping
column ranges, workspace reuse and signed results. Ordered pooled forward uses
no contribution-vector storage. Two pooled condensed equivalence tests also
pass. Source and logs: `/tmp/sdpx-sampled-direct-20260916/`. This is an unadopted
candidate, pending phase evidence and full-solve performance acceptance.

212923 partial receipts now contain a complete single-core ABBA and the first
four-core baseline, all numerical gates passing. Single-core medians are
74.435811/74.055095 s (baseline/candidate); scheduling does not change the
single-thread path, so this small variation is not a scheduling gain. Four- and
sixteen-core pairs remain pending. Fine KKT profile 212924 is dependency-held.


The direct-forward candidate's short local synthetic ABBA kernel screen uses
22 ordered blocks (side 16, 31 basis columns), shared variable ranges, fixed
BLAS thread count and exact equality before/after each timed batch. Float64
kernel median reductions at 2/4/8 workers are 7.60%/14.35%/7.05%; 512-bit
reductions are -0.09%/0.17%/1.27%. This is two batches per arm on a local host,
not full-solve acceptance. The high-precision result does not justify another
Ising full-solve campaign without stronger phase evidence. Preserve the
candidate externally; do not adopt or attribute these kernel percentages to
solver speed. Receipts: `/tmp/sdpx-sampled-direct-20260916/kernel-decision.json`.

212923's complete four-core ABBA now passes all gates: baseline 24.639679 s,
candidate 24.919369 s (1.14% slower). Sixteen-core samples remain pending;
no fine-lane scheduling gain is established.


### SDPB direction-solve source review

Reviewed SDPB commit `c5bd57e9ecee5f553901e4e9370e0014f402854c`, frozen under
`/tmp/sdpx-sdpb-core-20260916`. In `compute_Q.cxx`, per-block Cholesky and
`Trsm` prepare the off-diagonal factors once per iteration. The global Gram
matrix uses the shared-memory bigint SYRK context and is Cholesky factored.
`solve_schur_complement_equation.cxx` reuses these factors: local triangular
solves and coupling products, reduction into the border RHS, border solve,
then local reverse solves. The stale initialization comment mentions LU;
the actual implementation calls Cholesky. Block allocation creates MPI process
groups from block costs; it does not run independent duplicate solvers.

This supports the existing proposal to generalize DenseBlockSolver to positive
connected components with a retained equality border, sharing typed provider
kernels and factor reuse. It does not justify dropping SDPX's original-operator
refinement or assuming equivalent conditioning to SDPB. Preserve sparse fallback
and use the pending fine counters to establish whether reduced backsolve,
refinement, or PSD/operator work dominates before prioritizing implementation.

The first 16-core baseline in 212923 passes both numerical audits at 50
iterations: warm native 20.814071 s, direction solve 8.988936 s, KKT update
5.195204 s, cone scaling 1.805868 s. These remain single-cell phase observations;
16-core candidate/reverse cells are pending. The direction phase is still the
largest named phase; there is no evidence yet for its internal dominant kernel.


### Scheduling acceptance complete — adopted

PBS 212923 completed successfully. All 12 cells / 24 solves pass original
512-bit audits with 50 iterations; 623 source and four library/native identity
checks pass. Warm native medians baseline/candidate at 1/4/16 cores:
74.435811/74.055095 s, 24.639679/24.919369 s, 21.227065/19.612755 s.
Sixteen-core improvement is **7.60%**: both candidate samples are faster than
both baseline samples. Four cores regress **1.14%**, explicitly retained in the
record; single-core variation is 0.51% with an unchanged execution path.

Adopt the exact tested `cone_parallel.rs` task-budget change: up to four tasks
per existing pool worker, bounded by cone count and minimum work. No extra
threads, unsafe ownership, precision changes, input-specific or core-count
thresholds. All 283 solver library tests pass, including Float64/MPFR parallel
and failure-joining checks. Source comparison before integration confirmed this
was the only differing solver source file. This validates this Ising campaign,
not MOSEK/SDPB parity or 64/256-core scaling. Earlier partial notes above are
superseded by this final result. Evidence:
`/tmp/sdpx-scaling-20260916/pair-final/decision.json`.

212924 has started the frozen pre-adoption fine KKT profile build. Keep its
baseline identity; do not overwrite it with the newly adopted scheduling code.

Local release FFI build of the adopted source passes; dylib SHA256
`63ee51f1c892b76a8b0607b3f10777ba67e265284cb15fbe4b499cac280b8d94`. Prior stable library remains saved externally.


### Fine KKT profile complete; residual reuse candidate

PBS 212924 completed, all four original-coordinate audits pass; 335 source and
three library/native identity checks pass. Warm instrumented times are 76.551789 s
(1 core) and 21.147831 s (16). This is the frozen pre-scheduling baseline and
not a comparative speed claim. Aggregate single/16-core seconds:

| Operation | 1 core | 16 cores |
|---|---:|---:|
| Original-operator residual, 184 calls | 15.3153 | 3.3407 |
| Raw forward, 184 calls | 4.5689 | 0.9813 |
| Raw transpose, 184 calls | 4.1050 | 0.7474 |
| Raw scaling in/out | 9.9157 | 2.1873 |
| Reduced solve, including refinement | 2.2656 | 2.6496 |
| Assembly, 51 calls | 0.2801 | 0.0772 |
| Regularize/refactor, 51 calls | 1.5856 | 1.7803 |

Reduced solve includes initial backsolves (0.5871/0.7027 s), correction
backsolves (0.2799/0.3139 s) and reduced residuals (1.3909/1.6236 s); do not
add these to its parent again. Factor-only optimization cannot explain away
the single-core gap. On 16 cores the serial reduced solve becomes significant.
Evidence: `/tmp/sdpx-kkt-profile-20260916/final/decision.json`.

An external candidate reuses the original A*x-bz vector already evaluated by
solve_raw, only for the initial residual at that same returned x. Subsequent
residuals after adding a refinement correction recompute A*x. No new scratch,
precision conversion, relaxed stopping rule or omitted residual equation.
Subtraction/accumulation order differs, so numerical acceptance remains required.
The candidate applies to sampled and ordinary condensed operators at all types.
31 condensed tests pass, including a new cached-versus-fresh rounding bound and
a poisoned-cache check for a changed point at Float64 and MPFR precisions.
Source/logs: `/tmp/sdpx-residual-reuse-20260916/`. Not adopted: next run is a
frozen full-solve ABBA against the newly adopted scheduling baseline, with the
same original accuracy gates. Keep new builds separate from prior profile arms.


### Residual reuse isolated acceptance protocol

Completed PBS212926 froze 622 files and compared ABBA at 1/16 physical cores,
512 bits and BLAS1, first plus warm solves, with unchanged original-coordinate
gates and RSS. Only condensed.rs differed between arms. All 283 candidate core
tests passed. Protocol/evidence: `/tmp/sdpx-residual-reuse-20260916/`.
Final isolated results appear below; broader adoption was decided by PBS212929.


### Ordered row-gather prototype for reduced residuals

The measured 16-core reduced residual cost is 1.624 s and remains serial.
Existing symmetric CSC multiplication visits columns and scatters into output
rows. An external symbolic row-gather prototype records `(value index, x index)`
for each row in exactly that original traversal order, including duplicates,
then processes independent rows through a supplied existing Rayon pool.
It keeps numerical values in the authoritative CSC storage and retains the
same multiply/add expression and beta scaling. No numerical atomics, parallel
reduction tree, matrix-value clone, new thread pool or precision-specific engine.

Exact parity tests pass at Float64, 256 and 512 bits with 1/2/4 workers, upper
and lower storage, duplicates, gaps, zeros, cancellation, signed outputs and
updated values reusing the same symbolic map. Source/log:
`/tmp/sdpx-row-residual-20260916/`. This is an unintegrated kernel prototype;
cache memory, work granularity, pool reconfiguration, whole-solve correctness
and performance still need verification. It is independent of the residual
forward-reuse candidate currently running as PBS 212926.


The row-gather prototype is now integrated in an external solver copy, but the
separate SymmetricRowPlan was retired after locating the existing SparseParallel
component. Move that component/tests into the algebra layer and extend its
symbolic constructor for symmetric contributions; ordinary residuals and direct
KKT refinement share the same flattened indices, weighted row partitioning,
thread-pool configuration and value-update behavior. Symmetric multiplication
retains the original multiply/add expression rather than using gemv's specialized
alpha branches. The direct solver attaches the existing cone pool; the condensed
solver forwards its current pool on each update, including reconfiguration.
A precision-weighted work threshold keeps small products serial.

All 286 solver library tests pass in this external shared implementation,
including existing pool reconfiguration and live LP/SOCP tests plus symmetric
Float64/256/512 exact parity tests. Full-solve timing, metadata-memory impact and
explicit activation coverage for the new KKT path remain pending. No second
sparse engine or extra worker pool is introduced. Stable source remains unchanged.
Evidence: `/tmp/sdpx-row-residual-20260916/shared-test.log`.


Explicit new-KKT-path activation tests now pass at Float64, 256 and 512 bits.
A 192-variable symmetric system crosses the work threshold; configure its
existing pool through 4/1/2/4 workers, assert the effective pool width and stable
metadata allocation, update P values, and compare both residual vectors and
infinity norms exactly against the original serial implementation. This adds
three focused checks beyond the 286 passing library tests. A no-SDP build check
also passes after removing an unnecessary import introduced by the module move.
Evidence: `/tmp/sdpx-row-residual-20260916/activation-test.log` and
`no-sdp-check.log`. Performance and memory qualification remain pending.

The first residual-reuse candidate cell in 212926 passes both accuracy audits;
warm single-core baseline/candidate are 74.776033/72.385951 s. These are one
sample per arm, not the final paired result. Preserve all remaining ABBA cells
before deciding adoption; the stable source still contains only the previously
accepted scheduling change.


### Parallel residual isolated acceptance protocol

Completed PBS212927 froze 624 files and compared 16-core/512-bit ABBA, BLAS1,
first plus warm solves, unchanged original gates and RSS. Both arms included
accepted scheduling and excluded forward-residual reuse, isolating the shared
row-gather change. It followed PBS212926 without numerical overlap. Evidence:
`/tmp/sdpx-row-residual-20260916/`. Final results and combined adoption follow.


### Residual reuse isolated Ising result

212926 completed: all eight cells / sixteen solves pass 512-bit original
accuracy gates, all with 50 iterations. Source/library identities: 622/four
checks pass. Native warm medians baseline/candidate are **74.451782/72.293299 s**
at one core (**2.90% faster**) and **19.371735/18.318031 s** at sixteen cores
(**5.44% faster**). Subsequent Float64 and MPFR acceptance is recorded under PBS212929 below. Evidence: `/tmp/sdpx-residual-reuse-20260916/final/decision.json`.


### Medium independent dual audit gap

The historical medium driver independently checked original equalities and
primal PSD feasibility but used solver summaries for the dual side. An external
revised driver retains returned primal/equality/PSD dual data and independently
reconstructs stationarity, dual PSD feasibility, dual objective and gap after
native timing. Its formulas pass six analytic checks (including off-diagonal
trace contributions and rejected invalid data), plus four assertions on a real
Julia frontend equality+PSD problem. No solver-side certificate stage is added.

One stable-library medium run passes the historical gate but fails the added
stationarity/max(1,norm(q,Inf)) <= 1e-6 gate: **1.91649243224e-6**. Recomputing the
saved Float64 inputs/duals at 256-bit audit precision gives the same result,
so this is not Float64 audit accumulation error. Dual PSD violation is zero;
reconstructed dual objective agrees within 5e-16, and objective gap is 8.21546e-7.
This is a stricter independent metric than the old gate, not a new speed result
or a regression attributed to an unadopted candidate. Preserve the numerical
threshold and qualify both arms under the same audit; do not promote internal
Optimal into full-accuracy credit. Audit files and saved point:
`/tmp/sdpx-residual-broad-20260916/`. The stable solver/settings were unchanged.


### Parallel KKT residual Ising acceptance complete

212927 completed. All four cells / eight solves pass 512-bit audits at 50
iterations; every serialized x/s/z vector matches the baseline exactly. 624
source and four library/native checks pass. Sixteen-core native warm median
**19.251350 -> 15.133014 s**, a **21.39% reduction**. Candidate process RSS is
716540/727728 KiB versus baseline 731116/728628 KiB; no measured peak-memory
regression on this input. This is the shared row-gather candidate alone, excluding
forward-residual reuse. Other phase times also change, so do not attribute the
entire gain solely to one residual kernel. Broader acceptance and combination
with forward reuse remain pending; no production integration yet. Evidence:
`/tmp/sdpx-row-residual-20260916/final/decision.json`.

### Medium normalization clarified

Inspection of both SDPX and the local Clarabel.rs reference confirms the same
dual criterion: norm2(stationarity)/max(1,norm_inf(q)+norm2(x)+norm2(z)). From the
saved original point and frontend dual matrices (whose Frobenius norm equals
canonical svec norm), the independent 256-bit accumulation reconstructs
**2.51277821712e-7**, agreeing with the reported 2.51278e-7. Thus the point passes
the established 1e-6 criterion. The 1.91649e-6 target-only normalization is a
stricter, different diagnostic; it is not evidence of a faulty implementation
of the upstream stopping rule or a candidate regression.

Align the new external audit with that existing criterion at the unchanged
1e-6 tolerance, retaining raw norm2, infinity norm and target-only normalization
in reports. No solver stopping rule or in-flight benchmark gate changes.
The strengthened audit also checks original dual PSD feasibility and independently
reconstructed objectives/gap. Evidence:
`/tmp/sdpx-residual-broad-20260916/normalization-check.json`. Earlier wording of
a pending precision issue applies only to the stronger target-only diagnostic.


### Combined cross-problem acceptance protocol

Completed PBS212929 froze 666 files at `sdpx-combined-20260916-accept01`.
ABBA covered LP_bore3d, SOCP_axis_wide, SDP_truss2; medium/Ising at 1/16 cores;
SOCP_nb at 16 cores; ordinary nonsampled SDP at 256/512 bits. Forty-eight
receipts include separate high-precision audit processes. All are exposed
regression cases, not holdout evidence. Runs were sequential, BLAS1, with
original gates, native/API timing separated and RSS recorded. The 16-slot PBS
allocation was pinned to 1 or 16 physical cores, never a 128-core campaign.

The frozen assessor checks raw numeric bounds, precision, widths, arm identity,
resources and missing cells. Its tests reject excessive residuals, NaN, wrong
labels and missing resources. Identities: `assessor-identity.json`; tests:
`test_assessor.py`; source review: `integration-review.json`, all under
`/tmp/sdpx-combined-20260916/`. All 289 candidate core tests passed. Final
measurements and adoption follow; superseded interim samples are omitted here.

### Deferred scaling-cache experiment

Deferred isolated PSD scaling-cache experiment (`/tmp/sdpx-scaling-cache-20260916`):
the cone already caches the authoritative upper triangle of R R^T at every
scaling update. The high-precision condensed update recomputes the same SYRK.
Replace that redundant product with a copy of the existing cache, retaining
lower-triangle mirroring and all inverse-factor calculations. Identity resets
also update the source cache. Added exact cached-versus-recomputed checks at
Float64/256/512 bits; all 290 core tests pass. The release kernel screen passes exact output equality: at 512 bits, 500
order-16 SYRK calls cost about 0.0570 s versus 0.00010 s for cache copies;
order-32 costs about 0.415 s versus 0.00063 s. This confirms eliminated work,
but the absolute order-16 saving is only about 0.114 ms per call on this host.
Defer a dedicated cluster campaign: no >=2% full-solve gain is established.
No production code has been changed. This is
based on the combined candidate, separate from the completed frozen 212929 arms.



### Combined residual optimizations accepted and integrated

212929 completed: all 48 receipts pass fixed numerical gates; 666 source and
6 library/native hash checks pass. The reviewed nine source paths are integrated
exactly, including the shared sparse-row component move; the unrelated scaling-G
cache experiment is not included. External backup and evidence:
`/tmp/sdpx-combined-20260916/{pre-integration,final-decision.json,final-build-logs}`.

| Case / physical threads | Baseline native seconds | Candidate native seconds | Reduction |
|---|---:|---:|---:|
| Ising 512-bit / 1 | 74.050580 | 70.151168 | 5.27% |
| Ising 512-bit / 16 | 19.614802 | 14.070192 | 28.27% |
| medium Float64 / 1 | 11.413268 | 11.412418 | 0.01% |
| medium Float64 / 16 | 12.474882 | 11.554621 | 7.38% |
| ordinary SDP 256-bit / 1 | 11.509830 | 11.503130 | 0.06% |
| ordinary SDP 512-bit / 1 | 27.175027 | 26.689454 | 1.79% |

Small Float64 warm API changes span -0.53% to +1.76%; SOCP_nb at 16 cores
regresses 0.41%, not a material change in this screen. Medium-16 process RSS
increases from 969404 to 1030842 KiB (~60 MiB); Ising-16 RSS is effectively flat
(724004 to 723200 KiB). Candidate Ising speedup from 1 to 16 cores is about 4.99x,
only 31% parallel efficiency; medium still has no net multi-core acceleration.
These are same-host OpenBLAS comparisons, not new MOSEK/SDPB parity evidence.
Release build passes; the Julia frontend suite passes against the new library,
including prepared handles, original-coordinate outputs, high precision through
2048 bits and thread-budget tests. Production dylib SHA256:
`8e4017e2fcbaf220950ddc2a5a2239a4244ccb16ae518fa19298a1f7fddf664f`.
The bounded phase diagnostic is running as 212931.node220 in
`hpc:~/projects/sdpx-medium-profile-20260916-pilot01` (16 cores, 32 GiB, 45 min);
it uses the integrated numerical source with observation-only counters.


212931 diagnostic failure (preserved): all three children stopped before solver
execution because the copied Julia Manifest retained the previous campaign's
absolute SDPX path. Source-origin checks rejected it; there are no valid timings.
A first scoped retry is prepared in `sdpx-medium-profile-20260916-pilot02` with
only the environment binding corrected. Reuse the exact compiled diagnostic
library SHA256 `4b2d8e8a4cd38468c81dbdf4a0cf0afcbb48b220fd3247fbdec8e51e956e6fe5`;
retain the same numeric source, inputs, settings, evaluator and three cases.
The completed performance qualification 212929 and production build are unaffected.

Retry submitted as **212932.node220**; pilot01 remains intact. No second
compilation is needed; final source/library checks remain in the PBS script.


### Medium assembly bottleneck confirmed (212932)

The corrected three-cell diagnostic completed; original numerical gates pass,
364 source and 3 library hashes pass. Evidence is
`/tmp/sdpx-medium-profile-20260916/decision02.json`.
Medium assembly costs 7.441 s at one core and 7.834 s at 16; regularized refactor
is 1.317/1.307 s. Scaling synchronization is only 0.004/0.010 s. Thus assembly,
not factor-provider replacement or cached scaling-G, is the current main target.
Ising-16 differs: assembly 0.065 s, refactor 1.803 s, outer residual 2.083 s,
raw scaling 0.811+1.000 s. Preserve distinct measured priorities.

The original medium PSD supports contain 1887/1862/1887/1862/1862 columns,
8766015 contribution cells and 1781328 union-Schur cells. Before any possible
chordal transformation this exceeds the two-Schur-cache cutoff; the complete
Float64 contribution cache is only 66.88 MiB. An isolated candidate at
`/tmp/sdpx-assembly-budget-20260916` allows the greater of the existing relative
budget and 128 MiB, counted using the scalar storage size at every precision.
It changes only parallel-buffer eligibility, retaining original arithmetic and
ordered scattering. Validate actual activation, solve time and memory before
adoption; no production code or timing claim yet.

The isolated cache-budget candidate passes all 31 condensed tests. Existing
overlap tests were updated to expect bounded parallel activation instead of
the old relative-only fallback; they still require exact Schur values and
solutions versus serial execution, unchanged memory on pool reconfiguration,
and original-operator residual checks. No numerical comparisons were relaxed.


Assembly-budget screen submitted as **212933.node220** in
`hpc:~/projects/sdpx-assembly-budget-20260916-pair01`. Freeze 652 files; exact
candidate diff is allocation eligibility plus updated overlap expectations.
Use the integrated baseline library `98efe89d…`, build candidate before timing,
then eight ABBA cells (medium-16 and Ising-512/16). Allocation 16 cores/32 GiB,
45 minutes; medium processes <=300 s and Ising <=400 s. Capture native timing,
process RSS, iteration count and unchanged original-coordinate numerical gates.
The new environment binding was checked before submission. Local evaluator and
inherited gate hashes are in `/tmp/sdpx-assembly-budget-20260916/assessment-identity.json`.
This is a development screen, not a broad qualification or parity comparison.


212933 completed with all eight numerical gates, 652 source checks and four
library/native checks passing. Medium-16 improves 11.511656 -> 8.412762 s
(26.92%), process RSS 1022812 -> 1087782 KiB (~63 MiB). Ising-512/16 regresses
14.506874 -> 15.156954 s (4.48%); both candidate samples are slower than both
baseline samples. Do not adopt yet. The original Ising block supports require
9820 contribution cells, exactly the old two-Schur allowance (2*4910), so
buffer eligibility should already be true. Check actual result identity and
run an isolated reverse-order Ising repeat using the same two frozen libraries,
without rebuilding or interleaving medium, before attributing this to code.
Evidence: `/tmp/sdpx-assembly-budget-20260916/final-decision.json` and
`final-build-logs/`. The production library remains `8e4017e2…`.


The four original Ising warm outputs have exactly identical x/s/z and all take
50 iterations. Isolated reverse-order repeat submitted as **212934.node220**
in `sdpx-assembly-budget-20260916-repeat01`: candidate/baseline/baseline/candidate,
four Ising-only processes, same frozen binaries (no rebuild), 16 cores/32 GiB,
30 minutes, per-process timeout 400 s. All 626 source/input/library identities
are frozen, with the same numerical evaluator. Preserve the first observed
regression regardless of whether this repeat reproduces it.


212934 completed: all four numerical gates, 626 source and four library checks
pass. Isolated Ising medians are baseline 14.316636 s versus candidate
14.616446 s (2.09% slower). Keep both this result and the initial 4.48% regression;
the cause is not established. The identical binaries do not reproduce the full
initial gap, but neither does the repeat establish non-regression. No further
unchanged-library repeats are planned. **Do not adopt the 128 MiB floor yet.**
Medium's repeatable 26.92% gain establishes a useful parallel opportunity.
Next investigate processing overlapping block contributions in bounded batches
under the existing relative memory budget, preserving cone-order accumulation,
rather than retaining every block's contribution buffer simultaneously. Keep
one shared implementation across scalar precisions and leave the already-qualified
small-Ising scheduling behavior intact. Any replacement needs a fresh paired
measurement and all original numerical gates; no speed prediction is accepted.


Bounded-batch replacement implemented externally at
`/tmp/sdpx-assembly-batches-20260916/source`. It retains the original all-block
cache path when eligible. Otherwise a few reusable contribution slots consume
at most two Schur value arrays; contiguous batches compute concurrently and
scatter in original cone order before reusing slots. Shared compute/scatter
helpers avoid a second numerical implementation. No precision-specific path,
new pool, atomic accumulation or approximation is introduced.
Tests compare exact Schur entries and solve vectors against serial operation,
plus original-operator residuals, A/P updates and pool reconfiguration. Scratch
addresses and capacities must remain unchanged across iterations. Initial 31
condensed tests pass; all 290 core tests pass, including the added 512-bit overlap
case. Production code remains unchanged pending measured acceptance.
The next paired measurement should group cases rather than interleave medium
with Ising, and build both arms in the same campaign to reduce ambiguities.


Bounded-batch screen submitted as **212935.node220** in
`hpc:~/projects/sdpx-assembly-batches-20260916-pair01`: rebuild both arms with
identical toolchain/BLAS settings before timing; medium-16 ABBA followed by
Ising-512/16 BAAB (eight cells). Source/environment preflight checks passed,
651 files frozen; allocation 16 cores/32 GiB/45 minutes. Keep the original
300/400-second case limits and external precision gates. This replaces the
interleaved-case measurement layout, not its historical evidence. Assessor and
inherited gates are frozen in `/tmp/sdpx-assembly-batches-20260916/assessment-identity.json`.
No production code changed; measured acceptance remains pending.


Post-freeze edge-case validation for the batch candidate: two additional local
Float64/512-bit tests vary PSD live-column counts so slots shrink and grow
between blocks, while retaining empty/non-PSD blocks, A/P updates and thread
reconfiguration. Both pass exact serial Schur/solution comparisons, original
operator residuals, and stable scratch addresses/capacities. Only local test
code changed; the frozen cluster numerical source is untouched. The pre-addition
test file is retained as `/tmp/sdpx-assembly-batches-20260916/tests-frozen.rs`;
new results are in `heterogeneous-tests.log`. Job 212935 remains in its build phase.


### Bounded-batch screen completed — not adopted

PBS **212935.node220** completed with exit 0; all eight original numerical
receipts, 651 source checks and four library/native checks pass. Both arms were
rebuilt in the same job. Medium-16 ABBA medians: 11.489994 -> 10.937234 s
(4.81% gain), RSS 1020084 -> 1048798 KiB. Ising-512/16 BAAB medians:
14.170730 -> 14.498735 s (2.31% regression), RSS 725850 -> 728454 KiB.
The candidate remains external; the production dylib still hashes to `8e4017e2…`.
Evidence: `/tmp/sdpx-assembly-batches-20260916/final-decision.json`,
`final-build-logs/` and `phase-comparison.json`. Candidate Linux SHA256:
`2166d3c092a29473b4edf1d8040f154de992038144db1cbd683e7fe1a61488e3`.

Ising median phase times regress across cone scaling (1.05008 -> 1.10190 s),
KKT update (3.97761 -> 4.02451 s) and KKT solve (5.76957 -> 5.91396 s).
This does not isolate a numerical assembly cost or prove a compiler/layout
cause. Do not attribute the whole regression to the batch scheduler. The batch
candidate extracts shared compute/scatter helpers even for the original cached
path, so isolate that transformation before further algorithm changes. For
medium, inspect available workers per actual batch rather than assuming the
whole-problem block count represents simultaneous work. Any revised scheduling
must retain the same pool, exact cone-order accumulation and bounded memory.


### Batch-local sparse column scheduling — numerical prototype

External candidate `/tmp/sdpx-batch-lanes-20260916/source` assigns sparse-column
lanes using the reusable contribution-slot count rather than all PSD blocks.
Within a batch, independent block tasks can expose disjoint sparse output-column
tasks to the same Rayon pool; no extra threads, atomics or changed reduction order.
The original fully cached dispatch is unchanged relative to the previous batch
prototype. This is not yet an isolation of that prototype's shared-helper change.

Five overlapping 64-dimensional sparse PSD blocks exercise two contribution
slots and 2/4/8/1/4 worker reconfiguration. Float64 and 512-bit tests require exact
serial Schur equality after A updates, stable plan/output storage on repeat,
and the expected bounded lane count. Both targeted tests and all 294 core tests
pass (full suite 5.79 s). Initial test compilation used a non-Copy enum in an
array repetition; the test constructor was corrected without numerical changes.
Evidence: `identity.json`, `tests-fixed.log`, `full-tests.log` in that namespace.
No solve-time improvement is claimed; source remains external pending a bounded
paired screen and unchanged original-coordinate medium/Ising gates.


Batch-local lane candidate submitted as **212936.node220** in
`hpc:~/projects/sdpx-batch-lanes-20260916-pair01`. Both arms rebuild before
medium-16 ABBA and Ising-512/16 BAAB, eight cells on node49 with 16 cores,
32 GiB and 45 minutes. Existing 300/400-second process limits and all original
numerical gates remain fixed. Preflight verified exactly two candidate source
changes against the prior campaign, matching local tested hashes, the Julia
package's namespace binding, and 651 frozen files (generated Python bytecode
excluded). Submission initially queued. Assessor/driver identities are in
`/tmp/sdpx-batch-lanes-20260916/assessment-identity.json`.
The previous batch regression remains recorded; this submission does not
qualify that candidate or establish performance parity.


### Dense assembly attribution prepared

Static original-medium support analysis finds 1475–1504 dense-path columns per
PSD block versus 374–387 sparse-path columns. The current structural cost model
assigns only 0.20–0.37% of work to sparse pairs. This is before presolve/chordal,
not measured runtime attribution; it limits expectations for the sparse-only
batch-lane candidate. Evidence: `/tmp/sdpx-batch-lanes-20260916/medium-column-estimate.json`.

External diagnostic `/tmp/sdpx-dense-phases-20260916/source` instruments exact
column planning, dense transform/packing, tiled dense dots, tiled output stores
and sparse pairs. Atomic counters aggregate across workers; durations are summed
worker spans and overlap with selected-total parents, not additive wall time.
No numerical operators or production files changed. All 31 affected condensed
tests pass; an initial import-before-module-doc error was corrected before testing.

PBS **212937.node220** submitted with `afterany:212936.node220` and initially held
for that dependency. It runs medium-1, medium-16 and Ising-512/16 after the paired
screen, using the existing 16-core/32-GiB/45-minute allocation and case timeouts.
Preflight confirms exactly three instrumented source files, correct namespace
binding and 363 frozen files. Evaluator retains original numerical gates and
checks complete, balanced phase receipts. Parser checks accept complete counters
and reject a missing counter. Driver/evaluator identities are frozen locally in
`assessment-identity.json`; no performance conclusion is available yet.


### Research after repeated assembly candidates stalled

The [SDPARA sparse-Schur paper](https://optimization-online.org/wp-content/uploads/2010/09/2732.pdf)
(2010, §4.1–4.2) distributes stored Schur elements by estimated formula cost,
including reusable intermediate construction, and aligns sparse assembly storage
with the factorization input. This supports evaluating work-balanced dense
output tiles rather than equal PSD-block counts. That application is an SDPX
hypothesis, not a result of the paper. Preserve fixed per-entry arithmetic and
ordered cross-cone accumulation. First obtain 212937's actual dense-stage costs;
choose tile ownership and bounded scratch only for the measured dominant stage.

The newer [SDSL-Solver preprint](https://arxiv.org/html/2604.23979v1)
(April 2026, §3–4) uses Block Jacobi/BBD decomposition and hybrid MPI+OpenMP.
Its filtering/diagonal corrections apply to a preconditioner while Krylov uses
the original operator. Some reported NetworkPlan comparisons are individual
linear solves because reference IPM runs fail, not full optimizer comparisons.
Its 1e-8 linear-residual experiments do not establish SDPX's original-coordinate
accuracy or arbitrary-precision performance. Keep BBD/local elimination as a
future multi-node reference; do not introduce a filtered Krylov backend while
medium is assembly-bound and previous iterative candidates failed their gates.
Do not infer MPI, high-precision or MOSEK/SDPB parity from these source results.


### Batch-local lane screen completed; dense diagnostic repair

PBS212936 completed all eight numerical receipts, 651 source checks and four
library/native checks. Medium-16: 11.523612 -> 11.221488 s (2.62% gain), RSS
1020542 -> 1048454 KiB. Ising-512/16: 14.885211 -> 14.791965 s (0.63% gain),
RSS 720598 -> 719916 KiB. Candidate Ising samples span 14.455678–15.128252 s,
straddling both baseline samples; do not interpret 0.63% as established gain.
Medium samples both improve, but the current candidate still needs broader
qualification and resolution of earlier batch-path regressions before adoption.
Keep production unchanged and await dense-phase evidence rather than launching
another unchanged-binary repeat. Evidence: `/tmp/sdpx-batch-lanes-20260916/final-decision.json`
and `final-build-logs/`. Candidate Linux hash:
`32e2e9e0bf3324829eb0fe818c3d699157bf6daadb99e37f7e769f1851da4e9b`.

PBS212937 failed before any solve (exit1, 9 s): its copied retry script assumed
an existing candidate library and omitted compilation. `libraries/*` was absent.
No numerical or timing evidence resulted. Preserve `sdpx-dense-phases-20260916-pilot01`.
Repair in a fresh pilot02 namespace restores the explicit bounded release build
before library hashing and timing, retains identical instrumented source and
numeric gates, and rebinds the Julia environment. This is the first diagnosed
retry; a script parse check alone was insufficient to detect the omitted build.


Repaired dense-phase pilot submitted as **212941.node220**, initially queued.
Preflight verified the explicit build/copy/hash/solve ordering, unchanged numerical
source versus pilot01, correct package binding and 363 frozen files. Failed logs
are retained locally in `/tmp/sdpx-dense-phases-20260916/failed01`; retry script,
manifest, job ID and assessor identities are in `retry02/`. No duplicate live
diagnostic job remains.


### Bounded dense-dot row prototype

External source `/tmp/sdpx-dense-rows-20260916/source` splits independent rows of
an existing dense accumulation tile using the solver's current Rayon pool.
It retains one arithmetic helper for serial/parallel execution, original
entry/FMA order, unchanged transform packing and serial Schur publication.
No extra numeric scratch, new pool or atomics. Activation is restricted to the
serial-assembly fallback with an explicit configured pool and a sufficiently
large tile; the original all-block cached path remains outer-parallel. MPFR's
unbatched path is unchanged. This is separate from the batch-buffer candidates.

All 290 core tests pass, including a 513-column dense fixture requiring exact
serial/parallel results across 2/4/1/4 workers, coefficient/representative changes,
and stable accumulator allocation. Existing 150-column compact-storage tests
remain unchanged. The wider fixture initially inherited an invalid <50% packing
assumption; only that fixture-specific assertion was replaced with the full
panel bound, with numerical equality and independent matrix checks retained.
Logs and source identity are in the external namespace. No production change or
performance claim; await the frozen dense-phase profile before a timing campaign.


### Dense-phase attribution completed (212941)

The repaired diagnostic completed with exit0, all three numerical gates, 363
source checks and three library/native checks passing. Medium-1 spans (seconds):
selected assembly 7.43924; transform/pack 4.51821; dense dots 1.75298; publication
0.93777; sparse pairs 0.19876; exact-column planning 0.03121. Medium-16 shows
4.64422/1.82327/0.96039 s for transform/dot/publication, respectively. Its assembly
remains serial in this baseline, so additional pool threads do not accelerate it.
Ising sampled assembly does not enter the dense-dot path. All counts complete;
evidence: `/tmp/sdpx-dense-phases-20260916/final-decision.json` and
`retry02/build-logs/`. This instrumented run is not a speed comparison.

Transform/packing is the largest hotspot (about 61% of selected medium assembly).
The ready row-parallel prototype targets about 24%, so its potential is limited;
run one bounded whole-solve screen before deciding whether to keep it. Further
work should separate coefficient-product generation, GEMM and packing within
the dominant transform phase. Exact-column hash caching and sparse-only lane
tuning are low priorities given these actual measurements. Preserve all earlier
M4 and batch-candidate rejection evidence; do not repeat those unchanged.


Dense-row screen submitted as **212942.node220**, initially queued, namespace
`hpc:~/projects/sdpx-dense-rows-20260916-pair01`. Preflight confirms only condensed.rs
differs numerically between arms and matches the locally tested SHA256
`84137b1b73cf7db79d8069d535e7f501aa41eae6ca3d1267f4b8907762b15b73`.
Freeze 651 files; build both arms before medium-16 ABBA and Ising-512/16 BAAB.
Keep 16 cores/32 GiB/45 minutes and 300/400-second case limits, BLAS1, original
numeric gates and RSS. Assessor identities are frozen in the local namespace.
No production adoption; broader precision/one-core qualification remains required
if this screen produces a repeatable gain without regression.


### Transform substage diagnostic submitted

External `/tmp/sdpx-transform-phases-20260916/source` extends the qualified
observation-only dense probe with coefficient-product, GEMM and suffix-packing
counters. All 31 affected condensed tests pass; parser checks reject missing
GEMM counters. The evaluator requires matching chunk counts for the three new
stages and retains all original numerical gates. Parent/child worker spans
remain overlapping diagnostics, not additive wall-time or acceptance timings.

PBS **212943.node220** is dependency-held after **212942.node220** in
`hpc:~/projects/sdpx-transform-phases-20260916-pilot01`. Reuse the established
three diagnostic cases and 16-core/32-GiB/45-minute allocation. Preflight verifies
explicit build-before-timing, the correct package binding, exactly two additional
instrumentation-file changes from 212941 and 363 frozen files. Local identities
are in `identity.json` and `assessment-identity.json`. No production change;
the row-parallel screen remains running, and transform optimization awaits these
substage measurements rather than a speculative change of BLAS provider.


### Additional independent-family sources located

The official [DIMACS archive](https://archive.dimacs.rutgers.edu/Challenges/Seventh/Instances/)
provides copo14 (copositivity) and filter48_socp (PAM filter design). Downloaded
only these small first-family members, pinned compressed/raw SHA256, and inspected
SeDuMi metadata without solver construction or execution. copo14 has 1275 equations,
364 nonnegative coordinates and fourteen order-14 PSD blocks; filter48_socp has
969 equations, 931 nonnegative coordinates, one SOC49 and one PSD48. Despite its
name, filter48_socp is mixed SDP/SOCP, not a pure SOCP holdout.

Files/provenance: `/tmp/sdpx-dimacs-reservation-20260916/`. No matching case/family
was found in current catalog or searched sibling benchmark manifests, but full
exposure review and independently checked SeDuMi-to-svec conversion remain
necessary before catalog registration. No performance or generalization claim.
GitHub tree API inspection was rate-limited; the official DIMACS source supplied
the data instead. Large FIR files from CBLIB were not downloaded simply to fill
coverage; keep development budgets short. Pure SOCP holdout coverage remains open.


### Dense-row parallel candidate rejected (212942)

All eight original numerical receipts, 651 frozen source checks and four
library/native checks pass. Medium-16 regresses 11.659576 -> 11.960167 s
(2.58%), RSS 1000076 -> 1020138 KiB. Ising-512/16 regresses 14.215686 ->
15.244187 s (7.23%), RSS 725444 -> 725862 KiB. Reject the candidate; do not
integrate or run broader acceptance. Evidence: `/tmp/sdpx-dense-rows-20260916/final-decision.json`
and `final-build-logs/`. Production remains unchanged.

Ising never enters the batched dense-dot branch, so its regression is not
explained by dense-row parallelism itself. The candidate changes a shared generic
PSD workspace and helper structure; neither layout/code generation nor host
variation is established as the cause. Do not claim either without isolation.
Avoid further modifications of this rejected candidate while transform-stage
profile 212943 runs. Prefer minimal local kernel changes for the next hypothesis.

DIMACS candidate conversion has progressed without a solver run: copo14 maps to
1834 variables/3109 rows/4578 nonzeros; mixed filter48_socp maps to
2156 variables/3125 rows/48262 nonzeros. External `convert.py` forms a symmetric
upper-svec map, retains all original equations/cones, and checks its operators
and objective against independently reconstructed full symmetric matrices at
four deterministic random points per instance. All checks and existing finite
CSC/cone-shape validation pass. Source/JSON hashes, versions and errors are in
`/tmp/sdpx-dimacs-reservation-20260916/conversion.json`. These remain candidate
reservations, not registered fresh holdouts or evidence of solver success.


### Transform substage profile completed (212943)

All three original numerical gates, 363 source checks and three library/native
checks pass. Medium-1 transform/packing 4.47911 s divides into coefficient
products 1.32230 s, GEMM 2.74434 s and suffix packing 0.39802 s (1444 balanced
chunks); remaining setup is small. Medium-16 gives 1.39995/2.88572/0.41469 s,
respectively. Ising does not enter these stages. Source identities and receipts:
`/tmp/sdpx-transform-phases-20260916/final-decision.json`, `final-build-logs/`.
Do not compare this instrumented single run as an optimization gain.

Prioritize GEMM/input generation, preserving per-entry accumulation and the fixed
BLAS/thread budget. Evaluate a bounded panel or provider-kernel change without
retaining all cone Schur contributions or changing the generic PSD workspace
layout. Packing alone is too small to justify another architecture change.

### DIMACS holdouts registered without solving

Registered `SDP_copo14` and mixed `SDP_filter48_socp` as holdout-only, with source
and JSON hashes, DIMACS attribution, independent family/exposure notes and
180-second/4096-MiB per-process planned budgets. No solver constructed or run.
The available SDPX/sibling/reference benchmark text and manifests contain no
prior family names; deleted history is explicitly outside that evidence.
Pure SOCP coverage is still missing. Do not use these reservations for tuning.

`benchmark/research/import_sedumi.py` supplies reusable, hash-checked import with
optional NumPy/SciPy. Two hand-written convention/error tests and eight catalog
integrity/materialization tests pass. Its two outputs match the independently
audited external conversions byte-for-byte. Prior regression sets are unchanged;
all holdouts are integrity checked, not merely the first entry. No runtime solver
validation or certificate stage was added.


### Native GEMM output-column prototype

External `/tmp/sdpx-panel-gemm-20260916/source` extends the existing `xgemm_pool`
provider method for native Float64/Float32 by splitting disjoint output columns.
The condensed serial-assembly fallback passes its existing pool to batched GEMM;
all-block outer assembly still passes no inner pool. No PSD field/layout change,
new pool, numeric scratch or altered convergence rule. MPFR's existing provider
override remains unchanged. Tile zero retains serial semantics; one-worker and
small/degenerate cases retain the opaque call. The screen must keep BLAS1 to
avoid nested vendor threads.

All 290 core tests pass (5.30 s), including native transpose combinations,
nontrivial leading dimensions, alpha/beta accumulation, short final tiles,
1/4-worker and tile0/4/8 dispatch, and untouched padding/trailing sentinels.
The provider may choose a different microkernel for a smaller output width;
no bitwise-equivalence or whole-solve speed claim is made. Original-coordinate
acceptance remains required. Local source identities and logs are frozen in
`identity.json` and `full-tests.log`; production source is untouched.


Native panel GEMM screen submitted as **212944.node220**, initially queued, in
`hpc:~/projects/sdpx-panel-gemm-20260916-pair01`. Preflight verified the three
changed files against locally tested hashes, Julia binding and explicit build
ordering; 651 files frozen. Build both arms, then medium-16 ABBA and Ising-512/16
BAAB under BLAS1, 16 cores/32 GiB/45 minutes and unchanged 300/400-second case
limits. Assessor/driver hashes are recorded in the external namespace. No
production adoption or speed conclusion; broader acceptance follows only if
this complete paired screen qualifies.


### Coefficient-product fusion prototype

Independent external source `/tmp/sdpx-coefficient-fusion-20260916/source` groups
four consecutive contributions to the same coefficient-product output column,
retaining the exact scalar FMA order while reducing repeated accumulator loads
and stores. Only the existing generic coefficient kernel changes; no pool,
workspace-layout change, tolerance or new precision path. All 292 core tests pass;
new Float64/256/512-bit checks compare exact full buffers with the original loop,
including offset/stride padding and remainder groups.

A short release ABBA kernel screen (0.22 s total) passes exact output checks:
f64 n57 baseline 13.10/8.49 ms versus candidate 6.37/5.76 ms; n116 baseline
27.98/24.00 ms versus candidate 15.67/15.48 ms. Baseline drift is visible; do not
turn these into an end-to-end gain claim. At 512 bits/n16 both are about 24.5 ms,
with no clear gain. The dense synthetic kernel overstates medium coverage:
original-support static analysis puts 48.5–54.2% of dense-path plan entries in
four-entry groups, before presolve/chordal/exact reuse. No input-specific branch.
Evidence: `tests.log`, `kernel.log`, `medium-coverage.json`, `identity.json`.
This candidate is separate from running panel-GEMM acceptance; neither is merged.

Research library follow-up: the entire 70-test suite passes after adding DIMACS
reservations/import support (`/tmp/sdpx-dimacs-reservation-20260916/research-suite.log`).
No holdout solve was performed during these integrity/tool tests.


Coefficient fusion screen submitted as **212945.node220**, dependency-held after
212944, in `hpc:~/projects/sdpx-coefficient-fusion-20260916-pair01`. This isolates
the coefficient kernel against the integrated stable source: medium **one core**
ABBA, Ising-512/16 BAAB, BLAS1, eight cells. Preflight caught an ineffective
shell-quoted driver replacement before submission; editing/transferring the
local Python driver fixed it. The final check confirms Julia/solver/affinity
width1 for medium, unchanged width16 for Ising, exact tested candidate hash,
correct package binding and 652 frozen files. Build both arms before timing;
16-core/32-GiB/45-minute allocation with unchanged 300/400-second process limits.
Assessor expected labels were fixed before reading results; all original gates
are inherited unchanged. Broader nonsampled high-precision acceptance remains
necessary if the screen qualifies. Production is unchanged.

Panel-GEMM interim: all four medium gates pass, but median time rises
11.381786 -> 13.404283 s (~17.8% slower). Seven of eight receipts are available;
final Ising candidate remains pending. No adoption. A possible next hypothesis
is transposing the batched product/output so split GEMM calls share the small
scaling factor while reading disjoint large-panel regions. The present split
shares the whole large left operand across all calls; repeated BLAS packing is
plausible but not measured. Any layout experiment must account for packing and
publication costs, preserve numeric acceptance, and add no generic workspace
fields. Do not claim this hypothesis explains the observed slowdown yet.


### Native column-split GEMM rejected (212944)

All eight numerical receipts, 651 source checks and four library/native checks
pass. Medium-16 regresses 11.381786 -> 13.404283 s (17.77%), RSS 1010824 ->
1055380 KiB. Ising-512/16 is 14.510667 -> 14.520453 s (0.067% slower), RSS
721296 -> 729550 KiB. Reject the candidate; do not infer a meaningful Ising
change. Evidence: `/tmp/sdpx-panel-gemm-20260916/final-decision.json` and
`final-build-logs/`. Coefficient fusion 212945 has started independently.

### Transposed-panel numerical prototype

External `/tmp/sdpx-transposed-panel-20260916/source` stores each transformed
panel as the transpose, computing Ginv^T * panel^T rather than panel * Ginv.
This exposes many output columns with a small common left factor, avoiding the
previous split's shared large operand. Packing indices change consistently;
allocation size is unchanged and there are no new PSD workspace fields.
The original precision and provider API remain; changed BLAS microkernels may
reassociate floating-point operations, so original-coordinate gates remain
required. All 290 core tests pass. A local release diagnostic includes both
GEMM and packing at dimensions57/116 with requested BLAS1 and pool1/4/16; no
whole-solve claim or cluster submission before inspecting that diagnostic.


### Coefficient fusion screen completed (212945)

All eight original-coordinate numerical gates, 652 frozen-source checks and
four library/native checks pass. Medium-one-core median 11.399926 -> 11.403569 s
(0.032% slower, no gain); Ising-512/16 median 14.648398 -> 14.322739 s
(2.223% reduction). The Ising result barely clears the screening threshold;
repeatability and nonsampled high-precision coverage are still unqualified.
Do not integrate or attribute the gain to the kernel on these two samples.
RSS medians are 873426 -> 909726 KiB for medium and 729546 -> 727990 KiB
for Ising. Evidence: `/tmp/sdpx-coefficient-fusion-20260916/final-decision.json`
and `final-build-logs/`. Production remains unchanged.

### Isolate transposed layout from inner threading

The local transposed-panel release diagnostic passed every packed-coordinate
comparison, including packing costs. Sixteen workers regressed at both tested
sizes; one-worker results were better, with visible baseline drift. These are
short Accelerate microbenchmarks, not evidence of Linux whole-solve gains.
A separate external candidate `/tmp/sdpx-transpose-serial-20260916/source`
changes only the batched product orientation and matching pack indices in
`condensed.rs`, preserving allocation size and the existing provider/threading
interfaces. It excludes coefficient fusion and all native pool overrides.
All 289 core tests pass (5.41 s); source identity is recorded alongside the
candidate. Next qualify this layout with fixed-input whole-solve timing before
integration; no production change or speed claim yet.


Transposed-serial whole-solve screen submitted as **212946.node220**, confirmed
running, in `hpc:~/projects/sdpx-transpose-serial-20260916-pair01`.
Medium-one-core ABBA and Ising-512/16 BAAB; both libraries rebuilt before timing,
BLAS1, unchanged 300/400-second solve limits, 16 reserved physical cores,
32 GiB and 45-minute cap. Preflight verified the sole changed numerical file,
its tested SHA-256, package binding, driver widths and 652 frozen files.
The copied assessor and eight expected labels are fixed before results.

Coefficient-fusion attribution review: `sampled_adapter.jl` supplies a factor
for every active Gram cone; `compute_schur_selected` returns from its sampled
branch before either `coefficient_product` call. The recorded Ising route is
`sampled_factors` / `condensed_sampled_qdldl`. Thus the observed 2.223% Ising
difference does not demonstrate the changed kernel's benefit. Medium, the
actual target, is unchanged. Do not spend an unchanged Ising repeat to qualify
this candidate; shelve it unless evidence from an affected workload provides
a new reason. The generic high-precision kernel screen was also essentially
flat. No integration is warranted by these results.


### Residual stage overlap prototype

External `/tmp/sdpx-residual-overlap-20260916/source` evaluates the independent
original-operator residual products and H*z scaling concurrently on the same
existing Rayon pool when scaling already has multiple lanes. Each stage retains
its original arithmetic and internal block schedule; final residual addition and
norm remain after the join. Single-worker/fallback execution remains sequential.
No new pool, workspace field, numeric buffer, precision change or extra residual
acceptance criterion. This targets block-tail imbalance in the measured Ising
outer-residual phase (about 2.08 s), not a claim of removing its whole cost.

All 289 core tests pass (5.34 s, `final-tests.log`). Extended pooled-equivalence
tests compare every residual entry and infinity norm exactly between serial and
parallel implementations for both fresh and reused-forward paths, Float64/256
bits, pool reconfiguration and assembly fallback. Existing sampled solve tests
also pass at Float64/256/512 bits. An initial mistyped test filter matched zero
tests; the full suite was then run and verified to include the extended tests.
Source identities are stored in `identity.json`. This candidate is independent
of transpose/fusion changes and awaits whole-solve timing; production unchanged.
212946 transposed-layout campaign remains running on its original job handle.


Residual overlap screen submitted as **212947.node220**, dependency-held after
212946. Namespace `hpc:~/projects/sdpx-residual-overlap-20260916-pair01`;
medium/Ising both 16 physical workers, BLAS1, ABBA/BAAB, eight cells and unchanged
external gates. Preflight verifies the exact two tested files, 653 frozen files,
package binding and driver. Both arms build before timing; established
16-core/32-GiB/45-minute allocation and 300/400-second solve limits. The same
accepted stable baseline is used, with no transpose or coefficient-fusion edits.

212946 interim: all four medium numerical gates pass; native medians
11.388608 -> 11.243528 s (about 1.27% reduction), below the 2% retention threshold.
Ising cells remain incomplete; no final decision or integration. API/process
wall time includes first-run startup and is not substituted for native timing.


### Transposed serial layout: no qualifying whole-solve gain

212946 completed: all eight original-coordinate gates, 652 frozen-source and
four native/library checks pass. Medium-one-core 11.388608 -> 11.243528 s
(1.274%); Ising-512/16 14.415735 -> 14.220289 s (1.356%). Both are below the
retention threshold, and sampled Ising bypasses the changed batched product.
Do not integrate. `/tmp/sdpx-transpose-serial-20260916/final-decision.json` and
`final-build-logs/` retain full evidence. Dependent residual-overlap job212947
has started on its original handle.

### Larger Ising conditioning diagnosis (no solve)

Reviewed the existing iteration94, 768-bit Lambda11 replay: factor/CSC products
agree to roughly 1e-175 or better, while sampled equation1099 fails its unchanged
component-relative gate at 2.16e-22. This is not evidence of a mapping bug.
A new sealed-input diagnostic `/tmp/sdpx-lambda11-scales-20260916/summary.json`
loads the same input and saved point at768bits, without native build, solve or
EVD. Across1099 columns, nonzero A column infinity norms range 1.288231e-25 to
10.13978;162 are below1e-8. Objective coefficients span2.907108e-92 to13.32651.
The largest saved |x| is7.834819e78. Even |x_j|*||A_j||_inf reaches2.899914e58;
simple column normalization alone does not remove the solution's enormous
scale. Worst-audit column1099 has norm1.171106e-24, q=-2.907108e-92 and
x=5.291024e41. Existing cumulative Ruiz bounds are1e-4/1e4, but extending them
is not established as a cure: upstream residual norms explicitly undo Ruiz.
Investigate near-null directions/cancellation and alternate exact model
representations before another costly Lambda11 solve. Preserve all original
external gates and runtime stopping rules; no approximate rank removal or
point-dependent tolerance presented as a uniform guarantee.


SDPB comparison refinement: retrieved only the14 retained x-block files from
qualified job212626, verifying every SHA against its original provenance before
reading1099 coefficients in numeric block order. Its max |lambda| is1.515307e79,
versus7.834819e78 for the SDPX saved point; the worst SDPX audit column1099 has
SDPB lambda1.515307e79. Thus huge primal multipliers are also present in a valid
SDPB solution, not evidence by themselves of an SDPX defect or disposable null
space. Different optimal primal points need not match coefficientwise.
Evidence: `/tmp/sdpx-lambda11-scales-20260916/point-comparison.json`.

Source audit of SDPB `compute_dual_residues_and_error.cxx` finds an MPI maximum
of absolute residuals; `compute_feasible_and_termination.cxx` compares that value
directly with its threshold. It is not the external component-relative test.
Its pmp2sdp basis already incorporates sqrt(sample_scalings), which is preserved
in the shared Q input; adding the same scaling again is not a new optimization.
Equal numeric internal tolerances do not mean equal accuracy across these two
solver formulations. Keep fixed original-equation gates; do not copy an SDPB
absolute gate into SDPX or claim that a larger Ruiz cap solves the discrepancy.
Next conditioning work needs equation/trajectory evidence, rather than removing
large variables or treating their magnitude as failed convergence by itself.


### Isolate generic x86 FMA-call overhead

Read-only disassembly of the accepted Linux baseline identifies the Float64
`coefficient_product` loop calling through r15 for every scalar mul_add.
Relocation0xaecde8 resolves to local symbol `fma` at0xad5420, whose compiler
builtins implementation has runtime FMA dispatch. AVX FMA instructions elsewhere
in faer's kernels do not prove these generic scalar loops were vectorized.
This is direct code-generation evidence, not yet a measured speedup.

PBS **212948.node220**, dependency-held after212947, compares byte-identical
stable source arms with only candidate Rust flag `-C target-feature=+avx2,+fma`.
No fast-math, changed precision or arithmetic expression. Before building/running,
the compute-node script requires AVX2/FMA CPU flags; it records both RUSTFLAGS,
CPU details and library identities. Both arms build before medium-one-core ABBA
and Ising-512/16 BAAB with the unchanged external gates.652 frozen files,
16cores/32GiB/45min,300/400-second solve caps. Local evidence/scripts:
`/tmp/sdpx-fma-target-20260916`; remote matching `-pair01` namespace.
This is a CPU-specific experiment, not a portable distribution change. If it
wins, separately qualify a portable runtime-dispatched implementation or an
explicit supported CPU build; do not ship an AVX2-only library as universal.

An exact-rational raw-medium proportional-column census also completed without
solving: beyond sign-opposite reuse, only25/24/25/24/0 additional representatives
can be removed across the five blocks. Most apparent opportunity repeats the
already rejected signed reuse candidate. No new proportional-reuse branch is
justified by this raw, pre-Ruiz upper-bound screen. Evidence:
`/tmp/sdpx-proportional-screen-20260916/result.json`.


Residual-overlap212947 completed: all eight numerical gates,653 source and four
native/library checks pass. Medium16 median11.544091 ->11.335211s (1.81%);
Ising512/16 median14.440208 ->13.322176s (7.74%). This qualifies for broader
acceptance and a matched repeat across relevant worker counts, not immediate
integration or SDPB parity. Frozen full evidence:
`/tmp/sdpx-residual-overlap-20260916/final-decision.json`, `final-build-logs/`.
CPU-target212948 has started; keep its independent candidate separate.


Residual-overlap full acceptance submitted as **212949.node220**, confirmed
held after212948; remote `sdpx-residual-overlap-20260916-accept01`, local
`/tmp/sdpx-residual-overlap-20260916/acceptance`.666 frozen files; only the two
locally tested numerical/test files differ between arms. Both FFI libraries and
ordinary high-precision executables rebuild before timing with the same generic
CPU/BLAS configuration.56 predeclared receipts extend the former48-cell gate
with medium/Ising at4workers; original tolerances and metric limits unchanged.
The ordinary256/512-bit benchmark remains single-worker; do not present it as
nonsampled multiworker performance coverage. Pool numerical equivalence has
separate core-test evidence. Assessor negative checks pass and expected labels
are unique; hashes frozen locally before reading results. Allocation remains
16cores/32GiB,2-hour acceptance cap and per-process timeout limits. No holdout
input is consumed by this development regression.


CPU-target212948 interim: all four medium-one-core gates pass. Native median
11.416334 ->8.883278s (~22.2% reduction). Ising remains incomplete; no final
adoption or comparison to MOSEK. Disassembly confirms coefficient_product now
uses vfmadd213pd/sd directly, replacing per-element calls to the compiler-builtins
FMA dispatcher. Same source/settings/provider; this is a build-target benefit,
not a changed algorithm. Partial receipts:
`/tmp/sdpx-fma-target-20260916/current-decision.json`.

External portable prototype `/tmp/sdpx-runtime-fma-20260916/source` keeps a single
inlined dense-Schur arithmetic body. Its existing method dispatches once per
block to an AVX2/FMA target-feature wrapper only when both runtime CPU checks
pass; other platforms retain the generic body. No new dependency, workspace,
fast-math, per-element feature checks or duplicate algorithm. All289 library
tests pass on local ARM (5.44s), plus the changed compact-panel test. This only
validates fallback behavior: x86 compile/accelerated execution remain pending.
The compact-panel test now compares generic and accelerated outputs bitwise
when running on supported x86, including sparse skipping and changed widths.
The coefficient-product helper is not force-inlined in this prototype; inspect
its generated code and measure whole solves before assuming it captures all
of the whole-library target flag's gains.
Rust's documented runtime detection/target-feature pattern:
https://doc.rust-lang.org/stable/core/arch/
and https://doc.rust-lang.org/stable/reference/attributes/codegen.html .
Do not distribute the experiment's AVX2-only whole library as universal.


### CPU-target screen complete; portable test queued

212948 completed with all eight numerical gates,652 source checks and four
library/native checks passing. Medium-one-core11.416334 ->8.883278s (22.188%);
Ising512/16 14.662893 ->14.607267s (0.379%, no meaningful gain). Evidence:
`/tmp/sdpx-fma-target-20260916/final-decision.json`, `final-build-logs/`.
This establishes a qualifying CPU-specific Float64 build improvement only;
not portable adoption, algorithmic gain, or new MOSEK/Clarabel comparison.

Portable runtime dispatch screen **212950.node220** is held after full residual
acceptance212949. Namespace `sdpx-runtime-fma-20260916-pair01`;652 frozen files,
sole candidate file SHA378ae4d05050289479d1d7b0638f729c4e3ea7cd30a7c3564290aa6d8d3b0d3b.
No global target-feature flag. On the compute node, require actual AVX2/FMA
support and run the release compact-panel test comparing both bodies bitwise,
requiring exactly one passing test before rebuilding both FFI arms and timing.
Original medium-one-core ABBA, Ising512/16 BAAB, numerical gates and limits stay
fixed.16cores/32GiB/45min;900-second bounded build/test processes. This candidate
excludes residual overlap. Do not combine apparent percentages from independent
experiments; any combined release needs its own validation.


### Pure SOCP holdout reserved without solving

Added official CBLIB `strictmin_2D_43_dual` (geometric ARAP distortion) as
holdout-only `SOCP_strictmin_2D_43_dual`:101676 variables,111757 rows,
285726 A entries and10080 SOC5 blocks plus equalities. Source inspection found
no integer, PSD or power cones. Existing repository/reference exposure search
found no matching name/family; no claim about deleted history. Several rejected
selection candidates had integer/power/PSD variables or excessive download size;
none were solved or silently relaxed.

Conversion uses existing MOI/SDPX affine export without solver setup. Independent
CBF parsing verifies the full equality coefficient/constant multiset up to
row permutation/sign, exact objective and four deterministic SOC embeddings.
An initial verifier assumed equality rows came first; inspection showed MOI
places SOC rows first, and correcting that row layout made checks pass without
changing converted coefficients. The earlier environment lacked MOI as a direct
load dependency; the retry used the existing current-project/environment stack,
without adding packages. Compressed payload, provenance and CBLIB license now
live in the research holdout directory. All70 research tests pass. Evidence:
`/tmp/sdpx-socp-reservation-20260916/`. Reserve180seconds/4096MiB explicitly at
final acceptance; metadata does not override runner limits. Never use this case
for current optimization timing or tune on its final outcome.


### Enforce reserved per-case budgets at execution

`benchmark/research/run.py` now clamps process timeout and memory to any
reserved case cap, without extending a shorter caller/campaign limit. Effective
limits are saved in each process receipt. Invalid/nonfinite/nonpositive or
boolean caps are rejected before launch. This supersedes earlier notes that
reservation budgets were only descriptive: the four current holdouts now
actually cap execution at180seconds/4096MiB under this runner. External reference
adapter still rejects holdouts, so no reference path silently bypasses this cap.

All71 research tests pass (0.64s), including an execution-boundary test that
checks what reaches the owned-process supervisor, receipt values, shorter
campaign limits and malformed caps. No solver or holdout solve was run for these
tests. Evidence: `/tmp/sdpx-socp-reservation-20260916/budget-tests.log`.
212949 partial assessment has13/56 receipts and no numerical failures; no final
performance conclusion until all expected cells and final identities are checked.


Residual-overlap additional QA: ownership review confirms disjoint mutable
product/scaling workspaces and value-owned MPFR limbs, with only shared immutable
inputs across the join. Added local-only512-bit variants of pooled condensed
and overlapping-memory fallback equivalence tests. Both include exact residual
entry/norm comparisons for fresh/reused-forward paths and worker reconfiguration.
All15 selected MPFR512 tests pass (4.56s), including these two new variants.
Evidence: `/tmp/sdpx-residual-overlap-20260916/extended-512-tests.log` and
`extended-tests-identity.json`. Only the external test file changed; the numerical
source and already-frozen212949 campaign remain byte-identical. This is numerical
parallel coverage, not a new performance measurement. Latest assessed full-run
snapshot has24/56 receipts with no gate failures;212949 remains running. Completed
one-core ABBA medians: medium11.404927002→11.3715749815seconds (0.29% faster),
Ising51270.813399839→70.8485233165seconds (0.05% slower); both are noise, not
performance credit. Four-/sixteen-core and ordinary high-precision acceptance
remain incomplete.212950 is still dependency-held; no candidate is integrated.
Evidence: `/tmp/sdpx-residual-overlap-20260916/acceptance/current-decision.json`.


### SDPB timing-anchor provenance check

The copied Ising512 reference from212554 uses1e-34 internal tolerances and
records process elapsed time; the current screen uses1e-42 and warmed native
solve time. Its audited point remains an objective anchor at the fixed1e-30
external gate, but these archived times are not matched speed evidence. After
candidate acceptance, repeat SDPX/SDPB on the same allocated host and physical
widths with matched precision/tolerances, separate native/process timing and
unchanged original-coordinate audits. Do not infer parity from these old times.
Source hashes and review: `/tmp/sdpx-reference-provenance-20260916/review.json`.


### Residual-stage overlap accepted after full campaign212949

Integrated the reviewed residual-stage overlap into the shared condensed solver:
operator products and cone scaling use disjoint buffers on the existing pool;
serial fallback and per-output arithmetic order stay unchanged. No new buffers,
workers, tolerance changes or sampled-only branch. Includes exact residual
checks at Float64/256/512 bits, fresh/reused products and pool reconfiguration.

All56 expected campaign receipts pass unchanged numerical gates. Final666source
and6binary/library checks pass. One-/four-core differences are below2%; the
sixteen-core Ising512 median improves14.876019649→13.299804466seconds (10.60%),
replicating the initial7.74% direction. Medium16 improves11.606889869→
11.374204843seconds (2.00%, borderline; initial1.81%). Ordinary256 regresses0.92%;
ordinary512 is flat (+0.04%). Small LP/SOCP/SDP and SOCP_nb16 gains are below2%.
Acceptance is supported primarily by repeatable high-precision Ising16 benefit,
not a claim of general LP/SOCP speedup or MOSEK/SDPB parity. Candidate Ising1→16
scaling is5.33× (33.3% efficiency); medium remains essentially unscaled.

Evidence: `/tmp/sdpx-residual-overlap-20260916/acceptance/final-decision.json`
and `final-build-logs/`; integration identity and local check logs are in the
parent experiment directory. Integrated Rust suite:291/291 pass; release FFI
rebuilt successfully; Julia frontend:508 assertions across32 testsets pass.
The loaded FFI hash is recorded in `integration-review.json`.
Runtime AVX2/FMA screen212950 is now running independently against its frozen
pre-overlap baseline; any future combined candidate requires its own checks.


### Fresh matched Ising512 comparison submitted

PBS212965.node220 (`sdpx-matched-20260916-01`) is dependency-held after212950:
16physical cores/32GiB/1hour, controller deadline55minutes. Reuses the accepted
212949 candidate binary (SHA08b82b213931e5d3407929048f68e202176f47ad773d727a7c0e2dd7a9f92a1c)
and exact source/input; no rebuild or new numerical changes.370files frozen.
Pinned SDPB executable and dynamic dependencies are recorded by the controller.
At each of1/4/16physical cores: SDPX first+3warmed fresh solves, then3fresh SDPB
MPI solves, sequential on one node,512bits and1e-42internal tolerances. Every
returned point must pass unchanged1e-30original-coordinate audits and objective
agreement. Native/API/process timing and sampled process-tree/rank RSS remain
separate; no existing checkpoint reuse. A Linux numerical gate precedes timing.
Existing controller is copied externally with widths/core count and expected
point count generalized to this fixed sweep; production harness is unchanged.
Local preparation: `/tmp/sdpx-matched-20260916/`. Pending results are not parity
or scaling evidence. This campaign tests the accepted residual-overlap release,
not the still-experimental runtime FMA candidate or their combination.


### Runtime FMA screen: verified dispatch and remaining scalar helper

212950 is running; x86 compact-panel equivalence test actually executes and
passes (1test). Candidate disassembly contains vector FMA in the dense Schur
dot loop, but still calls out-of-line Float64 `coefficient_product`, whose
inner loop retains indirect scalar arithmetic calls. Thus this portable wrapper
does not yet reproduce the whole-library target-feature optimization. Evidence:
`/tmp/sdpx-runtime-fma-20260916/disassembly/`. Current3/8receipts have no numerical
failures; partial medium candidate median9.83536seconds, incomplete baseline.

External follow-up `/tmp/sdpx-runtime-fma-inline-20260916/` changes only that
shared helper annotation from `inline` to `inline(always)` on top of the frozen
runtime-dispatch candidate. It preserves one arithmetic implementation and
operation order, allowing the checked CPU context to reach this loop. All289
local core tests pass (5.30s); this is ARM fallback coverage, not x86 performance
or portability qualification. No follow-up numerical change is integrated or
submitted yet. Requires x86 exact-equivalence, generated-code inspection and
medium/Ising screen before further acceptance; MPFR code-size/regression remains
part of the check. Both runtime candidates exclude accepted residual overlap.


Inline follow-up submitted as212966.node220, dependency-held after matched
SDPB campaign212965.16cores/32GiB/45minutes,652frozen files. Baseline is the
runtime-FMA candidate from212950 (not the original generic source); sole source
change is `inline(always)` on `coefficient_product`, SHAa55ddb009553c9fb04601664859dcc97da8e8f24a0013b66c74d7687a75ea82b.
Before timing, run the actual x86 bitwise-equivalence test and rebuild both FFI
arms. Eight fixed medium1/Ising512-16cells, original limits and external gates;
no holdout or tolerance changes. This isolates incremental inlining benefit and
must not be presented as a combined release speedup. Parent screen212950 has
6/8receipts, no failures, completed medium medians11.4255677235→9.8353608815seconds;
Ising repetitions and final identities are still pending.


Runtime-dispatch screen212950 completed: all8numerical receipts pass,652source
and4library checks pass. Medium1 medians11.4255677235→9.8353608815seconds
(13.92% improvement); Ising512-16medians14.711022457→14.506035010seconds
(1.39%, below credit threshold). Explicit x86 bitwise test passes and disassembly
confirms vector FMA for dense Schur dots. The whole-library AVX target screen had
22.19% medium gain; remaining scalar coefficient helper motivates212966, not
an assumption that independent gains multiply. Evidence:
`/tmp/sdpx-runtime-fma-20260916/final-decision.json` and `build-logs/`.
No runtime-FMA code is integrated yet; retain accepted residual-overlap release
while evaluating the inline follow-up and eventual combined broad acceptance.
An initial local assessment raced an unfinished rsync and saw missing files;
waiting for that same transfer to finish and reassessing resolved it, with no
remote rerun or numerical repair.


Matched212965 is running on node49 with16distinct physical cores; Linux gate
passed. Preparation repair before any SDPB rank launch: copying the archived
SDPB executable with `copyfile` lost its execute bit. Restored owner execution
0644→0744, preserving SHA3ee1bee7417d853945bc94cfa7a39508d3266ebd1db9752f53509fb6613281fe.
Receipt: remote `results/212965.node220/executable-mode-repair.json`. No content,
settings, checkpoint or input changed; local preparation now uses `copy2`.

Timing interpretation: archived SDPB outputs expose integer-second `Solver
runtime` and millisecond `iterations.json` total_time/iter_time. Source review
shows runtime is measured from program start before input loading, so it must
not be mislabeled as pure iteration time. Retain these scopes separately from
SDPX native time, Julia API and process-wall time; use iteration records to
analyze per-iteration cost, not integer rounding to claim small speed changes.


Matched212965 SDPX leg completed: first+3warmed solves at1/4/16cores all pass
original-coordinate audits with50iterations each. Warm native medians:
71.192046877 /22.813339422 /13.2590836seconds; same-host speedups3.12× at4cores
and5.37× at16cores (78.0%/33.6% efficiency). Observed cell process-group RSS:
700092416 /703959040 /752492544bytes, including Julia and external audits.
These are SDPX scaling measurements only; SDPB repetitions are now running.
External summary and four numeric/protocol/command/incomplete-snapshot checks:
`/tmp/sdpx-matched-20260916/summarize.py` and `test_summary.py`. The summary checks
all21points, fixed512bits/1e-42/1e-30protocol, binary identity, actual thread/rank
binding, original metrics and objective agreement before declaring completion.
It keeps unlike timing scopes separate and never emits a premature cross-solver
ratio. Final whole-campaign identities remain required.


### Next parallel hypothesis: fuse scaling sync and cached block assembly

External candidate `/tmp/sdpx-sync-assembly-20260916/source` starts from the
accepted residual-overlap release. Existing outer parallel blocks now update
their scaling data and compute their own cached Schur contribution in one task,
removing the intervening all-block barrier. One extracted block-compute helper
serves fused and old paths; cone-order scatter and per-entry arithmetic order
remain unchanged. Inner parallel dispatch and uncached memory fallback retain
the existing two-stage path. No added cache, pool, precision-specific algorithm
or change to numerical gates.291core tests pass (5.59s), including exact serial/
pooled Schur and residual comparisons at Float64/256/512bits and reconfiguration.
A test-only wrapper plus comments were added afterward; production cargo check
passes. No cluster timing submitted and no production integration yet. Profile
predicted benefit is limited by existing scaling-sync cost; retain only if a
bounded same-host screen demonstrates repeatable gain without regressions.


Sync/assembly fusion screen213040.node220 submitted after212966, using the
accepted residual-overlap binary's exact numerical source as rebuilt baseline.
653files frozen; one changed numerical source file, no FMA candidate mixed in.
16cores/32GiB/45minutes; same eight-cell medium16ABBA/Ising512-16BAAB driver,
300/400-second solve caps and fixed numerical assessor. Both arms rebuild before
sequential timings. Final33condensed tests pass (2.28s) after the test-wrapper
annotation; full291test result and production check remain recorded. No production
change or performance claim yet. Local namespace `/tmp/sdpx-sync-assembly-20260916/`.


### Matched Ising512 completed: speed advantage, remaining iteration/scaling gap

212965 completed all21point audits (12SDPX,9SDPB) at512bits,1e-42internal and
1e-30external tolerances. Final370frozen-file checks pass; dependency/source
identities before/after match. Summary independently checks precision, actual
binary, input identities, fixed settings, physical/rank binding, point metrics
and all-objective agreement. Frozen legacy rank receipts lack hostname; locality
is verified using the one-host PBS nodefile and each rank PID/start-time key in
the locally sampled memory records. Five summary tests pass, including rejecting
missing/locality-mismatched rank records. No numerical gates were changed.

| Physical cores | SDPX warm native median (s) | SDPB iteration-time sum median (s) | SDPB process median (s) |
|---|---:|---:|---:|
|1|71.1920|127.056|129.5643|
|4|22.8133|35.847|37.3526|
|16|13.2591|15.660|18.1371|

Each median uses3repetitions. SDPX always50iterations, SDPB201recorded iterations.
Timing scopes differ: SDPX native solve versus summed SDPB per-iteration timers;
SDPB cumulative timer medians are127.340/36.014/15.909seconds. SDPX full process
cells include first+3warm solves and audits, so do not compare those cell totals
against one SDPB process. This finite benchmark shows an advantage from fewer
iterations, not universal SDPB parity or a rigorous equal-scope ratio. Average
iteration cost remains approximately2.3–3.4× higher for SDPX under these timers.
SDPX16core speedup5.37× (33.6% efficiency), SDPB iteration sum8.11× (50.7%).

Observed SDPX cell process-group RSS700092416/703959040/752492544bytes includes
Julia and external audits; SDPB solve-group medians99622912/290504704/1071054848
bytes include MPI ranks, with external audits separate. These are sampled sums
of RSS, not private/PSS memory or pure solver workspace. The larger Lambda11
accuracy issue,64/256cores and multi-node qualification remain unresolved.
Evidence: `/tmp/sdpx-matched-20260916/final-summary.json`, `final-source.txt`,
`results/212965.node220/`. Keep the sync/assembly experiment directed at measured
per-iteration and scaling gaps; retain it only after its own paired evidence.


Inline screen212966 generated-code check confirms coefficient AXPYs are now
inlined into the runtime-checked FMA body: vector `vfmadd213pd` precedes the GEMM
call, with scalar FMA tails and no out-of-line Float64 coefficient_product symbol.
Candidate library13132904bytes versus13114336baseline (+18568bytes). Actual x86
bitwise test passed. Evidence: `/tmp/sdpx-runtime-fma-inline-20260916/disassembly/`.
First2/8receipts pass; medium1single samples9.83893→8.99753seconds are preliminary,
not medians or integration evidence. Complete all repeated gates and library/source
checks before deciding, and retain separate acceptance for combined optimizations.


### Inline screen completed; combined runtime-FMA acceptance submitted

212966 passes all8numerical receipts,652source and4library checks. Relative to
runtime dispatch alone, medium1median9.829209406→8.9980119875seconds (8.46%).
Ising512-16median15.016494691→14.4040481005seconds (4.08%), but sampled assembly
bypasses the modified coefficient helper: do not attribute that timing change to
this arithmetic optimization. It is no observed regression in this screen.
Evidence: `/tmp/sdpx-runtime-fma-inline-20260916/final-decision.json`.

External combined candidate adds runtime AVX2/FMA+inlining to the accepted
residual-overlap source, preserving the exact common Schur body and current
512-bit tests. All291local core tests pass (5.63s). No production integration.
PBS213041.node220 is dependency-held after sync/assembly screen213040.666files
frozen;16cores/32GiB/2hours. Both FFI and ordinary-SDP binaries rebuild; actual
x86 equivalence test must pass before timing. Reuses the unchanged56-receipt
LP/SOCP/SDP, medium/Ising1/4/16 and ordinary256/512 acceptance protocol against
the accepted residual-overlap baseline. Fixed limits, tolerances and original
coordinate gates; no holdout exposure. This candidate excludes experimental
sync/assembly fusion. Evidence and scripts:
`/tmp/sdpx-runtime-fma-combined-20260916/acceptance/`.


### Sync/assembly fusion rejected after paired timing

213040 completed all8numerical gates plus653source/4library identity checks.
Medium16median11.365876934→11.567953869seconds (1.78% slower), Ising512-16
13.239480858→13.8865858515seconds (4.89% slower). Reject this candidate; no
production change and no unchanged repeat. Removing one barrier did not improve
whole-solve performance; do not infer an unmeasured scheduling cause from totals.
Evidence: `/tmp/sdpx-sync-assembly-20260916/final-decision.json` and `build-logs/`.
Current stable source keeps separate block scaling-sync and assembly stages.
Combined FMA acceptance213041 remains running and excludes the rejected fusion.


### September 16 publication verification

Both final speculative candidates were rejected; production Rust/Julia source
matches the fully accepted runtime-FMA snapshot byte-for-byte. Workspace tests
(`cargo test --workspace --locked --offline` with Accelerate and Faer) pass;
Julia frontend validation has 508 passing assertions. Research and Ising harness
tests also pass. Publish the stable engine, exact preprocessing, benchmark
protocol/holdout updates and associated documentation; retain known accuracy,
large-instance and distributed-scaling limitations above. No rejected candidate,
local binary, temporary timing artifact, credential or sibling repository is
included. Local logs: `/tmp/sdpx-publication-20260916-*.log`.
