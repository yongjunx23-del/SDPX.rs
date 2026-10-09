# SDPX development plan

Updated 2026-10-09. [AGENTS.md](AGENTS.md) defines working rules and numerical
contracts; [architecture](docs/ARCHITECTURE.md) defines modules/backends.
[Journal](docs/JOURNAL.md) holds full timings, job histories, source hashes,
failed attempts and audit evidence. This file keeps goals, current state,
open work, decisions and closed directions only; history goes to the journal.

## Goals

**Minimize time and peak memory to reach an audited solution at the requested
precision.** Preserve every numerical contract in AGENTS.md; an accuracy
failure never counts as a performance win.

- **Float64 (user, 2026-10-08):** beat MOSEK on every problem of the 58-case
  audited scoreboard (`t7/cases.txt`), at 1 and 16 threads.
- **MPFR (user, 2026-10-08):** beat SDPB at 768+ bits on the Ising problems
  (Λ19, Λ27, Λ35, ising11) with matched precision, thresholds, input and
  hardware. Λ35 runs at 1024 bits (user, 2026-10-09); the bar is SDPB at
  1024 bits, 745 it / 2762 s on one 64-core node. Target (user,
  2026-10-09): ≥2× faster than SDPB; 3× is a stretch on Λ19–Λ27, and on
  Λ35 it needs a shorter plateau.
- Memory: match SDPB peak PSS/RSS at matched thread counts; report
  speed/memory trade-offs rather than combining unmatched runs.
- PMP conversion matters only when it limits input-to-solution time or
  memory. Code size and uniform architecture are secondary; do not replace
  measured fast paths to meet a line count.

## Development and verification

[AGENTS.md](AGENTS.md) defines the workflow: one change, build the affected
crate (`--locked --offline`, one feature set), one matching E2E against a
frozen baseline with its original-coordinate audit. Pure refactors preserve
points. Order-changing changes pass the full-solve audit instead of bitwise
identity (no precision or refinement relaxation). Keep ≥2% end-to-end gains
or clear memory/correctness benefits. Long runs go to the cluster from
frozen sources. Frontends dispatch Float64 plus MPFR128/256/512/768/1024 by
default; `all-precisions` adds 128–2048 in steps of 64 (Lambda43 needs 1216).

Cluster gates (`~/projects/sdpx-ising11-scaling-20261007` on `hpc`):
- Build `t6bld.pbs`, with `-C target-feature=+fma` by default (`NOFMA=1` turns it off). Points are bitwise identical, and 1 thread is 2–18% faster.
- Float64 scoreboard `t7/t9run2.pbs` + `t7/t9_compare.py` (58 cases, 1/16 threads, every point audited by `native_oracle.jl`, s := b − Ax).
- Ising ABBA `t14abba.pbs` / `t15abba.pbs`, with audit `audit_point.jl` (`t4dgate.pbs`'s Λ27 audit call is broken; use `t14aud.pbs`).
- Small-case timing across nodes is bimodal: the conservative governor, a cold process, and MOSEK's warm minimum-of-3. Decide small Float64 timing with same-node interleaved A/B (`t13/abfam.py`).

## Current state (2026-10-09)

### Scoreboard against the goals

| Case (64 threads, 768 bits, 1e-42 unless noted) | SDPX | Reference |
|---|---|---|
| Λ27 spins 0–50 | 183 it / 258.5 s (1.41 s/it), audited | SDPB 265 it / 375 s (1.41 s/it) |
| spins 0–50 (s50) | 153 it / 153.8 s | SDPB 285 s |
| Λ19 | 121 it / 71.0 s, audited | — |
| ising11 (512 bits) | 54 it / 3.44 s | — |
| Λ35 spins 0–70, 1024 bits | Solved 683 it / 2426.5 s (3.55 s/it), audited; 768 bits Solved 691 it / 1886 s, audited | SDPB 1024 bits 745 it / 2762 s (3.71 s/it); 768 bits 746 it / 2009 s |
| Float64 58 cases (t16b = int5 + task 16) | 54 audited solves, 0 Solved-but-fail; wins 19 (1 thr), 12 (16 thr) | MOSEK |

These Ising figures include task 15. The Float64 figures are from t16b (int5 `a5334dd` + FMA, plus task 16).

- **Float64 geo-mean SDPX/MOSEK at 1 thread:**
  - int5: LP 2.53, SDP 1.65, SOCP 1.43 (16 threads: 2.18 / 2.29 / 1.77); int4 was 3.03 / 1.80 / 1.68.
  - Task 13, same-node A/B: LP 2.3, SDP 1.5, SOCP 1.5.
- **Float64 not Solved:** hinf3 (platform-sensitive), sched_100_50_orig, csdr3 (augmented/arrow end-game floor), f64_large (AlmostSolved/20, no Slater point; facial reduction).
- **Tiny LPs** lose on per-solve overhead: 2–20 ms against MOSEK's warm 1–5 ms.

### Retained work (details and evidence in the journal)

- **Release 0.9.1 (`4863de1`)** keeps the reviewed repairs: per-RHS refinement progress, accepted-iterate residual restoration, boundary validation, mixed-cone MPI exchanges. Ising11 MPFR512 is `Solved`/52 at one and four threads, with identical points and a 1e-30 original-coordinate audit.
- **Storage and exact products (2026-10-05..07):**
  - reused H·z and in-place gather decode;
  - borrowed SVD/eigen work and packed Grams;
  - shared exact-product CRT;
  - bound-leaf Schur destination reuse;
  - bounded and compact local-SOC exact products;
  - grouped residue block Schur (csdr3 −17%);
  - local cone arrow (crossing SOCP MPFR128 698.8 → 19.3 s);
  - memory/density size gates;
  - presolve F_p rank images;
  - PMP basis release;
  - ordinary residue GEMM ≥1024 bits at side ≥12.
- **Performance plan 2026-10-08 (q1/q2):**
  - Historical MPFR shift/pivot rule: Λ27 451 → 376 it. Superseded by task 17 below: no shift/replacement until a failed factor triggers escalation.
  - Assembly budget max(256 MiB, RAM/8).
  - Parallel border SYRK and tiled border factor use linked BLAS. The faer tile replacement was rejected after a convergence regression.
  - Dynamic pivot rule inside the dense Cholesky.
  - Binary64 PSD Gondzio correctors: medium 19 → 15 it, large 25 → 19 it.
- **Integration `0e42ad2` (2026-10-09):**
  - Task 9: Float64 Solved implies the original-coordinate audit (`tol_original` 1e-6, MPFR off); the returned s is the cone projection of b − Ax; thread-invariant factors.
  - Task 10: cost-based LP backend, presolve budget, condensed/augmented and chordal choices that never read the thread count.
  - Task 12: fixed-τ phase, MPFR only.
  - Task 14: data-scaled unit-fallback τ₀ (Λ27 376 → 183 it).
  - Per-factor arrow row block.
- **Task 15 (`perf-iter` `a68f72f`, merged `9454332`):**
  - Shared sampled linear products (bitwise identical).
  - Square-root-free Givens replay in the MPFR SVD (replay CPU −32%, audits pass).
  - Λ27 −4.9%, s50 −7.0%, Λ19 −5.3%, ising11 −4.3%.
- **Task 13 (`perf-small` `ad536de`, merged; int4 gate identical statuses/iterations/audits, geo-mean 0.835/0.911 vs int3):**
  - Lazy QDLDL parallel plan; one AMD ordering; Mersenne modular minor.
  - Receipt CPU clocks only with worker threads.
  - Binary64 arrow split thresholds; PSD scratch per cone order.
  - All 116 points bitwise identical; same-node geo-mean 0.86 at 1 and 16 threads.

### Profiles

- **Λ27 at 768 bits, 64 threads (task 15 base):**
  - Refined solves are 50%: 938 passes at about 120 ms each (prepare 23, reduced 14, recover 33, residual 51 ms). The outer corrections alone are 19%.
  - Cone scaling 21.6%: largest-cone SVD wall 0.30 s; replay was 54% of SVD CPU before task 15.
  - Factorization 15%.
  - Cycles: GMP `mul_basecase` 22%, RNS GEMM pipeline about 18%, rayon idle stealing about 8%.
- **Mixed Lambda27 at 1024 bits, 32 threads (frozen 0.9.1, failed point, 2349.9 s loop):**
  - KKT update 62.4% (Schur contributions 422.6 s, leaf factor/coupling 359.6 s).
  - Later KKT solves 14.1%; cone scaling 15.3%; border factor 0.6%.
  - 813 linear solves, 553 of them refinement corrections.
- **Float64 large SU(2) at `515708f`, 4 threads:**
  - Schur assembly 33.6 s (serial, CPU/wall 0.999) and refactor 37.5 s, out of an 84.2 s solve.
  - Refactor scaling stalls at 16 threads.
  - The assembly budget (2026-10-08 item 8) and the faer tiles came after this profile: large at 16 threads went 66 → 47.5 s.
- **Float64 small (task 13):**
  - Setup was 1.5–5.7 ms of QDLDL plan allocation on tiny LPs (fixed).
  - The out-of-line `fma` call was 5–9% of cycles without `+fma` (fixed by the build flag).
  - At 16 threads, concurrent small OpenBLAS calls serialize: summed `cone_wprod` time on mcp500 goes 33 → 2439 ms.

## Known failures (keep visible)

- **Λ35 conditioning:** the dual has no Slater point and both solvers retain a long gap plateau. Task 17 resolved the former 768-bit floor: Solved and audited at 768/1024 bits (scoreboard above). Shortening the plateau remains performance work.
- **Mixed Lambda27 MPFR1024 (frozen 0.9.1):** `Solved`/42 but fails the 1e-30 original-coordinate audit: dual 1.475e-23, primal 3.73e-28, PSD link 2.73e-26, componentwise dual 0.84355.
  - Primal variables reach 5.8e118.
  - The explicit-gate pilot is prepared but not run (decision below).
  - Not the same input as the 768-bit Λ27 above, which passes its audit.
- **Large SU(2) Float64 (n 7054):** every PSD block has a kernel shared by all A_j and b (30/30/30/30/60/12/14 of 95/92/94/92/186/74/71).
  - SDPX is AlmostSolved, and MOSEK's "optimal" points fail the 1e-6 audit (r_d about 1e-4).
  - Medium SU(2) is now Solved and audited after task 16; the large case still fails.
  - Remedy: facial reduction (`perf-facial`, opt-in, unverified).
- **Float64 unresolved cases:** hinf3, sched_100_50_orig, csdr3 and f64_large. Task 16 resolved qap6, gpp100, gpp124-1, gpp250-1 and gpp500-1.
- **Gravity Float64 default:** `Solved`/17 with r_d 4.01e-6 > 2e-6 before task 9. Not re-run since `tol_original`.
- **MPFR `condensed_graded`:** kept as designed.
  - Applying H through R fixes the synthetic test but costs +5.5% on Ising.
  - The synthetic failure remains.
- **Cluster test:** `sampled_integration::dim2_signed_parities_ruiz_f64` is AlmostSolved on Linux since ae2a827 and passes on macOS.

## Active work

| Task | Branch | Scope | Gate |
|---|---|---|---|
| 17 | `perf-l35b` | Merged `368ffe7` (no MPFR shifts, GMP basecase, opt-in host GMP). Remaining, not merged: (F) a start rule for SDPB 3.1 resampled inputs (Λ27-rs is a τ chase at the default start: 345 it vs 145 at τ₀ 1e-30); (G) `sdpx-pmp2sdp` resamples verified PyCFTBoot/SDPB.m blocks by default (`--keep-samples` opts out), with threads defaulting to the cores within a memory cap; then Λ35-rs at 1024 | Default solve on resampled inputs no worse than today on old inputs (Λ19 ≤121, Λ27 ≤183, Λ35 ≤683 it, audited); ising11, gravity256 and the old inputs no worse |
| 20 | `perf-t20` | Done, merged `23e0a99`: owner-MPI agreements (−1.4…8%, bitwise identical); `pmp2sdp --resample` (SDPB 3.1 points: Λ19 −38%, Λ27 −41% at τ₀ 1e-30, 0 refinement corrections). Matched SDPB and full resampled Λ35 comparisons are complete (journal). Remaining: start rule on 3.1 inputs (Λ27 345 it at default start), then Λ35-rs with task 17's factor changes | — |
| 21 | `perf-mc` | Multi-core scaling (user, 2026-10-09): 8–128-thread curves against SDPB on Λ27, Λ35 and mixed Λ27/1024 (`sdpx-mpi-20261004/inputs/mixed-L27-sdp-1024`, n 18703, the SDPB-benchmark-scale case; SDPB 2 paper: Λ43 mixed 2861 s → 59 s per iteration from 4 to 448 cores), with perf stat/record, lock, allocation and NUMA data. Then evidence-chosen fixes: persistent block ownership (SDPB-style affinity), per-socket layout for 128 cores, phase overlap, distributed border/Schur for mixed problems | Ising ABBA at 64 and 128 threads with audits; Λ35 pace; mixed Λ27 2nd-iteration time vs SDPB; Float64 scoreboard unchanged |
| 22 | `perf-f64mc` | Binary64 and non-PSD cone multi-thread scaling on large cases (user, 2026-10-09: small cases are too small for threads): gravity-large (n 20202, m 20299), SU(2) medium/large, gpp500, arch0, large LPs, against MOSEK at 1–32 threads; small cases only need a measured-work guard so 16 threads never loses to 1. Also owns lock-free receipts and the process-wide pool cache (the per-call pool sites are test-only). Phase 1: serial phases run about 2× slower at t>1 under the conservative governor because the unpinned main thread hops cores | 58-case scoreboard at 1 and 16 threads unchanged; same-node A/B; large cases audited; ising11 bitwise |
| 16 | `perf-endgame` | Done, merged `81b1549`: binary64 condensed Δs from the primal row, GMRES-IR on residual growth; 6 more Float64 cases Solved | — |

Task 17 items D/E passed their gates and are merged. The normalized 2×2 PMP is Solved/24. Resampling remains opt-in, and converter threads default to one; the proposed new defaults require the remaining start-rule gate.

Review repairs (2026-10-09): preserve sampled factors through singleton presolve, defer unused audit copies, bound split residue caches, reject overflow before result publication, use the existing pool for faer, and isolate complete receipts. Local audited checks preserve baseline points for medium/faer, ising11 and csdr3 at one/four threads. Linux affinity, owner assembly and MPI checks are recorded in the journal.

## Next work, in order

1. **Resampled Ising inputs (task 17).**
   - Fix the default-start τ chase without regressing old inputs or gravity256.
   - Then compare Λ35-rs at 1024 bits with the retained no-shift/GMP changes.
   - Gate: audited time at matched precision/settings against SDPB; the task 17 iteration limits above remain.
2. **Refinement corrections.**
   - Reprofile after no-shift factors and resampling: older task 15 inputs spent 50% in refined solves, but resampled runs can need no corrections.
   - Capping corrections or weakening acceptance remains prohibited. Investigate a representation floor only if the updated profile still identifies futile corrections.
3. **Float64 per-solve overhead (task 13 leads):**
   - Brandy's rational presolve pass costs 35–85 ms, because a budget counted in updates misses the GMP gcd cost.
   - Concurrent small OpenBLAS calls in cone lanes.
   - Size the shared pool by KKT work while keeping cheap cone tasks narrow. The cone-only cutoff limits gravity-large to four workers despite its roughly 206M-product bound Gram; check one/eight-thread audited solves before timing.
   - Pool creation (1–6 ms at 16 threads).
   - Receipt sampling on pooled solves.
4. **Float64 SU(2):**
   - Publish PSD Schur buffers by disjoint destination columns, preserving cone addition order. Reprofile first: the older large case spent 11.96 s in indexed publication; the current path still scatters serially.
   - Facial reduction (`perf-facial`) with tolerance-based kernel detection, gated by the audit (approved).
   - Eliminate empty (74) and equality-only (609) columns.
   - Two leaves plus a border for large's Schur columns (blocks 1–5 / 6–7 / both: 3729 / 1817 / 825; about 5× fewer factor flops).
5. **MPFR cone scaling:**
   - The largest-cone SVD sets the wall; replay is still 44% of SVD CPU.
   - Rayon idle stealing is about 8% of cycles.
   - Per-component pipelines for leaf sweeps and the border factor (order-changing; audit gate).
6. **Multi-node:**
   - (Border distribution closed: the border is 170 rows on Λ35, 2.1% at 4 nodes.)
   - Task 20 already batched refinement agreements; measure remaining collectives before changing them.
   - The single-process multi-owner path (74.8 vs 44.2 s).
   - Rank-local residue batch eligibility after partitioning (unmeasured).
   - `perf-border` `a80cfa1` is parked with its gate not run.
7. **Other open items:**
   - Exact arrow batching release ABBA, after the mixed-Λ27 accuracy decision.
   - Free-multiplier g0 SOCP (3.15 s / 15 it vs MOSEK 1.3 s / 13 it).
   - Owner-MPI correctors (port when an MPI SOC/LP workload exists).
   - Local cone arrow follow-ups: per-way residue scratch, and exponential and power cones as leaves.
   - Sparse QDLDL `solve_many` loops single solves.
   - csdr3 residue memory (about 25 MiB above 0.9.1 at four threads).
   - PMP conversion only when it limits the workflow.

Parked with WIP committed (journal 2026-10-09):
- `perf-fulldir` `6af5724`: regression.
- `perf-f64large` `50325c4`, `perf-threads` `946d4be`, `perf-facial` `a3a7271`: unverified.
- `perf-sharedrhs` is superseded by task 15.

## SDPB design reference (papers arXiv:1502.02033, 1909.09745; releases 2.7–3.1)

What SDPB does, and what it implies for SDPX:

| SDPB | SDPX today | Use |
|---|---|---|
| Infeasible primal-dual IPM from (x, X, y, Y) = (0, Ω_P I, 0, Ω_D I); no τ/κ. Ω = 1e40–1e60 for Λ19–Λ43 in the paper (default 1e20) | HSD (Clarabel) with τ/κ; τ₀ from the data (task 14); fixed-τ phase (task 12) | Λ35's dual has no Slater point, and HSD's τ collapses there. The fixed-τ phase is SDPX's route to the infeasible-IPM behaviour; consider an infeasible-start mode for PMP inputs if item 1 falls short |
| Separate primal and dual step lengths α_P, α_D, with γ = 0.7 | One HSD step for all variables, 0.99 of the boundary | A separate α_P/α_D is possible only with τ fixed; test inside the fixed-τ phase |
| XZ (HRVW/KSH/M) direction, symmetrized dY; Cholesky of X, Y | NT scaling with an MPFR SVD per cone | HKM measured: +12% iterations on ising11; closed |
| Mehrotra predictor (β = 0 if feasible, else 0.3) and corrector (β = r² or r, clamped by 0.1/0.3, as SDPA); two solves per iteration, no refinement | Predictor/corrector with 3 RHS (2 in the fixed-τ phase), two refinement levels plus about one outer correction each | SDPB relies on precision instead of refinement; SDPX refinement levels stay (contract), but futile corrections are item 2 |
| Free variables kept: T = [S −B; Bᵀ 0], blockwise Cholesky S = LLᵀ, Q = Bᵀ L⁻ᵀ L⁻¹ B, Cholesky of Q | Same arrow (leaves = S blocks, border = Q) | Already matched |
| SDPB 1 had "Cholesky stabilization": pivots below θ·geomean get +Λ, corrected exactly through a low-rank border U (Q′ by LU). **SDPB 2 removed it**: no regularization at all, raise precision instead | MPFR starts without shifts or pivot replacement; a failed factor escalates (task 17) | Retained and audited; further changes require new evidence |
| Termination: absolute max-norm primalError = max(\|p_i\|, \|P_ij\|), dualError = max\|d_i\|, and dualityGap = \|P−D\|/max(1, \|P+D\|) | Relative normalized residuals, plus the original-coordinate audit | "Matched thresholds" means the same numbers under different norms; the audit decides acceptance |
| Block timings from iteration 2, written to a file; worst-fit-decreasing assignment of blocks to cores | Owner cost histories (opt-in) | Same idea; adopt for multi-node only with a real scaling gain |
| SDPB 2: Elemental-distributed Q, a hand-written ring reduce-scatter (memory), Cholesky of Q distributed | Ordinary MPI replicates the border; owner path partitions leaves | Border distribution closed for Λ35 (2.1% at four nodes); reconsider only on a larger measured border |
| SDPB 3.0: Q by CRT residues into double BLAS (FLINT), in node-shared MPI windows split by `--maxSharedMemory`; about 2.5× faster than 2.7 | Exact residue GEMMs on faer per process | SDPX already has the residue products; node-shared panels matter only for multi-rank memory |
| SDPB 3.1: sample points that minimise interpolation error (lower condition numbers) | Opt-in `pmp2sdp --resample` | Matched comparisons completed; default-start work remains (task 17) |

## Measurement prerequisites

- Use validated dynamic OpenBLAS on the cluster, with identical thread and BLAS budgets.
- Changes that move the stopping point (directions, correctors, regularization) compare time to a matched final accuracy, e.g. csdr3 at 1e-12 (optimum −31.6721556) beside the pinned 1e-8 case.
- Checkpoints must have the same structure. Hot starts map through original coordinates.
- Bind input, settings, precision and hash before reusing an audit. Record native/API/process scope, status, iterations and peak memory.
- On the shared Mac, compare only inside one A/B/B/A session and report CPU seconds. Decide sub-5% changes on the cluster.
- Avoid node70: it kills jobs at start. OpenMPI uses TCP (`--mca btl self,vader,tcp`); `openib` hangs.

## Decisions

**Approved by the user:**
- 2026-10-07: shared-SOC arrow leaf primal columns skip the static shift.
- 2026-10-07: Gondzio correctors.
- 2026-10-09: facial reduction with tolerance-based kernel detection, gated by the original-coordinate audit.
- 2026-10-09: refinement of the full HSD direction including Δτ, with tolerances unchanged.
- 2026-10-09: a data-chosen τ₀.
- 2026-10-09: further contract decisions within the fixed-precision rules are delegated to the lead. Never lower precision, never loosen refinement acceptance, never promote AlmostSolved.

**Open:**
- **Mixed Lambda27/1024 explicit-gate pilot.** The frozen baseline binary, 32 cores, 64 GiB, 2 h, ≤100 iterations. Prepared and awaiting approval after two exhausted retries.
- **Independence from Clarabel.rs.** About 36% of non-test lines still match same-named Clarabel.rs files (Apache-2.0 notices kept).
  - Done (bitwise-neutral): the staged driver, banner/report, NOTICE.
  - Next neutral candidates: the `problemdata.rs` preprocessing pipeline, a uniform presolve/postsolve record, and the configuration printer.
  - Changing numerical policies conflicts with the "Clarabel-style convergence/refinement/regularization" rule and needs a decision.
- **Deferred until a concrete need:**
  - owner cost histories as a public interface;
  - ordinary/owner MPI convergence (only after real MPI E2Es; `direct_kkt_solver` stays in the settings schema);
  - Python/Julia bindings, a certificate product, backend unification and line-count rewrites;
  - sequential Schur writes;
  - dense-leaf packing (0.91% RSS);
  - distributed restart, broad sweeps and the old g0/application campaigns.

### Survey: other solvers (2026-10-07)

- **SDPA-GMP/QD/DD:** double-double and quad-double arithmetic (about 106/212 bits) is several times faster than MPFR at 128/256 bits. A `DoubleDouble` scalar would serve medium-precision solves.
- **MOSEK/HiGHS:**
  - Presolve (singleton columns, dualization) decides small and separable SOCPs.
  - Gondzio correctors (adopted).
- **Hypatia.jl:** a neighbourhood-based step, and interpolant-basis (WSOS) cones that avoid lifting PMPs to SDP.
- **COSMO.jl / SCS / CVXOPT:**
  - Clique merging (SDPX has it).
  - Indirect CG with warm starts for very large KKTs.

## Closed directions

Do not reopen without new evidence; numerical reasons, timings and sources
are in [the journal](docs/JOURNAL.md).

- Lower/mixed precision, relaxed refinement, NaN clamping and MᵀM eigenanalysis
  violate numerical contracts. Fixed-point SVD replay erased tiny MPFR values.
- HKM (SDPB XZ) or mixed NT/HKM PSD direction, and NT scaling through
  eig(LᵀSL): HKM's cone update is 6.5× cheaper, but ising11/512 takes 52 → 58
  iterations and 37.1 → 41.9 s; net ≤ 0 for the condensed sampled path.
  Task 19 re-measures it at Λ27/Λ35 scale, where scaling is 21.6% of an
  iteration (new evidence).
- Cone scaling by scheduling (task 19: LPT, inner ways, pool_ways ±0.5%; scaling is CPU-bound) and by a blocked Givens replay through residue GEMM (task 19b: +6–17% per iteration at L = 40–63). Fewer rotations (divide-and-conquer bidiagonal SVD) is the remaining SVD lever.
- Hypatia-style WSOS dual-barrier cone for the Ising PMPs (task 18):
  ising11 86 vs 54 it; Λ19 gap 4.6e-9 at 107 it vs SDPX's 73; the line
  search costs more than the factor.
- MPFR PSD Gondzio correctors (+33% per iteration); low-rank DSDP formulas
  (already covered); Strassen or Ozaki residue GEMM (≤3%); a backward-error
  refinement stop (changes the refinement rule).
- τ-chase restart (`auto_start_scale`, default off): the signature does not
  separate Λ27 from Λ19 (Λ19 222 vs 177 it). Adaptive tighter reduced target
  (item 4, 2026-10-08): the corrections sit at the outer rounding floor.
- Spare workers to the costliest blocks (task 15, `9ca6881`): neutral; the
  extra ways oversubscribe the residual join. Skipping a correction when the
  residual peak is in recovered rows: disproved (55 of 334 such corrections
  were useful).
- Fixed τ from the start on Λ35 (InsufficientProgress at 728); facial
  reduction of Λ35 at threshold 1e-60 cuts off the optimum.
- Full-direction refinement including Δτ (`perf-fulldir`): regression.
- SVD/eigen warm starts, terminal/scalar substitutions, zero-pair replay,
  cached small congruences and forced small dense factorization: no solve gain.
- Float64 step/panel/column-alias/recovery/GEMM variants and Group-FMA loop
  merge: rejected measurements; retain existing rounding and unrolled loops.
- Uniform/split-block MPI, excessive tasks, refinement fusion and allocator/
  intra-cone microtuning: no workload benefit. Real MPI gates precede removal.
- Fewer KKT/refinement solves cannot remove the batched affine/constant plus
  corrector minimum or required exact residual; two-pass residual dots regress.
- Replayable exact-row scans regress 256/512 primitives; pointer staging stays,
  with no new scalar API. GMP shifted add/wide exact dots and squaring yield
  no qualifying benefit; square trial reverted, exact-norm slices stay.
- Capacity/per-thread transform retention worsens parallel memory; triangular
  faer and 2D Gram fail across sizes/threads. Column tiling stays.
- RNS rescan bypass adds mutable/cloned-owner API without copy savings.
  Pair20 is exact but unpack 5.48× slower. Rounded-Z residue re-encoding costs
  memory; cached-Y exact Gram stays.
- The shared-operand SOC layout (+10.96% native); entry-compact residue chunks
  (`perf-compact`, +38% serial `rns.block_gemm`); whole-leaf support grouping
  (0.0573% fewer products).
- Additional serial residue-encoding scratch pooling (−0.53%, RSS +2.15%);
  scalar power-of-two shortcut removal (+0.89%); contiguous GEMM scalar
  fallback (not repeatable); in-place SVD column reflectors (−0.13%); packed
  upper congruence residues (RSS +6.27%); skinny left SVD projection batches
  (+1.96%/+5.82%); parallel shared-CRT normalization (+0.58%); diagonal CRT
  digit-column splitting (−0.82%); removing overwritten sampled/PSD
  initialization (+0.82%); sampled RHS pool propagation (RSS +10.37%);
  incremental RNS prime initialization (1.23%); sampled RHS matrix borrowing;
  adaptive SVD tiles (+3.47% on Ising19/768).
- Leading-diagonal slice and bound-coupling trial; OR objective-cost guard and
  G2 coupling (AND guard stays); prime streaming; zero sampled-PSD RHS and
  TRMM; external zero-multiply/zero-addend FMA and diagonal-Gram GEMV;
  power-of-two dot products and capped wide FMMA; PMP zero-term trimming and
  Serde `collect_str`; ordinary small-square residue GEMM at 512 bits (+2.47%).
- One exact residue Gram for the whole arrow border (Λ27 1024-bit 49.6 vs
  45.2 s on one node, 30.1 vs 21.5 s on two); removed.
- Dropping the condensed inner refinement on g0: refinement levels stay
  (2026-09-24 decision).
- Streamed Float64 dense Schur spans (RSS −15…28%, time +2.5…5.7%) and a
  precomputed alias index (−0.7%): identical points, not kept.
- Lower Gram triangle/common 16-column tiles on larger gravity: serial and
  eight-thread runs regress 19.9%/8.1%.
