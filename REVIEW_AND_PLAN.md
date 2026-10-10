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
| Λ35 spins 0–70, 1024 bits | Solved 435 it / 2098.8 s (4.82 s/it, other node), audited (perf-mc-impl mci19; was 683 it / 2426.5 s); 768 bits Solved 691 it / 1886 s, audited | SDPB 1024 bits 745 it / 2762 s (3.71 s/it); 768 bits 746 it / 2009 s |
| Float64 58 cases (t16b = int5 + task 16) | 54 audited solves, 0 Solved-but-fail; wins 19 (1 thr), 12 (16 thr) | MOSEK |

These Ising figures include task 15. The Float64 figures are from t16b (int5 `a5334dd` + FMA, plus task 16).

- **Float64 geo-mean SDPX/MOSEK at 1 thread:**
  - int5: LP 2.53, SDP 1.65, SOCP 1.43 (16 threads: 2.18 / 2.29 / 1.77); int4 was 3.03 / 1.80 / 1.68.
  - Task 13, same-node A/B: LP 2.3, SDP 1.5, SOCP 1.5.
- **Float64 not Solved:** hinf3 (platform-sensitive), sched_100_50_orig, csdr3 (augmented/arrow end-game floor), f64_large (AlmostSolved/20, no Slater point; facial reduction).
- **Tiny LPs** lose on per-solve overhead: 2–20 ms against MOSEK's warm 1–5 ms.

### Retained mechanisms and bottlenecks

- Exact residue products, packed Grams, bounded arrow kernels and shared
  operator products are retained. Their implementation is in the architecture;
  completed comparisons and source hashes are in the journal.
- Float64 uses original-coordinate acceptance, condensed slack recovery and
  continuation refinement after residual growth. Old small-case setup overhead
  repairs are retained; remaining presolve/pool/receipt costs need current profiles.
- MPFR starts without shifts or pivot replacement, escalating after failed
  factors. Resampled Ising can remove most refinement corrections; old profiles
  that attributed half the solve to refinement are not sufficient evidence now.
- Cone scaling is already close to its CPU balance limit. The largest SVD and
  replay remain important; blocked replay and extra-worker trials are closed.
- Mixed Lambda27's older profile was dominated by KKT assembly and leaf work,
  but its point failed the original audit. It is not accepted timing evidence.

## Known failures (keep visible)

- **Λ35 conditioning:** the dual has no Slater point and both solvers retain a long gap plateau. Task 17 resolved the former 768-bit floor: Solved and audited at 768/1024 bits (scoreboard above). Shortening the plateau remains performance work.
- **Mixed Lambda27 MPFR1024 (frozen 0.9.1):** `Solved`/42 but fails the 1e-30 original-coordinate audit: dual 1.475e-23, primal 3.73e-28, PSD link 2.73e-26, componentwise dual 0.84355.
  - Primal variables reach 5.8e118.
  - The explicit-gate pilot is prepared but not run (decision below).
  - Not the same input as the 768-bit Λ27 above, which passes its audit.
- **Large SU(2) Float64 (n 7054):** every PSD block has a kernel shared by all A_j and b (30/30/30/30/60/12/14 of 95/92/94/92/186/74/71).
  - SDPX is AlmostSolved, and MOSEK's "optimal" points fail the 1e-6 audit (r_d about 1e-4).
  - Medium SU(2) is now Solved and audited after task 16; the large case still fails.
  - Standalone common-kernel reduction shrinks the model but default
    augmented/faer remains AlmostSolved/21 on PBS 224492. A positive Schur
    separator is reverted and parked until an affected condensed-form solve
    is accepted.
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
| 21 | `perf-mc` | Multi-core scaling (user, 2026-10-09): 8–128-thread curves against SDPB on Λ27, Λ35 and mixed Λ27/1024 (`sdpx-mpi-20261004/inputs/mixed-L27-sdp-1024`, n 18703, the SDPB-benchmark-scale case; SDPB 2 paper: Λ43 mixed 2861 s → 59 s per iteration from 4 to 448 cores), with perf stat/record, lock, allocation and NUMA data. Then evidence-chosen fixes: persistent block ownership (SDPB-style affinity), per-socket layout for 128 cores, phase overlap, distributed border/Schur for mixed problems | Ising ABBA at 64 and 128 threads with audits; Λ35 pace; mixed Λ27 2nd-iteration time vs SDPB; Float64 scoreboard unchanged |
| 22 | `perf-f64mc` | Binary64 and non-PSD cone multi-thread scaling on large cases (user, 2026-10-09: small cases are too small for threads): gravity-large (n 20202, m 20299), SU(2) medium/large, gpp500, arch0, large LPs, against MOSEK at 1–32 threads; small cases only need a measured-work guard so 16 threads never loses to 1. Also owns lock-free receipts and the process-wide pool cache (the per-call pool sites are test-only). Phase 1: serial phases run about 2× slower at t>1 under the conservative governor because the unpinned main thread hops cores | 58-case scoreboard at 1 and 16 threads unchanged; same-node A/B; large cases audited; ising11 bitwise |

Task 21 first fixes (journal 2026-10-10, branch `perf-mc-impl`): 128-thread Λ27 −13%, Λ35-rs −14%, mixed Λ27/1024 −12% per iteration (64 threads −2…−9%), audited; the τ-chase restart is now on by default for MPFR unit starts (Λ27-rs 345 → 210 it). Mixed at 128 threads trades +13 GiB peak RSS for its gain.

Automatic τ rule (journal 2026-10-10, perf-mc-impl mci19, convergence contract, audited): large KKT start scales start at eps^(1/8); a start that still chases restarts once at eps^(1/3). Λ35-rs/1024 680 → 423 it, Λ35/1024 688 → 435, Λ27-rs 210 → 138; Λ27, Λ19-rs, ising11 unchanged. gravity256 (MPFR256, 4 threads) Solved 19 it, no restart; Float64 is untouched (wide types only). Not merged.

Task 17 items D/E passed their gates and are merged. The normalized 2×2 PMP is Solved/24. Resampling remains opt-in, and converter threads default to one; the proposed new defaults require the remaining start-rule gate.

Review repairs (2026-10-09): preserve sampled factors through singleton presolve, defer unused audit copies, bound split residue caches, reject overflow before result publication, use the existing pool for faer, and isolate complete receipts. Local audited checks preserve baseline points for medium/faer, ising11 and csdr3 at one/four threads. Linux affinity, owner assembly and MPI checks are recorded in the journal.

### Current performance changes (2026-10-09)

Release ABBA on PBS 224490 verifies ordered PSD publication: medium SU(2)
at four workers improves 3.203 → 2.948 s native (8.0%), preserving complete
points and the original 1e-6 audit. The standalone preprocessor reduces
medium rows 13,739 → 7,275 and nonzeros 138,192 → 73,869. With publication,
its reused solve takes 2.207 s and 353 MiB peak RSS: 31.1% less native time
and 14.9% less RSS than the old original solve. Preprocessing/imports and
lifting are separate costs. Sampled/MPFR inputs are rejected by this tool.

Hard rank/shared-owner assembly allowances and sampled storage pruning pass
actual MPI, the targeted allowance check and local Ising11 MPFR512 at
one/four workers with exact points and unchanged 1e-30 audits.

PBS 224493 binds to the job's eight allocated physical cores and isolates
KKT pool widening: gravity native 1.052 → 0.974 s (7.4%), API
1.055 → 0.978 s, peak RSS 300 → 318 MiB. Keep the wider KKT budget while
retaining the cone planning width. Balanced lower Gram tiles add no gain
on this repeat and are reverted. All points and original audits agree.
The earlier 1.34% pool result on node54 shows that the benefit depends on
the host; it is not a universal scaling claim.

Single-heavy sampled splitting is reverted after uniform and denser MPI
ABBA gates show no end-to-end benefit. Large reduced SU(2), checked on
PBS 224492, remains AlmostSolved/21 in both arms. Candidate lifted residuals
meet the numeric audit limits, but the status remains unaccepted. Its auto
form is augmented/faer, so the positive separator is reverted and parked
until a relevant condensed-form solve passes the original audit.

PBS 224487 A/B timings remain invalid because both executable hashes identify
the baseline. Archive timestamps reused cached Cargo artifacts. Force the
changed crate to rebuild after cache reuse, and verify binary hashes and
execution plans. The journal preserves all failed evidence and final decisions.

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
   - KKT-based pool widening is retained on gravity; profile other substantial non-PSD workloads before extending the policy. Balanced Gram tiles are closed after the allocated-core repeat shows no gain.
   - Pool creation (1–6 ms at 16 threads).
   - Receipt sampling on pooled solves.
4. **Float64 SU(2):**
   - Ordered column publication and external reduction pass medium's original audit; keep those measured improvements.
   - Resolve large's failure after reduction without changing acceptance rules. Its auto form is augmented, so a positive Schur separator needs a separate relevant gate before integration.
   - Large reduction removes 74 empty and 609 equality-only columns; recovery remains mandatory.
5. **MPFR cone scaling:**
   - The largest-cone SVD sets the wall; replay is still 44% of SVD CPU.
   - Rayon idle stealing is about 8% of cycles.
   - Per-component pipelines for leaf sweeps and the border factor (order-changing; audit gate).
6. **Multi-node:**
   - (Border distribution closed: the border is 170 rows on Λ35, 2.1% at 4 nodes.)
   - Task 20 already batched refinement agreements; measure remaining collectives before changing them.
   - The single-process multi-owner path (74.8 vs 44.2 s).
   - Single-heavy-owner sampled splitting is closed: correct on both MPI gates, but neither improves end-to-end time.
   - Profile prime-group underfill and NUMA residue placement on large calls;
     split complete inner products over output tiles only if they dominate.
   - Use `rns.cache.hit`/`rns.cache.rebuild` and profile dimensions to measure
     serial cached-operand rebuilding before the shared-CRT phase. Original
     Λ19/768 on PBS 224502 passes its audit; all 10,251 rebuilds have one
     granted way (largest operand 4,371 entries). No inner encoding change is
     justified there. Parallel encoding needs a larger call with material
     critical-path cost and spare ways; an unused same-operand cache branch
     is not a solver optimization.
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

## Parallel design reference

The architecture links primary SDPB, SCS and MOSEK sources. SDPX already
uses measured block costs, one shared pool, inner exact products and owner
partitioning. Remaining questions are substantial output-task balance,
prime-group underfill and producer/consumer NUMA placement. Measure those
on the affected shapes before adding a scheduler or distributed backend.
SDPB's different start, direction and convergence policies are described
in the journal; they do not override SDPX's numerical contracts.

## Measurement prerequisites

- Use validated dynamic OpenBLAS on the cluster, with identical thread and BLAS budgets.
- Changes that move the stopping point (directions, correctors, regularization) compare time to a matched final accuracy, e.g. csdr3 at 1e-12 (optimum −31.6721556) beside the pinned 1e-8 case.
- Checkpoints must have the same structure. Hot starts map through original coordinates.
- Bind input, settings, precision and hash before reusing an audit. Record native/API/process scope, status, iterations and peak memory.
- On the shared Mac, compare only inside one A/B/B/A session and report CPU seconds. Decide sub-5% changes on the cluster.
- Avoid node70: it kills jobs at start. OpenMPI uses TCP (`--mca btl self,vader,tcp`); `openib` hangs.

## Decisions

[AGENTS.md](AGENTS.md) governs numerical changes and approvals. Preserve
requested precision and every acceptance/refinement rule.

**Open:**
- **Mixed Lambda27/1024 explicit-gate pilot.** The frozen baseline binary, 32 cores, 64 GiB, 2 h, ≤100 iterations. Prepared and awaiting approval after two exhausted retries.
- **Deferred until a concrete need:**
  - owner cost histories as a public interface;
  - ordinary/owner MPI convergence (only after real MPI E2Es; `direct_kkt_solver` stays in the settings schema);
  - Python/Julia bindings, a certificate product, backend unification and line-count/independence rewrites; preserve attribution;
  - sequential Schur writes;
  - dense-leaf packing (0.91% RSS);
  - distributed restart, broad sweeps and the old g0/application campaigns.

## Closed directions

Do not reopen without new evidence; numerical reasons, timings and sources
are in [the journal](docs/JOURNAL.md).

- Lower/mixed precision, relaxed refinement, NaN clamping and MᵀM eigenanalysis
  violate numerical contracts. Fixed-point SVD replay erased tiny MPFR values.
- HKM (SDPB XZ) or mixed NT/HKM PSD direction, and NT scaling through
  eig(LᵀSL): HKM's cone update is 6.5× cheaper, but ising11/512 takes 52 → 58
  iterations and 37.1 → 41.9 s; net ≤ 0 for the condensed sampled path.
  Task 19 also declined HKM at larger scale; its proposed gains required
  contract changes.
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
- Balanced rectangular lower Gram tasks: correct and point-identical, but
  the allocated-core release repeat is neutral (+0.13% native time).
- Single-heavy sampled operator splitting: operator phases are 24–25%
  faster, but uniform and denser MPI full solves have no end-to-end gain.
