# SDPX development plan

Updated 2026-10-14. [AGENTS.md](AGENTS.md) defines working rules and numerical
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
identity (fixed precision and unchanged audit gates). Refinement scope changes
require the evidence specified in AGENTS.md. Keep ≥2% end-to-end gains
or clear memory/correctness benefits. Long runs go to the cluster from
frozen sources. Frontends dispatch Float64 plus MPFR128/256/512/768/1024 by
default; `all-precisions` adds 128–2048 in steps of 64 (Lambda43 needs 1216).

Cluster gates (`~/projects/sdpx-ising11-scaling-20261007` on `hpc`):
- Build `t6bld.pbs`, with `-C target-feature=+fma` by default (`NOFMA=1` turns it off). Points are bitwise identical, and 1 thread is 2–18% faster.
- Float64 scoreboard `t7/t9run2.pbs` + `t7/t9_compare.py` (58 cases, 1/16 threads, every point audited by `native_oracle.jl`, s := b − Ax).
- Ising ABBA `t14abba.pbs` / `t15abba.pbs`, with audit `audit_point.jl` (`t4dgate.pbs`'s Λ27 audit call is broken; use `t14aud.pbs`).
- Small-case timing across nodes is bimodal: the conservative governor, a cold process, and MOSEK's warm minimum-of-3. Decide small Float64 timing with same-node interleaved A/B (`t13/abfam.py`).

## Current state (2026-10-13)

### Scoreboard against the goals

| Case (64 threads, 768 bits, 1e-42 unless noted) | SDPX | Reference |
|---|---|---|
| Λ27 spins 0–50 | 170 it / 130.5 s (0.77 s/it), audited (Λ27-s50c, L71, 2026-10-13) | SDPB 265 it / 375 s (1.41 s/it) |
| spins 0–50 (s50) | 153 it / 153.8 s | SDPB 285 s |
| Λ19 | 121 it / 71.0 s, audited | — |
| ising11 (512 bits) | 54 it / 3.44 s | — |
| Λ35 spins 0–70, 1024 bits | Solved 357 it / 676.3 s (1.87 s/it), audited (L75, 2026-10-13; was 683 it / 2426.5 s); 768 bits Solved 357 it / 535.8 s, audited (L82, 2026-10-13; was 568 it / 742.1 s) | SDPB 1024 bits 745 it / 2762 s (3.71 s/it); 768 bits 746 it / 2009 s |
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
- Review repairs (2026-10-13): coherent full MPFR SVD, joined helper lifetimes,
  completed chunk dispatch, bounded/disjoint MPI keepers, packed upper
  congruence residues and reused SVD output storage. Local Ising512 points
  remain exactly unchanged; the journal records the matching gates.
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
- **Mixed Lambda27 MPFR1024 (frozen 0.9.1):** `Solved`/42 but fails the 1e-30 original-coordinate audit: dual 1.475e-23, primal 3.73e-28, PSD link 2.73e-26, componentwise dual 0.84355. The resampled input (2026-10-10) also fails (dual 2.34e-23) and takes 57 iterations.
- **Integration test failing on `main` (ee235b4) and `perf-mc-impl`:** `overshooting_start_still_solves` (NumericalError from τ₀ 1e-70), since `b718ace` removed the MPFR static shift. After an accepted 0.99 step, z leaves the PSD cone. A static shift of eps^(3/4) fixes it, while a step retake or refinement does not. Changing the default needs a decision (journal 2026-10-11). The chordal fixture tests were updated for `5b9980e`.
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
- **Goldstone LP/SOC dual-residual stall (MPFR256, tol 1e-27):** pres reaches 1e-29 while dres stays 1e-8…1e-11 with steps near 1; 512 bits stalls lower (1e-14). The dual part of the direction is inaccurate; under investigation after the step-stop change.
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

Levers L1–L9 and wave balance (journal 2026-10-11, perf-mc-impl, 64 threads, Λ35-rs):
- Kept:
  - exact-dot Y-solve rows (−3.1%; mixrs −7.9%, audited);
  - host GMP for cluster builds (−3.7%);
  - LPT-planned cone-lane ways (`2dff54b`, −2.5%, bitwise).
- Rejected:
  - planned ways for the KKT phases;
  - power-of-two balancing;
  - cached R residues;
  - full Newton step;
  - directed SVD sweep;
  - τ fall-rate chase detection.
- Next: the solve pipeline (about 45% of wall at f50); the SVD lever is done (below).

L8, chase restart keeps the side that did not grow (journal 2026-10-11, perf-mc-impl, convergence contract): Λ35-rs/1024 423 → 372 it (1511 → 1195 s); the other Ising inputs are bitwise unchanged. Audit 224854 accepted.

Step limits L10/L11 (journal 2026-10-11): the steady phase is limited by primal centrality in the largest cones (α_aff 0.65, step 0.90). MPFR curve search and split steps were rejected; split steps diverge on Λ27. The MPFR PSD Gondzio corrector cuts iterations 15–17% but costs 3–14% more time, so it was also rejected. Diagnostic: `SDPX_DBG_STEPCONE`. Next: cheaper KKT solves (the solve pipeline is 1.4 of 3.19 s per Λ35-rs iteration, 41–48 of 64 cores busy), which would also make the corrector pay.

Queues, cone SVD and tiled CRT (journal 2026-10-11, perf-mc-impl, now defaults). Each change is measured on Λ35-rs, 64 threads:
- largest-first job queues for the KKT block passes, residual products and step bounds: −6.2%, bitwise;
- queued border contributions: −2.6%, bitwise;
- dqds values with inverse-iteration vectors for the V-only cone SVD: −4.1% and −3.0%, audited;
- an output-tiled CRT for the RNS gemm and congruences: −8.4% per iteration, bitwise.

The f50 full solves with all of them are audited: Λ35-rs takes 372 iterations in 900 s (1216 s at mci41, −26%); Λ27 −8.5%, Λ27-rs −5.7%, Λ19-rs −6.6% against f46.
Kept since:
- residue products for short arrow-leaf groups (Λ35-rs −3.8% per iteration, bitwise, peak RSS 5.6 → 7.2 GB);
- cone SVD vectors from the shifted normal form within a factor two, with windowed Gram–Schmidt (Λ35-rs −5.5% per iteration; full solves −2.2 to −3.9%; audited);
- step bounds of KKT directions from one congruence, with W⁻ᵀΔs = −WΔz − W⁻ᵀc from the complementarity row (Λ35-rs −3.8% per iteration; full solves −2.2 to −6.0% at unchanged iteration counts; audited);
- the combined shift as one exact symmetric square (Λ35-rs −2.7% per iteration), and WΔz from the condensed recovery's pre-image when refinement made no correction (−2.1%; Λ27-rs −3.5%). Together the full solves are −1.9 to −6.2% at unchanged iteration counts, audited.
Rejected since: residue-major i32 congruence (gains only for n ≥ 96 under full load); reciprocal pivots in inverse iteration (+12%); blocked exact leaf LDL from order 96 (Λ27-rs +2.8%); board items inside cone SVD/Cholesky (−0.7%); queue help with board items in unsplit congruences and gemms (−1.2%, and its idle wait can livelock when a stolen join half starts a queue item); compact-WY V back-transform (+0.4% under full load). Their patches are kept in `~/.cache/sdpx-e2e/patches`.

Next:
- the solve pipeline: residual 0.246, recover 0.148, prepare 0.113 of 1.85 s per Λ35-rs iteration at mci60 (3.1 solves per iteration, three exact congruences per block and solve);
- refinement corrections in late iterations (f63): prepare + recover + residual take 44% of Λ35 (1586 corrections in 435 iterations), 41% of Λ27 and 37% of Λ35-rs. First find which rows exceed the ε^(3/4) tolerance and why, then make the raw solve more accurate or each correction cheaper. The tolerance itself stays;
- cone kernels under full load (n = 63, 64 copies, per cone): bidiagonalization 85 ms, reflectors 59, R/R⁻¹ 50, inverse iteration 43, Cholesky 24, dqds 19, L2ᵀL1 13 ms. A blocked triangular solve cuts R/R⁻¹ by 10.6 ms (−2.9% of the cone); panel bidiagonalization saves 5 ms. Neither yet reaches 2% of an iteration;
- Λ35-rs iterations: 169 iterations before the chase restart and about 120 to deflate the restarted primal scale (L1 detection is closed: no general early signal, at most about 25 iterations to gain).

Step-length stop (journal 2026-10-10, perf-mc-impl, convergence contract): only a zero step ends a solve; an unconverged end reports its best iterate; the exp/pow backtracking floor is eps^(1/4) in MPFR. Goldstone unitarity K4 case: InsufficientProgress/141 at gap 6.5e-6 → gap 4.5e-12 at 300; Float64 46/46 and csdr3 unchanged. Not merged.

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
   - The representation-floor stop is kept (L24, journal 2026-10-12): Λ27 −9.1%, Λ35 −1.4%.
   - The remaining corrections (Λ35 1496, Λ35-rs 554) are the x rows' productive contraction, set by the condensed solve's accuracy.
   - Capping corrections or weakening acceptance remains prohibited.
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
   - The V-only SVD takes dqds values and inverse-iteration vectors (QR and replay only on fallback). Its CPU splits into inverse iteration 35%, bidiagonalization 32%, reflectors 22% and dqds 11%. `scale cones` runs 63 of 64 cores busy, so its wall follows total SVD work.
   - Rayon idle stealing is about 8% of cycles.
   - Per-component pipelines for leaf sweeps and the border factor (order-changing; audit gate).
6. **Multi-node:**
   - (Border distribution closed: the border is 170 rows on Λ35, 2.1% at 4 nodes.)
   - Task 20 already batched refinement agreements; measure remaining collectives before changing them.
   - L6 rerun after the queue fix (journal 2026-10-12): owner layouts beat SDPB at 1, 2 and 4 nodes (Λ35/1024, 60 iterations: 4 nodes 16×16 108 s vs 126 s); the ordinary MPI path does not scale (201 s at 4 nodes). Owner ranks idle: 3-component ranks pay a second wave in the arrow panel solve (L26), Gram sync runs serially per block (L27), and scale-cone vs factor imbalance costs ~5 s at the shift barrier.
   - Owner follow-ups (journal 2026-10-12): kept L26 one-wave leaf chunks (−2.6%) and L28 rank-order reduce-scatter (−2.8%); rejected L25 planned contribution ways, L27 sync split hints and L29 pooled copies (all under 2%). 4 nodes 16×16 is now 97 s. Per-rank collective waits are 12–21 s spread over ~25 sites; site 100 costs ≤0.8 s, so merging the shift barrier is not worth it. Residual/recover/prepare run at 6–7 of 16 threads on owner ranks.
   - Owner follow-ups 2 (journal 2026-10-12):
     - Kept L33b, cone-queue helpers that run pending pool tasks when the rank has more workers than cones (−2.4%).
     - Kept L35, `SCHED_IDLE` core keepers on by default for multi-rank runs under a load-following governor: 95.10 → 87.50 s at 4 nodes 16×16 without receipts. One busy process loses 2% with keepers, hence the multi-rank default.
     - Rejected: L30 board items in cone kernels, L31 inverse-iteration chains, the MWAITX keeper variant (plain +6.5%) and the 12×21 / 36×7 layouts (L32).
     - Remaining on the critical rank: ~110 allreduces per iteration, 6.7 s total. The per-site minimum over ranks sums to ~2.8 s (communication floor); the rest is phase-to-phase imbalance, so removing every barrier (L4) is worth at most ~3%.
     - `scale cones` is set by the largest cone's serial MPFR SVD (~150 ms per cone per iteration).
     - Plain A legs differ 4% between nodes: compare within a node only.
   - Owner follow-ups 3 (journal 2026-10-12):
     - Kept L37b: the owner coupling panel `K⁻¹Cᵀ` is solved by exact-dot rows (4 nodes −3.0%, 1 node 4×16 −7.6%); full solves audited.
     - Rejected L37: the border Schur complement by residue GEMM (+10%).
     - Lockstep pair refinement deferred (≤8%).
     - Open, L41: in-process `--partitions` runs show multi-second stalls in scale cones (RNS gemm CRT maxima of 8–15 s). MPI ranks do not.
     - Kept L39: the owner path runs the core τ policy (chase restart, fixed-τ phase, best iterate). 4-node full solves: Λ35-rs 602 → 372 iterations (−42%), Λ27 379 → 223 (−43%), audited.
     - Rejected L40: owner KKT step bounds from recovery pre-images (−1.0% / −1.4%).
     - Kept L42: the owner path starts the rejected-KKT fallback at plain's τ₀ = 1/max d. Λ27 223 → 183 iterations (−20% on 4 nodes), audited.
     - Kept L43 (`eb78199`): a refinement floor for retained zero-cone rows, eps·(Σ|A_r||x| + |b_r|), on the plain and owner paths. Plain Λ35-rs −4.3%; 4-node Λ35-rs −3.3%, Λ27 −8.5%. Audited.
     - Rejected L44: the MPFR PSD Gondzio corrector. It saves 12–17% of iterations, but every call costs one combined solve, so plain is +2–8% and owner −0.4…+5.8%. Revisit only after the solve pipeline costs about half as much.
     - Next: the condensed solve pipeline. On plain Λ35-rs it takes about 0.98 of 2.02 s/it: residual 0.33, recover 0.21, prepare 0.17.
   - Follow-ups 4 (journal 2026-10-13, `perf-mc-impl` at `965db88`):
     - Kept:
       - L50/L53: the affine predictor column is left unrefined above binary64 (plain Λ35-rs −5.8%, Λ27 −5.3%; 4 nodes −5.4%/−6.2%; audited).
       - L52: no affine Hs·z congruence for PSD rows (−2.3%, bitwise).
       - L56/L57: RNS scratch without zero-fill and a streaming chunk encode (−2.2%, bitwise).
     - Plain Λ35-rs at 64 threads is now 372 iterations in about 650 s (1.74 s/it).
     - Rejected or refuted:
       - L51 earlier fixed-τ entry;
       - L58 chunk-major RNS chunks;
       - L59 panel leaf Y;
       - L60 residue contributions for light leaves (the `g·q² ≥ 2²¹` gate now has evidence);
       - L61 fused leaf contributions and L62 an `x86-64-v3` build (both under 2%);
       - L64 one stacked contribution product;
       - L66 an explicit-Ginv congruence (accuracy risk);
       - L67 freezing τ inside a chasing attempt.
     - Observed (L63): plain at 128 threads gains ≤ 3% over 64. Owner layouts on one 128-core node depend on the node: 8×16 −13% on node36 but +1.7% on node48, where 16×8 gives −3.2%. One rank × 64 threads takes 1633 s, 2.5× plain, so the owner path needs several ranks per node (task 46).
     - Observed (L68): deflation steps (≈ 0.89) are limited by primal centrality in the largest cones, and landing steps by a few off-centre small cones. Iteration levers that remain: a cheaper corrector solve, or a chase signal for a deep first start (L1 is closed).
   - Follow-ups 5 (journal 2026-10-13, L71):
     - Kept L71 as the default: a rejected or chase-scale KKT start keeps its shifted dual and starts only the primal at unit slacks (plain and owner paths). Plain −3.5…−35.5% on all ten Ising inputs (Λ35-rs 372 → 347 it, Λ27 183 → 170, t11-aux35 103 → 71); 4-node owner −6.1% / −9.1%; audited.
     - Rejected L72 (one owner cone queue, 0%), L73 (restart dual at the chased scale, +1.1%) and L74 (deep τ start: Λ35-rs/pf −37%, but Λ27-rs +74%, and no static predictor separates them). Refuted L77 (dual-residual polish). Dropped L78 (lazy componentwise check: its products are hidden).
     - Observed: the 1e-30 componentwise dual test holds Λ27-s50c for 19 iterations (170 vs 151); it stays.
     - Open items: see follow-ups 6.
   - Follow-ups 6 (journal 2026-10-13, L75):
     - Kept L75 as the default: a chase restart of a KKT-rejected input keeps the KKT dual when the chased ‖z‖∞ stayed within it. Λ35-s70-p1024 414 → 357 it (−32.5%), Λ35-s70/768 −15.3%; the other eight Ising inputs are identical; audited. 4-node owner (l75n5): 414 → 357 it, 659.6 → 458.5 s (−30.5%).
     - Rejected L76 (fixed τ on a settled-τ gap stall: never fires).
     - Observed: the Λ35-s70/768 plateau is primal growth within fixed τ (s/τ 2e76 → 1.7e95 at 0.08 dec/it). The chase restart's τ₀ = eps^(1/3) is 1.1e-77 at 768 bits, which starts the primal below its solution scale. L82, L80 and L81: see follow-ups 7. L79 (a chase predictor for KKT-rejected inputs) is parked.
   - Follow-ups 7 (journal 2026-10-13, L82):
     - Kept L82 as the default: the deep chase-restart scale is 2^-341 (eps^(1/3) at 1024 bits), clamped to [eps^(1/2), eps^(1/3)], so it no longer follows the precision. Λ35-s70/768 568 → 357 it (−28.1%), now on the 1024-bit path. Λ35-rs and Λ35-s70-p1024 are identical; audited.
     - Observed: a restart scale too shallow costs about 9 iterations per decade, too deep about 1.
     - Rejected L80 (τ capped at a scaled start): Λ27 −6…−15%, but Λ35-rs/pf and t11-aux35 +3–5%. Its premise (τ₀ bounds τ) is refuted. Rejected L81 (accept a stopped refinement candidate at its floor): neutral.
     - Closed L83 (journal 2026-10-13, L83). t11-red60 is a synthetic facial-reduction input. Its DualInfeasible verdict is verified in original coordinates (qᵀx = −1, ‖Ax + s‖∞ = 8.3e-78, s exactly PSD), and every start ends there. The saturated start costs +22% iterations (rejected). L84 (shorten the 58-iteration ray phase without touching the gate) is parked until a real excluded-point feasibility input is in the suite.
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

8. **Thermal matrix models (journal 2026-10-13, thermal-ore):**
   - The native operator relative entropy cone is in, on branch `thermal-ore`. two_L4 (53/128/256 bits), two_L6 (53/128) and one_L10 (53) are Solved and audited. two_L6/128 takes 7.94 s.
   - Review repair: owner RHS/dual coordinates, linear slack recovery and fresh-solve phase resets match the ordinary path. After an actual zero-base factor failure, the retry shift now has an `eps · max|diag(KKT)|` floor (initial shift and refinement tolerances unchanged). two_L4/128: ordinary Solved/45, owner Solved/44, both pass original 1e-20 and Arb audits; the former owner path stopped AlmostSolved/37. Extra T-row eigen/Cholesky whitening trials regressed ordinary convergence and are reverted.
   - 2026-10-14 (journal): the negated arrow takes the ORE rows as leaves; a failed arrow factor falls back in the z-first order. Binary64 inverses use BLAS, the dual test exits early (no CG at 53 bits), and ORE kernels split. Release T=1 against v0.9.2: one_L12/53 Solved 47 in 20.8 s (was 100 s, audited); one_L12/128 4 it 103 s (was 471 s). two_L4/two_L6 points are unchanged and audited.
   - 2026-10-14 (journal): leaf dots start at the row envelope, ORE kernels post to the scaling board, and wide dense Schur columns run on the pool. one_L12/128 4 it: T=1 −13.6%, T=4 −27%, bitwise points.
   - 2026-10-14 (journal): 128/192-bit residue products, recursive S₀ kernels above order 128, a halving CG budget and batched dual trials. At T=2: two_L6/128 −5…−11%, two_L6/256 −8…−10%, one_L12/128 (4 it) −22…−32%. A synthetic ORE n=20 input is Solved and audited (−33…−41%). Peak RSS on one_L12 rises 32–65%.
   - 2026-10-14 (journal): the arrow border factor takes exact dots above 64 bits, split in pivot panels (bitwise equal to serial). two_L6 −17…−19% at T=1 and −12…−14% at T=4; the synthetic input −27%/−13%. All points are audited. On the cluster, Ising Λ27/Λ35 at 64 threads is not yet measured.
   - 2026-10-14 (journal): arrow contributions take residue batches at 128–256 bits (`residue_product_applies`). Points are bitwise unchanged. T=1: two_L6 −4.5…−6.7%, synthetic −13%. T=4: neutral, because a lone residue product uses about 1.8 of 4 cores.
   - 2026-10-14 (journal): the ORE KKT row transform (N⁻¹A per column) runs on the solver pool. T=4: synthetic −8.1%, one_L12 −5.1%, two_L6/256 −2.7%. Points are bitwise unchanged.
   - 2026-10-14 (journal): ORE dual step tests take one B step, then CG preconditioned by the current iterate's S₀⁻¹ carried to the trial point. Synthetic −20.8% at T=4 and −15.0% at T=1 (S₀ factors 385 → 181); two_L6 and one_L12 within noise. Points are bitwise unchanged.
   - 2026-10-14 (journal): S₀'s columns are built in parallel pieces. T=4: synthetic −6.2%, one_L12 −2.6%. Points are bitwise unchanged. two_L8/128 ends MaxIterations (gap stalls near 6e-8 from iteration 75) and stays a known failure.
   - 2026-10-14 (journal): the ORE dual-test CG budget counts S₀'s column build, and each new scaling forgives one open test. Before this, the budget stuck at 0 and two_L6/256 factored S₀ 1151 times. two_L6/256 −8.3% at T=4 and −4.5% at T=1, two_L6/128 −2…−5%. Points are bitwise unchanged.
   - 2026-10-14 (journal): wide dense PSD Schur columns sum only over A's index support and compute packed entries when first read. The sparse/dense rule uses measured costs, so most sparse columns turn dense. two_L6 −6…−10% at T=1 and T=4, one_L12 (4 it) −3.9%. All solves are audited, ising11 included.
   - 2026-10-14 (journal): the MPFR auto backend takes K's or −K's arrow, whichever is eligible with less estimated dense work. On two_L6 that is −K's (five ORE leaves around 81 primal columns instead of one leaf and 186 ORE border rows): −21…−28% at T=1 and T=4, audited. Inputs with a single eligible arrow are unchanged.
   - 2026-10-14 (journal): affine ORE step searches run on the idle pool, as final searches already did, so their dual trials below αs are batched. two_L6 −5…−8% at T=4 and T=8. Points are bitwise unchanged.
   - 2026-10-14 (journal): the ORE combined shift runs its independent products in paired joins and reuses the scaling's oracle at s. two_L6 −3.6…−4.1% at T=4 and −0.8…−0.9% at T=1. Points are bitwise unchanged.
   - 2026-10-14 (journal): the bound-arrow memory gate uses the generic arrow's ceiling. Large dense MPFR LPs (the gravity restricted LP: 640 × 67k at 128 bits) no longer fall to serial QDLDL, which took 330 s per factorization on half the instance. Both sizes are Solved and audited. Open: the owner path's border assembly and pair solves for LP leaves (166–226 s vs 6–16 s plain on fix-a 256).
   - 2026-10-15 (journal): bound-arrow Gram primes are sized by the 4096-row product block, not the whole k (21 instead of 19–20 bits). Half-size gravity LP −11.2% at T=4 (release −10.2%); points are bitwise unchanged.
   - 2026-10-15 (journal): the bound Gram's ways share one residue accumulator per prime, with no per-way copies and no serial fold. Quarter-size gravity LP −2.6% at T=10 (faer), peak RSS −6%; points are bitwise unchanged.
   - 2026-10-15 (journal): local condensed kernels (owner path) run A's products on their pool's output lanes. fix-a 256 owner −6.7% at T=2; points are bitwise unchanged.
   - 2026-10-15 (journal): the owner border assembly takes one exact residue product per dense coupling. fix-a 256 owner −19.1% (release); points are bitwise unchanged. The owner path is still about 16× plain on dense-border LPs.
   - 2026-10-15 (journal): owner free columns (only border equalities touch them) join the border system and are eliminated exactly, instead of an escalated 1e-72 shift that buried S in rounding. fix-a 256 owner 121.9 → 39.0 s (−68%), 0 outer refinements instead of 762; audited; plain bitwise unchanged. Late-iteration peak RSS rises (2.4 → 3.4–4.0 GB). Next: diagonal local kernels via the bound Gram congruence (border_assemble 11.3 s + factor_response 4.8 s).
   - 2026-10-15 (journal): a structurally diagonal owner interior (LP bounds) takes R = D⁻¹Cᵀ by scaling and C·D⁻¹·Cᵀ by the exact diagonal congruence over cached Cᵀ residues. fix-a 256 owner 39.3 → 25.3 s (−35.8%), peak RSS 3.5–3.7 → 1.26 GB; partitions 2 −37%; audited. Next: serial condensed `residual`/`recover_rhs` in owner solves (7.9 s of 25.3).
   - 2026-10-15 (journal): owner recovery forms A·x only outside zero-cone rows, whose duals come from the border. fix-a 256 owner 25.3 → 22.5 s (−11.0%); points are bitwise unchanged. Next: the condensed `residual` (4.4 s).
   - 2026-10-15 (journal): wide exact forward products stream rows of ≥ 4096 entries column by column into per-row exact accumulators (contiguous values instead of scattered CSC reads). fix-a 256 owner −34.9% (T=2), −17% (partitions 2, T=10); plain −7.4% (T=2), −4.0% (T=10); points are bitwise unchanged. Next: the MPI-sharded forward products and the 640-row bound Gram (`arrow.local_schur`, 40 of 63 s on d6n640q at T=10).
   - 2026-10-15 (journal): the owner path runs plain's Gondzio centrality correctors on its orthant/SOC rows (border equalities and Ising PSD rows above binary64 are untouched). Owner iterations now match plain: fix-a 256 33 → 25 (−4.4…−9.8%), d6n640q 128 66 → 48 (−16.4%); audited; ising11 owner bitwise unchanged. Next: owner d6n640q 97.9 s vs plain 65.8 s at equal iterations.
   - 2026-10-15 (journal): over an agreed diagonal interior, owner solves skip the reduced-level refinement when the original level refines (same tolerances; GMRES-IR, unrefined columns and PSD owners keep it). fix-a 256 owner −23.6% (T=2) and −17.3% (partitions 2), bitwise; d6n640q −5.1%, audited; ising11 owner bitwise. Next: d6n640q owner 92 s vs plain 66 s, one original correction per right-hand side.
   - 2026-10-15 (journal): the owner's free-column border elimination y = S⁻¹g + W·x_f cancelled 14 digits on d6n640q. One refinement round on the bordered system with exact-dot residuals over the factored S brings raw residuals to plain's level (206 → 1 original corrections). d6n640q owner −19.3%, audited; fix-a neutral, audited; ising11 bitwise. Next: the serial pair-path back-substitution (~8.6 s of kkt update).
   - Next:
     - a full one_L12/128 solve with audit;
     - the S₀⁻¹ leaf factor from the cone's L⁻¹ (skip re-factoring the leaf);
     - the two_L8 gap stall (53: AlmostSolved; 128: stall at 3e-8).
   - Closed: Gram-row elimination of the cone (accuracy), μS₀ as the leaf inverse (inconsistent with the stored S₀⁻¹).
   - one_L8 and one_L10 at 128 bits are known failures. 2026-10-14 (journal): one_L8's optimum is not attained; the value keeps falling as ORE10 X(9,9) = Y(9,9) grows. one_L12 is likely the same.

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

[AGENTS.md](AGENTS.md) governs numerical changes and approvals. Requested
precision is fixed; other contract changes require measured, audited evidence.

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

- Lower/mixed precision and NaN clamping violate numerical contracts.
  Rejected: relaxed refinement gates and normal equations for singular values
  or ill-conditioned spectra. Shifted-normal reconstruction of vectors for
  spectra within a factor two is retained. Fixed-point SVD replay erased tiny MPFR values.
- HKM (SDPB XZ) or mixed NT/HKM PSD direction, and NT scaling through
  eig(LᵀSL): HKM's cone update is 6.5× cheaper, but ising11/512 takes 52 → 58
  iterations and 37.1 → 41.9 s; net ≤ 0 for the condensed sampled path.
  Task 19 also declined HKM at larger scale; its proposed gains required
  contract changes.
- Cone scaling by scheduling (task 19: LPT, inner ways, pool_ways ±0.5%; scaling is CPU-bound) and by a blocked Givens replay through residue GEMM (task 19b: +6–17% per iteration at L = 40–63). The retained right-only SVD now avoids QR rotations on its default path.
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
- 2026-10-13 (journal): chunk-major RNS chunks (L58), leaf Y row panels
  (L59), residue contributions below `g·q² = 2²¹` (L60, +4% on Λ27), fused
  leaf contributions (L61), `x86-64-v3` builds (L62), one stacked
  contribution product (L64, 7× slower), explicit-Ginv congruences (L66,
  squares κ), an earlier fixed-τ entry (L51) and freezing τ in a chasing
  attempt (L67). Changes that only fill idle cores move work; a lever must
  cut thread-seconds. Also the L44 corrector gated to short steps (L69), a
  warm chase restart (L70), one owner cone-scaling queue (L72), scaling the
  restart dual to the chased ‖z‖ (L73), a deep τ start without a chase
  predictor (L74), a dual-residual polish (L77) and a fixed-τ phase on a
  settled-τ gap stall (L76).
