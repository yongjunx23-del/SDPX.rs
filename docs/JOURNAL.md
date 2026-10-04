# SDPX experiment journal

Human-readable record of completed experiments and their conclusions, newest
first. Raw per-run rows (status, times, audit, RSS, binary hash) are appended
automatically by `benchmark/e2e/e2e.py` to `$SDPX_E2E_HOME/journal.jsonl`.
Add a dated entry here when an experiment is **kept or reverted**, in the
format: hypothesis → change → E2E result (case, arm, api s, audit) → decision.
Do not rewrite old entries; the plan (`REVIEW_AND_PLAN.md`) holds only current
status and next actions.

## 2026-09-27 — binary64 accuracy regression; residue-kernel buffer reuse

**Regression.** `dim2_signed_parities_ruiz_f64` (condensed, generic and
sampled) ended `AlmostSolved`. It passes at `origin/main` (e5ae2e3) and
fails from the first local snapshot (37067e5). Verbose traces matched
origin through iteration 12; then a direction broke primal feasibility
(pres 1e-11 → 0.13 at step 0.03). Debug toggles on the cluster isolated
three binary64 paths, each inherited from an MPFR speed change:
1. Condensed PSD `apply` used the squared `G = R·Rᵀ` / `Ginv` in one
   congruence. Origin applied the factor twice. Measured gap between the
   solver's `H·z` and the cones' `Wᵀ(W·z)`: up to 1e-8 (generic) and 1e-2
   (sampled) near convergence.
2. `Δs` reused that solver `H·z` (the cached scaled solution).
   Disabling the reuse fixed the sampled solve only.
3. The refinement-stall prediction (a 0.8 MPFR speedup) skipped refinement
   of the affine and combined right-hand sides. The debug trace showed only
   the constant RHS refined from iteration 8, with its residual growing
   1e-11 → 4.5e-7.

Fix (binary64 only; MPFR unchanged):
- `apply` goes through `R`/`Rinv`.
- No `H·z` reuse.
- No stall prediction: the blind skip, the in-loop skip and the
  cross-RHS floor are all off.

Variants:

| Build | Policy | test | `medium` |
|---|---|---|---|
| rns85 | fixes 1–2 only | fails | `Solved`/18, audit PASS (r_d 9.96e-7) |
| rns86 | all three | passes | `Solved`/18, audit FAIL (r_d 1.92e-6 > 1.75e-6) |
| rns87 | rns86 + the cross-RHS floor | fails | = rns85 |

Kept rns86: binary64 refinement then matches upstream, and the test pins
released behaviour. The floor prediction breaks a simple problem, and
`medium` was already a known failure (baseline `AlmostSolved`/19, r_d
2.85e-6). `condensed_graded_f64` (ignored as a known failure) passes and is
enabled. Ising Λ11 at 512 bits is bitwise identical (rns84 = rns85 = rns86
= rns87, audits pass).

**Residue-kernel buffer reuse and parallel encode** (rns85, bitwise):
- A per-thread pool (16 buffers, ≤32 MiB, best fit) backs the chunk
  matrices, stream scratch and CRT accumulators.
- Merged accumulators are released on pool workers.
- A split call encodes operand rows in parallel (≥256 rows per way).
- A new unit test checks split encode = serial encode bitwise.
- The temporary `SDPX_RNS_PROFILE` timers are removed.
- The CLI's glibc `mallopt` (mmap threshold 32 MiB, trim 256 MiB):
  Λ19 32 threads 26.96 → 25.67 s over 8 iterations (−4.8%).
- Λ19 at 768 bits, 8 iterations, same node, interleaved (`arms-rns86`).
  Points are bitwise identical across arms.

  | Threads | rns84 | rns84 + env `mallopt` | rns86 |
  |---|---|---|---|
  | 32 | 11.58 / 11.89 s | 11.38 / 10.88 s | 11.08 / 11.42 s |
  | 64 | 12.16 / 17.39 s | 15.88 / 15.28 s | 11.63 / 12.26 s |

  rns86 is the fastest and least variable at 64 threads. Kept.

## 2026-09-26 — generalizing to LP / SOCP / EXP / POW and binary64

Benchmarks are seeded synthetic families (`gen_cones.py`, `cones.pbs`,
32 threads):
- LP: box plus 10-nnz rows.
- SOCP: 6-dimensional cones.
- EXP: min Σexp(x) with equalities and a box.
- POW: budgeted power cones.
Full size for f64; 10× smaller at 256 bits (the full MPFR LP takes ~2
min/it in serial sparse LDL).

Baseline anatomy:
- 256-bit POW: step lengths are 72% of the solve.
- 256-bit EXP: step lengths 22%.
- LP and SOCP: KKT factorization (QDLDL) dominates at both precisions
  (f64 LP: 22.8 of 24.2 s).

Changes (rns80–82):
1. Screened backtracking for EXP/POW (bitwise). Each trial point and its
   membership residual are evaluated in binary64 with a rigorous error
   bound. Clear inside/outside trials skip the high-precision
   `log`/`exp`; near-boundary trials use the exact test. The α sequence and
   every decision are unchanged. Parity tests compare against the exact
   search on 400 random rays per cone side.
2. Parallel nonsymmetric step pass, with ≥64 nonsymmetric cones and a pool.
   The sequential fold ends at the minimum per-cone result along the chain
   α0·stepᵏ (monotone membership). Cones are evaluated in parallel from
   α0, then every cone is confirmed at the minimum; on disagreement the
   sequential fold runs.
3. Barrier backtracking evaluates per-cone barriers in the pool and sums
   them in cone order (bitwise).
4. binary64 KKT: build with `faer-sparse`. The existing auto rule already
   picks the supernodal multithreaded faer LDL for high-fill factors; it
   now also requires ≥1e8 flops, since small factors are faster with QDLDL.
5. The vector pool also serves binary64/32 above 65,536 entries.

Results (all Solved; 256-bit solutions bitwise equal to rns77 for rns81):
- POW 256: 7.40 → **3.80 s**.
- EXP 256: 2.25 → **1.92 s**.
- LP f64 with faer: 24.2 → **3.7–4.8 s**.
- SOCP f64 unchanged (threshold keeps QDLDL).
- EXP/POW f64 unchanged: trials are cheap, so the parallel pass does not
  pay at binary64.

Not generalized:
- The PSD-only items (certified λmin, SVD replay) have no counterpart in
  these cones: LP/SOC step lengths are closed-form.
- The generalized power cone keeps the unscreened search.
- The high-precision LP/SOCP bottleneck is the serial sparse LDL (faer does
  not support MPFR).

## 2026-09-26 — large parallel runs: rank balance and spare threads per leaf

Instrumentation (rns49, kept, numerically inert):
- `mpi.allreduce` phase.
- Per-site collective entry time in the receipt (`mpi.site_entry`).
- `SDPX_RECEIPT_ALL_RANKS=1`: every rank writes `<receipt>.rank<r>`.

Evidence, spins 0–50, 2 nodes × 4 ranks × 16 threads:
- Every rank spends 15–25% of the solve inside allreduce: 51–84 s of 337 s.
  Median latency is 6 µs over ~2,100 calls per iteration, so this is
  waiting, not latency.
- Largest waits: site 320 (border reduction of the original residual) and
  sites 239/240/242 (reduced solve).
- The KKT update does not scale: 166 s at 1×52 threads vs 167–185 s at 8×16.
  In the owner path, `owned.factor_response` (147 ms/it) runs one task per
  leaf, with only 3–4 leaves on 16 threads.

Changes (all kept):
1. Owner layout (rns50): the no-history plan now balances a cubic work
   estimate: columns³ + 8·side³ per PSD cone + the old structural weight.
   LPT is followed by a move/swap refinement of the busiest owner.
   - For the 26 blocks on 8 owners, max/mean cubic work drops 1.13 → 1.006.
   - Same node pair, 40 it: 75.1 → 72.2 s and 66.6 → 64.9 s.
   - Cost-history plans are unchanged.
2. Arrow leaves use spare threads (rns51, bitwise identical). When the pool
   has more threads than leaves:
   - The refactor's L⁻¹B border columns and YᵀD⁻¹Y columns run in parallel.
   - `solve_many` splits each leaf's panel into column chunks.
   Same node, 30 it:
   - Λ19, 64 thr: 47.1 / 52.1 → **42.1 / 43.7 s**.
   - Spins 0–50, 52 thr: 77.2 / 76.4 → 76.0 / 74.3 s.
   - 2 nodes, 8×16, 40 it: 68.5 → 66.3 s (factor_response 4.7 → 2.9 s).
3. Dense leaf LDLᵀ (rns52, bitwise identical): each step's trailing update
   runs over columns in parallel when a split is allowed and more than 32
   columns remain. The border (trunk) factor always splits, since all pool
   threads are idle then.

4. Busy-thread accounting (rns54/55/58, kept, numerically inert):
   - Every `timeit!` phase and every `receipt::start`/`finish` phase records
     process CPU (`cpu_s`) and main-thread CPU (`serial_s`).
   - The main thread sleeps inside `pool.install`, so `serial_s` is the
     phase's serial time.

   Findings, 30 it:
   - Average busy threads: 26 of 64 on Λ19 and 21 of 52 on spins 0–50.
   - Serial main-thread time is 20–23% of the solve: 7.5 of 36.9 s and
     15.7 of 66.7 s.
   - The KKT solve is 37–43% serial.
   - A main-thread `perf` profile (`perfmain.pbs`) shows MPFR vector work
     (waxpby/axpby) and elementwise multiplies.
5. Serial work moved onto the pool (rns56 + rns59, bitwise identical):
   - The fused path's linear-part products (the 54×2405 B coupling, in
     prepare and recover) now run through the existing `SparseParallel`
     plan inside the pool.
   - A thread-local vector pool, registered by the core solve loop: long
     (≥4096) MPFR elementwise vector ops on the main thread (scale,
     axpby, waxpby, hadamard, recip/sqrt, clip, norm_inf, is_finite) and
     `add_assign` split into 2048-element chunks.
   - The batched arrow `solve_many` trunk coupling runs over trunk rows,
     keeping leaf order per entry.

   Same node, 30 it, rns58 → rns59:
   | Case | Solve time | Serial time |
   |---|---|---|
   | Λ19, 32 thr | 38.3 / 38.3 → **34.1 / 34.5 s** | 7.2 → 3.6 s |
   | Λ19, 64 thr | 42.3 / 43.3 → **39.6 / 38.6 s** | 8.2 → 3.9 s |
   | Spins 0–50, 52 thr | 63.7 / 63.4 → **57.2 / 56.3 s** | 14.5 → 7.6 s |

   The KKT solve's serial time dropped 2.5 → 0.5 s.
6. Inner (arrow) refinement trace (`SDPX_TRACE_IR`, Λ19, full solve), 775
   passes in total:
   - 631 gain ≥1e10; typical: 1e45, then 1e11, then converged.
   - 120 stall.
   - A stall rule there would save <1%, so none was added.

7. rns60, bitwise: the componentwise dual error accumulates the linear
   operator column-parallel. Residual-update serial time 1.25 → 0.52 s.
8. rns61: owner ranks holding a single sampled block got a one-component
   local KKT. Arrow required ≥2 components, so these ranks fell back to
   scalar sparse QDLDL:
   - refactor median 36 vs 8.5 ms;
   - factor_response 25–27 s vs 9 s on two-block ranks.
   Arrow now also admits a single dense leaf (≥16 rows, ≥50% of the upper
   triangle stored). Sparse single components stay on QDLDL, and
   single-process runs are bitwise unchanged.
   - 2 nodes 16×8, 40 it: 64.3 → **58.8 s**.
   - factor_response spread across ranks 2.1–6.3 → 1.5–2.3 s.

9. rns62, bitwise: the exact inner residual (`ExactRows`) was split into
   ≥16-row chunks, but the 54 border rows hold ~1,260 entries each against
   ~140 for leaf rows. The chunks holding border rows set the critical
   path. Rows are now partitioned by entry count (4 tasks per worker).
   - ir.residual: Λ19 0.9 → 0.37 s; spins 0–50 2.7 → 1.0 s.
   - ir: 6.1 → 4.0 s on spins 0–50.
10. SVD anatomy (rns63 stage timers, Λ19 32 thr, median per cone):
    - total 174 ms: rotation replay onto U and V 90 ms, bidiagonalization
      29 ms, bidiagonal QR 28 ms, reflector accumulation 24 ms.
    - Replay is already one `mpfr_fmma` per output; block accumulation does
      not reduce work at n ≈ 45. Left as is.
11. Placement at 64 threads (Λ19, node43, rns61): one socket vs two, local
    vs interleaved memory all fall in 31.3–34.4 s (≤3%).
    - Same node: 32 thr 32.0–33.0 s vs 64 thr 34.3 s. Λ19 (14 leaves, 28
      cones) has no parallel slack beyond 32 threads.
    - Other nodes vary by ±10%; compare only within a job.

12. rns64/65, not bitwise in reported norms: the eight info norms took a
    scaled recurrence per entry (a division each) in one task per vector
    (~4 busy threads). For MPFR they are now sqrt of one exact dot product
    of the squared scaled entries (rounded once). This is more accurate;
    MPFR's exponent range needs no overflow scaling, and f64 is
    unchanged.
    - mu+info on spins 0–50: 1.17 → 0.55 s per 30 it.
    - The ising11 point stays bitwise.
    - The statistics test now compares MPFR norms within 256 ulp and
      checks the formulas on the summary's own norms.
13. rns67, bitwise: default start spent 2.7 s serially in
    `symmetric_initialization`, one full eigendecomposition per PSD cone
    for s and z. Cone margins now run in the cone pool and are folded in
    cone order.
    - Default start on spins 0–50: 5.3 → 1.8 s.

14. Reverted (rns69, bitwise): running the SVD's U pipeline (replay + left
    reflectors) and V pipeline side by side with `rayon::join`.
    - SVD median unchanged at 32 thr (183–185 ms), slightly worse at 64
      (133–146 → 139–147 ms).
    - The cone phase is throughput-bound; there are no idle threads.

15. rns70, owner path, not bitwise there: fused reduced refinement pass.
    - The reduced residual now also solves each owner's interior with the
      new residual and forms its border contribution.
    - One all-gather (site 323) carries the border residual, the
      correction's border RHS and the norm. This replaces sites
      320/321/322 plus 239/240/241/242 per inner pass.
    - The border residual and norm keep the old reduction order. The
      correction RHS is −e_border + Σ_rank c_rank; the old path seeded rank
      0 with −e_border.
    - `SDPX_FUSED_REDUCED=0` restores the old pass.
    - 2 nodes 16×8, 40 it, same nodes: 60.0 → 59.2 s; mean wait 13.8 → 12.9
      s (the refinement-site wait drops ~1 s; the rest is once-per-iteration
      sites).
16. rns71, not bitwise: certified Float64 step-length eigenvalue.
    - The step needs only λmin of the full-precision scaled matrix Δ, so
      a Float64 copy is solved with LAPACK.
    - The result is accepted when 8(n+1)ε‖Δ‖_F ≤ 1e-10·max(|λ|, 1/αmax):
      the conversion plus backward error then moves α by ≤1e-10
      relative. Otherwise it falls back to the 768-bit solver.
    - Step lengths are a free IPM parameter, scaled by the step fraction
      and kept interior.
    - eigmin 12–22 ms → 0.17 ms per call. Step phases: Λ19 2.7 → 1.15 s,
      spins 0–50 4.3 → 1.7 s per 30 it.
    - Solve per 30 it: Λ19 32 thr 28.5/31.4 → 26.7/28.7 s; spins 0–50 52 thr
      48.8/48.1 → 44.4/46.0 s.
    - ising11: 52 it, audit accepted, 9.04 → 8.17 s.

17. rns72/73: the Float64 λmin no longer uses LAPACK. Concurrent OpenBLAS
    calls from pool threads are not safe in every build. It is now pure
    Rust: Householder tridiagonalization without vectors plus Sturm
    bisection, with a more conservative bound 2(n+1)²ε‖Δ‖_F.
    - A unit test checks it against the MPFR eigensolver.
    - eigmin is 0.14 ms per call.
    - Spins 0–50, 52 thr, 30 it: 49.9 → 47.7 s.
    rns71 audited full solves (same step-length rule):
    - Λ19 32 thr **101.9 s**, 64 thr 106.4 s.
    - Spins 0–50, 52 thr **246.9 s**.

18. rns75, bitwise: the SVD rotation replay onto V was 64% of the SVD on
    spins 0–50 (133 of 209 ms). Its rows are independent, so they are now
    offered to idle pool threads on any pool worker, not only when a cone
    was granted ways.
    - Spins 0–50, 52 thr: cone phase 11.1 → 8.2 s per 30 it (busy 34 → 45).
    - Λ19 neutral (throughput-bound).
    - The same offer for bidiagonalization/reflector columns (rns76) was
      neutral and reverted.
    rns75 audited:
    - Λ19 32 thr 105.6 s; 64 thr **101.6 s**.
    - Spins 0–50, 52 thr **237.6 s**.
    - 2 nodes (TCP) 253.2 s.
19. rns77, bitwise: TCP allreduce latency is 72 µs vs 7.8 µs with `openib`.
    Each owner collective used 5–12 scalar allreduces for site / operation
    / length agreement (~1,460 per iteration). One handshake now carries
    them, plus the payload for all_true/agree, as a single vector
    max-allreduce of (x, −x) pairs; `World::agree_u32` also uses one.
    - 2 nodes 16×8 TCP, 40 it, same nodes: allreduces 50,989 → 7,458;
      solve 57.2 → **53.1 s**.

rns77 audited, 2 nodes 16×8 over TCP: **238.7 s** (4:07 wall). This equals
one node at 52 thr (237.6 s).

Layouts on rns77 over TCP, 40 it:
- 16×8: 53.6 s.
- 8×16: 58.6 s.
- 4×32: 73.6 s.
- 2×52: 103.2 s.

A 13-block rank on 52 threads is 2× slower per KKT phase than one process
with 26 blocks on 52 threads. Block-phase time is set by the per-block
critical path; beyond ~2 threads per block the residue-kernel prime split
barely shortens it.

Two hypotheses were tested and reverted (both neutral, same nodes):
- rns78: exact pooled rows for the owner reduced residual.
- rns79: single-local owner dispatch on the main thread (vector pool) plus
  entry-parallel border corrections.

Node70 is broken (home filesystem not mounted; SSH refused). Jobs placed
there hang at launch; 2-node jobs now name their nodes explicitly.

Spins 0–50 at 52 thr, 30 it: rns58 63.5 s → rns64 50.3–52.3 s.
2-node layouts with uniform per-rank structure (rns67, 40 it, same nodes):
- 16×8: 59.3 s.
- 14×9: 60.5 s.
- 26×4 (one block per rank): 58.1 s.
Waits are similar (12–14 s mean), so the remaining ~20% wait is per-round
jitter over ~38 small synchronizing rounds per iteration, not layout.

rns67 audited (norm change):
- Λ19 32 thr 114.8 s; 64 thr 110.0 s.
- Spins 0–50, 52 thr: 265.6 s (node9).
- 2 nodes 16×8: 233.3 s.

rns63 audited full solves:
- Λ19 32 thr **116.7 s**; 64 thr **109.9 s**.
- Spins 0–50, 52 thr: **257.5 s**.
- Spins 0–50, 2 nodes 16×8: **230.4 s** (4:04 wall).

Layouts, 2 nodes, rns60, 40 it: 16×8 63.5 s, 8×16 65.6 s, 4×32 84.8 s.
rns61, 2 nodes 16×8, audited: **258.8 s** solve (4:29 wall).

Audited full solves, rns60, idle nodes, 768 bits:
- Λ19 32 thr **111.6 s** (SDPB 204.2).
- Λ19 64 thr **123.3 s** (SDPB 152.8).
- Spins 0–50, 52 thr **291.2 s** (SDPB, 64 ranks: 328).
- Spins 0–50, 2 nodes 16×8: **279.3 s** (4:47 wall).
All 119/177 it; audits accepted.

2-node hangs (rns67 and rns71, several node pairs), diagnosed 2026-09-26.
Under the gdb wrapper, SIGTERM stops each rank and prints its backtrace.
- All 16 ranks were in the same collective: `reduce_max` →
  `WorldCollective::gather` → `World::gather_slice`.
- 12 ranks had finished the payload Allgatherv and waited in
  `exchange_status`. 4 ranks were still inside the Allgatherv
  (`gather_bytes`), polling `btl_openib_component_progress` /
  `mlx5_poll_cq`.
- So a transport message was never delivered: an OpenMPI 4.1.4 `openib`
  BTL problem. This build has no UCX, and `openib` is deprecated with known
  `MPI_THREAD_MULTIPLE` issues. The two earlier RIP=0 segfaults on the main
  thread are consistent with the same transport.
- Mitigation: `--mca btl self,vader,tcp` over ib0. All 2-node scripts
  now use it.
- Repeats of rns74 (same code, three node pairs) with `openib`:
  - one solved (247.3 s, audited);
  - one aborted when SDPX's wire check detected an undecodable payload
    ("wire encode/decode failed on at least one rank");
  - one hung.
- With TCP, 2 of 2 completed:
  - full solve 260.9 s, audited;
  - A/B 40 it on the same nodes: rns67 59.3 s vs rns74 56.4 s.
- Also added: the world `allreduce` now aborts with a backtrace if called
  from a non-control thread (none observed).

One unexplained SIGSEGV on rank 0 during the first rns50 2-node start. It
did not recur in five later runs on the same nodes; the 2-node scripts now
preload libSegFault for a backtrace.

## 2026-09-25 — how to overtake SDPB on very large problems: evidence

SDPB anatomy (`--verbosity 2` profiler, Λ19 spins 0–50, 64 ranks,
iterations 2–10, ≈ 0.85 s/it): search directions (predictor + corrector)
0.31 s (36%), step-length Choleskys of X+αdX, Y+αdY 0.21 s (24%), bilinear
pairings 0.18 s (21%), Schur complement + Q (block Cholesky 0.032, L⁻¹B
0.018, bigint syrk 0.011) + Cholesky(Q) 0.005 → 0.08 s (9%). At N = 54 SDPB
is per-block-work bound; its Q machinery is 9%. SDPB uses the residue
kernel only for Q = PᵀP; block Cholesky, Trsm and Cholesky(Q) are
Elemental BigFloat.

SDPX anatomy (rns42, 52 thr, 2.20 s/it): refinement solves 0.97 s (44%;
~5 passes per iteration: 3 right-hand sides of the HSD step plus
corrections, vs SDPB's 2), cone scaling (SVD-based NT) + step lengths
0.50 s (23%; SDPB needs only Choleskys), Schur assembly + arrow factor
0.23 s (10%). Per pass SDPX ≈ 1.3× one SDPB direction; the pass count is
the multiplier. Iterations: SDPX 177 vs SDPB 265 (Λ19: 119 vs 243).

Arrow (reduced system) implementation: leaf LDLᵀ (P³/3), L⁻¹B (P²·N),
BᵀS⁻¹B (P·N²/2) and the border factor (N³/3, serial) are all scalar MPFR;
10% of an iteration at N = 54 but the fastest-growing term (N ≈ Λ²/8 for a
single correlator, thousands for mixed correlators).

Precision (leaf LDLᵀ pivot spread, 768 bits): Λ19 1e121 at iteration 1 →
6e234 at the end; spins 0–50 1e134 → 1.4e250. At 512 bits both solvers
fail on Λ19: SDPB's block Cholesky breaks ("not numerically HPD"); SDPX
derails near iteration 20 identically with equilibration bounds 1e±38.5,
1e±58 or 1e±100. The late spread beyond 2^768 is benign (IPM ill-
conditioning near the optimum lies in irrelevant directions); the early
one is not. Precision is at parity and grows with problem size.

## 2026-09-25 — stalled outer corrections: skip by measured contraction

Evidence (IR trace, Λ19 spins 0–50): at 768 bits 596 of 603 corrections
gain < 10× (stall); at 1024 bits 482 of 534 solves need no correction and
corrections gain 1e8–1e19. Both precisions follow the same 177-iteration
trajectory. So the correction solve contracts by ~κ(S)·eps ≈ 0.8 at 768
bits (the reduced Schur complement is extremely ill-conditioned from the
first iteration, ‖x‖∞ ≈ 1e80), and the stalled passes do not change the
IPM path. G/Ginv/R/Rinv consistency was checked analytically (error ~
κ(L1)·eps ≈ 1e-217, not the cause).

Change: refinement records the improvement ratio of each right-hand
side's first correction; while the current factorization's last recorded
ratio is below `iterative_refinement_stop_ratio`, later right-hand sides
skip the correction. Reset at every KKT update (refactor); local path only
(MPI keeps the old rule). ising11 bitwise identical (no corrections there).

Audited full solves, idle nodes, 768 bits:
- Λ19 32 thr 171.3 → **154.9 s**, 64 thr 189.2 → **162.5 s** (SDPB 204.2 /
  152.8 s); solve passes 692 → 589; 119 it, audits accepted.
- Λ19 spins 0–50 52 thr 432.9 → **406.1 s**; passes 1073 → 871; 177 it,
  audit accepted.

Follow-up (rns46 → rns48, kept):
- Residual skip: when the stall is already predicted, the outer residual
  itself is skipped (only H·z is restored).
- MPI owner path gets the same prediction; the stall decision is agreed
  across ranks.
- Persistent stall floor: when a correction stalls, the relative residual
  it reached is kept across factorizations. On a later right-hand side, a
  first residual within `stop_ratio` of that floor skips the correction.

Audited, idle nodes, 768 bits, 177 / 119 iterations unchanged:
- Λ19 32 thr: 154.9 → **141.5 s**; passes 589 → 484.
- Spins 0–50, 1×52 thr: 406.1 → **370.1 s**.
- Spins 0–50, 2 nodes × 4 ranks × 16 thr: 341.8 (rns46) → **309.0 s** solve
  (322 s wall). SDPB is 328 s at 64 ranks on 1 node; its 2-node runs hang
  on this cluster.

Rejected in this round:
- Stopping on the floor after a correction: neutral.
- The always-on cone inner rule: slower.
- A separate KKT pool: no difference.

## 2026-09-24 — more threads than blocks: SDPB-style per-block workers

Question: SDPB speeds up from 32 to 64 ranks on Λ19 (28 blocks); SDPX got
slower. SDPB gives heavy blocks several ranks and splits their dense work.

Found: SDPX's equivalent ("ways", measured floor(cost/share) workers per
block) could not pay — any split call (`Split::Ways`/`Pool`) fell back to
the old path that encoded every prime of both operands up front, skipped
the residue caches and decoded separately. Changes (all bitwise, cluster
gate + new pool/ways tests for congruence and svec_quadratic):
- `stream_primes`: a split call gives each worker a contiguous prime range,
  streamed with its own scratch and CRT accumulator over the shared
  (cached) operands; accumulators merge in range order (exact integer digit
  sums; the fraction only feeds round() with a 1/4 margin); the final
  rounding pass is split over outputs. The old encode/decode path is gone.
- Cone lanes pass `ways = workers / lanes` to their kernels.
- Pool workers are pinned one per CPU when the process is bound to exactly
  that many CPUs (never for a wider mask, so unbound ranks cannot collide).

Measured (8 it, idle nodes, pinned): Λ19 32/64 thr before 12.0/14.1 s,
after 12.1/12.8 s; spins 0–50 52 thr 20.0 → 19.3 s (pinning −4%). `perf
stat` shows why 64 still does not beat 32 on Λ19: 32 thr 2.92 GHz, IPC
2.43; 64 thr 2.39 GHz, IPC 2.02 (instructions +3%). Per-core throughput
falls ~32% with 64 busy cores on the socket (spreading over both sockets:
2.47 GHz), while SDPX's per-iteration work is many small per-block kernels
(n≈45) with little left to split. SDPB's per-iteration work (dense Schur
complement per block, twice the iterations) splits far better. Guidance:
threads ≈ number of PSD blocks, MPI ranks for many blocks.

## 2026-09-24 — final audited set (rns36), idle nodes

Build rns36 (gated: 435 lib tests, ising11 bitwise vs rns33, audits pass).
768 bits, each run on an idle node's quietest NUMA domains, `--localalloc`:

| case | SDPX | SDPB (same method) |
|---|---|---|
| Λ19, 32 cores | **167.4 s**, 119 it, audit accepted | 204.2 s, 243 it |
| Λ19, 64 cores | 186.9 s (32-thread run: 167.4 s) | 152.8 s |
| Λ19 spins 0–50, 1 node 52 thr | 469.7 s, 177 it, audit accepted | 328 s (64 ranks), 557 s (32 ranks); 265 it |
| same, 1 node 4 ranks × 16 | 441 s wall, audit accepted | |
| same, 2 nodes 8 ranks × 16 | 363 s wall, audit accepted (one node shared with another of these runs) | SDPB 2-node hangs on this cluster |

Audit of every spins 0–50 point: primal 3.8e-99, dual 7.6e-92, gap 4.3e-89,
mapping 8.9e-31, objective vs SDPB 1.9e-42 — identical across 1 process,
4 ranks and 8 ranks.

## 2026-09-24 — IPM residual costs; refinement floor test (reverted)

All builds gated on the cluster (`gate.pbs`: lib tests, ising11 512-bit
audits, bitwise point comparison); no local runs.

Kept:
- IPM residual workspace keeps basis residues (as the KKT one): bitwise.
- Componentwise dual residual |A|ᵀ|z| for dim-1 sampled blocks via the exact
  quadratic kernel on |Q| and |x| (all terms nonnegative, rounded once):
  101 → 16 ms per iteration (Λ19 spins 0–50, 52 thr); IPM residual update
  1.79 → 1.06 s per 8 it; points bitwise identical.

Reverted (pinned A/B on idle nodes, 8 it, 2 reps):
- Cone inner parallelism for any pool (workers > 1): Λ19 32 thr 12.05 vs
  11.97 s, spins 0–50 52 thr 21.0 vs 20.8 s — neutral.
- Arioli–Demmel–Duff stop for outer refinement (ω = max|e_i| /
  (|b| + |K||x|)_i ≤ 8 eps, |A| via a magnitude copy of the operator, |H|
  via |G||Z||G|): triggered on 2 of 63 solves (Λ19) and 0 of 62 (spins
  0–50) while costing 1.7–2.9 s. The stalled corrections are not at the
  residual's rounding floor; they are limited by the accuracy of the
  condensed correction solve itself. An earlier cheap estimate using |Kx|
  instead of |K||x| is invalid (cancellation rows give ω ≈ 1).

## 2026-09-24 — thread width, NUMA memory, MPI layouts, clean-node finals

Measurement fixes: `numactl --membind=<several nodes>` fills the first
node, so 64-thread runs used one memory controller; all pinning now uses
`--localalloc` (first touch). Larger problem, 8 it: 32 thr 27.5 → 23.8 s,
48 thr 25.0 → 21.7 s; Λ19 unchanged. Nodes are shared even with
`singlejob` (loads 40–60 seen); `final.pbs` flags a run when a chosen
domain is > 5% busy, and finals are placed on idle nodes (`-l
nodes=nodeNN`). SDPB baselines were checked: busy ≤ 0.01.

Thread width (8 it, pinned, localalloc): Λ19 (28 blocks) 16/24/28/32/48/64
thr = 16.5/15.1/13.2/12.5/13.5/14.8 s; spread over both sockets at 64 thr
14.5 s (not bandwidth). Λ19 spins 0–50 (52 blocks) 32/48/52/64 = 23.8/21.7/
21.3/22.0 s. Best width ≈ PSD block count; beyond it extra workers add
stealing/interference without useful inner work (capping ways at 1 or a
1-thread global pool changed nothing). Not made automatic: problems with
few large blocks do benefit from inner parallelism, and no principled
threshold is available; guidance recorded instead.

MPI (owner path, per-rank `numactl` binding, 8 it): Λ19 1×32 12.7, 2×16
16.7, 4×16 14.6 s (ranks do not pay at 28 blocks). Larger problem 1×64
21.4, 4×16 19.7, 2 nodes 4×32 20.0, **2 nodes 8×16 16.5 s** (−25% vs one
64-thread process). SDPB on 2 nodes hangs on this cluster (default
transport after iteration 1; TCP fails with "unexpected process identifier"
because every node has identical docker0/bridge addresses; ib0-only
out-of-band also hangs before the first iteration).

Cone inner parallelism: enabled whenever a worker is spare (was ≥ 2 per
lane); bitwise identical (cluster gate), SVD tail work shared.

Clean-node audited finals (rns31, 768 bits):

| case | SDPX | SDPB |
|---|---|---|
| Λ19, 32 cores | **173.1 s**, 119 it, audit accepted | 204.2 s, 243 it |
| Λ19, 64 cores | 192.6 s (best within 64 cores: 173.1 s at 32 thr) | 152.8 s |
| Λ19 spins 0–50, 52 thr | 474.3 s, 177 it | 328 s (64 ranks), 265 it |

## 2026-09-24 — accuracy floor found and fixed: scaling + exact residuals

Hypothesis tested with a temporary trace (`SDPX_TRACE_IR=1`: per-solve
outer/inner refinement residuals, pass counts, stop reason, ‖x‖∞, KKT entry
range). Observed on Λ19 spins 0–50 at 768 bits: from iteration 2 on, every
outer refinement stops on the stall ratio after one correction (ratio
~1.3, later rejected); the first residual sits at ~1e-100 absolute while
the tolerance is 1e-172; no dynamic-regularization bumps. Λ19: ‖x‖∞ =
2.7e80 at the first solve, reduced-KKT entries 1e3…4e-95. Diagnosis: the
residual b − Kx is rounded per product, so its error is eps·|K||x|
(1e-231·1e131 ≈ 1e-100); corrections computed from it are noise, and the
Newton directions stay at that floor until the IPM stalls (gap 1e-28).

Fixes (both kept):
1. **Precision-scaled equilibration bounds.** Ruiz clipped cumulative
   scaling to [1e-4, 1e4], far too narrow for column scales spread over
   1e80. Default is now eps^(±1/4) for MPFR (binary64 keeps exactly 1e±4,
   medium bitwise unchanged); 1e±58 at 768 bits.
2. **Exact inner residual.** Each row of the reduced KKT residual,
   b_i − Σ K_ij x_j, is accumulated exactly and rounded once (`dot_fma`,
   exactdot window 32k bits) — Wilkinson refinement with an extra-precise
   residual. MPFR only.

Results (audited, 768 bits, pinned):
- Λ19 spins 0–50, 64 thr, wide bounds (1e±100): **Solved, 177 it**, audit
  accepted (primal 3.8e-99, dual 7.6e-92, gap 4.3e-89, mapping 8.9e-31,
  objective vs SDPB 1.9e-42). Before: InsufficientProgress at 345 it.
  SDPB needs 265 it at the same precision.
- Λ19, rns27 (both fixes): 32 thr **180.7 s** (was 225.1; SDPB 204.2),
  64 thr 190.5 s (was 241.2; SDPB 152.8); 119 it, audits accepted. Inner
  refinement steps 4261 → 775, inner-refinement time 44 → 10 s.
- ising11: 52 it (was 50), audits pass, residuals tighter.

## 2026-09-24 — 32/64-core round, part 3: audited finals vs SDPB, larger problem

Build rns22 (gemm-crate kernel products, SVD rotation replay, residue
caches; fused pass reverted). Same node, each run bound to the quietest
NUMA domains (SDPX `numactl`, SDPB `mpirun --cpu-set`), 768 bits:

| case | SDPX | SDPB |
|---|---|---|
| Λ19, 32 cores | 225.1 s, 119 it, Solved, audit accepted, 1.5 GiB | 204.2 s, 243 it |
| Λ19, 64 cores | 241.2 s, 119 it, Solved, audit accepted, 1.9 GiB | 152.8 s, 243 it |
| Λ19 spins 0–50, 64 cores | **InsufficientProgress**, 345 it, 954 s; audit fails (gap 1.3e-28, mapping 8.6e-21, objective vs SDPB 3.7e-9) | 328 s, 265 it, optimal |
| same, SDPX at 1024 bits | Solved, 177 it, 552 s, 4.3 GiB (audit at 768 bits pending) | — |

Session gain, pinned 8-it A/B pre-session (rns13) vs rns22: 32 thr 16.84 →
14.25 s (−15%), 64 thr 17.41 → 15.11 s (−13%); RSS +14–16%. Residue caches
alone (A/B with caches disabled): −3.3% (32 thr) / −7% (64 thr) for
+255/+315 MiB; kept.

Findings. (1) At 32 cores Λ19 is now within 10% of SDPB (was 1.64× at the
start of the round, though that figure was unpinned). (2) SDPX does not
gain from 32 to 64 threads on Λ19 (28 blocks): each sequential parallel
region costs about one whole block. (3) The larger input (52 PSD blocks,
57 k constraints) exposes a 768-bit accuracy floor: SDPX's primal residual
plateaus near 7e-29 while gap and dual residual reach 1e-157; SDPB solves
it at 768 bits. At 1024 bits SDPX converges in 177 iterations. This is the
first priority before any larger-problem speed claim.

Λ27 input: SDPB cannot solve the pinned-generator Λ27 (spins 0–26) at 768
or 1024 bits, with gap 1.412625 or 1.40 (Cholesky breakdown at 768;
maxComplementarity / maxIterations otherwise; primal error stuck at 2.67 in
all runs). Identical failure across gap and precision points to the
generated input, not physics. Not pursued; inputs kept under
`~/projects/sdpx-scaling-20260923-01/w6/`.

## 2026-09-24 — 32/64-core round, part 2: measurement, contention, fused pass

**Measurement.** Compute nodes are shared despite `naccesspolicy=singlejob`
(node178 ran another user's 30-core job; no cpuset, affinity 0–255), so
`taskset` on "the first T CPUs" collided with foreign threads: identical
runs varied 16–24 s. Every run now binds CPUs and memory to the quietest
whole NUMA domains (`pickcpus.py` + `numactl`); repeat runs agree to ±2–4%.
Earlier single-run comparisons (W1 cluster numbers, the removed buffer
pool's "+2% time", SDPB 64/128) were taken without this control.

**Allocator ruled out.** glibc here is 2.17, which ignores
`GLIBC_TUNABLES` (the first test was void). The legacy `MALLOC_*_`
variables cut page faults 258 k → 7 k and system time 13 → 9 s, but IP
time moved −2% (32 thr) / +1% (64 thr).

**OpenBLAS packing contention (fixed).** `perf` sample counts, 1 vs 32
threads, 8 iterations: GMP multiply +6%, kernel encode/CRT ≈ +10%, but
OpenBLAS `dgemm_kernel` +50% and its packing copies +180% (itcopy +370%):
every small dgemm packs into OpenBLAS's shared global buffer pool. The
residue kernel now calls the `gemm` crate (faer's backend; thread-local
packing, `Parallelism::None`); products are exact integers < 2^53, so bits
are identical (437 lib tests; ising11 A/B points identical). Pinned A/B,
8 it: `cone_wprod` CPU 11.5 → 7.8 s (32 thr), 23.7 → 11.1 s (64 thr; 1 thr
6.8 s); user CPU −5%; RSS −40…−100 MiB; IP 14.7 → 14.7 s (32), 16.2 →
15.5 s (64). Kept.

**Fused refinement pass (W2).** Recover + residual as one block task per
pass (block-local recover, `+x`, forward, adjoint and `H z`, then ordered
sums). Bitwise identical. Pinned A/B: its own phase −20% (4.8 → 3.8 s), IP
unchanged at 32 thr (15.2/14.9 vs 15.3/14.9), ±3% at 64. Blocks are
homogeneous, so the sum of per-phase maxima equals the max of per-block
sums; removing barriers does not shorten the critical path. Same-binary
on/off (pinned, 2 reps): off is 3–4% faster (32 thr 14.51 → 13.89 s, 64
thr 15.58 → 15.09 s). Reverted; the pure extractions it introduced
(`forward_leaf`, `adjoint_leaf`, `apply_block`, `solve_reduced`,
`recover_linear`, block-row adjoint slices) stay, bitwise neutral.

**Other pinned 8-it A/Bs (not adopted).** Ways over-decomposition
(`floor(f·cost/share)`): f = 4 → −3.5% at 32 thr, 0% at 64 thr, +140 MiB;
f = 8 → +3% / 0%. Cone inner parallelism at 32 thr (any spare worker instead
of ≥ 2 per lane): −1.6%, within noise.

**SVD rotation replay (W4 part).** The bidiagonal QR never reads U or V:
rotations are logged and replayed on row-major U/V, rows split across the
pool when inner parallelism is on (≥ 2 workers per cone lane). Bitwise
identical. 64 thr: scale cones 1.9 → 1.4 s, slowest SVD 240 → 170 ms, IP
−3%; 32 thr (inner parallelism off): unchanged. Kept.

## 2026-09-24 — 32/64-core round: baselines, cached encodings, where the time goes

Λ19 768-bit, cluster EPYC 7742 nodes, audited full solves unless marked 8-it.

| run | result |
|---|---|
| SDPB 64 ranks, 1 node | 158 s |
| SDPB 128 ranks, 2 nodes | hung after iteration 1 (no output for 90 min; walltime) |
| SDPX 64 threads (rns13) | 293 s, 119 it, Solved, audit accepted, 1.68 GiB |
| SDPX 2 nodes, 8 ranks × 16 (8-it) | 17.1 s vs 16.2 s for 1 node × 32: no gain |

W1 (streamed CRT in prime groups + cached residues of Q, V, Rinv, G, Ginv):
bitwise identical (kernel tests, ising11 A/B points identical, −4.7% at 4
threads locally). On the cluster, 8-it: 32 thr 16.23 → 16.61 s, 64 thr
17.79 → 17.24 s, RSS +21% (1192 → 1448 MiB at 32). Kept for now as
neutral; the encodings are not what limits 32+ threads.

Diagnosis (8-it phase totals, same build, 1 / 8 / 32 threads):
recover 19.7/4.4/2.5 s, prepare 19.6/3.9/2.0, residual 29.9/5.2/2.8, ir
27.9/5.4/2.7; IP iteration 184/33/16.6 (11× at 32). Plain MPFR code keeps
its per-call cost (cone_svd CPU 30.2 → 33.3 s), while residue-kernel phases
look 2–3× slower per call; that is the waiting time of pool-split callers,
not contention. Allocator tunables (glibc mmap/trim) changed nothing
(±5%, 1.3 M page faults per run). `perf stat`: only 12 of 32 CPUs busy (20
of 64); `perf` at 64 thr: 14% of cycles in rayon idle stealing
(crossbeam-epoch). The limit is the pass structure: each refinement pass
(63 per 8 iterations) runs 4–5 block-parallel phases, each waiting for its
slowest block. Next: one block task per pass (W2).

Λ27 (spins 0–26) at 768 bits: SDPB stops after ~128 iterations, "block_1
not numerically HPD" in the Schur Cholesky, max block condition number
1.2e319 (> 2^768). Λ27 comparisons need ≥ 1024 bits for both solvers;
SDPB rerun at 1024 submitted. New inputs (pinned generator, 768-bit XML,
converted with the pinned `pmp2sdp` at 768): `L27-s26` XML sha256
`6f1b106d…477a`, `L19-s50` `10210690…9f08`; Λ11 regeneration matched the
historical XML hash `aaadbc3c…b9c2b`.

## 2026-09-24 — decision: keep refinement tolerance; no cross-rank block split

Question: can outer/inner refinement do less work (about 7.3 passes and 36
inner refinements per Λ19 iteration)? The MPFR default reltol = abstol =
eps^(3/4) (7e-174 at 768 bits) is far below the 1e-42 / 1e-30 gates, so
looser values were tested as settings (no code change). Full audited solves:

| case | tolerance | status | it | time | audit |
|---|---|---|---|---|---|
| Λ19 768, 32 thr | 7e-174 (default) | Solved | 119 | 316 s | pass |
| Λ19 768, 32 thr | 1e-145 | AlmostSolved | 118 | 218 s | FAIL: mapping 2.6e-29 > 1e-30 |
| Λ19 768, 32 thr | 1e-116 (√eps) | InsufficientProgress | 182 | 295 s | FAIL: mapping 9e-28, gap 5e-49 |
| Λ11 512, 1 thr | 2.5e-116 (default) | Solved | 50 | 15.6 s | pass (87 refinements) |
| Λ11 512, 1 thr | 1e-97 / 8.6e-78 | Solved | 50 | ≈ same | pass, identical residuals (0 refinements) |

Decision: keep eps^(3/4). Near the optimum Λ19's reduced system is ill
conditioned enough that looser backward error degrades the directions and
the original-coordinate mapping gate fails. The refinement passes are the
price of the accuracy target, not waste.

Cross-rank splitting of one block (SDPB `Block_Map` with num_procs > 1) is
not pursued. SDPB's Elemental distributes each dense operation. In SDPX one
residue-kernel call per block takes a few ms and there are hundreds per
iteration, so an MPI round trip per call costs more than it saves, and
Λ19 has only 28 blocks. Heavy blocks already get extra threads within a
rank (measured ways). Multi-node pays when blocks greatly outnumber cores;
rank balance then uses the existing measured cost history
(`--cost-history-out/in`).

## 2026-09-24 — MPI owner path: rank pools reach the owner cones (SDPB source review)

SDPB source (67ebd53) review:
- `compute_search_direction` + `solve_schur_complement_equation`: the Schur
  complement is block-diagonal (per-block Cholesky L_j), coupled only
  through the small N×N `Q = Bᵀ S⁻¹ B`. Each direction does block-local
  solves, one global N-vector reduction, a Q solve and a broadcast, and
  no refinement. SDPX's reduced arrow system has the same shape (14 leaves
  = S_j, 54-row border = Q).
- `block_mapping`: a heavy block gets several ranks, light blocks share a
  rank (as in the paper). `bigint_syrk` computes Q per node in shared
  memory and then reduces across nodes.

First qualification of SDPX's owner-partitioned MPI path on Λ19 (768 bits,
8 iterations, IP loop; baseline 1 rank × 32 threads ≈ 18–20 s): 1 node
2×16 took 58.2 s, 4×8 took 38.0 s; 2 nodes 1×32 per node took 63.4 s, 2×16
per node 39.2 s. Cause: with `--partitions auto` each rank owns one
owner, `OwnedCones` parallelizes only across owners, and the owner's
`CompositeCone` has no pool. Cone scaling, step lengths and the owners'
sampled residual products therefore ran serially per rank (scale cones
19.6 s vs 3.2 s).

Fix: when a rank has fewer owners than workers, each owner's cones adopt
the rank pool (`CompositeCone::share_pool`, `ConeThreading::with_pool`;
a nested install runs inline). Owner residuals get the same pool for
sampled products. Debug assertions now state this condition
(`owner_cone_pools_ok`). Local `--partitions 2`: points bit-identical,
scale cones 2.83 → 1.43 s.

Cluster after the fix (8 iterations, IP loop):

| layout | before | after |
|---|---|---|
| 1 node 2×16 | 58.2 s | 26.6 s |
| 1 node 4×8 | 38.0 s | 21.8 s |
| 2 nodes 1×32 per node | 63.4 s | 26.9 s |
| 2 nodes 2×16 per node | 39.2 s | 22.1 s |

Threaded path unchanged (repeated A/B: 18.8/18.8 s vs 19.2/20.4 s).
Full 2-node solve (4 ranks × 16 threads): `Solved`, 119 it, audit passes,
399 s wall, 0.57 GB rank-0 RSS. Single node, 32 threads: 317.7 s.

Multi-node is correct but not yet faster than one node. All gathers
together take only 0.18 s per 8 iterations, so communication is not the
cost. Each pass is bounded by its slowest block. On top of that, the owned
KKT adds per-factorization border work (`owned.local_assemble` +
`factor_response` ≈ 0.3 s/it per rank) and 515 small batched solves per
8 iterations.

## 2026-09-24 — measured load balancing (SDPB 2.0, arXiv:1909.09745 §2.2)

Paper points applied or checked:
- §2.2.1/§2.2.2: block costs are measured, not modelled, and worst-fit
  gives expensive blocks several cores. SDPX's structural n³ model misses
  data-dependent costs: at Λ19, 8 threads, per-block time within one scaling
  call varies 4–10× (e.g. recover max 36 ms, mean 20 ms, min 9.5 ms) while
  block sizes only range over 41–47. The residue prime count follows
  exponent spreads.
- Implemented: each condensed PSD block and each sampled block workspace
  records its last measured work per action (seconds × ways). Before a
  pooled call, blocks above the fair share `total/W` get `floor(work/share)`
  ways (≤ 8), and their residue products split into that many prime groups
  (`rns_blas::with_split_hint`); the others stay serial. Results are
  bit-identical (the products are exact).
- Not applied: GMP instead of MPFR (§2.1), which conflicts with the MPFR
  contract. Per-core Q copies (§2.2.3–2.2.4) don't arise in the threaded
  solver; the Schur matrix is held twice (condensed plus reduced LDL), about
  88 MB at Λ19 — left as a follow-up.

Cluster, Λ19 768 bits, base = previous final build (`rns10`), 8 iterations, IP loop:
32 threads 19.86 → 18.66 s (−6%), 8 threads 36.76 → 35.09 s (−4.5%). Full
32-thread solve: 317.7 s wall (was 322.5 s), `Solved` 119 it, audit
passes; peak RSS 1.30 GiB (was 1.17; split blocks keep more buffers live).
Single runs. Kept for the time gain; the memory increase is noted.

The remaining 32-thread gap to SDPB (194 s) is structural. SDPB has one
global reduction per iteration (Q). SDPX runs about 7 outer solve passes
per iteration, each with several block-parallel barriers, and each barrier
waits for its slowest block.

## 2026-09-24 — multithread latency and memory (Λ19, 768 bits)

Memory: the receipt now records peak RSS (`receipt::peak_rss_bytes`, via
getrusage; also used by the FFI), and `SDPX_PROFILE` prints `MEMORY <step>`
marks. Local Mac, 4 threads, peak after each step, before → after:

| change | effect |
|---|---|
| sampled `materialize` assembles CSC column by column (was triplets + sort, ≈4× the final size); runs on the pool; bit-identical | materialize 1211 → 316 MiB; setup ≈10 s → 2 s |
| `install_sampled` frees the Ruiz-scaled copy before re-materializing | removes a transient 290 MB |
| the materialized A moves into problem data (`new_cow`) instead of being cloned | −290 MB |
| fully sampled problems (every PSD cone sampled, other cones zero): the condensed KKT keeps A's structure only, not its values | −250 MB |
| per-column A entry lists of sampled blocks released after install | −27 MB |
| exact sampled adjoint (below) replaces the per-block `wdiag` tables | −250 MB |

Peak RSS: 2352 → about 1050 MiB (local, 4 threads).

On the cluster, peak grew with threads (8 → 32: 1.1 → 2.0 GB). The
per-thread residue-kernel buffer pool retained every buffer that was live at
the same time. Measured twice each, 8 iterations:

| threads | pool | no pool |
|---|---|---|
| 32 | 1961/2047 MiB, 23.7/23.9 s | 1158/1155 MiB, 24.2/24.4 s |
| 8 | 1087/1095 MiB, 43.4/44.9 s | 909/913 MiB, 44.5/45.1 s |

The pool was removed. The isolated concurrency microbenchmark (repeated
identical allocations) favours the pool; the solver does not. Limiting glibc
arenas (`MALLOC_ARENA_MAX=2`) changed little (2021 → 1903 MiB), so the
growth was the retained buffers, not arena fragmentation.

Speed:
- Arrow LDL trunk coupling (54 trunk entries × 14 leaves of dots) was serial
  on the solve path. It now runs in parallel over trunk entries, each
  keeping its leaf order, so results are bit-identical. Cluster, 32 threads:
  `ir` 0.56 → 0.35 s/it, `trsv` 0.105 → 0.058 s/it.
- Exact sampled adjoint (`rns_blas::svec_quadratic`): for each dim-1 block
  it computes `v_k = Σ_{i≤j} c_ij x_t q_ik q_jk` (with `c = sqrt2` off the
  diagonal) as one exact residue expression, rounded once. Previously it
  was an exact dot against per-term-rounded `wdiag` weights, so the new form
  is more accurate. Test against a 6×-precision exact reference. Full local
  Λ19: `Solved`, 119 it, audit passes (same residuals).

Final tree (`rns10`, no pool), cluster, full audited Λ19 solves (`Solved`, 119 it,
audit passes each time: primal 4.8e-77, dual 6.3e-74, gap 5.6e-75,
reference 1.7e-42, mapping 5.7e-31):

| cores | SDPX start of 2026-09-23 | SDPX now (wall, peak RSS) | SDPB (wall) |
|---|---|---|---|
| 8 | — | 571.0 s, 0.93 GiB (job 220944) | 583.7 s, 8 × 115 MiB (job 220921, different node) |
| 32 | 394.7 s | 322.5 s, 1.17 GiB (job 220943) | 194 s (2026-09-22) |

At 8 cores SDPX is now level with SDPB (2% faster, single runs on
different nodes). At 32 cores it is 1.66× slower; the remaining gap is
per-pass latency of the residue kernels under shared-cache contention and
the per-block SVD critical path.

Cluster A/B (`rns7`, 8 iterations, IP loop): 32 threads 27.3 → 21.7 s (−21%),
8 threads 57.0 → 39.4 s (−31%). Full 32-thread solve (with the pool): 362.5 s,
119 it, audit passes. At 32 threads `residual` went up (0.49 → 0.68 s/it);
the kernels contend for shared cache there.

## 2026-09-23 — exact residue-BLAS kernel (SDPB `bigint_syrk` model)

Hypothesis: the gap to SDPB is MPFR dense products done one element at a time.
SDPB instead computes high-precision products as exact residue `dgemm`s.

Change (`algebra/dense/blas/rns_blas.rs`):
- Operands are aligned to one exponent and become exact integers. They are
  encoded modulo about 24-bit primes by a chunk × `2^(cj) mod p` `dgemm`.
- One exact `dgemm` runs per prime, the CRT is also a `dgemm`, and each
  output is rounded once (`MpFloat::from_scaled_integer`, direct RNE on the
  limbs).
- `gemm` returns the same correctly rounded exact dot as `exactdot`, so it is
  bit-identical to the per-entry path (test). The fused `congruence`
  A·X·Aᵀ keeps the intermediate exact and rounds once (test against a
  4×-precision exact reference). It changes low-order bits and is more
  accurate, as the exact-accumulation clause allows.
- Wired into the MPFR `gemm`, `pooled_gemm_sym` (upper only), the PSD
  condense/recover/apply congruences, the cone W products and the sampled
  forward `panel·Qᵀ`. Size gate: k ≥ 40 and m·n ≥ 1600 (break-even about 32³).

Pitfalls found on the cluster:
1. Per-call multi-MB allocations went through `mmap`, and the page faults
   serialized 32 threads (44 ms per call vs 6.9 ms alone). Fix: a per-thread
   buffer free list that is safe under re-entry, plus cached CRT and weight
   tables. After the fix: 9.0 ms at 32 threads vs 8.9 ms alone. OpenBLAS was
   not the cause.
2. A portable Rust f64 microkernel lost to BLAS (Mac Accelerate is 10× faster;
   cluster OpenBLAS 1–2×). Removed; native BLAS is exact for these integer
   products.
3. Splitting each block's product across primes inside the already
   block-parallel phases made 32 threads 16% slower. Only top-level calls
   (outside a rayon worker) split across primes now.

Also kept: sampled `materialize` on a setup pool of `max_threads` width, and
`wdiag` built on the cone pool. Both bit-identical.

Results. Cluster, Λ19, 768 bits, 8 iterations, IP-loop seconds vs the pooled-Ruiz
baseline (single runs, same node type):

| threads | base | new | change |
|---|---|---|---|
| 1 | 334.1 | 206.6 | −38% |
| 8 | 53.0 | 36.8 | −31% |
| 32 | 23.4 | 19.7 | −16% |

Setup at 32 threads: 4.9 → 3.7 s. Local full Λ19 solve (fused congruence, 4
threads): `Solved`, 119 it, audit passes with primal 4.8e-77, dual 6.3e-74,
gap 5.6e-75. ising11 `ab`: points identical, audits 2/2. medium: unchanged
known failure.

Full Λ19 solves on the cluster (768 bits, audit passes each time: primal
4.8e-77, dual 6.3e-74, gap 5.6e-75, reference 1.7e-42, mapping 5.7e-31):

| cores | SDPX before | SDPX now | SDPB | SDPB/SDPX now |
|---|---|---|---|---|
| 8 | — | 678.6 s wall, 119 it (job 220921) | 583.7 s wall, same node | 0.86 |
| 32 | 394.7 s, 119 it | 368.6 s, 119 it (job 220918) | 194 s, 243 it (2026-09-22) | 0.53 |

The 8-iteration windows overstate the full-solve gain (−16% at 32 threads
there, −7% end to end). Later iterations need more refinement: 7.3 outer
passes/it and about 36 inner reduced-system refinements/it. At 32 threads the
largest remaining main-thread items are `ir` 0.77 s/it (arrow triangular
solves scale only 3.2× from 1 to 32 threads) and scale cones 0.35 s/it
(per-block SVD critical path).

Deferred (step 3, SVD/eig blocking): at n ≈ 45 a blocked tridiagonalization's
trailing updates are 45×45×32, below the kernel's break-even. The SVD cost is
Givens application to V, not GEMM. No gain at Λ19 sizes without a different
algorithm, and the contract rules out eig-of-MᵀM and precision ladders.

## 2026-09-23 — Λ19 cluster scaling vs SDPB; one-sided SVD; pooled Ruiz

Cluster campaign `~/projects/sdpx-scaling-20260923-01` (UCAS HIAS, one 32-core
node, jobs 220810/220812/220830/220831). Λ19 input and reference from
`sdpx-sdpb-match-20260922-02`; 768 bits; internal 1e-42; NT (the `hkm` setting
no longer exists). SDPX pinned by `taskset`, SDPB via `mpirun --bind-to core`.
Single runs, preliminary.

8-iteration runs, native seconds including setup:

| cores | 1 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|
| SDPX threads | 378 | 117 | 71 | 46 | 38 |
| SDPB ranks | 133 | 46 | 23 | 14 | 10 |

Full solve on 32 cores: SDPX `Solved` in 119 it and 394.7 s. The external audit
passes: primal 4.8e-77, dual 6.3e-74, gap 5.6e-75, PSD 0, reference objective
1.7e-42, SDPB mapping 5.7e-31. The earlier HKM build took 456 s (125 it). SDPB
took 194 s (243 it, 2026-09-22). SDPX needs half the iterations, but each
iteration costs about 2.8× more on one core and about 4× more on 32.

Wall breakdown at 32 threads (`SDPX_PROFILE` timer tree, 8 it): setup 11.2 s
(serial), default start 3.9, kkt update 12.2, kkt solve 5.8, scale cones 1.9.
At 1 thread, the three RHS phases take 25 of 41.6 s/it: `prepare_rhs` 9.5,
`residual` 9.0 (scale 4.3, adj 2.2, fwd 2.4) and `recover_rhs` 6.6. That is 8
outer passes/it, each with three block congruences, plus about 24 inner
reduced-system refinements/it (`ir` 3.5 s/it; only 6.8× faster at 32
threads). Cone SVD/W/eig: 8 s/it thread-time. Past 16 threads the 28 blocks
cannot fill the pool. Receipt phase totals are summed over threads, so
cone_* phases that stay flat from 1 to 4 threads mean they scaled, not that
they stalled.

Kept (1): one-sided NT SVD (`SVDEngine::factor_right`, no U; Rinv from Vᵀ and
L1⁻¹). Λ11 A–B–B–A earlier this session, audit passes.

Kept (2): Ruiz `scale_data` rescales A on the cone pool over equal-nnz column
ranges (`lrscale_pooled`), with the same per-entry expression. Λ19 setup at
32 threads: 11.0/11.0 s → 4.9/4.9 s (cluster, 2 runs each). Mac, 4 threads:
22.4 → 13.0 s. Points bitwise identical on Λ19 and ising11. ising11 `ab`
at t4: 19.78 → 19.65 s, audits 2/2. Remaining serial setup: sampled
`materialize` (about 1.4 s on Mac) and `build_wdiag` (about 1.1 s). Both run
before the pool exists.

Refuted: the refinement passes do not come from regularization. On ising11
at 512 bits with static and dynamic regularization set to 1e-110 instead of
√eps, the solve still takes 50 it and ends `Solved`, but refinements rise
from 87 to 135 and linear solves from 250 to 292. Regularization is not the
lever. Diagnostic run only; defaults unchanged.

Clarabel.rs 0.11.1 (`sdp-accelerate,faer-sparse,serde`, qdldl, 1 thread) on
medium was stopped at iteration 13 after about 70 min (about 5.5 min/it).
Its sparse augmented KKT with dense PSD Hessian blocks is the cost; SDPX
solves medium in about 2.3 s. `sdp-accelerate` means the same in both crates:
Accelerate BLAS/LAPACK for dense cone kernels.

SDPB kernel review (source 67ebd53, `src/sdp_solve/SDP_Solver/run`). Per
iteration SDPB:
- builds the per-block Schur S_j from bilinear pairings (A_X_inv, A_Y),
  Cholesky-factors each block, forms P = L⁻¹B, and computes Q = PᵀP;
- computes Q with `bigint_syrk`: normalize P to fixed point, take residues
  modulo many small primes, run `cblas_dgemm`/`dsyrk` in double precision
  per prime, recombine with CRT (FLINT `mul_blas`), parallel over primes and
  windows across ranks;
- makes two direction solves (predictor and corrector) with no iterative
  refinement;
- computes step length from Cholesky(X), Cholesky(Y), a triangular
  congruence and a full Hermitian eig. There is no SVD.

In SDPX the Schur/Gram build is already cheap on Λ19 (`sync` 2.6 s/it at 1
thread). The time is in dense MPFR products done one element at a time
(congruences about 13 s/it, W products 2.8, Gram 2.6, sampled applies 4.6)
and in the sequential SVD/eig (5.2). SDPX's RNS uses scalar `u128`
accumulation with 60-bit primes, not BLAS.

## 2026-09-23 — module restructure and dev harness

- Solver source moved to `solver/{core,default,cones,kkt,sampled,distributed,chordal}`
  with per-module `tests/`; dead items removed; `distributed` compiled only
  with `sdp`. medium and ising11 outputs bitwise identical to `pre-restructure`.
- `benchmark/e2e/` added: pinned medium/ising11 inputs, committed Julia audit
  environment, `build`/`run`/`ab` commands. Replaces ad-hoc `/tmp` setups.

## 2026-09-22/23 — performance round (Apple M4, 4P+6E, single process)

Frozen binaries for this round were in `/tmp/sdpx-perf-20260922/` (not kept).
The text below is carried over unchanged from the plan's former §3–§4.


输入：`大规模矩阵/scripts/export_conic_json.py` 把 MOSEK export 转成原生 conic JSON
（medium n=1887、m=13739、nnz=138192，与 MOSEK metadata 一致；large n=7054、
m=42023）。设置 `settings1e6.json`（1e-6，与 MOSEK 记录同门槛），全部
`OPENBLAS/OMP/VECLIB/RAYON_NUM_THREADS=1`、`--threads 1`。

#### Float64 medium（MOSEK 1.89 s / 16 it）

| 臂 | native s（3 次中位） | it | 说明 |
|---|---:|---:|---|
| base（本轮工作树） | 5.44 | 18 | 旧 Julia 前端时代记录 7.98 s 不再可比 |
| + `pooled_gemm_sym` f64 走 BLAS | 4.23（−22%） | 18 | objective 逐位不变 |
| + PSD scaling 单次同构 G·X·G | 3.79（−3.5%，对前一臂同批 3.93） | 18 | objective、残差逐位不变 |

当前 medium ≈ 2.0× MOSEK。剩余相位（symblas 臂，solve 3.68 s）：assemble 2.20 s
（dot_scatter 1.12、transform 0.91、sparse 0.14）、cones_schur 0.52、refactor 0.46、
eigmin 0.15、ir+trsv 0.24。**Schur 装配占 60%**，其中 dot_scatter 实测约
1.4 GFLOP/s；当前实现按分段连续切片、8 项分组累加，transform GEMM 部分已接近 BLAS 峰值。

9 题开发集比较（LP/SOCP/SDP）两臂状态、迭代数全同；`SDP_arch0` objective 差 2e-10
（f64 BLAS 与标量 dot 舍入差），其余逐位一致。这是旧 development 结果，没有逐题附带
原问题审计，不能视作本计划的外部精度验收；large 未在本工作树复测（旧记录
129 s / 26 it，MOSEK 25.4 s）。

2026-09-23 当前 medium 快照（1887 变量、13725 约束，1 thread，1e-6）需与上述历史值
区分：未改动 release 基线与 Schur 四路 FMA 交错候选分别为 2.951 s 与 2.424 s
`api_seconds`（单次 −17.9%），外部墙钟却为 3.03 s 与 3.69 s，计时信号不一致。两臂
均 `AlmostSolved`、20 次迭代，x/s/z 逐位相同；原问题审计也相同但不通过：dual residual
2.85e-6、相对 gap 1.12e-6，超过 1e-6 门槛。此前 18 次 `Solved` 的历史输出也未通过
同一严格审计（dual residual 1.92e-6）。因此这项改动没有引入精度/状态变化，性能收益
暂记候选信号；当前 medium 的外部精度验收仍未通过，不能沿用旧记录称为已验收。

2026-09-23 补测：对同一 medium 分别运行 COSMO 0.8.11 与 Hypatia 0.10.2，求解器容差
均设为 1e-6，并使用相同的原坐标独立审计。COSMO 达到 5,000 次迭代上限（35.63 s；
外部原始 primal residual 2.05e-4、dual residual 8.49e-6、gap 2.71e-5）。Hypatia 在
31 次迭代后报告 `OPTIMAL`（130.85 s）；primal residual 7.18e-8、gap 8.36e-7 通过，
但 dual residual 2.154e-6 略超共同审计的 1.75e-6 门槛。两者都未通过该题的共同审计。
Hypatia 的 combined-stepper 可借鉴其共用 KKT 分解和多 RHS；SDPX 的 affine 与 combined
方向已经复用同一分解。它默认的稠密 QR-Cholesky，以及 COSMO 面向 ADMM 的 adaptive-rho /
Anderson 加速，不适合直接替换 SDPX 的稀疏 NT-IPM。

额外 Float64 诊断保留原曲线步和 0.99 最大步长：禁用曲线步得到更差的 18 次迭代结果
（dual 5.83e-6、gap 2.29e-6）；增加第三个更小曲线比例后结果明显变差（primal 3.05e-4）；
把 `max_step_fraction` 改成 0.995 或 0.98 均未通过（分别 dual 4.99e-6、gap 1.96e-6，
以及 primal 2.72e-3）。四个候选均已撤回。
基线日志显示第 19 次出现进度退化并恢复到前一可接受迭代，因此通用步长调参暂不作为下一项
优化。Schur 四路内核也尚未取得资格：有一次内部计时变快而墙钟变慢，暂不宣称有性能收益。

单 lane 的 transform→Schur dot 面板融合已按 E2E 结果撤回：候选保持 19 次迭代和与基线
相同的点及原坐标误差，native/API 为 3.123 s（基线 3.206 s），但 CLI 墙钟为 4.95 s
（基线 4.53 s）。没有可信的整题提速，且实现增加了专用代码，故恢复原有分块路径。
回退后的 fast-profile medium E2E 复现 `AlmostSolved`、19 次迭代、原坐标 dual residual
2.8486e-6 与 gap 1.1209e-6（审计门分别为 1.75e-6、1e-6）；状态和残差与基线一致，仍
未通过既有外部门槛，fast 与 release 的时间不作比较。

随后隔离两个可能改变 Schur 数值路径的实现。临时关闭结构零列压缩、改走完整 GEMM 后，
medium 仍是同一个 `AlmostSolved` 点和相同原坐标误差，故该分支不是当前精度偏差来源；
结构压缩路径已恢复。临时把 PSD `apply` 从合并后的 `G/Ginv` 两次 GEMM 改回因子 `R/Rinv`
的四次 GEMM，结果变为 18 次 `Solved`，gap 降至 7.54e-7，dual residual 降至 1.916e-6，
但仍超过 1.75e-6 外部门槛；API/native 增至 9.354 s，约为同配置单次变换的 3.27 倍。
因此恢复更快的合并变换。当前证据将优先级指向末端方向/poor-progress 轨迹，而不是 Schur
面板融合、结构零压缩或 componentwise 判据；后者只增加 full `Solved` 的额外条件，无法改变
当前 gap 未达 full tolerance 后的 poor-progress 回退。

#### MPFR Ising Λ11，512-bit，内部 1e-42（50 it，接受）

| 线程 | native s | kkt update | kkt solve | scale cones | step len |
|---:|---:|---:|---:|---:|---:|
| 1 | 44.6 | 14.2 | 7.3 | 11.2 | 7.6 |
| 8 | 8.3（5.4×） | 2.7 | 1.5 | 1.75 | 1.2 |

注：上表在低电量降频下测得；同臂接电复测 t1 = 22.33 s（约 2× 差距，跨电源状态
的绝对时间不可比，相位占比仍有效）。接电同批 A/B：

| 臂 | native s | it | 说明 |
|---|---:|---:|---|
| base（`mpfr_fma` 链 dot） | 22.33 | 50 | objective、receipt 残差逐位一致 |
| `exactdot` 定点精确累加 | 17.47（−21.8%） | 50 | 末端一次舍入；单批 E2E 初测，非稳定性能结论 |

在此基础上，MPFR SVD Givens 两分量改用两项 `exactdot` 后，512-bit Ising 同配置
release E2E 为 API/native 17.219/17.219 s，对照原旋转路径 17.730/17.728 s，单次约
−2.9%；两者均 `Solved`、50 次迭代，独立原问题审计 `accepted=true`。外部墙钟计时
方向相反，故只记为初步信号，不外推稳定收益。

2026-09-23，PSD 最小特征值路径把 `svec_to_mat` 后的对称矩阵缩放改为只算上三角并
镜像结果，避免重复 MPFR 乘法。Λ11 512-bit、单线程、内部 `1e-42` 的 release E2E
为 15.953 s、50 次迭代；独立原问题审计 `accepted=true`，原坐标 primal/dual/gap 均
小于 `1e-30`，峰值 RSS 104.8 MiB。相较此前 17.219 s 记录约快 7.4%；这是不同单次
计时之间的初步信号，尚不能排除机器状态波动。

改后采样归属：exactdot 31% inclusive，leaf 以 `__gmpn_addmul_1` 36% 为主（已受
原始乘法吞吐约束）；其次 `svd_rotate` 16%、`eig` 三对角化 7%、逐元素 mul_add 14%。

t1 相位累计：cone_svd 10.0 s（22%）、residual 7.8（scale 4.3、fwd 1.9、adj 1.6）、
cone_eigmin 4.5、recover_rhs 4.0、prepare_rhs 3.6、cone_wprod 3.3、sync 1.7、
cones_schur 1.4、refactor 1.4、ir 1.0、assemble 0.27。**MPFR 时间集中在锥（SVD/eig/W 乘积）
和 RHS/残差中的块同构，而不是 Schur 分解。** G·X·G 改动对 Ising 无变化
（43.6 s vs 43.6 s；residual.scale 未降，原因待查：sampled 块走 fused Condense/Recover，
普通 apply 路径不是瓶颈的假设尚未验证）。

#### MPFR 标量内核微基准（单核，每项 ns，k=8/32/128 基本不变）

| 位数 | `mpfr_fma` 链 | `mpn_mul_n`+移位+`mpn_add` 精确累加 | 比值 |
|---:|---:|---:|---:|
| 256 | 64–74 | 23–26 | 2.5–3.1× |
| 512 | 99–114 | 54–55 | 1.8–2.1× |
| 768 | 154–172 | 102–103 | 1.5–1.7× |
| 1024 | 223–241 | 172–174 | 1.3–1.4× |

上界估计未含真实指数对齐分支与末端一次舍入。内核已按 `mpn_mul_n` 全积、定点累加、
一次 `mpfr_set_z_2exp` 实现（`exactdot.rs`）并接入 `MpFloat::dot_fma`；Ising Λ11
单批 E2E 初测为 −21.8%，后续性能结论以固定精度完整求解和原问题残差为准。

### 当时的问题清单（收益 × 风险排序，含证据）

| # | 问题 | 证据 | 优化方向 | 风险 |
|---|---|---|---|---|
| 1 | f64 Schur dot_scatter 仍是 medium 主要热点 | 冻结 E2E profile：1.12 s / 3.7 s；按结构模式重排等 nnz 列的候选由 2.389 s 增至 2.519 s；新增稀疏左列等价复用的 A–B–B–A 为 2.338 s 对 2.324 s；两项均未提速，已撤回 | 保留现有四路 FMA 与分块实现；下一步应减少大块矩阵级工作，而非增加别名查找 | 中：medium 当前外部 1e-6 审计未通过，任何结果均标记为未验收 |
| 2 | 真正稠密的 f64 PSD 块可考虑平方根 Hessian 与 SYRK | Hypatia 的变换后 Gram 恒等式适用；当前 medium 五块的系数密度只有 0.386–0.710%，全量 SYRK 运算量远高于现有稀疏点积 | 仅在后续真实稠密 SDP 输入中评估全稠密块分支；不向当前 medium 强行引入 Gram 路径 | 中：额外内存及求和顺序变化需由完整求解评估 |
| 3 | MPFR SVD 的 Givens rotate 仍逐元素执行多次标量运算 | Ising Λ11 profile 中 svd_rotate 约 16% inclusive；两项 exactdot 候选已通过 Ising E2E 与外部精度审计，单次约 −2.9% | 保留候选；若继续做 MPFR E2E 优化，再试 WY/GEMM 或特征值内核 | 低到中：当前收益仍是单次测量 |
| 4 | MPFR SVD/eig/W 乘积仍占高精度主耗时 | 当前 Ising Λ11 完整求解阶段记录：SVD 4.85 s、eigmin 2.28 s、W 乘积 0.95 s / 总 16.19 s；Jordan 对称化与二项 `fmma` 的组合在同机 A–B–B–A 中为 16.218 s 对 16.459 s，50 次迭代、外部审计均通过；eig Householder 对称更新在新的配对中为 15.933 s 对 16.197 s，原问题审计通过。SVD 2 的幂归一化、eig 两项 `fmma`、SVD 常量缓存、Householder 分量除法倒数化均无收益并已撤回。整块缩放共用倒数初版在 t1/t4 快 0.60%/0.85%，补齐极端指数回退后 t4 反慢 0.66%，亦撤回。随后把 eig 三对角化的行内积改用现有精确累加接口，Λ11 512-bit A–B–B–A 在 t1 为 15.765 对 15.911 s（快 0.92%），t4 为 5.203 对 5.222 s（快 0.37%）；全部 50 次迭代、`Solved`，独立原问题审计通过 | 保留行内积精确累加；停止针对单一除法做小幅替换，优先找 SVD/eig 矩阵级冗余，再评估 WY/GEMM 或结构化 PSD Schur/W 乘积 | 中：配对收益小于 1%，尚不能外推跨机幅度 |
| 5 | RHS/残差路径重复做块同构与多波次求解 | 当前 Ising Λ11 阶段记录 prepare/recover/residual 分别为 1.06/1.21/1.73 s；solve_many 已存在；跳过 fused sampled recovery 的普通线性 GEMV 在同题配对 E2E 无可信收益，已撤回（其 CSC 本来不含采样 PSD 行） | 合并真正可共享的 RHS，并复用当前迭代的缩放乘积；优先审阅原算子剩余计算量 | 中：依赖关系需保持，按完整求解耗时评估 |
| 6 | condensed 外层和 Schur 内层存在精化工作重叠 | 静态调用审阅确认外层 correction 再调用带内层 IR 的 reduced solve；Λ11 512-bit 的 163 次 reduced solve 中仅 10 次来自外层 correction，内层累计 87 次修正（0.73 s），所以只在外层 correction 跳过内层 IR 的上限很小，且两层分别检查不同算子 | 暂不取消内层 IR；若 Float64 可验收工作负载显示其占比显著，再用独立候选检验 | 中：不能仅凭调用嵌套认定精化冗余，必须保留原问题精度 |
| 7 | BLAS 线程不完全纳入求解器线程预算 | settings.threads 主要控制锥池 | 明确线程预算，避免求解器池与 BLAS 过量订阅 | 中：仅在多线程 E2E 对比中验收 |
| 8 | 两条 MPI 路径并存，普通路径有逐 site gather | 既有跨节点记录有负扩展；owner-partitioned 已覆盖对称 LP/SOCP/SDP，但 CLI 无 `--partitions` 时和 C ABI 仍走普通路径，owner 路径尚不支持非对称锥或 augmented KKT，真实 MPI E2E 尚未合格 | 先补实际 MPI E2E 与默认路由/回退，再收敛普通 gather；不能仅凭 Ising mock 测试删掉通用路径。2026-09-23 复核：owner 路径仍拒绝非对称锥与 augmented KKT，本机无 MPI 运行时，故普通路径保留 | 高：受 runner 和集群资源支持限制 |
| 9 | dispatch 成本阈值散落且与机器相关 | rate、填充、Arrow、内存等常数来自启发式 | 先维持当前值；仅在目标完整求解明确受其影响时调整 | 低 |
| 10 | 代码规模与大文件职责重叠 | 当前约 41.5k 有效代码行；condensed_psd、sampled、compositecone 较大；内部未使用的 LU 与特征向量包装累计净减 211 个物理行 | 继续按生产调用证据去重，目标生产约 30k 有效代码行。2026-09-23 已完成目录重组（`solver/{core,default,cones,kkt,sampled,distributed,chordal}`，测试移入各模块 `tests/`，`distributed` 仅在 `sdp` 下编译）并删除库与测试均未使用的项；medium 与 Ising Λ11 E2E 输出逐位不变。行数基本未变，真正减量仍需去重 | 中 |
| 11 | release 构建较慢、debug 产物占用大 | 本轮记录：release 构建约 6 min，target/debug 约 59 GB | 迭代使用 fast profile；清理产物不作为性能里程碑验收。2026-09-23 已清理 target/debug（target 83 GB → 7 GB） | 低 |

## 2026-09-27 — Project review fixes

Kept the six fixes identified by the whole-project review:

- Native MPFR descriptor import is now an `unsafe` API with a documented
  storage-validity contract.
- Sampled solver export returns `Unsupported` before writing, avoiding loss
  of authoritative factors. Retain and serialize the original `JsonProblem`.
- Indexed matrix/vector updates validate all indices before mutation; paired
  index/value vectors must have equal lengths.
- QDLDL honors `dynamic_regularization_enable`; a zero pivot returns a failed
  factorization instead of panicking.
- Dynamic regularization settings are immutable after solver construction,
  preventing accepted updates that leave cached backend settings unchanged.
- Backtracking requires a finite factor strictly between zero and one, and
  cone searches stop if the step cannot decrease or underflows to zero.

Removed the README benchmark graphic, performance table, and section link at
the user's request. No numerical acceptance gates or precision changed.

Verification: rebuilt frozen fast arm `review-fixes-20260927`. Targeted
standalone reproductions passed for atomic failed updates, setting validation,
sampled export rejection and exact original-factor serialization. Safe Rust
descriptor import fails compilation with E0133. The disabled-regularization
reproduction now solves; a zero-pivot case returns `NumericalError` without a
panic. Invalid backtracking exits with a settings error; the valid exponential
cone input still solves. Reproduction sources and logs are under
`~/.cache/sdpx-e2e/review-20260927/`.

Pinned solves ran sequentially against unchanged settings and precision:

- `medium`: `Solved`, 18 iterations; existing external audit failure remains
  (`r_d=1.92e-6 > 1.75e-6`). Run `20260927-195346-663200-medium-review-fixes-20260927`.
- `ising11`: `Solved`, 52 iterations; external audit passes
  (primal `2.34e-45`, dual `4.29e-44`, gap `1.70e-43`). Run
  `20260927-195357-404136-ising11-review-fixes-20260927`.

Both cases have identical returned `x`, `s`, and `z` to frozen baseline
`review-20260927`. These fast-profile checks make no performance claim; no
headline status or benchmark number changed.

## 2026-09-27 — CSDR alpha-count 3 on the Mac

At the user's request, reconstructed the historical twice-subtracted CSDR
case from the surviving input generator: J=40, N_mu=200, N_a=15, N_x=1,
three alpha labels (0, -1/4, -1/2), power-6 quadrature; 8,400 spectral
variables, 42 equalities and 4,200 SOC3 cones. Cached Julia SDPX `db42fd2`
was used only by the input generator. BigFloat256 coefficients were exported
as decimal strings, without a Float64 intermediate.

On this Apple M4 / 16 GiB Mac, frozen release arm `csdr-current-20260927`
(current reviewed Rust changes) with MPFR256, four requested threads,
500-iteration/600-second limits and feasibility/absolute-gap/relative-gap
tolerances of `1e-8` returned `Solved` in 57 iterations. One preliminary run:
API/setup+solve 120.851 s, native 120.845 s, process wall 121.187 s.
The selected augmented QDLDL factorization used one thread; cone work used
four. No solver code or accuracy gate was changed for this experiment.

Independent BigFloat256 original-coordinate audit passed: primal residual
`1.24e-12`, dual `6.45e-63`, relative gap `3.79e-9`, and reconstructed physical
equality residual `6.32e-11`. Primal SOC margin was `-9.56e-13`, within the
existing `1e-8` gate; dual SOC margin was positive. Objective:
`-31.67385581061044`.

The historical Julia Float64x4 receipt reports a warmed median of 17.532 s,
101 iterations and objective `-31.672155970636577`. The raw time ratio is
6.89, with Rust slower in this run. This is not a validated like-for-like
performance comparison: the deleted frozen input cannot be hash-matched,
the objective differs by 0.00169984, arithmetic differs, and Julia was not
retimed. Repeated timings would not resolve those comparability limits.

Input, generator, audit, source hashes, complete logs and comparison receipt:
`~/.cache/sdpx-e2e/csdr-alpha3-20260927/` (`comparison.json`). Input JSON SHA256:
`a2c6b822aac133c4cd8ce734c4c2e09bedb218be8fa31c91305c10fa619302a1`.

## 2026-09-27 — Local SOC elimination and parallel equality Schur solve

Kept a structural `local_soc_arrow` backend beneath the existing augmented
KKT interface. It recognizes disjoint SOC3 blocks with two primal variables
each, block-local P, and an equality border. The current eligibility guard
requires at least eight local blocks, 1–128 equality rows and a bounded dense
working set. Selection is automatic; explicitly requesting `qdldl` preserves
the baseline route. Other cone structures retain their existing backend.

Reused the arrow implementation for batched RHS, local-factor reuse, parallel
elimination/recovery, and QDLDL fallback. Each five-coordinate leaf eliminates
its three negative cone coordinates before its two positive primal coordinates;
the global CSDR border is only 42 by 42. Local Schur entries use exact MPFR
dot accumulation rounded once, with fixed term order at every thread count.
There is no dense border-contribution buffer per SOC. Original augmented
residual refinement, static shifts, ×100 escalation, dynamic pivot shifts,
HSD recovery, convergence and infeasibility criteria remain in force.

On the same reconstructed CSDR input as the preceding entry, one serial
release A–B–B–A batch on the M4 Mac gave:

| Arm | API/setup+solve seconds | Median | Outcome |
|---|---|---|---|
| `csdr-current-20260927` (QDLDL) | 121.088, 121.214 | 121.151 s | Both Solved, 57 iterations, external audit pass |
| `csdr-local-soc-release-20260927` | 19.638, 19.299 | 19.468 s | Both Solved, 57 iterations, external audit pass |

Measured speedup: 6.223× (83.93% less time), at unchanged MPFR256, four-thread
budget, tolerances, input and preprocessing. Repeated points within each arm
are bitwise identical. The candidate reports four backend threads. Peak RSS
was approximately 369–375 MiB versus 280 MiB for QDLDL: extra block storage is
the memory tradeoff. Refinement corrections fell from 1,055 to 171 without
relaxing tolerances. The remaining linear-solver work is primarily local Schur
assembly and RHS coupling; the small border solve is negligible.

Candidate audit: primal `9.77e-13`, dual `2.86e-70`, relative gap `3.37e-9`,
physical equality residual `3.74e-62`; primal/dual SOC checks pass. The changed
elimination order changes the accepted point (objective about
`-31.67370757755`), so no cross-backend bitwise claim is made. The historical
Julia 17.532 s receipt remains an unmatched comparison, for the reasons above.

Checks: CSDR complete solves at one and four threads return bitwise-identical
`x/s/z`, objectives, status and iteration count. Pinned fast A–B–B–A checks
against `review-fixes-20260927` preserve points bitwise on both `ising11`
(all audits pass, 52 iterations) and `medium` (the same documented external
dual-residual failure, 18 iterations). Rustfmt checks on edited files and
`git diff --check` pass. MPI execution was not exercised on this Mac.

Evidence: `~/.cache/sdpx-e2e/csdr-alpha3-20260927/local-soc-release-abba/summary.json`,
`thread-parity.json`, `local-soc-pinned-checks.json`, and `local-soc-source/`.
Frozen arms also retain the new untracked source module under `source-extra/`,
because the harness's ordinary dirty patch includes tracked files only.

## 2026-09-27 — Development documentation cleanup

Rewrote the active plan in concise English, corrected current pinned-case
status, and separated historical cluster measurements from local evidence.
Ranked CSDR follow-ups: reproducible harness inputs/source identity, accounting
for time outside the solve phase, compact SOC panels/Schur assembly, and RHS
coupling. These are proposals, not new speedup claims. Shortened the development
skill and added a concise-English documentation rule to AGENTS.md. Historical
entries remain intact. Documentation links, English-only checks on the active
development files, and `git diff --check` pass; no solver rerun was needed.

## 2026-09-27 — Compact local SOC panels, reproducible CSDR checks and Ising profile

Kept the four CSDR follow-ups: registered reconstructed `csdr3` in the ignored
local E2E harness; frozen-arm manifests and copies now cover untracked source;
receipts retain inclusive setup timings; local SOC B/Y/Z panels store two
coordinates instead of five. RHS coupling uses one exact MPFR accumulation per
output, rounded once. Static/dynamic regularization, precision, tolerances,
accepted-iterate recovery and original augmented refinement are unchanged.

Release A–B–B–A on the M4, MPFR256, four threads, unchanged 1e-8 gates:

| Measure | Previous local SOC | Compact local SOC |
|---|---:|---:|
| Median API time | 20.026872 s | 15.956056 s |
| Median peak RSS | 370.0 MiB | 291.9 MiB |
| Schur assembly | 4.928 s | 3.155 s |
| Recorded RHS coupling | 2.009 s | 0.714 s |

One comparison batch: 20.3% less API time. All four points are `Solved`/57 and
pass the original-coordinate audit. Both arms retain 58 factor attempts and
171 refinements. Points repeat within each arm; candidate one/four-thread and
fast/release points are identical. The exact RHS accumulation changes old/new
low-order digits (maximum componentwise scaled distance below 5e-51), with
primal 9.77e-13, dual 2.86e-70 and gap 3.37e-9. The reconstructed input hash stays
`a2c6b822aac133c4cd8ce734c4c2e09bedb218be8fa31c91305c10fa619302a1`;
the missing historical Julia input still prevents a matched Julia comparison.

Setup now accounts for 6.153 s: presolve 5.991 s, equilibration 0.149 s, KKT
construction 0.011 s. The presolver performs exact rational equality elimination;
a conservative modular proof of full row rank is a future candidate, with exact
elimination retained whenever the proof is inconclusive. Setup timers are
inclusive and separate from solve timers; phase totals must not be added.

Pinned fast A–B–B–A checks preserve all medium and Ising11 points. Medium remains
`Solved`/18 with its known external dual-residual failure (1.92e-6 > 1.75e-6).
Ising11 remains `Solved`/52 and passes every original-coordinate audit. Its fast
profile was 13.3% slower, so release timing was checked before acceptance:
A=13.446812/13.435114 s; B=13.377951/14.381593 s, medians 13.440963/13.879772 s
(+3.3%). A separate diagnostic candidate run was 13.451 s. Preserve this variation;
no Ising kernel speedup is claimed and the confirmation is excluded from ABBA.

A single preliminary Ising11 release solve with four threads took 4.301 s and
returned the same point as one thread. One-thread phase medians identify direct
SVD (3.502 s inside 4.043 s cone scaling), sampled residual work (1.948 s), RHS
preparation (1.213 s) and recovery (1.368 s) as useful next targets. These timers
overlap. Four-thread cone scaling wall time is 1.204 s; summed per-cone SVD time
is not wall time. The plan prioritizes SVD subphase profiling, sampled cache and
application reuse, then scheduling on larger Ising inputs. No cluster work ran.

Harness checks verified untracked-source capture, rejection of source edits
during compilation, recovery from a corrupted input cache and rejection of an
auditor's nonzero exit even when an audit JSON exists. Rust formatting and
`git diff --check` pass. Active documentation remains concise and English.
Evidence: `~/.cache/sdpx-e2e/improve-20260927/summary.json`, logs and run paths;
frozen candidate `compact-soc-release-20260927` (binary SHA256
`b2ef4379f1a159000226ea378d5317191d3ed5b478b95f1c0e13e778a9e99d20`).

## 2026-09-27 — Ising SVD, sampled RHS and PSD scheduling

Kept three local Ising improvements:

- Direct SVD caches fixed QR constants and records only the requested U/V
  rotations in separate reusable buffers. QR, replay, bidiagonalization and
  reflector timings are now separate receipt phases. SVD arithmetic order and
  direct factorization are preserved; no Gram eigendecomposition is introduced.
- Scalar sampled RHS adjoints use cached exact quadratic forms at MPFR256 and
  above for eligible shapes (side >= 12, at least 16 unique basis columns).
  Packing the symmetric RHS with off-diagonal scale 2 avoids svec rounding;
  the residue kernel rounds once at the destination. Basis factors remain
  authoritative, cache validation checks their exact values, and unsupported
  shapes/exponent ranges retain the existing product path. The fallback reads
  the exactly symmetric RHS transposed for contiguous dot operands.
- Local MPFR PSD scaling offers the largest cones first through a shared work
  queue. Every task owns one cone, output order is preserved, all tasks finish
  before reporting a failure, and the existing pool/thread budget is reused.

A separate RHS panel-clear shortcut showed no useful complete-solve benefit and
was reverted. No precision, stopping criteria, regularization, refinement or
original-coordinate audit gate changed. No tests were removed.

Release A–B–B–A on the M4, pinned Ising11, MPFR512 and unchanged settings:

| Threads | Previous compact-SOC arm | Ising candidate | API reduction |
|---|---:|---:|---:|
| 1 | 13.395325 s | 13.208873 s | 1.4% |
| 8 | 3.238693 s | 3.149282 s | 2.8% |

One batch per width; the gains are modest. A single four-thread candidate run
was 4.070 s (preliminary). All nine release Ising solves passed the original-
coordinate audit at `Solved`/52, with 53 factor attempts and 90 refinements.
Candidate outputs repeat bitwise and are identical at one/four/eight threads.
The exact quadratic calculation changes old/new low-order digits: maximum
componentwise scaled distance below 6e-87. External primal/dual/gap remain
2.34e-45 / 4.29e-44 / 1.70e-43, with SDPB-reference agreement 4.62e-35.

One-thread phase medians: RHS preparation 1.125960 → 0.944683 s; total SVD
3.488466 → 3.456315 s. Candidate SVD subphases: bidiagonalization 0.349710 s,
QR 1.511023 s, replay 1.366551 s, reflectors 0.187023 s. Recovery remains
1.284668 s. Phase timers overlap, and per-cone totals are not threaded wall time.

The release medium A–B–B–A check returns identical points and the unchanged
known dual-residual audit failure (`Solved`/18; 1.92e-6 > 1.75e-6). Its median
API times are 2.126702/2.125896 s. Formatting and `git diff --check` pass.

Only Lambda11 is available locally. Larger Lambda19/spins 0–50 inputs were
requested; no cluster work ran and no larger-case performance claim is made.
Evidence: `~/.cache/sdpx-e2e/ising-opt-20260927/summary.json`, phase receipts,
rounding-distance report and logs. Frozen candidate `ising-opt-release-20260927`:
binary SHA256 `84bca7f66c81a9be425d60eba7a693a2deb644715e915ba16ef67abd5030688c`;
source manifest `0361d4e83c254b87b18699341ae623f9bbbd14814ff1036cd38d4e9f02fc9c6e`.


## 2026-09-27 — Official SDPB pilot and reconstructed Lambda43 input correction

Built official SDPB 3.1.0 (tag commit
`fec8e934bf03eb59b0f35ad76dd9b205dde537e6`) and the current SDPX release in
`hpc:~/projects/sdpx-ising43-20260927`. A same-node, four-physical-core
Lambda19 single-correlator pilot at 768 bits passed both original-coordinate
audits. One run per solver: SDPB 1101.777 s wall / 243 iterations /
445.36 MiB sampled peak PSS; SDPX 451.558 s / 119 iterations / 1524.34 MiB.
These are preliminary release measurements, not the requested Lambda43 result.
The SDPB solve/audit completed before its wrapper hit a process-exit memory
sampling race; the corrected SDPX retry skipped vanished process samples.
Raw outputs were retained, and both audited points remain accepted.

The 2019 paper's input is unavailable. The campaign reconstructs one fixed
3D mixed-correlator model at Lambda43, 1265 components, 117 blocks and 1216
bits. Review of arXiv:1603.04436 Eq. (2.5) found a missing cos²(theta) weight
on the isolated odd contribution. The first generation is excluded and
preserved; its converter and calibration are user-held. The v2 driver fixes
that coefficient and stamps the definition in metadata. Corrected generation
221696 follows the old job, conversion 221697 follows generation, and bounded
four-core calibration 221699 waits for conversion and release build 221698.
The 16/64-core scripts are prepared but unsubmitted; 256-core execution and
complete target audits remain pending. Three-iteration calibration does not
count as a successful numerical solve.

Evidence: `~/.cache/sdpx-e2e/ising43-cluster-20260927/pilot-evidence`,
`precision-grid-campaign.json`, and `generator/RECONSTRUCTION-v2.md`.
Old jobs 221692/221693/221695 and all their outputs remain preserved.


## 2026-09-27 — Exact MPFR precision grid in the CLI and C ABI

Kept support for every 64-bit increment from 128 through 2048 bits, including
1216. Binary64 remains 53 bits. Both frontends expand one shared registry,
so their supported types cannot drift. The Rust API retains `MpFloat<N>`
(`64*N` bits) and adds the convenience alias `Bits1216`. No arithmetic,
precision rounding, tolerances, convergence or refinement rules changed.
Unsupported widths, including 1000, still fail explicitly.

Verification: locked offline compiler check passed for the solver and C ABI;
the pinned Ising11 solve is `Solved`/52 and passes its existing audit. Medium
is `Solved`/18 and retains its documented dual residual 1.92e-6 > 1.75e-6
failure. Both solutions' x/s/z arrays are bitwise identical to the preceding
`ising-opt-release-20260927` results. The existing C ABI sampled-PSD solve
passed at 53, 512 and newly added 1216 bits, including reported working
precision and solved objective checks. Rustfmt and `git diff --check` pass.
The first exact test filter selected zero tests; rerunning the fully qualified
name executed and passed the intended test. No timing claim uses this fast arm.

Frozen arm: `precision-grid-20260927`; binary `d86a3bfeb06ec6dd36c09a5a960bc167c3f6f794d0909e74cf05139d4e40216f`;
source manifest `b65d94406d6558df0effecd29bc80e851464ee5120e5df397d5209b2dfa4f414`. Run records and summary are in
`~/.cache/sdpx-e2e/ising43-cluster-20260927/precision-validation.json`.
Cluster source archive SHA256:
`de8ef119a97e60c3c04d9a53d1f42f590b302c15b810fbe141c6a6fa6cdbd5fa`.
Release job 221698 builds into a new target directory; the previous binary
remains preserved. Its dependent calibration requires exact 1216-bit output
and accepts exit code 2 only with `MaxIterations`, which is a calibration
termination, never numerical success.


## 2026-09-27 — Lambda19 memory and runtime diagnosis

A diagnostic replay on node9 (job 221700, four physical cores, frozen release,
768 bits and unchanged settings) completed `Solved`/119, passed the external
audit and reproduced the earlier pilot's x/s/z bitwise. No numerical code
changed. Profiled aggregate peak PSS was 1538.89 MiB; retain the uninstrumented
1524.34 MiB and 451.558 s as the preliminary comparison row.

Cumulative process peak RSS by stage: input 31 MiB; sampled materialization
322; equilibration 432; KKT construction 751; first iteration 1283; second
iteration 1375; solved 1529. These high-water marks do not constitute a heap
allocation census. The retained A accounts for 274.44 MiB of allocated value
and index payload. Generic PSD axpy plans retain another 53.39 MiB of capacity
although the installed sampled path does not use them. Other live storage
includes scaling/Gram/recovery panels and residue caches. RNS idle scratch is
capped at 32 MiB per participating thread, while its process-global weight
and CRT table caches have no eviction policy; their actual share of this
run's peak is not yet measured.

Top-level solve phases in this instrumented release: KKT update 194.49 s
(43.4%), cone scaling 147.10 s (32.8%), KKT solve 48.27 s (10.8%). Setup is
3.45 s. SVD rotation replay is 62.9% of summed per-cone SVD time; per-cone
observations overlap across workers. Arrow refactor alone is 33.12 s.

Prioritize removing unused sampled metadata and retained expanded A for RAM,
then budget/share residue caches without discarding useful hot encodings.
For speed, investigate direct-SVD rotation replay and the full KKT update
path. Preserve the existing rounding, refinement and convergence rules.
No optimization or savings claim follows from this diagnosis alone.
Evidence: `~/.cache/sdpx-e2e/ising43-cluster-20260927/memory-profile/report.md`,
`summary.json`, raw receipt, memory trace and original-coordinate audit.


## 2026-09-28 — Sampled memory candidate and Rust PMP converter

Implemented in the shared checkout (uncommitted):

- Avoid generic PSD coefficient/AXPY plans for fully sampled KKT blocks;
  release those plans when installing partially sampled blocks too.
- Release `DefaultProblemData.A`'s expanded sampled coefficients after KKT
  construction. Preserve the full matrix norm/nnz for initialization and
  reporting. Its public `A` now contains the explicit linear component;
  `materialize_A()` reconstructs the equilibrated matrix on demand. Prepared
  preprocessing and distributed construction still receive the full matrix.
- Share immutable basis residue encodings across operator workspaces.
  Idle scratch has a process-wide 128 MiB cap and a 32 MiB per-thread cap;
  weight and CRT table caches each retain at most 64 MiB. Active kernel
  buffers and input-dependent caches are outside these retained-storage caps.
- Replay direct-SVD rotations in four-row tiles, reusing MPFR descriptors.
  Each destination still uses the same correctly rounded two-product sum.

Frozen fast arm `sampled-memory-fast-20260928`, source
`2205d2e21c0788c40a8ff39ae216099060d2c6bd08f601c1566cc797148c494e`,
binary `60e6beae0ee96f4df70b75cab2c01cac015b37cbb1fd99f0e3fd066487784a68`.
Ising11 A–B–B–A against `precision-grid-20260927`: all four points identical,
`Solved`/52, all original-coordinate audits pass. Medium: identical points,
`Solved`/18, existing dual-residual failure unchanged. Ising one/four-thread
points are identical; four-thread audit passed. Fast timings are development
checks, not release performance claims.

Cluster source archive SHA256
`fbbc9f2b8da5cfc5767762414043f936e401dbee316cf506229245687233b4f5`.
Release build job `221703.node220`; node9 four-core Lambda19 comparison job
`221704.node220` depends on that build. The driver runs A–B–B–A at MPFR768
with unchanged settings, audits outside the timer, and simultaneous process
PSS sampling. Existing binaries and pilot outputs are preserved. Submission
manifest: `~/.cache/sdpx-e2e/ising43-cluster-20260927/sampled-memory-v2-campaign.json`.

Added `sdpx-pmp`, a standalone Rust PMP conversion library and CLI. Adapted
SDPB 3.1.0's normalization, density sampling, moment basis and sampled block
format, retaining MIT provenance. Inputs are JSON or legacy XML; decimal
strings go straight to MPFR. General square polynomial matrices, both basis
parities, reduced prefactors, supplied sampling data, and automatic sampling
are supported. Output is uncompressed SDPB JSON, written block by block.
Mathematica, NSV, binary/ZIP output and MPI conversion are not implemented.

Initial converter checks (MPFR768 unless stated): supplied, generated, XML,
repeated-pole and isolated-zero upstream inputs converted; scalar outputs
match their reference files (largest scaled differences below 1e-190 for the
four reference comparisons). The upstream 2×2 XML coefficients agree below
3e-231. Nontrivial normalization agrees below 6e-77 against an upstream
512-bit reference (whose sampling stops at half precision). Three scalar
solves and the upstream 2×2 solve pass independent original-coordinate
1e-30 audits at unchanged 1e-42 solver tolerances; scalar objective also
agrees with `(sqrt(145)-1)/6`. All 31 CLI MPFR choices pass a constant PMP
conversion. Malformed shapes, normalization, samples and nonfinite inputs
are rejected, and existing output files are preserved.
Evidence: `~/.cache/sdpx-e2e/pmp-rust-20260928/`.

A separately constructed normalized 2×2 diagnostic returns `AlmostSolved`
after 24 iterations in both old and new binaries, with bitwise-identical
points. It is not accepted or promoted. Input SHA256 `b5dac5b0848b35f5fa47cdfaa52a57e1416a91817f3be872c247d75439dc1f81`;
analytical optimum 0.75; MPFR512, unchanged 1e-42 tolerances. This newly
observed baseline limitation does not invalidate the upstream converter
comparisons or the pinned solves.

Integration checking also exposed an existing no-SDP build failure from
unconditional split-hint calls; added a no-SDP passthrough. Five standalone
provider test harnesses lacked the public receipt import after earlier SVD
instrumentation; imports are repaired. Workspace release tests are pending.

Final standalone converter source was rebuilt with the workspace lockfile in
`pmp-rust-20260928/target-final`. JSON reading now avoids a whole-file text
copy; XML concatenates comment/CDATA-split scalar text, and supplied bases
accept trailing zero padding. All reference comparisons were repeated;
the three audited scalar SDP payloads are unchanged. Final binary/source
hashes and additional XML/padding checks are in `validation-v2/manifest.json`.

## 2026-09-28 — Move remaining integration work to the cluster

At the user's request, stopped the local release workspace compilation and
its two remaining rustc children. That interrupted run is not a test pass.
Large builds, full-suite checks and substantial numerical runs now use PBS;
local work is limited to editing, inspection and lightweight preparation.

Submitted `221705.node220` (8 cores, 64 GiB, four-hour limit), dependent on
completion of the Lambda19 comparison `221704.node220`. It builds and checks
the Rust converter against six upstream fixtures, exercises all 31 MPFR
choices and malformed-input rejection, solves four converted inputs and
runs independent original-coordinate audits. It then runs the release
workspace tests with the cluster's dynamic OpenBLAS provider and faer.
These checks are queued, not yet passed. The dependency keeps compilation
from interfering with the time/memory comparison.

Source hashes and the offline dependency closure were verified before PBS
submission. Archive SHA256:
`2fd7b083017ccbf4abf71b21adc1763e12a2e4a7383a048382d82d66c15814df`.
Local preparation: `~/.cache/sdpx-e2e/cluster-integration-20260928-v1/`.
Remote evidence: `~/projects/sdpx-ising43-20260927/integration-20260928-v1/`.
Existing remote sources, binaries, jobs and results are preserved.

## 2026-09-28 — Lambda19 memory qualification and remaining cluster checks

Kept sampled memory and SVD replay changes after release A–B–B–A job
`221704.node220` completed successfully. MPFR768, four physical cores on
node9, unchanged input and tolerances. All four solves are `Solved`/119 and
pass the original-coordinate audit; x/s/z are bitwise identical. Canonical
point SHA256: `5d7108b5a6066ca96103235f04ec3ae20c7d9d7735c32d46a118bd7a8255446f`.

| Arm | Wall seconds | Peak PSS MiB |
|---|---:|---:|
| A1 | 451.460906 | 1527.8730 |
| B1 | 448.820576 | 1112.7305 |
| B2 | 449.110645 | 1108.8076 |
| A2 | 451.758012 | 1538.1641 |

Medians: 451.609459 → 448.965610 seconds (−0.6%) and
1533.018555 → 1110.769043 MiB (−27.5%). One batch on shared hardware;
this establishes a memory improvement, with no substantial speed claim.
Evidence: `~/.cache/sdpx-e2e/ising43-cluster-20260927/memory-paired-v2/`;
remote `~/projects/sdpx-ising43-20260927/results/memory-v2-*` and
`results-sdpx-memory-v2.json`. Frozen source archive:
`fbbc9f2b8da5cfc5767762414043f936e401dbee316cf506229245687233b4f5`.

Linux converter checks in job `221705.node220` pass: six upstream fixtures,
all 31 MPFR dispatch choices, malformed-input/output-preservation checks,
and four complete solves with independent audits. Workspace release tests
are still pending; do not interpret the converter pass as a suite pass.

Additional candidates implemented, awaiting complete-solve qualification:

- Exact finite-field full-row-rank proof, with existing rational fallback.
- Exact bilinear sampled RHS for matrix-valued blocks, rounded once.
- Owned MPI matrix compaction and shared residual basis-cache preparation.
- Shared automatic worker budget respecting `RAYON_NUM_THREADS`.
- Public declaration for the existing `sdpx_mpi_world_size` C ABI function.

Release pinned and enlarged-matrix checks run in `221706.node220`, source
archive `a66d9db676f28d7494e96c63c6e7ac006c6f72c13cbdeeacbe3b6d4afb7c3bc7`.
Intermediate real MPI qualification is `221707.node220`. Final source,
including later cache/thread/argument-guard changes, is qualified separately
by `221708.node220`, archive
`3568263f750877c266e8e1d6f219b3d2be0c53b4202ef09c9b3d8f1cc2d622fc`.
That job requires release workspace tests, pinned and matrix solves, real
MPI CLI/C ABI audits, and automatic/explicit thread parity before writing
its accepted completion marker. Existing medium failure remains explicit.

Submitted the finite Lambda43 controller `221711.node220`, after successful
`221708` and corrected-input calibration `221699`. It freezes input and
binaries, calibrates 16/64/256 cores, then runs matched full 4/16/64/256-core
SDPX/SDPB comparisons sequentially. Multi-node memory uses simultaneous
summed process PSS, with physical CPU binding and output/input hashes.
Failures stop the chain; measured resource estimates above 120 hours also
stop for a resource decision. No full Lambda43 result is claimed.
Harness: `~/projects/sdpx-ising43-20260927/mixed-scaling-20260928-v3/`, archive
`67c14b75eb3cddd5f7a0efb3cc98e107eb85c5cbc179ea124c0cd4d7ac1ee02e`.
Eight malformed-result fixtures were rejected by its lightweight gate check.
Queued calibration now uses isolated Python 3.12 with NumPy 1.26.4; system
Python 3.6 lacks the harness's monotonic clock API. All heavy work uses PBS.

## 2026-09-28 — Integration failures observed in cluster status check

Job `221705.node220` ended with exit 101: solver library tests reported
440 passed, 5 failed and 43 ignored. Failures: sampled-factor residual sparse
plan assertion, arrow fallback refactor, and arrow storage updates at Float64,
MPFR512 and MPFR768. Converter comparisons and four audits had passed before
the workspace tests. These failures require diagnosis; the suite is not passed.

Job `221706.node220` ended with exit 1. Medium and Ising11 A–B–B–A points
match bitwise; all four Ising solves return `Solved`/52. The Ising audit process
could not load the required Julia `JSON` package, so no numerical audit result
was produced. CSDR and matrix checks were not reached. Existing outputs are
preserved for re-auditing after environment repair.

Dependent jobs `221707`, `221708` and `221711` have no execution artifacts and
are no longer recognized by PBS. They must not be reported as running or
queued. The Lambda43 generation chain remains active/held separately; full
scaling results are unavailable. The accepted Lambda19 memory result stands.

## 2026-09-28 — PMP converter speed candidate

Hoisted full-precision density constants, pole square roots and sampled-basis
scale roots from inner loops. Operation order and bracketed root convergence
are unchanged. Added optional `--threads N` and library
`write_sdp_with_threads`: independent blocks are claimed dynamically, written
one at a time per worker, and metadata retains input order. All workers join
before failure cleanup; the existing serial API and CLI default stay serial.
No solver engine or numerical tolerance changed.

Submitted cluster release qualification `221717.node220` (8 cores, 32 GiB,
two hours). The timing workload repeats the upstream pole fixture 16 times
at MPFR768, in old/new-serial/new-eight-worker/reverse order. This is a
converter workload, not an Ising solve or Lambda43 performance result.
Validation requires byte-identical files, a nontrivial MPFR1216 comparison,
all 31 registered precisions through the parallel API, error cleanup and
output preservation, plus four original-coordinate audited complete solves.

Source/harness archive SHA256:
`6d3f0d5d47b137d9814a75bb9040061d55d545109db2c48370b0fc7f0ee1c224`.
Evidence: `~/.cache/sdpx-e2e/pmp-speed-20260928-v1/` and remote
`~/projects/sdpx-ising43-20260927/pmp-speed-20260928-v1/`.
Results pending; no speedup is claimed at submission.

## 2026-09-28 — PMP speed candidate accepted

Cluster job `221717.node220` finished with exit 0 on node33. Release converter
medians on the same MPFR768 input (16 copies of the upstream pole fixture),
run sequentially in A–B–C–C–B–A order:

| Converter | Workers | Wall seconds |
|---|---:|---:|
| Previous | 1 | 20.472284 |
| Candidate | 1 | 18.794936 |
| Candidate | 8 | 2.818450 |

Serial time fell 8.2%; eight workers were 7.26× faster than the preceding
serial converter. One batch on shared cluster hardware; this does not
establish large-Ising conversion performance. Retained all changes.

All output JSON files (including metadata) are byte-identical across the
six upstream fixtures, timed runs and nontrivial 1216-bit comparison.
Parallel dispatch at all 31 MPFR precisions, invalid-worker cleanup, zero
worker rejection and preservation of existing output passed. Four converted
inputs returned `Solved` and passed independent original-coordinate audits:
supplied/automatic/XML/XML2, respectively 39/39/34/39 iterations, unchanged
768-bit precision and 1e-42 solver tolerances, audit threshold 1e-30.

Old binary SHA256:
`3352e9d4a9c71ac3b74787de98102390119ed770e67873e13ebadb48e5f4b1cc`.
New binary SHA256:
`947413c1ce80d4187a0d4af17c79d535ff193f7ad81e1f660ca80cbf3b755535`.
Timing input SHA256:
`3d2a9dbac2187bac629ddd2248ba0294ec4de7d42716f3a33bd75ff3b42628b8`.
Local evidence: `~/.cache/sdpx-e2e/pmp-speed-20260928-v1/evidence/`;
remote: `~/projects/sdpx-ising43-20260927/pmp-speed-20260928-v1/`.
`speed/summary.json` contains the full run rows and acceptance fields.
The original B1 timing/log files were copied to `evidence/timing-original/`
before the output-preservation check reused the B1 output path.

## 2026-09-28 — Rust converter versus SDPB pmp2sdp

Job `221720.node220` passed on node33, eight allocated physical CPU slots
with explicit affinity. Frozen Rust converter from `221717` and the existing
SDPB 3.1.0 build (`--version` reports `3.1.0-dirty`). Same input bytes,
precision and uncompressed JSON output; serial BLAS/OpenMP. SDPB uses MPI
ranks at width eight, Rust uses block workers. Each case/width runs sequential
SDPB–Rust–Rust–SDPB. No solver changes or new solve-performance claim.

Whole-command medians, including process/MPI startup and I/O:

| Input | Bits | Workers | Rust seconds | SDPB seconds |
|---|---:|---:|---:|---:|
| 16 repeated pole blocks, automatic sampling | 768 | 1 | 17.495162 | 3.403778 |
| Same | 768 | 8 | 2.443195 | 0.915778 |
| Ising Λ11 preflight XML, supplied sampling/bases | 1216 | 1 | 3.448615 | 4.056028 |
| Same | 1216 | 8 | 0.890525 | 2.527379 |

One preliminary batch on shared hardware. Startup variability is material:
SDPB's serial pole commands took 5.541 and 1.266 s, while its own conversion
timer reported 0.991 and 0.921 s. At eight workers Ising's SDPB conversion
timer reported 0.904/0.914 s versus 2.556/2.499 s whole-command time. Therefore
the 2.84× end-to-end Rust advantage on that input is largely startup; it is
not a demonstrated 2.84× arithmetic-kernel improvement. For automatic pole
sampling Rust is 2.67× slower end-to-end at eight workers. SDPB uses Newton
iteration with a half-precision stopping target; Rust retains full-working-
precision bisection. Safeguarded full-precision Newton is a future candidate,
requiring independent root/output and solve validation before adoption.

Objective, block dimensions, bases and coefficients were compared on every
run. Maximum scaled differences versus SDPB: 6.524e-229 for poles and
2.221e-351 for Ising, both below the predeclared 1e-100 comparison gate.
Each converter's numerical payload files are byte-identical across repeats
and thread/rank counts. The Ising input is the existing preflight model,
not corrected Λ43 or an original paper input; no scientific solve result is
claimed for this conversion benchmark.

Harness archive SHA256:
`d58bbe2bdab936a3d283b3906c17b6cd37836c9a15fab29579741ca64a970b6a`.
Local evidence: `~/.cache/sdpx-e2e/pmp-sdpb-compare-20260928-v1/evidence/`.
Remote inputs, raw outputs, affinity receipts and commands:
`~/projects/sdpx-ising43-20260927/pmp-sdpb-compare-20260928-v1/`.
`manifest.json` pins both binaries and inputs; `results/summary.json` retains
all timings and coefficient comparisons. Existing failed solver integration
checks remain open independently of this converter comparison.

## 2026-09-28 — Full-precision Newton and PMP memory candidate

Implemented safeguarded Newton with analytic density derivatives. A Newton
step is accepted only inside the sign bracket; invalid/non-improving steps
fall back to bisection. Near rounding noise, test a local sign bracket and
bisect to adjacent representable values. As before, an exactly zero computed
residual also stops. No half-precision target or relaxed root tolerance was
introduced. Root last digits may change, so complete solve audits are required.

Replaced the full-document XML tree with a buffered quick-xml event reader,
retaining comments/CDATA and numeric entity handling, duplicate-field checks,
DTD rejection and shape validation. Replaced roxmltree with pinned quick-xml
0.37.5 (existing memchr dependency). Parse MPFR polynomial coefficients for
one upper-triangle matrix entry at a time; nonidentical text in symmetric
entries still receives numeric equality checks.

Cluster job `221721.node220` (8 cores, 32 GiB, two hours) builds a frozen
release, runs upstream conversion checks and four solve audits, compares
sample points against prior bisection at every supported precision, exercises
XML validation and threaded cleanup, then compares old/new/SDPB on the two
existing workloads. Time runs are sequential, old–new–SDPB–SDPB–new–old.
Separate untimed memory runs sample the simultaneous sum of converter process
PSS, excluding launchers, at 20 ms intervals. No candidate result claimed yet.
Archive SHA256:
`9cd9d498fa925d04bd19252610cb334775c714d3bd42604c9194db69aae50905`.
Local preparation: `~/.cache/sdpx-e2e/pmp-newton-20260928-v1/`;
remote evidence: `~/projects/sdpx-ising43-20260927/pmp-newton-20260928-v1/`.

## 2026-09-28 — Newton/streaming XML candidate qualified

Job `221721.node220` completed with exit 0. All six upstream reference checks
and four original-coordinate solve audits passed. Sample points at every
64-bit precision from 128 through 2048 agree with the previous bisection
within the predeclared 64-epsilon scaled comparison. This comparison is a
validation bound, not a new root stopping tolerance. XML entities, CDATA,
comments, malformed-input rejection and parallel failure cleanup passed.
Repeated and threaded payloads are byte-identical within each converter;
the supplied-sampling Ising payload is also identical to the old converter.
Automatic sampling changes the last digits (maximum coefficient difference
from the prior Rust payload 9.565e-230 at MPFR768).

First candidate results at eight workers on node33, one preliminary paired
release batch, whole-command seconds:

| Input | Prior Rust | Newton/streaming Rust | SDPB 3.1.0 |
|---|---:|---:|---:|
| Automatic pole fixture, MPFR768 | See timing rows | 0.164588 | 1.118690 |
| Ising Λ11 preflight XML, MPFR1216 | 1.044801 | 0.915550 | 2.167771 |

Separate 20-ms PSS measurements on Ising: serial prior/new/SDPB
77.029/48.889/125.638 MiB; eight workers 100.369/94.436/160.401 MiB.
These are simultaneous sums over converter processes, not sums of independent
rank maxima. Timing includes process/MPI startup and JSON I/O; this is not a
large-Λ43 result or an arithmetic-kernel speed ratio. Candidate retained.
Local summaries: `~/.cache/sdpx-e2e/pmp-newton-20260928-v1/evidence/`.

A subsequent bounded memory refinement removes the temporary copy of each
entry's MPFR polynomial vectors and reuses the first XML scalar text buffer.
The normalization/Horner arithmetic order is unchanged. A fresh source and
separate namespace `pmp-memory-20260928-v2` will verify exact payload parity,
repeat the converter solve audits, and measure paired time and process PSS.

## 2026-09-28 — PMP coefficient buffer refinement accepted

Job `221722.node220` passed on node33. Kept direct parsing into normalized
polynomial vectors and reuse of the first XML text buffer. All six fixture
outputs and 31 precision-dispatch outputs are byte-identical to the qualified
Newton version. Split comment/CDATA/entity XML text and numerically equal,
textually different symmetric coefficients also produce identical output.
Four complete solves again pass unchanged original-coordinate audit gates.
The Newton source is unchanged from the 31-precision root qualification.

Release whole-command medians from old–new–SDPB–SDPB–new–old ("old" here is
Newton/streaming XML v1), fixed CPU affinity, same input bytes and precision:

| Input | Workers | Old seconds | Final Rust seconds | SDPB seconds |
|---|---:|---:|---:|---:|
| Automatic pole fixture, MPFR768 | 1 | 0.640312 | 0.615409 | 2.276507 |
| Same | 8 | 0.164680 | 0.164694 | 0.790443 |
| Ising Λ11 preflight XML, MPFR1216 | 1 | 3.324595 | 3.277324 | 4.242533 |
| Same | 8 | 0.941652 | 0.915465 | 2.120863 |

Separate untimed peak-PSS measurements, simultaneous sum over converter
processes at 20 ms intervals:

| Workers | Old MiB | Final Rust MiB | SDPB MiB |
|---|---:|---:|---:|
| 1 | 48.833984 | 48.130859 | 125.629883 |
| 8 | 93.345703 | 88.728516 | 159.581055 |

Final eight-worker whole-command speed ratios are 4.80× for automatic
sampling and 2.32× for Ising versus SDPB. Ising PSS is 44.4% lower at eight
workers and 61.7% lower serially. The incremental copy removal cuts PSS
another 4.9% at eight workers relative to the qualified Newton candidate.
Times and memory are preliminary single batches/probes on shared hardware.
Startup contributes materially to SDPB wall time; these are not universal
kernel-speed claims. Neither workload qualifies large-Λ43 conversion scaling.

Final source/harness archive SHA256:
`8759518cbd864b135213e6994b74d81f9c51f347cf84e9e0d447e26615748f89`.
Evidence: `~/.cache/sdpx-e2e/pmp-memory-20260928-v2/evidence/` and
`~/projects/sdpx-ising43-20260927/pmp-memory-20260928-v2/`.
The manifest pins inputs/binaries; bench rows, raw output payloads, audit
files, physical-CPU receipts and simultaneous-PSS traces remain on the cluster.
Local summaries retain all timing rows, memory peaks, parity results and
solve-audit status. No numerical tolerance or working precision was reduced.

## 2026-09-28 — Eight-hour improvement campaign and integration repair

Started the user-authorized eight-hour window at 03:04 China time, ending
11:04. SDPX remains the main task; PMP conversion is secondary. All substantial
builds, tests and solves run through PBS. No commits are requested.

The five integration assertions from 221705 assume pre-compaction matrix
storage or QDLDL regularization despite an explicitly disabled setting. The
sampled check now examines `materialize_A()` and verifies compact storage.
Arrow checks require both backends to reject the zero singleton pivot with
regularization disabled, preserve lazy materialization and recover when the
pivot is restored. Existing successful fallback/batch coverage is retained.
No production regularization or numerical acceptance rule was relaxed.

Installed the exact pinned Julia audit dependencies into the new campaign
namespace. Job 221727 stopped during compilation because a copied Cargo
cache reused an older arithmetic library lacking the modular-rank method.
Job 221728 excludes all workspace crate artifacts/fingerprints from its cache
and rebuilds the complete integration snapshot. Its source/harness archive is
`f417fbdb37c6d510b0c73da835b21c0ae7a5ed913e43d6ec5de498338f562d93`.
The failed artifacts remain intact; no numerical result is accepted yet.

A separate candidate reuses the cone's R·Rᵀ in KKT updates and applies the
existing exact residue-BLAS kernel to suitable MPFR symmetric products.
Aliased operand views share one encoding. The upper-triangle and rounding
contracts are retained. Job 221729 is the release/pinned/Lambda19/MPI gate;
archive `5e143f1b43946f303d6e840eb424542cbb6ed15fc1d2cc2b908ddbbfb07f3195`.
Comparisons are sequential per host with BLAS at one thread and explicit
physical CPU affinity. These are unqualified candidates, not measured gains.

### 2026-09-28 — Bounded PMP input and lazy solver workspace candidates

- PMP job `221730.node220` freezes bounded two-pass JSON/XML conversion with
  one block per worker plus the reader. Existing API, byte parity, all 31
  precisions, malformed-input cleanup and four complete solve audits precede
  matched Rust/SDPB conversion timing and simultaneous memory measurements.
  Archive SHA-256: `56b734b841af32523db87f23a52167e715c998b40062df9ffe7d16025f56a952`.
- Solver job `221739.node220` follows `221729` on node9. It removes unused left
  SVD storage, allocates paired PSD step work only when used, and reconstructs
  eligible exact residue products into their destination matrix. Numerical
  operations and gates are unchanged. It includes an eight-thread matrix solve
  to exercise paired workspaces, full integration and MPI/C ABI validation.
  Four distributed state checks now distinguish full operator nonzeros from
  compact linear storage. Archive SHA-256:
  `9d42114bb8a7a7da16219f4e6e7f802d1cac5ff75433d7632d9ed08c912642c4`.
- Both are pending candidates. No speed, memory or correctness outcome is
  claimed before the cluster evidence completes.

### 2026-09-28 — Integration fixture diagnosis and candidate repairs

- `221728` completed compilation but failed its solver suite: 438 passed,
  seven failed, 43 ignored. Four state assertions compared compact storage
  with expanded counts; three KKT checks rebuilt kernels from compact runtime
  state. Tests now restore construction matrices from the sampled factors.
  Independent original-operator residual checks remain unchanged. The initial
  sparse-plan and disabled-regularization repairs passed this run.
- Held unstarted `221739` to preserve its snapshot without repeating known
  fixture failures. Replacement workspace candidate `221743` includes the
  final test repair; archive SHA-256
  `ada675ba0bd5a06dc39ad9f48ea09fc719fc556855450fe9565d90b5c2a34906`.
- `221730` failed to compile because the cancellation flag was declared inside
  the thread scope. Moving it outside repairs its lifetime. `221742` compiled
  the corrected PMP snapshot and began conversion validation. Archive SHA-256:
  `bba7037c01eb2876751f5bfe7fb8397422084727a09f2a23664b01e4e8574459`.
- `221744` follows PMP validation. Its fast-profile A/B arms isolate skipping
  zero products during MPFR SVD rotation replay, preserving signed zeros. The
  rule matches the pinned MPFR 4.2.2 `fmma.c`/`add.c` zero paths. Nonzero and
  nonfinite paths are unchanged. Archive SHA-256:
  `2593a745b7beed3905960c0d30ba1114f67537fa4458c385004aa589287353f7`.
  No performance claim is made before complete-solve evidence and release
  confirmation.

### 2026-09-28 — Bounded PMP file conversion retained

`221742.node220` exited 0. Six fixture payload checks, every precision from
128 to 2048 in 64-bit steps, four complete original-coordinate solve audits,
12 in-memory API calls, header ordering, normalization, serial/threaded parity
and malformed/late/worker-error cleanup all passed. Existing output is refused;
workers finish before removing a failed call's own output directory.

One release A–B–C–C–B–A batch on node13, old Rust / streaming Rust / SDPB 3.1.0:

| Input | Workers | Old Rust (s) | Streaming Rust (s) | SDPB (s) |
|---|---:|---:|---:|---:|
| Automatic 16-pole, MPFR768 | 1 | 0.666 | 0.615 | 3.699 |
| Automatic 16-pole, MPFR768 | 8 | 0.164 | 0.189 | 0.922 |
| Ising Λ11 XML, MPFR1216 | 1 | 3.244 | 3.343 | 3.992 |
| Ising Λ11 XML, MPFR1216 | 8 | 0.791 | 0.740 | 1.967 |

Separate 20 ms simultaneous process-PSS passes on the Ising XML: serial
48.14 → 10.84 MiB (−77.5%); eight workers 88.63 → 77.79 MiB (−12.2%).
SDPB used 123.23 / 158.60 MiB. Retained for bounded memory and overlap of
parsing with conversion. Serial Ising time increased 3.1%; eight-worker time
fell 6.4%. Automatic eight-worker runs varied from about 0.11 to 0.21 s, so
its apparent Rust regression is not a reliable timing distinction. All times
include command startup and output; SDPB parallel time also includes MPI.
These are preliminary workload-specific measurements, not a Λ43 result.

Evidence: `~/.cache/sdpx-e2e/pmp-stream-20260928-v2/evidence/`; source archive
SHA-256 `bba7037c01eb2876751f5bfe7fb8397422084727a09f2a23664b01e4e8574459`.

### 2026-09-28 — PMP polynomial evaluation candidate submitted

`221745.node220` follows the solver replay comparison on node13. It removes
redundant leading Horner work after polynomial normalization; original lengths
still determine sample counts and basis sizes. Remaining arithmetic keeps its
order. All-zero polynomials are shortened only when their zero signs permit
it. Payload checks include signed zeros, sample `-0`, normalization and 128,
768 and 2048-bit inputs, followed by the existing complete converted-input
solve audits and matched conversion/memory comparison. Results are pending.
Archive SHA-256: `8d7368469d178f193492fe4fff44f724c888830e171566dad8de7e289a22b716`.

### 2026-09-28 — Sampled-input reader candidate submitted

`221751.node220` follows the PMP candidate on node13. It parses decimal strings
directly into the working scalar type without retaining each block's strings.
B rows retain only nonzeros plus their full width for dimension validation.
String-only wire input, finite/underflow checks, CSC ordering and factor values
are preserved; malformed decimals now carry JSON source positions. No Float64
intermediate is introduced for MPFR. The isolated fast comparison requires
pinned medium/Ising11 point parity, full Lambda19 audits and real MPI. Profiled
large solves also record peak memory reached at input completion. Release
confirmation is required before quoting a gain. Archive SHA-256:
`9c585463100d67111eaf0de09bd4bd078540a2a0adcb329469b0d6201f96013f`.

### 2026-09-28 — Shared transformed-basis cache candidate submitted

`221754.node220` isolates sharing the sampled forward/adjoint V encoding when
both inner dimensions select the same prime width. Different widths retain
separate caches. The existing exact-bit, exponent-alignment and prime-count
checks still control reuse; the larger adjoint encoding can serve a forward
prefix. No arithmetic or output ordering changes. Required checks are pinned
and Lambda19 complete solves, matrix-valued one/eight-thread parity, and real
MPI. This fast build is for selection; release confirmation remains required.
Archive SHA-256: `14b33dc290aa183cdf28724ed4a8bf0aed8f65eddcbdbb9d9a7d92ef6559a06e`.

### 2026-09-28 — Gram reuse retained after release and MPI checks

`221729.node220` exited 0. One release A–B–B–A batch on node9, four fixed
physical cores: Lambda19 MPFR768 median wall time 457.112 → 421.484 s
(−7.8%), simultaneous peak PSS 1121.23 → 1129.57 MiB (+0.7%). Every run
was `Solved`/119 with identical points and passing original-coordinate audit.
Medium and Ising11 one/four-thread points match; medium retains exactly its
known external failure. Real two-rank MPI A–B–B–A also preserves both cases'
points and audits. This qualifies the isolated Gram change against its frozen
preceding arm; it does not replace the pending full integration checks.

Retained cone Gram reuse, exact residue SYRK and encoding one copy of identical
GEMM operands. Evidence: `~/.cache/sdpx-e2e/gram-reuse-20260928-v1/evidence/`.
Source archive SHA-256:
`5e143f1b43946f303d6e840eb424542cbb6ed15fc1d2cc2b908ddbbfb07f3195`.

### 2026-09-28 — PMP output-row streaming candidate submitted

`221755.node220` follows workspace checks on node9 and successful polynomial
qualification. Basis and B rows are serialized immediately; c entries stream
individually. Decimal arrays no longer remain allocated for an entire output
block. JSON field order and decimal formatting stay unchanged. Normalized
constant coefficients are parsed again for B to bound retained memory; the
per-entry arithmetic sequence is unchanged. Worker joins and whole-call error
cleanup still precede publication of control.json. Byte parity, all precision
and converted-solve audits, signed zeros, error cleanup and matched SDPB/Rust
conversion time/PSS are required. Results are pending. Archive SHA-256:
`280baecde00c0c580c839c4ba288aefcdc52e371f7e581bc3ea6ba364bdc4e22`.

### 2026-09-28 — Packed residue-cache candidate submitted

`221758.node220` follows successful shared-V qualification on node13. Balanced
residues for prime widths up to 24 lie strictly between −2^23 and 2^23, so
three signed bytes preserve each integer exactly. Wider cached residues keep
f32 storage. Decoding sign-extends to i32 and converts exactly to the f64 BLAS
operand; cache identity and CRT arithmetic remain unchanged. This reduces
eligible retained residue payloads by 25%, not necessarily whole-process
peak memory. Fast-profile pinned/Lambda19, matrix one/eight-thread and real
MPI complete-solve checks will determine whether the tradeoff is worthwhile.
Release confirmation is required for performance claims. Archive SHA-256:
`af1e0c6f116ed35f3526f430dd777cf8e44839dd09ad9363ff580b769be0028d`.

### 2026-09-28 — Zero-pair SVD shortcut removed

The complete isolated fast-profile A–B–B–A comparisons from `221744` showed
no speed or memory benefit on medium, Ising11 or Lambda19. All four large
runs were `Solved`/119 with identical points and passing audits; the candidate
was slightly slower and its peak PSS slightly higher. No release performance
claim is made from this development selection. Removed only the zero-pair
branch and its lazy canonical-zero variable, restoring the frozen preceding
arithmetic file byte for byte. Earlier replay improvements are preserved.
The live job is allowed to finish its remaining MPI checks; all artifacts stay
under `~/.cache/sdpx-e2e/replay-zero-fast-20260928-v1/` and the matching cluster
namespace. This direction is closed without new profiling evidence.

`221744` subsequently exited 0: real two-rank MPI also preserved points and
audits. Numerical checks passed; the performance rejection is unchanged.
Reports are in `~/.cache/sdpx-e2e/replay-zero-fast-20260928-v1/evidence/`.

### 2026-09-28 — PMP zero-term shortcut removed

`221745.node220` exited 0. Six fixture payload checks, all 31 precisions, four
converted-input solve audits, in-memory API/thread parity, signed-zero padding
and failure cleanup passed. One release A–B–C–C–B–A batch on node13 measured
Ising1216 serial time 3.293 → 3.296 s; eight-worker medians 0.777 → 0.766 s,
with candidate runs spanning 0.666–0.866 s. Serial PSS was 10.84 MiB for both;
eight-worker PSS 77.28 → 77.47 MiB. This does not establish a useful gain.
Automatic short runs and SDPB runs also varied substantially; no new speedup
claim is drawn from them. Removed the trimming helper and calls, preserving
the independent row-output candidate. That frozen row experiment still
compares against this numerically qualified zero-term arm; the combined
release will compare final output streaming against accepted input streaming.

### 2026-09-28 — Workspace tests repaired; doctest link environment corrected

`221743` completed all compiled unit/integration suites without failures.
The solver unit suite reported 445 passed, zero failed, 43 ignored; all seven
previously failing distributed state/KKT fixtures passed with their original
residual checks. The final doctest stage failed one CscMatrix example because
rustdoc's linker lacked the external OpenBLAS search path (`-lopenblas`).
This is a harness link failure, not a numerical failure, and the full workspace
check is not yet marked passed.

Retry `221761.node220` preserves the exact frozen v2 source and copies its
build artifacts into a fresh target directory. RUSTDOCFLAGS and LIBRARY_PATH
now include the pinned provider directory. It reruns the full suite, then
continues release solve, matrix, MPI and C ABI qualification. The final
combined harness has the same environment repair. The original log is stored
under `~/.cache/sdpx-e2e/solver-workspace-20260928-v2/evidence/`.
Retry package SHA-256:
`c5c7a06039f5e952e179ed67840c89decae806b1ebc77c3080022ed3553d0c57`.

### 2026-09-28 — PMP row output retained after release qualification

`221755.node220` exited 0. Output files match the preceding converter byte
for byte. Six fixture checks, 31 precisions, signed zeros, 12 in-memory API
calls, thread parity and malformed/late/worker-error cleanup passed. All four
converted-input complete solves passed original-coordinate audits.

One release A–B–C–C–B–A batch on node9, preceding Rust / row-streaming Rust /
SDPB 3.1.0, including command startup and uncompressed JSON output:

| Input | Workers | Previous Rust (s) | Row output (s) | SDPB (s) |
|---|---:|---:|---:|---:|
| Automatic 16-pole, MPFR768 | 1 | 0.640 | 0.615 | 2.142 |
| Automatic 16-pole, MPFR768 | 8 | 0.164 | 0.164 | 0.690 |
| Ising Λ11 XML, MPFR1216 | 1 | 3.194 | 3.243 | 4.820 |
| Ising Λ11 XML, MPFR1216 | 8 | 0.865 | 0.690 | 1.967 |

Separate 20 ms PSS passes: serial 10.84 → 6.74 MiB (−37.9%), eight workers
77.62 → 39.98 MiB (−48.5%). SDPB used 125.81 / 158.65 MiB. Retained for
bounded output memory and the parallel improvement; serial time rose 1.5%.
These remain preliminary workload-specific numbers. Automatic/SDPB short
runs varied; no Λ43 result is inferred.

The frozen row experiment includes zero-term trimming in both Rust arms.
That independent shortcut was removed locally after showing no benefit. The
final combined converter will be compared directly with accepted input
streaming, including byte parity and complete converted-input solves.
Evidence: `~/.cache/sdpx-e2e/pmp-row-output-20260928-v1/evidence/`.
Archive SHA-256: `280baecde00c0c580c839c4ba288aefcdc52e371f7e581bc3ea6ba364bdc4e22`.

`221761` again passed the compiled suites but rustdoc rejected `-l`, a
compiler-only flag incorrectly copied into RUSTDOCFLAGS. Second scoped retry
`221766.node220` uses only `-L native=<provider>` there and keeps the provider
in LIBRARY_PATH. The pinned rustdoc accepts that option in a brief preflight.
Source and numerical settings remain unchanged; earlier outputs are preserved.
The combined harness has the same correction. Retry package SHA-256:
`50daaec583b342de9e9eecd1eb3f75d9e1783c319b066c4bdf346b7a487cee60`.

`221766` passed the entire Linux release workspace suite, including all four
solver doctests. The repaired rustdoc search path resolved the harness issue.
The solver unit suite remains 445 passed / zero failed / 43 ignored. Release
binary construction and complete-solve/MPI/C ABI checks follow in the same
job. Log: `~/.cache/sdpx-e2e/solver-workspace-20260928-v4/evidence/workspace-tests.log`.
This snapshot predates the later reader/cache/converter candidates; final
combined integration is still required.

### 2026-09-28 — Converter timing resolution corrected for final comparison

Inspection found that the earlier converter timing harness used
subprocess.run(timeout=...) with file outputs. Python's POSIX timeout wait
polls with delays up to 50 ms, visible in the short timings above. Those are
observed whole-command waits with coarse resolution; the reported 1.5%
serial row-output difference is inconclusive. Memory sampling, byte checks
and original-coordinate audits are unaffected. The larger parallel difference
exceeds this quantization, but precise final confirmation is still required.

The final combined harness now times a blocking Popen.wait(), with a separate
1200-second watchdog and its cancellation/join outside the measured interval.
It will compare accepted input streaming, final row streaming and SDPB anew.
The solver's pinned harness already uses blocking os.wait4 and is unaffected.

### 2026-09-28 — Direct sampled decimal reader retained for loading

Job `221751.node220` completed all pinned, Lambda19 and real two-rank MPI
A–B–B–A checks. Every point is identical; Ising audits pass and medium retains
its known failure. Direct MPFR parsing and sparse B-row construction remove
duplicate decimal buffers. The fast-profile input stage became faster and
used slightly less memory; whole-solve PSS increased slightly, so this is not
claimed as a solve-peak memory reduction. Retained for input loading, with
release timing deferred to the final combined comparison.

Evidence: `~/.cache/sdpx-e2e/sampled-reader-fast-20260928-v1/evidence/`.
Source archive: `9c585463100d67111eaf0de09bd4bd078540a2a0adcb329469b0d6201f96013f`.

### 2026-09-28 — Honor explicit glibc allocator thresholds

The CLI previously skipped its allocator defaults only when
MALLOC_MMAP_THRESHOLD_ was set, despite promising to respect user settings.
It now also skips them for MALLOC_TRIM_THRESHOLD_ or MALLOC_TOP_PAD_. Default
allocator behavior is unchanged. This compatibility fix is included in the
pending final CLI/pinned integration; no allocator performance gain is claimed.

### 2026-09-28 — Remaining MPI coverage and bounded cache candidate

Queued `221773.node220` after workspace qualification: ordinary two-rank MPI
exponential-cone and forced augmented-LP solves, compared with the accepted
release baseline. Analytical original-coordinate primal, dual, cone, gap and
objective checks cover routing outside the partitioned PSD cases. This is a
short validation job on the existing node9 allocation footprint.

A separate scratch candidate builds cached residues in 16-prime groups,
compressing each immediately instead of holding a full f64 encoding alongside
the final cache. Mathematical integer GEMM and residue order are unchanged.
It follows the packed-cache experiment on node13 and is not yet in production.
Pinned, Lambda19, matrix one/eight-thread and real MPI checks decide retention.
Package: `bounded-cache-fast-20260928-v1`, SHA-256
`90513e3b6d4003dc5fa95f3b0832e895aa3f98f2e17551affa9215a4e0b17c6a`.

### 2026-09-28 — Bootstrap example documentation path

Corrected the root README's bootstrap example link to its workspace location,
`crates/solver/examples/rust/example_bootstrap.rs`, and wrapped the sampled
storage description. No benchmark table was added. `git diff --check` passes.

### 2026-09-28 — PMP decimal formatting candidate

Queued `221779.node220` on node9 after the ordinary MPI check. A scratch-only
candidate uses Serde collect_str for MPFR values and serializes borrowed
numeric rows, avoiding an outer decimal String copy and Vec<String> rows.
JSON escaping and the scalar's Display format are preserved. The frozen
baseline is current row streaming without the rejected zero-term shortcut.

Acceptance includes fixture/precision byte parity, public API/thread/error
cleanup, four converted-input audits, and a matched release comparison with
blocking process waits plus a separate memory pass. No production formatting
change is kept yet. Package `pmp-format-stream-20260928-v1`, SHA-256
`1102bde3d80592ee02e2cd5fa340cdbff3c1e3ad897c8502f0575a86640ccddf`.

### 2026-09-28 — Shared transformed-basis residues retained

`221754.node220` exited zero. Compatible forward and adjoint products now
share one V residue cache; incompatible prime widths retain separate caches.
The existing fingerprint/alignment/prime-count checks remain unchanged.
All pinned and Lambda19 A–B–B–A points are identical, as are the matrix
one/eight-thread points and real two-rank MPI comparisons. External Ising and
matrix audits pass; medium retains its unchanged known failure.

The complete fast-profile comparison used less peak PSS with no observed
speed penalty. Retained for memory; performance figures await the final
release comparison. Evidence:
`~/.cache/sdpx-e2e/shared-v-cache-fast-20260928-v1/evidence/`.
Source package SHA-256:
`14b33dc290aa183cdf28724ed4a8bf0aed8f65eddcbdbb9d9a7d92ef6559a06e`.

### 2026-09-28 — Lazy workspaces/direct MPFR output retained; MPI qualified

`221766.node220` exited zero. Retained right-only SVD storage, lazy paired
PSD step workspaces and direct reconstruction into compact GEMM destinations.
The complete release workspace suite, pinned medium/Ising11/CSDR, Lambda19,
matrix one/eight-thread, real partitioned MPI, C ABI MPI and automatic-thread
budget checks pass their existing gates. Medium's known external failure is
unchanged. All refactor/thread comparison points are identical; all five
matrix original-coordinate audits pass.

One release A–B–B–A batch on node9 (AMD EPYC 7742, four physical cores),
Lambda19 MPFR768: median wall 422.058 → 420.344 s; peak PSS 1135.71 →
1106.67 MiB (−2.6%). The small timing difference is preliminary. Every arm
reports Solved/119 and passes the unchanged audit. This compares against
Gram reuse; the final combined comparison still uses the accepted sampled
memory baseline. Evidence:
`~/.cache/sdpx-e2e/solver-workspace-20260928-v4/evidence/qualification/`.
Frozen production source SHA-256:
`ada675ba0bd5a06dc39ad9f48ea09fc719fc556855450fe9565d90b5c2a34906`.

Additional job `221773.node220` exited zero. Ordinary MPI exponential and
forced augmented-LP cases pass independent MPFR512 primal/dual/cone/gap and
analytical-objective audits at unchanged tolerances. Baseline/candidate and
serial/two-rank points are identical. Evidence:
`~/.cache/sdpx-e2e/ordinary-mpi-20260928-v1/evidence/complete/`.

### 2026-09-28 — Serde formatting candidate rejected

`221779.node220` exited zero: all six fixture byte checks, 31 precisions,
API/thread/error cleanup and four original-coordinate audits pass. However,
serial Ising Λ11 XML at MPFR1216 slowed from 3.207 to 3.393 s in one precise
release comparison; both candidate runs were slower than both baseline runs.
Eight-worker medians were 0.646 / 0.659 s, with substantial run variation.
Peak PSS was essentially unchanged (6.72 / 6.73 MiB serial, 39.99 / 39.90 MiB
at eight workers). The automatic-pole case improved, but this did not justify
the Ising regression and added wrappers. Rejected; production was never edited.
Evidence: `~/.cache/sdpx-e2e/pmp-format-stream-20260928-v1/evidence/`.

A smaller alternative will bypass Display's outer String copy using MPFR's
existing direct decimal conversion, retaining ordinary Serde string encoding.
This avoids the streaming formatter overhead tested above and requires no new
JSON wrapper types or changes to public converter bounds.

The direct-decimal alternative is `221785.node220`, in the same node9 budget
with no overlapping timed run. Scalar gains a default decimal_string method;
MPFR delegates directly to its existing to_decimal(None). Four converter
formatting sites use it; custom scalar types retain Display through the default.
The candidate remains scratch-only pending the same full conversion checks.
Package `pmp-decimal-string-20260928-v1`, SHA-256
`b02d008b7aa9456f5485cb78f7ea2dda3dfdf0cb8c0cd4f81430b71a81a34c8d`.

### 2026-09-28 — Direct decimal strings retained

`221785.node220` exited zero. All six fixtures, 31 precisions, signed-zero
cases, API/thread/error cleanup and four converted-input audits pass with
byte-identical output. Retained the small Scalar default/MPFR override and
four PMP call-site substitutions; no Serde wrapper types were added.

One release A–B–C–C–B–A batch on node9, using blocking process waits:

| Input | Workers | Row-stream baseline | Direct decimal strings | SDPB 3.1.0 |
|---|---:|---:|---:|---:|
| Automatic 16-pole, MPFR768 | 1 | 0.612 s | 0.577 s | 2.173 s |
| Automatic 16-pole, MPFR768 | 8 | 0.150 s | 0.134 s | 0.633 s |
| Ising Λ11 XML, MPFR1216 | 1 | 3.218 s | 3.222 s | 3.994 s |
| Ising Λ11 XML, MPFR1216 | 8 | 0.695 s | 0.637 s | 2.056 s |

Serial Ising is essentially unchanged; both eight-worker candidate runs were
faster than both baseline runs (median −8.3%). Separate peak-PSS measurements
are similar: 6.73 / 6.77 MiB serial, 39.99 / 40.19 MiB with eight workers.
No memory reduction is claimed for this formatting change. SDPB short-run
variation is substantial, so these are preliminary workload-specific results.
Final composition will still be compared to accepted input streaming.
Evidence: `~/.cache/sdpx-e2e/pmp-decimal-string-20260928-v1/evidence/`.

### 2026-09-28 — Invalid cache lifetime candidate

Queued `221786.node220` on the now-free node9 allocation. On a cache miss,
the scratch candidate takes the invalid entry under its mutex and drops it
outside the lock before encoding a replacement. Active products retain their
own Arc, so their storage stays valid. Hit checks and numerical encoding are
unchanged. The earlier implementation retained the invalid storage through
replacement construction.

This is isolated against the packed-cache binary and is independent of bounded
prime-group encoding. Both changes require final combined qualification if
retained. Pinned, Lambda19, matrix one/eight-thread and real MPI checks run
sequentially within the job. Package `cache-release-fast-20260928-v1`, SHA-256
`804689dd5d20e8612489e9f47adaa6dbd16bd99cd5d8e75b0f6758d4910d8e50`.

### 2026-09-28 — Packed residue storage retained

`221758.node220` exited zero. Retained signed three-byte storage for balanced
residues at prime widths up to 24 bits, with exact f32 storage for 25-bit
primes. Complete pinned, Lambda19, matrix one/eight-thread and real MPI
comparisons preserve solution points. All applicable original-coordinate
audits pass; medium's known failure is unchanged. All five matrix audits pass.

The fast development comparison showed lower peak PSS with a small runtime
tradeoff. No release performance number is claimed for this isolated change;
the combined release comparison will determine the overall effect. Evidence:
`~/.cache/sdpx-e2e/packed-residue-fast-20260928-v1/evidence/`. Package SHA-256:
`af1e0c6f116ed35f3526f430dd777cf8e44839dd09ad9363ff580b769be0028d`.

The bounded-construction experiment `221774.node220` has started on node13.
Its candidate remains outside production until the same checks finish.

### 2026-09-28 — Early invalid-cache release retained

`221786.node220` exited zero. Pinned medium/Ising11, Lambda19, matrix
one/eight-thread and real two-rank MPI comparisons preserve solution points.
All applicable original-coordinate audits pass, including all five matrix
audits; medium's known failure is unchanged.

Retained the cache-miss lifetime change: take the invalid entry under the
mutex, then release it outside the lock before constructing its replacement.
Products already using an entry retain their own Arc. The complete fast
development batch showed a modest memory reduction with essentially unchanged
runtime; final release numbers remain pending. Production matches the frozen
candidate, with no other changes to its residue source. Evidence:
`~/.cache/sdpx-e2e/cache-release-fast-20260928-v1/evidence/`. Package SHA-256:
`804689dd5d20e8612489e9f47adaa6dbd16bd99cd5d8e75b0f6758d4910d8e50`.

### 2026-09-28 — Buffer streamed CLI result files

Code inspection found that the CLI already borrows solution vectors and
streams JSON, but writes result files through an unbuffered File. Added the
standard BufWriter around that file handle. The existing explicit flush
continues to propagate write errors; serialization and stdout behavior are
unchanged. This avoids repeated small file writes without materializing the
point. Final medium/Ising11 and combined release checks will qualify it; no
isolated speed claim is made.

### 2026-09-28 — Bounded cache construction retained

`221774.node220` exited zero. Pinned medium/Ising11, Lambda19, matrix
one/eight-thread and real two-rank MPI comparisons preserve solution points.
All applicable original-coordinate audits pass, including all five matrix
audits; medium's known failure is unchanged.

Retained cache construction in 16-prime groups. Each group is compressed
immediately into the final three-byte/f32 storage, avoiding a complete f64
encoding alongside the compressed cache. The fast development batch showed
lower peak PSS with essentially unchanged runtime. No isolated release number
is claimed. Combined it with the separately qualified early invalid-cache
release; the union differs from the bounded candidate only in that lifetime
block. Final combined release qualification remains required. Evidence:
`~/.cache/sdpx-e2e/bounded-cache-fast-20260928-v1/evidence/`. Package SHA-256:
`90513e3b6d4003dc5fa95f3b0832e895aa3f98f2e17551affa9215a4e0b17c6a`.

### 2026-09-28 — Combined release integration submitted

Submitted `221792.node220` after both final cache experiments exited zero.
The frozen source includes all retained solver/converter changes and buffered
CLI result files. The allocation consolidates the earlier two eight-core jobs
into 16 cores on node9; timed solver arms use the same four-core subset, and
matrix/converter arms use the same eight-core subset. Comparisons remain
sequential. Third-party build artifacts are reused, while production crate
fingerprints/artifacts are excluded from that cache.

Required checks: full release workspace suite; pinned medium/Ising11/CSDR;
Lambda19 A–B–B–A against the accepted sampled-memory baseline; matrix audits
and thread parity; partitioned and ordinary MPI; C ABI and automatic thread
budgets; final PMP byte/precision/API/cleanup/audits and matched comparisons
against accepted input streaming and SDPB 3.1.0. All numerical gates remain
unchanged. The automatic-budget check has 16 available CPUs and an explicit
RAYON_NUM_THREADS=4 cap.

Package `combined-release-20260928-v1`, SHA-256:
`bd26af0d3f7447e21674c59554fb974cedc428f339245f556d3474f402f9a4bc`.
Submission receipt and frozen source are under `~/.cache/sdpx-e2e/` in that
namespace. No combined performance result is available yet.

### 2026-09-28 — Clarify user documentation during release integration

Corrected the README to distinguish the standard solution JSON from detailed
phase receipts, enabled with SDPX_RECEIPT. Added the locked dependency and
single-test-thread options to its release test command. Replaced the stale
Ising11 limitation about all larger inputs with the specific pending Lambda43
comparison. These are documentation-only changes; the frozen production
source remains unchanged. Verified with git diff --check.

### 2026-09-28 — Combined source passes the full workspace suite

`221792.node220` completed the full release workspace suite: 697 passed,
0 failed, 59 ignored across 38 result groups, with no compiler warnings.
All four solver doctests pass. This source includes the combined packed,
bounded and early-release caches, direct decimal strings and buffered CLI
result files. Evidence: `combined-release-20260928-v1/evidence/progress/`
under `~/.cache/sdpx-e2e/`. The ordinary release build is now running; the
final numerical and performance comparisons remain pending.

### 2026-09-28 — Recover monitoring after intermittent SSH timeouts

The SSH monitor disconnected during release compilation. Read-only checks
confirmed that the route used PC's Tailscale subnet and that PC could reach
the cluster endpoint. A temporary SSH jump query confirmed the existing PBS
job was running; subsequent direct access recovered. No PBS job was restarted
or canceled, and no persistent network or SSH settings were changed.

The monitor now has bounded reconnection handling. `221792.node220` was again
confirmed running at approximately 01:28 UTC, still in the ordinary release
build. Its previously completed 697/0/59 workspace result is saved locally.
The unrelated request to cancel obsolete Lambda43 generation remains
unanswered; network recovery does not authorize that cancellation.

### 2026-09-28 — Final pinned release comparisons pass parity

Combined release job `221792.node220` completed the medium, Ising11 and CSDR
A–B–B–A checks against the accepted sampled-memory binary on node9. Every
comparison preserves solution points exactly; Ising11 and CSDR also match
across one/four threads and pass every external audit. Medium retains exactly
the same known dual-residual failure.

Preliminary API-time medians from this single batch: Ising11 25.505 → 24.901 s
at one thread and 7.594 → 7.469 s at four; CSDR 68.292 → 56.268 s at one
and 29.555 → 17.822 s at four. CSDR remains Solved/57 and Ising11 Solved/52.
These cluster results must not be compared directly with earlier Mac timings.
The larger Lambda19, matrix, MPI and PMP final checks are still running.
Evidence: `combined-release-20260928-v1/pinned-report.json` on the cluster;
final local evidence export will follow job completion.

### 2026-09-28 — Combined Lambda19 release comparison completed

Final job `221792.node220` completed A–B–B–A on node9 with four physical
cores, MPFR768 and unchanged settings/audits. Against the accepted
sampled-memory Rust binary, median whole-command time fell from 456.267684
to 422.140830 s (7.5%), and median sampled peak PSS from 1111.833984 to
911.539063 MiB (18.0%). Baseline runs took 456.509 and 456.026 s; candidate
runs took 422.020 and 422.261 s. All four are Solved/119, pass the original-
coordinate audit, and have identical points (`5d7108b5a6066ca9`).

This is one preliminary release batch, with sequential arms and a roughly
one-second process-memory sampler. PSS excludes compilation. Input loading
medians were 0.481 and 0.446 s; no separate strong loading-speed claim is
made from this small difference. These numbers compare the complete retained
composition directly; isolated optimization percentages are not added.
No new matched SDPB solver run was performed.

Stable reports are saved in
`~/.cache/sdpx-e2e/combined-release-20260928-v1/evidence/progress/`.
The final matrix, MPI/C ABI and PMP checks are still running.

### 2026-09-28 — Final combined integration and PMP comparison passed

`221792.node220` exited zero with `complete.json` accepted. The final frozen
source passes the full release workspace, pinned/Lambda19, matrix, partitioned
and ordinary MPI, C ABI, automatic thread-budget and PMP checks. Matrix
exact bilinear accumulation differs from the accepted baseline by at most
1.24e-131 in componentwise scaled distance; all five original-coordinate
audits pass and candidate one/eight-thread points agree exactly. The automatic
budget check had 16 available CPUs and respected the explicit four-thread cap.
Medium's known audit failure remains unchanged.

Final PMP results, one precise release A–B–C–C–B–A batch on node9:

| Case | Workers | Previous Rust (s) | Final Rust (s) | SDPB 3.1.0 (s) |
|---|---|---|---|---|
| 16 poles, MPFR768 | 1 | 0.611352 | 0.608633 | 2.326840 |
| 16 poles, MPFR768 | 8 | 0.133676 | 0.139603 | 0.673751 |
| Ising11 XML, MPFR1216 | 1 | 3.265003 | 3.207380 | 4.687906 |
| Ising11 XML, MPFR1216 | 8 | 0.855637 | 0.632936 | 2.520807 |

Separate 20 ms memory runs for Ising11 measured peak PSS of 10.868 →
6.771 MiB serial and 77.177 → 40.013 MiB at eight workers. SDPB used
125.620 and 160.474 MiB respectively. The eight-worker Rust Ising gain is
26.0% in time and 48.2% in memory; final Rust takes about one quarter of
SDPB's whole-command time on this input. The small eight-worker pole case
is slightly slower than preceding Rust, so these are case-specific results.

Comparisons include process/MPI startup and uncompressed JSON output, with
blocking process waits. Sampling stopping targets differ between tools; Rust
keeps full working precision. Six fixtures, all 31 precisions, signed-zero
byte parity, 12 library API checks, threaded output/error cleanup and four
converted-input audits pass. No Lambda43 conversion result is available.

Evidence: `~/.cache/sdpx-e2e/combined-release-20260928-v1/evidence/`. Full
artifact export is in progress; completed summary metadata is already local.
The development plan now keeps current results and open work together and
leaves isolated experiment history in this journal.

### 2026-09-28 — Final diagnostic profile submitted

After final integration exited zero, submitted `221805.node220` on idle node9
with four physical cores, 32 GiB and a 20-minute limit. It reuses the final
release binary at MPFR768 with unchanged Lambda19 settings. A 720-second
solver timeout plus the existing 300-second audit bound fit within that job.
Every preceding release solve took below 600 seconds and more than 25 minutes
remained in the requested campaign window at submission.

This is one diagnostic complete solve to refresh SVD/KKT phase priorities,
not another performance comparison. The point must equal the uninstrumented
release point and pass the same external audit. Package SHA-256:
`97f34de21474488c470f6262f6fa243813309c7dee37c4a60f6264683d76d5cf`.

### 2026-09-28 — Final profile confirms the next solver priorities

`221805.node220` exited zero. The instrumented Lambda19 MPFR768 run is
Solved/119, passes the original-coordinate audit and reproduces the final
uninstrumented x/s/z/sampled-y strings exactly. No production changes followed
this diagnostic. Current production was separately checked against all 244
files in the qualified frozen source.

Using the inclusive `wall.solve` timer (420.075534 s) as denominator:

| Phase | Wall seconds | Share | Average busy cores |
|---|---|---|---|
| KKT update | 168.394 | 40.1% | 3.53 |
| PSD cone scaling | 142.025 | 33.8% | 3.90 |
| KKT solve | 50.239 | 12.0% | 3.46 |
| Residual update | 15.158 | 3.6% | 3.54 |
| Combined RHS | 15.024 | 3.6% | 3.50 |

The KKT update scope includes the batched constant/affine RHS solves. The
refactor timer totals 33.153 s, about 7.9% of solver-loop wall time; the
40.1% update share must not be described as factorization alone. SVD rotation
replay is 62.3% of summed SVD phase time (298.644 of 479.357 s). Those
per-cone totals overlap across workers and cannot be added to outer wall
timings. The profile supports prioritizing remaining sampled RHS transforms
and SVD replay while retaining direct SVD, exact accumulation and unchanged
complete-solve gates. Previously rejected shortcuts stay closed.

Evidence: `~/.cache/sdpx-e2e/final-profile-20260928-v1/evidence/`. This profile
is diagnostic; its elapsed time and memory are not a paired speed comparison.

### 2026-09-28 — Release evidence export verified

Downloaded and verified 1,309 final release evidence files. The compressed
archive is 59,194,221 bytes with SHA-256
`70d60f4c01fe55125d4490fb8ad23b4dcb8c61569635c688d40fb99b70724f61`.
The export inventory records each file hash and any large omitted artifact
retained remotely. Local reports are under
`~/.cache/sdpx-e2e/combined-release-20260928-v1/evidence/complete/`; derived
comparison numbers are in the adjacent `release-summary.json`.

The maximum scaled SDP-coefficient differences from SDPB in the matched
PMP benchmark were 6.524e-229 for the 768-bit pole case and 2.221e-351 for
the 1216-bit Ising case. Rust outputs remain byte-identical to the accepted
input-streaming baseline.

### 2026-09-28 — SDPB comparison provenance checked

The benchmark manifest correctly records `SDPB 3.1.0-dirty`. A read-only
comparison against upstream commit `fec8e934bf03eb59b0f35ad76dd9b205dde537e6`
confirmed that all 314 production/build files match byte for byte. The only
tracked differences are end-of-line whitespace in `.gitignore` and
`docs/site_installs/EPFL.md`; transfer-created AppleDouble sidecars are also
present. No numerical source change was found, and no remote file was altered
or deleted. The plan now states the exact build provenance.

Evidence: `combined-release-20260928-v1/evidence/sdpb-source-provenance.json`
and `sdpb-upstream-source-manifest.json` under `~/.cache/sdpx-e2e/`. Manifest
SHA-256: `9a8ce7e617d5020d9361eb4c57cef6d9f2161c8339575772882ff0c0c3536bfb`.

The official [release page](https://github.com/davidsd/sdpb/releases/tag/3.1.0)
still marks 3.1.0 as the latest release, checked on 2026-09-28.

### 2026-09-28 — Preserve a concrete Lambda43 restart handoff

Saved final solver/converter hashes, upstream binary identities, corrected
input checks and pending jobs in
`~/.cache/sdpx-e2e/eight-hours-20260928/cluster-handoff.json` with a short
`CLUSTER_HANDOFF.md`. The held 221699 calibration still targets the older
precision-grid solver; its artifacts must remain historical, with a fresh
final-binary calibration before full scaling. The canonical conversion uses
one SDPB rank despite reserving eight cores, so a future converter comparison
must explicitly match workers/ranks. No job, dependency or held state was changed.

At the latest check the obsolete generator had run 11h43m without producing
its XML; corrected generation/conversion/calibration remained held. The
separate cancellation request is still unanswered. The full corrected
4/16/64/256-core scientific comparison is not complete.

### 2026-09-28 — Eight-hour optimization window completed

The requested 03:04–11:04 China-time window is complete. Final release and
diagnostic jobs exited zero, evidence is downloaded and verified, and the
retained production source and public C header match their qualified snapshots.
All substantial builds, solves and benchmarks ran in PBS. No commit was made.

The current plan records the measured solver/converter gains and the profile
priorities. Known numerical failures remain visible, and the corrected Lambda43
input and 4/16/64/256-core scientific comparison remain pending. No cancellation
approval arrived; the existing generation chain and its artifacts were preserved.


### 2026-09-28 — Keep short-wide equality rank presolve

For at most 256 equality rows, stream exact columns through modular elimination
before allocating the rational sparse-row fallback. A full-rank proof skips
redundancy work; inconclusive cases retain the existing fallback. No retained
coefficient or output coordinate changes.

Apple M4 (10 CPUs, 16 GiB), one thread, release arms
`gravity-lp-base-20260928` and `gravity-lp-column-rank-20260928`:
fixed-a gravity NN=50, nn=100, jj=100 has 49 equalities and 5,100 rho bounds.
One serial L–A–B–B–A–L batch compares the legacy local executable, published
source and the presolve candidate. Native medians in seconds:

| Precision / tolerance | Legacy | Published | Presolve candidate |
|---|---:|---:|---:|
| Float64 / 1e-6 | 6.242066 | 0.492486 | 0.350203 |
| MPFR128 / 1e-12 | 51.037731 | 23.674592 | 23.740829 |
| MPFR256 / 1e-18 | 37.574531 | 35.610655 | 35.660526 |

The new rank proof saves about 29% in Float64; MPFR totals are effectively
unchanged. All x/s/z outputs are identical across all three arms. MPFR audits
pass. Float64 is an existing failure: original dual residual 4.011564682e-6
exceeds its 2e-6 external gate, despite `Solved`/17. It remains a failure.
The first candidate Float64 process had a cold-start delay outside the native
timer; these figures exclude process startup, input loading and matrix assembly.
One balanced batch is preliminary evidence.

Pinned medium and Ising11 A–B–B–A checks preserve exact points. Ising11 passes
all audits; medium's known dual-residual failure is unchanged. Evidence:
`~/.cache/sdpx-e2e/gravity-lp-presolve-20260928/`.

### 2026-09-28 — Reject the gravity canonical-dual pilot

A separate 49-variable canonical-dual formulation returned `Solved`/29 at
MPFR128, but recovered original equality residual 2.895741356e-12 exceeded
its 1.840220876e-12 gate, both with augmented and forced condensed KKT.
The runner retains the original primal formulation. Diagnostic runs overlapped
compilation and supply no matched performance conclusion. Evidence: the
`dual-*` files in the preceding gravity cache directory.


### 2026-09-28 — Keep analytical local bounds and packed faer kernels

Frozen release `gravity-local-bounds-faer-six-20260928-v1` eliminates one/two
scalar bound rows per variable into a signed border system. Eligibility is
structural (at least 64 bounded variables, border at most 128, diagonal P).
The gravity border is 51×51. Float64 uses persistent faer panels for Schur
assembly and single/batched RHS; MPFR uses exact dot products. Original KKT
regularization, escalation, refinement, status and recovery remain unchanged.

One serial A–B–B–A batch per case/width on the Mac M4, against the frozen
column-rank-presolve release; native seconds:

| Bounds | Bits | Threads requested | Before | After |
|---|---:|---:|---:|---:|
| rho>=0 | 53 | 1 | 0.352648 | 0.181549 |
| rho>=0 | 128 | 1 | 23.660321 | 4.646493 |
| rho>=0 | 256 | 1 | 35.877941 | 8.707559 |
| rho>=0 | 128 | 4 | 22.573686 | 2.593730 |
| 0<=rho<=0.5 | 53 | 1 | 0.384939 | 0.316177 |
| 0<=rho<=0.5 | 128 | 1 | 19.222789 | 6.773212 |
| 0<=rho<=0.5 | 128 | 4 | 17.812955 | 3.991943 |

All MPFR and boxed Float64 audits pass. Unbounded Float64 retains its existing
4.01e-6 dual-residual failure against 2e-6; low-order differences from faer are
about 8e-15 in that residual, not a corrected audit. Algorithms differ in
low-order output digits; MPFR exact accumulation is retained. Candidate
one/four-thread MPFR x/s/z agree bitwise for both bounds. Float64 one/four-thread
parity was also verified in development (its dense products remain serial SIMD).
Peak process RSS is essentially unchanged: Float64 38.39→38.45 MiB,
MPFR128 84.56→84.60 MiB, MPFR256 114.64→117.13 MiB (median peaks).

Pinned medium and Ising11 A–B–B–A preserve exact points; medium's known failure
remains, Ising passes. CSDR4 A–B–B–A and CSDR1 A–B preserve exact points and
pass all audits. CSDR4 API median increases 11.036→11.647 s in this batch;
CSDR1 changes 30.778→30.498 s. No CSDR performance improvement is claimed.

User requested six compiled solver-CLI choices: Float64 (53), 128, 256, 512,
768 and 1024 bits, while LP comparisons stay at 53/128/256. The default release
built in 3m23s. `all-precisions` retains the former full CLI dispatch; native Rust,
PMP and C ABI precision support is unchanged. The preceding full-range compile
was deliberately stopped before completion when this preference arrived.
The external gravity runner adds `--rho-ub` and `--threads`; default unbounded
inputs remain identical. Its README records the new results and audit limitation.

Evidence: `~/.cache/sdpx-e2e/gravity-local-bounds-20260928/` (release summaries,
raw points, audits, receipts and qualification logs). Timings are preliminary,
from one balanced batch; all timing runs were isolated from compilation.


### 2026-09-28 — Keep packed LP residuals and exact coupling panels

Frozen release `gravity-lp-packed-20260928-v2`, compared against the preceding
six-precision local-bound release on the same Mac. Float64 now evaluates the
complete KKT residual with packed faer products plus the remaining sparse
entries, reading the unshifted parent KKT for every diagonal/local term.
Refinement tolerances, passes, stopping rules and regularization are unchanged.
MPFR packs coupling operands for contiguous exact dot products. Scalar leaves
reuse B as Y=L^-1 B and no longer retain duplicate Y/Z buffers. The selected
Schur triangle and exact accumulation preserve MPFR points bitwise.

One serial A–B–B–A release batch per case/width; medians:

| Bounds | Bits | Threads | Native seconds, before→after | Peak RSS MiB, before→after |
|---|---:|---:|---:|---:|
| rho>=0 | 53 | 1 | 0.181813→0.093192 | 38.41→34.61 |
| rho>=0 | 128 | 1 | 4.539465→4.192009 | 84.69→83.13 |
| rho>=0 | 256 | 1 | 8.462872→5.803461 | 117.10→110.15 |
| rho>=0 | 128 | 4 | 2.646597→1.921216 | 98.40→96.93 |
| 0<=rho<=0.5 | 53 | 1 | 0.312249→0.158931 | 41.95→38.11 |
| 0<=rho<=0.5 | 128 | 1 | 6.805359→5.732973 | 102.85→93.35 |
| 0<=rho<=0.5 | 128 | 4 | 4.392771→2.984800 | 116.92→107.54 |

These are preliminary results from one batch, excluding process startup,
JSON I/O and model generation. MPFR and boxed Float64 audits pass. The original
Float64 dual-residual failure persists (4.011564697e-6 against 2e-6); changes
relative to the preceding failed point are rounding-level, not an audit fix.
All MPFR A/B points are identical. Medium, Ising11 and CSDR A–B–B–A preserve
points exactly; Ising/CSDR pass and medium's known failure remains. CSDR is also
checked at one thread with A–B. No unrelated speed improvement is claimed.

Rejected before release: residue-BLAS Schur assembly on these very tall panels.
It preserves exact points but uses substantially more memory than packed exact
dots, especially with four threads. Diagnostic binaries, receipts and the
rejected source patch are retained under `speed-v2/rns-*` and `probe-compact*`.

The default release built in 3m23s without warnings. The opt-in full-precision
CLI passes `cargo check`. The minimal library also typechecks, with dead-code
warnings for optional SDP/MPI helpers; the normal release configuration is clean.
Evidence: `~/.cache/sdpx-e2e/gravity-local-bounds-20260928/speed-v2/`.

## 2026-09-29 — Accelerate GEMM retry: kernel kept, local-bounds RNS reverted again

Second attempt at the residue-BLAS bound Schur product on the gravity LP, now
with Accelerate `dgemm` inside the exact residue kernel (replacing `gemm`
crate; the pure-Rust `gemm` dependency is removed) and three memory controls:
configurable streamed prime-group caps, a `Packed` transient operand
(3-byte residues instead of re-encoding chunk panels per group), and an
entry-split parallel packed encode (`ChunkView`) so the encode scales with
threads. Buffers above 16 MiB are no longer retained by the buffer pool.

Results on `fix-a` MPFR256 (release, serial runs): t1 3.55 s vs 5.01 s
baseline (−29%), t4 1.73 vs 1.94 (−11%), t8 1.62 vs 1.63 (≈0). MPFR128 was
flat (≈−2%). But peak RSS was 299/613/584 MiB at t1/t4/t8 vs 115/126/128 MiB
for packed exact dots — 2.6–4.9×, the same failure that rejected the earlier
candidate. The local-bounds MPFR integration is reverted; the dot path stays.

Kept: the residue-kernel changes themselves (Accelerate inner GEMM, parallel
packed encode, pool cap) since `xgemm_upper_exact` already serves the sampled
and condensed paths — `ab ising11 grav-final rns-blas-v1` is bitwise
identical, all audits pass, timing −0.3%. Float64 bound Schur assembly moved
from faer to `blas::dgemm` under `sdp` (faer remains for sdp-less builds);
`ab medium grav-final rns-blas-v1` is bitwise identical at −17.9%
(2.60→2.13 s). Gravity Float64 median 0.073→0.064 s, RSS unchanged.


## 2026-09-29 — Shared-variable SOC3 elimination: direct-support dual pilot

Kept `shared_soc_arrow`, selected structurally by automatic LDL after the
existing `local_soc_arrow` check. A column touching one SOC3 belongs to that
leaf; shared or uncoupled primal columns and equality multipliers form the
border. Cross-leaf P/Schur edges decline the route. The existing 128-coordinate
border and 512 MiB dense-workspace guards bound eligibility. Full coupling
panels reuse the existing signed dense leaf factorization, ordered exact MPFR
Schur accumulation, parallel recovery, regularization, refinement and QDLDL
fallback. No input coefficients, solver tolerances or precision were changed.
For mixed SOC/orthant inputs, the existing condensed layer eliminates orthants
before this structural check.

The direct-support g0 upper pilot has 203 primals, 7,920 SOC3 and 947 scalar
nonnegative constraints. Eliminating 160 SOC+epigraph leaves and 7,760 SOC-only
leaves leaves 43 shared primal coordinates. This extends the csdr3 idea to a
shared primal border; the csdr3 equality-border route stays unchanged.

Matched release ABBA on Apple M4/16 GiB: MPFR512, 8 SDPX threads, BLAS/OpenMP 1,
identical `plus32_retry1/conic.json` and settings (`condensed`, feasibility
1e-40, absolute/relative gap 1e-30, max_iter 180). Native seconds:

| Arm | Runs | Median |
|---|---|---|
| `shared-soc-before-20260929` | 326.119, 329.025 | 327.572 |
| `shared-soc-after-20260929` | 100.297, 102.011 | 101.154 |

**3.238× native acceleration (69.12% reduction); 3.230× whole-process**
(329.980 → 102.154 s). Median peak RSS 747.6 → 894.6 MiB (+19.7%). All four
runs `Solved` in 81 iterations; each arm's repeated x/s/z are bitwise equal.
Across arms the objective differs by 2.783e-125 and the largest componentwise
relative primal difference is 1.751e-107. All four independent 210-decimal-digit
original-input affine, primal/dual SOC and gap audits pass 1e-30. This audits
the finite conic model; it does not replace the continuum bound certificate.

Median factorization 133.575 → 37.109 s; refinement solves 90.504 → 2.860 s;
ordinary triangular solves 38.211 → 6.812 s. Refinement count 643 → 117 with
unchanged gates. Shared-border Schur assembly is now 34.861 s. These timers
are nested; do not add them to estimate total wall time.

Checks: 15 Arrow unit tests pass, including new Float64/MPFR512 original KKT
residual and 1/4-thread parity checks, a 512-bit complete condensed SOCP with
known optimum and QDLDL comparison, and cross-leaf structural-zero rejection.
Pinned Ising11 MPFR512 audit passes (`Solved`/52). CSDR3 MPFR256 1/4-thread
audits pass (`Solved`/57), with bitwise identical points. `git diff --check`
passes. Existing worktree changes were preserved in both frozen arms.

Evidence and frozen arms: `~/.cache/sdpx-e2e/shared-soc-20260929/` and
`~/.cache/sdpx-e2e/arms/shared-soc-{before,after}-20260929/`. Input, settings,
binary and source hashes are retained. User-facing records are copied to
`massive case with real/analytical/numerical/direct_soc_rational_20260929/shared_soc_benchmark/`.
The quadrature optimum stays 2.904704694858; existing certified bounds do not
change merely from faster linear algebra.


## 2026-09-29 — Review implementation (F1–F14) and cluster checks

Implemented the review items in `REVIEW_AND_PLAN.md` and tested them on the
cluster (`~/projects/sdpx-review-20260929/`, PBS 221949–221978; source tarball
`~/.cache/sdpx-e2e/review-20260929.tar.gz`).

- **Provider finding.** Static `openblas-src` (`sdp-openblas`) is not safe for
  concurrent BLAS calls from the solver pool. With it, nine Float64
  pooled-equivalence tests fail on committed HEAD (release
  `820a44dd…`), and three residue-BLAS tests fail on the working tree, whose
  kernel calls `blas::dgemm` from pool threads. A mutex around that `dgemm`
  made the residue tests pass. With the pinned dynamic provider
  (`~/projects/sdpx-medium-20260921-dense01/providers`, `-l dylib=openblas`,
  features `sdp,blas-src,lapack-src,faer-sparse`) every suite passes: lib 446,
  `it` 195, ffi 13, pmp 3, arithmetic 36 (0 failed).
- **F5 (`tol_dual_qnorm`).** Medium, Float64, 1e-6, 1 thread: default
  `Solved`/18, audit r_d 1.916e-6 > 1.75e-6 (unchanged); with
  `tol_dual_qnorm=1e-6` `Solved`/19, r_d 6.14e-7, gap 2.41e-7, r_p 2.3e-8,
  audit passes. The pinned `medium` case now sets the option.
- **F6 rejected.** Curve search enabled at MPFR512 on ising11 (t1, taskset,
  ABBA ×2): 51 vs 52 iterations; native 29.26/26.94 s vs 27.53/26.79 s
  (medians 28.10 vs 27.16, +3.5%). Objectives equal to all printed digits. One
  saved iteration cannot cover the extra step-length trials; not kept.
- **F7 restart.** `--checkpoint runs/ising.ckpt --checkpoint-every 10` on
  ising11: `Solved`/52, 27.0 s, same objective, 413 KB file. `--restart` from
  the iteration-50 file: `Solved`/2, 1.6 s, same objective digits.
- **Unit tests added:** `checkpoint::{restart_from_checkpoint_solves,
  restart_rejects_different_problem}`, `pmp_solve::readme_pmp_solves_to_one`,
  `crates/pmp/tests/convert.rs` (3).
- Timings above are single quiet runs on a shared node and carry no
  performance claim beyond the F6 sign.


## 2026-09-29 — Exact integer multiplier in sampled RHS quadratics (kept for memory)

`rns_blas::svec_quadratic` now encodes an exact off-diagonal multiplier of 2
as the integer 2 with exponent zero. Previously the common factor
2^(P-2) in the integer quadratic was carried through the residue products
and CRT before cancellation by the final exponent. Removing it reduces the
CRT bound by P-2 bits without changing the exact expression or working
precision. Other multipliers retain their existing path. No new settings.

Quick check: fast-profile Ising11 MPFR512, four threads, `Solved`/52, audit
passes and x/s/z equal the preceding audited run. Evidence:
`~/.cache/sdpx-e2e/runs/20260929-122151-573866-ising11-rhs-integer-two-fast-20260929/`.
The rebuilt Linux release also passes Ising11 (`Solved`/52).

One release ABBA on Lambda19 MPFR768, four physical cores on node100
(AMD EPYC 7742), dynamic OpenBLAS with one BLAS thread. Same input/settings
hashes and CPU slots throughout. All four runs `Solved`/119, all original-
coordinate audits pass, and full x/s/z vectors are identical across arms.

| Metric (median) | Baseline | Candidate | Reduction |
|---|---:|---:|---:|
| Native solve time | 431.440 s | 428.168 s | 0.76% |
| API time | 431.487 s | 428.212 s | 0.76% |
| Whole process | 433.117 s | 428.955 s | 0.96% |
| Peak process-group PSS (1 s sampling) | 916.105 MiB | 897.039 MiB | 2.08% |
| Peak solver RSS (`wait4`) | 906.693 MiB | 890.703 MiB | 1.76% |

Native rows in ABBA order: 432.1846, 428.4468, 427.8884, 430.6956 s.
Both candidate memory peaks are below both baseline peaks. Kept for the
clear memory benefit; timing does not meet the 2% speed-improvement gate.
No additional suites or repeated ABBA runs were needed.

Evidence: `hpc:~/projects/sdpx-sampled-rhs-20260929/evidence-df773e72/`
(`final-report.json`, per-run audits, resource receipts, frozen binaries).
Frozen source archives and manifests are in the parent directory. Baseline
binary SHA-256 `c295295a61b95595415b9d3d4580c130cc9e10167a923813880646ec854dae5f`;
candidate `2eb108f56e047965063f4a4dabd93b451658e5a377f4f994214782a3132d83e5`.

Harness repairs: PBS 222003 stopped before solving (missing affinity env);
a copied-target timestamp issue had also reused the baseline binary for the
candidate. Forced source-mtime invalidation produced a distinct rebuilt
candidate before measurement. PBS 222004 completed Ising11 and Lambda19 A1,
then its legacy Julia audit environment failed to load JSON. PBS 222013
reused the saved A1 solve, audited with the working environment and completed
B1/B2/A2 on the same node/slots (exit 0). Failed logs are preserved; no failed
or stale-binary run contributes a timing row. Parent took over monitoring
when the user removed mandatory Luna delegation.

### 2026-09-30 — Experimental supernodal backend: first measurements

`direct_solve_method = "supernodal"` (opt-in; arrow backends stay the default).
Cluster node, taskset 0-3, pinned dynamic OpenBLAS, single runs (preliminary),
same settings as auto except the backend. All runs `Solved` with the same
iteration counts as auto; objectives agree (csdr3 within its 1e-8 tolerance).

| Case | auto (backend) | supernodal v1 | supernodal v2 |
|---|---|---|---|
| ising11 MPFR512 t1 | 25.08 s (condensed_sampled_arrow) | 24.47 s | 24.67 s |
| gravity boxed MPFR128 t4 | 3.24 s (local_bounds_arrow) | 10.08 s | 14.32 s |
| gravity boxed MPFR256 t1 | 14.97 s | 29.42 s | 36.60 s |
| csdr3 MPFR256 t4 | 17.71 s (local_soc_arrow) | 117.49 s | 108.12 s |

v1: fundamental supernodes, AMD order. v2: adds quasidefinite order
(zero-cone multipliers last), etree postorder, relaxed amalgamation, one
panel arena and in-place level-parallel solves. Receipts (SDPX_RECEIPT):
the dominant difference is refinement, not factorization — corrections
(`ir.solve` count) gravity 231 vs 26, csdr3 1081 vs 171, identical in v1 and
v2, so the reordering did not change solve accuracy. Relaxed amalgamation
doubled gravity panel entries (277,901 → 532,950) and slowed it. Open
question before any speed work: why the arrow solves reach the refinement
tolerance with far fewer corrections. Evidence: cluster
`~/projects/sdpx-review-20260929/sn/` (PBS 221989, 221990, 221992).

## 2026-09-30 — Bootstrap-readiness fixes (review items 3–7)

Cluster evidence under `~/projects/sdpx-review-20260929/` (PBS 222073–222083).

- **Static OpenBLAS (fixed, observed root cause).** `openblas-src` builds
  OpenBLAS single-threaded (`USE_THREAD` off, strings `SINGLE_THREADED`)
  and without `USE_LOCKING`; such a build is not safe to call from several
  threads, and the solver calls BLAS from its pool. `.cargo/config.toml` now
  sets `OPENBLAS_USE_LOCKING=1` for the build script. The cache key changes
  (`~/.local/share/openblas_build/23c72b552828359b`; seeded offline from
  the cleaned 0.3.32 source after reproducing the old key `eab3da505301bad2`
  with the same hash). With it, the static `sdp-openblas,faer-sparse` build
  passes the full lib suite (448/0; 12 failures before). The g0 release
  binary links the dynamic provider and was not affected.
- **Feasibility-mode stopping (not implemented, evidence).** In the HSD
  embedding the residuals fall with the gap: ising11 pres ≈ gap/10 and
  dres ≈ gap/2 at every iteration; primal feasibility at 1e-42 arrives at
  iteration 51 vs optimality at 52. SDPB's early stops rely on the
  infeasible-start "jump" to exact feasibility, which HSD does not have.
- **MPFR `condensed_graded` (kept as designed).** Applying H through R at
  MPFR (two congruences, as binary64) passes the graded test but costs
  +5.5% on ising11 (A–B–B–A t1: 24.60/24.53 vs 25.96/25.95 s, identical
  iterations and objective). The rounded G only loses entries below
  eps·‖G‖; IPM grading at MPFR (≈1/μ², ~1e84 at 1e-42) stays far above eps
  (~1e-154 at 512 bits). MPFR keeps G; the test stays ignored with this
  reason in its comment.
- **Checkpoint/restart (fixed) and hot start (new).** Files carry a
  structure fingerprint (dimensions, A/P patterns, cone list, sampled block
  shapes) that must match, a value fingerprint, and the writer's
  equilibration. Same values → exact internal continuation; different values
  → hot start through original coordinates into the new scaling. Writes now
  happen only after the progress checks accept the iterate. ising11: exact
  restart from iteration 50 → `Solved`/2, identical digits. Hot start on a
  nearby problem (every sampled `c`, `B` perturbed by relative 1e-4;
  objective moves 4.068 → 5.591): cold 120 it / 60.3 s; from the original's
  iteration 10/20/30/40/50 checkpoint: 104/88/75/88/119 it, 52.3/43.2/38.2/
  42.9/57.6 s. Hot-start from a mid-trajectory checkpoint (−37% at
  iteration 30); a converged iterate is too close to the boundary to help.
- **Normalized 2×2 PMP (diagnosed; fix needs approval).** Converges
  100×/iteration to gap 4.1e-42 at iteration 23; iteration 24 takes a
  combined step of 5.1e-36 (small-step exit, `AlmostSolved`). Not the pivot
  order (zero-cone rows last: same), not the prepared combined RHS (forced
  unprepared path: same). `SDPX_TRACE_IR` at that iteration: the first solve
  leaves residual 3.58 against ‖b‖ 1.05 and refinement diverges to 3.7e37;
  with `dynamic_regularization_enable=false` the same solve starts at 1e-38
  and refines to 1e-40, and the problem is `Solved`/24. Also `Solved` with
  static regularization off, equilibration off, or 768/1024 bits. Smaller
  replacement pivots (δ = 1e-100, 1e-110, 4.3e-116) do not help: replacing a
  legitimately tiny/wrong-sign dual pivot at all breaks the factor. On
  ising11 and csdr3 dynamic regularization never fires (δ changes leave the
  points bitwise identical). Proposed, pending approval (regularization
  contract): when refinement diverges after a dynamic-regularization
  replacement, refactor once without replacements.
- **MPI.** The loader now accepts the MPICH ABI (MPICH, Intel MPI, MVAPICH:
  fixed integer handles detected via `MPI_Get_library_version`) as well as
  OpenMPI, and prefers the library family of the launcher. Intel MPI 2021.3
  (cluster) gives points bitwise identical to OpenMPI in both partitioned and
  ordinary mode (ising11, 2 ranks × 2 threads). MPI points differ from serial
  ones in both implementations (pre-existing; serial 1- and 2-thread points
  agree). Λ19 768-bit, 8 iterations: 1 node 1×64 threads 10.01 s; 2 nodes
  16×8 OpenMPI/TCP 10.48 s; Intel MPI over InfiniBand (`verbs;ofi_rxm`)
  11.54 s. The second node is limited by the 28-block critical path and
  synchronizing rounds, not by transport latency; a multi-node gain needs a
  problem with many more blocks (Λ43).
- **Removed experiments (simplification).** `SDPX_RNS_OPS` (opt-in residue
  dots for sampled products) measured slower: ising11 t1 +8% (24.57 →
  26.56 s), Λ19 8-it t4 +27% (36.37 → 46.19 s); removed with its `Scalar`
  trait plumbing and tests (the default path also stops building an unused
  `qq` table). The experimental supernodal LDLᵀ is removed (slower on every
  measured case but a tie on ising11; see the 2026-09-30 entry above). All
  suites pass after removal (lib 442, it 196, arithmetic 36, pmp 3, ffi 13);
  workspace, tests and no-sdp builds warning-free; rustfmt clean.
- **Λ19 profile (current source, 768-bit, 4 threads, 8 it, solve 33.6 s).**
  KKT update 11.7 s wall (refinement residual 5.8, RHS recover 4.3, prepare
  2.7, factor 2.5 cpu-s); cone scaling 8.6 s wall (SVD 29.2 cpu-s, rotation
  replay 18.1 = 62%); KKT solve 4.7 s; default start 4.5 s. The replay is the
  largest single kernel (~15% of CPU).

## 2026-09-30 — Review fixes: replay, checkpoints and pool budget

- **Rejected fixed-point SVD replay.** Its absolute P+64-bit scale erased
  rotations of 1e-70 at 128 bits and 1e-200 at 512 bits. Removed the prototype
  and restored MPFR replay without changing tile order or parallelism.
- **Kept checkpoint geometry validation.** Power exponents, generalized
  exponents and tail dimension now belong to the structure fingerprint.
  The reproduced incompatible restart is rejected before cone scaling;
  changing only the generalized exponents is also rejected.
- **Kept checkpoint κ rescaling.** Hot starts use κ_new = κ_old·c_new/c_old;
  exact continuation copies κ unchanged. A box QP with c_old=0.04714045 and
  c_new=1 preserves the scaled embedding residual (error 6.1e-15). Both
  restart modes reach `Solved`/3 and pass the original-coordinate audit.
- **Kept explicit-pool residue merging.** Encoding and merging execute in
  the supplied pool. A 45×45 exact congruence at 512 bits verifies its output
  with two pool threads while `RAYON_NUM_THREADS=6`; the global pool remains
  uninitialized and can subsequently be configured.

Quick verification: fast-profile ising11 at 512 bits, one and two threads,
both `Solved`/52 with original-coordinate audits passing. x/s/z and sampled
y are identical across threads and to the frozen `rhs-integer-two-fast-20260929`
baseline. No formal performance claim.

Evidence: `~/.cache/sdpx-e2e/arms/review-fixes-20260930`, runs
`20260930-133557-085665-ising11-review-fixes-20260930` and
`20260930-133730-094390-ising11-review-fixes-20260930`; targeted reproducers
and outputs under `~/.cache/sdpx-e2e/review-fixes-20260930`.

## 2026-09-30 — Refactor round: scratch sharing, residue reuse, profiling

Branch `refactor/hotpath-20260930` (worktree
`~/.cache/sdpx-e2e/refactor-20260930/source`), based on snapshot commit
`ff04e7c` of the shared checkout's uncommitted work at 13:45 (another agent was
editing it). Cluster namespace `hpc:~/projects/sdpx-refactor-20260930/`.

Profile (Λ19 MPFR768, 8 it, 4 threads, node100). `perf`: GMP basecase multiply
26%, OpenBLAS residue GEMM 19%, RNS encode/CRT ~13%, SVD replay `fmma` 5%.
Callgrind (ising11, 512 bits, 1 thread): `exactdot::accumulate` 40% of
instructions. Massif (Λ19, 2 it, peak 755 MiB): cached residues 193 MiB (basis
adjoint 50, |basis| 49, Schur V 39, congruence 34, forward 29), retained RNS
buffer pool ~108 MiB, per-block PSD/sampled scratch and duplicate copies.

Kept (all bitwise identical to `ff04e7c` on medium t1/t4, csdr3 t4, ising11
t1/t4 and Λ19 8 it; lib 442, it 196, arithmetic 36, pmp 3, ffi 13 pass):
- One contiguous lane partition shared by cone and scaling schedulers.
- PSD cones keep only R, Rinv, G, λ, Λisqrt; Cholesky/SVD/eigen engines and
  work matrices are per-thread scratch (`algebra::scratch::with_scratch`).
- Condensed PSD blocks share per-thread congruence panels.
- |Q|ᵀ|X||Q| (componentwise dual residual) reads Q's cached residues with
  sign flips (balanced residues are odd); the |Q| copy and cache are gone.
- Residue-BLAS gate lowered from 40³ to the EPYC break-even 24³ (serial square
  GEMM: 1.14× at 512 bits, 1.20× at 768, 1.50× at 1024 at 24³; 1.7–2.4× at 40³).
  No pinned case has PSD orders 24–39, so this is not an end-to-end claim.
- Sampled Schur workspaces read the operator basis unless duplicate columns
  were merged; `product` and the sampled `panel`/`square` are per-thread.
- Unread Ginv of fused sampled blocks released (no per-iteration SYRK).
- Forward and adjoint sampled products share one basis residue encoding
  when their inner dimensions use one prime width (as V already did).
Λ19 8-it peak RSS (`/usr/bin/time`): 874 → 729 MB at `1a003f2`, 690 MB at
`19d73d9`, 668 MB at `128ddb0`, 652 MB (−25.4%) after releasing the
condensed blocks' R at MPFR precision (read only by binary64), 644 MB after
skipping per-nonzero PSD entry records in structure-only builds (76 MiB at the
setup peak), 628 MB (−28.1%) after dropping the unused A pattern in fused
sampled solves (`update_A`/`update_P` pattern checks became debug assertions);
native time unchanged (36.0–36.9 s). A second massif (1 it, `128ddb0`) put
setup and iteration heaps level at ~570 MiB; RSS tracks the heap high-water
because freed heap memory stays in the process, so pool-cap trimming barely
moves RSS.

Full Λ19 release A–B–B–A (MPFR768, 119 it, 4 pinned cores, node58, base
`ff04e7c` vs `19d73d9`+Ginv skip): all four `Solved`/119, points bitwise
identical across arms. Native 436.22 / 435.58 / 435.46 / 434.41 s (medians
435.3 vs 435.5, neutral). Peak RSS 907.5 / 718.9 / 720.5 / 911.8 MB (medians
909.7 → 719.7, −20.9%). Original-coordinate audits accepted for A1 and B1
(primal 4.8e-77, dual 6.3e-74, gap 5.6e-75, reference agreement 1.7e-42) and
ising11 t1 (2.3e-45 / 4.3e-44 / 1.7e-43). PBS 222144, 222147.

Final full Λ19 release A–B–B–A (same settings, node100, base `ff04e7c` vs
`40a7f3c` (head11)): all `Solved`/119, points bitwise identical.
Native 437.05 / 434.02 / 436.40 / 436.40 s (medians 436.7 vs 435.2, neutral).
Peak RSS 912.2 / 659.1 / 660.2 / 912.6 MB (medians 912.4 → 659.7, −27.7%).
Audits accepted for A1 and B1. Arithmetic 36, pmp 3, ffi 13, lib 442, it 196
pass; snapshot, no-sdp and all-precisions checks are warning-free. PBS
222157, 222158.

Also rejected: lowering the RNS buffer-pool caps to 8 MiB/thread, 32 MiB
total — Λ19 peak RSS 736 → 730 MB only; not worth the allocation churn risk.

Rejected: LAPACK dbdsqr bulge-chase direction for the MPFR bidiagonal QR.
Rotation counts changed −0.6% (Λ19) / +0.2% (ising11) and points differ; the
IPM bidiagonals already converge at the end the top-down chase targets.
Reverted. `ExactRows` negation-copy removal rejected: flips the sign of exact
zero residuals, breaking parity, for a ~1.2k-value allocation.

## 2026-10-01 — Float64 hot paths and exact MPFR kernels

Same branch; cluster PBS 222162–222176 (node100 unless noted), gravity on the
Apple M4. Every kept change passed lib (443) and integration (196) tests.

**Float64 vs MOSEK (medium, qnorm settings).** Same node, one core: MOSEK
11.2.2 product defaults 7.96 s / 21 it; SDPX 8.04 s / 19 it. The earlier
1.89 s MOSEK figure was an M4 measurement. Kept, in order (same-node A/B):

| Change | t1 | t4 | Points |
|---|---:|---:|---|
| base | 8.20 | 5.61 | — |
| FMA wrapper on packed Schur + pooled dot lanes (libm `fma` was 11%) | 6.92 | 4.99 | identical |
| stamp-and-sort Schur pattern, scattered positions (BTreeSet 4%) | 6.20 | 4.38 | identical |
| pooled tiled Cholesky, 128 tiles, one BLAS call per tile | 6.22 | 3.70 | t4 differs (t1 keeps dpotrf) |
| per-column support routing in transform, 8x4 FMA kernel | 5.93 | 3.55 | differs at rounding level |
| pooled dot lanes reuse scratch (memset 9.8% at t4) | 5.93 | 3.50 | identical |

Objectives agree to 1e-15 relative, residuals equivalent. Rejected with
same-node evidence: 32-column blocked publish (store 1.03 → 2.52 s),
dense-axis chunked dots (FMA phase 0.29 → 0.41 s; A is 0.3% dense, so the
dots are not bandwidth-bound and a dense GEMM would do ~100x the flops),
transpose-before-pack (transform 1.75 → 1.91 s). THP is `always` on the
nodes, so no madvise work.

**Exact kernels (bitwise identical by construction).** `exactdot` paid three
GMP calls per term; for N <= 4 limbs the schoolbook product and shifted add
are inlined and the sign accumulator is chosen branchlessly (8 limbs keep
GMP: 30 → 38 ns/term inline). `Scalar::dot_slices` scans contiguous operands
without collecting terms. Regular x regular products up to 256 bits round
inline to nearest-even (40k-case oracle vs `mpfr_mul`, ties and carry-out).
New tests: inline-vs-GMP accumulation equivalence (N = 1..4, carry stress),
slice-path oracle checks. The existing `exact_product` oracle returns 0 for
300 all-ones-mantissa terms at N = 2 (both dot paths agree on ~300); not
investigated further, the equivalence test covers that input.

Gravity fixed-a (M4, medians of three): MPFR128 t1 2.713 → 1.553 s, MPFR256
t1 5.117 → 3.224 s, t4 1.938 → 1.150 s, t8 1.683 → 1.008 s; x/s/z identical.
The bound Schur Gram is now ~8.9 ns/term at 256 bits (about 2x the multiply
floor); the residue route stays rejected for memory. Cluster: csdr3 MPFR256
t4 17.23 → 15.12 s, ising11/Λ19 (N > 4) unchanged within noise; all MPFR
points identical to base.

## 2026-10-01 — Exact arithmetic kernels, residue bound Gram, SDP-mandatory refactor

Same branch. Mac M4 gravity runs; cluster PBS 222198 (node100) for the
pinned cases. Every listed change passed lib, integration and (where
touched) arithmetic, FFI and PMP tests.

**Inline arithmetic (bitwise identical).** Correctly rounded nearest-even
multiply after the limb product (schoolbook <= 4 limbs, GMP `mpn_mul_n`
above) beats `mpfr_mul` through 19 limbs; inline add/sub (exact 2N+2-limb
buffer, gap >= P+2 returns the larger operand) wins through 8 limbs. Oracle
tests compare every inline width against MPFR, including ties, carry-out and
cancellation. PBS 222198: csdr3, ising11 t1/t4 and Λ19 points identical;
ising11 24.70 → 23.63 s (t1), 7.47 → 7.05 s (t4); csdr3 t4 17.23 → 16.03 s.

**Residue bound Gram (rounding-level change).** The local-bounds Gram is the
exact `Yᵀdiag(d)Y` rounded once (previously `Z = round(Y·d)` then an exact
`ZᵀY`). Y's residues are cached for the solve (3-byte store, built in
32k-entry blocks), only d is encoded per iteration, rows are scaled and
multiplied in 4096-row blocks, one prime per group, d's spread reserved at
160 bits so the cache is built once (measured spread 0 → 116 bits over the
solve). Gravity fixed-a: MPFR256 t1 3.57 → 2.39 s (RSS 115 → 179 MiB),
MPFR128 t1 1.72 → 1.29 s (88 → 134 MiB), MPFR256 t4 1.41 → 1.28 s (126 →
261 MiB; per-thread rns_blas buffer pools). Same iterations; independent
audits pass with unchanged residuals and gap.

**Refactor.** Removed the scalar RNS dot (`arithmetic/rns.rs`, 1068 lines):
MPFR xgemm/xsyrk fall back to the exact dot, identical bits at equal speed.
SDP support is mandatory: the `sdp` feature and its 327 cfg sites are gone,
BLAS/LAPACK always link (providers: `sdp-accelerate/openblas/mkl/netlib` or
RUSTFLAGS), `FloatT = Scalar + BlasFloatT + LDLConfiguration`. Removed the
never-called `DirectLDLSolverReqs` trait, unused `norm_one_scaled` and
`core_mut`, duplicate `tri`/`permute` helpers. Left for later (plan):
`timers` → `receipt` and the distributed/serial HSD loop merge.

**Follow-ups (same day).** Row-parallel `Aᵀdiag(d)A`: ways own leaf-row blocks
(k/ways, 256..4096 rows) and run every prime, adding partial residues mod p
(exact, split-independent). Gravity MPFR256 t4 RSS 274 → 202 MiB at equal time;
points identical. Sub-phase probe: per-prime loop 80% of the Gram, cache
check and d encode 16%, CRT 1%; near the floor of this method. Rejected: a
two-pass exact dot over cloneable iterators for `ExactRows::residual`
(refinement 0.70 → 0.79 s; rescanning scattered row gathers costs more than
collecting term pointers). Refinement passes: 19 per 78 solves on gravity
MPFR256; the time is the contract-required initial exact residual, so no
change. Gravity Float64: the opt-in `tol_dual_qnorm=1e-6` passes the audit
(dual residual 4.0e-6 → 7.3e-9, objective agrees with MPFR to 7e-8, 17 → 19
iterations); the gravity runner sets it, the solver default is unchanged.
The remaining Float64 gap to MOSEK (0.075 vs 0.043 s) is three refined KKT
solves per HSD iteration. PBS 222213 (refactored head): csdr3, ising11 t1/t4
and Λ19 points identical; lib 444, integration 201 passed.

Gravity fixed-a after this round (Apple M4, medians of three, load ~3):
Float64 0.075 s (audit passes), MPFR128 1.248 s, MPFR256 2.236 s (1 thread),
1.045 s (4), 1.001 s (8); peak RSS 130/175 MiB (MPFR128/256, 1 thread).

## 2026-10-01 — Timers/receipt merge, MPI dispatch, Float64 hot spots

`Timers` is a solve clock (setup + solve start); `timeit!` records only
receipt phases; the receipt's `setup_seconds_inclusive` holds the setup
total. The MPI partitioned solver already ran the shared core HSD loop; its
16 pooled/serial duplicate dispatch sites became `for_blocks!`/`all_blocks!`.
PBS 222216: ising11 under 2-rank Intel MPI (`--partitions auto`, `2`) and
in-process `--partitions 2` give points identical to the previous build
(MPI vs in-process already differed in the last bits before the change).
PBS 222217/222218: csdr3, ising11 t1/t4, Λ19 identical; tests pass.

Float64 medium profile (PBS 222214, t1): transform 31%, dense Cholesky 22%,
Schur store 18%, dot 5%. Kept (identical points): u32 Schur positions and
upper-triangle-only clearing with reused tiles in the dense block (RSS 353 →
318 MB t1, 579 → 539 MB t4 on node100; t4 3.60 → 3.48 s).

Gravity Float64 (in-process sampling of repeated solves): four-column
lockstep `Aᵀx` (each column keeps its term order) and branch-free f64
finite/inf-norm scans (identical points): 0.073 → 0.062 s; the coupling
panel products through BLAS instead of faer matvecs (rounding-level; same
19 iterations, audit and residuals unchanged): → 0.048 s quiet, 0.054 s at
load ~3. Remaining spread: leaf factor/solves ~23%, refinement residual 9%,
value updates 7%, sparse products 12%; an SoA leaf layout is the next lever.

**Later the same day.** Flat leaf factors for the Float64 bound solve and a
forward column scan in `update_entries` (identical points): gravity Float64
0.048 → 0.044 s. Λ19 profile (PBS 222220): GMP basecase 26%, residue-kernel
dgemm 16.5%, CRT/encode/memset ~15%; kept the uninitialized wide-multiply
buffer and no clearing before β=0 residue products, and narrow `fmma` and SVD
rotations through the exact dot (57 vs 122 ns at 4 limbs; oracle-tested).
Rejected: inlined shifted add above 4 limbs (12 limbs 63 vs 56 ns/term),
exact dots for `fmma` above 4 limbs (equal). PBS 222222/222223: MPFR cases
identical; csdr3 t4 16.1 → 15.1 s. Gravity table (M4, load 3–6): Float64
0.046 s, MPFR128 1.192 s, MPFR256 2.208 / 1.384 / 1.044 / 1.007 s at 1/2/4/8
threads.

## 2026-10-01 — Review corrections

Kept correctness fixes: narrow FMA2 declines to native `fmma` (one rounding),
zero operands retain MPFR's zero sign, and inline add/sub/multiply honor the
current MPFR exponent limits. The shared solve loop freezes its timer at
completion; pooled dense factors report their selected worker count.

Fast arm `review-fixes-20261001`: Ising11 MPFR512, 4 threads, `Solved`/52,
original-coordinate audit PASS; x/s/z/sampled_y identical to `review-20261001`.
The 128-bit reproducers match MPFR for the wide-exponent FMA2/SVD rotation,
signed zero, overflow and underflow. A small dense QP passes its analytical
solution check (max x error 2.83e-15), reports 4 workers, and its completed
timer stays fixed across an idle delay. No release timing claim.
Evidence: `~/.cache/sdpx-e2e/review-fixes-20261001`.

## 2026-10-02 — Larger gravity bound LP

`NN=100, nn=200, jj=200`: 20,202 variables, 99 equalities, 20,200 rho
leaves and a 101-column border. The Gram work grows about 15.5× from
`NN=50, nn=100, jj=100`; local positivity/box bounds already eliminate
analytically. Removed the 128-column admission cliff and charged packed
storage correctly against the dense-storage budget plus input storage.

One packed Y now owns the couplings; leaves no longer duplicate B, and the
rounded Z panel is allocated only if the exact residue kernel declines.
This removes two rho × border MPFR panels (186.8 MiB at this size/256 bits).
Constant-Y residue metadata is reused and invalidated on coupling writes;
upper Gram tiles skip redundant products and use the existing scratch pool.
Unrolled 2/3-variable signed LDL keeps the generic pivot/FMA order,
regularization and refinement. No precision or stopping-rule change.

Quick checks only: boxed gravity MPFR256 is `Solved`/33, x/s/z identical
to the frozen baseline and between 1/4 threads; original-coordinate audit
PASS. A Float64 129-column border is `Solved`/6 with an audit PASS and
selects `local_bounds_faer` (old: QDLDL). The runner's small boxed MPFR128/256
solves pass and share one serialized MPFR input; per-precision settings,
full points and original-coordinate audits are saved. Its coarse unboxed
model is genuinely `DualInfeasible`, identical to the baseline; not a pass.

**Kept:** PBS 222308 release ABBA on node9, MPFR256, 4 physical cores,
BLAS 1, frozen input and source manifests, validated dynamic OpenBLAS.
Native medians 29.769 → 27.836 s (−6.5%); API 29.776 → 27.843 s;
process 32.416 → 30.399 s (−6.2%). GNU-time solver-process peak RSS medians
1223.9 → 1087.8 MiB (−11.1%). All four runs are `Solved`/33 with identical
x/s/z; original-coordinate primal 3.28e-19, conic 3.40e-19, dual 1.50e-18
(gate 2e-18), gap 2.23e-19: PASS. The larger boxed case now selects
`local_bounds_arrow`, `Solved`/59, native 56.820 s, peak RSS 1149.8 MiB,
audit PASS (primal 1.28e-22, conic 1.48e-22, dual 1.47e-68, gap 6.89e-18).
Requested resources: 8 cores/32 GiB/30 minutes; job completed in 12:06,
exit 0. No precision/core sweep or broad test suite.
Receipt bound-Schur medians: 13.821 → 12.257 s; this remains 44% of native
solve time and is the next measured target for this larger model.

PBS 222307 reused the baseline executable from a shared Cargo target; its
comparison is invalid and was stopped. The repair isolates candidate
artifacts, cleans only the copied solver package and requires distinct
binary hashes before timed runs. Baseline binary `3be2c514…`, candidate
`68ab3d8c…`; complete hashes are in the result summary. Cache ownership
comments were clarified after the build; no subsequent executable change.
Evidence: `~/.cache/sdpx-e2e/gravity-improve-20261002`; remote
`~/projects/sdpx-gravity-improve-20261002/results-corrected`.

## 2026-10-02 — Balanced gravity Gram work and compact Float64 bound solves

**Kept:** exact bound-Gram row blocks now give every worker the same number
of cache-sized blocks. For 20,200 rows/four workers, five 4096-row blocks
became eight 2525-row blocks; this removes the worker with twice the work.
Modular accumulation and rounding are unchanged. Float64 flat leaves retain
only strict-lower factors and immutable indices; forward/backward 2/3-variable
solves are unrolled in the existing FMA order. No precision, regularization,
refinement or convergence change.

PBS 222316, node9, release A–B–B–A, BLAS 1, dynamic OpenBLAS. The same larger
gravity setup (`NN=100, nn=200, jj=200`) and frozen inputs/settings:

| Precision / threads | Native median, before → after | API median | Process median | Peak RSS median |
|---|---|---|---|---|
| Float64 / 1 | 2.500 → 2.357 s (−5.7%) | 2.506 → 2.364 s | 2.869 → 2.630 s | 162.8 → 163.0 MiB |
| MPFR256 / 4 | 28.015 → 23.238 s (−17.1%) | 28.022 → 23.244 s | 30.530 → 25.743 s | 1094.1 → 1085.0 MiB |

Every point is identical across each ABBA; Float64 is `Solved`/35, MPFR256
`Solved`/33. Original-coordinate audits pass: Float64 primal/conic/dual
2.34e-9/2.42e-9/1.07e-8, MPFR256 3.28e-19/3.40e-19/1.50e-18.
Local small boxed Float64 and MPFR256 solves also preserve points at 1/4
threads and pass their audits. KKT-update wall medians fall from 18.164 to
13.554 s on larger MPFR256; Float64 KKT solves fall from 0.547 to 0.494 s.

**Rejected:** saving one MPFR SVD rotation result through input/output aliasing.
Ising11 MPFR512/four cores ABBA: 7.178 → 7.118 s native (−0.84%), peak RSS
72.9 → 74.3 MiB. Points are identical, `Solved`/52, audit passes; the gain is
below 2% with no memory benefit. Restored only this trial's arithmetic edit.

Job exit 0. Evidence: `~/.cache/sdpx-e2e/gravity-ising-20261002-v2/cluster-results/verified-summary.json`;
remote `~/projects/sdpx-gravity-ising-20261002-v2/results`. The subsequent Ising trial is recorded below.

**Rejected:** analytical MPFR 2×2 terminal SVD (LAPACK DLASV2 ratios) plus
contiguous slice dots. Local Ising11 one/four-thread solves pass the audit;
a temporary 512-bit pair check passes signed/rank-deficient/wide-exponent
reconstruction and a 1024-bit value oracle. PBS 222323 release ABBA on node9:
7.196 → 7.180 s native (−0.21%), API 7.196 → 7.180 s, process 7.336 → 7.246 s.
RSS 73.6/76.9 MiB baseline versus 74.2/74.4 candidate gives no repeatable
memory benefit. All `Solved`/52; each arm repeats bitwise. The candidate
matches its locally audited point exactly; primal/dual/gap 2.34e-45/4.29e-44/
1.70e-43, PSD violations zero. Algorithm points differ from baseline by at
most 3.83e-87 scaled. Reverted the entire trial and its added license text.
Evidence: `gravity-ising-20261002-v2/contiguous/cluster-results/verified-summary.json`.

PBS 222324 profiles the existing Ising11 release on four cores: 12k user-cycle
samples, none lost. GMP basecase multiplication is 38.1%, limb shifts 7.4%.
This identifies small exact matrix products as a more useful next target
than the rejected scalar SVD edits. Evidence: the campaign's `profile/`.

**Rejected:** cached MPFR congruences at orders 12–23 (>=512 bits) through the
existing exact residue kernel. One small four-thread Ising11 E2E passes the
unchanged audit. PBS 222325/node9 release ABBA: native 7.194 → 7.229 s
(+0.50%), API 7.194 → 7.230 s, process 7.340 → 7.291 s; peak RSS 74.5 →
77.3 MiB (+3.73%). All `Solved`/52, repeats identical within each arm;
candidate point equals its locally audited point exactly, with the same
accepted original-coordinate residuals. Restored the original dispatch.
Evidence: `gravity-ising-20261002-v2/cached/cluster-results/verified-summary.json`.

Final fast arm `gravity-ising-20261002-kept` contains only the two accepted
gravity source edits relative to this task's baseline. Ising arithmetic,
SVD and matrix-product dispatch are restored byte for byte. All three Ising
trials are closed on this input: no >=2% complete-solve gain or clear memory
benefit. No large Ising claim, precision sweep or broad test suite. Jobs
222316/222323/222324/222325 completed with exit 0; all frozen source, input,
point and timing evidence stays in the campaign directories.

## 2026-10-02 — Gravity Float64 pooled BLAS and MOSEK comparison

**Kept:** split the packed Float64 bound Gram and forward/adjoint panel
products into disjoint BLAS tiles on the existing solver pool. Refinement
uses the same products against the original unshifted KKT. Keep the full
BLAS calls for serial execution and small products. No precision,
convergence, regularization or refinement changes; no new settings.

Local fast E2E: the existing NN50/nn100/jj100 Float64 input at one/four
threads, `Solved`/19, identical points within each candidate, full original
audit PASS. First trial tiled the serial calls too: larger-case native time
2.374 → 2.877 s (+21.2%). Rejected that serial change and restored full
calls before the final comparison.

PBS 222327 and dependent 222328 on node9, eight physical cores reserved,
32 GiB, validated dynamic OpenBLAS with one BLAS thread, release, frozen
sources/input/settings, serial A–B–B–A within each 1/4/8-thread comparison.
Both jobs completed with exit 0. Same larger gravity input:
NN100/nn200/jj200, 20,202 variables, 99 equalities, 20,200 rho bounds,
retained border 101. Float64 tolerance 1e-6 with opt-in
`tol_dual_qnorm=1e-6`; MPFR256 1e-18. All results pass the unchanged
original-coordinate primal, conic, dual, cone and gap gates.

Final SDPX Float64 medians (seconds):

| Threads | Native before → after | API before → after | Process before → after | Peak RSS before → after (MiB) |
|---|---|---|---|---|
| 1 | 2.367 → 2.405 (+1.6%) | 2.373 → 2.411 | 2.717 → 2.676 | 161.8 → 162.7 |
| 4 | 2.666 → 1.849 (−30.7%) | 2.672 → 1.855 | 2.991 → 2.134 | 263.3 → 268.9 |
| 8 | 2.426 → 2.363 (−2.6%, near noise) | 2.432 → 2.369 | 2.760 → 2.689 | 262.6 → 270.9 |

Keep for the repeated four-thread gain (first batch −27.8%, final −30.7%);
do not claim useful eight-thread scaling. Serial point is bitwise unchanged,
`Solved`/35; parallel points repeat and match at four/eight threads,
`Solved`/37. Different BLAS kernel shapes change the Newton trajectory and
rho values: max per-component scaled x/s difference from serial is 1.40%,
z 0.76%; the first variable/bound differs by only 5.98e-8. Parallel original
primal/dual/gap are 3.29e-10 / 1.36e-9 / 2.09e-10 (serial
2.34e-9 / 1.07e-8 / 1.64e-9). No accuracy gate is weakened.

MOSEK 11.2.2 native Python API, interior point, same Float64 matrix and
1e-6 internal tolerances. Convert the verified -I nonnegative rows to
variable lower bounds; map equality/bound multipliers back and use the
same original-coordinate audit. All `optimal`/21 iterations, audits PASS.

| Threads | Optimizer (s) | Python model + optimize (s) | Process (s) | Process RSS (MiB) |
|---|---|---|---|---|
| 1 | 1.187 | 1.771 | 3.869 | 355.9 |
| 4 | 0.950 | 1.515 | 2.670 | 357.9 |
| 8 | 0.912 | 1.478 | 2.643 | 357.7 |

MOSEK optimizer time includes its preprocessing but excludes Python model
construction. SDPX native includes numerical setup; its API clock starts
after JSON parsing. Report both scopes: at four threads SDPX is 1.95×
MOSEK optimizer time and 1.22× model + optimize wall time. Process clocks
include loading/recovery, exclude the subsequent independent audit. RSS
is process peak, not a native-library-only MOSEK allocation measurement.
The four-thread first variable agrees within 3.57e-7 across solvers.

Existing MPFR256 path, two fresh runs per count: native 72.853 / 23.082 /
15.451 s at 1/4/8 threads, process 75.341 / 25.585 / 17.972 s, RSS 910.6 /
1079.6 / 1149.2 MiB. All `Solved`/33, bitwise identical across counts,
audits PASS. Eight threads give 4.72× vs one and 1.49× vs four. This is a
scaling measurement; this round changes no MPFR kernels.

Remaining Float64 costs: four-thread sample KKT update 0.952 s, refinement
0.725 s (nested), Gram 0.298 s; eight-thread sample 1.426 / 0.976 / 0.493 s.
Next bounded candidates: avoid unused generic KKT residual row tables
(exact MPFR rows and packed Float64 residuals already supersede them),
compute only the lower Float64 Gram triangle and measure coarser BLAS tiles.
Investigate scaling/conditioning behind 37 vs 21 iterations without
relaxing convergence or deleting constraints.

Evidence: `~/.cache/sdpx-e2e/gravity-threads-20261002/cluster-results/summary.json`
and `serial-preserved/verified-summary.json` (final comparison), frozen fast
arm `gravity-threads-20261002-kept`. Remote complete raw points, receipts,
source and build logs: `~/projects/sdpx-gravity-threads-20261002/` and its
`serial-preserved/` directory. Final source differs from this task's frozen
baseline only in `arrow/local_bounds.rs`; all other shared source edits
are preserved. No broad test suite, new Ising campaign or precision sweep.

## 2026-10-02 — Lower bound Gram triangle and lazy residual rows

**Kept:** Float64 bound elimination computes only the lower `YᵀZ`
triangle, using disjoint column tiles and the existing worker pool. Keep
individually rounded `Z=Y*d`; replacing it with a square-root/SYRK operator
would change the arithmetic. One private 16-column tile serves small and
larger models on both BLAS providers. No model-name, provider or requested
thread-count branches, new settings or numerical-rule changes.

Generic symmetric residual row tables are now built lazily when iterative
refinement actually uses them. Exact MPFR rows and native packed backend
residuals take their existing precedence and avoid the duplicate table.
Generic QDLDL still gets its parallel table when needed. This storage change
preserves points; Float64 triangular BLAS changes rounding and trajectories.

Quick local fast checks: NN50/nn100/jj100 Float64 at one/four threads is
`Solved`/19 with identical candidate points. MPFR256 before/after at one/
four threads is `Solved`/25, bitwise identical; a forced-QDLDL Float64
before/after pair is `Solved`/19, bitwise identical. All original-coordinate
audits pass. No routine unit/integration suite or precision sweep.

PBS 222330, node9/EPYC 7742, release ABBA, frozen NN100/nn200/jj200 inputs
and sources; eight physical cores, 32 GiB, 45-minute limit, validated
dynamic OpenBLAS with one BLAS thread. Completed in 35:22, exit 0. All
candidate and cluster baseline solves are `Solved` and pass the unchanged
original-coordinate audit. PBS 222329 exited 1 before benchmarking because
the standalone diagnostic helper omitted thin-LTO linking flags; the retry
matched the release flags. Complete points, receipts and logs are preserved.

Memory-only control, four threads:

| Precision | Native before → after (s) | API (s) | Process (s) | Peak RSS (MiB) |
|---|---|---|---|---|
| Float64 | 1.858 → 1.766 (−5.0%) | 1.864 → 1.772 | 2.200 → 2.053 | 268.3 → 208.4 |
| MPFR256 | 23.356 → 23.191 (−0.7%, no speed claim) | 23.362 → 23.197 | 25.870 → 25.673 | 1077.5 → 1016.6 |

Control points are bitwise identical, at 37/33 iterations respectively.
The retained lower triangle plus lazy rows, Float64:

| Threads | Native before → after (s) | API (s) | Process (s) | Peak RSS (MiB) |
|---|---|---|---|---|
| 1 | 2.357 → 2.827 (+19.9%) | 2.363 → 2.833 | 2.644 → 3.125 | 163.8 → 162.5 |
| 4 | 1.744 → 1.603 (−8.1%) | 1.750 → 1.609 | 2.049 → 1.876 | 269.3 → 207.8 |
| 8 | 2.140 → 2.313 (+8.1%) | 2.146 → 2.319 | 2.435 → 2.627 | 268.3 → 208.0 |

Keep for the memory benefit, four-thread gain and Mac status improvement;
serial/eight-thread EPYC regressions remain. Gram medians at four threads
fall 0.327 → 0.304 s; at one thread they rise 0.466 → 0.798 s. Computing
fewer entries does not guarantee faster BLAS execution.

The user authorized local Mac M4 comparisons at 1/2/4 threads after SSH
timeouts. Same larger inputs/settings, release ABBA, Accelerate BLAS 1:

| Threads | Candidate native / API / process (s) | RSS (MiB) | Status |
|---|---|---|---|
| 1 | 0.594 / 0.597 / 1.320 | 162.5 | `Solved`/36 |
| 2 | 0.556 / 0.558 / 0.700 | 195.0 | `Solved`/37 |
| 4 | 0.513 / 0.516 / 0.656 | 198.1 | `Solved`/37 |

All candidate audits pass; two/four-thread points match. The one-thread
process median includes a startup outlier. Frozen baseline and memory-only
Float64 both return `AlmostSolved`/32, exit 2, on this Mac input at all three
counts. Their audit is failed by status; never promote them or quote an
audited speed gain against them. The triangle candidate finishes on
`local_bounds_faer`. Mac MPFR256 memory-only comparison saves 40.0 MiB
(1030.3 → 990.3), with identical `Solved`/33 points; 27.02 → 27.71 s native
gives no speed claim. MPFR arithmetic is unchanged by the triangle variant.

**Rejected:** 8- and 32-column alternatives. On EPYC/four threads, their
matched native changes are −6.2% and +8.3% respectively, vs −8.1% for 16.
Mac four-thread candidate medians are 0.534/0.513/0.500 s for 8/16/32 in
separate batches; that small Mac preference for 32 does not justify tuning
by provider, thread count or model. Keep the common 16-column geometry.
These results cover the recorded small/larger models and MPFR256, not every
larger size or precision.

Iteration diagnosis on the frozen four-thread baseline: standard gates
first/stably pass at 26; the opt-in `tol_dual_qnorm=1e-6` needs 37. Its
original-coordinate formula is `‖r_dual‖₂/(1+‖q‖∞)`, while the external
audit uses `‖r_dual‖∞`. At 26, qnorm is 2.40e-4; at 37, 9.60e-8. Corrected
the comments only. MOSEK 11.2.2 takes 21 interior-point iterations, retains
99 equalities and removes zero dependencies; its optimizer also includes
basis identification. The last 20 refinement improvement ratios are
1.164–3.514, below the unchanged stop ratio 5. Investigate late correction
quality/scaling next; preserve convergence, regularization and refinement.

Evidence: `~/.cache/sdpx-e2e/gravity-triangle-20261002/`, especially
`local-check.json`, `cluster-results/{memory,c16,c8,c32}/summary.json`,
`mac/results/*-v2/summary.json` and `iteration-analysis.json`. Remote full
artifacts: `~/projects/sdpx-gravity-triangle-20261002/`. Input SHA-256:
Float64 `b6f6b02e…`, MPFR256 `64bcac82…`; full hashes and source manifests
are frozen there. Baseline/c16 cluster binaries `bc0f0118…` / `81b0964c…`;
Mac `a06d5760…` / `f296ff74…`. Final executable code matches the tested c16
snapshot; subsequent edits only correct the two qnorm API comments and
update documentation. Other shared source edits are preserved.


## 2026-10-03 — preceding simplification snapshot

Historical record transferred from the development plan. This predates
the storage changes below; its suite counts do not describe their verification.

Simplification round (2026-10-03, Mac M4, one private copy of the then-current
working tree; evidence in the journal): production code 46,625 → 45,540
lines (crates −1,531 net including tests/examples). Removed the `snapshot` and
`bench` features, the dead `offset_values`/`rescale`/`xgemv`/SVD-selector/
`solve_many` default paths, the LDL-config wrappers and two unused
dependencies; both feature sets build warning-free. Every pinned case keeps its
point bitwise (`medium` `1a16dc99…`, `ising11` `616bcced…`, `csdr3`
`0f810778…`), all audits pass, RSS is unchanged, and release A–B–B–A is
neutral (medium 1 thread −0.7%, ising11 +0.4%, csdr3 4 threads +1.5%, all within
run-to-run noise of this shared host). One real speed fix: primitive `Scalar`
math is now `#[inline]` across the crate boundary. Release (thin LTO) already
inlined it, but non-LTO builds called `mul_add` once per FMA: `medium` in the
`fast` profile drops 6.12 → 2.33 s (1 thread, −62%) and 2.71 → 1.51 s (4
threads, −44%) with identical points, and Rust API consumers that do not use
LTO get the same gain. Full suites pass in the `fast` profile: solver lib
442 + bin 7 + `it` 201 + doc 4, arithmetic 33, pmp 3, ffi 13.


Frozen arms: `simp-final-fast` and `simp-final-rel`; source manifest SHA-256
`498ec00aa6afd66b146e69729c2b5f1a7d5674ad7e95a80eec09a419e3ed550d`.

## 2026-10-03 — ownership and storage simplification

**Kept for memory and simpler ownership.** Ordinary consumed JSON moves
P/q/A/b into the existing preprocessing setup; native borrowed calls retain
their copying semantics. CSC validation is shared, including zero-origin
column pointers. JSON still validates cone geometry and exports directly to
the writer instead of first constructing a complete string.

Dense-block LDL borrows the direct layer's current KKT when refactoring;
its duplicate CSC is gone. Dormant sparse fallback receives all current
shifted values before factorization. Dense factor and unshifted residual
matrices remain distinct. Fully dense upper Schur destinations use direct
triangular indices; sparse blocks retain position maps. Medium's recorded
layout saves about 27.2 MiB of dense-backend CSC values/indices and 27.6 MiB
of per-block index maps. The earlier description of seven value copies was
incorrect: it counted three value arrays and four index arrays.

MPFR SVD rotation replay uses already allocated output buffers. Primary PSD
scratch uses Vt after scaling has consumed its right vectors; parallel step
workers retain independent scratch. Sampled batched RHS snapshots store the
upper triangle and restore the exact symmetric entries. These changes leave
arithmetic, factorization shifts, refinement and convergence rules intact.

Quick matching gates only: medium Float64 and Ising11 MPFR512, including
one/four-thread Ising runs across the successive storage patches. All are
`Solved`, pass their original-coordinate audits, and reproduce every x/s/z
and sampled_y entry bitwise against frozen `simp-final-fast`. Medium is 19
iterations with the recorded opt-in qnorm setting; Ising11 is 52. A malformed
zero-origin CSC input remains rejected. One existing dense fallback diagnostic
passed to cover the concrete shifted-matrix ownership risk; no routine suite
or precision sweep was run for this batch.

Final fast arm: `streamline-schur-20261003-fast`. Single-run RSS (preliminary)
is 404.2 MiB for medium/four threads and 64.3 MiB for Ising11/four threads.
These are not release speed claims. Evidence: `~/.cache/sdpx-e2e/streamline-20261003/`,
including `quick-gates.json`, boundary rejection, source manifests and backups.

README and architecture now describe one generic engine, actual dependencies,
backend ownership, precision dispatch and the compact verification workflow.
The plan tracks remaining ownership and hot-path work; obsolete storage
attribution and completed targets were corrected.

Release comparison submitted as PBS 222351, eight reserved cores/32 GiB,
one-hour limit, node192. Frozen baseline/candidate archives, dynamic OpenBLAS
with one BLAS thread, and four physical solver cores. Sequential release ABBA
covers medium Float64 and existing Lambda19 MPFR768. Each case requires
`Solved`, complete point identity and an independent original-coordinate
audit; an audit is reused only after all points are proven identical.
Native/API/process timing and process peak RSS are captured separately.
Remote namespace: `~/projects/sdpx-streamline-20261003-0153/`.
Results pending; no Lambda43 or cancelled campaign was restarted.


## 2026-10-03 — archived application context from the plan

### Recorded g0 application campaign (2026-09-29)

Application context only; this plan does not authorize starting or resuming
jobs. Recheck live state and current user authorization before cluster work.
The recorded deployment uses the tested shared-SOC source snapshot `03dd5ae2c096d1593a1e92d59935dbf9692939bb`
in a new immutable cluster release. Build/MPFR512 original-coordinate validation:
PBS `221952.node220`, 8 cores (retry after the original node rebooted).
Follow-on PBS `221954.node220` reserves 64 cores/96 GiB for
at most four hours, four workers with 16-thread solves. Fifth-order upper and
rational third-order lower searches use 32/64/128 positive midpoint nodes,
then one additional tail Bernstein subdivision at 128 nodes; change one axis
at a time. Settings: 512 bits, feasibility 1e-40, gaps 1e-30, external conic
checks 1e-30, outward integration 1e-4 per spin. Two successive node changes
below 1e-3 are a finite quadrature gate, not global hierarchy optimality.
Rank only exact-tail/full-domain certified candidates. Also integrate disjoint
tail-spin savings for the existing lower candidate. Frozen input/source hashes
and all rejected rows live in the analytical project's
`numerical/cluster_soc_20260929/`; existing best certificates remain in force.


## 2026-10-03 — assemble directly into the KKT primal prefix

**Kept for memory and ownership.** Condensed construction already inserts
every diagonal; all built-in backends store the upper triangle. The Schur
pattern therefore equals the first n columns of the direct KKT, even with
retained equalities. Constructor debug assertions verify that invariant.
Assembly writes/clears/checks only that prefix and publishes it through the
same backend update_values notification. Retained cone tails and the
factorization/refinement path are unchanged. The standalone Schur CSC is
constructor-local; its persistent values, row indices and column pointers
are gone, along with the per-update transfer.

Recorded medium storage saving: 27.2 MiB. Fast process peak falls 404.2 →
390.6 MiB in these single runs; constructor overlaps mean the retained-storage
formula is not a peak-RSS prediction. Medium Float64/four threads and Ising11
MPFR512/four threads pass original-coordinate audits and keep all output
points bitwise identical to the prior frozen storage arm. Numerical
convergence, regularization/escalation and refinement remain unchanged.
Existing diagnostic assertions were mechanically updated to inspect the
same KKT prefix; no new scaffolding or routine suite was added.

Arm `streamline-prefix-20261003-fast`; `prefix-quick-gates.json` under the
current evidence directory. A bounded independent source review found no
concrete issue in backend notifications, failure handling, sampled or owner
paths. No new distributed protocol or MPI layout was introduced.

JSON export now uses a bounded BufWriter and explicitly propagates flush
errors. The existing one-variable save/load solve check (`json_io::test_json_io`)
passes with identical recovered points; only that test was run.

The generic faer lower-triangle Gram candidate remains frozen outside the
production source. Gravity Float64 one/four-thread quick solves pass and
have identical candidate points; small-case timing shows no clear gain.
A two-line SVD candidate removes redundant whole-workspace clearing and
initializes only the needed U identity. It is a separate write-pass trial;
retention awaits matched release evidence, without unsafe uninitialized data.


SVD clear trial: MPFR512/four-thread Ising11 remains `Solved`/52, with all
points bitwise identical and audit passing. The single fast run is 3.835 s
API versus 3.845 s for the prior arm, offering no promising gain. The patch
was restored; no release speed claim or broad follow-up was made. Saved
candidate and read-before-write analysis remain outside the repo. Continue
with larger measured ownership and Gram costs.


Cluster PBS 222351 completed the Float64 medium ABBA: native/API/process
medians 3.429/3.430/3.595 → 3.487/3.487/3.533 s; peak RSS 474.5 → 411.4 MiB
(−13.3%). All four points are identical and the original audit passes.
Native +1.7% gives no speed claim; retained for the clear memory benefit.
Lambda19 baseline is `Solved`/119, native 424.398 s, RSS 642.1 MiB, but
its Julia audit could not load JSON because JULIA_DEPOT_PATH was unset.
Job exited 1 before any Lambda19 candidate run; no numerical audit failure
was reported. PBS removed dependent 222352 without starting it. Both frozen
binaries and completed evidence remain intact. Resume uses installed audit
depots and the remaining B–B–A on the same node; record the interruption.

The direct KKT pattern is immutable after construction, including prepared
value updates. Removed the repeated expected-row-count scan and impossible
cache-rebuild branch from MPFR refinement; exact row lists are created once.
Ising11 MPFR512/four threads remains `Solved`/52 with identical complete
points and a passing original-coordinate audit. Arm
`streamline-fixed-pattern-20261003-fast`; no precision/convergence change.


Corrected audit resume: PBS 222353 on node192, eight cores/32 GiB/35 minutes,
reusing binaries `ac2b9f68…` / `030e4c51…`. JULIA_DEPOT_PATH points to existing
integration/tool depots. The saved Lambda19 baseline now passes the original
independent audit (`accepted=true`, `Solved`/119). Remaining B–B–A runs are
in progress; the original interrupted A run is retained in the matched batch
and the interruption is explicit in the summary. No binaries were rebuilt
for this operational repair.


Follow-up retry: PBS 222354 reserves eight cores/32 GiB/35 minutes on
node192 with afterok:222353. Frozen prefix/faer sources and two gravity
inputs are unchanged from the scheduler-removed 222352 attempt. New v2
scripts verify the repaired parent completion, preserve the old submission
evidence, and run medium prefix storage plus two-size Float64 triangle
comparisons serially. Payload SHA256 45ccb272…; no old campaign restarted.


## 2026-10-03 — halve exact residual index storage

**Kept for memory.** ExactRows now stores only incoming off-diagonal
mirrors; each residual reads its own CSC column from authoritative KKT
indices. For the upper-triangle KKT this reproduces the prior term order
exactly, including one diagonal. Full-row prefix work counts remain identical,
so pool task boundaries do too. Fixed-pattern value updates read current
values. Storage saving is 16×nnz(KKT) − 8×(dimension+1) bytes.

Ising11 MPFR512/four threads is Solved/52, all x/s/z/sampled_y values
bitwise identical, with the original-coordinate audit passing. Native/API
timing from this single fast solve is preliminary; no speed claim. Arm
streamline-exact-mirrors-20261003-fast, evidence in
~/.cache/sdpx-e2e/streamline-followup-20261003/exact-mirrors-quick-gate.json.
Bounded independent source review found no issue.


## 2026-10-03 — reuse compact SVD input and sampled panel capacity

MPFR GESVD now destroys compact tall/square input directly, as its existing
contract permits; wide or padded input retains the packed copy. Scaling,
bidiagonalization, QR, reflector reconstruction and output ordering use the
same arithmetic/order. Workspace removes m×n values; right-only square work
is 2n²+8n−2 values (20 KiB saved per worker on Ising11/512, about 307 KiB
on Lambda19/768 at maximum order 53).

Sampled TLS panels resize their existing Matrix buffers instead of replacing
them at each shape change. Every active entry is initialized by its existing
producer; residue caches remain attached to authoritative Q/V operands. A
serial Ising11 block sweep previously caused about 1.11 MB of panel and
0.74 MB of product allocations. Warm capacity avoids those shape-change
allocations; these are structural estimates, not measured RSS or speed.

Ising11 MPFR512/four threads passes its original audit, Solved/52 with full
points bitwise equal. Arm streamline-scratch-20261003-fast. This correctness
run overlapped the diagnostic compilation; its timing is excluded from speed
comparisons. The existing provider_contracts_512 check passes (one test,
216 filtered), covering tall/wide workspace and overwrite behavior. Its
tall input is restored before the second factorization, fixing the diagnostic
caller to respect GESVD input destruction. No broad suite ran.


Completed first storage batch (PBS 222351/222353, node192, four threads,
BLAS one). Medium medians above remain unchanged. Lambda19 native/API/process
424.553/424.553/425.111 → 424.224/424.224/424.827 s; RSS
642.561 → 633.982 MiB (−1.3%). All Solved/119, complete points identical,
original audit accepted. Native −0.08% is below the speed gate. The ABBA
interruption for the Julia depot repair remains explicit. Local evidence:
~/.cache/sdpx-e2e/streamline-20261003/cluster-complete-summary.json.

Latest trial: capacity-preserving PsdWork/BlockWork and minimal EigEngine
resize avoid reconstructing engine buffers at each cone order change. Strict
Cholesky upper triangles are zeroed after reshape; right-only SVD U stays
absent, small SVD builds it when needed; paired step workspace resizes lazily.
Independent review and Ising11 MPFR512/four threads pass; full points bitwise
identical, Solved/52, audit accepted. Arm streamline-psd-resize-20261003-fast.
Retention awaits full Lambda19 speed/RSS comparison: geometric Vec capacity
can cost more memory. No speed or memory benefit claimed for this trial yet.


Queued PBS 222355, node192/eight cores/32 GiB/45 minutes, afterok:222354.
One candidate release build, then Lambda19 MPFR768/four-thread ABBA against
the frozen prefix binary; BLAS one and timed processes serial. Source/input/
settings are hashed, audits reused only after full point identity to the
independently accepted original-coordinate point. Batch includes fixed/incoming
ExactRows, compact SVD input, sampled/PSD scratch capacity reuse. Source archive
2bfed0ec…, payload 89f6b090…; candidate remains a performance trial until
matched medians are available. Remote:
~/projects/sdpx-streamline-mpfr-20261003-0256/.


## 2026-10-03 — packed Float64 refinement trial

Dense-block systems with a complete upper primal prefix use DSPMV directly
on current authoritative KKT values, then FMA the retained CSC tail. Sparse
leading patterns keep the existing residual route. The direct layer skips
its full unshifted residual matrix only for this eligible backend/pattern;
factor storage, fallback, precision and all refinement rules are unchanged.
For medium (1,934 total KKT coordinates), this removes 28.54 MiB and the
per-factor cache population. No mixed precision or factor/operator reuse.

One medium Float64/four-thread fast solve is Solved/19 and audited. Native
1.521 s and RSS 364.3 MiB are preliminary; earlier prefix fast RSS was 390.6
MiB. Points differ slightly due to packed BLAS/tail association, so treat
this as a kernel change, not bitwise ownership refactor. Formal matched
release timing/RSS will decide. Arm streamline-packed-residual-20261003-fast;
evidence ~/.cache/sdpx-e2e/streamline-packed-20261003/quick-gate.json.


## 2026-10-03 — prefix kept; triangular faer replacement rejected

PBS 222354 completed all frozen release ABBA comparisons on node192.
Medium single-KKT prefix: native/API/process 3.523/3.523/3.632 →
3.418/3.418/3.460 s (native −2.97%); RSS 414.234 → 393.502 MiB
(−5.0%). All Solved/19, full points identical and audits accepted. Kept.

General triangular faer Gram (existing rounded Z preserved) is rejected
as a replacement. Small gravity serial/four-thread native medians
0.17761/0.17675 → 0.20214/0.17955 s (+13.8%/+1.6%). Larger gravity
one/four/eight-thread medians 2.85270/1.80306/2.07253 →
2.40339/1.86022/2.12240 s (−15.8%/+3.2%/+2.4%). All Solved and audited;
within-arm repeat points identical. Larger iterations: old 37/36/36 versus
trial 35/37/37, so the serial benefit includes a trajectory change.
Trial stays outside production; the common BLAS tile remains.

Evidence: ~/.cache/sdpx-e2e/streamline-followup-20261003/cluster-summary.json,
remote ~/projects/sdpx-streamline-followup-20261003-0218/results/.
The next threading candidate splits uneven column work into existing-size
2D lower tiles, preserving diagonal dot calls and rounded Z.


Packed-residual release decision queued in PBS 222356, afterok:222355,
node192/eight cores/32 GiB/15 minutes. One candidate build and medium
Float64/four-thread ABBA, BLAS one. Baseline is the frozen current source
from 222355; the only source differences are direct/solver.rs and
ldl/dense_block.rs. Independent audit per distinct point, repeat identity
within each arm. Payload c0e7f37f…, source dfc1f7e2…. Remote:
~/projects/sdpx-streamline-packed-20261003-0309/.


## 2026-10-03 — balanced lower-Gram tiles and batched-output ownership

Float64 bound Gram trial keeps the original serial calls and diagonal dots,
but replaces uneven off-diagonal column rectangles with independent 16×16
packed jobs and a safe scatter. Rounded Z and the lower-triangle operator
remain unchanged. The existing storage admission includes the scratch:
43,344 bytes at border 101, including metadata. The Mac quick one/four-thread
checks on the frozen larger input pass and reproduce complete points exactly;
these subsecond fast-profile timings are preliminary. PBS 222360 is held
behind 222356 for a release ABBA on both model sizes, small at one/four
threads and larger at one/four/eight, BLAS one. One candidate build; all timed
processes serial. Source 9a2e929e…, payload 7f82c7b3…. Evidence:
~/.cache/sdpx-e2e/streamline-bound-2d-20261003/;
remote ~/projects/sdpx-streamline-bound-2d-20261003-0325/.

DefaultKKTSystem now keeps accepted constant and variable solutions in its
existing two-column batch output, removing x1/z1/x2/z2 allocations and the
four post-batch copies. Constant retries, affine success flags, corrector
lifetimes and cached scaled products are unchanged. Structural retained
storage reduction: 2(n+m) scalar values (244 KiB for medium Float64).

Separate residue-cache trial recycles a sole-owned stale compressed payload
only when its representation and exact length match the replacement. Shared
or mismatched payloads retain early release. Cache checks and exact encoding
are unchanged; every recycled slot is fully overwritten. Potential allocation
traffic avoided is about 73 MiB per Lambda19 scaling update at stable prime
counts, based on the earlier cache inventory; this is not a measured peak-RSS
or speed improvement.

One Ising11 MPFR512/four-thread E2E passes, Solved/52, with full x/s/z/sampled_y
bitwise identical to the frozen workspace arm and an accepted original audit.
Arm streamline-batch-residue-20261003-fast. Retain batch ownership for its
clear storage reduction; residue reuse awaits matched release measurement.
Evidence ~/.cache/sdpx-e2e/streamline-batch-residue-20261003/quick-gate.json.
No additional tests or suite ran.


The narrower RHS ownership follow-up retains workx: its zero-coefficient
axpby still depends on old scratch values in failure/reset paths. Both conic
RHS slices replace workz/work_conic, with each producer fully overwriting them.
Together with batch output ownership this removes 2n+4m persistent scalar
values after the first batch (870,080 numeric bytes for Ising11/512).
Initial allocation shifts earlier; before the first batch this conic-slice
change alone adds 2n values. The matching Ising11 solve is bitwise identical
and audited, arm streamline-kkt-buffers-20261003-fast.

Generic arrow leaf contributions now store t(t+1)/2 values rather than t².
Dots are unchanged; each dense Schur entry receives the same subtraction in
leaf order, with its diagonal updated once. Local routes keep empty
contribution buffers. Ising11 group stats show 11 components and border 20:
167,200 numeric bytes saved at 512 bits. One matching E2E is Solved/52 with
all points bitwise identical and an accepted original audit. SDPX_PROFILE was
enabled on this correctness run to obtain group stats; its timing is excluded
from performance claims. Arm streamline-arrow-packed-20261003-fast.

PBS 222361 is held after 222360, node192/eight cores/32 GiB/45 minutes.
One release build then Ising11/512 and Lambda19/768 ABBA at four physical
cores and BLAS one. Only kktsystem.rs, arrow.rs and rns_blas.rs differ from
its frozen baseline. Includes HSD ownership, packed contributions and the
residue-reuse trial; excludes the later borrowed scaled-product view.
Source 392e2ee6…, payload 6d86b447…. Original audit reuse requires complete
point identity at matched inputs/precision/settings. Evidence:
~/.cache/sdpx-e2e/streamline-mpfr-buffers-20261003/;
remote ~/projects/sdpx-streamline-mpfr-buffers-20261003-0342/.


## 2026-10-03 — capacity retention rejected after release comparison

PBS 222355 completed, exit 0, node192/four physical cores/BLAS one. Lambda19
MPFR768 medians native/API/process 426.567/426.567/427.199 →
424.222/424.222/424.768 s (native −0.55%); peak RSS
630.129 → 642.457 MiB (+1.96%). All Solved/119, complete points identical,
original audit accepted. Speed is below the 2% gate and memory is worse.
Reverted sampled TLS panel/product capacity retention and
PSD/condensed/EigEngine resize, preserving prior owned-buffer and compact
SVD-input changes. These combined measurements do not isolate either
capacity patch's RSS effect. No further capacity trial retained.

PBS 222356 is now running. The frozen 222356/222360/222361 comparisons still
contain the rejected capacity code identically in both arms; their isolated
residual/tile/ownership comparisons remain scoped to those snapshots. No
jobs were restarted. Working source separately borrows the accepted variable
scaled product instead of storing hs1 (constant hs2 remains owned), awaiting
the matching Ising11 gate. Evidence:
~/.cache/sdpx-e2e/streamline-mpfr-20261003/cluster-summary.json.


## 2026-10-03 — packed Float64 refinement kept for memory

PBS 222356 completed, exit 0, node192/four physical cores/BLAS one. Medium
Float64 medians native/API/process 3.405/3.406/3.517 →
3.372/3.372/3.416 s; native −0.98% is below the speed gate. RSS
391.338 → 364.578 MiB (−6.84%). All Solved/19, repeat points identical
within each arm; changed BLAS association gives different cross-arm points,
both independently accepted by the original-coordinate audit. Retained for
memory. Evidence ~/.cache/sdpx-e2e/streamline-packed-20261003/cluster-summary.json.

Working source after both capacity rollbacks also borrows the variable
scaled-product view directly from the accepted KKT cache (batch column 1
for affine, column 0 for a fresh solve). hs2 remains owned across corrector
updates. Removes m MPFR scalar values plus copies; Float64 keeps its cone
operator path. One Ising11/512/four-thread check is Solved/52, original audit
accepted, complete points identical. Arm streamline-kept-buffers-20261003-fast.
No new trait, flag or setting. This change is outside frozen PBS 222361.


## 2026-10-03 — Gram trial rejected; MPFR input repair

PBS 222360 completed, exit 0. All gravity runs are Solved and independently
audited; complete points identical. Native release medians small/four threads
0.177833 → 0.181530 s (+2.08%); larger/four threads 1.632287 → 1.720101 s
(+5.38%); larger/eight threads 2.016786 → 2.070085 s (+2.64%). Serial code
is unchanged and its small-case timing difference is noise. No useful RSS
benefit. Reverted the 2D kernel, keeping existing column tiling. The external
owned-scratch alternative was never applied. Full medians/evidence:
~/.cache/sdpx-e2e/streamline-bound-2d-20261003/cluster-summary.json.

Latest kept working source after capacity/Gram rollbacks passes the small
Float64 gravity gate at four threads, BLAS one: Solved/19, accepted original
audit, full point bitwise identical to packed-refinement baseline. Arm
streamline-kept-lp-20261003-fast; fast timings are preliminary. Evidence:
~/.cache/sdpx-e2e/streamline-kept-buffers-20261003/gravity-small-quick-gate.json.

PBS 222361 failed preflight after 23 s, exit 1: its frozen Ising archive was
missing on the cluster. No compilation or timed solve ran. Failed logs and
manifest remain untouched. Retry 222362 uses a fresh namespace and the same
source/input/settings/comparison, with the local frozen archive supplied and
its recorded SHA-256 checked before submission. Source 392e2ee6… unchanged.
One release build and serial Ising11/512 plus Lambda19/768 ABBA, node192,
eight allocated cores/32 GiB/45 minutes, four physical timed cores/BLAS one.
State is submitted, awaiting live check. Evidence:
~/.cache/sdpx-e2e/streamline-mpfr-buffers-retry-20261003-0410/; remote
~/projects/sdpx-streamline-mpfr-buffers-retry-20261003-0410/.


## 2026-10-03 — direct SVD Vt and gravity correction diagnostic

MPFR GESVD now uses compact caller Vt as row-major replay scratch and final
sorted output for tall/square non-overwrite calls. Padded input/U staging
is independent; wide matrices and overwrite jobs retain their staging.
For square right-only factors, workspace falls from 2n²+8n−2 to n²+8n−2
scalar values, with the final Vt copy removed. Ising11/512, one thread:
Solved/52, independent original audit accepted, full x/s/z/sampled_y
bitwise equal to the frozen accepted point. Fast timing is preliminary.
Arm streamline-direct-vt-20261003-fast; evidence
~/.cache/sdpx-e2e/streamline-direct-vt-20261003/quick-gate.json.
Retained per-thread workspace saves its largest n² block, not the sum over
cones: Ising order 16/512 saves20 KiB per worker, at most80 KiB for four.
The summed logical traffic removed per scaling sweep is about 1.088 MiB.
Numerical failure may leave output scratch; failed-factor outputs are
unspecified and PSD scaling does not consume them. INFO unchanged.

External-only gravity diagnostic, same Float64 larger model/four threads:
Solved/37 and audit pass; two diagnostic returned points match exactly.
At factor 26/correction 114 equality 17 exact residual improves 3178×; new
dominant equality 84 worsens 2.58% to 9.38e−8. Residual rounding errors are
about 2e−16, too small to explain the stall. This bounded capture supports
targeting Newton correction quality while preserving refinement rules.
Gather/allocation measured 19.51ms over 269 residual calls, 3.8% of 515.77ms
native diagnostic time; includes tracing and is not a speed claim.
Primal leaves form contiguous point[2..20202]; trunk remains noncontiguous.
Evidence /tmp/sdpx-refine-row-20261003/two-rows/analysis.json. The trace patch
stays external; no production diagnostics or setting added.

Sampled persistent-Gram packing was inspected but not implemented: actual
net four-worker scratch-adjusted savings are 0.407 MiB on Ising11/512 and
7.388 MiB on Lambda19/768. It needs MPI transport/owner-cost changes and
an added copy per update; defer rather than introduce that complexity now.


## 2026-10-03 — contiguous bound residual views (trial)

Generic Float64 local-bound setup records whether bounded primal coupling
IDs form one contiguous range. Such models borrow that point slice and
write the forward BLAS product directly into their residual range; other
layouts retain the existing gather/product path. Beta=0 BLAS overwrites all
product entries, then rhs−product and the unshifted local FMA sequence stay
unchanged. Detection is structure-based, with no benchmark/size branch.
For larger gravity, temporary numeric payload per call falls 324,816→1,616
bytes (317.2→1.58 KiB); 269 calls avoid 82.9 MiB allocation traffic. This is a
structural transient-storage count, not a measured RSS benefit.

Small gravity53/four threads/BLAS one: Solved/19, original-coordinate audit
accepted and full point bitwise identical to frozen packed-refinement arm.
Arm streamline-residual-view-20261003-fast; one-run fast timing preliminary.
Evidence ~/.cache/sdpx-e2e/streamline-residual-view-20261003/quick-gate.json.
Prepared one release build/serial ABBA on small gravity 1/4, larger gravity 4
and direct-Vt Ising11/512 4 after 222362, node192/eight allocated cores/32 GiB/
20 minutes/BLAS one. Source bfa34c76… changes only arrow.rs,local_bounds.rs and
mpfr_svd.rs from 222356 packed baseline. Both frozen arms retain rejected
capacity code identically; no 2D trial and no later HSD/RNS changes. Evidence
~/.cache/sdpx-e2e/streamline-views-release-20261003-0424/.

Submission evidence: qsub returned 222363.node220; live dependency state check pending. Remote ~/projects/sdpx-streamline-views-20261003-0424/.


## 2026-10-03 — consumed sampled JSON ownership

The existing sampled setup constructor now accepts Cow inputs. Native/FFI
borrowed inputs keep their cloning behavior; consumed JSON moves P/q/b/
linear A directly. Removes the duplicate JSON identity branch and one
internal wrapper. Fingerprints still read identical original values before
any move/strip/scale. Expanded sampled A already moved into setup and was
already compacted after KKT construction; no additional full matrix removed.
Avoided setup copies (numeric payload/CSC indices only): Ising11/512
802,288B (0.765MiB), Lambda19/768 11,015,776 B (10.505MiB). These counts
exclude allocator/preprocessing replacements and are not peak-RSS claims.

Residual norms also remove a redundant vector finite scan: compute the same
infinity norm once, map any nonfinite scalar norm to positive infinity,
preserving previous NaN/infinity behavior. ExactRows and Float64 bounds
products/refinement rules unchanged. One Ising11/512/four-thread CLI solve
passes its independent audit with complete x/s/z/sampled_y bitwise identical
to the frozen point. Small Float64 gravity/four-thread gate also passes
with full point identity. Arm streamline-sampled-owned-20261003-fast; fast
timings preliminary. Evidence
~/.cache/sdpx-e2e/streamline-sampled-owned-20261003/{ising-quick-gate,quick-gate}.json.
These changes are outside frozen 222362/222363.


## 2026-10-03 — single-RHS bound panel GEMV (trial)

Float64 bound panel products use DGEMV when there is one RHS, retaining
DGEMM for batched RHS. Serial and existing output-owned tiles keep their
leading dimensions and thread-independent partitions. Operation shape
chooses the kernel; no model or size tuning added. Small gravity53/four
threads/BLAS one passes Solved/19, original-coordinate audit and full frozen
point identity. Fast timing is preliminary. Evidence
~/.cache/sdpx-e2e/streamline-bound-gemv-20261003/quick-gate.json.

PBS 222364.node220 is held afterok:222363 on node192/eight cores/32 GiB/
15 minutes. One release build followed by serial ABBA small gravity53 at
one/four threads and large53 at four, BLAS one. Frozen source 829d1eb1…
differs from 222363 views only in this GEMV change; both retain previous
capacity/direct-Vt/residual views and exclude later norm/JSON/HSD/RNS edits.
BLAS arithmetic may differ across arms: independent original audits and
within-arm repeat point identity are required. No speed claim or keep
decision until results. Remote ~/projects/sdpx-streamline-gemv-20261003-0435;
local ~/.cache/sdpx-e2e/streamline-gemv-release-20261003-0435/.


## 2026-10-03 — validate inputs once before common setup

Ordinary construction keeps its existing dimension, CSC and settings
checks intact at the boundary. Sampled construction checks dimensions/P/
settings against linear A inside the existing setup timer; SampledOperator
still validates linear CSC, finite factors and ranges, and materialization
still rejects nonfinite overflow/underflow. Common preparation trusts the
constructed canonical full A rather than scanning it again. No new flag or
helper; derived reduced cone-dimension gate remains. Invalid sampled inputs
with several faults may report a different first error; valid arithmetic,
fingerprints and timer scope are unchanged. Avoids two expanded row-index
passes (up to 36.6 MiB of index reads on Lambda19), a work count, not a
speed or RSS claim.

One Ising11/512/four-thread CLI solve is Solved/52, original audit passes,
and complete x/s/z/sampled_y equals the frozen audited point. Arm
streamline-boundary-validation-20261003-fast; evidence
~/.cache/sdpx-e2e/streamline-boundary-validation-20261003/quick-gate.json.


## 2026-10-03 — MPFR buffer batch completed; retained with memory tradeoff

Repaired retry PBS 222362 completed exit 0, node192/four physical cores/BLAS
one, release ABBA frozen sources. Ising11/512 native median 7.12980→7.12710s
(−0.038%, no speed claim), API 7.12994→7.12724s, process 7.30420→7.18786s;
peak RSS73.424→75.404 MiB (+2.70%). Lambda19/768 native422.86867→420.58172s
(−0.541%, no speed claim), API422.86882→420.58190s, process423.41134→421.11978s;
peak RSS639.957→628.102 MiB (−1.85%). All eight solves retain full x/s/z/
sampled_y identity and accepted independent seed audits; Solved/52 and119.

Keep HSD output/RHS ownership, packed generic-arrow contributions and RNS
sole-owned equal-size payload reuse for structural allocation/work savings
and the Lambda memory benefit. The combined batch does not isolate any
individual contribution; Ising peak RSS increases and native gains fail the
2% gate. Both arms retain former capacity/2D code equally; later borrowed
hs1/direct-Vt/views/JSON/norm changes are excluded. Failed222361 preflight
evidence remains intact. Evidence
~/.cache/sdpx-e2e/streamline-mpfr-buffers-retry-20261003-0410/cluster-summary.json.
222363 views comparison has started; GEMV 222364 remains dependent.

Exact norms now call existing dot_slices on their owned scaled values,
removing the generic address-pair collection (16 bytes/nonzero coordinate).
Across eight scans, at most64*(n+m) pointer bytes written per info update
are avoided; TLS highwater benefit depends on other dots. The scaled-value
Vec remains. Exact accumulation and the nonnegative-square fallback value
are unchanged, including zero/NaN/infinity behavior. Ising11/512/four threads
passes Solved/52, independent audit and full frozen point identity. No speed
claim. Evidence ~/.cache/sdpx-e2e/streamline-norm-slice-20261003/quick-gate.json.


## 2026-10-03 — PSD L2 lifetime and sampled absolute products

For PSD order>3, scaling packs the second input directly into mat2 lower,
zeros strict upper, and calls the same lower potrf there. L2 is consumed by
L2ᵀL1 before mat2 is reused for Rinv. Small fixed paths, paired dispatch,
rounded operands, failures and full GEMM operands are unchanged. Removes
one n² worker factor allocation plus a triangular copy. At retained max
orders Ising16/512 and current Lambda47/768, saves 20 KiB/241.61 KiB per worker.
The matching Lambda sweep avoids 6,014,848 B of logical triangular-copy traffic;
these are structural counts, not measured RSS or speed.

A private exactdot trial selects GMP mpn_sqr once for identical slices above
256 bits. Both GMP kernels produce identical exact 2N-limb products; all
scans/shifts/carry/fallback/final rounding code stays unchanged. No public API or
per-term runtime branch. PSD+square Ising11/512/four-thread CLI is Solved/52,
accepted original audit and full frozen x/s/z/sampled_y identity. Evidence
~/.cache/sdpx-e2e/streamline-psd-square-20261003/quick-gate.json.

Sampled absolute-adjoint scalar fallback hoists only the first rounded
weight*basis multiply into existing dvec; remaining multiplication and
+= order are identical. Ising removes 65,840 multiplications per call
(75,338→9,498 first products), around 3.42M over 52 calls. Existing dvec adds
25,760 B across 22 small first-use allocations at 512 bits, then reuses them;
no new field/cache. Exact residue path unchanged. One Ising11/512/four-thread
CLI again Solved/52, accepted audit, full frozen point identity. Evidence
~/.cache/sdpx-e2e/streamline-sampled-abs-20261003/quick-gate.json.

PBS 222365.node220 submitted afterok:222364, node192/eight cores/32 GiB/55 min.
Two release builds then serial ABBA Ising11/512 and Lambda19/768 on four
physical cores/BLAS one. Baseline 98c0a380… equals audited norm-slice source;
candidate 3fb0f27a… changes only exactdot.rs,psdtrianglecone.rs,sampled/mod.rs.
Both contain current capacity rollback/HSD/RNS/hs1/JSON/validation/normslice
and prior GEMV/view changes equally. No speed claim pending comparison;
source and input/settings hashes frozen. Later arrow lazy-work is excluded.
Evidence ~/.cache/sdpx-e2e/streamline-psd-abs-release-20261003-0451/;
remote ~/projects/sdpx-streamline-psd-abs-20261003-0451/.


## 2026-10-03 — allocate leaf single-RHS work only when used

Leaf w/v start empty; first_solve allocates both once, then fully overwrites
w from RHS and v from forward-scaled w before every read. One predictable
guard; no arithmetic or scratch-history change. Batch/factor/fallback work
is independent. Float64 BoundPanels bypass generic leaf single/batch work,
so w/v stay empty. Larger gravity's 20,200 two-variable leaves avoid 646,400 B
of numeric payload and 40,400 allocations; vector headers remain, allocator
overhead/RSS not claimed. Generic arrow still uses single RHS during
initialization/corrector/refinement, so allocates once on first use.

Small gravity/53/four threads is Solved/19 with full frozen point identity
and accepted original audit. Generic Ising11/512/four threads is Solved/52,
full x/s/z/sampled_y identity and accepted independent audit. Fast-profile
timings preliminary. Evidence
~/.cache/sdpx-e2e/streamline-arrow-lazy-20261003/{quick-gate,ising-quick-gate}.json.
This change is outside frozen 222365.

Weighted native SYRK considered, not implemented: allowed disabled dynamic
regularization accepts finite nonzero pivots of either sign. sqrt(d) would
turn negative weights into NaN, requiring a new fallback/sign restriction.
For positive weights it would batch only diagonal tiles (16.1% of gravity
lower entries), add square roots and change rounding, with no storage
benefit. Retain current Yᵀ(Y*d) contract; no build/run needed for this rejection.


## 2026-10-03 — residual-view comparison completed; retained

PBS 222363 completed exit 0, node192, serial release ABBA, BLAS one. Native
medians small gravity/53/t1:0.177952→0.175397s (−1.44%); small/t4:
0.178136→0.173379s (−2.67%); large/t4:1.683396→1.505325s (−10.58%).
Large API 1.686540→1.508407s; process 1.939116→1.766893s; peak RSS
209.650→208.693 MiB. All full points/iterations repeat and match across arms,
with accepted independent original audits. Keep contiguous residual views
for qualifying small/four and large/four speed gains and structural scratch
savings; no serial small qualifying gain and no size tuning.

Direct caller-Vt in the same batch: Ising11/512/t4 native 7.061926→7.100769s
(+0.55%, no speed claim), API 7.062055→7.100890s, process 7.130797→7.161972s,
peak RSS74.084→75.311 MiB (+1.66%). Solved/52, full points identical and seed
audit accepted. Retain direct-Vt for the explicit one-matrix/copy removal;
the small structural saving is not a measured RSS benefit. Both arms retain
old capacity code equally; frozen scope excludes later HSD/RNS/JSON/norm
changes. Evidence
~/.cache/sdpx-e2e/streamline-views-release-20261003-0424/cluster-summary.json.
GEMV 222364 now running, PSD/absolute 222365 held after it.


## 2026-10-03 — single-RHS GEMV completed; retained

PBS 222364 completed exit 0: release serial ABBA, node192, BLAS one. Small
gravity/53 at one thread: native 0.178934→0.132424s (−25.99%), API
0.179650→0.132949s, process 0.301777→0.182728s, RSS48.055→50.316 MiB
(+4.71%). At four threads: native 0.174198→0.132333s (−24.03%), API
0.174641→0.132783s, process 0.219188→0.181427s, RSS50.760 MiB unchanged.
Larger gravity/53/four: native 1.527260→1.346598s (−11.83%), API
1.530413→1.349827s, process 1.786632→1.624533s, RSS209.709→209.266 MiB.

Keep GEMV for one RHS; batched GEMM and existing tiles remain. Solved/19
small and36 large, all independent original audits accepted, repeat points
identical within arms. Large cross-arm points differ due to changed BLAS
rounding; convergence and refinement rules are unchanged. Both frozen arms
retain old capacity/direct-Vt/views equally and exclude later HSD/RNS/JSON/
norm changes. Evidence
~/.cache/sdpx-e2e/streamline-gemv-release-20261003-0435/cluster-summary.json.
222365 is now running its two builds; no new MPI or large campaign started.

Current Lambda dimension correction: matching hash6a492614… has28 PSD cones
of orders40–47, Σn=1,211, Σn²=52,493, Σtriangle=26,852, plus54 equalities
(m26,906). Thus L2 saves247,408B=241.609375KiB per largest worker and
6,014,848B of triangular-copy reads+writes per sweep. Max53/sum111,969
belong to a different cached input; those are not current222365 claims.
Evidence ~/.cache/sdpx-e2e/runs/l19-memadj/{point,audit}.json.

Universal packed DenseLeaf factors were inspected and deferred: current
Lambda could save5.695 MiB (~0.91% RSS), Ising0.365 MiB, medium none. It
would change every factor/solve index and need new variable-column parallel
splitting; pinned E2Es do not exercise the inner split. Keep the current
layout for this round. Absolute componentwise residual scans already skip
all work when their tolerance isNone; Ising explicitly enables1e-30, so
those measured scans supply an accuracy gate and must remain. No builds or
runs were needed for these two read-only decisions.

## 2026-10-03 — adjoint scratch/root reuse; bound trial; clean-build comparison

Sampled scalar adjoints no longer acquire the square/product panels they do
not read. The same panel operation remains in the off-diagonal branch;
rounding and store order are unchanged. Ising11's 22 scalar blocks have
14,252 scratch cells per sweep (1,140,160 logical bytes at 512 bits). Serial
shape churn represents 42 buffers/1,105,600 bytes per sweep; actual parallel
allocation savings depend on scheduling, and forward products still use
these panels. No measured RSS/speed claim yet.

PSD scaling copies singular values into persistent λ, then reuses dead SVD
singular-value scratch for their rounded roots. Λisqrt and Rinv use the same
roots, eliminating 322/1,211 square roots per Ising/current Lambda sweep.
No new buffer, numerical operation association, factorization or failure
policy. All successful SVD routes overwrite the scratch before its next use.
Ising11/512/four threads: Solved/52, full x/s/z/sampled_y identity and accepted
original audit. Fast timings preliminary. Evidence:
~/.cache/sdpx-e2e/streamline-adjoint-sqrt-20261003/quick-gate.json.

A separate two-coordinate bound factor trial uses the original coupling
instead of reconstructing it from the rounded factor/pivot product. It
passes small gravity53/19 and full-input256/25 original audits, with changed
rounding and unchanged iterations. The trial is out of the live tree pending
its release comparison. Evidence:
~/.cache/sdpx-e2e/streamline-bound-coupling-20261003/*.gate.json;
MPFR input hash dd8ecea76878d8f83d16618b16f6817187cead7ad505efc2eb2d2a3b53c0770a.

222365 build-scope correction: its two source archives differ in arithmetic,
but the candidate reused the baseline arithmetic artifact. Both exactdot
source mtimes (04:45:47/04:46:36) precede arithmetic invoked.timestamp
04:58:40; relative-path dependency fingerprint and shared target remained,
and only the baseline log compiled arithmetic. Thus this batch measures PSD
L2 +sampled absolute-product hoisting with baseline arithmetic, not GMP
squaring. Ising ABBA complete: native7.168404→6.977779s (−2.66%), all52
iterations/full points/audits pass. Lambda still running; no final conclusion
or isolated contribution claim. The source/effective-binary distinction and
failed cache evidence are preserved in the original packet/build logs.

PBS222366 submitted, dependency-held afterok222365. Fresh namespace
~/projects/sdpx-streamline-next-20261003-0520; eight cores/32GiB/45minutes,
node192, BLAS one. Four frozen arms explicitly clean both changed crates
(arithmetic and solver) each build, then serial four-physical-core ABBA:
one-bound coupling on small/large53 and small256; unused adjoint panels+
PSD roots on Ising512; GMP square alone on Ising512. Full source/hash/arm
scopes are in ~/.cache/sdpx-e2e/streamline-next-release-20261003-0520.

The same job includes an actual two-rank/two-thread-per-rank Ising512 MPI
ownership gate with fresh original-coordinate audits and matched full-point
comparison. External-only candidate removes OwnedKktSystem.offset and reuses
constant_work.s, which batched RHS packing does not read. Offset preparation
fully overwrites it before subtraction/recovery. Ising two-owner logical
payload savings:99,200/107,040B. Current Lambda formula112*(assigned PSD
rows+54) bytes/nonempty owner; two-/eight-owner means1,509,760/381,976B,
not individual assignments or RSS. Live MPI code stays unchanged until this
gate passes. No MPI speed claim or cancelled campaign restart.

## 2026-10-03 — leading sampled diagonal dot trial

For leading diagonal level r=0, the existing packed-triangle traversal is
exactly contiguous x[..tri(h)] and its wdiag column, for every block dimension.
Use the existing slice dot there; nonleading diagonal iteration is unchanged.
Float64 keeps its ordered FMA fold; MPFR keeps finite exact accumulation and
one rounding without indexed pointer collection. Exotic wide-exponent MPFR
fallbacks can differ in zero sign because generic dots omit benign zero
products and slice chains keep them; this is within the documented scalar
value contract, not a universal zero-bit identity claim. No accuracy gate
or arithmetic primitive changed.

Ising512/four threads: Solved/52, all x/s/z/sampled_y identical and original
audit accepted. Evidence ~/.cache/sdpx-e2e/streamline-diagonal-slice-20261003/
quick-gate.json. Fast timing remains preliminary. Across 224 recorded
adjoint sweeps, 16,875,712 pairs could avoid up to257.5MiB pointer writes plus
matching reads; zero dynamic entries reduce that traffic. TLS capacity is
already reused, so this is neither an allocation nor RSS claim.

PBS222367 submitted, dependency-held afterok222366. Frozen one-file release candidate:
~/.cache/sdpx-e2e/streamline-diagonal-release-20261003-0535. One new build and
serial Ising512/four-core ABBA against the 222366 panels executable, after
its actual MPI gate; eight cores/32GiB/15min on node192. Both changed local
packages are explicitly cleaned: the parent's final mul arm has different
arithmetic, despite old source mtimes. Full point identity is required to
reuse the accepted independent seed audit. No extra precision/thread sweep.

Immutable RNS scan bypass rejected without implementation: cache entries
already store only fingerprints, metadata and compressed residues, with no
duplicate MPFR image. Generic caches also serve mutable operands; shared
constants can accompany cloned bases at new addresses. Skipping validation
needs a new immutable-owner contract/tag propagated through kernels, beyond
this compact cleanup. Existing owner-invalidated diagonal-congruence cache
cannot be applied to general cache users. No build or solve needed.


## 2026-10-03 — PSD/absolute bundle comparison complete; retained

PBS222365 exit0, serial release ABBA node192/four physical cores/BLAS one.
Effective scope remains PSD L2 +sampled absolute-product hoisting, with
baseline arithmetic in both executables (GMP square not measured).
Ising11/512: native7.168404→6.977779s (−2.659%); API7.168542→6.977885s,
process7.306984→7.037934s, peak RSS73.927734→74.310547MiB (+0.518%).
Current Lambda19/768: native423.155426→423.780637s (+0.148%, no speed gain),
API423.155587→423.780773s, process423.680406→424.327083s,
peak RSS610.271484→605.953125MiB (−0.708%).

Keep bundle for qualifying Ising speed and Lambda memory/explicit L2 storage
removal, recording the small Ising RSS increase. All eight complete points
match the accepted independent seed audits and repeat across arms;
Solved52/119. No isolated attribution or current SDPB/MOSEK claim.
Evidence ~/.cache/sdpx-e2e/streamline-psd-abs-release-20261003-0451/
cluster-summary.json plus both retrieved build logs/exit-code.
222366 now running its four explicitly clean builds;222367 held after it.

## 2026-10-03 — narrow fused arithmetic and setup lifetimes

Trial `mul_add` retains the exact regular product and addend in a bounded
2N+2-limb window, then rounds once at the requested precision. N<=4 only:
128/256 bits; zeros, nonfinite values, distant exponents and >=512-bit values
retain native MPFR. No intermediate product rounding or precision lowering.
The exponent window reserves a full carry bit; existing integer reconstruction
handles cancellation and MPFR exponent limits. Independent source review found
no numerical issue.

External primitive audit covers random/window-edge values, cancellation,
midpoint tails, signed zeros/nonfinite combinations and changed MPFR exponent
limits. It passes. Near-input ABBA primitive screening: 128-bit33.284→20.386ns,
256-bit38.777→23.314ns; spread-input gains are smaller. 512-bit58.381→58.887ns
near and54.342→60.844ns spread: reject that width and retain native MPFR.
These are primitive screening results, not solver speed claims. Prototype and
logs: /tmp/sdpx-fma-prototype-20261003/locked-workspace; no probe added to repo.

Small full-input gravity256/four threads: Solved25, complete x/s/z/sampled_y
identity and accepted original-coordinate audit. Evidence:
~/.cache/sdpx-e2e/streamline-bound-coupling-20261003/narrow-fma-quick-gate.json.
Formal two-size isolated FMA packet prepared; no formal conclusion yet.

Constructor now builds KKT and compacts sampled A before allocating initial
variables/residuals and sparse workspace. KKT still sees the full numerical
matrix, sampled residuals use their authoritative operator, and nonsampled
compaction is a no-op. Removes full-A overlap with (4n+4m)*sizeof(T): current
Lambda19/768 n1211,m26906 gives12,596,416B (~12.0MiB). Persistent storage is
unchanged; this count is not measured peak RSS. Ising11/512/one thread:
Solved52, complete point identity and accepted original audit. Evidence:
~/.cache/sdpx-e2e/streamline-setup-lifetime-20261003/quick-gate.json.

PBS222368 submitted, held afterok222367, isolated narrow FMA on small/larger
gravity256/four physical cores, serial release ABBA/BLAS one. Fresh namespace
~/projects/sdpx-streamline-narrow-fma-20261003-0600; eight cores/32GiB/15min.
Candidate differs from frozen 222367 source only in arithmetic/lib.rs;
constructor reorder is excluded. Clean arithmetic and solver before building.
Remote larger input SHA64bcac82288dddf6505db00e22d0b417908c51cdefea11b59c371f190bfaa50c
verified. Frozen settings remain embedded; original audit and full point
identity required. Packet/evidence:
~/.cache/sdpx-e2e/streamline-narrow-fma-release-20261003-0600.

## 2026-10-03 — unused PSD scratch deleted

Jordan inverse no longer unpacks destination x into X: the existing symmetric
loop overwrites every matrix entry using only Z and lambda before packing X.
Removes per PSD sweep Ising4,754 writes/2,538 reads/2,216 discarded scale
multiplies; current Lambda52,493/26,852/25,641. Same used arithmetic/order.

Condensed BlockWork.vector now starts empty; both existing writers resize
then completely pack it before any reader. Sampled MPFR routes use matrices
only, so allocate no vector. Float64 factor application keeps its mandatory
svec roundtrip, and unsampled dense Schur assembly keeps its vector. Ising
serial shape changes over513 recorded sweeps imply~96.28MiB zero-buffer
allocation traffic/10,773 buffers removed; actual four-worker misses vary.
Max payload10,880B/worker(Ising16/512),126,336B/worker(Lambda47/768).
These structural work/storage counts are not RSS or speed measurements.

One combined Ising11/512/one-thread fast E2E: Solved52, complete point
identity and accepted original audit. Evidence:
~/.cache/sdpx-e2e/streamline-unused-psd-scratch-20261003/quick-gate.json.
External vector proof/patch: /tmp/sdpx-condensed-lazy-vector-20261003.

## 2026-10-03 — isolated decisions, exact-kernel trials and converter gate

PBS222366 completed four explicit-clean builds and20 serial ABBA runs. All
original audits and within-arm full-point repeats pass. G2 original coupling:
small53 native0.143786→0.156013s (+8.50%); larger53 1.355453→1.393373s
(+2.80%,36→37 iterations); small256 2.518020→2.503473s (−0.58%). Reject,
kept outside live tree. No accuracy policy changed to recover an iteration.

Unused adjoint panels/PSD roots: Ising512 native7.027142→7.130574s (+1.47%),
API7.027254→7.130678s, process7.097529→7.191769s; RSS73.900391→73.007813MiB
(−1.21%). Keep structural allocation/work removal and memory tradeoff, no
speed claim. MPI offset is not yet kept. Isolated GMP square: native
7.023570→7.047869s (+0.35%), API7.023679→7.047980s, process7.085238→7.108399s;
RSS74.582031→74.167969MiB with unchanged allocation strategy is not a clear
memory benefit. Reject; remove square dispatch/private specialization from
live exactdot. Slice-based norms remain. Source/results:
~/.cache/sdpx-e2e/streamline-next-release-20261003-0520/cluster-summary.json.

The job then failed before MPI execution: vendor Intel profile referenced
unset INCLUDE under nounset. Original exit1/logs preserved; PBS removed
dependent222367/222368 automatically. Fresh retry222369 reuses all four
binaries/20 measurements, temporarily disables nounset for vendor setup and
attempts only MPI then previously unstarted diagonal/FMA stages. Its MPI
solve completed, but audit setup failed before producing an audit file;
diagnosis pending. No MPI source patch accepted or full build rerun yet.

Narrow two-product FMMA trial shares scalar FMA shift/sign machinery (+46
net lines); retains two exact products, exponent-sum gap<=127, rounds once.
128/256 only; wider/zero/nonfinite dispatch and narrow distant-gap TLS path
unchanged. External adversarial primitive audit passes, 256 near46.190→28.204ns
and spread43.498→30.528ns. Primitive values do not establish solve speed.
Wide512 screen remains external: nearby rows−21.0%, tiny-angle+1.65%, distant
exponents+1.27%, partial success−3.58%, zero angle+2.51%. No current SVD angle
distribution measured; no wide FMMA solver claim or scalar FMA512 reopening.

Residue sampled quadratic trial multiplies upper-triangular M by Q using
DTRMM(L,U,N,N), after copying Q into existing T. Centered residues are integers
below2^(bits−1); existing h<=k_chunk=2^(55−2bits) bounds every signed partial
sum below2^53. BLAS order therefore preserves exact residues, CRT and final
rounding. Ising M*Q products141,178→75,338/prime/sweep; Lambda4,560,325→2,332,652.
New Q copies75,984/839,832B respectively. No whole-solve speed/storage claim.

Combined FMMA/TRMM Ising256/four threads: Solved52, complete x/s/z/sampled_y
identity, accepted original1e−30 audit. Existing audit script rejected256;
external copy adds that supported width while retaining every accuracy gate.
Original rejection log preserved, no repeated baseline solve. Evidence:
~/.cache/sdpx-e2e/streamline-fmma-narrow-20261003/{baseline-audit,quick-gate}.json.
Isolated source packets (one file per arm) prepared; no formal conclusion.

PMP B rows now collect checked decimal strings directly, removing the numeric
row Vec and one allocation per row. Same evaluations, rounding and decimal
contents; complete row remains checked before writing, cleanup unchanged.
README example128: all five converted files byte-identical to frozen converter;
converted output Solved20, original-coordinate audit accepted and analytical
optimum1 within1e−15. Evidence:
~/.cache/sdpx-e2e/streamline-pmp-row-20261003/quick-gate.json.

## 2026-10-03 — converter row reuse and audit setup repair

PMP B output now reuses one fixed-capacity Vec<String> per block across all
triangle(dim) × sample_count rows. Each row is fully evaluated, checked and
serialized before clear; individual decimal-string allocations remain. This
removes outer-vector allocation churn without changing rounding or output.
One128-bit/one-thread converter E2E: all five output files byte-identical to
the preceding converter, Solved20 and original-coordinate/analytical-optimum
audit accepted at1e−15. No solve or converter speed claim. Evidence:
~/.cache/sdpx-e2e/streamline-pmp-row-reuse-20261003/quick-gate.json.

PBS222369 exit1 was audit setup, not numerical failure: kept two-rank MPI
Solved52 completed, then Julia could not load JSON. Candidate MPI and the
diagonal/scalar-FMA stages never ran. Preserve original namespace/exit/logs.
Second and final scoped retry prepared in sdpx-streamline-repair2-20261003-0640;
explicit login Pkg.instantiate with precompilation disabled and a fresh first
depot passed, with frozen Project/Manifest hashes unchanged. Reuse the
successful baseline point/receipt after hash validation, audit it freshly,
then run candidate MPI and the two unstarted stages. No new whole build sweep.

Read-only gravity row evidence: row84 correction multiplier9.38130080676406
times the default negative static shift−1e−8 predicts−9.38130080676406e−8,
versus measured exact residual−9.380008677238313e−8. Bias-adjusted difference
1.29213e−11 is0.0138% of that residual. Actual factor shift/clamps still need
confirmation; this is an inference, not proof or a solver-quality fix. No
range issue appears in captured row values. Defer border equilibration:
power-of-two scaling normally reproduces the same rounded LDL, and general
scaling does not improve its componentwise bound merely by changing norms.
Prescribed regularization and refinement rules remain unchanged.

PBS222370 completed exit1 before MPI or any new build: fma/frozen.sha256
required upload.sha256, a transport checksum omitted from the archive. Root
payload hashes passed; actual nested archive coverage was incomplete. Audit
dependencies/imports are now ready, so this is a packaging failure. Preserve
all failure logs and source archives. The cluster skill's two automatic MPI
retries are exhausted; no third MPI job submitted. Prepare a concrete corrected
packet and continue the previously unstarted serial comparisons independently;
serial evidence cannot certify the MPI offset patch.
Pkg's first import precompiled its own stdlib cache in87s despite package
auto-precompilation being disabled; explicit audit imports then took14s.

## 2026-10-03 — gravity regularization residual confirmed

Four-line external diagnostic replay of the same frozen Float64/four-thread
gravity trajectory logs the actual factor shift/sign. Solved37, complete
x/s/z identity, prior accepted original-coordinate audit retained. At factor26/
refinement114 the applied shift is1e−8 and equality sign−1. Equality84 raw
residual−9.380008677238313e−8 versus static-shift contribution
−9.38130080676406e−8 leaves1.2921295257475887e−11, only0.013775% of raw error.
The prescribed regularization explains the dominant stalled row; residual
rounding was2.0078e−16. Equality18 correction residual is−5.1580e−11.
No further growth/range dump is justified by this evidence, and changing
convergence, regularization or refinement remains out of scope. Factor-only
equilibration stays deferred. Evidence: /tmp/sdpx-refine-shift-20261003/run/
{quick-gate,shift-analysis}.json. Diagnostic timing is not a performance claim.

## 2026-10-03 — serial comparisons resumed; wide FMMA screened

PBS222372 submitted8cores/32GiB/30min/node192. New serial-only parent
attestation verifies original222366 summary/binary/source and20 audited
rows; original whole-job failure remains1. Previously unstarted diagonal512
and scalarFMA256 source archives/gates are byte-identical. MPI offset remains
unaccepted. Real upload extraction and all nested payload checks passed;
transport checksum files are outside payload frozen lists. Evidence:
~/.cache/sdpx-e2e/streamline-serial-release-20261003-0700.

One external Ising512/four-thread count diagnostic: Solved52, complete
x/s/z/sampled_y identity and original1e−30auditPASS. Actual replay uses native
FMMA directly, so dot_fma2-only instrumentation would miss it. Observed1,144
replays/473,250rotations/7,037,369rowpairs: zero725,504; both-near6,024,057;
first-only27,621; second-only4,353; both-far255,834; special/unsafe-exponent0.
Eligible outputs85.8282%; near does not guarantee a gain (tiny-angle screen
regressed). Applying historical inclusive9.404% FMMA share andnear21.01%
primitive gain gives only~1.70% optimistic extrapolation before declines.
Defer512-bit wide FMMA unless a new matched profile provides stronger evidence.
Diagnostic timing is not usable as performance evidence. Artifacts:
/tmp/sdpx-fmma-svd-distribution-20261003/{quick-gate.json,README.md}.

SVD duplicate-expression refactor caches shift/d[lo] from its original guard
and passes that rounded quotient to the sole shifted_qr call. Operands and
all Demmel–Kahan/tiny-shift gates are unchanged; one full-precision division
per accepted shifted pass disappears. Ising512/one-thread fast: Solved52,
complete x/s/z/sampled_y identity and accepted original1e−30audit. This is
redundant-computation removal, with no measured speed claim. Evidence:
~/.cache/sdpx-e2e/streamline-svd-shift-ratio-20261003/quick-gate.json.

PBS222374 submitted/held afterok222372,8cores32GiB60minnode192. It builds
two isolated frozen arms and runs FMMA/Ising256, TRMM/Ising512 and
TRMM/Lambda768 ABBA sequentially. Original scalar-FMA/GMPsquare=true common
scope is preserved; live later edits excluded equally. Baseline is new
serial-only FMA stage; MPI acceptance is independent. Submission3 archive
SHA50bcdd006a2b35adc719e8be1a2a0629dc082184cc88c2bd3c4145cf1ff7cfbd.

MPI-only proposed third check is fully prepared/uploaded:31actual nested
checksum targets and Julia imports pass, no build,8cores32GiB5min. User
approval requested because ucas-hpc permits only two automatic retries;
no third MPI job submitted. Existing failures and all namespaces remain.

## 2026-10-03 — LP cost-normalization heuristic rejected

External one-line Ruiz cost guard AND→OR normalizes an LP/nonzero objective
and pure quadratic q=0; original nonzero/nonzero path stays identical. The
same AND exists in current upstream Clarabel.rs problemdata; rationale is
undocumented, so this was a heuristic trial, not a bug fix. Existing c-aware
recovery and all tolerance/regularization/refinement rules remain untouched.

Float64/four-thread small and larger gravity audits pass for both arms. Fast
screen small19→23iterations/native0.041586→0.047305s; larger37→38/native
0.443505→0.597182s. Timings are preliminary single observations, not formal
speed claims. More decisively, larger inner residual/solve count155→382
while unchanged outer refinement count114→117: cost scaling worsens the
regularized correction workload despite reducing the dual-coordinate shift.
Reject without a release build/sweep; default cost guard remains AND.
Evidence: /tmp/sdpx-ruiz-cost-20261003/screen-decision.json and both original
audits. This direction stays closed without new evidence.
Upstream: https://raw.githubusercontent.com/oxfordcontrol/Clarabel.rs/main/src/solver/implementations/default/problemdata.rs

Separately skip the provably zero P norm/mean on structurally empty P during
Ruiz. dwork is overwritten before reuse; objective guard and c update order
remain unchanged. Removes zero-fill/empty-column traversal/zero sum and one
division per pass, without an allocation or meaningful speed claim. This is
setup dead-work removal; matching MPFR gate pending.

## 2026-10-03 — setup gate and diagonal trial decision

Empty-P Ruiz norm skip: Ising512/four threads Solved52, complete
x/s/z/sampled_y identity against the one-thread quotient-refactor baseline,
original1e−30auditPASS. AND cost-normalization guard remains unchanged.
Setup dead-zero-work removal; no allocation/RSS or measured speed claim.
Evidence: ~/.cache/sdpx-e2e/streamline-ruiz-empty-p-20261003/quick-gate.json.

PBS222372 diagonal stage completes exit0; following scalar-FMA build runs.
Ising512/four physical cores serial ABBA native median7.1002421525→7.060563643s
(−0.5588%); API7.100345605→7.0606827335s; process7.224934403→7.121671025s;
RSS73.224609375→73.4609375MiB (+0.32%). All four Solved52/full seed identity
and accepted original audit. Reject leading sampled-diagonal slice branch:
no2% native gain, no allocation/footprint reduction (TLS term capacity was
already reused). Revert only that sampled/mod.rs branch; generic exact-norm
slice refactor remains. Frozen upcoming FMA/FMMA/TRMM arms retain this
obsolete code equally, so their isolated scope stays valid but is not a
latest-whole-tree certification. Evidence:
~/.cache/sdpx-e2e/streamline-serial-release-20261003-0700/diagonal/cluster-summary.json.

## 2026-10-03 — scalar FMA kept; sampled coefficients released

PBS222372 completes exit0. Narrow scalar FMA (128/256-bit, exact full
product/addend and one nearest-even rounding) is kept after four-thread
release ABBA: smaller gravity native median2.6205949855→2.562017252s
(−2.2353%), API2.6211167025→2.5625604865s, process3.056846518→2.926853616s.
Larger gravity native22.897270528→22.682627345s (−0.9374%); this is below
the speed acceptance threshold for that case. All eight runs exit0,
Solved25/33, complete point identity and accepted original-coordinate audits.
Peak RSS medians172.2402→169.0879MiB small and978.6719→981.4395MiB large;
no general memory claim. Frozen arms share obsolete square/diagonal trials,
so this comparison isolates FMA rather than certifying the latest tree.
Evidence: ~/.cache/sdpx-e2e/streamline-serial-release-20261003-0700/fma/cluster-summary.json.

Release retained_A numeric coefficients after sampled operator installation.
Reduced KKT already copied those values; later sampled nonempty A updates
are rejected at the public boundary, while ordinary A updates keep their
values and pattern. Ising512/four-thread fast Solved52, full x/s/z/sampled_y
identity and fresh original1e−30auditPASS. Removes515,200B (0.4913MiB) of
retained numeric storage for this input; no measured RSS or speed claim.
Evidence: ~/.cache/sdpx-e2e/streamline-sampled-retained-20261003/quick-gate.json.

PBS222374 now runs the frozen FMMA/TRMM comparisons after222372. Third
MPI retry remains unsubmitted pending the requested user approval.

## 2026-10-03 — FMMA decision and triangle-only storage

PBS222374 Ising256 stage completes four serial release runs at four physical
cores: native4.3307268385→4.2062329025s (−2.8747%), API4.330829663→4.2063300325s,
process4.468750425→4.263933320s. All Solved52/full seed point identity/original
auditPASS. Keep narrow two-product FMMA's exact stack kernel. Ising512 TRMM
native7.2582719375→7.191856043s (−0.9150%) does not qualify alone; wait for the
larger Lambda768 comparison, still running. Whole-job completion is pending.
Isolated frozen scope retains obsolete square/diagonal code equally; later
live changes excluded. Evidence: ~/.cache/sdpx-e2e/streamline-fmma-trmm-release-20261003-0630/partial-summary.json.

PSD sync now copies only authoritative upper G before the unchanged lower
mirror. All entries fully written with identical values/order. Ising512/four
threads Solved52, full x/s/z/sampled_y identity and original1e−30auditPASS.
Current Lambda removes25,641 scalar copies per sync (2.74MiB payload at768),
without a persistent-memory or speed claim. Evidence:
~/.cache/sdpx-e2e/streamline-psd-upper-copy-20261003/quick-gate.json.

PMP generated Hankel Cholesky stores row i with n−i entries, omitting the
unused lower triangle. Same arithmetic order; removes n(n−1)/2 MPFR cells
per active parity/worker. Five small128 converted files remain byte-identical;
converted-input solve/original audit pending. MPFR eigenvalues-only calls
also have a focused dead-reflector-storage candidate, with its gate pending.

Both gates now pass. Generated-basis PMP output solves at128/one thread,
Solved20, full x/s/z/sampled_y identity and fresh original1e−15audit including
the analytical optimum. Evidence: /tmp/sdpx-pmp-upper-basis-20261003/quick-gate.json.

Eigenvalues-only MPFR tridiagonalization skips packed reflector tails and
taus; later reduction uses only the trailing block, and only vector requests
consume packed reflectors. V requests stay unchanged. N scratch removes n
values; Ising largest initialized workspace saves1,280B at512 and current
Lambda saves5,264B at768. Full-sweep omitted stores include tau:175,520B/
2,868,656B respectively. Ising512/four threads Solved52, full point identity
and original1e−30auditPASS. Structural storage/dead-write benefit; no speed
claim. Evidence: ~/.cache/sdpx-e2e/streamline-eigen-values-only-20261003/quick-gate.json.

## 2026-10-03 — packed residue scratch kept; TRMM rejected

Exact diagonal congruence now packs only selected upper partial residues;
dense BLAS tiles/u/t stay unchanged. CRT folds at most16 primes per add,
bounding r active length by selected_entries*min(prime_count,16). Existing
absolute integer bounds keep grouped GEMM sums exact; frac retains original
prime order, one final rounding and original scatter. Also remove r.clear:
every retained r cell is overwritten before use. Gravity256/four-thread fast
Solved25, complete x/s/z identity and original1e−18auditPASS. Clear active
scratch/work reduction; pool capacities can remain larger, so no guaranteed
RSS or speed claim. Four-way partial lengths save40,800*prime_count bytes
small/161,600*prime_count larger; actual counts unrecorded. CRT-phase savings
are separate from the partial-phase maximum. Evidence:
/tmp/sdpx-rns-packed-upper-v2-20261003/quick-gate.json.

PBS222374 completes exit0, all twelve runs Solved52/119, full seed identity
and original auditsPASS. Narrow FMMA stays kept. Reject/revert only TRMM:
Ising512 native7.2582719375→7.191856043s (−0.9150%); Lambda768
428.7900560935→422.907335375s (−1.3719%), API428.7901958685→422.9074777355s,
process429.338523254→423.446107313s. Lambda RSS620084→622366KiB (+0.3680%);
no clear memory benefit and neither native gain meets2%. Packing/grouped
CRT edits remain. Frozen scope caveats still apply. Evidence:
~/.cache/sdpx-e2e/streamline-fmma-trmm-release-20261003-0630/cluster-summary.json.

User explicitly approved one third MPI retry. PBS222378 submitted after the
serial job completed,8cores32GiB5min/node192, MPI-only/no build. All18root/
13nested payload checks pass; corrected archive basename mpi-only-packet.tar.gz
matches SHA b036308014cbfe4581fe0d94d033b5570214611ee58d99769cedee676ac0e0b7.
Old failures remain unchanged. Offset acceptance awaits this real MPI gate.

## 2026-10-03 — MPI offset reuse accepted; denser cache screen deferred

PBS222378 completes MPI-only exit0. Actual panels candidate world2/partitions2/
two threads per rank: Solved52, complete x/s/z/sampled_y identity with the
preserved successful kept MPI point. Both receive fresh original512/1e−30
auditsPASS. Candidate process10.785s; baseline solve is reused, so this is
an ownership/correctness gate and not a speed comparison. Current
distributed/hsd_kkt.rs matched the frozen before byte-for-byte; apply only
the offset patch. constant_work.s now holds the offset after RHS packing,
removing99,200/107,040B per nonempty Ising owner, no RSS claim. Other later
source changes are not certified by this frozen comparison. Evidence:
~/.cache/sdpx-e2e/streamline-repair3-release-20261003-0650/mpi-results/gate.json.
Original222369/222370 failures remain unchanged.

External Pair20 cache prototype is deferred before a solver/cluster trial.
Exact codec checks pass; payload3N→5ceil(N/2) bytes saves~16.67% for prime
width≤20. Larger gravity y has2,040,200entries; saving1,020,100*cached_count
bytes, count unrecorded. Preliminary100k unpack screen remains5.48× slower
(7.669ms versus1.399ms/6.4M decoded values), including a compact bulk-iterator
attempt. No whole-solver inference/claim; no shared code or job. Existing
f32 Wide cache is exact: CACHE_MAX_BITS25 excludes26-bit primes before store.
Artifacts: /tmp/sdpx-rns-pair20-20261003/{audit-screen.json,integration-review.txt}.

## 2026-10-03 — Zero sampled RHS trial; shared PMP prefactors

MPFR sampled PSD RHS shortcut checks exact zeros, finite Rinv and every
compact Gram diagonal before skipping congruence and adjoint work. It fully
writes mat3c as +0 and performs the original ordered weight*+0 products,
preserving negative-weight signed zeros. Failed guards retain the original
path. Ising11 at 512 bits/four threads is Solved/52, with complete x/s/z/
sampled_y identity and a fresh original-coordinate audit PASS. A matched
release ABBA is being prepared; single-run timing supplies no speed claim
and the shortcut remains provisional. Baseline source was frozen before
applying this one-file change. Evidence:
/tmp/sdpx-zero-psd-rhs-20261003/{proof.txt,quick-gate.json}.

PMP conversion now borrows the original prefactor and sample-scale slice
when no reduced replacement exists. This removes repeated decimal parsing
(or the default exponential), a duplicate pole vector and one count-element
scale vector per applicable block. Explicit reduced values retain their
original parsing and arithmetic. All five converted files are byte-identical;
the 128-bit/one-thread solve is Solved/20 with complete point identity. The
preserved accepted original-coordinate 1e-15 audit is reused only after both
file and point identity. Kept for redundant work/storage removal; no measured
RSS or speed claim. Evidence:
/tmp/sdpx-pmp-shared-prefactor-20261003/quick-gate.json.

PBS 222420 submitted for this zero-RHS decision: 8 cores, 32 GiB, 20 minutes;
latest frozen baseline versus the single psd.rs change, two release builds
and one serial Ising512/four-thread ABBA with BLAS one. Packet SHA
36b9305a7f63ce0ea9bcf8201e45264e0d6e75274f69839cfee7638d7ad32bf1;
all 20 payload checks pass. Initial state queued. Later PMP borrowing and
bound-diagonal scratch edits are excluded equally from both frozen arms.
Evidence: ~/.cache/sdpx-e2e/streamline-zero-psd-release-20261003-1000/.

## 2026-10-03 — Bound diagonal shares solve scratch

Local bound Schur assembly now fills the existing solve RHS scratch with
its diagonal scales. Float64 removes recurring temporary diagonal vectors;
MPFR removes the separate persistent d vector and borrows vs[..leaf_count].
All later solves/refinement fully overwrite their live scratch, failed
refactors/fallbacks do not read it, and RNS consumes the borrowed diagonal
synchronously without caching it. Products, reduction order and BLAS/thread
geometry are unchanged. Small gravity four-thread Float64 and MPFR256 checks
are Solved/19 and /25, with complete point identity and original-coordinate
audits PASS at their unchanged gates. MPFR256 payload removed: 244,800 B
small and 969,600 B larger; Float64 recurring temporary payload: 40,800 B
and 161,600 B. Kept for storage/allocation removal; no speed or RSS claim.
Evidence: /tmp/sdpx-bound-diagonal-scratch-20261003/{review.md,quick-gate.json}.
This later patch is excluded equally from PBS 222420's frozen arms.

## 2026-10-03 — Finite empty quadratic trial

An unconditional empty-P HSD shortcut was rejected: quadratic forms still
evaluate zero times every coordinate, so division/subtraction overflow must
remain visible. Both xi transforms and Px are unchanged. Instead the CSC
quadratic form returns +0 only for empty coefficients and finite x/y;
otherwise the original loop remains identical. This replaces zero arithmetic
and column scans with finiteness predicates. Float64 243-case primitive
screen is bitwise equal; the proof covers every MPFR width and signed zeros.
Small gravity MPFR256/four-thread is Solved/25, complete points identical and
original 1e-18 audit PASS. Provisional pending matched release timing; no
speed/RSS claim. Evidence: /tmp/sdpx-empty-quad-form-20261003/quick-gate.json.

## 2026-10-03 — Zero multiplication primitive trials rejected

Two external MPFR zero-times-finite shortcuts preserve full storage and
sticky flags at 256/512 bits, including signed zeros, exponent limits and
the unchanged native 0*Inf/NaN path. Direct rustc OPT3 two-cycle primitive
ABBA screens show regular multiplication regressions of 13–24%; 512-bit
mixed workloads have no gain. Zero-only gains of 18–34% do not justify
retention. Both inline and outlined variants are rejected before any solver
edit/build/run; the live arithmetic source is unchanged. Evidence:
/tmp/sdpx-zero-mul-screen-20261003/README.md. These are primitive measurements,
not solver timings.

PBS 222420 completed with exit 0. All four Ising512 runs are Solved/52,
with identical complete points and accepted original-coordinate audits.
Native median 7.283575230→7.245817623 s (−0.5184%); API
7.283698340→7.245930858 s; process 7.435329426→7.309286140 s (−1.6952%).
RSS median 71,886→71,736 KiB is not a clear structural memory benefit.
Reject and revert the isolated zero sampled-RHS shortcut: gain is below 2%
and no persistent allocation is removed. Other PSD/eigen/RNS storage edits
remain. Evidence: streamline-zero-psd-release-20261003-1000/cluster-summary.json
under the external e2e cache. The subsequent frozen empty-quadratic packet
contains this now-rejected shortcut identically in both arms; its LP cases
have no sampled PSD blocks and do not execute it.

External diagonal-tile Gram GEMV is deferred after an Accelerate/BLAS-one
full-kernel primitive ABBA: 0.51250→0.50200 ms small (−2.05%, noisy),
2.56583→2.52450 ms larger (−1.61%). All lower entries are bitwise equal.
The kernel-level gain does not justify a solver gate; no shared edit/build
or solver run. Fixed tiles and current DGEMM remain. Evidence:
/tmp/sdpx-bound-gram-gemv-20261003/primitive/receipt.json.

PBS 222421 submitted after 222420 completed and the timing host was free.
8 cores, 32 GiB, 20 minutes; two release builds and three serial gravity
ABBAs (small MPFR256 primary, small/larger Float64 generality check).
Original tolerances and Float64 opt-in qnorm settings are unchanged.
All 31 payload checks pass, retained larger input SHA matches and original
audit imports work. Packet SHA e9f094c8872ae63515f7a179778e7a697f033e9f2035d8c236b9ebad0710947c.
Observed running/current solver baseline compiling. Evidence:
~/.cache/sdpx-e2e/streamline-empty-quad-release-20261003-1010/.

## 2026-10-03 — MPFR finite-empty quadratic kept

PBS 222421 completed exit 0; all 12 runs are Solved with same-host complete
points identical and original-coordinate audits PASS. Primary gravity256
native median 2.561517950→2.489656290 s (−2.8054%). Float64 small/larger
medians regress 2.153%/5.000%, with visible timing variation. Cached-binary
repeat 222422 failed before any solver: importing the driver's auto-main
created its old results path. The failed record remains. Repaired 222423
loads definitions only and completes exit 0, all 12 points/audits PASS.
MPFR repeat 2.584382886→2.486555736 s (−3.7853%); combined two-batch
medians 2.561517950→2.487710376 s (−2.8814%), API
2.562012505→2.488241016 s, process 2.958617792→2.840901744 s.
RSS 169,610→171,520 KiB (+1.126%); no memory benefit is claimed.
Float64 repeat small −0.187% gain/larger +3.432% gain; combined native
small/larger regress 0.247%/1.496%, so no reliable Float64 benefit.

Keep the finite-empty path only for MPFR (precision_bits>53), with identical
MPFR operations and proof; Float64 uses its original loop. This is arithmetic
backend selection, without parameter/name tuning. xi transforms, Px,
nonfinite behavior and numerical rules remain. Frozen 222421 contains the
now-rejected zero-PSD shortcut identically/inactive in both LP arms. Evidence:
~/.cache/sdpx-e2e/streamline-empty-quad-{release-20261003-1010,repeat-20261003-1045}/cluster-summary.json.

## 2026-10-03 — Equality plus orthant chunk reuse, quick gate

The existing orthant chunks now also serve exactly one positive-size MPFR
orthant with equality cones. Existing pool/admission/worker budgets and all
Zero callbacks remain; no new option or scheduling layer. Step bounds retain
original cone order and the existing MPFR minimum semantics. Mixed Float64
and MPI early routes are unchanged. Small gravity256 at one/four threads
is Solved/25 with identical full x/s/z/sampled_y and fresh original 1e-18
audits PASS. Fast timings are preliminary, not formal speed claims.
Evidence: /tmp/sdpx-zero-orthant-chunks-20261003/{proof.txt,quick-gate.json}.
Source is frozen for isolated small/larger256 four-thread release comparison
under streamline-zero-orthant-release-20261003-1045 in the external cache;
subsequent Float64 output-scratch reuse is excluded from both frozen arms.

## 2026-10-03 — Product normalization read-only trial rejected

External round_product avoids mutating the 2N-limb exact product and reads
shifted high limbs/guard/sticky directly. Full-storage/native nearest-even
checks at 256/512/768 and all-N proof pass, including midpoint/carry/range
and special paths. OPT3 mixed products regress 23.56% at 512 and 9.77%
at 768; reject before any shared apply/build/solve. Product/GMP work and
allocations are unchanged, and the GMP shift profile is not attributable
to this Rust pass. No precision tuning or extra gate. Evidence:
/tmp/sdpx-mul-normalize-read-20261003/{README.md,primitive-receipt.json}.

## 2026-10-03 — Float64 leaf outputs hold forward intermediates

Remove BoundPanels.work: forward leaf solves write their intermediates to
disjoint solution x leaf slots; the intervening trunk solve writes only
trunk IDs, and backward solves read each leaf before overwriting it.
All products, FMA order, subtract-zero operations and arbitrary RHS-column
indices remain. Small gravity Float64/four-thread is Solved/19, complete
points identical and fresh original 1e-6 audit PASS. Logical two-RHS
payload removed: 163,200 B small / 646,400 B larger, plus a Vec header.
Keep for storage removal, no speed/RSS claim. Evidence:
/tmp/sdpx-bound-forward-output-20261003/{proof.txt,quick-gate.json}.
This later patch is excluded equally from frozen orthant222424 arms.

PBS 222424 submitted/running:8 cores,32 GiB,20 minutes, two fresh release
builds and small/larger gravity256 four-thread sequential ABBAs, BLAS one.
Original1e-18 audits/settings and requested precision are unchanged.
All29 payload hashes pass; retained large input/full seed hashes and audit
imports verified; node192 free at submission. Runtime full source/cache/
input/seed checks PASS, baseline solver compiling. Only two cone files
differ; restricted empty-quadratic path is common, rejected zero-PSD absent.
PacketSHA33ff55f7a1d699132414f2db8e6bb16ad58ca380b070eab75790e96929785f48.
Evidence:~/.cache/sdpx-e2e/streamline-zero-orthant-release-20261003-1045/.

## 2026-10-03 — Zero vector scaling deferred after primitive screen

A batch-level zero coefficient path avoids per-element native MPFR
multiplication for finite values while preserving canonical zero storage,
signs, special-value native paths and sticky flags. Standalone OPT3 full
storage/flag gates pass at256/512; vector zero kernel improves about81%
with unchanged regular-vector timing. This does not change live scalar
multiplication or repeat the closed per-product zero shortcut. The current
gravity zero-scale path is only Px per residual; 132,652 small-case elements
at the measured primitive cost imply under0.33% of fast native time. Pool
scheduling remains, so no plausible2% total gain demonstrated. Defer before
shared edit/build/solve; no solver performance claim. External evidence:
/tmp/sdpx-zero-vector-scale-20261003/{candidate.rs,primitive-receipt.json}.

## 2026-10-03 — Fused Ruiz norm scan rejected

One external CSC scan accumulates row and column max norms with a shared
absolute value, preserving each reduction's entry order and P prefix.
Norm arrays are wire-identical for actual Float64 shapes and the small
MPFR256 screen, including special/empty cases. Three primitive ABBA cycles:
small Float640.4065→0.3569 ms (noisy), larger0.9937→1.3337 ms (+34.2%),
small2562.2264→2.1560 ms (−3.16%). Ten Ruiz passes project only0.704 ms
MPFR saving (about0.028% of cluster native2.5 s); larger Float64 regresses.
Reject before shared edit/build/solve, without size tuning. Evidence:
/tmp/sdpx-fused-ruiz-norms-20261003/receipt.json.

## 2026-10-03 — Fixed-dispatch verification prototype

A temporary copy of the existing CLI replaces only its precision macro
import/invocation with MPFR256 dispatch. The input/setup/engine/export body
remains unchanged; no published source or six-width precision list changes.
Warm-cache solver --lib build selects matching artifacts, then standalone
rustc uses fast-equivalent O3/16 codegen units, taking9.54 s. This is a
build experiment, not a matched compile or solver speed claim. Small
gravity256/four-thread is Solved/25 with exact complete points/backend
matching the full CLI and a fresh original1e-18 audit PASS. Compiled library
source manifest equals the full-CLI bound-output arm. No permanent helper
or frontend option added. Evidence:
/tmp/sdpx-fixed-dispatch-20261003/{build.json,quick-gate.json}.

## 2026-10-03 — Equality plus orthant parallelism kept

PBS 222424 completed exit 0. All eight runs are Solved/25 or /33, complete
accepted points identical and original-coordinate 1e-18 audits PASS.
At MPFR256/four threads, small gravity native median
2.520070578→2.376522952 s (−5.6962%); API 2.520563124→2.377018952 s,
process 2.937642764→2.711543206 s. Larger gravity native
22.334561223→21.385767833 s (−4.2481%); API 22.337694272→21.388856159 s,
process 24.853630249→23.866449878 s. Keep the isolated two-cone-file patch.
RSS small 167,654→169,408 KiB (+1.0462%), larger
989,632→992,686 KiB (+0.3086%); this is a speed benefit with a small memory
tradeoff. Child process RSS can include inherited parent pages.

Both frozen arms have the restricted MPFR empty-quadratic path, exclude the
later Float64 forward-output change and omit the rejected zero-PSD shortcut.
These numbers measure this patch, without a cumulative whole-tree claim.
Larger scaling falls 0.4246→0.1065 s; Schur assembly and iterative refinement
remain about 7.27/6.75 s and are the next substantial costs. Evidence:
~/.cache/sdpx-e2e/streamline-zero-orthant-release-20261003-1045/cluster-summary.json.

## 2026-10-03 — Zero-addend scalar FMA rejected

One external Ising11/MPFR512/four-thread diagnostic records 19,192,136 scalar
FMAs; 1,624,733 (8.4656%) have zero addend and eligible regular products.
Solved/52, complete points identical and fresh original 1e-30 audit PASS.
Historical inclusive scalar FMA CPU share is 5.40%; frequency weighting
suggests only 0.457% CPU even for complete elimination, not a measured
wall-time bound. MPFR already routes these zero-addend cases to multiplication.
Reject before a candidate, primitive comparison or live edit. Counter probes
affect diagnostic timing, which is not performance evidence. Evidence:
/tmp/sdpx-fma-zero-addend-20261003/{diagnostic.json,README.md}.

## 2026-10-03 — Parallel arrow diagnostic counters need no vector

Each started leaf keeps its existing regularization count on the stack and
adds nonzero counts to one joined diagnostic atomic. The Result collector,
pivot decisions, leaf arithmetic and fallback inputs are unchanged; successful
totals are independent of scheduling. One/four-thread small Float64 gravity
are Solved/19 with complete accepted points identical and fresh original
1e-6 audits PASS. Keep for removal of a recurring 40,800 B small / 161,600 B
larger temporary plus its zeroing and summation. No measured speed/RSS claim.
Regularization-heavy settings can contend once per affected leaf; no atomic
updates occur when the leaf needs no dynamic regularization. Evidence:
/tmp/sdpx-arrow-counter-20261003/{arrow-counter.patch,proof.txt,*gate.json}.
The solver library build and temporary fixed-width verification CLI use the
same fast profile and feature set; published frontend dispatch is unchanged.

## 2026-10-03 — Coupled leaf suffix and mutable IR scratch kept

Leaf v/batch_v retain only coupling_start..g, with all readers remapped.
Forward/backward w buffers remain complete; retained products and coupling
dot order are identical. Local SOC also trims its zero prefix; generic and
shared-SOC leaves retain start zero. Exact initial v allocation and first
batch reserve_exact avoid Vec::resize's four-element minimum capacity.
For single-bound gravity at MPFR256 and one single plus two batched RHSs,
active payload removed is 734,400/2,908,800 B small/larger. Requested Vec
capacities change from 4+4 to 1+2 values per leaf: 1,224,000/4,848,000 B
less capacity. These are capacity counts, not allocator usable size or RSS.

Direct iterative refinement negates its already mutable current/candidate
point, evaluates the same exact row terms, then restores the point before
every normal return. Double owned MPFR negation restores all storage fields;
pool joins complete before restoration. Immutable residual_full retains its
old temporary wrapper. Larger256 removes a 1,944,048 B temporary per IR
residual. A second sign-flip pass is added, so no speed benefit is assumed.

One batched build, then small gravity256/four-thread Solved/25 and
Ising11/512/four-thread Solved/52; complete accepted x/s/z/sampled_y identical
and fresh original 1e-18/1e-30 audits PASS. Keep both storage changes. This
gate also contains CRT product-buffer reuse, exercising identity selections;
nontrivial upper compaction still awaits an actual frozen Lambda19/768 cluster
gate. No new timing/RSS claim, and no new MPI certification. Evidence:
/tmp/sdpx-ir-leaf-crt-20261003/{source.json,build.json,gravity-gate.json,ising-gate.json};
/tmp/sdpx-{arrow-coupling-suffix,ir-inplace-neg,crt-prod-reuse}-20261003/.

PBS 222426 submitted, observed queued: 8 cores, 32 GiB, 20 minutes.
One frozen default-six-width release build with validated dynamic OpenBLAS,
then full Lambda19/768 and larger gravity256 profiled solves sequentially at
four physical cores/BLAS one. Lambda covers nontrivial upper CRT compaction;
gravity supplies current exact residual/Schur costs. All 22 payload checks,
Python3.6 syntax, retained inputs/full seed hashes pass; node192 was free.
Audits are bound to the same input/settings/precision and reused only after
complete x/s/z/sampled_y equality. Profiling adds overhead and wrapper RSS;
no speed/RSS comparison is claimed. A later parallel Result collection cleanup
is excluded from this frozen packet. Packet SHA:
46e9f8b99210a66c3f7fbb73e044ddeb6d6dea16136c67972da67a624208d6b2.
Evidence: ~/.cache/sdpx-e2e/streamline-crt-profile-20261003-1135/.

## 2026-10-03 — Arrow unit-result collection removed

Parallel leaf refactors use try_for_each instead of collecting Result<Vec<()>>.
Rayon1.12's Result collector uses WhileSome without opt_len, so Vec collection
allocates LinkedList nodes per completed task even for zero-sized units.
try_for_each keeps stack consumers and the existing best-effort cancellation.
Every started refactor publishes its diagnostic count after Ok or Err, and
joins complete before fallback; raw coefficients define fallback as before.
Successful numerical work/counts are identical. One/four-thread small
Float64 gravity are Solved/19, complete points identical and fresh original
1e-6 audits PASS. Keep for collection/allocation removal, no byte total or
speed/RSS claim. This later change is excluded from PBS222426's frozen source.
Evidence: /tmp/sdpx-arrow-result-collect-20261003/{result-collect.patch,proof.txt,*gate.json}.

PBS222426 observed running; frozen source/cache/input/settings/full seed/audit
checks PASS, fresh solver release compilation underway.

## 2026-10-03 — Upper CRT product reuse validated; current profiles retrieved

PBS222426 completed exit 0 after the fresh six-width release build. Lambda19/
768/four-thread is Solved/119 and larger gravity256/four-thread Solved/33.
Complete x/s/z/sampled_y equal their retained accepted seeds; input/settings/
precision/raw seed/full-point hashes and original audits are bound and checked.
Accepted original audits are reused only after actual complete field equality.
The Lambda gate exercises nontrivial upper selection, completing CRT product
reuse acceptance. Keep for removal of a separate group-residue buffer and
copy; no measured speed or RSS benefit is claimed. Later arrow try_for_each
is excluded from this frozen source.

These are single profiled runs, not matched timing comparisons: native/API
Lambda425.128567/425.128735 s, larger gravity21.536642/21.539822 s. Perf wrapper
process-tree peaks608,788/995,576 KiB are not isolated solver RSS comparisons.
Raw profiles remain on the cluster; receipts, complete points, audit bindings
and symbol/caller reports are retrieved locally. No samples were lost.
Flat cycles:u shares: gravity exact-residual closure22.47%, DGEMM kernel13.83%,
diagonal-congruence closure10.96%; Lambda GMP basecase multiply26.82%, DGEMM
kernel15.94%, CRT finish4.41%, chunk encoding3.26%. These CPU shares identify
work to investigate; they are not wall-time speedup bounds. Evidence:
~/.cache/sdpx-e2e/streamline-crt-profile-20261003-1135/cluster-summary.json;
results/*.{receipt.json,symbols.txt,stacks.txt,audit.json}.

## 2026-10-03 — Diagonal prime-group streaming rejected

An isolated candidate moves existing 16-prime CRT groups around diagonal
congruence row work. It bounds D residues and each way's partials to one
group, preserving every per-prime block/way/reduction order and output point.
It also adds group barriers, repeated D encoding GEMMs and scratch checkouts;
D chunks and CRT state now overlap row work. Static payload formulas show
tall shapes can improve but short/wide single-way shapes can worsen, so this
is not a general memory reduction.

Small gravity256 at one/four threads is Solved/25 in both frozen fast arms,
complete points identical and fresh original 1e-18 audits PASS. Preliminary
native one-thread1.981710→1.980009 s, four-thread0.830916→0.870374 s;
process peaks134.969→139.156/155.984→157.250 MiB. Single fast runs do not
establish a measured speed/RSS change. This screen offers no useful gain
to justify its extra scheduling or further release work. Reject and restore
only the isolated RNS file to SHA9b59a725e5919aeb01fa986e7382daf6bf95f2b87749d2fdbab165c71bffb969.
No settings, precision, convergence or other edits changed. Evidence:
/tmp/sdpx-diag-prime-stream-20261003/{review.md,decision.json,baseline,candidate}.

## 2026-10-03 — Slice-dot fallback aligned; bound coupling comparison submitted

The slice helper now routes nonfinite/window-declined sums through the
generic exact dot before its rounded fallback, preserving eligible-zero
filtering and underflow zero signs. Unsupported widths/unavailable scratch
keep their original direct chain. Exact finite arithmetic is unchanged and
scratch is restored before fallback reentry. Keep this small correctness
alignment with the existing same-value Scalar contract.

The bound-coupling trial uses existing slice dots for single and batched RHSs.
Packed batch values change from interleaved rows to contiguous columns;
each dot retains ascending leaf order. Single solves and Schur weights
refill/read the first leaf-count prefix. Generic/shared SOC and Float64
panels are unchanged. Reviewer finds no layout/fallback/TLS issue. Small
gravity256/four-thread is Solved/25, complete accepted points identical and
fresh original1e-18 audit PASS. Fast native0.816845 s is preliminary, not a
gain claim. Evidence: /tmp/sdpx-bound-couple-slices-20261003/.

PBS222429 submitted, observed queued:8 cores/32 GiB/20 minutes. Two frozen
default-six-width release builds and eight sequential fresh processes form
one ABBA per small/larger gravity256/four-thread case, BLAS one. Both arms
contain the fallback fix and rebuild arithmetic; only arrow coupling differs.
Earlier kept changes through69 are common; rejected prime streaming and
later prototypes are absent. All25 payload hashes (26 archive entries),
Python3.6 syntax and PBS syntax pass; node192 was free and no duplicate task
job existed. Packet SHA73964605ff01b4dbb05fd01899514ce4e571c6ec32d8e019e2f1b0abacc84f53.
Evidence: ~/.cache/sdpx-e2e/streamline-bound-couple-slices-20261003-1306/.

## 2026-10-03 — Wide two-product FMMA remains deferred

The fresh Lambda768 profile motivates rechecking the existing exact window
at512/768. External guard/row candidates pass native-oracle checks including
ties, cancellation, signs, exponent/range fallbacks and both row outputs.
Matched separate-binary primitive nearby-row medians512193.712→156.596 ns
(−19.16%),768305.582→268.029 ns (−12.29%). Other row classes show little
benefit and small768 regressions. Discard the earlier biased one-binary
old-helper comparison. Reported Lambda FMMA inclusive7.03% times nearby gain
suggests only0.864% synthetic all-near CPU saving; this is neither a rigorous
ceiling nor a solver prediction, and the actual eligibility mixture/host
effect is unknown. No storage is removed. Defer without live edits, crate
builds, solver gates or another cluster job. No new kernel is justified.
Evidence: /tmp/sdpx-wide-fmma-20261003/{README.md,matched-row-receipt.json}.

## 2026-10-03 — Replayable exact residual dots rejected

An external candidate scans and replays cloneable row iterators instead of
staging exact-dot terms. It preserves current finite values, fallback zero
signs, full storage and MPFR flags in 64 comparisons at256/512; finite sums
also match the native exact-dot oracle. Matched same-row primitive medians
regress at256 by15.78%/17.53%/22.17% for4/104/20,202 terms and at512 by
1.95%/5.12%/4.71%. Repeated indexed scans outweigh the existing reusable
pointer staging. Reject without a shared API/source change, crate build,
solver run or cluster job. These are primitive results, not solve timings.
Evidence: /tmp/sdpx-replay-dot-20261003/{replay-dot.patch,proof.txt,result.txt}.

## 2026-10-03 — Bound coupling slices rejected after matched release solves

PBS222429 completed exit0; fresh baseline/candidate six-width release builds
take6m38s/6m28s. All eight serial ABBA processes on small/larger gravity256
at four physical cores, BLAS one, are Solved/25 or33. Complete x/s/z/sampled_y
match accepted seeds; bound original1e-18 audits pass. Both arms contain and
rebuild the same slice-dot fallback correctness fix; only arrow coupling differs.

Small native/API medians2.338173/2.338724→2.347662/2.348176 s (+0.41%);
process2.772499→2.687822 s (−3.05%), child RSS167634→166888 KiB (−0.45%).
Larger native/API21.423684/21.426837→21.502933/21.506112 s (+0.37%);
process23.890360→23.953210 s (+0.26%), RSS988378→998948 KiB (+1.07%).
Process variation supplies no native solver gain; storage shape is unchanged.
Reject and restore only arrow SHAf5ebee7e37b27f3fb8d37b2bf89bab8a1657a71d5d55475abfa660c3ad51bb5e.
Keep common arithmetic fallback SHA297acc65452ee529cf32729f8f81ae5660d686e0e0164f9e58d0b3ce5ce7a06f.
No extra solve is needed after restoring the frozen baseline already gated.
Results/receipts/audit bindings/build logs/exit records retrieved and checked;
full outputs remain on the cluster. Linux wait4 child peaks can include
inherited pre-exec pages; no solver-only RSS benefit is claimed. Evidence:
~/.cache/sdpx-e2e/streamline-bound-couple-slices-20261003-1306/cluster-summary.json;
/tmp/sdpx-bound-couple-slices-20261003/decision.json.

## 2026-10-03 — CRT packing screen: fusion rejected, packing deferred

An external streamed bit bucket preserves24,615 sign/carry/spill cases and
improves isolated contiguous packing primitives12.57–28.54%. Combining it
with lazy per-digit CRT correction removes y writes and an unsafe pointer
wrapper, but matched strided reconstruction primitives regress255–463%.
Complete outputs and failure prefixes match. Reject this fusion without a
shared edit, crate build, solve or cluster job.

One root screen isolates packing with the original correction loop. The
first candidate process has a large startup transient; a no-rebuild ABBA
resolves it. Warm medians improve31-output512 layouts11.75–12.65%,768 upper
layouts2.91–8.42%, while the all-nonzero4371-output class is+0.98%. All
checksums match. These include strided digits, fraction rounding/correction,
packing, actual magnitude conversion and scatter, excluding reset/checksum.
Arguments are dynamically blackboxed and the same arithmetic rlib is used.
This is not a complete solve; the4.41% finish profile is not packing-only.
No convincing total-solve gain or meaningful storage benefit is established.
Defer packing without a live patch or solver run;8 B less logical scratch
per finish range is no RSS rationale. Evidence:
/tmp/sdpx-crt-pack-stream-20261003/{FINISH_REVIEW.md,finish-primitive-receipt.json,finish-pack-only-receipt.json}.

## 2026-10-03 — Regular-first exact-dot scan: small gate passed, trial pending

The classifier checks regular operands before eligible-zero products, so
ordinary pairs skip zero-kind predicates. Filtering, term order, exponent
window, fallback and scratch lifetime are unchanged. All128 full-storage/
MPFR-flag comparisons agree, including unavailable TLS and negative underflow
zero; finite sums match native exact-dot. Optimized specializations are six
instructions smaller. Matched row primitives show256 finite gains0.81–5.24%,
mixed-zero0–2.40%;512 is mostly neutral, including small mixed regressions.
These are arithmetic screens, not solve gains.

Apply only exactdot trial SHA07136e7abd265fffe47f34b708884deb34ea239dfdf1a2d068aaab5e9598e16c.
One small gravity256/four-thread fast E2E is Solved/25; complete x/s/z/sampled_y
match the accepted seed and fresh original1e-18 audit passes. Native0.818301 s
is preliminary. Live arrow is the restored kept baseline; RNS is unchanged.
The default six-width frontend remains unchanged; only the external quick
verification executable instantiates256. A release decision is pending;
reuse the exact249-build-file baseline already rebuilt in222429, then build
only a frozen candidate and compare small/larger256 serial ABBAs. Evidence:
/tmp/sdpx-dot-kind-order-20261003/{screen.json,frozen-hashes.json};
/tmp/sdpx-dot-kind-solver-20261003/fast/{build.json,gravity256-t4.gate.json}.

PBS222434 submitted, observed queued:8 cores/32 GiB/12 minutes. Reuse the
222429 baseline binary only after complete249-build-file manifest identity,
source/binary hashes, exit0 and its accepted solve/audit evidence are checked.
Fresh candidate rebuilds arithmetic+solver in a task-owned cache copy; normal
six-width release CLI, validated dynamic OpenBLAS. Two small/larger256 serial
ABBAs at four physical cores/BLAS one; precision, settings and audits unchanged.
All26 payload files and actual source archive extraction, Python3.6 syntax,
PBS syntax and upload hashes pass; node192 was free, no duplicate task job.
Only exactdot differs. Packet SHA472030bbf509b365481bfb767746dbf9c22b71bfc0ef6b16fe54b782a5192f55.
Cluster clock lags local time; archive timestamp warnings are not failures.
Evidence: ~/.cache/sdpx-e2e/stdstreamline-dot-kind-release-20261003-1410/.

## 2026-10-03 — Regular-first exact-dot classifier kept after release comparison

PBS222434 completed exit0; candidate rebuilt in6m33s, verified frozen baseline
reused. Eight fresh serial processes are Solved/25 or33 and full x/s/z/sampled_y
match bound seeds; original1e-18 audits pass. Both gravity256/four-thread
ABBAs use the same physical cores and dynamic BLAS one. Small native/API
medians2.365602/2.366111→2.297663/2.298172 s (−2.87%); process2.815907→2.638411
s (−6.30%), child RSS169240→166950 KiB (−1.35%). Larger native/API
21.397124/21.400342→21.484356/21.487478 s (+0.41%); process23.858191→24.009244
s (+0.63%), RSS993122→992998 KiB (−0.01%). Keep the general classifier on
the qualifying small native gain, record the larger regression without
size/precision tuning. Peak variation is not an isolated memory benefit.

Only classifier ordering differs from the reused baseline; later power-of-two
and wide-FMMA trials are excluded from this formal evidence. Candidate binary
SHAd9d0275decee923655ad1d6ed574a50f362c828b8997e79077007188c360e9ac.
Retrieved summary/receipts/audit bindings/build/source/exit records agree.
Evidence: ~/.cache/sdpx-e2e/stdstreamline-dot-kind-release-20261003-1410/cluster-summary.json.

## 2026-10-03 — Exact-dot power-of-two and capped wide-FMMA gates passed

The dot-only product trial recognizes either normalized power-of-two
mantissa and constructs its identical2N-limb product by shift/copy. Global
multiply/FMA are unchanged; alignment, signed accumulation, final rounding
and flags stay identical. N1/4/8 full-storage/flag comparisons pass; native
exact-dot oracle applies to admitted finite windows, while unchanged wide
window/TLS fallback is compared against its existing filtered FMA semantics.
Primitive short-row power mixtures improve, but ordinary104-term rows
regress3.53% at256 and1.58% at512. No uniform or solver gain is assumed.

The existing wide-FMMA proposal is reconsidered for a real512 solve: profile
caller shares are not reliable bounds and the near-row primitive512 gain is
19.16%. Limit both attempt guards and private invariant to12 limbs; the old
MAX32 proposal would also change unmeasured1024+ widths and is not applied.
N≤4 and N>12 keep their original routes. Existing full512/768 native-oracle,
sign/tie/cancellation/window/range/row checks apply. This shares the retained
narrow inline policy, not a new MPFR sticky-flag promise. Both row outputs
are computed before either store; partial success retains native fallback.

One batched fast library/temporary256+512 verification CLI build, then small
gravity256/four-thread Solved/25 and Ising11/512/four-thread Solved/52. Full
accepted points match and fresh original1e-18/1e-30 audits pass. Native
0.840176/3.804564 s are preliminary; default six-width frontend is unchanged.
Both trials remain unkept pending one scoped cluster release comparison;
regular-first classifier is common. Evidence:
/tmp/sdpx-dot-power2-product-20261003/{README.md,primitive-receipt.json};
/tmp/sdpx-wide-fmma-20261003/{limited-proof.txt,limited-wide-fmma.patch};
/tmp/sdpx-exact-arithmetic-bundle-20261003/fast/{build.json,gravity256-t4.gate.json,ising-gate.json}.


PBS222436 submitted, observed queued:8 cores/32 GiB/12 minutes,node192.
One fresh six-width candidate build; verified222434 candidate binary is the
baseline after source/cache exit/hash proof. Only exactdot/lib arithmetic
files differ. Three sequential ABBAs, small/larger gravity256 and Ising512,
four physical cores/BLAS one. Complete Ising point identity is mandatory
before accepted original1e-30 audit reuse; gravity original1e-18 unchanged.
All30 payload hashes, extracted source manifests, Python3.6/PBS syntax and
upload hash pass. Live node192 was free; no duplicate task job. No Lambda,
MPI, sweep or numerical-policy change. Packet SHA
652ad45ab9b249c5b3d581ead5170819b5d6220bd8cf8f8721b1c6ea6a1775a6.
Evidence: ~/.cache/sdpx-e2e/stdstreamline-exact-arithmetic-release-20261003-1424/{manifest.json,submission.json}.


## 2026-10-03 — Exact-dot power products and wide FMMA rejected

PBS222436 completes exit0; one six-width candidate build6m34s. All12 fresh
processes are Solved/25,33 or52 with complete accepted-point identity and
original1e-18 gravity/1e-30 Ising audit binding PASS. Same four physical cores,
BLAS one. Native/API medians: small2562.405072/2.405549→2.382296/2.382828 s
(−0.95%); larger25621.465954/21.469053→21.602672/21.605781 s (+0.64%);
Ising5127.224560/7.224652→7.187799/7.187915 s (−0.51%). Process medians
2.822294→2.722983,23.949879→24.071314,7.290584→7.256393 s. Child RSS
167534→168604,992522→990596,72342→72720 KiB; no storage benefit.

Reject both trials: no qualifying native/API solve gain or meaningful memory/
correctness benefit; short-row primitives do not justify retention. Small
process improvement does not repeat across cases or solver timing lanes.
Restore only exactdot to kept75 SHA07136e7a... and lib to4750e273... after
byte-exact trial/baseline checks, preserving filtered fallback/classifier and
others' changes. No Lambda, isolation rebuild or extra verification needed:
the restored frozen baseline already passes these actual solves. Candidate
binary SHA28fec42a2732f157f9b9b6e9ffb93661b4d0bb83ed76212e88f1b6a8fbe2c7f9.
Retrieved summary/receipts/audit bindings match; retrieval SHA
21f97a72a69a6f212524e6d42a85bc27797dceecb49ca6d871cade51a3543958.
Evidence: ~/.cache/sdpx-e2e/stdstreamline-exact-arithmetic-release-20261003-1424/cluster-summary.json;
/tmp/sdpx-exact-arithmetic-bundle-20261003/decision.json.

## 2026-10-03 — Current Float64 gravity/MOSEK refresh submitted

PBS222437 submitted/observed queued only after222436 completed:8 cores,
16 GiB,5 minutes,node192. Reuse verified kept222434 binary/source manifest;
no rebuild. Four sequential SDPX/MOSEK/MOSEK/SDPX ABBAs, actual distinct
small5102/larger20202 models at one/four physical cores, BLAS one. SDPX
explicit opt-in qnorm1e-6; historical default audit failure stays labeled.
MOSEK11.2.2 native lower bounds exactly reconstruct original NN rows; q,
equality coefficients and b unchanged. Read back actual optimizer/thread/
tolerances; original1e-6 audits run per distinct complete point outside timers.
Native optimizer/API setup/process clocks and child RSS remain distinct.
Twelve payload hashes, actual gzip/input mapping, syntax and upload checks
pass; existing MOSEK environment import succeeds, license acquired only in PBS.
Packet SHAaf66e43ca0387613bc2a677f777b8834ddb337add4df88575ff27f13e62a3a4a.
Evidence: ~/.cache/sdpx-e2e/stdstreamline-gravity-f64-refresh-20261003-1436/.

Benchmark research documentation is concise English; E2E documentation now
states fast build defaults and the medium opt-in/default audit distinction.
Only documentation changed; git diff --check passes.

## 2026-10-03 — Float64 gravity/MOSEK refresh completed

PBS222437 completed exit0 on node192, no build. All16 fresh serial processes
are Solved and all original1e-6 audits pass; repeated points match within
each solver/thread arm. Same small5102/larger20202 inputs, explicit SDPX
qnorm1e-6, MOSEK11.2.2 settings readback, one/four physical cores, BLAS one.
Native / setup-inclusive API / whole-process medians in seconds:

| Model / threads | SDPX native / API / process | MOSEK native / API / process | Iterations SDPX / MOSEK |
|---|---|---|---|
| Small /1 |0.142878 /0.143631 /0.221136|0.130044 /0.225713 /0.461692|19 /19|
| Small /4 |0.147198 /0.148004 /0.230793|0.108574 /0.192023 /0.433580|19 /19|
| Larger /1 |1.914302 /1.917414 /2.206585|1.160389 /1.716947 /2.878135|35 /21|
| Larger /4 |1.379647 /1.382930 /1.679530|0.976168 /1.541082 /2.737939|36 /21|

Larger SDPX one→four improves27.93%; small slows3.02%. Larger native ratio
SDPX/MOSEK1.65×/1.41× at one/four; small1.10×/1.36×. MOSEK API includes
Python model conversion/setup and optimize; SDPX API uses its native setup
and reported solve clock. Frontends differ, so native/API/process comparisons
remain distinct. Native timing includes each solver's own clock scope.
The iteration gap persists without a demonstrated numerical-policy-safe fix.
Do not compare historical129/25.4 s as a matched gain; default audit failures
remain historical failures, not newly certified by these opt-in runs.

Wait4 child RSS mediansKiB: small all arms179316; larger SDPX one/four
179316/211196, MOSEK364532/364542. Parent's parsed-input peak179316 is inherited
before exec, censoring smaller child peaks; native getrusage sees the same
floor. This is not reliable small-solver memory evidence or an isolated
memory benefit. Full outputs remain remote. Retrieved summary,16 receipts,
eight distinct full-point audit bindings and exit files agree. Retrieval SHA
f74cf799363579111fda215c830f01ba101cf767cf63d1373f590c11a92efdf1.
Evidence: ~/.cache/sdpx-e2e/stdstreamline-gravity-f64-refresh-20261003-1436/cluster-summary.json.

## 2026-10-03 — Duplicate diagonal-congruence reserve trial

Remove fixed160-bit scaling-spread reserve from the exact Aᵀdiag(d)A modulus
bound: the generic operand cache already retains eight spare prefix primes.
The actual3P+2da+dd+log2(k)+3 bound and CRT margin remain; prime width,
exact integer products/sums, once-rounding, owner invalidation and numerical
policies are unchanged. Growth beyond cache count uses the existing encoding
path. This removes overlapping reservations without a model or precision knob.
One fast256/four-thread small-gravity solve is Solved/25, complete accepted
x/s/z/sampled_y identity and fresh original1e-18 audit PASS. Native0.791704 s
is preliminary, not a matched gain. Candidate is unkept pending two-size
release comparison. RNS SHA9b59a725...→9af3ecb3...; arithmetic07136/4750 common.
Evidence: /tmp/sdpx-diag-cache-reserve-20261003/{manifest.json,proof.txt,fast/}.

Scalar power-of-two multiply remains external/deferred. Full storage/native
flags proof passes256/512/768; ordinary primitive products regress6.73/1.63/
1.83%, while controlled25–100% power mixtures improve. Actual solver scalar
eligibility is unknown; no solver/RSS claim or live arithmetic change.
Evidence: /tmp/sdpx-scalar-power2-mul-20261003/{README.md,primitive-receipt.json}.

PBS222441 submitted after upload/hash/source/PBS preflight and live node192
free/normal queue checks;8 cores32 GiB12 minutes. One fresh six-width candidate
build, verified222434 kept baseline reused, two serial small/largergravity256
four-thread ABBAs with fresh original1e-18 audits per distinct full point.
Only rns_blas differs; no Ising/Lambda/MPI/settings or wider campaign. All24
payload hashes pass. No duplicate task job; interrupted SSH checks preserved,
upload completed successfully before no-overwrite rename/submission. Packet
SHAa577da6cddc3112281b4a8df3156bdf650ad1ff0919df2257db6f5f8f0c588b4.
Evidence: ~/.cache/sdpx-e2e/stdstreamline-diag-cache-release-20261003-1449/submission.json.

## 2026-10-04 — Diagonal reserve removal kept after release gate

PBS222441 completed exit0 on node192; six-width candidate build6m33s, verified
kept222434 baseline reused. Eight fresh serial solves, two gravity256/t4
ABBAs, same four physical cores/BLAS one. All complete x/s/z/sampled_y equal
bound seeds and each other; fresh distinct original1e-18 audits pass. Small
native/API2.345343/2.345877→2.263761/2.264293 s (−3.48%); process2.795615→
2.631629 s (−5.87%); wait4 child RSS167288→161572 KiB (−3.42%). Larger
native/API21.449212/21.452416→20.544601/20.547775 s (−4.22%); process
23.932321→23.058827 s (−3.65%); child RSS986156→955568 KiB (−3.10%).
Wait4 scope may include inherited parent pages; no isolated byte attribution
or wider-precision speed claim. Keep the actual-spread reserve simplification,
without a model/precision parameter. Only rns_blas differs; later single-
product81 is excluded. Candidate binary SHA
a3dd2308fb3ed66b2c1517cda59a5d0916dab6f7b855fdf23435981eb4e4d1df.
Retrieved eight receipts/full-point/audit bindings and exit files agree. Archive
SHA1dc0a726cb8244625edb600f0edd99ee892697b9d5df5a6cbaec939ed035c34b.
Evidence: ~/.cache/sdpx-e2e/stdstreamline-diag-cache-release-20261003-1449/cluster-summary.json;
/tmp/sdpx-diag-cache-reserve-20261003/decision.json.

## 2026-10-04 — Single-product diagonal Gram scratch kept

Prime width guarantees k_chunk>=k until the minimum18-bit clamp, where
k_chunk=2^19. Diagonal row blocks are at most4096 rows, so every block fits
one exact BLAS product: the second m² product buffer is never read. Inline
that first product; retain one initialized m·min(m,32) buffer for upper tiles
(m² for full output), overwritten completely by beta0 without repeated
clear/resize. Reduce only selected entries before the same nested modular
addition; BLAS shapes/operands, exact integers, CRT, partitioning and numerical
policies are unchanged. A debug assertion states the chunk invariant.

One smallgravity256/t4 fast gate is Solved/25, complete accepted-point
identity and fresh original1e-18 audit PASS. Native0.813979 s is preliminary,
not a matched gain. Initialized product scratch per way falls2m²→m·min(m,32):
four-way logical removal114240/549440 bytes at borders51/101. Pooled capacity
and process RSS can differ; no RSS/speed claim. Keep for dead storage/work
removal, including larger general borders. Live RNS78+81 SHA
525010ebef1927ec169db444a50d38d604cba2e1b03c8cb5c45d9c4a6c41afe7.
Evidence: /tmp/sdpx-diag-single-product-20261003/{proof.txt,decision.json,fast/}.

Exact binary64→MPFR constructor80 stays external/deferred: native complete
storage/flags proof passes22,068 cases/arm at64/512, including tightened
ranges and invalid precision; Ising geometry has9,820 half conversions per
complete Schur update (nominal510,640 at52, retries not instrumented). Lambda
set_d flat0.01% supplies no qualifying workload opportunity. No solver build,
shared arithmetic change or counter registration scaffold.
Evidence: /tmp/sdpx-f64-conversion-20261003/{README.md,manifest.json}.

## 2026-10-04 — Balanced residue-product trial

Balanced integer residue operands below p/2, odd p<2^26, give |x*y/p|<2^24.
Reciprocal/product rounding error<2^-28 cannot cross the nearest half-integer
distance1/(2p)>2^-27; SHIFT rounds to the true nearest integer quotient.
The residual already lies in the balanced range, so remove two corrective
comparisons ONLY for diagonal scaling and CRT inverse normalization. General
BLAS-sum reduction stays. An181125-case bounded integer oracle/full-f64-bit
check covers actual18..26-bit primes, near-half quotients, endpoints/random
pairs and signedzero. An external draft lost signedzero by converting an
integer reference product; repaired before shared code changed.

Fast smallgravity256/t4 is Solved/25, complete accepted points identical and
fresh original1e-18 audit PASS. Native0.804005 s is preliminary; no speed
claim/retention decision until independent proof review and scoped release
comparison. Kept78+81 common; RNS525010eb...→29a6965c.... Scalar eligibility
diagnostic builds from frozen81 externally and never changes shared sources.
Evidence: /tmp/sdpx-balanced-residue-product-20261004/{manifest.json,proof.txt,proof.log,fast/}.

Independent82 review approves both current sites: reciprocal error<2^-29
plus product rounding≤2^-30 preserves the quotient margin. All five CRT
callers provide balanced reduced residues, as do cached A/d and symmetric
inverse u. Signedzero sequence is unchanged; arbitrary BLAS sums keep general
reduction. Preparing one formal81+82 bundle comparison against kept78-only
222441 source; no isolated82 gain claim.

## 2026-10-04 — Actual Ising scalar multiply eligibility

External diagnostic from frozen kept78+81 source, shared code unchanged.
Four relaxed counters reset/read immediately around actual solve(), including
worker tasks and excluding setup/serialization. Ising11/512/t4 Solved/52,
full accepted x/s/z/sampled_y identity and fresh original1e-30 audit PASS.
Scalar multiply calls51,191,124; initially admitted regular49,350,851; exact
power eligible4,857,105 (9.49% total/9.84% initially regular); thin-top-limb
operand6,476,778 (12.65% total). Atomic/scanning overhead invalidates timing/
RSS comparisons; counts do not include internal FMMA/exactdot/RNS products
or establish a solver gain/caller cost. Global power shortcut remains an
external proposal until an actual complete-solve comparison; no production
telemetry/API remains. Temporary512binary SHA
230d5a403ff0b19ec4fc2fb86bf1f6af9a36c822dfa0c4dc27a319455c76aea6.
Evidence: /tmp/sdpx-mul-eligibility-20261004/run/diagnostic.json.

The diagnostic replaced shared target library artifacts, not shared sources
or the shipped CLI executable. Frozen production binaries remain usable;
rebuild the matching production library before linking a new CLI.

## 2026-10-04 — Residue product bundle submitted

PBS222444 queued after live node192/queue check, packet/current249-file
source identity and remote32-payload hash preflight. Eight cores32 GiB12
minutes, one default six-width candidate build, completed222441 kept binary
reuse. Three serial small/larger gravity256 and Ising512 ABBAs, t4/BLAS1.
Only RNS differs: single-product scratch81 plus balanced-product82 versus
kept78; no isolated82 claim. Fresh gravity audits; Ising audit reuse requires
actual complete-point equality and exact input/settings/precision/seed/audit
bindings. No Lambda, MPI or settings change. Packet SHA
8053c29de21ec135826d96a367d7c46598ae64f2247a3d0d82f0007d5d20e147.
Evidence: ~/.cache/sdpx-e2e/stdstreamline-residue-product-release-20261003-1621/submission.json.

## 2026-10-04 — Sampled scratch trial and CRT index borrowing

Trial83 removes repeated clearing before full writes in sampled quadratic
and congruence scratch, and applies reviewed balanced residue-product
normalization to sampled weights. Independent lifetime/rounding review passes.
Memory84 borrows one immutable sorted output selection across streamed prime
workers rather than cloning each Vec: logical8·ways·selected.len() bytes and
those allocations removed. Synchronous joins, compaction, merge and final
addresses are unchanged. Diagonal packed/final selections stay alive; no
diagonal byte saving claim. Keep84 for removed copies after one Ising512/t4
fast gate: Solved52, complete bound-point identity, fresh original1e-30 audit
PASS. Native3.801946 s preliminary;83 still needs matched performance evidence.
First build E0597 exposed Drop order of the diagonal final binding; repaired
by declaring it before the accumulator, initializing at the existing remap
phase. Failed build retained, no solve attempted. A temporary helper metadata
SHA naming error happened after successful compile; repaired receipt from
verified frozen sources/existing binary, without rebuilding. Shared production
artifacts explicitly restored after external79; binary SHA
74a87250cbddfe61067b980d8f0ea7a75f5d6de45e28a50c6bf1e484b0865a76.
Evidence: /tmp/sdpx-crt-selection-borrow-20261004/{decision.json,run/gate.json}.

External79 actual Ising512/t4 paired fast screen: native3.904318→3.799252 s
(−2.69%), both Solved52, complete points identical, fresh1e-30 audits PASS.
No counters; explicit frozen-origin arithmetic/solver recompilation. Promising
for a scoped release comparison, no repeatable gain or production retention.
The previously measured ordinary-product regression remains a relevant risk
for gravity; native full-solve scope decides, not primitive interpolation.
Evidence: /tmp/sdpx-power2-mul-ising-20261004/decision.json.

## 2026-10-04 — Residue product bundle kept; next trials gated

PBS222444 completed exit0; six-width release build6m32s,12 fresh serial
solves, node192/four physical cores/BLAS1. Small native/API2.284858/2.285376
→2.263909/2.264431 s (−0.92%); larger20.475820/20.479033→19.777691/
19.780851 s (−3.41%); Ising5127.242045/7.242152→7.290033/7.290157 s
(+0.66%). All complete points match bound seeds and each other; gravity
fresh original1e-18 audits pass, Ising1e-30 accepted audit is reused only
after actual complete-point/input/settings/precision/hash equality. Keep
81+82 bundle for the qualifying larger gain and dead scratch benefit; no
isolated82 or cumulative speed claim. Child RSS small158850→160622 KiB
(+1.12%), larger961692→959774(−0.20%), Ising72700→71980(−0.99%); inherited
parent pages can affect wait4 peaks. Candidate binary SHA
b7e4db70241d055f759b3398588e095fd069f776778ac3bfc228090303fd30b2.
Remote summary SHA01e0f1e65ffedb5bd6f5a3e509c8159f1a6a193844a9e7d6fb5748d07d08c6a5.
Receipt/audit archive retrieval is pending intermittent SSH; completed
outputs remain remote, with archive SHA
d35f9d8109e6f7599370d1d886dc5c0c6be80e6acf896858169228f8cc96858c.
Evidence: ~/projects/sdpx-residue-product-release-20261003-1621/results/summary.json.

Later trial79+83 with kept84 also passes a smallgravity256/t4 fast gate:
Solved25, complete bound-point identity, fresh original1e-18 audit PASS.
Native0.756649 s is preliminary, not a matched gain. Existing512 Ising79
and83/84 gates pass. Source lib548ca49a/RNS6256e01; no precision, convergence,
regularization or refinement changes. Preparing one formal three-case bundle
comparison versus verified222444 candidate; no isolated79/83 gain claim.
Evidence: /tmp/sdpx-power2-combined-gravity-20261004/fast/gravity256-t4.gate.json.

222444 receipt retrieval recovered: exact archive/summary/binary identities,
all12 receipt/status/complete-point/audit/input/settings/precision bindings
and completed exit files pass local validation. Full raw outputs remain
remote. Evidence: ~/.cache/sdpx-e2e/stdstreamline-residue-product-release-20261003-1621/cluster-summary.json.

## 2026-10-04 — Balanced residue sum trial

Trial85 replaces quotient reduction only for sums of two balanced integer
residues in diagonal partials, diagonal merges and chunked GEMM accumulation.
Their exact sum has |sum|≤p−1<2^26; one recenter gives the unique balanced
residue. General GEMM-sum quotient arithmetic is unchanged; extracted center
shares its existing comparisons. Independent proof/site review and180945
integer-oracle/full-f64-bit checks pass, including signedzero and18..26-bit
primes. No precision/plan/partition/CRT/final rounding change. This trial was
applied after the next79+83+84 packet freeze and is excluded from that
comparison; no speed/retention claim yet.
Evidence: /tmp/sdpx-balanced-residue-sum-20261004/{manifest.json,review.txt,proof.log}.

## 2026-10-04 — Scalar / sampled scratch bundle submitted

PBS222447 queued after live node192/normal check, rootpacket/current249
identity and remote28-payload hash/PBS preflight. Eight cores32 GiB12
minutes. One fresh default six-width candidate build, explicit arithmetic
and solver clean; completed222444 candidate baseline/12rows reused with
exact summary/binary/source/exit validation. Three serial small/largergravity256
andIsing512 ABBAs, t4/BLAS1, unchanged audits/inputs/settings. Bundle79+83+84
only; later85 is excluded, no isolatedgain. No Lambda/MPI/sweep. Packet SHA
7bf42b5493653b5253377e6c453d4a9ecea1cbd6aced3986e7585d0642894e28.
Evidence: ~/.cache/sdpx-e2e/stdstreamline-rns-scalar-release-20261003-1640/submission.json.

85 is restored/deferred after local screening. Both forms Solved25, full
points/fresh1e-18 audits pass. First native0.861161 s; branchless signed
correction0.755571 s versus preceding0.756649 s baseline, all preliminary.
No clear complete-solve opportunity or memory benefit established; no
release claim/85 retention. Both candidate/proof/gate artifacts preserved.
Generic reduction remains unchanged; kept RNS6256e01 restored before next
sampled build. Evidence: /tmp/sdpx-balanced-residue-sum-20261004/decision.json.

## 2026-10-04 — Higher-order inverse-adjoint staging removed

Sampled dim>1 exact bilinear results write directly to caller output, then
retain the same per-pair weight*value multiplication/order. quadratic now
resizes only for dim1 duplicate-column projection; declined/partial CRT
fallback still assigns every output. No input/output alias, scheduling, MPI
exchange, operator or numerical-policy change. Removes one persistent
M=basis_cols·dim(dim+1)/2 value buffer per eligible block:80M bytes at512,
112M at768, plus its allocation/initialization. No measured speed/RSS claim.

One existing smallmatrix768/t1 fast E2E exercises four dim2 sampled parity
blocks (side18/16, rank34). Solved58, complete accepted x/s/z/sampled_y
identity and fresh original1e-30 audit PASS. Native5.257547 s is preliminary,
not compared with September history. Settings/input/seed/reference/binary
hashes bound; no conversion or extra tests. A local helper import failure
before any solve was repaired using the existing e2e import path; preserved
source/build/gate records. Keep the storage removal. Sampled source
87b39b9f...→4132ac0d...; binary SHA
c2be4d5f76518f8a14919c1c16fced83fea04b186e5bf2fa5cb363a8f068252c.
This later change is excluded from PBS222447 and has no later MPI acceptance.
Evidence: /tmp/sdpx-inverse-adjoint-direct-20261004/{decision.json,run/gate.json}.

## 2026-10-04 — Review pass: sampled kernel dedup, timer/knob cleanup, docs

Baseline arm `review-base-20261004-fast` (working tree at session start, Mac
M4, fast profile): medium Solved/19 2.14 s, csdr3 Solved/57 7.1 s, ising11
Solved/52 12.2 s (4 threads 3.76 s); all audits PASS. Points
`fd1437ee79ac39e0` / `0f810778fc459536` / `616bcced31fa1be3`.

Kept (pure refactors, identical points):

- Sampled adjoint/forward pair kernels: the diagonal `wdiag` dot (written 4×),
  off-diagonal panel dot (3×) and forward pair product (2×) in `sampled/mod.rs`
  and `split.rs` now share `diag_pair_dot`, `offdiag_pair_dots`,
  `adjoint_pairs` and `forward_pair`. ising11 512-bit points identical at 1 and
  4 threads; the dim2 matrix SDP (768 bits, Solved/58) x/s/z/sampled_y hash
  `6af13c11488e2f1c` identical at 1 and 4 threads.
- Receipt timing: manual `Instant` + `cpu_start`/`cpu_add` pairs (refactor,
  trsv/ir, residual, sync, assemble, cones_schur) use `receipt::start/finish`;
  `cpu_add` is removed. `PHASE trsv … ir …` and `PHASE sync … (inner_sampled)`
  lines become ordinary `PHASE` lines.
- Removed diagnostic env knobs `SDPX_TRACE_IR` and `SDPX_SERIAL_QDLDL` (no
  script used them; the latter was also hashed into the MPI agreement).
- `MpFloat::min/max` merge two identical branches.
- Arm `review-s2-fast`: all three pinned points identical to baseline.

Rejected (Float64 condensed Schur, medium, 1 thread, three or four
alternating runs without receipts, identical points everywhere):

- Streamed dense Schur assembly: transform the dense axis in spans of whole
  transform chunks (same chunk boundaries, same per-entry FMA order) instead
  of storing every column's packed suffix. Peak RSS 213 → 180 MiB (per-block
  span buffers) or 153 MiB (per-thread buffers), but native time +2.5% /
  +5.7% (2.14 → 2.19 / 2.26 s); the extra time is in the tile publish pass.
  4M-element spans: RSS −7%, time +4%. Multi-span correctness was checked with
  a 4K-element budget (identical 1- and 4-thread points). Not kept: Float64
  speed is the first priority and the memory gain does not offset it here.
- Precomputed previous-alias index replacing the per-column `(a0..a).find`
  scan in `accumulate_column`: −0.7% (2.142 → 2.125 s), below the 2% gate.

The MPFR SVD rotation replay is ~10% of ising11 time (all native
`mpfr_fmma` at 512 bits), but wide FMMA was already rejected (PBS222436,
Ising512 −0.51%); not reopened.

## 2026-10-04 — g0 SOC benchmark (medium): exact sparse products

Input: `SDPX_g0_benchmark` medium (12 modes, 1040 cells; m 1586, n 179,
nnz(A) 50830; 144 SOC3 + one 1154-row orthant; `condensed_shared_soc_arrow`).
Gate: the benchmark's independent 256-bit original-coordinate audit
(`audit_g0_bigfloat.jl`, target 1e-25). Mac M4, fast profile, one thread.

| Change (cumulative) | 256-bit native | iterations | audit dual residual |
|---|---|---|---|
| start of session | 6.65 s | 117 | 2.15e-30 (accepted) |
| orthant Schur: one exact accumulation per entry (precomputed plan) | 4.97 s | 117 | 2.15e-30 |
| MPFR `Aᵀx`: one exact accumulation per column (≥4 entries) | 4.01 s | 117 | 2.15e-30 |
| MPFR `Ax`: rows bucketed per call, one exact accumulation per row | 3.58 s | 117 | 2.15e-30 |
| orthant quotients via one reciprocal per row | 3.38 s | 117 | 2.15e-30 |

Same objective 2.7265361946230821 and status at every step; primal residual
5.3e-44 → 3.5e-44. These change MPFR rounding (fewer, exact accumulations),
so points differ from before; pinned csdr3 and ising11 still pass their audits,
medium (Float64) keeps point `fd1437ee79ac39e0`.

Float64: the orthant assembly formed `A/w` once per pair; forming it once per
row is bitwise identical: g0 53-bit 0.20 → 0.15 s, same AlmostSolved point
(the stall at dual residual 2.75e-6 is pre-existing; MOSEK also fails the gate).

Settings finding (no default change): the MPFR Ruiz bound default
eps^(1/4) (≈5e-20…2e19 at 256 bits) costs this model 117 vs 65 iterations.
With `equilibrate_min_scaling=1e-4, equilibrate_max_scaling=1e4` the 256-bit
solve takes 1.95 s / 65 iterations and the audit improves (dual 2.0e-33, gap
4.0e-32). ising11 also passes with those bounds (50 vs 52 iterations), but the
wide default was introduced for Λ19 spins 0–50, which stalled with 1e4, so a
data-driven default would need that case re-validated first.

## 2026-10-04 — g0 medium: condensed SOC elimination, dense panels (Float64 vs MOSEK)

Structure of the g0 medium input: 144 private columns h (objective 1, one
SOC3 plus one bound row each) and 35 shared columns y that appear in every
SOC and in 1010 dense orthant rows. MOSEK 11.1.3 (one thread) takes 39
iterations, 0.028 s native; SDPX Float64 took 39 iterations, 0.20 s.

Kept (Mac M4, fast profile, one thread; native seconds, warm runs):

| Change | Float64 | 256-bit | Notes |
|---|---|---|---|
| start | 0.20 s, 39 it | 6.65 s | |
| orthant rows sharing one column pattern → one SYRK | 0.067 s | 2.69 s | |
| condensed form eliminates SOC dim ≤ 16 via explicit W⁻¹ (H⁻¹ = η⁻²(2JwwᵀJ − J)); orthant and SOC rows form one Gram BᵀB: SYRK on columns present in ≥ half the rows, sparse pair plan for the rest | 0.053 s, 46 it | 2.79 s | replaces the retained-SOC arrow; reduced KKT is the 179-column arrow-structured Schur |
| Float64 A products through a dense panel of A's ≥25%-dense columns (condensed solves) | 0.034 s | — | |
| CSC SYMV keeps y[col] in a register (bitwise identical) | 0.028 s | — | inner-refinement residual |
| panel-only rows written directly, no panel zero-fill | ~0.027 s | — | bitwise identical |
| dense panel also in the per-iteration residual products | 0.025–0.026 s | — | |
| wide precision: panel-only orthant rows as Aᵀdiag(1/w²)A with cached residues | — | 2.58 s | |

Accuracy (independent g0 audit): with SOC elimination Float64 ends
AlmostSolved at 46 iterations with gap 4.9e-11 and dual residual 2.1e-6
(MOSEK: Optimal, gap 5.6e-10, dual 8.7e-6; old SDPX: gap 5.2e-6, dual 2.75e-6).
The audit's primal residual uses the reported s: MOSEK's runner reports
s = b − Ax (residual ≈ 0 by construction); SDPX reports its iterate s, whose
consistency with x on tiny-scale rows is 1.6e-4. The slack b − Ax is in the
cone for both. 256-bit stays Solved/117 with dual 2.15e-30, accepted.
Pinned medium (point fd1437ee79ac39e0), csdr3 and ising11 still pass.

Measured and not kept: dropping the condensed inner refinement (0.034 →
0.030 s) — refinement levels were kept by an earlier decision (2026-09-24).

## 2026-10-04 — Arrow residue Gram rejected; panel thread invariance

Arm `gram` replaced the per-leaf exact-dot border contributions of the generic
arrow (MPFR) by one exact residue Gram Σ YᵀD⁻¹Y over all owned leaves, with
residues summed across ranks by `MPI_Allreduce(SUM)` before one CRT finish.
Λ27, 1024 bits, 3 iterations, 32 threads/node (UCAS):

| Arm | Nodes | IP | refactor | contributions / gram | allreduce |
|---|---|---|---|---|---|
| dist5 (exact dots) | 1 | 150.5 s | 77.4 s | 45.2 s | — |
| gram | 1 | 149.9 s | 81.4 s | 49.6 s | — |
| dist5 | 2 | 142.7 s | 49.8 s | 21.5 s | 0.7 s |
| gram | 2 | 138.8 s | 52.6 s | 30.1 s | 7.1 s |

A local sample (mixed Λ11, 768 bits) puts the time inside the residue
products themselves (encoding and CRT < 1%). No gain; the Gram path, the
`RankReduce` trait and the MPI sum handle were removed. The 4-node run
(222592, hosts node186/189/191/70) printed no PHASE lines in 25 min; node70
is the known bad host.

Test-suite repair after the g0 work (all 442 + 201 tests pass):

- Float64 residual products used the dense panel only when the sparse row
  plan was absent, i.e. at one thread; results depended on thread count. The
  panel now takes precedence at every thread count (replicated MPI keeps the
  row plan); g0 points identical at 1 and 4 threads (`be316de8fc` Float64,
  `8cb93bb840` 256-bit).
- Test oracles follow the kept rounding: orthant quotients via one reciprocal
  per row, MPFR entries rounded once from the exact sum; SocElim blocks are
  eliminated like orthants; the shared-SOC arrow test uses the augmented
  form (condensed eliminates SOC3).

Pinned after these changes: medium Solved/19 point `fd1437ee79ac39e0`,
csdr3 Solved/57 PASS, ising11 Solved/52 PASS; ising11 at 1/2/3 MPI ranks
Solved/52 (6.4/4.1/4.4 s, 2 threads each).

## 2026-10-05 — Multi-node Λ27 after 0.9.0; SDPB at matched precision/threads

Input: mixed Λ27 (m 544,653, n 18,703, 117 sampled PSD blocks, arrow border
524), 1024 bits, 32 threads per node (SDPX: one rank per node; SDPB: 32 ranks
per node), UCAS cluster, OpenMPI over ib0 TCP, node70 excluded (it hangs
multi-node jobs: 222592, 222616, 222619, 222620 were deleted for that).

SDPX 0.9.0, 3 iterations (`wall.solve` includes the start iteration):

| Nodes | wall.solve | IP (3 it) | refactor | kkt solve | scale cones | process |
|---|---|---|---|---|---|---|
| 1 (222600) | 190.7 s | 152.4 s | 81.0 s | 13.6 s | 24.0 s | 5:52 |
| 2 (222621) | 162.2 s | 137.0 s | 50.5 s | 24.5 s | 19.3 s | 5:33 |
| 4 (222602) | 132.9 s | 109.5 s | 31.1 s | 20.7 s | 14.8 s | 5:20 |

Per-rank load 65–77 s and setup ~90 s (presolve_rank 24 s) are replicated.
Scaling stops at the parts every rank repeats: border solves, residual
scaling and sampled products in the KKT solves (~45 s at 4 nodes, not
shrinking), m-vector gathers (11.6 s), and cone-scaling tails.

SDPB (same input via `pmp2sdp --precision 1024`, thresholds 1e-30), cumulative
solver time after iterations 1/2/3: 1 node 112/255/398 s (process 12:20);
2 nodes 39/105/181 s (4:52); 4 nodes 54/111/195 s (5:22). A 4-node run
sharing node31 with the 1-node job took 477 s and is discarded.

Full solve, 4 nodes: SDPX (arm with rejected balance changes, numerically
identical to 0.9.0) Solved in 42 iterations, solve 1608 s, process 28:59
(job 222615).

Rejected (same-allocation A/B, 4 nodes, 3 iterations, job 222629:
0.9.0 128.4/124.9 s vs 132.4 s):

- Fused sampled RHS under MPI (owners condense/recover, adjoints gathered):
  works at 1–4 ranks but disables forward reuse in the residual (26 instead
  of 12 forward products) and doubles recover scaling; 4 nodes 161.9 s vs
  132.9 s.
- Cost-balanced (h²·kmax) block partition for sharded sampled products,
  largest-first cone scaling under MPI with inner splitting, and rank-local
  scaling tiles: bitwise identical, no gain (sampled adjoint and solves
  slightly slower).
- Leaf cost model by coupled border width: leaves finish unevenly
  (`arrow.border_sum` wait 2 → 6 s on rank 0).

### MPFR thread-count dependence in sparse products (bug in 0.9.0, fixed)

0.9.0 made MPFR CSC `gemv` exact per output (columns with ≥ 4 entries; rows
when nnz ≥ 4m) but the pooled `SparseParallel` lanes and the rank-sharded
`product_sharded` kept the rounded per-term chain. One-thread and pooled runs
therefore differed: ising11 512-bit point `fecc053686f5` at 1 thread vs
`2930df70fec0` at 4 threads (both Solved/52, audits pass); mixed Λ11 differed
likewise; g0 256 was unaffected. Fix: one per-output kernel
(`csc::wide_output`) used by gemv, the pooled lanes, the sharded product and
the residual products (the separate all-exact residual kernel is removed).
After the fix ising11 and Λ11 points agree at 1 and 4 threads
(`fecc053686f5`); g0 256 changes to `01a08d6b4f` (Solved/117, audit accepted,
primal 7.5e-44); medium/csdr3/ising11 audits pass. A regression test
(`wide_exact_parity_*`) covers long columns and dense rows; the old
`kernel_equivalence` matrix only had short columns.

### Kept after 0.9.0: aligned single-exchange products (same allocation A/B)

Measurement method: `aba.pbs` runs A B A B in one PBS allocation (4 nodes ×
32 threads, exclusive request, node70 excluded); cross-allocation runs vary
±5% and are not used for decisions.

| Arm (job) | wall.solve A/B/A/B | Verdict |
|---|---|---|
| coupled columns (222634, 2 nodes) | 161.0 / 155.6 / 161.7 / 160.8 | −2.0%, refactor −10%: kept |
| coupled columns (222635, 4 nodes) | 132.3 / 132.9 / 135.0 / 128.4 | −1.4%: kept |
| + exact-rule fix, sharded linear part (222640) | 132.2 / 141.7 / 131.7 / 142.8 | +8%: linear forward gathered all m rows |
| + aligned partition, active-row linear forward (222647) | 132.3 / 128.1 / 131.1 / 132.4 | −1%: forward rows ran 8 tasks only |
| + one task per active row (222652) | 124.8 / 118.9 / 124.7 / 115.2 | −6.2%: kept |

The sampled linear part of Λ27 is 544,653 × 18,703 with 5.2 M entries in
525 rows (the equalities). Aligned partition: the sampled products use the
condensed scaling partition, so prepare skips the scaling exchange, recover
skips the forward exchange, and the residual exchanges `ez` once; the
partial `H·z` is gathered once when copied out after refinement. Points are
bitwise identical to the non-aligned path at 1–4 local ranks.

Full solves, 4 nodes × 32 threads, 1024 bits: SDPX Solved/42, solve 1608 s,
process 28:59 (222615); SDPB "found primal-dual optimal solution"/125,
solver runtime 2592 s, process 43:17, 689 MB per process (222630, node3/4/5/7).

### Measured cone costs and packed Grams (same allocation A/B, 4 nodes)

| Arm (job) | wall.solve A/B/A/B | Notes |
|---|---|---|
| align4 vs measured cone costs (222659) | 121.0 / 117.5 / 119.5 / 115.5 | SVD CPU per rank 173/148/186/202 → 172/168/174/182 s; kept |
| align4 vs + packed upper Gram exchange (222664) | 123.6 / 122.2 / 127.3 / 120.1 | `sync` 7.4/8.0 → 5.0/4.8 s; kept |

Cone scaling wall stays ~15 s per 4 scalings: each rank's scaling CPU is
~70 s (2.2 s on 32 threads), but single 88×88 1024-bit cones take 2.4–3.4 s
(`CONE_COSTS` profile line), so the phase is bounded by one cone's latency.
Further scaling of this phase needs parallelism inside one cone's SVD.

1-node full solves (1024 bits, 32 threads): SDPX Solved/42, solve 2170 s,
process 36:19, load 3.4 s with parallel block parsing (222643).
