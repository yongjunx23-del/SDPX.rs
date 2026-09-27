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

