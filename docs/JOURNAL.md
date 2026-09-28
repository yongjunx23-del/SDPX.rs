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
