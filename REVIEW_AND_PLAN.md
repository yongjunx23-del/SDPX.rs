# SDPX development plan

Updated 2026-10-08. [AGENTS.md](AGENTS.md) defines working rules and numerical
contracts; [architecture](docs/ARCHITECTURE.md) defines modules/backends.
[Journal](docs/JOURNAL.md) holds full timings, job histories, source hashes,
failed attempts and audit evidence. This file keeps priorities, current
state, open work and closed directions.

## First priority: solver performance

**Minimize time and peak memory to reach an audited solution at the requested
precision.** Preserve every numerical contract in AGENTS.md; accuracy failure
cannot count as a performance win.

- Float64 target: audited medium/large within 1.3×/2× MOSEK time.
- MPFR target: beat SDPB with matched precision, tolerances, input and hardware.
- Memory target: match SDPB peak PSS/RSS at matched thread counts; report
  speed/memory tradeoffs rather than combining unmatched runs.
- Solver performance comes first; PMP conversion matters when it limits total
  input-to-solution time or memory. Keep useful specialized elimination;
  general backends must earn their place in complete solves.
- Code size, uniform architecture and new interfaces are secondary. Remove
  useful duplication without replacing measured fast paths to meet a line count.

## Development and verification

[AGENTS.md](AGENTS.md) defines the workflow: one change, build the affected
crate (`--locked --offline`, one feature set), one matching E2E against a
frozen baseline with its original-coordinate audit. Pure refactors preserve
points; algorithm changes pass status and audit. Batch related edits before
one matched release ABBA; keep ≥2% end-to-end gains or clear memory/
correctness benefits. Long runs go to the cluster from frozen sources.
Frontends dispatch Float64 plus MPFR128/256/512/768/1024 by default;
`all-precisions` adds 128–2048 in steps of 64 (Lambda43 needs 1216).

## Current state

### Verified solver and architecture

Release 0.9.1 (`4863de1`) keeps the reviewed numerical repairs: per-RHS
refinement progress, accepted-iterate residual restoration, shared boundary
validation and the corrected mixed-cone MPI exchanges. Ising11 MPFR512 is
`Solved`/52 at one/four threads with identical points and an original-
coordinate 1e-30 audit. Actual two-rank MPI checks cover sampled/ordinary
products, mixed orthant rows, accepted product reuse and gather decoding.
Default Float64 failures remain labeled below; qnorm is explicit opt-in.

Owner setup has one constructor/layout route; local/global layouts share
immutable storage, fingerprinting stays opt-in, and reductions use existing
buffers. History training/reuse passes the MPFR256 equality/orthant E2E
locally and on two MPI ranks with identical audited points.

Latest retained changes (individual measurements and source bindings are in
[the journal](docs/JOURNAL.md)):

| Area | Retained benefit and evidence |
|---|---|
| Condensed and MPI storage | Reuse accepted H·z, decode gathers in place, and omit owner/fused-only staging. Logical Lambda27/1024 reductions include 149.6 MiB of H·z vectors and 74.8 MiB of gather staging; local and actual two-rank Ising11 points/audits match. No aggregate Lambda27 RSS claim. |
| PSD and sampled storage | Borrow dead SVD/eigen work, omit unused Gram/coordinate tables, pack MPFR Grams/results and reduce rotation records. Distinct scalar inverse adjoints write quadratics directly; compacted bases retain projection. Matching original-coordinate and threading audits pass; individual timing/memory tradeoffs remain in the journal. |
| Shared exact-product CRT | GEMM, congruence, quadratic and bilinear kernels share one accumulator and existing prime scheduling; contiguous outputs omit index vectors. Matched small release cases reduce RSS by 1.33–24.83%, with native changes from −1.49% to +4.96%. These are separate measurements; no summed gain or full Ising/Lambda27 claim. |
| Float64 Schur storage | Defer the unused serial tile buffer during parallel assembly. Medium release RSS 359.75 → 350.90 MiB (−2.46%), native +1.31%; full audited points match. |
| Bound-leaf elimination | Omit unused dense couplings, masks/indices and pivot signs. Float64/MPFR products reuse the Schur destination, removing `border² × sizeof(T)` retained bytes; matching gravity full fields/audits pass. Float64 reuse release native is flat (−0.08%), no RSS claim. Forward/backward phases use disjoint 1024-leaf ranges on the existing pool. Larger gravity, PBS223608/223609, four threads / BLAS one: native −5.88%, reverse-order repeat −3.41%; single-thread +1.91%/+0.99%, no RSS claim. Full fields and audits match. |
| Local SOC exact products | Bounded row groups retain rounded Z and round the complete sum once. Initial CSDR3/256 release: serial native −20.97%, RSS +11.44%; four-thread speed below gate, RSS +43.45%. PBS223610/512/four threads: native +0.98%, RSS +22.81%. Direct encoding removes two MPFR staging panels per worker. Subsequent compact row supports: matched256 release native −2.59%/−6.42% at one/four threads; PBS223611/512/four threads native −4.25%. Full fields/audits match; no meaningful row-support RSS gain. |
| Local SOC storage | Keep raw coupling row 0 in its unchanged Y row; B stores only raw row 1. Remove 176,400 scalars / 8.08 MiB on CSDR3/256, with identical full fields and original audit. Updates, finite checks, retries and fallback preserve original raw values. Logical memory benefit only. |
| Grouped residue block Schur | Rows sorted by exponent window; 256-row groups with own window and primes, exact group integers summed before one rounding. csdr3/256 release ABBA −17.1% (four threads, RSS 238 → 214 MiB) and −6.7% (one thread); identical points. |
| Local cone arrow (MPFR) | Any mix of small orthant/SOC leaves around the equality rows (oversized cones join the border), compressed per-leaf couplings, border ordered by coupling degree, exact residue border Schur from 128 bits with Z formed while encoding. Crossing SOCP MPFR128 698.8 → 19.3 s, Solved/63, objective within 8e-24, RSS 953 → 728 MiB. Parity 49/49 identical. |
| Size gates and presolve | Local/shared SOC border caps replaced by memory and density rules (300-row synthetic: MPFR256 280 → 7.6 s); presolve rank proof by direct F_p images, stride-ordered columns, m ≤ 1024 (crossing 3.9 s → 10 ms). |
| PMP conversion | Release parity bases before constraint coefficient work. At 256 samples/MPFR1024, remove 2.27 MiB of overlapping scalar storage per worker plus row headers. Tiny MPFR128 output is byte-identical; matched current-solver points and the original audit pass. Logical memory benefit only. |

Sparse row-plan setup also reuses offsets as fill cursors, removing
`rows × sizeof(usize)` temporary bytes per construction at every precision.
Gravity256 full fields and original audit match; no speed or peak-RSS claim.
Diagonal congruence releases its unused scaling residues after row products
join, allowing CRT to reuse or free the storage. Gravity256 full fields and
audit match; logical overlap removal, with no formal speed or RSS claim.
CRT table construction keeps one large-integer cofactor at a time instead
of all prime quotients; Ising11/1024 full fields and original audit match.
This reduces temporary construction storage; cached table sizes are unchanged.

Ordinary square GEMMs of side ≥12 use the existing exact residue kernel at
≥1024 bits. Matched Mac Ising11/1024 release ABBA, four threads / BLAS one:
native 9.836518 → 9.555854 s (−2.85%), peak process RSS +4.69%; full fields
and original audits match. Lower widths and other kernel eligibility stay
unchanged. This is a small-case speed/memory tradeoff, with no larger-case
or combined-patch claim. The Ising audit now checks the requested precision
with the same 1e-30 equations and tolerance, removing its 512/768 allowlist.

The shared-operand SOC layout reduces RSS relative to private row groups but
regresses native time 10.96% in release and is reverted. PBS223610 completed
the private layout's frozen MPFR512 comparison: no verified parallel gain.
Whole-leaf support grouping offers only 0.0573% fewer products on CSDR3.
Compact common row supports are kept after matched release gains;
PBS223611 confirms their higher-precision gain from frozen sources.

The earlier combined accepted-product / borrowed-work / gather-decode release
check on small Ising11/512 was native −1.80%, RSS −0.10%; it establishes no
≥2% speed gain or aggregate benefit for later patches. Sampled RHS borrowing
is reverted after a warmed repeat showed no benefit and possible extra
retained TLS workspaces. Scalar power-of-two multiplication
is already in release 0.9.1; removing its exact mantissa-copy shortcut regressed
release Ising11/512 by 0.89%. Leave it in place.

### Matched release check (2026-10-07)

Release A/B/B/A against 0.9.1 on the Mac (journal): ce09de1 and e2470e6 leave
medium and ising11 bitwise unchanged (±1%); since 10-03 medium is −6.6% time
and −34% RSS, ising11 −2.8% and −7%. csdr3/256: ce09de1's residue local Schur
gives −13.1% at four threads and −26% at one with identical points, at +36–40%
peak RSS at four threads (+6% at one). The correctors then shorten csdr3 at
the pinned 1e-8 by 12% only by stopping earlier (objective error 2.9e-2 versus
1.6e-3). At a matched 1e-12 they save 23% of the iterations (103 → 79) but no
time (wall +3.1%, CPU −2.3%): corrector solves and the serial SOC corrections
take 26% of the solve. Owner-MPI runs never apply correctors.

### Full-solve accuracy gate

PBS223602 checked frozen 0.9.1 sources differing only in arrow residue batching,
release ABBA, Lambda27/1024, 32 cores / BLAS one, 64 GiB, six hours. Local
memory changes are excluded. The branch preserves exact products and leaf
subtraction order. Earlier three-iteration windows improved 6.40%; they do
not establish complete-solve performance.

Preserve both prior failures: PBS223591 reused the wrong candidate binary
and failed the small branch-count preflight; PBS223593 completed baseline
`Solved`/42 (2441.387 s native, 21163.313 MiB peak RSS) but its Julia audit
exhausted memory before any candidate ran. PBS223602 is the second scoped
retry. The blockwise audit completed in 598.277 s / 1157.227 MiB, then failed
the unchanged 1e-30 gates: primal 3.730e-28, dual 1.475e-23, PSD mapping link
2.733e-26 and componentwise dual 0.84355. Exit 1; no candidate ran. Diagnose
the worst original equation and solver/audit normalization before further
timing. Both automatic retries are spent. The repaired audit matches all
Ising11 gates (objective/gap summation differs by 2.24e-154). No audited
large comparison yet. Evidence: `$SDPX_E2E_HOME/work/resumed-091-arrow-full-20261006/`,
remote `retry2/`.

The user approved a separate saved-point diagnosis after those two retries.
PBS223605 completed on eight cores / 16 GiB, one actual Julia thread and
no new solver run: 331.86 s process, 1138.254 MiB peak RSS, exit 0. Layout
and failed dual metrics match exactly. Primal variables reach 5.832e118;
standard residual normalization includes that large norm. The worst relative
row (block77/sample127, zero cost) has absolute residual 1.286e-61, term
work 1.525e-61 and relative error 0.84355; the worst absolute row is a
different sample (block78/sample0). Both optional qnorm/componentwise gates
were off. Next accuracy pilot should enable the existing gates explicitly;
defaults and convergence rules stay unchanged. The saved point remains FAIL.
The accuracy pilot is prepared and uploaded (existing frozen baseline binary,
32 cores / 64 GiB / two hours, at most 100 iterations), awaiting additional
approval after the exhausted retries. Small Ising11/512 with both gates at
1e-30 remains `Solved`/52 and passes the unchanged audit.

### Historical performance reference

The following measured sources precede the 2026-10-05 repairs; do not treat
them as current-tree timings or add gains from different comparisons.

- Mixed Lambda27/1024 full solves, 32 threads per node: four-node SDPX
  `Solved`/42, native 1369 s versus SDPB optimal/125, 2592 s; one-node SDPX
  2106 s versus SDPB 4014 s. Sources/allocations are recorded in the journal.
- Matched Float64 gravity, explicit qnorm 1e-6, PBS222437, all audits pass:
  larger SDPX 1.914/1.380 s (one/four threads) versus MOSEK11.2.2
  1.160/0.976 s; small 0.143/0.147 s versus 0.130/0.109 s. Larger SDPX
  takes 35/36 iterations versus MOSEK21: diagnose the gap without changing
  convergence, regularization or refinement rules.
- Packed residue outputs: complete Ising19/768 RSS −4.72%, native unchanged
  (PBS223509 combined binary). Adaptive SVD tiles regressed Ising19/768
  by 3.47% on 32 cores (PBS223513) and were removed; four-row replay stays.

Full experiment history, source bindings, old timing tables and unretrieved
receipts (including PBS222447) remain in the journal. They do not authorize
restarting old campaigns.

## Profile

Shares of the 2026-10-07 release runs of e2470e6 (receipt phases):

- `medium` Float64, one thread: KKT update 83% of the solve, of which Schur
  assembly 66% (coefficient transform 37%, dot/scatter 24%) and dense
  refactor 12%. Four threads give only 1.43×: the refactor stays 0.25 s and
  the Schur assembly speeds up 1.67× (next work 12).
- `ising11` MPFR512, one thread: cone scaling 32% (SVD 28%: bidiagonal QR
  12%, rotation replay 11%), KKT update 33%, KKT solve 16%, refinement
  residuals 14%, RHS recover/prepare scaling 10%/7%. GMP limb multiplication
  inside exact dots was ~45% of samples on 10-04.
- `csdr3` MPFR256, four threads, 1e-8: KKT update 45% (refactor 26%, residue
  local Schur 22.5%), step length with correctors 31% (corrector solves 18%),
  iterative refinement 27%. At 1e-12 corrector work is 26% of the solve.

Cluster 222426 cycle samples: gravity exact residual 22.5%, DGEMM 13.8%,
diagonal congruence 11.0%; Lambda GMP multiplication 26.8%, DGEMM 15.9%,
CRT finish 4.4%. Profile before a factorizer rewrite, SIMD, communication
overlap or buffer pool.

Larger gravity keeps 20,202 variables, 99 equalities, 20,200 bounds and a
border of 101; lower Gram triangle/common 16-column tiles help four-thread
EPYC but regress serial/eight-thread runs by 19.9%/8.1%. The historical
Float64 iteration gap (opt-in qnorm at 37 iterations versus MOSEK 21) is
explained 99.986% by static regularization in a shift probe; stopping,
regularization and refinement stay unchanged.

## Known failures (keep visible)

- **Mixed Lambda27 MPFR1024:** frozen 0.9.1 baseline is `Solved`/42 but fails
  the original-coordinate 1e-30 gate (dual 1.475e-23, primal 3.730e-28,
  PSD link 2.733e-26, componentwise dual 0.84355). Diagnose before timing;
  convergence and audit rules remain unchanged.
- **Gravity Float64 default:** `Solved`/17, original dual residual 4.01e-6
  exceeds 2e-6. MPFR128/256 pass; opt-in qnorm runs are reported separately.
- **Medium Float64 default:** 1.916e-6 exceeds 1.75e-6. Opt-in
  `tol_dual_qnorm=1e-6` gives `Solved`/19, r_d 6.14e-7, gap 2.41e-7, audit pass.
  Audit uses `‖r_d‖∞ ≤ tol·(1+‖q‖∞)`; standard solver normalization also uses
  x/z magnitudes. Evidence supports the mismatch; defaults remain failures.
- **SDP_control3:** 29 iterations, primal residual ≈4.98e-4; deferred.
- **MPFR `condensed_graded`:** kept as designed. Applying H through R fixes
  the synthetic test but costs +5.5% on Ising; rounded G loses grading below
  eps·‖G‖, outside observed MPFR IPM states. Known synthetic failure remains.
- **Normalized 2×2 PMP:** `AlmostSolved`/24 at MPFR512/1e-42. A tiny dual pivot
  is replaced by dynamic regularization and refinement diverges; dynamic
  regularization off or 768 bits solves it. The proposed no-replacement
  refactor requires approval because it changes the regularization contract.
- **Float64 large:** historical129 s/26 versus MOSEK25.4 s is superseded by
  the scoped222437 audited refresh above; it is not a matched optimization
  comparison with that history. Mixed Λ27 has the matched SDPB comparison above.
- **Large SU(2) path Float64 (n 7054, 2026-10-08):** `AlmostSolved` after
  24–25 iterations at 1–32 threads, `NumericalError`/23 at 64; the dense block
  factor falls back to sparse LDL near convergence. MOSEK reaches optimal in
  18–30 iterations (plan item 10).
- **Λ35 spins 0–70 at 768 bits:** `MaxIterations` 1000, μ stuck at 3e-24 by
  the default static and dynamic shifts (plan item 1). SDPB also stalls at
  768 bits (gap ~1e-15 after iteration 225); compare the two at 1024 bits.

## Concrete next work

### Performance plan (2026-10-08)

Measured on idle 64-core allocations of the UCAS EPYC nodes; evidence and
the full tables are in the journal ("diagnosis: Λ35 768-bit stall…" and
"iteration count: start scale…"). MPFR at 768 bits, 1e-42; Float64 at 1e-6.

| Case | SDPX b0088bb | Reference |
|---|---|---|
| Λ27 spins 0–50, one node | 451 it, 943 s (2.08 s/it); τ₀ 1e-30: 181 it, 385 s | SDPB 265 it, 375 s (1.41 s/it) |
| Λ35 spins 0–70, one node | MaxIterations 1000 (3.19 s/it), μ stuck at 3e-24 | SDPB 746 it, 2004 s (2.68 s/it), after a gap plateau of 1e-13 to 1e-17 over iterations 200–650 |
| spins 0–50, one node | 177 it, 252 s | SDPB 265 it, 285 s |
| medium Float64, 1 / 8 / 64 threads | 5.57 / 3.03 / 3.97 s, 19 it | MOSEK 5.24 / 2.60 (16 threads) / 3.20 s, 16 it |
| large Float64, 1 / 16 / 64 threads | AlmostSolved: 132 / 66 / 70 s (NumericalError at 64) | MOSEK optimal: 82 / 15.2 / 26 s |

The SDPB gap is Λ27: 1.7x the iterations (the τ-chase) and 1.5x the time
per iteration. Refined solves are 55% of a Λ27 iteration and 47% of a Λ35
one: SDPX solves three right-hand sides (the homogeneous constant one
included), each with two refinement levels and about one outer correction;
SDPB solves twice without refinement. Factorization is 13–18%, cone scaling
(one MPFR SVD per cone, the largest cone sets the wall) 15–16%. At 768 bits
Λ35 defeats both solvers. Float64 loses at many threads to serial or
collapsing kernels, and on large to a dense-factor fallback near convergence.

Work in this order. Each item is one change with its own gate (AGENTS.md);
algorithm changes pass full solves and the original-coordinate audit.

**Status 2026-10-08 (uncommitted, journal "performance plan items…"):**
done 1, 8, 9, 10, 12; item 2 shipped off by default (Λ19 regression); item 4
reverted (corrections are at the outer rounding floor); item 5 was already in
place (largest-first cone jobs, row-split replay); item 3 not done (Λ27
escapes a 250-iteration gap plateau, so a floor stop would end good solves).
Open: 3 (needs a discriminating signal), 6, 7, 11, 13, 14; Float64 large
still AlmostSolved (cause not the dense fallback); MOSEK still 3x faster on
large at 16 threads.

**Goals (user, 2026-10-08):** beat MOSEK on every Float64 problem; beat
SDPB at 768+ bits on the Ising problems (Λ19, Λ27, Λ35, ising11). Current
gaps (2026-10-09):
- Λ27: τ₀ = 1e-40 gives 156 it / 276 s and the default 183 it / 312–368 s,
  against SDPB's 265 / 375 s. SDPX is still slower per iteration (1.77 vs
  1.41 s).
- Λ35: SDPX does not reach 1e-42 at 768 bits. The primal diverges and the
  dual has no Slater point, with no exact face. SDPX and SDPB both escape a
  plateau at about iteration 650. SDPX then stalls at about 1e-26 objective
  accuracy (1200 it, 3428 s), while SDPB finishes in 746 it / 2009 s. At
  1024 bits SDPX solves Λ35 in 743 it / 3565 s.
- Float64 suite (58 cases, every point audited): SDPX loses 46 at 1 thread
  and 50 at 16. Geo-mean SDPX/MOSEK is 5.4× (1.7× where MOSEK takes
  ≥ 0.1 s). Four cases report Solved but fail the audit.
- Float64 large: SDPX 41–47 s, AlmostSolved or NumericalError; MOSEK 15–29 s,
  and its points fail the audit. Medium is at parity: 3.17 s versus
  CVXPY→MOSEK's 2.96 s.

Active branches (2026-10-09; two agents, at the user's request):
- `perf-audit`, Float64 suite:
  - Task 9: Solved must imply an audit pass (control1, hinf3, sched_100_*;
    s ≠ b − Ax; thread-dependent outcomes).
  - Task 10: cost-based LP backend, presolve and SDP formulation choice,
    independent of thread count.
- `perf-l35`, Ising, task 12: a fixed-τ phase after the HSD start, which
  drops the constant right-hand side (one refined solve in three); the Λ35
  end game after its plateau escape; a progress stop that lets the plateau
  be traversed. Task 11 closed facial reduction for Λ35 (no exact face) and
  fixed τ from the start (journal 2026-10-09).

Parked, with WIP committed on each branch (journal 2026-10-09):
- `perf-sharedrhs` `d34869c`: Λ27 −2.4%, unaudited.
- `perf-fulldir` `6af5724`: regression.
- `perf-tau0` `0fccf2f`: ties the default.
- `perf-f64large` `50325c4`, `perf-threads` `946d4be`, `perf-facial`
  `a3a7271`: unverified.
- `perf-border` `a80cfa1`: gate not run.

**Survey leads (2026-10-08; data in `~/.cache/sdpx-e2e/work/su2-large-scaling-20261008/`):**

- **Float64 SU(2) medium and large have no Slater point.**
  - Every PSD block has a kernel shared by all A_j and b. Large: 30/30/30/30/60/12/14 of 95/92/94/92/186/74/71. The relations are sparse, with coefficients 1, −1, −4/3.
  - MOSEK's "optimal" points fail the 1e-6 audit (r_d≈1e-4).
  - Fix: facial reduction in presolve (`perf-facial`, opt-in). The detection tolerance is a contract decision.
  - Presolve already drops large's 1317 dependent equalities. Columns that are empty (74) or appear only in equalities (609) are still not eliminated.
- **Float64 factor ordering.** Large's Schur columns split into blocks 1–5 only (3729), blocks 6–7 only (1817) and both (825). Two leaves plus a border would cut factor flops about 5×; refactor was 65% of one-thread time. Next: tiled Cholesky lookahead, phase lanes capped by work, position-blocked Schur dots.
- **MPFR, in order:**
  1. τ₀ from the data (SDPB uses Ω = 1e20, and 1e60 for Λ43).
  2. Refining the full direction including Δτ, with batched corrections. This changes the refinement scope, so it needs approval.
  3. A dataflow pipeline per component, with idle workers parked.
  4. BLAS-3 rotation replay in the MPFR SVD (replay is 63% of SVD time).
  5. A distributed border factor.
- **Closed:**
  - HKM or mixed NT/HKM, and NT scaling through eig(LᵀSL).
  - MPFR PSD Gondzio correctors (about +33% per iteration).
  - Low-rank DSDP formulas (already covered).
  - Strassen or Ozaki residue GEMM (≤3%).
  - A backward-error refinement stop (changes the refinement rule).

**Approved by the user (2026-10-09):**
- facial reduction with a tolerance-based kernel detection, gated by the
  original-coordinate audit;
- refinement of the full HSD direction (including Δτ), with tolerances
  unchanged;
- a data-chosen τ₀ that may cost ising11 iterations, if every Ising case
  still beats SDPB.

The user also delegated further contract decisions within the fixed precision
rules (AGENTS.md "Numerical contracts").

**A. Convergence of large sampled problems**

1. **MPFR regularization scale (needs approval: regularization contract).**
   The default static shift eps^(3/4) (6.8e-174 at 768 bits) stops Λ35 at μ
   3e-24, and with it lowered to 1e-215 the dynamic rule (pivots below
   eps^(3/4) set to sqrt(eps) = 3.6e-116) stops it again at 1.5e-28 (from
   τ₀ 1e-30 too: μ flat, gap ~3e-15); with both relaxed, 768 bits tracks the
   1024-bit run. 40 refinement steps per solve do not help. Candidates:
   smaller constants (eps^(7/8)…eps) relying on the existing escalation when
   a factor fails, or shifts proportional to the reduced matrix diagonal,
   with a replacement δ far below sqrt(eps). Pending evidence: 768 bits with
   both shifts relaxed and τ₀ 1e-30, 1024 bits with τ₀ 1e-30, SDPB at 1024
   bits. Gate: Λ35
   Solved and audited; ising11, spins 0–50, Λ19, Λ27, csdr3, gravity256,
   mpfr-dev and the normalized 2x2 PMP failure no worse; Float64 defaults
   unchanged (`settings.rs` `linear_default`; shifts in
   `kkt/direct/solver.rs`, pivots in `kkt/ldl/arrow.rs`).
2. **Automatic start scale.** τ₀ 1e-30 gives Λ27 181 iterations and 1e-10 to
   1e-12 gives spins 0–50 132, but ising11 wants 1 (77 at 1e-20). Step 1:
   verbose τ and ‖x‖∞ columns; traces of ising11, spins 0–50, Λ19, Λ27, Λ35
   and the Float64 SDP set. Step 2: one rule for every input — restart once,
   on the τ-chase signature (τ falling several decades while the gap stalls),
   from a τ₀ extrapolated from the trace, inside the same iteration budget
   (the restart path in `core/solver.rs`). Gate: no case more than 5% more
   iterations; Λ27 at most 250.
3. **Stop at a precision floor.** When μ has not fallen for k iterations and
   refinement no longer contracts, stop with a status that names the floor
   (no precision switch) instead of running to MaxIterations (Λ35: 3200 s).

**B. Per-iteration cost of the sampled MPFR path**

4. **Outer refinement corrections** (Λ35 about 0.75 of 3.19 s, Λ27 about
   0.6 of 2.08 s; one correction = prepare + reduced solve + recover + exact
   full residual). Step 1: receipt counters per right-hand side for the
   outer residual before and after the correction, and the inner residual and
   steps. Step 2: aim the reduced refinement (`kkt/direct/solver.rs::refine`)
   at the full-system tolerance divided by the measured amplification, so the
   first recovered solution passes the unchanged outer test (an inner step
   costs about an eighth of an outer correction). Expected up to −25% per
   iteration if the corrections are inner-accuracy driven; if they sit at the
   outer rounding floor instead, report that and stop (no rule change).
5. **Cone scaling latency** (0.31–0.52 s/it). Per-cone times first, then
   largest-first order on the cone pool and row-split SVD replay (each
   rotation acts on rows of V independently, so bitwise identical) only when
   threads exceed cones; adaptive SVD tiles lost 3.5% when cones outnumbered
   threads.
6. **Thread runtime.** About 11% of Λ35's 64-thread cycles are rayon idle
   stealing, and busy-waiting lowers the clock (ising11 3.1 → 2.1 GHz).
   Count parallel regions per iteration, merge short ones, size lanes to the
   work (shared with item 11).
7. **Per-component parallelism.** Leaf sweeps, single-column solves and
   the border factor of one component run serially (a Λ27 rank with two
   components runs at ~5x CPU/wall on 16 threads). Split them with an
   order-independent exact accumulation (changes bits; audit gate).

**C. Float64 SDP on many threads**

8. **Parallel Schur assembly at every size.** Large's assembly is serial at
   every thread count (33 s) because `parallel_assembly_allowed` caps the
   per-block buffers at 256 MiB. Replace the cap with cone-order waves:
   blocks whose buffers fit a budget derived from available memory run in
   parallel and publish in cone order (bitwise identical); a block larger
   than the budget runs alone, its parallel tiles adding straight into the
   Schur matrix (disjoint positions, same per-entry order). Expected large at
   16 threads 66 → ~35 s.
9. **Dense factor kernels.** `tiled_potrf` issues concurrent single-thread
   OpenBLAS calls (79% of large's 64-thread samples, packing 37%), has no
   lookahead, and packs/unpacks, factors the border (`dsyrk`, `dpotrf`) and
   scans for finiteness serially. Run tile GEMM/SYRK/TRSM on faer (as the
   residue products), add one-panel lookahead, parallelize the border SYRK
   and scans; compare with faer's parallel dense Cholesky. Gate: medium and
   large audits; medium refactor at 8 threads 0.54 → ≤0.25 s, no loss at 64.
10. **No sparse fallback near convergence.** `dense_block` rejects a factor
    with any pivot below the dynamic threshold (1e-13) and refactors with
    faer's sparse LDL; large's last factorizations take that path and the
    solve ends AlmostSolved/NumericalError where MOSEK converges. Apply the same
    dynamic pivot rule (eps, δ, sign) inside the dense diagonal tiles so the
    dense path completes, then compare large's last iterations with MOSEK.
11. **Many threads on small problems** (medium 3.03 s at 8 threads, 3.97 s
    at 64). Measure per-lane busy time, then cap each phase's lanes by its
    work.
12. **Iterations** (medium 19 versus MOSEK 16 at equal time per iteration).
    PSD Gondzio correctors (orthant/SOC/τκ have them): one W-scaled
    eigendecomposition per corrector is small next to Float64 assembly.
    Compare at matched final accuracy.

**D. Multi-node**

13. Distribute the arrow border factor (Λ35 border n 4071 replicated on every
    rank, about 143 single-RHS solves per iteration).
14. Batch the refinement agreement flags (2120 of 6113 collectives in 30
    spins 0–50 iterations at four nodes).

Compare Λ35 at 1024 bits (SDPX and SDPB); Λ27 stays at 768.

### Other open items

| Order | Action | Acceptance / constraint |
|---|---|---|
| 0 | Resolve Lambda27's original-coordinate accuracy gate | Saved-point diagnosis is complete; run the prepared explicit-gate pilot after approval. Full primal/dual/PSD/mapping audit still decides acceptance. |
| 1 | Finish exact arrow batching measurement | After the accuracy gate, complete audited release ABBA. Preserve rounded Z and leaf subtraction order; diagnostic windows are insufficient. |
| 2 | Optimize dominant sampled/PSD phases | Use the latest receipts to locate remaining serial work after shared congruence CRT. Largest-cone SVD latency and exact factor products remain; rejected SVD tile/QR directions stay closed. |
| 3 | Remove unused storage/setup work | Packed results/Grams, shared product CRT, implicit indices and sampled owner/fused-only buffers are kept. Further removal needs a live unused allocation; Gram and scaling owners may differ. Preserve shifted factors versus unshifted residual operators. |
| 4 | Improve LP/SOC thread balance and Float64 iteration gap | Compact SOC row supports improve256/512 native time; duplicate raw row 0 is removed. Further storage changes need a live allocation. Use gravity/MOSEK receipts for the iteration gap; preserve convergence, regularization and refinement. |
| 5 | Scale only for the active workload | Frozen sources and bounded resources. Lambda43/1216 and broad core sweeps remain later work; avoid node70. |
| 6 | Improve PMP when it limits the workflow | Secondary to the solver; measure whole-command time/memory and output equivalence. |
| 7 | Free-multiplier g0 SOCP per-iteration cost (3.15 s/15 it vs MOSEK 1.3 s/13 it) | Batch refinement over the constant/affine RHS (one residual pass for both), fewer full-KKT residual passes, then compare a dual (16-row Schur) form as MOSEK's presolve does. Keep the 46-problem Float64 set and csdr3 unchanged. |
| 8 | Lessons from other solvers (2026-10-07 survey, below) | Each item needs its own audited A/B; none is adopted yet. |
| 9 | Centrality (Float64 LP/SOC iteration gap) | Done 2026-10-07 (approved): Gondzio correctors on orthant, SOC and τκ; parity iterations −14.5%, gravity 37 → 22, sched all Solved. csdr3 at matched 1e-12: 103 → 79 iterations at equal time; the 1e-8 gain is an earlier stop. PSD rows have no correction (eigendecomposition per corrector); measure before adding. |
| 10 | Corrector overhead | Parallel corrections kept 2026-10-07 (bitwise identical; csdr3 and csdr3-tight −2.7%); csdr3 accepts 38/57 and 70/112 corrector solves. Remaining at 1e-12: 1.9 s of corrector solves and 0.38 s of extra step lengths. Same-session csdr3-tight against ce09de1 (no correctors): −2.1% wall, −1.8% CPU at equal objective error, so correctors now gain slightly at matched accuracy. |
| 11 | csdr3 residue memory | Grouped windows/primes kept 2026-10-07 (−17.1%/−6.7% time, RSS 238 → 214 MiB at four threads). Remaining above 0.9.1: ~25 MiB at four threads (per-way i128 slots and chunk buffers). Entry-compact chunks and per-column splits are superseded. |
| 14 | Local cone arrow follow-ups (crossing SOCP) | Done 2026-10-07: degree-ordered border (no fallback), Z formed while encoding, presolve images; 19.3 s, 728 MiB. Open: per-way residue scratch (~60 MB per worker at a 300-wide border; a capped prime group cost 8% time), exponential and power cones as leaves (no MPFR test problem yet). Binary64 stays augmented (AlmostSolved/47 at 1.4e-6 versus faer 7.9e-9). |
| 15 | Sparse QDLDL path (problems with no arrow layout) | Coalescing subtrees past the 500k-cell plan cap is bitwise identical but 24% slower on a trunk-dominated KKT (ordered trunk replay); a plan needs leaf-dominated work to pay. `solve_many` still loops single solves. Measure on a workload that keeps QDLDL. |
| 12 | Medium four-thread scaling | Superseded by plan items 8–11 (cluster runs at 1–64 threads against MOSEK, 2026-10-08). |
| 16 | Multi-node sampled Ising (owner MPI path) | History 2026-10-07/08 in the journal: component-matched rank layouts, residue GEMMs on faer, parallel owner border products, start scale and opt-in GMRES-IR. Full solves one node 32/64/96 threads 300/247/263 s vs SDPB 536/285/370 s on spins 0–50; Λ27 30 it 1/2/4 nodes 58.0/56.3/53.3 s vs SDPB 40/34/42 s. Open beyond plan items 2, 7, 13 and 14: the single-process multi-owner path (26 in-process owners 74.8 vs plain 44.2 s) and component-size spread. Gate: spins 0–50 30 it on idle nodes, bitwise against the same rank count; algorithm changes by full solves + audit. Known: `sampled_integration::dim2_signed_parities_ruiz_f64` returns AlmostSolved on the cluster (Linux) at ae2a827 and after; passes on macOS. |
| 13 | Owner-MPI correctors | `OwnedCones`/`OwnedVariables` keep the no-op defaults, so SOC/LP iterates differ from single-process runs. The matched-accuracy gain is small (item 10) and no MPI SOC/LP workload is active; port when one is, with a real MPI E2E. |

## Measurement prerequisites

Apply these to affected paths; they are not a project-wide gate:

- Use validated dynamic OpenBLAS on the cluster. The static-provider issue
  needs a matched runtime check before provider changes; build success is
  not timing equivalence. Keep thread/BLAS budgets identical.
- Freeze an audited Float64 baseline with explicit opt-in qnorm settings.
  Preserve default failures; do not relax external gates or stopping rules.
- Checkpoint structure must match. Values/equilibration distinguish continuation
  and hot start; write accepted iterates and map hot starts through original
  coordinates. Historical tests do not validate later sources automatically.
- Preserve original input/settings/precision/hash binding before reusing an
  accepted audit after actual full-point equality. Record native/API/process
  scope, status, iterations and peak memory with evidence.
- Changes that move the stopping point (directions, correctors,
  regularization) compare time to a matched final accuracy, e.g. csdr3 at
  1e-12 (optimum −31.6721556) beside the pinned 1e-8 case. Fewer iterations or
  a faster 1e-8 stop alone are not a speedup.
- On the shared Mac, compare only inside one A/B/B/A session and report CPU
  seconds: identical binaries varied 28% and session medians drifted 15% on
  2026-10-07. Decide sub-5% changes on a quiet host or the cluster.

## Decisions still needed / deferred

- **Approved 2026-10-07 (regularization contract):** shared-SOC arrow leaf
  primal columns skip the static shift (pivot `P_jj + aᵀH⁻¹a > 0` by the fixed
  elimination order; any factor failure restores it). Evidence: JOURNAL
  2026-10-07. Other structures keep the shift until measured and approved.
- **Independence from Clarabel.rs (requested 2026-10-07):** ≈36% of non-test
  lines still match a same-named Clarabel.rs file (verbatim: qdldl, chordal,
  cone files, info_print, data_updating, datamaps). Apache-2.0 requires keeping
  the notices of derived files. Staged plan, each stage behaviour-neutral on
  the pinned cases: (1) SDPX-owned problem/presolve/postsolve pipeline and
  output (banner, settings schema); (2) replace the trait-object
  `ProblemData/Variables/Residuals/KKTSystem` scaffolding with SDPX's own
  state types; (3) own cone and KKT interfaces around the arrow/condensed
  backends; (4) retire unused Clarabel features. Conflicts with the standing
  "Clarabel-style convergence/refinement/regularization" rule; needs a
  decision on which numerical policies may change. Done (bitwise-neutral,
  2026-10-07): SDPX staged driver, banner/report, NOTICE/metadata, upstream
  notes. Next neutral candidates: preprocessing pipeline in `problemdata.rs`
  (keep its copy avoidance), a uniform presolve reduction/postsolve record,
  and the configuration printer.

### Survey: what other solvers do better (2026-10-07)

- **SDPB (high precision):** HRVW/KSH/M (XZ) direction, so each block needs
  only Cholesky factors of X and Y; SDPX's NT scaling needs an MPFR SVD per
  PSD cone (≈3 s per 88×88 cone at 1024 bits, the largest serial latency).
  Measured 2026-10-07 (JOURNAL): HKM cone update 6.5× cheaper, but +12%
  iterations on ising11 MPFR512; net ≤ 0 on Λ27 shares. Closed. SDPB 3 forms the
  Schur complement with blocked RNS (FLINT) BLAS, as SDPX's residue GEMM, and
  distributes it with Elemental Cholesky.
- **SDPA-GMP/QD/DD:** double-double and quad-double arithmetic (≈106/212
  bits) is several times faster than MPFR at 128/256 bits; a `DoubleDouble`
  scalar would serve medium-precision solves.
- **MOSEK/HiGHS:** presolve (singleton columns, dualization) decides small and
  separable SOCPs; Gondzio multiple centrality correctors cut iterations.
- **Hypatia.jl:** neighborhood-based step with combined directions and
  interpolant-basis polynomial (WSOS) cones that avoid lifting PMPs to SDP.
- **COSMO.jl / SCS / CVXOPT:** clique merging (SDPX has it), indirect CG
  solves with warm starts for very large KKTs, structured KKT exploitation.

- Owner cost histories remain an opt-in public interface until a real scaling
  campaign establishes their value. Constructor duplication and repeated
  fingerprint plumbing are removed; distributed-arrow scaling above does
  not establish a benefit from cost histories.
- Ordinary per-site and owner MPI paths converge only after real MPI E2Es
  (local OpenMPI: conda env `sdpx-mpi`). Owner setup/reductions now pass a
  two-rank E2E. Keep `direct_kkt_solver` in the public settings schema,
  validated at the boundary; its redundant internal assertion is removed.
- Python/Julia bindings, certificate product, mandatory backend unification,
  communication abstraction, blanket panic removal and line-count rewrites.
  Julia remains input generation/audit only; current API/CLI/C ABI stay supported.
- Sequential Schur writes require column reordering/rounding changes and undo
  suffix compaction. Dense-leaf packing saves only ~5.695 MiB (0.91% RSS)
  against broad index/parallel changes; revisit for a larger workload need.
- Reusing caller Vt for internal SVD V removes no retained PSD matrix. Borrowed
  eigen work limits order88 savings to 350 scalars, zero after indexed fallback;
  no implementation is justified by this storage benefit.
- Distributed restart, broad sweeps and new physics searches await concrete
  need. Old g0/application campaigns stay stopped unless explicitly resumed.

## Closed directions

Do not reopen without new evidence; numerical reasons, timings and sources
are in [the journal](docs/JOURNAL.md).

- Lower/mixed precision, relaxed refinement, NaN clamping and MᵀM eigenanalysis
  violate numerical contracts. Fixed-point SVD replay erased tiny MPFR values.
- HKM (SDPB XZ) PSD direction: 6.5× cheaper cone update, +12% iterations at
  MPFR512 (ising11 58 vs 52); net ≤ 0 for the condensed sampled path.
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
  faer and 2D Gram fail across sizes/threads. Column tiling stays; unused
  scratch alternative was never applied.
- RNS rescan bypass adds mutable/cloned-owner API without copy savings.
  Pair20 is exact but unpack 5.48× slower; potential payload saving is not RSS.
  Rounded-Z residue re-encoding costs memory; cached-Y exact Gram stays.
- Additional serial residue-encoding scratch pooling: MPFR1024 equality
  release ABBA gains only 0.53% native time and increases RSS 2.15%; reverted.
- Scalar power-of-two shortcut removal: release Ising11/512 regresses 0.89%,
  with identical audited points and no clear memory benefit; existing code stays.
- Contiguous GEMM scalar fallback: Ising11 release ABBA −4.78% with 20%
  baseline drift, reverse BAAB +14.14%; no repeatable speed or storage
  benefit. Reverted; packed SYRK/Gram storage stays.
- In-place SVD column reflectors: Ising11 release native −0.13%, below
  the speed gate; copy traffic is not retained storage. Reverted.
- Packed upper congruence residues: native −0.90%/−0.82% on order24/88,
  below the speed gate; order88 RSS +6.27%. Reverted; full staging stays.
- Skinny left SVD projection batches: exact points, but fast-profile native
  +1.96%/+5.82% at one/four threads. Reverted without a release campaign.
- Parallel shared-CRT normalization: order88 release native +0.58%, RSS
  +1.08%. Reverted; fraction conversion and mapped compaction stay serial.
- Diagonal CRT digit-column splitting: small gravity256/four fast screen
  native −0.82%, RSS flat, no storage removal; restored without a release
  campaign. Existing serial diagonal reconstruction stays.
- Removing overwritten sampled/PSD initialization: Ising11/512/one-thread
  release native +0.82%, RSS −0.06%, no retained-storage benefit. Reverted.
- Sampled RHS pool propagation: forward alone native −0.67%; adding scalar
  adjoint pooling gives −1.86% and RSS +10.37%. Reverted; neither clears the gate.
- Incremental RNS prime initialization: release 24-row sampled solve gains
  only 1.23%; tiny table storage does not justify added cache lifecycle. Reverted.
- Sampled RHS matrix borrowing: warmed release Ising11 shows no speed/RSS
  benefit, and longer TLS checkouts may retain more worker workspaces; reverted.
- Leading-diagonal slice and bound-coupling trial: below gate/no clear memory
  benefit; reverted. Shared filtered-FMA fallback fix stays.
- OR objective-cost guard and G2 coupling increase iterations/work or regress
  across sizes; AND guard stays. Prime streaming has poor fast-screen scaling
  and short-wide memory counterexamples; reverted without release comparison.
- Zero sampled-PSD RHS and TRMM: audited but below speed gate/no clear memory
  benefit; reverted, other residue storage reductions remain.
- External zero-multiply/zero-addend FMA and diagonal-Gram GEMV: primitive
  regressions, tiny eligibility or sub-gate noisy kernels; no solver claim.
  Power-of-two dot products and capped wide FMMA fail the actual release gate;
  wider fused arithmetic stays native. Exact
  binary64 constructors pass native proofs but have no qualifying workload scope.
- PMP zero-term trimming and Serde `collect_str`: rejected measured variants.
- Ordinary small-square residue GEMM at512: Ising11/four-thread release
  native +2.47%, RSS +2.37%, full fields/audits match; reverted. Global
  24-row/product-area eligibility stays; QR/replay are unaffected.
- One exact residue Gram for the whole arrow border (all leaves and ranks,
  residues summed by allreduce): Λ27 1024-bit 49.6 s versus 45.2 s for
  per-leaf exact dots on one node, 30.1 s versus 21.5 s on two; removed.
- Dropping the condensed inner refinement on g0 (0.034 → 0.030 s): refinement
  levels stay (2026-09-24 decision).
- Streamed Float64 dense Schur spans (RSS −15…28%, time +2.5…5.7% on medium)
  and a precomputed alias index (−0.7%): identical points, not kept.

Cluster: OpenMPI TCP (`--mca btl self,vader,tcp`), `openib` hangs; avoid node70.
