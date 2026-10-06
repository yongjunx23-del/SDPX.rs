# SDPX development plan

Updated 2026-10-06. [AGENTS.md](AGENTS.md) defines working rules and numerical
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

The 2026-10-05 review fixes are verified: Ising11 MPFR512 at one and four
threads is `Solved`/52, passes the original-coordinate 1e-30 audit and
returns identical points. A two-rank MPI KKT check covers serial and pooled
sampled products, mixed orthant rows and an empty arrow border. Focused
checkpoint, recovery, cone-input and C ABI checks pass. Validation is shared
at preparation; every RHS now measures its own refinement progress, and
rollback refreshes accepted-iterate residuals before reduced convergence.
The repaired Ising Lambda19 baseline is now retimed below.

Owner setup now has one route for explicit, automatic and MPI partitions.
Fingerprinting stays opt-in at the JSON boundary; ordinary solves avoid
unused history metadata. Local/global layouts share immutable storage, and
owner reductions reuse existing output buffers. The MPFR256 equality/orthant
QP is `Solved`/19 locally and on two MPI ranks, including history recording
and reuse, with identical baseline points and a 1e-30 original-coordinate
audit. This removes 150 physical Rust lines; no speedup is claimed.

The first exact-BLAS arrow trial is rejected: release ABBA on eight cluster
cores gave Ising19 MPFR768 native medians 332.888 → 349.592 s (+5.02%) and
RSS 725.756 → 855.352 MiB (+17.86%), with identical audited `Solved`/119
points. Mixed Lambda27 MPFR1024 three-iteration windows gave 824.563 →
831.410 s (+0.83%); these are incomplete diagnostics, not audited solutions.
The disconnected-QR trial is also removed: its completed diagnostic A/B
showed only −0.35%, and PBS223495 reached its two-hour limit before ABBA
finished. Preserve its completed rows; the shell exit marker alone does not
establish campaign completion.

Packed upper-triangle residues reduce retained per-prime output storage.
PBS223509 completed its eight-core release ABBA: Ising19 MPFR768 native
332.946 → 333.185 s (+0.07%), process RSS 731.760 → 697.197 MiB (−4.72%),
all audited `Solved`/119 with identical points. Its combined candidate also
contained adaptive SVD tiles, so the RSS result is for that frozen binary.
Lambda27 MPFR1024 baseline/packed three-iteration windows were
822.329 → 769.710 s (−6.40%), RSS unchanged, identical `MaxIterations`/3
points. Batched arrow products still need a complete audited release ABBA;
the windows do not establish full-solve performance.

Approved PBS223513 completed on 32 cores within 45 minutes. Isolating the
SVD tile change gave complete Ising19 233.151 → 241.238 s (+3.47%), RSS
1435.098 → 1452.141 MiB; Lambda27 windows improved only 0.93%. All points
match and complete solves pass the audit. Remove adaptive tiles; keep the
existing four-row parallel replay. Smaller tiles and disconnected QR are
closed directions without new evidence.

The later local arrow admission rules preserve column-parallel dots for
short leaves and underfilled batches, use residues above 256 bits, and
bound result storage relative to Y. Equality-only problems now admit the
shared pool under the existing work cutoff; direct backends receive it
before initial thread reporting. MPFR1024 equality E2Es at one/four **actual**
workers are `Solved`/0, call the new residue branch, preserve full baseline
points and pass the original-coordinate 1e-50 audit. The earlier one-bound
fixture preserves points too, but misses its raw dual gate in the baseline
as well (1.127e-50 > 1e-50); it is not an audit pass. Ising11 MPFR512 remains
`Solved`/52 with identical points and 1e-30 audits at one/four threads.
A focused residue check covers all transposes and the 32-column boundary.
These admission/pool changes are outside the completed cluster binaries.
See the journal for source and evidence bindings.

The complete Lambda27 MPFR1024 baseline/latest ABBA is prepared under
`$SDPX_E2E_HOME/work/exact-arrow-full-20261006/`: frozen source, input hashes,
32 cores, 64 GiB, at most six hours. Its independent original-coordinate
1e-30 audit passes the small Ising11 preflight. It is **not submitted**;
performance work is paused at the user's request. Resuming this longer run
still needs approval beyond the completed 45-minute allocation.

The performance results below describe their frozen sources before the
2026-10-05 correctness fixes; they are not timings of the repaired tree.

All three pinned cases pass on the 2026-10-04 working tree (Mac M4, `fast`
profile, preliminary single-run timings, not performance claims):

| Case | Status | Native | Backend | Audit |
|---|---|---|---|---|
| `medium` Float64, qnorm 1e-6, 1 thread | Solved/19 | 2.12 s | `condensed_dense_block` | PASS, r_d 6.14e-7 |
| `csdr3` MPFR256 SOC, 4 threads | Solved/57 | 7.0 s | `local_soc_arrow` | PASS |
| `ising11` MPFR512 sampled, 1 / 4 threads | Solved/52 | 12.2 / 3.7 s | `condensed_sampled_arrow` | PASS, 1e-30 |

g0 SOC benchmark, medium (`SDPX_g0_benchmark`, Mac M4, one thread, native
seconds; condensed SOC elimination + dense panels, 2026-10-04):

| Precision | Before | Now | Reference | Status / audit |
|---|---|---|---|---|
| Float64 | 0.20 s, 39 it | 0.023–0.024 s, 46 it | MOSEK 0.0286 s, 39 it | `AlmostSolved` (dual 3.3e-7 > 1e-9); objective within 2e-11 of the 256-bit answer (MOSEK 1e-7); audit dual 2.1e-6 (MOSEK 8.7e-6) |
| MPFR256 | 6.65 s, 117 it | 2.55 s, 117 it | — | Solved, audit accepted (dual 2.15e-30) |

The g0 audit's primal residual uses the reported s; SDPX reports its iterate
(1.6e-4 consistency on tiny rows), MOSEK reports b − Ax. Ruiz bounds
1e-4/1e4 (a settings option) cut 256-bit to 65 iterations / ~1.95 s.

Mixed Λ27 at 1024 bits, 32 threads per node (SDPX one rank per node, SDPB
32 ranks per node, thresholds 1e-30), full solves: 4 nodes SDPX Solved/42,
solve 1369 s, process 23 min (head a0a6645; 0.9.0: 1608 s); SDPB
optimal/125, solver 2592 s, process 43 min. 1 node: SDPX 2106 s, SDPB 4014 s.
Per iteration (3-iteration runs) SDPX 51/46/37 s on 1/2/4 nodes vs SDPB
143/70/65 s. Kept since 0.9.0 (same-allocation A/B, 4 nodes): coupled-column
arrow work and aligned single-exchange sampled/scaling products, −6%.

Distributed arrow (mixed Λ27, MPFR1024, 3 iterations, 32 threads per node,
UCAS cluster): per-iteration time ~670 s → ~50 s on one node (leaf
contributions on the fly, column-parallel leaves, cap by KKT size). Two
nodes: refactor 77.4 → 49.8 s, IP 150.5 → 142.7 s; the remaining cost is
replicated cone scaling and m-vector gathers. The four-node run (job 222592,
host list including node70) produced no output and was left to its walltime.

Latest matched cluster comparisons (node192, four threads, BLAS one, release
native medians; each row is its own frozen comparison, gains do not add up):

| Kept change | Native result | Memory / accuracy |
|---|---|---|
| Remove duplicate diagonal reserve, gravity256 | small −3.48%, larger −4.22% | RSS −3.42%/−3.10%; points/audits identical |
| Single-product scratch + balanced residues | small256 −0.92%, larger256 −3.41%, Ising512 +0.66% | dead scratch removed; 12 points/audits identical |
| Regular-first exact-dot scan, gravity256 | small −2.87%, larger +0.41% | points/audits identical |
| Zero/orthant pool, gravity256 | small −5.70%, larger −4.25% | RSS +1.05%/+0.31%; points/audits identical |
| Narrow two-product FMMA, Ising256 | −2.87% | N ≤ 4 only; points/audits identical |
| Narrow scalar FMA, gravity256 | small −2.24%, larger −0.94% | N ≤ 4 only; points/audits identical |
| One-RHS GEMV, gravity53 | small −26%/−24% (1/4 threads), larger −11.8% | audits pass; rounding differs across arms |
| Contiguous residual views, gravity53 | small −2.67%, larger −10.58% (4 threads) | serial small below gate |
| KKT prefix, medium53 | −3.0% | RSS −5.0% |

Float64 versus MOSEK 11.2.2 (PBS222437, explicit qnorm 1e-6, all audits
pass): larger gravity 1.914/1.380 s (1/4 threads) versus 1.160/0.976 s;
small 0.143/0.147 s versus 0.130/0.109 s. Larger qnorm runs take 35/36
iterations versus MOSEK 21. The matched SDPB results above cover mixed Λ27.

Pending: PBS222447 (sampled scratch clearing and scalar power-of-two
multiplication, three cases against kept 222444). Later generic changes have
no acceptance from that earlier MPI offset-reuse gate (222378); the later
MPI changes have their own Λ27 evidence above. Full job histories are in the journal.

## Profile

Shares of the 2026-10-04 local runs (receipt phases, one thread):

- `medium` Float64: KKT update 82% of the solve, of which condensed PSD Schur
  assembly 67% (coefficient transform 37%, dot/scatter 26%) and dense
  refactor 12%.
- `ising11` MPFR512: PSD scaling 31% (SVD 27%: bidiagonal QR 12%, rotation
  replay 11%), KKT solve 16%, RHS scaling 17%, refinement residuals 15%.
  GMP limb multiplication inside exact dots is ~45% of samples overall.
- `csdr3` MPFR256, four threads: refactor 43% of wall (arrow local Schur
  38%), iterative refinement 24%. The local Schur is one exact dot per border
  pair over every leaf (structurally `YᵀZ`); an exact residue product would
  give identical bits, but the analogous bound-leaf trial was rejected for
  peak RSS, so measure RSS before trying it.

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

## Concrete next work

| Order | Action | Acceptance / constraint |
|---|---|---|
| 0 | Multi-node scaling and MPFR kernel cost | Existing aligned exchanges, packed Grams, pooled border work and leaf splitting remain. Batched residue products improved Λ27 eight-core diagnostic windows by 6.40%; next gate is a complete audited release ABBA of the latest admission rules. Smaller SVD tiles regressed complete Ising19 at 32 cores and are removed; disconnected QR is also rejected. Preserve rounded Z and leaf subtraction order; Y = L⁻¹B contains rounded recurrences and cannot be replaced by one exact GEMM. Scaling latency remains bounded by the largest cone; GMP `addmul_1` is ~80% of contribution samples, not 80% of the whole solve. Avoid node70. |
| 1 | Profile remaining exact residual and sampled/PSD work | Diagonal reserve222441 is kept; single-product scratch gate passes. Target measured complete-solve costs with unchanged exact operator/rounding. |
| 2 | Optimize Lambda sampled RHS/KKT and PSD scaling | Target current dominant costs; ≥2% audited solve gain or clear memory/correctness benefit. Preserve exact operator/rounding contracts. |
| 3 | Inspect remaining storage/setup lifetimes | Preserve shifted factorization versus unshifted residual operators; structural counts alone do not establish RSS. |
| 4 | Improve LP/SOC/general sparse task balance | Keep rounded Z, numerical rules and visible thread regressions. Do not reopen late-correction work without new shift-probe evidence. |
| 5 | Diagnose the remaining Float64 iteration gap | Use completed222437 small/larger comparison and phase receipts; separate setup/solve costs. Preserve numerical contracts. |
| 6 | Scale only when the active workload needs it | Bounded frozen sources; Lambda43/1216 and 4/16/64/256-core campaigns remain later work. Recheck inputs/jobs before resuming. |
| 7 | Improve PMP only when workflow cost warrants it | Whole-command time/memory and output equivalence; secondary to solver. |

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

## Decisions still needed / deferred

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
- Distributed restart, broad sweeps and new physics searches await concrete
  need. Old g0/application campaigns stay stopped unless explicitly resumed.

## Closed directions

Do not reopen without new evidence; numerical reasons, timings and sources
are in [the journal](docs/JOURNAL.md).

- Lower/mixed precision, relaxed refinement, NaN clamping and MᵀM eigenanalysis
  violate numerical contracts. Fixed-point SVD replay erased tiny MPFR values.
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
  wider arithmetic stays native. Scalar power-of-two multiplication is a
  separate deferred external proposal, with no solver gain assumed. Exact
  binary64 constructors pass native proofs but have no qualifying workload scope.
- PMP zero-term trimming and Serde `collect_str`: rejected measured variants.
- One exact residue Gram for the whole arrow border (all leaves and ranks,
  residues summed by allreduce): Λ27 1024-bit 49.6 s versus 45.2 s for
  per-leaf exact dots on one node, 30.1 s versus 21.5 s on two; removed.
- Dropping the condensed inner refinement on g0 (0.034 → 0.030 s): refinement
  levels stay (2026-09-24 decision).
- Streamed Float64 dense Schur spans (RSS −15…28%, time +2.5…5.7% on medium)
  and a precomputed alias index (−0.7%): identical points, not kept.

Cluster: OpenMPI TCP (`--mca btl self,vader,tcp`), `openib` hangs; avoid node70.
