# SDPX development plan

Updated 2026-10-04. [AGENTS.md](AGENTS.md) defines working rules and numerical
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
iterations versus MOSEK 21. No current matched SDPB comparison exists.

Pending: PBS222447 (sampled scratch clearing and scalar power-of-two
multiplication, three cases against kept 222444). Later generic changes have
no MPI acceptance; MPI offset reuse passed its real two-rank gate (222378)
only for that exact patch. Full job histories are in the journal.

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
  comparison with that history. No new matched SDPB comparison is available.

## Concrete next work

| Order | Action | Acceptance / constraint |
|---|---|---|
| 0 | Multi-node scaling of the ordinary MPI path | Distributed arrow is in place. Next: shard cone scaling (`affine_ds` and friends need all λ), avoid replicated m-vector gathers, fused sampled path under MPI, then a 1/2/4-node Λ27 campaign avoiding node70. Retire the owner-partitioned path only after it is beaten. |
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

- Owner cost histories, fingerprint plumbing and `with_identity` constructor
  ladder stay until a real scaling campaign establishes their value. Two-node
  benefit remains unproven; historical Lambda runs showed no advantage.
- Ordinary per-site and owner MPI paths converge only after real MPI E2Es
  (local OpenMPI: conda env `sdpx-mpi`). `direct_kkt_solver` removal changes
  the settings schema and needs a decision.
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
