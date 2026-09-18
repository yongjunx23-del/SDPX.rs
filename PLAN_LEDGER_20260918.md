# A90 ledger — 2026-09-18 plan execution

Baseline `0c04025`. Working diff `78a9d0e26f13beb8` on top of it.
Frozen identities: see `/tmp/sdpx-phase0/baseline_manifest.json`.

Local timing was **not** used as evidence: `pi` held 75% of a 10-core host and the
battery sat at 4% charging. All headline numbers are cluster PBS (node3, AMD
EPYC 7742, 64 physical cores on package 0) with per-width `taskset` pinning.

## Status per work package

| ID | Status | Evidence |
|---|---|---|
| A00 | done | `baseline_manifest.json`: HEAD, dirty-diff hash, Cargo.lock, Julia manifest, toolchain, load/battery exclusion |
| A01 | done (gate **open, not closed**) | larger-Lambda11 768-bit dual consistency `2.16e-22` vs the `1e-30` gate (PERFORMANCE_PLAN.md). Preserved as a failure; not worked around |
| A02 | done | span diagnostic: `/tmp/sdpx-spans-20260918/spans.jsonl` |
| A03 | not started | pool/budget ownership audit |
| A10 | **done** | `crates/arithmetic/src/dyadic.rs`, 8 tests |
| A11 | **done** | `crates/arithmetic/src/integer.rs`, 12 tests |
| A12 | **blocked by measurement** | not built — see `A13_decision.md` |
| A13 | **reject** | `A13_decision.md` |
| A20 | done (measurement) | 3.06 RHS applications per iteration, measured earlier with temporary counters |
| A21 | **done (interface + tests only)** | `solve_many` + `SolveCounters`, 3 tests x f64/256/512 |
| A22 | not started | needs the lazy constant RHS; the main loop does not call `solve_many` yet |
| A30–A33 | not started | no qualified target: Ising512 QDLDL is 2.4%, so the Amdahl cap for the whole decomposition phase is `1/(1-0.024) = 1.0246`, i.e. 2.46% |
| A40–A42 | **deferred** | entry gate unmet: no qualified memory-bound instance |

## A02 measurement (the Phase-1 gate)

75,636 `gemm` + 2,222 `syrk` calls in one 512-bit Ising512 solve, no non-finite
operands:

| kernel | max m x n x k | exponent span |
|---|---|---|
| gemm | 16 x 31 x 16 | 576 bits (exp -540 .. +36) |
| syrk | 31 x 31 x 16 | 85 bits |

Cross-check: `75636 * 16*31*16 * 118 ns = 70.8 s`, i.e. the solve.

Consequences for a modular backend:

- Image width is about `2 * prec + span` = 1600 bits at 512-bit -> **81 moduli**
  for signed CRT uniqueness (`prod(p) > 2B`), beyond any 64-prime table.
- Contraction depth is `k <= 16` by construction (PSD block orders 12..16).
- Measured: projecting one kernel's 752 operands into 81 moduli costs
  **1.318 ms** against **0.936 ms** for the kernel's entire MPFR FMA work —
  **1.41x**, projection alone.
- Scales as `prec^2` (moduli x reduction width) versus MPFR's ~`prec^1.6`, so the
  disadvantage grows to an estimated 4-5x at 1024 bits while `m*n*k` does not
  change.

Verdict: CRT cannot win when `k / K <= 0.29`. Rejected on measurement, as
PERFORMANCE_PLAN section 1.3 ("small matrices / unsuitable data keep the existing
implementation") and section 10.3 ("small kernel fast, whole solve not fast")
both allow.

## End-to-end acceptance

### Local (correctness only; timings inflated by host load)

| width | status | 1e-30 audit (first / warmed) | objective |
|---|---|---|---|
| 1 | Solved | `true` / `true` | `-0.28388466632834991310779120004428472028243171823494868` |
| 4 | Solved | `true` / `true` | identical to 1 thread |

Gap `3.346e-43` at both widths; objective is bit-identical to the reference audit.

### Cluster (job 213475, node3, PBS, all cells `accepted: true`)

| width | previous candidate (213461) | this tree (213475) | delta | speedup vs w1 |
|---|---|---|---|---|
| 1 | 60.870 | 60.381 | -0.80% | 1.00x |
| 4 | 20.375 | 20.119 | -1.26% | 3.00x |
| 8 | 13.269 | 13.230 | -0.29% | 4.56x |
| 16 | 12.233 | 12.181 | -0.43% | 4.96x |
| 64 | 15.388 | 14.381 | **-6.54%** | 4.20x |

Parallel efficiency `S_p / p`: 75% (4), 57% (8), 31% (16), 6.6% (64).

Caveat: node3 is shared with other tenants' jobs, so the sub-1% deltas at
widths <= 16 are inside the noise band. The 64-core delta is larger than that
band and is attributable to the congruence column tiling, which is the **only**
live behavioural change in this tree (at widths <= 16 the tiling is switched off
by construction, so those widths should be arithmetically identical).

## What is live versus inert in this tree

Live:

1. **Congruence column tiling** (`condensed.rs`): one block's congruence GEMM is
   split into disjoint output-column tiles when the pool is wider than the block
   count. Bit-exact, covered by `wide_pool_splits_congruence_tiles_{f64,256,512}`,
   worth -6.5% at 64 cores.

Inert (compiles, tested, but not on the solve path):

2. `dyadic.rs` / `integer.rs` — exact dyadic view and exact product image. Correct
   and independently useful as an oracle and planning tool. The compiler reports
   `counters` / `solve_many` / `solve_many_by_columns` as never used outside tests,
   which is the honest statement that A21 does not yet change the dataflow.

## Rollback boundary

- `A10`/`A11` are additive-only (`crates/arithmetic/src/{dyadic,integer}.rs` plus
  the `mod`/`pub use` lines and `pub(crate)` on three `MpFloat` fields).
- `A21` is additive-only (trait defaults + one override + one counter field).
- The congruence tiling is the only change that can alter arithmetic scheduling;
  it is bit-exact and reverting `condensed.rs` and `condensed_parallel_tests.rs`
  to `0c04025` removes it entirely.

## Remaining uncertainty

- The 64-core gain needs a same-job A/B on a quiet node to be fully trusted.
- Phase 3 has no qualified large model: the only candidates found (587-block
  CSDR, `n=1992`) do not converge from SDPX's default start and are time- rather
  than memory-bound, so they cannot justify the distributed phase either.
- Closing A01 (larger-Lambda11 dual consistency at 768 bits) is a precondition
  for any large-model scalability claim.

## Review round (same day): contract fixes, 3D Ising receipts, serial-LDL bound

Baseline for this round is the A90 working tree above, not `0c04025` alone.

**Live in the tree now:** the five review fixes (MPFR reduced tolerances,
single-pass condensed/sampled setup indexing, rejected no-op API surface, dead
`recompile` removal, panic-to-error and partial-column guards). Evidence and the
phase tables are in `PERFORMANCE_PLAN.md`, section "September 18 static-review
round". Frozen receipts: `/tmp/sdpx-review-artifacts/`.

**Inert, unchanged from A90:** `dyadic.rs` / `integer.rs`, `solve_many` /
`SolveCounters` (now annotated `#[allow(dead_code)]` so the build is warning-free
while A22 is unattempted).

**Correction to A30.** "Ising512 QDLDL is 2.4%" does not describe this route.
The reduced factorization is 3.03 s of a 58.3 s one-worker solve (5.2%) but does
not scale at all (3.00 s at eight workers), so it is 23% of the eight-worker
solve; with its triangular solves (1.81x) the reduced LDL is 42% of that cell.
The Amdahl cap for parallelizing it is ~1.7x at eight workers and grows with
width, which also explains the 64-core efficiency collapse (6.6%) better than
the 2.4% figure did. Phase-1/Phase-2 conclusions in A12/A13/A21 are unaffected:
CRT was rejected on kernel measurements, and the interface work stays inert.

**Verified, not claimed:** 498 Rust tests and 673 Julia tests pass on this tree;
the 3D Ising Lambda=11 case reproduces 1.00x/1.76x/2.99x/4.52x at 1/2/4/8
workers with every point accepted at 1e-30; the setup fix is -20.4% on prepare
for a 200-block chordal-shaped problem and a no-op on the 22-block Ising case.

## SDPB-referenced optimization round (same day): no net speedup, corrected cost model

Baseline for this round is the review-fixed tree above. Full record in
`PERFORMANCE_PLAN.md`, section "September 18 SDPB-referenced optimization
round".

**Landed:** nothing that changes timing. Two candidates were implemented or
evaluated to the point of measurement and rejected: fused in-place MPFR
accumulators (neutral: w1 -0.5%, w8 unchanged, changes MPFR rounding) and the
two-gemm `Ginv*M*Ginv` congruence (blocked by
`graded_action_tests::condensed_graded_*`, which guards the factored form's
conditioning). CRT + double BLAS and a dense border Cholesky were rejected on
arithmetic before implementation.

**Corrected:** `kkt update` is not mostly factorization. Per solve at one worker
it is 10.22 s = scaling sync 1.78 + assembly 0.26 + value updates and QDLDL
factor 1.52 + the **constant-term solve ~6.6 s**, a third right-hand side that
shares the factorization. The reduced factorization is 20.5-23.7 ms with 192,519
inner iterations, i.e. ~110 ns per iteration, at the intrinsic cost of a 512-bit
MPFR operation. The thread-independent fraction at eight workers is therefore
~21-25%, not 42%.

**Next, in order, all arithmetic-preserving:** level-scheduled parallel QDLDL
(up to ~15% at w8, high risk), batching the constant and affine right-hand sides
(~2-4%), pooling the serial norm fraction of `residual and info` (~2-3%). The
per-block NT-scaling SVD (21.6% at one worker) is only reachable by an
algorithmic change that alters iterates.

**Certification:** 498 Rust and 673 Julia tests pass on the final tree; the 3D
Ising case re-confirms 12.99/13.10/13.25 s at eight workers and
58.73/58.40/58.56 s at one worker, 50 iterations, all external audits at 1e-30.

**Procedure note:** `git checkout -- crates/arithmetic/src/lib.rs` during the
revert also discarded that file's uncommitted A90 additions (`mod dyadic`,
`mod integer`, `from_mpfr_descriptor`, `pub(crate)` limbs). It was restored
byte-for-byte from the frozen diagnostic copy and the arithmetic test count
returned to 20. Never `git checkout --` a file that carries uncommitted work in
this tree.

## Ranked-optimization round, 2026-09-18 evening

The four ranked targets from PERFORMANCE_PLAN were worked in order; the first and
third were retired on structural bounds and the second was implemented and sent
to the cluster for its timing cell.

| target | outcome | basis |
|---|---|---|
| 1 level-parallel sparse LDL | **rejected, code reverted** | measured critical path: 158,575 of 196,448 inner iterations over 51 levels -> 1.24x ceiling |
| 2 decouple the lane count from the worker count | **implemented and retained** | cluster 213568 ABBA: -4.07% at w8, -3.39% at w16, neutral at w1/w32/w64, objectives bit-identical |
| 3 batch the constant and affine right-hand sides | **rejected on its bound** | at most the 1.10 s constant solve at w8, and batching removes only per-call overhead |
| 4 measured-cost scheduling | subsumed by 2 | the plan is scored by LPT makespan over the real per-block cost array, not by the worker count |

Item 1's rejection rests on a diagnostic build (`/tmp/sdpx-ldlprobe-20260918`,
`SDPX_LDL_DUMP`) that dumped the reduced matrix and its factor pattern at the
first numerical factorization of the production Ising solve: `n = 342`,
`nnz(L) = 11,218`, 196,448 inner iterations, 51 dependency levels whose last 27
are single columns of the dense border. The 130 lines written for it (recorded
row lists, derived schedule, replay path) were reverted; `qdldl.rs` is again
byte-identical to its commit.

Item 2 changes `crates/solver/src/solver/core/kktsolvers/condensed.rs`:
`scaling_dispatch` scores candidate lane counts `{workers, 2*workers,
4*workers, blocks}` by the LPT makespan of their contiguous cost partition,
picks the best, and derives the intra-block congruence tile count from the
longest lane against an equal share of the pool. Arithmetic is untouched and
500 Rust tests pass, including the serial-versus-64-worker equality fixtures.

Timing cell (completed, exit 0, 40/40 points): PBS job **213568** in
`hpc:~/projects/sdpx-lane-dispatch-20260918-pair01`, one node, `ppn=64`, both
arms built in-job from a frozen source manifest (692 files, `frozen.sha256`),
`diff -ru baseline/crates candidate/crates` recorded as the arm difference, ABBA
order, two rounds per width at 1/8/16/32/64 workers, `taskset`-pinned to the
first package's physical cores. Two earlier submissions (213565, 213566) were
killed by site policy with `exit_status=-9` and no output because they requested
`ppn=128`; the same script at `ppn=64` (213567 probe, then 213568) runs.

Result, median of four `taskset`-pinned points per arm per width: 61.121 vs
60.798 s at w1 (the code-identical control, -0.5% ordering offset), 13.627 vs
13.073 s at w8 (-4.07%), 12.314 vs 11.897 s at w16 (-3.39%), 13.127 vs 13.206 s
at w32 (+0.6%), 14.506 vs 14.471 s at w64 (-0.2%). All 40 points exited 0 with
an accepted 1e-30 external audit in 50 iterations; the objective strings are
bit-identical between arms and the two library hashes differ, so the gain is
scheduling only. Corrected for the w1 offset the w8 and w16 figures are about
-3.5% and -2.9%. The wide-pool ceiling is untouched: 64 workers stay slower than
16, because 22 blocks cannot fill a 64-worker pool.

The local host could not supply this cell: with Low Power Mode enabled and the
battery at 6-7%, the unchanged certified library measured 23-47 s at eight
workers against 13.0 s earlier the same day, and 71.4 s at one worker against
58.5 s, so the paired ABBA there was discarded rather than reported.
