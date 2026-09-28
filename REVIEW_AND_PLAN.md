# SDPX development plan

Updated 2026-09-28. Working rules and numerical contracts: [AGENTS.md](AGENTS.md).
Completed experiments: [docs/JOURNAL.md](docs/JOURNAL.md).

## Goals

- Float64: audited medium/large solves within 1.3x/2x MOSEK time.
- MPFR: competitive with SDPB at matched precision, tolerances and hardware.
- Reproducible serial, threaded and MPI results; reduce production code toward
  30k lines by removing duplication.

## Current status

| Case | Latest evidence | Limitations |
|---|---|---|
| Gravity LP, Mac M4 | Latest packed kernels: Float64 0.182→0.093 s, MPFR128 4.54→4.19 s, MPFR256 8.46→5.80 s at one thread; MPFR128 1.92 s at four | One release batch per width; peak RSS decreases. MPFR points identical and audits pass; unbounded Float64 retains its dual-residual failure |
| Medium, Float64 | `Solved`, 18 iterations; dual residual 1.92e-6 exceeds the 1.75e-6 external gate | Still fails; user deferred numerical work |
| Ising11, MPFR512, node9 | Final release A–B–B–A: 25.505 → 24.901 s at one thread; 7.594 → 7.469 s at four; `Solved`/52, all audits pass | One batch per width; exact points across versions and threads; Lambda43 scaling pending |
| CSDR alpha-count 3, MPFR256, node9 | Final release A–B–B–A: 68.292 → 56.268 s at one thread; 29.555 → 17.822 s at four; `Solved`/57, all audits pass | One batch per width; exact points across versions and threads |
| Ising Lambda19, MPFR768, node9 | Final release A–B–B–A, 4 cores: 456.268 → 422.141 s (−7.5%); peak PSS 1111.83 → 911.54 MiB (−18.0%); all `Solved`/119 and audited | One batch against accepted Rust baseline; all points identical. Final integration passed; no new matched SDPB solver comparison |
| Lambda19 spins 0–50, MPFR768 | Historical rns77: 237.6 s on one node, 238.7 s on two nodes over TCP; audited | No demonstrated two-node advantage |
| Large, Float64 | Historical 129 s / 26 iterations | Current revision needs a fresh baseline |
| Mixed Ising Lambda43, MPFR1216 | Exact-precision release built; SDPB 3.1.0 built; corrected input generation queued | Corrected isolated σ/ε constraint queued; 4/16/64/256-core comparison and full audit pending |

CSDR uses automatic `local_soc_arrow`: independent SOC3 blocks, two variables
per block, block-local P, 1–128 equality rows and bounded workspace. One/four
thread points are bitwise identical. Medium points remain unchanged.
The earlier compact-coupling change rounds each MPFR sum once and differed
below 5e-51 in scaled distance from its preceding backend.
The old Julia CSDR median (17.532 s) is **not a matched comparison**: its frozen
input is missing, the reconstructed objective differs, and arithmetic differs.

The solver CLI now builds Float64, MPFR128/256/512/768/1024 by default.
Add `all-precisions` when rebuilding the CLI for Lambda43 at 1216 bits or
other nondefault precisions. Existing cluster arms, native Rust, PMP and C ABI
precision support are unchanged.

## Known failures

- **Gravity Float64:** `Solved`/17 at 1e-6, but original dual residual 4.011564682e-6 exceeds the 2e-6 external gate. The solver normalizes residuals by variable norms as well as data norms; retain those convergence rules. MPFR128/256 pass.
- **Medium:** external dual-residual failure above; investigate any new worsening.
- **SDP_control3:** 29 iterations, primal residual about 4.98e-4; input hash
  `d3a7cc27…6e84`. Deferred by the user; retain in the full input collection.
- **MPFR `condensed_graded`:** ignored extended regression still fails.
  Its Float64 counterpart was fixed and enabled on 2026-09-27.
- **Normalized 2×2 PMP diagnostic:** `AlmostSolved`/24 at MPFR512 and `1e-42`,
  identically in the preceding and candidate solver. The analytical optimum
  is 0.75; status remains a failure. Input and evidence are under
  `~/.cache/sdpx-e2e/pmp-rust-20260928/validation-v1/matrix2*`.

## Latest integration

Eight-hour campaign: 2026-09-28 03:04–11:04 China time. SDPX is primary;
PMP conversion is secondary. Substantial work runs in PBS.

Combined release `221792.node220` exited zero:

- Full workspace: **697 passed, 0 failed, 59 ignored**, including doctests
  and seven repaired distributed fixtures; no compiler warnings.
- Pinned medium/Ising11/CSDR and Lambda19 A–B–B–A preserve exact points.
  Ising11/CSDR also match across one/four threads. Medium's failure is unchanged.
- Matrix-valued sampled constraints pass all five audits and one/eight-thread
  parity. Exact bilinear accumulation changes baseline low-order digits by at
  most 1.24e-131 in componentwise scaled distance; audit gates are unchanged.
- Real two-rank MPI, ordinary exponential/augmented-LP routing and C ABI audits
  pass. Automatic settings respect a four-thread cap with 16 CPUs available.
- PMP passes six fixture checks, all 31 precisions, signed-zero byte parity,
  12 library API checks, thread/error-cleanup checks and four converted solves.

Evidence: `~/.cache/sdpx-e2e/combined-release-20260928-v1/evidence/`.
Diagnostic profile `221805` also exited zero, passed its audit and reproduced
the uninstrumented point exactly. Its wall time is not a matched benchmark.

## PMP conversion

Final release comparison on node9, Ising11 XML at MPFR1216. Times include
process/MPI startup and uncompressed JSON output; memory is sampled peak PSS.

| Workers | Previous → current time (s) | SDPB 3.1.0 (s) | Previous → current PSS (MiB) | SDPB PSS (MiB) |
|---|---|---|---|---|
| 1 | 3.265 → 3.207 | 4.688 | 10.87 → 6.77 | 125.62 |
| 8 | 0.856 → 0.633 | 2.521 | 77.18 → 40.01 | 160.47 |

One preliminary A–B–C–C–B–A batch, with separate memory runs. The eight-worker
Ising time improves 26.0% and memory 48.2%; it is about 4.0x faster than SDPB
on this case. Gains vary: the small 16-pole case at eight workers changes from
0.134 to 0.140 s, while SDPB takes 0.674 s. Automatic sampling stopping targets
differ between the implementations; Rust retains full working precision.
The SDPB build identifies as `3.1.0-dirty`; all 314 production/build files match
upstream commit `fec8e934`. Tracked differences are documentation/ignore-file
line endings only. No Lambda43 conversion result is available.

## Pending cluster study

The older Lambda43 generator `221692` is still running without an output.
Corrected generation, conversion and calibration (`221696/697/699`) remain
held. Cancellation approval for the obsolete chain is unanswered; artifacts
and jobs are preserved. The `ucas-hpc` skill requires approval before cancellation.
Unstarted `221739` is also held because its snapshot lacks the final fixture repair.
The corrected model, calibration and 4/16/64/256-core comparison remain pending.
A restart manifest with final binary hashes is saved in
`~/.cache/sdpx-e2e/eight-hours-20260928/cluster-handoff.json`. The held
calibration uses an older solver and does not qualify the final binary.

## Retained improvements

- Reuse the PSD scaling Gram in condensed KKT assembly; reuse identical RNS
  SYRK operands. Direct SVD and all numerical contracts remain unchanged.
- Avoid unused left singular-vector storage and allocate the second PSD step
  workspace on demand; reconstruct eligible exact products in their destination.
- Share compatible basis-residue caches, store residues exactly in three-byte
  integers/f32, encode bounded prime groups and release invalid storage early.
- Parse sampled decimal strings directly, skip sparse zero coefficients, move loaded
  data and buffer streamed CLI result files.
- Stream PMP input blocks and output rows; hold MPFR coefficients for one
  matrix entry per worker; use direct MPFR decimal strings.
- Support CLI/C ABI precision in every 64-bit increment from 128 to 2048,
  including 1216. Keep original-coordinate output and status unchanged.
- Use compact SOC coupling and exact modular rank proofs with fallback for
  CSDR; cache exact scalar/matrix sampled forms and schedule larger PSD cones first.

Rejected candidates and isolated experiments remain in [the journal](docs/JOURNAL.md).

## Next work, in order

| Priority | Work | Evidence and acceptance |
|---|---|---|
| 1 | Complete the authorized mixed-Ising cluster comparison | Freeze the corrected Lambda43 input at 1216 bits; compare SDPX and official SDPB 3.1.0 at 4/16/64/256 cores with memory and original-coordinate audits |
| 2 | Measure distributed scaling and memory | Final MPI/C ABI correctness passed; measure per-rank memory and scaling on the corrected large model |
| 3 | Reduce sampled RHS work and SVD replay cost | Final four-core profile: KKT update 40.1%, PSD scaling 33.8%, KKT solve 12.0% of solver-loop time; replay is 62.3% of summed SVD work. Preserve direct SVD and audited points |
| 4 | Consolidate shared solver/distributed code | Final routing and budget checks pass; reduce duplication with exact complete-solve parity |
| 5 | Revisit Float64 performance when resumed | Keep the deferred medium and SDP_control3 accuracy failures visible |

Ising already uses sampled condensation and an arrow backend. Its PSD blocks
need NT scaling and direct SVD; the SOC3 elimination does not apply. Current
solver tables report release medians: API time for pinned cases and whole-command
time for Lambda19. Cluster and earlier Mac timings are not directly comparable.
The KKT update timer includes batched constant/affine RHS work; factorization
itself is about 7.9% of solver-loop time. Phase timers and per-cone SVD totals
overlap across workers. Do not sum them.

## Validation

Follow [AGENTS.md](AGENTS.md). Preserve precision, tolerances, status and
original-coordinate audits. Refactors require exact point parity; MPI changes
require real cluster MPI. Known failures stay failures. Use release arms for
quoted timings and record kept/reverted changes in the append-only journal.

## Closed directions

- Serde collect_str output wrappers: removed from consideration after the
  serial Ising conversion slowed without reducing peak memory.

- PMP zero-term trimming: removed after release parity/audits passed but the
  matched Ising conversion showed no convincing speed or memory benefit.

- Zero-pair SVD replay shortcut: removed after the complete fast-profile
  comparison showed no speed or memory benefit; points and audits matched.

Do not reopen without new evidence; details and rejected candidates are in the journal.

- Lower MPFR precision, Float64 factorization of MPFR systems, relaxed refinement
  tolerances, NaN pivot clamping, or replacing direct SVD with eigenanalysis of MᵀM.
- SVD/eigenvector warm starts; small scalar SVD/eigenvalue substitutions without
  complete-solve benefit; forcing small sparse systems into dense factorization.
- Rejected Float64 step tuning, panel fusion, column reorder/alias schemes,
  fused-recovery GEMV removal and four-GEMM PSD application.
- Uniform MPI layouts, cross-rank splitting of individual blocks, ordinary MPI
  scaling, or excessive task splitting without new end-to-end evidence.
- Retry of refinement-pass fusion, allocator tuning or intra-cone parallelism
  based only on microbenchmarks. Existing trials showed no reliable solve benefit.

For the previously tested OpenMPI cluster, use TCP (`--mca btl self,vader,tcp`);
`openib` produced hangs/corruption. Avoid the previously invalid node70.
A valid large-N bootstrap input is required before qualifying large-border scaling.
