# SDPX development plan

Updated 2026-10-30. [AGENTS.md](AGENTS.md) defines working rules and numerical
contracts; [architecture](docs/ARCHITECTURE.md) defines modules/backends.
[Journal](docs/JOURNAL.md) holds full timings, job histories, source hashes,
failed attempts and audit evidence. This file keeps goals, current state,
open work, decisions and closed directions only; history goes to the journal.

Finite conic RHS (2026-10-25, kept): the infinity sentinel now applies only
to declared nonnegative-cone bounds. Finite PSD, SOC and equality offsets
are preserved through singleton collapse and presolve, and included in the
original-coordinate audit. A feasible large-PSD reproducer changes from
false `PrimalInfeasible` to audited `Solved` at 512 bits; the singleton/SOC
tail/equality checks also pass with presolve enabled and disabled. See the
journal for the unchanged 1e-16 audit gate and evidence hashes.

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

## Current state (2026-10-30)

Version 0.9.4 adds the BFSS L13 parallel arrow path, the ORE limit cone and
the 2026-10-30 review repairs: data updates rewrite the stored original copy,
b admits +inf only in orthant rows, the limit cone's trials share the ORE
allowance, the border Schur complement is factored in place (one t² array
less) and PMP resampling works again on the Ising blocks (Λ19/512: 89 instead
of 105 iterations; `d2b1fb5` had resampled no block). Open review findings
are listed under next work.

Version 0.9.3 includes the measured Float64 kernels and BFSS ORE parallel
work recorded below. Its 25-iteration BFSS scaling table measures kernel
progress; the L11u/128 full run still ends InsufficientProgress. The scoreboard
below retains its dated source versions and is not a fresh 0.9.3 comparison.

Review repairs are integrated: initialized residue storage, factor-authoritative
original acceptance for sampled/owned solves, exact wide-exponent dot fallback,
charged row merges, bounded ORE temporaries, serial MPFR decoding and general
PMP pole/memory handling (`263f027`). Local regressions and Linux release
builds pass. PBS 228121 passes medium, Ising512 at one/four threads and MPI,
strict-gate rejection and converted PMP solves; 228241 passes the C ABI.
PBS 228246 reproduces the audited ORE point at one/eight threads, preserves
the large two-iteration point, and confirms exactly one OS thread for a serial
heavy-cone run. All scoped gates pass at unchanged tolerances. Evidence and
runner history are in the journal.

The central-PSD MPFR SVD correction retains the unshifted matrix scale for
clustering and pivot thresholds. A complete 512-bit four-by-four PSD probe
passes its independent 592-bit audit; the direction discrepancy falls from
0.1976 to 8.654e-63. The BFSS L11 convergence failure remains open. Its
experimental limit cone and primal continuation are separate from this
production correction; see the 2026-10-23 journal entry.

BFSS L13 t5/128 (`bfss-l13-threads`, `25d4286`, 2026-10-27): the parallel
arrow path replaces the release's serial QDLDL, whose first factor runs
longer than 6 h. At 64 threads an iteration takes 89–94 s, 5.1× faster than
the forced release arrow, so a 300-iteration solve takes about 8 h. Points
agree with the release and the forced arrow. At 512 bits v7 peaks at
187.5 GiB, where the forced arrow does not fit a node (journal 2026-10-27).
Since `21bb0b8` (journal 2026-10-30) the generic arrow reads its values from
the KKT and ORE rows store A sparse: t5 peaks at 63.3 GiB at 128 bits (two
iterations, v0.9.4 88.9) and 116.5 GiB at 512 bits (one iteration, v0.9.4
171.0), with identical points and an IP iteration 11% faster at 128 bits.
v8 (`46d1558`) refactors a wide leaf by exact dots when its recursive factor
fails; on v7, a restart of r2.368 max from iteration 85 fell back to serial
QDLDL. A restart reloads the iterate exactly, but its first iteration uses
the first-iteration corrector, so it trails an uninterrupted run by about one
iteration.

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
- The 2×2/3×3 Jacobi tests are scale-relative, fixing the former
  `overshooting_start_still_solves` failure without a regularization change
  (journal 2026-10-16).
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
  - Primal variables reach 5.8e118.
  - The explicit-gate pilot is prepared but not run (decision below).
  - Not the same input as the 768-bit Λ27 above, which passes its audit.
- **BFSS L11u/128:** InsufficientProgress/107. Its unchanged point and
  capped 256/512-bit timings are kernel evidence; they do not establish an
  accepted full solution (journal 2026-10-21).
- **BFSS null(A) in x:** the L11 and t5 primal variables are not identified
  along null(A). At 512 bits, t5's first step puts up to 8.6e48 into x, so
  the ‖x‖-normalized residuals (3e-41) no longer measure feasibility
  (journal 2026-10-27).
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
| 17 | `perf-l35b` | MPFR arithmetic/start changes and default PMP resampling are merged. Remaining: shorten the Λ35 gap plateau and measure matched peak memory. | Audited full solves with matched precision, settings and inputs; preserve the pinned and gravity gates. |
| 21 | `perf-mc` | Multi-core scaling (user, 2026-10-09): 8–128-thread curves against SDPB on Λ27, Λ35 and mixed Λ27/1024 (`sdpx-mpi-20261004/inputs/mixed-L27-sdp-1024`, n 18703, the SDPB-benchmark-scale case; SDPB 2 paper: Λ43 mixed 2861 s → 59 s per iteration from 4 to 448 cores), with perf stat/record, lock, allocation and NUMA data. Then evidence-chosen fixes: persistent block ownership (SDPB-style affinity), per-socket layout for 128 cores, phase overlap, distributed border/Schur for mixed problems | Ising ABBA at 64 and 128 threads with audits; Λ35 pace; mixed Λ27 2nd-iteration time vs SDPB; Float64 scoreboard unchanged |
| 22 | `perf-f64mc` | Binary64 and non-PSD cone multi-thread scaling on large cases (user, 2026-10-09: small cases are too small for threads): gravity-large (n 20202, m 20299), SU(2) medium/large, gpp500, arch0, large LPs, against MOSEK at 1–32 threads; small cases only need a measured-work guard so 16 threads never loses to 1. Also owns lock-free receipts and the process-wide pool cache (the per-call pool sites are test-only). Phase 1: serial phases run about 2× slower at t>1 under the conservative governor because the unpinned main thread hops cores | 58-case scoreboard at 1 and 16 threads unchanged; same-node A/B; large cases audited; ising11 bitwise |

The task 17/21 start, step-limit, cone and residue changes are merged.
Their measurements and rejected alternatives are recorded in the journal
(2026-10-10 through 2026-10-13). Current kernel and iteration priorities are
listed below; the historical branch states do not require another merge.

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

### Open review findings (2026-10-30)

From the max-effort review of `crates/` at `7737088` (journal 2026-10-30);
the BFSS figures are for L13 t5 (t = 13,156).

- Arrow failures fall back to serial QDLDL silently (`arrow.rs`
  `use_arrow || factor_fallback()`), which takes hours per factor on t5.
  Triggers left after `46d1558`: the non-finite precheck, `factor_bounds`
  (no exact retry), exact-duplicate rows in `factor_dots` and the border
  factor. Returning the failure to regularization escalation changes a
  contract: measure it first.
- Owner/MPI Float64 recovery lacks the retained-row rule of `04442ef`, so
  owner solves of qap6/gpp may stall AlmostSolved. Needs owner Float64 runs.
- Arrow admission uses fixed 24× work ratios that were never calibrated; t5
  is at 12.3×, and a 25% larger trunk flips it to serial QDLDL.
- Fixed 2026-10-30 (`21bb0b8`): leaf h/b, the border C and the dense ORE
  rows are gone (t5 peak −25.6 GiB at 128 bits, −54.5 GiB at 512). Closed:
  skipping zero parts in the ORE congruences (about 0.4% of an L10u solve).

### Performance plan (2026-10-16)

Ranked by the Amdahl-bounded gain on the goal workloads per unit of effort (journal 2026-10-16 has the cost map and sources). Float64 is the goal furthest from done (wins 19/58 at 1 thread, 12/58 at 16), so its items come first. Kinds: B bitwise, O audited (rounding order), P policy (audit + journal + plan line), D diagnostic.

- **Tier A (Mac-local, in this order):**
  - A1 F64-SETUP/PRE-1 (B/P): split setup per stage on the sub-10 ms LPs, then charge the presolver's rational pass by operand limbs (brandy's pass is 35–85 ms of GMP gcd). Gate: brandy/bore3d Solved and audited; medium, ising11, csdr3 bitwise. **Merged 2026-10-16** (`b4e5907`, `2c8e90b`): points and removed rows bitwise identical on 18 LPs, the SOCP families and the three e2e cases; Mac setup brandy 17.9→1.9 ms, bore3d 4.8→0.5, bnl1 5.4→1.8; SOCP_nb rank +0.7 ms. Confirm on x86 (B2). Open: the t7/t9 scoreboard times with `SDPX_RECEIPT` on, which costs tiny LPs 0.2–0.55 ms on the Mac.
  - A2 F64-FAIL (P): diagnose, then fix hinf3, sched_100_50_orig, f64_large and csdr3 (Float64); +3 wins at 1 thread, +4 at 16 at today's speed. **Diagnosed 2026-10-16, no admissible fix:** sched_100_50_orig fails because the expanded 4741-dim SOC reaches w₀ 7.6e3 (cond(W²) ≈ 5e16 > 1/eps), and refinement cannot repair the auxiliary row. hinf3 passes locally with a 2–3× margin. Next: a KKT form that never builds W² for large-w SOCs (W-scaled rows), which probably also covers csdr3; f64_large and csdr3 are untested locally.
  - A3 F64-TRSV-1 (B): fused multi-column QDLDL solve (agg −5.6%, bnl1 −5.5% bound). **Merged 2026-10-16** (`dacb804`): points bitwise identical on 11 QDLDL inputs; Mac instructions −2.4…−6.1%. Kept provisionally (under 5% on the Mac); confirm with an x86 release ABBA on agg/bnl1.
  - A4 F64-PSDCOR-1 (O): binary64 PSD corrector right-hand side via dsyevr and one GEMM over the nonzero columns (arch0 15.5%, gpp124-1 14% shares). **Reverted 2026-10-16** (`cbc209e`): on x86 it turns SDP_gpp250-1 from Solved/20 into AlmostSolved/39 (r_d 3.5e-6), while hinf3 becomes an audited solve. The Mac passes both. A retry must keep gpp250-1 Solved on x86. Candidates: keep the matrix-form products and one-GEMM rebuild with dsyev or dsyevd vectors, or find which part of the change tips gpp250-1.
  - A5 F64-PSDREF (D): solves per iteration, outer refinements and offending rows on gpp100, gpp124-1, qap6, theta1. **Diagnosed 2026-10-16:** refinement is 28–38% of these solves (factorization 6–28%). Late z rows sit at the binary64 floor of H·z, so each late column ends with one non-productive correction. Bounds: a more accurate raw solve ≤8–10% (M effort); a PSD floor stop nets about 0. F64-AFF-1 is rejected (qap6 AlmostSolved). Larger T=1 targets: step-length eigmin 19–24% and cone scaling 11–13% on gpp; gpp100's GMRES continuation is about 11%.
  - A6 ITER-1 preparation (P): `sdpx-pmp2sdp` resamples by default (`--keep-samples` opts out), converter threads from cores within a memory cap. **Merged 2026-10-16** (`34e6283`): Λ19-xml/512 default conversion is byte-identical to the old `--resample`, and it solves in 89 it, audit accepted (105 it on kept samples). Conversion on the Mac: 1.8 s → 8.2 s (10 workers). On a PBS node, pass `--threads` (the default sees all cores).
  - A7 KKT-1 (P): x-first outer corrections in a decision-equivalent form. Only if the scoreboard keeps xml inputs (about 0 on resampled inputs).
  - A8 ARITH-K (B): 12/16-limb Karatsuba microbenchmark first; an E2E only at ≥10% per product under concurrent load.
  - A9 F64-ADMIT (B): binary64 PSD step admission by Cholesky for curve trials and corrector gains, capped step components, contiguous correction rebuild. **Kept 2026-10-18:** points bitwise identical; x86 job 226585 min-of-reps geomean 0.953 at T=1 and 0.924 at T=16 (control2 0.73, qap5 0.74, theta1 0.86, gpp100 0.90). Gated to PSD blocks of n ≥ 8 (truss2, n = 4, paid twice). Job 226592 (jittered): dsyevd correction vectors rejected (+3–4%, control1 audit lost; with dgesdd, gpp500-1 NumericalError). dgesdd SVDs from n = 40: **rejected 2026-10-18** (x86 job 226609, 5 jitters on 8 SDPs: geomean 0.999 at T=1 and 0.981 at T=16; gpp124-1 +7–11% with an AlmostSolved jitter, theta1 +11–13%, gpp500-1 +8%).
  - A10 F64-PROJ (B): one original-coordinate slack projection per solve (the termination check's projection is reused), and a contiguous spectral rebuild. **Kept 2026-10-18:** bitwise; Mac mcp250-1 −12.8%, theta2 −2.8%. x86 (job 226609, A10–A14 together, min of reps): mcp500-1 −14%/−16% (T=1/T=16).
  - A11 F64-SCHURDOT (B): dense-block Schur dots 16 outputs wide. **Kept 2026-10-18:** bitwise; Mac planted80 −8.5%, control3 −3.6%. x86 (226609, with A10–A14): planted80 −9%/−6%, control3 +9%/+4% (small-case spread).
  - A12 F64-SVEC (B): tiled contiguous svec conversions and a fused symmetrization in the binary64 H-apply. **Kept 2026-10-18:** bitwise; Mac gpp100 −2.6%, qap6 −2.5%, theta1 −3.3%.
  - A13 F64-LDLAUTO (P): the supernodal faer factor for every factor with flops/nnz(L) ≥ 40 (no 1e8 flop floor). **Kept 2026-10-18:** 5 of 47 local inputs change backend, all audited at the same iteration counts; Mac mcp250-1 −37%, nb_L1 −17%, axis_wide −17%, nb −8%. **x86 regression (226609):** nb +21%/+38% (T=1/T=16), nb_L1 +13%/+41%, nb_L2_bessel +25%/+52%, axis_wide +138%/+162%; mcp250-1 −25%/−2% (mixed with A10). **Floor restored 2026-10-19 (t24 Part F, 226629):** without the floor, x86 nb +26%/+46%, nb_L1 +23%/+44%, nb_L2_bessel +33%/+65%, axis_wide +151%/+142%, mcp250-1 −18%/+9%; audits pass on both. faer runs only from 1e8 flops.
  - A14 F64-QREPLAY (B): QDLDL refactors replay the row pattern the first factorization recorded instead of walking the etree. **Kept 2026-10-18:** bitwise; Mac nql60 −5.5%, sched_100_100 −6.4%, qssp30 −5.2%, LP_bnl1 −3.9%; +4 B per nnz(L). x86 (226609, with A10–A13): nql30 −15%, qssp30 −8.5%, sched_100_100 −7%, sched_50_50_scaled −12%; tiny-LP timings undecidable (±20–100% spread).
  - A15 F64-USYM (B): binary64 symmetric products (`pooled_gemm_sym`) by upper column panels, not a full gemm. **Kept 2026-10-18:** bitwise on all 17 local SDPs once the last panel takes the remainder (a one-column panel changed arch0); gpp100 −2.0%, gpp124-1 −2.1%, gpp250-1 −5%. **x86 confirmed** (226609 Part D): bitwise on 21 SDPs; gpp250-1 −8.6%/−9.2% (T=1/T=16), arch0 −8.2%/−5.2%, gpp124-1 −6.0%/−4.7%, gpp100 −3.0%/−4.9%; geomean 0.994/0.982.
  - A16 F64-UPPER (B): the binary64 H-apply drops both mirrors and reads only upper triangles (`gemm_sym_upper`, `upper_to_svec`). **Kept 2026-10-18:** bitwise on all 17 local SDPs and medium; Mac gpp100 −4.3%, gpp124-1 −3.9%, arch0 −3.4%, qap6 −2.9%, gpp250-1 −2.4%. **x86 confirmed** (job 226619 Part U): bitwise on 21 SDPs; geomean 0.979 (T=1) and 0.968 (T=16); qap6 −9.7%, planted80 −3.1%, arch0 −2.3%, gpp124-1 −1.8%/−4.1%, gpp100 −1.6%/−3.3%.
  - A17 PRE-MRANK (B): the presolve rank image eliminates column-sorted vector rows, not `BTreeMap`s. **Kept 2026-10-18:** bitwise on all 51 local cases (exact integer arithmetic, platform-independent); setup nql60 23.9 → 15.1 ms; Mac nql60 −2.9%, nql30 −4.1%.
  - A19 F64-LTPAIR (B): the QDLDL D(L+I)ᵀ solves (single and two-RHS) run the dot products of independent adjacent columns interleaved. **Kept 2026-10-18, provisional on x86:** bitwise on 48 local cases and medium; Mac, single-RHS step: chainsing −6.4%, qssp30 −4.9%, bnl1 −4.8%, agg −4.2%, nql60 −3.1%; two-RHS step: a further −1.4% to −3.7%.
  - A20 F64-CPROD (B): the dense Schur transform forms G·Aⱼ in 16-row register blocks, one store per element, in the AXPYs' term order. **Kept 2026-10-18, provisional on x86:** bitwise on 17 local SDPs, medium and ising11; Mac planted80 −4.9%, control3 −2.6%, the rest within ±0.8%. **x86 confirmed (t24 Part G, with A17/A19):** bitwise on 57 cases; SDP geomean −3.5% (T=1), −4.9% (T=16).
  - A21 F64-LPAIR (B): QDLDL refactors sweep recorded fundamental supernode pairs (c, c + 1) together, in the column sweeps' operation order. **Kept 2026-10-18:** bitwise on 39 LP/SOCP inputs, medium, ising11 and csdr3; Mac nql60 −13.4%, nql30 −9.6%, qssp30 −5.7%, bnl1 −3.7%; provisional on x86. **Extended 2026-10-18** to fundamental supernode runs of up to eight columns (16 lost): bitwise on 51 local inputs and the pinned cases; Mac against pairs nql60 −9.5%, nql30 −6.3%, qssp30 −6.7%, sched_100_100 −1.7%. Forward solves sweep the same runs (a table from etree/Lnz): bitwise; Mac nql60 −7.3%, nql30 −6.1%, qssp30 −2.5%, chainsing_1000_2 −2.4%, sched_100_100 +0.5%.
  - A22 F64-LCONES (P): binary64 admits the local cone arrow when its dense work (class Schur products plus the t³/3 border factor) is at most twice the AMD sparse LDL flops, with each coupling class run as stacked BLAS panels (one GEMM for its Schur part and one per solve direction). **Kept 2026-10-19:** only nb (ratio 0.99) and nb_L2_bessel (1.34) change, both audited at unchanged iteration counts; Mac nb 182.0 → 73.8 ms (−59%), nb_L2_bessel −33% (T=16 −61%/−37%). The other 49 local inputs and the pinned cases are bitwise identical. Ratio 2.3–19 cases lose 1.5–4.5× with panels (sambal, nb_L1, axis_many, three LPs). The binary64 crossing SOCP ends AlmostSolved at the iteration cap on both paths. Provisional on x86.
  - A23 F64-SYMVRUN (B): the serial binary64 refinement residual sweeps runs of up to four adjacent dense constraint columns (≥64 rows, 4/5 shared) over their merged row list, so the columns' sums form independent chains. **Kept 2026-10-19:** bitwise on 51 local inputs and medium; Mac nb −10.0%, nb_L2_bessel −6.9%, nb_L1 −4.4%, bnl1 −1.6%; a 16-row minimum lost 6% on agg. Provisional on x86.
  - A24 F64-CORBUDGET (P): fewer binary64 centrality correctors. **Rejected 2026-10-19:** correctors are 43% of gpp124-1 (15 of 35 attempts rejected); off or capped at one, gpp gains 19–28% but qap6 +43%, theta1 +13% and hinf3/sched end AlmostSolved; three rejection-driven rules each lose arch0 or sched statuses for at most a 1.5% geomean gain.
  - A18 F64-BANDEIG (C, contract-free policy): the PSD centrality correction computes only the eigenpairs outside the band [0.1σμ, 10σμ] (dsytrd, dstemr on each side, dormtr on those k vectors; median k/n ≈ 0.03 on gpp100), not all of them by dsyev. Not bitwise. Local: all 17 SDPs Solved, audits pass, iterations ±1–2. Mac median: gpp100 −9.7%, arch0 −12.9%, theta2 −5.7%, qap6 −5.7%, gpp124-1 −4.3%. **Rejected 2026-10-18** (x86 job 226619 Part E, 8 jitters): gpp250-1 ends AlmostSolved in 2 of 8 jitters (0 of 8 on the base), geomean 1.020 (T=1) and 0.984 (T=16); small blocks slower (hinf2 +22%, truss1 +25%). Patch `~/.cache/sdpx-e2e/patches/band-eigen.patch`.
  - Measurements first: M10 (Float64 counts, setup split, failure traces), M1 (Λ19-xml/L27-s50c correction split), M2 (goal-width profiles: L19-rs/768, L27-s50c/768, L35-rs-1024), M4/M5 (scale-cones split timers, dqds counters), M3 (CRT digit histogram), M9 (kernel microbenchmarks).
- **Tier B (cluster; authority granted 2026-10-16, bounded jobs under the `to` prefix):** B0 frozen x86 deploy of thermal-ore with the Λ35-rs ABBA (**done 2026-10-16**: `395a3b6` vs v0.9.2, node26, 64 threads: Λ35-rs −1.0%, Λ27-s50c −1.4%, same iterations, audits accepted; the 58-case scoreboard of `2709067` is job 226044), scoreboard re-measure with audits (L19-s50/768 is the only row below 2×, 1.85× on Oct 9 data), SDPB 3.1.0 references (Λ19/768, ising11 at 768) and same-node peak PSS; B1 ITER-1 matched resampled scoreboard (**done 2026-10-16** against official SDPB 3.1.0, node26, 64 cores: Λ19-rs/768 5.67×, Λ27-rs/768 4.95×, Λ35-rs/1024 4.48×, Λ19-xml/768 4.19×, ising11/768 4.70×, all audited; the Oct 9 "matched" rows used SDPB 67ebd53; open: L19-s50/768 and peak memory against 3.1.0); B2 58-case Float64 scoreboard at HEAD and same-node abfam for A1–A4 (**done 2026-10-16**: 54 audited solves; A4 reverted for gpp250-1; v0.9.2 → `211a192` same-node LP/SOCP 0.22–0.98, SDP flat, tiny cases neutral after the one-thread input pool fix; open: theta1 T=16 1.22); B3 F64-THR (binary64 width from measured work, task 22; **done 2026-10-18**: binary64 cones whose largest cone holds over four fifths of the work, or that hold under 2²¹ cost units, keep PSD phases serial and drop a pool the KKT factor does not use. node7, 16 threads against 1: arch0 2.03 → 1.02, gpp100 1.23 → 1.02, theta2 1.42 → 1.14, control3 1.39 → 0.97, mcp250 1.24 → 0.97, nb 1.20 → 0.98, statuses and iterations unchanged; the solve stays pinned to the first CPU. Open: theta2, mcp500 1.10, chainsing_1 1.37. Pinning only with a pool was rejected (geometric mean 1.005; qap6 +45%). Sub-200 ms T=16/T=1 ratios carry a run-order clock bias: CPU 0 is warm for T=16); B4 x86 width limits for the lean kernels (ARITH-3) and an x86 12/16-limb product (ARITH-2); B5 CRT-1 top-down partial CRT, only if M3 and the x86 share bound it ≥2%; B6 goal-width confirmation and the cone bundle (CONE-1 exact-dot Cholesky, blocked R⁻¹, first Gram–Schmidt skip); B7 task 21 OWN-128/HI-T1; B8 RCPT-1; B9 faer under load (measurement).
- **Tier C (break-even first):** SOLVE-1 recovery from an exactly rounded difference (pays below about 0.11 s/it extra on Λ35-rs); ITER-3 deep-first τ start with an online overshoot restart (rs/pf inputs); F64-ITER step-limit diagnosis; F64-AFF-1; CONE-3 dqds shifts; the MPFR PSD corrector pays when t_call < (ΔN/calls)·t_it (re-measure t_call at HEAD).
- **Off-goal track (thermal-ore/gravity, after the goal items):** RES-1 column-streamed direct refinement residual, KKT-4 D-exponent-banded congruence primes, ORE-HS exact-dot `hs_prod`, ORE-AFF/ORE-SEC step searches, ARITH-4 short-dot fixed costs, ORE-LEAF, RES-4.
  - BFSS-THR (branch bfss-threads; journal 2026-10-20). **Kept 2026-10-20:** the MPFR KKT value update runs per cone in parallel, with one value plan per index map. The arrow leaf phase runs as board lanes from outside the pool and forms its residue products in-phase within `BATCH_WORK_BYTES`.
    - BFSS L11u, 256 bits, node7, 25 iterations, bitwise identical to 66e64b7: T=8 −3.0%, T=16 −8.0% (21.50 → 19.75 s/it), T=32 −14.9%, T=64 −9.6%.
    - The 128-bit full run reproduces InsufficientProgress/107 bitwise. 512 bits −8.4%, L10 −1.0%, csdr3 −1.9%; binary64 nb within 1%.
    - Rejected: the parallel ORE step search (L10 +3.5%) and the blocked exact Y solve (slower on x86).
    - Open on this case: ORE scaling (Schur inverse and Cholesky, 4.3 s/it at 15.6 busy); step lengths (2.3 s/it at about 4 busy, on the S₀ critical path); the Y solve's exact dots (38% of cycles). T=64 spends 3.4× the CPU of T=16 for 1.3× the speed.
  - ORE-SCALE (branch ore-scale; journal 2026-10-20). **Kept 2026-10-20:** the residue GEMM copies strided sub-block operands dense and runs its exponent scans, residue decode, chunk residues and large zero fills over its ways. Its prime groups give every way a column panel. The ORE cone's serial O(n²–n³) passes run as items.
    - BFSS L11u, 256 bits, 25 iterations, bitwise identical to c80da23: T=32 −23.0% (node7), T=64 −27.6% (node61). Scale cones at T=64 4.91 → 2.00 s/it; the refactor −23%; RSS +5.6% at T=64.
    - Gates: the 128-bit full run reproduces InsufficientProgress/107 bitwise; 512 bits and L10 bitwise; csdr3 identical.
    - Open: at T=64 the n = 38 cone (1.9 s/it) is GEMM compute plus helpers shared with the n = 34 cones; the base-case dot kernels add 0.28 s.
    - Review of 18b84c5 (2026-10-20). Two defects, both in `rns_blas::gemm`; neither changes the bits. Both are fixed in 036585c (RNS-X86, merged 2026-10-21).
      - The panel-fill prime groups bypass `GROUP_SCRATCH_BYTES`. On tall packed operands this means up to 16 primes of streamed scratch, for example the distributed border product at T=64: about 3.4 → 5.8 GB per call.
      - `prod` moved from `take_scratch` to `take_zeroed`, which zeroes a reused buffer again even though `prime_columns` writes every entry. This undoes L56.
  - ORE-DIAG (diagnosis 2026-10-21; journal 2026-10-21). The ORE cone's barrier, derivatives, S₀, dual test and corrector are correct (256-bit check against an independent D). On BFSS L11u:
    - The k = 1 (n = 38) cone's Dikin dual certificate caps the affine step at a median 0.136 (≈ √(2/ν)) in 91 of 107 iterations, so σ ≈ 0.64 and the solve needs 107 iterations.
    - The stall is an unattained optimum along (0, W, W) KMS directions: |x| 1e9, dual above primal through x·(q + Aᵀz).
    - Rejected so far: σ from an uncapped step and wider dual certificates (ORE-SIGMA), and primal–dual scaling (ORE-PD). Open on the model side: bound or remove the (0, W, W) directions.
  - ORE-SIGMA (branch bfss-ore-sigma; journal 2026-10-21). **Rejected 2026-10-21.**
    - σ from a step the certificate does not cap (variants A–E): L10 InsufficientProgress at 10–24 iterations.
    - Wider sound dual certificates do not lengthen steps. The Dikin test at the current s gives L10 +4.6% and L11u IP/98 with a wider gap. A Newton certificate gives L10 214 iterations (+125%), and its point fails the audit's Dikin certificate.
    - The cap is the primal-scaled direction at the certificate edge (δ ≈ 1). The tree is reverted to 01fc57f.
  - ORE-PD (branch bfss-ore-pd; journal 2026-10-21). **Rejected 2026-10-21.** Primal–dual Dahl–Andersen ORE scaling (Newton shadow point, rank-4 W folded into the ORE KKT block).
    - L10/128: Solved/159 against 198 (−19.7%), but 140 against 117 s (+20%); the audit passes.
    - The iteration drop is below the one-third bar.
    - Untried: a warm-started shadow point, true border rows, and the PD neighbourhood test.
  - BFSS-MERGE (2026-10-21; journal 2026-10-21). ORE-STEP (74da6d7), RNS-X86 (036585c) and EXACTDOT-X86 (c64f2a4, aec492a) are merged into bfss-threads (2539893). Every gate is bitwise: 128-bit full run, 512 bits, L10, ising11 and csdr3.
    - BFSS L11u, 256 bits, 25 iterations, node180, against 66e64b7: T=8 −34.2%, T=16 −40.1% (21.82 → 12.91 s/it), T=32 −45.2%, T=64 −45.6%.
    - 128-bit full run: 1411.7 → 928.9 s. Peak RSS +12% at T=64 and +20% on the 128-bit full run (not attributed).
    - LANE-HANG (fixed 2026-10-21; journal 2026-10-21): the arrow leaf lanes from c80da23 could deadlock on L10/128 at T=16. With the start barrier (61fcd50, merged as cb444d9), hangs went from 3 of 4 runs on node55 and 5 of 32 in the lane-hang repro to 0 of 6 and 0 of 32. All gates are bitwise and timing is neutral.

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
   - 2026-10-15 (journal): the pair-path interior back-substitution now runs on the owner pool (bitwise). fix-a −3.1% (T=2) and −7.1% (partitions 2); d6n640q −6.5% (69.5 s vs plain 63–66 s).
   - 2026-10-15 (journal): condensed cone scaling queues one job per lane of linear-cost blocks instead of one per block. The 20200 one-row orthants had serialized on the queue lock. fix-a owner −2.8% (T=2) and −9.6% (partitions 2); d6n640q −2.4%; bitwise.
   - 2026-10-15 (journal): granted RNS ways post to the cone-scaling board, and one-prime groups split each prime's product by output columns. The cone queue's idle workers had never stolen the Rayon splits of the ORE Schur products. one_L12/128 (4 it) T=4 −30.5% (faer) and −7.2% (Accelerate); two_L6, fix-a and d6q are neutral; bitwise.
   - 2026-10-15 (journal): RNS products skip the exact-zero inner terms of triangular right operands (per-column inner ranges, 32-column tiles), and the ORE L⁻¹ recursion forms its second product transposed. one_L12/128 (4 it) −10.8% T=4 faer, −5.0% Accelerate, −13.7% T=1; others neutral; bitwise.
   - 2026-10-15 (journal): exactly accumulated arrow leaves from order 256 factor recursively. Envelope cuts split them into independent halves; otherwise the coupling is one exact product per level. The blocked exact form at ≥256 bits is removed. one_L12/128 (4 it) −6.4% T=4 faer, −13.6% Accelerate, −19.5% T=1; two_L8 −3.3%. syn20 and d6n640q are audited; ising11, fix-a and two_L4/L6 are bitwise. A recursive Y solve was rejected (+10%: the plain halving ignored the envelope cut and zero columns).
   - 2026-10-15 (journal): exact diagonal congruences (owner border C·D⁻¹·Cᵀ, bound Gram) multiply 512-row blocks with 23-bit primes instead of 4096-row blocks with 21-bit primes. d6n640q −14.7% owner, −12.5% plain, −17.1% faer; fix-a −10…−13%; bitwise.
   - 2026-10-15 (journal): a cut-aware recursive leaf Y solve over the coupled columns was rejected. T=1 −3.8%, but T=4 +0.1% (Accelerate) and +1.9% (faer), two_L8 +1.7%, syn20 +3.5%. The column-parallel sweep is already balanced at 4 threads.
   - 2026-10-15 (journal): cone-queue helpers stay while a cone keeps posting items (each helped item moves the deadline to now plus the last slowest cone), and they yield the core when idle. A fixed deadline had left the ORE cone of a slower phase to run alone (2.97 s against 1.66 s helped). one_L12/128 (4 it) −8.1% T=4 faer, −5.7% T=8 and −7.8% T=16 Accelerate; ising11 neutral at T=10/16, −2.0% at T=24; bitwise. Wider generic-gemm primes and wider triangular tiles were probed: within noise.
   - 2026-10-15 (journal): the ORE Hs product splits S₀⁻¹'s rows (bitwise), and condensed ORE blocks run under measured ways. one_L12/128 (4 it) −4.4% T=4 faer, −7.0% T=8 Accelerate; T=1 and two_L6/L8 are flat; syn20 −3.1%; bitwise. A parallel CRT fold for exact diagonal congruences was rejected on the Mac (−0.9% d6n640q owner); re-test it at 64 threads.
   - 2026-10-15 L-P1 (kept): the presolver's dense rank minor gets the time of the sparse modular pass (64 dense updates per sparse unit, measured 0.49 vs 37 ns). d6n640q proves its 640 rows in 0.03 s instead of failing after 2.2 s: full owner solve −3.1% wall, bitwise.
   - 2026-10-15 L-P2 (kept): high-precision JSON arrays keep their decimal text during the parse and decode on the `--threads` pool. Process wall: d6n640q 8 it −5.8% owner / −6.5% plain, fix-a −3.8% / −7.4%, csdr3 −4.3%; bitwise.
   - 2026-10-16 L-M1 (rejected): an unrolled two-carry-chain row product for 5–12-limb mantissas on aarch64 (microbench 8 limbs 22.2 → 12.7 ns). Instructions −11…−19%, but cycles ising11/512 T=1 −1.8% and Λ19/512 T=4 +1.3%; not inlined, +3.1% and +3.2%. Patches `lm1-mul-rows.patch` and `lm1n-mul-rows-noinline.patch`. The staged 16-limb Karatsuba (`lm1b-kara16-staged.rs`) stays unapplied.
   - 2026-10-15 L-R2a (rejected): a product-ready `f64` residue cache borrowed on hits. Λ19-xml/512 T=4 instructions −1.6%, but cycles +4.7%, wall +9.9% and RSS 622 → 824 MiB. Patch `lr2a-full-cache-rejected.patch`.
   - 2026-10-15 L-H1 (rejected): the residue-cache fingerprint in parallel per-word lanes. `rns.cache.hit` 3.14 → 0.55 s per Λ19 solve, but that is about 1% of busy CPU: Λ19/512 T=4 cycles +0.3%, ising11 flat. Patch `lh1-fingerprint-lanes-rejected.patch`.
   - 2026-10-15 L-C2 (rejected): CRT digit pack through a 128-bit window, 43.6 vs 39.5 ns per 50-digit pack; the carry chain is already the critical path. Patch `lc2-pack-window-rejected.patch`.
   - 2026-10-15 (diagnosis, no change): a per-row bound for exact diagonal congruences, |C| < 2^(3P)·Σ_r 2^(2(xa_r−lo_a)+(xd_r−lo_d)), saves only 10–11 bits against 3P + 2da + dd + log2 k (the largest d rows meet the widest A rows). Over a full d6n640q owner solve (Solved, 49 Gram calls) it would cut 21 of 1186 primes (−1.8%), about 1.3% end to end; not pursued. Probe patch `lg8-row-bound-probe.patch`.
   - 2026-10-15 L-D1 (kept): `MpFloat` division at 128–1024 bits (2 ≤ N ≤ 16, regular operands) is a Rust port of GMP's `mpn_sbpi1_div_qr` (3-by-2 inverse, quotient limbs via `mpn_submul_1`), rounded once to nearest-even; equal to `mpfr_div` bit for bit (new test at N = 1/2/3/4/8/12/16 with exact, power-of-two, all-ones and one-unit-apart quotients). Per division −52% (128 bits), −35% (256), −17% (512), −9% (1024). Λ19-xml/512 T=4 wall −4.7%, cycles −2.7%; ising11/512 T=1 cycles −1.5%; two_L8/128 8 it cycles −4.3%; two_L6/128 and /256 −0.9…−1.5%; bitwise.
   - 2026-10-15 L-F1 (kept): `MpFloat` FMA at 512–1024 bits (N ≤ 16, addend below the product's top) keeps the GMP product in place, adds the shifted addend with GMP assembly over N limbs, and rounds once; equal to `mpfr_fma` bit for bit (new test at N = 4/8/12/16 with cancellations). Per FMA −13% (512 bits), −11% (768), −2% (1024). Λ19-xml/512 T=4 cycles −2.3%, instructions −5.0%, wall −0.9%; ising11/512 T=1 wall −3.2%, T=2 −0.8%; bitwise. A Rust Comba product lost to GMP 63.5 vs 20.0 ns.
   - 2026-10-15 L-G7 (rejected): parallel block encode of the constant-A residue cache. Rebuild 0.49 → 0.40 s (serial 0.42 → 0.06 s), once per solve, about 0.7% of an 8-iteration d6q run. Patch `lg7-parallel-cache-encode-rejected.patch`.
   - 2026-10-15 L-G6 (kept): exact diagonal congruences cover the upper triangle with halves cut at multiples of 32 columns (one product per off-diagonal rectangle) and 32-column tiles inside ≤ 64-wide leaves; the multiply count is unchanged. Bitwise identical; instructions d6q −1.8…−2.3% (Accelerate), −4.1% (faer); solve d6q plain −11%, faer owner −5.5%, fix-a −1.8…−2.8%, d6q owner ±noise. Square leaves were rejected (fix-a +2…+3%).
   - 2026-10-15 L-X3 (rejected): bounded single-pass exact dots for adjoint columns (`wide_output`). 0 bound misses, but retired instructions moved only −0.85% (fix-a owner) and ±1% (d6q owner). Short-column dots are dominated by the per-term work, not the exponent scan. Patch `lx3-bounded-adjoint-rejected.patch`.
   - 2026-10-15 L-X2 (kept): long-row exact products (`forward_long`) bound each output's exponent sums from per-row A ranges recorded at planning plus x's range, so they skip the exponent-scan pass. Terms are checked against the bound; a miss falls back to the scan. Bitwise identical; d6q owner −7.1%, fix-a owner −6.4%, plain ±noise (2% exposure).
   - 2026-10-15 L-X1 (kept): exact dot products add each term's shifted limbs into 128-bit slots and resolve carries once per dot (no per-term carry chain; 512 bits drops `lshift`/`add_n`). Bitwise identical; d6q 8 it −4.1% owner / −3.8% plain, fix-a −1.9%/−3.4%, two_L8 −5.9%, syn20 −7.3%, ising11 −1…−2%. Needs an x86 check on the cluster.
   - 2026-10-15 L-P4 (kept): with more than one MPI rank, the input check hashes binary payloads on the `--threads` pool instead of serializing every value to decimal. Probe: d6n640q 5.63 → 0.31 s per rank (T=4), fix-a 1.34 → 0.08 s. A multi-rank end-to-end run is pending a cluster deploy. L-P3 (Ruiz row ranges by entries) was rejected: the norm pass went 0.29 → 0.72 s.
   - 2026-10-15 (diagnosis, no change): the Mac is an M4 with 4 performance + 6 efficiency cores and one shared AMX unit. d6n640q owner (8 it) takes 33.1 s at T=1, 13.05 at T=4 and 12.42 at T=10, so phases at par 3–4.5 on T=10 are saturated, not serial. Thread scaling past 4 must be judged on the cluster (L-Q1, L-T5 and L-G5 wait for a deploy). The d6n640q Gram needs 21–28 primes of 23 bits (needed bits 3P + 2·40 + dd + 17, dd ≤ 146). Partitioned (owner) solves work for every cone type: two_L4/L6 (ORE), csdr3 (SOC), d6n640q (LP), Λ19 (PSD). Λ19-xml/512 T=4: plain 59.8 s, owner 69.3 s (105 it each).
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

**Taken 2026-10-16 (delegated contract authority):**
- Mac keeps under 5% stay the L-F1/L-D1 practice: a bitwise change is kept on Mac cycles provisionally and confirmed or limited on x86 in B0/B4. Mac keep metrics use goal-width inputs (L19-rs/768, L27-s50c/768, L35-rs-1024) besides Λ19-xml/512.
- Task 17 item F is met: at the default start Λ19-rs takes 89 it, Λ27-rs 127 and Λ35-rs 347, all audited (limits 121/183/683). Item G (resampled conversion by default) proceeds as A6/B1.
- Thermal-ore/gravity items rank after the goal items.
- The MPFR PSD corrector reopens only when t_call < (ΔN/calls)·t_it at HEAD (replaces "about half the cost").

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
- Binary64 PSD (2026-10-18): split congruence (`trmm` + `syr2k`, geomean
  1.04–1.07 on x86; t25 kernel 1.005–1.28× slower at k = 16–250), static regularization 1e-9/1e-10 (new AlmostSolved and
  failures on SOCP/LP), 'I'-range eigmin (iteration counts move both ways).
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
- 2026-10-15/16 (journal): L-M1 two-chain row products (inlined or not), Rust
  Comba and row-wise 8-limb products, an 8-limb Karatsuba (GMP's Toom-22
  crossover is 26 limbs on M1, 19 on Zen 2), L-C2, L-H1, L-R2a, L-G7, L-G8,
  L-X3; the √2-split quadratic congruence (two CRTs at about 3P bits cost more
  than the 25% fewer prime products save); pre-image step bounds through
  corrections and skipping the dead affine z congruence (≤1.7%).


QICS centering (2026-10-25, kept): the experimental feasible-primal frontend
uses projected local-norm contraction near a center. The exact original-start
L11 comparison reproduces the old Armijo failure at iteration 267 and the
modified solve passes every uncapped/original-cone audit at unchanged
512-bit precision and tolerances. The current-main port passes the small
limit-cone and retained L11 checkpoint gates. See the QICS journal entry and
PR #11 for the run inventory; the default HSD algorithm is unchanged.
