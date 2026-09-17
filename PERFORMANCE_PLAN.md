# SDPX performance plan

Target: Float64 near or above Clarabel.rs/MOSEK and high-precision SDP near or
above SDPB, without losing accuracy. These are measured targets, not promises.
Contracts: [AGENTS.md](AGENTS.md). Evaluation: [benchmark/research](benchmark/research/README.md).

## Local small-block elimination trial (2026-09-16)

The user stopped the CSDR comparison: the stable Rust solve took 280.017 s
(256 bits, 57 iterations, Optimal), while the legacy run was interrupted;
there is no completed speed ratio. A shared Float64/MPFR candidate eliminated
independent SOC3 / PSD2 blocks and retained the global border (48 variables
for the m400-a3 structure), with original-KKT refinement and QDLDL fallback.
Eight focused numerical tests passed, but a short 128-block / 8-equality
Float64 case returned AlmostOptimal where the baseline returned Optimal at
1e-12. Repeated timings were unstable. The candidate is **rejected for default
integration**; production source and the stable release library are restored.
No tolerance relaxation, approximate rank reduction, or full CSDR rerun.
Candidate patch, library and receipts: `/tmp/sdpx-local-blocks-20260916/`.
Next work must first recover the strict convergence gate, then show repeatable
end-to-end gains. Sparse QDLDL already performs elimination; the smaller
explicit border is not itself evidence of a speedup.

Second trial: reviewed the historical `fixed_trace_q3.jl` Cholesky/TRSM/SYRK
path and MFLA's scaled 2x2 solve, then replaced scalar Schur updates with one
whitened-panel Gram contraction through the existing shared provider. Eight
focused tests passed. Same-process ABBA short checks still rejected it:
MPFR256 medians 0.067875→0.070556 s (8 equalities) and 0.253255→0.318860 s
(45 equalities); Float64/8 retained the AlmostOptimal regression. No production
integration or full CSDR rerun. Evidence: `/tmp/sdpx-local-gram-20260916/RESULT.md`.

## Continuous optimization objective

Continue evidence-driven optimization without a preset end date. Each cycle:
profile the current frozen baseline; formulate one testable bottleneck hypothesis;
implement the smallest shared-precision change; run focused numerical checks and
bounded paired benchmarks; retain repeatable gains or revert. Short development
screens have explicit timeouts; broader acceptance runs only for promising changes.
Failures remain in coverage reports. Preserve original-coordinate accuracy,
fixed precision, default preprocessing and the single-engine architecture.

After several unsuccessful hypotheses, or when the dominant cost changes, review
recent relevant papers and mature source implementations, then resume with a new
measurable hypothesis. Do not repeatedly benchmark the same failed idea without
new evidence. Reserve a new independent family before making a new generalization
claim: the current fresh group is already exposed.

Immediate order: residual-stage overlap (212949) and runtime AVX2/FMA dispatch
with shared-kernel inlining (213041) have passed full acceptance and are integrated.
The combined candidate passed 291 Rust core tests, 508 Julia assertions across 32
testsets, all 56 cluster numerical receipts, 666 source and 6 binary identity checks.
The actual x86 generic/accelerated bitwise-equivalence fixture also passed.
Matched Ising512 comparison (212965) completed all 21 point audits. Refresh profiling
and matched Float64 references on the accepted build next. Sync/assembly fusion
(213040) was rejected for measured regressions. Preserve four unexposed holdouts
for final generalization checks.
Resolve the remaining medium/hinf3 and larger Lambda11 accuracy issues; continue
high-precision iteration-cost optimization and real 1/4/16/64/256-core,
multi-node single-solve qualification. Small Ising timing does not close those
larger-scale requirements.
Publish verified code and matching cluster snapshots under the user's existing
authorization. Default to working without subagents. No speed or parity claim
without matched evidence; no claim of automatic background research while Goal
is paused. The old Goal's five-equation requirement and no-push restriction are
superseded by current project instructions and explicit publication authorization.

## Current evidence — September 16, 2026

September 17 provider assessment: keep MPFR/GMP scalar semantics and the existing
small-block kernels. First compare high-precision factorization/TRSM/SVD against
MPLAPACK and large Gram products against BFLA's existing exact FLINT bridge;
include allocation and representation conversion in timings. The BFLA bridge
currently uses text transport and serial restricted SYRK, so it is a reference,
not a ready Rust production backend. Exact Gram rounded once need not match
sequential MPFR accumulation. Arb approximate products or dropped ball radii do
not establish the required accuracy. Elemental remains a distributed-backend
evaluation, not an assumed single-core speedup.

The medium batched second product in `compute_schur_dense_impl` reaches the BLAS
provider; compare hardware-appropriate Float64 BLAS there before changing the
algorithm. The previously rejected active-row tail GEMM remains rejected.
FFI now forwards existing core `sdp-mkl`, `pardiso-mkl`, and `pardiso-panua`
features without changing defaults or dependencies. Locked offline Cargo metadata
validates all three mappings; locked offline feature resolution for
`sdp-mkl,pardiso-mkl` passes. Receipts: `/tmp/sdpx-provider-metadata-20260917.json`
and `/tmp/sdpx-provider-tree-20260917.txt`. Native MKL/PARDISO builds, provider
numerical qualification, timing and memory comparisons are still pending;
this wiring earns no performance credit. No full CSDR test was restarted.

September 17 Float64 backend screen, PBS 213336 (completed, exit 0): existing
OpenBLAS versus oneMKL on one pinned CPU, sequential ABBA processes, seven warmed
samples of ten GEMMs each, after 120 ms warmup per shape. Shapes follow medium's
PSD orders 57/59/60/116 and the current `(64*n,n)*(n,n)` batched product. Inputs
are synthetic dyadic values with every first-call output checked against an
exact integer-product oracle; these are not captured solver matrices or a
solver numerical qualification. Both providers report one thread and symbol
ownership is recorded. MKL speedups (AB / BA): 1.083/1.091, 1.074/1.084,
1.281/1.207, 1.150/1.122. Process peak RSS: OpenBLAS 18,304–18,684 KiB;
MKL 24,044 KiB. Host load and runtime-dispatched child-library identities were
not exhaustively recorded, so this is a directional screen, not promotion.
Evidence: `/tmp/sdpx-blas-screen-20260917/` (source/script hashes, library hashes,
symbol ownership, raw samples, resource logs and summary). Next: frozen current
solver builds using each provider, complete dependency identity, matched medium
ABBA with original-coordinate accuracy and LP/SOCP/SDP regression checks. Keep
the production provider unchanged until that comparison passes. High-precision
MPLAPACK/FLINT experiments remain pending.

The full medium provider comparison is prepared in
`/tmp/sdpx-provider-solve-20260917/`: 218 current source files frozen with per-file
hashes, identical source for both providers, existing original-coordinate audit,
one cold and one warmed fresh solve per ABBA process, 240-second process limits,
single CPU affinity, native/API timing, RSS and loaded-file identities. The
existing cluster MKL runtime avoids downloading static MKL dependencies; both
arms use the same provider-neutral Cargo features with explicit native linking.
Python compilation and PBS shell syntax checks pass. Source upload was rejected
by automatic approval review pending explicit payload/destination authorization
for `hpc:~/projects/sdpx-provider-solve-20260917/`. Only the empty remote directory
has been created; no full-solve PBS job has been submitted. See `handoff.json`
for archive/script hashes and the precise pending state. No timing claim or
production-provider change follows from this preparation.

Independent local work while upload authorization is pending: borrowed-input
MPFR FMA candidate in `/tmp/sdpx-borrowed-fma-20260917/`. A typed `mul_add_ref`
borrows the three inputs and retains independent output limbs; GEMM's existing
fused modes call it without changing precision, accumulation order or rounding.
No production source change. Release locked/offline build passes; the existing
GEMM/SYRK test binary passes all 14 tests (128 through 2048 bits, transposes,
strides, cancellation and untouched padding). A same-binary local ABBA screen
compares baseline and candidate provider modules on identical non-dyadic inputs,
checking complete output equality after each arm. For 64-order GEMM, AB/BA
speedups are 1.069/1.063 at 256 bits, 1.071/1.070 at 512 bits, 1.023/1.031 at
1024 bits. The 16-order samples are noisier. Seven samples per arm follow
100 ms warmup; macOS affinity/background load was not controlled, so these
figures are screening evidence only. Scalar dot/cancellation checks also pass.
Candidate patch, raw samples, build/test logs and hashes remain outside the
repository. Next gate is a frozen complete high-precision solve comparison;
no solver speedup, Ising improvement or adoption is established yet. This is
distinct from the previously rejected in-place arithmetic-assignment candidate.

Local full Ising512 comparison prepared in
`/tmp/sdpx-borrowed-fma-solve-20260917/` using existing frozen sampled input,
reference and macOS audit driver. Both current-source release libraries built;
candidate has only the two borrowed-FMA file changes. A shared-target candidate
build first reused stale arithmetic metadata and failed; rebuilding in an
independent target succeeded. Preserve both build logs. All 483 frozen files
are checked before/after runs. The first baseline process completed both solves
and passed both external audits at 50 iterations, but `/usr/bin/time -l` failed
on sandboxed `sysctl kern.clockrate`; the wrapper exit correctly invalidated the
attempt and stopped before candidate execution. The repaired runner uses direct
Julia plus `wait4` process RSS/CPU accounting, keeps a 240-second process-group
timeout, and restarts ABBA under `retry-01/` without overwriting failed evidence.
The repaired ABBA comparison completed: all eight first/warmed solves pass the
original-coordinate audits, all have 50 iterations, and all returned x/s/z and
statuses match exactly. The 483 frozen files remain unchanged. Warmed native
times in ABBA order are 76.4565 / 55.7918 / 55.5693 / 56.2690 seconds; API times
are 76.5386 / 55.8753 / 55.6521 / 56.3558 seconds. AB speedup is 1.3704 but BA
is only 1.0126, below the 1.02 threshold. The first baseline is visibly unstable
(first call 47.0993s, warmed 76.4565s); do not quote the pooled 1.1918 median
ratio as a demonstrated gain. Process peak RSS in the same order is
912064512 / 928071680 / 935657472 / 928907264 bytes. No OS affinity was enforced;
a mid-run process inspection found no competing numerical solver, but does not
establish controlled hardware conditions. Reject this candidate for insufficient
repeatable full-solve benefit; keep it isolated, with no production adoption.
`decision.json`, `retry-01/`, failed-attempt logs and `evidence.sha256` retain the
complete evidence. Precision/tolerances/settings and gates were unchanged.

September 17 next independent candidate: cache each orthant row's `A/w`
quotients during condensed Schur assembly, reducing divisions from
`d+d*(d+1)/2` to `d` per degree-d row while preserving each rounded quotient
and the original FMA order. Scratch is sized once to the largest row per cone
and overwritten every assembly, including after scaling/A updates. Isolated
source and evidence: `/tmp/sdpx-orthant-division-20260917/`. Thirteen orthant
tests pass, including new bitwise assembly comparisons at Float64/256/512 bits,
nonunit weights, empty rows, repeated assembly and A updates. A separate ABBA
row microkernel screen (degrees 8/64, seven samples, 100 calls per sample)
shows 2.50–3.18x at 256/512 bits with exact output equality; Float64 shows no
benefit and small rows regress. The microkernel excludes Schur index lookup,
full assembly, factorization and solver work; do not extrapolate its speedup.
Do not adopt the uniform cache. Next evaluate a precision-aware cost choice
within the same assembly loop, retaining Float64's current computation, then
measure complete MPFR assembly and a mixed orthant/SDP solve. Pure LP defaults
do not select this condensed path, and no Ising improvement is claimed.

Follow-up `/tmp/sdpx-orthant-assembly-20260917/`: MPFR-only row caching now
retains the literal original Float64 row loop and allocates no Float64 row
scratch. Fourteen orthant tests pass. Full assembly comparisons include Schur
lookup, clearing, scattering and finiteness checks; MPFR speedups remain about
2.5–2.8x. Float64 64-degree timings still vary around a small regression, so
full Float64 solver regression remains required despite unchanged arithmetic.

A single-binary ABBA full-solve diagnostic compares the original and cached
assembly through a test-only switch. Its 32-variable, 256-NN-row, PSD32 problem
has known unique optimum x=1; dense inactive NN rows repeat, so this is a
structural stressor, not independent benchmark coverage. Default preprocessing
and automatic KKT selection are retained, with actual condensed selection
asserted. At 256/512 bits the internal tolerances are 1e-24/1e-42 and external
original-coordinate primal/dual/gap and cone bounds are 1e-20/1e-30. All 24
solves pass; returned x/s/z and iteration counts agree exactly within precision
(17/26 iterations). Each arm has one first call and two warmed fresh solves.
Warmed native ABBA medians: 256-bit 2.06256/1.93011/1.93282/2.07384s,
512-bit 4.82749/4.46088/4.46990/4.82939s. AB/BA speed ratios are
1.0686/1.0730 and 1.0822/1.0804, respectively. No affinity or per-solve RSS was
measured, and separate production release arms remain unqualified. Retain this
experimental candidate for independent mixed-cone, Float64 and Ising regression;
do not adopt yet. The clean candidate patch excludes all test-only switching
and duplicate reference assembly. Sources, binary hash, raw logs, summaries,
assertion-based mathematical checks and limitations remain in the external
directory. The production solver is unchanged.

Nonrepeating-row follow-up `/tmp/sdpx-orthant-diverse-20260917/` verifies that
all 224 dense NN rows are distinct even modulo positive scaling. This changes
the synthetic coefficients, not the solver candidate, and retains the known
x=1 optimum; it is not a new public holdout family. Float64 is added with
1e-8 internal / 1e-6 external tolerances; MPFR gates remain unchanged. All 36
ABBA fresh solves pass original-coordinate checks and exact point agreement
within each precision (7/15/24 iterations at Float64/256/512). Warmed native
ABBA medians are 2.00695/1.88055/1.88169/2.01535s at 256 bits and
4.49895/4.14954/4.15171/4.50715s at 512 bits, giving AB/BA speed ratios
1.0672/1.0710 and 1.0842/1.0856. Float64 passes correctness, but its 7–12 ms
warmed samples show substantial startup drift; claim no Float64 speedup or
qualified timing regression result. The same isolated binary passes 292 core
tests (one benchmark filtered, one full-solve diagnostic ignored).

Next release qualification is staged in `/tmp/sdpx-orthant-release-20260917/`.
All current Cargo/crates/Julia files match the earlier frozen baseline source,
allowing reuse of its traced baseline FFI library. A clean candidate containing
only the production assembly patch, without the test-only reference/switch,
is building in its own target directory. `source.json` and `baseline.json`
record identities and build provenance. Separate-library Float64/medium/Ising
timing and memory checks remain pending; production is not modified or published.

Independent release candidate built successfully and matches its frozen source.
The local library-level ABBA runner at `/tmp/sdpx-orthant-release-20260917/`
uses direct Julia processes with wait4 RSS/CPU accounting, 240-second process
timeouts, one first and one warmed fresh solve per cell, and file verification
between cells. No competing numerical process was observed at launch; macOS
affinity remains uncontrolled. Medium completed all eight original-coordinate
gates at 18 iterations, with identical returned primal values, equality duals
and matrix duals across both libraries. Warmed native ABBA times:
2.95466/2.99322/2.99854/2.98568s; API:
2.99448/3.02536/3.02583/3.01289s. Candidate regressions are 1.31%/0.43%, below
the 2% screen limit; no Float64 speed credit. Peak process RSS:
1102643200/1122074624/1101955072/1090748416 bytes (includes Julia/setup/audits
and both solves, not per-solve native memory). Results and point-equality checks
are under `results/medium-*`, `medium-summary.json`, `medium-points.json`.
Ising512 ABBA completed: all eight solves pass at 50 iterations, with identical
returned x/s/z and status. Warmed native times are
36.50510/36.52823/36.50642/36.53842s; candidate/baseline ratios are
1.00063/0.99912. This fixture has no orthant rows, so it earns regression credit
only. All 295 frozen files remain unchanged; see `regression-summary.json`.
Independent-release mixed orthant/PSD qualification subsequently passed all 24
solves, including known optimum, original-coordinate primal/dual equations, gap
and cone bounds. Every returned x/s/z, status and iteration count matches within
each precision. Unique dense rows avoid the first fixture's repeated rows; this
is still a synthetic structural fixture, not an independent public holdout.
256-bit warmed native ABBA medians: 1.80655/1.69050/1.70042/1.84177s
(15 iterations; AB/BA speedup 1.069/1.083). 512-bit:
4.51174/4.19308/4.18047/4.53450s (24 iterations; 1.076/1.085).
Each cell runs a first solve plus two warmed fresh solves, with a 120-second
process timeout; process RSS is about 557/568 MB at 256/512 bits, including Julia
and audits. All frozen files were reverified; source matched current production
before integration. Evidence: `mixed-summary.json`, `mixed-frozen.json`,
`mixed-results/` under `/tmp/sdpx-orthant-release-20260917/`.
The precision-gated orthant row quotient cache is now integrated, preserving
Float64's original loop and MPFR quotient/FMA order. Integrated release checks
pass: 294 core tests and 23 LP/SOCP/SDP/mixed-conic/data-update tests. The final
numerical source differs from the timed candidate only in whitespace and a
pattern trailing comma; `integration-source.json` records that check and hashes.
No public holdout or MOSEK/SDPB parity claim follows from this local improvement.

Reproducing the archived Ising KKT instrumentation on current numerical sources
produces identical instrumented files (`/tmp/sdpx-profile-provenance-20260917/`).
The prior four-thread profile remains applicable: Schur assembly is only about
0.051s of the 11.86s warmed solve. Complete profiling of cone scaling/SVD, step
length and residual updates before selecting a replacement high-precision
provider; do not add nested timing counters together.

September 17 extended Ising phase diagnostic: frozen current source, isolated
26-span instrumentation, 512 bits, one configured thread, one first and one
warmed solve, 240-second process cap. All four instrumented/control solves pass
the unchanged external audit at 50 iterations and return identical x/s/z.
Instrumented first/warm: 72.869/69.794s; subsequent uninstrumented control:
74.778/71.249s. This is a sequential diagnostic/control pair, not an ABBA speed
claim. Both are much slower than the earlier 36.5s session; cause is unqualified,
so historical times must not be used to infer an instrumentation regression.
MPFR/GMP archive hashes and solver build features match the prior release.

Warmed exclusive outer spans cover 68.622s of 69.794s: affine/combined KKT solves
14.401/16.822s (44.7% together), KKT update 12.689s (18.2%), cone scaling 8.181s
(11.7%), affine/combined bounds 6.367/6.197s (18.0% together), residual update
2.541s (3.6%). Nested scaling eigensolve is 5.433s (7.8% of total); nested Schur
assembly only 0.265s (0.38%). Do not add these nested counters to the outer spans.
High precision actually uses an eigensolve of M-transpose-M for NT scaling;
Float64 retains SVD. Thus prioritize direction/residual matrix operations and
the common symmetric-eigensolver path used by scaling and step bounds, rather
than assuming Gram assembly or global factorization is the largest opportunity.
Next bounded hypothesis: exploit exact symmetry in MPFR Householder rank-two
updates, comparing the original update order and eigenpair residuals before any
full-solver timing. MPLAPACK comparison should include SYEVR, not just GESVD.
No new numerical change is adopted by this diagnostic. Evidence and all frozen
identities: `/tmp/sdpx-ising-phases-20260917/phases.json` and adjacent receipts.

September 17 symmetric Householder update candidate (isolated, not adopted):
compute the upper triangular rank-two update once and mirror it. MPFR nearest
rounding, products, additions, reflectors and eigensolver stopping rules are
unchanged. Exact comparisons of the packed matrix, diagonal, off-diagonal and
reflectors pass at 256/512/1024 bits for n=0/1/2/3/12/16/32/64 with zero,
repeated-diagonal, signed non-dyadic and scaled inputs (96 combinations).
Same-binary sequential ABBA kernel screen uses two warmups and five samples of
three calls per arm, 120-second cap. AB/BA speedups at n=12/16/32/64:
256-bit 1.281/1.244, 1.268/1.348, 1.390/1.382, 1.431/1.438;
512-bit 1.212/1.225, 1.285/1.281, 1.372/1.378, 1.404/1.413.
These include resetting the input matrix and are kernel-only synthetic results;
no full-solve credit. All 297 core tests and 25 dense-provider tests pass
(the timing test is ignored in each correctness run). Independent release Ising
ABBA completed: all eight solves pass at 50 iterations with identical x/s/z.
Warmed native times 66.6210/65.1787/66.3656/65.5639s; AB speedup 1.0221,
BA 0.9879. Reject for production: no repeatable >=2% full-solve improvement.
RSS 915456000/954597376/932364288/957612032 bytes includes Julia and audits;
all 268 frozen files reverified. Evidence, source and binary identities:
`/tmp/sdpx-symmetric-update-20260917/`.

Before the next arithmetic candidate, qualify NT scaling on graded SPD inputs:
the current high-precision M-transpose-M route and negative-eigenvalue clamp
need explicit conditioning coverage against same-precision direct SVD.
This concern is now reproduced at 256 bits: delta=2^-160, t=2^-80,
S=[[delta,t],[t,2]], Z=[[1,1-delta],[1-delta,1]] are SPD and both Cholesky
factorizations succeed after the actual svec round trip. Direct SVD retains
sigma_min=6.8422776578e-49, with singular-product/determinant relative error
8.55e-50. The current Gram route returns success=true, lambda_min=0 and nonfinite
R/Rinv; determinant relative error is 1. This is a cone-level counterexample,
not a claim that the existing Ising benchmark fails. An isolated restoration of
the common direct-SVD scaling path passes this counterexample and all 295 core
tests (`fix-tests.log`). Independent-release Ising first-solve ABBA now passes
all four unchanged external audits at 512 bits and 50 iterations; native times
66.2612/70.1115/70.7975/66.2673s, API
67.3419/72.0744/71.3190/66.7898s. The direct-SVD fix costs 5.81%/6.84% in AB/BA.
Each cell is one fresh-process solve (no warmed sample), with a 180-second cap;
points match within each arm, and all 268 frozen files reverify. Peak process RSS
905953280/929480704/945242112/933576704 bytes includes Julia and audits.
This is an accuracy fix with a measured cost, not a speedup. Float64 medium also
passes first/warm original-coordinate audits at 18 iterations, with identical
primal/equality-dual/matrix-dual values to the prior baseline (no new Float64
speed claim). The direct-SVD path is now integrated; an expanded 128–2048-bit
conditioning regression checks the singular-product/determinant identity and
R-times-Rinv. All integrated checks pass: 295 core tests (including all six
MPFR precision modes in the new regression), 23 LP/SOCP/SDP/mixed/update tests
and 22 dense-provider tests. The numerical source exactly matches the qualified
isolated release; `integration-source.json` records production file hashes.
This candidate has a correctness justification independent of the performance
retention threshold. See `solver-summary.json` and `results/` in the same folder.
Evidence: `/tmp/sdpx-nt-conditioning-20260917/retry-check.log` and source snapshot.
Also inspect native library
implementations before assuming speedups: the pinned gmp-mpfr-sys 1.6.8 MPFR
`src/dot.c` allocates n product temporaries at summed operand precision and calls
`mpfr_sum`; it is not an allocation-free replacement for the hot FMA loop.

September 17 live FMA accumulator experiment (isolated): keep one owned MPFR
destination throughout an ordered dot product, borrowing input descriptors and
rounding every FMA exactly as before. The pinned MPFR manual explicitly permits
input/output reuse; its Rust custom-descriptor helpers are already inline, so
reimplementing those helpers is not a useful optimization. Six arithmetic tests
pass, including 128–2048-bit comparisons, cancellation, empty products, signed
zero, infinity/NaN and unchanged inputs. Same-binary ABBA dot screens show about
1.04–1.12x speedups across lengths 8/16/64/256 at 256/512/1024 bits; at 512 bits,
length 16 is 1.109/1.118x. These are scalar-only results, not solver credit.
GEMM retains existing FMA precision selection. All transpose/leading-dimension
checks pass; 296 isolated core tests and 40 provider/parallel/workspace tests
pass (timing tests ignored). Matrix AB/BA speedups at n=16/32/64 are
256-bit 1.095/1.061, 1.052/1.074, 1.085/1.084;
512-bit 1.071/1.060, 1.062/1.048, 1.012/1.020.
The rebuilt, default-loaded production library first passed a 256-bit SDP smoke
test, ensuring the direct-SVD correctness fix was present in the actual baseline.
Independent-release Ising first-solve ABBA then passes all four unchanged audits
at 50 iterations with identical x/s/z across both arms. Native times:
39.23635/38.17474/38.11063/39.27043s (AB/BA 1.0278/1.0304x); API:
40.03465/40.01321/38.40458/39.54388s. These are fresh-process first solves, not
warmed timing. Historical ~70s sessions are not comparable; no affinity is
enforced. RSS 907821056/916094976/892043264/910327808 bytes includes Julia and
audits. All 268 frozen files reverified. The live accumulator and GEMM adapter
are integrated. Final production checks pass: 6 arithmetic tests, 295 core
tests and 37 provider/parallel/workspace tests. The default FFI library was
rebuilt and its default-loader 256-bit SDP smoke test passes (14 iterations).
A regression explicitly distinguishes ordered FMA from once-rounded exact dot
and separate multiply/add. Evidence: `/tmp/sdpx-fma-accumulator-20260917/`.

September 17 SYRK live-accumulator candidate (isolated, not adopted): reuse
`dot_fma` for the existing fused precision modes, preserving ordered FMA and
alpha/beta/triangle semantics. 1,080 baseline comparisons cover all six
precisions, U/L, N/T/C, rectangular/empty products, padded leading dimensions,
alpha/beta and serial/two-lane pool paths. All pass; 296 core tests pass, one
ignored timing test. Same-binary ABBA (seven three-call samples, two warm calls)
at 256/512 bits and orders 16/32/64 shows 1.034–1.089x kernel speedups across
both transpose modes. This is not solver credit. Independent-library Ising
ABBA native times were 44.25548/41.32696/37.96624/42.51806s (1.071/1.120x),
but an additional BA confirmation was 42.80283/38.05422s (0.889x). All six
unchanged original-coordinate audits pass, with identical x/s/z/status and 50
iterations. All frozen identities were reverified. No affinity is enforced;
large same-arm variability and gains disproportionate to the kernel screen
make whole-solver attribution unreliable. Do not promote or repeat this timing
campaign further. Production source is unchanged. Keep the isolated candidate
for a future controlled-host comparison, without solver performance credit.
Next: reprofile the current direct-SVD plus GEMM-accumulator baseline, because
the earlier stage breakdown used the retired Gram-based NT scaling path;
prioritize the resulting direction/scaling hotspots over more small-kernel
screens. Evidence: `/tmp/sdpx-syrk-accumulator-20260917/`.

September 17 current-baseline phase refresh (completed, isolated): current
production includes direct SVD and the accepted GEMM accumulator, not the
rejected SYRK candidate. Four original-coordinate audits pass; all returned
x/s/z/status/iterations match. Control first/warm native 37.79753/37.80434s;
profile 37.84738/37.83468s, 50 iterations. All frozen identities reverified.
Warm outer phases: affine/combined solve 7.180/8.436s (41.3% combined), cone
scaling 6.746s (17.8%), KKT update 6.311s (16.7%), affine/combined bounds
3.319/3.228s (17.3%). Nested SVD is 5.730s (15.1%), with bidiagonal iteration
4.656s (12.3%); Schur assembly only 0.137s (0.36%). Congruence calls total
7.850s (20.7%), of which GEMMs are 4.370s (11.6%). Nested percentages overlap
outer phases and must not be summed. Production is unchanged. Source review
finds `svec_to_mat`/`mat_to_svec` reevaluate correctly rounded high-precision
FRAC_1_SQRT_2 inside each off-diagonal element, and svec expansion repeats the
same product for mirrored entries. Next hypothesis: hoist this unchanged
constant once per conversion and reuse the mirrored product, without global
mutable caches or changing MPFR rounding. Prioritize this conversion overhead,
then SVD bidiagonal iteration; bulk Schur acceleration is not this Ising
instance's leading opportunity. Evidence: `/tmp/sdpx-current-phases-20260917/`.
The conversion-hoisting candidate is isolated in `/tmp/sdpx-svec-constant-20260917/`.
Float64 plus all six MPFR precisions pass exact baseline comparisons at orders
0/1/2/3/8/16. A 512-bit same-binary ABBA conversion-pair screen (two warmups,
seven batches of 50 calls, immutable input) at orders 8/16/32 yields
13.70/12.84x, 17.16/17.07x and 18.64/18.94x. These are conversion-only gains.
296 core tests and 23 LP/SOCP/SDP/mixed/update tests pass; release FFI builds.
Independent-library Ising ABBA native times are
60.16795/48.80130/49.24649/58.26923s (1.233/1.183x; paired-arm medians
59.21859 -> 49.02389s, 17.2% less native time). All four original-coordinate
audits pass at 512 bits/50 iterations with identical x/s/z/status. Float64
medium first/warm ABBA also passes all eight audits with identical points and
18 iterations: warm 5.013875/4.995525/5.040042/5.013875s (1.0037/0.9948x),
essentially unchanged. All 295 frozen identities were reverified. Native
Ising/API/RSS raw receipts remain in the experiment; these same-session
numbers must not be compared with earlier ~38s profiles. The candidate is
accepted and its exact production source is integrated, preserving all unrelated
source hashes. The default FFI library was rebuilt successfully and its
256-bit default-loader SDP smoke passes at 14 iterations. Integration source
and library identities are recorded and reverified. No global constant cache,
new dependency, precision change or solver algorithm change.

September 17 MPFR rotation candidate (isolated, not adopted): SVD's column
rotations reuse coefficient descriptors and one owned scratch value, borrow
inputs until both independently owned outputs are ready, and retain all four
separately rounded products plus subtraction/addition. No FMA substitution.
Six-precision scalar checks include cancellation distinguishing separate
products from FMA, signed zeros, infinities/NaN and output independence. Full
SVD comparisons pass for 72 shape/precision/conditioning combinations (tall,
wide, square, rank-deficient and anisotropic diagonal matrices), including
identical singular values and both vector matrices. Same-binary SVD ABBA
(two warmups; seven three-call samples) at n=8/16/32 yields
256-bit 1.064/1.086x, 1.112/1.106x, 1.120/1.134x;
512-bit 1.072/1.072x, 1.104/1.087x, 1.114/1.173x.
These are factorization-only gains. All 296 core and 23 dense checks pass
(timing tests ignored). Full Ising ABBA against the accepted conversion-hoisting
baseline is 49.64043/48.86401/49.71158/49.90513s (1.0159/1.0039x).
All four external audits pass with identical x/s/z/status and 50 iterations;
frozen identities reverify. Reject: the full-solver improvements do not reach
the repeatable 2% retention threshold. No further sweep of this candidate.
Production source hashes are unchanged; preserve the accepted conversion fix.
Evidence: `/tmp/sdpx-rotation-20260917/`.
A separate next hypothesis is MPFR 4.2.2's native `mpfr_hypot`: the pinned
source uses three working values at guard precision and a rounding loop, so
allocation cost must be timed rather than assuming a faster single API call.
Its correctly rounded result differs from the current scaled multi-operation
formula; qualification must use accuracy/residual tests, not require old-bit
identity. No hypot change is included in this rotation experiment.

September 17 native MPFR hypot candidate (isolated, not adopted): replace the
provider's scaled multi-operation norm with the pinned MPFR 4.2.2 correctly
rounded `mpfr_hypot`, using the existing owned binary-call wrapper. Fixed
precision and RNDN are unchanged, but results can differ from the old formula.
306 bounded-exponent-gap cases across 128–2048 bits and exponents -4096/0/4096
match a certified rounding interval: exact promotion/square/sum at 8192 bits,
directed square roots, and agreement of both endpoints rounded to the target.
This oracle shares MPFR bindings but does not call hypot. Pythagorean triples,
large operand gaps, signed zeros and Inf/NaN semantics also pass.
Same-binary scalar ABBA (32 fixed pairs, 20 warm batches, seven 100-batch samples)
shows 128-bit 1.139/1.070x, 256-bit 1.427/1.437x, 512-bit 1.692/1.692x,
768-bit 1.883/1.876x, 1024-bit 2.117/2.150x, 2048-bit 2.738/2.736x.
These are scalar-only gains. All 296 core and 23 dense tests pass, including
explicit SVD reconstruction/left/right orthogonality at orders 8/16/32 across
six precisions and the ill-conditioned NT regression. Full-SVD ABBA at
256-bit n=8/16/32 is 1.039/1.057x, 1.036/1.053x, 1.023/1.010x;
512-bit is 1.081/1.083x, 1.068/1.051x, 1.022/1.069x. The scalar gain does
not translate proportionally to SVD. Full Ising ABBA is
51.81860/50.10608/50.59220/51.25506s (1.0342/1.0131x; aggregate arm medians
51.53683 -> 50.34914s). All four unchanged original-coordinate audits pass,
all statuses are optimal and all iteration counts are 50. Repeats within each
arm have identical points. Baseline/candidate scaled infinity differences are
x 2.35e-120, s 8.16e-120, z 5.02e-103; these compare trajectories, not an
independent accuracy oracle. All frozen identities reverify. Do not promote:
the aggregate ~2.4% speed gain is marginal and the separate pairs do not both
clear 2% on this unpinned host. No further local sweep; prioritize the known
sampled-operator constant overhead. The rejected rotation helper is absent.
Production source hashes are unchanged. Evidence: `/tmp/sdpx-hypot-20260917/`.
Static follow-up: `SampledBlockWorkspace::forward_terms` and `adjoint_terms`
still construct SQRT_2/FRAC_1_SQRT_2 inside element loops. They lie in the
measured forward/transpose direction paths. Test per-call constant hoisting
next, separately from hypot; retain both actual constant values and arithmetic
association, including off-diagonal multiplicities and shared-column ordering.

September 17 sampled-operator constant hoisting (accepted and integrated):
`forward_terms` computes SQRT_2 and FRAC_1_SQRT_2 once per call only when its
shape needs them; `adjoint_terms` similarly reuses FRAC_1_SQRT_2. Both constants
still use their independently correctly rounded implementation. Multiplication
association, authoritative factors and contribution order are unchanged; no
cross-call cache or precision-specific solver path is added. All 378 baseline
comparisons pass across Float64/six MPFR precisions, dimensions 1/2/3, basis
rows 1/3/8, columns 0/1/5 and two alpha values, for both forward and adjoint.
512-bit same-binary ABBA (dim 1/2, basis rows 8/16/32, 16 columns; two warmups,
seven three-call samples) yields forward 1.21–1.41x, adjoint 1.44–1.70x.
These are operator-only gains. All 296 core tests and 27 sampled
solver/integration/update tests pass, including pooled-operator cases. The
frozen-library Ising512 ABBA native times are 51.861781/48.923151/48.761441/
51.061305s: AB/BA speedups 1.0601/1.0472, medians 51.461543 -> 48.842296s
(5.09% less time). All four solves pass unchanged original-coordinate audits,
with identical x/s/z/status and 50 iterations. Float64 medium passes all eight
first/warm audits with identical returned points and 18 iterations. Warm AB/BA
ratios are 1.0653/1.0026; no regression observed, but do not claim a repeatable
Float64 gain on this unpinned host. Frozen source/library/input identities pass.
The exact qualified production change is integrated; root FFI rebuilt and a
256-bit sampled PSD4 default-loader smoke passes, exercising both off-diagonal
constant branches and external original-coordinate residuals. Production source
and library hashes reverify. Hypot/rotation/SYRK candidates remain absent.
Evidence: `/tmp/sdpx-sampled-constants-20260917/` (solver-summary.json,
integration-source.json, production-library.json and production-smoke.log).
Further static audit finds inner-loop constants in ordinary SDP's
`psd_entry`/sparse Schur contractions and dense Schur packing. Assess those on
an independent non-sampled SDP family after this qualification; the Ising
sampled path does not establish their end-to-end benefit. Setup materialization
and two isolated SOC/Jacobi calls are lower priorities until measured.

September 17 ordinary SDP coefficient-product constant screen (isolated):
`coefficient_product` hoists FRAC_1_SQRT_2 once when the cached plan contains
an off-diagonal coefficient; purely diagonal/empty plans skip its evaluation.
FMA order, duplicate-coordinate plan and storage layout are unchanged. Exact
baseline comparisons pass for 280 combinations: Float64 and all six MPFR
precisions, orders 0/1/2/8/16, diagonal/mixed/dense/empty coefficients, offsets
0/2 and padded leading dimensions. Frozen-source same-binary 512-bit ABBA
(seven samples of five calls after three warmups) gives 1.32–2.23x for mixed
and dense orders 8/16/32, with diagonal cases within approximately 1%.
These are kernel-only, unpinned-host results. Production remains unchanged;
do not infer whole-solver or Ising gains. Evidence:
`/tmp/sdpx-coefficient-constants-20260917/` (source hashes, reference helper,
screen log and summary). All 296 core tests pass (one timing test ignored).
Full non-sampled SDP screening uses the 16-order dense pencil Q*diag(x-1)*Q',
with a rational Householder Q constructed directly at 512 bits and known
optimum x=1. Fixed tolerance 1e-42, external residual/gap/solution and PSD
Gershgorin gates 1e-30, default preprocessing and single-thread condensed KKT.
Twelve solves (ABBA processes, first plus two warm fresh solves each) pass;
x/s/z/status/factorization and 24 iterations match exactly. Warm process medians
are 1.517095/1.302430/1.267460/1.312772s, AB/BA speedups 1.1648/1.0357.
The large between-process variation limits the speed estimate; no affinity was
enforced. Inputs, source and both library hashes reverify; production unchanged.
This exercises off-diagonal coefficient products, unlike the prior orthant
fixture's diagonal PSD coefficients. Float64 medium and a second MPFR precision
remain required before integration; do not claim broad SDP/Ising improvement.
Receipts: ordinary.jl, frozen-solve.json, solve-summary.json and receipt-*.json
in the same external experiment directory.

September 17 coefficient-product qualification extension (not promoted):
The same frozen libraries/input generator passed twelve 256-bit ordinary SDP
solves at the unchanged 1e-42 tolerance and 1e-30 external gates. All returned
points and 24 iterations match. Warm ABBA process medians are
0.594515/0.565323/0.569262/0.602496s (1.0516/1.0584x).
Float64 medium passes all eight first/warm audits with identical points and
18 iterations, but warm ABBA times are 2.948915/6.116076/4.791032/4.575879s.
Both pairs are slower (ratios 0.4822/0.9551), with substantial host variation.
A late process snapshot showed the benchmark as the only CPU-heavy numerical
process; it does not establish CPU affinity/frequency or explain the earlier
variation. Do not attribute this entirely to noise or claim no regression.
Production remains unchanged; no candidate adoption. Frozen identities reverify.
Receipts: extended-summary.json, frozen-extended.json, ordinary256-*/ and
medium-*/ under `/tmp/sdpx-coefficient-constants-20260917/`.
Next candidate should avoid the new dynamic plan scan for hardware Float64,
where the square-root constant is already compile-time constant. Keep one
coefficient-product implementation with a scalar-capability branch, then test
that revision separately; high-precision gains do not waive Float64 acceptance.

September 17 coefficient-product revision 2 (accepted and integrated):
The shared helper bypasses the new plan scan for hardware precision <=53 bits,
where FRAC_1_SQRT_2 is constant; higher precision retains conditional per-call
hoisting. Same arithmetic order, no new solver path/cache. All 280 exact
Float64/128–2048-bit comparisons and 296 core tests pass. Frozen ABBA whole
ordinary SDP runs at fixed 1e-42 tolerances and 1e-30 external gates pass all
24 solves at 256/512 bits, with identical x/s/z/status and 24 iterations.
512-bit warm medians: 1.410909/1.207939/1.159990/1.241474s (1.1680/1.0702x).
256-bit warm medians: 0.888700/0.840186/0.862791/0.886167s (1.0577/1.0271x).
Float64 medium passes all eight audits, exact returned points and 18 iterations;
warm 4.621607/4.579073/4.524019/4.549596s (1.0093/1.0057x), essentially flat.
Host is unpinned; these gains are fixture-specific, not broad library parity.
Do not erase v1's regression or attribute it conclusively to the scan.
Integrated the exact qualified production helper, rebuilt default FFI, and passed
a 256-bit dense-coefficient SDP4 default-loader smoke with original-coordinate
checks and known primal/dual optimum. Source/library identities reverified.
Evidence: `/tmp/sdpx-coefficient-v2-20260917/` (screen/solve/extended summaries,
integration-source.json, production-library.json, production-smoke.log).
Next: independently qualify sparse Schur contractions' repeated SQRT_2, or
profile the new baseline before a MPLAPACK SVD/provider experiment. Neither the
ordinary dense pencil nor sampled Ising establishes sparse-path speed benefit.

September 17 sparse Schur constant screen (isolated, not promoted):
A first candidate computes SQRT_2 once per column-pair contraction after checking
whether a diagonal/off-diagonal combination exists; Float64 skips classification.
All 756 baseline comparisons pass (Float64/all six MPFR precisions, orders
1/3/8, diagonal/off-diagonal/mixed columns, lengths 0/1/5/16). Same-binary
512-bit ABBA shows approximately 1.76–1.98x on mixed/mixed lengths 4–32 and
2.92–3.52x on diagonal/off-diagonal lengths 4–32. Single-entry off/off is
4–7% slower; several pure-category samples also regress or vary. Nanosecond
small cases are noisy, but do not dismiss the extra classification overhead.
No full-solver gain or production adoption is claimed. Evidence:
`/tmp/sdpx-sparse-constant-20260917/` (frozen source, screen-summary.json,
core-tests.log and status.json). Revise toward once-per-block assembly constant
preparation shared with parallel sparse lanes, removing per-pair scans before
whole-solver qualification. Use a sparse invertible coefficient pencil with
mixed diagonal/off-diagonal support; the dense Householder pencil from the
previous experiment does not establish sparse-path benefit. Keep the accepted
dense-coefficient and sampled-operator changes unchanged.

September 17 sparse Schur revision 2 (isolated, not adopted):
Replace per-pair classification by one immutable SQRT_2 owned by PsdBlock,
initialized at that block's fixed precision (zero placeholder for orders <=1,
where mixed coordinates cannot occur). Serial contractions and parallel sparse
lanes receive this same scalar. No global cache or mutable shared scratch.
All 756 exact baseline comparisons and 296 core tests pass, including existing
parallel assembly/update coverage. Same-binary 512-bit ABBA shows 3.23–3.51x
for diagonal/off-diagonal pairs (including one-entry pairs at 3.42/3.47x),
1.78–2.22x for mixed/mixed lengths 4–32. Pure diagonal/off-diagonal categories
are roughly flat (small fluctuations up to about 2%); no per-pair scan remains.
The screen excludes one-time block setup and is not whole-solver evidence.
Production remains unchanged. Source and screen identities reverify.
Evidence: `/tmp/sdpx-sparse-v2-20260917/` (candidate.patch, screen-summary.json,
core-tests.log, status.json). Next full fixture is prepared as sparse.jl:
32-order Q*diag(x-1)*Q' using a sparse strictly diagonally dominant Q, known
optimum x=1 and dual identity. Verify actual post-preprocessing sparse-path
coverage and unchanged original-coordinate gates before timing/qualification.
The script is prepared but not executed; full solver/Float64 regression pending.

September 17 sparse Schur revision 2 full-solver screen (still isolated):
The frozen 32-order sparse pencil ran with default Ruiz/presolve/chordal enabled.
Verbose receipts show chordal expansion from 32 to 296 variables, 718 rows,
1008 A entries and five PSD cones with svec lengths 78/406/78/78/78. Thus do
not describe the timed problem as an untouched single 32-order block. The
transformation introduces sparse separator columns, but exact per-block sparse
contraction counts were not instrumented; no phase-level attribution is claimed.
All twelve 512-bit solves pass fixed 1e-42 tolerances and 1e-30 external
original-coordinate residual/gap/known-optimum/PSD gates. Returned x/s/z/status,
factorization and 23 iterations match exactly. ABBA warm process medians are
18.311634/17.512598/17.539860/17.934535s (1.0456/1.0225x). The candidate's
whole-solver gain is much smaller than its kernel speedup. Source/input/library
identities reverify; RSS is recorded and host affinity was not enforced.
Evidence: `/tmp/sdpx-sparse-v2-20260917/` (frozen-solve.json, solve-summary.json,
solve-*.log/json and receipt-*.json). Production is unchanged. Complete one
second-precision leg and Float64 medium before adoption; use one first plus one
warm solve per subsequent process to keep development time bounded rather than
repeating this ~4-minute 512-bit screen. Keep the 120-second process timeout.

September 17 sparse Schur revision 2 (accepted and integrated):
Additional ABBA runs pass eight 256-bit sparse-pencil solves and eight Float64
medium solves with exact x/s/z (or medium's original-coordinate points), statuses
and iteration counts across arms. 256-bit warm times are
8.289261/8.059643/12.671504/12.721206s (1.0285/1.0039x): substantial host drift,
no repeatable >=2% gain at this precision. Float64 medium is essentially flat:
4.557563/4.526983/4.557238/4.580696s (1.0068/1.0051x), 18 iterations. Retain
for the preceding repeated 512-bit >2% gains, without extending that claim to
256-bit or Float64. All frozen identities pass; 296 core tests and 756 exact
kernel comparisons already cover the unchanged candidate. Integrated the exact
qualified helper, rebuilt default FFI, and verified its Julia default loader.
The new SDP8 loader smoke initially contained an invalid assertion requiring
the particular dual identity solution, although dual optima are nonunique.
Preserve that failed script/log. Replaced it with unchanged-tolerance external
stationarity/gap and PSD checks; both frozen baseline and production pass with
the same complete point hash and 16 iterations. Dual Gershgorin lower bound is
0.578875 (>0), independently establishing PSD here. No solver tolerance or
acceptance rule was relaxed. Production source/library identities reverify.
Evidence: `/tmp/sdpx-sparse-v2-20260917/` (extended-summary.json,
integration-source.json, production-library.json, both smoke logs and retained
invalid-dual-oracle log). This closes the local constant-hoisting candidate,
not provider replacement, large-scale accuracy or distributed qualification.

September 17 MPLAPACK provider experiment (build/smoke only):
Public upstream cloned to `/tmp/sdpx-mplapack-20260917/source`, frozen clean
commit ddf3b4ba8ed65fa101245c472aecb8112ff32920. Built temporary MPC 1.3.1
from the cached gmp-mpfr-sys source against the exact existing SDPX MPFR/GMP
static libraries; project dependencies and production backend are unchanged.
Configured CMake for MPFR only and built mplapack_mpfr_opt (1061 translation
units); all other precision/GPU backends disabled. OpenMP is unavailable in
this Apple Clang configuration: single-thread evidence only, no scaling claim.
The standalone full-U/full-VT Rgesvd driver compiled/linked successfully.
Four deterministic rational-matrix cases (256/512 bits, orders 8/16) pass
independent Julia audits at twice the working precision: finite descending
nonnegative singular values, relative reconstruction and U/VT orthogonality,
threshold 1000*n*2^(1-bits). Input/output conversion is outside timing; two
warmups and seven calls are recorded, but these are not an SDPX comparison.
Do not infer a speedup from raw timings, which already show host/warmup drift.
Identity/build/test receipts: identity.json, smoke-summary.json, status.json,
configure/build logs and per-case audit logs in that directory. Next: current
Rust provider on the exact same rounded inputs and matched ABBA scheduling,
then ill-conditioned singular-value/NT regression and actual PSD shapes before
any FFI/provider integration. Current smoke does not establish robust SVD or
whole-solver equivalence. Public build documentation:
https://github.com/nakatamaho/mplapack/blob/ddf3b4ba8ed65fa101245c472aecb8112ff32920/README.cmake.md

September 17 MPLAPACK vs current Rust SVD screen (no provider change):
Frozen current Rust provider and native MPLAPACK use identical rounded rational
inputs, full U/VT, query-sized reused workspace, two warmups/seven measurements
per process and sequential ABBA scheduling. Input reset/output conversion are
outside timing; no affinity is enforced. Independent twice-precision audits
verify exact cross-provider input equality and reconstruction/orthogonality for
all 56 output sets: Rgesvd at 256/512 bits and orders 8/16/32/64 (32 sets),
Rgesdd at both precisions and orders 16/32/64 (24 sets). Tests pass; this does
not replace conditioning or whole-solver acceptance.
For Rgesvd, current Rust is faster throughout this screen. On the less noisy
16–64 orders Rust/native median ratios are 0.660–0.860 (roughly 14–34% less
kernel time). The 256-bit order8 runs show marked host drift; no extra claim.
Rgesdd improves the comparison only locally: 512-bit order32 Rust/native ratios
1.1797/1.1475, while 256-bit order32 is only 1.0199/1.0187 and orders16/64
remain slower than Rust. Do not select a production backend from one matrix
family/order or assume larger matrices necessarily favor the native library.
Evidence: `/tmp/sdpx-svd-compare-20260917/` (summary.json, summary-large.json,
summary-dd.json, audit logs, frozen identities, provider-identity.json).
Next capture representative actual PSD scaling matrices and test ill-conditioned
singular values before considering a provider/algorithm switch. Production
source/dependencies remain unchanged; no whole-solver speedup is established.

September 17 SVD conditioning screen (no production change):
Reproduce the existing graded NT fixture at each of 128/256/512/768/1024/2048
bits: build SPD S/Z with delta=2^(-5*bits/8), apply the actual svec round-trip
arithmetic, use current MPFR Cholesky/GEMM to construct Lz'*Ls, and export the
rounded matrix. Compare current Rust SVD, MPLAPACK Rgesvd and Rgesdd on exactly
that input. Also embed the block in identity32 at 256/512 bits to exercise a
larger divided-and-conquer call. All 24 cases pass independent twice-precision
checks: input equality, finite positive sorted singular values, reconstruction,
U/VT orthogonality and smallest-singular-value relative error <2^(-bits/4).
The small-value reference uses the exact rounded 2x2 input determinant and the
stable formula sigma_min=abs(det)/sigma_max, not a working-precision Gram solve.
The largest error/gate ratio is 1.319e-17 at 128 bits; other precisions have
larger margins. This does not qualify native NT R/Rinv integration or arbitrary
conditioning families. Runtime tolerances/providers are unchanged.
Evidence: `/tmp/sdpx-svd-conditioning-20260917/` (frozen.json, build logs,
summary.json, audit-summary.json and input/output files). First harness build
failed on a missing Scalar trait import; corrected locally, preserving its log.
Next: capture a small bounded set of real Ising scaling matrices at different
iteration stages and compare those before any provider switch. Synthetic
order32 speedups alone do not justify selecting a production dispatch threshold.

September 17 actual Ising SVD capture/replay (no production change):
Frozen production and isolated capture FFI each completed one full Ising512 solve:
unchanged external audits pass, identical x/s/z/status and 50 iterations. The
capture hook reads only nine predetermined non-query SVD calls (1/2/16/32/64/
128/256/384/512), exports rounded input values, and is absent from production.
These are call indices, not independently recorded iteration labels. Actual
captured PSD orders are 12/13/15, so the synthetic order32 Rgesdd advantage does
not describe this sample. No capture-run timing is a performance claim.
Replay uses the uninstrumented frozen Rust provider, not the diagnostic routine,
plus native Rgesvd and Rgesdd in R/V/D/D/V/R order, each with two warmups and
seven timed calls. All 54 outputs pass independent twice-precision reconstruction
and orthogonality audits, cross-provider input equality and positive-singular-
value checks; cross-provider singular-value differences are recorded separately
and are not an independent oracle. Rust/native timing ratios range approximately
0.576–0.793 for Rgesvd and 0.582–0.784 for Rgesdd: Rust uses roughly 21–42%
less kernel time throughout this sample. No affinity is enforced. Consequently
do not pursue a blanket MPLAPACK SVD replacement for this Ising family. This
says nothing about other routines, larger orders or distributed backends.
Evidence: `/tmp/sdpx-svd-capture-20260917/` (capture-summary.json,
replay-summary.json, singular-comparison.json, frozen identities and audit logs).
Next prioritize remaining direction/residual costs or evaluate other provider
kernels with measured relevance; retain actual SVD inputs for future algorithm
work instead of repeatedly rerunning the full Ising solve. Production unchanged.

September 17 refreshed current Ising512 profile (diagnostic only):
Frozen current production includes the accepted svec, sampled, coefficient and
sparse-Schur constant changes. Instrumentation-only deltas were recovered from
hash-matched prior sources and applied with zero fuzz to the new snapshot;
no old numerical source replaced current code. All 37 stage probes live outside
the repository. Verified baseline library SHA256
5d49d2a311e01f12bc989a501f1a5f1d3553396998ff20cbb34cb8251c980d77.
Sequential baseline/profile processes each run first plus one warm solve, with
240-second process caps, one thread and unchanged 512-bit/audit settings.
All four external audits pass, 50 iterations and bit-identical x/s/z/status.
Frozen files and production source identities pass after measurement.
Baseline first/warm native times 29.172154/29.184779 seconds; diagnostic
29.014780/28.966002 seconds. Unpinned host: these are phase measurements,
not a speedup claim and not comparable with older sessions' absolute times.
Warm profile: affine+combined direction solves 10.805392 seconds (37.3%);
cone scaling 6.421652 (22.2%), KKT update 5.255722 (18.1%), and bounds
4.523094 (15.6%). Nested SVD 5.720260 (19.7%), bidiagonal SVD iteration
4.647028 (16.0%), congruence 4.331263 (15.0%), its GEMMs 4.135109 (14.3%),
outer residual 4.031330 (13.9%), bound eigenvalues 2.464255 (8.5%).
Do not sum nested stages with their parents. Cholesky 0.145308 (~0.5%) and
Schur assembly 0.136748 (~0.5%) are low priorities for this particular sample;
large-matrix Gram/FLINT and distributed hypotheses remain separate workloads.
Next short screen: hoist immutable precision constants and sqrt(2) from the
bidiagonal SVD loop/shift routine without changing operation association,
deflation or convergence criteria. Replay captured actual matrices first,
including tiny-singular-value conditioning checks; only a repeatable kernel
benefit justifies a whole-solve candidate. Direction/congruence GEMMs remain
the next substantial provider target, ahead of MPLAPACK Cholesky here.
Evidence: /tmp/sdpx-refreshed-phases-20260917/ (instrumentation.patch,
instrumentation-origins.json, source-before.json, frozen-run.json, build.log,
summary.json and four original-coordinate audits). Production numerical source
and dependencies unchanged; no backend promoted.

September 17 SVD constant-hoisting screen rejected:
External provider-only baseline/candidate harnesses freeze current source and
reuse the nine actual Ising SVD inputs. Candidate computes epsilon, sqrt(epsilon)
and sqrt(2) once per bidiagonal decomposition and passes sqrt(2) into small_shift;
no reassociation, tolerance or deflation changes. Sequential ABBA, two warmups
and seven measured calls per process, 30-second process caps. All 36 actual-input
outputs are byte-identical across arms. Eight graded conditioning inputs add
16 exact baseline/candidate outputs: 2x2 at all six 128–2048-bit precisions and
embedded order32 at 256/512. All 52 outputs independently pass reconstruction,
orthogonality and (for conditioning cases) tiny-singular-value relative checks
at twice working precision. Frozen production source remains unchanged.
Actual-input baseline/candidate timing ratios are 0.981–1.006 across the 18
paired comparisons: no repeatable >=2% benefit; most slightly favor baseline.
Reject the candidate without spending more full-solver timing. No production
change. Do not infer an unmeasured code-generation or branching cause.
The runner's final identity check mistakenly included its actively written
run.log. Its failure is retained; finalize.py verifies every other frozen file,
all completed per-process logs, exact outputs and all 52 independent audit
records, then reconstructs summary.json without rerunning timing. This is a
harness bookkeeping failure, not a numerical failure; no check was weakened.
Evidence: /tmp/sdpx-svd-constants-20260917/ (setup.py, frozen.json, build.log,
run.log, finalize.py, results, audit-captures.log, audit-conditioning.log,
summary.json). Next investigate actual congruence GEMM shapes/data reuse or
bound-eigen kernels; do not repeat this unchanged constant candidate.

September 17 congruence/GEMM provider screen (no backend change):
Source inspection rules out simply omitting the second GEMM's lower triangle:
mat_to_svec deliberately combines upper and lower rounded entries. Replacing
both with one triangle is not an operation-preserving optimization and requires
its own numerical analysis, not an assumed symmetry shortcut.
Frozen current MPFR GEMM versus the existing optimized MPLAPACK Rgemm build:
256/512 bits, square orders12/15/32, NN/NT/TN, alpha=1/beta=0, bounded rational
inputs constructed at working precision. Sequential Rust/native/native/Rust,
two warmups plus seven measurements each, 30-second process caps. All72outputs
pass twice-precision multiplication audits and exact cross-provider input checks.
The Rust and native libmpfr.a/libgmp.a hashes match. Source and benchmark input/
binary identities verify unchanged. Both kernels use declared working precision;
accumulation implementations can differ, so output equality is not required.
Across36paired comparisons Rust/native median ratios0.466–0.811: Rust uses
about19–53%less kernel time throughout this screen. No native GEMM replacement
is justified for these shapes. Synthetic square inputs, unpinned CPU, no setup/
conversion timing: do not infer whole-Ising speedup or performance of large
Schur products from this screen. Current production remains unchanged.
Evidence: /tmp/sdpx-gemm-provider-20260917/ (frozen.json, source-before.json,
provider-identity.json, build.log, native-build.log, run.log, audit.log,
summary.json and72outputs). Next review reusable intermediate products in
scaling/direction operations against mature implementations; avoid repeating
unchanged MPLAPACK small-GEMM or constant-hoisting experiments.

September 17 cached-Hessian action rejected on conditioning:
Reviewed local Clarabel PSD mul_Hs: it retains W then W-transpose, four GEMMs;
current SDPX follows that factored action. External candidate reuses upper-
authoritative G=R*R-transpose, mirrors it into existing scratch, and applies
G*X*G using two GEMMs, one implementation for every precision. The existing
295 release core tests pass, but a new analytic directional regression rejects
this candidate before any whole-solver timing.
Use R=[[1,0],[1,t]], t=epsilon(T), and authoritative smat(x)=a*[[1,-1],[-1,1]],
a=1/sqrt(2) rounded at working precision. The factored response's bottom-right
entry is a*t^4 and is representable in every tested type. Rounded G loses t^2
from 1+t^2, so the proposed action returns zero instead. All7precisions
(Float64 and128/256/512/768/1024/2048) reproduce relative error1, while the
factored reference passes the analytic relative bound. This is a directional
conditioning test; an absolute error gate scaled by max(1,norm(y)) would hide it.
Do not promote this candidate even though ordinary core tests pass. Add the
seven analytic checks to production psd_hessian_tests.rs; runtime source remains
unchanged. Evidence: /tmp/sdpx-hessian-action-20260917/ (source-before.json,
setup.py, core-tests.log, graded-test.rs, graded-tests.log, numerical-source-
verification.json). All seven production graded tests pass (release, both
Accelerate and Faer features); see production-graded-tests.log.
Next priority: the existing high-precision condensed PsdBlock::apply already
uses a similar G/Ginv reassociation. This counterexample identifies a risk,
not yet a demonstrated failure of a complete solve. Reproduce it specifically
at that operator boundary and assess a conditioning-safe factored fallback;
do not extend the shortcut or infer an Ising failure without evidence.
Reference: https://docs.rs/clarabel/latest/src/clarabel/solver/core/cones/psdtrianglecone.rs.html

September 17 condensed fixed-precision conditioning repair (accepted correctness):
Reproduced the analytic counterexample at the actual PsdBlock::apply boundary,
using consistent R/Rinv pairs, both forward and inverse actions. Float64 passes;
all six MPFR precisions lose the tiny response on the existing two-GEMM Gram
shortcut (12failed action checks). This is an operator-level correctness failure,
not a claim that previous Ising results failed their original-coordinate audits.
Restore the existing four-product factorized action at every precision. Delete
the precision-specific fast path, the unused G cache and its update work, and
avoid forming Ginv for sampled blocks that no longer consume it. Ordinary Schur
assembly still uses its required Ginv and existing regularization/refinement;
this repair does not establish that every ill-conditioned Schur problem is solved.
Seven new regression tests exercise14directional cases including Float64. All
309release core tests pass, including serial/pooled paths and the prior seven
PSD Hessian regressions. One implementation now serves all precisions.
Frozen same-session sequential baseline/candidate/candidate/baseline Ising512
acceptance: all four original-coordinate audits pass,50iterations each. First
solve native median28.979713→31.226579seconds (+7.75%); process caps240seconds,
one thread, no affinity. Retain as a correctness repair, explicitly not a speedup.
Source/dependency/input/library identities pass; exact point identity is not
required for the changed arithmetic association. Production source now equals
the tested candidate; matching FFI library installed from the frozen build.
Evidence: /tmp/sdpx-condensed-conditioning-20260917/ (baseline-tests.log,
candidate-tests.log, candidate-build.log, frozen-run.json, summary.json,
source-before.json, integrate.py, production-library.json). Default-loader
256-bit ordinary-SDP smoke passes: known primal optimum, PSD dual bound,
stationarity, feasibility and gap at unchanged external tolerance;16iterations.
See production-smoke.log; production-library.json identifies the current DLL.
Use this repaired baseline for all further speed claims. Existing unmatched
older benchmark numbers do not describe the repaired engine. Remaining broad
accuracy/large-instance and cluster publication/scaling work stays open.

September 17 Sturm-cache screen rejected for whole-solve performance:
Confirmed step bounds already request eigenvalues only and indexed minimum at
high precision; there is no unnecessary eigenvector construction to remove.
Candidate caches the immutable tridiagonal tiny-pivot scale once and reuses
unchanged endpoint Sturm counts during isolation. Existing RQI safeguards,
final counts, 32-step budgets and QL fallback remain unchanged. No new tolerance,
precision change or cache surviving a call. All309core tests pass.216comparisons
at all six MPFR precisions, orders1/2/3/12/15/32, repeated/random spectra and
first/middle/last indices preserve exact Option eigenvalues and both iteration
counters. Same-binary512-bit kernel ABBA speedups: order12 1.332/1.287,
order15 1.248/1.246, order32 1.378/1.386. Kernel screen includes10calls/sample,
two warmups and five measured samples per arm; synthetic tridiagonals only.
Full repaired-baseline Ising512 ABBA:53.385811/50.970108/51.989464/51.231799s.
All4original-coordinate audits pass,50iterations and exactly equal x/s/z/status.
Paired speedups1.0474/0.9854; median52.308805→51.479786s is only1.58%lower
and not repeatable >=2%. Reject production integration; do not repeat unchanged.
No affinity; do not compare these absolute times with earlier sessions. Source,
inputs and binaries verify unchanged, and baseline DLL matches the accepted
condensed-conditioning repair. Production source/library remain that repaired
baseline. Evidence: /tmp/sdpx-sturm-cache-20260917/ (source-before.json,
core-tests.log, screen-mpfr.rs, screen.log, screen-summary.json, frozen-run.json,
ffi-build.log, summary.json and4raw/audited outputs). integrate.py was prepared
with the paired threshold assertion but was not run.
Next inspect availability of the larger Ising accuracy fixture and use a bounded
screen of the repaired factorized path against its prior failure. This will
inform accuracy work before further small-kernel optimization or multicore claims.

September 17 larger Lambda11 saved-point scaling replay (no full solve):
Old qualified input and iteration94point are locally available. Old768-bit,
8-core solve used1287.85native seconds and failed the unchanged external audit;
a fresh long solve is deferred until a bounded diagnostic supplies a hypothesis.
At768bits, compile the unchanged sampled input, verify SHA
bb1fa49da0d461ebba2b9539412222e5dc134553dfad8f7bc128b23acf61ed1d, and extract
original-coordinate s/z into28PSD blocks of orders36–43. Record raw-point,
fixture and current repaired source/binary hashes. This is not a replay of
internal equilibrated solver state, previous step directions or the full solver.
Current direct-SVD NT scaling succeeds on all28blocks; R/Rinv and84actions
are finite. Probe each block with z, s and deterministic signed rational entries.
Compare factorized Hs action with cached-G congruence and measure Hs*z versus s.
Largest norm-relative action discrepancy2.200892e-167; largest NT identity
residual2.314896e-167, both in block0/order36 with z. These norms do not bound
all near-null directions and do not undo the analytic regression requiring the
factorized path. Nor do they attribute the original external~2.16e-22 residual
to scaling: no comparable scaling defect appears at this saved point.
Diagnostic runtime7.8459seconds, peak process RSS25,198,592bytes,180-second cap.
All frozen source/input/binary identities pass; production unchanged. Initial
harness compilation lacked concrete num_traits imports; corrected, preserving
build-initial.log. Evidence: /tmp/sdpx-lambda11-scaling-20260917/ (export.jl,
input.json, source-before.json, frozen.json, replay.log, receipt.json,
summary.json). This is new conditioning evidence, not new Lambda11 acceptance,
performance credit or proof that the recent repair fixes its old failed solve.
Next focus on the recorded gap between global normalized runtime residuals and
component-relative sampled audits at enormous primal multipliers. Any exact
model transformation or stricter-tolerance study must preserve original external
gates and be labeled separately from the existing matched1e-42protocol; no
retired independent runtime certificate or pointwise tolerance guarantee.

September 17 exact variable-unit diagnostic (new hypothesis, not integration):
Evaluate x=D*xhat, Ahat=A*D, qhat=D*q, unchanged b/s/z on the saved Lambda11
point at768bits. Every D diagonal is a positive power of two derived from input
coefficients, with no approximate rank change. Four choices: identity, uniform
2^256, inverse column-norm exponent, inverse abs(q) exponent (zero q falls back
to its A column norm). Recomputed sparse products verify exact Ahat*xhat=A*x,
Ahat-transpose*z=D*(A-transpose*z), inverse variable mapping and objective dot
product. All coefficients remain finite at working precision. Existing global
feasibility formulas and1e-42tolerance are unchanged; norm reductions in this
Julia diagnostic model the formula and are not a bitwise Rust runtime replay.
Identity: dual6.4426e-142, passes. Uniform256: dual3.9533e-64, still passes.
Column-norm scaling: dual7.7598e-122, still passes. Objective-based powers
(exponents -3..305): dual1.737148e-24, rejects this previously accepted point;
primal1.4172e-89 still passes. norm(xhat)8.0080e26, norm(qhat)1.9987 versus
original norm(x)8.1775e78. This exposes one failure without adding a runtime
certificate, changing a convergence test or tightening its tolerance.
Important cost: maxabs(Ahat)=7.6339e67 versus10.1398original; objective-based
units may damage factorization conditioning. This single saved-point result is
not a guarantee of componentwise accuracy, better convergence or throughput.
Do not set new defaults or tune exponents against the failed point. Next qualify
exact sampled-factor scaling and bounded factorization/solve screens, followed
by independent LP/SOCP/SDP holdouts before any production policy. Full Lambda11
and matching SDPB acceptance remain outstanding.
Evidence: /tmp/sdpx-lambda11-units-20260917/ (check.jl, frozen.json,
summary.json, receipt.json);24.19seconds process wall, peak RSS2,299,543,552bytes,
120-second cap. No new solve or production changes. Original input SHA and
saved-point SHA are recorded; unchanged helper/runtime formula hashes verified.

September 17 objective-unit cross-family/representation qualification:
Use catalog-verified exposed development fixtures LP_afiro, SOCP_sambal and
SDP_truss1; no reserved holdout consumed. The accepted repaired DLL and copied
Julia frontend are frozen. Baseline/objective-power units each solve all3cases
at Float64 (internal1e-8, external1e-6) and256bits (internal1e-42, external1e-30),
with defaults on, one thread,200iterations,20-second native/40-second process
caps. Map x back before checking returned-slack feasibility, primal/dual cones,
stationarity and objective gap. P is confirmed zero; this experiment does not
claim QP support. Objective dot products map exactly at returned points.
All12effective cases pass. Iterations baseline→units: Float64 LP8→7, SOCP11→11,
SDP11→11;256-bit LP25→24, SOCP46→46, SDP49→49. Exponents only -3..2 here;
these small fixtures do not establish stability at Lambda11's -3..305 range.
One sample per arm, no interleaving or per-process memory study: no performance
or memory claim. Two original256-bit SDP audit executions failed because Julia
BigFloat eigmin requested an unsupported indexed eigvals method. Keep those logs;
audit v2 uses minimum(eigvals) at the same precision/gate. Only the two affected
cases were rerun;10unaffected receipts reused,14total solver invocations. No
numerical threshold weakened. Evidence: /tmp/sdpx-units-screen-20260917/
(frozen.json, original summary/results, check-v2.jl, repair/, final-summary.json).

Separately qualify Lambda11's Julia factor-authoritative representation: multiply
linear CSC columns and each sampled block's weights by the same exact powers,
leaving bases/rows/cones unchanged. At768bits, all28blocks and1099variables retain
exact forward and adjoint products under the coordinate map for the saved point,
deterministic signed probes and coordinate probes (three pairs). No approximate
factor replacement or dense PSD input introduced. This tests Julia reference
products only, not the Rust operator, KKT factorization or fresh solve. Runtime
21.52seconds, peak RSS2,259,828,736bytes,120-second cap; all frozen helper/data
identities pass. Evidence: /tmp/sdpx-units-factors-20260917/ (factor-products.jl,
check.jl, frozen.json, summary.json, receipt.json). Production unchanged.
The bounded native screen has now completed (768bits, one thread, default
preprocessing, max_iter1, native30s/outer90s limits). Both processes exited0;
44 frozen file identities revalidate unchanged. Baseline reports Solved at
iteration0 in13.4702 native seconds, but original-coordinate infinity residuals
are primal46.0155 and dual92.2868. Its returned primal vector has 2-norm2.0301e78;
the internal normalized residuals are only5.5995e-76/2.5994e-76. This is not an
accurate solve or a performance result. The convergence/is_solved bodies match
the local Clarabel reference exactly, including the kappa/tau<=1 condition;
do not replace those criteria with the retired independent runtime gates.

Objective units terminate at iteration_limit1 in34.7785 native seconds, with
original infinity residuals52534.4/32.6660 and internal dual residual1.5709e37.
The native time limit is checked at solver boundaries, not a hard process cap;
the90-second process cap was respected. Process peak RSS is3,192,750,080 and
3,525,443,584bytes respectively, including Julia and external diagnostics.
Neither arm earns accuracy or speed credit. Do not promote objective scaling:
it avoids this immediate Solved outcome but has not established conditioning
or convergence. Preserve the earlier path-resolution failures and both points.
Evidence: /tmp/sdpx-units-native-20260917/ (summary.json, review.json, frozen.json,
logs and raw points). Production remains unchanged. Next isolate initialization
and input scaling against the authoritative sampled operator before any long
Lambda11 or backend performance campaign; retain original-coordinate audits.

External residual replay isolates the iteration0 anomaly further. At768bits,
recompute both saved returned points using the Julia sampled-factor operator
and separately the original CSC matrix. Baseline forward/adjoint infinity
differences are1.55e-174/1.73e-228; objective-unit returned point differences
are1.54e-171/2.89e-229. Both representations reproduce the large absolute
residuals. Baseline factor-normalized residuals reproduce the native reported
5.5995e-76/2.5994e-76 to rounding accuracy, while primal/dual costs both round
to approximately-61.0083. Thus the observed outcome is explained by the large
norm denominator and tiny objective gap, not a CSC-versus-factor discrepancy
at these points. This is an external Julia replay, not general native-operator
qualification. Candidate residuals in this replay use original variable units
and must not be compared directly with its transformed native residuals.
Runtime11.68seconds,90-second cap, frozen inputs/helpers unchanged; no solve or
production change. Evidence: /tmp/sdpx-initial-residual-20260917/.

Code review: symmetric default_start solves two KKT right-hand sides when P=0,
then shifts s/z into cone interiors and sets tau=kappa=1. The existing unit
initializer instead sets x=0 and cone unit s/z. Next isolated hypothesis:
screen this existing unit initializer for the symmetric sampled problem to
avoid the huge KKT-derived starting x, without changing stopping criteria,
precision or the operator. A successful first step is insufficient: require
bounded convergence plus original-coordinate acceptance and cross-family
regression before considering any default initialization policy.

Unit-start candidate screened in /tmp/sdpx-unit-start-20260917/: frozen current
source, one isolated change replacing default_start with the existing unit
initializer for all cones. Release core tests pass308/309. The dependent-equality
bordered-refinement regression returns PrimalInfeasible but fails its external
A' z ray-residual requirement (1e-8 times ||z||inf). The exact failing test
reproduces twice in the same compiled binary. This is a QP with a PSD block and
dependent equalities; it rules out unconditional unit initialization as a general
default, not every possible LP-specific initialization policy. Preserve this
regression and its gate. Build driver stopped before FFI construction, so there
is no Lambda11 result for this candidate. All production source files still
match the pre-experiment snapshot. Candidate patch, build/test logs, repeated
failure logs and binary identity are retained with decision.json. Next study
an initialization strategy specific to P=0, with this QP path unchanged, and
qualify it on the full affected family before changing defaults.

P=0-only initialization screen: /tmp/sdpx-linear-unit-start-20260917/ changes
only solve_initial_point's linear-objective branch to zero x/s/z before the
existing symmetric cone-interior shift. QP initialization and stopping rules
are unchanged; initial identity KKT factorization is still performed. All309
release core tests pass and FFI builds. The verbose Lambda11 run exposes an
existing formatter panic: info_print.rs assumes LowerExp always contains 'e',
whereas MpFloat delegates to Display (zero/nonfinite strings can lack 'e').
Original failure scripts/logs/receipts are preserved in verbose-failure/.

With only verbose disabled, the same candidate completes one iteration at
768bits/one thread, status iteration_limit, finite point, native16.4632seconds,
process26.3773seconds, RSS3,814,473,728bytes. Original infinity residuals are
primal4.18362 and dual82.57425; relative gap1.99865. Internal residuals again
become tiny (7.19e-76/8.99e-75), so avoiding iteration0 Solved has not repaired
the normalization problem. No accuracy or speed credit and no production
integration. Next fix the independently exposed diagnostic formatter, then
use a bounded multi-iteration screen to test whether this initializer provides
real convergence rather than merely delaying premature termination. Frozen
identities passed; preserve both failed and successful process receipts.

Follow-up: fixed the independent logging defect in production source by
preserving exponent-free MPFR displays in _exp_str_reformat. Two focused release
tests pass: real MpFloat768 formatting for signed zero/Inf/NaN and existing
exponent sign/padding behavior. Initial test compilation used a private parsing
helper; corrected to public FromStr, retaining both compile logs at
/tmp/sdpx-formatting-tests[-retry]-20260917.log. No numerical behavior changes;
the installed FFI library has not yet been rebuilt with this display-only fix.

The frozen P=0 initialization candidate (still verbose=false and independent of
the formatting patch) was screened at max_iter6/native60s/outer90s,768bits and
one thread. It exceeded the outer cap and was terminated after90.1845seconds;
process RSS3,752,624,128bytes. No result/point was returned, so neither iteration
count nor residual progress is established. Preserve the timeout as a failed
screen, not a solve-time estimate. Source/environment identities pass after
termination. Evidence: /tmp/sdpx-linear-unit-multistep-20260917/ (summary.json,
frozen.json, scripts and logs). Do not extend this unchanged long run: first
add the qualified display fix to the isolated candidate and obtain bounded
iteration-stage diagnostics. Initialization remains experimental and unmerged.

Bounded stage diagnostic completed in /tmp/sdpx-unit-stage-20260917/. Same frozen
P=0 candidate plus the qualified display fix and external-only elapsed probes;
768bits/one thread, max_iter6/native60s/outer90s. Verbose iteration0 now prints
without panic. Four steps complete before the90.1073s outer timeout; last event
is scaling_begin at iteration4. No returned point or full-accuracy credit.
RSS3,849,601,024bytes; all frozen identities pass. The log establishes forward
progress, not a stuck factorization. Completed cone scalings after first step
take3.60–3.72s; KKT updates for steps2–4 take6.11/2.95/6.20s, affine direction
spans3.58/3.60/6.77s, affine bound spans~1.10s, residual products~0.65s.
These are instrumented, unpinned observations, not interleaved speed evidence;
stage counters omit combined-solve/bound separation. Relative gap progresses
2.00,2.35,5.96,0.453 through steps1–4, while normalized residuals remain tiny.
Unit initialization postpones the incorrect immediate acceptance but has not
demonstrated convergence or solved original-coordinate accuracy. Do not keep
repeating this screen unchanged. Review coefficient/variable-unit conditioning
and mature initialization strategies before another full-size run; preserve
the existing test gates. Receipt, parsed stages.json, source and raw logs remain
outside the repository. Production initialization is still unchanged.

Local reference review changes the next hypothesis. Hypatia initializes cone
points and obtains x from QR/LSQR least squares (Solvers/process.jl), rather
than using an all-zero x universally. Its optional numerical-rank preprocessing
must not be copied under our no-approximate-rank-reduction contract. COSMO's
Ruiz scaling follows a similar norm-based pattern and does not establish a
cure for the huge legitimate Lambda11 multipliers. Existing sampled frontend
also contains a Gram/free-variable formulation, but switching blindly would
discard the compact sampled Schur advantage.

Algebraic dual-orientation replay at768bits: take x_d=z, q_d=b,
A_d=[A';-selector_PSD], b_d=[-q;0], s_d=[0;z_PSD], z_d=[-x;s_PSD].
The zero-cone dual coordinates stay free; PSD coordinates are orthonormal svec.
For the same iteration0 point, the equivalent dual's normalized primal residual
is0.0936146 rather than2.5994e-76, so the unchanged Clarabel stopping criteria
reject it. The objective-scaled returned point similarly gives2.66345e-4.
Residual mapping identities pass; no full cone/solution audit or native solve.
The equivalent dual has22,196variables and23,275constraints instead of1,099
variables, so explicit generic dualization is not yet a performance proposal.
This is a mathematical dual in svec units, not execution of the older Gram
compiler (which stores unscaled upper triangles). A useful next design must
preserve sampled block elimination while choosing a numerically appropriate
orientation, rather than introduce another dense solve or relax accuracy gates.
Replay12.10seconds under90-second cap, source hashes unchanged; returned-point
hashes verified against the prior frozen replay receipt. Evidence:
/tmp/sdpx-dual-form-replay-20260917/ (summary.json, reference-review.json,
frozen.json, receipt.json, scripts). No production numerical change.

Dual KKT elimination prototype: split original A=[B;C] into equality and PSD
rows. For dual variables (u,v) and multipliers (lambda,eta), the unregularized
Newton equations reduce to [0 B; B' -C'HC]*(u,lambda), with v/eta reconstructed
by block actions. Thus the reduced order is n+k (Lambda11:1099+20), not the
explicit dual's22,196-variable dimension. This is an algebraic feasibility
result, not a new integrated backend or an established speedup.

Diagonal regularization requires care. With primal shift dp and dual shift dd,
J=H+dd*I, W=(I+dp*J)^(-1), T=J*W. The exact reduced matrix becomes
[dp*I B; B' -(C'TC+dd*I)], with corresponding W/T RHS and backsubstitution.
An external Julia prototype compares this elimination against full dense KKT
LU for48cases (256/512/768bits, regularized/unregularized, ordinary and graded
SPD metrics); all solution-difference and original-system RHS-residual checks
pass at sqrt(eps) diagnostic bounds. It takes2.54seconds under60-second cap.
No mixed precision, solver tolerance change or rank reduction. These synthetic
kernel checks do not establish cone scaling or full conic convergence.

Do not implement this by blindly swapping R/Rinv in SampledBlockWorkspace:
T generally loses the congruence structure used by the primitive Gram formula.
An exact rational 2x2 diagonal example G=diag(2,3), dp=1/16, dd=1/32 shows
t(4)*t(9)!=t(6)^2, so a single diagonal congruence cannot represent T.
Next resolve regularization/refinement at the reduced-system level while
preserving original-operator accuracy and the current numerical contract before
building a shared dual-orientation kernel. Evidence:
/tmp/sdpx-dual-schur-20260917/ (check.jl, summary.json, receipt.json,
regularized-congruence-counterexample.json). Production unchanged.

Reduced-only regularization prototype now tested against original dual KKT
residuals at the same working precision. Retain H and sampled congruence actions;
factor only [delta*I B; B' -(C'HC+delta*I)], reuse that factor for correction
solves, and evaluate the full unregularized operator for refinement. Diagnostic
uses the existing MPFR linear-default scale sqrt(eps)*sqrt(sqrt(eps)), maximum
10corrections and stop ratio5; no precision changes. This is an algebraic LU
prototype, not native QDLDL/dynamic-pivot or solver-status qualification.

Across48cases (256/512/768bits, four spectral grades and four RHS seeds), all24
ordinary/moderately graded cases meet the linear residual target in0–1corrections.
The24strongly graded cases stall. Crucially, a follow-up full-KKT LU comparison
also fails the same strict RHS residual target on all24of those cases: do not
attribute every failure to reduced regularization. At the extreme grade the
regularized direction can still differ substantially from full LU, so common
audit failure is not evidence of numerical equivalence or safe acceptance.
This supports feasibility on well-resolved systems but does not qualify
Lambda11, replace its original-coordinate audit, or authorize a new runtime
gate. Next implement a bounded native operator/Schur orientation prototype with
existing refinement and explicit ill-conditioned regression coverage; avoid
assuming that regularization changes alone cure representational scaling.
Evidence: /tmp/sdpx-dual-refinement-20260917/ (48case summary, histories,
full-LU residuals, source identities and receipts). A script initially used
BigInt for a negative power; corrected to BigFloat before completed measurements,
with failed script/log preserved. Each completed run takes about2seconds under
60-second caps; no production numerical change.

Native orientation qualification: four isolated Rust tests pass for Float64,
256/512/768bits, polynomial block dimensions1/2/3 with signed/zero weights and
noncommuting SPD slack pairs. Build real NT cone scaling; compare every sampled
Gram entry against materialized coefficient columns contracted with the native
factored cone.mul_Hs action. The existing workspace needs L=R' for C'HC;
passing R without transpose is deliberately tested and distinguished by every
fixture. This validates reuse of the existing GEMM/SYRK/primitive-entry kernel,
not complete dual Newton directions, severe conditioning or conic convergence.
Evidence: /tmp/sdpx-dual-native-20260917/ (test.rs, test.log, source snapshot,
source-before.json, review.json). Tests run0.01seconds after release compilation;
all production source hashes match the snapshot. No new production backend.

A simpler next native adapter may reuse CondensedKKTSolver itself. For original
A=[B;C], construct surrogate PSD scaling H_surrogate=H_dual^(-1) by swapping
the cone s/z inputs. Given dual RHS (r_u,r_v,r_lambda,r_eta), solve the existing
system with bx=-r_lambda-C'*r_eta and bz=[r_u;r_v]. Its outputs (lambda,t_B,t_C)
map to u=-t_B, v=-t_C-r_eta, eta=C*lambda-r_v. Algebra gives the exact
unregularized dual Newton equations. Existing reduced regularization/refinement
can remain shared, but the reconstructed original dual residual must be
qualified externally; transformation can amplify rounding. Test this complete
direction mapping before adding another solver class or promising a full
Lambda11 improvement. In particular, the NT scaling swap and recovery signs
need native fixed-precision checks, not inference from the Gram test alone.

Complete native direction mapping now passes4tests/48direction cases: Float64,
256/512/768bits; ordinary CSC and factor-authoritative sampled paths; two
noncommuting PSD scaling updates and three known-solution RHS per path.
Construct the same CondensedKKTSolver with zero P, swap s/z for its surrogate
scaling, retain default regularization/refinement, transform RHS and recover
(u,v,lambda,eta) as above. Check recovered directions against the generating
solution and the original dual KKT equations at16*sqrt(eps) external test bounds.
Both paths pass; test execution0.01s after release compilation. This supports
sharing the existing factorization and refinement implementation rather than
maintaining a second Schur solver. The prototype remains test-only, outside the
repository: /tmp/sdpx-dual-direction-20260917/ (test.rs, test.log, source-before.json,
source snapshot, review.json). All production source hashes remain unchanged.

Remaining before dual-orientation integration: mapping the homogeneous embedding
(tau/kappa, constant RHS and objectives), accepted-iterate/status/ray recovery,
scaling and preprocessing coordinates, and preserving efficient sampled
forward/adjoint application without constructing the large explicit dual A.
The48direction cases do not prove any of those, severe-conditioning robustness,
or full Lambda11 convergence. Start with small explicit primal/dual reference
problems to verify embedding and recovery, then introduce the matrix-free
orientation adapter while retaining this native direction test as an oracle.

Explicit conic dual recovery screen passes15paired fixtures/30native solves:
Float64/256/512bits; LP/SOCP/SDP with known optimum1, plus primal-infeasible
and unbounded LP examples. Standard solver, default structural preprocessing
and Ruiz, one thread, unchanged1e-8/1e-42 internal tolerances and1e-6/1e-30
external gates; max150iterations/native5seconds per solve,60seconds total cap.
For optimal dual results recover x=-z_d[equality], s=[0;z_d[cone]], z=x_d;
check original affine equations, both cone memberships, objective gap and known
optimum. For rays, swap primal/dual infeasibility interpretation and verify
normalized original ray equations, cone membership and strictly signed objective.
All mapped checks pass,4.50seconds total, frozen scripts/library unchanged;
frontend/environment identities match the prior frozen receipt. No speed claim.
Evidence: /tmp/sdpx-dual-conic-20260917/ (summary.json, frozen.json, receipt.json,
environment-review.json, scripts and logs).

This uses the explicit dual with the existing standard HSD solver; it validates
these endpoint/recovery contracts, not a custom embedding, arbitrary infeasibility
pathologies, or the compact sampled adapter. Original-arm statuses were checked;
the detailed external checks here apply to the mapped dual outputs. No production
dualization policy or source change. Next connect the already-tested mapped KKT
solve to an orientation-aware data/operator representation, preserving physical
cone coordinates, preprocessing reconstruction and accepted-iterate recovery.

Adapter scaling prerequisite verified: explicit dual Ruiz can change the PSD
selector from -I to -D with nonuniform positive diagonal D. Do not disable Ruiz
or incorrectly treat that selector as identity. If the top equality block is
[B';C'], absorb D into Cbar=D^(-1)C and solve the surrogate with Abar=[B;Cbar],
bx=-r_lambda-Cbar'*r_eta, bz=[r_u;D^(-1)r_v]. Recover
u=-t_B, v=D^(-1)(-t_C-r_eta), eta=Cbar*lambda-D^(-1)r_v.
This preserves the PSD H action; no dense D^(-1)HD^(-1) representation is needed.

Native nonuniform-diagonal direction tests pass4tests/48cases across Float64,
256/512/768bits, ordinary/sampled paths, two scaling updates, identity,
power-of-two and rational diagonals. Recovered directions and all four original
KKT equation groups meet the same16*sqrt(eps) test bounds. No production source
changes; execution0.00s after release build. Evidence:
/tmp/sdpx-dual-diagonal-20260917/ (test.rs, test.log, snapshot, review.json).
These are constructed diagonal fixtures, not actual Ruiz/preprocessing end-to-end
qualification. The orientation adapter still needs connection to the standard
solver loop; preserve this distinction from the earlier explicit-dual solves.

First full-loop mapped adapter now implemented and passes an isolated screen:
/tmp/sdpx-dual-adapter-20260917/. A test-only KKTSolver wrapper recognizes the
explicit dual's leading equalities and diagonal PSD selector after the standard
constructor's preprocessing/equilibration; reconstructs the small surrogate A;
uses inverse NT factors from current cones; and delegates factorization and
refinement to CondensedKKTSolver. DefaultKKTSystem, homogeneous embedding,
initialization, step selection, convergence and solution recovery are unchanged.
The private test-only scaling helper copies only factors consumed by condensed;
it is not a generally initialized cone and must not escape to other consumers.

Three tests/18complete solves compare standard versus mapped backend for a
known-optimum2x2SDP across Float64/256/512bits, three positive selector scales
and nonuniform variable units. Default Ruiz, presolve and chordal settings stay
enabled. Both arms solve every case; paired iteration counts match (5–6 at
Float64,22–23 at high precision). Original-coordinate equations, objective=-1,
gap, recovered primal optimum1 and PSD membership checks pass at1e-6/1e-30
external gates; internal tolerances remain1e-8/1e-42. Test execution0.04seconds
after57.75s release compilation; not timing evidence. Initial build missed a
CoreSettings import; corrected with initial-build-failure.log preserved.
Production source hashes match pre-experiment identities; no adapter integrated.

Limits: this prototype still materializes explicit dual data, covers only PSD
selectors and fixed P=0, and explicitly rejects update_A rather than claiming
prepared-update support. It has no large sampled input, multi-block/free-variable
or adapter-specific ray qualification yet. Next expand those contracts on small
fixtures, then connect factor-authoritative sampled operator metadata to avoid
large explicit dual matrices. Do not count the18solves as Lambda11 qualification
or a demonstrated performance improvement. Source, adapter.rs, test.log and
review.json retain the exact implementation and results outside the repository.

Mapped-adapter contracts expanded: three further tests/18complete solves pass
for Float64/256/512bits. A two-block2x2SDP with a retained free dual variable
and coupled primal equality has known optimum4; additional trace-inconsistent
and improving-ray SDP fixtures test both infeasibility directions. Standard
and mapped backends match statuses and iteration counts in every pair. Check
original mapped affine/stationarity equations, gap, known optimum, PSD principal
minors, and normalized ray equations plus strict objective sign. Assert the free
variable survives preprocessing rather than silently testing only k=0. All
default preprocessing/equilibration settings remain enabled;1e-6/1e-30 external
gates unchanged. Execution0.06s after58.83s release build, no timing claim.

Evidence: /tmp/sdpx-dual-adapter-contracts-20260917/ (extra.rs, source snapshot,
test.log, review.json); initial Rust comparison-token typo preserved in
initial-build-failure.log. Production source identities unchanged. These
small PSD fixtures qualify more adapter contracts but not arbitrary cone types,
prepared data updates, poor conditioning or large sampled problems. Next attach
factor-authoritative operator metadata: dual forward/adjoint products should
reuse original sampled transpose/forward products and selector operations,
with equilibration represented explicitly. Do not regenerate approximate factors
from rounded CSC or assume nonuniform dual-variable scaling preserves a simple
PSD congruence inside the original sampled operator.

Factor-authoritative dual product prototype implemented outside production:
E*[A';-selector]*D reuses the original SampledOperator transpose for its upper
forward rows and its forward product for the adjoint. Diagonal scaling and
selector terms use persistent input/output scratch; original factors are never
regenerated from CSC. No explicit dual matrix is built inside these products.
The explicit CSC exists only in test reference construction. Four native tests
pass at Float64/256/512/768bits:96forward/adjoint pairs and24adjoint identities,
two overlapping-column PSD blocks, polynomial dimensions1/2/3, signed/zero
weights, identity/nonuniform rational scaling, four alpha/beta combinations,
and repeated workspace reuse with input immutability checks. External comparison
uses the existing4096*eps scale bound; this is operator qualification, not
bitwise equivalence to differently associated materialized arithmetic.

Evidence: /tmp/sdpx-dual-operator-20260917/ (test.rs, test.log, snapshot,
review.json). Execution0.00s after release compilation; production source
identities unchanged. Arbitrary nonuniform row scales in these algebra fixtures
do not authorize non-cone-preserving Ruiz scaling. Next connect the tested
product metadata to DefaultResiduals and the mapped KKT adapter together;
retain standard cone-compatible equilibration, preprocessing and solution
recovery. A full solver that still uses CSC residuals must not be described
as factor-authoritative merely because its Schur backend uses sampled factors.

Combined factor-loop prototype now passes3tests/6complete solves at
Float64/256/512bits. In the isolated source, DefaultResiduals receives the
tested dual product workspace; forward/adjoint invocation counters confirm it
is actually used. The mapped KKT wrapper's RHS/recovery products and its inner
CondensedKKTSolver use a scaled copy of the same original sampled factors.
Column factors are the dual equality-row equilibration; PSD row factors are
inverse dual cone-row equilibration (verified uniform per block); free rows
retain their dual variable scales. No factors are fitted to rounded CSC data.

The fixture has one free dual coordinate, a2x2PSD block, nonorthogonal rank-one
sample bases and a trace constraint with known original optimum1. Default
preprocessing/Ruiz/chordal settings remain enabled and structural dimensions
are asserted unchanged for this fixture. Standard versus factor-backed arms
both pass original factor-operator equations, objective/gap and PSD checks at
1e-6/1e-30; iteration counts match (5Float64,22at256/512bits). Internal gates
remain1e-8/1e-42. Test execution0.02s after release build, no timing claim.
Evidence: /tmp/sdpx-dual-factor-loop-20260917/ (test.rs, isolated residual/operator/
adapter source, test.log, review.json). Production source hashes unchanged.

The test still materializes explicit dual A for the standard setup/preprocessing
and baseline; it does not establish end-to-end matrix-free setup or memory gains.
The hooks are test-only and prepared data updates remain unsupported. Before
large Lambda11, provide a bounded constructor path that installs the compact
backend without first constructing an expensive generic dual KKT, then validate
larger factor data with actual equilibration and original-coordinate audits.

Accepted runtime dispatch preserves the arithmetic implementation and checks AVX2
and FMA before entering the specialized kernel, retaining the portable fallback.
Against the accepted residual-overlap baseline on node49, medium native medians
at 1/4/16 cores improve 11.344→9.017 / 11.288→8.898 / 11.380→8.950 seconds
(20.5–21.4%). LP_bore3d/SOCP_axis_wide/SDP_truss2 are 0.38/1.58/1.48% slower;
SOCP_nb16 is 1.08% faster. Ordinary 256/512-bit SDP improves 1.88/0.54%, below
the performance-credit threshold. Sampled Ising bypasses the modified helper;
its observed 1.01/0.87/4.20% changes do not establish a causal speed benefit.
Precision and numerical gates are unchanged. These Linux timings are not a
matched MOSEK comparison. Production source SHA 94114e53, macOS FFI 948f6aa0.
Evidence: `/tmp/sdpx-runtime-fma-combined-20260916/acceptance/final-decision.json`,
`acceptance/build-logs/`, `frontend-verification.json`, `integration-review.json`.
The publication scope retains only accepted changes. The isolated profile includes all 13 stage
counters, explicitly fixing the previous diagnostic's missing scaling-sync span. PBS 213042.node220
completed medium at 1/16 and Ising512 at 16 physical cores in isolation.
All numerical/profile gates, 363 source and 3 library checks pass. Instrumented
cargo check passed. Medium assembly remains 5.079/5.348 seconds at 1/16 cores,
with refactor 1.300/1.300 and reduced refinement 1.103/0.580 seconds. Refresh
assembly's transform/dot/store breakdown before the next kernel hypothesis;
previous unsuccessful parallel assembly variants remain rejected. Ising16
assembly is only 0.059 seconds: prioritize refactor (1.763), outer residual
(1.246), and raw input/output scaling (0.732/0.891), not Schur assembly for this
sampled instance. Reduced and refinement/backsolve counters nest; never sum
parent and child times. These observations are not a paired speed comparison.
Evidence: `/tmp/sdpx-post-fma-profile-20260916/final-decision.json` and `build-logs/`.
Fine assembly diagnostic 213043 completed with all numerical/counter gates,
363 frozen-source and 3 binary checks passing. On medium1, worker-span totals:
coefficient product 0.408s, GEMM 2.748s, suffix packing 0.411s, dense dot 0.215s,
store 1.088s, sparse pairs 0.195s. Transform total 3.581s and selected total
5.111s include children; do not add them together. Medium16 remains slower in
these serial-dominated stages. No production instrumentation.
Evidence: `/tmp/sdpx-post-fma-detail-20260916/final-decision.json` and `build-logs/`.

Store-transpose screen 213044 completed all eight numerical receipts and
653 source/four binary checks. Reject it: medium1 improves only 0.11%
(9.0570→9.0468s), while Ising512-16 regresses 10.57% (12.9608→14.3306s).
No integration and no unchanged repeat. Its local 291 core tests passed, but
numerical correctness alone does not justify the extra workspace.
Evidence: `/tmp/sdpx-store-transpose-20260916/final-decision.json` and `build-logs/`.

Independent MPFR candidate uses a shared mutable-descriptor helper for output
construction and in-place assignment. The existing +=/-=/*=/ /=/%= operators
call the same MPFR operations with unchanged fixed precision and nearest-even
rounding; destination storage is exclusively borrowed and no descriptor escapes.
[MPFR variable conventions](https://www.mpfr.org/mpfr-current/mpfr.html#MPFR-Variable-Conventions)
permit input/output aliasing. The candidate preserves copy ownership and tests
128/256/512/768/1024/2048 bits, signed zeros, infinities, NaNs, cancellation,
large exponents and reused destinations. All 6 arithmetic and 291 solver core
tests pass. Warm local add/sub ABBA microtests reduce elapsed time by
19.86/31.16/38.39% at 256/512/1024 bits; this is not whole-solve evidence.
Isolated screen 213045.node220 completed after 213044;
medium1/Ising512-16 ABBA, 653 frozen files, release arithmetic tests, rebuilt
baseline/candidate FFI, fixed accuracy gates and final identities. It starts
from the accepted runtime-FMA source, excluding the store-transpose candidate.
16 cores/32 GiB/45 minutes. No production arithmetic change or adoption claim.
Evidence: `/tmp/sdpx-mpfr-assign-20260916/`.
All eight numerical receipts and 653 source/four binary checks pass, but reject
the candidate for insufficient whole-solve benefit: medium1 9.0068→9.0260s
(0.21% slower), Ising512-16 13.2363→13.2115s (0.19% faster). The microbenchmark
improvement did not translate into a qualifying solver improvement. No integration.
The stable runtime-FMA/residual-overlap version remains the publication candidate.






Latest matched high-precision evidence: at 512 bits and fixed 1e-42 internal /
1e-30 external gates, Ising SDPX warm-native medians at 1/4/16 physical cores are
71.19/22.81/13.26 s, versus SDPB per-iteration timer sums of 127.06/35.85/15.66 s.
These clocks have different scopes. SDPX takes 50 iterations, SDPB 201; SDPX's
iteration cost and parallel efficiency remain worse. This qualifies one finite
Ising problem, not general SDPB parity. Detailed receipts and scope appear below.
Runtime FMA and inlining passed short screens, but remain outside production
until the combined broad campaign finishes.

Latest local single-core medium refresh is SDPX **3.055060 s / 18 iterations**
versus MOSEK 11.1.3 **1.921669 s / 16 iterations**: native reported medians give
**1.59x slower**. Two fresh-process samples each, interleaved SDPX/MOSEK/MOSEK/SDPX;
328 frozen files unchanged, identical original raw coefficients verified after
MOSEK export. SDPX passes its additional independent dual audit; the retained
MOSEK checker covers original equalities, PSD feasibility and nonmissing gap,
without an identical dual-stationarity oracle. The frozen MOSEK driver explicitly uses CVXPY `eps=1e-6` (not MOSEK
default tolerances); SDPX primary internal tolerances are also 1e-6, with
solver-specific stopping rules. Expanded MOSEK parameters are recorded in
`effective-settings.json`. This remains a scoped comparison, not parity evidence.
The prior 3.159440/1.945962 pair is historical, not a controlled speedup baseline.
Evidence: `/tmp/sdpx-local-refresh-20260916/summary.json` and `results/`.
A supplemental MOSEK solve now exports its original x/equality/PSD dual point
and passes the exact shared 256-bit-accumulation Julia dual auditor used for
SDPX. Equality dual signs are converted from CVXPY; nine analytic sign/trace
checks pass. Upstream-normalized stationarity 1.6242e-8, gap 3.7197e-7, dual PSD
violation zero, original equality residual 1.0383e-9 and primal minimum eigenvalue
-1.7982e-14. Original raw inputs and shared helper match byte-for-byte.
This validates the supplemental point, not retrospectively the timed points
which were not exported. Evidence: `/tmp/sdpx-medium-common-audit-20260916/`.

This is native solve time at the original 1e-6 external gate, not broad parity.
Input `../大规模矩阵/models/medium/SU2path465var1887.json` has SHA256
`b1d973ac564e206fbb29aaad0a9b947c112f1130956aab31e79d899023c30048`,
1887 variables, 47 equalities and PSD orders 60, 57, 59, 57, 116.
Exact presolve removes 14 equalities. The 1e-8 full-accuracy failure remains.
A newly independent original-model dual audit also finds stationarity/max(1,
norm(q,Inf)) = 1.91649e-6 at the stable returned point, above a 1e-6 gate.
The historical timing pair passed its original primal/gap checks; it does not
prove that stronger target-only stationarity acceptance. Independently
reconstructed Clarabel normalization is 2.51278e-7 and passes the established
1e-6 criterion. See the normalization clarification below.

A fresh group was fixed before running: six previously reserved public inputs
from ClarabelBenchmarks revision `3679912c6bbd3f64c5c962f9d1c09d524561c412`
and three deterministic planted LP/SOCP/SDP-interface diagnostics (two SOCP,
one SDP). All nine are now exposed regression cases. Four holdouts are now reserved without solver execution: LP_ship04s,
SDP_copo14, SDP_filter48_socp and pure SOCP_strictmin_2D_43_dual. Source provenance,
conversion checks and 180-second/4096-MiB execution caps are recorded below.
These remain unavailable for optimization tuning; passing final holdout acceptance
is still unproven. The previously downloaded ss30 is another truss-topology instance; db_shear_wall appears in a historical exclusion list. Neither proves a new independent family. Both were format-converted without solver execution, and are excluded from the new holdout pending genuinely independent sources. Synthetic planted optima do not establish representative performance.
MOSEK passes 9/9, SDPX passes 8/9. SDP_hinf3 returns AlmostOptimal: original
primal residual 3.12e-4 exceeds 1.32e-5, and dual residual 2.64e-6 exceeds 2e-6.
Explicit condensed formulation also fails and is not adopted.

Warmed API medians in milliseconds, one thread, unchanged original-coordinate
1e-6 gate, default preprocessing. Entries below summarize the first matched run;
reverse-order and additional samples are retained externally.

| Case | SDPX baseline | Exact-presolve candidate | MOSEK | SDPX gate |
|---|---:|---:|---:|---|
| LP_bore3d | 7.047 | 3.534 | 0.808 | pass |
| LP_e226 | 5.491 | 5.449 | 2.928 | pass |
| LP_grow7 | 13.477 | 3.705 | 3.035 | pass |
| LP_sc50a | 0.271 | 0.265 | 0.329 | pass |
| SDP_hinf3 | 1.338 | 1.302 | 3.812 | fail; no speed credit |
| SDP_truss2 | 3.029 | 3.023 | 6.394 | pass |
| SOCP_axis_many (synthetic) | 7.043 | 7.109 | 3.645 | pass |
| SOCP_axis_wide (synthetic) | 7.300 | 13.324 | 6.227 | pass |
| SDP_planted80 (synthetic) | 59.237 | 59.470 | 19.584 | pass |

SOCP_axis_wide's apparent regression did not reproduce: retaining all 18 warm
samples per arm gives 7.266 versus 7.217 ms. The planted SDP takes MOSEK one
iteration and SDPX seven; it cannot support a general SDP ranking.
Process high-water RSS in the first paired run is 659/65 MiB (SDPX/MOSEK)
for bore3d and 674/83 MiB for planted80. These include Julia/Python and the
external oracle, not solver-only memory; do not attribute the whole gap to
native matrix storage.

The exact-presolve candidate peels structurally independent singleton-column
rows before bounded GMP elimination. It never deletes a row merely because it
was peeled. LP_grow7 previously spent 10.9 ms in presolve; all 140 equality rows
are structurally independent. Repeated local LP gains are about 50% on bore3d
and 73% on grow7. The candidate is frozen externally; **rejected for now after an Ising regression**.

Later local medium timings changed to 6.82/11.28/12.40/6.72 s (baseline/candidate/
candidate/baseline). Numerical outputs match, but these measurements do not
qualify adoption. Local Ising baseline also shifted from about 39.6 to 85–86 s;
no causal kernel conclusion is drawn from this changed timing environment.

### Fixed-core pilot observations

PBS 212917 on node48 has completed the Float64 ABBA cells: bore3d warm API
medians are 14.430 -> 6.819 ms, grow7 28.833 -> 7.252 ms, and axis_wide
26.595 -> 26.572 ms. All pass. Medium native medians are 61.045098 ->
60.788815 s, with 18 iterations and original gates passing; its prior local
regression does not reproduce here. Ising paired acceptance subsequently completed; its regression decision is below.
These Linux numbers use Netlib and must not be compared directly to macOS
Accelerate numbers or treated as a best-provider comparison with MOSEK.

A raw-medium sparsity screen of HDSDP-style intermediate contractions predicts
no winning columns under the current flop cost model, at either Float64 or
512 bits. This is a structural estimate, not a kernel timing or a proof that
M4 is useless elsewhere. Avoid adding a route solely to match HDSDP's checklist.
PBS **212918.node220** completed on node49. This experiment
compared the same candidate source and LP64
arithmetic using static Netlib versus installed OpenBLAS 0.3.29 on one fixed
physical core. Both libraries and input/operator audits are preserved; optimized
BLAS may explain much of Linux medium's cost. The build and runtime preflight
confirm OpenBLAS 0.3.29, the Zen kernel, LP64 integers, one thread, and loading
from the immutable campaign native-library copy. Completed paired timings and their scope are recorded below.
Experiment snapshots: `/tmp/sdpx-fresh-20260916/pilot-observation-01/` and
`hpc:~/projects/sdpx-blas-20260916-pilot01`.

## Retained implementation

- Dense positive-block Cholesky with the correctly signed bordered complement,
  existing refinement and sparse fallback; packed Schur dot fusion.
- Exact coefficient reuse and block-local sparsity ordering share Float64/MPFR
  logic. Equality is checked after scaling and updates; hashes never suffice.
  Compact triangular Float64 panels reduce memory; MPFR uses streamed storage.
- Factor-authoritative sampled operators, exact sampled-basis reuse, structured
  direction/residual products and bounded threaded contributions.
- Bounded two-direction curve search for Float64 symmetric cones: no extra KKT
  solve. Medium improved 3.339455 -> 3.148746 s, 19 -> 18 iterations. MPFR keeps
  its original step policy because the same heuristic regressed Ising.
- One exact GMP equality presolver for Float64/MPFR, with original-coordinate
  recovery and sampled-factor remapping. Proof limits: 512 rows, one million
  elimination entries, 2048-bit rational components. Inconsistent RHS and
  unproven dependencies are retained. Medium row reduction itself is not a
  measured speedup.
- MPFR allocation test now counts only its calling thread, avoiding false failures
  caused by concurrently running provider tests. This changes test instrumentation.

Rejected prototypes are outside the repository: lagged-factor GMRES (all attempts
failed its residual gate), additional recentering solves, sign-opposite reuse,
COSMO eigensolver workspace/minimum-eigenvalue caching, short-dot specialization,
active-row tail GEMM, and alternative Float64 crossover. They did not deliver
repeatable qualifying full-solve improvements. Do not reintroduce them without
new evidence. Prior receipts: `/tmp/sdpx-{curve,unified,cosmo,signed}-20260916/`.

## Immediate acceptance and accuracy work

1. Qualify the residual-stage overlap candidate with **212949.node220**:
   56 fixed receipts covering LP/SOCP/SDP, medium and Ising at1/4/16 workers,
   plus ordinary256/512-bit single-worker regression. The initial paired screen
   passed all gates and reduced Ising512/16 time7.74%; medium16 improved1.81%.
   Do not integrate before broader acceptance. Independently, **212948.node220**
   tests byte-identical stable source with AVX2/FMA enabled after disassembly
   proved scalar indirect FMA calls in a hot generic-x86 kernel. Separate build
   gains from algorithm gains and qualify CPU compatibility before deployment.
   Both candidates are independent of rejected fusion/layout/batch experiments.
2. Preserve the integrated residual-reuse and parallel row-gather baseline.
   PBS **212929** passed all 48 numerical receipts; gains were 7.38% on
   medium-16, 5.27% on Ising-512/1 and 28.27% on Ising-512/16. Local release
   verification passed 508 assertions in 32 Julia testsets. Stable library SHA256:
   `8e4017e2fcbaf220950ddc2a5a2239a4244ccb16ae518fa19298a1f7fddf664f`.
   Evidence: `/tmp/sdpx-combined-20260916/final-decision.json` and build/test logs.
   These are matched SDPX comparisons, not new MOSEK or SDPB parity results.
3. Diagnose medium's 1e-8 and SDP_hinf3 full-accuracy failures before interpreting
   full-accuracy speed. Preserve existing residual normalization and distinguish
   the stronger target-only diagnostic. No tolerance/status promotion. Failed
   regularization and precision-preserving residual experiments need a new
   hypothesis before another sweep.
4. Larger Ising/Lambda11 remains unqualified: the previous fixed 768-bit run had
   sampled dual consistency 2.16e-22 versus the 1e-30 gate. The saved-point replay rules out a material CSC/factor mapping mismatch;
   diagnose conditioning and near-null solution growth before large-scale comparisons. The unsolved LP, SDP and pure-SOCP reservations now cover separate
   source families; exposed development problems cannot establish generalization.
5. After numerical qualification, compare matching providers, precision and
   tolerances against Clarabel.rs/MOSEK/SDPB, then extend real single-solve
   scaling beyond 16 cores. Ising currently reaches about 4.99x at 16 cores;
   medium has no demonstrated net multicore speedup. MPI and 64/256-core
   multi-node acceptance remain unfinished.

Keep rejected prototypes and detailed receipts outside production. Historical
experiments and failure diagnoses below remain evidence, not active work orders.

## Multi-node work after numerical qualification

The user authorizes **1, 4, 16, 64, 256 physical cores**, cluster execution and
GitHub/cluster publication of the verified result. SDPX currently has a shared
thread pool and serial MPFR QDLDL, **no distributed MPI solver**. MPI launch
receipts now use global ranks and explicit per-host CPU maps (five binding tests
pass); the existing controller still measures one node. Running multiple
independent full solves measures throughput, not single-solve scaling.

Implement in this order, keeping one solver algorithm and optional typed backends:

1. Measure per-iteration cone scaling, structured assembly, factorization, backsolve,
   reductions and idle time; record block costs and memory. Reuse scheduling plans
   instead of rebuilding them during every cone operation. Demonstrate 1/4/16/64
   single-node behavior before attributing any limit to networking.
2. Exploit exact block-arrow structure in the condensed Newton system. Local SPD
   factors and triangular solves form a smaller global complement. Preserve signs,
   regularization, full residual refinement and direct fallback. Share arithmetic
   logic across Float64/MPFR; do not force unrelated sparse LP/SOCP into SDP blocks.
3. Distribute independent blocks across MPI process groups with measured-cost
   allocation. Own block matrices and scratch locally; communicate the required
   coupling data. Within a large block, use a mature distributed dense provider
   through a narrow native interface instead of writing another matrix library.
4. Qualify the provider's arbitrary-precision storage, arithmetic and serialization.
   The SDPB Elemental fork's GMP arithmetic is not automatically equivalent to
   our MPFR rounding contract. No conversion through Float64, raw MPFR-pointer
   transfer or reduced precision in communication/reductions.
5. On a qualified large frozen problem, compare both solvers sequentially at each
   width on matched nodes. Verify physical cores versus SMT, MPI rank placement,
   node count, BLAS budgets and memory. Report T1/Tp, T1/(p*Tp), iteration counts,
   per-iteration phase time, communication time and aggregate/per-rank RSS.
   Include cross-node 256-core execution and failures; a tiny 22-block Ising case
   alone cannot establish useful 256-core scaling.
6. Adopt only validated improvements, remove unused experiment plumbing, then
   commit/push the reviewed source and publish an immutable matching cluster
   snapshot with source, dependency, library and input hashes. Publication remains
   pending; do not label unfinished MPI work as released.

## Source guidance

[MOSEK's GAMS documentation](https://www.gams.com/53/docs/S_MOSEK.html) exposes
solve-form choice, multiple correctors, ordering/dense-column handling and bounded
presolve work. These suggest experiments, not knowledge of proprietary internals.
Approximate folding and bound perturbations conflict with the present contract.
[QOCO](https://arxiv.org/html/2503.12658v4) suggests precomputing fixed structural
work; its specialized solver generation and different infeasibility machinery
are not replacements for SDPX's general homogeneous solver.
[Proximal stabilized SDP](https://doi.org/10.1007/s10589-024-00614-3) is a possible
conditioning direction, requiring algorithm-level accuracy and cost validation.

[SDPB's scaling paper](https://arxiv.org/html/1909.09745v1) and
[implementation](https://github.com/davidsd/sdpb) motivate hierarchical block
ownership, measured load balancing, distributed matrix kernels and reducing
shared global operations. Inspect `allocate_blocks.cxx` and `compute_Q.cxx`.
COSMO's clique merging is already represented; ADMM projection and Anderson
acceleration cannot be copied unchanged into an NT interior-point step.

### HDSDP source review

Reviewed COPT-Public/HDSDP at `8842fedd13cddcd021f047dd9a9f289aa2fc90b2`.
Its `interface/hdsdp_conic_sdp.c` chooses M2–M5 per coefficient using rank,
nonzeros and remaining-column work. M1 eigen-decomposition is explicitly disabled.
Prioritize a measured comparison of the missing intermediate-product strategy
against SDPX's existing sparse-pair and packed-dense routes; reuse one typed
planner and precision-specific cost estimates. Existing sampled rank-one and
sparsity paths are not new opportunities merely because HDSDP has them too.

Do not copy `1e-10` rank-one extraction from `dense_opts.c`/`sparse_opts.c`:
use authoritative factors or exact structure. Its restarted PCG with reused
Cholesky applies to an SPD Schur system, not the full indefinite SDPX KKT;
our rejected lagged-factor experiment is still relevant counter-evidence.
The dual-scaling algorithm could reduce scaling work but changes iterates,
primal recovery and convergence behavior; it is a separate research direction,
not a replacement NT formula. The inspected public C kernels use double and
threaded numerical providers, not SDPB-style distributed MPFR/MPI matrices.
No HDSDP-derived numerical implementation or speedup is claimed from this review.
Source: [HDSDP](https://github.com/COPT-Public/HDSDP/tree/8842fedd13cddcd021f047dd9a9f289aa2fc90b2),
[paper sections 5 and 7](https://arxiv.org/html/2207.13862v2).

### Completed first cluster acceptance

PBS 212917 completed. All 12 Ising512 solves pass the original 1e-42 internal /
1e-30 external protocol at 50 iterations. Four native observations per arm give
baseline 73.673786 s, exact-presolve candidate 76.203102 s (+3.43%), and combined
MPFR candidate 73.013663 s (-0.90%). The presolve candidate's x/s/z are exactly
identical to baseline. The regression is real in this paired campaign, but its
mechanism is unresolved; do not attribute it to changed iterates. Restore the
production presolver and keep the LP improvement as evidence for a revised trial.
The combined MPFR prototype does not meet the 2% full-solve adoption threshold.
Raw receipts: `/tmp/sdpx-fresh-20260916/pilot-final/`.

A generic HDSDP M4 prototype now passes 284 solver library tests outside the
workspace. Its contraction uses G*A and evaluates only requested left entries;
no numerical rank extraction or arithmetic conversion. A 12-case kernel screen
covers Float64/256/512 bits and four sparsity configurations. In two moderate
sparsity MPFR cases M4 beats both existing kernels by about 2–3x; Float64 favors
batched/dense products. These are local microbenchmarks, not solver speedups.
The external prototype selects M4 with a structural cost model for streamed wide
arithmetic and updates scheduling cost estimates; full-solve acceptance remains.
Source/tests: `/tmp/sdpx-hdsdp-m4-20260916/`. No production M4 route is adopted.

### Completed Linux BLAS paired result

The Float64 medium ABBA cells in PBS 212918 pass at 18 iterations:
Netlib 60.863673 / 61.014321 s, OpenBLAS 11.389320 / 11.382507 s.
Paired native medians are 60.938997 -> 11.385913 s (5.35x speed ratio).
This uses the identical frozen presolve-candidate source, one physical core,
and unchanged original-coordinate gates. It isolates provider configuration;
it is neither an algorithmic speedup nor a MOSEK comparison. Confirm the provider
on restored stable source before publishing. The high-precision cross-provider
audit completed successfully; it has only one process per provider and does not
establish a repeatable Ising speedup. Keep OpenBLAS as the leading Linux provider candidate.
The restored production presolver passes all seven exact-dependency tests.

### M4 whole-solve development screen

A fixed ordinary CSC SDP (PSD order 32, 12 variables, 24 coefficient entries
per sparse column, deterministic signed coefficients, known optimum zero) now
passes eight independent original-coordinate audits. Default Ruiz/presolve/chordal
remain on; this diagnostic explicitly selects condensed KKT. At 256 bits,
internal 1e-24 and external 1e-20 gates give 24 iterations, baseline/candidate
native medians 5.773249 / 5.634718 s (2.40% lower). At 512 bits, internal 1e-42
and external 1e-30 give 41 iterations, 14.413975 / 14.092439 s (2.23% lower).
ABBA order is fixed and every primal/dual residual, gap, PSD violation and
known-optimum check passes. Local exploratory results do not establish broad
performance or Ising speedup. Keep the prototype external until independent
ordinary/sampled and fixed-core acceptance. Receipts and frozen executables:
`/tmp/sdpx-hdsdp-m4-20260916/pairs/`, `decision.json`, `m4-solve-*`.
PBS 212918 has completed with successful Ising output audits and unchanged
native-library hashes; no OpenBLAS-related accuracy failure was observed in
that campaign. Stable-source provider qualification remains the next Linux step.

### Completed stable-source M4 acceptance

PBS **212920.node220** completed on node49 with exit 0, staged at
`hpc:~/projects/sdpx-m4-20260916-accept02`. The prior attempt 212919 exited
101 while building the example: its feature list omitted `sdpx-solver/sdp`.
No numerical cells ran. Failure records remain in accept01. The corrected
feature combination passes a local example compilation check; source algorithms,
inputs and tolerance settings are unchanged. Both frozen arms use the restored
stable presolver and the same copied OpenBLAS LP64 provider. Their only numerical
source difference is the generic M4 assembly route and its work estimates.
The script builds both libraries/executables before timing, then runs ABBA
LP/SOCP/SDP and medium, ordinary SDP at 256/512 bits, and Ising512. Every timed
solve binds one physical core; audits remain outside timing. Both builds have
completed. All 36 receipts and original numerical gates pass. Native median
seconds (baseline/candidate): medium 11.387008/11.298859, ordinary256
11.503297/11.494823, ordinary512 26.901631/26.373859, Ising512
73.822490/75.962958. M4 is **rejected**: Ising regresses 2.90%; other gains
remain below 2%. Final native/library identity checks pass. The stable-source
OpenBLAS arm is numerically qualified on these inputs, not MOSEK/SDPB parity.
Receipts: `/tmp/sdpx-hdsdp-m4-20260916/acceptance-final/`; decision:
`/tmp/sdpx-hdsdp-m4-20260916/acceptance-decision.json`.
Two additional local tests pass: M4 selection and original Schur entries after
coefficient changes/stored-zero updates with repeated-column reuse, at 256 and
512 bits. These test-only additions are outside the frozen cluster source;
the numerical implementation is unchanged. No candidate is promoted yet.

The external campaign assessor requires all 36 expected receipts, checks each
individual numerical result, and withholds acceptance on missing/failed cells.
Tampering tests reject NaN, infinity, negative and out-of-tolerance residuals even
when the stored accepted flag is true. `assess_cluster.py` remains outside the
repository; these are experiment checks, not new runtime solver validation.

### Accuracy isolation and acceptance evaluator repair

A frozen hinf3 diagnostic disables only the Float64 curve step outside production.
Both arms still return AlmostSolved at 22 iterations, with identical printed
original primal residual 3.118910e-4, dual residual 2.637993e-6 and gap 3.358287e-9.
This does not support the curve as the cause; do not remove the qualified medium
step optimization on this evidence. Observation-only instrumentation isolates the stop to KKT at iteration 22:
the factorizer returns success but the initial refinement residual is NaN.
Earlier constant-RHS solve residuals deteriorate to about 0.05 for RHS norm 19.1.
PSD scaling and the small-step gate do not fail first. An isolated Faer substitution also fails (iteration 26, initial refinement
residual NaN; original primal residual 6.983969e-4 and dual 2.771077e-6).
Backend substitution alone is insufficient. Next inspect PSD-derived KKT
numerical ranges and regularization/refinement stability; retain upstream checks. Logs and candidate are in
`/tmp/sdpx-accuracy-20260916/`; the stable production library hash is unchanged.

The M4 campaign's Float64 ABBA cells pass: medium medians 11.387008 / 11.298859 s
(baseline/candidate, only 0.77% lower). Ordinary 256-bit SDP also passes, at
11.503297 / 11.494823 s (0.07% lower). These gains do not qualify adoption.
The 512-bit ordinary pair also passes at 26.901631 / 26.373859 s (1.96% lower,
still below threshold). Ising acceptance has now completed and rejects M4, as recorded above. The external assessor's
exact-decimal tolerance comparison falsely rejected the normal MPFR round-trip
representation of 1e-20. It now compares the tolerance at the declared binary
precision; original residual gates still use the exact decimal limits. Tests
reject real tolerance changes and invalid/out-of-bound metrics. Solver and
frozen campaign evaluators are unchanged. Snapshot: `observation-02` under the
external M4 experiment directory.

The finer hinf3 trace identifies factor growth: at iteration 21, KKT max is
4.97e13 and L max is 3.87e8. At iteration 22, KKT remains finite (max 7.11e14),
but L contains infinity after 10 dynamic pivot regularizations. D and Dinv
remain finite, so the existing Dinv-only refactor check reports success; the
subsequent solve/refinement correctly rejects NaN. No invalid result is promoted.
Prioritize scaling/regularization against factor growth, not extra refinement
of already nonfinite factors. Trace: `/tmp/sdpx-accuracy-20260916/factor-overflow.json`.

A bounded diagnostic sweep of the existing static regularization constant
(1e-8, 1e-6, 1e-4, 1e-2; unchanged full tolerances, one core) returns
AlmostSolved at 22, 19, 62, 200 iterations respectively. None qualifies an
accuracy fix. Do not raise the global default on this evidence. The external
`accuracy-probe` executable permits further targeted settings tests without
recompiling the production library; artifacts remain outside the repository.

A four-sweep symmetric KKT equilibration prototype applies D*K*D, scales RHS
and recovered solutions consistently, and retains refinement in the original
KKT coordinates. It avoids the observed hinf3 overflow but returns AlmostSolved
at 57 iterations: original primal residual worsens from 3.11891e-4 to 1.12629e-3,
although dual residual and gap improve. Reject this simple scaling route; neither
finite factors nor a small gap establish full convergence. Prototype and paired
raw points: `/tmp/sdpx-accuracy-20260916/scaled*`. No production numerical edit.

### Research after stalled accuracy experiments

Reviewed local Hypatia `systemsolvers/qrchol.jl`: it eliminates equality directions
with QR, then builds a reduced SPD matrix from Hessian products; square-root
oracles form Gram products when available. This is not a drop-in stable solver
for the full SDPX augmented system, and forming a Gram matrix can still square
conditioning. Reuse operators where applicable rather than adding a second engine.

[HiPO (2025), sections 4.5–4.6](https://arxiv.org/html/2508.04370v1) addresses
pivot growth with local pivot selection and scale-aware regularization. A first
small diagnostic retains the magnitude of a wrong-sign pivot when restoring its
expected sign, instead of replacing a large pivot by the tiny fixed shift. This
is an isolated hypothesis, not an implementation of HiPO's full pivot rule.
Original KKT refinement and accuracy gates remain authoritative.

[Well-conditioned SDP reformulation](https://arxiv.org/html/2407.14013v2)
derives a reformulation/preconditioner with assumptions on the spectral splitting,
centering and strengthened uniqueness. It is relevant to large low-rank SDPs,
but does not establish generic conditioning or speed for our medium/Ising inputs.
Any future spectral partition must preserve the complete Newton operator; no
approximate rank deletion or mixed precision is authorized.

The magnitude-preserving pivot diagnostic also fails: AlmostSolved after 27
iterations. It is rejected; no global pivot policy changes. Evidence:
`/tmp/sdpx-accuracy-20260916/pivot-decision.json`. A subsequent diagnostic should
compare a mature pivoting factorization on the same small failing KKT before
committing to a new regularization or reduced-system design.

A dense partial-pivot LU diagnostic reuses the existing LAPACK `gesv` provider
on the same statically regularized KKT, retaining original-operator refinement.
It reaches internal Solved in 46 iterations but violates the external primal
gate (1.46883e-4 > 1.31853e-5). Same-T compensated residual accumulation with
FMA product-error recovery reduces this to 20 iterations, but primal 8.70847e-5
and dual 2.37563e-6 still fail. Internal tolerances 1e-9 and 1e-10 both stall at
103 iterations (diagnostics only; defaults unchanged). Reject adoption: stable
pivoting and more accurate residuals alone do not repair this ill-conditioned
case. This prototype refactors each RHS and is not a production performance
implementation. Evidence: `/tmp/sdpx-accuracy-20260916/lu-compensation-decision.json`.

### Stable Ising scaling pilot

PBS **212922.node220** submitted to node49 with 16 physical-core slots, 32 GiB,
2 hours; initially queued. Campaign: `hpc:~/projects/sdpx-scaling-20260916-pilot01`.
Reuse the qualified stable OpenBLAS library from completed 212920; no numerical
source rebuild or candidate changes. Width order is 1/4/16/16/4/1, one process at
a time, each first plus one warmed fresh solve. Internal 1e-42, external 1e-30,
512 bits and one BLAS thread remain unchanged. Pin disjoint physical CPUs within
the allocation and verify effective cone/factor widths. The external driver only
extends its width whitelist to 16. GNU time RSS includes Julia and audit work.
The same Ising input has 322 variables, 20 equalities and 22 PSD blocks of orders
12–16; it is a scaling diagnostic, not evidence for large/multinode performance.

Code inspection: cone scaling and selected Schur work use the reusable worker
pool, while the MPFR condensed QDLDL factor remains serial. Existing raw receipts
report thread counts but no phase profile; infer no measured bottleneck shares
from counts alone. Use the scaling results to choose a phase profile and next
parallel implementation; avoid further hinf3 parameter sweeps without a new
structural hypothesis. Script receipt: `/tmp/sdpx-scaling-20260916/`.

The first scaling cell passes all original metrics and effective-plan checks:
1-core native first/warm times 74.964629/74.936771 s, whole-process peak RSS
669972 KiB. Other widths/reverse samples are pending; no speedup yet. External
assessor `/tmp/sdpx-scaling-20260916/assess.py` checks all six expected cells,
input identities, every point and audit, actual widths, affinity and resource
receipts. Tests reject nonfinite/out-of-bound metrics and mismatched widths.

An isolated scheduling candidate creates up to four tasks per pool worker,
bounded by cone count and existing minimum work per task. Pool thread count,
contiguous disjoint ownership, single-orthant chunking and precision are
unchanged. It targets static-lane stragglers without a new scheduler or raw
pointers. Source: `/tmp/sdpx-scaling-20260916/fine-lanes/`. The focused ownership/join test passes with four lanes on two workers,
including a failed lane whose siblings still complete. Paired full-solve
acceptance is still required before adoption; no measured gain.

Forward scaling cells all pass: native warmed times 74.936771 s (1 core),
24.660085 s (4), 21.164499 s (16). Whole-process RSS is respectively
669972/689800/731060 KiB. Reverse samples are still pending; these preliminary
values show diminishing returns beyond four cores, not a completed scaling claim.

The fine-lane candidate passes pooled condensed equivalence tests at Float64 and
256 bits, in addition to its ownership/join check. PBS **212923.node220** is
submitted with `afterok:212922.node220`, initially dependency-held. Frozen campaign:
`hpc:~/projects/sdpx-lanes-20260916-pair01`. Source comparison confirms only
`cone_parallel.rs` differs between arms; both share observation-only printing
of existing phase timers after `solve()`. Build both libraries before any timing.
Run ABBA at each width 1/4/16, first plus one warmed solve per process, 512 bits,
unchanged internal/external gates, fixed physical affinity, BLAS one thread and
process RSS. Allocation 16 cores/32 GiB/2 hours; no numerical concurrency with
212922. Results and phase shares are pending; no scheduler change is adopted.

### Completed 1/4/16-core Ising baseline

PBS 212922 completed; all six cells / twelve solves pass original-coordinate
512-bit gates and effective-thread checks. Frozen file checks pass. Warm native
medians: **74.929828 s (1), 24.698106 s (4), 21.460200 s (16)**. Relative speedups
are **3.03x** at four cores and **3.49x** at sixteen (parallel efficiencies about
76% and 22%). Peak process RSS ranges 654–662 MiB, 674–680 MiB and 713–714 MiB
respectively, including Julia/audits. This qualifies only the small 22-block
Ising input; large/multinode scaling remains unqualified. Final receipts:
`/tmp/sdpx-scaling-20260916/final/decision.json`.

Dependency 212923 has released and is running its builds. Its existing timer
output distinguishes cone phases from the overall KKT update/solve; it cannot
separate Schur assembly from factorization inside KKT. Add finer instrumentation
in a subsequent frozen experiment only if the measured phase requires it.
The external pair assessor also preserves timer hierarchy/units and refuses
missing or malformed profiles, avoiding double-counted parent/child times.

### Structural factorization opportunity

A no-solve analysis of the exact frozen Ising operator finds 11 positive-block
support components of orders 24,25,27,29,31,31,31,31,31,31,31; P has no entries.
All 20 original equalities connect all components. Dense component storage would
use 9498 scalars versus 103684 for a full 322-square positive block (excluding
border, factors and workspace). This is a storage count, not measured memory or
speed. Input/mapping identities match the qualified campaign. Evidence:
`/tmp/sdpx-scaling-20260916/structure.json`.

The existing DenseBlockSolver handles one dense positive block in Float64 only;
its density threshold excludes this sparse collection of dense components, and
MPFR uses QDLDL. Investigate generalizing that existing elimination to typed,
independent connected positive components and a retained equality border. Reuse
provider Cholesky/triangular/Gram kernels and original-operator refinement; retain
sparse fallback for failed SPD factors or unsuitable structure. Discover structure
once from symbolic support (including P), preserve updates and all equalities,
and do not introduce problem-name special cases or a separate Ising solver.
Profile evidence must still justify adoption and parallel work granularity.

### First measured phase profile (212923, partial)

The first completed baseline cell passes both original-coordinate audits at
512 bits and 50 iterations. Warm native solve is 74.498816 s: KKT direction
solve 34.092678 s (45.8%), KKT update 14.101460 s (18.9%), cone scaling
8.290972 s (11.1%). These are aggregate phase timings from one single-core
cell, not candidate gains or a parallel phase breakdown. Remaining ABBA cells
are still running; no scheduler change is adopted.

Prioritize decomposing the direction-solve phase before implementing the
component-factor proposal. `CondensedKKTSolver::solve_raw` includes two scaling
applications, sampled forward/transpose operations and a reduced solve; outer
refinement repeats this work and evaluates the original operator. The reduced
DirectLDL solve also retains its own refinement. Counts alone do not prove
redundancy: preserve both accuracy contracts. The next isolated profile should
separate reduced backsolve/refinement, sampled operations, scaling applications,
and original-operator residual work, while counting correction calls. Likewise
split KKT assembly from refactor before assigning a factorization speedup budget.
Evidence: `/tmp/sdpx-scaling-20260916/first-phase-decision.json`.


### Fine KKT diagnostic prepared

An external observation-only copy instruments twelve aggregate counters on the
solve thread, flushing after native timers finalize. Raw solve stages are scaling
in, transpose, reduced solve, forward, scaling out; additional counters cover
original residuals, assembly, regularize/refactor, reduced refinement, initial
and correction backsolves, and reduced residuals. Counters are nested where stated;
do not sum a parent with its children. No numerical operations or gates change.
The Float64 and MPFR-256 pooled condensed equivalence tests pass (2 tests).
Profile parser checks reject negative, missing and duplicate counters.

PBS **212924.node220**, initially dependency-held after successful 212923, runs
one first/warm diagnostic at 1 and 16 physical cores, sequentially, 512 bits,
unchanged tolerances, BLAS one thread. Allocation: 16 cores, 32 GiB, 45 minutes;
build timeout 900 s, each process timeout 400 s. Frozen 335 files under
`hpc:~/projects/sdpx-kkt-profile-20260916-pilot01`. Local scripts and isolated
source: `/tmp/sdpx-kkt-profile-20260916/`. Instrumented timings locate costs;
they do not establish a performance improvement. Production source is unchanged.


### Direct parallel sampled forward candidate

An isolated candidate avoids temporary forward contributions and their serial
scatter when sampled descriptors are ordered by their validated disjoint row
ranges. Safe recursive slice splitting writes each block directly into its output
rows. Unsorted descriptors retain the existing path; shared-column transpose
accumulation keeps descriptor order. One typed implementation covers Float64 and
MPFR; no input-name branch or raw pointer. Single-thread behavior is unchanged.

Six exact operator tests pass for ordered/unsorted Float64, 256-bit and 512-bit
inputs, including changing alpha/beta, gaps, empty basis blocks, overlapping
column ranges, workspace reuse and signed results. Ordered pooled forward uses
no contribution-vector storage. Two pooled condensed equivalence tests also
pass. Source and logs: `/tmp/sdpx-sampled-direct-20260916/`. This is an unadopted
candidate, pending phase evidence and full-solve performance acceptance.

212923 partial receipts now contain a complete single-core ABBA and the first
four-core baseline, all numerical gates passing. Single-core medians are
74.435811/74.055095 s (baseline/candidate); scheduling does not change the
single-thread path, so this small variation is not a scheduling gain. Four- and
sixteen-core pairs remain pending. Fine KKT profile 212924 is dependency-held.


The direct-forward candidate's short local synthetic ABBA kernel screen uses
22 ordered blocks (side 16, 31 basis columns), shared variable ranges, fixed
BLAS thread count and exact equality before/after each timed batch. Float64
kernel median reductions at 2/4/8 workers are 7.60%/14.35%/7.05%; 512-bit
reductions are -0.09%/0.17%/1.27%. This is two batches per arm on a local host,
not full-solve acceptance. The high-precision result does not justify another
Ising full-solve campaign without stronger phase evidence. Preserve the
candidate externally; do not adopt or attribute these kernel percentages to
solver speed. Receipts: `/tmp/sdpx-sampled-direct-20260916/kernel-decision.json`.

212923's complete four-core ABBA now passes all gates: baseline 24.639679 s,
candidate 24.919369 s (1.14% slower). Sixteen-core samples remain pending;
no fine-lane scheduling gain is established.


### SDPB direction-solve source review

Reviewed SDPB commit `c5bd57e9ecee5f553901e4e9370e0014f402854c`, frozen under
`/tmp/sdpx-sdpb-core-20260916`. In `compute_Q.cxx`, per-block Cholesky and
`Trsm` prepare the off-diagonal factors once per iteration. The global Gram
matrix uses the shared-memory bigint SYRK context and is Cholesky factored.
`solve_schur_complement_equation.cxx` reuses these factors: local triangular
solves and coupling products, reduction into the border RHS, border solve,
then local reverse solves. The stale initialization comment mentions LU;
the actual implementation calls Cholesky. Block allocation creates MPI process
groups from block costs; it does not run independent duplicate solvers.

This supports the existing proposal to generalize DenseBlockSolver to positive
connected components with a retained equality border, sharing typed provider
kernels and factor reuse. It does not justify dropping SDPX's original-operator
refinement or assuming equivalent conditioning to SDPB. Preserve sparse fallback
and use the pending fine counters to establish whether reduced backsolve,
refinement, or PSD/operator work dominates before prioritizing implementation.

The first 16-core baseline in 212923 passes both numerical audits at 50
iterations: warm native 20.814071 s, direction solve 8.988936 s, KKT update
5.195204 s, cone scaling 1.805868 s. These remain single-cell phase observations;
16-core candidate/reverse cells are pending. The direction phase is still the
largest named phase; there is no evidence yet for its internal dominant kernel.


### Scheduling acceptance complete — adopted

PBS 212923 completed successfully. All 12 cells / 24 solves pass original
512-bit audits with 50 iterations; 623 source and four library/native identity
checks pass. Warm native medians baseline/candidate at 1/4/16 cores:
74.435811/74.055095 s, 24.639679/24.919369 s, 21.227065/19.612755 s.
Sixteen-core improvement is **7.60%**: both candidate samples are faster than
both baseline samples. Four cores regress **1.14%**, explicitly retained in the
record; single-core variation is 0.51% with an unchanged execution path.

Adopt the exact tested `cone_parallel.rs` task-budget change: up to four tasks
per existing pool worker, bounded by cone count and minimum work. No extra
threads, unsafe ownership, precision changes, input-specific or core-count
thresholds. All 283 solver library tests pass, including Float64/MPFR parallel
and failure-joining checks. Source comparison before integration confirmed this
was the only differing solver source file. This validates this Ising campaign,
not MOSEK/SDPB parity or 64/256-core scaling. Earlier partial notes above are
superseded by this final result. Evidence:
`/tmp/sdpx-scaling-20260916/pair-final/decision.json`.

212924 has started the frozen pre-adoption fine KKT profile build. Keep its
baseline identity; do not overwrite it with the newly adopted scheduling code.

Local release FFI build of the adopted source passes; dylib SHA256
`63ee51f1c892b76a8b0607b3f10777ba67e265284cb15fbe4b499cac280b8d94`. Prior stable library remains saved externally.


### Fine KKT profile complete; residual reuse candidate

PBS 212924 completed, all four original-coordinate audits pass; 335 source and
three library/native identity checks pass. Warm instrumented times are 76.551789 s
(1 core) and 21.147831 s (16). This is the frozen pre-scheduling baseline and
not a comparative speed claim. Aggregate single/16-core seconds:

| Operation | 1 core | 16 cores |
|---|---:|---:|
| Original-operator residual, 184 calls | 15.3153 | 3.3407 |
| Raw forward, 184 calls | 4.5689 | 0.9813 |
| Raw transpose, 184 calls | 4.1050 | 0.7474 |
| Raw scaling in/out | 9.9157 | 2.1873 |
| Reduced solve, including refinement | 2.2656 | 2.6496 |
| Assembly, 51 calls | 0.2801 | 0.0772 |
| Regularize/refactor, 51 calls | 1.5856 | 1.7803 |

Reduced solve includes initial backsolves (0.5871/0.7027 s), correction
backsolves (0.2799/0.3139 s) and reduced residuals (1.3909/1.6236 s); do not
add these to its parent again. Factor-only optimization cannot explain away
the single-core gap. On 16 cores the serial reduced solve becomes significant.
Evidence: `/tmp/sdpx-kkt-profile-20260916/final/decision.json`.

An external candidate reuses the original A*x-bz vector already evaluated by
solve_raw, only for the initial residual at that same returned x. Subsequent
residuals after adding a refinement correction recompute A*x. No new scratch,
precision conversion, relaxed stopping rule or omitted residual equation.
Subtraction/accumulation order differs, so numerical acceptance remains required.
The candidate applies to sampled and ordinary condensed operators at all types.
31 condensed tests pass, including a new cached-versus-fresh rounding bound and
a poisoned-cache check for a changed point at Float64 and MPFR precisions.
Source/logs: `/tmp/sdpx-residual-reuse-20260916/`. Not adopted: next run is a
frozen full-solve ABBA against the newly adopted scheduling baseline, with the
same original accuracy gates. Keep new builds separate from prior profile arms.


### Residual reuse isolated acceptance protocol

Completed PBS212926 froze 622 files and compared ABBA at 1/16 physical cores,
512 bits and BLAS1, first plus warm solves, with unchanged original-coordinate
gates and RSS. Only condensed.rs differed between arms. All 283 candidate core
tests passed. Protocol/evidence: `/tmp/sdpx-residual-reuse-20260916/`.
Final isolated results appear below; broader adoption was decided by PBS212929.


### Ordered row-gather prototype for reduced residuals

The measured 16-core reduced residual cost is 1.624 s and remains serial.
Existing symmetric CSC multiplication visits columns and scatters into output
rows. An external symbolic row-gather prototype records `(value index, x index)`
for each row in exactly that original traversal order, including duplicates,
then processes independent rows through a supplied existing Rayon pool.
It keeps numerical values in the authoritative CSC storage and retains the
same multiply/add expression and beta scaling. No numerical atomics, parallel
reduction tree, matrix-value clone, new thread pool or precision-specific engine.

Exact parity tests pass at Float64, 256 and 512 bits with 1/2/4 workers, upper
and lower storage, duplicates, gaps, zeros, cancellation, signed outputs and
updated values reusing the same symbolic map. Source/log:
`/tmp/sdpx-row-residual-20260916/`. This is an unintegrated kernel prototype;
cache memory, work granularity, pool reconfiguration, whole-solve correctness
and performance still need verification. It is independent of the residual
forward-reuse candidate currently running as PBS 212926.


The row-gather prototype is now integrated in an external solver copy, but the
separate SymmetricRowPlan was retired after locating the existing SparseParallel
component. Move that component/tests into the algebra layer and extend its
symbolic constructor for symmetric contributions; ordinary residuals and direct
KKT refinement share the same flattened indices, weighted row partitioning,
thread-pool configuration and value-update behavior. Symmetric multiplication
retains the original multiply/add expression rather than using gemv's specialized
alpha branches. The direct solver attaches the existing cone pool; the condensed
solver forwards its current pool on each update, including reconfiguration.
A precision-weighted work threshold keeps small products serial.

All 286 solver library tests pass in this external shared implementation,
including existing pool reconfiguration and live LP/SOCP tests plus symmetric
Float64/256/512 exact parity tests. Full-solve timing, metadata-memory impact and
explicit activation coverage for the new KKT path remain pending. No second
sparse engine or extra worker pool is introduced. Stable source remains unchanged.
Evidence: `/tmp/sdpx-row-residual-20260916/shared-test.log`.


Explicit new-KKT-path activation tests now pass at Float64, 256 and 512 bits.
A 192-variable symmetric system crosses the work threshold; configure its
existing pool through 4/1/2/4 workers, assert the effective pool width and stable
metadata allocation, update P values, and compare both residual vectors and
infinity norms exactly against the original serial implementation. This adds
three focused checks beyond the 286 passing library tests. A no-SDP build check
also passes after removing an unnecessary import introduced by the module move.
Evidence: `/tmp/sdpx-row-residual-20260916/activation-test.log` and
`no-sdp-check.log`. Performance and memory qualification remain pending.

The first residual-reuse candidate cell in 212926 passes both accuracy audits;
warm single-core baseline/candidate are 74.776033/72.385951 s. These are one
sample per arm, not the final paired result. Preserve all remaining ABBA cells
before deciding adoption; the stable source still contains only the previously
accepted scheduling change.


### Parallel residual isolated acceptance protocol

Completed PBS212927 froze 624 files and compared 16-core/512-bit ABBA, BLAS1,
first plus warm solves, unchanged original gates and RSS. Both arms included
accepted scheduling and excluded forward-residual reuse, isolating the shared
row-gather change. It followed PBS212926 without numerical overlap. Evidence:
`/tmp/sdpx-row-residual-20260916/`. Final results and combined adoption follow.


### Residual reuse isolated Ising result

212926 completed: all eight cells / sixteen solves pass 512-bit original
accuracy gates, all with 50 iterations. Source/library identities: 622/four
checks pass. Native warm medians baseline/candidate are **74.451782/72.293299 s**
at one core (**2.90% faster**) and **19.371735/18.318031 s** at sixteen cores
(**5.44% faster**). Subsequent Float64 and MPFR acceptance is recorded under PBS212929 below. Evidence: `/tmp/sdpx-residual-reuse-20260916/final/decision.json`.


### Medium independent dual audit gap

The historical medium driver independently checked original equalities and
primal PSD feasibility but used solver summaries for the dual side. An external
revised driver retains returned primal/equality/PSD dual data and independently
reconstructs stationarity, dual PSD feasibility, dual objective and gap after
native timing. Its formulas pass six analytic checks (including off-diagonal
trace contributions and rejected invalid data), plus four assertions on a real
Julia frontend equality+PSD problem. No solver-side certificate stage is added.

One stable-library medium run passes the historical gate but fails the added
stationarity/max(1,norm(q,Inf)) <= 1e-6 gate: **1.91649243224e-6**. Recomputing the
saved Float64 inputs/duals at 256-bit audit precision gives the same result,
so this is not Float64 audit accumulation error. Dual PSD violation is zero;
reconstructed dual objective agrees within 5e-16, and objective gap is 8.21546e-7.
This is a stricter independent metric than the old gate, not a new speed result
or a regression attributed to an unadopted candidate. Preserve the numerical
threshold and qualify both arms under the same audit; do not promote internal
Optimal into full-accuracy credit. Audit files and saved point:
`/tmp/sdpx-residual-broad-20260916/`. The stable solver/settings were unchanged.


### Parallel KKT residual Ising acceptance complete

212927 completed. All four cells / eight solves pass 512-bit audits at 50
iterations; every serialized x/s/z vector matches the baseline exactly. 624
source and four library/native checks pass. Sixteen-core native warm median
**19.251350 -> 15.133014 s**, a **21.39% reduction**. Candidate process RSS is
716540/727728 KiB versus baseline 731116/728628 KiB; no measured peak-memory
regression on this input. This is the shared row-gather candidate alone, excluding
forward-residual reuse. Other phase times also change, so do not attribute the
entire gain solely to one residual kernel. Broader acceptance and combination
with forward reuse remain pending; no production integration yet. Evidence:
`/tmp/sdpx-row-residual-20260916/final/decision.json`.

### Medium normalization clarified

Inspection of both SDPX and the local Clarabel.rs reference confirms the same
dual criterion: norm2(stationarity)/max(1,norm_inf(q)+norm2(x)+norm2(z)). From the
saved original point and frontend dual matrices (whose Frobenius norm equals
canonical svec norm), the independent 256-bit accumulation reconstructs
**2.51277821712e-7**, agreeing with the reported 2.51278e-7. Thus the point passes
the established 1e-6 criterion. The 1.91649e-6 target-only normalization is a
stricter, different diagnostic; it is not evidence of a faulty implementation
of the upstream stopping rule or a candidate regression.

Align the new external audit with that existing criterion at the unchanged
1e-6 tolerance, retaining raw norm2, infinity norm and target-only normalization
in reports. No solver stopping rule or in-flight benchmark gate changes.
The strengthened audit also checks original dual PSD feasibility and independently
reconstructed objectives/gap. Evidence:
`/tmp/sdpx-residual-broad-20260916/normalization-check.json`. Earlier wording of
a pending precision issue applies only to the stronger target-only diagnostic.


### Combined cross-problem acceptance protocol

Completed PBS212929 froze 666 files at `sdpx-combined-20260916-accept01`.
ABBA covered LP_bore3d, SOCP_axis_wide, SDP_truss2; medium/Ising at 1/16 cores;
SOCP_nb at 16 cores; ordinary nonsampled SDP at 256/512 bits. Forty-eight
receipts include separate high-precision audit processes. All are exposed
regression cases, not holdout evidence. Runs were sequential, BLAS1, with
original gates, native/API timing separated and RSS recorded. The 16-slot PBS
allocation was pinned to 1 or 16 physical cores, never a 128-core campaign.

The frozen assessor checks raw numeric bounds, precision, widths, arm identity,
resources and missing cells. Its tests reject excessive residuals, NaN, wrong
labels and missing resources. Identities: `assessor-identity.json`; tests:
`test_assessor.py`; source review: `integration-review.json`, all under
`/tmp/sdpx-combined-20260916/`. All 289 candidate core tests passed. Final
measurements and adoption follow; superseded interim samples are omitted here.

### Deferred scaling-cache experiment

Deferred isolated PSD scaling-cache experiment (`/tmp/sdpx-scaling-cache-20260916`):
the cone already caches the authoritative upper triangle of R R^T at every
scaling update. The high-precision condensed update recomputes the same SYRK.
Replace that redundant product with a copy of the existing cache, retaining
lower-triangle mirroring and all inverse-factor calculations. Identity resets
also update the source cache. Added exact cached-versus-recomputed checks at
Float64/256/512 bits; all 290 core tests pass. The release kernel screen passes exact output equality: at 512 bits, 500
order-16 SYRK calls cost about 0.0570 s versus 0.00010 s for cache copies;
order-32 costs about 0.415 s versus 0.00063 s. This confirms eliminated work,
but the absolute order-16 saving is only about 0.114 ms per call on this host.
Defer a dedicated cluster campaign: no >=2% full-solve gain is established.
No production code has been changed. This is
based on the combined candidate, separate from the completed frozen 212929 arms.



### Combined residual optimizations accepted and integrated

212929 completed: all 48 receipts pass fixed numerical gates; 666 source and
6 library/native hash checks pass. The reviewed nine source paths are integrated
exactly, including the shared sparse-row component move; the unrelated scaling-G
cache experiment is not included. External backup and evidence:
`/tmp/sdpx-combined-20260916/{pre-integration,final-decision.json,final-build-logs}`.

| Case / physical threads | Baseline native seconds | Candidate native seconds | Reduction |
|---|---:|---:|---:|
| Ising 512-bit / 1 | 74.050580 | 70.151168 | 5.27% |
| Ising 512-bit / 16 | 19.614802 | 14.070192 | 28.27% |
| medium Float64 / 1 | 11.413268 | 11.412418 | 0.01% |
| medium Float64 / 16 | 12.474882 | 11.554621 | 7.38% |
| ordinary SDP 256-bit / 1 | 11.509830 | 11.503130 | 0.06% |
| ordinary SDP 512-bit / 1 | 27.175027 | 26.689454 | 1.79% |

Small Float64 warm API changes span -0.53% to +1.76%; SOCP_nb at 16 cores
regresses 0.41%, not a material change in this screen. Medium-16 process RSS
increases from 969404 to 1030842 KiB (~60 MiB); Ising-16 RSS is effectively flat
(724004 to 723200 KiB). Candidate Ising speedup from 1 to 16 cores is about 4.99x,
only 31% parallel efficiency; medium still has no net multi-core acceleration.
These are same-host OpenBLAS comparisons, not new MOSEK/SDPB parity evidence.
Release build passes; the Julia frontend suite passes against the new library,
including prepared handles, original-coordinate outputs, high precision through
2048 bits and thread-budget tests. Production dylib SHA256:
`8e4017e2fcbaf220950ddc2a5a2239a4244ccb16ae518fa19298a1f7fddf664f`.
The bounded phase diagnostic is running as 212931.node220 in
`hpc:~/projects/sdpx-medium-profile-20260916-pilot01` (16 cores, 32 GiB, 45 min);
it uses the integrated numerical source with observation-only counters.


212931 diagnostic failure (preserved): all three children stopped before solver
execution because the copied Julia Manifest retained the previous campaign's
absolute SDPX path. Source-origin checks rejected it; there are no valid timings.
A first scoped retry is prepared in `sdpx-medium-profile-20260916-pilot02` with
only the environment binding corrected. Reuse the exact compiled diagnostic
library SHA256 `4b2d8e8a4cd38468c81dbdf4a0cf0afcbb48b220fd3247fbdec8e51e956e6fe5`;
retain the same numeric source, inputs, settings, evaluator and three cases.
The completed performance qualification 212929 and production build are unaffected.

Retry submitted as **212932.node220**; pilot01 remains intact. No second
compilation is needed; final source/library checks remain in the PBS script.


### Medium assembly bottleneck confirmed (212932)

The corrected three-cell diagnostic completed; original numerical gates pass,
364 source and 3 library hashes pass. Evidence is
`/tmp/sdpx-medium-profile-20260916/decision02.json`.
Medium assembly costs 7.441 s at one core and 7.834 s at 16; regularized refactor
is 1.317/1.307 s. Scaling synchronization is only 0.004/0.010 s. Thus assembly,
not factor-provider replacement or cached scaling-G, is the current main target.
Ising-16 differs: assembly 0.065 s, refactor 1.803 s, outer residual 2.083 s,
raw scaling 0.811+1.000 s. Preserve distinct measured priorities.

The original medium PSD supports contain 1887/1862/1887/1862/1862 columns,
8766015 contribution cells and 1781328 union-Schur cells. Before any possible
chordal transformation this exceeds the two-Schur-cache cutoff; the complete
Float64 contribution cache is only 66.88 MiB. An isolated candidate at
`/tmp/sdpx-assembly-budget-20260916` allows the greater of the existing relative
budget and 128 MiB, counted using the scalar storage size at every precision.
It changes only parallel-buffer eligibility, retaining original arithmetic and
ordered scattering. Validate actual activation, solve time and memory before
adoption; no production code or timing claim yet.

The isolated cache-budget candidate passes all 31 condensed tests. Existing
overlap tests were updated to expect bounded parallel activation instead of
the old relative-only fallback; they still require exact Schur values and
solutions versus serial execution, unchanged memory on pool reconfiguration,
and original-operator residual checks. No numerical comparisons were relaxed.


Assembly-budget screen submitted as **212933.node220** in
`hpc:~/projects/sdpx-assembly-budget-20260916-pair01`. Freeze 652 files; exact
candidate diff is allocation eligibility plus updated overlap expectations.
Use the integrated baseline library `98efe89d…`, build candidate before timing,
then eight ABBA cells (medium-16 and Ising-512/16). Allocation 16 cores/32 GiB,
45 minutes; medium processes <=300 s and Ising <=400 s. Capture native timing,
process RSS, iteration count and unchanged original-coordinate numerical gates.
The new environment binding was checked before submission. Local evaluator and
inherited gate hashes are in `/tmp/sdpx-assembly-budget-20260916/assessment-identity.json`.
This is a development screen, not a broad qualification or parity comparison.


212933 completed with all eight numerical gates, 652 source checks and four
library/native checks passing. Medium-16 improves 11.511656 -> 8.412762 s
(26.92%), process RSS 1022812 -> 1087782 KiB (~63 MiB). Ising-512/16 regresses
14.506874 -> 15.156954 s (4.48%); both candidate samples are slower than both
baseline samples. Do not adopt yet. The original Ising block supports require
9820 contribution cells, exactly the old two-Schur allowance (2*4910), so
buffer eligibility should already be true. Check actual result identity and
run an isolated reverse-order Ising repeat using the same two frozen libraries,
without rebuilding or interleaving medium, before attributing this to code.
Evidence: `/tmp/sdpx-assembly-budget-20260916/final-decision.json` and
`final-build-logs/`. The production library remains `8e4017e2…`.


The four original Ising warm outputs have exactly identical x/s/z and all take
50 iterations. Isolated reverse-order repeat submitted as **212934.node220**
in `sdpx-assembly-budget-20260916-repeat01`: candidate/baseline/baseline/candidate,
four Ising-only processes, same frozen binaries (no rebuild), 16 cores/32 GiB,
30 minutes, per-process timeout 400 s. All 626 source/input/library identities
are frozen, with the same numerical evaluator. Preserve the first observed
regression regardless of whether this repeat reproduces it.


212934 completed: all four numerical gates, 626 source and four library checks
pass. Isolated Ising medians are baseline 14.316636 s versus candidate
14.616446 s (2.09% slower). Keep both this result and the initial 4.48% regression;
the cause is not established. The identical binaries do not reproduce the full
initial gap, but neither does the repeat establish non-regression. No further
unchanged-library repeats are planned. **Do not adopt the 128 MiB floor yet.**
Medium's repeatable 26.92% gain establishes a useful parallel opportunity.
Next investigate processing overlapping block contributions in bounded batches
under the existing relative memory budget, preserving cone-order accumulation,
rather than retaining every block's contribution buffer simultaneously. Keep
one shared implementation across scalar precisions and leave the already-qualified
small-Ising scheduling behavior intact. Any replacement needs a fresh paired
measurement and all original numerical gates; no speed prediction is accepted.


Bounded-batch replacement implemented externally at
`/tmp/sdpx-assembly-batches-20260916/source`. It retains the original all-block
cache path when eligible. Otherwise a few reusable contribution slots consume
at most two Schur value arrays; contiguous batches compute concurrently and
scatter in original cone order before reusing slots. Shared compute/scatter
helpers avoid a second numerical implementation. No precision-specific path,
new pool, atomic accumulation or approximation is introduced.
Tests compare exact Schur entries and solve vectors against serial operation,
plus original-operator residuals, A/P updates and pool reconfiguration. Scratch
addresses and capacities must remain unchanged across iterations. Initial 31
condensed tests pass; all 290 core tests pass, including the added 512-bit overlap
case. Production code remains unchanged pending measured acceptance.
The next paired measurement should group cases rather than interleave medium
with Ising, and build both arms in the same campaign to reduce ambiguities.


Bounded-batch screen submitted as **212935.node220** in
`hpc:~/projects/sdpx-assembly-batches-20260916-pair01`: rebuild both arms with
identical toolchain/BLAS settings before timing; medium-16 ABBA followed by
Ising-512/16 BAAB (eight cells). Source/environment preflight checks passed,
651 files frozen; allocation 16 cores/32 GiB/45 minutes. Keep the original
300/400-second case limits and external precision gates. This replaces the
interleaved-case measurement layout, not its historical evidence. Assessor and
inherited gates are frozen in `/tmp/sdpx-assembly-batches-20260916/assessment-identity.json`.
No production code changed; measured acceptance remains pending.


Post-freeze edge-case validation for the batch candidate: two additional local
Float64/512-bit tests vary PSD live-column counts so slots shrink and grow
between blocks, while retaining empty/non-PSD blocks, A/P updates and thread
reconfiguration. Both pass exact serial Schur/solution comparisons, original
operator residuals, and stable scratch addresses/capacities. Only local test
code changed; the frozen cluster numerical source is untouched. The pre-addition
test file is retained as `/tmp/sdpx-assembly-batches-20260916/tests-frozen.rs`;
new results are in `heterogeneous-tests.log`. Job 212935 remains in its build phase.


### Bounded-batch screen completed — not adopted

PBS **212935.node220** completed with exit 0; all eight original numerical
receipts, 651 source checks and four library/native checks pass. Both arms were
rebuilt in the same job. Medium-16 ABBA medians: 11.489994 -> 10.937234 s
(4.81% gain), RSS 1020084 -> 1048798 KiB. Ising-512/16 BAAB medians:
14.170730 -> 14.498735 s (2.31% regression), RSS 725850 -> 728454 KiB.
The candidate remains external; the production dylib still hashes to `8e4017e2…`.
Evidence: `/tmp/sdpx-assembly-batches-20260916/final-decision.json`,
`final-build-logs/` and `phase-comparison.json`. Candidate Linux SHA256:
`2166d3c092a29473b4edf1d8040f154de992038144db1cbd683e7fe1a61488e3`.

Ising median phase times regress across cone scaling (1.05008 -> 1.10190 s),
KKT update (3.97761 -> 4.02451 s) and KKT solve (5.76957 -> 5.91396 s).
This does not isolate a numerical assembly cost or prove a compiler/layout
cause. Do not attribute the whole regression to the batch scheduler. The batch
candidate extracts shared compute/scatter helpers even for the original cached
path, so isolate that transformation before further algorithm changes. For
medium, inspect available workers per actual batch rather than assuming the
whole-problem block count represents simultaneous work. Any revised scheduling
must retain the same pool, exact cone-order accumulation and bounded memory.


### Batch-local sparse column scheduling — numerical prototype

External candidate `/tmp/sdpx-batch-lanes-20260916/source` assigns sparse-column
lanes using the reusable contribution-slot count rather than all PSD blocks.
Within a batch, independent block tasks can expose disjoint sparse output-column
tasks to the same Rayon pool; no extra threads, atomics or changed reduction order.
The original fully cached dispatch is unchanged relative to the previous batch
prototype. This is not yet an isolation of that prototype's shared-helper change.

Five overlapping 64-dimensional sparse PSD blocks exercise two contribution
slots and 2/4/8/1/4 worker reconfiguration. Float64 and 512-bit tests require exact
serial Schur equality after A updates, stable plan/output storage on repeat,
and the expected bounded lane count. Both targeted tests and all 294 core tests
pass (full suite 5.79 s). Initial test compilation used a non-Copy enum in an
array repetition; the test constructor was corrected without numerical changes.
Evidence: `identity.json`, `tests-fixed.log`, `full-tests.log` in that namespace.
No solve-time improvement is claimed; source remains external pending a bounded
paired screen and unchanged original-coordinate medium/Ising gates.


Batch-local lane candidate submitted as **212936.node220** in
`hpc:~/projects/sdpx-batch-lanes-20260916-pair01`. Both arms rebuild before
medium-16 ABBA and Ising-512/16 BAAB, eight cells on node49 with 16 cores,
32 GiB and 45 minutes. Existing 300/400-second process limits and all original
numerical gates remain fixed. Preflight verified exactly two candidate source
changes against the prior campaign, matching local tested hashes, the Julia
package's namespace binding, and 651 frozen files (generated Python bytecode
excluded). Submission initially queued. Assessor/driver identities are in
`/tmp/sdpx-batch-lanes-20260916/assessment-identity.json`.
The previous batch regression remains recorded; this submission does not
qualify that candidate or establish performance parity.


### Dense assembly attribution prepared

Static original-medium support analysis finds 1475–1504 dense-path columns per
PSD block versus 374–387 sparse-path columns. The current structural cost model
assigns only 0.20–0.37% of work to sparse pairs. This is before presolve/chordal,
not measured runtime attribution; it limits expectations for the sparse-only
batch-lane candidate. Evidence: `/tmp/sdpx-batch-lanes-20260916/medium-column-estimate.json`.

External diagnostic `/tmp/sdpx-dense-phases-20260916/source` instruments exact
column planning, dense transform/packing, tiled dense dots, tiled output stores
and sparse pairs. Atomic counters aggregate across workers; durations are summed
worker spans and overlap with selected-total parents, not additive wall time.
No numerical operators or production files changed. All 31 affected condensed
tests pass; an initial import-before-module-doc error was corrected before testing.

PBS **212937.node220** submitted with `afterany:212936.node220` and initially held
for that dependency. It runs medium-1, medium-16 and Ising-512/16 after the paired
screen, using the existing 16-core/32-GiB/45-minute allocation and case timeouts.
Preflight confirms exactly three instrumented source files, correct namespace
binding and 363 frozen files. Evaluator retains original numerical gates and
checks complete, balanced phase receipts. Parser checks accept complete counters
and reject a missing counter. Driver/evaluator identities are frozen locally in
`assessment-identity.json`; no performance conclusion is available yet.


### Research after repeated assembly candidates stalled

The [SDPARA sparse-Schur paper](https://optimization-online.org/wp-content/uploads/2010/09/2732.pdf)
(2010, §4.1–4.2) distributes stored Schur elements by estimated formula cost,
including reusable intermediate construction, and aligns sparse assembly storage
with the factorization input. This supports evaluating work-balanced dense
output tiles rather than equal PSD-block counts. That application is an SDPX
hypothesis, not a result of the paper. Preserve fixed per-entry arithmetic and
ordered cross-cone accumulation. First obtain 212937's actual dense-stage costs;
choose tile ownership and bounded scratch only for the measured dominant stage.

The newer [SDSL-Solver preprint](https://arxiv.org/html/2604.23979v1)
(April 2026, §3–4) uses Block Jacobi/BBD decomposition and hybrid MPI+OpenMP.
Its filtering/diagonal corrections apply to a preconditioner while Krylov uses
the original operator. Some reported NetworkPlan comparisons are individual
linear solves because reference IPM runs fail, not full optimizer comparisons.
Its 1e-8 linear-residual experiments do not establish SDPX's original-coordinate
accuracy or arbitrary-precision performance. Keep BBD/local elimination as a
future multi-node reference; do not introduce a filtered Krylov backend while
medium is assembly-bound and previous iterative candidates failed their gates.
Do not infer MPI, high-precision or MOSEK/SDPB parity from these source results.


### Batch-local lane screen completed; dense diagnostic repair

PBS212936 completed all eight numerical receipts, 651 source checks and four
library/native checks. Medium-16: 11.523612 -> 11.221488 s (2.62% gain), RSS
1020542 -> 1048454 KiB. Ising-512/16: 14.885211 -> 14.791965 s (0.63% gain),
RSS 720598 -> 719916 KiB. Candidate Ising samples span 14.455678–15.128252 s,
straddling both baseline samples; do not interpret 0.63% as established gain.
Medium samples both improve, but the current candidate still needs broader
qualification and resolution of earlier batch-path regressions before adoption.
Keep production unchanged and await dense-phase evidence rather than launching
another unchanged-binary repeat. Evidence: `/tmp/sdpx-batch-lanes-20260916/final-decision.json`
and `final-build-logs/`. Candidate Linux hash:
`32e2e9e0bf3324829eb0fe818c3d699157bf6daadb99e37f7e769f1851da4e9b`.

PBS212937 failed before any solve (exit1, 9 s): its copied retry script assumed
an existing candidate library and omitted compilation. `libraries/*` was absent.
No numerical or timing evidence resulted. Preserve `sdpx-dense-phases-20260916-pilot01`.
Repair in a fresh pilot02 namespace restores the explicit bounded release build
before library hashing and timing, retains identical instrumented source and
numeric gates, and rebinds the Julia environment. This is the first diagnosed
retry; a script parse check alone was insufficient to detect the omitted build.


Repaired dense-phase pilot submitted as **212941.node220**, initially queued.
Preflight verified the explicit build/copy/hash/solve ordering, unchanged numerical
source versus pilot01, correct package binding and 363 frozen files. Failed logs
are retained locally in `/tmp/sdpx-dense-phases-20260916/failed01`; retry script,
manifest, job ID and assessor identities are in `retry02/`. No duplicate live
diagnostic job remains.


### Bounded dense-dot row prototype

External source `/tmp/sdpx-dense-rows-20260916/source` splits independent rows of
an existing dense accumulation tile using the solver's current Rayon pool.
It retains one arithmetic helper for serial/parallel execution, original
entry/FMA order, unchanged transform packing and serial Schur publication.
No extra numeric scratch, new pool or atomics. Activation is restricted to the
serial-assembly fallback with an explicit configured pool and a sufficiently
large tile; the original all-block cached path remains outer-parallel. MPFR's
unbatched path is unchanged. This is separate from the batch-buffer candidates.

All 290 core tests pass, including a 513-column dense fixture requiring exact
serial/parallel results across 2/4/1/4 workers, coefficient/representative changes,
and stable accumulator allocation. Existing 150-column compact-storage tests
remain unchanged. The wider fixture initially inherited an invalid <50% packing
assumption; only that fixture-specific assertion was replaced with the full
panel bound, with numerical equality and independent matrix checks retained.
Logs and source identity are in the external namespace. No production change or
performance claim; await the frozen dense-phase profile before a timing campaign.


### Dense-phase attribution completed (212941)

The repaired diagnostic completed with exit0, all three numerical gates, 363
source checks and three library/native checks passing. Medium-1 spans (seconds):
selected assembly 7.43924; transform/pack 4.51821; dense dots 1.75298; publication
0.93777; sparse pairs 0.19876; exact-column planning 0.03121. Medium-16 shows
4.64422/1.82327/0.96039 s for transform/dot/publication, respectively. Its assembly
remains serial in this baseline, so additional pool threads do not accelerate it.
Ising sampled assembly does not enter the dense-dot path. All counts complete;
evidence: `/tmp/sdpx-dense-phases-20260916/final-decision.json` and
`retry02/build-logs/`. This instrumented run is not a speed comparison.

Transform/packing is the largest hotspot (about 61% of selected medium assembly).
The ready row-parallel prototype targets about 24%, so its potential is limited;
run one bounded whole-solve screen before deciding whether to keep it. Further
work should separate coefficient-product generation, GEMM and packing within
the dominant transform phase. Exact-column hash caching and sparse-only lane
tuning are low priorities given these actual measurements. Preserve all earlier
M4 and batch-candidate rejection evidence; do not repeat those unchanged.


Dense-row screen submitted as **212942.node220**, initially queued, namespace
`hpc:~/projects/sdpx-dense-rows-20260916-pair01`. Preflight confirms only condensed.rs
differs numerically between arms and matches the locally tested SHA256
`84137b1b73cf7db79d8069d535e7f501aa41eae6ca3d1267f4b8907762b15b73`.
Freeze 651 files; build both arms before medium-16 ABBA and Ising-512/16 BAAB.
Keep 16 cores/32 GiB/45 minutes and 300/400-second case limits, BLAS1, original
numeric gates and RSS. Assessor identities are frozen in the local namespace.
No production adoption; broader precision/one-core qualification remains required
if this screen produces a repeatable gain without regression.


### Transform substage diagnostic submitted

External `/tmp/sdpx-transform-phases-20260916/source` extends the qualified
observation-only dense probe with coefficient-product, GEMM and suffix-packing
counters. All 31 affected condensed tests pass; parser checks reject missing
GEMM counters. The evaluator requires matching chunk counts for the three new
stages and retains all original numerical gates. Parent/child worker spans
remain overlapping diagnostics, not additive wall-time or acceptance timings.

PBS **212943.node220** is dependency-held after **212942.node220** in
`hpc:~/projects/sdpx-transform-phases-20260916-pilot01`. Reuse the established
three diagnostic cases and 16-core/32-GiB/45-minute allocation. Preflight verifies
explicit build-before-timing, the correct package binding, exactly two additional
instrumentation-file changes from 212941 and 363 frozen files. Local identities
are in `identity.json` and `assessment-identity.json`. No production change;
the row-parallel screen remains running, and transform optimization awaits these
substage measurements rather than a speculative change of BLAS provider.


### Additional independent-family sources located

The official [DIMACS archive](https://archive.dimacs.rutgers.edu/Challenges/Seventh/Instances/)
provides copo14 (copositivity) and filter48_socp (PAM filter design). Downloaded
only these small first-family members, pinned compressed/raw SHA256, and inspected
SeDuMi metadata without solver construction or execution. copo14 has 1275 equations,
364 nonnegative coordinates and fourteen order-14 PSD blocks; filter48_socp has
969 equations, 931 nonnegative coordinates, one SOC49 and one PSD48. Despite its
name, filter48_socp is mixed SDP/SOCP, not a pure SOCP holdout.

Files/provenance: `/tmp/sdpx-dimacs-reservation-20260916/`. No matching case/family
was found in current catalog or searched sibling benchmark manifests, but full
exposure review and independently checked SeDuMi-to-svec conversion remain
necessary before catalog registration. No performance or generalization claim.
GitHub tree API inspection was rate-limited; the official DIMACS source supplied
the data instead. Large FIR files from CBLIB were not downloaded simply to fill
coverage; keep development budgets short. Pure SOCP holdout coverage remains open.


### Dense-row parallel candidate rejected (212942)

All eight original numerical receipts, 651 frozen source checks and four
library/native checks pass. Medium-16 regresses 11.659576 -> 11.960167 s
(2.58%), RSS 1000076 -> 1020138 KiB. Ising-512/16 regresses 14.215686 ->
15.244187 s (7.23%), RSS 725444 -> 725862 KiB. Reject the candidate; do not
integrate or run broader acceptance. Evidence: `/tmp/sdpx-dense-rows-20260916/final-decision.json`
and `final-build-logs/`. Production remains unchanged.

Ising never enters the batched dense-dot branch, so its regression is not
explained by dense-row parallelism itself. The candidate changes a shared generic
PSD workspace and helper structure; neither layout/code generation nor host
variation is established as the cause. Do not claim either without isolation.
Avoid further modifications of this rejected candidate while transform-stage
profile 212943 runs. Prefer minimal local kernel changes for the next hypothesis.

DIMACS candidate conversion has progressed without a solver run: copo14 maps to
1834 variables/3109 rows/4578 nonzeros; mixed filter48_socp maps to
2156 variables/3125 rows/48262 nonzeros. External `convert.py` forms a symmetric
upper-svec map, retains all original equations/cones, and checks its operators
and objective against independently reconstructed full symmetric matrices at
four deterministic random points per instance. All checks and existing finite
CSC/cone-shape validation pass. Source/JSON hashes, versions and errors are in
`/tmp/sdpx-dimacs-reservation-20260916/conversion.json`. These remain candidate
reservations, not registered fresh holdouts or evidence of solver success.


### Transform substage profile completed (212943)

All three original numerical gates, 363 source checks and three library/native
checks pass. Medium-1 transform/packing 4.47911 s divides into coefficient
products 1.32230 s, GEMM 2.74434 s and suffix packing 0.39802 s (1444 balanced
chunks); remaining setup is small. Medium-16 gives 1.39995/2.88572/0.41469 s,
respectively. Ising does not enter these stages. Source identities and receipts:
`/tmp/sdpx-transform-phases-20260916/final-decision.json`, `final-build-logs/`.
Do not compare this instrumented single run as an optimization gain.

Prioritize GEMM/input generation, preserving per-entry accumulation and the fixed
BLAS/thread budget. Evaluate a bounded panel or provider-kernel change without
retaining all cone Schur contributions or changing the generic PSD workspace
layout. Packing alone is too small to justify another architecture change.

### DIMACS holdouts registered without solving

Registered `SDP_copo14` and mixed `SDP_filter48_socp` as holdout-only, with source
and JSON hashes, DIMACS attribution, independent family/exposure notes and
180-second/4096-MiB per-process planned budgets. No solver constructed or run.
The available SDPX/sibling/reference benchmark text and manifests contain no
prior family names; deleted history is explicitly outside that evidence.
Pure SOCP coverage is still missing. Do not use these reservations for tuning.

`benchmark/research/import_sedumi.py` supplies reusable, hash-checked import with
optional NumPy/SciPy. Two hand-written convention/error tests and eight catalog
integrity/materialization tests pass. Its two outputs match the independently
audited external conversions byte-for-byte. Prior regression sets are unchanged;
all holdouts are integrity checked, not merely the first entry. No runtime solver
validation or certificate stage was added.


### Native GEMM output-column prototype

External `/tmp/sdpx-panel-gemm-20260916/source` extends the existing `xgemm_pool`
provider method for native Float64/Float32 by splitting disjoint output columns.
The condensed serial-assembly fallback passes its existing pool to batched GEMM;
all-block outer assembly still passes no inner pool. No PSD field/layout change,
new pool, numeric scratch or altered convergence rule. MPFR's existing provider
override remains unchanged. Tile zero retains serial semantics; one-worker and
small/degenerate cases retain the opaque call. The screen must keep BLAS1 to
avoid nested vendor threads.

All 290 core tests pass (5.30 s), including native transpose combinations,
nontrivial leading dimensions, alpha/beta accumulation, short final tiles,
1/4-worker and tile0/4/8 dispatch, and untouched padding/trailing sentinels.
The provider may choose a different microkernel for a smaller output width;
no bitwise-equivalence or whole-solve speed claim is made. Original-coordinate
acceptance remains required. Local source identities and logs are frozen in
`identity.json` and `full-tests.log`; production source is untouched.


Native panel GEMM screen submitted as **212944.node220**, initially queued, in
`hpc:~/projects/sdpx-panel-gemm-20260916-pair01`. Preflight verified the three
changed files against locally tested hashes, Julia binding and explicit build
ordering; 651 files frozen. Build both arms, then medium-16 ABBA and Ising-512/16
BAAB under BLAS1, 16 cores/32 GiB/45 minutes and unchanged 300/400-second case
limits. Assessor/driver hashes are recorded in the external namespace. No
production adoption or speed conclusion; broader acceptance follows only if
this complete paired screen qualifies.


### Coefficient-product fusion prototype

Independent external source `/tmp/sdpx-coefficient-fusion-20260916/source` groups
four consecutive contributions to the same coefficient-product output column,
retaining the exact scalar FMA order while reducing repeated accumulator loads
and stores. Only the existing generic coefficient kernel changes; no pool,
workspace-layout change, tolerance or new precision path. All 292 core tests pass;
new Float64/256/512-bit checks compare exact full buffers with the original loop,
including offset/stride padding and remainder groups.

A short release ABBA kernel screen (0.22 s total) passes exact output checks:
f64 n57 baseline 13.10/8.49 ms versus candidate 6.37/5.76 ms; n116 baseline
27.98/24.00 ms versus candidate 15.67/15.48 ms. Baseline drift is visible; do not
turn these into an end-to-end gain claim. At 512 bits/n16 both are about 24.5 ms,
with no clear gain. The dense synthetic kernel overstates medium coverage:
original-support static analysis puts 48.5–54.2% of dense-path plan entries in
four-entry groups, before presolve/chordal/exact reuse. No input-specific branch.
Evidence: `tests.log`, `kernel.log`, `medium-coverage.json`, `identity.json`.
This candidate is separate from running panel-GEMM acceptance; neither is merged.

Research library follow-up: the entire 70-test suite passes after adding DIMACS
reservations/import support (`/tmp/sdpx-dimacs-reservation-20260916/research-suite.log`).
No holdout solve was performed during these integrity/tool tests.


Coefficient fusion screen submitted as **212945.node220**, dependency-held after
212944, in `hpc:~/projects/sdpx-coefficient-fusion-20260916-pair01`. This isolates
the coefficient kernel against the integrated stable source: medium **one core**
ABBA, Ising-512/16 BAAB, BLAS1, eight cells. Preflight caught an ineffective
shell-quoted driver replacement before submission; editing/transferring the
local Python driver fixed it. The final check confirms Julia/solver/affinity
width1 for medium, unchanged width16 for Ising, exact tested candidate hash,
correct package binding and 652 frozen files. Build both arms before timing;
16-core/32-GiB/45-minute allocation with unchanged 300/400-second process limits.
Assessor expected labels were fixed before reading results; all original gates
are inherited unchanged. Broader nonsampled high-precision acceptance remains
necessary if the screen qualifies. Production is unchanged.

Panel-GEMM interim: all four medium gates pass, but median time rises
11.381786 -> 13.404283 s (~17.8% slower). Seven of eight receipts are available;
final Ising candidate remains pending. No adoption. A possible next hypothesis
is transposing the batched product/output so split GEMM calls share the small
scaling factor while reading disjoint large-panel regions. The present split
shares the whole large left operand across all calls; repeated BLAS packing is
plausible but not measured. Any layout experiment must account for packing and
publication costs, preserve numeric acceptance, and add no generic workspace
fields. Do not claim this hypothesis explains the observed slowdown yet.


### Native column-split GEMM rejected (212944)

All eight numerical receipts, 651 source checks and four library/native checks
pass. Medium-16 regresses 11.381786 -> 13.404283 s (17.77%), RSS 1010824 ->
1055380 KiB. Ising-512/16 is 14.510667 -> 14.520453 s (0.067% slower), RSS
721296 -> 729550 KiB. Reject the candidate; do not infer a meaningful Ising
change. Evidence: `/tmp/sdpx-panel-gemm-20260916/final-decision.json` and
`final-build-logs/`. Coefficient fusion 212945 has started independently.

### Transposed-panel numerical prototype

External `/tmp/sdpx-transposed-panel-20260916/source` stores each transformed
panel as the transpose, computing Ginv^T * panel^T rather than panel * Ginv.
This exposes many output columns with a small common left factor, avoiding the
previous split's shared large operand. Packing indices change consistently;
allocation size is unchanged and there are no new PSD workspace fields.
The original precision and provider API remain; changed BLAS microkernels may
reassociate floating-point operations, so original-coordinate gates remain
required. All 290 core tests pass. A local release diagnostic includes both
GEMM and packing at dimensions57/116 with requested BLAS1 and pool1/4/16; no
whole-solve claim or cluster submission before inspecting that diagnostic.


### Coefficient fusion screen completed (212945)

All eight original-coordinate numerical gates, 652 frozen-source checks and
four library/native checks pass. Medium-one-core median 11.399926 -> 11.403569 s
(0.032% slower, no gain); Ising-512/16 median 14.648398 -> 14.322739 s
(2.223% reduction). The Ising result barely clears the screening threshold;
repeatability and nonsampled high-precision coverage are still unqualified.
Do not integrate or attribute the gain to the kernel on these two samples.
RSS medians are 873426 -> 909726 KiB for medium and 729546 -> 727990 KiB
for Ising. Evidence: `/tmp/sdpx-coefficient-fusion-20260916/final-decision.json`
and `final-build-logs/`. Production remains unchanged.

### Isolate transposed layout from inner threading

The local transposed-panel release diagnostic passed every packed-coordinate
comparison, including packing costs. Sixteen workers regressed at both tested
sizes; one-worker results were better, with visible baseline drift. These are
short Accelerate microbenchmarks, not evidence of Linux whole-solve gains.
A separate external candidate `/tmp/sdpx-transpose-serial-20260916/source`
changes only the batched product orientation and matching pack indices in
`condensed.rs`, preserving allocation size and the existing provider/threading
interfaces. It excludes coefficient fusion and all native pool overrides.
All 289 core tests pass (5.41 s); source identity is recorded alongside the
candidate. Next qualify this layout with fixed-input whole-solve timing before
integration; no production change or speed claim yet.


Transposed-serial whole-solve screen submitted as **212946.node220**, confirmed
running, in `hpc:~/projects/sdpx-transpose-serial-20260916-pair01`.
Medium-one-core ABBA and Ising-512/16 BAAB; both libraries rebuilt before timing,
BLAS1, unchanged 300/400-second solve limits, 16 reserved physical cores,
32 GiB and 45-minute cap. Preflight verified the sole changed numerical file,
its tested SHA-256, package binding, driver widths and 652 frozen files.
The copied assessor and eight expected labels are fixed before results.

Coefficient-fusion attribution review: `sampled_adapter.jl` supplies a factor
for every active Gram cone; `compute_schur_selected` returns from its sampled
branch before either `coefficient_product` call. The recorded Ising route is
`sampled_factors` / `condensed_sampled_qdldl`. Thus the observed 2.223% Ising
difference does not demonstrate the changed kernel's benefit. Medium, the
actual target, is unchanged. Do not spend an unchanged Ising repeat to qualify
this candidate; shelve it unless evidence from an affected workload provides
a new reason. The generic high-precision kernel screen was also essentially
flat. No integration is warranted by these results.


### Residual stage overlap prototype

External `/tmp/sdpx-residual-overlap-20260916/source` evaluates the independent
original-operator residual products and H*z scaling concurrently on the same
existing Rayon pool when scaling already has multiple lanes. Each stage retains
its original arithmetic and internal block schedule; final residual addition and
norm remain after the join. Single-worker/fallback execution remains sequential.
No new pool, workspace field, numeric buffer, precision change or extra residual
acceptance criterion. This targets block-tail imbalance in the measured Ising
outer-residual phase (about 2.08 s), not a claim of removing its whole cost.

All 289 core tests pass (5.34 s, `final-tests.log`). Extended pooled-equivalence
tests compare every residual entry and infinity norm exactly between serial and
parallel implementations for both fresh and reused-forward paths, Float64/256
bits, pool reconfiguration and assembly fallback. Existing sampled solve tests
also pass at Float64/256/512 bits. An initial mistyped test filter matched zero
tests; the full suite was then run and verified to include the extended tests.
Source identities are stored in `identity.json`. This candidate is independent
of transpose/fusion changes and awaits whole-solve timing; production unchanged.
212946 transposed-layout campaign remains running on its original job handle.


Residual overlap screen submitted as **212947.node220**, dependency-held after
212946. Namespace `hpc:~/projects/sdpx-residual-overlap-20260916-pair01`;
medium/Ising both 16 physical workers, BLAS1, ABBA/BAAB, eight cells and unchanged
external gates. Preflight verifies the exact two tested files, 653 frozen files,
package binding and driver. Both arms build before timing; established
16-core/32-GiB/45-minute allocation and 300/400-second solve limits. The same
accepted stable baseline is used, with no transpose or coefficient-fusion edits.

212946 interim: all four medium numerical gates pass; native medians
11.388608 -> 11.243528 s (about 1.27% reduction), below the 2% retention threshold.
Ising cells remain incomplete; no final decision or integration. API/process
wall time includes first-run startup and is not substituted for native timing.


### Transposed serial layout: no qualifying whole-solve gain

212946 completed: all eight original-coordinate gates, 652 frozen-source and
four native/library checks pass. Medium-one-core 11.388608 -> 11.243528 s
(1.274%); Ising-512/16 14.415735 -> 14.220289 s (1.356%). Both are below the
retention threshold, and sampled Ising bypasses the changed batched product.
Do not integrate. `/tmp/sdpx-transpose-serial-20260916/final-decision.json` and
`final-build-logs/` retain full evidence. Dependent residual-overlap job212947
has started on its original handle.

### Larger Ising conditioning diagnosis (no solve)

Reviewed the existing iteration94, 768-bit Lambda11 replay: factor/CSC products
agree to roughly 1e-175 or better, while sampled equation1099 fails its unchanged
component-relative gate at 2.16e-22. This is not evidence of a mapping bug.
A new sealed-input diagnostic `/tmp/sdpx-lambda11-scales-20260916/summary.json`
loads the same input and saved point at768bits, without native build, solve or
EVD. Across1099 columns, nonzero A column infinity norms range 1.288231e-25 to
10.13978;162 are below1e-8. Objective coefficients span2.907108e-92 to13.32651.
The largest saved |x| is7.834819e78. Even |x_j|*||A_j||_inf reaches2.899914e58;
simple column normalization alone does not remove the solution's enormous
scale. Worst-audit column1099 has norm1.171106e-24, q=-2.907108e-92 and
x=5.291024e41. Existing cumulative Ruiz bounds are1e-4/1e4, but extending them
is not established as a cure: upstream residual norms explicitly undo Ruiz.
Investigate near-null directions/cancellation and alternate exact model
representations before another costly Lambda11 solve. Preserve all original
external gates and runtime stopping rules; no approximate rank removal or
point-dependent tolerance presented as a uniform guarantee.


SDPB comparison refinement: retrieved only the14 retained x-block files from
qualified job212626, verifying every SHA against its original provenance before
reading1099 coefficients in numeric block order. Its max |lambda| is1.515307e79,
versus7.834819e78 for the SDPX saved point; the worst SDPX audit column1099 has
SDPB lambda1.515307e79. Thus huge primal multipliers are also present in a valid
SDPB solution, not evidence by themselves of an SDPX defect or disposable null
space. Different optimal primal points need not match coefficientwise.
Evidence: `/tmp/sdpx-lambda11-scales-20260916/point-comparison.json`.

Source audit of SDPB `compute_dual_residues_and_error.cxx` finds an MPI maximum
of absolute residuals; `compute_feasible_and_termination.cxx` compares that value
directly with its threshold. It is not the external component-relative test.
Its pmp2sdp basis already incorporates sqrt(sample_scalings), which is preserved
in the shared Q input; adding the same scaling again is not a new optimization.
Equal numeric internal tolerances do not mean equal accuracy across these two
solver formulations. Keep fixed original-equation gates; do not copy an SDPB
absolute gate into SDPX or claim that a larger Ruiz cap solves the discrepancy.
Next conditioning work needs equation/trajectory evidence, rather than removing
large variables or treating their magnitude as failed convergence by itself.


### Isolate generic x86 FMA-call overhead

Read-only disassembly of the accepted Linux baseline identifies the Float64
`coefficient_product` loop calling through r15 for every scalar mul_add.
Relocation0xaecde8 resolves to local symbol `fma` at0xad5420, whose compiler
builtins implementation has runtime FMA dispatch. AVX FMA instructions elsewhere
in faer's kernels do not prove these generic scalar loops were vectorized.
This is direct code-generation evidence, not yet a measured speedup.

PBS **212948.node220**, dependency-held after212947, compares byte-identical
stable source arms with only candidate Rust flag `-C target-feature=+avx2,+fma`.
No fast-math, changed precision or arithmetic expression. Before building/running,
the compute-node script requires AVX2/FMA CPU flags; it records both RUSTFLAGS,
CPU details and library identities. Both arms build before medium-one-core ABBA
and Ising-512/16 BAAB with the unchanged external gates.652 frozen files,
16cores/32GiB/45min,300/400-second solve caps. Local evidence/scripts:
`/tmp/sdpx-fma-target-20260916`; remote matching `-pair01` namespace.
This is a CPU-specific experiment, not a portable distribution change. If it
wins, separately qualify a portable runtime-dispatched implementation or an
explicit supported CPU build; do not ship an AVX2-only library as universal.

An exact-rational raw-medium proportional-column census also completed without
solving: beyond sign-opposite reuse, only25/24/25/24/0 additional representatives
can be removed across the five blocks. Most apparent opportunity repeats the
already rejected signed reuse candidate. No new proportional-reuse branch is
justified by this raw, pre-Ruiz upper-bound screen. Evidence:
`/tmp/sdpx-proportional-screen-20260916/result.json`.


Residual-overlap212947 completed: all eight numerical gates,653 source and four
native/library checks pass. Medium16 median11.544091 ->11.335211s (1.81%);
Ising512/16 median14.440208 ->13.322176s (7.74%). This qualifies for broader
acceptance and a matched repeat across relevant worker counts, not immediate
integration or SDPB parity. Frozen full evidence:
`/tmp/sdpx-residual-overlap-20260916/final-decision.json`, `final-build-logs/`.
CPU-target212948 has started; keep its independent candidate separate.


Residual-overlap full acceptance submitted as **212949.node220**, confirmed
held after212948; remote `sdpx-residual-overlap-20260916-accept01`, local
`/tmp/sdpx-residual-overlap-20260916/acceptance`.666 frozen files; only the two
locally tested numerical/test files differ between arms. Both FFI libraries and
ordinary high-precision executables rebuild before timing with the same generic
CPU/BLAS configuration.56 predeclared receipts extend the former48-cell gate
with medium/Ising at4workers; original tolerances and metric limits unchanged.
The ordinary256/512-bit benchmark remains single-worker; do not present it as
nonsampled multiworker performance coverage. Pool numerical equivalence has
separate core-test evidence. Assessor negative checks pass and expected labels
are unique; hashes frozen locally before reading results. Allocation remains
16cores/32GiB,2-hour acceptance cap and per-process timeout limits. No holdout
input is consumed by this development regression.


CPU-target212948 interim: all four medium-one-core gates pass. Native median
11.416334 ->8.883278s (~22.2% reduction). Ising remains incomplete; no final
adoption or comparison to MOSEK. Disassembly confirms coefficient_product now
uses vfmadd213pd/sd directly, replacing per-element calls to the compiler-builtins
FMA dispatcher. Same source/settings/provider; this is a build-target benefit,
not a changed algorithm. Partial receipts:
`/tmp/sdpx-fma-target-20260916/current-decision.json`.

External portable prototype `/tmp/sdpx-runtime-fma-20260916/source` keeps a single
inlined dense-Schur arithmetic body. Its existing method dispatches once per
block to an AVX2/FMA target-feature wrapper only when both runtime CPU checks
pass; other platforms retain the generic body. No new dependency, workspace,
fast-math, per-element feature checks or duplicate algorithm. All289 library
tests pass on local ARM (5.44s), plus the changed compact-panel test. This only
validates fallback behavior: x86 compile/accelerated execution remain pending.
The compact-panel test now compares generic and accelerated outputs bitwise
when running on supported x86, including sparse skipping and changed widths.
The coefficient-product helper is not force-inlined in this prototype; inspect
its generated code and measure whole solves before assuming it captures all
of the whole-library target flag's gains.
Rust's documented runtime detection/target-feature pattern:
https://doc.rust-lang.org/stable/core/arch/
and https://doc.rust-lang.org/stable/reference/attributes/codegen.html .
Do not distribute the experiment's AVX2-only whole library as universal.


### CPU-target screen complete; portable test queued

212948 completed with all eight numerical gates,652 source checks and four
library/native checks passing. Medium-one-core11.416334 ->8.883278s (22.188%);
Ising512/16 14.662893 ->14.607267s (0.379%, no meaningful gain). Evidence:
`/tmp/sdpx-fma-target-20260916/final-decision.json`, `final-build-logs/`.
This establishes a qualifying CPU-specific Float64 build improvement only;
not portable adoption, algorithmic gain, or new MOSEK/Clarabel comparison.

Portable runtime dispatch screen **212950.node220** is held after full residual
acceptance212949. Namespace `sdpx-runtime-fma-20260916-pair01`;652 frozen files,
sole candidate file SHA378ae4d05050289479d1d7b0638f729c4e3ea7cd30a7c3564290aa6d8d3b0d3b.
No global target-feature flag. On the compute node, require actual AVX2/FMA
support and run the release compact-panel test comparing both bodies bitwise,
requiring exactly one passing test before rebuilding both FFI arms and timing.
Original medium-one-core ABBA, Ising512/16 BAAB, numerical gates and limits stay
fixed.16cores/32GiB/45min;900-second bounded build/test processes. This candidate
excludes residual overlap. Do not combine apparent percentages from independent
experiments; any combined release needs its own validation.


### Pure SOCP holdout reserved without solving

Added official CBLIB `strictmin_2D_43_dual` (geometric ARAP distortion) as
holdout-only `SOCP_strictmin_2D_43_dual`:101676 variables,111757 rows,
285726 A entries and10080 SOC5 blocks plus equalities. Source inspection found
no integer, PSD or power cones. Existing repository/reference exposure search
found no matching name/family; no claim about deleted history. Several rejected
selection candidates had integer/power/PSD variables or excessive download size;
none were solved or silently relaxed.

Conversion uses existing MOI/SDPX affine export without solver setup. Independent
CBF parsing verifies the full equality coefficient/constant multiset up to
row permutation/sign, exact objective and four deterministic SOC embeddings.
An initial verifier assumed equality rows came first; inspection showed MOI
places SOC rows first, and correcting that row layout made checks pass without
changing converted coefficients. The earlier environment lacked MOI as a direct
load dependency; the retry used the existing current-project/environment stack,
without adding packages. Compressed payload, provenance and CBLIB license now
live in the research holdout directory. All70 research tests pass. Evidence:
`/tmp/sdpx-socp-reservation-20260916/`. Reserve180seconds/4096MiB explicitly at
final acceptance; metadata does not override runner limits. Never use this case
for current optimization timing or tune on its final outcome.


### Enforce reserved per-case budgets at execution

`benchmark/research/run.py` now clamps process timeout and memory to any
reserved case cap, without extending a shorter caller/campaign limit. Effective
limits are saved in each process receipt. Invalid/nonfinite/nonpositive or
boolean caps are rejected before launch. This supersedes earlier notes that
reservation budgets were only descriptive: the four current holdouts now
actually cap execution at180seconds/4096MiB under this runner. External reference
adapter still rejects holdouts, so no reference path silently bypasses this cap.

All71 research tests pass (0.64s), including an execution-boundary test that
checks what reaches the owned-process supervisor, receipt values, shorter
campaign limits and malformed caps. No solver or holdout solve was run for these
tests. Evidence: `/tmp/sdpx-socp-reservation-20260916/budget-tests.log`.
212949 partial assessment has13/56 receipts and no numerical failures; no final
performance conclusion until all expected cells and final identities are checked.


Residual-overlap additional QA: ownership review confirms disjoint mutable
product/scaling workspaces and value-owned MPFR limbs, with only shared immutable
inputs across the join. Added local-only512-bit variants of pooled condensed
and overlapping-memory fallback equivalence tests. Both include exact residual
entry/norm comparisons for fresh/reused-forward paths and worker reconfiguration.
All15 selected MPFR512 tests pass (4.56s), including these two new variants.
Evidence: `/tmp/sdpx-residual-overlap-20260916/extended-512-tests.log` and
`extended-tests-identity.json`. Only the external test file changed; the numerical
source and already-frozen212949 campaign remain byte-identical. This is numerical
parallel coverage, not a new performance measurement. Latest assessed full-run
snapshot has24/56 receipts with no gate failures;212949 remains running. Completed
one-core ABBA medians: medium11.404927002→11.3715749815seconds (0.29% faster),
Ising51270.813399839→70.8485233165seconds (0.05% slower); both are noise, not
performance credit. Four-/sixteen-core and ordinary high-precision acceptance
remain incomplete.212950 is still dependency-held; no candidate is integrated.
Evidence: `/tmp/sdpx-residual-overlap-20260916/acceptance/current-decision.json`.


### SDPB timing-anchor provenance check

The copied Ising512 reference from212554 uses1e-34 internal tolerances and
records process elapsed time; the current screen uses1e-42 and warmed native
solve time. Its audited point remains an objective anchor at the fixed1e-30
external gate, but these archived times are not matched speed evidence. After
candidate acceptance, repeat SDPX/SDPB on the same allocated host and physical
widths with matched precision/tolerances, separate native/process timing and
unchanged original-coordinate audits. Do not infer parity from these old times.
Source hashes and review: `/tmp/sdpx-reference-provenance-20260916/review.json`.


### Residual-stage overlap accepted after full campaign212949

Integrated the reviewed residual-stage overlap into the shared condensed solver:
operator products and cone scaling use disjoint buffers on the existing pool;
serial fallback and per-output arithmetic order stay unchanged. No new buffers,
workers, tolerance changes or sampled-only branch. Includes exact residual
checks at Float64/256/512 bits, fresh/reused products and pool reconfiguration.

All56 expected campaign receipts pass unchanged numerical gates. Final666source
and6binary/library checks pass. One-/four-core differences are below2%; the
sixteen-core Ising512 median improves14.876019649→13.299804466seconds (10.60%),
replicating the initial7.74% direction. Medium16 improves11.606889869→
11.374204843seconds (2.00%, borderline; initial1.81%). Ordinary256 regresses0.92%;
ordinary512 is flat (+0.04%). Small LP/SOCP/SDP and SOCP_nb16 gains are below2%.
Acceptance is supported primarily by repeatable high-precision Ising16 benefit,
not a claim of general LP/SOCP speedup or MOSEK/SDPB parity. Candidate Ising1→16
scaling is5.33× (33.3% efficiency); medium remains essentially unscaled.

Evidence: `/tmp/sdpx-residual-overlap-20260916/acceptance/final-decision.json`
and `final-build-logs/`; integration identity and local check logs are in the
parent experiment directory. Integrated Rust suite:291/291 pass; release FFI
rebuilt successfully; Julia frontend:508 assertions across32 testsets pass.
The loaded FFI hash is recorded in `integration-review.json`.
Runtime AVX2/FMA screen212950 is now running independently against its frozen
pre-overlap baseline; any future combined candidate requires its own checks.


### Fresh matched Ising512 comparison submitted

PBS212965.node220 (`sdpx-matched-20260916-01`) is dependency-held after212950:
16physical cores/32GiB/1hour, controller deadline55minutes. Reuses the accepted
212949 candidate binary (SHA08b82b213931e5d3407929048f68e202176f47ad773d727a7c0e2dd7a9f92a1c)
and exact source/input; no rebuild or new numerical changes.370files frozen.
Pinned SDPB executable and dynamic dependencies are recorded by the controller.
At each of1/4/16physical cores: SDPX first+3warmed fresh solves, then3fresh SDPB
MPI solves, sequential on one node,512bits and1e-42internal tolerances. Every
returned point must pass unchanged1e-30original-coordinate audits and objective
agreement. Native/API/process timing and sampled process-tree/rank RSS remain
separate; no existing checkpoint reuse. A Linux numerical gate precedes timing.
Existing controller is copied externally with widths/core count and expected
point count generalized to this fixed sweep; production harness is unchanged.
Local preparation: `/tmp/sdpx-matched-20260916/`. Pending results are not parity
or scaling evidence. This campaign tests the accepted residual-overlap release,
not the still-experimental runtime FMA candidate or their combination.


### Runtime FMA screen: verified dispatch and remaining scalar helper

212950 is running; x86 compact-panel equivalence test actually executes and
passes (1test). Candidate disassembly contains vector FMA in the dense Schur
dot loop, but still calls out-of-line Float64 `coefficient_product`, whose
inner loop retains indirect scalar arithmetic calls. Thus this portable wrapper
does not yet reproduce the whole-library target-feature optimization. Evidence:
`/tmp/sdpx-runtime-fma-20260916/disassembly/`. Current3/8receipts have no numerical
failures; partial medium candidate median9.83536seconds, incomplete baseline.

External follow-up `/tmp/sdpx-runtime-fma-inline-20260916/` changes only that
shared helper annotation from `inline` to `inline(always)` on top of the frozen
runtime-dispatch candidate. It preserves one arithmetic implementation and
operation order, allowing the checked CPU context to reach this loop. All289
local core tests pass (5.30s); this is ARM fallback coverage, not x86 performance
or portability qualification. No follow-up numerical change is integrated or
submitted yet. Requires x86 exact-equivalence, generated-code inspection and
medium/Ising screen before further acceptance; MPFR code-size/regression remains
part of the check. Both runtime candidates exclude accepted residual overlap.


Inline follow-up submitted as212966.node220, dependency-held after matched
SDPB campaign212965.16cores/32GiB/45minutes,652frozen files. Baseline is the
runtime-FMA candidate from212950 (not the original generic source); sole source
change is `inline(always)` on `coefficient_product`, SHAa55ddb009553c9fb04601664859dcc97da8e8f24a0013b66c74d7687a75ea82b.
Before timing, run the actual x86 bitwise-equivalence test and rebuild both FFI
arms. Eight fixed medium1/Ising512-16cells, original limits and external gates;
no holdout or tolerance changes. This isolates incremental inlining benefit and
must not be presented as a combined release speedup. Parent screen212950 has
6/8receipts, no failures, completed medium medians11.4255677235→9.8353608815seconds;
Ising repetitions and final identities are still pending.


Runtime-dispatch screen212950 completed: all8numerical receipts pass,652source
and4library checks pass. Medium1 medians11.4255677235→9.8353608815seconds
(13.92% improvement); Ising512-16medians14.711022457→14.506035010seconds
(1.39%, below credit threshold). Explicit x86 bitwise test passes and disassembly
confirms vector FMA for dense Schur dots. The whole-library AVX target screen had
22.19% medium gain; remaining scalar coefficient helper motivates212966, not
an assumption that independent gains multiply. Evidence:
`/tmp/sdpx-runtime-fma-20260916/final-decision.json` and `build-logs/`.
No runtime-FMA code is integrated yet; retain accepted residual-overlap release
while evaluating the inline follow-up and eventual combined broad acceptance.
An initial local assessment raced an unfinished rsync and saw missing files;
waiting for that same transfer to finish and reassessing resolved it, with no
remote rerun or numerical repair.


Matched212965 is running on node49 with16distinct physical cores; Linux gate
passed. Preparation repair before any SDPB rank launch: copying the archived
SDPB executable with `copyfile` lost its execute bit. Restored owner execution
0644→0744, preserving SHA3ee1bee7417d853945bc94cfa7a39508d3266ebd1db9752f53509fb6613281fe.
Receipt: remote `results/212965.node220/executable-mode-repair.json`. No content,
settings, checkpoint or input changed; local preparation now uses `copy2`.

Timing interpretation: archived SDPB outputs expose integer-second `Solver
runtime` and millisecond `iterations.json` total_time/iter_time. Source review
shows runtime is measured from program start before input loading, so it must
not be mislabeled as pure iteration time. Retain these scopes separately from
SDPX native time, Julia API and process-wall time; use iteration records to
analyze per-iteration cost, not integer rounding to claim small speed changes.


Matched212965 SDPX leg completed: first+3warmed solves at1/4/16cores all pass
original-coordinate audits with50iterations each. Warm native medians:
71.192046877 /22.813339422 /13.2590836seconds; same-host speedups3.12× at4cores
and5.37× at16cores (78.0%/33.6% efficiency). Observed cell process-group RSS:
700092416 /703959040 /752492544bytes, including Julia and external audits.
These are SDPX scaling measurements only; SDPB repetitions are now running.
External summary and four numeric/protocol/command/incomplete-snapshot checks:
`/tmp/sdpx-matched-20260916/summarize.py` and `test_summary.py`. The summary checks
all21points, fixed512bits/1e-42/1e-30protocol, binary identity, actual thread/rank
binding, original metrics and objective agreement before declaring completion.
It keeps unlike timing scopes separate and never emits a premature cross-solver
ratio. Final whole-campaign identities remain required.


### Next parallel hypothesis: fuse scaling sync and cached block assembly

External candidate `/tmp/sdpx-sync-assembly-20260916/source` starts from the
accepted residual-overlap release. Existing outer parallel blocks now update
their scaling data and compute their own cached Schur contribution in one task,
removing the intervening all-block barrier. One extracted block-compute helper
serves fused and old paths; cone-order scatter and per-entry arithmetic order
remain unchanged. Inner parallel dispatch and uncached memory fallback retain
the existing two-stage path. No added cache, pool, precision-specific algorithm
or change to numerical gates.291core tests pass (5.59s), including exact serial/
pooled Schur and residual comparisons at Float64/256/512bits and reconfiguration.
A test-only wrapper plus comments were added afterward; production cargo check
passes. No cluster timing submitted and no production integration yet. Profile
predicted benefit is limited by existing scaling-sync cost; retain only if a
bounded same-host screen demonstrates repeatable gain without regressions.


Sync/assembly fusion screen213040.node220 submitted after212966, using the
accepted residual-overlap binary's exact numerical source as rebuilt baseline.
653files frozen; one changed numerical source file, no FMA candidate mixed in.
16cores/32GiB/45minutes; same eight-cell medium16ABBA/Ising512-16BAAB driver,
300/400-second solve caps and fixed numerical assessor. Both arms rebuild before
sequential timings. Final33condensed tests pass (2.28s) after the test-wrapper
annotation; full291test result and production check remain recorded. No production
change or performance claim yet. Local namespace `/tmp/sdpx-sync-assembly-20260916/`.


### Matched Ising512 completed: speed advantage, remaining iteration/scaling gap

212965 completed all21point audits (12SDPX,9SDPB) at512bits,1e-42internal and
1e-30external tolerances. Final370frozen-file checks pass; dependency/source
identities before/after match. Summary independently checks precision, actual
binary, input identities, fixed settings, physical/rank binding, point metrics
and all-objective agreement. Frozen legacy rank receipts lack hostname; locality
is verified using the one-host PBS nodefile and each rank PID/start-time key in
the locally sampled memory records. Five summary tests pass, including rejecting
missing/locality-mismatched rank records. No numerical gates were changed.

| Physical cores | SDPX warm native median (s) | SDPB iteration-time sum median (s) | SDPB process median (s) |
|---|---:|---:|---:|
|1|71.1920|127.056|129.5643|
|4|22.8133|35.847|37.3526|
|16|13.2591|15.660|18.1371|

Each median uses3repetitions. SDPX always50iterations, SDPB201recorded iterations.
Timing scopes differ: SDPX native solve versus summed SDPB per-iteration timers;
SDPB cumulative timer medians are127.340/36.014/15.909seconds. SDPX full process
cells include first+3warm solves and audits, so do not compare those cell totals
against one SDPB process. This finite benchmark shows an advantage from fewer
iterations, not universal SDPB parity or a rigorous equal-scope ratio. Average
iteration cost remains approximately2.3–3.4× higher for SDPX under these timers.
SDPX16core speedup5.37× (33.6% efficiency), SDPB iteration sum8.11× (50.7%).

Observed SDPX cell process-group RSS700092416/703959040/752492544bytes includes
Julia and external audits; SDPB solve-group medians99622912/290504704/1071054848
bytes include MPI ranks, with external audits separate. These are sampled sums
of RSS, not private/PSS memory or pure solver workspace. The larger Lambda11
accuracy issue,64/256cores and multi-node qualification remain unresolved.
Evidence: `/tmp/sdpx-matched-20260916/final-summary.json`, `final-source.txt`,
`results/212965.node220/`. Keep the sync/assembly experiment directed at measured
per-iteration and scaling gaps; retain it only after its own paired evidence.


Inline screen212966 generated-code check confirms coefficient AXPYs are now
inlined into the runtime-checked FMA body: vector `vfmadd213pd` precedes the GEMM
call, with scalar FMA tails and no out-of-line Float64 coefficient_product symbol.
Candidate library13132904bytes versus13114336baseline (+18568bytes). Actual x86
bitwise test passed. Evidence: `/tmp/sdpx-runtime-fma-inline-20260916/disassembly/`.
First2/8receipts pass; medium1single samples9.83893→8.99753seconds are preliminary,
not medians or integration evidence. Complete all repeated gates and library/source
checks before deciding, and retain separate acceptance for combined optimizations.


### Inline screen completed; combined runtime-FMA acceptance submitted

212966 passes all8numerical receipts,652source and4library checks. Relative to
runtime dispatch alone, medium1median9.829209406→8.9980119875seconds (8.46%).
Ising512-16median15.016494691→14.4040481005seconds (4.08%), but sampled assembly
bypasses the modified coefficient helper: do not attribute that timing change to
this arithmetic optimization. It is no observed regression in this screen.
Evidence: `/tmp/sdpx-runtime-fma-inline-20260916/final-decision.json`.

External combined candidate adds runtime AVX2/FMA+inlining to the accepted
residual-overlap source, preserving the exact common Schur body and current
512-bit tests. All291local core tests pass (5.63s). No production integration.
PBS213041.node220 is dependency-held after sync/assembly screen213040.666files
frozen;16cores/32GiB/2hours. Both FFI and ordinary-SDP binaries rebuild; actual
x86 equivalence test must pass before timing. Reuses the unchanged56-receipt
LP/SOCP/SDP, medium/Ising1/4/16 and ordinary256/512 acceptance protocol against
the accepted residual-overlap baseline. Fixed limits, tolerances and original
coordinate gates; no holdout exposure. This candidate excludes experimental
sync/assembly fusion. Evidence and scripts:
`/tmp/sdpx-runtime-fma-combined-20260916/acceptance/`.


### Sync/assembly fusion rejected after paired timing

213040 completed all8numerical gates plus653source/4library identity checks.
Medium16median11.365876934→11.567953869seconds (1.78% slower), Ising512-16
13.239480858→13.8865858515seconds (4.89% slower). Reject this candidate; no
production change and no unchanged repeat. Removing one barrier did not improve
whole-solve performance; do not infer an unmeasured scheduling cause from totals.
Evidence: `/tmp/sdpx-sync-assembly-20260916/final-decision.json` and `build-logs/`.
Current stable source keeps separate block scaling-sync and assembly stages.
Combined FMA acceptance213041 remains running and excludes the rejected fusion.


### September 16 publication verification

Both final speculative candidates were rejected; production Rust/Julia source
matches the fully accepted runtime-FMA snapshot byte-for-byte. Workspace tests
(`cargo test --workspace --locked --offline` with Accelerate and Faer) pass;
Julia frontend validation has 508 passing assertions. Research and Ising harness
tests also pass. Publish the stable engine, exact preprocessing, benchmark
protocol/holdout updates and associated documentation; retain known accuracy,
large-instance and distributed-scaling limitations above. No rejected candidate,
local binary, temporary timing artifact, credential or sibling repository is
included. Local logs: `/tmp/sdpx-publication-20260916-*.log`.


### September 17 compact dual constructor: first real Lambda11 step

The isolated constructor now selects the existing condensed solver through the
mapped dual adapter before constructing a generic KKT backend. Constructor-only
qualification passed3tests/6solves (Float64/256/512); evidence:
`/tmp/sdpx-dual-constructor-20260917/test.log`. No production integration.

The first real768-bit Lambda11 screen exposed non-bit-identical cumulative Ruiz
row scales within PSD blocks. The strict uniform-factor assertion failed before
solve; do not relax it or turn off Ruiz. Preserve the compile failure (missing
One/Zero imports), successful retry and failed run in
`/tmp/sdpx-dual-lambda11-native-20260917/`. Process7.083s, peakRSS2426732544bytes;
all192frozen identities passed. This is a fixture/integration failure, not a
solver convergence result.

A separate isolated candidate follows existing install_sampled's exact uniform
PSD row-scale convention: canonicalize each block's e/einv, rebuild dual A/b/q
from original input and final scales before constructing KKT. Precision,
regularization, refinement, Ruiz, presolve, chordal and stopping criteria remain
unchanged. Nine adapter/factor/solution/ray tests pass atFloat64/256/512 in0.13s.

Real input SHA bb1fa49da0d461ebba2b9539412222e5dc134553dfad8f7bc128b23acf61ed1d:
1099original variables,20equalities,28PSD blocks; explicit dual22196variables,
23275rows,1794016nonzeros. Uses factor-authoritative residual/RHS/recovery and
sampled Schur; no fitted replacement for authoritative CSC data. Reduced KKT
order1119; setup still materializes explicit matrices and has substantial memory.
One thread,768bits,1e-42tolerances,max_iter1,native30s/outer90s. Completed with
MaxIterations at1, native29.686906s/process32.601064s, peakRSS2572173312bytes.
Initial internal primal residual6.58e-3 correctly prevents the previous iter0
false acceptance; after one step it is2.4894e-5. Original factor-coordinate
absolute residuals: primal23.5889, dual1.7191e-174; original objective-53.7639.
Both operator directions were invoked twice. No full accuracy acceptance:
primal feasibility remains poor, and this screen lacks full cone/gap audits.
One unpinned execution establishes neither a speedup nor end-to-end convergence.

Evidence: `/tmp/sdpx-dual-lambda11-uniform-20260917/` (review.json,test.log,
frozen-run.json,run.log,result.json,receipt.json,decision.json). All192frozen
identities and185production source/dependency identities pass. Only this plan
is updated in production. Next run a bounded multistep screen exporting the
original-coordinate point for independent cone/gap/equation audits; keep the
prototype external until full numerical qualification. Provider substitutions
remain lower priority than this unresolved large-instance correctness issue.


### September 17 Lambda11 compact dual multistep: convergence remains unqualified

External candidate is numerically identical to the one-step uniform-scaling
prototype; the only source delta changes max_iter1->6, native time30->60seconds
and exports original-coordinate x/s/z. Outer hard limit100seconds. Release build
passes. No production numerical changes;185source/dependency identities match.

At768bits/one thread/default preprocessing and unchanged1e-42internal tolerances,
the run completes4iterations, exits MaxTime, native86.703857s/process90.705518s,
peakRSS2604122112bytes. Native time is checked at iteration boundaries and is
not a hard deadline; the external limit was not reached.192frozen identities
pass. No wall-clock comparison or performance credit from this unpinned run.

Independent Julia factor-operator audit parses the frozen768-bit input/point at
768bits, then evaluates at1536bits, including56PSD eigensystems for28blocks.
Audit7.926526s, peakRSS732397568bytes; input, point, helper and environment hashes
unchanged. It does not change solver precision. Final original-coordinate values:

- Primal absolute residual1.21684; scale-normalized residual8.83145e-3.
- Dual absolute residual1.72066e-174; normalized1.29115e-175.
- Relative objective gap1.25356e-2; primal objective-58.46163, dual-57.72878.
- Primal PSD violation0; dual PSD violation1.17446e-2; equality slack0.

The1e-30point gates fail. Internal dual-orientation primal residual2.14847e-6
must not substitute for these original-coordinate results. Avoid interpreting
four steps or small dual-equation residuals as full feasibility. Reference
objective agreement and full sampled mapping qualification are still outstanding.
Evidence: `/tmp/sdpx-dual-lambda11-multistep-20260917/` (frozen-run.json,run.log,
result.json,point.json,receipt.json,audit-frozen.json,audit.json,audit-receipt.json,
decision.json). The nine prior small tests are not rerun because only experiment
limits/output changed. Next instrument bounded scaling/direction phases and
inspect original cone/equation errors; do not rerun an unchanged longer campaign
or promote this prototype on its internally reported residuals.


### September 17 Lambda11 dual-path profile prioritizes inner directions

Saved-point localization finds negative eigenvalues in all28original dual PSD
blocks after4iterations; the worst normalized violation1.17446e-2 is block9,
order38 (minimum eigenvalue-0.0144942). This early MaxTime point does not by itself
prove a Newton-system defect. Evidence: multistep/localization.json.

New isolated diagnostic keeps the same numerical method, limits to2iterations,
and inserts11strictly matched outer-timer patch hunks plus mapped-backend spans.
Release build passes; one768-bit/single-thread run completes MaxIterations2:
native48.751996s, process52.710580s, peakRSS2604023808bytes. All192frozen identities
and185production source/dependency identities pass. No full accuracy or speed
acceptance; this is unpinned diagnostic phase attribution, not paired timing.

Outer stages: affine direction8.581119s (17.60%), combined direction9.258156s
(18.99%), cone scaling7.040953s (14.44%), KKT update6.653487s (13.65%), residual
update1.995514s (4.09%), affine/combined bounds2.192070/2.193508s. Nested mapped
solves include initialization:9calls22.230523s, of which inner condensed solve
19.252998s; mapped RHS9calls1.387019s. Surrogate updates3calls2.309543s also include
initialization. Do not sum these nested totals with outer stages or treat the
internal update as all of the outer KKT-update time (which includes RHS solves).

Next prioritize inner condensed solve_raw scaling/operator products and residual
refinement, preferably capturing representative RHS/scaling for bounded replay.
Existing original/refined direction checks remain unchanged; no removal of
refinement or cached-G shortcut is authorized by this profile. SVD-library
replacement and raw mapping overhead are lower priorities for this prototype.
Evidence: `/tmp/sdpx-dual-lambda11-profile-20260917/` (setup-review.json,build.log,
frozen-run.json,run.log,result.json,point.json,receipt.json,analyze.py,stages.json).
Stable production numerical source remains unchanged; prototype qualification
and large-instance accuracy remain outstanding.

September 18 sampled ordinary-lane parallelization, single-parallel-level repair
and all-precision GEMM/SYRK fusion (accepted):
Measured on the frozen Ising512 sampled SDP, 512 bit, 50 iterations. The sampled
operator's ordinary CSC part was the only unbounded serial section: apply and
apply_transpose called linear.gemv on the calling thread over all ordinary
nonzeros before any block work. Both now dispatch through the existing
row/column SparseParallel plan, which reuses the CSC gemv scalar branches
(scale_output and accumulate), so every output accumulates in the same order as
the serial product and results are bit-identical; three new tests at Float64,
256 and 512 bit assert exact equality at widths 1/2/4/8 and assert that the plan
owns real lanes rather than a serial fallback. residuals.rs deliberately keeps
its sampled-route plan disabled because that route reaches the operator, not
the residual gemv.
The condenser handed the pool to every sampled block whenever the pool exceeded
the block count, so each block spawned its own GEMM/SYRK tiles and pair lanes on
top of the outer block level. This contradicted the module's own
single-parallel-level rule. The pool now reaches sampled inner work only for the
dominant block; the outer block level is the only parallel level for the rest.
Seven previously dead assertions were also found: the pooled split writers
accumulated into w.forward/w.adjoint, which the workspace only zeroes when its
length grows, so a second pooled call added to the first call's result. The
split branches now clear their buffer; pooled_operators at Float64/256/512 bit
reproduces the serial product exactly and had been failing before the fix.
gemm and syrk still gated fused accumulation on matches!(N,4|8|12|16), so 128,
384, 1536 and 2048 bit silently used the slower indexed product while 256, 512,
768 and 1024 bit did not. Both gates are removed; the descriptor form is not
specific to a limb count. Fourteen mpfr_gemm_syrk, fourteen mpfr_parallel, nine
mpfr_workspace and twenty-two mpfr_dense checks pass, and the Julia suite covers
128-2048 bit.
Rejected in this round, with measurements kept as negative results.
Whole-round cone lanes (one lane per worker instead of workers*4): measured
worse, w8 13.35 -> 14.37 and w16 12.47 -> 14.11 seconds. Balanced lanes follow a
structural cost model, so the spare tasks are what lets work stealing absorb
model error; the deliberate over-splitting stays.
Pool width capped by cone count: 15.39 -> 14.60 seconds at w64 with every audit
passing, but the reported cone_threads became 23 instead of the requested 64 and
the frozen benchmark plan gate requires cone_threads == width. Reverted; the
frozen campaign is unchanged and the request remains a contract, not a hint.
Order-preserving parallel QDLDL: the measured elimination tree of the 342-square
reduced KKT is a single-root caterpillar with widest level 11 and depth 51, so
subtree parallelism cannot exceed about eleven ways on a phase that costs 2.4
percent. Not written.
Cross-node: deferred by decision. The condensed Schur is n-by-n, so Ising512
would reduce 322-square (6.6 MB, reducible once per iteration), but the 1483
block instance is 14424-square, about 13 GB per iteration, which no allreduce
serves. SDPB avoids the gather with Elemental's block-cyclic distributed
Cholesky. A first-level rank sharding with replicated vectors and O(m+n)
allreduce is feasible but changes the Schur assembly summation order, which is a
contract decision, so no MPI backend was written.
Scaling curve measured on node3 in one socket, 64 physical cores, all six cells
accepted by the external original-coordinate audit: w1 60.52, w4 20.24, w8 13.35,
w16 12.47, w64 14.60-15.5 seconds. The curve is flat from 8 to 16 threads and
negative at 64, and the first 64 physical cores are one package, so the residual
limit is lane count (23 cones) and phase barriers rather than serial code.
Per-iteration accounting: 3.06 KKT solves per iteration, 3 congruence
applications per solve (two in solve_raw, one in the residual), 9.2 applications
per iteration; iterative refinement measured zero extra passes; one application
costs 22 blocks times 4 GEMMs of n-cubed (sum 80738) in 38.1 ms, or 118 ns per
fused multiply-add. Borrowable SDPB directions, none implemented here: batch the
per-block congruences into large products so CRT plus hardware BLAS
(BigInt_Shared_Memory_Syrk) has something to amortize, and replace the
structural block cost model with measured block costs as allocate_blocks does.
Evidence: local /tmp/sdpx-ising-local-124-20260917/ (final1, final4b, cnt1,
etree3, ph1, ph4); cluster jobs 213439, 213452, 213461, 213462, 213463 under
~/projects/sdpx-scaling-64-20260917/results/. Rust workspace 477 checks pass and
the Julia suite reports 673 of 673.
