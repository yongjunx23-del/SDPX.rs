# Singleton-cover admission: correctness passes, performance inconclusive (2026-11-11)

**Draft only. The fixed performance gate is INCONCLUSIVE; do not merge.**
This report covers only newly compiled binaries and newly audited solves on
2026-11-11. No measurements from an earlier host are used.

## Change and numerical scope

The existing condensed PSD exact-congruence path can now admit a coefficient
whose nonzeros share one diagonal anchor, in addition to the existing two-anchor
case. Both admission helpers validate against the actual anchors before padding
a singleton with a distinct second index. The dummy index must not admit extra
edges, and the same index must not be used twice. Existing two-anchor ordering,
finite/exponent scans, shape/precision/parallelism gates, residue bounds, kernel,
packing and scratch layout are unchanged. No public API, dependency, precision
or convergence tolerance changes.

This intentionally replaces an eligible coefficient-product plus matrix-product
rounding sequence with an exact congruence rounded once. Cross-arm points and
refinement counts need not match. Each arm must independently satisfy the
unchanged original-coordinate accuracy gate. C4 within-arm repeat stability
passed in the completed campaign; the conditional C3 performance comparison
was not eligible and was not run.

## Source and build identity

- Baseline main commit: `88f5100c7dba0ade45634408bf793d8fcab3d59b`.
- Baseline tree: `4080323fa8a8137ccbe74151ed56c341b6518c22`.
- Tested local candidate commit: `682afd950e2fc6f50f278b87acbf5d782cb3eacd`.
- Tested candidate tree: `6d74456ce69aebb0ba8619e00b327f35e1d0ae48`.
- Tested patch SHA-256: `1273f59c7005c4cf1414d495d2a6287f179d2211a65de31c88af1e521669cdf3`.

The publication commit metadata may differ from that local commit; the core
Git tree and all existing Rust blobs must match exactly. Documentation is a
separate child change, so the full publication tree is not the tested core tree.

Rust/Cargo 1.98.1, Linux x86-64, release profile, default/serde/faer-sparse
features, native CPU code generation, thin LTO and one codegen unit. The generic
production solver library is linked through a reviewed single-precision,
CLI-derived validation adapter for each requested width. **This is not a
benchmark of the production multi-precision CLI.** Both 256-bit arms use the
same adapter source and normalized recipe. The 512-bit candidate adapter uses
the same frozen candidate library. Source, libraries, adapters and providers
are immutable and independently reviewed. A falsely fresh shared-target
candidate archive was rejected before any solve; the actual candidate library
was rebuilt into a new empty target.

Direct pthread OpenBLAS 0.3.29 is used with BLAS/OMP set to one thread. All five
solves actually used four cone and four backend threads, in one process/rank.
All six dynamic providers match between arms. OpenBLAS file SHA-256:
`10d1c640ecccd21063a7d80f90e949b1d48202830b6ec0b52b2796c5d0fbce63`.

| Adapter | Binary SHA-256 |
|---|---|
| A256 | `2fcd3c1ff2907fefb8c5cc0c5d5d7905ba36117690e607040db3c236f96ad51c` |
| B256 | `0f35d9b06e8ffbdc66d4a5cfe6d973069cca9d5033c46725b68ba84a301f3f0d` |
| B512 | `fee607153b8434cbeee2d5462a1dc33a021e247325cb9be6029100a60a7fce72` |

## Focused and full correctness checks

Four new focused tests and two unchanged regressions pass: 135,360 true-256-bit
once-rounded full-output oracle comparisons, 106,128 residue outputs and 14,265
PSD NaN-intermediate sentinel comparisons; unchanged coverage includes 18,240
512-bit upper outputs, 8,844 residues and 132 balanced-halving comparisons.
The sentinel proves actual new-path execution on those test matrices. The
complete C3/C4 solves below use only ordinary built-in receipts, which do not
distinguish genuine singleton/two-cover cohorts or every kernel route. Counter
changes are indirect evidence only. Formatting
check also passes. This is not a claim that a full local workspace suite ran.

Focused-gate manifest SHA-256:
`49faae5569170402c19e7e369f92a715e5b4a41b73d54c7fc12d01afb85e853f`.
Independent full-solve packet review: **PASS**. All five original audits were
recomputed from the saved points and matched exactly. Review receipt SHA-256:
`2aad78e7b97c40f74b18079398a171132c558d245d23b55746c121f249655a8b`.
Final functional summary SHA-256:
`157d449cba2cfd1f843c4699652843b5517e63edc7fbcfd86320060d26258df4`.
The local 204-file proof manifest SHA-256 is
`b27b5ebddd7018579269aedc95032f8014d65fde4f7e0ffc178b0b1d23806eec`.
These identities do not turn the validation references into performance samples.

All five fixed-order validation references report `Solved`, finite complete
points and all eight original-coordinate metrics at or below `1e-30`. The
independent checker reads the original SDPA records and returned x/s/z, not the
solver's reported objectives or residuals. It uses 180 decimal digits; this is
a numerical audit, not a rigorous interval certificate. The exact formulas and
normalizations were frozen before any complete solver point, after eight
analytic parser/auditor selftests and independent review.

For recomputed p = c^T x and d = trace(F_0 Z), the denominators are
D_p = 1 + sqrt(sum_blocks ||F_0,block||_F^2), D_d = 1 + ||c||_2 and
D_g = 1 + |p| + |d|. Frobenius norms include both symmetric entries.
The eight metrics are ||returned S - raw S||_F / D_p;
||[trace(F_i Z)-c_i]_i||_2 / D_d; |p-d| / D_g;
|trace(raw S Z)| / D_g; |trace(returned S Z)| / D_g;
and the maximum negative-eigenvalue violations across blocks for raw S,
returned S and Z, divided by D_p, D_p and D_d respectively.
PSD eigendecomposition residual and orthogonality checks plus a numerical
cushion are mandatory. These are DIMACS-style definitions, not a claim of
exact standard DIMACS conventions.

Checker source SHA-256:
`0cc3cdc864f8c60c19c4ae2449070ba55d9e0ce969c964927089732e6eab4eb3`.
Checker contract SHA-256:
`39f8195949a88a4665326ec654d2452289389fd148883823f13b42f74b20d524`.
Audit/input tool manifest SHA-256:
`0126b501f48078185df60b074fc2ba980857cf2f89973d4e1d1f198bf07e44e9`.

| Validation reference | Bits | Iterations | Native seconds, descriptive only | Largest normalized audit metric |
|---|---:|---:|---:|---:|
| A / control4 | 256 | 60 | 10.008024196 | 9.847549962408E-32 |
| B / control4 | 256 | 60 | 9.463371787 | 9.847549962408E-32 |
| A / control3 | 256 | 58 | 4.272744422 | 1.373763894653E-31 |
| B / control3 | 256 | 58 | 6.263193395 | 1.373763894653E-31 |
| B / control4 | 512 | 69 | 22.687221263 | 2.324137147418E-31 |

These five times were excluded from the performance sample by advance rule.
They are retained, not selected or discarded, and cannot establish a speedup or
nonregression. Native time is the internal recorded setup plus solve clock;
loading, process startup, serialization, audit and pauses are excluded. It is
crosschecked exactly against the receipt solve clock.

| Reference | Slack | Stationarity | Gap | Raw complementarity | Returned complementarity | Raw PSD | Returned PSD | Dual PSD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A256 control4 | 1.092943065063E-71 | 4.993855242038E-35 | 9.838385221376E-32 | 9.847549962408E-32 | 9.847549962408E-32 | 0 | 0 | 0 |
| B256 control4 | 1.093337116366E-71 | 4.993855242038E-35 | 9.838385221376E-32 | 9.847549962408E-32 | 9.847549962408E-32 | 0 | 0 | 0 |
| A256 control3 | 6.160030727622E-72 | 3.289272038330E-35 | 1.373100413426E-31 | 1.373763894653E-31 | 1.373763894653E-31 | 0 | 0 | 0 |
| B256 control3 | 5.905145422944E-72 | 3.289272038330E-35 | 1.373100413426E-31 | 1.373763894653E-31 | 1.373763894653E-31 | 0 | 0 | 0 |
| B512 control4 | 9.611458347499E-149 | 2.746636429201E-44 | 2.324137147418E-31 | 2.324137147418E-31 | 2.324137147418E-31 | 0 | 0 | 0 |

Values above are rounded for presentation; decisions use the full saved values.

### A256 control4 evidence identities

- point_sha256: `33ee85d895b0ab8da15ceebe27a6cab4c29a7aaea8c49ec4af71f889080e0efa`.
- receipt_sha256: `0ae0ca6ab3806400814e0d5ce82a6c1ba05627f9fff40607bc587acc1f20694b`.
- audit_sha256: `258a9ad632653220dfdb6a32af81e518b6b6753c87d46713a99346651bbdec87`.
- reference_sha256: `c74d2fe7984cb8010341c3f4ecb40db7afc73918e6823ddf42dc2b3f62502822`.
- Mathematical counters: batches=51, factor_attempts=52, factorizations=52, linear_solves=160, outer_refinements=27, refinements=0, rhs_applied=133.

### B256 control4 evidence identities

- point_sha256: `5c64d256922368eeaa1f4bb5b7d0081781ab8a26e53cb2b4fc85778421a945bd`.
- receipt_sha256: `bd93f682e2cf03b388de7bcc0d9bcbfaa0bdd1d245c46b37781949f6b911dd6d`.
- audit_sha256: `9c38e4f4824a99a3368e0abc3a72d095ee391c2ae2b9725903f4432649b647f9`.
- reference_sha256: `1a253db41db355674ef673e40242997e0f1e4a749dd727d32325cab2d88fb034`.
- Mathematical counters: batches=51, factor_attempts=52, factorizations=52, linear_solves=161, outer_refinements=28, refinements=0, rhs_applied=133.

### A256 control3 evidence identities

- point_sha256: `a3735fdd1f348d2c9cfcc20705d5ecc342d70d30994a8f4bbcc1dccdfefb2006`.
- receipt_sha256: `d7022aa4062b8e304d159087a028c409bd89fc8d303f2f34f94c1bcb990da0f3`.
- audit_sha256: `64da80e41e3a959e16ec9e8debf14bfbc9b1028b6f1a2d27c3c589c247635619`.
- reference_sha256: `9101ef71f9a33685eb8cda9883f7dfd03818c5764c200eb2760847357ebb1ab0`.
- Mathematical counters: batches=49, factor_attempts=50, factorizations=50, linear_solves=158, outer_refinements=28, refinements=0, rhs_applied=130.

### B256 control3 evidence identities

- point_sha256: `242262cd72ca6645be51d407c7fdc55337eb4ea4d2e211ad27d7542cd9522bcc`.
- receipt_sha256: `66bd0ccf6f337689c5a30650e9179540b662c519c418798b170b066a2e0258ae`.
- audit_sha256: `0950dcd35da5d3f3d20fb4cd1f2ea012737a95723d9ab8e1d74055f636f06b36`.
- reference_sha256: `ad7ed92c68934d7043390358c6ccb6152f74e665acb13d6002ef6ab11a64f713`.
- Mathematical counters: batches=49, factor_attempts=50, factorizations=50, linear_solves=158, outer_refinements=28, refinements=0, rhs_applied=130.

### B512 control4 evidence identities

- point_sha256: `22636342a4f46c4d439e42a19bf6eced78e9f1fd945515d4ef51b082d40d0f43`.
- receipt_sha256: `7e564df45c83a1c85199cd4394b1bd15f2f0c9932dbae9156d7cb788d6018f55`.
- audit_sha256: `accabe0a3c69f65786c65a1c2366f217920e10a09aec3b8ea4c261f4772e7880`.
- reference_sha256: `db35d0afada519d0d776e836037b4d02f83fc061cbf575e2e41f97acfb8ebe5b`.
- Mathematical counters: batches=60, factor_attempts=61, factorizations=61, linear_solves=163, outer_refinements=0, refinements=0, rhs_applied=163.

## Public input provenance and settings

Brian Borchers SDPLIB 1.2 data, retrieved through the pinned SDPA-GMP mirror
commit `ca110db5ea1cc46e811b70dfea9cbb25db74448d`:

- [control3.dat-s.xz](https://raw.githubusercontent.com/nakatamaho/sdpa-gmp/ca110db5ea1cc46e811b70dfea9cbb25db74448d/SDPLIB/data/control3.dat-s.xz): decoded 160879 bytes; decoded SHA-256 `931c6f1cb5b70ad7907d73067642c935228244c3908ebc4912001271d0fdb009`; compressed SHA-256 `e63d334cc8c930b9f7c29b173951d061ab4c7766caeb5205097850f80442a69a`.
- [control4.dat-s.xz](https://raw.githubusercontent.com/nakatamaho/sdpa-gmp/ca110db5ea1cc46e811b70dfea9cbb25db74448d/SDPLIB/data/control4.dat-s.xz): decoded 389895 bytes; decoded SHA-256 `f1086b1a6abdc82bc1b27e6d4ad8fd7986fbb7e73c86e6475f2ab8db16da3815`; compressed SHA-256 `ea1d8a673bf95f3dfa0ec49e7b4b4acb8d608e32d0d6eec1858b379ebcfb993a`.

C3 has 136 variables and PSD blocks 30 and 15; C4 has 231 variables and PSD
blocks 40 and 20. Both arms use identical converted input and settings bytes.
The SDPA convention is S = sum(x_i F_i) - F_0, with q = c, A_i = -svec(F_i),
b = -svec(F_0), and sqrt(2) off-diagonal packing. Conversion uses 180-digit
numerical decimal strings; there is no f64 conversion stage.

Settings override only `verbose=true`, `time_limit=120`, `max_threads=4`, and
`tol_gap_abs=tol_gap_rel=tol_feas=tol_original="1e-30"`; all other settings are
unchanged main defaults. Settings-file SHA-256:
`85d554342a952b7ba1dac601a5a1529a365d69908d8339b2a8da8817e39b2c97`.
No input data, SDPA-GMP implementation source, benchmark runner, frozen point,
binary or raw log is included in this checkpoint. Benchmark material stays
local under the repository policy.

## Preregistered performance decision and merge restrictions

A fixed current-host, true-256-bit, release comparison was preregistered before
any complete native point. A = current main, B = this candidate. C4 uses eight
balanced ABBA/BAAB blocks (32 solves), and passes only if the one-sided 95%
upper bound for the block geometric B/A ratio is at most 0.98. Only then does
C3 run 32 balanced blocks (128 solves), requiring its upper bound at most 1.02.
Block log ratio is half the sum of B log times minus A log times; bounds use
t(.95,7)=1.89457860509000738947 and t(.95,31)=1.69551878254586556264,
respectively, with sample standard deviation and no pooled rescue. Missing,
failed or interrupted legs make the attempt incomplete; there is no replacement,
extra repeat or extension. Every timed leg must pass all eight original audits
and exact within-arm point/counter/settings/runtime references. All raw times
and descriptive medians, order/half effects and resource data are retained.

Protocol SHA-256:
`7d6cac844c039390530ae625dbad82daa79bb95d8d9fcc51ed0bf28fd59871e4`.
Fixed-sequence SHA-256:
`c4f4ce0bfc438d41da27edabc4992df50f84c6f1e636362e9aef87ae485ba3f1`.
Native-timer contract SHA-256:
`c2d2d640b3ca7149a7c08180a7e39840f651c23ee772210d1442a1905fc796ae`.

Approximate independence/stationarity and the Student-t block-error model are
assumptions, not guarantees. Deterministic order, drift, autocorrelation and
heavy-tail limitations must be reported. Results can describe only the tested
cases, precision, thread budget and host, not a universal speed guarantee.

The completed C4 comparison below did not pass its performance gate. Exact-head
GitHub CI is tracked separately; even green CI cannot override this result.
PR #24 remains draft and unmerged. No measured performance gain or general
nonregression is established.


## Completed C4 campaign: INCONCLUSIVE, not promotable

All 32 prescribed C4 legs completed, with all 256 original-coordinate metrics,
complete within-arm points, seven mathematical counters, settings, precision,
thread budgets and immutable provenance checks passing. Independent final
review recomputed the result from every raw leg and verified the same decision.
C3 was **not run**, exactly as required by the conditional protocol. There was
no replacement, discarded leg, extension, pooled data or post-hoc time correction.

- Block geometric B/A ratio: 0.7657594631275376.
- Model bounds: [0.5572769140708023, 1.0522372999195762].
- Mean log ratio: -0.26688717535992756; sample SD: 0.47445412942957005.
- Upper log bound: 0.05091865914412769, which fails the required <=log(0.98).
- Descriptive native medians: A 11.186386648500001 s, B 10.4687420595 s.

The model bounds are the preregistered one-sided 95% endpoints, equivalently a
two-sided 90% interval, not a two-sided 95% interval. Severe drift/dispersion and
the monitoring defect below undermine performance attribution. Neither the
nominal geometric ratio nor the favorable median establishes a speedup or
rescues the failed decision. No overall or case-specific performance gain is
retained. Functional correctness is separate from performance acceptance.

### All 32 primary times and diagnostic clocks

A = main, B = candidate. Native seconds below retain the raw primary values.
Other clocks are rounded for presentation. Solve CPU/wall are matching inner
solve scopes; unlike native time, they exclude constructor/setup and post-process.
CPU includes useful arithmetic, runtime/OS work and spinning. Guard clocks are
external-controller inclusive values, **not exclusive overhead**.

| Leg | Block | Order | Arm | Native s | Solve CPU s | Solve wall s | Guard wall s, inclusive | Guard CPU s, inclusive |
|---:|---:|:---:|:---:|---:|---:|---:|---:|---:|
| 1 | 1 | BAAB | B | 10.21399691 | 31.914436 | 10.149090 | 0.031336 | 0.031279 |
| 2 | 1 | BAAB | A | 9.296431943 | 29.156583 | 9.259997 | 0.022042 | 0.021883 |
| 3 | 1 | BAAB | A | 10.290457295 | 31.455188 | 10.248584 | 0.032707 | 0.032673 |
| 4 | 1 | BAAB | B | 12.5319708 | 35.294784 | 12.493258 | 0.241183 | 0.239913 |
| 5 | 2 | BAAB | B | 10.5513676 | 32.969164 | 10.507853 | 0.058507 | 0.058427 |
| 6 | 2 | BAAB | A | 12.397912625 | 36.895222 | 12.355209 | 0.266777 | 0.266591 |
| 7 | 2 | BAAB | A | 10.06562644 | 31.296176 | 10.014718 | 0.078861 | 0.078563 |
| 8 | 2 | BAAB | B | 11.160710875 | 33.866893 | 11.062002 | 0.100478 | 0.100434 |
| 9 | 3 | ABBA | A | 29.155759424 | 74.529052 | 29.072108 | 19.676965 | 19.107802 |
| 10 | 3 | ABBA | B | 31.651350081 | 84.124473 | 31.588988 | 11.377004 | 9.581863 |
| 11 | 3 | ABBA | B | 8.952468916 | 27.941096 | 8.904416 | 0.080735 | 0.080671 |
| 12 | 3 | ABBA | A | 9.147098965 | 27.700007 | 9.108624 | 0.097647 | 0.097835 |
| 13 | 4 | ABBA | A | 10.039807678 | 29.319644 | 9.994660 | 0.101591 | 0.101558 |
| 14 | 4 | ABBA | B | 10.947291319 | 33.561612 | 10.904418 | 0.113254 | 0.113276 |
| 15 | 4 | ABBA | B | 9.516771014 | 30.182038 | 9.474359 | 0.080912 | 0.080867 |
| 16 | 4 | ABBA | A | 8.952814163 | 27.195136 | 8.906507 | 0.115941 | 0.115745 |
| 17 | 5 | ABBA | A | 11.094234152 | 34.378237 | 11.054707 | 0.142009 | 0.141981 |
| 18 | 5 | ABBA | B | 10.386116519 | 30.905804 | 10.347991 | 0.294850 | 0.294811 |
| 19 | 5 | ABBA | B | 10.62068404 | 32.283331 | 10.561021 | 0.130731 | 0.130687 |
| 20 | 5 | ABBA | A | 10.029598511 | 31.720912 | 9.988281 | 0.144127 | 0.144093 |
| 21 | 6 | BAAB | B | 10.283894484 | 30.687399 | 10.224040 | 0.135569 | 0.134802 |
| 22 | 6 | BAAB | A | 14.819951781 | 30.663880 | 11.440072 | 14.454774 | 12.988499 |
| 23 | 6 | BAAB | A | 20.438437006 | 51.626479 | 20.399937 | 5.197083 | 5.054390 |
| 24 | 6 | BAAB | B | 10.301221954 | 32.376393 | 10.263680 | 0.151908 | 0.151868 |
| 25 | 7 | ABBA | A | 36.317009749 | 88.629097 | 36.194402 | 85.629715 | 80.419522 |
| 26 | 7 | ABBA | B | 9.510975153 | 29.697289 | 9.466380 | 0.164595 | 0.164555 |
| 27 | 7 | ABBA | B | 10.588867628 | 32.811524 | 10.547056 | 0.193158 | 0.193135 |
| 28 | 7 | ABBA | A | 31.051270757 | 74.836776 | 30.995259 | 69.900504 | 63.899166 |
| 29 | 8 | BAAB | B | 10.647415026000001 | 32.195040 | 10.571873 | 0.209158 | 0.209119 |
| 30 | 8 | BAAB | A | 11.278539145 | 33.879115 | 11.228872 | 0.238310 | 0.238273 |
| 31 | 8 | BAAB | A | 32.26351903 | 86.490293 | 32.215658 | 42.755582 | 30.512998 |
| 32 | 8 | BAAB | B | 9.922533008 | 30.966659 | 9.878013 | 0.216498 | 0.216483 |

### Variability and external-monitor limitation

The external controller permits successful periodic SIGALRM handler reentry
while recursively scanning the growing output directory. Three per-leg guard
wall accumulations exceed their child elapsed intervals (legs 25, 28 and 31).
For example, leg 25 reports 85.630 s guard wall and 80.420 s guard CPU during
about 36.466 s of child monitoring. These are per-invocation snapshot deltas,
not whole-campaign totals. Recursive inclusive accumulation is a code-supported
explanation; exact nesting depth and exclusive controller CPU were not recorded.
They must not be subtracted from native times or interpreted as measured
exclusive overhead. The monitor can itself perturb scheduling, but the existing
data cannot quantify its causal share or exclude external contention.

A's first/second-half median native times were 10.052717059/17.6291943935 s,
while B's were 10.7493294595/10.3436692365 s. Matching solve CPU also rose
substantially for A, so the spikes are not merely wall-time descheduling.
Within-arm exact points and seven counters do not prove identical low-level
execution: the recorded RNS prime-group route count ranges from 688 to 699
for A and 689 to 705 for B. The contribution of dynamic split/path mix is
unquantified. No subset or half is a replacement performance estimate.

One leg's native time includes 3.379 s of setup, which is outside cpu.solve;
loading, audit, pause and bookkeeping clocks are also outside the primary
scope. Requested pauses were always 2 s, but observed pauses ranged from
2.00010 to 4.38127 s. No cap failure was recorded; polling is not a hard kernel
limit, and reentry/delays can weaken its responsiveness.

Available metadata showed a nine-CPU allowed mask, not proof of nine dedicated
physical cores. Quota/pressure histories were unavailable; external activity
remains unknown. No governor, affinity, priority, security or cache settings
were changed, and no unknown process was stopped. No claim identifies the sole
cause of the observed variation.

The next proposed experiment first validates a new bounded-cost,
non-reentrant external supervisor with fabricated children/files and explicit
stop/resource/overhead tests. Only after review would a separately preregistered
independent experiment be considered. This is not an extension or rescue of
the completed trial. The solver source and original frozen evidence stay intact.

All raw campaign points, receipts, audits, logs and private analysis remain
local. This document is a curated numeric report, not a redistributed benchmark
suite or input. Full timing CSV SHA-256:
`b6ab9020bfdd38c333416036aa296c8f7b76864b88746c61d3c30e47cac4c2d8`.
All-data variability report SHA-256:
`cb3c57f0b9b3d79a264cddffcaa45fe2321a58a3c9c03d65dcffa9fdf1c71b45`.
Complete campaign summary SHA-256:
`f3e9fb7afd11f665f6373bfe4120cbac80f31ee8b10a0a37f5810806499989f0`.
Final independent campaign review SHA-256:
`802c620f59cf6f5798b79ee4b6f4cb6183e0cf1840942b486d88767c1582e020`.
Final campaign summary SHA-256:
`86fb0b13e52e80b15d344b23ee89d1d89ddabe7835ab8cd1da3606e48a75e7c5`.
The local 404-file campaign proof manifest SHA-256 is
`99356fdc9d93f40ae478404c96076a1cb335828c8598be92cc33683ca52faaaf`.
