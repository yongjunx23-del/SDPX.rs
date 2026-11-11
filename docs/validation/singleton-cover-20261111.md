# Singleton-cover admission: fresh correctness checkpoint (2026-11-11)

**Draft checkpoint. Performance is pending; do not merge.**
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
unchanged original-coordinate accuracy gate. Within-arm repeat stability is a
separate pending performance-campaign requirement.

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
complete C3/C4 solves below are uninstrumented and do **not** observe full-solve
cohort/kernel counts. Counter changes are indirect evidence only. Formatting
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

## Pending performance decision and merge restrictions

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

Performance campaign, exact-head GitHub CI and final review remain pending.
Do not merge unless all applicable correctness, performance and review gates
pass. No measured performance gain or general nonregression is claimed here.
