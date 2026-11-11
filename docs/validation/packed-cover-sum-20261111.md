# Packed exact two-cover sums: performance inconclusive, do not merge

**DRAFT / DO NOT MERGE. The fixed release experiment is inconclusive; no speed benefit has been established.** This change
is independent of the unmerged singleton-admission experiment in PR #24.
It is based on main `88f5100c7dba0ade45634408bf793d8fcab3d59b` and retains
main's requirement for exactly two cover anchors.

## Source and numerical contract

Source commit: [`2d3310c44016f6af2050349f0c30f095e8f6de2f`](https://github.com/yongjunx23-del/SDPX.rs/commit/2d3310c44016f6af2050349f0c30f095e8f6de2f).
Its tree `fd8b9361b7203482c157658819c7aa07a4055ce0` exactly matches the tested
local source tree (local commit `4282d788a82f59fd310ee8e02fb9e7041967caf8`).
Upload commit metadata differs; this documentation is a separate child.
All 255 Rust mode/blob entries remain unchanged by the documentation, with
compact sorted `[path,mode,type,blob]` JSON SHA-256
`ebe4d6606ca4029108e570f8d3e192593b771f653e8f022448a9e896d7b95b54`.

Only `two_cover_product` changes in production. After the unchanged
`V = op(A) B^T`, directly pack the upper triangle of `U V^T + V U^T` rather
than materializing full `W = U V^T` and reading `W + W^T`. Off-diagonals use
four products; diagonals compute the two-product `q` and then `q + q`.
Nominal multiply/add counts and the caller's scratch allocation are unchanged.
No RSS reduction is claimed. Linux's default integer-GEMM dispatcher here is
`faer_seq`, not OpenBLAS; the first GEMM is unchanged.

Balanced residues satisfy `|U|, |V| <= (p-1)/2`, with `p < 2^26`.
Every four-term absolute partial sum is less than `p^2 < 2^52`. Products,
partial sums, cancellation and fused forms therefore remain exact binary64
integers, and reduction receives the same integer as before. This is not a
precision reduction: requested MPFR precision, Plan, primes, encoding, caches,
CRT, single final rounding, admission and scheduling remain unchanged.

Production RNS file SHA-256:
`932e5f576cdb799a63f0493f5161d6f97b48c38fd8cfd3a836d2fe56deab0d0d`.
Test file SHA-256:
`f5aa44d21f1864e99512419b6e1213dcc92314b5c8c91ed048be632279e6fe4b`.

## Checks actually completed

The existing two tests remain byte-identical. One new synthetic i128 oracle
forms full `X op(A)^T`, then `op(A)` times that result modulo each prime;
it does not reuse floating GEMM, production reduction/halving or the rank-two
identity in the reference calculation.

- Existing residue test: 8,844 outputs and 132 balanced-half comparisons.
- Existing full-congruence test: 18,240 512-bit upper comparisons, including
  N/T, padded input, cache miss/hit and structural/shape fallbacks.
- New independent i128 test: 64 cases, 23,584 outputs, including 11,792 exact
  cancellation outputs, eligible prime extrema, odd intersections, diagonal
  doubling, N/T, padding and packed-tail sentinels.
- Exactly one complete C4 solve: true 256 bits, requested/actual four cone
  and four backend workers, BLAS/OMP one, actual `condensed_arrow` /
  `condensed`. `Solved`, 60 iterations. The complete serialized point,
  all seven mathematical counters, effective settings and runtime identity
  exactly match the independently audited main-A reference.
- A separate 180-decimal-digit original-coordinate audit passed all eight
  normalized metrics at the unchanged `1e-30` gate.

Counters: batches 51, factor attempts 52, factorizations 52, linear solves
160, outer refinements 27, refinements 0, RHS applications 133.

Audit normalized values below are rounded for display; the gate was evaluated
at 180 decimal digits:

- slack_equation: `1.092943E-71`
- stationarity: `4.993855E-35`
- gap: `9.838385E-32`
- raw_complementarity: `9.847550E-32`
- returned_complementarity: `9.847550E-32`
- raw_primal_psd: `0`
- returned_primal_psd: `0`
- dual_psd: `0`

These are custom normalized original-coordinate checks, not a rigorous
interval certificate. Matrix parity treats signed zeros equivalently, as the
unchanged tests did; the complete SDP serialized point was compared exactly.

## Build and evidence limits

Rust/Cargo 1.98.1, Linux, native CPU flags, default + serde + faer-sparse,
matching six linked providers (Debian pthread OpenBLAS 0.3.29 among them).
This is a correctness-only `fast` profile run through the reviewed
precision-256 CLI-derived adapter, **not** a full multi-precision CLI benchmark.
Production binary SHA-256:
`40122173eceafe13d9eb2048b6a8ce30d0a159ed2636b02cf64cda4d6a68c231`.
The recorded native binary hash is an artifact identity, not a reproducible
binary-build promise across hosts or different Git metadata.

Local whole-solver lib-test compilation exceeded its 1,200-second cap; it is
retained as a failed build, not a passing suite. The accepted replacement
compiled the entire byte-identical production RNS module and the exact three
test bodies in a focused component harness. Receipt/profiling shims were inert
only under the explicitly disabled test environment. The real production
library/adapter was separately built and its dependency closure verified to
contain no harness or shims before the complete SDP/audit run. This is targeted
component testing, not a full local workspace-suite pass.

Solver/auditor were single-process, threads allowed, with AS 2 GiB, CPU 600 s,
per-file output 2 MiB and core 0; solver external/internal limits 150/120 s,
audit external 300 s. Known aggregate output and disk reserve were checked
before/after stages, not enforced as a kernel aggregate quota. AS can influence
memory planning; these are constrained correctness checks, not proof of
ordinary-resource route equivalence. Child exits/reaping and actual limits
were verified. No claim of exclusive host use is made.

Independent functional closeout SHA-256:
`8727fcb45493f8a8b3856eb7be5047113a0522bf29fd42ebf63ff5e7d3c4d956`.
Independent focused-test/production-build proof manifest SHA-256:
`007c19b18b35428e66145c89a0a979fd88be6b52e3d1452319dc783cd200e1c7`.

## Input provenance and what remains

Public SDPLIB `control4.dat-s`, retrieved from its pinned compressed source
in the separately distributed [SDPA-GMP dataset](https://github.com/nakatamaho/sdpa-gmp/blob/ca110db5ea1cc46e811b70dfea9cbb25db74448d/SDPLIB/data/control4.dat-s.xz).
No SDPA-GMP implementation or dataset file is included in this repository.
Raw input SHA-256:
`f1086b1a6abdc82bc1b27e6d4ad8fd7986fbb7e73c86e6475f2ab8db16da3815`.
Converted input SHA-256:
`91eb458cfcfb2181ad2a8c06405beaa0bd92ada1a3bf75cd70f6f0bd9b5775ff`.
Strict settings SHA-256:
`85d554342a952b7ba1dac601a5a1529a365d69908d8339b2a8da8817e39b2c97`.
Settings retain `1e-30` absolute/relative gap, feasibility and original-coordinate
tolerances, four workers and a 120-second internal limit; other defaults and
full effective settings match main A.

The static C4 classification contains 190 existing two-cover coefficient
matrices; that is not a measured call count or CPU share. The completed release
comparison below did not pass its prespecified performance gate. C3 regression
and complete 512-bit SDP runs were therefore not started. Keep this PR draft;
correctness or CI success does not establish a performance benefit. Benchmark
inputs, suites, runners, raw logs, frozen points and binaries remain local per
repository policy; only this curated record and source changes are published.

## Completed fixed release comparison: inconclusive

This is a distinct prospective experiment, not a repeat or pooling of PR24 or
the failed custom-observer attempt. The production Rust blobs above stayed
unchanged. Baseline is main `88f5100c`; candidate is the exact tested source tree
`fd8b9361`. Frozen release-256 binary SHA-256 values:

- A: `2fcd3c1ff2907fefb8c5cc0c5d5d7905ba36117690e607040db3c236f96ad51c`
- B: `130efd2a6c184240e1e984295c2e7b272e57d18cf3efa3fc9edf4f856b7435f1`

Both used O3, thin LTO, one codegen unit, native CPU flags, the same toolchain,
features, six linked providers, C4 input and strict settings identified above.
True 256 bits; requested and actual four cone/four backend workers; BLAS/OMP
one and Rayon four. No integer-GEMM override, so Linux's default is faer_seq.
These are receipt-enabled native-solve timings through matching narrow
CLI-derived adapters, without the temporary custom observer. Built-in receipt
phase/CPU clocks remain enabled. They are not full multi-precision CLI,
zero-overhead, exclusive-host or ordinary-resource timings. Hashing/audits
occurred outside the native solve clock; no time was subtracted or corrected.

The protocol and explicit order were independently reviewed and frozen before
any release point. Exactly one A and one B validation-only complete solve/audit
preceded eight fixed four-leg blocks. Validation roles are excluded forever
from the primary. A fixed two-second pause followed every audited leg,
including validations and the final leg. No sample replacement, extension,
repeat, old-data pooling or alternate primary was permitted.

For each block, `L = (log(B1)+log(B2)-log(A1)-log(A2))/2`. The decision uses
`mean(L) + t(0.95,7)*sd(L,ddof=1)/sqrt(8)`, with
`t(0.95,7)=1.894578605061305`. Retention required its exponent to be at most
`0.98`, a one-sided 95% upper bound demonstrating at least a 2% gain under the
stated model. The displayed pair of endpoints is a two-sided 90% Student
interval, not a two-sided 95% interval. Only C4 passing would permit the
predeclared 32-block C3 noninferiority phase (upper bound at most `1.02`).

All 34 complete C4 solves were `Solved/60`, with exact full main-A serialized
point, all seven counters, effective settings and runtime parity. Every solve
passed all eight separate original-coordinate 180-digit `1e-30` audit gates
(272 individual metrics). All 68 solver/auditor process-limit and reaping
receipts passed. No incomplete or favorable-only numerical samples were kept.

Native seconds below are rounded to nine decimal places; all 32 timed legs
are shown in their frozen order. Validation-only A/B: `10.768462386` / `10.424065907` seconds, excluded.

| Block | Order | Native seconds in order |
| --- | --- | --- |
| 1 | BAAB | 10.371415587, 10.151110225, 9.483509393, 9.256414835 |
| 2 | ABBA | 10.047714132, 9.568222797, 10.050019552, 10.162039455 |
| 3 | ABBA | 10.293281269, 9.699359139, 11.994353503, 10.271384564 |
| 4 | BAAB | 11.007662561, 10.549730372, 10.408860146, 9.951295999 |
| 5 | ABBA | 10.160335992, 10.900959607, 9.907424052, 9.906510616 |
| 6 | BAAB | 9.930551842, 10.572056747, 10.835022052, 9.934485584 |
| 7 | BAAB | 9.388445217, 10.870576650, 11.743613674, 11.051190855 |
| 8 | ABBA | 11.710029059, 12.336577256, 12.435949852, 10.672942035 |

**Result: INCONCLUSIVE.** Geometric B/A is `0.996827713825668`; the two-sided
90% interval is `[0.953243923890503, 1.04240422220100]`. The one-sided 95%
upper bound `1.04240422220100` misses the required `0.98`. This neither proves
a speedup nor establishes neutrality. Native arm medians are A `10.3510707075`
and B `10.0006577755` seconds; their favorable descriptive ratio `0.96614718`
does not override the failed primary gate.

Order diagnostics also differ: ABBA geometric B/A `1.03965540`, BAAB
`0.95576427`; first/second halves `1.00381098` / `0.98989303`. These are retained
as descriptions, not a basis for selecting samples, adjusting time, identifying
a host-load cause or rescuing the experiment. The Student model assumes
independent, approximately normal block logs; balance cannot remove nonlinear
drift, autocorrelation or heavy tails, and eight blocks may be underpowered.
No exclusive host use or causal explanation for the variability is claimed.

C3 validation/timing was correctly **not run**. There was no repeat or later
512-bit run. The whole campaign exited normally in `678.116063391` seconds;
explicit controller bookkeeping before its final write was `21.211494619`
seconds. All-data API/process/CPU/RSS values remain in the sealed local packet;
they are descriptive, not alternative acceptance endpoints. Per-child AS2GiB
may affect planning. CPU600s, per-file2MiB, solver external/internal150/120s,
audit300s, campaign5400s/C4phase1800s, known-output256MiB and10GiB disk reserve
were predeclared. Output/disk/parent bookkeeping checks are boundary checks,
not continuous aggregate kernel quotas. No generic process/history scanner ran
during child execution; cooperative locking does not imply host exclusivity.

Independent final closeout passed, confirming the complete data, roles,
sequence, bounds, provenance, all numerical gates and the absence of C3. A
passing closeout verifies the inconclusive result; it does not turn it into a
performance pass.

Evidence SHA-256 values:

- Protocol: `a0cb40ad3a7fa5504ba4a5b48329b5aebe9961e9d4b9eb0110b639b3d689e259`
- Sequence: `e6d639cc230546965d4d8cec2050d8405984b877786e99cd4202cf8ff702e6b1`
- All34 timing/usage CSV: `8e2376ba729240c984cab288d4d9575f7d8ce0760f2457a827b6ec4c6dee20ba`
- Result: `9f12b2f0a54db5971416a554ae9174c97468fa9e153ca837280b53099588f338`
- 490-file proof manifest: `efb7b6c016561a3fdc04efd8fddb1c0e96d4a61dcd2a4b341a17eb79fb8872c2`
- Independent closeout: `c1c3cbbdbde0bbe861fb98b7abb4c4f276f4ad54bb089fc22cbedeb2b7ad96d3`

CI run [43019210047](https://github.com/yongjunx23-del/SDPX.rs/actions/runs/43019210047)
passed all five jobs on documentation head `1b788e60`, whose Rust blobs match
the tested source. Any later documentation-head CI is distinct. The full local
lib-test cap failure remains recorded above. No speed/memory benefit or merge
is justified, and the experiment stays draft/unmerged. A distinct next research
step is measuring the currently unclassified serial-cover CRT/encoding cost;
no such diagnostic or new optimization result is claimed here.
