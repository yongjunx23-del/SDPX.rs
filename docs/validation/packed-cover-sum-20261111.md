# Packed exact two-cover sums: functional validation, performance pending

**DRAFT / DO NOT MERGE. No speed benefit has been established.** This change
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

No release performance comparison or complete 512-bit SDP run is claimed.
The static C4 classification contains 190 existing two-cover coefficient
matrices; that is not a measured call count or CPU share. Performance,
regression, needed precision validation, exact-head CI and final review remain
pending before promotion. Keep this PR draft and do not merge on functional
checks or CI alone. Benchmark inputs, suites, runners, raw logs, frozen points
and binaries remain local per repository policy; only this curated record and
source changes are published.
