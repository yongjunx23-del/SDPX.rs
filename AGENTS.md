# SDPX

One Rust conic solver (HSD interior point, Clarabel.rs-derived) with a native
Rust API, the `sdpx` CLI, a C ABI (`crates/ffi`) and the `sdpx-pmp2sdp`
converter (`crates/pmp`). Julia in this repo only generates inputs or audits
results; it never loads or drives the solver.

Priorities: [REVIEW_AND_PLAN.md](REVIEW_AND_PLAN.md). Architecture, backend
selection and build features: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
Experiment history: [docs/JOURNAL.md](docs/JOURNAL.md). Toolchain and pinned
cases: [sdpx-development](.agents/skills/sdpx-development/SKILL.md).

## Architecture

One generic engine, `FloatT = Scalar + BlasFloatT + LDLConfiguration`, serves
Float64 and every MPFR width. BLAS/LAPACK are always linked (no SDP-less build).

- `crates/arithmetic` — `MpFloat<N>` (MPFR storage, owned limbs). Products and
  sums of regular values round inline to nearest-even (multiply through 1280
  bits, add/sub through 512; MPFR otherwise, bitwise equal). `exactdot`:
  every dot product is the exact sum rounded once (inlined limb kernel through
  256 bits, GMP above). Wire/serde formats.
- `crates/solver/src/algebra` — CSC and dense matrices, `BlasFloatT` (system
  BLAS/LAPACK for f64; MPFR kernels for wide types), exact residue GEMMs
  (`rns_blas`: products, congruences, `Aᵀdiag(d)A`, with constant-operand
  residue caches), per-thread scratch.
- `solver/core` — the Clarabel-derived HSD predictor/corrector loop and its
  traits. `solver/default` — problem setup (Ruiz, presolve, chordal), variables,
  residuals, info, settings, JSON I/O.
- `solver/cones` — zero, nonnegative, SOC, exponential, power, generalized
  power and PSD-triangle cones, plus the cone worker pool.
- `solver/kkt` — `direct`: the augmented system factored by QDLDL (serial or
  elimination-tree parallel), faer, `dense_block` (pooled tiled Cholesky) or an
  arrow LDL that eliminates local structure (`local_bounds`, `local_soc`,
  `shared_soc`), chosen by `auto`; `condensed`: PSD/orthant blocks eliminated
  into a Schur complement, factored by the direct layer. Iterative refinement.
- `solver/sampled` — SDPB-style factored PSD blocks (bases × sample weights).
- `solver/chordal`; `solver/distributed`: owner-partitioned data, variables,
  residuals and KKT for MPI, plugged into the same core HSD loop
  (`for_blocks!`/`all_blocks!` run owner blocks on the pool or serially);
  `mpi.rs`.
- `receipt` (phase wall/CPU timings via `timeit!`, RSS), `timers` (the solve
  clock).
- `crates/pmp` — PMP → sampled SDP converter. `crates/ffi` — C ABI.

## How to work

Highest priority: a fast, memory-efficient solver at unchanged accuracy.
This is a personal research project: favor fast development and quick
end-to-end verification over broad test coverage. Make a small useful change,
run one matching solve, and continue once it passes.

- **No over-design.** Solve the current problem. No speculative options,
  traits, feature flags, config knobs or abstraction layers "for later".
- **No over-defensive code.** Validate external input once at the boundary
  (`PreparedProblem` setup, JSON reader, C ABI). Internal code trusts
  internal invariants: no re-checking CSC format, dimensions or signs that
  the solver itself built. Use `debug_assert!` for internal invariants, not
  runtime fallbacks. No catch-all fallback paths for states that cannot occur.
- **Keep code compact.** Prefer deleting duplication over adding wrappers.
  Code size is secondary to measured solver performance. Do not rewrite fast
  code to meet a line-count target. Don't change comments without reason.
- Preserve edits made by others in the shared checkout; never reset or clean
  the tree wholesale. Temporary logs, profiles and frozen copies go outside
  the repo.
- Preserve upstream attribution and licenses.

## Numerical contracts (do not change without explicit approval)

- Precision: no silent precision lowering, mixed-precision factorization,
  or precision-ladder warm starts. MPFR values own their storage.
  Correctly rounded exact accumulation (RNS, `exactdot`) is allowed.
- Keep Clarabel-style convergence, reduced tolerances, infeasibility
  detection, regularization (including escalation) and iterative refinement.
  Never promote `AlmostSolved`; `tol_feas_componentwise` stays off by default.
- Keep original-coordinate outputs and accepted-iterate recovery. Sampled
  factors define their operator; never swap in rounded materializations.
- Direct solves default to Ruiz + presolve + chordal; prepared handles keep
  Ruiz and disable structural preprocessing.
- No benchmark-name branches. Never weaken an accuracy gate to pass a check.
  Known failures stay labeled as failures (see the plan).

## Build: recompile only what changed

```sh
export CARGO_HOME=/Users/xuyongjun/.local/share/sdpx-toolchain/cargo
export RUSTUP_HOME=/Users/xuyongjun/.local/share/sdpx-toolchain/rustup
export PATH="$CARGO_HOME/bin:$PATH"
F=sdp-accelerate,faer-sparse   # macOS (also the plain-build default); cluster uses its validated BLAS provider
```

- Always `--locked --offline`, and always `-p <crate>` for the crate you
  touched. Never build `--workspace` in the edit loop.
- Keep one feature set per session (`$F`). Changing features rebuilds the
  whole solver.
- Build directly for the matching E2E run. Use `cargo check --locked --offline
  -p sdpx-solver --features $F` only when it helps resolve compile errors;
  do not require both check and build on every change.
- Dev/test profile keeps only line tables; dependencies are prebuilt at
  opt-level 2. Don't change profiles to chase one run.
- `fast` profile for local end-to-end runs; `release` only for quoted timings.
- Frontends compile six precisions by default (53, 128, 256, 512, 768, 1024).
  Add `all-precisions` only when you need another width (e.g. 1216 for
  Lambda43) — it makes the C ABI build ~6× slower.

## Verification: one quick end-to-end check

The default gate is one small complete solve that exercises the changed
path. Reuse an existing input and its original-coordinate audit. Do not run
unit tests or integration suites in addition as a routine gate.

```sh
python3 benchmark/e2e/e2e.py build --arm NAME
python3 benchmark/e2e/e2e.py run CASE --arm NAME
# CASE: medium (Float64), ising11 (MPFR/SDP), csdr3 (SOC)
```

- Docs/comments: `git diff --check`; no build or solve.
- Full suites only when asked or for a risky change:
  `cargo test --locked --offline --profile fast -p sdpx-solver --features $F`
  (also `-p sdpx-arithmetic`, `-p sdpx-pmp`, `-p sdpx-ffi --features $F`);
  the solver suite takes about five minutes cold.
- Solver change: pick the smallest relevant case, one precision and one
  thread count. Do not run all three pinned cases or all precision widths.
- CLI/API/C ABI/converter change: one small E2E through the affected entry
  point; converter checks include solving the converted output.
- Threading change: the same small solve at one and the affected thread
  count, comparing points and audits. MPI changes need an actual MPI run.
- Reuse the frozen baseline. A pure refactor must preserve its points;
  algorithm changes must pass status and the original-coordinate audit.
- Known failures remain labeled; they do not block unrelated work. Investigate
  a new failure or worse residual with the smallest reproducer.
- Stop verification when the matching check passes. Extra tests are only for
  a concrete unresolved failure/risk or an explicit user request. Do not add
  test scaffolding, defensive cases or coverage campaigns by default.
- Existing tests may help diagnose a specific problem; no mass test deletion
  or suite reorganization is part of this workflow change.
- No automatic full-suite, clippy, cross-solver comparison or release gate.
  Publishing alone does not trigger a broad test campaign.

## Performance evidence without slowing the edit loop

One fast-profile E2E is enough to advance development; its timing is
preliminary. Batch related edits before formal timing. For a performance
claim or choosing between candidates, run one release ABBA on the affected
case, sequentially on one host:

```sh
python3 benchmark/e2e/e2e.py build --arm NAME --profile release
python3 benchmark/e2e/e2e.py ab CASE OLD NEW
```

Use identical precision, settings, input and thread/BLAS budgets. Keep a
speed optimization on a repeatable ≥2% end-to-end gain, or a clear memory or
correctness benefit. Report the median and native/API/process scope. Repeat
only if noise leaves the decision unresolved. Record a short kept/rejected
entry in the journal; do not reopen closed directions without new evidence.

## Cluster work and delegation

- Keep small E2E checks local. Send long builds, solves and substantial
  benchmarks to the cluster using `ucas-hpc`; use its validated dynamic
  OpenBLAS configuration until the recorded static-provider issue is fixed.
- Work yourself by default, including job submission and monitoring. Use a
  subagent only when genuinely needed for independent, bounded work; a long
  run alone does not require delegation. No mandatory agent or model.
- Freeze source and inputs, use bounded resources, and record job state,
  exit code, status, audit, time, memory and evidence paths.
- Keep timed runs serial per host. Compare frozen sources, not a moving tree.
- Submit only work needed for the active task. Do not restart cancelled jobs,
  launch old campaigns or expand to large sweeps merely because the plan
  mentions them. No long run is required for a documentation-only edit.
