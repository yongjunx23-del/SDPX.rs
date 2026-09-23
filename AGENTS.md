# SDPX

## Working agreement

- Follow the [workspace delegation policy](../AGENTS.md) when independent work
  benefits from delegation. Keep tightly coupled changes together and preserve
  edits made by other workers in the shared checkout.
- Give workers a concrete file ownership and acceptance check. Comparable
  numerical timings run serially on each host; freeze benchmark sources before
  measuring them.
- Complete requested checks and preserve sibling/reference repositories,
  attribution, licenses, and temporary experiment artifacts outside the repo.

## Solver contracts

- Keep one Rust solver with a native API, CLI, and C ABI. Julia programs in
  this repository are independent input generators or numerical auditors; they
  must not load a solver package or act as a frontend.
- Preserve precision, original-coordinate outputs, accepted-iterate recovery,
  convergence/infeasibility criteria, regularization, and refinement. Do not
  weaken accuracy gates, mix precisions, use precision-ladder warm starts, or
  add benchmark-name branches.
- Runtime numerical checks follow Clarabel.rs. Preserve its reduced tolerances,
  regularization/refinement and infeasibility semantics; do not reintroduce the
  retired independent certificate stage or status promotion. Tests and benchmark
  audits inspect returned points and rays outside solver timing.
- The optional `tol_feas_componentwise` dual-feasibility criterion is disabled
  by default. Keep global checks, external gates, and `AlmostSolved` semantics.
- Direct solves default to Ruiz, presolve, and chordal preprocessing. Prepared
  handles retain Ruiz and disable structural preprocessing for reusable updates.
- MPFR values must own their storage. Sampled factors define their operator;
  do not replace authoritative factors with rounded materializations.
- Prefer small typed interfaces and keep adapted production code focused. Aim
  for 20–30k production lines by removing duplication, not by hiding code behind
  wrappers or compressing formatting. Retain upstream attribution.

## Fast development workflow

For the active performance goal, use one representative complete E2E solve as
the per-change acceptance: check returned status, original-coordinate
residuals/gap, and total solve time at unchanged precision and tolerances. Build
the CLI when needed to run that solve. Do not require focused tests, a case
screen, repeated A/B for correctness, microbenchmarks, thread matrices, or the
full suite at each milestone. For a performance comparison, when practical run
frozen A/B executables sequentially in an interleaved A–B–B–A order on one
host; concurrent solves contend for the same CPU. Keep existing test and
benchmark tools available for other explicitly requested work.

```sh
# Reuse this fast CLI build for local end-to-end checks.
cargo build --locked --offline --profile fast -p sdpx-solver --bin sdpx \
  --features sdp-accelerate,faer-sparse
```

Do not use the nine-case set or full workspace suite as routine gates for this
goal. New numerical outcomes must satisfy existing status, original-coordinate
residual, gap, and objective tolerances. Iteration count and time are diagnostics
unless performance is the subject; do not require exact iteration/objective
matches after a valid algorithm change. Keep known failures labeled as failures,
and never weaken accuracy gates to make a check pass. Fix warnings in touched
code when convenient; unrelated warnings do not block.

Use `--profile fast` for development E2E and `--profile release` for performance
E2E. Compare identical settings and report timing scope. Treat one noisy timing
as preliminary rather than adding a repeat requirement. MOSEK/SDPB and cluster
comparisons are later plan milestones, not per-edit checks.

Read the relevant entry:

- [README](README.md): native installation, API, architecture, and licenses.
- [Review and plan](REVIEW_AND_PLAN.md): current priorities and unqualified work.
- [sdpx-development](.agents/skills/sdpx-development/SKILL.md): build setup,
  precision ownership, and benchmark routing.
- [Benchmark protocol](benchmark/research/README.md): research experiments.
  Use the available `ucas-hpc` skill for cluster work.
