---
name: sdpx-development
description: Build or validate the SDPX Rust core, native CLI, C ABI, precision paths, and benchmark contracts.
---

# SDPX development

Applies to the Rust-backed SDPX solver, native CLI, and versioned C ABI. Julia
programs are allowed only as independent benchmark input generators or
original-coordinate audits; there is no Julia solver frontend or package
environment to load.

## Fast development loop

For the current performance goal, use one representative complete E2E solve per
change. Check status, original-coordinate residuals and gap at unchanged
precision/tolerances, and record full solve time. Do not require focused tests,
a multi-case screen, repeated A/B, microbenchmarks, a thread matrix, or the full
suite as routine acceptance gates. Preserve known failures and never relax
accuracy thresholds. Existing tools remain available when explicitly requested
for another purpose.

`--profile fast` uses release arithmetic with LTO disabled and parallel codegen.
Use it for local CLI builds; use `release` only for performance numbers.

```sh
export CARGO_HOME=/Users/xuyongjun/.local/share/sdpx-toolchain/cargo
export RUSTUP_HOME=/Users/xuyongjun/.local/share/sdpx-toolchain/rustup
export PATH="$CARGO_HOME/bin:$PATH"

cargo build --locked --offline --profile fast -p sdpx-solver --bin sdpx \
  --features sdp-accelerate,faer-sparse
```

The `fast` binary lands in `target/fast/sdpx`; copy it out before the next
rebuild when an immutable comparison arm is needed. Build the CLI only when
needed for a complete E2E solve. Use `release` for quoted performance
measurements; avoid rebuilding or rerunning unchanged E2E arms.

Focused tests and full suites remain available when explicitly requested, but
they are not milestone gates for this optimization goal.

## Precision changes

MPFR destinations and scratch values must own storage. Shallow array copies do
not prove ownership. Do not change global precision or rounding concurrently.
For this goal, check precision preservation and the original-coordinate result
through the representative high-precision E2E solve; do not add a per-mode test
matrix. Decimal JSON values must reach MPFR without a Float64 intermediate.

If explicitly testing sibling BFLA/MFLA providers, use separate processes and
environments with `--gcthreads=1`; their legacy acceptance is not a prerequisite
for ordinary Rust-core changes.

## Benchmarks

Load only the relevant driver instructions:

- [Research library](../../../benchmark/research/README.md): fixed suites and native CLI comparisons.
- [Float64](../../../benchmark/float64/README.md): native CLI and Clarabel/MOSEK reference adapters.
- [MPFR](../../../benchmark/mpfr/README.md): provider kernels.
- [Parallel](../../../benchmark/parallel/README.md): orthant/sampled diagnostics.
- [Ising](../../../benchmark/ising/README.md): matched native SDPX/SDPB acceptance.

Use external immutable arms and pinned BLAS/solver budgets. Keep process,
native/API, audit, and memory scopes separate. Documentation does not start a
campaign. Actual cluster work follows `ucas-hpc` and the user's explicit
campaign scope.

## Performance work

For an optimization, use existing evidence to choose the hotspot and run one
matching complete solve in `release`. Keep settings identical; verify
original-coordinate precision and report time, memory if already available,
precision, and thread count. Use external solvers, Ising, or cluster only at the
corresponding plan milestone. Do not require repeated timing runs; label a noisy
single E2E measurement preliminary.
The existing `SDP_control3` accuracy failure remains a failure; it is neither
waived nor a routine gate for unrelated changes.

Do not create a second solver or benchmark framework. Reuse sampled operators,
cone/Arrow pools, original-coordinate audits, and the existing watchdog. Do not
restart rejected experiments without new workload or cost evidence.
