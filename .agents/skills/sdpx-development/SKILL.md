---
name: sdpx-development
description: Build or validate the SDPX Rust core and Julia frontend, including precision changes and performance experiments.
---

# SDPX development

Applies to the current Rust-backed project, not the stable sibling `SDPX.jl`.
Follow [AGENTS.md](../../../AGENTS.md); do not load legacy solver/physics rules.

## Build and checks

From the project root, use the pinned Cargo.lock. Local setup:

```sh
export CARGO_HOME=/Users/xuyongjun/.local/share/sdpx-toolchain/cargo
export RUSTUP_HOME=/Users/xuyongjun/.local/share/sdpx-toolchain/rustup
export PATH="$CARGO_HOME/bin:$PATH"
export JULIA_DEPOT_PATH=/Users/xuyongjun/Desktop/project/SDPX/rebuild-env-depot:$HOME/.julia

cargo test --locked --offline -p sdpx-solver --features sdp-accelerate --lib FILTER
julia --startup-file=no --project=julia/SDPX.jl -t1 --gcthreads=1 julia/SDPX.jl/test/runtests.jl
```

Replace FILTER with an affected test, or use `--test NAME` for its integration
target. Add `faer-sparse` when exercising that backend. Release-library builds
use `-p sdpx-ffi --features sdp-accelerate,faer-sparse`; Linux uses the campaign's
pinned BLAS feature, commonly `sdp-openblas`. Resolve a missing offline cache
without changing lock versions.

Julia 1.12.6 is the local baseline; 1.13 is a separate compatibility leg.
`SDPX_LIBRARY` selects the matching frozen library. Check that the environment
loads this package rather than the sibling. CI work follows the affected
workflow; this project currently has no `.github/workflows` directory.

## Precision changes

MPFR destinations and scratch values must have independently owned storage.
Shallow array copies do not prove ownership. Do not change global precision or
rounding concurrently. Check the changed modes and relevant cancellation,
factorization residual, aliasing and reuse cases.

If explicitly testing sibling BFLA/MFLA providers, use separate processes and
environments with `--gcthreads=1`; their legacy acceptance is not a prerequisite
for ordinary Rust-core changes.

## Benchmarks

Load only the relevant driver instructions:
- [Research library](../../../benchmark/research/README.md): fixed suites and screen/full comparisons.
- [Float64](../../../benchmark/float64/README.md): Clarabel.rs/MOSEK adapters.
- [MPFR](../../../benchmark/mpfr/README.md): provider kernels.
- [Parallel](../../../benchmark/parallel/README.md): orthant/sampled diagnostics.
- [Ising](../../../benchmark/ising/README.md): matched SDPX/SDPB acceptance.

Use external immutable arms and pinned BLAS/solver budgets. Keep the evaluator
fixed. A short screen guides development but does not establish performance parity.
Local focused checks are allowed; actual cluster work uses `ucas-hpc` and the
user's campaign scope. Documentation does not itself start a campaign.
