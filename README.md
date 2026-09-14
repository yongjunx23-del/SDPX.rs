# SDPX

A Julia interface to a Rust conic solver, with BLAS/LAPACK and MPFR/GMP numerical backends.

SDPX solves

```math
\min_x \tfrac12 x^T P x + q^T x \quad\text{subject to}\quad Ax+s=b,\;s\in\mathcal K.
```

The Rust engine is adapted from [Clarabel.rs](https://github.com/oxfordcontrol/Clarabel.rs). The Julia modeling interface reuses SDPX.jl. This project has its own directory and package identity; the existing stable solver remains available in the sibling `../SDPX.jl`.

## Features

- LP, convex QP, SOCP, SDP, exponential, power and generalized power cones.
- One predictor/corrector engine for Float64 and fixed 128, 256, 512, 768, 1024 and 2048-bit MPFR arithmetic.
- Julia modeling and MathOptInterface integration; prepared problems with reusable `q` and `b` updates.
- Bulk C ABI: data crosses the language boundary during preparation, updates and result retrieval. Solver iterations remain in Rust.
- Automatic augmented/condensed KKT selection, preserving sparse Schur blocks and the original Newton equations. `Settings(kkt_form=:augmented)` or `:condensed` selects a formulation explicitly.
- Cached worker pools for independent cone phases in Float64 and MPFR arithmetic.
- Original-coordinate primal, dual and slack outputs, ordinary infeasibility detection and accepted-iterate recovery. Runtime convergence and linear-solve checks follow Clarabel.rs; independent accuracy checks run in tests and benchmarks. Direct solves enable Ruiz equilibration, presolve and chordal decomposition by default.

## Installation

Requires a Rust toolchain, a C toolchain for GMP/MPFR, and Julia 1.12 or later. From this directory, on macOS:

```sh
cargo build --locked --release -p sdpx-ffi --features sdp-accelerate,faer-sparse
julia --project=julia/SDPX.jl -e 'using Pkg; Pkg.instantiate()'
```

On Linux, use `sdp-openblas,faer-sparse` instead. OpenBLAS source builds also require a Fortran compiler. The dependency versions are pinned in `Cargo.lock`. Julia can invoke the build with `include("julia/SDPX.jl/deps/build.jl")`; `CARGO` selects the Cargo executable.

Start Julia with `--project=julia/SDPX.jl`. The package finds `target/release/libsdpx` automatically; `SDPX_LIBRARY` can select an explicitly built library.

```julia
using SDPX, SparseArrays

# min x subject to x >= 1
result = solve_conic([1.0], sparse(reshape([-1.0], 1, 1)), [-1.0],
                     [NonnegativeConeT(1)]; return_result=true)
status(result), result.x
```

High precision uses `BigFloat` input/output with a fixed precision selected for the prepared handle. Unsupported precision values are rejected. Decimal transport preserves the supplied bits without a Float64 intermediate.

```julia
setprecision(BigFloat, 256) do
    settings = Settings(BigFloat; precision_bits=256)
    solve_conic(BigFloat[1], sparse(reshape(BigFloat[-1], 1, 1)), BigFloat[-1],
                [NonnegativeConeT(1)]; settings, return_result=true)
end
```

`prepare(...)`, `solve!(problem; q=..., b=...)` and `close(problem)` support repeated solves. Julia's reusable handles retain Ruiz but disable presolve and chordal rewriting so that the original input structure stays updateable. Direct `solve` / `solve_conic` and model optimization use all three defaults. See [Julia usage](julia/SDPX.jl/README.md) and the [PMP2SDP callback example](julia/SDPX.jl/test/pmp_callback.jl).

## Architecture

| Component | Responsibility |
|---|---|
| `julia/SDPX.jl` | Modeling, MOI, bulk conversion and result recovery |
| `crates/ffi` / `include/sdpx.h` | Versioned C ABI and Rust-owned prepared handles |
| `crates/solver` | Homogeneous embedding, cones, KKT, refinement and termination |
| `crates/arithmetic` | Independently owned fixed-precision MPFR values |

Float64 uses native BLAS/LAPACK with QDLDL or optional multithreaded Faer for sparse KKT factorization. The condensed backend eliminates PSD/orthant rows, retains other cones and equalities, and preserves the structural Schur sparsity. PSD cones cache matrix-sized scaling factors instead of a dense Hessian over packed cone coordinates. Both formulations use the same embedding, accepted-iterate recovery and native linear-solve refinement.

Set `Limits(threads=...)` for independent cone work, eligible sparse residual products, sampled block operators and Float64 Faer factorization. MPFR uses serial QDLDL and a dense provider with Householder/bidiagonal-QR SVD and symmetric Jacobi eigenanalysis; cone blocks can execute in parallel. Large single orthants also split independent elementwise phases across the existing pool, preserving reduction order. `execution_plan(result)` reports the formulation, factorization width and cone pool size. These are configured capacities, not a count of busy cores. Native BLAS threads are configured separately; use one BLAS thread when measuring cone/Faer scaling and set `RAYON_NUM_THREADS` to the requested factorization width.

An explicit `sampled_program` input retains shared basis factors and uses paired operators plus NT Gram Schur assembly. Ordinary CSC input keeps its existing semantics. Sampled input falls back to materialized CSC when structural preprocessing changes its blocks. The optional PMP2SDP extension supplies this representation without a mandatory frontend dependency. Float64 and 256/512-bit integration tests cover this path, including preprocessing fallback, updates, shared columns and deterministic threaded operator results. Dominant blocks can use the existing pool for MPFR matrix products, with triangular SYRK work balanced across tasks. More configured workers do not guarantee a whole-solve speedup.

BFLA/MFLA, CRT acceleration and MPI are not implemented as backends in this project. The stable sibling retains its existing capabilities. The Julia frontend and shared library must both use ABI 3; an older library is rejected explicitly.

## Verification

The [benchmark library and research loop](benchmark/research/README.md) provide pinned development/regression inputs, separately reserved holdouts, paired time/RSS measurements and accuracy-first candidate decisions. See [the agent protocol](benchmark/research/program.md) for bounded optimization cycles; its checks stay outside solver timing.

```sh
cargo test --locked --release --workspace --features sdpx-ffi/sdp-accelerate,sdpx-ffi/faer-sparse -- --test-threads=1
julia --project=julia/SDPX.jl julia/SDPX.jl/test/runtests.jl
```

The frozen macOS candidate `a52c22f3366d` passes 376 Rust tests, 508 checks each on Julia 1.12.6 and 1.13.0, and 40 optional PMP2SDP callback/extension checks. Coverage includes upstream solver cases, MPFR ownership, dense decomposition residuals, all six precision modes, preprocessing and original-coordinate recovery, mixed-cone condensation, dependent equalities, Julia model/MOI conversions and serial/parallel consistency. PSD step-length eigenanalysis now shares the existing pool, retaining the original ordered step-bound reduction. These checks establish correctness for the tested cases, not performance parity.

The [Float64 benchmark driver](benchmark/float64/README.md) checks original-coordinate residuals outside timing. On the fixed conic10 and disjoint holdout15 suites, candidate `e5972b714552` passes 4/10 and 12/15 optimal-point gates; fresh MOSEK passes 7/10 and 12/15. Across 14 jointly accepted cases, SDPX's geometric-mean warmed end-to-end time is 1.54 times MOSEK's. Both use the same external gate; MOSEK's product-default internal tolerances differ. MOSEK's unvalidated `LP_agg` certificate is excluded from optimal-point comparisons.

Fresh default Clarabel.rs passes 11/15 holdout cases. On those 11 common accepted points, SDPX/Clarabel's geometric-mean time ratio is 0.91, largely driven by `SDP_arch0` (4.51 versus 18.36 s); several small LPs remain 4–18% slower. Two Clarabel cases have incomplete supervision/cleanup outcomes and no solver status or complete memory receipt; they earn no speed credit. Failed points, AlmostSolved and incomplete runs remain in the denominators. These observations do not establish broad superiority or a repeated 2% optimization gain.

Memory figures describe whole processes, including startup and external audits. On the 14 common MOSEK points, Julia/SDPX peak RSS is 658–739 MiB and Python/MOSEK is 63–164 MiB. The 11 accepted standalone Rust Clarabel holdout processes use 7–296 MiB. These different frontends prevent treating the figures as solver-kernel memory measurements. Timings exclude first-call compilation and use three warmed solves on conic10 and seven on holdout15.

The matched Ising512 campaign `212627.node220` on `node7` measured the following native solver medians (three repetitions, 512-bit arithmetic, public tolerance `1e-42`, unchanged external tolerance `1e-30`). All 28 returned points passed the external audit. Both solvers ran sequentially within one cluster allocation; SDPX used threads and SDPB used MPI ranks pinned to the same physical-core budgets.

| Cores | SDPX seconds | SDPB seconds |
|---|---:|---:|
| 1 | 135.00 | 127 |
| 2 | 74.17 | 69 |
| 4 | 42.46 | 36 |
| 8 | 30.24 | 25 |

This campaign uses source `a52c22f3366d` and input SHA-256 `e4484eb8895e504a8f5c83651a5b964172e24c7060db24ddf09b737dcc78fddf`. SDPX took 50 iterations; SDPB's final logged iteration was 201. Eight-core scaling is 4.46× versus 5.08×; SDPX remains about 21% slower at eight cores. SDPB native time has whole-second resolution; the solvers' stopping norms and per-iteration work differ. SDPX whole-invocation RSS is 615–676 MiB; SDPB's sampled aggregate group peak reaches 521 MiB at eight ranks. Neither is a kernel-only memory figure.

For this same small input on macOS, PSD step-length parallelism reduces eight-thread native medians from 36.55 to 17.07 s; reverse-order measurements give 36.38 versus 17.05 s (2.13–2.14× faster). Single-thread medians remain about 68 s. All 24 forward/reverse points pass the unchanged 512-bit accuracy protocol and return identical decimal vectors and iteration counts. These configured macOS thread budgets are separate from the physical-core cluster measurements above.

The larger Lambda11 input has 28 PSD blocks of orders 36–43. Its 512-bit SDPB reference failed Cholesky positive-definiteness; a separate fixed-768-bit reference passed. SDPX's upstream normalization, with a denominator near `2.03e78`, initially accepted an inaccurate point. A separately declared SDPX feasibility tolerance `1e-140` produces 94 iterations and global residuals around `1e-65`, but sampled dual consistency remains `2.16e-22`, above the unchanged external `1e-30` gate. This run earns no speed credit; a qualified large-instance comparison remains unresolved. Both 768-bit experiments retain the same 512-bit-converted input bytes. SDPB keeps absolute internal tolerances `1e-42`; equal numeric internal tolerances do not imply equal accuracy across these solvers.

## References and license

Clarabel's homogeneous-embedding interior-point design is a modern, mature foundation, not a universal best algorithm. [The Clarabel paper](https://link.springer.com/article/10.1007/s12532-026-00320-7) describes its quadratic-objective and cone treatment. [Hypatia](https://arxiv.org/abs/2107.04262) is a useful reference for more general cones, and [SDPB](https://arxiv.org/abs/1909.09745) for structured, parallel high-precision SDP.

The Rust core retains Clarabel's Apache-2.0 license and attribution. Reused SDPX.jl code retains its MIT notice in `provenance/SDPX.jl-LICENSE`. Source mappings and original file hashes are in `provenance/`. Native dependency licenses remain with their packages.
