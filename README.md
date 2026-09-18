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

For x86 backend evaluation, `sdpx-ffi` also forwards `sdp-mkl`,
`pardiso-mkl`, and `pardiso-panua` to the Rust solver. Select one BLAS provider
per build; PARDISO features select a sparse solver backend separately. These
optional configurations require their native dependencies and are not qualified
performance defaults. Changing Julia's BLAS alone does not change the BLAS linked
into the Rust library.

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

`prepare(...)`, `solve!(problem; q=..., b=...)` and `close(problem)` support repeated solves.
This reuses the prepared workspace, not a primal/dual warm-start point; each solve
uses the core's default initialization. Explicit warm starts are unsupported. Julia's reusable handles retain Ruiz but disable presolve and chordal rewriting so that the original input structure stays updateable. Direct `solve` / `solve_conic` and model optimization use all three defaults. See the [PMP2SDP callback example](julia/SDPX.jl/test/pmp_callback.jl).

## Architecture

| Component | Responsibility |
|---|---|
| `julia/SDPX.jl` | Modeling, MOI, bulk conversion and result recovery |
| `crates/ffi` / `include/sdpx.h` | Versioned C ABI and Rust-owned prepared handles |
| `crates/solver` | Homogeneous embedding, cones, KKT, refinement and termination |
| `crates/arithmetic` | Independently owned fixed-precision MPFR values |

Float64 uses native BLAS/LAPACK with QDLDL or optional multithreaded Faer for sparse KKT factorization. Batched Schur assembly uses runtime-checked AVX2/FMA on supported x86 CPUs, with the same arithmetic implementation and a portable fallback. Auto selection can use dense block Cholesky for a near-dense positive block with a small negative border, retaining sparse fallback and refinement. The condensed backend eliminates PSD/orthant rows, retains other cones and equalities, and preserves the structural Schur sparsity. PSD cones cache matrix-sized scaling factors instead of a dense Hessian over packed cone coordinates. Both formulations use the same embedding, accepted-iterate recovery and native linear-solve refinement. Exact repeated coefficient columns share transforms across precisions; Float64 batches them, while MPFR streams them with bounded scratch. Presolve uses one bounded GMP rational elimination for exact redundant equalities, including their right-hand sides, and restores original-coordinate slacks and duals.

Set `Limits(threads=...)` for independent cone work, eligible sparse residual products, sampled block operators and Float64 Faer factorization. MPFR uses serial QDLDL and a dense provider with Householder/bidiagonal-QR SVD and symmetric Householder/QL eigenanalysis. NT scaling uses direct SVD at every precision; cone blocks can execute in parallel. Fused accumulation in the MPFR BLAS applies at every precision from 128 to 2048 bits. Large single orthants also split independent elementwise phases across the existing pool, preserving reduction order. `execution_plan(result)` reports the formulation, factorization width and cone pool size. These are configured capacities, not a count of busy cores; the reported cone pool size is the requested budget, so a request above the number of independent cones can be slower than a smaller one. Native BLAS threads are configured separately; use one BLAS thread when measuring cone/Faer scaling and set `RAYON_NUM_THREADS` to the requested factorization width.

An explicit `sampled_program` input retains shared basis factors and uses paired operators plus NT Gram Schur assembly. Exactly equal basis vectors share Gram columns at the working precision; canonical variables and weights remain unchanged. Ordinary CSC input keeps its existing semantics. Exact equality and infinite-bound row reductions retain the factors and remap block offsets; chordal block rewriting uses the materialized fallback. The optional PMP2SDP extension supplies this representation without a mandatory frontend dependency. Float64 and 256/512-bit integration tests cover this path, including row recovery, chordal fallback, updates, shared columns and deterministic threaded operator results. The ordinary CSC part of a sampled operator also uses the pool through row and column lanes, reproducing the serial product exactly. Dominant blocks can use the existing pool for MPFR matrix products, with triangular SYRK work balanced across tasks; every other block keeps the outer block level as its only parallel level. More configured workers do not guarantee a whole-solve speedup.

BFLA/MFLA, CRT acceleration and MPI are not implemented as backends in this project. The stable sibling retains its existing capabilities. The Julia frontend and shared library must both use ABI 3; an older library is rejected explicitly.

## Verification

The [benchmark library and research loop](benchmark/research/README.md) provide pinned development/regression inputs, separately reserved holdouts, paired time/RSS measurements and accuracy-first candidate decisions. Experiments have explicit budgets and completion criteria; accuracy checks stay outside solver timing.

```sh
cargo test --locked --release --workspace --features sdpx-ffi/sdp-accelerate,sdpx-ffi/faer-sparse -- --test-threads=1
julia --project=julia/SDPX.jl julia/SDPX.jl/test/runtests.jl
```

Measured counts on the reviewed tree: 498 Rust tests pass across all workspace
targets (321 solver-library unit tests and 4 doc-tests included in that total),
and the Julia suite reports 673 of 673. Quote these with the command above, not
as a fixed project property.

## Performance status

Performance parity is not established. On the retained Float64 conic10/holdout
campaign, SDPX/MOSEK warmed API-time ratio was 1.54 across 14 jointly accepted
cases; optimal-point counts were 4/10 and 12/15 for SDPX versus 7/10 and 12/15
for MOSEK. Across 11 accepted Clarabel.rs holdout cases the ratio was 0.91,
dominated by one SDP case. These historical observations do not establish broad
superiority; native solver and frontend memory measurements have different scopes.

The medium dense SDP development case now has a 3.15 s native median versus
3.34 s in a matched forward/reverse comparison (18 versus 19 iterations,
one thread, unchanged 1e-6 tolerances and external gates). A bounded quadratic
curve search reuses the existing predictor/corrector directions without extra
KKT solves, building on compact Schur assembly and exact coefficient reuse.
Five independent LP/SOCP/SDP cases and the 344 Rust checks of that round pass;
[Verification](#verification) records the current count and its command. At 1e-8 tolerances,
both baseline and candidate return AlmostOptimal on medium, so that accuracy
remains unqualified. Retained MOSEK is 1.89 s / 16 iterations; broad parity
is not established. Current optimization is single-core; the curve is enabled
only for Float64 symmetric cones, leaving high precision and nonsymmetric cones
on their existing step strategy. Exact presolve now removes 14 redundant medium
equalities without approximate rank tests; its matched timing is unchanged.
The unified paths preserve the 512-bit Ising solution, but do not show a new
Ising speedup. Enabling the curve at high precision was slower and was rejected.
See [current priorities and evidence](PERFORMANCE_PLAN.md).

On the local 3D Ising Lambda=11 sampled case (512-bit, 322 variables, 2558 rows,
50 iterations, every returned point audited externally at 1e-30), native-solve
medians are 58.6 s / 33.4 s / 19.6 s / 13.0 s at 1/2/4/8 cone workers, i.e.
1.00x / 1.76x / 2.99x / 4.52x. Measured phase shares at eight workers are 41%
KKT solves, 26% KKT update (of which the reduced MPFR factorization is a
thread-independent 3.0 s), 15% cone scaling and 11% step lengths. The reduced
Schur factorization and its triangular solves do not scale (1.0x and 1.8x at
eight workers) and account for about 42% of that solve, so they are the current
scaling barrier and the first target for any further parallel work; see the
review round in [the performance plan](PERFORMANCE_PLAN.md).

No SDPX-versus-SDPB speed claim is made here. Earlier campaigns ran on other
sources, hosts and core counts than the current tree, so their seconds are not
comparable and must not be quoted as current evidence. Scaling is re-measured on
one node at a fixed core set before any such claim is restated; the receipts live
in `PERFORMANCE_PLAN.md` under an explicit campaign identity.

The larger Lambda11 case still fails its sampled dual-consistency gate, and no
large-SDP superiority is claimed.

## Julia API details

PSD rows use upper-column svec packing with square-root-of-two off-diagonal
scaling. `Model`, `variable!` and `constraint!` use ordinary symmetric matrices.
MOI accepts Float64 models with affine or quadratic objectives and nonnegative,
zero, second-order, PSD triangle, exponential and power cone constraints.
Constraint primal/dual getters recover original MOI coordinates, including PSD
trace scaling, interval duals and infeasibility rays. Dual signs follow MOI's
constraint convention for both minimization and maximization. Raw optimizer
attributes accept writable `Settings` fields plus `threads`/`verbosity` aliases;
`limits` and `tolerances` are constructor/read-only groups, not raw setters.
BigFloat direct/model calls require matching
`Settings(BigFloat; precision_bits=...)`. Prepared settings are fixed at creation;
`solve_time` excludes Julia conversion and result copying.

```julia
p = prepare([1.0], sparse(reshape([-1.0], 1, 1)), [-1.0], [NonnegativeConeT(1)])
try
    result = solve!(p; b=[-2.0])
finally
    close(p)
end
```

Optional PMP2SDP tests use an isolated environment with that package developed
alongside this frontend. Julia sources derive from the sibling's modeling,
storage, compiler, result and MOI layers; their MIT notice remains in
`julia/SDPX.jl/LICENSE`. The solver executes in Rust.

## References and license

Clarabel's homogeneous-embedding interior-point design is a modern, mature foundation, not a universal best algorithm. [The Clarabel paper](https://link.springer.com/article/10.1007/s12532-026-00320-7) describes its quadratic-objective and cone treatment. [Hypatia](https://arxiv.org/abs/2107.04262) is a useful reference for more general cones, and [SDPB](https://arxiv.org/abs/1909.09745) for structured, parallel high-precision SDP.

The Rust core retains Clarabel's Apache-2.0 license and attribution. Reused SDPX.jl code retains its MIT notice in `provenance/SDPX.jl-LICENSE`. Source mappings and original file hashes are in `provenance/`. Native dependency licenses remain with their packages.
