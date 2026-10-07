# SDPX architecture

SDPX is one homogeneous self-dual (HSD) interior-point engine, derived from
Clarabel.rs, generic over the scalar
`FloatT = Scalar + BlasFloatT + LDLConfiguration`. The same code serves
Float64 and every MPFR width. The native Rust API, the `sdpx` CLI and the C
ABI are thin frontends over it; the PMP converter only produces inputs, and
Julia scripts only generate inputs or audit outputs.

This file describes the design. Working rules and numerical contracts are in
[AGENTS.md](../AGENTS.md); measured status, known failures and open work are
in [REVIEW_AND_PLAN.md](../REVIEW_AND_PLAN.md); experiment evidence (timings,
job IDs, rejected trials) is in [JOURNAL.md](JOURNAL.md).

## Crates

| Crate | Responsibility |
|---|---|
| `crates/arithmetic` | `Scalar` trait, `MpFloat<N>` (MPFR value with owned limbs), inline rounded arithmetic, `exactdot`, decimal and wire formats. |
| `crates/solver` | Generic solver, native Rust API and the `sdpx` CLI. |
| `crates/pmp` | PMP JSON/XML reader and the `sdpx-pmp2sdp` converter to sampled SDP inputs. |
| `crates/ffi` | C ABI and prepared handles; header in `include/sdpx.h`. |

PMP conversion streams blocks and writes rows directly. Each worker releases
the parsed/generated basis after both parity arrays are written, before
building normalized `c`/`B` coefficients.

## Solver modules

| Module | Responsibility |
|---|---|
| `solver/core` | HSD predictor/corrector loop and the traits it drives (data, variables, residuals, KKT, info, solution); callbacks and checkpoints. |
| `solver/default` | The standard problem: `PreparedProblem` setup, Ruiz equilibration, presolve, settings, convergence tests, JSON I/O, data updates. |
| `solver/cones` | Zero, nonnegative, SOC, exponential, power, generalized power and PSD-triangle cones; the composite cone and its worker pool. |
| `solver/kkt` | Augmented (`direct`) and condensed formulations, LDL backends (`ldl`) and iterative refinement. |
| `solver/sampled` | Factored PSD operators (bases × sample weights) for bootstrap SDPs. |
| `solver/chordal` | Sparse PSD chordal decomposition and solution recovery. |
| `solver/distributed`, `mpi.rs` | Owner-partitioned implementation of the core traits; MPI loaded at runtime. |
| `algebra` | CSC/dense matrices, `BlasFloatT`, MPFR dense kernels, exact residue (RNS) products in `dense/blas/rns_blas.rs`, per-thread scratch. |
| `qdldl` | QDLDL factorization with elimination-tree parallel scheduling. |
| `receipt`, `timers` | Phase wall/CPU timings and peak RSS (`SDPX_RECEIPT`, `SDPX_PROFILE`); the solve clock. |

## Setup

External input is validated at the boundary: `PreparedProblem` checks
dimensions, CSC structure, cone parameters and settings for every frontend;
the readers and C ABI validate their transport formats. Internal code trusts
the structures the solver builds and uses `debug_assert!` for its invariants.

Ordinary direct solves apply presolve, chordal decomposition and Ruiz
equilibration. Presolve removes infinite orthant bounds and exactly dependent
equalities, merges SOC tail coordinates with zero rows of `A` (a constant
tail makes the cone an orthant row), and fixes variables whose only nonzero is
a single-entry orthant or equality row (objective constant reported in both
costs); postsolve lifts `x`, `s`, `z` and certificates. Prepared handles keep Ruiz but disable structural
preprocessing so later data updates keep the prepared structure. Consumed
JSON problems move their owned P, q, b and A into setup instead of copying.
Explicit, automatic and MPI owner partitions use one setup route; MPI derives
the local owner from its communicator. Cost-history input fingerprints are
computed once at the JSON boundary, only for history import or recording.
Ordinary solves do not collect history metadata. Immutable local/global
layouts share storage when all owners are local.
Sparse row-plan construction reuses row offsets as insertion cursors, then
restores their starts; no temporary cursor copy is allocated. Entry order
and the retained row layout are unchanged.
Solutions and residuals are reported in original coordinates. Checkpoints
store the internal iterate with its equilibration; hot starts map through
original coordinates. Loading rejects nonfinite values and nonpositive
homogenization or scaling factors before installing the iterate.

Sampled problems keep the factored operator authoritative: once installed,
the retained linear-A values are released and the reduced KKT storage owns
the coefficients. Rounded materializations never replace the operator.
Fully sampled KKT setup retains column patterns without generic PSD entry
coordinates or value plans. Sampled MPFR workspaces retain packed upper Grams;
SYRK and MPI gather read/write that layout directly. Float64 keeps dense BLAS
Gram storage. V is allocated on its first owner update; global MPI non-owners
use only the gathered Gram. Fused adjoints allocate on first use, so global
MPI keeps none. The fused RHS matrix is allocated only on that path;
serial and owner-local solves retain it. Gram and scaling ownership can
differ, so other scaling matrices remain available to their consumers.
Condensed PSD blocks keep R for Float64 application and G for MPFR application;
the unused matrix is empty. Ginv remains for Schur assembly when needed.
MPFR SVD stores rotations in separate U/V logs without redundant factor flags;
the logs reuse thread-local capacity, and the four-row parallel replay keeps
each row's scalar rotation order.
Serial exact sampled adjoints emit workspace results directly in block/column
order; pooled adjoints use the same writer to fill their contribution buffers.
Inverse scalar adjoints with distinct basis columns write exact quadratics
directly into that output; compacted bases retain an indexed projection.

## Iteration

Each HSD iteration:

1. updates residuals and the convergence/infeasibility tests;
2. scales the cones (NT scaling; PSD blocks use Cholesky + SVD);
3. updates and factors the KKT operator;
4. solves the affine and combined predictor/corrector directions — the
   constant and affine right-hand sides share one batched solve;
5. chooses the step — symmetric-cone problems with orthant or SOC rows first
   apply up to two Gondzio centrality correctors (orthant products, SOC
   spectral values of the scaled Jordan product and τκ pushed into
   [0.1σμ, 10σμ], kept only when the step grows by 1%, skipped once α ≥ 0.9),
   and Float64 symmetric problems also try a quadratic curve on the affine and
   combined directions — and updates the iterate, keeping the previous
   accepted iterate for recovery.

Newton solves use iterative refinement against the true, unshifted operator;
batched right-hand sides share each correction solve, and on the parallel
Float64 row plan each residual pass (MPFR exact residual rows and serial
Float64 residuals run per right-hand side).
At MPFR precision every refinement residual row is an exact dot product
rounded once, so refinement keeps working on badly scaled bootstrap systems.
Every right-hand side uses its own residual and measured correction gain;
stalls from another right-hand side or factorization do not skip refinement.
Condensed solves retain the final accepted H·z product in refinement workspace
and snapshot only earlier right-hand sides. MPI completes the product gather
before marking it valid; rejected refinement steps restore the accepted product.
Clarabel stopping rules, reduced tolerances, infeasibility detection,
regularization with escalation and the distinct `AlmostSolved` status are
part of the engine. There is no low-precision solver, mixed-precision
factorization or precision-ladder warm start.

## KKT formulation and backends

`kkt_form` is `auto`, `augmented` or `condensed`.

- **Augmented** factors the full quasi-definite system.
- **Condensed** eliminates PSD, nonnegative and small (dimension ≤ 16)
  second-order rows into a Schur complement on the primal variables and keeps
  the other cone rows. An eliminated SOC block contributes BᵀB with
  B = W⁻¹A_K, using the explicit NT factor
  W⁻¹ = η⁻¹[[w₀, −w₁ᵀ], [−w₁, I + w₁w₁ᵀ/(1+w₀)]]. Orthant and SOC rows share one
  Gram: columns present in at least half the rows form a dense panel (one
  SYRK), the rest use a precomputed pair plan; at MPFR precision every entry
  is an exact sum rounded once, and rows touching only panel columns use a
  cached exact residue congruence Aᵀdiag(1/w²)A. `auto` condenses
  ordinary PSD data when it has at least 256 PSD `svec` coordinates and the
  reduced dimension is at most a quarter of that, comparing structural
  storage estimates of both forms. Sampled blocks apply their factored
  operator directly. Condensed assembly writes into the primal prefix of the
  direct layer's KKT matrix, so there is one authoritative KKT CSC.

With `direct_solve_method = "auto"`, the direct layer tries the structural
backends in order, then the scalar's general backend. The active backend is
reported as `linear_solver`.

| Order | Backend | Selection |
|---|---|---|
| 1 | `local_soc_arrow` | ≥ 8 SOC3 leaves with a nonempty equality border of ≤ 128 coordinates and no cross-leaf coupling. |
| 2 | `shared_soc_arrow` | Zero and ≥ 8 SOC3 cones only (augmented form; condensed eliminates SOC3), shared primal border of ≤ 128 coordinates, workspace ≤ 512 MiB (the arrow memory cap). A leaf's single primal column follows its SPD cone block and keeps its true diagonal (no static shift) until a factorization fails (approved 2026-10-07). |
| 3 | `local_bounds_faer` / `local_bounds_arrow` | ≥ 64 variables with one or two local bound rows, diagonal `P` and a nonempty equality/free border, admitted by added storage. Float64 needs `faer-sparse`; MPFR uses exact bound Gram products. |
| 4 | `dense_block` | Float64: an eligible dense leading positive block, pooled tiled Cholesky. |
| 5 | `arrow` | Disconnected positive dense leaves around a negative border, estimated workspace ≤ max(512 MiB, 8 × KKT storage) (Float64 and MPFR). Leaf contributions are recomputed in bounded batches. |
| 6 | `faer` / `qdldl` | General sparse LDL. Float64 with `faer-sparse` uses faer when estimated flops ≥ 1e8 and flops per factor nonzero ≥ 40, else QDLDL. MPFR uses QDLDL. |

Packed local bound leaves store their couplings in shared panels; their
unused dense coupling matrices, per-border coupling index/mask vectors and
pivot signs are not allocated. The bound factorizer supplies pivot signs;
generic and unpacked local leaves initialize their sign storage once.
Float64 flat bound-leaf forward/backward solves share the existing pool in
1024-leaf ranges. Each leaf owns disjoint output entries and retains its
arithmetic order; phases join before the intervening panel products.
Bound products write their Gram into the existing Schur destination
(Float64 lower triangle, MPFR upper triangle) before the unchanged
shifted-border subtraction and mirror. The MPFR exact kernel and dot fallback
share that destination; no separate bound Gram is retained.

Eligible MPFR local SOC Schur products encode at most 256 coupling rows per
group directly from leaf operands, without gathered MPFR panels. The full
finite/exponent scan derives common row supports; equal supports share compact
products, while dense rows retain the original layout. Workers scatter and sum
reduced residues across groups, then one shared CRT rounds
each entry once before subtracting it from the shifted border. Upper ZᵀY
uses the same products as the original lower YᵀZ, retaining scalar-rounded Z.
Residue/product scratch stays private to each worker; no full tall operand
cache is retained. Small shapes and declined plans use the original exact dots.

Local SOC leaves keep raw coupling row 0 in the unchanged first row of Y;
B retains only raw row 1. Updates, scales, finite checks and fallback
materialization read the original values there. Refactors reload Y row 1
before applying the same forward solve and diagonal scaling.

Above 256 bits, substantial generic MPFR arrow leaves form scalar-rounded
Z = Y ⊙ dinv, then compute the upper YᵀZ product through exact residues.
Result storage must stay proportional to Y; batches fill the worker pool,
hold one packed upper result per worker and scatter disjoint border columns.
Each entry subtracts leaves in their original order.
Short leaves and underfilled batches retain column-parallel exact dots. The plan records the current
performance acceptance gate for these candidates.

`qdldl` can be pinned for any scalar and `faer` for Float64. A failed arrow
factorization falls back to QDLDL and a failed dense block to its sparse
backend; both keep the requested precision and refine against the original
operator. Static shifts are applied to the factorization only; the residual
operator stays unshifted.

## Arithmetic

`MpFloat<N>` owns its limbs. For regular values, multiplication through 1280
bits and addition/subtraction through 512 bits round inline to nearest-even;
128/256-bit FMA and two-product FMMA use exact stack kernels with one
rounding. All other cases call MPFR and produce the same bits.
Inline power-of-two products copy the other mantissa and adjust its exponent;
results outside MPFR's exponent range retain MPFR handling.

`exactdot` accumulates the exact sum of products in fixed point and rounds
once (inline limb products through 256 bits, GMP above). The result does not
depend on term order, partitioning or thread count. Large dense MPFR
products, congruences (`Aᵀdiag(d)A`) and bound Grams use exact residue (RNS)
kernels with caches for operands that stay fixed during a solve; CRT
reconstruction rounds each entry once. GEMM, general congruence and sampled
quadratic/bilinear products use one streaming and prime-partition implementation.
CRT constants are built from one temporary large-integer cofactor at a time.
Pooled calls share one CRT accumulator:
independent prime products run in parallel, then disjoint digit columns update
the accumulator. Fractional residues and mapped compaction keep their serial
order. Congruence retains full square residue staging; quadratic/bilinear
tasks share operand staging and keep private matrix/product scratch. Diagonal
congruence retains its row partitions and releases scaling residues after
their joined products, before CRT allocates its buffers.
Ordinary square GEMMs of side ≥12 use the residue kernel from 1024 bits;
SYRK, upper products and cached congruences keep their existing eligibility.
Each ordinary GEMM still rounds separately, including congruence fallback
intermediates.
MPFR transposed SYRK uses contiguous exact slice dots when residues are
unprofitable. Upper GEMM requests stream packed
triangle residues through CRT. Contiguous results use implicit indices;
dense upper triangles keep explicit maps. Temporary BLAS tiles cover only the needed
rows of each column tile. MPFR sparse products `Ax`/`Aᵀx`
accumulate each output row/column exactly when it has enough entries.
Ordinary scalar operations and factorizations stay at the requested
precision. Binary64 products with A route columns that are at least a quarter
full through a dense BLAS panel (`algebra/csc/dense_columns.rs`), independent
of the thread count.

`BlasFloatT` maps Float64 to system BLAS/LAPACK and wide scalars to the MPFR
dense kernels (GEMM/SYRK through `exactdot`/RNS, Cholesky, triangular
solves, symmetric eigensolver and a bidiagonal-QR SVD whose rotations are
logged and replayed row-parallel). Four-row replay tiles preserve each row's
rotation order. Float32 is configured only for dense kernel tests.

## Parallelism and memory

One worker pool per solve (`max_threads`) runs cone blocks, factorization
and long vector work; dominant blocks split their tiles over the same pool.
Equality-only problems also share this pool with KKT kernels when they meet
the existing work cutoff. Backends receive it before initial thread reporting.
Splits preserve each output's operation order, so results are independent of
the thread count. Scratch is per thread (`algebra/scratch.rs`) and reused
across iterations; dense kernels reuse dead input/output storage instead of
allocating temporaries. Parallel packed Float64 Schur tiles use worker
scratch; the per-block dense accumulator is allocated only for serial
assembly. PSD scaling lends its consumed second Cholesky
matrix to the right-only SVD as workspace, then restores the matrix length
before constructing the inverse, including on SVD failure. Margins and step
bounds also lend a dead matrix to the eigensolver, retaining only its integer
workspace separately. BLAS providers must support concurrent calls from
workers (source-built OpenBLAS needs
`USE_LOCKING=1`).

MPI is loaded at runtime. The ordinary MPI path replicates input data and
shards the expensive work by blocks. Allgathered scalars decode directly into
their destination through the canonical wire codec; the collective format
checks and fatal decode agreement remain. Sampled forward/adjoint products
and condensed scaling share one rank partition for pooled problems containing
only sampled PSD and zero cones, so a condensed solve keeps intermediate
vectors on their owning rank and exchanges each result once. Ordered owned
forward products write directly into those output rows when no gather is
requested. Serial sampled
products and mixed ordinary cones keep the complete scaling exchange;
the sampled operator's linear part is sharded by columns (adjoint) or by
its entry-holding rows (forward). Sharded, pooled and serial sparse products
use the same per-output arithmetic, so results do not depend on the thread
count. The generic `arrow` backend distributes its
leaves: each rank factors a cost-balanced contiguous range of leaves, the
border Schur complement is summed in rank order and factored on every rank,
and solves gather the leaf solutions. Exact refinement residual rows are
split across ranks by work. `--partitions N|auto` selects owner partitioning,
where ranks own whole blocks and share the equality Schur complement, and
`--cost-history-in/out` feeds measured block costs to the balancer. Both
paths plug into the same core HSD loop. Owner sums fold gathered values into
existing buffers in rank order, retaining the collective sequence and
failure handling without allocating a separate folded result.

## Build features

BLAS/LAPACK are always linked. Without a provider feature, `build.rs` links
Accelerate on macOS and the system dynamic OpenBLAS on other Unix targets
(`OPENBLAS_LIB_DIR` adds a search path; `SDPX_BLAS_LINK=none` defers to
`RUSTFLAGS`, for example `-l dylib=openblas`).

| Feature | Effect |
|---|---|
| `sdp-accelerate`, `sdp-openblas`, `sdp-mkl`, `sdp-netlib` | BLAS/LAPACK provider (solver and FFI crates). |
| `faer-sparse` (default) | Float64 faer sparse LDL and packed local-bound kernels. |
| `serde` (solver default) | JSON input, output and settings; required by `sdpx`. |
| `all-precisions` | Frontend MPFR dispatch for 128–2048 bits in steps of 64 (default: 128, 256, 512, 768, 1024). Native `MpFloat<N>` types are unaffected. |

The converter dispatches only MPFR widths. Build and verification commands
are in [AGENTS.md](../AGENTS.md) and the
[development skill](../.agents/skills/sdpx-development/SKILL.md).
