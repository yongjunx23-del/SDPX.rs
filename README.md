# SDPX

SDPX is a Rust conic solver with native Float64 and fixed precision MPFR/GMP
backends. It solves

```math
\min_x \tfrac12 x^T P x + q^T x \quad\text{subject to}\quad Ax+s=b,
\;s\in\mathcal K.
```

The supported solver surfaces are the Rust API, the `sdpx` command-line
executable, and the versioned C ABI. Julia appears only in independent
benchmark and audit scripts; it is not a solver interface or a required
runtime dependency.

## Build and run

The Rust workspace needs a Rust toolchain and a C toolchain for GMP/MPFR. On
macOS, a release CLI build is:

```sh
cargo build --locked --release -p sdpx-solver --bin sdpx \
  --features sdp-accelerate,faer-sparse
target/release/sdpx problem.json --threads 8 --output solution.json
target/release/sdpx /path/to/sdpb-json --precision 768 \
  --settings settings.json
```

On Linux use `sdp-openblas,faer-sparse` instead. The CLI has no Julia runtime
dependency. `sdpx --help` lists options. Its JSON receipt includes original
coordinate `x`, `s`, and `z`, status, iterations, precision, factorization and
thread receipts, and separate input, API, and native solve times. Exit code 0
denotes a full solved or infeasible status; code 2 denotes incomplete or
reduced accuracy; code 1 denotes input or I/O failure. `AlmostSolved` is not
promoted to a full result.

Conic JSON uses `P`, `q`, `A`, `b`, `cones`, and `settings` fields. `P` is an
upper-triangle CSC matrix. Float64 inputs use JSON numbers. MPFR coefficients
and settings use decimal strings; exact integer tokens are also accepted, while
fractional JSON numbers are rejected so they cannot pass through Float64 first.
The optional `--settings` file replaces the input settings object. With no
override, Ruiz equilibration, presolve, chordal preprocessing, and the NT
direction retain their native defaults. The CLI supports 53-bit Float64 and
128, 256, 512, 768, 1024, and 2048-bit MPFR arithmetic. PSD cones use NT. The optional
`--partitions auto` owner-local path supports local pools and MPI.
Distributed performance has not been qualified.
The optional MPI loader supports OpenMPI's exported-handle ABI and requires
`MPI_THREAD_MULTIPLE`, including when the host initializes MPI. Rust embedding
applications can initialize `MpiContext` before reading distributed inputs,
agree on their configuration, and call `finish()` on that same thread after
all solver work has joined (`mpi_finalize()` remains available). Host-owned
MPI remains the host's responsibility. The CLI initializes and finalizes MPI
explicitly, checks rank input/configuration agreement, and writes results and
the main receipt only on rank 0. MPI input must be a file or directory;
stdin broadcasting is not implemented.

Sampled SDP input accepts an uncompressed `pmp2sdp --outputFormat=json`
directory. The reader preserves factor-authoritative PSD blocks and returns
the sample variables, original SDPB `y`, block/parity row maps, and the
objective constant. Normalization metadata is not applied twice. The optional
`tol_feas_componentwise` setting adds a componentwise dual-feasibility check
to the existing full `Solved` checks; it is disabled by default.

Use SDPB's `pmp2sdp` to construct sample points, weights and bilinear bases.
SDPX consumes those decimal factors directly at the selected precision; it
does not resample the polynomial or introduce a separate sampling scheme.
Input-generation precision and solve precision must be recorded separately:
reading a lower-precision input at higher precision cannot recover lost digits.

## Rust API

The JSON reader and solver share the CLI input path:

```rust
use sdpx_solver::solver::*;
use std::fs::File;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let problem = JsonProblem::<f64>::read(File::open("problem.json")?)?;
    let mut solver = problem.into_solver()?;
    solver.solve();
    Ok(())
}
```

`read_sdpb_sampled::<T>(directory)` builds the same structured sampled input
used by the CLI. `JsonProblem::write` preserves factors and MPFR decimal
digits.

## C ABI

`include/sdpx.h` defines ABI version 4. `crates/ffi` owns prepared handles and
copies all input arrays during `sdpx_prepare` or `sdpx_prepare_sampled`.
`sdpx_solve`, `sdpx_update`, `sdpx_get_info`, `sdpx_result_f64`,
`sdpx_result_decimal`, and `sdpx_destroy` provide the lifecycle and bulk
result operations. Indices are zero-based, `P` is upper triangular, and MPFR
decimal buffers preserve the requested precision. ABI return codes are
separate from solver status in `sdpx_info`.

## Architecture

| Component | Responsibility |
|---|---|
| `crates/solver` | Homogeneous embedding, cones, KKT systems, refinement, termination, and structured sampled operators |
| `sdpx` CLI / `JsonProblem` | Native JSON input, settings, precision selection, and solution receipts |
| `crates/ffi` / `include/sdpx.h` | Versioned C ABI and Rust-owned prepared handles |
| `crates/arithmetic` | Independently owned fixed-precision MPFR values and arithmetic |
| `benchmark/research` | Fixed inputs, native process protocol, and external original-coordinate gates |

Float64 uses native BLAS/LAPACK with QDLDL or optional multithreaded Faer
factorization. MPFR uses owned GMP/MPFR scalars and serial QDLDL with bounded
parallel cone work. The condensed backend eliminates PSD/orthant rows while
preserving the Newton equations; augmented and condensed forms share
embedding, recovery, and refinement rules. Native receipts report the actual
`linear_solver`, `linear_solver_threads`, `cone_threads`, and `kkt_form`; these
are configured capacities, not measurements of busy cores.

The sampled operator retains shared basis factors and weights as the numerical
input. Materialization is a checked fallback for structural preprocessing, not
the source of sampled coefficients. Exact equality reduction preserves factor
metadata; a chordal structural rewrite may use a materialized fallback and
change coordinates, so acceptance audits the original problem independently.

## Verification and benchmarks

Use risk-based checks, not a full regression after every edit. Run the focused
test for the changed package and behavior. For a single cone/kernel change,
test that cone and precision; add a matching solve only if the focused test
does not cover it. Use the three-case LP/SOCP/SDP screen for shared defaults,
convergence/status, input/output, or cross-cone changes. The nine-case full
development set is for performance candidates and integration milestones.
Documentation-only edits need `git diff --check`; run benchmark-tool tests only
when that tooling changed.

For local end-to-end work, use `--profile fast` and reuse the binary. Use the
benchmark runner's three-case `--profile screen` when the CLI/result path
changed (`--stage development --threads 1`). Check status,
original-coordinate residuals, gap, and objective against existing tolerances;
iteration count and runtime are diagnostic unless being measured. A valid
algorithm change need not preserve exact iterations or objective digits.

At integration milestones or before release, run the full workspace suite:

```sh
cargo test --locked --release --workspace \
  --features sdpx-ffi/sdp-accelerate,sdpx-ffi/faer-sparse -- --test-threads=1
```

Run extended MPFR/threading checks only when changing the corresponding pool,
precision, parallel, or condensed-congruence behavior. Keep the known
`condensed_graded` failure labeled as a failure; do not describe it as passing.

```sh
cargo test --locked -p sdpx-solver --features sdp-accelerate,faer-sparse \
  --lib mpfr -- --ignored --test-threads=1
cargo test --locked -p sdpx-solver --features sdp-accelerate \
  --lib condensed_graded -- --ignored --test-threads=1
```

The [research protocol](benchmark/research/README.md) fixes input identities,
native CLI process scope, precision, settings, and independent audits for
candidate comparisons. `benchmark/float64` and `benchmark/parallel` launch the
same native executable directly. Their Julia programs only generate inputs or
run external original-coordinate audits, with oracle time and memory recorded
outside solver timing. The Ising controller follows the same rule for its
independent high-precision audit.

Use external immutable arms and pinned BLAS/solver budgets. Measurements run
serially on each host; no benchmark command starts a cluster campaign by
itself. Accuracy failures, timeouts, OOMs, and incomplete process receipts
remain in the denominator.

Recorded MOSEK and SDPB runs use different inputs, settings, machines, and
source revisions. They are reference context only and do not establish
performance parity for this solver.

## References and licenses

The Rust core is adapted from [Clarabel.rs](https://github.com/oxfordcontrol/Clarabel.rs)
and retains its Apache-2.0 attribution. Historical adapted source mappings are
listed in [`provenance/reuse.json`](provenance/reuse.json); the retained MIT
notice is [`provenance/SDPX.jl-LICENSE`](provenance/SDPX.jl-LICENSE). The
fixed-precision SVD port records its GenericLinearAlgebra MIT notice in that
mapping. Native BLAS, LAPACK, GMP, MPFR, and benchmark dependency licenses
remain with their respective packages.

[SDPB](https://arxiv.org/abs/1909.09745) documents the structured sampled SDP
form used by the native reader. [Hypatia](https://arxiv.org/abs/2107.04262)
and the [Clarabel paper](https://link.springer.com/article/10.1007/s12532-026-00320-7)
provide related cone-solver references.
