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
`--partitions auto` owner-local path supports local pools and MPI. For
multi-node runs, start with one rank per 8–16 cores. If OpenMPI's `openib`
transport misbehaves (hangs, or "wire encode/decode failed"), select TCP with
`--mca btl self,vader,tcp --mca btl_tcp_if_include <ib-or-eth-interface>`.
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

Solver source layout (`crates/solver/src/solver/`; the public API is the flat
`sdpx_solver::solver::*` facade):

| Module | Contents |
|---|---|
| `core/` | Generic HSD predictor/corrector loop, component traits, core settings |
| `default/` | Standard-format implementation: data, presolve, Ruiz, variables, residuals, KKT system, info |
| `cones/` | Cone implementations and the composite cone |
| `kkt/` | KKT solvers: `condensed/`, `direct/` (augmented LDL), `ldl/` backends, refinement |
| `sampled/` | Factor-authoritative sampled PSD operator and SDPB-style input |
| `distributed/` | Owner-partitioned MPI implementation of the core traits and collectives |
| `chordal/` | Chordal decomposition |

Unit tests live in each module's `tests/` directory.

Float64 uses native BLAS/LAPACK with QDLDL, or, with the `faer-sparse`
feature, faer's multithreaded supernodal LDLᵀ for high-fill factorizations.
MPFR uses owned GMP/MPFR scalars:
- Dense products run through an exact residue-number-system kernel: exact
  accumulation, one rounding.
- Block-structured KKT systems use a parallel arrow LDLᵀ.
- Otherwise QDLDL.
Cones, blocks and long vector operations share one worker pool. Step lengths
use certified binary64 screens with full-precision fallback, for PSD λmin and
for the exponential/power backtracking. The condensed backend eliminates PSD/orthant rows while
preserving the Newton equations; augmented and condensed forms share
embedding, recovery, and refinement rules. Native receipts report the actual
`linear_solver`, `linear_solver_threads`, `cone_threads`, and `kkt_form`; these
are configured capacities, not measurements of busy cores.

The sampled operator retains shared basis factors and weights as the numerical
input. Materialization is a checked fallback for structural preprocessing, not
the source of sampled coefficients. Exact equality reduction preserves factor
metadata; a chordal structural rewrite may use a materialized fallback and
change coordinates, so acceptance audits the original problem independently.

## Performance

Audited full solves at 768 bits on 2× AMD EPYC 7742 nodes (see
[CHANGELOG.md](CHANGELOG.md) and [docs/JOURNAL.md](docs/JOURNAL.md)):

| Problem | SDPX 0.8 | SDPB |
|---|---|---|
| Ising Λ19, 32 threads / 32 ranks | 102–106 s | 204 s |
| Ising Λ19, 64 threads / 64 ranks | 102 s | 153 s |
| Λ19 spins 0–50, one node (52 threads / 64 ranks) | 238 s | 328 s |

SDPX needs fewer iterations (119 vs 243 and 177 vs 265). Both solvers used
the same precision and tolerances, and each SDPX result passed an
independent original-coordinate audit.

On glibc Linux the CLI raises malloc's `mmap` and trim thresholds at start-up
(32 MiB / 256 MiB). Otherwise, repeated multi-megabyte work buffers are
mapped and unmapped on every call, and the page faults and TLB shootdowns
stop per-block kernels scaling across threads. Setting
`MALLOC_MMAP_THRESHOLD_` in the environment keeps glibc's own policy.
Programs that embed the library can apply the same `mallopt` settings.

## Verification and benchmarks

Each change is checked with one complete solve of a pinned case plus an
independent original-coordinate audit:

```sh
python3 benchmark/e2e/e2e.py build
python3 benchmark/e2e/e2e.py run medium      # Float64
python3 benchmark/e2e/e2e.py run ising11     # MPFR 512-bit sampled SDP
```

[AGENTS.md](AGENTS.md) says which check each kind of change needs. Known
failures are listed in [REVIEW_AND_PLAN.md](REVIEW_AND_PLAN.md). Before a
release, run the full suite:

```sh
cargo test --locked --release --workspace \
  --features sdpx-ffi/sdp-accelerate,sdpx-ffi/faer-sparse -- --test-threads=1
```

On Linux use `sdpx-ffi/sdp-openblas` instead of `sdp-accelerate`.

The [research protocol](benchmark/research/README.md) handles the fixed
multi-case suites and the MOSEK/Clarabel reference comparisons.
[benchmark/ising](benchmark/ising/README.md) handles SDPB comparisons and
scaling. Recorded MOSEK and SDPB runs use different inputs, settings, machines
and source revisions, so they are context only, not evidence of performance
parity.

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
