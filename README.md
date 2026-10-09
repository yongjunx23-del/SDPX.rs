# SDPX

SDPX is a Rust conic solver derived from Clarabel.rs, with one homogeneous
self-dual interior-point engine for Float64 and fixed-precision MPFR arithmetic.
It solves

$$
\min_x \; \tfrac12 x^\top P x + q^\top x
\quad\text{subject to}\quad Ax+s=b,\;s\in\mathcal K.
$$

Supported cones are zero, nonnegative, second-order, exponential, power,
generalized power and PSD triangle. Sampled bootstrap SDPs retain their PSD
coefficients as bases and sample weights through presolve; these factors
remain the operator used by the solver.

Use the native Rust API, the `sdpx` CLI or the C ABI in
[`include/sdpx.h`](include/sdpx.h). The separate `sdpx-pmp2sdp` converter
turns polynomial matrix programs into sampled SDP inputs. Julia scripts in
this repository generate inputs or audit results; they do not drive the solver.

## Build

Requires Rust 1.85 or newer, a native build toolchain for the GMP/MPFR dependency,
and BLAS/LAPACK. Choose one provider; BLAS/LAPACK are linked for every solver
build. After dependencies are cached, build the required crate:

```sh
# macOS: Accelerate; use sdp-openblas instead on Linux.
F=sdp-accelerate,faer-sparse
cargo build --locked --offline --profile fast -p sdpx-solver --bin sdpx --features "$F"
cargo build --locked --offline --profile fast -p sdpx-pmp

# Optional C ABI library.
cargo build --locked --offline --profile fast -p sdpx-ffi --features "$F"
```

The executables are in `target/fast/`. Use `release` for measured performance.
The BLAS provider must support concurrent calls from solver workers;
`.cargo/config.toml` enables locking for source-built OpenBLAS.
[Architecture](docs/ARCHITECTURE.md#build-features) describes provider features
and direct linking.

The solver CLI and C ABI support 53-bit Float64 and MPFR at 128, 256, 512,
768 and 1024 bits. The converter supports those MPFR widths and defaults to
768 bits. Add `all-precisions` to the relevant crate's feature list to expose
every MPFR width from 128 through 2048 in steps of 64. Native Rust
`MpFloat<N>` types do not depend on that dispatch feature. A solve keeps its
requested precision throughout factorization and refinement.

## CLI usage

```sh
# Conic JSON: Float64 by default, or explicitly selected MPFR precision.
target/fast/sdpx problem.json --output solution.json
target/fast/sdpx problem.json --precision 256 --threads 8 --output solution.json

# Convert SDPB JSON/XML PMP input, then solve the sampled SDP directory.
target/fast/sdpx-pmp2sdp --input problem.xml --output problem-sdp --precision 768 --threads 8
target/fast/sdpx problem-sdp --precision 768 --threads 8 --output solution.json
```

`sdpx INPUT` accepts conic JSON, a sampled SDP directory, or `-` for JSON
stdin. `--settings FILE` replaces input settings; unspecified fields use
core defaults. Progress goes to stderr and the result goes to stdout unless
`--output` is supplied. The converter refuses existing output paths.

`--checkpoint FILE [--checkpoint-every N]` saves the accepted iterate
(default interval: 10 iterations). `--restart FILE` continues the same input
or starts a nearby input with matching structure and precision. Checkpoints
require the ordinary solver, without `--partitions`; loading rejects
nonfinite values and nonpositive homogenization or scaling factors.
`SDPX_RECEIPT=FILE` records phase timings and peak RSS. Run each executable
with `--help` for its full argument list, including MPI partitioning.

## Rust API

```rust
use sdpx_solver::algebra::CscMatrix;
use sdpx_solver::solver::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Minimize x subject to 0 <= x <= 1.
    let p = CscMatrix::<f64>::zeros((1, 1));
    let q = vec![1.0];
    let a = CscMatrix::from(&[[-1.0], [1.0]]);
    let b = vec![0.0, 1.0];
    let cones = [NonnegativeConeT(2)];
    let mut solver = DefaultSolver::new(
        &p, &q, &a, &b, &cones, DefaultSettings::default(),
    )?;
    solver.solve();
    println!("{:?}: {:?}", solver.solution.status, solver.solution.x);
    Ok(())
}
```

Use `sdpx_arithmetic::Bits256` or another `MpFloat<N>` type in place of
`f64` for MPFR. PSD cones use upper-triangular column order with
off-diagonals scaled by $\sqrt2$ (`svec`). For sampled SDP loading and
objective recovery, see the
[bootstrap example](crates/solver/examples/rust/example_bootstrap.rs).
Other [Rust examples](crates/solver/examples/rust/) cover ordinary SDP,
SOC, updates and callbacks.

## Documentation and development

- [Architecture](docs/ARCHITECTURE.md): modules, arithmetic, KKT selection and build features.
- [Development plan](REVIEW_AND_PLAN.md): priorities, measured results and known failures.
- [Journal](docs/JOURNAL.md): experiment history and evidence.
- [Working rules](AGENTS.md): numerical contracts and verification workflow.
- [Changelog](CHANGELOG.md): release notes.

For a solver edit, build the affected crate and run one matching end-to-end
case with its original-coordinate audit:

```sh
python3 benchmark/e2e/e2e.py build --arm NAME
python3 benchmark/e2e/e2e.py run ising11 --arm NAME
```

Pinned cases are `medium` (Float64), `ising11` (MPFR/SDP) and `csdr3` (SOC).
Documentation edits need only `git diff --check`. Detailed toolchain and
input instructions are in the
[development skill](.agents/skills/sdpx-development/SKILL.md).

## Citation and attribution

```bibtex
@software{sdpx_rs,
  author = {Yongjun Xu},
  title  = {{SDPX.rs}: High-Precision Conic Solver in Pure Rust for the Numerical Bootstrap},
  url    = {https://github.com/yongjunx23-del/SDPX.rs},
  year   = {2026}
}
```

The solver core and HSD embedding are adapted from
[Clarabel.rs](https://github.com/oxfordcontrol/Clarabel.rs); sampled SDP and
PMP conversion algorithms are adapted from [SDPB](https://github.com/davidsd/sdpb).
Solver, arithmetic and FFI crates use Apache-2.0; the PMP crate uses MIT.
Upstream attribution and licenses are retained in [provenance/](provenance/).
