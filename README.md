<h1 align="center">SDPX</h1>

<p align="center">
  <b>Arbitrary-precision interior-point conic optimization in Rust,<br>built for the  bootstrap.</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/version-0.8.0-e8590c" alt="version 0.8.0">
  <img src="https://img.shields.io/badge/rust-2021-1c7ed6" alt="Rust 2021">
  <img src="https://img.shields.io/badge/precision-53%20to%202048%20bits-495057" alt="precision 53 to 2048 bits">
  <img src="https://img.shields.io/badge/license-Apache--2.0-2f9e44" alt="Apache-2.0">
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#quick-start">Quick start</a> •
  <a href="#bootstrap-example">Bootstrap example</a> •
  <a href="#license">License</a>
</p>

SDPX is a primal-dual interior-point solver for convex conic programs,
written in Rust. It solves

$$
\begin{array}{rl}
\min & \tfrac12 x^\top P x + q^\top x \\\\
\text{s.t.} & Ax + s = b,\quad s \in \mathcal K
\end{array}
$$

in binary64 or in fixed precision from 128 to 2048 bits (GMP/MPFR). Here
$\mathcal K$ is a product of zero, nonnegative, second-order, exponential,
power and positive-semidefinite cones. It reads SDPB's `pmp2sdp` output
directly, so a polynomial-matrix bootstrap problem goes from `pmp2sdp` to an
audited 768-bit solution with no Julia or Python runtime in the solver.

## Features

- **Bootstrap-native input.** Sampled PSD blocks from `pmp2sdp` stay in
  factored form (bilinear bases × sample weights). Expanded sparse coefficients
  are released after preprocessing and KKT setup; iterations apply the factors
  directly.
- **Precision you choose.** Binary64, or MPFR from 128 to 2048 bits in
  64-bit increments in the CLI and C ABI, including 1216. The Rust API
  accepts `MpFloat<N>` for `64*N` bits.
  Dense high-precision products use an exact residue-number-system kernel:
  they accumulate exactly and round once.
- **Fast at scale.** Threads work per block (SDPB-style load balancing), and
  a parallel arrow LDLᵀ factors the KKT system. An optional
  owner-partitioned MPI path spans several nodes.
- **Robust.** A homogeneous self-dual embedding detects infeasibility,
  Nesterov–Todd scaling handles every symmetric cone, and iterative
  refinement runs against the original operator. Step lengths use
  certified binary64 screens.
- **Every cone.** Zero, nonnegative, second-order, exponential, power,
  generalized power and PSD cones, with chordal decomposition and presolve.
- **Pure Rust surface.** A library (`sdpx-solver`), a CLI (`sdpx`), and
  exact decimal I/O that never rounds through binary64.

## Quick start

**Set up** (Rust ≥ 1.85 and a C compiler for GMP/MPFR; on Linux use
`sdp-openblas` instead of `sdp-accelerate`):

```sh
cargo install --git https://github.com/yongjunx23-del/SDPX.rs sdpx-solver --bin sdpx --features sdp-accelerate,faer-sparse
```

**Use:** `sdpx problem.json --output solution.json`, or
`sdpx my-bootstrap-sdp/ --precision 768 --threads 32` on a `pmp2sdp` directory.

As a library, add `sdpx-solver` (and `sdpx-arithmetic` for MPFR types) from
this repository. A small SDP in binary64:

```rust
use sdpx_solver::algebra::*;
use sdpx_solver::solver::*;

fn main() {
    // minimise tr(X) subject to <A, X> = 1 over 2×2 PSD matrices X,
    // with X in svec form (x11, √2·x12, x22).
    let p = CscMatrix::zeros((3, 3));
    let q = vec![1.0, 0.0, 1.0];
    let s2 = 2f64.sqrt();
    let a = CscMatrix::from(&[
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
        [1.0, s2, 2.0], // <A, X> with A = [1 1; 1 2]
    ]);
    let b = vec![0.0, 0.0, 0.0, 1.0];
    let cones = [PSDTriangleConeT(2), ZeroConeT(1)];

    let mut solver = DefaultSolver::new(&p, &q, &a, &b, &cones, DefaultSettings::default()).unwrap();
    solver.solve();
    println!("{:?}  objective = {}", solver.solution.status, solver.solution.obj_val);
}
```

## PMP conversion

Convert SDPB JSON or XML directly in Rust:

```sh
cargo build --locked --release -p sdpx-pmp
./target/release/sdpx-pmp2sdp --input problem.json --output problem-sdp --precision 768
sdpx problem-sdp --precision 768
```

The [converter guide](crates/pmp/README.md) covers normalization, prefactors,
sampling, supported formats and the Rust API.

## Bootstrap example

Solve an SDPB-format bootstrap problem at 768 bits with 32 threads, to a
10⁻⁴² gap. See [the Rust example](crates/solver/examples/rust/example_bootstrap.rs); run it with
`cargo run --release --example bootstrap --features sdp-openblas -- ising-lambda19/`.

```rust
use sdpx_arithmetic::Bits768;
use sdpx_solver::solver::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Directory written by `pmp2sdp --outputFormat=json`.
    let dir = std::env::args().nth(1).expect("usage: bootstrap <pmp2sdp-json-dir>");
    let sdp = read_sdpb_sampled::<Bits768>(dir)?;
    let mut problem = sdp.problem;

    let tol: Bits768 = "1e-42".parse().unwrap();
    problem.settings.tol_gap_abs = tol;
    problem.settings.tol_gap_rel = tol;
    problem.settings.tol_feas = tol;
    problem.settings.max_threads = 32;
    problem.settings.verbose = true;

    let mut solver = problem.into_solver()?;
    solver.solve();

    // Add the objective constant to recover SDPB's objective.
    let objective = solver.solution.obj_val + sdp.objective_constant;
    println!("{:?} after {} iterations", solver.solution.status, solver.solution.iterations);
    println!("objective = {}", objective.to_decimal(Some(50)));
    Ok(())
}
```

The first `sdp.num_equalities` entries of `-solution.z` are SDPB's `y`. The
CLI writes the same point with status, precision and thread use. Set
`SDPX_RECEIPT=receipt.json` to record detailed phase timings.

`max_threads = 0` uses available CPUs, capped by a positive
`RAYON_NUM_THREADS` value. Explicit thread settings take precedence.

For sampled solvers, `solver.data.A` retains only the explicit linear
component after setup. Use `solver.data.materialize_A()` when you need the
full equilibrated matrix; this allocates its expanded coefficients.

## Crates

| Crate | Contents |
|---|---|
| `sdpx-solver` | Solver, cones, KKT systems, sampled bootstrap operator, `sdpx` CLI |
| `sdpx-arithmetic` | Fixed-precision MPFR scalars (`Bits128` … `Bits2048`) and exact dot products |
| `sdpx-pmp` | PMP conversion library and `sdpx-pmp2sdp` CLI (JSON/XML → sampled SDP) |
| `sdpx-ffi` | Stable C ABI over the solver, for non-Rust hosts |

Run the test suite with
`cargo test --locked --release --workspace --features sdpx-ffi/sdp-openblas,sdpx-ffi/faer-sparse -- --test-threads=1`.

## Citing

If SDPX helps your bootstrap study, please cite this repository and the
works it builds on:

- [Clarabel](https://link.springer.com/article/10.1007/s12532-026-00320-7)
  (interior-point algorithm and Rust core);
- [SDPB](https://arxiv.org/abs/1909.09745) (sampled bootstrap SDP form);
- [Hypatia](https://arxiv.org/abs/2107.04262) (conic methods).

## License

Apache-2.0. The Rust core is adapted from
[Clarabel.rs](https://github.com/oxfordcontrol/Clarabel.rs) and keeps its
attribution. The provenance of adapted code is in
[`provenance/`](provenance/). GMP, MPFR, BLAS and LAPACK keep their own
licenses.

The PMP converter is adapted from SDPB 3.1.0 under the
[MIT license](provenance/SDPB-LICENSE).
