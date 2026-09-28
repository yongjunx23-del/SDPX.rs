<h1 align="center">SDPX</h1>

<p align="center">
  <b>High-Precision Conic Solver in Pure Rust for the Numerical Bootstrap</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/version-0.8.0-e8590c" alt="version 0.8.0">
  <img src="https://img.shields.io/badge/rust-2021%20(%E2%89%A51.85)-1c7ed6" alt="Rust 2021">
  <img src="https://img.shields.io/badge/precision-53%20to%202048%20bits-495057" alt="precision 53 to 2048 bits">
  <img src="https://img.shields.io/badge/license-Apache--2.0-2f9e44" alt="Apache-2.0">
</p>

SDPX is an arbitrary-precision primal-dual interior-point solver for convex conic programs, engineered in pure Rust:

$$\begin{aligned}
\min_{x} \quad & \tfrac{1}{2} x^\top P x + q^\top x \\
\text{s.t.} \quad & A x + s = b, \quad s \in \mathcal{K}
\end{aligned}$$

where $\mathcal{K}$ is any Cartesian product of **Zero**, **Nonnegative**, **Second-Order (SOC)**, **Exponential**, **Power**, **Generalized Power**, and **Positive Semidefinite (PSD)** cones.

---

## Key Features

- **Built for the Numerical Bootstrap:** Designed for general numerical bootstrap problems (conformal bootstrap, S-matrix bootstrap, matrix models, and polynomial optimization). Ingests Polynomial Matrix Programs (PMP) directly via built-in `sdpx-pmp2sdp`. Sampled PSD blocks stay in memory-efficient factored form (bilinear bases $\times$ sample weights), avoiding expanding massive coefficient matrices into RAM.
- **Precision You Choose:** The solver CLI defaults to Float64 and MPFR 128, 256, 512, 768 and 1024 bits. Build with `all-precisions` to include every 64-bit step from 128 through 2048. Eligible dense high-precision products use an exact Residue Number System (RNS) kernel that accumulates exactly without intermediate rounding.
- **Fast & Scalable:** Multi-threaded block-level parallelism (SDPB-style load balancing via Rayon), parallel Arrow $\text{LDL}^\top$, Faer sparse solver, certified binary64 step-length screening, and optional multi-node MPI partitioning.
- **Zero Runtime Dependencies:** Standalone CLI tools and pure Rust libraries. No Julia, Python, or Mathematica required at solve time. Includes a stable C ABI (`include/sdpx.h`) for foreign language integration.

---

## Installation & Build

Requires **Rust $\ge$ 1.85** and system **GMP/MPFR** libraries (`brew install gmp mpfr` on macOS, `apt install libgmp-dev libmpfr-dev` on Linux).

```sh
# macOS (Apple Accelerate + Faer)
cargo build --release -p sdpx-solver --bin sdpx --features sdp-accelerate,faer-sparse

# Linux (OpenBLAS + Faer)
cargo build --release -p sdpx-solver --bin sdpx --features sdp-openblas,faer-sparse
```

These commands build `target/release/sdpx` with six precision choices. Add
`all-precisions` to the feature list for the complete precision range.
Build the converter separately with `cargo build --release -p sdpx-pmp`.
The native Rust API, converter and C ABI retain their complete precision support.

---

## Ordinary SDP Usage

SDPX solves standard SDPs using symmetric vector format (`svec`, upper-triangular order with off-diagonals scaled by $\sqrt{2}$).

### Rust API Example

```rust
use sdpx_solver::algebra::*;
use sdpx_solver::solver::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 3x3 PSD variable in svec format: dimension = 3 * 4 / 2 = 6
    // svec: [X11, sqrt(2)*X12, X22, sqrt(2)*X13, sqrt(2)*X23, X33]
    let n = 3;
    let nvec = (n * (n + 1)) / 2;

    // Minimize tr(X) = X11 + X22 + X33
    let p = CscMatrix::zeros((nvec, nvec));
    let q = vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0];

    // Constraint: <A_0, X> = 1
    let s2 = 2.0_f64.sqrt();
    let a = CscMatrix::from(&[
        [-1.0,  0.0,  0.0,  0.0,  0.0,  0.0],
        [ 0.0, -s2,   0.0,  0.0,  0.0,  0.0],
        [ 0.0,  0.0, -1.0,  0.0,  0.0,  0.0],
        [ 0.0,  0.0,  0.0, -s2,   0.0,  0.0],
        [ 0.0,  0.0,  0.0,  0.0, -s2,   0.0],
        [ 0.0,  0.0,  0.0,  0.0,  0.0, -1.0],
        [ 1.0,  4.0,  3.0,  8.0, 10.0,  6.0], // <A_0, X> = 1
    ]);
    let b = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];

    // Cones: 3x3 PSD cone (slack rows 0..5) and 1 equality constraint (row 6)
    let cones = vec![PSDTriangleConeT(n), ZeroConeT(1)];

    let mut solver = DefaultSolver::new(&p, &q, &a, &b, &cones, DefaultSettings::default())?;
    solver.solve();

    println!("Status: {:?}", solver.solution.status);
    println!("Objective: {:.8}", solver.solution.obj_val);
    Ok(())
}
```

### CLI Solve

```sh
# Solve a conic JSON problem in standard Float64:
sdpx problem.json --output solution.json

# Solve at 256-bit precision with 8 threads:
sdpx problem.json --precision 256 --threads 8 --output solution.json
```

---

## LPs with local bounds

Encode `rho >= 0` as `-rho + s = 0`, and an optional upper bound
`rho <= c` as `rho + s = c`, using nonnegative slack cones. With automatic
KKT selection, eligible problems eliminate these local directions and factor
only the equality/free-variable border. Eligibility requires at least 64
bounded variables, one or two bound rows per variable, a border of at most
128 coordinates, and diagonal `P`.

Float64 builds with `faer-sparse` use packed faer matrix products and batched
RHS kernels (`local_bounds_faer`). MPFR uses exact accumulation
(`local_bounds_arrow`). Both retain full-system regularization and refinement;
failed factorizations fall back to QDLDL. Explicit `direct_solve_method="qdldl"`
selects the original backend.

---

## PMP to SDP Workflow (Bootstrap)

Polynomial Matrix Programs (PMP) constrain polynomial matrices to be positive semidefinite for all $x \ge 0$:

$$\text{Maximize } y \quad \text{s.t.} \quad M_0(x) + \sum_{i} y_i M_i(x) \succeq 0 \quad \forall x \ge 0$$

### 1. PMP Problem JSON (`problem.json`)

```json
{
  "objective": ["0", "1"],
  "PositiveMatrixWithPrefactorArray": [
    {
      "prefactor": { "constant": "1", "base": "0.5", "poles": [] },
      "polynomials": [[[ ["1", "0", "1"], ["-1"] ]]]
    }
  ]
}
```
*Layout: `polynomials[row][col][coordinate][degree_coeff]`, encoding $1 - y + x^2 \ge 0$.*

### 2. End-to-End CLI Pipeline

```sh
# Step 1: Convert PMP (JSON or SDPB XML) to a sampled SDP directory at 768 bits
sdpx-pmp2sdp --input problem.json --output problem-sdp --precision 768 --threads 8

# Step 2: Solve the sampled SDP with 32 threads
sdpx problem-sdp --precision 768 --threads 32 --output solution.json
```

### 3. Rust API Workflow

```rust
use sdpx_arithmetic::Bits768;
use sdpx_pmp::PolynomialMatrixProgram;
use sdpx_solver::solver::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Convert PMP into a sampled SDP directory
    let pmp = PolynomialMatrixProgram::read("problem.json")?;
    pmp.write_sdp_with_threads::<Bits768>("problem-sdp", 8)?;

    // 2. Load sampled SDP and configure high-precision tolerances
    let sdp = read_sdpb_sampled::<Bits768>("problem-sdp")?;
    let mut problem = sdp.problem;
    let tol: Bits768 = "1e-45".parse()?;
    problem.settings.tol_gap_abs = tol;
    problem.settings.tol_feas = tol;
    problem.settings.max_threads = 16;

    // 3. Solve
    let mut solver = problem.into_solver()?;
    solver.solve();

    // 4. Recover objective & SDPB dual multipliers (y = -z)
    let objective = solver.solution.obj_val + sdp.objective_constant;
    println!("Status: {:?}", solver.solution.status);
    println!("Objective: {}", objective.to_decimal(Some(40)));

    let y: Vec<Bits768> = solver.solution.z[..sdp.num_equalities]
        .iter()
        .map(|&v| -v)
        .collect();
    println!("Dual multiplier y: {:?}", y);

    Ok(())
}
```

---

## CLI Reference

### `sdpx`
```text
sdpx INPUT [--precision BITS] [--threads N] [--output FILE] [--settings FILE]
```
- `INPUT`: Conic JSON file, SDPB sampled directory, or `-` for stdin.
- `--precision BITS`: `53` (Float64, default), or multiples of 64 from `128` through `2048` (MPFR).
- `--threads N`: Parallel cone workers and factorization threads (default: CPU cores).
- `--output FILE`: Output solution JSON path (default: stdout).
- `SDPX_RECEIPT=receipt.json`: (Env var) Records peak RSS and per-phase microsecond timings.

### `sdpx-pmp2sdp`
```text
sdpx-pmp2sdp --input PMP.json|PMP.xml --output NEW_DIR [--precision BITS] [--threads N]
```
- `--input FILE`: Input PMP file in SDPB JSON or XML format.
- `--output DIR`: Uncompressed target SDP directory (refuses to overwrite existing paths).
- `--precision BITS`: Multiples of 64 from `128` through `2048` (default: `768`).
- `--threads N`: Number of independent polynomial blocks to convert concurrently.

---

## Workspace Structure

| Crate | Purpose |
|---|---|
| [`sdpx-solver`](crates/solver/) | Interior-point conic solver core, cones, KKT systems, and `sdpx` CLI |
| [`sdpx-arithmetic`](crates/arithmetic/) | MPFR scalar types (`Bits128`–`Bits2048`), decimal I/O, exact RNS dot products |
| [`sdpx-pmp`](crates/pmp/) | PMP-to-SDP transformation library and `sdpx-pmp2sdp` CLI |
| [`sdpx-ffi`](crates/ffi/) | C ABI (`include/sdpx.h`) for integration with C/C++, Julia, and Python |

---

## Citation

If SDPX helps your research, please cite:

```bibtex
@software{sdpx_rs,
  author = {Yongjun Xu},
  title  = {{SDPX.rs}: High-Precision Conic Solver in Pure Rust for the Numerical Bootstrap},
  url    = {https://github.com/yongjunx23-del/SDPX.rs},
  year   = {2026}
}
```

### Acknowledgements

- **[Clarabel.rs](https://github.com/oxfordcontrol/Clarabel.rs):** The interior-point conic solver core and homogeneous self-dual embedding are adapted from Clarabel.rs.
- **[SDPB](https://github.com/davidsd/sdpb):** The sampled bootstrap SDP formulation and PMP conversion algorithms are adapted from SDPB.

---

## License

Apache-2.0. Upstream code provenance and licenses (Clarabel.rs, SDPB) are documented in [`provenance/`](provenance/).
