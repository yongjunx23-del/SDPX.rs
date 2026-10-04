# Rust PMP converter

`sdpx-pmp2sdp` converts polynomial matrix programs to SDPB sampled JSON.
It is a standalone Rust library and CLI; it needs GMP/MPFR, but no solver,
MPI, BLAS, Julia or Python runtime.

```sh
cargo build --locked --release -p sdpx-pmp
./target/release/sdpx-pmp2sdp --input problem.json --output problem-sdp --precision 768
./target/release/sdpx problem-sdp --precision 768
```

Use `--threads 8` to convert up to eight independent blocks concurrently.
The default is one worker. The CLI reads blocks as workers become available.
Each worker holds one input block and writes output row by row. Choose fewer
workers when memory is limited. Output is identical across thread counts.
Single-block inputs use one worker.

Input: SDPB JSON or legacy XML. Output: a new, uncompressed SDP directory.
Existing paths are refused. The CLI supports MPFR at 128, 256, 512, 768 or 1024 bits by default —
`all-precisions` builds accept every 64-bit increment from 128 through 2048.
The library accepts `MpFloat<N>` with at least 128 bits.
Use the same precision when solving the converted problem.

## Input

JSON numbers are decimal strings. Polynomial coefficients are ordered from
constant to highest degree. The matrix layout is
`polynomials[row][column][objective_coordinate][degree]`.
For example, maximize `y` subject to `1 - y + x² ≥ 0` for all `x ≥ 0`:

```json
{
  "objective": ["0", "1"],
  "PositiveMatrixWithPrefactorArray": [{
    "polynomials": [[[["1", "0", "1"], ["-1"]]]]
  }]
}
```

Without normalization the first coordinate is fixed to one. With
`normalization`, its dot product with the coordinates is fixed to one; the
largest absolute normalization entry is eliminated, as in SDPB. Remaining
coordinates retain their original order. Recover the eliminated coordinate
`w[k] = (1 - sum(n[i] * w[i], i != k)) / n[k]` from `normalization.json`.

Optional block fields:

- `prefactor`: `{ "constant": "1", "base": "0.5", "poles": ["-1"] }`.
  Constants must be positive, bases positive, and poles nonpositive.
- `reducedPrefactor`, `samplePoints`, `sampleScalings`,
  `reducedSampleScalings`.
- `bilinearBasis` for both parities, or `bilinearBasis_0` and
  `bilinearBasis_1` separately. Each contains polynomial coefficient lists;
  polynomial `j` has degree `j`, with optional trailing zeros.

Missing sampling data is generated from the prefactor's integrated density;
missing bases are generated from the sampled moment matrix. Automatic
sampling of nonconstant constraints requires a base below one. The default
prefactor is `exp(-x)` (one for constant constraints). XML uses SDPB's
`<sdp>` / `<polynomialVectorMatrices>` schema.

## Library

```rust
use sdpx_arithmetic::Bits768;
use sdpx_pmp::PolynomialMatrixProgram;

fn main() -> sdpx_pmp::Result<()> {
    let pmp = PolynomialMatrixProgram::read("problem.json")?;
    pmp.write_sdp::<Bits768>("problem-sdp")
}
```

The library also provides `write_sdp_with_threads::<Bits768>(path, 8)`.
For bounded input memory, use `sdpx_pmp::convert_file::<Bits768>(input, output, 8)`.
It scans the header, then converts blocks without retaining the whole program.
JSON and XML field order is unrestricted. Conversion keeps
MPFR coefficients for one matrix entry per worker and never expands PSD
factors into a coefficient matrix. It rejects malformed dimensions, asymmetric
matrices, non-finite values, decimal underflow and singular moment matrices.

The algorithms and formats are adapted from SDPB 3.1.0's MIT-licensed
`pmp2sdp`; see [provenance](../../provenance/pmp2sdp.json).
Sampling uses safeguarded Newton iteration, then bisection to neighboring
representable values at full working precision; upstream uses Newton iteration
with a half-precision stopping target. Generated coefficients can therefore
differ in their last digits. Supplied sample data
is preserved at the requested precision.

Mathematica `.m`, `.nsv` lists, compressed output, binary output and MPI
conversion are not supported. Conversion does not evaluate Mathematica code.
