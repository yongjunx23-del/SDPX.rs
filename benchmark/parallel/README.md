# Native synthetic scaling diagnostics

The orthant fixture is a deterministic separable LP: row `(j−1)r+k` imposes
`x[j] ≥ k/r`, so `x*=ones(n)` and the optimum is `n`. `sampled.jl` uses the
fixed factor-authoritative PSD fixture `Q=I+(1/4)J/n`, weights `−1`,
`q=ones`, and `b=−svec(I)`. These are scheduling diagnostics, not holdout or
reference-parity evidence.

The Rust executable is the only solver. Julia is used only to construct the
wire input and perform the independent audit; neither script imports `SDPX.jl`.
Freeze the executable and source identity before a run:

```sh
export SDPX_CLI=/absolute/frozen/target/release/sdpx
export SDPX_SOURCE_ID='commit=<sha>;dirty_patch_sha256=<sha>'
export SDPX_OUTPUT_DIR=/tmp/orthant-frozen-53
export SDPX_BITS=53 SDPX_N=10000 SDPX_ROWS_PER_VAR=16
sh SDPX/benchmark/parallel/sweep.sh
```

`SDPX_CLI` must be an executable absolute path. `JULIA` may select the runtime
used for the generator/audit, and `SDPX_BITS=256` or `512` selects MPFR in a new
output directory. Widths 1/2/4/8 run sequentially; each recorded point launches
one fresh native CLI process. An existing output directory is refused and all
failure records remain visible.

Each width emits one receipt with one cold and three warm *fresh CLI* samples.
`api_seconds` is native setup plus solve, `native_seconds` is the solver timer,
`load_seconds` is input loading, and `cli_e2e_seconds` includes process startup
and result serialization. Input construction and the original-coordinate audit
are outside all native scopes. The Float64 gate remains 1e-8 internally and
1e-6 for research; MPFR uses `sqrt(eps)` and requires `GenericLinearAlgebra` for
the independent PSD eigensolver. A failed sample remains in the denominator and
cannot earn timing credit.

For the sampled diagnostic, use the same variables and invoke `sampled.jl`
directly, for example:

```sh
export SDPX_BITS=512 SDPX_CLI=/absolute/frozen/target/release/sdpx
export SDPX_SOURCE_ID='commit=<sha>;dirty_patch_sha256=<sha>'
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
mkdir /tmp/sampled-candidate-512
for width in 1 2 4 8; do
  SDPX_THREADS=$width SDPX_OUTPUT=/tmp/sampled-candidate-512/width-$width.json \
    julia --startup-file=no -t1 --gcthreads=1 \
      SDPX/benchmark/parallel/sampled.jl \
      > /tmp/sampled-candidate-512/width-$width.log 2>&1 || true
done
```

The sampled solver receives the original `sampled` factors and never the dense
audit matrix. Receipts retain status, objective/residual gates, precision,
direction, cone/factor thread receipts and all native timing fields. These
synthetic results do not establish Ising or general SDP performance.
