# Single-orthant scaling diagnostic

This synthetic separable LP isolates scheduling over one large nonnegative cone.
It is not holdout, reference-parity, or general solver performance evidence.
For each variable j and k=1…r, row (j−1)r+k imposes x[j]≥k/r.
The objective is sum(x), so x*=ones(n) and the optimum is n. The deterministic
rational bounds are constructed directly in the selected arithmetic.

Use a reviewed frozen Julia package and matching shared library; do not change
source or dependencies during the sweep. The caller must supply source identity,
including commit and dirty patch hash. Library, driver, manifest, actual numerical
input and configuration hashes are captured separately. Run no other numerical
benchmark concurrently.

```sh
export SDPX_LIBRARY=/absolute/frozen/libsdpx.dylib
export SDPX_EXPECTED_SOURCE=/absolute/frozen/julia/SDPX.jl
export SDPX_SOURCE_ID='commit=<sha>;dirty_patch_sha256=<sha>'
export JULIA_DEPOT_PATH=/Users/xuyongjun/Desktop/project/SDPX/rebuild-env-depot:$HOME/.julia
export SDPX_OUTPUT_DIR=/tmp/orthant-frozen-53
export SDPX_BITS=53 SDPX_N=10000 SDPX_ROWS_PER_VAR=16
sh SDPX/benchmark/parallel/sweep.sh
```

The default environment is `/tmp/sdpx-rust-acceptance/float64-env-16a`; override
`SDPX_PROJECT` if needed. It must resolve SDPX to `SDPX_EXPECTED_SOURCE` and contain
JSON. `JULIA` may select a runtime. Set `SDPX_BITS=256` or `512` for BigFloat in a
new output directory. Dimensions are explicit and unchanged across widths.
For a single process call `orthant.jl` with `SDPX_THREADS` and `SDPX_OUTPUT` instead.
The sweep runs widths 1/2/4/8 sequentially in separate processes, Julia and BLAS
width 1. It refuses an existing output directory; the driver uses exclusive file
creation. Failures are retained and cause nonzero exit, with no fallback solve.

Each width performs one cold and three warm **fresh public solve_conic calls**.
Input construction and independent original-coordinate primal, dual, cone, gap,
and analytic objective validation are outside the timer. Validation uses the
unchanged native default tolerance: 1e-8 for Float64, sqrt(eps(BigFloat)) for MPFR.
BigFloat checks and serialized residual strings preserve precision. Settings use
defaults except presolve=false (retain all rows), augmented KKT and thread budget.
The API selects Float64 factorization automatically; every run must report QDLDL,
one factor thread, requested precision and an admissible actual cone width.
There is no unsupported backend override. Reduced-accuracy statuses fail.

Receipts include native solve time and frontend wall time, iterations, provider,
actual factor/cone widths, Julia allocation bytes and GC time. Public native setup
time is unavailable and explicitly null. Frontend time includes fresh model/handle
creation, native work, conversion and teardown; it is not a native setup measure.
Allocation receipts count Julia-managed allocations, not native allocations.
Warm medians are emitted only when all four runs pass. A single 3-warm sweep is a
bounded diagnostic; stable ≥2% speed credit requires comparable repeated evidence.

## Dominant sampled PSD diagnostic

`sampled.jl` predeclares one dense sampled PSD with Q=I+(1/4)J/n, weights −1,
q=ones and b=−svec(I). Default order is 64 for Float64, 24 for MPFR; `SDPX_N`
overrides it before the campaign. In exact arithmetic x=ones gives a PSD slack,
and Y=n/(n−1)(I−J/n) is a dual feasible point with objective n, proving optimality.
Non-dyadic entries (including n=24) are rounded once at the selected precision;
the unchanged accuracy gate tests the returned objective against n. Every Q entry
is nonzero, preventing a chordal split. This fixed synthetic fixture targets one
dominant sampled block, not a general SDP corpus or reference-parity claim.

Use the same explicit source/library/identity, bits, thread and output variables
as `orthant.jl`. JSON and SDPX are required; **MPFR also requires the existing
GenericLinearAlgebra package as a direct dependency in the benchmark environment**.
It is used only for independent PSD eigenaudits. No benchmark adds dependencies.
The existing Float64 environment suffices for the Float64 audit once it resolves
to the declared source. Example, with identities/depot already set:

```sh
export SDPX_BITS=512 SDPX_PROJECT=/absolute/frozen-benchmark-env
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
mkdir /tmp/sampled-candidate-512
failed=0
for width in 1 2 4 8; do
  SDPX_THREADS=$width RAYON_NUM_THREADS=$width \
  SDPX_OUTPUT=/tmp/sampled-candidate-512/width-$width.json \
  julia --startup-file=no -t1 --gcthreads=1 --project="$SDPX_PROJECT" \
    SDPX/benchmark/parallel/sampled.jl \
    > /tmp/sampled-candidate-512/width-$width.log 2>&1 || failed=1
done
test "$failed" = 0
```

One cold plus three warm fresh factor adaptations and public solves are timed.
All default numerical settings remain, with condensed KKT and the requested width
explicit. Independent materialization from Q, primal/dual affine checks, PSD
checks on primal/returned slack/dual, gap and known-objective checks stay outside
timing at 1e-8 or sqrt(eps(BigFloat)). Passing requires Optimal and the actual
`condensed_sampled_qdldl` receipt, requested precision and cone pool width, and
one QDLDL factor thread. This does not infer internal kernel widths from the factor
thread receipt. Failed runs retain records and receive no median timing credit.
Native setup time is unavailable (null). RSS is `Sys.maxrss()` in bytes: cumulative
process high-water memory including input, compilation and earlier audits, not a
per-solve allocation measurement. The existing `float64/resource_probe.py` can
wrap the Julia command for an additional whole-child RSS receipt. BigFloat audit
values are serialized as full decimal strings; Julia allocations exclude native
allocations. Compare equal inputs/settings across widths and source candidates.
