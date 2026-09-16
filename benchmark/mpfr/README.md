# MPFR dense provider diagnostic

`bench.py prepare` copies an explicitly named frozen source and adds one private,
ignored test module. It never changes the supplied source or its Cargo.lock.
The actual production `XgemmScalar` / `XsyrkScalar` traits are crate-private, so
this avoids adding public production APIs. No dependencies are added.

```sh
python3 benchmark/mpfr/bench.py prepare --source /absolute/frozen-baseline --output /tmp/mpfr-baseline
# Run only when the coordinated numerical timing slot is free:
python3 benchmark/mpfr/bench.py run --output /tmp/mpfr-baseline --cargo /absolute/cargo --rustc /absolute/rustc --features sdp-accelerate
```

On Linux select the same existing provider feature/build environment used for
qualification (for example `sdp-openblas`). Dependencies must already be cached:
the build always uses `--locked --offline`. Existing outputs are never reused;
failed runs retain their logs and require a fresh output directory. Source trees
with symlinks are rejected. All target artifacts/logs stay beneath the external
output directory. Use explicit real Cargo/rustc executables from one toolchain.

Repeat preparation/run with the candidate frozen source and the **same** harness,
compiler, provider, environment and physical-core allocation. The receipt hashes
all source files before/after insertion, Cargo.lock, generator, harness, compiler,
Cargo executable, test executable, raw log and all decimal files. Cargo JSON build
logs retain actual dependency/compiler artifacts. Record host/affinity and native
provider versions alongside these receipts when reporting measurements.

The 90 measurements cover 128/256/512/768/1024/2048 bits, square PSD-shaped orders
12/16/64, GEMM NN/NT with alpha=1 beta=0, GEMM TN with alpha=3/7 beta=2/11,
and upper/no-transpose plus lower/transpose SYRK. Inputs are deterministic
non-dyadic rational numbers constructed directly at the working precision.
Setup, first call and nine warmed calls (configurable odd count >=7) are separate.
Each measured sample batches a fixed `ceil(64^3/n^3)` calls, after at least
150 ms warming that particular kernel. `warm_ns` and `median_ns` are per-call
averages; `warm_batch_ns` retains the summed call timings. Resetting the output
is excluded from each call's timer. This reduces short-sample scheduling noise;
reverse the baseline/candidate execution order to check remaining order effects.
Input reset, allocations, equality checks and decimal output are outside kernel
timing. Setup is shared across kernels for each precision/size, so do not sum its
repeated receipt value. Each warmed output must equal its first-call output.

For assignment-only changes, compare the complete `input_files` and `output_files`
maps in both `run/results.json` files; require exact equality, including hashes.
Round-trippable decimal files are retained for investigating any difference.
Untouched SYRK triangles contain the deterministic seed and are included in the
check. This comparison checks preservation, not an independent mathematical oracle.
Compare warmed medians only after output equality; setup and first calls remain
separate. Provider microbenchmark gains are **not** end-to-end solver gains.

Preparation has static validation only until the operator compiles and runs
the harness. No numerical qualification is implied by successfully generating it.
