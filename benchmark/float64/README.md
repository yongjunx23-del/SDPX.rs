# Float64 conic benchmark adapter

`sdpx_runner.jl` is a standalone native adapter. It reads the portable conic
JSON input, launches the exact `sdpx` executable once per sample, and uses
Julia only for the independent original-coordinate oracle. It never imports
`SDPX.jl` or loads an SDPX shared library.

```sh
export SDPX_CLI=/absolute/frozen/target/release/sdpx
export SDPX_SOURCE_ID='commit=<sha>;dirty_patch_sha256=<sha>'
export SDPX_BENCH_THREADS=1
julia --startup-file=no benchmark/float64/sdpx_runner.jl input.json --runs=4 --tol=1e-6
```

The result separates `native_seconds` (solver timer), `api_seconds` (native
setup plus solve), `load_seconds` (JSON input), and `cli_e2e_seconds` (whole
fresh process). A warm row is a later fresh CLI process, never an in-process
warmed solve. Finiteness, affine/cone feasibility, gap, objective, and full
solver status gates remain outside native timing; failures stay in the sample
denominator.

The research controller invokes the same executable directly for each case and
records the native JSON receipt. Its canonical arm configuration is:

```json
{
  "source": "/absolute/frozen/candidate",
  "cli": "/absolute/frozen/candidate/target/release/sdpx",
  "env": {"RAYON_NUM_THREADS": "1"},
  "provider_files": []
}
```

`cli` is required and must be absolute. An old `library` entry may be retained
for identity history, but it is never used as a solver binding. Optional Julia
oracle settings belong under `oracle` and are not required to run native
Float64 benchmarks.

`benchmark/float64/run.py` is retained for Clarabel and MOSEK reference legs
only. Its former `--engine new` SDPX frontend leg is retired; native SDPX
comparisons use `benchmark/research/run.py pair` with the explicit `cli` arm
contract.
