# SDPX Julia frontend

This package builds models and calls the new SDPX Rust library. It has its own UUID and does not load the stable sibling solver. Float64 and explicit BigFloat precisions 128, 256, 512, 768, 1024, and 2048 use the same bulk C ABI. Numerical qualification belongs to the workspace acceptance results; unsupported native modes fail explicitly.

Build with `Pkg.build("SDPX")` after installing Rust and resolving the workspace lockfile. `CARGO` can select a Cargo executable. Set `SDPX_LIBRARY` to select an existing shared library, otherwise the frontend checks the new workspace's `target/release` and `target/debug` (or `CARGO_TARGET_DIR`).

The frontend requires ABI 3 and explicitly rejects older libraries. ABI 3 retains
the 80-byte settings and 104-byte info layouts and uses preprocessing flags for
Ruiz, presolve and chordal decomposition. Rebuild the library together with this
frontend; archived frontends retain their matching archived libraries.

```julia
using SDPX, SparseArrays
r = solve_conic([1.0], sparse([-1.0;;]), [-1.0], [NonnegativeConeT(1)]; return_result=true)
p = prepare([1.0], sparse([-1.0;;]), [-1.0], [NonnegativeConeT(1)])
try
    r = solve!(p; b=[-2.0])
finally
    close(p)
end
```

The direct form minimizes `x'P*x/2 + q'x` subject to `A*x+s=b`, with `s` in the supplied cones. PSD rows use upper-column svec packing with square-root-of-two off-diagonal scaling. `Model`, `variable!`, `constraint!`, and result recovery use ordinary symmetric matrix coordinates. MOI currently accepts Float64 models. BigFloat direct/model calls require matching `Settings(BigFloat; precision_bits=...)`; exact decimal transport never uses the Float64 result summaries. Prepared settings are fixed when the handle is created. `solve_time` reports Rust solver time, excluding Julia input conversion and output copying.

Direct solves default to `Settings(equilibration=:ruiz, presolve_enable=true,
chordal_decomposition_enable=true)`. Each setting can be disabled explicitly.
Julia `prepare` disables the two structural transformations in its private settings
copy to guarantee later `q`/`b` updates; it preserves the requested equilibration
and does not mutate caller settings. `result.info` records these effective enabled
settings. An enabled transformation may leave an already simplified problem unchanged.

`Settings(kkt_form=:auto)` lets the Rust engine select the KKT representation;
`:augmented` and `:condensed` request it explicitly. `Limits(threads=n)` budgets
the cone worker pool and eligible KKT factorization separately. Small cone
workloads remain serial. Float64 uses faer for multithreaded factorization when
the library includes `faer-sparse`; otherwise factorization uses serial QDLDL.
BigFloat uses serial QDLDL and can run independent cone blocks in parallel.
These settings are not a total process thread limit: native BLAS has a separate
budget and should use one thread when parallel cone workers execute PSD kernels.
`execution_plan(result)` reports the actual `kkt_form`, `factorization` (for example
`condensed_qdldl`), `precision_bits`, `backend_threads` and `cone_threads`.

Run `test/runtests.jl` in this package's isolated environment. It requires all supported arithmetic modes. Optional `test/pmp_callback.jl` runs with the existing PMP2SDP checkout developed into a separate test environment: its solver-neutral callback accepts this package directly, including high-precision results, without adapting or importing the old solver.

The model, storage, compiler, result, legacy ingestion, output policy, and MOI layers derive from the stable sibling `SDPX.jl/src/frontend`, `src/outputs.jl`, `src/runtime.jl` (result accessors), and `src/moi/optimizer.jl`. Their MIT copyright and license are retained in `LICENSE`. Numerical solver code is not copied into this Julia package.
