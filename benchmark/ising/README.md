# Ising512 comparison

This driver repeats the frozen SDPB Ising sampled problem using the Julia frontend and Rust core. `sampled/` and `audit_helpers.jl` preserve the prior compiler, provenance and original-coordinate audit. The numerical protocol below fixes tolerances, thread budgets and timing scope.

Prepare a fresh campaign directory with `source/`, the unchanged `common-sdp/` and `input.sha256`, a copied SDPB executable `sdpb` and its environment script, a pinned Julia `env/` containing this SDPX, JSON 0.21.4 and GenericLinearAlgebra, accepted prior `reference/sdpb1-audit.json` and `reference/input.json`, and `benchmark/ising/`. Record source hashes in `source-identity.json`. Adjust campaign/toolchain paths and a currently healthy PBS node in `comparison.pbs`. Fetch locked Cargo dependencies before submission; sustained builds run in PBS. `pic.cmake` supplies position-independent Fortran objects required by the shared library.

The controller performs a small Linux gate, then serially runs SDPX 1/2/4/8-thread cells with one first call and three warmed fresh solves each. Every point must pass external audits and actual precision/KKT/cone-thread checks. It then runs three fresh SDPB references at each matching width. Any failed/incomplete stage stops subsequent work. The source, environment, inputs and loaded library must remain frozen throughout.

Results under `results/<job>/` include raw points, audit JSON, exact commands, solver/API/compilation/process timings, immutable identities and private CPU metadata. `memory.json` reports the sampled peak sum of process-tree/MPI-rank RSS and individual process peaks, separately from GNU time's maximum RSS. Shared pages are counted once per process; aggregate RSS is not PSS. Compare native solver and process startup costs separately, and keep first-call and warmed timings distinct.

This condensed MPFR configuration reports one QDLDL thread and the cone worker budget. Condensed scaling, operator applications and eligible Schur contributions share that pool; contribution caching is bounded and overlapping sums retain a fixed order. MPFR factorization remains serial. Linux Netlib/faer serves Float64; MPFR retains its high-precision arithmetic.

## Numerical protocol

This campaign repeats the frozen Ising512 sampled SDP without regenerating or changing its finite problem. The prior common-sdp, input manifest, SDPB executable and sampled-primal compiler retain SHA256 identities. Primal variables are SDPB sample-equation multipliers; equality constraints are the original sampled B transpose equations; PSD slacks use sqrt(2)-scaled upper-column-major svec. There is no rank reduction or benchmark-only solver branch.

Both solvers use 512-bit arithmetic, 1e-42 internal primal/dual/gap tolerances and 1000 maximum iterations. SDPX uses its default Ruiz, presolve and chordal settings, with automatic KKT selection. Archived campaigns that disabled preprocessing remain separate baselines and must retain their original settings labels. Every returned first/warmed point is audited outside solver timing: normalized original-coordinate primal/dual residuals, relative gap, PSD violations and sampled mapping residuals must be <=1e-30, with optimal status and relative objective agreement <=1e-30 against the previously accepted frozen SDPB reference. GenericLinearAlgebra BigFloat eigenvalues provide external PSD checks. These checks qualify only the fixed finite SDP.

One ordinary 8-core, 48 GB, four-hour PBS allocation builds the locked Linux release with Netlib PIC and faer enabled. A Linux analytic Float64/MPFR512 gate precedes candidate execution. SDPX widths 1, 2, 4 and 8 run sequentially in separate fresh Julia processes, each with one first call and three warmed fresh solves. Every solve starts a new solver and keeps no iterate/checkpoint. Explicit budgets bind each Julia process to distinct physical cores, set BLAS/OpenMP to one and require returned metadata to report condensed QDLDL, one KKT thread, and the requested cone thread count.

The single-thread candidate cell is the bounded pilot. Any failure stops the campaign; a finite maximum wall budget prevents launching work beyond the allocation. Only after all candidate points pass does the controller run three fresh SDPB MPI processes at each width 1/2/4/8 on the same node, each independently audited. No numerical benchmarks run concurrently. The build is capped at 45 minutes. The controller deadline is the earlier of three hours after build completion or 3h40m after PBS script start, leaving at least 20 minutes of allocation headroom for receipts. Each native SDPX call has an 850-second limit, each SDPB process a 900-second limit, and each candidate cell an external 3550-second cap.

Preserve source/library/environment/input/helper hashes before and after, exact commands and CPU binding, raw points, all audits, actual solver plan, input construction, first-call compilation, native solve and public API timing, and whole-process time. A 250 ms sampler sums current RSS over process-group descendants and explicitly identified MPI ranks, while separately tracking each process peak; shared pages count per process, so this is RSS sum rather than PSS. GNU time's maximum RSS remains a separate raw measure. Timeouts and failures remain incomplete; only complete compatible repetitions support median comparisons or speed credit (at least 2%).

## Multi-node preparation

`rank_exec.py` records global MPI rank, local rank and hostname. Multi-node
launches require `SDPX_RANK_CPU_MAP`, a JSON file mapping each exact compute-node
hostname to its preselected physical CPU IDs. Binding is checked against inherited
affinity; global-rank receipt filenames prevent cross-node collisions. The caller
must establish physical topology inside the allocation before writing that map.
The existing controller and RSS sampler remain single-node; this binding support
does not implement an SDPX MPI backend or multi-node memory aggregation. The
1/4/16/64/256-core campaign and numerical prerequisites are tracked in
[the performance plan](../../PERFORMANCE_PLAN.md).
