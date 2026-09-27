# Ising scaling benchmark

`scaling.py` runs the sampled Ising SDP through the frozen native
`crates/solver/src/bin/sdpx.rs` executable.  The CLI accepts the SDPB sampled
JSON directory directly, so this benchmark no longer loads the retired SDPX.jl
frontend or materializes a Julia conic problem.

Each sample is one fresh CLI process and one solve.  There is no warmed,
repeated-in-process, iterate, or checkpoint timing mode.  The receipt's
`native_seconds` covers the numerical setup/solve core timer, `api_seconds`
covers native `into_solver` plus solve, and `load_seconds` covers sampled-input
parsing/compile.  Process startup/output and the independent `audit_point.jl`
process are outside those timers.  Results label this scope as
`fresh native CLI process`.

## Frozen configuration

The configuration is JSON and must use absolute paths.  `cli` identifies the
frozen native executable.  `source` is the frozen native source snapshot and is
hashed in the identity receipt; the executable and source are bound as one arm.
The optional `oracle` object identifies the Julia
runtime/project used only by the independent `GenericLinearAlgebra`/JSON
audit; it does not load `SDPX.jl`.

```json
{
  "cli": "/tmp/sdpx-rust-entry-20260922/sdpx",
  "source": "/campaign/native-source",
  "input": "/tmp/sdpx-refreshed-phases-20260917/common-sdp",
  "input_sha256": "e4484eb8895e504a8f5c83651a5b964172e24c7060db24ddf09b737dcc78fddf",
  "reference": "/path/to/reference-audit.json",
  "oracle": {
    "julia": "/absolute/path/to/julia",
    "project": "/campaign/audit-env"
  },
  "settings": "/campaign/ising-settings.json",
  "bits": 512,
  "plans": {
    "1": {"factorization": "condensed_sampled_arrow", "backend_threads": 1},
    "2": {"factorization": "condensed_sampled_arrow", "backend_threads": 2},
    "4": {"factorization": "condensed_sampled_arrow", "backend_threads": 4},
    "8": {"factorization": "condensed_sampled_arrow", "backend_threads": 8}
  }
}
```

`settings` can instead be an object and is written before launch.  It must
retain 1000 iterations and `tol_feas`, `tol_gap_abs` and `tol_gap_rel` at
`1e-42`; if `tol_feas_componentwise` is supplied, it may be any positive value
through the external `1e-30` gate and is preserved exactly.  A typical native
settings file is:

```json
{
  "max_iter": 1000,
  "time_limit": 120.0,
  "tol_feas": "1e-42",
  "tol_gap_abs": "1e-42",
  "tol_gap_rel": "1e-42",
  "max_threads": 8,
  "verbose": true
}
```

PSD cones use NT. Plan entries may use `linear_solver` and
`linear_solver_threads`; the older `factorization`/`backend_threads` spellings
remain accepted for receipts produced by the current CLI.

The input manifest hash is computed from every JSON file in the sampled
directory and must agree with the accepted reference (`input_sha256`, or the
legacy `mapping.input_sha256` field).  The identity receipt hashes the native
executable, optional source/provider files, input, reference, settings and all
benchmark/audit helpers before and after the run.  A changed identity makes
the campaign fail.

Returned settings must represent the requested MPFR values at the declared
precision. The driver compares exact nearest/even binary rounding, since a
round-trip decimal such as the MPFR representation of `1e-42` need not equal
the input decimal string. It rejects changed or missing settings; this does
not relax solver tolerances or the independent residual gates.

## Local screen and confirmation

```sh
python3 benchmark/ising/scaling.py --config /campaign/ising.json \
  --output /campaign/screen --widths 1,2,4,8 --seconds 600 --cell-seconds 180

python3 benchmark/ising/scaling.py --config /campaign/ising.json \
  --output /campaign/confirm --widths 1,2,4,8 --repetitions 3 \
  --seconds 1200 --cell-seconds 180
```

The one-round invocation is labelled `screen`; three repetitions are the
qualification shape and use three fresh processes per width.  Width order is
reversed in the middle repetition.  Every cell must finish within its whole
process cap, pass the native receipt gate, and pass the independent
original-coordinate audit.  Internal solver tolerances stay at `1e-42` and the
external primal, dual, gap, PSD, sampled-mapping and reference-objective gates
stay at `1e-30`.  A non-finite or failed point remains a failed sample and is
never omitted from the denominator.

An independently generated SDPB point can be checked with the same
original-coordinate oracle, without loading a solver frontend:

```sh
julia --startup-file=no benchmark/ising/audit_sdpb.jl \
  /campaign/sampled-input /campaign/sdpb-out /campaign/sdpb-audit.json \
  512 /campaign/accepted-reference.json
```

`SDPB_OUT` must contain the per-block `x_*.txt` and `X_matrix_*.txt` files,
`y.txt`, `Y_matrix_*.txt`, and SDPB's `out.txt` status log.  The audit converts
those files through the shared sampled mapping, checks the optimal status,
precision-tagged arbitrary-precision residual/PSD/objective gates, and writes
`accepted` only when every gate and the reference objective agree within
`1e-30`.  The harness verifies the solver's precision setting in its log; this
oracle process is outside solve timings.

On Linux the native process is pinned to unique physical cores from the
inherited affinity; PBS must reserve those cores.  macOS reports that hard
affinity is unavailable.  BLAS/OpenMP are held at one and the CLI's
`cone_threads`/`linear_solver_threads` receipts are checked independently.

`native_seconds`, `api_seconds`, `load_seconds`, iterations, solver metadata,
process RSS, commands and audit receipts are retained.  Summary speedup uses
the native solve time median, `S(p)=T(1)/T(p)`, and efficiency; audit and process
startup are reported separately.  Profile runs are diagnostic and do not earn
speed credit.  Run candidate arms sequentially under the shared research slot
lock.

`comparison.pbs` launches `scaling.py` using explicit
`SDPX_ISING_CONFIG` and `SDPX_ISING_OUTPUT` paths. Cluster timings remain
separate from local development and require a bounded campaign.
