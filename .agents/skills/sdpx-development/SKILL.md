---
name: sdpx-development
description: Build and verify the SDPX Rust solver, CLI and C ABI with pinned Float64/MPFR solves.
---

# SDPX development

Follow [AGENTS.md](../../../AGENTS.md) for rules and checks, and
[REVIEW_AND_PLAN.md](../../../REVIEW_AND_PLAN.md) for current priorities.

## Build

`benchmark/e2e/e2e.py build --arm NAME` configures the local toolchain when
`CARGO_HOME` is unset. For direct Cargo commands:

```sh
export CARGO_HOME=/Users/xuyongjun/.local/share/sdpx-toolchain/cargo
export RUSTUP_HOME=/Users/xuyongjun/.local/share/sdpx-toolchain/rustup
export PATH="$CARGO_HOME/bin:$PATH"
```

Build offline with the lockfile. Use `sdp-accelerate,faer-sparse` on macOS or
`sdp-openblas,faer-sparse` on ordinary Linux builds. On the cluster, keep
the validated dynamic OpenBLAS linkage described by `ucas-hpc`. Use `fast` for development and
`--profile release` for reported timings. Compare frozen arms: builds overwrite
`target/PROFILE/sdpx`.

## Evidence

`$SDPX_E2E_HOME` (default `~/.cache/sdpx-e2e`) stores frozen `arms/`, verified
`data/`, audited `runs/` and `journal.jsonl`. Preserve input/settings hashes,
binary identity and source changes. Frozen arms include tracked patches, a
build-input manifest and copies of untracked source files.

- Pinned cases: `medium`, `ising11` and `csdr3`; commands are in AGENTS.md.
- Ising audit: `benchmark/e2e/audit-env`; set `SDPX_E2E_JULIA` to select Julia.
  Audit at the point's requested precision with the unchanged 1e-30 gate.
- Other Float64 inputs: CLI solve, then `benchmark/research/native.py` audit.
- `csdr3` pins the reconstructed CSDR input and audit. It is not the missing
  historical Julia input. Check one/four-thread parity for SOC changes.
- External comparisons: `benchmark/{research,ising}/README.md`.
- Cluster work: use `ucas-hpc` within the user's authorized scope.

Julia is for input generation and audits. For BFLA/MFLA provider experiments,
use separate processes/environments and `--gcthreads=1`. Use fresh output paths.
