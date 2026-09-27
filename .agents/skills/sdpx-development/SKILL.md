---
name: sdpx-development
description: Build, run and check the SDPX Rust solver (CLI, C ABI, Float64/MPFR paths) with the pinned end-to-end harness.
---

# SDPX development

Rules, contracts and the per-change check table are in
[AGENTS.md](../../../AGENTS.md). This skill covers only environment details.

## Toolchain

`benchmark/e2e/e2e.py build` sets these automatically when `CARGO_HOME` is unset:

```sh
export CARGO_HOME=/Users/xuyongjun/.local/share/sdpx-toolchain/cargo
export RUSTUP_HOME=/Users/xuyongjun/.local/share/sdpx-toolchain/rustup
export PATH="$CARGO_HOME/bin:$PATH"
```

Builds are `--locked --offline`. Features are `sdp-accelerate,faer-sparse` on
macOS and `sdp-openblas,faer-sparse` on Linux. `fast` (no LTO, parallel
codegen, incremental) is for development; `release` is for quoted timings.
`target/fast/sdpx` is overwritten by every build, so always compare frozen arms
(`e2e.py build --arm NAME`).

## Harness state

`$SDPX_E2E_HOME` (default `~/.cache/sdpx-e2e`) contains `arms/` (frozen
binaries with commit and dirty-patch identity), `data/` (unpacked inputs,
hash-checked), `runs/` (result, stderr, settings and audit per solve) and
`journal.jsonl`. The Ising audit uses the committed Julia project
`benchmark/e2e/audit-env` (JSON, GenericLinearAlgebra, SHA); set
`SDPX_E2E_JULIA` to choose the Julia binary.

## Beyond the pinned cases

- Other Float64 inputs: `sdpx INPUT.json --settings S.json --output R.json`, then
  audit with `benchmark/research/native.py` (`native.audit`).
- MOSEK/Clarabel comparisons, the 9-case development set and holdout:
  `benchmark/research/README.md`.
- Ising scaling and SDPB comparisons: `benchmark/ising/README.md`; cluster work
  uses the `ucas-hpc` skill.
- If testing sibling BFLA/MFLA providers, use separate processes and
  environments with `--gcthreads=1`.
