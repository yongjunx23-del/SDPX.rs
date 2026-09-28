# SDPX

One Rust HSD conic solver (Clarabel.rs-derived) with a native Rust API, the
`sdpx` CLI and a C ABI. Float64 and MPFR share one engine. Julia in this repo
only generates inputs or audits results; it is never a solver frontend.

This file is the single source for **how to work**. Other documents:

| Document | Holds |
|---|---|
| [REVIEW_AND_PLAN.md](REVIEW_AND_PLAN.md) | Current status, known failures, priorities, closed directions |
| [docs/JOURNAL.md](docs/JOURNAL.md) | Completed experiments and their conclusions (append-only) |
| [README.md](README.md) | Users: overview, quick start, Rust examples, performance |
| `benchmark/e2e/README.md` | The per-change check tool |
| `benchmark/{research,ising,float64,parallel,mpfr}/README.md` | Milestone and external comparisons only |

`benchmark/` and `docs/archive/` are local working copies, git-ignored and not
published; keep inputs and harness changes there, never in commits.

## Development loop

Every change is judged by one complete solve of the matching pinned case:
status, original-coordinate audit, and time at unchanged precision and
tolerances.

```sh
python3 benchmark/e2e/e2e.py build                  # fast profile → frozen arm
python3 benchmark/e2e/e2e.py run medium             # Float64 path
python3 benchmark/e2e/e2e.py run ising11 --arm NAME # MPFR / sampled path
python3 benchmark/e2e/e2e.py ab ising11 BASE CAND   # performance: A-B-B-A
```

Pick the check by what the change touches:

| Change | Check |
|---|---|
| Refactor, move, dead-code removal | `ab` on both cases; `points identical: yes` required |
| Float64 numerics or performance | `ab medium` (release arms for quoted numbers) |
| MPFR, cones, sampled operator, condensed KKT | `ab ising11`; local SOC also `ab csdr3` |
| Input/output, CLI, settings | `run` on both cases |
| Threading | `run CASE --threads N` for the affected widths |
| MPI | real MPI E2E on a host with MPI; mock tests are not enough |
| Docs only | `git diff --check` |

Do not add unit tests, multi-case screens, repeated A/B, microbenchmarks or the
full suite as routine gates. Run them only when asked or when the change is to
that tooling. Before a release or integration milestone, run
`cargo test --locked --release --workspace --features sdpx-ffi/sdp-accelerate,sdpx-ffi/faer-sparse -- --test-threads=1`.

Rules for results:

- A numerical outcome passes only if status and the external audit pass the
  existing gates. Known failures are listed in the plan. They stay failures and
  do not block unrelated work. A new failure or a worse residual must be
  investigated.
- Iterations and time are diagnostics unless performance is the subject. A
  valid algorithm change need not reproduce old iteration counts or objective
  digits.
- Timing: `fast` for development, `release` for any number you quote. Run arms
  sequentially on one host, never concurrently. A single run is preliminary; say so.
- Run large builds, full-suite tests and substantial numerical runs on the
  cluster through PBS. Keep local work to editing, inspection and lightweight
  preparation.
- Record: when a candidate is kept or reverted, add one dated entry to
  `docs/JOURNAL.md` and update the plan's status table if a headline number
  moved. Raw run rows are appended automatically.
- Scratch files, logs and profiles go in `$SDPX_E2E_HOME` (default
  `~/.cache/sdpx-e2e`) or the session scratchpad, never in the repo. Do not rely
  on `/tmp` for anything a later session needs.

## Numerical contracts (never weaken)

- Keep working precision, original-coordinate outputs, accepted-iterate
  recovery, convergence and infeasibility criteria, reduced tolerances,
  regularization (including ×100 escalation, at most 3 levels) and refinement,
  all following Clarabel.rs. `AlmostSolved` is never promoted. Do not
  reintroduce an independent certificate stage or status promotion.
- Do not relax accuracy gates, mix precisions, factor MPFR problems in Float64,
  use precision-ladder warm starts, or branch on benchmark names.
- `tol_feas_componentwise` stays optional and off by default.
- Direct solves default to Ruiz, presolve and chordal. Prepared handles keep
  Ruiz and disable structural preprocessing for reusable updates.
- MPFR values own their storage; decimal inputs reach MPFR without a Float64
  intermediate. Sampled factors define the operator; never replace them with
  rounded materializations.
- MPFR accumulation keeps per-term FMA order. The one exception is exact
  accumulation rounded once at the destination (RNS, `exactdot`), which is
  more accurate.
- High-precision NT scaling uses a direct SVD. Do not replace it with an
  eigendecomposition of `MᵀM`; keep the ill-conditioned SPD regression.
- Parallel and distributed results equal serial results bitwise, or the
  difference is documented (MPFR reductions fold in rank order).
- PSD cones use NT only.

## Code

- Layout: `crates/solver/src/solver/{core,default,cones,kkt,sampled,distributed,chordal}`;
  unit tests in each module's `tests/`. The public API is the flat
  `sdpx_solver::solver::*` facade.
- Target ≤ 30k production lines by removing duplication, not by wrapping or
  compressing. Prefer small typed interfaces. Keep upstream attribution
  (`provenance/`).
- Fix warnings in touched code. Run `rustfmt` on files you edit, not on the
  whole tree.
- Write maintained documentation in concise English; keep experiment detail
  in `docs/JOURNAL.md` and current decisions in `REVIEW_AND_PLAN.md`.
- Commit only when asked. End commit messages with the attribution line the
  harness supplies.

## Working with other agents

Work alone on small or tightly coupled changes. Delegate only independent,
bounded work, giving each worker a file ownership and an acceptance command
(one of the `e2e.py` checks above). Preserve other workers' uncommitted edits
in the shared checkout; commit or stash them first if a change must span them.
Timed runs stay serial per host. Cluster work uses the `ucas-hpc` skill and
runs only within the scope the user authorizes.
