# Per-change E2E check

`e2e.py` builds a frozen CLI, solves a pinned case in a fresh process, runs the
independent original-coordinate audit outside all timers, and appends one row
to `$SDPX_E2E_HOME/journal.jsonl` (default `~/.cache/sdpx-e2e`).

```sh
python3 benchmark/e2e/e2e.py build [--profile release] [--arm NAME]
python3 benchmark/e2e/e2e.py run medium|ising11 [--arm NAME | --cli PATH] [--threads N]
python3 benchmark/e2e/e2e.py ab CASE ARM_A ARM_B [--order ABBA]
python3 benchmark/e2e/e2e.py list
```

`run` exits 0 only when the audit passes. When a case has a known failure, it
prints the note from `cases.json` next to the failing result. `ab` prints each
run, the median `api_seconds` per arm, whether all returned points are bitwise
identical, and the audit pass count.

| Case | Input | Precision / tolerance | Audit |
|---|---|---|---|
| `medium` | SU(2) path SDP, n=1887, m=13739 (from `大规模矩阵` MOSEK export) | Float64, 1e-6 | `research/native.py`, 1e-6 |
| `ising11` | SDPB `pmp2sdp` JSON, Ising Λ11 | 512-bit, internal 1e-42 | `ising/audit_point.jl` vs accepted SDPB reference, 1e-30 |

Inputs are committed under `data/` and verified against the SHA256 in
`cases.json` when unpacked. The medium hash is of the JSON file; the ising11
hash is the sampled-directory manifest hash used by `benchmark/ising`. To add a
case, commit a compressed input, then add an entry with its hash, settings and
audit.
