# Reference adapters

`mosek_runner.py` is the retained native MOSEK benchmark adapter recovered from
`sdpx-perf-20260913-production03-01`. Its cone conversion, product defaults,
original-coordinate oracle, and timing scopes are preserved. `providers.py`
contains the corresponding installed-distribution fingerprint function.

Run through [references.py](../../research/references.py), which hashes both
adapter sources and installed reference packages. These tools do not contain or
invoke the retired Julia solver. MOSEK remains an optional separately licensed
reference dependency.
