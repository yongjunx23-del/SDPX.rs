# Public regression inputs

Losslessly compressed original JSON inputs, recovered from retained SDPX benchmark
snapshots. Decompressed SHA256, upstream source URLs/hashes, conversion conventions,
and historical membership remain in [catalog.json](../../research/catalog.json).
These are exposed regression/development data, not the reserved holdout.

Recovery sources: `sdpx-perf-20260913-production03-01` and `sdpx-families/data`
on the existing project cluster. No retired solver code was restored.

Validate with `python3 benchmark/research/catalog.py verify --workspace .. --suite regression`.
