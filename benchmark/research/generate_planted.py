#!/usr/bin/env python3
"""Reproduce the three September 16 synthetic fixtures; no solver is called.

Each model has x*=0, b=s*, q=-A^T z*, with s*,z* complementary
cone elements. They are diagnostics, not representative public benchmarks.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
data = args.output.resolve()
data.mkdir(parents=True, exist_ok=False)
manifest = []

def matrix(m, n, cols):
    ptr = [0]
    rows = []
    vs = []
    for col in cols:
        for i, v in sorted(col.items()):
            if v:
                rows.append(i)
                vs.append(v)
        ptr.append(len(rows))
    return dict(m=m, n=n, colptr=ptr, rowval=rows, nzval=vs)

def save(name, n, m, cones, b, z, cols, family, seed):
    q = [-sum((v * z[i] for i, v in col.items())) for col in cols]
    obj = dict(P=matrix(n, n, [{} for _ in range(n)]), q=q, A=matrix(m, n, cols), b=b, cones=cones)
    raw = (json.dumps(obj, separators=(',', ':')) + '\n').encode()
    sha = hashlib.sha256(raw).hexdigest()
    (data / (name + '.json')).write_bytes(raw)
    manifest.append(dict(name=name, family=family, n=n, m=m, cones=cones, json_sha256=sha, seed=seed, source='fresh planted KKT; x*=0, complementary diagonal/axis s,z; objective=0', nnz_A=len(obj['A']['nzval'])))
for name, n, blocks, size, seed in [('SOCP_axis_many', 128, 32, 17, 2026091601), ('SOCP_axis_wide', 192, 2, 257, 2026091602)]:
    r = random.Random(seed)
    m = blocks * size
    b = [0.0] * m
    z = [0.0] * m
    for k in range(blocks):
        b[k * size] = b[k * size + 1] = 1.0
        z[k * size] = 1.0
        z[k * size + 1] = -1.0
    cols = [{i: r.randint(-8, 8) / 8 for i in r.sample(range(m), min(m, 40))} for _ in range(n)]
    save(name, n, m, [{'SecondOrderConeT': size} for _ in range(blocks)], b, z, cols, 'SOCP', seed)
name = 'SDP_planted80'
n = 300
d = 80
m = d * (d + 1) // 2
seed = 2026091603
r = random.Random(seed)
b = [0.0] * m
z = [0.0] * m
for j in range(d):
    if j < d // 2:
        b[j * (j + 1) // 2 + j] = 1.0
    else:
        z[j * (j + 1) // 2 + j] = 1.0
cols = [{i: r.randint(-8, 8) / 8 for i in r.sample(range(m), 160)} for _ in range(n)]
save(name, n, m, [{'PSDTriangleConeT': d}], b, z, cols, 'SDP', seed)
(data / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
