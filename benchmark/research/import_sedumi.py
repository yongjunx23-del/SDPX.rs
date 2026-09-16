#!/usr/bin/env python3
"""Convert real SeDuMi LP/SOCP/SDP data to SDPX JSON, without running a solver.

Requires NumPy/SciPy. PSD variables use symmetric upper svec coordinates;
free variables remain free. Rotated and complex cones are not supported.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from scipy.sparse import csc_matrix, eye, vstack


def vector(value):
    return np.asarray(value.toarray() if hasattr(value, 'toarray') else value).reshape(-1)


def csc(matrix):
    matrix = matrix.tocsc()
    matrix.sum_duplicates()
    matrix.sort_indices()
    return dict(m=matrix.shape[0], n=matrix.shape[1], colptr=matrix.indptr.tolist(),
                rowval=matrix.indices.tolist(), nzval=matrix.data.tolist())


def convert_data(data):
    D, c, b, k = data['A'].tocsc(), vector(data['c']), vector(data['b']), data['K']
    if set(k._fieldnames) - {'f', 'l', 'q', 'r', 's'} or np.any(getattr(k, 'r', 0)):
        raise ValueError('unsupported SeDuMi cone fields')
    if any(np.iscomplexobj(v) for v in (D.data, c, b)):
        raise ValueError('complex SeDuMi data are not supported')
    if not all(np.isfinite(v).all() for v in (D.data, c, b)):
        raise ValueError('nonfinite input')
    def sizes(field):
        values = np.atleast_1d(getattr(k, field, []))
        if any(v < 0 or int(v) != v for v in values):
            raise ValueError('invalid cone dimension')
        return [int(v) for v in values if v]
    fs, ls, qs, ss = (sizes(field) for field in ('f', 'l', 'q', 's'))
    if len(fs) > 1 or len(ls) > 1:
        raise ValueError('free and nonnegative dimensions must be scalar')
    f, l = sum(fs), sum(ls)
    linear = f + l + sum(qs)
    rows, cols, vals = list(range(linear)), list(range(linear)), [1.] * linear
    full = packed = linear
    for n in ss:
        for j in range(n):
            for i in range(j + 1):
                rows.append(full + i + j*n); cols.append(packed)
                vals.append(1. if i == j else 1/np.sqrt(2.))
                if i != j:
                    rows.append(full + j + i*n); cols.append(packed)
                    vals.append(1/np.sqrt(2.))
                packed += 1
        full += n*n
    if D.shape != (len(b), full) or len(c) != full:
        raise ValueError('SeDuMi matrix/cone dimensions disagree')
    lift = csc_matrix((vals, (rows, cols)), shape=(full, packed))
    E = D @ lift
    cones = [{'ZeroConeT': len(b)}] if len(b) else []
    if l: cones.append({'NonnegativeConeT': l})
    cones.extend({'SecondOrderConeT': n} for n in qs)
    cones.extend({'PSDTriangleConeT': n} for n in ss)
    A = vstack([E, -eye(packed, format='csc')[f:]], format='csc')
    return dict(P=csc(csc_matrix((packed, packed))), q=vector(lift.T @ c).tolist(),
                A=csc(A), b=np.r_[b, np.zeros(packed-f)].tolist(), cones=cones)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--source-sha256', required=True)
    args = parser.parse_args()
    if hashlib.sha256(args.source.read_bytes()).hexdigest() != args.source_sha256:
        parser.error('source SHA256 mismatch')
    result = convert_data(loadmat(args.source, squeeze_me=True, struct_as_record=False))
    with args.output.open('x') as output:
        output.write(json.dumps(result, separators=(',', ':'), allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
