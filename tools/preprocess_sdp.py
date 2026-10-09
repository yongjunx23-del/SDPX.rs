#!/usr/bin/env python3
"""Reduce Float64 equality/PSD JSON before constructing the Rust solver.

Requires NumPy and SciPy.

    python3 tools/preprocess_sdp.py reduce original.json reduced.json recovery.json
    sdpx reduced.json --precision 53 --output reduced-point.json
    python3 tools/preprocess_sdp.py lift original.json recovery.json reduced-point.json point.json

Audit point.json against original.json with the existing original-coordinate
audit and its unchanged tolerance. Timings in the solver point describe the
reduced solve; the preprocessing command reports its separate elapsed time.

PSD reduction selects a principal submatrix after verifying every common-kernel
relation against every original data matrix. It introduces no PSD coefficients
or block couplings. Equality-only, objective-free variables are eliminated in
independent sparse components; their primal values and equality multipliers are
recovered after the solve. Sampled factors and MPFR inputs are unsupported.
"""
import argparse
from fractions import Fraction
import gzip
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

EPS = np.finfo(np.float64).eps
VERIFY = 32 * EPS
CANDIDATE = 1e-10
ROOT2 = np.sqrt(2.0)


def read(path):
    raw = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def write(path, value):
    raw = json.dumps(value, separators=(",", ":"), allow_nan=False).encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def numbers(values):
    if any(type(v) not in (int, float) for v in values):
        raise ValueError("requires Float64 JSON numbers; MPFR/string coefficients are unsupported")
    out = np.asarray(values, dtype=np.float64)
    if not np.isfinite(out).all():
        raise ValueError("coefficients must be finite")
    return out


def matrix(data):
    out = sp.csc_matrix((numbers(data["nzval"]), data["rowval"], data["colptr"]),
                        shape=(data["m"], data["n"]))
    out.check_format(full_check=True)
    if not out.has_canonical_format:
        raise ValueError("requires sorted, unique CSC rows")
    out.eliminate_zeros()
    return out


def wire(value):
    value = value.tocsc()
    value.eliminate_zeros()
    value.sort_indices()
    return {"m": value.shape[0], "n": value.shape[1], "colptr": value.indptr.tolist(),
            "rowval": value.indices.tolist(), "nzval": value.data.tolist()}


def problem(data):
    if data.get("sampled") or data.get("precision_bits", 53) != 53:
        raise ValueError("only explicit Float64 inputs are supported; sampled factors are unsupported")
    a, p = matrix(data["A"]), matrix(data["P"])
    q, b = numbers(data["q"]), numbers(data["b"])
    n = len(q)
    if a.shape != (len(b), n) or p.shape != (n, n):
        raise ValueError("inconsistent problem dimensions")
    ranges, start = [], 0
    for cone in data["cones"]:
        if len(cone) != 1 or next(iter(cone)) not in ("ZeroConeT", "PSDTriangleConeT"):
            raise ValueError("only equality and PSD cones are supported")
        kind, size = next(iter(cone.items()))
        if type(size) is not int or size < 0:
            raise ValueError("invalid cone dimension")
        count = size if kind == "ZeroConeT" else size * (size + 1) // 2
        ranges.append((start, count, kind, size))
        start += count
    if start != len(b):
        raise ValueError("cone dimensions do not match b")
    return a, p, q, b, ranges


def snapped(values):
    out = values.copy()
    for i, v in enumerate(values.flat):
        rational = float(Fraction(float(v)).limit_denominator(1024))
        if abs(rational - v) <= 1e-9 * max(1.0, abs(v)):
            out.flat[i] = rational
    return out


def verified(residual, bound):
    return bool(np.all(np.abs(residual) <= VERIFY * bound))


def relation_error(residual, bound):
    ratio = np.divide(np.abs(residual), bound, out=np.zeros_like(bound), where=bound > 0)
    return float(ratio.max()) if ratio.size else 0.0


def face(a, b, start, side):
    count = side * (side + 1) // 2
    packed = a[start:start + count].tocoo()
    br = np.flatnonzero(b[start:start + count])
    row = np.r_[packed.row, br]
    col = np.r_[packed.col, np.full(len(br), a.shape[1])]
    val = np.r_[packed.data, b[start + br]]
    j, i = np.tril_indices(side)
    ii, jj = i[row], j[row]
    off = ii != jj
    val = val / np.where(off, ROOT2, 1.0)
    # Stack all symmetric data matrices, including B; its kernel is their
    # common kernel. Sparse construction keeps the original column supports.
    stack = sp.csc_matrix((np.r_[val, val[off]],
                           (np.r_[col * side + ii, col[off] * side + jj[off]],
                            np.r_[jj, ii[off]])), shape=((a.shape[1] + 1) * side, side))
    gram = (stack.T @ stack).toarray()
    d = np.sqrt(np.diag(gram))
    scale = np.outer(d, d)
    gram = np.divide(gram, scale, out=np.zeros_like(gram), where=scale > 0)
    residual = np.diag(gram).copy()
    lower = np.zeros_like(gram)
    remaining, pivots = list(range(side)), []
    for k in range(side):
        t = remaining[int(np.argmax(residual[remaining]))]
        if residual[t] <= CANDIDATE:
            break
        remaining.remove(t)
        pivots.append(t)
        lower[t, k] = np.sqrt(residual[t])
        lower[remaining, k] = (gram[remaining, t] - lower[remaining, :k] @ lower[t, :k]) / lower[t, k]
        residual[remaining] -= lower[remaining, k] ** 2
    relations, worst = [], 0.0
    absolute = abs(stack)
    for t in remaining:
        coeff = (la.solve_triangular(lower[pivots, :len(pivots)].T, lower[t, :len(pivots)],
                                     lower=False, check_finite=False) * d[t] / d[pivots]
                 if d[t] else np.zeros(len(pivots)))
        for candidate in (snapped(coeff), coeff):
            v = np.zeros(side)
            v[t], v[pivots] = 1.0, -candidate
            residual, bound = stack @ v, absolute @ np.abs(v)
            if verified(residual, bound):
                worst = max(worst, relation_error(residual, bound))
                relations.append([t, [[q, float(c)] for q, c in zip(pivots, candidate) if c != 0]])
                break
    if not relations:
        return None
    removed = {t for t, _ in relations}
    return {"start": start, "side": side, "keep": [i for i in range(side) if i not in removed],
            "relations": relations, "residual": worst}


def equality_components(a, p, q, equality):
    active = (p.getnnz(axis=0) + p.getnnz(axis=1) > 0) | (q != 0)
    free = (~active) & (a[~equality].getnnz(axis=0) == 0)
    empty = np.flatnonzero(free & (a.getnnz(axis=0) == 0))
    columns = np.flatnonzero(free & (a.getnnz(axis=0) != 0))
    f = a[equality][:, columns]
    pattern = f.copy()
    pattern.data[:] = 1.0
    count, labels = connected_components(pattern.T @ pattern, directed=False)
    eqrows = np.flatnonzero(equality)
    components = []
    for label in range(count):
        cols = columns[labels == label]
        rows = eqrows[np.unique(a[equality][:, cols].indices)]
        f = a[rows][:, cols].toarray()
        _, r, permutation = la.qr(f, mode="economic", pivoting=True, check_finite=False)
        diag = np.abs(np.diag(r))
        rank = int(np.count_nonzero(diag > CANDIDATE * diag.max()))
        selected = permutation[:rank]
        _, _, roworder = la.qr(f[:, selected].T, mode="economic", pivoting=True, check_finite=False)
        pivot_rows, other_rows = roworder[:rank], roworder[rank:]
        pivot = f[pivot_rows][:, selected]
        coeff = la.solve(pivot.T, f[other_rows][:, selected].T, check_finite=False).T
        for candidate in (snapped(coeff), coeff):
            residual = f[other_rows] - candidate @ f[pivot_rows]
            bound = np.abs(f[other_rows]) + np.abs(candidate) @ np.abs(f[pivot_rows])
            if verified(residual, bound):
                components.append({"columns": cols.tolist(), "selected": cols[selected].tolist(),
                                   "pivot_rows": rows[pivot_rows].tolist(),
                                   "other_rows": rows[other_rows].tolist(), "coeff": wire(sp.csc_matrix(candidate)),
                                   "residual": relation_error(residual, bound)})
                break
    return empty, components


def reduce(data):
    a, p, q, b, ranges = problem(data)
    equality = np.zeros(a.shape[0], dtype=bool)
    for start, count, kind, _ in ranges:
        if kind == "ZeroConeT":
            equality[start:start + count] = True
    empty, components = equality_components(a, p, q, equality)
    rowkeep, colkeep = np.ones(a.shape[0], dtype=bool), np.ones(a.shape[1], dtype=bool)
    colkeep[empty] = False
    for comp in components:
        rowkeep[comp["pivot_rows"]], colkeep[comp["columns"]] = False, False
    faces, cones = [], []
    for start, count, kind, side in ranges:
        if kind == "PSDTriangleConeT" and side:
            found = face(a, b, start, side)
            if found:
                faces.append(found)
                rowkeep[start:start + count] = False
                keep = found["keep"]
                rows = [start + j * (j + 1) // 2 + i for k, j in enumerate(keep) for i in keep[:k + 1]]
                rowkeep[rows] = True
                side = len(keep)
        size = int(np.count_nonzero(rowkeep[start:start + count])) if kind == "ZeroConeT" else side
        if size:
            cones.append({kind: size})
    rows, columns = np.flatnonzero(rowkeep), np.flatnonzero(colkeep)
    rowindex = np.full(a.shape[0], -1, dtype=int)
    rowindex[rows] = np.arange(len(rows))
    qr, qc, qv = [np.arange(len(rows))], [rows], [np.ones(len(rows))]
    for comp in components:
        coeff = matrix(comp["coeff"]).tocoo()
        qr.append(rowindex[np.asarray(comp["other_rows"])[coeff.row]])
        qc.append(np.asarray(comp["pivot_rows"])[coeff.col])
        qv.append(-coeff.data)
    transform = sp.csr_matrix((np.concatenate(qv), (np.concatenate(qr), np.concatenate(qc))),
                              shape=(len(rows), a.shape[0]))
    out = {"P": wire(p[columns][:, columns]), "q": q[columns].tolist(),
           "A": wire(transform @ a[:, columns]), "b": (transform @ b).tolist(), "cones": cones}
    if "settings" in data:
        out["settings"] = data["settings"]
    recovery = {"precision_bits": 53, "rows": rows.tolist(), "columns": columns.tolist(),
                "faces": faces, "equalities": components,
                "summary": {"original_n": a.shape[1], "reduced_n": len(columns),
                            "original_m": a.shape[0], "reduced_m": len(rows),
                            "empty_columns": len(empty), "equality_columns": sum(len(c["columns"]) for c in components),
                            "equality_rows": sum(len(c["pivot_rows"]) for c in components),
                            "psd_orders": [[f["side"], len(f["keep"])] for f in faces],
                            "original_nnz": a.nnz, "reduced_nnz": len(out["A"]["nzval"]),
                            "relation_tolerance": VERIFY,
                            "max_relation_residual": max([f["residual"] for f in faces] +
                                                         [c["residual"] for c in components], default=0.0)}}
    return out, recovery


def smat(values, side):
    j, i = np.tril_indices(side)
    values = np.asarray(values) / np.where(i == j, 1.0, ROOT2)
    out = np.zeros((side, side))
    out[i, j], out[j, i] = values, values
    return out


def svec(value):
    j, i = np.tril_indices(value.shape[0])
    return value[i, j] * np.where(i == j, 1.0, ROOT2)


def lift(data, recovery, point):
    a, p, q, b, _ = problem(data)
    if point.get("precision_bits") != 53 or recovery.get("precision_bits") != 53:
        raise ValueError("lifting requires a Float64 point and Float64 recovery metadata")
    columns, rows = recovery["columns"], recovery["rows"]
    rx, rs, rz = numbers(point["x"]), numbers(point["s"]), numbers(point["z"])
    if len(rx) != len(columns) or len(rs) != len(rows) or len(rz) != len(rows):
        raise ValueError("point dimensions do not match the reduced problem")
    x, s, z = np.zeros(len(q)), np.zeros(len(b)), np.zeros(len(b))
    x[columns], s[rows], z[rows] = rx, rs, rz
    certificate = "Infeasible" in point["status"]
    for comp in recovery["equalities"]:
        pivots, selected = comp["pivot_rows"], comp["selected"]
        rhs = -a[pivots][:, columns] @ rx
        if not certificate:
            rhs += b[pivots]
        x[selected] = la.solve(a[pivots][:, selected].toarray(), rhs, check_finite=False)
        z[pivots] = -matrix(comp["coeff"]).T @ z[comp["other_rows"]]
    for f in recovery["faces"]:
        start, side, keep = f["start"], f["side"], f["keep"]
        w = np.zeros((side, len(keep)))
        w[keep, np.arange(len(keep))] = 1.0
        lookup = {q: k for k, q in enumerate(keep)}
        for t, coeff in f["relations"]:
            for qindex, c in coeff:
                w[t, lookup[qindex]] = c
        retained = [start + j * (j + 1) // 2 + i for k, j in enumerate(keep) for i in keep[:k + 1]]
        s[start:start + side * (side + 1) // 2] = svec(w @ smat(s[retained], len(keep)) @ w.T)
    out = dict(point, x=x.tolist(), s=s.tolist(), z=z.tolist())
    if not certificate:
        fullp = p + p.T - sp.diags(p.diagonal())
        quadratic = float(x @ (fullp @ x)) / 2
        out["objective"], out["dual_objective"] = float(q @ x) + quadratic, -float(b @ z) - quadratic
    out["preprocessing"] = {k: recovery[k] for k in ("input_sha256", "reduced_sha256", "summary")}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command")
    r = commands.add_parser("reduce")
    for name in ("input", "output", "recovery"):
        r.add_argument(name, type=Path)
    l = commands.add_parser("lift")
    for name in ("input", "recovery", "point", "output"):
        l.add_argument(name, type=Path)
    args = parser.parse_args()
    if args.command is None:
        parser.error("choose reduce or lift")
    started = time.perf_counter()
    data, identity = read(args.input)
    if args.command == "reduce":
        reduced, recovery = reduce(data)
        recovery["input_sha256"] = identity
        recovery["reduced_sha256"] = write(args.output, reduced)
        write(args.recovery, recovery)
        print(json.dumps({**recovery["summary"], "preprocess_seconds": time.perf_counter() - started}))
    else:
        recovery, _ = read(args.recovery)
        if recovery["input_sha256"] != identity:
            raise ValueError("original input does not match the recovery metadata")
        point, _ = read(args.point)
        write(args.output, lift(data, recovery, point))
        print(json.dumps({"output": str(args.output), "lift_seconds": time.perf_counter() - started}))


if __name__ == "__main__":
    main()
