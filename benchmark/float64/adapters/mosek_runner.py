#!/usr/bin/env python3
"""MOSEK third leg for the fixed upstream conic benchmark.

Consumes the same Clarabel JSON bytes as ``runner/sdpx_runner.jl`` and
``runner/rust-harness/src/main.rs``:

    min q'x   s.t.   A x + s = b,   s in K      (P = 0 only)

and writes one JSON line per instance to stdout, plus one header line.  The
summary/run fields mirror ``sdpx_runner.jl`` so ``compare.py`` and any reader of
the Julia/Rust legs can consume the output unchanged.

Native API only: a fresh ``mosek.Task`` is built per repetition. By default
PSD/NN/Zero problems use native matrix variables for the negative algebraic
dual; other cones use AFE/ACC. ``--formulation afe`` retains the ACC baseline.
No CVXPY, no warm start. Model construction is included in setup timing.

Conventions
-----------
* Clarabel PSD rows are the upper triangle, column by column, with
  off-diagonals scaled by ``sqrt(2)`` (``v = sqrt(2) * X[i, j]``, i <= j).
* MOSEK's ``appendsvecpsdconedomain`` is the lower triangle, column by column,
  with the non-diagonal elements rescaled (MOSEK Python API doc: "vectorizations
  of the lower-triangular part of a positive semidefinite matrix, with the
  non-diagonal elements additionally rescaled").
* Both are the same scaled ``svec`` cone, so a symmetric entry ``X[i, j]``
  (i <= j) sits at Clarabel index ``cl(i, j)`` and at MOSEK index
  ``mk(j, i)``.  The ACC therefore references its AFE rows in MOSEK order
  (``afe_order[mk] = cl``), and ``getaccdoty`` is mapped back through the
  inverse permutation before the oracle runs.  ``--selftest`` checks that
  index algebra on random symmetric matrices without touching MOSEK.

Settings discipline
-------------------
Only ``MSK_IPAR_NUM_THREADS = 1`` is set.  Presolve, scaling and every
tolerance stay at the MOSEK product defaults; the header's ``default_receipt``
records the actual conic tolerances, presolve mode and so on read back from a
fresh task, so the metadata states measured values instead of assumed ones.
The independent oracle gate is ``--tol`` (default 1e-6), applied to the
original-coordinate residuals exactly as in the Julia and Rust legs.

Usage
-----
    python3 mosek_runner.py fixtures/json/*.json --runs=4 --tol=1e-6
    python3 mosek_runner.py --selftest

Exit codes: 0 all instances passed, 1 at least one instance failed,
2 the MOSEK Python package is unavailable, 3 a license error was raised.
"""

from __future__ import annotations

import os

# Pin BLAS/OpenMP threads before numpy, scipy or MOSEK load any runtime.
for _thread_env in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_env, "1")

import argparse
import json
import math
import sys
import time

import numpy as np
import scipy.sparse as sp

try:
    import mosek
except ImportError as _import_error:  # pragma: no cover - environment dependent
    mosek = None
    MOSEK_IMPORT_ERROR: Exception | None = _import_error
else:
    MOSEK_IMPORT_ERROR = None


ALLOWED_CONES = (
    "ZeroConeT",
    "NonnegativeConeT",
    "SecondOrderConeT",
    "PSDTriangleConeT",
)

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_NO_SOLVER = 2
EXIT_LICENSE = 3

# MOSEK result codes that mean "no usable license", not "bad model".
LICENSE_RESCODES = frozenset(
    name
    for name in (
        "err_license",
        "err_license_cannot_allocate",
        "err_license_cannot_connect",
        "err_license_expired",
        "err_license_feature",
        "err_license_invalid_hostid",
        "err_license_max",
        "err_license_moseklm_daemon",
        "err_license_no_server_line",
        "err_license_no_server_support",
        "err_license_old_server_version",
        "err_license_server",
        "err_license_server_version",
        "err_license_version",
        "err_missing_license_file",
        "err_file_license",
        "err_optimizer_license",
        "err_platform_not_licensed",
        "err_prob_license",
        "err_size_license",
        "err_size_license_con",
        "err_size_license_intvar",
        "err_size_license_var",
    )
)

SOLSTA_CLASS = {
    "optimal": "optimal",
    "prim_feas": "almost_solved",
    "dual_feas": "almost_solved",
    "prim_and_dual_feas": "almost_solved",
    "integer_optimal": "almost_solved",
    "prim_infeas_cer": "primal_infeasible",
    "dual_infeas_cer": "dual_infeasible",
    "prim_illposed_cer": "ill_posed",
    "dual_illposed_cer": "ill_posed",
    "unknown": "unknown",
}


class Unsupported(Exception):
    """The fixture is outside this leg's contract (nonzero P, unknown cone)."""


class LicenseUnavailable(Exception):
    """MOSEK raised a license result code."""


# ---------------------------------------------------------------------------
# JSON plumbing
# ---------------------------------------------------------------------------


def _json_safe(value):
    """Convert numpy scalars/arrays to plain JSON types; non-finite -> null."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating,)):
        as_float = float(value)
        return as_float if math.isfinite(as_float) else None
    return value


def emit(obj) -> None:
    """Write one compact JSON line and flush, so a reader can stream it."""
    sys.stdout.write(json.dumps(_json_safe(obj), separators=(",", ":"), allow_nan=False))
    sys.stdout.write("\n")
    sys.stdout.flush()


def enum_name(value) -> str:
    return str(getattr(value, "__name__", value))


def license_name(err) -> str:
    rescode = getattr(err, "errno", None)
    return enum_name(rescode) if rescode is not None else "unknown"


def is_license_error(err: BaseException) -> bool:
    if mosek is None or not isinstance(err, mosek.Error):
        return False
    return license_name(err) in LICENSE_RESCODES


# ---------------------------------------------------------------------------
# Problem input
# ---------------------------------------------------------------------------


def _csc(d) -> sp.csc_matrix:
    return sp.csc_matrix(
        (
            np.asarray(d["nzval"], dtype=np.float64),
            np.asarray(d["rowval"], dtype=np.int64),
            np.asarray(d["colptr"], dtype=np.int64),
        ),
        shape=(int(d["m"]), int(d["n"])),
    )


def read_problem_json(path: str):
    with open(path, "r") as stream:
        data = json.load(stream)
    for key in ("P", "q", "A", "b", "cones"):
        if key not in data:
            raise Unsupported(f"fixture {path} has no '{key}' field")
    q = np.asarray(data["q"], dtype=np.float64)
    b = np.asarray(data["b"], dtype=np.float64)
    A = _csc(data["A"])
    P = _csc(data["P"])
    n = q.size
    if q.ndim != 1 or b.ndim != 1:
        raise Unsupported(f"fixture {path} has non-vector q or b")
    if A.shape != (b.size, n):
        raise Unsupported(
            f"fixture {path} A is {A.shape}, expected ({b.size}, {n})"
        )
    if P.shape != (n, n):
        raise Unsupported(f"fixture {path} P is {P.shape}, expected ({n}, {n})")
    if P.nnz != 0:
        raise Unsupported(
            f"fixture {path} has P with {P.nnz} nonzeros; this leg is P = 0 only"
        )
    cones = parse_cones(data["cones"], path)
    cone_rows = sum(cone_length(tag, param) for tag, param in cones)
    if cone_rows != b.size:
        raise Unsupported(
            f"fixture {path} cone rows {cone_rows} != b length {b.size}"
        )
    return q, A, b, cones


def parse_cones(raw, path: str):
    cones = []
    for entry in raw:
        if not isinstance(entry, dict) or len(entry) != 1:
            raise Unsupported(f"fixture {path} has malformed cone entry {entry!r}")
        tag, param = next(iter(entry.items()))
        if tag not in ALLOWED_CONES:
            raise Unsupported(
                f"fixture {path} uses cone {tag}; supported: {', '.join(ALLOWED_CONES)}"
            )
        dim = int(param)
        if dim <= 0:
            raise Unsupported(f"fixture {path} has nonpositive {tag} dimension {dim}")
        cones.append((tag, dim))
    return cones


def cone_length(tag: str, param: int) -> int:
    if tag == "PSDTriangleConeT":
        return param * (param + 1) // 2
    return param


def build_blocks(cones) -> list:
    """Row layout plus the PSD svec row order / dual permutation."""
    blocks = []
    start = 0
    for tag, param in cones:
        length = cone_length(tag, param)
        block = {
            "tag": tag,
            "param": param,
            "length": length,
            "start": start,
        }
        if tag == "PSDTriangleConeT":
            afe_order, cl_to_mk = svec_maps(param)
            block["order"] = [start + cl for cl in afe_order]
            block["perm"] = cl_to_mk
        else:
            block["order"] = list(range(start, start + length))
        blocks.append(block)
        start += length
    return blocks, start


# ---------------------------------------------------------------------------
# PSD svec index algebra
# ---------------------------------------------------------------------------


def clarabel_upper_pairs(k: int):
    """Clarabel svec: upper triangle by columns, index i + j(j-1)/2, i <= j."""
    return [(i, j) for j in range(1, k + 1) for i in range(1, j + 1)]


def mosek_lower_pairs(k: int):
    """MOSEK svec: lower triangle by columns, index over (row, col), row >= col."""
    return [(i, j) for j in range(1, k + 1) for i in range(j, k + 1)]


def svec_maps(k: int):
    """Return ``(afe_order, cl_to_mk)`` for one PSD block of side ``k``.

    ``afe_order[mk]`` is the Clarabel row index that must sit at MOSEK svec
    position ``mk``; ``cl_to_mk[cl]`` is its inverse and maps a returned
    ``getaccdoty`` entry back to Clarabel row order.
    """
    cl_pairs = clarabel_upper_pairs(k)
    mk_index = {pair: idx for idx, pair in enumerate(mosek_lower_pairs(k))}
    cl_index = {pair: idx for idx, pair in enumerate(cl_pairs)}
    afe_order = [None] * len(cl_pairs)
    cl_to_mk = [None] * len(cl_pairs)
    for (i, j) in cl_pairs:
        cl = cl_index[(i, j)]
        mk = mk_index[(j, i)]  # X[i, j] = X[j, i]; MOSEK stores the lower copy
        afe_order[mk] = cl
        cl_to_mk[cl] = mk
    return afe_order, cl_to_mk


def svec_upper(matrix: np.ndarray) -> np.ndarray:
    """Clarabel-style scaled svec of a symmetric matrix."""
    k = matrix.shape[0]
    return np.asarray(
        [
            matrix[i - 1, j - 1] if i == j else math.sqrt(2.0) * matrix[i - 1, j - 1]
            for (i, j) in clarabel_upper_pairs(k)
        ],
        dtype=np.float64,
    )


def svec_lower(matrix: np.ndarray) -> np.ndarray:
    """MOSEK-style scaled svec of a symmetric matrix."""
    k = matrix.shape[0]
    return np.asarray(
        [
            matrix[i - 1, j - 1] if i == j else math.sqrt(2.0) * matrix[i - 1, j - 1]
            for (i, j) in mosek_lower_pairs(k)
        ],
        dtype=np.float64,
    )


def selftest() -> int:
    """Solver-free check of the PSD index algebra used for the ACC mapping."""
    rng = np.random.default_rng(20260912)
    failures = []
    for k in range(1, 9):
        afe_order, cl_to_mk = svec_maps(k)
        dim = k * (k + 1) // 2
        if sorted(afe_order) != list(range(dim)):
            failures.append(f"k={k}: afe_order is not a bijection")
        if sorted(cl_to_mk) != list(range(dim)):
            failures.append(f"k={k}: cl_to_mk is not a bijection")
        for cl, mk in enumerate(cl_to_mk):
            if afe_order[mk] != cl:
                failures.append(f"k={k}: maps are not inverses at cl={cl}")
                break
        m = rng.standard_normal((k, k))
        m = 0.5 * (m + m.T)
        upper = svec_upper(m)
        lower = svec_lower(m)
        mismatch = np.abs(upper - lower[cl_to_mk]).max(initial=0.0)
        if mismatch > 1e-12:
            failures.append(f"k={k}: svec mismatch {mismatch:g}")
    failures.extend(stub_domain_check())
    failures.extend(native_bar_check())
    if failures:
        for line in failures:
            print(line, file=sys.stderr)
        return EXIT_FAIL
    print("selftest: PSD svec permutation consistent for k=1..8; "
          "ACC dimensions and native bar mixed-cone coefficients/point mapping consistent")
    return EXIT_PASS


class _StubTask:
    """Records the API calls build_task makes, so domain dimensions are testable."""

    def __init__(self, env=None):
        self.calls = []
        self.domains = []

    def __getattr__(self, name):
        def record(*args, **kwargs):
            self.calls.append((name, args))
            if name.startswith("append") and name.endswith("domain"):
                self.domains.append((name, args[0]))
            return len(self.domains)
        return record

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def stub_domain_check() -> list:
    """Verify the domain dimension passed for each cone equals its ACC length."""
    if mosek is None:  # pragma: no cover - environment dependent
        return ["mosek unavailable"]
    stub = _StubTask()
    ns = type("ns", (), {})()
    ns.Task = lambda env=None, *a, **k: stub
    ns.iparam = type("i", (), {"num_threads": 0})()
    ns.boundkey = type("b", (), {"fr": 0})()
    ns.objsense = type("o", (), {"minimize": 0})()
    saved = globals()["mosek"]
    globals()["mosek"] = ns
    try:
        q = np.zeros(2)
        A = sp.identity(2, format="csc")
        b = np.zeros(1 + 6 + 1)
        blocks, _ = build_blocks(
            [("ZeroConeT", 1), ("PSDTriangleConeT", 3), ("NonnegativeConeT", 1)]
        )
        task, accs = build_task(None, q, A, b, blocks)
    finally:
        globals()["mosek"] = saved
    expected = [("appendrzerodomain", 1), ("appendsvecpsdconedomain", 6),
                ("appendrplusdomain", 1)]
    observed = [(name, int(dim)) for name, dim in task.domains]
    if observed != expected:
        return [f"domain dimensions {observed} != expected {expected}"]
    if len(accs) != 3:
        return [f"expected 3 ACCs, got {len(accs)}"]
    return []


# ---------------------------------------------------------------------------
# Cone distance (independent original-coordinate oracle)
# ---------------------------------------------------------------------------


def psd_matrix(vector: np.ndarray, k: int) -> np.ndarray:
    matrix = np.zeros((k, k), dtype=np.float64)
    idx = 0
    for (i, j) in clarabel_upper_pairs(k):
        value = float(vector[idx])
        if i == j:
            matrix[i - 1, j - 1] = value
            matrix[j - 1, i - 1] = value
        else:
            scaled = value / math.sqrt(2.0)
            matrix[i - 1, j - 1] = scaled
            matrix[j - 1, i - 1] = scaled
        idx += 1
    return matrix


def cone_distance(v: np.ndarray, blocks, dual: bool = False) -> float:
    if not np.all(np.isfinite(v)):
        return float("nan")
    worst = 0.0
    for block in blocks:
        tag = block["tag"]
        length = block["length"]
        seg = v[block["start"]:block["start"] + length]
        if tag == "ZeroConeT":
            dist = 0.0 if dual else float(np.max(np.abs(seg), initial=0.0))
        elif tag == "NonnegativeConeT":
            dist = max(0.0, -float(np.min(seg, initial=0.0)))
        elif tag == "SecondOrderConeT":
            dist = max(0.0, float(np.linalg.norm(seg[1:])) - float(seg[0]))
        else:
            matrix = psd_matrix(seg, int(block["param"]))
            eig = np.linalg.eigvalsh(matrix)
            dist = max(0.0, -float(eig[0]))
        worst = max(worst, dist)
    return worst


def oracle(x, z, q, A, b, blocks, tol: float, status: str, status_class: str,
           finite_solution: bool) -> dict:
    s = b - A @ x
    r_p_x = cone_distance(s, blocks)
    r_d = float(np.max(np.abs(A.T @ z + q), initial=0.0)) if q.size else 0.0
    dist_k_z = cone_distance(z, blocks, dual=True)
    qx = float(q @ x)
    bz = float(b @ z)
    gap = abs(qx + bz) / (1.0 + abs(qx))

    tol_feas = tol * (1.0 + float(np.max(np.abs(b), initial=0.0)))
    tol_dual = tol * (1.0 + float(np.max(np.abs(q), initial=0.0)))
    finite = bool(finite_solution and np.all(np.isfinite(x)) and np.all(np.isfinite(z)))
    residuals_ok = bool(
        finite
        and r_p_x <= tol_feas
        and r_d <= tol_dual
        and dist_k_z <= tol_dual
        and gap <= tol
    )
    # MOSEK supplies no external certificate object; the strict optimal status
    # is the solver's claim and the residual gate above is the independent check.
    certified = status_class == "optimal"
    return {
        "status": status,
        "status_class": status_class,
        "certified": certified,
        "cert_available": False,
        "finite": finite,
        "r_p_x": r_p_x,
        "dist_K_s": r_p_x,
        "r_d": r_d,
        "dist_Kstar_z": dist_k_z,
        "gap": gap,
        "residuals_ok": residuals_ok,
        "pass": bool(certified and residuals_ok),
        "tol_feas": tol_feas,
        "tol_dual": tol_dual,
        "tol_gap": tol,
        "primal_objective": qx,
        "dual_objective": -bz,
    }


# ---------------------------------------------------------------------------
# MOSEK task construction / solve
# ---------------------------------------------------------------------------


def build_task(env, q, A, b, blocks):
    """Fresh task in Clarabel standard form; returns (task, acc_indices)."""
    n = q.size
    m = b.size
    task = mosek.Task(env)
    task.putintparam(mosek.iparam.num_threads, 1)
    task.appendvars(int(n))
    task.putvarboundsliceconst(0, int(n), mosek.boundkey.fr, 0.0, 0.0)
    task.putobjsense(mosek.objsense.minimize)
    task.putclist(list(range(int(n))), [float(v) for v in q])

    task.appendafes(int(m))
    # s = b - A x, so F = -A and g = b.
    coo = sp.coo_matrix(A)
    if coo.nnz:
        task.putafefentrylist(
            coo.row.astype(np.int64),
            coo.col.astype(np.int32),
            (-coo.data).astype(np.float64),
        )
    task.putafegslice(0, int(m), [float(v) for v in b])

    acc_indices = []
    for block in blocks:
        tag = block["tag"]
        # Domain dimension is the ACC/cone vector length.  For PSD that is the
        # triangular count k(k+1)/2, not the matrix side k.
        length = int(block["length"])
        if tag == "ZeroConeT":
            domidx = task.appendrzerodomain(length)
        elif tag == "NonnegativeConeT":
            domidx = task.appendrplusdomain(length)
        elif tag == "SecondOrderConeT":
            domidx = task.appendquadraticconedomain(length)
        elif tag == "PSDTriangleConeT":
            domidx = task.appendsvecpsdconedomain(length)
        else:  # pragma: no cover - parse_cones already rejects this
            raise Unsupported(f"unsupported cone {tag}")
        task.appendacc(domidx, [int(i) for i in block["order"]], None)
        acc_indices.append(len(acc_indices))
    return task, acc_indices


def doty_to_clarabel(doty_by_block, blocks) -> np.ndarray:
    """Map per-ACC ``getaccdoty`` vectors back to Clarabel row order."""
    total = sum(block["length"] for block in blocks)
    z = np.zeros(total, dtype=np.float64)
    for block, doty in zip(blocks, doty_by_block):
        seg = np.asarray(doty, dtype=np.float64)
        start = block["start"]
        length = block["length"]
        if seg.size != length:
            raise RuntimeError(
                f"ACC dual length {seg.size} != {length} for cone {block['tag']}"
            )
        if block["tag"] == "PSDTriangleConeT":
            for cl, mk in enumerate(block["perm"]):
                z[start + cl] = seg[mk]
        else:
            z[start:start + length] = seg
    return z


def select_formulation(blocks, formulation="auto"):
    supported = all(v["tag"] in {"ZeroConeT", "NonnegativeConeT", "PSDTriangleConeT"}
                    for v in blocks)
    if formulation == "bar" and not supported:
        raise Unsupported("native bar formulation supports PSD, nonnegative and zero cones")
    if formulation == "auto":
        return "bar" if supported and any(v["tag"] == "PSDTriangleConeT" for v in blocks) else "afe"
    if formulation not in {"bar", "afe"}:
        raise ValueError(f"unknown formulation {formulation}")
    return formulation


def build_formulated_task(env, q, A, b, blocks, formulation="auto"):
    """Build either ACC primal or native-bar negative dual, without tuning defaults.

    Native problem: min b'z, A'z=-q, z in K*. Its equality dual is original x;
    its dual slack is original s=b-Ax. Matrix coefficients are smat(A[:,i]),
    hence off-diagonal entries divide by sqrt(2), because trace doubles them.
    https://docs.mosek.com/latest/pythonapi/tutorial-sdo-shared.html
    """
    selected = select_formulation(blocks, formulation)
    if selected == "afe":
        task, accs = build_task(env, q, A, b, blocks)
        return task, {"formulation": "afe", "accs": accs}
    task = mosek.Task(env)
    try:
        task.putintparam(mosek.iparam.num_threads, 1)
        task.putobjsense(mosek.objsense.minimize)
        task.appendcons(int(q.size))
        task.putconboundlist(list(range(q.size)), [mosek.boundkey.fx] * q.size,
                             (-q).tolist(), (-q).tolist())
        scalar_rows, scalar_bounds, bars = [], [], []
        for block in blocks:
            if block["tag"] == "PSDTriangleConeT":
                bars.append(block)
            else:
                scalar_rows.extend(range(block["start"], block["start"] + block["length"]))
                scalar_bounds.extend([mosek.boundkey.fr if block["tag"] == "ZeroConeT"
                                      else mosek.boundkey.lo] * block["length"])
        task.appendvars(len(scalar_rows))
        if scalar_rows:
            ids = list(range(len(scalar_rows)))
            task.putvarboundlist(ids, scalar_bounds, [0.] * len(ids), [0.] * len(ids))
            task.putclist(ids, b[scalar_rows].tolist())
            scalar_A = A[scalar_rows, :].tocoo()
            task.putaijlist(scalar_A.col.tolist(), scalar_A.row.tolist(), scalar_A.data.tolist())
        task.appendbarvars([int(block["param"]) for block in bars])
        cj, ck, cl, cv = [], [], [], []
        ai, aj, ak, al, av = [], [], [], [], []
        for j, block in enumerate(bars):
            start, length = block["start"], block["length"]
            pairs = clarabel_upper_pairs(block["param"])
            lower_row = np.asarray([col - 1 for row, col in pairs], dtype=np.int32)
            lower_col = np.asarray([row - 1 for row, col in pairs], dtype=np.int32)
            scale = np.where(lower_row == lower_col, 1., 1. / math.sqrt(2.))
            objective = b[start:start + length] * scale
            nz = np.flatnonzero(objective)
            if nz.size:
                cj.extend([j] * nz.size)
                ck.extend(lower_row[nz].tolist())
                cl.extend(lower_col[nz].tolist())
                cv.extend(objective[nz].tolist())
            coeff = A[start:start + length, :].tocoo()
            if coeff.nnz:
                ai.extend(coeff.col.tolist())
                aj.extend([j] * coeff.nnz)
                ak.extend(lower_row[coeff.row].tolist())
                al.extend(lower_col[coeff.row].tolist())
                av.extend((coeff.data * scale[coeff.row]).tolist())
        if cv:
            task.putbarcblocktriplet(cj, ck, cl, cv)
        if av:
            task.putbarablocktriplet(ai, aj, ak, al, av)
        return task, {"formulation": "bar", "scalar_rows": scalar_rows, "bars": bars}
    except BaseException:
        task.__exit__(None, None, None)
        raise


def recover_point(task, mapping, blocks, dual_sign=1.0):
    """Return original x,z. Original slack is b-Ax, checked outside timing."""
    sol = mosek.soltype.itr
    if mapping["formulation"] == "afe":
        x = np.asarray(task.getxx(sol), dtype=np.float64)
        z = doty_to_clarabel([task.getaccdoty(sol, acc) for acc in mapping["accs"]], blocks)
        return x, dual_sign * z
    x = np.asarray(task.gety(sol), dtype=np.float64)
    z = np.zeros(sum(block["length"] for block in blocks), dtype=np.float64)
    if mapping["scalar_rows"]:
        z[mapping["scalar_rows"]] = task.getxx(sol)
    for j, block in enumerate(mapping["bars"]):
        raw = np.asarray(task.getbarxj(sol, j), dtype=np.float64)
        scale = np.asarray([1. if row == col else math.sqrt(2.)
                            for row, col in mosek_lower_pairs(block["param"])])
        z[block["start"]:block["start"] + block["length"]] = (raw * scale)[block["perm"]]
    return x, z


def original_status(status, mapping):
    if mapping["formulation"] == "bar":
        return {"prim_infeas_cer": "dual_infeas_cer", "dual_infeas_cer": "prim_infeas_cer",
                "prim_feas": "dual_feas", "dual_feas": "prim_feas"}.get(status, status)
    return status


def native_bar_fixture():
    """Strictly complementary manufactured optimum with multiple PSD blocks."""
    rng = np.random.default_rng(76129)
    blocks, m = build_blocks([("ZeroConeT", 2), ("PSDTriangleConeT", 2),
                             ("NonnegativeConeT", 2), ("PSDTriangleConeT", 3)])
    A = sp.csc_matrix(rng.standard_normal((m, 3)))
    x = np.asarray([0.3, -0.7, 1.1])
    z = np.asarray([0.4, -0.8] + [0.] * (m - 2))
    for block in blocks:
        start, length = block["start"], block["length"]
        if block["tag"] == "PSDTriangleConeT":
            r = rng.standard_normal((block["param"], block["param"]))
            z[start:start + length] = svec_upper(r @ r.T + np.eye(block["param"]))
        elif block["tag"] == "NonnegativeConeT":
            z[start:start + length] = [0.5, 1.5]
    return -np.asarray(A.T @ z), A, np.asarray(A @ x), blocks, x, z


def native_bar_check():
    """No optimizer: reconstruct native trace coefficients and check exact mapping."""
    saved = globals()["mosek"]
    stub = _StubTask()
    ns = type("ns", (), {})()
    ns.Task = lambda env=None: stub
    ns.iparam = type("i", (), {"num_threads": 0})()
    ns.boundkey = type("b", (), {"fr": 0, "lo": 1, "fx": 2})()
    ns.objsense = type("o", (), {"minimize": 0})()
    ns.soltype = type("s", (), {"itr": 0})()
    globals()["mosek"] = ns
    try:
        q, A, b, blocks, expected_x, expected_z = native_bar_fixture()
        task, mapping = build_formulated_task(None, q, A, b, blocks)
        scalar = mapping["scalar_rows"]
        reconstructed_A = np.zeros(A.shape)
        reconstructed_b = np.zeros(b.size)
        bar_by_id = mapping["bars"]
        for name, args in task.calls:
            if name == "putclist":
                for index, value in zip(*args):
                    reconstructed_b[scalar[index]] = value
            elif name == "putaijlist":
                for col, index, value in zip(*args):
                    reconstructed_A[scalar[index], col] = value
            elif name in {"putbarcblocktriplet", "putbarablocktriplet"}:
                for entry in zip(*args):
                    col, j, row, lowercol, value = entry if name == "putbarablocktriplet" else (None, *entry)
                    block = bar_by_id[j]
                    original_row = block["start"] + row * (row + 1) // 2 + lowercol
                    value *= 1. if row == lowercol else math.sqrt(2.)
                    if col is None:
                        reconstructed_b[original_row] += value
                    else:
                        reconstructed_A[original_row, col] += value
        np.testing.assert_allclose(reconstructed_A, A.toarray(), rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(reconstructed_b, b, rtol=1e-14, atol=1e-14)
        bounds = [args for name, args in task.calls if name == "putconboundlist"][0]
        np.testing.assert_allclose(bounds[2], -q, rtol=0, atol=0)
        np.testing.assert_allclose(bounds[3], -q, rtol=0, atol=0)
        assert [args for name, args in task.calls if name == "appendcons"] == [(q.size,)]
        assert not any(name == "appendafes" for name, _ in task.calls)
        task.gety = lambda sol: expected_x
        task.getxx = lambda sol: expected_z[scalar]
        def getbarxj(sol, j):
            block = bar_by_id[j]
            matrix = psd_matrix(expected_z[block["start"]:block["start"] + block["length"]], block["param"])
            return [matrix[row - 1, col - 1] for row, col in mosek_lower_pairs(block["param"])]
        task.getbarxj = getbarxj
        x, z = recover_point(task, mapping, blocks)
        np.testing.assert_allclose(x, expected_x, rtol=0, atol=0)
        np.testing.assert_allclose(z, expected_z, rtol=1e-14, atol=1e-14)
        checked = oracle(x, z, q, A, b, blocks, 1e-12, "optimal", "optimal", True)
        assert checked["pass"], checked
        assert abs(float(q @ x + b @ z)) < 1e-12
        assert original_status("prim_infeas_cer", mapping) == "dual_infeas_cer"
        assert original_status("dual_infeas_cer", mapping) == "prim_infeas_cer"
        soc, _ = build_blocks([("SecondOrderConeT", 3)])
        assert select_formulation(soc) == "afe"
        assert select_formulation(blocks, "afe") == "afe"
        return []
    except Exception as error:
        return [f"native bar mapping: {type(error).__name__}: {error}"]
    finally:
        globals()["mosek"] = saved


def run_once(env, q, A, b, blocks, tol: float, dual_sign: float, formulation="auto") -> dict:
    setup_start = time.perf_counter()
    task, mapping = build_formulated_task(env, q, A, b, blocks, formulation)
    setup_s = time.perf_counter() - setup_start
    try:
        solve_start = time.perf_counter()
        task.optimize()
        solve_s = time.perf_counter() - solve_start

        retrieve_start = time.perf_counter()
        solsta = task.getsolsta(mosek.soltype.itr)
        native_status = enum_name(solsta)
        status = original_status(native_status, mapping)
        status_class = SOLSTA_CLASS.get(status, status)
        try:
            iterations = int(task.getintinf(mosek.iinfitem.intpnt_iter))
        except mosek.Error:
            iterations = 0
        x, z = recover_point(task, mapping, blocks, dual_sign)
        retrieve_s = time.perf_counter() - retrieve_start

        finite_solution = x.size == q.size and np.all(np.isfinite(x))
        record = oracle(
            x, z, q, A, b, blocks, tol, status, status_class, finite_solution
        )
        record.update(
            ok=True,
            error="",
            formulation=mapping["formulation"],
            native_status=native_status,
            setup_s=setup_s,
            solve_s=solve_s,
            e2e_s=setup_s + solve_s,
            iterations=iterations,
            x_norm=float(np.linalg.norm(x)),
            timings={
                "setup_s": setup_s,
                "optimize_s": solve_s,
                "retrieve_s": retrieve_s,
                "total_s": setup_s + solve_s + retrieve_s,
            },
        )
        return record
    finally:
        task.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="MOSEK third leg over the fixed Clarabel JSON fixtures"
    )
    parser.add_argument("files", nargs="*", help="Clarabel JSON problem files")
    parser.add_argument("--runs", type=int, default=4,
                        help="fresh Task setup + solve repetitions per instance")
    parser.add_argument("--tol", type=float, default=1e-6,
                        help="independent oracle tolerance")
    parser.add_argument("--dual-sign", type=float, default=1.0,
                        help="sign applied to the ACC dual before the oracle")
    parser.add_argument("--formulation", choices=["auto", "bar", "afe"], default="auto",
                        help="auto uses native bar negative dual for PSD/NN/Zero; otherwise AFE")
    parser.add_argument("--selftest", action="store_true",
                        help="run the solver-free PSD index self-test and exit")
    args = parser.parse_args(argv)
    if not args.selftest and not args.files:
        parser.error("at least one problem JSON is required (or use --selftest)")
    if args.runs < 1:
        parser.error("--runs must be >= 1")
    if not math.isfinite(args.tol) or args.tol <= 0:
        parser.error("--tol must be finite and > 0")
    return args


def emit_not_run(files, reason: str, detail: str, code: str | None = None) -> None:
    for path in files:
        name = os.path.basename(path)
        if name.endswith(".json"):
            name = name[:-5]
        emit({
            "impl": "MOSEK",
            "instance": name,
            "not_run": True,
            "reason": reason,
            "license_code": code,
            "error": detail,
            "runs_n": 0,
            "pass": False,
            "runs": [],
        })


def default_receipt(env) -> dict:
    with mosek.Task(env) as task:
        return {
            "presolve_use": int(task.getintparam(mosek.iparam.presolve_use)),
            "intpnt_co_tol_pfeas": float(task.getdouparam(mosek.dparam.intpnt_co_tol_pfeas)),
            "intpnt_co_tol_dfeas": float(task.getdouparam(mosek.dparam.intpnt_co_tol_dfeas)),
            "intpnt_co_tol_rel_gap": float(task.getdouparam(mosek.dparam.intpnt_co_tol_rel_gap)),
        }


def main(argv=None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.selftest:
        return selftest()

    if mosek is None:
        emit({
            "impl": "MOSEK",
            "solver": "mosek",
            "not_run": True,
            "reason": "solver_unavailable",
            "error": f"mosek import failed: {MOSEK_IMPORT_ERROR}",
            "runs": args.runs,
            "tol": args.tol,
        })
        emit_not_run(args.files, "solver_unavailable",
                     f"mosek import failed: {MOSEK_IMPORT_ERROR}")
        return EXIT_NO_SOLVER

    try:
        env = mosek.Env()
        defaults = default_receipt(env)
    except mosek.Error as err:
        reason = "license_unavailable" if is_license_error(err) else "solver_error"
        emit({
            "impl": "MOSEK",
            "solver": "mosek",
            "mosek_version": list(mosek.Env.getversion()),
            "python": sys.version.split()[0],
            "not_run": True,
            "reason": reason,
            "license_code": license_name(err) if reason == "license_unavailable" else None,
            "error": f"{enum_name(err.errno)}: {err.msg}",
            "runs": args.runs,
            "tol": args.tol,
        })
        emit_not_run(args.files, reason,
                     f"{enum_name(err.errno)}: {err.msg}",
                     license_name(err) if reason == "license_unavailable" else None)
        return EXIT_LICENSE if reason == "license_unavailable" else EXIT_FAIL

    emit({
        "impl": "MOSEK",
        "mosek_version": list(mosek.Env.getversion()),
        "python": sys.version.split()[0],
        "runs": args.runs,
        "tol": args.tol,
        "dual_sign": args.dual_sign,
        "timing": "fresh Task setup + solve per run (no reuse)",
        "api": "Task native bar negative dual or AFE/ACC primal",
        "formulation": args.formulation,
        "threads": 1,
        "tolerances": "MOSEK product defaults (no tolerance parameter set)",
        "presolve": "MOSEK product default (unchanged)",
        "default_receipt": defaults,
    })

    all_pass = True
    for index, path in enumerate(args.files):
        name = os.path.basename(path)
        if name.endswith(".json"):
            name = name[:-5]
        t0 = time.perf_counter()
        try:
            q, A, b, cones = read_problem_json(path)
            blocks, total_rows = build_blocks(cones)
            if total_rows != b.size:
                raise Unsupported(
                    f"fixture {path} cone rows {total_rows} != b length {b.size}"
                )
            if q.size == 0:
                raise Unsupported(f"fixture {path} has no variables")
        except (Unsupported, ValueError, KeyError, OSError) as err:
            all_pass = False
            emit({
                "impl": "MOSEK",
                "instance": name,
                "n": 0,
                "m": 0,
                "cones": "",
                "runs_n": args.runs,
                "t_json_s": time.perf_counter() - t0,
                "cold_e2e_s": None,
                "warm_e2e_median_s": None,
                "setup_median_s": None,
                "solve_median_s": None,
                "tol_feas": None,
                "tol_dual": None,
                "tol_gap": args.tol,
                "pass": False,
                "error": str(err),
                "runs": [{
                    "ok": False,
                    "error": str(err),
                    "setup_s": None,
                    "solve_s": None,
                    "e2e_s": None,
                    "pass": False,
                }],
            })
            continue

        t_json_s = time.perf_counter() - t0
        cone_str = ",".join(f"{tag}:{param}" for tag, param in cones)
        records = []
        try:
            for _ in range(args.runs):
                try:
                    records.append(run_once(env, q, A, b, blocks, args.tol,
                                            args.dual_sign, args.formulation))
                except mosek.Error as err:
                    if is_license_error(err):
                        raise LicenseUnavailable(f"{enum_name(err.errno)}: {err.msg}")
                    records.append({
                        "ok": False,
                        "error": f"{enum_name(err.errno)}: {err.msg}",
                        "setup_s": None,
                        "solve_s": None,
                        "e2e_s": None,
                        "pass": False,
                    })
                except Exception as err:  # noqa: BLE001 - record, never fake a pass
                    records.append({
                        "ok": False,
                        "error": f"{type(err).__name__}: {err}",
                        "setup_s": None,
                        "solve_s": None,
                        "e2e_s": None,
                        "pass": False,
                    })
        except LicenseUnavailable as err:
            emit({
                "impl": "MOSEK",
                "solver": "mosek",
                "mosek_version": list(mosek.Env.getversion()),
                "not_run": True,
                "reason": "license_unavailable",
                "error": str(err),
                "runs": args.runs,
                "tol": args.tol,
            })
            # Only the current and remaining files are not-run; earlier
            # instances already emitted their real result rows.
            emit_not_run(args.files[index:], "license_unavailable", str(err))
            return EXIT_LICENSE

        ok_records = [rec for rec in records if rec.get("ok")]
        passed = (
            len(ok_records) == args.runs
            and all(rec["pass"] for rec in ok_records)
        )
        all_pass = all_pass and passed

        e2e = [rec["setup_s"] + rec["solve_s"] for rec in ok_records]
        setups = [rec["setup_s"] for rec in ok_records]
        solves = [rec["solve_s"] for rec in ok_records]
        cold = e2e[0] if e2e else None
        warm = median(e2e[1:]) if len(e2e) > 1 else None
        first = ok_records[0] if ok_records else None

        emit({
            "impl": "MOSEK",
            "instance": name,
            "n": int(q.size),
            "m": int(b.size),
            "cones": cone_str,
            "runs_n": args.runs,
            "t_json_s": t_json_s,
            "cold_e2e_s": cold,
            "warm_e2e_median_s": warm,
            "setup_median_s": median(setups) if setups else None,
            "solve_median_s": median(solves) if solves else None,
            "cost_primal": first["primal_objective"] if first else None,
            "tol_feas": first["tol_feas"] if first else None,
            "tol_dual": first["tol_dual"] if first else None,
            "tol_gap": args.tol,
            "pass": passed,
            "runs": records,
        })

    return EXIT_PASS if all_pass else EXIT_FAIL


def median(values) -> float:
    ordered = sorted(float(v) for v in values)
    if not ordered:
        return float("nan")
    n = len(ordered)
    if n % 2:
        return ordered[n // 2]
    return 0.5 * (ordered[n // 2 - 1] + ordered[n // 2])


if __name__ == "__main__":
    sys.exit(main())
