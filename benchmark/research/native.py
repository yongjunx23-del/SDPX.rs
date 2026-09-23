#!/usr/bin/env python3
"""Small native-CLI benchmark helpers.

The research controller deliberately keeps this module independent of the
Julia package.  It understands the portable JSON wire format emitted by the
Rust ``sdpx`` binary, computes the original-coordinate cone checks used by the
old benchmark gate, and creates the two deterministic MPFR diagnostic inputs.
"""
import hashlib
import json
import math
from pathlib import Path


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _number(value):
    """Decode either a Float64 JSON number or an MPFR decimal string.

    The audit intentionally works in host Float64.  Native MPFR samples retain
    their decimal wire values in the raw receipt; a precision-aware external
    oracle can be run separately for those diagnostic points.  Float64 research
    gates, which are the promotion protocol, are fully checked here.
    """
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def _csc(data):
    return int(data["m"]), int(data["n"]), list(map(int, data["colptr"])), \
        list(map(int, data["rowval"])), [_number(x) for x in data["nzval"]]


def _matvec(csc, x):
    m, n, colptr, rowval, nzval = csc
    out = [0.0] * m
    for col in range(n):
        value = x[col] if col < len(x) else math.nan
        for k in range(colptr[col], colptr[col + 1]):
            row = rowval[k]
            if 0 <= row < m:
                out[row] += nzval[k] * value
    return out


def _tmatvec(csc, y):
    _m, n, colptr, rowval, nzval = csc
    out = [0.0] * n
    for col in range(n):
        total = 0.0
        for k in range(colptr[col], colptr[col + 1]):
            row = rowval[k]
            total += nzval[k] * y[row]
        out[col] = total
    return out


def _symmetric_matvec(csc, x):
    """Apply a CSC upper-triangle quadratic matrix as its symmetric operator."""
    _m, n, colptr, rowval, nzval = csc
    out = [0.0] * n
    for col in range(n):
        for k in range(colptr[col], colptr[col + 1]):
            row = rowval[k]
            value = nzval[k]
            if row == col:
                out[col] += value * x[col]
            elif 0 <= row < n:
                out[row] += value * x[col]
                out[col] += value * x[row]
    return out


def _psd_matrix(values, side):
    matrix = [[0.0 for _ in range(side)] for _ in range(side)]
    index = 0
    root2 = math.sqrt(2.0)
    for col in range(side):
        for row in range(col + 1):
            value = values[index]
            index += 1
            if row == col:
                matrix[row][col] = value
            else:
                value /= root2
                matrix[row][col] = value
                matrix[col][row] = value
    return matrix


def _eigmin(matrix):
    try:
        import numpy as np  # optional on small cluster utility nodes
        return float(np.linalg.eigvalsh(np.asarray(matrix, dtype=float))[0])
    except ImportError:
        # The small SDP smoke fixtures only contain 1x1/2x2 blocks.  Keep a
        # deterministic fallback and fail closed for larger blocks.
        n = len(matrix)
        if n == 0:
            return 0.0
        if n == 1:
            return matrix[0][0]
        if n == 2:
            a, b, c = matrix[0][0], matrix[0][1], matrix[1][1]
            return (a + c - math.sqrt((a - c) ** 2 + 4.0 * b * b)) / 2.0
        raise RuntimeError("Float64 cone audit needs numpy for PSD order > 2")


def _cone_distance(values, cones, *, dual=False):
    if any(not math.isfinite(v) for v in values):
        return math.nan
    worst = 0.0
    offset = 0
    for cone in cones:
        if not isinstance(cone, dict) or len(cone) != 1:
            return math.nan
        tag, parameter = next(iter(cone.items()))
        side = int(parameter)
        if tag in ("ZeroConeT", "NonnegativeConeT", "SecondOrderConeT"):
            length = side
        elif tag == "PSDTriangleConeT":
            length = side * (side + 1) // 2
        else:
            return math.nan
        block = values[offset:offset + length]
        if len(block) != length:
            return math.nan
        if tag == "ZeroConeT":
            distance = 0.0 if dual else max((abs(v) for v in block), default=0.0)
        elif tag == "NonnegativeConeT":
            distance = max(0.0, -min(block, default=0.0))
        elif tag == "SecondOrderConeT":
            distance = max(0.0, math.sqrt(sum(v * v for v in block[1:])) - block[0])
        else:
            distance = max(0.0, -_eigmin(_psd_matrix(block, side)))
        worst = max(worst, distance)
        offset += length
    return worst


def _sampled_matrix(problem):
    """Materialize sampled factors for the independent audit.

    This is used only after a native solve and never supplied to the solver;
    the factor-authoritative ``sampled`` field remains the numerical input.
    """
    m, n, _colptr, _rowval, _nzval = _csc(problem["A"])
    result = [[0.0] * n for _ in range(m)]
    for block in problem.get("sampled", []):
        start_row = int(block["row_start"])
        start_col = int(block["column_start"])
        dim = int(block["dim"])
        rows = int(block["basis_rows"])
        cols = int(block["basis_cols"])
        basis = [_number(v) for v in block["basis"]]
        weights = [_number(v) for v in block["weights"]]
        expected_basis = rows * cols
        if len(basis) != expected_basis:
            raise ValueError("sampled basis size mismatch")
        index = 0
        for s in range(dim):
            for r in range(s + 1):
                for k in range(cols):
                    column = start_col + index
                    weight = weights[index]
                    index += 1
                    if not (0 <= column < n):
                        raise ValueError("sampled column outside input")
                    if s == r:
                        # The q q' block is symmetric.  Visit only its upper
                        # triangle; visiting both (a,b) and (b,a) would double
                        # every off-diagonal svec entry.
                        for a in range(rows):
                            qa = basis[a + rows * k]
                            for b in range(a, rows):
                                qb = basis[b + rows * k]
                                i, j = s * rows + a, r * rows + b
                                coefficient = weight * qa * qb * (1.0 if a == b else math.sqrt(2.0))
                                row = start_row + j * (j + 1) // 2 + i
                                result[row][column] += coefficient
                    else:
                        # For s>r, sym(e_r e_s') contributes one half to the
                        # two transposed blocks.  The row ordering means this
                        # orientation already visits each svec row once.
                        for a in range(rows):
                            qa = basis[a + rows * k]
                            for b in range(rows):
                                qb = basis[b + rows * k]
                                i, j = s * rows + a, r * rows + b
                                lo, hi = min(i, j), max(i, j)
                                coefficient = weight * 0.5 * qa * qb * math.sqrt(2.0)
                                row = start_row + hi * (hi + 1) // 2 + lo
                                result[row][column] += coefficient
        # ``result`` stores rows as lists; continue with the next factor block.
    return result


def materialize(problem):
    """Return a dense Float64 view of the ordinary plus sampled operator."""
    m, n, colptr, rowval, nzval = _csc(problem["A"])
    rows = [[0.0] * n for _ in range(m)]
    for col in range(n):
        for k in range(colptr[col], colptr[col + 1]):
            rows[rowval[k]][col] += nzval[k]
    if problem.get("sampled"):
        sampled = _sampled_matrix(problem)
        for i in range(m):
            for j in range(n):
                rows[i][j] += sampled[i][j]
    return rows


def _dense_apply(rows, vector):
    return [sum(value * vector[j] for j, value in enumerate(row)) for row in rows]


def _dense_transpose_apply(rows, vector):
    if not rows:
        return []
    out = [0.0] * len(rows[0])
    for i, row in enumerate(rows):
        for j, value in enumerate(row):
            out[j] += value * vector[i]
    return out


def audit(problem, result, tolerance=1e-6):
    """Independent original-coordinate gate for a native result object."""
    q = [_number(v) for v in problem.get("q", [])]
    b = [_number(v) for v in problem.get("b", [])]
    x = [_number(v) for v in result.get("x", [])]
    z = [_number(v) for v in result.get("z", [])]
    returned_s = [_number(v) for v in result.get("s", [])]
    try:
        A_csc = _csc(problem["A"])
        P = _csc(problem.get("P", {"m": len(q), "n": len(q), "colptr": [0] * (len(q) + 1),
                                     "rowval": [], "nzval": []}))
    except (KeyError, TypeError, ValueError, IndexError):
        A_csc = None
        P = None
    sampled_rows = None
    if problem.get("sampled"):
        try:
            sampled_rows = _sampled_matrix(problem)
        except (KeyError, TypeError, ValueError, IndexError):
            sampled_rows = None
    finite = all(math.isfinite(v) for vector in (q, b, x, z, returned_s) for v in vector)
    if A_csc is None or (problem.get("sampled") and sampled_rows is None) \
            or len(x) != len(q) or len(z) != len(b) or len(returned_s) != len(b):
        finite = False
    if not finite:
        return {"pass": False, "finite": False, "status": str(result.get("status", "Error"))}
    ax = _matvec(A_csc, x)
    atz = _tmatvec(A_csc, z)
    if sampled_rows is not None:
        sampled_ax = _dense_apply(sampled_rows, x)
        sampled_atz = _dense_transpose_apply(sampled_rows, z)
        ax = [left + right for left, right in zip(ax, sampled_ax)]
        atz = [left + right for left, right in zip(atz, sampled_atz)]
    p_x = _symmetric_matvec(P, x) if P is not None else [0.0] * len(x)
    slack = [b[i] - ax[i] for i in range(len(b))]
    rp = max((abs(returned_s[i] - slack[i]) for i in range(len(b))), default=0.0)
    rp_x = _cone_distance(slack, problem.get("cones", []))
    dist_s = _cone_distance(returned_s, problem.get("cones", []))
    rd = max((abs(p_x[j] + atz[j] + q[j]) for j in range(len(q))), default=0.0)
    dist_z = _cone_distance(z, problem.get("cones", []), dual=True)
    p_xx = sum(x[i] * p_x[i] for i in range(len(x)))
    q_linear = sum(q[i] * x[i] for i in range(len(q)))
    primal_objective = q_linear + 0.5 * p_xx
    bz = sum(b[i] * z[i] for i in range(len(b)))
    # The native receipt's dual objective is -b'z - 1/2*x'Px.  Therefore the
    # primal/dual gap is x'Px + q'x + b'z; deriving it from the two objective
    # expressions keeps the P contribution from being counted incorrectly.
    dual_objective = -bz - 0.5 * p_xx
    gap = abs(primal_objective - dual_objective) / (1.0 + abs(primal_objective))
    tol_feas = tolerance * (1.0 + max((abs(v) for v in b), default=0.0))
    tol_dual = tolerance * (1.0 + max((abs(v) for v in q), default=0.0))
    residuals = (rp, dist_s, rp_x, rd, dist_z, gap)
    solver_optimal = str(result.get("status", "")) in ("Solved", "Optimal", "optimal", "solved")
    passed = solver_optimal and all(math.isfinite(v) and v <= tol_feas for v in (rp, dist_s, rp_x)) \
        and all(math.isfinite(v) and v <= tol_dual for v in (rd, dist_z)) \
        and math.isfinite(gap) and gap <= tolerance
    return {
        "pass": bool(passed), "finite": True, "status": str(result.get("status", "Error")),
        "solver_optimal": solver_optimal, "r_p": rp, "dist_K_s": dist_s,
        "r_p_x": rp_x, "r_d": rd, "dist_Kstar_z": dist_z, "gap": gap,
        "tol_feas": tol_feas, "tol_dual": tol_dual,
        "primal_objective": _number(result.get("objective")),
        "dual_objective": _number(result.get("dual_objective")),
        "solver_primal_affine_residual": _number(result.get("primal_residual")),
        "solver_dual_affine_residual": _number(result.get("dual_residual")),
        "residuals": list(residuals),
    }


def _decimal(value, bits):
    if bits == 53:
        return float(value)
    # Decimal's precision is decimal digits; retain a generous margin over the
    # binary target while keeping generated files compact.
    from decimal import Decimal, localcontext
    with localcontext() as context:
        context.prec = max(32, int(bits * 0.302 + 12))
        # Decimal(float) exposes the binary approximation.  ``repr`` is the
        # shortest round-trippable decimal and is sufficient for callers that
        # already provide a scalar rather than an exact rational.
        if isinstance(value, float):
            value = repr(value)
        return format(Decimal(value), ".%dg" % context.prec)


def _decimal_fraction(numerator, denominator, bits):
    """Render an exact rational with enough decimal digits for ``bits``."""
    if bits == 53:
        return numerator / denominator
    from decimal import Decimal, localcontext
    with localcontext() as context:
        context.prec = max(32, int(bits * 0.302 + 12))
        value = Decimal(numerator) / Decimal(denominator)
        return format(value, ".%dg" % context.prec)


def _csc_zero(m, n, bits):
    zero = _decimal(0, bits)
    return {"m": m, "n": n, "colptr": [0] * (n + 1), "rowval": [], "nzval": []}


def synthetic_problem(entry, bits):
    """Build one of the fixed research MPFR diagnostics as native JSON."""
    runner = entry.get("runner")
    parameters = entry.get("parameters", {})
    if runner == "orthant":
        n = int(parameters["n"])
        repeats = int(parameters["rows_per_variable"])
        m = n * repeats
        rowval, nzval, colptr = [], [], [0]
        for col in range(n):
            for k in range(repeats):
                rowval.append(col * repeats + k)
                nzval.append(_decimal(-1, bits))
            colptr.append(len(rowval))
        return {
            "P": _csc_zero(n, n, bits),
            "q": [_decimal(1, bits)] * n,
            "A": {"m": m, "n": n, "colptr": colptr, "rowval": rowval, "nzval": nzval},
            "b": [_decimal_fraction(-(k + 1), repeats, bits)
                  for _j in range(n) for k in range(repeats)],
            "cones": [{"NonnegativeConeT": m}],
            "settings": {"verbose": False, "max_threads": 1, "presolve_enable": False,
                         "kkt_form": "augmented"},
        }
    if runner == "sampled":
        n = int(parameters["n"])
        m = n * (n + 1) // 2
        q = [_decimal(1, bits)] * n
        basis = []
        rho = _decimal_fraction(1, 4 * n, bits)
        diagonal = _decimal_fraction(4 * n + 1, 4 * n, bits)
        for k in range(n):
            for i in range(n):
                basis.append(diagonal if i == k else rho)
        # The factor operator uses one primitive column per sampled vector.
        sampled = [{"row_start": 0, "column_start": 0, "dim": 1,
                    "basis_rows": n, "basis_cols": n, "basis": basis,
                    "weights": [_decimal(-1, bits)] * n}]
        b = []
        for col in range(n):
            for row in range(col + 1):
                b.append(_decimal(-1 if row == col else 0, bits))
        return {
            "P": _csc_zero(n, n, bits), "q": q,
            "A": _csc_zero(m, n, bits), "b": b,
            "cones": [{"PSDTriangleConeT": n}], "sampled": sampled,
            "settings": {"verbose": False, "max_threads": 1, "kkt_form": "condensed"},
        }
    raise ValueError("unsupported native synthetic runner: " + str(runner))


def wire_input_hash(problem):
    return hashlib.sha256(
        (json.dumps(problem, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    ).hexdigest()
