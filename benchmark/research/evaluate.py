#!/usr/bin/env python3
"""Evaluate predeclared, paired campaigns. No solver execution or statistical claims.

Schema 1: identity/source_sha256, artifact_sha256, harness_sha256,
catalog_sha256, host_id, environment_sha256; identity_unchanged=true;
qualification_passed reports external qualification for the exact identity;
contract with precision_bits/threads/blas_threads and declared tolerance/timing
identifiers; expected_cases with id/family/group; records with case_id/family,
input_sha256/settings_sha256, precision_bits/threads, phase, repetition,
order_block (AB/BA), passed/status, time_s/rss_bytes.
Each case needs one cold and at least three warm samples in EACH order block.
Failed samples remain required; their time/RSS may be null. Passing samples
need positive finite time and RSS. Repetition keys must match between arms.
"""
import argparse
import json
import math
import statistics
from pathlib import Path

HASH_FIELDS = ("source_sha256", "artifact_sha256", "harness_sha256",
               "catalog_sha256", "environment_sha256")
MATCH_IDENTITIES = ("harness_sha256", "catalog_sha256", "environment_sha256", "host_id")
ORDERS = ("AB", "BA")
STATUSES = ("Solved", "Optimal")


def _positive(value, integer=False):
    try:
        return (type(value) is int if integer else type(value) in (int, float)) and value > 0 and math.isfinite(value)
    except OverflowError:
        return False


def _hash(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _validate(data, arm, screen=False):
    errors, cases, rows = [], {}, {}
    def bad(message):
        errors.append(f"{arm}: {message}")
    if not isinstance(data, dict):
        return [f"{arm}: campaign must be an object"], cases, rows
    if type(data.get("schema_version")) is not int or data["schema_version"] != 1:
        bad("schema_version must be 1")
    if data.get("identity_unchanged") is not True:
        bad("source/artifact identity was not confirmed unchanged")
    if type(data.get("qualification_passed")) is not bool:
        bad("qualification_passed must explicitly report external qualification")
    identity = data.get("identity", {})
    if not isinstance(identity, dict):
        identity = {}
    for name in HASH_FIELDS:
        if not _hash(identity.get(name)):
            bad(f"invalid identity.{name}")
    if not isinstance(identity.get("host_id"), str) or not identity["host_id"].strip():
        bad("missing host_id")
    contract = data.get("contract")
    if not isinstance(contract, dict):
        contract = {}
        bad("missing contract")
    for name in ("precision_bits", "threads", "blas_threads"):
        if not _positive(contract.get(name), integer=True):
            bad(f"invalid contract.{name}")
    for name in ("tolerance_id", "timing_scope"):
        if not isinstance(contract.get(name), str) or not contract[name].strip():
            bad(f"missing contract.{name}")
    try:
        json.dumps(contract, allow_nan=False)
    except (ValueError, TypeError):
        bad("contract is not finite JSON")
    expected = data.get("expected_cases")
    if not isinstance(expected, list) or not expected:
        bad("expected_cases must be a nonempty list")
        expected = []
    for case in expected:
        if not isinstance(case, dict) or any(not isinstance(case.get(k), str) or not case[k].strip() for k in ("id", "family", "group")):
            bad("invalid expected case")
            continue
        if case["id"] in cases:
            bad(f"duplicate expected case {case['id']}")
        cases[case["id"]] = case
    records = data.get("records")
    if not isinstance(records, list):
        bad("records must be a list")
        records = []
    for row in records:
        if not isinstance(row, dict):
            bad("record must be an object")
            continue
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or case_id not in cases:
            bad("unknown case in records")
            continue
        if row.get("family") != cases[case_id]["family"]:
            bad(f"{case_id}: family mismatch")
        if row.get("phase") not in ("cold", "warm") or row.get("order_block") not in ORDERS:
            bad(f"{case_id}: invalid phase/order_block")
            continue
        repetition = row.get("repetition")
        if type(repetition) is not int or repetition < 0:
            bad(f"{case_id}: invalid repetition")
            continue
        key = (case_id, row["order_block"], row["phase"], repetition)
        if key in rows:
            bad(f"duplicate record {key}")
        rows[key] = row
        for name in ("input_sha256", "settings_sha256"):
            if not _hash(row.get(name)):
                bad(f"{case_id}: invalid {name}")
        for name in ("precision_bits", "threads"):
            if type(row.get(name)) is not int or row[name] != contract.get(name):
                bad(f"{case_id}: {name} differs from contract")
        if type(row.get("passed")) is not bool or not isinstance(row.get("status"), str):
            bad(f"{case_id}: missing pass/status evidence")
        if row.get("passed") is True and row.get("status") not in STATUSES:
            bad(f"{case_id}: passed sample has non-optimal status")
        if row.get("passed") is True and (row.get("error") or row.get("incomplete")):
            bad(f"{case_id}: passed sample reports an error or incomplete execution")
        for name, integer in (("time_s", False), ("rss_bytes", True)):
            value = row.get(name)
            if value is None and row.get("passed") is False:
                continue
            if not _positive(value, integer):
                bad(f"{case_id}: invalid or missing {name}")
    for case_id in cases:
        hashes = {(r.get("input_sha256"), r.get("settings_sha256")) for k, r in rows.items()
                  if k[0] == case_id and _hash(r.get("input_sha256")) and _hash(r.get("settings_sha256"))}
        if len(hashes) > 1:
            bad(f"{case_id}: input/settings vary within campaign")
        for order in (("AB",) if screen else ORDERS):
            group = [k for k in rows if k[:2] == (case_id, order)]
            if screen:
                if set(group) != {(case_id, order, "cold", 0), (case_id, order, "warm", 1)}:
                    bad(f"{case_id}/{order}: screen requires exactly cold 0 and warm 1")
                if any(k[0] == case_id and k[1] != "AB" for k in rows):
                    bad(f"{case_id}: screen permits only AB")
                continue
            if sum(k[2] == "cold" for k in group) != 1 or sum(k[2] == "warm" for k in group) < 3:
                bad(f"{case_id}/{order}: require one cold and at least three warm samples")
    return errors, cases, rows


def _coverage(cases, rows, screen=False):
    passing = [case for case in cases if any(k[0] == case for k in rows) and
               all(sum(k[:3] == (case, order, "cold") for k in rows) == 1 and
                   sum(k[:3] == (case, order, "warm") for k in rows) >= (1 if screen else 3) for order in (("AB",) if screen else ORDERS)) and
               all(r.get("passed") is True for k, r in rows.items() if k[0] == case)]
    return {"required_cases": len(cases), "recorded_samples": len(rows),
            "passed_samples": sum(r.get("passed") is True for r in rows.values()),
            "fully_passing_cases": passing}


def _geomean(values):
    return math.exp(statistics.mean(math.log(v) for v in values))


def _median(values):
    values = sorted(values)
    middle = len(values) // 2
    if len(values) % 2:
        return values[middle]
    low, high = values[middle - 1:middle + 1]
    return low + (high - low) / 2


def _evaluate_full(baseline, candidate, *, min_speedup=1.02, max_family_ratio=1.02,
             max_case_ratio=1.10, max_memory_ratio=1.10):
    """Ratios are candidate/baseline; speedup is its reciprocal.

    Equal-weight families contain equal-weight cases. Each order block must
    independently pass speed, regression and memory limits. No failed sample
    is removed to make a speed comparison possible.
    """
    policy = dict(min_speedup=min_speedup, max_family_ratio=max_family_ratio,
                  max_case_ratio=max_case_ratio, max_memory_ratio=max_memory_ratio)
    result = {"schema_version": 1, "verdict": "incomplete", "reasons": [], "policy": policy,
              "assumptions": ["fixed complete case catalog; no pass-only filtering",
                  "one cold plus at least three warm samples per case and order",
                  "equal family weights, equal case weights within each family",
                  "each AB/BA block independently clears all limits",
                  "RSS ratio uses maximum across cold and warm samples per case",
                  "descriptive paired comparisons, not statistical significance"]}
    if any(not _positive(v) for v in policy.values()) or min_speedup < 1.02:
        result["reasons"] = ["policy limits must be positive finite numbers; minimum speedup must be >= 1.02"]
        result["policy"] = None
        return result
    ea, ca, ra = _validate(baseline, "baseline")
    eb, cb, rb = _validate(candidate, "candidate")
    result["coverage"] = {"baseline": _coverage(ca, ra), "candidate": _coverage(cb, rb)}
    result["reasons"] = ea + eb
    if ea or eb:
        return result
    if ca != cb:
        result["reasons"].append("expected case catalogs differ")
    for name in MATCH_IDENTITIES:
        if baseline["identity"][name] != candidate["identity"][name]:
            result["reasons"].append(f"comparison identity mismatch: {name}")
    if baseline["contract"] != candidate["contract"]:
        result["reasons"].append("contracts differ")
    if ra.keys() != rb.keys():
        result["reasons"].append("sample repetition keys differ")
    for key in ra.keys() & rb.keys():
        if any(ra[key][f] != rb[key][f] for f in ("input_sha256", "settings_sha256", "precision_bits", "threads")):
            result["reasons"].append(f"sample contract mismatch: {key}")
    if result["reasons"]:
        return result
    if not baseline["qualification_passed"] or not candidate["qualification_passed"]:
        improved = (not baseline["qualification_passed"] and candidate["qualification_passed"] and
                    all(r["passed"] for r in rb.values()))
        result.update(verdict="correctness_only" if improved else "discard",
                      reasons=["both arms require matching external qualification for speed promotion"])
        return result
    bad_a = {k for k, r in ra.items() if not r["passed"]}
    bad_b = {k for k, r in rb.items() if not r["passed"]}
    if bad_a or bad_b:
        if bad_b < bad_a:
            result.update(verdict="correctness_only", reasons=["strictly improved required-sample acceptance; speed promotion unavailable"])
        else:
            result.update(verdict="discard", reasons=["failed required numerical gates; no speed comparison"])
        return result
    if all(baseline["identity"][name] == candidate["identity"][name] for name in ("source_sha256", "artifact_sha256")):
        result.update(verdict="discard", reasons=["identical source and artifact; timing noise is not a candidate improvement"])
        return result
    comparisons = {}
    for order in (*ORDERS, "combined"):
        per_case = {}
        for case_id in ca:
            keys = [k for k in ra if k[0] == case_id and (order == "combined" or k[1] == order)]
            warm = [k for k in keys if k[2] == "warm"]
            a = _median(ra[k]["time_s"] for k in warm)
            b = _median(rb[k]["time_s"] for k in warm)
            memory = max(rb[k]["rss_bytes"] for k in keys) / max(ra[k]["rss_bytes"] for k in keys)
            if not _positive(b / a) or not _positive(memory):
                result["reasons"] = ["comparison ratio outside finite representable range"]
                return result
            per_case[case_id] = {"family": ca[case_id]["family"], "baseline_median_s": a,
                                 "candidate_median_s": b, "ratio": b / a, "memory_ratio": memory}
        families = {}
        for family in sorted({c["family"] for c in ca.values()}):
            families[family] = _geomean([v["ratio"] for v in per_case.values() if v["family"] == family])
        ratio = _geomean(list(families.values()))
        if not _positive(ratio) or not _positive(1 / ratio):
            result["reasons"] = ["aggregate speedup outside finite representable range"]
            return result
        comparisons[order] = {"cases": per_case, "family_ratios": families,
                              "family_balanced_ratio": ratio, "speedup": 1 / ratio}
        if 1 / ratio < min_speedup:
            result["reasons"].append(f"{order}: aggregate speedup below {min_speedup}")
        for family, value in families.items():
            if value > max_family_ratio:
                result["reasons"].append(f"{order}: family regression exceeds limit: {family}")
        for case_id, value in per_case.items():
            if value["ratio"] > max_case_ratio:
                result["reasons"].append(f"{order}: case regression exceeds limit: {case_id}")
            if value["memory_ratio"] > max_memory_ratio:
                result["reasons"].append(f"{order}: memory regression exceeds limit: {case_id}")
    result["comparisons"] = comparisons
    result["verdict"] = "discard" if result["reasons"] else "keep"
    return result


def _reduced(data):
    contract = data.get('contract', {}) if isinstance(data, dict) else {}
    return isinstance(contract, dict) and (contract.get('profile') == 'screen'
        or contract.get('repetitions', 4) != 4
        or contract.get('blocks', list(ORDERS)) != list(ORDERS)
        or contract.get('speed_credit') is False)


def evaluate(baseline, candidate, *, min_speedup=1.02, max_family_ratio=1.02,
             max_case_ratio=1.10, max_memory_ratio=1.10):
    if not (_reduced(baseline) or _reduced(candidate)):
        return _evaluate_full(baseline, candidate, min_speedup=min_speedup,
            max_family_ratio=max_family_ratio, max_case_ratio=max_case_ratio,
            max_memory_ratio=max_memory_ratio)
    reason = 'reduced screen protocol: one cold + one warm, AB only; speed_credit: false'
    result = dict(schema_version=1, verdict='screen_fail', speed_credit=False,
                  reasons=[reason], policy={'timing_gate': False, 'timing_warning_ratio': max_family_ratio},
                  timing_observations=[])
    errors = []
    if not _positive(max_family_ratio):
        result['policy'] = None
        errors.append('invalid max_family_ratio')
    ea, ca, ra = _validate(baseline, 'baseline', screen=True)
    eb, cb, rb = _validate(candidate, 'candidate', screen=True)
    errors.extend(ea + eb)
    result['coverage'] = {'baseline': _coverage(ca, ra, True), 'candidate': _coverage(cb, rb, True)}
    for data in (baseline, candidate):
        contract = data.get('contract', {}) if isinstance(data, dict) else {}
        if not isinstance(contract, dict) or any(contract.get(k) != v for k, v in
                dict(profile='screen', repetitions=2, blocks=['AB'], speed_credit=False,
                     stage='development', threads=1).items()):
            errors.append('screen contract must declare the fixed development/1-thread/2-repetition/AB protocol')
    if errors:
        result['reasons'] += errors
        return result
    if ca != cb:
        errors.append('expected case catalogs differ')
    for name in MATCH_IDENTITIES:
        if baseline['identity'][name] != candidate['identity'][name]:
            errors.append(f'comparison identity mismatch: {name}')
    if baseline['contract'] != candidate['contract']:
        errors.append('contracts differ')
    if ra.keys() != rb.keys():
        errors.append('sample repetition keys differ')
    for key in ra.keys() & rb.keys():
        if any(ra[key][f] != rb[key][f] for f in ('input_sha256', 'settings_sha256', 'precision_bits', 'threads')):
            errors.append(f'sample contract mismatch: {key}')
    if any(r['passed'] is not True for r in (*ra.values(), *rb.values())):
        errors.append('failed required numerical gates')
    if errors:
        result['reasons'] += errors
        return result
    comparisons = {}
    for case_id in ca:
        keys = [k for k in ra if k[0] == case_id and k[2] == 'warm']
        a, b = (_median(rows[k]['time_s'] for k in keys) for rows in (ra, rb))
        ratio = b / a
        if not _positive(ratio):
            errors.append(f'{case_id}: comparison ratio outside finite representable range')
            continue
        comparisons[case_id] = dict(baseline_median_s=a, candidate_median_s=b, ratio=ratio)
        if ratio > max_family_ratio:
            result['timing_observations'].append(
                f'{case_id}: ratio {ratio:.3g}; repeat in a timed campaign before deciding')
    result['comparisons'] = comparisons
    result['reasons'] += errors
    result['verdict'] = 'screen_fail' if errors else 'screen_pass'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-speedup", type=float, default=1.02)
    parser.add_argument("--max-family-ratio", type=float, default=1.02)
    parser.add_argument("--max-case-ratio", type=float, default=1.10)
    parser.add_argument("--max-memory-ratio", type=float, default=1.10)
    args = parser.parse_args()
    try:
        result = evaluate(json.loads(args.baseline.read_text()), json.loads(args.candidate.read_text()),
                          min_speedup=args.min_speedup, max_family_ratio=args.max_family_ratio,
                          max_case_ratio=args.max_case_ratio, max_memory_ratio=args.max_memory_ratio)
    except (OSError, ValueError, TypeError, OverflowError) as exc:
        result = {"schema_version": 1, "verdict": "incomplete", "reasons": [str(exc)]}
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return 0 if result["verdict"] == "keep" else 1


if __name__ == "__main__":
    raise SystemExit(main())
