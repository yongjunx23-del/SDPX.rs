"""Adversarial normalized-campaign checks; no solver or benchmark execution."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location("research_evaluate", Path(__file__).resolve().parents[1] / "evaluate.py")
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)
evaluate = module.evaluate


def campaign(candidate=False, families=("LP", "SOCP")):
    out = {"schema_version": 1, "identity_unchanged": True, "qualification_passed": True,
           "identity": {k: "a" * 64 for k in module.HASH_FIELDS},
           "contract": {"precision_bits": 256, "threads": 4, "blas_threads": 1,
                        "tolerance_id": "fixed-256-v1", "timing_scope": "native solve"},
           "expected_cases": [], "records": []}
    out["identity"]["host_id"] = "host-1"
    if candidate:
        out["identity"]["source_sha256"] = "b" * 64
        out["identity"]["artifact_sha256"] = "c" * 64
    for i, family in enumerate(families):
        name = f"case-{i}"
        out["expected_cases"].append({"id": name, "family": family, "group": "training"})
        for order in module.ORDERS:
            for repetition in range(4):
                out["records"].append({"case_id": name, "family": family,
                    "input_sha256": "d" * 64, "settings_sha256": "e" * 64,
                    "precision_bits": 256, "threads": 4,
                    "phase": "cold" if repetition == 0 else "warm", "repetition": repetition,
                    "order_block": order, "passed": True, "status": "Solved",
                    "time_s": 100 if repetition == 0 else (0.9 if candidate else 1.0),
                    "rss_bytes": 1000})
    return out


class EvaluateTests(unittest.TestCase):
    def pair(self):
        return campaign(), campaign(True)

    def screen_pair(self):
        a, b = self.pair()
        for data in (a,b):
            data['contract'].update(profile='screen', repetitions=2, blocks=['AB'],
                                    speed_credit=False, stage='development', threads=1)
            data['records'] = [r for r in data['records'] if r['order_block']=='AB' and r['repetition']<2]
            for row in data['records']:
                row['threads'] = 1
        return a,b

    def test_screen_never_earns_speed_credit(self):
        a,b = self.screen_pair()
        for row in b['records']:
            row['time_s'] = .001
        result = evaluate(a,b)
        self.assertEqual(result['verdict'], 'screen_pass')
        self.assertFalse(result['speed_credit'])
        self.assertIn('reduced', result['reasons'][0])
        for field, value in [('repetitions',3), ('blocks',['BA']), ('profile','full')]:
            changed = copy.deepcopy(b)
            changed['contract'][field] = value
            self.assertEqual(evaluate(a,changed)['verdict'], 'screen_fail')
        a,b = self.pair()
        for data in (a,b):
            data['contract']['repetitions'] = 2
        self.assertEqual(evaluate(a,b)['verdict'], 'screen_fail')

    def test_screen_regression_and_missing_failed_or_changed_samples(self):
        a,b = self.screen_pair()
        b['records'][1]['time_s'] = 1.021
        self.assertEqual(evaluate(a,b)['verdict'], 'screen_fail')
        for mutate in (lambda c: c['records'].pop(),
                       lambda c: c['records'][1].update(passed=False),
                       lambda c: c['records'][1].update(repetition=99),
                       lambda c: c['identity'].update(catalog_sha256='f'*64)):
            a,b = self.screen_pair()
            mutate(b)
            self.assertEqual(evaluate(a,b)['verdict'], 'screen_fail')

    def test_qualified_balanced_speedup(self):
        a, b = self.pair()
        r = evaluate(a, b)
        self.assertEqual(r["verdict"], "keep")
        self.assertAlmostEqual(r["comparisons"]["combined"]["family_balanced_ratio"], .9)
        self.assertEqual(len(r["coverage"]["candidate"]["fully_passing_cases"]), 2)

    def test_same_artifact_and_source_cannot_win_from_noise(self):
        a, b = self.pair()
        b["identity"] = copy.deepcopy(a["identity"])
        self.assertEqual(evaluate(a, b)["verdict"], "discard")

    def test_missing_duplicate_unknown_and_sample_key_mismatch(self):
        a, b = self.pair()
        for mutation in (lambda c: c["records"].pop(),
                         lambda c: c["records"].append(copy.deepcopy(c["records"][0])),
                         lambda c: c["records"][0].update(case_id="other"),
                         lambda c: c["records"][1].update(repetition=99)):
            with self.subTest(mutation=mutation):
                c = copy.deepcopy(b)
                mutation(c)
                self.assertEqual(evaluate(a, c)["verdict"], "incomplete")

    def test_failed_case_not_dropped_and_correctness_only(self):
        a, b = self.pair()
        a["records"][1].update(passed=False, status="MaxIterations", time_s=None, rss_bytes=None)
        r = evaluate(a, b)
        self.assertEqual(r["verdict"], "correctness_only")
        self.assertNotIn("comparisons", r)
        b["records"][1].update(passed=False, status="MaxIterations", time_s=None, rss_bytes=None)
        self.assertEqual(evaluate(a, b)["verdict"], "discard")
        # Moving failure to another required point is not improved coverage.
        b["records"][1].update(passed=True, status="Solved", time_s=.9, rss_bytes=1000)
        b["records"][2].update(passed=False, status="NumericalError")
        self.assertEqual(evaluate(a, b)["verdict"], "discard")

    def test_missing_baseline_case_is_not_correctness_credit(self):
        a, b = self.pair()
        a["records"] = a["records"][:8]
        r = evaluate(a, b)
        self.assertEqual(r["verdict"], "incomplete")
        self.assertEqual(len(r["coverage"]["baseline"]["fully_passing_cases"]), 1)

    def test_nonfinite_zero_negative_and_boolean_measurements(self):
        a, b = self.pair()
        for value in (float("nan"), float("inf"), -1, 0, True, "0.9", None):
            for field in ("time_s", "rss_bytes"):
                with self.subTest(value=value, field=field):
                    c = copy.deepcopy(b)
                    c["records"][1][field] = value
                    self.assertEqual(evaluate(a, c)["verdict"], "incomplete")

    def test_identity_contract_precision_settings_and_input_mismatches(self):
        a, b = self.pair()
        changes = [("identity", "host_id", "another-host"),
                   ("identity", "harness_sha256", "f" * 64),
                   ("identity", "catalog_sha256", "f" * 64),
                   ("identity", "environment_sha256", "f" * 64),
                   ("contract", "timing_scope", "frontend"),
                   ("contract", "tolerance_id", "looser"),
                   ("contract", "precision_bits", 128)]
        for section, field, value in changes:
            c = copy.deepcopy(b); c[section][field] = value
            self.assertEqual(evaluate(a, c)["verdict"], "incomplete", field)
        for field, value in [("precision_bits", 128), ("threads", 8),
                             ("settings_sha256", "f" * 64), ("input_sha256", "f" * 64)]:
            c = copy.deepcopy(b); c["records"][1][field] = value
            self.assertEqual(evaluate(a, c)["verdict"], "incomplete", field)
        b["identity_unchanged"] = False
        self.assertEqual(evaluate(a, b)["verdict"], "incomplete")

    def test_numerical_status_cannot_be_promoted_by_pass_boolean(self):
        a, b = self.pair()
        b["records"][1]["status"] = "AlmostSolved"
        self.assertEqual(evaluate(a, b)["verdict"], "incomplete")

    def test_qualification_is_required_and_separate_from_smoke_gates(self):
        a, b = self.pair()
        b["qualification_passed"] = False
        self.assertEqual(evaluate(a, b)["verdict"], "discard")
        a["qualification_passed"] = False
        self.assertEqual(evaluate(a, b)["verdict"], "discard")
        b["qualification_passed"] = True
        self.assertEqual(evaluate(a, b)["verdict"], "correctness_only")

    def test_order_dependence_cannot_win(self):
        a, b = self.pair()
        for row in b["records"]:
            if row["phase"] == "warm":
                row["time_s"] = .6 if row["order_block"] == "AB" else 1.0
        r = evaluate(a, b)
        self.assertEqual(r["verdict"], "discard")
        self.assertTrue(any(reason.startswith("BA: aggregate") for reason in r["reasons"]))

    def test_family_balance_and_regression_limits(self):
        a, b = campaign(families=("LP", "LP", "SOCP")), campaign(True, ("LP", "LP", "SOCP"))
        for row in b["records"]:
            if row["phase"] == "warm":
                row["time_s"] = .7 if row["family"] == "LP" else 1.03
        r = evaluate(a, b)
        self.assertEqual(r["verdict"], "discard")
        self.assertAlmostEqual(r["comparisons"]["AB"]["family_balanced_ratio"], math.sqrt(.7 * 1.03))
        self.assertTrue(any("family regression" in s for s in r["reasons"]))
        self.assertEqual(evaluate(a, b, max_family_ratio=1.04)["verdict"], "keep")

    def test_case_and_memory_regressions_include_cold_memory(self):
        a, b = self.pair()
        for row in b["records"]:
            if row["case_id"] == "case-0" and row["phase"] == "warm":
                row["time_s"] = 1.11
        r = evaluate(a, b, max_family_ratio=2)
        self.assertTrue(any("case regression" in s for s in r["reasons"]))
        a, b = self.pair(); b["records"][0]["rss_bytes"] = 1101
        self.assertEqual(evaluate(a, b)["verdict"], "discard")

    def test_policy_cannot_relax_two_percent_or_accept_nan(self):
        a, b = self.pair()
        self.assertEqual(evaluate(a, b, min_speedup=1.0)["verdict"], "incomplete")
        self.assertEqual(evaluate(a, b, max_memory_ratio=float("nan"))["verdict"], "incomplete")

    def test_malformed_hash_and_contradictory_execution_evidence(self):
        a, b = self.pair()
        for field, value in (("input_sha256", []), ("settings_sha256", {}),
                             ("error", "watchdog failed"), ("incomplete", True)):
            c = copy.deepcopy(b)
            c["records"][1][field] = value
            self.assertEqual(evaluate(a, c)["verdict"], "incomplete")

    def test_extreme_finite_times_do_not_emit_infinity(self):
        a, b = self.pair()
        for row in a["records"]:
            row["time_s"] = 1e308
        for row in b["records"]:
            row["time_s"] = 9e307
        r = evaluate(a, b)
        self.assertEqual(r["verdict"], "keep")
        json.dumps(r, allow_nan=False)
        for row in a["records"]:
            row["time_s"] = 1.0
        for row in b["records"]:
            row["time_s"] = 5e-324
        r = evaluate(a, b)
        self.assertEqual(r["verdict"], "incomplete")
        json.dumps(r, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
