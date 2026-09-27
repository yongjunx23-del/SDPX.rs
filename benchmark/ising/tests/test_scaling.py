import importlib.util
import json
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("scaling", Path(__file__).parents[1] / "scaling.py")
scaling = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scaling)


class ResultGateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.write("input.json", {"input_sha256": "frozen"})
        self.write("fresh-raw.json", {
            "status": "Solved", "precision_bits": 512,
            "native_seconds": 2, "api_seconds": 3, "load_seconds": 0.1,
            "iterations": 50, "x": ["0"], "s": ["0"], "z": ["0"],
            "linear_solver": "condensed_sampled_arrow",
            "linear_solver_threads": 1, "cone_threads": 1,
        })
        self.write("fresh-audit.json", {"accepted": True, "optimal": True})

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, path, data):
        (self.root / path).write_text(json.dumps(data))

    def test_complete_point_is_one_fresh_process(self):
        point = scaling.cell(self.root, "frozen")
        self.assertEqual(len(point["solves"]), 1)
        self.assertEqual(point["solves"][0]["label"], "fresh-process")
        self.assertEqual(point["solves"][0]["partitions"], 1)
        self.assertIn("native CLI process", point["solves"][0]["timing_scope"])

    def test_failure_is_not_a_sample(self):
        self.write("fresh-audit.json", {"accepted": False, "optimal": True})
        with self.assertRaises(ValueError):
            scaling.cell(self.root, "frozen")

    def test_changed_input(self):
        with self.assertRaises(ValueError):
            scaling.cell(self.root, "different")

    def test_incomplete_and_nonfinite(self):
        (self.root / "fresh-raw.json").unlink()
        with self.assertRaises(FileNotFoundError):
            scaling.cell(self.root, "frozen")
        self.write("fresh-raw.json", {
            "status": "Solved", "precision_bits": 512,
            "native_seconds": "NaN", "api_seconds": 3, "load_seconds": 0.1,
            "x": ["0"], "s": ["0"], "z": ["0"],
            "linear_solver": "condensed_sampled_arrow",
            "linear_solver_threads": 1, "cone_threads": 1,
        })
        with self.assertRaises(ValueError):
            scaling.cell(self.root, "frozen")

    def test_old_warmed_julia_layout_is_rejected(self):
        raw = json.loads((self.root / "fresh-raw.json").read_text())
        (self.root / "fresh-raw.json").unlink()
        (self.root / "fresh-audit.json").unlink()
        self.write("first-raw.json", raw)
        self.write("warmed-1-raw.json", raw)
        self.write("first-audit.json", {"accepted": True, "optimal": True})
        self.write("warmed-1-audit.json", {"accepted": True, "optimal": True})
        with self.assertRaises(FileNotFoundError):
            scaling.cell(self.root, "frozen")


    def test_removed_direction_settings(self):
        with self.assertRaises(ValueError):
            scaling._settings_data({"settings": {"psd_direction": "nt"}})
        with self.assertRaises(ValueError):
            scaling._settings_data({"settings": {"psd_direction": "hkm"}})

    def native_case(self):
        raw = json.loads((self.root / "fresh-raw.json").read_text())
        settings = dict(max_iter=1000, tol_feas="1e-42", tol_gap_abs="1e-42",
                        tol_gap_rel="1e-42", tol_feas_componentwise="1e-30")
        raw["settings"] = dict(settings)
        config = dict(bits=512, settings=settings,
                      plans={"1": dict(linear_solver="condensed_sampled_arrow",
                                       linear_solver_threads=1)})
        return raw, config

    def validation_case(self, partitions):
        sampled = self.root / "sampled"
        sampled.mkdir(exist_ok=True)
        (sampled / "control.json").write_text("{}")
        (sampled / "objectives.json").write_text("{}")
        cli = self.root / "sdpx"
        cli.write_text("#!/bin/sh\n")
        cli.chmod(0o755)
        source = self.root / "source"
        source.mkdir(exist_ok=True)
        audit = self.root / "julia"
        audit.write_text("")
        input_hash = scaling.sampled_input_hash(sampled)
        reference = self.root / "reference.json"
        self.write(reference.name, {"accepted": True, "input_sha256": input_hash})
        return dict(cli=str(cli), source=str(source), input=str(sampled),
                    reference=str(reference), audit_julia=str(audit),
                    settings=dict(max_iter=1000, tol_feas="1e-42",
                                  tol_gap_abs="1e-42", tol_gap_rel="1e-42"),
                    plans={}, partitions=partitions)

    def history_validation_case(self, partitions=None):
        config = self.validation_case(partitions)
        history = self.root / "cost-history.json"
        history.write_text("{\"schema\":1}\n")
        config["cost_history_in"] = str(history)
        return config, history

    def test_partitions_config_must_be_a_positive_integer(self):
        for value in (1, 3):
            with self.subTest(value=value):
                validated = scaling._validate_config(self.validation_case(value))
                self.assertEqual(validated["partitions"], value)
        self.assertEqual(scaling._validate_config(self.validation_case("auto"))["partitions"], "auto")
        for value in (0, -1, True, 1.0, "2"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                scaling._validate_config(self.validation_case(value))



    def test_history_requires_partitions_and_output_is_boolean_opt_in(self):
        config, _ = self.history_validation_case()
        with self.assertRaisesRegex(ValueError, "cost history requires configured partitions"):
            scaling._validate_config(config)
        config, _ = self.history_validation_case(3)
        config["cost_history_out"] = "cost-history.json"
        with self.assertRaisesRegex(ValueError, "cost_history_out must be a boolean"):
            scaling._validate_config(config)
        config["cost_history_out"] = True
        validated = scaling._validate_config(config)
        self.assertEqual(validated["partitions"], 3)
        self.assertTrue(validated["cost_history_out"])
        self.assertEqual(Path(validated["cost_history_in"]).resolve(),
                         (self.root / "cost-history.json").resolve())

    def test_history_identity_hash_changes_with_input_bytes(self):
        config, history = self.history_validation_case(3)
        with patch.object(scaling, "_native_arm_identity", return_value={"source_sha256": "source"}):
            first = scaling.identity(config)
            history.write_text("{\"schema\":2}\n")
            second = scaling.identity(config)
        self.assertEqual(first["cost_history_in"]["path"], str(history.resolve()))
        self.assertEqual(first["cost_history_in_sha256"],
                         first["cost_history_in"]["sha256"])
        self.assertNotEqual(first["cost_history_in_sha256"], second["cost_history_in_sha256"])

    def test_auto_requires_positive_width_plan_and_matches_actual_count(self):
        raw, config = self.native_case()
        config["partitions"] = "auto"
        raw["partitions"] = 3
        with self.assertRaises(ValueError):
            scaling._check_native_result(raw, config, 1)

        config["plans"]["1"]["partitions"] = 3
        scaling._check_native_result(raw, config, 1)
        raw["partitions"] = 2
        with self.assertRaises(ValueError):
            scaling._check_native_result(raw, config, 1)

        for bad in (0, -1, True, 1.0, "3"):
            config["plans"]["1"]["partitions"] = bad
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                scaling._check_native_result(raw, config, 1)

    def test_auto_cell_keeps_requested_mode_and_reports_actual_count(self):
        raw, config = self.native_case()
        config["partitions"] = "auto"
        config["plans"]["1"]["partitions"] = 3
        raw["partitions"] = 3
        case = json.loads((self.root / "input.json").read_text())
        case["partitions"] = "auto"
        self.write("input.json", case)
        self.write("fresh-raw.json", raw)
        point = scaling.cell(self.root, "frozen", config=config, width=1)
        self.assertEqual(point["input"]["partitions"], "auto")
        self.assertEqual(point["solves"][0]["partitions"], 3)

        case["partitions"] = 3
        self.write("input.json", case)
        with self.assertRaises(ValueError):
            scaling.cell(self.root, "frozen", config=config, width=1)

        case["partitions"] = "auto"
        raw["partitions"] = 2
        self.write("input.json", case)
        self.write("fresh-raw.json", raw)
        with self.assertRaises(ValueError):
            scaling.cell(self.root, "frozen", config=config, width=1)

    def test_auto_command_flag_is_forwarded(self):
        raw, config = self.native_case()
        config.update(partitions="auto", input_sha256="frozen", cli="sdpx", input="input")
        config["plans"]["1"]["partitions"] = 3
        raw["partitions"] = 3
        directory = self.root / "run"
        captured = {}

        def supervise(command, **_kwargs):
            captured[command[0]] = list(command)
            if command[0] == "sdpx":
                output = Path(command[command.index("--output") + 1])
                output.write_text(json.dumps(raw))
            elif command[0] == "audit":
                Path(command[1]).write_text(json.dumps({"accepted": True, "optimal": True}))
            return {"incomplete": False, "process_exit_code": 0, "peak_rss_mib": 0}

        with patch.object(scaling, "_native_env", return_value={}), \
                patch.object(scaling, "_pin_command", side_effect=lambda command, *_: command), \
                patch.object(scaling, "_audit_command", side_effect=lambda _config, _raw, audit: ["audit", str(audit)]):
            scaling._run_one(config, directory, 1, [], supervise, 10, 64, False)
        command = captured["sdpx"]
        self.assertEqual(command[command.index("--partitions") + 1], "auto")
        self.assertNotIn("--cost-history-in", command)
        self.assertNotIn("--cost-history-out", command)


    def test_history_flags_and_per_run_output_metadata_are_recorded(self):
        raw, config = self.native_case()
        history = self.root / "trained-history.json"
        history.write_text("{\"schema\":1}\n")
        config.update(partitions="auto", input_sha256="frozen", cli="sdpx", input="input",
                      cost_history_in=str(history), cost_history_out=True)
        config["plans"]["1"]["partitions"] = 3
        raw["partitions"] = 3
        directory = self.root / "history-run"
        captured = {}

        def supervise(command, **_kwargs):
            captured[command[0]] = list(command)
            if command[0] == "sdpx":
                output = Path(command[command.index("--output") + 1])
                output.write_text(json.dumps(raw))
                exported = Path(command[command.index("--cost-history-out") + 1])
                exported.write_text("{\"trained\":true}\n")
            elif command[0] == "audit":
                Path(command[1]).write_text(json.dumps({"accepted": True, "optimal": True}))
            return {"incomplete": False, "process_exit_code": 0, "peak_rss_mib": 0}

        with patch.object(scaling, "_native_env", return_value={}), \
                patch.object(scaling, "_pin_command", side_effect=lambda command, *_: command), \
                patch.object(scaling, "_audit_command", side_effect=lambda _config, _raw, audit: ["audit", str(audit)]):
            value, _ = scaling._run_one(config, directory, 1, [], supervise, 10, 64, False)

        command = captured["sdpx"]
        self.assertEqual(command[command.index("--partitions") + 1], "auto")
        self.assertEqual(command[command.index("--cost-history-in") + 1], str(history.resolve()))
        output_path = Path(command[command.index("--cost-history-out") + 1])
        self.assertEqual(output_path, directory / "cost-history.json")
        self.assertEqual(value["history"]["output"]["path"], str(output_path.resolve()))
        self.assertEqual(value["history"]["output"]["scope"],
                         "fresh per-run directory; generated output, not fixed input")
        receipt = json.loads((directory / "solver-process.json").read_text())
        self.assertEqual(receipt["cli_command"], command)
        self.assertEqual(receipt["history"]["input"]["sha256"], scaling.research.sha(history))

    def test_history_metadata_rejects_changed_input_or_fixed_output_path(self):
        raw, config = self.native_case()
        history = self.root / "trained-history.json"
        history.write_text("{\"schema\":1}\n")
        config.update(partitions=3, cost_history_in=str(history), cost_history_out=True)
        raw["partitions"] = 3
        case = json.loads((self.root / "input.json").read_text())
        case["partitions"] = 3
        output = self.root / "cost-history.json"
        output.write_text("{\"trained\":true}\n")
        case["history"] = {
            "input": {"path": str(history.resolve()), "sha256": scaling.research.sha(history)},
            "output": {"path": str(output.resolve()), "sha256": scaling.research.sha(output),
                       "scope": "fresh per-run directory; generated output, not fixed input"},
        }
        self.write("input.json", case)
        self.write("fresh-raw.json", raw)
        scaling.cell(self.root, "frozen", config=config, width=1)
        history.write_text("{\"schema\":2}\n")
        with self.assertRaises(ValueError):
            scaling.cell(self.root, "frozen", config=config, width=1)
        case["history"]["input"]["sha256"] = scaling.research.sha(history)
        case["history"]["output"]["path"] = str(self.root.parent / "fixed-history.json")
        self.write("input.json", case)
        with self.assertRaises(ValueError):
            scaling.cell(self.root, "frozen", config=config, width=1)

    def test_real_mpfr_roundtrip_settings(self):
        raw, config = self.native_case()
        # Real 512-bit CLI output: the last digit differs from the decimal
        # input even though it round-trips to exactly the requested MPFR value.
        for key in ("tol_feas", "tol_gap_abs", "tol_gap_rel", "tol_feas_componentwise"):
            exponent = -30 if key == "tol_feas_componentwise" else -42
            raw["settings"][key] = "1." + "0" * 154 + "4e" + str(exponent)
        scaling._check_native_result(raw, config, 1)

    def test_mpfr_identity_rounds_ties_even(self):
        for text, expected in (("1.125", 1), ("1.375", Fraction(3, 2)),
                               ("1.875", 2), ("-1.125", -1),
                               ("0.140625", Fraction(1, 8)), ("0", 0)):
            with self.subTest(text=text):
                self.assertEqual(scaling._mpfr_setting(text, 3, "test"), expected)
        self.assertEqual(scaling._mpfr_setting("1e-42", 53, "test"),
                         Fraction.from_float(float("1e-42")))
        for bits in (512, 768):
            with localcontext() as ctx:
                ctx.prec = 1000
                neighbor = Decimal(1) + Decimal(2) ** (1 - bits)
            self.assertNotEqual(scaling._mpfr_setting(neighbor, bits, "test"), 1)

    def test_changed_or_binary64_rounded_tolerance_rejected(self):
        for bad in ("1e-41", str(Decimal.from_float(float("1e-42"))), "NaN"):
            raw, config = self.native_case()
            raw["settings"]["tol_feas"] = bad
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                scaling._check_native_result(raw, config, 1)

    def test_componentwise_setting_must_match_requested_value(self):
        for bad in (None, "1e-29", "1e-31"):
            raw, config = self.native_case()
            raw["settings"]["tol_feas_componentwise"] = bad
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                scaling._check_native_result(raw, config, 1)
        raw, config = self.native_case()
        config["settings"].pop("tol_feas_componentwise")
        with self.assertRaises(ValueError):
            scaling._check_native_result(raw, config, 1)
        raw["settings"].pop("tol_feas_componentwise")
        scaling._check_native_result(raw, config, 1)

    def test_configured_partitions_match_native_and_cell_metadata(self):
        raw, config = self.native_case()
        config["partitions"] = 3
        raw["partitions"] = 3
        case = json.loads((self.root / "input.json").read_text())
        case["partitions"] = 3
        self.write("input.json", case)
        scaling._check_native_result(raw, config, 1)
        self.write("fresh-raw.json", raw)
        point = scaling.cell(self.root, "frozen", config=config, width=1)
        self.assertEqual(point["solves"][0]["partitions"], 3)

        raw["partitions"] = 2
        self.write("fresh-raw.json", raw)
        with self.assertRaises(ValueError):
            scaling.cell(self.root, "frozen", config=config, width=1)

        raw["partitions"] = 3
        self.write("fresh-raw.json", raw)
        case["partitions"] = 2
        self.write("input.json", case)
        with self.assertRaises(ValueError):
            scaling.cell(self.root, "frozen", config=config, width=1)

    def test_missing_settings_and_residual_failure_stay_rejected(self):
        raw, config = self.native_case()
        raw.pop("settings")
        with self.assertRaises(ValueError):
            scaling._check_native_result(raw, config, 1)
        raw, config = self.native_case()
        raw["dual_residual"] = "1.00000000000000000000000001e-42"
        with self.assertRaises(ValueError):
            scaling._check_native_result(raw, config, 1)


if __name__ == "__main__":
    unittest.main()
