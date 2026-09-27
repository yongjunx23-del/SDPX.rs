"""Native research-controller boundary tests (no solver execution)."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


run = module("research_run_native_test", HERE / "run.py")
catalog = module("research_catalog_native_test", HERE / "catalog.py")


class RunTests(unittest.TestCase):
    def _cli(self, root):
        path = root / "sdpx"
        path.write_text("#!/bin/sh\nexit 0\n")
        path.chmod(0o755)
        return path

    def _source(self, root):
        source = root / "source"
        (source / "crates").mkdir(parents=True)
        (source / "include").mkdir()
        (source / "Cargo.toml").write_text("[workspace]\n")
        (source / "Cargo.lock").write_text("version = 3\n")
        return source

    def _config(self, root):
        source = self._source(root)
        cli = self._cli(root)
        return {"source": str(source), "cli": str(cli), "env": {"RAYON_NUM_THREADS": "1"}}

    def test_arm_identity_requires_native_cli_and_binds_executable(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = self._config(root)
            first = run.arm_identity(config)
            self.assertEqual(first["artifact_sha256"], first["cli_sha256"])
            self.assertEqual(first["cli"], str((root / "sdpx").resolve()))
            (root / "sdpx").write_text("#!/bin/sh\nexit 1\n")
            self.assertNotEqual(first["cli_sha256"], run.arm_identity(config)["cli_sha256"])
            with self.assertRaisesRegex(ValueError, "requires cli"):
                run.arm_identity({"source": config["source"]})
            with self.assertRaisesRegex(ValueError, "absolute executable"):
                run.arm_identity(dict(config, cli="sdpx"))

    def test_config_env_has_no_library_or_julia_binding(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = self._config(root)
            identity = run.arm_identity(config)
            env = run.config_env(config, identity, 4)
            self.assertEqual(env["SDPX_CLI"], str(Path(config["cli"]).resolve()))
            self.assertEqual(env["SDPX_THREADS"], "4")
            self.assertNotIn("SDPX_LIBRARY", env)
            self.assertNotIn("SDPX_EXPECTED_SOURCE", env)

    def test_run_case_launches_one_native_process_per_sample(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / "data"
            data.mkdir()
            problem = {"P": {"m": 1, "n": 1, "colptr": [0, 0], "rowval": [], "nzval": []},
                       "q": [1.0],
                       "A": {"m": 1, "n": 1, "colptr": [0, 1], "rowval": [0], "nzval": [-1.0]},
                       "b": [-1.0], "cones": [{"NonnegativeConeT": 1}]}
            (data / "x.json").write_text(json.dumps(problem) + "\n")
            entry = {"name": "x", "family": "LP", "runner": "float64", "json_path": "x.json",
                     "json_sha256": run.sha(data / "x.json")}
            config = self._config(root)
            launched = []

            def owned(command, **kwargs):
                launched.append(command)
                output = Path(command[command.index("--output") + 1])
                output.write_text(json.dumps({
                    "version": "0.7.0", "precision_bits": 53, "status": "Solved",
                    "iterations": 2, "objective": 1.0, "dual_objective": 1.0,
                    "primal_residual": 0.0, "dual_residual": 0.0,
                    "x": [1.0], "s": [0.0], "z": [1.0],
                    "native_seconds": 0.01, "api_seconds": 0.02, "load_seconds": 0.001,
                    "threads_requested": 1,
                    "cone_threads": 1, "linear_solver": "qdldl",
                    "linear_solver_threads": 1, "kkt_form": "augmented", "settings": {}
                }))
                return {"process_exit_code": 0, "cleanup_confirmed": True,
                        "native_highwater": {"max_rss_bytes": 4096}}

            with patch.object(run.native_helpers(), "audit", return_value={"pass": True}):
                rows = run.run_case(owned, config, run.arm_identity(config), entry, data,
                                    53, 1, "AB", root / "out", 60, 512, repetitions=3)
            self.assertEqual(len(launched), 3)
            self.assertTrue(all(command[0] == str(Path(config["cli"]).resolve()) for command in launched))
            self.assertEqual([row["phase"] for row in rows], ["cold", "warm", "warm"])
            self.assertTrue(all(row["passed"] for row in rows))
            self.assertEqual(rows[1]["time_s"], 0.02)
            self.assertNotIn("julia", rows[0]["runtime"])

            # A real A/B comparison uses different binaries at different paths.
            # Artifact provenance must remain distinct without rejecting equal
            # numerical settings as a sample contract mismatch.
            other_root = root / "candidate"
            other_root.mkdir()
            other = self._config(other_root)
            Path(other["cli"]).write_text("#!/bin/sh\n# different build\nexit 0\n")
            other_identity = run.arm_identity(other)
            self.assertNotEqual(run.arm_identity(config)["artifact_sha256"],
                                other_identity["artifact_sha256"])
            other_rows = run.run_case(owned, other, other_identity, entry, data,
                                      53, 1, "AB", root / "other-out", 60, 512, repetitions=1)
            self.assertTrue(other_rows[0]["passed"])
            self.assertNotEqual(rows[0]["runtime"]["cli"], other_rows[0]["runtime"]["cli"])
            self.assertNotEqual(rows[0]["runtime"]["cli_sha256"], other_rows[0]["runtime"]["cli_sha256"])
            self.assertEqual(rows[0]["settings_sha256"], other_rows[0]["settings_sha256"])

    def test_settings_identity_retains_numerical_and_backend_contract(self):
        settings = dict(tol_feas=1e-8, equilibrate_enable=True, presolve_enable=True,
                        chordal_decomposition_enable=True)
        runtime = dict(cli="/baseline/sdpx", cli_sha256="a" * 64, version="0.7.0",
                       status="Solved", precision_bits=53, linear_solver="qdldl",
                       linear_solver_threads=1, cone_threads=1,
                       kkt_form="augmented", threads_requested=1)
        reference = run.settings_identity(settings, runtime)
        self.assertEqual(reference, run.settings_identity(settings, dict(
            runtime, cli="/candidate/sdpx", cli_sha256="b" * 64, version="0.7.1")))
        for key, value in dict(tol_feas=1e-6, equilibrate_enable=False,
                               presolve_enable=False, chordal_decomposition_enable=False).items():
            with self.subTest(setting=key):
                self.assertNotEqual(reference, run.settings_identity(dict(settings, **{key: value}), runtime))
        for key, value in dict(precision_bits=256, linear_solver="faer",
                               linear_solver_threads=2, cone_threads=2,
                               kkt_form="condensed", threads_requested=2).items():
            with self.subTest(runtime=key):
                self.assertNotEqual(reference, run.settings_identity(settings, dict(runtime, **{key: value})))

    def test_run_case_retains_failed_native_slots(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / "data"
            data.mkdir()
            (data / "x.json").write_text(json.dumps({"P": {"m": 0, "n": 1, "colptr": [0, 0], "rowval": [], "nzval": []},
                "q": [1.0], "A": {"m": 0, "n": 1, "colptr": [0, 0], "rowval": [], "nzval": []},
                "b": [], "cones": []}) + "\n")
            entry = {"name": "x", "family": "LP", "runner": "float64", "json_path": "x.json",
                     "json_sha256": run.sha(data / "x.json")}
            config = self._config(root)
            calls = []

            def owned(command, **kwargs):
                calls.append(command)
                return {"process_exit_code": 1, "cleanup_confirmed": True,
                        "incomplete": True, "native_highwater": {"max_rss_bytes": 4096}}

            rows = run.run_case(owned, config, run.arm_identity(config), entry, data,
                                53, 1, "AB", root / "out", 60, 512, repetitions=2)
            self.assertEqual(len(calls), 2)
            self.assertTrue(all(not row["passed"] for row in rows))
            self.assertEqual([row["repetition"] for row in rows], [0, 1])

    def test_screen_configuration_defaults(self):
        args = SimpleNamespace(profile=None, stage="development", threads=1,
                               precision_bits=53, timeout=None, budget_seconds=None)
        protocol = run.profile_options(args)
        self.assertEqual(protocol["profile"], "screen")
        self.assertEqual((args.timeout, args.budget_seconds), (120, 180))
        self.assertEqual(protocol["repetitions"], 2)
        self.assertEqual(protocol["blocks"], ["AB"])
        self.assertFalse(protocol["speed_credit"])

    def test_float64_input_cannot_be_narrowed(self):
        with self.assertRaises(ValueError):
            run.validate_cases({"instances": [{"name": "x", "runner": "float64"}]}, 256)

    def test_synthetic_input_is_decimal_on_mpfr_wire(self):
        helper = run.native_helpers()
        value = helper.synthetic_problem({"runner": "orthant", "parameters": {"n": 1, "rows_per_variable": 2}}, 256)
        self.assertIsInstance(value["q"][0], str)
        self.assertIsInstance(value["b"][0], str)
        self.assertNotIn(".", value["q"][0])

    def test_audit_quadratic_gap_matches_primal_dual_objectives(self):
        helper = run.native_helpers()
        # Unconstrained min 1/2*x^2-x has x=1 and objective -1/2.  The
        # independent gate must count the full x'Px term in primal-dual gap.
        problem = {
            "P": {"m": 1, "n": 1, "colptr": [0, 1], "rowval": [0], "nzval": [1]},
            "q": [-1],
            "A": {"m": 0, "n": 1, "colptr": [0, 0], "rowval": [], "nzval": []},
            "b": [], "cones": [],
        }
        result = {"status": "Solved", "x": [1], "s": [], "z": [],
                  "objective": -0.5, "dual_objective": -0.5}
        checked = helper.audit(problem, result, tolerance=1e-12)
        self.assertTrue(checked["pass"], checked)
        self.assertEqual(checked["gap"], 0.0)


if __name__ == "__main__":
    unittest.main()
