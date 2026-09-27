#!/usr/bin/env python3
"""Bounded Ising scaling driver for the native SDPX command-line solver.

Each measured point is one fresh native process.  The independent Julia
``audit_point.jl`` process runs after the solver and is excluded from all
solver/API timings.  Keep outputs outside the source tree so a campaign can
prove that its input, executable and harness stayed immutable.
"""
import argparse
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import statistics
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
spec = importlib.util.spec_from_file_location("research", HERE.parent / "research/run.py")
research = importlib.util.module_from_spec(spec)
spec.loader.exec_module(research)

SUPPORTED_WIDTHS = (1, 2, 4, 8, 16, 32, 64)
INTERNAL_TOLERANCE = "1e-42"
EXTERNAL_TOLERANCE = "1e-30"


def _absolute_file(value, name):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty absolute path")
    path = Path(value)
    if not path.is_absolute() or not path.is_file():
        raise ValueError(f"{name} must name an existing absolute file")
    return path.resolve()


def _absolute_dir(value, name):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty absolute directory")
    path = Path(value)
    if not path.is_absolute() or not path.is_dir():
        raise ValueError(f"{name} must name an existing absolute directory")
    return path.resolve()


def _sha(path):
    return research.sha(path)


def sampled_input_hash(directory):
    """Return the sampled-reader input hash without loading solver arithmetic.

    This is the same manifest hash used by ``sampled_to_conic.jl`` and by the
    native sampled JSON adapter: every JSON file is hashed individually, then
    sorted by file name before the manifest is hashed.
    """
    directory = _absolute_dir(str(directory), "input")
    files = sorted(p for p in directory.iterdir() if p.is_file() and p.suffix == ".json")
    required = {"control.json", "objectives.json"}
    names = {p.name for p in files}
    if not required <= names:
        raise ValueError("sampled input must contain control.json and objectives.json")
    entries = "".join(f"{p.name}\0{_sha(p)}\n" for p in files)
    return hashlib.sha256(entries.encode()).hexdigest()


def _native_arm_identity(config):
    """Fingerprint a frozen native arm through the shared research helper."""
    return research.arm_identity(config)


def _native_env(config, width):
    """Use the shared environment helper for native CLI thread identity."""
    arm = _native_arm_identity(config)
    return research.config_env(config, arm, width)


def _audit_runtime(config):
    # ``julia`` is retained only as an independent GenericLinearAlgebra/JSON
    # audit runtime.  ``audit_julia`` is preferred to make package retirement
    # explicit, while the old key remains a compatible spelling.
    oracle = config.get("oracle")
    if isinstance(oracle, dict):
        value = oracle.get("julia", oracle.get("runtime"))
    else:
        value = config.get("audit_julia", config.get("julia"))
    return _absolute_file(value, "audit_julia")


def _audit_project(config):
    oracle = config.get("oracle")
    if isinstance(oracle, dict):
        value = oracle.get("project")
    else:
        value = config.get("audit_project", config.get("julia_project"))
    if value is None:
        return None
    return _absolute_dir(value, "audit_project")


def _settings_data(config):
    settings = config.get("settings")
    if settings is None:
        raise ValueError("settings is required to retain the 1e-42 native solver gate")
    if isinstance(settings, str):
        settings = research.read(_absolute_file(settings, "settings"))
    if not isinstance(settings, dict):
        raise ValueError("settings must be a JSON path or object")
    settings = dict(settings)
    if "psd_direction" in settings:
        raise ValueError("psd_direction has been removed; delete it from settings")
    return settings


def _decimal_text(value, name):
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise ValueError(f"settings {name} must be a finite decimal")
    if not result.is_finite():
        raise ValueError(f"settings {name} must be finite")
    return result


def _mpfr_setting(value, bits, name):
    """Exact round-to-nearest/even identity at the declared MPFR precision.

    MPFR's round-trip decimal output need not equal the input decimal (1e-42
    is not binary-exact). Compare represented values, without a tolerance or
    a binary64 intermediate. Residual acceptance gates remain separate.
    """
    value = Fraction(_decimal_text(value, name))
    if not value:
        return value
    sign = -1 if value < 0 else 1
    value = abs(value)
    exponent = value.numerator.bit_length() - value.denominator.bit_length()
    scale = Fraction(2) ** exponent
    if value < scale:
        scale /= 2
    quantum = scale / (1 << (bits - 1))
    scaled = value / quantum
    significand, remainder = divmod(scaled.numerator, scaled.denominator)
    twice_remainder = 2 * remainder
    if twice_remainder > scaled.denominator or (
            twice_remainder == scaled.denominator and significand % 2):
        significand += 1
    return sign * significand * quantum


def _positive_partition(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _cost_history_input(config):
    """Resolve an optional immutable cost-history input file.

    History is deliberately a file rather than an opaque value so the
    campaign identity can hash the exact bytes that the native CLI consumes.
    The output side is handled separately and is always created below a fresh
    per-run directory.
    """
    value = config.get("cost_history_in")
    if value is None:
        return None
    return _absolute_file(value, "cost_history_in")


def _cost_history_output_enabled(config):
    """Return whether this route asks the native CLI to export history."""
    if "cost_history_out" not in config:
        return False
    value = config["cost_history_out"]
    if not isinstance(value, bool):
        raise ValueError("cost_history_out must be a boolean opt-in")
    return value


def _cost_history_metadata(config, output_path=None):
    """Describe history inputs/outputs without treating generated output as input.

    The returned object is suitable for per-run receipts.  ``output_path`` is
    supplied only after the native process has written its file, so its hash is
    never folded into the immutable campaign identity.
    """
    history_input = _cost_history_input(config)
    metadata = {}
    if history_input is not None:
        metadata["input"] = {"path": str(history_input), "sha256": research.sha(history_input)}
    if output_path is not None:
        output_path = Path(output_path).resolve()
        if not output_path.is_file():
            raise RuntimeError("native solver did not produce cost-history output")
        metadata["output"] = {
            "path": str(output_path),
            "sha256": research.sha(output_path),
            "scope": "fresh per-run directory; generated output, not fixed input",
        }
    return metadata


def _validate_config(config):
    if not isinstance(config, dict):
        raise ValueError("config must be a JSON object")
    cli = _absolute_file(config.get("cli"), "cli")
    if not os.access(cli, os.X_OK):
        raise ValueError("cli must be executable")
    _absolute_dir(config.get("source"), "source")
    input_dir = _absolute_dir(config.get("input"), "input")
    reference = _absolute_file(config.get("reference"), "reference")
    audit_julia = _audit_runtime(config)
    audit_project = _audit_project(config)
    bits = config.get("bits", 512)
    if isinstance(bits, bool) or bits not in (512, 768):
        raise ValueError("bits must be 512 or 768")
    partitions = config.get("partitions")
    if partitions is not None and partitions != "auto":
        _positive_partition(partitions, "partitions")
    history_input = _cost_history_input(config)
    history_output = _cost_history_output_enabled(config)
    if (history_input is not None or history_output) and partitions is None:
        raise ValueError("cost history requires configured partitions")
    if "psd_direction" in config:
        raise ValueError("psd_direction has been removed; delete it from config")
    expected_hash = config.get("input_sha256")
    actual_hash = sampled_input_hash(input_dir)
    if expected_hash is not None and expected_hash != actual_hash:
        raise ValueError("input_sha256 does not match sampled JSON directory")
    if not isinstance(config.get("plans", {}), dict):
        raise ValueError("plans must be an object keyed by thread width")
    settings = _settings_data(config)
    if settings.get("max_iter") != 1000:
        raise ValueError("settings max_iter must remain 1000")
    for key in ("tol_feas", "tol_gap_abs", "tol_gap_rel"):
        if _decimal_text(settings.get(key), key) != Decimal(INTERNAL_TOLERANCE):
            raise ValueError(f"settings {key} must remain {INTERNAL_TOLERANCE}")
    if "tol_feas_componentwise" in settings:
        supplied = settings["tol_feas_componentwise"]
        if supplied is not None:
            componentwise = _decimal_text(supplied, "tol_feas_componentwise")
            if componentwise <= 0 or componentwise > Decimal(EXTERNAL_TOLERANCE):
                raise ValueError(f"settings tol_feas_componentwise must be in (0, {EXTERNAL_TOLERANCE}]")
    if "time_limit" in settings and settings["time_limit"] is not None:
        if _decimal_text(settings["time_limit"], "time_limit") <= 0:
            raise ValueError("settings time_limit must be positive")
    # A reference must be accepted and correspond to the sampled input.  The
    # accepted point stores input_sha256 under mapping in older receipts.
    reference_data = research.read(reference)
    if reference_data.get("accepted") is not True:
        raise ValueError("reference audit was not accepted")
    reference_hash = reference_data.get("input_sha256")
    if reference_hash is None and isinstance(reference_data.get("mapping"), dict):
        reference_hash = reference_data["mapping"].get("input_sha256")
    if reference_hash != actual_hash:
        raise ValueError("reference audit belongs to another sampled input")
    normalized = dict(config, cli=str(cli), input=str(input_dir), reference=str(reference),
                      audit_julia=str(audit_julia), audit_project=str(audit_project) if audit_project else None,
                      bits=bits, input_sha256=actual_hash)
    if history_input is not None:
        normalized["cost_history_in"] = str(history_input)
    if "cost_history_out" in config:
        normalized["cost_history_out"] = history_output
    return normalized


def identity(config):
    """Capture immutable arm, input, reference and benchmark-harness hashes."""
    config = _validate_config(config)
    inputs = Path(config["input"])
    files = {str(p.relative_to(inputs)): research.sha(p)
             for p in sorted(inputs.rglob("*")) if p.is_file()}
    if not files:
        raise ValueError("empty Ising input")
    helpers = [HERE / name for name in
               ("scaling.py", "gate.jl", "comparison.pbs",
                "audit_helpers.jl", "audit_point.jl")]
    helpers += list((HERE / "sampled").glob("*.jl"))
    helpers += [Path(__file__).resolve(), HERE.parent / "research/run.py",
                HERE.parent / "research/native.py",
                HERE.parent / "float64/run.py", HERE.parent / "float64/resource_probe.py"]
    helper_hashes = {str(p.relative_to(ROOT)): research.sha(p) for p in helpers if p.is_file()}
    result = dict(arm=_native_arm_identity(config), input_files=files,
                  input_sha256=sampled_input_hash(inputs),
                  reference=research.sha(config["reference"]), harness=helper_hashes,
                  config=config, partitions=config.get("partitions", 1))
    history_input = _cost_history_input(config)
    if history_input is not None:
        history_sha = research.sha(history_input)
        # Keep generated cost-history output out of this immutable identity.
        # A subsequent invocation can use its recorded output path as the
        # next invocation's cost_history_in and will then hash those bytes.
        result["cost_history_in"] = {"path": str(history_input), "sha256": history_sha}
        result["cost_history_in_sha256"] = history_sha
    settings = config.get("settings")
    if isinstance(settings, str):
        settings_path = _absolute_file(settings, "settings")
        result["settings_sha256"] = research.sha(settings_path)
    elif settings is not None:
        result["settings_sha256"] = research.digest(settings)
    return result


def physical_cpus():
    if not hasattr(os, "sched_getaffinity"):
        return []  # macOS has no supported hard-affinity API; report this.
    cores = {}
    for cpu in sorted(os.sched_getaffinity(0)):
        p = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        package = p / "physical_package_id"
        core = p / "core_id"
        if not package.is_file() or not core.is_file():
            return []
        key = (package.read_text().strip(), core.read_text().strip())
        cores.setdefault(key, cpu)
    return [cpu for _, cpu in sorted(cores.items(), key=lambda item: tuple(map(int, item[0])))]


def _external_output(path, config):
    path = Path(path).resolve()
    root = ROOT.resolve()
    try:
        path.relative_to(root)
    except ValueError:
        pass
    else:
        raise ValueError("campaign output must be outside the SDPX source tree")
    for name in ("source", "input"):
        base = Path(config[name]).resolve()
        try:
            path.relative_to(base)
        except ValueError:
            continue
        raise ValueError(f"campaign output must be outside frozen {name}")


def _read_json(path):
    return research.read(path)


def _expected_plan(config, width):
    plans = config.get("plans", {})
    plan = plans.get(str(width), plans.get(width))
    if plan is None:
        raise ValueError(f"missing native plan for width {width}")
    if not isinstance(plan, dict):
        raise ValueError(f"plan for width {width} must be an object")
    if config.get("partitions") == "auto":
        _positive_partition(plan.get("partitions"),
                           f"plan partitions for width {width}")
    return plan


def _check_native_result(raw, config, width):
    required = ("status", "precision_bits", "native_seconds", "api_seconds", "load_seconds",
                "x", "s", "z", "linear_solver", "linear_solver_threads", "cone_threads", "settings")
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError("native result missing fields: " + ", ".join(missing))
    if raw["status"] != "Solved":
        raise ValueError(f"native status is {raw['status']!r}, expected Solved")
    if raw["precision_bits"] != config["bits"]:
        raise ValueError("native precision does not match config")
    actual_settings = raw.get("settings")
    if not isinstance(actual_settings, dict):
        raise ValueError("native settings must be an object")
    if actual_settings.get("max_iter") != 1000:
        raise ValueError("native settings max_iter mismatch")
    for key in ("tol_feas", "tol_gap_abs", "tol_gap_rel"):
        if _mpfr_setting(actual_settings.get(key), config["bits"], key) != _mpfr_setting(
                INTERNAL_TOLERANCE, config["bits"], key):
            raise ValueError(f"native settings {key} mismatch")
    key = "tol_feas_componentwise"
    componentwise = actual_settings.get(key)
    expected = _settings_data(config).get(key)
    if (componentwise is None) != (expected is None) or (
            componentwise is not None and _mpfr_setting(componentwise, config["bits"], key)
            != _mpfr_setting(expected, config["bits"], key)):
        raise ValueError("native settings tol_feas_componentwise mismatch")
    for key in ("native_seconds", "api_seconds"):
        value = float(raw[key])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"invalid native timing: {key}")
    for key in ("primal_residual", "dual_residual"):
        if key in raw and _decimal_text(raw[key], key) > Decimal(INTERNAL_TOLERANCE):
            raise ValueError(f"native {key} exceeds {INTERNAL_TOLERANCE}")
    load = float(raw["load_seconds"])
    if not math.isfinite(load) or load < 0:
        raise ValueError("invalid native timing: load_seconds")
    if not isinstance(raw["x"], list) or not isinstance(raw["s"], list) or not isinstance(raw["z"], list):
        raise ValueError("native vectors must be arrays")
    if not isinstance(raw["linear_solver"], str) or not raw["linear_solver"]:
        raise ValueError("native linear_solver metadata is missing")
    plan = _expected_plan(config, width)
    configured_partitions = config.get("partitions")
    if configured_partitions == "auto":
        expected_partitions = _positive_partition(
            plan.get("partitions"), f"plan partitions for width {width}")
        reported_partitions = _positive_partition(
            raw.get("partitions"), "native partitions")
        if reported_partitions != expected_partitions:
            raise ValueError("native partition plan mismatch")
    elif configured_partitions is not None:
        reported_partitions = _positive_partition(
            raw.get("partitions"), "native partitions")
        if reported_partitions != configured_partitions:
            raise ValueError("native partition plan mismatch")
    expected_cone = plan.get("cone_threads", width)
    expected_linear_threads = plan.get("linear_solver_threads", plan.get("backend_threads"))
    if expected_cone is not None and raw["cone_threads"] != expected_cone:
        raise ValueError("native cone thread plan mismatch")
    if expected_linear_threads is not None and raw["linear_solver_threads"] != expected_linear_threads:
        raise ValueError("native linear solver thread plan mismatch")
    expected_solver = plan.get("linear_solver", plan.get("factorization"))
    if expected_solver is not None and raw["linear_solver"] != expected_solver:
        raise ValueError("native linear solver plan mismatch")
    if raw.get("threads_requested", width) != width:
        raise ValueError("native requested thread budget mismatch")


def cell(directory, expected_hash, config=None, width=None):
    """Validate one fresh native solve and its independent audit receipt.

    ``fresh-raw.json`` is intentionally singular: the native CLI starts one
    solver per process and has no warmed/in-process timing mode.
    """
    directory = Path(directory)
    case = _read_json(directory / "input.json")
    if case["input_sha256"] != expected_hash:
        raise ValueError("input identity mismatch")
    raw = _read_json(directory / "fresh-raw.json")
    audit = _read_json(directory / "fresh-audit.json")
    if audit.get("accepted") is not True or audit.get("optimal") is not True:
        raise ValueError("failed original-coordinate audit")
    if config is not None:
        expected_partitions = config.get("partitions", 1)
        reported_partitions = case.get("partitions", 1)
        if expected_partitions == "auto":
            if reported_partitions != "auto":
                raise ValueError("cell partition metadata mismatch")
        else:
            _positive_partition(reported_partitions, "cell partitions")
            if reported_partitions != expected_partitions:
                raise ValueError("cell partition metadata mismatch")
        expected_history_input = _cost_history_input(config)
        expected_history_output = _cost_history_output_enabled(config)
        history = case.get("history")
        if expected_history_input is None and not expected_history_output:
            if history is not None:
                raise ValueError("unexpected cost history metadata")
        else:
            if not isinstance(history, dict):
                raise ValueError("missing cost history metadata")
            if expected_history_input is None:
                if "input" in history:
                    raise ValueError("cost history input metadata mismatch")
            else:
                input_metadata = history.get("input")
                if not isinstance(input_metadata, dict):
                    raise ValueError("missing cost history input metadata")
                if input_metadata.get("path") != str(expected_history_input.resolve()):
                    raise ValueError("cost history input path mismatch")
                if input_metadata.get("sha256") != research.sha(expected_history_input):
                    raise ValueError("cost history input hash mismatch")
            output_metadata = history.get("output")
            if expected_history_output:
                if not isinstance(output_metadata, dict):
                    raise ValueError("missing cost history output metadata")
                output_path = Path(output_metadata.get("path", "")).resolve()
                if output_path.parent != directory.resolve():
                    raise ValueError("cost history output must stay in the per-run directory")
                if output_metadata.get("scope") != (
                        "fresh per-run directory; generated output, not fixed input"):
                    raise ValueError("cost history output scope metadata mismatch")
                if output_metadata.get("sha256") != research.sha(output_path):
                    raise ValueError("cost history output hash mismatch")
            elif output_metadata is not None:
                raise ValueError("unexpected cost history output metadata")
    if config is not None and width is not None:
        _check_native_result(raw, config, width)
    native, api, load = (float(raw[k]) for k in ("native_seconds", "api_seconds", "load_seconds"))
    if not all(math.isfinite(v) and v > 0 for v in (native, api)) or not math.isfinite(load) or load < 0:
        raise ValueError("invalid timing")
    result = dict(input=case, solves=[dict(label="fresh-process", native_seconds=native,
                                           api_seconds=api, load_seconds=load,
                                           iterations=raw.get("iterations"),
                                           status=raw["status"],
                                           precision_bits=raw["precision_bits"],
                                           linear_solver=raw["linear_solver"],
                                           linear_solver_threads=raw["linear_solver_threads"],
                                           cone_threads=raw["cone_threads"],
                                           partitions=raw.get("partitions", 1),
                                           timing_scope="native CLI process: native/API/load sub-timers; process and audit separate")],
                  audit=audit)
    if config is not None and case.get("history") is not None:
        result["history"] = case["history"]
    return result


def _settings_path(config, directory):
    settings = config.get("settings")
    if settings is None:
        return None, None
    if isinstance(settings, str):
        path = _absolute_file(settings, "settings")
        return path, None
    if not isinstance(settings, dict):
        raise ValueError("settings must be a JSON path or object")
    # Keep generated settings outside the solver output's measured process. It
    # is written before launch and is hashed through the config identity.
    path = Path(directory) / "native-settings.json"
    path.write_text(json.dumps(settings, indent=2, sort_keys=True) + "\n")
    return path, settings


def _pin_command(command, cpus, width, env):
    if cpus:
        pin = ",".join(map(str, cpus[:width]))
        env["SDPX_RANK_CPUS"] = pin
        if platform.system() == "Linux":
            return ["taskset", "-c", pin] + command
    return command


def _audit_command(config, raw_path, audit_path):
    command = [config["audit_julia"], "--startup-file=no", "-t1", "--gcthreads=1"]
    if config.get("audit_project"):
        command.append("--project=" + config["audit_project"])
    command += [str(HERE / "audit_point.jl"), config["input"], str(raw_path),
                str(audit_path), str(config["bits"]), config["reference"]]
    return command


def _run_one(config, directory, width, cpus, supervise, timeout, memory_mib, profile):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    settings_path, _ = _settings_path(config, directory)
    history_input = _cost_history_input(config)
    history_output_enabled = _cost_history_output_enabled(config)
    if (history_input is not None or history_output_enabled) and config.get("partitions") is None:
        raise ValueError("cost history requires configured partitions")
    history_output = directory / "cost-history.json" if history_output_enabled else None
    raw_path = directory / "fresh-raw.json"
    audit_path = directory / "fresh-audit.json"
    env = _native_env(config, width)
    for key in ("OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS",
                "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = "1"
    env["RAYON_NUM_THREADS"] = str(width)
    env.pop("SDPX_RANK_CPUS", None)
    env.pop("SDPX_PROFILE", None)
    env.pop("SDPX_RECEIPT", None)
    if profile:
        env["SDPX_PROFILE"] = "1"
        env["SDPX_RECEIPT"] = str(directory / "phases.json")
    cli_command = [config["cli"], config["input"], "--precision", str(config["bits"]),
                   "--threads", str(width),
                   "--output", str(raw_path), "--quiet"]
    partitions = config.get("partitions")
    if partitions is not None:
        cli_command += ["--partitions", str(partitions)]
    if settings_path is not None:
        cli_command += ["--settings", str(settings_path)]
    if history_input is not None:
        cli_command += ["--cost-history-in", str(history_input)]
    if history_output is not None:
        cli_command += ["--cost-history-out", str(history_output)]
    command = _pin_command(cli_command, cpus, width, env)
    with (directory / "solver.stdout").open("w") as stdout, (directory / "solver.stderr").open("w") as stderr:
        process = supervise(command, env=env, stdout=stdout, stderr=stderr,
                            timeout=timeout, memory_limit_mib=memory_mib)
    process_receipt = dict(process, command=command,
                           timing_scope="native CLI process; audit excluded")
    # Preserve the exact unwrapped argv as well as the optional CPU pinning
    # wrapper.  This is especially useful for history train/export and reload
    # runs, where the flags must be auditable independently of taskset.
    if history_input is not None or history_output is not None:
        process_receipt["cli_command"] = cli_command
        process_receipt["history"] = _cost_history_metadata(config)
    research.write(directory / "solver-process.json", process_receipt)
    if process.get("incomplete") or process.get("process_exit_code") != 0:
        raise RuntimeError("failed native solve: " + str(directory))
    if not raw_path.is_file():
        raise RuntimeError("native solver did not produce JSON output")
    history = None
    if history_input is not None or history_output is not None:
        history = _cost_history_metadata(config, history_output)
        process_receipt["history"] = history
        research.write(directory / "solver-process.json", process_receipt)
    audit_command = _audit_command(config, raw_path, audit_path)
    with (directory / "audit.stdout").open("w") as stdout, (directory / "audit.stderr").open("w") as stderr:
        audit_process = supervise(audit_command, env=env, stdout=stdout, stderr=stderr,
                                  timeout=min(timeout, 300), memory_limit_mib=memory_mib)
    research.write(directory / "audit-process.json", dict(audit_process, command=audit_command,
                      timing_scope="independent original-coordinate audit; excluded from solver timing"))
    if audit_process.get("incomplete") or audit_process.get("process_exit_code") != 0 or not audit_path.is_file():
        raise RuntimeError("failed independent original-coordinate audit: " + str(directory))
    case = dict(input_sha256=config["input_sha256"], cli=config["cli"], bits=config["bits"],
                width=width, timing_scope="fresh native CLI process",
                audit_scope="independent Julia GenericLinearAlgebra process outside solve timing",
                profile=profile, partitions=partitions if partitions is not None else 1)
    if history is not None:
        case["history"] = history
    research.write(directory / "input.json", case)
    return cell(directory, config["input_sha256"], config=config, width=width), process


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--widths", default="1,2,4,8")
    parser.add_argument("--repetitions", type=int, choices=(1, 3), default=1)
    parser.add_argument("--seconds", type=float, default=600)
    parser.add_argument("--cell-seconds", type=float, default=180)
    parser.add_argument("--memory-mib", type=int, default=4096)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    try:
        widths = [int(x) for x in args.widths.split(",")]
    except ValueError:
        parser.error("widths must be distinct supported powers of two through 64")
    if not widths or len(set(widths)) != len(widths) or any(w not in SUPPORTED_WIDTHS for w in widths):
        parser.error("widths must be distinct supported powers of two through 64")
    if min(args.seconds, args.cell_seconds, args.memory_mib) <= 0:
        parser.error("resource bounds must be positive")
    raw_config = research.read(args.config)
    try:
        config = _validate_config(raw_config)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    for width in widths:
        _expected_plan(config, width)
    cpus = physical_cpus()
    available = os.cpu_count() or 1
    if (cpus and max(widths) > len(cpus)) or max(widths) > available:
        parser.error("insufficient available physical cores")
    try:
        _external_output(args.output, config)
    except ValueError as error:
        parser.error(str(error))
    args.output.mkdir(parents=True, exist_ok=False)
    before = identity(config)
    research.write(args.output / "identity-before.json", before)
    result = dict(accepted=False, immutable=False, profile="qualification" if args.repetitions == 3 and not args.profile else "screen",
                  host=platform.node(), platform=platform.platform(), cpus=cpus,
                  affinity_enforced=bool(cpus), bits=config["bits"], widths=widths,
                  partitions=config.get("partitions", 1),
                  expected_cells=args.repetitions * len(widths), cells=[], summary={})
    deadline = time.monotonic() + args.seconds
    supervise = research.supervisor()
    try:
        with research.slot(Path(config.get("state", "/tmp/sdpx-ising-state"))):
            for repetition in range(args.repetitions):
                order = widths if repetition % 2 == 0 else widths[::-1]
                for width in order:
                    remaining = deadline - time.monotonic()
                    if remaining < args.cell_seconds:
                        raise RuntimeError("insufficient campaign budget for another complete cell")
                    directory = args.output / (f"r{repetition + 1}-t{width}")
                    plan = _expected_plan(config, width)
                    try:
                        value, process = _run_one(config, directory, width, cpus, supervise,
                                                  args.cell_seconds, args.memory_mib, args.profile)
                    except Exception as error:
                        # Keep a failed slot in the denominator before stopping
                        # the sweep; no failed point is silently dropped.
                        result["cells"].append(dict(directory=str(directory), width=width,
                                                    repetition=repetition + 1, accepted=False,
                                                    failure=str(error), timing_scope="fresh native CLI process"))
                        result["failed_cells"] = sum(1 for row in result["cells"]
                                                      if row.get("accepted") is False)
                        result["completed_cells"] = len(result["cells"]) - result["failed_cells"]
                        result["failure_denominator"] = result["expected_cells"]
                        research.write(args.output / "summary.json", result)
                        raise
                    result["cells"].append(dict(directory=str(directory), width=width,
                                                repetition=repetition + 1, accepted=True, process=process,
                                                plan=plan, **value))
                    research.write(args.output / "summary.json", result)
                    print(f"completed fresh native r{repetition + 1} t{width}", flush=True)
            for width in widths:
                rows = [x for x in result["cells"] if x["width"] == width]
                fresh = [x["solves"][0] for x in rows]
                actual_partitions = [x["partitions"] for x in fresh]
                if len(set(actual_partitions)) != 1:
                    raise RuntimeError(f"inconsistent native partition count for width {width}")
                result["summary"][str(width)] = dict(
                    native_seconds=statistics.median(x["native_seconds"] for x in fresh),
                    api_seconds=statistics.median(x["api_seconds"] for x in fresh),
                    load_seconds=statistics.median(x["load_seconds"] for x in fresh),
                    iterations=[x["iterations"] for x in fresh], repetitions=len(rows),
                    timing_scope="one fresh native CLI process per sample",
                    partitions=actual_partitions[0],
                    requested_partitions=config.get("partitions", 1),
                    peak_group_rss_mib=max(x["process"].get("peak_rss_mib") or 0 for x in rows))
            if 1 in widths and not args.profile:
                t1 = result["summary"]["1"]["native_seconds"]
                for width in widths:
                    row = result["summary"][str(width)]
                    row["speedup"] = t1 / row["native_seconds"]
                    row["efficiency"] = row["speedup"] / width
            for width in widths:
                result["summary"][str(width)]["speed_credit"] = not args.profile
            result["accepted"] = True
    except Exception as exc:
        result["error"] = str(exc)
    finally:
        result["failed_cells"] = sum(1 for row in result["cells"] if row.get("accepted") is False)
        result["completed_cells"] = len(result["cells"]) - result["failed_cells"]
        result["failure_denominator"] = result["expected_cells"]
        try:
            after = identity(config)
            research.write(args.output / "identity-after.json", after)
            result["immutable"] = before == after
        except Exception as error:
            result["immutable"] = False
            result["identity_error"] = str(error)
        result["accepted"] &= result["immutable"]
        research.write(args.output / "summary.json", result)
    return 0 if result["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
