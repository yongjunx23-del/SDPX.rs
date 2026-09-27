#!/usr/bin/env python3
"""Per-change end-to-end check for SDPX: build, solve a pinned case, audit.

    e2e.py build [--profile fast|release] [--arm NAME]
    e2e.py run CASE [--arm NAME | --cli PATH] [--threads N]
    e2e.py ab CASE ARM_A ARM_B [--order ABBA] [--threads N]
    e2e.py list

Cases and their gates live in cases.json. Inputs are committed under data/
and unpacked on first use. Frozen executables ("arms"), run outputs and the
append-only journal live in $SDPX_E2E_HOME (default ~/.cache/sdpx-e2e), never
in the repository or /tmp. Timing is one fresh CLI process per solve; the
independent original-coordinate audit runs afterwards, outside every timer.
"""
import argparse
import datetime
import hashlib
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tarfile
import time
import gzip
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
HOME = Path(os.environ.get("SDPX_E2E_HOME", Path.home() / ".cache" / "sdpx-e2e"))
CASES = json.loads((HERE / "cases.json").read_text())
TOOLCHAIN = Path.home() / ".local/share/sdpx-toolchain"
THREAD_VARS = ("RAYON_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS")

sys.path.insert(0, str(HERE.parent / "research"))
sys.path.insert(0, str(HERE.parent / "ising"))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git(*args):
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True).stdout.strip()


def cargo_env():
    env = dict(os.environ)
    if "CARGO_HOME" not in env and (TOOLCHAIN / "cargo").is_dir():
        env["CARGO_HOME"] = str(TOOLCHAIN / "cargo")
        env["RUSTUP_HOME"] = str(TOOLCHAIN / "rustup")
        env["PATH"] = f"{TOOLCHAIN / 'cargo/bin'}:{env['PATH']}"
    return env


# ---------------------------------------------------------------- build

def build(args):
    blas = "sdp-accelerate" if platform.system() == "Darwin" else "sdp-openblas"
    cmd = ["cargo", "build", "--locked", "--offline", "--profile", args.profile,
           "-p", "sdpx-solver", "--bin", "sdpx", "--features", f"{blas},faer-sparse"]
    subprocess.run(cmd, cwd=REPO, env=cargo_env(), check=True)
    commit = git("rev-parse", "--short", "HEAD")
    diff = git("diff", "HEAD")
    dirty = hashlib.sha256(diff.encode()).hexdigest()[:8] if diff else ""
    name = args.arm or f"{commit}{'-' + dirty if dirty else ''}-{args.profile}"
    arm = HOME / "arms" / name
    if arm.exists() and not args.force:
        sys.exit(f"arm {name} exists; pass --force or --arm NEW_NAME")
    arm.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPO / "target" / args.profile / "sdpx", arm / "sdpx")
    (arm / "arm.json").write_text(json.dumps({
        "commit": git("rev-parse", "HEAD"), "dirty_diff_sha256": dirty or None,
        "profile": args.profile, "features": [blas, "faer-sparse"],
        "built": now(), "sdpx_sha256": sha256(arm / "sdpx")}, indent=2))
    if diff:
        (arm / "dirty.patch").write_text(diff)
    print(name)


# ---------------------------------------------------------------- inputs

def materialize(name):
    case = CASES[name]
    src = HERE / case["input"]
    dest = HOME / "data" / name
    stamp = dest / ".sha256"
    if not (stamp.exists() and stamp.read_text() == case["input_sha256"]):
        shutil.rmtree(dest, ignore_errors=True)
        dest.mkdir(parents=True)
        if case["kind"] == "file":
            target = dest / "input.json"
            with gzip.open(src) as fin, open(target, "wb") as fout:
                shutil.copyfileobj(fin, fout)
            got = sha256(target)
        else:
            with tarfile.open(src) as tar:
                tar.extractall(dest, filter="data")
            import scaling
            got = scaling.sampled_input_hash(next(p for p in dest.iterdir() if p.is_dir()))
        if got != case["input_sha256"]:
            shutil.rmtree(dest)
            sys.exit(f"{name}: input hash {got} != pinned {case['input_sha256']}")
        stamp.write_text(case["input_sha256"])
    if case["kind"] == "file":
        return dest / "input.json"
    return next(p for p in dest.iterdir() if p.is_dir())


# ---------------------------------------------------------------- run + audit

def now():
    return datetime.datetime.now().isoformat(timespec="seconds")


def resolve_cli(args):
    if args.cli:
        return Path(args.cli).resolve(), "cli:" + str(Path(args.cli).resolve())
    if args.arm:
        path = HOME / "arms" / args.arm / "sdpx"
        if not path.exists():
            sys.exit(f"unknown arm {args.arm}; see `e2e.py list`")
        return path, args.arm
    path = REPO / "target" / "fast" / "sdpx"
    if not path.exists():
        sys.exit("no --arm/--cli given and target/fast/sdpx missing; run `e2e.py build`")
    return path, "working-tree:target/fast"


def point_hash(result):
    point = {k: result.get(k) for k in ("x", "s", "z")}
    return hashlib.sha256(json.dumps(point, sort_keys=True).encode()).hexdigest()[:16]


def audit(name, case, input_path, result_path, out_dir):
    spec = case["audit"]
    if spec["kind"] == "python":
        import native
        a = native.audit(json.loads(input_path.read_text()),
                         json.loads(result_path.read_text()), spec["tolerance"])
        keep = ("pass", "r_p", "r_d", "gap", "dist_K_s", "dist_Kstar_z", "tol_feas", "tol_dual")
        summary = {k: a[k] for k in keep}
        (out_dir / "audit.json").write_text(json.dumps(a, indent=2, default=str))
        return summary
    julia = os.environ.get("SDPX_E2E_JULIA", shutil.which("julia") or "julia")
    cmd = [julia, "--startup-file=no", "-t1", "--gcthreads=1",
           f"--project={HERE / 'audit-env'}", str(HERE.parent / "ising" / "audit_point.jl"),
           str(input_path), str(result_path), str(out_dir / "audit.json"),
           str(case["bits"]), str(HERE / spec["reference"])]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    (out_dir / "audit.log").write_text(proc.stdout + proc.stderr)
    if not (out_dir / "audit.json").exists():
        return {"pass": False, "error": (proc.stderr or proc.stdout).strip().splitlines()[-1:]}
    a = json.loads((out_dir / "audit.json").read_text())
    fields = ("primal", "dual", "gap", "primal_psd", "dual_psd", "reference_objective_agreement")
    summary = {"pass": a.get("accepted") is True}
    summary.update({k: float(a[k]) for k in fields if k in a})
    return summary


def solve(name, cli, label, threads):
    case = CASES[name]
    input_path = materialize(name)
    threads = threads or case["threads"]
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    out = HOME / "runs" / f"{stamp}-{name}-{label.replace('/', '_').replace(':', '_')}"
    out.mkdir(parents=True)
    settings = out / "settings.json"
    settings.write_text(json.dumps(case["settings"], indent=2))
    result_path = out / "result.json"
    env = dict(os.environ, **{v: str(threads) for v in THREAD_VARS})
    cmd = [str(cli), str(input_path), "--settings", str(settings), "--threads", str(threads),
           "--output", str(result_path), "--quiet"]
    if case["bits"] != 53:
        cmd[2:2] = ["--precision", str(case["bits"])]
    t0 = time.perf_counter()
    with open(out / "stderr.log", "w") as err:
        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=err)
        _, status, usage = os.wait4(proc.pid, 0)
    wall = time.perf_counter() - t0
    code = os.waitstatus_to_exitcode(status)
    rss_mib = usage.ru_maxrss / (1 << 20 if platform.system() == "Darwin" else 1 << 10)
    row = {"time": now(), "case": name, "arm": label, "threads": threads, "exit": code,
           "wall_seconds": round(wall, 3), "rss_mib": round(rss_mib, 1),
           "cli_sha256": sha256(cli), "out": str(out)}
    if result_path.exists():
        r = json.loads(result_path.read_text())
        row.update(status=r["status"], iterations=r["iterations"],
                   api_seconds=r.get("api_seconds"), native_seconds=r.get("native_seconds"),
                   objective=str(r.get("objective"))[:24], point=point_hash(r))
        row["audit"] = audit(name, case, input_path, result_path, out)
    else:
        row["audit"] = {"pass": False, "error": "no result written"}
    (out / "row.json").write_text(json.dumps(row, indent=2))
    HOME.mkdir(parents=True, exist_ok=True)
    with open(HOME / "journal.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
    return row


def fmt(row):
    a = row["audit"]
    gates = " ".join(f"{k}={v:.2e}" for k, v in a.items()
                     if isinstance(v, float) and k not in ("tol_feas", "tol_dual"))
    api = row.get("api_seconds")
    return (f"{row['case']:8} {row['arm'][:34]:34} {row.get('status', 'ERROR'):13} "
            f"it={row.get('iterations', '-'):>3} api={api if api is None else f'{api:.3f}'}s "
            f"wall={row['wall_seconds']:.2f}s audit={'PASS' if a.get('pass') else 'FAIL'} {gates}")


def run(args):
    cli, label = resolve_cli(args)
    row = solve(args.case, cli, label, args.threads)
    print(fmt(row))
    known = CASES[args.case].get("known_failure")
    if not row["audit"].get("pass") and known:
        print(f"known failure: {known}")
    print(f"output: {row['out']}")
    sys.exit(0 if row["audit"].get("pass") else 1)


def ab(args):
    arms = {"A": args.arm_a, "B": args.arm_b}
    rows = {"A": [], "B": []}
    for key in args.order:
        a = argparse.Namespace(cli=None, arm=arms[key])
        cli, label = resolve_cli(a)
        row = solve(args.case, cli, label, args.threads)
        rows[key].append(row)
        print(f"[{key}] {fmt(row)}", flush=True)
    med = {k: statistics.median(r["api_seconds"] for r in v if r.get("api_seconds") is not None)
           for k, v in rows.items() if v}
    points = {r.get("point") for v in rows.values() for r in v}
    print(f"\nmedian api_seconds  A={med['A']:.3f}  B={med['B']:.3f}  "
          f"B/A={med['B'] / med['A']:.3f} ({(med['B'] / med['A'] - 1) * 100:+.1f}%)")
    print(f"points identical across all runs: {'yes' if len(points) == 1 else 'no'}")
    print(f"audits: A {sum(r['audit'].get('pass', False) for r in rows['A'])}/{len(rows['A'])} "
          f"B {sum(r['audit'].get('pass', False) for r in rows['B'])}/{len(rows['B'])}")


def list_(_):
    for name, case in CASES.items():
        print(f"{name:10} {case['description']}")
    arms = sorted((HOME / "arms").glob("*/arm.json")) if (HOME / "arms").exists() else []
    print(f"\narms in {HOME / 'arms'}:")
    for a in arms:
        meta = json.loads(a.read_text())
        print(f"  {a.parent.name:40} {meta['profile']:8} built {meta['built']}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="build the CLI and freeze it as a named arm")
    b.add_argument("--profile", default="fast", choices=("fast", "release"))
    b.add_argument("--arm")
    b.add_argument("--force", action="store_true")
    b.set_defaults(fn=build)
    r = sub.add_parser("run", help="solve and audit one pinned case")
    r.add_argument("case", choices=CASES)
    r.add_argument("--arm")
    r.add_argument("--cli")
    r.add_argument("--threads", type=int)
    r.set_defaults(fn=run)
    c = sub.add_parser("ab", help="interleaved A/B comparison of two frozen arms")
    c.add_argument("case", choices=CASES)
    c.add_argument("arm_a")
    c.add_argument("arm_b")
    c.add_argument("--order", default="ABBA")
    c.add_argument("--threads", type=int)
    c.set_defaults(fn=ab)
    sub.add_parser("list", help="list cases and frozen arms").set_defaults(fn=list_)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
