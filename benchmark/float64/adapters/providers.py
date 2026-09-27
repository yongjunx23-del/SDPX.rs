"""Fingerprint reference distributions without importing solver runtimes."""
import json
import subprocess

def python_provider_fingerprint(executable):
    """Hash installed distributions using the selected interpreter, without importing solvers."""
    probe = r"""
import hashlib, importlib.metadata as metadata, json, pathlib, sys
receipt = {"executable": str(pathlib.Path(sys.executable).resolve()), "python": sys.version, "packages": {}, "complete": True}
for name in ("mosek", "numpy", "scipy"):
    try:
        distribution = metadata.distribution(name)
        files = {}
        for entry in distribution.files or []:
            path = pathlib.Path(distribution.locate_file(entry)).resolve()
            if path.is_file() and path.suffix != ".pyc" and "__pycache__" not in path.parts:
                h = hashlib.sha256()
                with path.open("rb") as stream:
                    for block in iter(lambda: stream.read(1048576), b""):
                        h.update(block)
                files[str(path)] = h.hexdigest()
        if not files:
            raise ValueError("distribution has no inspectable installed files")
        receipt["packages"][name] = {"version": distribution.version, "files": files}
    except Exception as err:
        receipt["complete"] = False
        receipt["packages"][name] = {"error": type(err).__name__ + ": " + str(err)}
print(json.dumps(receipt, sort_keys=True))
"""
    try:
        result = subprocess.run([executable, "-c", probe], text=True, capture_output=True, timeout=60)
        if result.returncode:
            return {"complete": False, "exit_code": result.returncode, "error": result.stderr}
        return json.loads(result.stdout)
    except (OSError, subprocess.TimeoutExpired, ValueError) as err:
        return {"complete": False, "error": f"{type(err).__name__}: {err}"}

