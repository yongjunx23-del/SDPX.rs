#!/usr/bin/env python3
"""Adapt the retained Rust benchmark in a disposable build directory.

The numerical oracle and fresh-solve timing scopes are reused unchanged. Only
solver imports, source receipts and optional untimed phase output are adapted.
"""
import argparse,hashlib,json,subprocess,tomllib
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--reference',type=Path,required=True)
p.add_argument('--source',type=Path,required=True)
p.add_argument('--identity',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.source=a.source.resolve();a.reference=a.reference.resolve()
identity=json.loads(a.identity.read_text())
assert all(hashlib.sha256((a.source/k).read_bytes()).hexdigest()==v for k,v in identity['files'].items())
a.output.mkdir(parents=True,exist_ok=False);(a.output/'src').mkdir()
original=(a.reference/'src/main.rs').read_text()
s=original.replace('use clarabel::','use sdpx_solver::')
s=s.replace('unwrap_or_else(|_| "controlled".into())',
            'unwrap_or_else(|_| "reference_default".into())')
s=s.replace('Benchmark harness for the *reference* Clarabel.rs solver.','SDPX core harness, adapted from the retained Clarabel.rs benchmark.')
s=s.replace('"clarabel_version": env!("PINNED_CLARABEL_VERSION")','"solver_version": "0.1.0"')
s=s.replace('"reference_revision": env!("PINNED_CLARABEL_REVISION")','"source_id": "'+identity['source_id']+'"')
s=s.replace('    settings\n}\n\n#[cfg(test)]', '''    // Match the public FFI's input policy; reduced tolerances retain defaults.
    settings.input_sparse_dropzeros = false;
    settings
}

#[cfg(test)]''')
s=s.replace('"solver_max_threads": 1,', '''"solver_max_threads": 1,
            "actual_factorization": info.linsolver.name,
            "actual_backend_threads": info.linsolver.threads,
            "actual_cone_threads": solver.cones.cone_threads(),''')
s=s.replace('\\"impl\\":\\"clarabel.rs\\",\\"source\\":\\"local reference crate\\"','\\"impl\\":\\"SDPX\\",\\"source\\":\\"frozen Rust core\\"')
needle='    let sol = &solver.solution;'
assert s.count(needle)==1
s=s.replace(needle,'''    if std::env::var_os("SDPX_BENCH_PHASE_TIMERS").is_some() {
        if let Some(timers) = &solver.timers { timers.print(); }
    }
'''+needle)
(a.output/'src/main.rs').write_text(s)
manifest='''[package]
name="sdpx-kernel-benchmark"
version="0.1.0"
edition="2021"
[dependencies]
sdpx-solver={path=SOLVER_PATH,features=["sdp-accelerate","faer-sparse","serde"]}
serde={version="1",features=["derive"]}
serde_json="1"
[profile.release]
debug=false
'''.replace('SOLVER_PATH',json.dumps(str(a.source/'crates/solver')))
(a.output/'Cargo.toml').write_text(manifest)
subprocess.run(['cargo','generate-lockfile','--offline','--manifest-path',str(a.output/'Cargo.toml')],check=True)
original_lock=tomllib.loads((a.source/'Cargo.lock').read_text())
locked=tomllib.loads((a.output/'Cargo.lock').read_text())
keys=lambda d:(d['name'],d['version'],d.get('source'),d.get('checksum'))
known={keys(d) for d in original_lock['package']}
assert all(keys(d) in known for d in locked['package'] if 'source' in d), 'provider version differs from frozen candidate lock'
receipt={'source_id':identity['source_id'],'reference_source':str(a.reference),'reference_main_sha256':hashlib.sha256(original.encode()).hexdigest(),'adapted_main_sha256':hashlib.sha256(s.encode()).hexdigest(),'cargo_lock_sha256':hashlib.sha256((a.output/'Cargo.lock').read_bytes()).hexdigest(),'profile':'same release profile as retained reference harness: debug=false; default opt-level=3, no LTO','scope':'direct Rust setup+solve; Julia frontend and runtime excluded; oracle and input/tolerances unchanged'}
(a.output/'identity.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(a.output)
