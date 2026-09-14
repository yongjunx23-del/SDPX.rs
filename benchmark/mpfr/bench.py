#!/usr/bin/env python3
"""Prepare a frozen-copy harness, or explicitly build/run one. Standard library only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

HERE=Path(__file__).resolve().parent
MODULE='crates/solver/src/mpfr_dense_benchmark.rs'
LIB='crates/solver/src/lib.rs'
DECL='\n#[cfg(all(test, feature = "sdp"))]\nmod mpfr_dense_benchmark;\n'
ENV_RECEIPT_KEYS = (
    'CARGO_HOME', 'RUSTUP_HOME', 'RUSTUP_TOOLCHAIN', 'CARGO_TARGET_DIR', 'RUSTC',
    'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS',
    'OMP_NUM_THREADS', 'RAYON_NUM_THREADS', 'SDPX_MPFR_OUTPUT', 'SDPX_MPFR_REPEATS',
)


def run_owned(command, *, timeout, status_path, **kwargs):
    # Separate POSIX session: a timeout stops only this invocation and its
    # compiler/test children, never a concurrent benchmark's process group.
    if os.name != 'posix': raise ValueError('owned process-group execution requires POSIX')
    started = time.monotonic()
    process = subprocess.Popen(command, start_new_session=True, **kwargs)
    state = 'interrupted'
    try:
        code = process.wait(timeout=timeout)
        state = 'passed' if code == 0 else 'failed'
        if code: raise subprocess.CalledProcessError(code, command)
    except BaseException as error:
        if isinstance(error, subprocess.TimeoutExpired): state = 'timeout'
        if state in ('timeout', 'interrupted'):
            try: os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError: pass
            process.wait(timeout=10)
        raise
    finally:
        save(status_path, dict(command=command, timeout_seconds=timeout,
             elapsed_seconds=time.monotonic()-started, status=state,
             returncode=process.returncode))


def sha(data): return hashlib.sha256(data).hexdigest()
def tree(root):
    result={}
    for p in sorted(root.rglob('*')):
        if p.is_symlink(): raise ValueError(f'symlink unsupported: {p}')
        if p.is_file(): result[p.relative_to(root).as_posix()]=sha(p.read_bytes())
        elif not p.is_dir(): raise ValueError(f'nonregular entry: {p}')
    return result

def save(path,data):
    with path.open('x') as f: json.dump(data,f,indent=2,sort_keys=True); f.write('\n')

def prepare(source,out):
    source=source.resolve(strict=True)
    if out.is_symlink(): raise ValueError('output symlink is unsupported')
    out=out.resolve()
    if out.exists() or out.is_relative_to(source) or source.is_relative_to(out):
        raise ValueError('output must be absent and disjoint from source')
    original=tree(source)
    text=(source/LIB).read_text()
    if 'mod mpfr_dense_benchmark' in text or (source/MODULE).exists(): raise ValueError('already instrumented')
    out.mkdir(parents=True,exist_ok=False)
    target=out/'source'; shutil.copytree(source,target,symlinks=True)
    if tree(target)!=original or tree(source)!=original: raise ValueError('source changed during copy')
    (target/LIB).write_text(text+DECL)
    (target/MODULE).write_bytes((HERE/'dense.rs').read_bytes())
    expected=dict(original); expected[LIB]=sha((text+DECL).encode()); expected[MODULE]=sha((HERE/'dense.rs').read_bytes())
    if tree(target)!=expected: raise ValueError('unexpected generated changes')
    if tree(source)!=original: raise ValueError('source changed during harness insertion')
    save(out/'prepare.json',dict(original_source=str(source),original_files=original,harness_files=expected,
        cargo_lock_sha256=original['Cargo.lock'],harness_sha256=expected[MODULE],
        generator_sha256=sha(Path(__file__).read_bytes()),
        purpose='MPFR provider microbenchmark; not end-to-end solver gain'))
    print(out/'prepare.json')

def run(out,cargo,rustc,features,repeats,build_timeout=1800,test_timeout=900):
    if not (0<build_timeout<86400 and 0<test_timeout<86400): raise ValueError('timeouts must be positive and below one day')
    if repeats<7 or repeats%2!=1: raise ValueError('odd repeats >=7 required')
    receipt=json.loads((out/'prepare.json').read_text()); source=out/'source'
    if tree(source)!=receipt['harness_files']: raise ValueError('prepared source changed')
    work=out/'run'; work.mkdir(exist_ok=False)
    env=os.environ.copy()
    for k in ['OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','RAYON_NUM_THREADS']:
        env[k]='1'
    env.update(CARGO_TARGET_DIR=str(work/'target'),RUSTC=rustc)
    command=[cargo,'test','-p','sdpx-solver','--lib','--no-run','--release','--locked','--offline',
             '--no-default-features','--features',features,'--message-format=json']
    compiler=subprocess.check_output([rustc,'-vV'],env=env,text=True,timeout=30)
    cargo_version=subprocess.check_output([cargo,'-V'],env=env,text=True,timeout=30)
    with (work/'build.jsonl').open('x') as stdout,(work/'build.stderr').open('x') as stderr:
        run_owned(command,timeout=build_timeout,status_path=work/'build-status.json',cwd=source,env=env,stdout=stdout,stderr=stderr)
    artifacts=[]
    for line in (work/'build.jsonl').read_text().splitlines():
        try: obj=json.loads(line)
        except json.JSONDecodeError: continue
        if obj.get('reason')=='compiler-artifact' and obj.get('executable') and obj['target']['name']=='sdpx_solver':
            artifacts.append(obj['executable'])
    if len(artifacts)!=1: raise ValueError(f'expected one solver test binary, found {artifacts}')
    binary=Path(artifacts[0]); binary_hash=sha(binary.read_bytes())
    output=work/'values'; output.mkdir()
    env.update(SDPX_MPFR_OUTPUT=str(output),SDPX_MPFR_REPEATS=str(repeats))
    test=[str(binary),'mpfr_dense_benchmark::dense_microbenchmark','--exact','--ignored','--nocapture','--test-threads=1']
    save(work/'execution.json',dict(build_command=command,test_command=test,compiler=compiler,cargo=cargo_version,
        rustc_path=rustc,rustc_sha256=sha(Path(rustc).read_bytes()),cargo_path=cargo,cargo_sha256=sha(Path(cargo).read_bytes()),
        binary_sha256=binary_hash,features=features,repeats=repeats,
        build_timeout_seconds=build_timeout,test_timeout_seconds=test_timeout,
        environment={k:env[k] for k in ENV_RECEIPT_KEYS if k in env}))
    with (work/'raw.log').open('x') as log:
        run_owned(test,timeout=test_timeout,status_path=work/'test-status.json',env=env,stdout=log,stderr=subprocess.STDOUT)
    if tree(source)!=receipt['harness_files'] or sha(binary.read_bytes())!=binary_hash: raise ValueError('source/binary changed')
    values=tree(output)
    metrics=[json.loads(line.split('MPFR_DENSE ',1)[1]) for line in (work/'raw.log').read_text().splitlines() if 'MPFR_DENSE ' in line]
    if len(metrics)!=90 or len(values)!=144: raise ValueError('incomplete kernel/input/output set')
    save(work/'metrics.json',metrics)
    save(work/'results.json',dict(decimal_files_sha256=values,raw_log_sha256=sha((work/'raw.log').read_bytes()),
        input_files={p:h for p,h in values.items() if not p.endswith('_output.txt')},
        output_files={p:h for p,h in values.items() if p.endswith('_output.txt')},
        comparison='Require identical input/output file hashes for assignment-only candidate; timings separate from correctness'))
    print(work/'results.json')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='mode',required=True)
    a=sub.add_parser('prepare'); a.add_argument('--source',type=Path,required=True); a.add_argument('--output',type=Path,required=True)
    a=sub.add_parser('run'); a.add_argument('--output',type=Path,required=True)
    a.add_argument('--cargo',required=True); a.add_argument('--rustc',required=True)
    a.add_argument('--features',required=True,help='existing native provider feature, e.g. sdp-accelerate or sdp-openblas')
    a.add_argument('--repeats',type=int,default=9)
    a.add_argument('--build-timeout',type=int,default=1800)
    a.add_argument('--test-timeout',type=int,default=900)
    a=p.parse_args()
    if a.mode=='prepare': prepare(a.source,a.output)
    else:
        cargo=str(Path(a.cargo).resolve(strict=True)); rustc=str(Path(a.rustc).resolve(strict=True))
        run(a.output.resolve(strict=True),cargo,rustc,a.features,a.repeats,a.build_timeout,a.test_timeout)
