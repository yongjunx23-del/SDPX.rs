#!/usr/bin/env python3
"""Bounded Float64 reference receipts; no cross-solver promotion or holdout access."""
import argparse
import importlib.util
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('research_reference_common', HERE / 'run.py')
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)


def identity(args, env):
    workspace = args.workspace.resolve()
    suite = HERE.parent / 'float64/adapters/providers.py'
    providers = {str(p.resolve()): common.sha(p) for p in args.provider_file}
    executable = args.python if args.engine == 'mosek' else args.binary
    if executable is None or not executable.is_absolute() or not executable.is_file():
        raise ValueError('reference executable must be an existing absolute path')
    if args.engine == 'mosek':
        source = [suite, HERE.parent / 'float64/adapters/mosek_runner.py']
        packages = common.load_module('reference_provider_probe', suite).python_provider_fingerprint(str(executable))
    else:
        if not args.source.is_dir() or any(not (args.source / f).is_file() for f in ('Cargo.toml', 'Cargo.lock')):
            raise ValueError('Clarabel source must include Cargo.toml and Cargo.lock')
        source = [p for p in args.source.rglob('*') if p.is_file()
                  and not {'.git', 'target', '__pycache__'} & set(p.relative_to(args.source).parts)]
        packages = {'complete': True, 'scope': 'supplied native provider files; Rust dependencies pinned by source lockfile'}
    files = {str(p.resolve()): common.sha(p) for p in source}
    return dict(source_sha256=common.digest(files), source_files=files,
                artifact_sha256=common.sha(executable), executable=str(executable.resolve()),
                provider_files=providers, python_providers=packages,
                environment_sha256=common.digest(env), harness_sha256=common.harness_identity(),
                host_id=common.host_identity())


def dataset(data):
    manifest = common.read(data / 'manifest.json')
    if manifest['suite']['role'] == 'holdout':
        raise ValueError('reference v1 refuses holdout access')
    common.validate_catalog(manifest)
    common.validate_stage(manifest, manifest['suite']['role'])
    entries = common.validate_cases(manifest, 53)
    reserved = common.read(HERE / 'catalog.json')['cases']
    held = {c['json_sha256'] for c in reserved if 'holdout' in c['roles']}
    for entry in entries:
        if entry.get('runner', 'float64') != 'float64' or entry['json_sha256'] in held:
            raise ValueError('only non-holdout materialized Float64 inputs are supported')
        name = entry['name']
        if not name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in name):
            raise ValueError('unsafe case name')
        path = (data / entry['json_path']).resolve()
        if not common._is_relative_to(path, data) or common.sha(path) != entry['json_sha256']:
            raise ValueError('input path or hash mismatch: ' + name)
    return manifest, entries


def run(args):
    if args.threads != 1:
        raise ValueError('reference engines support only threads=1')
    data, output = args.data.resolve(), args.output.resolve()
    manifest, entries = dataset(data)
    protected = [args.workspace.resolve(), data]
    if args.source:
        protected.append(args.source.resolve())
    if any(common._is_relative_to(output, p) for p in protected):
        raise ValueError('output must be outside workspace, source and dataset')
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    env.update({k: '1' for k in common.THREAD_ENV})
    env.update(SDPX_RUST_ARM='reference_default', RAYON_NUM_THREADS='1', SDPX_BENCH_THREADS='1', SDPX_THREADS='1')
    # Record relevant configuration, never license values or unrelated secrets.
    receipt_env = {k: v for k, v in env.items() if k in common.THREAD_ENV or k in
                   ('SDPX_RUST_ARM', 'RAYON_NUM_THREADS', 'SDPX_BENCH_THREADS', 'SDPX_THREADS') or
                   k.startswith(('LD_', 'DYLD_', 'PYTHON'))}
    result = dict(schema_version=1, engine=args.engine, precision_bits=53, threads=1, blas_threads=1,
                  external_tolerance=1e-6, runs=4, timeout_s=900, memory_limit_mib=4096,
                  comparison_scope='separate product-default reference receipts; no cross-solver promotion',
                  binary_source_binding='supplied source and executable are independently hashed; no inferred build attestation',
                  provider_scope='MOSEK installed package files; Clarabel source lockfile and explicitly supplied native files',
                  warm_selection='exact repetitions 1,2,3; never filter failed runs',
                  expected_cases=[e['name'] for e in entries], denominator=len(entries),
                  environment=receipt_env, manifest_sha256=common.sha(data / 'manifest.json'),
                  input_sha256={e['name']: e['json_sha256'] for e in entries}, rows=[], passed=False)
    with common.slot(args.state):
        blocked = False
        try:
            result['before'] = identity(args, receipt_env)
            if not result['before']['python_providers']['complete']:
                raise ValueError('reference provider unavailable or fingerprint incomplete')
            owned = common.supervisor()
            command = ([str(args.binary.resolve())] if args.engine == 'clarabel' else
                       [str(args.python.resolve()), str(HERE.parent / 'float64/adapters/mosek_runner.py')])
            for entry in entries:
                row = dict(case_id=entry['name'], passed=False, raw_rows=[])
                result['rows'].append(row)
                if blocked:
                    row.update(error='prior process cleanup unconfirmed', not_run=True)
                    continue
                prefix = output / entry['name']
                path = (data / entry['json_path']).resolve()
                if common.sha(path) != entry['json_sha256']:
                    row.update(error='input changed before launch', not_run=True)
                    continue
                cmd = command + [str(path), '--runs=4', '--tol=1e-6']
                with prefix.with_suffix('.stdout').open('w') as stdout, prefix.with_suffix('.stderr').open('w') as stderr:
                    process = owned(cmd, env=env, stdout=stdout, stderr=stderr, timeout=900, memory_limit_mib=4096)
                row.update(command=cmd, process=process)
                blocked = process.get('cleanup_confirmed') is not True
                for line in prefix.with_suffix('.stdout').read_text().splitlines():
                    try:
                        row['raw_rows'].append(json.loads(line, parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v))))
                    except ValueError:
                        row.setdefault('non_json_lines', []).append(line)
                matches = [r for r in row['raw_rows'] if isinstance(r, dict) and r.get('instance') == path.stem]
                aggregate = matches[0] if len(matches) == 1 else {}
                runs = aggregate.get('runs', [])
                row['passed'] = (process.get('process_exit_code') == 0 and not process.get('incomplete') and not blocked
                                 and aggregate.get('pass') is True and isinstance(runs, list) and len(runs) == 4
                                 and all(isinstance(r, dict) and r.get('pass') is True for r in runs))
                common.write(output / 'reference_summary.json', result)
            if blocked:
                raise ValueError('cleanup unconfirmed; no further provider probe permitted')
            result['after'] = identity(args, receipt_env)
            result['identity_unchanged'] = (result['before'] == result['after'] and
                result['manifest_sha256'] == common.sha(data / 'manifest.json') and
                all(common.sha(data / e['json_path']) == e['json_sha256'] for e in entries))
            result['passed'] = result['identity_unchanged'] and all(r['passed'] for r in result['rows'])
        except Exception as error:
            result['error'] = type(error).__name__ + ': ' + str(error)
        recorded = {r['case_id'] for r in result['rows']}
        result['rows'] += [dict(case_id=e['name'], passed=False, not_run=True,
                                error=result.get('error', 'not completed')) for e in entries if e['name'] not in recorded]
        common.write(output / 'reference_summary.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'workspace', 'output', 'state'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--engine', choices=('clarabel', 'mosek'), required=True)
    p.add_argument('--binary', type=Path)
    p.add_argument('--source', type=Path)
    p.add_argument('--python', type=Path)
    p.add_argument('--provider-file', type=Path, action='append', default=[])
    p.add_argument('--threads', type=int, choices=(1,), default=1)
    args = p.parse_args()
    if args.engine == 'clarabel' and (args.binary is None or args.source is None):
        p.error('Clarabel requires --binary and --source')
    if args.engine == 'mosek' and args.python is None:
        p.error('MOSEK requires --python')
    try:
        return 0 if run(args)['passed'] else 1
    except (OSError, ValueError, KeyError) as error:
        p.exit(1, str(error) + '\n')


if __name__ == '__main__':
    raise SystemExit(main())
