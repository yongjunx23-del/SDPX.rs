#!/usr/bin/env python3
"""Sequential, source-bound experiments; reuse the existing solver/oracle drivers."""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
THREAD_ENV = ('OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS',
              'OMP_NUM_THREADS', 'NUMEXPR_NUM_THREADS')


def _is_relative_to(path, base):
    # Path.is_relative_to is 3.9+; retain the relative-path fallback for
    # supported cluster Python 3.6 launchers.
    try:
        Path(path).relative_to(base)
        return True
    except ValueError:
        return False


def read(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + '\n')


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def settings_identity(settings, runtime):
    numerical = {k: v for k, v in settings.items() if k not in ('verbose', 'outputs')} if isinstance(settings, dict) else settings
    # Executables are deliberately different between arms. Keep their identity
    # in the arm/row receipts, and validate outcomes separately from settings.
    comparable = {k: v for k, v in runtime.items()
                  if k not in ('cli', 'cli_sha256', 'version', 'status')}
    return digest({'numerical': numerical, 'runtime': comparable})


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def native_helpers():
    return load_module('sdpx_research_native', HERE / 'native.py')


def precision_oracle(config, input_path, result_path, bits, timeout, log_path):
    """Run the independent Julia oracle for MPFR points.

    The oracle receives only JSON input and the native result.  It never loads
    the SDPX Julia package and its process wall time is outside solve timing.
    """
    spec = config.get('oracle', {})
    if spec and not isinstance(spec, dict):
        return {'pass': False, 'finite': False, 'status': 'Error',
                'error': 'oracle must be an object'}
    julia = (spec.get('julia') if isinstance(spec, dict) else None) or os.environ.get('JULIA') or shutil.which('julia')
    if not julia:
        return {'pass': False, 'finite': False, 'status': 'Error',
                'error': 'MPFR audit requires an independent Julia oracle executable'}
    julia_path = Path(julia)
    if julia_path.is_absolute() and not julia_path.is_file():
        return {'pass': False, 'finite': False, 'status': 'Error',
                'error': f'oracle Julia executable is not a file: {julia_path}'}
    command = [str(julia_path), '--startup-file=no']
    project = spec.get('project') if isinstance(spec, dict) else None
    if project:
        project_path = Path(project).resolve()
        if not project_path.is_dir():
            return {'pass': False, 'finite': False, 'status': 'Error',
                    'error': f'oracle project is not a directory: {project_path}'}
        command.append('--project=' + str(project_path))
    command += [str(HERE / 'native_oracle.jl'), str(input_path), str(result_path), str(bits)]
    try:
        proc = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              universal_newlines=True, timeout=min(float(timeout), 120.0))
    except (OSError, subprocess.SubprocessError) as error:
        return {'pass': False, 'finite': False, 'status': 'Error', 'error': str(error)}
    Path(log_path).write_text(proc.stdout + proc.stderr)
    parsed = None
    for line in reversed(proc.stdout.splitlines()):
        try:
            parsed = json.loads(line)
            break
        except ValueError:
            continue
    if not isinstance(parsed, dict):
        parsed = {'pass': False, 'finite': False, 'status': 'Error',
                  'error': proc.stderr.strip() or 'oracle emitted no JSON'}
    if proc.returncode != 0:
        parsed['pass'] = False
        parsed.setdefault('error', proc.stderr.strip() or 'oracle rejected point')
    parsed['oracle_command'] = command
    return parsed


def source_files(root):
    root = Path(root).resolve()
    paths = [root / 'Cargo.toml', root / 'Cargo.lock']
    for directory in ('crates', 'include'):
        paths += [p for p in (root / directory).rglob('*') if p.is_file()
                  and not {'target', '.git', '__pycache__'} & set(p.relative_to(root).parts)]
    if not paths or any(not p.is_file() for p in paths):
        raise ValueError('source must contain the Rust workspace, lockfile and native crates')
    if any(p.is_symlink() for p in paths):
        raise ValueError('source files must be regular owned files, not symlinks')
    return {str(p.relative_to(root)): sha(p) for p in sorted(paths)}


def arm_identity(config):
    source = Path(config['source']).resolve()
    if 'cli' not in config:
        raise ValueError('native arm config requires cli (absolute sdpx executable)')
    cli = Path(config['cli'])
    if not cli.is_absolute() or not cli.is_file() or not os.access(cli, os.X_OK):
        raise ValueError('cli must name an absolute executable, not a PATH lookup')
    cli = cli.resolve()
    files = source_files(source)
    providers = {str(Path(p).resolve()): sha(p) for p in config.get('provider_files', [])}
    # A legacy FFI library may be recorded for historical receipts, but it is
    # never substituted for the native CLI identity.
    legacy_library = config.get('library')
    legacy_hash = None
    if legacy_library is not None:
        library = Path(legacy_library)
        if not library.is_absolute() or not library.is_file():
            raise ValueError('legacy library path is present but is not a regular file')
        legacy_hash = sha(library.resolve())
    effective = dict(os.environ, **config.get('env', {}))
    runtime_env = {k: v for k, v in effective.items() if k.startswith(('DYLD_', 'LD_', 'RAYON_', 'OMP_', 'MKL_', 'OPENBLAS_'))}
    oracle = config.get('oracle', {})
    if oracle and not isinstance(oracle, dict):
        raise ValueError('oracle must be an object when supplied')
    oracle_identity = {}
    for key in ('julia', 'project'):
        value = oracle.get(key) if isinstance(oracle, dict) else None
        if value:
            path = Path(value).resolve()
            if not path.exists():
                raise ValueError(f'oracle {key} path does not exist: {path}')
            oracle_identity[key] = sha(path) if path.is_file() else digest({str(p): sha(p) for p in sorted(path.rglob('*')) if p.is_file()})
    result = {'source_sha256': digest(files), 'artifact_sha256': sha(cli),
            'cli_sha256': sha(cli), 'cli': str(cli),
            'source_files': files, 'provider_files': providers,
            'environment_sha256': digest({'oracle': oracle_identity,
                                         'provider_hashes': sorted(providers.values()),
                                         'env': config.get('env', {}), 'runtime_env': runtime_env,
                                         'blas': config.get('blas', 'default'),
                                         'source_binding': config.get('source_binding', 'independent source and CLI hashes')}),
            'source_binding': config.get('source_binding', 'independent source and CLI hashes')}
    if legacy_hash is not None:
        result['legacy_library_sha256'] = legacy_hash
    if oracle_identity:
        result['oracle'] = oracle_identity
    return result


def harness_identity():
    paths = list(HERE.glob('*.py')) + list(HERE.glob('*.jl'))
    paths += list((ROOT / 'benchmark/float64/adapters').glob('*.py'))
    paths += [HERE / 'catalog.json']
    paths += [ROOT / 'benchmark/float64' / f for f in ('run.py', 'resource_probe.py', 'sdpx_runner.jl')]
    paths += [ROOT / 'benchmark/parallel' / f for f in ('orthant.jl', 'sampled.jl', 'sweep.sh')]
    paths += [HERE / 'native.py', HERE / 'native_oracle.jl']
    return digest({str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)})


def host_identity():
    return digest({'node': platform.node(), 'platform': platform.platform(),
                   'machine': platform.machine(), 'cpus': os.cpu_count(),
                   'affinity': sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None})


@contextmanager
def slot(state):
    state = Path(state)
    state.mkdir(parents=True, exist_ok=True)
    with (state / 'numerical.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def group_rss_kib(pgid):
    result = subprocess.run(['ps', '-axo', 'pgid=,rss='], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, universal_newlines=True,
                            check=True)
    total = 0
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) == 2 and int(fields[0]) == pgid:
            total += int(fields[1])
    return total


def supervisor():
    module = load_module('sdpx_owned_supervisor', ROOT / 'benchmark/float64/run.py')
    return module.with_native_highwater(module.owned_supervisor(group_rss_kib))


def config_env(config, identity, threads):
    env = dict(os.environ, **config.get('env', {}))
    for key in THREAD_ENV:
        env[key] = '1'
    env.update(SDPX_SOURCE_ID=identity['source_sha256'], SDPX_BENCH_THREADS=str(threads),
               SDPX_THREADS=str(threads), RAYON_NUM_THREADS=str(threads),
               SDPX_BENCH_PREPROCESSING='default', SDPX_BENCH_BLAS=config.get('blas', 'default'))
    # Native diagnostic scripts use this explicit binding; the command itself
    # is still passed as an argv entry so PATH lookups cannot change an arm.
    env['SDPX_CLI'] = str(Path(config['cli']).resolve())
    return env


def qualification_matches(config, identity, harness):
    if not config.get('qualification'):
        return False
    receipt = read(config['qualification'])
    return (receipt.get('passed') is True and receipt.get('identity_unchanged') is True
            and receipt.get('source_sha256') == identity['source_sha256']
            and receipt.get('artifact_sha256') == identity['artifact_sha256']
            and receipt.get('environment_sha256') == identity['environment_sha256']
            and receipt.get('harness_sha256') == harness
            and bool(receipt.get('commands')))


def validate_cases(manifest, bits):
    entries = manifest['instances']
    if not entries or len({e['name'] for e in entries}) != len(entries):
        raise ValueError('empty or duplicate case denominator')
    for entry in entries:
        runner = entry.get('runner', 'float64')
        if runner not in ('float64', 'orthant', 'sampled'):
            raise ValueError('unsupported runner: ' + runner)
        if runner == 'float64' and bits != 53:
            raise ValueError('Float64 JSON cannot silently become a high-precision fixture')
        modes = entry.get('precision_bits', [53] if runner == 'float64' else [256, 512])
        if isinstance(modes, int):
            modes = [modes]
        if bits not in modes:
            raise ValueError(f"{entry['name']} does not support requested precision")
    return entries


def validate_stage(manifest, stage):
    role = manifest['suite']['role']
    expected = 'development' if role == 'mpfr-dev' else role
    if stage != expected:
        raise ValueError(f'manifest role {role} cannot run as {stage}')
    if manifest['suite'].get('independent_oracle_tolerance') != 1e-6:
        raise ValueError('external Float64 gate must remain 1e-6')
    if manifest['suite'].get('denominator') != len(manifest['instances']):
        raise ValueError('manifest denominator mismatch')
    if any('holdout' in e.get('roles', []) for e in manifest['instances']) and stage != 'holdout':
        raise ValueError('held-out inputs cannot enter development')


def validate_catalog(manifest):
    library = load_module('sdpx_pinned_catalog', HERE / 'catalog.py')
    catalog = library.load_catalog()
    if manifest['suite'].get('catalog_sha256') != library.digest(library.encoded(catalog)):
        raise ValueError('manifest was not materialized from the current pinned catalog')
    expected = library.select_cases(catalog, manifest['suite']['role'])
    received = [{k: v for k, v in e.items() if k != 'json_path'} for e in manifest['instances']]
    if received != expected:
        raise ValueError('manifest case membership/roles differ from the pinned catalog')


def external_output(path, configs):
    path = path.resolve()
    protected = [ROOT.resolve()] + [Path(c['source']).resolve() for c in configs]
    if any(_is_relative_to(path, p) for p in protected):
        raise ValueError('experiment output must be outside project and source snapshots')


def run_case(owned, config, identity, entry, data, bits, threads, block, out, timeout, memory, repetitions=4):
    """Run one native CLI solve per sample and retain every failure slot.

    The CLI process is the numerical process.  ``api_seconds`` is the primary
    comparable time (native setup plus solve); ``native_seconds`` and the
    supervisor-measured whole-process duration are retained separately.
    """
    budget = entry.get('reservation', {}).get('budget', {})
    limits = [timeout, memory]
    for index, key in enumerate(('process_seconds', 'memory_mib')):
        if key in budget:
            value = budget[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError('invalid reserved case limit: ' + key)
            limits[index] = min(limits[index], value)
    timeout, memory = limits
    name = entry['name']
    if not name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in name):
        raise ValueError('case name must be a safe filename')
    runner = entry.get('runner', 'float64')
    prefix = out / f'{block}_{name}'
    out.mkdir(parents=True, exist_ok=True)
    env = config_env(config, identity, threads)
    cli = Path(config['cli']).resolve()
    if not cli.is_file() or not os.access(cli, os.X_OK):
        raise ValueError('cli must name a regular native executable')
    helper = native_helpers()
    if runner == 'float64':
        input_path = (data / entry['json_path']).resolve()
        input_sha = entry.get('json_sha256')
        if sha(input_path) != input_sha:
            raise ValueError('input hash changed: ' + name)
        problem = read(input_path)
    else:
        driver = ROOT / 'benchmark/parallel' / (runner + '.jl')
        if sha(driver) != entry['source']['sha256']:
            raise ValueError('recipe changed since catalog reservation: ' + name)
        problem = helper.synthetic_problem(entry, bits)
        input_path = prefix.with_name(prefix.name + '.input.json')
        if input_path.exists():
            raise ValueError('refusing to overwrite generated native input: ' + str(input_path))
        write(input_path, problem)
        input_sha = sha(input_path)
    records = []
    for index in range(repetitions):
        result_path = prefix.with_name(prefix.name + f'.run-{index}.json')
        stdout_path = prefix.with_name(prefix.name + f'.run-{index}.stdout')
        stderr_path = prefix.with_name(prefix.name + f'.run-{index}.stderr')
        process_path = prefix.with_name(prefix.name + f'.run-{index}.process.json')
        command = [str(cli), str(input_path), '--precision', str(bits), '--threads', str(threads),
                   '--quiet', '--output', str(result_path)]
        if 'psd_direction' in config:
            raise ValueError('psd_direction has been removed; delete it from config')
        settings_path = config.get('settings')
        if settings_path:
            settings_path = Path(settings_path).resolve()
            if not settings_path.is_file():
                raise ValueError('settings path is not a regular file')
            command += ['--settings', str(settings_path)]
        started = time.monotonic()
        with stdout_path.open('w') as stdout, stderr_path.open('w') as stderr:
            receipt = owned(command, env=env, stdout=stdout, stderr=stderr,
                            timeout=timeout, memory_limit_mib=memory)
        elapsed = time.monotonic() - started
        write(process_path, dict(receipt, command=command, timeout_s=timeout,
                                 memory_limit_mib=memory, cli_e2e_seconds=elapsed))
        result = {}
        error = None
        if result_path.is_file():
            try:
                result = read(result_path)
            except (OSError, ValueError, TypeError) as exc:
                error = f'invalid native result: {exc}'
        else:
            error = 'native result file missing'
        status = result.get('status', 'Error') if isinstance(result, dict) else 'Error'
        normalized_status = {'optimal': 'Optimal', 'solved': 'Solved'}.get(str(status).lower(), status)
        tolerance = 1e-6 if bits == 53 else 2.0 ** (-bits / 2.0)
        try:
            validation = helper.audit(problem, result, tolerance=tolerance)
        except (OSError, RuntimeError, TypeError, ValueError, KeyError, IndexError) as exc:
            validation = {'pass': False, 'finite': False, 'status': str(status), 'error': str(exc)}
        if bits != 53 and result_path.is_file():
            validation = precision_oracle(config, input_path, result_path, bits, timeout,
                                          prefix.with_name(prefix.name + f'.run-{index}.oracle.log'))
        runtime = dict(cli=str(cli), cli_sha256=sha(cli), version=result.get('version'),
                       precision_bits=result.get('precision_bits'), status=status,
                       linear_solver=result.get('linear_solver'),
                       linear_solver_threads=result.get('linear_solver_threads'),
                       cone_threads=result.get('cone_threads'),
                       kkt_form=result.get('kkt_form'), threads_requested=result.get('threads_requested'))
        settings = result.get('settings', {}) if isinstance(result, dict) else {}
        process_ok = (receipt.get('process_exit_code') == 0 and not receipt.get('incomplete', False)
                      and receipt.get('cleanup_confirmed') is True)
        passed = bool(process_ok and validation.get('pass') is True
                      and result.get('precision_bits') == bits and result.get('threads_requested') == threads)
        if error:
            validation = dict(validation, error=error)
        native_seconds = result.get('native_seconds')
        api_seconds = result.get('api_seconds')
        time_s = api_seconds if isinstance(api_seconds, (int, float)) and math.isfinite(api_seconds) else elapsed
        rss = receipt.get('native_highwater', {})
        rss_bytes = rss.get('max_rss_bytes') if isinstance(rss, dict) else None
        records.append(dict(case_id=name, family=entry['family'], input_sha256=input_sha,
                            precision_bits=bits, threads=threads, order_block=block,
                            phase='cold' if index == 0 else 'warm', repetition=index, passed=passed,
                            sample_scope='fresh_cli_process',
                            status=normalized_status, raw_status=status, time_s=time_s,
                            cli_e2e_seconds=elapsed, native_seconds=native_seconds,
                            api_seconds=api_seconds, load_seconds=result.get('load_seconds'),
                            rss_bytes=rss_bytes, settings_sha256=settings_identity(settings, runtime),
                            runtime=runtime, validation=validation, raw_sample=result,
                            process_receipt=str(process_path), error=error,
                            incomplete=not process_ok))
    return records


def consume_holdout(args, candidate_identity, manifest_hash, entries):
    if not args.development_result:
        raise ValueError('holdout requires the kept development comparison for this exact candidate')
    development = read(args.development_result)
    candidate_path = Path(args.development_result).parent / 'candidate.json'
    candidate = read(candidate_path)
    baseline = read(Path(args.development_result).parent / 'baseline.json')
    evaluator = load_module('sdpx_holdout_precondition', HERE / 'evaluate.py')
    if (development.get('verdict') != 'keep' or evaluator.evaluate(baseline, candidate)['verdict'] != 'keep'
            or candidate['contract'].get('stage') != 'development'
            or candidate['contract'].get('precision_bits') != args.precision_bits
            or candidate['contract'].get('threads') != args.threads
            or candidate['identity'].get('harness_sha256') != harness_identity()
            or any(candidate['identity'].get(k) != candidate_identity[k]
                                                for k in ('source_sha256', 'artifact_sha256'))):
        raise ValueError('development result does not qualify this candidate')
    ledger = Path(args.state) / 'holdout-access.jsonl'
    prior = [json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []
    inputs = sorted({h for e in entries for h in (e.get('json_sha256'), e.get('source_sha256')) if h})
    if not inputs or any(set(r.get('input_hashes', [])) & set(inputs) for r in prior):
        raise ValueError('holdout already exposed: retain it as regression and prepare a new version')
    with ledger.open('a') as stream:
        stream.write(json.dumps({'manifest_sha256': manifest_hash, 'candidate': candidate_identity['source_sha256'],
                                 'input_hashes': inputs, 'output': str(args.output.resolve()), 'time': time.time()}) + '\n')


def profile_options(args):
    profile = getattr(args, 'profile', None)
    if profile is None:
        profile = ('screen' if args.stage == 'development' and args.threads == 1
                   and getattr(args, 'precision_bits', 53) == 53 else 'full')
    if profile not in ('full', 'screen'):
        raise ValueError('unknown profile')
    screen = profile == 'screen'
    if screen and (args.stage != 'development' or args.threads != 1):
        raise ValueError('screen requires --stage development and --threads 1')
    clamps = {}
    for field, limit in (('timeout', 120 if screen else 900),
                         ('budget_seconds', 600 if screen else 3600)):
        value = getattr(args, field, None)
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError('timeout and budget must be finite and positive')
        if screen and value is not None and value > limit:
            clamps[field] = {'requested': value, 'effective': limit}
        default = 180 if screen and field == 'budget_seconds' else limit
        setattr(args, field, default if value is None else min(value, limit) if screen else value)
    return dict(profile=profile, repetitions=2 if screen else 4,
                blocks=['AB'] if screen else ['AB', 'BA'],
                **({'speed_credit': False, 'budget_clamps': clamps} if screen else {}))


def profile_cases(manifest, profile):
    validate_catalog(manifest)
    if profile == 'full':
        return manifest['instances']
    if manifest['suite']['role'] != 'development':
        raise ValueError('screen requires the development catalog')
    library = load_module('sdpx_screen_catalog', HERE / 'catalog.py')
    selected = library.screen_cases(library.load_catalog())
    by_name = {e['name']: e for e in manifest['instances']}
    return [by_name[e['name']] for e in selected]

def pair(args):
    protocol = getattr(args, '_protocol', None) or profile_options(args)
    if args.output.exists():
        raise ValueError('output must not exist; preserve every experiment')
    data = args.data.resolve()
    manifest = read(data / 'manifest.json')
    validate_stage(manifest, args.stage)
    entries = profile_cases(manifest, protocol['profile'])
    validate_cases(dict(instances=entries), args.precision_bits)
    configs = {name: read(path) for name, path in [('baseline', args.baseline), ('candidate', args.candidate)]}
    external_output(args.output, configs.values())
    before = {name: arm_identity(config) for name, config in configs.items()}
    harness = harness_identity()
    manifest_hash = sha(data / 'manifest.json')
    if args.stage == 'holdout':
        consume_holdout(args, before['candidate'], manifest_hash, entries)
    args.output.mkdir(parents=True)
    scope = ('native CLI child: one fresh JSON solve per process; time_s=api_seconds '
             '(native setup+solve), native_seconds=solver solve, load_seconds=input load; '
             'CLI process e2e and external audit are separate')
    contract = dict(precision_bits=args.precision_bits, threads=args.threads, blas_threads=1,
                    tolerance_id='internal_full_1e-8_external_1e-6' if args.precision_bits == 53 else 'native_precision_external_sqrt_eps',
                    timing_scope=scope, stage=args.stage, timeout_s=args.timeout,
                    campaign_budget_s=args.budget_seconds,
                    memory_limit_mib=args.memory_mib, **protocol,
                    memory_scope='wait4 entire native CLI child including input/result serialization')
    if protocol['profile'] == 'screen':
        print('SCREEN: reduced protocol; speed_credit: false', flush=True)
    campaigns = {}
    for name in configs:
        (args.output / name).mkdir()
        campaigns[name] = dict(schema_version=1, identity={k: before[name][k] for k in
                                 ('source_sha256', 'artifact_sha256', 'environment_sha256')},
                               contract=contract, expected_cases=[dict(id=e['name'], family=e['family'],
                                   group=e.get('group', e.get('size_group', e['family']))) for e in entries],
                               records=[], identity_unchanged=False,
                               qualification_passed=qualification_matches(configs[name], before[name], harness))
        campaigns[name]['identity'].update(harness_sha256=harness, catalog_sha256=manifest_hash, host_id=host_identity())
        write(args.output / name / 'before.json', before[name])
    owned = supervisor()
    deadline = time.monotonic() + args.budget_seconds
    interruption = None
    try:
        # Pair each case, reverse both arms and case order in the second block.
        for block, names, selected in [('AB', ['baseline', 'candidate'], entries),
                                        ('BA', ['candidate', 'baseline'], list(reversed(entries)))]:
            if block not in contract['blocks']:
                continue
            for entry in selected:
                for name in names:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError('predeclared campaign wall-clock budget exhausted')
                    records = run_case(owned, configs[name], before[name], entry, data,
                                       args.precision_bits, args.threads, block, args.output / name,
                                       min(args.timeout, remaining), args.memory_mib, repetitions=contract['repetitions'])
                    campaigns[name]['records'].extend(records)
                    write(args.output / (name + '.json'), campaigns[name])
                    print(f"{block} {name} {entry['name']}: {'pass' if all(r['passed'] for r in records) else 'fail'}", flush=True)
    except TimeoutError as error:
        interruption = str(error)
    finally:
        for name in configs:
            try:
                after = arm_identity(configs[name])
                inputs_ok = all(sha(data / e['json_path']) == e['json_sha256'] for e in entries if 'json_path' in e)
                campaigns[name]['identity_unchanged'] = (before[name] == after and harness == harness_identity()
                    and manifest_hash == sha(data / 'manifest.json') and inputs_ok)
                write(args.output / name / 'after.json', after)
            except (OSError, ValueError) as error:
                campaigns[name]['identity_error'] = str(error)
            write(args.output / (name + '.json'), campaigns[name])
    evaluator = load_module('sdpx_research_evaluate', HERE / 'evaluate.py')
    result = evaluator.evaluate(campaigns['baseline'], campaigns['candidate'])
    if interruption:
        result['verdict'] = 'screen_fail' if protocol['profile'] == 'screen' else 'incomplete'
        result['reasons'].append(interruption)
    write(args.output / 'comparison.json', result)
    with (Path(args.state) / 'results.jsonl').open('a') as ledger:
        ledger.write(json.dumps(dict(time=time.time(), output=str(args.output.resolve()), stage=args.stage,
            manifest_sha256=manifest_hash, source_ids={n: before[n]['source_sha256'] for n in before},
            verdict=result['verdict'], reasons=result['reasons']), allow_nan=False) + '\n')
    print(json.dumps(result, indent=2))
    return 0 if result['verdict'] in ('keep', 'discard', 'correctness_only', 'screen_pass') else 1


def qualify(args):
    config = read(args.config)
    external_output(args.output, [config])
    commands = read(args.commands)
    if not isinstance(commands, list) or not commands or any(not isinstance(c, list) or not c or
            any(not isinstance(v, str) for v in c) for c in commands):
        raise ValueError('commands must be a nonempty JSON array of argv arrays; no shell interpolation')
    args.output.mkdir(parents=True, exist_ok=False)
    before = arm_identity(config)
    harness = harness_identity()
    env = config_env(config, before, 1)
    owned = supervisor()
    results = []
    deadline = time.monotonic() + args.budget_seconds
    for i, command in enumerate(commands):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        with (args.output / f'{i}.stdout').open('w') as stdout, (args.output / f'{i}.stderr').open('w') as stderr:
            receipt = owned(command, env=env, stdout=stdout, stderr=stderr,
                            timeout=min(args.timeout, remaining), memory_limit_mib=args.memory_mib)
        results.append(dict(receipt, command=command))
        if receipt.get('process_exit_code') != 0 or receipt.get('incomplete'):
            break
    after = arm_identity(config)
    unchanged = before == after and harness == harness_identity()
    passed = unchanged and len(results) == len(commands) and all(
        r.get('process_exit_code') == 0 and r.get('cleanup_confirmed') and not r.get('incomplete') for r in results)
    receipt = dict(passed=bool(passed), identity_unchanged=unchanged, commands=results,
                   harness_sha256=harness, **{k: before[k] for k in ('source_sha256', 'artifact_sha256', 'environment_sha256')})
    write(args.output / 'qualification.json', receipt)
    return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action')
    sub.required = True
    fingerprint = sub.add_parser('fingerprint')
    fingerprint.add_argument('--config', type=Path, required=True)
    pair_parser = sub.add_parser('pair')
    for key in ('baseline', 'candidate', 'data', 'output'):
        pair_parser.add_argument('--' + key, type=Path, required=True)
    pair_parser.add_argument('--precision-bits', type=int, choices=(53, 256, 512), default=53)
    pair_parser.add_argument('--threads', type=int, choices=(1, 2, 4, 8), default=1)
    pair_parser.add_argument('--stage', choices=('smoke', 'development', 'regression', 'holdout'), default='development')
    pair_parser.add_argument('--profile', choices=('full', 'screen'), default=None,
                             help='default: screen for single-thread Float64 development; full otherwise')
    pair_parser.add_argument('--development-result', type=Path)
    qualify_parser = sub.add_parser('qualify')
    for key in ('config', 'commands', 'output'):
        qualify_parser.add_argument('--' + key, type=Path, required=True)
    for p in (pair_parser, qualify_parser):
        p.add_argument('--state', type=Path, default=Path.home() / '.cache/sdpx-research')
        p.add_argument('--timeout', type=float, default=None if p is pair_parser else 900)
        p.add_argument('--memory-mib', type=float, default=4096)
        p.add_argument('--budget-seconds', type=float, default=None if p is pair_parser else 3600)
    args = parser.parse_args()
    if args.action == 'fingerprint':
        print(json.dumps(arm_identity(read(args.config)), indent=2))
        return 0
    if args.action == 'pair':
        args._protocol = profile_options(args)
    if any(not math.isfinite(v) or v <= 0 for v in (args.timeout, args.memory_mib, args.budget_seconds)):
        parser.error('timeout and memory bounds must be finite and positive')
    with slot(args.state):
        return pair(args) if args.action == 'pair' else qualify(args)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError) as error:
        print(f'Experiment stopped: {error}', file=sys.stderr)
        sys.exit(2)
