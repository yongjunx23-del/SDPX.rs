#!/usr/bin/env python3
"""Run retained reference legs under the sequential watchdog.

The former ``--engine new`` SDPX frontend leg is retired.  Native SDPX
research comparisons use ``benchmark/research/run.py pair`` with an explicit
``cli`` arm configuration; this wrapper remains for Clarabel/MOSEK reference
measurements only.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import signal
import subprocess
import time

HERE = Path(__file__).resolve().parent


class CleanupUnconfirmed(RuntimeError):
    """The campaign must not launch another numerical process."""


def owned_supervisor(group_rss_kib):
    # One closure per campaign; a failed cleanup permanently closes its launch gate.
    blocked = None

    def group_gone(pgid, errors):
        try:
            os.killpg(pgid, 0)
            return False
        except ProcessLookupError:
            return True
        except OSError as error:
            errors.append(f'group probe: {type(error).__name__}: {error}')
            return False

    def cleanup(process, errors):
        for sig, grace in ((signal.SIGTERM, 0.5), (signal.SIGKILL, 2.0)):
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                pass
            except OSError as error:
                errors.append(f'{sig.name}: {type(error).__name__}: {error}')
            deadline = time.monotonic() + grace
            while True:
                try:
                    process.poll()  # Reap the direct child before probing the group.
                except OSError as error:
                    errors.append(f'child reap: {type(error).__name__}: {error}')
                if group_gone(process.pid, errors):
                    return process.returncode is not None
                if time.monotonic() >= deadline or errors:
                    break
                time.sleep(0.05)
        return False

    def run(cmd, *, env, stdout, stderr, timeout, memory_limit_mib=0):
        nonlocal blocked
        if blocked is not None:
            raise CleanupUnconfirmed(blocked)
        started = time.monotonic()
        result = dict(peak_rss_mib=None, memory_limit_mib=memory_limit_mib)
        try:
            process = subprocess.Popen(cmd, env=env, stdout=stdout, stderr=stderr,
                                       start_new_session=True)
        except OSError as error:
            return dict(result, reason='launch_failed', incomplete=True,
                        error=f'{type(error).__name__}: {error}', process_exit_code=None)
        try:
            while process.poll() is None:
                if memory_limit_mib:
                    rss = group_rss_kib(process.pid) / 1024
                    result['peak_rss_mib'] = max(result['peak_rss_mib'] or 0, rss)
                    if rss > memory_limit_mib:
                        result.update(reason='resource_limit', incomplete=True,
                                      error=f'process-group RSS exceeded {memory_limit_mib:g} MiB')
                        break
                remaining = timeout - (time.monotonic() - started)
                if remaining <= 0:
                    result.update(reason='timeout', incomplete=True,
                                  error=f'whole process exceeded {timeout:g}s')
                    break
                try:
                    process.wait(timeout=min(0.2, remaining))
                except subprocess.TimeoutExpired:
                    pass
        except (OSError, subprocess.SubprocessError) as error:
            result.update(reason='watchdog_error', incomplete=True,
                          error=f'RSS watchdog failed: {type(error).__name__}: {error}')
        except BaseException:
            if not cleanup(process, []):
                blocked = f'Campaign stopped: cleanup unconfirmed for owned process group {process.pid}'
            raise
        errors = []
        confirmed = group_gone(process.pid, errors) and process.returncode is not None
        if not confirmed:
            confirmed = cleanup(process, errors)
        result.update(process_exit_code=process.returncode, cleanup_confirmed=confirmed,
                      cleanup_errors=errors)
        if not confirmed:
            result.setdefault('reason', 'cleanup_unconfirmed')
            result.setdefault('error', 'owned process group cleanup could not be confirmed')
            result['incomplete'] = True
            blocked = f'Campaign stopped: cleanup unconfirmed for owned process group {process.pid}'
        return result
    return run


def with_native_highwater(run_owned):
    def measured(cmd, **kwargs):
        receipt_path = Path(kwargs['stdout'].name).with_suffix('.resource.json')
        wrapped = [sys.executable, str(HERE / 'resource_probe.py'), str(receipt_path), '--', *cmd]
        result = run_owned(wrapped, **kwargs)
        receipt = {'complete': False, 'receipt_path': str(receipt_path),
                   'scope': 'single solver child process high-water RSS; not simultaneous process-group peak'}
        if receipt_path.is_file():
            try:
                recorded = json.loads(receipt_path.read_text())
                if recorded.get('command') != cmd:
                    raise ValueError('resource receipt command mismatch')
                if recorded.get('complete') and recorded.get('child_exit_code') != result.get('process_exit_code'):
                    raise ValueError('resource receipt exit status mismatch')
                receipt.update(recorded)
                if recorded.get('launch_error'):
                    result.setdefault('reason', 'launch_failed')
                    result.setdefault('error', recorded['launch_error'])
                    result['incomplete'] = True
            except (OSError, ValueError) as error:
                receipt['error'] = str(error)
        else:
            receipt['error'] = 'no completed wait4 receipt; watchdog interruption may prevent collection'
        result['native_highwater'] = receipt
        result['sampled_group_includes_resource_probe'] = True
        return result
    return measured


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, required=True,
                   help='retained final-selection directory')
    p.add_argument('--engine', choices=['rust_clarabel', 'mosek'], required=True)
    p.add_argument('--reference-arm', choices=['reference_default', 'controlled'],
                   help='Clarabel.rs preprocessing; defaults on, controlled disables it explicitly')
    p.add_argument('--threads', type=int, choices=[1], default=1,
                   help='reference legs are fixed to one solver thread')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--native-highwater', action='store_true',
                   help='add OS child high-water RSS; retain sampled process-group watchdog')
    args, forwarded = p.parse_known_args()
    if args.reference_arm is not None and args.engine != 'rust_clarabel':
        p.error('--reference-arm applies only to --engine rust_clarabel')
    if args.engine == 'rust_clarabel':
        # An inherited controlled-arm environment must not disable product defaults.
        args.reference_arm = args.reference_arm or 'reference_default'
        os.environ['SDPX_RUST_ARM'] = args.reference_arm
    if args.native_highwater and (not hasattr(os, 'wait4') or sys.platform not in ('darwin', 'linux')):
        p.error('--native-highwater requires wait4 on macOS or Linux')
    os.environ['SDPX_BENCH_THREADS'] = str(args.threads)
    # Faer uses the global Rayon pool; cone workers use their own bounded pool.
    os.environ['RAYON_NUM_THREADS'] = str(args.threads)
    for key in ('OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS',
                'OMP_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    ref = args.reference.resolve()
    manifest = json.loads((ref / 'data/manifest.json').read_text())
    assert len(manifest['instances']) == 10
    assert manifest['suite']['independent_oracle_tolerance'] == 1e-6
    for entry in manifest['instances']:
        data = (ref / 'data' / entry['json_path']).read_bytes()
        assert hashlib.sha256(data).hexdigest() == entry['json_sha256']
    # Fixed acceptance settings cannot be accidentally overridden downstream.
    fixed = {'--data', '--legs', '--selection', '--runs', '--timeout', '--memory-limit-mib'}
    if any(x.split('=')[0] in fixed for x in forwarded):
        p.error('data, legs, selection, runs and resource bounds are fixed by this wrapper')
    spec = importlib.util.spec_from_file_location('retained_suite', ref / 'adapters/run_suite.py')
    suite = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(suite)
    suite.run_owned = owned_supervisor(suite.group_rss_kib)
    if args.native_highwater:
        suite.run_owned = with_native_highwater(suite.run_owned)
    # Fingerprint this wrapper and adapter in addition to retained reference code.
    extra = ['--rust-source', str(HERE)]
    leg = args.engine
    sys.argv = [str(ref / 'adapters/run_suite.py'), *forwarded, *extra,
                '--output', str(args.output),
                '--data', str(ref / 'data'), '--legs', leg, '--selection', 'full',
                '--runs', '4', '--timeout', '900', '--memory-limit-mib', '4096']
    try:
        code = suite.main()
    except CleanupUnconfirmed as error:
        (args.output / "supervision_abort.json").write_text(json.dumps(
            dict(incomplete=True, reason="cleanup_unconfirmed", error=str(error)),
            indent=2) + "\n")
        return 2
    metadata_path = args.output / 'run_metadata.json'
    metadata = json.loads(metadata_path.read_text())
    metadata['reference_arm'] = args.reference_arm
    if args.engine == 'rust_clarabel':
        enabled = args.reference_arm == 'reference_default'
        metadata['requested_preprocessing'] = dict(
            equilibrate_enable=enabled, presolve_enable=enabled,
            chordal_decomposition_enable=enabled)
    metadata['requested_native_threads'] = args.threads
    metadata['native_blas_thread_budget'] = 1
    metadata['rayon_global_thread_budget'] = args.threads
    metadata['thread_budget_scope'] = 'cone worker pool and eligible KKT factorization; not total process threads'
    metadata['backend_thread_receipt_scope'] = 'configured factorization width; not busy CPU cores or measured utilization'
    metadata['native_highwater_enabled'] = args.native_highwater
    metadata['memory_measurement_scope'] = (
        'native_highwater: wait4 single solver child; peak_rss_mib: sampled process group including probe'
        if args.native_highwater else 'peak_rss_mib: sampled process group; short-lived peaks can be missed')
    metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
    return code


if __name__ == '__main__':
    sys.exit(main())
