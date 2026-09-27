#!/usr/bin/env python3
"""Async cluster bridge for family-tree campaigns (PBS over ssh, fire-and-forget).

All numerical computation runs on the cluster. This tool never solves locally;
local python only stages snapshots, renders job scripts, and tracks state.

Commands:
  preflight   read-only cluster check (connectivity, queues, toolchains)
  sync-base   upload the shared baseline snapshot once per campaign
  submit      stage a variant snapshot, qsub one PBS job, return immediately
  poll        qstat live jobs, fetch finished verdicts, release cores
  fetch       pull a full remote results directory for a recorded job

Budget and tree state live in familytree state (see tree.py). submit holds the
state slot for its whole run so concurrent workers cannot overbook cores.
poll holds the slot while applying verdicts via tree.apply_record (no nested
locking). qdel is deliberately absent: cancelling a queued/running job needs
human approval; cut trees keep reservations until poll records them.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
try:
    import tree as familytree
except ImportError:  # loaded as a test module under another name
    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        'familytree', HERE / 'tree.py')
    familytree = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(familytree)

REMOTE_BASE = '$HOME/projects/sdpx-families'

_REMOTE_HOME = {}


def _remote_home(remote):
    """Absolute remote $HOME. Local scp uses SFTP (no shell expansion)."""
    if remote not in _REMOTE_HOME:
        proc = ssh(remote, 'printf %s "$HOME"', timeout=60)
        if proc.returncode != 0 or not proc.stdout.strip():
            raise ValueError(
                f'cannot resolve remote HOME: {proc.stderr.strip()}')
        _REMOTE_HOME[remote] = proc.stdout.strip()
    return _REMOTE_HOME[remote]


def _remote_base(remote):
    return f'{_remote_home(remote)}/projects/sdpx-families'
SNAPSHOT_DIRS = ('crates', 'include')
DEFAULT_FEATURES = 'sdp-openblas,faer-sparse'
JOBID_RE = re.compile(r'^(\d[\w.-]*)')
LIVE_JOB = ('submitted', 'queued', 'running', 'unknown')


def _run(argv, timeout=300):
    # ``capture_output`` and ``text`` were added after the Python 3.6 runtime
    # still used by the cluster bridge.  Keep the equivalent older spelling.
    return subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          universal_newlines=True, timeout=timeout)


def ssh(remote, script, timeout=300):
    return _run(['ssh', remote, script], timeout=timeout)


def check_snapshot(snapshot):
    snapshot = Path(snapshot)
    for directory in SNAPSHOT_DIRS:
        if not (snapshot / directory).is_dir():
            raise ValueError(
                f'snapshot {snapshot} lacks {directory}/; not an SDPX root')
    harness = snapshot / 'benchmark' / 'research' / 'run.py'
    if not harness.is_file():
        raise ValueError(f'snapshot {snapshot} lacks benchmark/research/run.py')
    return snapshot.resolve()


def stage_snapshot(snapshot, stagedir):
    stagedir = Path(stagedir)
    stagedir.mkdir(parents=True, exist_ok=True)
    tarball = stagedir / 'snapshot.tar.gz'
    if tarball.exists():
        tarball.unlink()
    with tarfile.open(tarball, 'w:gz') as tar:
        for root, dirs, files in os.walk(snapshot):
            dirs[:] = [d for d in dirs if d not in ('target', '.git')]
            for name in files:
                full = Path(root) / name
                tar.add(full, arcname=full.relative_to(snapshot))
    digest = hashlib.sha256(tarball.read_bytes()).hexdigest()
    return tarball, digest


JOB_SCRIPT = r'''#!/bin/bash
#PBS -N @@JOBNAME@@
#PBS -q normal
#PBS -l nodes=1:ppn=@@CORES@@
#PBS -l mem=@@MEMGB@@gb
#PBS -l walltime=@@WALLTIME@@
#PBS -j oe
set -euo pipefail
BASE="$HOME/projects/sdpx-families/@@CAMPAIGN@@"
NODE_DIR="$BASE/@@TREE@@/@@NODE@@"
CAND_SRC="$NODE_DIR/source"
BASE_SRC="$BASE/_base/source"
WORK="$NODE_DIR/work"
OUT="$NODE_DIR/results"
mkdir -p "$WORK" "$OUT"
CARGO="$HOME/.cargo/bin/cargo"
cd "$CAND_SRC"
$CARGO build --locked --offline --release -p sdpx-solver --bin sdpx --features "@@FEATURES@@"
if [ ! -x "$BASE_SRC/target/release/sdpx" ]; then
    (
        flock -x 200
        if [ ! -x "$BASE_SRC/target/release/sdpx" ]; then
            cd "$BASE_SRC"
            $CARGO build --locked --offline --release -p sdpx-solver --bin sdpx --features "@@FEATURES@@"
        fi
    ) 200>"$BASE_SRC/.build.lock"
fi
HARNESS="$CAND_SRC/benchmark/research"
python3 "$HARNESS/catalog.py" materialize --suite @@SUITE@@ \
  --workspace "$HOME/projects/sdpx-families/data" --output "$WORK/data"
python3 - "$WORK" "$CAND_SRC" "$BASE_SRC" <<'EOF'
import json, sys
work, cand, base = sys.argv[1:4]
def arm(src):
    return {'source': src, 'cli': src + '/target/release/sdpx',
            'blas': 'default'}
json.dump(arm(cand), open(work + '/candidate.json', 'w'))
json.dump(arm(base), open(work + '/baseline.json', 'w'))
EOF
python3 "$HARNESS/run.py" pair --profile @@PROFILE@@ --stage @@SUITE@@ \
  --threads @@THREADS@@ --precision-bits 53 --state "$WORK/state" \
  --baseline "$WORK/baseline.json" --candidate "$WORK/candidate.json" \
  --data "$WORK/data" --output "$OUT/pair" 2>&1 | tee "$OUT/run.log"
'''


def render_job(args, tree_id):
    text = JOB_SCRIPT
    mapping = {'JOBNAME': f'sdpx-{tree_id}-{args.node}',
               'CORES': str(args.cores), 'MEMGB': str(args.mem_gb),
               'WALLTIME': args.walltime, 'CAMPAIGN': args.campaign,
               'TREE': tree_id, 'NODE': args.node,
               'FEATURES': args.features, 'SUITE': args.stage,
               'PROFILE': args.profile, 'THREADS': str(args.threads)}
    for key, value in mapping.items():
        text = text.replace(f'@@{key}@@', value)
    if '@@' in text:
        raise ValueError('unrendered job template token remains')
    return text


def cmd_preflight(args):
    proc = ssh(args.remote,
               'hostname; echo ---; qstat -q; echo ---; '
               'command -v python3; ls $HOME/.cargo/bin/cargo; echo ---; qstat -an | head -30')
    print(proc.stdout, end='')
    if proc.returncode != 0:
        print(proc.stderr, end='', file=sys.stderr)
        return 1
    return 0


def upload_and_setup(remote, campaign, dest, tarball, timeout):
    """Upload tarball to dest/source under the campaign and fetch Rust deps."""
    dest = '/'.join(familytree.check_remote_name('dest-part', part)
                    for part in (dest or '').split('/'))
    campaign = familytree.check_remote_name('campaign', campaign)
    base = _remote_base(remote)
    flat = dest.replace('/', '-')
    incoming = f'{base}/{campaign}/incoming/{flat}.tar.gz.part'
    mkdir = ssh(remote, f'mkdir -p {base}/{campaign}/incoming',
                timeout=120)
    if mkdir.returncode != 0:
        raise ValueError(f'remote mkdir failed: {mkdir.stderr.strip()}')
    scp = _run(['scp', '-q', str(tarball), f'{remote}:{incoming}'],
               timeout=timeout)
    if scp.returncode != 0:
        raise ValueError(f'scp failed: {scp.stderr.strip()}')
    script = (
        'set -euo pipefail; '
        f'BASE={REMOTE_BASE}/{campaign}; DIR="$BASE/{dest}"; '
        'mkdir -p "$BASE/incoming" "$DIR"; rm -rf "$DIR/source"; '
        'mkdir -p "$DIR/source"; '
        f'tar -xzf "$BASE/incoming/{flat}.tar.gz.part" -C "$DIR/source"; '
        f'rm -f "$BASE/incoming/{flat}.tar.gz.part"; '
        'cd "$DIR/source" && $HOME/.cargo/bin/cargo fetch --locked'
    )
    proc = ssh(remote, script, timeout=timeout)
    if proc.returncode != 0:
        raise ValueError(f'remote setup failed: {proc.stdout}\n{proc.stderr}')
    return f'$HOME/projects/sdpx-families/{campaign}/{dest}'


def cmd_sync_base(args):
    snapshot = check_snapshot(args.snapshot)
    with tempfile.TemporaryDirectory() as tmp:
        tarball, digest = stage_snapshot(snapshot, Path(tmp))
        upload_and_setup(args.remote, args.campaign, '_base', tarball,
                         args.timeout)
    print(f'base ready: {args.campaign}/_base sha256={digest}')
    return 0


def cmd_submit(args):
    if args.cores in (3, 4):
        raise ValueError('ppn=3/4 stay queued; use ppn=8 or another size')
    if args.cores < args.threads:
        raise ValueError('ppn cores must cover benchmark threads')
    if args.profile == 'screen' and (args.stage != 'development'
                                     or args.threads != 1):
        raise ValueError('screen requires --stage development and --threads 1')
    snapshot = check_snapshot(args.snapshot)
    campaign = familytree.check_remote_name('campaign', args.campaign)
    with familytree.slot(args.state) as state:
        data = familytree.load(state)
        node = data['nodes'].get(args.node)
        if node is None:
            raise ValueError(f'unknown node {args.node}')
        if data['trees'][node['tree']]['status'] not in familytree.ACTIVE_STATUSES:
            raise ValueError(f'tree {node["tree"]} is not active')
        if node['status'] not in ('design', 'ready'):
            raise ValueError(f'node {args.node} is {node["status"]}, '
                             'already submitted or judged')
        if familytree.live_cores(data) + args.cores > data['core_budget']:
            raise ValueError(
                f'budget exceeded: live {familytree.live_cores(data)} + '
                f'{args.cores} > {data["core_budget"]}')
        tree_id = node['tree']
        stagedir = familytree.staging_dir(state, tree_id, args.node)
        tarball, digest = stage_snapshot(snapshot, stagedir)
        upload_and_setup(args.remote, campaign, f'{tree_id}/{args.node}',
                         tarball, args.timeout)
        script_text = render_job(args, tree_id)
        script_remote = (f'{_remote_base(args.remote)}/{campaign}/{tree_id}/{args.node}/job.sh')
        with tempfile.NamedTemporaryFile('w', suffix='.sh',
                                         delete=False) as handle:
            handle.write(script_text)
            local_script = handle.name
        try:
            scp = _run(['scp', '-q', local_script,
                        f'{args.remote}:{script_remote}'], timeout=120)
            if scp.returncode != 0:
                raise ValueError(f'job script upload failed: {scp.stderr.strip()}')
            qsub = ssh(args.remote, f'qsub {script_remote}', timeout=120)
            if qsub.returncode != 0:
                raise ValueError(f'qsub failed: {qsub.stderr.strip()}')
            match = JOBID_RE.search(qsub.stdout.strip().splitlines()[0]
                                    if qsub.stdout.strip() else '')
            if not match:
                raise ValueError(f'unparseable qsub output: {qsub.stdout!r}')
            job_id = match.group(1)
            if '.' not in job_id and qsub.stdout.strip():
                job_id = qsub.stdout.strip().split()[0]
        finally:
            os.unlink(local_script)
        data['jobs'][job_id] = {
            'node': args.node, 'cores': args.cores, 'threads': args.threads,
            'profile': args.profile, 'stage': args.stage,
            'payload_sha256': digest, 'campaign': campaign,
            'status': 'submitted', 'submitted': familytree._now(),
        }
        node['status'] = 'submitted'
        node['job_id'] = job_id
        node['cores'] = args.cores
        familytree.save(state, data)
        familytree.log_event(state, {'event': 'submit', 'tree': tree_id,
                                     'node': args.node, 'job': job_id,
                                     'cores': args.cores,
                                     'payload_sha256': digest})
        print(f'{job_id} (node {args.node}; live '
              f'{familytree.live_cores(data)}/{data["core_budget"]} cores)')
    return 0


QSTAT_MAP = {'Q': 'queued', 'H': 'queued', 'W': 'queued', 'S': 'queued',
             'R': 'running', 'T': 'running', 'E': 'running'}


def poll_remote(remote, job_ids):
    """One ssh call: job_state + exit_status per job, plus verdict presence."""
    parts = []
    for job_id in job_ids:
        parts.append(
            f'echo "== {job_id}"; '
            f'qstat -f "{job_id}" 2>&1 | grep -E "(job_state|exit_status|Unknown Job Id)" || true'
        )
    script = '\n'.join(parts)
    proc = ssh(remote, script, timeout=180)
    if proc.returncode != 0:
        raise ValueError(f'qstat failed: {proc.stderr.strip()}')
    # NOTE: `qstat -f` indents attribute lines, so match unanchored.
    states = {}
    current, info = None, {}
    for line in proc.stdout.splitlines():
        if line.startswith('== '):
            if current:
                states[current] = info
            current, info = line[3:].strip(), {}
            continue
        if '=' in line:
            key, _, value = line.partition('=')
            info[key.strip()] = value.strip()
        elif 'Unknown Job Id' in line:
            info['Unknown Job Id'] = line.strip()
    if current:
        states[current] = info
    return states


def cmd_poll(args):
    with familytree.slot(args.state) as state:
        data = familytree.load(state)
        live = {jid: job for jid, job in data['jobs'].items()
                if job['status'] in LIVE_JOB}
        if not live:
            print('no live jobs')
            return 0
        states = poll_remote(args.remote, list(live))
        for job_id, job in live.items():
            info = states.get(job_id, {})
            raw = info.get('job_state')
            node = data['nodes'][job['node']]
            if raw in QSTAT_MAP:
                job['status'] = ('submitted' if job['status'] == 'submitted'
                                 and QSTAT_MAP[raw] == 'queued'
                                 else QSTAT_MAP[raw])
                node['status'] = ('submitted' if job['status'] in
                                  ('submitted', 'queued') else 'running')
            elif raw == 'C':
                finish_job(args, state, data, job_id, job,
                           info.get('exit_status') == '0')
            elif raw == 'F':
                finish_job(args, state, data, job_id, job, False)
            elif 'Unknown Job Id' in json.dumps(info):
                check_vanished(args, state, data, job_id, job)
            else:
                job['status'] = 'unknown'
        familytree.save(state, data)
        done = sum(1 for j in data['jobs'].values()
                   if j['status'] in ('done', 'failed'))
        print(f'live {familytree.live_cores(data)}/{data["core_budget"]} cores; '
              f'{len(live)} polled, {done} finished total')
    return 0


def finish_job(args, state, data, job_id, job, exited_ok):
    node = data['nodes'][job['node']]
    tree_id = node['tree']
    outdir = familytree.outputs_dir(state, tree_id, job['node'])
    outdir.mkdir(parents=True, exist_ok=True)
    base = _remote_base(args.remote)
    comparison_remote = (
        f'{base}/{job["campaign"]}/{tree_id}/{job["node"]}'
        f'/results/pair/comparison.json')
    log_remote = (f'{base}/{job["campaign"]}/{tree_id}/{job["node"]}'
                  f'/results/run.log')
    comparison_local = outdir / 'comparison.json'
    got = _run(['scp', '-q',
                f'{args.remote}:{comparison_remote}', str(comparison_local)],
               timeout=180)
    _run(['scp', '-q', f'{args.remote}:{log_remote}',
          str(outdir / 'run.log')], timeout=180)
    if got.returncode != 0 or not comparison_local.is_file():
        record_incomplete(state, data, job_id, job, 'comparison.json missing')
        return
    payload = json.loads(comparison_local.read_text())
    message, event = familytree.apply_record(
        data, job_id, payload.get('verdict'), comparison_local)
    familytree.log_event(state, event)
    print(message)


def record_incomplete(state, data, job_id, job, note):
    """Mark a dead job incomplete and release its cores. No verdict file."""
    node = data['nodes'][job['node']]
    tree_id = node['tree']
    job['status'] = 'failed'
    job['finished'] = familytree._now()
    released = job['cores']
    job['cores'] = 0
    node['status'] = 'failed'
    node['verdict'] = 'incomplete'
    familytree.log_event(state, {'event': 'record', 'tree': tree_id,
                                 'node': job['node'], 'job': job_id,
                                 'verdict': 'incomplete',
                                 'comparison_sha256': None,
                                 'cores_released': released, 'note': note})
    print(f'{job["node"]} incomplete ({note}; released {released} cores)')


def check_vanished(args, state, data, job_id, job):
    """Job reaped from the server: fetch verdict if present, else incomplete."""
    node = data['nodes'][job['node']]
    tree_id = node['tree']
    comparison_remote = (
        f'{REMOTE_BASE}/{job["campaign"]}/{tree_id}/{job["node"]}'
        f'/results/pair/comparison.json')
    proc = ssh(args.remote, f'test -f {comparison_remote} && echo PRESENT',
               timeout=60)
    if 'PRESENT' in proc.stdout:
        finish_job(args, state, data, job_id, job, True)
    else:
        # Fully reaped with no verdict file: nothing more will arrive.
        record_incomplete(state, data, job_id, job,
                          'job reaped with no comparison.json')


def cmd_fetch(args):
    with familytree.slot(args.state) as state:
        data = familytree.load(state)
        job = data['jobs'].get(args.job)
        if job is None:
            raise ValueError(f'unknown job {args.job}')
        node = data['nodes'][job['node']]
        remote_dir = (f'{_remote_base(args.remote)}/{job["campaign"]}/{node["tree"]}/'
                      f'{job["node"]}/results')
        local_dir = familytree.outputs_dir(state, node['tree'], job['node'])
        local_dir.mkdir(parents=True, exist_ok=True)
        proc = _run(['scp', '-qr', f'{args.remote}:{remote_dir}/',
                     str(local_dir / 'results')], timeout=600)
        if proc.returncode != 0:
            raise ValueError(f'fetch failed: {proc.stderr.strip()}')
        print(f'{args.job} -> {local_dir / "results"}')
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', type=Path,
                        default=familytree.default_state_dir())
    parser.add_argument('--remote', default='hpc')
    parser.add_argument('--timeout', type=int, default=1200)
    sub = parser.add_subparsers(dest='action')
    sub.required = True
    preflight = sub.add_parser('preflight')
    preflight.set_defaults(func=cmd_preflight)
    base = sub.add_parser('sync-base')
    base.add_argument('--campaign', required=True)
    base.add_argument('--snapshot', type=Path, required=True)
    base.set_defaults(func=cmd_sync_base)
    submit = sub.add_parser('submit')
    submit.add_argument('--campaign', required=True)
    submit.add_argument('--node', required=True)
    submit.add_argument('--snapshot', type=Path, required=True)
    submit.add_argument('--cores', type=int, default=8)
    submit.add_argument('--mem-gb', type=int, default=32)
    submit.add_argument('--walltime', default='04:00:00')
    submit.add_argument('--profile', choices=('full', 'screen'),
                        default='screen')
    submit.add_argument('--stage',
                        choices=('smoke', 'development', 'regression',
                                 'holdout'),
                        default='development')
    submit.add_argument('--threads', type=int, choices=(1, 2, 4, 8),
                        default=1)
    submit.add_argument('--features', default=DEFAULT_FEATURES)
    submit.set_defaults(func=cmd_submit)
    poll = sub.add_parser('poll')
    poll.add_argument('--campaign', required=True)
    poll.set_defaults(func=cmd_poll)
    fetch = sub.add_parser('fetch')
    fetch.add_argument('--job', required=True)
    fetch.set_defaults(func=cmd_fetch)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (OSError, ValueError, KeyError) as error:
        print(f'Cluster bridge stopped: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
