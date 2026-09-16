#!/usr/bin/env python3
"""Family-tree campaign state: trees, binary rounds, core budget, merge/cut/adopt.

One coordinator owns this state. Each live tree is owned by one tree worker.
Every round of a tree produces exactly two candidate variants (binary split);
trees keep bifurcating until the main agent cuts, merges, or adopts them.

State lives in <state>/families.json (schema_version 1) plus an append-only
<state>/results.jsonl log. All mutations take an exclusive file lock.
Core budget counts PBS-allocated cores of live jobs (see hpc.py).
"""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

SCHEMA_VERSION = 1
DEFAULT_CORE_BUDGET = 640
DEFAULT_MAX_TREES = 32
VERDICTS = ('keep', 'discard', 'correctness_only', 'incomplete',
            'screen_pass', 'screen_fail')
TREE_STATUSES = ('live', 'cut', 'merged', 'adopted', 'superseded')
# Adopted trees stay branchable: adoption is a milestone, optimization never
# stops. Cut/merged trees are dead ends awaiting purge.
ACTIVE_STATUSES = ('live', 'adopted')
NODE_STATUSES = ('design', 'ready', 'submitted', 'running', 'done', 'failed')
JOB_STATUSES = ('submitted', 'queued', 'running', 'done', 'failed', 'unknown')
_NAME_OK = re.compile(r'^[A-Za-z0-9_][A-Za-z0-9._-]{0,127}$')


def default_state_dir():
    return Path(os.environ.get('SDPX_FAMILY_STATE',
                               Path.home() / '.cache' / 'sdpx-families'))


@contextmanager
def slot(state_dir):
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    with open(state_dir / '.lock', 'w') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield state_dir
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _now():
    return time.strftime('%Y-%m-%dT%H:%M:%S%z')


def blank(core_budget=DEFAULT_CORE_BUDGET, max_trees=DEFAULT_MAX_TREES):
    return {'schema_version': SCHEMA_VERSION, 'core_budget': core_budget,
            'max_live_trees': max_trees, 'next_tree_id': 1, 'next_node_id': 1,
            'created': _now(), 'champion': None,
            'trees': {}, 'nodes': {}, 'jobs': {}}


def families_path(state_dir):
    return Path(state_dir) / 'families.json'


def staging_dir(state_dir, tree_id, node_id=None):
    base = Path(state_dir) / 'staging' / tree_id
    return base / node_id if node_id else base


def outputs_dir(state_dir, tree_id, node_id=None):
    base = Path(state_dir) / 'outputs' / tree_id
    return base / node_id if node_id else base


def check_remote_name(kind, value):
    if not _NAME_OK.match(value or ''):
        raise ValueError(f'unsafe {kind} for remote path: {value!r}')
    return value


def load(state_dir):
    path = families_path(state_dir)
    if not path.exists():
        raise ValueError(f'uninitialized family state {state_dir}; run tree.py init')
    data = json.loads(path.read_text())
    if data.get('schema_version') != SCHEMA_VERSION:
        raise ValueError('unsupported families.json schema_version')
    for node_id, node in data.get('nodes', {}).items():
        node.setdefault('id', node_id)
    return data


def save(state_dir, data):
    path = families_path(state_dir)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')
    os.replace(tmp, path)


def log_event(state_dir, event):
    event = dict(event)
    event.setdefault('ts', _now())
    with open(Path(state_dir) / 'results.jsonl', 'a') as handle:
        handle.write(json.dumps(event, sort_keys=True) + '\n')


def live_trees(data):
    return {tid: t for tid, t in data['trees'].items() if t['status'] == 'live'}


def live_cores(data):
    total = 0
    for job in data['jobs'].values():
        if job['status'] in ('submitted', 'queued', 'running', 'unknown'):
            total += job['cores']
    return total


def tree_tip(data, tree_id):
    """Latest node of a tree: highest round, then highest node sequence."""
    nodes = [n for n in data['nodes'].values() if n['tree'] == tree_id]
    if not nodes:
        return None
    return sorted(nodes, key=lambda n: (n['round'], n['seq']))[-1]


def cmd_init(args):
    with slot(args.state) as state:
        path = families_path(state)
        if path.exists() and not args.force:
            raise ValueError(f'{path} exists; use --force to reinitialize')
        save(state, blank(args.core_budget, args.max_trees))
        print(f'initialized {path} '
              f'(budget={args.core_budget} cores, max_live_trees={args.max_trees})')
    return 0


def cmd_plant(args):
    with slot(args.state) as state:
        data = load(state)
        if len(live_trees(data)) >= data['max_live_trees']:
            raise ValueError(
                f'live tree cap reached ({data["max_live_trees"]}); '
                'cut or merge a tree first')
        tree_id = f'T{data["next_tree_id"]}'
        data['next_tree_id'] += 1
        node_id = f'N{data["next_node_id"]}'
        data['next_node_id'] += 1
        data['trees'][tree_id] = {
            'name': args.name, 'hypothesis': args.hypothesis,
            'status': 'live', 'created': _now(), 'nodes': [node_id],
            'root_source_sha256': args.source_sha,
            'merged_into': None, 'cut_reason': None, 'adopt_reason': None,
        }
        data['nodes'][node_id] = {
            'id': node_id, 'tree': tree_id, 'round': 0, 'kind': 'root', 'seq': 1,
            'parent': None, 'merged_from': None, 'hypothesis': args.hypothesis,
            'source_sha256': args.source_sha, 'status': 'ready',
            'job_id': None, 'cores': 0, 'verdict': None,
            'comparison_sha256': None,
        }
        save(state, data)
        log_event(state, {'event': 'plant', 'tree': tree_id, 'node': node_id,
                          'name': args.name})
        print(f'{tree_id} {node_id}')
    return 0


def cmd_branch(args):
    with slot(args.state) as state:
        data = load(state)
        tree = data['trees'].get(args.tree)
        if tree is None:
            raise ValueError(f'unknown tree {args.tree}')
        if tree['status'] not in ACTIVE_STATUSES:
            raise ValueError(
                f'tree {args.tree} is {tree["status"]}, not branchable')
        parent = args.parent
        if parent is None:
            tip = tree_tip(data, args.tree)
            parent = tip['id'] if tip else None
        else:
            if parent not in data['nodes'] or data['nodes'][parent]['tree'] != args.tree:
                raise ValueError(f'parent {parent} is not a node of {args.tree}')
        tip_round = max([n['round'] for n in data['nodes'].values()
                         if n['tree'] == args.tree] or [0])
        made = []
        seq_base = max([n['seq'] for n in data['nodes'].values()
                        if n['tree'] == args.tree] or [0])
        for i, (hypothesis, sha) in enumerate(
                ((args.ha, args.sa), (args.hb, args.sb))):
            node_id = f'N{data["next_node_id"]}'
            data['next_node_id'] += 1
            data['nodes'][node_id] = {
                'id': node_id, 'tree': args.tree, 'round': tip_round + 1, 'kind': 'variant',
                'seq': seq_base + i + 1, 'parent': parent, 'merged_from': None,
                'hypothesis': hypothesis, 'source_sha256': sha,
                'status': 'design', 'job_id': None, 'cores': 0,
                'verdict': None, 'comparison_sha256': None,
            }
            tree['nodes'].append(node_id)
            made.append(node_id)
        save(state, data)
        log_event(state, {'event': 'branch', 'tree': args.tree,
                          'round': tip_round + 1, 'parent': parent,
                          'nodes': made})
        print(' '.join(made))
    return 0


def apply_record(data, job_id, verdict, comparison):
    """Record a finished job verdict on loaded state; releases cores.

    Takes no lock; callers must hold the state slot. Returns (message, event).
    """
    if verdict not in VERDICTS:
        raise ValueError(f'verdict must be one of {VERDICTS}')
    comparison = Path(comparison)
    if not comparison.is_file():
        raise ValueError(f'comparison file missing: {comparison}')
    payload = json.loads(comparison.read_text())
    if payload.get('verdict') != verdict:
        raise ValueError('recorded verdict does not match comparison.json verdict')
    digest = hashlib.sha256(comparison.read_bytes()).hexdigest()
    job = data['jobs'].get(job_id)
    if job is None:
        raise ValueError(f'unknown job {job_id}')
    node = data['nodes'][job['node']]
    job['status'] = 'done' if verdict in (
        'keep', 'correctness_only', 'screen_pass') else 'failed'
    job['finished'] = _now()
    released = job['cores']
    job['cores'] = 0
    node['status'] = 'done' if job['status'] == 'done' else 'failed'
    node['verdict'] = verdict
    node['comparison_sha256'] = digest
    event = {'event': 'record', 'tree': node['tree'], 'node': job['node'],
             'job': job_id, 'verdict': verdict,
             'comparison_sha256': digest, 'cores_released': released}
    message = (f'{job["node"]} {verdict} (released {released} cores; '
               f'live {live_cores(data)}/{data["core_budget"]})')
    return message, event


def cmd_record(args):
    with slot(args.state) as state:
        data = load(state)
        message, event = apply_record(data, args.job, args.verdict,
                                      args.comparison)
        save(state, data)
        log_event(state, event)
        print(message)
    return 0


def cmd_cut(args):
    with slot(args.state) as state:
        data = load(state)
        tree = data['trees'].get(args.tree)
        if tree is None:
            raise ValueError(f'unknown tree {args.tree}')
        if tree['status'] not in ACTIVE_STATUSES:
            raise ValueError(f'tree {args.tree} is already {tree["status"]}')
        tree['status'] = 'cut'
        tree['cut_reason'] = args.reason
        save(state, data)
        log_event(state, {'event': 'cut', 'tree': args.tree,
                          'reason': args.reason})
        live = [j for j in data['jobs'].values()
                if data['nodes'][j['node']]['tree'] == args.tree
                and j['status'] in ('submitted', 'queued', 'running', 'unknown')]
        print(f'{args.tree} cut; {len(live)} live job(s) keep their '
              f'reservations until poll records them (qdel needs approval)')
    return 0


def cmd_merge(args):
    with slot(args.state) as state:
        data = load(state)
        if args.merge_from == args.merge_into:
            raise ValueError('cannot merge a tree into itself')
        src = data['trees'].get(args.merge_from)
        dst = data['trees'].get(args.merge_into)
        if src is None or dst is None:
            raise ValueError('unknown tree in merge')
        if src['status'] not in ACTIVE_STATUSES or dst['status'] not in ACTIVE_STATUSES:
            raise ValueError('merge needs two active (live or adopted) trees')
        src['status'] = 'merged'
        src['merged_into'] = args.merge_into
        src['cut_reason'] = f'merged: {args.reason}'
        graft = None
        if args.graft_node:
            if args.graft_node not in data['nodes']:
                raise ValueError(f'unknown graft node {args.graft_node}')
            tip_round = max([n['round'] for n in data['nodes'].values()
                             if n['tree'] == args.merge_into] or [0])
            seq_base = max([n['seq'] for n in data['nodes'].values()
                            if n['tree'] == args.merge_into] or [0])
            graft = f'N{data["next_node_id"]}'
            data['next_node_id'] += 1
            data['nodes'][graft] = {
                'tree': args.merge_into, 'round': tip_round + 1,
                'kind': 'variant', 'seq': seq_base + 1,
                'parent': (tree_tip(data, args.merge_into) or {}).get('id'),
                'merged_from': [args.merge_from, args.graft_node],
                'hypothesis': f'merge of {args.merge_from}/{args.graft_node}: {args.reason}',
                'source_sha256': data['nodes'][args.graft_node]['source_sha256'],
                'status': 'design', 'job_id': None, 'cores': 0,
                'verdict': None, 'comparison_sha256': None,
                'id': graft,
            }
            dst['nodes'].append(graft)
        save(state, data)
        log_event(state, {'event': 'merge', 'from': args.merge_from,
                          'into': args.merge_into, 'graft': graft,
                          'reason': args.reason})
        print(f'{args.merge_from} merged into {args.merge_into}'
              + (f' graft {graft}' if graft else ''))
    return 0


def cmd_adopt(args):
    with slot(args.state) as state:
        data = load(state)
        tree = data['trees'].get(args.tree)
        if tree is None:
            raise ValueError(f'unknown tree {args.tree}')
        if tree['status'] not in ('live', 'adopted'):
            raise ValueError(f'tree {args.tree} is {tree["status"]}')
        for tid, other in data['trees'].items():
            if tid != args.tree and other['status'] == 'adopted':
                if not args.supersede:
                    raise ValueError(
                        f'{tid} already adopted; use --supersede to replace')
                other['status'] = 'superseded'
                log_event(state, {'event': 'supersede', 'tree': tid,
                                  'by': args.tree})
        tree['status'] = 'adopted'
        tree['adopt_reason'] = args.reason
        node = data['nodes'].get(args.node)
        if node is None or node['tree'] != args.tree:
            raise ValueError(f'champion node {args.node} is not on {args.tree}')
        if node['verdict'] not in ('keep', 'screen_pass', 'correctness_only'):
            raise ValueError(
                f'champion node {args.node} has verdict {node["verdict"]!r}; '
                'adopt a judged-improved node')
        data['champion'] = {'tree': args.tree, 'node': args.node,
                            'verdict': node['verdict'], 'ts': _now(),
                            'reason': args.reason}
        save(state, data)
        log_event(state, {'event': 'adopt', 'tree': args.tree,
                          'node': args.node, 'reason': args.reason})
        print(f'{args.tree} adopted at {args.node}; optimization continues '
              f'from the champion (never stops)')
    return 0


def cmd_purge(args):
    with slot(args.state) as state:
        data = load(state)
        tree = data['trees'].get(args.tree)
        if tree is None:
            raise ValueError(f'unknown tree {args.tree}')
        if tree['status'] not in ('cut', 'merged'):
            raise ValueError(
                f'purge refuses {args.tree} with status {tree["status"]}; '
                'only cut or merged routes are deleted')
        live = [jid for jid, j in data['jobs'].items()
                if data['nodes'][j['node']]['tree'] == args.tree
                and j['status'] in ('submitted', 'queued', 'running', 'unknown')]
        if live:
            raise ValueError(
                f'purge refuses {args.tree}: live jobs {live} still hold '
                'reservations; wait for poll to record them first')
        unjudged = [nid for nid in tree['nodes']
                    if data['nodes'][nid]['verdict'] is None]
        if unjudged and not args.force:
            raise ValueError(
                f'purge refuses {args.tree}: unjudged nodes {unjudged}; '
                'use --force only after confirming their outputs are fetched')
        campaign = check_remote_name('campaign', args.campaign)
        tid = check_remote_name('tree', args.tree)
        removed = []
        local_stage = staging_dir(state, args.tree)
        if local_stage.exists():
            import shutil
            shutil.rmtree(local_stage)
            removed.append(str(local_stage))
        # Fetched verdicts (outputs/<tree>/comparison.json) and results.jsonl
        # are the permanent record and are never deleted by purge.
        proc = subprocess.run(
            ['ssh', args.remote, 'rm', '-rf',
             f"$HOME/projects/sdpx-families/{campaign}/{tid}"],
            capture_output=True, text=True, timeout=120)
        if proc.returncode != 0:
            raise ValueError(f'remote purge failed: {proc.stderr.strip()}')
        removed.append(f'{args.remote}:~/projects/sdpx-families/'
                       f'{campaign}/{tid}')
        tree['purged'] = _now()
        save(state, data)
        log_event(state, {'event': 'purge', 'tree': args.tree,
                          'removed': removed})
        print(f'{args.tree} purged: ' + '; '.join(removed))
    return 0


def cmd_status(args):
    with slot(args.state) as state:
        data = load(state)
        if args.format == 'json':
            print(json.dumps({
                'live_trees': len(live_trees(data)),
                'max_live_trees': data['max_live_trees'],
                'live_cores': live_cores(data),
                'core_budget': data['core_budget'],
                'trees': data['trees'], 'nodes': data['nodes'],
                'jobs': data['jobs'], 'champion': data.get('champion')}, indent=2, sort_keys=True))
            return 0
        print(f'trees live {len(live_trees(data))}/{data["max_live_trees"]}  '
              f'cores live {live_cores(data)}/{data["core_budget"]}')
        champ = data.get('champion')
        if champ:
            print(f'champion {champ["tree"]}/{champ["node"]} '
                  f'({champ["verdict"]})')
        for tid in sorted(data['trees'], key=lambda t: int(t[1:])):
            tree = data['trees'][tid]
            depth = max([n['round'] for n in data['nodes'].values()
                         if n['tree'] == tid] or [0])
            verdicts = {}
            for nid in tree['nodes']:
                verdict = data['nodes'][nid]['verdict']
                if verdict:
                    verdicts[verdict] = verdicts.get(verdict, 0) + 1
            jobs = sum(1 for j in data['jobs'].values()
                       if data['nodes'][j['node']]['tree'] == tid
                       and j['status'] in ('submitted', 'queued', 'running',
                                           'unknown'))
            print(f'  {tid} {tree["status"]:9s} depth={depth} '
                  f'live_jobs={jobs} verdicts={verdicts or "-"} '
                  f'{tree["name"]}')
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', type=Path, default=default_state_dir())
    sub = parser.add_subparsers(dest='action')
    sub.required = True
    init = sub.add_parser('init')
    init.add_argument('--core-budget', type=int, default=DEFAULT_CORE_BUDGET)
    init.add_argument('--max-trees', type=int, default=DEFAULT_MAX_TREES)
    init.add_argument('--force', action='store_true')
    init.set_defaults(func=cmd_init)
    plant = sub.add_parser('plant')
    plant.add_argument('--name', required=True)
    plant.add_argument('--hypothesis', required=True)
    plant.add_argument('--source-sha', required=True)
    plant.set_defaults(func=cmd_plant)
    branch = sub.add_parser('branch')
    branch.add_argument('--tree', required=True)
    branch.add_argument('--ha', required=True, help='variant A hypothesis')
    branch.add_argument('--hb', required=True, help='variant B hypothesis')
    branch.add_argument('--sa', required=True, help='variant A source sha256')
    branch.add_argument('--sb', required=True, help='variant B source sha256')
    branch.add_argument('--parent', default=None)
    branch.set_defaults(func=cmd_branch)
    record = sub.add_parser('record')
    record.add_argument('--job', required=True)
    record.add_argument('--verdict', required=True, choices=VERDICTS)
    record.add_argument('--comparison', required=True)
    record.set_defaults(func=cmd_record)
    cut = sub.add_parser('cut')
    cut.add_argument('--tree', required=True)
    cut.add_argument('--reason', required=True)
    cut.set_defaults(func=cmd_cut)
    merge = sub.add_parser('merge')
    merge.add_argument('--from', dest='merge_from', required=True)
    merge.add_argument('--into', dest='merge_into', required=True)
    merge.add_argument('--reason', required=True)
    merge.add_argument('--graft-node', default=None)
    merge.set_defaults(func=cmd_merge)
    adopt = sub.add_parser('adopt')
    adopt.add_argument('--tree', required=True)
    adopt.add_argument('--reason', required=True)
    adopt.add_argument('--node', required=True,
                       help='judged-improved node becoming champion')
    adopt.add_argument('--supersede', action='store_true')
    adopt.set_defaults(func=cmd_adopt)
    purge = sub.add_parser('purge')
    purge.add_argument('--tree', required=True)
    purge.add_argument('--remote', default='hpc')
    purge.add_argument('--campaign', required=True)
    purge.add_argument('--force', action='store_true')
    purge.set_defaults(func=cmd_purge)
    status = sub.add_parser('status')
    status.add_argument('--format', choices=('text', 'json'), default='text')
    status.set_defaults(func=cmd_status)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (OSError, ValueError, KeyError) as error:
        print(f'Family stopped: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
