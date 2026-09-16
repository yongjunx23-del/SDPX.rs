#!/usr/bin/env python3
"""Bind an MPI rank to its preselected physical CPU and record its identity."""
import json
import os
import pathlib
import socket
import sys


def selected_cpu(env, hostname, allowed):
    rank = int(env['OMPI_COMM_WORLD_RANK'])
    local_rank = int(env['OMPI_COMM_WORLD_LOCAL_RANK'])
    size = int(env['OMPI_COMM_WORLD_SIZE'])
    if not 0 <= rank < size or local_rank < 0:
        raise ValueError('Invalid MPI rank identity')
    if env.get('SDPX_RANK_CPU_MAP'):
        # Hostnames and CPU IDs come from the allocation's topology preflight.
        # Never reuse another host's logical CPU numbers implicitly.
        mapping = json.loads(pathlib.Path(env['SDPX_RANK_CPU_MAP']).read_text())
        cpus = mapping[hostname]
    else:
        cpus = list(map(int, env['SDPX_RANK_CPUS'].split(',')))
        if int(env['OMPI_COMM_WORLD_LOCAL_SIZE']) != size:
            raise ValueError('Multi-node binding requires a per-host CPU map')
    if (not cpus or any(type(c) is not int or c < 0 for c in cpus)
            or len(set(cpus)) != len(cpus)):
        raise ValueError('CPU map must contain unique nonnegative CPU IDs')
    if local_rank >= len(cpus) or cpus[local_rank] not in allowed:
        raise ValueError('MPI rank escaped selected physical core subset')
    return rank, local_rank, cpus[local_rank]


def main():
    if len(sys.argv) < 2:
        raise ValueError('Expected executable and arguments')
    host = socket.gethostname()
    allowed = sorted(os.sched_getaffinity(0))
    rank, local_rank, cpu = selected_cpu(os.environ, host, allowed)
    os.sched_setaffinity(0, {cpu})
    actual = sorted(os.sched_getaffinity(0))
    if actual != [cpu]:
        raise RuntimeError('Rank affinity was not applied')
    stat = pathlib.Path('/proc/self/stat').read_text()
    start_ticks = int(stat[stat.rfind(')') + 2:].split()[19])
    receipt = dict(rank=rank, local_rank=local_rank, hostname=host,
                   inherited_cpus=allowed, actual_cpus=actual, pid=os.getpid(),
                   parent_pid=os.getppid(), process_group=os.getpgrp(),
                   start_ticks=start_ticks)
    # Global ranks are unique across hosts; local ranks are not.
    path = pathlib.Path(os.environ['SDPX_RANK_RECEIPTS']) / ('rank-%d.json' % rank)
    tmp = path.with_suffix('.tmp')
    with tmp.open('x') as f:
        json.dump(receipt, f)
        f.write('\n')
    tmp.rename(path)
    os.execv(sys.argv[1], sys.argv[1:])


if __name__ == '__main__':
    main()
