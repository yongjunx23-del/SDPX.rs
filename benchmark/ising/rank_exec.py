#!/usr/bin/env python3
import json, os, pathlib, sys
rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
cpus = list(map(int, os.environ['SDPX_RANK_CPUS'].split(',')))
allowed = sorted(os.sched_getaffinity(0))
if rank >= len(cpus) or cpus[rank] not in allowed:
    raise RuntimeError('MPI rank escaped selected physical core subset')
os.sched_setaffinity(0, {cpus[rank]})
actual = sorted(os.sched_getaffinity(0))
path = pathlib.Path(os.environ['SDPX_RANK_RECEIPTS']) / ('rank-%d.json'%rank)
stat=pathlib.Path('/proc/self/stat').read_text()
start_ticks=int(stat[stat.rfind(')')+2:].split()[19])
receipt=dict(rank=rank, inherited_cpus=allowed, actual_cpus=actual, pid=os.getpid(), parent_pid=os.getppid(), process_group=os.getpgrp(), start_ticks=start_ticks)
tmp=path.with_suffix('.tmp')
tmp.write_text(json.dumps(receipt)+'\n')
tmp.rename(path)
os.execv(sys.argv[1], sys.argv[1:])
