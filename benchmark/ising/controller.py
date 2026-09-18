#!/usr/bin/env python3
"""Sequential, bounded condensed-Ising scaling campaign inside one PBS allocation."""
import hashlib, json, os, pathlib, re, signal, subprocess, time
from decimal import Decimal, localcontext

B = pathlib.Path(__file__).resolve().parents[2]
H = B / 'benchmark/ising'
D = B / 'results' / os.environ['PBS_JOBID']
D.mkdir(parents=True, exist_ok=False)
J = str(pathlib.Path.home() / 'tools/julia-1.12.6/bin/julia')
WIDTHS = (1, 2, 4, 8)
WARMED = 3
RSS_SAMPLE_SECONDS = 0.25
# Leave PBS time for the build, gates and final receipts. No unbounded followups.
deadline = time.monotonic() + min(3 * 3600, float(os.environ["SDPX_CAMPAIGN_DEADLINE_EPOCH"])-time.time())
cpus, physical = [], set()
for cpu in sorted(os.sched_getaffinity(0)):
    topology = pathlib.Path('/sys/devices/system/cpu/cpu%d/topology' % cpu)
    key = ((topology/'physical_package_id').read_text().strip(), (topology/'core_id').read_text().strip())
    if key not in physical:
        physical.add(key)
        cpus.append(cpu)
    if len(cpus) == 8:
        break
assert len(cpus) == 8
os.sched_setaffinity(0, set(cpus))
os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
    JULIA_NUM_GC_THREADS='1', SDPX_LIBRARY=str(B/'source/target/release/libsdpx.so'),
    SDPX_FROZEN_ROOT=str(B/'source'), SDPX_REFERENCE_AUDIT=str(B/'reference/sdpb1-audit.json'),
    # The README describes the sampled problem; without this the harness silently
    # falls back to the materialized CSC route and measures a different workload.
    SDPX_ISING_SAMPLED='1')


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def identity():
    result = {}
    for sub in ('source/crates', 'source/julia', 'source/include', 'source/Cargo.toml',
                'source/Cargo.lock', 'source/target/release/libsdpx.so', 'env', 'common-sdp',
                'benchmark', 'sdpb', 'reference', 'source-identity.json', 'input.sha256'):
        path = B/sub
        for p in ([path] if path.is_file() else sorted(path.rglob('*'))):
            if p.is_file():
                result[str(p.relative_to(B))] = hashlib.sha256(p.read_bytes()).hexdigest()
    for executable in (B/'sdpb', B/'source/target/release/libsdpx.so', pathlib.Path(J)):
        linkage = subprocess.check_output(['ldd', str(executable)]).decode()
        if 'not found' in linkage:
            raise RuntimeError('Unresolved dynamic library for ' + str(executable))
        for name in re.findall(r'(?:=>\s+)?(/\S+)\s+\(', linkage):
            library = pathlib.Path(name).resolve()
            result['external-library:' + str(library)] = hashlib.sha256(library.read_bytes()).hexdigest()
    result['julia-executable:' + J] = hashlib.sha256(pathlib.Path(J).read_bytes()).hexdigest()
    return result


def proc_snapshot():
    """Linux RSS pages, parent, process group and start-time identity; no external packages."""
    records = {}
    for p in pathlib.Path('/proc').iterdir():
        if not p.name.isdigit():
            continue
        try:
            text = (p/'stat').read_text()
            fields = text[text.rfind(')')+2:].split()
            records[int(p.name)] = (int(fields[1]), int(fields[2]), int(fields[19]),
                                    max(0, int(fields[21])) * os.sysconf('SC_PAGE_SIZE'))
        except (OSError, ValueError, IndexError, UnicodeError):
            pass  # A process may exit between directory enumeration and read.
    return records


class MemorySampler:
    def __init__(self, pid, directory):
        self.pid, self.directory = pid, directory
        self.known = set()
        self.peak_group = 0
        self.peak_members = []
        self.per_process = {}
        self.samples = 0

    def sample(self):
        records = proc_snapshot()
        # MPI may create separate process groups; follow descendants and rank receipts too.
        roots = {self.pid}
        for path in self.directory.glob('rank-*.json'):
            receipt = json.loads(path.read_text())
            pid = receipt['pid']
            if pid in records and records[pid][2] == receipt['start_ticks']:
                roots.add(pid)
        selected = {pid for pid, (_, group, start, _) in records.items()
                    if group == self.pid or (pid, start) in self.known or pid in roots}
        while True:
            children = {pid for pid, (parent, _, _, _) in records.items() if parent in selected}
            new = children - selected
            if not new:
                break
            selected.update(new)
        total = 0
        for pid in selected:
            _, _, start, rss = records[pid]
            self.known.add((pid, start))
            key = '%d:%d' % (pid, start)
            self.per_process[key] = max(self.per_process.get(key, 0), rss)
            total += rss
        self.samples += 1
        if total > self.peak_group:
            self.peak_group, self.peak_members = total, sorted(selected)

    def result(self):
        return {'sample_interval_seconds': RSS_SAMPLE_SECONDS, 'samples': self.samples,
                'observed_peak_group_rss_bytes': self.peak_group,
                'peak_group_pids': self.peak_members, 'per_process_peak_rss_bytes': self.per_process,
                'observed_max_single_process_rss_bytes': max(self.per_process.values(), default=0),
                'scope': 'Sum of resident pages of launcher/process-group descendants and MPI ranks; shared pages counted per process; sampled peak, not PSS or exact continuous maximum.'}

    def terminate(self, signum):
        self.sample()
        try:
            os.killpg(self.pid, signum)
        except ProcessLookupError:
            pass
        records = proc_snapshot()
        for pid, start in self.known:
            if pid in records and records[pid][2] == start:
                try:
                    os.kill(pid, signum)
                except ProcessLookupError:
                    pass


state = {'cpus': cpus, 'bits': 512, 'internal_tolerance': '1e-42', 'external_tolerance': '1e-30',
         'widths': WIDTHS, 'sdpx_warmed_repetitions': WARMED, 'sdpb_repetitions': 3,
         'runs': {}, 'accepted': False, 'stage': 'gate'}
# Capture the compute host explicitly; the PBS job suffix is only a server name.
try:
    qstat = subprocess.run(['qstat','-f',os.environ['PBS_JOBID']],stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=15)
    allowed = {'job_state','exec_host','Resource_List.nodes','Resource_List.mem',
               'Resource_List.walltime','resources_used.walltime','Exit_status'}
    fields, current = {}, None
    for line in qstat.stdout.decode('utf-8',errors='replace').splitlines():
        match = re.match(r'^\s*([A-Za-z_][A-Za-z0-9_.]*)\s*=\s*(.*)$',line)
        if match:
            current = match.group(1)
            if current in allowed:
                fields[current] = match.group(2).strip()
        elif current == 'exec_host' and line[:1].isspace():
            fields[current] += line.strip()
        else:
            current = None
    qstat_receipt = {'returncode':qstat.returncode,'fields':fields}
    # Do not retain raw stdout/stderr: qstat may expose Variable_List values.
except (OSError,subprocess.TimeoutExpired) as exc:
    qstat_receipt = {'error':str(exc)}
write(D/'host.json', {'hostname':os.uname().nodename,'uname':list(os.uname()),
    'pbs_job_id':os.environ['PBS_JOBID'],'pbs_nodefile':pathlib.Path(os.environ['PBS_NODEFILE']).read_text() if os.environ.get('PBS_NODEFILE') else None,
    'qstat':qstat_receipt,'selected_cpus':cpus,'physical_package_core_pairs':sorted(physical)})
before = identity()
write(D/'identity-before.json', before)
write(D/'state.json', state)
(D/'cpuinfo.txt').write_text(pathlib.Path('/proc/cpuinfo').read_text())
(D/'lscpu.txt').write_text(subprocess.check_output(['lscpu']).decode())


def run(name, args, timeout, affinity, extra_env=None):
    if deadline-time.monotonic() < 60:
        raise RuntimeError('Campaign time budget exhausted before ' + name)
    timeout = min(timeout, deadline-time.monotonic())
    d = D/name
    d.mkdir()
    env = dict(os.environ, JULIA_NUM_THREADS=str(len(affinity)),
               SDPX_RANK_CPUS=','.join(map(str, affinity)))
    if extra_env:
        env.update(extra_env)
    start = time.monotonic()
    with (d/'process.log').open('w') as log:
        process = subprocess.Popen(['/usr/bin/time', '-v', '-o', str(d/'time.txt')] + args,
            stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True,
            preexec_fn=lambda: os.sched_setaffinity(0, set(affinity)))
        sampler = MemorySampler(process.pid, d)
        timedout = False
        try:
            while process.poll() is None:
                sampler.sample()
                if time.monotonic()-start >= timeout:
                    timedout = True
                    sampler.terminate(signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        pass
                    # The launcher may exit before its descendants. Reap those too.
                    sampler.terminate(signal.SIGKILL)
                    process.wait()
                    break
                time.sleep(RSS_SAMPLE_SECONDS)
            sampler.sample()
            rc = process.wait()
        except BaseException:
            sampler.terminate(signal.SIGKILL)
            process.wait()
            raise
    memory = sampler.result()
    write(d/'memory.json', memory)
    result = {'returncode': rc, 'timeout': timedout, 'elapsed_seconds': time.monotonic()-start,
              'command': args, 'affinity': affinity, 'memory': memory}
    state['runs'][name] = result
    write(D/'state.json', state)
    if rc != 0 or timedout:
        raise RuntimeError('Failed or incomplete run: ' + name)
    return d, result


def julia(width):
    return [J, '--startup-file=no', '--gcthreads=1', '--threads='+str(width), '--project='+str(B/'env')]


def audit(path):
    value = json.loads(path.read_text())
    if value.get('accepted') is not True:
        raise RuntimeError('Failed numerical audit: ' + str(path))
    return value


try:
    run('gate', julia(1) + [str(H/'gate.jl')], 300, cpus[:1])
    expected_input = json.loads((B/'reference/input.json').read_text())['input_sha256']
    objectives = []
    state['stage'] = 'sdpx'
    previous_cell = 0
    for width in WIDTHS:
        # Once t1 establishes runtime, do not start an allocation-overrunning sweep.
        if previous_cell and deadline-time.monotonic() < previous_cell + 1200:
            raise RuntimeError('Insufficient bounded time for another candidate cell and SDPB references')
        name = 'sdpx-t%d' % width
        directory, result = run(name, julia(width) + [str(H/'run.jl'), str(B/'common-sdp'),
            'sdpx', str(D/name), str(width), str(WARMED)], 3550, cpus[:width])
        previous_cell = result['elapsed_seconds']
        case = json.loads((directory/'input.json').read_text())
        if case['input_sha256'] != expected_input:
            raise RuntimeError('Candidate input hash mismatch')
        for label in ['first'] + ['warmed-%d' % i for i in range(1, WARMED+1)]:
            objectives.append(audit(directory/(label+'-audit.json'))['objective'])
    # Candidate points all passed; now produce fresh matched reference repetitions.
    state['stage'] = 'sdpb'
    write(D/'state.json', state)
    for repetition in range(1, 4):
        for width in WIDTHS:
            name = 'sdpb-t%d-r%d' % (width, repetition)
            directory, _ = run(name, ['mpirun', '--bind-to', 'none', '-np', str(width),
                'python3', str(H/'rank_exec.py'), str(B/'sdpb'), '-s', str(B/'common-sdp'),
                '-o', str(D/name/'sdpb-out'), '--checkpointDir', str(D/name/'checkpoint'),
                '--writeSolution=x,y,X,Y', '--precision', '512', '--dualityGapThreshold', '1e-42',
                '--primalErrorThreshold', '1e-42', '--dualErrorThreshold', '1e-42', '--maxIterations', '1000'],
                900, cpus[:width], {'SDPX_RANK_RECEIPTS': str(D/name)})
            precision = re.search(r'precision\s*\(actual\)\s*=\s*(\d+)\s*\(\s*(\d+)\s*\)',
                                  (directory/'process.log').read_text())
            if not precision or tuple(map(int, precision.groups())) != (512, 512):
                raise RuntimeError('SDPB actual precision mismatch')
            run('audit-'+name, julia(1) + [str(H/'run.jl'), str(B/'common-sdp'),
                'sdpb', str(directory), '1', '0'], 300, cpus[:1])
            objectives.append(audit(directory/'audit.json')['objective'])
    with localcontext() as context:
        context.prec = 170
        values = [Decimal(x) for x in objectives]
        if len(values) != 28 or not all(v.is_finite() for v in values):
            raise RuntimeError('Missing or nonfinite audited point')
        agreement = (max(values)-min(values))/max([Decimal(1)] + [abs(v) for v in values])
        state['objective_agreement'] = str(agreement)
        state['accepted'] = agreement <= Decimal('1e-30')
    state['stage'] = 'complete'
except Exception as exc:
    state['error'] = str(exc)
    state['accepted'] = False
finally:
    after = identity()
    write(D/'identity-after.json', after)
    state['immutable'] = before == after
    state['accepted'] = state['accepted'] and state['immutable']
    write(D/'state.json', state)
raise SystemExit(0 if state['accepted'] else 1)
