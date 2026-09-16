"""Tests for the async cluster bridge (hpc.py). Offline only: ssh/scp mocked."""
import importlib.util
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    sys.modules[name] = obj
    return obj


tree = module('tree', HERE / 'tree.py')
hpc = module('research_hpc_test', HERE / 'hpc.py')


def run(argv):
    stdout = io.StringIO()
    with redirect_stdout(stdout):
        code = hpc.main(argv)
    return code, stdout.getvalue()


def fake_snapshot(root):
    root = Path(root)
    for directory in ('crates', 'include', 'julia'):
        (root / directory).mkdir(parents=True)
    harness = root / 'benchmark' / 'research'
    harness.mkdir(parents=True)
    (harness / 'run.py').write_text('# harness\n')
    (root / 'target' / 'x').mkdir(parents=True)
    (root / 'target' / 'x' / 'big.o').write_text('build artifact')
    (root / '.git' / 'y').mkdir(parents=True)
    (root / '.git' / 'y' / 'z').write_text('vcs')
    return root


class HpcTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.state = str(Path(self.tmp.name) / 'fam')
        tree.main(['--state', self.state, 'init'])
        tree.main(['--state', self.state, 'plant', '--name', 't',
                   '--hypothesis', 'h', '--source-sha', '0' * 64])
        # Local scp needs absolute remote paths; seed the HOME cache so
        # no test touches the network.
        hpc._REMOTE_HOME['hpc'] = '/home/test'

    def tearDown(self):
        self.tmp.cleanup()

    def test_check_snapshot_rejects_non_root(self):
        with self.assertRaises(ValueError):
            hpc.check_snapshot(Path(self.tmp.name))
        snap = fake_snapshot(Path(self.tmp.name) / 'snap')
        self.assertEqual(hpc.check_snapshot(snap), snap.resolve())

    def test_stage_snapshot_excludes_build_and_vcs(self):
        snap = fake_snapshot(Path(self.tmp.name) / 'snap')
        tarball, digest = hpc.stage_snapshot(
            snap, Path(self.tmp.name) / 'stage')
        self.assertTrue(tarball.is_file())
        names = tarfile.open(tarball).getnames()
        self.assertIn('benchmark/research/run.py', names)
        self.assertFalse([n for n in names if n.startswith('target/')])
        self.assertFalse([n for n in names if n.startswith('.git/')])
        import hashlib
        self.assertEqual(digest, hashlib.sha256(
            tarball.read_bytes()).hexdigest())

    def test_render_job_no_tokens_left(self):
        args = SimpleNamespace(node='N1', cores=8, mem_gb=32,
                               walltime='04:00:00', profile='screen',
                               stage='development', threads=1,
                               julia_bin='/j/julia', features='f1',
                               campaign='demo')
        text = hpc.render_job(args, 'T1')
        self.assertNotIn('@@', text)
        self.assertIn('ppn=8', text)
        self.assertIn('$HOME/.cargo/bin/cargo', text)
        self.assertIn('sdpx-families/demo', text)
        self.assertIn('--stage development', text)

    def test_submit_validation_before_network(self):
        snap = str(fake_snapshot(Path(self.tmp.name) / 'snap'))
        base = ['--state', self.state, 'submit', '--campaign', 'demo',
                '--node', 'N1', '--snapshot', snap]
        self.assertEqual(hpc.main(base + ['--cores', '3']), 2)
        self.assertEqual(hpc.main(base + ['--cores', '8', '--profile',
                                          'screen', '--stage', 'holdout']), 2)
        self.assertEqual(hpc.main(base + ['--cores', '2', '--threads',
                                          '4']), 2)

    def add_live_job(self, status='queued', cores=8):
        data = tree.load(Path(self.state))
        data['jobs']['42.host'] = {'node': 'N1', 'cores': cores,
                                   'threads': 1, 'profile': 'screen',
                                   'stage': 'development',
                                   'campaign': 'demo', 'status': status}
        data['nodes']['N1']['status'] = status
        data['nodes']['N1']['job_id'] = '42.host'
        data['nodes']['N1']['cores'] = cores
        tree.save(Path(self.state), data)

    def test_poll_no_live_jobs(self):
        code, out = run(['--state', self.state, 'poll',
                         '--campaign', 'demo'])
        self.assertEqual(code, 0)
        self.assertIn('no live jobs', out)

    def test_poll_state_transitions(self):
        self.add_live_job('queued')
        fake = {'42.host': {'job_state': 'R'}}
        with patch.object(hpc, 'poll_remote', return_value=fake):
            code, out = run(['--state', self.state, 'poll',
                             '--campaign', 'demo'])
        self.assertEqual(code, 0)
        data = tree.load(Path(self.state))
        self.assertEqual(data['jobs']['42.host']['status'], 'running')
        self.assertEqual(data['nodes']['N1']['status'], 'running')

    def scp_with_comparison(self, verdict):
        def fake_run(argv, timeout=300):
            if 'comparison.json' in argv[2]:
                Path(argv[3]).write_text(json.dumps({'verdict': verdict}))
            return SimpleNamespace(returncode=0, stderr='')
        return fake_run

    def test_poll_completed_records_verdict(self):
        self.add_live_job('running')
        fake = {'42.host': {'job_state': 'C', 'exit_status': '0'}}
        with patch.object(hpc, 'poll_remote', return_value=fake), \
             patch.object(hpc, '_run',
                          side_effect=self.scp_with_comparison('keep')):
            code, out = run(['--state', self.state, 'poll',
                             '--campaign', 'demo'])
        self.assertEqual(code, 0)
        self.assertIn('N1 keep', out)
        data = tree.load(Path(self.state))
        self.assertEqual(data['jobs']['42.host']['status'], 'done')
        self.assertEqual(tree.live_cores(data), 0)

    def test_poll_completed_missing_comparison_incomplete(self):
        self.add_live_job('running')
        fake = {'42.host': {'job_state': 'C', 'exit_status': '0'}}
        with patch.object(hpc, 'poll_remote', return_value=fake), \
             patch.object(hpc, '_run', return_value=SimpleNamespace(
                 returncode=1, stderr='no such file')):
            code, out = run(['--state', self.state, 'poll',
                             '--campaign', 'demo'])
        self.assertEqual(code, 0)
        self.assertIn('incomplete', out)
        data = tree.load(Path(self.state))
        self.assertEqual(data['nodes']['N1']['verdict'], 'incomplete')
        self.assertEqual(tree.live_cores(data), 0)

    def test_vanished_without_verdict_goes_incomplete(self):
        self.add_live_job('running')
        fake = {'42.host': {'Unknown Job Id': 'Unknown Job Id 42.host'}}
        ssh_out = SimpleNamespace(returncode=0, stdout='', stderr='')
        with patch.object(hpc, 'poll_remote', return_value=fake), \
             patch.object(hpc, 'ssh', return_value=ssh_out):
            code, out = run(['--state', self.state, 'poll',
                             '--campaign', 'demo'])
        self.assertEqual(code, 0)
        self.assertIn('incomplete', out)
        data = tree.load(Path(self.state))
        self.assertEqual(data['jobs']['42.host']['status'], 'failed')
        self.assertEqual(data['nodes']['N1']['verdict'], 'incomplete')
        # Reaped jobs release their core reservation.
        self.assertEqual(tree.live_cores(data), 0)

    def test_remote_home_cached(self):
        hpc._REMOTE_HOME.pop('mockremote', None)
        with patch.object(hpc, 'ssh', return_value=SimpleNamespace(
                returncode=0, stdout='/public/home/u', stderr='')) as m:
            self.assertEqual(hpc._remote_home('mockremote'),
                             '/public/home/u')
            self.assertEqual(hpc._remote_home('mockremote'),
                             '/public/home/u')
            m.assert_called_once()
        del hpc._REMOTE_HOME['mockremote']
        with self.assertRaises(ValueError):
            with patch.object(hpc, 'ssh', return_value=SimpleNamespace(
                    returncode=1, stdout='', stderr='boom')):
                hpc._remote_home('mockremote')

    def test_fetch_unknown_job(self):
        code, _ = run(['--state', self.state, 'fetch', '--job', 'nope'])
        self.assertEqual(code, 2)


if __name__ == '__main__':
    unittest.main()
