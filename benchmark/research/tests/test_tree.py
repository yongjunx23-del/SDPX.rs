"""Tests for the family-tree campaign state machine (tree.py).

Offline only: state lives in a temp dir, purge's ssh call is mocked.
"""
import importlib.util
import json
import io
from contextlib import redirect_stdout
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


tree = module('research_tree_test', HERE / 'tree.py')


def run(argv):
    stdout = io.StringIO()
    with redirect_stdout(stdout):
        code = tree.main(argv)
    return code, stdout.getvalue()


class TreeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.state = str(Path(self.tmp.name) / 'fam')
        code, _ = run(['--state', self.state, 'init'])
        self.assertEqual(code, 0)

    def tearDown(self):
        self.tmp.cleanup()

    def data(self):
        return tree.load(Path(self.state))

    def plant(self, name='speed', hypothesis='h', sha='0' * 64):
        code, out = run(['--state', self.state, 'plant',
                         '--name', name, '--hypothesis', hypothesis,
                         '--source-sha', sha])
        self.assertEqual(code, 0)
        return out.split()  # [tree_id, node_id]

    def test_plant_branch_binary_round(self):
        tid, root = self.plant()
        code, out = run(['--state', self.state, 'branch', '--tree', tid,
                         '--ha', 'A', '--hb', 'B',
                         '--sa', 'a' * 64, '--sb', 'b' * 64])
        self.assertEqual(code, 0)
        kids = out.split()
        self.assertEqual(len(kids), 2)
        data = self.data()
        for kid in kids:
            node = data['nodes'][kid]
            self.assertEqual(node['parent'], root)
            self.assertEqual(node['round'], 1)
            self.assertEqual(node['status'], 'design')
        self.assertEqual(data['nodes'][kids[0]]['hypothesis'], 'A')

    def test_live_tree_cap(self):
        with tempfile.TemporaryDirectory() as temp:
            state = str(Path(temp) / 's')
            run(['--state', state, 'init', '--max-trees', '1'])
            run(['--state', state, 'plant', '--name', 'a',
                 '--hypothesis', 'h', '--source-sha', '0' * 64])
            code, _ = run(['--state', state, 'plant', '--name', 'b',
                         '--hypothesis', 'h', '--source-sha', '0' * 64])
            self.assertEqual(code, 2)

    def test_branch_cut_tree_refused(self):
        tid, _ = self.plant()
        run(['--state', self.state, 'cut', '--tree', tid, '--reason', 'bad'])
        code, _ = run(['--state', self.state, 'branch', '--tree', tid,
                         '--ha', 'A', '--hb', 'B',
                         '--sa', 'a' * 64, '--sb', 'b' * 64])
        self.assertEqual(code, 2)

    def add_job(self, node, cores=8):
        data = self.data()
        job_id = f'999.{node}'
        data['jobs'][job_id] = {'node': node, 'cores': cores,
                                'threads': 1, 'profile': 'screen',
                                'stage': 'development', 'campaign': 'demo',
                                'status': 'running'}
        data['nodes'][node]['status'] = 'running'
        data['nodes'][node]['job_id'] = job_id
        data['nodes'][node]['cores'] = cores
        tree.save(Path(self.state), data)
        return job_id

    def comparison(self, verdict):
        path = Path(self.tmp.name) / f'comparison-{verdict}.json'
        path.write_text(json.dumps({'verdict': verdict}))
        return str(path)

    def test_record_keep_releases_cores(self):
        tid, root = self.plant()
        job = self.add_job(root, cores=8)
        data = self.data()
        self.assertEqual(tree.live_cores(data), 8)
        code, out = run(['--state', self.state, 'record', '--job', job,
                         '--verdict', 'keep',
                         '--comparison', self.comparison('keep')])
        self.assertEqual(code, 0)
        self.assertIn('released 8 cores', out)
        data = self.data()
        self.assertEqual(tree.live_cores(data), 0)
        self.assertEqual(data['jobs'][job]['status'], 'done')
        self.assertEqual(data['nodes'][root]['verdict'], 'keep')

    def test_record_discard_marks_failed(self):
        _, root = self.plant()
        job = self.add_job(root)
        code, _ = run(['--state', self.state, 'record', '--job', job,
                       '--verdict', 'discard',
                       '--comparison', self.comparison('discard')])
        self.assertEqual(code, 0)
        data = self.data()
        self.assertEqual(data['jobs'][job]['status'], 'failed')
        self.assertEqual(data['nodes'][root]['status'], 'failed')

    def test_record_verdict_mismatch_refused(self):
        _, root = self.plant()
        job = self.add_job(root)
        code, _ = run(['--state', self.state, 'record', '--job', job,
                         '--verdict', 'keep',
                         '--comparison', self.comparison('discard')])
        self.assertEqual(code, 2)

    def test_adopt_needs_judged_improved_node(self):
        tid, root = self.plant()
        code, _ = run(['--state', self.state, 'adopt', '--tree', tid,
                         '--node', root, '--reason', 'too early'])
        self.assertEqual(code, 2)
        job = self.add_job(root)
        run(['--state', self.state, 'record', '--job', job,
             '--verdict', 'keep', '--comparison', self.comparison('keep')])
        code, out = run(['--state', self.state, 'adopt', '--tree', tid,
                         '--node', root, '--reason', 'wins'])
        self.assertEqual(code, 0)
        self.assertIn('never stops', out)
        data = self.data()
        self.assertEqual(data['trees'][tid]['status'], 'adopted')
        self.assertEqual(data['champion']['node'], root)
        # Adopted trees stay branchable.
        code, _ = run(['--state', self.state, 'branch', '--tree', tid,
                       '--ha', 'A', '--hb', 'B',
                       '--sa', 'a' * 64, '--sb', 'b' * 64])
        self.assertEqual(code, 0)

    def test_adopt_second_needs_supersede(self):
        t1, n1 = self.plant(name='one')
        t2, n2 = self.plant(name='two')
        for node in (n1, n2):
            job = self.add_job(node)
            run(['--state', self.state, 'record', '--job', job,
                 '--verdict', 'keep',
                 '--comparison', self.comparison('keep')])
        run(['--state', self.state, 'adopt', '--tree', t1,
             '--node', n1, '--reason', 'first'])
        code, _ = run(['--state', self.state, 'adopt', '--tree', t2,
                         '--node', n2, '--reason', 'second'])
        self.assertEqual(code, 2)
        code, _ = run(['--state', self.state, 'adopt', '--tree', t2,
                       '--node', n2, '--reason', 'second', '--supersede'])
        self.assertEqual(code, 0)
        data = self.data()
        self.assertEqual(data['trees'][t1]['status'], 'superseded')
        self.assertEqual(data['champion']['node'], n2)

    def test_merge_graft(self):
        t1, _ = self.plant(name='one')
        t2, n2 = self.plant(name='two')
        code, out = run(['--state', self.state, 'merge', '--from', t2,
                         '--into', t1, '--reason', 'combine',
                         '--graft-node', n2])
        self.assertEqual(code, 0)
        data = self.data()
        self.assertEqual(data['trees'][t2]['status'], 'merged')
        graft = out.split()[-1]
        self.assertEqual(data['nodes'][graft]['merged_from'], [t2, n2])
        self.assertEqual(data['nodes'][graft]['tree'], t1)
        code, _ = run(['--state', self.state, 'merge', '--from', t1,
                         '--into', t1, '--reason', 'self'])
        self.assertEqual(code, 2)

    def test_purge_guards(self):
        tid, root = self.plant()
        # Live tree refused.
        code, _ = run(['--state', self.state, 'purge', '--tree', tid,
                       '--campaign', 'demo'])
        self.assertEqual(code, 2)
        # Live jobs refused.
        run(['--state', self.state, 'cut', '--tree', tid, '--reason', 'bad'])
        self.add_job(root)
        code, _ = run(['--state', self.state, 'purge', '--tree', tid,
                       '--campaign', 'demo'])
        self.assertEqual(code, 2)

    def test_purge_removes_staging_keeps_outputs(self):
        tid, root = self.plant()
        run(['--state', self.state, 'cut', '--tree', tid, '--reason', 'bad'])
        state = Path(self.state)
        staged = tree.staging_dir(state, tid, root)
        staged.mkdir(parents=True)
        (staged / 'snapshot.tar.gz').write_text('x')
        outdir = tree.outputs_dir(state, tid, root)
        outdir.mkdir(parents=True)
        (outdir / 'comparison.json').write_text(
            json.dumps({'verdict': 'discard'}))
        with patch.object(tree.subprocess, 'run') as mock_run:
            mock_run.return_value = type('P', (),
                                         {'returncode': 0, 'stderr': ''})()
            code, _ = run(['--state', self.state, 'purge', '--tree', tid,
                           '--campaign', 'demo', '--force'])
        self.assertEqual(code, 0)
        mock_run.assert_called_once()
        remote_cmd = mock_run.call_args[0][0]
        self.assertIn('sdpx-families/demo/T1', remote_cmd[-1])
        self.assertFalse(tree.staging_dir(state, tid).exists())
        self.assertTrue((outdir / 'comparison.json').is_file())
        self.assertIn('purged', self.data()['trees'][tid])

    def test_remote_name_safety(self):
        for bad in ('../x', 'a b', 'a;b', '', '..', 'x' * 200):
            with self.assertRaises(ValueError):
                tree.check_remote_name('tree', bad)
        self.assertEqual(tree.check_remote_name('tree', 'T1-ok_2'), 'T1-ok_2')
        self.assertEqual(tree.check_remote_name('dest', '_base'), '_base')


if __name__ == '__main__':
    unittest.main()
