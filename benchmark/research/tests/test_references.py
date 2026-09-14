"""Reference orchestration only: mocked solvers, no numerical execution."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('research_references_test', HERE / 'references.py')
ref = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref)


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.real_validate_catalog=ref.common.validate_catalog
        guard=patch.object(ref.common,'validate_catalog')
        self.catalog_guard=guard.start()
        self.addCleanup(guard.stop)

    def fixture(self, root):
        workspace = root / 'workspace'; workspace.mkdir()
        data = root / 'data'; data.mkdir()
        entries = []
        for name in ('first', 'second'):
            path = data / (name + '.json'); path.write_text('{}')
            entries.append(dict(name=name, family='LP', json_path=path.name,
                                json_sha256=ref.common.sha(path), roles=['smoke']))
        ref.common.write(data / 'manifest.json', dict(suite=dict(role='smoke', denominator=2,
                         independent_oracle_tolerance=1e-6), instances=entries))
        return SimpleNamespace(workspace=workspace, data=data, output=root / 'output',
            state=root / 'state', engine='clarabel', binary=Path(sys.executable),
            source=workspace, python=None, provider_file=[], threads=1)

    def owned(self, commands, *, fail=False, cleanup=True):
        def call(cmd, **kw):
            commands.append((cmd, kw['env']))
            self.assertEqual(cmd[-2:], ['--runs=4', '--tol=1e-6'])
            self.assertEqual(kw['timeout'], 900)
            self.assertEqual(kw['memory_limit_mib'], 4096)
            rows = [dict(impl='reference', threads=1), dict(instance=Path(cmd[-3]).stem,
                    **{'pass':not fail}, runs=[{'pass':not fail, 'e2e_s':1.0}] * 4)]
            kw['stdout'].write('\n'.join(map(json.dumps, rows)))
            return dict(process_exit_code=1 if fail else 0, cleanup_confirmed=cleanup,
                        native_highwater=dict(complete=True, max_rss_bytes=4096))
        return call

    def test_fixed_defaults_raw_headers_and_all_case_denominator(self):
        with tempfile.TemporaryDirectory() as d:
            args = self.fixture(Path(d)); commands = []
            with patch.object(ref, 'identity', return_value={'python_providers':{'complete':True}}), \
                 patch.object(ref.common, 'supervisor', return_value=self.owned(commands)), \
                 patch.dict(ref.os.environ, {'SDPX_RUST_ARM':'controlled', 'OMP_NUM_THREADS':'8'}):
                result = ref.run(args)
            self.assertTrue(result['passed'])
            self.catalog_guard.assert_called_once()
            self.assertEqual(result['denominator'], 2)
            self.assertEqual(len(result['rows']), 2)
            self.assertEqual(result['rows'][0]['raw_rows'][0]['threads'], 1)
            self.assertEqual(commands[0][1]['SDPX_RUST_ARM'], 'reference_default')
            self.assertEqual(commands[0][1]['OMP_NUM_THREADS'], '1')
            self.assertEqual(result['rows'][0]['process']['native_highwater']['max_rss_bytes'], 4096)

    def test_failed_points_are_retained(self):
        with tempfile.TemporaryDirectory() as d:
            args = self.fixture(Path(d)); commands=[]
            with patch.object(ref, 'identity', return_value={'python_providers':{'complete':True}}), \
                 patch.object(ref.common, 'supervisor', return_value=self.owned(commands, fail=True)):
                result = ref.run(args)
            self.assertFalse(result['passed'])
            self.assertEqual(len(commands), 2)
            self.assertTrue(all(not r['passed'] and len(r['raw_rows'][1]['runs']) == 4 for r in result['rows']))

    def test_cleanup_blocks_next_launch_and_provider_probe(self):
        with tempfile.TemporaryDirectory() as d:
            args = self.fixture(Path(d)); commands=[]
            with patch.object(ref, 'identity', return_value={'python_providers':{'complete':True}}) as probe, \
                 patch.object(ref.common, 'supervisor', return_value=self.owned(commands, cleanup=False)):
                result = ref.run(args)
            self.assertEqual(len(commands), 1)
            self.assertEqual(probe.call_count, 1)
            self.assertFalse(result['passed'])
            self.assertTrue(result['rows'][1]['not_run'])

    def test_unavailable_provider_is_failure_for_every_case(self):
        with tempfile.TemporaryDirectory() as d:
            args = self.fixture(Path(d))
            with patch.object(ref, 'identity', return_value={'python_providers':{'complete':False}}), \
                 patch.object(ref.common, 'supervisor') as launch:
                result = ref.run(args)
            launch.assert_not_called()
            self.assertFalse(result['passed'])
            self.assertEqual(len(result['rows']), 2)
            self.assertTrue(all(r['not_run'] for r in result['rows']))

    def test_mosek_uses_stable_workspace_runner(self):
        with tempfile.TemporaryDirectory() as d:
            args = self.fixture(Path(d)); args.engine='mosek'; args.python=Path(sys.executable)
            commands=[]
            with patch.object(ref, 'identity', return_value={'python_providers':{'complete':True}}), \
                 patch.object(ref.common, 'supervisor', return_value=self.owned(commands)):
                result=ref.run(args)
            self.assertTrue(result['passed'])
            self.assertEqual(commands[0][0][:2], [str(args.python.resolve()),
                str(args.workspace.resolve()/'SDPX.jl/benchmark/mosek_runner.py')])

    def test_unpinned_catalog_is_rejected_before_launch(self):
        with tempfile.TemporaryDirectory() as d:
            args=self.fixture(Path(d))
            self.catalog_guard.side_effect=self.real_validate_catalog
            with patch.object(ref.common,'supervisor') as launch:
                with self.assertRaises((ValueError,KeyError)):
                    ref.run(args)
            launch.assert_not_called()

    def test_clarabel_source_artifact_and_provider_pins(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); args=self.fixture(root)
            (args.source/'Cargo.toml').write_text('[package]')
            (args.source/'Cargo.lock').write_text('version=3')
            args.binary=root/'binary'; args.binary.write_text('binary')
            provider=root/'provider'; provider.write_text('library'); args.provider_file=[provider]
            first=ref.identity(args,{})
            provider.write_text('changed library')
            self.assertNotEqual(first['provider_files'],ref.identity(args,{})['provider_files'])
            args.binary.write_text('changed binary')
            self.assertNotEqual(first['artifact_sha256'],ref.identity(args,{})['artifact_sha256'])
            (args.source/'Cargo.lock').write_text('changed dependency')
            self.assertNotEqual(first['source_sha256'],ref.identity(args,{})['source_sha256'])

    def test_changed_identity_invalidates_every_pass(self):
        with tempfile.TemporaryDirectory() as d:
            args = self.fixture(Path(d))
            with patch.object(ref, 'identity', side_effect=[{'python_providers':{'complete':True},'sha':1},
                                                          {'python_providers':{'complete':True},'sha':2}]), \
                 patch.object(ref.common, 'supervisor', return_value=self.owned([])):
                result = ref.run(args)
            self.assertFalse(result['passed'])
            self.assertFalse(result['identity_unchanged'])

    def test_holdout_and_disguised_reserved_hash_refused(self):
        with tempfile.TemporaryDirectory() as d:
            args = self.fixture(Path(d)); path=args.data/'manifest.json'
            data=ref.common.read(path); data['suite']['role']='holdout'; ref.common.write(path,data)
            with self.assertRaisesRegex(ValueError,'holdout'):
                ref.dataset(args.data)
            data['suite']['role']='smoke'
            held=next(c for c in ref.common.read(HERE/'catalog.json')['cases'] if 'holdout' in c['roles'])
            data['instances'][0]['json_sha256']=held['json_sha256']; ref.common.write(path,data)
            with self.assertRaisesRegex(ValueError,'holdout'):
                ref.dataset(args.data)

    def test_help_and_thread_cap(self):
        result=subprocess.run([sys.executable,str(HERE/'references.py'),'--help'], capture_output=True,text=True)
        self.assertEqual(result.returncode,0)
        self.assertIn('--threads {1}',result.stdout)


if __name__ == '__main__':
    unittest.main()
