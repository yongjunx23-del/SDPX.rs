"""Test the actual catalog/controller boundary and execution binding, without solvers."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


run = module('research_run_test', HERE / 'run.py')
catalog = module('research_catalog_test', HERE / 'catalog.py')
fixtures = module('research_eval_fixtures', HERE / 'tests/test_evaluate.py')


class RunTests(unittest.TestCase):
    def test_materialized_name_mapping_and_failures_retained(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = catalog.materialize(catalog.load_catalog(), HERE.parents[2], root / 'data', 'smoke')
            entries = run.read(manifest)['instances']
            run.validate_catalog(run.read(manifest))
            tampered=run.read(manifest)
            tampered['instances'][0]['roles']=['development']
            with self.assertRaisesRegex(ValueError,'membership/roles'):
                run.validate_catalog(tampered)
            out = root / 'logs'; out.mkdir()
            config = dict(source=str(root), library=str(root / 'lib'), julia=sys.executable,
                          julia_project=str(root), blas='default')
            def owned(command, **kwargs):
                source = Path(command[-3])
                # The retained runner names output after the actual file stem.
                header = dict(impl='SDPX Julia/Rust', julia='1.12.6', blas_config='test-BLAS')
                sample = dict(status='optimal', **{'pass': True}, settings={'precision_bits':53}, e2e_s=0.01)
                row = dict(instance=source.stem, **{'pass':True}, runs=[sample] * 4)
                kwargs['stdout'].write(json.dumps(header) + '\n' + json.dumps(row) + '\n')
                return dict(process_exit_code=0, cleanup_confirmed=True, native_highwater={'max_rss_bytes':1024})
            for entry in entries:
                rows = run.run_case(owned, config, {'source_sha256':'a'*64}, entry, root/'data',
                                    53, 1, 'AB', out, 60, 512)
                self.assertEqual(len(rows), 4)
                self.assertTrue(all(r['passed'] and r['case_id'] == entry['name'] for r in rows))
                self.assertTrue(all(r['status'] == 'Optimal' and r['raw_status'] == 'optimal' for r in rows))
            def failed(command, **kwargs):
                return dict(process_exit_code=1, cleanup_confirmed=True)
            rows = run.run_case(failed, config, {'source_sha256':'a'*64}, entries[0], root/'data',
                                53, 1, 'BA', out, 60, 512)
            self.assertEqual(len(rows), 4)
            self.assertTrue(all(r['passed'] is False for r in rows))

    def test_screen_configuration_and_catalog_restriction(self):
        args = SimpleNamespace(profile='screen', stage='development', threads=1,
                               timeout=None, budget_seconds=None)
        protocol = run.profile_options(args)
        self.assertEqual((args.timeout, args.budget_seconds), (120, 600))
        self.assertEqual(protocol['repetitions'], 2)
        self.assertEqual(protocol['blocks'], ['AB'])
        self.assertFalse(protocol['speed_credit'])
        args.timeout, args.budget_seconds = 900, 3600
        protocol = run.profile_options(args)
        self.assertEqual(protocol['budget_clamps']['timeout'], {'requested':900, 'effective':120})
        self.assertEqual(args.budget_seconds, 600)
        for stage, threads in [('smoke',1), ('regression',1), ('holdout',1),
                               ('development',2), ('development',4), ('development',8)]:
            args.stage, args.threads = stage, threads
            with self.assertRaisesRegex(ValueError, 'screen requires'):
                run.profile_options(args)
        pinned = catalog.load_catalog()
        entries = catalog.select_cases(pinned, 'development')
        manifest = dict(suite=dict(role='development', catalog_sha256=catalog.digest(catalog.encoded(pinned))),
                        instances=entries)
        self.assertEqual([e['name'] for e in run.profile_cases(manifest, 'screen')],
                         [e['name'] for e in catalog.screen_cases(pinned)])
        manifest['instances'] = entries[:-1]
        with self.assertRaisesRegex(ValueError, 'membership'):
            run.profile_cases(manifest, 'screen')
        full = SimpleNamespace(stage='development', threads=4, timeout=None, budget_seconds=None)
        self.assertEqual(run.profile_options(full)['repetitions'], 4)
        self.assertEqual((full.timeout, full.budget_seconds), (900,3600))

    def test_screen_pair_executes_ab_only_and_records_terminal_verdict(self):
        pinned = catalog.load_catalog()
        entries = catalog.select_cases(pinned, 'development')
        manifest = dict(suite=dict(role='development', denominator=len(entries),
            independent_oracle_tolerance=1e-6, catalog_sha256=catalog.digest(catalog.encoded(pinned))),
            instances=entries)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run.write(root/'manifest.json', manifest)
            config = root/'config.json'
            run.write(config, {'source':str(root/'source')})
            identity = {key:'a'*64 for key in ('source_sha256','artifact_sha256','environment_sha256')}
            for verdict, expected_code in [('screen_pass',0), ('screen_fail',1)]:
                args = SimpleNamespace(profile='screen', stage='development', threads=1, precision_bits=53,
                    timeout=900, budget_seconds=3600, memory_mib=4096, baseline=config, candidate=config,
                    data=root, output=root/verdict, state=root)
                real_loader = run.load_module
                def loader(name, path):
                    if name == 'sdpx_research_evaluate':
                        return SimpleNamespace(evaluate=lambda a,b: dict(verdict=verdict, reasons=[], speed_credit=False))
                    return real_loader(name,path)
                with patch.object(run, 'arm_identity', return_value=identity), \
                     patch.object(run, 'supervisor'), patch.object(run, 'run_case', return_value=[{'passed':True}]) as launch, \
                     patch.object(run, 'load_module', side_effect=loader), patch('builtins.print'):
                    self.assertEqual(run.pair(args), expected_code)
                expected = [e['name'] for e in catalog.screen_cases(pinned) for _ in range(2)]
                self.assertEqual([call.args[3]['name'] for call in launch.call_args_list], expected)
                self.assertTrue(all(call.args[7]=='AB' and call.kwargs['repetitions']==2 for call in launch.call_args_list))
                contract = run.read(args.output/'candidate.json')['contract']
                self.assertEqual(contract['budget_clamps']['budget_seconds']['requested'],3600)
                self.assertEqual(contract['campaign_budget_s'],600)
                self.assertFalse(contract['speed_credit'])
            self.assertEqual([json.loads(line)['verdict'] for line in (root/'results.jsonl').read_text().splitlines()],
                             ['screen_pass','screen_fail'])

    def test_screen_repetition_plumbing_and_failure_slots(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = catalog.materialize(catalog.load_catalog(), HERE.parents[2], root/'data', 'smoke')
            entry = run.read(path)['instances'][0]
            config = dict(source=temp, library=temp+'/lib', julia=sys.executable, julia_project=temp)
            def failed(command, **kwargs):
                self.assertIn('--runs=2', command)
                return dict(process_exit_code=1, cleanup_confirmed=True)
            rows = run.run_case(failed, config, {'source_sha256':'a'*64}, entry, root/'data',
                                53, 1, 'AB', root, 60, 512, repetitions=2)
            self.assertEqual([(r['phase'], r['repetition']) for r in rows], [('cold',0), ('warm',1)])
            self.assertTrue(all(not r['passed'] for r in rows))

    def test_stage_cannot_relabel_holdout(self):
        manifest = dict(suite={'role':'holdout', 'denominator':1, 'independent_oracle_tolerance':1e-6},
                        instances=[{'roles':['holdout']}])
        with self.assertRaises(ValueError):
            run.validate_stage(manifest, 'development')
        run.validate_stage(manifest, 'holdout')
        manifest['suite']['role'] = 'development'
        with self.assertRaises(ValueError):
            run.validate_stage(manifest, 'development')

    def test_generated_recipe_hash_checked_before_launch(self):
        entry = dict(name='recipe', family='LP', runner='orthant', source={'sha256':'0'*64},
                     parameters={'n':3,'rows_per_variable':2})
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            config=dict(source=temp,library=temp+'/lib',julia=sys.executable,julia_project=temp)
            def forbidden(*a, **kw):
                self.fail('must reject a stale recipe before launching')
            with self.assertRaisesRegex(ValueError,'recipe changed'):
                run.run_case(forbidden,config,{'source_sha256':'a'*64},entry,root,256,1,'AB',root,60,512)

    def test_float64_json_cannot_be_narrowed_precision_input(self):
        with self.assertRaises(ValueError):
            run.validate_cases({'instances':[{'name':'x','runner':'float64'}]}, 256)

    def test_runtime_and_path_provider_are_bound(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); source=root/'source'; source.mkdir()
            (source/'Cargo.toml').write_text('[workspace]\n')
            (source/'Cargo.lock').write_text('version=3\n')
            project=root/'env'; project.mkdir()
            provider=root/'provider'; provider.mkdir(); f=provider/'x.jl'; f.write_text('x=1')
            (project/'Project.toml').write_text('[deps]\n')
            (project/'Manifest.toml').write_text('[[deps.Provider]]\npath='+json.dumps(str(provider))+'\n')
            lib=root/'lib'; lib.write_text('test')
            config=dict(source=str(source),julia_project=str(project),library=str(lib),julia=sys.executable)
            with patch.dict(run.os.environ, {'JULIA_DEPOT_PATH':'/one'}):
                first=run.arm_identity(config)
            with patch.dict(run.os.environ, {'JULIA_DEPOT_PATH':'/two'}):
                second=run.arm_identity(config)
            self.assertNotEqual(first['environment_sha256'],second['environment_sha256'])
            f.write_text('x=2')
            self.assertNotEqual(first['provider_files'],run.arm_identity(config)['provider_files'])
            config['julia']='julia'
            with self.assertRaisesRegex(ValueError,'absolute executable'):
                run.arm_identity(config)

    def test_holdout_tracks_input_hashes_not_manifest_names(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); a=fixtures.campaign(); b=fixtures.campaign(True)
            for c in (a,b):
                c['contract']['stage']='development'
                c['identity']['harness_sha256']=run.harness_identity()
            report=fixtures.evaluate(a,b)
            run.write(root/'baseline.json',a);run.write(root/'candidate.json',b)
            run.write(root/'comparison.json',report)
            args=SimpleNamespace(development_result=root/'comparison.json', state=root,
                                  output=root/'output',precision_bits=256,threads=4)
            entries=[{'json_sha256':'e'*64,'source_sha256':'f'*64}]
            run.consume_holdout(args,b['identity'],'1'*64,entries)
            with self.assertRaisesRegex(ValueError,'already exposed'):
                run.consume_holdout(args,b['identity'],'2'*64,entries)
            b['contract']['stage']='smoke';run.write(root/'candidate.json',b)
            with self.assertRaisesRegex(ValueError,'does not qualify'):
                run.consume_holdout(args,b['identity'],'3'*64,[{'json_sha256':'d'*64}])

    def test_unconfirmed_cleanup_stops_further_launches(self):
        # Reuse the real supervisor closure, including its persistent blocked state.
        watchdog=module('watchdog_test',HERE.parent/'float64/run.py')
        class Child:
            pid=12345
            returncode=None
            def poll(self): return None
            def wait(self,timeout): raise run.subprocess.TimeoutExpired('x',timeout)
        with patch.object(watchdog.subprocess,'Popen',return_value=Child()) as launch, \
             patch.object(watchdog.os,'killpg',side_effect=OSError('probe failed')), \
             patch.object(watchdog.time,'sleep'):
            owned=watchdog.owned_supervisor(lambda _:0)
            receipt=owned(['fake'],env={},stdout=None,stderr=None,timeout=0,memory_limit_mib=0)
            self.assertFalse(receipt['cleanup_confirmed'])
            with self.assertRaises(watchdog.CleanupUnconfirmed):
                owned(['another'],env={},stdout=None,stderr=None,timeout=0,memory_limit_mib=0)
            self.assertEqual(launch.call_count,1)


if __name__=='__main__': unittest.main()
