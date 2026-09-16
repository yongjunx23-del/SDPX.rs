"""Metadata/integrity tests; never start a numerical solver."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location('catalog', Path(__file__).parents[1]/'catalog.py')
c = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(c)

class CatalogTests(unittest.TestCase):
    def setUp(self):
        self.catalog = c.load_catalog()

    def load(self, cases):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'catalog.json'
            extras = copy.deepcopy(c.select_cases(self.catalog, 'development'))
            path.write_bytes(c.encoded(dict(schema_version=1, cases=cases + extras,
                selection_policy=self.catalog['selection_policy'])))
            return c.load_catalog(path)

    def test_screen_selection_is_fixed_order_and_strict(self):
        names = [x['name'] for x in c.screen_cases(self.catalog)]
        self.assertEqual(set(names), {'LP_afiro', 'SOCP_sambal', 'SDP_truss1'})
        self.assertEqual(names, [x['name'] for x in self.catalog['cases'] if x['name'] in names])
        for invalid in ([], ['missing'], ['smoke_LP'], ['LP_afiro'] * 5,
                        [x['name'] for x in c.select_cases(self.catalog, 'development')]):
            changed = copy.deepcopy(self.catalog)
            changed['selection_policy']['screen_cases'] = invalid
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'catalog.json'
                path.write_bytes(c.encoded(changed))
                with self.assertRaisesRegex(ValueError, 'screen_cases'):
                    c.load_catalog(path)

    def test_exposed_sets_and_balanced_development(self):
        cases=c.select_cases(self.catalog,'regression')
        self.assertEqual(len(cases),55)
        self.assertEqual(len({x['json_sha256'] for x in cases}),len(cases))
        self.assertEqual(set().union(*(set(x['legacy_sets']) for x in cases)),
                         {'conic18','conic10','holdout','sdplib-scale3','refresh-20260916'})
        for family in ('LP','SOCP','SDP'):
            self.assertEqual(sum(x['family']==family for x in c.select_cases(self.catalog,'development')),3)
        for x in cases:self.assertNotIn('holdout',x['roles'])

    def test_deduplication_and_conflicts(self):
        a=copy.deepcopy(self.catalog['cases'][0]);b=copy.deepcopy(a)
        b['name']='alias';b['roles']=['development']
        result=self.load([a,b])['cases'][:1]
        self.assertEqual(len(result),1)
        self.assertEqual(result[0]['aliases'],['alias'])
        self.assertIn('development',result[0]['roles'])
        b['name']=a['name'];b['json_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'conflicting'):self.load([a,b])

    def test_exposed_holdout_rejected(self):
        a=copy.deepcopy(c.select_cases(self.catalog,'regression')[0])
        a['roles'].append('holdout');a['reservation']={'date':'2026-09-15'}
        with self.assertRaisesRegex(ValueError,'exposed'):self.load([a])

    def test_smoke_analytic_primal_dual_witnesses(self):
        witnesses={'LP':([1,2],[0,0],[1,1],3),
                   'SOCP':([5],[5,3,4],[1,-.6,-.8],5),
                   'SDP':([2],[1,0,0],[0,0,1],2)}
        for family,(x,s,z,optimum) in witnesses.items():
            p=c.smoke_problem(family);A=p['A'];Ax=[0]*A['m'];Atz=[0]*A['n']
            for j in range(A['n']):
                for k in range(A['colptr'][j],A['colptr'][j+1]):
                    i=A['rowval'][k];v=A['nzval'][k];Ax[i]+=v*x[j];Atz[j]+=v*z[i]
            for lhs,rhs in zip([a+b for a,b in zip(Ax,s)],p['b']):self.assertAlmostEqual(lhs,rhs)
            for v,q in zip(Atz,p['q']):self.assertAlmostEqual(v+q,0)
            self.assertAlmostEqual(sum(q*v for q,v in zip(p['q'],x)),optimum)
            self.assertAlmostEqual(-sum(b*v for b,v in zip(p['b'],z)),optimum)
            for v in (s,z):
                if family=='LP':self.assertGreaterEqual(min(v),0)
                elif family=='SOCP':self.assertGreaterEqual(v[0]+1e-14,math.hypot(*v[1:]))
                else:
                    self.assertGreaterEqual(min(v[0],v[2]),0)
                    self.assertGreaterEqual(v[0]*v[2]-v[1]**2/2,0)

    def test_materialize_integrity_and_empty_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);workspace=root/'workspace';workspace.mkdir();output=root/'cache'
            manifest=c.materialize(self.catalog,workspace,output,'smoke')
            obj=json.loads(manifest.read_text())
            self.assertEqual(obj['suite']['independent_oracle_tolerance'],1e-6)
            for case in obj['instances']:
                self.assertEqual(c.digest((output/case['json_path']).read_bytes()),case['json_sha256'])
            with self.assertRaisesRegex(ValueError,'empty'):c.materialize(self.catalog,workspace,output,'smoke')
            with self.assertRaisesRegex(ValueError,'outside'):c.materialize(self.catalog,workspace,workspace/'bad','smoke')
            case=copy.deepcopy(obj['instances'][0]);case['source']={'kind':'workspace','path':'input.json'}
            (workspace/'input.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):
                c.materialize(dict(cases=[case]),workspace,root/'bad','smoke')
            self.assertFalse((root/'bad').exists())

    def test_consumed_reservation_is_regression_and_compressed_integrity(self):
        self.assertEqual([x['name'] for x in c.select_cases(self.catalog,'holdout')],['LP_ship04s','SDP_copo14','SDP_filter48_socp','SOCP_strictmin_2D_43_dual'])
        workspace=Path(__file__).resolve().parents[4]
        exposed=[x for x in self.catalog['cases'] if x.get('exposure')]
        self.assertEqual(len(exposed),9)
        for case in exposed:
            self.assertIn('legacy_regression',case['roles'])
            self.assertNotIn('holdout',case['roles'])
            self.assertEqual(c.digest(c._payload(case,workspace)),case['json_sha256'])
        with tempfile.TemporaryDirectory() as tmp:
            reserved=c.materialize(self.catalog,workspace,Path(tmp)/'reserved','holdout')
            manifest=json.loads(reserved.read_text())
            self.assertEqual([x['name'] for x in manifest['instances']],['LP_ship04s','SDP_copo14','SDP_filter48_socp','SOCP_strictmin_2D_43_dual'])
            for case in manifest['instances']:
                self.assertNotIn('exposure',case)
                self.assertEqual(c.digest(c._payload(case,workspace)),case['json_sha256'])
            empty=dict(self.catalog,cases=[x for x in self.catalog['cases'] if 'holdout' not in x['roles']])
            with self.assertRaisesRegex(ValueError,'no reserved cases'):
                c.materialize(empty,workspace,Path(tmp)/'cache','holdout')

    def test_mpfr_recipes_have_no_fictional_input_hash(self):
        recipes=c.select_cases(self.catalog,'mpfr-dev')
        self.assertEqual(len(recipes),3)
        for case in recipes:
            self.assertNotIn('json_sha256',case)
            self.assertEqual(case['precision_bits'],[256,512])
            self.assertEqual(case['source']['kind'],'driver')
            self.assertNotIn('holdout',case['roles'])

if __name__=='__main__':unittest.main()
