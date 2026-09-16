#!/usr/bin/env python3
"""Pinned benchmark inputs; no solver execution or outcome-based selection."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

DEFAULT_CATALOG = Path(__file__).with_name('catalog.json')
SUITES = ('smoke', 'development', 'regression', 'holdout', 'mpfr-dev')

def digest(data):
    return hashlib.sha256(data).hexdigest()

def _is_relative_to(path, base):
    # Path.is_relative_to is 3.9+; cluster nodes run 3.6.
    try:
        Path(path).relative_to(base)
        return True
    except ValueError:
        return False

def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()

def _csc(m, n, entries):
    entries = sorted(entries, key=lambda x: (x[1], x[0]))
    return dict(m=m, n=n, colptr=[sum(c < j for _,c,_ in entries) for j in range(n+1)],
                rowval=[r for r,_,_ in entries], nzval=[v for _,_,v in entries])

def smoke_problem(kind):
    """Three analytic minima in min q'x, Ax+s=b, s in K convention."""
    if kind == 'LP':
        n,m,q,b,cones,entries = 2,2,[1.,1.],[-1.,-2.],[{'NonnegativeConeT':2}],[(0,0,-1.),(1,1,-1.)]
    elif kind == 'SOCP':
        n,m,q,b,cones,entries = 1,3,[1.],[0.,3.,4.],[{'SecondOrderConeT':3}],[(0,0,-1.)]
    elif kind == 'SDP':
        # Packed upper svec of diag(t-1,t-2); off diagonal is exactly zero.
        n,m,q,b,cones,entries = 1,3,[1.],[-1.,0.,-2.],[{'PSDTriangleConeT':2}],[(0,0,-1.),(2,0,-1.)]
    else:
        raise ValueError('unknown smoke family')
    return dict(P=_csc(n,n,[]),q=q,A=_csc(m,n,entries),b=b,cones=cones)

def load_catalog(path=DEFAULT_CATALOG):
    catalog=json.loads(Path(path).read_text())
    if catalog.get('schema_version') != 1:
        raise ValueError('unsupported catalog schema')
    # Deduplicate exact identities, preserving aliases and historical membership.
    names, hashes, cases = {}, {}, []
    for original in catalog['cases']:
        case=dict(original)
        name=case['name']
        sha=case.get('json_sha256') or 'recipe:'+digest(encoded(case['source']))+digest(encoded(case.get('parameters',{})))
        if name in names and names[name] != sha:
            raise ValueError(f'conflicting input hashes for {name}')
        names[name]=sha
        if sha in hashes:
            kept=hashes[sha]
            if any(kept[k]!=case[k] for k in ('family','n','m','cones')):
                raise ValueError('identical input hash has inconsistent structure')
            kept['aliases']=sorted(set(kept.get('aliases',[])+[name])-{kept['name']})
            kept['origins']=kept.get('origins',[])+case.get('origins',[])
            for key in ('roles','legacy_sets'):
                kept[key]=sorted(set(kept.get(key,[])+case.get(key,[])))
        else:
            hashes[sha]=case
            cases.append(case)
    for case in cases:
        if 'holdout' in case['roles'] and (case.get('legacy_sets') or 'development' in case['roles'] or 'legacy_regression' in case['roles']):
            raise ValueError('exposed inputs cannot be fresh holdout')
        if 'holdout' in case['roles'] and not case.get('reservation'):
            raise ValueError('fresh holdout needs explicit reservation provenance')
    result = dict(catalog,cases=cases)
    screen_cases(result)
    return result

def screen_cases(catalog):
    policy = catalog.get('selection_policy', {})
    names = policy.get('screen_cases') if isinstance(policy, dict) else None
    if (not isinstance(names, list) or not 1 <= len(names) <= 4
            or any(not isinstance(n, str) for n in names) or len(set(names)) != len(names)):
        raise ValueError('screen_cases must declare 1 to 4 unique development names')
    development = {c['name'] for c in catalog['cases'] if 'development' in c['roles']}
    if not set(names) < development:
        raise ValueError('screen_cases must be a strict subset of development cases')
    return [c for c in catalog['cases'] if c['name'] in names]

def select_cases(catalog, suite):
    if suite not in SUITES:
        raise ValueError(f'unknown suite: {suite}')
    role='legacy_regression' if suite=='regression' else suite
    return [case for case in catalog['cases'] if role in case['roles']]

def _payload(case, workspace):
    if case.get('runner','float64') != 'float64':
        source=(workspace/case['source']['path']).resolve()
        if not _is_relative_to(source, workspace) or digest(source.read_bytes()) != case['source']['sha256']:
            raise ValueError(f"recipe source hash mismatch: {case['name']}")
        return None
    if case['source']['kind']=='generated':
        data=encoded(smoke_problem(case['family']))
    elif case['source']['kind']=='workspace':
        source=(workspace/case['source']['path']).resolve()
        if not _is_relative_to(source, workspace):
            raise ValueError('input source escapes workspace')
        data=source.read_bytes()
        if case['source'].get('compression') == 'gzip':
            data=gzip.decompress(data)
    else:
        raise ValueError('unknown source kind')
    if digest(data)!=case['json_sha256']:
        raise ValueError(f"input hash mismatch: {case['name']}")
    obj=json.loads(data)
    if (obj['A']['n'],obj['A']['m'],len(obj['A']['nzval']),obj['cones']) != (case['n'],case['m'],case['nnz_A'],case['cones']):
        raise ValueError(f"input structure mismatch: {case['name']}")
    return data

def verify(catalog, workspace, suite):
    workspace=Path(workspace).resolve()
    cases=select_cases(catalog,suite)
    if not cases:
        raise ValueError(f'{suite} has no reserved cases; legacy holdout is exposed development data')
    for case in cases:
        _payload(case,workspace)
    return cases

def materialize(catalog, workspace, output, suite):
    workspace, output=Path(workspace).resolve(),Path(output).resolve()
    if _is_relative_to(output, workspace):
        raise ValueError('materialization cache must be outside the workspace')
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError('output must be an empty directory')
    cases=verify(catalog,workspace,suite)  # Fail integrity preflight before writing.
    output.mkdir(parents=True,exist_ok=True)
    (output/'json').mkdir()
    records=[]
    for index,case in enumerate(cases):
        if case.get('runner','float64') != 'float64':
            records.append(dict(case))
            continue
        name=case['name']
        if not name or any(ch not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for ch in name):
            raise ValueError('case name is not safe for a JSON filename')
        relative=f'json/{name}.json'
        data=_payload(case,workspace)
        (output/relative).write_bytes(data)
        records.append(dict(case,json_path=relative))
    manifest=dict(suite=dict(name=f'research-{suite}-v1', independent_oracle_tolerance=1e-6,
        precision='MPFR' if suite=='mpfr-dev' else 'Float64',threads=1,repetitions=4,order_blocks=['AB','BA'],total_samples_per_arm=8,
        timeout_seconds_per_case_and_solver=900,memory_limit_mib=4096,
        denominator=len(records),pilot=[c['name'] for c in records],
        selection_basis='pinned metadata only; no solver outcomes',role=suite,
        catalog_sha256=digest(encoded(catalog))),instances=records)
    path=output/'manifest.json';path.write_bytes(encoded(manifest));return path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('list','verify','materialize'))
    p.add_argument('--catalog',type=Path,default=DEFAULT_CATALOG)
    p.add_argument('--suite',choices=SUITES,required=True)
    p.add_argument('--workspace',type=Path)
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    try:
        catalog=load_catalog(args.catalog)
        if args.command=='list':
            print(json.dumps(select_cases(catalog,args.suite),indent=2));return
        if args.workspace is None:p.error('--workspace is required')
        if args.command=='verify':
            print(json.dumps(dict(verified=len(verify(catalog,args.workspace,args.suite)))));return
        if args.output is None:p.error('--output is required')
        print(materialize(catalog,args.workspace,args.output,args.suite))
    except (OSError,ValueError,KeyError) as error:
        p.error(str(error))

if __name__=='__main__':main()
