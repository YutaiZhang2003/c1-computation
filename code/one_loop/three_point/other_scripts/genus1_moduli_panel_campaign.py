"""Sharded, cost-balanced moduli experiments using a pinned CFT runtime.

Workers read only their small geometry batches. Geometry, physical weights,
spectral reuse and results have separate immutable identities. Matrix-model
values are never inputs to planning or quadrature acceptance.
"""

from genus1_source_paths import ROOT as _SOURCE_ROOT, source_path as _source_path
import argparse
from collections import Counter,defaultdict
from dataclasses import asdict
import heapq
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys

sys.dont_write_bytecode=True


def assign(costs,workers):
    """Longest-first allocation, covering each immutable unit exactly once."""
    if type(workers) is not int or workers<1:raise ValueError('positive worker count required')
    if any(not math.isfinite(c) or c<0 for c in costs):raise ValueError('finite nonnegative costs required')
    heap=[(0.,i) for i in range(workers)];jobs=[[] for _ in heap]
    for index in sorted(range(len(costs)),key=lambda i:(-costs[i],i)):
        load,worker=heapq.heappop(heap);jobs[worker].append(index)
        heapq.heappush(heap,(load+costs[index],worker))
    return jobs,[sum(costs[i] for i in row) for row in jobs]


def configure(root):
    global f,g,source,parents
    loc=json.loads((root/'locations.json').read_text());remote=bool(os.environ.get('SLURM_JOB_ID'))
    source=Path(loc['source_remote' if remote else 'source'])
    parents=[Path(row['remote' if remote else 'local']) for row in loc['parents']]
    sys.path.insert(0,str(source/'runtime/source/code/tasks'))
    sys.path.insert(0,str(source/'runtime/source/code/one_loop/three_point/other_scripts'))
    import genus1_fresh_analytic
    f=genus1_fresh_analytic;p=f.load(source)
    if f.fingerprint(p)!=loc['source_plan_sha256']:raise ValueError('changed CFT source plan')
    path=root/'pilot/genus1_moduli_panels.py'
    spec=importlib.util.spec_from_file_location('moduli_campaign_geometry',path)
    g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
    return p


def checked(root,p):
    d=f.spectrum.read(root/'design.json')
    if d['source_plan_sha256']!=f.fingerprint(p):raise ValueError('foreign CFT runtime')
    for rel,sha in d['pilot_sources'].items():
        if f.shared.sha(root/rel)!=sha:raise ValueError('changed moduli implementation')
    if len(parents)!=len(d['parent_identities']):raise ValueError('changed parent experiment count')
    for parent,identity in zip(parents,d['parent_identities']):
        if f.fingerprint(f.spectrum.read(parent/'design.json'))!=identity:raise ValueError('changed parent experiment')
    return d


def prepare(args):
    root=args.root;root.mkdir(parents=True,exist_ok=False);(root/'pilot').mkdir()
    shutil.copy2(Path(__file__),root/'pilot/genus1_moduli_panel_campaign.py')
    shutil.copy2(_source_path('code/one_loop/three_point/moduli_integration/genus1_moduli_panels.py'),root/'pilot/genus1_moduli_panels.py')
    old=json.loads((args.source/'plan.json').read_text())['payload']
    parent_rows=[dict(local=str(p.resolve()),remote=r) for p,r in zip(args.parent,args.parent_remote)]
    if len(args.parent)!=len(args.parent_remote):raise ValueError('each parent needs a remote location')
    (root/'locations.json').write_text(json.dumps(dict(source=str(args.source.resolve()),source_remote=args.source_remote,
        source_plan_sha256=json.loads((args.source/'plan.json').read_text())['sha256'],parents=parent_rows),indent=2)+'\n')
    p=configure(root);physics=f.make_plan(p['config']);atlas=f.make_atlas(physics.t)
    parent_designs=[f.spectrum.read(p/'design.json') for p in parents]
    if any(d['source_plan_sha256']!=f.fingerprint(p) for d in parent_designs):raise ValueError('incompatible parent CFT source')
    selected=parent_designs[0]['selected'];primary=selected['primary:']['cell'];pair=selected['bulk:pair_face_analytic']['cell']
    planner=f.spectrum.Planner(p,atlas);samples={};rules={};weights={};grids={};routing={};primary_cache={}
    catalogs=[]
    for i,base in enumerate([source,*parents]):
        evidence=base/'evidence'
        for ready in (evidence/'prepared/spectrum/grids').glob('*/ready.json'):
            catalogs.append((ready.parent.name,i,ready))
        for ready in (base/'prepared/spectrum/grids').glob('*/ready.json'):
            catalogs.append((ready.parent.name,i,ready))
    available={}
    for key,i,path in catalogs:available.setdefault(key,(i,path))
    def register(value,is_primary):
        cache_key=None
        if is_primary:
            cache_key=(value.sector,value.s1,value.s2)
            if cache_key in primary_cache:return primary_cache[cache_key]
        key,spec=planner.grid(value,is_primary,p['profile']);f.check_momentum_grid(p,spec)
        if key not in grids:
            grids[key]=dict(kind=spec['case']['kind'],axes=[len(r['nodes']) for r in spec['grid']['rules']])
            if key in available:
                origin,path=available[key];ready=f.spectrum.read(path)
                if ready['grid_sha256']!=key:raise ValueError('foreign bank receipt')
                routing[key]=dict(origin=origin,ready_sha256=f.shared.sha(path))
            else:f.spectrum.publish(f.spectrum.grid_directory(root,key)/'spec.json',spec)
        if cache_key is not None:primary_cache[cache_key]=key
        return key
    def add(name,sample,weight,tag='all'):
        if not math.isfinite(weight) or weight<=0:raise ValueError('invalid quadrature weight')
        key=f.fingerprint(sample)
        if key not in samples:
            if sample['kind']=='primary':values=[f.ShortPoint.from_record(sample['short'])]
            else:values=[f.request_from_record(r) for r in sample['requests']]
            samples[key]=dict(sample,grids=[register(v,sample['kind']=='primary') for v in values])
        weights[key]=max(weights.get(key,0.),weight)
        rules.setdefault(name,[]).append([key,float(weight),tag])
    if args.mode=='followup':
        # Independent length and angular checks, with no automatic refinement.
        for name,n,a in (('primary_l3_a3',3,3),('primary_l4_a3',4,3),('primary_l4_a4',4,4)):
            for short,w in g.primary_cell_rule(physics,primary,n,a,angular_heads=True):
                add(name,dict(kind='primary',short=short.record()),w)
        pair_rules=[('pair_h3_p3',(2,3,3,3)),('pair_h4_p3',(2,4,3,3)),('pair_h4_p4',(2,4,4,4))]
        comparisons=[('primary_lengths','primary_l3_a3','primary_l4_a3'),
                     ('primary_angles','primary_l4_a3','primary_l4_a4'),
                     ('pair_height','pair_h3_p3','pair_h4_p3'),('pair_puncture','pair_h4_p3','pair_h4_p4')]
    elif args.mode=='pair-real-check':
        # The other directions are completed parent results. Only this cell's
        # missing real-part check needs fresh CFT work; do not replay them.
        if (pair['term']!='bulk:pair_face_analytic' or
                pair['bounds']!=[[0.,1.],[0.,.5],[0.,1.],[0.,1.]]):
            raise ValueError('the lower-height bulk pair cell is required')
        pair_rules=[('pair_r3_h3_p3',(3,3,3,3))]
        comparisons=[]
    else:raise ValueError('explicit supported campaign mode required')
    for name,orders in pair_rules:
        for sample,meta in g.pair_rule(physics,'bulk',orders,bounds=pair['bounds'][:2],separation_splits=(2.,4.,8.)):
            add(name,dict(kind='collision',requests=f.encoded([asdict(r) for r in sample.requests])),sample.jacobian,meta['puncture'])
    fresh=sorted(set(grids)-set(routing));grouped=defaultdict(list)
    for key,s in samples.items():grouped[tuple(s['grids'])].append(key)
    batches=[];costs=[]
    for group in grouped.values():
        for start in range(0,len(group),32):
            keys=group[start:start+32];index=len(batches);rel=f'geometry/batch_{index:05d}.json'
            payload=dict(samples={k:dict(samples[k],max_weight=weights[k]) for k in keys})
            f.spectrum.publish(root/rel,payload)
            batches.append(dict(path=rel,sha256=f.shared.sha(root/rel),points=len(keys),point_ids_sha256=f.fingerprint(sorted(keys))))
            costs.append(len(keys)*(.111 if samples[keys[0]]['kind']=='primary' else .080))
    rule_path='geometry/rules.json';f.spectrum.publish(root/rule_path,rules)
    bank_costs=[(100. if grids[k]['kind']=='primary' else 20.3)*math.prod(grids[k]['axes'])/512 for k in fresh]
    allocations,loads=assign(bank_costs,args.workers);contracts,contract_loads=assign(costs,args.workers)
    if len(samples)>300000 or len(fresh)>6000:raise ValueError('bounded diagnostic work cap exceeded')
    if args.mode=='pair-real-check' and (len(samples)>8000 or len(fresh)>2500):
        raise ValueError('single pair-cell work cap exceeded')
    d=dict(format='genus1-sharded-moduli-campaign-v1',mode=args.mode,created_utc=f.shared.utc(),
        source_plan_sha256=f.fingerprint(p),parent_identities=[f.fingerprint(d) for d in parent_designs],
        pilot_sources={str(path.relative_to(root)):f.shared.sha(path) for path in sorted((root/'pilot').glob('*.py'))},
        selected=selected,physics=p['physics'],momentum_mode=p['momentum_mode'],elliptic_orders=p['elliptic_orders'],
        comparisons=comparisons,batches=batches,rules=dict(path=rule_path,sha256=f.shared.sha(root/rule_path)),
        fresh_banks=fresh,routing=routing,workers=args.workers,prepare=allocations,contract=contracts,
        summary=dict(points=len(samples),new_banks=len(fresh),reused_banks=len(routing),batches=len(batches),
            fresh_by_family=dict(Counter(grids[k]['kind'] for k in fresh)),
            banks_by_momentum_axes=dict(Counter(str(tuple(v['axes'])) for v in grids.values())),
            rule_points={k:len(v) for k,v in rules.items()},estimated_prepare_max_seconds=max(loads),
            estimated_contract_max_seconds=max(contract_loads)),
        full_amplitude_certified=False,physical_cutoffs_changed=False,momentum_design_changed=False)
    if args.mode=='pair-real-check':
        d['acceptance']=dict(candidate='pair_h3_p3',candidate_orders=[2,3,3,3],
            maximum_sum_directional_relative_indicator=.01,
            maximum_real_direction_relative_indicator=.005,
            interpretation='local stability indicators, not error bounds',
            automatic_refinement=False,matrix_model_used=False)
    f.spectrum.publish(root/'design.json',d);print(json.dumps(d['summary']),flush=True)


class Banks:
    def __init__(self,root,d):
        self.local=f.spectrum.Banks(root);self.inputs=[f.spectrum.Banks(p) for p in [source,*parents]]
        self.routing=d['routing'];self.verified=set()
    def get(self,key):
        if key not in self.routing:return self.local.get(key)
        route=self.routing[key];bank=self.inputs[route['origin']]
        if key not in self.verified:
            if f.shared.sha(f.spectrum.grid_directory(bank.output,key)/'ready.json')!=route['ready_sha256']:
                raise ValueError('changed reused CFT bank')
            self.verified.add(key)
        return bank.get(key)


def batch(root,d,index):
    row=d['batches'][index];path=root/row['path']
    if f.shared.sha(path)!=row['sha256']:raise ValueError('changed geometry batch')
    samples=f.spectrum.read(path)['samples']
    if len(samples)!=row['points'] or f.fingerprint(sorted(samples))!=row['point_ids_sha256']:
        raise ValueError('changed point coverage')
    return samples


def completed_worker(root,d,identity,phase,rank):
    """A terminal receipt is reusable only while its assigned work is intact."""
    receipt=root/f'timings/{phase}_{rank:03d}.json'
    if not receipt.exists():return False
    row=f.spectrum.read(receipt)
    if (row['design_sha256']!=identity or row['phase']!=phase or row['rank']!=rank):
        raise ValueError('foreign completed worker receipt')
    if phase=='prepare':
        for i in d['prepare'][rank]:f.spectrum.ready(root,d['fresh_banks'][i])
    elif phase=='contract':
        for i in d['contract'][rank]:
            samples=batch(root,d,i);result=f.spectrum.read(root/f'results/batch_{i:05d}.json')
            if result['design_sha256']!=identity or set(result['values'])!=set(samples):
                raise ValueError('completed worker has incomplete results')
    else:raise ValueError('unknown worker phase')
    return True


def worker(root,phase):
    p=configure(root);d=checked(root,p);identity=f.fingerprint(d);rank=int(os.environ['SLURM_PROCID'])
    if int(os.environ['SLURM_NTASKS'])!=d['workers'] or not 0<=rank<d['workers']:
        raise ValueError('changed worker allocation')
    if completed_worker(root,d,identity,phase,rank):return
    start=f.clock()
    if phase=='prepare':
        for i in d['prepare'][rank]:
            key=d['fresh_banks'][i]
            r=f.spectrum.build(root,p,key,cpu_budget=300,max_new_chunks=64,max_new_elliptic_nodes=1024)
            if not r['complete_rule']:raise RuntimeError(str(r))
    elif phase=='contract':
        from genus1_boundary_batch import GeometryStore
        physics=f.make_plan(p['config']);atlas=f.make_atlas(physics.t)
        geometry=GeometryStore(atlas,root/f'work/geometry/{rank:03d}')
        runtime=f.spectrum.Runtime(root,p,atlas,geometry,point_budget=1e-12);runtime.banks=Banks(root,d)
        try:
            for i in d['contract'][rank]:
                path=root/f'results/batch_{i:05d}.json';samples=batch(root,d,i)
                if path.exists():
                    r=f.spectrum.read(path)
                    if r['design_sha256']!=identity or set(r['values'])!=set(samples):raise ValueError('foreign result')
                    continue
                tick=f.clock();keys=list(samples);values={}
                if all(s['kind']=='primary' for s in samples.values()):
                    shorts=[f.ShortPoint.from_record(samples[k]['short']) for k in keys]
                    grids=[samples[k]['grids'][0] for k in keys]
                    vs=runtime.primary(shorts,grids,geometry.prepare(shorts),physics.policy,physics.angle_order)
                    values={k:f.boundary.amplitude_value(v,1.).record() for k,v in zip(keys,vs)}
                else:
                    for key,s in samples.items():
                        parts=[runtime.collide(f.request_from_record(a),grid,s['max_weight']).scaled(a['multiplicity'])
                            for a,grid in zip(s['requests'],s['grids'])]
                        values[key]=f.boundary.amplitude_value(f.combine(parts),1.).record()
                f.spectrum.publish(path,dict(design_sha256=identity,values=values,**f.timing(tick)))
        finally:geometry.close()
    else:raise ValueError('unknown worker phase')
    f.spectrum.publish(root/f'timings/{phase}_{rank:03d}.json',dict(design_sha256=identity,
        phase=phase,rank=rank,job_id=os.environ['SLURM_JOB_ID'],**f.timing(start)))


def reduce(root):
    p=configure(root);d=checked(root,p);identity=f.fingerprint(d);values={}
    for i,descriptor in enumerate(d['batches']):
        row=f.spectrum.read(root/f'results/batch_{i:05d}.json');data=row['values']
        if (row['design_sha256']!=identity or len(data)!=descriptor['points'] or
                f.fingerprint(sorted(data))!=descriptor['point_ids_sha256'] or set(values)&set(data)):
            raise ValueError('foreign, overlapping, or incomplete contraction')
        values.update(data)
    if len(values)!=d['summary']['points']:raise ValueError('incomplete geometry coverage')
    if f.shared.sha(root/d['rules']['path'])!=d['rules']['sha256']:raise ValueError('changed quadrature weights')
    rules=f.spectrum.read(root/d['rules']['path']);results={}
    for name,rows in rules.items():
        tags=sorted({tag for _,_,tag in rows})
        results[name]=dict(value=math.fsum(values[k]['value'][0]*w for k,w,_ in rows),
            block_indicator=math.fsum(values[k]['errors']['block']*w for k,w,_ in rows),
            physical_measure=math.fsum(w for _,w,_ in rows),points=len(rows),
            by_panel={tag:math.fsum(values[k]['value'][0]*w for k,w,t in rows if tag==t) for tag in tags})
    comparisons={}
    for name,low,high in d['comparisons']:
        a,b=results[low]['value'],results[high]['value'];delta=b-a
        comparisons[name]=dict(low=low,high=high,delta=delta,indicator=abs(delta),
            relative_indicator=abs(delta)/max(abs(b),1e-300))
    timings={}
    for phase in ('prepare','contract'):
        ts=[f.spectrum.read(root/f'timings/{phase}_{i:03d}.json') for i in range(d['workers'])]
        if any(t['design_sha256']!=identity for t in ts):raise ValueError('foreign timing')
        timings[phase]=dict(cpu_seconds=sum(t['cpu_seconds'] for t in ts),max_wall_seconds=max(t['wall_seconds'] for t in ts))
    r=dict(design_sha256=identity,source_plan_sha256=d['source_plan_sha256'],results=results,
        comparisons=comparisons,timings=timings,summary=d['summary'],full_amplitude_certified=False,
        amplitude_replaced=False,indicator_is_error_bound=False)
    f.spectrum.publish(root/'result.json',r);print(json.dumps(r,indent=2),flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('action',choices=('prepare','worker','reduce'))
    a.add_argument('--root',type=Path,required=True);a.add_argument('--source',type=Path)
    a.add_argument('--source-remote');a.add_argument('--parent',type=Path,action='append',default=[])
    a.add_argument('--parent-remote',action='append',default=[]);a.add_argument('--mode',default='followup')
    a.add_argument('--workers',type=int,default=192);a.add_argument('--phase',choices=('prepare','contract'))
    args=a.parse_args()
    if args.action=='prepare':prepare(args)
    elif args.action=='worker':worker(args.root,args.phase)
    else:reduce(args.root)
