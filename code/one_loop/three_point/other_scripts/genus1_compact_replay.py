"""Replay the existing compact observations; never draw new MC points."""
from __future__ import annotations

# genus1_fixed_compact_replay
# Bounded replay of the frozen 64 compact observations on corrected grids.
# 
# The saved coordinates, masks, quadrature weights and replicate labels are the
# sampling design. There is no random generator here. Preparation and contraction
# are separate, restartable operations; a partial profile is never an estimate.


from genus1_source_paths import ROOT as _SOURCE_ROOT, source_path as _source_path
from pathlib import Path
import json
import math
import resource
import sys
import time

import numpy as np

from genus1_boundary_cubature import Value,combine,fingerprint,MissingBoundaryInput
from genus1_native_bank import source_manifest, check_sources, chunk_nodes
from genus1_native_bank import digest, variants
from genus1_three_point_channel_atlas import make_atlas
from genus1_production_support import make_plan
from genus1_boundary_integration import compact_estimate,amplitude_value
import genus1_projection_preparation as preparation

FORMAT='genus1-fixed-compact-replay-v1'


def settings(value):
    c=json.loads(json.dumps(value))
    if set(c)!={'profiles','production_profile','coefficient_orders','absolute_tolerance'}:
        raise ValueError('explicit compact profiles, production label, native orders and tolerance required')
    if len(c['profiles'])<2 or c['production_profile'] not in c['profiles']:
        raise ValueError('at least two compact momentum profiles required')
    for label,row in c['profiles'].items():
        if not label.isalnum() or set(row)-{'momentum_orders','tail_orders','width_scales'}:
            raise ValueError('invalid compact profile')
        for key in ('momentum_orders','tail_orders'):
            if len(row.get(key,[]))!=3 or any(type(n) is not int or n<1 for n in row[key]):
                raise ValueError('three positive compact core/tail orders required')
        row.setdefault('width_scales',[1.,1.,1.])
        if len(row['width_scales'])!=3 or any(isinstance(x,bool) or not math.isfinite(x) or x<=0 for x in row['width_scales']):
            raise ValueError('positive finite compact momentum widths required')
    if len({fingerprint(row) for row in c['profiles'].values()})!=len(c['profiles']):
        raise ValueError('compact comparison profiles must differ')
    if len(c['coefficient_orders'])!=3 or any(type(n) is not int or not 2<=n<=10 for n in c['coefficient_orders']):
        raise ValueError('three native compact orders with two retained prefixes required')
    if not math.isfinite(c['absolute_tolerance']) or c['absolute_tolerance']<=0:
        raise ValueError('positive compact comparison tolerance required')
    return c


def initialize(root,source,config,physics):
    from genus1_production_support import checked
    from genus1_boundary_spectrum import publish
    root=Path(root);source=Path(source);raw=checked(source);config=settings(config)
    locations=raw['locations']
    if (not raw['complete'] or raw['points']!=64 or len(locations)!=64 or
            raw['estimate']['plan']!=physics.record() or
            {(r['replicate'],r['index']) for r in locations}!={(r,i) for r in range(16) for i in range(4)} or
            any(r['status']!='complete' or not math.isfinite(r['geometry']['weight']) or
                r['geometry']['weight']<=0 or type(r['geometry']['regular']) is not bool for r in locations)):
        raise ValueError('complete original 16x4 compact geometry and observations required')
    sources=source_manifest()
    command=_source_path('code/one_loop/three_point/other_scripts/genus1_compact_replay.py')
    sources['code/one_loop/three_point/other_scripts/genus1_compact_replay.py']=digest(command)
    plan=dict(format=FORMAT,config=config,physics=physics.record(),source_compact_sha256=digest(source),
        locations=locations,source_files=sources,new_mc_points=0)
    publish(root/'plan.json',plan)
    return plan


def load(root):
    from genus1_boundary_spectrum import read
    p=read(Path(root)/'plan.json')
    if p['format']!=FORMAT or settings(p['config'])!=p['config']:raise ValueError('invalid compact replay plan')
    check_sources(p)
    return p


def point_spec(p,label,index):
    if label not in p['config']['profiles'] or type(index) is not int or not 0<=index<64:
        raise ValueError('frozen compact profile and point index required')
    row=p['locations'][index];g=row['geometry'];atlas=make_atlas(p['physics']['t'])
    xyz=[complex(*g[k]) for k in ('w1','w2','tau')]
    choice=atlas.choose_channel(*xyz)
    if not atlas.sewing_chart(*xyz,choice.channel,choice.pair).admissible:
        raise ValueError('frozen compact point has no admissible chosen chart')
    case=dict(id='bank',kind='full',geometry={k:g[k] for k in ('w1','w2','tau')},
              coefficient_orders=p['config']['coefficient_orders'])
    case['geometry'].update(channel=choice.channel,pair=None if choice.pair is None else list(choice.pair))
    return row,atlas,case


def work(root,p,label,index,*,cpu_budget=600.,max_new_chunks=16):
    from genus1_production_support import checked, save
    from genus1_production_support import ownership, cpu_deadline
    from genus1_boundary_spectrum import publish
    from genus1_channel_evaluation import full_values
    from genus1_boundary_cubature import TimeLimit
    root=Path(root);folder=root/'points'/label/f'{index:02d}';result_path=folder/'result.json'
    identity=dict(plan_sha256=fingerprint(p),profile=label,index=index)
    if result_path.exists():
        result=checked(result_path)
        if result['identity']!=identity:raise ValueError('foreign compact result')
        return dict(complete=True,cached=True,result=result)
    if not math.isfinite(cpu_budget) or cpu_budget<=0 or type(max_new_chunks) is not int or max_new_chunks<1:
        raise ValueError('positive compact CPU and chunk budgets required')
    cpu=time.process_time();wall=time.monotonic();new=0;failure=None;result=None
    with ownership(root/'claims'/label/f'{index:02d}'):
        try:
            with cpu_deadline(cpu_budget):
                row=p['locations'][index]
                if not row['geometry']['regular']:
                    value=Value(0.,dict(block=0.,arithmetic=0.,momentum=None,endpoint=None),
                        {'native:'+name:0. for name,_ in variants(p['config']['coefficient_orders']) if name!='full'})
                    nodes=0
                else:
                    row,atlas,case=point_spec(p,label,index);target=folder/'input'
                    if not (target/'plan.json').exists():
                        preparation.prepare(target,t=p['physics']['t'],cases=[case],
                            **p['config']['profiles'][label],chunk_size=8)
                    subplan=preparation.read_json(target/'plan.json')
                    if subplan['cases']['bank']['case']!=case:raise ValueError('changed compact point preparation')
                    expected=preparation.spec(atlas,case,**{
                        'orders':p['config']['profiles'][label]['momentum_orders'],
                        'tails':p['config']['profiles'][label]['tail_orders'],
                        'width_scales':p['config']['profiles'][label]['width_scales']})[2]
                    if subplan['cases']['bank']['rules']!=expected:raise ValueError('changed compact momentum grid')
                    missing=[i for i in range(len(chunk_nodes(subplan))) if not (target/'banks'/f'{i:07d}').exists()]
                    selected=missing[:max_new_chunks]
                    def progress(event):
                        nonlocal new
                        if event['stage']=='committed_chunk':new+=1
                    if selected:preparation.build(target,start=selected[0],stop=selected[-1]+1,progress=progress)
                    if len(missing)>max_new_chunks:raise TimeLimit('compact preparation chunk allocation reached')
                    bank=preparation.load_bank(target,'bank');g=preparation.geometry(case)
                    values,report=full_values(bank,atlas,g,bank.orders)
                    if not report['passed']:raise FloatingPointError('compact full-CFT arithmetic unresolved')
                    errors=[r['absolute_arithmetic_log_bound'] for r in report['prefixes']]
                    arithmetic=math.exp(max(errors))
                    native=Value(values[0],dict(block=max(abs(values[0]-values[1:])),arithmetic=arithmetic,
                        momentum=None,endpoint=None),{'native:'+name:values[i] for i,(name,_) in enumerate(variants(bank.orders)) if i})
                    value=amplitude_value(native,row['geometry']['weight']);nodes=len(bank.arrays['momenta'])
                result=dict(identity=identity,replicate=row['replicate'],index=row['index'],geometry=row['geometry'],
                    value=value.record(),momentum_nodes=nodes,new_mc_points=0)
                save(result_path,result)
        except Exception as exc:failure=dict(type=type(exc).__name__,message=str(exc))
        attempt=dict(identity=identity,complete=result is not None,new_chunks=new,failure=failure,
            cpu_seconds=time.process_time()-cpu,wall_seconds=time.monotonic()-wall)
        import uuid
        save(root/'attempts'/(str(uuid.uuid4())+'.json'),attempt)
    return attempt


def reduce(root,p,physics,*,publish_result=True):
    from genus1_production_support import checked, save
    if physics.record()!=p['physics']:raise ValueError('compact reduction physics changed')
    root=Path(root);profiles={};missing=[];values={}
    old=np.zeros((16,4))
    for row in p['locations']:old[row['replicate'],row['index']]=Value.from_record(row['value']).value.real
    for label in p['config']['profiles']:
        rows=[];data=np.full((16,4),np.nan)
        for i,original in enumerate(p['locations']):
            path=root/'points'/label/f'{i:02d}'/'result.json'
            if not path.exists():missing.append([label,i]);continue
            r=checked(path)
            if (r['identity']!=dict(plan_sha256=fingerprint(p),profile=label,index=i) or
                    r['geometry']!=original['geometry'] or (r['replicate'],r['index'])!=(original['replicate'],original['index'])):
                raise ValueError('compact replay changed the frozen coordinates, masks, labels or weights')
            v=Value.from_record(r['value']);rows.append(v);data[r['replicate'],r['index']]=v.value.real
        if len(rows)!=64:continue
        estimate=compact_estimate(data,physics,density_identity=dict(replay_plan_sha256=fingerprint(p),profile=label))
        aggregate=combine(rows,[1/64]*64)
        estimate['errors']=aggregate.errors
        old_delta=compact_estimate(data-old,physics,density_identity={'paired_with':'original_64'})
        profiles[label]=dict(estimate=estimate,paired_old_change=old_delta['mean'],
            paired_old_halfwidth95=old_delta['sampling_halfwidth95'],aggregate=aggregate.record())
        values[label]=data
    comparisons={};production=p['config']['production_profile'];tol=p['config']['absolute_tolerance']
    if production in values:
        for label,data in values.items():
            if label==production:continue
            delta=compact_estimate(values[production]-data,physics,density_identity={'paired_profiles':[label,production]})
            indicator=abs(delta['mean'])+delta['sampling_halfwidth95']
            comparisons[label]=dict(mean_change=delta['mean'],paired_halfwidth95=delta['sampling_halfwidth95'],
                indicator=indicator,passed=indicator<=tol)
    all_complete=not missing
    block=None if production not in profiles else profiles[production]['estimate']['errors']['block']
    result=dict(format=FORMAT,plan_sha256=fingerprint(p),source_compact_sha256=p['source_compact_sha256'],
        profiles=profiles,production_profile=production,missing=missing,complete=all_complete,
        momentum_comparisons=comparisons,absolute_tolerance=tol,
        momentum_indicator_met=all_complete and all(v['passed'] for v in comparisons.values()),
        block_indicator=block,block_indicator_met=block is not None and block<=tol,
        new_mc_points=0,rigorous_error_bound=False,independent_validation=False)
    if publish_result:save(root/'result.json',result)
    return result


def available(release,physics):
    """Use only a complete replay pinned to this release's original 64 points."""
    from genus1_production_support import checked
    root=Path(release)/'compact_replay';path=root/'result.json'
    if not path.exists():return None,None
    p=load(root);r=checked(path)
    if (p['source_compact_sha256']!=digest(Path(release)/'inputs/compact.json') or
            p['physics']!=physics.record() or r['plan_sha256']!=fingerprint(p) or
            r['source_compact_sha256']!=p['source_compact_sha256']):
        raise ValueError('compact replay belongs to a different frozen sample or physics')
    # Re-read the point receipts; a cached summary alone is not completion proof.
    fresh=reduce(root,p,physics,publish_result=False)
    if fresh!=r:raise ValueError('compact replay summary differs from the complete point receipts; reduce again')
    estimate=r['profiles'][r['production_profile']]['estimate'] if r['complete'] else None
    return estimate,r


def worker(root,p,index,*,cpu_budget=45.,wall_budget=3300.,max_attempts=100,max_new_chunks=16):
    """One bounded array slot; retry only a committed, incomplete allocation.

    A slot is (momentum profile, existing observation). This function neither
    submits jobs nor advances scientific orders. An arithmetic or provenance
    failure ends the slot, and unfinished work returns a non-success status.
    """
    import signal
    import uuid
    from genus1_production_support import save
    labels=sorted(p['config']['profiles'])
    if type(index) is not int or not 0<=index<64*len(labels):raise ValueError('compact array slot outside frozen plan')
    if (not math.isfinite(wall_budget) or wall_budget<=0 or not math.isfinite(cpu_budget) or cpu_budget<=0
            or type(max_attempts) is not int or max_attempts<1):raise ValueError('positive bounded worker allocation required')
    label=labels[index//64];point=index%64;start=time.monotonic();cpu=time.process_time();stop=[];attempts=[]
    previous=signal.signal(signal.SIGUSR1,lambda *_:stop.append(True))
    try:
        while len(attempts)<max_attempts and not stop:
            remaining=wall_budget-(time.monotonic()-start)
            if remaining<=1:break
            result=work(root,p,label,point,cpu_budget=min(cpu_budget,remaining),max_new_chunks=max_new_chunks)
            attempts.append(result)
            if result['complete'] or result.get('failure',{}).get('type')!='TimeLimit':break
    finally:signal.signal(signal.SIGUSR1,previous)
    complete=bool(attempts and attempts[-1]['complete'])
    result=dict(plan_sha256=fingerprint(p),array_index=index,profile=label,index=point,
        complete=complete,attempts=len(attempts),last_attempt=attempts[-1] if attempts else None,
        stop_requested=bool(stop),wall_seconds=time.monotonic()-start,process_cpu_seconds=time.process_time()-cpu,
        process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
        new_mc_points=0)
    save(Path(root)/'workers'/(str(uuid.uuid4())+'.json'),result)
    return result


# genus1_compact_replay
# Replay the existing compact observations; never draw new MC points.

import argparse
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))
import genus1_production_support as shared
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('init','work','worker','reduce'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--source-release',type=Path)
    p.add_argument('--config',type=Path);p.add_argument('--profile');p.add_argument('--index',type=int)
    p.add_argument('--cpu-budget',type=float,default=600.);p.add_argument('--max-new-chunks',type=int,default=16)
    p.add_argument('--wall-budget',type=float,default=3300.);p.add_argument('--max-attempts',type=int,default=100)
    a=p.parse_args()
    if a.stage=='init':
        if a.source_release is None or a.config is None:p.error('init needs source-release and explicit config')
        physics=shared.make_plan(shared.checked(a.source_release/'plan.json')['config'])
        plan=initialize(a.output,a.source_release/'inputs/compact.json',json.loads(a.config.read_text()),physics)
        result=dict(plan_sha256=shared.fingerprint(plan),points=64,new_mc_points=0)
    else:
        plan=load(a.output)
        # Use the same public physics constructor as the source release.
        from genus1_production_support import template, make_plan
        config=template();config['run']['t']=plan['physics']['t']
        config['run'].update({k:plan['physics']['policy'][k] for k in ('q_max','pair_radius','triple_radius')})
        config['run'].update({k:plan['physics'][k] for k in ('collision_height_split','angle_order')})
        physics=make_plan(config)
        if physics.record()!=plan['physics']:raise ValueError('compact replay physics reconstruction changed')
        if a.stage=='worker':
            if a.index is None:p.error('worker requires a global frozen array slot')
            result=worker(a.output,plan,a.index,cpu_budget=a.cpu_budget,wall_budget=a.wall_budget,
                                  max_attempts=a.max_attempts,max_new_chunks=a.max_new_chunks)
        elif a.stage=='work':
            if a.profile is None or a.index is None:p.error('work needs profile and frozen point index')
            result=work(a.output,plan,a.profile,a.index,cpu_budget=a.cpu_budget,max_new_chunks=a.max_new_chunks)
        else:result=reduce(a.output,plan,physics)
    print(json.dumps(result,indent=2,allow_nan=False))
    return 2 if a.stage in ('work','worker') and not result.get('complete',False) else 0


if __name__=='__main__':raise SystemExit(main())
