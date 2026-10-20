"""Fresh, bounded implementation of the analytic-boundary production proposal.

No source release, imported mesh, amplitude component, or coefficient bank is
accepted. Only the retained analytic estimator is evaluated. Preparation,
contraction, compact sampling, and local cell refinement have separate receipts.
"""

from genus1_source_paths import ROOT as _SOURCE_ROOT, source_path as _source_path
from dataclasses import asdict
import argparse
from collections import Counter
import itertools
import json
import math
from pathlib import Path
import resource
import signal
import sys
import time

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import genus1_production_support as shared
import genus1_boundary_integration as boundary
import genus1_boundary_spectrum as spectrum
from genus1_production_support import template, make_plan, controls
from genus1_analytic_boundary_layer import compact_samples, collision_plan, MIN_HEIGHT
from genus1_boundary_cubature import Value, combine, fingerprint, gauss_rule
from genus1_moduli_sampling import ShortPoint, regular_mask
from genus1_three_point_channel_atlas import make_atlas
from genus1_native_bank import encode_cusp_value as encoded
from genus1_collision_sampling import Request

FORMAT = 'genus1-fresh-analytic-proposal-v1'
ROOT = _SOURCE_ROOT


def clock():
    return time.process_time(), time.monotonic()


def timing(start):
    return dict(cpu_seconds=time.process_time()-start[0], wall_seconds=time.monotonic()-start[1],
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024))


def load(root):
    p=spectrum.read(Path(root)/'plan.json')
    if p['format']!=FORMAT or p['imports']:
        raise ValueError('fresh analytic plan required')
    if make_plan(p['config']).record()!=p['physics']:
        raise ValueError('changed analytic partition')
    for rel,digest in p['source_files'].items():
        if shared.sha(ROOT/rel)!=digest:raise ValueError('changed frozen source: '+rel)
    return p


def fixed_momentum_sampling():
    return dict(total_nodes=[8,9,9],head_orders=[5,6,6],split_scales=1.5,
                tail_width_scales=[.25,.275,.275],placement_policy='moduli-head-tail-v1')


def select_final_cells(initial,roots,config):
    """Choose one bisection per unresolved root from saved initial diagnostics."""
    initial=Path(initial)
    old=spectrum.read(initial/'plan.json');result=spectrum.read(initial/'000_result.json')
    diagnostics=json.loads((initial/'coarse_cell_indicators.json').read_text())
    if (old['config']!=config or result['wave']!=0 or result['plan_sha256']!=fingerprint(old)
            or diagnostics['source_plan_sha256']!=fingerprint(old)
            or diagnostics['source_result_sha256']!=fingerprint(result)):
        raise ValueError('initial diagnostics do not belong to this analytic proposal')
    by_id={r['cell']['id']:r for r in diagnostics['cells']}
    if set(by_id)!={c['id'] for c in roots} or len(by_id)!=len(diagnostics['cells']):
        raise ValueError('initial diagnostic roots differ from the retained domain')
    allocation=controls(config);counts=Counter(c['term'] for c in roots);cells=[];selected=[]
    for term,component in result['components'].items():
        error=math.fsum(r['indicator'] for r in by_id.values() if r['cell']['term']==term)
        if not math.isclose(error,component['quadrature_indicator'],rel_tol=1e-12,abs_tol=1e-18):
            raise ValueError('initial cell indicators differ from the component receipt')
    for cell in roots:
        row=by_id[cell['id']]
        if row['cell']!=cell:raise ValueError('changed initial cell geometry')
        term=cell['term'];tolerance=allocation[term].absolute_tolerance
        if (result['components'][term]['quadrature_indicator']>tolerance
                and row['indicator']>tolerance/counts[term]):
            axis=row['axis']
            if type(axis) is not int or not 0<=axis<len(cell['bounds']):raise ValueError('invalid diagnostic axis')
            cells.extend(bisect(cell,axis));selected.append(dict(parent=cell['id'],term=term,axis=axis))
        else:cells.append(cell)
    return cells,dict(mode='single_final_refinement',maximum_refinement_passes=1,
        source_plan_sha256=fingerprint(old),source_boundary_sha256=fingerprint(result),
        source_diagnostics_sha256=shared.sha(initial/'coarse_cell_indicators.json'),
        selection='bisect each unresolved root once along its largest initial directional indicator',
        selected_roots=selected,imported_cft_values=0)


def initialize(root,*,initial=None,momentum_mode=None):
    root=Path(root)
    if (root/'plan.json').exists():raise FileExistsError('preserve the existing fresh plan')
    # Moduli refinement and momentum allocation are independent choices.
    # Preserve the explicit fixed-pass entry point; permit the audited regional
    # policy on exactly the same once-refined geometry.
    momentum_mode=('fixed_899' if initial is not None else 'regional') if momentum_mode is None else momentum_mode
    if momentum_mode not in ('regional','fixed_899'):
        raise ValueError('momentum mode must be regional or fixed_899')
    config=template();config['compact_seed']=20260923
    raw=json.loads((ROOT/'code/one_loop/three_point/other_scripts/genus1_retained_t025_candidate.json').read_text())
    if momentum_mode=='fixed_899':
        raw['generic_total_node_limit']=9
        raw['profiles']['9']=dict(momentum_orders=[8,9,9],tail_orders=[3,3,3],
            momentum_sampling=fixed_momentum_sampling(),kernels={'primary':dict(
                momentum_sampling=dict(fixed_momentum_sampling(),axis_order='cut_short_short'))})
    settings=spectrum.settings(raw)
    physics=make_plan(config)
    sources={}
    from genus1_source_paths import production_sources
    for path in production_sources():
        if path.is_file() and path.suffix in ('.py','.json','.sbatch'):
            rel=str(path.relative_to(ROOT));sources[rel]=shared.copy(path,root/'runtime/source'/rel)
    sources['PACKAGE_MAP.json']=shared.copy(ROOT/'PACKAGE_MAP.json',root/'runtime/source/PACKAGE_MAP.json')
    p=dict(format=FORMAT,created_utc=shared.utc(),config=config,physics=physics.record(),
        spectrum=settings,elliptic_orders=[2,4],profile=9,momentum_mode=momentum_mode,imports=[],
        source_files=sources,quadrature=dict(low_order=2,high_order=3,adaptive='bisect one coordinate of the largest-error cell',
            controls={k:asdict(v) for k,v in controls(config).items()}),
        compact=dict(seed=config['compact_seed'],scrambles=16,points_per_scramble=4,
            coefficient_orders=[4,4,4],momentum_sampling=dict(total_nodes=8,head_orders=5,
                split_scales=1.5,tail_width_scales=.25)),
        diagnostics=dict(integrated_remainder=False,max_locations=48,max_parents=96,cpu_fraction=.1),
        full_amplitude_certified=False)
    atlas=make_atlas(physics.t)
    cells=[]
    for term in boundary.term_names('analytic'):
        for cell in boundary.term_cells(term,physics,atlas):
            row=dict(term=term,key=cell.key,bounds=[list(b) for b in cell.bounds],depth=0)
            row['id']=fingerprint(row);cells.append(row)
    root_count=len(cells)
    if initial is not None:
        cells,p['refinement']=select_final_cells(initial,cells,config)
        p['quadrature'].update(adaptive=None,directional_probes=False)
    if momentum_mode=='fixed_899':
        p['compact']['momentum_sampling']=fixed_momentum_sampling()
    spectrum.publish(root/'plan.json',p)
    locations=[]
    for replicate in range(16):
        b=compact_samples(2,seed=config['compact_seed'],replicate=replicate,policy=physics.policy)
        for index in range(4):
            locations.append(dict(replicate=replicate,index=index,geometry=dict(
                w1=[b.w1[index].real,b.w1[index].imag],w2=[b.w2[index].real,b.w2[index].imag],
                tau=[b.tau[index].real,b.tau[index].imag],weight=float(b.weight[index]),regular=bool(b.regular[index]))))
    spectrum.publish(root/'compact/locations.json',dict(plan_sha256=fingerprint(p),locations=locations))
    spectrum.publish(root/'waves/000_request.json',dict(plan_sha256=fingerprint(p),wave=0,cells=cells,
        active_cells=[r['id'] for r in cells],previous_wave=None,parents_replaced=[]))
    return dict(plan_sha256=fingerprint(p),root_cells=root_count,staged_cells=len(cells),compact_points=64,imports=[])


def point_id(cell,u):
    # Child cells keep their root namespace; shared coordinates are computed once.
    return fingerprint([cell['key'],[float(v).hex() for v in u]])


def cell_nodes(cell,low=2,high=3,*,directional_probes=True):
    """Embedded integrals and the original proposal's directional probes."""
    bounds=cell['bounds'];d=len(bounds);points={};rules={}
    for order in (low,high):
        nodes,weights=gauss_rule(order);rows=[]
        for index in itertools.product(range(order),repeat=d):
            u=[(a+b)/2+(b-a)/2*nodes[i] for (a,b),i in zip(bounds,index)]
            w=math.prod((b-a)/2*weights[i] for (a,b),i in zip(bounds,index))
            key=point_id(cell,u);points[key]=list(map(float,u));rows.append([key,float(w)])
        rules[str(order)]=rows
    probes=[];middle=[(a+b)/2 for a,b in bounds]
    for axis,(a,b) in enumerate(bounds) if directional_probes else ():
        entry={}
        for order in (low,high):
            nodes,weights=gauss_rule(order);rows=[]
            for node,w in zip(nodes,weights):
                u=middle.copy();u[axis]+=(b-a)/2*node
                key=point_id(cell,u);points[key]=list(map(float,u));rows.append([key,float(w/2)])
            entry[str(order)]=rows
        probes.append(entry)
    return points,dict(cell=cell,rules=rules,probes=probes)


def sample_at(cell,u,physics,atlas):
    term=cell['term'];kind,label=term.split(':')
    if kind=='primary':
        order,band=(int(v[-1]) for v in cell['key'].split(':')[-2:])
        short,jac=boundary.short_map(u,tuple(map(int,label)),order,band,physics.policy)
        if not regular_mask(short,physics.policy.pair_radius,MIN_HEIGHT):
            return dict(kind='zero',jacobian=0.,reason='collision_ownership')
        return dict(kind='primary',short=short.record(),jacobian=jac)
    sample=collision_plan(atlas,label,boundary.collision_unit(u,label,kind),region=kind,
        policy=physics.policy,cutoff=physics.collision_height_split)
    if not sample.requests:
        if sample.exact_zero_reason is None:raise ValueError('unexplained empty collision sample')
        return dict(kind='zero',jacobian=0.,reason=sample.exact_zero_reason)
    return dict(kind='collision',jacobian=sample.jacobian,requests=encoded([asdict(r) for r in sample.requests]))


def request_from_record(row):
    g=dict(row['geometry'])
    for key in ('tau','delta_total','shape','w1','w2'):
        if key in g and g[key] is not None:g[key]=complex(*g[key])
    if g.get('pair') is not None:g['pair']=tuple(g['pair'])
    return Request(row['kind'],g,row['multiplicity'])


def check_momentum_grid(p,spec):
    """Enforce the chosen allocation independently of the moduli mesh."""
    actual=tuple(len(r['nodes']) for r in spec['grid']['rules'])
    if any(n not in (8,9) for n in actual):
        raise ValueError('fresh production must use eight or nine TOTAL nodes')
    cut=spec['grid']['cut_axis']
    mode=p.get('momentum_mode','fixed_899' if p.get('refinement',{}).get('mode')=='single_final_refinement' else 'regional')
    if mode=='fixed_899':
        expected=tuple(8 if i==cut else 9 for i in range(3)) if cut is not None else (8,9,9)
    elif mode=='regional':
        expected=[8,8,8]
        if cut is not None:
            controls=p['spectrum']['profiles'][str(p['profile'])]['kernels']['primary']
            refinement=controls.get('short_refinement')
            short=ShortPoint.from_record(spec['case']['geometry']['short'])
            if refinement is not None:
                for offset,length in ((1,short.s1),(2,short.s2)):
                    if length<refinement['maximum_short_length']:
                        expected[(cut+offset)%3]=refinement['total_nodes']
        expected=tuple(expected)
    else:raise ValueError('unknown momentum allocation mode')
    if actual!=expected:
        raise ValueError('actual momentum grid differs from '+mode+' allocation')
    return actual


def stage(root,p,wave):
    """Enumerate this bounded wave and derive its exact, fresh spectral grids."""
    if p.get('refinement',{}).get('mode')=='single_final_refinement' and wave!=0:
        raise ValueError('only one final refinement pass is authorized')
    root=Path(root);start=clock();path=root/f'waves/{wave:03d}_mesh.json'
    if path.exists():return spectrum.read(path)['summary']
    request=spectrum.read(root/f'waves/{wave:03d}_request.json')
    if request['plan_sha256']!=fingerprint(p):raise ValueError('foreign wave')
    physics=make_plan(p['config']);atlas=make_atlas(physics.t,special_dps=p['spectrum']['special_dps'])
    planner=spectrum.Planner(p,atlas);new={};grids=set();byterm=Counter();integral_ids=set()
    for cell in request['cells']:
        points,rule=cell_nodes(cell,directional_probes=p['quadrature'].get('directional_probes',True))
        spectrum.publish(root/'cells'/f'{cell["id"]}.json',rule)
        for rows in rule['rules'].values():integral_ids.update(k for k,_ in rows)
        for key,u in points.items():
            point_path=root/'points'/key/'sample.json'
            if point_path.exists():continue
            sample=sample_at(cell,u,physics,atlas)
            if sample['kind']=='primary':
                short=ShortPoint.from_record(sample['short'])
                sample['grids']=[planner.register(root,short,True,p['profile'])]
            elif sample['kind']=='collision':
                sample['grids']=[planner.register(root,request_from_record(r),False,p['profile']) for r in sample['requests']]
            else:sample['grids']=[]
            sample.update(point_id=key,term=cell['term'],namespace=cell['key'],unit=u,plan_sha256=fingerprint(p))
            spectrum.publish(point_path,sample);new[key]=sample;byterm[cell['term']]+=1;grids.update(sample['grids'])
    # Restart after a partial stage: reconstruct the whole wave from its cell pins.
    wave_points={}
    for cell in request['cells']:
        points,_=cell_nodes(cell,directional_probes=p['quadrature'].get('directional_probes',True))
        for key in points:wave_points[key]=spectrum.read(root/'points'/key/'sample.json')
    grids={g for r in wave_points.values() for g in r['grids']}
    bank_keys=sorted(g for g in grids if not (spectrum.grid_directory(root,g)/'ready.json').exists())
    todo=[k for k in wave_points if not (root/'points'/k/'value.json').exists()]
    batches=[todo[i:i+24] for i in range(0,len(todo),24)]
    counts=Counter(r['term'] for r in wave_points.values());axis_counts=Counter();families=Counter()
    for key in grids:
        s=spectrum.read(spectrum.grid_directory(root,key)/'spec.json');nn=check_momentum_grid(p,s)
        axis_counts[str(nn)]+=1;families[s['case']['kind']]+=1
    summary=dict(wave=wave,cells=len(request['cells']),integral_nodes=len(integral_ids),
        moduli_points=len(wave_points),split_probe_nodes=len(wave_points)-len(integral_ids),
        zero_points=sum(r['kind']=='zero' for r in wave_points.values()),fresh_banks=len(bank_keys),
        banks_by_family=dict(families),banks_by_axis_counts=dict(axis_counts),points_by_term=dict(counts),
        contraction_batches=len(batches),preparation=timing(start),imports=[])
    spectrum.publish(path,dict(plan_sha256=fingerprint(p),wave=wave,bank_keys=bank_keys,batches=batches,
        points=list(wave_points),summary=summary))
    shared.save(root/f'timings/mesh_{wave:03d}.json',summary['preparation'])
    return summary


def bank_work(root,p,wave,index):
    mesh=spectrum.read(Path(root)/f'waves/{wave:03d}_mesh.json');key=mesh['bank_keys'][index];start=clock()
    attempts=[]
    while time.monotonic()-start[1]<1550:
        r=spectrum.build(root,p,key,cpu_budget=240,max_new_chunks=64,max_new_elliptic_nodes=1024)
        attempts.append(r)
        if r['complete_rule']:break
        if (r.get('failure') or {}).get('type')!='TimeLimit':break
    r=dict(plan_sha256=fingerprint(p),wave=wave,index=index,grid=key,complete=bool(attempts[-1]['complete_rule']),
        attempts=len(attempts),failure=attempts[-1]['failure'],**timing(start))
    shared.save(Path(root)/f'timings/bank_{wave:03d}_{index:05d}.json',r)
    return r


def contract(root,p,wave,index):
    from genus1_boundary_batch import GeometryStore
    root=Path(root);start=clock();physics=make_plan(p['config']);atlas=make_atlas(physics.t)
    mesh=spectrum.read(root/f'waves/{wave:03d}_mesh.json');keys=mesh['batches'][index]
    geometry=GeometryStore(atlas,root/f'work/geometry/{wave:03d}_{index:05d}')
    runtime=spectrum.Runtime(root,p,atlas,geometry,point_budget=1e-10)
    new=0
    try:
        for key in keys:
            path=root/'points'/key/'value.json'
            if path.exists():continue
            r=spectrum.read(root/'points'/key/'sample.json');jac=r['jacobian']
            if r['kind']=='zero':v=Value(0.,dict(block=0.,arithmetic=0.,momentum=None,endpoint=None))
            elif r['kind']=='primary':
                short=ShortPoint.from_record(r['short']);prepared=geometry.prepare([short])
                v=boundary.amplitude_value(runtime.primary([short],r['grids'],prepared,physics.policy,physics.angle_order)[0],jac)
            else:
                values=[runtime.collide(request_from_record(a),g,jac).scaled(a['multiplicity'])
                        for a,g in zip(r['requests'],r['grids'])]
                v=boundary.amplitude_value(combine(values),jac)
            spectrum.publish(path,dict(plan_sha256=fingerprint(p),sample_sha256=shared.sha(root/'points'/key/'sample.json'),value=v.record()))
            new+=1
    finally:geometry.close()
    result=dict(complete=True,plan_sha256=fingerprint(p),wave=wave,index=index,new_points=new,**timing(start))
    shared.save(root/f'timings/contract_{wave:03d}_{index:05d}.json',result)
    return result


def compact_work(root,p,index):
    """New full-CFT banks with the plan's explicit total-node budget."""
    import genus1_projection_preparation as prep
    from genus1_channel_evaluation import full_values
    from genus1_native_bank import variants
    root=Path(root);folder=root/f'compact/points/{index:02d}';start=clock();path=folder/'value.json'
    if path.exists():return dict(complete=True,cached=True,index=index)
    row=spectrum.read(root/'compact/locations.json')['locations'][index];g=row['geometry'];c=p['compact']
    prep_start=clock();nodes=0
    if not g['regular']:
        v=Value(0.,dict(block=0.,arithmetic=0.,momentum=None,endpoint=None))
        prep_time=timing(prep_start);eval_start=clock()
    else:
        atlas=make_atlas(p['physics']['t']);w1,w2,tau=(complex(*g[k]) for k in ('w1','w2','tau'))
        choice=atlas.choose_channel(w1,w2,tau)
        if not atlas.sewing_chart(w1,w2,tau,choice.channel,choice.pair).admissible:
            raise ValueError('fresh compact point needs a valid chart')
        case=dict(id='bank',kind='full',geometry={k:g[k] for k in ('w1','w2','tau')},coefficient_orders=c['coefficient_orders'])
        case['geometry'].update(channel=choice.channel,pair=None if choice.pair is None else list(choice.pair))
        target=folder/'input'
        from genus1_projection_momentum import total_sampling_controls, check_total_nodes
        budget=total_sampling_controls(c['momentum_sampling'])
        if not (target/'plan.json').exists():
            prep.prepare(target,t=p['physics']['t'],cases=[case],momentum_orders=budget['total_nodes'],
                tail_orders=[n-h for n,h in zip(budget['total_nodes'],budget['head_orders'])],
                momentum_sampling=c['momentum_sampling'],chunk_size=32)
        plan=prep.read_json(target/'plan.json')
        check_total_nodes(plan['cases']['bank']['rules'],budget)
        from genus1_native_bank import chunk_nodes
        missing=[i for i in range(len(chunk_nodes(plan))) if not (target/'banks'/f'{i:07d}').exists()]
        if missing:prep.build(target,start=missing[0],stop=missing[-1]+1)
        bank=prep.load_bank(target,'bank');prep_time=timing(prep_start);eval_start=clock()
        values,report=full_values(bank,atlas,prep.geometry(case),bank.orders)
        if not report['passed']:raise FloatingPointError('fresh compact arithmetic unresolved')
        e=[r['absolute_arithmetic_log_bound'] for r in report['prefixes']]
        native=Value(values[0],dict(block=float(max(abs(values[0]-values[1:]))),arithmetic=math.exp(max(e)),momentum=None,endpoint=None),
            {'native:'+name:values[i] for i,(name,_) in enumerate(variants(bank.orders)) if i})
        v=boundary.amplitude_value(native,g['weight']);nodes=len(bank.arrays['momenta'])
    r=dict(plan_sha256=fingerprint(p),location=row,value=v.record(),momentum_tuples=nodes,
        preparation=prep_time,contraction=timing(eval_start),**timing(start))
    spectrum.publish(path,r);shared.save(root/f'timings/compact_{index:02d}.json',r)
    return dict(complete=True,index=index,**{k:r[k] for k in ('cpu_seconds','wall_seconds','momentum_tuples')})


def estimate_cell(root,p,cell_id):
    root=Path(root);rule=spectrum.read(root/'cells'/f'{cell_id}.json');values={}
    def sum_rule(rows):
        for key,_ in rows:
            if key not in values:
                record=spectrum.read(root/'points'/key/'value.json')
                if record['plan_sha256']!=fingerprint(p) or record['sample_sha256']!=shared.sha(root/'points'/key/'sample.json'):
                    raise ValueError('foreign point value')
                values[key]=Value.from_record(record['value'])
        return combine((values[k] for k,_ in rows),(w for _,w in rows))
    low,high=(sum_rule(rule['rules'][str(n)]) for n in (2,3))
    scores=[abs(sum_rule(a['3']).value-sum_rule(a['2']).value) for a in rule['probes']]
    axis=max(range(len(scores)),key=lambda i:scores[i]) if scores else None
    return dict(cell=rule['cell'],value=high.record(),quadrature_indicator=abs(high.value-low.value),
        coarse_value=low.record(),axis=axis,directional_indicators=scores)


def bisect(cell,axis):
    a,b=cell['bounds'][axis];mid=(a+b)/2;rows=[]
    for interval in ([a,mid],[mid,b]):
        row={k:v for k,v in cell.items() if k!='id'};row['bounds']=[list(x) for x in cell['bounds']]
        row['bounds'][axis]=interval;row['depth']=cell['depth']+1;row['id']=fingerprint(row);rows.append(row)
    return rows


def reduce_wave(root,p,wave):
    root=Path(root);start=clock();request=spectrum.read(root/f'waves/{wave:03d}_request.json')
    estimates=[estimate_cell(root,p,key) for key in request['active_cells']]
    counts=Counter();existing=set()
    for path in (root/'points').glob('*/sample.json'):
        sample=spectrum.read(path);counts[sample['term']]+=1;existing.add(sample['point_id'])
    terms={};children=[];replaced=[]
    final_pass=p.get('refinement',{}).get('mode')=='single_final_refinement'
    if final_pass and wave!=0:raise ValueError('only one final refinement pass is authorized')
    for term in boundary.term_names('analytic'):
        rows=[r for r in estimates if r['cell']['term']==term];value=combine(Value.from_record(r['value']) for r in rows)
        error=math.fsum(r['quadrature_indicator'] for r in rows);c=p['quadrature']['controls'][term]
        roots=4 if term.startswith('primary:') else len(boundary.collision_boxes(term.split(':')[1],term.split(':')[0]))
        enough_splits=final_pass or len(rows)-roots>=c['minimum_splits']
        status='quadrature_indicator_met' if enough_splits and error<=c['absolute_tolerance'] else 'work_limit'
        if final_pass and error>c['absolute_tolerance']:status='single_refinement_limit'
        if not final_pass and (error>c['absolute_tolerance'] or not enough_splits):
            for candidate in sorted(rows,key=lambda r:r['quadrature_indicator'],reverse=True):
                if candidate['cell']['depth']>=c['max_depth']:continue
                proposed=bisect(candidate['cell'],candidate['axis'])
                nodes=set().union(*(set(cell_nodes(r)[0]) for r in proposed));new=len(nodes-existing)
                if counts[term]+new<=c['max_evaluations']:
                    children.extend(proposed);replaced.append(candidate['cell']['id']);status='local_refinement_planned';break
        terms[term]=dict(value=value.record(),quadrature_indicator=error,tolerance=c['absolute_tolerance'],
            status=status,moduli_points=counts[term],cell_count=len(rows))
    next_wave=None
    if children:
        next_wave=wave+1
        active=[key for key in request['active_cells'] if key not in replaced]+[r['id'] for r in children]
        spectrum.publish(root/f'waves/{next_wave:03d}_request.json',dict(plan_sha256=fingerprint(p),wave=next_wave,
            cells=children,active_cells=active,previous_wave=wave,parents_replaced=replaced))
    total=combine(Value.from_record(r['value']) for r in terms.values())
    result=dict(plan_sha256=fingerprint(p),wave=wave,complete_domain=True,components=terms,
        boundary=total.record(),quadrature_indicator=math.fsum(r['quadrature_indicator'] for r in terms.values()),
        next_wave=next_wave,full_amplitude_certified=False,**timing(start))
    if final_pass:result['stopping_reason']='one_final_refinement_completed_no_further_passes'
    spectrum.publish(root/f'waves/{wave:03d}_result.json',result)
    shared.save(root/f'timings/reduce_{wave:03d}.json',timing(start))
    return result


def assemble(root,p,wave):
    root=Path(root);r=spectrum.read(root/f'waves/{wave:03d}_result.json');data=[];rows=[]
    if r['plan_sha256']!=fingerprint(p) or r['next_wave'] is not None:
        raise ValueError('final assembly requires this plan and terminal bounded refinement')
    request=spectrum.read(root/f'waves/{wave:03d}_request.json')
    leaves=[estimate_cell(root,p,key) for key in request['active_cells']]
    for term,component in r['components'].items():
        values=[Value.from_record(row['value']) for row in leaves if row['cell']['term']==term]
        if combine(values).record()!=component['value']:
            raise ValueError('boundary receipt differs from saved values')
        if component['moduli_points']>p['quadrature']['controls'][term]['max_evaluations']:
            raise ValueError('component work allowance exceeded')
    import numpy as np
    observations=np.full((16,4),np.nan)
    locations=spectrum.read(root/'compact/locations.json')['locations']
    for i,location in enumerate(locations):
        row=spectrum.read(root/f'compact/points/{i:02d}/value.json')
        if row['plan_sha256']!=fingerprint(p) or row['location']!=location:raise ValueError('foreign compact observation')
        v=Value.from_record(row['value']);observations[location['replicate'],location['index']]=v.value.real;rows.append(v)
    compact=boundary.compact_estimate(observations,make_plan(p['config']),density_identity=dict(fresh_plan_sha256=fingerprint(p)))
    compact['errors']=combine(rows,[1/64]*64).errors
    for path in (root/'timings').glob('*.json'):
        row=shared.checked(path);data.append(dict(stage=path.name.split('_')[0],cpu_seconds=row['cpu_seconds'],wall_seconds=row['wall_seconds']))
    stages={}
    for row in data:
        dest=stages.setdefault(row['stage'],dict(cpu_seconds=0.,worker_wall_seconds=0.,tasks=0))
        dest['cpu_seconds']+=row['cpu_seconds'];dest['worker_wall_seconds']+=row['wall_seconds'];dest['tasks']+=1
    result=dict(format=FORMAT,plan_sha256=fingerprint(p),wave=wave,complete_domain=True,
        amplitude=compact['mean']+r['boundary']['value'][0],compact=compact,boundary=r,
        imports=[],integrated_remainder=False,full_amplitude_certified=False,
        unknown_errors=['integrated omitted analytic remainders, including full triple OPE',
                        'global cutoff stability','integrated momentum quadrature error'],
        timing_by_stage=stages,process_cpu_seconds=sum(a['cpu_seconds'] for a in stages.values()),
        timing_definition='Useful process CPU and summed worker wall time; scheduler elapsed is separate.',created_utc=shared.utc())
    spectrum.publish(root/f'amplitude_wave_{wave:03d}.json',result)
    return result


def main():
    def interrupted(signum,frame):
        # Unwind coefficient-bank ownership before a scheduler requeue.
        # SystemExit must escape the bounded numerical retry loop.
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,interrupted)
    if hasattr(signal,'SIGUSR1'):signal.signal(signal.SIGUSR1,interrupted)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('init','stage','bank','contract','compact','reduce','assemble','verify'))
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--wave',type=int,default=0)
    parser.add_argument('--index',type=int)
    parser.add_argument('--initial',type=Path,help='Saved initial diagnostics for exactly one final refinement')
    parser.add_argument('--momentum-mode',choices=('regional','fixed_899'),
                        help='Choose momentum allocation independently of the moduli refinement')
    a=parser.parse_args()
    if a.stage=='init':result=initialize(a.root,initial=a.initial,momentum_mode=a.momentum_mode)
    else:
        p=load(a.root)
        if a.stage=='stage':result=stage(a.root,p,a.wave)
        elif a.stage=='bank':result=bank_work(a.root,p,a.wave,a.index)
        elif a.stage=='contract':result=contract(a.root,p,a.wave,a.index)
        elif a.stage=='compact':result=compact_work(a.root,p,a.index)
        elif a.stage=='reduce':result=reduce_wave(a.root,p,a.wave)
        elif a.stage=='assemble':result=assemble(a.root,p,a.wave)
        else:result=dict(verified=True,plan_sha256=fingerprint(p))
    print(json.dumps(result,indent=2,allow_nan=False))
    return 2 if result.get('complete') is False else 0


if __name__=='__main__':raise SystemExit(main())
