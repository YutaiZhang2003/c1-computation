"""Bounded, direct quadrature stability checks on a fixed retained amplitude.

No fitted control, new stochastic estimator, momentum refinement, or automatic
change to the reference amplitude. Geometry is frozen before CFT evaluation.
"""

from genus1_source_paths import ROOT as _SOURCE_ROOT, source_path as _source_path
import argparse
from collections import Counter, defaultdict
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import sys
sys.dont_write_bytecode=True

SOURCE_PLAN='4fe44ca0aeb6854200489322cce6d58068a9c7d7a53cd11b374989736a8bc675'
REFERENCE_SHA='15a21cc7570dd39b488b05131b761caca947a0263d4e66ea36cc3277ac2360b7'
FIRST_SHA='5d6edd15acb4de531e72978b4e53b0264aecde3ead9f446a02a9e500952b2bfd'
BROAD_SHA='ae64b8b20568eef35e3a2eaa7f7bf1b67f8ee32c71e0e8addaa57da1979f79b3'
CHECK=(0.,.2,.45,.7,1.)
PRIMARY_RULES={'base':(1,1),'head_h2':(2,2),'head_h4':(4,4),
               'head_check':(CHECK,CHECK),'outer_h2':(2,1),'inner_h2':(1,2)}
TAIL_RULES=['base',*[f'axis{i}_{s}' for i in range(4) for s in ('h2','h4','check')],'mixed_h2']

def configure(root):
    global f,g,production,campaign,locations,p
    locations=json.loads((root/'locations.json').read_text())
    remote=str(root).startswith(locations['remote_diagnostics'])
    runtime=Path(locations['runtime']['remote' if remote else 'local'])
    sys.path.insert(0,str(runtime/'runtime/source/code/tasks'))
    sys.path.insert(0,str(runtime/'runtime/source/code/one_loop/three_point/other_scripts'))
    import genus1_fresh_analytic as f
    p=f.load(runtime)
    if f.fingerprint(p)!=SOURCE_PLAN:raise ValueError('changed frozen CFT runtime')
    import genus1_moduli_amplitude as production
    campaign=production.campaign;campaign.f=f
    sources=[Path(locations['remote_diagnostics'] if remote else locations['local_diagnostics'])/x
             for x in locations['bank_sources']]
    campaign.source,campaign.parents=sources[0],sources[1:]
    spec=importlib.util.spec_from_file_location('stability_geometry',root/'pilot/genus1_moduli_panels.py')
    g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
    return p

def source_interval(bounds,parts=1):
    lo,hi=bounds;xs,ws=f.gauss_rule(3);edges=g.reference_edges(parts)
    for left,right in zip(edges[:-1],edges[1:]):
        a,b=lo+(hi-lo)*float(left),lo+(hi-lo)*float(right)
        for x,w in zip(xs,ws):yield float((a+b)/2+(b-a)/2*x),float((b-a)/2*w)

def primary_observations(cell,parts,physics):
    """Original band-zero length rule; only designated angular heads change."""
    sector=tuple(map(int,cell['term'].split(':')[1]))
    ordering,band=(int(v[-1]) for v in cell['key'].split(':')[-2:])
    if band!=0:raise ValueError('this bounded test only covers short maximum length')
    for (u0,w0),(u1,w1) in itertools.product(*(list(source_interval(b)) for b in cell['bounds'][:2])):
        short,jac=f.boundary.short_map([u0,u1,.5,.5],sector,ordering,band,physics.policy)
        if min(short.s1,short.s2)>2*physics.policy.pair_radius:continue
        lw=jac/(2*math.pi)**2*math.prod((w0,w1))
        for x,y,aw in g.angular_head_rule(short.s1,short.s2,physics.policy.pair_radius,3,tail_order=2,
                x1_bounds=tuple(g.TAU*u for u in cell['bounds'][2]),
                x2_bounds=tuple(g.TAU*u for u in cell['bounds'][3]),
                near_head_subdivisions=parts,near_head_distance=2*physics.policy.pair_radius):
            point=f.ShortPoint(sector,short.s1,short.s2,x,y)
            if not f.regular_mask(point,physics.policy.pair_radius,f.MIN_HEIGHT):
                raise ValueError('changed physical ownership')
            yield dict(kind='primary',short=point.record()),float(lw*aw),'near_head_length_nodes'

def sequence(q0,q2,q4,check):
    """A conditional geometric-increment estimate, explicitly layout-checked."""
    if not all(math.isfinite(v) for v in (q0,q2,q4,check)):raise ValueError('nonfinite quadrature')
    floor=max(1e-18,64*sys.float_info.epsilon*max(map(abs,(q0,q2,q4,check))))
    d1,d2=q2-q0,q4-q2;layout=abs(check-q4)
    out=dict(base=q0,h2=q2,h4=q4,nonuniform4=check,first_change=d1,second_change=d2,
        fine_shift=q4-q0,layout_discrepancy=layout,observed_ratio=None,observed_order=None,
        extrapolated_value=None,remaining_estimate=None,arithmetic_bound=None,
        resolution_screen='floating-point heuristic; no propagated primary arithmetic bound')
    if max(abs(d1),abs(d2),layout)<=floor:
        out.update(status='below_resolution',remaining_estimate=floor)
    elif d1*d2>0 and floor<abs(d2)<abs(d1):
        ratio=abs(d2/d1);signed=d2*ratio/(1-ratio)
        out.update(observed_ratio=ratio,observed_order=-math.log2(ratio),extrapolated_value=q4+signed)
        # A single successful triplet is not a certified asymptotic regime.
        if abs(check-(q4+signed))<=max(abs(d2),floor):
            out.update(status='conditionally_consistent',remaining_estimate=max(abs(signed),layout,floor))
        else:out.update(status='layout_disagrees')
    else:out.update(status='non_asymptotic')
    out['observed_reference_envelope']=max(abs(q-q0) for q in (q2,q4,check))
    out['reference_allowance']=(abs(q4-q0)+out['remaining_estimate']
        if out['remaining_estimate'] is not None else None)
    return out

def prepare(args):
    root=args.root.resolve()
    if (root/'design.json').exists():raise ValueError('cannot rebuild a frozen stability design')
    root.mkdir(parents=True,exist_ok=True);(root/'pilot').mkdir(exist_ok=True)
    work=_SOURCE_ROOT;D=work/'output/diagnostics'
    availability=json.loads(args.availability.read_text())
    sources=sorted({r['source'] for r in availability['banks'].values()})
    if any(r['prefix'] not in ('','evidence') for r in availability['banks'].values()):raise ValueError('unknown bank layout')
    loc=dict(local_diagnostics=str(D),remote_diagnostics='/n/netscratch/yin_lab/Lab/yutaizhang',
        runtime=dict(local=str(D/'genus1_t025_tailored_384_20260923'),
            remote='/n/netscratch/yin_lab/Lab/yutaizhang/genus1_t025_tailored_384_20260923'),bank_sources=sources)
    (root/'locations.json').write_text(json.dumps(loc,indent=2)+'\n')
    for path in (Path(__file__),work/'code/one_loop/three_point/other_scripts/genus1_moduli_amplitude.py',
            work/'code/one_loop/three_point/other_scripts/genus1_moduli_panel_campaign.py',work/'code/one_loop/three_point/moduli_integration/genus1_moduli_panels.py'):
        shutil.copy2(path,root/'pilot'/path.name)
    configure(root);physics=f.make_plan(p['config']);atlas=f.make_atlas(.25);pins={}
    def read(path):
        value=f.spectrum.read(path);pins[str(path.relative_to(D))]=f.shared.sha(path);return value
    refpath=D/'genus1_upper_pair_t025_refinement_384_20260924/amplitude.json'
    if f.shared.sha(refpath)!=REFERENCE_SHA:raise ValueError('changed reference amplitude')
    reference=read(refpath)
    current=read(D/'genus1_primary_length_angle_t025_t095_384_20260924/energies/t0250/amplitude.json')
    basepair=read(D/'genus1_upper_pair_t025_refinement_384_20260924/baseline.json')
    selected=[r for r in current['primary_cells'] if r['cell']['key'].endswith(':band0')]
    tail=[r for r in basepair['cells'] if r['cell']['term']=='tail:pair_face_analytic']
    if len(selected)!=24 or len(tail)!=2:raise ValueError('changed tested region')
    ids={r['cell']['id'] for r in selected};cache={};primary_keys={};original_weights={}
    first=D/'genus1_t025_primary_first_384_20260923';fd=read(first/'design.json')
    if f.fingerprint(fd)!=FIRST_SHA:raise ValueError('foreign primary source')
    for i,b in enumerate(fd['batches']):
        if b['cell_id'] not in ids:continue
        geometry=read(first/b['path']);values=read(first/f'results/batch_{i:05d}.json')
        if f.shared.sha(first/b['path'])!=b['sha256'] or values['design_sha256']!=FIRST_SHA or set(geometry['samples'])!=set(values['values']):
            raise ValueError('foreign primary point evidence')
        for key,s in geometry['samples'].items():
            cache[key]=values['values'][key]
            sp=f.ShortPoint.from_record(s['short']);primary_keys[(sp.sector,sp.s1,sp.s2)]=s['grids'][0]
            original_weights[(geometry['cell_id'],key)]=math.fsum(w for rule,w,tag in geometry['weights'][key] if rule=='high')
    broad=D/'genus1_moduli_error_t025_t095_384_20260924/energies/t0250';bd=read(broad/'design.json')
    if f.fingerprint(bd)!=BROAD_SHA:raise ValueError('foreign saved tail geometry')
    class Builder(production.Builder):
        def __init__(self):
            self.root,self.p,self.batch_size=root,p,256;self.f=f;self.planner=f.spectrum.Planner(p,atlas)
            self.grids={};self.routing={};self.primary_cache={};self.available={}
            self.batches=[];self.costs=[];self.cells=[];self.reset()
        def use_grid(self,key,spec=None):
            if key in self.grids:return key
            if spec is None:
                # Only tail keys reach here. Read their original immutable specification.
                path=f.spectrum.grid_directory(broad,key)/'spec.json'
                if not path.exists():
                    route=availability['banks'][key]
                    path=D/route['source']/route['prefix']/'prepared/spectrum/grids'/key/'spec.json'
                spec=read(path)
            if f.fingerprint(spec['grid'])!=key:raise ValueError('foreign bank specification')
            f.check_momentum_grid(p,spec)
            self.grids[key]=dict(kind=spec['case']['kind'],axes=[len(r['nodes']) for r in spec['grid']['rules']])
            if key in availability['banks']:
                r=availability['banks'][key]
                self.routing[key]=dict(origin=sources.index(r['source']),ready_sha256=r['ready_sha256'])
            else:f.spectrum.publish(f.spectrum.grid_directory(root,key)/'spec.json',spec)
            return key
        def register(self,value,primary):
            if not primary:raise ValueError('tail geometry must use pinned saved grid keys')
            c=(value.sector,value.s1,value.s2)
            if c in self.primary_cache:return self.primary_cache[c]
            # Pin the actual saved nodes and weights at these exact lengths.
            # Regenerating the same floating-point rule on another platform
            # can change its serialized bits; no approximate key matching is used.
            key=primary_keys[c];route=availability['banks'][key]
            spec=read(D/route['source']/route['prefix']/'prepared/spectrum/grids'/key/'spec.json')
            self.primary_cache[c]=self.use_grid(key,spec);return key
        def saved(self,rule,key,s,weight,tag):
            raw={k:s[k] for k in ('kind','requests')}
            if f.fingerprint(raw)!=key:raise ValueError('changed raw collision identity')
            if key not in self.samples:
                # Base values are exact imports; their banks are never loaded.
                grids=list(s['grids']) if rule=='base' else [self.use_grid(k) for k in s['grids']]
                self.samples[key]=dict(raw,grids=grids,max_weight=weight)
                self.weights[key]=[]
            elif self.samples[key]['grids']!=s['grids']:
                raise ValueError('same collision geometry has different actual momentum grids')
            self.samples[key]['max_weight']=max(self.samples[key]['max_weight'],weight)
            self.weights[key].append([rule,weight,tag]);self.rules[rule]+=1
    builder=Builder();base_checks=0
    for row in selected:
        cell=row['cell']
        for name,parts in PRIMARY_RULES.items():
            for sample,w,tag in primary_observations(cell,parts,physics):
                key=f.fingerprint(sample)
                if name=='base':
                    expected=original_weights.get((cell['id'],key))
                    if key not in cache or expected is None or not math.isclose(w,expected,rel_tol=2e-13,abs_tol=1e-18):
                        raise ValueError('base geometry differs from retained primary quadrature')
                    base_checks+=1
                builder.add(name,sample,w,tag)
        builder.finish_cell(cell)
    runtime=Path(loc['runtime']['local'])
    for row in tail:
        cell=row['cell'];rules=read(runtime/f"cells/{cell['id']}.json")
        for oldkey,w in rules['rules']['3']:
            vp=runtime/f'points/{oldkey}/value.json'
            if not vp.exists():vp=runtime/f'evidence/points/{oldkey}/value.json'
            value=read(vp);candidates=[runtime/f'{prefix}points/{oldkey}/sample.json' for prefix in ('','evidence/')]
            sp=next((q for q in candidates if q.exists() and f.shared.sha(q)==value['sample_sha256']),None)
            if sp is None or value['plan_sha256']!=SOURCE_PLAN:raise ValueError('foreign tail base value')
            s=read(sp)
            if s['kind']=='zero':continue
            raw={k:s[k] for k in ('kind','requests')};key=f.fingerprint(raw)
            cache[key]=f.Value.from_record(value['value']).scaled(1/s['jacobian']).record()
            builder.saved('base',key,s,float(w*s['jacobian']),'legacy_pair')
        saved=next(c for c in bd['cells'] if c['cell']['id']==cell['id'])
        for i in saved['batch_indices']:
            b=bd['batches'][i];geometry=read(broad/b['path'])
            if f.shared.sha(broad/b['path'])!=b['sha256']:raise ValueError('changed tail check geometry')
            for key,s in geometry['samples'].items():
                for rule,w,tag in geometry['weights'][key]:
                    if rule not in TAIL_RULES:raise ValueError('unplanned tail rule')
                    builder.saved(rule,key,s,w,tag)
        builder.finish_cell(cell)
    used_cache={};new_counts=Counter()
    for b in builder.batches:
        geo=f.spectrum.read(root/b['path']);fresh=sum(k not in cache for k in geo['samples'])
        b['new_points']=fresh;used_cache.update({k:cache[k] for k in geo['samples'] if k in cache})
        new_counts['primary' if next(iter(geo['samples'].values()))['kind']=='primary' else 'tail_pair']+=fresh
    fresh=sorted(set(builder.grids)-set(builder.routing));specpins={k:f.shared.sha(f.spectrum.grid_directory(root,k)/'spec.json') for k in fresh}
    f.spectrum.publish(root/'reused_values.json',used_cache)
    f.spectrum.publish(root/'baseline.json',dict(reference=reference,cells=selected+tail,source_files=pins,
        reference_file_sha256=REFERENCE_SHA,primary_base_points_checked=base_checks))
    preparations,ploads=campaign.assign([20.3*math.prod(builder.grids[k]['axes'])/512 for k in fresh],args.workers)
    costs=[b['new_points']*(.055273 if b['cell_id'] in ids else .08) for b in builder.batches]
    contractions,cloads=production.assign_geometry_groups(costs,[b['coordinate_group'] for b in builder.batches],args.workers)
    summary=dict(points=sum(b['points'] for b in builder.batches),new_points=dict(new_counts),
        reused_points=len(used_cache),new_banks=len(fresh),reused_banks=len(builder.routing),
        batches=len(builder.batches),cells=len(builder.cells),estimated_useful_cpu_seconds=sum(costs)+sum(ploads),
        estimated_prepare_max_seconds=max(ploads),estimated_contract_max_seconds=max(cloads))
    d=dict(format='genus1-direct-stability-v1',t=.25,source_plan_sha256=SOURCE_PLAN,reference_file_sha256=REFERENCE_SHA,
        geometry_scope=dict(primary_band=0,near_head_vertical_separation=.25,angular_head_order=3,broad_tail_order=2,
            primary_rules=PRIMARY_RULES,tail_rules=TAIL_RULES,tail_orders=[3,3,3,3],nonuniform_edges=CHECK),
        cells=builder.cells,batches=builder.batches,fresh_banks=fresh,routing=builder.routing,spec_sha256=specpins,
        prepare=preparations,contract=contractions,workers=args.workers,summary=summary,
        banks_by_momentum_axes=dict(Counter(str(tuple(v['axes'])) for v in builder.grids.values())),
        input_sha256={n:f.shared.sha(root/n) for n in ('locations.json','baseline.json','reused_values.json')},
        pilot_sources={str(q.relative_to(root)):f.shared.sha(q) for q in sorted((root/'pilot').glob('*.py'))},
        momentum_design_changed=False,physical_cutoffs_changed=False,central_value_replaced=False,
        matrix_model_used=False,surrogate_used=False,automatic_refinement=False,created_utc=f.shared.utc())
    f.spectrum.publish(root/'design.json',d);print(json.dumps(summary,indent=2),flush=True)

def checked(root):
    configure(root);d=f.spectrum.read(root/'design.json')
    if d['source_plan_sha256']!=SOURCE_PLAN or d['reference_file_sha256']!=REFERENCE_SHA:raise ValueError('foreign study')
    for rel,sha in {**d['input_sha256'],**d['pilot_sources']}.items():
        if f.shared.sha(root/rel)!=sha:raise ValueError('changed input '+rel)
    return d

def worker(root,phase):
    d=checked(root);identity=f.fingerprint(d);rank=int(os.environ['SLURM_PROCID'])
    if int(os.environ['SLURM_NTASKS'])!=d['workers'] or not 0<=rank<d['workers']:raise ValueError('changed allocation')
    if campaign.completed_worker(root,d,identity,phase,rank):return
    start=f.clock()
    if phase=='prepare':
        for i in d['prepare'][rank]:
            key=d['fresh_banks'][i]
            if f.shared.sha(f.spectrum.grid_directory(root,key)/'spec.json')!=d['spec_sha256'][key]:raise ValueError('changed bank input')
            while True:
                row=f.spectrum.build(root,p,key,cpu_budget=240,max_new_chunks=64,max_new_elliptic_nodes=1024)
                if row['complete_rule']:break
                if (row.get('failure') or {}).get('type')!='TimeLimit':raise RuntimeError(str(row))
    elif phase=='contract':
        from genus1_boundary_batch import GeometryStore
        physics=f.make_plan(p['config']);atlas=f.make_atlas(.25);reuse=f.spectrum.read(root/'reused_values.json')
        geometry=GeometryStore(atlas,root/f'work/geometry/{rank:03d}')
        runtime=f.spectrum.Runtime(root,p,atlas,geometry,point_budget=1e-12);runtime.banks=campaign.Banks(root,d)
        try:
            for i in d['contract'][rank]:
                samples=campaign.batch(root,d,i);path=root/f'results/batch_{i:05d}.json'
                if path.exists():
                    r=f.spectrum.read(path)
                    if r['design_sha256']!=identity or set(r['values'])!=set(samples):raise ValueError('foreign saved result')
                    continue
                tick=f.clock();values={k:reuse[k] for k in samples if k in reuse};keys=[k for k in samples if k not in reuse]
                if keys and samples[keys[0]]['kind']=='primary':
                    shorts=[f.ShortPoint.from_record(samples[k]['short']) for k in keys]
                    grids=[samples[k]['grids'][0] for k in keys]
                    vs=runtime.primary(shorts,grids,geometry.prepare(shorts),physics.policy,physics.angle_order)
                    values.update({k:f.boundary.amplitude_value(v,1.).record() for k,v in zip(keys,vs)})
                else:
                    for key in keys:
                        s=samples[key];vs=[runtime.collide(f.request_from_record(a),grid,s['max_weight']).scaled(a['multiplicity'])
                            for a,grid in zip(s['requests'],s['grids'])]
                        values[key]=f.boundary.amplitude_value(f.combine(vs),1.).record()
                f.spectrum.publish(path,dict(design_sha256=identity,values=values,new_points=len(keys),**f.timing(tick)))
        finally:geometry.close()
    else:raise ValueError('unknown phase')
    f.spectrum.publish(root/f'timings/{phase}_{rank:03d}.json',dict(design_sha256=identity,phase=phase,rank=rank,
        job_id=os.environ.get('SLURM_JOB_ID','local-smoke'),**f.timing(start)))

def reduce(root):
    d=checked(root);identity=f.fingerprint(d);baseline=f.spectrum.read(root/'baseline.json')
    old={c['cell']['id']:c for c in baseline['cells']};cells=[];seen=set();timing=Counter()
    for c in d['cells']:
        partial=defaultdict(list);counts=Counter();is_primary=c['cell']['term'].startswith('primary:')
        rules=set(PRIMARY_RULES if is_primary else TAIL_RULES)
        for i in c['batch_indices']:
            if i in seen:raise ValueError('duplicate batch')
            seen.add(i);b=d['batches'][i];geo=f.spectrum.read(root/b['path'])
            if f.shared.sha(root/b['path'])!=b['sha256']:raise ValueError('changed quadrature')
            row=f.spectrum.read(root/f'results/batch_{i:05d}.json')
            if row['design_sha256']!=identity:raise ValueError('foreign result')
            sums,_,ns=production.batch_estimates(geo,row['values'],f.Value,f.combine,allowed_rules=rules)
            counts.update(ns)
            for rule,v in sums.items():partial[rule].append(v)
            timing['contraction_cpu_seconds']+=row['cpu_seconds']
        if dict(counts)!=c['rule_points'] or set(partial)!=rules:raise ValueError('incomplete rule')
        totals={k:f.combine(v).value.real for k,v in partial.items()};prior=old[c['cell']['id']]
        if not is_primary and not math.isclose(totals['base'],prior['high']['value'][0],rel_tol=1e-12,abs_tol=1e-18):
            raise ValueError('tail base no longer reconstructs current amplitude')
        if is_primary:
            assessments={'angular_heads':sequence(*(totals[k] for k in ('base','head_h2','head_h4','head_check')))}
            mixed=totals['head_h2']-totals['outer_h2']-totals['inner_h2']+totals['base']
            actual2=totals['head_h2']-totals['base']
        else:
            assessments={f'axis{i}':sequence(totals['base'],*(totals[f'axis{i}_{s}'] for s in ('h2','h4','check'))) for i in range(4)}
            mixed=totals['mixed_h2']-totals['base']-math.fsum(totals[f'axis{i}_h2']-totals['base'] for i in range(4))
            actual2=totals['mixed_h2']-totals['base']
        valid=all(v['reference_allowance'] is not None for v in assessments.values())
        conditional=math.fsum(v['reference_allowance'] for v in assessments.values())+abs(mixed) if valid else None
        observed=math.fsum(v['observed_reference_envelope'] for v in assessments.values())+abs(mixed)
        cells.append(dict(cell=c['cell'],rules=totals,assessments=assessments,mixed_discrepancy=mixed,
            actual_h2_shift=actual2,conditional_allowance_at_reference=conditional,observed_stability_envelope=observed,
            previous_cell_indicator=prior['indicator'],unresolved=not valid,
            previous_indicator_covers_untested_directions=is_primary))
    if seen!=set(range(len(d['batches']))):raise ValueError('incomplete studied domain')
    for phase in ('prepare','contract'):
        for rank in range(d['workers']):
            row=f.spectrum.read(root/f'timings/{phase}_{rank:03d}.json')
            if row['design_sha256']!=identity or row['phase']!=phase or row['rank']!=rank:raise ValueError('missing worker completion')
            timing[phase+'_worker_cpu_seconds']+=row['cpu_seconds']
    families={}
    for name in ('primary_heads','tail_pair'):
        rows=[c for c in cells if c['cell']['term'].startswith('primary:')==(name=='primary_heads')]
        valid=all(not c['unresolved'] for c in rows)
        families[name]=dict(cells=len(rows),unresolved_cells=sum(c['unresolved'] for c in rows),
            sequence_status_counts=dict(Counter(a['status'] for c in rows for a in c['assessments'].values())),
            actual_h2_shift=math.fsum(c['actual_h2_shift'] for c in rows),
            sum_absolute_h2_shifts=math.fsum(abs(c['actual_h2_shift']) for c in rows),
            summed_mixed_allowance=math.fsum(abs(c['mixed_discrepancy']) for c in rows),
            conditional_allowance_at_reference=math.fsum(c['conditional_allowance_at_reference'] for c in rows) if valid else None,
            observed_stability_envelope=math.fsum(c['observed_stability_envelope'] for c in rows),
            previous_selected_cell_indicator=math.fsum(c['previous_cell_indicator'] for c in rows))
    ref=baseline['reference'];delta=math.fsum(c['actual_h2_shift'] for c in cells)
    result=dict(design_sha256=identity,t=.25,reference_amplitude=ref['amplitude'],reference_unchanged=True,
        complete_planned_study=True,total_physical_error_bar=None,
        actual_combined_two_panel_amplitude=ref['amplitude']+delta,actual_combined_two_panel_shift=delta,
        families=families,cells=cells,timings=dict(timing),previous_moduli_indicator=ref['moduli_indicator'],
        block_indicator=ref['block_indicator'],compact_sampling_halfwidth95=ref['compact_sampling_halfwidth95'],
        unknown_errors=ref['unknown_errors'],untested_primary_length_ratio_and_other_regions_retained=True,
        deterministic_estimate_is_confidence_interval=False,full_amplitude_certified=False,matrix_model_used=False,
        limitations=['A three-level sequence plus a layout check supplies conditional empirical evidence, not a certified bound.',
            'The primary check keeps original short-length nodes and covers only band-zero near-pair heads.',
            'One combined two-panel check does not calibrate higher mixed-direction effects.',
            'Failed sequences receive no extrapolated error bar; their observed range is not a bound.',
            'Momentum, analytic remainders, global cutoff stability and primary arithmetic bounds remain unquantified.'],
        created_utc=f.shared.utc())
    f.spectrum.publish(root/'result.json',result)
    report=['# Direct moduli stability study at t=0.25','',f"Reference amplitude: {ref['amplitude']:.16g} (unchanged).",'',
        f"Actual simultaneous two-panel substitution: {result['actual_combined_two_panel_amplitude']:.16g}; signed change {delta:+.8g}.",'',
        '| Tested family | Unresolved cells | Conditional reference allowance | Observed stability envelope |',
        '|---|---:|---:|---:|']
    for name,r in families.items():
        a=r['conditional_allowance_at_reference'];text='unresolved' if a is None else f'{a:.8g} ({100*a/abs(ref["amplitude"]):.4g}%)'
        report.append(f"| {name} | {r['unresolved_cells']}/{r['cells']} | {text} | {r['observed_stability_envelope']:.8g} |")
    report+=['','These are direct deterministic quadrature comparisons. No fitted function or matrix-model prediction enters the estimate.',
        '',*result['limitations'],'',f"Block sensitivity: {ref['block_indicator']:.8g}. Compact pointwise 95% half-width: {ref['compact_sampling_halfwidth95']:.8g}."]
    (root/'REPORT.md').write_text('\n'.join(report)+'\n')
    f.spectrum.publish(root/'completion.json',dict(design_sha256=identity,result_file_sha256=f.shared.sha(root/'result.json'),
        report_file_sha256=f.shared.sha(root/'REPORT.md'),complete_planned_study=True,created_utc=f.shared.utc()))
    print(json.dumps({k:result[k] for k in ('reference_amplitude','actual_combined_two_panel_shift','families')},indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('prepare','worker','reduce'))
    parser.add_argument('--root',required=True,type=Path);parser.add_argument('--availability',type=Path)
    parser.add_argument('--workers',type=int,default=384);parser.add_argument('--phase',choices=('prepare','contract'))
    args=parser.parse_args()
    if args.action=='prepare':prepare(args)
    elif args.action=='worker':worker(args.root,args.phase)
    else:reduce(args.root)
