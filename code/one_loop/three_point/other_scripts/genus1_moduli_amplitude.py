"""Bounded full-domain moduli production on a pinned retained CFT estimator.

The pilot supplies numerical choices, not an amplitude correction. Every
primary sector and the full pair domain are integrated. Unchanged compact
and projected-triple contributions are explicit, identity-checked imports.
Neither planning nor acceptance uses the matrix-model target.
"""

from genus1_source_paths import ROOT as _SOURCE_ROOT, source_path as _source_path
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import json
import math
from pathlib import Path
import shutil
import sys

sys.dont_write_bytecode = True
import genus1_moduli_panel_campaign as campaign


def assign_geometry_groups(costs, groups, workers):
    """Co-locate identical coordinate batches; retain every labeled CFT call."""
    if len(costs)!=len(groups):raise ValueError('one geometry group per work unit required')
    members=defaultdict(list)
    for i,key in enumerate(groups):members[key].append(i)
    units=list(members.values())
    packed,loads=campaign.assign([math.fsum(costs[i] for i in row) for row in units],workers)
    return [[i for unit in row for i in units[unit]] for row in packed],loads


def source_result(root, p):
    row = campaign.f.spectrum.read(root/'amplitude_wave_000.json')
    if (row['plan_sha256'] != campaign.f.fingerprint(p) or
            not row['complete_domain'] or row['boundary']['next_wave'] is not None):
        raise ValueError('complete retained source amplitude required')
    expected = set(campaign.f.boundary.term_names('analytic'))
    if set(row['boundary']['components']) != expected:
        raise ValueError('incomplete source component manifest')
    if row['compact']['plan'] != p['physics']:
        raise ValueError('compact import has different physics')
    return row


def validate_boxes(rows, dimensions):
    if any(len(b)!=dimensions or any(not 0<=a<z<=1 for a,z in b) for b in rows):
        raise ValueError('invalid domain cell')
    volume=math.fsum(math.prod(z-a for a,z in b) for b in rows)
    if not math.isclose(volume,1.,rel_tol=0,abs_tol=1e-14):
        raise ValueError('cells do not cover their root')
    for i,a in enumerate(rows):
        for b in rows[:i]:
            if all(min(x[1],y[1])>max(x[0],y[0]) for x,y in zip(a,b)):
                raise ValueError('overlapping domain cells')


def validate_cells(cells, *, require_pair=False):
    """Check the saved rectangular partition independently of quadrature nodes."""
    grouped = defaultdict(list)
    for c in cells:
        if c['id'] != campaign.f.fingerprint({k:v for k,v in c.items() if k != 'id'}):
            raise ValueError('changed source cell identity')
        if c['term'].startswith('primary:'): grouped[c['key']].append(c['bounds'])
    expected = {f'primary:{s}:order{o}:band{b}' for s in ('012','021','102','120','201','210')
                for o in (0,1) for b in (0,1)}
    if set(grouped) != expected: raise ValueError('missing labeled primary sector')
    for rows in grouped.values():validate_boxes(rows,4)
    if require_pair:
        for region in ('bulk','tail'):
            rows=[c['bounds'] for c in cells if c['term']==region+':pair_face_analytic']
            validate_boxes(rows,4)
            if region=='bulk':
                if any(b[2:]!=[[0.,1.],[0.,1.]] for b in rows):
                    raise ValueError('bulk pair replacement requires unsplit puncture support')
                validate_boxes([b[:2] for b in rows],2)


class Builder:
    """Keep at most one cell in memory; pin each batch's weights with its nodes."""
    def __init__(self, root, p, atlas, batch_size):
        self.root,self.p,self.batch_size = root,p,batch_size
        self.f=campaign.f;self.planner=self.f.spectrum.Planner(p,atlas)
        self.grids={};self.routing={};self.primary_cache={};self.available={}
        self.batches=[];self.costs=[];self.cells=[]
        for i,base in enumerate([campaign.source,*campaign.parents]):
            for prefix in (base/'evidence',base):
                for path in (prefix/'prepared/spectrum/grids').glob('*/ready.json'):
                    self.available.setdefault(path.parent.name,(i,path))
        self.reset()

    def reset(self):
        self.samples={};self.weights={};self.rules=Counter()

    def register(self, value, primary):
        f=self.f;cache=(value.sector,value.s1,value.s2) if primary else None
        if cache is not None and cache in self.primary_cache:return self.primary_cache[cache]
        key,spec=self.planner.grid(value,primary,self.p['profile']);f.check_momentum_grid(self.p,spec)
        if key not in self.grids:
            self.grids[key]=dict(kind=spec['case']['kind'],axes=[len(r['nodes']) for r in spec['grid']['rules']])
            if key in self.available:
                origin,path=self.available[key];ready=f.spectrum.read(path)
                if ready['grid_sha256'] != key:raise ValueError('foreign reused bank')
                self.routing[key]=dict(origin=origin,ready_sha256=f.shared.sha(path))
            else:f.spectrum.publish(f.spectrum.grid_directory(self.root,key)/'spec.json',spec)
        if cache is not None:self.primary_cache[cache]=key
        return key

    def add(self, rule, sample, weight, tag='all'):
        if not math.isfinite(weight) or weight<=0:raise ValueError('finite positive physical weight required')
        f=self.f;key=f.fingerprint(sample)
        if key not in self.samples:
            primary=sample['kind']=='primary'
            values=([f.ShortPoint.from_record(sample['short'])] if primary else
                    [f.request_from_record(r) for r in sample['requests']])
            self.samples[key]=dict(sample,grids=[self.register(v,primary) for v in values],max_weight=weight)
            self.weights[key]=[]
        self.samples[key]['max_weight']=max(weight,self.samples[key]['max_weight'])
        self.weights[key].append([rule,float(weight),tag]);self.rules[rule]+=1

    def finish_cell(self, cell):
        groups=defaultdict(list);f=self.f
        for key,s in self.samples.items():groups[tuple(s['grids'])].append(key)
        before=len(self.batches)
        for group in groups.values():
            for start in range(0,len(group),self.batch_size):
                keys=group[start:start+self.batch_size];index=len(self.batches)
                rel=f'geometry/batch_{index:05d}.json'
                payload=dict(cell_id=cell['id'],samples={k:self.samples[k] for k in keys},
                             weights={k:self.weights[k] for k in keys})
                f.spectrum.publish(self.root/rel,payload)
                primary=self.samples[keys[0]]['kind']=='primary'
                # The existing GeometryStore separates its sector-independent
                # log-coordinate/features cache from sector matter factors.
                # Only byte-identical physical coordinate lists are grouped.
                common=([{k:v for k,v in self.samples[key]['short'].items() if k!='sector'}
                         for key in keys] if primary else [cell['id'],index])
                self.batches.append(dict(path=rel,sha256=f.shared.sha(self.root/rel),
                    cell_id=cell['id'],points=len(keys),point_ids_sha256=f.fingerprint(sorted(keys)),
                    coordinate_group=f.fingerprint(common)))
                self.costs.append(len(keys)*(.111 if primary else .080))
        self.cells.append(dict(cell=cell,rule_points=dict(self.rules),points=len(self.samples),
                               batch_indices=list(range(before,len(self.batches)))))
        print(json.dumps(dict(cell=cell['id'],term=cell['term'],points=len(self.samples),
                              grids=len(self.grids))),flush=True)
        self.reset()


def prepare(args):
    root=args.root;root.mkdir(parents=True,exist_ok=False);(root/'pilot').mkdir()
    for path in (Path(__file__),Path(campaign.__file__),
                 _source_path('code/one_loop/three_point/moduli_integration/genus1_moduli_panels.py')):
        shutil.copy2(path,root/'pilot'/path.name)
    if len(args.parent)!=len(args.parent_remote):raise ValueError('parent locations must be paired')
    (root/'locations.json').write_text(json.dumps(dict(source=str(args.source.resolve()),
        source_remote=args.source_remote,source_plan_sha256=json.loads((args.source/'plan.json').read_text())['sha256'],
        parents=[dict(local=str(p.resolve()),remote=r) for p,r in zip(args.parent,args.parent_remote)]),indent=2)+'\n')
    p=campaign.configure(root);f=campaign.f;g=campaign.g;physics=f.make_plan(p['config'])
    original=source_result(campaign.source,p)
    request=f.spectrum.read(campaign.source/'waves/000_request.json')
    if request['plan_sha256']!=f.fingerprint(p):raise ValueError('foreign domain partition')
    validate_cells(request['cells'],require_pair=True)
    choices=json.loads(args.choices.read_text())
    expected={'primary_low','primary_high','pair_bulk_low','pair_bulk_high','pair_tail_low','pair_tail_high',
              'separation_splits','readiness'}
    if set(choices)!=expected:raise ValueError('explicit complete numerical choices required')
    readiness=choices['readiness'];path=Path(readiness['local_result'])
    if f.shared.sha(path)!=readiness['result_file_sha256']:raise ValueError('pilot readiness artifact changed')
    pilot_result=f.spectrum.read(path)
    parent_designs=[f.spectrum.read(x/'design.json') for x in campaign.parents]
    if any(d['source_plan_sha256']!=f.fingerprint(p) for d in parent_designs):raise ValueError('foreign CFT parent')
    if pilot_result['design_sha256'] not in [f.fingerprint(d) for d in parent_designs]:
        raise ValueError('readiness pilot is not a pinned parent')
    for label in expected-{'readiness','separation_splits'}:
        row=choices[label];size=2 if label.startswith('primary') else 4
        if len(row)!=size or any(type(n) is not int or not 2<=n<=6 for n in row):
            raise ValueError('bounded, explicit moduli orders required')
    for family in ('primary','pair_bulk','pair_tail'):
        low,high=(choices[family+'_'+s] for s in ('low','high'))
        if low==high or any(a>b for a,b in zip(low,high)):
            raise ValueError('one distinct nondecreasing comparison required')
    if args.batch_size<1 or args.batch_size>1024:raise ValueError('bounded batch size required')
    imported={k:v for k,v in original['boundary']['components'].items() if ':triple_' in k}
    if len(imported)!=6:raise ValueError('all six projected triple terms required')
    imports=dict(source_plan_sha256=f.fingerprint(p),
        amplitude_file_sha256=f.shared.sha(campaign.source/'amplitude_wave_000.json'),
        compact=original['compact'],components=imported,unknown_errors=original['unknown_errors'])
    f.spectrum.publish(root/'imports.json',imports)
    f.spectrum.publish(root/'readiness.json',dict(choices=choices,pilot_result=pilot_result))
    builder=Builder(root,p,f.make_atlas(physics.t),args.batch_size)
    for cell in request['cells']:
        if not cell['term'].startswith('primary:'):continue
        for label in ('low','high'):
            length_order,angle_order=choices['primary_'+label]
            for short,w in g.primary_cell_rule(physics,cell,length_order,angle_order,angular_heads=True):
                builder.add(label,dict(kind='primary',short=short.record()),w)
        builder.finish_cell(cell)
    # The old tail pair cells split a fractional puncture coordinate. The new
    # exact torus partition replaces their union, once, at every tail height.
    for region in ('bulk','tail'):
        bounds=([c['bounds'][:2] for c in request['cells'] if c['term']=='bulk:pair_face_analytic']
                if region=='bulk' else [[[0.,1.],[0.,1.]]])
        for b in bounds:
            cell=dict(term=region+':pair_face_analytic',key=region+':physical_pair',bounds=b)
            cell['id']=f.fingerprint(cell)
            for label in ('low','high'):
                orders=choices['pair_'+region+'_'+label]
                for sample,meta in g.pair_rule(physics,region,orders,bounds=b,
                                             separation_splits=choices['separation_splits']):
                    builder.add(label,dict(kind='collision',requests=f.encoded([asdict(r) for r in sample.requests])),
                                sample.jacobian,meta['puncture'])
            builder.finish_cell(cell)
    fresh=sorted(set(builder.grids)-set(builder.routing))
    costs=[(100. if builder.grids[k]['kind']=='primary' else 20.3)*math.prod(builder.grids[k]['axes'])/512 for k in fresh]
    preparations,loads=campaign.assign(costs,args.workers)
    contractions,cloads=assign_geometry_groups(builder.costs,
        [b['coordinate_group'] for b in builder.batches],args.workers)
    points=sum(b['points'] for b in builder.batches)
    if points>4000000 or len(fresh)>24000:raise ValueError('bounded full-production work cap exceeded')
    d=dict(format='genus1-streamed-moduli-amplitude-v1',mode='amplitude',created_utc=f.shared.utc(),
        source_plan_sha256=f.fingerprint(p),source_domain_sha256=f.fingerprint(request),
        parent_identities=[f.fingerprint(d) for d in parent_designs],
        pilot_sources={str(q.relative_to(root)):f.shared.sha(q) for q in sorted((root/'pilot').glob('*.py'))},
        physics=p['physics'],momentum_mode=p['momentum_mode'],elliptic_orders=p['elliptic_orders'],
        imports_sha256=f.shared.sha(root/'imports.json'),readiness_sha256=f.shared.sha(root/'readiness.json'),
        choices=choices,cells=builder.cells,batches=builder.batches,fresh_banks=fresh,routing=builder.routing,
        workers=args.workers,prepare=preparations,contract=contractions,
        boundary_indicator_budget=p['config']['boundary_controls']['total_absolute_tolerance'],
        component_tolerances={k:v['absolute_tolerance'] for k,v in p['quadrature']['controls'].items()},
        summary=dict(points=points,new_banks=len(fresh),reused_banks=len(builder.routing),batches=len(builder.batches),
            cells=len(builder.cells),fresh_by_family=dict(Counter(builder.grids[k]['kind'] for k in fresh)),
            coordinate_groups=len({b['coordinate_group'] for b in builder.batches}),
            banks_by_momentum_axes=dict(Counter(str(tuple(v['axes'])) for v in builder.grids.values())),
            estimated_prepare_max_seconds=max(loads),estimated_contract_max_seconds=max(cloads)),
        full_amplitude_certified=False,physical_cutoffs_changed=False,momentum_design_changed=False,
        automatic_refinement=False,maximum_planned_comparisons=1)
    f.spectrum.publish(root/'design.json',d);print(json.dumps(d['summary']),flush=True)


def batch_estimates(payload, values, Value, combine, *, allowed_rules=('low','high')):
    """Reduce bounded point evidence, preserving unknown errors as unknown."""
    if set(payload['samples'])!=set(values) or set(payload['weights'])!=set(values):
        raise ValueError('incomplete point/weight/value coverage')
    rows=defaultdict(list);panels=defaultdict(list);counts=Counter()
    for key,record in values.items():
        v=Value.from_record(record)
        for rule,w,tag in payload['weights'][key]:
            if rule not in allowed_rules or not math.isfinite(w) or w<=0:
                raise ValueError('invalid production weight')
            rows[rule].append(v.scaled(w));panels[(rule,tag)].append(v.value.real*w);counts[rule]+=1
    return ({k:combine(v) for k,v in rows.items()},
            {k:math.fsum(v) for k,v in panels.items()},counts)


def reduce(root):
    p=campaign.configure(root);f=campaign.f;d=campaign.checked(root,p);identity=f.fingerprint(d)
    if d['format']!='genus1-streamed-moduli-amplitude-v1':raise ValueError('full amplitude design required')
    if (f.shared.sha(root/'imports.json')!=d['imports_sha256'] or
            f.shared.sha(root/'readiness.json')!=d['readiness_sha256']):raise ValueError('changed pinned evidence')
    imports=f.spectrum.read(root/'imports.json')
    if f.shared.sha(campaign.source/'amplitude_wave_000.json')!=imports['amplitude_file_sha256']:
        raise ValueError('changed source amplitude')
    original=source_result(campaign.source,p);cell_rows=[];seen=set();points=0
    for c in d['cells']:
        partial=defaultdict(list);panels=defaultdict(list);counts=Counter()
        for i in c['batch_indices']:
            if i in seen:raise ValueError('a geometry batch was reduced twice')
            seen.add(i);descriptor=d['batches'][i];path=root/descriptor['path']
            if f.shared.sha(path)!=descriptor['sha256']:raise ValueError('changed geometry or weights')
            payload=f.spectrum.read(path);row=f.spectrum.read(root/f'results/batch_{i:05d}.json')
            if (payload['cell_id']!=c['cell']['id'] or descriptor['cell_id']!=c['cell']['id'] or
                row['design_sha256']!=identity or len(row['values'])!=descriptor['points'] or
                f.fingerprint(sorted(row['values']))!=descriptor['point_ids_sha256']):
                raise ValueError('foreign or incomplete batch result')
            sums,tags,ns=batch_estimates(payload,row['values'],f.Value,f.combine)
            for rule,v in sums.items():partial[rule].append(v)
            for tag,v in tags.items():panels[tag].append(v)
            counts.update(ns);points+=len(row['values'])
        if dict(counts)!=c['rule_points'] or set(partial)!={'low','high'}:
            raise ValueError('incomplete quadrature rule')
        low,high=(f.combine(partial[k]) for k in ('low','high'))
        cell_rows.append(dict(cell=c['cell'],low=low.record(),high=high.record(),
            indicator=abs(high.value-low.value),signed_change=(high.value-low.value).real,
            by_panel={r:{tag:math.fsum(v) for (rule,tag),v in panels.items() if rule==r} for r in ('low','high')}))
    if seen!=set(range(len(d['batches']))) or points!=d['summary']['points']:
        raise ValueError('incomplete production coverage')
    components=dict(imports['components'])
    for term in sorted({c['cell']['term'] for c in cell_rows}):
        rows=[c for c in cell_rows if c['cell']['term']==term]
        high=f.combine(f.Value.from_record(c['high']) for c in rows)
        low=f.combine(f.Value.from_record(c['low']) for c in rows)
        indicator=math.fsum(c['indicator'] for c in rows);tol=d['component_tolerances'][term]
        components[term]=dict(value=high.record(),low=low.record(),quadrature_indicator=indicator,
            tolerance=tol,status='quadrature_indicator_met' if indicator<=tol else 'planned_comparison_limit',
            cell_count=len(rows),imported=False)
    if set(components)!=set(f.boundary.term_names('analytic')):raise ValueError('missing amplitude component')
    total=f.combine(f.Value.from_record(v['value']) for v in components.values())
    indicator=math.fsum(v['quadrature_indicator'] for v in components.values())
    amplitude=total.value.real+imports['compact']['mean']
    timings={}
    phases=d.get('timing_phases',['prepare','contract'])
    if phases not in (['prepare','contract'],['contract']):raise ValueError('invalid timing phases')
    if d['fresh_banks'] and 'prepare' not in phases:raise ValueError('missing bank preparation accounting')
    for phase in phases:
        ts=[f.spectrum.read(root/f'timings/{phase}_{i:03d}.json') for i in range(d['workers'])]
        if any(t['design_sha256']!=identity for t in ts):raise ValueError('foreign timing')
        timings[phase]=dict(cpu_seconds=math.fsum(t['cpu_seconds'] for t in ts),
                           max_wall_seconds=max(t['wall_seconds'] for t in ts))
    r=dict(format=d['format'],design_sha256=identity,source_plan_sha256=d['source_plan_sha256'],
        amplitude=amplitude,previous_amplitude=original['amplitude'],amplitude_change=amplitude-original['amplitude'],
        complete_domain=True,components=components,cells=cell_rows,boundary=total.record(),compact=imports['compact'],
        moduli_indicator=indicator,relative_moduli_indicator=indicator/abs(amplitude),
        boundary_indicator_budget=d['boundary_indicator_budget'],
        boundary_indicator_met=indicator<=d['boundary_indicator_budget'],
        imported_terms=sorted(imports['components']),compact_imported=True,summary=d['summary'],timings=timings,
        unknown_errors=imports['unknown_errors'],full_amplitude_certified=False,indicator_is_error_bound=False,
        automatic_refinement=False,created_utc=f.shared.utc())
    if (root/'amplitude.json').exists():
        # Recomputing the reduction must compare equal without changing the
        # original completion time. publish still rejects every other change.
        existing=f.spectrum.read(root/'amplitude.json')
        r['created_utc']=existing['created_utc']
    f.spectrum.publish(root/'amplitude.json',r)
    print(json.dumps({k:v for k,v in r.items() if k not in ('cells','components')},indent=2),flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('action',choices=('prepare','worker','reduce'))
    a.add_argument('--root',type=Path,required=True);a.add_argument('--source',type=Path);a.add_argument('--source-remote')
    a.add_argument('--parent',type=Path,action='append',default=[]);a.add_argument('--parent-remote',action='append',default=[])
    a.add_argument('--choices',type=Path);a.add_argument('--workers',type=int,default=384)
    a.add_argument('--batch-size',type=int,default=256);a.add_argument('--phase',choices=('prepare','contract'))
    args=a.parse_args()
    if args.action=='prepare':prepare(args)
    elif args.action=='worker':campaign.worker(args.root,args.phase)
    else:reduce(args.root)
