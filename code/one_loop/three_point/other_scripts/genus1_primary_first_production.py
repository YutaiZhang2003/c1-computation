"""First retained amplitude with tested primary angular panels and exact reuse.

Preserve the frozen short-length and momentum nodes throughout all primary
sectors. Import the unchanged collision and compact components explicitly.
No coefficient preparation or automatic refinement is permitted.
"""

from genus1_source_paths import ROOT as _SOURCE_ROOT, source_path as _source_path
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import shutil
import sys

sys.dont_write_bytecode=True
import genus1_moduli_amplitude as production
campaign=production.campaign


def source_length_slices(f, physics, cell, order):
    """Match the original cell_nodes floating-point arithmetic exactly."""
    sector=tuple(map(int,cell['term'].split(':')[1]))
    ordering,band=(int(v[-1]) for v in cell['key'].split(':')[-2:])
    nodes,weights=f.gauss_rule(order)
    for i,nm in enumerate(nodes):
        for j,nr in enumerate(nodes):
            u=[(a+b)/2+(b-a)/2*x for (a,b),x in zip(cell['bounds'][:2],(nm,nr))]
            short,jac=f.boundary.short_map([*u,.5,.5],sector,ordering,band,physics.policy)
            weight=jac/(2*math.pi)**2*math.prod((b-a)/2*w
                for (a,b),w in zip(cell['bounds'][:2],(weights[i],weights[j])))
            yield short,weight


def verified_reuse(f, regional, p):
    d=f.spectrum.read(regional/'design.json');identity=f.fingerprint(d)
    if d['source_plan_sha256']!=f.fingerprint(p):raise ValueError('foreign local trial')
    for rel,sha in d['sources'].items():
        if f.shared.sha(regional/rel)!=sha:raise ValueError('changed local trial source')
    for rel,key in (('points.json','points_sha256'),('bank_requests.json','banks_sha256')):
        if f.shared.sha(regional/rel)!=d[key]:raise ValueError('changed local trial inputs')
    q=f.spectrum.read(regional/'points.json');values={}
    for i,keys in enumerate(q['batches']):
        path=regional/f'results/batch_{i:04d}.json';row=f.spectrum.read(path)
        if row['design_sha256']!=identity or set(row['values'])!=set(keys):
            raise ValueError('incomplete local trial batch')
        for key,v in row['values'].items():
            sample=dict(kind='primary',short=q['points'][key]['short']);new=f.fingerprint(sample)
            if new in values:raise ValueError('duplicate local trial point')
            values[new]=dict(value=v,grid=q['points'][key]['grid'],
                             source_batch=path.name,source_batch_sha256=f.shared.sha(path))
    if len(values)!=len(q['points']):raise ValueError('missing local trial point')
    result=f.spectrum.read(regional/'result.json')
    if result['design_sha256']!=identity:raise ValueError('foreign local result')
    return values,dict(design=d,result=result,design_sha256=identity,
        result_file_sha256=f.shared.sha(regional/'result.json'),
        saved_reference_comparison=json.loads((regional/'saved_reference_comparison.json').read_text()),
        accepted_limitation='The user accepts the observed 3.3% selected-cell difference for this first production; this is not a global error bound.')


def prepare(args):
    root=args.root;root.mkdir(parents=True,exist_ok=False);(root/'pilot').mkdir()
    for path in (Path(__file__),Path(production.__file__),Path(campaign.__file__),
                 _source_path('code/one_loop/three_point/moduli_integration/genus1_moduli_panels.py')):
        shutil.copy2(path,root/'pilot'/path.name)
    (root/'locations.json').write_text(json.dumps(dict(source=str(args.source.resolve()),
        source_remote=args.source_remote,source_plan_sha256=json.loads((args.source/'plan.json').read_text())['sha256'],
        parents=[]),indent=2)+'\n')
    p=campaign.configure(root);f=campaign.f;g=campaign.g;physics=f.make_plan(p['config'])
    original=production.source_result(campaign.source,p)
    request=f.spectrum.read(campaign.source/'waves/000_request.json')
    if request['plan_sha256']!=f.fingerprint(p):raise ValueError('foreign source partition')
    production.validate_cells(request['cells'],require_pair=True)
    available,readiness=verified_reuse(f,args.regional,p)
    f.spectrum.publish(root/'readiness.json',readiness)
    imported={k:v for k,v in original['boundary']['components'].items() if not k.startswith('primary:')}
    if len(imported)!=8:raise ValueError('all unchanged collision terms required')
    f.spectrum.publish(root/'imports.json',dict(source_plan_sha256=f.fingerprint(p),
        amplitude_file_sha256=f.shared.sha(campaign.source/'amplitude_wave_000.json'),
        compact=original['compact'],components=imported,unknown_errors=original['unknown_errors']))
    builder=production.Builder(root,p,f.make_atlas(physics.t),64)
    for cell in request['cells']:
        if not cell['term'].startswith('primary:'):continue
        for label,n in (('low',2),('high',3)):
            for short,lw in source_length_slices(f,physics,cell,n):
                for x,y,aw in g.angular_head_rule(short.s1,short.s2,physics.policy.pair_radius,n,
                        x1_bounds=tuple(2*math.pi*u for u in cell['bounds'][2]),
                        x2_bounds=tuple(2*math.pi*u for u in cell['bounds'][3]),tail_order=2):
                    point=f.ShortPoint(short.sector,short.s1,short.s2,x,y)
                    if not f.regular_mask(point,physics.policy.pair_radius,f.MIN_HEIGHT):
                        raise ValueError('outside primary ownership')
                    builder.add(label,dict(kind='primary',short=point.record()),lw*aw)
        builder.finish_cell(cell)
    if set(builder.grids)!=set(builder.routing):raise ValueError('no new banks authorized in this production')
    imported_batches={};reused_points=0
    for i,b in enumerate(builder.batches):
        samples=f.spectrum.read(root/b['path'])['samples'];present=set(samples)&set(available)
        if not present:continue
        if present!=set(samples):raise ValueError('partial reusable batch needs explicit routing')
        if any(samples[k]['grids']!=[available[k]['grid']] for k in samples):
            raise ValueError('changed momentum grid on reused point')
        imported_batches[i]={k:available[k] for k in samples};reused_points+=len(samples)
    if reused_points!=len(available):raise ValueError('local trial not fully retained')
    pending=[i for i in range(len(builder.batches)) if i not in imported_batches]
    allocation,loads=campaign.assign([builder.costs[i] for i in pending],args.workers)
    contracts=[[pending[i] for i in row] for row in allocation]
    points=sum(b['points'] for b in builder.batches)
    if points>100000 or len(builder.grids)>400:raise ValueError('first-production work cap exceeded')
    d=dict(format='genus1-streamed-moduli-amplitude-v1',mode='primary-first',created_utc=f.shared.utc(),
        source_plan_sha256=f.fingerprint(p),source_domain_sha256=f.fingerprint(request),parent_identities=[],
        pilot_sources={str(q.relative_to(root)):f.shared.sha(q) for q in sorted((root/'pilot').glob('*.py'))},
        physics=p['physics'],momentum_mode=p['momentum_mode'],elliptic_orders=p['elliptic_orders'],
        imports_sha256=f.shared.sha(root/'imports.json'),readiness_sha256=f.shared.sha(root/'readiness.json'),
        choices=dict(length_nodes='exact frozen source G2/G3',angular_heads=[2,3],angular_tail=2,
            collision_clipped_tail='retain head order',update_primary=True,update_pair=False,
            momentum_nodes='unchanged eight/nine totals'),
        cells=builder.cells,batches=builder.batches,fresh_banks=[],routing=builder.routing,
        workers=args.workers,prepare=[[] for _ in range(args.workers)],contract=contracts,timing_phases=['contract'],
        imported_batch_indices=sorted(imported_batches),
        boundary_indicator_budget=p['config']['boundary_controls']['total_absolute_tolerance'],
        component_tolerances={k:v['absolute_tolerance'] for k,v in p['quadrature']['controls'].items()},
        summary=dict(points=points,new_evaluations=points-reused_points,reused_point_values=reused_points,
            new_banks=0,reused_banks=len(builder.routing),batches=len(builder.batches),
            new_batches=len(pending),imported_batches=len(imported_batches),cells=len(builder.cells),
            banks_by_momentum_axes=dict(Counter(str(tuple(v['axes'])) for v in builder.grids.values())),
            estimated_contract_max_seconds=max(loads),active_workers=sum(bool(v) for v in contracts)),
        full_amplitude_certified=False,physical_cutoffs_changed=False,momentum_design_changed=False,
        automatic_refinement=False,maximum_planned_comparisons=1)
    if sorted(i for rows in contracts for i in rows)+[]!=sorted(pending):
        raise ValueError('incomplete worker coverage')
    f.spectrum.publish(root/'design.json',d);identity=f.fingerprint(d)
    for i,rows in imported_batches.items():
        f.spectrum.publish(root/f'results/batch_{i:05d}.json',dict(design_sha256=identity,
            values={k:r['value'] for k,r in rows.items()},imported=True,
            origin_design_sha256=readiness['design_sha256'],
            source_batches={r['source_batch']:r['source_batch_sha256'] for r in rows.values()}))
    print(json.dumps(d['summary'],indent=2),flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('action',choices=('prepare','worker','reduce'))
    a.add_argument('--root',type=Path,required=True);a.add_argument('--source',type=Path);a.add_argument('--source-remote')
    a.add_argument('--regional',type=Path);a.add_argument('--workers',type=int,default=384)
    args=a.parse_args()
    if args.action=='prepare':prepare(args)
    elif args.action=='worker':campaign.worker(args.root,'contract')
    else:production.reduce(args.root)
