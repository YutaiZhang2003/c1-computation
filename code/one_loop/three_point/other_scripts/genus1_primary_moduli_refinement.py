"""Independent physical-length and angular checks in selected primary cells.

The frozen CFT runtime, energy-specific momentum rules, other cells and compact
observations are retained. New results live in a separate experiment. Candidate
selection and diagnostics never use the matrix-model prediction.
"""

from genus1_source_paths import ROOT as _SOURCE_ROOT, source_path as _source_path
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys

sys.dont_write_bytecode = True
import genus1_moduli_amplitude as production
campaign = production.campaign
RULES = dict(candidate=(3,3,3), maximum=(4,3,3), ratio=(3,4,3), angles=(3,3,4))
ENERGIES = {'t0250': .25, 't0950': .95}


def selected(cell):
    return (cell['term'].startswith('primary:') and cell['key'].endswith(':band1')
            and cell['bounds'] == [[0.,1.],[0.,.5],[0.,1.],[0.,1.]])


def close(a, b):
    return math.isclose(a, b, rel_tol=2e-12, abs_tol=2e-18)


def replace_primary(baseline, replacements, Value, combine):
    """Reconstruct complete primary components, retaining all other terms."""
    old = baseline['amplitude']
    cells = baseline['primary_cells']
    targets = {c['cell']['id'] for c in cells if selected(c['cell'])}
    if len(cells) != 48 or len(targets) != 12 or set(replacements) != targets:
        raise ValueError('exactly the twelve selected primary cells are required')
    if len({c['cell']['id'] for c in cells}) != len(cells):
        raise ValueError('duplicate source cell')
    components = deepcopy(old['components'])
    for term in sorted({c['cell']['term'] for c in cells}):
        rows = [c for c in cells if c['cell']['term'] == term]
        previous = combine(Value.from_record(c['high']) for c in rows)
        component = components[term]
        if (not close(previous.value.real, component['value']['value'][0]) or
                not close(previous.errors['block'], component['value']['errors']['block']) or
                not close(math.fsum(c['indicator'] for c in rows), component['quadrature_indicator'])):
            raise ValueError('source cells do not reconstruct the retained component')
        updated = [replacements.get(c['cell']['id'], c) for c in rows]
        component.update(value=combine(Value.from_record(c['high']) for c in updated).record(),
            low=None, quadrature_indicator=math.fsum(c['indicator'] for c in updated),
            quadrature_indicator_method='retained cells plus absolute independent primary directional changes',
            replaced_cell_ids=[c['cell']['id'] for c in rows if c['cell']['id'] in replacements])
    boundary = combine(Value.from_record(c['value']) for c in components.values())
    indicator = math.fsum(c['quadrature_indicator'] for c in components.values())
    amplitude = boundary.value.real + old['compact']['mean']
    return dict(t=baseline['t'], amplitude=amplitude, previous_amplitude=old['amplitude'],
        amplitude_change=amplitude-old['amplitude'], components=components,
        boundary=boundary.record(), compact=deepcopy(old['compact']),
        compact_replicate_means=deepcopy(old.get('compact_replicate_means')),
        moduli_indicator=indicator, relative_moduli_indicator=indicator/abs(amplitude),
        previous_moduli_indicator=old['moduli_indicator'],
        primary_cells=[replacements.get(c['cell']['id'], c) for c in cells],
        complete_domain=True, full_amplitude_certified=False, indicator_is_error_bound=False,
        unknown_errors=deepcopy(old['unknown_errors']), momentum_design_changed=False,
        physical_cutoffs_changed=False, compact_recomputed=False,
        replaced_cell_ids=sorted(replacements), unchanged_other_cells=True)


def configure(root, label=None):
    global f, g, base, locations
    locations = json.loads((root/'locations.json').read_text())
    remote = bool(os.environ.get('SLURM_JOB_ID')) or locations.get('remote_only', False)
    key = 'remote' if remote else 'local'
    source = Path(locations['runtime'][key])
    sys.path.insert(0, str(source/'runtime/source/code/tasks'))
    sys.path.insert(0, str(source/'runtime/source/code/one_loop/three_point/other_scripts'))
    import genus1_fresh_analytic
    f = genus1_fresh_analytic
    base = f.load(source)
    if f.fingerprint(base) != locations['source_plan_sha256']:
        raise ValueError('changed frozen CFT runtime')
    spec = importlib.util.spec_from_file_location('targeted_primary_geometry', root/'pilot/genus1_moduli_panels.py')
    g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
    campaign.f = f
    if label is not None:
        sources = [Path(r[key]) for r in locations['banks'][label]]
        # Local evidence may be stored in an extracted subdirectory.
        if not remote:
            sources = [p/'evidence' if (p/'evidence/prepared/spectrum/grids').exists() else p for p in sources]
        campaign.source, campaign.parents = sources[0], sources[1:]
    return base


def plan(t):
    p = deepcopy(base)
    p['config']['run']['t'] = t
    p['physics'] = f.make_plan(p['config']).record()
    return p


def baseline_import(root, curve_root, primary_root, label):
    curve = f.spectrum.read(curve_root/'curve.json')
    if f.fingerprint(curve) != locations['baseline_curve_sha256']:
        raise ValueError('changed baseline curve')
    t = ENERGIES[label]
    if t == .25:
        a = f.spectrum.read(curve_root/'anchor.json')['amplitude']
        before = f.spectrum.read(primary_root/'amplitude.json')
        cells = before['cells']
        pins = {str(curve_root/'anchor.json'):f.shared.sha(curve_root/'anchor.json'),
                str(primary_root/'amplitude.json'):f.shared.sha(primary_root/'amplitude.json')}
    else:
        a = f.spectrum.read(curve_root/f'energies/{label}/amplitude.json')
        receipt = f.spectrum.read(curve_root/f'sequential/completed/{label}.json')
        ed = next(e for e in curve['energies'] if e['label'] == label)
        if (a['curve_sha256'] != f.fingerprint(curve) or a['t'] != t or
                a['energy_design_sha256'] != ed['design_sha256'] or
                receipt['curve_sha256'] != f.fingerprint(curve) or receipt['energy'] != label or
                receipt['amplitude_sha256'] != f.shared.sha(curve_root/f'energies/{label}/amplitude.json')):
            raise ValueError('foreign baseline energy')
        cells = [c for c in a['cells'] if c['cell']['term'].startswith('primary:')]
        pins = {str(curve_root/f'energies/{label}/amplitude.json'):
                f.shared.sha(curve_root/f'energies/{label}/amplitude.json')}
    if not a['complete_domain'] or a['compact']['plan']['t'] != t:
        raise ValueError('complete baseline at the requested energy required')
    result = dict(t=t, amplitude=a, primary_cells=cells, source_files=pins,
                  baseline_curve_sha256=f.fingerprint(curve))
    # An identity replacement independently verifies every retained primary sum.
    same = {c['cell']['id']:c for c in cells if selected(c['cell'])}
    replay = replace_primary(result, same, f.Value, f.combine)
    if not close(replay['amplitude'], a['amplitude']) or not close(replay['moduli_indicator'], a['moduli_indicator']):
        raise ValueError('baseline replay failed')
    f.spectrum.publish(root/'baseline.json', result)
    return result


def prepare(args):
    root = args.root.resolve(); root.mkdir(parents=True, exist_ok=False); (root/'pilot').mkdir()
    for path in (Path(__file__),Path(production.__file__),Path(campaign.__file__),
                 _source_path('code/one_loop/three_point/moduli_integration/genus1_moduli_panels.py')):
        shutil.copy2(path, root/'pilot'/path.name)
    oldloc = json.loads((args.curve/'locations.json').read_text())
    loc = dict(runtime=dict(local=oldloc['source'],remote=oldloc['source_remote']),
        source_plan_sha256=oldloc['source_plan_sha256'],
        baseline_curve_sha256=json.loads((args.curve/'curve.json').read_text())['sha256'],
        banks={'t0250':[dict(local=oldloc['source'],remote=oldloc['source_remote'])],
               't0950':[dict(local=str(Path(oldloc['previous_curve'])/'energies/t095'),
                              remote=oldloc['previous_curve_remote']+'/energies/t095')]})
    (root/'locations.json').write_text(json.dumps(loc,indent=2)+'\n')
    summaries = {}
    for label,t in ENERGIES.items():
        configure(root,label); p=plan(t); eroot=root/'energies'/label
        imported=baseline_import(eroot,args.curve,Path(oldloc['anchor']),label)
        f.spectrum.publish(eroot/'plan.json',p)
        physics=f.make_plan(p['config'])
        builder=production.Builder(eroot,p,f.make_atlas(t),64)
        target=[c['cell'] for c in imported['primary_cells'] if selected(c['cell'])]
        for cell in target:
            for name,(nm,nr,na) in RULES.items():
                for short,w in g.primary_cell_rule(physics,cell,(nm,nr),na,
                        angular_heads=True,angular_tail_order=2):
                    builder.add(name,dict(kind='primary',short=short.record()),w)
            builder.finish_cell(cell)
        fresh=sorted(set(builder.grids)-set(builder.routing))
        costs=[100.*math.prod(builder.grids[k]['axes'])/512 for k in fresh]
        preparations,loads=campaign.assign(costs,args.workers)
        contracts,cloads=production.assign_geometry_groups(builder.costs,
            [b['coordinate_group'] for b in builder.batches],args.workers)
        d=dict(format='genus1-targeted-primary-moduli-v1',t=t,label=label,
            source_plan_sha256=f.fingerprint(base),plan_sha256=f.fingerprint(p),
            baseline_sha256=f.shared.sha(eroot/'baseline.json'),
            pilot_sources={str(q.relative_to(root)):f.shared.sha(q) for q in sorted((root/'pilot').glob('*.py'))},
            rules={k:list(v) for k,v in RULES.items()},cells=builder.cells,batches=builder.batches,
            fresh_banks=fresh,routing=builder.routing,workers=args.workers,
            prepare=preparations,contract=contracts,
            spec_sha256={k:f.shared.sha(f.spectrum.grid_directory(eroot,k)/'spec.json') for k in fresh},
            summary=dict(points=sum(b['points'] for b in builder.batches),fresh_banks=len(fresh),
                reused_banks=len(builder.routing),batches=len(builder.batches),cells=len(target),
                rule_points=dict(sum((Counter(c['rule_points']) for c in builder.cells),Counter())),
                banks_by_momentum_axes=dict(Counter(str(tuple(v['axes'])) for v in builder.grids.values())),
                estimated_prepare_max_seconds=max(loads),estimated_contract_max_seconds=max(cloads)),
            physical_panels='unchanged cell support; physical collision and modular transition splits',
            angular_tail_order=2,momentum_design_changed=False,physical_cutoffs_changed=False,
            diagnostic_flags_halt=False,indicator_is_error_bound=False)
        f.spectrum.publish(eroot/'design.json',d)
        summaries[label]=dict(design_sha256=f.fingerprint(d),summary=d['summary'])
    f.spectrum.publish(root/'experiment.json',dict(energies=summaries,order=list(ENERGIES),
        source_plan_sha256=loc['source_plan_sha256'],baseline_curve_sha256=loc['baseline_curve_sha256'],
        locations_sha256=f.shared.sha(root/'locations.json'),workers=args.workers,created_utc=f.shared.utc()))
    print(json.dumps(summaries,indent=2),flush=True)


def checked(root,label):
    configure(root,label); eroot=root/'energies'/label
    e=f.spectrum.read(root/'experiment.json');d=f.spectrum.read(eroot/'design.json')
    p=f.spectrum.read(eroot/'plan.json')
    if (f.shared.sha(root/'locations.json')!=e['locations_sha256'] or
            f.fingerprint(d)!=e['energies'][label]['design_sha256'] or
            f.fingerprint(p)!=d['plan_sha256'] or p!=plan(ENERGIES[label]) or
            f.shared.sha(eroot/'baseline.json')!=d['baseline_sha256'] or
            d['rules']!={k:list(v) for k,v in RULES.items()}):
        raise ValueError('changed experiment input')
    for rel,sha in d['pilot_sources'].items():
        if f.shared.sha(root/rel)!=sha:raise ValueError('changed experimental source: '+rel)
    return eroot,d,p


def worker(root,label,phase):
    eroot,d,p=checked(root,label); identity=f.fingerprint(d)
    rank=int(os.environ['SLURM_PROCID'])
    if int(os.environ['SLURM_NTASKS'])!=d['workers'] or not 0<=rank<d['workers']:
        raise ValueError('changed worker allocation')
    if campaign.completed_worker(eroot,d,identity,phase,rank):return
    start=f.clock()
    if phase=='prepare':
        for i in d['prepare'][rank]:
            key=d['fresh_banks'][i]
            if f.shared.sha(f.spectrum.grid_directory(eroot,key)/'spec.json')!=d['spec_sha256'][key]:
                raise ValueError('changed momentum grid specification')
            while True:
                result=f.spectrum.build(eroot,p,key,cpu_budget=240,max_new_chunks=64,max_new_elliptic_nodes=1024)
                if result['complete_rule']:break
                if (result.get('failure') or {}).get('type')!='TimeLimit':raise RuntimeError(str(result))
    elif phase=='contract':
        from genus1_boundary_batch import GeometryStore
        physics=f.make_plan(p['config']);atlas=f.make_atlas(physics.t)
        geometry=GeometryStore(atlas,eroot/f'work/geometry/{rank:03d}')
        runtime=f.spectrum.Runtime(eroot,p,atlas,geometry,point_budget=1e-12)
        runtime.banks=campaign.Banks(eroot,d)
        try:
            for i in d['contract'][rank]:
                samples=campaign.batch(eroot,d,i);path=eroot/f'results/batch_{i:05d}.json'
                if path.exists():
                    row=f.spectrum.read(path)
                    if row['design_sha256']!=identity or set(row['values'])!=set(samples):
                        raise ValueError('foreign completed contraction')
                    continue
                tick=f.clock();keys=list(samples)
                shorts=[f.ShortPoint.from_record(samples[k]['short']) for k in keys]
                grids=[samples[k]['grids'][0] for k in keys]
                values=runtime.primary(shorts,grids,geometry.prepare(shorts),physics.policy,physics.angle_order)
                f.spectrum.publish(path,dict(design_sha256=identity,
                    values={k:f.boundary.amplitude_value(v,1.).record() for k,v in zip(keys,values)},**f.timing(tick)))
        finally:geometry.close()
    else:raise ValueError('unknown worker phase')
    f.spectrum.publish(eroot/f'timings/{phase}_{rank:03d}.json',dict(design_sha256=identity,
        phase=phase,rank=rank,job_id=os.environ['SLURM_JOB_ID'],**f.timing(start)))


def reduce(root,label):
    eroot,d,p=checked(root,label);identity=f.fingerprint(d);cells={};seen=set()
    for c in d['cells']:
        partial=defaultdict(list);counts=Counter()
        for i in c['batch_indices']:
            if i in seen:raise ValueError('duplicate contraction in reduction')
            seen.add(i);b=d['batches'][i]
            if f.shared.sha(eroot/b['path'])!=b['sha256']:raise ValueError('changed geometry or weights')
            geometry=f.spectrum.read(eroot/b['path']);result=f.spectrum.read(eroot/f'results/batch_{i:05d}.json')
            if (result['design_sha256']!=identity or set(result['values'])!=set(geometry['samples']) or
                    geometry['cell_id']!=c['cell']['id']):raise ValueError('foreign or incomplete contraction')
            sums,_,ns=production.batch_estimates(geometry,result['values'],f.Value,f.combine,allowed_rules=set(RULES))
            counts.update(ns)
            for name,value in sums.items():partial[name].append(value)
        if dict(counts)!=c['rule_points'] or set(partial)!=set(RULES):raise ValueError('missing quadrature rule')
        totals={name:f.combine(rows) for name,rows in partial.items()};high=totals['candidate']
        delta={name:(totals[name].value-high.value).real for name in RULES if name!='candidate'}
        cells[c['cell']['id']]=dict(cell=c['cell'],high=high.record(),
            indicator=math.fsum(abs(v) for v in delta.values()),directional_deltas=delta,
            rules={name:v.record() for name,v in totals.items()},rule_points=dict(counts))
    if seen!=set(range(len(d['batches']))):raise ValueError('incomplete regional coverage')
    for phase in ('prepare','contract'):
        for rank in range(d['workers']):
            row=f.spectrum.read(eroot/f'timings/{phase}_{rank:03d}.json')
            if row['design_sha256']!=identity or row['phase']!=phase or row['rank']!=rank:
                raise ValueError('foreign completed worker')
    baseline=f.spectrum.read(eroot/'baseline.json')
    result=replace_primary(baseline,cells,f.Value,f.combine)
    result.update(design_sha256=identity,baseline_sha256=d['baseline_sha256'],
        direction_indicators={name:math.fsum(abs(c['directional_deltas'][name]) for c in cells.values())
                              for name in RULES if name!='candidate'},
        previous_selected_indicator=math.fsum(c['indicator'] for c in baseline['primary_cells'] if selected(c['cell'])),
        new_selected_indicator=math.fsum(c['indicator'] for c in cells.values()))
    result['block_indicator']=result['boundary']['errors']['block']+result['compact']['errors']['block']
    result['compact_sampling_halfwidth95']=result['compact']['sampling_halfwidth95']
    result['partial_numerical_envelope']=math.fsum((result['moduli_indicator'],result['block_indicator'],
                                                   result['compact_sampling_halfwidth95']))
    result['partial_envelope_is_confidence_interval']=False
    f.spectrum.publish(eroot/'amplitude.json',result)
    f.spectrum.publish(root/f'completed/{label}.json',dict(t=ENERGIES[label],design_sha256=identity,
        amplitude_file_sha256=f.shared.sha(eroot/'amplitude.json'),amplitude=result['amplitude'],
        created_utc=f.shared.utc(),complete_domain=True))
    print(json.dumps({k:result[k] for k in ('t','amplitude','amplitude_change','moduli_indicator',
        'relative_moduli_indicator','previous_selected_indicator','new_selected_indicator','direction_indicators')},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','worker','reduce'))
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--curve',type=Path)
    parser.add_argument('--workers',type=int,default=384)
    parser.add_argument('--energy',choices=tuple(ENERGIES));parser.add_argument('--phase',choices=('prepare','contract'))
    args=parser.parse_args()
    if args.action=='prepare':prepare(args)
    elif args.action=='worker':worker(args.root,args.energy,args.phase)
    else:reduce(args.root,args.energy)
