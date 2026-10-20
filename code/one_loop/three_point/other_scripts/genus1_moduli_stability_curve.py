"""Repeat one frozen, targeted moduli stability study across saved energies.

The moduli nodes/weights are the verified t=.25 study's geometry. Each energy
uses its own pinned Liouville banks and base observations. No central amplitude
is changed, and no matrix prediction enters the quadrature or dispersion.
"""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys

sys.dont_write_bytecode = True
T_VALUES = [.025] + [i/20 for i in range(1, 20)]
SOURCE_PLAN = '4fe44ca0aeb6854200489322cce6d58068a9c7d7a53cd11b374989736a8bc675'
TEMPLATE = '2d071ad574c75c53f70c380880a3e61641b9d1b89432cc0b4139c22e6aeb680d'
CURVE20 = 'f94dfa8c5d444652e29450fe1e083f82c29b84dcac914aece30cd43fb879b082'
CURVE10 = '2da7197be1e36176b1e26a706883258a5a17269b3ac8026d908fc67dd266af39'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    obj = json.loads(Path(path).read_text())
    if set(obj) != {'payload', 'sha256'} or digest(obj['payload']) != obj['sha256']:
        raise ValueError('Invalid immutable record: '+str(path))
    return obj['payload']


def utc():
    return datetime.now(timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def label(t):
    return f't{round(1000*t):04d}'


def configure(root):
    global f, production, old, D, template, td, experiment, curve, curve_root
    locations = json.loads((root/'locations.json').read_text())
    remote = str(root).startswith(locations['remote_diagnostics'])
    D = Path(locations['remote_diagnostics' if remote else 'local_diagnostics'])
    experiment = read(root/'experiment.json')
    for rel, pin in experiment['implementation_sha256'].items():
        require(sha(root/rel) == pin, 'Changed study implementation: '+rel)
    runtime = D/locations['runtime']
    sys.path.insert(0, str(runtime/'runtime/source/code/tasks'))
    import genus1_fresh_analytic
    f = genus1_fresh_analytic
    base = f.load(runtime)
    require(f.fingerprint(base) == SOURCE_PLAN, 'Changed frozen CFT runtime')
    sys.path.insert(0, str(root/'pilot'))
    import genus1_moduli_amplitude
    import genus1_moduli_stability
    production, old = genus1_moduli_amplitude, genus1_moduli_stability
    production.campaign.f = f
    template = D/locations['template']
    td = read(template/'design.json')
    require(digest(td) == TEMPLATE, 'Changed quadrature template')
    curve_root = D/locations['curve20']
    curve = read(curve_root/'curve.json')
    require(digest(curve) == CURVE20, 'Changed production curve')
    require(experiment['t_values'] == T_VALUES and experiment['workers'] == 384,
            'Changed energy/allocation request')
    return base


def template_batch(index):
    descriptor = td['batches'][index]
    path = template/descriptor['path']
    require(sha(path) == descriptor['sha256'], 'Changed moduli nodes/weights')
    row = read(path)
    require(len(row['samples']) == descriptor['points'] and
            digest(sorted(row['samples'])) == descriptor['point_ids_sha256'], 'Incomplete geometry')
    return row


def same_energy_plan(base, t):
    p = deepcopy(base)
    p['config']['run']['t'] = t
    p['physics'] = f.make_plan(p['config']).record()
    return p


def source_dataset(t):
    eroot = curve_root/'energies'/label(t)
    ed = read(eroot/'design.json')
    if ed['retained_sha256']:
        parent = D/'genus1_curve10_primary_384_20260923'
        identity = CURVE10
        source_energy = parent/'energies'/f't{round(t*100):03d}'
    else:
        parent, identity, source_energy = curve_root, CURVE20, eroot
    pd = read(parent/'curve.json')
    require(digest(pd) == identity, 'Changed point source')
    sd = read(source_energy/'design.json')
    require(sd['t'] == t, 'Foreign source energy')
    return parent, identity, source_energy, sd, read(parent/'mesh.json')


def load_baseline_observations(t):
    parent, identity, eroot, ed, mesh = source_dataset(t)
    tested_ids = {c['cell']['id'] for c in td['cells']}
    cache, physical_weights, primary_grids = {}, {}, {}
    pins = {str((eroot/'design.json').relative_to(D)): sha(eroot/'design.json')}
    for i, b in enumerate(mesh['batches']):
        if b['cell_id'] not in tested_ids:
            continue
        gp = parent/'mesh'/b['path']
        require(sha(gp) == b['sha256'], 'Changed saved baseline geometry')
        geometry = read(gp)
        rp = eroot/f'results/batch_{i:05d}.json'
        row = read(rp)
        require(row['curve_sha256'] == identity and row['energy_design_sha256'] == digest(ed)
                and set(row['values']) == set(geometry['samples']), 'Foreign/incomplete baseline observations')
        pins[str(rp.relative_to(D))] = sha(rp)
        for key, s in geometry['samples'].items():
            if key in cache:
                require(cache[key] == row['values'][key], 'Conflicting same-energy observation')
            cache[key] = row['values'][key]
            weight = math.fsum(w for rule, w, tag in geometry['weights'][key] if rule == 'high')
            physical_weights[(geometry['cell_id'], key)] = weight
            if s['kind'] == 'primary':
                sp = s['short']; coordinate = (tuple(sp['sector']), sp['s1'], sp['s2'])
                actual = ed['grid_map'][s['grids'][0]]
                require(coordinate not in primary_grids or primary_grids[coordinate] == actual,
                        'Ambiguous saved primary momentum grid')
                primary_grids[coordinate] = actual
    return cache, physical_weights, primary_grids, eroot, ed, pins


def prepare_energy(root, t):
    base = configure(root)
    require(t in T_VALUES and t != .25, 'Energy is not an unestimated curve point')
    eroot = root/'energies'/label(t)
    if (eroot/'design.json').exists():
        ed = read(eroot/'design.json')
        checked_energy(root, t)
        print(json.dumps(dict(t=t, reused_design=True, **ed['summary'])), flush=True)
        return
    eroot.mkdir(parents=True, exist_ok=True)
    p = read(curve_root/'energies'/label(t)/'plan.json')
    require(p == same_energy_plan(base, t), 'Physics changed beyond energy')
    f.spectrum.publish(eroot/'plan.json', p)
    reference_path = curve_root/'energies'/label(t)/'amplitude.json'
    reference = read(reference_path)
    pin = next(r for r in experiment['references'] if r['t'] == t)
    require(sha(reference_path) == pin['file_sha256'] and reference['complete_domain']
            and reference['t'] == t, 'Changed/incomplete reference amplitude')
    cache, original_weights, primary_grids, source_eroot, source_ed, source_pins = load_baseline_observations(t)
    require(source_ed['plan_sha256'] == digest(p), 'Foreign same-energy bank plan')
    planner = f.spectrum.Planner(p, f.make_atlas(t))
    mapping, routing, specs, primary_cache, used_cache = {}, {}, {}, {}, {}
    source_roots = [source_eroot, curve_root/'energies'/label(t)]
    counts, base_points, geometry_identity = Counter(), 0, []
    batch_new = []
    grid_records = {}

    def register(s, previous_grid, request=None):
        primary = s['kind'] == 'primary'
        if primary:
            sp = s['short']; coordinate = (tuple(sp['sector']), sp['s1'], sp['s2'])
            key = primary_grids[coordinate]
            if key not in specs:
                path = f.spectrum.grid_directory(source_eroot, key)/'spec.json'
                require(sha(path) == source_ed['spec_sha256'][key], 'Changed saved primary nodes')
                specs[key] = read(path)
        else:
            cache_key = digest(request)
            if cache_key not in primary_cache:
                value = f.request_from_record(request)
                key, spec = planner.grid(value, False, p['profile'])
                primary_cache[cache_key] = key
                specs.setdefault(key, spec)
            key = primary_cache[cache_key]
        require(previous_grid not in mapping or mapping[previous_grid] == key, 'Template alias split')
        mapping[previous_grid] = key
        return key

    for i, b in enumerate(td['batches']):
        geo = template_batch(i)
        nnew = 0
        for key, s in geo['samples'].items():
            raw = {k:s[k] for k in (('kind','short') if s['kind']=='primary' else ('kind','requests'))}
            require(digest(raw) == key, 'Changed physical sample identity')
            for rule, w, tag in geo['weights'][key]:
                if rule == 'base':
                    require(key in cache and math.isclose(w, original_weights.get((geo['cell_id'], key), -1),
                            rel_tol=2e-13, abs_tol=1e-18), 'Baseline quadrature does not match source')
                    base_points += 1
            if key in cache:
                used_cache[key] = cache[key]
            else:
                nnew += 1
            # Even imported points retain exact per-energy grid identity. Only
            # non-imported observations need prepared banks at execution time.
            for j, oldkey in enumerate(s['grids']):
                actual = register(s, oldkey, None if s['kind']=='primary' else s['requests'][j])
                if key not in cache:
                    grid_records[actual] = specs[actual]
            counts['primary' if s['kind']=='primary' else 'tail_pair'] += int(key not in cache)
        batch_new.append(nnew)
        geometry_identity.append(b['sha256'])
    fresh = []
    for key, spec in grid_records.items():
        require(digest(spec['grid']) == key, 'Invalid spectral identity')
        f.check_momentum_grid(p, spec)
        require(spec['grid']['t'] == t, 'Cross-energy bank')
        axes = [len(rule['nodes']) for rule in spec['grid']['rules']]
        require(all(n in (8,9) for n in axes), 'Changed momentum-node budget')
        for source in source_roots:
            folder = f.spectrum.grid_directory(source,key)
            if (folder/'ready.json').exists():
                ready = read(folder/'ready.json')
                require(ready['grid_sha256']==key and ready['spec_sha256']==sha(folder/'spec.json'),
                        'Foreign completed bank receipt')
                # The canonical grid contains energy, axes, nodes, weights,
                # native and elliptic orders. Descriptive example geometry
                # can differ while the bank is exactly the same.
                require(read(folder/'spec.json')['grid'] == spec['grid'], 'Saved bank differs from planned energy/grid')
                routing[key] = dict(source=str(source.relative_to(D)), ready_sha256=sha(folder/'ready.json'))
                break
        if key not in routing:
            f.spectrum.publish(f.spectrum.grid_directory(eroot,key)/'spec.json', spec)
            fresh.append(key)
    fresh.sort()
    reference_cells = {c['cell']['id']:c for c in reference['cells']}
    require({c['cell']['id'] for c in td['cells']} <= set(reference_cells), 'Incomplete retained domain')
    f.spectrum.publish(eroot/'baseline.json', dict(reference=reference,
        cells=[reference_cells[c['cell']['id']] for c in td['cells']], source_files=source_pins))
    f.spectrum.publish(eroot/'reused_values.json', used_cache)
    costs = [20.3*math.prod(len(r['nodes']) for r in specs[k]['grid']['rules'])/512 for k in fresh]
    preparations, ploads = production.campaign.assign(costs,384)
    ccosts = [n*(.055273 if td['cells'][next(j for j,c in enumerate(td['cells']) if i in c['batch_indices'])]['cell']['term'].startswith('primary:') else .08)
              for i,n in enumerate(batch_new)]
    contracts, cloads = production.assign_geometry_groups(ccosts,[b['coordinate_group'] for b in td['batches']],384)
    for assignments, total in ((preparations,len(fresh)),(contracts,len(td['batches']))):
        require(sorted(i for worker in assignments for i in worker)==list(range(total)), 'Invalid worker coverage')
    summary = dict(points=td['summary']['points'],new_points=dict(counts),reused_points=len(used_cache),
        fresh_banks=len(fresh),reused_banks=len(routing),batches=len(td['batches']),cells=len(td['cells']),
        base_points_verified=base_points,estimated_useful_cpu_seconds=math.fsum(costs)+math.fsum(ccosts),
        estimated_prepare_max_seconds=max(ploads,default=0),estimated_contract_max_seconds=max(cloads,default=0))
    design = dict(format='genus1-targeted-stability-energy-v1',t=t,workers=384,experiment_sha256=digest(experiment),
        template_sha256=TEMPLATE,plan_sha256=digest(p),reference_file_sha256=sha(reference_path),
        mapping=mapping,routing=routing,fresh_banks=fresh,
        spec_sha256={k:sha(f.spectrum.grid_directory(eroot,k)/'spec.json') for k in fresh},
        inputs={n:sha(eroot/n) for n in ('plan.json','baseline.json','reused_values.json')},
        prepare=preparations,contract=contracts,batch_new_points=batch_new,summary=summary,
        momentum_axes=dict(Counter(str(tuple(len(r['nodes']) for r in s['grid']['rules'])) for s in grid_records.values())))
    f.spectrum.publish(eroot/'design.json',design)
    print(json.dumps(dict(t=t,design_sha256=digest(design),**summary)),flush=True)


def checked_energy(root,t):
    eroot=root/'energies'/label(t); ed=read(eroot/'design.json'); p=read(eroot/'plan.json')
    require(ed['experiment_sha256']==digest(experiment) and ed['template_sha256']==TEMPLATE
            and ed['t']==t and ed['plan_sha256']==digest(p) and p['physics']['t']==t,
            'Changed energy identity')
    for rel,pin in ed['inputs'].items():require(sha(eroot/rel)==pin,'Changed energy input: '+rel)
    return eroot,ed,p


class RoutedBanks:
    def __init__(self,eroot,ed):
        self.local=f.spectrum.Banks(eroot);self.routing=ed['routing'];self.cache={};self.verified=set()
    def get(self,key):
        if key not in self.routing:return self.local.get(key)
        r=self.routing[key];source=D/r['source']
        if key not in self.verified:
            require(sha(f.spectrum.grid_directory(source,key)/'ready.json')==r['ready_sha256'],'Changed reused bank')
            self.verified.add(key)
        if str(source) not in self.cache:self.cache[str(source)]=f.spectrum.Banks(source)
        return self.cache[str(source)].get(key)


def validate_batch_result(row,ed,geometry):
    require(row['design_sha256']==digest(ed) and row['t']==ed['t']
            and set(row['values'])==set(geometry['samples']),'Foreign/incomplete result batch')


def worker(root,t,phase,only_index=None):
    configure(root);eroot,ed,p=checked_energy(root,t)
    rank=int(os.environ.get('SLURM_PROCID','0'))
    require(only_index is not None or int(os.environ['SLURM_NTASKS'])==384,'Changed allocation')
    assigned=ed[phase][rank] if only_index is None else [only_index]
    receipt=eroot/f'timings/{phase}_{rank:03d}.json'
    if only_index is None and receipt.exists():
        row=read(receipt)
        require(row['design_sha256']==digest(ed) and row['phase']==phase and row['rank']==rank,'Foreign worker receipt')
        for i in assigned:
            if phase=='prepare':f.spectrum.ready(eroot,ed['fresh_banks'][i])
            else:validate_batch_result(read(eroot/f'results/batch_{i:05d}.json'),ed,template_batch(i))
        return
    start=f.clock()
    if phase=='prepare':
        for i in assigned:
            key=ed['fresh_banks'][i]
            require(sha(f.spectrum.grid_directory(eroot,key)/'spec.json')==ed['spec_sha256'][key],'Changed bank input')
            while True:
                row=f.spectrum.build(eroot,p,key,cpu_budget=240,max_new_chunks=64,max_new_elliptic_nodes=1024)
                if row['complete_rule']:break
                require((row.get('failure') or {}).get('type')=='TimeLimit','CFT bank failed: '+str(row))
    elif phase=='contract':
        from genus1_boundary_batch import GeometryStore
        physics=f.make_plan(p['config']);atlas=f.make_atlas(t)
        geometry=GeometryStore(atlas,eroot/f'work/geometry/{rank:03d}')
        runtime=f.spectrum.Runtime(eroot,p,atlas,geometry,point_budget=1e-12);runtime.banks=RoutedBanks(eroot,ed)
        reused=read(eroot/'reused_values.json')
        try:
            for i in assigned:
                geo=template_batch(i);samples=geo['samples'];path=eroot/f'results/batch_{i:05d}.json'
                if path.exists():validate_batch_result(read(path),ed,geo);continue
                tick=f.clock();values={k:reused[k] for k in samples if k in reused}
                keys=[k for k in samples if k not in reused]
                require(len(keys)==ed['batch_new_points'][i],'Changed exact point reuse')
                if keys and samples[keys[0]]['kind']=='primary':
                    shorts=[f.ShortPoint.from_record(samples[k]['short']) for k in keys]
                    grids=[ed['mapping'][samples[k]['grids'][0]] for k in keys]
                    vs=runtime.primary(shorts,grids,geometry.prepare(shorts),physics.policy,physics.angle_order)
                    values.update({k:f.boundary.amplitude_value(v,1.).record() for k,v in zip(keys,vs)})
                else:
                    for key in keys:
                        s=samples[key]
                        vs=[runtime.collide(f.request_from_record(a),ed['mapping'][grid],s['max_weight']).scaled(a['multiplicity'])
                            for a,grid in zip(s['requests'],s['grids'])]
                        values[key]=f.boundary.amplitude_value(f.combine(vs),1.).record()
                f.spectrum.publish(path,dict(design_sha256=digest(ed),t=t,values=values,new_points=len(keys),**f.timing(tick)))
        finally:geometry.close()
    else:raise ValueError('Unknown phase')
    if only_index is None:
        f.spectrum.publish(receipt,dict(design_sha256=digest(ed),phase=phase,rank=rank,
            job_id=os.environ['SLURM_JOB_ID'],**f.timing(start)))


def dispersion(reference,cells):
    require(len(cells)==26,'Expected all tested cells')
    variants=[]
    for name,pr,tr in [('original','base','base'),('two_panels','head_h2','axis3_h2'),
                       ('four_panels','head_h4','axis3_h4'),('four_nonuniform_panels','head_check','axis3_check')]:
        delta=math.fsum(c['rules'][pr if c['cell']['term'].startswith('primary:') else tr]-c['rules']['base'] for c in cells)
        variants.append(dict(label=name,shift=delta,amplitude=reference+delta))
    shifts=[v['shift'] for v in variants]
    mean=statistics.mean(shifts);sd=statistics.stdev(shifts);rms=math.sqrt(statistics.mean(v*v for v in shifts))
    require(math.isclose(rms*rms,mean*mean+3*sd*sd/4,rel_tol=1e-12,abs_tol=1e-30),'Variance identity failed')
    return dict(variants=variants,mean_shift=mean,grid_standard_deviation=sd,rms_shift=rms,
                relative_rms=rms/abs(reference),relative_standard_deviation=sd/abs(reference),
                observed_shift_min=min(shifts),observed_shift_max=max(shifts),
                probabilities_assigned=False,is_confidence_interval=False,full_error_bar=None)


def reduce_energy(root,t):
    configure(root);eroot,ed,p=checked_energy(root,t)
    if (root/'completed'/f'{label(t)}.json').exists():return completed(root,t)
    baseline=read(eroot/'baseline.json');prior={c['cell']['id']:c for c in baseline['cells']}
    cells=[];seen=set();timings=Counter()
    for c in td['cells']:
        primary=c['cell']['term'].startswith('primary:');rules=set(old.PRIMARY_RULES if primary else old.TAIL_RULES)
        partial=defaultdict(list);counts=Counter()
        for i in c['batch_indices']:
            require(i not in seen,'Duplicate reduction batch');seen.add(i)
            geo=template_batch(i);row=read(eroot/f'results/batch_{i:05d}.json');validate_batch_result(row,ed,geo)
            sums,_,ns=production.batch_estimates(geo,row['values'],f.Value,f.combine,allowed_rules=rules)
            for name,v in sums.items():partial[name].append(v)
            counts.update(ns);timings['contraction_cpu_seconds']+=row['cpu_seconds']
        require(dict(counts)==c['rule_points'] and set(partial)==rules,'Incomplete quadrature rules')
        totals={k:f.combine(v).value.real for k,v in partial.items()}
        if not primary:
            require(math.isclose(totals['base'],prior[c['cell']['id']]['high']['value'][0],rel_tol=1e-12,abs_tol=1e-18),
                    'Tail baseline no longer matches reference')
        assessments=({'angular_heads':old.sequence(*(totals[k] for k in ('base','head_h2','head_h4','head_check')))}
                     if primary else {f'axis{i}':old.sequence(totals['base'],*(totals[f'axis{i}_{s}'] for s in ('h2','h4','check'))) for i in range(4)})
        mixed=(totals['head_h2']-totals['outer_h2']-totals['inner_h2']+totals['base'] if primary
               else totals['mixed_h2']-totals['base']-math.fsum(totals[f'axis{i}_h2']-totals['base'] for i in range(4)))
        cells.append(dict(cell=c['cell'],rules=totals,assessments=assessments,mixed_discrepancy=mixed,
            actual_h2_shift=totals['head_h2' if primary else 'mixed_h2']-totals['base'],
            previous_cell_indicator=prior[c['cell']['id']]['indicator']))
    require(seen==set(range(len(td['batches']))),'Incomplete tested moduli coverage')
    for phase in ('prepare','contract'):
        for rank in range(384):
            row=read(eroot/f'timings/{phase}_{rank:03d}.json')
            require(row['design_sha256']==digest(ed) and row['phase']==phase and row['rank']==rank,'Missing worker completion')
            timings[phase+'_worker_cpu_seconds']+=row['cpu_seconds']
    reference=baseline['reference'];a0=reference['amplitude'];delta=math.fsum(c['actual_h2_shift'] for c in cells)
    out=dict(t=t,design_sha256=digest(ed),experiment_sha256=digest(experiment),reference_amplitude=a0,
        central_value_unchanged=True,complete_planned_study=True,complete_domain=reference['complete_domain'],
        source_amplitude_sha256=ed['reference_file_sha256'],statistics=dispersion(a0,cells),cells=cells,
        actual_combined_two_panel_amplitude=a0+delta,actual_combined_two_panel_shift=delta,timings=dict(timings),
        previous_moduli_indicator=reference['moduli_indicator'],unknown_errors=reference['unknown_errors'],
        other_moduli_directions_retain_existing_diagnostics=True,total_physical_error_bar=None,
        matrix_model_used_for_estimation=False)
    f.spectrum.publish(eroot/'result.json',out)
    receipt=dict(t=t,label=label(t),design_sha256=digest(ed),experiment_sha256=digest(experiment),
        result_file_sha256=sha(eroot/'result.json'),reference_amplitude=a0,
        rms=out['statistics']['rms_shift'],grid_sd=out['statistics']['grid_standard_deviation'],completed_utc=utc())
    f.spectrum.publish(root/'completed'/f'{label(t)}.json',receipt)
    print(json.dumps(receipt),flush=True)
    return out


def completed(root,t):
    receipt=read(root/'completed'/f'{label(t)}.json')
    path=root/'energies'/label(t)/'result.json'
    require(receipt['experiment_sha256']==digest(experiment) and receipt['t']==t
            and receipt['result_file_sha256']==sha(path),'Invalid energy completion')
    result=read(path)
    require(result['t']==t and result['experiment_sha256']==digest(experiment)
            and result['complete_planned_study'] and result['complete_domain'], 'Incomplete amplitude coverage')
    if t != .25:
        _,ed,_=checked_energy(root,t)
        require(receipt['design_sha256']==result['design_sha256']==digest(ed),'Changed completed design')
    return result


def report(root):
    configure(root);values=[completed(root,t) for t in T_VALUES]
    if (root/'completion.json').exists():
        receipt=read(root/'completion.json')
        require(receipt['experiment_sha256']==digest(experiment)
                and receipt['curve_result_sha256']==sha(root/'curve_result.json')
                and receipt['report_sha256']==sha(root/'REPORT.md'),'Changed final report')
        return
    rows=[]
    for value in values:
        t=value['t'];a=value['reference_amplitude'];stats=value['statistics']
        target=t**3*(t-1)*(t-2)*(1+t-t*t/2)/96
        rows.append(dict(t=t,amplitude=a,matrix_model=target,relative_difference=a/target-1,
                         rms=stats['rms_shift'],grid_sd=stats['grid_standard_deviation'],
                         relative_rms=stats['relative_rms'],mean_shift=stats['mean_shift']))
    f.spectrum.publish(root/'curve_result.json',dict(experiment_sha256=digest(experiment),rows=rows,
        complete=True,central_values_unchanged=True,matrix_used_for_estimation=False,
        rms_scope='matched primary near-pair heads and tail vertical-coordinate rules',
        diagnostic_not_total_confidence_interval=True))
    lines=['# Twenty-point moduli stability curve','',
        'Saved production amplitudes are unchanged. Each energy has its own full-amplitude quadrature-variant RMS and SD.',
        'The t=0.25 shifts are imported from its completed study; an additive change in unchanged regions does not change those absolute shifts.',
        'RMS is computed about the fixed saved reference; SD is about the mean of the four matched variants. No division by sqrt(4).',
        'These targeted moduli statistics are not total confidence intervals. Other moduli directions, blocks, compact sampling and unquantified errors remain separate.','',
        '| t | Saved amplitude | RMS / amplitude | Grid SD / amplitude | Mean shift / amplitude | MQM difference |',
        '|---:|---:|---:|---:|---:|---:|']
    for r in rows:lines.append(f"| {r['t']:.3f} | {r['amplitude']:.12g} | {100*r['relative_rms']:.4f}% | {100*r['grid_sd']/abs(r['amplitude']):.4f}% | {100*r['mean_shift']/r['amplitude']:+.4f}% | {100*r['relative_difference']:+.4f}% |")
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    f.spectrum.publish(root/'completion.json',dict(experiment_sha256=digest(experiment),
        curve_result_sha256=sha(root/'curve_result.json'),report_sha256=sha(root/'REPORT.md'),
        points=20,completed_utc=utc()))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','worker','reduce','report'))
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--t',type=float)
    parser.add_argument('--phase',choices=('prepare','contract'));parser.add_argument('--only-index',type=int)
    args=parser.parse_args();root=args.root.resolve()
    if args.action=='prepare':prepare_energy(root,args.t)
    elif args.action=='worker':worker(root,args.t,args.phase,args.only_index)
    elif args.action=='reduce':reduce_energy(root,args.t)
    else:report(root)
