"""Fixed-design, pole-free amplitude curve with a pinned parallel allocation.

Geometry and its physical weights are shared; energy-dependent CFT banks and
values are distinct. The completed anchor is imported only at its own energy.
No matrix-model value participates in planning or numerical acceptance.
"""

from genus1_source_paths import ROOT as _SOURCE_ROOT, source_path as _source_path
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import csv
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import signal
import sys
import time

sys.dont_write_bytecode = True
import genus1_moduli_amplitude as production
campaign = production.campaign
T_VALUES = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
FORMAT = 'genus1-fixed-design-curve-v1'
FINAL_T_VALUES = [.025] + [i/20 for i in range(1,20)]
FINAL_FORMAT = 'genus1-final-pair-design-curve-v2'


def configure(root):
    global f, source, anchor, base
    locations = json.loads((root/'locations.json').read_text())
    remote = bool(os.environ.get('SLURM_JOB_ID')) or locations.get('remote_only', False)
    source = Path(locations['source_remote' if remote else 'source'])
    anchor = Path(locations['anchor_remote' if remote else 'anchor'])
    sys.path.insert(0, str(source/'runtime/source/code/tasks'))
    sys.path.insert(0, str(source/'runtime/source/code/one_loop/three_point/other_scripts'))
    import genus1_fresh_analytic
    f = genus1_fresh_analytic
    base = f.load(source)
    if f.fingerprint(base) != locations['source_plan_sha256']:
        raise ValueError('changed frozen scientific runtime')
    campaign.f, campaign.source, campaign.parents = f, source, []
    return base


def energy_plan(t):
    if not math.isfinite(t) or not 0 < t < 1:
        raise ValueError('the supported pole-free chamber is 0<t<1')
    p = deepcopy(base)
    p['config']['run']['t'] = t
    p['physics'] = f.make_plan(p['config']).record()
    # The inherited refinement records describe the fixed partition's origin;
    # they do not authorize any further refinement at this energy.
    return p


def check_plan(p):
    expected = energy_plan(p['physics']['t'])
    if p != expected:
        raise ValueError('curve point changed settings beyond its energy')


def checked(root):
    d = f.spectrum.read(root/'curve.json')
    if d['format'] not in (FORMAT, FINAL_FORMAT) or d['source_plan_sha256'] != f.fingerprint(base):
        raise ValueError('foreign curve or scientific runtime')
    for rel, digest in d['files'].items():
        if f.shared.sha(root/rel) != digest:
            raise ValueError('changed frozen curve input: '+rel)
    expected_t = FINAL_T_VALUES if d['format'] == FINAL_FORMAT else T_VALUES
    if d['workers'] != 384 or d['t_values'] != expected_t:
        raise ValueError('changed curve allocation or energy sequence')
    return d


def payload(root, mesh, index):
    descriptor = mesh['batches'][index]
    path = root/'mesh'/descriptor['path']
    if f.shared.sha(path) != descriptor['sha256']:
        raise ValueError('changed shared moduli geometry')
    row = f.spectrum.read(path)
    if (len(row['samples']) != descriptor['points'] or
            f.fingerprint(sorted(row['samples'])) != descriptor['point_ids_sha256']):
        raise ValueError('changed moduli sample coverage')
    return row


def component_reduce(mesh, get_batch, tolerances, *, require_complete=True):
    """Stream every cell exactly once; low/high differences remain indicators."""
    cells, seen, point_count = [], set(), 0
    for c in mesh['cells']:
        partial = defaultdict(list)
        counts = Counter()
        expected_rules = {'high','height','puncture','real'} if c.get('directional_pair') else {'low','high'}
        for i in c['batch_indices']:
            if i in seen:
                raise ValueError('duplicate batch in moduli reduction')
            seen.add(i)
            geometry, values = get_batch(i)
            if geometry['cell_id'] != c['cell']['id']:
                raise ValueError('batch belongs to a different cell')
            sums, _, ns = production.batch_estimates(geometry, values, f.Value, f.combine,allowed_rules=expected_rules)
            for rule, value in sums.items():
                partial[rule].append(value)
            counts.update(ns)
            point_count += len(values)
        if dict(counts) != c['rule_points'] or set(partial) != expected_rules:
            raise ValueError('incomplete moduli comparison rules')
        totals = {k:f.combine(v) for k,v in partial.items()}
        high = totals['high']
        if c.get('directional_pair'):
            delta = dict(torus_real=(totals['real'].value-high.value).real,
                         height=(totals['height'].value-high.value).real,
                         puncture=(totals['puncture'].value-totals['height'].value).real)
            cells.append(dict(cell=c['cell'],high=high.record(),
                indicator=math.fsum(abs(v) for v in delta.values()),directional_deltas=delta,
                rules={k:v.record() for k,v in totals.items()},rule_points=dict(counts)))
        else:
            low = totals['low']
            cells.append(dict(cell=c['cell'],low=low.record(),high=high.record(),
                              indicator=abs(high.value-low.value)))
    expected_batches = set(mesh.get('active_batch_indices', range(len(mesh['batches']))))
    if seen != expected_batches or point_count != mesh['points']:
        raise ValueError('incomplete boundary coverage')
    components = {}
    for term in sorted({c['cell']['term'] for c in cells}):
        rows = [c for c in cells if c['cell']['term'] == term]
        high = f.combine(f.Value.from_record(c['high']) for c in rows)
        low = (f.combine(f.Value.from_record(c['low']) for c in rows)
               if all('low' in c for c in rows) else None)
        indicator = math.fsum(c['indicator'] for c in rows)
        components[term] = dict(value=high.record(), low=None if low is None else low.record(),
            quadrature_indicator=indicator, tolerance=tolerances[term], cell_count=len(rows),
            status='quadrature_indicator_met' if indicator <= tolerances[term] else 'planned_comparison_limit')
    expected_terms = set(f.boundary.term_names('analytic'))
    if not set(components) <= expected_terms or (require_complete and set(components) != expected_terms):
        raise ValueError('missing retained amplitude term')
    total = f.combine(f.Value.from_record(v['value']) for v in components.values())
    return components, cells, total, math.fsum(v['quadrature_indicator'] for v in components.values())


def prepare(args):
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root/'pilot').mkdir()
    for path in (Path(__file__), Path(production.__file__), Path(campaign.__file__),
                 _source_path('code/one_loop/three_point/other_scripts/genus1_moduli_amplitude_report.py')):
        shutil.copy2(path, root/'pilot'/path.name)
    locations = dict(source=str(args.source.resolve()), source_remote=args.source_remote,
        anchor=str(args.anchor.resolve()), anchor_remote=args.anchor_remote,
        source_plan_sha256=json.loads((args.source/'plan.json').read_text())['sha256'])
    (root/'locations.json').write_text(json.dumps(locations, indent=2)+'\n')
    configure(root)
    old = production.source_result(source, base)
    ad = f.spectrum.read(anchor/'design.json')
    ar = f.spectrum.read(anchor/'amplitude.json')
    if (ad['source_plan_sha256'] != f.fingerprint(base) or
            ar['design_sha256'] != f.fingerprint(ad) or not ar['complete_domain'] or
            ad['mode'] != 'primary-first' or base['physics']['t'] != .25):
        raise ValueError('completed accepted t=0.25 anchor required')
    request = f.spectrum.read(source/'waves/000_request.json')
    if request['plan_sha256'] != f.fingerprint(base):
        raise ValueError('foreign retained partition')
    production.validate_cells(request['cells'], require_pair=True)
    if {c['cell']['id'] for c in ad['cells']} != {
            c['id'] for c in request['cells'] if c['term'].startswith('primary:')}:
        raise ValueError('anchor primary partition changed')
    imports = dict(t=.25, amplitude=ar, anchor_design_sha256=f.fingerprint(ad),
        anchor_amplitude_sha256=f.shared.sha(anchor/'amplitude.json'),
        scientific_limitations=old['unknown_errors'])
    f.spectrum.publish(root/'anchor.json', imports)
    atlas = f.make_atlas(.25)
    builder = production.Builder(root/'mesh', base, atlas, 64)
    anchor_values = {}
    # Copy the accepted primary rule exactly, including clipped angular tails.
    for c in ad['cells']:
        for i in c['batch_indices']:
            b = ad['batches'][i]
            if f.shared.sha(anchor/b['path']) != b['sha256']:
                raise ValueError('changed anchor geometry')
            row = f.spectrum.read(anchor/b['path'])
            value = f.spectrum.read(anchor/f'results/batch_{i:05d}.json')
            if value['design_sha256'] != f.fingerprint(ad) or set(value['values']) != set(row['samples']):
                raise ValueError('incomplete anchor contraction')
            for key, s in row['samples'].items():
                sample = dict(kind='primary', short=s['short'])
                if f.fingerprint(sample) != key:
                    raise ValueError('changed primary point identity')
                anchor_values[key] = value['values'][key]
                for rule, w, tag in row['weights'][key]:
                    builder.add(rule, sample, w, tag)
        builder.finish_cell(c['cell'])
    collision_checks = []
    physics = f.make_plan(base['config'])
    for cell in request['cells']:
        if cell['term'].startswith('primary:'):
            continue
        points, rules = f.cell_nodes(cell, directional_probes=False)
        samples = {}
        for oldkey, u in points.items():
            s = f.sample_at(cell, u, physics, atlas)
            saved = f.spectrum.read(source/'points'/oldkey/'sample.json')
            if any(saved[k] != v for k, v in s.items()) or s['kind'] != 'collision':
                raise ValueError('collision geometry changed or unexpected zero')
            samples[oldkey] = s
            collision_checks.append(dict(cell=cell, unit=u, sample=s))
            value_path = source/'points'/oldkey/'value.json'
            if not value_path.exists():
                value_path = source/'evidence/points'/oldkey/'value.json'
            v = f.spectrum.read(value_path)
            if v['plan_sha256'] != f.fingerprint(base) or v['sample_sha256'] != f.shared.sha(source/'points'/oldkey/'sample.json'):
                raise ValueError('foreign collision reference value')
            key = f.fingerprint(dict(kind='collision', requests=s['requests']))
            anchor_values[key] = f.Value.from_record(v['value']).scaled(1/s['jacobian']).record()
        for order, label in (('2', 'low'), ('3', 'high')):
            for oldkey, weight in rules['rules'][order]:
                s = samples[oldkey]
                builder.add(label, dict(kind='collision', requests=s['requests']), weight*s['jacobian'])
        builder.finish_cell(cell)
    mesh = dict(cells=builder.cells, batches=builder.batches,
                points=sum(b['points'] for b in builder.batches), reference_grids=builder.grids,
                component_tolerances={k:v['absolute_tolerance'] for k,v in base['quadrature']['controls'].items()})
    if len(mesh['cells']) != 74 or mesh['points'] > 60000 or len(builder.grids) != 787:
        raise ValueError('fixed-design work census changed')
    f.spectrum.publish(root/'mesh.json', mesh)
    def anchor_batch(i):
        row = payload(root, mesh, i)
        return row, {k:anchor_values[k] for k in row['samples']}
    components, _, total, indicator = component_reduce(mesh, anchor_batch, mesh['component_tolerances'])
    replay = total.value.real+ar['compact']['mean']
    differences = {k:components[k]['value']['value'][0]-v['value']['value'][0] for k,v in ar['components'].items()}
    if (abs(replay-ar['amplitude']) > 2e-16 or abs(indicator-ar['moduli_indicator']) > 2e-16 or
            max(map(abs, differences.values())) > 2e-16):
        raise ValueError('full anchor reduction replay failed')
    del anchor_values
    f.spectrum.publish(root/'validation.json', dict(anchor_replay=replay, anchor_amplitude=ar['amplitude'],
        amplitude_difference=replay-ar['amplitude'], component_differences=differences,
        moduli_indicator_difference=indicator-ar['moduli_indicator'], cft_recomputed=False,
        domain_cells=74, retained_terms=len(components), fixed_geometry=True))
    compact_rows = f.spectrum.read(source/'compact/locations.json')['locations']
    all_batches = [payload(root, mesh, i) for i in range(len(mesh['batches']))]
    energies = []
    prep_units, prep_costs, contract_units, contract_costs, groups = [], [], [], [], []
    for t in T_VALUES:
        if t == .25:
            continue
        label = f't{round(100*t):03d}'
        eroot = root/'energies'/label
        p = energy_plan(t)
        f.spectrum.publish(eroot/'plan.json', p)
        f.spectrum.publish(eroot/'compact/locations.json', dict(plan_sha256=f.fingerprint(p), locations=compact_rows))
        planner = f.spectrum.Planner(p, f.make_atlas(t))
        new_atlas, new_physics = planner.atlas, f.make_plan(p['config'])
        for row in collision_checks:
            if f.sample_at(row['cell'], row['unit'], new_physics, new_atlas) != row['sample']:
                raise ValueError('collision moduli geometry depends on energy; explicit replanning required')
        mapping, specs, cache = {}, {}, {}
        for row in all_batches:
            for s in row['samples'].values():
                primary = s['kind'] == 'primary'
                records = [s['short']] if primary else s['requests']
                for oldkey, record in zip(s['grids'], records):
                    cache_key = (('primary', tuple(record['sector']), record['s1'], record['s2']) if primary
                                 else ('collision', f.fingerprint(record)))
                    if cache_key not in cache:
                        value = f.ShortPoint.from_record(record) if primary else f.request_from_record(record)
                        key, spec = planner.grid(value, primary, p['profile'])
                        f.check_momentum_grid(p, spec)
                        cache[cache_key] = key
                        if key not in specs:
                            specs[key] = spec
                            f.spectrum.publish(f.spectrum.grid_directory(eroot, key)/'spec.json', spec)
                    key = cache[cache_key]
                    if oldkey in mapping and mapping[oldkey] != key:
                        raise ValueError('a reference grid alias split at this energy')
                    mapping[oldkey] = key
        if set(mapping) != set(builder.grids) or set(mapping.values()) & set(builder.grids):
            raise ValueError('incomplete energy routing or cross-energy bank reuse')
        banks = sorted(specs)
        ed = dict(t=t, label=label, plan_sha256=f.fingerprint(p), grid_map=mapping, banks=banks,
            spec_sha256={k:f.shared.sha(f.spectrum.grid_directory(eroot,k)/'spec.json') for k in banks},
            compact_locations_sha256=f.shared.sha(eroot/'compact/locations.json'),
            banks_by_family=dict(Counter(s['case']['kind'] for s in specs.values())),
            banks_by_axes=dict(Counter(str(tuple(len(r['nodes']) for r in s['grid']['rules'])) for s in specs.values())))
        f.spectrum.publish(eroot/'design.json', ed)
        energies.append(dict(label=label, t=t, design_sha256=f.fingerprint(ed)))
        for key in banks:
            s = specs[key]
            prep_units.append(dict(energy=label, kind='bank', key=key))
            prep_costs.append((100. if s['case']['kind'] == 'primary' else 20.3)*math.prod(len(r['nodes']) for r in s['grid']['rules'])/512)
        for i in range(len(compact_rows)):
            prep_units.append(dict(energy=label, kind='compact', index=i))
            prep_costs.append(180.)
        for i, b in enumerate(mesh['batches']):
            contract_units.append(dict(energy=label, kind='contract', index=i))
            contract_costs.append(builder.costs[i])
            groups.append((label, b['coordinate_group']))
        print(json.dumps(dict(t=t, banks=len(banks), boundary_points=mesh['points'], compact_points=len(compact_rows))), flush=True)
    preparations, ploads = campaign.assign(prep_costs, 384)
    contracts, cloads = production.assign_geometry_groups(contract_costs, groups, 384)
    for units, allocation in ((prep_units, preparations), (contract_units, contracts)):
        if sorted(i for row in allocation for i in row) != list(range(len(units))):
            raise ValueError('parallel allocation does not cover every unit exactly once')
    files = {str(q.relative_to(root)):f.shared.sha(q) for q in sorted((root/'pilot').glob('*.py'))}
    for rel in ('anchor.json', 'mesh.json', 'validation.json'):
        files[rel] = f.shared.sha(root/rel)
    for e in energies:
        for name in ('plan.json', 'design.json'):
            rel = 'energies/'+e['label']+'/'+name
            files[rel] = f.shared.sha(root/rel)
    curve = dict(format=FORMAT, created_utc=f.shared.utc(), t_values=T_VALUES,
        source_plan_sha256=f.fingerprint(base), anchor_design_sha256=f.fingerprint(ad),
        files=files, workers=384, partition='shared', energies=energies,
        prepare_units=prep_units, contract_units=contract_units, prepare=preparations, contract=contracts,
        summary=dict(amplitudes=10, fresh_amplitudes=9, imported_amplitudes=1,
            new_boundary_banks=sum(len(f.spectrum.read(root/'energies'/e['label']/'design.json')['banks']) for e in energies),
            fresh_boundary_points=9*mesh['points'], fresh_compact_points=9*len(compact_rows),
            contraction_batches=len(contract_units), estimated_prepare_max_seconds=max(ploads),
            estimated_contract_max_seconds=max(cloads)),
        pole_free_chamber='0<t<1, conservative direct-real-contour chamber; not an assertion of the first three-point pole location',
        automatic_refinement=False, momentum_design_changed=False, physical_cutoffs_changed=False,
        unknown_errors=old['unknown_errors'], full_amplitude_certified=False)
    f.spectrum.publish(root/'curve.json', curve)
    print(json.dumps(dict(curve_sha256=f.fingerprint(curve), **curve['summary']), indent=2), flush=True)


def energy_context(root, d, label):
    entry = next(e for e in d['energies'] if e['label'] == label)
    eroot = root/'energies'/label
    ed, p = (f.spectrum.read(eroot/name) for name in ('design.json', 'plan.json'))
    check_plan(p)
    if f.fingerprint(ed) != entry['design_sha256'] or ed['plan_sha256'] != f.fingerprint(p):
        raise ValueError('foreign energy design')
    if f.shared.sha(eroot/'compact/locations.json') != ed['compact_locations_sha256']:
        raise ValueError('changed compact samples')
    if ed.get('retained_sha256') and f.shared.sha(eroot/'retained.json') != ed['retained_sha256']:
        raise ValueError('changed retained energy contributions')
    return eroot, ed, p


def validate_result(row, identity, entry, samples):
    if (row['curve_sha256'] != identity or row['energy_design_sha256'] != entry or
            set(row['values']) != set(samples)):
        raise ValueError('foreign energy or incomplete contraction result')


def energy_indices(d, phase, label):
    if label not in {e['label'] for e in d['energies']}:
        raise ValueError('unknown curve energy')
    return [i for i,u in enumerate(d[phase+'_units']) if u['energy'] == label]


def validate_sequence(d, sequence):
    labels = [e['label'] for e in d['energies']]
    workers = sequence['workers']
    if (sequence['order'] != labels or type(workers) is not int or
            not 1 <= workers <= d['workers']):
        raise ValueError('changed sequential energy order or allocation')
    if set(sequence['allocation']) != set(labels):
        raise ValueError('missing energy allocation')
    for label in labels:
        for phase in ('prepare','contract'):
            rows = sequence['allocation'][label][phase]
            if len(rows) != workers or sorted(i for row in rows for i in row) != energy_indices(d,phase,label):
                raise ValueError('duplicate, missing or foreign energy work unit')


def execution_directory(root):
    name = os.environ.get('GENUS1_CURVE_EXECUTION','sequential')
    if not name or Path(name).name != name or name in ('.','..'):
        raise ValueError('execution directory must be a single relative directory name')
    return root/name


def sequential_checked(root, d):
    sequence = f.spectrum.read(execution_directory(root)/'execution.json')
    if sequence['curve_sha256'] != f.fingerprint(d):
        raise ValueError('sequential dispatch belongs to another curve')
    for rel,sha in sequence['files'].items():
        if f.shared.sha(root/rel) != sha:
            raise ValueError('changed sequential implementation: '+rel)
    validate_sequence(d,sequence)
    return sequence


def completed_energy(root, d, sequence, label):
    e = next(e for e in d['energies'] if e['label'] == label)
    receipt = f.spectrum.read(execution_directory(root)/'completed'/f'{label}.json')
    path = root/'energies'/label/'amplitude.json'
    if (receipt['execution_sha256'] != f.fingerprint(sequence) or
            receipt['curve_sha256'] != f.fingerprint(d) or receipt['energy'] != label or
            receipt['amplitude_sha256'] != f.shared.sha(path)):
        raise ValueError('foreign or changed completed energy')
    value = f.spectrum.read(path)
    if (value['curve_sha256'] != f.fingerprint(d) or
            value['energy_design_sha256'] != e['design_sha256'] or not value['complete_domain']):
        raise ValueError('incomplete energy amplitude')
    return value,receipt


def require_predecessor(root,d,sequence,label):
    index = sequence['order'].index(label)
    if index:
        completed_energy(root,d,sequence,sequence['order'][index-1])


def worker(root, phase, energy=None):
    configure(root)
    d = checked(root)
    identity = f.fingerprint(d)
    rank = int(os.environ['SLURM_PROCID'])
    sequence = sequential_checked(root,d) if energy is not None else None
    workers = d['workers'] if sequence is None else sequence['workers']
    if int(os.environ['SLURM_NTASKS']) != workers or not 0 <= rank < workers:
        raise ValueError(f'{workers} single-core workers required by the execution plan')
    start = f.clock()
    mesh = f.spectrum.read(root/'mesh.json')
    contexts = {}
    if sequence is not None:
        require_predecessor(root,d,sequence,energy)
    assigned = d[phase][rank] if sequence is None else sequence['allocation'][energy][phase][rank]
    units = d[phase+'_units']
    def context(label):
        if label not in contexts:
            eroot, ed, p = energy_context(root, d, label)
            c = dict(root=eroot, design=ed, plan=p)
            if phase == 'contract':
                from genus1_boundary_batch import GeometryStore
                atlas = f.make_atlas(p['physics']['t'])
                c['geometry'] = GeometryStore(atlas, eroot/f'work/geometry/{rank:03d}')
                c['runtime'] = f.spectrum.Runtime(eroot, p, atlas, c['geometry'], point_budget=1e-12)
                c['physics'] = f.make_plan(p['config'])
            contexts[label] = c
        return contexts[label]
    try:
        # Group by energy to bound bank caches and avoid interleaving context setup.
        for index in sorted(assigned, key=lambda i: units[i]['energy']):
            unit = units[index]
            c = context(unit['energy'])
            eroot, ed, p = c['root'], c['design'], c['plan']
            tick = f.clock()
            if unit['kind'] == 'bank':
                key = unit['key']
                if f.shared.sha(f.spectrum.grid_directory(eroot,key)/'spec.json') != ed['spec_sha256'][key]:
                    raise ValueError('changed spectral specification')
                while True:
                    result = f.spectrum.build(eroot, p, key, cpu_budget=240,
                                             max_new_chunks=64, max_new_elliptic_nodes=1024)
                    if result['complete_rule']:
                        break
                    if (result.get('failure') or {}).get('type') != 'TimeLimit' or time.monotonic()-tick[1] > 1550:
                        raise RuntimeError('incomplete bank '+key+': '+str(result))
            elif unit['kind'] == 'compact':
                f.compact_work(eroot, p, unit['index'])
                row = f.spectrum.read(eroot/f"compact/points/{unit['index']:02d}/value.json")
                locations = f.spectrum.read(eroot/'compact/locations.json')['locations']
                if row['plan_sha256'] != f.fingerprint(p) or row['location'] != locations[unit['index']]:
                    raise ValueError('foreign compact observation')
            elif unit['kind'] == 'contract':
                i = unit['index']
                row = payload(root, mesh, i)
                samples = row['samples']
                path = eroot/f'results/batch_{i:05d}.json'
                if path.exists():
                    validate_result(f.spectrum.read(path), identity, f.fingerprint(ed), samples)
                else:
                    keys = list(samples)
                    geometry, runtime, physics = c['geometry'], c['runtime'], c['physics']
                    if all(s['kind'] == 'primary' for s in samples.values()):
                        shorts = [f.ShortPoint.from_record(samples[k]['short']) for k in keys]
                        grids = [ed['grid_map'][samples[k]['grids'][0]] for k in keys]
                        values = runtime.primary(shorts, grids, geometry.prepare(shorts), physics.policy, physics.angle_order)
                        values = {k:f.boundary.amplitude_value(v,1.).record() for k,v in zip(keys,values)}
                    else:
                        values = {}
                        for k, s in samples.items():
                            parts = [runtime.collide(f.request_from_record(a),ed['grid_map'][grid],s['max_weight']).scaled(a['multiplicity'])
                                     for a,grid in zip(s['requests'],s['grids'])]
                            values[k] = f.boundary.amplitude_value(f.combine(parts),1.).record()
                    f.spectrum.publish(path, dict(curve_sha256=identity, energy_design_sha256=f.fingerprint(ed),
                                                  values=values, **f.timing(tick)))
            else:
                raise ValueError('unknown curve work unit')
            receipt = root/f'operations/{phase}/{index:06d}.json'
            if not receipt.exists():
                f.spectrum.publish(receipt, dict(curve_sha256=identity, unit=unit, complete=True,
                    job_id=os.environ['SLURM_JOB_ID'], rank=rank, **f.timing(tick)))
            else:
                saved = f.spectrum.read(receipt)
                if saved['curve_sha256'] != identity or saved['unit'] != unit or not saved['complete']:
                    raise ValueError('foreign completed work receipt')
            # Keep only this energy's live caches. Work is sorted by energy.
            for label in list(contexts):
                if label != unit['energy']:
                    if 'geometry' in contexts[label]:
                        contexts[label]['geometry'].close()
                    del contexts[label]
    except BaseException as exc:
        f.shared.save(root/f'failures/{energy or "all"}_{phase}_{rank:03d}_{os.environ["SLURM_JOB_ID"]}.json',
            dict(curve_sha256=identity, energy=energy, phase=phase, rank=rank, error=repr(exc), **f.timing(start)))
        raise
    finally:
        for c in contexts.values():
            if 'geometry' in c:
                c['geometry'].close()
    path = root/f'timings/{energy or "all"}_{phase}_{rank:03d}_{os.environ["SLURM_JOB_ID"]}.json'
    f.spectrum.publish(path, dict(curve_sha256=identity, energy=energy, phase=phase, rank=rank, completed=len(assigned), **f.timing(start)))


def reduce(root, energy=None):
    configure(root)
    d = checked(root)
    identity = f.fingerprint(d)
    sequence = sequential_checked(root,d) if energy is not None else None
    if sequence is not None:
        require_predecessor(root,d,sequence,energy)
        if (execution_directory(root)/'completed'/f'{energy}.json').exists():
            value,_ = completed_energy(root,d,sequence,energy)
            print(json.dumps(dict(energy=energy,amplitude=value['amplitude'],already_complete=True)),flush=True)
            return
    start = f.clock()
    mesh = f.spectrum.read(root/'mesh.json')
    # Every scheduled unit, not just every result file, must be accounted for.
    timings = {}
    for phase in ('prepare', 'contract'):
        rows = []
        for i, unit in enumerate(d[phase+'_units']):
            if energy is not None and unit['energy'] != energy:
                continue
            r = f.spectrum.read(root/f'operations/{phase}/{i:06d}.json')
            if r['curve_sha256'] != identity or r['unit'] != unit or not r['complete']:
                raise ValueError('incomplete parallel work coverage')
            rows.append(r)
        timings[phase] = dict(units=len(rows), cpu_seconds=math.fsum(r['cpu_seconds'] for r in rows),
                              summed_worker_wall_seconds=math.fsum(r['wall_seconds'] for r in rows))
    amplitudes = {}
    for e in d['energies']:
        if energy is not None and e['label'] != energy:
            continue
        eroot, ed, p = energy_context(root,d,e['label'])
        def get_batch(i):
            row = payload(root,mesh,i)
            values = f.spectrum.read(eroot/f'results/batch_{i:05d}.json')
            validate_result(values,identity,e['design_sha256'],row['samples'])
            return row,values['values']
        active_mesh = mesh
        retained = None
        if ed.get('retained_sha256'):
            retained = f.spectrum.read(eroot/'retained.json')
            active_mesh = dict(mesh, cells=[c for c in mesh['cells'] if c.get('directional_pair')],
                active_batch_indices=ed['batch_indices'],
                points=sum(mesh['batches'][i]['points'] for i in ed['batch_indices']))
        components,cells,total,indicator = component_reduce(active_mesh,get_batch,
            mesh['component_tolerances'],require_complete=retained is None)
        import numpy as np
        if retained is not None:
            from genus1_pair_cell_refinement import replace_cell, TERM
            if len(cells)!=1 or retained['plan_sha256']!=f.fingerprint(p):
                raise ValueError('foreign retained energy or replacement scope')
            replacement=cells[0]
            components,total=replace_cell(retained['amplitude'],retained['pair_cells'],
                replacement['cell']['id'],dict(value=replacement['high'],
                    points=replacement['rule_points']['high']),replacement['indicator'],f.Value,f.combine)
            cells=[c for c in retained['amplitude']['cells'] if c['cell']['id']!=replacement['cell']['id']]+cells
            indicator=math.fsum(c['quadrature_indicator'] for c in components.values())
            compact=retained['amplitude']['compact']
            replicate_means=retained['compact_replicate_means']
        else:
            observations = np.full((16,4),np.nan)
            values = []
            for i,location in enumerate(f.spectrum.read(eroot/'compact/locations.json')['locations']):
                row = f.spectrum.read(eroot/f'compact/points/{i:02d}/value.json')
                if row['plan_sha256'] != f.fingerprint(p) or row['location'] != location:
                    raise ValueError('foreign compact contribution')
                v = f.Value.from_record(row['value'])
                observations[location['replicate'],location['index']] = v.value.real
                values.append(v)
            compact = f.boundary.compact_estimate(observations,f.make_plan(p['config']),
                                                density_identity=dict(fresh_plan_sha256=f.fingerprint(p)))
            compact['errors'] = f.combine(values,[1/64]*64).errors
            replicate_means=observations.mean(axis=1).tolist()
        amplitude = total.value.real+compact['mean']
        r = dict(t=e['t'],curve_sha256=identity,energy_design_sha256=e['design_sha256'],
            plan_sha256=f.fingerprint(p),amplitude=amplitude,components=components,cells=cells,
            compact=compact,boundary=total.record(),moduli_indicator=indicator,
            relative_moduli_indicator=indicator/max(abs(amplitude),1e-300),complete_domain=True,
            unknown_errors=d['unknown_errors'],full_amplitude_certified=False,indicator_is_error_bound=False)
        if d['format']==FINAL_FORMAT:
            pairs=[c for c in cells if 'directional_deltas' in c]
            if len(pairs)!=1:raise ValueError('one revised pair cell is required')
            c=pairs[0];scale=max(abs(c['high']['value'][0]),1e-300)
            r.update(compact_replicate_means=replicate_means,
                retained_contributions_imported=retained is not None,
                pair_diagnostics=dict(cell_id=c['cell']['id'],directional_deltas=c['directional_deltas'],
                    relative_indicator=c['indicator']/scale,
                    local_stability_flag=(c['indicator']/scale>.01 or abs(c['directional_deltas']['torus_real'])/scale>.005)))
        f.spectrum.publish(eroot/'amplitude.json',r)
        amplitudes[e['t']] = r
    if energy is not None:
        e = next(e for e in d['energies'] if e['label'] == energy)
        r = amplitudes[e['t']]
        from genus1_amplitude_report import matrix_amplitude
        target = float(matrix_amplitude(e['t']))
        receipt = dict(execution_sha256=f.fingerprint(sequence),curve_sha256=identity,energy=energy,
            amplitude_sha256=f.shared.sha(root/'energies'/energy/'amplitude.json'),
            t=e['t'],amplitude=r['amplitude'],matrix_model=target,
            relative_difference=(r['amplitude']-target)/target,
            relative_moduli_indicator=r['relative_moduli_indicator'],timings=timings,
            reduction=f.timing(start),completed_utc=f.shared.utc(),full_amplitude_certified=False)
        f.spectrum.publish(execution_directory(root)/'completed'/f'{energy}.json',receipt)
        print(json.dumps(receipt,indent=2),flush=True)
        return
    publish_curve_report(root,d,amplitudes,timings)


def finish_sequence(root):
    configure(root)
    d = checked(root)
    sequence = sequential_checked(root,d)
    amplitudes,receipts = {},[]
    for label in sequence['order']:
        value,receipt = completed_energy(root,d,sequence,label)
        amplitudes[value['t']] = value
        receipts.append(receipt)
    timings = {phase:{key:math.fsum(r['timings'][phase][key] for r in receipts)
        for key in ('units','cpu_seconds','summed_worker_wall_seconds')} for phase in ('prepare','contract')}
    timings['reduction'] = dict(units=len(receipts),cpu_seconds=math.fsum(r['reduction']['cpu_seconds'] for r in receipts),
        summed_worker_wall_seconds=math.fsum(r['reduction']['wall_seconds'] for r in receipts))
    publish_curve_report(root,d,amplitudes,timings)


def publish_curve_report(root,d,amplitudes,timings):
    identity = f.fingerprint(d)
    amplitudes[.25] = f.spectrum.read(root/'anchor.json')['amplitude']
    # The independent dual prediction is evaluated only after all reductions.
    from genus1_amplitude_report import matrix_amplitude, PAPER
    rows = []
    for t in d['t_values']:
        a = amplitudes[t]
        target = matrix_amplitude(t)
        rows.append(dict(t=t,amplitude=a['amplitude'],matrix_model=float(target),matrix_exact=str(target),
            relative_difference=(a['amplitude']-float(target))/float(target),
            moduli_indicator=a['moduli_indicator'],relative_moduli_indicator=a['relative_moduli_indicator'],
            compact_sampling_halfwidth95=a['compact']['sampling_halfwidth95'],
            block_indicator=a['boundary']['errors']['block']+a['compact']['errors']['block'],
            imported_anchor=(t==.25)))
        if d['format']==FINAL_FORMAT:
            row=rows[-1]
            row['partial_numerical_envelope']=(row['moduli_indicator']+row['block_indicator']+
                                                row['compact_sampling_halfwidth95'])
            row['relative_partial_numerical_envelope']=row['partial_numerical_envelope']/max(abs(a['amplitude']),1e-300)
            row['pair_relative_indicator']=a['pair_diagnostics']['relative_indicator']
            row['pair_stability_flag']=a['pair_diagnostics']['local_stability_flag']
    result = dict(curve_sha256=identity,complete=True,rows=rows,timings=timings,
        matrix_source=PAPER,matrix_used_for_planning=False,normalization_fitted=False,
        unknown_errors=d['unknown_errors'],full_amplitude_certified=False,indicator_is_error_bound=False)
    if d['format']==FINAL_FORMAT:
        import numpy as np
        means=np.asarray([amplitudes[t]['compact_replicate_means'] for t in d['t_values']]).T
        if means.shape!=(16,20) or not np.isfinite(means).all():
            raise ValueError('complete paired compact scrambles required')
        covariance=np.cov(means,rowvar=False,ddof=1)/len(means)
        for i,t in enumerate(d['t_values']):
            if not math.isclose(covariance[i,i],amplitudes[t]['compact']['sampling_se']**2,rel_tol=2e-11,abs_tol=1e-40):
                raise ValueError('compact covariance disagrees with pointwise sampling error')
        covariance_record=dict(curve_sha256=identity,t_values=d['t_values'],scrambles=16,
            covariance_of_compact_means=covariance.tolist(),replicate_means=means.tolist(),
            scope='compact Monte Carlo only; shared sample locations across energies',
            includes_moduli_or_block_errors=False)
        f.spectrum.publish(root/'compact_covariance.json',covariance_record)
        result.update(compact_covariance_sha256=f.shared.sha(root/'compact_covariance.json'),
            error_estimation=dict(moduli='sum of cell absolute comparisons; three separate directions in the revised pair cell',
                block='integrated retained-prefix comparisons',compact='pointwise Student-t half-width across 16 independent scrambles',
                partial_envelope='moduli indicator + block indicator + compact pointwise 95% half-width',
                partial_envelope_is_confidence_interval=False,unquantified_errors=d['unknown_errors']))
    f.spectrum.publish(root/'curve_result.json',result)
    with (root/'curve.csv').open('w') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    lines = ['# Fixed-design genus-one amplitude curve','',
        f"{len(d['t_values'])} amplitudes in the conservative pole-free chamber, with the accepted primary angular rule, "
        'eight/nine total momentum nodes and unchanged retained analytic terms. The t=0.25 amplitude '
        'is an identity-checked import. Matching completed contributions are reused only at their own energy.','',
        '| t | Amplitude | Matrix model | Difference | Moduli indicator / amplitude |',
        '|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['t']:.3f} | {row['amplitude']:.12g} | {row['matrix_model']:.12g} | {100*row['relative_difference']:+.4f}% | {100*row['relative_moduli_indicator']:.4f}% |")
    lines += ['',f'[Dual matrix model, equation (58)]({PAPER}). The prediction did not enter planning or acceptance.',
        '', 'Moduli and block indicators are convergence diagnostics, not certified uncertainty bounds. '
        'Compact sampling uncertainty is recorded separately in curve.csv. All points share the compact '
        'sample locations, so their sampling fluctuations are correlated. No automatic refinement is scheduled.',
        '', 'Unresolved: '+ '; '.join(d['unknown_errors'])+'.','']
    if d['format']==FINAL_FORMAT:
        lines += ['The final pair-cell candidate is (2,3,3,3) per physical panel. The torus-real, height '
            'and puncture comparisons were performed separately at every energy. Flagged comparisons are '
            'reported without changing the frozen design.', '',
            'The partial numerical envelope in curve.csv is the linear sum of the moduli indicator, '
            'block indicator and compact pointwise 95% half-width. It is not a total confidence interval '
            'and excludes the unquantified terms above. compact_covariance.json preserves the measured '
            'sampling correlation across the curve. No independence is assumed for deterministic moduli or block errors.', '']
    (root/'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(result,indent=2),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','worker','reduce','verify','finish-sequence','verify-sequence'))
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--source',type=Path)
    parser.add_argument('--source-remote')
    parser.add_argument('--anchor',type=Path)
    parser.add_argument('--anchor-remote')
    parser.add_argument('--phase',choices=('prepare','contract'))
    parser.add_argument('--energy')
    args = parser.parse_args()
    def interrupted(signum,frame):
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,interrupted)
    if args.action == 'prepare':
        prepare(args)
    elif args.action == 'worker':
        worker(args.root,args.phase,args.energy)
    elif args.action == 'reduce':
        reduce(args.root,args.energy)
    elif args.action == 'finish-sequence':
        finish_sequence(args.root)
    elif args.action == 'verify-sequence':
        configure(args.root)
        d = checked(args.root)
        sequence = sequential_checked(args.root,d)
        print(json.dumps(dict(curve_sha256=f.fingerprint(d),execution_sha256=f.fingerprint(sequence),
                              order=sequence['order'],summary=sequence['summary']),indent=2))
    else:
        configure(args.root)
        d = checked(args.root)
        print(json.dumps(dict(curve_sha256=f.fingerprint(d),summary=d['summary']),indent=2))
