"""One-cell physical-panel refinement with a single coarse/fine comparison."""

from genus1_source_paths import ROOT as _SOURCE_ROOT, source_path as _source_path
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict
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

CELL_ID = 'e966df0f9792868fef6ac3107d79732324778459aaeae46a315566957119aaf5'
TERM = 'bulk:pair_face_analytic'
ORDERS = {'low': (2, 3, 3, 3), 'high': (3, 4, 4, 4)}


def configure(root):
    global f, g, catalog
    loc = json.loads((root/'locations.json').read_text())
    remote = bool(os.environ.get('SLURM_JOB_ID'))
    runtime = Path(loc['runtime']['remote' if remote else 'local'])
    sys.path.insert(0, str(runtime/'runtime/source/code/tasks'))
    import genus1_fresh_analytic
    f = genus1_fresh_analytic
    p = f.load(runtime)
    if f.fingerprint(p) != loc['source_plan_sha256'] or p['physics']['t'] != .25:
        raise ValueError('foreign CFT runtime')
    campaign.f = f
    spec = importlib.util.spec_from_file_location('upper_pair_geometry', root/'pilot/genus1_moduli_panels.py')
    g = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(g)
    catalog = json.loads((root/'bank_catalog.json').read_text())
    sources = [Path(p) for p in catalog['sources']]
    if not remote:
        sources = [Path(loc['local_diagnostics'])/p.relative_to(loc['remote_diagnostics']) for p in sources]
    campaign.source, campaign.parents = sources[0], sources[1:]
    return p


class Builder(production.Builder):
    def __init__(self, root, p, atlas):
        self.root, self.p, self.batch_size = root, p, 32
        self.f = f
        self.planner = f.spectrum.Planner(p, atlas)
        self.grids, self.routing, self.primary_cache = {}, {}, {}
        self.batches, self.costs, self.cells = [], [], []
        self.reset()

    def register(self, request, primary):
        if primary:
            raise ValueError('only the selected pair cell is authorized')
        key, spec = self.planner.grid(request, False, self.p['profile'])
        f.check_momentum_grid(self.p, spec)
        if key not in self.grids:
            self.grids[key] = dict(kind=spec['case']['kind'], axes=[len(r['nodes']) for r in spec['grid']['rules']])
            if key in catalog['banks']:
                self.routing[key] = catalog['banks'][key]
            else:
                f.spectrum.publish(f.spectrum.grid_directory(self.root, key)/'spec.json', spec)
        return key


def prepare(root, prior):
    if (root/'design.json').exists():
        raise ValueError('immutable design already exists')
    (root/'pilot').mkdir(exist_ok=True)
    files = [Path(__file__), Path(production.__file__), Path(campaign.__file__),
             _source_path('code/one_loop/three_point/moduli_integration/genus1_moduli_panels.py')]
    for path in files:
        shutil.copy2(path, root/'pilot'/path.name)
    shutil.copy2(prior/'locations.json', root/'locations.json')
    p = configure(root)
    baseline = f.spectrum.read(prior/'energies/t0250/baseline.json')
    old = baseline['amplitude']
    if old['t'] != .25 or not old['complete_domain']:
        raise ValueError('complete same-energy starting amplitude required')
    matches = [row for row in baseline['cells'] if row['cell']['id'] == CELL_ID]
    if len(matches) != 1:
        raise ValueError('selected cell must occur exactly once')
    selected = matches[0]
    cell = selected['cell']
    if cell['term'] != TERM or cell['bounds'] != [[0.,1.],[.5,1.],[0.,1.],[0.,1.]]:
        raise ValueError('foreign target domain')
    f.spectrum.publish(root/'baseline.json', baseline)
    physics = f.make_plan(p['config'])
    builder = Builder(root, p, f.make_atlas(.25))
    measures = {}
    for name, orders in ORDERS.items():
        weights = []
        for sample, meta in g.pair_rule(physics, 'bulk', orders,
                bounds=cell['bounds'][:2], separation_splits=(2.,4.,8.)):
            builder.add(name, dict(kind='collision', requests=f.encoded([asdict(r) for r in sample.requests])),
                        sample.jacobian, meta['puncture'])
            weights.append(sample.jacobian)
        measures[name] = math.fsum(weights)
    builder.finish_cell(cell)
    fresh = sorted(set(builder.grids)-set(builder.routing))
    cost = [20.3*math.prod(builder.grids[k]['axes'])/512 for k in fresh]
    allocations, loads = campaign.assign(cost, 384)
    contractions, cloads = campaign.assign(builder.costs, 384)
    # This is a bounded one-cell design. No additional rule is selected by its result.
    summary = dict(points=builder.cells[0]['points'], fresh_banks=len(fresh), reused_banks=len(builder.routing),
        batches=len(builder.batches), rule_points=builder.cells[0]['rule_points'], physical_measures=measures,
        banks_by_momentum_axes=dict(Counter(str(tuple(v['axes'])) for v in builder.grids.values())),
        estimated_prepare_max_seconds=max(loads), estimated_contract_max_seconds=max(cloads))
    d = dict(format='genus1-upper-pair-refinement-v1', t=.25, source_plan_sha256=f.fingerprint(p),
        locations_sha256=f.shared.sha(root/'locations.json'), catalog_sha256=f.shared.sha(root/'bank_catalog.json'),
        baseline_sha256=f.shared.sha(root/'baseline.json'), target_cell_id=CELL_ID, orders=f.encoded(ORDERS),
        separation_splits=[2.,4.,8.], cells=builder.cells, batches=builder.batches, fresh_banks=fresh,
        spec_sha256={k:f.shared.sha(f.spectrum.grid_directory(root,k)/'spec.json') for k in fresh},
        routing=builder.routing, prepare=allocations, contract=contractions, workers=384, summary=summary,
        pilot_sources={str(path.relative_to(root)):f.shared.sha(path) for path in sorted((root/'pilot').glob('*.py'))},
        physical_cutoffs_changed=False, momentum_design_changed=False, baseline_replaced=False,
        indicator_method='sum of absolute low/high changes in annulus, near complement and far complement',
        indicator_is_error_bound=False, numerical_threshold_stops_production=False, matrix_model_used=False)
    f.spectrum.publish(root/'design.json', d)
    print(json.dumps(summary, indent=2), flush=True)


def checked(root):
    p = configure(root)
    d = f.spectrum.read(root/'design.json')
    if (d['source_plan_sha256'] != f.fingerprint(p) or d['target_cell_id'] != CELL_ID or
            d['orders'] != f.encoded(ORDERS) or d['workers'] != 384 or len(d['cells']) != 1):
        raise ValueError('changed target design')
    for name, key in [('locations.json','locations_sha256'), ('bank_catalog.json','catalog_sha256'), ('baseline.json','baseline_sha256')]:
        if f.shared.sha(root/name) != d[key]:
            raise ValueError('changed input: '+name)
    for rel, digest in d['pilot_sources'].items():
        if f.shared.sha(root/rel) != digest:
            raise ValueError('changed implementation: '+rel)
    return p, d


def worker(root, phase):
    p, d = checked(root)
    identity = f.fingerprint(d)
    rank = int(os.environ['SLURM_PROCID'])
    if int(os.environ['SLURM_NTASKS']) != d['workers'] or not 0 <= rank < d['workers']:
        raise ValueError('foreign allocation')
    if campaign.completed_worker(root, d, identity, phase, rank):
        return
    start = f.clock()
    if phase == 'prepare':
        for i in d['prepare'][rank]:
            key = d['fresh_banks'][i]
            if f.shared.sha(f.spectrum.grid_directory(root,key)/'spec.json') != d['spec_sha256'][key]:
                raise ValueError('changed bank specification')
            while True:
                result = f.spectrum.build(root,p,key,cpu_budget=240,max_new_chunks=64,max_new_elliptic_nodes=1024)
                if result['complete_rule']:
                    break
                if (result.get('failure') or {}).get('type') != 'TimeLimit':
                    raise RuntimeError(str(result))
    elif phase == 'contract':
        from genus1_boundary_batch import GeometryStore
        physics = f.make_plan(p['config'])
        atlas = f.make_atlas(.25)
        geometry = GeometryStore(atlas,root/f'work/geometry/{rank:03d}')
        runtime = f.spectrum.Runtime(root,p,atlas,geometry,point_budget=1e-12)
        runtime.banks = campaign.Banks(root,d)
        try:
            for i in d['contract'][rank]:
                samples = campaign.batch(root,d,i)
                path = root/f'results/batch_{i:05d}.json'
                if path.exists():
                    row = f.spectrum.read(path)
                    if row['design_sha256'] != identity or set(row['values']) != set(samples):
                        raise ValueError('foreign saved contraction')
                    continue
                tick = f.clock()
                values = {}
                for key, sample in samples.items():
                    parts = [runtime.collide(f.request_from_record(req),grid,sample['max_weight']).scaled(req['multiplicity'])
                             for req,grid in zip(sample['requests'],sample['grids'])]
                    values[key] = f.boundary.amplitude_value(f.combine(parts),1.).record()
                f.spectrum.publish(path,dict(design_sha256=identity,values=values,**f.timing(tick)))
        finally:
            geometry.close()
    else:
        raise ValueError('unsupported phase')
    f.spectrum.publish(root/f'timings/{phase}_{rank:03d}.json',dict(design_sha256=identity,
        phase=phase,rank=rank,job_id=os.environ['SLURM_JOB_ID'],**f.timing(start)))


def reduce(root):
    p, d = checked(root)
    identity = f.fingerprint(d)
    baseline = f.spectrum.read(root/'baseline.json')
    sums, panels, counts = defaultdict(list), defaultdict(list), Counter()
    for i, descriptor in enumerate(d['batches']):
        if f.shared.sha(root/descriptor['path']) != descriptor['sha256']:
            raise ValueError('changed geometry')
        geometry = f.spectrum.read(root/descriptor['path'])
        row = f.spectrum.read(root/f'results/batch_{i:05d}.json')
        if row['design_sha256'] != identity or geometry['cell_id'] != CELL_ID:
            raise ValueError('foreign contraction')
        total, tags, ns = production.batch_estimates(geometry,row['values'],f.Value,f.combine)
        for name,value in total.items():
            sums[name].append(value)
        for tag,value in tags.items():
            panels[tag].append(value)
        counts.update(ns)
    if dict(counts) != d['summary']['rule_points'] or set(sums) != {'low','high'}:
        raise ValueError('incomplete comparison')
    values = {name:f.combine(rows) for name,rows in sums.items()}
    by_panel = {name:{tag:math.fsum(v) for (rule,tag),v in panels.items() if rule==name} for name in values}
    if set(by_panel['low']) != set(by_panel['high']):
        raise ValueError('unmatched physical subregions')
    deltas = {tag:by_panel['high'][tag]-by_panel['low'][tag] for tag in by_panel['low']}
    indicator = math.fsum(abs(v) for v in deltas.values())
    old = baseline['amplitude']
    target = next(c for c in baseline['cells'] if c['cell']['id']==CELL_ID)
    kept = [c for c in baseline['cells'] if c['cell']['term']==TERM and c['cell']['id']!=CELL_ID]
    original_pair = f.combine([f.Value.from_record(c['high']) for c in kept]+[f.Value.from_record(target['high'])])
    if not math.isclose(original_pair.value.real,old['components'][TERM]['value']['value'][0],rel_tol=2e-12,abs_tol=1e-18):
        raise ValueError('inconsistent baseline cell decomposition')
    pair = f.combine([f.Value.from_record(c['high']) for c in kept]+[values['high']])
    components = deepcopy(old['components'])
    components[TERM].update(value=pair.record(), quadrature_indicator=math.fsum(c['indicator'] for c in kept)+indicator,
        replaced_cell_id=CELL_ID, quadrature_indicator_method=d['indicator_method'], status='fixed_refinement_complete')
    boundary = f.combine(f.Value.from_record(v['value']) for v in components.values())
    amplitude = boundary.value.real+old['compact']['mean']
    moduli = math.fsum(v['quadrature_indicator'] for v in components.values())
    if not math.isclose(moduli,old['moduli_indicator']-target['indicator']+indicator,rel_tol=2e-12,abs_tol=1e-18):
        raise ValueError('indicator replacement is inconsistent')
    timings = {}
    for phase in ('prepare','contract'):
        rows = [f.spectrum.read(root/f'timings/{phase}_{rank:03d}.json') for rank in range(d['workers'])]
        if any(row['design_sha256'] != identity or row['phase'] != phase or row['rank'] != rank for rank,row in enumerate(rows)):
            raise ValueError('foreign worker receipt')
        timings[phase] = dict(cpu_seconds=math.fsum(row['cpu_seconds'] for row in rows),max_wall_seconds=max(row['wall_seconds'] for row in rows))
    result = dict(design_sha256=identity,t=.25,complete_domain=True,amplitude=amplitude,
        previous_amplitude=old['amplitude'],amplitude_change=amplitude-old['amplitude'],
        moduli_indicator=moduli,relative_moduli_indicator=moduli/abs(amplitude),previous_moduli_indicator=old['moduli_indicator'],
        selected_cell=dict(cell=target['cell'],old_value=target['high'],old_indicator=target['indicator'],
            low=values['low'].record(),high=values['high'].record(),indicator=indicator,
            by_panel=by_panel,panel_deltas=deltas,orders=d['orders']),
        components=components,boundary=boundary.record(),compact=old['compact'],
        block_indicator=boundary.errors['block']+old['compact']['errors']['block'],
        compact_sampling_halfwidth95=old['compact']['sampling_halfwidth95'],
        indicator_improved=indicator<target['indicator'],baseline_replaced=False,indicator_is_error_bound=False,
        full_amplitude_certified=False,unknown_errors=old['unknown_errors'],timings=timings,summary=d['summary'],
        matrix_model_used_for_design=False,created_utc=f.shared.utc())
    f.spectrum.publish(root/'amplitude.json',result)
    f.spectrum.publish(root/'completion.json',dict(design_sha256=identity,t=.25,complete_domain=True,
        amplitude_file_sha256=f.shared.sha(root/'amplitude.json'),created_utc=f.shared.utc()))
    print(json.dumps({k:result[k] for k in ('amplitude','amplitude_change','moduli_indicator','relative_moduli_indicator','indicator_improved')},indent=2),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','worker','reduce'))
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--prior',type=Path)
    parser.add_argument('--phase',choices=('prepare','contract'))
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.root,args.prior)
    elif args.action == 'worker':
        worker(args.root,args.phase)
    else:
        reduce(args.root)
