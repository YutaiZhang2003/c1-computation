"""Replace one verified pair cell, retaining every other amplitude component.

The completed panel experiments supply the height/puncture comparisons.
Only the missing tau-real comparison is new. Acceptance uses those local
comparisons, never the matrix-model prediction. All reductions use saved CFT
values; this command cannot prepare banks or submit additional work.
"""
import argparse
from copy import deepcopy
import math
from pathlib import Path

import genus1_moduli_panel_campaign as campaign


TERM = 'bulk:pair_face_analytic'
REFERENCES = ('pair_h3_p3', 'pair_h4_p3', 'pair_h4_p4')
PROBE = 'pair_r3_h3_p3'


def artifact(root, relative):
    for base in (root, root/'evidence'):
        path = base/relative
        if path.is_file():
            return path
    raise FileNotFoundError(root/relative)


def close(actual, expected):
    if not math.isclose(actual, expected, rel_tol=2e-12, abs_tol=1e-18):
        raise ValueError(f'inconsistent saved reduction: {actual} != {expected}')


def rule_values(root, names, source_identity):
    """Verify and reduce only the requested rules from an immutable experiment."""
    f = campaign.f
    design_path = artifact(root, 'design.json')
    d = f.spectrum.read(design_path)
    identity = f.fingerprint(d)
    if d['source_plan_sha256'] != source_identity:
        raise ValueError('foreign CFT source in pair experiment')
    for relative, digest in d['pilot_sources'].items():
        if f.shared.sha(artifact(root, relative)) != digest:
            raise ValueError('changed pair experiment implementation')
    path = artifact(root, d['rules']['path'])
    if f.shared.sha(path) != d['rules']['sha256']:
        raise ValueError('changed pair quadrature weights')
    rules = f.spectrum.read(path)
    wanted = {key for name in names for key, _, _ in rules[name]}
    values, receipts = {}, {}
    for index, batch in enumerate(d['batches']):
        path = artifact(root, f'results/batch_{index:05d}.json')
        row = f.spectrum.read(path)
        keys = wanted.intersection(row['values'])
        if not keys:
            continue
        if (row['design_sha256'] != identity or
                len(row['values']) != batch['points'] or
                f.fingerprint(sorted(row['values'])) != batch['point_ids_sha256'] or
                keys.intersection(values)):
            raise ValueError('foreign, incomplete, or duplicate pair observations')
        geometry_path = artifact(root, batch['path'])
        if f.shared.sha(geometry_path) != batch['sha256']:
            raise ValueError('changed pair geometry')
        samples = f.spectrum.read(geometry_path)['samples']
        if set(samples) != set(row['values']):
            raise ValueError('pair observations do not cover their geometry')
        for key in keys:
            s = samples[key]
            if s['kind'] != 'collision' or f.fingerprint(
                    dict(kind=s['kind'], requests=s['requests'])) != key:
                raise ValueError('changed pair sample identity')
            values[key] = f.Value.from_record(row['values'][key])
        receipts[str(path.relative_to(root))] = f.shared.sha(path)
        if set(values) == wanted:
            break
    if set(values) != wanted:
        raise ValueError('missing pair observations')
    result_path = artifact(root, 'result.json')
    result = f.spectrum.read(result_path)
    if (result['design_sha256'] != identity or
            result['source_plan_sha256'] != source_identity):
        raise ValueError('foreign pair result')
    totals = {}
    for name in names:
        rows = rules[name]
        total = f.combine((values[k] for k, _, _ in rows), (w for _, w, _ in rows))
        close(total.value.real, result['results'][name]['value'])
        close(total.errors['block'], result['results'][name]['block_indicator'])
        if len(rows) != result['results'][name]['points']:
            raise ValueError('changed rule coverage')
        totals[name] = dict(value=total.record(), points=len(rows),
                            physical_measure=math.fsum(w for _, w, _ in rows))
    return totals, dict(design_sha256=identity, result_sha256=f.shared.sha(result_path),
                        verified_batches=receipts, cell=d['selected'][TERM]['cell'])


def original_cells(source, p):
    """Reconstruct both original pair cells, including signed block comparisons."""
    f = campaign.f
    mesh = f.spectrum.read(source/'waves/000_request.json')
    if mesh['plan_sha256'] != f.fingerprint(p):
        raise ValueError('foreign source cell partition')
    cells = mesh['cells']
    if isinstance(cells, dict):
        cells = list(cells.values())
    selected = [c for c in cells if c['term'] == TERM]
    # The frozen mesh describes the terminal children through its cell IDs.
    if not selected:
        raise ValueError('source mesh has no pair cells')
    results = []
    for cell in selected:
        rules = f.spectrum.read(source/f"cells/{cell['id']}.json")
        if (rules['cell'] != cell or
                cell['id'] != f.fingerprint({k:v for k,v in cell.items() if k != 'id'})):
            raise ValueError('changed source pair cell')
        totals = []
        for order in ('2', '3'):
            vals, weights = [], []
            for key, weight in rules['rules'][order]:
                row = f.spectrum.read(artifact(source, f'points/{key}/value.json'))
                sample = artifact(source, f'points/{key}/sample.json')
                if (row['plan_sha256'] != f.fingerprint(p) or
                        row['sample_sha256'] != f.shared.sha(sample)):
                    raise ValueError('foreign source pair value')
                vals.append(f.Value.from_record(row['value']))
                weights.append(weight)
            totals.append(f.combine(vals, weights))
        low, high = totals
        results.append(dict(cell=cell, low=low.record(), high=high.record(),
                            moduli_points=sum(len(rules['rules'][n]) for n in ('2','3')),
                            indicator=abs(high.value-low.value)))
    return results


def directional_assessment(totals, acceptance):
    """Sum absolute directional changes to avoid cancellation between checks."""
    value = lambda name: totals[name]['value']['value'][0]
    base = value(acceptance['candidate'])
    deltas = dict(torus_real=value(PROBE)-base,
                  height=value('pair_h4_p3')-base,
                  puncture=value('pair_h4_p4')-value('pair_h4_p3'))
    indicator = math.fsum(abs(v) for v in deltas.values())
    scale = max(abs(base), 1e-300)
    accepted = (indicator/scale <= acceptance['maximum_sum_directional_relative_indicator']
                and abs(deltas['torus_real'])/scale <= acceptance['maximum_real_direction_relative_indicator'])
    return dict(accepted=accepted, candidate=acceptance['candidate'], deltas=deltas,
                indicator=indicator, relative_indicator=indicator/scale,
                rule='sum of absolute changes in torus real part, height and puncture',
                indicator_is_error_bound=False, matrix_model_used=False)


def replace_cell(baseline, cells, cell_id, candidate, indicator, value_type, combine):
    """Replace exactly one cell and independently reassemble the component sum."""
    original = baseline['components'][TERM]
    close(combine(value_type.from_record(c['high']) for c in cells).value.real,
          original['value']['value'][0])
    close(math.fsum(c['indicator'] for c in cells), original['quadrature_indicator'])
    if sum(c['cell']['id'] == cell_id for c in cells) != 1:
        raise ValueError('replacement cell must occur exactly once')
    components = deepcopy(baseline['components'])
    kept = [c for c in cells if c['cell']['id'] != cell_id]
    total = combine([value_type.from_record(c['high']) for c in kept] +
                    [value_type.from_record(candidate['value'])])
    updated_indicator = math.fsum([c['indicator'] for c in kept] + [indicator])
    components[TERM] = dict(original, value=total.record(),
                            quadrature_indicator=updated_indicator,
                            status='targeted_cell_comparison_limit',
                            imported=False, replaced_cell_id=cell_id,
                            quadrature_indicator_method='retained cell indicators plus sum of three directional pair checks')
    components[TERM].pop('moduli_points', None)
    components[TERM]['retained_original_moduli_points'] = sum(c['moduli_points'] for c in kept)
    components[TERM]['candidate_moduli_points'] = candidate['points']
    boundary = combine(value_type.from_record(c['value']) for c in components.values())
    return components, boundary


def prepare_imports(root, baseline_root):
    p = campaign.configure(root)
    f = campaign.f
    d = campaign.checked(root, p)
    baseline_path = baseline_root/'amplitude.json'
    baseline = f.spectrum.read(baseline_path)
    if baseline['source_plan_sha256'] != f.fingerprint(p) or not baseline['complete_domain']:
        raise ValueError('complete amplitude with identical physics is required')
    totals, receipt = rule_values(campaign.parents[0], REFERENCES, f.fingerprint(p))
    cell = d['selected'][TERM]['cell']
    if receipt['cell'] != cell:
        raise ValueError('parent experiment covered a different pair cell')
    cells = original_cells(campaign.source, p)
    # Validate the replacement partition even before the pending probe arrives.
    replace_cell(baseline, cells, cell['id'], totals[REFERENCES[0]], 0., f.Value, f.combine)
    row = dict(design_sha256=f.fingerprint(d), source_plan_sha256=f.fingerprint(p),
               baseline_sha256=f.shared.sha(baseline_path), baseline=baseline,
               original_cells=cells, totals=totals, parent=receipt,
               reducer_source_sha256=f.shared.sha(Path(__file__)))
    f.spectrum.publish(root/'verified_imports.json', row)
    return row


def reduce(root):
    p = campaign.configure(root)
    f = campaign.f
    d = campaign.checked(root, p)
    imports = f.spectrum.read(root/'verified_imports.json')
    if (imports['design_sha256'] != f.fingerprint(d) or
            imports['reducer_source_sha256'] != f.shared.sha(Path(__file__))):
        raise ValueError('changed replacement design or reducer')
    totals, receipt = rule_values(root, (PROBE,), f.fingerprint(p))
    totals.update(imports['totals'])
    if receipt['cell'] != imports['parent']['cell']:
        raise ValueError('probe used a different cell')
    assessment = directional_assessment(totals, d['acceptance'])
    f.spectrum.publish(root/'assessment.json', dict(assessment, design_sha256=f.fingerprint(d),
                                                  totals=totals, probe=receipt))
    if not assessment['accepted']:
        return assessment
    baseline = imports['baseline']
    candidate = totals[assessment['candidate']]
    components, boundary = replace_cell(baseline, imports['original_cells'],
        receipt['cell']['id'], candidate, assessment['indicator'], f.Value, f.combine)
    amplitude = boundary.value.real+baseline['compact']['mean']
    indicator = math.fsum(c['quadrature_indicator'] for c in components.values())
    result = dict(format='genus1-targeted-pair-cell-amplitude-v1',
        design_sha256=f.fingerprint(d), source_plan_sha256=f.fingerprint(p),
        baseline_sha256=imports['baseline_sha256'],
        verified_imports_sha256=f.shared.sha(root/'verified_imports.json'),
        amplitude=amplitude, previous_amplitude=baseline['amplitude'],
        amplitude_change=amplitude-baseline['amplitude'],
        components=components, boundary=boundary.record(), compact=baseline['compact'],
        moduli_indicator=indicator, relative_moduli_indicator=indicator/abs(amplitude),
        boundary_indicator_budget=baseline['boundary_indicator_budget'],
        boundary_indicator_met=indicator<=baseline['boundary_indicator_budget'],
        replacement=dict(cell=receipt['cell'], candidate=candidate, assessment=assessment),
        complete_domain=True, full_amplitude_certified=False, indicator_is_error_bound=False,
        momentum_design_changed=False, physical_cutoffs_changed=False,
        unknown_errors=baseline['unknown_errors'], created_utc=f.shared.utc())
    path = root/'amplitude.json'
    if path.exists():
        result['created_utc'] = f.spectrum.read(path)['created_utc']
    f.spectrum.publish(path, result)
    return {key:result[key] for key in ('amplitude', 'amplitude_change', 'moduli_indicator',
                                      'relative_moduli_indicator', 'full_amplitude_certified')}


if __name__ == '__main__':
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare-imports', 'reduce'))
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--baseline', type=Path)
    args = parser.parse_args()
    if args.action == 'prepare-imports':
        row = prepare_imports(args.root, args.baseline)
        print(json.dumps(dict(verified_rules=list(row['totals']),
                              verified_cells=len(row['original_cells']))))
    else:
        print(json.dumps(reduce(args.root), indent=2))
