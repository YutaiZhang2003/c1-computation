"""Pin saved C14 inputs using bytes and bookkeeping; no numerical replay."""

# Resolve the package's import directories when this script is run directly.
import runpy as _six_point_runpy
from pathlib import Path as _SixPointPath
_six_point_package = next((p for p in _SixPointPath(__file__).resolve().parents
                          if (p / "run.py").is_file() and (p / "PACKAGE_MAP.json").is_file()), None)
if _six_point_package is not None:
    _six_point_runpy.run_path(str(_six_point_package / "run.py"))["bootstrap"]()

from pathlib import Path
import numpy as np

from six_point_refinement_statistics import c18_VERSION as VERSION
from six_point_refinement_statistics import digest
from six_point_refinement_statistics import read_manifest
from six_point_refinement_statistics import load_arrays
from six_point_refinement_statistics import seal
from six_point_refinement_statistics import write_json_once
from six_point_refinement_statistics import c18_gaps_pass as gaps_pass
from six_point_refinement_statistics import array_hash


def preflight(root):
    from six_point_refinement import continuation_source as source
    from six_point_refinement import continuation_verify_deployment as verify_deployment
    from six_point_refinement import continuation_group_data as group_data
    src = source(root)
    verify_deployment(root)
    destination = root / 'inputs/cache_manifest.json'
    if destination.exists():
        existing = read_manifest(destination)
        if existing['source_fingerprint'] != src['fingerprint']:
            raise ValueError('preflight identity changed')
        return {'reused_preflight': existing['fingerprint']}
    parent = Path(src['parent_root'])
    old_state = read_manifest(parent / 'stages/C14/state.json')
    old_results = read_manifest(parent / 'refined/curve_results.json')
    if (old_state['fingerprint'] != src['parent_state_fingerprint']
            or old_results['fingerprint'] != src['parent_results_fingerprint']):
        raise ValueError('C14 parent no longer matches the accepted result')
    selected, cleared = set(), set()
    for uid, expected in src['c14_diagnostic_sha256'].items():
        path = parent / f'stages/C14/diagnostics/{uid}.npz'
        if digest(path) != expected:
            raise ValueError('saved gate evidence changed: ' + uid)
        meta, arrays = load_arrays(path)
        if (old_state['states'][uid]['status'] != 'cap_reached_unconverged'
                or meta['orders'] != [10, 12, 14]
                or not np.isfinite(arrays['totals']).all()
                or arrays['denominator'] != abs(arrays['totals'][-1])
                or complex(*old_state['states'][uid]['new_baseline_I6']) != arrays['totals'][-1]):
            raise ValueError('saved C14 evidence is inconsistent')
        passed, _ = gaps_pass(arrays['numerators'], arrays['denominator'], arrays['totals'][-1])
        (cleared if passed else selected).add(uid)
    unresolved = {uid for uid, st in old_state['states'].items()
                  if st['status'] == 'cap_reached_unconverged'}
    if (selected != set(src['selected_ids']) or cleared != set(src['reclassified_ids'])
            or selected | cleared != unresolved or len(selected) != 450):
        raise ValueError('the 2% selection differs from the saved evidence')
    inventory, target_tables = {}, {}
    for energy in src['energies']:
        meta, table = load_arrays(parent / energy['targets'])
        if meta['fingerprint'] != energy['targets_fingerprint']:
            raise ValueError('saved point targets changed')
        target_tables[energy['label']] = (meta, table)
        point_meta, _ = load_arrays(parent / f"refined/{energy['label']}/pointwise.npz")
        row = next(r for r in old_results['results'] if r['label'] == energy['label'])
        if point_meta['fingerprint'] != row['refined_pointwise_fingerprint']:
            raise ValueError('accepted C14 amplitude changed')
        inventory[f"refined/{energy['label']}/pointwise.npz"] = point_meta['fingerprint']
    banks = sorted({src['states'][uid]['family_id'] for uid in selected})
    for bid in banks:
        relative = 'bank_aux/' + bid + '.npz'
        meta, arrays = load_arrays(parent / relative)
        if meta['parent_fingerprint'] != src['banks'][bid]['parent']['fingerprint']:
            raise ValueError('spectral rule identity differs')
        if any(not np.isfinite(v).all() for v in arrays.values()):
            raise ValueError('nonfinite saved spectral rule')
        inventory[relative] = meta['fingerprint']
    groups = read_manifest(root / 'groups/manifest.json')
    rows = 0
    for desc in groups['groups'].values():
        needed = [bid for bid in desc['banks'] if bid in banks]
        if not needed:
            continue
        data = group_data(root, desc)
        for bid in needed:
            _, aux = load_arrays(parent / ('bank_aux/' + bid + '.npz'))
            scatter = data['scatter_' + bid]
            if not np.array_equal(aux['p_values'].view(np.uint64), data['p_values'][scatter].view(np.uint64)):
                raise ValueError('saved node scatter differs')
        rows += len(np.unique(np.concatenate([data['scatter_' + bid] for bid in needed])))
    for uid in sorted(selected):
        st = src['states'][uid]
        tmeta, targets = target_tables[st['energy']]
        j, bid = st['target_index'], st['family_id']
        if targets['ids'][j] != uid or targets['family_ids'][j] != bid:
            raise ValueError('target index mismatch')
        expected = dict(uid=uid, parent_fingerprint=src['banks'][bid]['parent']['fingerprint'],
                        source_fingerprint=src['parent_source_fingerprint'],
                        target_fingerprint=tmeta['fingerprint'], q_hash=array_hash(targets['q'][j]))
        for n in (11, 12, 13, 14):
            relative = f'contractions/{uid}/C{n}.npz'
            meta, arrays = load_arrays(parent / relative)
            if (meta['order'] != n or any(meta['context'].get(k) != v for k, v in expected.items())
                    or arrays['chiral'].ndim != 2 or arrays['chiral'].shape[1] != 2
                    or not np.isfinite(arrays['chiral']).all()):
                raise ValueError('saved contraction identity differs: ' + relative)
            inventory[relative] = meta['fingerprint']
    result = seal(dict(version=VERSION, source_fingerprint=src['fingerprint'],
        arrays=inventory, selected_count=len(selected), reclassified_count=len(cleared),
        needed_banks=len(banks), new_C15_node_rows=rows,
        new_coefficient_evaluations=0, new_contractions=0, new_momentum_integrations=0,
        reused_parent_outputs=True))
    write_json_once(destination, result)
    return {k: result[k] for k in ('fingerprint', 'selected_count', 'reclassified_count',
                                  'needed_banks', 'new_C15_node_rows')}
