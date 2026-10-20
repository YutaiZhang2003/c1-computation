"""Apply selected-point corrections and assemble the accepted C14 and C18 amplitudes."""
from __future__ import annotations
import argparse
import json
import traceback
import numpy as np
from six_point_refinement_statistics import STATUSES
from six_point_refinement_statistics import apply_corrections

from pathlib import Path as _SixPointPath
from six_point_refinement_statistics import json_hash
from pathlib import Path
from decimal import Decimal
from six_point_refinement_statistics import digest
from six_point_refinement_statistics import read_manifest
from six_point_refinement_statistics import seal
from six_point_refinement_statistics import write_json_once
from six_point_refinement_statistics import load_arrays
from six_point_refinement_statistics import save_arrays
from six_point_refinement_statistics import paired_statistics
from six_point_refinement_statistics import point_slots
from collections import Counter
import os
from six_point_refinement_statistics import gaps_pass
from six_point_refinement_statistics import c18_VERSION as C18_VERSION, c18_STATUSES as C18_STATUSES


# apply_pointwise_amplitude_corrections
def pair(value):
    return [float(complex(value).real), float(complex(value).imag)]


def patch_pointwise(original, slots, changes, statuses, orders, source_fingerprint, state_fingerprint):
    """Preserve old diagnostics under explicit original names; patch full estimators."""
    arrays = {k: v.copy() for k, v in original.items()}
    old_diagnostics = ('raw_native_I6', 'treated_prefix_I6', 'direct_high_native_I6',
        'mixed_defect_I6', 'mixed_defect_assessed', 'native_relative_absolute_increments',
        'native_absolute_increments_I6', 'native_normalization_denominator_I6',
        'zero_absolute_weight_node_counts', 'baseline_prefix_pass', 'available_prefix_pass',
        'block_prefix_pass', 'orders', 'available_orders')
    for key in old_diagnostics:
        if key in arrays:
            arrays['original_' + key] = arrays.pop(key)
    delta = None
    for key in ('multilevel_native_I6', 'routed_full_I6', 'hybrid_I6'):
        patched, delta = apply_corrections(original[key][0], slots, changes)
        arrays[key] = patched[None]
    patched, _ = apply_corrections(original['baseline_correction_I6'][0], slots, changes)
    arrays['baseline_correction_I6'] = patched[None]
    status_grid = np.full((8, 256), 'not_targeted', dtype='<U40')
    order_grid = np.full((8, 256), 12, dtype=np.int64)
    refined_pass = original['baseline_prefix_pass'].copy()
    direct_new_available = np.zeros((1, 8, 256, len(original['variants'])), dtype=bool)
    for (r, n), status, order in zip(slots, statuses, orders):
        if status not in STATUSES or order not in range(12, 17):
            raise ValueError('unresolved numerical failure or incomplete stopping status')
        status_grid[r, n], order_grid[r, n] = status, order
        refined_pass[0, r, n] = status != 'cap_reached_unconverged'
    arrays.update(refinement_status=status_grid[None], refined_point_order=order_grid[None],
        refined_baseline_prefix_pass=refined_pass, refinement_delta_I6=delta[None],
        new_direct_control_comparison_available=direct_new_available,
        source_manifest_fingerprint=np.asarray(source_fingerprint),
        stopping_state_fingerprint=np.asarray(state_fingerprint))
    target_mask = np.zeros((8, 256), dtype=bool)
    for r, n in slots:
        target_mask[r, n] = True
    for key in ('multilevel_native_I6', 'routed_full_I6', 'hybrid_I6', 'baseline_correction_I6'):
        if not np.array_equal(arrays[key][0][~target_mask], original[key][0][~target_mask], equal_nan=True):
            raise ValueError('untargeted value changed during reassembly')
    for key in original:
        if key.startswith('collar_') and not np.array_equal(original[key], arrays[key], equal_nan=True):
            raise ValueError('saved collar or addback changed')
    if not np.isfinite(arrays['hybrid_I6']).all():
        raise ValueError('nonfinite refined full pointwise value')
    return arrays, delta


def covariance_rows(replicates):
    values = np.asarray(replicates, complex)
    if values.shape != (15, 8) or not np.isfinite(values).all():
        raise ValueError('fifteen complete common-scramble rows required')
    return dict(real=(np.cov(values.real, ddof=1) / 8).tolist(),
                imag=(np.cov(values.imag, ddof=1) / 8).tolist(),
                real_imag_cross=(np.cov(np.concatenate((values.real, values.imag)), ddof=1)[:15, 15:] / 8).tolist())


# stopping_policy
_six_point_package = next((p for p in _SixPointPath(__file__).resolve().parents
                          if (p / "run.py").is_file() and (p / "PACKAGE_MAP.json").is_file()), None)


C14_VERSION = 'sphere-six-point-selective-c14-cap-v1'


C14_CAP = 14


C14_STATUSES = ('passes_at_12_under_0p5pct', 'converged_at_13',
            'converged_at_14', 'cap_reached_unconverged')


def validate_final_state(state, source):
    if (state.get('fingerprint') != json_hash({k: v for k, v in state.items() if k != 'fingerprint'})
            or state.get('effective_cap') != C14_CAP or state.get('order') != C14_CAP
            or state.get('source_fingerprint') != source['fingerprint']
            or not state.get('cap_policy_fingerprint')
            or set(state['states']) != set(source['states'])
            or len(state['states']) != source['original_flagged']):
        raise ValueError('complete, sealed C14 stopping state required before assembly')
    for record in state['states'].values():
        status = record['status']
        expected = 12 if status == C14_STATUSES[0] else 13 if status == C14_STATUSES[1] else C14_CAP
        if status not in C14_STATUSES or record.get('final_order') != expected:
            raise ValueError('unfinished or higher-order state cannot enter C14 assembly')
    counts = {s: sum(r['status'] == s for r in state['states'].values()) for s in C14_STATUSES}
    if counts != state['counts']:
        raise ValueError('C14 stopping counts disagree with complete point records')


def report_policy(state):
    return dict(effective_cap=C14_CAP, original_planned_cap=16,
                cap_policy_fingerprint=state['cap_policy_fingerprint'],
                higher_order_refinement_deferred=True,
                cap_reason='User compute budget; refinement above C14 is future work.')


# assemble_final_amplitudes
def assemble_c14(root, state):
    root = Path(root)
    src = read_manifest(root / 'inputs/source_manifest.json')
    validate_final_state(state, src)
    policy = read_manifest(root / 'cap14/cap_policy.json')
    if state['cap_policy_fingerprint'] != policy['fingerprint']:
        raise ValueError('final state belongs to a different cap policy')
    policy_fields = report_policy(state)
    results, old_rows, new_rows, shift_rows, q_rows = [], [], [], [], []
    for energy in src['energies']:
        label, t = energy['label'], energy['t']
        path = Path(energy['original_pointwise'])
        if digest(path) != energy['original_pointwise_sha256']:
            raise ValueError('original pointwise output changed')
        with np.load(path, allow_pickle=False) as z:
            original = {k: z[k] for k in z.files}
        point_slots(original['original_replicate_ids'], original['original_point_indices'])
        meta, targets = load_arrays(root / energy['targets'])
        if meta['fingerprint'] != energy['targets_fingerprint']:
            raise ValueError('frozen target manifest changed')
        changes, statuses, orders, final_native = [], [], [], []
        for j, uid in enumerate(targets['ids']):
            record = state['states'][str(uid)]
            old = complex(targets['original_C12'][j])
            if record['status'] == C14_STATUSES[0]:
                value, order = old, 12
            else:
                decision = read_manifest(root / record['decision'])
                if decision['uid'] != uid or decision['status'] != record['status']:
                    raise ValueError('final stopping record/decision mismatch')
                value = complex(*decision['new_baseline_I6'])
                order = decision['order']
                if (order != record['final_order'] or order > C14_CAP
                        or (order == C14_CAP and decision.get('cap_policy_fingerprint') != policy['fingerprint'])):
                    raise ValueError('decision is outside the final C14 policy')
            if not np.isfinite(value):
                raise ValueError('numerical failure cannot become a zero or dropped point')
            changes.append(value - old)
            statuses.append(record['status'])
            orders.append(order)
            final_native.append(value)
        changes = np.asarray(changes, complex)
        arrays, delta = patch_pointwise(original, targets['slots'], changes, statuses, orders,
                                        src['fingerprint'], state['fingerprint'])
        point_meta = save_arrays(root / f'refined/{label}/pointwise.npz', arrays,
            dict(version=C14_VERSION, **policy_fields, source_pointwise_sha256=energy['original_pointwise_sha256'],
                source_manifest_fingerprint=src['fingerprint'], stopping_state_fingerprint=state['fingerprint'],
                original_diagnostics_prefix='original_', direct_new_control_values_computed=False,
                addback_reused_exactly_once=True, original_sample_count=2048))
        old_q = np.asarray([complex(*p) for p in energy['original_replicate_means_Q5']])
        old_i6 = old_q * (-40 * np.pi**3 * t**6)
        q_scale = -1 / (40 * np.pi**3 * t**6)
        a_scale = 1j / (8 * np.pi**3)
        i6_stats, new_i6, shifts_i6 = paired_statistics(old_i6, delta)
        q_stats, new_q, shifts_q = paired_statistics(old_q, delta * q_scale)
        old_a = old_q * (-5j * t**6)
        a_stats, new_a, shifts_a = paired_statistics(old_a, delta * a_scale)
        old_rows.append(old_a); new_rows.append(new_a); shift_rows.append(shifts_a); q_rows.append(new_q)
        counts = {name: statuses.count(name) for name in C14_STATUSES}
        cap = np.asarray([status == 'cap_reached_unconverged' for status in statuses])
        capped_native = np.asarray(final_native)[cap]
        capped_slots = targets['slots'][cap]
        b = original['variants'].tolist().index('baseline')
        capped_full = np.asarray([arrays['hybrid_I6'][0, r, n, b] for r, n in capped_slots])
        cap_contribution = dict(count=int(cap.sum()),
            signed_native_I6=pair(capped_native.sum() / 2048),
            absolute_native_I6=float(abs(capped_native).sum() / 2048),
            signed_mu4_A=pair(capped_native.sum() / 2048 * a_scale),
            absolute_mu4_A=float(abs(capped_native).sum() / 2048 * abs(a_scale)),
            signed_full_slot_I6=pair(capped_full.sum() / 2048),
            absolute_full_slot_I6=float(abs(capped_full).sum() / 2048),
            paired_capped_shift_mu4_A=pair(changes[cap].sum() / 2048 * a_scale),
            native_contribution_includes_source_density_and_momentum_weights=True,
            full_slot_contribution_also_includes_original_uniform_addback=True)
        row = dict(t=t, label=label, **policy_fields, I6=i6_stats, Q5=q_stats, mu4_A=a_stats,
            stopping_counts=counts, original_flagged=len(statuses),
            cap_reached_unconverged_contributions=cap_contribution,
            original_replicate_means_mu4_A=[pair(x) for x in old_a],
            refined_replicate_means_mu4_A=[pair(x) for x in new_a],
            paired_replicate_shifts_mu4_A=[pair(x) for x in shifts_a],
            refined_pointwise_fingerprint=point_meta['fingerprint'],
            original_pointwise_sha256=energy['original_pointwise_sha256'],
            sample_count=2048, points_per_scramble=256, scrambles=8,
            controls_updated_by_same_baseline_correction=True, original_addback_preserved=True)
        # This prediction is reached only after all actual stopping decisions are sealed.
        dt = Decimal(str(t))
        model_q = float((1-5*dt)*(2-5*dt)*(3-5*dt))
        row['matrix_model_comparison_after_frozen_decisions'] = dict(Q5=model_q,
            mu4_A=pair(-5j*t**6*model_q), frozen_state_fingerprint=state['fingerprint'])
        write_json_once(root / f'refined/{label}/summary.json', seal(dict(version=C14_VERSION, **row)))
        results.append(row)
    report = seal(dict(version=C14_VERSION, **policy_fields, complete_execution=True, energies=[e['t'] for e in src['energies']],
        results=results, source_manifest_fingerprint=src['fingerprint'], stopping_state_fingerprint=state['fingerprint'],
        amplitude_convention='A=i*I6/(8*pi^3*mu^4)=-5*i*t^6*Q5/mu^4; tables report mu^4*A',
        stopping_counts={name: sum(row['stopping_counts'][name] for row in results) for name in C14_STATUSES},
        original_mean_covariance_mu4_A=covariance_rows(old_rows), refined_mean_covariance_mu4_A=covariance_rows(new_rows),
        paired_shift_mean_covariance_mu4_A=covariance_rows(shift_rows), refined_mean_covariance_Q5=covariance_rows(q_rows),
        covariance_rank_bound=7, covariance_note='Common original scramble IDs; fifteen-energy covariance is singular with eight scrambles.',
        regenerated_parent_bank_outputs=0, rerun_original_amplitudes=0,
        new_direct_high_order_control_comparisons_available=False, total_accuracy_certified=False,
        uncertainty_note='Reported SEs are scramble sampling errors. The practical prefix rule does not bound all CFT or momentum tails. Finite capped failures remain included.'))
    write_json_once(root / 'refined/curve_results.json', report)
    lines = ['# Selective C14 full-amplitude refinement', '',
        'C14 is the final compute cap. Higher-order refinement is deferred to future work; finite unresolved C14 values remain included and flagged.', '',
        'All entries report the imaginary component of mu^4 A; complete complex values and component SEs are in curve_results.json. Paired shifts use the original eight common scrambles.', '',
        '| t | Original Im(mu^4 A) ± SE | Refined Im(mu^4 A) ± SE | Paired shift ± SE | C12 / C13 / C14 / capped at C14 |',
        '|---|---:|---:|---:|---|']
    for row in results:
        a = row['mu4_A']
        cells = [f'{a[k]["mean"][1]:.10g} ± {a[k]["se"][1]:.4g}' for k in ('original', 'refined', 'shift')]
        counts = ' / '.join(str(row['stopping_counts'][k]) for k in C14_STATUSES)
        lines.append(f'| {row["t"]:.4f} | ' + ' | '.join(cells) + ' | ' + counts + ' |')
    lines += ['', 'Capped finite values remain in the amplitudes with unresolved flags. Sampling SEs do not certify total accuracy. Original controls, source samples, weights and analytic addback are preserved. No parent-bank outputs or original amplitudes were regenerated.', '']
    path = root / 'refined/full_amplitudes.md'
    text = '\n'.join(lines)
    if path.exists() and path.read_text() != text:
        raise ValueError('existing final report differs')
    if not path.exists():
        path.write_text(text)
    write_json_once(root / 'completion.json', seal(dict(version=C14_VERSION, **policy_fields, complete=True,
        results_fingerprint=report['fingerprint'], stopping_state_fingerprint=state['fingerprint'],
        energy_count=15, original_flagged=src['original_flagged'], stopping_counts=report['stopping_counts'],
        regenerated_parent_bank_outputs=0, rerun_original_amplitudes=0)))
    return report


# finalize_refinement_and_assemble
def build_final_state(root, source, stage, previous, policy):
    """Read existing diagnostics only; preserve immutable worker decisions as evidence."""
    root = Path(root)
    labels = [e['label'] for e in source['energies']]
    if (policy['effective_cap'] != C14_CAP or policy['original_planned_cap'] != 16
            or policy['tolerance'] != .005 or source['tolerance'] != .005
            or source['base_order'] != 12 or source['cap'] != 16
            or len(labels) != 15 or len(set(labels)) != 15 or labels != policy['energy_labels']
            or source['fingerprint'] != policy['source_fingerprint']
            or previous['fingerprint'] != policy['prior_state_fingerprint']
            or previous['source_fingerprint'] != source['fingerprint'] or previous['order'] != 13
            or stage['fingerprint'] != policy['stage_fingerprint']
            or stage['source_fingerprint'] != source['fingerprint']
            or stage['prior_state'] != previous['fingerprint']
            or stage['order'] != C14_CAP or stage['window'] != [10, 12, 14]
            or stage['coefficient_degrees'] != [C14_CAP]):
        raise ValueError('C14 cap policy or frozen stage identity mismatch')
    if any((root / f'stages/C{n}').exists() for n in (15, 16)):
        raise ValueError('unexpected higher-order stage; reconcile before C14 finalization')
    before = previous['states']
    if (set(before) != set(source['states'])
            or len(before) != source['original_flagged']
            or len(before) != policy['original_flagged']
            or Counter(r['energy'] for r in before.values()) !=
                Counter({e['label']: e['original_flagged'] for e in source['energies']})):
        raise ValueError('all original targets across all fifteen energies are required')
    active = {uid for uid, record in before.items() if record['status'] == 'needs_14'}
    ids = stage['active_point_ids']
    evaluation_ids = [uid for task in stage['evaluations'] for uid in task['target_ids']]
    if (len(ids) != len(set(ids)) or set(ids) != active or len(active) != policy['active_points']
            or len(evaluation_ids) != len(set(evaluation_ids)) or set(evaluation_ids) != active):
        raise ValueError('C14 decision barrier has missing or duplicate active targets')
    states = {uid: dict(record) for uid, record in before.items()}
    for uid, record in before.items():
        if record['status'] not in (*C14_STATUSES[:2], 'needs_14'):
            raise ValueError('unexpected stopping status before C14')
        if uid not in active:
            order = 12 if record['status'] == C14_STATUSES[0] else 13
            if record.get('final_order') != order:
                raise ValueError('saved earlier stopping order changed')
    overlays = {}
    for uid in ids:
        decision_path = f'stages/C14/decisions/{uid}.json'
        d = read_manifest(root / decision_path)
        diagnostic_path = f'stages/C14/diagnostics/{uid}.npz'
        if d['diagnostic'] != diagnostic_path:
            raise ValueError('unexpected C14 diagnostic path')
        meta, arrays = load_arrays(root / diagnostic_path)
        totals, numerators = arrays['totals'], arrays['numerators']
        denominator = arrays['denominator']
        if (d['uid'] != uid or d['order'] != C14_CAP
                or d['bank_id'] != before[uid]['family_id']
                or d['stage_fingerprint'] != stage['fingerprint']
                or meta['fingerprint'] != d['diagnostic_fingerprint']
                or meta['uid'] != uid or meta['order'] != C14_CAP
                or meta['stage_fingerprint'] != stage['fingerprint']
                or meta['orders'] != [10, 12, 14]
                or totals.shape != (3,) or not np.isfinite(totals).all()
                or numerators.shape != (2,) or denominator.shape != ()
                or arrays['relative'].shape != (2,)
                or denominator != abs(totals[-1])
                or complex(*d['new_baseline_I6']) != totals[-1]):
            raise ValueError('invalid or nonfinite C14 point decision')
        passed, relative = gaps_pass(numerators, denominator, totals[-1])
        expected_worker_status = 'converged_at_14' if passed else 'needs_15'
        relative_json = [float(x) if np.isfinite(x) else None for x in relative]
        if (d['passed'] != bool(passed) or d['status'] != expected_worker_status
                or d['relative_gaps'] != relative_json
                or not np.array_equal(relative, arrays['relative'], equal_nan=True)):
            raise ValueError('C14 decision violates the unchanged strict two-gap rule')
        status = 'converged_at_14' if passed else 'cap_reached_unconverged'
        overlay_path = f'cap14/decisions/{uid}.json'
        overlay = {k: v for k, v in d.items() if k != 'fingerprint'}
        overlay.update(version=C14_VERSION, status=status, effective_cap=C14_CAP,
            cap_policy_fingerprint=policy['fingerprint'],
            original_worker_decision=decision_path,
            original_worker_decision_fingerprint=d['fingerprint'], original_worker_status=d['status'],
            higher_order_refinement_deferred=not bool(passed))
        overlays[overlay_path] = seal(overlay)
        states[uid].update(status=status, last_order=C14_CAP, final_order=C14_CAP,
                          new_baseline_I6=d['new_baseline_I6'], decision=overlay_path)
    counts = {s: sum(r['status'] == s for r in states.values()) for s in C14_STATUSES}
    state = seal(dict(version=C14_VERSION, order=C14_CAP, effective_cap=C14_CAP, original_planned_cap=16,
        stage_fingerprint=stage['fingerprint'], prior_state_fingerprint=previous['fingerprint'],
        source_fingerprint=source['fingerprint'], cap_policy_fingerprint=policy['fingerprint'],
        states=states, counts=counts, higher_order_refinement_deferred=True))
    validate_final_state(state, source)
    # No final state or amplitude is written until every required decision was checked.
    for path, overlay in overlays.items():
        write_json_once(root / path, overlay)
    write_json_once(root / 'stages/C14/state.json', state)
    return state


def finalize(root, expected_policy_fingerprint):
    root = Path(root).resolve()
    policy = read_manifest(root / 'cap14/cap_policy.json')
    if policy['fingerprint'] != expected_policy_fingerprint:
        raise ValueError('submitted C14 policy changed')
    deployment = read_manifest(root / 'deployment.json')
    if deployment['fingerprint'] != policy['original_deployment_fingerprint']:
        raise ValueError('original deployment identity changed')
    for relative, expected in {**deployment['files'], **policy['files']}.items():
        if digest(root / relative) != expected:
            raise ValueError('pinned code or saved input changed: ' + relative)
    source = read_manifest(root / 'inputs/source_manifest.json')
    gate = read_manifest(root / 'inputs/completion_gate.json')
    if not gate['passed'] or gate['fingerprint'] != source['gate']:
        raise ValueError('original completion gate changed')
    stage = read_manifest(root / 'stages/C14/manifest.json')
    previous = read_manifest(root / 'stages/C13/state.json')
    state = build_final_state(root, source, stage, previous, policy)
    assemble = assemble_c14
    report = assemble(root, state)
    result = dict(effective_cap=C14_CAP, stopping_counts=state['counts'],
                  stopping_state_fingerprint=state['fingerprint'],
                  results_fingerprint=report['fingerprint'], higher_order_jobs_submitted=0)
    write_json_once(root / f'progress/C14/advance-{os.environ.get("SLURM_JOB_ID", "local")}-0.json', result)
    return result


# assemble_amplitudes_from_c14_corrections
def c18_pair(value):
    return [float(complex(value).real), float(complex(value).imag)]


def c18_covariance_rows(replicates):
    values = np.asarray(replicates, complex)
    if values.shape != (15, 8) or not np.isfinite(values).all():
        raise ValueError('fifteen complete common-scramble rows required')
    return dict(real=(np.cov(values.real, ddof=1) / 8).tolist(),
                imag=(np.cov(values.imag, ddof=1) / 8).tolist(),
                real_imag_cross=(np.cov(np.concatenate((values.real, values.imag)), ddof=1)[:15, 15:] / 8).tolist())


def patch_c14_pointwise(original, slots, changes, all_slots, records, source_fp, state_fp):
    arrays = {k: v.copy() for k, v in original.items()}
    for key in ('refinement_status', 'refined_point_order', 'refined_baseline_prefix_pass',
                'refinement_delta_I6', 'source_manifest_fingerprint', 'stopping_state_fingerprint'):
        arrays['c14_' + key] = arrays[key].copy()
    for key in ('multilevel_native_I6', 'routed_full_I6', 'hybrid_I6', 'baseline_correction_I6'):
        patched, delta = apply_corrections(original[key][0], slots, changes)
        arrays[key] = patched[None]
    for (r, n), record in zip(all_slots, records):
        if record['status'] not in C18_STATUSES or record['refinement_required']:
            raise ValueError('amplitude cannot be assembled while a selected point is active')
        if record['status'] == 'cap_reached_unconverged' and record['final_order'] != 18:
            raise ValueError('only finite C18 failures can stop at the new cap')
        arrays['refinement_status'][0, r, n] = record['status']
        arrays['refined_point_order'][0, r, n] = record['final_order']
        arrays['refined_baseline_prefix_pass'][0, r, n] = record['convergence_pass']
    arrays['active_refinement_flag'] = np.zeros((1, 8, 256), dtype=bool)
    arrays['continuation_delta_I6'] = delta[None]
    arrays['refinement_delta_I6'] = original['refinement_delta_I6'] + delta[None]
    arrays['source_manifest_fingerprint'] = np.asarray(source_fp)
    arrays['stopping_state_fingerprint'] = np.asarray(state_fp)
    mask = np.zeros((8, 256), dtype=bool)
    for r, n in slots:
        mask[r, n] = True
    for key in ('multilevel_native_I6', 'routed_full_I6', 'hybrid_I6', 'baseline_correction_I6'):
        if not np.array_equal(original[key][0][~mask], arrays[key][0][~mask], equal_nan=True):
            raise ValueError('a nonselected numerical value changed')
    mutable = {'multilevel_native_I6', 'routed_full_I6', 'hybrid_I6', 'baseline_correction_I6',
               'refinement_status', 'refined_point_order', 'refined_baseline_prefix_pass',
               'refinement_delta_I6', 'source_manifest_fingerprint', 'stopping_state_fingerprint'}
    for key in original.keys() - mutable:
        if original[key].dtype.kind in 'fc':
            equal = np.array_equal(original[key], arrays[key], equal_nan=True)
        else:
            equal = np.array_equal(original[key], arrays[key])
        if not equal:
            raise ValueError('saved samples, control evidence, or addback changed: ' + key)
    if not np.isfinite(arrays['hybrid_I6']).all():
        raise ValueError('nonfinite full amplitude; points must not be dropped')
    return arrays, delta


def assemble_c18(root, state):
    from six_point_refinement import continuation_source as source
    from six_point_refinement import continuation_verify_deployment as verify_deployment
    root = Path(root)
    src = source(root)
    verify_deployment(root)
    if (state['source_fingerprint'] != src['fingerprint']
            or set(state['states']) != set(src['states'])
            or any(st['status'] not in C18_STATUSES or st['refinement_required'] for st in state['states'].values())):
        raise ValueError('all actual stopping decisions must be frozen')
    parent = Path(src['parent_root'])
    old_report = read_manifest(parent / 'refined/curve_results.json')
    if old_report['fingerprint'] != src['parent_results_fingerprint']:
        raise ValueError('C14 result identity changed')
    cache = read_manifest(root / 'inputs/cache_manifest.json')
    results, old_rows, new_rows, shift_rows, q_rows = [], [], [], [], []
    selected = set(src['selected_ids'])
    a_scale = 1j / (8 * np.pi**3)
    for energy in src['energies']:
        label, t = energy['label'], energy['t']
        relative = f'refined/{label}/pointwise.npz'
        old_meta, original = load_arrays(parent / relative)
        if old_meta['fingerprint'] != cache['arrays'][relative]:
            raise ValueError('accepted pointwise amplitude changed')
        point_slots(original['original_replicate_ids'], original['original_point_indices'])
        target_meta, targets = load_arrays(parent / energy['targets'])
        if target_meta['fingerprint'] != energy['targets_fingerprint']:
            raise ValueError('saved targets changed')
        changes, slots, final_native, selected_records = [], [], [], []
        all_records = [state['states'][str(uid)] for uid in targets['ids']]
        for j, target_uid in enumerate(targets['ids']):
            uid = str(target_uid)
            if uid not in selected:
                continue
            record = state['states'][uid]
            decision = read_manifest(root / record['decision'])
            if (decision['uid'] != uid or decision['status'] != record['status']
                    or decision['order'] != record['final_order']
                    or decision['passed'] != record['convergence_pass']
                    or decision['new_baseline_I6'] != record['new_baseline_I6']):
                raise ValueError('final point decision differs from the frozen state')
            value = complex(*decision['new_baseline_I6'])
            if not np.isfinite(value):
                raise ValueError('nonfinite selected value cannot be omitted')
            old_value = complex(*src['states'][uid]['new_baseline_I6'])
            changes.append(value - old_value)
            slots.append(targets['slots'][j])
            final_native.append(value)
            selected_records.append(record)
        changes, final_native = np.asarray(changes, complex), np.asarray(final_native, complex)
        arrays, delta = patch_c14_pointwise(original, slots, changes, targets['slots'], all_records,
                                           src['fingerprint'], state['fingerprint'])
        point_meta = save_arrays(root / relative, arrays,
            dict(version=C18_VERSION, parent_c14_pointwise_fingerprint=old_meta['fingerprint'],
                 source_manifest_fingerprint=src['fingerprint'], stopping_state_fingerprint=state['fingerprint'],
                 original_sample_count=2048, addback_reused_exactly_once=True,
                 corrections_relative_to='accepted C14 mixed-order pointwise amplitude',
                 direct_new_control_values_computed=False))
        old_row = next(r for r in old_report['results'] if r['label'] == label)
        old_a = np.asarray([complex(*p) for p in old_row['refined_replicate_means_mu4_A']])
        old_i6 = old_a / a_scale
        q_scale = -1 / (40 * np.pi**3 * t**6)
        i6_stats, new_i6, _ = paired_statistics(old_i6, delta)
        q_stats, new_q, _ = paired_statistics(old_i6 * q_scale, delta * q_scale)
        a_stats, new_a, shifts_a = paired_statistics(old_a, delta * a_scale)
        b = original['variants'].tolist().index('baseline')
        direct_old = original['hybrid_I6'][0, :, :, b].mean(axis=1) * a_scale
        direct_new = arrays['hybrid_I6'][0, :, :, b].mean(axis=1) * a_scale
        if (not np.allclose(direct_old, old_a, rtol=2e-12, atol=1e-18)
                or not np.allclose(direct_new, new_a, rtol=2e-12, atol=1e-18)
                or not np.allclose(new_a, new_i6 * a_scale, rtol=2e-12, atol=1e-18)):
            raise ValueError('paired reconstruction disagrees with the stored full pointwise amplitude')
        capped = np.asarray([r['status'] == 'cap_reached_unconverged' for r in selected_records])
        capped_full = np.asarray([arrays['hybrid_I6'][0, r, n, b]
                                  for (r, n), use in zip(slots, capped) if use])
        contributions = dict(count=int(capped.sum()),
            signed_native_mu4_A=c18_pair(final_native[capped].sum() / 2048 * a_scale),
            absolute_native_mu4_A=float(abs(final_native[capped]).sum() / 2048 * abs(a_scale)),
            signed_full_slot_mu4_A=c18_pair(capped_full.sum() / 2048 * a_scale),
            absolute_full_slot_mu4_A=float(abs(capped_full).sum() / 2048 * abs(a_scale)),
            signed_shift_from_C14_mu4_A=c18_pair(changes[capped].sum() / 2048 * a_scale))
        row = dict(t=t, label=label, I6=i6_stats, Q5=q_stats, mu4_A=a_stats,
            comparison_baseline='accepted C14 final amplitude', effective_cap=18, tolerance=0.02,
            original_flagged=len(all_records), selected_for_continuation=len(changes),
            stopping_counts=dict(Counter(r['status'] for r in all_records)),
            continuation_stopping_counts=dict(Counter(r['status'] for r in selected_records)),
            active_refinement_flags=0, cap_reached_unconverged_contributions=contributions,
            c14_replicate_means_mu4_A=[c18_pair(x) for x in old_a],
            refined_replicate_means_mu4_A=[c18_pair(x) for x in new_a],
            paired_replicate_shifts_mu4_A=[c18_pair(x) for x in shifts_a],
            refined_pointwise_fingerprint=point_meta['fingerprint'],
            c14_pointwise_fingerprint=old_meta['fingerprint'],
            sample_count=2048, points_per_scramble=256, scrambles=8,
            controls_updated_by_same_baseline_correction=True, original_addback_preserved=True,
            matrix_model_comparison_after_frozen_decisions=dict(
                old_row['matrix_model_comparison_after_frozen_decisions'], frozen_state_fingerprint=state['fingerprint']))
        write_json_once(root / f'refined/{label}/summary.json', seal(dict(version=C18_VERSION, **row)))
        results.append(row)
        old_rows.append(old_a); new_rows.append(new_a); shift_rows.append(shifts_a); q_rows.append(new_q)
    counts = dict(Counter(st['status'] for st in state['states'].values()))
    selected_counts = dict(Counter(state['states'][uid]['status'] for uid in selected))
    report = seal(dict(version=C18_VERSION, complete_execution=True, results=results,
        source_manifest_fingerprint=src['fingerprint'], stopping_state_fingerprint=state['fingerprint'],
        parent_results_fingerprint=old_report['fingerprint'], tolerance=0.02, cap=18,
        amplitude_convention='mu^4 A = i I6/(8 pi^3) = -5 i t^6 Q5',
        stopping_counts=counts, continuation_stopping_counts=selected_counts,
        selected_count=450, sample_energy_evaluations=30720, active_refinement_flags=0,
        cap_reached_unconverged=counts.get('cap_reached_unconverged', 0),
        c14_mean_covariance_mu4_A=c18_covariance_rows(old_rows),
        refined_mean_covariance_mu4_A=c18_covariance_rows(new_rows),
        paired_shift_mean_covariance_mu4_A=c18_covariance_rows(shift_rows),
        refined_mean_covariance_Q5=c18_covariance_rows(q_rows), covariance_rank_bound=7,
        regenerated_parent_bank_outputs=0, rerun_original_amplitudes=0,
        new_direct_high_order_control_comparisons_available=False,
        uncertainty_note='SEs measure scramble sampling error. Passing two 2% prefix gaps is not a tail bound. '
                         'Finite C18 failures are included and remain distinguishable despite clearing the active flag.'))
    write_json_once(root / 'refined/curve_results.json', report)
    lines = ['# Amplitude refinement at 2%, capped at C18', '',
             '| t | C14 Im(mu^4 A) ± SE | Refined Im(mu^4 A) ± SE | Paired shift ± SE | C18 unresolved |',
             '|---|---:|---:|---:|---:|']
    for row in results:
        cells = [f'{row["mu4_A"][k]["mean"][1]:.10g} ± {row["mu4_A"][k]["se"][1]:.4g}'
                 for k in ('original', 'refined', 'shift')]
        lines.append(f'| {row["t"]:.4f} | ' + ' | '.join(cells) +
                     f' | {row["cap_reached_unconverged_contributions"]["count"]} |')
    lines += ['', report['uncertainty_note'], '',
              'Only the 450 selected point values were updated. All original samples, controls, weights, '
              'accepted nonselected values and the analytic addback are retained. No manuscript or figure was changed.', '']
    path = root / 'refined/full_amplitudes.md'
    content = '\n'.join(lines)
    if path.exists() and path.read_text() != content:
        raise ValueError('existing amplitude report differs')
    if not path.exists():
        path.write_text(content)
    write_json_once(root / 'completion.json', seal(dict(version=C18_VERSION, complete=True,
        results_fingerprint=report['fingerprint'], stopping_state_fingerprint=state['fingerprint'],
        energy_count=15, selected_count=450, continuation_stopping_counts=selected_counts,
        active_refinement_flags=0, cap_reached_unconverged=report['cap_reached_unconverged'],
        regenerated_parent_bank_outputs=0, rerun_original_amplitudes=0)))
    return {'complete': True, 'fingerprint': report['fingerprint'], 'stopping_counts': selected_counts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--policy-fingerprint', required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(finalize(args.root, args.policy_fingerprint), indent=2), flush=True)
    except Exception:
        write_json_once(args.root / f'failures/C14-finalize-{os.environ.get("SLURM_JOB_ID", "local")}-0.json',
                        dict(traceback=traceback.format_exc(), effective_cap=C14_CAP,
                             accepted_outputs_retained=True, higher_order_jobs_submitted=0))
        raise


if __name__ == "__main__":
    main()
