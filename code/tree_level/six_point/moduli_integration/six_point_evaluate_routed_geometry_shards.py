"""Indexed geometry work with common heterogeneous block representations.

This module does not launch processes or choose resources. One Slurm worker
owns each geometry shard, with every momentum variant of a point kept together.
Unavailable high orders remain NaN. A short control ladder can screen a common
representation but cannot acquire a three-prefix or high-order certificate.
The labelled collar addback is computed and stored separately, then shared.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import math
from pathlib import Path

import numpy as np

import six_point_estimate_collar_corrected_integrand as routed
from six_point_route_collar_contributions import CollarCFTEvaluator, NORMALIZATION, point_from_plan
from six_point_sample_integration_geometry import SphereGeometryPlan, point_digest
from six_point_momentum_banks_and_reweighting import SphereSixPointMomentumBank, external_data

VERSION = 'sphere-six-point-indexed-direct-routing-v2'
BOUNDARY_POLICY = 'conditional-boundary-blocks-v1'


def settings(config):
    if not isinstance(config,dict) or config.get('policy') != BOUNDARY_POLICY:
        raise ValueError('explicit conditional-boundary-blocks-v1 policy required')
    defaults=dict(policy=BOUNDARY_POLICY,relative_tolerance=.0025,momentum_tolerance=.0025,
                  minimum_native_order=12,elliptic_orders=[8,12],dps=60,
                  maximum_nome_radius=.1,momentum_controls=None)
    if set(config)-set(defaults):
        raise ValueError(f'unknown boundary treatment settings: {sorted(set(config)-set(defaults))}')
    defaults.update(config)
    for key in ('relative_tolerance','momentum_tolerance','maximum_nome_radius'):
        value=defaults[key]
        if isinstance(value,bool) or not math.isfinite(value) or not 0<value<1:
            raise ValueError(f'{key} must lie strictly between zero and one')
    for key,minimum in (('minimum_native_order',3),('dps',30)):
        value=defaults[key]
        if isinstance(value,bool) or int(value)!=value or value<minimum:
            raise ValueError(f'{key} must be an integer >= {minimum}')
        defaults[key]=int(value)
    orders=defaults['elliptic_orders']
    if (not isinstance(orders,(list,tuple)) or len(orders)<2
            or any(isinstance(n,bool) or int(n)!=n or not 0<=n<=24 for n in orders)
            or list(orders)!=sorted(set(orders))):
        raise ValueError('at least two increasing elliptic orders through 24 required')
    defaults['elliptic_orders']=[int(n) for n in orders]
    controls=defaults['momentum_controls']
    if controls is not None and (not isinstance(controls,list) or not controls
            or len(set(controls))!=len(controls)
            or any(not isinstance(v,str) or not v for v in controls)):
        raise ValueError('momentum_controls must be a distinct nonempty list of variant names')
    return defaults


def _json_safe(value):
    if isinstance(value,dict):return {str(k):_json_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [_json_safe(v) for v in value]
    if isinstance(value,(complex,np.complexfloating)):return [float(value.real),float(value.imag)]
    if isinstance(value,np.generic):return _json_safe(value.item())
    if isinstance(value,float) and not math.isfinite(value):return None
    return value


def source_hashes():
    result = routed.source_hashes()
    for name in ('six_point_sample_integration_geometry', 'six_point_momentum_banks_and_reweighting'):
        path = Path(importlib.import_module(name).__file__).resolve()
        result[str(path)] = _digest(path)
    result[str(Path(__file__).resolve())] = _digest(__file__)
    return result


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def slice_plan(plan, indices):
    """Copy exact source records, preserving their full-plan indices and hash."""
    indices = np.asarray(indices)
    if (indices.ndim != 1 or not len(indices) or indices.dtype.kind not in 'iu'
            or len(set(indices.tolist())) != len(indices)
            or np.any(indices < 0) or np.any(indices >= len(plan.arrays['points']))):
        raise ValueError('distinct in-range original geometry indices required')
    arrays = {key: value[indices].copy() for key, value in plan.arrays.items()}
    metadata = dict(plan.metadata)
    prior = metadata.get('source_indices', list(range(len(plan.arrays['points']))))
    metadata.update(sample_count=len(indices), point_sha256=point_digest(arrays['points']),
        source_indices=[int(prior[i]) for i in indices],
        source_point_sha256=metadata.get('source_point_sha256', plan.metadata['point_sha256']),
        source_sample_count=metadata.get('source_sample_count', len(plan.arrays['points'])),
        sample_design='indexed-shard', amplitude_sample=False)
    return SphereGeometryPlan(metadata, arrays)


def slice_marks(marks, replicate, indices):
    """Slice marking records only; no coordinate selection or density changes."""
    indices = np.asarray(indices)
    if (indices.ndim != 1 or indices.dtype.kind not in 'iu' or not len(indices)
            or len(set(indices.tolist())) != len(indices)
            or isinstance(replicate, bool) or int(replicate) != replicate
            or not 0 <= replicate < len(marks['records'])
            or np.any(indices < 0) or np.any(indices >= len(marks['records'][replicate]))):
        raise ValueError('valid replicate and distinct source indices required')
    r = int(replicate)
    return dict(marks, inside=marks['inside'][r:r+1, indices].copy(),
                unhandled=marks['unhandled'][r:r+1, indices].copy(),
                records=[[marks['records'][r][int(i)] for i in indices]])


def apply_mixed_treatment(plans, energies, families, selected, qualified, raw,
                          native_increments, marks, config, *, orders, variants,
                          baseline, available_orders):
    """Common alternatives evaluated at each bank's ACTUAL retained orders.

    Adoption requires the baseline's two final gaps and every active control's
    available gaps to pass, plus all supplied variant/order values finite.
    ``block_prefix_pass`` is stricter: every active variant needs three actual
    prefixes through the requested minimum. ``available_prefix_pass`` is only
    an observed finite-ladder screen. Missing entries are never manufactured.
    Momentum changes here compare the highest order available on every active
    rule; the two-level estimator and direct-H availability are separate.
    """
    cfg = settings(config)
    orders, variants, energies = tuple(orders), list(variants), tuple(energies)
    if (len(orders) < 2 or list(orders) != sorted(set(orders))
            or any(isinstance(n, bool) or int(n) != n or n < 0 for n in orders)
            or len(set(variants)) != len(variants) or baseline not in variants):
        raise ValueError('ordered native prefixes and distinct named variants required')
    raw = np.asarray(raw, complex)
    selected, qualified = np.asarray(selected), np.asarray(qualified)
    available = np.asarray(available_orders)
    expected = (len(energies), len(plans), len(plans[0].arrays['points']))
    if (raw.shape != (*expected, len(variants), len(orders))
            or selected.shape != expected or selected.dtype.kind not in 'iu'
            or qualified.shape != expected or qualified.dtype != np.dtype(bool)
            or available.shape != (len(families), len(variants), len(orders))
            or available.dtype != np.dtype(bool)
            or np.shape(native_increments) != (*raw.shape[:-1], len(orders)-1)
            or np.shape(marks['inside']) != expected[1:]):
        raise ValueError('mixed routing array axes or availability types disagree')
    # Complete prefixes only: a missing intermediate order cannot be silently
    # bridged by an increment whose two endpoints were never evaluated.
    for row in available.reshape(-1, len(orders)):
        count = int(row.sum())
        if count < 2 or not row[:count].all() or row[count:].any():
            raise ValueError('every bank needs a contiguous available ladder of at least two prefixes')
    controls = cfg['momentum_controls']
    if controls is None:
        controls = [v for v in variants if v != baseline]
    if any(v not in variants or v == baseline for v in controls):
        raise ValueError('invalid active momentum control names')
    b = variants.index(baseline)
    active = [variants.index(v) for v in (baseline, *controls)]
    if any(available[f, b].sum() < 3 or orders[int(available[f, b].sum())-1] < cfg['minimum_native_order']
           for f in range(len(families))):
        raise ValueError('baseline requires three actual prefixes through minimum_native_order')
    counts = available.sum(axis=-1)
    values = raw.copy()
    shape = raw.shape[:3]
    variant_pass = np.zeros((*shape, len(variants)), bool)
    supplied_finite = np.zeros(shape, bool)
    has_three = np.zeros(shape, bool)
    high_available = np.zeros((*shape, len(variants)), bool)
    for key in np.ndindex(shape):
        f = int(selected[key])
        if not 0 <= f < len(families):
            if qualified[key]:
                raise ValueError('qualified point has no selected physical family')
            continue
        mask = available[f]
        if not np.isnan(raw[key][~mask]).all():
            raise ValueError('unavailable native orders must remain explicit NaNs')
        supplied_finite[key] = np.isfinite(raw[key][mask]).all()
        high_available[key] = mask[:, -1]
        has_three[key] = all(counts[f, v] >= 3 and orders[counts[f, v]-1] >= cfg['minimum_native_order'] for v in active)
        for v in range(len(variants)):
            count = int(counts[f, v])
            gaps = np.asarray(native_increments[key][v, max(0, count-3):count-1])
            variant_pass[key][v] = bool(np.isfinite(gaps).all() and len(gaps)
                and np.max(gaps) < cfg['relative_tolerance']
                and np.isfinite(raw[key][v, :count]).all())
    available_pass = qualified & supplied_finite & variant_pass[..., active].all(axis=-1)
    native_pass = available_pass.copy()
    methods = np.full(shape, 'native-unresolved', dtype='<U40')
    methods[available_pass] = 'native'
    methods[:, marks['inside']] = 'analytic-tested-collar'
    labels = [-1, *cfg['elliptic_orders']]
    completed = np.full((*raw.shape, len(labels)), complex(np.nan, np.nan))
    records = [dict(t=energies[e], replicate=r, point=i,
                    status='higher-native-order-required', retained_method=str(methods[e, r, i]))
               for e, r, i in np.ndindex(shape)
               if not available_pass[e, r, i] and not marks['inside'][r, i]]
    baseline_pass = qualified & supplied_finite & variant_pass[..., b]
    block_pass = available_pass & has_three
    momentum_pass = np.zeros(shape, bool)
    common_order = np.full(shape, -1, int)
    differences = {name: np.full(shape, np.nan) for name in controls}
    for key in np.ndindex(shape):
        if marks['inside'][key[1:]] or not 0 <= selected[key] < len(families):
            continue
        f = int(selected[key])
        last = min(int(counts[f, v]) for v in active)-1
        common_order[key] = orders[last]
        reference = values[key][b, last]
        denominator = abs(reference)
        passed = bool(controls) and np.isfinite(reference) and denominator > 0
        for name in controls:
            delta = abs(values[key][variants.index(name), last]-reference)/denominator if denominator else np.nan
            differences[name][key] = delta
            passed = passed and np.isfinite(delta) and delta < cfg['momentum_tolerance']
        momentum_pass[key] = passed
    for array in (baseline_pass, available_pass, native_pass, block_pass, momentum_pass):
        array[:, marks['inside']] = False
    marked = np.asarray([[row['requires_factorization'] for row in records_] for records_ in marks['records']], bool)
    return dict(version=VERSION, values=values, completed_values=completed,
        completed_elliptic_orders=labels, methods=methods, records=_json_safe(records), settings=cfg,
        baseline_prefix_pass=baseline_pass, available_prefix_pass=available_pass,
        native_prefix_pass=native_pass, block_prefix_pass=block_pass, three_prefixes_all_variants=has_three,
        full_high_available=high_available, momentum_pass=momentum_pass,
        momentum_comparison_order=common_order, momentum_differences=differences, marked=marked,
        unresolved_marked_blocks=marked[None] & ~marks['inside'][None] & ~block_pass,
        all_samples_retained=True, missing_high_orders_preserved=True,
        qualification_is_tail_bound=False, total_accuracy_certified=False)


def _context(energies, config):
    cfg = routed.settings(config)
    energies = [float(t) for t in energies]
    if not energies or len(set(energies)) != len(energies) or any(not 0 < t < 1/3 for t in energies):
        raise ValueError('distinct energies in the residue-free imaginary strip required')
    root = Path(__file__).resolve().parents[4]
    # Relocation of an identical frozen source tree is not a physics change.
    # Relative keys retain every dependency and avoid host-specific prefixes.
    sources = {str(Path(path).relative_to(root)): digest for path, digest in source_hashes().items()}
    return dict(version=VERSION, energies=energies, settings=cfg,
                normalization_id=NORMALIZATION, sources=sources)


def evaluate_marked_addback(energies, config):
    """Compute each energy/profile addback once, independently of any shard."""
    context = _context(energies, config)
    cfg = context['settings']
    shape = (len(energies), len(cfg['momentum_profiles']), len(cfg['elliptic_orders']))
    addback, lower = np.empty(shape, complex), np.empty(shape, complex)
    records = []
    for e, t in enumerate(energies):
        for p, profile in enumerate(cfg['momentum_profiles']):
            evaluator = CollarCFTEvaluator(t, profile=profile, block_orders=cfg['elliptic_orders'], dps=cfg['dps'])
            high = evaluator.prepare_addback(cfg['x_order'])
            low = evaluator.prepare_addback(cfg['x_order']-1)
            for o, n in enumerate(cfg['elliptic_orders']):
                addback[e, p, o] = high['by_order'][str(n)]
                lower[e, p, o] = low['by_order'][str(n)]
            records.append(dict(t=float(t), profile=profile, momentum_nodes=len(evaluator.ps),
                spectral_metadata=evaluator.grid['metadata'], normalization_id=NORMALIZATION))
            evaluator._slices.cache_clear()
            del evaluator
    if not np.isfinite(addback).all() or not np.isfinite(lower).all():
        raise FloatingPointError('nonfinite shared analytic addback')
    if context != _context(energies, config):
        raise RuntimeError('collar source changed during shared addback evaluation')
    return dict(context=context, context_identity=_identity(context), addback=addback,
                addback_lower_x=lower, records=records)


def save_marked_addback(result, path):
    """Write a portable NPZ plus bound JSON, refusing an existing destination."""
    path = Path(path)
    sidecar = path.with_suffix('.json')
    if path.suffix != '.npz' or path.exists() or sidecar.exists():
        raise ValueError('a new .npz addback destination is required')
    if result['context_identity'] != _identity(result['context']):
        raise ValueError('addback context identity mismatch')
    _validate_addback(result, result['context'])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as handle:
        np.savez_compressed(handle, addback=result['addback'], addback_lower_x=result['addback_lower_x'])
    metadata = {k: v for k, v in result.items() if k not in ('addback', 'addback_lower_x')}
    metadata['npz_sha256'] = _digest(path)
    with sidecar.open('x') as handle:
        json.dump(_json_safe(metadata), handle, indent=2, allow_nan=False)
    return dict(path=str(path.resolve()), sha256=metadata['npz_sha256'],
                metadata_sha256=_digest(sidecar), context_identity=result['context_identity'])


def load_marked_addback(path, energies, config):
    path = Path(path)
    metadata = json.loads(path.with_suffix('.json').read_text())
    context = _context(energies, config)
    if (metadata.get('context') != context or metadata.get('context_identity') != _identity(context)
            or metadata.get('npz_sha256') != _digest(path)):
        raise ValueError('shared addback source/config/energy/checksum mismatch')
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != {'addback', 'addback_lower_x'}:
            raise ValueError('shared addback array schema mismatch')
        arrays = {k: data[k].copy() for k in data.files}
    shape = (len(energies), len(context['settings']['momentum_profiles']), len(context['settings']['elliptic_orders']))
    if any(a.shape != shape or not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError('shared addback shape or finite coverage mismatch')
    return dict(metadata, **arrays)


def _validate_addback(result, context):
    cfg = context['settings']
    shape = (len(context['energies']), len(cfg['momentum_profiles']), len(cfg['elliptic_orders']))
    for name in ('addback', 'addback_lower_x'):
        value = np.asarray(result[name])
        if value.shape != shape or value.dtype.kind not in 'fc' or not np.isfinite(value).all():
            raise ValueError('shared addback shape/type/finite coverage mismatch')


def evaluate_marked_points(plans, energies, marks, config):
    """Evaluate only this shard's physical collar points, never an addback."""
    context = _context(energies, config)
    cfg = context['settings']
    shape = (len(energies), *marks['inside'].shape, len(cfg['momentum_profiles']), len(cfg['elliptic_orders']))
    arrays = {key: np.zeros(shape, complex) for key in ('full', 'leading', 'remainder', 'outer1')}
    records = []
    indices = np.argwhere(marks['inside'])
    if len(indices):
        for e, t in enumerate(energies):
            for p, profile in enumerate(cfg['momentum_profiles']):
                evaluator = CollarCFTEvaluator(t, profile=profile, block_orders=cfg['elliptic_orders'], dps=cfg['dps'])
                rows = []
                for r, i in indices:
                    try:
                        result = evaluator.evaluate(point_from_plan(plans[r], i))
                        for o, n in enumerate(cfg['elliptic_orders']):
                            for key in arrays:
                                arrays[key][e, r, i, p, o] = result['by_order'][str(n)][key]
                        rows.append(dict(replicate=int(r), point=int(i), diagnostics=result['diagnostics']))
                    except (ArithmeticError, ValueError, OverflowError) as error:
                        for array in arrays.values():
                            array[e, r, i, p] = complex(np.nan, np.nan)
                        rows.append(dict(replicate=int(r), point=int(i), failure=str(error)))
                records.append(dict(t=float(t), profile=profile, points=rows,
                    momentum_nodes=len(evaluator.ps), normalization_id=NORMALIZATION,
                    spectral_metadata=evaluator.grid['metadata']))
                evaluator._slices.cache_clear()
                del evaluator
    if context != _context(energies, config):
        raise RuntimeError('collar source changed during indexed point evaluation')
    return dict(context=context, context_identity=_identity(context), arrays=arrays,
                records=records, inside=marks['inside'].copy(), addback_computations=0)


def combine_marked(points_result, addback_result, energies, config):
    """Restore the existing collar schema; the caller adds the constant once."""
    context = _context(energies, config)
    for result in (points_result, addback_result):
        if result['context'] != context or result['context_identity'] != _identity(context):
            raise ValueError('point and analytic addback settings/source identities differ')
    cfg = context['settings']
    arrays = points_result['arrays']
    inside = np.asarray(points_result['inside'])
    if inside.ndim != 2 or not inside.size or inside.dtype != np.dtype(bool):
        raise ValueError('indexed point mask must be a nonempty replicate-by-point boolean array')
    shape = (len(energies), *inside.shape, len(cfg['momentum_profiles']), len(cfg['elliptic_orders']))
    if set(arrays) != {'full', 'leading', 'remainder', 'outer1'}:
        raise ValueError('indexed collar array schema mismatch')
    for value in arrays.values():
        value = np.asarray(value)
        if value.shape != shape or value.dtype.kind not in 'fc':
            raise ValueError('indexed collar arrays differ from physical sample inventory')
    _validate_addback(addback_result, context)
    addback, lower = addback_result['addback'], addback_result['addback_lower_x']
    full = arrays['full']
    denom = np.maximum(abs(full[:, :, :, -1, -1]), np.finfo(float).tiny)
    momentum = abs(full[:, :, :, -2, -1]-full[:, :, :, -1, -1])/denom
    block = abs(full[:, :, :, -1, -2]-full[:, :, :, -1, -1])/denom
    outer = abs(arrays['outer1'][:, :, :, -1, -1]-full[:, :, :, -1, -1])/denom
    passed = (momentum < cfg['relative_tolerance']) & (block < cfg['relative_tolerance']) & (outer < cfg['relative_tolerance'])
    for value in arrays.values():
        passed &= np.isfinite(value).all(axis=(-1, -2))
    for row in points_result['records']:
        if row['profile'] != cfg['momentum_profiles'][-1]:
            continue
        e = list(energies).index(row['t'])
        for point in row['points']:
            r, i = point['replicate'], point['point']
            change = point.get('diagnostics', {}).get('absolute_weighted_order_change', {}).get(str(cfg['elliptic_orders'][-2]))
            passed[e, r, i] &= change is not None and np.isfinite(change) and change < cfg['relative_tolerance']
    add_denom = np.maximum(abs(addback[:, -1, -1]), np.finfo(float).tiny)
    add_momentum = abs(addback[:, -2, -1]-addback[:, -1, -1])/add_denom
    add_middle = abs(lower[:, -1, -1]-addback[:, -1, -1])/add_denom
    add_block = abs(addback[:, -1, -2]-addback[:, -1, -1])/add_denom
    add_pass = ((add_momentum < cfg['relative_tolerance']) & (add_middle < cfg['relative_tolerance'])
                & (add_block < cfg['relative_tolerance']) & np.isfinite(addback).all(axis=(-1, -2))
                & np.isfinite(lower).all(axis=(-1, -2)))
    return dict(arrays=arrays, addback=addback, addback_lower_x=lower,
        momentum_change=momentum, block_change=block, outer_change=outer,
        prefix_pass=passed, addback_prefix_pass=add_pass,
        addback_momentum_change=add_momentum, addback_middle_change=add_middle,
        addback_block_change=add_block, records=points_result['records'], settings=cfg,
        shared_addback_identity=addback_result['context_identity'],
        addback_records=addback_result['records'], analytic_addback_once_per_integral=True)


def apply_treatment(plans, energies, families, selected, qualified, raw,
                    native_increments, marks, config, *, orders, variants, baseline):
    """Direct-series diagnostics when all momentum rules have the same prefix ladder."""
    available = np.ones((len(families), len(variants), len(orders)), dtype=bool)
    return apply_mixed_treatment(plans, energies, families, selected, qualified, raw,
        native_increments, marks, config, orders=orders, variants=variants,
        baseline=baseline, available_orders=available)
