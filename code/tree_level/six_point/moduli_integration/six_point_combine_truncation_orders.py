"""Two-level spectral sums with explicit missing-control and method records.

The inputs are FULL nonchiral momentum integrals, including original primary
powers, all four DOZZ factors, the dP^3/pi^3 measure, and any source-moduli
weight. A lower-order chiral block must never be subtracted in their place.
Observed mixed defects compare finite rules; they do not bound either tail.
"""
from __future__ import annotations

import hashlib
import importlib
import operator
from pathlib import Path

import numpy as np

VERSION = 'sphere-six-point-two-level-v1'


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f'{name} must be an integer')
    try:
        return operator.index(value)
    except TypeError as exc:
        raise ValueError(f'{name} must be an integer') from exc


def _mask(value, shape, name):
    result = np.asarray(value)
    if result.dtype != np.dtype(bool) or result.shape != shape:
        raise ValueError(f'{name} must be a boolean array of shape {shape}')
    return result.copy()


def _method(value, shape):
    result = np.asarray(value)
    if result.dtype.kind not in 'US':
        raise ValueError('explicit string representation identities required')
    if result.ndim == 0:
        result = np.full(shape, str(result.item()), dtype=object)
    elif result.shape != shape:
        raise ValueError('representation identities must match the sample shape')
    if any(not str(v).strip() for v in result.flat):
        raise ValueError('empty representation identity')
    return np.asarray(result, dtype=str)


def combine_two_level(baseline_low, baseline_high, control_low, *,
                      control_high=None, low_order=10, high_order=12,
                      baseline_method, control_method,
                      baseline_high_method=None, control_high_method=None,
                      eligible=None, high_available=None,
                      absolute_tolerance=None, provenance=None):
    """Return Q_control f_L + Q_baseline(f_H-f_L), for scalars or arrays.

    All three required arrays have EXACTLY the same shape; values are not
    broadcast across samples. Method strings identify the common physical
    representation (including any re-expansion axes/face/elliptic settings),
    separately from its L/H truncation. Omitting a high-method argument is an
    explicit assertion that it equals that rule's low method. This algebra
    function cannot independently verify caller-supplied physical provenance.

    ``eligible`` labels a separately treated domain; excluded entries remain
    NaN, never zero. A partially available control H needs an explicit boolean
    mask. Missing-H entries MUST be NaN. Such entries never pass a comparison.
    Supplied tolerances are ABSOLUTE physical units and assess only an observed
    finite-rule defect. No default accuracy tolerance or tail claim is made.
    """
    low_order, high_order = _integer(low_order, 'low order'), _integer(high_order, 'high order')
    if not 0 <= low_order < high_order:
        raise ValueError('require 0 <= low order < high order')
    low = np.asarray(baseline_low, dtype=complex)
    high = np.asarray(baseline_high, dtype=complex)
    other_low = np.asarray(control_low, dtype=complex)
    shape = low.shape
    if not low.size or high.shape != shape or other_low.shape != shape:
        raise ValueError('nonempty, exactly matching full-integrand shapes required')
    use = np.ones(shape, bool) if eligible is None else _mask(eligible, shape, 'eligible')
    for name, array in (('baseline low', low), ('baseline high', high), ('control low', other_low)):
        if not np.isfinite(array[use]).all():
            raise FloatingPointError(f'nonfinite {name} on requested samples')
    bm, cm = _method(baseline_method, shape), _method(control_method, shape)
    bh = bm if baseline_high_method is None else _method(baseline_high_method, shape)
    ch = cm if control_high_method is None else _method(control_high_method, shape)
    if np.any(use & ((bm != cm) | (bm != bh))):
        raise ValueError('two-level sums require a common representation on every rule and level')
    if control_high is None:
        if high_available is not None and _mask(high_available, shape, 'high available').any():
            raise ValueError('high availability cannot be asserted without high values')
        available = np.zeros(shape, bool)
        other_high = np.full(shape, complex(np.nan, np.nan))
    else:
        other_high = np.asarray(control_high, dtype=complex)
        if other_high.shape != shape:
            raise ValueError('control high shape differs from original samples')
        available = np.ones(shape, bool) if high_available is None else _mask(high_available, shape, 'high available')
        if not np.isfinite(other_high[available & use]).all():
            raise FloatingPointError('nonfinite available high control')
        if not np.isnan(other_high[~available]).all():
            raise ValueError('unavailable high controls must be explicit NaNs, never zero proxies')
    assessed = use & available
    if np.any(assessed & (bm != ch)):
        raise ValueError('direct high comparison changes physical representation')
    nan = complex(np.nan, np.nan)
    correction, estimate, direct, defect = (np.full(shape, nan) for _ in range(4))
    with np.errstate(over='raise', invalid='raise'):
        correction[use] = high[use] - low[use]
        estimate[use] = other_low[use] + correction[use]
        direct[assessed] = other_high[assessed]
        defect[assessed] = estimate[assessed] - direct[assessed]
    if not np.isfinite(estimate[use]).all() or not np.isfinite(defect[assessed]).all():
        raise FloatingPointError('two-level subtraction exceeded floating-point range')
    absolute = np.abs(defect)
    relative = np.full(shape, np.nan)
    defined = assessed & (abs(direct) > 0)
    relative[defined] = absolute[defined] / abs(direct[defined])
    passed = None
    if absolute_tolerance is not None:
        tolerance = np.asarray(absolute_tolerance, dtype=float)
        try:
            tolerance = np.broadcast_to(tolerance, shape)
        except ValueError as exc:
            raise ValueError('absolute tolerance shape mismatch') from exc
        if not np.isfinite(tolerance).all() or np.any(tolerance < 0):
            raise ValueError('finite nonnegative absolute tolerance required')
        passed = assessed & (absolute <= tolerance)
    complete = bool(assessed.all())
    if not complete:
        status = 'unassessed-points-remain'
    elif passed is None:
        status = 'observed-not-budgeted'
    else:
        status = 'observed-budget-pass' if passed.all() else 'observed-budget-fail'
    return dict(version=VERSION, low_order=low_order, high_order=high_order,
        estimate=estimate, baseline_correction=correction, direct_high=direct,
        signed_mixed_defect=defect, absolute_mixed_defect=absolute,
        relative_mixed_defect=relative, normalization_defined=defined,
        eligible=use, high_available=available, comparison_assessed=assessed,
        empirical_defect_pass=passed, global_mixed_defect_status=status,
        all_direct_high_comparisons_available=complete, common_method=bm,
        excluded_point_count=int((~use).sum()), missing_high_control_count=int((use & ~available).sum()),
        input_sample_count=low.size, all_sample_slots_retained=True,
        provenance=dict(provenance or {}),
        formula='Q_control f_L + Q_baseline(f_H-f_L)',
        defect_sign='two-level estimate minus direct control H',
        relative_denominator='absolute direct control H, undefined at zero',
        physical_context_verified_by_algebra=False,
        block_tail_certified=False, momentum_tail_certified=False,
        total_accuracy_certified=False)


def _statistics(values, mask, ids):
    """Regional contributions use full-source denominators, even if empty."""
    values, mask, ids = values.ravel(), mask.ravel(), ids.ravel()
    labels = np.unique(ids)
    means, counts = [], []
    for label in labels:
        rows = ids == label
        counts.append(int(rows.sum()))
        means.append(complex(values[rows & mask].sum() / counts[-1]))
    means = np.asarray(means)
    equal = len(set(counts)) == 1
    mean = complex(values[mask].sum() / len(values))
    se = dict(real=float(means.real.std(ddof=1) / np.sqrt(len(means))),
               imag=float(means.imag.std(ddof=1) / np.sqrt(len(means)))) if len(means)>1 and equal else None
    return dict(mean=dict(real=mean.real, imag=mean.imag),
        paired_scramble_se=se,
        replicate_means=[[float(x.real),float(x.imag)] for x in means],
        replicate_full_counts=counts, contributing_count=int(mask.sum()),
        full_sample_denominator=len(values),
        denominator_policy='sum over specified observed region divided by every original source sample',
        standard_error_requires_independent_scrambles=True)


def summarize_two_level(result, *, replicate_ids=None, regions=None):
    """Summarize observed paired defects without certifying missing controls.

    Regional statistics are contributions to the ORIGINAL full-sample mean,
    never conditional means. Unknown entries are labelled and cannot establish
    a global bound. For scalar inputs and one scramble the SE is unavailable.
    """
    use, assessed = result['eligible'], result['comparison_assessed']
    shape = use.shape
    ids = np.zeros(shape, dtype=int) if replicate_ids is None else np.asarray(replicate_ids)
    if ids.shape != shape or ids.dtype.kind not in 'iu':
        raise ValueError('integer replicate IDs must match original samples')
    all_regions = {'all':np.ones(shape,bool)}
    for name, mask in (regions or {}).items():
        if not isinstance(name,str) or not name or name == 'all':
            raise ValueError('distinct named regions required; all is reserved')
        all_regions[name] = _mask(mask,shape,'region')
    records = {}
    for name, region in all_regions.items():
        observed, requested = region & assessed, region & use
        records[name] = dict(source_region_count=int(region.sum()),
            evaluated_count=int(requested.sum()), assessed_count=int(observed.sum()),
            missing_high_count=int((requested & ~assessed).sum()),
            excluded_count=int((region & ~use).sum()),
            complete_for_source_region=bool(np.all(~region | assessed)),
            estimate_contribution=_statistics(result['estimate'], requested, ids),
            observed_direct_high_contribution=_statistics(result['direct_high'], observed, ids) if observed.any() or not region.any() else None,
            observed_signed_defect_contribution=_statistics(result['signed_mixed_defect'], observed, ids) if observed.any() or not region.any() else None,
            observed_absolute_defect_contribution=_statistics(result['absolute_mixed_defect'], observed, ids) if observed.any() or not region.any() else None)
    return dict(version=VERSION,low_order=result['low_order'],high_order=result['high_order'],
        global_mixed_defect_status=result['global_mixed_defect_status'],
        regions=records,full_sample_count=use.size,
        unavailable_comparisons_are_not_zero_errors=True,
        finite_rule_comparison_only=True,total_accuracy_certified=False)


def _hash_array(array):
    a=np.ascontiguousarray(array)
    return hashlib.sha256(str((a.shape,a.dtype.str)).encode()+a.tobytes()).hexdigest()


def source_hashes():
    paths=[Path(__file__).resolve()]
    paths += [Path(importlib.import_module(n).__file__).resolve() for n in
        ('six_point_momentum_banks_and_reweighting','six_point_batch_block_contraction','six_point_measure_block_convergence')]
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def evaluate_native_banks(baseline_bank, control_banks, q_values, *,
                          base_logarithms=0j, low_order=10, high_order=12,
                          batch_size=16):
    """Actually integrate heterogeneous L/H banks in a common NATIVE basis.

    This does not dispatch boundary repairs. All geometry/sample weights must
    already be in ``base_logarithms``. An L-only control has no invented H data.
    Full DOZZ, original proposal weights and every row come from each bank.
    Prefix diagnostics are reported separately; two orders are not a tail test.
    """
    from six_point_momentum_banks_and_reweighting import SphereSixPointMomentumBank
    from six_point_batch_block_contraction import SphereBatchEvaluator
    from six_point_measure_block_convergence import evaluate_native_prefixes
    low_order,high_order=_integer(low_order,'low order'),_integer(high_order,'high order')
    if not 0<=low_order<high_order:
        raise ValueError('require 0 <= low order < high order')
    if not isinstance(baseline_bank,SphereSixPointMomentumBank) or baseline_bank.order<high_order:
        raise ValueError('baseline requires an actual complete high-order bank')
    if not isinstance(control_banks,dict) or not control_banks:
        raise ValueError('at least one explicitly named control bank required')
    q=np.asarray(q_values,complex)
    if q.ndim==1:q=q[None,:]
    base=np.broadcast_to(np.asarray(base_logarithms,complex),(len(q),))
    identity=(baseline_bank.t,baseline_bank.topology,baseline_bank.incoming_slot)
    before=source_hashes()
    banks={'baseline':baseline_bank}
    if 'baseline' in control_banks:
        raise ValueError('baseline is a reserved control name')
    banks.update(control_banks)
    evaluated,receipts={},{}
    for name,bank in banks.items():
        if not isinstance(name,str) or not name or not isinstance(bank,SphereSixPointMomentumBank):
            raise ValueError('named complete momentum banks required')
        if (bank.t,bank.topology,bank.incoming_slot)!=identity or bank.provider_metadata!=baseline_bank.provider_metadata:
            raise ValueError('control mixes physical external data or coefficient provider')
        if bank.order<low_order:
            raise ValueError('control lacks the requested complete low prefix')
        orders=(low_order,high_order) if bank.order>=high_order else (low_order,)
        evaluated[name]=evaluate_native_prefixes(SphereBatchEvaluator(bank.as_momentum_arrays()),q,
            base_logarithms=base,orders=orders,batch_size=batch_size)
        receipts[name]=dict(bank_fingerprint=bank.fingerprint,proposal_fingerprint=bank.proposal.fingerprint,
            available_bank_order=bank.order,evaluated_orders=list(orders),momentum_node_count=len(bank.p_values),
            provider=bank.provider_metadata,bank_provenance=bank.provenance)
    combinations={}
    for name in control_banks:
        value=evaluated[name]['values'];b=evaluated['baseline']['values']
        combinations[name]=combine_two_level(b[low_order],b[high_order],value[low_order],
            control_high=value.get(high_order),low_order=low_order,high_order=high_order,
            baseline_method='native-total-degree',control_method='native-total-degree',
            provenance=dict(baseline=receipts['baseline'],control=receipts[name]))
    if source_hashes()!=before:
        raise RuntimeError('multilevel evaluation source changed during contraction')
    return dict(version=VERSION,physical_identity=dict(t=identity[0],topology=identity[1],incoming_slot=identity[2]),
        point_context=dict(q_sha256=_hash_array(q),base_logarithms_sha256=_hash_array(base)),
        evaluations=evaluated,combinations=combinations,bank_receipts=receipts,source_hashes=before,
        boundary_method='native-only; conditional boundary treatment not performed',
        boundary_qualification_performed=False,all_momentum_rows_retained=True,
        full_mixed_production_ready=False,total_accuracy_certified=False)


__all__=['VERSION','combine_two_level','summarize_two_level','evaluate_native_banks','source_hashes']
