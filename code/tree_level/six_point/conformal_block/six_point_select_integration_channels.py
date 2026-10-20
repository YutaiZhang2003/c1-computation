"""Explicit six-sphere channel preferences with unchanged source measure.

The diagonal star gauge is the nearest collision under simultaneous complex
rescaling of all three native variables. It screens a total-degree expansion;
it is neither absolute multivariate convergence nor a finite-order/tail bound.
Routing never changes the original all1440 proposal density or matter factor.
"""
from __future__ import annotations

from collections.abc import Mapping
import math

import numpy as np

ROUTING_VERSION = 'sphere-six-point-channel-routing-v1'
COORDINATE_LOG_TOLERANCE = 1e-9
POLICY_NAMES = ('absolute-minimum-v1', 'diagonal-minimum-v1', 'diagonal-star-priority-v1')
PAIR_LABELS = tuple((i,j) for i in range(6) for j in range(i+1,6))


def normalize_routing(policy=None):
    """Return a canonical policy dictionary, rejecting unused parameters.

    None preserves the existing absolute-gauge selection. Thresholds only
    affect the explicit star-priority policy; they are geometric preferences.
    No policy chooses the value closest to a target or labels a CFT sum passed.
    """
    if policy is None:
        record = {'name': 'absolute-minimum-v1'}
    elif isinstance(policy, str):
        record = {'name': policy}
    elif isinstance(policy, Mapping):
        record = dict(policy)
    else:
        raise TypeError('routing must be a named policy or mapping')
    name = record.get('name')
    if name not in POLICY_NAMES:
        raise ValueError('unknown sphere channel routing policy')
    allowed = {'name', 'comb_min', 'star_max'} if name == 'diagonal-star-priority-v1' else {'name'}
    if set(record) - allowed:
        raise ValueError('unused or unknown routing parameters')
    result = {'name': name}
    if name == 'diagonal-star-priority-v1':
        for key, default in (('comb_min', .6), ('star_max', .7)):
            value = record.get(key, default)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int,float,np.integer,np.floating)):
                raise ValueError('routing thresholds must be finite real numbers')
            value = float(value)
            if not math.isfinite(value) or not 0 <= value < 1 or (key == 'star_max' and value == 0):
                raise ValueError('comb_min must lie in [0,1), star_max in (0,1)')
            result[key] = value
    return result


def star_gauge(q, domain='absolute'):
    """Vectorized star collision gauge, with the last dimension equal to 3.

    In the absolute domain, the relevant phase-torus boundaries are
    r1+r2=1 and r3*(1+r2)=1. In the diagonal domain, set q -> lambda*q
    in (0,q1,1,1+q2,1/q3,infinity) and find every nonzero pair collision.
    The quadratic collision yields inverse roots of
    s²-q3*s-q2*q3=0; both complex roots must be included.
    """
    if domain not in ('absolute', 'diagonal'):
        raise ValueError('star gauge domain must be absolute or diagonal')
    values = np.asarray(q, dtype=complex)
    if values.ndim == 0 or values.shape[-1] != 3 or not np.isfinite(values).all():
        raise ValueError('finite native coordinates with last dimension 3 required')
    q1,q2,q3 = np.moveaxis(values,-1,0)
    r1,r2,r3 = abs(q1),abs(q2),abs(q3)
    if domain == 'absolute':
        return np.maximum(r1+r2, .5*(r3+np.sqrt(r3*r3+4*r2*r3)))
    discriminant = np.sqrt(q3*q3+4*q2*q3)
    return np.maximum.reduce((r1,r2,r3,abs(q1-q2),np.sqrt(r1*r3),
                             abs(.5*(q3+discriminant)),abs(.5*(q3-discriminant))))


def coordinate_fidelity_error(differences, topology, row):
    """Maximum error in the 15 labelled pair-log covariance identities.

    A representable q near a boundary can still lose a much smaller puncture
    separation when converted into a different chart. Reconstruct native
    target differences and compare log|D'_ij| with log|D_ij|+(s_i+s_j)/2,
    where s_i are the candidate's local scale logs in label order. Return
    infinity for malformed, collapsed or unrepresentable geometry. This is
    a coordinate identity check, not a CFT or spectral error estimate.
    """
    # Imported lazily: native geometry uses this module to route its candidates.
    from six_point_stable_puncture_coordinates import native_differences

    try:
        if topology not in ('comb','star') or len(row) != 5:
            return math.inf
        ordering, q, scales = tuple(row[2]), tuple(row[3]), np.asarray(row[4],float)
        if sorted(ordering) != list(range(6)) or scales.shape != (6,) or not np.isfinite(scales).all():
            return math.inf
        source = np.asarray(differences,complex)
        if source.shape != (6,6):
            return math.inf
        target = native_differences(topology,q,ordering)
        scales_by_label = scales[np.argsort(ordering)]
        source_pairs = np.asarray([abs(source[i,j]) for i,j in PAIR_LABELS])
        target_pairs = np.asarray([abs(target[i,j]) for i,j in PAIR_LABELS])
        if (not np.isfinite(source_pairs).all() or not np.isfinite(target_pairs).all()
                or np.any(source_pairs <= 0) or np.any(target_pairs <= 0)):
            return math.inf
        expected = np.log(source_pairs) + np.asarray([.5*(scales_by_label[i]+scales_by_label[j]) for i,j in PAIR_LABELS])
        return float(np.max(abs(np.log(target_pairs)-expected)))
    except (ValueError, TypeError, ArithmeticError, IndexError, OverflowError):
        return math.inf


def _faithful_best(rows, topology, differences, domain):
    candidates = []
    for row in rows:
        try:
            q = np.asarray(row[3],complex)
            if q.shape != (3,) or not np.isfinite(q).all() or not np.all((abs(q)>0)&(abs(q)<1)):
                continue
            computed = float(star_gauge(q,domain)) if topology == 'star' else float(abs(q).max())
            gauge = float(row[0])
            # Preserve the generating atlas's arithmetic and tie order.
            # abs(exp(log_r+i*phase)) can differ from exp(log_r) by an ulp.
            # A different domain or an inconsistent rank is still refused.
            if (not 0 <= computed < 1 or not 0 <= gauge < 1
                    or abs(computed-gauge) > 16*np.finfo(float).eps*max(1.,abs(gauge))):
                continue
            candidates.append((gauge,row))
        except (ValueError, TypeError, ArithmeticError, IndexError):
            continue
    # Sorting is stable, so equal-gauge orientations retain the native atlas's
    # order. Coordinate rejection searches the next candidate of this graph.
    for gauge,row in sorted(candidates,key=lambda value:value[0]):
        if coordinate_fidelity_error(differences,topology,row) <= COORDINATE_LOG_TOLERANCE:
            return row
    return None


def select_candidate(candidates, policy=None, differences=None):
    """Return (topology, row), preserving the native five-field row format.

    Candidates are {'comb': rows, 'star': rows}, sorted by their generating
    gauge and containing the available distinct tree orientations. New
    policies validate their gauge and skip coordinate-unfaithful candidates.
    Their star list must be generated in the diagonal domain; routing cannot
    recover orientations already removed by an absolute-domain filter.

    The absolute policy exactly preserves the prior first-per-graph selection,
    including comb winning a tie and the absence of a new fidelity veto.
    No density or source-measure field is read or changed by this function.
    """
    policy = normalize_routing(policy)
    if not isinstance(candidates, Mapping) or set(candidates) - {'comb','star'}:
        raise ValueError('comb/star candidate mapping required')
    if policy['name'] == 'absolute-minimum-v1':
        available = [(name,candidates[name][0]) for name in ('comb','star') if candidates.get(name)]
        if not available:
            raise ArithmeticError('no available absolute-domain sphere candidate')
        return min(available,key=lambda value:value[1][0])
    best = {name:_faithful_best(candidates.get(name,()),name,differences,'diagonal')
            for name in ('comb','star')}
    if policy['name'] == 'diagonal-star-priority-v1' and best['star'] is not None:
        if best['comb'] is not None and best['comb'][0] >= policy['comb_min'] and best['star'][0] <= policy['star_max']:
            return 'star', best['star']
    available = [(name,best[name]) for name in ('comb','star') if best[name] is not None]
    if not available:
        raise ArithmeticError('no coordinate-faithful sphere candidate in the routing domain')
    return min(available,key=lambda value:value[1][0])


__all__ = ['ROUTING_VERSION','COORDINATE_LOG_TOLERANCE','POLICY_NAMES','normalize_routing',
           'star_gauge','coordinate_fidelity_error','select_candidate']
