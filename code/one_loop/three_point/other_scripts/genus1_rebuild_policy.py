"""Acceptance policy for the corrected genus-one 30-frequency rebuild.

The user caps each native conformal-block order at 10 and authorizes retaining
unresolved cap-limited points provisionally. This does not relax sewing-domain,
coefficient-arithmetic or momentum-quadrature checks. No matrix target is used.
The requested empirical refinement tolerance is now 0.5%, not 0.1%; it is not
a rigorous total-amplitude error bound.
"""
from __future__ import annotations

import math


POLICY_VERSION = 'order10-first-pass-v1'
BLOCK_ORDER_CAP = 10
REFINEMENT_RTOL = 5e-3
MAXIMUM_CORES = 192
SOBOL_POWER = 13
SCRAMBLES = 16
FREQUENCY_T_VALUES = (.02,.05,.08,.12,.15,.18,.22,.25,.28,.32,.35,.38,
                      .42,.45,.48,.52,.55,.58,.62,.65,.68,.72,.75,.78,
                      .82,.85,.88,.92,.95,.98)


def require_orders(orders):
    if (len(orders) != 3 or any(isinstance(n, bool) or not math.isfinite(n) or
            int(n) != n or n < 0 or n > BLOCK_ORDER_CAP for n in orders)):
        raise ValueError('three native block orders between 0 and 10 required')
    return list(map(int, orders))


def classify_cft_point(*, orders, axis_changes, block_rtol=REFINEMENT_RTOL,
                       chart_admissible, coefficients_verified,
                       momentum_converged):
    """Classify already evaluated data; never manufacture a converged value.

    axis_changes contains two successive relative block-order changes per
    native sewing axis. For an order-10 axis the first is the 8->10 change.
    The changes are empirical diagnostics, not rigorous remainder bounds.
    """
    orders = require_orders(orders)
    if not math.isfinite(block_rtol) or not 0 < block_rtol < 1:
        raise ValueError('invalid block-refinement tolerance')
    if (len(axis_changes) != 3 or any(len(row) != 2 for row in axis_changes) or
            any(not math.isfinite(x) or x < 0 for row in axis_changes for x in row)):
        raise ValueError('two finite nonnegative block changes per axis required')
    failed = [i for i, row in enumerate(axis_changes) if max(row) > block_rtol]
    at_cap = [i for i, n in enumerate(orders) if n == BLOCK_ORDER_CAP]
    # Strings such as "false" must never become passing evidence by coercion.
    if any(type(x) is not bool for x in
           (chart_admissible, coefficients_verified, momentum_converged)):
        raise ValueError('explicit boolean prerequisite evidence required')
    prerequisites = dict(chart_admissible=chart_admissible,
                         coefficients_verified=coefficients_verified,
                         momentum_converged=momentum_converged)
    if not all(prerequisites.values()):
        status = 'unresolved_prerequisite'
    elif not failed:
        status = 'accepted_empirical_refinement'
    elif all(i in at_cap for i in failed):
        status = 'accepted_provisionally_at_block_cap'
    else:
        status = 'needs_block_refinement'
    return dict(policy_version=POLICY_VERSION, status=status,
                include_in_amplitude=status.startswith('accepted_'),
                provisional=status=='accepted_provisionally_at_block_cap',
                orders=list(orders), block_order_cap=BLOCK_ORDER_CAP,
                capped_axes=at_cap, failed_block_axes=failed,
                axes_to_refine=[i for i in failed if i not in at_cap],
                block_rtol=block_rtol, axis_changes=[list(row) for row in axis_changes],
                cap_8_to_10_relative_changes={str(i):axis_changes[i][0] for i in at_cap},
                prerequisites=prerequisites, rigorous_truncation_bound=None,
                qualification='Capped unresolved points are retained with a provisional marker; sampling error does not cover their unresolved block truncation.')


def next_orders(classification):
    """Refine only failed, uncapped axes; never extend beyond this pass's cap."""
    check_classification(classification)
    if not all(classification['prerequisites'].values()):
        raise ValueError('repair prerequisite failures before block refinement')
    orders = list(classification['orders'])
    for axis in classification['axes_to_refine']:
        orders[axis] = min(BLOCK_ORDER_CAP, orders[axis] + 2)
    return require_orders(orders)


def check_classification(row):
    """Reject legacy or edited acceptance flags at every downstream boundary."""
    expected = classify_cft_point(orders=row['orders'], axis_changes=row['axis_changes'],
        block_rtol=row['block_rtol'], **row['prerequisites'])
    for key, value in expected.items():
        if row.get(key) != value:
            raise ValueError('incompatible acceptance record: '+key)
    if row['block_rtol'] != REFINEMENT_RTOL:
        raise ValueError('first-pass refinement tolerance changed')
    return row


def encoded(z):
    z = complex(z)
    if not math.isfinite(z.real) or not math.isfinite(z.imag):
        raise ValueError('nonfinite CFT value or weighted contribution')
    return [z.real, z.imag]


def weighted_cft_record(classification, *, value, weight, cap_prefix_values,
                        point_id):
    """Attach an ALREADY normalized MC integration weight to one CFT value.

    The caller supplies its true Jacobian, stratum/sample normalization and
    any amplitude prefactor. A pointwise pilot cannot infer these weights.
    Axis-prefix changes are kept separately, not summed into an error bound.
    """
    check_classification(classification)
    if not classification['include_in_amplitude']:
        raise ValueError('unaccepted CFT data cannot enter amplitude reduction')
    if not point_id:
        raise ValueError('a stable sample/term identity is required')
    value, weight = complex(*encoded(value)), complex(*encoded(weight))
    axes = [str(i) for i in classification['capped_axes']]
    if set(cap_prefix_values) != set(axes):
        raise ValueError('every capped axis requires its actual order-8 prefix value')
    deltas = {}
    for axis in axes:
        prefix = complex(*encoded(cap_prefix_values[axis]))
        relative = abs(value-prefix)/max(abs(value), 1e-300)
        if not math.isclose(relative, classification['cap_8_to_10_relative_changes'][axis],
                            rel_tol=1e-9, abs_tol=1e-14):
            raise ValueError('order-8 prefix does not match its refinement record')
        deltas[axis] = encoded(weight*(value-prefix))
    return dict(point_id=point_id, classification=classification,
        contribution=encoded(weight*value), cap_8_to_10_weighted_changes=deltas)


def contribution_diagnostics(records):
    """Reduce additive, correctly normalized contributions without altering MC SE.

    Signed cap-limited contributions and absolute sums are both retained to
    expose cancellations. These are empirical diagnostics, NOT MC variances
    or rigorous bounds. Include region/stratum/scramble in each point identity.
    """
    rows = list(records)
    ids = [r['point_id'] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate weighted CFT record')
    for r in rows:
        c = check_classification(r['classification'])
        if not c['include_in_amplitude']:
            raise ValueError('unaccepted CFT point in amplitude result')
        if set(r['cap_8_to_10_weighted_changes']) != {str(i) for i in c['capped_axes']}:
            raise ValueError('missing capped-axis diagnostic')
    def total(values):
        zs = [complex(*encoded(z)) for z in values]
        return encoded(complex(math.fsum(z.real for z in zs), math.fsum(z.imag for z in zs)))
    provisional = [r for r in rows if r['classification']['provisional']]
    return dict(policy_version=POLICY_VERSION, block_order_cap=BLOCK_ORDER_CAP,
        provisional=bool(provisional), provisional_points=len(provisional), points=len(rows),
        provisional_contribution=total(complex(*r['contribution']) for r in provisional),
        provisional_absolute_contribution_sum=math.fsum(abs(complex(*r['contribution'])) for r in provisional),
        cap_8_to_10_weighted_changes={str(i): total(complex(*r['cap_8_to_10_weighted_changes'][str(i)])
            for r in rows if str(i) in r['cap_8_to_10_weighted_changes']) for i in range(3)},
        rigorous_truncation_bound=None, included_in_mc_standard_error=False)


def attach_contribution_diagnostics(result, records):
    """The future bank-backed amplitude reducer uses this without changing its SE."""
    if 'block_truncation_diagnostics' in result:
        raise ValueError('refusing to overwrite existing systematic diagnostics')
    diagnostics = contribution_diagnostics(records)
    return dict(result, provisional=bool(result.get('provisional', False) or diagnostics['provisional']),
                block_truncation_diagnostics=diagnostics)


def self_test():
    good = [[1e-5,2e-5]]*3
    common = dict(chart_admissible=True,coefficients_verified=True,momentum_converged=True)
    assert classify_cft_point(orders=[4,4,2],axis_changes=good,**common)['status']=='accepted_empirical_refinement'
    assert classify_cft_point(orders=[4,4,2],axis_changes=[[.004,.003]]*3,**common)['include_in_amplitude']
    assert not classify_cft_point(orders=[4,4,2],axis_changes=[[.006,.003]]*3,**common)['include_in_amplitude']
    bad = [[.02,.03],good[1],good[2]]
    capped = classify_cft_point(orders=[10,6,2],axis_changes=bad,**common)
    assert capped['include_in_amplitude'] and capped['provisional'] and capped['capped_axes']==[0]
    assert capped['cap_8_to_10_relative_changes']=={'0':.02}
    assert not classify_cft_point(orders=[8,6,2],axis_changes=bad,**common)['include_in_amplitude']
    mixed = classify_cft_point(orders=[10,6,2],axis_changes=[bad[0],bad[0],good[2]],**common)
    assert mixed['axes_to_refine']==[1] and not mixed['include_in_amplitude']
    for key in common:
        assert not classify_cft_point(orders=[10,6,2],axis_changes=bad,**dict(common,**{key:False}))['include_in_amplitude']
    passed_at_cap = classify_cft_point(orders=[10,6,2],axis_changes=good,**common)
    assert passed_at_cap['capped_axes']==[0] and not passed_at_cap['provisional']
    try:
        classify_cft_point(orders=[12,6,2],axis_changes=good,**common)
    except ValueError:
        pass
    else:
        raise AssertionError('block cap was not enforced')
    assert next_orders(capped) == [10,6,2]
    assert next_orders(mixed) == [10,8,2]
    weighted = weighted_cft_record(capped, value=1j, weight=2.,
                                  cap_prefix_values={'0':.98j}, point_id='test:0')
    output = attach_contribution_diagnostics({'mc_standard_error': .3}, [weighted])
    assert output['provisional'] and output['mc_standard_error'] == .3
    assert output['block_truncation_diagnostics']['provisional_contribution'] == [0.,2.]
    return dict(status='passed',block_order_cap=10,refinement_rtol=REFINEMENT_RTOL,provisional_cap_acceptance=True,
                marker_propagation=True,mc_standard_error_unchanged=True,
                below_cap_requires_refinement=True,other_accuracy_gates_preserved=True)


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(),indent=2))
