"""Momentum proposals and one-axis controls used by the six-point curve.

The saved profile fixes the core/tail panel orders and proposal scales.
This module contains no campaign preparation or scheduler operations.
The proposal definition and metadata are unchanged from the historical pilot.
"""
from __future__ import annotations

import numpy as np

from six_point_endpoint_and_tail_quadrature import panel_rule, endpoint_distances
from six_point_momentum_banks_and_reweighting import explicit_tensor_proposal

VARIANTS = ('baseline', 'core0', 'core1', 'core2', 'tail0', 'tail1', 'tail2', 'width_half')


def proposal(profile, variant):
    """Full dP tensor with the saved capped proposal scales, not scope scales."""
    if variant not in VARIANTS:
        raise ValueError('unknown momentum control')
    core, tail = np.array(profile['core'], int), np.array(profile['tail'], int)
    if variant.startswith('core'):
        core[int(variant[-1])] += 2
    if variant.startswith('tail'):
        tail[int(variant[-1])] += 2
    decays = np.array(profile['proposal_decays'], float) * (4 if variant == 'width_half' else 1)
    distances = endpoint_distances(profile['t'], profile['topology'], profile['incoming_slot'])
    rules = [panel_rule(a, d, int(c), int(n), split_multiple=profile['split_multiple'],
                        tail_scale=profile['tail_scale'])
             for a, d, c, n in zip(decays, distances, core, tail)]
    return explicit_tensor_proposal(rules, measure='dP', domain='[0,infinity)^3',
        metadata=dict(scheme='sphere-six-point-fixed-family-multilevel-v1',
            complete_rule=True, variant=variant, profile_name=profile['name'],
            anchor_decays=profile['anchor_decays'], proposal_decays=decays.tolist(),
            core_orders=core.tolist(), tail_orders=tail.tolist(),
            endpoint_distances=list(distances), split_multiple=profile['split_multiple'],
            tail_scale=profile['tail_scale'], conditional_poles_integrated_coupled=True,
            finite_momentum_cutoff=None, anchor_scope_is_error_bound=False))


__all__ = ['VARIANTS', 'proposal']
