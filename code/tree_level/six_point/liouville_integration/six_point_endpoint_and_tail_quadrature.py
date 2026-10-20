"""Explicit endpoint/core and infinite-tail proposals for sphere six points.

This promotes the rule tested in endpoint_cft_20260908 without changing any
existing production default. Anchors select nodes only; all CFT factors and
the original dP measure must still be evaluated at every node. Anchor matching
is a qualification screen, never a quadrature error estimate.
"""
from __future__ import annotations

from functools import lru_cache
import math
import numpy as np

from six_point_momentum_banks_and_reweighting import external_data, explicit_tensor_proposal

VERSION = "sphere-six-point-endpoint-core-tail-v1"


def endpoint_distances(t, topology, incoming_slot):
    external_data(t, incoming_slot)
    if topology == "star":
        return tuple(1 - 3*t if incoming_slot//2 == e else 1-t for e in range(3))
    if topology == "comb":
        return (1-3*t if incoming_slot in (0, 1) else 1-t, None,
                1-3*t if incoming_slot in (4, 5) else 1-t)
    raise ValueError("unknown six-point topology")


@lru_cache(None)
def _unit_legendre(order):
    if isinstance(order, (bool, np.bool_)) or int(order) != order or order < 1:
        raise ValueError("positive integer panel order required")
    x, w = np.polynomial.legendre.leggauss(int(order))
    x, w = (x + 1)/2, w/2
    x.setflags(write=False)
    w.setflags(write=False)
    return x, w


def panel_rule(anchor_decay, delta, core_order=8, tail_order=16, *,
               split_multiple=4., tail_scale=2.):
    """Return exact dP weights; delta=None denotes a conditional pole edge.

    w=1/sqrt(a), s=min(4*delta,w), or s=w without an anchored pole.
    The core is P=s*u; the tail is P=s-2*w*log(1-u). In particular,
    a delta of order one does not enlarge the core beyond the bulk width.
    """
    if not all(math.isfinite(v) and v > 0 for v in (anchor_decay, split_multiple, tail_scale)):
        raise ValueError("finite positive scales required")
    if delta is not None and not (math.isfinite(delta) and delta > 0):
        raise ValueError("anchored pole distance must be positive before the first wall")
    width = 1/math.sqrt(anchor_decay)
    split = width if delta is None else min(split_multiple*delta, width)
    u, v = _unit_legendre(core_order)
    x, y = _unit_legendre(tail_order)
    tail_width = tail_scale*width
    nodes = np.concatenate((split*u, split-tail_width*np.log1p(-x)))
    weights = np.concatenate((split*v, tail_width*y/(1-x)))
    if not (np.isfinite(nodes).all() and np.isfinite(weights).all()
            and np.all(nodes > 0) and np.all(weights > 0)):
        raise FloatingPointError("unrepresentable endpoint/tail rule")
    return nodes, weights


def named_variants():
    """Baseline and SIX independent one-axis controls; no cancelling all-axis step."""
    result = {"baseline": dict(core_orders=[8]*3, tail_orders=[16]*3)}
    for edge in range(3):
        core, tail = [8]*3, [16]*3
        core[edge] = 12
        result[f"core_axis{edge}"] = dict(core_orders=core, tail_orders=tail)
        core, tail = [8]*3, [16]*3
        tail[edge] = 24
        result[f"tail_axis{edge}"] = dict(core_orders=core, tail_orders=tail)
    return result


def endpoint_proposal(t, topology, incoming_slot, anchor_decays, *,
                      core_orders=(8, 8, 8), tail_orders=(16, 16, 16),
                      split_multiple=4., tail_scale=2., variant="baseline"):
    if any(len(x) != 3 for x in (anchor_decays, core_orders, tail_orders)):
        raise ValueError("three edge scales and panel orders required")
    distances = endpoint_distances(t, topology, incoming_slot)
    rules = [panel_rule(float(a), d, n, m, split_multiple=split_multiple, tail_scale=tail_scale)
             for a, d, n, m in zip(anchor_decays, distances, core_orders, tail_orders)]
    return explicit_tensor_proposal(rules, measure="dP", domain="[0,infinity)^3",
        metadata=dict(scheme=VERSION, complete_rule=True, variant=variant,
            anchor_decays=list(map(float, anchor_decays)), endpoint_distances=list(distances),
            core_orders=list(map(int, core_orders)), tail_orders=list(map(int, tail_orders)),
            split_multiple=float(split_multiple), tail_scale=float(tail_scale),
            conditional_poles_integrated_coupled=True,
            anchor_scope_is_error_bound=False,
            finite_momentum_cutoff=None))


__all__ = ["endpoint_distances", "panel_rule", "endpoint_proposal", "named_variants"]
