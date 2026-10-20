"""Momentum proposals and full DOZZ data for the bounded analytic-collar pilot.

Fixed comb ordering is [1,0,5,3,4,2], with the incoming leaf in slot1. The
helper builds no Virasoro coefficient tables. Endpoint scales choose nodes
only: all DOZZ factors and any caller-supplied radial moments remain exact
functions of the three independent momentum nodes.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import math
from pathlib import Path
import time

import mpmath as mp
import numpy as np

from liouville_torus import UpsilonB, log_yin_structure_constant_momentum
from six_point_endpoint_and_tail_quadrature import endpoint_distances, panel_rule
from six_point_momentum_banks_and_reweighting import external_data, explicit_tensor_proposal


VERSION = "analytic-collar-spectral-pilot-v1"
ORDERING = (1, 0, 5, 3, 4, 2)
PROFILES = {
    "coarse": ((3, 3, 3), (3, 4, 5)),  # 6*7*8=336 rows, 48 outer pairs
    "fine": ((4, 4, 4), (4, 5, 6)),    # 8*9*10=720 rows, 80 outer pairs
}


def _source_hashes():
    import liouville_torus
    import six_point_endpoint_and_tail_quadrature
    import six_point_momentum_banks_and_reweighting
    paths = [Path(__file__), Path(liouville_torus.__file__),
             Path(six_point_endpoint_and_tail_quadrature.__file__),
             Path(six_point_momentum_banks_and_reweighting.__file__)]
    return {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def build_collar_spectral_grid(t, outer_radii, *, profile="coarse", anchor_decays=None,
                               core_orders=None, tail_orders=None, special_dps=35,
                               split_multiple=4., tail_scale=2.):
    """Return reusable tensor nodes, dP^3/pi^3 weights, and four coupled DOZZs.

    The exact analytic radial moments have denominators P0²+4*t² and P2²+t².
    Their imaginary pole distances (2*t,t) are included in *proposal* endpoint
    scales, together with the DOZZ distances (1-3*t,1-t). No radial moment,
    polynomial, primary factor or Gaussian is absorbed into the returned DOZZ
    weights. ``log_weighted_dozz`` contains measure plus four DOZZs only.

    Keep outer_radii/anchor_decays fixed to the parent collar when testing
    different stitching cutoffs, so momentum nodes do not move with a cutoff.
    """
    external_momenta, external_weights = external_data(t, 1)
    t = float(t)
    radii = tuple(map(float, outer_radii))
    if len(radii) != 2 or not all(math.isfinite(r) and 0 < r < 1 for r in radii):
        raise ValueError("two finite parent outer radii in (0,1) are required")
    if profile not in PROFILES:
        raise ValueError("profile must be coarse or fine")
    if int(special_dps) != special_dps or special_dps < 25:
        raise ValueError("DOZZ special_dps must be an integer at least25")
    default_core, default_tail = PROFILES[profile]
    core = tuple(default_core if core_orders is None else core_orders)
    tail = tuple(default_tail if tail_orders is None else tail_orders)
    if len(core) != 3 or len(tail) != 3:
        raise ValueError("three core and tail panel orders are required")
    decays = ((max(1.,-2*math.log(radii[0])),1.,max(1.,-2*math.log(radii[1])))
              if anchor_decays is None else tuple(map(float,anchor_decays)))
    if len(decays) != 3 or not all(math.isfinite(a) and a >= 1 for a in decays):
        raise ValueError("three finite proposal decays >=1 are required (bulk width<=1)")
    dozz_distances = endpoint_distances(t, "comb", 1)
    radial_distances = (2*t, None, t)
    distances = (min(dozz_distances[0],2*t),None,min(dozz_distances[2],t))
    rules = [panel_rule(a,delta,n,m,split_multiple=split_multiple,tail_scale=tail_scale)
             for a,delta,n,m in zip(decays,distances,core,tail)]
    proposal = explicit_tensor_proposal(rules, measure="dP", domain="[0,infinity)^3",
        metadata=dict(scheme=VERSION, complete_rule=True, topology="comb",incoming_slot=1,
            ordering=list(ORDERING),t=t,profile=profile,parent_outer_radii=list(radii),
            anchor_decays=list(decays),core_orders=list(core),tail_orders=list(tail),
            dozz_endpoint_distances=list(dozz_distances),radial_endpoint_distances=list(radial_distances),
            effective_endpoint_distances=list(distances),split_multiple=float(split_multiple),
            tail_scale=float(tail_scale),proposal_factors_absorbed=False,
            finite_momentum_cutoff=None,quadrature_certified=False))
    axes = tuple(rule[0] for rule in rules)
    shape = tuple(len(axis) for axis in axes)
    indices = np.indices(shape).reshape(3,-1).T
    sources = _source_hashes()
    started = time.perf_counter()
    calls = 0
    # The exact requested t=.2 puts the incoming numerator at Upsilon_1(0).
    # Positive endpoint momentum nodes keep its denominator arguments off the
    # real Upsilon zeros. This is a genuine zero of the raw BRY coefficient;
    # the shift-based log routine otherwise raises at its gamma-function pole.
    external_zero_slots = [i for i,p in enumerate(external_momenta)
                           if (1+2j*p).imag == 0 and (1+2j*p).real == 0]
    # UpsilonB's implementation sets mp.mp.dps internally. Bound that side
    # effect so building weights cannot clamp a later high-precision block.
    with mp.workdps(int(special_dps)):
        special = UpsilonB(1., dps=int(special_dps))

        @lru_cache(None)
        def dozz(args):
            nonlocal calls
            calls += 1
            return complex(log_yin_structure_constant_momentum(special,*args))

        if external_zero_slots:
            logs = np.full(len(indices),complex(-math.inf,0.))
        else:
            left = np.array([dozz((external_momenta[0],external_momenta[1],p)) for p in axes[0]])
            middle_left = np.array([[dozz((p,external_momenta[2],q)) for q in axes[1]] for p in axes[0]])
            middle_right = np.array([[dozz((q,external_momenta[3],r)) for r in axes[2]] for q in axes[1]])
            right = np.array([dozz((r,external_momenta[4],external_momenta[5])) for r in axes[2]])
            logs = (left[indices[:,0]]+middle_left[indices[:,0],indices[:,1]]
                    +middle_right[indices[:,1],indices[:,2]]+right[indices[:,2]])
        dozz.cache_clear()
    if np.any(np.isnan(logs)) or np.any(np.isposinf(logs.real)) or not np.isfinite(logs.imag).all():
        raise ArithmeticError("nonfinite four-DOZZ weight on positive interior nodes")
    if sources != _source_hashes():
        raise RuntimeError("spectral scientific source changed during weight construction")
    outer_index = indices[:,0]*shape[2]+indices[:,2]
    outer_pairs = np.array([(p,r) for p in axes[0] for r in axes[2]])
    metadata = dict(proposal.metadata)
    metadata.update(source_hashes=sources,proposal_fingerprint=proposal.fingerprint,
                    special_dps=int(special_dps),unique_dozz_evaluations=calls,
                    dozz_seconds=time.perf_counter()-started,outer_pair_count=len(outer_pairs),
                    row_count=len(logs),tensor_shape=list(shape),
                    weight_content="full four BRY/DOZZ logarithms plus dP0*dP1*dP2/pi^3 only",
                    analytic_radial_shifts=[4*t*t,t*t],block_coefficients_built=False,
                    exact_external_upsilon_zero_slots=external_zero_slots,
                    raw_bry_zero=bool(external_zero_slots))
    return dict(p_values=proposal.p_values,internal_weights=1+proposal.p_values**2,
                axis_nodes=axes,axis_weights=tuple(rule[1] for rule in rules),
                log_measure=proposal.log_weights,log_dozz=logs,
                log_weighted_dozz=logs+proposal.log_weights,
                external_momenta=np.asarray(external_momenta),
                external_weights=np.asarray(external_weights),
                row_indices=indices,outer_pair_index=outer_index,middle_index=indices[:,1],
                outer_pairs=outer_pairs,metadata=metadata)


def log_outer_disk_moments(p_values, outer_radii, t, *, levels=(0,0)):
    """Exact product of two Euclidean disk moments, separate from DOZZ weights.

    Returns sum log[pi*R^(2s)/s], with s=P_outer²+(4t²,t²)+level.
    This integrates only the stated monomial. Matter remainders, x factors,
    mixed descendant coefficients, and other measure normalizations belong
    to the caller. Levels are equal holomorphic/antiholomorphic radial levels.
    """
    external_data(t,1)
    p = np.asarray(p_values,float)
    radii = np.asarray(outer_radii,float)
    levels = np.asarray(levels)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all() or np.any(p<0):
        raise ValueError("finite nonnegative N-by-3 momenta required")
    if radii.shape != (2,) or not np.isfinite(radii).all() or np.any(radii<=0) or np.any(radii>=1):
        raise ValueError("two outer radii in (0,1) required")
    if levels.shape != (2,) or not np.isfinite(levels).all() or np.any(levels<0) or np.any(levels!=levels.astype(int)):
        raise ValueError("two nonnegative integer radial levels required")
    s = p[:,(0,2)]**2+np.asarray((4*float(t)**2,float(t)**2))+levels
    if np.any(s<=0):
        raise ArithmeticError("disk moment is not absolutely convergent")
    return np.sum(math.log(math.pi)+2*s*np.log(radii)-np.log(s),axis=1)


__all__ = ["build_collar_spectral_grid","log_outer_disk_moments","PROFILES","VERSION"]
