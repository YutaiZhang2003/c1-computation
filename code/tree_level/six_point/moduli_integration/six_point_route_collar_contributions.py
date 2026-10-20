"""Physical CFT routing on a labelled, independently marked product collar.

The two short tubes retain the entire primary momentum continuum. Their
leading radial integrals are analytic; the finite-tube remainder is evaluated
pointwise with the degree-two Ward correction and complete matter factor.
The original 1,440-chart importance density is never recomputed or replaced.
Numerical prefix comparisons are diagnostics, not uncomputed-tail bounds.
"""
from __future__ import annotations

from dataclasses import asdict
from functools import lru_cache
import math

import numpy as np

from six_point_analytic_collar_subtraction import (
    CollarPatch, channel_exponents, regular_matter_log_ratio,
)
from six_point_collar_momentum_quadrature import (
    build_collar_spectral_grid, log_outer_disk_moments,
)
from six_point_collar_block_evaluator import SphereSixPointCollarBlock
from six_point_collar_counterterm_geometry import patch_geometry
from six_point_sample_integration_geometry import SphereGeometryPoint
from dataclasses import dataclass, field
from typing import Mapping
from six_point_integrate_collar_patch import external_weights, representation_for

VERSION = 'sphere-six-point-physical-collar-router-v1'
NORMALIZATION = 'sphere-six-point-I6-dP3-over-pi3-four-BRY-DOZZ-v1'
TESTED_PATCH = CollarPatch(.02, .01, (-.435, -.405, -.898, -.878))
TESTED_PATCH_NAME = 'pair01_pair24_pillow_corner_v1'
PROFILES = {'coarse': None, 'fine': None,
            'refined': ((6,6,6), (6,7,8)),
            'high': ((8,8,8), (8,9,10)),
            'reference': ((10,10,10), (10,11,12))}


@dataclass(frozen=True)
class CheckedIntegrand:
    """A scalar/vector integrand with its unchanged CFT resolution status.

    ``resolved=None`` means unchecked; it is never promoted to resolved by
    applying this control. Strict production callbacks should raise on a
    refused CFT evaluation rather than returning a replacement value.
    """
    value: object
    resolved: bool | None=None
    diagnostics: Mapping[str,object]=field(default_factory=dict)

    def __post_init__(self):
        if self.resolved is not None and not isinstance(self.resolved,(bool,np.bool_)):
            raise TypeError("resolved must be a boolean or None, never an inferred truth value")
        if not isinstance(self.diagnostics,Mapping):
            raise TypeError("integrand diagnostics must be a mapping")
        if self.resolved is not None:
            object.__setattr__(self,'resolved',bool(self.resolved))
        object.__setattr__(self,'diagnostics',dict(self.diagnostics))


def point_from_plan(plan, index):
    """Reconstruct a record without changing selected or source coordinates."""
    a, i = plan.arrays, int(index)
    return SphereGeometryPoint(tuple(a['points'][i]), ('comb','star')[int(a['topology'][i])],
        tuple(map(int,a['ordering'][i])), tuple(a['q'][i]), float(a['log_density'][i]),
        tuple(a['pair_logs'][i]), tuple(a['scale_logs'][i]), float(a['score'][i]),
        float(a['convergence_score'][i]), int(a['recovery'][i]),
        ('comb','star')[int(a['sampled_topology'][i])],
        tuple(map(int,a['sampled_ordering'][i])), tuple(a['sampled_q'][i]))


def _sum(log_weights, values):
    """Signed real amplitudes times complex DOZZ phases, retaining log scale."""
    weights = np.asarray(log_weights, np.clongdouble)
    values = np.asarray(values, np.longdouble)
    mask = (values != 0) & (weights.real != -np.inf)
    if not mask.any():
        return dict(value=0j, log_value=complex(-math.inf,0), cancellation_condition=None)
    logs = weights[mask] + np.log(abs(values[mask]))
    top = logs.real.max()
    scaled = np.sign(values[mask])*np.exp(logs-top)
    total = scaled.sum(dtype=np.clongdouble)
    if total == 0:
        return dict(value=0j, log_value=complex(-math.inf,0), cancellation_condition=None)
    log_value = complex(top+np.log(abs(total)), np.angle(total))
    return dict(value=_exp(log_value), log_value=log_value,
                cancellation_condition=float(abs(scaled).sum()/abs(total)))


def _exp(log_value):
    if log_value.real == -math.inf:
        return 0j
    with np.errstate(over='raise', invalid='raise'):
        value = complex(np.exp(np.clongdouble(log_value)))
    if not math.isfinite(value.real) or not math.isfinite(value.imag):
        raise FloatingPointError('collar density exceeds binary64 range')
    return value


class CollarCFTEvaluator:
    """Reusable four-point coefficient banks for one energy and momentum rule.

    This evaluator deliberately rejects other incoming slots: the present
    spectral-grid DOZZ assignment is verified for slot 1. A caller cannot
    silently relabel a physically different face as the tested patch.
    """
    def __init__(self, t, patch=TESTED_PATCH, *, profile='refined',
                 block_orders=(8,12), dps=60):
        self.t, self.patch = float(t), patch
        if not isinstance(patch, CollarPatch) or patch.ordering.index(0) != 1:
            raise ValueError('the current collar spectral rule requires incoming slot 1')
        orders = tuple(block_orders)
        if (not orders or any(isinstance(n,bool) or int(n)!=n or not 0<=n<=24 for n in orders)
                or list(orders) != sorted(set(orders))):
            raise ValueError('distinct increasing elliptic orders between 0 and 24 required')
        if profile not in PROFILES:
            raise ValueError('unknown collar momentum profile')
        self.block_orders, self.profile, self.dps = tuple(map(int,orders)), profile, int(dps)
        extra = PROFILES[profile]
        kwargs = {} if extra is None else dict(core_orders=extra[0], tail_orders=extra[1])
        self.grid = build_collar_spectral_grid(self.t,(patch.left_radius,patch.right_radius),
            profile=profile if extra is None else 'fine', **kwargs)
        self.ps = np.asarray(self.grid['p_values'],float)
        self.hs = 1+self.ps*self.ps
        self.logw = np.asarray(self.grid['log_weighted_dozz'],complex)
        self.ds = external_weights(self.t,patch.ordering)
        self.beta = np.asarray([channel_exponents(self.t,p,patch.ordering) for p in self.ps])
        self.modes = tuple(representation_for(self.ds,h) for h in self.hs)
        self.groups = {}
        for i,p in enumerate(self.ps):
            self.groups.setdefault((float(p[0]),float(p[2])),[]).append(i)
        self.cores, self.addbacks = {}, {}

    def _core(self, pair):
        if pair not in self.cores:
            self.cores[pair] = SphereSixPointCollarBlock(self.ds,
                (1+pair[0]**2,1+pair[1]**2), order=max(self.block_orders),dps=self.dps)
        return self.cores[pair]

    @lru_cache(maxsize=24)
    def _slices(self, x):
        values = {n: {k:np.empty(len(self.ps),complex)
                  for k in ((0,0),(1,0),(0,1),(2,0),(1,1),(0,2))}
                  for n in self.block_orders}
        for pair,indices in self.groups.items():
            core = self._core(pair)
            for i in indices:
                for n in self.block_orders:
                    slices = core.slices(self.hs[i,1],x,elliptic_order=n,
                                         outer_order=2,representation=self.modes[i])
                    for key in values[n]:
                        values[n][key][i] = complex(slices[key])
        for row in values.values():
            for vector in row.values():
                vector.flags.writeable = False
        return values

    def _diagnostics(self):
        return dict(version=VERSION, normalization_id=NORMALIZATION,
            momentum_profile=self.profile, momentum_nodes=len(self.ps),
            elliptic_orders=self.block_orders, outer_total_degree=2,
            all_momentum_nodes_retained=True,
            pointwise_disk_moments_included=False,
            representation_counts={m:self.modes.count(m) for m in ('standard','complete')},
            representation_policy='smaller first nome coefficient, fixed across prefixes',
            full_CFT_certified=False, momentum_convergence_checked=False,
            uncomputed_descendant_tail_bounded=False)

    def _point_values(self, q, *, log_measure_shift=0.):
        a,x,b = map(complex,q)
        if not self.patch.contains((a,x,b)) or min(abs(a),abs(b)) == 0:
            raise ValueError('a nonsingular point strictly inside this physical collar is required')
        log_ratio = float(regular_matter_log_ratio(self.t,self.patch.ordering,a,x,b))
        log_R0 = self.t*self.t*math.log(abs(1-x)) # incoming slot 1
        # This is a three-term scalar exponent, not a dense matrix operation.
        # Explicit summation also avoids spurious floating-status warnings
        # observed from the local BLAS path for finite, identical inputs.
        powers = np.sum((self.beta-2)*np.log(np.abs((a,x,b)))[None,:],axis=1)
        logs = self.logw + powers + log_R0 + log_measure_shift
        vectors, sums = {}, {}
        for n,s in self._slices(x).items():
            primary = np.abs(s[0,0])**2
            delta1 = a*s[1,0]+b*s[0,1]
            delta2 = delta1+a*a*s[2,0]+a*b*s[1,1]+b*b*s[0,2]
            remainder = np.expm1(log_ratio)*primary+math.exp(log_ratio)*(
                2*np.real(np.conj(s[0,0])*delta2)+np.abs(delta2)**2)
            outer1 = math.exp(log_ratio)*np.abs(s[0,0]+delta1)**2
            vectors[n] = dict(full=primary+remainder, leading=primary,
                              remainder=remainder,outer1=outer1)
            sums[str(n)] = {key:_sum(logs,value) for key,value in vectors[n].items()}
        last = self.block_orders[-1]
        full = sums[str(last)]['full']['value']
        denom = abs(full)
        change, flagged = {}, {}
        for n in self.block_orders:
            difference = abs(vectors[n]['full']-vectors[last]['full'])
            flag = difference > .01*np.maximum(abs(vectors[last]['full']),np.finfo(float).tiny)
            change[str(n)] = (_sum(logs.real,difference)['value'].real/denom if denom else None)
            flagged[str(n)] = dict(count=int(flag.sum()), reference_weight_fraction=(
                _sum(logs.real,np.where(flag,abs(vectors[last]['full']),0.))['value'].real/denom
                if denom else None), reference_weight_is_tail_bound=False)
        diag = self._diagnostics()
        diag.update(absolute_weighted_order_change=change, node_order_flags=flagged,
            outer1_vs2_relative=(abs(sums[str(last)]['outer1']['value']/full-1) if denom else None),
            leading_vs_outer2_relative=(abs(sums[str(last)]['remainder']['value']/full) if denom else None))
        return dict(by_order={n:{key:row['value'] for key,row in entries.items()}
                              for n,entries in sums.items()},
                    log_by_order={n:{key:row['log_value'] for key,row in entries.items()}
                                  for n,entries in sums.items()}, diagnostics=diag)

    def evaluate_q(self, q):
        """Density including full DOZZ and dP³/pi³, in d²a d²x d²b."""
        return self._point_values(q)

    def evaluate(self, point):
        """Importance-weighted density in the original frozen source frame."""
        geometry = patch_geometry(point,self.patch)
        if not geometry.inside:
            raise ValueError('point is outside the labelled collar')
        shift = -2*geometry.log_source_inverse_jacobian-geometry.source_log_density
        result = self._point_values(geometry.q,log_measure_shift=shift)
        result['geometry'] = asdict(geometry)
        result['diagnostics']['source_mixture_density_unchanged'] = True
        return result

    def prepare_addback(self, x_order=2):
        """Integral of precisely the leading pointwise counterterm on this patch.

        Only the two complex radial disks are analytic. The middle complex
        coordinate and all three coupled Liouville momenta remain quadrature.
        No overall string normalization or permutation multiplier is inserted.
        """
        if x_order in self.addbacks:
            return self.addbacks[x_order]
        xs,weights = self.patch.middle_rule(x_order)
        disk_logs = log_outer_disk_moments(self.ps,
            (self.patch.left_radius,self.patch.right_radius),self.t)
        parts = {n:[] for n in self.block_orders}
        for x,weight in zip(xs,weights):
            logs = self.logw+disk_logs+(self.beta[:,1]-2)*math.log(abs(x))
            logs += self.t*self.t*math.log(abs(1-x))+math.log(weight)
            for n,slices in self._slices(complex(x)).items():
                parts[n].append(_sum(logs,abs(slices[0,0])**2)['value'])
        result = dict(by_order={str(n):sum(values) for n,values in parts.items()},
            x_order=int(x_order), x_nodes=len(xs), momentum_nodes=len(self.ps),
            normalization_id=NORMALIZATION, patch=asdict(self.patch),
            radial_disks_analytic=True, middle_and_momenta_numerical=True,
            full_CFT_certified=False, analytic_addback_matched_to_pointwise_counterterm=True)
        self.addbacks[x_order] = result
        return result


class CollarRouter:
    """Bypass the native bank inside the marked patch; preserve outside values.

    The caller averages ``remainder`` over all original samples and adds the
    matched patch integral once. ``corrected`` instead distributes that same
    constant across every sample; averaging it gives the identical estimator.
    """
    def __init__(self,evaluator,*,x_order=2):
        self.evaluator, self.patch, self.x_order = evaluator,evaluator.patch,int(x_order)

    def contains(self,point):
        return patch_geometry(point,self.patch).inside

    def route(self,point,fallback):
        addback = self.evaluator.prepare_addback(self.x_order)['by_order'][str(max(self.evaluator.block_orders))]
        if self.contains(point):
            result = self.evaluator.evaluate(point)
            row = result['by_order'][str(max(self.evaluator.block_orders))]
            diagnostics = dict(result['diagnostics'], route=TESTED_PATCH_NAME,
                               finite_prefix_qualification_is_tail_bound=False)
            full = CheckedIntegrand(row['full'],None,diagnostics)
            remainder = CheckedIntegrand(row['remainder'],None,dict(diagnostics))
            control = row['leading']
        else:
            full = fallback(point)
            if not isinstance(full,CheckedIntegrand):
                raise TypeError('fallback must return CheckedIntegrand')
            declared = full.diagnostics.get('normalization_id')
            if declared is not None and declared != NORMALIZATION:
                raise ValueError('outside evaluator and collar normalization differ')
            remainder = full
            control = 0j
        corrected = CheckedIntegrand(remainder.value+addback,remainder.resolved,dict(remainder.diagnostics))
        return dict(full=full, control=control, remainder=remainder, corrected=corrected,
                    analytic_addback=addback, inside_patch=self.contains(point),
                    normalization_id=NORMALIZATION, source_mixture_density_unchanged=True)
