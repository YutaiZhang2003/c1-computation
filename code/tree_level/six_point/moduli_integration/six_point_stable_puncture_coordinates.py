"""Stable six-sphere atlas from factorized native puncture differences.

No large finite cross ratio is identified with infinity, and a small star
separation is never reconstructed by subtracting 1 from 1+q. Jacobians follow
from the product of all six local Möbius scales in the fixed puncture gauge.
"""
from __future__ import annotations

from functools import lru_cache
from fractions import Fraction
import math

import mpmath as mp
import numpy as np

import six_point_channel_coordinates as atlas
from six_point_radial_sampling_distribution import _power_disk_sample
from six_point_sample_integration_geometry import SphereGeometryPoint, PAIR_LABELS, TOPOLOGIES
from six_point_radial_sampling_distribution import (
    mixture_log_radial_density, normalize_radial_proposal, sample_source,
)

NATIVE_GEOMETRY_VERSION = "sphere-factorized-differences-v3-cancellation-safe-source"


@lru_cache(None)
def _orderings():
    orders = np.asarray(atlas.oriented_comb_orderings(), dtype=int)
    return orders, tuple(atlas.comb_tree_signature(o) for o in orders), tuple(atlas.star_tree_signature(o) for o in orders)


class NativeDifferenceTable(np.ndarray):
    """Determinants plus the native source needed for ambiguous support tests."""

    def __array_finalize__(self, source):
        self.native_source = getattr(source, "native_source", None)
        self.ambiguous_ratio_count = getattr(source, "ambiguous_ratio_count", 0)
        self.source_cancellation_refined = getattr(source, "source_cancellation_refined", False)


class _ExactComplex:
    """Gaussian rational used only to decide near-unit proposal membership.

    Native complex floats are exact binary rationals. Comparing squared
    numerator and denominator norms therefore decides |q|<1 without any
    precision-dependent sign convention, including exact unit-circle points.
    """

    def __init__(self, real=0, imag=0):
        self.real, self.imag = Fraction(real), Fraction(imag)

    @staticmethod
    def cast(value):
        return value if isinstance(value, _ExactComplex) else _ExactComplex(value)

    def __add__(self, other):
        other = self.cast(other)
        return _ExactComplex(self.real+other.real, self.imag+other.imag)

    __radd__ = __add__

    def __neg__(self):
        return _ExactComplex(-self.real, -self.imag)

    def __sub__(self, other):
        return self + (-self.cast(other))

    def __rsub__(self, other):
        return self.cast(other) - self

    def __mul__(self, other):
        other = self.cast(other)
        return _ExactComplex(self.real*other.real-self.imag*other.imag,
                             self.real*other.imag+self.imag*other.real)

    __rmul__ = __mul__

    @property
    def norm_squared(self):
        return self.real*self.real+self.imag*self.imag

    def __truediv__(self, other):
        other = self.cast(other)
        numerator = self*_ExactComplex(other.real,-other.imag)
        norm = other.norm_squared
        return _ExactComplex(numerator.real/norm,numerator.imag/norm)

    def __rtruediv__(self, other):
        return self.cast(other)/self


def _difference_entries(topology, q):
    q1, q2, q3 = q
    if topology == "comb":
        x, y = q1 * q2 * q3, q2 * q3
        values = {(1,0): x, (2,0): y, (3,0): q3, (4,0): 1,
                  (2,1): y*(1-q1), (3,1): q3*(1-q1*q2), (4,1): 1-x,
                  (3,2): q3*(1-q2), (4,2): 1-y, (4,3): 1-q3}
    elif topology == "star":
        values = {(1,0): q1, (2,0): 1, (3,0): 1+q2, (4,0): 1/q3,
                  (2,1): 1-q1, (3,1): (1-q1)+q2, (4,1): (1-q1*q3)/q3,
                  (3,2): q2, (4,2): (1-q3)/q3, (4,3): ((1-q3)-q2*q3)/q3}
    else:
        raise ValueError("unknown sphere graph")
    values.update({(5,i): 1 for i in range(5)})
    return values


def _source_cancellation_requires_exact(topology, q, values):
    """Detect loss of relative accuracy in source puncture separations.

    Factorized native coordinates avoid the small-q cancellation but still
    contain cross-pair differences such as 1-q1+q2 and 1-q3-q2*q3. Near those
    divisors a correct target-chart check cannot repair an inaccurate source
    table. Refine when a subtraction loses roughly three decimal digits;
    ordinary points retain their existing float arithmetic.
    """
    if any(value == 0 or not math.isfinite(abs(value)) for value in values.values()):
        return True
    q1,q2,q3=q
    if topology == "comb":
        checks=((1-q1,1+abs(q1)),(1-q2,1+abs(q2)),(1-q3,1+abs(q3)),
                (1-q1*q2,1+abs(q1)*abs(q2)),
                (1-q2*q3,1+abs(q2)*abs(q3)),
                (1-q1*q2*q3,1+abs(q1)*abs(q2)*abs(q3)))
    else:
        checks=((1-q1,1+abs(q1)),(1+q2,1+abs(q2)),(1-q3,1+abs(q3)),
                ((1-q1)+q2,1+abs(q1)+abs(q2)),
                (1-q1*q3,1+abs(q1)*abs(q3)),
                ((1-q3)-q2*q3,1+abs(q3)+abs(q2)*abs(q3)))
    return any(abs(value) <= 1e-3*scale for value,scale in checks)


def _exact_source_difference_entries(topology, q):
    """Round exact rational functions of the original binary floats once."""
    exact_q=tuple(_ExactComplex(z.real,z.imag) for z in q)
    values={}
    for key,value in _difference_entries(topology,exact_q).items():
        value=_ExactComplex.cast(value)
        try:
            values[key]=complex(float(value.real),float(value.imag))
        except (OverflowError,ValueError) as exc:
            raise ArithmeticError("native puncture difference is unrepresentable") from exc
    return values


def native_differences(topology, q, ordering=tuple(range(6))):
    """Homogeneous determinants D_ij in the exact sampled finite/∞ frame."""
    q = tuple(map(complex,q))
    ordering = tuple(ordering)
    if len(q)!=3 or any(z == 0 or not math.isfinite(abs(z)) for z in q):
        raise ValueError("nonzero finite native sewing parameters required")
    if tuple(sorted(ordering)) != tuple(range(6)):
        raise ValueError("six distinct puncture labels required")
    d = np.zeros((6, 6), dtype=complex).view(NativeDifferenceTable)
    d.native_source = (topology,q,ordering)
    values=_difference_entries(topology,q)
    if _source_cancellation_requires_exact(topology,q,values):
        values=_exact_source_difference_entries(topology,q)
        d.source_cancellation_refined=True
    for (i,j), value in values.items():
        if value == 0 or not math.isfinite(abs(value)):
            raise ArithmeticError("native puncture difference is zero or unrepresentable")
        a, b = ordering[i], ordering[j]
        d[a,b], d[b,a] = value, -value
    return d


def _exact_difference_norms(source):
    topology,q,ordering = source
    exact_q = tuple(_ExactComplex(z.real,z.imag) for z in q)
    table = {}
    for (i,j), value in _difference_entries(topology,exact_q).items():
        value = _ExactComplex.cast(value)
        a,b = ordering[i],ordering[j]
        table[tuple(sorted((a,b)))] = value.norm_squared
    return table


def _exact_ratio_log_radius(norms, topology, ordering, axis, cache):
    a,x,y,z,e,c = ordering
    factors = (((x,a),(y,c),(x,c),(y,a)),
               ((y,a),(z,c),(y,c),(z,a)),
               ((z,a),(e,c),(z,c),(e,a))) if topology=="comb" else (
               ((x,a),(y,c),(x,c),(y,a)),
               ((a,c),(z,y),(y,a),(z,c)),
               ((e,c),(y,a),(e,a),(y,c)))
    n1,n2,d1,d2 = factors[axis]
    # Norms remove determinant signs. Redundant orientations reuse the same
    # at most 90 cross-ratio moduli, including the star q2 sign convention.
    numerator = tuple(sorted((tuple(sorted(n1)),tuple(sorted(n2)))))
    denominator = tuple(sorted((tuple(sorted(d1)),tuple(sorted(d2)))))
    key = (numerator,denominator)
    if key in cache:
        return cache[key]
    norm_n = norms[numerator[0]]*norms[numerator[1]]
    norm_d = norms[denominator[0]]*norms[denominator[1]]
    if norm_n==0 or norm_d==0:
        raise ArithmeticError("exact proposal ratio is singular")
    difference = norm_n-norm_d
    # log1p retains an arbitrarily small signed distance from the unit
    # circle. Membership is the exact rational sign, even if float(log r)
    # subsequently underflows to signed zero.
    relative = difference/norm_d
    with mp.workdps(80):
        delta = mp.mpf(relative.numerator)/mp.mpf(relative.denominator)
        log_radius = float(mp.log1p(delta)/2)
    cache[key] = difference<0,log_radius
    return cache[key]


def atlas_candidates(differences, radial_power=None, *, radial_proposal=None, routing=None):
    """Full proposal density and target charts for an explicit CFT routing.

    Routing affects target eligibility/ranking only. Every supported proposal
    contributes to the density before the CFT-domain screen is applied.
    """
    from six_point_select_integration_channels import normalize_routing, star_gauge
    policy = normalize_routing(routing)
    proposal = normalize_radial_proposal(radial_proposal, radial_power=radial_power)
    source = getattr(differences,"native_source",None)
    exact_norms = None
    exact_ratio_cache = {}
    ambiguous_count = 0
    d = np.asarray(differences, dtype=complex)
    if d.shape != (6,6):
        raise ValueError("six-puncture determinant table required")
    with np.errstate(divide="ignore"):
        logs = np.log(np.abs(d)) + 1j * np.angle(d)
    orders, comb_signatures, star_signatures = _orderings()
    a, x, y, z, e, c = orders.T
    best = {"comb": {}, "star": {}}
    terms = []
    candidate_count = 0
    for topology in TOPOLOGIES:
        if topology == "comb":
            b = e
            qlogs = np.column_stack((
                logs[x,a]+logs[y,c]-logs[x,c]-logs[y,a],
                logs[y,a]+logs[z,c]-logs[y,c]-logs[z,a],
                logs[z,a]+logs[b,c]-logs[z,c]-logs[b,a],
            ))
        else:
            b = y
            qlogs = np.column_stack((
                logs[x,a]+logs[b,c]-logs[x,c]-logs[b,a],
                logs[a,c]+logs[z,b]-logs[b,a]-logs[z,c],
                logs[e,c]+logs[b,a]-logs[e,a]-logs[b,c],
            ))
        inside = np.isfinite(qlogs) & (qlogs.real<0)
        ambiguous = np.argwhere(np.isfinite(qlogs) & (np.abs(qlogs.real)<1e-12))
        if len(ambiguous):
            if source is None:
                raise ArithmeticError("ambiguous unit-circle support requires original native source provenance")
            if exact_norms is None:
                exact_norms = _exact_difference_norms(source)
            for original_index,axis in ambiguous:
                member,log_radius = _exact_ratio_log_radius(exact_norms,topology,orders[original_index],axis,exact_ratio_cache)
                inside[original_index,axis] = member
                qlogs[original_index,axis] = complex(log_radius,qlogs[original_index,axis].imag)
            ambiguous_count += len(ambiguous)
        valid = np.flatnonzero(np.all(inside, axis=1))
        if not len(valid):
            continue
        radii = np.exp(qlogs[valid].real)
        representable = np.all((radii > 0) & (radii < 1), axis=1)
        coords = radii * np.exp(1j*qlogs[valid].imag)
        logdet = logs[b,c].real+logs[b,a].real+logs[a,c].real
        # Derivatives in each original local coordinate, with 1/f at target ∞.
        target_orders = orders[valid]
        numerator = logs[target_orders,a[valid,None]].real+logs[b[valid],c[valid]].real[:,None]
        denominator = logs[target_orders,c[valid,None]].real+logs[b[valid],a[valid]].real[:,None]
        local = logdet[valid,None]-2*np.where(target_orders==c[valid,None],numerator,denominator)
        if not np.isfinite(local).all():
            raise ArithmeticError("nonfinite target local scale")
        native_jacobian = (qlogs[valid,1].real+2*qlogs[valid,2].real if topology=="comb"
                           else -2*qlogs[valid,2].real)
        # |dz_source/dq_target| = |J_target| / product_i |f'_local(i)|.
        log_inverse_jacobian = native_jacobian-local.sum(axis=1)
        density_terms = (mixture_log_radial_density(qlogs[valid].real, proposal)
                         -2*log_inverse_jacobian)
        terms.extend(density_terms.tolist())
        candidate_count += len(valid)
        scores = radii.max(axis=1)
        if topology == "comb":
            gauges = scores
        elif policy["name"] == "absolute-minimum-v1":
            # Preserve the original arithmetic (including exact tie choices)
            # for paired legacy replays.
            gauges = np.maximum(radii[:,0]+radii[:,1],
                .5*(radii[:,2]+np.sqrt(radii[:,2]**2+4*radii[:,1]*radii[:,2])))
        else:
            gauges = star_gauge(coords, domain="diagonal")
        signatures = comb_signatures if topology=="comb" else star_signatures
        for row, original_index in enumerate(valid):
            # Rounded boundary/underflow coordinates cannot certify a CFT
            # chart, but their exact proposal density term remains above.
            if not representable[row]:
                continue
            gauge = float(gauges[row])
            if gauge >= 1:
                continue
            signature = signatures[original_index]
            previous = best[topology].get(signature)
            if previous is None or gauge < previous[0]:
                best[topology][signature] = (gauge, float(scores[row]), tuple(map(int, orders[original_index])),
                                            tuple(coords[row]), tuple(local[row]))
    if not terms:
        raise ArithmeticError("no proposal orientation covers this sphere point")
    maximum = max(terms)
    density = maximum+math.log(math.fsum(math.exp(value-maximum) for value in terms))-math.log(1440)
    candidates = {name: sorted(values.values(),key=lambda row:row[0]) for name,values in best.items()}
    if isinstance(differences,NativeDifferenceTable):
        differences.ambiguous_ratio_count = ambiguous_count
    return density, candidates, candidate_count


def sample_native_geometry(point, *, radial_power=None, radial_proposal=None, routing=None):
    from six_point_select_integration_channels import normalize_routing, select_candidate
    policy = normalize_routing(routing)
    point = tuple(map(float,point))
    if len(point)!=7 or any(not math.isfinite(x) or not 0<=x<1 for x in point):
        raise ValueError("seven finite coordinates in [0,1) required")
    proposal = normalize_radial_proposal(radial_proposal, radial_power=radial_power)
    _, chart_uniform, qs = sample_source(point, proposal, _power_disk_sample)
    source_index = min(int(chart_uniform*1440),1439)
    source_topology = TOPOLOGIES[source_index//720]
    source_ordering = tuple(map(int,_orderings()[0][source_index%720]))
    d = native_differences(source_topology,qs,source_ordering)
    density,candidates,_ = atlas_candidates(d,radial_proposal=proposal,routing=policy)
    topology,row = select_candidate(candidates,policy,d)
    gauge,score,ordering,q,scales = row
    pair_logs = tuple(math.log(abs(d[i,j])) for i,j in PAIR_LABELS)
    return SphereGeometryPoint(point,topology,ordering,q,density,pair_logs,scales,score,gauge,0,
                               source_topology,source_ordering,qs)
