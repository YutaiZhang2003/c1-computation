"""Opt-in analytic boundary partition and direct compact-bulk sampling.

Collision strata keep priority at every height. The regular complement is
split by its longest necklace gap. This module defines exact geometry and
primary kernels; a geometric route never certifies an approximation error.
The historical campaign and bank resolvers do not import this module.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
from scipy.stats import qmc

from genus1_moduli_sampling import TAU, ShortPoint, from_torus, regular_mask
from genus1_nonseparating import log_length_kernel, _sum_logs
from genus1_cusp_control import _matter

MIN_HEIGHT = math.sqrt(3)/2
FORMAT = 'genus1-analytic-boundary-layer-v1'


@dataclass(frozen=True)
class BoundaryPolicy:
    q_max: float = .05
    pair_radius: float = .125
    triple_radius: float = .5

    def __post_init__(self):
        if not math.isfinite(self.q_max) or not 0 < self.q_max < math.exp(-TAU/3):
            raise ValueError('this compact map requires a height cap strictly above one')
        if not 0 < self.pair_radius < min(math.pi, TAU*MIN_HEIGHT/3):
            raise ValueError('pair radius must have a uniform longest-gap cylinder mask')
        from genus1_three_point_collision import CollisionRadii
        CollisionRadii(self.pair_radius, self.triple_radius)

    @property
    def length(self):
        return -math.log(self.q_max)

    @property
    def height_cap(self):
        return 3*self.length/TAU

    def record(self):
        return dict(format=FORMAT, q_max=self.q_max, pair_radius=self.pair_radius,
                    triple_radius=self.triple_radius, height_cap=self.height_cap,
                    geometry_only=True, approximation_certified=False)


def classify(w1, w2, tau, policy=BoundaryPolicy()):
    tau = complex(tau)
    if not -.5 <= tau.real < .5 or tau.imag < math.sqrt(1-tau.real**2)-1e-13:
        raise ValueError('standard modular fundamental domain required')
    point = from_torus(w1, w2, tau)
    if not regular_mask(point.short, policy.pair_radius, MIN_HEIGHT):
        return 'collision'
    return 'tube' if point.length >= policy.length else 'compact'


def gap_area(total, limit):
    """Area in dg1 dg2 with all three positive gaps below limit."""
    total = np.asarray(total, float)
    # Reflection about 3*limit/2 avoids cancellation near the upper endpoint.
    t = np.minimum(total, 3*limit-total)
    return np.where(t > 0, .5*(t*t-3*np.maximum(t-limit, 0)**2), 0.)


def gap_primitive(x, limit):
    """Integral of gap_area from zero to x, on 0 <= x <= 3*limit."""
    x = np.asarray(x, float)
    return sum(c*np.maximum(x-k*limit, 0)**3 for k,c in enumerate((1,-3,3,-1)))/6


def _gap_polygon(total, limit):
    poly = [np.array(v, float) for v in ((0,0),(limit,0),(limit,limit),(0,limit))]
    for bound, sign in ((total, 1.), (total-limit, -1.)):
        result = []
        for a,b in zip(poly, poly[1:]+poly[:1]):
            fa, fb = sign*(sum(a)-bound), sign*(sum(b)-bound)
            if fa <= 0:
                result.append(a)
            if (fa < 0 < fb) or (fb < 0 < fa):
                result.append(a+(b-a)*fa/(fa-fb))
        poly = result
    if len(poly) < 3:
        raise ValueError('zero-volume compact gap polygon')
    return np.asarray(poly)


def _gap_point(total, limit, u, v):
    poly = _gap_polygon(total, limit)
    center = np.mean(poly, axis=0)
    edges = np.roll(poly, -1, axis=0)
    a, b = poly-center, edges-center
    areas = abs(a[:,0]*b[:,1]-a[:,1]*b[:,0])/2
    cumulative = np.cumsum(areas)
    index = min(int(np.searchsorted(cumulative, u*cumulative[-1], side='right')), len(poly)-1)
    start = cumulative[index-1] if index else 0.
    r = math.sqrt(max(0., (u*cumulative[-1]-start)/areas[index]))
    pair = center+r*((1-v)*a[index]+v*b[index])
    return np.array([pair[0], pair[1], total-sum(pair)])


@dataclass(frozen=True)
class CompactBatch:
    tau: np.ndarray
    w1: np.ndarray
    w2: np.ndarray
    weight: np.ndarray
    regular: np.ndarray
    gaps: np.ndarray


def map_compact(unit, policy=BoundaryPolicy()):
    """Map six unit coordinates directly into the finite tube complement.

    tau1 is uniform. Height density is proportional to the admissible gap
    polygon area; gaps are uniform in that polygon. Both labeled cyclic
    orders are present. The weight is for d²tau d²w1 d²w2, before -Im/(4*pi).
    Collision points remain explicit zeros, never an unnormalized rejection.
    """
    u = np.asarray(unit, float)
    if u.ndim != 2 or u.shape[1] != 6 or np.any(~np.isfinite(u)) or np.any((u<0)|(u>=1)):
        raise ValueError('an N by 6 array in [0,1) is required')
    u = np.maximum(u, np.finfo(float).eps/2)
    limit = policy.length
    tau1 = u[:,0]-.5
    lower = TAU*np.sqrt(1-tau1*tau1)
    deficit_max = 3*limit-lower
    normalizer = gap_primitive(deficit_max, limit)
    target = (1-u[:,1])*normalizer
    left = np.zeros(len(u)); right = deficit_max.copy()
    for _ in range(52):
        mid = (left+right)/2
        smaller = gap_primitive(mid, limit) < target
        left = np.where(smaller, mid, left)
        right = np.where(smaller, right, mid)
    total = 3*limit-(left+right)/2
    code = 2*u[:,2]
    reverse = code >= 1
    gaps = np.array([_gap_point(t,limit,a,b) for t,a,b in zip(total,code%1,u[:,3])])
    y1 = gaps[:,0]; y2 = gaps[:,0]+gaps[:,1]
    w1 = TAU*u[:,4]+1j*np.where(reverse,y2,y1)
    w2 = TAU*u[:,5]+1j*np.where(reverse,y1,y2)
    tau = tau1+1j*total/TAU
    distance2 = np.full(len(u), np.inf)
    for z in (w1, w2, w1-w2):
        for n in (-1,0,1):
            d = z+n*TAU*tau
            real = d.real-TAU*np.rint(d.real/TAU)
            distance2 = np.minimum(distance2, real**2+d.imag**2)
    return CompactBatch(tau,w1,w2,2*TAU*normalizer,
                        distance2 >= policy.pair_radius**2,gaps)


def compact_samples(power, *, seed, replicate, policy=BoundaryPolicy()):
    if any(type(x) is not int or x < 0 for x in (power,seed,replicate)):
        raise ValueError('nonnegative integer power, seed and replicate required')
    state = int(np.random.SeedSequence([seed,replicate,0x434F4D50]).generate_state(1)[0])
    return map_compact(qmc.Sobol(6,scramble=True,seed=state).random_base2(power),policy)


def compact_plan(atlas, unit, *, policy=BoundaryPolicy(), channel=None):
    """Bank-backed worker request with the direct compact map and its weight."""
    from genus1_collision_sampling import Request, SamplePlan
    b = map_compact(np.asarray(unit,float)[None,:],policy)
    if not b.regular[0]:
        return SamplePlan(0., (), 'outside_compact_regular_inside_collision')
    w1,w2,tau = b.w1[0],b.w2[0],b.tau[0]
    if channel is None:
        choice = atlas.choose_channel(w1,w2,tau)
        channel,pair = choice.channel,choice.pair
    elif channel == 'necklace':
        pair = None
    else:
        raise ValueError('explicit channel override supports necklace only')
    atlas.sewing_chart(w1,w2,tau,channel,pair).require()
    return SamplePlan(float(b.weight[0]),
        (Request('full_reduced',dict(w1=w1,w2=w2,tau=tau,channel=channel,pair=pair)),))


COLLISION_STRATA = ('pair_face_remainder','pair_face_analytic',
    'triple_forest_remainder','triple_pair_face','triple_scale_face','triple_corner')


def collision_plan(atlas, stratum, unit, *, region, policy=BoundaryPolicy(), cutoff=8.):
    """Keep the full existing collision forest at the chosen radii and all heights."""
    from genus1_three_point_collision import CollisionRadii
    from genus1_collision_sampling import sample_plan
    if stratum not in COLLISION_STRATA:
        raise ValueError('one of the six collision terms is required')
    return sample_plan(atlas,stratum,unit,region=region,
        radii=CollisionRadii(policy.pair_radius,policy.triple_radius),cutoff=cutoff)


def necklace_slice_geometry(atlas, short):
    """Native short logs and physical cylinder factors for any longest-gap sector."""
    if atlas.external_momenta[0] != atlas.external_momenta[1]:
        raise ValueError('equal outgoing operators required')
    # After the cut, the two successive displacements are exactly
    # x1+i*s1 and (x2-x1)+i*s2 in every labeled sector. Subtracting
    # translated positions introduced sector-dependent rounding and even
    # lost a very small s2 beside a long s1. Keep the defining lengths.
    axis=(2-short.sector.index(2))%3
    logs=(complex(-short.s1,short.x1),complex(-short.s2,short.x2-short.x1))
    sine,gaussian = _matter(atlas,short)
    return dict(cut_axis=axis,short_logs=logs,
                sine=sine,gaussian=gaussian)


@dataclass(frozen=True)
class PrimarySlice:
    cut_momenta: np.ndarray
    log_coefficients: np.ndarray
    short_length: float
    sine: float
    gaussian: float

    def __post_init__(self):
        p=np.asarray(self.cut_momenta,float).copy(); z=np.asarray(self.log_coefficients,complex).copy()
        if (p.ndim!=1 or not len(p) or z.shape!=p.shape or np.any(~np.isfinite(p)) or np.any(p<=0)
                or np.any(~np.isfinite(z.imag)) or np.any(~np.isfinite(z.real)&(z.real!=-math.inf))
                or not all(math.isfinite(x) for x in (self.short_length,self.sine,self.gaussian))
                or self.short_length<=0 or self.gaussian<0):
            raise ValueError('invalid real-weight primary slice')
        p.flags.writeable=False; z.flags.writeable=False
        object.__setattr__(self,'cut_momenta',p); object.__setattr__(self,'log_coefficients',z)

    def kernel_logs(self, lower):
        return log_length_kernel(self.cut_momenta,lower_length=lower,short_length=self.short_length,
                                 coupling=math.sqrt(2*self.gaussian),log_prefactor=self.sine)

    def point(self,length):
        if not math.isfinite(length) or length<=0:
            raise ValueError('positive tube length required')
        matter=.5*math.log(TAU/(length+self.short_length))+self.sine+self.gaussian/(length+self.short_length)
        value,absolute=_sum_logs(self.log_coefficients-2*self.cut_momenta**2*length+matter)
        return dict(value=value,absolute_sum=absolute)

    def integrated(self, lower):
        value,absolute=_sum_logs(self.log_coefficients+self.kernel_logs(lower)-math.log(TAU))
        return dict(value=value,absolute_sum=absolute)


def primary_from_blocks(atlas, short, *, momenta, log_weights, structure_logs, blocks):
    """Use normalized B5 from a checked elliptic/native bank on the given grid.

    dP weights, full complex DOZZ / pi^3, short propagators and |B5|² appear
    exactly once. This function does not construct or validate a spectral bank.
    """
    g=necklace_slice_geometry(atlas,short)
    p=np.asarray(momenta,float); lw=np.asarray(log_weights,float)
    ds=np.asarray(structure_logs,complex); b=np.asarray(blocks,complex)
    if (p.ndim!=2 or p.shape[1]!=3 or any(a.shape!=(len(p),) for a in (lw,ds,b))
            or any(np.any(~np.isfinite(a)) for a in (p,lw,ds,b)) or np.any(p<=0)):
        raise ValueError('compatible finite physical spectral arrays required')
    axis=g['cut_axis']; exponent=sum(2*p[:,(axis+k)%3]**2*z.real
        for k,z in enumerate(g['short_logs'],1))
    block_log=np.full(len(b),-math.inf); good=abs(b)>0
    block_log[good]=2*np.log(abs(b[good]))
    return PrimarySlice(p[:,axis],lw+ds-3*math.log(math.pi)+exponent+block_log,
                        short.total,g['sine'],g['gaussian'])


def integrate_primary(short, nodes, *, policy=BoundaryPolicy(), angle_order=32):
    """Analytic length and strip twist; deterministic twist in the modular cap.

    The remaining density has four short moduli and sums over six sectors.
    Collisions are owned by collision_plan, never by a second tube addback.
    """
    if type(angle_order) is not int or angle_order<2:
        raise ValueError('angle order must be an integer >=2')
    if not math.isclose(nodes.short_length,short.total,rel_tol=1e-13,abs_tol=1e-13):
        raise ValueError('primary nodes and short geometry do not match')
    if not regular_mask(short,policy.pair_radius,MIN_HEIGHT):
        return dict(value=0j,absolute_sum=0.,angle_evaluations=0,owner='collision')
    floor=max(short.s1,short.s2,policy.length)
    if floor+short.total>=TAU:
        return dict(**nodes.integrated(floor),angle_evaluations=0,owner='tube')
    height=(floor+short.total)/TAU
    boundaries=[0.,math.pi]
    if MIN_HEIGHT<height<1:
        boundaries.insert(1,TAU*math.sqrt(1-height*height))
    from genus1_boundary_cubature import gauss_rule
    roots,weights=gauss_rule(angle_order)
    values=[]; absolutes=[]
    for a,b in zip(boundaries,boundaries[1:]):
        for u,w in zip(roots,weights):
            theta=a+(u+1)*(b-a)/2
            lower=max(floor,TAU*math.sqrt(1-(theta/TAU)**2)-short.total)
            result=nodes.integrated(lower); weight=w*(b-a)/TAU
            values.append(weight*result['value']); absolutes.append(weight*result['absolute_sum'])
    return dict(value=complex(math.fsum(x.real for x in values),math.fsum(x.imag for x in values)),
                absolute_sum=math.fsum(absolutes),angle_evaluations=len(values),owner='tube')
