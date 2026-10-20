"""Analytic primary subtraction on an explicit labelled six-point collar.

These functions use physical d²a d²x d²b measure. They never interpret a
selected-chart mask as a product collar or replace the atlas proposal density.
The analytic term retains the whole primary continuum; callers supply each
momentum and the complete middle four-point block. A finite descendant table
is a declared approximation, not a certified CFT integrand.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Mapping, Sequence

import numpy as np
from scipy.special import roots_jacobi
from numpy.polynomial.legendre import leggauss

DEFAULT_ORDERING = (1, 0, 5, 3, 4, 2)


def _ordering(ordering):
    result = tuple(ordering)
    if len(result) != 6 or set(result) != set(range(6)):
        raise ValueError("ordering must permute the six puncture labels")
    return result


def _energy(t):
    t = float(t)
    if not math.isfinite(t) or not 0 < t < 1/3:
        raise ValueError("the collar kernel requires 0 < t < 1/3")
    return t


@dataclass(frozen=True)
class CollarPatch:
    """One physical product collar, independent of CFT chart selection.

    middle_rectangle is (real_min, real_max, imag_min, imag_max). Its closed
    rectangle must avoid 0, 1 and the principal elliptic-K cut [1,infinity).
    Membership uses half-open bounds to make adjacent patches unambiguous.
    """
    left_radius: float
    right_radius: float
    middle_rectangle: tuple[float, float, float, float]
    ordering: tuple[int, ...] = DEFAULT_ORDERING

    def __post_init__(self):
        for name in ('left_radius', 'right_radius'):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0 < value < 1:
                raise ValueError("collar radii must be finite and lie in (0,1)")
            object.__setattr__(self, name, value)
        rectangle = tuple(map(float, self.middle_rectangle))
        if len(rectangle) != 4 or not all(map(math.isfinite, rectangle)):
            raise ValueError("four finite middle-rectangle bounds required")
        lo, hi, bottom, top = rectangle
        if not lo < hi or not bottom < top:
            raise ValueError("middle rectangle must have positive area")
        if bottom <= 0 <= top and (lo <= 0 <= hi or hi >= 1):
            raise ValueError("middle rectangle crosses a singularity or elliptic-K cut")
        # All finite regular pair separations must stay away from collisions.
        max_x = max(abs(complex(re, im)) for re in (lo, hi) for im in (bottom, top))
        if max_x * max(self.left_radius, self.right_radius) >= 1:
            raise ValueError("collar rectangle permits an additional pair collision")
        object.__setattr__(self, 'middle_rectangle', rectangle)
        object.__setattr__(self, 'ordering', _ordering(self.ordering))

    def contains(self, q: Sequence[complex]) -> bool:
        if len(q) != 3:
            raise ValueError("three native comb coordinates required")
        a, x, b = map(complex, q)
        if not all(math.isfinite(z.real) and math.isfinite(z.imag) for z in (a,x,b)):
            return False
        lo, hi, bottom, top = self.middle_rectangle
        return (abs(a) < self.left_radius and abs(b) < self.right_radius
                and lo <= x.real < hi and bottom <= x.imag < top)

    def middle_rule(self, order: int):
        order = _order(order)
        nodes, weights = leggauss(order)
        lo, hi, bottom, top = self.middle_rectangle
        re = (lo+hi)/2+(hi-lo)*nodes/2
        im = (bottom+top)/2+(top-bottom)*nodes/2
        return ((re[:,None]+1j*im[None,:]).ravel(),
                (np.outer(weights,weights)*(hi-lo)*(top-bottom)/4).ravel())


def _order(value):
    if int(value) != value or int(value) < 1:
        raise ValueError("positive integral quadrature order required")
    return int(value)


def channel_exponents(t, momenta, ordering=DEFAULT_ORDERING):
    """Physical powers beta in d²q |q|^(beta-2), including the Jacobian."""
    t = _energy(t)
    order = _ordering(ordering)
    ps = tuple(map(float, momenta))
    if len(ps) != 3 or any(not math.isfinite(p) or p < 0 for p in ps):
        raise ValueError("three finite nonnegative internal momenta required")
    signed = (5*t,)+(-t,)*5
    return tuple(2*p*p + .5*sum(signed[i] for i in order[:n])**2
                 for p,n in zip(ps,(2,3,4)))


def _log_abs_one_minus(z):
    z = np.asarray(z, dtype=complex)
    small = np.abs(z) < .5
    with np.errstate(divide='raise', invalid='raise'):
        if np.all(small):
            return .5*np.log1p(-2*z.real+z.real*z.real+z.imag*z.imag)
        # Near z=1 the squared-distance correction can round to -1 even
        # though the specified complex point is nonsingular. Compute the
        # actual distance there, without moving or regularizing the point.
        if not np.any(small):
            return np.log(np.abs(1-z))
        u = np.where(small,-2*z.real+z.real*z.real+z.imag*z.imag,0.)
        return .5*np.log1p(u)+np.log(np.where(small,1.,np.abs(1-z)))


def regular_matter_log_ratio(t, ordering, a, x, b):
    """log(R(a,x,b)/R(0,x,0)), continued regularly to a=0 or b=0."""
    t = _energy(t)
    order = _ordering(ordering)
    signed = np.asarray([5 if label == 0 else -1 for label in order], dtype=float)*t
    # Matter exponent -E_i E_j = s_i s_j on the imaginary ray E=i*s.
    terms = ((1,2,a),(1,3,a*x),(1,4,a*x*b),(2,4,x*b),(3,4,b))
    result = 0.
    for i,j,z in terms:
        result = result + signed[i]*signed[j]*_log_abs_one_minus(z)
    return result


def regular_matter(t, ordering, a, x, b):
    order = _ordering(ordering)
    t = _energy(t)
    s2,s3 = (5 if order[k] == 0 else -1 for k in (2,3))
    return np.exp(s2*s3*t*t*_log_abs_one_minus(x)
                  + regular_matter_log_ratio(t,order,a,x,b))


def _log_radius_ratio(inner, outer):
    return (math.log1p((inner-outer)/outer) if inner >= outer/2
            else math.log(inner)-math.log(outer))


def annulus_mass(beta, inner, outer):
    """Exact integral of d²q |q|^(beta-2) on inner<|q|<outer.

    Full discs require beta>0. Finite annuli also support beta=0 and negative
    beta, retaining the exact exponent dependence rather than a pole limit.
    """
    beta,inner,outer = map(float,(beta,inner,outer))
    if not all(map(math.isfinite,(beta,inner,outer))) or not 0 <= inner < outer:
        raise ValueError("finite beta and 0 <= inner < outer required")
    if inner == 0:
        if beta <= 0:
            raise ValueError("the full-disc primary integral diverges for beta<=0")
        return 2*math.pi*math.exp(beta*math.log(outer))/beta
    log_ratio = _log_radius_ratio(inner,outer)
    if beta == 0:
        return -2*math.pi*log_ratio
    return 2*math.pi*math.exp(beta*math.log(outer))*(-math.expm1(beta*log_ratio))/beta


@lru_cache(maxsize=512)
def _radial_rule(beta, inner, outer, order):
    """Probability rule for the exact radial power, with mass separated."""
    mass = annulus_mass(beta,inner,outer)
    if inner == 0 and beta >= 1e-8:
        nodes,weights = roots_jacobi(order,0.,beta-1)
        radii = outer*(nodes+1)/2
        weights = weights/weights.sum()
    else:
        nodes,weights = leggauss(order)
        u = (nodes+1)/2
        weights = weights/2
        if inner == 0:
            # Underflow is the regular a=0 limit, not a singular CFT call.
            radii = outer*np.exp(np.log(u)/beta)
        elif beta == 0:
            radii = np.exp(math.log(inner)-u*_log_radius_ratio(inner,outer))
        else:
            radii = outer*np.exp(np.log1p((1-u)*math.expm1(beta*_log_radius_ratio(inner,outer)))/beta)
    if np.any(~np.isfinite(radii)) or np.any(~np.isfinite(weights)):
        raise FloatingPointError("nonfinite collar quadrature")
    radii.flags.writeable = False
    weights.flags.writeable = False
    return radii,weights,mass


def integrate_outer_collars(slices: Mapping[tuple[int,int],complex], t, momenta, x,
                            left_radius, right_radius, *, ordering=DEFAULT_ORDERING,
                            radial_order=8, angular_order=8, inner_radii=(0.,0.)):
    """Integrate a finite outer block and exact regular matter over two collars.

    Returns the analytic leading term plus a stably evaluated numerical
    remainder. The result excludes |x|^(beta_middle-2), DOZZ and momentum
    measures. All slices must share the same complete middle normalization.
    No missing outer coefficient is inferred to vanish in the full CFT.
    """
    radial_order,angular_order = _order(radial_order),_order(angular_order)
    x = complex(x)
    if not math.isfinite(x.real) or not math.isfinite(x.imag) or x in (0,1):
        raise ValueError("finite nonsingular middle coordinate required")
    if not 0 < float(left_radius) < 1 or not 0 < float(right_radius) < 1:
        raise ValueError("collar radii must lie in (0,1)")
    if abs(x)*max(left_radius,right_radius) >= 1:
        raise ValueError("requested collar may contain an additional collision")
    if (0,0) not in slices:
        raise ValueError("the complete primary middle slice is required")
    coeffs = {}
    for key,value in slices.items():
        if (len(key)!=2 or any(int(k)!=k or k<0 for k in key)
                or not math.isfinite(complex(value).real) or not math.isfinite(complex(value).imag)):
            raise ValueError("nonnegative outer degrees and finite coefficients required")
        coeffs[tuple(map(int,key))] = complex(value)
    if len(inner_radii)!=2:
        raise ValueError("two inner collar radii required")
    beta_a,beta_x,beta_b = channel_exponents(t,momenta,ordering)
    ra,wa,ma = _radial_rule(beta_a,float(inner_radii[0]),float(left_radius),radial_order)
    rb,wb,mb = _radial_rule(beta_b,float(inner_radii[1]),float(right_radius),radial_order)
    phases = np.exp(2j*np.pi*np.arange(angular_order)/angular_order)
    av = (ra[:,None]*phases[None,:]).ravel()[:,None]
    bv = (rb[:,None]*phases[None,:]).ravel()[None,:]
    weights = np.outer(np.repeat(wa/angular_order,angular_order),
                       np.repeat(wb/angular_order,angular_order))
    f0 = coeffs[(0,0)]
    delta = np.zeros((av.size,bv.size),dtype=complex)
    for (i,j),coefficient in coeffs.items():
        if i+j:
            delta += coefficient*av**i*bv**j
    log_ratio = regular_matter_log_ratio(t,ordering,av,x,bv)
    ratio = np.exp(log_ratio)
    regular0 = float(regular_matter(t,ordering,0.,x,0.))
    primary_norm = abs(f0)**2
    remainder_density = (np.expm1(log_ratio)*primary_norm
                         + ratio*(2*np.real(f0.conjugate()*delta)+np.abs(delta)**2))
    mass = ma*mb*regular0
    leading = mass*primary_norm
    remainder = mass*float(np.sum(weights*remainder_density))
    direct = mass*float(np.sum(weights*ratio*np.abs(f0+delta)**2))
    total = leading+remainder
    result = dict(leading=leading,remainder=remainder,total=total,direct=direct,
                  relative_remainder=(abs(remainder/total) if total else None),
                  absolute_remainder_integral=mass*float(np.sum(weights*np.abs(remainder_density))),
                  beta=(beta_a,beta_x,beta_b),radial_order=radial_order,
                  angular_order=angular_order,outer_coefficients=len(coeffs),
                  quadrature_points=int(weights.size),full_CFT_certified=False)
    if any(not math.isfinite(result[key]) for key in ('leading','remainder','total','direct')):
        raise FloatingPointError("nonfinite collar integral")
    return result


def leading_patch_density(patch: CollarPatch, q, t, momenta, primary_slice):
    """Primary counterterm density in physical d²a d²x d²b measure.

    This pointwise function is for nonzero finite moduli samples; use analytic
    moments for the endpoints. No proposal weight or chart-selection mask is
    applied here. Multiplication by the appropriate inverse proposal density
    belongs in the sampler adapter.
    """
    if not patch.contains(q):
        return 0.
    a,x,b = map(complex,q)
    if a == 0 or b == 0:
        raise ValueError("use analytic collar moments at exact outer endpoints")
    betas = channel_exponents(t,momenta,patch.ordering)
    power = sum((beta-2)*math.log(abs(z)) for beta,z in zip(betas,(a,x,b)))
    return math.exp(power)*float(regular_matter(t,patch.ordering,0,x,0))*abs(complex(primary_slice))**2


__all__ = ['CollarPatch','channel_exponents','regular_matter',
           'regular_matter_log_ratio','annulus_mass','integrate_outer_collars',
           'leading_patch_density']
