"""Marked torus frames and pullbacks to the original integration coordinates.

The three-point reduced string density has modular weight (4,4): R'=|c*tau+d|^8 R.
The real Jacobian of (tau,w1,w2)->(tau',w1',w2') is |c*tau+d|^-8.
Pull back the density OR transform the measure, never both.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

TAU = 2*math.pi


def upper_half_plane(tau):
    tau = complex(tau)
    if not all(map(math.isfinite, (tau.real, tau.imag))) or tau.imag <= 0:
        raise ValueError('finite upper-half-plane modulus required')
    return tau


@dataclass(frozen=True)
class ModularTransform:
    a: int = 1
    b: int = 0
    c: int = 0
    d: int = 1

    def __post_init__(self):
        if any(type(x) is not int for x in self.matrix) or self.a*self.d-self.b*self.c != 1:
            raise ValueError('an integer SL(2,Z) matrix is required')

    @property
    def matrix(self):
        return (self.a, self.b, self.c, self.d)

    def denominator(self, tau):
        return self.c*upper_half_plane(tau)+self.d

    def modulus(self, tau):
        tau = upper_half_plane(tau)
        den = self.denominator(tau)
        value = (self.a*tau+self.b)/den
        # Avoid cancellation in the imaginary part near a rational cusp.
        return complex(value.real, tau.imag/abs(den)**2)

    def inverse(self):
        return ModularTransform(self.d, -self.b, -self.c, self.a)

    def compose(self, right):
        """self after right, including the sign acting on punctures."""
        a,b,c,d = self.matrix; e,f,g,h = right.matrix
        return ModularTransform(a*e+b*g, a*f+b*h, c*e+d*g, c*f+d*h)

    def point(self, w, tau, *, wrap=True):
        z = complex(w)/self.denominator(tau)
        return wrap_point(z, self.modulus(tau)) if wrap else z

    def log_measure_jacobian(self, tau, integrated_punctures=2):
        if type(integrated_punctures) is not int or integrated_punctures < 0:
            raise ValueError('nonnegative integrated puncture count required')
        return -(4+2*integrated_punctures)*math.log(abs(self.denominator(tau)))


def fractional_coordinates(w, tau):
    tau = upper_half_plane(tau); w = complex(w)
    if not math.isfinite(w.real) or not math.isfinite(w.imag):
        raise ValueError('finite puncture coordinate required')
    v = w.imag/(TAU*tau.imag)
    return (w.real/TAU-tau.real*v, v)


def wrap_point(w, tau):
    u,v = fractional_coordinates(w, tau)
    return TAU*((u % 1)+(v % 1)*tau)


def fundamental_transform(tau):
    """Return a marked map into |Re tau|<=1/2, |tau|>=1."""
    tau = upper_half_plane(tau); transform = ModularTransform()
    seen=set()
    for _ in range(128):
        value = transform.modulus(tau)
        shift = math.floor(value.real+.5)
        if shift:
            transform = ModularTransform(1,-shift,0,1).compose(transform)
            value = transform.modulus(tau)
        if abs(value) >= 1-8e-15:
            return transform
        key=min(transform.matrix,tuple(-n for n in transform.matrix))
        if key in seen:
            return _fundamental_transform_precise(tau)
        seen.add(key)
        transform = ModularTransform(0,-1,1,0).compose(transform)
    return _fundamental_transform_precise(tau)


def _fundamental_transform_precise(tau):
    """Resolve seam decisions when cancellation makes binary64 cycle.

    Both nominal S-related images can round inside |tau|=1 when evaluated
    through a large integer matrix. Recompute the discrete reduction decision
    using the exact binary64 input at higher precision; do not move the point.
    """
    import mpmath as mp
    with mp.workdps(60):
        original=mp.mpc(tau.real,tau.imag);transform=ModularTransform()
        for _ in range(256):
            a,b,c,d=transform.matrix
            value=(a*original+b)/(c*original+d)
            shift=int(mp.floor(value.real+mp.mpf('.5')))
            if shift:
                transform=ModularTransform(1,-shift,0,1).compose(transform)
                value-=shift
            if abs(value)>=1-mp.mpf('1e-50'):
                return transform
            transform=ModularTransform(0,-1,1,0).compose(transform)
    raise ArithmeticError('precise modular reduction did not terminate')


def candidate_transforms(tau, bound=4):
    """Bounded cycle search about a reduced basis, valid at arbitrary input tau.

    Identity is always retained; both matrix signs act differently on points.
    This is a candidate search, not a proof of global optimality.
    """
    if type(bound) is not int or not 0 <= bound <= 16:
        raise ValueError('modular search bound must be an integer from 0 to 16')
    tau = upper_half_plane(tau)
    yield ModularTransform()
    if bound == 0:
        return
    reduction = fundamental_transform(tau); reduced = reduction.modulus(tau)
    seen = {ModularTransform().matrix}
    for c in range(-bound,bound+1):
        for d in range(-bound,bound+1):
            if math.gcd(c,d) != 1:
                continue
            a,b = next((a,b) for a in range(-bound,bound+1)
                       for b in range(-bound,bound+1) if a*d-b*c == 1)
            value = (a*reduced+b)/(c*reduced+d)
            shift = math.floor(value.real+.5)
            transform = ModularTransform(a-shift*c,b-shift*d,c,d).compose(reduction)
            if transform.matrix not in seen:
                seen.add(transform.matrix)
                yield transform


def covariance(atlas, tau, transform):
    """Validate the on-shell three-point convention and expose each factor.

    The code's Liouville primary exponential has the eta vacuum removed.
    Its pullback has an extra exp(pi*(tau2'-tau2)/6), which cancels only
    after the eta oscillator factor is included.
    """
    from genus1_three_point_collision import koba_exponent
    dsum = sum(complex(d) for d in atlas.external_weights)
    kappas = sum(koba_exponent(atlas,a,b) for a,b in ((0,1),(0,2),(1,2)))
    weight = 2+2*dsum-kappas
    if abs(weight-8) > 2e-12 or abs(dsum.imag)+abs(kappas.imag) > 2e-12:
        raise ValueError('this pullback requires the on-shell three-point string density')
    log_rho = math.log(abs(transform.denominator(tau)))
    delta = math.pi*(transform.modulus(tau).imag-complex(tau).imag)/6
    return dict(log_rho=log_rho, reduced_log_pullback=-8*log_rho,
        measure_log_jacobian=transform.log_measure_jacobian(tau),
        liouville_log_pullback=-2*dsum.real*log_rho,
        cancelled_liouville_log_pullback=-2*dsum.real*log_rho+delta,
        common_log_pushforward=(2-kappas)*log_rho+delta,
        reduced_modular_weight=8)


def common_log_in_frame(atlas, original, transform):
    """Common factor in the evaluation frame, computed in a stable reduced frame.

    This avoids a theta-series cancellation at small transformed Im(tau).
    The Liouville sewing variables are still evaluated in the selected frame.
    """
    from genus1_native_log_evaluation import common_log, koba_log
    tau = original['tau']; reduction = fundamental_transform(tau)
    reduced_tau = reduction.modulus(tau)
    points = [reduction.point(original[k],tau) for k in ('w1','w2')]
    base = common_log(reduced_tau)+koba_log(atlas,*points,reduced_tau)
    base -= covariance(atlas,tau,reduction)['common_log_pushforward']
    return base+covariance(atlas,tau,transform)['common_log_pushforward']
