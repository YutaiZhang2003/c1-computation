#!/usr/bin/env python3
"""Separating-primary control variate for the current c=1 genus-two kernel.

Conventions: alpha'=1, dP/pi, BRY's 8*pi**4 kernel including Aut=Z2,
epsilon=Omega12, q_bridge=-2*pi*i*epsilon to leading order.  This module
does NOT fit a normalization to the matrix model.

The control keeps the entire bridge-momentum integral, not just its
P=0 asymptotic.  At fixed diagonal tori, writing L=-log(2*pi*|epsilon|),

  |epsilon|**2 K0 = 4 R Theta1 Theta2 / sqrt(y1*y2)
      * integral dP dA dB C(A,P,A) C(B,P,B)
        exp(-4*pi*y1*A**2-4*pi*y2*B**2-2*L*P**2) |H1 H2|**2.

The radial integral is analytic: integral_L0^infty exp(-2*L*P**2)dL
=exp(-2*L0*P**2)/(2*P**2).  Its apparent threshold singularity cancels
against the two DOZZ zeros.  For the product torus domain, including the
half-plane epsilon angle and ordering of the two tori, the integral is

  G(R,rho0) = pi*R * integral_0^infty dP exp(-2*L0*P**2) J_R(P)**2,
  J_R(P) = integral_F d2tau Theta_R(tau)/sqrt(y)
           * integral_0^infty dA [C(A,P,A)/P] exp(-4*pi*y*A**2)|H|**2.

The finite-epsilon Gottschling domain need not equal this product domain:
subtract the out-of-domain control integral when assembling an estimator.
Finite handle-block and momentum orders are numerical approximations and
must be varied.  A control-variate correction alone does not certify the
old regular-area proposal's standard errors.  New separating samples use
the log-length proposal provided here, including its explicit Jacobian.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import mpmath as mp
import numpy as np
from numpy.polynomial import chebyshev
from scipy.fft import dct
from scipy.special import gamma, roots_genlaguerre
from scipy.stats import qmc

try:
    from genus2_integrand_normalization import CURRENT_INTEGRATION_KERNEL_CONVENTION, LEGACY_V5_XI_INTEGRATION_KERNEL_CONVENTION, LEGACY_V6_BRY_INTEGRATION_KERNEL_CONVENTION, integration_kernel_scale_to_current
    from torus_sampling import proposal_transform, theta_terms
    from virasoro_blocks import TorusOnePointVirasoroBlock
except ImportError:  # pragma: no cover
    from plumbing.genus2_integrand_normalization import CURRENT_INTEGRATION_KERNEL_CONVENTION, LEGACY_V5_XI_INTEGRATION_KERNEL_CONVENTION, LEGACY_V6_BRY_INTEGRATION_KERNEL_CONVENTION, integration_kernel_scale_to_current
    from torus_sampling import proposal_transform, theta_terms
    from plumbing.virasoro_blocks import TorusOnePointVirasoroBlock


def tail_scale_from_legacy_xi(kernel_convention):
    """Explicit compatibility for frozen Xi tail campaigns; no implicit mixing."""
    if kernel_convention == CURRENT_INTEGRATION_KERNEL_CONVENTION:
        return integration_kernel_scale_to_current(LEGACY_V5_XI_INTEGRATION_KERNEL_CONVENTION)
    if kernel_convention == LEGACY_V6_BRY_INTEGRATION_KERNEL_CONVENTION:
        return 0.25
    if kernel_convention == LEGACY_V5_XI_INTEGRATION_KERNEL_CONVENTION:
        return 1.0
    raise ValueError(f"unsupported tail convention {kernel_convention!r}")


class RealUpsilonOne:
    """Vectorized central-line log Upsilon, checked against Barnes G.

    Interpolation is on a compact interval ONLY; extrapolation is forbidden.
    Values outside it use the original arbitrary-precision formula.  The
    validation error is exposed, and construction fails above tolerance.
    """

    def __init__(self, maximum: float = 32., degree: int = 384,
                 dps: int = 44, tolerance: float = 2e-10):
        if not math.isfinite(maximum) or maximum <= 0 or degree < 16:
            raise ValueError("invalid Upsilon interpolation interval/order")
        self.maximum, self.dps = float(maximum), int(dps)

        @lru_cache(maxsize=4096)
        def exact(x):
            with mp.workdps(self.dps):
                return float(2*mp.re(mp.log(mp.barnesg(1+mp.j*float(x)))))

        self.exact = exact
        # A DCT avoids the accumulated cosine-matrix rounding error of a
        # direct O(N²) coefficient construction when log Upsilon is large.
        nodes = np.cos(math.pi*(np.arange(degree+1)+.5)/(degree+1))
        values = np.array([exact((v+1)*maximum/2) for v in nodes])
        self.coefficients = dct(values,type=2)/(degree+1)
        self.coefficients[0] *= .5
        probes = np.r_[np.linspace(0., maximum, 129),
                       np.random.default_rng(20260830).uniform(0., maximum, 64)]
        self.validation_max_abs_error = float(np.max(np.abs(
            self(probes)-np.array([exact(v) for v in probes]))))
        if self.validation_max_abs_error > tolerance:
            raise ArithmeticError("Upsilon interpolation failed validation: "
                                  f"{self.validation_max_abs_error:g}")

    def __call__(self, x):
        values = np.abs(np.asarray(x, dtype=float))
        if np.any(~np.isfinite(values)):
            raise ValueError("Upsilon arguments must be finite")
        shape = values.shape
        flat = values.reshape(-1)
        inside = flat <= self.maximum
        out = np.empty_like(flat)
        out[inside] = chebyshev.chebval(
            2*flat[inside]/self.maximum-1, self.coefficients)
        outside_values,inverse = np.unique(flat[~inside],return_inverse=True)
        out[~inside] = np.array([self.exact(float(v)) for v in outside_values])[inverse]
        # The exact endpoint is important for the threshold identity.
        out[flat == 0.] = 0.
        return out.reshape(shape)


def reduced_diagonal_dozz(a, p, upsilon: RealUpsilonOne):
    """C(A,P,A)/(8*A**2*P), continuously extended to either zero."""
    a, p = np.broadcast_arrays(np.asarray(a, float), np.asarray(p, float))
    if np.any(a < 0) or np.any(p < 0):
        raise ValueError("physical momenta must be nonnegative")
    log_value = (2*upsilon(2*a)+upsilon(2*p)-upsilon(2*a+p)
                 -upsilon(2*a-p)-2*upsilon(p))
    return np.where(p == 0., 1., np.exp(log_value))


def torus_elliptic_block(q, a, p, order: int = 2):
    """Torus H at c=25, h_int=1+A², h_ext=1+P².

    Levels 0..4 use explicit inverse-Gram-matrix formulas (checked against
    the independent c recursion). Higher orders call the reference recursion.
    No claim is made that a finite order is an exact torus block.
    """
    if order < 0:
        raise ValueError("block order must be nonnegative")
    q, a, p = np.broadcast_arrays(np.asarray(q, complex),
                                 np.asarray(a, float), np.asarray(p, float))
    h, e = 1+a*a, 1+p*p
    if order in (3, 4):
        try:
            from torus_primary_low_order import torus_primary_h
        except ImportError:  # pragma: no cover - package-style execution
            from torus_primary_low_order import torus_primary_h
        return torus_primary_h(q, h, e, order)
    if order > 2:
        out = np.empty(q.shape, dtype=complex)
        for index in np.ndindex(q.shape):
            out[index] = TorusOnePointVirasoroBlock(
                25., float(h[index]), float(e[index])).elliptic_block(
                    complex(q[index]), order)
        return out
    out = np.ones_like(q)
    h1 = e*(e-1)/(2*h)
    if order >= 1:
        out += h1*q
    if order >= 2:
        g11, g12, g22 = 4*h+12.5, 6*h, 4*h*(2*h+1)
        d11, d12 = 4*e*(e-1), 2*e*(e*e-1)
        d22 = (8*h+4)*e*(e-1)+(e-2)*(e-1)*e*(e+1)
        h2 = (g22*d11-2*g12*d12+g11*d22)/(g11*g22-g12*g12)-h1
        out += h2*q*q
    return out


def torus_lattice_sum(x, y, radii, *, tolerance: float = 1e-14):
    """Poisson lattice sum: Theta_R = sqrt(y)/R * this value.

    The cutoff scales with BOTH R and 1/R; a fixed cutoff is unsafe for
    the R=15 study.  A zero-winding term remains at arbitrarily large y.
    """
    radii = np.asarray(radii, float)
    if radii.ndim != 1 or not len(radii) or np.any(~np.isfinite(radii)) or np.any(radii <= 0):
        raise ValueError("radii must be positive and finite")
    if not 0 < tolerance < 1:
        raise ValueError("tolerance must be in (0,1)")
    cutoff = int(math.ceil(math.sqrt(-math.log(tolerance)/(math.pi*math.sqrt(3)/2))
                           * max(float(np.max(radii)), 1/float(np.min(radii)))))+2
    coefficients, mk = theta_terms(radii, alpha_prime=1., cutoff=cutoff,
                                   tolerance=tolerance)
    x, y = np.asarray(x, float), np.asarray(y, float)
    if (x.ndim != 1 or x.shape != y.shape or np.any(~np.isfinite(x))
            or np.any(~np.isfinite(y)) or np.any(y < math.sqrt(3)/2-1e-13)):
        raise ValueError('torus vectors require finite coordinates and y>=sqrt(3)/2')
    out = np.empty((len(x), len(radii)))
    for start in range(0, len(x), 128):
        stop = min(start+128, len(x))
        out[start:stop] = np.einsum(
            "nrt,nt->nr", np.exp(-y[start:stop,None,None]*coefficients[None]),
            np.cos(2*math.pi*x[start:stop,None]*mk[None]), optimize=True)
    return out


def handle_expectation(x, y, p: float, *, upsilon: RealUpsilonOne,
                       momentum_order: int = 16, block_order: int = 2):
    """E_Gamma(3/2)[C/(8*A²*P) |H|²] after the handle Gaussian."""
    nodes, weights = roots_genlaguerre(momentum_order, .5)
    x, y = np.asarray(x, float), np.asarray(y, float)
    a = np.sqrt(nodes[None,:]/(4*math.pi*y[:,None]))
    q = np.exp(2*math.pi*(1j*x-y))[:,None]
    value = reduced_diagonal_dozz(a, p, upsilon)
    value *= np.abs(torus_elliptic_block(q, a, p, block_order))**2
    return value @ (weights/gamma(1.5))


def primary_scaled_kernel(tau1: complex, tau2: complex, length: float,
                          radii, *, upsilon: RealUpsilonOne,
                          momentum_order: int = 16, bridge_order: int = 24,
                          block_order: int = 2,
                          kernel_convention=CURRENT_INTEGRATION_KERNEL_CONVENTION):
    """Return rho²*K0, without constructing rho or a singular period matrix."""
    return primary_scaled_kernel_batch(np.array([tau1]),np.array([tau2]),
        np.array([length]),radii,upsilon=upsilon,momentum_order=momentum_order,
        bridge_order=bridge_order,block_order=block_order,
        kernel_convention=kernel_convention)[0]


def primary_scaled_kernel_batch(tau1, tau2, length, radii, *,
                                upsilon: RealUpsilonOne, momentum_order: int = 16,
                                bridge_order: int = 24, block_order: int = 2,
                                kernel_convention=CURRENT_INTEGRATION_KERNEL_CONVENTION):
    """Vectorized rho²*K0, including arbitrarily deep representable log lengths."""
    tau1,tau2,length = np.broadcast_arrays(np.asarray(tau1,complex),
        np.asarray(tau2,complex),np.asarray(length,float))
    if length.ndim != 1 or np.any(~np.isfinite(length)) or np.any(length <= 0):
        raise ValueError("bridge log lengths must be a positive finite vector")
    radii = np.asarray(radii, float)
    if np.any(tau1.imag <= 0) or np.any(tau2.imag <= 0):
        raise ValueError("diagonal tori must be in the upper half plane")
    lattice1 = torus_lattice_sum(tau1.real,tau1.imag,radii)
    lattice2 = torus_lattice_sum(tau2.real,tau2.imag,radii)
    nodes, weights = roots_genlaguerre(bridge_order, .5)
    expectation = np.zeros(len(length))
    for node,weight in zip(nodes,weights/gamma(1.5)):
        p = np.sqrt(node/(2*length))[:,None]
        h1 = handle_expectation(tau1.real,tau1.imag,p,upsilon=upsilon,
                                momentum_order=momentum_order,block_order=block_order)
        h2 = handle_expectation(tau2.real,tau2.imag,p,upsilon=upsilon,
                                momentum_order=momentum_order,block_order=block_order)
        expectation += weight*h1*h2
    # The elementary Gaussian factors, using C=8*A²*P*D on each torus.
    return (tail_scale_from_legacy_xi(kernel_convention)*lattice1*lattice2/radii[None,:]
            * (expectation/((tau1.imag*tau2.imag*length)**1.5
               *4*math.sqrt(2)*math.pi**1.5))[:,None])


def leading_integrated_tail(radii, rho0: float, *,
                            kernel_convention=CURRENT_INTEGRATION_KERNEL_CONVENTION):
    """Leading large-L diagnostic ONLY, not the full primary control."""
    length = bridge_length(rho0)
    radii = np.asarray(radii, float)
    return (tail_scale_from_legacy_xi(kernel_convention)*math.pi**2/(36*math.sqrt(2*math.pi*length))
            * (radii+2/radii+1/radii**3))


def bridge_length(rho0: float) -> float:
    if not math.isfinite(rho0) or not 0 < rho0 < 1/(2*math.pi):
        raise ValueError("rho0 must lie in (0,1/(2*pi))")
    return -math.log(2*math.pi*rho0)


def separating_length_proposal(u, rho0: float):
    """Draw L=L0/(1-u)^2, density sqrt(L0)/(2*L**(3/2)).

    Returns (L, log_rho, log_inverse_density_in_L). Never exponentiate
    log_rho merely to pass it to a routine that cannot handle underflow.
    The physical area factor is rho² dL dphi, phi in [0,pi].
    """
    unit = np.asarray(u, float)
    if np.any(~np.isfinite(unit)) or np.any(unit < 0) or np.any(unit >= 1):
        raise ValueError("unit coordinates must lie in [0,1)")
    l0 = bridge_length(rho0)
    length = l0/(1-unit)**2
    return length, -length-math.log(2*math.pi), math.log(2/math.sqrt(l0))+1.5*np.log(length)


@dataclass(frozen=True)
class SeparatingProposal:
    tau1: np.ndarray
    tau2: np.ndarray
    length: np.ndarray
    log_rho: np.ndarray
    angle: np.ndarray
    log_weight_over_rho_squared: np.ndarray

    def take(self, indices):
        return SeparatingProposal(self.tau1[indices],self.tau2[indices],
            self.length[indices],self.log_rho[indices],self.angle[indices],
            self.log_weight_over_rho_squared[indices])

    def domain_matrices(self):
        """For DOMAIN membership only. Tiny off-diagonals may underflow.

        Do not pass these matrices to the full CFT evaluator; use the saved
        log coordinates for a separately validated deep-tail evaluator.
        """
        omega = np.zeros((len(self.length),2,2),complex)
        omega[:,0,0],omega[:,1,1] = self.tau1,self.tau2
        omega[:,0,1] = np.exp(self.log_rho+1j*self.angle)
        omega[:,1,0] = omega[:,0,1]
        return omega

    def weighted_primary(self, radii, *, upsilon, **kwargs):
        scaled = primary_scaled_kernel_batch(self.tau1,self.tau2,self.length,
                                             radii,upsilon=upsilon,**kwargs)
        # Combine in logs; no product of an underflowed rho² and divergent K.
        return np.exp(np.log(scaled)+self.log_weight_over_rho_squared[:,None])


def separating_proposals_from_unit_cube(unit, rho0: float):
    """Six-dimensional, normalized proposal on the ordered product-torus tube.

    The physical inverse density is exp(log_weight_over_rho_squared)*rho².
    Folded torus ordering gives a factor 1/2; the epsilon half-plane gives pi.
    Apply the Gottschling indicator separately.  Rejected proposals are
    zero terms, NOT a reason to renormalize by the accepted-node count.
    """
    unit = np.asarray(unit,float)
    if (unit.ndim != 2 or unit.shape[1] != 6 or not len(unit)
            or np.any(~np.isfinite(unit)) or np.any(unit < 0) or np.any(unit >= 1)):
        raise ValueError("unit points must have shape (N,6) and lie in [0,1)")
    x1,y1,a1 = proposal_transform(unit[:,:2])
    x2,y2,a2 = proposal_transform(unit[:,2:4])
    first,second = x1+1j*y1,x2+1j*y2
    swap=y1>y2
    length,log_rho,log_inv_length_density=separating_length_proposal(unit[:,5],rho0)
    log_inv_torus_density=(math.log(4)+1.5*np.log(y1*y2)-.5*np.log(a1*a2))
    return SeparatingProposal(np.where(swap,second,first),np.where(swap,first,second),
        length,log_rho,math.pi*unit[:,4],math.log(math.pi/2)
        +log_inv_torus_density+log_inv_length_density)


@dataclass(frozen=True)
class TailEstimate:
    radii: np.ndarray
    replicates: np.ndarray
    rho0: float
    power: int
    momentum_order: int
    bridge_order: int
    block_order: int
    integration_kernel_convention: str = CURRENT_INTEGRATION_KERNEL_CONVENTION

    @property
    def mean(self):
        return self.replicates.mean(axis=0)

    @property
    def standard_error(self):
        return self.replicates.std(axis=0, ddof=1)/math.sqrt(len(self.replicates))


def integrate_primary_tail(radii, rho0: float, *, upsilon: RealUpsilonOne,
                            power: int = 10, replicates: int = 8,
                            seed: int = 20260831, momentum_order: int = 16,
                            bridge_order: int = 24, block_order: int = 2,
                            progress=None,
                            kernel_convention=CURRENT_INTEGRATION_KERNEL_CONVENTION) -> TailEstimate:
    """Integrate on ordered product tori, NOT yet restricted at finite rho.

    Each Sobol point contains TWO independently parameterized tori.  We
    average their product; squaring a noisy estimate of J would add bias.
    There is no additional unmarked-torus or genus-two reflection factor.
    """
    if power < 1 or replicates < 2 or min(momentum_order,bridge_order) < 1:
        raise ValueError("need positive orders and at least two replicates")
    radii = np.asarray(radii, float)
    length = bridge_length(rho0)
    bridge_nodes, bridge_weights = roots_genlaguerre(bridge_order, -.5)
    bridge_weights = bridge_weights/(2*math.sqrt(2*length))
    result = []
    for replicate in range(replicates):
        unit = qmc.Sobol(4,scramble=True,seed=seed+replicate).random_base2(power)
        x1,y1,a1 = proposal_transform(unit[:,:2])
        x2,y2,a2 = proposal_transform(unit[:,2:])
        lattice1 = torus_lattice_sum(x1,y1,radii)
        lattice2 = torus_lattice_sum(x2,y2,radii)
        # Theta/y² divided by torus proposal and by 4*pi from handle integral.
        base_product = (lattice1*lattice2 /
                        (4*math.pi**2*radii[None,:]**2*np.sqrt(a1*a2)[:,None]))
        accumulated = np.zeros(len(radii))
        for p,w in zip(np.sqrt(bridge_nodes/(2*length)), bridge_weights):
            h1 = handle_expectation(x1,y1,float(p),upsilon=upsilon,
                                    momentum_order=momentum_order,block_order=block_order)
            h2 = handle_expectation(x2,y2,float(p),upsilon=upsilon,
                                    momentum_order=momentum_order,block_order=block_order)
            accumulated += w*np.mean(base_product*(h1*h2)[:,None], axis=0)
        result.append(tail_scale_from_legacy_xi(kernel_convention)*8*math.pi*radii*accumulated)
        if progress is not None:
            progress(replicate,np.array(result[-1]))
    return TailEstimate(radii,np.array(result),rho0,power,momentum_order,
                        bridge_order,block_order,kernel_convention)


def integrate_separating_residual(scaled_full_kernel, radii, rho0: float, *,
                                 upsilon: RealUpsilonOne, power: int = 8,
                                 replicates: int = 8, seed: int = 20262901,
                                 momentum_order: int = 24, bridge_order: int = 32,
                                 block_order: int = 2,
                                 kernel_convention=CURRENT_INTEGRATION_KERNEL_CONVENTION) -> TailEstimate:
    """Matched-proposal estimator of integral_product (1_D*K - K0).

    Add this to the analytic primary integral AND an outside-tube bulk
    estimate. Do NOT also include old inside-tube contributions. This
    three-part split, unlike reusing the old regular-area tail sample, has
    a tail proposal matched to the singularity even if the control block
    is numerically truncated.

    ``scaled_full_kernel(proposal, radii)`` must return rho²*K with shape
    (N,R), using the provided log coordinates. It must either validate its
    deep-tail approximation or fail explicitly. There is deliberately no
    silent rho floor, zero-period fallback, or untested asymptotic switch.
    Domain exclusions are included as negative K0 terms on the full nominal
    proposal count. The callback is called only on in-domain points.
    """
    try:
        from genus2_siegel_fundamental_domain import in_gottschling_domain
    except ImportError:  # pragma: no cover
        from plumbing.genus2_siegel_fundamental_domain import in_gottschling_domain
    if power < 1 or replicates < 2:
        raise ValueError('need positive power and at least two replicates')
    radii=np.asarray(radii,float)
    result=[]
    for k in range(replicates):
        unit=qmc.Sobol(6,scramble=True,seed=seed+k).random_base2(power)
        proposal=separating_proposals_from_unit_cube(unit,rho0)
        inside=in_gottschling_domain(proposal.domain_matrices(),tolerance=0.)
        primary=primary_scaled_kernel_batch(proposal.tau1,proposal.tau2,proposal.length,
            radii,upsilon=upsilon,momentum_order=momentum_order,
            bridge_order=bridge_order,block_order=block_order,kernel_convention=kernel_convention)
        full=np.zeros_like(primary)
        if np.any(inside):
            evaluated=np.asarray(scaled_full_kernel(proposal.take(inside),radii),float)
            if evaluated.shape != primary[inside].shape or np.any(~np.isfinite(evaluated)) or np.any(evaluated<0):
                raise ArithmeticError('full log-coordinate evaluator returned invalid values')
            full[inside]=evaluated
        difference=full-primary
        with np.errstate(divide='ignore'):
            weighted=np.sign(difference)*np.exp(np.log(np.abs(difference))
                +proposal.log_weight_over_rho_squared[:,None])
        if np.any(~np.isfinite(weighted)):
            raise ArithmeticError('nonfinite separating residual weight')
        result.append(weighted.mean(axis=0))
    return TailEstimate(radii,np.array(result),rho0,power,momentum_order,bridge_order,block_order,kernel_convention)


def weighted_residual(full,control,nodes):
    full,control=np.asarray(full,float),np.asarray(control,float)
    if full.ndim!=2 or full.shape!=control.shape or len(full)!=len(nodes):raise ValueError('incompatible kernel arrays')
    if np.any(~np.isfinite(full)) or np.any(~np.isfinite(control)):raise ValueError('nonfinite full/control value')
    difference=np.where(np.array([n['in_domain'] for n in nodes])[:,None],full-control,0.)
    inv=np.array([n['log_weight_over_rho_squared'] for n in nodes])
    with np.errstate(divide='ignore',invalid='raise',over='raise'):
        return np.sign(difference)*np.exp(np.log(np.abs(difference))+inv[:,None])


def band_replicates(weighted,nodes,policy):
    expected=(len(policy['band_edges'])-1)*policy['replicates']*2**policy['power']
    if len(nodes)!=expected or len(weighted)!=expected or len({n['node_id'] for n in nodes})!=expected:
        raise ValueError('incomplete/duplicate nominal design')
    result=[]
    for band in range(len(policy['band_edges'])-1):
        values=[]
        for rep in range(policy['replicates']):
            selected=[i for i,n in enumerate(nodes) if n['band']==band and n['replicate']==rep]
            if len(selected)!=2**policy['power'] or {nodes[i]['index'] for i in selected}!=set(range(2**policy['power'])):
                raise ValueError('incomplete band/replicate')
            values.append(np.sum(np.asarray(weighted)[selected],axis=0)/2**policy['power'])
        result.append(values)
    return np.asarray(result)
