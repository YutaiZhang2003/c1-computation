"""Thermal kernels with explicit KK charges; no change to saved source kernels."""
from __future__ import annotations

# kinematics
# Explicit equal-outgoing Matsubara kinematics in the draft's conventions.

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ThermalKinematics:
    t: float
    R: float
    N: int = 1

    def __post_init__(self):
        if (type(self.N) is not int or self.N < 1 or
                not all(math.isfinite(v) and v > 0 for v in (self.t, self.R))):
            raise ValueError('positive frequency, radius and integer KK number required')
        if not math.isclose(self.t * self.R, 2 * self.N, rel_tol=2e-13):
            raise ValueError('Matsubara kinematics require t R = 2 N')

    @property
    def charges(self):
        return (self.N, self.N, -2 * self.N)

    def charge_position(self, positions):
        if len(positions) != 3:
            raise ValueError('three insertion positions required')
        return sum(n * z for n, z in zip(self.charges, positions))

    def record(self):
        return dict(t=self.t, R=self.R, N=self.N, charges=list(self.charges),
                    observable='finite thermal-LSZ Matsubara coefficient',
                    normalization='A_1to2 = g_s A_tree + g_s^3 A_genus1 + ...')


def matrix_prediction(kin):
    """Comparison only. Never used by quadrature or proposal construction."""
    t = kin.t
    coefficient = math.pi**3 / 12 * t**3 * (t - 1) * (t - 2)
    zero = coefficient * (1 + t - t*t/2)
    delta = coefficient / kin.R**2
    return dict(noncompact=zero, thermal=zero + delta, correction=delta,
                scaled_correction=coefficient)



# compact_boson
# Compact Euclidean boson factor relative to the noncompact torus integrand.
# 
# alpha'=1; z=2*pi*(x+tau*y); integer external momenta n_i/R and no
# external winding. The compact target zero mode is stripped separately.



from genus1_source_paths import ROOT as _SOURCE_ROOT, source_path as _source_path

import math
import numpy as np


from genus1_compact_boson import (
    phases, lattice_direct, lattice_poisson, compact_factor,
    validate as validate_lattice,
)


if __name__=="__main__":
    import json
    print(json.dumps(validate_lattice(),indent=2))


# thermal_boundary_kernels
# Compact-boson kernels for the retained genus-one boundary estimator.
# 
# These integrate the lattice sum BEFORE the tube/height integral. They do not
# restore descendant remainders omitted in the source boundary estimator.
# Collision projections use the lattice at the fused insertion configuration.


from genus1_source_paths import ROOT as _SOURCE_ROOT, source_path as _source_path
import math
from functools import lru_cache
import numpy as np
from scipy.special import logsumexp

TAU = 2 * math.pi


def primary_kernel_log(p, short, lower, R, *, theta=None, tolerance=1e-16):
    """Length integral, including 1/(2*pi) from the original moduli measure.

    theta=None performs the full twist average. Otherwise this is the average
    of the +theta and -theta densities needed in the circular modular cap.
    p contains only the Liouville momenta on the cut tube.
    """
    p = np.asarray(p, float)
    positions = short.positions()
    charge_position = positions[0] + positions[1] - 2 * positions[2]
    qx, qy = charge_position.real, charge_position.imag
    S, total = short.total, short.total + lower
    width = R * math.sqrt(2 * (-math.log(tolerance) + 4) / total)
    center = qy / total
    ell = np.arange(math.floor(center - width) - 1,
                    math.ceil(center + width) + 2, dtype=float)
    mcap = math.ceil(math.sqrt(2 * (-math.log(tolerance) + 4) /
                              (R * R * total))) + 1
    mm, ll = np.meshgrid(np.arange(-mcap, mcap + 1), ell, indexing="ij")
    mm, ll = mm.ravel(), ll.ravel()
    if theta is None:
        keep = mm * ll == 0
        mm, ll = mm[keep], ll[keep]
        phase = np.cos(mm * qx)
    else:
        phase = np.cos(mm * qx) * np.cos(mm * ll * theta)
    mass = (R * R * mm * mm + ll * ll / (R * R)) / 2
    decay = 2 * p[:, None] ** 2 + mass[None, :]
    logs = (-2 * p[:, None] ** 2 * lower - mass[None, :] * total +
            ll[None, :] * qy / (R * R) - np.log(decay))
    value, sign = logsumexp(logs, b=phase[None, :], axis=1, return_sign=True)
    if np.any(sign <= 0):
        raise FloatingPointError("nonpositive integrated compact-boson kernel")
    return value - math.log(TAU * R)


def primary_integral(short, p, log_coefficients, sine, R, policy, angle_order=32):
    from genus1_nonseparating import _sum_logs
    positions = short.positions()
    Q = positions[0]+positions[1]-2*positions[2]
    offset, components, _ = integrated_primary_components(
        tuple(map(float,p)),short.s1,short.s2,abs(Q.imag),R,policy.length,angle_order)
    kernel = components @ np.cos(np.arange(components.shape[1])*Q.real)
    if np.any(kernel <= 0):
        raise FloatingPointError("nonpositive integrated primary kernel")
    return _sum_logs(log_coefficients+sine+offset+np.log(kernel))[0]


@lru_cache(maxsize=4096)
def integrated_primary_components(p_tuple,s1,s2,qy,R,threshold,angle_order):
    """Integrate each spatial-winding harmonic once per physical short grid."""
    from genus1_boundary_cubature import gauss_rule
    from genus1_nonseparating import log_length_kernel
    p = np.asarray(p_tuple)
    S = s1+s2
    floor = max(s1,s2,threshold)
    total_min = floor+S
    if total_min >= TAU:
        rules = [(floor,None,1.)]
    else:
        height = total_min/TAU
        edges = [0.,math.pi]
        if math.sqrt(3)/2 < height < 1:
            edges.insert(1,TAU*math.sqrt(1-height*height))
        roots,weights = gauss_rule(angle_order)
        rules = []
        for a,b in zip(edges,edges[1:]):
            for node,weight in zip(roots,weights):
                theta = a+(node+1)*(b-a)/2
                lower = max(floor,TAU*math.sqrt(1-(theta/TAU)**2)-S)
                rules.append((lower,theta,weight*(b-a)/TAU))
    width = R*math.sqrt(2*41/total_min)
    center = qy/total_min
    ell = np.arange(math.floor(center-width)-2,math.ceil(center+width)+3,dtype=float)
    mcap = math.ceil(math.sqrt(2*41/(R*R*total_min)))+1
    modes = np.arange(mcap+1)
    mass = (R*R*modes[:,None]**2+ell[None,:]**2/(R*R))/2
    rate = 2*p[:,None,None]**2+mass[None,:,:]
    offset = -2*p*p*floor+qy*qy/(2*R*R*total_min)-np.log(2*p*p)-math.log(TAU*R)
    components = np.zeros((len(p),len(modes)))
    old_logs = []
    for lower,theta,weight in rules:
        logs = (-2*p[:,None,None]**2*lower-mass[None,:,:]*(lower+S)+
                ell[None,None,:]*qy/(R*R)-np.log(rate)-math.log(TAU*R))
        phase = ((modes[:,None]*ell[None,:] == 0).astype(float) if theta is None else
                 np.cos(modes[:,None]*ell[None,:]*theta))
        components += weight*np.sum(np.exp(logs-offset[:,None,None])*phase[None,:,:],axis=2)
        old_logs.append(math.log(weight)+log_length_kernel(p,lower_length=lower,
            short_length=S,coupling=qy/R,log_prefactor=0.)-math.log(TAU))
    components[:,1:] *= 2
    old_kernel = logsumexp(old_logs,axis=0)
    return offset,components,old_kernel


def height_log_integrals(decay, lower, upper, R, tau1, tolerance=1e-16):
    """Integral exp(-a*T)*Lambda_R(T,0)/sqrt(T) over a height interval."""
    decay = np.asarray(decay, float)
    ellcap = math.ceil(R * math.sqrt((-math.log(tolerance) + 4) /
                                   (math.pi * lower))) + 1
    mcap = math.ceil(math.sqrt((-math.log(tolerance) + 4) /
                              (math.pi * R * R * lower))) + 1
    mm, ll = np.meshgrid(np.arange(-mcap, mcap + 1),
                         np.arange(-ellcap, ellcap + 1), indexing="ij")
    mm, ll = mm.ravel(), ll.ravel()
    phase = np.cos(TAU * tau1 * mm * ll)
    mass = math.pi * (R * R * mm * mm + ll * ll / (R * R))
    rate = decay[..., None] + mass
    logs = -rate * lower - np.log(rate) - math.log(R)
    if not math.isinf(upper):
        logs += np.log(-np.expm1(-rate * (upper - lower)))
    value, sign = logsumexp(logs, b=phase, axis=-1, return_sign=True)
    if np.any(sign <= 0):
        raise FloatingPointError("nonpositive thermal height kernel")
    return value


def height_matrices(loop_nodes, tau1, region, cutoff=8., *, loop_order=2, R):
    from genus1_integrated_scale_collar import ETA_TERMS
    lower = math.sqrt(1 - tau1 * tau1) if region == "bulk" else cutoff
    upper = cutoff if region == "bulk" else math.inf
    p = np.asarray(loop_nodes, float)
    maximum = 2 * (loop_order + 7)
    decay = 4 * math.pi * p[:, None] ** 2 + TAU * np.arange(maximum + 1)
    logs = height_log_integrals(decay, lower, upper, R, tau1)
    scale = logs[:, 0]
    j = np.exp(logs - scale[:, None])
    h = np.zeros((len(p), loop_order + 1, loop_order + 1), complex)
    bound = np.zeros(h.shape, float)
    for left in range(loop_order + 1):
        for right in range(loop_order + 1):
            for a, ca in ETA_TERMS:
                for b, cb in ETA_TERMS:
                    v = j[:, left + right + a + b]
                    h[:, left, right] += ca * cb * np.exp(
                        1j * TAU * (left - right + a - b) * tau1) * v
                    bound[:, left, right] += v
    return h, bound, scale


def validate_boundary_kernels():
    """Compare analytic thermal kernels with independent numerical integrals."""
    from scipy.integrate import quad
    pass
    from genus1_moduli_sampling import ShortPoint
    errors = []
    R = 8 / 3
    short = ShortPoint((0, 2, 1), 1.2, 2.4, .7, 2.1)
    pos = short.positions()
    Qy = (pos[0] + pos[1]).imag
    for p in (.06, .35, .8):
        for theta in (0., .8, 2.3):
            lower = 4.
            def density(L):
                T = (L + short.total) / TAU
                prefactor = math.exp(-2*p*p*L + Qy*Qy/(2*R*R*(L+short.total))) / math.sqrt(T)
                return prefactor * sum(compact_factor(R, complex(sign*theta/TAU,T), pos, [1,1,-2])
                                       for sign in (-1,1)) / (2*TAU)
            numeric = quad(density, lower, np.inf, epsabs=1e-11, epsrel=2e-10)[0]
            analytic = math.exp(primary_kernel_log([p], short, lower, R, theta=theta)[0])
            errors.append(abs(numeric/analytic-1))
    for lower, upper in ((.93,8.), (8.,math.inf)):
        for decay in (.03,.5,3.):
            numeric = quad(lambda T: math.exp(-decay*T)/math.sqrt(T)*
                           lattice_poisson(R,complex(.3,T)), lower, upper,
                           epsabs=1e-18,epsrel=1e-11)[0]
            analytic = math.exp(height_log_integrals(decay,lower,upper,R,.3))
            errors.append(abs(numeric/analytic-1))
    maximum = max(errors)
    assert maximum < 2e-9, maximum
    return {"independent_quadrature_cases": len(errors), "max_relative_difference": maximum}


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(_SOURCE_ROOT/"code/one_loop"))
    from genus1 import project
    print(validate_boundary_kernels())


# kernels
# Thermal kernels with explicit KK charges; no change to saved source kernels.

import genus1_source_paths as bootstrap
bootstrap.install()
import math
import numpy as np
from scipy.special import logsumexp
TAU = 2 * math.pi


def lattice(kin, tau, positions):
    return compact_factor(kin.R, complex(tau), positions, kin.charges)


def pair_lattice(kin, tau, pair, total):
    from genus1_three_point_collision import reconstruct_points
    w1, w2 = reconstruct_points(tuple(pair), 0j, complex(total))
    return lattice(kin, tau, (w1, w2, 0j))


def primary_length_log(p, short, lower, kin, *, theta=None, tolerance=1e-16):
    """Analytic long-length integral, with the original dL/(2 pi) measure."""
    p = np.atleast_1d(np.asarray(p, float))
    if np.any(p <= 0):
        raise ValueError('positive cut momenta required')
    q = kin.charge_position(short.positions())
    total = short.total + lower
    width = kin.R * math.sqrt(2 * (-math.log(tolerance) + 4) / total)
    center = q.imag / total
    ell = np.arange(math.floor(center-width)-1, math.ceil(center+width)+2, dtype=float)
    mcap = math.ceil(math.sqrt(2 * (-math.log(tolerance)+4) / (kin.R**2*total))) + 1
    mm, ll = np.meshgrid(np.arange(-mcap, mcap+1), ell, indexing='ij')
    mm, ll = mm.ravel(), ll.ravel()
    if theta is None:
        keep = mm*ll == 0
        mm, ll = mm[keep], ll[keep]
        phase = np.cos(mm*q.real)
    else:
        phase = np.cos(mm*q.real)*np.cos(mm*ll*theta)
    mass = (kin.R**2*mm**2 + ll**2/kin.R**2)/2
    rate = 2*p[:, None]**2 + mass[None, :]
    logs = (-2*p[:, None]**2*lower - mass[None, :]*total
            + ll[None, :]*q.imag/kin.R**2 - np.log(rate))
    value, sign = logsumexp(logs, b=phase[None, :], axis=1, return_sign=True)
    if np.any(sign <= 0):
        raise FloatingPointError('nonpositive integrated thermal kernel')
    return value - math.log(TAU*kin.R)


def primary_kernel_logs(p, short, kin, threshold=-math.log(.05), angle_order=32):
    """Paired noncompact/thermal kernels at identical physical momenta."""
    p = np.atleast_1d(np.asarray(p, float))
    q = kin.charge_position(short.positions())
    offset, harmonics, noncompact = integrated_primary_components(
        tuple(map(float, p)), short.s1, short.s2, abs(q.imag),
        kin.R, threshold, angle_order)
    factor = harmonics @ np.cos(np.arange(harmonics.shape[1])*q.real)
    with np.errstate(divide='ignore', invalid='ignore'):
        thermal = offset + np.log(factor)
    bad = np.flatnonzero(factor <= 0)
    if len(bad):
        floor = max(threshold, short.s1, short.s2)
        if floor + short.total >= TAU:
            thermal[bad] = primary_length_log(p[bad], short, floor, kin)
        else:
            edges = [0., math.pi]
            h = (floor+short.total)/TAU
            if math.sqrt(3)/2 < h < 1:
                edges.insert(1, TAU*math.sqrt(1-h*h))
            roots, weights = np.polynomial.legendre.leggauss(angle_order)
            logs = []
            for a, b in zip(edges, edges[1:]):
                for x, w in zip(roots, weights):
                    theta = a+(x+1)*(b-a)/2
                    lower = max(floor, TAU*math.sqrt(1-(theta/TAU)**2)-short.total)
                    logs.append(math.log(w*(b-a)/TAU) +
                                primary_length_log(p[bad], short, lower, kin, theta=theta))
            thermal[bad] = logsumexp(logs, axis=0)
    if not np.all(np.isfinite(thermal)):
        raise FloatingPointError('nonfinite thermal kernel')
    return noncompact, thermal

