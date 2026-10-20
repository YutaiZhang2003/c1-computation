"""Fresh dP rules for the two thermally modified integrated endpoints."""
import genus1_source_paths as bootstrap
bootstrap.install()
import math
import numpy as np
from scipy.linalg import eigh_tridiagonal
from kernels import primary_kernel_logs, height_log_integrals


def weighted_rule(log_density, decay, order, *, construction_order=384, tail_exponent=100.):
    """Gaussian rule in P^2, constructed from a positive measure.

    Returned weights integrate dP. The finite construction interval is an
    exponentially suppressed construction tail, not a physical momentum cut.
    Its refinement remains a separate numerical check.
    """
    if type(order) is not int or not 1 <= order <= 32 or construction_order < 8*order or decay <= 0:
        raise ValueError('resolved positive quadrature controls required')
    maximum = math.sqrt(tail_exponent/decay)
    roots, weights = np.polynomial.legendre.leggauss(construction_order)
    p = maximum*(roots+1)/2
    logs = np.log(maximum*weights/2) + log_density(p)
    peak = float(max(logs)); mass = np.exp(logs-peak); norm = math.fsum(mass)
    x = decay*p*p; vector = np.sqrt(mass/norm)
    vectors, diagonal, offdiagonal = [], [], []
    for j in range(order):
        vectors.append(vector)
        product = x*vector; a = float(vector@product); diagonal.append(a)
        residual = product-a*vector
        if j:
            residual -= offdiagonal[-1]*vectors[-2]
        for _ in range(2):
            for v in vectors:
                residual -= (v@residual)*v
        if j < order-1:
            b = float(np.linalg.norm(residual))
            if b <= 0:
                raise ArithmeticError('quadrature lost rank')
            offdiagonal.append(b); vector = residual/b
    eigenvalues, eigenvectors = eigh_tridiagonal(diagonal, offdiagonal)
    if np.any(eigenvalues <= 0):
        raise ArithmeticError('nonpositive momentum node')
    nodes = np.sqrt(eigenvalues/decay)
    lw = peak+math.log(norm)+2*np.log(abs(eigenvectors[0]))-log_density(nodes)
    return dict(nodes=nodes.tolist(), weights=np.exp(lw).tolist(), log_weights=lw.tolist(),
                measure='dP', order=order, construction_order=construction_order,
                construction_tail_exponent=tail_exponent, construction_maximum_P=maximum,
                physical_momentum_cutoff=False, requires_new_CFT_bank=True)


def primary_rule(short, kin, slope, order, *, threshold=-math.log(.05), **controls):
    lower = max(threshold, short.s1, short.s2)
    curvature = min(float(slope), .5*lower)
    def log_density(p):
        return 2*np.log(p)+primary_kernel_logs(p, short, kin, threshold)[1]+curvature*p*p
    result = weighted_rule(log_density, 2*lower-curvature, order, **controls)
    result.update(scheme='thermal-primary-kernel-v2', curvature=curvature,
                  kinematics=kin.record(), endpoint_power=0)
    return result


def height_rule(kin, slope, order, *, tau1, region, cutoff=8., **controls):
    lower = math.sqrt(1-tau1*tau1) if region == 'bulk' else cutoff
    upper = cutoff if region == 'bulk' else math.inf
    curvature = min(float(slope), math.pi*lower)
    def log_density(p):
        return (2*np.log(p) + height_log_integrals(4*math.pi*p*p, lower, upper, kin.R, tau1)
                + curvature*p*p)
    result = weighted_rule(log_density, 4*math.pi*lower-curvature, order, **controls)
    result.update(scheme='thermal-height-kernel-v2', curvature=curvature,
                  kinematics=kin.record(), tau1=tau1, region=region, cutoff=cutoff,
                  endpoint_power=2 if region == 'bulk' else 0)
    return result
