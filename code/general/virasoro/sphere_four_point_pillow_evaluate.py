"""Analytic z derivatives of a fixed four-point elliptic block polynomial.

External weights are ordered at (0,z,1,infinity). ``f`` is the native
descendant-normalized block F4/z**(h-d0-dz), including its entire elliptic
prefactor. Coefficients are the fixed coefficients of H(q), in ascending
nome powers; they must belong to the supplied weights and central charge.

The principal elliptic map is evaluated locally. This routine does not select
a global analytic-continuation path or change a conformal-block channel.
All arithmetic follows the caller's mpmath precision. Internal guard digits
protect removable cancellations near z=0 and z=1; results are rounded back
to the caller's context. There is no numerical differentiation at runtime.
"""
from __future__ import annotations

import operator
from typing import Any, NamedTuple

import mpmath as mp


class PillowJet(NamedTuple):
    """Value, first/second z derivatives, and the elliptic nome."""

    f: Any
    df: Any
    d2f: Any
    nome: Any


def _series_product(a, b, order):
    return tuple(mp.fsum(a[j] * b[n-j]
                        for j in range(max(0, n-len(b)+1), min(n+1, len(a))))
                 for n in range(order+1))


def _unit_series_power(series, exponent, order):
    """Formal power with its logarithm normalized to zero at q=0."""
    if series[0] != 1:
        raise ValueError("a formal power requires unit constant coefficient")
    exponent = mp.mpmathify(exponent)
    result = [mp.mpf(1)]
    # series*y' = exponent*series'*y; no logarithm or binomial truncation.
    for n in range(1, order+1):
        result.append(mp.fsum(((exponent+1)*k-n) * series[k] * result[n-k]
                              for k in range(1, min(n+1, len(series)))) / n)
    return tuple(result)


def complete_q_coefficients(external4, h, Hcoefficients, central_charge=25):
    """Include the entire descendant prefactor, then truncate to the input order.

    Return ascending coefficients of E(q)=P(q)H(q), with the ordinary native
    primary power z**(h-d0-dz) still removed. Thus E_N=T_N[P*H_N] preserves
    every known Taylor coefficient. It is a different finite-order evaluation
    from P(q)*H_N(q). Never pass the returned coefficients as H coefficients
    to ``evaluate_pillow_jet``: that would include the prefactor twice.

    For a=(c-1)/24 and theta2r=theta2/(2*q**(1/4)), the unit-constant
    prefactor is exactly theta2r**(4*(a-h))*theta4**(4*(a-dz-d1))*
    theta3**(4*(a+h-d0-dinf)) as a germ at q=0. All formal powers use that
    germ's logarithm. Arithmetic follows the caller's MP precision; large
    weights can require guard digits for cancellation in this transformation.
    """
    external = tuple(map(mp.mpmathify, external4))
    if len(external) != 4:
        raise ValueError("exactly four external weights are required")
    h, c = map(mp.mpmathify, (h, central_charge))
    coefficients = tuple(map(mp.mpmathify, Hcoefficients))
    if not coefficients:
        raise ValueError("a nonempty H coefficient sequence is required")
    if not all(mp.isfinite(v) for v in (*external, h, c, *coefficients)):
        raise ValueError("finite weights and coefficients are required")
    order = len(coefficients)-1
    theta3 = [mp.mpf(0)]*(order+1)
    theta4 = theta3[:]
    theta2r = theta3[:]
    theta3[0] = theta4[0] = theta2r[0] = mp.mpf(1)
    for j in range(1, order+1):
        if j*j <= order:
            theta3[j*j] = 2
            theta4[j*j] = 2*(-1)**j
        if j*(j+1) <= order:
            theta2r[j*(j+1)] = 1
    d0, dz, d1, di = external
    a = (c-1)/24
    prefactor = _series_product(
        _unit_series_power(theta2r, 4*(a-h), order),
        _unit_series_power(theta4, 4*(a-dz-d1), order), order)
    prefactor = _series_product(prefactor,
        _unit_series_power(theta3, 4*(a+h-d0-di), order), order)
    return _series_product(prefactor, coefficients, order)


def evaluate_complete_q_jet(z, coefficients, *, order=None):
    """Evaluate a COMPLETE q polynomial and its first two z derivatives.

    Unlike ``evaluate_pillow_jet``, this function adds no prefactor. The
    coefficients must already use ``complete_q_coefficients``' normalization.
    The principal elliptic map is local; this does not choose a continuation
    path across a branch cut. No coefficient or truncation accuracy is inferred.
    """
    z = mp.mpmathify(z)
    coefficients = tuple(map(mp.mpmathify, coefficients))
    if not coefficients:
        raise ValueError("a nonempty complete coefficient sequence is required")
    n = len(coefficients)-1 if order is None else operator.index(order)
    if not 0 <= n < len(coefficients):
        raise ValueError("order outside coefficient table")
    if not all(mp.isfinite(v) for v in (z, *coefficients)):
        raise ValueError("finite coordinate and coefficients are required")
    if z == 0 or z == 1:
        raise ValueError("z=0 and z=1 require endpoint limits")
    if mp.im(z) == 0 and mp.re(z) >= 1:
        raise ValueError("z is on the principal elliptic-K cut; specify a complex side")
    distance = min(abs(z), abs(1-z), mp.mpf(1))
    guard = 12 + max(0, int(mp.ceil(-2*mp.log10(distance))))
    with mp.extradps(guard):
        K, E = mp.ellipk(z), mp.ellipe(z)
        nome = mp.exp(-mp.pi*mp.ellipk(1-z)/K)
        if not mp.isfinite(nome) or not 0 < abs(nome) < 1:
            raise ValueError("elliptic nome outside the open unit disc")
        K1 = (E-(1-z)*K)/(2*z*(1-z))
        q_log1 = mp.pi**2/(4*z*(1-z)*K*K)
        q_log2 = q_log1*(-1/z+1/(1-z)-2*K1/K)
        q1 = nome*q_log1
        q2 = nome*(q_log1*q_log1+q_log2)
        value, first, second = _horner_jet(coefficients[:n+1], nome)
        values = value, first*q1, second*q1*q1+first*q2, nome
        if not all(mp.isfinite(v) for v in values):
            raise ArithmeticError("nonfinite complete block value or derivative")
    return PillowJet(*(+(v) for v in values))


def _horner_jet(coefficients, q):
    """Evaluate H, dH/dq, d²H/dq² without dividing by H or q."""
    value, first, second = coefficients[-1], mp.mpf(0), mp.mpf(0)
    for coefficient in reversed(coefficients[:-1]):
        second = q*second + 2*first
        first = q*first + value
        value = q*value + coefficient
    return value, first, second


def evaluate_pillow_jet(external4, h, z, coefficients, central_charge=25, *, order=None):
    """Return ``PillowJet(f, df/dz, d²f/dz², nome)`` at fixed coefficients.

    The normalization exactly matches the experimental
    ``evaluate_from_coefficients``: the ordinary z**(h-d0-dz) primary power
    is removed, while all remaining pillow factors are included. ``order``
    selects a prefix of the coefficient sequence; by default all are used.

    With K=K(z) in the *parameter* convention, the analytic identities are
    log(q)'=pi²/[4*z*(1-z)*K²] and
    z*(1-z)*K''+(1-2*z)*K'-K/4=0. The theta logarithmic derivatives follow
    from theta3(q)²=2*K/pi. The actual prefactor value uses theta3 directly
    and the same individual complex powers as the reference evaluator.

    No accuracy of the supplied coefficients, h-recursion limit, or truncation
    is inferred here. z=0,1 and the principal K branch cut [1,infinity) are
    rejected; a cut boundary value needs an explicitly chosen complex side.
    """
    external = tuple(mp.mpmathify(weight) for weight in external4)
    if len(external) != 4:
        raise ValueError("exactly four external weights are required")
    h, z, c = map(mp.mpmathify, (h, z, central_charge))
    coefficients = tuple(mp.mpmathify(value) for value in coefficients)
    if not coefficients:
        raise ValueError("a nonempty coefficient sequence is required")
    n = len(coefficients)-1 if order is None else operator.index(order)
    if not 0 <= n < len(coefficients):
        raise ValueError("order outside coefficient table")
    coefficients = coefficients[:n+1]
    if not all(mp.isfinite(value) for value in (*external, h, z, c, *coefficients)):
        raise ValueError("finite weights, coordinate and coefficients are required")
    if z == 0 or z == 1:
        raise ValueError("z=0 and z=1 require endpoint limits")
    if mp.im(z) == 0 and mp.re(z) >= 1:
        raise ValueError("z is on the principal elliptic-K cut; specify a complex side")

    # K' from E-K and the prefactor's second logarithmic derivative can lose
    # up to two powers of the endpoint distance. The guard is relative to
    # caller precision, never a fixed precision clamp (including under mp.diff).
    distance = min(abs(z), abs(1-z), mp.mpf(1))
    guard = 12 + max(0, int(mp.ceil(-2*mp.log10(distance))))
    with mp.extradps(guard):
        one_minus = 1-z
        K, E = mp.ellipk(z), mp.ellipe(z)
        nome = mp.exp(-mp.pi*mp.ellipk(one_minus)/K)
        if not mp.isfinite(nome) or not 0 < abs(nome) < 1:
            raise ValueError("elliptic nome outside the open unit disc")
        K1 = (E-one_minus*K)/(2*z*one_minus)
        K2 = ((2*z-1)*K1+K/4)/(z*one_minus)
        K_log1 = K1/K
        K_log2 = K2/K-K_log1*K_log1
        q_log1 = mp.pi**2/(4*z*one_minus*K*K)
        q_log2 = q_log1*(-1/z+1/one_minus-2*K_log1)
        q1 = nome*q_log1
        q2 = nome*(q_log1*q_log1+q_log2)

        d0, dz, d1, di = external
        a = (c-1)/24
        alpha, beta = h-a, a-dz-d1
        gamma = 12*a-4*(d0+dz+d1+di)
        theta = mp.jtheta(3, 0, nome)
        prefactor = (16*nome/z)**alpha * one_minus**beta * theta**gamma
        p_log1 = alpha*(q_log1-1/z)-beta/one_minus+gamma*K_log1/2
        p_log2 = alpha*(q_log2+1/(z*z))-beta/(one_minus*one_minus)+gamma*K_log2/2
        H, Hq, Hqq = _horner_jet(coefficients, nome)
        Hz = Hq*q1
        Hzz = Hqq*q1*q1+Hq*q2
        f = prefactor*H
        df = prefactor*(Hz+p_log1*H)
        d2f = prefactor*(Hzz+2*p_log1*Hz+(p_log2+p_log1*p_log1)*H)
        values = f, df, d2f, nome
        if not all(mp.isfinite(value) for value in values):
            raise ArithmeticError("nonfinite pillow value or derivative")
    return PillowJet(*(+(value) for value in values))
