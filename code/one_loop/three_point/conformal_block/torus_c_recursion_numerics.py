"""Verified fixed-weight CCY c recursion for torus sewing graphs.

All residues, recursive sums and global seeds use arbitrary precision.  Exact
coincidences of intermediate poles are evaluated as a checked generic-weight
limit of the same c recursion.  PBW contraction and h recursion are never
production fallbacks.  The returned precision/refinement comparison is a
numerical stability diagnostic, not a bound on truncating the sewing series.
"""
from __future__ import annotations

from functools import lru_cache
from itertools import product
from math import factorial
import math

import mpmath as mp
import numpy as np

C_RECURSION_NUMERICS_VERSION = "mp-ccy-verified-v1"
C_RECURSION_BASE_DPS = 40
C_RECURSION_CHECK_DPS = 60
C_RECURSION_DEFAULT_RTOL = 2e-11
C_RECURSION_DEFAULT_ATOL = 2e-13


class CRecursionAccuracyError(ArithmeticError):
    """The requested coefficient accuracy could not be verified."""


def _orders(orders):
    if len(orders) not in (2, 3) or any(int(n) != n or n < 0 for n in orders):
        raise ValueError("two or three nonnegative integer sewing orders required")
    return tuple(int(n) for n in orders)


@lru_cache(None)
def _vacuum(order):
    result = [1] + [0] * order
    for oscillator in range(2, order + 1):
        for n in range(oscillator, order + 1):
            result[n] += result[n - oscillator]
    return tuple(result)


def mp_c_recursion_coefficients(kind, central_charge=25, *, external_weights,
                                internal_weights, orders, dps=50,
                                return_mpmath=False, coefficient_keys=None):
    """Compute one tensor with type-preserving multiprecision c recursion.

    ``return_mpmath`` retains arbitrary precision for weight-limit extraction;
    ordinary callers receive one final conversion to complex128.
    """
    return _mp_c_recursion(
        kind, central_charge, external_weights=external_weights,
        internal_weights=internal_weights, orders=orders, dps=dps,
        return_mpmath=return_mpmath, coefficient_keys=coefficient_keys)


def coefficient_region(orders, total_degree):
    """The source coefficients needed by a degree-limited frame conversion."""
    orders = _orders(orders)
    if isinstance(total_degree, bool) or int(total_degree) != total_degree or total_degree < 0:
        raise ValueError("nonnegative integer total coefficient degree required")
    return tuple(sorted((k for k in product(*(range(n + 1) for n in orders))
                         if sum(k) <= total_degree), key=lambda k: (sum(k), k)))


def _mp_c_recursion(kind, central_charge=25, *, external_weights,
                    internal_weights, orders, dps=50, return_mpmath=False,
                    seed_factory=None, q_values=None, total_order=None,
                    coefficient_keys=None):
    """Shared residue engine for coefficients and resummed pointwise blocks.

    With ``seed_factory`` the recursion counts null-level shifts only, and
    returns a one-element array containing the pointwise block. The factory
    is invoked inside the working-precision context with (external,q), then
    its callable is evaluated at every distinct shifted internal-weight
    tuple. Without it, the historical coefficient calculation is unchanged.
    """
    if kind not in {"necklace", "pair", "comb"}:
        raise ValueError("unknown torus c-recursion graph")
    pointwise = seed_factory is not None
    if pointwise != (q_values is not None):
        raise ValueError("pointwise recursion requires both q_values and a seed factory")
    if total_order is not None:
        if not pointwise or isinstance(total_order, bool) or int(total_order) != total_order or total_order < 0:
            raise ValueError("total_order is a nonnegative integer for pointwise recursion only")
        total_order = int(total_order)
    orders = _orders(orders)
    edges = len(orders)
    if edges != 3 and kind != "necklace":
        raise ValueError("pair and comb graphs require three sewing orders")
    if coefficient_keys is not None:
        if pointwise:
            raise ValueError("selected coefficient keys are not pointwise recursion orders")
        coefficient_keys = tuple(tuple(k) for k in coefficient_keys)
        if (len(set(coefficient_keys)) != len(coefficient_keys) or
                any(len(k) != edges or any(isinstance(x, bool) or int(x) != x or
                    x < 0 or x > orders[i] for i, x in enumerate(k)) for k in coefficient_keys)):
            raise ValueError("unique coefficient keys within the requested orders required")
    if len(external_weights) != edges or len(internal_weights) != edges:
        raise ValueError("one external and one internal weight per sewing edge required")
    if int(dps) != dps or dps < 30:
        raise ValueError("c-recursion precision must be an integer >=30")
    with mp.workdps(dps):
        external = tuple(mp.mpc(v) for v in external_weights)
        initial = tuple(mp.mpc(v) for v in internal_weights)
        initial_c = mp.mpc(central_charge)
        if not all(mp.isfinite(v) for v in (*external, *initial, initial_c)):
            raise ValueError("finite conformal weights and central charge required")
        if pointwise:
            if len(q_values) != edges:
                raise ValueError("one pointwise sewing parameter per edge required")
            qs = tuple(mp.mpc(q) for q in q_values)
            if not all(mp.isfinite(q) and abs(q) < 1 for q in qs):
                raise ValueError("finite pointwise parameters with |q_i| < 1 required")
            pointwise_seed = seed_factory(external, qs)
            q_powers = tuple(tuple(q ** n for n in range(order + 1))
                             for q, order in zip(qs, orders))

        # Use small cache keys while retaining exact MP equality and the
        # original addition for every shifted weight. Nearby weights and
        # different precisions must not be merged by rounding or symbolic
        # offsets, especially during the confluent-weight limit.
        weights, weight_ids = [], {}
        charges, charge_ids = [], {}

        def intern(value, values, ids):
            index = ids.get(value)
            if index is None:
                index = len(values)
                ids[value] = index
                values.append(value)
            return index

        external_ids = tuple(intern(h, weights, weight_ids) for h in external)
        initial_ids = tuple(intern(h, weights, weight_ids) for h in initial)
        initial_c_id = intern(initial_c, charges, charge_ids)
        pole_tolerance = mp.power(10, -dps + 10)
        null_levels = tuple(
            tuple((r, s, r * s) for r in range(2, n + 1)
                  for s in range(1, n // r + 1))
            for n in range(max(orders) + 1)
        )

        @lru_cache(None)
        def shifted_weight(weight_id, level):
            return intern(weights[weight_id] + level, weights, weight_ids)

        @lru_cache(None)
        def rising(x, n):
            return mp.fprod(x + j for j in range(n))

        @lru_cache(None)
        def norm(weight_id, level):
            return factorial(level) * rising(2 * weights[weight_id], level)

        @lru_cache(None)
        def two_leg(a, c, hi_id, hj_id, hk_id):
            # rho(L_-1^a h_i, h_j, L_-1^c h_k), at positions infinity,1,0.
            hi, hj, hk = weights[hi_id], weights[hj_id], weights[hk_id]
            if a == 0:
                return rising(hj + hk - hi, c)
            value = (hi + a - 1 + hj - hk - c) * two_leg(a - 1, c, hi_id, hj_id, hk_id)
            if c:
                value += c * (2 * hk + c - 1) * two_leg(a - 1, c - 1, hi_id, hj_id, hk_id)
            return value

        def three_leg(a, b, c, hi_id, hj_id, hk_id):
            # The middle L_-1 insertions differentiate the same monomial.
            hi, hj, hk = weights[hi_id], weights[hj_id], weights[hk_id]
            return (two_leg(a, c, hi_id, hj_id, hk_id)
                    * mp.fprod(hi + a - hj - hk - c - j for j in range(b)))

        @lru_cache(None)
        def global_seed(levels, hs):
            denominator = mp.fprod(norm(index, n) for index, n in zip(hs, levels))
            if kind == "necklace":
                numerator = mp.fprod(two_leg(levels[e], levels[(e + 1) % edges],
                                             hs[e], external_ids[e], hs[(e + 1) % edges])
                                     for e in range(edges))
                return numerator / denominator
            a, b, c = levels
            h0, h1, h2 = (weights[index] for index in hs)
            d0, d1, d2 = external
            if kind == "pair":
                numerator = (rising(h0 + d0 - d1, a)
                             * three_leg(b, a, c, hs[1], hs[0], hs[2])
                             * two_leg(c, b, hs[2], external_ids[2], hs[1]))
            else:
                numerator = (rising(h0 + d0 - d1, a)
                             * two_leg(b, a, hs[1], external_ids[2], hs[0])
                             * three_leg(c, b, c, hs[2], hs[1], hs[2]))
            return numerator / denominator

        @lru_cache(None)
        def regular_seed(levels, hs):
            if kind == "necklace":
                vacuum = _vacuum(min(levels))
                return mp.fsum(vacuum[k] * global_seed(tuple(n - k for n in levels), hs)
                               for k in range(min(levels) + 1) if vacuum[k])
            a, b, c = levels
            maximum = min(b, c) if kind == "pair" else c
            vacuum = _vacuum(maximum)
            return mp.fsum(vacuum[k] * global_seed(
                (a, b - k, c - k) if kind == "pair" else (a, b, c - k), hs)
                for k in range(maximum + 1) if vacuum[k])

        @lru_cache(None)
        def resummed_seed(hs):
            return pointwise_seed(tuple(weights[index] for index in hs))

        @lru_cache(None)
        def pole(r, s, weight_id):
            h = weights[weight_id]
            x = (r * s - 1 + 2 * h
                 + mp.sqrt((r - s) ** 2 + 4 * (r * s - 1) * h + 4 * h * h)) / (1 - r * r)
            b = mp.sqrt(x)
            numerator = -12 * x ** (2 * r * s - 1)
            denominator = (1 - r * r) * x * x - (1 - s * s)
            factors = [(p, ell) for p in range(1 - r, r + 1)
                       for ell in range(1 - s, s + 1)
                       if (p, ell) not in ((0, 0), (r, s))]
            # Algebraic x-1/x+1 cancellations in -dc/dh*A_rs happen before
            # numerical products; remaining rational-weight singularities
            # require the combined generic-weight limit of the full block.
            for np_, ne in ((1, -1), (1, 1)):
                for i, (p, ell) in enumerate(factors):
                    if p and p * ne == ell * np_:
                        denominator *= p // np_
                        factors.pop(i)
                        break
                else:
                    numerator *= np_ * x + ne
            denominator *= mp.fprod(p * x + ell for p, ell in factors)
            charge_id = intern(13 + 6 * (x + 1 / x), charges, charge_ids)
            return charge_id, b, numerator / denominator

        @lru_cache(None)
        def fusion(r, s, b, top, bottom):
            q = b + 1 / b
            lt, lb = mp.sqrt(q * q - 4 * top), mp.sqrt(q * q - 4 * bottom)
            return mp.fprod((lt + lb + p * b + ell / b)
                            * (lt - lb + p * b + ell / b) / 4
                            for p in range(1 - r, r, 2)
                            for ell in range(1 - s, s, 2))

        @lru_cache(None)
        def residue(r, s, edge, hs):
            values = tuple(weights[index] for index in hs)
            phase = False
            if kind == "necklace":
                before, after = (edge - 1) % edges, (edge + 1) % edges
                left, right = (values[before], external[before]), (values[after], external[edge])
            else:
                h0, h1, h2 = values
                d0, d1, d2 = external
            if kind == "pair":
                left, right = (((d1, d0), (h1, h2)),
                               ((h2, d2), (h0, h2)),
                               ((h1, d2), (h1, h0)))[edge]
                phase = edge in (0, 1)
            elif kind == "comb":
                left, right = (((d1, d0), (h1, d2)),
                               ((h0, d2), (h2, h2)),
                               ((h1, h2 + r * s), (h1, h2)))[edge]
            pc, b, prefactor = pole(r, s, hs[edge])
            value = prefactor * fusion(r, s, b, *left) * fusion(r, s, b, *right)
            if phase and r * s % 2:
                value = -value
            return pc, value

        @lru_cache(None)
        def transition(current_c, hs, edge, r, s):
            # This factor and shifted state are shared by all remaining
            # descendant levels at the same charge and internal weights.
            pc, res = residue(r, s, edge, hs)
            denominator = charges[current_c] - charges[pc]
            if abs(denominator) < pole_tolerance:
                if current_c == initial_c_id and hs == initial_ids:
                    raise CRecursionAccuracyError(
                        "target central charge is a c-recursion pole; "
                        "a generic-weight finite part is not a block value")
                raise ZeroDivisionError("confluent intermediate c-recursion poles")
            hh = list(hs)
            hh[edge] = shifted_weight(hs[edge], r * s)
            return res / denominator, pc, tuple(hh)

        @lru_cache(None)
        def coefficient(levels, current_c, hs):
            terms = [resummed_seed(hs) if pointwise else regular_seed(levels, hs)]
            for edge, n in enumerate(levels):
                for r, s, level in null_levels[n]:
                    if pointwise:
                        if qs[edge] == 0:
                            continue
                        if total_order is not None and sum(orders) - sum(levels) + level > total_order:
                            continue
                    factor, pc, hh = transition(current_c, hs, edge, r, s)
                    ns = list(levels)
                    ns[edge] -= level
                    if pointwise:
                        factor *= q_powers[edge][level]
                    terms.append(factor * coefficient(tuple(ns), pc, hh))
            return mp.fsum(terms)

        try:
            if pointwise:
                result = np.asarray([coefficient(orders, initial_c_id, initial_ids)], dtype=object)
            elif coefficient_keys is not None:
                result = np.asarray([coefficient(levels, initial_c_id, initial_ids)
                                     for levels in coefficient_keys], dtype=object)
            else:
                result = np.empty(tuple(n + 1 for n in orders), dtype=object)
                for levels in np.ndindex(result.shape):
                    result[levels] = coefficient(levels, initial_c_id, initial_ids)
            if not all(mp.isfinite(v) for v in result.flat):
                raise CRecursionAccuracyError("nonfinite multiprecision c-recursion coefficient")
            if return_mpmath:
                return result
            return _to_complex(result)
        finally:
            for fn in (shifted_weight, rising, norm, two_leg, global_seed,
                       regular_seed, resummed_seed, pole, fusion, residue, transition, coefficient):
                fn.cache_clear()
            weights.clear()
            weight_ids.clear()
            charges.clear()
            charge_ids.clear()


def _to_complex(tensor):
    result = np.asarray(tensor, dtype=np.complex128)
    if not np.all(np.isfinite(result)):
        raise CRecursionAccuracyError("c-recursion coefficients exceed complex128 range")
    return result


def _comparison(left, right, rtol, atol):
    left, right = _to_complex(left), _to_complex(right)
    delta = np.abs(left - right)
    scale = np.maximum(np.abs(left), np.abs(right))
    relative = float(np.max(delta / np.maximum(scale, 1.0)))
    return bool(np.all(delta <= atol + rtol * scale)), relative


def _weight_limit(kind, central_charge, external, internal, orders, *, dps, radius,
                  coefficient_keys=None):
    """Even Richardson limit in generic internal weights, through epsilon^4."""
    with mp.workdps(dps):
        direction = (mp.mpf(1), mp.sqrt(2), mp.sqrt(3))
        base = tuple(mp.mpc(h) for h in internal)
        even = []
        for divisor in (1, 2, 4):
            step = mp.mpf(radius) / divisor
            plus, minus = [mp_c_recursion_coefficients(
                kind, central_charge, external_weights=external,
                internal_weights=tuple(h + sign * step * d for h, d in zip(base, direction)),
                orders=orders, dps=dps, return_mpmath=True,
                coefficient_keys=coefficient_keys) for sign in (1, -1)]
            even.append((plus + minus) / 2)
        # Polynomial interpolation in epsilon^2 at zero, all MP scalars.
        return (even[0] - 20 * even[1] + 64 * even[2]) / 45


def stable_c_recursion_coefficients(kind, *, central_charge=25, external_weights,
                                   internal_weights, orders,
                                   rtol=C_RECURSION_DEFAULT_RTOL,
                                   atol=C_RECURSION_DEFAULT_ATOL,
                                   coefficient_keys=None):
    """Return ``(tensor, weight_limit_used, maximum_scaled_disagreement)``.

    Every accepted tensor passes a per-coefficient comparison at two working
    precisions. A confluent weight limit must additionally agree after its
    regulator radius is halved. Failures raise instead of entering a bank.
    """
    if not (math.isfinite(rtol) and rtol > 0 and math.isfinite(atol) and atol >= 0):
        raise ValueError("positive finite rtol and nonnegative finite atol required")
    orders = _orders(orders)
    if coefficient_keys is not None:
        coefficient_keys = tuple(tuple(k) for k in coefficient_keys)
        if not coefficient_keys:
            return np.empty(0, dtype=np.complex128), 0, 0.
    options = dict(external_weights=external_weights, internal_weights=internal_weights,
                   orders=orders, coefficient_keys=coefficient_keys)
    previous = None
    for dps in (C_RECURSION_BASE_DPS, C_RECURSION_CHECK_DPS, 90, 130):
        try:
            current = mp_c_recursion_coefficients(kind, central_charge, dps=dps, **options)
        except ZeroDivisionError:
            break
        if previous is not None:
            passed, error = _comparison(previous, current, rtol, atol)
            if passed:
                return current, 0, error
        previous = current
    else:
        raise CRecursionAccuracyError("c-recursion failed the 40/60/90/130 digit precision checks")

    # At exact rational/equal weights separate residues are undefined while
    # the c=25 physical block is regular. Keep the original target weights and
    # extract their analytic value from this same c recursion.
    for dps in (60, 90, 130):
        for radius in ("0.001", "0.0001", "0.00001"):
            try:
                first = _weight_limit(kind, central_charge, external_weights, internal_weights,
                                      orders, dps=dps, radius=radius,
                                      coefficient_keys=coefficient_keys)
                refined = _weight_limit(kind, central_charge, external_weights, internal_weights,
                                        orders, dps=dps, radius=str(mp.mpf(radius) / 2),
                                        coefficient_keys=coefficient_keys)
                passed, radius_error = _comparison(first, refined, rtol, atol)
                if not passed:
                    continue
                checked = _weight_limit(kind, central_charge, external_weights, internal_weights,
                                        orders, dps=dps + 20, radius=str(mp.mpf(radius) / 2),
                                        coefficient_keys=coefficient_keys)
                passed, precision_error = _comparison(refined, checked, rtol, atol)
                if passed:
                    return _to_complex(checked), 1, max(radius_error, precision_error)
            except ZeroDivisionError:
                continue
    raise CRecursionAccuracyError("confluent c-recursion weight limit did not converge")


__all__ = ["mp_c_recursion_coefficients", "stable_c_recursion_coefficients",
           "coefficient_region",
           "CRecursionAccuracyError", "C_RECURSION_NUMERICS_VERSION",
           "C_RECURSION_BASE_DPS", "C_RECURSION_CHECK_DPS",
           "C_RECURSION_DEFAULT_RTOL", "C_RECURSION_DEFAULT_ATOL"]
