"""Fixed-weight CCY c recursion with arbitrary-precision scalar arithmetic.

Same residues, global seed and star BPZ phases as ccy_sphere_six_point{,_star}.
No h recursion, c regulator, or descendant fallback is used. Caches are local
to one momentum triple, avoiding the old unbounded Ward-cache memory growth.
"""
from functools import lru_cache
from math import factorial

import mpmath as mp


def c_coefficients(topology, external_weights, internal_weights, order, *, dps=40,
                   central_charge=25, minimum_total_order=0, return_mpmath=False):
    if topology not in ("comb", "star"):
        raise ValueError("unknown topology")
    minimum_total_order = int(minimum_total_order)
    if order < 0 or dps < 25 or not 0 <= minimum_total_order <= order:
        raise ValueError("invalid order or precision")
    with mp.workdps(dps):
        external = tuple(mp.mpc(v) for v in external_weights)
        internal = tuple(mp.mpc(v) for v in internal_weights)
        if len(external) != 6 or len(internal) != 3:
            raise ValueError("six external and three internal weights required")

        # Intern exact MP values once, rather than hashing and comparing them
        # at every recurrence-cache lookup. Shifts still use the original MP
        # addition: neither rounding nor symbolic identification of weights
        # is used to merge states near a confluent pole.
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
        internal_ids = tuple(intern(h, weights, weight_ids) for h in internal)
        initial_charge = intern(mp.mpc(central_charge), charges, charge_ids)
        pole_tolerance = mp.power(10, -dps//2)
        null_levels = tuple(
            tuple((r, s, r*s) for r in range(2, n+1)
                  for s in range(1, n//r+1))
            for n in range(order+1)
        )

        @lru_cache(None)
        def shifted_weight(weight_id, level):
            return intern(weights[weight_id]+level, weights, weight_ids)

        @lru_cache(None)
        def rising(x, n):
            return mp.fprod(x+j for j in range(n))

        @lru_cache(None)
        def norm(weight_id, level):
            return factorial(level)*rising(2*weights[weight_id], level)

        @lru_cache(None)
        def two_leg(a, c, hi_id, hj_id, hk_id):
            # Ward identity restricted to global L_-1 descendants.
            hi, hj, hk = weights[hi_id], weights[hj_id], weights[hk_id]
            if a == 0:
                return rising(hj+hk-hi, c)
            value = (hi+a-1+hj-hk-c)*two_leg(a-1, c, hi_id, hj_id, hk_id)
            if c:
                value += c*(2*hk+c-1)*two_leg(a-1, c-1, hi_id, hj_id, hk_id)
            return value

        @lru_cache(None)
        def global_seed(levels, hs):
            a, b, c = levels
            h1, h2, h3 = (weights[index] for index in hs)
            d1, d2, d3, d4, d5, d6 = external
            denominator = mp.fprod(norm(index, n) for index, n in zip(hs, levels))
            if topology == "comb":
                numerator = (rising(h1+d2-d1, a)*two_leg(b, a, hs[1], external_ids[2], hs[0])
                             *two_leg(c, b, hs[2], external_ids[3], hs[1])*rising(h3+d5-d6, c))
            else:
                central = (two_leg(c, a, hs[2], hs[1], hs[0])
                           *mp.fprod(h3+c-h2-h1-a-j for j in range(b)))
                numerator = (rising(h1+d2-d1, a)*rising(h2+d4-d3, b)
                             *rising(h3+d5-d6, c)*central)
            return numerator/denominator

        @lru_cache(None)
        def pole(r, s, weight_id):
            h = weights[weight_id]
            x = (r*s-1+2*h+mp.sqrt((r-s)**2+4*(r*s-1)*h+4*h*h))/(1-r*r)
            b = mp.sqrt(x)
            numerator = -12*x**(2*r*s-1)
            denominator = (1-r*r)*x*x-(1-s*s)
            factors = [(p, ell) for p in range(1-r, r+1)
                       for ell in range(1-s, s+1) if (p, ell) not in ((0, 0), (r, s))]
            # Same universal x-1 and x+1 cancellations as minus_dc_dh_times_a_rs.
            for np_, ne in ((1, -1), (1, 1)):
                for i, (p, ell) in enumerate(factors):
                    if p and p*ne == ell*np_:
                        denominator *= p//np_
                        factors.pop(i)
                        break
                else:
                    numerator *= np_*x+ne
            denominator *= mp.fprod(p*x+ell for p, ell in factors)
            charge_id = intern(13+6*(x+1/x), charges, charge_ids)
            return charge_id, b, numerator/denominator

        @lru_cache(None)
        def fusion(r, s, b, top, bottom):
            q = b+1/b
            lt, lb = mp.sqrt(q*q-4*top), mp.sqrt(q*q-4*bottom)
            value = mp.mpc(1)
            for p in range(1-r, r, 2):
                for ell in range(1-s, s, 2):
                    shift = p*b+ell/b
                    value *= (lt+lb+shift)*(lt-lb+shift)/4
            return value

        @lru_cache(None)
        def residue(r, s, edge, hs):
            h1, h2, h3 = (weights[index] for index in hs)
            d1, d2, d3, d4, d5, d6 = external
            if topology == "comb":
                pairs = (((d1, d2), (h2, d3)), ((h1, d3), (h3, d4)),
                         ((h2, d4), (d6, d5)))
            else:
                pairs = (((d1, d2), (h3, h2)), ((d3, d4), (h3, h1)),
                         ((d6, d5), (h2, h1)))
            pc, b, prefactor = pole(r, s, hs[edge])
            left, right = pairs[edge]
            value = prefactor*fusion(r, s, b, *left)*fusion(r, s, b, *right)
            if topology == "star" and edge in (1, 2) and r*s % 2:
                value = -value
            return pc, value

        @lru_cache(None)
        def transition(c, hs, edge, r, s):
            # The residue/propagator and shifted module are independent of
            # the remaining descendant levels. Reuse them across that series.
            pc, res = residue(r, s, edge, hs)
            denominator = charges[c]-charges[pc]
            if abs(denominator) < pole_tolerance:
                raise ZeroDivisionError("c-recursion pole collision; no fallback allowed")
            hh = list(hs)
            hh[edge] = shifted_weight(hs[edge], r*s)
            return res/denominator, pc, tuple(hh)

        @lru_cache(None)
        def coefficient(levels, c, hs):
            total = global_seed(levels, hs)
            for edge, n in enumerate(levels):
                for r, s, level in null_levels[n]:
                    factor, pc, hh = transition(c, hs, edge, r, s)
                    ns = list(levels)
                    ns[edge] -= level
                    total += factor*coefficient(tuple(ns), pc, hh)
            return total

        convert = (lambda value: value) if return_mpmath else complex
        try:
            return {
                (a, b, c): convert(
                    coefficient((a, b, c), initial_charge, internal_ids))
                for a in range(order + 1)
                for b in range(order + 1 - a)
                for c in range(order + 1 - a - b)
                if a + b + c >= minimum_total_order
            }
        finally:
            # Recursive closures form cycles; free their potentially large
            # tables immediately instead of waiting for cyclic collection.
            for fn in (shifted_weight, rising, norm, two_leg, global_seed,
                       pole, fusion, residue, transition, coefficient):
                fn.cache_clear()
            weights.clear()
            weight_ids.clear()
            charges.clear()
            charge_ids.clear()
