"""Analytically grouped four-point elliptic h recursion at exactly c=25.

This is an arbitrary-precision numerical evaluation of a *formal* Laurent
recursion in x=b**2-1.  It never evaluates a small nonzero x.  Complete singular
shifted states are retained; only the physical root is reduced to its finite
part.  External weights are ordered (d_0,d_z,d_1,d_infinity).  Coefficients
multiply q**n, including the conventional 16**(r*s) in the residues.

The reusable setup separates the external weights from the physical exchanged
weight.  Its result is a table of rational functions of the latter, with poles
at 1-(r+s)**2/4, so evaluating further middle momenta needs no new recursion.

Jet bound: the null-norm pole order is nu=2*min(r,s)-1-(r==s)<=r*s-1.
Every vanishing shifted denominator has a simple zero.  Consequently a level-n
shifted coefficient has valuation >=-n.  For requested order N it suffices to
retain H_n[-n:N-n] and V_n[1-n:N-n+1], where
V_(n,rs)=R_rs H_(n-rs)(h_rs+rs).  Residues need powers through N-rs+1.
These bounds apply to complete paths and do not assume intermediate finite
parts.  The bounds and physical-root cancellations are checked in the setup.
"""

from __future__ import annotations

import functools
import decimal
import math
import time
from collections import OrderedDict
from typing import Any, Sequence

import mpmath as mp


SPHERE_FOUR_POINT_C25_VERSION = "formal-laurent-x-v1"


class C25CancellationError(ArithmeticError):
    """The formal root's principal parts did not cancel at working precision."""


def _mul(a, b, maximum):
    """Truncated ordinary polynomial product; works for integers or MP values."""
    out = [0] * (min(maximum + 1, len(a) + len(b) - 1))
    for i, ai in enumerate(a):
        if ai:
            for j in range(min(len(b), len(out) - i)):
                if b[j]:
                    out[i + j] += ai * b[j]
    return out


def _divide_linear(a, constant, slope, maximum):
    out = [0] * (maximum + 1)
    for i in range(maximum + 1):
        numerator = ((a[i] if i < len(a) else 0)
                     - (slope * out[i - 1] if i else 0))
        out[i] = numerator / constant if numerator else 0
    return out


@functools.lru_cache(maxsize=None)
def _structure(order):
    """Integer-only, precision-independent deformation and transition data."""
    labels = tuple((r, s) for r in range(1, order + 1)
                   for s in range(1, order // r + 1))
    records = []
    for r, s in labels:
        level = r * s
        linear = []
        zero_product = 1
        nu = 0
        for p in range(1 - r, r + 1):
            for ell in range(1 - s, s + 1):
                if (p, ell) in ((0, 0), (r, s)):
                    continue
                if p + ell == 0:
                    nu += 1
                    zero_product *= p
                else:
                    linear.append((p + ell, p))
        assert nu == 2 * min(r, s) - 1 - int(r == s)
        assert nu <= level - 1
        fusion_pairs = tuple((p, ell)
            for p in range(1 - r, r, 2)
            for ell in range(1 - s, s, 2)
            if p > 0 or (p == 0 and ell > 0))
        assert 2 * len(fusion_pairs) + int(level % 2) == level
        # delta h_rs = x*(a+b*x)/(4*(1+x)).  Its m-th power has
        # integer Taylor coefficients divided by 4**m.
        a, b = s * s - r * r, 1 - r * r
        powers = [((0, 1),)]
        for m in range(1, order):
            degree = order - 1 - m
            if degree < 0 or (a == 0 and b == 0):
                powers.append(())
                continue
            polynomial = [math.comb(m, i) * a ** (m - i) * b ** i
                          for i in range(min(m, degree) + 1)]
            inverse = [(-1) ** k * math.comb(m + k - 1, k)
                       for k in range(degree + 1)]
            product = _mul(polynomial, inverse, degree)
            powers.append(tuple((k + m, value) for k, value in enumerate(product)
                                if value))
        records.append({"label": (r, s), "level": level, "nu": nu,
                        "zero_product": zero_product, "linear": tuple(linear),
                        "fusion_pairs": fusion_pairs, "delta_powers": tuple(powers),
                        "pole_sum": r + s})
    transitions = {}
    for i, (a, b) in enumerate(labels):
        for j, (r, s) in enumerate(labels):
            if a * b + r * s > order:
                continue
            # 4*(1+x)*(h_ab+ab-h_rs).
            numerator = ((r + s) ** 2 - (a - b) ** 2,
                         2 * (r * r - a * a + a * b + r * s),
                         r * r - a * a)
            if numerator[0] == 0:
                assert numerator[1] != 0, "unexpected nonsimple shifted resonance"
            transitions[i, j] = numerator
    return tuple(records), transitions


def _fusion_numerator(p, ell, top, bottom):
    """Paired fusion factor times 16*(1+x)**2, without momentum roots."""
    delta, sigma = top - bottom, top + bottom
    v2 = _mul([p + ell, p], [p + ell, p], 4)
    v4 = _mul(v2, v2, 4)
    yv2 = _mul([1, 1], v2, 4)
    qv2 = _mul([4, 4, 1], v2, 4)  # (y+1)**2 * (p*y+ell)**2
    out = [0] * 5
    for i in range(5):
        out[i] = ((v4[i] if i < len(v4) else 0)
                  + 8 * sigma * (yv2[i] if i < len(yv2) else 0)
                  - 4 * (qv2[i] if i < len(qv2) else 0))
    for i, integer in enumerate((1, 2, 1)):
        out[i] += 16 * delta * delta * integer
    return out


def _residue(record, external, order):
    r, s = record["label"]
    level, nu = record["level"], record["nu"]
    maximum = order - level + 1 + nu
    d0, dz, d1, dinf = external
    # After pairing both fusion products the powers of y reduce to y^(+/-1).
    constant = (8 * (d0 - dz) * (dinf - d1) if level % 2 else (external[0] * 0 + 1) / 2)
    out = [constant / record["zero_product"]]
    for p, ell in record["fusion_pairs"]:
        out = _mul(out, _fusion_numerator(p, ell, d0, dz), maximum)
        out = _mul(out, _fusion_numerator(p, ell, dinf, d1), maximum)
    if level % 2:
        out = _mul(out, [1, 1], maximum)
    else:
        out = _divide_linear(out, 1, 1, maximum)
    for constant, slope in record["linear"]:
        out = _divide_linear(out, constant, slope, maximum)
    out.extend([0] * (maximum + 1 - len(out)))
    return -nu, tuple(out)


def _laurent_product(a_low, a, b_low, b, low, high):
    result = [0] * (high - low + 1)
    shift = a_low + b_low - low
    for i, ai in enumerate(a):
        if not ai:
            continue
        start = max(0, -i - shift)
        stop = min(len(b), len(result) - i - shift)
        for j in range(start, stop):
            if b[j]:
                result[i + j + shift] += ai * b[j]
    return result


def _shifted_division(values, numerator, order):
    """V[1-n:N-n+1] / D -> H[-n:N-n], via a quadratic recurrence."""
    p0, p1, p2 = numerator
    out = [0] * (order + 1)
    if p0:
        for k in range(1, order + 1):
            rhs = 4 * (values[k - 1] + (values[k - 2] if k >= 2 else 0))
            value = rhs - p1 * out[k - 1] - (p2 * out[k - 2] if k >= 2 else 0)
            out[k] = value / p0 if value else 0
    else:
        for k in range(order + 1):
            rhs = 4 * (values[k] + (values[k - 1] if k else 0))
            value = rhs - (p2 * out[k - 1] if k else 0)
            out[k] = value / p1 if value else 0
    return out


class FourPointC25:
    """Reusable exact-c25 grouped setup for fixed external weights.

    ``coefficients(h)`` returns H_0,...,H_N in the ordinary elliptic nome q.
    Setup starts at ``dps`` decimal digits and, by default, raises working
    precision until a fixed cancellation tolerance passes. ``requested_dps``
    and ``dps`` record the requested and actual working precision. Returned MP
    numbers can be evaluated under the caller's context, including derivatives.
    Passing the cancellation check is an arithmetic diagnostic, not a block-tail
    bound or a substitute for independent PBW/regulator validation.
    """

    def __init__(self, external_weights: Sequence[Any], order: int = 24,
                 dps: int = 60, *, cancellation_tolerance=None, arithmetic="decimal",
                 adaptive_precision=True, max_dps=None):
        if len(external_weights) != 4:
            raise ValueError("four external weights (d0,dz,d1,dinf) are required")
        if int(order) != order or not 0 <= int(order) <= 24:
            raise ValueError("order must be an integer between 0 and 24")
        if int(dps) != dps or int(dps) < 30:
            raise ValueError("dps must be an integer at least 30")
        if arithmetic not in ("mpmath", "decimal"):
            raise ValueError("arithmetic must be 'mpmath' or 'decimal'")
        self.order, self.dps = int(order), int(dps)
        self.requested_dps = self.dps
        if max_dps is None:
            max_dps = max(300, self.requested_dps)
        if int(max_dps) != max_dps or max_dps < self.requested_dps:
            raise ValueError("max_dps must be an integer at least dps")
        maximum_dps = int(max_dps) if adaptive_precision else self.requested_dps
        self.arithmetic = arithmetic
        start = time.perf_counter()
        with mp.workdps(self.dps):
            self.external_weights = tuple(mp.mpmathify(x) for x in external_weights)
            if not all(mp.isfinite(x) for x in self.external_weights):
                raise ValueError("external weights must be finite")
            tolerance = (mp.power(10, -(self.dps // 2)) if cancellation_tolerance is None
                         else mp.mpf(cancellation_tolerance))
            if not 0 < tolerance < 1:
                raise ValueError("cancellation_tolerance must lie between zero and one")
        if arithmetic == "decimal" and any(mp.im(value) for value in self.external_weights):
            raise ValueError("Decimal arithmetic requires real external weights")
        attempts = []
        while True:
            attempt_start = time.perf_counter()
            try:
                with mp.workdps(self.dps):
                    if arithmetic == "decimal":
                        with decimal.localcontext() as context:
                            context.prec = self.dps
                            decimal_external = tuple(decimal.Decimal(mp.nstr(mp.re(value), self.dps))
                                                     for value in self.external_weights)
                            rational, detail = self._build(
                                decimal.Decimal(mp.nstr(tolerance, self.dps)), decimal_external)
                            self._rational = tuple(tuple((s, k, mp.mpf(str(value)))
                                                          for s, k, value in row)
                                                   for row in rational)
                    else:
                        self._rational, detail = self._build(tolerance, self.external_weights)
                attempts.append({"working_dps": self.dps, "status": "passed",
                                 "seconds": time.perf_counter() - attempt_start,
                                 "scaled_residual": detail["maximum_principal_part_scaled_residual"]})
                break
            except C25CancellationError as error:
                attempts.append({"working_dps": self.dps, "status": "failed",
                                 "seconds": time.perf_counter() - attempt_start,
                                 "reason": str(error)})
                if self.dps >= maximum_dps:
                    raise C25CancellationError(
                        f"formal c25 precision budget exhausted at {self.dps} digits; "
                        f"requested {self.requested_dps}; {error}") from error
                self.dps = min(maximum_dps, self.dps + max(20, self.requested_dps // 2))
        with mp.workdps(self.dps):
            self.diagnostics = {
                "backend": SPHERE_FOUR_POINT_C25_VERSION,
                "central_charge": 25, "deformation": "formal x=b^2-1; no sampled regulator",
                "order": self.order, "dps": self.dps, "arithmetic": self.arithmetic,
                "requested_dps": self.requested_dps, "working_dps": self.dps,
                "adaptive_precision": bool(adaptive_precision),
                "maximum_working_dps": maximum_dps, "precision_attempts": attempts,
                "jet_bound": "H_n[-n,N-n], V_n[1-n,N-n+1], R_rs through N-rs+1",
                "cancellation_tolerance": mp.nstr(tolerance, self.dps),
                **detail,
            }
        self.diagnostics["setup_seconds"] = time.perf_counter() - start
        self._coefficient_cache = OrderedDict()
        self._coefficient_diagnostics = OrderedDict()
        self.diagnostics["coefficient_cache_capacity"] = 256

    def _build(self, tolerance, external):
        order = self.order
        records, transitions = _structure(order)
        residues = [_residue(rec, external, order) for rec in records]
        states = {}
        rational = [()]
        maximum_principal_relative = 0
        maximum_principal_absolute = 0
        maximum_constant_condition = 1
        state_count = transition_count = 0
        for n in range(1, order + 1):
            terms = {}
            for j, rec in enumerate(records):
                if rec["level"] > n:
                    continue
                remaining = n - rec["level"]
                h_low, h_values = ((0, (external[0] * 0 + 1,)) if remaining == 0
                                   else (-remaining, states[remaining, j]))
                terms[j] = _laurent_product(*residues[j], h_low, h_values,
                                            1 - n, order - n + 1)
            # Group every coalescing root pole before extracting its finite part.
            grouped = {}
            scales = {}
            for j, values in terms.items():
                rec = records[j]
                for m, power in enumerate(rec["delta_powers"][:n]):
                    if not power or power[0][0] >= n:
                        continue
                    key = (rec["pole_sum"], m + 1)
                    target = grouped.setdefault(key, [0] * n)
                    scale = scales.setdefault(key, [0] * n)
                    for k, integer in power:
                        if k >= n:
                            break
                        for vi in range(n - k):
                            contribution = values[vi] * integer
                            target[vi + k] += contribution
                            scale[vi + k] += abs(contribution)
            result = []
            for (pole_sum, power), values in grouped.items():
                scale = scales[pole_sum, power]
                denominator = 4 ** (power - 1)
                for k in range(n - 1):
                    relative = abs(values[k]) / max(1, scale[k])
                    maximum_principal_relative = max(maximum_principal_relative, relative)
                    maximum_principal_absolute = max(maximum_principal_absolute,
                                                     abs(values[k]) / denominator)
                constant = values[-1] / denominator
                if constant:
                    maximum_constant_condition = max(maximum_constant_condition,
                        scale[-1] / max(1, abs(values[-1])))
                    result.append((pole_sum, power, constant))
            rational.append(tuple(result))
            # These are shifted-state Laurent series, not their finite parts.
            for i, rec in enumerate(records):
                if rec["level"] + n > order:
                    continue
                total = [0] * (order + 1)
                for j, values in terms.items():
                    divided = _shifted_division(values, transitions[i, j], order)
                    for k, value in enumerate(divided):
                        total[k] += value
                    transition_count += 1
                states[n, i] = tuple(total)
                state_count += 1
        if maximum_principal_relative > tolerance:
            raise C25CancellationError(
                "physical-root Laurent principal parts failed cancellation: "
                f"relative residual {str(maximum_principal_relative)} "
                f"> {str(tolerance)}; increase dps or investigate the recurrence")
        detail = {
            "kac_labels": len(records), "shifted_states": state_count,
            "shifted_transition_evaluations": transition_count,
            "maximum_principal_part_scaled_residual": str(maximum_principal_relative),
            "maximum_principal_part_absolute_residual": str(maximum_principal_absolute),
            "maximum_root_group_sum_condition_proxy": str(maximum_constant_condition),
            "principal_parts_checked": True,
            "rational_terms": sum(len(row) for row in rational),
            "block_truncation_certified": False,
        }
        return tuple(rational), detail

    def coefficients(self, internal_weight: Any):
        """Evaluate all coefficient rational functions at a positive real h."""
        with mp.workdps(self.dps):
            h = mp.mpmathify(internal_weight)
            if not mp.isfinite(h) or mp.im(h) or mp.re(h) <= 0:
                raise ValueError("this physical-root API requires finite real h>0")
            h = mp.re(h)
            cache_key = h._mpf_
            if cache_key in self._coefficient_cache:
                self._coefficient_cache.move_to_end(cache_key)
                return self._coefficient_cache[cache_key]
            denominator_cache = {}
            maximum_condition = mp.mpf(1)
            out = [mp.mpf(1)]
            for row in self._rational[1:]:
                terms = []
                for pole_sum, power, value in row:
                    key = pole_sum, power
                    if key not in denominator_cache:
                        denominator_cache[key] = (h - 1 + mp.mpf(pole_sum * pole_sum) / 4) ** (-power)
                    terms.append(value * denominator_cache[key])
                value = mp.fsum(terms)
                maximum_condition = max(maximum_condition,
                    mp.fsum(abs(term) for term in terms) / max(mp.mpf(1), abs(value)))
                out.append(value)
            result = tuple(out)
            self._coefficient_cache[cache_key] = result
            self._coefficient_diagnostics[cache_key] = {
                "maximum_rational_sum_condition_proxy": mp.nstr(maximum_condition, 20),
                "dps": self.dps, "coefficient_relative_accuracy_certified": False}
            if len(self._coefficient_cache) > 256:
                expired, _ = self._coefficient_cache.popitem(last=False)
                del self._coefficient_diagnostics[expired]
            return result

    def coefficient_diagnostics(self, internal_weight):
        """Per-root sum-condition proxy; not an independent precision certificate."""
        self.coefficients(internal_weight)
        with mp.workdps(self.dps):
            key = mp.re(mp.mpmathify(internal_weight))._mpf_
            return dict(self._coefficient_diagnostics[key])

    def coefficients_many(self, internal_weights: Sequence[Any]):
        """Reuse the external setup for each supplied physical exchanged weight."""
        return tuple(self.coefficients(h) for h in internal_weights)


def compute_four_point_c25(*, external_weights, internal_weight, order=24, dps=60):
    """Convenience one-weight interface; retain FourPointC25 for quadrature reuse."""
    return FourPointC25(external_weights, order=order, dps=dps).coefficients(internal_weight)
