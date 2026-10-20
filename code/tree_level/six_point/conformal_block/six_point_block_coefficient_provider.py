"""Verified sphere six-point CCY coefficient provider.

This module controls arithmetic and analytic continuation around the sphere
comb/star recurrence. It does not evaluate torus graph seeds, h recursions, or
PBW fallbacks. Verification concerns coefficient arithmetic; it is not an
estimate of the discarded conformal-block series or momentum-integral tails.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import mpmath as mp

from six_point_arbitrary_precision_c_recursion import c_coefficients


SPHERE_C_PROVIDER_VERSION = "sphere-ccy-verified-v1"


class SphereCoefficientAccuracyError(ArithmeticError):
    """The requested sphere coefficient arithmetic could not be verified."""


def _complex_table(table):
    converted = {key: complex(value) for key, value in table.items()}
    if not all(math.isfinite(value.real) and math.isfinite(value.imag)
               for value in converted.values()):
        raise SphereCoefficientAccuracyError("sphere coefficients exceed complex128 range")
    return converted


def _compare(left, right, *, rtol, atol):
    if left.keys() != right.keys():
        raise SphereCoefficientAccuracyError("precision checks produced different degree sets")
    left, right = _complex_table(left), _complex_table(right)
    maximum = 0.0
    passed = True
    for key, value in left.items():
        scale = max(abs(value), abs(right[key]))
        error = abs(value - right[key])
        maximum = max(maximum, error / max(1.0, scale))
        passed = passed and error <= atol + rtol * scale
    return passed, maximum


def _check_target_modules(central_charge, internal, order):
    """A weight-limit continuation must not regularize a true target Kac pole."""
    with mp.workdps(60):
        c = mp.mpc(central_charge)
        q = mp.sqrt((c - 1) / 6)
        b = (q + mp.sqrt(q * q - 4)) / 2
        if b == 0:
            raise ValueError("invalid central-charge uniformization")
        for weight in internal:
            h = mp.mpc(weight)
            for r in range(1, order + 1):
                for s in range(1, order // r + 1):
                    degenerate = (q * q - (r * b + s / b) ** 2) / 4
                    if abs(h - degenerate) < mp.mpf("1e-40"):
                        raise SphereCoefficientAccuracyError(
                            "target internal module is degenerate; an explicit quotient "
                            "or pole prescription is required"
                        )


def _weight_limit(topology, external, internal, order, *, central_charge,
                  dps, radius, minimum_total_order):
    with mp.workdps(dps):
        base = tuple(mp.mpc(weight) for weight in internal)
        directions = (mp.mpf(1), mp.sqrt(2), mp.sqrt(3))
        even = []
        for divisor in (1, 2, 4):
            step = mp.mpf(radius) / divisor
            values = [c_coefficients(
                topology, external,
                tuple(h + sign * step * d for h, d in zip(base, directions)),
                order, central_charge=central_charge, dps=dps,
                minimum_total_order=minimum_total_order, return_mpmath=True,
            ) for sign in (1, -1)]
            even.append({key: (values[0][key] + values[1][key]) / 2 for key in values[0]})
        # Extrapolate in the squared weight displacement, through its fourth power.
        return {key: (even[0][key] - 20 * even[1][key] + 64 * even[2][key]) / 45
                for key in even[0]}


@dataclass
class VerifiedSphereCProvider:
    """An explicit, independently precision-checked coefficient dependency.

    ``minimum_total_order`` and ``topology_orders`` support existing staged
    shell construction. Complete momentum banks should use their defaults.
    All memoized recurrence state remains local to one primitive evaluation.
    """

    base_dps: int = 40
    check_dps: int | None = None
    maximum_dps: int = 130
    rtol: float = 2.0e-11
    atol: float = 2.0e-13
    minimum_total_order: int = 0
    topology_orders: dict[str, int] | None = None
    _verified_tables: int = field(default=0, init=False, repr=False)
    _weight_limits: int = field(default=0, init=False, repr=False)
    _maximum_dps_used: int = field(default=0, init=False, repr=False)
    _maximum_disagreement: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self):
        if self.check_dps is None:
            self.check_dps = self.base_dps + 20
        if any(int(n) != n for n in (self.base_dps, self.check_dps, self.maximum_dps)):
            raise ValueError("working precisions must be integer decimal digits")
        if not 25 <= self.base_dps < self.check_dps <= self.maximum_dps:
            raise ValueError("require 25 <= base_dps < check_dps <= maximum_dps")
        self.base_dps, self.check_dps, self.maximum_dps = (
            int(self.base_dps), int(self.check_dps), int(self.maximum_dps)
        )
        if not (math.isfinite(self.rtol) and self.rtol > 0
                and math.isfinite(self.atol) and self.atol >= 0):
            raise ValueError("positive finite rtol and nonnegative finite atol required")
        if int(self.minimum_total_order) != self.minimum_total_order or self.minimum_total_order < 0:
            raise ValueError("minimum_total_order must be a nonnegative integer")
        self.minimum_total_order = int(self.minimum_total_order)
        if self.topology_orders is not None:
            self.topology_orders = dict(self.topology_orders)
            if any(kind not in {"comb", "star"} or int(n) != n or n < 0
                   for kind, n in self.topology_orders.items()):
                raise ValueError("invalid topology order overrides")
            self.topology_orders = {kind: int(n) for kind, n in self.topology_orders.items()}

    def metadata(self):
        return {
            "backend": "verified-fixed-weight-c",
            "version": SPHERE_C_PROVIDER_VERSION,
            "representation": "native-plumbing-descendant",
            "truncation": "total-degree",
            "base_dps": self.base_dps,
            "check_dps": self.check_dps,
            "maximum_dps": self.maximum_dps,
            "rtol": self.rtol,
            "atol": self.atol,
            "minimum_total_order": self.minimum_total_order,
            "topology_orders": None if self.topology_orders is None else dict(self.topology_orders),
            "h_or_pbw_fallback": False,
        }

    def diagnostics(self):
        return {
            "verified_tables": self._verified_tables,
            "weight_limit_count": self._weight_limits,
            "maximum_dps_used": self._maximum_dps_used,
            "maximum_scaled_disagreement": self._maximum_disagreement,
        }

    def _accept(self, table, *, dps, error, weight_limit):
        converted = _complex_table(table)
        self._verified_tables += 1
        self._weight_limits += int(weight_limit)
        self._maximum_dps_used = max(self._maximum_dps_used, dps)
        self._maximum_disagreement = max(self._maximum_disagreement, error)
        return converted

    def coefficients(self, topology, external_weights, internal_weights, order, *,
                     central_charge=25, minimum_total_order=None):
        if topology not in {"comb", "star"}:
            raise ValueError("sphere topology must be comb or star")
        if int(order) != order or order < 0:
            raise ValueError("order must be a nonnegative integer")
        order = int(order)
        if self.topology_orders is not None:
            order = min(order, int(self.topology_orders.get(topology, order)))
        minimum = self.minimum_total_order if minimum_total_order is None else minimum_total_order
        if int(minimum) != minimum or not 0 <= minimum <= order:
            raise ValueError("invalid minimum total degree")
        if len(external_weights) != 6 or len(internal_weights) != 3:
            raise ValueError("sphere six-point sewing requires six external and three internal weights")
        if not all(math.isfinite(complex(value).real) and math.isfinite(complex(value).imag)
                   for value in (*external_weights, *internal_weights, central_charge)):
            raise ValueError("finite CFT parameters are required")
        options = dict(central_charge=central_charge, minimum_total_order=int(minimum))
        ladder = sorted({self.base_dps, self.check_dps,
                         min(self.maximum_dps, self.check_dps + 30), self.maximum_dps})
        previous = None
        for dps in ladder:
            try:
                current = c_coefficients(topology, external_weights, internal_weights,
                                         order, dps=dps, **options)
            except ZeroDivisionError:
                break
            if previous is not None:
                passed, error = _compare(previous, current, rtol=self.rtol, atol=self.atol)
                if passed:
                    return self._accept(current, dps=dps, error=error, weight_limit=False)
            previous = current
        else:
            raise SphereCoefficientAccuracyError("sphere c-recursion precision ladder did not converge")

        _check_target_modules(central_charge, internal_weights, order)
        for dps in sorted({max(60, self.check_dps), max(90, self.check_dps), self.maximum_dps - 20}):
            if dps + 20 > self.maximum_dps:
                continue
            for radius in ("0.001", "0.0001", "0.00001"):
                try:
                    first = _weight_limit(topology, external_weights, internal_weights, order,
                                          dps=dps, radius=radius, **options)
                    half_radius = str(float(radius) / 2)
                    refined = _weight_limit(topology, external_weights, internal_weights, order,
                                            dps=dps, radius=half_radius, **options)
                    passed, radius_error = _compare(first, refined, rtol=self.rtol, atol=self.atol)
                    if not passed:
                        continue
                    checked = _weight_limit(topology, external_weights, internal_weights, order,
                                            dps=dps + 20, radius=half_radius, **options)
                    passed, precision_error = _compare(refined, checked, rtol=self.rtol, atol=self.atol)
                    if passed:
                        return self._accept(checked, dps=dps + 20,
                            error=max(radius_error, precision_error), weight_limit=True)
                except ZeroDivisionError:
                    continue
        raise SphereCoefficientAccuracyError("sphere confluent c-recursion weight limit did not converge")

    def __call__(self, topology, *, central_charge, external_weights, internal_weights,
                 order1, order2=None, order3=None, max_total_order=None):
        orders = (order1, order1 if order2 is None else order2,
                  order1 if order3 is None else order3,
                  order1 if max_total_order is None else max_total_order)
        if len(set(orders)) != 1:
            raise ValueError("verified sphere provider requires total-degree truncation")
        return self.coefficients(topology, external_weights, internal_weights, orders[0],
                                 central_charge=central_charge)


__all__ = ["VerifiedSphereCProvider", "SphereCoefficientAccuracyError", "SPHERE_C_PROVIDER_VERSION"]
