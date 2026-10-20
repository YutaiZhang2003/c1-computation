#!/usr/bin/env python3
"""CCY recursions for the sphere five-point Virasoro block.

The external primaries ``(d1,...,d5)`` are inserted at

``(0, z1, z2, 1, infinity)``,

and the two internal weights ``(h1,h2)`` propagate in the linear channel

``((d1 d2) -> h1, (h1 d3) -> h2, (h2 d4 d5))``.

Following Cho--Collier--Yin (CCY), arXiv:1703.09805, equation (3.21), the
descendant expansion variables are

``q1 = z1 / z2`` and ``q2 = z2``.

All coefficient functions in this module return only the descendant series
``sum F[n1,n2] q1**n1 q2**n2``.  Three Liouville structure constants and the
primary-coordinate factor

``z1**(h1-d1-d2) * z2**(h2-d3-h1)``

are deliberately excluded.  The imaginary-frequency production uses the
fixed-weight central-charge (c) recursion.  The CCY internal-weight (h)
recursion, specialized from their equation (3.26), and the defining
descendant contraction are retained as independent checks.

At ``c=25`` the individual terms of the h-recursion are resonant because
several Kac weights coincide at ``b=1``.  ``sphere_five_point_h_c25_limit``
defines the finite answer by evaluating at ``b=exp(eta)`` and extrapolating
in ``eta**2`` to zero.  This regulated h-recursion is not used by the
imaginary-frequency production, which evaluates the c-recursion at
``c=25``.
"""

from __future__ import annotations

from c1_paths import PACKAGE_ROOT, data_root, package_path, module_path

import cmath
import math
from functools import lru_cache
from typing import Mapping, Sequence

import numpy as np

try:
    from ccy_genus2_block import (
        b_from_c_rs_h,
        c_rs_from_h,
        fusion_polynomial_for_weights,
        minus_dc_dh_times_a_rs,
    )
    from virasoro_blocks import (
        central_charge_to_b,
        degenerate_weight,
        fusion_polynomial_for_weights as fixed_c_fusion_polynomial_for_weights,
        zamolodchikov_a_rs,
    )
    from virasoro_plumbing_graph import (
        inverse_verma_gram_matrix,
        rho_primary_descendants,
    )
except ImportError:  # pragma: no cover - supports package-style execution
    from ccy_genus2_block import (
        b_from_c_rs_h,
        c_rs_from_h,
        fusion_polynomial_for_weights,
        minus_dc_dh_times_a_rs,
    )
    from virasoro_blocks import (
        central_charge_to_b,
        degenerate_weight,
        fusion_polynomial_for_weights as fixed_c_fusion_polynomial_for_weights,
        zamolodchikov_a_rs,
    )
    from virasoro_plumbing_graph import (
        inverse_verma_gram_matrix,
        rho_primary_descendants,
    )


CoefficientTable = dict[tuple[int, int], complex]


def _validate_orders(order1: int, order2: int | None) -> tuple[int, int]:
    order1 = int(order1)
    order2 = order1 if order2 is None else int(order2)
    if order1 < 0 or order2 < 0:
        raise ValueError("block orders must be non-negative")
    return order1, order2


def _weights(
    external_weights: Sequence[complex],
    internal_weights: Sequence[complex],
) -> tuple[tuple[complex, ...], tuple[complex, complex]]:
    if len(external_weights) != 5:
        raise ValueError("external_weights must contain (d1,d2,d3,d4,d5)")
    if len(internal_weights) != 2:
        raise ValueError("internal_weights must contain (h1,h2)")
    return (
        tuple(complex(value) for value in external_weights),
        (complex(internal_weights[0]), complex(internal_weights[1])),
    )


def _rising(value: complex, order: int) -> complex:
    out = 1.0 + 0.0j
    for offset in range(int(order)):
        out *= complex(value) + offset
    return out


def _factorial_rising_norm(weight: complex, level: int) -> complex:
    return math.factorial(int(level)) * _rising(2.0 * complex(weight), int(level))


def sphere_five_point_global_coefficient(
    level1: int,
    level2: int,
    *,
    external_weights: Sequence[complex],
    internal_weights: Sequence[complex],
) -> complex:
    """Return one coefficient of the global five-point linear block."""

    level1, level2 = _validate_orders(level1, level2)
    external, internal = _weights(external_weights, internal_weights)
    d1, d2, d3, d4, d5 = external
    h1, h2 = internal
    left = _rising(h1 + d2 - d1, level1)
    middle = rho_primary_descendants(
        (1,) * level2,
        (),
        (1,) * level1,
        h2,
        d3,
        h1,
        0.0,
    )
    right = _rising(h2 + d4 - d5, level2)
    denominator = (
        _factorial_rising_norm(h1, level1)
        * _factorial_rising_norm(h2, level2)
    )
    if denominator == 0.0:
        raise ZeroDivisionError("global block has a singular internal weight")
    return complex(left * middle * right / denominator)


def sphere_five_point_global_coefficients(
    *,
    external_weights: Sequence[complex],
    internal_weights: Sequence[complex],
    order1: int,
    order2: int | None = None,
    max_total_order: int | None = None,
) -> CoefficientTable:
    """Return a rectangular or total-degree-truncated global coefficient table."""

    order1, order2 = _validate_orders(order1, order2)
    if max_total_order is not None and int(max_total_order) < 0:
        raise ValueError("max_total_order must be non-negative")
    return {
        (n1, n2): sphere_five_point_global_coefficient(
            n1,
            n2,
            external_weights=external_weights,
            internal_weights=internal_weights,
        )
        for n1 in range(order1 + 1)
        for n2 in range(order2 + 1)
        if max_total_order is None or n1 + n2 <= int(max_total_order)
    }


def sphere_five_point_direct_coefficients(
    *,
    central_charge: complex,
    external_weights: Sequence[complex],
    internal_weights: Sequence[complex],
    order1: int,
    order2: int | None = None,
    max_total_order: int | None = None,
) -> CoefficientTable:
    """Evaluate the defining two-edge Verma-module descendant contraction."""

    order1, order2 = _validate_orders(order1, order2)
    external, internal = _weights(external_weights, internal_weights)
    d1, d2, d3, d4, d5 = external
    h1, h2 = internal
    c_value = complex(central_charge)
    out: CoefficientTable = {}

    gram1 = {
        level: inverse_verma_gram_matrix(level, h1, c_value)[:2]
        for level in range(order1 + 1)
    }
    gram2 = {
        level: inverse_verma_gram_matrix(level, h2, c_value)[:2]
        for level in range(order2 + 1)
    }
    for n1 in range(order1 + 1):
        for n2 in range(order2 + 1):
            if max_total_order is not None and n1 + n2 > int(max_total_order):
                continue
            basis1, inverse1 = gram1[n1]
            basis2, inverse2 = gram2[n2]
            coefficient = 0.0 + 0.0j
            for a1_index, a1 in enumerate(basis1):
                for b1_index, b1 in enumerate(basis1):
                    metric1 = inverse1[a1_index, b1_index]
                    if metric1 == 0.0:
                        continue
                    left = rho_primary_descendants(
                        b1,
                        (),
                        (),
                        h1,
                        d2,
                        d1,
                        c_value,
                    )
                    if left == 0.0:
                        continue
                    for a2_index, a2 in enumerate(basis2):
                        right = rho_primary_descendants(
                            (),
                            (),
                            a2,
                            d5,
                            d4,
                            h2,
                            c_value,
                        )
                        if right == 0.0:
                            continue
                        for b2_index, b2 in enumerate(basis2):
                            metric2 = inverse2[a2_index, b2_index]
                            if metric2 == 0.0:
                                continue
                            middle = rho_primary_descendants(
                                b2,
                                (),
                                a1,
                                h2,
                                d3,
                                h1,
                                c_value,
                            )
                            coefficient += metric1 * metric2 * left * middle * right
            out[(n1, n2)] = complex(coefficient)
    return out


def sphere_five_point_c_coefficients(
    *,
    central_charge: complex,
    external_weights: Sequence[complex],
    internal_weights: Sequence[complex],
    order1: int,
    order2: int | None = None,
    max_total_order: int | None = None,
    pole_tolerance: float = 1.0e-12,
) -> CoefficientTable:
    """Return coefficients from the CCY fixed-weight ``c``-recursion."""

    order1, order2 = _validate_orders(order1, order2)
    external, internal = _weights(external_weights, internal_weights)
    d1, d2, d3, d4, d5 = external
    initial_h1, initial_h2 = internal
    c_value = complex(central_charge)
    pole_tolerance = float(pole_tolerance)
    if pole_tolerance <= 0.0:
        raise ValueError("pole_tolerance must be positive")

    @lru_cache(maxsize=None)
    def coefficient(
        n1: int,
        n2: int,
        current_c: complex,
        h1: complex,
        h2: complex,
    ) -> complex:
        total = sphere_five_point_global_coefficient(
            n1,
            n2,
            external_weights=external,
            internal_weights=(h1, h2),
        )
        for r in range(2, n1 + 1):
            for s in range(1, n1 // r + 1):
                null_level = r * s
                pole_c = c_rs_from_h(r, s, h1)
                denominator = current_c - pole_c
                if abs(denominator) < pole_tolerance:
                    raise ZeroDivisionError(
                        "five-point c-recursion encountered an edge-1 pole collision"
                    )
                b_pole = b_from_c_rs_h(r, s, h1)
                residue = (
                    minus_dc_dh_times_a_rs(r, s, h1)
                    * fusion_polynomial_for_weights(r, s, b_pole, d1, d2)
                    * fusion_polynomial_for_weights(r, s, b_pole, h2, d3)
                )
                total += residue / denominator * coefficient(
                    n1 - null_level,
                    n2,
                    pole_c,
                    h1 + null_level,
                    h2,
                )
        for r in range(2, n2 + 1):
            for s in range(1, n2 // r + 1):
                null_level = r * s
                pole_c = c_rs_from_h(r, s, h2)
                denominator = current_c - pole_c
                if abs(denominator) < pole_tolerance:
                    raise ZeroDivisionError(
                        "five-point c-recursion encountered an edge-2 pole collision"
                    )
                b_pole = b_from_c_rs_h(r, s, h2)
                residue = (
                    minus_dc_dh_times_a_rs(r, s, h2)
                    * fusion_polynomial_for_weights(r, s, b_pole, h1, d3)
                    * fusion_polynomial_for_weights(r, s, b_pole, d5, d4)
                )
                total += residue / denominator * coefficient(
                    n1,
                    n2 - null_level,
                    pole_c,
                    h1,
                    h2 + null_level,
                )
        return complex(total)

    return {
        (n1, n2): coefficient(n1, n2, c_value, initial_h1, initial_h2)
        for n1 in range(order1 + 1)
        for n2 in range(order2 + 1)
        if max_total_order is None or n1 + n2 <= int(max_total_order)
    }


def sphere_five_point_h_coefficients(
    *,
    external_weights: Sequence[complex],
    internal_weights: Sequence[complex],
    order1: int,
    order2: int | None = None,
    max_total_order: int | None = None,
    central_charge: complex | None = None,
    b: complex | None = None,
    pole_tolerance: float = 1.0e-12,
) -> CoefficientTable:
    """Return coefficients from the CCY sphere-linear ``h``-recursion.

    Exactly one of ``central_charge`` or ``b`` may be supplied.  Supplying
    ``b`` is useful for the explicit ``b=exp(eta)`` regulator near ``c=25``.
    """

    order1, order2 = _validate_orders(order1, order2)
    external, internal = _weights(external_weights, internal_weights)
    d1, d2, d3, d4, d5 = external
    h1_initial, h2_initial = internal
    if central_charge is not None and b is not None:
        raise ValueError("supply central_charge or b, not both")
    if b is None:
        if central_charge is None:
            raise ValueError("central_charge or b is required")
        b_value = central_charge_to_b(complex(central_charge))
    else:
        b_value = complex(b)
    if b_value == 0.0:
        raise ValueError("b must be nonzero")
    pole_tolerance = float(pole_tolerance)
    if pole_tolerance <= 0.0:
        raise ValueError("pole_tolerance must be positive")

    a_initial = h2_initial - h1_initial
    e1_initial = d1 - h1_initial
    e5_initial = d5 - h1_initial

    @lru_cache(maxsize=None)
    def coefficient(
        n1: int,
        n2: int,
        h1: complex,
        a: complex,
        e1: complex,
        e5: complex,
    ) -> complex:
        total = 1.0 + 0.0j if n1 == 0 and n2 == 0 else 0.0 + 0.0j
        for r in range(1, n1 + 1):
            for s in range(1, n1 // r + 1):
                null_level = r * s
                pole_h = degenerate_weight(r, s, b_value)
                denominator = h1 - pole_h
                if abs(denominator) < pole_tolerance:
                    raise ZeroDivisionError(
                        "five-point h-recursion encountered an edge-1 Kac pole"
                    )
                residue = (
                    zamolodchikov_a_rs(r, s, b_value)
                    * fixed_c_fusion_polynomial_for_weights(
                        r, s, b_value, pole_h + e1, d2
                    )
                    * fixed_c_fusion_polynomial_for_weights(
                        r, s, b_value, pole_h + a, d3
                    )
                )
                total += residue / denominator * coefficient(
                    n1 - null_level,
                    n2,
                    pole_h + null_level,
                    a - null_level,
                    e1 - null_level,
                    e5 - null_level,
                )
        for r in range(1, n2 + 1):
            for s in range(1, n2 // r + 1):
                null_level = r * s
                pole_h = degenerate_weight(r, s, b_value)
                denominator = h1 + a - pole_h
                if abs(denominator) < pole_tolerance:
                    raise ZeroDivisionError(
                        "five-point h-recursion encountered an edge-2 Kac pole"
                    )
                residue = (
                    zamolodchikov_a_rs(r, s, b_value)
                    * fixed_c_fusion_polynomial_for_weights(
                        r, s, b_value, pole_h - a, d3
                    )
                    * fixed_c_fusion_polynomial_for_weights(
                        r, s, b_value, pole_h - a + e5, d4
                    )
                )
                total += residue / denominator * coefficient(
                    n1,
                    n2 - null_level,
                    pole_h - a,
                    a + null_level,
                    e1,
                    e5,
                )
        return complex(total)

    return {
        (n1, n2): coefficient(
            n1,
            n2,
            h1_initial,
            a_initial,
            e1_initial,
            e5_initial,
        )
        for n1 in range(order1 + 1)
        for n2 in range(order2 + 1)
        if max_total_order is None or n1 + n2 <= int(max_total_order)
    }


def sphere_five_point_h_c25_limit(
    *,
    external_weights: Sequence[complex],
    internal_weights: Sequence[complex],
    order1: int,
    order2: int | None = None,
    max_total_order: int | None = None,
    regulator_etas: Sequence[float] = (0.16, 0.13, 0.10, 0.075, 0.055),
    polynomial_degree: int = 3,
) -> tuple[CoefficientTable, CoefficientTable]:
    """Return the regulated ``c=25`` h-recursion and an error estimate.

    Since ``c(exp(eta)) = 25 + O(eta**2)``, every coefficient is fitted as a
    polynomial in ``eta**2``.  The error table is the difference between the
    requested fit and the fit with one lower degree.
    """

    order1, order2 = _validate_orders(order1, order2)
    etas = tuple(float(value) for value in regulator_etas)
    if len(etas) < 3 or any(not math.isfinite(value) or value <= 0.0 for value in etas):
        raise ValueError("regulator_etas must contain at least three positive finite values")
    degree = int(polynomial_degree)
    if degree < 1 or degree >= len(etas):
        raise ValueError("polynomial_degree must lie between one and len(regulator_etas)-1")

    samples = [
        sphere_five_point_h_coefficients(
            external_weights=external_weights,
            internal_weights=internal_weights,
            order1=order1,
            order2=order2,
            max_total_order=max_total_order,
            b=cmath.exp(eta),
        )
        for eta in etas
    ]
    keys = tuple(samples[0])
    x_values = np.asarray([eta * eta for eta in etas], dtype=float)
    values: CoefficientTable = {}
    errors: CoefficientTable = {}
    for key in keys:
        y_values = np.asarray([sample[key] for sample in samples], dtype=complex)
        high = np.polynomial.polynomial.polyfit(x_values, y_values, degree)
        low = np.polynomial.polynomial.polyfit(x_values, y_values, degree - 1)
        values[key] = complex(high[0])
        errors[key] = complex(high[0] - low[0])
    return values, errors


def evaluate_sphere_five_point_series(
    q1: complex,
    q2: complex,
    coefficients: Mapping[tuple[int, int], complex],
) -> complex:
    """Evaluate a supplied bivariate descendant coefficient table."""

    q1 = complex(q1)
    q2 = complex(q2)
    return complex(
        sum(
            complex(value) * q1**int(levels[0]) * q2**int(levels[1])
            for levels, value in coefficients.items()
        )
    )


def sphere_five_point_primary_factor(
    z1: complex,
    z2: complex,
    *,
    external_weights: Sequence[complex],
    internal_weights: Sequence[complex],
) -> complex:
    """Return the primary-coordinate power in the stated linear channel."""

    external, internal = _weights(external_weights, internal_weights)
    d1, d2, d3, _, _ = external
    h1, h2 = internal
    return complex(
        complex(z1) ** (h1 - d1 - d2)
        * complex(z2) ** (h2 - d3 - h1)
    )

