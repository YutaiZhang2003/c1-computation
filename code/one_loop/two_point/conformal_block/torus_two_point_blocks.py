#!/usr/bin/env python3
"""Direct and recursive Virasoro torus two-point blocks in two channels.

The descendant sums provide an independent low-level baseline. Both channels
implement the Cho--Collier--Yin fixed-weight central-charge recursion. The
generic necklace block is evaluated in its original sewing variables:

* flat torus coordinate ``z ~ z + 2*pi ~ z + 2*pi*tau``;
* ``q = exp(2*pi*i*tau)``;
* necklace cylinders ``q1 = exp(i*z)``, ``q2 = q/q1``;
* OPE plumbing coordinate ``v = exp(-i*z) - 1``.

Only the descendant series is returned by the coefficient builders.  The
primary propagation and flat-frame factors are inserted by the two block
evaluation functions below.
"""

from __future__ import annotations

from c1_paths import PACKAGE_ROOT, data_root, package_path, module_path

import cmath
import math
from functools import lru_cache

import numpy as np

try:
    from torus_descendant_blocks import (
        gram_matrix,
        rho_descendant_external,
        rho_primary_external,
    )
except ImportError:  # pragma: no cover - package-style execution
    from torus_descendant_blocks import (
        gram_matrix,
        rho_descendant_external,
        rho_primary_external,
    )

try:
    from ccy_genus2_block import (
        b_from_c_rs_h,
        c_rs_from_h,
        fusion_polynomial_for_weights,
        minus_dc_dh_times_a_rs,
    )
except ImportError:  # pragma: no cover - package-style execution
    from ccy_genus2_block import (
        b_from_c_rs_h,
        c_rs_from_h,
        fusion_polynomial_for_weights,
        minus_dc_dh_times_a_rs,
    )


def _validate_order(order: int) -> int:
    order = int(order)
    if order < 0:
        raise ValueError("block orders must be non-negative")
    return order


@lru_cache(maxsize=None)
def _basis_and_inverse_gram(
    h: complex,
    c: complex,
    level: int,
) -> tuple[tuple[tuple[int, ...], ...], np.ndarray]:
    basis, gram = gram_matrix(complex(h), complex(c), int(level))
    return basis, np.linalg.inv(gram)


def necklace_descendant_coefficients(
    c: complex,
    h1: complex,
    h2: complex,
    d1: complex,
    d2: complex,
    order1: int,
    order2: int,
) -> np.ndarray:
    """Return ``a[n1,n2]`` in the necklace descendant block.

    This is equation (3.2) of Cho--Collier--Yin specialized to two
    insertions, evaluated directly from Virasoro Ward identities.
    """
    order1 = _validate_order(order1)
    order2 = _validate_order(order2)
    c = complex(c)
    h1 = complex(h1)
    h2 = complex(h2)
    d1 = complex(d1)
    d2 = complex(d2)
    coefficients = np.zeros((order1 + 1, order2 + 1), dtype=np.complex128)

    for n1 in range(order1 + 1):
        basis1, inverse1 = _basis_and_inverse_gram(h1, c, n1)
        for n2 in range(order2 + 1):
            basis2, inverse2 = _basis_and_inverse_gram(h2, c, n2)
            value = 0.0 + 0.0j
            for a1_index, a1 in enumerate(basis1):
                for b1_index, b1 in enumerate(basis1):
                    inverse_edge1 = inverse1[a1_index, b1_index]
                    for a2_index, a2 in enumerate(basis2):
                        rho1 = rho_primary_external(b1, a2, h1, d1, h2, c)
                        for b2_index, b2 in enumerate(basis2):
                            value += (
                                inverse_edge1
                                * inverse2[a2_index, b2_index]
                                * rho1
                                * rho_primary_external(b2, a1, h2, d2, h1, c)
                            )
            coefficients[n1, n2] = value
    return coefficients


def necklace_c_recursion_coefficients(
    c: complex,
    h1: complex,
    h2: complex,
    d1: complex,
    d2: complex,
    order1: int,
    order2: int,
) -> np.ndarray:
    """Return the native ``q1,q2`` coefficients from fixed-weight c recursion.

    The large-c seed is the cyclic global block times the vacuum oscillator
    character in ``q1*q2``. Each edge supplies its own CCY null-state residues.
    Coincident intermediate poles are resolved by a checked weight limit of
    the same c recursion; no descendant or h-recursion fallback is used.
    """
    from torus_c_recursion_numerics import stable_c_recursion_coefficients

    coefficients, _, _ = stable_c_recursion_coefficients(
        "necklace", central_charge=c, external_weights=(d1, d2),
        internal_weights=(h1, h2), orders=(order1, order2))
    return coefficients


def ope_descendant_coefficients(
    c: complex,
    h_loop: complex,
    h_ope: complex,
    d1: complex,
    d2: complex,
    q_order: int,
    v_order: int,
) -> np.ndarray:
    """Return ``a[n,m]`` in the OPE descendant block.

    This is equation (4.34) of Cho--Collier--Yin, evaluated directly.
    ``n`` is the torus-loop level and ``m`` the OPE level.
    """
    q_order = _validate_order(q_order)
    v_order = _validate_order(v_order)
    c = complex(c)
    h_loop = complex(h_loop)
    h_ope = complex(h_ope)
    d1 = complex(d1)
    d2 = complex(d2)
    coefficients = np.zeros((q_order + 1, v_order + 1), dtype=np.complex128)

    for n in range(q_order + 1):
        loop_basis, inverse_loop = _basis_and_inverse_gram(h_loop, c, n)
        for m in range(v_order + 1):
            ope_basis, inverse_ope = _basis_and_inverse_gram(h_ope, c, m)
            value = 0.0 + 0.0j
            for n_index, n_state in enumerate(loop_basis):
                for m_index, m_state in enumerate(loop_basis):
                    inverse_loop_edge = inverse_loop[n_index, m_index]
                    for p_index, p_state in enumerate(ope_basis):
                        rho_torus = rho_descendant_external(
                            n_state,
                            p_state,
                            m_state,
                            h_loop,
                            h_ope,
                            c,
                        )
                        for q_index, q_state in enumerate(ope_basis):
                            value += (
                                inverse_loop_edge
                                * inverse_ope[p_index, q_index]
                                * rho_torus
                                * rho_primary_external(
                                    q_state,
                                    (),
                                    h_ope,
                                    d1,
                                    d2,
                                    c,
                                )
                            )
            coefficients[n, m] = value
    return coefficients


@lru_cache(maxsize=None)
def _sl2_descendant_norm(weight: complex, level: int) -> complex:
    """Return ``<h|L_1^n L_-1^n|h>`` for the global descendant."""

    value = 1.0 + 0.0j
    for offset in range(int(level)):
        value *= (offset + 1) * (2.0 * complex(weight) + offset)
    return complex(value)


@lru_cache(maxsize=None)
def _vacuum_character_without_lminus1(order: int) -> tuple[int, ...]:
    r"""Coefficients of ``prod_(n>=2) (1-q^n)^(-1)``."""

    order = _validate_order(order)
    coefficients = [0] * (order + 1)
    coefficients[0] = 1
    for oscillator in range(2, order + 1):
        for level in range(oscillator, order + 1):
            coefficients[level] += coefficients[level - oscillator]
    return tuple(coefficients)


@lru_cache(maxsize=None)
def ope_large_c_global_coefficient(
    h_loop: complex,
    h_ope: complex,
    d1: complex,
    d2: complex,
    q_level: int,
    v_level: int,
) -> complex:
    r"""Return one coefficient of the global OPE plumbing block.

    Only the ``L_-1`` state propagates on each internal edge at large
    central charge.  This evaluates the same two three-point tensors as the
    defining descendant sum, divided by their global Gram norms.
    """

    q_level = _validate_order(q_level)
    v_level = _validate_order(v_level)
    h_loop = complex(h_loop)
    h_ope = complex(h_ope)
    d1 = complex(d1)
    d2 = complex(d2)
    loop_state = (1,) * q_level
    ope_state = (1,) * v_level
    numerator = rho_descendant_external(
        loop_state,
        ope_state,
        loop_state,
        h_loop,
        h_ope,
        0.0,
    ) * rho_primary_external(
        ope_state,
        (),
        h_ope,
        d1,
        d2,
        0.0,
    )
    denominator = _sl2_descendant_norm(
        h_loop,
        q_level,
    ) * _sl2_descendant_norm(h_ope, v_level)
    if denominator == 0.0:
        raise ZeroDivisionError("singular global OPE descendant norm")
    return complex(numerator / denominator)


@lru_cache(maxsize=None)
def _ope_large_c_seed_coefficient(
    h_loop: complex,
    h_ope: complex,
    d1: complex,
    d2: complex,
    q_level: int,
    v_level: int,
) -> complex:
    r"""Return the CCY regular term ``U_c`` at one bidegree.

    The regular term is the global two-edge block times the genus-one
    vacuum oscillator seed ``prod_(n>=2)(1-q^n)^(-1)``.
    """

    vacuum = _vacuum_character_without_lminus1(q_level)
    return complex(
        sum(
            vacuum[oscillator_level]
            * ope_large_c_global_coefficient(
                h_loop,
                h_ope,
                d1,
                d2,
                q_level - oscillator_level,
                v_level,
            )
            for oscillator_level in range(q_level + 1)
        )
    )


def _ope_loop_edge_c_residue(
    r: int,
    s: int,
    h_loop: complex,
    h_ope: complex,
) -> complex:
    """CCY residue for the tadpole/torus-loop internal edge."""

    level = int(r) * int(s)
    b_pole = b_from_c_rs_h(int(r), int(s), complex(h_loop))
    return complex(
        minus_dc_dh_times_a_rs(int(r), int(s), complex(h_loop))
        * fusion_polynomial_for_weights(
            int(r),
            int(s),
            b_pole,
            complex(h_ope),
            complex(h_loop) + level,
        )
        * fusion_polynomial_for_weights(
            int(r),
            int(s),
            b_pole,
            complex(h_ope),
            complex(h_loop),
        )
    )


def _ope_fusion_edge_c_residue(
    r: int,
    s: int,
    h_ope: complex,
    h_loop: complex,
    d1: complex,
    d2: complex,
) -> complex:
    """CCY residue for the edge joining the two external primaries."""

    b_pole = b_from_c_rs_h(int(r), int(s), complex(h_ope))
    return complex(
        minus_dc_dh_times_a_rs(int(r), int(s), complex(h_ope))
        * fusion_polynomial_for_weights(
            int(r),
            int(s),
            b_pole,
            complex(h_loop),
            complex(h_loop),
        )
        * fusion_polynomial_for_weights(
            int(r),
            int(s),
            b_pole,
            complex(d1),
            complex(d2),
        )
    )


def ope_c_recursion_coefficients(
    c: complex,
    h_loop: complex,
    h_ope: complex,
    d1: complex,
    d2: complex,
    q_order: int,
    v_order: int,
    *,
    pole_tolerance: float = 1.0e-12,
    collision_regulator: float = 0.001,
) -> np.ndarray:
    r"""Return the OPE descendant series from the CCY ``c``-recursion.

    This is the coefficient form of Cho--Collier--Yin equations (4.35)--
    (4.36).  The returned normalization is identical to
    :func:`ope_descendant_coefficients`: primary powers and flat-cylinder
    factors are not included.
    """

    q_order = _validate_order(q_order)
    v_order = _validate_order(v_order)
    c = complex(c)
    h_loop = complex(h_loop)
    h_ope = complex(h_ope)
    d1 = complex(d1)
    d2 = complex(d2)
    pole_tolerance = float(pole_tolerance)
    if pole_tolerance <= 0.0:
        raise ValueError("pole_tolerance must be positive")
    collision_regulator = float(collision_regulator)
    if collision_regulator < 0.0:
        raise ValueError("collision_regulator must be non-negative")

    @lru_cache(maxsize=None)
    def coefficient(
        q_level: int,
        v_level: int,
        current_c: complex,
        current_h_loop: complex,
        current_h_ope: complex,
    ) -> complex:
        total = _ope_large_c_seed_coefficient(
            current_h_loop,
            current_h_ope,
            d1,
            d2,
            q_level,
            v_level,
        )
        for r in range(2, q_level + 1):
            for s in range(1, q_level // r + 1):
                null_level = r * s
                pole_c = c_rs_from_h(r, s, current_h_loop)
                denominator = current_c - pole_c
                if abs(denominator) < pole_tolerance:
                    raise ZeroDivisionError(
                        "torus OPE c-recursion hit a loop-edge pole "
                        f"c_({r},{s})={pole_c!r}"
                    )
                total += (
                    _ope_loop_edge_c_residue(
                        r,
                        s,
                        current_h_loop,
                        current_h_ope,
                    )
                    / denominator
                    * coefficient(
                        q_level - null_level,
                        v_level,
                        pole_c,
                        current_h_loop + null_level,
                        current_h_ope,
                    )
                )
        for r in range(2, v_level + 1):
            for s in range(1, v_level // r + 1):
                null_level = r * s
                pole_c = c_rs_from_h(r, s, current_h_ope)
                denominator = current_c - pole_c
                if abs(denominator) < pole_tolerance:
                    raise ZeroDivisionError(
                        "torus OPE c-recursion hit a fusion-edge pole "
                        f"c_({r},{s})={pole_c!r}"
                    )
                total += (
                    _ope_fusion_edge_c_residue(
                        r,
                        s,
                        current_h_ope,
                        current_h_loop,
                        d1,
                        d2,
                    )
                    / denominator
                    * coefficient(
                        q_level,
                        v_level - null_level,
                        pole_c,
                        current_h_loop,
                        current_h_ope + null_level,
                    )
                )
        return complex(total)

    try:
        coefficients = np.empty((q_order + 1, v_order + 1), dtype=np.complex128)
        for q_level in range(q_order + 1):
            for v_level in range(v_order + 1):
                coefficients[q_level, v_level] = coefficient(
                    q_level,
                    v_level,
                    c,
                    h_loop,
                    h_ope,
                )
        return coefficients
    except ZeroDivisionError:
        if collision_regulator == 0.0:
            raise

    # Equal or specially related internal weights can make distinct CCY
    # simple poles coincide at intermediate recursive states.  The complete
    # block is regular although the termwise recursion is not.  Approach the
    # requested weights symmetrically from generic points, then eliminate the
    # leading even regulator error.
    symmetric_values = []
    for scale in (1.0, 0.5):
        delta = collision_regulator * scale
        plus = ope_c_recursion_coefficients(
            c,
            h_loop + delta,
            h_ope - delta,
            d1,
            d2,
            q_order,
            v_order,
            pole_tolerance=pole_tolerance,
            collision_regulator=0.0,
        )
        minus = ope_c_recursion_coefficients(
            c,
            h_loop - delta,
            h_ope + delta,
            d1,
            d2,
            q_order,
            v_order,
            pole_tolerance=pole_tolerance,
            collision_regulator=0.0,
        )
        symmetric_values.append(0.5 * (plus + minus))
    return np.asarray(
        (4.0 * symmetric_values[1] - symmetric_values[0]) / 3.0,
        dtype=np.complex128,
    )


def _truncated_product(left: np.ndarray, right: np.ndarray, order: int) -> np.ndarray:
    return np.convolve(left, right)[: order + 1]


@lru_cache(maxsize=None)
def modular_lambda_series(order: int) -> np.ndarray:
    """Return coefficients of ``lambda(q)`` through ``q**order``.

    The nome convention is ``q=exp(i*pi*tau)``.  Thus
    ``lambda(q)=theta_2(q)^4/theta_3(q)^4=16*q-128*q^2+...``.
    """
    order = _validate_order(order)
    if order == 0:
        return np.zeros(1, dtype=np.complex128)

    theta2_reduced = np.zeros(order + 1, dtype=np.complex128)
    for n in range(order + 1):
        exponent = n * (n + 1)
        if exponent > order:
            break
        theta2_reduced[exponent] += 1.0
    theta3 = np.zeros(order + 1, dtype=np.complex128)
    theta3[0] = 1.0
    for n in range(1, order + 1):
        exponent = n * n
        if exponent > order:
            break
        theta3[exponent] += 2.0

    numerator_reduced = np.array([1.0 + 0.0j])
    denominator = np.array([1.0 + 0.0j])
    for _ in range(4):
        numerator_reduced = _truncated_product(numerator_reduced, theta2_reduced, order)
        denominator = _truncated_product(denominator, theta3, order)

    inverse_denominator = np.zeros(order + 1, dtype=np.complex128)
    inverse_denominator[0] = 1.0 / denominator[0]
    for n in range(1, order + 1):
        inverse_denominator[n] = -sum(
            denominator[k] * inverse_denominator[n - k]
            for k in range(1, min(n, len(denominator) - 1) + 1)
        ) / denominator[0]
    reduced_ratio = _truncated_product(numerator_reduced, inverse_denominator, order)
    result = np.zeros(order + 1, dtype=np.complex128)
    result[1:] = 16.0 * reduced_ratio[:order]
    return result


def power_composition_matrix(series: np.ndarray, input_order: int, output_order: int) -> np.ndarray:
    """Return ``T[n,k]=[x^k] series(x)^n``."""
    input_order = _validate_order(input_order)
    output_order = _validate_order(output_order)
    series = np.asarray(series, dtype=np.complex128)[: output_order + 1]
    transform = np.zeros((input_order + 1, output_order + 1), dtype=np.complex128)
    transform[0, 0] = 1.0
    for n in range(1, input_order + 1):
        transform[n] = _truncated_product(transform[n - 1], series, output_order)
    return transform


def ope_coefficients_in_z(coefficients: np.ndarray, z_order: int | None = None) -> np.ndarray:
    """Re-expand the OPE descendant series using ``v=exp(-i*z)-1``."""
    coefficients = np.asarray(coefficients, dtype=np.complex128)
    v_order = coefficients.shape[1] - 1
    if z_order is None:
        z_order = v_order
    z_order = _validate_order(z_order)
    v_series = np.zeros(z_order + 1, dtype=np.complex128)
    for n in range(1, z_order + 1):
        v_series[n] = (-1.0j) ** n / math.factorial(n)
    transform = power_composition_matrix(v_series, v_order, z_order)
    return coefficients @ transform


def elliptic_nome(cross_ratio: complex) -> complex:
    """Return ``E(x)=exp[-pi K(1-x)/K(x)]`` on principal branches."""
    import scipy.special  # local import keeps the coefficient baseline lightweight

    x = complex(cross_ratio)
    if abs(x) < 1.0e-200:
        # lambda(q)=16q+O(q^2).  This also avoids the logarithmic singularity
        # of hyp2f1 at 1-x when x underflows in the deep torus cusp.
        return x / 16.0
    k_x = scipy.special.hyp2f1(0.5, 0.5, 1.0, x)
    k_one_minus_x = scipy.special.hyp2f1(0.5, 0.5, 1.0, 1.0 - x)
    return cmath.exp(-math.pi * k_one_minus_x / k_x)


def evaluate_bivariate(coefficients: np.ndarray, x: complex, y: complex) -> complex:
    """Evaluate a coefficient matrix by nested Horner summation."""
    coefficients = np.asarray(coefficients, dtype=np.complex128)
    value = 0.0 + 0.0j
    for row in coefficients[::-1]:
        row_value = 0.0 + 0.0j
        for coefficient in row[::-1]:
            row_value = row_value * y + coefficient
        value = value * x + row_value
    return complex(value)


def necklace_flat_block(
    coefficients_in_q: np.ndarray,
    *,
    c: complex,
    h1: complex,
    h2: complex,
    z: complex,
    tau: complex,
) -> complex:
    """Evaluate the full chiral necklace block from native sewing coefficients."""
    log_q1 = 1.0j * complex(z)
    log_q2 = 1.0j * (2.0 * math.pi * complex(tau) - complex(z))
    q1 = cmath.exp(log_q1)
    q2 = cmath.exp(log_q2)
    descendant = evaluate_bivariate(
        coefficients_in_q,
        q1,
        q2,
    )
    primary = cmath.exp(
        (complex(h1) - complex(c) / 24.0) * log_q1
        + (complex(h2) - complex(c) / 24.0) * log_q2
    )
    return primary * descendant


def ope_flat_block(
    coefficients_in_z: np.ndarray,
    *,
    c: complex,
    h_loop: complex,
    h_ope: complex,
    d1: complex,
    d2: complex,
    z: complex,
    tau: complex,
) -> complex:
    """Evaluate the full chiral OPE block in the flat torus frame.

    The implementation is currently restricted to equal external weights,
    which is the case needed for the c=1 reflection amplitude.  The factor
    ``(2 sin(z/2))^(-2d)`` converts the plumbing-disc frame to the flat
    cylinder frame.
    """
    if abs(complex(d1) - complex(d2)) > 1.0e-12:
        raise ValueError("the flat-frame helper currently expects d1=d2")
    z = complex(z)
    tau = complex(tau)
    q = cmath.exp(2.0j * math.pi * tau)
    v = cmath.exp(-1.0j * z) - 1.0
    descendant = evaluate_bivariate(coefficients_in_z, q, z)
    primary = (
        cmath.exp((complex(h_loop) - complex(c) / 24.0) * (2.0j * math.pi * tau))
        * cmath.exp(complex(h_ope) * cmath.log(v))
        * cmath.exp(-2.0 * complex(d1) * cmath.log(2.0 * cmath.sin(z / 2.0)))
    )
    return primary * descendant
