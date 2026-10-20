#!/usr/bin/env python3
"""Cho-Collier-Yin genus-two Virasoro block in the plumbing frame.

The block implemented here is the genus-two, no-puncture block built in the
theta-graph frame by gluing two 2-holed discs.  In the notation of Cho,
Collier, and Yin,

    F = sum q1^|A| q2^|C| q3^|E|
        G_h1^{AB} G_h2^{CD} G_h3^{EF}
        rho(L_-A h1, L_-C h2, L_-E h3)
        rho(L_-B h1, L_-D h2, L_-F h3).

The public evaluator uses their central-charge recursion.  The large-c regular
part is written as a global SL(2) block times the c=infinity vacuum block.  The
global block can either be total-degree truncated for coefficient checks or
resummed along one edge into a Gauss hypergeometric function.  A stable
two-dimensional recurrence generates the normalized coefficients on the two
remaining edges, whose total-level shells are converged adaptively.  The
vacuum block is approximated, when requested, by the theta-graph Schottky
primitive-class product truncated by word length and oscillator level.

As in CCY's generic plumbing construction, this block contains the descendant
powers q^level only; the separated primary propagator prod_i q_i^h_i is applied
by the Liouville sewing wrapper.  The low-level edge order follows the
three-point tensor slots ``(infinity, one, zero)``; geometry-facing callers
must convert from the theta chart's ``(zero, one, infinity)`` order.
"""

from __future__ import annotations

from c1_paths import PACKAGE_ROOT, data_root, package_path, module_path

import argparse
import cmath
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Iterator, Sequence

try:
    from genus2_vacuum_blocks import oscillator_log_factor, schottky_vacuum_block
    from plumbing_algorithms import (
        generators_for_theta,
        inverse_letter,
        reduced_words,
        schottky_theta_period_matrix_cross_ratio,
        theta_cusp_surviving_multipliers,
        word_mobius,
    )
    from virasoro_blocks import (
        fusion_polynomial,
        momentum_from_weight,
        zamolodchikov_a_rs,
    )
except ImportError:  # pragma: no cover - supports package-style execution
    from genus2_vacuum_blocks import oscillator_log_factor, schottky_vacuum_block
    from plumbing_algorithms import (
        generators_for_theta,
        inverse_letter,
        reduced_words,
        schottky_theta_period_matrix_cross_ratio,
        theta_cusp_surviving_multipliers,
        word_mobius,
    )
    from virasoro_blocks import (
        fusion_polynomial,
        momentum_from_weight,
        zamolodchikov_a_rs,
    )


def parse_complex(value: str) -> complex:
    return complex(value.replace("i", "j"))


def format_complex(value: complex) -> str:
    return f"{value.real:+.12e}{value.imag:+.12e}j"


def _as_complex(value: complex | float | int) -> complex:
    return complex(value)


def _validate_order(order: int) -> int:
    order = int(order)
    if order < 0:
        raise ValueError("order must be non-negative")
    return order


def rising_pochhammer(value: complex, order: int) -> complex:
    """Return value (value+1) ... (value+order-1)."""
    order = _validate_order(order)
    out = 1.0 + 0.0j
    value = _as_complex(value)
    for n in range(order):
        out *= value + n
    return out


def falling_pochhammer(value: complex, order: int) -> complex:
    """Return value (value-1) ... (value-order+1)."""
    order = _validate_order(order)
    out = 1.0 + 0.0j
    value = _as_complex(value)
    for n in range(order):
        out *= value - n
    return out


@lru_cache(maxsize=32768)
def gauss_hypergeometric_2f1(
    a: complex,
    b: complex,
    c: complex,
    z: complex,
    *,
    tolerance: float = 1.0e-15,
    max_terms: int = 10000,
) -> complex:
    r"""Return ``2F1(a,b;c;z)`` by its adaptively converged Gauss series.

    Plumbing parameters lie in ``|z| < 1``, where the defining series is the
    fastest and least ambiguous numerical representation.  Unlike the block
    recursion order, ``max_terms`` is only a safety cap: summation stops when
    three successive terms meet ``tolerance``.  Keeping this evaluator local
    avoids turning SciPy or arbitrary precision arithmetic into a dependency
    of every genus-two integrand call.
    """

    a = _as_complex(a)
    b = _as_complex(b)
    c = _as_complex(c)
    z = _as_complex(z)
    tolerance = float(tolerance)
    max_terms = int(max_terms)
    if not math.isfinite(tolerance) or not 0.0 < tolerance < 1.0:
        raise ValueError("hypergeometric tolerance must lie strictly between zero and one")
    if max_terms <= 0:
        raise ValueError("hypergeometric max_terms must be positive")
    if abs(z) >= 1.0:
        raise ValueError("the plumbing-frame hypergeometric series requires |z| < 1")
    if z == 0.0:
        return 1.0 + 0.0j

    total = 1.0 + 0.0j
    term = 1.0 + 0.0j
    converged_terms = 0
    for level in range(1, max_terms + 1):
        denominator = (c + level - 1) * level
        if denominator == 0.0:
            raise ZeroDivisionError("2F1 has a singular lower parameter")
        term *= (a + level - 1) * (b + level - 1) * z / denominator
        total += term
        if not _is_finite_complex(total) or not _is_finite_complex(term):
            raise ArithmeticError("2F1 series produced a non-finite value")
        if term == 0.0:
            return complex(total)
        if abs(term) <= tolerance * max(1.0, abs(total)):
            converged_terms += 1
            if converged_terms >= 3:
                return complex(total)
        else:
            converged_terms = 0
    raise ArithmeticError(
        f"2F1 series did not converge in {max_terms} terms at z={z!r}"
    )


def sl2_descendant_norm(weight: complex, level: int) -> complex:
    """Return <h| L_1^n L_-1^n |h> = n! (2h)_n."""
    level = _validate_order(level)
    return math.factorial(level) * rising_pochhammer(2.0 * _as_complex(weight), level)


def rho_lminus1_two_edge(i_level: int, k_level: int, h1: complex, h2: complex, h3: complex) -> complex:
    r"""Return s_{ik}=rho(L_-1^i h1, h2, L_-1^k h3).

    This is the closed form quoted by Cho-Collier-Yin for the global block.
    """
    i_level = _validate_order(i_level)
    k_level = _validate_order(k_level)
    h1 = _as_complex(h1)
    h2 = _as_complex(h2)
    h3 = _as_complex(h3)
    total = 0.0 + 0.0j
    for p_level in range(min(i_level, k_level) + 1):
        total += (
            math.factorial(i_level)
            / (math.factorial(p_level) * math.factorial(i_level - p_level))
            * falling_pochhammer(2.0 * h3 + k_level - 1.0, p_level)
            * falling_pochhammer(k_level, p_level)
            * rising_pochhammer(h3 + h2 - h1, k_level - p_level)
            * rising_pochhammer(h1 + h2 - h3 + p_level - k_level, i_level - p_level)
        )
    return total


def normalized_rho_lminus1_two_edge(
    i_level: int,
    k_level: int,
    h1: complex,
    h2: complex,
    h3: complex,
) -> complex:
    r"""Return ``rho(i,0,k)/sqrt(norm_i norm_k)`` without overflow.

    The raw three-point coefficient and both descendant norms grow
    factorially even though their normalized ratio remains moderate.  The
    resummed global block may need outer levels well above 64 at large
    plumbing multipliers, where forming those quantities separately both
    overflows and amplifies cancellation.  We initialize the ``p=0`` term in
    logarithmic form and obtain the remaining terms by their exact adjacent
    ratio before summing real and imaginary parts with ``math.fsum``.
    """

    i_level = _validate_order(i_level)
    k_level = _validate_order(k_level)
    h1 = _as_complex(h1)
    h2 = _as_complex(h2)
    h3 = _as_complex(h3)
    a = h3 + h2 - h1
    b = h1 + h2 - h3

    def log_product(start: complex, count: int) -> complex:
        total = 0.0 + 0.0j
        for offset in range(count):
            factor = start + offset
            if factor == 0.0:
                raise ZeroDivisionError(
                    "normalized global-block coefficient encountered an exact zero"
                )
            total += cmath.log(factor)
        return total

    log_t0 = (
        log_product(a, k_level)
        + log_product(b - k_level, i_level)
        - 0.5
        * (
            math.lgamma(i_level + 1.0)
            + log_product(2.0 * h1, i_level)
            + math.lgamma(k_level + 1.0)
            + log_product(2.0 * h3, k_level)
        )
    )
    term = cmath.exp(log_t0)
    terms = [term]
    for p_level in range(min(i_level, k_level)):
        denominator = (
            (p_level + 1.0)
            * (a + k_level - p_level - 1.0)
            * (b + p_level - k_level)
        )
        if denominator == 0.0:
            # Exact degenerate weights require taking the finite combined
            # limit rather than dividing the adjacent-term ratio by zero.
            # The Liouville continuum does not hit these isolated values, so
            # fail loudly instead of silently returning a corrupted seed.
            raise ZeroDivisionError(
                "normalized global-block adjacent-term ratio is singular"
            )
        term *= (
            (i_level - p_level)
            * (2.0 * h3 + k_level - 1.0 - p_level)
            * (k_level - p_level)
            / denominator
        )
        terms.append(term)
    summed = complex(
        math.fsum(value.real for value in terms),
        math.fsum(value.imag for value in terms),
    )
    absolute_sum = math.fsum(abs(value) for value in terms)
    condition = absolute_sum / max(abs(summed), 1.0e-300)
    estimated_relative_error = condition * math.ulp(1.0)
    if (
        not _is_finite_complex(summed)
        or not math.isfinite(condition)
        or estimated_relative_error > 1.0e-13
    ):
        return _normalized_rho_lminus1_two_edge_multiprecision(
            i_level,
            k_level,
            h1,
            h2,
            h3,
            cancellation_condition=condition,
        )
    return summed


def normalized_rho_lminus1_two_edge_table(
    max_order: int,
    h1: complex,
    h2: complex,
    h3: complex,
) -> tuple[tuple[complex, ...], ...]:
    r"""Generate all normalized two-edge coefficients through ``max_order``.

    Write

    ``b_ik = rho(L_-1^i h1, h2, L_-1^k h3)``
    ``       / sqrt(i! (2 h1)_i k! (2 h3)_k)``.

    The exponential generating function of the unnormalized coefficients is

    ``(1-w)^(h1-h2-h3) (1-u)^(h3-h1-h2)``
    ``    * (1-u w)^(h2-h1-h3)``.

    Its first-order differential equation in ``u`` gives a three-neighbour
    recurrence for ``b_ik``.  Filling the rectangular table therefore costs
    O(max_order^2), rather than evaluating the finite descendant sum at every
    lattice point.  Rows are indexed by ``i`` and columns by ``k``.

    The production Liouville path has positive real weights, for which all
    normalization square roots are positive.  Generic complex weights use the
    principal square-root continuation anchored at ``b_00 = 1``.
    """

    max_order = _validate_order(max_order)
    h1 = _as_complex(h1)
    h2 = _as_complex(h2)
    h3 = _as_complex(h3)
    exponent_w = h1 - h2 - h3
    exponent_u = h3 - h1 - h2
    exponent_uw = h2 - h1 - h3

    table = [
        [0.0 + 0.0j for _ in range(max_order + 1)]
        for _ in range(max_order + 1)
    ]
    table[0][0] = 1.0 + 0.0j

    # The i=0 boundary is the normalized coefficient sequence of
    # (1-w)^exponent_w.
    for k_level in range(max_order):
        denominator = cmath.sqrt(
            (k_level + 1.0) * (2.0 * h3 + k_level)
        )
        if denominator == 0.0:
            raise ZeroDivisionError(
                "normalized two-edge recurrence has a singular h3 boundary"
            )
        table[0][k_level + 1] = (
            (k_level - exponent_w)
            * table[0][k_level]
            / denominator
        )

    for k_level in range(max_order + 1):
        for i_level in range(max_order):
            rhs = (i_level - exponent_u) * table[i_level][k_level]
            if k_level > 0:
                h3_denominator = 2.0 * h3 + k_level - 1.0
                if h3_denominator == 0.0:
                    raise ZeroDivisionError(
                        "normalized two-edge recurrence has a singular h3 factor"
                    )
                rhs += (
                    (i_level - exponent_uw)
                    * cmath.sqrt(k_level / h3_denominator)
                    * table[i_level][k_level - 1]
                )
                if i_level > 0:
                    h1_denominator = 2.0 * h1 + i_level - 1.0
                    if h1_denominator == 0.0:
                        raise ZeroDivisionError(
                            "normalized two-edge recurrence has a singular h1 factor"
                        )
                    rhs += (
                        (exponent_u + exponent_uw - i_level + 1.0)
                        * cmath.sqrt(
                            i_level
                            * k_level
                            / (h1_denominator * h3_denominator)
                        )
                        * table[i_level - 1][k_level - 1]
                    )

            denominator = cmath.sqrt(
                (i_level + 1.0) * (2.0 * h1 + i_level)
            )
            if denominator == 0.0:
                raise ZeroDivisionError(
                    "normalized two-edge recurrence has a singular h1 boundary"
                )
            value = rhs / denominator
            if not _is_finite_complex(value):
                raise ArithmeticError(
                    "normalized two-edge recurrence produced a non-finite value"
                )
            table[i_level + 1][k_level] = complex(value)

    return tuple(tuple(row) for row in table)


def normalized_rho_lminus1_two_edge_shells(
    max_outer_order: int,
    h1: complex,
    h2: complex,
    h3: complex,
) -> Iterator[tuple[complex, ...]]:
    r"""Yield normalized ``b_ik`` coefficients one total-level shell at a time.

    Shell ``n`` is returned in increasing ``i`` order as

    ``(b_0n, b_1,n-1, ..., b_n0)``.

    This is the same three-neighbour recurrence used by
    :func:`normalized_rho_lminus1_two_edge_table`, but it only materializes
    shells that the adaptive global-block sum actually requests.  In
    particular, convergence at shell ``n`` costs ``O(n^2)`` storage and work,
    independent of a much larger safety cap ``max_outer_order``.
    """

    maximum = _validate_order(max_outer_order)
    h1 = _as_complex(h1)
    h2 = _as_complex(h2)
    h3 = _as_complex(h3)
    exponent_w = h1 - h2 - h3
    exponent_u = h3 - h1 - h2
    exponent_uw = h2 - h1 - h3

    # Row i contains b_{i,k} for every k reached so far.  At shell n each
    # target row receives exactly its next k entry, so no max-cap rectangle is
    # allocated up front.
    rows: list[list[complex]] = [[1.0 + 0.0j]]
    yield (1.0 + 0.0j,)

    for outer_level in range(1, maximum + 1):
        previous_k = outer_level - 1
        boundary_denominator = cmath.sqrt(
            outer_level * (2.0 * h3 + previous_k)
        )
        if boundary_denominator == 0.0:
            raise ZeroDivisionError(
                "normalized two-edge recurrence has a singular h3 boundary"
            )
        rows[0].append(
            (previous_k - exponent_w)
            * rows[0][previous_k]
            / boundary_denominator
        )

        for i_level in range(outer_level):
            k_level = outer_level - 1 - i_level
            rhs = (i_level - exponent_u) * rows[i_level][k_level]
            if k_level > 0:
                h3_denominator = 2.0 * h3 + k_level - 1.0
                if h3_denominator == 0.0:
                    raise ZeroDivisionError(
                        "normalized two-edge recurrence has a singular h3 factor"
                    )
                rhs += (
                    (i_level - exponent_uw)
                    * cmath.sqrt(k_level / h3_denominator)
                    * rows[i_level][k_level - 1]
                )
                if i_level > 0:
                    h1_denominator = 2.0 * h1 + i_level - 1.0
                    if h1_denominator == 0.0:
                        raise ZeroDivisionError(
                            "normalized two-edge recurrence has a singular h1 factor"
                        )
                    rhs += (
                        (exponent_u + exponent_uw - i_level + 1.0)
                        * cmath.sqrt(
                            i_level
                            * k_level
                            / (h1_denominator * h3_denominator)
                        )
                        * rows[i_level - 1][k_level - 1]
                    )

            denominator = cmath.sqrt(
                (i_level + 1.0) * (2.0 * h1 + i_level)
            )
            if denominator == 0.0:
                raise ZeroDivisionError(
                    "normalized two-edge recurrence has a singular h1 boundary"
                )
            value = complex(rhs / denominator)
            if not _is_finite_complex(value):
                raise ArithmeticError(
                    "normalized two-edge recurrence produced a non-finite value"
                )
            target_row = i_level + 1
            if target_row == len(rows):
                rows.append([])
            if len(rows[target_row]) != k_level:
                raise AssertionError("lazy two-edge recurrence lost its triangular order")
            rows[target_row].append(value)

        yield tuple(
            rows[i_level][outer_level - i_level]
            for i_level in range(outer_level + 1)
        )


def _normalized_rho_lminus1_two_edge_multiprecision(
    i_level: int,
    k_level: int,
    h1: complex,
    h2: complex,
    h3: complex,
    *,
    cancellation_condition: float,
) -> complex:
    """Recover a cancellation-dominated normalized coefficient with mpmath."""

    try:
        import mpmath as mp
    except ImportError as exc:  # pragma: no cover - production environment has mpmath
        raise ImportError(
            "high-level theta global-block resummation requires mpmath when "
            "the normalized descendant coefficient is cancellation dominated"
        ) from exc

    if math.isfinite(cancellation_condition) and cancellation_condition > 1.0:
        lost_digits = max(0, math.ceil(math.log10(cancellation_condition)))
    else:
        lost_digits = 80
    dps = max(50, min(200, lost_digits + 30))

    def mp_complex(value: complex):
        value = complex(value)
        return mp.mpc(value.real, value.imag)

    with mp.workdps(dps):
        mh1 = mp_complex(h1)
        mh2 = mp_complex(h2)
        mh3 = mp_complex(h3)
        a = mh3 + mh2 - mh1
        b = mh1 + mh2 - mh3
        denominator = mp.sqrt(
            mp.factorial(i_level)
            * mp.rf(2 * mh1, i_level)
            * mp.factorial(k_level)
            * mp.rf(2 * mh3, k_level)
        )
        term = mp.rf(a, k_level) * mp.rf(b - k_level, i_level) / denominator
        total = term
        for p_level in range(min(i_level, k_level)):
            ratio_denominator = (
                (p_level + 1)
                * (a + k_level - p_level - 1)
                * (b + p_level - k_level)
            )
            if ratio_denominator == 0:
                raise ZeroDivisionError(
                    "multiprecision normalized coefficient encountered a singular ratio"
                )
            term *= (
                (i_level - p_level)
                * (2 * mh3 + k_level - 1 - p_level)
                * (k_level - p_level)
                / ratio_denominator
            )
            total += term
        return complex(total)


def rho_lminus1_triple(
    i_level: int,
    j_level: int,
    k_level: int,
    h1: complex,
    h2: complex,
    h3: complex,
) -> complex:
    r"""Return rho(L_-1^i h1, L_-1^j h2, L_-1^k h3)."""
    i_level = _validate_order(i_level)
    j_level = _validate_order(j_level)
    k_level = _validate_order(k_level)
    prefactor = rising_pochhammer(
        _as_complex(h1) + i_level - _as_complex(h2) - j_level + 1.0 - _as_complex(h3) - k_level,
        j_level,
    )
    return prefactor * rho_lminus1_two_edge(i_level, k_level, h1, h2, h3)


def genus2_global_sl2_block(
    h1: complex,
    h2: complex,
    h3: complex,
    q1: complex,
    q2: complex,
    q3: complex,
    order: int,
) -> complex:
    """Return the total-degree truncated genus-two global SL(2) block."""
    order = _validate_order(order)
    h1 = _as_complex(h1)
    h2 = _as_complex(h2)
    h3 = _as_complex(h3)
    q1 = _as_complex(q1)
    q2 = _as_complex(q2)
    q3 = _as_complex(q3)
    total = 0.0 + 0.0j
    for i_level in range(order + 1):
        norm1 = sl2_descendant_norm(h1, i_level)
        for j_level in range(order + 1 - i_level):
            norm2 = sl2_descendant_norm(h2, j_level)
            for k_level in range(order + 1 - i_level - j_level):
                rho = rho_lminus1_triple(i_level, j_level, k_level, h1, h2, h3)
                total += (
                    (q1**i_level)
                    * (q2**j_level)
                    * (q3**k_level)
                    * rho
                    * rho
                    / (norm1 * norm2 * sl2_descendant_norm(h3, k_level))
                )
    return total


@dataclass(frozen=True)
class ResummedGlobalBlockDiagnostics:
    """Convergence record for the direct theta global-block shell sum."""

    value: complex
    converged: bool
    outer_order_reached: int
    shell_norms: tuple[float, ...]
    partial_sums: tuple[complex, ...]


def genus2_global_sl2_block_resummed_diagnostics(
    h1: complex,
    h2: complex,
    h3: complex,
    q1: complex,
    q2: complex,
    q3: complex,
    *,
    tolerance: float = 1.0e-13,
    max_outer_order: int = 64,
) -> ResummedGlobalBlockDiagnostics:
    r"""Evaluate the direct theta global-block formula and retain its shells.

    For fixed levels ``i,k`` on the infinity and zero slots, the complete
    one-slot descendant tower is

    ``2F1(a_ik, a_ik; 2 h2; q2)``,

    where ``a_ik = h2 + h3 + k - h1 - i``.  Thus only the outer ``i,k``
    shells require adaptive summation.  This decouples the accuracy of the
    large-c regular term from the finite depth of the Virasoro pole recursion.
    """

    h1 = _as_complex(h1)
    h2 = _as_complex(h2)
    h3 = _as_complex(h3)
    q1 = _as_complex(q1)
    q2 = _as_complex(q2)
    q3 = _as_complex(q3)
    tolerance = float(tolerance)
    max_outer_order = _validate_order(max_outer_order)
    if not math.isfinite(tolerance) or not 0.0 < tolerance < 1.0:
        raise ValueError("global-block tolerance must lie strictly between zero and one")
    if max_outer_order < 3:
        raise ValueError("resummed theta global block needs max_outer_order >= 3")

    total = 0.0 + 0.0j
    converged_shells = 0
    shell_norms: list[float] = []
    partial_sums: list[complex] = []
    hyper_tolerance = min(1.0e-15, 0.1 * tolerance)
    edge1_powers = tuple(q1**level for level in range(max_outer_order + 1))
    edge3_powers = tuple(q3**level for level in range(max_outer_order + 1))
    normalized_coefficient_shells = normalized_rho_lminus1_two_edge_shells(
        max_outer_order,
        h1,
        h2,
        h3,
    )
    hypergeometric_by_level_difference: dict[int, complex] = {}
    for outer_level, normalized_shell in enumerate(normalized_coefficient_shells):
        shell = 0.0 + 0.0j
        shell_norm = 0.0
        for i_level in range(outer_level + 1):
            k_level = outer_level - i_level
            edge_factor = edge1_powers[i_level] * edge3_powers[k_level]
            if edge_factor == 0.0:
                continue
            normalized_rho = normalized_shell[i_level]
            level_difference = i_level - k_level
            hypergeometric = hypergeometric_by_level_difference.get(level_difference)
            if hypergeometric is None:
                parameter = h2 + h3 - h1 - level_difference
                hypergeometric = gauss_hypergeometric_2f1(
                    parameter,
                    parameter,
                    2.0 * h2,
                    q2,
                    tolerance=hyper_tolerance,
                )
                hypergeometric_by_level_difference[level_difference] = hypergeometric
            term = (
                edge_factor
                * normalized_rho
                * normalized_rho
                * hypergeometric
            )
            shell += term
            shell_norm += abs(term)
        total += shell
        shell_norms.append(float(shell_norm))
        partial_sums.append(complex(total))
        if shell_norm <= tolerance * max(1.0, abs(total)):
            converged_shells += 1
            if converged_shells >= 3:
                return ResummedGlobalBlockDiagnostics(
                    value=complex(total),
                    converged=True,
                    outer_order_reached=outer_level,
                    shell_norms=tuple(shell_norms),
                    partial_sums=tuple(partial_sums),
                )
        else:
            converged_shells = 0

    return ResummedGlobalBlockDiagnostics(
        value=complex(total),
        converged=False,
        outer_order_reached=max_outer_order,
        shell_norms=tuple(shell_norms),
        partial_sums=tuple(partial_sums),
    )


def genus2_global_sl2_block_resummed(
    h1: complex,
    h2: complex,
    h3: complex,
    q1: complex,
    q2: complex,
    q3: complex,
    *,
    tolerance: float = 1.0e-13,
    max_outer_order: int = 64,
) -> complex:
    r"""Return the direct, partially resummed genus-two theta global block."""

    diagnostics = genus2_global_sl2_block_resummed_diagnostics(
        h1,
        h2,
        h3,
        q1,
        q2,
        q3,
        tolerance=tolerance,
        max_outer_order=max_outer_order,
    )
    if diagnostics.converged:
        return diagnostics.value
    raise ArithmeticError(
        "resummed theta global block did not converge by outer order "
        f"{max_outer_order}; last shell norm={diagnostics.shell_norms[-1]:.6e}"
    )


def _matrix_eigenvalue_ratio(a: complex, b: complex, c: complex, d: complex) -> complex:
    trace = a + d
    determinant = a * d - b * c
    discriminant = trace * trace - 4.0 * determinant
    root = cmath.sqrt(discriminant)
    lam_plus = 0.5 * (trace + root)
    lam_minus = 0.5 * (trace - root)
    large = lam_plus if abs(lam_plus) >= abs(lam_minus) else lam_minus
    small = lam_minus if abs(lam_plus) >= abs(lam_minus) else lam_plus
    if abs(large) == 0:
        raise ZeroDivisionError("Mobius word has two zero eigenvalues")
    if abs(small) == 0:
        return 0.0 + 0.0j
    return small / large


def _inverse_word(word: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(inverse_letter(letter) for letter in reversed(word))


def _cyclic_rotations(word: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    if not word:
        return (word,)
    return tuple(word[idx:] + word[:idx] for idx in range(len(word)))


def _is_power_word(word: tuple[int, ...]) -> bool:
    length = len(word)
    for period in range(1, length):
        if length % period == 0 and word == word[:period] * (length // period):
            return True
    return False


def primitive_conjugacy_words(max_word_len: int) -> tuple[tuple[int, ...], ...]:
    """Return canonical primitive conjugacy classes for a rank-two Schottky group."""
    max_word_len = _validate_order(max_word_len)
    seen: set[tuple[int, ...]] = set()
    classes: list[tuple[int, ...]] = []
    for word in reduced_words(2, max_word_len):
        if not word:
            continue
        if word[0] == inverse_letter(word[-1]):
            continue
        if _is_power_word(word):
            continue
        orbit = _cyclic_rotations(word) + _cyclic_rotations(_inverse_word(word))
        key = min(orbit)
        if key in seen:
            continue
        seen.add(key)
        classes.append(key)
    return tuple(classes)


@lru_cache(maxsize=None)
def _primitive_conjugacy_words_cached(max_word_len: int) -> tuple[tuple[int, ...], ...]:
    return primitive_conjugacy_words(max_word_len)


@lru_cache(maxsize=128)
def genus2_vacuum_seed_schottky(
    q_zero: complex,
    q_one: complex,
    q_infty: complex,
    *,
    max_word_len: int = 3,
    oscillator_level_max: int = 12,
    word_tail_tolerance: float | None = None,
    minimum_word_length: int = 5,
) -> complex:
    r"""Approximate the theta-frame product over unoriented primitive classes.

    CCY write exponent ``-1/2`` while summing oriented conjugacy classes.  The
    primitive-word helper identifies each word with its inverse, so every
    retained class carries exponent ``-1``.
    """
    max_word_len = _validate_order(max_word_len)
    oscillator_level_max = _validate_order(oscillator_level_max)
    if oscillator_level_max < 2 or max_word_len == 0:
        return 1.0 + 0.0j

    surviving = theta_cusp_surviving_multipliers(q_zero, q_one, q_infty)
    if surviving is not None:
        log_value = 0.0 + 0.0j
        for multiplier in surviving:
            contribution, _ = oscillator_log_factor(
                multiplier,
                max_mode=oscillator_level_max,
                tolerance=1.0e-14,
            )
            log_value += contribution
        return cmath.exp(log_value)

    result = schottky_vacuum_block(
        generators_for_theta(q_zero, q_one, q_infty),
        max_word_length=max_word_len,
        max_mode=oscillator_level_max,
        word_tail_tolerance=word_tail_tolerance,
        minimum_word_length=minimum_word_length,
        channel="theta",
        q_values=(complex(q_zero), complex(q_one), complex(q_infty)),
    )
    if word_tail_tolerance is not None and (
        result.primitive_word_tail_estimate is None
        or result.primitive_word_tail_estimate > float(word_tail_tolerance)
    ):
        raise RuntimeError(
            "theta CCY vacuum seed exhausted its word-length safety cap "
            "before reaching the requested primitive-word tail"
        )
    return result.value


def b_square_rs_from_h(r: int, s: int, h: complex) -> complex:
    """Return b_{r,s}(h)^2 in the CCY c-recursion convention."""
    if r < 2 or s < 1:
        raise ValueError("c-recursion uses r >= 2 and s >= 1")
    h = _as_complex(h)
    radical = (r - s) ** 2 + 4.0 * (r * s - 1.0) * h + 4.0 * h * h
    return (r * s - 1.0 + 2.0 * h + cmath.sqrt(radical)) / (1.0 - r * r)


def central_charge_from_b_square(b_square: complex) -> complex:
    b_square = _as_complex(b_square)
    return 13.0 + 6.0 * (b_square + 1.0 / b_square)


def c_rs_from_h(r: int, s: int, h: complex) -> complex:
    return central_charge_from_b_square(b_square_rs_from_h(r, s, h))


def dc_rs_dh(r: int, s: int, h: complex) -> complex:
    """Return derivative d c_{r,s}(h) / d h."""
    x = b_square_rs_from_h(r, s, h)
    dh_dx = 0.25 * ((1.0 - r * r) - (1.0 - s * s) / (x * x))
    dc_dx = 6.0 * (1.0 - 1.0 / (x * x))
    if abs(dh_dx) == 0:
        raise ZeroDivisionError("dc_rs_dh hit a branch point")
    return dc_dx / dh_dx


def b_from_c_rs_h(r: int, s: int, h: complex) -> complex:
    return cmath.sqrt(b_square_rs_from_h(r, s, h))


def fusion_polynomial_for_weights(
    r: int,
    s: int,
    b: complex,
    top_weight: complex,
    bottom_weight: complex,
) -> complex:
    lambda_top = momentum_from_weight(top_weight, b)
    lambda_bottom = momentum_from_weight(bottom_weight, b)
    return fusion_polynomial(r, s, b, lambda_top, lambda_bottom)


def minus_dc_dh_times_a_rs(r: int, s: int, h: complex) -> complex:
    r"""Return ``-dc_{r,s}/dh * A_{r,s}`` with universal cancellations.

    With ``x=b^2``,

        A_{r,s} = 1/2 x^(2rs-1) / prod'(p x + ell),

    where the primed product omits ``(p,ell)=(0,0),(r,s)``.  Also

        dc/dh = 24 (x^2 - 1) / ((1-r^2)x^2 - (1-s^2)).

    This representation lets us cancel the ``x-1`` and ``x+1`` zeros before
    evaluating the product.  That avoids the resonant ``0 * infinity`` form
    seen at ``c=25`` in higher-order recursion terms.
    """

    x = b_square_rs_from_h(r, s, h)
    numerator = -12.0 * (x ** (2 * r * s - 1))
    denominator = (1.0 - r * r) * x * x - (1.0 - s * s)

    denominator_factors: list[tuple[int, int]] = []
    for p in range(1 - r, r + 1):
        for ell in range(1 - s, s + 1):
            if (p, ell) in {(0, 0), (r, s)}:
                continue
            denominator_factors.append((p, ell))

    numerator_factors: list[tuple[int, int]] = [(1, -1), (1, 1)]
    remaining_numerator_factors: list[tuple[int, int]] = []
    for num_p, num_ell in numerator_factors:
        matched_index = None
        matched_scale = 1
        for idx, (den_p, den_ell) in enumerate(denominator_factors):
            if den_p != 0 and den_p * num_ell == den_ell * num_p:
                matched_index = idx
                matched_scale = den_p // num_p
                break
        if matched_index is None:
            remaining_numerator_factors.append((num_p, num_ell))
        else:
            denominator *= matched_scale
            denominator_factors.pop(matched_index)

    for p, ell in remaining_numerator_factors:
        numerator *= p * x + ell
    for p, ell in denominator_factors:
        denominator *= p * x + ell
    if abs(denominator) == 0:
        raise ZeroDivisionError(f"singular simplified A_rs prefactor for r={r}, s={s}, h={h!r}")
    return numerator / denominator


def _is_finite_complex(value: complex) -> bool:
    return math.isfinite(value.real) and math.isfinite(value.imag)


def ccy_residue_prefactor_for_weights(
    r: int,
    s: int,
    h_edge: complex,
    top_weight: complex,
    bottom_weight: complex,
) -> complex:
    r"""Return the finite CCY residue prefactor.

    The formal prefactor is

        -dc_{r,s}(h)/dh * A_{r,s}(b_{r,s}(h)) * P_{r,s}^2.

    At resonant points, for example at c=25 along recursive shifted weights,
    the separate factors can look like 0 * infinity because A_{r,s} has a
    vanishing denominator while dc/dh vanishes.  The product has a finite
    analytic limit.  We evaluate that limit directly instead of requiring the
    caller to regulate b away from the resonant value.
    """

    h_edge = _as_complex(h_edge)
    top_weight = _as_complex(top_weight)
    bottom_weight = _as_complex(bottom_weight)

    def direct(current_h: complex) -> complex:
        b_pole = b_from_c_rs_h(r, s, current_h)
        polynomial = fusion_polynomial_for_weights(r, s, b_pole, top_weight, bottom_weight)
        return minus_dc_dh_times_a_rs(r, s, current_h) * polynomial * polynomial

    try:
        value = direct(h_edge)
        if _is_finite_complex(value):
            return value
    except ZeroDivisionError:
        pass

    scale = max(1.0, abs(h_edge))
    samples: list[tuple[float, complex]] = []
    for relative_step in (1.0e-5, 3.0e-6, 1.0e-6, 3.0e-7, 1.0e-7, 3.0e-8):
        step = relative_step * scale
        try:
            value = direct(h_edge + step)
        except ZeroDivisionError:
            continue
        if _is_finite_complex(value):
            samples.append((step, value))

    if not samples:
        raise ZeroDivisionError(f"could not resolve resonant CCY residue for r={r}, s={s}, h={h_edge!r}")
    if len(samples) == 1:
        return samples[-1][1]

    step_a, value_a = samples[-2]
    step_b, value_b = samples[-1]
    if step_a == step_b:
        return value_b
    return (step_a * value_b - step_b * value_a) / (step_a - step_b)


@dataclass(frozen=True)
class CCYGenus2BlockResult:
    value: complex
    c: complex
    h1: complex
    h2: complex
    h3: complex
    q1: complex
    q2: complex
    q3: complex
    order: int
    include_vacuum_seed: bool
    vacuum_word_len: int
    vacuum_oscillator_level_max: int
    global_block_resummed: bool = False
    global_block_tolerance: float = 0.0
    global_block_max_outer_order: int = 0
    partial_fraction_pole_count: int = 0
    partial_fraction_coefficient_count: int = 0
    partial_fraction_max_pole_order: int = 0
    collision_regulated: bool = False
    collision_regulator_error: float = 0.0
    collision_regulator_scale: float = 0.0


class ConfluentPoleError(ArithmeticError):
    """Raised when the generic-weight CCY recursion reaches a repeated c-pole."""


@dataclass(frozen=True)
class CollisionRegulatedValue:
    """Symmetric generic-weight limit of a block at a confluent pole."""

    value: complex
    error_estimate: float
    relative_scale: float
    representative_partial_fraction: "PartialFractionInC"


def _matching_pole(
    poles: dict[complex, list[complex]],
    pole: complex,
    tolerance: float,
) -> complex | None:
    """Return an exactly identical pole key.

    ``tolerance`` is retained in the signature for API compatibility, but it
    must not define pole identity: CCY's simple-pole representation applies at
    generic weights, and two nearby generic poles are mathematically distinct.
    Repeated algebraic poles generated from identical recursion data have the
    same binary complex value and are merged exactly.
    """

    del tolerance
    for existing in poles:
        if existing == pole:
            return existing
    return None


def _extrapolate_to_zero(
    abscissae: Sequence[float],
    values: Sequence[complex],
) -> complex:
    """Evaluate the Lagrange interpolant through ``values`` at zero."""

    if len(abscissae) != len(values) or not abscissae:
        raise ValueError("extrapolation requires equally sized nonempty data")
    total = 0.0 + 0.0j
    for index, (abscissa, value) in enumerate(zip(abscissae, values)):
        coefficient = 1.0
        for other_index, other_abscissa in enumerate(abscissae):
            if other_index == index:
                continue
            denominator = abscissa - other_abscissa
            if denominator == 0.0:
                raise ValueError("extrapolation abscissae must be distinct")
            coefficient *= -other_abscissa / denominator
        total += coefficient * complex(value)
    return complex(total)


def _normalized_collision_direction(
    count: int,
    direction: Sequence[float] | None,
) -> tuple[float, ...]:
    """Return a centered, pairwise-distinct regulator direction."""

    if count <= 1:
        raise ValueError("a collision regulator requires at least two weights")
    if direction is None:
        raw = tuple(float(2 * index - (count - 1)) for index in range(count))
    else:
        if len(direction) != count:
            raise ValueError(f"collision regulator direction must contain {count} entries")
        raw = tuple(float(value) for value in direction)
    if any(not math.isfinite(value) for value in raw):
        raise ValueError("collision regulator direction must be finite")
    mean = sum(raw) / count
    centered = tuple(value - mean for value in raw)
    norm = max(abs(value) for value in centered)
    if norm == 0.0 or len(set(centered)) != count:
        raise ValueError("collision regulator direction must have pairwise-distinct entries")
    return tuple(value / norm for value in centered)


def collision_regulated_partial_fraction_value(
    *,
    build_partial_fraction: Callable[[tuple[complex, ...]], "PartialFractionInC"],
    weights: Sequence[complex],
    central_charge: complex,
    pole_tolerance: float,
    relative_scale: float = 1.0e-3,
    direction: Sequence[float] | None = None,
) -> CollisionRegulatedValue:
    r"""Return the generic-weight limit at an exact collision.

    The physical central charge is never moved.  For a normalized generic
    direction ``eta`` we evaluate

    ``[F(h + eps eta, c) + F(h - eps eta, c)] / 2``

    at three geometrically decreasing ``eps`` values.  The symmetric values
    are analytic in ``eps^2``; quadratic extrapolation in that variable gives
    the confluent limit and the difference from the two-point extrapolation is
    retained as an error estimate.
    """

    if not math.isfinite(float(relative_scale)) or float(relative_scale) <= 0.0:
        raise ValueError("collision regulator scale must be finite and positive")
    weight_tuple = tuple(complex(weight) for weight in weights)
    eta = _normalized_collision_direction(len(weight_tuple), direction)
    absolute_scale = max(1.0, *(abs(weight) for weight in weight_tuple))
    steps = tuple(float(relative_scale) * factor for factor in (1.0, 0.5, 0.25))
    symmetric_values: list[complex] = []
    representative: PartialFractionInC | None = None

    for step in steps:
        signed_values: list[complex] = []
        for sign in (1.0, -1.0):
            shifted = tuple(
                weight + sign * step * absolute_scale * component
                for weight, component in zip(weight_tuple, eta)
            )
            partial = build_partial_fraction(shifted)
            if representative is None or (step == steps[-1] and sign > 0.0):
                representative = partial
            signed_values.append(
                partial.value(complex(central_charge), pole_tolerance=pole_tolerance)
            )
        symmetric_values.append(0.5 * (signed_values[0] + signed_values[1]))

    squared_steps = tuple(step * step for step in steps)
    extrapolated = _extrapolate_to_zero(squared_steps, symmetric_values)
    lower_order = _extrapolate_to_zero(squared_steps[-2:], symmetric_values[-2:])
    if representative is None:  # pragma: no cover - steps is statically nonempty
        raise RuntimeError("collision regulator produced no partial fractions")
    return CollisionRegulatedValue(
        value=extrapolated,
        error_estimate=float(abs(extrapolated - lower_order)),
        relative_scale=float(relative_scale),
        representative_partial_fraction=representative,
    )


@dataclass
class PartialFractionInC:
    """Finite generic-weight partial-fraction representation in ``c``.

    The represented function is

        constant + sum_p sum_{k>=1} coeff[p][k-1] / (c-p)^k.

    Pole identity is exact.  Numerically nearby generic poles remain distinct;
    an exact collision encountered in a shifted lower block raises
    :class:`ConfluentPoleError` so the public evaluator can take the complete
    generic-weight limit.
    """

    constant: complex = 0.0 + 0.0j
    poles: dict[complex, list[complex]] | None = None

    def __post_init__(self) -> None:
        if self.poles is None:
            self.poles = {}

    def add_constant(self, value: complex) -> None:
        self.constant += complex(value)

    def add_pole_coefficient(
        self,
        pole: complex,
        order: int,
        coefficient: complex,
        *,
        pole_tolerance: float,
    ) -> None:
        if order <= 0:
            raise ValueError("partial-fraction pole order must be positive")
        coefficient = complex(coefficient)
        if abs(coefficient) == 0:
            return
        pole = complex(pole)
        key = _matching_pole(self.poles or {}, pole, pole_tolerance)
        if key is None:
            key = pole
            self.poles[key] = []
        coeffs = self.poles[key]
        while len(coeffs) < order:
            coeffs.append(0.0 + 0.0j)
        coeffs[order - 1] += coefficient

    def add(self, other: "PartialFractionInC", *, pole_tolerance: float) -> None:
        self.add_constant(other.constant)
        for pole, coeffs in (other.poles or {}).items():
            for idx, coeff in enumerate(coeffs, start=1):
                self.add_pole_coefficient(pole, idx, coeff, pole_tolerance=pole_tolerance)

    def finite_part_at(self, pole: complex, *, pole_tolerance: float) -> complex:
        pole = complex(pole)
        value = self.constant
        for existing, coeffs in (self.poles or {}).items():
            if existing == pole:
                continue
            delta = pole - existing
            value += sum(coeff / (delta**order) for order, coeff in enumerate(coeffs, start=1))
        return value

    def pole_coefficients_at(self, pole: complex, *, pole_tolerance: float) -> tuple[complex, ...]:
        key = _matching_pole(self.poles or {}, complex(pole), pole_tolerance)
        if key is None:
            return ()
        return tuple(self.poles[key])

    def add_residue_times_laurent_at(
        self,
        *,
        pole: complex,
        residue: complex,
        subblock: "PartialFractionInC",
        pole_tolerance: float,
    ) -> None:
        """Add a generic-weight CCY residue evaluated at ``c=pole``.

        CCY's c-recursion assumes generic internal weights.  If ``subblock``
        itself has the same pole, its value at ``c=pole`` is undefined and the
        complete expression must instead be obtained as a confluent limit in
        the weights.  Promoting the subblock Laurent coefficients to
        higher-order poles loses the necessary weight derivatives and is not a
        valid continuation.
        """
        residue = complex(residue)
        coincident = subblock.pole_coefficients_at(
            pole,
            pole_tolerance=pole_tolerance,
        )
        if any(coefficient != 0 for coefficient in coincident):
            raise ConfluentPoleError(
                f"lower CCY block has a pole coincident with outer pole {complex(pole)!r}"
            )
        finite_part = subblock.finite_part_at(pole, pole_tolerance=pole_tolerance)
        self.add_pole_coefficient(
            pole,
            1,
            residue * finite_part,
            pole_tolerance=pole_tolerance,
        )

    def value(self, c: complex, *, pole_tolerance: float) -> complex:
        c = complex(c)
        total = self.constant
        for pole, coeffs in (self.poles or {}).items():
            delta = c - pole
            if abs(delta) < pole_tolerance:
                raise ZeroDivisionError(f"requested c={c!r} lies on a c-recursion pole {pole!r}")
            total += sum(coeff / (delta**order) for order, coeff in enumerate(coeffs, start=1))
        return total

    @property
    def max_pole_order(self) -> int:
        return max((len(coeffs) for coeffs in (self.poles or {}).values()), default=0)

    @property
    def pole_count(self) -> int:
        return len(self.poles or {})

    @property
    def coefficient_count(self) -> int:
        return sum(len(coeffs) for coeffs in (self.poles or {}).values())


def ccy_genus2_block_partial_fraction(
    *,
    h1: complex,
    h2: complex,
    h3: complex,
    q1: complex,
    q2: complex,
    q3: complex,
    order: int,
    include_vacuum_seed: bool = True,
    vacuum_word_len: int = 3,
    vacuum_oscillator_level_max: int = 12,
    vacuum_word_tail_tolerance: float | None = None,
    vacuum_minimum_word_length: int = 5,
    resum_global_block: bool = False,
    global_block_tolerance: float = 1.0e-13,
    global_block_max_outer_order: int = 64,
    pole_tolerance: float = 1.0e-12,
) -> PartialFractionInC:
    """Return the CCY genus-two block as a partial fraction in c.

    The edge/weight order is the descendant-tensor slot order
    ``(infinity, one, zero)``.  The Schottky vacuum seed is converted back to
    the geometric ``(zero, one, infinity)`` order at its call boundary.
    """
    order = _validate_order(order)
    q1 = _as_complex(q1)
    q2 = _as_complex(q2)
    q3 = _as_complex(q3)
    global_block_tolerance = float(global_block_tolerance)
    global_block_max_outer_order = int(global_block_max_outer_order)

    vacuum_seed = (
        genus2_vacuum_seed_schottky(
            q3,
            q2,
            q1,
            max_word_len=vacuum_word_len,
            oscillator_level_max=vacuum_oscillator_level_max,
            word_tail_tolerance=vacuum_word_tail_tolerance,
            minimum_word_length=vacuum_minimum_word_length,
        )
        if include_vacuum_seed
        else 1.0 + 0.0j
    )

    @lru_cache(maxsize=None)
    def recurse(current_h1: complex, current_h2: complex, current_h3: complex, remaining: int) -> PartialFractionInC:
        if resum_global_block:
            global_seed = genus2_global_sl2_block_resummed(
                current_h1,
                current_h2,
                current_h3,
                q1,
                q2,
                q3,
                tolerance=global_block_tolerance,
                max_outer_order=global_block_max_outer_order,
            )
        else:
            global_seed = genus2_global_sl2_block(
                current_h1,
                current_h2,
                current_h3,
                q1,
                q2,
                q3,
                remaining,
            )
        seed = vacuum_seed * global_seed
        total = PartialFractionInC(constant=seed)
        weights = (current_h1, current_h2, current_h3)
        q_values = (q1, q2, q3)
        fusion_pairs = (
            (current_h3, current_h2),
            (current_h3, current_h1),
            (current_h1, current_h2),
        )
        for edge in range(3):
            h_edge = weights[edge]
            for r in range(2, remaining + 1):
                for s in range(1, remaining // r + 1):
                    level = r * s
                    if level > remaining:
                        continue
                    pole_c = c_rs_from_h(r, s, h_edge)
                    top_weight, bottom_weight = fusion_pairs[edge]
                    residue = (q_values[edge] ** level) * ccy_residue_prefactor_for_weights(
                        r, s, h_edge, top_weight, bottom_weight
                    )
                    shifted = list(weights)
                    shifted[edge] = h_edge + level
                    subblock = recurse(shifted[0], shifted[1], shifted[2], remaining - level)
                    total.add_residue_times_laurent_at(
                        pole=pole_c,
                        residue=residue,
                        subblock=subblock,
                        pole_tolerance=pole_tolerance,
                    )
        return total

    return recurse(_as_complex(h1), _as_complex(h2), _as_complex(h3), order)


def ccy_genus2_block(
    *,
    c: complex,
    h1: complex,
    h2: complex,
    h3: complex,
    q1: complex,
    q2: complex,
    q3: complex,
    order: int,
    include_vacuum_seed: bool = True,
    vacuum_word_len: int = 3,
    vacuum_oscillator_level_max: int = 12,
    vacuum_word_tail_tolerance: float | None = None,
    vacuum_minimum_word_length: int = 5,
    resum_global_block: bool = False,
    global_block_tolerance: float = 1.0e-13,
    global_block_max_outer_order: int = 64,
    pole_tolerance: float = 1.0e-12,
    collision_aware: bool = True,
    collision_regulator_scale: float = 1.0e-3,
    collision_regulator_direction: Sequence[float] | None = None,
) -> CCYGenus2BlockResult:
    """Evaluate the CCY genus-two block by c-recursion.

    The pole recursion is truncated by total plumbing degree.  When
    ``resum_global_block`` is true, its regular seed is nevertheless evaluated
    to all global-descendant levels; otherwise the legacy strict power-series
    truncation is retained.  The optional vacuum seed is a finite Schottky
    primitive-class product and is therefore an approximation to the exact
    c=infinity vacuum block.

    ``(h1,q1)``, ``(h2,q2)``, and ``(h3,q3)`` occupy the
    ``(infinity, one, zero)`` slots of the three-point descendant tensor.
    """
    order = _validate_order(order)
    c = _as_complex(c)
    q1 = _as_complex(q1)
    q2 = _as_complex(q2)
    q3 = _as_complex(q3)
    global_block_tolerance = float(global_block_tolerance)
    global_block_max_outer_order = int(global_block_max_outer_order)
    collision_regulator_scale = float(collision_regulator_scale)
    if (
        not math.isfinite(collision_regulator_scale)
        or collision_regulator_scale <= 0.0
    ):
        raise ValueError("collision regulator scale must be finite and positive")

    if collision_aware:
        weights = (_as_complex(h1), _as_complex(h2), _as_complex(h3))

        def build_partial(current_weights: tuple[complex, ...]) -> PartialFractionInC:
            return ccy_genus2_block_partial_fraction(
                h1=current_weights[0],
                h2=current_weights[1],
                h3=current_weights[2],
                q1=q1,
                q2=q2,
                q3=q3,
                order=order,
                include_vacuum_seed=include_vacuum_seed,
                vacuum_word_len=vacuum_word_len,
                vacuum_oscillator_level_max=vacuum_oscillator_level_max,
                vacuum_word_tail_tolerance=vacuum_word_tail_tolerance,
                vacuum_minimum_word_length=vacuum_minimum_word_length,
                resum_global_block=resum_global_block,
                global_block_tolerance=global_block_tolerance,
                global_block_max_outer_order=global_block_max_outer_order,
                pole_tolerance=pole_tolerance,
            )

        regulated = False
        regulator_error = 0.0
        regulator_scale = 0.0
        try:
            partial_fraction = build_partial(weights)
            value = partial_fraction.value(c, pole_tolerance=pole_tolerance)
        except ConfluentPoleError:
            regulated_value = collision_regulated_partial_fraction_value(
                build_partial_fraction=build_partial,
                weights=weights,
                central_charge=c,
                pole_tolerance=pole_tolerance,
                relative_scale=collision_regulator_scale,
                direction=collision_regulator_direction,
            )
            partial_fraction = regulated_value.representative_partial_fraction
            value = regulated_value.value
            regulated = True
            regulator_error = regulated_value.error_estimate
            regulator_scale = regulated_value.relative_scale
        return CCYGenus2BlockResult(
            value=value,
            c=c,
            h1=weights[0],
            h2=weights[1],
            h3=weights[2],
            q1=q1,
            q2=q2,
            q3=q3,
            order=order,
            include_vacuum_seed=include_vacuum_seed,
            vacuum_word_len=int(vacuum_word_len),
            vacuum_oscillator_level_max=int(vacuum_oscillator_level_max),
            global_block_resummed=bool(resum_global_block),
            global_block_tolerance=(global_block_tolerance if resum_global_block else 0.0),
            global_block_max_outer_order=(
                global_block_max_outer_order if resum_global_block else 0
            ),
            partial_fraction_pole_count=partial_fraction.pole_count,
            partial_fraction_coefficient_count=partial_fraction.coefficient_count,
            partial_fraction_max_pole_order=partial_fraction.max_pole_order,
            collision_regulated=regulated,
            collision_regulator_error=regulator_error,
            collision_regulator_scale=regulator_scale,
        )

    vacuum_seed = (
        genus2_vacuum_seed_schottky(
            q3,
            q2,
            q1,
            max_word_len=vacuum_word_len,
            oscillator_level_max=vacuum_oscillator_level_max,
            word_tail_tolerance=vacuum_word_tail_tolerance,
            minimum_word_length=vacuum_minimum_word_length,
        )
        if include_vacuum_seed
        else 1.0 + 0.0j
    )

    @lru_cache(maxsize=None)
    def recurse(current_c: complex, current_h1: complex, current_h2: complex, current_h3: complex, remaining: int) -> complex:
        if resum_global_block:
            global_seed = genus2_global_sl2_block_resummed(
                current_h1,
                current_h2,
                current_h3,
                q1,
                q2,
                q3,
                tolerance=global_block_tolerance,
                max_outer_order=global_block_max_outer_order,
            )
        else:
            global_seed = genus2_global_sl2_block(
                current_h1,
                current_h2,
                current_h3,
                q1,
                q2,
                q3,
                remaining,
            )
        seed = vacuum_seed * global_seed
        total = seed
        weights = (current_h1, current_h2, current_h3)
        q_values = (q1, q2, q3)
        fusion_pairs = (
            (current_h3, current_h2),
            (current_h3, current_h1),
            (current_h1, current_h2),
        )
        for edge in range(3):
            h_edge = weights[edge]
            for r in range(2, remaining + 1):
                for s in range(1, remaining // r + 1):
                    level = r * s
                    if level > remaining:
                        continue
                    pole_c = c_rs_from_h(r, s, h_edge)
                    denominator = current_c - pole_c
                    if abs(denominator) < pole_tolerance:
                        raise ZeroDivisionError(
                            f"central charge is too close to c_({r},{s})({h_edge!r})={pole_c!r}"
                        )
                    b_pole = b_from_c_rs_h(r, s, h_edge)
                    top_weight, bottom_weight = fusion_pairs[edge]
                    polynomial = fusion_polynomial_for_weights(r, s, b_pole, top_weight, bottom_weight)
                    residue = (
                        -dc_rs_dh(r, s, h_edge)
                        * (q_values[edge] ** level)
                        * zamolodchikov_a_rs(r, s, b_pole)
                        * polynomial
                        * polynomial
                        / denominator
                    )
                    shifted = list(weights)
                    shifted[edge] = h_edge + level
                    total += residue * recurse(pole_c, shifted[0], shifted[1], shifted[2], remaining - level)
        return total

    return CCYGenus2BlockResult(
        value=recurse(c, _as_complex(h1), _as_complex(h2), _as_complex(h3), order),
        c=c,
        h1=_as_complex(h1),
        h2=_as_complex(h2),
        h3=_as_complex(h3),
        q1=q1,
        q2=q2,
        q3=q3,
        order=order,
        include_vacuum_seed=include_vacuum_seed,
        vacuum_word_len=int(vacuum_word_len),
        vacuum_oscillator_level_max=int(vacuum_oscillator_level_max),
        global_block_resummed=bool(resum_global_block),
        global_block_tolerance=(global_block_tolerance if resum_global_block else 0.0),
        global_block_max_outer_order=(
            global_block_max_outer_order if resum_global_block else 0
        ),
    )


def liouville_c_and_weight(b: float, momentum: float) -> tuple[float, float]:
    q_background = b + 1.0 / b
    return 1.0 + 6.0 * q_background * q_background, 0.25 * q_background * q_background + momentum * momentum


def run() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the CCY genus-two Virasoro block.")
    parser.add_argument("--b", type=float, help="Liouville b; if set, c and h_i are read from P_i.")
    parser.add_argument("--p1", type=float, default=0.2)
    parser.add_argument("--p2", type=float, default=0.3)
    parser.add_argument("--p3", type=float, default=0.25)
    parser.add_argument("--c", type=parse_complex)
    parser.add_argument("--h1", type=parse_complex)
    parser.add_argument("--h2", type=parse_complex)
    parser.add_argument("--h3", type=parse_complex)
    parser.add_argument("--q1", type=parse_complex, required=True, help="CCY infinity-slot edge")
    parser.add_argument("--q2", type=parse_complex, required=True, help="CCY one-slot edge")
    parser.add_argument("--q3", type=parse_complex, required=True, help="CCY zero-slot edge")
    parser.add_argument("--order", type=int, default=2)
    parser.add_argument("--no-vacuum-seed", action="store_true")
    parser.add_argument("--vacuum-word-len", type=int, default=3)
    parser.add_argument("--vacuum-oscillator-level-max", type=int, default=12)
    parser.add_argument("--resum-global-block", action="store_true")
    parser.add_argument("--global-block-tolerance", type=float, default=1.0e-13)
    parser.add_argument("--global-block-max-outer-order", type=int, default=64)
    parser.add_argument("--period-matrix-word-len", type=int, default=5)
    args = parser.parse_args()

    if args.b is not None:
        c_value, h1 = liouville_c_and_weight(args.b, args.p1)
        _, h2 = liouville_c_and_weight(args.b, args.p2)
        _, h3 = liouville_c_and_weight(args.b, args.p3)
    else:
        if args.c is None or args.h1 is None or args.h2 is None or args.h3 is None:
            raise ValueError("pass either --b with --p1/--p2/--p3 or explicit --c/--h1/--h2/--h3")
        c_value, h1, h2, h3 = args.c, args.h1, args.h2, args.h3

    result = ccy_genus2_block(
        c=c_value,
        h1=h1,
        h2=h2,
        h3=h3,
        q1=args.q1,
        q2=args.q2,
        q3=args.q3,
        order=args.order,
        include_vacuum_seed=not args.no_vacuum_seed,
        vacuum_word_len=args.vacuum_word_len,
        vacuum_oscillator_level_max=args.vacuum_oscillator_level_max,
        resum_global_block=args.resum_global_block,
        global_block_tolerance=args.global_block_tolerance,
        global_block_max_outer_order=args.global_block_max_outer_order,
    )

    print("CCY genus-two Virasoro block")
    print(f"  c={format_complex(result.c)}")
    print(f"  h1={format_complex(result.h1)}")
    print(f"  h2={format_complex(result.h2)}")
    print(f"  h3={format_complex(result.h3)}")
    print(f"  q1={format_complex(result.q1)}")
    print(f"  q2={format_complex(result.q2)}")
    print(f"  q3={format_complex(result.q3)}")
    print(f"  order={result.order}")
    print(f"  vacuum seed={result.include_vacuum_seed}")
    print(f"  global block resummed={result.global_block_resummed}")
    print(f"  c-poles={result.partial_fraction_pole_count}")
    print(f"  c-pole coefficients={result.partial_fraction_coefficient_count}")
    print(f"  max c-pole order={result.partial_fraction_max_pole_order}")
    print(f"  value={format_complex(result.value)}")
    omega = schottky_theta_period_matrix_cross_ratio(
        result.q1,
        result.q2,
        result.q3,
        max_word_len=args.period_matrix_word_len,
    )
    print(f"  period matrix word length={args.period_matrix_word_len}")
    print("  Omega=")
    for row in omega:
        print("    " + "  ".join(format_complex(complex(entry)) for entry in row))


if __name__ == "__main__":
    run()
