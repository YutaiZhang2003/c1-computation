#!/usr/bin/env python3
"""Virasoro torus one-point and multipoint necklace conformal blocks.

This module implements the fixed-weight central-charge recursion for the
non-trivial elliptic part of the genus-one one-point Virasoro block.  The
older Zamolodchikov/Poghossian internal-weight recursion remains available as
an independent check.  It also implements the simultaneous internal-weight
recursion for torus ``N``-point necklace blocks, with dedicated two- and
three-point interfaces.  The CFT data, such as three-point coefficients, are
intentionally not included here: this file only supplies the universal chiral
blocks multiplying them.
"""

from __future__ import annotations

import argparse
import cmath
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from numbers import Integral

import numpy as np


Number = complex


def parse_complex(value: str) -> complex:
    return complex(value.replace("i", "j"))


def _as_complex(value: complex | float | int) -> complex:
    return complex(value)


def central_charge_to_background_charge(c: complex | float) -> complex:
    """Return Q with c = 1 + 6 Q^2, using the principal square-root."""
    return cmath.sqrt((_as_complex(c) - 1.0) / 6.0)


def central_charge_to_b(c: complex | float, branch: int = 1) -> complex:
    """Return one solution of b + 1/b = Q, where c = 1 + 6 Q^2."""
    if branch not in {-1, 1}:
        raise ValueError("branch must be +1 or -1")
    q_background = central_charge_to_background_charge(c)
    return 0.5 * (q_background + branch * cmath.sqrt(q_background * q_background - 4.0))


def background_charge(b: complex) -> complex:
    return b + 1.0 / b


def weight_from_momentum(lam: complex, b: complex) -> complex:
    """Return Delta_lambda = (Q^2 - lambda^2)/4."""
    q_background = background_charge(b)
    return 0.25 * (q_background * q_background - lam * lam)


def momentum_from_weight(weight: complex | float, b: complex, sign: int = 1) -> complex:
    """Return lambda from Delta = (Q^2 - lambda^2)/4."""
    if sign not in {-1, 1}:
        raise ValueError("sign must be +1 or -1")
    q_background = background_charge(b)
    return sign * cmath.sqrt(q_background * q_background - 4.0 * _as_complex(weight))


def degenerate_momentum(r: int, s: int, b: complex) -> complex:
    return r * b + s / b


def degenerate_weight(r: int, s: int, b: complex) -> complex:
    return weight_from_momentum(degenerate_momentum(r, s, b), b)


def shifted_degenerate_weight(r: int, s: int, b: complex) -> complex:
    """Return Delta_{r,s}+rs = Delta_{r,-s}."""
    return degenerate_weight(r, -s, b)


def zamolodchikov_a_rs(r: int, s: int, b: complex) -> complex:
    """Universal A_rs factor for the Delta-recursion."""
    product = 1.0 + 0.0j
    for p in range(1 - r, r + 1):
        for ell in range(1 - s, s + 1):
            if (p, ell) in {(0, 0), (r, s)}:
                continue
            product *= 1.0 / (p * b + ell / b)
    return 0.5 * product


def fusion_polynomial(
    r: int,
    s: int,
    b: complex,
    lambda_top: complex,
    lambda_bottom: complex,
) -> complex:
    """Fusion polynomial P_c^{rs}[Delta_top / Delta_bottom].

    We use the momentum convention Delta_lambda = (Q^2 - lambda^2)/4.
    """
    product = 1.0 + 0.0j
    for p in range(1 - r, r, 2):
        for ell in range(1 - s, s, 2):
            shift = p * b + ell / b
            product *= 0.5 * (lambda_top + lambda_bottom + shift)
            product *= 0.5 * (lambda_top - lambda_bottom + shift)
    return product


def torus_one_point_residue(r: int, s: int, b: complex, external_lambda: complex) -> complex:
    """Residue multiplying the lower-level torus one-point block."""
    lambda_rs = degenerate_momentum(r, s, b)
    lambda_r_minus_s = degenerate_momentum(r, -s, b)
    return (
        zamolodchikov_a_rs(r, s, b)
        * fusion_polynomial(r, s, b, external_lambda, lambda_r_minus_s)
        * fusion_polynomial(r, s, b, external_lambda, lambda_rs)
    )


def c_recursion_b_square(r: int, s: int, internal_weight: complex) -> complex:
    r"""Return ``b_{r,s}(h)^2`` in the fixed-weight CCY convention.

    The representative with ``r >= 2`` removes the duplicate Kac labels that
    would otherwise occur under ``b -> 1/b``.
    """

    if r < 2 or s < 1:
        raise ValueError("c-recursion uses r >= 2 and s >= 1")
    h = _as_complex(internal_weight)
    radical = (r - s) ** 2 + 4.0 * (r * s - 1.0) * h + 4.0 * h * h
    return (r * s - 1.0 + 2.0 * h + cmath.sqrt(radical)) / (1.0 - r * r)


def c_recursion_pole(r: int, s: int, internal_weight: complex) -> complex:
    """Return the central-charge pole ``c_{r,s}(h)`` at fixed ``h``."""

    b_square = c_recursion_b_square(r, s, internal_weight)
    return 13.0 + 6.0 * (b_square + 1.0 / b_square)


def _minus_dc_dh_times_a_rs(r: int, s: int, internal_weight: complex) -> complex:
    r"""Return the finite product ``-dc_{r,s}/dh A_{r,s}``.

    Evaluating the derivative and ``A_{r,s}`` separately produces a spurious
    ``0 * infinity`` at resonant recursive states.  This rational-in-``b^2``
    form performs the universal cancellation first, matching the higher-genus
    CCY implementation.
    """

    x = c_recursion_b_square(r, s, internal_weight)
    numerator = -12.0 * (x ** (2 * r * s - 1))
    denominator = (1.0 - r * r) * x * x - (1.0 - s * s)

    denominator_factors: list[tuple[int, int]] = []
    for p in range(1 - r, r + 1):
        for ell in range(1 - s, s + 1):
            if (p, ell) in {(0, 0), (r, s)}:
                continue
            denominator_factors.append((p, ell))

    remaining_numerator_factors: list[tuple[int, int]] = []
    for num_p, num_ell in ((1, -1), (1, 1)):
        matched_index = None
        matched_scale = 1
        for index, (den_p, den_ell) in enumerate(denominator_factors):
            if den_p != 0 and den_p * num_ell == den_ell * num_p:
                matched_index = index
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
    if denominator == 0.0:
        raise ZeroDivisionError(
            f"singular simplified c-recursion prefactor for r={r}, s={s}"
        )
    return numerator / denominator


def fusion_polynomial_for_weights(
    r: int,
    s: int,
    b: complex,
    top_weight: complex,
    bottom_weight: complex,
) -> complex:
    """Fusion polynomial with both arguments supplied as weights."""

    return fusion_polynomial(
        r,
        s,
        b,
        momentum_from_weight(top_weight, b),
        momentum_from_weight(bottom_weight, b),
    )


def torus_one_point_c_recursion_residue(
    r: int,
    s: int,
    internal_weight: complex,
    external_weight: complex,
) -> complex:
    r"""Return the fixed-weight ``c``-pole residue for a tadpole edge.

    This is the genus-one specialization of the CCY handle residue,

    ``-dc/dh A_rs P(d,h+rs) P(d,h)``.
    """

    h = _as_complex(internal_weight)
    d = _as_complex(external_weight)
    level = r * s
    b_pole = cmath.sqrt(c_recursion_b_square(r, s, h))
    return (
        _minus_dc_dh_times_a_rs(r, s, h)
        * fusion_polynomial_for_weights(r, s, b_pole, d, h + level)
        * fusion_polynomial_for_weights(r, s, b_pole, d, h)
    )


def torus_one_point_large_c_elliptic_coefficients(
    internal_weight: complex,
    external_weight: complex,
    order: int,
) -> list[complex]:
    r"""Return coefficients of the eta-stripped large-``c`` seed.

    The all-level global block is

    ``G=(1-q)^(d-1) 2F1(d,2h+d-1;2h;q)``.

    The full large-``c`` Virasoro seed contains the oscillator product
    ``prod_{n>=2}(1-q^n)^-1``.  Multiplication by the Euler product that
    defines the eta-stripped elliptic block leaves

    ``H_infinity=(1-q) G=(1-q)^d 2F1(d,2h+d-1;2h;q)``.
    """

    if order < 0:
        raise ValueError("order must be non-negative")
    h = _as_complex(internal_weight)
    d = _as_complex(external_weight)
    hypergeometric = [1.0 + 0.0j]
    binomial = [1.0 + 0.0j]
    for n in range(1, order + 1):
        denominator = (2.0 * h + n - 1.0) * n
        if denominator == 0.0:
            raise ZeroDivisionError("large-c torus seed has singular internal weight")
        hypergeometric.append(
            hypergeometric[-1]
            * (d + n - 1.0)
            * (2.0 * h + d + n - 2.0)
            / denominator
        )
        binomial.append(-binomial[-1] * (d - n + 1.0) / n)
    return [
        sum(binomial[k] * hypergeometric[n - k] for k in range(n + 1))
        for n in range(order + 1)
    ]


def partition_numbers(order: int) -> list[int]:
    """Coefficients of prod_{n>=1} (1-q^n)^(-1) through q^order."""
    if order < 0:
        raise ValueError("order must be non-negative")
    values = [0] * (order + 1)
    values[0] = 1
    for part in range(1, order + 1):
        for n in range(part, order + 1):
            values[n] += values[n - part]
    return values


def dedekind_eta(q: complex, *, tolerance: float = 1.0e-15, max_terms: int = 100000) -> complex:
    """Return eta(q)=q^(1/24) prod_{n>=1}(1-q^n) for |q|<1."""
    q = _as_complex(q)
    if not 0 < abs(q) < 1:
        raise ValueError("Dedekind eta expects 0 < |q| < 1")
    product = 1.0 + 0.0j
    q_power = q
    for _ in range(max_terms):
        product *= 1.0 - q_power
        if abs(q_power) < tolerance:
            return (q ** (1.0 / 24.0)) * product
        q_power *= q
    raise RuntimeError("Dedekind eta product did not converge")


@dataclass(frozen=True)
class TorusOnePointBlockParameters:
    c: complex
    internal_weight: complex
    external_weight: complex
    b: complex
    external_lambda: complex


class TorusOnePointVirasoroBlock:
    """Numerical torus one-point Virasoro block for a fixed channel."""

    def __init__(
        self,
        c: complex | float,
        internal_weight: complex | float,
        external_weight: complex | float,
        *,
        b: complex | float | None = None,
        external_lambda: complex | float | None = None,
        recursion: str = "c",
        pole_tolerance: float = 1.0e-13,
    ) -> None:
        if b is None:
            b_value = central_charge_to_b(c)
        else:
            b_value = _as_complex(b)
        if external_lambda is None:
            external_lambda_value = momentum_from_weight(external_weight, b_value)
        else:
            external_lambda_value = _as_complex(external_lambda)

        self.params = TorusOnePointBlockParameters(
            c=_as_complex(c),
            internal_weight=_as_complex(internal_weight),
            external_weight=_as_complex(external_weight),
            b=b_value,
            external_lambda=external_lambda_value,
        )
        if recursion not in {"c", "h"}:
            raise ValueError("recursion must be 'c' or 'h'")
        self.recursion = recursion
        self.pole_tolerance = pole_tolerance

    @lru_cache(maxsize=None)
    def _h_coefficient(self, n: int, internal_weight: complex) -> complex:
        if n < 0:
            return 0.0j
        if n == 0:
            return 1.0 + 0.0j

        total = 0.0j
        b = self.params.b
        for r in range(1, n + 1):
            for s in range(1, n // r + 1):
                level = r * s
                pole = degenerate_weight(r, s, b)
                denominator = internal_weight - pole
                if abs(denominator) < self.pole_tolerance:
                    raise ZeroDivisionError(
                        "internal weight is too close to the degenerate pole "
                        f"Delta_({r},{s})={pole!r}"
                    )
                residue = torus_one_point_residue(r, s, b, self.params.external_lambda)
                shifted_weight = shifted_degenerate_weight(r, s, b)
                total += residue * self._h_coefficient(n - level, shifted_weight) / denominator
        return total

    @lru_cache(maxsize=None)
    def _c_coefficient(
        self,
        n: int,
        central_charge: complex,
        internal_weight: complex,
    ) -> complex:
        if n < 0:
            return 0.0j
        seed = torus_one_point_large_c_elliptic_coefficients(
            internal_weight,
            self.params.external_weight,
            n,
        )[n]
        total = seed
        for r in range(2, n + 1):
            for s in range(1, n // r + 1):
                level = r * s
                pole = c_recursion_pole(r, s, internal_weight)
                denominator = central_charge - pole
                if abs(denominator) < self.pole_tolerance:
                    raise ZeroDivisionError(
                        "central charge is too close to the fixed-weight pole "
                        f"c_({r},{s})={pole!r}"
                    )
                residue = torus_one_point_c_recursion_residue(
                    r,
                    s,
                    internal_weight,
                    self.params.external_weight,
                )
                total += (
                    residue
                    * self._c_coefficient(
                        n - level,
                        pole,
                        internal_weight + level,
                    )
                    / denominator
                )
        return total

    def elliptic_coefficients(self, order: int) -> list[complex]:
        """Return H_n for H(q)=sum_n q^n H_n through q^order."""
        if order < 0:
            raise ValueError("order must be non-negative")
        if self.recursion == "h":
            return [
                self._h_coefficient(n, self.params.internal_weight)
                for n in range(order + 1)
            ]
        return [
            self._c_coefficient(n, self.params.c, self.params.internal_weight)
            for n in range(order + 1)
        ]

    def elliptic_block(self, q: complex, order: int) -> complex:
        """Return the non-trivial elliptic block H(q) truncated at q^order."""
        q = _as_complex(q)
        return sum(coeff * (q**n) for n, coeff in enumerate(self.elliptic_coefficients(order)))

    def descendant_coefficients(self, order: int) -> list[complex]:
        """Return coefficients of eta(q)^(-1) q^(1/24) H(q).

        Equivalently, these are coefficients of
        prod_{n>=1}(1-q^n)^(-1) H(q) through q^order.
        """
        h_coeffs = self.elliptic_coefficients(order)
        partitions = partition_numbers(order)
        out: list[complex] = []
        for n in range(order + 1):
            out.append(sum(partitions[n - k] * h_coeffs[k] for k in range(n + 1)))
        return out

    def chiral_block(self, q: complex, order: int, *, include_prefactor: bool = True) -> complex:
        """Return F(q) = q^(h-c/24) prod(1-q^n)^(-1) H(q), truncated."""
        q = _as_complex(q)
        series = sum(coeff * (q**n) for n, coeff in enumerate(self.descendant_coefficients(order)))
        if include_prefactor:
            series *= q ** (self.params.internal_weight - self.params.c / 24.0)
        return series

    def chiral_block_exact_eta(
        self,
        q: complex,
        order: int,
        *,
        eta_tolerance: float = 1.0e-15,
    ) -> complex:
        """Return q^(h-(c-1)/24) eta(q)^(-1) H(q), truncating only H(q)."""
        q = _as_complex(q)
        known_exponent = self.params.internal_weight - (self.params.c - 1.0) / 24.0
        return (
            q**known_exponent
            * self.elliptic_block(q, order)
            / dedekind_eta(q, tolerance=eta_tolerance)
        )


def _necklace_orders(orders: int | Sequence[int], point_count: int) -> tuple[int, ...]:
    """Normalize one rectangular truncation order per necklace cylinder."""

    if isinstance(orders, Integral):
        normalized = (int(orders),) * point_count
    else:
        normalized = tuple(int(order) for order in orders)
        if len(normalized) != point_count:
            raise ValueError(f"orders must contain {point_count} entries")
    if any(order < 0 for order in normalized):
        raise ValueError("block orders must be non-negative")
    return normalized


def _necklace_q_values(q_values: Sequence[complex], point_count: int) -> tuple[complex, ...]:
    normalized = tuple(_as_complex(q) for q in q_values)
    if len(normalized) != point_count:
        raise ValueError(f"q_values must contain {point_count} entries")
    return normalized


@dataclass(frozen=True)
class TorusNecklaceBlockParameters:
    """Fixed CFT data for a torus multipoint necklace block."""

    c: complex
    internal_weights: tuple[complex, ...]
    external_weights: tuple[complex, ...]
    b: complex


class TorusNecklaceVirasoroBlock:
    r"""Internal-weight recursion for a torus necklace block with ``N >= 2``.

    The internal edge ``i`` carries ``internal_weights[i]`` and ``q_values[i]``.
    The primary of weight ``external_weights[i]`` joins edge ``i`` to edge
    ``i+1`` (cyclically).  The reduced block is the function ``f`` of
    arXiv:1703.09805, eq. (3.20); the non-degenerate torus character in the
    product nome is restored by :meth:`descendant_coefficients`.

    This simultaneous-weight recursion is special to ``N >= 2``.  The torus
    one-point block has a different elliptic recursion and remains implemented
    by :class:`TorusOnePointVirasoroBlock`.
    """

    def __init__(
        self,
        c: complex | float,
        internal_weights: Sequence[complex | float],
        external_weights: Sequence[complex | float],
        *,
        b: complex | float | None = None,
        pole_tolerance: float = 1.0e-13,
    ) -> None:
        internal = tuple(_as_complex(weight) for weight in internal_weights)
        external = tuple(_as_complex(weight) for weight in external_weights)
        if len(internal) < 2:
            raise ValueError("torus necklace h-recursion requires at least two points")
        if len(external) != len(internal):
            raise ValueError("internal_weights and external_weights must have the same length")
        if pole_tolerance < 0:
            raise ValueError("pole_tolerance must be non-negative")
        b_value = central_charge_to_b(c) if b is None else _as_complex(b)
        if b_value == 0.0:
            raise ValueError("b must be nonzero")
        self.params = TorusNecklaceBlockParameters(
            c=_as_complex(c),
            internal_weights=internal,
            external_weights=external,
            b=b_value,
        )
        self.pole_tolerance = float(pole_tolerance)

    @property
    def point_count(self) -> int:
        return len(self.params.internal_weights)

    @lru_cache(maxsize=None)
    def _reduced_coefficient(
        self,
        levels: tuple[int, ...],
        internal_weights: tuple[complex, ...],
    ) -> complex:
        """Return one multivariate coefficient of the reduced block."""

        if any(level < 0 for level in levels):
            return 0.0j
        if not any(levels):
            return 1.0 + 0.0j

        total = 0.0j
        b = self.params.b
        external = self.params.external_weights
        point_count = self.point_count
        for edge, available_level in enumerate(levels):
            for r in range(1, available_level + 1):
                for s in range(1, available_level // r + 1):
                    null_level = r * s
                    pole = degenerate_weight(r, s, b)
                    denominator = internal_weights[edge] - pole
                    if abs(denominator) < self.pole_tolerance:
                        raise ZeroDivisionError(
                            "internal weight is too close to the degenerate pole "
                            f"Delta_({r},{s})={pole!r} on necklace edge {edge}"
                        )

                    # Equation (3.20) keeps all differences h_j-h_i fixed while
                    # taking the residue in h_i.  In absolute weights this first
                    # translates every h_j by Delta_rs-h_i and then raises the
                    # null edge itself by rs.
                    common_shift = pole - internal_weights[edge]
                    shifted_weights = tuple(
                        weight + common_shift + (null_level if index == edge else 0)
                        for index, weight in enumerate(internal_weights)
                    )
                    left_edge = (edge - 1) % point_count
                    right_edge = (edge + 1) % point_count
                    residue = (
                        zamolodchikov_a_rs(r, s, b)
                        * fusion_polynomial_for_weights(
                            r,
                            s,
                            b,
                            internal_weights[left_edge] + common_shift,
                            external[left_edge],
                        )
                        * fusion_polynomial_for_weights(
                            r,
                            s,
                            b,
                            internal_weights[right_edge] + common_shift,
                            external[edge],
                        )
                    )
                    remaining_levels = list(levels)
                    remaining_levels[edge] -= null_level
                    total += (
                        residue
                        * self._reduced_coefficient(
                            tuple(remaining_levels),
                            shifted_weights,
                        )
                        / denominator
                    )
        return total

    def reduced_coefficients(self, orders: int | Sequence[int]) -> np.ndarray:
        r"""Return coefficients of the character-stripped necklace block ``f``."""

        normalized_orders = _necklace_orders(orders, self.point_count)
        coefficients = np.zeros(
            tuple(order + 1 for order in normalized_orders),
            dtype=np.complex128,
        )
        for levels in np.ndindex(coefficients.shape):
            coefficients[levels] = self._reduced_coefficient(
                levels,
                self.params.internal_weights,
            )
        return coefficients

    def descendant_coefficients(self, orders: int | Sequence[int]) -> np.ndarray:
        r"""Return the full necklace descendant coefficients.

        If ``f`` is the reduced coefficient tensor, this restores

        ``prod_(m>=1) (1-(q_1 ... q_N)^m)^(-1)``.
        """

        normalized_orders = _necklace_orders(orders, self.point_count)
        reduced = self.reduced_coefficients(normalized_orders)
        partitions = partition_numbers(min(normalized_orders))
        coefficients = np.zeros_like(reduced)
        for levels in np.ndindex(coefficients.shape):
            coefficients[levels] = sum(
                partitions[diagonal_level]
                * reduced[tuple(level - diagonal_level for level in levels)]
                for diagonal_level in range(min(levels) + 1)
            )
        return coefficients

    @staticmethod
    def _evaluate_coefficients(coefficients: np.ndarray, q_values: tuple[complex, ...]) -> complex:
        value = 0.0j
        for levels in np.ndindex(coefficients.shape):
            monomial = coefficients[levels]
            for q, level in zip(q_values, levels):
                monomial *= q**level
            value += monomial
        return complex(value)

    def reduced_block(
        self,
        q_values: Sequence[complex],
        orders: int | Sequence[int],
    ) -> complex:
        """Evaluate the character-stripped necklace block at ``q_values``."""

        q_tuple = _necklace_q_values(q_values, self.point_count)
        return self._evaluate_coefficients(self.reduced_coefficients(orders), q_tuple)

    def chiral_block(
        self,
        q_values: Sequence[complex],
        orders: int | Sequence[int],
        *,
        include_prefactor: bool = True,
    ) -> complex:
        r"""Evaluate the necklace block, optionally including primary propagation.

        With ``include_prefactor=True`` the multiplier is
        ``prod_i q_i**(h_i-c/24)`` in the cylinder convention of eq. (3.1).
        """

        q_tuple = _necklace_q_values(q_values, self.point_count)
        value = self._evaluate_coefficients(self.descendant_coefficients(orders), q_tuple)
        if include_prefactor:
            for q, internal_weight in zip(q_tuple, self.params.internal_weights):
                value *= q ** (internal_weight - self.params.c / 24.0)
        return value


class TorusTwoPointVirasoroBlock(TorusNecklaceVirasoroBlock):
    """Two-point specialization of the torus necklace h-recursion."""

    def __init__(
        self,
        c: complex | float,
        h1: complex | float,
        h2: complex | float,
        d1: complex | float,
        d2: complex | float,
        *,
        b: complex | float | None = None,
        pole_tolerance: float = 1.0e-13,
    ) -> None:
        super().__init__(
            c,
            (h1, h2),
            (d1, d2),
            b=b,
            pole_tolerance=pole_tolerance,
        )


class TorusThreePointVirasoroBlock(TorusNecklaceVirasoroBlock):
    """Three-point specialization of the torus necklace h-recursion."""

    def __init__(
        self,
        c: complex | float,
        h1: complex | float,
        h2: complex | float,
        h3: complex | float,
        d1: complex | float,
        d2: complex | float,
        d3: complex | float,
        *,
        b: complex | float | None = None,
        pole_tolerance: float = 1.0e-13,
    ) -> None:
        super().__init__(
            c,
            (h1, h2, h3),
            (d1, d2, d3),
            b=b,
            pole_tolerance=pole_tolerance,
        )


def torus_one_point_elliptic_block(
    c: complex | float,
    internal_weight: complex | float,
    external_weight: complex | float,
    q: complex | float,
    order: int,
    *,
    b: complex | float | None = None,
    external_lambda: complex | float | None = None,
) -> complex:
    return TorusOnePointVirasoroBlock(
        c,
        internal_weight,
        external_weight,
        b=b,
        external_lambda=external_lambda,
    ).elliptic_block(_as_complex(q), order)


def torus_one_point_chiral_block(
    c: complex | float,
    internal_weight: complex | float,
    external_weight: complex | float,
    q: complex | float,
    order: int,
    *,
    b: complex | float | None = None,
    external_lambda: complex | float | None = None,
    include_prefactor: bool = True,
) -> complex:
    return TorusOnePointVirasoroBlock(
        c,
        internal_weight,
        external_weight,
        b=b,
        external_lambda=external_lambda,
    ).chiral_block(_as_complex(q), order, include_prefactor=include_prefactor)


def torus_one_point_chiral_block_exact_eta(
    c: complex | float,
    internal_weight: complex | float,
    external_weight: complex | float,
    q: complex | float,
    order: int,
    *,
    b: complex | float | None = None,
    external_lambda: complex | float | None = None,
    eta_tolerance: float = 1.0e-15,
) -> complex:
    return TorusOnePointVirasoroBlock(
        c,
        internal_weight,
        external_weight,
        b=b,
        external_lambda=external_lambda,
    ).chiral_block_exact_eta(_as_complex(q), order, eta_tolerance=eta_tolerance)


def format_complex(z: complex) -> str:
    return f"{z.real:+.12e}{z.imag:+.12e}j"


def run() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a torus one-point Virasoro block.")
    parser.add_argument("--c", type=parse_complex, required=True)
    parser.add_argument("--internal-weight", type=parse_complex, required=True)
    parser.add_argument("--external-weight", type=parse_complex, required=True)
    parser.add_argument("--q", type=parse_complex, required=True)
    parser.add_argument("--order", type=int, default=8)
    parser.add_argument("--b", type=parse_complex)
    parser.add_argument("--external-lambda", type=parse_complex)
    args = parser.parse_args()

    block = TorusOnePointVirasoroBlock(
        args.c,
        args.internal_weight,
        args.external_weight,
        b=args.b,
        external_lambda=args.external_lambda,
    )
    print("torus one-point Virasoro block")
    print(f"  c={format_complex(block.params.c)}")
    print(f"  b={format_complex(block.params.b)}")
    print(f"  internal h={format_complex(block.params.internal_weight)}")
    print(f"  external h={format_complex(block.params.external_weight)}")
    print(f"  external lambda={format_complex(block.params.external_lambda)}")
    print(f"  q={format_complex(args.q)}")
    print(f"  order={args.order}")
    print("  elliptic H coefficients:")
    for n, coeff in enumerate(block.elliptic_coefficients(args.order)):
        print(f"    H[{n}]={format_complex(coeff)}")
    print(f"  H(q)={format_complex(block.elliptic_block(args.q, args.order))}")
    print(f"  F(q)={format_complex(block.chiral_block(args.q, args.order))}")


if __name__ == "__main__":
    run()
