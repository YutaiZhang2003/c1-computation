"""Pillow coordinates for the aligned real genus-zero comb cell."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import mpmath as mp

from .kac import as_mpmath


Number = Any


@dataclass(frozen=True)
class AlignedNomes:
    """The exact segment nomes obtained from an ordered real sphere point."""

    q: Number
    segment_nomes: tuple[Number, ...]
    right_products: tuple[Number, ...]
    coordinate_residuals: tuple[Number, ...]
    product_residual: Number


def elliptic_nome(z: Number) -> Number:
    r"""Return ``q=exp[-pi K(1-z)/K(z)]``."""

    z = as_mpmath(z)
    return mp.exp(-mp.pi * mp.ellipk(1 - z) / mp.ellipk(z))


def theta3_from_nome(q: Number) -> Number:
    return mp.jtheta(3, mp.mpf(0), as_mpmath(q))


def cross_ratio_from_nome(
    q: Number,
    *,
    tolerance: Number | None = None,
    max_terms: int = 10000,
) -> Number:
    r"""Evaluate ``z=16q prod_n[(1+q^(2n))/(1+q^(2n-1))]^8``."""

    q = as_mpmath(q)
    if abs(q) >= 1:
        raise ValueError("the elliptic product requires |q|<1")
    tolerance = (
        mp.power(10, -(mp.mp.dps - 8)) if tolerance is None else abs(as_mpmath(tolerance))
    )
    value: Number = 16 * q
    for n in range(1, int(max_terms) + 1):
        value *= ((1 + q ** (2 * n)) / (1 + q ** (2 * n - 1))) ** 8
        if abs(q) ** (2 * n - 1) < tolerance:
            return value
    raise RuntimeError("modular-lambda product did not converge")


def mobile_position_from_right_product(
    p_right: Number,
    q: Number,
    *,
    tolerance: Number | None = None,
    max_terms: int = 10000,
) -> Number:
    r"""Return the exact aligned position ``t=4 p_right Y(q/p_right,p_right)``."""

    p_right, q = as_mpmath(p_right), as_mpmath(q)
    p_left = q / p_right
    if max(abs(p_left), abs(p_right), abs(q)) >= 1:
        raise ValueError("the aligned product requires |p_left|,|p_right|,|q|<1")
    tolerance = (
        mp.power(10, -(mp.mp.dps - 8)) if tolerance is None else abs(as_mpmath(tolerance))
    )
    y: Number = (1 + p_left) ** 2
    for n in range(1, int(max_terms) + 1):
        q_odd = q ** (2 * n - 1)
        q_even = q ** (2 * n)
        y *= ((1 + q_even) / (1 + q_odd)) ** 4
        y *= (
            (1 + p_left ** (2 * n + 1) * p_right ** (2 * n))
            * (1 + p_left ** (2 * n - 1) * p_right ** (2 * n))
            / (
                (1 + p_left ** (2 * n) * p_right ** (2 * n - 1))
                * (1 + p_left ** (2 * n - 2) * p_right ** (2 * n - 1))
            )
        ) ** 2
        if max(
            abs(q_even), abs(p_left) ** (2 * n), abs(p_right) ** (2 * n)
        ) < tolerance:
            return 4 * p_right * y
    raise RuntimeError("aligned mobile-position product did not converge")


def coordinates_from_segment_nomes(
    segment_nomes: Sequence[Number],
) -> tuple[Number, tuple[Number, ...]]:
    """Map segment nomes to ``(z,(t_1,...,t_{n-4}))``."""

    nomes = tuple(as_mpmath(value) for value in segment_nomes)
    if not nomes:
        raise ValueError("at least one segment nome is required")
    q = mp.fprod(nomes)
    z = cross_ratio_from_nome(q)
    mobile_positions = []
    for split in range(1, len(nomes)):
        p_right = mp.fprod(nomes[split:])
        mobile_positions.append(mobile_position_from_right_product(p_right, q))
    return z, tuple(mobile_positions)


def _invert_one_mobile(t: Number, q: Number, bisections: int) -> tuple[Number, Number]:
    guard = mp.power(10, -(mp.mp.dps - 8))
    lower = q * (1 + guard)
    upper = 1 - guard
    for _ in range(bisections):
        midpoint = (lower + upper) / 2
        if mobile_position_from_right_product(midpoint, q) < t:
            lower = midpoint
        else:
            upper = midpoint
    p_right = (lower + upper) / 2
    residual = abs(mobile_position_from_right_product(p_right, q) - t)
    return p_right, residual


def invert_aligned_coordinates(
    z: Number,
    mobile_positions: Sequence[Number] = (),
    *,
    dps: int = 50,
    bisections: int | None = None,
) -> AlignedNomes:
    r"""Invert an ordered real sphere cell into exact segment nomes.

    The input must obey ``0<z<t_1<...<t_{n-4}<1``.  Complex continuation is
    intentionally not guessed; for complex moduli, supply segment nomes
    directly to the recursion and choose branches in the reconstruction.
    """

    dps = int(dps)
    if dps < 20:
        raise ValueError("dps must be at least 20")
    with mp.workdps(dps):
        z_value = as_mpmath(z)
        mobiles = tuple(as_mpmath(value) for value in mobile_positions)
        ordered = (z_value, *mobiles, mp.mpf(1))
        if any(mp.im(value) != 0 for value in ordered):
            raise ValueError("automatic inversion is restricted to the ordered real cell")
        if not all(ordered[index] < ordered[index + 1] for index in range(len(ordered) - 1)):
            raise ValueError("coordinates must obey 0<z<t_1<...<t_m<1")
        if z_value <= 0:
            raise ValueError("coordinates must obey 0<z<t_1<...<t_m<1")
        q = elliptic_nome(z_value)
        if not mobiles:
            return AlignedNomes(
                q=+q,
                segment_nomes=(+q,),
                right_products=(),
                coordinate_residuals=(),
                product_residual=mp.mpf(0),
            )
        steps = (
            int(math.ceil((dps + 8) * math.log2(10)))
            if bisections is None
            else int(bisections)
        )
        if steps < 10:
            raise ValueError("bisections must be at least 10")
        right_products = []
        residuals = []
        for t in mobiles:
            p_right, residual = _invert_one_mobile(t, q, steps)
            right_products.append(p_right)
            residuals.append(residual)
        if not all(
            right_products[index] < right_products[index + 1]
            for index in range(len(right_products) - 1)
        ):
            raise AssertionError("the inverse aggregate nomes are not ordered")
        nomes = [q / right_products[0]]
        for index in range(len(right_products) - 1):
            nomes.append(right_products[index] / right_products[index + 1])
        nomes.append(right_products[-1])
        if not all(0 < value < 1 for value in nomes):
            raise AssertionError("the inverse map left the real plumbing polydisc")
        product_residual = abs(mp.fprod(nomes) - q)
        return AlignedNomes(
            q=+q,
            segment_nomes=tuple(+value for value in nomes),
            right_products=tuple(+value for value in right_products),
            coordinate_residuals=tuple(+value for value in residuals),
            product_residual=+product_residual,
        )

