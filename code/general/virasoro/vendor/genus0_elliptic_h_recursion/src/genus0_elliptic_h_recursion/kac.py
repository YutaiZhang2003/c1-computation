"""Kac weights, Zamolodchikov residues, and fusion polynomials."""

from __future__ import annotations

from typing import Any

import mpmath as mp


Number = Any


def as_mpmath(value: Number) -> Number:
    """Convert strings and Python numeric values to an mpmath number."""

    return mp.mpmathify(value)


def background_data(central_charge: Number) -> tuple[Number, Number]:
    r"""Return ``(Q,b)`` with ``c=1+6 Q^2`` and ``Q=b+b^{-1}``."""

    central_charge = as_mpmath(central_charge)
    q_background = mp.sqrt((central_charge - 1) / 6)
    b = (q_background + mp.sqrt(q_background**2 - 4)) / 2
    return q_background, b


def degenerate_weight(
    alpha: int,
    beta: int,
    q_background: Number,
    b: Number,
) -> Number:
    r"""Return ``h_{alpha,beta}=(Q^2-(alpha*b+beta/b)^2)/4``."""

    momentum = alpha * b + beta / b
    return (q_background**2 - momentum**2) / 4


def zamolodchikov_a(alpha: int, beta: int, b: Number) -> Number:
    r"""Return the standard inverse-null-norm factor ``A_{alpha,beta}``."""

    value = mp.mpf("0.5")
    for p in range(1 - alpha, alpha + 1):
        for ell in range(1 - beta, beta + 1):
            if (p, ell) in {(0, 0), (alpha, beta)}:
                continue
            value /= p * b + ell / b
    return value


def fusion_polynomial(
    alpha: int,
    beta: int,
    *,
    top: Number,
    bottom: Number,
    q_background: Number,
    b: Number,
) -> Number:
    r"""Return the Virasoro fusion polynomial in weight variables.

    The momentum branch drops out of the complete polynomial.  The mpmath
    principal square root is used consistently for numerical evaluation.
    """

    lambda_top = mp.sqrt(q_background**2 - 4 * top)
    lambda_bottom = mp.sqrt(q_background**2 - 4 * bottom)
    value: Number = mp.mpf(1)
    for p in range(1 - alpha, alpha, 2):
        for ell in range(1 - beta, beta, 2):
            shift = p * b + ell / b
            value *= (lambda_top + lambda_bottom + shift) / 2
            value *= (lambda_top - lambda_bottom + shift) / 2
    return value
