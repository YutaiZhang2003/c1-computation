#!/usr/bin/env python3
"""Momentum quadrature helpers for Liouville plumbing integrals."""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass
from numbers import Integral
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class MomentumQuadratureRule:
    """Quadrature rule for one Liouville momentum edge."""

    nodes: tuple[float, ...]
    weights: tuple[float, ...]
    split_point: float | None
    gaussian_width: float
    momentum_domain: str
    p_max_used: bool
    p_max: float | None
    laguerre_alpha: float | None = None
    effective_decay_coefficient: float | None = None
    effective_q_abs: float | None = None
    asymptotic_tempered: bool = False


@dataclass(frozen=True)
class CorrelatedMomentumPairRule:
    """Joint quadrature rule for two positive Liouville momenta."""

    first_nodes: tuple[float, ...]
    second_nodes: tuple[float, ...]
    weights: tuple[float, ...]
    radial_order: int
    angular_order: int
    first_decay_coefficient: float
    second_decay_coefficient: float
    radial_laguerre_alpha: float
    angular_jacobi_alpha: float
    angular_jacobi_beta: float
    momentum_domain: str = "[0,infinity)^2"


def normalize_quadrature_orders(
    q_values: Sequence[complex],
    quadrature_order: int | Sequence[int],
) -> tuple[int, ...]:
    """Return one positive quadrature order per plumbing edge."""

    edge_count = len(q_values)
    if isinstance(quadrature_order, Integral):
        orders = (int(quadrature_order),) * edge_count
    else:
        orders = tuple(int(order) for order in quadrature_order)
        if len(orders) != edge_count:
            raise ValueError(
                f"expected {edge_count} edge quadrature orders, got {len(orders)}"
            )
    if any(order <= 0 for order in orders):
        raise ValueError("quadrature orders must be positive")
    return orders


def gaussian_width_from_q(q_value: complex, *, log_q_abs: float | None = None) -> float:
    """Return the near-threshold width from ``exp(-2 |log q| P^2)``."""

    if log_q_abs is not None:
        log_q_abs = float(log_q_abs)
        if not math.isfinite(log_q_abs) or log_q_abs >= 0.0:
            raise ValueError("log|q| must be finite and negative")
        return float(1.0 / math.sqrt(-2.0 * log_q_abs))
    q_abs = abs(complex(q_value))
    if not 0.0 < q_abs < 1.0:
        raise ValueError("momentum width requires 0<|q|<1")
    return float(1.0 / math.sqrt(2.0 * abs(math.log(q_abs))))


def _interval_rule(start: float, stop: float, order: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if order <= 0:
        raise ValueError("quadrature order must be positive")
    if stop <= start:
        return (), ()
    nodes, weights = np.polynomial.legendre.leggauss(int(order))
    midpoint = 0.5 * (float(start) + float(stop))
    half_width = 0.5 * (float(stop) - float(start))
    mapped_nodes = tuple(float(midpoint + half_width * node) for node in nodes)
    mapped_weights = tuple(float(half_width * weight / math.pi) for weight in weights)
    return mapped_nodes, mapped_weights


def uniform_momentum_rule(p_max: float, quadrature_order: int) -> MomentumQuadratureRule:
    """Return the original global Gauss rule on ``[0, p_max]``."""

    if p_max <= 0.0:
        raise ValueError("p_max must be positive")
    nodes, weights = _interval_rule(0.0, float(p_max), int(quadrature_order))
    return MomentumQuadratureRule(
        nodes=nodes,
        weights=weights,
        split_point=None,
        gaussian_width=float("nan"),
        momentum_domain="[0,p_max]",
        p_max_used=True,
        p_max=float(p_max),
    )


def edge_scaled_momentum_rule(
    q_value: complex,
    p_max: float,
    near_order: int,
    *,
    tail_order: int | None = None,
    split_widths: float = 4.0,
    log_q_abs: float | None = None,
) -> MomentumQuadratureRule:
    """Return a two-piece rule adapted to the edge's primary propagator.

    The primary part of the nonchiral integrand contains
    ``exp(-2 |log |q|| P^2)``.  For very small plumbing parameters, most of the
    mass sits close to ``P=0``; a single low-order Gauss rule on ``[0,p_max]``
    can miss it.  This rule places one interval on ``[0, split]`` with
    ``split = split_widths / sqrt(2 |log |q||)`` and a coarser tail interval.
    """

    if p_max <= 0.0:
        raise ValueError("p_max must be positive")
    if split_widths <= 0.0:
        raise ValueError("split_widths must be positive")
    width = gaussian_width_from_q(q_value, log_q_abs=log_q_abs)
    split = min(float(p_max), float(split_widths) * width)
    if split >= float(p_max) * (1.0 - 1.0e-14):
        nodes, weights = _interval_rule(0.0, float(p_max), int(near_order))
        return MomentumQuadratureRule(
            nodes=nodes,
            weights=weights,
            split_point=None,
            gaussian_width=width,
            momentum_domain="[0,p_max]",
            p_max_used=True,
            p_max=float(p_max),
        )

    if tail_order is None:
        tail_order = max(2, int(near_order) // 2)
    near_nodes, near_weights = _interval_rule(0.0, split, int(near_order))
    tail_nodes, tail_weights = _interval_rule(split, float(p_max), int(tail_order))
    return MomentumQuadratureRule(
        nodes=near_nodes + tail_nodes,
        weights=near_weights + tail_weights,
        split_point=float(split),
        gaussian_width=width,
        momentum_domain="[0,p_max]",
        p_max_used=True,
        p_max=float(p_max),
    )


def generalized_gaussian_momentum_rule(
    q_value: complex,
    quadrature_order: int,
    *,
    laguerre_alpha: float,
    decay_coefficient: float,
    log_q_abs: float | None = None,
    effective_q_abs: float | None = None,
    asymptotic_tempered: bool = False,
) -> MomentumQuadratureRule:
    r"""Return a generalized-Laguerre rule for ``P in [0,infinity)``.

    The change of variables is ``P=sigma sqrt(u)`` with
    ``sigma=1/sqrt(decay_coefficient)``.  The returned weights undo the
    generalized-Laguerre weight ``u^alpha exp(-u)``, so callers continue to
    pass the complete momentum integrand.  This makes the rule exact as a
    quadrature transformation for any ``alpha > -1`` and any positive decay
    coefficient; choosing them only changes convergence, not the integral.
    """

    if quadrature_order <= 0:
        raise ValueError("quadrature order must be positive")
    laguerre_alpha = float(laguerre_alpha)
    decay_coefficient = float(decay_coefficient)
    if not math.isfinite(laguerre_alpha) or laguerre_alpha <= -1.0:
        raise ValueError("generalized-Laguerre alpha must be finite and greater than -1")
    if not math.isfinite(decay_coefficient) or decay_coefficient <= 0.0:
        raise ValueError("Gaussian decay coefficient must be positive and finite")
    try:
        from scipy.special import roots_genlaguerre
    except ImportError as exc:  # pragma: no cover - depends on optional SciPy
        raise ImportError(
            "primary-gaussian momentum quadrature requires SciPy; "
            "uniform and edge-scaled rules require only NumPy"
        ) from exc
    width = 1.0 / math.sqrt(decay_coefficient)
    u_nodes, u_weights = roots_genlaguerre(
        int(quadrature_order), laguerre_alpha
    )
    nodes = tuple(float(width * math.sqrt(float(node))) for node in u_nodes)
    weights = tuple(
        float(
            width
            * float(weight)
            * math.exp(float(node))
            * float(node) ** (-laguerre_alpha - 0.5)
            / (2.0 * math.pi)
        )
        for node, weight in zip(u_nodes, u_weights)
    )
    return MomentumQuadratureRule(
        nodes=nodes,
        weights=weights,
        split_point=None,
        gaussian_width=width,
        momentum_domain="[0,infinity)",
        p_max_used=False,
        p_max=None,
        laguerre_alpha=laguerre_alpha,
        effective_decay_coefficient=decay_coefficient,
        effective_q_abs=effective_q_abs,
        asymptotic_tempered=bool(asymptotic_tempered),
    )


def primary_gaussian_momentum_rule(
    q_value: complex,
    quadrature_order: int,
    *,
    log_q_abs: float | None = None,
) -> MomentumQuadratureRule:
    r"""Return the production rule matched to ``exp(-2|log q| P^2)``."""

    log_abs = (
        float(log_q_abs)
        if log_q_abs is not None
        else math.log(abs(complex(q_value)))
    )
    return generalized_gaussian_momentum_rule(
        q_value,
        quadrature_order,
        laguerre_alpha=-0.5,
        decay_coefficient=-2.0 * log_abs,
        log_q_abs=log_q_abs,
        effective_q_abs=(math.exp(log_abs) if log_abs > -745.0 else 0.0),
    )


def threshold_gaussian_momentum_rule(
    q_value: complex,
    quadrature_order: int,
    *,
    log_q_abs: float | None = None,
) -> MomentumQuadratureRule:
    r"""Use the exact b=1 DOZZ threshold zero in the Laguerre weight.

    Each edge contributes ``P^2`` to the full genus-two b=1 DOZZ measure, so
    ``alpha=+1/2`` makes the transformed residual finite and generically
    nonzero at ``P=0``.  The primary Gaussian decay remains unchanged.
    """

    log_abs = (
        float(log_q_abs)
        if log_q_abs is not None
        else math.log(abs(complex(q_value)))
    )
    return generalized_gaussian_momentum_rule(
        q_value,
        quadrature_order,
        laguerre_alpha=0.5,
        decay_coefficient=-2.0 * log_abs,
        log_q_abs=log_q_abs,
        effective_q_abs=(math.exp(log_abs) if log_abs > -745.0 else 0.0),
    )


def threshold_polar_momentum_pair_rule(
    q_first: complex,
    q_second: complex,
    radial_order: int,
    angular_order: int,
    *,
    log_q_abs_first: float | None = None,
    log_q_abs_second: float | None = None,
    angular_jacobi_alpha: float = 0.5,
    angular_jacobi_beta: float = 0.5,
) -> CorrelatedMomentumPairRule:
    r"""Return a joint threshold rule in radial-angular scaled coordinates.

    Write ``x_i=sqrt(a_i) P_i`` with ``a_i=-2 log|q_i|`` and transform

    ``v=x_first^2+x_second^2``,
    ``y=(x_first^2-x_second^2)/v``.

    The exact threshold-Gaussian factor and Jacobian become

    ``v^2 exp(-v) sqrt(1-y^2) dv dy``.

    Generalized Laguerre with ``alpha=2`` therefore handles the common radial
    direction.  The default Gauss-Jacobi exponents ``alpha=beta=1/2`` absorb
    the threshold angular factor.  Different exponents can shift and narrow
    the angular proposal around a known DOZZ ridge.  The returned weights undo
    both reference weights and include ``1/pi^2``, so callers evaluate the
    complete, unmodified two-momentum integrand.  This is an exact change of
    variables; correlation adaptation changes convergence only.
    """

    radial_order = int(radial_order)
    angular_order = int(angular_order)
    if radial_order <= 0 or angular_order <= 0:
        raise ValueError("radial and angular orders must be positive")
    angular_jacobi_alpha = float(angular_jacobi_alpha)
    angular_jacobi_beta = float(angular_jacobi_beta)
    if (
        not math.isfinite(angular_jacobi_alpha)
        or not math.isfinite(angular_jacobi_beta)
        or angular_jacobi_alpha <= -1.0
        or angular_jacobi_beta <= -1.0
    ):
        raise ValueError("angular Jacobi exponents must be finite and greater than -1")

    def decay(q_value: complex, supplied_log_abs: float | None) -> float:
        log_abs = (
            float(supplied_log_abs)
            if supplied_log_abs is not None
            else math.log(abs(complex(q_value)))
        )
        coefficient = -2.0 * log_abs
        if not math.isfinite(coefficient) or coefficient <= 0.0:
            raise ValueError("polar momentum rule requires finite 0<|q|<1")
        return coefficient

    first_decay = decay(q_first, log_q_abs_first)
    second_decay = decay(q_second, log_q_abs_second)
    try:
        from scipy.special import roots_genlaguerre, roots_jacobi
    except ImportError as exc:  # pragma: no cover - depends on optional SciPy
        raise ImportError(
            "threshold-polar momentum quadrature requires SciPy"
        ) from exc

    radial_nodes, radial_weights = roots_genlaguerre(radial_order, 2.0)
    angular_nodes, angular_weights = roots_jacobi(
        angular_order, angular_jacobi_alpha, angular_jacobi_beta
    )
    first_nodes: list[float] = []
    second_nodes: list[float] = []
    weights: list[float] = []
    decay_jacobian = math.sqrt(first_decay * second_decay)
    for radial_node, radial_weight in zip(radial_nodes, radial_weights):
        v = float(radial_node)
        radial_undo = float(radial_weight) * math.exp(v) / (v * v)
        for angular_node, angular_weight in zip(angular_nodes, angular_weights):
            y = float(angular_node)
            first_x = math.sqrt(0.5 * v * (1.0 + y))
            second_x = math.sqrt(0.5 * v * (1.0 - y))
            first_nodes.append(first_x / math.sqrt(first_decay))
            second_nodes.append(second_x / math.sqrt(second_decay))
            weights.append(
                radial_undo
                * float(angular_weight)
                / (
                    4.0
                    * math.pi**2
                    * decay_jacobian
                    * (1.0 - y) ** (angular_jacobi_alpha + 0.5)
                    * (1.0 + y) ** (angular_jacobi_beta + 0.5)
                )
            )
    return CorrelatedMomentumPairRule(
        first_nodes=tuple(first_nodes),
        second_nodes=tuple(second_nodes),
        weights=tuple(weights),
        radial_order=radial_order,
        angular_order=angular_order,
        first_decay_coefficient=first_decay,
        second_decay_coefficient=second_decay,
        radial_laguerre_alpha=2.0,
        angular_jacobi_alpha=angular_jacobi_alpha,
        angular_jacobi_beta=angular_jacobi_beta,
    )


def cft_asymptotic_momentum_rule(
    q_value: complex,
    quadrature_order: int,
    *,
    log_q_abs: float | None = None,
    minimum_decay_fraction: float = 0.75,
) -> MomentumQuadratureRule:
    r"""Experimental threshold- and heavy-block-adapted momentum rule.

    The diagonal global-block saddle replaces ``q`` by

    ``q_eff = q * (2 / (1 + sqrt(1-q)))^2``.

    The full DOZZ asymptotic is not yet included.  When the block-only model
    would remove too much decay, the proposal is therefore tempered to retain
    ``minimum_decay_fraction`` of the primary Gaussian coefficient.  The
    returned rule still integrates the original, unmodified integrand.
    """

    minimum_decay_fraction = float(minimum_decay_fraction)
    if not 0.0 < minimum_decay_fraction <= 1.0:
        raise ValueError("minimum decay fraction must lie in (0,1]")
    q_value = complex(q_value)
    q_abs = abs(q_value)
    if not 0.0 < q_abs < 1.0:
        raise ValueError("CFT-asymptotic momentum rule requires 0<|q|<1")
    primary_log_abs = (
        float(log_q_abs) if log_q_abs is not None else math.log(q_abs)
    )
    square_root = cmath.sqrt(1.0 - q_value)
    if square_root.real < 0.0:
        square_root = -square_root
    block_ratio = (2.0 / (1.0 + square_root)) ** 2
    effective_log_abs = primary_log_abs + math.log(abs(block_ratio))
    primary_decay = -2.0 * primary_log_abs
    block_decay = -2.0 * effective_log_abs
    decay_floor = minimum_decay_fraction * primary_decay
    decay_coefficient = max(decay_floor, block_decay)
    tempered = block_decay < decay_floor
    return generalized_gaussian_momentum_rule(
        q_value,
        quadrature_order,
        laguerre_alpha=0.5,
        decay_coefficient=decay_coefficient,
        log_q_abs=log_q_abs,
        effective_q_abs=(
            math.exp(effective_log_abs) if effective_log_abs > -745.0 else 0.0
        ),
        asymptotic_tempered=tempered,
    )


def momentum_quadrature_rules(
    q_values: Sequence[complex],
    *,
    p_max: float | None,
    quadrature_order: int | Sequence[int],
    quadrature_scheme: str = "uniform",
    tail_order: int | None = None,
    split_widths: float = 4.0,
    log_q_abs_values: Sequence[float] | None = None,
) -> tuple[MomentumQuadratureRule, ...]:
    """Return one momentum quadrature rule per plumbing edge.

    ``p_max`` is required by the finite-interval schemes.  The Gaussian
    schemes integrate ``[0,infinity)`` and explicitly record
    ``p_max_used=False`` and ``p_max=None`` on every returned rule; any
    supplied value is ignored by this low-level constructor.
    """

    orders = normalize_quadrature_orders(q_values, quadrature_order)
    if log_q_abs_values is None:
        log_abs_values: tuple[float | None, ...] = (None,) * len(q_values)
    else:
        supplied = tuple(float(value) for value in log_q_abs_values)
        if len(supplied) != len(q_values):
            raise ValueError(
                f"expected {len(q_values)} log|q| values, got {len(supplied)}"
            )
        log_abs_values = supplied

    if quadrature_scheme == "uniform":
        if p_max is None:
            raise ValueError("uniform momentum quadrature requires p_max")
        return tuple(
            uniform_momentum_rule(float(p_max), order) for order in orders
        )
    if quadrature_scheme == "edge-scaled":
        if p_max is None:
            raise ValueError("edge-scaled momentum quadrature requires p_max")
        return tuple(
            edge_scaled_momentum_rule(
                value,
                float(p_max),
                order,
                tail_order=tail_order,
                split_widths=split_widths,
                log_q_abs=log_q_abs,
            )
            for value, order, log_q_abs in zip(q_values, orders, log_abs_values)
        )
    if quadrature_scheme == "primary-gaussian":
        return tuple(
            primary_gaussian_momentum_rule(value, order, log_q_abs=log_q_abs)
            for value, order, log_q_abs in zip(q_values, orders, log_abs_values)
        )
    if quadrature_scheme == "threshold-gaussian":
        return tuple(
            threshold_gaussian_momentum_rule(value, order, log_q_abs=log_q_abs)
            for value, order, log_q_abs in zip(q_values, orders, log_abs_values)
        )
    if quadrature_scheme == "cft-asymptotic":
        return tuple(
            cft_asymptotic_momentum_rule(value, order, log_q_abs=log_q_abs)
            for value, order, log_q_abs in zip(q_values, orders, log_abs_values)
        )
    raise ValueError(f"unsupported quadrature scheme {quadrature_scheme!r}")
