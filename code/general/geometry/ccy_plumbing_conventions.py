#!/usr/bin/env python3
"""Cho-Collier-Yin plumbing-frame sewing conventions.

CCY build higher-genus Virasoro blocks by plumbing holed spheres with
``SL(2,C)`` maps and contracting descendant three-point functions with inverse
Gram matrices.  For an internal primary of weight ``h`` and descendant level
``N``, the sewing operator contributes ``q^(h+N)``.  Their block convention
keeps the descendant power ``q^N`` inside the Virasoro block and separates the
primary factor ``q^h`` as an overall prefactor.

This module contains only that bookkeeping.  It deliberately does not include
cylinder Casimir factors or any conformal-frame anomaly.
"""

from __future__ import annotations

import cmath
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Callable
from typing import TypeVar


_T = TypeVar("_T")

THETA_GEOMETRY_EDGE_ORDER = ("q_zero", "q_one", "q_infinity")
THETA_CCY_EDGE_ORDER = ("q_infinity", "q_one", "q_zero")
GLASSES_GEOMETRY_EDGE_ORDER = ("q_left", "q_right", "q_bridge")
GENUS2_PLUMBING_COORDINATE_CONVENTION = "genus2-geometric-edge-order-v1"
GENUS2_PLUMBING_EDGE_ORDERS = {
    "theta": THETA_GEOMETRY_EDGE_ORDER,
    "glasses": GLASSES_GEOMETRY_EDGE_ORDER,
}
GENUS2_CFT_DESCENDANT_EDGE_ORDERS = {
    "theta": THETA_CCY_EDGE_ORDER,
    "glasses": GLASSES_GEOMETRY_EDGE_ORDER,
}
GENUS2_FREE_BOSON_EDGE_ORDERS = GENUS2_PLUMBING_EDGE_ORDERS
LOG_Q_BINARY64_SURROGATE_FLOOR = -690.0


@dataclass(frozen=True)
class Genus2PlumbingCoordinates:
    """One topology-labelled genus-two plumbing-coordinate contract."""

    channel: str
    edge_names: tuple[str, str, str]
    q_values: tuple[complex, complex, complex]
    log_q_values: tuple[complex, complex, complex]

    @property
    def named_q_values(self) -> dict[str, complex]:
        return dict(zip(self.edge_names, self.q_values))

    @property
    def named_log_q_values(self) -> dict[str, complex]:
        return dict(zip(self.edge_names, self.log_q_values))


def genus2_channel_ordered_values(
    channel: str,
    values: Iterable[_T] | Mapping[str, _T],
    *,
    label: str = "values",
) -> tuple[_T, _T, _T]:
    """Freeze three values in the declared geometric edge order."""

    topology = str(channel).strip().lower()
    try:
        edge_order = GENUS2_PLUMBING_EDGE_ORDERS[topology]
    except KeyError as exc:
        raise ValueError(f"unsupported genus-two channel {channel!r}") from exc
    if isinstance(values, Mapping):
        missing = [edge for edge in edge_order if edge not in values]
        extra = sorted(set(values) - set(edge_order))
        if missing or extra:
            raise ValueError(
                f"{label} keys do not match {topology} edges: "
                f"missing={missing}, extra={extra}"
            )
        items = tuple(values[edge] for edge in edge_order)
    else:
        items = tuple(values)
        if len(items) != 3:
            raise ValueError(
                f"{label} for {topology} must contain three entries in edge "
                f"order {edge_order}"
            )
    return items  # type: ignore[return-value]


def genus2_channel_q_values(
    channel: str,
    values: Iterable[complex] | Mapping[str, complex],
    *,
    label: str = "q_values",
) -> tuple[complex, complex, complex]:
    """Return complex plumbing parameters in the channel's geometric order."""

    ordered = genus2_channel_ordered_values(channel, values, label=label)
    return tuple(complex(value) for value in ordered)  # type: ignore[return-value]


def validate_genus2_plumbing_coordinates(
    channel: str,
    q_values: Iterable[complex] | Mapping[str, complex],
    *,
    log_q_values: Iterable[complex] | Mapping[str, complex] | None = None,
    relative_tolerance: float = 1.0e-8,
) -> Genus2PlumbingCoordinates:
    """Validate named edge order and the chosen logarithmic branch.

    For representable coordinates this enforces ``exp(log_q_e) == q_e``.
    Deep-cusp code may use a nonzero binary64 surrogate for an underflowed
    descendant coordinate.  In that case the surrogate must have the same
    phase as the true logarithm, be exponentially small, and be no smaller
    than the true (unrepresentable) magnitude.
    """

    topology = str(channel).strip().lower()
    q_tuple = genus2_channel_q_values(topology, q_values)
    edge_names = GENUS2_PLUMBING_EDGE_ORDERS[topology]
    if log_q_values is None:
        if any(value == 0.0j for value in q_tuple):
            raise ValueError("zero plumbing parameters require explicit log(q) values")
        log_q_tuple = tuple(cmath.log(value) for value in q_tuple)
    else:
        log_q_tuple = tuple(
            complex(value)
            for value in genus2_channel_ordered_values(
                topology,
                log_q_values,
                label="log_q_values",
            )
        )

    tolerance = float(relative_tolerance)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("relative_tolerance must be positive and finite")
    for edge_name, q_value, log_q in zip(edge_names, q_tuple, log_q_tuple):
        if not (
            math.isfinite(q_value.real)
            and math.isfinite(q_value.imag)
            and 0.0 < abs(q_value) < 1.0
        ):
            raise ValueError(
                f"{topology} edge {edge_name} must have finite 0 < |q| < 1"
            )
        if not (
            math.isfinite(log_q.real)
            and math.isfinite(log_q.imag)
            and log_q.real < 0.0
        ):
            raise ValueError(
                f"{topology} edge {edge_name} must have a finite log(q) "
                "with negative real part"
            )

        expected_phase = cmath.exp(1.0j * log_q.imag)
        actual_phase = q_value / abs(q_value)
        if abs(actual_phase - expected_phase) > 10.0 * tolerance:
            raise ValueError(
                f"{topology} edge {edge_name} q/log(q) phases disagree"
            )
        expected = cmath.exp(log_q)
        relative_error = abs(q_value - expected) / max(
            abs(q_value),
            abs(expected),
            1.0e-300,
        )
        if relative_error > tolerance:
            expected_surrogate = cmath.exp(
                complex(
                    max(log_q.real, LOG_Q_BINARY64_SURROGATE_FLOOR),
                    log_q.imag,
                )
            )
            surrogate_relative_error = abs(q_value - expected_surrogate) / max(
                abs(q_value),
                abs(expected_surrogate),
                1.0e-300,
            )
            valid_surrogate = (
                log_q.real < LOG_Q_BINARY64_SURROGATE_FLOOR
                and surrogate_relative_error <= tolerance
            )
            if not valid_surrogate:
                raise ValueError(
                    f"{topology} edge {edge_name} does not satisfy "
                    f"exp(log_q)=q (relative error {relative_error:.3e})"
                )

    return Genus2PlumbingCoordinates(
        channel=topology,
        edge_names=edge_names,
        q_values=q_tuple,
        log_q_values=log_q_tuple,  # type: ignore[arg-type]
    )


def genus2_plumbing_coordinate_metadata(
    coordinates: Genus2PlumbingCoordinates,
    *,
    formatter: Callable[[complex], str] = str,
) -> dict[str, str]:
    """Serialize both positional and explicitly named edge conventions."""

    metadata = {
        "plumbing_channel": coordinates.channel,
        "plumbing_edge_order": ",".join(coordinates.edge_names),
        "plumbing_coordinate_convention": (
            GENUS2_PLUMBING_COORDINATE_CONVENTION
        ),
        "cft_descendant_edge_order": ",".join(
            GENUS2_CFT_DESCENDANT_EDGE_ORDERS[coordinates.channel]
        ),
        "free_boson_edge_order": ",".join(
            GENUS2_FREE_BOSON_EDGE_ORDERS[coordinates.channel]
        ),
    }
    for index, (edge_name, q_value, log_q) in enumerate(
        zip(
            coordinates.edge_names,
            coordinates.q_values,
            coordinates.log_q_values,
        ),
        start=1,
    ):
        metadata[f"q{index}_edge_name"] = edge_name
        metadata[edge_name] = formatter(q_value)
        metadata[f"log_{edge_name}"] = formatter(log_q)
    return metadata


def theta_geometry_to_ccy_order(values: Iterable[_T]) -> tuple[_T, _T, _T]:
    r"""Map theta plumbing labels ``(zero, one, infinity)`` to CCY slot order.

    The Schottky/period-map API labels the three tubes by their punctures
    ``(0,1,infinity)``.  The CCY descendant tensor is ordered as
    ``(infinity,1,0)``.  Keeping this reversal explicit prevents the geometric
    ``q_zero`` tube from being attached to the infinity descendant slot.
    """

    items = tuple(values)
    if len(items) != 3:
        raise ValueError("theta edge ordering requires exactly three values")
    ordered = items[2], items[1], items[0]
    if THETA_CCY_EDGE_ORDER != tuple(reversed(THETA_GEOMETRY_EDGE_ORDER)):
        raise AssertionError("theta CCY edge-order contract is inconsistent")
    return ordered


def ccy_primary_propagator(q_values: Iterable[complex], weights: Iterable[complex]) -> complex:
    """Return the CCY separated primary sewing factor ``prod_e q_e^h_e``."""
    q_tuple = tuple(q_values)
    weight_tuple = tuple(weights)
    if len(q_tuple) != len(weight_tuple):
        raise ValueError("q_values and weights must have the same length")
    propagator = 1.0 + 0.0j
    for q_value, weight in zip(q_tuple, weight_tuple):
        propagator *= complex(q_value) ** complex(weight)
    return propagator


def ccy_raw_sewing_propagator(
    q_values: Iterable[complex],
    weights: Iterable[complex],
    *,
    diagnostic_shift: float = 0.0,
    log_q_values: Iterable[complex] | None = None,
) -> complex:
    """Return the raw CCY primary propagator, optionally with an explicit diagnostic shift.

    ``diagnostic_shift`` multiplies the CCY factor by ``prod_e q_e^(-shift)``.
    It is not part of the CCY plumbing-frame definition; it exists only for
    controlled normalization diagnostics.
    """
    return cmath.exp(
        ccy_raw_sewing_log_propagator(
            q_values,
            weights,
            diagnostic_shift=diagnostic_shift,
            log_q_values=log_q_values,
        )
    )


def ccy_raw_sewing_log_propagator(
    q_values: Iterable[complex],
    weights: Iterable[complex],
    *,
    diagnostic_shift: float = 0.0,
    log_q_values: Iterable[complex] | None = None,
) -> complex:
    r"""Return the logarithm of the raw CCY primary propagator.

    This is the production representation used when the propagator is
    multiplied by DOZZ constants.  Keeping

    ``sum_e (h_e-shift) log(q_e)``

    unexponentiated allows its large negative real part to cancel a large
    positive ``log(DOZZ)`` before either factor is formed.  The ordinary
    :func:`ccy_raw_sewing_propagator` API is now just one exponential of this
    function.
    """

    q_tuple = tuple(complex(q_value) for q_value in q_values)
    weight_tuple = tuple(complex(weight) for weight in weights)
    if len(q_tuple) != len(weight_tuple):
        raise ValueError("q_values and weights must have the same length")
    if log_q_values is not None:
        log_q_tuple = tuple(complex(value) for value in log_q_values)
        if len(log_q_tuple) != len(weight_tuple):
            raise ValueError("log_q_values and weights must have the same length")
        return sum(
            (weight - float(diagnostic_shift)) * log_q
            for weight, log_q in zip(weight_tuple, log_q_tuple)
        )
    return sum(
        (weight - float(diagnostic_shift)) * cmath.log(q_value)
        for q_value, weight in zip(q_tuple, weight_tuple)
    )


def liouville_threshold_weight(b: float) -> float:
    """Return the lower edge ``Q^2/4`` of the continuous Liouville weights."""
    q_background = float(b) + 1.0 / float(b)
    return 0.25 * q_background * q_background


def liouville_threshold_modulus_factor(q_values: Iterable[complex], *, b: float) -> float:
    """Return ``|prod_e q_e^(Q^2/4)|^2`` for the diagonal raw CCY integrand."""
    exponent = 2.0 * liouville_threshold_weight(b)
    factor = 1.0
    for q_value in q_values:
        factor *= abs(complex(q_value)) ** exponent
    return factor
