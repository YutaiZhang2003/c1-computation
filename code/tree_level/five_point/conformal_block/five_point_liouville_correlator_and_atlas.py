#!/usr/bin/env python3
"""Five-point c=25 Liouville correlator in the CCY linear-channel atlas.

The module supplies three layers needed by the sphere ``1->4`` integrand:

* projectively stable Mobius maps, including punctures at infinity;
* the fifteen trivalent-tree channels of ``Mbar_{0,5}``, with orientation
  chosen to minimize the two CCY plumbing parameters;
* the double Liouville-momentum integral of three BRY-normalized DOZZ
  constants times the regulated ``c=25`` five-point block.

The antiholomorphic block uses the same analytically continued weights at
``conjugate(q)``, not complex-conjugated coefficients.  This distinction is
essential when the external energies carry the ``i epsilon`` prescription.
"""

from __future__ import annotations

Particle = int

from c1_paths import PACKAGE_ROOT, data_root, package_path, module_path

import cmath
import itertools
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np

try:
    from five_point_virasoro_block_recursion import (
        evaluate_sphere_five_point_series,
        sphere_five_point_c_coefficients,
        sphere_five_point_h_c25_limit,
    )
    from liouville_torus import UpsilonB, yin_structure_constant_momentum
except ImportError:  # pragma: no cover - supports package-style execution
    from five_point_virasoro_block_recursion import (
        evaluate_sphere_five_point_series,
        sphere_five_point_c_coefficients,
        sphere_five_point_h_c25_limit,
    )
    from liouville_torus import UpsilonB, yin_structure_constant_momentum


# Re-export the historical names for existing five-point callers.
from sphere_mobius_geometry import (
    INFINITY,
    ProjectivePoint,
    _homogeneous,
    _determinant,
    MobiusMap,
    mobius_to_zero_one_infinity,
)


from typing import Mapping, Sequence

@dataclass(frozen=True, order=True)
class BoundaryDivisor:
    """A stable 2+3 partition of five labels, named by its two-label side."""

    cherry: tuple[Particle, Particle]

    def __post_init__(self) -> None:
        normalized = tuple(sorted(int(label) for label in self.cherry))
        if len(normalized) != 2 or normalized[0] == normalized[1]:
            raise ValueError("a boundary cherry must contain two distinct labels")
        object.__setattr__(self, "cherry", normalized)

    def is_compatible_with(self, other: "BoundaryDivisor") -> bool:
        """Return whether the two divisors meet in ``Mbar_{0,5}``."""

        return set(self.cherry).isdisjoint(other.cherry)


@dataclass(frozen=True, order=True)
class BoundaryCorner:
    """A trivalent five-leaf tree, equivalently two disjoint cherries."""

    divisors: tuple[BoundaryDivisor, BoundaryDivisor]
    middle_label: Particle


def five_point_boundary_divisors(labels: Sequence[Particle] = range(5)) -> tuple[BoundaryDivisor, ...]:
    """Return all ten boundary divisors of ``Mbar_{0,5}``."""

    normalized = tuple(int(label) for label in labels)
    if len(normalized) != 5 or len(set(normalized)) != 5:
        raise ValueError("labels must contain five distinct particle labels")
    return tuple(BoundaryDivisor(pair) for pair in itertools.combinations(normalized, 2))


def five_point_boundary_corners(labels: Sequence[Particle] = range(5)) -> tuple[BoundaryCorner, ...]:
    """Return all fifteen compatible divisor pairs (five-leaf trees)."""

    normalized = tuple(int(label) for label in labels)
    divisors = five_point_boundary_divisors(normalized)
    label_set = set(normalized)
    corners: list[BoundaryCorner] = []
    for left_index, left in enumerate(divisors):
        for right in divisors[left_index + 1 :]:
            if not left.is_compatible_with(right):
                continue
            remaining = label_set - set(left.cherry) - set(right.cherry)
            if len(remaining) != 1:
                raise AssertionError("compatible five-point cherries must leave one label")
            corners.append(
                BoundaryCorner(
                    divisors=(left, right),
                    middle_label=next(iter(remaining)),
                )
            )
    return tuple(corners)


def signed_channel_energy(
    divisor: BoundaryDivisor,
    signed_energies: Mapping[Particle, complex] | Sequence[complex],
) -> complex:
    """Return the signed target-time energy flowing through a divisor."""

    if isinstance(signed_energies, Mapping):
        return complex(sum(complex(signed_energies[label]) for label in divisor.cherry))
    return complex(sum(complex(signed_energies[label]) for label in divisor.cherry))


def equal_outgoing_signed_energies(omega: complex) -> tuple[complex, ...]:
    """Return ``(+4 omega,-omega,-omega,-omega,-omega)`` for ``1->4``."""

    omega = complex(omega)
    return (4.0 * omega, -omega, -omega, -omega, -omega)


def divergent_spin_zero_levels(channel_energy: complex) -> tuple[int, ...]:
    """Return descendant levels whose momentum interval contains divergences."""

    threshold = 0.25 * complex(channel_energy) ** 2
    real_threshold = threshold.real
    if real_threshold <= 0.0:
        return ()
    # Strict inequality is required.  A level exactly at the endpoint has a
    # zero-length P interval and therefore contributes nothing.
    upper = max(0, int(math.ceil(real_threshold) - 1))
    return tuple(level for level in range(upper + 1) if level < real_threshold)


@dataclass(frozen=True)
class LinearChannel:
    """One oriented five-leaf linear channel in the CCY frame."""

    ordering: tuple[int, int, int, int, int]
    positions: tuple[complex, complex, complex, complex, None]
    q1: complex
    q2: complex
    mobius: MobiusMap
    local_scales: tuple[complex, complex, complex, complex, complex]
    score: float

    @property
    def convergence_score(self) -> float:
        """Radial gauge of the nested five-point CCY bidisk."""

        return self.score

    @property
    def left_cherry(self) -> tuple[int, int]:
        return self.ordering[0], self.ordering[1]

    @property
    def right_cherry(self) -> tuple[int, int]:
        return self.ordering[3], self.ordering[4]

    @property
    def middle_label(self) -> int:
        return self.ordering[2]


def _oriented_tree_orderings(labels: Sequence[int] = range(5)) -> Iterable[tuple[int, ...]]:
    """Yield the 120 oriented representatives of the fifteen tree channels."""

    for corner in five_point_boundary_corners(labels):
        left_base = corner.divisors[0].cherry
        right_base = corner.divisors[1].cherry
        for swap_sides in (False, True):
            left, right = (right_base, left_base) if swap_sides else (left_base, right_base)
            for swap_left in (False, True):
                oriented_left = left[::-1] if swap_left else left
                for swap_right in (False, True):
                    oriented_right = right[::-1] if swap_right else right
                    yield (
                        oriented_left[0],
                        oriented_left[1],
                        corner.middle_label,
                        oriented_right[0],
                        oriented_right[1],
                    )


def oriented_tree_orderings(labels: Sequence[int] = range(5)) -> tuple[tuple[int, ...], ...]:
    """Return all 120 oriented linear frames of the fifteen tree channels."""

    return _oriented_tree_orderings_cached(tuple(int(label) for label in labels))


@lru_cache(maxsize=None)
def _oriented_tree_orderings_cached(
    labels: tuple[int, ...],
) -> tuple[tuple[int, ...], ...]:
    return tuple(_oriented_tree_orderings(labels))


@dataclass(frozen=True)
class LinearChannelForwardMap:
    """The original ``(z_in,z_out)`` chart and its holomorphic Jacobian."""

    positions: tuple[complex, complex, complex, complex, None]
    complex_jacobian: complex

    @property
    def area_jacobian(self) -> float:
        return float(abs(self.complex_jacobian) ** 2)


@dataclass(frozen=True)
class _Dual2:
    value: complex
    derivative1: complex = 0.0 + 0.0j
    derivative2: complex = 0.0 + 0.0j

    def __add__(self, other: object) -> "_Dual2":
        other_dual = _as_dual(other)
        return _Dual2(
            self.value + other_dual.value,
            self.derivative1 + other_dual.derivative1,
            self.derivative2 + other_dual.derivative2,
        )

    __radd__ = __add__

    def __neg__(self) -> "_Dual2":
        return _Dual2(-self.value, -self.derivative1, -self.derivative2)

    def __sub__(self, other: object) -> "_Dual2":
        return self + (-_as_dual(other))

    def __rsub__(self, other: object) -> "_Dual2":
        return _as_dual(other) - self

    def __mul__(self, other: object) -> "_Dual2":
        other_dual = _as_dual(other)
        return _Dual2(
            self.value * other_dual.value,
            self.derivative1 * other_dual.value + self.value * other_dual.derivative1,
            self.derivative2 * other_dual.value + self.value * other_dual.derivative2,
        )

    __rmul__ = __mul__

    def __truediv__(self, other: object) -> "_Dual2":
        other_dual = _as_dual(other)
        if other_dual.value == 0.0:
            raise ZeroDivisionError("the linear-channel forward map hit a boundary")
        inverse = 1.0 / other_dual.value
        return _Dual2(
            self.value * inverse,
            (self.derivative1 - self.value * inverse * other_dual.derivative1) * inverse,
            (self.derivative2 - self.value * inverse * other_dual.derivative2) * inverse,
        )

    def __rtruediv__(self, other: object) -> "_Dual2":
        return _as_dual(other) / self


def _as_dual(value: object) -> _Dual2:
    if isinstance(value, _Dual2):
        return value
    return _Dual2(complex(value))


def _dual_determinant(
    left: tuple[_Dual2, _Dual2],
    right: tuple[_Dual2, _Dual2],
) -> _Dual2:
    return left[0] * right[1] - left[1] * right[0]


def linear_channel_to_original_chart(
    q1: complex,
    q2: complex,
    ordering: Sequence[int],
) -> LinearChannelForwardMap:
    r"""Map an oriented CCY bidisc to ``(z_in,z_out,0,1,infinity)``.

    The returned complex Jacobian is
    ``det d(z_in,z_out)/d(q1,q2)``.  Its squared modulus is the four-real-
    dimensional area Jacobian.
    """

    ordering_tuple = tuple(int(label) for label in ordering)
    if len(ordering_tuple) != 5 or set(ordering_tuple) != set(range(5)):
        raise ValueError("ordering must be a permutation of labels 0,...,4")
    q1_dual = _Dual2(complex(q1), 1.0 + 0.0j, 0.0 + 0.0j)
    q2_dual = _Dual2(complex(q2), 0.0 + 0.0j, 1.0 + 0.0j)
    x_dual = q1_dual * q2_dual
    y_dual = q2_dual
    zero = (_Dual2(0.0), _Dual2(1.0))
    x_point = (x_dual, _Dual2(1.0))
    y_point = (y_dual, _Dual2(1.0))
    one = (_Dual2(1.0), _Dual2(1.0))
    infinity = (_Dual2(1.0), _Dual2(0.0))
    by_label = {
        label: point
        for label, point in zip(ordering_tuple, (zero, x_point, y_point, one, infinity))
    }
    fixed_zero = by_label[2]
    fixed_one = by_label[3]
    fixed_infinity = by_label[4]
    normalization = _dual_determinant(fixed_one, fixed_infinity) / _dual_determinant(
        fixed_one, fixed_zero
    )

    def transform(point: tuple[_Dual2, _Dual2]) -> _Dual2:
        return (
            _dual_determinant(point, fixed_zero)
            / _dual_determinant(point, fixed_infinity)
            * normalization
        )

    z_incoming = transform(by_label[0])
    z_outgoing = transform(by_label[1])
    jacobian = (
        z_incoming.derivative1 * z_outgoing.derivative2
        - z_incoming.derivative2 * z_outgoing.derivative1
    )
    return LinearChannelForwardMap(
        positions=(
            complex(z_incoming.value),
            complex(z_outgoing.value),
            0.0 + 0.0j,
            1.0 + 0.0j,
            None,
        ),
        complex_jacobian=complex(jacobian),
    )


def linear_channel_positions_by_label(
    q1: complex,
    q2: complex,
    ordering: Sequence[int],
) -> tuple[ProjectivePoint, ...]:
    """Return label-ordered positions in the channel's own ``0,x,y,1,inf`` gauge."""

    ordering_tuple = tuple(int(label) for label in ordering)
    if len(ordering_tuple) != 5 or set(ordering_tuple) != set(range(5)):
        raise ValueError("ordering must be a permutation of labels 0,...,4")
    by_label: list[ProjectivePoint] = [None] * 5
    for label, point in zip(
        ordering_tuple,
        (0.0 + 0.0j, complex(q1) * complex(q2), complex(q2), 1.0 + 0.0j, None),
    ):
        by_label[label] = point
    return tuple(by_label)


def linear_channel_complex_jacobian_to_chart(
    q1: complex,
    q2: complex,
    ordering: Sequence[int],
    *,
    fixed_zero: int,
    fixed_one: int,
    fixed_infinity: int,
    moving_labels: Sequence[int],
) -> complex:
    """Return ``det d(z_m1,z_m2)/d(q1,q2)`` in an arbitrary fixed-label gauge."""

    ordering_tuple = tuple(int(label) for label in ordering)
    moving = tuple(int(label) for label in moving_labels)
    fixed = (int(fixed_zero), int(fixed_one), int(fixed_infinity))
    if len(ordering_tuple) != 5 or set(ordering_tuple) != set(range(5)):
        raise ValueError("ordering must be a permutation of labels 0,...,4")
    if len(moving) != 2 or len(set(moving)) != 2:
        raise ValueError("moving_labels must contain two distinct labels")
    if len(set(fixed)) != 3 or set(fixed) & set(moving) or set(fixed) | set(moving) != set(range(5)):
        raise ValueError("fixed and moving labels must partition labels 0,...,4")

    q1_dual = _Dual2(complex(q1), 1.0 + 0.0j, 0.0 + 0.0j)
    q2_dual = _Dual2(complex(q2), 0.0 + 0.0j, 1.0 + 0.0j)
    x_dual = q1_dual * q2_dual
    y_dual = q2_dual
    zero = (_Dual2(0.0), _Dual2(1.0))
    x_point = (x_dual, _Dual2(1.0))
    y_point = (y_dual, _Dual2(1.0))
    one = (_Dual2(1.0), _Dual2(1.0))
    infinity = (_Dual2(1.0), _Dual2(0.0))
    by_label = {
        label: point
        for label, point in zip(ordering_tuple, (zero, x_point, y_point, one, infinity))
    }
    zero_point = by_label[fixed[0]]
    one_point = by_label[fixed[1]]
    infinity_point = by_label[fixed[2]]
    normalization = _dual_determinant(one_point, infinity_point) / _dual_determinant(
        one_point, zero_point
    )

    def transform(point: tuple[_Dual2, _Dual2]) -> _Dual2:
        return (
            _dual_determinant(point, zero_point)
            / _dual_determinant(point, infinity_point)
            * normalization
        )

    first = transform(by_label[moving[0]])
    second = transform(by_label[moving[1]])
    return complex(
        first.derivative1 * second.derivative2
        - first.derivative2 * second.derivative1
    )


def linear_channel_from_ordering(
    positions: Sequence[ProjectivePoint],
    ordering: Sequence[int],
) -> LinearChannel:
    """Construct one CCY channel from an external-label ordering."""

    if len(positions) != 5 or len(ordering) != 5:
        raise ValueError("positions and ordering must each contain five entries")
    ordering_tuple = tuple(int(label) for label in ordering)
    if set(ordering_tuple) != set(range(5)):
        raise ValueError("ordering must be a permutation of labels 0,...,4")
    a, b, c, d, e = ordering_tuple
    transform = mobius_to_zero_one_infinity(positions[a], positions[d], positions[e])
    x = transform(positions[b])
    y = transform(positions[c])
    if x is None or y is None or y == 0.0:
        raise ValueError("the ordered channel is degenerate")
    q1 = complex(x / y)
    q2 = complex(y)
    transformed = (0.0 + 0.0j, complex(x), complex(y), 1.0 + 0.0j, None)
    scales = tuple(transform.local_scale(positions[label]) for label in ordering_tuple)
    return LinearChannel(
        ordering=ordering_tuple,
        positions=transformed,
        q1=q1,
        q2=q2,
        mobius=transform,
        local_scales=scales,
        score=float(max(abs(q1), abs(q2))),
    )


def best_linear_channels(
    positions: Sequence[ProjectivePoint],
    *,
    limit: int = 2,
    convergence_radius: float = 1.0,
) -> tuple[LinearChannel, ...]:
    """Return the best distinct oriented tree channels for one configuration."""

    if len(positions) != 5:
        raise ValueError("positions must contain five punctures")
    limit = int(limit)
    if limit <= 0:
        raise ValueError("limit must be positive")
    best_by_tree: dict[tuple[tuple[int, int], tuple[int, int]], LinearChannel] = {}
    for ordering in oriented_tree_orderings():
        try:
            channel = linear_channel_from_ordering(positions, ordering)
        except (ValueError, ZeroDivisionError):
            continue
        if channel.score >= float(convergence_radius):
            continue
        signature = tuple(
            sorted(
                (
                    tuple(sorted(channel.left_cherry)),
                    tuple(sorted(channel.right_cherry)),
                )
            )
        )
        previous = best_by_tree.get(signature)
        if previous is None or channel.score < previous.score:
            best_by_tree[signature] = channel
    candidates = list(best_by_tree.values())
    candidates.sort(key=lambda item: item.score)
    if not candidates:
        raise RuntimeError("no convergent five-point linear channel was found")
    return tuple(candidates[:limit])


def liouville_weights(external_momenta: Sequence[complex]) -> tuple[complex, ...]:
    """Return the ``c=25`` weights ``d_i=1+P_i^2``."""

    if len(external_momenta) != 5:
        raise ValueError("external_momenta must contain five values")
    return tuple(1.0 + complex(momentum) ** 2 for momentum in external_momenta)


def liouville_primary_covariance(
    channel: LinearChannel,
    external_weights: Sequence[complex],
) -> complex:
    """Return the nonchiral primary covariance from the original chart."""

    if len(external_weights) != 5:
        raise ValueError("external_weights must contain five values")
    return complex(cmath.exp(liouville_primary_covariance_log(channel, external_weights)))


def liouville_primary_covariance_log(
    channel: LinearChannel,
    external_weights: Sequence[complex],
) -> complex:
    """Return the logarithm of the nonchiral primary covariance factor."""

    if len(external_weights) != 5:
        raise ValueError("external_weights must contain five values")
    ordered_weights = tuple(complex(external_weights[label]) for label in channel.ordering)
    logarithm = 0.0 + 0.0j
    for scale, weight in zip(channel.local_scales, ordered_weights):
        absolute_scale = abs(scale)
        if absolute_scale == 0.0:
            raise ZeroDivisionError("a channel has a vanishing puncture scale")
        logarithm += 2.0 * weight * math.log(absolute_scale)
    return complex(logarithm)


def timelike_free_boson_factor(
    positions: Sequence[ProjectivePoint],
    signed_energies: Sequence[complex],
) -> complex:
    r"""Return ``prod_finite |z_i-z_j|^{-k_i k_j}`` in the fixed chart."""

    return complex(cmath.exp(timelike_free_boson_log_factor(positions, signed_energies)))


def timelike_free_boson_log_factor(
    positions: Sequence[ProjectivePoint],
    signed_energies: Sequence[complex],
) -> complex:
    r"""Return the logarithm of the fixed-chart timelike correlator."""

    if len(positions) != 5 or len(signed_energies) != 5:
        raise ValueError("positions and signed_energies must contain five values")
    logarithm = 0.0 + 0.0j
    for left in range(5):
        if positions[left] is None:
            continue
        for right in range(left + 1, 5):
            if positions[right] is None:
                continue
            separation = abs(complex(positions[left]) - complex(positions[right]))
            if separation == 0.0:
                raise ZeroDivisionError("the timelike correlator hit a collision")
            logarithm -= (
                complex(signed_energies[left])
                * complex(signed_energies[right])
                * math.log(separation)
            )
    return complex(logarithm)


def _gauss_legendre_nodes(order: int, upper: float) -> tuple[np.ndarray, np.ndarray]:
    order = int(order)
    upper = float(upper)
    if order <= 0 or not math.isfinite(upper) or upper <= 0.0:
        raise ValueError("quadrature order and upper endpoint must be positive")
    nodes, weights = np.polynomial.legendre.leggauss(order)
    return 0.5 * upper * (nodes + 1.0), 0.5 * upper * weights


def _momentum_cutoff(q_value: complex, tail_tolerance: float) -> float:
    absolute_q = abs(complex(q_value))
    if not 0.0 < absolute_q < 1.0:
        raise ValueError("the five-point momentum integral requires 0<|q_i|<1")
    return max(
        2.0,
        math.sqrt(-math.log(float(tail_tolerance)) / (-2.0 * math.log(absolute_q))) + 1.0,
    )


@dataclass(frozen=True)
class FivePointLiouvilleResult:
    """One channel evaluation of the Liouville five-point function."""

    value: complex
    block_scheme: str
    block_order: int
    momentum_order: int
    momentum_cutoffs: tuple[float, float]
    channel: LinearChannel
    c25_h_fit_error_bound: float


def liouville_five_point_correlator(
    channel: LinearChannel,
    *,
    external_momenta: Sequence[complex],
    block_order: int = 4,
    momentum_order: int = 12,
    momentum_tail_tolerance: float = 1.0e-10,
    block_scheme: str = "h",
    special: UpsilonB | None = None,
    h_regulator_etas: Sequence[float] = (0.16, 0.13, 0.10, 0.075, 0.055),
) -> FivePointLiouvilleResult:
    r"""Evaluate the double-momentum five-point Liouville correlator.

    ``block_scheme='h'`` is the requested CCY h-recursion with its ``c=25``
    Kac-collision extrapolation.  ``block_scheme='c'`` is the independent
    exact-c recursion used for checks.
    """

    if block_scheme not in {"h", "c"}:
        raise ValueError("block_scheme must be 'h' or 'c'")
    block_order = int(block_order)
    momentum_order = int(momentum_order)
    if block_order < 0 or momentum_order <= 0:
        raise ValueError("block_order must be non-negative and momentum_order positive")
    momenta = tuple(complex(value) for value in external_momenta)
    if len(momenta) != 5:
        raise ValueError("external_momenta must contain five values")
    ordered_momenta = tuple(momenta[label] for label in channel.ordering)
    ordered_weights = liouville_weights(ordered_momenta)
    if special is None:
        special = UpsilonB(1.0, dps=35)

    p1_max = _momentum_cutoff(channel.q1, momentum_tail_tolerance)
    p2_max = _momentum_cutoff(channel.q2, momentum_tail_tolerance)
    p1_nodes, p1_weights = _gauss_legendre_nodes(momentum_order, p1_max)
    p2_nodes, p2_weights = _gauss_legendre_nodes(momentum_order, p2_max)
    total = 0.0 + 0.0j
    max_fit_error = 0.0
    pa, pb, pc, pd, pe = ordered_momenta
    q1_bar = channel.q1.conjugate()
    q2_bar = channel.q2.conjugate()

    for p1, quadrature_weight1 in zip(p1_nodes, p1_weights):
        first_structure = yin_structure_constant_momentum(special, pa, pb, float(p1))
        if first_structure == 0.0:
            continue
        h1 = 1.0 + float(p1) ** 2
        for p2, quadrature_weight2 in zip(p2_nodes, p2_weights):
            middle_structure = yin_structure_constant_momentum(
                special, float(p1), pc, float(p2)
            )
            last_structure = yin_structure_constant_momentum(
                special, float(p2), pd, pe
            )
            structure_product = first_structure * middle_structure * last_structure
            if structure_product == 0.0:
                continue
            h2 = 1.0 + float(p2) ** 2
            if block_scheme == "h":
                coefficients, fit_errors = sphere_five_point_h_c25_limit(
                    external_weights=ordered_weights,
                    internal_weights=(h1, h2),
                    order1=block_order,
                    order2=block_order,
                    max_total_order=block_order,
                    regulator_etas=h_regulator_etas,
                    polynomial_degree=min(3, len(tuple(h_regulator_etas)) - 1),
                )
                fit_error = sum(
                    abs(error)
                    * abs(channel.q1) ** levels[0]
                    * abs(channel.q2) ** levels[1]
                    for levels, error in fit_errors.items()
                )
                max_fit_error = max(max_fit_error, float(fit_error))
            else:
                coefficients = sphere_five_point_c_coefficients(
                    central_charge=25.0,
                    external_weights=ordered_weights,
                    internal_weights=(h1, h2),
                    order1=block_order,
                    order2=block_order,
                    max_total_order=block_order,
                )
            holomorphic = evaluate_sphere_five_point_series(
                channel.q1, channel.q2, coefficients
            )
            antiholomorphic = evaluate_sphere_five_point_series(
                q1_bar, q2_bar, coefficients
            )
            primary_product = (
                abs(channel.positions[1])
                ** (2.0 * (h1 - ordered_weights[0] - ordered_weights[1]))
                * abs(channel.positions[2])
                ** (2.0 * (h2 - ordered_weights[2] - h1))
            )
            total += (
                quadrature_weight1
                * quadrature_weight2
                / (math.pi * math.pi)
                * structure_product
                * primary_product
                * holomorphic
                * antiholomorphic
            )

    return FivePointLiouvilleResult(
        value=complex(total),
        block_scheme=block_scheme,
        block_order=block_order,
        momentum_order=momentum_order,
        momentum_cutoffs=(p1_max, p2_max),
        channel=channel,
        c25_h_fit_error_bound=max_fit_error,
    )


def five_point_worldsheet_integrand(
    positions: Sequence[ProjectivePoint],
    signed_energies: Sequence[complex],
    *,
    channel: LinearChannel | None = None,
    block_order: int = 4,
    momentum_order: int = 12,
    block_scheme: str = "h",
    special: UpsilonB | None = None,
) -> FivePointLiouvilleResult:
    """Evaluate the raw matter integrand in the original fixed-puncture chart."""

    if channel is None:
        channel = best_linear_channels(positions, limit=1)[0]
    external_momenta = tuple(0.5 * abs(1) * complex(value) for value in signed_energies)
    # Liouville momenta are the analytic energies divided by two without the
    # incoming/outgoing sign.  Callers therefore pass signed energies only to
    # the timelike factor and we remove their signs explicitly here.
    external_momenta = tuple(
        0.5 * (complex(value) if index == 0 else -complex(value))
        for index, value in enumerate(signed_energies)
    )
    liouville = liouville_five_point_correlator(
        channel,
        external_momenta=external_momenta,
        block_order=block_order,
        momentum_order=momentum_order,
        block_scheme=block_scheme,
        special=special,
    )
    weights = liouville_weights(external_momenta)
    covariance = liouville_primary_covariance(channel, weights)
    timelike = timelike_free_boson_factor(positions, signed_energies)
    return FivePointLiouvilleResult(
        value=complex(liouville.value * covariance * timelike),
        block_scheme=liouville.block_scheme,
        block_order=liouville.block_order,
        momentum_order=liouville.momentum_order,
        momentum_cutoffs=liouville.momentum_cutoffs,
        channel=channel,
        c25_h_fit_error_bound=liouville.c25_h_fit_error_bound,
    )
