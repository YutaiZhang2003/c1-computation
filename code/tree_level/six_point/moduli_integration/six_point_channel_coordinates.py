#!/usr/bin/env python3
"""Linear-channel plumbing atlas for the six-punctured sphere.

An oriented comb frame places the six labelled punctures at

``(0, q1*q2*q3, q2*q3, q3, 1, infinity)``.

All 720 label orderings are retained as proposal charts.  They represent the
90 labelled comb trees with their eight useful orientations.  The redundant
oriented mixture is intentional: it resolves OPE collars symmetrically and
gives an exact multiple-coverage density for QMC integration.
"""

from __future__ import annotations

from c1_paths import PACKAGE_ROOT, data_root, package_path, module_path

import itertools
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Sequence

try:
    from sphere_mobius_geometry import (
        MobiusMap,
        ProjectivePoint,
        mobius_to_zero_one_infinity,
    )
except ImportError:  # pragma: no cover - supports package-style execution
    from sphere_mobius_geometry import (
        MobiusMap,
        ProjectivePoint,
        mobius_to_zero_one_infinity,
    )


@dataclass(frozen=True)
class SixPointLinearChannel:
    """One oriented six-leaf comb channel in the CCY frame."""

    ordering: tuple[int, int, int, int, int, int]
    positions: tuple[complex, complex, complex, complex, complex, None]
    q1: complex
    q2: complex
    q3: complex
    mobius: MobiusMap
    local_scales: tuple[complex, ...]
    score: float

    @property
    def convergence_score(self) -> float:
        """Radial gauge controlling total-degree convergence of the comb series."""

        return self.score

    @property
    def left_cherry(self) -> tuple[int, int]:
        return self.ordering[0], self.ordering[1]

    @property
    def right_cherry(self) -> tuple[int, int]:
        return self.ordering[4], self.ordering[5]

    @property
    def middle_split(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        left = tuple(sorted(self.ordering[:3]))
        right = tuple(sorted(self.ordering[3:]))
        return tuple(sorted((left, right)))  # type: ignore[return-value]


@dataclass(frozen=True)
class SixPointForwardMap:
    """The original three-moving-puncture chart and holomorphic Jacobian."""

    positions: tuple[complex, complex, complex, complex, complex, None]
    complex_jacobian: complex

    @property
    def area_jacobian(self) -> float:
        return float(abs(self.complex_jacobian) ** 2)


@dataclass(frozen=True)
class SixPointStarChannel:
    """One oriented three-cherry star channel."""

    ordering: tuple[int, int, int, int, int, int]
    positions: tuple[complex, complex, complex, complex, complex, None]
    q1: complex
    q2: complex
    q3: complex
    mobius: MobiusMap
    local_scales: tuple[complex, ...]
    score: float

    @property
    def convergence_score(self) -> float:
        """Radial gauge of the absolute-convergence domain of the star series.

        In the canonical frame ``(0,q1,1,1+q2,1/q3,infinity)``, the
        complete Reinhardt domain containing the origin is bounded by the
        cross-cherry collision divisors ``q1=1+q2`` and
        ``q3*(1+q2)=1``.  Along a common radial rescaling of all three
        variables their gauges are respectively ``|q1|+|q2|`` and the
        positive root of ``s**2-|q3|*s-|q2*q3|=0``.

        A value below one is therefore the necessary collision-domain test
        and is directly comparable with ``max(abs(qi))`` in a comb chart.
        """

        radius1, radius2, radius3 = (
            abs(self.q1),
            abs(self.q2),
            abs(self.q3),
        )
        infinity_gauge = 0.5 * (
            radius3 + math.sqrt(radius3 * radius3 + 4.0 * radius2 * radius3)
        )
        return float(max(radius1 + radius2, infinity_gauge))

    @property
    def cherries(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            sorted(
                tuple(sorted(self.ordering[offset : offset + 2]))
                for offset in (0, 2, 4)
            )
        )


@dataclass(frozen=True)
class _Dual3:
    value: complex
    gradient: tuple[complex, complex, complex] = (0.0j, 0.0j, 0.0j)

    def __add__(self, other: object) -> "_Dual3":
        other_dual = _as_dual(other)
        return _Dual3(
            self.value + other_dual.value,
            tuple(left + right for left, right in zip(self.gradient, other_dual.gradient)),
        )

    __radd__ = __add__

    def __neg__(self) -> "_Dual3":
        return _Dual3(-self.value, tuple(-value for value in self.gradient))

    def __sub__(self, other: object) -> "_Dual3":
        return self + (-_as_dual(other))

    def __rsub__(self, other: object) -> "_Dual3":
        return _as_dual(other) - self

    def __mul__(self, other: object) -> "_Dual3":
        other_dual = _as_dual(other)
        return _Dual3(
            self.value * other_dual.value,
            tuple(
                left * other_dual.value + self.value * right
                for left, right in zip(self.gradient, other_dual.gradient)
            ),
        )

    __rmul__ = __mul__

    def __truediv__(self, other: object) -> "_Dual3":
        other_dual = _as_dual(other)
        if other_dual.value == 0.0:
            raise ZeroDivisionError("the six-point forward map hit a boundary")
        inverse = 1.0 / other_dual.value
        return _Dual3(
            self.value * inverse,
            tuple(
                (left - self.value * inverse * right) * inverse
                for left, right in zip(self.gradient, other_dual.gradient)
            ),
        )

    def __rtruediv__(self, other: object) -> "_Dual3":
        return _as_dual(other) / self


def _as_dual(value: object) -> _Dual3:
    if isinstance(value, _Dual3):
        return value
    return _Dual3(complex(value))


def _dual_determinant(
    left: tuple[_Dual3, _Dual3],
    right: tuple[_Dual3, _Dual3],
) -> _Dual3:
    return left[0] * right[1] - left[1] * right[0]


def _determinant3(rows: Sequence[Sequence[complex]]) -> complex:
    if len(rows) != 3 or any(len(row) != 3 for row in rows):
        raise ValueError("a 3 by 3 matrix is required")
    a, b, c = rows
    return complex(
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


@lru_cache(maxsize=None)
def _oriented_comb_orderings_cached(
    normalized: tuple[int, ...],
) -> tuple[tuple[int, ...], ...]:
    if len(normalized) != 6 or len(set(normalized)) != 6:
        raise ValueError("labels must contain six distinct entries")
    return tuple(itertools.permutations(normalized))


def oriented_comb_orderings(labels: Sequence[int] = range(6)) -> tuple[tuple[int, ...], ...]:
    """Return all 720 oriented comb frames."""

    return _oriented_comb_orderings_cached(tuple(int(label) for label in labels))


def oriented_star_orderings(labels: Sequence[int] = range(6)) -> tuple[tuple[int, ...], ...]:
    """Return all 720 oriented star frames.

    For ``(a,b,c,d,e,f)`` the pairs occupy the central zero, one, and
    infinity slots and are placed at ``(0,q1)``, ``(1,1+q2)``, and
    ``(1/q3,infinity)``.
    """

    return oriented_comb_orderings(labels)


def star_tree_signature(ordering: Sequence[int]) -> tuple[tuple[int, int], ...]:
    """Return the unordered three-cherry partition of a star ordering."""

    order = tuple(int(label) for label in ordering)
    if len(order) != 6 or set(order) != set(range(6)):
        raise ValueError("ordering must permute labels 0,...,5")
    return tuple(
        sorted(tuple(sorted(order[offset : offset + 2])) for offset in (0, 2, 4))
    )


def comb_tree_signature(ordering: Sequence[int]) -> tuple[object, ...]:
    """Return the unoriented labelled-tree signature of one comb ordering."""

    order = tuple(int(label) for label in ordering)
    if len(order) != 6 or set(order) != set(range(6)):
        raise ValueError("ordering must permute labels 0,...,5")
    cherries = tuple(sorted((tuple(sorted(order[:2])), tuple(sorted(order[4:])))))
    middle = tuple(sorted((tuple(sorted(order[:3])), tuple(sorted(order[3:])))))
    return cherries + (middle,)


def linear_channel_positions_by_label(
    q1: complex,
    q2: complex,
    q3: complex,
    ordering: Sequence[int],
) -> tuple[ProjectivePoint, ...]:
    """Return label-ordered positions in an oriented comb gauge."""

    order = tuple(int(label) for label in ordering)
    if len(order) != 6 or set(order) != set(range(6)):
        raise ValueError("ordering must permute labels 0,...,5")
    q1, q2, q3 = complex(q1), complex(q2), complex(q3)
    frame = (0.0j, q1 * q2 * q3, q2 * q3, q3, 1.0 + 0.0j, None)
    by_label: list[ProjectivePoint] = [None] * 6
    for label, point in zip(order, frame):
        by_label[label] = point
    return tuple(by_label)


def linear_channel_from_plumbing_coordinates(
    q1: complex,
    q2: complex,
    q3: complex,
    ordering: Sequence[int],
) -> SixPointLinearChannel:
    """Construct the exact proposal channel without reconstructing its ``q`` values.

    This is the numerically stable inverse of
    :func:`linear_channel_positions_by_label` for a chart already expressed in
    its canonical ``(0,1,infinity)`` gauge.  Keeping the supplied plumbing
    coordinates is important in deep collars: reconstructing ratios of very
    small puncture positions can round a nonzero ``q`` to zero.
    """

    order = tuple(int(label) for label in ordering)
    positions = linear_channel_positions_by_label(q1, q2, q3, order)
    q_values = (complex(q1), complex(q2), complex(q3))
    if any(value == 0.0 for value in q_values):
        raise ValueError("linear plumbing coordinates must be nonzero")
    identity = MobiusMap(1.0 + 0.0j, 0.0j, 0.0j, 1.0 + 0.0j)
    return SixPointLinearChannel(
        ordering=order,
        positions=tuple(positions[label] for label in order),
        q1=q_values[0],
        q2=q_values[1],
        q3=q_values[2],
        mobius=identity,
        local_scales=(1.0 + 0.0j,) * 6,
        score=float(max(abs(value) for value in q_values)),
    )


def linear_channel_from_ordering(
    positions: Sequence[ProjectivePoint],
    ordering: Sequence[int],
) -> SixPointLinearChannel:
    """Construct one oriented comb channel from six labelled punctures."""

    if len(positions) != 6 or len(ordering) != 6:
        raise ValueError("positions and ordering must each contain six entries")
    order = tuple(int(label) for label in ordering)
    if set(order) != set(range(6)):
        raise ValueError("ordering must permute labels 0,...,5")
    a, b, c, d, e, f = order
    transform = mobius_to_zero_one_infinity(positions[a], positions[e], positions[f])
    x1 = transform(positions[b])
    x2 = transform(positions[c])
    x3 = transform(positions[d])
    if x1 is None or x2 is None or x3 is None or x2 == 0.0 or x3 == 0.0:
        raise ValueError("the ordered six-point channel is degenerate")
    q1 = complex(x1 / x2)
    q2 = complex(x2 / x3)
    q3 = complex(x3)
    scales = tuple(transform.local_scale(positions[label]) for label in order)
    return SixPointLinearChannel(
        ordering=order,
        positions=(0.0j, complex(x1), complex(x2), complex(x3), 1.0 + 0.0j, None),
        q1=q1,
        q2=q2,
        q3=q3,
        mobius=transform,
        local_scales=scales,
        score=float(max(abs(q1), abs(q2), abs(q3))),
    )


def star_channel_positions_by_label(
    q1: complex,
    q2: complex,
    q3: complex,
    ordering: Sequence[int],
) -> tuple[ProjectivePoint, ...]:
    """Return label-ordered positions in an oriented star gauge."""

    order = tuple(int(label) for label in ordering)
    if len(order) != 6 or set(order) != set(range(6)):
        raise ValueError("ordering must permute labels 0,...,5")
    q1, q2, q3 = complex(q1), complex(q2), complex(q3)
    if q3 == 0.0:
        raise ValueError("the infinity-cherry plumbing parameter must be nonzero")
    frame = (0.0j, q1, 1.0 + 0.0j, 1.0 + q2, 1.0 / q3, None)
    by_label: list[ProjectivePoint] = [None] * 6
    for label, point in zip(order, frame):
        by_label[label] = point
    return tuple(by_label)


def star_channel_from_plumbing_coordinates(
    q1: complex,
    q2: complex,
    q3: complex,
    ordering: Sequence[int],
) -> SixPointStarChannel:
    """Construct the exact canonical star proposal from its supplied ``q`` values."""

    order = tuple(int(label) for label in ordering)
    positions = star_channel_positions_by_label(q1, q2, q3, order)
    q_values = (complex(q1), complex(q2), complex(q3))
    if any(value == 0.0 for value in q_values):
        raise ValueError("star plumbing coordinates must be nonzero")
    identity = MobiusMap(1.0 + 0.0j, 0.0j, 0.0j, 1.0 + 0.0j)
    return SixPointStarChannel(
        ordering=order,
        positions=tuple(positions[label] for label in order),
        q1=q_values[0],
        q2=q_values[1],
        q3=q_values[2],
        mobius=identity,
        local_scales=(1.0 + 0.0j,) * 6,
        score=float(max(abs(value) for value in q_values)),
    )


def star_channel_from_ordering(
    positions: Sequence[ProjectivePoint],
    ordering: Sequence[int],
) -> SixPointStarChannel:
    """Construct one oriented star channel from six labelled punctures."""

    if len(positions) != 6 or len(ordering) != 6:
        raise ValueError("positions and ordering must each contain six entries")
    order = tuple(int(label) for label in ordering)
    if set(order) != set(range(6)):
        raise ValueError("ordering must permute labels 0,...,5")
    a, b, c, d, e, f = order
    transform = mobius_to_zero_one_infinity(positions[a], positions[c], positions[f])
    first = transform(positions[b])
    second = transform(positions[d])
    third = transform(positions[e])
    if first is None or second is None or third is None or third == 0.0:
        raise ValueError("the ordered star channel is degenerate")
    q1 = complex(first)
    q2 = complex(second - 1.0)
    q3 = complex(1.0 / third)
    scales = tuple(transform.local_scale(positions[label]) for label in order)
    return SixPointStarChannel(
        ordering=order,
        positions=(0.0j, q1, 1.0 + 0.0j, 1.0 + q2, 1.0 / q3, None),
        q1=q1,
        q2=q2,
        q3=q3,
        mobius=transform,
        local_scales=scales,
        score=float(max(abs(q1), abs(q2), abs(q3))),
    )


def best_linear_channels(
    positions: Sequence[ProjectivePoint],
    *,
    limit: int = 2,
    convergence_radius: float = 1.0,
) -> tuple[SixPointLinearChannel, ...]:
    """Return the best distinct convergent comb-tree channels."""

    if len(positions) != 6:
        raise ValueError("positions must contain six punctures")
    limit = int(limit)
    if limit <= 0:
        raise ValueError("limit must be positive")
    best_by_tree: dict[tuple[object, ...], SixPointLinearChannel] = {}
    for ordering in oriented_comb_orderings():
        try:
            channel = linear_channel_from_ordering(positions, ordering)
        except (ArithmeticError, ValueError, ZeroDivisionError):
            continue
        if channel.score >= float(convergence_radius):
            continue
        signature = comb_tree_signature(ordering)
        previous = best_by_tree.get(signature)
        if previous is None or channel.score < previous.score:
            best_by_tree[signature] = channel
    candidates = sorted(best_by_tree.values(), key=lambda item: item.score)
    if not candidates:
        raise RuntimeError("no convergent six-point comb channel was found")
    return tuple(candidates[:limit])


def best_star_channels(
    positions: Sequence[ProjectivePoint],
    *,
    limit: int = 2,
    convergence_radius: float = 1.0,
) -> tuple[SixPointStarChannel, ...]:
    """Return the best distinct absolutely convergent star-tree channels."""

    if len(positions) != 6:
        raise ValueError("positions must contain six punctures")
    limit = int(limit)
    if limit <= 0:
        raise ValueError("limit must be positive")
    best_by_tree: dict[tuple[tuple[int, int], ...], SixPointStarChannel] = {}
    for ordering in oriented_star_orderings():
        try:
            channel = star_channel_from_ordering(positions, ordering)
        except (ArithmeticError, ValueError, ZeroDivisionError):
            continue
        if channel.convergence_score >= float(convergence_radius):
            continue
        signature = star_tree_signature(ordering)
        previous = best_by_tree.get(signature)
        if previous is None or channel.convergence_score < previous.convergence_score:
            best_by_tree[signature] = channel
    candidates = sorted(best_by_tree.values(), key=lambda item: item.convergence_score)
    if not candidates:
        raise RuntimeError("no convergent six-point star channel was found")
    return tuple(candidates[:limit])


def _dual_frame(
    q1: complex,
    q2: complex,
    q3: complex,
    ordering: Sequence[int],
) -> dict[int, tuple[_Dual3, _Dual3]]:
    order = tuple(int(label) for label in ordering)
    if len(order) != 6 or set(order) != set(range(6)):
        raise ValueError("ordering must permute labels 0,...,5")
    q1_dual = _Dual3(complex(q1), (1.0 + 0.0j, 0.0j, 0.0j))
    q2_dual = _Dual3(complex(q2), (0.0j, 1.0 + 0.0j, 0.0j))
    q3_dual = _Dual3(complex(q3), (0.0j, 0.0j, 1.0 + 0.0j))
    x1 = q1_dual * q2_dual * q3_dual
    x2 = q2_dual * q3_dual
    x3 = q3_dual
    frame = (
        (_Dual3(0.0j), _Dual3(1.0 + 0.0j)),
        (x1, _Dual3(1.0 + 0.0j)),
        (x2, _Dual3(1.0 + 0.0j)),
        (x3, _Dual3(1.0 + 0.0j)),
        (_Dual3(1.0 + 0.0j), _Dual3(1.0 + 0.0j)),
        (_Dual3(1.0 + 0.0j), _Dual3(0.0j)),
    )
    return {label: point for label, point in zip(order, frame)}


def linear_channel_complex_jacobian_to_chart(
    q1: complex,
    q2: complex,
    q3: complex,
    ordering: Sequence[int],
    *,
    fixed_zero: int,
    fixed_one: int,
    fixed_infinity: int,
    moving_labels: Sequence[int],
) -> complex:
    """Return det d(z_m1,z_m2,z_m3)/d(q1,q2,q3) in a fixed gauge."""

    order = tuple(int(label) for label in ordering)
    moving = tuple(int(label) for label in moving_labels)
    fixed = (int(fixed_zero), int(fixed_one), int(fixed_infinity))
    if len(order) != 6 or set(order) != set(range(6)):
        raise ValueError("ordering must permute labels 0,...,5")
    if len(moving) != 3 or len(set(moving)) != 3:
        raise ValueError("moving_labels must contain three distinct labels")
    if len(set(fixed)) != 3 or set(fixed) & set(moving) or set(fixed) | set(moving) != set(range(6)):
        raise ValueError("fixed and moving labels must partition labels 0,...,5")

    by_label = _dual_frame(q1, q2, q3, order)
    zero_point = by_label[fixed[0]]
    one_point = by_label[fixed[1]]
    infinity_point = by_label[fixed[2]]
    normalization = _dual_determinant(one_point, infinity_point) / _dual_determinant(
        one_point, zero_point
    )

    def transform(point: tuple[_Dual3, _Dual3]) -> _Dual3:
        return (
            _dual_determinant(point, zero_point)
            / _dual_determinant(point, infinity_point)
            * normalization
        )

    transformed = tuple(transform(by_label[label]) for label in moving)
    return _determinant3(tuple(item.gradient for item in transformed))


def _star_dual_frame(
    q1: complex,
    q2: complex,
    q3: complex,
    ordering: Sequence[int],
) -> dict[int, tuple[_Dual3, _Dual3]]:
    order = tuple(int(label) for label in ordering)
    if len(order) != 6 or set(order) != set(range(6)):
        raise ValueError("ordering must permute labels 0,...,5")
    q1_dual = _Dual3(complex(q1), (1.0 + 0.0j, 0.0j, 0.0j))
    q2_dual = _Dual3(complex(q2), (0.0j, 1.0 + 0.0j, 0.0j))
    q3_dual = _Dual3(complex(q3), (0.0j, 0.0j, 1.0 + 0.0j))
    frame = (
        (_Dual3(0.0j), _Dual3(1.0 + 0.0j)),
        (q1_dual, _Dual3(1.0 + 0.0j)),
        (_Dual3(1.0 + 0.0j), _Dual3(1.0 + 0.0j)),
        (_Dual3(1.0 + 0.0j) + q2_dual, _Dual3(1.0 + 0.0j)),
        (_Dual3(1.0 + 0.0j) / q3_dual, _Dual3(1.0 + 0.0j)),
        (_Dual3(1.0 + 0.0j), _Dual3(0.0j)),
    )
    return {label: point for label, point in zip(order, frame)}


def star_channel_complex_jacobian_to_chart(
    q1: complex,
    q2: complex,
    q3: complex,
    ordering: Sequence[int],
    *,
    fixed_zero: int,
    fixed_one: int,
    fixed_infinity: int,
    moving_labels: Sequence[int],
) -> complex:
    """Return a star-chart holomorphic Jacobian in a fixed-label gauge."""

    order = tuple(int(label) for label in ordering)
    moving = tuple(int(label) for label in moving_labels)
    fixed = (int(fixed_zero), int(fixed_one), int(fixed_infinity))
    if len(order) != 6 or set(order) != set(range(6)):
        raise ValueError("ordering must permute labels 0,...,5")
    if len(moving) != 3 or len(set(moving)) != 3:
        raise ValueError("moving_labels must contain three distinct labels")
    if len(set(fixed)) != 3 or set(fixed) & set(moving) or set(fixed) | set(moving) != set(range(6)):
        raise ValueError("fixed and moving labels must partition labels 0,...,5")
    by_label = _star_dual_frame(q1, q2, q3, order)
    zero_point = by_label[fixed[0]]
    one_point = by_label[fixed[1]]
    infinity_point = by_label[fixed[2]]
    normalization = _dual_determinant(one_point, infinity_point) / _dual_determinant(
        one_point, zero_point
    )

    def transform(point: tuple[_Dual3, _Dual3]) -> _Dual3:
        return (
            _dual_determinant(point, zero_point)
            / _dual_determinant(point, infinity_point)
            * normalization
        )

    transformed = tuple(transform(by_label[label]) for label in moving)
    return _determinant3(tuple(item.gradient for item in transformed))


def linear_channel_to_original_chart(
    q1: complex,
    q2: complex,
    q3: complex,
    ordering: Sequence[int],
) -> SixPointForwardMap:
    r"""Map a comb tridisc to ``(z0,z1,z2,0,1,infinity)``."""

    positions = linear_channel_positions_by_label(q1, q2, q3, ordering)
    channel = linear_channel_from_ordering(positions, ordering)
    # The original chart fixes labels (3,4,5) at (0,1,infinity).
    transform = mobius_to_zero_one_infinity(positions[3], positions[4], positions[5])
    moving = tuple(transform(positions[label]) for label in (0, 1, 2))
    if any(value is None for value in moving):
        raise ValueError("an original-chart moving puncture was mapped to infinity")
    jacobian = linear_channel_complex_jacobian_to_chart(
        channel.q1,
        channel.q2,
        channel.q3,
        ordering,
        fixed_zero=3,
        fixed_one=4,
        fixed_infinity=5,
        moving_labels=(0, 1, 2),
    )
    return SixPointForwardMap(
        positions=(
            complex(moving[0]),
            complex(moving[1]),
            complex(moving[2]),
            0.0j,
            1.0 + 0.0j,
            None,
        ),
        complex_jacobian=jacobian,
    )


def oriented_tridisc_log_mixture_density_in_frame(
    positions: Sequence[ProjectivePoint],
    selected_ordering: Sequence[int],
    *,
    radial_power: float,
) -> float:
    r"""Return the 720-chart log proposal density in a selected comb gauge."""

    selected = tuple(int(label) for label in selected_ordering)
    if len(selected) != 6 or set(selected) != set(range(6)):
        raise ValueError("selected_ordering must permute labels 0,...,5")
    if len(positions) != 6:
        raise ValueError("positions must contain six label-ordered punctures")
    radial_power = float(radial_power)
    if not 0.0 < radial_power <= 2.0:
        raise ValueError("radial_power must lie in (0,2]")
    fixed_zero, fixed_one, fixed_infinity = selected[0], selected[4], selected[5]
    moving_labels = (selected[1], selected[2], selected[3])
    base_log_density = math.log(radial_power / (2.0 * math.pi))
    logarithmic_terms: list[float] = []
    orderings = oriented_comb_orderings()
    for ordering in orderings:
        try:
            channel = linear_channel_from_ordering(positions, ordering)
            radii = (abs(channel.q1), abs(channel.q2), abs(channel.q3))
            if any(not 0.0 < radius < 1.0 for radius in radii):
                continue
            jacobian = linear_channel_complex_jacobian_to_chart(
                channel.q1,
                channel.q2,
                channel.q3,
                ordering,
                fixed_zero=fixed_zero,
                fixed_one=fixed_one,
                fixed_infinity=fixed_infinity,
                moving_labels=moving_labels,
            )
            absolute_jacobian = abs(jacobian)
            if not math.isfinite(absolute_jacobian) or absolute_jacobian <= 0.0:
                continue
            logarithmic_terms.append(
                3.0 * base_log_density
                + (radial_power - 2.0) * sum(math.log(radius) for radius in radii)
                - 2.0 * math.log(absolute_jacobian)
            )
        except (ArithmeticError, OverflowError, ValueError, ZeroDivisionError):
            continue
    if not logarithmic_terms:
        raise ArithmeticError("no oriented tridisc contributes to the mixture density")
    maximum = max(logarithmic_terms)
    return float(
        maximum
        + math.log(sum(math.exp(value - maximum) for value in logarithmic_terms))
        - math.log(len(orderings))
    )


def mixed_atlas_log_density_in_frame(
    positions: Sequence[ProjectivePoint],
    *,
    fixed_zero: int,
    fixed_one: int,
    fixed_infinity: int,
    moving_labels: Sequence[int],
    radial_power: float,
) -> float:
    r"""Return the equal 720-comb plus 720-star proposal density.

    The density is expressed in the gauge specified by the three fixed labels.
    This routine is used regardless of whether the sampled chart was a comb or
    a star, so multiple coverage is removed exactly across both topologies.
    """

    return mixed_atlas_density_and_channels_in_frame(
        positions, fixed_zero=fixed_zero, fixed_one=fixed_one,
        fixed_infinity=fixed_infinity, moving_labels=moving_labels,
        radial_power=radial_power, collect_channels=False,
    )[0]


def mixed_atlas_density_and_channels_in_frame(
    positions: Sequence[ProjectivePoint],
    *,
    fixed_zero: int,
    fixed_one: int,
    fixed_infinity: int,
    moving_labels: Sequence[int],
    radial_power: float,
    collect_channels: bool = True,
) -> tuple[float, tuple[SixPointLinearChannel, ...], tuple[SixPointStarChannel, ...]]:
    """Enumerate the exact proposal and retain the existing selector candidates.

    Tree insertion order and strict score comparisons preserve selector ties.
    Zero-coordinate candidates are retained for the caller's source-chart
    recovery, while their singular proposal densities are excluded as before.
    """
    if len(positions) != 6:
        raise ValueError("positions must contain six label-ordered punctures")
    moving = tuple(int(label) for label in moving_labels)
    fixed = (int(fixed_zero), int(fixed_one), int(fixed_infinity))
    if len(moving) != 3 or len(set(moving)) != 3:
        raise ValueError("moving_labels must contain three distinct labels")
    if len(set(fixed)) != 3 or set(fixed) & set(moving) or set(fixed) | set(moving) != set(range(6)):
        raise ValueError("fixed and moving labels must partition labels 0,...,5")
    radial_power = float(radial_power)
    if not 0.0 < radial_power <= 2.0:
        raise ValueError("radial_power must lie in (0,2]")
    base_log_density = math.log(radial_power / (2.0 * math.pi))
    logarithmic_terms: list[float] = []
    orderings = oriented_comb_orderings()
    best_comb = {}
    best_star = {}

    for topology in ("comb", "star"):
        for ordering in orderings:
            try:
                if topology == "comb":
                    channel = linear_channel_from_ordering(positions, ordering)
                    if collect_channels and channel.score < 1.0:
                        signature = comb_tree_signature(ordering)
                        previous = best_comb.get(signature)
                        if previous is None or channel.score < previous.score:
                            best_comb[signature] = channel
                else:
                    channel = star_channel_from_ordering(positions, ordering)
                    if collect_channels and channel.convergence_score < 1.0:
                        signature = star_tree_signature(ordering)
                        previous = best_star.get(signature)
                        if previous is None or channel.convergence_score < previous.convergence_score:
                            best_star[signature] = channel
                radii = (abs(channel.q1), abs(channel.q2), abs(channel.q3))
                if any(not 0.0 < radius < 1.0 for radius in radii):
                    continue
                jacobian_function = (linear_channel_complex_jacobian_to_chart
                                     if topology == "comb"
                                     else star_channel_complex_jacobian_to_chart)
                jacobian = jacobian_function(
                    channel.q1, channel.q2, channel.q3, ordering,
                    fixed_zero=fixed[0], fixed_one=fixed[1],
                    fixed_infinity=fixed[2], moving_labels=moving,
                )
                absolute_jacobian = abs(jacobian)
                if not math.isfinite(absolute_jacobian) or absolute_jacobian <= 0.0:
                    continue
                logarithmic_terms.append(
                    3.0 * base_log_density
                    + (radial_power - 2.0) * sum(math.log(radius) for radius in radii)
                    - 2.0 * math.log(absolute_jacobian)
                )
            except (ArithmeticError, OverflowError, ValueError, ZeroDivisionError):
                continue
    if not logarithmic_terms:
        raise ArithmeticError("no six-point plumbing chart contributes to the density")
    maximum = max(logarithmic_terms)
    density = float(
        maximum
        + math.log(sum(math.exp(value - maximum) for value in logarithmic_terms))
        - math.log(2 * len(orderings))
    )
    return (density,
            tuple(sorted(best_comb.values(), key=lambda channel: channel.score)),
            tuple(sorted(best_star.values(), key=lambda channel: channel.convergence_score)))


def liouville_primary_covariance_log(
    channel: SixPointLinearChannel,
    external_weights: Sequence[complex],
) -> complex:
    """Return the log nonchiral primary covariance from a fixed chart."""

    if len(external_weights) != 6:
        raise ValueError("external_weights must contain six values")
    ordered_weights = tuple(complex(external_weights[label]) for label in channel.ordering)
    logarithm = 0.0 + 0.0j
    for scale, weight in zip(channel.local_scales, ordered_weights):
        absolute_scale = abs(scale)
        if absolute_scale == 0.0:
            raise ZeroDivisionError("a channel has a vanishing puncture scale")
        logarithm += 2.0 * weight * math.log(absolute_scale)
    return complex(logarithm)


def timelike_free_boson_log_factor(
    positions: Sequence[ProjectivePoint],
    signed_energies: Sequence[complex],
) -> complex:
    r"""Return log prod_finite |z_i-z_j|^{-k_i k_j}."""

    if len(positions) != 6 or len(signed_energies) != 6:
        raise ValueError("positions and signed_energies must contain six values")
    logarithm = 0.0 + 0.0j
    for left in range(6):
        if positions[left] is None:
            continue
        for right in range(left + 1, 6):
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


__all__ = [
    "SixPointForwardMap",
    "SixPointLinearChannel",
    "SixPointStarChannel",
    "best_linear_channels",
    "best_star_channels",
    "comb_tree_signature",
    "liouville_primary_covariance_log",
    "linear_channel_complex_jacobian_to_chart",
    "linear_channel_from_plumbing_coordinates",
    "linear_channel_from_ordering",
    "linear_channel_positions_by_label",
    "linear_channel_to_original_chart",
    "mixed_atlas_log_density_in_frame",
    "mixed_atlas_density_and_channels_in_frame",
    "oriented_comb_orderings",
    "oriented_star_orderings",
    "oriented_tridisc_log_mixture_density_in_frame",
    "timelike_free_boson_log_factor",
    "star_channel_complex_jacobian_to_chart",
    "star_channel_from_plumbing_coordinates",
    "star_channel_from_ordering",
    "star_channel_positions_by_label",
    "star_tree_signature",
]
