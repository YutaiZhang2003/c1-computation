#!/usr/bin/env python3
"""Geometry and low-cost numerical checks for the five-point Liouville atlas."""

from __future__ import annotations

from c1_paths import PACKAGE_ROOT, data_root, package_path, module_path

import random

try:
    from sphere_five_point_liouville import (
        INFINITY,
        best_linear_channels,
        linear_channel_complex_jacobian_to_chart,
        linear_channel_positions_by_label,
        linear_channel_to_original_chart,
        linear_channel_from_ordering,
        mobius_to_zero_one_infinity,
    )
except ImportError:  # pragma: no cover - supports package-style execution
    from sphere_five_point_liouville import (
        INFINITY,
        best_linear_channels,
        linear_channel_complex_jacobian_to_chart,
        linear_channel_positions_by_label,
        linear_channel_to_original_chart,
        linear_channel_from_ordering,
        mobius_to_zero_one_infinity,
    )


def _point_error(value: complex | None, target: complex | None) -> float:
    if value is None or target is None:
        return 0.0 if value is target else float("inf")
    return abs(complex(value) - complex(target))


def check_mobius_special_cases() -> None:
    cases = (
        (0.0, 1.0, INFINITY),
        (INFINITY, 0.0, 1.0),
        (1.2 - 0.4j, INFINITY, -0.7 + 0.2j),
        (-0.3 + 0.8j, 0.9 + 0.1j, 1.7 - 0.6j),
    )
    maximum_error = 0.0
    minimum_scale = float("inf")
    for zero, one, infinity in cases:
        transform = mobius_to_zero_one_infinity(zero, one, infinity)
        maximum_error = max(
            maximum_error,
            _point_error(transform(zero), 0.0),
            _point_error(transform(one), 1.0),
            _point_error(transform(infinity), INFINITY),
        )
        minimum_scale = min(
            minimum_scale,
            abs(transform.local_scale(zero)),
            abs(transform.local_scale(one)),
            abs(transform.local_scale(infinity)),
        )
    print("Mobius channel frames")
    print(f"  max landmark error={maximum_error:.3e}")
    print(f"  minimum local-coordinate scale={minimum_scale:.3e}")
    if maximum_error > 1.0e-12 or minimum_scale <= 0.0:
        raise AssertionError("Mobius frame or local-coordinate scale is incorrect")


def check_known_linear_frame() -> None:
    positions = (0.0, 0.12 + 0.03j, 0.31 + 0.08j, 1.0, INFINITY)
    channel = linear_channel_from_ordering(positions, (0, 1, 2, 3, 4))
    print("\nknown CCY linear frame")
    print(f"  q1={channel.q1!r}, q2={channel.q2!r}, score={channel.score:.3e}")
    if abs(channel.q1 - positions[1] / positions[2]) > 1.0e-14:
        raise AssertionError("q1 is not z1/z2")
    if abs(channel.q2 - positions[2]) > 1.0e-14:
        raise AssertionError("q2 is not z2")
    if channel.convergence_score != channel.score:
        raise AssertionError("the nested five-point bidisk has an inconsistent gauge")


def check_random_atlas_coverage() -> None:
    random.seed(1729)
    worst_score = 0.0
    for _ in range(80):
        z1 = complex(random.uniform(-3.0, 3.0), random.uniform(-3.0, 3.0))
        z2 = complex(random.uniform(-3.0, 3.0), random.uniform(-3.0, 3.0))
        if min(abs(z1), abs(z1 - 1), abs(z2), abs(z2 - 1), abs(z1 - z2)) < 1.0e-3:
            continue
        channels = best_linear_channels((z1, z2, 0.0, 1.0, INFINITY), limit=2)
        worst_score = max(worst_score, channels[0].score)
        if channels[0].score >= 1.0:
            raise AssertionError("the selected five-point block is outside its bidisc")
    print("\nrandom five-point atlas coverage")
    print(f"  worst best-channel score={worst_score:.6f}")


def check_forward_map_roundtrip_and_jacobian() -> None:
    ordering = (2, 0, 4, 1, 3)
    q1 = 0.31 + 0.07j
    q2 = -0.22 + 0.18j
    forward = linear_channel_to_original_chart(q1, q2, ordering)
    recovered = linear_channel_from_ordering(forward.positions, ordering)
    step = 1.0e-6
    shifted1_plus = linear_channel_to_original_chart(q1 + step, q2, ordering).positions
    shifted1_minus = linear_channel_to_original_chart(q1 - step, q2, ordering).positions
    shifted2_plus = linear_channel_to_original_chart(q1, q2 + step, ordering).positions
    shifted2_minus = linear_channel_to_original_chart(q1, q2 - step, ordering).positions
    dz0_dq1 = (shifted1_plus[0] - shifted1_minus[0]) / (2.0 * step)
    dz1_dq1 = (shifted1_plus[1] - shifted1_minus[1]) / (2.0 * step)
    dz0_dq2 = (shifted2_plus[0] - shifted2_minus[0]) / (2.0 * step)
    dz1_dq2 = (shifted2_plus[1] - shifted2_minus[1]) / (2.0 * step)
    finite_difference = dz0_dq1 * dz1_dq2 - dz0_dq2 * dz1_dq1
    print("\nlinear-channel forward map")
    print(f"  q roundtrip error={max(abs(recovered.q1-q1),abs(recovered.q2-q2)):.3e}")
    print(f"  Jacobian finite-difference error={abs(forward.complex_jacobian-finite_difference):.3e}")
    if max(abs(recovered.q1 - q1), abs(recovered.q2 - q2)) > 1.0e-12:
        raise AssertionError("the oriented bidisc map does not roundtrip")
    if abs(forward.complex_jacobian - finite_difference) > 2.0e-9:
        raise AssertionError("the oriented bidisc Jacobian is incorrect")


def check_arbitrary_gauge_jacobian() -> None:
    ordering = (4, 1, 3, 0, 2)
    q1 = -0.27 + 0.11j
    q2 = 0.38 - 0.09j
    selected = (2, 0, 4, 1, 3)
    positions = linear_channel_positions_by_label(q1, q2, ordering)
    channel = linear_channel_from_ordering(positions, selected)
    analytic = linear_channel_complex_jacobian_to_chart(
        channel.q1,
        channel.q2,
        selected,
        fixed_zero=ordering[0],
        fixed_one=ordering[3],
        fixed_infinity=ordering[4],
        moving_labels=(ordering[1], ordering[2]),
    )
    step = 1.0e-6

    def moving_coordinates(first: complex, second: complex) -> tuple[complex, complex]:
        trial_positions = linear_channel_positions_by_label(first, second, selected)
        transform = mobius_to_zero_one_infinity(
            trial_positions[ordering[0]],
            trial_positions[ordering[3]],
            trial_positions[ordering[4]],
        )
        return (
            complex(transform(trial_positions[ordering[1]])),
            complex(transform(trial_positions[ordering[2]])),
        )

    plus1 = moving_coordinates(channel.q1 + step, channel.q2)
    minus1 = moving_coordinates(channel.q1 - step, channel.q2)
    plus2 = moving_coordinates(channel.q1, channel.q2 + step)
    minus2 = moving_coordinates(channel.q1, channel.q2 - step)
    first_derivative = tuple((plus1[i] - minus1[i]) / (2.0 * step) for i in range(2))
    second_derivative = tuple((plus2[i] - minus2[i]) / (2.0 * step) for i in range(2))
    finite_difference = (
        first_derivative[0] * second_derivative[1]
        - second_derivative[0] * first_derivative[1]
    )
    error = abs(analytic - finite_difference)
    print("\narbitrary fixed-label gauge")
    print(f"  Jacobian finite-difference error={error:.3e}")
    if error > 2.0e-8:
        raise AssertionError("the arbitrary-gauge bidisc Jacobian is incorrect")


def run() -> None:
    check_mobius_special_cases()
    check_known_linear_frame()
    check_random_atlas_coverage()
    check_forward_map_roundtrip_and_jacobian()
    check_arbitrary_gauge_jacobian()
    print("\nall sphere five-point Liouville atlas checks passed")


if __name__ == "__main__":
    run()
