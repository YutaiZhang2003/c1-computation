#!/usr/bin/env python3
"""Low-cost checks for the first sphere-five Liouville contour crossing."""

from __future__ import annotations

from c1_paths import PACKAGE_ROOT, data_root, package_path, module_path

import numpy as np

try:
    from sphere_five_point_equal_energy import (
        EqualEnergyFivePointKernel,
        _gauss_legendre_grid,
    )
except ImportError:  # pragma: no cover - package-style execution
    from sphere_five_point_equal_energy import (
        EqualEnergyFivePointKernel,
        _gauss_legendre_grid,
    )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def check_power_rule_moments() -> None:
    for power in (1.0, 1.25, 2.0):
        nodes, weights = _gauss_legendre_grid(12, 6.0, power=power)
        zeroth = float(np.sum(weights))
        first = float(np.sum(weights * nodes))
        require(abs(zeroth - 6.0) < 2.0e-13, "power rule misses a constant")
        # For fractional power the transformed linear moment is not a
        # polynomial in the Jacobi coordinate, but should already be accurate
        # at this deliberately low order.
        require(abs(first - 18.0) < 2.0e-5, "power rule misses a linear moment")


def check_continuation_wall_and_residue() -> None:
    below_real = EqualEnergyFivePointKernel(
        0.38j,
        block_order=2,
        momentum_order=4,
        momentum_maximum=4.0,
        momentum_power=1.25,
        block_scheme="c",
        liouville_contour="real",
        special_dps=25,
    )
    below_continued = EqualEnergyFivePointKernel(
        0.38j,
        block_order=2,
        momentum_order=4,
        momentum_maximum=4.0,
        momentum_power=1.25,
        block_scheme="c",
        liouville_contour="continued",
        special_dps=25,
    )
    require(below_continued.crossed_pole is None, "a residue appeared below t=2/5")
    for real_entries, continued_entries in zip(
        below_real.entries_by_incoming_slot,
        below_continued.entries_by_incoming_slot,
    ):
        require(
            all(
                abs(left.log_weighted_structure_constant - right.log_weighted_structure_constant)
                < 1.0e-14
                for left, right in zip(real_entries, continued_entries)
            ),
            "continued and real grids differ below the first wall",
        )

    above = EqualEnergyFivePointKernel(
        0.42j,
        block_order=2,
        momentum_order=4,
        momentum_maximum=4.0,
        momentum_power=1.25,
        block_scheme="c",
        liouville_contour="continued",
        special_dps=35,
    )
    require(abs(above.crossed_pole - 0.05j) < 2.0e-15, "crossed pole is misplaced")
    require(
        abs(above.crossed_cherry_residue - 0.23955696) < 2.0e-7,
        "crossed DOZZ residue changed",
    )
    require(not above.discrete_entries_by_incoming_slot[2], "middle channel got a residue")
    require(
        all(above.discrete_entries_by_incoming_slot[index] for index in (0, 1, 3, 4)),
        "an incoming--outgoing cherry is missing its residue",
    )


def check_second_wall_guard() -> None:
    try:
        EqualEnergyFivePointKernel(
            0.5j,
            block_order=0,
            momentum_order=2,
            block_scheme="c",
            liouville_contour="continued",
        )
    except ValueError:
        return
    raise AssertionError("the first-wall continuation was used at the second pinch")


def main() -> None:
    check_power_rule_moments()
    check_continuation_wall_and_residue()
    check_second_wall_guard()
    print("all sphere-five contour-continuation checks passed")


if __name__ == "__main__":
    main()
