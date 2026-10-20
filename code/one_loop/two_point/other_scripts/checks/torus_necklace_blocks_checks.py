#!/usr/bin/env python3
"""Low-level checks for torus two- and three-point h-recursion blocks."""

from __future__ import annotations

from c1_paths import PACKAGE_ROOT, data_root, package_path, module_path

from itertools import product

import numpy as np

try:
    from torus_descendant_blocks import gram_matrix, rho_primary_external
    from torus_two_point_blocks import necklace_descendant_coefficients
    from virasoro_blocks import (
        TorusThreePointVirasoroBlock,
        TorusTwoPointVirasoroBlock,
    )
except ImportError:  # pragma: no cover - supports package-style execution
    from torus_descendant_blocks import gram_matrix, rho_primary_external
    from torus_two_point_blocks import necklace_descendant_coefficients
    from virasoro_blocks import (
        TorusThreePointVirasoroBlock,
        TorusTwoPointVirasoroBlock,
    )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def direct_necklace_descendant_coefficients(
    c: complex,
    internal_weights: tuple[complex, ...],
    external_weights: tuple[complex, ...],
    orders: tuple[int, ...],
) -> np.ndarray:
    """Evaluate the defining descendant sum for a small necklace block."""

    shape = tuple(order + 1 for order in orders)
    coefficients = np.zeros(shape, dtype=np.complex128)
    point_count = len(internal_weights)
    for levels in np.ndindex(shape):
        edge_data: list[tuple[tuple[tuple[int, ...], ...], np.ndarray]] = []
        for weight, level in zip(internal_weights, levels):
            basis, gram = gram_matrix(weight, c, level)
            edge_data.append((basis, np.linalg.inv(gram)))
        dimensions = tuple(len(basis) for basis, _ in edge_data)

        value = 0.0j
        for ket_indices in product(*(range(dimension) for dimension in dimensions)):
            for bra_indices in product(*(range(dimension) for dimension in dimensions)):
                term = 1.0 + 0.0j
                for edge in range(point_count):
                    next_edge = (edge + 1) % point_count
                    basis, inverse_gram = edge_data[edge]
                    next_basis, _ = edge_data[next_edge]
                    term *= inverse_gram[ket_indices[edge], bra_indices[edge]]
                    term *= rho_primary_external(
                        basis[bra_indices[edge]],
                        next_basis[ket_indices[next_edge]],
                        internal_weights[edge],
                        external_weights[edge],
                        internal_weights[next_edge],
                        c,
                    )
                value += term
        coefficients[levels] = value
    return coefficients


def check_two_point_against_direct_descendants() -> None:
    c = 30.0
    internal_weights = (1.37, 1.82)
    external_weights = (0.91, 0.73)
    orders = (3, 3)
    recursion = TorusTwoPointVirasoroBlock(
        c,
        *internal_weights,
        *external_weights,
    ).descendant_coefficients(orders)
    direct = necklace_descendant_coefficients(
        c,
        *internal_weights,
        *external_weights,
        *orders,
    )
    error = float(np.max(np.abs(recursion - direct)))
    print("torus two-point h-recursion")
    print(f"  max coefficient error through (3,3): {error:.6e}")
    require(error < 2.0e-11, "two-point h-recursion disagrees with descendant sums")


def check_three_point_against_direct_descendants() -> None:
    c = 30.0
    internal_weights = (1.17, 1.43, 1.88)
    external_weights = (0.31, 0.52, 0.79)
    orders = (2, 2, 2)
    recursion = TorusThreePointVirasoroBlock(
        c,
        *internal_weights,
        *external_weights,
    ).descendant_coefficients(orders)
    direct = direct_necklace_descendant_coefficients(
        c,
        internal_weights,
        external_weights,
        orders,
    )
    error = float(np.max(np.abs(recursion - direct)))
    print("\ntorus three-point h-recursion")
    print(f"  max coefficient error through (2,2,2): {error:.6e}")
    require(error < 2.0e-11, "three-point h-recursion disagrees with descendant sums")


def check_rectangular_orders_and_prefactor() -> None:
    c = 30.0
    internal_weights = (1.21, 1.54, 1.93)
    external_weights = (0.29, 0.48, 0.81)
    q_values = (0.011 + 0.002j, -0.008 + 0.003j, 0.006 - 0.001j)
    orders = (1, 2, 3)
    block = TorusThreePointVirasoroBlock(
        c,
        *internal_weights,
        *external_weights,
    )
    coefficients = block.descendant_coefficients(orders)
    descendant_value = sum(
        coefficients[levels]
        * np.prod([q**level for q, level in zip(q_values, levels)])
        for levels in np.ndindex(coefficients.shape)
    )
    expected = descendant_value * np.prod(
        [
            q ** (weight - c / 24.0)
            for q, weight in zip(q_values, internal_weights)
        ]
    )
    actual = block.chiral_block(q_values, orders)

    print("\nrectangular truncation and primary propagation")
    print(f"  coefficient shape: {coefficients.shape}")
    print(f"  |direct evaluation - chiral_block|: {abs(actual-expected):.6e}")
    require(coefficients.shape == (2, 3, 4), "rectangular orders were not preserved")
    require(abs(actual - expected) < 2.0e-14, "primary propagation prefactor is wrong")


def check_input_validation() -> None:
    generic = TorusThreePointVirasoroBlock(
        30.0,
        1.1,
        1.4,
        1.8,
        0.3,
        0.5,
        0.7,
    )
    try:
        generic.descendant_coefficients((2, 2))
    except ValueError:
        pass
    else:
        raise AssertionError("wrong number of truncation orders was accepted")

    try:
        generic.chiral_block((0.01, 0.02), 2)
    except ValueError:
        pass
    else:
        raise AssertionError("wrong number of necklace nomes was accepted")


def run() -> None:
    check_two_point_against_direct_descendants()
    check_three_point_against_direct_descendants()
    check_rectangular_orders_and_prefactor()
    check_input_validation()
    print("\nall torus necklace h-recursion checks passed")


if __name__ == "__main__":
    run()
