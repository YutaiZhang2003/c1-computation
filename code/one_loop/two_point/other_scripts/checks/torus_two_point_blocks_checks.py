#!/usr/bin/env python3
"""Checks for native-q torus two-point c recursion and its independent baseline."""

from __future__ import annotations

from c1_paths import PACKAGE_ROOT, data_root, package_path, module_path

import cmath
import math

import numpy as np

try:
    from torus_descendant_blocks import torus_one_point_descendant_coefficients
    from torus_two_point_blocks import (
        elliptic_nome,
        modular_lambda_series,
        necklace_c_recursion_coefficients,
        necklace_flat_block,
        necklace_descendant_coefficients,
        ope_c_recursion_coefficients,
        ope_coefficients_in_z,
        ope_descendant_coefficients,
    )
except ImportError:  # pragma: no cover
    from torus_descendant_blocks import torus_one_point_descendant_coefficients
    from torus_two_point_blocks import (
        elliptic_nome,
        modular_lambda_series,
        necklace_c_recursion_coefficients,
        necklace_flat_block,
        necklace_descendant_coefficients,
        ope_c_recursion_coefficients,
        ope_coefficients_in_z,
        ope_descendant_coefficients,
    )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def check_level_zero_and_one() -> None:
    c = 25.0
    h1 = 1.37
    h2 = 1.82
    d = 0.91
    necklace = necklace_descendant_coefficients(c, h1, h2, d, d, 1, 1)
    ope = ope_descendant_coefficients(c, h1, h2, d, d, 1, 1)
    print("level-zero checks")
    print(f"  necklace[0,0]={necklace[0,0]!r}")
    print(f"  ope[0,0]={ope[0,0]!r}")
    require(abs(necklace[0, 0] - 1.0) < 1.0e-13, "wrong necklace primary term")
    require(abs(ope[0, 0] - 1.0) < 1.0e-13, "wrong OPE primary term")
    require(np.all(np.isfinite(necklace)), "nonfinite necklace level-one coefficients")
    require(np.all(np.isfinite(ope)), "nonfinite OPE level-one coefficients")


def check_identity_necklace_reduces_to_character() -> None:
    c = 25.0
    h = 1.43
    order = 4
    coefficients = necklace_descendant_coefficients(c, h, h, 0.0, 0.0, order, order)
    # Identity insertions force equal descendant levels.  The diagonal is the
    # Verma-module partition number p(n).
    expected = np.array([1.0, 1.0, 2.0, 3.0, 5.0])
    diagonal = np.diag(coefficients)
    off_diagonal = coefficients - np.diag(diagonal)
    print("\nidentity necklace check")
    print(f"  diagonal={diagonal.tolist()}")
    print(f"  max off diagonal={np.max(np.abs(off_diagonal)):.6e}")
    require(np.max(np.abs(diagonal - expected)) < 2.0e-10, "wrong torus character")
    require(np.max(np.abs(off_diagonal)) < 2.0e-10, "identity changes descendant level")


def check_ope_zero_ope_level_is_torus_one_point() -> None:
    c = 25.0
    h_loop = 1.37
    h_ope = 1.82
    d = 0.91
    order = 4
    coefficients = ope_descendant_coefficients(c, h_loop, h_ope, d, d, order, 0)
    torus = torus_one_point_descendant_coefficients(c, h_loop, h_ope, (), order)
    difference = coefficients[:, 0] - np.asarray(torus)
    print("\nOPE m=0 torus one-point check")
    print(f"  max difference={np.max(np.abs(difference)):.6e}")
    require(np.max(np.abs(difference)) < 2.0e-10, "OPE loop contraction is wrong")


def check_ope_c_recursion_against_descendants() -> None:
    cases = (
        (26.215, 1.31, 1.73, 0.41, 0.56),
        (25.0, 1.37, 1.82, 0.91, 0.91),
        (25.0, 1.50, 1.50, 0.96, 0.96),
        (8.7, 1.07, 0.91, 0.31, 0.47),
    )
    largest_error = 0.0
    for c, h_loop, h_ope, d1, d2 in cases:
        direct = ope_descendant_coefficients(
            c,
            h_loop,
            h_ope,
            d1,
            d2,
            4,
            5,
        )
        recursive = ope_c_recursion_coefficients(
            c,
            h_loop,
            h_ope,
            d1,
            d2,
            4,
            5,
        )
        largest_error = max(
            largest_error,
            float(
                np.max(
                    np.abs(recursive - direct)
                    / np.maximum(1.0, np.abs(direct))
                )
            ),
        )
    print("\nOPE CCY c-recursion check")
    print(f"  max relative coefficient error={largest_error:.6e}")
    require(
        largest_error < 5.0e-11,
        "OPE c-recursion disagrees with direct descendant sewing",
    )


def check_necklace_c_recursion_against_descendants() -> None:
    from torus_c_recursion_numerics import CRecursionAccuracyError

    try:
        necklace_c_recursion_coefficients(-2., 1., 1.29, .91, .81, 2, 0)
    except CRecursionAccuracyError:
        pass
    else:
        raise AssertionError("a true target-c pole was silently regularized")
    cases = (
        (29.0, 1.31, 1.73, 0.87, 1.09, (3, 3)),
        (25.0, 1.37, 1.82, 0.91, 0.91, (6, 3)),
        (25.0, 1.0, 1.0, 0.9375, 0.9375, (4, 3)),
        (25.0, 1.25, 1.25, 0.9375, 0.9375, (6, 3)),
        (25.0, 10.0, 30.0, 0.91, 0.91, (6, 3)),
    )
    maximum = 0.0
    for c, h1, h2, d1, d2, orders in cases:
        direct = necklace_descendant_coefficients(c, h1, h2, d1, d2, *orders)
        recursive = necklace_c_recursion_coefficients(c, h1, h2, d1, d2, *orders)
        maximum = max(maximum, float(np.max(abs(recursive-direct)/np.maximum(abs(direct), 1))))
        np.testing.assert_allclose(recursive, direct, rtol=2e-10, atol=2e-11)
        # Check the evaluated block against an independent native-q sum,
        # including a point close to the edge of the q1 convergence disc.
        for z, tau in ((1.1+0.2j, 0.2+1.1j), (0.7+1.8j, 0.1+2j)):
            log1, log2 = 1j*z, 1j*(2*math.pi*tau-z)
            expected = sum(direct[n,m]*cmath.exp(n*log1+m*log2)
                           for n,m in np.ndindex(direct.shape))
            expected *= cmath.exp((h1-c/24)*log1+(h2-c/24)*log2)
            actual = necklace_flat_block(recursive,c=c,h1=h1,h2=h2,z=z,tau=tau)
            require(abs(actual-expected) < 2e-10*max(abs(expected),1e-100),
                    "necklace evaluator changed the sewing variables")
    character = necklace_c_recursion_coefficients(25.,1.43,1.43,0.,0.,4,4)
    np.testing.assert_allclose(character,np.diag([1.,1.,2.,3.,5.]),rtol=2e-11,atol=2e-11)
    print("\nnecklace native-q CCY c-recursion check")
    print(f"  max scaled coefficient error={maximum:.6e}")


def _lambda_numeric(q: complex, terms: int = 80) -> complex:
    theta2_reduced = sum(q ** (n * (n + 1)) for n in range(terms))
    theta2 = 2.0 * q ** 0.25 * theta2_reduced
    theta3 = 1.0 + 2.0 * sum(q ** (n * n) for n in range(1, terms))
    return (theta2 / theta3) ** 4


def check_sphere_elliptic_nome() -> None:
    order = 8
    series = modular_lambda_series(order)
    expected_head = np.array([0.0, 16.0, -128.0, 704.0, -3072.0])
    print("\nmodular lambda check")
    print(f"  head={series[:5].tolist()}")
    require(np.max(np.abs(series[:5] - expected_head)) < 1.0e-10, "wrong lambda series")

    hat_q = 0.012 + 0.007j
    lambda_truncated = sum(series[n] * hat_q**n for n in range(order + 1))
    lambda_direct = _lambda_numeric(hat_q)
    require(abs(lambda_truncated - lambda_direct) < 2.0e-11, "lambda series evaluation failed")
    recovered = elliptic_nome(lambda_direct)
    print(f"  |E(lambda(q))-q|={abs(recovered-hat_q):.6e}")
    require(abs(recovered - hat_q) < 2.0e-11, "elliptic nome is not inverse to lambda")



def check_ope_z_reexpansion() -> None:
    rng = np.random.default_rng(4321)
    original = rng.normal(size=(3, 5)) + 1j * rng.normal(size=(3, 5))
    transformed = ope_coefficients_in_z(original, 12)
    q = 0.002 + 0.001j
    z = 0.08 + 0.04j
    v = cmath.exp(-1j * z) - 1.0
    direct = sum(original[n, m] * q**n * v**m for n in range(3) for m in range(5))
    recomposed = sum(
        transformed[n, m] * q**n * z**m
        for n in range(3)
        for m in range(13)
    )
    print("\nOPE z re-expansion check")
    print(f"  difference={abs(direct-recomposed):.6e}")
    require(abs(direct - recomposed) < 2.0e-11, "OPE z re-expansion failed")


def run() -> None:
    check_level_zero_and_one()
    check_identity_necklace_reduces_to_character()
    check_ope_zero_ope_level_is_torus_one_point()
    check_ope_c_recursion_against_descendants()
    check_necklace_c_recursion_against_descendants()
    check_sphere_elliptic_nome()
    check_ope_z_reexpansion()
    print("\nall torus two-point block checks passed")


if __name__ == "__main__":
    run()
