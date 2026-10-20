#!/usr/bin/env python3
"""PBW anchors for the c-only genus-one three-point production backend.

The full mode includes the order-(8,3,3) necklace tensor at which the former
h-recursion had a 39.6% same-order block error. Neither amplitude targets nor
agreement between recursion implementations are used as the test oracle.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from genus1_two_point_worldsheet import MomentumRule
from torus_three_point_blocks import necklace_descendant_coefficients_three_point
from torus_three_point_blocks import pair_ope_direct_coefficients, comb_ope_direct_coefficients
from torus_c_recursion_numerics import (
    stable_c_recursion_coefficients, C_RECURSION_NUMERICS_VERSION,
    CRecursionAccuracyError,
)


EXTERNAL = (.96484375, .96484375, .859375)


def check_case(kind, internal, orders):
    started = time.monotonic()
    tested, regulated, error = stable_c_recursion_coefficients(
        kind, external_weights=EXTERNAL, internal_weights=internal, orders=orders)
    if kind == "necklace":
        expected = necklace_descendant_coefficients_three_point(25., internal, EXTERNAL, orders)
    else:
        builder = pair_ope_direct_coefficients if kind == "pair" else comb_ope_direct_coefficients
        expected = builder(25., external_weights=EXTERNAL, internal_weights=internal, orders=orders)
    difference = np.abs(tested - expected)
    # Per-coefficient acceptance; a large unrelated coefficient must not hide
    # corruption of a small retained coefficient.
    assert np.all(difference <= 3e-11 + 3e-10 * np.abs(expected)), (
        kind, internal, orders, np.unravel_index(difference.argmax(), difference.shape),
        float(difference.max()))
    powers = tuple(np.asarray([q**n for n in range(m + 1)])
                   for q, m in zip((.6, .2, .01), orders))
    reference_block = np.einsum("ijk,i,j,k", expected, *powers)
    computed_block = np.einsum("ijk,i,j,k", tested, *powers)
    return dict(kind=kind, weights=list(internal), orders=list(orders),
                max_absolute_coefficient_error=float(difference.max()),
                max_scaled_coefficient_error=float(np.max(difference / np.maximum(1., abs(expected)))),
                relative_polynomial_error=float(abs(computed_block-reference_block) / max(abs(reference_block), 1e-280)),
                weight_limit_used=int(regulated), arithmetic_disagreement=float(error),
                seconds=time.monotonic()-started)


def run(full=False):
    # c=-2,h=1 has a singular level-two Gram matrix. A physical target pole
    # must not be replaced by a finite generic-weight extrapolation.
    try:
        stable_c_recursion_coefficients(
            "necklace", central_charge=-2., external_weights=EXTERNAL,
            internal_weights=(1., 1.29, 1.41), orders=(2, 0, 0))
    except CRecursionAccuracyError:
        pass
    else:
        raise AssertionError("a true target-c pole was silently regularized")
    rules = tuple(MomentumRule.power_legendre(5., 14, 2.+.137*e) for e in range(3))
    endpoint = tuple(float(1.+r.nodes[0]**2) for r in rules)
    cases = [(kind, h, (3, 2, 2))
             for kind in ("necklace", "pair", "comb")
             for h in ((1.04, 1.49, 2.44), endpoint, (2., 2., 2.))]
    if full:
        cases += [("necklace", (10., 1.1, 1.2), (8, 3, 3)),
                  ("pair", endpoint, (4, 3, 3)),
                  ("comb", endpoint, (4, 3, 3))]
    results = []
    for kind, h, orders in cases:
        row = check_case(kind, h, orders)
        results.append(row)
        print(json.dumps(row), flush=True)
    print(json.dumps(dict(numerics=C_RECURSION_NUMERICS_VERSION,
                          pbw_cases_passed=len(results), full=full)), flush=True)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    run(parser.parse_args().full)
