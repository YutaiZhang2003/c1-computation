#!/usr/bin/env python3
"""Evaluate the CCY Liouville genus-two recursion near the Bolza point.

The Bolza curve is represented by the standard period matrix

    Omega_B = 1/2 [[1 + i sqrt(2), 1], [1, 1 + i sqrt(2)]].

The raw Bolza representative is not a useful theta-graph plumbing point: the
leading CCY edge q for the off-diagonal entry lies on the unit circle.  This
driver applies a fixed Sp(4,Z) move to the same marked surface, then an
integral B-translation to put logarithms on the principal theta branch.  The
resulting leading theta q-values have absolute value about 0.0517.

At present the script uses those leading theta q-values for the CCY recursion.
It can also attempt the all-order theta inverse, but that inverse is not yet
reliable at the Bolza point and is therefore not the default.
"""

from __future__ import annotations

from c1_paths import PACKAGE_ROOT, data_root, package_path, module_path

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    from liouville_genus2 import format_complex
    from liouville_genus2_ccy import liouville_genus2_ccy_partition
    from plumbing_algorithms import solve_theta_inverse_from_omega
except ImportError:  # pragma: no cover - supports package-style execution
    from liouville_genus2 import format_complex
    from liouville_genus2_ccy import liouville_genus2_ccy_partition
    from plumbing_algorithms import solve_theta_inverse_from_omega


def _block(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    return np.block([[a, b], [c, d]]).astype(int)


def _sp_generators() -> dict[str, np.ndarray]:
    identity = np.eye(2, dtype=int)
    zero = np.zeros((2, 2), dtype=int)
    s_matrix = _block(zero, -identity, identity, zero)
    shear = np.asarray([[1, 1], [0, 1]], dtype=int)
    gl_shear = _block(shear, zero, zero, np.linalg.inv(shear).T.astype(int))
    swap = np.asarray([[0, 1], [1, 0]], dtype=int)
    handle_swap = _block(swap, zero, zero, swap)
    return {
        "S": s_matrix,
        "GL": gl_shear,
        "SW": handle_swap,
    }


def bolza_period_matrix() -> np.ndarray:
    return np.asarray(
        [
            [0.5 + 0.5j * math.sqrt(2.0), 0.5],
            [0.5, 0.5 + 0.5j * math.sqrt(2.0)],
        ],
        dtype=np.complex128,
    )


def symplectic_transform(matrix: np.ndarray, omega: np.ndarray) -> np.ndarray:
    a = matrix[:2, :2]
    b = matrix[:2, 2:]
    c = matrix[2:, :2]
    d = matrix[2:, 2:]
    return (a @ omega + b) @ np.linalg.inv(c @ omega + d)


def bolza_theta_chart_transform() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    generators = _sp_generators()
    matrix = np.eye(4, dtype=int)
    for name in ("S", "GL", "SW"):
        matrix = generators[name] @ matrix

    branch_shift = np.asarray([[0, 1], [1, 1]], dtype=int)
    translation = _block(np.eye(2, dtype=int), branch_shift, np.zeros((2, 2), dtype=int), np.eye(2, dtype=int))
    total_matrix = translation @ matrix
    return matrix, translation, total_matrix


def leading_theta_q_from_omega(omega: np.ndarray) -> tuple[complex, complex, complex]:
    two_pi_i = 2.0j * math.pi
    q3 = np.exp(two_pi_i * omega[0, 1])
    q1 = np.exp(two_pi_i * (omega[0, 0] - omega[0, 1]))
    q2 = np.exp(two_pi_i * (omega[1, 1] - omega[0, 1]))
    return complex(q1), complex(q2), complex(q3)


def modular_det_factor(matrix: np.ndarray, omega: np.ndarray) -> complex:
    c = matrix[2:, :2]
    d = matrix[2:, 2:]
    return complex(np.linalg.det(c @ omega + d))


@dataclass(frozen=True)
class BolzaCCYResult:
    b: float
    central_charge: float
    block_order: int
    p_max: float
    quadrature_order: int
    dps: int
    q1: str
    q2: str
    q3: str
    q_abs: float
    modular_det_abs: float
    value: str
    value_abs: float
    note: str


def print_matrix(name: str, matrix: np.ndarray) -> None:
    print(f"  {name}:")
    for row in matrix:
        print("    " + "  ".join(format_complex(complex(value)) for value in row))


def run(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate CCY recursion at the Bolza point in a theta plumbing chart.")
    parser.add_argument("--b", type=float, default=1.0)
    parser.add_argument("--block-order", type=int, default=3)
    parser.add_argument("--p-max", type=float, default=2.0)
    parser.add_argument("--quadrature-order", type=int, default=5)
    parser.add_argument("--dps", type=int, default=24)
    parser.add_argument("--vacuum-word-len", type=int, default=2)
    parser.add_argument("--vacuum-oscillator-level-max", type=int, default=6)
    parser.add_argument("--no-vacuum-seed", action="store_true")
    parser.add_argument("--attempt-inverse", action="store_true")
    parser.add_argument("--inverse-word-len", type=int, default=3)
    parser.add_argument("--inverse-b-order", type=int, default=140)
    parser.add_argument("--out-json", type=Path)
    args = parser.parse_args(argv)

    omega_bolza = bolza_period_matrix()
    chart_matrix, branch_translation, total_matrix = bolza_theta_chart_transform()
    omega_chart_raw = symplectic_transform(chart_matrix, omega_bolza)
    omega_chart = symplectic_transform(total_matrix, omega_bolza)
    q1, q2, q3 = leading_theta_q_from_omega(omega_chart)
    determinant = modular_det_factor(total_matrix, omega_bolza)

    print("Bolza CCY recursion pilot")
    print_matrix("Omega_Bolza", omega_bolza)
    print_matrix("Omega_theta_raw", omega_chart_raw)
    print_matrix("Omega_theta_principal_branch", omega_chart)
    print("  Sp(4,Z) word: S GL SW, followed by B-shift [[0,1],[1,1]]")
    print(f"  |det(C Omega + D)|={abs(determinant):.12e}")
    print(f"  leading q1={format_complex(q1)}  |q1|={abs(q1):.12e}")
    print(f"  leading q2={format_complex(q2)}  |q2|={abs(q2):.12e}")
    print(f"  leading q3={format_complex(q3)}  |q3|={abs(q3):.12e}")

    if args.attempt_inverse:
        inverse = solve_theta_inverse_from_omega(
            omega_chart,
            max_word_len=args.inverse_word_len,
            b_order=args.inverse_b_order,
            max_nfev=80,
        )
        print("  theta inverse attempt:")
        print(f"    success={inverse.success}, nfev={inverse.nfev}, residual={inverse.max_abs_residual:.12e}")
        print(f"    q0={format_complex(inverse.q_zero)}  |q0|={abs(inverse.q_zero):.12e}")
        print(f"    q1={format_complex(inverse.q_one)}  |q1|={abs(inverse.q_one):.12e}")
        print(f"    qi={format_complex(inverse.q_infty)}  |qi|={abs(inverse.q_infty):.12e}")
        print(f"    health={inverse.health_message}")

    result = liouville_genus2_ccy_partition(
        b=args.b,
        q1=q1,
        q2=q2,
        q3=q3,
        block_order=args.block_order,
        p_max=args.p_max,
        quadrature_order=args.quadrature_order,
        dps=args.dps,
        propagator_shift=0.0,
        include_vacuum_seed=not args.no_vacuum_seed,
        vacuum_word_len=args.vacuum_word_len,
        vacuum_oscillator_level_max=args.vacuum_oscillator_level_max,
        include_cosmological_prefactor=False,
        store_samples=False,
    )

    note = "leading theta q-values; exact all-order theta inverse not yet used"
    summary = BolzaCCYResult(
        b=result.b,
        central_charge=result.central_charge,
        block_order=result.block_order,
        p_max=result.p_max,
        quadrature_order=result.quadrature_order,
        dps=result.dps,
        q1=format_complex(q1),
        q2=format_complex(q2),
        q3=format_complex(q3),
        q_abs=abs(q1),
        modular_det_abs=abs(determinant),
        value=format_complex(result.value),
        value_abs=abs(result.value),
        note=note,
    )
    print("  CCY Liouville partition:")
    print(f"    b={summary.b:.12g}, c={summary.central_charge:.12g}")
    print(f"    block order={summary.block_order}, quadrature order={summary.quadrature_order}, Pmax={summary.p_max:.12g}")
    print(f"    value={summary.value}")
    print(f"    note={summary.note}")

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(asdict(summary), indent=2) + "\n")
        print(f"  wrote {args.out_json}")


if __name__ == "__main__":
    run()
