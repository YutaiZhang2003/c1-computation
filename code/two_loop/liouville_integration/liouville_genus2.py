#!/usr/bin/env python3
"""Genus-two Liouville partition function in the pair-of-tori plumbing channel.

The first implemented approximation is the primary term in the separating
bridge sewing.  It keeps the full torus one-point Virasoro blocks on the two
genus-one components, and integrates over the primary Liouville momentum
propagating through the bridge.
"""

from __future__ import annotations

from c1_paths import PACKAGE_ROOT, data_root, package_path, module_path

import argparse
import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np

try:
    from liouville_torus import (
        UpsilonB,
        estimate_p_max,
        lambda_from_yin_momentum,
        liouville_weight_from_lambda,
        q_from_tau,
        validate_nonresonant_b_for_block,
        yin_structure_constant_momentum,
    )
    from virasoro_blocks import TorusOnePointVirasoroBlock
except ImportError:  # pragma: no cover - supports package-style execution
    from liouville_torus import (
        UpsilonB,
        estimate_p_max,
        lambda_from_yin_momentum,
        liouville_weight_from_lambda,
        q_from_tau,
        validate_nonresonant_b_for_block,
        yin_structure_constant_momentum,
    )
    from virasoro_blocks import TorusOnePointVirasoroBlock


def parse_complex(value: str) -> complex:
    return complex(value.replace("i", "j"))


def format_complex(value: complex) -> str:
    return f"{value.real:+.12e}{value.imag:+.12e}j"


def _validate_plumbing_q(name: str, value: complex) -> complex:
    value = complex(value)
    if not 0 < abs(value) < 1:
        raise ValueError(f"{name} must satisfy 0 < |{name}| < 1")
    return value


def liouville_weight_from_yin_momentum(b: float, momentum: float | complex) -> complex:
    """Return h_P=Q^2/4+P^2 for the Xi/Yin Liouville momentum P."""
    q_background = b + 1.0 / b
    momentum = complex(momentum)
    return 0.25 * q_background * q_background + momentum * momentum


def bridge_primary_sewing_exponent(
    *,
    b: float,
    bridge_momentum: float,
    include_vacuum_energy: bool = True,
) -> float:
    """Return the real exponent in the primary bridge factor.

    With the default cylinder vacuum-energy shift this is h_P-c/24.  For real
    Liouville momentum it simplifies to P^2-1/24, but the unsimplified formula
    is kept here to make the sewing convention explicit.
    """
    q_background = b + 1.0 / b
    central_charge = 1.0 + 6.0 * q_background * q_background
    exponent = liouville_weight_from_yin_momentum(b, bridge_momentum)
    if include_vacuum_energy:
        exponent -= central_charge / 24.0
    if abs(exponent.imag) > 1.0e-13:
        raise ValueError("bridge exponent is not real for the supplied momentum")
    return float(exponent.real)


def bridge_primary_sewing_factor(
    q_bridge: complex,
    *,
    b: float,
    bridge_momentum: float,
    include_vacuum_energy: bool = True,
) -> float:
    """Return |q_bridge|^(2(h_P-c/24)) for the primary bridge state."""
    q_bridge = _validate_plumbing_q("q_bridge", q_bridge)
    exponent = bridge_primary_sewing_exponent(
        b=b,
        bridge_momentum=bridge_momentum,
        include_vacuum_energy=include_vacuum_energy,
    )
    return abs(q_bridge) ** (2.0 * exponent)


@dataclass(frozen=True)
class LiouvilleGenus2BridgeSample:
    bridge_momentum: float
    bridge_measure_weight: float
    bridge_sewing_factor: float
    left_torus_one_point: complex
    right_torus_one_point: complex
    contribution: complex


@dataclass(frozen=True)
class LiouvilleGenus2PairOfToriResult:
    value: complex
    q1: complex
    q2: complex
    q_bridge: complex
    b: float
    central_charge: float
    block_order: int
    bridge_p_max: float
    handle_p_max_left: float
    handle_p_max_right: float
    bridge_quadrature_order: int
    handle_quadrature_order: int
    dps: int
    include_bridge_vacuum_energy: bool
    include_cosmological_prefactor: bool
    samples: tuple[LiouvilleGenus2BridgeSample, ...]


class _TorusOnePointWorkspace:
    """Reusable handle-momentum quadrature for one torus component."""

    def __init__(
        self,
        *,
        b: float,
        q: complex,
        block_order: int,
        p_max: float,
        quadrature_order: int,
        special: UpsilonB,
        mu: complex,
        include_cosmological_prefactor: bool,
    ) -> None:
        if p_max <= 0:
            raise ValueError("handle p_max must be positive")
        if quadrature_order <= 0:
            raise ValueError("handle quadrature_order must be positive")

        self.b = b
        self.q = _validate_plumbing_q("q", q)
        self.block_order = int(block_order)
        self.p_max = float(p_max)
        self.quadrature_order = int(quadrature_order)
        self.special = special
        self.mu = complex(mu)
        self.include_cosmological_prefactor = bool(include_cosmological_prefactor)

        self.q_background = b + 1.0 / b
        self.central_charge = 1.0 + 6.0 * self.q_background * self.q_background
        nodes, weights = np.polynomial.legendre.leggauss(self.quadrature_order)
        midpoint = 0.5 * self.p_max
        self._nodes = tuple(float(midpoint * (node + 1.0)) for node in nodes)
        self._measure_weights = tuple(float(midpoint * weight / math.pi) for weight in weights)

    def one_point(self, external_momentum: float) -> complex:
        external_lambda = lambda_from_yin_momentum(external_momentum)
        external_weight = liouville_weight_from_lambda(self.b, external_lambda)
        total = 0.0 + 0.0j
        for p, measure_weight in zip(self._nodes, self._measure_weights):
            internal_weight = 0.25 * self.q_background * self.q_background + p * p
            structure_constant = yin_structure_constant_momentum(
                self.special,
                p,
                external_momentum,
                p,
                mu=self.mu,
                include_cosmological_prefactor=self.include_cosmological_prefactor,
            )
            block = TorusOnePointVirasoroBlock(
                self.central_charge,
                internal_weight,
                external_weight,
                b=self.b,
                external_lambda=external_lambda,
            )
            chiral_block = block.chiral_block_exact_eta(self.q, self.block_order)
            total += measure_weight * structure_constant * abs(chiral_block) ** 2
        return total


def liouville_genus2_pair_of_tori(
    *,
    b: float,
    q1: complex,
    q2: complex,
    q_bridge: complex,
    block_order: int,
    bridge_p_max: float | None = None,
    handle_p_max: float | None = None,
    handle_p_max_left: float | None = None,
    handle_p_max_right: float | None = None,
    bridge_quadrature_order: int = 8,
    handle_quadrature_order: int = 12,
    dps: int = 40,
    mu: complex = 1.0,
    include_bridge_vacuum_energy: bool = True,
    include_cosmological_prefactor: bool = False,
    tail_tolerance: float = 1.0e-14,
    safety_margin: float = 1.0,
) -> LiouvilleGenus2PairOfToriResult:
    r"""Compute the leading separating-channel genus-two Liouville partition.

    The implemented quantity is

        int dP3/pi |q_bridge|^(2(h3-c/24)) G(P3,q1) G(P3,q2),

    where G(Pext,q) is the full scalar Liouville torus one-point integral for
    the external primary V_{Pext}.  This includes all handle descendants through
    the torus one-point blocks, but keeps only the primary bridge state.  Bridge
    descendants start at order q_bridge and are a separate future recursion.
    """
    if b <= 0:
        raise ValueError("b must be positive")
    if block_order < 0:
        raise ValueError("block_order must be non-negative")
    if bridge_quadrature_order <= 0:
        raise ValueError("bridge_quadrature_order must be positive")

    q1 = _validate_plumbing_q("q1", q1)
    q2 = _validate_plumbing_q("q2", q2)
    q_bridge = _validate_plumbing_q("q_bridge", q_bridge)
    validate_nonresonant_b_for_block(b, block_order)

    if bridge_p_max is None:
        bridge_p_max = estimate_p_max(
            q_bridge,
            tail_tolerance=tail_tolerance,
            safety_margin=safety_margin,
        )
    if handle_p_max is not None and (handle_p_max_left is not None or handle_p_max_right is not None):
        raise ValueError("use either handle_p_max or the left/right handle cutoffs, not both")
    if handle_p_max is None:
        if handle_p_max_left is None:
            handle_p_max_left = estimate_p_max(q1, tail_tolerance=tail_tolerance, safety_margin=safety_margin)
        if handle_p_max_right is None:
            handle_p_max_right = estimate_p_max(q2, tail_tolerance=tail_tolerance, safety_margin=safety_margin)
    else:
        handle_p_max_left = handle_p_max
        handle_p_max_right = handle_p_max
    if bridge_p_max <= 0 or handle_p_max_left <= 0 or handle_p_max_right <= 0:
        raise ValueError("all momentum cutoffs must be positive")

    special = UpsilonB(b=b, dps=dps)
    left = _TorusOnePointWorkspace(
        b=b,
        q=q1,
        block_order=block_order,
        p_max=float(handle_p_max_left),
        quadrature_order=handle_quadrature_order,
        special=special,
        mu=mu,
        include_cosmological_prefactor=include_cosmological_prefactor,
    )
    right = _TorusOnePointWorkspace(
        b=b,
        q=q2,
        block_order=block_order,
        p_max=float(handle_p_max_right),
        quadrature_order=handle_quadrature_order,
        special=special,
        mu=mu,
        include_cosmological_prefactor=include_cosmological_prefactor,
    )

    nodes, weights = np.polynomial.legendre.leggauss(bridge_quadrature_order)
    midpoint = 0.5 * float(bridge_p_max)
    samples: list[LiouvilleGenus2BridgeSample] = []
    total = 0.0 + 0.0j
    for node, weight in zip(nodes, weights):
        bridge_momentum = float(midpoint * (node + 1.0))
        bridge_measure_weight = float(midpoint * weight / math.pi)
        bridge_factor = bridge_primary_sewing_factor(
            q_bridge,
            b=b,
            bridge_momentum=bridge_momentum,
            include_vacuum_energy=include_bridge_vacuum_energy,
        )
        left_value = left.one_point(bridge_momentum)
        right_value = right.one_point(bridge_momentum)
        contribution = bridge_measure_weight * bridge_factor * left_value * right_value
        samples.append(
            LiouvilleGenus2BridgeSample(
                bridge_momentum=bridge_momentum,
                bridge_measure_weight=bridge_measure_weight,
                bridge_sewing_factor=bridge_factor,
                left_torus_one_point=left_value,
                right_torus_one_point=right_value,
                contribution=contribution,
            )
        )
        total += contribution

    q_background = b + 1.0 / b
    central_charge = 1.0 + 6.0 * q_background * q_background
    return LiouvilleGenus2PairOfToriResult(
        value=total,
        q1=q1,
        q2=q2,
        q_bridge=q_bridge,
        b=float(b),
        central_charge=float(central_charge),
        block_order=int(block_order),
        bridge_p_max=float(bridge_p_max),
        handle_p_max_left=float(handle_p_max_left),
        handle_p_max_right=float(handle_p_max_right),
        bridge_quadrature_order=int(bridge_quadrature_order),
        handle_quadrature_order=int(handle_quadrature_order),
        dps=int(dps),
        include_bridge_vacuum_energy=bool(include_bridge_vacuum_energy),
        include_cosmological_prefactor=bool(include_cosmological_prefactor),
        samples=tuple(samples),
    )


def _resolve_q(name: str, q_value: complex | None, tau_value: complex | None) -> complex:
    if (q_value is None) == (tau_value is None):
        raise ValueError(f"provide exactly one of --{name} or --tau-{name[-1]}")
    return q_value if q_value is not None else q_from_tau(tau_value)


def run(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the leading pair-of-tori genus-two Liouville partition function."
    )
    parser.add_argument("--b", type=float, required=True)
    parser.add_argument("--q1", type=parse_complex)
    parser.add_argument("--q2", type=parse_complex)
    parser.add_argument("--tau-1", type=parse_complex)
    parser.add_argument("--tau-2", type=parse_complex)
    parser.add_argument("--q-bridge", type=parse_complex, required=True)
    parser.add_argument("--block-order", type=int, default=3)
    parser.add_argument("--bridge-p-max", type=float)
    parser.add_argument("--handle-p-max", type=float)
    parser.add_argument("--handle-p-max-left", type=float)
    parser.add_argument("--handle-p-max-right", type=float)
    parser.add_argument("--bridge-quadrature-order", type=int, default=8)
    parser.add_argument("--handle-quadrature-order", type=int, default=12)
    parser.add_argument("--dps", type=int, default=40)
    parser.add_argument("--mu", type=parse_complex, default=1.0 + 0.0j)
    parser.add_argument(
        "--omit-bridge-vacuum-energy",
        action="store_true",
        help="use |q_bridge|^(2 h_P) instead of |q_bridge|^(2(h_P-c/24))",
    )
    parser.add_argument(
        "--include-cosmological-prefactor",
        action="store_true",
        help="include the momentum-independent DOZZ cosmological prefactor",
    )
    args = parser.parse_args(argv)

    try:
        q1 = _resolve_q("q1", args.q1, args.tau_1)
        q2 = _resolve_q("q2", args.q2, args.tau_2)
        result = liouville_genus2_pair_of_tori(
            b=args.b,
            q1=q1,
            q2=q2,
            q_bridge=args.q_bridge,
            block_order=args.block_order,
            bridge_p_max=args.bridge_p_max,
            handle_p_max=args.handle_p_max,
            handle_p_max_left=args.handle_p_max_left,
            handle_p_max_right=args.handle_p_max_right,
            bridge_quadrature_order=args.bridge_quadrature_order,
            handle_quadrature_order=args.handle_quadrature_order,
            dps=args.dps,
            mu=args.mu,
            include_bridge_vacuum_energy=not args.omit_bridge_vacuum_energy,
            include_cosmological_prefactor=args.include_cosmological_prefactor,
        )
    except ValueError as exc:
        parser.error(str(exc))

    print("Liouville genus-two pair-of-tori partition")
    print("  bridge approximation: primary state only")
    print(f"  b={result.b:.12g}")
    print(f"  c={result.central_charge:.12g}")
    print(f"  q1={format_complex(result.q1)}")
    print(f"  q2={format_complex(result.q2)}")
    print(f"  q_bridge={format_complex(result.q_bridge)}")
    print(f"  block order={result.block_order}")
    print(f"  bridge P cutoff={result.bridge_p_max:.12g}")
    print(f"  left handle P cutoff={result.handle_p_max_left:.12g}")
    print(f"  right handle P cutoff={result.handle_p_max_right:.12g}")
    print(f"  bridge quadrature order={result.bridge_quadrature_order}")
    print(f"  handle quadrature order={result.handle_quadrature_order}")
    print(f"  value={format_complex(result.value)}")


if __name__ == "__main__":
    run()
