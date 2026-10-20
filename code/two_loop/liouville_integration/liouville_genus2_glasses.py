#!/usr/bin/env python3
"""Raw Liouville genus-two plumbing integral in the glasses frame."""

from __future__ import annotations

from c1_paths import PACKAGE_ROOT, data_root, package_path, module_path

import argparse
import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import mpmath as mp

try:
    from ccy_plumbing_conventions import (
        GLASSES_GEOMETRY_EDGE_ORDER,
        ccy_raw_sewing_log_propagator,
    )
    from ccy_genus2_glasses_block import ccy_genus2_glasses_block
    from liouville_momentum_quadrature import (
        momentum_quadrature_rules,
        normalize_quadrature_orders,
    )
    from liouville_torus import (
        UpsilonB,
        estimate_p_max,
        log_yin_structure_constant_momentum,
    )
except ImportError:  # pragma: no cover - supports package-style execution
    from ccy_plumbing_conventions import (
        GLASSES_GEOMETRY_EDGE_ORDER,
        ccy_raw_sewing_log_propagator,
    )
    from ccy_genus2_glasses_block import ccy_genus2_glasses_block
    from liouville_momentum_quadrature import (
        momentum_quadrature_rules,
        normalize_quadrature_orders,
    )
    from liouville_torus import (
        UpsilonB,
        estimate_p_max,
        log_yin_structure_constant_momentum,
    )


def parse_complex(value: str) -> complex:
    return complex(value.replace("i", "j"))


def format_complex(value: complex) -> str:
    return f"{value.real:+.12e}{value.imag:+.12e}j"


def _validate_plumbing_q(name: str, value: complex) -> complex:
    value = complex(value)
    if not 0 < abs(value) < 1:
        raise ValueError(f"{name} must satisfy 0 < |{name}| < 1")
    return value


def liouville_central_charge(b: float) -> float:
    q_background = b + 1.0 / b
    return 1.0 + 6.0 * q_background * q_background


def liouville_weight_from_momentum(b: float, momentum: float) -> float:
    q_background = b + 1.0 / b
    return 0.25 * q_background * q_background + momentum * momentum


def build_glasses_log_structure_tables(
    *,
    special: UpsilonB,
    left_nodes: Sequence[float],
    right_nodes: Sequence[float],
    bridge_nodes: Sequence[float],
    mu: complex = 1.0,
    include_cosmological_prefactor: bool = False,
) -> tuple[
    tuple[tuple[mp.mpc, ...], ...],
    tuple[tuple[mp.mpc, ...], ...],
]:
    r"""Build the two exact glasses-vertex tables in their named edge order.

    The entries are

    ``left[i][k]  = log C(P_left[i],  P_bridge[k], P_left[i])`` and
    ``right[j][k] = log C(P_right[j], P_bridge[k], P_right[j])``.

    Keeping this operation separate makes the channel incidence and lookup
    order directly auditable.  It is an exact cache of the logarithmic DOZZ
    evaluator: no interpolation or momentum approximation is introduced.
    """

    left_table = tuple(
        tuple(
            log_yin_structure_constant_momentum(
                special,
                p_left,
                p_bridge,
                p_left,
                mu=mu,
                include_cosmological_prefactor=include_cosmological_prefactor,
            )
            for p_bridge in bridge_nodes
        )
        for p_left in left_nodes
    )
    right_table = tuple(
        tuple(
            log_yin_structure_constant_momentum(
                special,
                p_right,
                p_bridge,
                p_right,
                mu=mu,
                include_cosmological_prefactor=include_cosmological_prefactor,
            )
            for p_bridge in bridge_nodes
        )
        for p_right in right_nodes
    )
    return left_table, right_table


def liouville_genus2_glasses_density(
    *,
    special: UpsilonB,
    b: float,
    q_left: complex,
    q_right: complex,
    q_bridge: complex,
    p_left: float,
    p_right: float,
    p_bridge: float,
    block_order: int,
    mu: complex = 1.0,
    propagator_shift: float = 0.0,
    include_vacuum_seed: bool = True,
    vacuum_word_len: int = 6,
    vacuum_oscillator_level_max: int = 30,
    vacuum_word_tail_tolerance: float | None = None,
    vacuum_minimum_word_length: int = 5,
    resum_global_block: bool = True,
    global_block_tolerance: float = 1.0e-13,
    include_cosmological_prefactor: bool = False,
) -> complex:
    r"""Return the glasses-channel Liouville integrand density.

    The two pair-of-pants vertices carry structure constants

        C(P_left, P_bridge, P_left) C(P_right, P_bridge, P_right).

    The density is normalized so that integrating against
    ``dP_left dP_right dP_bridge / pi^3`` gives the diagonal contribution.
    Following CCY, the block contains descendant powers and the raw local
    propagator is the separated primary factor ``prod_e q_e^h_e``.  A nonzero
    ``propagator_shift`` is only an extra diagnostic multiplier.
    """
    central_charge = liouville_central_charge(b)
    h_left = liouville_weight_from_momentum(b, p_left)
    h_right = liouville_weight_from_momentum(b, p_right)
    h_bridge = liouville_weight_from_momentum(b, p_bridge)
    log_structure_left = log_yin_structure_constant_momentum(
        special,
        p_left,
        p_bridge,
        p_left,
        mu=mu,
        include_cosmological_prefactor=include_cosmological_prefactor,
    )
    log_structure_right = log_yin_structure_constant_momentum(
        special,
        p_right,
        p_bridge,
        p_right,
        mu=mu,
        include_cosmological_prefactor=include_cosmological_prefactor,
    )
    block = ccy_genus2_glasses_block(
        c=central_charge,
        h_left=h_left,
        h_right=h_right,
        h_bridge=h_bridge,
        q_left=q_left,
        q_right=q_right,
        q_bridge=q_bridge,
        order=block_order,
        include_vacuum_seed=include_vacuum_seed,
        vacuum_word_len=vacuum_word_len,
        vacuum_oscillator_level_max=vacuum_oscillator_level_max,
        vacuum_word_tail_tolerance=vacuum_word_tail_tolerance,
        vacuum_minimum_word_length=vacuum_minimum_word_length,
        resum_global_block=resum_global_block,
        global_block_tolerance=global_block_tolerance,
    ).value
    log_propagator = ccy_raw_sewing_log_propagator(
        (q_left, q_right, q_bridge),
        (h_left, h_right, h_bridge),
        diagnostic_shift=propagator_shift,
    )
    if block == 0:
        return 0.0 + 0.0j
    log_density = (
        log_structure_left
        + log_structure_right
        + 2 * mp.re(log_propagator)
        + 2 * mp.log(abs(block))
        - 3 * mp.log(mp.pi)
    )
    return complex(mp.exp(log_density))


@dataclass(frozen=True)
class LiouvilleGenus2GlassesSample:
    p_left: float
    p_right: float
    p_bridge: float
    measure_weight: float
    structure_left: complex
    structure_right: complex
    log_structure_left: complex
    log_structure_right: complex
    block: complex
    propagator: complex
    contribution: complex


@dataclass(frozen=True)
class LiouvilleGenus2GlassesResult:
    value: complex
    b: float
    central_charge: float
    q_left: complex
    q_right: complex
    q_bridge: complex
    block_order: int
    p_max: float
    quadrature_order: int
    quadrature_orders: tuple[int, int, int]
    dps: int
    propagator_shift: float
    include_vacuum_seed: bool
    vacuum_word_len: int
    vacuum_oscillator_level_max: int
    global_block_resummed: bool
    global_block_tolerance: float
    include_cosmological_prefactor: bool
    log_q_values: tuple[complex, complex, complex] | None
    samples: tuple[LiouvilleGenus2GlassesSample, ...]

    @property
    def channel(self) -> str:
        return "glasses"

    @property
    def edge_names(self) -> tuple[str, str, str]:
        return GLASSES_GEOMETRY_EDGE_ORDER

    @property
    def q_values(self) -> tuple[complex, complex, complex]:
        return self.q_left, self.q_right, self.q_bridge


def liouville_genus2_glasses_partition(
    *,
    b: float,
    q_left: complex,
    q_right: complex,
    q_bridge: complex,
    log_q_values: Sequence[complex] | None = None,
    block_order: int,
    p_max: float | None = None,
    quadrature_order: int | Sequence[int] = 4,
    dps: int = 40,
    mu: complex = 1.0,
    propagator_shift: float | None = None,
    include_vacuum_seed: bool = True,
    vacuum_word_len: int = 6,
    vacuum_oscillator_level_max: int = 30,
    vacuum_word_tail_tolerance: float | None = None,
    vacuum_minimum_word_length: int = 5,
    resum_global_block: bool = True,
    global_block_tolerance: float = 1.0e-13,
    include_cosmological_prefactor: bool = False,
    tail_tolerance: float = 1.0e-12,
    safety_margin: float = 1.0,
    quadrature_scheme: str = "uniform",
    quadrature_tail_order: int | None = None,
    quadrature_split_widths: float = 4.0,
    store_samples: bool = True,
) -> LiouvilleGenus2GlassesResult:
    r"""Evaluate the truncated raw Liouville genus-two integral in the glasses frame.

    This raw plumbing-frame object includes explicit local propagators.  It is
    not, by itself, the physical modular-invariant genus-two partition function.
    The CCY local propagator is ``prod_e q_e^h_e``.  The default propagator
    shift is zero; nonzero shifts are diagnostic only unless a conformal-frame
    normalization has been derived.

    ``log_q_values`` preserves the chosen sewing logarithms when an
    exponentiated handle edge is below floating-point range. The representable
    ``q`` arguments remain the descendant-expansion coordinates.
    """
    if b <= 0:
        raise ValueError("b must be positive")
    if block_order < 0:
        raise ValueError("block_order must be non-negative")
    q_left = _validate_plumbing_q("q_left", q_left)
    q_right = _validate_plumbing_q("q_right", q_right)
    q_bridge = _validate_plumbing_q("q_bridge", q_bridge)
    log_q_tuple = (
        tuple(complex(value) for value in log_q_values)
        if log_q_values is not None
        else None
    )
    if log_q_tuple is not None and (
        len(log_q_tuple) != 3
        or any(not math.isfinite(value.real) or not math.isfinite(value.imag) or value.real >= 0.0 for value in log_q_tuple)
    ):
        raise ValueError("log_q_values must contain three finite logarithms with negative real part")
    quadrature_orders = normalize_quadrature_orders(
        (q_left, q_right, q_bridge), quadrature_order
    )
    central_charge = liouville_central_charge(b)
    if propagator_shift is None:
        propagator_shift = 0.0

    if p_max is None:
        p_max = max(
            estimate_p_max(q_left, tail_tolerance=tail_tolerance, safety_margin=safety_margin),
            estimate_p_max(q_right, tail_tolerance=tail_tolerance, safety_margin=safety_margin),
            estimate_p_max(q_bridge, tail_tolerance=tail_tolerance, safety_margin=safety_margin),
        )
    if p_max <= 0:
        raise ValueError("p_max must be positive")

    special = UpsilonB(b=b, dps=dps)
    rules = momentum_quadrature_rules(
        (q_left, q_right, q_bridge),
        p_max=float(p_max),
        quadrature_order=quadrature_orders,
        quadrature_scheme=quadrature_scheme,
        tail_order=quadrature_tail_order,
        split_widths=quadrature_split_widths,
        log_q_abs_values=(
            tuple(value.real for value in log_q_tuple)
            if log_q_tuple is not None
            else None
        ),
    )

    total = 0.0 + 0.0j
    samples: list[LiouvilleGenus2GlassesSample] = []
    # Each glasses vertex depends on one handle momentum and the bridge
    # momentum, but not on the momentum of the opposite handle.  Computing it
    # inside the three-dimensional loop repeated the identical DOZZ value
    # once for every node on that opposite handle.  Precompute the two exact
    # vertex tables; no interpolation or approximation is involved.
    left_weights = tuple(
        liouville_weight_from_momentum(b, momentum)
        for momentum in rules[0].nodes
    )
    right_weights = tuple(
        liouville_weight_from_momentum(b, momentum)
        for momentum in rules[1].nodes
    )
    bridge_weights = tuple(
        liouville_weight_from_momentum(b, momentum)
        for momentum in rules[2].nodes
    )
    log_structure_left_table, log_structure_right_table = (
        build_glasses_log_structure_tables(
            special=special,
            left_nodes=rules[0].nodes,
            right_nodes=rules[1].nodes,
            bridge_nodes=rules[2].nodes,
            mu=mu,
            include_cosmological_prefactor=include_cosmological_prefactor,
        )
    )
    for idx_left, p_left in enumerate(rules[0].nodes):
        h_left = left_weights[idx_left]
        for idx_right, p_right in enumerate(rules[1].nodes):
            h_right = right_weights[idx_right]
            for idx_bridge, p_bridge in enumerate(rules[2].nodes):
                h_bridge = bridge_weights[idx_bridge]
                measure_weight = rules[0].weights[idx_left] * rules[1].weights[idx_right] * rules[2].weights[idx_bridge]
                log_structure_left = log_structure_left_table[idx_left][idx_bridge]
                log_structure_right = log_structure_right_table[idx_right][idx_bridge]
                block = ccy_genus2_glasses_block(
                    c=central_charge,
                    h_left=h_left,
                    h_right=h_right,
                    h_bridge=h_bridge,
                    q_left=q_left,
                    q_right=q_right,
                    q_bridge=q_bridge,
                    order=block_order,
                    include_vacuum_seed=include_vacuum_seed,
                    vacuum_word_len=vacuum_word_len,
                    vacuum_oscillator_level_max=vacuum_oscillator_level_max,
                    vacuum_word_tail_tolerance=vacuum_word_tail_tolerance,
                    vacuum_minimum_word_length=vacuum_minimum_word_length,
                    resum_global_block=resum_global_block,
                    global_block_tolerance=global_block_tolerance,
                ).value
                log_propagator = ccy_raw_sewing_log_propagator(
                    (q_left, q_right, q_bridge),
                    (h_left, h_right, h_bridge),
                    diagnostic_shift=propagator_shift,
                    log_q_values=log_q_tuple,
                )
                if block == 0 or measure_weight == 0:
                    contribution = 0.0 + 0.0j
                else:
                    log_contribution = (
                        mp.log(measure_weight)
                        + log_structure_left
                        + log_structure_right
                        + 2 * mp.re(log_propagator)
                        + 2 * mp.log(abs(block))
                    )
                    contribution = complex(mp.exp(log_contribution))
                total += contribution
                if store_samples:
                    structure_left = complex(mp.exp(log_structure_left))
                    structure_right = complex(mp.exp(log_structure_right))
                    propagator = complex(mp.exp(mp.mpc(log_propagator)))
                    samples.append(
                        LiouvilleGenus2GlassesSample(
                            p_left=p_left,
                            p_right=p_right,
                            p_bridge=p_bridge,
                            measure_weight=measure_weight,
                            structure_left=structure_left,
                            structure_right=structure_right,
                            log_structure_left=complex(log_structure_left),
                            log_structure_right=complex(log_structure_right),
                            block=block,
                            propagator=propagator,
                            contribution=contribution,
                        )
                    )

    return LiouvilleGenus2GlassesResult(
        value=total,
        b=float(b),
        central_charge=float(central_charge),
        q_left=q_left,
        q_right=q_right,
        q_bridge=q_bridge,
        block_order=int(block_order),
        p_max=float(p_max),
        quadrature_order=max(quadrature_orders),
        quadrature_orders=quadrature_orders,
        dps=int(dps),
        propagator_shift=float(propagator_shift),
        include_vacuum_seed=bool(include_vacuum_seed),
        vacuum_word_len=int(vacuum_word_len),
        vacuum_oscillator_level_max=int(vacuum_oscillator_level_max),
        global_block_resummed=bool(resum_global_block),
        global_block_tolerance=(float(global_block_tolerance) if resum_global_block else 0.0),
        include_cosmological_prefactor=bool(include_cosmological_prefactor),
        log_q_values=log_q_tuple,  # type: ignore[arg-type]
        samples=tuple(samples),
    )


def run(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate the raw Liouville genus-two glasses plumbing integral.")
    parser.add_argument("--b", type=float, required=True)
    parser.add_argument("--q-left", type=parse_complex, required=True)
    parser.add_argument("--q-right", type=parse_complex, required=True)
    parser.add_argument("--q-bridge", type=parse_complex, required=True)
    parser.add_argument("--block-order", type=int, default=1)
    parser.add_argument("--p-max", type=float)
    parser.add_argument("--quadrature-order", type=int, default=4)
    parser.add_argument(
        "--quadrature-scheme",
        choices=(
            "uniform",
            "edge-scaled",
            "primary-gaussian",
            "threshold-gaussian",
            "cft-asymptotic",
        ),
        default="uniform",
    )
    parser.add_argument("--quadrature-tail-order", type=int)
    parser.add_argument("--quadrature-split-widths", type=float, default=4.0)
    parser.add_argument("--dps", type=int, default=40)
    parser.add_argument("--mu", type=parse_complex, default=1.0 + 0.0j)
    parser.add_argument(
        "--propagator-shift",
        type=float,
        help="diagnostic only; defaults to 0 in the raw glasses plumbing frame",
    )
    parser.add_argument("--no-vacuum-seed", action="store_true")
    parser.add_argument("--vacuum-word-len", type=int, default=6)
    parser.add_argument("--vacuum-oscillator-level-max", type=int, default=30)
    parser.add_argument(
        "--truncate-global-block",
        action="store_true",
        help="restore the legacy total-degree cutoff on the global SL(2) seed",
    )
    parser.add_argument("--global-block-tolerance", type=float, default=1.0e-13)
    parser.add_argument("--include-cosmological-prefactor", action="store_true")
    parser.add_argument("--no-store-samples", action="store_true")
    args = parser.parse_args(argv)

    result = liouville_genus2_glasses_partition(
        b=args.b,
        q_left=args.q_left,
        q_right=args.q_right,
        q_bridge=args.q_bridge,
        block_order=args.block_order,
        p_max=args.p_max,
        quadrature_order=args.quadrature_order,
        quadrature_scheme=args.quadrature_scheme,
        quadrature_tail_order=args.quadrature_tail_order,
        quadrature_split_widths=args.quadrature_split_widths,
        dps=args.dps,
        mu=args.mu,
        propagator_shift=args.propagator_shift,
        include_vacuum_seed=not args.no_vacuum_seed,
        vacuum_word_len=args.vacuum_word_len,
        vacuum_oscillator_level_max=args.vacuum_oscillator_level_max,
        resum_global_block=not args.truncate_global_block,
        global_block_tolerance=args.global_block_tolerance,
        include_cosmological_prefactor=args.include_cosmological_prefactor,
        store_samples=not args.no_store_samples,
    )

    print("Raw Liouville genus-two glasses plumbing integral")
    print("  frame=separating/glasses CCY")
    print(f"  b={result.b:.12g}")
    print(f"  c={result.central_charge:.12g}")
    print(f"  q_left={format_complex(result.q_left)}")
    print(f"  q_right={format_complex(result.q_right)}")
    print(f"  q_bridge={format_complex(result.q_bridge)}")
    print(f"  block order={result.block_order}")
    print(f"  P cutoff={result.p_max:.12g}")
    print(f"  quadrature order={result.quadrature_order}")
    print(f"  propagator shift={result.propagator_shift:.12g}")
    print(f"  vacuum seed={result.include_vacuum_seed}")
    print(f"  global block resummed={result.global_block_resummed}")
    print(f"  value={format_complex(result.value)}")


if __name__ == "__main__":
    run()
