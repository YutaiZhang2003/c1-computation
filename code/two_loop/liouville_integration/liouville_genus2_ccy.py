#!/usr/bin/env python3
"""Raw Liouville genus-two plumbing integral from the CCY block.

This is the theta-graph/two-holed-disc plumbing channel of Cho-Collier-Yin.  It
is distinct from the earlier pair-of-tori approximation: all three internal
Liouville momenta meet at each of two pair-of-pants vertices, and the Virasoro
block is the genus-two CCY block.
"""

from __future__ import annotations

from c1_paths import PACKAGE_ROOT, data_root, package_path, module_path

import argparse
import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import mpmath as mp

try:
    from ccy_plumbing_conventions import (
        THETA_GEOMETRY_EDGE_ORDER,
        ccy_raw_sewing_log_propagator,
        theta_geometry_to_ccy_order,
    )
    from ccy_genus2_block import ccy_genus2_block
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
        THETA_GEOMETRY_EDGE_ORDER,
        ccy_raw_sewing_log_propagator,
        theta_geometry_to_ccy_order,
    )
    from ccy_genus2_block import ccy_genus2_block
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


def liouville_genus2_ccy_density(
    *,
    special: UpsilonB,
    b: float,
    q1: complex,
    q2: complex,
    q3: complex,
    p1: float,
    p2: float,
    p3: float,
    block_order: int,
    mu: complex = 1.0,
    propagator_shift: float = 0.0,
    include_vacuum_seed: bool = True,
    vacuum_word_len: int = 3,
    vacuum_oscillator_level_max: int = 12,
    vacuum_word_tail_tolerance: float | None = None,
    vacuum_minimum_word_length: int = 5,
    resum_global_block: bool = True,
    global_block_tolerance: float = 1.0e-13,
    global_block_max_outer_order: int = 64,
    include_cosmological_prefactor: bool = False,
) -> complex:
    r"""Return the CCY Liouville integrand density.

    ``(q1,q2,q3)`` and ``(p1,p2,p3)`` use geometric
    ``(zero,one,infinity)`` order.

    The density is normalized so that integrating it against
    ``dP1 dP2 dP3`` gives

        int dP1 dP2 dP3 / pi^3 C(P1,P2,P3)^2
            | prod_i q_i^h_i F_CCY |^2.

    A nonzero ``propagator_shift`` applies an additional diagnostic factor
    ``prod_i q_i^(-propagator_shift)`` on top of the CCY sewing convention.
    """
    central_charge = liouville_central_charge(b)
    h1 = liouville_weight_from_momentum(b, p1)
    h2 = liouville_weight_from_momentum(b, p2)
    h3 = liouville_weight_from_momentum(b, p3)
    log_structure_constant = log_yin_structure_constant_momentum(
        special,
        p1,
        p2,
        p3,
        mu=mu,
        include_cosmological_prefactor=include_cosmological_prefactor,
    )
    ccy_q = theta_geometry_to_ccy_order((q1, q2, q3))
    ccy_h = theta_geometry_to_ccy_order((h1, h2, h3))
    block = ccy_genus2_block(
        c=central_charge,
        h1=ccy_h[0],
        h2=ccy_h[1],
        h3=ccy_h[2],
        q1=ccy_q[0],
        q2=ccy_q[1],
        q3=ccy_q[2],
        order=block_order,
        include_vacuum_seed=include_vacuum_seed,
        vacuum_word_len=vacuum_word_len,
        vacuum_oscillator_level_max=vacuum_oscillator_level_max,
        vacuum_word_tail_tolerance=vacuum_word_tail_tolerance,
        vacuum_minimum_word_length=vacuum_minimum_word_length,
        resum_global_block=resum_global_block,
        global_block_tolerance=global_block_tolerance,
        global_block_max_outer_order=global_block_max_outer_order,
    ).value
    log_propagator = ccy_raw_sewing_log_propagator(
        (q1, q2, q3),
        (h1, h2, h3),
        diagnostic_shift=propagator_shift,
    )
    if block == 0:
        return 0.0 + 0.0j
    log_density = (
        2 * log_structure_constant
        + 2 * mp.re(log_propagator)
        + 2 * mp.log(abs(block))
        - 3 * mp.log(mp.pi)
    )
    return complex(mp.exp(log_density))


@dataclass(frozen=True)
class LiouvilleGenus2CCYSample:
    p1: float
    p2: float
    p3: float
    measure_weight: float
    structure_constant: complex
    log_structure_constant: complex
    block: complex
    propagator: complex
    contribution: complex


@dataclass(frozen=True)
class LiouvilleGenus2CCYResult:
    value: complex
    b: float
    central_charge: float
    q1: complex
    q2: complex
    q3: complex
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
    global_block_max_outer_order: int
    include_cosmological_prefactor: bool
    log_q_values: tuple[complex, complex, complex] | None
    samples: tuple[LiouvilleGenus2CCYSample, ...]

    @property
    def channel(self) -> str:
        return "theta"

    @property
    def edge_names(self) -> tuple[str, str, str]:
        return THETA_GEOMETRY_EDGE_ORDER

    @property
    def q_values(self) -> tuple[complex, complex, complex]:
        return self.q1, self.q2, self.q3


def liouville_genus2_ccy_partition(
    *,
    b: float,
    q1: complex,
    q2: complex,
    q3: complex,
    log_q_values: Sequence[complex] | None = None,
    block_order: int,
    p_max: float | None = None,
    quadrature_order: int | Sequence[int] = 4,
    dps: int = 40,
    mu: complex = 1.0,
    propagator_shift: float | None = None,
    include_vacuum_seed: bool = True,
    vacuum_word_len: int = 3,
    vacuum_oscillator_level_max: int = 12,
    vacuum_word_tail_tolerance: float | None = None,
    vacuum_minimum_word_length: int = 5,
    resum_global_block: bool = True,
    global_block_tolerance: float = 1.0e-13,
    global_block_max_outer_order: int = 64,
    include_cosmological_prefactor: bool = False,
    tail_tolerance: float = 1.0e-12,
    safety_margin: float = 1.0,
    quadrature_scheme: str = "uniform",
    quadrature_tail_order: int | None = None,
    quadrature_split_widths: float = 4.0,
    store_samples: bool = True,
) -> LiouvilleGenus2CCYResult:
    r"""Evaluate the truncated raw Liouville genus-two integral in the CCY frame.

    The arguments ``(q1,q2,q3)`` are the geometric theta tubes
    ``(q_zero,q_one,q_infinity)``.  They are reversed internally when passed
    to the CCY descendant tensor, whose slots are ordered
    ``(infinity,one,zero)``.

    The implemented integral is

        int dP1 dP2 dP3 / pi^3 C(P1,P2,P3)^2
            | prod_i q_i^h_i F_CCY(h_i;q_i) |^2.

    This raw plumbing-frame object includes explicit local propagators.  It is
    not, by itself, the physical modular-invariant genus-two partition function.
    The natural shift in this raw CCY convention is zero; nonzero shifts are
    diagnostic only unless a conformal-frame normalization has been derived.

    ``log_q_values`` preserves the chosen sewing logarithms when an
    exponentiated edge is below floating-point range. The representable ``q``
    arguments are still used for descendant powers, whose omitted corrections
    are then below binary64 precision.
    """
    if b <= 0:
        raise ValueError("b must be positive")
    if block_order < 0:
        raise ValueError("block_order must be non-negative")
    q1 = _validate_plumbing_q("q1", q1)
    q2 = _validate_plumbing_q("q2", q2)
    q3 = _validate_plumbing_q("q3", q3)
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
    quadrature_orders = normalize_quadrature_orders((q1, q2, q3), quadrature_order)
    central_charge = liouville_central_charge(b)
    if propagator_shift is None:
        propagator_shift = 0.0

    if p_max is None:
        p_max = max(
            estimate_p_max(q1, tail_tolerance=tail_tolerance, safety_margin=safety_margin),
            estimate_p_max(q2, tail_tolerance=tail_tolerance, safety_margin=safety_margin),
            estimate_p_max(q3, tail_tolerance=tail_tolerance, safety_margin=safety_margin),
        )
    if p_max <= 0:
        raise ValueError("p_max must be positive")

    special = UpsilonB(b=b, dps=dps)
    rules = momentum_quadrature_rules(
        (q1, q2, q3),
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
    samples: list[LiouvilleGenus2CCYSample] = []
    ccy_q = theta_geometry_to_ccy_order((q1, q2, q3))
    for idx1, p1 in enumerate(rules[0].nodes):
        h1 = liouville_weight_from_momentum(b, p1)
        for idx2, p2 in enumerate(rules[1].nodes):
            h2 = liouville_weight_from_momentum(b, p2)
            for idx3, p3 in enumerate(rules[2].nodes):
                h3 = liouville_weight_from_momentum(b, p3)
                measure_weight = rules[0].weights[idx1] * rules[1].weights[idx2] * rules[2].weights[idx3]
                log_structure_constant = log_yin_structure_constant_momentum(
                    special,
                    p1,
                    p2,
                    p3,
                    mu=mu,
                    include_cosmological_prefactor=include_cosmological_prefactor,
                )
                ccy_h = theta_geometry_to_ccy_order((h1, h2, h3))
                block = ccy_genus2_block(
                    c=central_charge,
                    h1=ccy_h[0],
                    h2=ccy_h[1],
                    h3=ccy_h[2],
                    q1=ccy_q[0],
                    q2=ccy_q[1],
                    q3=ccy_q[2],
                    order=block_order,
                    include_vacuum_seed=include_vacuum_seed,
                    vacuum_word_len=vacuum_word_len,
                    vacuum_oscillator_level_max=vacuum_oscillator_level_max,
                    vacuum_word_tail_tolerance=vacuum_word_tail_tolerance,
                    vacuum_minimum_word_length=vacuum_minimum_word_length,
                    resum_global_block=resum_global_block,
                    global_block_tolerance=global_block_tolerance,
                    global_block_max_outer_order=global_block_max_outer_order,
                ).value
                log_propagator = ccy_raw_sewing_log_propagator(
                    (q1, q2, q3),
                    (h1, h2, h3),
                    diagnostic_shift=propagator_shift,
                    log_q_values=log_q_tuple,
                )
                if block == 0 or measure_weight == 0:
                    contribution = 0.0 + 0.0j
                else:
                    log_contribution = (
                        mp.log(measure_weight)
                        + 2 * log_structure_constant
                        + 2 * mp.re(log_propagator)
                        + 2 * mp.log(abs(block))
                    )
                    contribution = complex(mp.exp(log_contribution))
                total += contribution
                if store_samples:
                    propagator = complex(mp.exp(mp.mpc(log_propagator)))
                    structure_constant = complex(
                        mp.exp(log_structure_constant)
                    )
                    samples.append(
                        LiouvilleGenus2CCYSample(
                            p1=p1,
                            p2=p2,
                            p3=p3,
                            measure_weight=measure_weight,
                            structure_constant=structure_constant,
                            log_structure_constant=complex(
                                log_structure_constant
                            ),
                            block=block,
                            propagator=propagator,
                            contribution=contribution,
                        )
                    )

    return LiouvilleGenus2CCYResult(
        value=total,
        b=float(b),
        central_charge=float(central_charge),
        q1=q1,
        q2=q2,
        q3=q3,
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
        global_block_max_outer_order=(
            int(global_block_max_outer_order) if resum_global_block else 0
        ),
        include_cosmological_prefactor=bool(include_cosmological_prefactor),
        log_q_values=log_q_tuple,  # type: ignore[arg-type]
        samples=tuple(samples),
    )


def run(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate the raw Liouville genus-two CCY plumbing integral.")
    parser.add_argument("--b", type=float, required=True)
    parser.add_argument("--q1", type=parse_complex, required=True, help="geometric q_zero")
    parser.add_argument("--q2", type=parse_complex, required=True, help="geometric q_one")
    parser.add_argument("--q3", type=parse_complex, required=True, help="geometric q_infinity")
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
        help="diagnostic only; defaults to 0 in the raw CCY theta-graph frame",
    )
    parser.add_argument("--no-vacuum-seed", action="store_true")
    parser.add_argument("--vacuum-word-len", type=int, default=3)
    parser.add_argument("--vacuum-oscillator-level-max", type=int, default=12)
    parser.add_argument(
        "--truncate-global-block",
        action="store_true",
        help="restore the legacy total-degree cutoff on the global SL(2) seed",
    )
    parser.add_argument("--global-block-tolerance", type=float, default=1.0e-13)
    parser.add_argument("--global-block-max-outer-order", type=int, default=64)
    parser.add_argument("--include-cosmological-prefactor", action="store_true")
    parser.add_argument("--no-store-samples", action="store_true")
    args = parser.parse_args(argv)

    result = liouville_genus2_ccy_partition(
        b=args.b,
        q1=args.q1,
        q2=args.q2,
        q3=args.q3,
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
        global_block_max_outer_order=args.global_block_max_outer_order,
        include_cosmological_prefactor=args.include_cosmological_prefactor,
        store_samples=not args.no_store_samples,
    )

    print("Raw Liouville genus-two CCY plumbing integral")
    print("  frame=theta-graph CCY")
    print(f"  b={result.b:.12g}")
    print(f"  c={result.central_charge:.12g}")
    print(f"  q1={format_complex(result.q1)}")
    print(f"  q2={format_complex(result.q2)}")
    print(f"  q3={format_complex(result.q3)}")
    print(f"  block order={result.block_order}")
    print(f"  P cutoff={result.p_max:.12g}")
    print(f"  quadrature order={result.quadrature_order}")
    print(f"  propagator shift={result.propagator_shift:.12g}")
    print(f"  vacuum seed={result.include_vacuum_seed}")
    print(f"  global block resummed={result.global_block_resummed}")
    print(f"  value={format_complex(result.value)}")


if __name__ == "__main__":
    run()
