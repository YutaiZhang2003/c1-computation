#!/usr/bin/env python3
"""CCY c-recursion for the genus-two separating/glasses pants graph.

The theta graph in :mod:`ccy_genus2_block` is only one genus-two plumbing
frame.  This module applies the generic Cho-Collier-Yin plumbing-frame rule to
the separating graph: two one-holed tori joined by a bridge tube.

The chiral block is

    F = sum qL^|A| qR^|C| qB^|E| G_L^{AB} G_R^{CD} G_B^{EF}
        rho(A_L, E_B, B_L) rho(C_R, F_B, D_R).

As in CCY's generic plumbing construction, these are descendant powers only;
the separated primary propagator qL^hL qR^hR qB^hB is applied by the Liouville
sewing wrapper.

The two handle edges are tadpoles, so their null-vector residue is the torus
one-point factor P[hB, h+rs] P[hB, h].  The bridge edge is ordinary and gives
P[hL, hL] P[hR, hR].  The global block factorizes into three Gauss
hypergeometric functions and can therefore be evaluated to all descendant
levels before truncating the Virasoro pole recursion.
"""

from __future__ import annotations

from c1_paths import PACKAGE_ROOT, data_root, package_path, module_path

import argparse
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Sequence

try:
    from ccy_genus2_block import (
        PartialFractionInC,
        ConfluentPoleError,
        _as_complex,
        _is_finite_complex,
        _validate_order,
        b_from_c_rs_h,
        c_rs_from_h,
        collision_regulated_partial_fraction_value,
        format_complex,
        fusion_polynomial_for_weights,
        gauss_hypergeometric_2f1,
        minus_dc_dh_times_a_rs,
        parse_complex,
        rho_lminus1_triple,
        sl2_descendant_norm,
    )
    from genus2_vacuum_blocks import glasses_vacuum_block
except ImportError:  # pragma: no cover - supports package-style execution
    from ccy_genus2_block import (
        PartialFractionInC,
        ConfluentPoleError,
        _as_complex,
        _is_finite_complex,
        _validate_order,
        b_from_c_rs_h,
        c_rs_from_h,
        collision_regulated_partial_fraction_value,
        format_complex,
        fusion_polynomial_for_weights,
        gauss_hypergeometric_2f1,
        minus_dc_dh_times_a_rs,
        parse_complex,
        rho_lminus1_triple,
        sl2_descendant_norm,
    )
    from genus2_vacuum_blocks import glasses_vacuum_block


PairFunction = Callable[[complex, int], tuple[complex, complex]]


@dataclass(frozen=True)
class CCYGenus2GlassesBlockResult:
    value: complex
    c: complex
    h_left: complex
    h_right: complex
    h_bridge: complex
    q_left: complex
    q_right: complex
    q_bridge: complex
    order: int
    include_vacuum_seed: bool
    vacuum_word_len: int
    vacuum_oscillator_level_max: int
    global_block_resummed: bool
    global_block_tolerance: float
    partial_fraction_pole_count: int
    partial_fraction_coefficient_count: int
    partial_fraction_max_pole_order: int
    collision_regulated: bool = False
    collision_regulator_error: float = 0.0
    collision_regulator_scale: float = 0.0


def genus2_global_glasses_sl2_block(
    h_left: complex,
    h_right: complex,
    h_bridge: complex,
    q_left: complex,
    q_right: complex,
    q_bridge: complex,
    order: int,
) -> complex:
    """Return the total-degree truncated global block for the glasses graph."""
    order = _validate_order(order)
    h_left = _as_complex(h_left)
    h_right = _as_complex(h_right)
    h_bridge = _as_complex(h_bridge)
    q_left = _as_complex(q_left)
    q_right = _as_complex(q_right)
    q_bridge = _as_complex(q_bridge)
    total = 0.0 + 0.0j
    for left_level in range(order + 1):
        norm_left = sl2_descendant_norm(h_left, left_level)
        for right_level in range(order + 1 - left_level):
            norm_right = sl2_descendant_norm(h_right, right_level)
            for bridge_level in range(order + 1 - left_level - right_level):
                left_rho = rho_lminus1_triple(
                    left_level,
                    bridge_level,
                    left_level,
                    h_left,
                    h_bridge,
                    h_left,
                )
                right_rho = rho_lminus1_triple(
                    right_level,
                    bridge_level,
                    right_level,
                    h_right,
                    h_bridge,
                    h_right,
                )
                total += (
                    (q_left**left_level)
                    * (q_right**right_level)
                    * (q_bridge**bridge_level)
                    * left_rho
                    * right_rho
                    / (norm_left * norm_right * sl2_descendant_norm(h_bridge, bridge_level))
                )
    return total


def torus_one_point_global_sl2_block_resummed(
    internal_weight: complex,
    external_weight: complex,
    q: complex,
    *,
    tolerance: float = 1.0e-15,
) -> complex:
    r"""Return the all-level torus one-point global block.

    This is the Pfaff-transformed version of CCY equation (4.28), chosen so
    that the Gauss-series argument remains the plumbing parameter ``q``:

    ``(1-q)^(d-1) 2F1(d, 2h+d-1; 2h; q)``.
    """

    internal_weight = _as_complex(internal_weight)
    external_weight = _as_complex(external_weight)
    q = _as_complex(q)
    return (1.0 - q) ** (external_weight - 1.0) * gauss_hypergeometric_2f1(
        external_weight,
        2.0 * internal_weight + external_weight - 1.0,
        2.0 * internal_weight,
        q,
        tolerance=tolerance,
    )


def genus2_global_glasses_sl2_block_resummed(
    h_left: complex,
    h_right: complex,
    h_bridge: complex,
    q_left: complex,
    q_right: complex,
    q_bridge: complex,
    *,
    tolerance: float = 1.0e-15,
) -> complex:
    r"""Return the exact all-level glasses global block as three ``2F1``s.

    The two handle sums are torus one-point global blocks with external
    weight ``h_bridge``.  The remaining bridge descendants give
    ``2F1(h_bridge,h_bridge;2 h_bridge;q_bridge)`` independently of the
    handle levels, so the complete three-variable sum factorizes.
    """

    h_left = _as_complex(h_left)
    h_right = _as_complex(h_right)
    h_bridge = _as_complex(h_bridge)
    q_left = _as_complex(q_left)
    q_right = _as_complex(q_right)
    q_bridge = _as_complex(q_bridge)
    return (
        torus_one_point_global_sl2_block_resummed(
            h_left,
            h_bridge,
            q_left,
            tolerance=tolerance,
        )
        * torus_one_point_global_sl2_block_resummed(
            h_right,
            h_bridge,
            q_right,
            tolerance=tolerance,
        )
        * gauss_hypergeometric_2f1(
            h_bridge,
            h_bridge,
            2.0 * h_bridge,
            q_bridge,
            tolerance=tolerance,
        )
    )


@lru_cache(maxsize=128)
def glasses_vacuum_seed_schottky(
    q_left: complex,
    q_right: complex,
    q_bridge: complex,
    *,
    max_word_len: int = 6,
    oscillator_level_max: int = 30,
    word_tail_tolerance: float | None = None,
    minimum_word_length: int = 5,
) -> complex:
    """Return the c=infinity vacuum seed in the separating Schottky frame."""
    result = glasses_vacuum_block(
        q_left,
        q_right,
        q_bridge,
        max_word_length=max_word_len,
        max_mode=oscillator_level_max,
        word_tail_tolerance=word_tail_tolerance,
        minimum_word_length=minimum_word_length,
    )
    if word_tail_tolerance is not None and (
        result.primitive_word_tail_estimate is None
        or result.primitive_word_tail_estimate > float(word_tail_tolerance)
    ):
        raise RuntimeError(
            "glasses CCY vacuum seed exhausted its word-length safety cap "
            "before reaching the requested primitive-word tail"
        )
    return result.value


def _residue_prefactor_from_pair_functions(
    r: int,
    s: int,
    h_edge: complex,
    first_pair: PairFunction,
    second_pair: PairFunction,
) -> complex:
    """Return ``-dc/dh A_rs P(first_pair) P(second_pair)``.

    The pair functions receive the current degenerate weight and the null
    level.  This matters for tadpole edges, where one fusion polynomial uses
    the shifted weight ``h_edge + rs``.
    """

    level = r * s
    h_edge = _as_complex(h_edge)

    def direct(current_h: complex) -> complex:
        b_pole = b_from_c_rs_h(r, s, current_h)
        first_top, first_bottom = first_pair(current_h, level)
        second_top, second_bottom = second_pair(current_h, level)
        return (
            minus_dc_dh_times_a_rs(r, s, current_h)
            * fusion_polynomial_for_weights(r, s, b_pole, first_top, first_bottom)
            * fusion_polynomial_for_weights(r, s, b_pole, second_top, second_bottom)
        )

    try:
        value = direct(h_edge)
        if _is_finite_complex(value):
            return value
    except ZeroDivisionError:
        pass

    scale = max(1.0, abs(h_edge))
    samples: list[tuple[float, complex]] = []
    for relative_step in (1.0e-5, 3.0e-6, 1.0e-6, 3.0e-7, 1.0e-7, 3.0e-8):
        step = relative_step * scale
        try:
            value = direct(h_edge + step)
        except ZeroDivisionError:
            continue
        if _is_finite_complex(value):
            samples.append((step, value))

    if not samples:
        raise ZeroDivisionError(f"could not resolve glasses residue for r={r}, s={s}, h={h_edge!r}")
    if len(samples) == 1:
        return samples[-1][1]

    step_a, value_a = samples[-2]
    step_b, value_b = samples[-1]
    if step_a == step_b:
        return value_b
    return (step_a * value_b - step_b * value_a) / (step_a - step_b)


def handle_residue_prefactor(r: int, s: int, handle_weight: complex, bridge_weight: complex) -> complex:
    """Residue prefactor for a self-glued handle edge."""
    bridge_weight = _as_complex(bridge_weight)
    return _residue_prefactor_from_pair_functions(
        r,
        s,
        handle_weight,
        lambda h, level: (bridge_weight, h + level),
        lambda h, level: (bridge_weight, h),
    )


def bridge_residue_prefactor(
    r: int,
    s: int,
    bridge_weight: complex,
    left_weight: complex,
    right_weight: complex,
) -> complex:
    """Residue prefactor for the separating bridge edge."""
    left_weight = _as_complex(left_weight)
    right_weight = _as_complex(right_weight)
    return _residue_prefactor_from_pair_functions(
        r,
        s,
        bridge_weight,
        lambda h, level: (left_weight, left_weight),
        lambda h, level: (right_weight, right_weight),
    )


def ccy_genus2_glasses_block_partial_fraction(
    *,
    h_left: complex,
    h_right: complex,
    h_bridge: complex,
    q_left: complex,
    q_right: complex,
    q_bridge: complex,
    order: int,
    include_vacuum_seed: bool = True,
    vacuum_word_len: int = 6,
    vacuum_oscillator_level_max: int = 30,
    vacuum_word_tail_tolerance: float | None = None,
    vacuum_minimum_word_length: int = 5,
    resum_global_block: bool = False,
    global_block_tolerance: float = 1.0e-13,
    pole_tolerance: float = 1.0e-12,
) -> PartialFractionInC:
    """Return the separating genus-two block as a partial fraction in c."""
    order = _validate_order(order)
    q_left = _as_complex(q_left)
    q_right = _as_complex(q_right)
    q_bridge = _as_complex(q_bridge)
    global_block_tolerance = float(global_block_tolerance)

    vacuum_seed = (
        glasses_vacuum_seed_schottky(
            q_left,
            q_right,
            q_bridge,
            max_word_len=vacuum_word_len,
            oscillator_level_max=vacuum_oscillator_level_max,
            word_tail_tolerance=vacuum_word_tail_tolerance,
            minimum_word_length=vacuum_minimum_word_length,
        )
        if include_vacuum_seed
        else 1.0 + 0.0j
    )

    @lru_cache(maxsize=None)
    def recurse(
        current_h_left: complex,
        current_h_right: complex,
        current_h_bridge: complex,
        remaining: int,
    ) -> PartialFractionInC:
        if resum_global_block:
            global_seed = genus2_global_glasses_sl2_block_resummed(
                current_h_left,
                current_h_right,
                current_h_bridge,
                q_left,
                q_right,
                q_bridge,
                tolerance=global_block_tolerance,
            )
        else:
            global_seed = genus2_global_glasses_sl2_block(
                current_h_left,
                current_h_right,
                current_h_bridge,
                q_left,
                q_right,
                q_bridge,
                remaining,
            )
        seed = vacuum_seed * global_seed
        total = PartialFractionInC(constant=seed)

        edge_data = (
            ("left", current_h_left, q_left),
            ("right", current_h_right, q_right),
            ("bridge", current_h_bridge, q_bridge),
        )
        for edge_name, h_edge, q_value in edge_data:
            for r in range(2, remaining + 1):
                for s in range(1, remaining // r + 1):
                    level = r * s
                    if level > remaining:
                        continue
                    pole_c = c_rs_from_h(r, s, h_edge)
                    if edge_name == "left":
                        residue_prefactor = handle_residue_prefactor(r, s, h_edge, current_h_bridge)
                        shifted = (h_edge + level, current_h_right, current_h_bridge)
                    elif edge_name == "right":
                        residue_prefactor = handle_residue_prefactor(r, s, h_edge, current_h_bridge)
                        shifted = (current_h_left, h_edge + level, current_h_bridge)
                    else:
                        residue_prefactor = bridge_residue_prefactor(
                            r,
                            s,
                            h_edge,
                            current_h_left,
                            current_h_right,
                        )
                        shifted = (current_h_left, current_h_right, h_edge + level)
                    subblock = recurse(shifted[0], shifted[1], shifted[2], remaining - level)
                    total.add_residue_times_laurent_at(
                        pole=pole_c,
                        residue=(q_value**level) * residue_prefactor,
                        subblock=subblock,
                        pole_tolerance=pole_tolerance,
                    )
        return total

    return recurse(_as_complex(h_left), _as_complex(h_right), _as_complex(h_bridge), order)


def ccy_genus2_glasses_block(
    *,
    c: complex,
    h_left: complex,
    h_right: complex,
    h_bridge: complex,
    q_left: complex,
    q_right: complex,
    q_bridge: complex,
    order: int,
    include_vacuum_seed: bool = True,
    vacuum_word_len: int = 6,
    vacuum_oscillator_level_max: int = 30,
    vacuum_word_tail_tolerance: float | None = None,
    vacuum_minimum_word_length: int = 5,
    resum_global_block: bool = False,
    global_block_tolerance: float = 1.0e-13,
    pole_tolerance: float = 1.0e-12,
    collision_regulator_scale: float = 1.0e-3,
    collision_regulator_direction: Sequence[float] | None = None,
) -> CCYGenus2GlassesBlockResult:
    """Evaluate the CCY block in the separating/glasses plumbing frame."""
    c = _as_complex(c)
    weights = (
        _as_complex(h_left),
        _as_complex(h_right),
        _as_complex(h_bridge),
    )
    global_block_tolerance = float(global_block_tolerance)
    collision_regulator_scale = float(collision_regulator_scale)
    if (
        not math.isfinite(collision_regulator_scale)
        or collision_regulator_scale <= 0.0
    ):
        raise ValueError("collision regulator scale must be finite and positive")

    def build_partial(current_weights: tuple[complex, ...]) -> PartialFractionInC:
        return ccy_genus2_glasses_block_partial_fraction(
            h_left=current_weights[0],
            h_right=current_weights[1],
            h_bridge=current_weights[2],
            q_left=q_left,
            q_right=q_right,
            q_bridge=q_bridge,
            order=order,
            include_vacuum_seed=include_vacuum_seed,
            vacuum_word_len=vacuum_word_len,
            vacuum_oscillator_level_max=vacuum_oscillator_level_max,
            vacuum_word_tail_tolerance=vacuum_word_tail_tolerance,
            vacuum_minimum_word_length=vacuum_minimum_word_length,
            resum_global_block=resum_global_block,
            global_block_tolerance=global_block_tolerance,
            pole_tolerance=pole_tolerance,
        )

    regulated = False
    regulator_error = 0.0
    regulator_scale = 0.0
    try:
        partial_fraction = build_partial(weights)
        value = partial_fraction.value(c, pole_tolerance=pole_tolerance)
    except ConfluentPoleError:
        regulated_value = collision_regulated_partial_fraction_value(
            build_partial_fraction=build_partial,
            weights=weights,
            central_charge=c,
            pole_tolerance=pole_tolerance,
            relative_scale=collision_regulator_scale,
            direction=collision_regulator_direction,
        )
        partial_fraction = regulated_value.representative_partial_fraction
        value = regulated_value.value
        regulated = True
        regulator_error = regulated_value.error_estimate
        regulator_scale = regulated_value.relative_scale

    return CCYGenus2GlassesBlockResult(
        value=value,
        c=c,
        h_left=weights[0],
        h_right=weights[1],
        h_bridge=weights[2],
        q_left=_as_complex(q_left),
        q_right=_as_complex(q_right),
        q_bridge=_as_complex(q_bridge),
        order=_validate_order(order),
        include_vacuum_seed=bool(include_vacuum_seed),
        vacuum_word_len=int(vacuum_word_len),
        vacuum_oscillator_level_max=int(vacuum_oscillator_level_max),
        global_block_resummed=bool(resum_global_block),
        global_block_tolerance=(global_block_tolerance if resum_global_block else 0.0),
        partial_fraction_pole_count=partial_fraction.pole_count,
        partial_fraction_coefficient_count=partial_fraction.coefficient_count,
        partial_fraction_max_pole_order=partial_fraction.max_pole_order,
        collision_regulated=regulated,
        collision_regulator_error=regulator_error,
        collision_regulator_scale=regulator_scale,
    )


def liouville_c_and_weight(b: float, momentum: float) -> tuple[float, float]:
    q_background = b + 1.0 / b
    return 1.0 + 6.0 * q_background * q_background, 0.25 * q_background * q_background + momentum * momentum


def run() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the CCY genus-two glasses-frame Virasoro block.")
    parser.add_argument("--b", type=float, help="Liouville b; if set, c and h_i are read from P_i.")
    parser.add_argument("--p-left", type=float, default=0.2)
    parser.add_argument("--p-right", type=float, default=0.3)
    parser.add_argument("--p-bridge", type=float, default=0.25)
    parser.add_argument("--c", type=parse_complex)
    parser.add_argument("--h-left", type=parse_complex)
    parser.add_argument("--h-right", type=parse_complex)
    parser.add_argument("--h-bridge", type=parse_complex)
    parser.add_argument("--q-left", type=parse_complex, required=True)
    parser.add_argument("--q-right", type=parse_complex, required=True)
    parser.add_argument("--q-bridge", type=parse_complex, required=True)
    parser.add_argument("--order", type=int, default=2)
    parser.add_argument("--no-vacuum-seed", action="store_true")
    parser.add_argument("--vacuum-word-len", type=int, default=6)
    parser.add_argument("--vacuum-oscillator-level-max", type=int, default=30)
    parser.add_argument("--resum-global-block", action="store_true")
    parser.add_argument("--global-block-tolerance", type=float, default=1.0e-13)
    args = parser.parse_args()

    if args.b is not None:
        c_value, h_left = liouville_c_and_weight(args.b, args.p_left)
        _, h_right = liouville_c_and_weight(args.b, args.p_right)
        _, h_bridge = liouville_c_and_weight(args.b, args.p_bridge)
    else:
        if args.c is None or args.h_left is None or args.h_right is None or args.h_bridge is None:
            raise ValueError("pass either --b with momenta or explicit --c/--h-left/--h-right/--h-bridge")
        c_value = args.c
        h_left = args.h_left
        h_right = args.h_right
        h_bridge = args.h_bridge

    result = ccy_genus2_glasses_block(
        c=c_value,
        h_left=h_left,
        h_right=h_right,
        h_bridge=h_bridge,
        q_left=args.q_left,
        q_right=args.q_right,
        q_bridge=args.q_bridge,
        order=args.order,
        include_vacuum_seed=not args.no_vacuum_seed,
        vacuum_word_len=args.vacuum_word_len,
        vacuum_oscillator_level_max=args.vacuum_oscillator_level_max,
        resum_global_block=args.resum_global_block,
        global_block_tolerance=args.global_block_tolerance,
    )

    print("CCY genus-two glasses-frame Virasoro block")
    print(f"  c={format_complex(result.c)}")
    print(f"  h_left={format_complex(result.h_left)}")
    print(f"  h_right={format_complex(result.h_right)}")
    print(f"  h_bridge={format_complex(result.h_bridge)}")
    print(f"  q_left={format_complex(result.q_left)}")
    print(f"  q_right={format_complex(result.q_right)}")
    print(f"  q_bridge={format_complex(result.q_bridge)}")
    print(f"  order={result.order}")
    print(f"  vacuum seed={result.include_vacuum_seed}")
    print(f"  global block resummed={result.global_block_resummed}")
    print(f"  c-poles={result.partial_fraction_pole_count}")
    print(f"  c-pole coefficients={result.partial_fraction_coefficient_count}")
    print(f"  max c-pole order={result.partial_fraction_max_pole_order}")
    print(f"  value={format_complex(result.value)}")


if __name__ == "__main__":
    run()
