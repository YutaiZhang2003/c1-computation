"""Native necklace, pair, and linear-channel torus three-point blocks."""
from __future__ import annotations

#!/usr/bin/env python3
# torus_three_point_blocks
# Direct PBW and fixed-weight c-recursive torus three-point necklace blocks.
# 
# All coefficient arrays use the native cylinder nomes ``(q1,q2,q3)`` and
# contain descendants only; primary propagation belongs to the correlator.




from c1_paths import PACKAGE_ROOT, data_root, package_path, module_path

from itertools import product
from functools import lru_cache
import math

import numpy as np

from ccy_genus2_block import (
    b_from_c_rs_h,
    c_rs_from_h,
    fusion_polynomial_for_weights,
    minus_dc_dh_times_a_rs,
)

try:
    from torus_descendant_blocks import gram_matrix, rho_primary_external
except ImportError:  # pragma: no cover
    from torus_descendant_blocks import gram_matrix, rho_primary_external


def necklace_descendant_coefficients_three_point(
    c: complex,
    internal_weights: tuple[complex, complex, complex],
    external_weights: tuple[complex, complex, complex],
    orders: tuple[int, int, int],
) -> np.ndarray:
    """Evaluate the defining three-edge descendant contraction exactly.

    This finite-level sewing sum is the regular c=25 value of the block.  It
    is also the reference used to validate the regulated h-recursion, whose
    individual residues are resonant at b=1.
    """
    if len(internal_weights) != 3 or len(external_weights) != 3 or len(orders) != 3:
        raise ValueError("the torus three-point necklace requires three edges")
    if any(int(order) < 0 for order in orders):
        raise ValueError("block orders must be non-negative")

    edge_data: list[list[tuple[tuple[tuple[int, ...], ...], np.ndarray]]] = []
    for weight, maximum_level in zip(internal_weights, orders):
        levels = []
        for level in range(int(maximum_level) + 1):
            basis, gram = gram_matrix(weight, c, level)
            levels.append((basis, np.linalg.inv(gram)))
        edge_data.append(levels)

    shape = tuple(int(order) + 1 for order in orders)
    coefficients = np.zeros(shape, dtype=np.complex128)
    for levels in np.ndindex(shape):
        dimensions = tuple(len(edge_data[edge][levels[edge]][0]) for edge in range(3))
        value = 0.0 + 0.0j
        for ket_indices in product(*(range(dimension) for dimension in dimensions)):
            for bra_indices in product(*(range(dimension) for dimension in dimensions)):
                term = 1.0 + 0.0j
                for edge in range(3):
                    next_edge = (edge + 1) % 3
                    basis, inverse_gram = edge_data[edge][levels[edge]]
                    next_basis, _ = edge_data[next_edge][levels[next_edge]]
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


def necklace_c_recursion_coefficients_three_point(
    c: complex,
    internal_weights: tuple[complex, complex, complex],
    external_weights: tuple[complex, complex, complex],
    orders: tuple[int, int, int],
    *,
    pole_tolerance: float = 1.0e-12,
    dps: int | None = None,
) -> np.ndarray:
    """Return the native necklace series from fixed-weight CCY c recursion.

    The regular large-c term is the cyclic SL(2) descendant contraction times
    ``prod_(m>=2) (1-(q1*q2*q3)**m)**(-1)``. At a pole belonging to edge i,
    only its weight is shifted by rs; all neighboring weights remain fixed.

    Set ``dps`` to use the shared arbitrary-precision c-recursion primitive.
    Both primitives deliberately raise on confluent poles.
    Numerical continuation/precision checks belong to the shared c-recursion
    stability wrapper, rather than to a central-charge or h-recursion fallback.
    """
    if len(internal_weights) != 3 or len(external_weights) != 3 or len(orders) != 3:
        raise ValueError("the torus three-point necklace requires three edges")
    normalized_orders = tuple(int(order) for order in orders)
    if any(order < 0 for order in normalized_orders):
        raise ValueError("block orders must be non-negative")
    if not math.isfinite(pole_tolerance) or pole_tolerance <= 0.0:
        raise ValueError("pole_tolerance must be finite and positive")
    if dps is not None:
        from torus_c_recursion_numerics import mp_c_recursion_coefficients

        return mp_c_recursion_coefficients(
            "necklace", c, external_weights=external_weights,
            internal_weights=internal_weights, orders=normalized_orders, dps=dps,
        )
    internal = tuple(complex(weight) for weight in internal_weights)
    external = tuple(complex(weight) for weight in external_weights)
    vacuum = [0] * (min(normalized_orders) + 1)
    vacuum[0] = 1
    for oscillator in range(2, len(vacuum)):
        for level in range(oscillator, len(vacuum)):
            vacuum[level] += vacuum[level - oscillator]

    @lru_cache(maxsize=None)
    def rising(weight: complex, level: int) -> complex:
        return math.prod(weight + offset for offset in range(level))

    @lru_cache(maxsize=None)
    def two_leg(
        bra_level: int,
        ket_level: int,
        h_bra: complex,
        d: complex,
        h_ket: complex,
    ) -> complex:
        # <h_bra|L_1^n V_d(1) L_-1^m|h_ket>, from the global Ward identity.
        if bra_level == 0:
            return rising(d + h_ket - h_bra, ket_level)
        value = (h_bra + bra_level - 1 + d - h_ket - ket_level) * two_leg(
            bra_level - 1, ket_level, h_bra, d, h_ket
        )
        if ket_level:
            value += ket_level * (2 * h_ket + ket_level - 1) * two_leg(
                bra_level - 1, ket_level - 1, h_bra, d, h_ket
            )
        return value

    @lru_cache(maxsize=None)
    def global_coefficient(levels: tuple[int, int, int], hs: tuple[complex, ...]) -> complex:
        numerator = math.prod(
            two_leg(levels[edge], levels[(edge + 1) % 3], hs[edge],
                    external[edge], hs[(edge + 1) % 3])
            for edge in range(3)
        )
        denominator = math.prod(
            math.factorial(level) * rising(2 * weight, level)
            for weight, level in zip(hs, levels)
        )
        if denominator == 0:
            raise ZeroDivisionError("singular global necklace descendant norm")
        return numerator / denominator

    @lru_cache(maxsize=None)
    def residue(r: int, s: int, edge: int, hs: tuple[complex, ...]) -> tuple[complex, complex]:
        pole_c = c_rs_from_h(r, s, hs[edge])
        b_pole = b_from_c_rs_h(r, s, hs[edge])
        previous, following = (edge - 1) % 3, (edge + 1) % 3
        value = (
            minus_dc_dh_times_a_rs(r, s, hs[edge])
            * fusion_polynomial_for_weights(r, s, b_pole, hs[previous], external[previous])
            * fusion_polynomial_for_weights(r, s, b_pole, hs[following], external[edge])
        )
        return pole_c, value

    @lru_cache(maxsize=None)
    def coefficient(levels: tuple[int, int, int], current_c: complex, hs: tuple[complex, ...]) -> complex:
        total = sum(
            vacuum[oscillator_level]
            * global_coefficient(tuple(n - oscillator_level for n in levels), hs)
            for oscillator_level in range(min(levels) + 1)
        )
        for edge, available in enumerate(levels):
            for r in range(2, available + 1):
                for s in range(1, available // r + 1):
                    pole_c, weight = residue(r, s, edge, hs)
                    denominator = current_c - pole_c
                    if abs(denominator) < pole_tolerance:
                        raise ZeroDivisionError(
                            f"necklace c-recursion hit an edge-{edge} pole c_({r},{s})={pole_c!r}"
                        )
                    next_levels, next_weights = list(levels), list(hs)
                    next_levels[edge] -= r * s
                    next_weights[edge] += r * s
                    total += weight / denominator * coefficient(
                        tuple(next_levels), pole_c, tuple(next_weights)
                    )
        return complex(total)

    try:
        coefficients = np.empty(tuple(order + 1 for order in normalized_orders), dtype=np.complex128)
        for levels in np.ndindex(coefficients.shape):
            coefficients[levels] = coefficient(levels, complex(c), internal)
        return coefficients
    finally:
        for cached in (rising, two_leg, global_coefficient, residue, coefficient):
            cached.cache_clear()


#!/usr/bin/env python3
# torus_three_point_ope_blocks
# Pair-OPE channel for a primary torus three-point Virasoro block.
# 
# Two external primaries of weights ``d_a,d_b`` fuse through ``h_ope``.  The
# resulting descendant is inserted in a two-edge torus necklace together with
# the spectator primary ``d_s``.  The three plumbing variables are ordered as
# ``(v,q_first,q_second)``.  Only the descendant series is returned; primary
# propagation and the disc-to-flat-cylinder factor are inserted by the
# worldsheet kernel.
# 
# Both an exact finite-level contraction and the fixed-weight CCY
# central-charge recursion are supplied.  The large-c regular term is the
# three-edge global block convolved with the torus vacuum oscillators in the
# product ``q_first*q_second``.




from c1_paths import PACKAGE_ROOT, data_root, package_path, module_path

import math
from functools import lru_cache

import numpy as np

try:
    from ccy_genus2_block import (
        b_from_c_rs_h,
        c_rs_from_h,
        fusion_polynomial_for_weights,
        minus_dc_dh_times_a_rs,
    )
    from virasoro_plumbing_graph import (
        inverse_verma_gram_matrix,
        rho_primary_descendants,
    )
except ImportError:  # pragma: no cover - package-style execution
    from ccy_genus2_block import (
        b_from_c_rs_h,
        c_rs_from_h,
        fusion_polynomial_for_weights,
        minus_dc_dh_times_a_rs,
    )
    from virasoro_plumbing_graph import (
        inverse_verma_gram_matrix,
        rho_primary_descendants,
    )


def _validate_orders(orders: tuple[int, int, int]) -> tuple[int, int, int]:
    if len(orders) != 3:
        raise ValueError("pair-OPE orders must be (ope,first_loop,second_loop)")
    normalized = tuple(int(order) for order in orders)
    if any(order < 0 for order in normalized):
        raise ValueError("block orders must be non-negative")
    return normalized  # type: ignore[return-value]


def _validate_weights(
    external_weights: tuple[complex, complex, complex],
    internal_weights: tuple[complex, complex, complex],
) -> tuple[tuple[complex, complex, complex], tuple[complex, complex, complex]]:
    if len(external_weights) != 3 or len(internal_weights) != 3:
        raise ValueError("three external and three internal weights are required")
    return (
        tuple(complex(value) for value in external_weights),
        tuple(complex(value) for value in internal_weights),
    )  # type: ignore[return-value]


def _rising(value: complex, order: int) -> complex:
    out = 1.0 + 0.0j
    for offset in range(int(order)):
        out *= complex(value) + offset
    return complex(out)


def _global_norm(weight: complex, level: int) -> complex:
    return complex(math.factorial(int(level)) * _rising(2.0 * weight, level))


@lru_cache(maxsize=None)
def _vacuum_character_without_lminus1(order: int) -> tuple[int, ...]:
    coefficients = [0] * (int(order) + 1)
    coefficients[0] = 1
    for oscillator in range(2, int(order) + 1):
        for level in range(oscillator, int(order) + 1):
            coefficients[level] += coefficients[level - oscillator]
    return tuple(coefficients)


def pair_ope_global_coefficient(
    levels: tuple[int, int, int],
    *,
    external_weights: tuple[complex, complex, complex],
    internal_weights: tuple[complex, complex, complex],
) -> complex:
    """Return one coefficient of the graph-global ``SL(2)`` block."""
    n_ope, n_first, n_second = _validate_orders(levels)
    external, internal = _validate_weights(external_weights, internal_weights)
    d_a, d_b, d_s = external
    h_ope, h_first, h_second = internal
    # The local plane has d_b at zero and d_a at v.  In the repository's
    # rho convention this is rho(h_ope,d_a,d_b).
    outer = _rising(h_ope + d_a - d_b, n_ope)
    fusion_vertex = rho_primary_descendants(
        (1,) * n_first,
        (1,) * n_ope,
        (1,) * n_second,
        h_first,
        h_ope,
        h_second,
        0.0,
    )
    spectator_vertex = rho_primary_descendants(
        (1,) * n_second,
        (),
        (1,) * n_first,
        h_second,
        d_s,
        h_first,
        0.0,
    )
    denominator = (
        _global_norm(h_ope, n_ope)
        * _global_norm(h_first, n_first)
        * _global_norm(h_second, n_second)
    )
    if denominator == 0.0:
        raise ZeroDivisionError("singular global pair-OPE descendant norm")
    return complex(outer * fusion_vertex * spectator_vertex / denominator)


def pair_ope_large_c_coefficient(
    levels: tuple[int, int, int],
    *,
    external_weights: tuple[complex, complex, complex],
    internal_weights: tuple[complex, complex, complex],
) -> complex:
    """Return the CCY regular term including the torus vacuum seed."""
    n_ope, n_first, n_second = _validate_orders(levels)
    vacuum = _vacuum_character_without_lminus1(min(n_first, n_second))
    return complex(
        sum(
            vacuum[oscillator_level]
            * pair_ope_global_coefficient(
                (n_ope, n_first - oscillator_level, n_second - oscillator_level),
                external_weights=external_weights,
                internal_weights=internal_weights,
            )
            for oscillator_level in range(min(n_first, n_second) + 1)
        )
    )


def pair_ope_direct_coefficients(
    central_charge: complex,
    *,
    external_weights: tuple[complex, complex, complex],
    internal_weights: tuple[complex, complex, complex],
    orders: tuple[int, int, int],
) -> np.ndarray:
    """Evaluate the defining three-edge descendant contraction."""
    orders = _validate_orders(orders)
    external, internal = _validate_weights(external_weights, internal_weights)
    d_a, d_b, d_s = external
    h_ope, h_first, h_second = internal
    c_value = complex(central_charge)
    grams = tuple(
        {
            level: inverse_verma_gram_matrix(level, weight, c_value)[:2]
            for level in range(order + 1)
        }
        for weight, order in zip(internal, orders)
    )
    coefficients = np.zeros(tuple(order + 1 for order in orders), dtype=np.complex128)
    for levels in np.ndindex(coefficients.shape):
        bases_and_metrics = tuple(grams[edge][levels[edge]] for edge in range(3))
        (basis_ope, inverse_ope), (basis_first, inverse_first), (
            basis_second,
            inverse_second,
        ) = bases_and_metrics
        value = 0.0 + 0.0j
        for a_ope_index, a_ope in enumerate(basis_ope):
            for b_ope_index, b_ope in enumerate(basis_ope):
                metric_ope = inverse_ope[a_ope_index, b_ope_index]
                outer = rho_primary_descendants(
                    b_ope,
                    (),
                    (),
                    h_ope,
                    d_a,
                    d_b,
                    c_value,
                )
                for a_first_index, a_first in enumerate(basis_first):
                    for b_first_index, b_first in enumerate(basis_first):
                        metric_first = inverse_first[a_first_index, b_first_index]
                        for a_second_index, a_second in enumerate(basis_second):
                            fusion = rho_primary_descendants(
                                b_first,
                                a_ope,
                                a_second,
                                h_first,
                                h_ope,
                                h_second,
                                c_value,
                            )
                            if fusion == 0.0:
                                continue
                            for b_second_index, b_second in enumerate(basis_second):
                                metric_second = inverse_second[
                                    a_second_index,
                                    b_second_index,
                                ]
                                spectator = rho_primary_descendants(
                                    b_second,
                                    (),
                                    a_first,
                                    h_second,
                                    d_s,
                                    h_first,
                                    c_value,
                                )
                                value += (
                                    metric_ope
                                    * metric_first
                                    * metric_second
                                    * outer
                                    * fusion
                                    * spectator
                                )
        coefficients[levels] = value
    return coefficients


def pair_ope_c_recursion_coefficients(
    central_charge: complex,
    *,
    external_weights: tuple[complex, complex, complex],
    internal_weights: tuple[complex, complex, complex],
    orders: tuple[int, int, int],
    pole_tolerance: float = 1.0e-12,
    dps: int | None = None,
) -> np.ndarray:
    """Return the fixed-weight CCY central-charge recursion coefficients.

    ``dps`` selects the type-preserving multiprecision arithmetic. Production
    callers use ``stable_c_recursion_coefficients('pair', ...)`` to additionally
    verify precision and any confluent-weight limit.
    """
    if dps is not None:
        from torus_c_recursion_numerics import mp_c_recursion_coefficients
        return mp_c_recursion_coefficients(
            "pair", central_charge, external_weights=external_weights,
            internal_weights=internal_weights, orders=orders, dps=dps)
    orders = _validate_orders(orders)
    external, initial_internal = _validate_weights(external_weights, internal_weights)
    d_a, d_b, d_s = external
    pole_tolerance = float(pole_tolerance)
    if pole_tolerance <= 0.0:
        raise ValueError("pole_tolerance must be positive")

    @lru_cache(maxsize=None)
    def coefficient(
        n_ope: int,
        n_first: int,
        n_second: int,
        current_c: complex,
        h_ope: complex,
        h_first: complex,
        h_second: complex,
    ) -> complex:
        levels = (n_ope, n_first, n_second)
        weights = (h_ope, h_first, h_second)
        total = pair_ope_large_c_coefficient(
            levels,
            external_weights=external,
            internal_weights=weights,
        )
        edge_data = (
            (n_ope, h_ope, (d_b, d_a), (h_first, h_second), 0, True),
            (n_first, h_first, (h_second, d_s), (h_ope, h_second), 1, True),
            (n_second, h_second, (h_first, d_s), (h_first, h_ope), 2, False),
        )
        for edge_order, edge_weight, first_pair, second_pair, edge, bpz_phase in edge_data:
            for r in range(2, edge_order + 1):
                for s in range(1, edge_order // r + 1):
                    null_level = r * s
                    pole_c = c_rs_from_h(r, s, edge_weight)
                    denominator = current_c - pole_c
                    if abs(denominator) < pole_tolerance:
                        raise ZeroDivisionError(
                            f"pair-OPE c-recursion hit an edge-{edge} pole"
                        )
                    b_pole = b_from_c_rs_h(r, s, edge_weight)
                    residue = (
                        minus_dc_dh_times_a_rs(r, s, edge_weight)
                        * fusion_polynomial_for_weights(r, s, b_pole, *first_pair)
                        * fusion_polynomial_for_weights(r, s, b_pole, *second_pair)
                    )
                    if bpz_phase and null_level % 2:
                        residue = -residue
                    next_levels = list(levels)
                    next_levels[edge] -= null_level
                    next_weights = list(weights)
                    next_weights[edge] += null_level
                    total += residue / denominator * coefficient(
                        *next_levels,
                        pole_c,
                        *next_weights,
                    )
        return complex(total)

    coefficients = np.empty(tuple(order + 1 for order in orders), dtype=np.complex128)
    for levels in np.ndindex(coefficients.shape):
        coefficients[levels] = coefficient(
            *levels,
            complex(central_charge),
            *initial_internal,
        )
    return coefficients


def pair_ope_coefficients_in_elliptic_loop_nomes(
    coefficients: np.ndarray,
    *,
    first_loop_order: int | None = None,
    second_loop_order: int | None = None,
) -> np.ndarray:
    """Fail closed for archived callers; do not re-expand recursion output."""
    raise ValueError("elliptic coefficient conversion removed; evaluate raw sewing q")


def pair_ope_coefficients_in_local_delta(
    coefficients: np.ndarray,
    *,
    delta_order: int | None = None,
) -> np.ndarray:
    """Fail closed: native v (or the comb ratio) is supplied at evaluation."""
    raise ValueError("local-delta coefficient conversion removed; evaluate native sewing variables")


def comb_ope_global_coefficient(
    levels: tuple[int, int, int],
    *,
    external_weights: tuple[complex, complex, complex],
    internal_weights: tuple[complex, complex, complex],
) -> complex:
    """Global coefficient for two successive OPEs followed by a torus loop."""
    n_pair, n_total, n_loop = _validate_orders(levels)
    external, internal = _validate_weights(external_weights, internal_weights)
    d_a, d_b, d_s = external
    h_pair, h_total, h_loop = internal
    outer = _rising(h_pair + d_a - d_b, n_pair)
    second_fusion = rho_primary_descendants(
        (1,) * n_total,
        (),
        (1,) * n_pair,
        h_total,
        d_s,
        h_pair,
        0.0,
    )
    tadpole = rho_primary_descendants(
        (1,) * n_loop,
        (1,) * n_total,
        (1,) * n_loop,
        h_loop,
        h_total,
        h_loop,
        0.0,
    )
    denominator = (
        _global_norm(h_pair, n_pair)
        * _global_norm(h_total, n_total)
        * _global_norm(h_loop, n_loop)
    )
    if denominator == 0.0:
        raise ZeroDivisionError("singular global comb-OPE descendant norm")
    return complex(outer * second_fusion * tadpole / denominator)


def comb_ope_large_c_coefficient(
    levels: tuple[int, int, int],
    *,
    external_weights: tuple[complex, complex, complex],
    internal_weights: tuple[complex, complex, complex],
) -> complex:
    """Large-c comb seed convolved with the torus vacuum oscillators."""
    n_pair, n_total, n_loop = _validate_orders(levels)
    vacuum = _vacuum_character_without_lminus1(n_loop)
    return complex(
        sum(
            vacuum[oscillator_level]
            * comb_ope_global_coefficient(
                (n_pair, n_total, n_loop - oscillator_level),
                external_weights=external_weights,
                internal_weights=internal_weights,
            )
            for oscillator_level in range(n_loop + 1)
        )
    )


def comb_ope_direct_coefficients(
    central_charge: complex,
    *,
    external_weights: tuple[complex, complex, complex],
    internal_weights: tuple[complex, complex, complex],
    orders: tuple[int, int, int],
) -> np.ndarray:
    """Exact descendant contraction for the fully local torus comb chart."""
    orders = _validate_orders(orders)
    external, internal = _validate_weights(external_weights, internal_weights)
    d_a, d_b, d_s = external
    h_pair, h_total, h_loop = internal
    c_value = complex(central_charge)
    grams = tuple(
        {
            level: inverse_verma_gram_matrix(level, weight, c_value)[:2]
            for level in range(order + 1)
        }
        for weight, order in zip(internal, orders)
    )
    coefficients = np.zeros(tuple(order + 1 for order in orders), dtype=np.complex128)
    for levels in np.ndindex(coefficients.shape):
        (basis_pair, inverse_pair), (basis_total, inverse_total), (
            basis_loop,
            inverse_loop,
        ) = tuple(grams[edge][levels[edge]] for edge in range(3))
        value = 0.0 + 0.0j
        for a_pair_index, a_pair in enumerate(basis_pair):
            for b_pair_index, b_pair in enumerate(basis_pair):
                metric_pair = inverse_pair[a_pair_index, b_pair_index]
                outer = rho_primary_descendants(
                    b_pair,
                    (),
                    (),
                    h_pair,
                    d_a,
                    d_b,
                    c_value,
                )
                for a_total_index, a_total in enumerate(basis_total):
                    for b_total_index, b_total in enumerate(basis_total):
                        metric_total = inverse_total[a_total_index, b_total_index]
                        second_fusion = rho_primary_descendants(
                            b_total,
                            (),
                            a_pair,
                            h_total,
                            d_s,
                            h_pair,
                            c_value,
                        )
                        if second_fusion == 0.0:
                            continue
                        for a_loop_index, a_loop in enumerate(basis_loop):
                            for b_loop_index, b_loop in enumerate(basis_loop):
                                metric_loop = inverse_loop[a_loop_index, b_loop_index]
                                tadpole = rho_primary_descendants(
                                    b_loop,
                                    a_total,
                                    a_loop,
                                    h_loop,
                                    h_total,
                                    h_loop,
                                    c_value,
                                )
                                value += (
                                    metric_pair
                                    * metric_total
                                    * metric_loop
                                    * outer
                                    * second_fusion
                                    * tadpole
                                )
        coefficients[levels] = value
    return coefficients


def comb_ope_c_recursion_coefficients(
    central_charge: complex,
    *,
    external_weights: tuple[complex, complex, complex],
    internal_weights: tuple[complex, complex, complex],
    orders: tuple[int, int, int],
    pole_tolerance: float = 1.0e-12,
    dps: int | None = None,
) -> np.ndarray:
    """Fixed-weight CCY recursion for the fully local torus comb chart.

    ``dps`` selects multiprecision arithmetic; the shared stable helper also
    enforces precision and confluent-weight-limit checks for production.
    """
    if dps is not None:
        from torus_c_recursion_numerics import mp_c_recursion_coefficients
        return mp_c_recursion_coefficients(
            "comb", central_charge, external_weights=external_weights,
            internal_weights=internal_weights, orders=orders, dps=dps)
    orders = _validate_orders(orders)
    external, initial_internal = _validate_weights(external_weights, internal_weights)
    d_a, d_b, d_s = external
    pole_tolerance = float(pole_tolerance)
    if pole_tolerance <= 0.0:
        raise ValueError("pole_tolerance must be positive")

    @lru_cache(maxsize=None)
    def coefficient(
        n_pair: int,
        n_total: int,
        n_loop: int,
        current_c: complex,
        h_pair: complex,
        h_total: complex,
        h_loop: complex,
    ) -> complex:
        levels = (n_pair, n_total, n_loop)
        weights = (h_pair, h_total, h_loop)
        total = comb_ope_large_c_coefficient(
            levels,
            external_weights=external,
            internal_weights=weights,
        )
        ordinary_edges = (
            (n_pair, h_pair, (d_b, d_a), (h_total, d_s), 0),
            (n_total, h_total, (h_pair, d_s), (h_loop, h_loop), 1),
        )
        for edge_order, edge_weight, first_pair, second_pair, edge in ordinary_edges:
            for r in range(2, edge_order + 1):
                for s in range(1, edge_order // r + 1):
                    null_level = r * s
                    pole_c = c_rs_from_h(r, s, edge_weight)
                    denominator = current_c - pole_c
                    if abs(denominator) < pole_tolerance:
                        raise ZeroDivisionError(
                            f"comb-OPE c-recursion hit an edge-{edge} pole"
                        )
                    b_pole = b_from_c_rs_h(r, s, edge_weight)
                    residue = (
                        minus_dc_dh_times_a_rs(r, s, edge_weight)
                        * fusion_polynomial_for_weights(r, s, b_pole, *first_pair)
                        * fusion_polynomial_for_weights(r, s, b_pole, *second_pair)
                    )
                    next_levels = list(levels)
                    next_levels[edge] -= null_level
                    next_weights = list(weights)
                    next_weights[edge] += null_level
                    total += residue / denominator * coefficient(
                        *next_levels,
                        pole_c,
                        *next_weights,
                    )

        for r in range(2, n_loop + 1):
            for s in range(1, n_loop // r + 1):
                null_level = r * s
                pole_c = c_rs_from_h(r, s, h_loop)
                denominator = current_c - pole_c
                if abs(denominator) < pole_tolerance:
                    raise ZeroDivisionError("comb-OPE c-recursion hit the loop-edge pole")
                b_pole = b_from_c_rs_h(r, s, h_loop)
                residue = (
                    minus_dc_dh_times_a_rs(r, s, h_loop)
                    * fusion_polynomial_for_weights(
                        r,
                        s,
                        b_pole,
                        h_total,
                        h_loop + null_level,
                    )
                    * fusion_polynomial_for_weights(
                        r,
                        s,
                        b_pole,
                        h_total,
                        h_loop,
                    )
                )
                total += residue / denominator * coefficient(
                    n_pair,
                    n_total,
                    n_loop - null_level,
                    pole_c,
                    h_pair,
                    h_total,
                    h_loop + null_level,
                )
        return complex(total)

    coefficients = np.empty(tuple(order + 1 for order in orders), dtype=np.complex128)
    for levels in np.ndindex(coefficients.shape):
        coefficients[levels] = coefficient(
            *levels,
            complex(central_charge),
            *initial_internal,
        )
    return coefficients


