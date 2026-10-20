#!/usr/bin/env python3
"""Direct Virasoro conformal blocks for trivalent plumbing graphs.

This module implements the defining descendant sum in the plane plumbing
frame of Cho--Collier--Yin, arXiv:1703.09805.  Each pair of pants is represented
by a trivalent vertex whose ordered slots are the insertions at infinity, one,
and zero.  Each internal edge contracts two descendant states with the inverse
Verma-module Gram matrix.

The returned block contains the descendant powers ``q_e**level`` only.  The
primary propagation factor ``prod_e q_e**h_e`` and any CFT three-point
structure constants are deliberately not included.
"""

from __future__ import annotations

from c1_paths import PACKAGE_ROOT, data_root, package_path, module_path

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Mapping, Sequence

import numpy as np

try:
    from virasoro_descendant_algebra import (
        Descendant,
        State,
        act_virasoro_mode,
        descendant_inner_product,
        descendant_level,
        integer_partitions,
    )
except ImportError:  # pragma: no cover - supports package-style execution
    from virasoro_descendant_algebra import (
        Descendant,
        State,
        act_virasoro_mode,
        descendant_inner_product,
        descendant_level,
        integer_partitions,
    )


INFINITY_SLOT = 0
ONE_SLOT = 1
ZERO_SLOT = 2
SLOT_NAMES = ("infinity", "one", "zero")

PowerSeriesItems = tuple[tuple[int, complex], ...]


def verma_descendant_basis(level: int) -> tuple[Descendant, ...]:
    """Return the PBW basis ``L_-A |h>`` at one non-negative level."""

    level = int(level)
    if level < 0:
        raise ValueError("level must be non-negative")
    return tuple(integer_partitions(level, min_part=1))


@lru_cache(maxsize=None)
def verma_gram_matrix(
    level: int,
    weight: complex,
    central_charge: complex,
) -> tuple[tuple[Descendant, ...], tuple[tuple[complex, ...], ...]]:
    """Return the generic-primary Verma Gram matrix at fixed level."""

    basis = verma_descendant_basis(level)
    matrix = tuple(
        tuple(
            descendant_inner_product(
                left,
                right,
                h=complex(weight),
                c=complex(central_charge),
                vacuum=False,
            )
            for right in basis
        )
        for left in basis
    )
    return basis, matrix


def inverse_verma_gram_matrix(
    level: int,
    weight: complex,
    central_charge: complex,
) -> tuple[tuple[Descendant, ...], np.ndarray, float]:
    """Return a basis, inverse Gram matrix, and 2-norm condition number.

    The generic Verma basis is singular at degenerate weights.  Such a point
    requires either a quotient module or analytic continuation and is not
    silently regularized here.
    """

    basis, matrix_items = verma_gram_matrix(level, complex(weight), complex(central_charge))
    matrix = np.asarray(matrix_items, dtype=complex)
    try:
        inverse = np.linalg.inv(matrix)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "singular Verma Gram matrix at "
            f"level={level}, h={weight!r}, c={central_charge!r}; "
            "degenerate and vacuum modules require an explicit quotient"
        ) from exc
    return basis, inverse, float(np.linalg.cond(matrix))


def _series_items(series: Mapping[int, complex]) -> PowerSeriesItems:
    return tuple(
        (int(shift), complex(coefficient))
        for shift, coefficient in sorted(series.items())
        if coefficient != 0
    )


def _add_series(*terms: tuple[complex, PowerSeriesItems]) -> PowerSeriesItems:
    out: dict[int, complex] = {}
    for scale, items in terms:
        for shift, coefficient in items:
            out[shift] = out.get(shift, 0.0j) + complex(scale) * coefficient
    return _series_items(out)


def _primary_commutator_differential(
    items: PowerSeriesItems,
    *,
    mode: int,
    middle_weight: complex,
    base_exponent: complex,
) -> PowerSeriesItems:
    r"""Apply ``z^m(z d/dz + (m+1)h_2)`` to a power series.

    The series is represented as

    ``sum_k coefficient[k] z**(base_exponent + k)``.
    """

    mode = int(mode)
    middle_weight = complex(middle_weight)
    base_exponent = complex(base_exponent)
    out: dict[int, complex] = {}
    for shift, coefficient in items:
        new_shift = shift + mode
        factor = base_exponent + shift + (mode + 1) * middle_weight
        out[new_shift] = out.get(new_shift, 0.0j) + coefficient * factor
    return _series_items(out)


@lru_cache(maxsize=None)
def _two_leg_matrix_element_series(
    desc_infinity: Descendant,
    desc_zero: Descendant,
    h_infinity: complex,
    h_one: complex,
    h_zero: complex,
    central_charge: complex,
) -> PowerSeriesItems:
    r"""Return ``rho(desc_inf, h_one, desc_zero | z)`` as a power series.

    This is the base case in which the insertion at ``z`` is primary.  Positive
    bra modes are commuted through that primary using

    ``[L_m,V_h(z)] = z^m(z d/dz + (m+1)h)V_h(z)``.
    """

    desc_infinity = tuple(desc_infinity)
    desc_zero = tuple(desc_zero)
    h_infinity = complex(h_infinity)
    h_one = complex(h_one)
    h_zero = complex(h_zero)
    central_charge = complex(central_charge)
    exponent = h_infinity - h_one - h_zero

    if desc_infinity:
        mode = int(desc_infinity[0])
        rest_infinity = desc_infinity[1:]
        commutator_term = _primary_commutator_differential(
            _two_leg_matrix_element_series(
                rest_infinity,
                desc_zero,
                h_infinity,
                h_one,
                h_zero,
                central_charge,
            ),
            mode=mode,
            middle_weight=h_one,
            base_exponent=exponent,
        )
        acted_zero = act_virasoro_mode(
            mode,
            {desc_zero: 1.0 + 0.0j},
            h=h_zero,
            c=central_charge,
            vacuum=False,
        )
        terms: list[tuple[complex, PowerSeriesItems]] = [(1.0, commutator_term)]
        for resulting_desc, coefficient in acted_zero.items():
            terms.append(
                (
                    coefficient,
                    _two_leg_matrix_element_series(
                        rest_infinity,
                        resulting_desc,
                        h_infinity,
                        h_one,
                        h_zero,
                        central_charge,
                    ),
                )
            )
        return _add_series(*terms)

    if desc_zero:
        mode = int(desc_zero[0])
        rest_zero = desc_zero[1:]
        # <h_inf|L_-mode=0, so V L_-mode = -[L_-mode,V].
        return _add_series(
            (
                -1.0,
                _primary_commutator_differential(
                    _two_leg_matrix_element_series(
                        (),
                        rest_zero,
                        h_infinity,
                        h_one,
                        h_zero,
                        central_charge,
                    ),
                    mode=-mode,
                    middle_weight=h_one,
                    base_exponent=exponent,
                ),
            )
        )

    return ((0, 1.0 + 0.0j),)


def _rho_apply_mode(
    mode: int,
    desc: Descendant,
    *,
    weight: complex,
    central_charge: complex,
) -> State:
    return act_virasoro_mode(
        int(mode),
        {tuple(desc): 1.0 + 0.0j},
        h=complex(weight),
        c=complex(central_charge),
        vacuum=False,
    )


def _rho_state_sum(
    state_infinity: State,
    desc_one: Descendant,
    state_zero: State,
    *,
    h_infinity: complex,
    h_one: complex,
    h_zero: complex,
    central_charge: complex,
) -> complex:
    total = 0.0j
    for desc_infinity, coefficient_infinity in state_infinity.items():
        for desc_zero, coefficient_zero in state_zero.items():
            total += (
                coefficient_infinity
                * coefficient_zero
                * rho_primary_descendants(
                    desc_infinity,
                    desc_one,
                    desc_zero,
                    h_infinity,
                    h_one,
                    h_zero,
                    central_charge,
                )
            )
    return complex(total)


@lru_cache(maxsize=None)
def rho_primary_descendants(
    desc_infinity: Descendant,
    desc_one: Descendant,
    desc_zero: Descendant,
    h_infinity: complex,
    h_one: complex,
    h_zero: complex,
    central_charge: complex,
) -> complex:
    r"""Return the normalized plane descendant three-point function at ``z=1``.

    The result is

    ``rho(L_-A h_inf, L_-B h_one, L_-C h_zero | 1)``,

    normalized to one for three primaries.  The recursion is the exact
    Virasoro Ward recursion of Appendix A of arXiv:1703.09805.
    """

    desc_infinity = tuple(desc_infinity)
    desc_one = tuple(desc_one)
    desc_zero = tuple(desc_zero)
    h_infinity = complex(h_infinity)
    h_one = complex(h_one)
    h_zero = complex(h_zero)
    central_charge = complex(central_charge)

    if not desc_one:
        return complex(
            sum(
                coefficient
                for _, coefficient in _two_leg_matrix_element_series(
                    desc_infinity,
                    desc_zero,
                    h_infinity,
                    h_one,
                    h_zero,
                    central_charge,
                )
            )
        )

    mode = int(desc_one[0])
    rest_one = desc_one[1:]
    if mode <= 0:
        raise ValueError("descendant labels must contain positive Virasoro mode numbers")

    if mode == 1:
        exponent = (
            h_infinity
            + descendant_level(desc_infinity)
            - h_one
            - descendant_level(rest_one)
            - h_zero
            - descendant_level(desc_zero)
        )
        return complex(exponent) * rho_primary_descendants(
            desc_infinity,
            rest_one,
            desc_zero,
            h_infinity,
            h_one,
            h_zero,
            central_charge,
        )

    total = 0.0j
    max_m = max(
        0,
        descendant_level(desc_infinity) - mode,
        descendant_level(desc_zero) + 1,
    )
    for m_value in range(max_m + 1):
        coefficient = math.comb(mode - 2 + m_value, mode - 2)
        state_infinity = _rho_apply_mode(
            mode + m_value,
            desc_infinity,
            weight=h_infinity,
            central_charge=central_charge,
        )
        if state_infinity:
            total += coefficient * _rho_state_sum(
                state_infinity,
                rest_one,
                {desc_zero: 1.0 + 0.0j},
                h_infinity=h_infinity,
                h_one=h_one,
                h_zero=h_zero,
                central_charge=central_charge,
            )

        state_zero = _rho_apply_mode(
            m_value - 1,
            desc_zero,
            weight=h_zero,
            central_charge=central_charge,
        )
        if state_zero:
            total += coefficient * ((-1) ** mode) * _rho_state_sum(
                {desc_infinity: 1.0 + 0.0j},
                rest_one,
                state_zero,
                h_infinity=h_infinity,
                h_one=h_one,
                h_zero=h_zero,
                central_charge=central_charge,
            )
    return complex(total)


@dataclass(frozen=True, order=True)
class EdgeEndpoint:
    """One graph half-edge attached to a specified three-point slot."""

    vertex: int
    slot: int

    def __post_init__(self) -> None:
        if int(self.vertex) < 0:
            raise ValueError("vertex indices must be non-negative")
        if int(self.slot) not in (INFINITY_SLOT, ONE_SLOT, ZERO_SLOT):
            raise ValueError("slot must be INFINITY_SLOT, ONE_SLOT, or ZERO_SLOT")


@dataclass(frozen=True)
class PlumbingEdge:
    """One sewn internal edge with two graph endpoints."""

    name: str
    endpoints: tuple[EdgeEndpoint, EdgeEndpoint]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("edge names must be nonempty")
        if len(self.endpoints) != 2:
            raise ValueError("each plumbing edge must have exactly two endpoints")
        if self.endpoints[0] == self.endpoints[1]:
            raise ValueError("an edge cannot use the same vertex slot twice")


@dataclass(frozen=True)
class TrivalentPlumbingGraph:
    """Closed connected trivalent plumbing graph with ordered local slots."""

    name: str
    vertex_count: int
    edges: tuple[PlumbingEdge, ...]

    def __post_init__(self) -> None:
        if int(self.vertex_count) <= 0:
            raise ValueError("vertex_count must be positive")
        if not self.edges:
            raise ValueError("a closed plumbing graph must contain at least one edge")
        names = [edge.name for edge in self.edges]
        if len(set(names)) != len(names):
            raise ValueError("plumbing edge names must be unique")

        endpoints = [endpoint for edge in self.edges for endpoint in edge.endpoints]
        if len(set(endpoints)) != len(endpoints):
            raise ValueError("each vertex slot must belong to exactly one edge")
        expected = {
            EdgeEndpoint(vertex, slot)
            for vertex in range(int(self.vertex_count))
            for slot in (INFINITY_SLOT, ONE_SLOT, ZERO_SLOT)
        }
        if set(endpoints) != expected:
            missing = sorted(expected - set(endpoints))
            extra = sorted(set(endpoints) - expected)
            raise ValueError(f"graph is not closed trivalent: missing={missing}, extra={extra}")

        parent = list(range(int(self.vertex_count)))

        def find(vertex: int) -> int:
            while parent[vertex] != vertex:
                parent[vertex] = parent[parent[vertex]]
                vertex = parent[vertex]
            return vertex

        def union(left: int, right: int) -> None:
            left_root = find(left)
            right_root = find(right)
            if left_root != right_root:
                parent[right_root] = left_root

        for edge in self.edges:
            union(edge.endpoints[0].vertex, edge.endpoints[1].vertex)
        if len({find(vertex) for vertex in range(int(self.vertex_count))}) != 1:
            raise ValueError("plumbing graph must be connected")

    @property
    def genus(self) -> int:
        """Return the first Betti number ``E - V + 1``."""

        return len(self.edges) - int(self.vertex_count) + 1


@dataclass(frozen=True)
class PlumbingBlockResult:
    """One total-degree truncated direct plumbing-graph block."""

    graph_name: str
    genus: int
    central_charge: complex
    edge_weights: tuple[complex, ...]
    q_values: tuple[complex, ...]
    max_total_level: int
    value: complex
    coefficient_by_levels: dict[tuple[int, ...], complex]
    contribution_by_levels: dict[tuple[int, ...], complex]
    max_gram_condition_number: float


def genus2_theta_graph() -> TrivalentPlumbingGraph:
    """Return the two-vertex, three-edge theta plumbing graph."""

    return TrivalentPlumbingGraph(
        name="genus2_theta",
        vertex_count=2,
        edges=(
            PlumbingEdge("q1", (EdgeEndpoint(0, INFINITY_SLOT), EdgeEndpoint(1, INFINITY_SLOT))),
            PlumbingEdge("q2", (EdgeEndpoint(0, ONE_SLOT), EdgeEndpoint(1, ONE_SLOT))),
            PlumbingEdge("q3", (EdgeEndpoint(0, ZERO_SLOT), EdgeEndpoint(1, ZERO_SLOT))),
        ),
    )


def genus2_glasses_graph() -> TrivalentPlumbingGraph:
    """Return the two-loop glasses graph with its bridge in the middle slot."""

    return TrivalentPlumbingGraph(
        name="genus2_glasses",
        vertex_count=2,
        edges=(
            PlumbingEdge("q_left", (EdgeEndpoint(0, INFINITY_SLOT), EdgeEndpoint(0, ZERO_SLOT))),
            PlumbingEdge("q_right", (EdgeEndpoint(1, INFINITY_SLOT), EdgeEndpoint(1, ZERO_SLOT))),
            PlumbingEdge("q_bridge", (EdgeEndpoint(0, ONE_SLOT), EdgeEndpoint(1, ONE_SLOT))),
        ),
    )


def genus3_tetrahedral_graph() -> TrivalentPlumbingGraph:
    """Return the K4 graph in the slot convention of the tetrahedral chart."""

    puncture_slots = (
        {1: ZERO_SLOT, 2: ONE_SLOT, 3: INFINITY_SLOT},
        {0: ZERO_SLOT, 2: ONE_SLOT, 3: INFINITY_SLOT},
        {0: ZERO_SLOT, 1: ONE_SLOT, 3: INFINITY_SLOT},
        {0: ZERO_SLOT, 1: ONE_SLOT, 2: INFINITY_SLOT},
    )
    edge_pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    edges = tuple(
        PlumbingEdge(
            f"q{left}{right}",
            (
                EdgeEndpoint(left, puncture_slots[left][right]),
                EdgeEndpoint(right, puncture_slots[right][left]),
            ),
        )
        for left, right in edge_pairs
    )
    return TrivalentPlumbingGraph(name="genus3_tetrahedral", vertex_count=4, edges=edges)


def genus3_whitehead_graph() -> TrivalentPlumbingGraph:
    r"""Return a non-tetrahedral graph obtained by moving the K4 ``q01`` seam.

    Cutting the ``q01`` seam exposes the four legs ``q02,q03,q12,q13``.
    The alternate pairing ``(q02,q12)|(q03,q13)`` gives two pairs of
    parallel edges, joined by the new ``q01`` seam and by ``q23``.  This is a
    genuine Whitehead move of the pants decomposition, not an edge relabeling
    or a change of homology marking.

    Vertex zero carries ``(q01,q12,q02)`` in infinity/one/zero slots, vertex
    one carries ``(q01,q13,q03)``, and vertices two and three retain the slot
    assignments of the corresponding tetrahedral pants.
    """

    return TrivalentPlumbingGraph(
        name="genus3_whitehead_q01",
        vertex_count=4,
        edges=(
            PlumbingEdge("q02", (EdgeEndpoint(0, ZERO_SLOT), EdgeEndpoint(2, ZERO_SLOT))),
            PlumbingEdge("q12", (EdgeEndpoint(0, ONE_SLOT), EdgeEndpoint(2, ONE_SLOT))),
            PlumbingEdge("q03", (EdgeEndpoint(1, ZERO_SLOT), EdgeEndpoint(3, ZERO_SLOT))),
            PlumbingEdge("q13", (EdgeEndpoint(1, ONE_SLOT), EdgeEndpoint(3, ONE_SLOT))),
            PlumbingEdge(
                "q01",
                (EdgeEndpoint(0, INFINITY_SLOT), EdgeEndpoint(1, INFINITY_SLOT)),
            ),
            PlumbingEdge(
                "q23",
                (EdgeEndpoint(2, INFINITY_SLOT), EdgeEndpoint(3, INFINITY_SLOT)),
            ),
        ),
    )


def _edge_values(
    graph: TrivalentPlumbingGraph,
    values: Sequence[complex] | Mapping[str, complex],
    *,
    label: str,
) -> tuple[complex, ...]:
    if isinstance(values, Mapping):
        missing = [edge.name for edge in graph.edges if edge.name not in values]
        extra = sorted(set(values) - {edge.name for edge in graph.edges})
        if missing or extra:
            raise ValueError(f"{label} keys do not match graph edges: missing={missing}, extra={extra}")
        return tuple(complex(values[edge.name]) for edge in graph.edges)
    if len(values) != len(graph.edges):
        raise ValueError(f"{label} must contain {len(graph.edges)} entries")
    return tuple(complex(value) for value in values)


def _fixed_total_compositions(total: int, length: int) -> tuple[tuple[int, ...], ...]:
    if length == 0:
        return ((),) if total == 0 else ()
    out: list[tuple[int, ...]] = []
    for first in range(total + 1):
        for tail in _fixed_total_compositions(total - first, length - 1):
            out.append((first,) + tail)
    return tuple(out)


def direct_plumbing_graph_block(
    graph: TrivalentPlumbingGraph,
    *,
    central_charge: complex,
    edge_weights: Sequence[complex] | Mapping[str, complex],
    q_values: Sequence[complex] | Mapping[str, complex],
    max_total_level: int,
) -> PlumbingBlockResult:
    r"""Evaluate the direct descendant sum for a closed plumbing graph.

    For edge levels ``n_e``, the coefficient is

    ``prod_e (G_e[n_e]^-1) * prod_v rho_v``,

    with all descendant indices contracted according to the graph.  The
    truncation is by total plumbing degree ``sum_e n_e <= max_total_level``.
    """

    max_total_level = int(max_total_level)
    if max_total_level < 0:
        raise ValueError("max_total_level must be non-negative")
    central_charge = complex(central_charge)
    weights = _edge_values(graph, edge_weights, label="edge_weights")
    q_tuple = _edge_values(graph, q_values, label="q_values")

    gram_data: dict[tuple[int, int], tuple[tuple[Descendant, ...], np.ndarray]] = {}
    max_condition = 1.0
    for edge_index, weight in enumerate(weights):
        for level in range(max_total_level + 1):
            basis, inverse, condition = inverse_verma_gram_matrix(level, weight, central_charge)
            gram_data[(edge_index, level)] = (basis, inverse)
            max_condition = max(max_condition, condition)

    endpoint_weights: dict[EdgeEndpoint, complex] = {}
    for edge_index, edge in enumerate(graph.edges):
        for endpoint in edge.endpoints:
            endpoint_weights[endpoint] = weights[edge_index]

    coefficient_by_levels: dict[tuple[int, ...], complex] = {}
    contribution_by_levels: dict[tuple[int, ...], complex] = {}
    total_value = 0.0j

    for total_level in range(max_total_level + 1):
        for levels in _fixed_total_compositions(total_level, len(graph.edges)):
            endpoint_states: dict[EdgeEndpoint, Descendant] = {}
            coefficient = 0.0j

            def contract_edge(edge_index: int, metric_product: complex) -> None:
                nonlocal coefficient
                if edge_index == len(graph.edges):
                    vertex_product = 1.0 + 0.0j
                    for vertex in range(graph.vertex_count):
                        endpoints = tuple(EdgeEndpoint(vertex, slot) for slot in range(3))
                        vertex_product *= rho_primary_descendants(
                            endpoint_states[endpoints[INFINITY_SLOT]],
                            endpoint_states[endpoints[ONE_SLOT]],
                            endpoint_states[endpoints[ZERO_SLOT]],
                            endpoint_weights[endpoints[INFINITY_SLOT]],
                            endpoint_weights[endpoints[ONE_SLOT]],
                            endpoint_weights[endpoints[ZERO_SLOT]],
                            central_charge,
                        )
                        if vertex_product == 0:
                            break
                    coefficient += metric_product * vertex_product
                    return

                edge = graph.edges[edge_index]
                basis, inverse = gram_data[(edge_index, levels[edge_index])]
                for left_index, left_descendant in enumerate(basis):
                    endpoint_states[edge.endpoints[0]] = left_descendant
                    for right_index, right_descendant in enumerate(basis):
                        inverse_element = inverse[left_index, right_index]
                        if inverse_element == 0:
                            continue
                        endpoint_states[edge.endpoints[1]] = right_descendant
                        contract_edge(edge_index + 1, metric_product * inverse_element)

            contract_edge(0, 1.0 + 0.0j)
            q_factor = math.prod(
                q_value**level
                for q_value, level in zip(q_tuple, levels)
            )
            contribution = coefficient * q_factor
            coefficient_by_levels[levels] = complex(coefficient)
            contribution_by_levels[levels] = complex(contribution)
            total_value += contribution

    return PlumbingBlockResult(
        graph_name=graph.name,
        genus=graph.genus,
        central_charge=central_charge,
        edge_weights=weights,
        q_values=q_tuple,
        max_total_level=max_total_level,
        value=complex(total_value),
        coefficient_by_levels=coefficient_by_levels,
        contribution_by_levels=contribution_by_levels,
        max_gram_condition_number=float(max_condition),
    )
