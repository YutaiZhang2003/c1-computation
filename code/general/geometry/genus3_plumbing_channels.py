#!/usr/bin/env python3
"""The five marked trivalent genus-three plumbing channels.

The channel conventions are those fixed in
``plumbing/genus3/notes/genus3_period_matrix_note.tex``.  Every channel
has four three-punctured spheres, six ordered plumbing edges, a deterministic
spanning tree, and three chord cycles.  The endpoint slot of every named
plumbing parameter is part of the public channel contract, rather than being
inferred later by a CFT or period-matrix consumer.

Besides the graph data used by the Virasoro recursion, this module constructs
three exact Schottky generators in a common root-sphere coordinate.  A
generator follows the tree from the root to a chord, crosses the chord, and
returns along the tree.  This works uniformly for ordinary edges, parallel
edges, and self-plumbing loops.
"""

from __future__ import annotations

from c1_paths import PACKAGE_ROOT, data_root, package_path, module_path

import itertools
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable, Mapping, Sequence

import mpmath as mp

try:
    from plumbing_algorithms import (
        GeneratorData,
        IDENTITY,
        INF,
        Mobius,
        mobius_fixed_points,
        plumbing_transition,
    )
    from virasoro_plumbing_graph import (
        INFINITY_SLOT,
        ONE_SLOT,
        ZERO_SLOT,
        EdgeEndpoint,
        PlumbingEdge,
        TrivalentPlumbingGraph,
    )
except ImportError:  # pragma: no cover - supports package-style execution
    from plumbing_algorithms import (
        GeneratorData,
        IDENTITY,
        INF,
        Mobius,
        mobius_fixed_points,
        plumbing_transition,
    )
    from virasoro_plumbing_graph import (
        INFINITY_SLOT,
        ONE_SLOT,
        ZERO_SLOT,
        EdgeEndpoint,
        PlumbingEdge,
        TrivalentPlumbingGraph,
    )


GENUS3_CHANNEL_NAMES = (
    "one-tadpole-double-triangle",
    "opposite-double-edge-cycle",
    "tetrahedron",
    "three-tadpole-star",
    "two-tadpoles-double-bridge",
)

# This is the public plumbing-parameter contract.  Geometry, CFT, cluster,
# and command-line code must name parameters with these keys before converting
# them to the internal canonical tuple order.
GENUS3_CHANNEL_EDGE_NAMES = {
    "one-tadpole-double-triangle": (
        "q02",
        "q03_1",
        "q03_2",
        "q11",
        "q12",
        "q23",
    ),
    "opposite-double-edge-cycle": (
        "q02",
        "q03_1",
        "q03_2",
        "q12_1",
        "q12_2",
        "q13",
    ),
    "tetrahedron": (
        "q01",
        "q02",
        "q03",
        "q12",
        "q13",
        "q23",
    ),
    "three-tadpole-star": (
        "q01",
        "q02",
        "q03",
        "q11",
        "q22",
        "q33",
    ),
    "two-tadpoles-double-bridge": (
        "q02",
        "q03_1",
        "q03_2",
        "q11",
        "q13",
        "q22",
    ),
}

# The endpoint-slot contract for every named plumbing parameter.  Each item is
#
#     q_name: ((source sphere, source puncture), (target sphere, target puncture))
#
# and the plumbing relation is u_source u_target = q_name.  Endpoint order is
# fixed as well because it enters the marked Schottky paths, even though the
# local plumbing equation itself is symmetric.  Together with
# GENUS3_CHANNEL_EDGE_NAMES this table is the single source of truth for
# passing q's to geometry and to the CCY (infinity, one, zero) trinion slots.
GENUS3_CHANNEL_Q_ENDPOINT_SLOTS = {
    "one-tadpole-double-triangle": {
        "q02": ((0, "zero"), (2, "zero")),
        "q03_1": ((0, "one"), (3, "zero")),
        "q03_2": ((0, "infty"), (3, "one")),
        "q11": ((1, "zero"), (1, "one")),
        "q12": ((1, "infty"), (2, "one")),
        "q23": ((2, "infty"), (3, "infty")),
    },
    "opposite-double-edge-cycle": {
        "q02": ((0, "zero"), (2, "zero")),
        "q03_1": ((0, "one"), (3, "zero")),
        "q03_2": ((0, "infty"), (3, "one")),
        "q12_1": ((1, "zero"), (2, "one")),
        "q12_2": ((1, "one"), (2, "infty")),
        "q13": ((1, "infty"), (3, "infty")),
    },
    "tetrahedron": {
        "q01": ((0, "zero"), (1, "zero")),
        "q02": ((0, "one"), (2, "zero")),
        "q03": ((0, "infty"), (3, "zero")),
        "q12": ((1, "one"), (2, "one")),
        "q13": ((1, "infty"), (3, "one")),
        "q23": ((2, "infty"), (3, "infty")),
    },
    "three-tadpole-star": {
        "q01": ((0, "zero"), (1, "zero")),
        "q02": ((0, "one"), (2, "zero")),
        "q03": ((0, "infty"), (3, "zero")),
        "q11": ((1, "one"), (1, "infty")),
        "q22": ((2, "one"), (2, "infty")),
        "q33": ((3, "one"), (3, "infty")),
    },
    "two-tadpoles-double-bridge": {
        "q02": ((0, "zero"), (2, "zero")),
        "q03_1": ((0, "one"), (3, "zero")),
        "q03_2": ((0, "infty"), (3, "one")),
        "q11": ((1, "zero"), (1, "one")),
        "q13": ((1, "infty"), (3, "infty")),
        "q22": ((2, "one"), (2, "infty")),
    },
}

# The six independent entries of a symmetric genus-three period matrix.
# With tau_e = log(q_e)/(2 pi i), the logarithmic plumbing degeneration is
#
#   vec(Omega_log) = S_G tau,
#   (S_G)_{(i,j),e} = C_{G,i,e} C_{G,j,e},
#
# in this component order.  Keeping the order public makes the slope matrix
# directly comparable with the six named plumbing parameters above.
GENUS3_SYMMETRIC_PERIOD_INDEX_ORDER = (
    (0, 0),
    (1, 1),
    (2, 2),
    (0, 1),
    (0, 2),
    (1, 2),
)

_PUNCTURE_SLOTS = (ZERO_SLOT, ONE_SLOT, INFINITY_SLOT)
_SLOT_PUNCTURES = {
    ZERO_SLOT: "zero",
    ONE_SLOT: "one",
    INFINITY_SLOT: "infty",
}


@dataclass(frozen=True)
class Genus3PlumbingChannel:
    """One fixed genus-three plumbing graph and marking."""

    name: str
    adjacency: tuple[tuple[int, ...], ...]
    graph: TrivalentPlumbingGraph
    tree_edges: tuple[int, ...]
    chord_edges: tuple[int, ...]
    cycle_matrix: tuple[tuple[int, ...], ...]

    @property
    def edge_names(self) -> tuple[str, ...]:
        return tuple(edge.name for edge in self.graph.edges)

    @property
    def genus(self) -> int:
        return self.graph.genus

    @property
    def vertex_edge_indices(self) -> tuple[tuple[int, int, int], ...]:
        """Return incident edge indices in infinity/one/zero slot order."""

        indices: list[list[int | None]] = [
            [None, None, None] for _ in range(self.graph.vertex_count)
        ]
        for edge_index, edge in enumerate(self.graph.edges):
            for endpoint in edge.endpoints:
                indices[endpoint.vertex][endpoint.slot] = edge_index
        if any(item is None for vertex in indices for item in vertex):
            raise RuntimeError(f"channel {self.name!r} is missing a vertex slot")
        return tuple(
            tuple(int(item) for item in vertex)
            for vertex in indices
        )

    @property
    def vertex_q_names(self) -> tuple[tuple[str, str, str], ...]:
        """Return named q's at each vertex in CCY infinity/one/zero order."""

        return tuple(
            tuple(self.edge_names[edge_index] for edge_index in vertex)
            for vertex in self.vertex_edge_indices
        )


def _upper_signature(matrix: Sequence[Sequence[int]]) -> tuple[int, ...]:
    return tuple(
        int(matrix[left][right])
        for left in range(len(matrix))
        for right in range(left, len(matrix))
    )


def _permuted_matrix(
    matrix: Sequence[Sequence[int]],
    permutation: Sequence[int],
) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(int(matrix[permutation[left]][permutation[right]]) for right in range(4))
        for left in range(4)
    )


def _canonical_signature(matrix: Sequence[Sequence[int]]) -> tuple[int, ...]:
    return min(
        _upper_signature(_permuted_matrix(matrix, permutation))
        for permutation in itertools.permutations(range(4))
    )


def _weak_compositions(total: int, length: int) -> Iterable[tuple[int, ...]]:
    if length == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for tail in _weak_compositions(total - first, length - 1):
            yield (first,) + tail


def _is_connected(matrix: Sequence[Sequence[int]]) -> bool:
    seen = {0}
    frontier = [0]
    while frontier:
        vertex = frontier.pop()
        for neighbor, multiplicity in enumerate(matrix[vertex]):
            if neighbor != vertex and multiplicity and neighbor not in seen:
                seen.add(neighbor)
                frontier.append(neighbor)
    return len(seen) == len(matrix)


@lru_cache(maxsize=1)
def enumerate_cubic_adjacency_matrices() -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Enumerate the five connected cubic multigraphs on four vertices."""

    pairs = tuple(
        (left, right)
        for left in range(4)
        for right in range(left, 4)
    )
    representatives: dict[tuple[int, ...], tuple[tuple[int, ...], ...]] = {}
    for multiplicities in _weak_compositions(6, len(pairs)):
        degrees = [0, 0, 0, 0]
        matrix = [[0] * 4 for _ in range(4)]
        for multiplicity, (left, right) in zip(multiplicities, pairs):
            matrix[left][right] = multiplicity
            matrix[right][left] = multiplicity
            if left == right:
                degrees[left] += 2 * multiplicity
            else:
                degrees[left] += multiplicity
                degrees[right] += multiplicity
        if degrees != [3, 3, 3, 3] or not _is_connected(matrix):
            continue
        matrix_tuple = tuple(tuple(row) for row in matrix)
        signature = _canonical_signature(matrix_tuple)
        if signature == _upper_signature(matrix_tuple):
            representatives[signature] = matrix_tuple
    return tuple(representatives[key] for key in sorted(representatives))


_CHANNEL_NAMES_BY_SIGNATURE = {
    (0, 0, 1, 2, 0, 2, 1, 0, 0, 0): "opposite-double-edge-cycle",
    (0, 0, 1, 2, 1, 0, 1, 1, 0, 0): "two-tadpoles-double-bridge",
    (0, 0, 1, 2, 1, 1, 0, 0, 1, 0): "one-tadpole-double-triangle",
    (0, 1, 1, 1, 0, 1, 1, 0, 1, 0): "tetrahedron",
    (0, 1, 1, 1, 1, 0, 0, 1, 0, 1): "three-tadpole-star",
}


def _graph_name(channel_name: str) -> str:
    if channel_name == "tetrahedron":
        # Preserve equality with the original public K4 graph constructor.
        return "genus3_tetrahedral"
    return "genus3_" + channel_name.replace("-", "_")


def _edge_list(
    matrix: Sequence[Sequence[int]],
    *,
    channel_name: str,
    graph_name: str,
) -> tuple[PlumbingEdge, ...]:
    try:
        slot_contract = GENUS3_CHANNEL_Q_ENDPOINT_SLOTS[channel_name]
    except KeyError as exc:  # pragma: no cover - protects the public table
        raise RuntimeError(f"missing q-slot contract for {channel_name!r}") from exc
    puncture_slots = {puncture: slot for slot, puncture in _SLOT_PUNCTURES.items()}
    used_slots: list[list[int]] = [[], [], [], []]
    edges: list[PlumbingEdge] = []
    for left in range(4):
        for right in range(left, 4):
            multiplicity = int(matrix[left][right])
            for copy in range(multiplicity):
                suffix = f"_{copy + 1}" if multiplicity > 1 else ""
                name = f"q{left}{right}{suffix}"
                try:
                    endpoint_contract = slot_contract[name]
                except KeyError as exc:
                    raise RuntimeError(
                        f"{channel_name!r} has no endpoint-slot contract for {name!r}"
                    ) from exc
                endpoint_vertices = tuple(
                    int(vertex) for vertex, _ in endpoint_contract
                )
                if endpoint_vertices != (left, right):
                    raise RuntimeError(
                        f"{channel_name!r} endpoint vertices for {name!r} are "
                        f"{endpoint_vertices}, expected {(left, right)}"
                    )
                endpoints: list[EdgeEndpoint] = []
                for vertex, puncture in endpoint_contract:
                    try:
                        slot = puncture_slots[puncture]
                    except KeyError as exc:
                        raise RuntimeError(
                            f"{channel_name!r} uses invalid puncture {puncture!r} "
                            f"for {name!r}"
                        ) from exc
                    used_slots[vertex].append(slot)
                    endpoints.append(EdgeEndpoint(vertex, slot))
                edges.append(
                    PlumbingEdge(
                        name,
                        tuple(endpoints),
                    )
                )
    generated_names = tuple(edge.name for edge in edges)
    contract_names = tuple(slot_contract)
    if generated_names != contract_names:
        raise RuntimeError(
            f"{channel_name!r} q-slot keys {contract_names} do not match "
            f"the generated edge order {generated_names}"
        )
    expected_slots = sorted(_PUNCTURE_SLOTS)
    if (
        len(edges) != 6
        or any(sorted(vertex_slots) != expected_slots for vertex_slots in used_slots)
    ):
        raise RuntimeError(f"{graph_name!r} is not a six-edge trivalent graph")
    return tuple(edges)


def _spanning_tree(
    edges: Sequence[PlumbingEdge],
    vertex_count: int,
) -> tuple[int, ...]:
    parent = list(range(vertex_count))

    def root(vertex: int) -> int:
        while parent[vertex] != vertex:
            parent[vertex] = parent[parent[vertex]]
            vertex = parent[vertex]
        return vertex

    selected: list[int] = []
    for edge_index, edge in enumerate(edges):
        left, right = (endpoint.vertex for endpoint in edge.endpoints)
        if left == right:
            continue
        left_root = root(left)
        right_root = root(right)
        if left_root == right_root:
            continue
        parent[right_root] = left_root
        selected.append(edge_index)
    if len(selected) != vertex_count - 1:
        raise RuntimeError("genus-three plumbing graph is not connected")
    return tuple(selected)


def _tree_path(
    edges: Sequence[PlumbingEdge],
    tree_edges: Sequence[int],
    start: int,
    stop: int,
) -> tuple[tuple[int, int, int], ...]:
    """Return ``(edge, source endpoint, target endpoint)`` along the tree."""

    if start == stop:
        return ()
    adjacency: dict[int, list[tuple[int, int, int]]] = {
        vertex: [] for vertex in range(4)
    }
    for edge_index in tree_edges:
        left, right = (endpoint.vertex for endpoint in edges[edge_index].endpoints)
        adjacency[left].append((right, edge_index, 0))
        adjacency[right].append((left, edge_index, 1))

    parent: dict[int, tuple[int, int, int] | None] = {start: None}
    frontier = [start]
    while frontier:
        vertex = frontier.pop(0)
        if vertex == stop:
            break
        for neighbor, edge_index, source_endpoint in adjacency[vertex]:
            if neighbor not in parent:
                parent[neighbor] = (vertex, edge_index, source_endpoint)
                frontier.append(neighbor)
    if stop not in parent:
        raise RuntimeError("tree path was not found")

    reverse_path: list[tuple[int, int, int]] = []
    vertex = stop
    while vertex != start:
        previous, edge_index, source_endpoint = parent[vertex]  # type: ignore[misc]
        reverse_path.append((edge_index, source_endpoint, 1 - source_endpoint))
        vertex = previous
    return tuple(reversed(reverse_path))


def _marked_cycles(
    edges: Sequence[PlumbingEdge],
    tree_edges: Sequence[int],
) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]:
    chords = tuple(index for index in range(len(edges)) if index not in tree_edges)
    rows: list[tuple[int, ...]] = []
    for chord_index in chords:
        chord = edges[chord_index]
        start = chord.endpoints[0].vertex
        stop = chord.endpoints[1].vertex
        row = [0] * len(edges)
        row[chord_index] = 1
        for edge_index, source_endpoint, _ in _tree_path(
            edges,
            tree_edges,
            stop,
            start,
        ):
            row[edge_index] += 1 if source_endpoint == 0 else -1
        rows.append(tuple(row))
    return chords, tuple(rows)


@lru_cache(maxsize=1)
def enumerate_genus3_channels() -> tuple[Genus3PlumbingChannel, ...]:
    """Return all five channels in the order used by the genus-three note."""

    channels_by_name: dict[str, Genus3PlumbingChannel] = {}
    for matrix in enumerate_cubic_adjacency_matrices():
        signature = _upper_signature(matrix)
        try:
            name = _CHANNEL_NAMES_BY_SIGNATURE[signature]
        except KeyError as exc:  # pragma: no cover - protects classification
            raise RuntimeError(f"unclassified genus-three graph {signature}") from exc
        graph_name = _graph_name(name)
        edges = _edge_list(
            matrix,
            channel_name=name,
            graph_name=graph_name,
        )
        graph = TrivalentPlumbingGraph(
            name=graph_name,
            vertex_count=4,
            edges=edges,
        )
        tree_edges = _spanning_tree(edges, graph.vertex_count)
        chord_edges, cycle_matrix = _marked_cycles(edges, tree_edges)
        channel = Genus3PlumbingChannel(
            name=name,
            adjacency=matrix,
            graph=graph,
            tree_edges=tree_edges,
            chord_edges=chord_edges,
            cycle_matrix=cycle_matrix,
        )
        if channel.genus != 3 or len(channel.chord_edges) != 3:
            raise RuntimeError(f"invalid genus-three marking for {name!r}")
        if channel.edge_names != GENUS3_CHANNEL_EDGE_NAMES[name]:
            raise RuntimeError(
                f"generated edge order for {name!r} disagrees with the "
                "explicit genus-three plumbing contract"
            )
        channels_by_name[name] = channel
    if set(channels_by_name) != set(GENUS3_CHANNEL_NAMES):
        raise RuntimeError("genus-three channel enumeration did not return the fixed five")
    return tuple(channels_by_name[name] for name in GENUS3_CHANNEL_NAMES)


def genus3_channel_by_name(name: str) -> Genus3PlumbingChannel:
    """Resolve a note channel name, graph name, or ``k4`` alias."""

    normalized = str(name).strip().lower().replace("_", "-")
    if normalized in {"k4", "tetrahedral", "genus3-tetrahedral"}:
        normalized = "tetrahedron"
    for channel in enumerate_genus3_channels():
        if normalized in {
            channel.name,
            channel.graph.name.lower().replace("_", "-"),
        }:
            return channel
    available = ", ".join(GENUS3_CHANNEL_NAMES)
    raise ValueError(f"unknown genus-three channel {name!r}; choose one of {available}")


def genus3_channel_for_graph(
    graph: TrivalentPlumbingGraph,
) -> Genus3PlumbingChannel | None:
    """Return the registered channel equal to ``graph``, if any."""

    for channel in enumerate_genus3_channels():
        if graph == channel.graph:
            return channel
    return None


def genus3_channel_leading_period_slope_matrix(
    channel: str | Genus3PlumbingChannel,
) -> tuple[tuple[int, ...], ...]:
    """Return the exact logarithmic period slope in named edge order.

    Rows use :data:`GENUS3_SYMMETRIC_PERIOD_INDEX_ORDER`; columns use
    ``channel.edge_names``.  Thus this integer matrix is the derivative of
    the symmetric period vector with respect to
    ``tau_e = log(q_e)/(2*pi*i)``.  The derivative with respect to ``log q_e``
    is the returned matrix divided by ``2*pi*i``.

    The matrix need not have full rank.  Separating fixtures have zero
    logarithmic columns, while further tropical Torelli degeneracies can make
    distinct nonseparating edge lengths indistinguishable at logarithmic
    order.  Their plumbing parameters enter the finite and higher-order
    period map instead.
    """

    resolved = (
        genus3_channel_by_name(channel)
        if isinstance(channel, str)
        else channel
    )
    cycles = resolved.cycle_matrix
    return tuple(
        tuple(
            int(cycles[left][edge] * cycles[right][edge])
            for edge in range(len(resolved.graph.edges))
        )
        for left, right in GENUS3_SYMMETRIC_PERIOD_INDEX_ORDER
    )


def genus3_channel_q_values(
    channel: str | Genus3PlumbingChannel,
    values: Sequence[complex] | Mapping[str, complex],
    *,
    label: str = "q_values",
) -> tuple[complex, ...]:
    """Validate named values and return the internal fixed edge order.

    A six-entry sequence remains supported for backward compatibility and is
    interpreted in :data:`GENUS3_CHANNEL_EDGE_NAMES` order.
    """

    resolved = (
        genus3_channel_by_name(channel)
        if isinstance(channel, str)
        else channel
    )
    if isinstance(values, Mapping):
        missing = [name for name in resolved.edge_names if name not in values]
        extra = sorted(set(values) - set(resolved.edge_names))
        if missing or extra:
            raise ValueError(
                f"{label} keys do not match {resolved.name} edges: "
                f"missing={missing}, extra={extra}"
            )
        result = tuple(complex(values[name]) for name in resolved.edge_names)
    else:
        if len(values) != 6:
            raise ValueError(f"{label} must contain six entries")
        result = tuple(complex(value) for value in values)
    return result


def _validate_q_values(
    channel: Genus3PlumbingChannel,
    q_values: Sequence[complex] | Mapping[str, complex],
) -> tuple[complex, ...]:
    result = genus3_channel_q_values(channel, q_values)
    for name, value in zip(channel.edge_names, result):
        if (
            not math.isfinite(value.real)
            or not math.isfinite(value.imag)
            or not 0.0 < abs(value) < 1.0
        ):
            raise ValueError(f"{name} must be finite and satisfy 0 < |{name}| < 1")
    return result


def genus3_channel_transition_maps(
    channel: str | Genus3PlumbingChannel,
    q_values: Sequence[complex] | Mapping[str, complex],
) -> tuple[tuple[Mobius, Mobius], ...]:
    """Return endpoint-0 to endpoint-1 maps and their inverses."""

    resolved = (
        genus3_channel_by_name(channel)
        if isinstance(channel, str)
        else channel
    )
    q_tuple = _validate_q_values(resolved, q_values)
    maps: list[tuple[Mobius, Mobius]] = []
    for edge, q_value in zip(resolved.graph.edges, q_tuple):
        source, target = edge.endpoints
        forward = plumbing_transition(
            _SLOT_PUNCTURES[source.slot],
            _SLOT_PUNCTURES[target.slot],
            q_value,
        )
        maps.append((forward, forward.inv()))
    return tuple(maps)


def _compose_edge_path(
    transition_maps: Sequence[tuple[Mobius, Mobius]],
    path: Sequence[tuple[int, int, int]],
) -> Mobius:
    result = IDENTITY
    for edge_index, source_endpoint, _ in path:
        result = transition_maps[edge_index][source_endpoint].compose(result)
    return result


def _fixed_point_multiplier(
    transform: Mobius,
    point: complex | None,
) -> complex:
    if point is INF:
        if transform.c != 0 or transform.a == 0:
            raise ValueError("infinity is not a regular fixed point")
        return transform.d / transform.a
    return transform.deriv(point)


def _mp_matrix_multiply(left: Any, right: Any) -> Any:
    return tuple(
        tuple(
            sum(left[row][inner] * right[inner][column] for inner in range(2))
            for column in range(2)
        )
        for row in range(2)
    )


def _mp_projective_inverse(matrix: Any) -> Any:
    """Return a projectively equivalent inverse without dividing by det."""

    return (
        (matrix[1][1], -matrix[0][1]),
        (-matrix[1][0], matrix[0][0]),
    )


def _mp_local_coordinate_matrix(puncture: str) -> Any:
    zero = mp.mpc(0)
    one = mp.mpc(1)
    if puncture == "zero":
        return ((one, zero), (zero, one))
    if puncture == "one":
        return ((one, -one), (zero, one))
    if puncture == "infty":
        return ((zero, one), (one, zero))
    raise ValueError(f"unknown puncture {puncture!r}")


def _stable_path_multiplier(
    channel: Genus3PlumbingChannel,
    q_values: Sequence[complex],
    path: Sequence[tuple[int, int, int]],
    *,
    dps: int = 80,
) -> complex:
    """Return a deep-cusp-stable attracting eigenvalue ratio.

    A fixed-point derivative in binary64 loses the small projective
    determinant when ``a*d`` and ``b*c`` nearly cancel.  Rebuilding the
    elementary plumbing maps at high precision and evaluating
    ``det(M)/lambda_large**2`` avoids that cancellation as well as the
    cancellation in the small quadratic root.
    """

    with mp.workdps(int(dps)):
        transition_maps: list[tuple[Any, Any]] = []
        for edge, q_value in zip(channel.graph.edges, q_values):
            source, target = edge.endpoints
            source_map = _mp_local_coordinate_matrix(
                _SLOT_PUNCTURES[source.slot]
            )
            target_inverse = _mp_projective_inverse(
                _mp_local_coordinate_matrix(_SLOT_PUNCTURES[target.slot])
            )
            q_mp = mp.mpc(float(q_value.real), float(q_value.imag))
            q_over_u = ((mp.mpc(0), q_mp), (mp.mpc(1), mp.mpc(0)))
            forward = _mp_matrix_multiply(
                target_inverse,
                _mp_matrix_multiply(q_over_u, source_map),
            )
            transition_maps.append(
                (forward, _mp_projective_inverse(forward))
            )

        matrix: Any = (
            (mp.mpc(1), mp.mpc(0)),
            (mp.mpc(0), mp.mpc(1)),
        )
        for edge_index, source_endpoint, _ in path:
            matrix = _mp_matrix_multiply(
                transition_maps[edge_index][source_endpoint],
                matrix,
            )

        trace = matrix[0][0] + matrix[1][1]
        determinant = (
            matrix[0][0] * matrix[1][1]
            - matrix[0][1] * matrix[1][0]
        )
        discriminant = mp.sqrt(trace * trace - 4 * determinant)
        candidates = (
            (trace + discriminant) / 2,
            (trace - discriminant) / 2,
        )
        large = max(candidates, key=abs)
        if large == 0:
            raise ValueError("marked genus-three cycle has zero eigenvalue")
        multiplier_mp = determinant / (large * large)
        multiplier = complex(
            float(mp.re(multiplier_mp)),
            float(mp.im(multiplier_mp)),
        )
    if multiplier == 0.0:
        raise ValueError(
            "marked genus-three multiplier lies below binary64 range"
        )
    return multiplier


def _fixed_point_eigenvalue_magnitude(
    transform: Mobius,
    point: complex | None,
) -> float:
    """Return the matrix eigenvalue modulus associated with a fixed point."""

    if point is INF:
        return float(abs(transform.a))
    return float(abs(transform.c * point + transform.d))


def _generator_preserving_cycle_orientation(
    transform: Mobius,
    *,
    stable_multiplier: complex | None = None,
) -> GeneratorData:
    """Attach fixed-point data without replacing the marked cycle by its inverse."""

    first, second = mobius_fixed_points(transform)
    first_eigenvalue = _fixed_point_eigenvalue_magnitude(transform, first)
    second_eigenvalue = _fixed_point_eigenvalue_magnitude(transform, second)
    if first_eigenvalue >= second_eigenvalue:
        attracting, repelling = first, second
    else:
        attracting, repelling = second, first
    multiplier = (
        complex(stable_multiplier)
        if stable_multiplier is not None
        else _fixed_point_multiplier(transform, attracting)
    )
    if (
        not math.isfinite(multiplier.real)
        or not math.isfinite(multiplier.imag)
        or not 0.0 < abs(multiplier) < 1.0
    ):
        raise ValueError(
            "marked genus-three cycle did not produce a loxodromic "
            f"generator with |k|<1: |k|={abs(multiplier):.6g}"
        )
    return GeneratorData(
        gamma=transform,
        attracting=attracting,
        repelling=repelling,
        multiplier=multiplier,
    )


def generators_for_genus3_channel(
    channel: str | Genus3PlumbingChannel,
    q_values: Sequence[complex] | Mapping[str, complex],
    *,
    root_vertex: int = 0,
) -> tuple[GeneratorData, GeneratorData, GeneratorData]:
    """Construct the three marked Schottky generators in one root coordinate."""

    resolved = (
        genus3_channel_by_name(channel)
        if isinstance(channel, str)
        else channel
    )
    if not 0 <= int(root_vertex) < resolved.graph.vertex_count:
        raise ValueError("root_vertex is outside the genus-three graph")
    q_tuple = _validate_q_values(resolved, q_values)
    maps = genus3_channel_transition_maps(resolved, q_tuple)
    generators: list[GeneratorData] = []
    for chord_index in resolved.chord_edges:
        chord = resolved.graph.edges[chord_index]
        chord_start = chord.endpoints[0].vertex
        chord_stop = chord.endpoints[1].vertex
        path = (
            _tree_path(
                resolved.graph.edges,
                resolved.tree_edges,
                int(root_vertex),
                chord_start,
            )
            + ((chord_index, 0, 1),)
            + _tree_path(
                resolved.graph.edges,
                resolved.tree_edges,
                chord_stop,
                int(root_vertex),
            )
        )
        transform = _compose_edge_path(maps, path)
        generators.append(
            _generator_preserving_cycle_orientation(
                transform,
                stable_multiplier=_stable_path_multiplier(
                    resolved,
                    q_tuple,
                    path,
                ),
            )
        )
    if len(generators) != 3:
        raise RuntimeError(f"channel {resolved.name!r} did not produce rank three")
    return generators[0], generators[1], generators[2]


__all__ = [
    "GENUS3_CHANNEL_EDGE_NAMES",
    "GENUS3_CHANNEL_NAMES",
    "GENUS3_SYMMETRIC_PERIOD_INDEX_ORDER",
    "Genus3PlumbingChannel",
    "enumerate_cubic_adjacency_matrices",
    "enumerate_genus3_channels",
    "generators_for_genus3_channel",
    "genus3_channel_by_name",
    "genus3_channel_for_graph",
    "genus3_channel_leading_period_slope_matrix",
    "genus3_channel_q_values",
    "genus3_channel_transition_maps",
]
