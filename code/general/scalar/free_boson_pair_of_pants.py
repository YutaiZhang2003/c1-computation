#!/usr/bin/env python3
"""Direct rank-one Heisenberg-VOA sewing on trivalent plumbing graphs.

This module is a finite-level diagnostic, not a production free-boson
evaluator.  In genus three its generic endpoint/BPZ contraction disagrees
with the all-level Schottky product in several channel slot patterns.  Use
``free_boson_plumbing.genus3_free_boson_product`` for numerical quotients.

This module computes the free-boson oscillator partition function by the same
pair-of-pants sewing operation used for the CCY Virasoro blocks.  It does not
use Schottky primitive-word products internally.

The Heisenberg vacuum module has basis

    a_{-lambda_1} ... a_{-lambda_k} |0>,

labelled by integer partitions ``lambda``.  Its Gram matrix is diagonal with
norm ``z_lambda = product_n n^m_n m_n!``.  The pants coefficient is

    rho(lambda, mu, nu) = <lambda| Y(mu, 1) |nu>,

and is evaluated by Wick contractions of the current and its derivatives.
The sewing propagator is ``q^L0``: no ``-c/24`` shift is inserted.
"""

from __future__ import annotations

from c1_paths import PACKAGE_ROOT, data_root, package_path, module_path

import argparse
import math
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from typing import Iterable, Mapping, Sequence

try:
    from genus3_plumbing_channels import (
        GENUS3_CHANNEL_NAMES,
        Genus3PlumbingChannel,
        genus3_channel_by_name,
        genus3_channel_q_values,
    )
    from virasoro_plumbing_graph import (
        TrivalentPlumbingGraph,
        genus2_glasses_graph,
        genus2_theta_graph,
    )
except ImportError:  # pragma: no cover - supports package-style execution
    from genus3_plumbing_channels import (
        GENUS3_CHANNEL_NAMES,
        Genus3PlumbingChannel,
        genus3_channel_by_name,
        genus3_channel_q_values,
    )
    from virasoro_plumbing_graph import (
        TrivalentPlumbingGraph,
        genus2_glasses_graph,
        genus2_theta_graph,
    )


Partition = tuple[int, ...]


def parse_complex(value: str) -> complex:
    return complex(value.replace("i", "j"))


def format_complex(value: complex) -> str:
    value = complex(value)
    return f"{value.real:+.12e}{value.imag:+.12e}j"


@lru_cache(maxsize=None)
def integer_partitions(total: int, max_part: int | None = None) -> tuple[Partition, ...]:
    """Return integer partitions of ``total`` in nonincreasing order."""

    total = int(total)
    if total < 0:
        return ()
    if total == 0:
        return ((),)
    if max_part is None:
        max_part = total
    out: list[Partition] = []
    for part in range(min(int(max_part), total), 0, -1):
        for tail in integer_partitions(total - part, part):
            out.append((part,) + tail)
    return tuple(out)


def partition_level(partition: Partition) -> int:
    return int(sum(partition))


@lru_cache(maxsize=None)
def heisenberg_gram_norm(partition: Partition) -> int:
    r"""Return ``z_lambda = product_n n^m_n m_n!``."""

    norm = 1
    for mode, multiplicity in Counter(partition).items():
        if mode <= 0:
            raise ValueError("Heisenberg creation modes must be positive")
        norm *= (int(mode) ** int(multiplicity)) * math.factorial(int(multiplicity))
    return int(norm)


def _remove_index(values: tuple[int, ...], index: int) -> tuple[int, ...]:
    return values[:index] + values[index + 1 :]


def _bra_middle_contraction(bra_mode: int, middle_mode: int) -> int:
    r"""Return ``<0|a_bra D_middle J(1)|0>``."""

    if bra_mode < middle_mode:
        return 0
    return int(bra_mode * math.comb(bra_mode - 1, middle_mode - 1))


def _middle_ket_contraction(middle_mode: int, ket_mode: int) -> int:
    r"""Return ``<0|D_middle J(1) a_-ket|0>``."""

    sign = -1 if (middle_mode - 1) % 2 else 1
    return int(sign * ket_mode * math.comb(ket_mode + middle_mode - 1, middle_mode - 1))


def _bra_ket_contraction(bra_mode: int, ket_mode: int) -> int:
    return int(bra_mode) if bra_mode == ket_mode else 0


@lru_cache(maxsize=None)
def _heisenberg_pairing_sum(
    bra: tuple[int, ...],
    middle: tuple[int, ...],
    ket: tuple[int, ...],
) -> int:
    """Sum complete Wick pairings with no same-insertion contractions."""

    field_count = len(bra) + len(middle) + len(ket)
    if field_count == 0:
        return 1
    if field_count % 2:
        return 0

    total = 0
    if bra:
        mode = bra[0]
        rest_bra = bra[1:]
        for index, partner in enumerate(middle):
            contraction = _bra_middle_contraction(mode, partner)
            if contraction:
                total += contraction * _heisenberg_pairing_sum(
                    rest_bra,
                    _remove_index(middle, index),
                    ket,
                )
        for index, partner in enumerate(ket):
            contraction = _bra_ket_contraction(mode, partner)
            if contraction:
                total += contraction * _heisenberg_pairing_sum(
                    rest_bra,
                    middle,
                    _remove_index(ket, index),
                )
        return int(total)

    if middle:
        mode = middle[0]
        rest_middle = middle[1:]
        for index, partner in enumerate(ket):
            contraction = _middle_ket_contraction(mode, partner)
            if contraction:
                total += contraction * _heisenberg_pairing_sum(
                    bra,
                    rest_middle,
                    _remove_index(ket, index),
                )
        return int(total)

    return 0


@lru_cache(maxsize=None)
def heisenberg_three_point(bra: Partition, middle: Partition, ket: Partition) -> int:
    r"""Return ``<bra|Y(middle,1)|ket>`` in the Heisenberg vacuum module.

    For ``middle=(n_1,...,n_k)``, the vertex operator is the normal-ordered
    product of ``(n_i-1)`` derivatives of the current.  Wick contractions
    among fields belonging to the same insertion are excluded.
    """

    bra = tuple(int(mode) for mode in bra)
    middle = tuple(int(mode) for mode in middle)
    ket = tuple(int(mode) for mode in ket)
    if any(mode <= 0 for mode in bra + middle + ket):
        raise ValueError("Heisenberg partitions must contain positive modes")
    return _heisenberg_pairing_sum(bra, middle, ket)


@dataclass(frozen=True)
class HeisenbergPlumbingResult:
    channel: str
    edge_names: tuple[str, ...]
    q_values: tuple[complex, ...]
    max_total_level: int
    chiral_value: complex
    nonchiral_value: float
    level_contributions: dict[tuple[int, ...], complex]


def _states_by_level(max_total_level: int) -> tuple[tuple[Partition, ...], ...]:
    if max_total_level < 0:
        raise ValueError("max_total_level must be non-negative")
    return tuple(integer_partitions(level) for level in range(int(max_total_level) + 1))


@lru_cache(maxsize=None)
def _fixed_total_compositions(
    total: int,
    length: int,
) -> tuple[tuple[int, ...], ...]:
    if length == 0:
        return ((),) if total == 0 else ()
    return tuple(
        (first,) + tail
        for first in range(total + 1)
        for tail in _fixed_total_compositions(total - first, length - 1)
    )


def _graph_edge_values(
    graph: TrivalentPlumbingGraph,
    values: Sequence[complex] | Mapping[str, complex],
) -> tuple[complex, ...]:
    edge_names = tuple(edge.name for edge in graph.edges)
    if isinstance(values, Mapping):
        missing = [name for name in edge_names if name not in values]
        extra = sorted(set(values) - set(edge_names))
        if missing or extra:
            raise ValueError(
                f"q_values keys do not match graph edges: missing={missing}, extra={extra}"
            )
        q_values = tuple(complex(values[name]) for name in edge_names)
    else:
        if len(values) != len(graph.edges):
            raise ValueError(f"q_values must contain {len(graph.edges)} entries")
        q_values = tuple(complex(value) for value in values)
    if any(
        not math.isfinite(value.real)
        or not math.isfinite(value.imag)
        or not 0.0 < abs(value) < 1.0
        for value in q_values
    ):
        raise ValueError("plumbing parameters must be finite and satisfy 0 < |q| < 1")
    return q_values


def _vertex_edge_indices(
    graph: TrivalentPlumbingGraph,
) -> tuple[tuple[int, int, int], ...]:
    indices: list[list[int | None]] = [
        [None, None, None] for _ in range(graph.vertex_count)
    ]
    for edge_index, edge in enumerate(graph.edges):
        for endpoint in edge.endpoints:
            indices[endpoint.vertex][endpoint.slot] = edge_index
    if any(item is None for vertex in indices for item in vertex):
        raise RuntimeError("trivalent graph is missing a local puncture slot")
    return tuple(
        tuple(int(item) for item in vertex)
        for vertex in indices
    )


@lru_cache(maxsize=None)
def _heisenberg_graph_level_coefficient(
    graph: TrivalentPlumbingGraph,
    levels: tuple[int, ...],
) -> complex:
    states_by_edge = tuple(integer_partitions(level) for level in levels)
    vertex_edges = _vertex_edge_indices(graph)
    coefficient = 0.0 + 0.0j
    for edge_states in product(*states_by_edge):
        numerator = 1
        for infinity_edge, one_edge, zero_edge in vertex_edges:
            rho = heisenberg_three_point(
                edge_states[infinity_edge],
                edge_states[one_edge],
                edge_states[zero_edge],
            )
            if rho == 0:
                numerator = 0
                break
            numerator *= rho
        if numerator == 0:
            continue
        denominator = math.prod(heisenberg_gram_norm(state) for state in edge_states)
        coefficient += numerator / denominator
    return complex(coefficient)


def heisenberg_plumbing_graph_partition(
    graph: TrivalentPlumbingGraph,
    q_values: Sequence[complex] | Mapping[str, complex],
    *,
    max_total_level: int,
    channel: str | None = None,
) -> HeisenbergPlumbingResult:
    """Direct Heisenberg state sewing on any closed trivalent graph."""

    if int(max_total_level) < 0:
        raise ValueError("max_total_level must be non-negative")
    q_tuple = _graph_edge_values(graph, q_values)
    total = 0.0 + 0.0j
    level_contributions: dict[tuple[int, ...], complex] = {}
    for total_level in range(int(max_total_level) + 1):
        for levels in _fixed_total_compositions(total_level, len(graph.edges)):
            coefficient = _heisenberg_graph_level_coefficient(graph, levels)
            if coefficient == 0.0:
                continue
            contribution = coefficient * math.prod(
                q_value**level
                for q_value, level in zip(q_tuple, levels)
            )
            if contribution != 0.0:
                level_contributions[levels] = complex(contribution)
                total += contribution
    return HeisenbergPlumbingResult(
        channel=str(channel or graph.name),
        edge_names=tuple(edge.name for edge in graph.edges),
        q_values=q_tuple,
        max_total_level=int(max_total_level),
        chiral_value=complex(total),
        nonchiral_value=float(abs(total) ** 2),
        level_contributions=level_contributions,
    )


def genus3_heisenberg_plumbing_partition(
    channel: str | Genus3PlumbingChannel,
    q_values: Sequence[complex] | Mapping[str, complex],
    *,
    max_total_level: int,
) -> HeisenbergPlumbingResult:
    """Direct rank-one Heisenberg sewing in any genus-three channel."""

    resolved = genus3_channel_by_name(channel) if isinstance(channel, str) else channel
    q_tuple = genus3_channel_q_values(resolved, q_values)
    return heisenberg_plumbing_graph_partition(
        resolved.graph,
        q_tuple,
        max_total_level=max_total_level,
        channel=resolved.name,
    )


def theta_heisenberg_plumbing_partition(
    q_zero: complex,
    q_one: complex,
    q_infty: complex,
    *,
    max_total_level: int,
) -> HeisenbergPlumbingResult:
    r"""Direct Heisenberg sewing sum in the theta graph.

    The chiral quantity is

      sum_{lambda,mu,nu} q_infty^|lambda| q_one^|mu| q_zero^|nu|
        rho(lambda,mu,nu)^2 / (z_lambda z_mu z_nu),

    truncated by ``|lambda|+|mu|+|nu| <= max_total_level``.
    """

    q_values = (complex(q_zero), complex(q_one), complex(q_infty))
    states = _states_by_level(int(max_total_level))
    total = 0.0 + 0.0j
    level_contributions: dict[tuple[int, int, int], complex] = {}
    # rho is ordered as (infinity bra, insertion at one, zero ket), whereas
    # generators_for_theta and the public plumbing API use (zero, one, infinity).
    for level_infty in range(int(max_total_level) + 1):
        for level_one in range(int(max_total_level) + 1 - level_infty):
            for level_zero in range(int(max_total_level) + 1 - level_infty - level_one):
                coefficient = 0.0 + 0.0j
                for state_infty in states[level_infty]:
                    norm_infty = heisenberg_gram_norm(state_infty)
                    for state_one in states[level_one]:
                        norm_one = heisenberg_gram_norm(state_one)
                        for state_zero in states[level_zero]:
                            rho = heisenberg_three_point(state_infty, state_one, state_zero)
                            if rho == 0:
                                continue
                            norm_zero = heisenberg_gram_norm(state_zero)
                            coefficient += (rho * rho) / (norm_infty * norm_one * norm_zero)
                contribution = (
                    (q_values[0] ** level_zero)
                    * (q_values[1] ** level_one)
                    * (q_values[2] ** level_infty)
                    * coefficient
                )
                if contribution != 0.0:
                    level_contributions[(level_zero, level_one, level_infty)] = contribution
                    total += contribution
    return HeisenbergPlumbingResult(
        channel="theta",
        edge_names=("q_zero", "q_one", "q_infty"),
        q_values=q_values,
        max_total_level=int(max_total_level),
        chiral_value=total,
        nonchiral_value=float(abs(total) ** 2),
        level_contributions=level_contributions,
    )


def glasses_heisenberg_plumbing_partition(
    q_left: complex,
    q_right: complex,
    q_bridge: complex,
    *,
    max_total_level: int,
) -> HeisenbergPlumbingResult:
    r"""Direct Heisenberg sewing sum in the separating glasses graph.

    The chiral quantity is

      sum_{lambda,mu,nu} qL^|lambda| qR^|mu| qB^|nu|
        rho(lambda,nu,lambda) rho(mu,nu,mu)
        / (z_lambda z_mu z_nu),

    truncated by total sewing level.
    """

    q_values = (complex(q_left), complex(q_right), complex(q_bridge))
    states = _states_by_level(int(max_total_level))
    total = 0.0 + 0.0j
    level_contributions: dict[tuple[int, int, int], complex] = {}
    for level_left in range(int(max_total_level) + 1):
        for level_right in range(int(max_total_level) + 1 - level_left):
            for level_bridge in range(int(max_total_level) + 1 - level_left - level_right):
                coefficient = 0.0 + 0.0j
                for state_left in states[level_left]:
                    norm_left = heisenberg_gram_norm(state_left)
                    for state_right in states[level_right]:
                        norm_right = heisenberg_gram_norm(state_right)
                        for state_bridge in states[level_bridge]:
                            rho_left = heisenberg_three_point(state_left, state_bridge, state_left)
                            if rho_left == 0:
                                continue
                            rho_right = heisenberg_three_point(state_right, state_bridge, state_right)
                            if rho_right == 0:
                                continue
                            norm_bridge = heisenberg_gram_norm(state_bridge)
                            coefficient += (rho_left * rho_right) / (
                                norm_left * norm_right * norm_bridge
                            )
                contribution = (
                    (q_values[0] ** level_left)
                    * (q_values[1] ** level_right)
                    * (q_values[2] ** level_bridge)
                    * coefficient
                )
                if contribution != 0.0:
                    level_contributions[(level_left, level_right, level_bridge)] = contribution
                    total += contribution
    return HeisenbergPlumbingResult(
        channel="glasses",
        edge_names=("q_left", "q_right", "q_bridge"),
        q_values=q_values,
        max_total_level=int(max_total_level),
        chiral_value=total,
        nonchiral_value=float(abs(total) ** 2),
        level_contributions=level_contributions,
    )


def run(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Direct Heisenberg pair-of-pants sewing sum."
    )
    parser.add_argument(
        "--channel",
        choices=("theta", "glasses", *GENUS3_CHANNEL_NAMES),
        required=True,
    )
    parser.add_argument("--q", type=parse_complex, nargs="+", required=True)
    parser.add_argument("--max-total-level", type=int, default=6)
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.channel in GENUS3_CHANNEL_NAMES:
        if len(args.q) != 6:
            parser.error("a genus-three channel requires six --q values")
        result = genus3_heisenberg_plumbing_partition(
            args.channel,
            args.q,
            max_total_level=args.max_total_level,
        )
    elif len(args.q) != 3:
        parser.error("a genus-two channel requires three --q values")
    elif args.channel == "theta":
        result = theta_heisenberg_plumbing_partition(*args.q, max_total_level=args.max_total_level)
    else:
        result = glasses_heisenberg_plumbing_partition(*args.q, max_total_level=args.max_total_level)

    print(f"Heisenberg pair-of-pants plumbing: {result.channel}")
    print(f"  q = ({', '.join(format_complex(value) for value in result.q_values)})")
    print(f"  max total level = {result.max_total_level}")
    print(f"  chiral value = {format_complex(result.chiral_value)}")
    print(f"  nonchiral value = {result.nonchiral_value:.16e}")
    print(f"  nonzero level tuples = {len(result.level_contributions)}")


if __name__ == "__main__":
    run()
