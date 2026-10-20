"""General fixed-difference elliptic h-recursion for sphere comb blocks."""

from __future__ import annotations

import functools
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Iterator, Sequence

import mpmath as mp

from .kac import (
    as_mpmath,
    background_data,
    degenerate_weight,
    fusion_polynomial,
    zamolodchikov_a,
)


Number = Any
MultiIndex = tuple[int, ...]


class RecursionWorkspace:
    """Bounded exact MP reuse across nodes at fixed external weights.

    Precision is part of every key. Values are never interpolated or reused
    at a different MP precision, central charge, or set of fusion weights.
    A preparer owns the workspace; it is not shared between worker threads.
    """

    def __init__(self, maximum_entries: int = 16384):
        if type(maximum_entries) is not int or maximum_entries < 1:
            raise ValueError("positive workspace capacity required")
        self.maximum_entries = maximum_entries
        self.cache = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key, calculate):
        key = (mp.mp.prec, *key)
        if key in self.cache:
            self.hits += 1
            self.cache.move_to_end(key)
            return self.cache[key]
        self.misses += 1
        value = calculate()
        self.cache[key] = value
        if len(self.cache) > self.maximum_entries:
            self.cache.popitem(last=False)
        return value


class KacPoleError(ZeroDivisionError):
    """Raised when a requested parameter point is too close to a Kac pole."""


@dataclass(frozen=True)
class PoleContext:
    """The smallest recursion denominator encountered while building a table."""

    magnitude: Number
    denominator: Number
    edge: int
    alpha: int
    beta: int
    levels: MultiIndex
    current_h: Number
    current_differences: tuple[Number, ...]
    pole_weight: Number


@dataclass(frozen=True)
class RecursionTable:
    """A total-degree-truncated reduced elliptic block ``H_n``."""

    coefficients: dict[MultiIndex, Number]
    order: int
    central_charge: Number
    external_weights: tuple[Number, ...]
    internal_weights: tuple[Number, ...]
    dps: int
    minimum_pole: PoleContext | None

    @property
    def point_count(self) -> int:
        return len(self.external_weights)

    @property
    def edge_count(self) -> int:
        return len(self.internal_weights)

    def evaluate(
        self,
        segment_nomes: Sequence[Number],
        *,
        order: int | None = None,
    ) -> Number:
        """Evaluate the reduced block in the raw segment nomes ``p_i``."""

        if len(segment_nomes) != self.edge_count:
            raise ValueError(
                f"expected {self.edge_count} segment nomes, got {len(segment_nomes)}"
            )
        truncation = self.order if order is None else int(order)
        if not 0 <= truncation <= self.order:
            raise ValueError(f"order must lie between 0 and {self.order}")
        with mp.workdps(self.dps):
            nomes = tuple(as_mpmath(value) for value in segment_nomes)
            total: Number = mp.mpf(0)
            for levels, coefficient in self.coefficients.items():
                if sum(levels) <= truncation:
                    monomial: Number = coefficient
                    for nome, level in zip(nomes, levels):
                        monomial *= nome**level
                    total += monomial
            return +total

    def shell(self, segment_nomes: Sequence[Number], level: int) -> Number:
        """Evaluate only the homogeneous total-degree ``level`` shell."""

        level = int(level)
        if not 0 <= level <= self.order:
            raise ValueError(f"level must lie between 0 and {self.order}")
        with mp.workdps(self.dps):
            nomes = tuple(as_mpmath(value) for value in segment_nomes)
            total: Number = mp.mpf(0)
            for levels, coefficient in self.coefficients.items():
                if sum(levels) == level:
                    monomial: Number = coefficient
                    for nome, power in zip(nomes, levels):
                        monomial *= nome**power
                    total += monomial
            return +total


def _compositions(total: int, dimension: int) -> Iterator[MultiIndex]:
    if dimension == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for tail in _compositions(total - first, dimension - 1):
            yield (first, *tail)


def total_degree_indices(dimension: int, order: int) -> Iterator[MultiIndex]:
    """Yield all nonnegative multi-indices of total degree at most ``order``."""

    dimension, order = int(dimension), int(order)
    if dimension < 1:
        raise ValueError("dimension must be positive")
    if order < 0:
        raise ValueError("order must be nonnegative")
    for total in range(order + 1):
        yield from _compositions(total, dimension)


def _validate_problem(
    external_weights: Sequence[Number],
    internal_weights: Sequence[Number],
    order: int,
) -> tuple[int, int]:
    point_count = len(external_weights)
    if point_count < 4:
        raise ValueError("a sphere block needs at least four external weights")
    edge_count = point_count - 3
    if len(internal_weights) != edge_count:
        raise ValueError(
            f"an n={point_count} comb block requires {edge_count} internal weights"
        )
    if int(order) != order or int(order) < 0:
        raise ValueError("order must be a nonnegative integer")
    return point_count, edge_count


def _edge_nome_scale(edge: int, edge_count: int) -> int:
    # For four points both endpoint normalizations live on the same edge.
    if edge_count == 1:
        return 16
    if edge in {0, edge_count - 1}:
        return 4
    return 1


def compute_h_recursion(
    *,
    central_charge: Number,
    external_weights: Sequence[Number],
    internal_weights: Sequence[Number],
    order: int,
    dps: int = 50,
    pole_tolerance: Number | None = None,
    workspace: RecursionWorkspace | None = None,
) -> RecursionTable:
    r"""Compute the reduced elliptic block through a total degree.

    External weights must be ordered as

    ``(d_0, d_z, mu_1, ..., mu_{n-4}, d_1, d_infinity)``.

    Internal weights are ordered from the ``(0,z)`` end to the ``(1,inf)``
    end.  The recursion sends all internal weights to infinity together while
    holding ``a_i=h_i-h_1`` fixed.
    """

    _, edge_count = _validate_problem(external_weights, internal_weights, order)
    dps = int(dps)
    if dps < 20:
        raise ValueError("dps must be at least 20")
    with mp.workdps(dps):
        c_value = as_mpmath(central_charge)
        external = tuple(as_mpmath(value) for value in external_weights)
        internal = tuple(as_mpmath(value) for value in internal_weights)
        h_base = internal[0]
        initial_differences = tuple(value - h_base for value in internal)
        q_background, b = background_data(c_value)
        tolerance = None if pole_tolerance is None else abs(as_mpmath(pole_tolerance))
        if tolerance is not None and tolerance == 0:
            raise ValueError("pole_tolerance must be positive")
        mobile_weights = external[2:-2]
        minimum: PoleContext | None = None

        @functools.lru_cache(maxsize=None)
        def pole_data(alpha: int, beta: int) -> tuple[Number, Number]:
            def calculate():
                return (degenerate_weight(alpha, beta, q_background, b),
                        zamolodchikov_a(alpha, beta, b))
            return (calculate() if workspace is None else
                    workspace.get(('pole', alpha, beta, q_background, b), calculate))

        def fusion(alpha, beta, *, top, bottom, q_background, b):
            def calculate():
                return fusion_polynomial(alpha, beta, top=top, bottom=bottom,
                                         q_background=q_background, b=b)
            return (calculate() if workspace is None else workspace.get(
                ('fusion', alpha, beta, top, bottom, q_background, b), calculate))

        def record_denominator(
            denominator: Number,
            *,
            edge: int,
            alpha: int,
            beta: int,
            levels: MultiIndex,
            current_h: Number,
            current_differences: tuple[Number, ...],
            pole: Number,
        ) -> None:
            nonlocal minimum
            magnitude = abs(denominator)
            context = PoleContext(
                magnitude=+magnitude,
                denominator=+denominator,
                edge=edge + 1,
                alpha=alpha,
                beta=beta,
                levels=levels,
                current_h=+current_h,
                current_differences=tuple(+value for value in current_differences),
                pole_weight=+pole,
            )
            if minimum is None or magnitude < minimum.magnitude:
                minimum = context
            if tolerance is not None and magnitude < tolerance:
                raise KacPoleError(
                    "recursion denominator is below pole_tolerance: "
                    f"edge={edge + 1}, (alpha,beta)=({alpha},{beta}), "
                    f"levels={levels}, |denominator|={mp.nstr(magnitude, 12)}"
                )

        @functools.lru_cache(maxsize=None)
        def coefficient(
            levels: MultiIndex,
            current_h: Number,
            current_differences: tuple[Number, ...],
        ) -> Number:
            total: Number = mp.mpf(1) if not any(levels) else mp.mpf(0)
            for edge, available in enumerate(levels):
                for alpha in range(1, available + 1):
                    for beta in range(1, available // alpha + 1):
                        null_level = alpha * beta
                        pole, null_norm = pole_data(alpha, beta)
                        difference = current_differences[edge]
                        denominator = current_h + difference - pole
                        record_denominator(
                            denominator,
                            edge=edge,
                            alpha=alpha,
                            beta=beta,
                            levels=levels,
                            current_h=current_h,
                            current_differences=current_differences,
                            pole=pole,
                        )

                        if edge == 0:
                            left = fusion(
                                alpha,
                                beta,
                                top=external[0],
                                bottom=external[1],
                                q_background=q_background,
                                b=b,
                            )
                        else:
                            left = fusion(
                                alpha,
                                beta,
                                top=pole
                                + current_differences[edge - 1]
                                - difference,
                                bottom=mobile_weights[edge - 1],
                                q_background=q_background,
                                b=b,
                            )
                        if edge == edge_count - 1:
                            right = fusion(
                                alpha,
                                beta,
                                top=external[-1],
                                bottom=external[-2],
                                q_background=q_background,
                                b=b,
                            )
                        else:
                            right = fusion(
                                alpha,
                                beta,
                                top=pole
                                + current_differences[edge + 1]
                                - difference,
                                bottom=mobile_weights[edge],
                                q_background=q_background,
                                b=b,
                            )

                        remainder = list(levels)
                        remainder[edge] -= null_level
                        shifted_differences = list(current_differences)
                        if edge == 0:
                            shifted_h = pole + null_level
                            shifted_differences[0] = mp.mpf(0)
                            for other in range(1, edge_count):
                                shifted_differences[other] -= null_level
                        else:
                            shifted_h = pole - difference
                            shifted_differences[edge] += null_level
                        residue = (
                            mp.mpf(_edge_nome_scale(edge, edge_count)) ** null_level
                            * null_norm
                            * left
                            * right
                            / denominator
                        )
                        total += residue * coefficient(
                            tuple(remainder),
                            shifted_h,
                            tuple(shifted_differences),
                        )
            return total

        coefficients = {
            levels: +coefficient(levels, h_base, initial_differences)
            for levels in total_degree_indices(edge_count, int(order))
        }
        return RecursionTable(
            coefficients=coefficients,
            order=int(order),
            central_charge=+c_value,
            external_weights=tuple(+value for value in external),
            internal_weights=tuple(+value for value in internal),
            dps=dps,
            minimum_pole=minimum,
        )
