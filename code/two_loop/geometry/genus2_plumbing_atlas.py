#!/usr/bin/env python3
"""Mapping-class atlas diagnostics for genus-two plumbing coordinates.

For a fixed period matrix, this module searches symplectic markings of the two
genus-two pants-graph topologies.  Its finite-q inverse follows the shared
hybrid policy: normalized holomorphic one-forms in the bulk, rescaled
multiprecision holomorphic one-forms in mixed cusps, adaptive Schottky words
only when all three plumbing parameters are small, and explicit agreement
checks in the common transition region.  Leading plumbing formulae or a certified
holomorphic period-map table supply initial ``Omega -> q`` values.

The resulting q score is a numerical chart diagnostic, not by itself an error
bound for a Virasoro block.  ``reference-q-envelope`` below means only that the
chart lies inside the q range reached by the saved real c=25 order-12 overlap
runs.  An integration run must still compare successive recursion and
quadrature orders at its actual period matrix.
"""

from __future__ import annotations

from c1_paths import PACKAGE_ROOT, data_root, package_path, module_path

import argparse
import cmath
import json
import math
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Literal, Mapping, Sequence

import numpy as np

try:
    from bolza_ccy_recursion import bolza_period_matrix, leading_theta_q_from_omega
    from bolza_torus_plumbing_reach import (
        enumerate_symplectic_words,
        period_matrix_is_riemann,
        transform_omega,
    )
    from genus2_holomorphic_period_table import (
        HolomorphicPeriodMapTable,
        PeriodMapSeed,
    )
    from genus2_period_table import Genus2PeriodMapTable
    from genus2_hybrid_period_map import (
        HybridPeriodMapConfig,
        MULTIPRECISION_HOLOMORPHIC_ALGORITHM,
        SCHOTTKY_ALGORITHM,
        classify_period_map_region,
        evaluate_schottky_period_map,
        hybrid_period_matrix,
        is_schottky_algorithm,
        refine_multiprecision_holomorphic_inverse,
        refine_schottky_inverse,
    )
    from liouville_genus2 import format_complex, parse_complex
    from plumbing_algorithms import (
        glasses_collocation_period_matrix,
        generators_for_glasses,
        generators_for_theta,
        genus2_symmetric_period_vector,
        q_from_tau,
        solve_theta_collocation,
        symmetrized_period_matrix,
        tau_from_q,
    )
except ImportError:  # pragma: no cover - supports package-style execution
    from bolza_ccy_recursion import bolza_period_matrix, leading_theta_q_from_omega
    from bolza_torus_plumbing_reach import (
        enumerate_symplectic_words,
        period_matrix_is_riemann,
        transform_omega,
    )
    from genus2_holomorphic_period_table import (
        HolomorphicPeriodMapTable,
        PeriodMapSeed,
    )
    from genus2_period_table import Genus2PeriodMapTable
    from genus2_hybrid_period_map import (
        HybridPeriodMapConfig,
        MULTIPRECISION_HOLOMORPHIC_ALGORITHM,
        SCHOTTKY_ALGORITHM,
        classify_period_map_region,
        evaluate_schottky_period_map,
        hybrid_period_matrix,
        is_schottky_algorithm,
        refine_multiprecision_holomorphic_inverse,
        refine_schottky_inverse,
    )
    from liouville_genus2 import format_complex, parse_complex
    from plumbing_algorithms import (
        glasses_collocation_period_matrix,
        generators_for_glasses,
        generators_for_theta,
        genus2_symmetric_period_vector,
        q_from_tau,
        solve_theta_collocation,
        symmetrized_period_matrix,
        tau_from_q,
    )


TWO_PI_I = 2.0j * math.pi
MAX_TAU_IMAG = 102.0
LOG_Q_BINARY64_FLOOR = -690.0
# Compatibility constant retained for older sampling drivers.  It is not a
# period-map backend threshold.
MULTIPRECISION_FALLBACK_Q_MAX = 0.4
HOLOMORPHIC_COLLOCATION_MIN_Q = 1.0e-12
HOLOMORPHIC_COLLOCATION_MAX_TAU_IMAG = -math.log(
    HOLOMORPHIC_COLLOCATION_MIN_Q
) / (2.0 * math.pi)
Topology = Literal["theta", "glasses"]


def symplectic_matrix_csv_fields(
    matrix: Sequence[Sequence[int]],
    *,
    prefix: str = "symplectic_m",
) -> dict[str, int]:
    """Flatten an exact 4x4 marking into stable CSV columns."""

    array = np.asarray(matrix)
    if array.shape != (4, 4):
        raise ValueError("a genus-two symplectic marking must be 4x4")
    rounded = np.rint(array).astype(np.int64)
    if not np.array_equal(array, rounded):
        raise ValueError("a symplectic marking must have exact integer entries")
    form = np.block(
        [
            [np.zeros((2, 2), dtype=np.int64), np.eye(2, dtype=np.int64)],
            [-np.eye(2, dtype=np.int64), np.zeros((2, 2), dtype=np.int64)],
        ]
    )
    if not np.array_equal(rounded.T @ form @ rounded, form):
        raise ValueError("saved marking is not symplectic")
    return {
        f"{prefix}{row}{column}": int(rounded[row, column])
        for row in range(4)
        for column in range(4)
    }


def symplectic_matrix_from_csv_row(
    row: Mapping[str, object],
    *,
    prefix: str = "symplectic_m",
) -> np.ndarray | None:
    """Recover a saved exact marking, or return ``None`` for legacy rows."""

    keys = [f"{prefix}{i}{j}" for i in range(4) for j in range(4)]
    present = [str(row.get(key, "")).strip() != "" for key in keys]
    if not any(present):
        return None
    if not all(present):
        raise ValueError("saved symplectic marking has incomplete matrix columns")
    values = [int(str(row[key]).strip()) for key in keys]
    matrix = np.asarray(values, dtype=np.int64).reshape(4, 4)
    # Reuse the exact-integer and symplectic checks before accepting disk data.
    symplectic_matrix_csv_fields(matrix, prefix=prefix)
    return matrix


@dataclass(frozen=True)
class LeadingMarking:
    """A symplectic marking ranked using the degeneration formula only."""

    topology: Topology
    word: str
    matrix: tuple[tuple[int, ...], ...]
    omega: np.ndarray
    leading_q: tuple[complex, complex, complex]
    leading_q_abs: tuple[float, float, float]
    leading_q_max: float
    leading_q_spread: float
    leading_log_q: tuple[complex, complex, complex] | None = None
    table_seed: PeriodMapSeed | None = None


@dataclass(frozen=True)
class PlumbingChartResult:
    """A finite-q inverse-period check for one plumbing chart."""

    topology: Topology
    word: str
    matrix: tuple[tuple[int, ...], ...]
    modular_det_abs: float
    omega_chart: tuple[tuple[str, ...], ...]
    integer_branch: tuple[tuple[int, ...], ...]
    q: tuple[str, str, str]
    log_q: tuple[str, str, str]
    tau: tuple[str, str, str]
    q_abs: tuple[float, float, float]
    q_max: float
    leading_q_abs: tuple[float, float, float]
    leading_q_max: float
    max_schottky_multiplier: float
    period_algorithm: str
    period_map_region: str
    plumbing_geometry_margin: float
    period_overlap_residual: float | None
    inverse_seed_source: str
    collocation_basis_order: int | None
    collocation_samples_per_seam: int | None
    collocation_seam_residual: float | None
    collocation_symmetry_error: float | None
    inverse_success: bool
    inverse_message: str
    inverse_nfev: int
    period_max_residual: float
    period_map_stability: float
    # Compatibility alias retained for readers of the original Schottky-only
    # atlas JSON schema.  In bulk charts this is a collocation-basis diagnostic.
    forward_word_stability: float
    word_length: int
    stability_word_length: int
    status: str


@dataclass(frozen=True)
class PlumbingAtlasResult:
    """Best checked charts found for one period matrix."""

    omega: tuple[tuple[str, ...], ...]
    search_depth: int
    prefilter_count: int
    q_reference_max: float
    period_tolerance: float
    stability_tolerance: float
    best_topology: str | None
    best_q_max: float | None
    coverage_status: str
    charts: tuple[PlumbingChartResult, ...]
    note: str


@dataclass(frozen=True)
class _CertifiedCollocationCandidate:
    """One local inverse reached from one table/degeneration seed."""

    seed_source: str
    success: bool
    message: str
    nfev: int
    q: tuple[complex, complex, complex]
    log_q: tuple[complex, complex, complex]
    forward: np.ndarray
    branch: np.ndarray
    residual: float
    stability: float
    seam_residual: float
    symmetry_error: float
    basis_order: int
    samples_per_seam: int
    max_multiplier: float
    algorithm: str = "holomorphic-form-collocation"
    low_order: int = 0
    high_order: int = 0
    region: str = "unvalidated"
    geometry_margin: float = math.nan
    overlap_residual: float | None = None


def leading_period_seed(marking: LeadingMarking) -> PeriodMapSeed:
    """Return the degeneration-formula seed without a ``q -> Omega`` call."""

    log_q = marking.leading_log_q
    if log_q is None:
        log_q = tuple(cmath.log(value) for value in marking.leading_q)
    return PeriodMapSeed(
        q=tuple(complex(value) for value in marking.leading_q),
        log_q=tuple(complex(value) for value in log_q),
        source="leading-plumbing-formula",
    )


def inverse_seeds_for_marking(
    marking: LeadingMarking,
    *,
    period_table: HolomorphicPeriodMapTable | Genus2PeriodMapTable | None = None,
    table_seed_count: int = 4,
    include_leading_seed: bool = True,
) -> tuple[PeriodMapSeed, ...]:
    """Collect table and leading seeds without evaluating a forward map."""

    seeds: list[PeriodMapSeed] = []
    if marking.table_seed is not None:
        seeds.append(marking.table_seed)
    if period_table is not None:
        seeds.extend(
            period_table.nearest_seeds(
                marking.topology,
                marking.omega,
                count=int(table_seed_count),
            )
        )
    if include_leading_seed:
        seeds.append(leading_period_seed(marking))

    unique: list[PeriodMapSeed] = []
    seen: set[tuple[float, ...]] = set()
    for seed in seeds:
        key = tuple(
            round(component, 14)
            for value in seed.q
            for component in (complex(value).real, complex(value).imag)
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(seed)
    return tuple(unique)


def _as_omega(omega: Sequence[Sequence[complex]]) -> np.ndarray:
    out = np.asarray(omega, dtype=np.complex128)
    if out.shape != (2, 2):
        raise ValueError(f"omega must be 2x2, got {out.shape}")
    out = 0.5 * (out + out.T)
    if not period_matrix_is_riemann(out):
        raise ValueError("omega is not in the genus-two Siegel upper half-space")
    return out


def format_chart_coordinate(value: complex) -> str:
    """Round-trip binary64 coordinates, including very large cusp logarithms.

    The display formatter's 12 decimal places can move a certified log(q)
    outside its absolute period tolerance when Im(tau) is several million.
    Serialized coordinates are computational inputs, not display labels.
    """
    value = complex(value)
    return f"{value.real:.17e}{value.imag:+.17e}j"

def _format_matrix(matrix: np.ndarray) -> tuple[tuple[str, ...], ...]:
    return tuple(tuple(format_chart_coordinate(value) for value in row) for row in matrix)


def _integer_matrix(matrix: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(value) for value in row) for row in matrix)


def _symplectic_det_factor(matrix: np.ndarray, omega: np.ndarray) -> complex:
    c = matrix[2:, :2]
    d = matrix[2:, 2:]
    return complex(np.linalg.det(c @ omega + d))


@lru_cache(maxsize=None)
def _symplectic_words(search_depth: int) -> tuple[tuple[str, np.ndarray], ...]:
    return tuple(enumerate_symplectic_words(int(search_depth)))


def leading_q_for_topology(omega: np.ndarray, topology: Topology) -> tuple[complex, complex, complex]:
    """Return the leading sewing coordinates in one fixed marking.

    These coordinates are for ranking only.  Accepted atlas charts always use
    the finite-q inverse result returned by :func:`certify_marking`.
    """

    omega = _as_omega(omega)
    # Some rejected symplectic images have exponentially large leading
    # coordinates.  They are discarded by _valid_leading_q; suppressing the
    # intermediate overflow keeps a full-domain scan diagnostically quiet.
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        if topology == "theta":
            return leading_theta_q_from_omega(omega)
        if topology == "glasses":
            return (
                complex(np.exp(TWO_PI_I * omega[0, 0])),
                complex(np.exp(TWO_PI_I * omega[1, 1])),
                complex(-TWO_PI_I * omega[0, 1]),
            )
    raise ValueError(f"unknown topology {topology!r}")


def leading_log_q_for_topology(
    omega: np.ndarray,
    topology: Topology,
) -> tuple[complex, complex, complex]:
    """Return natural logarithms of the leading sewing coordinates.

    Keeping these logarithms avoids losing a good cusp chart when one tube is
    longer than the binary64 range for its exponentiated ``q`` coordinate.
    """

    omega = _as_omega(omega)
    if topology == "theta":
        taus = (
            omega[0, 0] - omega[0, 1],
            omega[1, 1] - omega[0, 1],
            omega[0, 1],
        )
        return tuple(complex(TWO_PI_I * tau) for tau in taus)  # type: ignore[return-value]
    if topology == "glasses":
        bridge = complex(-TWO_PI_I * omega[0, 1])
        if bridge == 0.0:
            raise ValueError("the leading glasses bridge coordinate vanishes")
        return (
            complex(TWO_PI_I * omega[0, 0]),
            complex(TWO_PI_I * omega[1, 1]),
            cmath.log(bridge),
        )
    raise ValueError(f"unknown topology {topology!r}")


def _q_surrogate_from_log(log_q: complex) -> complex:
    return cmath.exp(complex(max(float(log_q.real), LOG_Q_BINARY64_FLOOR), float(log_q.imag)))


def _q_abs_from_log(log_q: complex) -> float:
    return math.exp(float(log_q.real)) if log_q.real > -745.0 else 0.0


def _valid_leading_log_q(log_q_values: Sequence[complex]) -> bool:
    return all(
        math.isfinite(value.real)
        and math.isfinite(value.imag)
        and value.real < 0.0
        for value in log_q_values
    )


def shortlist_markings(
    omega: Sequence[Sequence[complex]],
    topology: Topology,
    *,
    search_depth: int = 3,
    count: int = 6,
) -> list[LeadingMarking]:
    """Rank symplectic markings by their leading maximum edge ``|q|``."""

    source = _as_omega(omega)
    candidates: list[LeadingMarking] = []
    seen_q: set[tuple[float, ...]] = set()
    for word, matrix in _symplectic_words(int(search_depth)):
        try:
            transformed = transform_omega(matrix, source)
        except np.linalg.LinAlgError:
            continue
        transformed = 0.5 * (transformed + transformed.T)
        if not period_matrix_is_riemann(transformed):
            continue
        try:
            log_q_values = leading_log_q_for_topology(transformed, topology)
        except ValueError:
            continue
        if not _valid_leading_log_q(log_q_values):
            continue
        q_values = tuple(_q_surrogate_from_log(value) for value in log_q_values)
        q_abs = tuple(_q_abs_from_log(value) for value in log_q_values)
        q_max = math.exp(max(value.real for value in log_q_values))
        # Integral B translations give the same exponentiated sewing data.
        # Avoid spending nonlinear solves on those duplicate branches.
        q_key = tuple(
            round(component, 13)
            for value in log_q_values
            for component in (value.real, math.remainder(value.imag, 2.0 * math.pi))
        )
        if q_key in seen_q:
            continue
        seen_q.add(q_key)
        candidates.append(
            LeadingMarking(
                topology=topology,
                word=word,
                matrix=_integer_matrix(matrix),
                omega=transformed,
                leading_q=tuple(complex(value) for value in q_values),
                leading_q_abs=q_abs,
                leading_q_max=q_max,
                leading_q_spread=(
                    math.exp(max(value.real for value in log_q_values) - min(value.real for value in log_q_values))
                    if max(value.real for value in log_q_values) - min(value.real for value in log_q_values) < 700.0
                    else math.inf
                ),
                leading_log_q=log_q_values,
            )
        )
    def ranking_key(item: LeadingMarking) -> tuple[float, float, float, float, str]:
        return (
            0.0,
            round(item.leading_q_max, 13),
            (
                -item.leading_log_q[2].real
                if item.topology == "theta" and item.leading_log_q is not None
                else 0.0
            ),
            math.log(item.leading_q_spread),
            item.word,
        )

    candidates.sort(key=ranking_key)
    return candidates[: int(count)]


def fundamental_table_markings(
    omega: Sequence[Sequence[complex]],
    topology: Topology,
    *,
    period_table: Genus2PeriodMapTable | None,
    count: int = 4,
) -> list[LeadingMarking]:
    """Turn nearby fundamental table rows into exact marked chart targets."""

    if period_table is None or not period_table.has_fundamental_index:
        return []
    source = _as_omega(omega)
    out: list[LeadingMarking] = []
    for item in period_table.nearest_fundamental_seeds(
        topology, source, count=int(count)
    ):
        # Extremely deep cusps can make an otherwise exact integral
        # symplectic image lose positive definiteness in complex128 through
        # cancellation.  A nearest-table row with such an unusable image is
        # only one optional seed; it must not abort the complete chart search.
        try:
            marked = _as_omega(item.omega_marked)
        except ValueError:
            continue
        try:
            log_q_values = leading_log_q_for_topology(marked, topology)
        except ValueError:
            continue
        if not _valid_leading_log_q(log_q_values):
            continue
        q_values = tuple(_q_surrogate_from_log(value) for value in log_q_values)
        q_abs = tuple(_q_abs_from_log(value) for value in log_q_values)
        spread_log = max(value.real for value in log_q_values) - min(
            value.real for value in log_q_values
        )
        out.append(
            LeadingMarking(
                topology=topology,
                word=f"fundamental-table:{item.row_id}",
                matrix=item.matrix_fund_to_raw,
                omega=marked,
                leading_q=tuple(complex(value) for value in q_values),
                leading_q_abs=q_abs,
                leading_q_max=math.exp(max(value.real for value in log_q_values)),
                leading_q_spread=(
                    math.exp(spread_log) if spread_log < 700.0 else math.inf
                ),
                leading_log_q=log_q_values,
                table_seed=item.seed,
            )
        )
    return out


def best_leading_score(
    omega: Sequence[Sequence[complex]],
    topology: Topology,
    *,
    search_depth: int = 3,
) -> float:
    """Return the best leading ``max |q_e|`` score at finite search depth."""

    candidates = shortlist_markings(omega, topology, search_depth=search_depth, count=1)
    return candidates[0].leading_q_max if candidates else math.inf


def _atlas_marking_search_key(marking: LeadingMarking) -> tuple[object, ...]:
    """Rank theta and glasses markings in one topology-neutral queue.

    The nonlinear inverse is the expensive part of the atlas search.  Its
    early-stop path must therefore try the marking with the best cheap
    degeneration score first, regardless of which topology produced it.
    Historically the nested ``theta`` then ``glasses`` loops could accept a
    merely usable theta chart without ever testing a substantially smaller-q
    glasses chart.

    The rounded primary key is consistent with the per-topology shortlist.
    Equal-score ties prefer glasses because its separating-channel evaluator
    is production-certified and this avoids retaining the historical theta
    ordering as an implicit tie-break; no finite q advantage is assigned.
    """

    return (
        round(float(marking.leading_q_max), 13),
        0 if marking.topology == "glasses" else 1,
        math.log(float(marking.leading_q_spread)),
        marking.word,
    )


def _competitive_early_stop_topologies(
    markings: Sequence[LeadingMarking],
    *,
    score_ratio: float,
) -> frozenset[Topology]:
    """Return topologies that need finite-q comparison before early stop.

    A degeneration score is an excellent coarse selector but is not the final
    finite-q answer.  If the best theta and glasses scores are close, both
    topologies are cheap enough to certify once; otherwise only the clearly
    preferred topology blocks early stopping.
    """

    ratio = float(score_ratio)
    if not math.isfinite(ratio) or ratio < 1.0:
        raise ValueError("early-stop competitive score ratio must be finite and >= 1")
    best: dict[Topology, float] = {}
    for marking in markings:
        score = float(marking.leading_q_max)
        best[marking.topology] = min(score, best.get(marking.topology, math.inf))
    if not best:
        return frozenset()
    minimum = min(best.values())
    return frozenset(
        topology for topology, score in best.items() if score <= ratio * minimum
    )


def _prioritize_competitive_topology_heads(
    markings: Sequence[LeadingMarking],
    competitive_topologies: frozenset[Topology],
) -> list[LeadingMarking]:
    """Place one best marking from each near-tied topology at queue front."""

    ranked = list(markings)
    if len(competitive_topologies) <= 1:
        return ranked
    heads: list[LeadingMarking] = []
    rest: list[LeadingMarking] = []
    seen_heads: set[Topology] = set()
    for marking in ranked:
        if (
            marking.topology in competitive_topologies
            and marking.topology not in seen_heads
        ):
            seen_heads.add(marking.topology)
            heads.append(marking)
        else:
            rest.append(marking)
    return heads + rest


def _period_difference_mod_integer(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Subtract period matrices after removing symmetric integral B shifts."""

    difference = np.asarray(left - right, dtype=np.complex128)
    branch = np.rint(difference.real).astype(int)
    branch = np.rint(0.5 * (branch + branch.T)).astype(int)
    return difference - branch


def _pack_theta_q(q_values: Sequence[complex]) -> np.ndarray:
    taus = tuple(tau_from_q(complex(value)) for value in q_values)
    return np.asarray([component for tau in taus for component in (tau.real, tau.imag)], dtype=float)


def _unpack_theta_q(x: np.ndarray) -> tuple[complex, complex, complex]:
    return tuple(
        q_from_tau(complex(float(x[index]), float(x[index + 1])))
        for index in (0, 2, 4)
    )  # type: ignore[return-value]


def _pack_glasses_q(q_values: Sequence[complex]) -> np.ndarray:
    tau1 = tau_from_q(complex(q_values[0]))
    tau2 = tau_from_q(complex(q_values[1]))
    q3 = complex(q_values[2])
    return np.asarray([tau1.real, tau1.imag, tau2.real, tau2.imag, q3.real, q3.imag], dtype=float)


def _unpack_glasses_q(x: np.ndarray) -> tuple[complex, complex, complex]:
    return (
        q_from_tau(complex(float(x[0]), float(x[1]))),
        q_from_tau(complex(float(x[2]), float(x[3]))),
        complex(float(x[4]), float(x[5])),
    )


def _collocation_orders(topology: Topology, q_values: Sequence[complex]) -> tuple[int, int, int, int]:
    q_max = max(abs(value) for value in q_values)
    if topology == "theta":
        if q_max <= 0.2:
            low_basis, high_basis = 20, 24
        elif q_max <= 0.3:
            low_basis, high_basis = 24, 32
        else:
            low_basis, high_basis = 32, 40
        return low_basis, 4 * low_basis, high_basis, 4 * high_basis
    if q_max <= 0.2:
        low_basis, high_basis = 20, 28
    elif q_max <= 0.3:
        low_basis, high_basis = 24, 32
    else:
        low_basis, high_basis = 32, 40
    return low_basis, 6 * low_basis, high_basis, 6 * high_basis


def _collocation_forward(
    topology: Topology,
    q_values: Sequence[complex],
    *,
    basis_order: int,
    samples_per_seam: int,
) -> tuple[np.ndarray, float, float]:
    if topology == "theta":
        result = solve_theta_collocation(
            *q_values,
            basis_order=int(basis_order),
            samples_per_seam=int(samples_per_seam),
        )
        return (
            symmetrized_period_matrix(result.omega),
            float(result.max_seam_residual),
            float(result.omega_symmetry_error),
        )
    omega, seam_residual, symmetry_error = glasses_collocation_period_matrix(
        *q_values,
        basis_order=int(basis_order),
        samples_per_seam=int(samples_per_seam),
    )
    return symmetrized_period_matrix(omega), float(seam_residual), float(symmetry_error)


def _refine_collocation_inverse(
    marking: LeadingMarking,
    initial_q: Sequence[complex],
    *,
    basis_order: int,
    samples_per_seam: int,
    max_nfev: int,
    q3_component_bound: float,
) -> tuple[
    bool,
    str,
    int,
    tuple[complex, complex, complex],
    np.ndarray,
    np.ndarray,
    float,
    float,
    float,
]:
    """Refine one leading/table seed with the normalized holomorphic-form map."""

    from scipy.optimize import least_squares

    initial_forward, _, _ = _collocation_forward(
        marking.topology,
        initial_q,
        basis_order=basis_order,
        samples_per_seam=samples_per_seam,
    )
    # Determine the integral B-cycle branch using the same holomorphic-form
    # convention that is used throughout the refinement.
    branch = np.rint((initial_forward - marking.omega).real).astype(int)
    branch = np.rint(0.5 * (branch + branch.T)).astype(int)
    target_on_branch = marking.omega + branch
    target_vector = genus2_symmetric_period_vector(target_on_branch)
    if marking.topology == "theta":
        x0 = _pack_theta_q(initial_q)
        lower = np.asarray([-np.inf, 1.0e-12, -np.inf, 1.0e-12, -np.inf, 1.0e-12], dtype=float)
        upper = np.asarray(
            [
                np.inf,
                HOLOMORPHIC_COLLOCATION_MAX_TAU_IMAG,
                np.inf,
                HOLOMORPHIC_COLLOCATION_MAX_TAU_IMAG,
                np.inf,
                HOLOMORPHIC_COLLOCATION_MAX_TAU_IMAG,
            ],
            dtype=float,
        )
        unpack = _unpack_theta_q
    else:
        x0 = _pack_glasses_q(initial_q)
        bound = float(q3_component_bound)
        lower = np.asarray([-np.inf, 1.0e-12, -np.inf, 1.0e-12, -bound, -bound], dtype=float)
        upper = np.asarray(
            [
                np.inf,
                HOLOMORPHIC_COLLOCATION_MAX_TAU_IMAG,
                np.inf,
                HOLOMORPHIC_COLLOCATION_MAX_TAU_IMAG,
                bound,
                bound,
            ],
            dtype=float,
        )
        x0 = np.minimum(np.maximum(x0, lower), upper)
        unpack = _unpack_glasses_q

    def residual(x: np.ndarray) -> np.ndarray:
        q_values = unpack(x)
        if any(
            not HOLOMORPHIC_COLLOCATION_MIN_Q <= abs(value) < 1.0
            for value in q_values
        ):
            return 1.0e6 * np.ones(6, dtype=float)
        try:
            # Least-squares may briefly probe nearly singular seams.  They are
            # penalty points, so contain their floating warnings and reject the
            # resulting non-finite map below.
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                forward, seam_residual, symmetry_error = _collocation_forward(
                    marking.topology,
                    q_values,
                    basis_order=basis_order,
                    samples_per_seam=samples_per_seam,
                )
        except Exception:
            return 1.0e6 * np.ones(6, dtype=float)
        if (
            not np.all(np.isfinite(forward))
            or not math.isfinite(seam_residual)
            or not math.isfinite(symmetry_error)
        ):
            return 1.0e6 * np.ones(6, dtype=float)
        return genus2_symmetric_period_vector(forward) - target_vector

    opt = least_squares(
        residual,
        x0,
        bounds=(lower, upper),
        max_nfev=int(max_nfev),
        diff_step=1.0e-5,
        x_scale="jac",
        xtol=1.0e-10,
        ftol=1.0e-10,
        gtol=1.0e-10,
    )
    q_values = unpack(opt.x)
    forward, seam_residual, symmetry_error = _collocation_forward(
        marking.topology,
        q_values,
        basis_order=basis_order,
        samples_per_seam=samples_per_seam,
    )
    max_residual = float(np.max(np.abs(forward - target_on_branch)))
    return (
        bool(opt.success),
        str(opt.message),
        int(opt.nfev),
        q_values,
        forward,
        branch,
        max_residual,
        seam_residual,
        symmetry_error,
    )


def _chart_status(
    *,
    success: bool,
    residual: float,
    stability: float,
    q_max: float,
    q_reference_max: float,
    period_tolerance: float,
    stability_tolerance: float,
) -> str:
    if not success or not math.isfinite(residual) or residual > period_tolerance:
        return "inverse-failed"
    if not math.isfinite(stability) or stability > stability_tolerance:
        return "period-map-unstable"
    if q_max <= q_reference_max:
        return "reference-q-envelope"
    return "requires-recursion-order-study"


def _collocation_candidate_from_seed(
    marking: LeadingMarking,
    seed: PeriodMapSeed,
    *,
    max_nfev: int,
    q3_component_bound: float,
) -> _CertifiedCollocationCandidate:
    """Run and certify one local inverse without another period-map backend."""

    if len(seed.q) != 3 or len(seed.log_q) != 3:
        raise ValueError("an inverse seed must contain three q and log(q) values")
    minimum_log_q = math.log(HOLOMORPHIC_COLLOCATION_MIN_Q)
    if any(complex(value).real < minimum_log_q for value in seed.log_q) or any(
        not HOLOMORPHIC_COLLOCATION_MIN_Q <= abs(complex(value)) < 1.0
        for value in seed.q
    ):
        raise ValueError(
            "seed is outside the current normalized-holomorphic-form conditioning "
            f"range |q_e|>={HOLOMORPHIC_COLLOCATION_MIN_Q:.0e}"
        )

    low_basis, low_samples, high_basis, high_samples = _collocation_orders(
        marking.topology,
        seed.q,
    )
    direct = _refine_collocation_inverse(
        marking,
        seed.q,
        basis_order=high_basis,
        samples_per_seam=high_samples,
        max_nfev=max_nfev,
        q3_component_bound=q3_component_bound,
    )
    (
        direct_success,
        direct_message,
        direct_nfev,
        q_values,
        forward,
        branch,
        residual,
        seam_residual,
        symmetry_error,
    ) = direct

    # If the local solve crosses an adaptive-order boundary, rerun the inverse
    # at the order appropriate to the final q rather than certifying the seed's
    # original order.
    final_orders = _collocation_orders(marking.topology, q_values)
    if final_orders[2:] != (high_basis, high_samples):
        low_basis, low_samples, high_basis, high_samples = final_orders
        direct = _refine_collocation_inverse(
            marking,
            q_values,
            basis_order=high_basis,
            samples_per_seam=high_samples,
            max_nfev=max_nfev,
            q3_component_bound=q3_component_bound,
        )
        (
            direct_success,
            direct_message,
            extra_nfev,
            q_values,
            forward,
            branch,
            residual,
            seam_residual,
            symmetry_error,
        ) = direct
        direct_nfev += extra_nfev

    forward_low, _, _ = _collocation_forward(
        marking.topology,
        q_values,
        basis_order=low_basis,
        samples_per_seam=low_samples,
    )
    basis_stability = float(
        np.max(np.abs(_period_difference_mod_integer(forward, forward_low)))
    )
    stability = max(
        basis_stability,
        float(seam_residual),
        float(symmetry_error),
    )
    log_q_values = tuple(cmath.log(value) for value in q_values)

    # This multiplier is retained only as a downstream CFT diagnostic.  It is
    # not used to compute a period matrix or to accept the inverse.
    try:
        generators = (
            generators_for_theta(*q_values)
            if marking.topology == "theta"
            else generators_for_glasses(*q_values)
        )
        max_multiplier = max(
            float(abs(generator.multiplier)) for generator in generators
        )
    except Exception:
        max_multiplier = math.nan

    return _CertifiedCollocationCandidate(
        seed_source=seed.source,
        success=bool(direct_success),
        message=str(direct_message),
        nfev=int(direct_nfev),
        q=tuple(complex(value) for value in q_values),
        log_q=tuple(complex(value) for value in log_q_values),
        forward=np.asarray(forward, dtype=np.complex128),
        branch=np.asarray(branch, dtype=int),
        residual=float(residual),
        stability=float(stability),
        seam_residual=float(seam_residual),
        symmetry_error=float(symmetry_error),
        basis_order=int(high_basis),
        samples_per_seam=int(high_samples),
        max_multiplier=float(max_multiplier),
        algorithm="holomorphic-form-collocation",
        low_order=int(low_basis),
        high_order=int(high_basis),
    )


def _schottky_candidate_from_seed(
    marking: LeadingMarking,
    seed: PeriodMapSeed,
    *,
    word_length: int,
    max_nfev: int,
    q3_component_bound: float,
    period_tolerance: float,
    stability_tolerance: float,
) -> _CertifiedCollocationCandidate:
    """Refine and certify a long-cusp seed with adaptive Schottky words."""

    policy_tolerance = min(float(period_tolerance), float(stability_tolerance))
    policy = HybridPeriodMapConfig(
        tolerance=policy_tolerance,
        agreement_tolerance=max(policy_tolerance, float(period_tolerance)),
        minimum_schottky_word=max(4, int(word_length) - 1),
        maximum_schottky_word=max(9, int(word_length) + 2),
    )
    inverse = refine_schottky_inverse(
        marking.topology,
        marking.omega,
        seed.q,
        initial_log_q=seed.log_q,
        config=policy,
        fit_word_length=max(5, int(word_length)),
        max_nfev=max_nfev,
        q3_component_bound=q3_component_bound,
    )
    try:
        generators = (
            generators_for_theta(*inverse.q)
            if marking.topology == "theta"
            else generators_for_glasses(*inverse.q)
        )
        max_multiplier = max(float(abs(generator.multiplier)) for generator in generators)
    except Exception:
        max_multiplier = math.nan
    return _CertifiedCollocationCandidate(
        seed_source=seed.source,
        success=bool(inverse.success),
        message=str(inverse.message),
        nfev=int(inverse.nfev),
        q=inverse.q,
        log_q=inverse.log_q,
        forward=np.asarray(inverse.omega, dtype=np.complex128),
        branch=np.asarray(inverse.branch, dtype=int),
        residual=float(inverse.residual),
        stability=float(inverse.stability),
        seam_residual=math.nan,
        symmetry_error=float(inverse.symmetry_error),
        basis_order=0,
        samples_per_seam=0,
        max_multiplier=float(max_multiplier),
        algorithm=SCHOTTKY_ALGORITHM,
        low_order=int(inverse.word_length),
        high_order=int(inverse.validation_word_length),
    )


def _bounded_log_update(value: complex, *, maximum: float = 2.0) -> complex:
    """Limit one analytic log-q correction to its local table chart."""

    magnitude = abs(value)
    if not math.isfinite(magnitude):
        return 0.0j
    if magnitude <= float(maximum):
        return complex(value)
    return complex(value) * (float(maximum) / magnitude)


def _table_first_schottky_candidate_from_seed(
    marking: LeadingMarking,
    seed: PeriodMapSeed,
    *,
    maximum_word: int,
    maximum_corrections: int,
    period_tolerance: float,
    stability_tolerance: float,
) -> _CertifiedCollocationCandidate:
    """Certify a transported table seed before attempting nonlinear inversion.

    This path is deliberately restricted to the all-small-q region.  It
    evaluates the adaptive Schottky product once and, when the transported
    seed is close but not yet inside the requested residual bar, applies at
    most a few analytic corrections in log(q).  In the two-method overlap it
    additionally requires one holomorphic/Schottky forward agreement check.
    The expensive multi-seed least-squares inverse remains the caller's
    fallback.
    """

    if len(seed.q) != 3 or len(seed.log_q) != 3:
        raise ValueError("a table-first inverse seed must contain three q and log(q) values")
    tolerance = min(float(period_tolerance), float(stability_tolerance))
    region_policy = HybridPeriodMapConfig(
        tolerance=tolerance,
        agreement_tolerance=float(period_tolerance),
        maximum_collocation_basis=160,
    )
    seed_region, _ = classify_period_map_region(
        marking.topology,
        seed.q,
        config=region_policy,
        log_q_values=seed.log_q,
    )
    all_small_regions = {"schottky-all-small", "two-method-overlap"}
    if seed_region not in all_small_regions:
        raise ValueError(
            "table-first Schottky inversion is restricted to the all-small-q region"
        )

    schottky_policy = HybridPeriodMapConfig(
        tolerance=tolerance,
        agreement_tolerance=float(period_tolerance),
        minimum_schottky_word=2,
        maximum_schottky_word=max(4, int(maximum_word)),
        crosscheck_overlap=False,
    )
    logs = tuple(complex(value) for value in seed.log_q)
    q_values = tuple(_q_surrogate_from_log(value) for value in logs)
    seed_label = (
        "transported table seed"
        if "period-table" in seed.source
        else "leading cusp seed"
    )
    best: _CertifiedCollocationCandidate | None = None
    for correction_count in range(max(0, int(maximum_corrections)) + 1):
        region, geometry = classify_period_map_region(
            marking.topology,
            q_values,
            config=region_policy,
            log_q_values=logs,
        )
        if region not in all_small_regions:
            break
        evaluation = evaluate_schottky_period_map(
            marking.topology,
            q_values,
            config=schottky_policy,
            log_q_values=logs,
        )
        raw_difference = np.asarray(evaluation.omega - marking.omega, dtype=np.complex128)
        branch = np.rint(raw_difference.real).astype(int)
        branch = np.rint(0.5 * (branch + branch.T)).astype(int)
        difference = raw_difference - branch
        residual = float(np.max(np.abs(difference)))
        stability = float(max(evaluation.error_estimate, evaluation.symmetry_error))
        try:
            generators = (
                generators_for_theta(*q_values)
                if marking.topology == "theta"
                else generators_for_glasses(*q_values)
            )
            max_multiplier = max(
                float(abs(generator.multiplier)) for generator in generators
            )
        except Exception:
            max_multiplier = math.nan
        success = bool(
            evaluation.converged
            and residual <= float(period_tolerance)
            and stability <= float(stability_tolerance)
        )
        overlap_residual: float | None = None
        crosscheck_failed = False
        if success and region == "two-method-overlap":
            overlap_policy = HybridPeriodMapConfig(
                tolerance=tolerance,
                agreement_tolerance=float(period_tolerance),
                maximum_collocation_basis=160,
                minimum_schottky_word=2,
                maximum_schottky_word=max(4, int(maximum_word)),
                crosscheck_overlap=True,
            )
            overlap = hybrid_period_matrix(
                marking.topology,
                q_values,
                config=overlap_policy,
                log_q_values=logs,
            )
            holomorphic = overlap.holomorphic
            schottky = overlap.schottky
            overlap_residual = overlap.overlap_residual
            crosscheck_failed = bool(
                holomorphic is None
                or schottky is None
                or not holomorphic.converged
                or not schottky.converged
                or overlap_residual is None
                or overlap_residual > float(period_tolerance)
            )
            if not crosscheck_failed:
                holomorphic_residual = float(
                    np.max(
                        np.abs(
                            _period_difference_mod_integer(
                                holomorphic.omega,
                                marking.omega,
                            )
                        )
                    )
                )
                residual = max(residual, holomorphic_residual)
                stability = max(
                    stability,
                    float(holomorphic.error_estimate),
                    float(overlap_residual),
                )
                success = bool(
                    residual <= float(period_tolerance)
                    and stability <= float(stability_tolerance)
                )
            else:
                success = False
        candidate = _CertifiedCollocationCandidate(
            seed_source=f"{seed.source}:direct-all-small",
            success=success,
            message=(
                f"{seed_label} passed one adaptive Schottky certificate"
                if correction_count == 0 and success
                else (
                    f"{seed_label} passed after {correction_count} "
                    "analytic log-q correction(s)"
                    if success
                    else (
                        f"direct all-small Schottky attempt {correction_count + 1} did not pass; "
                        f"residual={residual:.3e}, tail={stability:.3e}"
                    )
                )
            ),
            nfev=correction_count + 1,
            q=tuple(complex(value) for value in q_values),  # type: ignore[arg-type]
            log_q=tuple(complex(value) for value in logs),  # type: ignore[arg-type]
            forward=np.asarray(evaluation.omega, dtype=np.complex128),
            branch=np.asarray(branch, dtype=int),
            residual=residual,
            stability=stability,
            seam_residual=math.nan,
            symmetry_error=float(evaluation.symmetry_error),
            basis_order=0,
            samples_per_seam=0,
            max_multiplier=float(max_multiplier),
            algorithm=SCHOTTKY_ALGORITHM,
            low_order=int(evaluation.low_order),
            high_order=int(evaluation.high_order),
            region=region,
            geometry_margin=float(geometry.minimum_margin),
            overlap_residual=overlap_residual,
        )
        if best is None or (not candidate.success, candidate.residual, candidate.stability) < (
            not best.success,
            best.residual,
            best.stability,
        ):
            best = candidate
        if success or crosscheck_failed or correction_count >= int(maximum_corrections):
            break

        if marking.topology == "theta":
            updates = (
                -TWO_PI_I * (difference[0, 0] - difference[0, 1]),
                -TWO_PI_I * (difference[1, 1] - difference[0, 1]),
                -TWO_PI_I * difference[0, 1],
            )
        else:
            bridge_is_resolved = logs[2].real >= math.log(
                max(tolerance * 0.01, 1.0e-300)
            )
            bridge_update = (
                TWO_PI_I * difference[0, 1] / q_values[2]
                if bridge_is_resolved and q_values[2] != 0.0j
                else 0.0j
            )
            updates = (
                -TWO_PI_I * difference[0, 0],
                -TWO_PI_I * difference[1, 1],
                bridge_update,
            )
        proposed_logs = tuple(
            complex(
                min(-1.0e-12, old.real + update.real),
                old.imag + update.imag,
            )
            for old, raw_update in zip(logs, updates)
            for update in (_bounded_log_update(raw_update),)
        )
        logs = proposed_logs  # type: ignore[assignment]
        q_values = tuple(_q_surrogate_from_log(value) for value in logs)

    if best is None:
        raise RuntimeError("table-first Schottky seed left the all-small-q region")
    return best


def _multiprecision_candidate_from_seed(
    marking: LeadingMarking,
    seed: PeriodMapSeed,
    *,
    max_nfev: int,
    period_tolerance: float,
    stability_tolerance: float,
    maximum_collocation_basis: int = 160,
) -> _CertifiedCollocationCandidate:
    """Refine one mixed-cusp seed with rescaled holomorphic forms only."""

    policy_tolerance = min(float(period_tolerance), float(stability_tolerance))
    policy = HybridPeriodMapConfig(
        tolerance=policy_tolerance,
        agreement_tolerance=max(policy_tolerance, float(period_tolerance)),
        maximum_collocation_basis=int(maximum_collocation_basis),
    )
    inverse = refine_multiprecision_holomorphic_inverse(
        marking.topology,
        marking.omega,
        seed.q,
        initial_log_q=seed.log_q,
        config=policy,
        max_nfev=min(int(max_nfev), 36),
    )
    try:
        generators = (
            generators_for_theta(*inverse.q)
            if marking.topology == "theta"
            else generators_for_glasses(*inverse.q)
        )
        max_multiplier = max(
            float(abs(generator.multiplier)) for generator in generators
        )
    except Exception:
        max_multiplier = math.nan
    return _CertifiedCollocationCandidate(
        seed_source=seed.source,
        success=bool(inverse.success),
        message=str(inverse.message),
        nfev=int(inverse.nfev),
        q=inverse.q,
        log_q=inverse.log_q,
        forward=np.asarray(inverse.omega, dtype=np.complex128),
        branch=np.asarray(inverse.branch, dtype=int),
        residual=float(inverse.residual),
        stability=float(inverse.stability),
        seam_residual=float(inverse.seam_residual),
        symmetry_error=float(inverse.symmetry_error),
        basis_order=int(inverse.high_order),
        samples_per_seam=(4 if marking.topology == "theta" else 6)
        * int(inverse.high_order),
        max_multiplier=float(max_multiplier),
        algorithm=MULTIPRECISION_HOLOMORPHIC_ALGORITHM,
        low_order=int(inverse.low_order),
        high_order=int(inverse.high_order),
        region=inverse.region,
    )


def _hybrid_recertify_candidate(
    marking: LeadingMarking,
    candidate: _CertifiedCollocationCandidate,
    *,
    word_length: int,
    period_tolerance: float,
    stability_tolerance: float,
    maximum_collocation_basis: int = 160,
) -> _CertifiedCollocationCandidate:
    """Apply the shared regional forward certificate to an inverse candidate."""

    policy_tolerance = min(float(period_tolerance), float(stability_tolerance))
    policy = HybridPeriodMapConfig(
        tolerance=policy_tolerance,
        agreement_tolerance=float(period_tolerance),
        maximum_collocation_basis=int(maximum_collocation_basis),
        minimum_schottky_word=max(4, int(word_length) - 1),
        maximum_schottky_word=max(9, int(word_length) + 2),
    )
    certified = hybrid_period_matrix(
        marking.topology,
        candidate.q,
        config=policy,
        log_q_values=candidate.log_q,
    )
    selected = (
        certified.schottky
        if is_schottky_algorithm(certified.algorithm)
        else certified.holomorphic
    )
    if selected is None:
        raise RuntimeError("hybrid period-map selection returned no method diagnostics")
    difference = np.asarray(certified.omega - marking.omega, dtype=np.complex128)
    branch = np.rint(difference.real).astype(int)
    branch = np.rint(0.5 * (branch + branch.T)).astype(int)
    residual = float(
        np.max(np.abs(certified.omega - (marking.omega + branch)))
    )
    sample_factor = 4 if marking.topology == "theta" else 6
    overlap_text = (
        ""
        if certified.overlap_residual is None
        else f", overlap={certified.overlap_residual:.3e}"
    )
    # The regional hybrid evaluation is the final certificate.  Do not carry
    # a failed flag from the seed backend after a different selected backend
    # has produced an admissible, converged map with certified residual and
    # stability.  This occurs in the two-method overlap when a Schottky-seeded
    # inverse misses its own tail criterion but the holomorphic evaluation and
    # the explicit method-overlap check both pass.
    recertified_success = bool(
        certified.geometry.valid
        and selected.converged
        and math.isfinite(residual)
        and residual <= float(period_tolerance)
        and math.isfinite(certified.error_estimate)
        and max(certified.error_estimate, certified.overlap_residual or 0.0)
        <= float(stability_tolerance)
    )
    return _CertifiedCollocationCandidate(
        seed_source=candidate.seed_source,
        success=recertified_success,
        message=(
            f"{candidate.message}; hybrid region={certified.region}, "
            f"error={certified.error_estimate:.3e}{overlap_text}"
        ),
        nfev=candidate.nfev,
        q=candidate.q,
        log_q=candidate.log_q,
        forward=np.asarray(certified.omega, dtype=np.complex128),
        branch=branch,
        residual=residual,
        stability=float(
            max(certified.error_estimate, certified.overlap_residual or 0.0)
        ),
        seam_residual=float(selected.seam_residual),
        symmetry_error=float(selected.symmetry_error),
        basis_order=(
            int(selected.high_order)
            if not is_schottky_algorithm(certified.algorithm)
            else 0
        ),
        samples_per_seam=(
            sample_factor * int(selected.high_order)
            if not is_schottky_algorithm(certified.algorithm)
            else 0
        ),
        max_multiplier=candidate.max_multiplier,
        algorithm=certified.algorithm,
        low_order=int(selected.low_order),
        high_order=int(selected.high_order),
        region=certified.region,
        geometry_margin=float(certified.geometry.minimum_margin),
        overlap_residual=certified.overlap_residual,
    )


def certify_marking(
    source_omega: Sequence[Sequence[complex]],
    marking: LeadingMarking,
    *,
    word_length: int = 5,
    stability_step: int = 1,
    max_nfev: int = 120,
    q3_component_bound: float = 0.98,
    q_reference_max: float = 0.16,
    period_tolerance: float = 2.0e-6,
    stability_tolerance: float = 2.0e-6,
    inverse_seeds: Sequence[PeriodMapSeed] | None = None,
    table_first_schottky: bool = False,
    table_first_only: bool = False,
    table_first_maximum_word: int = 5,
    table_first_maximum_corrections: int = 2,
    maximum_collocation_basis: int = 160,
) -> PlumbingChartResult:
    """Invert one marking with the adaptive holomorphic/Schottky split."""

    source = _as_omega(source_omega)
    matrix = np.asarray(marking.matrix, dtype=int)
    seeds = tuple(inverse_seeds) if inverse_seeds is not None else (
        leading_period_seed(marking),
    )
    attempts: list[_CertifiedCollocationCandidate] = []
    failures: list[str] = []
    for seed in seeds:
        try:
            seed_region, _ = classify_period_map_region(
                marking.topology,
                seed.q,
                config=HybridPeriodMapConfig(
                    tolerance=min(float(period_tolerance), float(stability_tolerance)),
                    agreement_tolerance=float(period_tolerance),
                    maximum_collocation_basis=int(maximum_collocation_basis),
                ),
                log_q_values=seed.log_q,
            )
            if table_first_schottky and seed_region in {
                "schottky-all-small",
                "two-method-overlap",
            }:
                table_candidate = _table_first_schottky_candidate_from_seed(
                    marking,
                    seed,
                    maximum_word=table_first_maximum_word,
                    maximum_corrections=table_first_maximum_corrections,
                    period_tolerance=period_tolerance,
                    stability_tolerance=stability_tolerance,
                )
                attempts.append(table_candidate)
                if (
                    table_candidate.success
                    and table_candidate.residual <= float(period_tolerance)
                    and table_candidate.stability <= float(stability_tolerance)
                ):
                    break
                if table_first_only:
                    continue
            elif table_first_only:
                continue
            if seed_region in {"schottky-all-small", "two-method-overlap"}:
                candidate = _schottky_candidate_from_seed(
                    marking,
                    seed,
                    word_length=word_length,
                    max_nfev=max_nfev,
                    q3_component_bound=q3_component_bound,
                    period_tolerance=period_tolerance,
                    stability_tolerance=stability_tolerance,
                )
            elif seed_region == "holomorphic-mixed-cusp":
                candidate = _multiprecision_candidate_from_seed(
                    marking,
                    seed,
                    max_nfev=max_nfev,
                    period_tolerance=period_tolerance,
                    stability_tolerance=stability_tolerance,
                    maximum_collocation_basis=maximum_collocation_basis,
                )
            else:
                candidate = _collocation_candidate_from_seed(
                    marking,
                    seed,
                    max_nfev=max_nfev,
                    q3_component_bound=q3_component_bound,
                )
            certified_candidate = _hybrid_recertify_candidate(
                marking,
                candidate,
                word_length=word_length,
                period_tolerance=period_tolerance,
                stability_tolerance=stability_tolerance,
                maximum_collocation_basis=maximum_collocation_basis,
            )
            attempts.append(certified_candidate)
            if (
                certified_candidate.success
                and certified_candidate.residual <= float(period_tolerance)
                and certified_candidate.stability <= float(stability_tolerance)
            ):
                break
        except Exception as exc:
            failures.append(f"{seed.source}: {type(exc).__name__}: {exc}")

    if attempts:
        status_order = {
            "reference-q-envelope": 0,
            "requires-recursion-order-study": 1,
            "period-map-unstable": 2,
            "inverse-failed": 3,
        }

        def candidate_key(candidate: _CertifiedCollocationCandidate) -> tuple[object, ...]:
            q_max_value = max(abs(value) for value in candidate.q)
            candidate_status = _chart_status(
                success=candidate.success,
                residual=candidate.residual,
                stability=candidate.stability,
                q_max=q_max_value,
                q_reference_max=q_reference_max,
                period_tolerance=period_tolerance,
                stability_tolerance=stability_tolerance,
            )
            return (
                status_order[candidate_status],
                q_max_value,
                candidate.residual,
                candidate.stability,
                candidate.seed_source,
            )

        candidate = min(attempts, key=candidate_key)
        success = candidate.success
        message = f"{candidate.seed_source}: {candidate.message}"
        if failures:
            message += "; other seeds: " + " | ".join(failures)
        nfev = candidate.nfev
        q_values = candidate.q
        log_q_values = candidate.log_q
        q_abs = tuple(abs(value) for value in q_values)
        forward = candidate.forward
        branch = candidate.branch
        residual = candidate.residual
        stability = candidate.stability
        max_multiplier = candidate.max_multiplier
        period_algorithm = candidate.algorithm
        period_map_region = candidate.region
        geometry_margin = candidate.geometry_margin
        overlap_residual = candidate.overlap_residual
        inverse_seed_source = candidate.seed_source
        if not is_schottky_algorithm(candidate.algorithm):
            collocation_basis_order = candidate.basis_order
            collocation_samples = candidate.samples_per_seam
            collocation_seam_residual = candidate.seam_residual
            collocation_symmetry_error = candidate.symmetry_error
        else:
            collocation_basis_order = None
            collocation_samples = None
            collocation_seam_residual = None
            collocation_symmetry_error = candidate.symmetry_error
        low_order = candidate.low_order
        high_order = candidate.high_order
    else:
        success = False
        message = "no inverse seed passed the hybrid period-map certificate"
        if failures:
            message += ": " + " | ".join(failures)
        nfev = 0
        q_values = marking.leading_q
        log_q_values = (
            marking.leading_log_q
            if marking.leading_log_q is not None
            else tuple(cmath.log(value) for value in marking.leading_q)
        )
        q_abs = marking.leading_q_abs
        forward = marking.omega
        branch = np.zeros((2, 2), dtype=int)
        residual = math.inf
        stability = math.inf
        max_multiplier = math.inf
        period_algorithm = "holomorphic-form-collocation-unsupported"
        period_map_region = "uncovered"
        geometry_margin = math.nan
        overlap_residual = None
        inverse_seed_source = "none"
        collocation_basis_order = None
        collocation_samples = None
        collocation_seam_residual = None
        collocation_symmetry_error = None
        low_order = 0
        high_order = 0

    q_max = max(q_abs)
    status = _chart_status(
        success=success,
        residual=residual,
        stability=stability,
        q_max=q_max,
        q_reference_max=q_reference_max,
        period_tolerance=period_tolerance,
        stability_tolerance=stability_tolerance,
    )
    return PlumbingChartResult(
        topology=marking.topology,
        word=marking.word,
        matrix=marking.matrix,
        modular_det_abs=float(abs(_symplectic_det_factor(matrix, source))),
        omega_chart=_format_matrix(marking.omega),
        integer_branch=_integer_matrix(branch),
        q=tuple(format_chart_coordinate(value) for value in q_values),  # type: ignore[arg-type]
        log_q=tuple(format_chart_coordinate(value) for value in log_q_values),  # type: ignore[arg-type]
        tau=tuple(format_chart_coordinate(value / TWO_PI_I) for value in log_q_values),  # type: ignore[arg-type]
        q_abs=q_abs,
        q_max=q_max,
        leading_q_abs=marking.leading_q_abs,
        leading_q_max=marking.leading_q_max,
        max_schottky_multiplier=max_multiplier,
        period_algorithm=period_algorithm,
        period_map_region=period_map_region,
        plumbing_geometry_margin=geometry_margin,
        period_overlap_residual=overlap_residual,
        inverse_seed_source=inverse_seed_source,
        collocation_basis_order=collocation_basis_order,
        collocation_samples_per_seam=collocation_samples,
        collocation_seam_residual=collocation_seam_residual,
        collocation_symmetry_error=collocation_symmetry_error,
        inverse_success=success,
        inverse_message=message,
        inverse_nfev=nfev,
        period_max_residual=residual,
        period_map_stability=stability,
        forward_word_stability=stability,
        word_length=low_order if is_schottky_algorithm(period_algorithm) else 0,
        stability_word_length=high_order if is_schottky_algorithm(period_algorithm) else 0,
        status=status,
    )


def table_first_all_small_chart(
    omega: Sequence[Sequence[complex]],
    *,
    period_table: Genus2PeriodMapTable | None,
    table_seed_count: int = 4,
    leading_search_depth: int = 3,
    leading_prefilter_count: int = 4,
    maximum_word: int = 5,
    maximum_corrections: int = 2,
    q_reference_max: float = 0.16,
    period_tolerance: float = 1.0e-6,
    stability_tolerance: float = 1.0e-6,
) -> PlumbingChartResult | None:
    """Return a directly certified deep-cusp table chart, if one is available.

    Nearby v3 table rows provide exact fundamental-to-plumbing markings and
    transported finite-q seeds.  A shallow leading-form search also supplies
    the correct cusp marking when the target lies beyond the table's minimum-q
    range; table neighbours in that marked chart are still considered before
    the asymptotic seed.  Candidates outside the genuine Schottky all-small
    region are skipped.  Failure is intentionally nonfatal so the caller can
    fall back to the complete mapping-class atlas.
    """

    if period_table is None or not period_table.has_fundamental_index:
        return None
    source = _as_omega(omega)
    table_markings = [
        marking
        for topology in ("theta", "glasses")
        for marking in fundamental_table_markings(
            source,
            topology,
            period_table=period_table,
            count=int(table_seed_count),
        )
        if marking.table_seed is not None
    ]
    leading_markings = [
        marking
        for topology in ("theta", "glasses")
        for marking in shortlist_markings(
            source,
            topology,
            search_depth=int(leading_search_depth),
            count=int(leading_prefilter_count),
        )
    ]
    candidates: list[tuple[LeadingMarking, PeriodMapSeed]] = []
    seen: set[tuple[object, ...]] = set()
    for marking in (*table_markings, *leading_markings):
        seeds = (
            (marking.table_seed,)
            if marking.table_seed is not None
            else inverse_seeds_for_marking(
                marking,
                period_table=period_table,
                table_seed_count=table_seed_count,
                include_leading_seed=True,
            )
        )
        for seed in seeds:
            if seed is None:
                continue
            key = (
                marking.topology,
                *tuple(value for row in marking.matrix for value in row),
                *tuple(
                    round(component, 12)
                    for value in seed.log_q
                    for component in (value.real, math.remainder(value.imag, 2.0 * math.pi))
                ),
            )
            if key in seen:
                continue
            seen.add(key)
            candidates.append((marking, seed))
    candidates.sort(
        key=lambda item: (
            math.exp(max(value.real for value in item[1].log_q)),
            0 if item[1].source.startswith("fundamental-period-table-v3:") else 1,
            math.inf if item[1].table_distance is None else float(item[1].table_distance),
            item[0].topology,
            item[0].word,
        )
    )
    for marking, seed in candidates:
        chart = certify_marking(
            source,
            marking,
            word_length=max(3, int(maximum_word) - 1),
            q_reference_max=q_reference_max,
            period_tolerance=period_tolerance,
            stability_tolerance=stability_tolerance,
            inverse_seeds=(seed,),
            table_first_schottky=True,
            table_first_only=True,
            table_first_maximum_word=maximum_word,
            table_first_maximum_corrections=maximum_corrections,
        )
        if chart.status in {"reference-q-envelope", "requires-recursion-order-study"}:
            return chart
    return None


def table_first_mixed_cusp_chart(
    omega: Sequence[Sequence[complex]],
    *,
    period_table: Genus2PeriodMapTable | None,
    leading_search_depth: int = 3,
    leading_prefilter_count: int = 4,
    table_seed_count: int = 4,
    maximum_candidates: int = 12,
    q_reference_max: float = 0.16,
    period_tolerance: float = 1.0e-5,
    stability_tolerance: float = 1.0e-5,
) -> PlumbingChartResult | None:
    """Try transported mixed-cusp seeds before the generic atlas search.

    The generic atlas historically entered many finite-q nonlinear inversions
    before reaching the one-small-q marking.  A portable table seed already
    identifies that regime, so rank those independent marking/seed pairs and
    promote them directly to the rescaled multiprecision solver.  Failure is
    nonfatal and leaves the complete atlas as the final fallback.
    """

    if period_table is None or not period_table.has_fundamental_index:
        return None
    source = _as_omega(omega)
    policy = HybridPeriodMapConfig(
        tolerance=min(float(period_tolerance), float(stability_tolerance)),
        agreement_tolerance=float(period_tolerance),
        maximum_collocation_basis=160,
    )
    markings = [
        marking
        for topology in ("theta", "glasses")
        for marking in (
            *shortlist_markings(
                source,
                topology,
                search_depth=int(leading_search_depth),
                count=int(leading_prefilter_count),
            ),
            *fundamental_table_markings(
                source,
                topology,
                period_table=period_table,
                count=int(table_seed_count),
            ),
        )
    ]
    candidates: list[tuple[LeadingMarking, PeriodMapSeed]] = []
    seen: set[tuple[object, ...]] = set()
    for marking in markings:
        for seed in inverse_seeds_for_marking(
            marking,
            period_table=period_table,
            table_seed_count=table_seed_count,
            include_leading_seed=True,
        ):
            try:
                region, _ = classify_period_map_region(
                    marking.topology,
                    seed.q,
                    config=policy,
                    log_q_values=seed.log_q,
                )
            except Exception:
                continue
            if region != "holomorphic-mixed-cusp":
                continue
            key = (
                marking.topology,
                *tuple(value for row in marking.matrix for value in row),
                *tuple(
                    round(component, 12)
                    for value in seed.log_q
                    for component in (
                        value.real,
                        math.remainder(value.imag, 2.0 * math.pi),
                    )
                ),
            )
            if key in seen:
                continue
            seen.add(key)
            candidates.append((marking, seed))
    candidates.sort(
        key=lambda item: (
            item[0].leading_q_max,
            math.inf
            if item[1].table_distance is None
            else float(item[1].table_distance),
            item[0].topology,
            item[0].word,
            item[1].source,
        )
    )
    for marking, seed in candidates[: max(1, int(maximum_candidates))]:
        chart = certify_marking(
            source,
            marking,
            word_length=6,
            max_nfev=36,
            q_reference_max=q_reference_max,
            period_tolerance=period_tolerance,
            stability_tolerance=stability_tolerance,
            inverse_seeds=(seed,),
        )
        if chart.status in {"reference-q-envelope", "requires-recursion-order-study"}:
            return chart
    return None


def build_plumbing_atlas(
    omega: Sequence[Sequence[complex]],
    *,
    search_depth: int = 3,
    prefilter_count: int = 4,
    word_length: int = 5,
    stability_step: int = 1,
    max_nfev: int = 120,
    q_reference_max: float = 0.16,
    period_tolerance: float = 2.0e-6,
    stability_tolerance: float = 2.0e-6,
    stop_at_reference: bool = False,
    stop_at_usable_q_max: float | None = None,
    early_stop_competitive_score_ratio: float = 1.02,
    period_table: HolomorphicPeriodMapTable | Genus2PeriodMapTable | None = None,
    table_seed_count: int = 4,
    include_leading_seed: bool = True,
) -> PlumbingAtlasResult:
    """Build a hybrid-certified theta/glasses chart atlas.

    If supplied, ``period_table`` provides nearest-neighbour initial values.
    Every returned candidate is nevertheless recomputed and certified by the
    live hybrid period map.
    """

    source = _as_omega(omega)
    charts: list[PlumbingChartResult] = []
    markings: list[LeadingMarking] = []
    seen_markings: set[tuple[object, ...]] = set()
    early_stop_found = False
    for topology in ("theta", "glasses"):
        table_markings = (
            fundamental_table_markings(
                source,
                topology,
                period_table=(
                    period_table
                    if isinstance(period_table, Genus2PeriodMapTable)
                    else None
                ),
                count=table_seed_count,
            )
            if period_table is not None
            else []
        )
        enumerated_markings = shortlist_markings(
            source, topology, search_depth=search_depth, count=prefilter_count
        )
        for candidate in (*table_markings, *enumerated_markings):
            matrix_key = tuple(value for row in candidate.matrix for value in row)
            # The same homology marking defines distinct theta and glasses
            # pants charts, so topology is part of the de-duplication key.
            marking_key = (candidate.topology, *matrix_key)
            if marking_key in seen_markings:
                continue
            seen_markings.add(marking_key)
            markings.append(candidate)

    # Put both topologies into one queue before any nonlinear solve.  This
    # preserves the inexpensive early-stop policy while removing its former
    # theta-first selection bias.
    markings.sort(key=_atlas_marking_search_key)
    competitive_topologies = _competitive_early_stop_topologies(
        markings,
        score_ratio=early_stop_competitive_score_ratio,
    )
    markings = _prioritize_competitive_topology_heads(
        markings,
        competitive_topologies,
    )
    attempted_topologies: set[Topology] = set()
    for marking in markings:
        inverse_seeds = inverse_seeds_for_marking(
            marking,
            period_table=period_table,
            table_seed_count=table_seed_count,
            include_leading_seed=include_leading_seed,
        )
        chart = certify_marking(
            source,
            marking,
            word_length=word_length,
            stability_step=stability_step,
            max_nfev=max_nfev,
            q_reference_max=q_reference_max,
            period_tolerance=period_tolerance,
            stability_tolerance=stability_tolerance,
            inverse_seeds=inverse_seeds,
        )
        charts.append(chart)
        attempted_topologies.add(marking.topology)
        stop_candidate_found = any(
            (
                stop_at_reference
                and item.status == "reference-q-envelope"
            )
            or (
                stop_at_usable_q_max is not None
                and item.status
                in {"reference-q-envelope", "requires-recursion-order-study"}
                and item.q_max <= float(stop_at_usable_q_max)
            )
            for item in charts
        )
        if stop_candidate_found and competitive_topologies <= attempted_topologies:
            early_stop_found = True
            break

    status_order = {
        "reference-q-envelope": 0,
        "requires-recursion-order-study": 1,
        "period-map-unstable": 2,
        "inverse-failed": 3,
    }
    charts.sort(key=lambda item: (status_order[item.status], item.q_max, item.period_max_residual, item.word))
    usable = [
        item
        for item in charts
        if item.status in {"reference-q-envelope", "requires-recursion-order-study"}
    ]
    best = usable[0] if usable else None
    if best is None:
        coverage_status = "uncovered-at-current-search-settings"
    elif best.status == "reference-q-envelope":
        coverage_status = "period-chart-inside-reference-q-envelope"
    else:
        coverage_status = "period-chart-found-but-block-order-unvalidated"

    return PlumbingAtlasResult(
        omega=_format_matrix(source),
        search_depth=int(search_depth),
        prefilter_count=int(prefilter_count),
        q_reference_max=float(q_reference_max),
        period_tolerance=float(period_tolerance),
        stability_tolerance=float(stability_tolerance),
        best_topology=None if best is None else best.topology,
        best_q_max=None if best is None else best.q_max,
        coverage_status=coverage_status,
        charts=tuple(charts),
        note=(
            "Theta and glasses are the two genus-two trivalent pants-graph topologies. "
            "The finite symplectic search supplies homology-marking images, but does not yet "
            "enumerate Torelli-distinct pants decompositions or multiple inverse-period roots. "
            "Final q-to-Omega evaluations use normalized holomorphic forms in the "
            "bulk, rescaled multiprecision holomorphic forms in mixed cusps, and "
            "adaptive Schottky words only when all three q values are small; the "
            "shared method-boundary policy enforces numerical agreement. "
            + (
                "Certified table rows supplied initial q values and were freshly refined. "
                if period_table is not None
                else "The degeneration formula supplied the initial q values. "
            )
            + (
                "Theta and glasses candidates were ranked together by their leading q score; "
                "near-tied topologies were both checked before certification stopped at the "
                "requested q threshold. "
                if early_stop_found
                else ""
            )
            + "The q<=0.16 label records the existing c=25 order-12 radial benchmark envelope; "
            "it is not a theorem or a replacement for order doubling at each integration node."
        ),
    )


def _omega_from_args(args: argparse.Namespace) -> np.ndarray:
    if args.bolza:
        return bolza_period_matrix()
    return np.asarray(
        [
            [args.omega11, args.omega12],
            [args.omega12, args.omega22],
        ],
        dtype=np.complex128,
    )


def _load_period_seed_table(path: Path):
    """Load either the older holomorphic-only or mixed-backend table schema."""

    import csv

    with path.open(newline="") as handle:
        fieldnames = set(csv.DictReader(handle).fieldnames or ())
    if {"result_schema_version", "actual_backend"}.issubset(fieldnames):
        return Genus2PeriodMapTable.from_csv(path)
    return HolomorphicPeriodMapTable.from_csv(path)


def run(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Find efficient genus-two plumbing charts for a period matrix.")
    parser.add_argument("--bolza", action="store_true", help="use the standard Bolza period matrix")
    parser.add_argument("--omega11", type=parse_complex, default=1.0j)
    parser.add_argument("--omega12", type=parse_complex, default=0.2j)
    parser.add_argument("--omega22", type=parse_complex, default=1.0j)
    parser.add_argument("--search-depth", type=int, default=3)
    parser.add_argument("--prefilter-count", type=int, default=4)
    parser.add_argument(
        "--word-length",
        type=int,
        default=5,
        help="deprecated compatibility option; no Schottky period map is evaluated",
    )
    parser.add_argument(
        "--stability-step",
        type=int,
        default=1,
        help="deprecated compatibility option; collocation uses adaptive basis orders",
    )
    parser.add_argument("--max-nfev", type=int, default=120)
    parser.add_argument("--q-reference-max", type=float, default=0.16)
    parser.add_argument("--period-tolerance", type=float, default=2.0e-6)
    parser.add_argument("--stability-tolerance", type=float, default=2.0e-6)
    parser.add_argument(
        "--period-table",
        type=Path,
        help="optional certified q-to-Omega table used only for inverse seeds",
    )
    parser.add_argument("--table-seed-count", type=int, default=4)
    parser.add_argument(
        "--table-only-seeds",
        action="store_true",
        help="do not append the leading plumbing seed when a period table is supplied",
    )
    parser.add_argument("--out-json", type=Path)
    args = parser.parse_args(argv)

    period_table = (
        None
        if args.period_table is None
        else _load_period_seed_table(args.period_table)
    )

    result = build_plumbing_atlas(
        _omega_from_args(args),
        search_depth=args.search_depth,
        prefilter_count=args.prefilter_count,
        word_length=args.word_length,
        stability_step=args.stability_step,
        max_nfev=args.max_nfev,
        q_reference_max=args.q_reference_max,
        period_tolerance=args.period_tolerance,
        stability_tolerance=args.stability_tolerance,
        period_table=period_table,
        table_seed_count=args.table_seed_count,
        include_leading_seed=not args.table_only_seeds,
    )
    print("Genus-two plumbing atlas")
    print(f"  coverage={result.coverage_status}")
    print(f"  best topology={result.best_topology}, best max |q|={result.best_q_max}")
    for index, chart in enumerate(result.charts, start=1):
        print(
            f"  {index:2d}. {chart.topology:7s} {chart.status:36s} "
            f"max|q|={chart.q_max:.6g} residual={chart.period_max_residual:.3e} "
            f"map-step={chart.period_map_stability:.3e} "
            f"algorithm={chart.period_algorithm} word={chart.word}"
        )

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(asdict(result), indent=2) + "\n")
        print(f"  wrote {args.out_json}")


if __name__ == "__main__":
    run()
