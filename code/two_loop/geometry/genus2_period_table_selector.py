#!/usr/bin/env python3
"""Atlas-aware q selector for the genus-two period table.

Targets are drawn from the invariant Siegel measure on Gottschling's
fundamental domain.  For every target this module searches finite-depth
symplectic markings in both plumbing topologies, retains the best valid chart,
and also retains near-best charts in a controlled overlap band.  It returns q
coordinates only; no finite-q period map is evaluated here.
"""

from __future__ import annotations

from c1_paths import PACKAGE_ROOT, data_root, package_path, module_path

import cmath
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterator, Literal

import numpy as np
from scipy.stats import qmc

try:
    from bolza_torus_plumbing_reach import enumerate_symplectic_words
    from genus2_hybrid_period_map import plumbing_geometry
    from genus2_siegel_fundamental_domain import (
        INVARIANT_WEIGHT_MAX,
        in_gottschling_domain,
        minkowski_proposals_from_unit_cube,
    )
except ImportError:  # pragma: no cover - package-style execution
    from bolza_torus_plumbing_reach import enumerate_symplectic_words
    from genus2_hybrid_period_map import plumbing_geometry
    from genus2_siegel_fundamental_domain import (
        INVARIANT_WEIGHT_MAX,
        in_gottschling_domain,
        minkowski_proposals_from_unit_cube,
    )


Topology = Literal["theta", "glasses"]
TWO_PI_I = 2.0j * math.pi


@dataclass(frozen=True)
class SelectedAtlasPoint:
    target_index: int
    target_omega: np.ndarray
    topology: Topology
    marking_word: str
    marking_matrix_index: int
    q: tuple[complex, complex, complex]
    log_q: tuple[complex, complex, complex]
    leading_q_max: float
    best_valid_q_max: float
    score_ratio: float
    overlap_chart: bool
    geometry_margin: float
    search_depth: int
    tail_refined: bool


@lru_cache(maxsize=None)
def _words_and_matrices(depth: int) -> tuple[tuple[str, np.ndarray], ...]:
    return tuple(enumerate_symplectic_words(int(depth)))


def _rqmc_invariant_targets(count: int, *, seed: int) -> np.ndarray:
    """Generate deterministic randomized-QMC invariant-domain targets."""

    count = int(count)
    if count < 1:
        raise ValueError("atlas target count must be positive")
    # The seventh Sobol coordinate performs the rejection step.  Increase the
    # proposal power deterministically until enough invariant points survive.
    power = max(8, int(math.ceil(math.log2(3.5 * count))))
    while True:
        unit = qmc.Sobol(7, scramble=True, seed=int(seed)).random_base2(power)
        omega, weight, _ = minkowski_proposals_from_unit_cube(unit[:, :6])
        domain = np.asarray(in_gottschling_domain(omega), dtype=bool)
        accepted = domain & (unit[:, 6] < weight / INVARIANT_WEIGHT_MAX)
        selected = omega[accepted]
        if len(selected) >= count:
            return np.asarray(selected[:count], dtype=np.complex128)
        power += 1


def _transform_batch(matrix: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """Apply one Sp(4,Z) matrix to a batch of 2x2 period matrices."""

    a = np.asarray(matrix[:2, :2], dtype=np.float64)
    b = np.asarray(matrix[:2, 2:], dtype=np.float64)
    c = np.asarray(matrix[2:, :2], dtype=np.float64)
    d = np.asarray(matrix[2:, 2:], dtype=np.float64)
    numerator = np.einsum("ij,njk->nik", a, omega) + b
    denominator = np.einsum("ij,njk->nik", c, omega) + d
    determinant = denominator[:, 0, 0] * denominator[:, 1, 1] - denominator[:, 0, 1] * denominator[:, 1, 0]
    inverse = np.empty_like(denominator)
    inverse[:, 0, 0] = denominator[:, 1, 1] / determinant
    inverse[:, 0, 1] = -denominator[:, 0, 1] / determinant
    inverse[:, 1, 0] = -denominator[:, 1, 0] / determinant
    inverse[:, 1, 1] = denominator[:, 0, 0] / determinant
    transformed = np.einsum("nij,njk->nik", numerator, inverse)
    return 0.5 * (transformed + np.swapaxes(transformed, 1, 2))


def _log_scores(transformed: np.ndarray, topology: Topology) -> np.ndarray:
    if topology == "theta":
        logs = np.stack(
            (
                TWO_PI_I * (transformed[:, 0, 0] - transformed[:, 0, 1]),
                TWO_PI_I * (transformed[:, 1, 1] - transformed[:, 0, 1]),
                TWO_PI_I * transformed[:, 0, 1],
            ),
            axis=1,
        )
        valid = np.all(np.isfinite(logs), axis=1) & np.all(logs.real < 0.0, axis=1)
        score = np.max(logs.real, axis=1)
    else:
        handle_logs = np.stack(
            (TWO_PI_I * transformed[:, 0, 0], TWO_PI_I * transformed[:, 1, 1]),
            axis=1,
        )
        bridge = -TWO_PI_I * transformed[:, 0, 1]
        bridge_abs = np.abs(bridge)
        valid = (
            np.all(np.isfinite(handle_logs), axis=1)
            & np.all(handle_logs.real < 0.0, axis=1)
            & np.isfinite(bridge_abs)
            & (bridge_abs > 0.0)
            & (bridge_abs < 1.0)
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            score = np.maximum(np.max(handle_logs.real, axis=1), np.log(bridge_abs))
    return np.where(valid, score, math.inf)


def _insert_top_k(
    scores: np.ndarray,
    indices: np.ndarray,
    candidate: np.ndarray,
    matrix_index: int,
) -> None:
    """Insert one score vector into sorted per-target top-k arrays."""

    carry_score = candidate.copy()
    carry_index = np.full(len(candidate), int(matrix_index), dtype=np.int32)
    for rank in range(scores.shape[1]):
        replace = carry_score < scores[:, rank]
        old_score = scores[:, rank].copy()
        old_index = indices[:, rank].copy()
        scores[replace, rank] = carry_score[replace]
        indices[replace, rank] = carry_index[replace]
        carry_score[replace] = old_score[replace]
        carry_index[replace] = old_index[replace]


def _search_top_k(
    omega: np.ndarray,
    *,
    depth: int,
    count: int,
) -> dict[Topology, tuple[np.ndarray, np.ndarray]]:
    words = _words_and_matrices(int(depth))
    result: dict[Topology, tuple[np.ndarray, np.ndarray]] = {}
    for topology in ("theta", "glasses"):
        scores = np.full((len(omega), int(count)), math.inf, dtype=np.float64)
        indices = np.full((len(omega), int(count)), -1, dtype=np.int32)
        for matrix_index, (_, matrix) in enumerate(words):
            transformed = _transform_batch(matrix, omega)
            _insert_top_k(scores, indices, _log_scores(transformed, topology), matrix_index)
        result[topology] = (scores, indices)
    return result


def _logs_and_q(
    omega: np.ndarray,
    topology: Topology,
    *,
    log_floor: float,
) -> tuple[tuple[complex, complex, complex], tuple[complex, complex, complex]]:
    if topology == "theta":
        raw_logs = (
            complex(TWO_PI_I * (omega[0, 0] - omega[0, 1])),
            complex(TWO_PI_I * (omega[1, 1] - omega[0, 1])),
            complex(TWO_PI_I * omega[0, 1]),
        )
    else:
        bridge = complex(-TWO_PI_I * omega[0, 1])
        if bridge == 0.0j:
            raise ValueError("leading glasses bridge vanished")
        raw_logs = (
            complex(TWO_PI_I * omega[0, 0]),
            complex(TWO_PI_I * omega[1, 1]),
            cmath.log(bridge),
        )
    logs = tuple(complex(max(value.real, float(log_floor)), value.imag) for value in raw_logs)
    q_values = tuple(cmath.exp(value) for value in logs)
    return logs, q_values  # type: ignore[return-value]


def _candidate_points_for_target(
    target_index: int,
    target: np.ndarray,
    searches: dict[Topology, tuple[np.ndarray, np.ndarray]],
    local_index: int,
    *,
    depth: int,
    q_abs_max: float,
    q_abs_floor: float,
    overlap_ratio: float,
    overlap_q_abs_max: float,
    minimum_geometry_margin: float,
    tail_refined: bool,
) -> list[SelectedAtlasPoint]:
    provisional: list[tuple[float, Topology, int, np.ndarray, tuple[complex, ...], tuple[complex, ...], float]] = []
    for topology in ("theta", "glasses"):
        scores, matrix_indices = searches[topology]
        words = _words_and_matrices(int(depth))
        for log_score, matrix_index in zip(scores[local_index], matrix_indices[local_index]):
            if matrix_index < 0 or not math.isfinite(float(log_score)):
                continue
            word, matrix = words[int(matrix_index)]
            transformed = _transform_batch(matrix, target[np.newaxis, ...])[0]
            try:
                logs, q_values = _logs_and_q(
                    transformed, topology, log_floor=math.log(float(q_abs_floor))
                )
                geometry = plumbing_geometry(topology, q_values, log_q_values=logs)
            except (ValueError, OverflowError):
                continue
            q_max = math.exp(float(log_score))
            if q_max > float(q_abs_max) or geometry.minimum_margin < float(minimum_geometry_margin):
                continue
            provisional.append(
                (
                    float(log_score),
                    topology,
                    int(matrix_index),
                    transformed,
                    logs,
                    q_values,
                    float(geometry.minimum_margin),
                )
            )
    if not provisional:
        return []
    provisional.sort(key=lambda item: (item[0], item[1], item[2]))
    best_log_score = provisional[0][0]
    selected: list[SelectedAtlasPoint] = []
    seen: set[tuple[object, ...]] = set()
    for log_score, topology, matrix_index, _, logs, q_values, margin in provisional:
        if log_score > best_log_score + math.log(float(overlap_ratio)):
            continue
        if log_score > best_log_score + 1.0e-14 and math.exp(log_score) > float(
            overlap_q_abs_max
        ):
            continue
        key = (
            topology,
            *(round(value.real, 13) for value in logs),
            *(round(math.remainder(value.imag, 2.0 * math.pi), 13) for value in logs),
        )
        if key in seen:
            continue
        seen.add(key)
        q_max = math.exp(log_score)
        selected.append(
            SelectedAtlasPoint(
                target_index=int(target_index),
                target_omega=np.asarray(target, dtype=np.complex128),
                topology=topology,
                marking_word=_words_and_matrices(int(depth))[matrix_index][0],
                marking_matrix_index=int(matrix_index),
                q=tuple(complex(value) for value in q_values),  # type: ignore[arg-type]
                log_q=tuple(complex(value) for value in logs),  # type: ignore[arg-type]
                leading_q_max=float(q_max),
                best_valid_q_max=float(math.exp(best_log_score)),
                score_ratio=float(math.exp(log_score - best_log_score)),
                overlap_chart=bool(log_score > best_log_score + 1.0e-14),
                geometry_margin=float(margin),
                search_depth=int(depth),
                tail_refined=bool(tail_refined),
            )
        )
    return selected


def iter_selected_atlas_points(payload: dict[str, object]) -> Iterator[SelectedAtlasPoint]:
    """Yield the selected theta/glasses chart union without evaluating Omega(q)."""

    design = payload["atlas_design"]  # type: ignore[index]
    domain = payload["q_domain"]  # type: ignore[index]
    count = 1 << int(design["target_sample_power"])
    targets = _rqmc_invariant_targets(count, seed=int(payload["seed"]))
    batch_size = int(design["selector_batch_size"])
    primary_depth = int(design["search_depth"])
    tail_depth = int(design["tail_search_depth"])
    rescue_depth = int(design["rescue_search_depth"])
    top_k = int(design["markings_per_topology"])
    tail_top_k = int(design["tail_markings_per_topology"])
    rescue_top_k = int(design["rescue_markings_per_topology"])
    tail_trigger = float(design["tail_refine_q_abs_min"])
    common = {
        "q_abs_max": float(domain["q_abs_max"]),
        "q_abs_floor": float(domain["q_abs_tail_min"]),
        "overlap_ratio": float(design["overlap_score_ratio"]),
        "overlap_q_abs_max": float(design["overlap_q_abs_max"]),
        "minimum_geometry_margin": float(design["minimum_selector_geometry_margin"]),
    }
    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        batch = targets[start:stop]
        primary = _search_top_k(batch, depth=primary_depth, count=top_k)
        selected_by_local: list[list[SelectedAtlasPoint]] = []
        tail_local_indices: list[int] = []
        for local_index, target in enumerate(batch):
            selected = _candidate_points_for_target(
                start + local_index,
                target,
                primary,
                local_index,
                depth=primary_depth,
                tail_refined=False,
                **common,
            )
            best = min((point.best_valid_q_max for point in selected), default=math.inf)
            if best >= tail_trigger and tail_depth > primary_depth:
                tail_local_indices.append(local_index)
            selected_by_local.append(selected)
        if tail_local_indices:
            tail_targets = batch[np.asarray(tail_local_indices, dtype=int)]
            refined = _search_top_k(
                tail_targets, depth=tail_depth, count=tail_top_k
            )
            for tail_position, local_index in enumerate(tail_local_indices):
                selected_by_local[local_index] = _candidate_points_for_target(
                    start + local_index,
                    batch[local_index],
                    refined,
                    tail_position,
                    depth=tail_depth,
                    tail_refined=True,
                    **common,
                )
        rescue_local_indices = [
            local_index
            for local_index, selected in enumerate(selected_by_local)
            if not selected
        ]
        if rescue_local_indices and rescue_depth > tail_depth:
            rescue_targets = batch[np.asarray(rescue_local_indices, dtype=int)]
            rescued = _search_top_k(
                rescue_targets, depth=rescue_depth, count=rescue_top_k
            )
            for rescue_position, local_index in enumerate(rescue_local_indices):
                selected_by_local[local_index] = _candidate_points_for_target(
                    start + local_index,
                    batch[local_index],
                    rescued,
                    rescue_position,
                    depth=rescue_depth,
                    tail_refined=True,
                    **common,
                )
        for selected in selected_by_local:
            yield from selected
