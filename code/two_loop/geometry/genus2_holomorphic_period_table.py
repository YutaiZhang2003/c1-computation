#!/usr/bin/env python3
"""Reference table and calibrated Schottky envelope for the genus-two period map.

The table itself will be generated separately.  This module fixes the data
contract needed by the inverse atlas: every usable row must have been produced
and certified with normalized holomorphic one-forms.  A table lookup supplies
only initial ``q`` values; the atlas still performs a fresh collocation inverse
and higher-order certificate at the requested period matrix.

The same reference rows may later be used to construct Schottky-validity
cells.  A Schottky evaluation is admissible only when its query lies inside a
certified cell and that cell's conservative error bound is below the requested
tolerance.  In particular, neither ``min|q|`` nor ``max|q|`` alone activates a
Schottky backend.
"""

from __future__ import annotations

import cmath
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Sequence

import numpy as np


Topology = Literal["theta", "glasses"]
HOLOMORPHIC_PERIOD_ALGORITHM = "holomorphic-form-collocation"
CALIBRATED_SCHOTTKY_ALGORITHM = "calibrated-schottky"


def _parse_complex(value: object) -> complex:
    return complex(str(value).strip().replace("i", "j"))


def _symmetric_period_matrix(
    omega11: complex,
    omega12: complex,
    omega22: complex,
) -> np.ndarray:
    return np.asarray(
        [[complex(omega11), complex(omega12)], [complex(omega12), complex(omega22)]],
        dtype=np.complex128,
    )


def _period_distance_mod_integer(left: np.ndarray, right: np.ndarray) -> float:
    """Relative symmetric-coordinate distance after integral B shifts."""

    difference = np.asarray(left, dtype=np.complex128) - np.asarray(
        right, dtype=np.complex128
    )
    branch = np.rint(difference.real).astype(int)
    branch = np.rint(0.5 * (branch + branch.T)).astype(int)
    difference -= branch
    vector = np.asarray(
        [difference[0, 0], difference[0, 1], difference[1, 1]],
        dtype=np.complex128,
    )
    target = np.asarray(
        [right[0, 0], right[0, 1], right[1, 1]],
        dtype=np.complex128,
    )
    return float(np.linalg.norm(vector) / max(float(np.linalg.norm(target)), 1.0))


@dataclass(frozen=True)
class PeriodMapSeed:
    """One initial value for a fresh holomorphic-form inverse solve."""

    q: tuple[complex, complex, complex]
    log_q: tuple[complex, complex, complex]
    source: str
    table_distance: float | None = None


@dataclass(frozen=True)
class HolomorphicPeriodTableEntry:
    """One certified ``q -> Omega`` table row."""

    row_id: str
    topology: Topology
    q: tuple[complex, complex, complex]
    omega: np.ndarray
    basis_order: int
    samples_per_seam: int
    basis_stability: float
    seam_residual: float
    symmetry_error: float
    geometry_margin: float
    schottky_word_length: int | None = None
    schottky_reference_residual: float | None = None
    schottky_word_stability: float | None = None
    schottky_symmetry_error: float | None = None
    period_algorithm: str = HOLOMORPHIC_PERIOD_ALGORITHM
    certified: bool = True

    def __post_init__(self) -> None:
        if self.topology not in {"theta", "glasses"}:
            raise ValueError(f"unknown plumbing topology {self.topology!r}")
        if len(self.q) != 3 or any(
            not (
                math.isfinite(complex(value).real)
                and math.isfinite(complex(value).imag)
                and 0.0 < abs(complex(value)) < 1.0
            )
            for value in self.q
        ):
            raise ValueError("a period-table row must contain three finite 0<|q|<1 values")
        omega = np.asarray(self.omega, dtype=np.complex128)
        if omega.shape != (2, 2) or not np.all(np.isfinite(omega)):
            raise ValueError("a period-table row must contain a finite 2x2 period matrix")
        if float(np.max(np.abs(omega - omega.T))) > 1.0e-10:
            raise ValueError("a period-table period matrix must be symmetric")
        if self.period_algorithm != HOLOMORPHIC_PERIOD_ALGORITHM:
            raise ValueError(
                "production period-table rows must use normalized holomorphic one-forms"
            )
        if self.basis_order < 2 or self.samples_per_seam < 2 * self.basis_order:
            raise ValueError("invalid period-table basis or seam-sampling order")
        diagnostics = (
            self.basis_stability,
            self.seam_residual,
            self.symmetry_error,
            self.geometry_margin,
        )
        if any(not math.isfinite(float(value)) for value in diagnostics):
            raise ValueError("period-table diagnostics must be finite")
        schottky_diagnostics = (
            self.schottky_reference_residual,
            self.schottky_word_stability,
            self.schottky_symmetry_error,
        )
        supplied = [value is not None for value in schottky_diagnostics]
        if self.schottky_word_length is not None and not all(supplied):
            raise ValueError("a Schottky word length needs complete comparison diagnostics")
        if any(supplied) and not all(supplied):
            raise ValueError("Schottky comparison diagnostics must be all present or absent")
        if all(supplied):
            if self.schottky_word_length is None or self.schottky_word_length < 2:
                raise ValueError("a Schottky comparison needs a positive word length")
            if any(
                not math.isfinite(float(value)) or float(value) < 0.0
                for value in schottky_diagnostics
                if value is not None
            ):
                raise ValueError("Schottky comparison diagnostics must be finite and nonnegative")

    @classmethod
    def from_csv_row(cls, row: dict[str, str]) -> "HolomorphicPeriodTableEntry":
        return cls(
            row_id=str(row["row_id"]),
            topology=str(row["topology"]),  # type: ignore[arg-type]
            q=tuple(_parse_complex(row[f"q{index}"]) for index in (1, 2, 3)),  # type: ignore[arg-type]
            omega=_symmetric_period_matrix(
                _parse_complex(row["omega11"]),
                _parse_complex(row["omega12"]),
                _parse_complex(row["omega22"]),
            ),
            basis_order=int(row["basis_order"]),
            samples_per_seam=int(row["samples_per_seam"]),
            basis_stability=float(row["basis_stability"]),
            seam_residual=float(row["seam_residual"]),
            symmetry_error=float(row["symmetry_error"]),
            geometry_margin=float(row["geometry_margin"]),
            schottky_word_length=(
                None
                if not row.get("schottky_word_length", "").strip()
                else int(row["schottky_word_length"])
            ),
            schottky_reference_residual=(
                None
                if not row.get("schottky_reference_residual", "").strip()
                else float(row["schottky_reference_residual"])
            ),
            schottky_word_stability=(
                None
                if not row.get("schottky_word_stability", "").strip()
                else float(row["schottky_word_stability"])
            ),
            schottky_symmetry_error=(
                None
                if not row.get("schottky_symmetry_error", "").strip()
                else float(row["schottky_symmetry_error"])
            ),
            period_algorithm=str(row["period_algorithm"]),
            certified=str(row["certified"]).strip().lower() in {"1", "true", "yes"},
        )


class HolomorphicPeriodMapTable:
    """In-memory nearest-neighbour index for certified period-map rows."""

    def __init__(self, entries: Iterable[HolomorphicPeriodTableEntry]):
        self.entries = tuple(entries)

    @classmethod
    def from_csv(cls, path: Path | str) -> "HolomorphicPeriodMapTable":
        with Path(path).open(newline="") as handle:
            return cls(
                HolomorphicPeriodTableEntry.from_csv_row(row)
                for row in csv.DictReader(handle)
            )

    def nearest_seeds(
        self,
        topology: Topology,
        target_omega: Sequence[Sequence[complex]],
        *,
        count: int = 4,
    ) -> tuple[PeriodMapSeed, ...]:
        """Return nearest certified rows in the requested marked topology."""

        target = np.asarray(target_omega, dtype=np.complex128)
        if target.shape != (2, 2):
            raise ValueError(f"target period matrix must be 2x2, got {target.shape}")
        ranked: list[tuple[float, HolomorphicPeriodTableEntry]] = []
        for entry in self.entries:
            if (
                entry.topology != topology
                or not entry.certified
                or entry.period_algorithm != HOLOMORPHIC_PERIOD_ALGORITHM
                or entry.geometry_margin <= 0.0
            ):
                continue
            ranked.append((_period_distance_mod_integer(entry.omega, target), entry))
        ranked.sort(key=lambda item: (item[0], item[1].row_id))

        seeds: list[PeriodMapSeed] = []
        seen: set[tuple[float, ...]] = set()
        for distance, entry in ranked:
            key = tuple(
                round(component, 14)
                for value in entry.q
                for component in (complex(value).real, complex(value).imag)
            )
            if key in seen:
                continue
            seen.add(key)
            q = tuple(complex(value) for value in entry.q)
            seeds.append(
                PeriodMapSeed(
                    q=q,  # type: ignore[arg-type]
                    log_q=tuple(cmath.log(value) for value in q),  # type: ignore[arg-type]
                    source=f"holomorphic-period-table:{entry.row_id}",
                    table_distance=float(distance),
                )
            )
            if len(seeds) >= int(count):
                break
        return tuple(seeds)


def _phase_distance(left: float, right: float) -> float:
    return abs((float(left) - float(right) + math.pi) % (2.0 * math.pi) - math.pi)


@dataclass(frozen=True)
class SchottkyValidityCell:
    """One six-real-dimensional cell calibrated against holomorphic forms."""

    cell_id: str
    topology: Topology
    center_q: tuple[complex, complex, complex]
    log_abs_radius: tuple[float, float, float]
    phase_radius: tuple[float, float, float]
    word_length: int
    validation_point_count: int
    boundary_point_count: int
    interior_point_count: int
    reference_table_sha256: str
    max_reference_residual: float
    max_word_stability: float
    max_symmetry_error: float
    min_geometry_margin: float
    safety_factor: float = 2.0
    certified: bool = True

    def __post_init__(self) -> None:
        if self.topology not in {"theta", "glasses"}:
            raise ValueError(f"unknown plumbing topology {self.topology!r}")
        if len(self.center_q) != 3 or any(
            not (
                math.isfinite(complex(value).real)
                and math.isfinite(complex(value).imag)
                and 0.0 < abs(complex(value)) < 1.0
            )
            for value in self.center_q
        ):
            raise ValueError("a Schottky cell needs three finite 0<|q|<1 centers")
        if len(self.log_abs_radius) != 3 or any(
            not math.isfinite(float(value)) or float(value) <= 0.0
            for value in self.log_abs_radius
        ):
            raise ValueError("all Schottky-cell log-modulus radii must be positive")
        if len(self.phase_radius) != 3 or any(
            not math.isfinite(float(value)) or not (0.0 < float(value) <= math.pi)
            for value in self.phase_radius
        ):
            raise ValueError("all Schottky-cell phase radii must lie in (0, pi]")
        if self.word_length < 2 or self.validation_point_count < 1:
            raise ValueError("invalid Schottky word length or validation-point count")
        if (
            self.boundary_point_count < 1
            or self.interior_point_count < 1
            or self.boundary_point_count + self.interior_point_count
            != self.validation_point_count
        ):
            raise ValueError("Schottky cells need consistent boundary/interior counts")
        digest = self.reference_table_sha256.strip().lower()
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("Schottky cells must identify the reference table by SHA-256")
        diagnostics = (
            self.max_reference_residual,
            self.max_word_stability,
            self.max_symmetry_error,
        )
        if any(not math.isfinite(float(value)) or float(value) < 0.0 for value in diagnostics):
            raise ValueError("Schottky-cell errors must be finite and nonnegative")
        if not math.isfinite(self.min_geometry_margin) or self.min_geometry_margin <= 0.0:
            raise ValueError("a Schottky cell needs a positive calibrated geometry margin")
        if not math.isfinite(self.safety_factor) or self.safety_factor < 1.0:
            raise ValueError("Schottky-cell safety factor must be at least one")

    @property
    def error_bound(self) -> float:
        return float(
            self.safety_factor
            * max(
                self.max_reference_residual,
                self.max_word_stability,
                self.max_symmetry_error,
            )
        )

    def contains(self, q_values: Sequence[complex]) -> bool:
        q = tuple(complex(value) for value in q_values)
        if len(q) != 3 or any(not (0.0 < abs(value) < 1.0) for value in q):
            return False
        for value, center, modulus_radius, angle_radius in zip(
            q,
            self.center_q,
            self.log_abs_radius,
            self.phase_radius,
        ):
            if abs(math.log(abs(value)) - math.log(abs(center))) > modulus_radius:
                return False
            if _phase_distance(cmath.phase(value), cmath.phase(center)) > angle_radius:
                return False
        return True

    @classmethod
    def from_csv_row(cls, row: dict[str, str]) -> "SchottkyValidityCell":
        return cls(
            cell_id=str(row["cell_id"]),
            topology=str(row["topology"]),  # type: ignore[arg-type]
            center_q=tuple(
                _parse_complex(row[f"center_q{index}"]) for index in (1, 2, 3)
            ),  # type: ignore[arg-type]
            log_abs_radius=tuple(
                float(row[f"log_abs_radius{index}"]) for index in (1, 2, 3)
            ),  # type: ignore[arg-type]
            phase_radius=tuple(
                float(row[f"phase_radius{index}"]) for index in (1, 2, 3)
            ),  # type: ignore[arg-type]
            word_length=int(row["word_length"]),
            validation_point_count=int(row["validation_point_count"]),
            boundary_point_count=int(row["boundary_point_count"]),
            interior_point_count=int(row["interior_point_count"]),
            reference_table_sha256=str(row["reference_table_sha256"]),
            max_reference_residual=float(row["max_reference_residual"]),
            max_word_stability=float(row["max_word_stability"]),
            max_symmetry_error=float(row["max_symmetry_error"]),
            min_geometry_margin=float(row["min_geometry_margin"]),
            safety_factor=float(row.get("safety_factor", 2.0)),
            certified=str(row["certified"]).strip().lower() in {"1", "true", "yes"},
        )


@dataclass(frozen=True)
class SchottkyValidityCertificate:
    cell_id: str
    topology: Topology
    word_length: int
    error_bound: float
    validation_point_count: int
    boundary_point_count: int
    interior_point_count: int
    min_geometry_margin: float
    reference_table_sha256: str


class SchottkyValidityEnvelope:
    """Collection of cells in which Schottky has a reference error bound."""

    def __init__(
        self,
        cells: Iterable[SchottkyValidityCell],
        *,
        minimum_validation_points: int = 64,
        minimum_interior_points: int = 8,
    ):
        if minimum_validation_points < 1:
            raise ValueError("minimum_validation_points must be positive")
        if minimum_interior_points < 1:
            raise ValueError("minimum_interior_points must be positive")
        self.cells = tuple(cells)
        self.minimum_validation_points = int(minimum_validation_points)
        self.minimum_interior_points = int(minimum_interior_points)

    @classmethod
    def from_csv(
        cls,
        path: Path | str,
        *,
        minimum_validation_points: int = 64,
        minimum_interior_points: int = 8,
    ) -> "SchottkyValidityEnvelope":
        with Path(path).open(newline="") as handle:
            return cls(
                (SchottkyValidityCell.from_csv_row(row) for row in csv.DictReader(handle)),
                minimum_validation_points=minimum_validation_points,
                minimum_interior_points=minimum_interior_points,
            )

    def certificate(
        self,
        topology: Topology,
        q_values: Sequence[complex],
        *,
        tolerance: float,
    ) -> SchottkyValidityCertificate | None:
        """Return the strongest admissible cell, or ``None`` outside the envelope."""

        admissible = [
            cell
            for cell in self.cells
            if cell.certified
            and cell.topology == topology
            and cell.validation_point_count >= self.minimum_validation_points
            and cell.interior_point_count >= self.minimum_interior_points
            and cell.error_bound <= float(tolerance)
            and cell.contains(q_values)
        ]
        if not admissible:
            return None
        cell = min(
            admissible,
            key=lambda value: (value.error_bound, -value.validation_point_count, value.cell_id),
        )
        return SchottkyValidityCertificate(
            cell_id=cell.cell_id,
            topology=cell.topology,
            word_length=cell.word_length,
            error_bound=cell.error_bound,
            validation_point_count=cell.validation_point_count,
            boundary_point_count=cell.boundary_point_count,
            interior_point_count=cell.interior_point_count,
            min_geometry_margin=cell.min_geometry_margin,
            reference_table_sha256=cell.reference_table_sha256,
        )
