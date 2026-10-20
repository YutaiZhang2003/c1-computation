#!/usr/bin/env python3
"""Indexed mixed-backend genus-two period table.

The table is indexed in coordinates adapted to plumbing rather than in the
raw real and imaginary entries of ``Omega``.  Forward queries use

``(log|q_e|, cos(arg q_e), sin(arg q_e))``

and inverse queries first apply the topology-specific leading ``Omega -> q``
map and use the same feature construction.  A local affine forward fit is
performed on ``Omega - Omega_leading``; the logarithmic cusp singularity is
therefore removed before interpolation.
"""

from __future__ import annotations

from c1_paths import PACKAGE_ROOT, data_root, package_path, module_path

import cmath
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Sequence

import numpy as np
from scipy.spatial import cKDTree

try:
    from bolza_torus_plumbing_reach import transform_omega
    from genus2_holomorphic_period_table import PeriodMapSeed
    from genus2_hybrid_period_map import period_difference_mod_integer
    from genus2_period_table_grid import DEFAULT_CONFIG, load_config
    from plumbing_algorithms import theta_leading_period_matrix
except ImportError:  # pragma: no cover - package-style execution
    from bolza_torus_plumbing_reach import transform_omega
    from genus2_holomorphic_period_table import PeriodMapSeed
    from genus2_hybrid_period_map import period_difference_mod_integer
    from genus2_period_table_grid import DEFAULT_CONFIG, load_config
    from plumbing_algorithms import theta_leading_period_matrix


Topology = Literal["theta", "glasses"]
TWO_PI_I = 2.0j * math.pi
J4 = np.block(
    [
        [np.zeros((2, 2), dtype=np.int64), np.eye(2, dtype=np.int64)],
        [-np.eye(2, dtype=np.int64), np.zeros((2, 2), dtype=np.int64)],
    ]
)


def parse_complex(value: object) -> complex:
    return complex(str(value).strip().replace("i", "j"))


def format_complex(value: complex) -> str:
    number = complex(value)
    return f"{number.real:+.17e}{number.imag:+.17e}j"


def symmetric_omega(omega11: complex, omega12: complex, omega22: complex) -> np.ndarray:
    return np.asarray(
        [[complex(omega11), complex(omega12)], [complex(omega12), complex(omega22)]],
        dtype=np.complex128,
    )


def leading_omega(topology: Topology, q_values: Sequence[complex]) -> np.ndarray:
    q = tuple(complex(value) for value in q_values)
    if len(q) != 3 or any(value == 0.0j for value in q):
        raise ValueError("leading period map needs three nonzero q values")
    if topology == "theta":
        return theta_leading_period_matrix(*q)
    if topology == "glasses":
        return symmetric_omega(
            cmath.log(q[0]) / TWO_PI_I,
            q[2] / (-TWO_PI_I),
            cmath.log(q[1]) / TWO_PI_I,
        )
    raise ValueError(f"unknown topology {topology!r}")


def leading_log_q_from_omega(
    topology: Topology,
    omega_values: Sequence[Sequence[complex]],
) -> tuple[complex, complex, complex]:
    omega = np.asarray(omega_values, dtype=np.complex128)
    if omega.shape != (2, 2) or not np.all(np.isfinite(omega)):
        raise ValueError("period matrix must be a finite 2x2 matrix")
    if topology == "theta":
        return (
            complex(TWO_PI_I * (omega[0, 0] - omega[0, 1])),
            complex(TWO_PI_I * (omega[1, 1] - omega[0, 1])),
            complex(TWO_PI_I * omega[0, 1]),
        )
    if topology == "glasses":
        bridge = complex(-TWO_PI_I * omega[0, 1])
        if bridge == 0.0j:
            raise ValueError("leading glasses bridge is zero")
        return (
            complex(TWO_PI_I * omega[0, 0]),
            complex(TWO_PI_I * omega[1, 1]),
            cmath.log(bridge),
        )
    raise ValueError(f"unknown topology {topology!r}")


def q_feature_from_logs(
    log_q: Sequence[complex],
    *,
    log_abs_min: float,
    log_abs_max: float,
) -> np.ndarray:
    """Return a dimensionless 9-vector with circular phase coordinates."""

    logs = tuple(complex(value) for value in log_q)
    if len(logs) != 3 or not log_abs_min < log_abs_max:
        raise ValueError("invalid log-q feature arguments")
    radial_scale = log_abs_max - log_abs_min
    feature: list[float] = []
    for value in logs:
        if not math.isfinite(value.real) or not math.isfinite(value.imag):
            raise ValueError("log-q feature values must be finite")
        phase = math.remainder(value.imag, 2.0 * math.pi)
        feature.extend(
            (
                (value.real - log_abs_min) / radial_scale,
                0.5 * math.cos(phase),
                0.5 * math.sin(phase),
            )
        )
    return np.asarray(feature, dtype=np.float64)


def q_feature(
    q_values: Sequence[complex],
    *,
    log_abs_min: float,
    log_abs_max: float,
) -> np.ndarray:
    q = tuple(complex(value) for value in q_values)
    if len(q) != 3 or any(value == 0.0j for value in q):
        raise ValueError("q feature needs three nonzero values")
    return q_feature_from_logs(
        tuple(cmath.log(value) for value in q),
        log_abs_min=log_abs_min,
        log_abs_max=log_abs_max,
    )


def omega_feature(
    topology: Topology,
    omega: Sequence[Sequence[complex]],
    *,
    log_abs_min: float,
    log_abs_max: float,
) -> np.ndarray:
    return q_feature_from_logs(
        leading_log_q_from_omega(topology, omega),
        log_abs_min=log_abs_min,
        log_abs_max=log_abs_max,
    )


def fundamental_omega_coordinates(
    omega_values: Sequence[Sequence[complex]],
) -> np.ndarray:
    """Return six smooth coordinates for a Gottschling-domain KD tree.

    The real entries are already bounded in a fundamental representative.
    The imaginary entries are expressed in the same common-scale, anisotropy,
    and mixing coordinates used by the global moduli sampler.  Standardizing
    these six coordinates with table-wide moments prevents a large cusp scale
    from overwhelming the compact directions in nearest-neighbour searches.
    """

    omega = np.asarray(omega_values, dtype=np.complex128)
    if omega.shape != (2, 2) or not np.all(np.isfinite(omega)):
        raise ValueError("fundamental period matrix must be a finite 2x2 matrix")
    omega = 0.5 * (omega + omega.T)
    y = np.asarray(omega.imag, dtype=np.float64)
    if float(np.min(np.linalg.eigvalsh(y))) <= 0.0 or y[0, 0] <= 0.0:
        raise ValueError("fundamental period matrix must lie in the Siegel upper half-space")
    return np.asarray(
        [
            2.0 * omega[0, 0].real,
            2.0 * omega[0, 1].real,
            2.0 * omega[1, 1].real,
            math.log(float(y[0, 0])),
            math.log(float(y[1, 1] / y[0, 0])),
            float(2.0 * y[0, 1] / y[0, 0]),
        ],
        dtype=np.float64,
    )


def symplectic_inverse(matrix: Sequence[Sequence[int]]) -> np.ndarray:
    """Return the exact integer inverse of one ``Sp(4,Z)`` matrix."""

    value = np.asarray(matrix, dtype=np.int64)
    if value.shape != (4, 4) or not np.array_equal(value.T @ J4 @ value, J4):
        raise ValueError("fundamental period-table marking is not symplectic")
    inverse = -J4 @ value.T @ J4
    if not np.array_equal(inverse @ value, np.eye(4, dtype=np.int64)):
        raise ValueError("exact symplectic inverse failed")
    return inverse


@dataclass(frozen=True)
class PeriodTableEntry:
    row_id: str
    topology: Topology
    q: tuple[complex, complex, complex]
    omega: np.ndarray
    actual_backend: str
    precision_tier: str
    error_estimate: float
    geometry_margin: float
    certified: bool

    @classmethod
    def from_csv_row(cls, row: dict[str, str]) -> "PeriodTableEntry":
        return cls(
            row_id=str(row["row_id"]),
            topology=str(row["topology"]),  # type: ignore[arg-type]
            q=tuple(parse_complex(row[f"q{edge}"]) for edge in (1, 2, 3)),  # type: ignore[arg-type]
            omega=symmetric_omega(
                parse_complex(row["omega11"]),
                parse_complex(row["omega12"]),
                parse_complex(row["omega22"]),
            ),
            actual_backend=str(row["actual_backend"]),
            precision_tier=str(row.get("actual_precision_tier") or row["precision_tier"]),
            error_estimate=float(row["error_estimate"]),
            geometry_margin=float(row["geometry_margin"]),
            certified=str(row["certified"]).strip().lower() in {"1", "true", "yes"},
        )

    def validate(self) -> None:
        if self.topology not in {"theta", "glasses"}:
            raise ValueError(f"unknown topology {self.topology!r}")
        if len(self.q) != 3 or any(
            not (math.isfinite(value.real) and math.isfinite(value.imag) and 0.0 < abs(value) < 1.0)
            for value in self.q
        ):
            raise ValueError(f"invalid q values in table row {self.row_id}")
        omega = np.asarray(self.omega, dtype=np.complex128)
        if omega.shape != (2, 2) or not np.all(np.isfinite(omega)):
            raise ValueError(f"invalid Omega in table row {self.row_id}")
        if float(np.max(np.abs(omega - omega.T))) > 1.0e-10:
            raise ValueError(f"nonsymmetric Omega in table row {self.row_id}")
        if not math.isfinite(self.error_estimate) or self.error_estimate < 0.0:
            raise ValueError(f"invalid error estimate in table row {self.row_id}")
        if not math.isfinite(self.geometry_margin) or self.geometry_margin <= 0.0:
            raise ValueError(f"invalid geometry margin in table row {self.row_id}")


@dataclass(frozen=True)
class LocalInterpolationResult:
    omega: np.ndarray
    neighbor_row_ids: tuple[str, ...]
    maximum_feature_distance: float
    local_fit_residual: float


@dataclass(frozen=True)
class FundamentalPeriodSeed:
    """A table seed together with its exact fundamental-to-chart marking."""

    seed: PeriodMapSeed
    topology: Topology
    row_id: str
    table_distance: float
    matrix_fund_to_raw: tuple[tuple[int, ...], ...]
    omega_marked: np.ndarray
    omega_fund: np.ndarray
    raw_to_fund_residual: float
    domain_margin: float


class Genus2PeriodMapTable:
    """KD-tree indices for forward interpolation and inverse-solver seeds."""

    def __init__(
        self,
        entries: Iterable[PeriodTableEntry],
        *,
        q_abs_min: float,
        q_abs_max: float,
        _portable_features: dict[Topology, tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None,
        _fundamental_data: dict[str, np.ndarray] | None = None,
    ) -> None:
        if not 0.0 < q_abs_min < q_abs_max < 1.0:
            raise ValueError("invalid period-table q bounds")
        accepted = tuple(entry for entry in entries if entry.certified)
        if not accepted:
            raise ValueError("period table contains no certified rows")
        for entry in accepted:
            entry.validate()
        self.entries = accepted
        self.log_abs_min = math.log(float(q_abs_min))
        self.log_abs_max = math.log(float(q_abs_max))
        self._indices: dict[Topology, np.ndarray] = {}
        self._q_features: dict[Topology, np.ndarray] = {}
        self._omega_features: dict[Topology, np.ndarray] = {}
        self._q_trees: dict[Topology, cKDTree] = {}
        self._omega_trees: dict[Topology, cKDTree] = {}
        for topology in ("theta", "glasses"):
            indices = np.asarray(
                [index for index, entry in enumerate(accepted) if entry.topology == topology],
                dtype=np.int64,
            )
            if indices.size == 0:
                continue
            portable = None if _portable_features is None else _portable_features.get(topology)
            if portable is None:
                q_features = np.vstack(
                    [
                        q_feature(
                            accepted[index].q,
                            log_abs_min=self.log_abs_min,
                            log_abs_max=self.log_abs_max,
                        )
                        for index in indices
                    ]
                )
                inverse_features = np.vstack(
                    [
                        omega_feature(
                            topology,
                            accepted[index].omega,
                            log_abs_min=self.log_abs_min,
                            log_abs_max=self.log_abs_max,
                        )
                        for index in indices
                    ]
                )
            else:
                stored_indices, q_features, inverse_features = portable
                stored_indices = np.asarray(stored_indices, dtype=np.int64)
                q_features = np.asarray(q_features, dtype=np.float64)
                inverse_features = np.asarray(inverse_features, dtype=np.float64)
                if not np.array_equal(stored_indices, indices):
                    raise ValueError(f"portable {topology} entry indices do not match table rows")
                expected_shape = (indices.size, 9)
                if q_features.shape != expected_shape or inverse_features.shape != expected_shape:
                    raise ValueError(f"portable {topology} features have the wrong shape")
            self._indices[topology] = indices
            self._q_features[topology] = q_features
            self._omega_features[topology] = inverse_features
            self._q_trees[topology] = cKDTree(q_features)
            self._omega_trees[topology] = cKDTree(inverse_features)
        self._fundamental_omega: np.ndarray | None = None
        self._sp4_raw_to_fund: np.ndarray | None = None
        self._fundamental_domain_margins: np.ndarray | None = None
        self._fundamental_transform_residuals: np.ndarray | None = None
        self._fundamental_feature_center: np.ndarray | None = None
        self._fundamental_feature_scale: np.ndarray | None = None
        self._fundamental_features: np.ndarray | None = None
        self._fundamental_trees: dict[Topology, cKDTree] = {}
        if _fundamental_data is not None:
            omega_fund = np.asarray(_fundamental_data["omega_fund"], dtype=np.complex128)
            sp4 = np.asarray(_fundamental_data["sp4_raw_to_fund"], dtype=np.int64)
            margins = np.asarray(_fundamental_data["domain_margins"], dtype=np.float64)
            residuals = np.asarray(
                _fundamental_data["transform_residuals"], dtype=np.float64
            )
            center = np.asarray(
                _fundamental_data["fundamental_feature_center"], dtype=np.float64
            )
            scale = np.asarray(
                _fundamental_data["fundamental_feature_scale"], dtype=np.float64
            )
            features = np.asarray(
                _fundamental_data["fundamental_features"], dtype=np.float64
            )
            size = len(accepted)
            if (
                omega_fund.shape != (size, 3)
                or sp4.shape != (size, 4, 4)
                or margins.shape != (size,)
                or residuals.shape != (size,)
                or center.shape != (6,)
                or scale.shape != (6,)
                or features.shape != (size, 6)
                or np.any(~np.isfinite(features))
                or np.any(~np.isfinite(center))
                or np.any(~np.isfinite(scale))
                or np.any(scale <= 0.0)
            ):
                raise ValueError("portable fundamental period-table arrays are inconsistent")
            self._fundamental_omega = omega_fund
            self._sp4_raw_to_fund = sp4
            self._fundamental_domain_margins = margins
            self._fundamental_transform_residuals = residuals
            self._fundamental_feature_center = center
            self._fundamental_feature_scale = scale
            self._fundamental_features = features
            for topology, indices in self._indices.items():
                self._fundamental_trees[topology] = cKDTree(features[indices])

    @classmethod
    def from_csv(
        cls,
        table_path: Path | str,
        *,
        config_path: Path | str = DEFAULT_CONFIG,
    ) -> "Genus2PeriodMapTable":
        config = load_config(config_path)
        domain = config["q_domain"]
        with Path(table_path).open(newline="") as handle:
            entries = tuple(PeriodTableEntry.from_csv_row(row) for row in csv.DictReader(handle))
        return cls(
            entries,
            q_abs_min=float(domain["q_abs_tail_min"]),
            q_abs_max=float(domain["q_abs_max"]),
        )

    @classmethod
    def from_portable_index(
        cls,
        index_path: Path | str,
        *,
        verify_table_path: Path | str | None = None,
    ) -> "Genus2PeriodMapTable":
        """Load all query-critical rows and features without parsing the CSV."""

        path = Path(index_path)
        with np.load(path, allow_pickle=False) as archive:
            version = int(np.asarray(archive["schema_version"]).reshape(-1)[0])
            if version not in {2, 3}:
                raise ValueError(f"unsupported portable period-table schema {version}")
            expected_table_sha256 = str(np.asarray(archive["table_sha256"]).reshape(-1)[0])
            if verify_table_path is not None:
                digest = hashlib.sha256()
                with Path(verify_table_path).open("rb") as handle:
                    for block in iter(lambda: handle.read(1 << 20), b""):
                        digest.update(block)
                verification_digest = (
                    str(np.asarray(archive["canonical_table_sha256"]).reshape(-1)[0])
                    if version >= 3 and "canonical_table_sha256" in archive
                    else expected_table_sha256
                )
                if digest.hexdigest() != verification_digest:
                    raise ValueError("portable index does not match its canonical CSV table")
            row_ids = np.asarray(archive["row_ids"]).astype(str)
            topology_codes = np.asarray(archive["topology_codes"], dtype=np.int8)
            q_values = np.asarray(archive["q_values"], dtype=np.complex128)
            omega_values = np.asarray(archive["omega_values"], dtype=np.complex128)
            actual_backends = np.asarray(archive["actual_backends"]).astype(str)
            precision_tiers = np.asarray(archive["precision_tiers"]).astype(str)
            error_estimates = np.asarray(archive["error_estimates"], dtype=np.float64)
            geometry_margins = np.asarray(archive["geometry_margins"], dtype=np.float64)
            log_abs_bounds = np.asarray(archive["log_abs_bounds"], dtype=np.float64)
            size = row_ids.size
            if (
                topology_codes.shape != (size,)
                or q_values.shape != (size, 3)
                or omega_values.shape != (size, 3)
                or actual_backends.shape != (size,)
                or precision_tiers.shape != (size,)
                or error_estimates.shape != (size,)
                or geometry_margins.shape != (size,)
                or log_abs_bounds.shape != (2,)
            ):
                raise ValueError("portable period-table arrays have inconsistent shapes")
            if np.any((topology_codes < 0) | (topology_codes > 1)):
                raise ValueError("portable period table has an unknown topology code")
            entries = tuple(
                PeriodTableEntry(
                    row_id=str(row_ids[index]),
                    topology=("theta" if topology_codes[index] == 0 else "glasses"),
                    q=tuple(complex(value) for value in q_values[index]),  # type: ignore[arg-type]
                    omega=symmetric_omega(*omega_values[index]),
                    actual_backend=str(actual_backends[index]),
                    precision_tier=str(precision_tiers[index]),
                    error_estimate=float(error_estimates[index]),
                    geometry_margin=float(geometry_margins[index]),
                    certified=True,
                )
                for index in range(size)
            )
            portable_features: dict[
                Topology, tuple[np.ndarray, np.ndarray, np.ndarray]
            ] = {}
            for topology in ("theta", "glasses"):
                prefix = topology
                indices_key = f"{prefix}_entry_indices"
                if indices_key not in archive:
                    continue
                portable_features[topology] = (
                    np.asarray(archive[indices_key], dtype=np.int64),
                    np.asarray(archive[f"{prefix}_q_features"], dtype=np.float64),
                    np.asarray(archive[f"{prefix}_omega_features"], dtype=np.float64),
                )
            fundamental_data = None
            if version >= 3:
                required = (
                    "omega_fund",
                    "sp4_raw_to_fund",
                    "domain_margins",
                    "transform_residuals",
                    "fundamental_feature_center",
                    "fundamental_feature_scale",
                    "fundamental_features",
                )
                missing = [key for key in required if key not in archive]
                if missing:
                    raise ValueError(
                        f"portable fundamental period table is missing arrays {missing}"
                    )
                fundamental_data = {
                    key: np.asarray(archive[key]).copy() for key in required
                }
        return cls(
            entries,
            q_abs_min=math.exp(float(log_abs_bounds[0])),
            q_abs_max=math.exp(float(log_abs_bounds[1])),
            _portable_features=portable_features,
            _fundamental_data=fundamental_data,
        )

    @property
    def has_fundamental_index(self) -> bool:
        return bool(self._fundamental_omega is not None)

    def _query(
        self,
        topology: Topology,
        feature: np.ndarray,
        *,
        count: int,
        inverse: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        trees = self._omega_trees if inverse else self._q_trees
        if topology not in trees:
            raise ValueError(f"table has no certified {topology} rows")
        size = len(self._indices[topology])
        k = min(max(int(count), 1), size)
        distances, local_indices = trees[topology].query(feature, k=k)
        distances = np.atleast_1d(np.asarray(distances, dtype=np.float64))
        local_indices = np.atleast_1d(np.asarray(local_indices, dtype=np.int64))
        return distances, self._indices[topology][local_indices]

    def nearest_q_entries(
        self,
        topology: Topology,
        q_values: Sequence[complex],
        *,
        count: int = 8,
    ) -> tuple[tuple[float, PeriodTableEntry], ...]:
        feature = q_feature(
            q_values,
            log_abs_min=self.log_abs_min,
            log_abs_max=self.log_abs_max,
        )
        distances, indices = self._query(topology, feature, count=count, inverse=False)
        return tuple((float(distance), self.entries[int(index)]) for distance, index in zip(distances, indices))

    def nearest_seeds(
        self,
        topology: Topology,
        target_omega: Sequence[Sequence[complex]],
        *,
        count: int = 8,
    ) -> tuple[PeriodMapSeed, ...]:
        feature = omega_feature(
            topology,
            target_omega,
            log_abs_min=self.log_abs_min,
            log_abs_max=self.log_abs_max,
        )
        distances, indices = self._query(topology, feature, count=count, inverse=True)
        return tuple(
            PeriodMapSeed(
                q=entry.q,
                log_q=tuple(cmath.log(value) for value in entry.q),  # type: ignore[arg-type]
                source=f"genus2-period-table-v2:{entry.row_id}",
                table_distance=float(distance),
            )
            for distance, index in zip(distances, indices)
            for entry in (self.entries[int(index)],)
        )

    def nearest_omega_entries(
        self,
        topology: Topology,
        target_omega: Sequence[Sequence[complex]],
        *,
        count: int = 8,
    ) -> tuple[tuple[float, PeriodTableEntry], ...]:
        """Return inverse-index neighbours in topology-adapted coordinates."""

        feature = omega_feature(
            topology,
            target_omega,
            log_abs_min=self.log_abs_min,
            log_abs_max=self.log_abs_max,
        )
        distances, indices = self._query(topology, feature, count=count, inverse=True)
        return tuple(
            (float(distance), self.entries[int(index)])
            for distance, index in zip(distances, indices)
        )

    def nearest_fundamental_seeds(
        self,
        topology: Topology,
        target_omega: Sequence[Sequence[complex]],
        *,
        count: int = 4,
    ) -> tuple[FundamentalPeriodSeed, ...]:
        """Return nearby table charts with exact fundamental-to-raw markings.

        The stored finite-q correction is transported to the target by adding
        the change predicted by the topology-specific leading period map.  It
        remains only an inverse seed: the live solver recomputes and certifies
        every returned plumbing coordinate.
        """

        if (
            self._fundamental_omega is None
            or self._sp4_raw_to_fund is None
            or self._fundamental_domain_margins is None
            or self._fundamental_transform_residuals is None
            or self._fundamental_feature_center is None
            or self._fundamental_feature_scale is None
            or topology not in self._fundamental_trees
        ):
            return ()
        target = np.asarray(target_omega, dtype=np.complex128)
        if target.shape != (2, 2) or not np.all(np.isfinite(target)):
            raise ValueError("fundamental inverse target must be a finite 2x2 matrix")
        target = 0.5 * (target + target.T)
        feature = (
            fundamental_omega_coordinates(target) - self._fundamental_feature_center
        ) / self._fundamental_feature_scale
        indices = self._indices[topology]
        k = min(max(int(count), 1), len(indices))
        distances, local_indices = self._fundamental_trees[topology].query(feature, k=k)
        distances = np.atleast_1d(np.asarray(distances, dtype=np.float64))
        local_indices = np.atleast_1d(np.asarray(local_indices, dtype=np.int64))
        out: list[FundamentalPeriodSeed] = []
        for distance, local_index in zip(distances, local_indices):
            index = int(indices[int(local_index)])
            entry = self.entries[index]
            fund_to_raw = symplectic_inverse(self._sp4_raw_to_fund[index])
            marked = transform_omega(fund_to_raw, target)
            marked = 0.5 * (marked + marked.T)
            row_logs = tuple(cmath.log(value) for value in entry.q)
            try:
                target_leading = leading_log_q_from_omega(topology, marked)
                row_leading = leading_log_q_from_omega(topology, entry.omega)
                transported_logs = tuple(
                    row_log + target_log - row_log_leading
                    for row_log, target_log, row_log_leading in zip(
                        row_logs, target_leading, row_leading
                    )
                )
                if any(
                    not math.isfinite(value.real)
                    or not math.isfinite(value.imag)
                    or value.real >= 0.0
                    for value in transported_logs
                ):
                    transported_logs = row_logs
            except ValueError:
                transported_logs = row_logs
            transported_q = tuple(
                cmath.exp(complex(max(value.real, -690.0), value.imag))
                for value in transported_logs
            )
            seed = PeriodMapSeed(
                q=transported_q,  # type: ignore[arg-type]
                log_q=transported_logs,  # type: ignore[arg-type]
                source=f"fundamental-period-table-v3:{entry.row_id}",
                table_distance=float(distance),
            )
            out.append(
                FundamentalPeriodSeed(
                    seed=seed,
                    topology=topology,
                    row_id=entry.row_id,
                    table_distance=float(distance),
                    matrix_fund_to_raw=tuple(
                        tuple(int(value) for value in row) for row in fund_to_raw
                    ),
                    omega_marked=np.asarray(marked, dtype=np.complex128),
                    omega_fund=symmetric_omega(*self._fundamental_omega[index]),
                    raw_to_fund_residual=float(
                        self._fundamental_transform_residuals[index]
                    ),
                    domain_margin=float(self._fundamental_domain_margins[index]),
                )
            )
        return tuple(out)

    def inverse_index_spacing(self, topology: Topology, *, quantile: float = 0.995) -> float:
        """Return a leave-one-out inverse-feature spacing diagnostic."""

        if topology not in self._omega_trees:
            raise ValueError(f"table has no certified {topology} rows")
        if not 0.0 < float(quantile) <= 1.0:
            raise ValueError("spacing quantile must lie in (0,1]")
        features = self._omega_features[topology]
        if len(features) < 2:
            raise ValueError("at least two rows are needed for a spacing diagnostic")
        distances, _ = self._omega_trees[topology].query(features, k=2)
        return float(np.quantile(np.asarray(distances)[:, 1], float(quantile)))

    def interpolate_omega(
        self,
        topology: Topology,
        q_values: Sequence[complex],
        *,
        count: int = 24,
        ridge: float = 1.0e-10,
    ) -> LocalInterpolationResult:
        """Fit the regular correction to the leading plumbing period map."""

        q = tuple(complex(value) for value in q_values)
        neighbours = self.nearest_q_entries(topology, q, count=max(count, 8))
        if neighbours[0][0] < 1.0e-14:
            entry = neighbours[0][1]
            return LocalInterpolationResult(
                omega=np.asarray(entry.omega, dtype=np.complex128).copy(),
                neighbor_row_ids=(entry.row_id,),
                maximum_feature_distance=float(neighbours[0][0]),
                local_fit_residual=0.0,
            )
        query_logs = tuple(cmath.log(value) for value in q)
        rows: list[list[float]] = []
        targets: list[list[float]] = []
        weights: list[float] = []
        row_ids: list[str] = []
        for distance, entry in neighbours:
            logs = tuple(cmath.log(value) for value in entry.q)
            delta = [value.real - center.real for value, center in zip(logs, query_logs)]
            delta.extend(
                math.remainder(value.imag - center.imag, 2.0 * math.pi)
                for value, center in zip(logs, query_logs)
            )
            rows.append([1.0, *delta])
            correction = period_difference_mod_integer(entry.omega, leading_omega(topology, entry.q))
            vector = (correction[0, 0], correction[0, 1], correction[1, 1])
            targets.append([component for value in vector for component in (value.real, value.imag)])
            weights.append(1.0 / max(float(distance), 1.0e-12))
            row_ids.append(entry.row_id)
        design = np.asarray(rows, dtype=np.float64)
        target = np.asarray(targets, dtype=np.float64)
        sqrt_weight = np.sqrt(np.asarray(weights, dtype=np.float64))[:, None]
        weighted_design = sqrt_weight * design
        weighted_target = sqrt_weight * target
        # Regularize slopes only.  Penalizing the intercept biases even a
        # perfectly constant correction when the neighbour cloud is locally
        # rank deficient.
        penalty = math.sqrt(float(ridge)) * np.diag(
            np.asarray([0.0] + [1.0] * (design.shape[1] - 1), dtype=np.float64)
        )
        augmented_design = np.vstack((weighted_design, penalty))
        augmented_target = np.vstack(
            (weighted_target, np.zeros((penalty.shape[0], target.shape[1]), dtype=np.float64))
        )
        coefficients, _, _, _ = np.linalg.lstsq(
            augmented_design, augmented_target, rcond=None
        )
        predicted = coefficients[0]
        correction_vector = tuple(
            complex(predicted[2 * index], predicted[2 * index + 1]) for index in range(3)
        )
        correction = symmetric_omega(*correction_vector)
        fitted = design @ coefficients
        local_residual = float(np.max(np.abs(fitted - target)))
        return LocalInterpolationResult(
            omega=leading_omega(topology, q) + correction,
            neighbor_row_ids=tuple(row_ids),
            maximum_feature_distance=max(float(item[0]) for item in neighbours),
            local_fit_residual=local_residual,
        )

    def write_portable_index(self, path: Path | str, *, table_sha256: str) -> None:
        """Save certified rows and features; rebuild KD trees portably at load time."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {
            "schema_version": np.asarray([2], dtype=np.int64),
            "table_sha256": np.asarray([str(table_sha256)]),
            "log_abs_bounds": np.asarray([self.log_abs_min, self.log_abs_max]),
            "row_ids": np.asarray([entry.row_id for entry in self.entries]),
            "topology_codes": np.asarray(
                [0 if entry.topology == "theta" else 1 for entry in self.entries],
                dtype=np.int8,
            ),
            "q_values": np.asarray([entry.q for entry in self.entries], dtype=np.complex128),
            "omega_values": np.asarray(
                [
                    (entry.omega[0, 0], entry.omega[0, 1], entry.omega[1, 1])
                    for entry in self.entries
                ],
                dtype=np.complex128,
            ),
            "actual_backends": np.asarray([entry.actual_backend for entry in self.entries]),
            "precision_tiers": np.asarray([entry.precision_tier for entry in self.entries]),
            "error_estimates": np.asarray(
                [entry.error_estimate for entry in self.entries], dtype=np.float64
            ),
            "geometry_margins": np.asarray(
                [entry.geometry_margin for entry in self.entries], dtype=np.float64
            ),
        }
        for topology in ("theta", "glasses"):
            if topology not in self._indices:
                continue
            arrays[f"{topology}_entry_indices"] = self._indices[topology]
            arrays[f"{topology}_q_features"] = self._q_features[topology]
            arrays[f"{topology}_omega_features"] = self._omega_features[topology]
        np.savez_compressed(destination, **arrays)


def table_metadata(table_path: Path | str) -> dict[str, object]:
    path = Path(table_path)
    return {
        "table": str(path),
        "index_coordinates": {
            "forward": "normalized log|q| and circular q phases",
            "inverse": "leading topology-specific Omega->q followed by the same q features",
            "forward_interpolation": "local affine fit to Omega-Omega_leading",
        },
        "note": "The table interpolant supplies a fast value or inverse seed; certified calculations still run the direct solver.",
    }
