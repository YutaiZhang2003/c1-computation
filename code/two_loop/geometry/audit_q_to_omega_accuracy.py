#!/usr/bin/env python3
"""Audit plumbing coordinates against their target period matrices.

The shared hybrid policy uses normalized holomorphic one-forms in the bulk,
adaptive Schottky words in long cusps, and both maps in their overlap.  A saved
plumbing coordinate is tested in three distinct ways:

1. forward ``q -> Omega`` residual against the marked target;
2. basis/seam convergence of the normalized holomorphic one-forms;
3. movement of ``q`` after re-solving the direct inverse problem when needed.

An optional six-real-dimensional calibration envelope can strengthen a
Schottky certificate, but is not required in the cusp where collocation is
ill-conditioned.  No scalar-q interval is deliberately left unsupported.
"""

from __future__ import annotations

from c1_paths import PACKAGE_ROOT, data_root, package_path, module_path

import argparse
import csv
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import least_squares

THIS_FILE = Path(__file__).resolve()
sys.path.insert(0, str(THIS_FILE.parent))

from bolza_torus_plumbing_reach import (  # noqa: E402
    enumerate_symplectic_words,
    transform_omega,
)
from genus2_plumbing_atlas import (  # noqa: E402
    HOLOMORPHIC_COLLOCATION_MAX_TAU_IMAG,
    HOLOMORPHIC_COLLOCATION_MIN_Q,
    symplectic_matrix_csv_fields,
    symplectic_matrix_from_csv_row,
)
from genus2_holomorphic_period_table import (  # noqa: E402
    SchottkyValidityEnvelope,
)
from genus2_hybrid_period_map import (  # noqa: E402
    HOLOMORPHIC_ALGORITHM,
    MULTIPRECISION_HOLOMORPHIC_ALGORITHM,
    HybridPeriodMapConfig,
    hybrid_period_matrix,
    is_schottky_algorithm,
    refine_multiprecision_holomorphic_inverse,
    refine_schottky_inverse,
)
from plumbing_algorithms import (  # noqa: E402
    glasses_collocation_period_matrix,
    genus2_symmetric_period_vector,
    q_from_tau,
    solve_theta_collocation,
    symmetrized_period_matrix,
    tau_from_q,
)


DEFAULT_INPUT = Path(
    str(package_path('plumbing/results/genus2_c1_moduli_mc/pilot_R1_N64_refined/refined_samples.csv'))
)
DEFAULT_OUTPUT = Path(
    str(package_path('plumbing/results/genus2_c1_moduli_mc/pilot_R1_N64_q_to_omega_audit'))
)


@dataclass(frozen=True)
class CollocationRefinedInverse:
    success: bool
    message: str
    nfev: int
    q: tuple[complex, complex, complex]
    fit_residual: float
    branch: np.ndarray
    seam_residual: float
    symmetry_error: float


@dataclass(frozen=True)
class PeriodMapValidation:
    q: tuple[complex, complex, complex]
    period_algorithm: str
    refined: bool
    fixed_q_residual: float
    fixed_q_stability: float
    final_residual: float
    final_stability: float
    seam_residual: float
    symmetry_error: float
    low_order: int
    high_order: int
    validation_order: int
    reinverse_success: bool
    reinverse_message: str
    reinverse_nfev: int
    max_tau_shift: float
    log_q: tuple[complex, complex, complex] | None = None
    validity_cell_id: str | None = None
    certified_error_bound: float | None = None
    validity_reference_table_sha256: str | None = None
    period_map_region: str | None = None
    overlap_residual: float | None = None
    agreement_tolerance: float | None = None

    @property
    def fixed_q_word_step(self) -> float:
        """Compatibility alias for pre-collocation result readers."""

        return self.fixed_q_stability

    @property
    def final_word_step(self) -> float:
        """Compatibility alias for pre-collocation result readers."""

        return self.final_stability


DEEP_CUSP_THRESHOLD = HOLOMORPHIC_COLLOCATION_MIN_Q


def collocation_orders(
    topology: str,
    q_values: Sequence[complex],
) -> tuple[int, int, int, int, int, int]:
    """Return low/high/validation Laurent and seam-sampling orders."""

    q_max = max(abs(complex(value)) for value in q_values)
    if topology == "theta":
        if q_max <= 0.2:
            low, high = 20, 24
        elif q_max <= 0.3:
            low, high = 24, 32
        else:
            low, high = 32, 44
        validation = high + 4
        return low, 4 * low, high, 4 * high, validation, 4 * validation
    if topology == "glasses":
        if q_max <= 0.2:
            low, high = 20, 28
        elif q_max <= 0.3:
            low, high = 24, 32
        else:
            low, high = 32, 44
        validation = high + 4
        return low, 6 * low, high, 6 * high, validation, 6 * validation
    raise ValueError(f"unknown topology {topology!r}")


def collocation_period_from_q(
    topology: str,
    q_values: Sequence[complex],
    *,
    basis_order: int,
    samples_per_seam: int,
) -> tuple[np.ndarray, float, float]:
    """Compute ``Omega`` from normalized holomorphic one-forms."""

    q = tuple(complex(value) for value in q_values)
    if topology == "theta":
        result = solve_theta_collocation(
            *q,
            basis_order=int(basis_order),
            samples_per_seam=int(samples_per_seam),
        )
        return (
            symmetrized_period_matrix(result.omega),
            float(result.max_seam_residual),
            float(result.omega_symmetry_error),
        )
    if topology == "glasses":
        omega, seam_residual, symmetry_error = glasses_collocation_period_matrix(
            *q,
            basis_order=int(basis_order),
            samples_per_seam=int(samples_per_seam),
        )
        return (
            symmetrized_period_matrix(omega),
            float(seam_residual),
            float(symmetry_error),
        )
    raise ValueError(f"unknown topology {topology!r}")


def period_difference_mod_integer(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    difference = np.asarray(left - right, dtype=np.complex128)
    branch = np.rint(difference.real).astype(int)
    branch = np.rint(0.5 * (branch + branch.T)).astype(int)
    return difference - branch


def period_max_residual(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.max(np.abs(period_difference_mod_integer(left, right))))


def _pack_q(topology: str, q_values: Sequence[complex]) -> np.ndarray:
    if topology == "theta":
        taus = [tau_from_q(complex(value)) for value in q_values]
        return np.asarray(
            [component for tau in taus for component in (tau.real, tau.imag)],
            dtype=float,
        )
    tau1 = tau_from_q(complex(q_values[0]))
    tau2 = tau_from_q(complex(q_values[1]))
    q3 = complex(q_values[2])
    return np.asarray(
        [tau1.real, tau1.imag, tau2.real, tau2.imag, q3.real, q3.imag],
        dtype=float,
    )


def _unpack_q(topology: str, values: np.ndarray) -> tuple[complex, complex, complex]:
    if topology == "theta":
        return tuple(
            q_from_tau(complex(float(values[index]), float(values[index + 1])))
            for index in (0, 2, 4)
        )  # type: ignore[return-value]
    return (
        q_from_tau(complex(float(values[0]), float(values[1]))),
        q_from_tau(complex(float(values[2]), float(values[3]))),
        complex(float(values[4]), float(values[5])),
    )


def refine_inverse_collocation(
    topology: str,
    target: np.ndarray,
    initial_q: Sequence[complex],
    *,
    basis_order: int,
    samples_per_seam: int,
    max_nfev: int,
    q3_component_bound: float = 0.98,
) -> CollocationRefinedInverse:
    """Re-solve the inverse map using normalized holomorphic one-forms."""

    initial_tuple = tuple(complex(value) for value in initial_q)
    initial_forward, _, _ = collocation_period_from_q(
        topology,
        initial_tuple,
        basis_order=basis_order,
        samples_per_seam=samples_per_seam,
    )
    branch = np.rint((initial_forward - target).real).astype(int)
    branch = np.rint(0.5 * (branch + branch.T)).astype(int)
    target_on_branch = target + branch
    target_vector = genus2_symmetric_period_vector(target_on_branch)
    x0 = _pack_q(topology, initial_tuple)
    if topology == "theta":
        lower = np.asarray(
            [-np.inf, 1.0e-12, -np.inf, 1.0e-12, -np.inf, 1.0e-12]
        )
        upper = np.asarray(
            [
                np.inf,
                HOLOMORPHIC_COLLOCATION_MAX_TAU_IMAG,
                np.inf,
                HOLOMORPHIC_COLLOCATION_MAX_TAU_IMAG,
                np.inf,
                HOLOMORPHIC_COLLOCATION_MAX_TAU_IMAG,
            ]
        )
    else:
        bound = float(q3_component_bound)
        lower = np.asarray(
            [-np.inf, 1.0e-12, -np.inf, 1.0e-12, -bound, -bound]
        )
        upper = np.asarray(
            [
                np.inf,
                HOLOMORPHIC_COLLOCATION_MAX_TAU_IMAG,
                np.inf,
                HOLOMORPHIC_COLLOCATION_MAX_TAU_IMAG,
                bound,
                bound,
            ]
        )
        x0 = np.minimum(np.maximum(x0, lower), upper)

    def residual(values: np.ndarray) -> np.ndarray:
        q_values = _unpack_q(topology, values)
        if any(
            not (HOLOMORPHIC_COLLOCATION_MIN_Q <= abs(value) < 1.0)
            for value in q_values
        ):
            return 1.0e6 * np.ones(6)
        try:
            forward, seam_residual, symmetry_error = collocation_period_from_q(
                topology,
                q_values,
                basis_order=basis_order,
                samples_per_seam=samples_per_seam,
            )
        except Exception:
            return 1.0e6 * np.ones(6)
        if (
            not np.all(np.isfinite(forward))
            or not math.isfinite(seam_residual)
            or not math.isfinite(symmetry_error)
        ):
            return 1.0e6 * np.ones(6)
        return genus2_symmetric_period_vector(forward) - target_vector

    optimum = least_squares(
        residual,
        x0,
        bounds=(lower, upper),
        max_nfev=int(max_nfev),
        diff_step=1.0e-5,
        x_scale="jac",
        xtol=1.0e-11,
        ftol=1.0e-11,
        gtol=1.0e-11,
    )
    q_values = _unpack_q(topology, optimum.x)
    forward, seam_residual, symmetry_error = collocation_period_from_q(
        topology,
        q_values,
        basis_order=basis_order,
        samples_per_seam=samples_per_seam,
    )
    return CollocationRefinedInverse(
        success=bool(optimum.success),
        message=str(optimum.message),
        nfev=int(optimum.nfev),
        q=q_values,
        fit_residual=float(np.max(np.abs(forward - target_on_branch))),
        branch=branch,
        seam_residual=seam_residual,
        symmetry_error=symmetry_error,
    )


def tau_shift_mod_integer(
    left_q: Sequence[complex],
    right_q: Sequence[complex],
) -> float:
    shifts = []
    for left, right in zip(left_q, right_q):
        left_tau = tau_from_q(complex(left))
        right_tau = tau_from_q(complex(right))
        difference = left_tau - right_tau
        difference -= round(difference.real)
        shifts.append(abs(difference))
    return float(max(shifts))


def validate_or_refine_period_map(
    topology: str,
    target: np.ndarray,
    initial_q: Sequence[complex],
    *,
    word_length: int,
    word_step: int,
    tolerance: float,
    reinverse_validation_word_length: int,
    reinverse_max_nfev: int,
    initial_log_q: Sequence[complex] | None = None,
    schottky_envelope: SchottkyValidityEnvelope | None = None,
) -> PeriodMapValidation:
    """Validate one saved ``q`` with the calibrated hybrid period-map policy."""

    initial_tuple = tuple(complex(value) for value in initial_q)
    initial_log_tuple = (
        tuple(complex(value) for value in initial_log_q)
        if initial_log_q is not None
        else None
    )
    hybrid_config = HybridPeriodMapConfig(
        tolerance=float(tolerance),
        agreement_tolerance=float(tolerance),
        minimum_schottky_word=max(3, int(word_length) - int(word_step)),
        maximum_schottky_word=max(
            int(reinverse_validation_word_length),
            int(word_length) + 2,
        ),
    )
    hybrid = hybrid_period_matrix(
        topology,  # type: ignore[arg-type]
        initial_tuple,
        config=hybrid_config,
        log_q_values=initial_log_tuple,
        schottky_envelope=schottky_envelope,
    )
    if is_schottky_algorithm(hybrid.algorithm):
        schottky = hybrid.schottky
        if schottky is None:
            raise RuntimeError("hybrid cusp selection returned no Schottky diagnostics")
        fixed_residual = period_max_residual(hybrid.omega, target)
        if fixed_residual <= tolerance:
            return PeriodMapValidation(
                q=initial_tuple,  # type: ignore[arg-type]
                period_algorithm=hybrid.algorithm,
                refined=False,
                fixed_q_residual=fixed_residual,
                fixed_q_stability=max(
                    schottky.error_estimate, hybrid.overlap_residual or 0.0
                ),
                final_residual=fixed_residual,
                final_stability=max(
                    schottky.error_estimate, hybrid.overlap_residual or 0.0
                ),
                seam_residual=math.nan,
                symmetry_error=schottky.symmetry_error,
                low_order=schottky.low_order,
                high_order=schottky.high_order,
                validation_order=schottky.high_order,
                reinverse_success=True,
                reinverse_message=(
                    "fixed q passed the adaptive hybrid Schottky certificate"
                ),
                reinverse_nfev=0,
                max_tau_shift=0.0,
                log_q=initial_log_tuple,
                validity_cell_id=schottky.validity_cell_id,
                certified_error_bound=schottky.error_estimate,
                validity_reference_table_sha256=(
                    schottky.validity_reference_table_sha256
                ),
                period_map_region=hybrid.region,
                overlap_residual=hybrid.overlap_residual,
                agreement_tolerance=hybrid.agreement_tolerance,
            )

        refined_schottky = refine_schottky_inverse(
            topology,  # type: ignore[arg-type]
            target,
            initial_tuple,
            initial_log_q=initial_log_tuple,
            config=hybrid_config,
            fit_word_length=max(5, int(word_length)),
            max_nfev=reinverse_max_nfev,
        )
        if not refined_schottky.success:
            raise RuntimeError(
                "adaptive Schottky period-map re-inversion failed certification: "
                f"residual={refined_schottky.residual:.3e}, "
                f"stability={refined_schottky.stability:.3e}"
            )
        return PeriodMapValidation(
            q=refined_schottky.q,
            period_algorithm=hybrid.algorithm,
            refined=True,
            fixed_q_residual=fixed_residual,
            fixed_q_stability=max(
                schottky.error_estimate, hybrid.overlap_residual or 0.0
            ),
            final_residual=refined_schottky.residual,
            final_stability=refined_schottky.stability,
            seam_residual=math.nan,
            symmetry_error=refined_schottky.symmetry_error,
            low_order=refined_schottky.word_length,
            high_order=refined_schottky.validation_word_length,
            validation_order=refined_schottky.validation_word_length,
            reinverse_success=True,
            reinverse_message=refined_schottky.message,
            reinverse_nfev=refined_schottky.nfev,
            max_tau_shift=tau_shift_mod_integer(refined_schottky.q, initial_tuple),
            log_q=refined_schottky.log_q,
            certified_error_bound=refined_schottky.stability,
            period_map_region=refined_schottky.region,
            overlap_residual=refined_schottky.overlap_residual,
            agreement_tolerance=hybrid_config.agreement_tolerance,
        )

    if hybrid.algorithm == MULTIPRECISION_HOLOMORPHIC_ALGORITHM:
        holomorphic = hybrid.holomorphic
        if holomorphic is None or not holomorphic.used_multiprecision:
            raise RuntimeError(
                "mixed-cusp selection returned no multiprecision holomorphic diagnostics"
            )
        fixed_residual = period_max_residual(hybrid.omega, target)
        fixed_stability = max(
            holomorphic.error_estimate, hybrid.overlap_residual or 0.0
        )
        if fixed_residual <= tolerance and holomorphic.converged:
            return PeriodMapValidation(
                q=initial_tuple,  # type: ignore[arg-type]
                period_algorithm=hybrid.algorithm,
                refined=False,
                fixed_q_residual=fixed_residual,
                fixed_q_stability=fixed_stability,
                final_residual=fixed_residual,
                final_stability=fixed_stability,
                seam_residual=holomorphic.seam_residual,
                symmetry_error=holomorphic.symmetry_error,
                low_order=holomorphic.low_order,
                high_order=holomorphic.high_order,
                validation_order=holomorphic.high_order,
                reinverse_success=True,
                reinverse_message=(
                    "fixed q passed the rescaled multiprecision holomorphic certificate"
                ),
                reinverse_nfev=0,
                max_tau_shift=0.0,
                log_q=(
                    initial_log_tuple
                    if initial_log_tuple is not None
                    else tuple(complex(np.log(value)) for value in initial_tuple)
                ),
                certified_error_bound=fixed_stability,
                period_map_region=hybrid.region,
                overlap_residual=hybrid.overlap_residual,
                agreement_tolerance=hybrid.agreement_tolerance,
            )
        refined_mp = refine_multiprecision_holomorphic_inverse(
            topology,  # type: ignore[arg-type]
            target,
            initial_tuple,
            initial_log_q=initial_log_tuple,
            config=hybrid_config,
            max_nfev=reinverse_max_nfev,
        )
        if not refined_mp.success:
            raise RuntimeError(
                "multiprecision holomorphic re-inversion failed certification: "
                f"residual={refined_mp.residual:.3e}, "
                f"stability={refined_mp.stability:.3e}"
            )
        return PeriodMapValidation(
            q=refined_mp.q,
            period_algorithm=MULTIPRECISION_HOLOMORPHIC_ALGORITHM,
            refined=True,
            fixed_q_residual=fixed_residual,
            fixed_q_stability=fixed_stability,
            final_residual=refined_mp.residual,
            final_stability=refined_mp.stability,
            seam_residual=refined_mp.seam_residual,
            symmetry_error=refined_mp.symmetry_error,
            low_order=refined_mp.low_order,
            high_order=refined_mp.high_order,
            validation_order=refined_mp.high_order,
            reinverse_success=True,
            reinverse_message=refined_mp.message,
            reinverse_nfev=refined_mp.nfev,
            max_tau_shift=tau_shift_mod_integer(refined_mp.q, initial_tuple),
            log_q=refined_mp.log_q,
            certified_error_bound=refined_mp.stability,
            period_map_region=refined_mp.region,
            overlap_residual=None,
            agreement_tolerance=hybrid_config.agreement_tolerance,
        )

    low_order, low_samples, high_order, high_samples, validation_order, validation_samples = (
        collocation_orders(topology, initial_tuple)
    )
    # When one tube is already exponentially long, the Laurent collocation
    # matrix loses a few digits even though the normalized periods remain
    # stable.  Keep the period residual at the requested tolerance, while
    # allowing the independent seam diagnostic its observed conditioning
    # floor in this mixed bulk/cusp regime.
    seam_floor = (
        1.0e-5
        if min(abs(value) for value in initial_tuple) < 1.0e-10
        else 1.0e-6
    )
    diagnostic_tolerance = max(float(tolerance), seam_floor)
    low, _, _ = collocation_period_from_q(
        topology,
        initial_tuple,
        basis_order=low_order,
        samples_per_seam=low_samples,
    )
    sample_factor = 4 if topology == "theta" else 6
    while True:
        high, seam_residual, symmetry_error = collocation_period_from_q(
            topology,
            initial_tuple,
            basis_order=high_order,
            samples_per_seam=high_samples,
        )
        if (
            seam_residual <= diagnostic_tolerance
            and symmetry_error <= diagnostic_tolerance
        ) or high_order >= 72:
            break
        high_order += 4
        high_samples = sample_factor * high_order
        validation_order = high_order + 4
        validation_samples = sample_factor * validation_order
    fixed_residual = period_max_residual(high, target)
    fixed_stability = period_max_residual(high, low)
    if (
        fixed_residual <= tolerance
        and fixed_stability <= tolerance
        and seam_residual <= diagnostic_tolerance
        and symmetry_error <= diagnostic_tolerance
    ):
        return PeriodMapValidation(
            q=initial_tuple,  # type: ignore[arg-type]
            period_algorithm="holomorphic-form-collocation",
            refined=False,
            fixed_q_residual=fixed_residual,
            fixed_q_stability=max(
                fixed_stability, hybrid.overlap_residual or 0.0
            ),
            final_residual=fixed_residual,
            final_stability=max(
                fixed_stability, hybrid.overlap_residual or 0.0
            ),
            seam_residual=seam_residual,
            symmetry_error=symmetry_error,
            low_order=low_order,
            high_order=high_order,
            validation_order=high_order,
            reinverse_success=True,
            reinverse_message="fixed q passed the holomorphic-form certificate",
            reinverse_nfev=0,
            max_tau_shift=0.0,
            period_map_region=hybrid.region,
            overlap_residual=hybrid.overlap_residual,
            agreement_tolerance=hybrid.agreement_tolerance,
        )

    refined = refine_inverse_collocation(
        topology,
        target,
        initial_tuple,
        basis_order=high_order,
        samples_per_seam=high_samples,
        max_nfev=reinverse_max_nfev,
    )
    fit, fit_seam, fit_symmetry = collocation_period_from_q(
        topology,
        refined.q,
        basis_order=high_order,
        samples_per_seam=high_samples,
    )
    validation, validation_seam, validation_symmetry = collocation_period_from_q(
        topology,
        refined.q,
        basis_order=validation_order,
        samples_per_seam=validation_samples,
    )
    final_hybrid = hybrid_period_matrix(
        topology,  # type: ignore[arg-type]
        refined.q,
        config=hybrid_config,
    )
    final_residual = period_max_residual(final_hybrid.omega, target)
    final_stability = max(
        period_max_residual(validation, fit),
        final_hybrid.error_estimate,
        final_hybrid.overlap_residual or 0.0,
    )
    final_seam = max(refined.seam_residual, fit_seam, validation_seam)
    final_symmetry = max(refined.symmetry_error, fit_symmetry, validation_symmetry)
    if (
        not refined.success
        or refined.fit_residual > tolerance
        or final_residual > tolerance
        or final_stability > tolerance
        or final_seam > diagnostic_tolerance
        or final_symmetry > diagnostic_tolerance
    ):
        raise RuntimeError(
            "holomorphic-form period-map re-inversion failed certification: "
            f"success={refined.success}, fit={refined.fit_residual:.3e}, "
            f"validation={final_residual:.3e}, step={final_stability:.3e}, "
            f"seam={final_seam:.3e}, symmetry={final_symmetry:.3e}, "
            f"tolerance={tolerance:.3e}"
        )
    return PeriodMapValidation(
        q=refined.q,
        period_algorithm=final_hybrid.algorithm,
        refined=True,
        fixed_q_residual=fixed_residual,
        fixed_q_stability=fixed_stability,
        final_residual=final_residual,
        final_stability=final_stability,
        seam_residual=final_seam,
        symmetry_error=final_symmetry,
        low_order=low_order,
        high_order=high_order,
        validation_order=validation_order,
        reinverse_success=refined.success,
        reinverse_message=refined.message,
        reinverse_nfev=refined.nfev,
        max_tau_shift=tau_shift_mod_integer(refined.q, initial_tuple),
        period_map_region=final_hybrid.region,
        overlap_residual=final_hybrid.overlap_residual,
        agreement_tolerance=final_hybrid.agreement_tolerance,
    )


def _source_omega(row: dict[str, str]) -> np.ndarray:
    return np.asarray(
        [
            [
                float(row["x11"]) + 1j * float(row["y11"]),
                float(row["x12"]) + 1j * float(row["y12"]),
            ],
            [
                float(row["x12"]) + 1j * float(row["y12"]),
                float(row["x22"]) + 1j * float(row["y22"]),
            ],
        ],
        dtype=np.complex128,
    )


def _select_reinverse_indices(rows: Sequence[dict[str, object]], count: int) -> set[int]:
    order = sorted(range(len(rows)), key=lambda index: float(rows[index]["q_max"]))
    positions = np.linspace(0, len(order) - 1, min(int(count), len(order))).round().astype(int)
    selected = {order[int(position)] for position in positions}
    selected.update(
        index for index, row in enumerate(rows) if row["topology"] == "glasses"
    )
    return selected


def audit_rows(
    source_rows: Sequence[dict[str, str]],
    *,
    reinverse_count: int,
    reinverse_max_nfev: int,
    symplectic_depth: int,
    period_tolerance: float,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Audit saved coordinates with collocation at two Laurent orders."""

    matrix_by_word = dict(enumerate_symplectic_words(symplectic_depth))
    rows: list[dict[str, object]] = []

    def source_marking(source: dict[str, str]) -> np.ndarray:
        saved = symplectic_matrix_from_csv_row(source)
        if saved is not None:
            return saved
        word = source["symplectic_word"]
        if word not in matrix_by_word:
            raise ValueError(
                f"symplectic word {word!r} is absent at depth {symplectic_depth} "
                "and the row has no saved exact marking"
            )
        return matrix_by_word[word]

    for source in source_rows:
        word = source["symplectic_word"]
        marking = source_marking(source)
        target = transform_omega(marking, _source_omega(source))
        target = 0.5 * (target + target.T)
        q_values = tuple(complex(source[key]) for key in ("q1", "q2", "q3"))
        topology = source["topology"]
        low_order, low_samples, high_order, high_samples, val_order, val_samples = (
            collocation_orders(topology, q_values)
        )
        row: dict[str, object] = {
            "sample_index": int(source["sample_index"]),
            "topology": topology,
            "search_stage": source["search_stage"],
            "symplectic_word": word,
            **symplectic_matrix_csv_fields(marking),
            "q1": source["q1"],
            "q2": source["q2"],
            "q3": source["q3"],
            "q_max": max(abs(value) for value in q_values),
            "saved_period_residual": float(source["period_residual"]),
            "saved_map_stability": float(
                source.get("period_map_stability", source.get("period_word_stability", "nan"))
            ),
            "period_algorithm": "holomorphic-form-collocation",
            "low_basis_order": low_order,
            "low_samples_per_seam": low_samples,
            "high_basis_order": high_order,
            "high_samples_per_seam": high_samples,
            "validation_basis_order": val_order,
            "validation_samples_per_seam": val_samples,
        }
        if min(abs(value) for value in q_values) < DEEP_CUSP_THRESHOLD:
            saved_logs = (
                tuple(complex(source[f"log_q{edge}"]) for edge in (1, 2, 3))
                if all(source.get(f"log_q{edge}", "") for edge in (1, 2, 3))
                else None
            )
            hybrid = hybrid_period_matrix(
                topology,  # type: ignore[arg-type]
                q_values,
                config=HybridPeriodMapConfig(
                    tolerance=float(period_tolerance),
                    agreement_tolerance=float(period_tolerance),
                ),
                log_q_values=saved_logs,
            )
            selected_method = (
                hybrid.schottky
                if is_schottky_algorithm(hybrid.algorithm)
                else hybrid.holomorphic
            )
            if selected_method is None:
                raise RuntimeError("deep-cusp hybrid audit returned no method diagnostics")
            fixed_residual = period_max_residual(hybrid.omega, target)
            row.update(
                {
                    "period_algorithm": hybrid.algorithm,
                    "period_map_region": hybrid.region,
                    "support_error": "",
                    "low_basis_order": selected_method.low_order,
                    "high_basis_order": selected_method.high_order,
                    "validation_basis_order": selected_method.high_order,
                    "fixed_q_period_residual": fixed_residual,
                    "fixed_q_basis_step": selected_method.error_estimate,
                    "fixed_q_seam_residual": selected_method.seam_residual,
                    "fixed_q_symmetry_error": selected_method.symmetry_error,
                    "fixed_q_pass": (
                        fixed_residual <= period_tolerance
                        and selected_method.error_estimate <= period_tolerance
                        and selected_method.symmetry_error <= period_tolerance
                    ),
                }
            )
            rows.append(row)
            continue

        low, low_seam, low_symmetry = collocation_period_from_q(
            topology,
            q_values,
            basis_order=low_order,
            samples_per_seam=low_samples,
        )
        high, high_seam, high_symmetry = collocation_period_from_q(
            topology,
            q_values,
            basis_order=high_order,
            samples_per_seam=high_samples,
        )
        diagnostic_tolerance = max(
            float(period_tolerance),
            1.0e-5 if min(abs(value) for value in q_values) < 1.0e-10 else 1.0e-6,
        )
        fixed_residual = period_max_residual(high, target)
        fixed_step = period_max_residual(high, low)
        seam_residual = max(low_seam, high_seam)
        symmetry_error = max(low_symmetry, high_symmetry)
        row.update(
            {
                "support_error": "",
                "fixed_q_period_residual": fixed_residual,
                "fixed_q_basis_step": fixed_step,
                "fixed_q_seam_residual": seam_residual,
                "fixed_q_symmetry_error": symmetry_error,
                "fixed_q_pass": (
                    fixed_residual <= period_tolerance
                    and fixed_step <= period_tolerance
                    and seam_residual <= diagnostic_tolerance
                    and symmetry_error <= diagnostic_tolerance
                ),
            }
        )
        rows.append(row)

    selected = _select_reinverse_indices(rows, reinverse_count)
    selected.update(
        index
        for index, row in enumerate(rows)
        if not row["fixed_q_pass"]
        and row["period_algorithm"] == "holomorphic-form-collocation"
    )
    for index, row in enumerate(rows):
        supported = row["period_algorithm"] == "holomorphic-form-collocation"
        reinverse_selected = index in selected and supported
        row["collocation_reinverse_selected"] = reinverse_selected
        row["collocation_reinverse_success"] = False
        row["collocation_reinverse_message"] = ""
        row["collocation_reinverse_nfev"] = 0
        row["collocation_fit_residual"] = math.nan
        row["collocation_validation_residual"] = math.nan
        row["collocation_validation_step"] = math.nan
        row["collocation_validation_seam_residual"] = math.nan
        row["collocation_validation_symmetry_error"] = math.nan
        row["max_tau_shift_from_saved_q"] = math.nan
        row["max_q_abs_relative_shift"] = math.nan
        row["final_period_map_pass"] = bool(row["fixed_q_pass"])
        if not reinverse_selected:
            continue
        source = source_rows[index]
        target = transform_omega(source_marking(source), _source_omega(source))
        target = 0.5 * (target + target.T)
        old_q = tuple(complex(source[key]) for key in ("q1", "q2", "q3"))
        high_order = int(row["high_basis_order"])
        high_samples = int(row["high_samples_per_seam"])
        validation_order = int(row["validation_basis_order"])
        validation_samples = int(row["validation_samples_per_seam"])
        refined = refine_inverse_collocation(
            source["topology"],
            target,
            old_q,
            basis_order=high_order,
            samples_per_seam=high_samples,
            max_nfev=reinverse_max_nfev,
        )
        fit_forward, fit_seam, fit_symmetry = collocation_period_from_q(
            source["topology"],
            refined.q,
            basis_order=high_order,
            samples_per_seam=high_samples,
        )
        validation, validation_seam, validation_symmetry = collocation_period_from_q(
            source["topology"],
            refined.q,
            basis_order=validation_order,
            samples_per_seam=validation_samples,
        )
        validation_residual = period_max_residual(validation, target)
        validation_step = period_max_residual(validation, fit_forward)
        seam_residual = max(refined.seam_residual, fit_seam, validation_seam)
        symmetry_error = max(refined.symmetry_error, fit_symmetry, validation_symmetry)
        diagnostic_tolerance = max(
            float(period_tolerance),
            1.0e-5 if min(abs(value) for value in refined.q) < 1.0e-10 else 1.0e-6,
        )
        row.update(
            {
                "collocation_reinverse_success": refined.success,
                "collocation_reinverse_message": refined.message,
                "collocation_reinverse_nfev": refined.nfev,
                "collocation_fit_residual": refined.fit_residual,
                "collocation_validation_residual": validation_residual,
                "collocation_validation_step": validation_step,
                "collocation_validation_seam_residual": seam_residual,
                "collocation_validation_symmetry_error": symmetry_error,
                "max_tau_shift_from_saved_q": tau_shift_mod_integer(
                    refined.q, old_q
                ),
                "max_q_abs_relative_shift": max(
                    abs(abs(new) / abs(old) - 1.0)
                    for new, old in zip(refined.q, old_q)
                ),
                "refined_q1": repr(refined.q[0]),
                "refined_q2": repr(refined.q[1]),
                "refined_q3": repr(refined.q[2]),
                "final_period_map_pass": (
                    refined.success
                    and refined.fit_residual <= period_tolerance
                    and validation_residual <= period_tolerance
                    and validation_step <= period_tolerance
                    and seam_residual <= diagnostic_tolerance
                    and symmetry_error <= diagnostic_tolerance
                ),
            }
        )

    fixed_failures = [row for row in rows if not row["fixed_q_pass"]]
    reinverted = [row for row in rows if row["collocation_reinverse_selected"]]

    def quantiles(key: str, selected_rows: Sequence[dict[str, object]]) -> dict[str, float]:
        values = np.asarray([float(row[key]) for row in selected_rows], dtype=float)
        values = values[np.isfinite(values)]
        if not len(values):
            return {}
        return {
            "minimum": float(np.min(values)),
            "median": float(np.quantile(values, 0.5)),
            "q90": float(np.quantile(values, 0.9)),
            "maximum": float(np.max(values)),
        }

    summary = {
        "sample_count": len(rows),
        "period_algorithm": "adaptive-hybrid",
        "period_tolerance": period_tolerance,
        "fixed_q_pass_count": len(rows) - len(fixed_failures),
        "fixed_q_failure_count": len(fixed_failures),
        "fixed_q_failure_indices": [row["sample_index"] for row in fixed_failures],
        "unsupported_count": sum(
            bool(row.get("support_error")) for row in rows
        ),
        "fixed_q_period_residual_quantiles": quantiles(
            "fixed_q_period_residual", rows
        ),
        "fixed_q_basis_step_quantiles": quantiles(
            "fixed_q_basis_step", rows
        ),
        "reinverse_count": len(reinverted),
        "reinverse_success_count": sum(
            bool(row["collocation_reinverse_success"]) for row in reinverted
        ),
        "final_period_map_pass_count": sum(
            bool(row["final_period_map_pass"]) for row in rows
        ),
        "final_period_map_failure_indices": [
            row["sample_index"] for row in rows if not row["final_period_map_pass"]
        ],
        "reinverse_tau_shift_quantiles": quantiles(
            "max_tau_shift_from_saved_q", reinverted
        ),
        "reinverse_validation_residual_quantiles": quantiles(
            "collocation_validation_residual", reinverted
        ),
        "interpretation": (
            "Bulk period matrices are computed from normalized holomorphic one-forms; "
            "long cusps use adaptive Schottky words. Low/high/validation orders "
            "separate truncation from nonlinear optimizer error, and binary64 "
            "underflow points retain their supplied log(q) coordinates."
        ),
    }
    return rows, summary


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_plot(path: Path, rows: Sequence[dict[str, object]], tolerance: float) -> None:
    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(tempfile.gettempdir()) / "stringmc-matplotlib"),
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    q_max = np.asarray([float(row["q_max"]) for row in rows])
    residual = np.asarray([float(row["fixed_q_period_residual"]) for row in rows])
    step = np.asarray([float(row["fixed_q_basis_step"]) for row in rows])
    selected = np.asarray(
        [bool(row["collocation_reinverse_selected"]) for row in rows]
    )
    tau_shift = np.asarray(
        [float(row["max_tau_shift_from_saved_q"]) for row in rows]
    )

    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.7))
    axes[0].semilogy(q_max, residual, "o", color="#16717c", alpha=0.85)
    axes[0].axhline(tolerance, color="#a34a34", linestyle="--")
    axes[0].set_title("Saved q against high-basis Omega")
    axes[0].set_ylabel("max period residual")

    axes[1].semilogy(q_max, step, "o", color="#c28c2c", alpha=0.85)
    axes[1].axhline(tolerance, color="#a34a34", linestyle="--")
    axes[1].set_title("Low-to-high Laurent-basis step")
    axes[1].set_ylabel("max period change")

    axes[2].semilogy(q_max[selected], tau_shift[selected], "o", color="#704c8c")
    axes[2].set_title("q movement after collocation re-inversion")
    axes[2].set_ylabel(r"max $|\Delta\tau_e|$ modulo integers")
    for axis in axes:
        axis.set_xlabel(r"saved $\max |q_e|$")
        axis.grid(True, which="both", color="#d9d9d6", alpha=0.8)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Genus-two plumbing period-map accuracy", fontsize=15, fontweight="semibold")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def run(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Audit saved q-to-Omega accuracy with holomorphic one-forms."
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--reinverse-count", type=int, default=12)
    parser.add_argument("--reinverse-max-nfev", type=int, default=80)
    parser.add_argument("--symplectic-depth", type=int, default=4)
    parser.add_argument("--period-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(list(argv) if argv is not None else None)

    source_rows = list(csv.DictReader(args.input_csv.open()))
    rows, summary = audit_rows(
        source_rows,
        reinverse_count=args.reinverse_count,
        reinverse_max_nfev=args.reinverse_max_nfev,
        symplectic_depth=args.symplectic_depth,
        period_tolerance=args.period_tolerance,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "q_to_omega_audit.csv"
    json_path = args.out_dir / "summary.json"
    png_path = args.out_dir / "q_to_omega_accuracy.png"
    _write_csv(csv_path, rows)
    json_path.write_text(
        json.dumps(
            {
                "scope": (
                    "Normalized-holomorphic-form forward residuals for all saved "
                    "nodes and higher-basis re-inversion on a q-stratified subset."
                ),
                "input_csv": str(args.input_csv),
                **summary,
            },
            indent=2,
        )
        + "\n"
    )
    _write_plot(png_path, rows, args.period_tolerance)
    print("Genus-two q-to-Omega accuracy audit")
    print(
        f"  fixed-q pass={summary['fixed_q_pass_count']}/"
        f"{summary['sample_count']} at tolerance {args.period_tolerance:.1e}"
    )
    print(
        f"  final pass after selective re-inversion="
        f"{summary['final_period_map_pass_count']}/{summary['sample_count']}"
    )
    print(
        f"  high-basis residual max="
        f"{summary['fixed_q_period_residual_quantiles']['maximum']:.3e}"
    )
    print(
        f"  reinverse success={summary['reinverse_success_count']}/"
        f"{summary['reinverse_count']}; max tau shift="
        f"{summary['reinverse_tau_shift_quantiles'].get('maximum', math.nan):.3e}"
    )
    print(f"  wrote {csv_path}")
    print(f"  wrote {json_path}")
    print(f"  wrote {png_path}")


if __name__ == "__main__":
    run()
