#!/usr/bin/env python3
"""Guarded Schottky period evaluation for calibrated regions only.

This module intentionally exposes no unguarded production ``q -> Omega``
function.  Callers must provide a :class:`SchottkyValidityEnvelope` built by
comparison with normalized holomorphic one-forms.  A request outside that
envelope fails instead of guessing from a scalar ``q`` threshold.
"""

from __future__ import annotations

from c1_paths import PACKAGE_ROOT, data_root, package_path, module_path

from dataclasses import dataclass
from typing import Sequence

import numpy as np

try:
    from genus2_holomorphic_period_table import (
        CALIBRATED_SCHOTTKY_ALGORITHM,
        SchottkyValidityCertificate,
        SchottkyValidityEnvelope,
        Topology,
    )
    from plumbing_algorithms import (
        generators_for_glasses,
        generators_for_theta,
        schottky_period_matrix_cross_ratio,
    )
except ImportError:  # pragma: no cover - package-style execution
    from genus2_holomorphic_period_table import (
        CALIBRATED_SCHOTTKY_ALGORITHM,
        SchottkyValidityCertificate,
        SchottkyValidityEnvelope,
        Topology,
    )
    from plumbing_algorithms import (
        generators_for_glasses,
        generators_for_theta,
        schottky_period_matrix_cross_ratio,
    )


class UncertifiedSchottkyRegion(RuntimeError):
    """Raised when a query has no sufficiently accurate calibrated cell."""


@dataclass(frozen=True)
class CalibratedSchottkyEvaluation:
    omega: np.ndarray
    symmetry_error: float
    word_stability: float
    algorithm: str
    certificate: SchottkyValidityCertificate


def _period_at_word(
    topology: Topology,
    q_values: Sequence[complex],
    word_length: int,
) -> tuple[np.ndarray, float]:
    q = tuple(complex(value) for value in q_values)
    generators = (
        generators_for_theta(*q)
        if topology == "theta"
        else generators_for_glasses(*q)
    )
    raw = np.asarray(
        schottky_period_matrix_cross_ratio(
            generators,
            max_word_len=int(word_length),
        ),
        dtype=np.complex128,
    )
    if raw.shape != (2, 2) or not np.all(np.isfinite(raw)):
        raise FloatingPointError("calibrated Schottky evaluation returned a nonfinite matrix")
    branch = int(round((raw[0, 1] - raw[1, 0]).real))
    lower_aligned = raw[1, 0] + branch
    symmetry_error = float(abs(raw[0, 1] - lower_aligned))
    off_diagonal = 0.5 * (raw[0, 1] + lower_aligned)
    omega = np.asarray(
        [[raw[0, 0], off_diagonal], [off_diagonal, raw[1, 1]]],
        dtype=np.complex128,
    )
    return omega, symmetry_error


def calibrated_schottky_period_from_q(
    topology: Topology,
    q_values: Sequence[complex],
    *,
    envelope: SchottkyValidityEnvelope,
    tolerance: float,
) -> CalibratedSchottkyEvaluation:
    """Evaluate Schottky only under a holomorphic-reference certificate."""

    certificate = envelope.certificate(
        topology,
        q_values,
        tolerance=float(tolerance),
    )
    if certificate is None:
        raise UncertifiedSchottkyRegion(
            "q is outside the calibrated Schottky validity envelope at "
            f"tolerance {float(tolerance):.3e}"
        )
    high, symmetry_error = _period_at_word(
        topology,
        q_values,
        certificate.word_length,
    )
    low, _ = _period_at_word(
        topology,
        q_values,
        certificate.word_length - 1,
    )
    difference = high - low
    integer_branch = np.rint(difference.real).astype(int)
    integer_branch = np.rint(0.5 * (integer_branch + integer_branch.T)).astype(int)
    word_stability = float(np.max(np.abs(difference - integer_branch)))
    if max(symmetry_error, word_stability) > certificate.error_bound:
        raise RuntimeError(
            "Schottky query violated its calibrated runtime bound: "
            f"symmetry={symmetry_error:.3e}, word_step={word_stability:.3e}, "
            f"bound={certificate.error_bound:.3e}, cell={certificate.cell_id!r}"
        )
    return CalibratedSchottkyEvaluation(
        omega=high,
        symmetry_error=symmetry_error,
        word_stability=word_stability,
        algorithm=CALIBRATED_SCHOTTKY_ALGORITHM,
        certificate=certificate,
    )
