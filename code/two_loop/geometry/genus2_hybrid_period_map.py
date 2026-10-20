#!/usr/bin/env python3
"""Adaptive full-chart genus-two ``q -> Omega`` period-map policy.

The two numerical representations have complementary conditioning:

* normalized-holomorphic-form collocation is preferred in the ordinary bulk;
* the Schottky cross-ratio series is preferred in long plumbing cusps;
* in their common domain both are evaluated and must agree modulo a symmetric
  integral B-period shift.

There is no scalar-q hole between the two methods.  Every *geometrically valid*
theta or glasses plumbing chart is assigned to a preferred method.  The
transition path raises Laurent/word cutoffs and tries both methods before it
fails.  A chart whose excised coordinate disks overlap is rejected: covering
that moduli point requires another pants decomposition, not a more expensive
evaluation of an invalid chart.
"""

from __future__ import annotations

from c1_paths import PACKAGE_ROOT, data_root, package_path, module_path

import cmath
import math
from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

try:
    from ccy_plumbing_conventions import validate_genus2_plumbing_coordinates
    from genus2_holomorphic_period_table import SchottkyValidityEnvelope
    from genus2_calibrated_schottky import (
        UncertifiedSchottkyRegion,
        calibrated_schottky_period_from_q,
    )
    from plumbing_algorithms import (
        genus2_symmetric_period_vector,
        generators_for_glasses,
        generators_for_theta,
        glasses_collocation_period_matrix,
        q_from_tau,
        schottky_period_matrix_cross_ratio,
        schottky_period_matrix_cross_ratio_multiprecision,
        solve_theta_collocation,
        symmetrized_period_matrix,
    )
except ImportError:  # pragma: no cover - package-style execution
    from ccy_plumbing_conventions import (
        validate_genus2_plumbing_coordinates,
    )
    from genus2_holomorphic_period_table import SchottkyValidityEnvelope
    from genus2_calibrated_schottky import (
        UncertifiedSchottkyRegion,
        calibrated_schottky_period_from_q,
    )
    from plumbing_algorithms import (
        genus2_symmetric_period_vector,
        generators_for_glasses,
        generators_for_theta,
        glasses_collocation_period_matrix,
        q_from_tau,
        schottky_period_matrix_cross_ratio,
        schottky_period_matrix_cross_ratio_multiprecision,
        solve_theta_collocation,
        symmetrized_period_matrix,
    )


Topology = Literal["theta", "glasses"]
TWO_PI_I = 2.0j * math.pi
HOLOMORPHIC_ALGORITHM = "holomorphic-form-collocation"
MULTIPRECISION_HOLOMORPHIC_ALGORITHM = (
    "multiprecision-rescaled-holomorphic-collocation"
)
SCHOTTKY_ALGORITHM = "adaptive-schottky"
CALIBRATED_SCHOTTKY_ALGORITHM = "calibrated-schottky"


class InvalidPlumbingGeometry(ValueError):
    """Raised when the standard plumbing disks overlap in the requested chart."""


class HybridPeriodMapFailure(RuntimeError):
    """Raised when neither numerical representation reaches the requested bar."""


@dataclass(frozen=True)
class HybridPeriodMapConfig:
    """Numerical bars and work ceilings for the hybrid period map."""

    tolerance: float = 1.0e-6
    agreement_tolerance: float = 1.0e-6
    collocation_min_q: float = 1.0e-12
    collocation_comfortable_min_q: float = 1.0e-10
    schottky_all_small_q_max_theta: float = 0.15
    schottky_all_small_q_max_glasses: float = 0.20
    method_boundary_log_half_width: float = 0.18
    # Deprecated compatibility field.  New dispatch uses the two explicit
    # topology-dependent all-small thresholds above.
    overlap_q_max: float = 0.16
    minimum_geometry_margin: float = 1.0e-10
    comfortable_geometry_margin: float = 2.0e-2
    maximum_collocation_basis: int = 72
    minimum_schottky_word: int = 4
    maximum_schottky_word: int = 9
    schottky_tail_safety_factor: float = 2.0
    crosscheck_overlap: bool = True
    require_convergence: bool = True

    def __post_init__(self) -> None:
        positive = (
            self.tolerance,
            self.agreement_tolerance,
            self.collocation_min_q,
            self.collocation_comfortable_min_q,
            self.schottky_all_small_q_max_theta,
            self.schottky_all_small_q_max_glasses,
            self.method_boundary_log_half_width,
            self.overlap_q_max,
            self.minimum_geometry_margin,
            self.comfortable_geometry_margin,
            self.schottky_tail_safety_factor,
        )
        if any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in positive):
            raise ValueError("hybrid period-map tolerances and thresholds must be positive")
        if self.collocation_comfortable_min_q < self.collocation_min_q:
            raise ValueError("comfortable collocation threshold cannot be below its hard floor")
        if self.minimum_geometry_margin > self.comfortable_geometry_margin:
            raise ValueError("minimum geometry margin cannot exceed the comfortable margin")
        if not (
            self.schottky_all_small_q_max_theta < 1.0
            and self.schottky_all_small_q_max_glasses < 1.0
        ):
            raise ValueError("all-small Schottky thresholds must be below one")
        if self.maximum_collocation_basis < 8:
            raise ValueError("maximum collocation basis is too small")
        if not (2 <= self.minimum_schottky_word <= self.maximum_schottky_word):
            raise ValueError("invalid Schottky word range")


@dataclass(frozen=True)
class PlumbingGeometry:
    topology: Topology
    # One seam radius per edge.  In theta these are the radii on the first
    # trinion.  In glasses they are respectively the zero-side radii of the
    # left/right self-sewings and the left bridge radius; the opposite radii
    # are fixed by r_h r_hbar = |q_e|.
    radii: tuple[float, float, float]
    sphere_margins: tuple[tuple[float, float, float], ...]
    minimum_margin: float
    valid: bool


@dataclass(frozen=True)
class MethodEvaluation:
    algorithm: str
    omega: np.ndarray
    converged: bool
    error_estimate: float
    low_order: int
    high_order: int
    seam_residual: float
    symmetry_error: float
    used_multiprecision: bool
    calibrated: bool
    message: str
    validity_cell_id: str | None = None
    validity_reference_table_sha256: str | None = None


@dataclass(frozen=True)
class HybridPeriodMapResult:
    topology: Topology
    q: tuple[complex, complex, complex]
    log_q: tuple[complex, complex, complex]
    omega: np.ndarray
    algorithm: str
    region: str
    geometry: PlumbingGeometry
    error_estimate: float
    overlap_residual: float | None
    agreement_tolerance: float
    holomorphic: MethodEvaluation | None
    schottky: MethodEvaluation | None
    selection_reason: str


@dataclass(frozen=True)
class AdaptiveSchottkyInverseResult:
    success: bool
    message: str
    nfev: int
    q: tuple[complex, complex, complex]
    log_q: tuple[complex, complex, complex]
    omega: np.ndarray
    branch: np.ndarray
    residual: float
    stability: float
    symmetry_error: float
    word_length: int
    validation_word_length: int
    used_multiprecision: bool
    region: str
    overlap_residual: float | None


@dataclass(frozen=True)
class MultiprecisionHolomorphicInverseResult:
    """Certified logarithmic inverse for a mixed cusp."""

    success: bool
    message: str
    nfev: int
    q: tuple[complex, complex, complex]
    log_q: tuple[complex, complex, complex]
    omega: np.ndarray
    branch: np.ndarray
    residual: float
    stability: float
    seam_residual: float
    symmetry_error: float
    low_order: int
    high_order: int
    region: str


def is_schottky_algorithm(algorithm: str) -> bool:
    return str(algorithm) in {SCHOTTKY_ALGORITHM, CALIBRATED_SCHOTTKY_ALGORITHM}


def schottky_all_small_limit(
    topology: Topology, config: HybridPeriodMapConfig
) -> float:
    if topology == "theta":
        return float(config.schottky_all_small_q_max_theta)
    if topology == "glasses":
        return float(config.schottky_all_small_q_max_glasses)
    raise ValueError(f"unknown topology {topology!r}")


def _as_q_and_logs(
    q_values: Sequence[complex],
    log_q_values: Sequence[complex] | None,
) -> tuple[tuple[complex, complex, complex], tuple[complex, complex, complex]]:
    q = tuple(complex(value) for value in q_values)
    if len(q) != 3:
        raise ValueError("a genus-two period map needs three plumbing parameters")
    if log_q_values is None:
        if any(
            not (
                math.isfinite(value.real)
                and math.isfinite(value.imag)
                and 0.0 < abs(value) < 1.0
            )
            for value in q
        ):
            raise ValueError("plumbing parameters must be finite and satisfy 0 < |q_e| < 1")
        logs = tuple(cmath.log(value) for value in q)
    else:
        logs = tuple(complex(value) for value in log_q_values)
        if len(logs) != 3 or any(
            not (
                math.isfinite(value.real)
                and math.isfinite(value.imag)
                and value.real < 0.0
            )
            for value in logs
        ):
            raise ValueError("log(q) values must be finite with negative real part")
        if any(abs(value) >= 1.0 for value in q):
            raise ValueError("plumbing parameters must satisfy |q_e| < 1")
        # A zero/surrogate q is allowed only when its true logarithm is supplied.
        q = tuple(
            value
            if value != 0.0j
            else cmath.exp(complex(max(log_value.real, -690.0), log_value.imag))
            for value, log_value in zip(q, logs)
        )  # type: ignore[assignment]
    return q, logs  # type: ignore[return-value]


def _q_abs_from_logs(log_q: Sequence[complex]) -> tuple[float, float, float]:
    return tuple(
        math.exp(value.real) if value.real > -745.0 else 0.0
        for value in log_q
    )  # type: ignore[return-value]


def _sphere_disk_margins(r_zero: float, r_one: float, r_infty: float) -> tuple[float, float, float]:
    """Return clearances for the disks centered at 0, 1, and infinity."""

    return (
        1.0 - r_zero - r_one,
        1.0 / r_infty - r_zero,
        1.0 / r_infty - (1.0 + r_one),
    )


def _theta_q_nonoverlap_slacks(
    q_abs: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Return the three pairwise disk slacks after eliminating all radii."""

    r_zero, r_one, r_infty = (math.sqrt(value) for value in q_abs)
    return (
        1.0 - r_zero - r_one,          # zero versus one
        1.0 - r_zero * r_infty,        # zero versus infinity
        1.0 - r_infty * (1.0 + r_one), # one versus infinity
    )


def _glasses_q_nonoverlap_slacks(
    q_abs: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Return the radius-eliminated left, right, and bridge slacks."""

    left_handle, right_handle, bridge = q_abs
    left_bridge_max = (1.0 - left_handle) / (1.0 + left_handle)
    right_bridge_max = (1.0 - right_handle) / (1.0 + right_handle)
    return (
        1.0 - left_handle,
        1.0 - right_handle,
        left_bridge_max * right_bridge_max - bridge,
    )


def plumbing_nonoverlap_slacks(
    topology: Topology,
    q_values: Sequence[complex],
    *,
    log_q_values: Sequence[complex] | None = None,
) -> tuple[float, float, float]:
    """Return a complete q-only non-overlap certificate.

    Every component is a strict pairwise-disk slack.  The requested plumbing
    graph admits non-overlapping seam circles if and only if all three values
    are positive.  No seam radii are unknowns in this predicate.
    """

    _q, logs = _as_q_and_logs(q_values, log_q_values)
    q_abs = _q_abs_from_logs(logs)
    if topology == "theta":
        return _theta_q_nonoverlap_slacks(q_abs)
    if topology == "glasses":
        return _glasses_q_nonoverlap_slacks(q_abs)
    raise ValueError(f"unknown topology {topology!r}")


def _positive_radius_from_log(log_radius: float) -> float:
    """Exponentiate a seam radius without returning binary64 zero."""

    return math.exp(max(float(log_radius), -690.0))


def _balanced_glasses_radii(
    q_abs: tuple[float, float, float],
    logs: tuple[complex, complex, complex],
) -> tuple[tuple[float, float, float], tuple[tuple[float, float, float], ...], float, bool]:
    """Return a constructive certificate for the complete glasses domain.

    The self-sewing parameters are ``a,b`` and the bridge parameter is ``c``.
    If

        c < ((1-a)/(1+a)) ((1-b)/(1+b)),

    the bridge radii are chosen to use the same fraction of their two maximum
    allowed values.  For each fixed bridge radius, the zero-side handle radius
    is the geometric mean of its allowed interval endpoints.  This balances
    the two logarithmic vertex slacks and supplies stable collocation seams.
    """

    a, b, c = q_abs
    left_bridge_max = (1.0 - a) / (1.0 + a)
    right_bridge_max = (1.0 - b) / (1.0 + b)
    log_left_max = math.log(left_bridge_max)
    log_right_max = math.log(right_bridge_max)
    log_bridge = float(logs[2].real)
    q_slacks = _glasses_q_nonoverlap_slacks(q_abs)
    valid = bool(min(q_slacks) > 0.0)

    if not valid:
        symmetric = tuple(
            max(math.sqrt(value), 1.0e-300) for value in q_abs
        )
        margins = (
            _sphere_disk_margins(symmetric[0], symmetric[2], symmetric[0]),
            _sphere_disk_margins(symmetric[1], symmetric[2], symmetric[1]),
        )
        minimum = min(
            *q_slacks,
            *(value for sphere in margins for value in sphere),
        )
        return symmetric, margins, float(minimum), False

    # u/A = v/B = sqrt(c/(A B)), with u v = c.
    log_left_bridge = 0.5 * (log_left_max + log_bridge - log_right_max)
    log_right_bridge = 0.5 * (log_right_max + log_bridge - log_left_max)
    left_bridge = _positive_radius_from_log(log_left_bridge)
    right_bridge = _positive_radius_from_log(log_right_bridge)

    # For fixed bridge radius u, the left zero-side radius lies in
    # (a(1+u), 1-u).  Its geometric midpoint balances the two constraints.
    log_left_zero = 0.5 * (
        float(logs[0].real) + math.log1p(-(left_bridge * left_bridge))
    )
    log_right_zero = 0.5 * (
        float(logs[1].real) + math.log1p(-(right_bridge * right_bridge))
    )
    left_zero = _positive_radius_from_log(log_left_zero)
    right_zero = _positive_radius_from_log(log_right_zero)
    left_infty = _positive_radius_from_log(float(logs[0].real) - log_left_zero)
    right_infty = _positive_radius_from_log(float(logs[1].real) - log_right_zero)
    margins = (
        _sphere_disk_margins(left_zero, left_bridge, left_infty),
        _sphere_disk_margins(right_zero, right_bridge, right_infty),
    )
    minimum = min(
        *q_slacks,
        *(value for sphere in margins for value in sphere),
    )
    return (
        (left_zero, right_zero, left_bridge),
        margins,
        float(minimum),
        bool(minimum > 0.0),
    )


def plumbing_geometry(
    topology: Topology,
    q_values: Sequence[complex],
    *,
    log_q_values: Sequence[complex] | None = None,
) -> PlumbingGeometry:
    """Return a necessary-and-sufficient standard-trinion seam certificate."""

    q, logs = _as_q_and_logs(q_values, log_q_values)
    q_abs = _q_abs_from_logs(logs)
    radii = tuple(math.sqrt(value) for value in q_abs)
    if any(radius == 0.0 for radius in radii):
        # A truly underflowed cusp disk is disjoint from every finite disk.
        radii = tuple(max(radius, 1.0e-300) for radius in radii)
    if topology == "theta":
        margins = (_sphere_disk_margins(radii[0], radii[1], radii[2]),) * 2
    elif topology == "glasses":
        radii, margins, minimum, valid = _balanced_glasses_radii(q_abs, logs)
        return PlumbingGeometry(
            topology=topology,
            radii=radii,
            sphere_margins=margins,
            minimum_margin=float(minimum),
            valid=valid,
        )
    else:
        raise ValueError(f"unknown topology {topology!r}")
    q_slacks = _theta_q_nonoverlap_slacks(q_abs)
    minimum = min(
        *q_slacks,
        *(value for sphere in margins for value in sphere),
    )
    return PlumbingGeometry(
        topology=topology,
        radii=radii,  # type: ignore[arg-type]
        sphere_margins=margins,
        minimum_margin=float(minimum),
        valid=bool(minimum > 0.0),
    )


def period_difference_mod_integer(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    difference = np.asarray(left, dtype=np.complex128) - np.asarray(
        right, dtype=np.complex128
    )
    branch = np.rint(difference.real).astype(int)
    branch = np.rint(0.5 * (branch + branch.T)).astype(int)
    return difference - branch


def period_max_residual(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.max(np.abs(period_difference_mod_integer(left, right))))


def _collocation_initial_orders(topology: Topology, q_max: float) -> tuple[int, int]:
    if topology == "theta":
        if q_max <= 0.2:
            return 20, 24
        if q_max <= 0.3:
            return 24, 32
        return 32, 40
    if q_max <= 0.2:
        return 20, 28
    if q_max <= 0.3:
        return 24, 32
    return 32, 40


def _collocation_at_order(
    topology: Topology,
    q: tuple[complex, complex, complex],
    order: int,
) -> tuple[np.ndarray, float, float]:
    geometry = plumbing_geometry(topology, q)
    sample_factor = 4 if topology == "theta" else 6
    samples = sample_factor * int(order)
    if topology == "theta":
        result = solve_theta_collocation(
            *q,
            basis_order=int(order),
            samples_per_seam=samples,
            radii=geometry.radii,
        )
        return (
            symmetrized_period_matrix(result.omega),
            float(result.max_seam_residual),
            float(result.omega_symmetry_error),
        )
    omega, seam, symmetry = glasses_collocation_period_matrix(
        *q,
        basis_order=int(order),
        samples_per_seam=samples,
        radii=geometry.radii,
    )
    return symmetrized_period_matrix(omega), float(seam), float(symmetry)


def evaluate_holomorphic_period_map(
    topology: Topology,
    q_values: Sequence[complex],
    *,
    config: HybridPeriodMapConfig,
    log_q_values: Sequence[complex] | None = None,
    allow_below_conditioning_floor: bool = False,
) -> MethodEvaluation:
    """Raise the Laurent basis until collocation passes its local certificate."""

    q, logs = _as_q_and_logs(q_values, log_q_values)
    q_abs = _q_abs_from_logs(logs)
    geometry = plumbing_geometry(topology, q, log_q_values=logs)
    if geometry.minimum_margin <= config.minimum_geometry_margin:
        raise InvalidPlumbingGeometry(
            "standard plumbing disks are not safely disjoint: "
            f"margin={geometry.minimum_margin:.3e}, required>{config.minimum_geometry_margin:.3e}"
        )
    if min(q_abs) < config.collocation_min_q and not allow_below_conditioning_floor:
        raise ValueError(
            "q is below the holomorphic collocation conditioning floor: "
            f"min|q|={min(q_abs):.3e} < {config.collocation_min_q:.3e}"
        )

    low_order, high_order = _collocation_initial_orders(topology, max(q_abs))
    low, low_seam, low_symmetry = _collocation_at_order(topology, q, low_order)
    last = low
    last_order = low_order
    last_seam = low_seam
    last_symmetry = low_symmetry
    while True:
        high, seam, symmetry = _collocation_at_order(topology, q, high_order)
        step = period_max_residual(high, last)
        error = max(step, seam, symmetry, last_seam, last_symmetry)
        converged = bool(
            np.all(np.isfinite(high))
            and math.isfinite(error)
            and error <= config.tolerance
        )
        if converged or high_order >= config.maximum_collocation_basis:
            return MethodEvaluation(
                algorithm=HOLOMORPHIC_ALGORITHM,
                omega=high,
                converged=converged,
                error_estimate=float(error),
                low_order=int(last_order),
                high_order=int(high_order),
                seam_residual=float(max(seam, last_seam)),
                symmetry_error=float(max(symmetry, last_symmetry)),
                used_multiprecision=False,
                calibrated=True,
                message=(
                    "successive Laurent bases, seam residual, and symmetry passed"
                    if converged
                    else "collocation reached its maximum Laurent basis without passing"
                ),
            )
        last = high
        last_order = high_order
        last_seam = seam
        last_symmetry = symmetry
        high_order += 4


def evaluate_multiprecision_holomorphic_period_map(
    topology: Topology,
    q_values: Sequence[complex],
    *,
    config: HybridPeriodMapConfig,
    log_q_values: Sequence[complex] | None = None,
) -> MethodEvaluation:
    """Evaluate the rescaled holomorphic-form backend without an import cycle."""

    q, logs = _as_q_and_logs(q_values, log_q_values)
    try:
        from genus2_multiprecision_collocation import (
            BACKEND_READY,
            evaluate_multiprecision_holomorphic_period_map as evaluate,
        )
    except ImportError:  # pragma: no cover - package-style execution
        from genus2_multiprecision_collocation import (
            BACKEND_READY,
            evaluate_multiprecision_holomorphic_period_map as evaluate,
        )
    if not BACKEND_READY:
        raise HybridPeriodMapFailure(
            "multiprecision rescaled holomorphic-form backend is unavailable"
        )
    result = evaluate(
        topology,
        q,
        log_q_values=logs,
        tolerance=float(config.tolerance),
        maximum_basis=int(config.maximum_collocation_basis),
    )
    if not isinstance(result, MethodEvaluation):
        # A package-style caller can load the same source as both
        # ``genus2_hybrid_period_map`` and ``plumbing.genus2_hybrid_period_map``.
        # Normalize the otherwise identical dataclass rather than rejecting a
        # valid backend result on Python class identity alone.
        required = (
            "algorithm",
            "omega",
            "converged",
            "error_estimate",
            "low_order",
            "high_order",
            "seam_residual",
            "symmetry_error",
            "used_multiprecision",
            "calibrated",
            "message",
        )
        if not all(hasattr(result, name) for name in required):
            raise TypeError("multiprecision holomorphic backend returned the wrong type")
        result = MethodEvaluation(
            **{
                name: getattr(result, name)
                for name in MethodEvaluation.__dataclass_fields__
            }
        )
    if result.algorithm != MULTIPRECISION_HOLOMORPHIC_ALGORITHM:
        result = MethodEvaluation(
            **{
                name: (
                    MULTIPRECISION_HOLOMORPHIC_ALGORITHM
                    if name == "algorithm"
                    else getattr(result, name)
                )
                for name in MethodEvaluation.__dataclass_fields__
            }
        )
    return result


def _align_schottky_matrix(raw: np.ndarray) -> tuple[np.ndarray, float]:
    value = np.asarray(raw, dtype=np.complex128)
    if value.shape != (2, 2) or not np.all(np.isfinite(value)):
        raise FloatingPointError("Schottky period map returned a nonfinite 2x2 matrix")
    branch = int(round((value[0, 1] - value[1, 0]).real))
    lower = value[1, 0] + branch
    symmetry = float(abs(value[0, 1] - lower))
    off_diagonal = 0.5 * (value[0, 1] + lower)
    aligned = np.asarray(
        [[value[0, 0], off_diagonal], [off_diagonal, value[1, 1]]],
        dtype=np.complex128,
    )
    return aligned, symmetry


def _ordinary_schottky(
    topology: Topology,
    q: tuple[complex, complex, complex],
    word_length: int,
) -> np.ndarray:
    generators = (
        generators_for_theta(*q)
        if topology == "theta"
        else generators_for_glasses(*q)
    )
    return schottky_period_matrix_cross_ratio(
        generators,
        max_word_len=int(word_length),
    )


def _schottky_at_word(
    topology: Topology,
    q: tuple[complex, complex, complex],
    logs: tuple[complex, complex, complex],
    word_length: int,
    *,
    force_multiprecision: bool,
) -> tuple[np.ndarray, float, bool]:
    if not force_multiprecision:
        try:
            raw = _ordinary_schottky(topology, q, word_length)
            aligned, symmetry = _align_schottky_matrix(raw)
            return aligned, symmetry, False
        except (ArithmeticError, FloatingPointError, ValueError, np.linalg.LinAlgError):
            pass
    raw = schottky_period_matrix_cross_ratio_multiprecision(
        topology,
        q,
        max_word_len=int(word_length),
        log_q_values=logs,
    )
    aligned, symmetry = _align_schottky_matrix(raw)
    return aligned, symmetry, True


def evaluate_schottky_period_map(
    topology: Topology,
    q_values: Sequence[complex],
    *,
    config: HybridPeriodMapConfig,
    log_q_values: Sequence[complex] | None = None,
    envelope: SchottkyValidityEnvelope | None = None,
) -> MethodEvaluation:
    """Raise the Schottky word cutoff and estimate the remaining geometric tail."""

    q, logs = _as_q_and_logs(q_values, log_q_values)
    if envelope is not None:
        try:
            calibrated = calibrated_schottky_period_from_q(
                topology,
                q,
                envelope=envelope,
                tolerance=config.tolerance,
            )
        except UncertifiedSchottkyRegion:
            calibrated = None
        if calibrated is not None:
            certificate = calibrated.certificate
            return MethodEvaluation(
                algorithm=CALIBRATED_SCHOTTKY_ALGORITHM,
                omega=symmetrized_period_matrix(calibrated.omega),
                converged=True,
                error_estimate=float(certificate.error_bound),
                low_order=int(certificate.word_length - 1),
                high_order=int(certificate.word_length),
                seam_residual=math.nan,
                symmetry_error=float(calibrated.symmetry_error),
                used_multiprecision=False,
                calibrated=True,
                message=f"passed calibrated Schottky cell {certificate.cell_id}",
                validity_cell_id=certificate.cell_id,
                validity_reference_table_sha256=certificate.reference_table_sha256,
            )

    q_abs = _q_abs_from_logs(logs)
    # When every multiplier is tiny, the true higher-word corrections can be
    # far below binary64 roundoff.  Summing exponentially many numerically
    # trivial words then produces an increasing *roundoff* step and the
    # geometric-tail test incorrectly reports infinity.  Multiprecision is
    # inexpensive in this regime because convergence occurs by word 2--3.
    force_mp = bool(
        min(value.real for value in logs) < math.log(1.0e-10)
        or max(q_abs) <= 1.0e-2
    )
    previous, previous_symmetry, used_mp = _schottky_at_word(
        topology,
        q,
        logs,
        config.minimum_schottky_word - 1,
        force_multiprecision=force_mp,
    )
    previous_step: float | None = None
    for word in range(config.minimum_schottky_word, config.maximum_schottky_word + 1):
        current, symmetry, current_mp = _schottky_at_word(
            topology,
            q,
            logs,
            word,
            force_multiprecision=force_mp or used_mp,
        )
        used_mp = used_mp or current_mp
        step = period_max_residual(current, previous)
        if previous_step is None or previous_step == 0.0:
            tail = step
        else:
            ratio = step / previous_step
            tail = (
                math.inf
                if not math.isfinite(ratio) or ratio >= 1.0
                else config.schottky_tail_safety_factor * step * ratio / (1.0 - ratio)
            )
        error = max(float(tail), float(symmetry), float(previous_symmetry))
        # A single word step does not determine a tail ratio.  Require two
        # successive steps before declaring convergence, even if the first
        # movement happens to lie below the requested bar.
        converged = bool(
            previous_step is not None
            and np.all(np.isfinite(current))
            and math.isfinite(error)
            and error <= config.tolerance
        )
        if converged or word >= config.maximum_schottky_word:
            return MethodEvaluation(
                algorithm=SCHOTTKY_ALGORITHM,
                omega=current,
                converged=converged,
                error_estimate=float(error),
                low_order=int(word - 1),
                high_order=int(word),
                seam_residual=math.nan,
                symmetry_error=float(max(symmetry, previous_symmetry)),
                used_multiprecision=used_mp,
                calibrated=False,
                message=(
                    "successive Schottky words passed the geometric-tail estimate"
                    if converged
                    else "Schottky series reached its maximum word without passing"
                ),
            )
        previous = current
        previous_symmetry = symmetry
        previous_step = step
    raise AssertionError("unreachable Schottky word loop")


def classify_period_map_region(
    topology: Topology,
    q_values: Sequence[complex],
    *,
    config: HybridPeriodMapConfig,
    log_q_values: Sequence[complex] | None = None,
) -> tuple[str, PlumbingGeometry]:
    """Return the agreed all-small/mixed-cusp numerical region."""

    q, logs = _as_q_and_logs(q_values, log_q_values)
    q_abs = _q_abs_from_logs(logs)
    geometry = plumbing_geometry(topology, q, log_q_values=logs)
    if geometry.minimum_margin <= config.minimum_geometry_margin:
        return "invalid-chart-geometry", geometry
    q_min = min(q_abs)
    q_max = max(q_abs)
    schottky_limit = schottky_all_small_limit(topology, config)
    if q_max <= schottky_limit:
        if (
            config.crosscheck_overlap
            and q_min >= config.collocation_comfortable_min_q
            and geometry.minimum_margin >= config.comfortable_geometry_margin
        ):
            return "two-method-overlap", geometry
        return "schottky-all-small", geometry
    if (
        q_min < config.collocation_comfortable_min_q
        or geometry.minimum_margin < config.comfortable_geometry_margin
    ):
        return "holomorphic-mixed-cusp", geometry
    return "holomorphic-bulk", geometry


def hybrid_period_matrix(
    topology: Topology,
    q_values: Sequence[complex],
    *,
    config: HybridPeriodMapConfig | None = None,
    log_q_values: Sequence[complex] | None = None,
    schottky_envelope: SchottkyValidityEnvelope | None = None,
) -> HybridPeriodMapResult:
    """Evaluate the preferred map and enforce agreement wherever both are good."""

    policy = config or HybridPeriodMapConfig()
    q, logs = _as_q_and_logs(q_values, log_q_values)
    coordinates = validate_genus2_plumbing_coordinates(
        topology,
        q,
        log_q_values=logs,
    )
    topology = coordinates.channel  # type: ignore[assignment]
    q = coordinates.q_values
    logs = coordinates.log_q_values
    region, geometry = classify_period_map_region(
        topology,
        q,
        config=policy,
        log_q_values=logs,
    )
    if region == "invalid-chart-geometry":
        raise InvalidPlumbingGeometry(
            "the requested plumbing chart has overlapping coordinate disks; "
            f"minimum margin={geometry.minimum_margin:.3e}; choose another marking"
        )

    holomorphic: MethodEvaluation | None = None
    schottky: MethodEvaluation | None = None
    overlap_residual: float | None = None

    if region == "schottky-all-small":
        schottky = evaluate_schottky_period_map(
            topology,
            q,
            config=policy,
            log_q_values=logs,
            envelope=schottky_envelope,
        )
        selected = schottky
        reason = "all three q parameters are inside the topology-specific Schottky region"
        if not schottky.converged:
            try:
                holomorphic = evaluate_multiprecision_holomorphic_period_map(
                    topology,
                    q,
                    config=policy,
                    log_q_values=logs,
                )
            except (
                ValueError,
                ArithmeticError,
                InvalidPlumbingGeometry,
                HybridPeriodMapFailure,
                np.linalg.LinAlgError,
            ):
                holomorphic = None
            if holomorphic is not None:
                selected = min(
                    (schottky, holomorphic),
                    key=lambda item: (not item.converged, item.error_estimate),
                )
                reason = (
                    "Schottky reached its word ceiling; rescaled holomorphic "
                    "forms supplied the independent fallback"
                    if holomorphic.converged
                    else "both all-small representations reached their work ceilings"
                )
    elif region == "holomorphic-mixed-cusp":
        holomorphic = evaluate_multiprecision_holomorphic_period_map(
            topology,
            q,
            config=policy,
            log_q_values=logs,
        )
        selected = holomorphic
        reason = (
            "a mixed cusp or narrow plumbing geometry requires rescaled "
            "multiprecision holomorphic forms; Schottky is not used"
        )
    elif region == "holomorphic-bulk":
        holomorphic = evaluate_holomorphic_period_map(
            topology,
            q,
            config=policy,
            log_q_values=logs,
        )
        if holomorphic.converged:
            selected = holomorphic
            reason = "ordinary finite-q bulk uses adaptive holomorphic forms"
        else:
            promoted = evaluate_multiprecision_holomorphic_period_map(
                topology, q, config=policy, log_q_values=logs
            )
            selected = min(
                (holomorphic, promoted),
                key=lambda item: (not item.converged, item.error_estimate),
            )
            holomorphic = selected
            reason = "binary64 bulk collocation missed its bar and was promoted"
    else:
        # Only the all-small method-boundary band evaluates both methods.
        try:
            holomorphic = evaluate_holomorphic_period_map(
                topology,
                q,
                config=policy,
                log_q_values=logs,
            )
        except (ValueError, ArithmeticError, InvalidPlumbingGeometry, np.linalg.LinAlgError):
            # Near a deep separating seam, binary64 Laurent basis powers
            # can overflow even though the logarithmic Schottky map is
            # well resolved. Keep the independently gated alternate path.
            holomorphic = None
        try:
            schottky = evaluate_schottky_period_map(
                topology,
                q,
                config=policy,
                log_q_values=logs,
                envelope=schottky_envelope,
            )
        except (ValueError, ArithmeticError, np.linalg.LinAlgError):
            schottky = None
        converged = [
            item for item in (holomorphic, schottky) if item is not None and item.converged
        ]
        if holomorphic is not None and schottky is not None:
            overlap_residual = period_max_residual(holomorphic.omega, schottky.omega)
            if (
                holomorphic.converged
                and schottky.converged
                and overlap_residual > policy.agreement_tolerance
            ):
                raise HybridPeriodMapFailure(
                    "holomorphic and Schottky period matrices disagree in their overlap: "
                    f"residual={overlap_residual:.3e}, bar={policy.agreement_tolerance:.3e}"
                )
        if not converged:
            candidates = [item for item in (holomorphic, schottky) if item is not None]
            if not candidates:
                raise HybridPeriodMapFailure("neither period-map representation could be evaluated")
            selected = min(candidates, key=lambda item: item.error_estimate)
        else:
            selected = min(converged, key=lambda item: item.error_estimate)
        reason = (
            "both methods passed and agreed; selected the smaller estimated truncation error"
            if len(converged) == 2
            else "all-small boundary computation selected the only method that passed"
        )

    if policy.require_convergence and not selected.converged:
        diagnostics = ", ".join(
            f"{item.algorithm}={item.error_estimate:.3e}"
            for item in (holomorphic, schottky)
            if item is not None
        )
        raise HybridPeriodMapFailure(
            "no period-map method reached the requested numerical bar "
            f"{policy.tolerance:.3e} ({diagnostics})"
        )
    return HybridPeriodMapResult(
        topology=topology,
        q=q,
        log_q=logs,
        omega=symmetrized_period_matrix(selected.omega),
        algorithm=selected.algorithm,
        region=region,
        geometry=geometry,
        error_estimate=float(selected.error_estimate),
        overlap_residual=overlap_residual,
        agreement_tolerance=float(policy.agreement_tolerance),
        holomorphic=holomorphic,
        schottky=schottky,
        selection_reason=reason,
    )


def _q_surrogate_from_log(value: complex) -> complex:
    return cmath.exp(complex(max(float(value.real), -690.0), float(value.imag)))


def refine_multiprecision_holomorphic_inverse(
    topology: Topology,
    target_omega: np.ndarray,
    initial_q: Sequence[complex],
    *,
    initial_log_q: Sequence[complex] | None = None,
    config: HybridPeriodMapConfig | None = None,
    max_nfev: int = 24,
) -> MultiprecisionHolomorphicInverseResult:
    """Refine a mixed-cusp inverse using only rescaled holomorphic forms."""

    policy = config or HybridPeriodMapConfig()
    q_seed, log_seed = _as_q_and_logs(initial_q, initial_log_q)
    target = symmetrized_period_matrix(np.asarray(target_omega, dtype=np.complex128))
    if target.shape != (2, 2) or not np.all(np.isfinite(target)):
        raise ValueError("multiprecision inverse target must be a finite 2x2 matrix")

    x0 = np.asarray(
        [component for value in log_seed for component in (value.real, value.imag)],
        dtype=np.float64,
    )
    lower: list[float] = []
    upper: list[float] = []
    for value in log_seed:
        # A portable table row may store a cusp edge only down to its sampling
        # floor even when the marked target lies much farther down that cusp.
        # Eight log units were therefore too narrow for mixed-cusp recovery:
        # the correct handle can be O(50) log units away while the other two
        # edges remain finite.  Keep the phase local, but allow the analytic
        # singular-period update to reach the target radial scale.
        radial_window = max(64.0, 2.0 * abs(float(value.real)))
        lower.extend((float(value.real) - radial_window, float(value.imag) - math.pi))
        upper.extend(
            (
                min(-1.0e-12, float(value.real) + radial_window),
                float(value.imag) + math.pi,
            )
        )
    lower_array = np.asarray(lower, dtype=np.float64)
    upper_array = np.asarray(upper, dtype=np.float64)
    x0 = np.minimum(np.maximum(x0, lower_array), upper_array)

    def unpack(values: np.ndarray):
        logs = tuple(
            complex(float(values[index]), float(values[index + 1]))
            for index in (0, 2, 4)
        )
        q = tuple(_q_surrogate_from_log(value) for value in logs)
        return q, logs

    def evaluate(values: np.ndarray) -> tuple[MethodEvaluation, np.ndarray, float]:
        q, logs = unpack(values)
        result = evaluate_multiprecision_holomorphic_period_map(
            topology, q, config=policy, log_q_values=logs
        )
        difference = period_difference_mod_integer(result.omega, target)
        return result, difference, float(np.max(np.abs(difference)))

    # In a mixed cusp the singular part of Omega is exactly linear in log(q).
    # Use its analytic inverse as a fixed-point/Newton preconditioner.  This
    # avoids a numerical six-column Jacobian, for which every column would
    # require another expensive rescaled collocation solve.
    current_x = x0.copy()
    best_x = current_x.copy()
    best_evaluation: MethodEvaluation | None = None
    best_residual = math.inf
    evaluations = 0
    # Mixed-cusp fixed-point iterations can converge slowly even when every
    # forward map is well resolved.  Honour the caller's budget: a hidden
    # 16-step cap previously stopped recoverable inverses just above tolerance.
    maximum_evaluations = max(1, int(max_nfev))
    message = "analytic leading-period log-q refinement reached its evaluation limit"
    for _ in range(maximum_evaluations):
        try:
            current_evaluation, difference, residual_value = evaluate(current_x)
        except Exception as exc:
            message = f"multiprecision holomorphic evaluation failed: {type(exc).__name__}: {exc}"
            break
        evaluations += 1
        if residual_value < best_residual:
            best_residual = residual_value
            best_x = current_x.copy()
            best_evaluation = current_evaluation
        if current_evaluation.converged and residual_value <= policy.tolerance:
            message = (
                "transported table seed passed without refinement"
                if evaluations == 1
                else "analytic leading-period log-q refinement converged"
            )
            break

        if topology == "theta":
            updates = (
                -TWO_PI_I * (difference[0, 0] - difference[0, 1]),
                -TWO_PI_I * (difference[1, 1] - difference[0, 1]),
                -TWO_PI_I * difference[0, 1],
            )
        else:
            _, current_logs = unpack(current_x)
            bridge = _q_surrogate_from_log(current_logs[2])
            updates = (
                -TWO_PI_I * difference[0, 0],
                -TWO_PI_I * difference[1, 1],
                TWO_PI_I * difference[0, 1] / bridge,
            )
        proposed = current_x.copy()
        for edge, update in enumerate(updates):
            # The table seed is already local.  A cap prevents an anomalous
            # branch or near-zero glasses bridge from leaving that chart.
            magnitude = abs(update)
            bounded = update if magnitude <= 8.0 else update * (8.0 / magnitude)
            proposed[2 * edge] += bounded.real
            proposed[2 * edge + 1] += bounded.imag
        current_x = np.minimum(np.maximum(proposed, lower_array), upper_array)

    if best_evaluation is None:
        q_final, log_final = unpack(x0)
        final = evaluate_multiprecision_holomorphic_period_map(
            topology, q_final, config=policy, log_q_values=log_final
        )
        evaluations += 1
    else:
        q_final, log_final = unpack(best_x)
        final = best_evaluation
    difference = np.asarray(final.omega - target, dtype=np.complex128)
    branch = np.rint(difference.real).astype(int)
    branch = np.rint(0.5 * (branch + branch.T)).astype(int)
    residual_value = period_max_residual(final.omega, target)
    region, _ = classify_period_map_region(
        topology, q_final, config=policy, log_q_values=log_final
    )
    success = bool(
        final.converged
        and residual_value <= policy.tolerance
        and not is_schottky_algorithm(final.algorithm)
    )
    return MultiprecisionHolomorphicInverseResult(
        success=success,
        message=(
            message
            if success
            else f"{message}; validation residual={residual_value:.3e}, "
            f"map error={final.error_estimate:.3e}"
        ),
        nfev=evaluations,
        q=tuple(complex(value) for value in q_final),  # type: ignore[arg-type]
        log_q=tuple(complex(value) for value in log_final),  # type: ignore[arg-type]
        omega=np.asarray(final.omega, dtype=np.complex128),
        branch=np.asarray(branch, dtype=int),
        residual=float(residual_value),
        stability=float(final.error_estimate),
        seam_residual=float(final.seam_residual),
        symmetry_error=float(final.symmetry_error),
        low_order=int(final.low_order),
        high_order=int(final.high_order),
        region=region,
    )


def refine_schottky_inverse(
    topology: Topology,
    target_omega: np.ndarray,
    initial_q: Sequence[complex],
    *,
    initial_log_q: Sequence[complex] | None = None,
    config: HybridPeriodMapConfig | None = None,
    fit_word_length: int = 5,
    max_nfev: int = 80,
    q3_component_bound: float = 0.98,
) -> AdaptiveSchottkyInverseResult:
    """Refine a cusp inverse at fixed word order, then certify it adaptively.

    Theta edges are optimized in logarithmic ``tau`` variables.  The glasses
    handles use the same variables, while its separating bridge is optimized
    directly unless it is already too small to affect ``Omega`` at the
    requested tolerance; in that case its supplied logarithm is preserved.
    """

    from scipy.optimize import least_squares

    policy = config or HybridPeriodMapConfig()
    q_seed, log_seed = _as_q_and_logs(initial_q, initial_log_q)
    if max(_q_abs_from_logs(log_seed)) > schottky_all_small_limit(topology, policy):
        raise HybridPeriodMapFailure(
            "Schottky inverse is forbidden outside the all-small-q region"
        )
    target = symmetrized_period_matrix(np.asarray(target_omega, dtype=np.complex128))
    if target.shape != (2, 2) or not np.all(np.isfinite(target)):
        raise ValueError("Schottky inverse target must be a finite 2x2 matrix")
    fit_word = max(int(fit_word_length), policy.minimum_schottky_word)

    def theta_unpack(values: np.ndarray):
        taus = tuple(
            complex(float(values[index]), float(values[index + 1]))
            for index in (0, 2, 4)
        )
        logs = tuple(2.0j * math.pi * value for value in taus)
        q = tuple(_q_surrogate_from_log(value) for value in logs)
        return q, logs

    frozen_bridge = bool(
        topology == "glasses"
        and log_seed[2].real < math.log(max(policy.tolerance * 0.01, 1.0e-300))
    )
    if topology == "theta":
        tau_seed = tuple(value / (2.0j * math.pi) for value in log_seed)
        x0 = np.asarray(
            [component for value in tau_seed for component in (value.real, value.imag)],
            dtype=float,
        )
        upper_imag = max(110.0, float(max(x0[1], x0[3], x0[5])) + 10.0)
        lower = np.asarray([-np.inf, 1.0e-12] * 3, dtype=float)
        upper = np.asarray([np.inf, upper_imag] * 3, dtype=float)
        unpack = theta_unpack
    elif topology == "glasses":
        tau1 = log_seed[0] / (2.0j * math.pi)
        tau2 = log_seed[1] / (2.0j * math.pi)
        # Deep nonseparating handles can have Im(tau) far above 110 while
        # their logarithmic plumbing coordinate remains perfectly usable.
        # Preserve the supplied cusp depth instead of clipping the inverse
        # seed to the historical bulk-oriented ceiling.
        upper_imag = max(110.0, float(tau1.imag), float(tau2.imag)) + 10.0
        if frozen_bridge:
            x0 = np.asarray([tau1.real, tau1.imag, tau2.real, tau2.imag], dtype=float)
            lower = np.asarray([-np.inf, 1.0e-12, -np.inf, 1.0e-12], dtype=float)
            upper = np.asarray(
                [np.inf, upper_imag, np.inf, upper_imag], dtype=float
            )

            def unpack(values: np.ndarray):
                taus = (
                    complex(float(values[0]), float(values[1])),
                    complex(float(values[2]), float(values[3])),
                )
                logs = (2.0j * math.pi * taus[0], 2.0j * math.pi * taus[1], log_seed[2])
                q = (
                    _q_surrogate_from_log(logs[0]),
                    _q_surrogate_from_log(logs[1]),
                    _q_surrogate_from_log(log_seed[2]),
                )
                return q, logs

        else:
            bridge = q_seed[2]
            bound = float(q3_component_bound)
            x0 = np.asarray(
                [tau1.real, tau1.imag, tau2.real, tau2.imag, bridge.real, bridge.imag],
                dtype=float,
            )
            lower = np.asarray(
                [-np.inf, 1.0e-12, -np.inf, 1.0e-12, -bound, -bound],
                dtype=float,
            )
            upper = np.asarray(
                [np.inf, upper_imag, np.inf, upper_imag, bound, bound],
                dtype=float,
            )
            x0 = np.minimum(np.maximum(x0, lower), upper)

            def unpack(values: np.ndarray):
                taus = (
                    complex(float(values[0]), float(values[1])),
                    complex(float(values[2]), float(values[3])),
                )
                bridge_value = complex(float(values[4]), float(values[5]))
                if bridge_value == 0.0j:
                    bridge_value = complex(1.0e-300, 0.0)
                logs = (2.0j * math.pi * taus[0], 2.0j * math.pi * taus[1], cmath.log(bridge_value))
                q = (
                    _q_surrogate_from_log(logs[0]),
                    _q_surrogate_from_log(logs[1]),
                    bridge_value,
                )
                return q, logs

    else:
        raise ValueError(f"unknown topology {topology!r}")

    def residual(values: np.ndarray) -> np.ndarray:
        q, logs = unpack(values)
        if any(value.real >= 0.0 for value in logs):
            return 1.0e6 * np.ones(6, dtype=float)
        try:
            # The box constraints below only enforce |q_e|<1.  For the
            # glasses chart that is weaker than disjoint plumbing disks, so
            # an unconstrained least-squares step can leave the very chart
            # whose inverse is being certified.  Reject such trial points in
            # the objective; otherwise the optimizer may find a formally
            # small period residual and fail only in final validation.
            if max(_q_abs_from_logs(logs)) > schottky_all_small_limit(
                topology, policy
            ):
                return 1.0e6 * np.ones(6, dtype=float)
            geometry = plumbing_geometry(topology, q, log_q_values=logs)
            if geometry.minimum_margin <= policy.minimum_geometry_margin:
                return 1.0e6 * np.ones(6, dtype=float)
            forward, _, _ = _schottky_at_word(
                topology,
                q,
                logs,
                fit_word,
                # Match the forward certificate's precision policy. Fitting
                # noisy binary64 cross ratios and then validating them in
                # multiprecision can leave a false residual floor.
                force_multiprecision=bool(
                    min(value.real for value in logs) < math.log(1.0e-10)
                    or max(_q_abs_from_logs(logs)) <= 1.0e-2
                ),
            )
        except Exception:
            return 1.0e6 * np.ones(6, dtype=float)
        # Schottky logarithms can jump by an integral B-period under an
        # arbitrarily small phase perturbation.  Remove that branch at every
        # optimizer evaluation instead of freezing a discontinuous residual.
        return genus2_symmetric_period_vector(
            period_difference_mod_integer(forward, target)
        )

    optimum = least_squares(
        residual,
        x0,
        bounds=(lower, upper),
        max_nfev=int(max_nfev),
        xtol=1.0e-10,
        ftol=1.0e-10,
        # A fixed 1e-10 gradient bar can stop before a deep separating
        # point meets its rho-scaled period tolerance. Certification below
        # remains the acceptance criterion; this only permits refinement.
        gtol=max(10 * np.finfo(float).eps, min(1.0e-10, policy.tolerance * 0.01)),
    )
    q_final, log_final = unpack(optimum.x)
    certificate = hybrid_period_matrix(
        topology,
        q_final,
        config=policy,
        log_q_values=log_final,
    )
    final_branch = np.rint((certificate.omega - target).real).astype(int)
    final_branch = np.rint(0.5 * (final_branch + final_branch.T)).astype(int)
    residual_value = period_max_residual(certificate.omega, target)
    schottky = certificate.schottky
    if schottky is None:
        raise HybridPeriodMapFailure("cusp inverse validation did not use Schottky")
    success = bool(optimum.success and residual_value <= policy.tolerance and schottky.converged)
    return AdaptiveSchottkyInverseResult(
        success=success,
        message=(
            str(optimum.message)
            if success
            else f"{optimum.message}; validation residual={residual_value:.3e}"
        ),
        nfev=int(optimum.nfev),
        q=tuple(complex(value) for value in q_final),  # type: ignore[arg-type]
        log_q=tuple(complex(value) for value in log_final),  # type: ignore[arg-type]
        omega=np.asarray(certificate.omega, dtype=np.complex128),
        branch=np.asarray(final_branch, dtype=int),
        residual=float(residual_value),
        stability=float(
            max(schottky.error_estimate, certificate.overlap_residual or 0.0)
        ),
        symmetry_error=float(schottky.symmetry_error),
        word_length=int(fit_word),
        validation_word_length=int(schottky.high_order),
        used_multiprecision=bool(schottky.used_multiprecision),
        region=certificate.region,
        overlap_residual=certificate.overlap_residual,
    )
