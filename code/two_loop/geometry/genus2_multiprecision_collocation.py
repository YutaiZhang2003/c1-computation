#!/usr/bin/env python3
"""Certified rescaled holomorphic-form solver for mixed plumbing cusps.

The raw Laurent collocation matrix becomes badly scaled when one plumbing
parameter is tiny while the others remain finite.  The period matrix has a
known logarithmic singular part, whereas

    R(q) = Omega(q) - Omega_leading(q)

is regular at the plumbing divisor.  This backend solves ``R`` on safe,
phase-preserving surrogate seams, extrapolates it to the requested tiny
coordinates, and restores the exact logarithmic part with ``mpmath``.  Two
nested extrapolation windows and two Laurent orders provide independent error
estimates.  Thus arbitrary precision is used only where it is mathematically
needed; the large regular collocation systems stay in the well-scaled
binary64 regime.
"""

from __future__ import annotations

from c1_paths import PACKAGE_ROOT, data_root, package_path, module_path

import cmath
import itertools
import math
from typing import Sequence

import mpmath as mp
import numpy as np

try:
    from genus2_hybrid_period_map import (
        HOLOMORPHIC_ALGORITHM,
        MethodEvaluation,
        _collocation_at_order,
        _collocation_initial_orders,
        period_max_residual,
        plumbing_geometry,
    )
    from genus2_period_table import leading_omega
    from plumbing_algorithms import (
        glasses_basis_index,
        solve_constrained_collocation,
        theta_boundary_pair,
    )
except ImportError:  # pragma: no cover - package-style execution
    from genus2_hybrid_period_map import (
        HOLOMORPHIC_ALGORITHM,
        MethodEvaluation,
        _collocation_at_order,
        _collocation_initial_orders,
        period_max_residual,
        plumbing_geometry,
    )
    from genus2_period_table import leading_omega
    from plumbing_algorithms import (
        glasses_basis_index,
        solve_constrained_collocation,
        theta_boundary_pair,
    )


BACKEND_READY = True
TWO_PI = 2.0 * math.pi
RESCALE_TRIGGER = 1.0e-9
EXTRAPOLATION_RADII = (1.0e-6, 1.0e-7, 1.0e-8)
SCALED_BASIS_TRIGGER = 1.0e-4
SCALED_SEAM_GATE = 1.0e-7


def _stable_lstsq(
    lhs: np.ndarray,
    rhs: np.ndarray,
    *,
    rcond: float = 1.0e-14,
) -> np.ndarray:
    """Solve a scaled complex least-squares system with a QR fallback.

    NumPy's default divide-and-conquer SVD occasionally fails to converge for
    the very anisotropic mixed-cusp matrices on the cluster LAPACK build.  A
    column-pivoted QR solve does not need singular values and is stable for the
    already logarithmically scaled systems used here.  Keep the SVD as the
    usual path and invoke QR only on an explicit failure or nonfinite result.
    """

    values = np.asarray(lhs, dtype=np.complex128)
    target = np.asarray(rhs, dtype=np.complex128)
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(target)):
        raise FloatingPointError("mixed-cusp least-squares system is nonfinite")
    coefficients: np.ndarray | None = None
    try:
        candidate = np.linalg.lstsq(values, target, rcond=float(rcond))[0]
        if np.all(np.isfinite(candidate)):
            coefficients = np.asarray(candidate, dtype=np.complex128)
    except np.linalg.LinAlgError:
        pass
    if coefficients is None:
        from scipy.linalg import lstsq as scipy_lstsq

        candidate = scipy_lstsq(
            values,
            target,
            cond=float(rcond),
            lapack_driver="gelsy",
            check_finite=True,
        )[0]
        coefficients = np.asarray(candidate, dtype=np.complex128)
    if not np.all(np.isfinite(coefficients)):
        raise FloatingPointError("mixed-cusp least-squares solution is nonfinite")
    return coefficients


def _basis_log(puncture: str, n: int, z: complex) -> complex:
    """Return a logarithm of one Laurent basis function without exponentiating."""

    if puncture == "zero":
        return -int(n) * cmath.log(z)
    if puncture == "one":
        return -int(n) * cmath.log(z - 1.0)
    if puncture == "infty":
        return 1.0j * math.pi + (int(n) - 2) * cmath.log(z)
    raise ValueError(f"unknown puncture {puncture!r}")


def _theta_scaled_system(
    q: tuple[complex, complex, complex],
    order: int,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, str, int]], np.ndarray, tuple[float, ...]]:
    """Build theta seam/period matrices in a logarithmically scaled basis."""

    samples = 4 * int(order)
    index = glasses_basis_index(int(order))
    column_count = len(index)
    radii = tuple(math.sqrt(abs(value)) for value in q)
    log_values = np.full((3 * samples, column_count), complex(-math.inf, 0.0), dtype=np.complex128)
    seam_data = (("zero", q[0], radii[0]), ("one", q[1], radii[1]), ("infty", q[2], radii[2]))
    row = 0
    sphere_columns = {
        sphere: [(column, puncture, n) for column, (basis_sphere, puncture, n) in enumerate(index) if basis_sphere == sphere]
        for sphere in (0, 1)
    }
    for puncture, sewing, radius in seam_data:
        for sample in range(samples):
            phase = 2.0 * math.pi * sample / samples
            local = radius * cmath.exp(1.0j * phase)
            if puncture == "zero":
                left = local
                right = sewing / local
                derivative = -sewing / (left * left)
            elif puncture == "one":
                left = 1.0 + local
                right = 1.0 + sewing / local
                derivative = -sewing / ((left - 1.0) * (left - 1.0))
            else:
                left = 1.0 / local
                right = local / sewing
                derivative = -1.0 / (sewing * left * left)
            for column, basis_puncture, n in sphere_columns[0]:
                log_values[row, column] = _basis_log(basis_puncture, n, left)
            log_minus_derivative = cmath.log(-derivative)
            for column, basis_puncture, n in sphere_columns[1]:
                log_values[row, column] = log_minus_derivative + _basis_log(
                    basis_puncture, n, right
                )
            row += 1

    column_logs = np.max(log_values.real, axis=0)
    log_two_pi = math.log(2.0 * math.pi)
    for column, (sphere, puncture, n) in enumerate(index):
        constrained = sphere == 0 and n == 1 and puncture in {"zero", "one"}
        if constrained:
            column_logs[column] = max(column_logs[column], log_two_pi)
    finite = np.isfinite(log_values.real)
    matrix = np.zeros_like(log_values)
    matrix[finite] = np.exp(
        log_values[finite]
        - np.broadcast_to(column_logs, log_values.shape)[finite]
    )
    periods = np.zeros((2, column_count), dtype=np.complex128)
    for column, (sphere, puncture, n) in enumerate(index):
        if sphere == 0 and puncture == "zero" and n == 1:
            periods[0, column] = 2.0j * math.pi * math.exp(-column_logs[column])
        if sphere == 0 and puncture == "one" and n == 1:
            periods[1, column] = 2.0j * math.pi * math.exp(-column_logs[column])
    return matrix, periods, index, column_logs, radii


def _glasses_scaled_system(
    q: tuple[complex, complex, complex],
    order: int,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, str, int]], np.ndarray, tuple[float, ...]]:
    """Build glasses seam/period matrices without forming huge Laurent powers."""

    samples = 6 * int(order)
    index = glasses_basis_index(int(order))
    column_count = len(index)
    row_count = 3 * samples
    first_logs = np.full(
        (row_count, column_count), complex(-math.inf, 0.0), dtype=np.complex128
    )
    second_logs = np.full_like(first_logs, complex(-math.inf, 0.0))
    radii = plumbing_geometry("glasses", q).radii
    sphere_columns = {
        sphere: [
            (column, puncture, n)
            for column, (basis_sphere, puncture, n) in enumerate(index)
            if basis_sphere == sphere
        ]
        for sphere in (0, 1)
    }

    row = 0
    for sphere, sewing, radius in ((0, q[0], radii[0]), (1, q[1], radii[1])):
        log_minus_inverse_q = cmath.log(-1.0 / sewing)
        for sample in range(samples):
            phase = 2.0 * math.pi * sample / samples
            left = radius * cmath.exp(1.0j * phase)
            right = left / sewing
            for column, puncture, n in sphere_columns[sphere]:
                first_logs[row, column] = _basis_log(puncture, n, left)
                second_logs[row, column] = log_minus_inverse_q + _basis_log(
                    puncture, n, right
                )
            row += 1

    for sample in range(samples):
        phase = 2.0 * math.pi * sample / samples
        local = radii[2] * cmath.exp(1.0j * phase)
        left = 1.0 + local
        right = 1.0 + q[2] / local
        derivative = -q[2] / (left - 1.0) ** 2
        for column, puncture, n in sphere_columns[0]:
            first_logs[row, column] = _basis_log(puncture, n, left)
        log_minus_derivative = cmath.log(-derivative)
        for column, puncture, n in sphere_columns[1]:
            first_logs[row, column] = log_minus_derivative + _basis_log(
                puncture, n, right
            )
        row += 1

    column_logs = np.maximum(
        np.max(first_logs.real, axis=0), np.max(second_logs.real, axis=0)
    )
    log_two_pi = math.log(2.0 * math.pi)
    for column, (_sphere, puncture, n) in enumerate(index):
        if puncture == "zero" and n == 1:
            column_logs[column] = max(column_logs[column], log_two_pi)

    broadcast_logs = np.broadcast_to(column_logs, first_logs.shape)
    matrix = np.zeros_like(first_logs)
    for values in (first_logs, second_logs):
        finite = np.isfinite(values.real)
        matrix[finite] += np.exp(values[finite] - broadcast_logs[finite])

    periods = np.zeros((2, column_count), dtype=np.complex128)
    for column, (sphere, puncture, n) in enumerate(index):
        if puncture == "zero" and n == 1:
            periods[sphere, column] = 2.0j * math.pi * math.exp(-column_logs[column])
    return matrix, periods, index, column_logs, radii


def _scaled_exponential(value: complex, subtract_log: float) -> complex:
    shifted = complex(value.real - float(subtract_log), value.imag)
    if shifted.real < -745.0:
        return 0.0j
    if shifted.real > 700.0:
        with mp.workdps(100):
            out = mp.exp(_mp_complex(value) - mp.mpf(repr(float(subtract_log))))
            return complex(float(mp.re(out)), float(mp.im(out)))
    return cmath.exp(shifted)


def _scaled_basis_integral(
    puncture: str,
    n: int,
    z0: complex,
    z1: complex,
    column_log: float,
) -> complex:
    if puncture == "zero":
        if n == 1:
            return cmath.log(z1 / z0) * math.exp(-column_log)
        power = 1 - int(n)
        return (
            _scaled_exponential(power * cmath.log(z1), column_log)
            - _scaled_exponential(power * cmath.log(z0), column_log)
        ) / power
    if puncture == "one":
        if n == 1:
            return cmath.log((z1 - 1.0) / (z0 - 1.0)) * math.exp(-column_log)
        power = 1 - int(n)
        return (
            _scaled_exponential(power * cmath.log(z1 - 1.0), column_log)
            - _scaled_exponential(power * cmath.log(z0 - 1.0), column_log)
        ) / power
    if puncture == "infty":
        power = int(n) - 1
        return -(
            _scaled_exponential(power * cmath.log(z1), column_log)
            - _scaled_exponential(power * cmath.log(z0), column_log)
        ) / power
    raise ValueError(f"unknown puncture {puncture!r}")


def _theta_scaled_b_periods(
    forms: np.ndarray,
    index: list[tuple[int, str, int]],
    column_logs: np.ndarray,
    q: tuple[complex, complex, complex],
    radii: tuple[float, ...],
) -> np.ndarray:
    phases = (0.37, 2.11, 1.23)
    zero = theta_boundary_pair("zero", q[0], radii[0], phases[0])
    one = theta_boundary_pair("one", q[1], radii[1], phases[1])
    infinity = theta_boundary_pair("infty", q[2], radii[2], phases[2])
    omega = np.zeros((2, 2), dtype=np.complex128)
    for cycle, target in enumerate((zero, one)):
        integral = np.zeros(len(index), dtype=np.complex128)
        for column, (sphere, puncture, n) in enumerate(index):
            if sphere == 0:
                z0, z1 = infinity[0], target[0]
            else:
                z0, z1 = target[1], infinity[1]
            integral[column] = _scaled_basis_integral(
                puncture, n, z0, z1, float(column_logs[column])
            )
        omega[:, cycle] = integral @ forms
    return omega


def _scaled_one_pole_log_spiral(
    z_outer: complex,
    q: complex,
    column_log: float,
    *,
    order: int = 800,
) -> complex:
    """Integrate ``dz/(z-1)`` along a handle spiral in the scaled basis."""

    log_q = cmath.log(q)
    nodes, weights = np.polynomial.legendre.leggauss(int(order))
    terms: list[complex] = []
    scale = math.exp(-float(column_log))
    for node, weight in zip(nodes, weights):
        parameter = 0.5 * (float(node) + 1.0)
        z = z_outer * cmath.exp(parameter * log_q)
        terms.append(complex(weight) * z * log_q / (z - 1.0) * 0.5 * scale)
    return complex(
        math.fsum(value.real for value in terms),
        math.fsum(value.imag for value in terms),
    )


def _glasses_scaled_b_periods(
    forms: np.ndarray,
    index: list[tuple[int, str, int]],
    column_logs: np.ndarray,
    q: tuple[complex, complex, complex],
    radii: tuple[float, ...],
) -> np.ndarray:
    omega = np.zeros((2, 2), dtype=np.complex128)
    outer_points = (
        (radii[0] / abs(q[0])) * cmath.exp(1.3j),
        (radii[1] / abs(q[1])) * cmath.exp(1.7j),
    )
    for cycle, (sewing, outer) in enumerate(zip(q[:2], outer_points)):
        inner = outer * sewing
        integrals = np.zeros(len(index), dtype=np.complex128)
        for column, (sphere, puncture, n) in enumerate(index):
            if sphere != cycle:
                continue
            if puncture == "one" and n == 1:
                integrals[column] = _scaled_one_pole_log_spiral(
                    outer, sewing, float(column_logs[column])
                )
            else:
                integrals[column] = _scaled_basis_integral(
                    puncture,
                    n,
                    outer,
                    inner,
                    float(column_logs[column]),
                )
        omega[:, cycle] = integrals @ forms
    return omega


def _theta_collocation_at_order_rescaled(
    q: tuple[complex, complex, complex],
    order: int,
) -> tuple[np.ndarray, float, float]:
    matrix, periods, index, column_logs, radii = _theta_scaled_system(q, int(order))
    # The theta A-period constraints each fix one residue coefficient, so we
    # can eliminate them exactly instead of finding a numerically fragile SVD
    # null space.  Compensated-residual iterative refinement then recovers the
    # least-squares solution lost to cancellation in the highly anisotropic
    # mixed-cusp systems.
    fixed = [
        next(
            column
            for column, (sphere, puncture, n) in enumerate(index)
            if sphere == 0 and puncture == wanted and n == 1
        )
        for wanted in ("zero", "one")
    ]
    free = np.asarray([column for column in range(len(index)) if column not in fixed], dtype=int)
    lhs = matrix[:, free]
    forms = np.zeros((len(index), 2), dtype=np.complex128)

    def compensated_product(values: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
        out = np.empty(values.shape[0], dtype=np.complex128)
        for row, row_values in enumerate(values):
            products = row_values * coefficients
            out[row] = complex(
                math.fsum(float(value.real) for value in products),
                math.fsum(float(value.imag) for value in products),
            )
        return out

    for form in range(2):
        fixed_column = fixed[form]
        fixed_value = 1.0 / periods[form, fixed_column]
        forms[fixed_column, form] = fixed_value
        rhs = -matrix[:, fixed_column] * fixed_value
        coefficients = _stable_lstsq(lhs, rhs)
        for _ in range(3):
            residual = rhs - compensated_product(lhs, coefficients)
            correction = _stable_lstsq(lhs, residual)
            coefficients += correction
            if float(np.max(np.abs(correction))) <= 1.0e-14 * max(
                float(np.max(np.abs(coefficients))), 1.0
            ):
                break
        forms[free, form] = coefficients
    omega_raw = _theta_scaled_b_periods(forms, index, column_logs, q, radii)
    upper = complex(omega_raw[0, 1])
    lower = complex(omega_raw[1, 0])
    lower_aligned = lower + int(round((upper - lower).real))
    symmetry = abs(upper - lower_aligned)
    omega = np.asarray(omega_raw, dtype=np.complex128)
    omega[0, 1] = omega[1, 0] = 0.5 * (upper + lower_aligned)
    seam = float(
        max(
            np.max(np.abs(compensated_product(matrix, forms[:, form])))
            for form in range(2)
        )
    )
    return omega, seam, float(symmetry)


def _glasses_collocation_at_order_rescaled(
    q: tuple[complex, complex, complex],
    order: int,
) -> tuple[np.ndarray, float, float]:
    matrix, periods, index, column_logs, radii = _glasses_scaled_system(q, int(order))
    fixed = [
        next(
            column
            for column, (basis_sphere, puncture, n) in enumerate(index)
            if basis_sphere == sphere and puncture == "zero" and n == 1
        )
        for sphere in (0, 1)
    ]
    free = np.asarray([column for column in range(len(index)) if column not in fixed], dtype=int)
    lhs = matrix[:, free]
    forms = np.zeros((len(index), 2), dtype=np.complex128)

    def compensated_product(values: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
        out = np.empty(values.shape[0], dtype=np.complex128)
        for row, row_values in enumerate(values):
            products = row_values * coefficients
            out[row] = complex(
                math.fsum(float(value.real) for value in products),
                math.fsum(float(value.imag) for value in products),
            )
        return out

    for form in range(2):
        fixed_column = fixed[form]
        fixed_value = 1.0 / periods[form, fixed_column]
        forms[fixed_column, form] = fixed_value
        rhs = -matrix[:, fixed_column] * fixed_value
        coefficients = _stable_lstsq(lhs, rhs)
        for _ in range(3):
            residual = rhs - compensated_product(lhs, coefficients)
            correction = _stable_lstsq(lhs, residual)
            coefficients += correction
            if float(np.max(np.abs(correction))) <= 1.0e-14 * max(
                float(np.max(np.abs(coefficients))), 1.0
            ):
                break
        forms[free, form] = coefficients

    omega_raw = _glasses_scaled_b_periods(forms, index, column_logs, q, radii)
    upper = complex(omega_raw[0, 1])
    lower = complex(omega_raw[1, 0])
    lower_aligned = lower + int(round((upper - lower).real))
    symmetry = abs(upper - lower_aligned)
    omega = np.asarray(omega_raw, dtype=np.complex128)
    omega[0, 1] = omega[1, 0] = 0.5 * (upper + lower_aligned)
    seam = float(
        max(
            np.max(np.abs(compensated_product(matrix, forms[:, form])))
            for form in range(2)
        )
    )
    return omega, seam, float(symmetry)


def _direct_scaled_adaptive(
    topology: str,
    q: tuple[complex, complex, complex],
    tolerance: float,
    maximum_basis: int,
) -> MethodEvaluation:
    """Raise a logarithmically scaled Laurent basis without raw-power overflow."""

    evaluator = (
        _theta_collocation_at_order_rescaled
        if topology == "theta"
        else _glasses_collocation_at_order_rescaled
    )
    initial_low, initial_high = _collocation_initial_orders(
        topology, max(abs(value) for value in q)
    )
    high = max(int(initial_high), 40)
    low = max(int(initial_low), high - 8)
    previous, previous_seam, previous_symmetry = evaluator(q, low)
    while True:
        current, seam, symmetry = evaluator(q, high)
        error = max(
            period_max_residual(current, previous),
            float(symmetry),
            float(previous_symmetry),
        )
        seam_residual = max(float(seam), float(previous_seam))
        converged = bool(
            np.all(np.isfinite(current))
            and math.isfinite(error)
            and error <= float(tolerance)
            and seam_residual <= SCALED_SEAM_GATE
        )
        if converged or high >= int(maximum_basis):
            return MethodEvaluation(
                algorithm=f"multiprecision-scaled-{topology}-holomorphic-collocation",
                omega=current,
                converged=converged,
                error_estimate=float(error),
                low_order=int(low),
                high_order=int(high),
                seam_residual=float(seam_residual),
                symmetry_error=float(max(symmetry, previous_symmetry)),
                used_multiprecision=True,
                calibrated=True,
                message=(
                    "logarithmically scaled successive Laurent bases passed"
                    if converged
                    else "scaled collocation reached its Laurent-basis ceiling"
                ),
            )
        previous = current
        previous_seam = seam
        previous_symmetry = symmetry
        low = high
        high = min(high + 8, int(maximum_basis))


def _validate_inputs(
    topology: str,
    q_values: Sequence[complex],
    log_q_values: Sequence[complex],
) -> tuple[tuple[complex, complex, complex], tuple[complex, complex, complex]]:
    if topology not in {"theta", "glasses"}:
        raise ValueError(f"unknown plumbing topology {topology!r}")
    q = tuple(complex(value) for value in q_values)
    logs = tuple(complex(value) for value in log_q_values)
    if len(q) != 3 or len(logs) != 3:
        raise ValueError("multiprecision collocation needs three q and log(q) values")
    if any(
        not (math.isfinite(value.real) and math.isfinite(value.imag) and value.real < 0.0)
        for value in logs
    ):
        raise ValueError("log(q) values must be finite with negative real part")
    if any(not (math.isfinite(value.real) and math.isfinite(value.imag)) for value in q):
        raise ValueError("q values must be finite")
    return q, logs  # type: ignore[return-value]


def _mp_complex(value: complex):
    number = complex(value)
    return mp.mpc(mp.mpf(repr(number.real)), mp.mpf(repr(number.imag)))


def _exact_leading_omega(
    topology: str,
    q: tuple[complex, complex, complex],
    logs: tuple[complex, complex, complex],
    *,
    dps: int,
) -> np.ndarray:
    """Evaluate the singular plumbing formula without binary64 cancellation."""

    with mp.workdps(int(dps)):
        tau = tuple(_mp_complex(value) / (2 * mp.pi * mp.j) for value in logs)
        if topology == "theta":
            values = (
                (tau[0] + tau[2], tau[2]),
                (tau[2], tau[1] + tau[2]),
            )
        else:
            bridge = _mp_complex(q[2]) / (-2 * mp.pi * mp.j)
            values = ((tau[0], bridge), (bridge, tau[1]))
        return np.asarray(
            [[complex(float(mp.re(value)), float(mp.im(value))) for value in row] for row in values],
            dtype=np.complex128,
        )


def _surrogate_q(
    q: tuple[complex, complex, complex],
    small_indices: tuple[int, ...],
    radii: tuple[float, ...],
) -> tuple[complex, complex, complex]:
    out = list(q)
    for edge, radius in zip(small_indices, radii):
        phase = cmath.phase(q[edge])
        out[edge] = float(radius) * cmath.exp(1.0j * phase)
    return tuple(out)  # type: ignore[return-value]


def _tensor_extrapolate(
    values: dict[tuple[float, ...], np.ndarray],
    nodes: tuple[tuple[float, float], ...],
    targets: tuple[float, ...],
) -> np.ndarray:
    """Multilinearly interpolate/extrapolate a matrix on a tensor grid."""

    result = np.zeros((2, 2), dtype=np.complex128)
    for radii in itertools.product(*(pair for pair in nodes)):
        weight = 1.0
        for edge, radius in enumerate(radii):
            left, right = nodes[edge]
            other = right if radius == left else left
            weight *= (targets[edge] - other) / (radius - other)
        result += float(weight) * values[tuple(float(value) for value in radii)]
    return result


def _regular_correction_grid(
    topology: str,
    q: tuple[complex, complex, complex],
    small_indices: tuple[int, ...],
    radii_by_edge: tuple[tuple[float, ...], ...],
    order: int,
) -> tuple[dict[tuple[float, ...], np.ndarray], float, float]:
    values: dict[tuple[float, ...], np.ndarray] = {}
    max_seam = 0.0
    max_symmetry = 0.0
    for radii in itertools.product(*radii_by_edge):
        surrogate = _surrogate_q(q, small_indices, tuple(float(value) for value in radii))
        use_scaled = bool(min(abs(value) for value in surrogate) < SCALED_BASIS_TRIGGER)
        try:
            if use_scaled:
                evaluator = (
                    _theta_collocation_at_order_rescaled
                    if topology == "theta"
                    else _glasses_collocation_at_order_rescaled
                )
                omega, seam, symmetry = evaluator(surrogate, int(order))
            else:
                omega, seam, symmetry = _collocation_at_order(topology, surrogate, int(order))
        except (OverflowError, FloatingPointError, np.linalg.LinAlgError):
            evaluator = (
                _theta_collocation_at_order_rescaled
                if topology == "theta"
                else _glasses_collocation_at_order_rescaled
            )
            omega, seam, symmetry = evaluator(surrogate, int(order))
        values[tuple(float(value) for value in radii)] = omega - leading_omega(topology, surrogate)
        max_seam = max(max_seam, float(seam))
        max_symmetry = max(max_symmetry, float(symmetry))
    return values, max_seam, max_symmetry


def _direct_adaptive(
    topology: str,
    q: tuple[complex, complex, complex],
    tolerance: float,
    maximum_basis: int,
) -> MethodEvaluation:
    """Fallback for a promoted finite-q row that does not need cusp rescaling."""

    low, high = _collocation_initial_orders(topology, max(abs(value) for value in q))
    high = max(int(high), 32)
    low = max(int(low), high - 8)
    previous, previous_seam, previous_symmetry = _collocation_at_order(topology, q, low)
    while True:
        current, seam, symmetry = _collocation_at_order(topology, q, high)
        error = max(
            period_max_residual(current, previous),
            float(seam),
            float(symmetry),
            float(previous_seam),
            float(previous_symmetry),
        )
        converged = bool(np.all(np.isfinite(current)) and error <= float(tolerance))
        if converged or high >= int(maximum_basis):
            return MethodEvaluation(
                algorithm=HOLOMORPHIC_ALGORITHM,
                omega=current,
                converged=converged,
                error_estimate=float(error),
                low_order=int(low),
                high_order=int(high),
                seam_residual=float(max(seam, previous_seam)),
                symmetry_error=float(max(symmetry, previous_symmetry)),
                used_multiprecision=False,
                calibrated=True,
                message=(
                    "promoted finite-q collocation passed successive-basis checks"
                    if converged
                    else "promoted finite-q collocation reached its basis ceiling"
                ),
            )
        previous = current
        previous_seam = seam
        previous_symmetry = symmetry
        low = high
        high = min(high + 8, int(maximum_basis))


def evaluate_multiprecision_holomorphic_period_map(
    topology: str,
    q: Sequence[complex],
    *,
    log_q_values: Sequence[complex],
    tolerance: float,
    maximum_basis: int,
) -> MethodEvaluation:
    """Return a certified mixed-cusp period matrix or an explicit failure."""

    q_values, logs = _validate_inputs(topology, q, log_q_values)
    geometry = plumbing_geometry(topology, q_values, log_q_values=logs)
    if not geometry.valid:
        raise ValueError("multiprecision collocation received invalid plumbing geometry")

    minimum_q = min(math.exp(value.real) for value in logs)
    small_indices = tuple(
        edge for edge, value in enumerate(logs) if value.real < math.log(RESCALE_TRIGGER)
    )
    if not small_indices:
        if minimum_q < SCALED_BASIS_TRIGGER:
            return _direct_scaled_adaptive(
                topology, q_values, tolerance, maximum_basis
            )
        try:
            return _direct_adaptive(topology, q_values, tolerance, maximum_basis)
        except (OverflowError, FloatingPointError, np.linalg.LinAlgError):
            return _direct_scaled_adaptive(
                topology, q_values, tolerance, maximum_basis
            )

    target_radii = tuple(math.exp(logs[edge].real) for edge in small_indices)
    initial_low, initial_high = _collocation_initial_orders(
        topology, max(math.exp(value.real) for value in logs)
    )
    high_order = max(int(initial_high), 32)
    low_order = max(int(initial_low), high_order - 8)
    outer_nodes = tuple((EXTRAPOLATION_RADII[0], EXTRAPOLATION_RADII[1]) for _ in small_indices)
    inner_nodes = tuple((EXTRAPOLATION_RADII[1], EXTRAPOLATION_RADII[2]) for _ in small_indices)
    all_radii = tuple(EXTRAPOLATION_RADII for _ in small_indices)
    inner_radii = tuple(EXTRAPOLATION_RADII[1:] for _ in small_indices)

    exact_leading = _exact_leading_omega(topology, q_values, logs, dps=100)
    while True:
        high_grid, high_seam, high_symmetry = _regular_correction_grid(
            topology, q_values, small_indices, all_radii, high_order
        )
        low_grid, low_seam, low_symmetry = _regular_correction_grid(
            topology, q_values, small_indices, inner_radii, low_order
        )
        outer = _tensor_extrapolate(high_grid, outer_nodes, target_radii)
        inner = _tensor_extrapolate(high_grid, inner_nodes, target_radii)
        low_inner = _tensor_extrapolate(low_grid, inner_nodes, target_radii)
        omega = exact_leading + inner
        extrapolation_error = period_max_residual(inner, outer)
        basis_error = period_max_residual(inner, low_inner)
        # The raw seam residual is expressed in local Laurent coordinates and
        # is not invariant under the cusp rescaling.  Period stability and the
        # nested extrapolation are the direct Omega-error estimators.  We keep
        # a separate conservative absolute seam gate to catch a genuinely bad
        # boundary solve without treating coordinate amplification as period
        # error.
        error = max(
            extrapolation_error,
            basis_error,
            high_symmetry,
            low_symmetry,
        )
        seam_gate = max(SCALED_SEAM_GATE, 1_000.0 * float(tolerance))
        converged = bool(
            np.all(np.isfinite(omega))
            and math.isfinite(error)
            and error <= tolerance
            and max(high_seam, low_seam) <= seam_gate
        )
        if converged or high_order >= int(maximum_basis):
            return MethodEvaluation(
                algorithm="multiprecision-rescaled-holomorphic-collocation",
                omega=0.5 * (omega + omega.T),
                converged=converged,
                error_estimate=float(error),
                low_order=int(low_order),
                high_order=int(high_order),
                seam_residual=float(max(high_seam, low_seam)),
                symmetry_error=float(max(high_symmetry, low_symmetry)),
                used_multiprecision=True,
                calibrated=True,
                message=(
                    "regular correction passed nested cusp extrapolation and Laurent-basis checks"
                    if converged
                    else "rescaled cusp collocation reached its Laurent-basis ceiling"
                ),
            )
        low_order = high_order
        high_order = min(high_order + 8, int(maximum_basis))
