#!/usr/bin/env python3
"""Canonical genus-two ``bc`` ghost density in period-matrix coordinates.

The three b-zero modes are absorbed by the coordinates
``(Omega_11, Omega_12, Omega_22)``.  This module evaluates the remaining
nonchiral scalar coefficient multiplying

    |dOmega_11 dOmega_12 dOmega_22|^2.

It is obtained by dividing the dimensionless sewing-normalized critical
26-boson Mumford measure by 26 dimensionless real-scalar partitions in the
same convention.  Xi's conversion multiplies the critical seed by
``(4 pi^2 alpha')^-26`` and each scalar by
``(4 pi^2 alpha')^-1``; these factors cancel from the standalone ghost
density.  The period-matrix dependence and sewing prefactor are therefore
unchanged.  The Bergman determinant constant remains explicit here because
it cancels from the final critical integrand.
"""

from __future__ import annotations

from c1_paths import PACKAGE_ROOT, data_root, package_path, module_path

import argparse
import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np

try:
    from conformal_frame_labels import UNIT_AREA_BERGMAN_FRAME
    from free_boson_plumbing import bergman_petersson_norm_delta2
except ImportError:  # pragma: no cover - supports package-style execution
    from conformal_frame_labels import UNIT_AREA_BERGMAN_FRAME
    from free_boson_plumbing import bergman_petersson_norm_delta2


@dataclass(frozen=True)
class CanonicalBCGhostDensity:
    """Components of the canonical genus-two ghost density."""

    omega: np.ndarray
    conformal_frame: str
    worldsheet_area: float
    det_im_omega: float
    chi10: complex
    petersson_norm_delta2: float
    mumford_prefactor: complex
    bergman_determinant_constant: float
    scalar_partition_without_target_zero_mode: float
    critical_matter_ghost_density: float
    bc_ghost_density: float
    simplified_bc_ghost_density: float
    log_bc_ghost_density: float


def _validated_period_matrix(omega: np.ndarray) -> np.ndarray:
    omega = np.asarray(omega, dtype=np.complex128)
    if omega.shape != (2, 2):
        raise ValueError("omega must be a 2x2 period matrix")
    if not np.allclose(omega, omega.T, rtol=0.0, atol=1.0e-12):
        raise ValueError("omega must be symmetric")
    if float(np.min(np.linalg.eigvalsh(np.imag(omega)))) <= 0.0:
        raise ValueError("Im(omega) must be positive definite")
    return omega


def canonical_bc_ghost_density(
    omega: np.ndarray,
    *,
    mumford_prefactor: complex = 2j * math.pi,
    bergman_determinant_constant: float = 1.0,
    theta_nmax: int | None = None,
    theta_tol: float = 1.0e-12,
    chi10_normalization: str = "product",
) -> CanonicalBCGhostDensity:
    r"""Return the genus-two ``bc`` coefficient in the unit-area Bergman frame.

    The metric is ``g_B = sum(Y^(-1)_IJ omega_I conjugate(omega_J))`` and has
    worldsheet area one.  The critical Mumford density is Weyl independent.
    Dividing it by 26 scalar path integrals evaluated in this same metric
    defines the standalone ghost determinant and its three ``b`` zero-mode
    insertions in the period-dual Beltrami basis.

    With ``Y = Im(Omega)`` and the dimensionless repository convention

    ``chi10 = product_even theta[delta](0|Omega)^2``, the critical Mumford
    coefficient is

    ``|N_Phi|^2 / (det(Y)^13 |chi10|^2)``.

    The KKK Bergman determinant formula gives one dimensionless scalar
    partition function

    ``Z_X^B = C_B^(-1/2) F(Omega)^(-1/6)``,

    where ``F = det(Y)^(5/2) |chi10|^(1/2)``.  Dividing the critical
    coefficient by ``(Z_X^B)^26`` isolates the ghost density,

    ``G_bc^B = |N_Phi|^2 C_B^13 det(Y)^(-13/6) |chi10|^(1/6)``.

    The default stores the coefficient against Moore's raw theta product.
    The string-note form is ``chi10_note = 2^-12 chi10_product`` with
    ``N_Phi = 2 pi i``; equivalently, the final raw-product integrand applies
    the multiplier ``2^24`` nonchirally.  This makes the separating limit
    equal to the product of two normalized genus-one Mumford forms.  D'Hoker--Phong
    instead have ``N_Phi = pi^-12`` for the raw product.  The resulting
    nonchiral ratio ``(2 pi)^26`` compares those determinant-line
    normalizations.  Xi's separate physical-momentum conversion is
    ``Z_X^Xi=Z_X^p/(4 pi^2 alpha')`` at genus two and cancels between the
    critical seed and 26 scalar denominators in the ghost extraction.
    ``C_B`` is deliberately not guessed because it cancels from the
    matter--ghost combination.
    """

    omega = _validated_period_matrix(omega)
    determinant_constant = float(bergman_determinant_constant)
    if not math.isfinite(determinant_constant) or determinant_constant <= 0.0:
        raise ValueError("bergman_determinant_constant must be positive and finite")
    prefactor = complex(mumford_prefactor)
    if abs(prefactor) == 0.0:
        raise ValueError("mumford_prefactor must be nonzero")

    det_im, chi10, petersson_norm = bergman_petersson_norm_delta2(
        omega,
        theta_nmax=theta_nmax,
        theta_tol=theta_tol,
        chi10_normalization=chi10_normalization,
    )
    abs_chi10 = abs(chi10)
    if abs_chi10 == 0.0:
        raise ValueError("chi10 vanishes at this period matrix")

    log_scalar_partition = -0.5 * math.log(determinant_constant) - math.log(petersson_norm) / 6.0
    scalar_partition = math.exp(log_scalar_partition)
    log_critical_density = (
        2.0 * math.log(abs(prefactor)) - 13.0 * math.log(det_im) - 2.0 * math.log(abs_chi10)
    )
    critical_density = math.exp(log_critical_density)
    log_ghost_density = log_critical_density - 26.0 * log_scalar_partition
    ghost_density = math.exp(log_ghost_density)

    simplified_log = (
        2.0 * math.log(abs(prefactor))
        + 13.0 * math.log(determinant_constant)
        - (13.0 / 6.0) * math.log(det_im)
        + (1.0 / 6.0) * math.log(abs_chi10)
    )
    simplified_density = math.exp(simplified_log)

    return CanonicalBCGhostDensity(
        omega=omega.copy(),
        conformal_frame=UNIT_AREA_BERGMAN_FRAME,
        worldsheet_area=1.0,
        det_im_omega=det_im,
        chi10=chi10,
        petersson_norm_delta2=petersson_norm,
        mumford_prefactor=prefactor,
        bergman_determinant_constant=determinant_constant,
        scalar_partition_without_target_zero_mode=scalar_partition,
        critical_matter_ghost_density=critical_density,
        bc_ghost_density=ghost_density,
        simplified_bc_ghost_density=simplified_density,
        log_bc_ghost_density=log_ghost_density,
    )


def period_volume_jacobian_abs_squared(modular_denominator_determinant: complex) -> float:
    r"""Return the Jacobian of ``|dOmega_11 dOmega_12 dOmega_22|^2``.

    For ``Omega'=(A Omega+B)(C Omega+D)^(-1)``, pass
    ``det(C Omega+D)``.  The holomorphic period volume has weight ``-3`` at
    genus two, so its absolute square has weight ``-6``.
    """

    determinant = complex(modular_denominator_determinant)
    if abs(determinant) == 0.0:
        raise ValueError("modular denominator determinant must be nonzero")
    return float(abs(determinant) ** -6.0)


def _parse_omega(values: Iterable[str]) -> np.ndarray:
    parsed = tuple(complex(value) for value in values)
    if len(parsed) != 3:
        raise ValueError("expected Omega_11 Omega_12 Omega_22")
    return np.asarray([[parsed[0], parsed[1]], [parsed[1], parsed[2]]], dtype=np.complex128)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the canonical genus-two bc ghost density.")
    parser.add_argument(
        "omega",
        nargs=3,
        metavar=("OMEGA11", "OMEGA12", "OMEGA22"),
        help="complex period-matrix entries",
    )
    parser.add_argument("--theta-nmax", type=int, default=None)
    parser.add_argument("--bergman-constant", type=float, default=1.0)
    args = parser.parse_args()

    result = canonical_bc_ghost_density(
        _parse_omega(args.omega),
        theta_nmax=args.theta_nmax,
        bergman_determinant_constant=args.bergman_constant,
    )
    print("Canonical genus-two bc ghost density")
    print(f"  det Im(Omega)        = {result.det_im_omega:.16e}")
    print(f"  |chi10|              = {abs(result.chi10):.16e}")
    print(f"  Bergman F            = {result.petersson_norm_delta2:.16e}")
    print(f"  Z_X^B                = {result.scalar_partition_without_target_zero_mode:.16e}")
    print(f"  critical density     = {result.critical_matter_ghost_density:.16e}")
    print(f"  bc ghost density     = {result.bc_ghost_density:.16e}")
    print(f"  simplified density   = {result.simplified_bc_ghost_density:.16e}")


if __name__ == "__main__":
    main()
