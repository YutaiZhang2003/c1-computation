#!/usr/bin/env python3
"""Genus-two ``c=1`` string integrand in period-matrix coordinates.

The raw coefficient returned here multiplies

    |dOmega_11 dOmega_12 dOmega_22|^2

and uses the selected ``chi10`` normalization.  The result also exposes a
``factorization_normalized_density`` whose separating Mumford-form residue is
one and a ``string_note_kernel_density``.  The latter is the former multiplied
by the BRY coefficient ``8*pi^4*sqrt(alpha')``, including the generic
unpunctured genus-two automorphism factor 1/2, with only
``g_s^2`` stripped and ``mu^-1=2*pi*g_s``. The historical field name
``string_note_kernel_density`` is retained for saved-schema compatibility;
its convention tag is BRY v7. Intrinsic CFT partitions are unchanged.
The kernel is integrated over the existing ordinary Siegel domain with the
existing weights. Do not multiply the automorphism half into those weights
again. The inherited absolute genus-two normalization remains
uncertified; local consistency checks do not fix every ghost/sewing factor.

The
Liouville and scalar arguments may either both be canonical/Bergman-frame
partition functions or both be plumbing-frame partition functions evaluated
with the same local coordinates.  In the second case the scalar Weyl factor
cancels algebraically from ``Z_L / Z_X**25``.

The scalar plumbing input must be the full noncompact scalar partition with
its constant target-space zero mode removed: oscillator sewing times the two
loop-momentum Gaussian.  It must not be the oscillator product alone.  The
production input uses dimensionless momenta ``p=sqrt(alpha') k`` and measure
``d^2p``.  Before the critical-boson replacement is assembled, it is converted
to Xi's per-target-volume measure ``prod_I dk_I/(2 pi)``.
"""

from __future__ import annotations

from c1_paths import PACKAGE_ROOT, data_root, package_path, module_path

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np


COMPACT_THETA_IMPLEMENTATION = "auto-direct-poisson-v2-stable-cutoff"
DIMENSIONLESS_SEWING_SCALAR_NORMALIZATION = "dimensionless-p-dp"
XI_PHYSICAL_MOMENTUM_SCALAR_NORMALIZATION = "xi-k-dk-over-2pi-per-volume"
SCALAR_NORMALIZATIONS = frozenset(
    {
        DIMENSIONLESS_SEWING_SCALAR_NORMALIZATION,
        XI_PHYSICAL_MOMENTUM_SCALAR_NORMALIZATION,
    }
)

try:
    from conformal_frame_labels import (
        GLASSES_PLUMBING_FRAME,
        MATTER_CONFORMAL_FRAMES,
        THETA_PLUMBING_FRAME,
        UNIT_AREA_BERGMAN_FRAME,
    )
    from free_boson_plumbing import igusa_chi10_evaluation_genus2
    from genus2_integrand_normalization import (
        CURRENT_INTEGRATION_KERNEL_CONVENTION,
        c1_bry_genus2_kernel_multiplier,
        mumford_factorization_normalization,
        xi_compact_target_zero_mode,
        xi_genus2_scalar_over_dimensionless,
    )
except ImportError:  # pragma: no cover - supports package-style execution
    from conformal_frame_labels import (
        GLASSES_PLUMBING_FRAME,
        MATTER_CONFORMAL_FRAMES,
        THETA_PLUMBING_FRAME,
        UNIT_AREA_BERGMAN_FRAME,
    )
    from free_boson_plumbing import igusa_chi10_evaluation_genus2
    from genus2_integrand_normalization import (
        CURRENT_INTEGRATION_KERNEL_CONVENTION,
        c1_bry_genus2_kernel_multiplier,
        mumford_factorization_normalization,
        xi_compact_target_zero_mode,
        xi_genus2_scalar_over_dimensionless,
    )


@dataclass(frozen=True)
class SameFrameMatterPartitions:
    """Liouville and scalar partition functions in one declared frame."""

    conformal_frame: Literal["bergman:unit-area", "plumbing:theta", "plumbing:glasses"]
    liouville_partition: float
    noncompact_scalar_partition: float
    noncompact_scalar_normalization: str = DIMENSIONLESS_SEWING_SCALAR_NORMALIZATION

    def __post_init__(self) -> None:
        if self.conformal_frame not in MATTER_CONFORMAL_FRAMES:
            raise ValueError(f"unsupported conformal frame {self.conformal_frame!r}")
        _positive_finite(self.liouville_partition, "liouville_partition")
        _positive_finite(self.noncompact_scalar_partition, "noncompact_scalar_partition")
        if self.noncompact_scalar_normalization not in SCALAR_NORMALIZATIONS:
            raise ValueError(
                "unsupported noncompact scalar normalization "
                f"{self.noncompact_scalar_normalization!r}"
            )


@dataclass(frozen=True)
class PlumbingToBergmanMatterConversion:
    """Explicit conversion of a same-frame plumbing pair to Bergman frame."""

    plumbing_matter: SameFrameMatterPartitions
    bergman_matter: SameFrameMatterPartitions
    plumbing_over_bergman_scalar_factor: float
    plumbing_quotient: float
    bergman_quotient: float
    quotient_relative_error: float


@dataclass(frozen=True)
class Genus2C1StringIntegrand:
    """Raw, local-CFT, and string-note densities before moduli integration."""

    omega: np.ndarray
    matter_conformal_frame: str
    alpha_prime: float
    chi10_normalization: str
    radius: float
    physical_radius: float
    lattice_nmax: int
    compact_theta_algorithm: str
    compact_theta_momentum_nmax: int
    compact_theta_winding_nmax: int
    compact_winding_sum: float
    compact_lattice_factor: float
    compact_target_zero_mode: float
    det_im_omega: float
    chi10: complex
    chi10_log_abs: float
    critical_mumford_density: float
    liouville_partition: float
    input_noncompact_scalar_partition: float
    input_noncompact_scalar_normalization: str
    noncompact_scalar_partition: float
    liouville_over_scalar_25: float
    density: float
    log_density: float
    factorization_normalization: float
    factorization_normalized_density: float
    factorization_normalized_log_density: float
    string_note_kernel_convention: str
    string_note_kernel_multiplier: float
    string_note_kernel_density: float
    string_note_kernel_log_density: float


@dataclass(frozen=True)
class Genus2C1StringIntegrandGeometry:
    """Matter-independent geometry shared by low/high CFT evaluations."""

    omega: np.ndarray
    alpha_prime: float
    chi10_normalization: str
    radius: float
    physical_radius: float
    lattice_nmax: int
    compact_theta_algorithm: str
    compact_theta_momentum_nmax: int
    compact_theta_winding_nmax: int
    compact_winding_sum: float
    compact_lattice_factor: float
    compact_target_zero_mode: float
    det_im_omega: float
    chi10: complex
    chi10_log_abs: float
    critical_mumford_density: float
    critical_mumford_log_density: float
    factorization_normalization: float
    string_note_kernel_multiplier: float


def _validated_period_matrix(omega: np.ndarray) -> np.ndarray:
    omega = np.asarray(omega, dtype=np.complex128)
    if omega.shape != (2, 2):
        raise ValueError("omega must be a 2x2 period matrix")
    if not np.allclose(omega, omega.T, rtol=0.0, atol=1.0e-12):
        raise ValueError("omega must be symmetric")
    if float(np.min(np.linalg.eigvalsh(np.imag(omega)))) <= 0.0:
        raise ValueError("Im(omega) must be positive definite")
    return omega


def _safe_exp(log_value: float) -> float:
    """Exponentiate while preserving a usable log when float range is exceeded."""

    if log_value > math.log(np.finfo(np.float64).max):
        return math.inf
    if log_value < math.log(np.nextafter(0.0, 1.0)):
        return 0.0
    return math.exp(log_value)


def _positive_finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return value


def plumbing_matter_to_bergman(
    plumbing_matter: SameFrameMatterPartitions,
    *,
    bergman_scalar_partition: float,
) -> PlumbingToBergmanMatterConversion:
    r"""Convert a paired plumbing-frame ``(Z_L, Z_X)`` to Bergman frame.

    If ``W = Z_X^pl / Z_X^B``, then ``Z_L^B = Z_L^pl / W^25``.  The
    frame-independent quotient is checked explicitly before returning.
    """

    if not plumbing_matter.conformal_frame.startswith("plumbing:"):
        raise ValueError("plumbing_matter must be declared in a plumbing frame")
    bergman_scalar = _positive_finite(bergman_scalar_partition, "bergman_scalar_partition")
    plumbing_scalar = float(plumbing_matter.noncompact_scalar_partition)
    plumbing_liouville = float(plumbing_matter.liouville_partition)
    frame_factor = plumbing_scalar / bergman_scalar
    bergman_liouville = plumbing_liouville / frame_factor**25
    bergman_matter = SameFrameMatterPartitions(
        conformal_frame=UNIT_AREA_BERGMAN_FRAME,
        liouville_partition=bergman_liouville,
        noncompact_scalar_partition=bergman_scalar,
        noncompact_scalar_normalization=plumbing_matter.noncompact_scalar_normalization,
    )
    plumbing_quotient = plumbing_liouville / plumbing_scalar**25
    bergman_quotient = bergman_liouville / bergman_scalar**25
    relative_error = abs(bergman_quotient / plumbing_quotient - 1.0)
    return PlumbingToBergmanMatterConversion(
        plumbing_matter=plumbing_matter,
        bergman_matter=bergman_matter,
        plumbing_over_bergman_scalar_factor=frame_factor,
        plumbing_quotient=plumbing_quotient,
        bergman_quotient=bergman_quotient,
        quotient_relative_error=relative_error,
    )


def _winding_quadratic_matrix(omega: np.ndarray) -> np.ndarray:
    """Return the real quadratic form for ``(m + Omega n)``."""

    x = np.asarray(np.real(omega), dtype=np.float64)
    y = np.asarray(np.imag(omega), dtype=np.float64)
    y_inverse = np.linalg.inv(y)
    return np.block(
        [
            [y_inverse, y_inverse @ x],
            [x @ y_inverse, x @ y_inverse @ x + y],
        ]
    )


def _winding_quadratic_eigenvalue_lower_bound(omega: np.ndarray) -> float:
    r"""Return a stable positive lower bound for the winding quadratic form.

    Writing ``Omega = X + iY``, the four-dimensional form factors as

    ``Q = L.T diag(Y^-1, Y) L``,  ``L = [[I, X], [0, I]]``.

    Directly diagonalizing ``Q`` is unreliable at a nonseparating cusp:
    its condition number can exceed ``1/eps`` even though ``Y`` itself is
    perfectly well resolved.  The factorization instead gives the rigorous
    bound

    ``lambda_min(Q) >= min(lambda_min(Y), 1/lambda_max(Y)) sigma_min(L)^2``.

    Using a lower bound is deliberately conservative for a truncation box.
    """

    x = np.asarray(np.real(omega), dtype=np.float64)
    y = np.asarray(np.imag(omega), dtype=np.float64)
    y_eigenvalues = np.linalg.eigvalsh(y)
    y_minimum = float(y_eigenvalues[0])
    y_maximum = float(y_eigenvalues[-1])
    if not math.isfinite(y_minimum) or not math.isfinite(y_maximum) or y_minimum <= 0.0:
        raise ValueError("Im(omega) must have finite positive eigenvalues")

    identity = np.eye(2, dtype=np.float64)
    shear = np.block([[identity, x], [np.zeros((2, 2), dtype=np.float64), identity]])
    shear_minimum_singular_value = float(np.linalg.svd(shear, compute_uv=False)[-1])
    lower_bound = (
        min(y_minimum, 1.0 / y_maximum) * shear_minimum_singular_value**2
    )
    if not math.isfinite(lower_bound) or lower_bound <= 0.0:
        raise ValueError("could not resolve a positive winding-quadratic cutoff bound")
    return lower_bound


def _automatic_lattice_nmax(omega: np.ndarray, radius: float, tolerance: float) -> int:
    smallest_eigenvalue = _winding_quadratic_eigenvalue_lower_bound(omega)
    exponent = -math.log(max(min(float(tolerance), 0.1), 1.0e-16))
    estimate = math.sqrt(exponent / (math.pi * radius * radius * smallest_eigenvalue))
    return max(2, int(math.ceil(estimate)) + 1)


@dataclass(frozen=True)
class CompactBosonWindingEvaluation:
    """A compact-boson theta value together with its truncation data."""

    value: float
    algorithm: Literal["direct", "poisson"]
    momentum_nmax: int
    winding_nmax: int
    estimated_term_count: int


def _integer_vectors(nmax: int) -> np.ndarray:
    integers = np.arange(-int(nmax), int(nmax) + 1, dtype=np.float64)
    return np.stack(np.meshgrid(integers, integers, indexing="ij"), axis=-1).reshape(-1, 2)


def _automatic_poisson_nmax(
    omega: np.ndarray,
    radius: float,
    tolerance: float,
) -> tuple[int, int]:
    r"""Choose box cutoffs after Poisson resumming the ``m`` lattice.

    The resummed summand is bounded using the smallest eigenvalue of
    ``Im(omega)``.  The returned cutoffs belong respectively to the dual
    momentum ``k`` and the original winding ``n``.
    """

    y = np.asarray(np.imag(omega), dtype=np.float64)
    smallest_eigenvalue = float(np.min(np.linalg.eigvalsh(y)))
    exponent = -math.log(max(min(float(tolerance), 0.1), 1.0e-16))
    momentum_estimate = math.sqrt(
        exponent * radius * radius / (math.pi * smallest_eigenvalue)
    )
    winding_estimate = math.sqrt(
        exponent / (math.pi * radius * radius * smallest_eigenvalue)
    )
    return (
        max(2, int(math.ceil(momentum_estimate)) + 1),
        max(2, int(math.ceil(winding_estimate)) + 1),
    )


def _direct_winding_sum(
    omega: np.ndarray,
    radius: float,
    nmax: int,
) -> float:
    x = np.asarray(np.real(omega), dtype=np.float64)
    y = np.asarray(np.imag(omega), dtype=np.float64)
    y_inverse = np.linalg.inv(y)
    vectors = _integer_vectors(nmax)

    total = 0.0
    coefficient = math.pi * radius * radius
    for winding_b in vectors:
        shifted_a = vectors + x @ winding_b
        quadratic_a = np.einsum("ni,ij,nj->n", shifted_a, y_inverse, shifted_a)
        quadratic_b = float(winding_b @ y @ winding_b)
        exponent = -coefficient * np.maximum(quadratic_a + quadratic_b, 0.0)
        total += float(np.sum(np.exp(exponent)))
    return _positive_finite(total, "compact winding sum")


def _poisson_resummed_winding_sum(
    omega: np.ndarray,
    radius: float,
    momentum_nmax: int,
    winding_nmax: int,
) -> float:
    r"""Evaluate the genus-two theta sum after Poisson resumming ``m``.

    For ``Omega = X + iY`` the identity used is

    ``Theta_R = sqrt(det(Y))/R^2 sum_{k,n} exp[-pi k.Y.k/R^2
       -pi R^2 n.Y.n] exp[2 pi i k.X.n]``.

    Symmetric boxes let us replace the last phase by its cosine exactly.
    This representation is especially efficient at nonseparating cusps,
    where the direct ``m`` cutoff grows like ``sqrt(max eig(Y))``.
    """

    x = np.asarray(np.real(omega), dtype=np.float64)
    y = np.asarray(np.imag(omega), dtype=np.float64)
    determinant = float(np.linalg.det(y))
    momentum_vectors = _integer_vectors(momentum_nmax)
    winding_vectors = _integer_vectors(winding_nmax)
    momentum_quadratic = np.einsum(
        "ni,ij,nj->n", momentum_vectors, y, momentum_vectors
    )
    momentum_weight = np.exp(
        -math.pi * np.maximum(momentum_quadratic, 0.0) / (radius * radius)
    )

    total = 0.0
    # Deep cusp matrices make most nonzero winding terms underflow exactly to
    # zero.  Do not form their unused oscillatory phases: on some BLAS builds
    # the next matmul otherwise inherits misleading floating-point warnings.
    # The skipped terms are already zero in the working float64 arithmetic.
    log_smallest_subnormal = math.log(float(np.nextafter(0.0, 1.0)))
    for winding in winding_vectors:
        winding_quadratic = max(
            float(np.einsum("i,ij,j->", winding, y, winding)),
            0.0,
        )
        winding_exponent = -math.pi * radius * radius * winding_quadratic
        if winding_exponent < log_smallest_subnormal:
            continue
        winding_weight = math.exp(winding_exponent)
        x_winding = np.einsum("ij,j->i", x, winding)
        phase = 2.0 * math.pi * np.einsum(
            "ni,i->n",
            momentum_vectors,
            x_winding,
        )
        total += winding_weight * float(np.dot(momentum_weight, np.cos(phase)))

    value = math.sqrt(determinant) * total / (radius * radius)
    return _positive_finite(value, "Poisson-resummed compact winding sum")


def compact_boson_winding_evaluation_genus2(
    omega: np.ndarray,
    radius: float,
    *,
    lattice_nmax: int | None = None,
    tolerance: float = 1.0e-12,
    algorithm: Literal["auto", "direct", "poisson"] = "auto",
) -> CompactBosonWindingEvaluation:
    """Evaluate the winding theta function and expose the chosen branch."""

    omega = _validated_period_matrix(omega)
    radius = _positive_finite(radius, "radius")
    if not math.isfinite(float(tolerance)) or float(tolerance) <= 0.0:
        raise ValueError("tolerance must be positive and finite")
    if algorithm not in {"auto", "direct", "poisson"}:
        raise ValueError("algorithm must be 'auto', 'direct', or 'poisson'")
    if lattice_nmax is not None and int(lattice_nmax) < 0:
        raise ValueError("lattice_nmax must be nonnegative")
    if algorithm == "poisson" and lattice_nmax is not None:
        raise ValueError("lattice_nmax is a direct-sum cutoff and cannot select poisson")

    direct_nmax: int | None = None
    direct_cost: int | None = None
    if algorithm in {"auto", "direct"}:
        direct_nmax = (
            int(lattice_nmax)
            if lattice_nmax is not None
            else _automatic_lattice_nmax(omega, radius, tolerance)
        )
        direct_cost = (2 * direct_nmax + 1) ** 4

    momentum_nmax: int | None = None
    winding_nmax: int | None = None
    poisson_cost: int | None = None
    if algorithm in {"auto", "poisson"}:
        momentum_nmax, winding_nmax = _automatic_poisson_nmax(
            omega,
            radius,
            tolerance,
        )
        poisson_cost = (2 * momentum_nmax + 1) ** 2 * (2 * winding_nmax + 1) ** 2

    selected = algorithm
    if selected == "auto":
        assert direct_cost is not None and poisson_cost is not None
        # An explicit legacy cutoff requests the direct box exactly.  With an
        # automatic cutoff, choose the representation with fewer summands.
        selected = (
            "direct"
            if lattice_nmax is not None or direct_cost <= poisson_cost
            else "poisson"
        )

    if selected == "direct":
        assert direct_nmax is not None and direct_cost is not None
        return CompactBosonWindingEvaluation(
            value=_direct_winding_sum(omega, radius, direct_nmax),
            algorithm="direct",
            momentum_nmax=direct_nmax,
            winding_nmax=direct_nmax,
            estimated_term_count=direct_cost,
        )
    assert momentum_nmax is not None and winding_nmax is not None and poisson_cost is not None
    return CompactBosonWindingEvaluation(
        value=_poisson_resummed_winding_sum(
            omega,
            radius,
            momentum_nmax,
            winding_nmax,
        ),
        algorithm="poisson",
        momentum_nmax=momentum_nmax,
        winding_nmax=winding_nmax,
        estimated_term_count=poisson_cost,
    )


def compact_boson_winding_sum_genus2(
    omega: np.ndarray,
    radius: float,
    *,
    lattice_nmax: int | None = None,
    tolerance: float = 1.0e-12,
    algorithm: Literal["auto", "direct", "poisson"] = "auto",
) -> float:
    r"""Return the genus-two compact-boson classical winding sum.

    For ``r=R_phys/sqrt(alpha')``, this evaluates

    ``sum_{m,n in Z^2} exp[-pi r^2
       (m + conjugate(Omega)n)^T Im(Omega)^(-1) (m + Omega n)]``.

    In Xi's physical-momentum convention the compact partition is
    ``(2 pi R_phys) * Z_X,Xi * winding_sum``.  The historical dimensionless
    sewing convention instead wrote ``r * Z_X,p * winding_sum``; the complete
    integrand conversion between them is handled by
    :func:`genus2_c1_string_integrand_density`.
    """

    return compact_boson_winding_evaluation_genus2(
        omega,
        radius,
        lattice_nmax=lattice_nmax,
        tolerance=tolerance,
        algorithm=algorithm,
    ).value


def genus2_c1_string_integrand_geometry(
    omega: np.ndarray,
    radius: float,
    *,
    alpha_prime: float = 1.0,
    mumford_prefactor: complex = 2j * math.pi,
    lattice_nmax: int | None = None,
    lattice_tolerance: float = 1.0e-12,
    lattice_algorithm: Literal["auto", "direct", "poisson"] = "auto",
    theta_nmax: int | None = None,
    theta_tolerance: float = 1.0e-12,
    chi10_normalization: str = "product",
) -> Genus2C1StringIntegrandGeometry:
    """Compute all radius/period data shared by matter truncations."""

    omega = _validated_period_matrix(omega)
    radius = _positive_finite(radius, "radius")
    alpha_prime = _positive_finite(alpha_prime, "alpha_prime")
    prefactor = complex(mumford_prefactor)
    if abs(prefactor) == 0.0:
        raise ValueError("mumford_prefactor must be nonzero")

    winding_evaluation = compact_boson_winding_evaluation_genus2(
        omega,
        radius,
        lattice_nmax=lattice_nmax,
        tolerance=lattice_tolerance,
        algorithm=lattice_algorithm,
    )
    compact_target_zero_mode = xi_compact_target_zero_mode(radius, alpha_prime)
    compact_factor = compact_target_zero_mode * winding_evaluation.value
    det_im = float(np.linalg.det(np.asarray(np.imag(omega), dtype=np.float64)))
    chi10_evaluation = igusa_chi10_evaluation_genus2(
        omega,
        nmax=theta_nmax,
        tol=theta_tolerance,
        normalization=chi10_normalization,
    )
    log_critical_density = (
        2.0 * math.log(abs(prefactor))
        - 26.0 * math.log(4.0 * math.pi * math.pi * alpha_prime)
        - 13.0 * math.log(det_im)
        - 2.0 * chi10_evaluation.log_abs
    )
    return Genus2C1StringIntegrandGeometry(
        omega=omega.copy(),
        alpha_prime=alpha_prime,
        chi10_normalization=chi10_normalization,
        radius=radius,
        physical_radius=math.sqrt(alpha_prime) * radius,
        lattice_nmax=max(
            winding_evaluation.momentum_nmax,
            winding_evaluation.winding_nmax,
        ),
        compact_theta_algorithm=winding_evaluation.algorithm,
        compact_theta_momentum_nmax=winding_evaluation.momentum_nmax,
        compact_theta_winding_nmax=winding_evaluation.winding_nmax,
        compact_winding_sum=winding_evaluation.value,
        compact_lattice_factor=compact_factor,
        compact_target_zero_mode=compact_target_zero_mode,
        det_im_omega=det_im,
        chi10=chi10_evaluation.value,
        chi10_log_abs=chi10_evaluation.log_abs,
        critical_mumford_density=_safe_exp(log_critical_density),
        critical_mumford_log_density=log_critical_density,
        factorization_normalization=mumford_factorization_normalization(
            chi10_normalization
        ),
        string_note_kernel_multiplier=(
            c1_bry_genus2_kernel_multiplier(alpha_prime)
        ),
    )


def genus2_c1_string_integrand_density_from_geometry(
    geometry: Genus2C1StringIntegrandGeometry,
    *,
    matter_partitions: SameFrameMatterPartitions,
) -> Genus2C1StringIntegrand:
    """Attach one same-frame matter quotient to precomputed geometry."""

    if not isinstance(geometry, Genus2C1StringIntegrandGeometry):
        raise TypeError("geometry must be a Genus2C1StringIntegrandGeometry instance")
    if not isinstance(matter_partitions, SameFrameMatterPartitions):
        raise TypeError("matter_partitions must be a SameFrameMatterPartitions instance")
    liouville_partition = _positive_finite(
        matter_partitions.liouville_partition,
        "liouville_partition",
    )
    input_scalar_partition = _positive_finite(
        matter_partitions.noncompact_scalar_partition,
        "noncompact_scalar_partition",
    )
    if (
        matter_partitions.noncompact_scalar_normalization
        == DIMENSIONLESS_SEWING_SCALAR_NORMALIZATION
    ):
        scalar_partition = (
            input_scalar_partition
            * xi_genus2_scalar_over_dimensionless(geometry.alpha_prime)
        )
    else:
        scalar_partition = input_scalar_partition

    log_matter_quotient = (
        math.log(liouville_partition) - 25.0 * math.log(scalar_partition)
    )
    matter_quotient = _safe_exp(log_matter_quotient)
    log_density = (
        geometry.critical_mumford_log_density
        + math.log(geometry.compact_lattice_factor)
        + log_matter_quotient
    )
    density = _safe_exp(log_density)
    factorization_normalized_log_density = (
        log_density + math.log(geometry.factorization_normalization)
    )
    factorization_normalized_density = _safe_exp(
        factorization_normalized_log_density
    )
    string_note_kernel_log_density = (
        factorization_normalized_log_density
        + math.log(geometry.string_note_kernel_multiplier)
    )
    string_note_kernel_density = _safe_exp(string_note_kernel_log_density)

    return Genus2C1StringIntegrand(
        omega=geometry.omega.copy(),
        matter_conformal_frame=matter_partitions.conformal_frame,
        alpha_prime=geometry.alpha_prime,
        chi10_normalization=geometry.chi10_normalization,
        radius=geometry.radius,
        physical_radius=geometry.physical_radius,
        lattice_nmax=geometry.lattice_nmax,
        compact_theta_algorithm=geometry.compact_theta_algorithm,
        compact_theta_momentum_nmax=geometry.compact_theta_momentum_nmax,
        compact_theta_winding_nmax=geometry.compact_theta_winding_nmax,
        compact_winding_sum=geometry.compact_winding_sum,
        compact_lattice_factor=geometry.compact_lattice_factor,
        compact_target_zero_mode=geometry.compact_target_zero_mode,
        det_im_omega=geometry.det_im_omega,
        chi10=geometry.chi10,
        chi10_log_abs=geometry.chi10_log_abs,
        critical_mumford_density=geometry.critical_mumford_density,
        liouville_partition=liouville_partition,
        input_noncompact_scalar_partition=input_scalar_partition,
        input_noncompact_scalar_normalization=(
            matter_partitions.noncompact_scalar_normalization
        ),
        noncompact_scalar_partition=scalar_partition,
        liouville_over_scalar_25=matter_quotient,
        density=density,
        log_density=log_density,
        factorization_normalization=geometry.factorization_normalization,
        factorization_normalized_density=factorization_normalized_density,
        factorization_normalized_log_density=factorization_normalized_log_density,
        string_note_kernel_convention=CURRENT_INTEGRATION_KERNEL_CONVENTION,
        string_note_kernel_multiplier=geometry.string_note_kernel_multiplier,
        string_note_kernel_density=string_note_kernel_density,
        string_note_kernel_log_density=string_note_kernel_log_density,
    )


def genus2_c1_string_integrand_density(
    omega: np.ndarray,
    radius: float,
    *,
    matter_partitions: SameFrameMatterPartitions,
    alpha_prime: float = 1.0,
    mumford_prefactor: complex = 2j * math.pi,
    lattice_nmax: int | None = None,
    lattice_tolerance: float = 1.0e-12,
    lattice_algorithm: Literal["auto", "direct", "poisson"] = "auto",
    theta_nmax: int | None = None,
    theta_tolerance: float = 1.0e-12,
    chi10_normalization: str = "product",
) -> Genus2C1StringIntegrand:
    r"""Return coefficients of the genus-two period-matrix volume form.

    ``omega`` is the period matrix in the integration (normally Siegel
    fundamental-domain) frame.  The modular-covariant Mumford/Igusa
    coefficient and all theta-function evaluations, including the compact
    winding sum, are evaluated directly at this matrix and multiply the
    fundamental-domain coordinate volume. A plumbing marking is used only
    for the same-frame Liouville/free-boson quotient supplied through
    ``matter_partitions``; it must not replace ``omega`` here.

    ``radius`` is the dimensionless radius
    ``r=R_phys/sqrt(alpha')``.  ``matter_partitions`` pairs ``Z_L`` and
    ``Z_X`` under one explicit frame label.  In either one common canonical
    frame or one plumbing frame ``a``, Xi's normalization gives

    ``I_2 = |N_Phi|^2 / [(4 pi^2 alpha')^26 det(Y)^13 |chi10|^2]
             * (2 pi R_phys) Theta_r * Z_L^a / (Z_X,Xi^a)^25``.

    The default scalar input is the dimensionless sewing partition
    ``Z_X,p``.  It is converted internally according to

    ``Z_X,Xi = Z_X,p / (4 pi^2 alpha')``.

    Consequently all large powers cancel and the complete Xi-normalized
    density is ``1/(2 pi sqrt(alpha'))`` times the earlier dimensionless
    expression at fixed ``r``.  This conversion must be applied to the
    critical seed, compact zero mode, and scalar denominator together.

    For plumbing inputs this identity follows from
    ``Z_X^pl = W Z_X^B`` and ``Z_L^pl = W^25 Z_L^B``.  Thus no separately
    evaluated Weyl factor or Bergman determinant constant remains.

    ``density`` keeps the requested cusp-form convention unchanged for
    channel diagnostics.  ``factorization_normalized_density`` multiplies it
    by the exact Mumford-form separating-residue conversion (``2^24`` for the
    default raw theta product).  In the normalized scalar and Liouville state
    conventions this fixes the local CFT sewing coefficient.

    The compatibility field ``string_note_kernel_density`` is now the
    BRY v7 amplitude kernel, including the automorphism half:

    ``K_BRY = 8*pi^4*sqrt(alpha') * factorization_normalized_density``.

    Thus ``F_2/g_s^2 = integral_F K_BRY d^6Omega`` with ``g_s=g_BRY``.
    The extra sqrt(alpha') compensates the intermediate physical-momentum
    density's inverse length unit; at fixed r the final kernel is
    alpha'-independent. At alpha'=1 this is exactly one eighth of Xi v5:
    one quarter from the BRY coupling conversion, then the automorphism half.

    Saved Xi rows are converted explicitly at assembly, including absolute
    errors. CFT quotients and proposal weights are unchanged. The current
    geometric domain weights receive no further 1/2: it is already in K_BRY.
    This is a conversion of the working coefficient, not an independent
    determination of its unresolved absolute ghost/sewing normalization.
    """

    geometry = genus2_c1_string_integrand_geometry(
        omega,
        radius,
        alpha_prime=alpha_prime,
        mumford_prefactor=mumford_prefactor,
        lattice_nmax=lattice_nmax,
        lattice_tolerance=lattice_tolerance,
        lattice_algorithm=lattice_algorithm,
        theta_nmax=theta_nmax,
        theta_tolerance=theta_tolerance,
        chi10_normalization=chi10_normalization,
    )
    return genus2_c1_string_integrand_density_from_geometry(
        geometry,
        matter_partitions=matter_partitions,
    )
