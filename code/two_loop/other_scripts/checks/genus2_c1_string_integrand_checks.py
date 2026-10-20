#!/usr/bin/env python3
"""Checks for the explicit genus-two ``c=1`` string integrand."""

from __future__ import annotations

from c1_paths import PACKAGE_ROOT, data_root, package_path, module_path

import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np

try:
    from conformal_frame_labels import (
        GLASSES_PLUMBING_FRAME,
        THETA_PLUMBING_FRAME,
        UNIT_AREA_BERGMAN_FRAME,
    )
    from free_boson_pair_of_pants import (
        glasses_heisenberg_plumbing_partition,
        theta_heisenberg_plumbing_partition,
    )
    from free_boson_plumbing import (
        bergman_scalar_partition_candidate,
        noncompact_scalar_loop_momentum_factor,
        noncompact_scalar_zero_mode_factor,
        xi_noncompact_scalar_loop_momentum_factor,
    )
    from genus2_bc_ghost import canonical_bc_ghost_density
    from genus2_c1_string_integrand import (
        DIMENSIONLESS_SEWING_SCALAR_NORMALIZATION,
        SameFrameMatterPartitions,
        XI_PHYSICAL_MOMENTUM_SCALAR_NORMALIZATION,
        compact_boson_winding_evaluation_genus2,
        compact_boson_winding_sum_genus2,
        genus2_c1_string_integrand_density,
        genus2_c1_string_integrand_density_from_geometry,
        genus2_c1_string_integrand_geometry,
        plumbing_matter_to_bergman,
    )
    from genus2_integrand_normalization import (
        RAW_PRODUCT_FACTORIZATION_NORMALIZATION,
        CURRENT_INTEGRATION_KERNEL_CONVENTION,
        c1_bry_genus2_kernel_multiplier,
        xi_compact_target_zero_mode,
        xi_full_replacement_over_dimensionless,
        xi_genus2_scalar_over_dimensionless,
    )
except ImportError:  # pragma: no cover - supports package-style execution
    from conformal_frame_labels import (
        GLASSES_PLUMBING_FRAME,
        THETA_PLUMBING_FRAME,
        UNIT_AREA_BERGMAN_FRAME,
    )
    from free_boson_pair_of_pants import (
        glasses_heisenberg_plumbing_partition,
        theta_heisenberg_plumbing_partition,
    )
    from free_boson_plumbing import (
        bergman_scalar_partition_candidate,
        noncompact_scalar_loop_momentum_factor,
        noncompact_scalar_zero_mode_factor,
        xi_noncompact_scalar_loop_momentum_factor,
    )
    from genus2_bc_ghost import canonical_bc_ghost_density
    from genus2_c1_string_integrand import (
        DIMENSIONLESS_SEWING_SCALAR_NORMALIZATION,
        SameFrameMatterPartitions,
        XI_PHYSICAL_MOMENTUM_SCALAR_NORMALIZATION,
        compact_boson_winding_evaluation_genus2,
        compact_boson_winding_sum_genus2,
        genus2_c1_string_integrand_density,
        genus2_c1_string_integrand_density_from_geometry,
        genus2_c1_string_integrand_geometry,
        plumbing_matter_to_bergman,
    )
    from genus2_integrand_normalization import (
        RAW_PRODUCT_FACTORIZATION_NORMALIZATION,
        CURRENT_INTEGRATION_KERNEL_CONVENTION,
        c1_bry_genus2_kernel_multiplier,
        xi_compact_target_zero_mode,
        xi_full_replacement_over_dimensionless,
        xi_genus2_scalar_over_dimensionless,
    )


OMEGA = np.asarray(
    [[0.12 + 1.15j, 0.08 + 0.04j], [0.08 + 0.04j, -0.09 + 0.92j]],
    dtype=np.complex128,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _relative_error(value: float, target: float) -> float:
    return abs(float(value) - float(target)) / max(abs(float(target)), 1.0e-300)


def _period_matrix(data: dict[str, object], key: str) -> np.ndarray:
    rows = data[key]
    return np.asarray([[complex(value) for value in row] for row in rows], dtype=np.complex128)


def check_weyl_factor_cancellation() -> None:
    canonical_scalar = 0.73
    canonical_liouville = 1.17
    frame_factors = (0.0416214, 1.37)
    canonical_matter = SameFrameMatterPartitions(
        conformal_frame=UNIT_AREA_BERGMAN_FRAME,
        liouville_partition=canonical_liouville,
        noncompact_scalar_partition=canonical_scalar,
    )
    canonical = genus2_c1_string_integrand_density(
        OMEGA,
        1.2,
        matter_partitions=canonical_matter,
        lattice_nmax=7,
        theta_nmax=10,
    )

    print("algebraic plumbing-frame cancellation")
    for index, frame_factor in enumerate(frame_factors):
        plumbing_scalar = frame_factor * canonical_scalar
        plumbing_liouville = frame_factor**25 * canonical_liouville
        plumbing_matter = SameFrameMatterPartitions(
            conformal_frame=THETA_PLUMBING_FRAME if index == 0 else GLASSES_PLUMBING_FRAME,
            liouville_partition=plumbing_liouville,
            noncompact_scalar_partition=plumbing_scalar,
        )
        conversion = plumbing_matter_to_bergman(
            plumbing_matter,
            bergman_scalar_partition=canonical_scalar,
        )
        plumbing = genus2_c1_string_integrand_density(
            OMEGA,
            1.2,
            matter_partitions=plumbing_matter,
            lattice_nmax=7,
            theta_nmax=10,
        )
        reconstructed = genus2_c1_string_integrand_density(
            OMEGA,
            1.2,
            matter_partitions=conversion.bergman_matter,
            lattice_nmax=7,
            theta_nmax=10,
        )
        error = _relative_error(plumbing.density, canonical.density)
        print(f"  W={frame_factor:.7g}: relative density error = {error:.6e}")
        _require(error < 5.0e-14, "the scalar Weyl factor did not cancel")
        _require(
            _relative_error(reconstructed.density, plumbing.density) < 5.0e-14,
            "explicit Bergman reconstruction disagrees with the plumbing quotient",
        )
        _require(conversion.quotient_relative_error < 5.0e-14, "matter quotient changed frames")

    try:
        SameFrameMatterPartitions(
            conformal_frame="unspecified",  # type: ignore[arg-type]
            liouville_partition=1.0,
            noncompact_scalar_partition=1.0,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("an unspecified matter frame was accepted")


def check_shared_integrand_geometry() -> None:
    """Low/high matter quotients must reuse one unchanged geometry object."""

    matter_low = SameFrameMatterPartitions(
        conformal_frame=THETA_PLUMBING_FRAME,
        liouville_partition=1.17,
        noncompact_scalar_partition=0.73,
    )
    matter_high = SameFrameMatterPartitions(
        conformal_frame=THETA_PLUMBING_FRAME,
        liouville_partition=1.19,
        noncompact_scalar_partition=0.731,
    )
    geometry = genus2_c1_string_integrand_geometry(
        OMEGA,
        1.2,
        lattice_nmax=7,
        theta_nmax=10,
    )
    low = genus2_c1_string_integrand_density_from_geometry(
        geometry,
        matter_partitions=matter_low,
    )
    high = genus2_c1_string_integrand_density_from_geometry(
        geometry,
        matter_partitions=matter_high,
    )
    direct = genus2_c1_string_integrand_density(
        OMEGA,
        1.2,
        matter_partitions=matter_low,
        lattice_nmax=7,
        theta_nmax=10,
    )
    _require(low.log_density == direct.log_density, "shared geometry changed the density")
    _require(low.chi10 == high.chi10, "low/high matter changed the shared chi10")
    _require(
        low.compact_winding_sum == high.compact_winding_sum,
        "low/high matter changed the shared compact lattice sum",
    )
    expected_log_ratio = (
        math.log(matter_high.liouville_partition / matter_low.liouville_partition)
        - 25.0
        * math.log(
            matter_high.noncompact_scalar_partition
            / matter_low.noncompact_scalar_partition
        )
    )
    _require(
        abs((high.log_density - low.log_density) - expected_log_ratio) < 2.0e-14,
        "shared geometry did not isolate the low/high matter quotient",
    )


def check_xi_free_scalar_normalization() -> None:
    r"""Check Xi's ``dk/(2 pi)`` measure and the correlated replacement."""

    alpha_prime = 1.7
    radius = 1.2
    scalar_dimensionless = 0.73
    liouville = 1.17
    scalar_xi = (
        scalar_dimensionless
        * xi_genus2_scalar_over_dimensionless(alpha_prime)
    )

    loop_dimensionless = noncompact_scalar_loop_momentum_factor(OMEGA)
    loop_xi = xi_noncompact_scalar_loop_momentum_factor(
        OMEGA,
        alpha_prime=alpha_prime,
    )
    _require(
        _relative_error(
            loop_xi,
            loop_dimensionless / (4.0 * math.pi**2 * alpha_prime),
        )
        < 2.0e-15,
        "Xi's genus-two scalar loop measure has the wrong normalization",
    )

    dimensionless_input = genus2_c1_string_integrand_density(
        OMEGA,
        radius,
        alpha_prime=alpha_prime,
        matter_partitions=SameFrameMatterPartitions(
            conformal_frame=UNIT_AREA_BERGMAN_FRAME,
            liouville_partition=liouville,
            noncompact_scalar_partition=scalar_dimensionless,
            noncompact_scalar_normalization=(
                DIMENSIONLESS_SEWING_SCALAR_NORMALIZATION
            ),
        ),
        lattice_nmax=7,
        theta_nmax=10,
    )
    xi_input = genus2_c1_string_integrand_density(
        OMEGA,
        radius,
        alpha_prime=alpha_prime,
        matter_partitions=SameFrameMatterPartitions(
            conformal_frame=UNIT_AREA_BERGMAN_FRAME,
            liouville_partition=liouville,
            noncompact_scalar_partition=scalar_xi,
            noncompact_scalar_normalization=(
                XI_PHYSICAL_MOMENTUM_SCALAR_NORMALIZATION
            ),
        ),
        lattice_nmax=7,
        theta_nmax=10,
    )
    _require(
        _relative_error(dimensionless_input.density, xi_input.density) < 2.0e-14,
        "dimensionless and Xi-normalized scalar inputs give different densities",
    )
    _require(
        _relative_error(
            dimensionless_input.noncompact_scalar_partition,
            scalar_xi,
        )
        < 2.0e-15,
        "the integrand did not expose Xi's scalar partition",
    )

    old_log_density = (
        2.0 * math.log(2.0 * math.pi)
        - 13.0 * math.log(dimensionless_input.det_im_omega)
        - 2.0 * dimensionless_input.chi10_log_abs
        + math.log(radius * dimensionless_input.compact_winding_sum)
        + math.log(liouville)
        - 25.0 * math.log(scalar_dimensionless)
    )
    observed_net_factor = math.exp(
        dimensionless_input.log_density - old_log_density
    )
    expected_net_factor = xi_full_replacement_over_dimensionless(alpha_prime)
    print("\nXi free-scalar zero-mode normalization")
    print(f"  Z_X^Xi / Z_X^p       = {scalar_xi / scalar_dimensionless:.16e}")
    print(f"  full Xi/old factor   = {observed_net_factor:.16e}")
    print(f"  expected             = {expected_net_factor:.16e}")
    _require(
        _relative_error(observed_net_factor, expected_net_factor) < 3.0e-14,
        "the correlated Xi scalar conversion does not reduce to 1/(2 pi sqrt(alpha'))",
    )
    _require(
        _relative_error(
            dimensionless_input.compact_target_zero_mode,
            2.0 * math.pi * math.sqrt(alpha_prime) * radius,
        )
        < 2.0e-15,
        "the Xi compact target zero mode is not 2 pi R_phys",
    )
    _require(
        _relative_error(
            dimensionless_input.string_note_kernel_multiplier,
            8.0 * math.pi**4 * math.sqrt(alpha_prime),
        )
        < 2.0e-15,
        "the final c=1 kernel did not apply the BRY unit-compensated multiplier",
    )


def check_factorization_normalization_exposure() -> None:
    matter = SameFrameMatterPartitions(
        conformal_frame=UNIT_AREA_BERGMAN_FRAME,
        liouville_partition=1.17,
        noncompact_scalar_partition=0.73,
    )
    product = genus2_c1_string_integrand_density(
        OMEGA,
        1.2,
        matter_partitions=matter,
        lattice_nmax=7,
        theta_nmax=10,
        chi10_normalization="product",
    )
    normalized_cusp = genus2_c1_string_integrand_density(
        OMEGA,
        1.2,
        matter_partitions=matter,
        lattice_nmax=7,
        theta_nmax=10,
        chi10_normalization="string_note_2^-12",
    )

    print("\nseparating Mumford-residue normalization")
    print(f"  raw product multiplier       = {product.factorization_normalization:.16e}")
    print(
        "  normalized/raw density      = "
        f"{product.factorization_normalized_density / product.density:.16e}"
    )
    print(
        "  cusp-convention agreement   = "
        f"{normalized_cusp.density / product.factorization_normalized_density:.16e}"
    )
    print(
        "  string-note kernel/local CFT= "
        f"{product.string_note_kernel_density / product.factorization_normalized_density:.16e}"
    )
    _require(
        product.factorization_normalization == RAW_PRODUCT_FACTORIZATION_NORMALIZATION,
        "the raw theta-product integrand has the wrong Mumford-residue multiplier",
    )
    _require(
        _relative_error(
            product.factorization_normalized_density,
            RAW_PRODUCT_FACTORIZATION_NORMALIZATION * product.density,
        )
        < 2.0e-14,
        "the exposed Mumford-residue-normalized density is inconsistent",
    )
    _require(
        _relative_error(normalized_cusp.density, product.factorization_normalized_density) < 2.0e-13,
        "the normalized cusp form and explicit product multiplier disagree",
    )
    _require(
        normalized_cusp.factorization_normalization == 1.0,
        "the Mumford-residue-normalized cusp form retained an extra multiplier",
    )
    _require(
        product.string_note_kernel_convention
        == normalized_cusp.string_note_kernel_convention
        == CURRENT_INTEGRATION_KERNEL_CONVENTION,
        "the string-note kernel convention is not explicit",
    )
    _require(
        _relative_error(
            product.string_note_kernel_density,
            c1_bry_genus2_kernel_multiplier()
            * product.factorization_normalized_density,
        )
        < 2.0e-14,
        "the string-note real-measure multiplier is missing from the integrand",
    )


def check_compact_sum_in_saved_markings() -> None:
    nodes = json.loads((data_root('two_loop') / 'production/parent/design/nodes.json').read_text())
    node = min((n for n in nodes if n['in_domain']), key=lambda n: n['y1']+n['y2'])
    bridge = np.exp(node['log_rho'] + 1j*node['angle'])
    omega_theta = np.array([[node['x1']+1j*node['y1'], bridge],
                            [bridge, node['x2']+1j*node['y2']]])
    # This exact symplectic handle permutation is the retained production's
    # glass marking; no retired overlap fixture is needed.
    omega_glasses = omega_theta[::-1, ::-1].copy()

    print("\ncompact winding sum in the saved symplectic markings")
    for radius in (0.8, 1.0, 1.3):
        glasses = compact_boson_winding_sum_genus2(omega_glasses, radius, lattice_nmax=12)
        theta = compact_boson_winding_sum_genus2(omega_theta, radius, lattice_nmax=12)
        error = _relative_error(theta, glasses)
        print(f"  R={radius:.1f}: theta/glasses={theta / glasses:.16e}, error={error:.6e}")
        _require(error < 2.0e-12, "compact winding sum is inconsistent between markings")


def check_genus2_t_duality_scaling() -> None:
    radius = 1.25
    direct = radius * compact_boson_winding_sum_genus2(OMEGA, radius, lattice_nmax=12)
    dual_radius = 1.0 / radius
    dual = dual_radius * compact_boson_winding_sum_genus2(OMEGA, dual_radius, lattice_nmax=12)
    actual_ratio = direct / dual
    expected_ratio = radius**-2
    error = _relative_error(actual_ratio, expected_ratio)

    print("\ngenus-two compact-boson T-duality scaling")
    print(f"  Z(R) / Z(1/R) = {actual_ratio:.16e}")
    print(f"  expected R^(-2) = {expected_ratio:.16e}")
    print(f"  relative error   = {error:.6e}")
    _require(error < 2.0e-12, "the compact zero-mode factor has the wrong genus-two T-duality weight")


def check_poisson_resummation() -> None:
    print("\nPoisson-resummed compact-boson theta function")
    for radius in (0.8, 1.0, 1.3, 2.0):
        direct = compact_boson_winding_sum_genus2(
            OMEGA,
            radius,
            lattice_nmax=14,
            algorithm="direct",
        )
        poisson = compact_boson_winding_evaluation_genus2(
            OMEGA,
            radius,
            tolerance=1.0e-14,
            algorithm="poisson",
        )
        error = _relative_error(poisson.value, direct)
        print(
            f"  R={radius:.1f}: direct/resummed error={error:.6e}, "
            f"cutoffs=({poisson.momentum_nmax},{poisson.winding_nmax})"
        )
        _require(error < 2.0e-12, "Poisson resummation disagrees with the direct theta sum")

    deep_cusp = np.asarray(
        [
            [0.13 + 4.4j, 0.08 + 0.03j],
            [0.08 + 0.03j, -0.11 + 1.411039e6j],
        ],
        dtype=np.complex128,
    )
    radius = 1.6
    direct_radius = compact_boson_winding_evaluation_genus2(
        deep_cusp,
        radius,
        tolerance=1.0e-13,
    )
    dual_radius = compact_boson_winding_evaluation_genus2(
        deep_cusp,
        1.0 / radius,
        tolerance=1.0e-13,
    )
    actual_ratio = radius * direct_radius.value / ((1.0 / radius) * dual_radius.value)
    duality_error = _relative_error(actual_ratio, radius**-2)
    print(
        "  deep cusp: "
        f"algorithm={direct_radius.algorithm}, "
        f"cutoffs=({direct_radius.momentum_nmax},{direct_radius.winding_nmax}), "
        f"T-duality error={duality_error:.6e}"
    )
    _require(
        direct_radius.algorithm == dual_radius.algorithm == "poisson",
        "the automatic evaluator did not route the deep cusp to Poisson resummation",
    )
    _require(
        max(
            direct_radius.momentum_nmax,
            direct_radius.winding_nmax,
            dual_radius.momentum_nmax,
            dual_radius.winding_nmax,
        )
        <= 6,
        "the resummed deep-cusp cutoff is unexpectedly large",
    )
    _require(duality_error < 2.0e-12, "the resummed theta sum violates T-duality")

    # This is the extreme nonseparating-cusp period matrix from production
    # sample r004-c01-p00000118.  Diagonalizing its assembled 4x4 winding
    # quadratic form is ill-conditioned and produced a platform-dependent
    # negative eigenvalue on the cluster even though Im(Omega) is positive.
    production_cusp = np.asarray(
        [
            [
                0.28335479367524385 + 1.112566439249399j,
                -0.33641934487968683 + 0.1718225172857308j,
            ],
            [
                -0.33641934487968683 + 0.1718225172857308j,
                0.29582383669912815 + 6896429848.499076j,
            ],
        ],
        dtype=np.complex128,
    )
    production_evaluation = compact_boson_winding_evaluation_genus2(
        production_cusp,
        1.0,
        tolerance=1.0e-12,
    )
    print(
        "  production cusp: "
        f"algorithm={production_evaluation.algorithm}, "
        f"cutoffs=({production_evaluation.momentum_nmax},"
        f"{production_evaluation.winding_nmax}), "
        f"value={production_evaluation.value:.16e}"
    )
    _require(
        production_evaluation.algorithm == "poisson",
        "the production cusp was not routed to Poisson resummation",
    )
    _require(
        max(
            production_evaluation.momentum_nmax,
            production_evaluation.winding_nmax,
        )
        <= 6,
        "the production-cusp Poisson cutoff is unexpectedly large",
    )
    _require(
        math.isfinite(production_evaluation.value) and production_evaluation.value > 0.0,
        "the production-cusp compact-boson theta value is not positive and finite",
    )

    # This R=15 production geometry previously evaluated phases for terms
    # whose winding weights had already underflowed to zero.  The value was
    # finite, but NumPy emitted misleading matmul overflow warnings.
    large_radius_cusp = np.asarray(
        [
            [
                0.36468237 + 1.20736049j,
                -0.29718603 + 0.357496592j,
            ],
            [
                -0.29718603 + 0.357496592j,
                0.06395721 + 12712.1738j,
            ],
        ],
        dtype=np.complex128,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        large_radius_evaluation = compact_boson_winding_evaluation_genus2(
            large_radius_cusp,
            15.0,
            tolerance=1.0e-13,
            algorithm="poisson",
        )
    _require(
        math.isfinite(large_radius_evaluation.value)
        and large_radius_evaluation.value > 0.0,
        "the R=15 deep-cusp compact-boson theta value is not positive and finite",
    )






def main() -> None:
    check_weyl_factor_cancellation()
    check_shared_integrand_geometry()
    check_xi_free_scalar_normalization()
    check_factorization_normalization_exposure()
    check_compact_sum_in_saved_markings()
    check_genus2_t_duality_scaling()
    check_poisson_resummation()
    print("\ngenus-two c=1 string integrand checks passed")


if __name__ == "__main__":
    if sys.argv[1:] == ["--compact-theta-only"]:
        check_poisson_resummation()
        print("\ncompact-boson theta checks passed")
    else:
        main()
