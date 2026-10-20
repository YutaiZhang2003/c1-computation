"""Sampling densities, weighted moduli integration, and independent-scramble errors."""
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from five_point_integrand_and_quadrature import EqualEnergyFivePointKernel
import cmath
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import numpy as np
from scipy.stats import qmc
from five_point_liouville_correlator_and_atlas import linear_channel_complex_jacobian_to_chart, linear_channel_from_ordering, linear_channel_positions_by_label, linear_channel_to_original_chart, oriented_tree_orderings


@dataclass(frozen=True)
class EqualEnergyQMCResult:
    omega: complex
    estimates: tuple[complex, ...]
    mean: complex
    standard_error_real: float
    standard_error_imag: float
    samples_per_replicate: int
    replicates: int
    block_order: int
    momentum_order: int
    block_scheme: str

    def to_json(self) -> dict[str, object]:
        return {
            "omega_real": self.omega.real,
            "omega_imag": self.omega.imag,
            "estimate_real": self.mean.real,
            "estimate_imag": self.mean.imag,
            "standard_error_real": self.standard_error_real,
            "standard_error_imag": self.standard_error_imag,
            "replicate_estimates": [
                {"real": value.real, "imag": value.imag} for value in self.estimates
            ],
            "samples_per_replicate": self.samples_per_replicate,
            "replicates": self.replicates,
            "block_order": self.block_order,
            "momentum_order": self.momentum_order,
            "block_scheme": self.block_scheme,
        }


def _plane_map(radial_coordinate: float, angular_coordinate: float) -> tuple[complex, float]:
    """Map the unit square to the complex plane and return its area Jacobian."""

    radial_coordinate = float(radial_coordinate)
    angular_coordinate = float(angular_coordinate)
    radius = math.tan(0.5 * math.pi * radial_coordinate)
    angle = 2.0 * math.pi * angular_coordinate
    secant_squared = 1.0 + radius * radius
    jacobian = math.pi * math.pi * radius * secant_squared
    return complex(radius * cmath.exp(1.0j * angle)), jacobian


def integrate_convergent_equal_energy_qmc(
    kernel: EqualEnergyFivePointKernel,
    *,
    sobol_power: int = 7,
    replicates: int = 4,
    seed: int = 20260823,
) -> EqualEnergyQMCResult:
    """Integrate a subtraction-free complex-energy point over ``C^2``.

    This diagnostic is intentionally unavailable when any boundary has a
    power divergence; the physical calculation must go through the complete
    forest rather than silently using the raw integral.
    """

    if kernel.requires_power_subtraction():
        raise ValueError(
            "this energy requires the sphere-five-point boundary subtraction forest"
        )
    sobol_power = int(sobol_power)
    replicates = int(replicates)
    if sobol_power < 1 or replicates < 2:
        raise ValueError("sobol_power must be positive and replicates at least two")
    estimates: list[complex] = []
    for replicate in range(replicates):
        sampler = qmc.Sobol(d=4, scramble=True, seed=int(seed) + replicate)
        points = sampler.random_base2(sobol_power)
        values: list[complex] = []
        for point in points:
            z_in, jacobian_in = _plane_map(point[0], point[1])
            z_out, jacobian_out = _plane_map(point[2], point[3])
            try:
                value = kernel.integrand(z_in, z_out)
            except (ArithmeticError, RuntimeError, ValueError, ZeroDivisionError):
                # Sobol points never sit exactly on a boundary.  A failed
                # interior point is therefore a genuine numerical error.
                raise
            values.append(value * jacobian_in * jacobian_out)
        estimates.append(complex(np.mean(np.asarray(values, dtype=complex))))
    estimate_array = np.asarray(estimates, dtype=complex)
    mean = complex(np.mean(estimate_array))
    return EqualEnergyQMCResult(
        omega=kernel.omega,
        estimates=tuple(estimates),
        mean=mean,
        standard_error_real=float(
            np.std(estimate_array.real, ddof=1) / math.sqrt(replicates)
        ),
        standard_error_imag=float(
            np.std(estimate_array.imag, ddof=1) / math.sqrt(replicates)
        ),
        samples_per_replicate=2**sobol_power,
        replicates=replicates,
        block_order=kernel.block_order,
        momentum_order=kernel.momentum_order,
        block_scheme=kernel.block_scheme,
    )


def _power_disk_sample(
    radial_uniform: float,
    angular_uniform: float,
    radial_power: float,
) -> tuple[complex, float]:
    r"""Sample the unit disc with radial CDF ``r**radial_power``.

    Returns the complex point and its probability density with respect to
    plane area.
    """

    radial_power = float(radial_power)
    if not 0.0 < radial_power <= 2.0:
        raise ValueError("radial_power must lie in (0,2]")
    u = min(max(float(radial_uniform), np.nextafter(0.0, 1.0)), np.nextafter(1.0, 0.0))
    radius = u ** (1.0 / radial_power)
    angle = 2.0 * math.pi * float(angular_uniform)
    density = radial_power / (2.0 * math.pi) * radius ** (radial_power - 2.0)
    return complex(radius * cmath.exp(1.0j * angle)), float(density)


def oriented_bidisc_mixture_density(
    positions: Sequence[complex | None],
    *,
    radial_power: float,
) -> float:
    """Return the 120-chart bidisc mixture density in the original chart."""

    orderings = oriented_tree_orderings()
    total = 0.0
    for ordering in orderings:
        try:
            channel = linear_channel_from_ordering(positions, ordering)
        except (ValueError, ZeroDivisionError):
            continue
        radius1 = abs(channel.q1)
        radius2 = abs(channel.q2)
        if not (0.0 < radius1 < 1.0 and 0.0 < radius2 < 1.0):
            continue
        forward = linear_channel_to_original_chart(
            channel.q1, channel.q2, ordering
        )
        if forward.area_jacobian <= 0.0:
            continue
        density1 = float(radial_power) / (2.0 * math.pi) * radius1 ** (
            float(radial_power) - 2.0
        )
        density2 = float(radial_power) / (2.0 * math.pi) * radius2 ** (
            float(radial_power) - 2.0
        )
        total += density1 * density2 / forward.area_jacobian
    mixture = total / len(orderings)
    if not math.isfinite(mixture) or mixture <= 0.0:
        raise ArithmeticError("the oriented-bidisc mixture density is non-positive")
    return float(mixture)


def oriented_bidisc_log_mixture_density_in_frame(
    positions: Sequence[complex | None],
    selected_ordering: Sequence[int],
    *,
    radial_power: float,
) -> float:
    r"""Return the log mixture density in a selected linear-channel gauge.

    The selected ordering fixes its labels ``(a,d,e)`` at ``(0,1,inf)`` and
    integrates the labels ``(b,c)``.  Each of the 120 proposal charts is
    transformed directly into that same gauge.  A log-sum-exp keeps the
    density finite in arbitrarily deep OPE collars.
    """

    selected = tuple(int(label) for label in selected_ordering)
    if len(selected) != 5 or set(selected) != set(range(5)):
        raise ValueError("selected_ordering must permute labels 0,...,4")
    if len(positions) != 5:
        raise ValueError("positions must contain five label-ordered punctures")
    radial_power = float(radial_power)
    if not 0.0 < radial_power <= 2.0:
        raise ValueError("radial_power must lie in (0,2]")
    fixed_zero, fixed_one, fixed_infinity = selected[0], selected[3], selected[4]
    moving_labels = (selected[1], selected[2])
    base_log_density = math.log(radial_power / (2.0 * math.pi))
    logarithmic_terms: list[float] = []
    orderings = oriented_tree_orderings()
    for ordering in orderings:
        try:
            channel = linear_channel_from_ordering(positions, ordering)
            radius1 = abs(channel.q1)
            radius2 = abs(channel.q2)
            if not (0.0 < radius1 < 1.0 and 0.0 < radius2 < 1.0):
                continue
            jacobian = linear_channel_complex_jacobian_to_chart(
                channel.q1,
                channel.q2,
                ordering,
                fixed_zero=fixed_zero,
                fixed_one=fixed_one,
                fixed_infinity=fixed_infinity,
                moving_labels=moving_labels,
            )
            absolute_jacobian = abs(jacobian)
            if not math.isfinite(absolute_jacobian) or absolute_jacobian <= 0.0:
                continue
            logarithmic_terms.append(
                2.0 * base_log_density
                + (radial_power - 2.0) * (math.log(radius1) + math.log(radius2))
                - 2.0 * math.log(absolute_jacobian)
            )
        except (ArithmeticError, OverflowError, ValueError, ZeroDivisionError):
            continue
    if not logarithmic_terms:
        raise ArithmeticError("no oriented bidisc contributes to the mixture density")
    maximum = max(logarithmic_terms)
    return float(
        maximum
        + math.log(sum(math.exp(value - maximum) for value in logarithmic_terms))
        - math.log(len(orderings))
    )


def integrate_convergent_equal_energy_atlas_qmc(
    kernel: EqualEnergyFivePointKernel,
    *,
    sobol_power: int = 7,
    replicates: int = 4,
    radial_power: float = 0.08,
    seed: int = 20260823,
) -> EqualEnergyQMCResult:
    r"""Integrate a convergent point using the 120-chart plumbing mixture.

    Sampling in plumbing coordinates resolves all ten boundary divisors and
    all fifteen corners.  The mixture-density denominator removes multiple
    coverage exactly.  A small ``radial_power`` importance-samples the
    integrable but sharp OPE collars of imaginary-energy kinematics.
    """

    if kernel.requires_power_subtraction():
        raise ValueError(
            "this energy requires the sphere-five-point boundary subtraction forest"
        )
    sobol_power = int(sobol_power)
    replicates = int(replicates)
    radial_power = float(radial_power)
    if sobol_power < 1 or replicates < 2:
        raise ValueError("sobol_power must be positive and replicates at least two")
    if not 0.0 < radial_power <= 2.0:
        raise ValueError("radial_power must lie in (0,2]")
    orderings = oriented_tree_orderings()
    estimates: list[complex] = []
    for replicate in range(replicates):
        sampler = qmc.Sobol(d=5, scramble=True, seed=int(seed) + replicate)
        points = sampler.random_base2(sobol_power)
        values: list[complex] = []
        for point in points:
            q1, _density1 = _power_disk_sample(point[0], point[1], radial_power)
            q2, _density2 = _power_disk_sample(point[2], point[3], radial_power)
            ordering_index = min(int(point[4] * len(orderings)), len(orderings) - 1)
            selected_ordering = orderings[ordering_index]
            positions = linear_channel_positions_by_label(q1, q2, selected_ordering)
            log_mixture_density = oriented_bidisc_log_mixture_density_in_frame(
                positions,
                selected_ordering,
                radial_power=radial_power,
            )
            values.append(
                kernel.integrand_linear_gauge_weighted(
                    q1,
                    q2,
                    selected_ordering,
                    logarithmic_weight=-log_mixture_density,
                )
            )
        estimates.append(complex(np.mean(np.asarray(values, dtype=complex))))
    estimate_array = np.asarray(estimates, dtype=complex)
    mean = complex(np.mean(estimate_array))
    return EqualEnergyQMCResult(
        omega=kernel.omega,
        estimates=tuple(estimates),
        mean=mean,
        standard_error_real=float(
            np.std(estimate_array.real, ddof=1) / math.sqrt(replicates)
        ),
        standard_error_imag=float(
            np.std(estimate_array.imag, ddof=1) / math.sqrt(replicates)
        ),
        samples_per_replicate=2**sobol_power,
        replicates=replicates,
        block_order=kernel.block_order,
        momentum_order=kernel.momentum_order,
        block_scheme=kernel.block_scheme,
    )


def write_qmc_result(
    result: EqualEnergyQMCResult,
    path: Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_json(), indent=2) + "\n")

