#!/usr/bin/env python3
"""RQMC integration of the genus-one c=1 two-point worldsheet amplitude.

The calculation is performed at ``omega=i*x``, ``0<x<1``.  In this strip the
DOZZ momentum contours stay on the positive real axes and the worldsheet
integral is convergent.  The torus cusp is handled by a sequence of explicit
cutoffs, followed by an inverse-cutoff fit; no matrix-model formula is used in
the estimator or in the fit.
"""

from __future__ import annotations

from c1_paths import PACKAGE_ROOT, data_root, package_path, module_path

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import qmc

try:
    from c1_string_conventions import BRY_GENUS1_TWO_POINT_PREFACTOR, CURRENT_STRING_COUPLING_CONVENTION
    from genus1_two_point_worldsheet import (
        LiouvilleTorusTwoPoint,
        MomentumRule,
        reduced_worldsheet_integrand,
    )
except ImportError:  # pragma: no cover
    from c1_string_conventions import BRY_GENUS1_TWO_POINT_PREFACTOR, CURRENT_STRING_COUPLING_CONVENTION
    from genus1_two_point_worldsheet import (
        LiouvilleTorusTwoPoint,
        MomentumRule,
        reduced_worldsheet_integrand,
    )


def _complex_record(value: complex) -> dict[str, float]:
    value = complex(value)
    return {"real": float(value.real), "imag": float(value.imag)}


def integrate_cutoffs_one_replicate(
    correlator: LiouvilleTorusTwoPoint,
    *,
    cutoffs: tuple[float, ...],
    sobol_power: int,
    seed: int,
    epsilon: float,
    collision_radius: float,
) -> tuple[np.ndarray, dict[str, float]]:
    sampler = qmc.Sobol(d=4, scramble=True, seed=int(seed))
    points = sampler.random_base2(int(sobol_power))
    estimates = np.zeros(len(cutoffs), dtype=np.complex128)
    maximum_absolute_sample = 0.0
    ope_count = 0

    for cutoff_index, cutoff in enumerate(cutoffs):
        values = np.zeros(len(points), dtype=np.complex128)
        for point_index, point in enumerate(points):
            tau1 = float(point[0]) - 0.5
            tau2_min = math.sqrt(1.0 - tau1 * tau1)
            tau2 = tau2_min + float(point[1]) * (float(cutoff) - tau2_min)
            tau = tau1 + 1.0j * tau2
            r1 = float(point[2])
            r2 = 0.5 * float(point[3])
            z = 2.0 * math.pi * (r1 + r2 * tau)

            local_distance = min(abs(z), abs(z - 2.0 * math.pi))
            if local_distance < 2.0 * math.pi * epsilon:
                ope_count += 1
            if local_distance >= collision_radius:
                integrand = reduced_worldsheet_integrand(
                    correlator,
                    z,
                    tau,
                    epsilon=epsilon,
                )
                # Twice the lower-half-torus domain, with its full collision
                # disc removed.  The removed disc is restored analytically.
                bulk_z_integral_sample = 4.0 * math.pi**2 * tau2 * integrand
            else:
                bulk_z_integral_sample = 0.0 + 0.0j
            collision_disc = correlator.leading_collision_disc(tau, collision_radius)
            tau_jacobian = float(cutoff) - tau2_min
            values[point_index] = tau_jacobian * (
                bulk_z_integral_sample + collision_disc
            )
        estimates[cutoff_index] = np.mean(values)
        maximum_absolute_sample = max(maximum_absolute_sample, float(np.max(np.abs(values))))

    diagnostics = {
        "maximum_absolute_sample": maximum_absolute_sample,
        "ope_fraction_all_cutoffs": float(ope_count / (len(points) * len(cutoffs))),
    }
    return estimates, diagnostics


def fit_cusp_limit(cutoffs: np.ndarray, estimates: np.ndarray) -> tuple[complex, complex, complex]:
    """Fit ``I(T)=I_inf+a/T+b/T^2`` without reference to a target answer."""
    design = np.column_stack(
        [np.ones_like(cutoffs), 1.0 / cutoffs, 1.0 / (cutoffs * cutoffs)]
    )
    real_fit, *_ = np.linalg.lstsq(design, estimates.real, rcond=None)
    imag_fit, *_ = np.linalg.lstsq(design, estimates.imag, rcond=None)
    combined = real_fit + 1.0j * imag_fit
    return complex(combined[0]), complex(combined[1]), complex(combined[2])


def integrate_tau_slice_one_replicate(
    correlator: LiouvilleTorusTwoPoint,
    *,
    tau2: float,
    sobol_power: int,
    seed: int,
    epsilon: float,
    collision_radius: float,
) -> complex:
    """Integrate ``tau1`` and ``z`` at fixed large ``tau2``."""
    sampler = qmc.Sobol(d=3, scramble=True, seed=int(seed))
    points = sampler.random_base2(int(sobol_power))
    values = np.zeros(len(points), dtype=np.complex128)
    tau2 = float(tau2)
    for point_index, point in enumerate(points):
        tau1 = float(point[0]) - 0.5
        tau = tau1 + 1.0j * tau2
        # Rectangular torus coordinates used in Appendix B.2:
        # z=2*pi*(r1+r2*tau), with 0<r2<1/2.
        r1 = float(point[1])
        r2 = 0.5 * float(point[2])
        z = 2.0 * math.pi * (r1 + r2 * tau)
        local_distance = min(abs(z), abs(z - 2.0 * math.pi))
        if local_distance >= collision_radius:
            bulk = (
                4.0
                * math.pi**2
                * tau2
                * reduced_worldsheet_integrand(
                    correlator,
                    z,
                    tau,
                    epsilon=epsilon,
                )
            )
        else:
            bulk = 0.0 + 0.0j
        values[point_index] = bulk + correlator.leading_collision_disc(
            tau,
            collision_radius,
        )
    return complex(np.mean(values))


def fit_tau_integrand_tail(
    tau2_values: np.ndarray,
    slice_values: np.ndarray,
) -> tuple[complex, complex, complex]:
    """Fit ``a0*t^-2+a1*t^-5/3+a2*t^-3`` as in BRY Appendix B.2."""
    design = np.column_stack(
        [tau2_values**-2.0, tau2_values ** (-5.0 / 3.0), tau2_values**-3.0]
    )
    real_fit, *_ = np.linalg.lstsq(design, slice_values.real, rcond=None)
    imag_fit, *_ = np.linalg.lstsq(design, slice_values.imag, rcond=None)
    combined = real_fit + 1.0j * imag_fit
    return complex(combined[0]), complex(combined[1]), complex(combined[2])


def integrated_fitted_tail(
    tail_start: float,
    coefficients: tuple[complex, complex, complex],
) -> complex:
    a0, a1, a2 = coefficients
    tail_start = float(tail_start)
    return (
        a0 / tail_start
        + 1.5 * a1 * tail_start ** (-2.0 / 3.0)
        + 0.5 * a2 / (tail_start * tail_start)
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    if not 0.0 < args.x < 1.0:
        raise ValueError("the direct real-contour computation requires 0 < x < 1")
    cutoffs = tuple(float(value) for value in args.cutoffs.split(","))
    if len(cutoffs) < 3 or sorted(cutoffs) != list(cutoffs):
        raise ValueError("provide at least three increasing cusp cutoffs")
    if cutoffs[0] <= 1.0:
        raise ValueError("the smallest cusp cutoff must exceed 1")

    momentum_rule = MomentumRule.power_legendre(
        args.p_max,
        args.momentum_order,
        args.momentum_power,
    )
    correlator = LiouvilleTorusTwoPoint(
        1.0j * args.x,
        momentum_rule=momentum_rule,
        necklace_orders=(args.necklace_order_first, args.necklace_order_second),
        ope_orders=(args.ope_q_order, args.ope_z_order),
        necklace_backend=getattr(
            args,
            "necklace_backend",
            "c-recursion",
        ),
        ope_backend=getattr(args, "ope_backend", "c-recursion"),
        h_recursion_regulator=getattr(args, "h_recursion_regulator", 0.04),
        h_recursion_weight_regulator=getattr(
            args,
            "h_recursion_weight_regulator",
            0.001,
        ),
        h_recursion_audit_tolerance=getattr(
            args,
            "h_recursion_audit_tolerance",
            1.0e-7,
        ),
        special_dps=args.dps,
    )

    replicate_values: list[np.ndarray] = []
    replicate_diagnostics: list[dict[str, float]] = []
    tail_slices = tuple(float(value) for value in args.tail_slices.split(","))
    if len(tail_slices) < 3 or min(tail_slices) < cutoffs[-1]:
        raise ValueError("tail slices must contain at least three values at or above the last cutoff")
    replicate_slice_values: list[np.ndarray] = []
    replicate_tail_coefficients: list[tuple[complex, complex, complex]] = []
    replicate_tail_integrals: list[complex] = []
    replicate_final_values: list[complex] = []
    for replicate in range(args.replicates):
        values, diagnostics = integrate_cutoffs_one_replicate(
            correlator,
            cutoffs=cutoffs,
            sobol_power=args.sobol_power,
            seed=args.seed + replicate,
            epsilon=args.epsilon,
            collision_radius=args.collision_radius,
        )
        replicate_values.append(values)
        replicate_diagnostics.append(diagnostics)
        slices = np.asarray(
            [
                integrate_tau_slice_one_replicate(
                    correlator,
                    tau2=tau2,
                    sobol_power=args.tail_sobol_power,
                    seed=args.seed + 10000 + 97 * replicate + slice_index,
                    epsilon=args.epsilon,
                    collision_radius=args.collision_radius,
                )
                for slice_index, tau2 in enumerate(tail_slices)
            ]
        )
        tail_coefficients = fit_tau_integrand_tail(np.asarray(tail_slices), slices)
        tail_integral = integrated_fitted_tail(cutoffs[-1], tail_coefficients)
        final_value = complex(values[-1] + tail_integral)
        replicate_slice_values.append(slices)
        replicate_tail_coefficients.append(tail_coefficients)
        replicate_tail_integrals.append(tail_integral)
        replicate_final_values.append(final_value)
        print(
            f"replicate {replicate + 1}/{args.replicates}: "
            + ", ".join(
                f"T={cutoff:g}: {value.real:+.9e}{value.imag:+.2e}j"
                for cutoff, value in zip(cutoffs, values)
            )
            + f", fitted tail={tail_integral.real:+.4e}, final={final_value.real:+.9e}",
            flush=True,
        )

    replicate_array = np.asarray(replicate_values)
    means = np.mean(replicate_array, axis=0)
    if args.replicates > 1:
        standard_errors = np.std(replicate_array, axis=0, ddof=1) / math.sqrt(args.replicates)
    else:
        standard_errors = np.full(len(cutoffs), np.nan + 1.0j * np.nan)

    fitted_replicates = np.asarray(
        [fit_cusp_limit(np.asarray(cutoffs), values)[0] for values in replicate_array]
    )
    fitted_limit, fitted_a, fitted_b = fit_cusp_limit(np.asarray(cutoffs), means)
    if args.replicates > 1:
        fitted_standard_error = np.std(fitted_replicates, ddof=1) / math.sqrt(args.replicates)
    else:
        fitted_standard_error = complex(float("nan"), float("nan"))

    slice_array = np.asarray(replicate_slice_values)
    slice_means = np.mean(slice_array, axis=0)
    tail_coefficient_means = tuple(
        np.mean(np.asarray(replicate_tail_coefficients), axis=0)
    )
    tail_integral_mean = complex(np.mean(replicate_tail_integrals))
    final_mean = complex(np.mean(replicate_final_values))
    if args.replicates > 1:
        final_standard_error = complex(
            float(np.std(np.asarray(replicate_final_values).real, ddof=1) / math.sqrt(args.replicates)),
            float(np.std(np.asarray(replicate_final_values).imag, ddof=1) / math.sqrt(args.replicates)),
        )
    else:
        final_standard_error = complex(float("nan"), float("nan"))

    result: dict[str, object] = {
        "calculation": "direct c=1 genus-one two-point worldsheet integral",
        "blind_freeze": True,
        "blind_freeze_statement": (
            "The estimator, cutoff treatment, and reported native amplitude do "
            "not evaluate a matrix-model or literature target."
        ),
        "native_convention": True,
        "native_normalization": "A_1^ws(omega)=2*pi^2*i*g_s^2*I_1(omega)",
        "string_coupling_convention": CURRENT_STRING_COUPLING_CONVENTION,
        "omega": _complex_record(correlator.omega),
        "x": float(args.x),
        "domain": "omega=i*x, 0<x<1; no DOZZ pole crosses the real P contours",
        "patch_epsilon": float(args.epsilon),
        "collision_disc": {
            "radius": float(args.collision_radius),
            "treatment": "leading OPE term integrated analytically",
        },
        "momentum_rule": {
            "kind": "Gauss-Legendre after P=p_max*u^power",
            "p_max": float(args.p_max),
            "order": int(args.momentum_order),
            "power": float(args.momentum_power),
        },
        "block_orders": {
            "necklace_q1": int(args.necklace_order_first),
            "necklace_q2": int(args.necklace_order_second),
            "ope_q": int(args.ope_q_order),
            "ope_z": int(args.ope_z_order),
        },
        "block_backends": {
            "necklace": correlator.necklace_backend,
            "ope": correlator.ope_backend,
            "h_recursion_c_regulator": float(correlator.h_recursion_regulator),
            "h_recursion_weight_regulator": float(
                correlator.h_recursion_weight_regulator
            ),
            "h_recursion_audit_tolerance": float(
                correlator.h_recursion_audit_tolerance
            ),
            "h_recursion_audit_max_relative_error": float(
                correlator.h_recursion_audit_max_relative_error
            ),
            "h_recursion_node_count": int(correlator.h_recursion_node_count),
            "h_recursion_direct_fallback_count": int(
                correlator.h_recursion_fallback_count
            ),
            "frame_conversion": (
                "primary powers, q^(-c/24), and the OPE "
                "(2*sin(z/2))^(-2*d) factor are applied outside both recursions"
            ),
        },
        "rqmc": {
            "sobol_power": int(args.sobol_power),
            "points_per_replicate": int(2**args.sobol_power),
            "replicates": int(args.replicates),
            "seed": int(args.seed),
            "seed_scheme": "bulk replicate r uses seed+r",
        },
        "tail_rqmc": {
            "sobol_power": int(args.tail_sobol_power),
            "points_per_slice_replicate": int(2**args.tail_sobol_power),
            "replicates": int(args.replicates),
            "seed_scheme": (
                "tail replicate r and slice j use seed+10000+97*r+j"
            ),
        },
        "special_dps": int(args.dps),
        "cutoffs": [float(value) for value in cutoffs],
        "cutoff_means": [_complex_record(value) for value in means],
        "cutoff_standard_errors": [_complex_record(value) for value in standard_errors],
        "replicate_values": [
            [_complex_record(value) for value in row] for row in replicate_array
        ],
        "replicate_diagnostics": replicate_diagnostics,
        "cusp_fit": {
            "legacy_cumulative_ansatz": "I(T)=I_infinity+a/T+b/T^2",
            "I_infinity": _complex_record(fitted_limit),
            "a": _complex_record(fitted_a),
            "b": _complex_record(fitted_b),
            "rqmc_standard_error": _complex_record(fitted_standard_error),
            "replicate_limits": [_complex_record(value) for value in fitted_replicates],
            "tau_integrand_ansatz": "f(t)=a0*t^-2+a1*t^-5/3+a2*t^-3",
            "tail_start": float(cutoffs[-1]),
            "tau2_slices": [float(value) for value in tail_slices],
            "slice_means": [_complex_record(value) for value in slice_means],
            "mean_coefficients": {
                "a0": _complex_record(tail_coefficient_means[0]),
                "a1": _complex_record(tail_coefficient_means[1]),
                "a2": _complex_record(tail_coefficient_means[2]),
            },
            "mean_integrated_tail": _complex_record(tail_integral_mean),
            "final_I": _complex_record(final_mean),
            "final_rqmc_standard_error": _complex_record(final_standard_error),
            "replicate_finals": [_complex_record(value) for value in replicate_final_values],
        },
        "native_amplitude": {
            "A_over_i_gs2": _complex_record(BRY_GENUS1_TWO_POINT_PREFACTOR * final_mean),
            "rqmc_standard_error": _complex_record(
                BRY_GENUS1_TWO_POINT_PREFACTOR * final_standard_error
            ),
        },
        "upsilon_cache_size": int(len(correlator.special._log_cache)),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    print(
        "I_final="
        f"{final_mean.real:+.12e}{final_mean.imag:+.3e}j "
        f"(RQMC SE {final_standard_error.real:.3e})"
    )
    return result


def parser() -> argparse.ArgumentParser:
    out = argparse.ArgumentParser()
    out.add_argument("--x", type=float, required=True)
    out.add_argument("--p-max", type=float, default=5.0)
    out.add_argument("--momentum-order", type=int, default=12)
    out.add_argument("--momentum-power", type=float, default=2.0)
    out.add_argument("--necklace-order-first", type=int, default=5)
    out.add_argument("--necklace-order-second", type=int, default=5)
    out.add_argument("--ope-q-order", type=int, default=2)
    out.add_argument("--ope-z-order", type=int, default=7)
    out.add_argument(
        "--necklace-backend",
        choices=("c-recursion", "direct-descendants"),
        default="c-recursion",
    )
    out.add_argument(
        "--ope-backend",
        choices=("c-recursion", "direct-descendants"),
        default="c-recursion",
    )
    out.add_argument("--h-recursion-regulator", type=float, default=0.04)
    out.add_argument("--h-recursion-weight-regulator", type=float, default=0.001)
    out.add_argument("--h-recursion-audit-tolerance", type=float, default=1.0e-7)
    out.add_argument("--epsilon", type=float, default=0.15)
    out.add_argument("--collision-radius", type=float, default=0.15)
    out.add_argument("--cutoffs", default="3,4,6,8")
    out.add_argument("--sobol-power", type=int, default=11)
    out.add_argument("--replicates", type=int, default=4)
    out.add_argument("--tail-slices", default="8,10,12,16,20")
    out.add_argument("--tail-sobol-power", type=int, default=10)
    out.add_argument("--seed", type=int, default=170507151)
    out.add_argument("--dps", type=int, default=26)
    out.add_argument(
        "--output",
        default=str(package_path('plumbing/results/genus1_two_point_worldsheet/native_q_pilot.json')),
    )
    return out


if __name__ == "__main__":
    run(parser().parse_args())
