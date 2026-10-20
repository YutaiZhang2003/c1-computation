"""Evaluate the full genus-two integrand at one saved moduli sampling point."""
from __future__ import annotations
import argparse
import cmath
import json
import math
import time
from typing import Mapping
import numpy as np
from audit_q_to_omega_accuracy import PeriodMapValidation, validate_or_refine_period_map
from bolza_torus_plumbing_reach import transform_omega
from ccy_plumbing_conventions import genus2_plumbing_coordinate_metadata, validate_genus2_plumbing_coordinates
from conformal_frame_labels import GLASSES_PLUMBING_FRAME, THETA_PLUMBING_FRAME
from genus2_c1_string_integrand import COMPACT_THETA_IMPLEMENTATION, DIMENSIONLESS_SEWING_SCALAR_NORMALIZATION, SameFrameMatterPartitions, XI_PHYSICAL_MOMENTUM_SCALAR_NORMALIZATION, genus2_c1_string_integrand_density_from_geometry, genus2_c1_string_integrand_geometry
from genus2_integrand_normalization import GENUS2_LOCAL_FUNDAMENTAL_DOMAIN_WEIGHT_FIELD, genus2_automorphism_metadata, xi_full_replacement_over_dimensionless, xi_genus2_scalar_over_dimensionless
from genus2_plumbing_atlas import symplectic_matrix_csv_fields
from genus2_holomorphic_period_table import SchottkyValidityEnvelope
from genus2_hybrid_period_map import MULTIPRECISION_HOLOMORPHIC_ALGORITHM
from genus2_period_table import Genus2PeriodMapTable
from genus2_liouville_evaluation import evaluate_liouville_rescaled
from genus2_scalar_evaluation import NoncompactScalarFixedCapConvergence, evaluate_noncompact_scalar_detailed, evaluate_noncompact_scalar_fixed_cap_convergence, scalar_word_safety_cap
from genus2_integration_geometry import _complex_string, validate_covariant_period_frames, select_plumbing_chart, table_first_period_validation, cached_period_solution_objects
from genus2_moduli_estimation import omega_from_csv_row, node_fundamental_domain_integration_weight, kernel_det_im_power



def evaluate_node(
    source_row: dict[str, str],
    args: argparse.Namespace,
    schottky_envelope: SchottkyValidityEnvelope | None = None,
    period_table: Genus2PeriodMapTable | None = None,
    period_recovery_solution: Mapping[str, object] | None = None,
) -> dict[str, object]:
    started = time.time()
    stage = "initialize"
    omega_fundamental = omega_from_csv_row(source_row)
    det_y = float(np.linalg.det(omega_fundamental.imag))
    sample_index = int(source_row["sample_index"])
    integration_weight = node_fundamental_domain_integration_weight(source_row)
    det_im_power = kernel_det_im_power(source_row)
    vacuum_word_tail_tolerance = (
        None
        if args.disable_vacuum_word_tail_check
        else args.vacuum_word_tail_tolerance
    )
    row: dict[str, object] = {
        "sample_index": sample_index,
        "status": "failed",
        "error": "",
        "det_im_omega": det_y,
        "x11": float(omega_fundamental[0, 0].real),
        "x12": float(omega_fundamental[0, 1].real),
        "x22": float(omega_fundamental[1, 1].real),
        "y11": float(omega_fundamental[0, 0].imag),
        "y12": float(omega_fundamental[0, 1].imag),
        "y22": float(omega_fundamental[1, 1].imag),
        "igusa_measure_frame": "siegel-fundamental-domain",
        GENUS2_LOCAL_FUNDAMENTAL_DOMAIN_WEIGHT_FIELD: integration_weight,
        "kernel_det_im_power": det_im_power,
    }
    for key, value in source_row.items():
        if key == "sampling_scheme" or key.startswith("rqmc_"):
            row[key] = value
    try:
        stage = "period-map-chart"
        if period_recovery_solution is None:
            chart, search_stage = select_plumbing_chart(
                omega_fundamental,
                q_reference_max=args.q_reference_max,
                refine_above_q=args.refine_above_q,
                base_search_depth=args.base_search_depth,
                base_prefilter_count=args.base_prefilter_count,
                base_word_length=args.base_word_length,
                base_period_tolerance=args.base_period_tolerance,
                base_stability_tolerance=args.base_stability_tolerance,
                refined_search_depth=args.refined_search_depth,
                refined_prefilter_count=args.refined_prefilter_count,
                refined_word_length=args.refined_word_length,
                refined_period_tolerance=args.refined_period_tolerance,
                refined_stability_tolerance=args.refined_stability_tolerance,
                early_stop_competitive_score_ratio=(
                    args.early_stop_competitive_score_ratio
                ),
                period_table=period_table,
                table_first_period_tolerance=args.period_validation_tolerance,
                table_first_stability_tolerance=args.period_validation_tolerance,
                table_first_maximum_word=args.table_first_maximum_word,
                table_first_maximum_corrections=args.table_first_maximum_corrections,
            )
            period_certificate: PeriodMapValidation | None = None
        else:
            chart, period_certificate = cached_period_solution_objects(
                source_row,
                period_recovery_solution,
                tolerance=args.period_validation_tolerance,
            )
            search_stage = "cached-period-recovery"
        atlas_q_values = tuple(complex(value) for value in chart.q)
        atlas_log_q_values = tuple(complex(value) for value in chart.log_q)
        # The exact integer marking is the authoritative frame map.  Parsing
        # the atlas' human-readable decimal Omega loses an absolute amount of
        # precision proportional to a very long cusp entry and previously
        # caused false failures of the 2e-9 algebraic consistency check.
        omega_chart = transform_omega(
            np.asarray(chart.matrix, dtype=np.int64),
            omega_fundamental,
        )
        omega_chart = 0.5 * (omega_chart + omega_chart.T)
        validate_covariant_period_frames(
            omega_fundamental,
            chart.matrix,
            omega_chart,
        )
        if period_certificate is None:
            if search_stage == "direct-all-small":
                period_certificate = table_first_period_validation(
                    chart,
                    atlas_q_values,
                    atlas_log_q_values,
                    tolerance=args.period_validation_tolerance,
                )
            else:
                period_certificate = validate_or_refine_period_map(
                    chart.topology,
                    omega_chart,
                    atlas_q_values,
                    word_length=args.period_validation_word_length,
                    word_step=args.period_validation_word_step,
                    tolerance=args.period_validation_tolerance,
                    reinverse_validation_word_length=(
                        args.period_reinverse_validation_word_length
                    ),
                    reinverse_max_nfev=args.period_reinverse_max_nfev,
                    initial_log_q=atlas_log_q_values,
                    schottky_envelope=schottky_envelope,
                )
        certified_period_algorithms = {
            "holomorphic-form-collocation",
            MULTIPRECISION_HOLOMORPHIC_ALGORITHM,
            "adaptive-schottky",
            "calibrated-schottky",
        }
        if (
            chart.period_algorithm not in certified_period_algorithms
            or period_certificate.period_algorithm not in certified_period_algorithms
        ):
            raise RuntimeError(
                "production node used an uncertified period-map backend: "
                f"atlas={chart.period_algorithm!r}, "
                f"certificate={period_certificate.period_algorithm!r}"
            )
        q_values = period_certificate.q
        log_q_values = period_certificate.log_q or tuple(cmath.log(value) for value in q_values)
        plumbing_coordinates = validate_genus2_plumbing_coordinates(
            chart.topology,
            q_values,
            log_q_values=log_q_values,
        )
        q_values = plumbing_coordinates.q_values
        log_q_values = plumbing_coordinates.log_q_values
        frame = THETA_PLUMBING_FRAME if chart.topology == "theta" else GLASSES_PLUMBING_FRAME
        scalar_high_cap = scalar_word_safety_cap(
            q_values,
            base_cap=args.scalar_word_high,
            large_q_cap=args.scalar_word_high_large_q,
            large_q_threshold=args.scalar_large_q_threshold,
        )

        stage = "noncompact-scalar-low"
        scalar_fixed_cap_convergence: (
            NoncompactScalarFixedCapConvergence | None
        ) = None
        if scalar_high_cap > args.scalar_word_low:
            scalar_fixed_cap_convergence = (
                evaluate_noncompact_scalar_fixed_cap_convergence(
                    chart.topology,
                    q_values,
                    omega_chart,
                    log_q_values=log_q_values,
                    initial_word_length=args.scalar_word_low,
                    maximum_word_length=scalar_high_cap,
                    word_length_increment=args.scalar_word_increment,
                    max_mode=args.scalar_max_mode,
                    tolerance=args.scalar_tolerance,
                    powered_relative_tolerance=(
                        args.scalar_powered_relative_tolerance
                    ),
                    confirmation_steps=args.scalar_confirmation_steps,
                )
            )
            if not scalar_fixed_cap_convergence.converged:
                raise RuntimeError(
                    f"{chart.topology} scalar did not pass the required successive "
                    "fixed-word-cap comparisons after 25th-power amplification"
                )
            scalar_low_evaluation = (
                scalar_fixed_cap_convergence.steps[0].evaluation
            )
            scalar_high_evaluation = scalar_fixed_cap_convergence.evaluation
        else:
            scalar_low_evaluation = evaluate_noncompact_scalar_detailed(
                chart.topology,
                q_values,
                omega_chart,
                log_q_values=log_q_values,
                word_length=args.scalar_word_low,
                max_mode=args.scalar_max_mode,
                tolerance=args.scalar_tolerance,
                word_tail_tolerance=args.scalar_word_tail_tolerance,
                minimum_word_length=args.scalar_minimum_word_length,
                powered_relative_tolerance=args.scalar_powered_relative_tolerance,
            )
            scalar_high_evaluation = scalar_low_evaluation
        scalar_low, scalar_tail_low, scalar_primitives_low = (
            scalar_low_evaluation.legacy_tuple()
        )
        scalar_high, scalar_tail_high, scalar_primitives_high = (
            scalar_high_evaluation.legacy_tuple()
        )
        stage = "liouville-low"
        liouville_low = evaluate_liouville_rescaled(
            chart.topology,
            q_values,
            log_q_values=log_q_values,
            block_order=args.block_order_low,
            quadrature_order=args.quadrature_order_low,
            quadrature_scheme=args.quadrature_scheme,
            dps=args.dps,
            vacuum_word_length=args.vacuum_word_length,
            vacuum_oscillator_level=args.vacuum_oscillator_level,
            vacuum_word_tail_tolerance=vacuum_word_tail_tolerance,
            vacuum_minimum_word_length=args.vacuum_minimum_word_length,
            global_block_tolerance=args.global_block_tolerance,
            global_block_max_outer_order=args.global_block_max_outer_order,
        )
        if (
            args.block_order_high == args.block_order_low
            and args.quadrature_order_high == args.quadrature_order_low
        ):
            liouville_high = liouville_low
        else:
            stage = "liouville-high"
            liouville_high = evaluate_liouville_rescaled(
                chart.topology,
                q_values,
                log_q_values=log_q_values,
                block_order=args.block_order_high,
                quadrature_order=args.quadrature_order_high,
                quadrature_scheme=args.quadrature_scheme,
                dps=args.dps,
                vacuum_word_length=args.vacuum_word_length,
                vacuum_oscillator_level=args.vacuum_oscillator_level,
                vacuum_word_tail_tolerance=vacuum_word_tail_tolerance,
                vacuum_minimum_word_length=args.vacuum_minimum_word_length,
                global_block_tolerance=args.global_block_tolerance,
                global_block_max_outer_order=args.global_block_max_outer_order,
            )

        stage = "fundamental-frame-integrand-geometry"
        integrand_geometry = genus2_c1_string_integrand_geometry(
            omega_fundamental,
            args.radius,
            lattice_tolerance=args.lattice_tolerance,
            theta_tolerance=args.theta_tolerance,
            chi10_normalization="product",
        )
        low_matter = SameFrameMatterPartitions(
            conformal_frame=frame,
            liouville_partition=liouville_low.scaled_partition,
            noncompact_scalar_partition=scalar_low,
        )
        high_matter = SameFrameMatterPartitions(
            conformal_frame=frame,
            liouville_partition=liouville_high.scaled_partition,
            noncompact_scalar_partition=scalar_high,
        )
        stage = "fundamental-frame-integrand-low"
        low = genus2_c1_string_integrand_density_from_geometry(
            integrand_geometry,
            matter_partitions=low_matter,
        )
        stage = "fundamental-frame-integrand-high"
        high = (
            low
            if high_matter == low_matter
            else genus2_c1_string_integrand_density_from_geometry(
                integrand_geometry,
                matter_partitions=high_matter,
            )
        )
        stage = "log-density-recombination"
        unit_mumford_log_low = (
            low.factorization_normalized_log_density
            + liouville_low.log_threshold_factor
        )
        unit_mumford_log_high = (
            high.factorization_normalized_log_density
            + liouville_high.log_threshold_factor
        )
        string_note_log_low = (
            low.string_note_kernel_log_density
            + liouville_low.log_threshold_factor
        )
        string_note_log_high = (
            high.string_note_kernel_log_density
            + liouville_high.log_threshold_factor
        )
        factorized_density_low = math.exp(unit_mumford_log_low)
        factorized_density_high = math.exp(unit_mumford_log_high)
        string_note_density_low = math.exp(string_note_log_low)
        string_note_density_high = math.exp(string_note_log_high)
        det_im_log_factor = float(det_im_power) * math.log(det_y)
        unit_mumford_transformed_low = math.exp(
            det_im_log_factor + unit_mumford_log_low
        )
        unit_mumford_transformed_high = math.exp(
            det_im_log_factor + unit_mumford_log_high
        )
        transformed_low = math.exp(det_im_log_factor + string_note_log_low)
        transformed_high = math.exp(det_im_log_factor + string_note_log_high)
        stage = "serialize-success"
        row.update(
            {
                "status": "ok",
                "search_stage": search_stage,
                "topology": chart.topology,
                "matter_frame": frame,
                "compact_theta_frame": "siegel-fundamental-domain",
                "liouville_scalar_quotient_frame": frame,
                "symplectic_word": chart.word,
                **symplectic_matrix_csv_fields(chart.matrix),
                "chart_status": chart.status,
                "atlas_period_algorithm": chart.period_algorithm,
                "atlas_period_map_region": chart.period_map_region,
                "atlas_plumbing_geometry_margin": chart.plumbing_geometry_margin,
                "atlas_period_overlap_residual": (
                    ""
                    if chart.period_overlap_residual is None
                    else chart.period_overlap_residual
                ),
                "atlas_inverse_seed_source": chart.inverse_seed_source,
                "atlas_q1": _complex_string(atlas_q_values[0]),
                "atlas_q2": _complex_string(atlas_q_values[1]),
                "atlas_q3": _complex_string(atlas_q_values[2]),
                "atlas_log_q1": _complex_string(atlas_log_q_values[0]),
                "atlas_log_q2": _complex_string(atlas_log_q_values[1]),
                "atlas_log_q3": _complex_string(atlas_log_q_values[2]),
                "q1": _complex_string(q_values[0]),
                "q2": _complex_string(q_values[1]),
                "q3": _complex_string(q_values[2]),
                "log_q1": _complex_string(log_q_values[0]),
                "log_q2": _complex_string(log_q_values[1]),
                "log_q3": _complex_string(log_q_values[2]),
                **genus2_plumbing_coordinate_metadata(
                    plumbing_coordinates,
                    formatter=_complex_string,
                ),
                "q1_abs": math.exp(log_q_values[0].real) if log_q_values[0].real > -745.0 else 0.0,
                "q2_abs": math.exp(log_q_values[1].real) if log_q_values[1].real > -745.0 else 0.0,
                "q3_abs": math.exp(log_q_values[2].real) if log_q_values[2].real > -745.0 else 0.0,
                "atlas_q_max": chart.q_max,
                "q_max": math.exp(max(value.real for value in log_q_values)),
                "period_algorithm": period_certificate.period_algorithm,
                "period_map_region": period_certificate.period_map_region or "",
                "period_overlap_residual": (
                    ""
                    if period_certificate.overlap_residual is None
                    else period_certificate.overlap_residual
                ),
                "period_agreement_tolerance": (
                    ""
                    if period_certificate.agreement_tolerance is None
                    else period_certificate.agreement_tolerance
                ),
                "period_validity_cell_id": period_certificate.validity_cell_id or "",
                "period_validity_reference_table_sha256": (
                    period_certificate.validity_reference_table_sha256 or ""
                ),
                "period_certified_error_bound": (
                    ""
                    if period_certificate.certified_error_bound is None
                    else period_certificate.certified_error_bound
                ),
                "fixed_cft_order": (
                    args.block_order_low == args.block_order_high
                    and args.quadrature_order_low == args.quadrature_order_high
                ),
                "block_order_low": args.block_order_low,
                "block_order_high": args.block_order_high,
                "quadrature_order_low": args.quadrature_order_low,
                "quadrature_order_high": args.quadrature_order_high,
                "scalar_word_low": args.scalar_word_low,
                "scalar_word_high": args.scalar_word_high,
                "scalar_word_high_large_q": args.scalar_word_high_large_q,
                "scalar_large_q_threshold": args.scalar_large_q_threshold,
                "scalar_selected_word_safety_cap": scalar_high_cap,
                "scalar_word_tail_tolerance": args.scalar_word_tail_tolerance,
                "scalar_powered_relative_tolerance": (
                    args.scalar_powered_relative_tolerance
                ),
                "scalar_word_increment": args.scalar_word_increment,
                "scalar_confirmation_steps": args.scalar_confirmation_steps,
                "scalar_fixed_cap_convergence_basis": (
                    "successive fixed-cap values of Z_X^-25"
                    if scalar_fixed_cap_convergence is not None
                    else "legacy theta/rank-reduced evaluation"
                ),
                "scalar_fixed_cap_converged": (
                    ""
                    if scalar_fixed_cap_convergence is None
                    else scalar_fixed_cap_convergence.converged
                ),
                "scalar_fixed_cap_terminal_powered_relative_change": (
                    ""
                    if scalar_fixed_cap_convergence is None
                    else scalar_fixed_cap_convergence.terminal_powered_relative_change
                ),
                "scalar_fixed_cap_history": (
                    ""
                    if scalar_fixed_cap_convergence is None
                    else json.dumps(
                        scalar_fixed_cap_convergence.records(),
                        separators=(",", ":"),
                    )
                ),
                "scalar_minimum_word_length": args.scalar_minimum_word_length,
                "vacuum_word_length": args.vacuum_word_length,
                "vacuum_oscillator_level": args.vacuum_oscillator_level,
                "vacuum_word_tail_tolerance": (
                    ""
                    if vacuum_word_tail_tolerance is None
                    else vacuum_word_tail_tolerance
                ),
                "vacuum_word_tail_check_enabled": (
                    vacuum_word_tail_tolerance is not None
                ),
                "vacuum_minimum_word_length": args.vacuum_minimum_word_length,
                "period_residual": chart.period_max_residual,
                "period_map_stability": chart.period_map_stability,
                "period_cusp_word_length": args.period_validation_word_length,
                "period_cusp_word_step": args.period_validation_word_step,
                "period_validation_tolerance": args.period_validation_tolerance,
                "period_fixed_q_residual": period_certificate.fixed_q_residual,
                "period_fixed_q_map_step": period_certificate.fixed_q_stability,
                "period_q_refined": period_certificate.refined,
                "period_final_residual": period_certificate.final_residual,
                "period_final_map_step": period_certificate.final_stability,
                "period_map_low_order": period_certificate.low_order,
                "period_map_high_order": period_certificate.high_order,
                "period_map_validation_order": period_certificate.validation_order,
                "period_seam_residual": period_certificate.seam_residual,
                "period_symmetry_error": period_certificate.symmetry_error,
                "period_reinverse_success": period_certificate.reinverse_success,
                "period_reinverse_message": period_certificate.reinverse_message,
                "period_reinverse_nfev": period_certificate.reinverse_nfev,
                "period_reinverse_max_tau_shift": period_certificate.max_tau_shift,
                "liouville_low": liouville_low.raw_partition,
                "liouville_high": liouville_high.raw_partition,
                "liouville_log_low": liouville_low.log_partition,
                "liouville_log_high": liouville_high.log_partition,
                "liouville_scaled_low": liouville_low.scaled_partition,
                "liouville_scaled_high": liouville_high.scaled_partition,
                "liouville_threshold_log": liouville_high.log_threshold_factor,
                "liouville_relative_change": (
                    liouville_high.scaled_partition
                    / liouville_low.scaled_partition
                    - 1.0
                ),
                "scalar_low": scalar_low,
                "scalar_high": scalar_high,
                "scalar_input_normalization": DIMENSIONLESS_SEWING_SCALAR_NORMALIZATION,
                "scalar_integrand_normalization": XI_PHYSICAL_MOMENTUM_SCALAR_NORMALIZATION,
                "xi_scalar_over_dimensionless": xi_genus2_scalar_over_dimensionless(),
                "xi_full_replacement_over_dimensionless": (
                    xi_full_replacement_over_dimensionless()
                ),
                "scalar_relative_change": scalar_high / scalar_low - 1.0,
                "scalar_primitive_count_low": scalar_primitives_low,
                "scalar_primitive_count_high": scalar_primitives_high,
                "scalar_effective_word_length_low": (
                    scalar_low_evaluation.effective_word_length
                ),
                "scalar_effective_word_length_high": (
                    scalar_high_evaluation.effective_word_length
                ),
                "scalar_mode_tail_low": scalar_tail_low,
                "scalar_mode_tail_high": scalar_tail_high,
                "scalar_word_tail_low": (
                    ""
                    if scalar_low_evaluation.primitive_word_chiral_log_tail_estimate
                    is None
                    else scalar_low_evaluation.primitive_word_chiral_log_tail_estimate
                ),
                "scalar_word_tail_high": (
                    ""
                    if scalar_high_evaluation.primitive_word_chiral_log_tail_estimate
                    is None
                    else scalar_high_evaluation.primitive_word_chiral_log_tail_estimate
                ),
                "scalar_powered_log_error_estimate_high": (
                    ""
                    if scalar_high_evaluation.powered_log_error_estimate() is None
                    else scalar_high_evaluation.powered_log_error_estimate()
                ),
                "scalar_powered_relative_error_estimate_high": (
                    ""
                    if scalar_high_evaluation.powered_relative_error_estimate() is None
                    else scalar_high_evaluation.powered_relative_error_estimate()
                ),
                "chi10_abs": abs(high.chi10),
                "chi10_log_abs": high.chi10_log_abs,
                "compact_winding_sum": high.compact_winding_sum,
                "lattice_nmax": high.lattice_nmax,
                "compact_theta_algorithm": high.compact_theta_algorithm,
                "compact_theta_implementation": COMPACT_THETA_IMPLEMENTATION,
                "compact_theta_momentum_nmax": high.compact_theta_momentum_nmax,
                "compact_theta_winding_nmax": high.compact_theta_winding_nmax,
                "factorized_density_low": factorized_density_low,
                "factorized_density_high": factorized_density_high,
                "factorized_log_density_low": unit_mumford_log_low,
                "factorized_log_density_high": unit_mumford_log_high,
                "integration_kernel_convention": high.string_note_kernel_convention,
                **genus2_automorphism_metadata(),
                "string_note_kernel_multiplier": high.string_note_kernel_multiplier,
                "string_note_kernel_density_low": string_note_density_low,
                "string_note_kernel_density_high": string_note_density_high,
                "string_note_kernel_log_density_low": string_note_log_low,
                "string_note_kernel_log_density_high": string_note_log_high,
                "unit_mumford_transformed_integrand_low": unit_mumford_transformed_low,
                "unit_mumford_transformed_integrand_high": unit_mumford_transformed_high,
                "transformed_integrand_low": transformed_low,
                "transformed_integrand_high": transformed_high,
                "node_contribution_low": (
                    integration_weight * transformed_low
                ),
                "node_contribution_high": (
                    integration_weight * transformed_high
                ),
                "combined_relative_change": transformed_high / transformed_low - 1.0,
            }
        )
    except Exception as exc:  # noqa: BLE001 - every failed node is saved.
        if bool(getattr(args, "raise_on_error", False)):
            raise
        row["error"] = f"{type(exc).__name__} at {stage}: {exc}"
    row["runtime_seconds"] = time.time() - started
    return row

