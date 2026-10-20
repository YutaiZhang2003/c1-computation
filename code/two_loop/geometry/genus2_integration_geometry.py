"""Plumbing-chart selection, period-map checks, and saved coordinate reconstruction."""
from __future__ import annotations
import argparse
import math
from typing import Mapping, Sequence
import numpy as np
from audit_q_to_omega_accuracy import PeriodMapValidation
from bolza_torus_plumbing_reach import transform_omega
from ccy_plumbing_conventions import validate_genus2_plumbing_coordinates
from genus2_plumbing_atlas import MULTIPRECISION_FALLBACK_Q_MAX, PlumbingChartResult, best_leading_score, build_plumbing_atlas, symplectic_matrix_from_csv_row, table_first_all_small_chart, table_first_mixed_cusp_chart
from genus2_hybrid_period_map import MULTIPRECISION_HOLOMORPHIC_ALGORITHM
from genus2_period_table import Genus2PeriodMapTable
from genus2_siegel_fundamental_domain import in_gottschling_domain



USABLE_CHART_STATUSES = {"reference-q-envelope", "requires-recursion-order-study"}

def _complex_string(value: complex) -> str:
    value = complex(value)
    return f"{value.real:+.16e}{value.imag:+.16e}j"


def _matrix_from_strings(values: Sequence[Sequence[str]]) -> np.ndarray:
    return np.asarray([[complex(value) for value in row] for row in values], dtype=np.complex128)


def validate_covariant_period_frames(
    omega_fundamental: np.ndarray,
    marking: Sequence[Sequence[int]],
    omega_plumbing: np.ndarray,
    *,
    tolerance: float = 2.0e-9,
) -> None:
    """Keep the modular-covariant kernel and invariant ratios in their frames."""

    fundamental = np.asarray(omega_fundamental, dtype=np.complex128)
    plumbing = np.asarray(omega_plumbing, dtype=np.complex128)
    if not bool(in_gottschling_domain(fundamental, tolerance=float(tolerance))):
        raise ValueError("the Igusa/measure period matrix is outside the Gottschling domain")
    expected_plumbing = transform_omega(np.asarray(marking, dtype=np.int64), fundamental)
    mismatch = float(np.max(np.abs(expected_plumbing - plumbing)))
    if mismatch > float(tolerance):
        raise ValueError(
            "the plumbing period matrix is not the saved Sp(4,Z) image of the "
            f"fundamental-domain matrix (mismatch={mismatch:.3e})"
        )


def _best_usable_chart(atlas: object) -> PlumbingChartResult | None:
    usable = [chart for chart in atlas.charts if chart.status in USABLE_CHART_STATUSES]
    return min(usable, key=lambda chart: (chart.q_max, chart.period_max_residual)) if usable else None


def select_plumbing_chart(
    omega: np.ndarray,
    *,
    q_reference_max: float,
    refine_above_q: float,
    base_search_depth: int,
    base_prefilter_count: int,
    base_word_length: int,
    base_period_tolerance: float,
    base_stability_tolerance: float,
    refined_search_depth: int,
    refined_prefilter_count: int,
    refined_word_length: int,
    refined_period_tolerance: float,
    refined_stability_tolerance: float,
    early_stop_competitive_score_ratio: float = 1.02,
    period_table: Genus2PeriodMapTable | None = None,
    table_first_period_tolerance: float | None = None,
    table_first_stability_tolerance: float | None = None,
    table_first_maximum_word: int = 5,
    table_first_maximum_corrections: int = 2,
) -> tuple[PlumbingChartResult, str]:
    fast_period_tolerance = (
        float(base_period_tolerance)
        if table_first_period_tolerance is None
        else float(table_first_period_tolerance)
    )
    fast_stability_tolerance = (
        float(base_stability_tolerance)
        if table_first_stability_tolerance is None
        else float(table_first_stability_tolerance)
    )
    table_chart = table_first_all_small_chart(
        omega,
        period_table=period_table,
        leading_search_depth=base_search_depth,
        leading_prefilter_count=max(2, base_prefilter_count),
        maximum_word=table_first_maximum_word,
        maximum_corrections=table_first_maximum_corrections,
        q_reference_max=q_reference_max,
        period_tolerance=fast_period_tolerance,
        stability_tolerance=fast_stability_tolerance,
    )
    if table_chart is not None:
        return table_chart, "direct-all-small"

    mixed_chart = table_first_mixed_cusp_chart(
        omega,
        period_table=period_table,
        leading_search_depth=base_search_depth,
        leading_prefilter_count=max(2, base_prefilter_count),
        maximum_candidates=max(8, 4 * base_prefilter_count),
        q_reference_max=q_reference_max,
        period_tolerance=fast_period_tolerance,
        stability_tolerance=fast_stability_tolerance,
    )
    if mixed_chart is not None:
        return mixed_chart, "direct-mixed-cusp"

    refined_first = False
    if refined_search_depth > base_search_depth:
        base_leading = min(
            best_leading_score(omega, topology, search_depth=base_search_depth)
            for topology in ("theta", "glasses")
        )
        refined_leading = min(
            best_leading_score(omega, topology, search_depth=refined_search_depth)
            for topology in ("theta", "glasses")
        )
        refined_first = bool(
            base_leading > MULTIPRECISION_FALLBACK_Q_MAX
            and refined_leading <= MULTIPRECISION_FALLBACK_Q_MAX
        )

    if refined_first:
        refined = build_plumbing_atlas(
            omega,
            search_depth=refined_search_depth,
            prefilter_count=refined_prefilter_count,
            word_length=refined_word_length,
            max_nfev=180,
            q_reference_max=q_reference_max,
            period_tolerance=refined_period_tolerance,
            stability_tolerance=refined_stability_tolerance,
            stop_at_reference=True,
            stop_at_usable_q_max=MULTIPRECISION_FALLBACK_Q_MAX,
            early_stop_competitive_score_ratio=early_stop_competitive_score_ratio,
            period_table=period_table,
        )
        refined_chart = _best_usable_chart(refined)
        if refined_chart is not None:
            return refined_chart, "refined"

    base = build_plumbing_atlas(
        omega,
        search_depth=base_search_depth,
        prefilter_count=base_prefilter_count,
        word_length=base_word_length,
        q_reference_max=q_reference_max,
        period_tolerance=base_period_tolerance,
        stability_tolerance=base_stability_tolerance,
        stop_at_reference=True,
        stop_at_usable_q_max=MULTIPRECISION_FALLBACK_Q_MAX,
        early_stop_competitive_score_ratio=early_stop_competitive_score_ratio,
        period_table=period_table,
    )
    candidates: list[tuple[str, PlumbingChartResult]] = []
    base_chart = _best_usable_chart(base)
    if base_chart is not None:
        candidates.append(("base", base_chart))

    base_needs_refinement = base_chart is None or base_chart.q_max > refine_above_q
    if base_needs_refinement:
        refined = build_plumbing_atlas(
            omega,
            search_depth=refined_search_depth,
            prefilter_count=refined_prefilter_count,
            word_length=refined_word_length,
            max_nfev=180,
            q_reference_max=q_reference_max,
            period_tolerance=refined_period_tolerance,
            stability_tolerance=refined_stability_tolerance,
            stop_at_reference=True,
            stop_at_usable_q_max=MULTIPRECISION_FALLBACK_Q_MAX,
            early_stop_competitive_score_ratio=early_stop_competitive_score_ratio,
            period_table=period_table,
        )
        refined_chart = _best_usable_chart(refined)
        if refined_chart is not None:
            candidates.append(("refined", refined_chart))

    if not candidates:
        raise RuntimeError("no finite-q plumbing chart passed the period-map checks")
    stage, chart = min(candidates, key=lambda item: (item[1].q_max, item[1].period_max_residual))
    return chart, stage


def table_first_period_validation(
    chart: PlumbingChartResult,
    q_values: Sequence[complex],
    log_q_values: Sequence[complex],
    *,
    tolerance: float,
) -> PeriodMapValidation:
    """Promote the table-first forward check to the final period certificate."""

    if chart.period_map_region not in {"schottky-all-small", "two-method-overlap"}:
        raise ValueError("table-first certificate is not in the all-small-q region")
    if chart.period_max_residual > float(tolerance):
        raise ValueError(
            "table-first period residual exceeds the final numerical bar: "
            f"{chart.period_max_residual:.3e} > {float(tolerance):.3e}"
        )
    if chart.period_map_stability > float(tolerance):
        raise ValueError(
            "table-first Schottky tail estimate exceeds the final numerical bar: "
            f"{chart.period_map_stability:.3e} > {float(tolerance):.3e}"
        )
    symmetry_error = (
        math.nan
        if chart.collocation_symmetry_error is None
        else float(chart.collocation_symmetry_error)
    )
    return PeriodMapValidation(
        q=tuple(complex(value) for value in q_values),  # type: ignore[arg-type]
        log_q=tuple(complex(value) for value in log_q_values),  # type: ignore[arg-type]
        period_algorithm=chart.period_algorithm,
        refined=bool(chart.inverse_nfev > 1),
        fixed_q_residual=float(chart.period_max_residual),
        fixed_q_stability=float(chart.period_map_stability),
        final_residual=float(chart.period_max_residual),
        final_stability=float(chart.period_map_stability),
        seam_residual=math.nan,
        symmetry_error=symmetry_error,
        low_order=int(chart.word_length),
        high_order=int(chart.stability_word_length),
        validation_order=int(chart.stability_word_length),
        reinverse_success=True,
        reinverse_message=(
            "the table/leading all-small seed passed the direct adaptive Schottky certificate; "
            "no nonlinear re-inversion was used"
        ),
        reinverse_nfev=0,
        max_tau_shift=0.0,
        certified_error_bound=float(chart.period_map_stability),
        period_map_region=chart.period_map_region,
        overlap_residual=chart.period_overlap_residual,
        agreement_tolerance=float(tolerance),
    )


def _optional_float(row: Mapping[str, object], key: str) -> float | None:
    value = str(row.get(key, "")).strip()
    return None if value == "" else float(value)


def _cached_period_value(
    row: Mapping[str, object],
    *keys: str,
    default: object | None = None,
) -> object:
    """Return the first nonempty cached-period field among compatible schemas."""

    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip() != "":
            return value
    if default is not None:
        return default
    raise KeyError(f"cached period solution lacks all of {keys!r}")


def cached_period_solution_objects(
    source_row: Mapping[str, str],
    recovery_row: Mapping[str, object],
    *,
    tolerance: float,
) -> tuple[argparse.Namespace, PeriodMapValidation]:
    """Rehydrate a certified period chart without repeating Omega-to-q."""

    sample_index = int(source_row["sample_index"])
    if int(str(recovery_row.get("sample_index", "-1"))) != sample_index:
        raise ValueError("period-recovery cache sample index does not match the source row")
    source_node_id = str(source_row.get("rqmc_node_id", ""))
    recovered_node_id = str(recovery_row.get("rqmc_node_id", ""))
    if source_node_id and recovered_node_id and recovered_node_id != source_node_id:
        raise ValueError("period-recovery cache node id does not match the source row")
    if str(recovery_row.get("status", "")) != "ok":
        raise ValueError(
            "period-recovery cache has no accepted solution: "
            f"{recovery_row.get('error', 'unknown failure')}"
        )
    matrix = symplectic_matrix_from_csv_row(recovery_row)
    if matrix is None:
        raise ValueError("period-recovery cache lacks the exact Sp(4,Z) marking")
    topology = str(recovery_row.get("topology", ""))
    if topology not in {"theta", "glasses"}:
        raise ValueError(f"invalid cached plumbing topology {topology!r}")

    final_residual = float(
        _cached_period_value(
            recovery_row,
            "final_period_residual",
            "period_final_residual",
        )
    )
    final_stability = float(
        _cached_period_value(
            recovery_row,
            "final_period_map_step",
            "period_final_map_step",
        )
    )
    if final_residual > float(tolerance) or final_stability > float(tolerance):
        raise ValueError(
            "cached period solution exceeds the requested numerical bar: "
            f"residual={final_residual:.3e}, stability={final_stability:.3e}, "
            f"bar={float(tolerance):.3e}"
        )
    q_values = tuple(
        complex(
            str(
                _cached_period_value(
                    recovery_row,
                    f"final_q{index}",
                    f"q{index}",
                )
            )
        )
        for index in range(1, 4)
    )
    log_q_values = tuple(
        complex(
            str(
                _cached_period_value(
                    recovery_row,
                    f"final_log_q{index}",
                    f"log_q{index}",
                )
            )
        )
        for index in range(1, 4)
    )
    atlas_q_values = tuple(
        complex(str(recovery_row.get(f"atlas_q{index}", q_values[index - 1])))
        for index in range(1, 4)
    )
    atlas_log_q_values = tuple(
        complex(
            str(recovery_row.get(f"atlas_log_q{index}", log_q_values[index - 1]))
        )
        for index in range(1, 4)
    )
    final_coordinates = validate_genus2_plumbing_coordinates(
        topology,
        q_values,
        log_q_values=log_q_values,
    )
    atlas_coordinates = validate_genus2_plumbing_coordinates(
        topology,
        atlas_q_values,
        log_q_values=atlas_log_q_values,
    )
    q_values = final_coordinates.q_values
    log_q_values = final_coordinates.log_q_values
    atlas_q_values = atlas_coordinates.q_values
    atlas_log_q_values = atlas_coordinates.log_q_values
    certificate_algorithm = str(recovery_row["period_algorithm"])
    inferred_region = (
        "schottky-all-small"
        if "schottky" in certificate_algorithm
        else (
            "holomorphic-mixed-cusp"
            if certificate_algorithm == MULTIPRECISION_HOLOMORPHIC_ALGORITHM
            else "holomorphic-bulk"
        )
    )
    cached_region = (
        str(recovery_row.get("period_map_region", ""))
        or str(recovery_row.get("atlas_period_map_region", ""))
        or inferred_region
    )
    chart = argparse.Namespace(
        topology=topology,
        word=str(recovery_row.get("symplectic_word", "cached-period-recovery")),
        matrix=tuple(tuple(int(value) for value in row) for row in matrix),
        status=str(recovery_row.get("chart_status", "requires-recursion-order-study")),
        period_algorithm=str(recovery_row["atlas_period_algorithm"]),
        period_map_region=(
            str(recovery_row.get("atlas_period_map_region", "")) or cached_region
        ),
        plumbing_geometry_margin=float(
            recovery_row.get("atlas_plumbing_geometry_margin", "nan") or "nan"
        ),
        period_overlap_residual=_optional_float(
            recovery_row, "atlas_period_overlap_residual"
        ),
        inverse_seed_source=str(recovery_row.get("atlas_inverse_seed_source", "")),
        q=tuple(_complex_string(value) for value in atlas_q_values),
        log_q=tuple(_complex_string(value) for value in atlas_log_q_values),
        q_max=float(recovery_row.get("atlas_q_max", max(abs(value) for value in q_values))),
        period_max_residual=float(
            _cached_period_value(
                recovery_row,
                "atlas_period_residual",
                "period_residual",
            )
        ),
        period_map_stability=float(
            _cached_period_value(
                recovery_row,
                "atlas_period_map_stability",
                "period_map_stability",
            )
        ),
    )
    certificate = PeriodMapValidation(
        q=q_values,  # type: ignore[arg-type]
        log_q=log_q_values,  # type: ignore[arg-type]
        period_algorithm=certificate_algorithm,
        refined=str(
            _cached_period_value(
                recovery_row,
                "q_refined",
                "period_q_refined",
                default="False",
            )
        ).lower()
        in {"1", "true", "yes"},
        fixed_q_residual=float(
            _cached_period_value(
                recovery_row,
                "fixed_q_period_residual",
                "period_fixed_q_residual",
            )
        ),
        fixed_q_stability=float(
            _cached_period_value(
                recovery_row,
                "fixed_q_period_map_step",
                "period_fixed_q_map_step",
            )
        ),
        final_residual=final_residual,
        final_stability=final_stability,
        seam_residual=float(recovery_row.get("period_seam_residual", "nan") or "nan"),
        symmetry_error=float(recovery_row.get("period_symmetry_error", "nan") or "nan"),
        low_order=int(float(recovery_row.get("period_map_low_order", 0) or 0)),
        high_order=int(float(recovery_row.get("period_map_high_order", 0) or 0)),
        validation_order=int(
            float(recovery_row.get("period_map_validation_order", 0) or 0)
        ),
        reinverse_success=True,
        reinverse_message=str(
            _cached_period_value(
                recovery_row,
                "reinverse_message",
                "period_reinverse_message",
                default="cached certificate",
            )
        ),
        reinverse_nfev=int(
            float(
                _cached_period_value(
                    recovery_row,
                    "reinverse_nfev",
                    "period_reinverse_nfev",
                    default=0,
                )
            )
        ),
        max_tau_shift=float(
            _cached_period_value(
                recovery_row,
                "max_tau_shift",
                "period_reinverse_max_tau_shift",
                default=0.0,
            )
        ),
        validity_cell_id=str(recovery_row.get("period_validity_cell_id", "")) or None,
        certified_error_bound=(
            _optional_float(recovery_row, "period_certified_error_bound")
            or final_stability
        ),
        validity_reference_table_sha256=(
            str(recovery_row.get("period_validity_reference_table_sha256", "")) or None
        ),
        period_map_region=cached_region,
        overlap_residual=_optional_float(recovery_row, "period_overlap_residual"),
        agreement_tolerance=(
            _optional_float(recovery_row, "period_agreement_tolerance")
            or float(tolerance)
        ),
    )
    return chart, certificate

