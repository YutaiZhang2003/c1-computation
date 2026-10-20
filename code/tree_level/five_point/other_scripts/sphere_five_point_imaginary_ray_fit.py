#!/usr/bin/env python3
"""Freeze a target-free quadratic fit to the sphere 1->4 imaginary-ray data.

This module is deliberately ignorant of every matrix-model coefficient.  It
reads only the ``points`` array of the worldsheet audit, fits

    Q_4(i t) = a + b t + c t^2,

and records the source hash, selected points, residuals, and covariance.  The
near-second-wall point at t=0.49 is excluded from the primary fit and retained
as a declared sensitivity diagnostic.
"""

from __future__ import annotations

from c1_paths import PACKAGE_ROOT, data_root, package_path, module_path

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


POINTS_STATUS = "imaginary_ray_worldsheet_points_frozen_for_target_free_fit"
FIT_STATUS = "imaginary_ray_worldsheet_quadratic_fit_frozen_for_separate_comparison"
MAXIMUM_PRIMARY_T = 0.48


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _plain_point(point: dict[str, Any]) -> dict[str, Any]:
    """Copy only worldsheet data and contour metadata used by this fit."""

    copied: dict[str, Any] = {
        "t": float(point["t"]),
        "Q": float(point["Q"]),
        "Q_standard_error": float(point["Q_standard_error"]),
        "block_order": int(point["block_order"]),
        "sobol_power": int(point["sobol_power"]),
        "contour": str(point["contour"]),
    }
    if "residue_status" in point:
        copied["residue_status"] = str(point["residue_status"])
    if "Q_block_order_4_6_8" in point:
        copied["Q_block_order_4_6_8"] = [
            float(value) for value in point["Q_block_order_4_6_8"]
        ]
    return copied


def quadratic_fit(points: list[dict[str, Any]]) -> dict[str, Any]:
    """Return an unweighted least-squares quadratic and residual covariance.

    The fit is intentionally unweighted because the stored pointwise errors
    are QMC errors, while block truncation and other discretization effects
    are correlated and are not represented by a full covariance matrix.
    """

    if len(points) < 4:
        raise ValueError("at least four points are required for an audited quadratic fit")
    t = np.asarray([point["t"] for point in points], dtype=float)
    q = np.asarray([point["Q"] for point in points], dtype=float)
    design = np.column_stack((np.ones_like(t), t, t**2))
    coefficients, *_ = np.linalg.lstsq(design, q, rcond=None)
    residuals = q - design @ coefficients
    degrees_of_freedom = len(points) - 3
    residual_variance = float(residuals @ residuals / degrees_of_freedom)
    covariance = np.linalg.inv(design.T @ design) * residual_variance
    coefficient_errors = np.sqrt(np.diag(covariance))
    return {
        "number_of_points": len(points),
        "degree": 2,
        "weighting": "unweighted ordinary least squares",
        "weighting_reason": (
            "stored standard errors are QMC-only and do not form the full "
            "correlated block/discretization covariance"
        ),
        "coefficients_in_t": [float(value) for value in coefficients],
        "coefficient_standard_errors_from_residual_scatter": [
            float(value) for value in coefficient_errors
        ],
        "coefficient_covariance_from_residual_scatter": [
            [float(value) for value in row] for row in covariance
        ],
        "residuals": [float(value) for value in residuals],
        "maximum_absolute_residual": float(np.max(np.abs(residuals))),
        "rms_residual": float(np.sqrt(np.mean(residuals**2))),
    }


def freeze_points(audit_path: Path, output_path: Path) -> dict[str, Any]:
    """Mechanically extract a points-only input from the historical audit."""

    audit = json.loads(audit_path.read_text())
    if "points" not in audit:
        raise ValueError("worldsheet audit has no points array")
    points = [_plain_point(point) for point in audit["points"]]
    result: dict[str, Any] = {
        "status": POINTS_STATUS,
        "program": Path(__file__).name,
        "lineage_source_worldsheet_audit": audit_path.name,
        "lineage_source_worldsheet_audit_sha256": sha256(audit_path),
        "extraction": "mechanical copy of points and contour metadata only",
        "target_information_present": False,
        "points": points,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def freeze_fit(input_path: Path, output_path: Path) -> dict[str, Any]:
    source = json.loads(input_path.read_text())
    if source.get("status") != POINTS_STATUS:
        raise ValueError("fit requires the frozen points-only imaginary-ray table")
    if source.get("target_information_present") is not False:
        raise ValueError("points-only fit input must explicitly exclude target information")
    points = [_plain_point(point) for point in source["points"]]
    t_values = [point["t"] for point in points]
    if t_values != sorted(t_values) or len(set(t_values)) != len(t_values):
        raise ValueError("worldsheet t values must be unique and increasing")

    primary_points = [point for point in points if point["t"] <= MAXIMUM_PRIMARY_T]
    diagnostic_points = [point for point in points if point["t"] > MAXIMUM_PRIMARY_T]
    if not diagnostic_points:
        raise ValueError("expected at least one declared near-wall diagnostic point")

    result: dict[str, Any] = {
        "status": FIT_STATUS,
        "program": Path(__file__).name,
        "source_worldsheet_points": input_path.name,
        "source_worldsheet_points_sha256": sha256(input_path),
        "lineage_source_worldsheet_audit": source["lineage_source_worldsheet_audit"],
        "lineage_source_worldsheet_audit_sha256": source[
            "lineage_source_worldsheet_audit_sha256"
        ],
        "target_information_used": False,
        "fit_ansatz": "Q_4(i*t)=a+b*t+c*t^2",
        "fit_degree_assumption": (
            "degree two is an explicit analytic ansatz; it is not inferred "
            "from the matrix-model coefficients"
        ),
        "primary_selection": f"all points with t <= {MAXIMUM_PRIMARY_T}",
        "primary_points": primary_points,
        "diagnostic_points_excluded_from_primary_fit": diagnostic_points,
        "primary_fit": quadratic_fit(primary_points),
        "all_points_sensitivity_fit": quadratic_fit(points),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    base = data_root("tree_level") / "sphere_five_point_1to4"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audit-input",
        type=Path,
        default=base / "worldsheet_imaginary_ray_audit_20260823.json",
    )
    parser.add_argument(
        "--points-output",
        type=Path,
        default=base / "worldsheet_imaginary_ray_points_frozen.json",
    )
    parser.add_argument(
        "--fit-output",
        type=Path,
        default=base / "worldsheet_imaginary_ray_fit_frozen.json",
    )
    arguments = parser.parse_args()
    freeze_points(arguments.audit_input, arguments.points_output)
    print(json.dumps(freeze_fit(arguments.points_output, arguments.fit_output), indent=2))


if __name__ == "__main__":
    main()
