#!/usr/bin/env python3
"""Merge and fit the 30-point sphere ``1->4`` worldsheet campaign.

No matrix-model coefficient appears in this module.  The historical frozen
points and the new twelve-point extension are hash-verified, merged without
discarding a datum, and fitted with the pre-existing degree-two ansatz before
any external comparison is allowed.
"""

from __future__ import annotations

from c1_paths import PACKAGE_ROOT, data_root, package_path, module_path

import argparse
import hashlib
import json
import math
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from sphere_five_point_30point_worldsheet_extension import STATUS as EXTENSION_STATUS
from sphere_five_point_imaginary_ray_fit import FIT_STATUS as HISTORICAL_FIT_STATUS
from sphere_five_point_imaginary_ray_fit import POINTS_STATUS as HISTORICAL_POINTS_STATUS


SCAN_STATUS = "sphere_1to4_worldsheet_30point_merged_before_external_comparison"
POINTS_STATUS = "sphere_1to4_worldsheet_30point_points_frozen_for_target_free_fit"
FIT_STATUS = "sphere_1to4_worldsheet_30point_quadratic_fit_frozen_for_separate_comparison"
MAXIMUM_PRIMARY_T = 0.48


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _verify_extension(path: Path, manifest_path: Path) -> tuple[dict[str, Any], str]:
    payload = json.loads(path.read_text())
    manifest = json.loads(manifest_path.read_text())
    digest = sha256_file(path)
    if digest != manifest.get("sha256"):
        raise RuntimeError("extension does not match its freeze manifest")
    if payload.get("status") != EXTENSION_STATUS or manifest.get("status") != EXTENSION_STATUS:
        raise RuntimeError("extension has the wrong frozen status")
    if payload.get("matrix_model_information_used") is not False:
        raise RuntimeError("extension does not certify target-free production")
    if len(payload.get("points", [])) != 12 or manifest.get("point_count") != 12:
        raise RuntimeError("extension must contain exactly twelve points")
    return payload, digest


def _verify_historical(
    points_path: Path, fit_path: Path
) -> tuple[dict[str, Any], str, str]:
    points = json.loads(points_path.read_text())
    fit = json.loads(fit_path.read_text())
    points_digest = sha256_file(points_path)
    fit_digest = sha256_file(fit_path)
    if points.get("status") != HISTORICAL_POINTS_STATUS:
        raise RuntimeError("historical source is not the frozen points-only table")
    if fit.get("status") != HISTORICAL_FIT_STATUS:
        raise RuntimeError("historical fit has the wrong frozen status")
    if fit.get("source_worldsheet_points_sha256") != points_digest:
        raise RuntimeError("historical points changed after their original fit")
    if points.get("target_information_present") is not False:
        raise RuntimeError("historical points contain external target information")
    if fit.get("target_information_used") is not False:
        raise RuntimeError("historical fit used external target information")
    if len(points.get("points", [])) != 18:
        raise RuntimeError("historical table must contain eighteen points")
    return points, points_digest, fit_digest


def _plain_point(point: dict[str, Any], cohort: str) -> dict[str, Any]:
    copied: dict[str, Any] = {
        "t": float(point["t"]),
        "Q": float(point["Q"]),
        "Q_standard_error": float(point["Q_standard_error"]),
        "block_order": int(point["block_order"]),
        "sobol_power": int(point["sobol_power"]),
        "contour": str(point["contour"]),
        "scan_cohort": cohort,
        "fit_role": (
            "primary" if float(point["t"]) <= MAXIMUM_PRIMARY_T else "near-wall diagnostic"
        ),
    }
    optional = (
        "residue_status",
        "Q_block_order_4_6_8",
        "Q_imaginary_part",
        "raw_integral",
        "raw_standard_error",
        "replicate_estimates",
        "momentum_order",
        "replicates",
        "seed",
    )
    for key in optional:
        if key in point:
            copied[key] = point[key]
    return copied


def merge_and_freeze(
    historical_points_path: Path,
    historical_fit_path: Path,
    extension_path: Path,
    extension_manifest_path: Path,
    output_path: Path,
    manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    historical, historical_hash, historical_fit_hash = _verify_historical(
        historical_points_path, historical_fit_path
    )
    extension, extension_hash = _verify_extension(extension_path, extension_manifest_path)
    points = [
        *(_plain_point(point, "known_base") for point in historical["points"]),
        *(_plain_point(point, "new_extension") for point in extension["points"]),
    ]
    points.sort(key=lambda point: point["t"])
    t_values = [point["t"] for point in points]
    if len(points) != 30 or len(set(t_values)) != 30:
        raise RuntimeError("merged design must contain thirty distinct points")
    if t_values != sorted(t_values):
        raise RuntimeError("merged t values are not increasing")
    diagnostics = [point for point in points if point["fit_role"] != "primary"]
    if len(diagnostics) != 1 or not math.isclose(diagnostics[0]["t"], 0.49):
        raise RuntimeError("exactly the historical t=0.49 diagnostic must be excluded")
    payload: dict[str, Any] = {
        "status": SCAN_STATUS,
        "program": Path(__file__).name,
        "calculation": "genus-zero equal-energy sphere 1->4 amplitude",
        "matrix_model_information_used": False,
        "normalization": "Q_4=I_5/(16*pi^2*omega^5); mu^3*A=4*i*omega^5*Q_4",
        "point_count": 30,
        "primary_point_count": 29,
        "diagnostic_point_count": 1,
        "design": {
            "known_point_count": 18,
            "new_point_count": 12,
            "primary_selection": f"all points with t <= {MAXIMUM_PRIMARY_T}",
            "t_values": t_values,
        },
        "provenance": {
            "historical_points": str(historical_points_path.resolve()),
            "historical_points_sha256": historical_hash,
            "historical_fit": str(historical_fit_path.resolve()),
            "historical_fit_sha256": historical_fit_hash,
            "extension": str(extension_path.resolve()),
            "extension_sha256": extension_hash,
            "extension_manifest": str(extension_manifest_path.resolve()),
            "matrix_model_information_used": False,
        },
        "points": points,
    }
    write_json(output_path, payload)
    manifest = {
        "status": SCAN_STATUS,
        "artifact": str(output_path.resolve()),
        "sha256": sha256_file(output_path),
        "frozen_on": date.today().isoformat(),
        "point_count": 30,
        "primary_point_count": 29,
        "t_values": t_values,
        "matrix_model_information_used": False,
    }
    write_json(manifest_path, manifest)
    return payload, manifest


def _fit(t: np.ndarray, q: np.ndarray, sigma: np.ndarray | None) -> dict[str, Any]:
    design = np.column_stack((np.ones_like(t), t, t**2))
    if sigma is None:
        coefficients, *_ = np.linalg.lstsq(design, q, rcond=None)
        residuals = q - design @ coefficients
        residual_variance = float(residuals @ residuals / (len(t) - 3))
        covariance = residual_variance * np.linalg.inv(design.T @ design)
        chi_squared = None
        weighting = "unweighted ordinary least squares"
    else:
        inverse_variance = 1.0 / sigma**2
        normal = design.T @ (inverse_variance[:, None] * design)
        covariance = np.linalg.inv(normal)
        coefficients = covariance @ design.T @ (inverse_variance * q)
        residuals = q - design @ coefficients
        chi_squared = float(np.sum((residuals / sigma) ** 2))
        weighting = "diagonal QMC inverse-variance sensitivity fit"
    return {
        "number_of_points": len(t),
        "degree": 2,
        "weighting": weighting,
        "coefficients_in_t": [float(value) for value in coefficients],
        "coefficient_standard_errors": [
            float(value) for value in np.sqrt(np.diag(covariance))
        ],
        "coefficient_covariance": [
            [float(value) for value in row] for row in covariance
        ],
        "residuals": [float(value) for value in residuals],
        "chi_squared": chi_squared,
        "degrees_of_freedom": len(t) - 3,
        "maximum_absolute_residual": float(np.max(np.abs(residuals))),
        "rms_residual": float(math.sqrt(np.mean(residuals**2))),
    }


def freeze_points_and_fit(
    scan_path: Path,
    scan_manifest_path: Path,
    points_path: Path,
    fit_path: Path,
    fit_manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    scan = json.loads(scan_path.read_text())
    manifest = json.loads(scan_manifest_path.read_text())
    scan_hash = sha256_file(scan_path)
    if scan_hash != manifest.get("sha256"):
        raise RuntimeError("merged scan does not match its freeze manifest")
    if scan.get("status") != SCAN_STATUS or manifest.get("status") != SCAN_STATUS:
        raise RuntimeError("merged scan has the wrong frozen status")
    if scan.get("matrix_model_information_used") is not False:
        raise RuntimeError("merged scan does not certify target-free production")
    points_payload = {
        "status": POINTS_STATUS,
        "program": Path(__file__).name,
        "source_worldsheet_scan": str(scan_path.resolve()),
        "source_worldsheet_scan_sha256": scan_hash,
        "source_freeze_manifest": str(scan_manifest_path.resolve()),
        "source_freeze_manifest_sha256": sha256_file(scan_manifest_path),
        "target_information_present": False,
        "point_count": 30,
        "points": scan["points"],
    }
    write_json(points_path, points_payload)

    points = points_payload["points"]
    t = np.asarray([point["t"] for point in points], dtype=float)
    q = np.asarray([point["Q"] for point in points], dtype=float)
    sigma = np.asarray([point["Q_standard_error"] for point in points], dtype=float)
    cohort = np.asarray([point["scan_cohort"] for point in points])
    primary = t <= MAXIMUM_PRIMARY_T
    known = cohort == "known_base"
    new = cohort == "new_extension"
    if int(np.sum(primary)) != 29 or int(np.sum(known & primary)) != 17 or int(np.sum(new)) != 12:
        raise RuntimeError("merged fit cohort counts do not match the design")
    fit_payload: dict[str, Any] = {
        "status": FIT_STATUS,
        "program": Path(__file__).name,
        "source_worldsheet_points": str(points_path.resolve()),
        "source_worldsheet_points_sha256": sha256_file(points_path),
        "source_worldsheet_scan": str(scan_path.resolve()),
        "source_worldsheet_scan_sha256": scan_hash,
        "target_information_used": False,
        "fit_ansatz": "Q_4(i*t)=a+b*t+c*t^2",
        "fit_degree_assumption": "the pre-existing sphere 1->4 quadratic design",
        "primary_selection": f"all points with t <= {MAXIMUM_PRIMARY_T}",
        "primary_fit": _fit(t[primary], q[primary], None),
        "known_base_primary_fit": _fit(t[known & primary], q[known & primary], None),
        "new_extension_fit": _fit(t[new], q[new], None),
        "all_30point_sensitivity_fit": _fit(t, q, None),
        "qmc_weighted_primary_sensitivity_fit": _fit(
            t[primary], q[primary], sigma[primary]
        ),
        "diagnostic_points_excluded_from_primary_fit": [
            point for point in points if point["fit_role"] != "primary"
        ],
    }
    write_json(fit_path, fit_payload)
    fit_manifest = {
        "status": FIT_STATUS,
        "artifact": str(fit_path.resolve()),
        "sha256": sha256_file(fit_path),
        "frozen_on": date.today().isoformat(),
        "point_count": 30,
        "primary_point_count": 29,
        "target_information_used": False,
    }
    write_json(fit_manifest_path, fit_manifest)
    return fit_payload, fit_manifest


def main() -> None:
    base = data_root("tree_level") / "sphere_five_point_1to4"
    run_dir = base / "blind30_20260824"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--historical-points",
        type=Path,
        default=base / "worldsheet_imaginary_ray_points_frozen.json",
    )
    parser.add_argument(
        "--historical-fit",
        type=Path,
        default=base / "worldsheet_imaginary_ray_fit_frozen.json",
    )
    parser.add_argument(
        "--extension",
        type=Path,
        default=run_dir / "worldsheet_extension_12point.json",
    )
    parser.add_argument(
        "--extension-manifest",
        type=Path,
        default=run_dir / "worldsheet_extension_12point_frozen.json",
    )
    parser.add_argument(
        "--scan-output",
        type=Path,
        default=run_dir / "worldsheet_scan_30point.json",
    )
    parser.add_argument(
        "--scan-manifest",
        type=Path,
        default=run_dir / "worldsheet_scan_30point_frozen.json",
    )
    parser.add_argument(
        "--points-output",
        type=Path,
        default=run_dir / "worldsheet_points_30point_frozen.json",
    )
    parser.add_argument(
        "--fit-output",
        type=Path,
        default=run_dir / "worldsheet_quadratic_fit_30point_frozen.json",
    )
    parser.add_argument(
        "--fit-manifest",
        type=Path,
        default=run_dir / "worldsheet_quadratic_fit_30point_manifest.json",
    )
    arguments = parser.parse_args()
    merge_and_freeze(
        arguments.historical_points,
        arguments.historical_fit,
        arguments.extension,
        arguments.extension_manifest,
        arguments.scan_output,
        arguments.scan_manifest,
    )
    fit, _ = freeze_points_and_fit(
        arguments.scan_output,
        arguments.scan_manifest,
        arguments.points_output,
        arguments.fit_output,
        arguments.fit_manifest,
    )
    print(json.dumps(fit, indent=2))


if __name__ == "__main__":
    main()
