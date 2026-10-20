#!/usr/bin/env python3
"""Regression checks for the sphere ``1->4`` 30-point campaign artifacts."""

from __future__ import annotations

from c1_paths import PACKAGE_ROOT, data_root, package_path, module_path

import hashlib
import json
from pathlib import Path

import numpy as np


BASE = data_root("tree_level") / "sphere_five_point_1to4" / "blind30_20260824"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def verified(name: str, manifest_name: str) -> dict[str, object]:
    path = BASE / name
    manifest = json.loads((BASE / manifest_name).read_text())
    require(sha256_file(path) == manifest["sha256"], f"stale manifest for {name}")
    return json.loads(path.read_text())


def check_extension() -> None:
    extension = verified(
        "worldsheet_extension_12point.json",
        "worldsheet_extension_12point_frozen.json",
    )
    expected = [0.19, 0.21, 0.23, 0.27, 0.29, 0.31, 0.33, 0.35, 0.37, 0.39, 0.44, 0.47]
    points = extension["points"]
    require([point["t"] for point in points] == expected, "extension design drifted")
    require(extension["matrix_model_information_used"] is False, "extension is not blind")
    require(len({point["seed"] for point in points}) == 12, "extension seeds overlap")
    require(
        max(abs(point["Q_imaginary_part"]) for point in points) < 2.0e-15,
        "extension has a non-negligible imaginary contamination",
    )


def check_merge_and_fit() -> None:
    scan = verified("worldsheet_scan_30point.json", "worldsheet_scan_30point_frozen.json")
    points = scan["points"]
    t = np.asarray([point["t"] for point in points], dtype=float)
    q = np.asarray([point["Q"] for point in points], dtype=float)
    require(len(points) == 30 and len(set(t)) == 30, "merged table is not 30-point")
    require(np.all(np.diff(t) > 0.0), "merged table is not increasing")
    require(np.sum(t <= 0.48) == 29, "wrong primary-point count")
    require(np.sum(np.asarray([point["scan_cohort"] for point in points]) == "new_extension") == 12, "wrong new cohort count")

    fit = verified(
        "worldsheet_quadratic_fit_30point_frozen.json",
        "worldsheet_quadratic_fit_30point_manifest.json",
    )
    require(fit["target_information_used"] is False, "fit is not target-free")
    require(fit["source_worldsheet_scan_sha256"] == sha256_file(BASE / "worldsheet_scan_30point.json"), "fit source hash is stale")
    primary = t <= 0.48
    design = np.column_stack((np.ones(np.sum(primary)), t[primary], t[primary] ** 2))
    coefficients, *_ = np.linalg.lstsq(design, q[primary], rcond=None)
    require(np.allclose(coefficients, fit["primary_fit"]["coefficients_in_t"], rtol=0.0, atol=2.0e-13), "primary fit is not reproducible")


def check_audit_and_comparison() -> None:
    audit = verified(
        "worldsheet_newpoint_block_audit_frozen.json",
        "worldsheet_newpoint_block_audit_manifest.json",
    )
    require(audit["matrix_model_information_used"] is False, "audit is not target-free")
    require([point["t"] for point in audit["points"]] == [0.31, 0.35, 0.44], "audit design drifted")

    comparison = json.loads((BASE / "matrix_comparison_30point.json").read_text())
    require(comparison["point_count"] == 30, "comparison lost points")
    require(comparison["primary_point_count"] == 29, "comparison primary count drifted")
    require(comparison["verified_worldsheet_fit"]["sha256"] == sha256_file(BASE / "worldsheet_quadratic_fit_30point_frozen.json"), "comparison fit hash is stale")
    new_qmc = comparison["qmc_only_pointwise_comparisons"]["new_extension_12point"]
    new_conservative = comparison["conservative_pointwise_comparisons"]["new_extension_12point"]
    require(abs(new_qmc["chi_squared"] - 8.01238766690237) < 1.0e-10, "new-point statistic drifted")
    require(new_conservative["chi_squared"] < new_qmc["chi_squared"], "audit did not enlarge conservative errors")
    require((BASE / "amplitude_comparison_30point.png").stat().st_size > 100_000, "comparison plot is missing")


def check_code_separation() -> None:
    for name in (
        "sphere_five_point_30point_worldsheet_extension.py",
        "sphere_five_point_30point_worldsheet_fit.py",
        "sphere_five_point_30point_audit_summary.py",
    ):
        source = module_path(name).read_text()
        require("MATRIX_COEFFICIENTS_IN_T" not in source, f"matrix coefficients leaked into {name}")
    comparison_source = module_path("sphere_five_point_30point_matrix_comparison").read_text()
    require("MATRIX_COEFFICIENTS_IN_T" in comparison_source, "downstream comparator lacks its target")


def main() -> None:
    check_extension()
    check_merge_and_fit()
    check_audit_and_comparison()
    check_code_separation()
    print("PASS sphere 1->4 blind 30-point extension, fit, audit, and comparison")


if __name__ == "__main__":
    main()
