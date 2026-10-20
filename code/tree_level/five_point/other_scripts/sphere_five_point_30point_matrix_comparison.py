#!/usr/bin/env python3
"""Compare the independently frozen 30-point sphere ``1->4`` fit downstream."""

from __future__ import annotations

from c1_paths import PACKAGE_ROOT, data_root, package_path, module_path

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from sphere_five_point_30point_worldsheet_fit import FIT_STATUS, SCAN_STATUS
from sphere_five_point_30point_audit_summary import STATUS as AUDIT_STATUS



MATRIX_COEFFICIENTS_IN_T = np.asarray([2.0, -12.0, 16.0])


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify(path: Path, manifest_path: Path, status: str) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    manifest = json.loads(manifest_path.read_text())
    if sha256_file(path) != manifest.get("sha256"):
        raise RuntimeError(f"artifact does not match its freeze manifest: {path}")
    if payload.get("status") != status or manifest.get("status") != status:
        raise RuntimeError(f"artifact has the wrong frozen status: {path}")
    return payload


def reduced_amplitude(t: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    return coefficients[0] + coefficients[1] * t + coefficients[2] * t**2


def normalized_amplitude(t: np.ndarray, reduced: np.ndarray) -> np.ndarray:
    return -4.0 * t**5 * reduced


def _plot_error(point: dict[str, Any]) -> float:
    qmc = float(point["Q_standard_error"])
    if "Q_block_order_4_6_8" not in point:
        return qmc
    values = np.asarray(point["Q_block_order_4_6_8"], dtype=float)
    block_spread = float(np.max(np.abs(values - float(point["Q"]))))
    return max(qmc, block_spread)


def _metrics(
    difference: np.ndarray,
    sigma: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float | int]:
    selected = difference[mask]
    pulls = selected / sigma[mask]
    return {
        "point_count": int(np.sum(mask)),
        "chi_squared": float(pulls @ pulls),
        "degrees_of_freedom": int(np.sum(mask)),
        "maximum_absolute_pull": float(np.max(np.abs(pulls))),
        "maximum_absolute_Q4_difference": float(np.max(np.abs(selected))),
        "rms_Q4_difference": float(math.sqrt(np.mean(selected**2))),
    }


def _coefficient_comparison(fit: dict[str, Any]) -> dict[str, Any]:
    coefficients = np.asarray(fit["coefficients_in_t"], dtype=float)
    errors = np.asarray(fit["coefficient_standard_errors"], dtype=float)
    covariance = np.asarray(fit["coefficient_covariance"], dtype=float)
    difference = coefficients - MATRIX_COEFFICIENTS_IN_T
    return {
        "worldsheet_coefficients_in_t": [float(x) for x in coefficients],
        "matrix_model_coefficients_in_t": [
            float(x) for x in MATRIX_COEFFICIENTS_IN_T
        ],
        "fit_minus_matrix_coefficients": [float(x) for x in difference],
        "coefficient_relative_errors": [
            float(abs(x / y))
            for x, y in zip(difference, MATRIX_COEFFICIENTS_IN_T, strict=True)
        ],
        "marginal_difference_in_fit_standard_errors": [
            float(x) for x in difference / errors
        ],
        "joint_coefficient_chi_squared": float(
            difference @ np.linalg.solve(covariance, difference)
        ),
        "joint_coefficient_degrees_of_freedom": 3,
    }


def _real_roots(coefficients: np.ndarray) -> list[float]:
    roots = np.roots([coefficients[2], coefficients[1], coefficients[0]])
    return sorted(float(root.real) for root in roots if abs(root.imag) < 1.0e-10)


def compare(
    scan_path: Path,
    scan_manifest_path: Path,
    fit_path: Path,
    fit_manifest_path: Path,
    audit_path: Path,
    audit_manifest_path: Path,
    output_path: Path,
    figure_path: Path,
) -> dict[str, Any]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    scan = _verify(scan_path, scan_manifest_path, SCAN_STATUS)
    fit = _verify(fit_path, fit_manifest_path, FIT_STATUS)
    scan_hash = sha256_file(scan_path)
    if fit.get("source_worldsheet_scan_sha256") != scan_hash:
        raise RuntimeError("frozen fit was not derived from the verified scan")
    if fit.get("target_information_used") is not False:
        raise RuntimeError("worldsheet fit does not certify target-free production")
    audit = _verify(audit_path, audit_manifest_path, AUDIT_STATUS)
    if audit.get("matrix_model_information_used") is not False:
        raise RuntimeError("worldsheet audit does not certify target-free production")
    if audit.get("verified_production_scan", {}).get("sha256") != scan_hash:
        raise RuntimeError("worldsheet audit does not refer to the verified scan")

    points = scan["points"]
    t = np.asarray([point["t"] for point in points], dtype=float)
    q_worldsheet = np.asarray([point["Q"] for point in points], dtype=float)
    qmc_sigma = np.asarray([point["Q_standard_error"] for point in points], dtype=float)
    conservative_sigma = np.asarray([_plot_error(point) for point in points], dtype=float)
    cohort = np.asarray([point["scan_cohort"] for point in points])
    role = np.asarray([point["fit_role"] for point in points])
    known = cohort == "known_base"
    new = cohort == "new_extension"
    primary = role == "primary"
    diagnostic = ~primary
    if (int(np.sum(known)), int(np.sum(new)), int(np.sum(primary)), int(np.sum(diagnostic))) != (18, 12, 29, 1):
        raise RuntimeError("comparison cohort counts do not match the frozen design")
    index_by_t = {round(float(value), 12): index for index, value in enumerate(t)}
    audit_indices = []
    audit_q = []
    audit_sigma = []
    for record in audit["points"]:
        index = index_by_t[round(float(record["t"]), 12)]
        audit_indices.append(index)
        audit_q.append(float(record["audit_Q4"]))
        audit_sigma.append(float(record["audit_Q4_qmc_standard_error"]))
        conservative_sigma[index] = max(
            conservative_sigma[index], float(record["conservative_standard_error"])
        )
    audit_indices_array = np.asarray(audit_indices, dtype=int)
    audit_q_array = np.asarray(audit_q, dtype=float)
    audit_sigma_array = np.asarray(audit_sigma, dtype=float)

    q_matrix = reduced_amplitude(t, MATRIX_COEFFICIENTS_IN_T)
    difference = q_worldsheet - q_matrix
    fit_coefficients = np.asarray(fit["primary_fit"]["coefficients_in_t"], dtype=float)
    amplitude_worldsheet = normalized_amplitude(t, q_worldsheet)
    amplitude_matrix = normalized_amplitude(t, q_matrix)

    dense_t = np.linspace(float(t.min()) - 0.01, 0.50, 700)
    dense_matrix = reduced_amplitude(dense_t, MATRIX_COEFFICIENTS_IN_T)
    dense_fit = reduced_amplitude(dense_t, fit_coefficients)
    figure, axes = plt.subplots(3, 1, figsize=(8.7, 10.0), sharex=True)
    axes[0].plot(dense_t, dense_matrix, color="#222222", linewidth=1.8, label="matrix model")
    axes[0].plot(dense_t, dense_fit, color="#315b9d", linewidth=1.5, linestyle="--", label="frozen worldsheet quadratic fit")
    styles = (
        (known & primary, "o", "#d36c45", "17 known primary points"),
        (new, "D", "#26876b", "12 new points"),
    )
    for mask, marker, color, label in styles:
        axes[0].errorbar(t[mask], q_worldsheet[mask], yerr=conservative_sigma[mask], fmt=marker, markersize=4.2, capsize=2.0, color=color, ecolor=color, label=label)
        axes[1].errorbar(t[mask], amplitude_worldsheet[mask], yerr=4.0 * t[mask] ** 5 * conservative_sigma[mask], fmt=marker, markersize=4.2, capsize=2.0, color=color, ecolor=color)
        axes[2].errorbar(t[mask], difference[mask], yerr=conservative_sigma[mask], fmt=marker, markersize=4.2, capsize=2.0, color=color, ecolor=color)
    axes[0].errorbar(t[diagnostic], q_worldsheet[diagnostic], yerr=conservative_sigma[diagnostic], fmt="D", markerfacecolor="none", markeredgecolor="#9467bd", ecolor="#9467bd", capsize=2.0, label="t=0.49 diagnostic")
    axes[1].errorbar(t[diagnostic], amplitude_worldsheet[diagnostic], yerr=4.0 * t[diagnostic] ** 5 * conservative_sigma[diagnostic], fmt="D", markerfacecolor="none", markeredgecolor="#9467bd", ecolor="#9467bd", capsize=2.0)
    axes[2].errorbar(t[diagnostic], difference[diagnostic], yerr=conservative_sigma[diagnostic], fmt="D", markerfacecolor="none", markeredgecolor="#9467bd", ecolor="#9467bd", capsize=2.0)
    axes[0].errorbar(t[audit_indices_array], audit_q_array, yerr=audit_sigma_array, fmt="s", markersize=4.0, capsize=2.0, color="#315b9d", ecolor="#315b9d", label="3 block-order audits")
    axes[1].errorbar(t[audit_indices_array], normalized_amplitude(t[audit_indices_array], audit_q_array), yerr=4.0 * t[audit_indices_array] ** 5 * audit_sigma_array, fmt="s", markersize=4.0, capsize=2.0, color="#315b9d", ecolor="#315b9d")
    axes[2].errorbar(t[audit_indices_array], audit_q_array - q_matrix[audit_indices_array], yerr=audit_sigma_array, fmt="s", markersize=4.0, capsize=2.0, color="#315b9d", ecolor="#315b9d")
    axes[1].plot(dense_t, normalized_amplitude(dense_t, dense_matrix), color="#222222", linewidth=1.8)
    axes[1].plot(dense_t, normalized_amplitude(dense_t, dense_fit), color="#315b9d", linewidth=1.5, linestyle="--")
    for axis in axes:
        axis.axhline(0.0, color="#777777", linewidth=0.7)
        axis.axvline(0.4, color="#e76f9a", linestyle=":", linewidth=1.0)
        axis.axvline(0.5, color="#9467bd", linestyle=":", linewidth=1.0)
        axis.grid(alpha=0.18)
    axes[0].set_ylabel(r"$Q_4(i t)$")
    axes[1].set_ylabel(r"$\mu^3\mathcal{A}^{\rm tree}_{1\to4}(i t)$")
    axes[2].set_ylabel(r"$Q_4^{\rm WS}-Q_4^{\rm MM}$")
    axes[2].set_xlabel(r"$t$ in $\omega=i t$")
    axes[0].legend(frameon=False, fontsize=8.5, ncol=2)
    figure.suptitle(r"Sphere $1\to4$: independently frozen 30-point worldsheet fit vs. matrix model")
    figure.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_path, dpi=220)
    plt.close(figure)

    point_records = []
    for index, point in enumerate(points):
        point_records.append(
            {
                "t": float(t[index]),
                "scan_cohort": str(cohort[index]),
                "fit_role": str(role[index]),
                "worldsheet_Q4": float(q_worldsheet[index]),
                "worldsheet_Q4_qmc_standard_error": float(qmc_sigma[index]),
                "worldsheet_Q4_conservative_plot_error": float(conservative_sigma[index]),
                "matrix_model_Q4": float(q_matrix[index]),
                "Q4_difference": float(difference[index]),
                "Q4_qmc_pull": float(difference[index] / qmc_sigma[index]),
                "Q4_conservative_pull": float(difference[index] / conservative_sigma[index]),
                "worldsheet_amplitude": float(amplitude_worldsheet[index]),
                "matrix_model_amplitude": float(amplitude_matrix[index]),
            }
        )

    masks = {
        "primary_29point": primary,
        "known_primary_17point": known & primary,
        "new_extension_12point": new,
        "all_30point_including_diagnostic": np.ones(len(t), dtype=bool),
    }
    result: dict[str, Any] = {
        "comparison_order": (
            "the 30-point worldsheet scan and target-free quadratic fits were "
            "hash-frozen before this program evaluated the matrix-model coefficients"
        ),
        "verified_worldsheet_scan": {
            "path": str(scan_path.resolve()),
            "sha256": scan_hash,
        },
        "verified_worldsheet_fit": {
            "path": str(fit_path.resolve()),
            "sha256": sha256_file(fit_path),
        },
        "verified_worldsheet_block_audit": {
            "path": str(audit_path.resolve()),
            "sha256": sha256_file(audit_path),
            "timing": "post-comparison convergence sensitivity; primary fit unchanged",
        },
        "point_count": 30,
        "primary_point_count": 29,
        "matrix_model_reduced_amplitude": "Q_4(i*t)=2-12*t+16*t^2",
        "matrix_model_normalized_amplitude": "mu^3*A_tree(i*t)=-4*t^5*Q_4(i*t)",
        "worldsheet_primary_quadratic_fit": fit["primary_fit"],
        "coefficient_comparisons": {
            "primary_29point": _coefficient_comparison(fit["primary_fit"]),
            "known_primary_17point": _coefficient_comparison(fit["known_base_primary_fit"]),
            "new_extension_12point": _coefficient_comparison(fit["new_extension_fit"]),
            "all_30point_sensitivity": _coefficient_comparison(fit["all_30point_sensitivity_fit"]),
        },
        "worldsheet_fit_real_roots": _real_roots(fit_coefficients),
        "matrix_model_real_roots": _real_roots(MATRIX_COEFFICIENTS_IN_T),
        "qmc_only_pointwise_comparisons": {
            key: _metrics(difference, qmc_sigma, mask) for key, mask in masks.items()
        },
        "conservative_pointwise_comparisons": {
            key: _metrics(difference, conservative_sigma, mask) for key, mask in masks.items()
        },
        "newpoint_block_order_audits": audit["points"],
        "points": point_records,
        "figure": str(figure_path.resolve()),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    run_dir = data_root("tree_level") / "sphere_five_point_1to4" / "blind30_20260824"
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan", type=Path, default=run_dir / "worldsheet_scan_30point.json")
    parser.add_argument("--scan-manifest", type=Path, default=run_dir / "worldsheet_scan_30point_frozen.json")
    parser.add_argument("--fit", type=Path, default=run_dir / "worldsheet_quadratic_fit_30point_frozen.json")
    parser.add_argument("--fit-manifest", type=Path, default=run_dir / "worldsheet_quadratic_fit_30point_manifest.json")
    parser.add_argument("--audit", type=Path, default=run_dir / "worldsheet_newpoint_block_audit_frozen.json")
    parser.add_argument("--audit-manifest", type=Path, default=run_dir / "worldsheet_newpoint_block_audit_manifest.json")
    parser.add_argument("--output", type=Path, default=run_dir / "matrix_comparison_30point.json")
    parser.add_argument("--figure", type=Path, default=run_dir / "amplitude_comparison_30point.png")
    arguments = parser.parse_args()
    result = compare(arguments.scan, arguments.scan_manifest, arguments.fit, arguments.fit_manifest, arguments.audit, arguments.audit_manifest, arguments.output, arguments.figure)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
