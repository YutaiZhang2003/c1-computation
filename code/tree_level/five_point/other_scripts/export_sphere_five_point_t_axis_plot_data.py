#!/usr/bin/env python3
"""Export the frozen sphere 1->4 points for the t-axis amplitude figure."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


from c1_paths import PACKAGE_ROOT as ROOT
RUN_DIR = (
    ROOT
    / "data set"
    / "tree_level"
    / "sphere_five_point_1to4"
    / "blind30_20260824"
)
FIGURE_DIR = ROOT / "Machine note" / "tree_level" / "figures"
SCAN_STATUS = "sphere_1to4_worldsheet_30point_merged_before_external_comparison"
AUDIT_STATUS = "sphere_1to4_worldsheet_newpoint_block_audit_frozen"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verified(path: Path, manifest_path: Path, status: str) -> tuple[dict[str, object], str]:
    payload = json.loads(path.read_text())
    manifest = json.loads(manifest_path.read_text())
    digest = sha256_file(path)
    if digest != manifest.get("sha256"):
        raise RuntimeError(f"stale freeze manifest for {path.name}")
    if payload.get("status") != status or manifest.get("status") != status:
        raise RuntimeError(f"wrong frozen status for {path.name}")
    return payload, digest


def conservative_q_error(
    point: dict[str, object], audit_by_t: dict[float, dict[str, object]]
) -> float:
    candidates = [float(point["Q_standard_error"])]
    block_values = point.get("Q_block_order_4_6_8")
    if block_values is not None:
        central = float(point["Q"])
        candidates.append(max(abs(float(value) - central) for value in block_values))
    audit = audit_by_t.get(round(float(point["t"]), 12))
    if audit is not None:
        candidates.append(float(audit["conservative_standard_error"]))
    return max(candidates)


def row(point: dict[str, object], q_error: float) -> tuple[float, ...]:
    t = float(point["t"])
    q_worldsheet = float(point["worldsheet_Q4"])
    amplitude_real = -4.0 * t**5 * q_worldsheet
    amplitude_real_error = 4.0 * t**5 * q_error
    return t, amplitude_real, amplitude_real_error


def write_table(path: Path, rows: list[tuple[float, ...]]) -> None:
    lines = ["t amplitude_real amplitude_real_error"]
    lines.extend(" ".join(f"{value:.17g}" for value in values) for values in rows)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scan",
        type=Path,
        default=RUN_DIR / "worldsheet_scan_30point.json",
    )
    parser.add_argument(
        "--scan-manifest",
        type=Path,
        default=RUN_DIR / "worldsheet_scan_30point_frozen.json",
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=RUN_DIR / "worldsheet_newpoint_block_audit_frozen.json",
    )
    parser.add_argument(
        "--audit-manifest",
        type=Path,
        default=RUN_DIR / "worldsheet_newpoint_block_audit_manifest.json",
    )
    parser.add_argument("--output-dir", type=Path, default=FIGURE_DIR)
    arguments = parser.parse_args()

    scan, scan_hash = verified(arguments.scan, arguments.scan_manifest, SCAN_STATUS)
    audit, audit_hash = verified(arguments.audit, arguments.audit_manifest, AUDIT_STATUS)
    if scan.get("matrix_model_information_used") is not False:
        raise RuntimeError("worldsheet scan is not target-free")
    if audit.get("matrix_model_information_used") is not False:
        raise RuntimeError("worldsheet audit is not target-free")
    if audit.get("verified_production_scan", {}).get("sha256") != scan_hash:
        raise RuntimeError("block-order audit does not certify this frozen scan")
    audit_by_t = {
        round(float(record["t"]), 12): record for record in audit["points"]
    }

    no_residue: list[tuple[float, ...]] = []
    residue: list[tuple[float, ...]] = []
    qmc_only_points: list[float] = []
    block_checked_points: list[float] = []
    inflated_points: list[float] = []
    for point in scan["points"]:
        if point["fit_role"] != "primary":
            continue
        t = float(point["t"])
        error = conservative_q_error(point, audit_by_t)
        values = row(
            {
                "t": t,
                "worldsheet_Q4": float(point["Q"]),
            },
            error,
        )
        has_block_check = (
            "Q_block_order_4_6_8" in point or round(t, 12) in audit_by_t
        )
        (block_checked_points if has_block_check else qmc_only_points).append(t)
        if error > float(point["Q_standard_error"]) * (1.0 + 1.0e-12):
            inflated_points.append(t)
        if t > 0.4:
            residue.append(values)
        else:
            no_residue.append(values)

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    no_residue_path = arguments.output_dir / "sphere_1to4_t_no_residue.dat"
    residue_path = arguments.output_dir / "sphere_1to4_t_residue.dat"
    write_table(no_residue_path, no_residue)
    write_table(residue_path, residue)
    certificate = {
        "status": "verified_error_bar_provenance_and_propagation",
        "verified_scan_sha256": scan_hash,
        "verified_block_audit_sha256": audit_hash,
        "primary_point_count": len(no_residue) + len(residue),
        "ordinary_contour_point_count": len(no_residue),
        "residue_prescription_point_count": len(residue),
        "block_order_checked_t": block_checked_points,
        "qmc_only_t": qmc_only_points,
        "points_inflated_above_qmc_error_by_block_audit": inflated_points,
        "q_error_definition": (
            "maximum of the production scrambled-QMC standard error, the stored "
            "block-order spread when available, and the frozen independent "
            "block-order audit error when available"
        ),
        "amplitude_error_propagation": "sigma_A=4*t^5*sigma_Q4",
        "coverage_note": (
            "verified conservative numerical estimates, not rigorous confidence "
            "intervals; points without block-order checks use QMC error only"
        ),
        "plot_data_sha256": {
            no_residue_path.name: sha256_file(no_residue_path),
            residue_path.name: sha256_file(residue_path),
        },
    }
    certificate_path = arguments.output_dir / "sphere_1to4_error_bar_audit.json"
    certificate_path.write_text(json.dumps(certificate, indent=2) + "\n")
    print(json.dumps(certificate, indent=2))


if __name__ == "__main__":
    main()
