#!/usr/bin/env python3
"""Freeze target-free block-order audits of representative new 1->4 points."""

from __future__ import annotations

from c1_paths import PACKAGE_ROOT, data_root, package_path, module_path

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any

from sphere_five_point_30point_worldsheet_fit import SCAN_STATUS


STATUS = "sphere_1to4_worldsheet_newpoint_block_audit_frozen"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def summarize(
    scan_path: Path,
    scan_manifest_path: Path,
    audit_paths: list[Path],
    output_path: Path,
    manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    scan = json.loads(scan_path.read_text())
    scan_manifest = json.loads(scan_manifest_path.read_text())
    scan_hash = sha256_file(scan_path)
    if scan_hash != scan_manifest.get("sha256"):
        raise RuntimeError("production scan does not match its freeze manifest")
    if scan.get("status") != SCAN_STATUS:
        raise RuntimeError("audit requires the frozen 30-point production scan")
    if scan.get("matrix_model_information_used") is not False:
        raise RuntimeError("production scan does not certify target-free computation")
    production = {round(float(point["t"]), 12): point for point in scan["points"]}
    records = []
    source_records = []
    for path in audit_paths:
        audit = json.loads(path.read_text())
        source_records.append({"path": str(path.resolve()), "sha256": sha256_file(path)})
        settings = audit["settings"]
        if "no matrix-model values" not in str(audit.get("description", "")):
            raise RuntimeError("audit artifact does not certify target-free computation")
        for point in audit["points"]:
            key = round(float(point["t"]), 12)
            if key not in production or production[key]["scan_cohort"] != "new_extension":
                raise RuntimeError("audit point is absent from the new production cohort")
            original = production[key]
            audit_q = float(point["Q"]["real"])
            audit_sigma = float(point["Q_standard_error"]["real"])
            spread = abs(audit_q - float(original["Q"]))
            records.append(
                {
                    "t": float(point["t"]),
                    "production_block_order": int(original["block_order"]),
                    "production_Q4": float(original["Q"]),
                    "production_Q4_qmc_standard_error": float(original["Q_standard_error"]),
                    "audit_block_order": int(settings["block_order"]),
                    "audit_Q4": audit_q,
                    "audit_Q4_qmc_standard_error": audit_sigma,
                    "independent_block_order_spread": spread,
                    "conservative_standard_error": max(
                        float(original["Q_standard_error"]), audit_sigma, spread
                    ),
                }
            )
    records.sort(key=lambda record: record["t"])
    if [record["t"] for record in records] != [0.31, 0.35, 0.44]:
        raise RuntimeError("expected audits at t=0.31, 0.35, and 0.44")
    payload: dict[str, Any] = {
        "status": STATUS,
        "program": Path(__file__).name,
        "matrix_model_information_used": False,
        "selection_basis": {
            "0.31": "representative real-contour interior point",
            "0.35": "block-order transition region below the first wall",
            "0.44": "representative residue-corrected point above the first wall",
        },
        "verified_production_scan": {
            "path": str(scan_path.resolve()),
            "sha256": scan_hash,
        },
        "source_audits": source_records,
        "points": records,
    }
    write_json(output_path, payload)
    manifest = {
        "status": STATUS,
        "artifact": str(output_path.resolve()),
        "sha256": sha256_file(output_path),
        "frozen_on": date.today().isoformat(),
        "point_count": len(records),
        "matrix_model_information_used": False,
    }
    write_json(manifest_path, manifest)
    return payload, manifest


def main() -> None:
    run_dir = data_root("tree_level") / "sphere_five_point_1to4" / "blind30_20260824"
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan", type=Path, default=run_dir / "worldsheet_scan_30point.json")
    parser.add_argument("--scan-manifest", type=Path, default=run_dir / "worldsheet_scan_30point_frozen.json")
    parser.add_argument(
        "--audits",
        nargs="+",
        type=Path,
        default=(
            run_dir / "worldsheet_newpoint_block8_audit.json",
            run_dir / "worldsheet_newpoint_block6_audit.json",
        ),
    )
    parser.add_argument("--output", type=Path, default=run_dir / "worldsheet_newpoint_block_audit_frozen.json")
    parser.add_argument("--manifest", type=Path, default=run_dir / "worldsheet_newpoint_block_audit_manifest.json")
    arguments = parser.parse_args()
    _, manifest = summarize(arguments.scan, arguments.scan_manifest, list(arguments.audits), arguments.output, arguments.manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
