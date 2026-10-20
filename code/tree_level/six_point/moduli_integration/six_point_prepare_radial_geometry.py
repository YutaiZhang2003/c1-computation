#!/usr/bin/env python3
"""Prepare audited face-stratified six-point geometry and old-data reuse maps.

This creates geometry only.  It does not build CFT banks, evaluate an
amplitude, submit jobs, or reinterpret historical weights as a new estimate.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
import runpy

if __name__ == "__main__" and not __package__:
    runpy.run_path(str(Path(__file__).resolve().parents[4] / "run.py"))["bootstrap"]()

import numpy as np

from six_point_select_integration_channels import normalize_routing
from six_point_sample_integration_geometry import SphereGeometryPlan, digest, geometry_sources
from six_point_radial_sampling_distribution import (
    face_stratified_radial_proposal, stratified_sobol_points,
)
from six_point_check_sampling_variance import (
    require_radial_proposal_second_moment_condition,
)


VERSION = "sphere-six-point-face-radial-geometry-preparation-v1"
PHYSICAL_REUSE_FIELDS = (
    "sampled_topology", "sampled_ordering", "sampled_q", "topology", "ordering",
    "q", "pair_logs", "scale_logs", "score", "convergence_score", "recovery",
)


def write(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError("preserve existing preparation output: " + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def historical_prefix_reuse(path, plan, eee_count):
    """Verify that a nested historical EEE prefix has identical physical geometry."""
    path = Path(path).resolve()
    sidecar = path.with_suffix(".json")
    metadata = json.loads(sidecar.read_text())
    if metadata.get("data_sha256") != digest(path):
        raise ValueError("historical geometry checksum mismatch: " + str(path))
    endpoint = plan.metadata["radial_proposal"]["reference_endpoint_power"]
    if abs(float(metadata.get("radial_power")) - endpoint) > 2e-15:
        raise ValueError("historical geometry did not use the endpoint proposal")
    with np.load(path, allow_pickle=False) as old:
        if len(old["points"]) < eee_count:
            raise ValueError("historical plan has no complete nested EEE prefix")
        for key in PHYSICAL_REUSE_FIELDS:
            if not np.array_equal(old[key][:eee_count], plan.arrays[key][:eee_count]):
                raise ValueError("historical EEE physical geometry changed: " + key)
        old_density = old["log_density"][:eee_count]
    new_density = plan.arrays["log_density"][:eee_count]
    ratio = np.exp(old_density - new_density)
    bound = 1 / plan.metadata["radial_proposal"]["reference_endpoint_weight"]
    if not np.isfinite(ratio).all() or np.any(ratio > bound * (1 + 2e-13)):
        raise ValueError("historical-to-mixture density conversion violates its lower bound")
    return {
        "historical_path": str(path),
        "historical_data_sha256": metadata["data_sha256"],
        "old_indices": list(range(eee_count)),
        "new_indices": list(range(eee_count)),
        "reusable_CFT_point_count": eee_count,
        "physical_geometry_bit_identical": True,
        "importance_conversion": "multiply pointwise CFT term by exp(old_log_density-new_log_density)",
        "analytic_addback_conversion": "do not reweight; add the matched integral once",
        "minimum_density_ratio_old_over_new": float(ratio.min()),
        "maximum_density_ratio_old_over_new": float(ratio.max()),
        "maximum_allowed_density_ratio_old_over_new": float(bound),
    }


def derive_evaluation_config(template_path, output, result):
    """Retarget one same-energy configuration without copying its CFT banks."""
    template_path = Path(template_path).resolve()
    output = Path(output).resolve()
    config = json.loads(template_path.read_text())
    t = float(result["t"])
    if [float(value) for value in config.get("energies", ())] != [t]:
        raise ValueError("template configuration must contain exactly the prepared energy")
    if not isinstance(config.get("families"), list) or not config["families"]:
        raise ValueError("template configuration has no reusable CFT bank families")
    if any(float(row.get("t")) != t for row in config["families"]):
        raise ValueError("template configuration mixes another energy into its CFT banks")
    for family in config["families"]:
        for name, value in list(family["variants"].items()):
            descriptor = {"path": value} if isinstance(value, str) else dict(value)
            source = (template_path.parent / descriptor["path"]).resolve()
            descriptor["path"] = os.path.relpath(source, output)
            family["variants"][name] = (descriptor["path"] if isinstance(value, str)
                                          else descriptor)
    config.pop("input_manifest", None)
    config.update(
        sample_design=result["sample_design"],
        base_seed=result["base_seed"],
        replicates=result["replicates"],
        sobol_power=result["sobol_power"],
        radial_power=result["radial_proposal"]["reference_endpoint_power"],
        radial_proposal=result["radial_proposal"],
        radial_power_prescription=(
            "face-stratified mixture with endpoint p_E=2*t^2 and configured bulk p_B"),
        geometry_plans=[{
            "replicate": row["replicate"], "seed": row["seed"],
            "path": row["path"], "point_sha256": row["point_sha256"],
        } for row in result["geometry_plans"]],
    )
    return config


def prepare(output, *, t, base_seed, replicates=8, sobol_power=8,
            bulk_power=0.75, routing=None, historical_geometry=None,
            template_config=None):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("preserve existing radial preparation: " + str(output))
    proposal = face_stratified_radial_proposal(t, bulk_power=bulk_power)
    diagnostic = require_radial_proposal_second_moment_condition(t, proposal)
    policy = normalize_routing(routing or {"name": "diagonal-minimum-v1"})
    output.mkdir(parents=True)
    records, reuse = [], []
    historical = Path(historical_geometry).resolve() if historical_geometry else None
    for replicate in range(int(replicates)):
        points, strata = stratified_sobol_points(proposal, total_power=int(sobol_power),
            base_seed=int(base_seed), replicate=replicate)
        plan = SphereGeometryPlan.build(points, radial_proposal=proposal,
            radial_strata=strata, routing=policy)
        relative = Path("geometry") / f"replicate_{replicate}.npz"
        plan.save(output / relative)
        record = {"replicate": replicate, "seed": int(base_seed) + replicate,
                  "path": str(relative), "point_sha256": plan.metadata["point_sha256"],
                  "summary": plan.summary(), "radial_strata": strata}
        records.append(record)
        if historical is not None:
            eee_count = next(row["count"] for row in strata if row["component"] == "EEE")
            reuse.append({"replicate": replicate, **historical_prefix_reuse(
                historical / f"replicate_{replicate}.npz", plan, eee_count)})
    result = {
        "version": VERSION,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "t": float(t),
        "base_seed": int(base_seed),
        "replicates": int(replicates),
        "sobol_power": int(sobol_power),
        "points_per_replicate": 1 << int(sobol_power),
        "sample_design": "stratified-sobol-radial-mixture",
        "preparation_source_sha256": digest(__file__),
        "radial_proposal": proposal,
        "routing": policy,
        "geometry_plans": records,
        "sampling_diagnostic": diagnostic,
        "geometry_sources": geometry_sources("native"),
        "historical_reuse": reuse,
        "historical_reusable_CFT_point_count": sum(row["reusable_CFT_point_count"] for row in reuse),
        "new_non_EEE_point_count": int(replicates) * (1 << int(sobol_power))
            - sum(row["reusable_CFT_point_count"] for row in reuse),
        "complete_amplitude_computed": False,
        "jobs_submitted": False,
    }
    if template_config is not None:
        template = Path(template_config).resolve()
        config = derive_evaluation_config(template, output, result)
        write(output / "evaluation_config.json", config)
        result.update(
            template_config=str(template), template_config_sha256=digest(template),
            evaluation_config="evaluation_config.json",
            evaluation_config_sha256=digest(output / "evaluation_config.json"),
            reused_CFT_bank_family_count=len(config["families"]),
        )
    write(output / "preparation.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--t", type=float, required=True)
    parser.add_argument("--base-seed", type=int, default=2026088400)
    parser.add_argument("--replicates", type=int, default=8)
    parser.add_argument("--sobol-power", type=int, default=8)
    parser.add_argument("--bulk-power", type=float, default=0.75)
    parser.add_argument("--historical-geometry", type=Path)
    parser.add_argument("--template-config", type=Path)
    args = parser.parse_args()
    result = prepare(args.output, t=args.t, base_seed=args.base_seed,
        replicates=args.replicates, sobol_power=args.sobol_power,
        bulk_power=args.bulk_power, historical_geometry=args.historical_geometry,
        template_config=args.template_config)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "plans": len(result["geometry_plans"]),
        "reusable_CFT_points": result["historical_reusable_CFT_point_count"],
        "new_points": result["new_non_EEE_point_count"],
    }, indent=2))


if __name__ == "__main__":
    main()
