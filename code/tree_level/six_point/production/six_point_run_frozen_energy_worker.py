"""Yin-192 worker for one prepared sphere 1->5 energy release.

The release is immutable after preparation.  This adapter only builds the
declared coefficient shards, assembles them, evaluates the declared 2048-point
Sobol net, and writes an audit receipt.  It deliberately has no t=.3205
constant: the energy and p=2*t**2 are read from the bound release manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import socket
import time


WORKERS = 192


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")
    os.replace(tmp, path)


def initialize(directory):
    directory = Path(directory).resolve()
    release = json.loads((directory / "input_release.json").read_text())
    if release.get("version") not in {"sphere-six-point-yin-curve-energy-v2",
                                      "sphere-six-point-yin-radial-mixture-energy-v1",
                                      "sphere-six-point-yin-radial-production-energy-v2"}:
        raise ValueError("unsupported energy release")
    if release.get("workers") != WORKERS or release.get("partition") != "yin":
        raise ValueError("release is not bound to Yin-192")
    t = float(release["t"])
    if not 0 < t < 1 / 3:
        raise ValueError("energy is outside the residue-free strip")
    expected_p = 2 * t * t
    if abs(float(release["radial_power"]) - expected_p) > 2e-15:
        raise ValueError("analytic p=2*t**2 binding mismatch")
    if os.environ.get("SLURM_JOB_ID"):
        if os.environ.get("SLURM_JOB_PARTITION") != "yin":
            raise ValueError("wrong scheduler partition")
        if socket.gethostname().split(".")[0] not in set(release["nodes"]):
            raise ValueError("worker is outside the authorized Yin nodes")
    for relative, expected in release["files"].items():
        path = (directory / relative).resolve()
        if not path.is_relative_to(directory) or not path.is_file() or digest(path) != expected:
            raise ValueError("energy release input changed: " + relative)
    runpy.run_path(str(directory / "source/run.py"))["bootstrap"]()
    return directory, release


def existing_scan_output(output, *, scope, geometry_slice):
    """Validate and reuse a completed immutable scan output."""
    output = Path(output)
    if not output.exists():
        return None
    required = [output / name for name in ("summary.json", "input_manifest.json", "pointwise.npz")]
    if not all(path.is_file() for path in required):
        raise ValueError("existing scan output is incomplete: " + str(output))
    summary = json.loads(required[0].read_text())
    manifest = json.loads(required[1].read_text())
    if (summary.get("execution_scope") != scope
            or summary.get("geometry_slice") != geometry_slice
            or summary.get("identity") != manifest.get("identity")
            or digest(required[2]) != summary.get("pointwise_sha256")
            or not summary.get("finite_all_evaluated_values")):
        raise ValueError("existing scan output failed its immutable receipt checks: " + str(output))
    return summary


def existing_assembly_output(output):
    """Validate and reuse a completed full-source assembly."""
    output = Path(output)
    if not output.exists():
        return None
    summary_path, pointwise_path = output / "summary.json", output / "pointwise.npz"
    if not summary_path.is_file() or not pointwise_path.is_file():
        raise ValueError("existing assembly output is incomplete: " + str(output))
    summary = json.loads(summary_path.read_text())
    if (summary.get("execution_scope") != "full-source"
            or not summary.get("exact_original_index_coverage")
            or not summary.get("all_original_source_geometry_and_density_hashes_match")
            or not summary.get("complete_finite_amplitude")
            or digest(pointwise_path) != summary.get("pointwise_sha256")):
        raise ValueError("existing assembly output failed its immutable receipt checks: " + str(output))
    return summary


def execute(directory, action, index=0):
    directory, release = initialize(directory)
    t = float(release["t"])
    if action == "validate":
        return dict(passed=True, t=t, radial_power=release["radial_power"], workers=WORKERS)
    if action == "build":
        from sphere_six_point_multilevel_catalog import build_shard
        tasks = json.loads((directory / "build_tasks.json").read_text())
        if not 0 <= index < WORKERS:
            raise ValueError("build worker index outside 192-worker array")
        completed = []
        started = time.time()
        for task in tasks[index::WORKERS]:
            build_shard(directory / task["catalog"], task["start"], task["stop"],
                        directory / task["output"])
            completed.append(task["output"])
            write(directory / "progress/build" / f"worker_{index:03d}.json",
                  dict(worker=index, assigned=len(tasks[index::WORKERS]), completed=completed,
                       seconds=time.time() - started))
        return dict(action=action, worker=index, completed=len(completed), seconds=time.time()-started)
    if action == "assemble":
        from sphere_six_point_multilevel_catalog import assemble_views
        from sphere_six_point_momentum_banks import SphereSixPointMomentumBank
        inventory = json.loads((directory / "catalog_inventory.json").read_text())
        if not 0 <= index < WORKERS:
            raise ValueError("assemble worker index outside 192-worker array")
        completed = []
        for record in inventory[index::WORKERS]:
            catalog = directory / record["catalog"]
            banks = assemble_views(catalog, sorted((catalog.parent / "shards").glob("shard_*.npz")))
            for bank in banks.values():
                variant = bank.proposal.metadata["variant"]
                destination = directory / record["variants"][variant]
                if destination.exists():
                    if SphereSixPointMomentumBank.load(destination).fingerprint != bank.fingerprint:
                        raise ValueError("existing assembled bank differs")
                else:
                    bank.save(destination)
            completed.append(record["name"])
        return dict(action=action, worker=index, completed=completed)
    if action == "addback":
        from sphere_six_point_parallel_routing import (evaluate_marked_addback,
            load_marked_addback, save_marked_addback)
        config = json.loads((directory / "collar_routing.json").read_text())
        output = directory / "shared_addback.npz"
        sidecar = output.with_suffix(".json")
        if output.exists() or sidecar.exists():
            if not output.is_file() or not sidecar.is_file():
                raise ValueError("partial shared addback output exists")
            result = load_marked_addback(output, [t], config)
            return dict(action=action, t=t, reused=True,
                        path=str(output), context_identity=result["context_identity"])
        result = evaluate_marked_addback([t], config)
        return save_marked_addback(result, output)
    if action == "preflight":
        from sphere_six_point_multilevel_scan import run_multilevel
        config = directory / "full_config.json"
        geometry_slice = dict(replicate=0, start=0, stop=1)
        output = directory / "preflight_output"
        existing = existing_scan_output(output, scope="geometry-shard",
                                        geometry_slice=geometry_slice)
        if existing is not None:
            return dict(action=action, t=t, reused=True, result=existing)
        # A one-point indexed execution validates imports, banks, geometry and
        # the addback context without claiming an amplitude result.
        result = run_multilevel(config, output,
                                geometry_slice=geometry_slice,
                                shared_addback=directory / "shared_addback.npz")
        return dict(action=action, t=t, result=result)
    if action == "full":
        from sphere_six_point_multilevel_scan import run_multilevel
        tasks = json.loads((directory / "full_tasks.json").read_text())
        if not 0 <= index < WORKERS:
            raise ValueError("full worker index outside 192-worker array")
        completed = []
        reused = []
        started = time.time()
        for task in tasks[index::WORKERS]:
            output = directory / task["output"]
            geometry_slice = {k: task[k] for k in ("replicate", "start", "stop")}
            existing = existing_scan_output(output, scope="geometry-shard",
                                            geometry_slice=geometry_slice)
            if existing is None:
                run_multilevel(directory / task["config"], output,
                               geometry_slice=geometry_slice,
                               shared_addback=directory / "shared_addback.npz")
            else:
                reused.append(task["output"])
            completed.append(task["output"])
            write(directory / "progress/full" / f"worker_{index:03d}.json",
                  dict(worker=index, assigned=len(tasks[index::WORKERS]), completed=completed,
                       reused=reused, seconds=time.time() - started))
        return dict(action=action, worker=index, completed=len(completed), reused=len(reused),
                    seconds=time.time()-started)
    if action == "reduce":
        from sphere_six_point_multilevel_scan import assemble_multilevel
        if not 0 <= index < 2:
            raise ValueError("reduce index outside two-stage reduction slot")
        design = "main"
        tasks = json.loads((directory / "full_tasks.json").read_text())
        paths = [directory / task["output"] for task in tasks]
        output = directory / "amplitude" / design
        existing = existing_assembly_output(output)
        result = existing if existing is not None else assemble_multilevel(paths, output)
        return dict(action=action, t=t, output=str(output), reused=existing is not None, result=result)
    if action == "audit":
        import numpy as np
        summary_path = directory / "amplitude/main/summary.json"
        pointwise_path = directory / "amplitude/main/pointwise.npz"
        summary = json.loads(summary_path.read_text())
        if digest(pointwise_path) != summary.get("pointwise_sha256"):
            raise ValueError("assembled pointwise checksum mismatch")
        if (summary.get("execution_scope") != "full-source"
                or summary.get("evaluated_point_count") != 2048
                or not summary.get("exact_original_index_coverage")
                or not summary.get("all_original_source_geometry_and_density_hashes_match")
                or not summary.get("complete_finite_amplitude")):
            raise ValueError("assembled amplitude does not cover the exact complete source")
        with np.load(pointwise_path, allow_pickle=False) as data:
            values = data["hybrid_I6"]
            points = data["points"]
            variants = data["variants"].tolist()
            energies = data["energies"].tolist()
        if values.shape[0:3] != (1, 8, 256) or not np.isfinite(values).all():
            raise ValueError("full 2048-point amplitude shape or finiteness check failed")
        if variants != summary.get("variants") or energies != [t] or "baseline" not in variants:
            raise ValueError("assembled physical or variant axes disagree with the release")
        scale = -1 / (40 * np.pi**3 * t**6)
        scaled = values[0] * scale
        baseline_index = variants.index("baseline")
        baseline = scaled[:, :, baseline_index]
        replicate_means = baseline.mean(axis=1)
        mean = replicate_means.mean()
        standard_error = complex(replicate_means.real.std(ddof=1),
                                 replicate_means.imag.std(ddof=1)) / np.sqrt(len(replicate_means))
        normalization = max(abs(mean), np.finfo(float).tiny)
        controls = {}
        for index, name in enumerate(variants):
            difference = scaled[:, :, index] - baseline
            controls[name] = dict(
                signed_mean_real=float(difference.real.mean()),
                signed_mean_imag=float(difference.imag.mean()),
                mean_absolute_difference=float(np.abs(difference).mean()),
                relative_mean_absolute_difference=float(np.abs(difference).mean() / normalization))
        relative_se = float(abs(standard_error) / normalization)
        maximum_control = max(row["relative_mean_absolute_difference"] for row in controls.values())
        numerical_failure_count = len(summary.get("numerical_failures", []))
        scope_complete = summary.get("unresolved_scope_count") == 0 and numerical_failure_count == 0
        result = dict(version="sphere-six-point-yin-curve-audit-v2", t=t,
            radial_power=release["radial_power"], sample_count=int(points.shape[0] * points.shape[1]),
            mean_real=float(mean.real), mean_imag=float(mean.imag),
            replicate_means=[[float(value.real), float(value.imag)] for value in replicate_means],
            scramble_se_real=float(standard_error.real), scramble_se_imag=float(standard_error.imag),
            empirical_relative_scramble_se=relative_se,
            control_differences=controls,
            maximum_control_relative_mean_absolute_difference=maximum_control,
            target_relative_accuracy=0.01,
            one_percent_empirical_screen=bool(scope_complete and relative_se <= .01 and maximum_control <= .01),
            finite=True, exact_complete_source=True, unresolved_scope_count=summary.get("unresolved_scope_count"),
            numerical_failure_count=numerical_failure_count,
            baseline_prefix_unresolved_count=summary.get("baseline_prefix_unresolved_count"),
            available_prefix_unresolved_count=summary.get("available_prefix_unresolved_count"),
            analytic_patch_count=summary.get("analytic_patch_count"),
            total_accuracy_certified=False,
            input_release_sha256=digest(directory / "input_release.json"),
            summary_sha256=digest(summary_path), pointwise_sha256=digest(pointwise_path))
        write(directory / "final_audit.json", result)
        return result
    raise ValueError("unknown action")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("action", choices=("validate", "build", "assemble", "addback", "preflight", "full", "reduce", "audit"))
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(execute(args.directory, args.action, args.index), indent=2, allow_nan=False, default=str))
