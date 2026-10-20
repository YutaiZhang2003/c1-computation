"""Build the lightweight inputs from an existing production_integrated directory.

This maintainer command is unnecessary for normal result recovery. It preserves
the numerical sample values and the ordered quadrature weights without rounding.
The original dataset is read-only; the destination must not exist.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import io
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def portable_example_readme(contents, sector):
    """Keep reference-table descriptions without obsolete exporter commands."""
    if not sector:
        return contents.split("The tables use explicit reference momenta")[0] + (
            "These are retained reference examples, with parameters and generation "
            "receipts beside the tables. They illustrate block conventions and are "
            "not complete production coefficient banks. Historical exporter "
            "commands in the receipts refer to the original source packages.\n\n"
            "Use the [repository README](../../../../README.md) for standalone "
            "recovery and the organized computational source.\n")
    before, rest = contents.split("From the extracted package root,", 1)
    after = rest.split("## From a block to a correlator", 1)[1]
    after = after.split("The main [package README]")[0]
    return before + (
        "The JSON files and `generation_receipt.json` preserve the original "
        "generation settings. These reference examples are checksum-verified; "
        "the historical standalone exporters are not part of this repository.\n\n"
        "## From a block to a correlator") + after + (
        "See the [repository README](../../../../../README.md) for standalone "
        "recovery, conventions and production entry points.\n")


def compact_moduli_archive(source, destination):
    """Replace hashed geometry dictionaries by ordered index/weight arrays."""
    reports = {}
    with zipfile.ZipFile(source) as archive:
        for group in ("main", "primary_t0250", "primary_t0950", "upper_t0250", "rms"):
            design = json.loads(archive.read(group + "/design.json"))
            ids = design["sample_ids"]
            index = {key: i for i, key in enumerate(ids)}
            if len(index) != len(ids):
                raise ValueError("Duplicate sample IDs")
            terms = {cell["cell"]["id"]: cell["cell"]["term"] for cell in design["cells"]}
            rules = defaultdict(lambda: [[], []])
            coverage = set()
            for i in range(len(design["batches"])):
                record = json.loads(archive.read(group + f"/geometry/{i:05d}.json"))
                if set(record["samples"]) != set(record["weights"]):
                    raise ValueError("Geometry/weight coverage mismatch")
                if record["cell_id"] not in terms:
                    raise ValueError("Unknown integration cell")
                for key, rows in record["weights"].items():
                    sample_index = index[key]
                    coverage.add(sample_index)
                    for rule, weight, _ in rows:
                        if not np.isfinite(weight) or weight <= 0:
                            raise ValueError("Invalid quadrature weight")
                        entry = rules[record["cell_id"], rule]
                        entry[0].append(sample_index)
                        entry[1].append(weight)
            if coverage != set(range(len(ids))):
                raise ValueError("Incomplete sample coverage")
            metadata, arrays = [], {}
            for i, ((cell_id, rule), (indices, weights)) in enumerate(rules.items()):
                metadata.append({"cell_id": cell_id, "rule": rule, "array": str(i)})
                arrays[f"indices_{i}"] = np.asarray(indices, dtype=np.int32)
                arrays[f"weights_{i}"] = np.asarray(weights, dtype=np.float64)
                # No sorting, coalescing, or precision reduction is applied.
                np.testing.assert_array_equal(arrays[f"indices_{i}"], indices)
                np.testing.assert_array_equal(arrays[f"weights_{i}"], weights)
            target = destination / group
            target.mkdir(parents=True)
            np.savez_compressed(target / "weights.npz", **arrays)
            with np.load(target / "weights.npz", allow_pickle=False) as restored:
                for key, value in arrays.items():
                    np.testing.assert_array_equal(restored[key], value)
            write_json(target / "design.json", {
                "format": "ordered-moduli-weights-v1",
                "sample_count": len(ids), "terms": terms, "rules": metadata,
                "original_sample_ids_sha256": hashlib.sha256(
                    json.dumps(ids, separators=(",", ":")).encode()).hexdigest(),
            })
            count = 0
            for name in archive.namelist():
                if name.startswith(group + "/values/") and name.endswith(".npz"):
                    path = destination / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    contents = archive.read(name)
                    path.write_bytes(contents)
                    if path.read_bytes() != contents:
                        raise ValueError("Changed sample array archive")
                    count += 1
            reports[group] = {"samples": len(ids), "rules": len(rules),
                             "value_files": count,
                             "ordered_weights_and_sample_values_preserved": True}
    return reports


def build(source, output):
    source, output = source.resolve(), output.resolve()
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Source and destination must be separate directories")
    from recover_production_results import verify_data
    original_verified = verify_data(source)
    original = json.loads((source / "DATASET_MANIFEST.json").read_text())
    originals = {row["path"]: row for row in original["files"]}
    output.mkdir(parents=True, exist_ok=False)
    archive_name = "one_loop/three_point/inputs/momentum_integrated_samples.zip"
    omitted = {
        archive_name: "Replaced by ordered numeric weights and unchanged sample-value files.",
        "two_loop/inputs/momentum_integrated/bulk_samples.csv":
            "Verbose geometry descriptions; density, physical weight and replicate remain in bulk_values.npz.",
        "one_loop/three_point_thermal/collision_mesh.json":
            "Geometry for rerunning collision integrations; the accepted integrated addbacks are retained.",
        "tree_level/six_point/reference/sphere_1to5_im_amplitude.tex":
            "Figure source replaced by its embedded numerical table.",
        "README.md": "Replaced by the repository README.",
    }
    transforms = {}
    for name, record in sorted(originals.items()):
        if name in omitted:
            continue
        path, target = source / name, output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if name.startswith("one_loop/three_point_thermal/sampling/") and path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as arrays:
                keep = {key: arrays[key] for key in arrays.files
                        if key not in {"unit", "coordinates"}}
                np.savez_compressed(target, **keep)
                with np.load(target, allow_pickle=False) as restored:
                    for key, values in keep.items():
                        np.testing.assert_array_equal(restored[key], values)
            transforms[name] = {
                "original_sha256": record["sha256"],
                "preserved_fields": sorted(keep),
                "omitted_fields": ["unit", "coordinates"],
                "preserved_arrays_equal": True,
            }
        elif name.startswith("computation_examples/coefficient_examples/") and path.name == "README.md":
            target.write_text(portable_example_readme(
                path.read_text(), sector=path.parent.name != "coefficient_examples"))
            transforms[name] = {
                "original_sha256": record["sha256"],
                "change": "Updated links and removed commands for absent historical exporters; numerical tables unchanged.",
            }
        else:
            shutil.copyfile(path, target)
            if digest(target) != record["sha256"]:
                raise ValueError("Changed original input: " + name)
    report = compact_moduli_archive(
        source / archive_name, output / "one_loop/three_point/inputs/moduli_samples")
    tex = source / "tree_level/six_point/reference/sphere_1to5_im_amplitude.tex"
    embedded = []
    for line in tex.read_text().splitlines():
        parts = line.rstrip("\\").split()
        if len(parts) == 3:
            try:
                embedded.append([float(value) for value in parts])
            except ValueError:
                pass
    if len(embedded) != 15:
        raise ValueError("Unexpected six-point reference table")
    np.savetxt(output / "tree_level/six_point/reference/sphere_1to5_native.dat",
               embedded, header="t ImA SE", comments="", fmt="%.17g")
    files = []
    for path in sorted(output.rglob("*")):
        if not path.is_file():
            continue
        name = str(path.relative_to(output))
        row = {"path": name, "bytes": path.stat().st_size, "sha256": digest(path)}
        if name in originals:
            row["original_sha256"] = originals[name]["sha256"]
        files.append(row)
    manifest = {
        "format": "c1-light-production-inputs-v1",
        "data_stage": "after Liouville momentum integration",
        "coefficient_banks_included": False,
        "source_manifest_sha256": digest(source / "DATASET_MANIFEST.json"),
        "original_dataset": original_verified,
        "files": files, "transformed_files": transforms,
        "moduli_archive_compaction": report,
        "omitted_files": [{
            "path": name, "reason": reason, "sha256": originals[name]["sha256"]
        } for name, reason in omitted.items()],
        "scope": "Numerical reduction inputs. Full production geometry and coefficient banks are not included.",
    }
    write_json(output / "DATASET_MANIFEST.json", manifest)
    print(json.dumps(verify_data(output), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.source, args.output)
