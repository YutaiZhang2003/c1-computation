"""Explicit reuse of validated rows from a different frozen coefficient producer.

The supported mapping is deliberately exact: identical physical policy, complete
catalog momentum bits, row ordering and degrees. The original arrays and producer
evidence remain bound into the destination catalog and every assembled bank.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import runpy

import numpy as np

EXPORT_VERSION = "sphere-six-point-verified-catalog-export-v1"
IMPORT_VERSION = "sphere-six-point-exact-producer-import-v1"
ARRAYS = ("p_values", "row_max_orders", "row_offsets", "coefficients_flat", "log_dozz")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _semantic(physical):
    return {k:v for k,v in physical.items() if k != "source_dependencies"}


def export_catalog(catalog_path, shard_paths, expected_hashes):
    """Run inside the ORIGINAL frozen source environment, never the new loader."""
    from six_point_build_momentum_banks import _array_hash, _hash
    from six_point_mixed_order_node_catalog import load_catalog, _orders, _read_shard
    catalog_path = Path(catalog_path)
    manifest, nodes, _ = load_catalog(catalog_path)
    orders = _orders(catalog_path, manifest)
    records, present = [], np.zeros(len(nodes), dtype=bool)
    for path in map(Path, shard_paths):
        sha = digest(path)
        if expected_hashes.get(str(path.resolve())) != sha:
            raise ValueError("original shard differs from the preservation inventory")
        metadata, _, _, _ = _read_shard(path, manifest, nodes, orders)
        start, stop = metadata["row_start"], metadata["row_stop"]
        if present[start:stop].any():
            raise ValueError("overlapping imported intervals")
        present[start:stop] = True
        with np.load(path, allow_pickle=False) as data:
            hashes = {key:_array_hash(data[key]) for key in ARRAYS}
        records.append(dict(row_start=start, row_stop=stop,
            original_file_sha256=sha, original_shard_fingerprint=metadata["fingerprint"],
            original_filename=path.name, array_hashes=hashes,
            degree_groups_sha256=_hash(metadata["degree_groups"]),
            original_build_seconds=metadata["build_seconds"]))
    result = dict(version=EXPORT_VERSION, producer_catalog_fingerprint=manifest["fingerprint"],
        producer_physical_fingerprint=manifest["physical_fingerprint"],
        producer_physical=manifest["physical"], producer_implementation_sources=manifest["implementation_sources"],
        node_hash=_array_hash(nodes), orders_hash=_array_hash(orders),
        exporter_source_sha256=digest(__file__), producer_source_root="../../imports/original_source",
        shards=sorted(records, key=lambda r:r["row_start"]))
    return dict(result, identity=_hash(result))


def validate_spec(spec, manifest, nodes, orders, catalog_path):
    from six_point_build_momentum_banks import _array_hash, _hash
    if (spec.get("version") != EXPORT_VERSION
            or spec.get("identity") != _hash({k:v for k,v in spec.items() if k != "identity"})
            or spec.get("exporter_source_sha256") != digest(__file__)
            or _semantic(spec["producer_physical"]) != _semantic(manifest["physical"])
            or spec["producer_physical_fingerprint"] != _hash(spec["producer_physical"])
            or spec["node_hash"] != _array_hash(nodes)
            or spec["orders_hash"] != _array_hash(orders)
            or not spec.get("producer_implementation_sources")):
        raise ValueError("import producer, physical policy, exact nodes or degrees differ")
    if spec.get("producer_source_root") != "../../imports/original_source":
        raise ValueError("unexpected original-source location")
    source_root = (Path(catalog_path).parent / spec["producer_source_root"]).resolve()
    dependencies = dict(spec["producer_implementation_sources"])
    for relative, sha in (spec["producer_physical"].get("source_dependencies") or {}).items():
        if relative in dependencies and dependencies[relative] != sha:
            raise ValueError("conflicting original producer source identities")
        dependencies[relative] = sha
    for relative, sha in dependencies.items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or digest(source_root/path) != sha:
            raise ValueError("original producer source closure changed: " + relative)
    present = np.zeros(len(nodes), dtype=bool)
    for record in spec["shards"]:
        start, stop = record["row_start"], record["row_stop"]
        if (type(start) is not int or type(stop) is not int or not 0 <= start < stop <= len(nodes)
                or present[start:stop].any() or set(record["array_hashes"]) != set(ARRAYS)
                or record["array_hashes"]["p_values"] != _array_hash(nodes[start:stop])
                or record["array_hashes"]["row_max_orders"] != _array_hash(orders[start:stop])):
            raise ValueError("invalid, overlapping or incorrectly mapped import interval")
        present[start:stop] = True


def bind_imports(catalog_path, exported):
    """Bind immutable export evidence before any destination shards are built."""
    from six_point_build_momentum_banks import _hash, _atomic_file
    from six_point_mixed_order_node_catalog import load_catalog, _orders
    catalog_path = Path(catalog_path)
    manifest, nodes, _ = load_catalog(catalog_path)
    orders = _orders(catalog_path, manifest)
    validate_spec(exported, manifest, nodes, orders, catalog_path)
    if "producer_import" in manifest:
        if manifest["producer_import"] != exported:
            raise ValueError("catalog already binds a different producer import")
        return manifest
    if any((catalog_path.parent / "shards").glob("*.npz")):
        raise ValueError("cannot change catalog identity after destination shards exist")
    spec = {k:v for k,v in manifest.items() if k != "fingerprint"}
    spec["producer_import"] = exported
    updated = dict(spec, fingerprint=_hash(spec))
    _atomic_file(catalog_path, lambda handle:handle.write(json.dumps(updated, indent=2)+"\n"), "w")
    return load_catalog(catalog_path)[0]


def record_for(manifest, start, stop):
    candidates = manifest.get("producer_import", {}).get("shards", [])
    for record in candidates:
        a, b = record["row_start"], record["row_stop"]
        if max(start, a) < min(stop, b):
            if (start, stop) != (a, b):
                raise ValueError("shard interval splits or overlaps an imported chunk")
            return record
    return None


def origin_for(manifest, record):
    exported = manifest["producer_import"]
    return dict(version=IMPORT_VERSION, export_identity=exported["identity"],
        producer_catalog_fingerprint=exported["producer_catalog_fingerprint"],
        producer_physical=exported["producer_physical"],
        producer_implementation_sources=exported["producer_implementation_sources"],
        original_file_sha256=record["original_file_sha256"],
        original_shard_fingerprint=record["original_shard_fingerprint"],
        row_mapping="identical binary64 nodes, row positions and complete degrees",
        fresh_coefficient_rows_computed=0)


def validate_imported_arrays(manifest, metadata, arrays):
    from six_point_build_momentum_banks import _array_hash, _hash
    record = record_for(manifest, metadata["row_start"], metadata["row_stop"])
    if record is None:
        if "import_origin" in metadata:
            raise ValueError("unapproved producer import")
        return
    if (metadata.get("import_origin") != origin_for(manifest, record)
            or any(_array_hash(arrays[k]) != record["array_hashes"][k] for k in ARRAYS)
            or _hash(metadata["degree_groups"]) != record["degree_groups_sha256"]
            or metadata["build_seconds"] != record["original_build_seconds"]):
        raise ValueError("imported coefficients, DOZZ or original provenance changed")


def materialize_import(catalog_path, original_shard, output):
    from six_point_build_momentum_banks import _atomic_file, _hash
    from six_point_mixed_order_node_catalog import load_catalog, _orders, _read_shard, _shard_hash
    catalog_path, original_shard, output = map(Path, (catalog_path, original_shard, output))
    manifest, nodes, _ = load_catalog(catalog_path)
    orders = _orders(catalog_path, manifest)
    with np.load(original_shard, allow_pickle=False) as data:
        original = json.loads(data["metadata"].item())
        arrays = {k:data[k].copy() for k in ARRAYS}
    record = record_for(manifest, original["row_start"], original["row_stop"])
    exported = manifest.get("producer_import", {})
    if (record is None or digest(original_shard) != record["original_file_sha256"]
            or original["fingerprint"] != record["original_shard_fingerprint"]
            or original["catalog_fingerprint"] != exported["producer_catalog_fingerprint"]
            or original["physical_fingerprint"] != exported["producer_physical_fingerprint"]):
        raise ValueError("original artifact does not match the verified export")
    metadata = dict(original, catalog_fingerprint=manifest["fingerprint"],
        physical_fingerprint=manifest["physical_fingerprint"], import_origin=origin_for(manifest, record))
    metadata["fingerprint"] = _shard_hash(metadata, *(arrays[k] for k in ARRAYS))
    validate_imported_arrays(manifest, metadata, arrays)
    if output.exists():
        existing, _, _, _ = _read_shard(output, manifest, nodes, orders)
        if existing != metadata:
            raise ValueError("existing destination differs from the approved import")
        return existing
    _atomic_file(output, lambda handle:np.savez_compressed(handle,
        metadata=json.dumps(metadata, allow_nan=False), **arrays))
    return _read_shard(output, manifest, nodes, orders)[0]


def export_preserved_energy(directory, preservation, output):
    """CLI entry: bootstrap original source and verify its bound release first."""
    directory, preservation, output = map(Path, (directory, preservation, output))
    release = json.loads((directory / "input_release.json").read_text())
    for relative, expected in release["files"].items():
        if digest(directory / relative) != expected:
            raise ValueError("original frozen release changed: " + relative)
    runpy.run_path(str(directory / "source/run.py"))["bootstrap"]()
    inventory = json.loads(preservation.read_text())
    hashes = {str((directory.parent / row["path"]).resolve()):row["sha256"] for row in inventory["files"]}
    tasks = json.loads((directory / "build_tasks.json").read_text())
    catalogs = {}
    for task in tasks:
        shard = directory / task["output"]
        if shard.is_file():
            catalogs.setdefault(task["catalog"], []).append(shard)
    exports = {name:export_catalog(directory/name, shards, hashes) for name,shards in catalogs.items()}
    result = dict(original_energy=release["t"], original_release_sha256=digest(directory/"input_release.json"),
        preservation_manifest_sha256=digest(preservation), catalogs=exports)
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(result, indent=2, allow_nan=False)+"\n"
    if output.exists() and output.read_text() != encoded:
        raise ValueError("existing export differs")
    output.write_text(encoded)
    return dict(catalogs=len(exports), shards=sum(len(v["shards"]) for v in exports.values()), output=str(output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("preservation", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(export_preserved_energy(args.directory, args.preservation, args.output), indent=2))
