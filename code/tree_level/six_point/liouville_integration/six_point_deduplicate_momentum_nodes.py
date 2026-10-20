"""Weight-independent, exact-node construction for verified sphere CFT banks.

A catalog is NOT a quadrature rule. It stores one physical coefficient/DOZZ
problem per distinct binary64 momentum tuple and scatters the results back
onto each original, complete proposal with its original weights. Catalogs
never interpolate nodes or share rows across physical/provider/source inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import time

import numpy as np

from six_point_build_momentum_banks import (
    _array_hash, _atomic_file, _check_source_dependencies, _hash, _provider,
    load_bank_build,
)
from six_point_momentum_banks_and_reweighting import (
    MomentumProposal, SphereSixPointMomentumBank, build_momentum_bank, degree_keys,
)

CATALOG_VERSION = "sphere-six-point-exact-node-catalog-v1"
CATALOG_SHARD_VERSION = "sphere-six-point-exact-node-catalog-shard-v1"
_PHYSICAL_KEYS = ("t", "topology", "incoming_slot", "order", "special_dps",
                  "coefficient_provider", "source_dependencies")


def _bits(points):
    points = np.ascontiguousarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("catalog nodes must have three binary64 components")
    return points.view(np.dtype((np.void, 3 * points.dtype.itemsize))).reshape(-1)


def _same_bits(left, right):
    return left.shape == right.shape and np.array_equal(_bits(left), _bits(right))


def _sources():
    root = Path(__file__).resolve().parents[4]
    names = ("six_point_build_momentum_banks", "six_point_momentum_banks_and_reweighting",
             "six_point_block_coefficient_provider", "six_point_arbitrary_precision_c_recursion",
             "liouville_torus", "liouville_momentum_quadrature", "c1_paths")
    paths = [Path(__file__).resolve()]
    paths += [Path(importlib.import_module(name).__file__).resolve() for name in names]
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths}


def _physical(manifest):
    return {name: manifest.get(name) for name in _PHYSICAL_KEYS}


def prepare_catalog(directory, bank_manifests):
    """Prepare one physical group's exact row union; build no CFT values.

    Supply baseline manifests first to put their unique nodes in a contiguous
    initial prefix. Different physical groups require separate catalogs.
    Original manifests/proposals are copied into the catalog for portability.
    The returned JSON manifest names each view by its proposal fingerprint.
    """
    paths = [Path(p) for p in bank_manifests]
    if not paths:
        raise ValueError("at least one complete bank manifest is required")
    loaded = [(path, *load_bank_build(path)) for path in paths]
    physical = _physical(loaded[0][1])
    if any(_physical(manifest) != physical for _, manifest, _ in loaded):
        raise ValueError("catalog cannot mix energy, graph, slot, order, provider, DOZZ precision or sources")
    unique_views = {}
    for path, manifest, proposal in loaded:
        if manifest["proposal_file"] != "proposal.npz":
            raise ValueError("catalog requires the portable standard proposal.npz manifest layout")
        unique_views.setdefault(proposal.fingerprint, (path, manifest, proposal))
    loaded = list(unique_views.values())
    combined = np.concatenate([proposal.p_values for _, _, proposal in loaded])
    # np.unique on byte strings distinguishes signed zero and one-ULP changes.
    # Restore first appearance so adding later views cannot reorder old nodes.
    _, first, inverse = np.unique(_bits(combined), return_index=True, return_inverse=True)
    appearance = np.argsort(first)
    remap = np.empty(len(appearance), dtype=np.int64)
    remap[appearance] = np.arange(len(appearance), dtype=np.int64)
    inverse = remap[inverse]
    nodes = combined[first[appearance]].copy()
    directory = Path(directory)
    views, scatters = [], {}
    offset = 0
    for index, (path, manifest, proposal) in enumerate(loaded):
        view_id = proposal.fingerprint
        scatter = inverse[offset:offset + len(proposal.p_values)].astype(np.int64)
        offset += len(proposal.p_values)
        key = f"scatter_{index}"
        scatters[key] = scatter
        views.append(dict(view_id=view_id, source_label=path.parent.name,
            manifest_file=f"views/{view_id}/manifest.json",
            bank_manifest_fingerprint=manifest["fingerprint"],
            proposal_fingerprint=proposal.fingerprint, node_count=len(scatter),
            unique_nodes_used=len(np.unique(scatter)), scatter_key=key,
            scatter_hash=_array_hash(scatter)))
    spec = dict(version=CATALOG_VERSION, is_quadrature=False, complete_rule=False,
        physical=physical, physical_fingerprint=_hash(physical),
        implementation_sources=_sources(), nodes_file="nodes.npz",
        node_count=len(nodes), node_hash=_array_hash(nodes),
        coefficient_count=len(degree_keys(physical["order"])),
        deduplication="exact binary64 tuple bits; first occurrence; no weights or interpolation",
        sum_view_rows=sum(v["node_count"] for v in views), views=views)
    manifest = dict(spec, fingerprint=_hash(spec))
    destination = directory / "catalog.json"
    if destination.exists():
        previous, _, _ = load_catalog(destination)
        if previous != manifest:
            raise ValueError("catalog directory already describes different physical inputs or views")
        return destination
    for path, bank_manifest, proposal in loaded:
        target = directory / "views" / proposal.fingerprint
        for name in ("manifest.json", "proposal.npz"):
            data = (path if name == "manifest.json" else path.parent / name).read_bytes()
            _atomic_file(target / name, lambda handle, content=data: handle.write(content))
    _atomic_file(directory / "nodes.npz", lambda handle: np.savez_compressed(
        handle, p_values=nodes, **scatters))
    _atomic_file(destination, lambda handle: handle.write(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"), "w")
    return destination


def load_catalog(path):
    """Validate catalog, copied complete proposals, exact scatter and sources."""
    path = Path(path)
    manifest = json.loads(path.read_text())
    spec = {k: v for k, v in manifest.items() if k != "fingerprint"}
    if (spec.get("version") != CATALOG_VERSION or spec.get("is_quadrature") is not False
            or spec.get("complete_rule") is not False or manifest.get("fingerprint") != _hash(spec)):
        raise ValueError("catalog format, identity or non-quadrature marker mismatch")
    physical = manifest["physical"]
    if manifest["physical_fingerprint"] != _hash(physical):
        raise ValueError("catalog physical identity mismatch")
    _provider(physical["coefficient_provider"])
    _check_source_dependencies(physical.get("source_dependencies"))
    _check_source_dependencies(manifest["implementation_sources"])
    with np.load(path.parent / manifest["nodes_file"], allow_pickle=False) as data:
        nodes = data["p_values"].copy()
        scatters = {view["scatter_key"]: data[view["scatter_key"]].copy()
                    for view in manifest["views"]}
    if (nodes.dtype != np.dtype(np.float64) or nodes.shape != (manifest["node_count"], 3)
            or not np.all(np.isfinite(nodes)) or np.any(nodes < 0)
            or _array_hash(nodes) != manifest["node_hash"]
            or len(np.unique(_bits(nodes))) != len(nodes)):
        raise ValueError("catalog exact node content/uniqueness mismatch")
    if manifest["coefficient_count"] != len(degree_keys(physical["order"])):
        raise ValueError("catalog coefficient prefix is incomplete")
    views = {}
    used = np.zeros(len(nodes), bool)
    for view in manifest["views"]:
        view_id = view["view_id"]
        if view_id in views:
            raise ValueError("duplicate catalog view identity")
        bank_manifest, proposal = load_bank_build(path.parent / view["manifest_file"])
        scatter = scatters[view["scatter_key"]]
        if (scatter.dtype != np.dtype(np.int64) or scatter.shape != (view["node_count"],)
                or np.any(scatter < 0) or np.any(scatter >= len(nodes))
                or _array_hash(scatter) != view["scatter_hash"]):
            raise ValueError("catalog scatter index content/shape mismatch")
        if (_physical(bank_manifest) != physical
                or bank_manifest["fingerprint"] != view["bank_manifest_fingerprint"]
                or proposal.fingerprint != view_id or proposal.fingerprint != view["proposal_fingerprint"]
                or not _same_bits(nodes[scatter], proposal.p_values)
                or len(np.unique(scatter)) != view["unique_nodes_used"]):
            raise ValueError("catalog view physical/node identity does not exactly match its proposal")
        used[scatter] = True
        views[view_id] = dict(metadata=view, manifest=bank_manifest, proposal=proposal, scatter=scatter)
    if not views or not used.all() or sum(v["metadata"]["node_count"] for v in views.values()) != manifest["sum_view_rows"]:
        raise ValueError("catalog has orphan nodes or inconsistent complete views")
    return manifest, nodes, views


def _shard_hash(metadata, points, coefficients, logs):
    return _hash(dict(metadata={k: v for k, v in metadata.items() if k != "fingerprint"},
        points=_array_hash(points), coefficients=_array_hash(coefficients), dozz=_array_hash(logs)))


def _diagnostics(provenance, count):
    before = provenance.get("coefficient_diagnostics_before", {})
    after = provenance.get("coefficient_diagnostics", {})
    if (before.get("verified_tables") != 0 or after.get("verified_tables") != count
            or provenance.get("coefficient_table_reused") is not False):
        raise ValueError("catalog shard lacks fresh independently verified unique-row evidence")
    for key in ("verified_tables", "weight_limit_count", "maximum_dps_used"):
        if not isinstance(after.get(key), int) or after[key] < 0:
            raise ValueError("invalid catalog coefficient diagnostics")
    if (after["weight_limit_count"] > count
            or not np.isfinite(after.get("maximum_scaled_disagreement", np.nan))
            or after["maximum_scaled_disagreement"] < 0):
        raise ValueError("invalid catalog precision/weight-limit evidence")
    return after


def _read_catalog_shard(path, manifest, nodes):
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(data["metadata"].item())
        points, coefficients, logs = data["p_values"], data["coefficients"], data["log_dozz"]
    if (metadata.get("version") != CATALOG_SHARD_VERSION
            or metadata.get("is_quadrature") is not False or metadata.get("complete_rule") is not False
            or metadata.get("fingerprint") != _shard_hash(metadata, points, coefficients, logs)
            or metadata.get("catalog_fingerprint") != manifest["fingerprint"]
            or metadata.get("physical_fingerprint") != manifest["physical_fingerprint"]):
        raise ValueError("catalog shard identity, physical inputs or content mismatch")
    start, stop = metadata.get("row_start"), metadata.get("row_stop")
    if (not isinstance(start, int) or not isinstance(stop, int)
            or not 0 <= start < stop <= len(nodes) or metadata.get("catalog_node_count") != len(nodes)):
        raise ValueError("invalid catalog shard coverage")
    if not _same_bits(points, nodes[start:stop]):
        raise ValueError("catalog shard nodes do not match exact tuple bits")
    if (coefficients.shape != (stop - start, manifest["coefficient_count"])
            or not np.all(np.isfinite(coefficients))
            or not np.allclose(coefficients[:, 0], 1, atol=1e-12, rtol=0)
            or logs.shape != (stop - start,) or np.any(np.isnan(logs))
            or np.any(np.isposinf(logs.real)) or not np.all(np.isfinite(logs.imag))
            or tuple(map(tuple, metadata.get("coefficient_keys", ()))) != degree_keys(manifest["physical"]["order"])):
        raise ValueError("catalog shard has invalid/incomplete coefficient or DOZZ rows")
    _diagnostics(metadata.get("provenance", {}), stop - start)
    if metadata["provenance"].get("special_dps") != manifest["physical"]["special_dps"]:
        raise ValueError("catalog shard DOZZ precision mismatch")
    return metadata, coefficients, logs


def build_catalog_shard(catalog_path, start, stop, output):
    """Build only unique rows [start,stop); output is never a quadrature bank."""
    manifest, nodes, _ = load_catalog(catalog_path)
    if int(start) != start or int(stop) != stop or not 0 <= start < stop <= len(nodes):
        raise ValueError("nonempty exclusive catalog row interval required")
    start, stop = int(start), int(stop)
    if Path(output).exists():
        old, _, _ = _read_catalog_shard(output, manifest, nodes)
        if (old["row_start"], old["row_stop"]) != (start, stop):
            raise ValueError("existing catalog shard covers a different interval")
        return old
    physical = manifest["physical"]
    # Dummy unit weights exist solely to call the shared physical row builder.
    # complete_rule=False prevents evaluation or saving as a quadrature bank.
    subset = MomentumProposal(nodes[start:stop], np.zeros(stop - start), dict(
        measure="dP1*dP2*dP3/pi^3", domain="catalog rows, not quadrature",
        scheme="exact-node-catalog-internal-builder", complete_rule=False,
        catalog_fingerprint=manifest["fingerprint"]))
    before = time.perf_counter()
    bank = build_momentum_bank(physical["t"], physical["topology"], physical["incoming_slot"],
        subset, physical["order"], coefficient_provider=_provider(physical["coefficient_provider"]),
        special_dps=physical["special_dps"])
    _diagnostics(bank.provenance, stop - start)
    # A long coefficient build must not publish a frozen-source claim after
    # sources or the catalog changed while it was running.
    current, _, _ = load_catalog(catalog_path)
    if current["fingerprint"] != manifest["fingerprint"]:
        raise ValueError("catalog changed during unique-row construction")
    if Path(output).exists():
        existing, _, _ = _read_catalog_shard(output, manifest, nodes)
        if (existing["row_start"], existing["row_stop"]) != (start, stop):
            raise ValueError("concurrent catalog output covers a different interval")
        return existing
    metadata = dict(version=CATALOG_SHARD_VERSION, is_quadrature=False, complete_rule=False,
        catalog_fingerprint=manifest["fingerprint"], physical_fingerprint=manifest["physical_fingerprint"],
        catalog_node_count=len(nodes), row_start=start, row_stop=stop,
        coefficient_keys=bank.coefficient_keys, provenance=bank.provenance,
        build_seconds=time.perf_counter() - before)
    metadata["fingerprint"] = _shard_hash(metadata, subset.p_values,
        bank.coefficient_matrix, bank.log_structure_constants)
    _atomic_file(output, lambda handle: np.savez_compressed(handle,
        metadata=json.dumps(metadata, allow_nan=False), p_values=subset.p_values,
        coefficients=bank.coefficient_matrix, log_dozz=bank.log_structure_constants))
    return metadata


def assemble_catalog(catalog_path, shard_paths, *, selected_views=None, output_directory=None):
    """Restore selected COMPLETE proposals with exact required-row coverage.

    Every supplied shard is checked and overlap is rejected. Rows outside the
    selected views may remain unbuilt; no required row may be missing. Returned
    banks retain their original weights and carry honest catalog reuse evidence.
    No verification count is multiplied by the number of proposal views.
    """
    manifest, nodes, views = load_catalog(catalog_path)
    selected = list(views) if selected_views is None else list(selected_views)
    if not selected or len(selected) != len(set(selected)) or any(v not in views for v in selected):
        raise ValueError("select distinct complete catalog view IDs")
    required = np.unique(np.concatenate([views[v]["scatter"] for v in selected]))
    coefficients = np.empty((len(required), manifest["coefficient_count"]), complex)
    logs = np.empty(len(required), complex)
    present = np.zeros(len(nodes), bool)
    receipts = []
    for path in shard_paths:
        metadata, matrix, dozz = _read_catalog_shard(path, manifest, nodes)
        start, stop = metadata["row_start"], metadata["row_stop"]
        if present[start:stop].any():
            raise ValueError("catalog shard coverage overlaps or duplicates existing rows")
        present[start:stop] = True
        lo, hi = np.searchsorted(required, [start, stop])
        if hi > lo:
            take = required[lo:hi] - start
            coefficients[lo:hi] = matrix[take]
            logs[lo:hi] = dozz[take]
        receipts.append(dict(fingerprint=metadata["fingerprint"], row_start=start,
            row_stop=stop, build_seconds=metadata["build_seconds"],
            coefficient_diagnostics=metadata["provenance"]["coefficient_diagnostics"]))
    if not np.all(present[required]):
        missing = required[~present[required]]
        raise ValueError(f"selected catalog views lack {len(missing)} required unique rows; first missing row {missing[0]}")
    receipts.sort(key=lambda r: r["row_start"])
    ds = [r["coefficient_diagnostics"] for r in receipts]
    merged = dict(verified_tables=sum(d["verified_tables"] for d in ds),
        weight_limit_count=sum(d["weight_limit_count"] for d in ds),
        maximum_dps_used=max(d["maximum_dps_used"] for d in ds),
        maximum_scaled_disagreement=max(d["maximum_scaled_disagreement"] for d in ds),
        scope="all supplied unique-node source shards; not fresh per-view builds")
    banks = {}
    physical = manifest["physical"]
    for view_id in selected:
        view = views[view_id]
        scatter = view["scatter"]
        index = np.searchsorted(required, scatter)
        provenance = dict(node_catalog_fingerprint=manifest["fingerprint"],
            node_catalog_physical_fingerprint=manifest["physical_fingerprint"],
            original_bank_manifest_fingerprint=view["manifest"]["fingerprint"],
            exact_binary64_scatter_hash=_array_hash(scatter),
            complete_view_row_coverage=True, complete_catalog_coverage=bool(present.all()),
            coefficient_table_reused=True, fresh_coefficient_tables_built_during_assembly=0,
            verified_unique_tables_used_by_view=int(len(np.unique(scatter))),
            original_view_row_count=len(scatter), supplied_unique_verified_tables=int(present.sum()),
            coefficient_diagnostics=merged, special_dps=physical["special_dps"],
            source_shards=receipts, reuse_scope="exact physical nodes; original proposal weights restored without interpolation")
        bank = SphereSixPointMomentumBank(physical["t"], physical["topology"], physical["incoming_slot"],
            physical["order"], view["proposal"], coefficients[index], logs[index],
            physical["coefficient_provider"], provenance)
        banks[view_id] = bank
    current, _, _ = load_catalog(catalog_path)
    if current["fingerprint"] != manifest["fingerprint"]:
        raise ValueError("catalog changed during view assembly")
    if output_directory is not None:
        existing_views = set()
        # Preflight every destination before writing any selected view.
        for view_id, bank in banks.items():
            destination = Path(output_directory) / f"{view_id}.npz"
            if destination.exists():
                existing = SphereSixPointMomentumBank.load(destination)
                if existing.fingerprint != bank.fingerprint:
                    raise ValueError("existing assembled output has a different bank fingerprint")
                # Preserve the existing file and its actual source provenance.
                banks[view_id] = existing
                existing_views.add(view_id)
        for view_id, bank in banks.items():
            if view_id not in existing_views:
                bank.save(Path(output_directory) / f"{view_id}.npz")
    return banks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("directory", type=Path)
    prepare.add_argument("bank_manifests", type=Path, nargs="+")
    build = commands.add_parser("build-shard")
    build.add_argument("catalog", type=Path)
    build.add_argument("--start", type=int, required=True)
    build.add_argument("--stop", type=int, required=True)
    build.add_argument("--output", type=Path, required=True)
    assemble = commands.add_parser("assemble")
    assemble.add_argument("catalog", type=Path)
    assemble.add_argument("shards", type=Path, nargs="+")
    assemble.add_argument("--view", action="append", dest="views")
    assemble.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        path = prepare_catalog(args.directory, args.bank_manifests)
        manifest, _, _ = load_catalog(path)
        result = dict(catalog=str(path), fingerprint=manifest["fingerprint"],
            unique_nodes=manifest["node_count"], sum_view_rows=manifest["sum_view_rows"],
            view_ids=[v["view_id"] for v in manifest["views"]], is_quadrature=False)
    elif args.command == "build-shard":
        result = build_catalog_shard(args.catalog, args.start, args.stop, args.output)
    else:
        banks = assemble_catalog(args.catalog, args.shards,
            selected_views=args.views, output_directory=args.output_directory)
        result = {view: dict(path=str(args.output_directory / f"{view}.npz"),
            fingerprint=bank.fingerprint, nodes=len(bank.p_values),
            verified_unique_tables_used=bank.provenance.get("verified_unique_tables_used_by_view"))
            for view, bank in banks.items()}
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()


__all__ = ["prepare_catalog", "load_catalog", "build_catalog_shard", "assemble_catalog",
           "CATALOG_VERSION", "CATALOG_SHARD_VERSION"]
