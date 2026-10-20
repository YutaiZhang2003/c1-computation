"""Exact momentum-node catalogs with a different complete degree per view.

Each binary64 tuple is evaluated once at its maximum requested degree. Shards
store variable-length COMPLETE coefficient rows, never padded or inferred
coefficients. Assembly restores the original proposal weights and requested
prefix. A catalog/shard is not a quadrature rule or a convergence certificate.
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
from six_point_deduplicate_momentum_nodes import _bits, _same_bits, _diagnostics
from six_point_momentum_banks_and_reweighting import (
    MomentumProposal, SphereSixPointMomentumBank, build_momentum_bank, degree_keys,
)

CATALOG_VERSION = "sphere-six-point-multilevel-catalog-v1"
SHARD_VERSION = "sphere-six-point-multilevel-catalog-shard-v1"
_PHYSICAL_KEYS = ("t", "topology", "incoming_slot", "special_dps",
                  "coefficient_provider", "source_dependencies")


def _sources():
    root = Path(__file__).resolve().parents[4]
    modules = ("six_point_build_momentum_banks", "six_point_deduplicate_momentum_nodes",
               "six_point_momentum_banks_and_reweighting", "six_point_block_coefficient_provider",
               "six_point_arbitrary_precision_c_recursion", "liouville_torus",
               "liouville_momentum_quadrature", "c1_paths", "six_point_reuse_saved_coefficients")
    paths = [Path(__file__).resolve()] + [Path(importlib.import_module(m).__file__).resolve()
                                         for m in modules]
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def _physical(manifest):
    return {name: manifest.get(name) for name in _PHYSICAL_KEYS}


def _histogram(orders):
    keys, counts = np.unique(orders, return_counts=True)
    return {str(int(k)): int(n) for k, n in zip(keys, counts)}


def _offsets(orders):
    counts = {int(n): len(degree_keys(int(n))) for n in np.unique(orders)}
    return np.concatenate((np.zeros(1, dtype=np.int64),
        np.cumsum([counts[int(n)] for n in orders], dtype=np.int64)))


def prepare_catalog(directory, bank_manifests):
    """Prepare nodes only. View IDs are original BANK manifest fingerprints.

    The same proposal requested at two degrees produces two views. Input order
    determines stable first-occurrence node order; overlapping views assign
    the exact tuple the maximum requested degree, including 14/16 overlays.
    """
    paths = [Path(p) for p in bank_manifests]
    if not paths:
        raise ValueError("at least one complete bank manifest is required")
    loaded = [(path, *load_bank_build(path)) for path in paths]
    physical = _physical(loaded[0][1])
    if any(_physical(m) != physical for _, m, _ in loaded):
        raise ValueError("multilevel catalog cannot mix energy, graph, slot, provider, DOZZ precision or sources")
    unique = {}
    for path, manifest, proposal in loaded:
        if manifest["proposal_file"] != "proposal.npz":
            raise ValueError("portable standard proposal.npz layout required")
        unique.setdefault(manifest["fingerprint"], (path, manifest, proposal))
    loaded = list(unique.values())
    combined = np.concatenate([p.p_values for _, _, p in loaded])
    _, first, inverse = np.unique(_bits(combined), return_index=True, return_inverse=True)
    appearance = np.argsort(first)
    remap = np.empty(len(appearance), dtype=np.int64)
    remap[appearance] = np.arange(len(appearance), dtype=np.int64)
    inverse = remap[inverse]
    nodes = combined[first[appearance]].copy()
    orders = np.full(len(nodes), -1, dtype=np.int64)
    views, scatters, cursor = [], {}, 0
    for index, (path, manifest, proposal) in enumerate(loaded):
        scatter = inverse[cursor:cursor + len(proposal.p_values)].astype(np.int64)
        cursor += len(scatter)
        np.maximum.at(orders, scatter, manifest["order"])
        view_id, key = manifest["fingerprint"], f"scatter_{index}"
        scatters[key] = scatter
        views.append(dict(view_id=view_id, source_label=path.parent.name,
            manifest_file=f"views/{view_id}/manifest.json",
            bank_manifest_fingerprint=manifest["fingerprint"],
            proposal_fingerprint=proposal.fingerprint, requested_order=manifest["order"],
            coefficient_count=manifest["coefficient_count"], node_count=len(scatter),
            unique_nodes_used=len(np.unique(scatter)), scatter_key=key,
            scatter_hash=_array_hash(scatter)))
    spec = dict(version=CATALOG_VERSION, is_quadrature=False, complete_rule=False,
        physical=physical, physical_fingerprint=_hash(physical), implementation_sources=_sources(),
        nodes_file="nodes.npz", node_count=len(nodes), node_hash=_array_hash(nodes),
        node_max_orders_hash=_array_hash(orders), build_order_histogram=_histogram(orders),
        maximum_order=int(orders.max()), sum_view_rows=sum(v["node_count"] for v in views),
        total_stored_coefficient_count=int(_offsets(orders)[-1]), views=views,
        coefficient_key_order="ascending (total degree,a,b,c); complete prefix per row",
        deduplication="exact binary64 momentum bits; first occurrence; maximum requested degree; no interpolation")
    manifest = dict(spec, fingerprint=_hash(spec))
    directory = Path(directory)
    destination = directory / "catalog.json"
    if destination.exists():
        previous, _, _ = load_catalog(destination)
        if previous != manifest:
            raise ValueError("catalog directory already describes different inputs/views")
        return destination
    for path, original, _ in loaded:
        target = directory / "views" / original["fingerprint"]
        for name in ("manifest.json", "proposal.npz"):
            data = (path if name == "manifest.json" else path.parent / name).read_bytes()
            _atomic_file(target / name, lambda handle, content=data: handle.write(content))
    _atomic_file(directory / "nodes.npz", lambda handle: np.savez_compressed(
        handle, p_values=nodes, node_max_orders=orders, **scatters))
    _atomic_file(destination, lambda handle: handle.write(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"), "w")
    load_catalog(destination)
    return destination


def load_catalog(path):
    """Validate all source/node/degree/scatter identities; return manifest,nodes,views.

    Per-node build orders are stored as int64 ``node_max_orders`` in nodes.npz.
    The manifest exposes their verified histogram and hash for job planning.
    """
    path = Path(path)
    manifest = json.loads(path.read_text())
    spec = {k: v for k, v in manifest.items() if k != "fingerprint"}
    if (spec.get("version") != CATALOG_VERSION or spec.get("is_quadrature") is not False
            or spec.get("complete_rule") is not False or manifest.get("fingerprint") != _hash(spec)):
        raise ValueError("invalid multilevel catalog identity/format")
    physical = manifest["physical"]
    if set(physical) != set(_PHYSICAL_KEYS) or manifest["physical_fingerprint"] != _hash(physical):
        raise ValueError("multilevel physical identity mismatch")
    _provider(physical["coefficient_provider"])
    _check_source_dependencies(physical.get("source_dependencies"))
    if manifest.get("implementation_sources") != _sources():
        raise ValueError("multilevel implementation source closure changed")
    with np.load(path.parent / manifest["nodes_file"], allow_pickle=False) as data:
        nodes, orders = data["p_values"].copy(), data["node_max_orders"].copy()
        scatters = {v["scatter_key"]: data[v["scatter_key"]].copy() for v in manifest["views"]}
    if (nodes.dtype != np.float64 or nodes.shape != (manifest["node_count"], 3)
            or not np.all(np.isfinite(nodes)) or np.any(nodes < 0)
            or _array_hash(nodes) != manifest["node_hash"]
            or len(np.unique(_bits(nodes))) != len(nodes)):
        raise ValueError("multilevel exact-node identity/uniqueness mismatch")
    if (orders.dtype != np.int64 or orders.shape != (len(nodes),) or np.any(orders < 0)
            or _array_hash(orders) != manifest["node_max_orders_hash"]):
        raise ValueError("multilevel build degree identity mismatch")
    views, needed = {}, np.full(len(nodes), -1, dtype=np.int64)
    for view in manifest["views"]:
        view_id = view["view_id"]
        if view_id in views or view["manifest_file"] != f"views/{view_id}/manifest.json":
            raise ValueError("duplicate or nonportable multilevel view")
        original, proposal = load_bank_build(path.parent / view["manifest_file"])
        scatter = scatters[view["scatter_key"]]
        if (scatter.dtype != np.int64 or scatter.shape != (view["node_count"],)
                or np.any(scatter < 0) or np.any(scatter >= len(nodes))
                or _array_hash(scatter) != view["scatter_hash"]):
            raise ValueError("invalid multilevel scatter")
        if (_physical(original) != physical or original["fingerprint"] != view_id
                or view_id != view["bank_manifest_fingerprint"]
                or original["order"] != view["requested_order"]
                or original["coefficient_count"] != view["coefficient_count"]
                or proposal.fingerprint != view["proposal_fingerprint"]
                or not _same_bits(nodes[scatter], proposal.p_values)
                or len(np.unique(scatter)) != view["unique_nodes_used"]):
            raise ValueError("multilevel view differs from original physical inputs, nodes or degree")
        np.maximum.at(needed, scatter, original["order"])
        views[view_id] = dict(metadata=view, manifest=original, proposal=proposal, scatter=scatter)
    if (not views or not np.array_equal(orders, needed)
            or _histogram(orders) != manifest["build_order_histogram"]
            or int(orders.max()) != manifest["maximum_order"]
            or int(_offsets(orders)[-1]) != manifest["total_stored_coefficient_count"]
            or sum(v["metadata"]["node_count"] for v in views.values()) != manifest["sum_view_rows"]):
        raise ValueError("multilevel maximum degrees or complete-view coverage are inconsistent")
    if "producer_import" in manifest:
        from six_point_reuse_saved_coefficients import validate_spec
        validate_spec(manifest["producer_import"], manifest, nodes, orders, path)
    return manifest, nodes, views


def _orders(path, manifest):
    with np.load(Path(path).parent / manifest["nodes_file"], allow_pickle=False) as data:
        orders = data["node_max_orders"].copy()
    if _array_hash(orders) != manifest["node_max_orders_hash"]:
        raise ValueError("node degrees changed after catalog validation")
    return orders


def _shard_hash(metadata, points, orders, offsets, coefficients, logs):
    return _hash(dict(metadata={k: v for k, v in metadata.items() if k != "fingerprint"},
        points=_array_hash(points), orders=_array_hash(orders), offsets=_array_hash(offsets),
        coefficients=_array_hash(coefficients), dozz=_array_hash(logs)))


def _read_shard(path, manifest, nodes, orders):
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(data["metadata"].item())
        points, row_orders = data["p_values"], data["row_max_orders"]
        offsets, coefficients, logs = data["row_offsets"], data["coefficients_flat"], data["log_dozz"]
    if (metadata.get("version") != SHARD_VERSION or metadata.get("is_quadrature") is not False
            or metadata.get("complete_rule") is not False
            or metadata.get("catalog_fingerprint") != manifest["fingerprint"]
            or metadata.get("physical_fingerprint") != manifest["physical_fingerprint"]
            or metadata.get("fingerprint") != _shard_hash(metadata, points, row_orders, offsets, coefficients, logs)):
        raise ValueError("multilevel shard identity/content mismatch")
    start, stop = metadata.get("row_start"), metadata.get("row_stop")
    if (type(start) is not int or type(stop) is not int or not 0 <= start < stop <= len(nodes)
            or metadata.get("catalog_node_count") != len(nodes)):
        raise ValueError("invalid multilevel shard interval")
    expected = orders[start:stop]
    if (points.dtype != np.float64 or not _same_bits(points, nodes[start:stop])
            or row_orders.dtype != np.int64 or not np.array_equal(row_orders, expected)
            or offsets.dtype != np.int64 or not np.array_equal(offsets, _offsets(expected))):
        raise ValueError("multilevel shard nodes, build degrees or row boundaries differ")
    if (coefficients.dtype != np.complex128 or coefficients.shape != (int(offsets[-1]),)
            or not np.all(np.isfinite(coefficients)) or not np.all(coefficients[offsets[:-1]] == 1)
            or logs.dtype != np.complex128 or logs.shape != (stop-start,)
            or np.any(np.isnan(logs)) or np.any(np.isposinf(logs.real)) or not np.all(np.isfinite(logs.imag))):
        raise ValueError("multilevel shard lacks finite complete coefficient/DOZZ rows")
    groups = metadata.get("degree_groups", [])
    if len(groups) != len(np.unique(expected)):
        raise ValueError("multilevel degree-group evidence is incomplete")
    seen = set()
    for group in groups:
        order = group["order"]
        local = np.flatnonzero(expected == order).tolist()
        if (type(order) is not int or order in seen or not local
                or group.get("local_row_indices") != local or group.get("row_count") != len(local)
                or tuple(map(tuple, group.get("coefficient_keys", ()))) != degree_keys(order)
                or group["provenance"].get("special_dps") != manifest["physical"]["special_dps"]):
            raise ValueError("multilevel degree-group verification scope mismatch")
        _diagnostics(group["provenance"], len(local))
        seen.add(order)
    from six_point_reuse_saved_coefficients import validate_imported_arrays
    validate_imported_arrays(manifest, metadata, dict(p_values=points, row_max_orders=row_orders,
        row_offsets=offsets, coefficients_flat=coefficients, log_dozz=logs))
    return metadata, offsets, coefficients, logs


def build_shard(catalog_path, start, stop, output):
    """Build every unique row once, at its maximum degree, within [start,stop)."""
    manifest, nodes, _ = load_catalog(catalog_path)
    orders = _orders(catalog_path, manifest)
    if isinstance(start, bool) or isinstance(stop, bool) or int(start) != start or int(stop) != stop or not 0 <= start < stop <= len(nodes):
        raise ValueError("nonempty exclusive multilevel row interval required")
    start, stop = int(start), int(stop)
    if Path(output).exists():
        old, _, _, _ = _read_shard(output, manifest, nodes, orders)
        if (old["row_start"], old["row_stop"]) != (start, stop):
            raise ValueError("existing multilevel shard covers a different interval")
        return old
    from six_point_reuse_saved_coefficients import record_for
    if record_for(manifest, start, stop) is not None:
        raise ValueError("approved imported shard is missing; materialize it before construction")
    physical, row_orders = manifest["physical"], orders[start:stop]
    offsets = _offsets(row_orders)
    coefficients = np.empty(int(offsets[-1]), dtype=np.complex128)
    logs, groups = np.empty(stop-start, dtype=np.complex128), []
    before = time.perf_counter()
    for order in np.unique(row_orders):
        local = np.flatnonzero(row_orders == order)
        proposal = MomentumProposal(nodes[start+local], np.zeros(len(local)), dict(
            measure="dP1*dP2*dP3/pi^3", domain="unique catalog nodes; not quadrature",
            scheme="multilevel-catalog-internal-builder", complete_rule=False,
            catalog_fingerprint=manifest["fingerprint"]))
        bank = build_momentum_bank(physical["t"], physical["topology"], physical["incoming_slot"],
            proposal, int(order), coefficient_provider=_provider(physical["coefficient_provider"]),
            special_dps=physical["special_dps"])
        _diagnostics(bank.provenance, len(local))
        for j, i in enumerate(local):
            coefficients[offsets[i]:offsets[i+1]] = bank.coefficient_matrix[j]
        logs[local] = bank.log_structure_constants
        groups.append(dict(order=int(order), local_row_indices=local.tolist(), row_count=len(local),
                           coefficient_keys=bank.coefficient_keys, provenance=bank.provenance))
    current, _, _ = load_catalog(catalog_path)
    if current["fingerprint"] != manifest["fingerprint"]:
        raise ValueError("multilevel catalog changed during row construction")
    if Path(output).exists():
        old, _, _, _ = _read_shard(output, manifest, nodes, orders)
        if (old["row_start"], old["row_stop"]) != (start, stop):
            raise ValueError("concurrent multilevel output covers a different interval")
        return old
    metadata = dict(version=SHARD_VERSION, is_quadrature=False, complete_rule=False,
        catalog_fingerprint=manifest["fingerprint"], physical_fingerprint=manifest["physical_fingerprint"],
        catalog_node_count=len(nodes), row_start=start, row_stop=stop, degree_groups=groups,
        build_seconds=time.perf_counter()-before)
    metadata["fingerprint"] = _shard_hash(metadata, nodes[start:stop], row_orders, offsets, coefficients, logs)
    _atomic_file(output, lambda handle: np.savez_compressed(handle,
        metadata=json.dumps(metadata, allow_nan=False), p_values=nodes[start:stop],
        row_max_orders=row_orders, row_offsets=offsets, coefficients_flat=coefficients, log_dozz=logs))
    return metadata


def assemble_views(catalog_path, shard_paths, *, selected_views=None, output_directory=None):
    """Return ordinary complete banks keyed by original bank-manifest view IDs.

    Every required unique row must exist at its declared maximum build degree.
    Extra irrelevant missing rows are allowed only for selected complete views.
    Supplied duplicate/overlapping shards are always rejected. Output files are
    ``<view_id>.npz`` and existing unequal banks are never overwritten.
    """
    manifest, nodes, views = load_catalog(catalog_path)
    orders = _orders(catalog_path, manifest)
    selected = list(views) if selected_views is None else list(selected_views)
    if not selected or len(selected) != len(set(selected)) or any(v not in views for v in selected):
        raise ValueError("select distinct complete multilevel view IDs")
    required = np.unique(np.concatenate([views[v]["scatter"] for v in selected]))
    offsets = _offsets(orders[required])
    coefficients = np.empty(int(offsets[-1]), dtype=np.complex128)
    logs = np.empty(len(required), dtype=np.complex128)
    present, receipts, diagnostics = np.zeros(len(nodes), bool), [], []
    for path in shard_paths:
        metadata, row_offsets, flat, dozz = _read_shard(path, manifest, nodes, orders)
        start, stop = metadata["row_start"], metadata["row_stop"]
        if present[start:stop].any():
            raise ValueError("multilevel shard intervals overlap")
        present[start:stop] = True
        lo, hi = np.searchsorted(required, [start, stop])
        for i in range(lo, hi):
            j = required[i]-start
            coefficients[offsets[i]:offsets[i+1]] = flat[row_offsets[j]:row_offsets[j+1]]
            logs[i] = dozz[j]
        receipt = dict(fingerprint=metadata["fingerprint"], row_start=start, row_stop=stop,
                       build_seconds=metadata["build_seconds"], degree_groups=metadata["degree_groups"])
        if "import_origin" in metadata:
            receipt["import_origin"] = metadata["import_origin"]
        receipts.append(receipt)
        diagnostics.extend(g["provenance"]["coefficient_diagnostics"] for g in metadata["degree_groups"])
    if not np.all(present[required]):
        raise ValueError(f"multilevel views lack {int((~present[required]).sum())} required unique rows")
    receipts.sort(key=lambda r: r["row_start"])
    merged = dict(verified_tables=sum(d["verified_tables"] for d in diagnostics),
        weight_limit_count=sum(d["weight_limit_count"] for d in diagnostics),
        maximum_dps_used=max(d["maximum_dps_used"] for d in diagnostics),
        maximum_scaled_disagreement=max(d["maximum_scaled_disagreement"] for d in diagnostics),
        scope="all supplied unique-node source shards at their declared maximum degrees; not fresh per-view builds")
    physical, banks = manifest["physical"], {}
    for view_id in selected:
        view, scatter = views[view_id], views[view_id]["scatter"]
        order = view["manifest"]["order"]
        count = len(degree_keys(order))
        index = np.searchsorted(required, scatter)
        matrix = np.stack([coefficients[offsets[i]:offsets[i]+count] for i in index])
        provenance = dict(multilevel_catalog_fingerprint=manifest["fingerprint"],
            multilevel_catalog_physical_fingerprint=manifest["physical_fingerprint"],
            original_bank_manifest_fingerprint=view_id, requested_order=order,
            exact_binary64_scatter_hash=_array_hash(scatter), complete_view_row_coverage=True,
            complete_catalog_coverage=bool(present.all()), coefficient_table_reused=True,
            fresh_coefficient_tables_built_during_assembly=0,
            verified_unique_tables_used_by_view=len(np.unique(scatter)),
            source_max_order_counts=_histogram(orders[np.unique(scatter)]),
            original_view_row_count=len(scatter), supplied_unique_verified_tables=int(present.sum()),
            coefficient_diagnostics=merged, special_dps=physical["special_dps"], source_shards=receipts,
            reuse_scope="exact physical nodes; complete lower-degree prefix; original proposal weights; no interpolation")
        banks[view_id] = SphereSixPointMomentumBank(physical["t"], physical["topology"], physical["incoming_slot"],
            order, view["proposal"], matrix, logs[index], physical["coefficient_provider"], provenance)
    current, _, _ = load_catalog(catalog_path)
    if current["fingerprint"] != manifest["fingerprint"]:
        raise ValueError("multilevel catalog changed during assembly")
    if output_directory is not None:
        existing = set()
        for view_id, bank in banks.items():
            target = Path(output_directory) / f"{view_id}.npz"
            if target.exists():
                old = SphereSixPointMomentumBank.load(target)
                if old.fingerprint != bank.fingerprint:
                    raise ValueError("existing multilevel view output differs")
                banks[view_id] = old
                existing.add(view_id)
        for view_id, bank in banks.items():
            if view_id not in existing:
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
        output = prepare_catalog(args.directory, args.bank_manifests)
        result = dict(catalog=str(output), fingerprint=load_catalog(output)[0]["fingerprint"])
    elif args.command == "build-shard":
        result = build_shard(args.catalog, args.start, args.stop, args.output)
    else:
        banks = assemble_views(args.catalog, args.shards, selected_views=args.views,
                               output_directory=args.output_directory)
        result = {v: dict(path=str(args.output_directory / f"{v}.npz"), order=b.order,
                          fingerprint=b.fingerprint) for v, b in banks.items()}
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()


__all__ = ["prepare_catalog", "load_catalog", "build_shard", "assemble_views",
           "CATALOG_VERSION", "SHARD_VERSION"]
