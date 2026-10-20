"""Parallel, coverage-checked construction of explicit sphere momentum banks.

A manifest fixes all momentum rows, weights, physical inputs and c-recursion
policy before workers run. Worker shards are deliberately a different format
from usable quadrature banks. Assembly requires exact contiguous row coverage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np

from six_point_block_coefficient_provider import VerifiedSphereCProvider
from six_point_momentum_banks_and_reweighting import (
    MomentumProposal, SphereSixPointMomentumBank, build_momentum_bank,
    degree_keys, external_data,
)


BANK_BUILD_VERSION = "sphere-six-point-bank-build-v1"
SHARD_VERSION = "sphere-six-point-momentum-shard-v1"
PROPOSAL_VERSION = "sphere-six-point-momentum-proposal-v1"


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _array_hash(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(str(value.dtype).encode() + str(value.shape).encode()
                          + value.tobytes()).hexdigest()


def _check_source_dependencies(dependencies):
    if dependencies is None:
        return
    if not isinstance(dependencies, dict) or not dependencies:
        raise ValueError("source dependency fingerprints must be a nonempty mapping")
    source_root = Path(__file__).resolve().parents[4]
    for relative, expected in dependencies.items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("source dependency paths must be relative to the staged source root")
        path = source_root / path
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("bank-build source dependency changed: " + relative)


def _atomic_file(path, writer, mode="wb"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode=mode, dir=path.parent,
                prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            writer(handle)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _provider(metadata):
    names = ("base_dps", "check_dps", "maximum_dps", "rtol", "atol",
             "minimum_total_order", "topology_orders")
    provider = VerifiedSphereCProvider(**{key: metadata[key] for key in names})
    if provider.metadata() != metadata:
        raise ValueError("manifest coefficient policy/version does not match this worker")
    if metadata.get("minimum_total_order") != 0 or metadata.get("topology_orders") is not None:
        raise ValueError("parallel banks require complete coefficient prefixes without shell overrides")
    return provider


def prepare_bank_build(directory, *, t, topology, incoming_slot, proposal, order,
                       coefficient_provider=None, special_dps=35, source_dependencies=None):
    """Write a portable manifest and its complete explicit quadrature rule."""
    external_data(t, incoming_slot)
    if topology not in ("comb", "star"):
        raise ValueError("sphere topology must be comb or star")
    degree_keys(order)
    if proposal.metadata.get("complete_rule", True) is not True:
        raise ValueError("a worker shard cannot define a complete momentum proposal")
    if int(special_dps) != special_dps or special_dps < 15:
        raise ValueError("at least fifteen integer DOZZ decimal digits required")
    provider = coefficient_provider or VerifiedSphereCProvider()
    policy = provider.metadata()
    _provider(policy)
    _check_source_dependencies(source_dependencies)
    directory = Path(directory)
    proposal_file = directory / "proposal.npz"
    proposal_metadata = dict(version=PROPOSAL_VERSION, metadata=proposal.metadata,
                             fingerprint=proposal.fingerprint)
    spec = dict(version=BANK_BUILD_VERSION, t=float(t), topology=topology,
                incoming_slot=int(incoming_slot), order=int(order),
                special_dps=int(special_dps), coefficient_provider=policy,
                proposal_file="proposal.npz", proposal_fingerprint=proposal.fingerprint,
                node_count=len(proposal.p_values), coefficient_count=len(degree_keys(order)))
    if source_dependencies is not None:
        spec["source_dependencies"] = dict(source_dependencies)
    manifest = dict(spec, fingerprint=_hash(spec))
    manifest_file = directory / "manifest.json"
    # A prepared directory is reusable only for the identical physical build.
    if manifest_file.exists():
        previous, old_proposal = load_bank_build(manifest_file)
        if previous != manifest or old_proposal.fingerprint != proposal.fingerprint:
            raise ValueError("build directory already describes a different momentum bank")
        return manifest_file
    _atomic_file(proposal_file, lambda handle: np.savez_compressed(handle,
        metadata=json.dumps(proposal_metadata), p_values=proposal.p_values,
        log_weights=proposal.log_weights))
    _atomic_file(manifest_file, lambda handle: handle.write(json.dumps(manifest, indent=2) + "\n"), "w")
    return manifest_file


def load_bank_build(path):
    """Validate the manifest and full explicit node/weight identity."""
    path = Path(path)
    manifest = json.loads(path.read_text())
    fingerprint = manifest.get("fingerprint")
    spec = {key: value for key, value in manifest.items() if key != "fingerprint"}
    if spec.get("version") != BANK_BUILD_VERSION or fingerprint != _hash(spec):
        raise ValueError("bank-build manifest identity/version mismatch")
    _check_source_dependencies(manifest.get("source_dependencies"))
    external_data(manifest["t"], manifest["incoming_slot"])
    if manifest["topology"] not in ("comb", "star"):
        raise ValueError("invalid manifest topology")
    _provider(manifest["coefficient_provider"])
    proposal_path = path.parent / manifest["proposal_file"]
    with np.load(proposal_path, allow_pickle=False) as payload:
        metadata = json.loads(payload["metadata"].item())
        if metadata.get("version") != PROPOSAL_VERSION:
            raise ValueError("explicit momentum proposal format mismatch")
        proposal = MomentumProposal(payload["p_values"], payload["log_weights"], metadata["metadata"])
    if (proposal.fingerprint != metadata.get("fingerprint")
            or proposal.fingerprint != manifest["proposal_fingerprint"]
            or len(proposal.p_values) != manifest["node_count"]
            or proposal.metadata.get("complete_rule", True) is not True):
        raise ValueError("manifest momentum proposal content mismatch")
    if manifest["coefficient_count"] != len(degree_keys(manifest["order"])):
        raise ValueError("manifest coefficient degree count mismatch")
    return manifest, proposal


def _shard_fingerprint(metadata, points, weights, coefficients, log_dozz):
    return _hash(dict(metadata={k: v for k, v in metadata.items() if k != "fingerprint"},
        points=_array_hash(points), weights=_array_hash(weights),
        coefficients=_array_hash(coefficients), log_dozz=_array_hash(log_dozz)))


def build_shard(manifest_path, start, stop, output):
    """Build exclusive rows [start,stop), or reuse that exact validated shard."""
    manifest, proposal = load_bank_build(manifest_path)
    if (int(start) != start or int(stop) != stop
            or not 0 <= start < stop <= len(proposal.p_values)):
        raise ValueError("a nonempty valid exclusive momentum-row range is required")
    start, stop = int(start), int(stop)
    if Path(output).exists():
        metadata, _, _ = _read_shard(output, manifest, proposal)
        if (metadata["row_start"], metadata["row_stop"]) != (start, stop):
            raise ValueError("existing worker output covers a different momentum row interval")
        return metadata
    subset = MomentumProposal(proposal.p_values[start:stop], proposal.log_weights[start:stop],
        dict(measure="dP1*dP2*dP3/pi^3", scheme="explicit-build-shard",
             domain="subset; not a quadrature rule", complete_rule=False,
             full_proposal_fingerprint=proposal.fingerprint, row_start=start, row_stop=stop))
    before = time.perf_counter()
    bank = build_momentum_bank(manifest["t"], manifest["topology"], manifest["incoming_slot"],
        subset, manifest["order"], coefficient_provider=_provider(manifest["coefficient_provider"]),
        special_dps=manifest["special_dps"])
    metadata = dict(version=SHARD_VERSION, complete_rule=False,
        manifest_fingerprint=manifest["fingerprint"], proposal_fingerprint=proposal.fingerprint,
        t=manifest["t"], topology=manifest["topology"], incoming_slot=manifest["incoming_slot"],
        order=manifest["order"], special_dps=manifest["special_dps"],
        coefficient_provider=manifest["coefficient_provider"], row_start=start, row_stop=stop,
        full_node_count=len(proposal.p_values), coefficient_keys=bank.coefficient_keys,
        provenance=bank.provenance, build_seconds=time.perf_counter() - before)
    metadata["fingerprint"] = _shard_fingerprint(metadata, subset.p_values,
                        subset.log_weights, bank.coefficient_matrix, bank.log_structure_constants)
    _atomic_file(output, lambda handle: np.savez_compressed(handle, metadata=json.dumps(metadata),
        p_values=subset.p_values, log_weights=subset.log_weights,
        coefficients=bank.coefficient_matrix, log_dozz=bank.log_structure_constants))
    return metadata


def _read_shard(path, manifest, proposal):
    with np.load(path, allow_pickle=False) as payload:
        metadata = json.loads(payload["metadata"].item())
        points, weights = payload["p_values"], payload["log_weights"]
        coefficients, logs = payload["coefficients"], payload["log_dozz"]
    if metadata.get("version") != SHARD_VERSION or metadata.get("complete_rule") is not False:
        raise ValueError("a dedicated incomplete worker-shard format is required")
    if metadata.get("fingerprint") != _shard_fingerprint(metadata, points, weights, coefficients, logs):
        raise ValueError("worker shard content fingerprint mismatch")
    expected = dict(manifest_fingerprint=manifest["fingerprint"],
        proposal_fingerprint=proposal.fingerprint, t=manifest["t"], topology=manifest["topology"],
        incoming_slot=manifest["incoming_slot"], order=manifest["order"],
        special_dps=manifest["special_dps"], coefficient_provider=manifest["coefficient_provider"],
        full_node_count=len(proposal.p_values))
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("worker shard physical inputs, proposal or coefficient policy mismatch")
    start, stop = metadata["row_start"], metadata["row_stop"]
    if not (isinstance(start, int) and isinstance(stop, int)
            and 0 <= start < stop <= len(proposal.p_values)):
        raise ValueError("invalid worker shard row interval")
    if (not np.array_equal(points, proposal.p_values[start:stop])
            or not np.array_equal(weights, proposal.log_weights[start:stop])):
        raise ValueError("worker shard does not contain the exact specified proposal rows/weights")
    if tuple(map(tuple, metadata["coefficient_keys"])) != degree_keys(manifest["order"]):
        raise ValueError("worker shard coefficient degrees do not match the complete prefix")
    if coefficients.shape != (stop - start, manifest["coefficient_count"]) or logs.shape != (stop - start,):
        raise ValueError("worker shard coefficient/DOZZ shapes do not match its row coverage")
    diagnostics = metadata["provenance"]["coefficient_diagnostics"]
    if diagnostics.get("verified_tables") != stop - start:
        raise ValueError("worker shard lacks one verified c-recursion table per momentum row")
    if metadata["provenance"].get("coefficient_table_reused") is not False:
        raise ValueError("fresh worker shards cannot claim archived coefficient reuse")
    return metadata, coefficients, logs


def assemble_bank(manifest_path, shard_paths, output=None):
    """Assemble exactly one complete coverage of the manifest's ordered rows."""
    manifest, proposal = load_bank_build(manifest_path)
    if not shard_paths:
        raise ValueError("at least one worker shard is required")
    shards = [_read_shard(path, manifest, proposal) for path in shard_paths]
    shards.sort(key=lambda value: value[0]["row_start"])
    expected_start = 0
    for metadata, _, _ in shards:
        if metadata["row_start"] != expected_start:
            raise ValueError("worker shard coverage contains a gap, overlap or duplicate")
        expected_start = metadata["row_stop"]
    if expected_start != len(proposal.p_values):
        raise ValueError("worker shard coverage omits the final momentum rows")
    diagnostics = [item[0]["provenance"]["coefficient_diagnostics"] for item in shards]
    merged = dict(verified_tables=sum(d["verified_tables"] for d in diagnostics),
        weight_limit_count=sum(d["weight_limit_count"] for d in diagnostics),
        maximum_dps_used=max(d["maximum_dps_used"] for d in diagnostics),
        maximum_scaled_disagreement=max(d["maximum_scaled_disagreement"] for d in diagnostics))
    provenance = dict(parallel_build_manifest_fingerprint=manifest["fingerprint"],
        complete_contiguous_row_coverage=True, coefficient_diagnostics=merged,
        special_dps=manifest["special_dps"], coefficient_table_reused=False,
        shards=[dict(fingerprint=item[0]["fingerprint"], row_start=item[0]["row_start"],
                     row_stop=item[0]["row_stop"], build_seconds=item[0]["build_seconds"],
                     coefficient_diagnostics=item[0]["provenance"]["coefficient_diagnostics"])
                for item in shards])
    bank = SphereSixPointMomentumBank(manifest["t"], manifest["topology"], manifest["incoming_slot"],
        manifest["order"], proposal, np.concatenate([item[1] for item in shards]),
        np.concatenate([item[2] for item in shards]), manifest["coefficient_provider"], provenance)
    if output is not None:
        bank.save(output)
    return bank


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-shard", help="build exclusive manifest rows [start,stop)")
    build.add_argument("manifest", type=Path)
    build.add_argument("--start", type=int, required=True)
    build.add_argument("--stop", type=int, required=True)
    build.add_argument("--output", type=Path, required=True)
    assemble = commands.add_parser("assemble", help="require complete nonoverlapping row coverage")
    assemble.add_argument("manifest", type=Path)
    assemble.add_argument("shards", type=Path, nargs="+")
    assemble.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build-shard":
        metadata = build_shard(args.manifest, args.start, args.stop, args.output)
        result = dict(shard=str(args.output), fingerprint=metadata["fingerprint"],
                      row_start=metadata["row_start"], row_stop=metadata["row_stop"],
                      build_seconds=metadata["build_seconds"], complete_rule=False)
    else:
        bank = assemble_bank(args.manifest, args.shards, args.output)
        result = dict(bank=str(args.output), fingerprint=bank.fingerprint,
                      node_count=len(bank.p_values), order=bank.order,
                      complete_rule=True, coefficient_diagnostics=bank.provenance["coefficient_diagnostics"])
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()


__all__ = ["prepare_bank_build", "load_bank_build", "build_shard", "assemble_bank",
           "BANK_BUILD_VERSION", "SHARD_VERSION"]
