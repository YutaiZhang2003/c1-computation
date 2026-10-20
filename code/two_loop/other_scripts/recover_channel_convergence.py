"""Recover observed Liouville convergence changes from fixed-geometry receipts."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from check_channel_convergence import ROOT, digest, load_baseline


def maximum_available(values):
    available = [abs(value) for value in values if value is not None]
    return max(available) if available else None


def recover(receipts, output):
    receipts, output = Path(receipts), Path(output)
    data = receipts.parent
    source_path = data / "source_manifest.json"
    source_hash = digest(source_path)
    source = json.loads(source_path.read_text())
    driver = next(item for item in source["files"]
                  if item["path"].endswith("/check_channel_convergence.py"))
    if digest(data / "driver.py") != driver["sha256"]:
        raise ValueError("Changed archived convergence driver")
    baselines, computed = {}, {}
    for path in sorted(receipts.glob("*.json")):
        record = json.loads(path.read_text())
        pair = (record["row"], record["channel"])
        if pair not in baselines:
            baselines[pair] = load_baseline(*pair)
        baseline_path, baseline, sources = baselines[pair]
        if (not record["completed"]
                or record["source_manifest_sha256"] != source_hash
                or record["driver_sha256"] != driver["sha256"]
                or record["baseline_sha256"] != digest(baseline_path)
                or record["target"] != baseline["target"]
                or record["target_sha256"] != baseline["target_sha256"]
                or record["chart"] != baseline["chart"]
                or record["scalar_partition"] != baseline["scalar"]["partition"]
                or record["unchanged_numerical_sources"] != sources):
            raise ValueError(f"Inconsistent convergence receipt: {path}")
        settings = record["settings"]
        if settings["quadrature_scheme"] != "primary-gaussian":
            raise ValueError("Convergence run changed the node-placement rule")
        key = pair + (settings["block_order"],
                      settings["momentum_nodes_per_edge"], settings["dps"])
        if key in computed:
            raise ValueError(f"Duplicate convergence point: {key}")
        liouville = record["liouville"]
        log_ql = (math.log(liouville["scaled_partition"])
                  + liouville["log_threshold_factor"]
                  - 25 * math.log(record["scalar_partition"]))
        delta = math.expm1(log_ql - baseline["log_Q_L"])
        if (not math.isclose(log_ql, record["log_Q_L"], rel_tol=0, abs_tol=1e-12)
                or not math.isclose(math.exp(log_ql), record["Q_L"], rel_tol=2e-12)
                or not math.isclose(delta, record["relative_change_from_N12_n12"],
                                    rel_tol=0, abs_tol=2e-12)):
            raise ValueError(f"Inconsistent partition arithmetic: {path}")
        computed[key] = dict(row=pair[0], channel=pair[1],
                             block_order=key[2], nodes_per_edge=key[3], dps=key[4],
                             Q_L=math.exp(log_ql), log_Q_L=log_ql,
                             relative_change_from_N12_n12=delta)
    required = {(row, channel, order, nodes, 50)
                for row in range(1, 6) for channel in ("glasses", "theta")
                for order, nodes in ((12, 16),)}
    required.add((1, "glasses", 12, 12, 50))
    if not required.issubset(computed):
        raise ValueError(f"Missing convergence points: {sorted(required - computed.keys())}")
    control = computed[(1, "glasses", 12, 12, 50)]
    if abs(control["relative_change_from_N12_n12"]) > 2e-12:
        raise ValueError("The fixed-geometry driver did not recover its baseline")
    comparisons = []
    for row in range(1, 6):
        item = dict(row=row, channel_difference_N12_n12=math.expm1(
            baselines[(row, "theta")][1]["log_Q_L"]
            - baselines[(row, "glasses")][1]["log_Q_L"]))
        for channel in ("glasses", "theta"):
            item[f"{channel}_nodes_12_to_16"] = computed[
                (row, channel, 12, 16, 50)]["relative_change_from_N12_n12"]
            item[f"{channel}_block_12_to_18"] = computed.get(
                (row, channel, 18, 12, 50), {}).get("relative_change_from_N12_n12")
        item["channel_difference_N12_n16"] = math.expm1(
            computed[(row, "theta", 12, 16, 50)]["log_Q_L"]
            - computed[(row, "glasses", 12, 16, 50)]["log_Q_L"])
        item["channel_difference_N18_n12"] = (
            math.expm1(computed[(row, "theta", 18, 12, 50)]["log_Q_L"]
                       - computed[(row, "glasses", 18, 12, 50)]["log_Q_L"])
            if all((row, channel, 18, 12, 50) in computed
                   for channel in ("theta", "glasses")) else None)
        comparisons.append(item)
    report = dict(
        format="c1-channel-convergence-recovery-v1",
        scope="Observed Liouville changes at fixed saved geometry and scalar normalization",
        rigorous_error_bound=None, fixed_geometry_baseline_recovered=True,
        receipt_count=len(computed), comparisons=comparisons,
        missing_block_checks=[
            dict(row=row, channel=channel, block_order=18, nodes_per_edge=12)
            for row in range(1, 6) for channel in ("theta", "glasses")
            if (row, channel, 18, 12, 50) not in computed],
        maximum_observed_node_change=max(
            abs(item[f"{channel}_nodes_12_to_16"])
            for item in comparisons for channel in ("glasses", "theta")),
        maximum_observed_block_change=maximum_available(
            item[f"{channel}_block_12_to_18"]
            for item in comparisons for channel in ("glasses", "theta")),
        maximum_channel_difference_N12_n16=max(
            abs(item["channel_difference_N12_n16"]) for item in comparisons),
        maximum_channel_difference_N18_n12=maximum_available(
            item["channel_difference_N18_n12"] for item in comparisons),
        maxima_by_channel={
            channel: dict(
                nodes_12_to_16=max(abs(item[f"{channel}_nodes_12_to_16"])
                                   for item in comparisons),
                block_12_to_18=maximum_available(
                    item[f"{channel}_block_12_to_18"] for item in comparisons),
                completed_block_checks=sum(
                    item[f"{channel}_block_12_to_18"] is not None
                    for item in comparisons))
            for channel in ("glasses", "theta")},
        evaluations=[computed[key] for key in sorted(computed)])
    output.mkdir(parents=True, exist_ok=True)
    with (output / "convergence.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    (output / "convergence.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipts", type=Path,
                        default=ROOT / "data/channel_convergence/receipts")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = recover(args.receipts, args.output)
    print(json.dumps({key: value for key, value in report.items()
                      if key not in {"comparisons", "evaluations"}}, indent=2))


if __name__ == "__main__":
    main()
