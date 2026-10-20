"""Recover observed Liouville convergence changes from fixed-geometry receipts."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
import math
from pathlib import Path

from check_channel_convergence import ROOT, digest, load_baseline


def maximum_available(values):
    available = [abs(value) for value in values if value is not None]
    return max(available) if available else None


def export_current_table(convergence_report, output):
    """Recover the current N12/n16 draft table from verified integral results."""
    from liouville_momentum_quadrature import primary_gaussian_momentum_rule

    output = Path(output)
    current = {(item["row"], item["channel"]): item
               for item in convergence_report["evaluations"]
               if (item["block_order"], item["nodes_per_edge"], item["dps"])
               == (12, 16, 50)}
    rows, momentum_rules = [], []
    for row in range(1, 6):
        glasses, theta = current[(row, "glasses")], current[(row, "theta")]
        log_ratio = theta["log_Q_L"] - glasses["log_Q_L"]
        rows.append(dict(row=row, Q_L_glasses=glasses["Q_L"],
                         Q_L_theta=theta["Q_L"],
                         theta_over_glasses=math.exp(log_ratio),
                         relative_difference=math.expm1(log_ratio)))
        for channel in ("glasses", "theta"):
            _, baseline, _ = load_baseline(row, channel)
            chart = baseline["chart"]
            momentum_rules.append(dict(
                row=row, channel=channel,
                rules=[asdict(primary_gaussian_momentum_rule(
                    complex(q), 16, log_q_abs=complex(log_q).real))
                    for q, log_q in zip(chart["q"], chart["log_q"])]))
    data = ROOT / "data/channel_convergence"
    with (data / "current_channel_comparison.csv").open(newline="") as stream:
        reference = list(csv.DictReader(stream))
    if len(reference) != 5:
        raise ValueError("Expected five saved channel-comparison rows")
    for actual, expected in zip(rows, reference):
        for key, value in actual.items():
            if not math.isclose(float(value), float(expected[key]), rel_tol=2e-12, abs_tol=0):
                raise ValueError(f"Current channel comparison differs at row {actual['row']}: {key}")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "channel_comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    maximum = max(abs(row["relative_difference"]) for row in rows)
    report = dict(
        rows=5, block_order=12, momentum_nodes_per_edge=16, dps=50,
        quadrature_scheme="primary-gaussian",
        maximum_relative_channel_difference=maximum,
        agrees_within_1e_minus_7=maximum < 1e-7,
        output_precision="Full stored binary64 values; no display rounding",
        ratios_use_unrounded_integrals=True,
        coordinates_are_rounded_manuscript_values=True,
        scope="Current draft table: empirical cross-channel agreement, not an absolute error bound")
    (output / "channel_comparison_verification.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n")
    (output / "channel_momentum_quadrature.json").write_text(json.dumps(dict(
        quadrature_scheme="primary-gaussian", laguerre_alpha=-.5,
        node_placement="P_j = sqrt(u_j / (-2 log |q_edge|))",
        rows=momentum_rules), indent=2, allow_nan=False) + "\n")
    return report


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
    parser.add_argument("--table-output", type=Path,
                        help="Also recover the current draft's N12/n16 comparison")
    args = parser.parse_args()
    report = recover(args.receipts, args.output)
    if args.table_output is not None:
        report["current_draft_table"] = export_current_table(report, args.table_output)
    print(json.dumps({key: value for key, value in report.items()
                      if key not in {"comparisons", "evaluations"}}, indent=2))


if __name__ == "__main__":
    main()
