"""Vary Liouville quadrature and block orders at the saved table geometries.

The scalar partition, plumbing coordinates, homology marking, and period
matrix are held fixed. This measures Liouville discretization changes only.
Run through run.py; each invocation evaluates one row and one channel.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import time

ROOT = Path(__file__).resolve().parents[3]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_baseline(row, channel):
    path = ROOT / "data/channel_comparison/receipts" / (
        f"row{row}_{channel}_N12_n12.json")
    baseline = json.loads(path.read_text())
    targets = ROOT / "data/channel_comparison/targets.json"
    target = next(item for item in json.loads(targets.read_text())["rows"]
                  if item["row"] == row)
    if (not baseline["completed"] or baseline["row"] != row
            or baseline["channel"] != channel or baseline["target"] != target
            or baseline["target_sha256"] != digest(targets)):
        raise ValueError("Invalid baseline target or receipt")
    settings = baseline["settings"]
    if (settings["block_order"], settings["momentum_nodes_per_edge"],
            settings["quadrature_scheme"], settings["dps"]) != (
            12, 12, "primary-gaussian", 50):
        raise ValueError("Unexpected baseline integration settings")
    chart = baseline["chart"]
    if (not chart["inverse_success"]
            or max(chart["period_max_residual"], chart["period_map_stability"])
            > settings["period_tolerance"]):
        raise ValueError("Uncertified baseline geometry")
    source_path = ROOT / "data/channel_comparison/source_manifests" / (
        baseline["source_manifest_sha256"] + ".json")
    if digest(source_path) != baseline["source_manifest_sha256"]:
        raise ValueError("Changed baseline source manifest")
    # These directories contain all production numerical dependencies.
    # Recovery tools and table drivers are orchestration, not numerical kernels.
    checked = []
    for item in json.loads(source_path.read_text())["files"]:
        name = item["path"]
        if (name.endswith(".py")
                and name.startswith(("code/general/", "code/two_loop/"))
                and not name.startswith(("code/general/recovery/",
                                         "code/two_loop/other_scripts/"))):
            if digest(ROOT / name) != item["sha256"]:
                raise ValueError(f"Numerical source differs from baseline: {name}")
            checked.append(dict(path=name, sha256=item["sha256"]))
    log_zl = (math.log(baseline["liouville"]["scaled_partition"])
              + baseline["liouville"]["log_threshold_factor"])
    log_ql = log_zl - 25 * math.log(baseline["scalar"]["partition"])
    if not math.isclose(log_ql, baseline["log_Q_L"], rel_tol=0, abs_tol=1e-12):
        raise ValueError("Inconsistent baseline partition")
    return path, baseline, checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--row", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--channel", choices=("theta", "glasses"), required=True)
    parser.add_argument("--block-order", type=int, required=True)
    parser.add_argument("--nodes", type=int, required=True)
    parser.add_argument("--dps", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.block_order < 0 or args.nodes < 1 or args.dps < 30:
        parser.error("Require nonnegative block order, positive nodes, dps >= 30")
    baseline_path, baseline, checked = load_baseline(args.row, args.channel)
    settings = dict(
        block_order=args.block_order, momentum_nodes_per_edge=args.nodes,
        quadrature_scheme="primary-gaussian", dps=args.dps,
        vacuum_word_length=12, vacuum_oscillator_level=40,
        vacuum_word_tail_tolerance=1e-12, vacuum_minimum_word_length=5,
        resum_global_block=True, global_block_tolerance=1e-13,
        global_block_max_outer_order=96, b=1, c=25)
    filename = (f"row{args.row}_{args.channel}_N{args.block_order}"
                f"_n{args.nodes}_dps{args.dps}.json")
    path = args.output / filename
    if path.exists():
        raise FileExistsError(path)
    print(json.dumps(dict(row=args.row, channel=args.channel, settings=settings,
                          baseline_Q_L=baseline["Q_L"],
                          unchanged_numerical_sources=len(checked))), flush=True)
    if args.dry_run:
        return
    from genus2_liouville_evaluation import evaluate_liouville_rescaled

    started = time.monotonic()
    chart = baseline["chart"]
    liouville = evaluate_liouville_rescaled(
        args.channel, tuple(complex(q) for q in chart["q"]),
        log_q_values=tuple(complex(q) for q in chart["log_q"]),
        block_order=args.block_order, quadrature_order=args.nodes,
        quadrature_scheme=settings["quadrature_scheme"], dps=args.dps,
        vacuum_word_length=settings["vacuum_word_length"],
        vacuum_oscillator_level=settings["vacuum_oscillator_level"],
        vacuum_word_tail_tolerance=settings["vacuum_word_tail_tolerance"],
        vacuum_minimum_word_length=settings["vacuum_minimum_word_length"],
        resum_global_block=settings["resum_global_block"],
        global_block_tolerance=settings["global_block_tolerance"],
        global_block_max_outer_order=settings["global_block_max_outer_order"])
    log_ql = liouville.log_partition - 25 * math.log(
        baseline["scalar"]["partition"])
    record = dict(
        format="c1-channel-convergence-v1", completed=True,
        completed_utc=datetime.now(timezone.utc).isoformat(),
        row=args.row, channel=args.channel, target=baseline["target"],
        target_sha256=baseline["target_sha256"],
        baseline_receipt=baseline_path.name,
        baseline_sha256=digest(baseline_path),
        baseline_Q_L=baseline["Q_L"],
        baseline_log_Q_L=baseline["log_Q_L"],
        source_manifest_sha256=digest(ROOT / "SOURCE_MANIFEST.json"),
        driver_sha256=digest(__file__), unchanged_numerical_sources=checked,
        settings=settings, chart=chart,
        scalar_partition=baseline["scalar"]["partition"],
        liouville=dict(
            scaled_partition=liouville.scaled_partition,
            log_threshold_factor=liouville.log_threshold_factor,
            log_partition=liouville.log_partition),
        log_Q_L=log_ql, Q_L=math.exp(log_ql),
        relative_change_from_N12_n12=math.expm1(log_ql - baseline["log_Q_L"]),
        elapsed_seconds=time.monotonic() - started,
        python=platform.python_version())
    args.output.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    print(f"Completed {filename}: Q_L={record['Q_L']:.16g}; "
          f"relative_change={record['relative_change_from_N12_n12']:.9g}; "
          f"seconds={record['elapsed_seconds']:.1f}", flush=True)


if __name__ == "__main__":
    main()
