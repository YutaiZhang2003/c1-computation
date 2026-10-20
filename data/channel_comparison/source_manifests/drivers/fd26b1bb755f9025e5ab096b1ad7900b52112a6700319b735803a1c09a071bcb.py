"""Recompute theta/glasses Q_L at the five printed manuscript period matrices.

Run through run.py. Each evaluate invocation computes one row in one channel.
Receipts include the inverse-period certificate and both partition functions.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(clean(value), indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def select_chart(omega, topology, tolerance):
    from genus2_plumbing_atlas import shortlist_markings, certify_marking
    from genus2_hybrid_period_map import (
        HybridPeriodMapConfig, refine_multiprecision_holomorphic_inverse)
    failures = []
    for marking in shortlist_markings(omega, topology, search_depth=3, count=8):
        print(f"Trying {topology} marking {marking.word}; "
              f"leading q_max={marking.leading_q_max:.6g}", flush=True)
        chart = certify_marking(
            omega, marking, word_length=6, max_nfev=160,
            q_reference_max=.25, period_tolerance=tolerance,
            stability_tolerance=tolerance, maximum_collocation_basis=160)
        print(f"Chart {chart.status}; residual={chart.period_max_residual:.3g}; "
              f"stability={chart.period_map_stability:.3g}", flush=True)
        if (chart.inverse_success and chart.period_max_residual <= tolerance
                and chart.period_map_stability <= tolerance
                and chart.status in {"reference-q-envelope", "requires-recursion-order-study"}):
            return asdict(chart)
        # The generic atlas may choose unscaled collocation at tiny handle q.
        # Use its existing rescaled backend explicitly if that solve overflows.
        if min(marking.leading_q_abs) < 1e-4:
            print("Trying the current rescaled holomorphic-form backend", flush=True)
            try:
                inverse = refine_multiprecision_holomorphic_inverse(
                    topology, marking.omega, marking.leading_q,
                    initial_log_q=marking.leading_log_q, max_nfev=64,
                    config=HybridPeriodMapConfig(
                        tolerance=tolerance, agreement_tolerance=tolerance,
                        maximum_collocation_basis=160))
                if inverse.success and max(inverse.residual, inverse.stability) <= tolerance:
                    return dict(
                        topology=topology, word=marking.word, matrix=marking.matrix,
                        omega_chart=[[str(x) for x in row] for row in marking.omega],
                        integer_branch=inverse.branch.tolist(),
                        q=[str(x) for x in inverse.q],
                        log_q=[str(x) for x in inverse.log_q],
                        q_abs=[abs(x) for x in inverse.q],
                        q_max=max(abs(x) for x in inverse.q),
                        inverse_success=True, inverse_message=inverse.message,
                        inverse_nfev=inverse.nfev,
                        period_max_residual=inverse.residual,
                        period_map_stability=inverse.stability,
                        period_algorithm="rescaled-holomorphic-forms",
                        low_order=inverse.low_order, high_order=inverse.high_order,
                        seam_residual=inverse.seam_residual,
                        symmetry_error=inverse.symmetry_error,
                        status="certified-rescaled-inverse")
                failures.append(inverse.message)
            except (ValueError, ArithmeticError, RuntimeError) as exc:
                failures.append(f"Rescaled backend: {type(exc).__name__}: {exc}")
        failures.append(chart.inverse_message)
    raise RuntimeError(f"No certified {topology} chart: {failures}")


def evaluate(args):
    from genus2_liouville_evaluation import evaluate_liouville_rescaled
    from genus2_scalar_evaluation import evaluate_noncompact_scalar_detailed
    targets = json.loads(args.targets.read_text())
    selected = next(row for row in targets["rows"] if row["row"] == args.row)
    omega = np.asarray([[complex(x) for x in row] for row in selected["omega"]])
    if not np.allclose(omega, omega.T) or np.min(np.linalg.eigvalsh(omega.imag)) <= 0:
        raise ValueError("Target is not a Riemann period matrix")
    args.output.mkdir(parents=True, exist_ok=True)
    stem = f"row{args.row}_{args.channel}"
    path = args.output/f"{stem}_N{args.block_order}_n{args.nodes}.json"
    if path.exists():
        raise FileExistsError(path)
    geometry_path = args.output/f"{stem}_geometry.json"
    start = time.monotonic()
    if geometry_path.exists():
        geometry = json.loads(geometry_path.read_text())
        if (geometry["target_sha256"] != digest(args.targets)
                or geometry["row"] != args.row or geometry["channel"] != args.channel):
            raise ValueError("Changed cached geometry target")
        chart = geometry["chart"]
        if max(chart["period_max_residual"], chart["period_map_stability"]) > args.period_tolerance:
            raise ValueError("Cached geometry does not meet the requested tolerance")
    else:
        chart = select_chart(omega, args.channel, args.period_tolerance)
        save(geometry_path, dict(row=args.row, channel=args.channel,
             target_sha256=digest(args.targets), target=selected, chart=chart))
    if args.geometry_only:
        print(geometry_path)
        return
    q = tuple(complex(value) for value in chart["q"])
    logs = tuple(complex(value) for value in chart["log_q"])
    omega_chart = np.asarray([[complex(value) for value in row] for row in chart["omega_chart"]])
    print("Evaluating the noncompact scalar", flush=True)
    scalar = evaluate_noncompact_scalar_detailed(
        args.channel, q, omega_chart, log_q_values=logs,
        word_length=12, max_mode=120, tolerance=1e-14,
        word_tail_tolerance=1e-12, minimum_word_length=5,
        powered_relative_tolerance=1e-9)
    print(f"Evaluating Liouville: block N={args.block_order}, nodes={args.nodes}", flush=True)
    liouville = evaluate_liouville_rescaled(
        args.channel, q, log_q_values=logs,
        block_order=args.block_order, quadrature_order=args.nodes,
        quadrature_scheme="primary-gaussian", dps=args.dps,
        vacuum_word_length=12, vacuum_oscillator_level=40,
        vacuum_word_tail_tolerance=1e-12, vacuum_minimum_word_length=5,
        resum_global_block=True, global_block_tolerance=1e-13,
        global_block_max_outer_order=96)
    log_zx = math.log(scalar.partition)
    log_ql = liouville.log_partition - 25*log_zx
    record = dict(
        format="c1-channel-comparison-receipt-v1", completed=True,
        completed_utc=datetime.now(timezone.utc).isoformat(),
        row=args.row, channel=args.channel, target=selected,
        coordinates_are_rounded_manuscript_values=True,
        target_sha256=digest(args.targets),
        source_manifest_sha256=digest(ROOT/"SOURCE_MANIFEST.json"),
        driver_sha256=digest(Path(__file__)),
        settings=dict(block_order=args.block_order, momentum_nodes_per_edge=args.nodes,
                      quadrature_scheme="primary-gaussian", dps=args.dps, b=1, c=25,
                      vacuum_word_length=12, vacuum_oscillator_level=40,
                      scalar_word_length=12, scalar_max_mode=120,
                      period_tolerance=args.period_tolerance),
        chart=chart, scalar=asdict(scalar),
        liouville=dict(scaled_partition=liouville.scaled_partition,
                       log_threshold_factor=liouville.log_threshold_factor,
                       log_partition=liouville.log_partition),
        log_Z_X=log_zx, log_Q_L=log_ql, Q_L=math.exp(log_ql),
        elapsed_seconds=time.monotonic()-start,
        python=sys.version.split()[0])
    save(path, record)
    print(f"Finished {stem}: Q_L={record['Q_L']:.16g}; {record['elapsed_seconds']:.1f}s", flush=True)


def recover_receipts(receipts, output, block_order=12, nodes=12):
    """Recompute Q_L and channel ratios from saved scalar/Liouville integrals."""
    import csv
    rows, records = [], []
    receipts, output = Path(receipts), Path(output)
    for index in range(1, 6):
        values = {}
        for channel in ("glasses", "theta"):
            path = receipts/f"row{index}_{channel}_N{block_order}_n{nodes}.json"
            record = json.loads(path.read_text())
            if not record["completed"] or record["row"] != index or record["channel"] != channel:
                raise ValueError("Unexpected channel receipt")
            settings = record["settings"]
            if settings["block_order"] != block_order or settings["momentum_nodes_per_edge"] != nodes:
                raise ValueError("Unexpected block or momentum order")
            liouville = record["liouville"]
            log_zl = math.log(liouville["scaled_partition"])+liouville["log_threshold_factor"]
            log_zx = math.log(record["scalar"]["partition"])
            log_ql = log_zl-25*log_zx
            np.testing.assert_allclose(log_zl, liouville["log_partition"], rtol=0, atol=1e-12)
            np.testing.assert_allclose(log_ql, record["log_Q_L"], rtol=0, atol=1e-12)
            values[channel] = math.exp(log_ql)
            np.testing.assert_allclose(values[channel], record["Q_L"], rtol=2e-12, atol=0)
            records.append(record)
        ratio = values["theta"]/values["glasses"]
        rows.append(dict(row=index, Q_L_glasses=values["glasses"],
                         Q_L_theta=values["theta"], theta_over_glasses=ratio,
                         relative_difference=ratio-1))
    output.mkdir(parents=True, exist_ok=True)
    with (output/"channel_comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = dict(
        rows=5, block_order=block_order, momentum_nodes_per_edge=nodes,
        coordinates_are_rounded_manuscript_values=True,
        maximum_relative_channel_difference=max(abs(row["relative_difference"]) for row in rows),
        all_period_residuals_below_tolerance=all(
            row["chart"]["period_max_residual"] <= row["settings"]["period_tolerance"]
            and row["chart"]["period_map_stability"] <= row["settings"]["period_tolerance"]
            for row in records),
        same_unrounded_inputs_as_original_table=False,
        scope="Fresh comparison at the rounded matrices printed in the draft.")
    save(output/"channel_comparison_verification.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    worker = sub.add_parser("evaluate")
    worker.add_argument("--targets", type=Path, default=ROOT/"data/channel_comparison/targets.json")
    worker.add_argument("--row", type=int, choices=range(1, 6), required=True)
    worker.add_argument("--channel", choices=("theta", "glasses"), required=True)
    worker.add_argument("--output", type=Path, required=True)
    worker.add_argument("--block-order", type=int, default=12)
    worker.add_argument("--nodes", type=int, default=12)
    worker.add_argument("--dps", type=int, default=50)
    worker.add_argument("--period-tolerance", type=float, default=2e-10)
    worker.add_argument("--geometry-only", action="store_true")
    assemble = sub.add_parser("assemble")
    assemble.add_argument("--receipts", type=Path, required=True)
    assemble.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "evaluate":
        evaluate(args)
    else:
        print(json.dumps(recover_receipts(args.receipts, args.output), indent=2))


if __name__ == "__main__":
    main()
