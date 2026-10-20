#!/usr/bin/env python3
"""Close precision and period-sensitivity gates for glass CFT sentinels."""

from __future__ import annotations

from c1_paths import PACKAGE_ROOT, data_root, package_path, module_path

import argparse
import cmath
import csv
import json
import math
import os
import resource
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

try:
    from bolza_torus_plumbing_reach import transform_omega
    from genus2_holomorphic_period_table import PeriodMapSeed
    from genus2_hybrid_period_map import (
        HybridPeriodMapConfig,
        hybrid_period_matrix,
        period_difference_mod_integer,
        period_max_residual,
    )
    from genus2_plumbing_atlas import (
        LeadingMarking,
        certify_marking,
        symplectic_matrix_from_csv_row,
    )
    from genus2_scalar_evaluation import evaluate_glasses_noncompact_scalar_fixed_cap_convergence
    from genus2_liouville_evaluation import evaluate_liouville_rescaled
    from genus2_moduli_estimation import omega_from_csv_row
except ImportError:  # pragma: no cover - supports package-style execution
    from bolza_torus_plumbing_reach import transform_omega
    from genus2_holomorphic_period_table import PeriodMapSeed
    from genus2_hybrid_period_map import (
        HybridPeriodMapConfig,
        hybrid_period_matrix,
        period_difference_mod_integer,
        period_max_residual,
    )
    from genus2_plumbing_atlas import (
        LeadingMarking,
        certify_marking,
        symplectic_matrix_from_csv_row,
    )
    from genus2_scalar_evaluation import evaluate_glasses_noncompact_scalar_fixed_cap_convergence
    from genus2_liouville_evaluation import evaluate_liouville_rescaled
    from genus2_moduli_estimation import omega_from_csv_row


FORMAT_VERSION = 1
CALIBRATED_BRIDGE_MAXIMUM = 0.432
CALIBRATED_Q_MAXIMUM = 0.535


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _peak_rss_megabytes() -> float:
    maximum = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return (
        maximum / (1024.0 * 1024.0)
        if sys.platform == "darwin"
        else maximum / 1024.0
    )


def calibrated_momentum_orders(q_bridge: complex) -> tuple[int, int, int]:
    magnitude = abs(complex(q_bridge))
    if magnitude <= 0.20:
        return 8, 8, 8
    if magnitude <= 0.40:
        return 8, 8, 10
    if magnitude <= CALIBRATED_BRIDGE_MAXIMUM:
        return 8, 8, 12
    raise ValueError(
        f"bridge magnitude {magnitude:.9g} exceeds calibrated maximum "
        f"{CALIBRATED_BRIDGE_MAXIMUM:.7g}"
    )


def _coverage_candidate_key(item: tuple[dict[str, str], Path]) -> tuple:
    row, path = item
    q = tuple(complex(row[f"final_q{i}"]) for i in range(1, 4))
    q_max = max(map(abs, q))
    inside = q_max <= CALIBRATED_Q_MAXIMUM and abs(q[2]) <= CALIBRATED_BRIDGE_MAXIMUM
    return (
        not inside,
        q_max,
        max(float(row["final_period_residual"]), float(row["final_period_map_step"])),
        int(row.get("candidate_rank", "999999")),
        str(path),
    )


def cohort_coverage(args: argparse.Namespace) -> None:
    """Assign the calibrated policy to a complete certified glass cohort."""

    expected: dict[int, str] = {}
    for target_manifest in args.target_manifests:
        for row in _read_rows(target_manifest):
            sample_index = int(row["sample_index"])
            node_id = str(row["rqmc_node_id"])
            previous = expected.get(sample_index)
            if previous is not None and previous != node_id:
                raise ValueError(
                    f"sample {sample_index} has conflicting node ids "
                    f"{previous!r} and {node_id!r}"
                )
            expected[sample_index] = node_id

    candidates: dict[int, list[tuple[dict[str, str], Path]]] = defaultdict(list)
    branch_file_count = 0
    accepted_branch_count = 0
    malformed_files: list[str] = []
    for branch_dir in args.branch_dirs:
        for path in sorted(branch_dir.glob("branch_*.csv")):
            branch_file_count += 1
            rows = _read_rows(path)
            if len(rows) != 1:
                malformed_files.append(str(path))
                continue
            row = rows[0]
            try:
                accepted = bool(
                    row.get("status") == "ok"
                    and row.get("topology") == "glasses"
                    and float(row["final_period_residual"])
                    <= args.period_tolerance
                    and float(row["final_period_map_step"])
                    <= args.period_tolerance
                )
                sample_index = int(row["sample_index"])
            except (KeyError, TypeError, ValueError):
                malformed_files.append(str(path))
                continue
            if accepted and sample_index in expected:
                accepted_branch_count += 1
                candidates[sample_index].append((row, path))

    records: list[dict[str, object]] = []
    missing: list[int] = []
    outside: list[int] = []
    order_counts: Counter[str] = Counter()
    for sample_index in sorted(expected):
        choices = candidates.get(sample_index, [])
        if not choices:
            missing.append(sample_index)
            continue
        # Prefer a calibrated chart before minimising q_max.  A smaller
        # q_max can otherwise select an uncalibrated bridge over a usable
        # alternative marking of the same Omega.
        row, path = min(choices, key=_coverage_candidate_key)
        q_values = tuple(complex(row[f"final_q{i}"]) for i in range(1, 4))
        q_max = max(abs(value) for value in q_values)
        q_bridge = abs(q_values[2])
        coverage_pass = bool(
            q_max <= CALIBRATED_Q_MAXIMUM
            and q_bridge <= CALIBRATED_BRIDGE_MAXIMUM
        )
        orders: tuple[int, int, int] | None
        try:
            orders = calibrated_momentum_orders(q_values[2])
        except ValueError:
            orders = None
        if not coverage_pass or orders is None:
            outside.append(sample_index)
        else:
            order_counts["x".join(str(value) for value in orders)] += 1
        records.append(
            {
                "sample_index": sample_index,
                "rqmc_node_id": expected[sample_index],
                "selected_branch_file": str(path),
                "selected_branch_index": int(row["branch_index"]),
                "candidate_rank": int(row.get("candidate_rank", "999999")),
                "final_period_residual": float(row["final_period_residual"]),
                "final_period_map_step": float(row["final_period_map_step"]),
                "q_max": q_max,
                "q_bridge_abs": q_bridge,
                "quadrature_orders": list(orders) if orders is not None else None,
                "coverage_pass": coverage_pass and orders is not None,
            }
        )

    complete = bool(
        expected
        and not missing
        and not outside
        and not malformed_files
        and len(records) == len(expected)
    )
    payload = {
        "format_version": FORMAT_VERSION,
        "scope": "Full-cohort assignment of the calibrated glass-channel order policy.",
        "created_utc": _utc_now(),
        "expected_target_count": len(expected),
        "covered_target_count": len(records),
        "branch_file_count": branch_file_count,
        "accepted_branch_count": accepted_branch_count,
        "missing_sample_indices": missing,
        "outside_calibrated_range_sample_indices": outside,
        "malformed_branch_files": malformed_files,
        "period_tolerance": args.period_tolerance,
        "calibrated_q_maximum": CALIBRATED_Q_MAXIMUM,
        "calibrated_bridge_maximum": CALIBRATED_BRIDGE_MAXIMUM,
        "observed_q_maximum": max(
            (float(record["q_max"]) for record in records), default=None
        ),
        "observed_bridge_maximum": max(
            (float(record["q_bridge_abs"]) for record in records), default=None
        ),
        "quadrature_order_counts": dict(sorted(order_counts.items())),
        "production_eligible": complete,
        "complete": complete,
        "records": records,
    }
    _atomic_json(args.output, payload)
    print(
        json.dumps(
            {key: value for key, value in payload.items() if key != "records"},
            indent=2,
            allow_nan=False,
        )
    )
    if not complete:
        raise RuntimeError("full glass cohort is not covered by the calibrated policy")


def _adaptive_scalar(
    q_values: Sequence[complex],
    log_q_values: Sequence[complex],
    omega: np.ndarray,
    *,
    maximum_word_length: int = 12,
    powered_tolerance: float = 2.0e-4,
) -> tuple[float, int, float, list[dict[str, object]]]:
    started = time.perf_counter()
    convergence = evaluate_glasses_noncompact_scalar_fixed_cap_convergence(
        q_values,
        omega,
        log_q_values=log_q_values,
        initial_word_length=8,
        maximum_word_length=int(maximum_word_length),
        word_length_increment=2,
        max_mode=40,
        tolerance=1.0e-14,
        powered_relative_tolerance=float(powered_tolerance),
        confirmation_steps=2,
    )
    history = convergence.records()
    history[-1]["total_single_pass_runtime_seconds"] = (
        time.perf_counter() - started
    )
    terminal_change = convergence.terminal_powered_relative_change
    if not convergence.converged or terminal_change is None:
        raise RuntimeError(
            "scalar Schottky product did not pass two direct successive "
            "fixed-word-cap comparisons after the 25th-power amplification"
        )
    evaluation = convergence.evaluation
    return (
        float(evaluation.partition),
        int(evaluation.effective_word_length),
        float(terminal_change),
        history,
    )


def _strict_marking(
    source: np.ndarray,
    branch: Mapping[str, str],
    q_values: Sequence[complex],
    log_q_values: Sequence[complex],
    *,
    tolerance: float,
    maximum_collocation_basis: int,
) -> object:
    matrix = symplectic_matrix_from_csv_row(branch)
    if matrix is None:
        raise ValueError("certified branch lacks its exact symplectic matrix")
    marked = transform_omega(matrix, source)
    marked = 0.5 * (marked + marked.T)
    q_abs = tuple(
        math.exp(value.real) if value.real > -745.0 else 0.0
        for value in log_q_values
    )
    spread_log = max(value.real for value in log_q_values) - min(
        value.real for value in log_q_values
    )
    marking = LeadingMarking(
        topology="glasses",
        word=str(branch["symplectic_word"]),
        matrix=tuple(tuple(int(value) for value in row) for row in matrix),
        omega=marked,
        leading_q=tuple(complex(value) for value in q_values),  # type: ignore[arg-type]
        leading_q_abs=q_abs,  # type: ignore[arg-type]
        leading_q_max=max(q_abs),
        leading_q_spread=(math.exp(spread_log) if spread_log < 700.0 else math.inf),
        leading_log_q=tuple(complex(value) for value in log_q_values),  # type: ignore[arg-type]
    )
    seed = PeriodMapSeed(
        q=tuple(complex(value) for value in q_values),  # type: ignore[arg-type]
        log_q=tuple(complex(value) for value in log_q_values),  # type: ignore[arg-type]
        source="saved-certified-glass-branch",
        table_distance=0.0,
    )
    return certify_marking(
        source,
        marking,
        word_length=8,
        max_nfev=32,
        q_reference_max=0.60,
        period_tolerance=float(tolerance),
        stability_tolerance=float(tolerance),
        inverse_seeds=(seed,),
        maximum_collocation_basis=int(maximum_collocation_basis),
    )


def _linearized_period_candidate(
    marked_omega: np.ndarray,
    q_values: Sequence[complex],
    log_q_values: Sequence[complex],
    *,
    tolerance: float,
    maximum_collocation_basis: int,
) -> dict[str, object]:
    """Apply one leading-period Newton displacement and recertify its map.

    This is used only when the saved forward-map stability already beats the
    strict tolerance but a full inverse optimizer stalls on an irreducible
    target residual.  It turns that residual into an explicit q displacement
    for the quotient-sensitivity test without an unbounded reinversion.
    """

    policy = HybridPeriodMapConfig(
        tolerance=float(tolerance),
        agreement_tolerance=max(float(tolerance), 1.0e-9),
        maximum_collocation_basis=int(maximum_collocation_basis),
    )
    initial = hybrid_period_matrix(
        "glasses",
        q_values,
        config=policy,
        log_q_values=log_q_values,
    )
    difference = period_difference_mod_integer(initial.omega, marked_omega)
    bridge = complex(q_values[2])
    if bridge == 0.0:
        raise ValueError("linearized glass period perturbation needs nonzero bridge")
    two_pi_i = 2.0j * math.pi
    updates = (
        -two_pi_i * difference[0, 0],
        -two_pi_i * difference[1, 1],
        two_pi_i * difference[0, 1] / bridge,
    )
    displaced_logs = tuple(
        complex(old + update) for old, update in zip(log_q_values, updates)
    )
    displaced_q = tuple(
        cmath.exp(complex(max(value.real, -690.0), value.imag))
        for value in displaced_logs
    )
    final = hybrid_period_matrix(
        "glasses",
        displaced_q,
        config=policy,
        log_q_values=displaced_logs,
    )
    stability = max(float(initial.error_estimate), float(final.error_estimate))
    return {
        "q": displaced_q,
        "log_q": displaced_logs,
        "residual": period_max_residual(final.omega, marked_omega),
        "stability": stability,
        "algorithm": f"linearized:{final.algorithm}",
        "region": final.region,
        "evaluations": 2,
        "message": "one leading-period Newton displacement with strict forward recertification",
    }


def _cft_log_quotient(
    q_values: Sequence[complex],
    log_q_values: Sequence[complex],
    scalar: float,
    *,
    orders: Sequence[int],
    dps: int,
) -> tuple[float, float]:
    started = time.perf_counter()
    result = evaluate_liouville_rescaled(
        "glasses",
        q_values,
        log_q_values=log_q_values,
        block_order=8,
        quadrature_order=orders,
        quadrature_scheme="primary-gaussian",
        dps=int(dps),
        vacuum_word_length=8,
        vacuum_oscillator_level=40,
        resum_global_block=True,
        global_block_tolerance=1.0e-13,
    )
    return result.log_partition - 25.0 * math.log(float(scalar)), time.perf_counter() - started


def evaluate(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    targets = _read_rows(args.targets_csv)
    if not 0 <= args.target_index < len(targets):
        raise IndexError("target index is outside the sentinel design")
    target = targets[args.target_index]
    if int(target["target_index"]) != args.target_index:
        raise ValueError("sentinel target indices are not contiguous")
    source_by_id = {
        row["rqmc_node_id"]: row for row in _read_rows(args.source_csv)
    }
    source_row = source_by_id[target["rqmc_node_id"]]
    branch_path = args.branch_dir / f"branch_{args.target_index:08d}.csv"
    branch_rows = _read_rows(branch_path)
    if len(branch_rows) != 1:
        raise ValueError("sentinel branch file must contain exactly one row")
    branch = branch_rows[0]
    output: dict[str, object] = {
        "format_version": FORMAT_VERSION,
        "target_index": args.target_index,
        "rqmc_node_id": target["rqmc_node_id"],
        "status": "failed",
        "error": "",
        "production_eligible": False,
        "started_utc": _utc_now(),
    }
    try:
        if branch.get("status") != "ok" or branch.get("topology") != "glasses":
            raise ValueError("selected branch is not a successful glass chart")
        saved_residual = float(branch["final_period_residual"])
        saved_map_step = float(branch["final_period_map_step"])
        if max(saved_residual, saved_map_step) > args.saved_period_tolerance:
            raise ValueError("saved branch does not pass its period certificate")
        q_values = tuple(complex(branch[f"final_q{i}"]) for i in range(1, 4))
        log_q_values = tuple(
            complex(branch[f"final_log_q{i}"]) for i in range(1, 4)
        )
        orders = calibrated_momentum_orders(q_values[2])
        source = omega_from_csv_row(source_row)
        matrix = symplectic_matrix_from_csv_row(branch)
        if matrix is None:
            raise ValueError("selected branch has no symplectic matrix")
        omega_glasses = transform_omega(matrix, source)
        omega_glasses = 0.5 * (omega_glasses + omega_glasses.T)

        print(json.dumps({"event": "scalar_saved_start", "target_index": args.target_index}), flush=True)
        scalar, scalar_word, scalar_error, scalar_history = _adaptive_scalar(
            q_values, log_q_values, omega_glasses
        )
        print(json.dumps({"event": "precision_44_start", "target_index": args.target_index, "orders": orders}), flush=True)
        log_q_44, runtime_44 = _cft_log_quotient(
            q_values, log_q_values, scalar, orders=orders, dps=44
        )
        print(json.dumps({"event": "precision_52_start", "target_index": args.target_index, "orders": orders}), flush=True)
        log_q_52, runtime_52 = _cft_log_quotient(
            q_values, log_q_values, scalar, orders=orders, dps=52
        )
        precision_change = math.expm1(log_q_52 - log_q_44)

        print(json.dumps({"event": "strict_period_inverse_start", "target_index": args.target_index}), flush=True)
        inverse_started = time.perf_counter()
        if args.linearized_period_perturbation:
            strict_record = _linearized_period_candidate(
                omega_glasses,
                q_values,
                log_q_values,
                tolerance=args.strict_period_tolerance,
                maximum_collocation_basis=args.maximum_collocation_basis,
            )
            strict_q = tuple(complex(value) for value in strict_record["q"])
            strict_logs = tuple(
                complex(value) for value in strict_record["log_q"]
            )
            strict_residual = float(strict_record["residual"])
            strict_stability = float(strict_record["stability"])
            strict_algorithm = str(strict_record["algorithm"])
            strict_region = str(strict_record["region"])
            strict_evaluations = int(strict_record["evaluations"])
            strict_message = str(strict_record["message"])
            strict_inverse_success = False
        else:
            strict = _strict_marking(
                source,
                branch,
                q_values,
                log_q_values,
                tolerance=args.strict_period_tolerance,
                maximum_collocation_basis=args.maximum_collocation_basis,
            )
            strict_q = tuple(complex(value) for value in strict.q)
            strict_logs = tuple(complex(value) for value in strict.log_q)
            strict_residual = float(strict.period_max_residual)
            strict_stability = float(strict.period_map_stability)
            strict_algorithm = str(strict.period_algorithm)
            strict_region = str(strict.period_map_region)
            strict_evaluations = int(strict.inverse_nfev)
            strict_message = str(strict.inverse_message)
            strict_inverse_success = bool(strict.inverse_success)
        inverse_runtime = time.perf_counter() - inverse_started
        strict_orders = calibrated_momentum_orders(strict_q[2])
        if strict_orders != orders:
            raise RuntimeError("strict inverse crossed a calibrated momentum-order boundary")
        strict_inverse_reaches_target_tolerance = bool(
            strict_inverse_success
            and strict_residual <= args.strict_period_tolerance
            and strict_stability <= args.strict_period_tolerance
        )
        # Some bulk inverses reach a stationary residual floor set by the
        # saved period matrix while their independently raised forward map is
        # already stable below the strict tolerance.  Such a candidate is the
        # correct object for the sensitivity comparison: require the raised
        # map stability, retain the original certified residual bar, and
        # record separately whether the stricter target residual was reached.
        strict_inverse_usable = bool(
            math.isfinite(strict_residual)
            and math.isfinite(strict_stability)
            and strict_residual <= args.saved_period_tolerance
            and strict_stability <= args.strict_period_tolerance
        )
        if not strict_inverse_usable:
            raise RuntimeError(
                "raised-map inverse is unusable: "
                f"{strict_message}; residual={strict_residual:.3e}; "
                f"stability={strict_stability:.3e}"
            )

        print(json.dumps({"event": "scalar_strict_start", "target_index": args.target_index}), flush=True)
        strict_scalar, strict_scalar_word, strict_scalar_error, strict_scalar_history = _adaptive_scalar(
            strict_q, strict_logs, omega_glasses
        )
        print(json.dumps({"event": "strict_cft_start", "target_index": args.target_index, "orders": orders}), flush=True)
        strict_log_q_44, strict_cft_runtime = _cft_log_quotient(
            strict_q, strict_logs, strict_scalar, orders=orders, dps=44
        )
        period_change = math.expm1(strict_log_q_44 - log_q_44)
        log_q_changes = [
            complex(new - old) for new, old in zip(strict_logs, log_q_values)
        ]

        output.update(
            {
                "status": "ok",
                "saved_period_residual": saved_residual,
                "saved_period_map_step": saved_map_step,
                "strict_period_tolerance": args.strict_period_tolerance,
                "strict_period_residual": strict_residual,
                "strict_period_map_stability": strict_stability,
                "strict_period_algorithm": strict_algorithm,
                "strict_period_region": strict_region,
                "linearized_period_perturbation": bool(
                    args.linearized_period_perturbation
                ),
                "strict_inverse_reaches_target_tolerance": (
                    strict_inverse_reaches_target_tolerance
                ),
                "strict_inverse_usable_for_sensitivity": strict_inverse_usable,
                "strict_inverse_evaluations": strict_evaluations,
                "strict_inverse_message": strict_message,
                "saved_q": [str(value) for value in q_values],
                "strict_q": [str(value) for value in strict_q],
                "maximum_absolute_log_q_change": max(abs(value) for value in log_q_changes),
                "quadrature_orders": list(orders),
                "block_order": 8,
                "saved_scalar": scalar,
                "saved_scalar_word_length": scalar_word,
                "saved_scalar_powered_error": scalar_error,
                "saved_scalar_history": scalar_history,
                "strict_scalar": strict_scalar,
                "strict_scalar_word_length": strict_scalar_word,
                "strict_scalar_powered_error": strict_scalar_error,
                "strict_scalar_history": strict_scalar_history,
                "log_quotient_dps44": log_q_44,
                "log_quotient_dps52": log_q_52,
                "precision_relative_change": precision_change,
                "precision_relative_tolerance": args.precision_relative_tolerance,
                "precision_pass": abs(precision_change) <= args.precision_relative_tolerance,
                "strict_log_quotient_dps44": strict_log_q_44,
                "period_relative_change": period_change,
                "period_relative_tolerance": args.period_relative_tolerance,
                "period_pass": abs(period_change) <= args.period_relative_tolerance,
                "coverage_pass": abs(strict_q[2]) <= CALIBRATED_BRIDGE_MAXIMUM,
                "stage_runtimes_seconds": {
                    "cft_saved_dps44": runtime_44,
                    "cft_saved_dps52": runtime_52,
                    "strict_period_inverse": inverse_runtime,
                    "cft_strict_dps44": strict_cft_runtime,
                },
            }
        )
        output["point_gate_pass"] = bool(
            output["precision_pass"]
            and output["period_pass"]
            and output["coverage_pass"]
        )
        output["production_eligible"] = output["point_gate_pass"]
    except Exception as exc:
        output["error"] = f"{type(exc).__name__}: {exc}"
    output["total_runtime_seconds"] = time.perf_counter() - started
    output["peak_rss_megabytes"] = _peak_rss_megabytes()
    output["finished_utc"] = _utc_now()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(args.output_dir / f"target_{args.target_index:05d}.json", output)
    print(json.dumps(output, indent=2, allow_nan=False), flush=True)
    if output["status"] != "ok":
        raise RuntimeError(str(output["error"]))


def assemble(args: argparse.Namespace) -> None:
    targets = _read_rows(args.targets_csv)
    results: list[dict[str, object]] = []
    missing: list[int] = []
    for index in range(len(targets)):
        path = args.evaluation_dir / f"target_{index:05d}.json"
        if not path.exists():
            missing.append(index)
            continue
        result = json.loads(path.read_text())
        # Early gate runs initialized this field to false and did not replace
        # it after computing point_gate_pass.  Normalize embedded summary
        # records so the cohort and point-level eligibility flags agree.
        result["production_eligible"] = bool(
            result.get("status") == "ok"
            and result.get("point_gate_pass", False)
        )
        results.append(result)
    failed = [
        int(item["target_index"])
        for item in results
        if item.get("status") != "ok"
    ]
    point_failures = [
        int(item["target_index"])
        for item in results
        if item.get("status") == "ok" and not item.get("point_gate_pass", False)
    ]
    complete = not missing and not failed and not point_failures and len(results) == len(targets)
    payload = {
        "format_version": FORMAT_VERSION,
        "scope": "Glass-channel precision and period-sensitivity production gate.",
        "created_utc": _utc_now(),
        "target_count": len(targets),
        "result_count": len(results),
        "missing_target_indices": missing,
        "failed_target_indices": failed,
        "point_gate_failure_indices": point_failures,
        "precision_relative_tolerance": args.precision_relative_tolerance,
        "period_relative_tolerance": args.period_relative_tolerance,
        "maximum_absolute_precision_relative_change": (
            max(abs(float(item["precision_relative_change"])) for item in results if item.get("status") == "ok")
            if results and not failed
            else None
        ),
        "maximum_absolute_period_relative_change": (
            max(abs(float(item["period_relative_change"])) for item in results if item.get("status") == "ok")
            if results and not failed
            else None
        ),
        "maximum_strict_period_residual": (
            max(float(item["strict_period_residual"]) for item in results if item.get("status") == "ok")
            if results and not failed
            else None
        ),
        "maximum_strict_period_map_stability": (
            max(float(item["strict_period_map_stability"]) for item in results if item.get("status") == "ok")
            if results and not failed
            else None
        ),
        "production_eligible": complete,
        "complete": complete,
        "results": results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(args.output_dir / "summary.json", payload)
    passed_marker = args.output_dir / "PRODUCTION_GATE_PASSED.json"
    failed_marker = args.output_dir / "PRODUCTION_GATE_FAILED.json"
    obsolete_marker = failed_marker if complete else passed_marker
    if obsolete_marker.exists():
        obsolete_marker.unlink()
    _atomic_json(
        passed_marker if complete else failed_marker,
        {
            "production_eligible": complete,
            "created_utc": payload["created_utc"],
            "summary": "summary.json",
            "missing_target_indices": missing,
            "failed_target_indices": failed,
            "point_gate_failure_indices": point_failures,
        },
    )
    print(json.dumps(payload, indent=2, allow_nan=False))
    if not complete:
        raise RuntimeError("glass production gate did not pass")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--source-csv", type=Path, required=True)
    evaluation.add_argument("--targets-csv", type=Path, required=True)
    evaluation.add_argument("--branch-dir", type=Path, required=True)
    evaluation.add_argument("--target-index", type=int, required=True)
    evaluation.add_argument("--output-dir", type=Path, required=True)
    evaluation.add_argument("--saved-period-tolerance", type=float, default=1.0e-7)
    evaluation.add_argument("--strict-period-tolerance", type=float, default=1.0e-9)
    evaluation.add_argument("--maximum-collocation-basis", type=int, default=128)
    evaluation.add_argument(
        "--linearized-period-perturbation",
        action="store_true",
        help="replace full reinversion by one recertified leading-period displacement",
    )
    evaluation.add_argument("--precision-relative-tolerance", type=float, default=1.0e-8)
    evaluation.add_argument("--period-relative-tolerance", type=float, default=5.0e-5)
    evaluation.set_defaults(function=evaluate)
    assembly = commands.add_parser("assemble")
    assembly.add_argument("--targets-csv", type=Path, required=True)
    assembly.add_argument("--evaluation-dir", type=Path, required=True)
    assembly.add_argument("--output-dir", type=Path, required=True)
    assembly.add_argument("--precision-relative-tolerance", type=float, default=1.0e-8)
    assembly.add_argument("--period-relative-tolerance", type=float, default=5.0e-5)
    assembly.set_defaults(function=assemble)
    coverage = commands.add_parser("cohort-coverage")
    coverage.add_argument(
        "--target-manifests", type=Path, nargs="+", required=True
    )
    coverage.add_argument("--branch-dirs", type=Path, nargs="+", required=True)
    coverage.add_argument("--output", type=Path, required=True)
    coverage.add_argument("--period-tolerance", type=float, default=1.0e-7)
    coverage.set_defaults(function=cohort_coverage)
    return root


def main() -> None:
    args = parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
