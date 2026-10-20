#!/usr/bin/env python3
"""Post-freeze BRY fit of the genus-one two-point imaginary-energy scan.

This module is intentionally separate from the worldsheet runner.  It refuses
inputs that are not marked ``blind_freeze=true``, converts the already saved
native integral to BRY amplitude units, and only then introduces the analytic
three-term basis used in Balthazar--Rodriguez--Yin.
"""

from __future__ import annotations

from c1_paths import PACKAGE_ROOT, data_root, package_path, module_path

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


DEFAULT_SCAN_DIR = Path(
    str(package_path('plumbing/results/genus1_two_point_worldsheet/imaginary_hybrid_hc_t_scan10_n256_v2'))
)
BRY_REPORTED_COEFFICIENTS = (1.018, 1.028, 1.0344)


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bry_design_matrix(t_values: np.ndarray) -> np.ndarray:
    r"""Basis for ``-i*A(it)=(-a*t^2+2*b*t^4-c*t^5)/24``."""
    t_values = np.asarray(t_values, dtype=float)
    return np.column_stack(
        [-t_values**2, 2.0 * t_values**4, -t_values**5]
    ) / 24.0


def analytic_reduced_amplitude(t_values: np.ndarray) -> np.ndarray:
    """Matrix-model prediction in the reduced BRY amplitude convention."""
    return bry_design_matrix(np.asarray(t_values, dtype=float)) @ np.ones(3)


def _correlation(covariance: np.ndarray) -> np.ndarray:
    scales = np.sqrt(np.diag(covariance))
    return covariance / np.outer(scales, scales)


def load_frozen_records(scan_dir: Path) -> list[tuple[Path, dict[str, object]]]:
    paths = sorted(scan_dir.glob("t*/worldsheet_blind.json"))
    if len(paths) < 3:
        raise ValueError(f"need at least three frozen points below {scan_dir}")

    loaded: list[tuple[Path, dict[str, object]]] = []
    for path in paths:
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("blind_freeze") is not True:
            raise ValueError(f"refusing non-frozen input {path}")
        if record.get("native_convention") is not True:
            raise ValueError(f"refusing non-native input {path}")
        # Both generations store the SAME reduced I; do not divide I by four.
        try:
            from c1_string_conventions import genus1_two_point_source_convention
        except ImportError:  # pragma: no cover
            from c1_string_conventions import genus1_two_point_source_convention
        genus1_two_point_source_convention(record)
        if "0<x<1" not in str(record.get("domain", "")):
            raise ValueError(f"input is outside the direct safe strip: {path}")
        loaded.append((path, record))

    loaded.sort(key=lambda item: float(item[1]["x"]))
    t_values = [float(record["x"]) for _, record in loaded]
    if any(right <= left for left, right in zip(t_values, t_values[1:])):
        raise ValueError("stored t values are not strictly increasing")
    return loaded


def fit_frozen_records(
    loaded: Sequence[tuple[Path, Mapping[str, object]]],
) -> dict[str, object]:
    """Fit central values and shared-scramble replicates in BRY units."""
    t_values = np.asarray([float(record["x"]) for _, record in loaded])
    native_i1 = np.asarray(
        [float(record["cusp_fit"]["final_I"]["real"]) for _, record in loaded]
    )
    native_imag = np.asarray(
        [float(record["cusp_fit"]["final_I"]["imag"]) for _, record in loaded]
    )
    native_se = np.asarray(
        [
            abs(float(record["cusp_fit"]["final_rqmc_standard_error"]["real"]))
            for _, record in loaded
        ]
    )
    if np.any(native_se <= 0.0) or not np.all(np.isfinite(native_se)):
        raise ValueError("every point must have a finite positive RQMC error")
    if np.max(np.abs(native_imag)) > 1.0e-12:
        raise ValueError("the imaginary-ray native integrals are not real")

    # This convention change is performed only after all native inputs above
    # have passed their blind-freeze checks.
    reduced_amplitude = 0.5 * native_i1
    reduced_se = 0.5 * native_se
    design = bry_design_matrix(t_values)
    weights = 1.0 / reduced_se**2
    normal = design.T @ (weights[:, None] * design)
    rhs = design.T @ (weights * reduced_amplitude)
    coefficients = np.linalg.solve(normal, rhs)
    diagonal_covariance = np.linalg.inv(normal)
    fitted = design @ coefficients
    residuals = reduced_amplitude - fitted
    diagonal_chi_squared = float(np.sum((residuals / reduced_se) ** 2))

    unweighted_coefficients, *_ = np.linalg.lstsq(
        design,
        reduced_amplitude,
        rcond=None,
    )

    replicate_lists = [record["cusp_fit"]["replicate_finals"] for _, record in loaded]
    replicate_count = len(replicate_lists[0])
    if replicate_count < 2 or any(
        len(values) != replicate_count for values in replicate_lists
    ):
        raise ValueError("all points must contain the same number of replicates")
    replicate_amplitudes = 0.5 * np.asarray(
        [
            [float(replicate_lists[point][replicate]["real"]) for point in range(len(loaded))]
            for replicate in range(replicate_count)
        ]
    )
    replicate_coefficients = np.asarray(
        [
            np.linalg.solve(normal, design.T @ (weights * values))
            for values in replicate_amplitudes
        ]
    )
    shared_scramble_covariance = (
        np.cov(replicate_coefficients, rowvar=False, ddof=1) / replicate_count
    )
    shared_scramble_standard_errors = np.sqrt(
        np.diag(shared_scramble_covariance)
    )

    analytic = analytic_reduced_amplitude(t_values)
    common_denominator = float(np.sum(weights * analytic * analytic))
    common_coefficient = float(
        np.sum(weights * analytic * reduced_amplitude) / common_denominator
    )
    replicate_common_coefficients = np.asarray(
        [
            np.sum(weights * analytic * values) / common_denominator
            for values in replicate_amplitudes
        ]
    )
    shared_common_error = float(
        np.std(replicate_common_coefficients, ddof=1) / math.sqrt(replicate_count)
    )
    common_residuals = reduced_amplitude - common_coefficient * analytic
    common_chi_squared = float(np.sum((common_residuals / reduced_se) ** 2))

    rows: list[dict[str, object]] = []
    for index, (path, _) in enumerate(loaded):
        rows.append(
            {
                "t": float(t_values[index]),
                "input": str(path),
                "input_sha256": _sha256(path),
                "native_I1": float(native_i1[index]),
                "native_rqmc_standard_error": float(native_se[index]),
                "bry_minus_i_amplitude": float(reduced_amplitude[index]),
                "bry_rqmc_standard_error": float(reduced_se[index]),
                "analytic_bry_minus_i_amplitude": float(analytic[index]),
                "analytic_native_I1": float(2.0 * analytic[index]),
                "worldsheet_over_analytic": float(
                    reduced_amplitude[index] / analytic[index]
                ),
            }
        )

    return {
        "calculation": "post-freeze BRY fit of genus-one two-point worldsheet data",
        "ordering_statement": (
            "Every input was serialized in native convention with blind_freeze=true "
            "before this module introduced the BRY basis or analytic comparison."
        ),
        "input_count": len(loaded),
        "notation": {
            "native": "A_1^ws=2*pi^2*i*g_BRY^2*I_1; frozen Xi inputs have identical I_1",
            "post_freeze_transform": "-i*A_BRY^(1)=I_1/2",
            "fit_form": "-i*A_BRY^(1)(i*t)=(-a*t^2+2*b*t^4-c*t^5)/24",
        },
        "weighted_fit": {
            "coefficients": {
                "a": float(coefficients[0]),
                "b": float(coefficients[1]),
                "c": float(coefficients[2]),
            },
            "shared_scramble_rqmc_standard_errors": {
                "a": float(shared_scramble_standard_errors[0]),
                "b": float(shared_scramble_standard_errors[1]),
                "c": float(shared_scramble_standard_errors[2]),
            },
            "shared_scramble_covariance": shared_scramble_covariance.tolist(),
            "replicate_coefficients": replicate_coefficients.tolist(),
            "independent_error_approximation": {
                "standard_errors": np.sqrt(np.diag(diagonal_covariance)).tolist(),
                "correlation": _correlation(diagonal_covariance).tolist(),
                "chi_squared": diagonal_chi_squared,
                "degrees_of_freedom": len(loaded) - 3,
            },
            "weighted_design_condition_number": float(
                np.linalg.cond(design / reduced_se[:, None])
            ),
        },
        "unweighted_fit": {
            "a": float(unweighted_coefficients[0]),
            "b": float(unweighted_coefficients[1]),
            "c": float(unweighted_coefficients[2]),
        },
        "common_shape_fit": {
            "definition": "a=b=c=kappa",
            "kappa": common_coefficient,
            "shared_scramble_rqmc_standard_error": shared_common_error,
            "replicate_kappa": replicate_common_coefficients.tolist(),
            "independent_error_chi_squared": common_chi_squared,
            "degrees_of_freedom": len(loaded) - 1,
        },
        "analytic_comparison": {
            "coefficients": {"a": 1.0, "b": 1.0, "c": 1.0},
            "common_kappa": 1.0,
            "native_I1": "(-t^2+2*t^4-t^5)/12",
            "bry_minus_i_amplitude": "(-t^2+2*t^4-t^5)/24",
            "resonance_t_2": {
                "native_I1": -1.0 / 3.0,
                "bry_minus_i_amplitude": -1.0 / 6.0,
            },
            "bry_reported_fit": {
                "a": BRY_REPORTED_COEFFICIENTS[0],
                "b": BRY_REPORTED_COEFFICIENTS[1],
                "c": BRY_REPORTED_COEFFICIENTS[2],
                "source": "Balthazar--Rodriguez--Yin, arXiv:1705.07151",
            },
        },
        "rows": rows,
        "limitations": [
            "RQMC errors are based on only four shared scrambles.",
            "The shared-scramble errors exclude block, momentum, and tail-model systematics.",
            "The t^4 and t^5 basis columns are strongly correlated on 0<t<1.",
        ],
    }


def parser() -> argparse.ArgumentParser:
    out = argparse.ArgumentParser(
        description="Fit frozen imaginary-energy worldsheet data in BRY notation."
    )
    out.add_argument("--scan-dir", type=Path, default=DEFAULT_SCAN_DIR)
    out.add_argument(
        "--output",
        type=Path,
        help="default: <scan-dir>/bry_postfreeze_fit.json",
    )
    out.add_argument(
        "--stdout-only",
        action="store_true",
        help="print the result without writing an artifact",
    )
    return out


def main() -> None:
    args = parser().parse_args()
    loaded = load_frozen_records(args.scan_dir)
    result = fit_frozen_records(loaded)
    if not args.stdout_only:
        output = args.output or args.scan_dir / "bry_postfreeze_fit.json"
        _atomic_json(output, result)
        print(f"wrote {output}")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
