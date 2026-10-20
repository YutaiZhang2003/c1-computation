#!/usr/bin/env python3
"""Run the frozen ten-point imaginary-energy worldsheet smoke design.

This module is deliberately worldsheet-only.  It imports the established
genus-one two-point integrator, fixes the numerical design used in the local
scan, and serializes one blind artifact per value of ``t`` in ``omega=i*t``.
No comparison curve or target coefficient is defined here.

The comparison/fitting stage lives in ``fit_genus1_two_point_bry_scan.py`` and
can run only after every point below has been serialized with
``blind_freeze=true``.
"""

from __future__ import annotations

from c1_paths import PACKAGE_ROOT, data_root, package_path, module_path

import argparse
import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

try:
    from integrate_genus1_two_point_worldsheet import run as run_one_point
except ImportError:  # pragma: no cover - package-style execution
    from integrate_genus1_two_point_worldsheet import run as run_one_point


DEFAULT_T_VALUES = (0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95)
DEFAULT_OUTPUT_DIR = Path(
    str(package_path('plumbing/results/genus1_two_point_worldsheet/imaginary_native_q_smoke_t_scan10_n256_v1'))
)


@dataclass(frozen=True)
class SmokeDesign:
    """Numerical parameters frozen before the ten-point scan."""

    p_max: float = 6.0
    momentum_order: int = 16
    momentum_power: float = 2.0
    necklace_order_first: int = 6
    necklace_order_second: int = 3
    ope_q_order: int = 3
    ope_z_order: int = 8
    necklace_backend: str = "c-recursion"
    ope_backend: str = "c-recursion"
    h_recursion_regulator: float = 0.04
    h_recursion_weight_regulator: float = 0.001
    h_recursion_audit_tolerance: float = 1.0e-7
    epsilon: float = 0.15
    collision_radius: float = 0.10
    cutoffs: str = "3,4,6,8"
    sobol_power: int = 8
    replicates: int = 4
    tail_slices: str = "8,10,12,16,20"
    tail_sobol_power: int = 7
    seed: int = 170507151
    dps: int = 28


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


def parse_t_values(text: str) -> tuple[float, ...]:
    values = tuple(float(piece) for piece in text.split(",") if piece.strip())
    if not values:
        raise ValueError("at least one value of t is required")
    if any(not 0.0 < value < 1.0 for value in values):
        raise ValueError("the blind real-contour scan requires 0<t<1")
    if any(right <= left for left, right in zip(values, values[1:])):
        raise ValueError("t values must be strictly increasing")
    return values


def point_tag(t: float) -> str:
    """Stable tag for the two-decimal smoke grid, e.g. ``0.05 -> t005``."""
    hundredths = int(round(100.0 * float(t)))
    if not math.isclose(float(t), hundredths / 100.0, abs_tol=1.0e-12):
        raise ValueError("the review driver expects t values on a 0.01 grid")
    return f"t{hundredths:03d}"


def point_namespace(t: float, output: Path, design: SmokeDesign) -> argparse.Namespace:
    """Build exactly the namespace consumed by the established integrator."""
    return argparse.Namespace(x=float(t), output=str(output), **asdict(design))


def validate_frozen_record(
    record: Mapping[str, object],
    *,
    t: float,
    design: SmokeDesign,
) -> None:
    """Fail closed if a stored point does not match the frozen blind design."""
    if record.get("blind_freeze") is not True:
        raise ValueError(f"t={t:g}: record is not marked blind_freeze=true")
    if not math.isclose(float(record["x"]), float(t), abs_tol=1.0e-14):
        raise ValueError(f"t={t:g}: stored x does not match the requested point")
    domain = str(record.get("domain", ""))
    if "0<x<1" not in domain:
        raise ValueError(f"t={t:g}: stored domain is not the direct safe strip")

    momentum = record["momentum_rule"]
    expected_momentum = {
        "p_max": design.p_max,
        "order": design.momentum_order,
        "power": design.momentum_power,
    }
    for key, expected in expected_momentum.items():
        if not math.isclose(float(momentum[key]), float(expected), abs_tol=1.0e-14):
            raise ValueError(f"t={t:g}: momentum-rule field {key!r} changed")

    blocks = record["block_orders"]
    expected_blocks = {
        "necklace_q1": design.necklace_order_first,
        "necklace_q2": design.necklace_order_second,
        "ope_q": design.ope_q_order,
        "ope_z": design.ope_z_order,
    }
    for key, expected in expected_blocks.items():
        if int(blocks[key]) != int(expected):
            raise ValueError(f"t={t:g}: block-order field {key!r} changed")

    backends = record.get("block_backends")
    if backends is not None:
        expected_backends = {
            "necklace": design.necklace_backend,
            "ope": design.ope_backend,
            "h_recursion_c_regulator": design.h_recursion_regulator,
            "h_recursion_weight_regulator": design.h_recursion_weight_regulator,
            "h_recursion_audit_tolerance": design.h_recursion_audit_tolerance,
        }
        for key, expected in expected_backends.items():
            actual = backends.get(key)
            if isinstance(expected, float):
                if not math.isclose(float(actual), expected, abs_tol=1.0e-14):
                    raise ValueError(f"t={t:g}: block-backend field {key!r} changed")
            elif actual != expected:
                raise ValueError(f"t={t:g}: block-backend field {key!r} changed")

    rqmc = record["rqmc"]
    expected_rqmc = {
        "sobol_power": design.sobol_power,
        "points_per_replicate": 2**design.sobol_power,
        "replicates": design.replicates,
        "seed": design.seed,
    }
    for key, expected in expected_rqmc.items():
        if int(rqmc[key]) != int(expected):
            raise ValueError(f"t={t:g}: RQMC field {key!r} changed")

    # These two fields were added after the first local scan.  New artifacts
    # must carry them; legacy points remain readable but are flagged explicitly
    # in the generated inspection manifest below.
    tail_rqmc = record.get("tail_rqmc")
    if tail_rqmc is not None:
        expected_tail_rqmc = {
            "sobol_power": design.tail_sobol_power,
            "points_per_slice_replicate": 2**design.tail_sobol_power,
            "replicates": design.replicates,
        }
        for key, expected in expected_tail_rqmc.items():
            if int(tail_rqmc[key]) != int(expected):
                raise ValueError(f"t={t:g}: tail-RQMC field {key!r} changed")
    if (
        record.get("special_dps") is not None
        and int(record["special_dps"]) != design.dps
    ):
        raise ValueError(f"t={t:g}: special-function precision changed")

    if not math.isclose(
        float(record["patch_epsilon"]), design.epsilon, abs_tol=1.0e-14
    ):
        raise ValueError(f"t={t:g}: channel-switch epsilon changed")
    if not math.isclose(
        float(record["collision_disc"]["radius"]),
        design.collision_radius,
        abs_tol=1.0e-14,
    ):
        raise ValueError(f"t={t:g}: collision radius changed")

    expected_cutoffs = [float(value) for value in design.cutoffs.split(",")]
    if [float(value) for value in record["cutoffs"]] != expected_cutoffs:
        raise ValueError(f"t={t:g}: cutoff list changed")
    cusp = record["cusp_fit"]
    expected_slices = [float(value) for value in design.tail_slices.split(",")]
    if [float(value) for value in cusp["tau2_slices"]] != expected_slices:
        raise ValueError(f"t={t:g}: tail-slice list changed")
    if cusp.get("tau_integrand_ansatz") != "f(t)=a0*t^-2+a1*t^-5/3+a2*t^-3":
        raise ValueError(f"t={t:g}: tail ansatz changed")


def stored_point_path(output_dir: Path, t: float) -> Path:
    return output_dir / point_tag(t) / "worldsheet_blind.json"


def inspect_existing_scan(
    output_dir: Path,
    t_values: Sequence[float],
    design: SmokeDesign,
) -> dict[str, object]:
    points: list[dict[str, object]] = []
    for t in t_values:
        path = stored_point_path(output_dir, t)
        if not path.is_file():
            raise FileNotFoundError(path)
        record = json.loads(path.read_text(encoding="utf-8"))
        validate_frozen_record(record, t=t, design=design)
        points.append(
            {
                "t": float(t),
                "path": str(path),
                "sha256": _sha256(path),
                "I1": float(record["cusp_fit"]["final_I"]["real"]),
                "rqmc_standard_error": abs(
                    float(record["cusp_fit"]["final_rqmc_standard_error"]["real"])
                ),
                "legacy_metadata_gaps": [
                    field
                    for field in ("tail_rqmc", "special_dps")
                    if record.get(field) is None
                ],
            }
        )
    return {
        "calculation": "blind genus-one two-point imaginary-energy scan",
        "comparison_stage_present": False,
        "t_values": [float(value) for value in t_values],
        "design": asdict(design),
        "legacy_metadata_statement": (
            "The first local scan predates serialization of tail_rqmc and "
            "special_dps. Missing fields are listed per point; the wrapper "
            "records the command-level values used for exact reproduction."
        ),
        "points": points,
    }


def run_scan(
    output_dir: Path,
    t_values: Sequence[float],
    design: SmokeDesign,
) -> dict[str, object]:
    for t in t_values:
        output = stored_point_path(output_dir, t)
        if output.exists():
            raise FileExistsError(
                f"refusing to overwrite frozen point {output}; use --check-existing "
                "or choose a new --output-dir"
            )
        result = run_one_point(point_namespace(t, output, design))
        validate_frozen_record(result, t=t, design=design)

    manifest = inspect_existing_scan(output_dir, t_values, design)
    manifest_path = output_dir / "worldsheet_scan_manifest.json"
    _atomic_json(manifest_path, manifest)
    print(f"wrote {manifest_path}")
    return manifest


def parser() -> argparse.ArgumentParser:
    out = argparse.ArgumentParser(
        description="Run or validate the blind ten-point imaginary-energy scan."
    )
    out.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    out.add_argument(
        "--t-values",
        default=",".join(f"{value:.2f}" for value in DEFAULT_T_VALUES),
    )
    out.add_argument(
        "--check-existing",
        action="store_true",
        help="validate stored points and print their manifest without writing",
    )
    out.add_argument(
        "--write-manifest",
        action="store_true",
        help="with --check-existing, refresh worldsheet_scan_manifest.json",
    )
    return out


def main() -> None:
    args = parser().parse_args()
    design = SmokeDesign()
    t_values = parse_t_values(args.t_values)
    if args.check_existing:
        manifest = inspect_existing_scan(args.output_dir, t_values, design)
        if args.write_manifest:
            manifest_path = args.output_dir / "worldsheet_scan_manifest.json"
            _atomic_json(manifest_path, manifest)
            print(f"wrote {manifest_path}")
        print(json.dumps(manifest, indent=2, allow_nan=False))
        return
    run_scan(args.output_dir, t_values, design)


if __name__ == "__main__":
    main()
