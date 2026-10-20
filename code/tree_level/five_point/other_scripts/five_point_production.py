"""Sphere 1->4 scan and the frozen twelve-point extension."""
from __future__ import annotations
from c1_paths import data_root
import argparse
import json
import math
from pathlib import Path
from c1_observables import amplitude_from_reduced
from five_point_integrand_and_quadrature import EqualEnergyFivePointKernel
from five_point_moduli_integration import integrate_convergent_equal_energy_atlas_qmc
import hashlib
from datetime import date
from typing import Any

def scan_point(
    t_value: float,
    *,
    block_order: int,
    momentum_order: int,
    momentum_maximum: float,
    momentum_panels: int,
    momentum_power: float,
    block_scheme: str,
    liouville_contour: str,
    sobol_power: int,
    replicates: int,
    radial_power: float,
    seed: int,
) -> dict[str, object]:
    """Evaluate one convergent point and return a JSON-ready record."""

    t_value = float(t_value)
    if not math.isfinite(t_value) or t_value <= 0.0:
        raise ValueError("each t value must be positive and finite")
    omega = 1.0j * t_value
    kernel = EqualEnergyFivePointKernel(
        omega,
        block_order=block_order,
        momentum_order=momentum_order,
        momentum_maximum=momentum_maximum,
        momentum_panels=momentum_panels,
        momentum_power=momentum_power,
        block_scheme=block_scheme,
        liouville_contour=liouville_contour,
    )
    result = integrate_convergent_equal_energy_atlas_qmc(
        kernel,
        sobol_power=sobol_power,
        replicates=replicates,
        radial_power=radial_power,
        seed=seed,
    )
    denominator = 16.0 * math.pi**2 * omega**5
    reduced = result.mean / denominator
    denominator_magnitude = abs(denominator)
    # Division by i*positive swaps the raw real/imaginary uncertainties.
    reduced_standard_error_real = (
        result.standard_error_imag / denominator_magnitude
    )
    reduced_standard_error_imag = (
        result.standard_error_real / denominator_magnitude
    )
    return {
        "t": t_value,
        "omega": {"real": omega.real, "imag": omega.imag},
        "raw_integral": {"real": result.mean.real, "imag": result.mean.imag},
        "raw_standard_error": {
            "real": result.standard_error_real,
            "imag": result.standard_error_imag,
        },
        "Q": {"real": reduced.real, "imag": reduced.imag},
        "Q_standard_error": {
            "real": reduced_standard_error_real,
            "imag": reduced_standard_error_imag,
        },
        "replicate_estimates": [
            {"real": value.real, "imag": value.imag}
            for value in result.estimates
        ],
        "amplitude_gs": amplitude_from_reduced(
            result.mean, complex(result.standard_error_real, result.standard_error_imag),
            factor=2j * math.pi, genus=0, punctures=5, reduced_quantity="I5",
            replicates=result.estimates,
        ),
    }


def scan_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--t",
        type=float,
        nargs="+",
        default=(0.38, 0.42, 0.46, 0.48),
        help="positive imaginary-energy coordinates, omega=i*t",
    )
    parser.add_argument("--block-order", type=int, default=6)
    parser.add_argument("--momentum-order", type=int, default=16)
    parser.add_argument("--momentum-maximum", type=float, default=6.0)
    parser.add_argument("--momentum-panels", type=int, default=1)
    parser.add_argument(
        "--momentum-power",
        type=float,
        default=1.25,
        help=(
            "power in P=Pmax*u^power; 5/4 improves threshold resolution "
            "without placing c-recursion nodes too close to P=0"
        ),
    )
    parser.add_argument("--block-scheme", choices=("h", "c"), default="c")
    parser.add_argument(
        "--liouville-contour",
        choices=("real", "continued"),
        default="continued",
        help=(
            "real uses P>=0; continued adds the crossed-pole residue needed "
            "across the t=2/5 DOZZ endpoint pinch"
        ),
    )
    parser.add_argument("--sobol-power", type=int, default=10)
    parser.add_argument("--replicates", type=int, default=4)
    parser.add_argument("--radial-power", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=3201)
    parser.add_argument(
        "--output",
        type=Path,
        default=data_root("tree_level")
        / "sphere_five_point_1to4"
        / "worldsheet_extended_pilot.json",
    )
    arguments = parser.parse_args()

    records: list[dict[str, object]] = []
    for t_value in arguments.t:
        print(f"evaluating omega=i*{t_value:.6g}", flush=True)
        try:
            record = scan_point(
                t_value,
                block_order=arguments.block_order,
                momentum_order=arguments.momentum_order,
                momentum_maximum=arguments.momentum_maximum,
                momentum_panels=arguments.momentum_panels,
                momentum_power=arguments.momentum_power,
                block_scheme=arguments.block_scheme,
                liouville_contour=arguments.liouville_contour,
                sobol_power=arguments.sobol_power,
                replicates=arguments.replicates,
                radial_power=arguments.radial_power,
                seed=arguments.seed,
            )
        except Exception as error:  # preserve successful points in a long scan
            record = {
                "t": float(t_value),
                "error_type": type(error).__name__,
                "error": str(error),
            }
        records.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)

    payload = {
        "description": (
            "Worldsheet-only extension on omega=i*t; no matrix-model values "
            "were imported or evaluated."
        ),
        "settings": {
            "block_order": arguments.block_order,
            "momentum_order": arguments.momentum_order,
            "momentum_maximum": arguments.momentum_maximum,
            "momentum_panels": arguments.momentum_panels,
            "momentum_power": arguments.momentum_power,
            "block_scheme": arguments.block_scheme,
            "liouville_contour": arguments.liouville_contour,
            "continuation": (
                "real P>=0 double integral plus -2i times the residue at "
                "P=(5/2)omega-i on an incoming-outgoing cherry edge"
            ),
            "validity_range": "0<t<1/2; a second DOZZ pole family pinches at t=1/2",
            "sobol_power": arguments.sobol_power,
            "replicates": arguments.replicates,
            "radial_power": arguments.radial_power,
            "seed": arguments.seed,
        },
        "points": records,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {arguments.output}", flush=True)


STATUS = "sphere_1to4_worldsheet_extension_frozen_before_external_comparison"


BASE_SEED = 20277001


SEED_STRIDE = 8


DESIGN: tuple[tuple[float, int, int, str], ...] = (
    (0.19, 6, 10, "real"),
    (0.21, 6, 10, "real"),
    (0.23, 6, 10, "real"),
    (0.27, 6, 10, "real"),
    (0.29, 6, 10, "real"),
    (0.31, 6, 10, "real"),
    (0.33, 6, 10, "real"),
    (0.35, 8, 9, "continued"),
    (0.37, 8, 9, "continued"),
    (0.39, 8, 9, "continued"),
    (0.44, 8, 9, "continued"),
    (0.47, 8, 9, "continued"),
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _settings() -> dict[str, Any]:
    return {
        "momentum_order": 20,
        "momentum_maximum": 6.0,
        "momentum_panels": 1,
        "momentum_power": 1.25,
        "block_scheme": "c",
        "replicates": 4,
        "radial_power": 0.15,
        "base_seed": BASE_SEED,
        "per_point_seed_stride": SEED_STRIDE,
        "first_contour_wall": 0.4,
        "second_contour_wall": 0.5,
    }


def _empty_payload() -> dict[str, Any]:
    return {
        "status": "sphere_1to4_worldsheet_extension_in_progress",
        "calculation": "genus-zero equal-energy sphere 1->4 amplitude",
        "matrix_model_information_used": False,
        "kinematics": "omega=i*t; signed energies (4*omega,-omega,-omega,-omega,-omega)",
        "normalization": "Q_4=I_5/(16*pi^2*omega^5); mu^3*A=4*i*omega^5*Q_4",
        "moduli_prescription": "120-chart plumbing-coordinate mixture",
        "liouville_contour": (
            "real below t=0.35; continued implementation from t=0.35, "
            "with the first crossed residue active only for t>0.4"
        ),
        "settings": _settings(),
        "design": {
            "target_merged_point_count": 30,
            "historical_point_count": 18,
            "extension_point_count": len(DESIGN),
            "extension_t_values": [entry[0] for entry in DESIGN],
            "selection": (
                "interlace broad historical gaps across the validated ray; "
                "retain the existing t=0.249 and 0.251 zero probes; avoid "
                "t=0.25 and both contour walls"
            ),
        },
        "points": [],
    }


def _load_resume(path: Path) -> dict[str, Any]:
    if not path.exists():
        return _empty_payload()
    payload = json.loads(path.read_text())
    if payload.get("matrix_model_information_used") is not False:
        raise RuntimeError("resume artifact does not certify target-free production")
    if payload.get("settings") != _settings():
        raise RuntimeError("resume artifact settings differ from the frozen design")
    if payload.get("design", {}).get("extension_t_values") != [x[0] for x in DESIGN]:
        raise RuntimeError("resume artifact has a different extension design")
    return payload


def _flat_record(
    raw: dict[str, Any],
    *,
    block_order: int,
    sobol_power: int,
    contour: str,
    seed: int,
) -> dict[str, Any]:
    t_value = float(raw["t"])
    q = raw["Q"]
    sigma = raw["Q_standard_error"]
    q_real = float(q["real"])
    q_imag = float(q["imag"])
    sigma_real = float(sigma["real"])
    if not all(math.isfinite(value) for value in (q_real, q_imag, sigma_real)):
        raise ArithmeticError("non-finite worldsheet output")
    if sigma_real <= 0.0:
        raise ArithmeticError("non-positive worldsheet standard error")
    if contour == "real":
        residue_status = "not applicable"
    elif t_value < 0.4:
        residue_status = "inactive below t=2/5"
    else:
        residue_status = "included"
    return {
        "t": t_value,
        "Q": q_real,
        "Q_imaginary_part": q_imag,
        "Q_standard_error": sigma_real,
        "raw_integral": raw["raw_integral"],
        "raw_standard_error": raw["raw_standard_error"],
        "replicate_estimates": raw["replicate_estimates"],
        "amplitude_gs": raw["amplitude_gs"],
        "block_order": block_order,
        "momentum_order": 20,
        "sobol_power": sobol_power,
        "replicates": 4,
        "contour": contour,
        "residue_status": residue_status,
        "seed": seed,
    }


def run(output_path: Path, manifest_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = _load_resume(output_path)
    completed = {round(float(point["t"]), 12): point for point in payload["points"]}
    for index, (t_value, block_order, sobol_power, contour) in enumerate(DESIGN):
        key = round(t_value, 12)
        if key in completed:
            print(f"skipping completed omega=i*{t_value:.6g}", flush=True)
            continue
        seed = BASE_SEED + SEED_STRIDE * index
        print(
            f"evaluating omega=i*{t_value:.6g}, block={block_order}, "
            f"sobol=2^{sobol_power}, seed={seed}",
            flush=True,
        )
        raw = scan_point(
            t_value,
            block_order=block_order,
            momentum_order=20,
            momentum_maximum=6.0,
            momentum_panels=1,
            momentum_power=1.25,
            block_scheme="c",
            liouville_contour=contour,
            sobol_power=sobol_power,
            replicates=4,
            radial_power=0.15,
            seed=seed,
        )
        completed[key] = _flat_record(
            raw,
            block_order=block_order,
            sobol_power=sobol_power,
            contour=contour,
            seed=seed,
        )
        payload["points"] = [completed[value] for value in sorted(completed)]
        write_json(output_path, payload)

    expected = [round(entry[0], 12) for entry in DESIGN]
    actual = [round(float(point["t"]), 12) for point in payload["points"]]
    if actual != expected:
        raise RuntimeError(f"extension is incomplete: found {actual}, expected {expected}")
    payload["status"] = STATUS
    # Completed checkpoints may predate the explicit amplitude export. Derive
    # it from their unchanged raw integrals; never convert an amplitude twice.
    for point in payload["points"]:
        if "amplitude_gs" not in point:
            point["amplitude_gs"] = amplitude_from_reduced(
                complex(**point["raw_integral"]), complex(**point["raw_standard_error"]),
                factor=2j * math.pi, genus=0, punctures=5, reduced_quantity="I5",
                replicates=[complex(**value) for value in point["replicate_estimates"]],
            )
    payload["point_count"] = len(payload["points"])
    write_json(output_path, payload)
    manifest = {
        "status": STATUS,
        "artifact": str(output_path.resolve()),
        "sha256": sha256_file(output_path),
        "frozen_on": date.today().isoformat(),
        "point_count": len(payload["points"]),
        "t_values": [point["t"] for point in payload["points"]],
        "matrix_model_information_used": False,
    }
    write_json(manifest_path, manifest)
    return payload, manifest


def extension_main() -> None:
    run_dir = (
        data_root("tree_level")
        / "sphere_five_point_1to4"
        / "blind30_20260824"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=run_dir / "worldsheet_extension_12point.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=run_dir / "worldsheet_extension_12point_frozen.json",
    )
    arguments = parser.parse_args()
    _, manifest = run(arguments.output, arguments.manifest)
    print(json.dumps(manifest, indent=2))


def main():
    import sys
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('scan', 'extend'))
    args, rest = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    return (scan_main if args.command == 'scan' else extension_main)()

if __name__ == '__main__':
    main()
