"""Assemble thermal 1->1 receipts and optionally render the draft-style figure."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from genus1_two_point_thermal_production import load_settings
from genus1_two_point_coefficient_bank import validate_metadata


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def matrix_model(radius):
    x = 1 / radius
    return math.pi**2 / 6 * x*x * (x*x - 1)


def collect(scan_dir, radii=None):
    """Preserve the plotted coefficient and its sampling errors from each receipt."""
    radii = load_settings()["radii"] if radii is None else list(radii)
    rows, hashes = [], {}
    for radius in sorted(radii):
        path = Path(scan_dir) / f"R{radius:g}.json"
        d = json.loads(path.read_text())
        validate_metadata(d.get("kernel", {}))
        if d["R"] != radius or not d["collision_remainder_included"]:
            raise ValueError("Receipt does not cover the full collision disc: " + str(path))
        R = d["R"]
        source = str(path.relative_to(Path(scan_dir).parent))
        hashes[source] = digest(path)
        rows.append(dict(R=R,T=d["T"],omega_imag=1/R,power_of_gs=2,
            amplitude_real=0.,amplitude_imag=d["thermal_imag"],
            sampling_SE=d["sampling_SE_thermal"],
            zero_temperature_imag=d["zero_temperature_imag"],
            thermal_correction_imag=d["thermal_correction_imag"],
            thermal_correction_sampling_SE=d["sampling_SE_correction"],
            matrix_model_imag=matrix_model(R),
            matrix_model_thermal_correction_imag=math.pi**2/6/R**4*(1/R-1),
            fractional_difference=d["thermal_imag"]/matrix_model(R)-1,
            status="finite_block_and_momentum_worldsheet_quadrature",source=source))
    return rows, hashes


def assemble(scan_dir, output_dir, radii=None):
    rows, hashes = collect(scan_dir, radii)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "one_loop_1to1_R.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(
        normalization="Im A^(T,(1))_(1->1) = 2*pi^2*Re I_T; coefficient of g_s^2; compact zero mode stripped.",
        external_KK=[1,-1], external_winding=[0,0],
        worldsheet_normalization_fitted_to_matrix_model=False,
        two_point=rows, source_sha256=hashes,
        limitations=["Sampling standard errors exclude finite momentum and conformal-block errors.",
                     "Radii share random scrambles; estimates at different radii are correlated."])
    receipt = output_dir / "one_loop_amplitudes_R.json"
    receipt.write_text(json.dumps(summary, indent=2) + "\n")
    manifest = {p.name: digest(p) for p in (csv_path, receipt)}
    (output_dir / "delivery_manifest.json").write_text(json.dumps(
        dict(format="thermal-one-loop-two-point-delivery-v1", files=manifest), indent=2) + "\n")
    return rows


def plot(rows, output_dir):
    import numpy as np
    import matplotlib.pyplot as plt
    import plot_thermal_tree_radius as style
    from plot_thermal_one_loop_radius import draw_panel, matrix_component, unspecialized_matrix_component
    style.configure_style()
    radius = np.array([r["R"] for r in rows])
    amplitude = np.array([r["amplitude_imag"] for r in rows])
    np.testing.assert_allclose(matrix_component(radius,1), unspecialized_matrix_component(radius,1), rtol=2e-13)
    fig, ax = plt.subplots(figsize=(4.8, 3.1))
    fig.subplots_adjust(left=.155, right=.985, bottom=.19, top=.975)
    draw_panel(ax,1,radius,amplitude)
    stem = Path(output_dir) / "torus_1to1_thermal_R"
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=320)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scan-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--radii", type=float, nargs="+")
    p.add_argument("--plot", action="store_true")
    a = p.parse_args()
    rows = assemble(a.scan_dir,a.output_dir,a.radii)
    if a.plot:
        plot(rows,a.output_dir)
    print(json.dumps(dict(points=len(rows), output=str(a.output_dir))))


if __name__ == "__main__":
    main()
