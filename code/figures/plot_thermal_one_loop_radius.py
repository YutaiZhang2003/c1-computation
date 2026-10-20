#!/usr/bin/env python3
"""Format the saved finite-temperature one-loop scans for draft Section 4.

Use the complete-collision two-point scan and the refined draft three-point
baseline plus independently integrated thermal corrections. The latter
remains preliminary. This exporter never synthesizes numerical data from
the matrix-model prediction.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import plot_thermal_tree_radius as tree
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "output/thermal_one_loop_radius"
PDF_OUTPUT = ROOT / "output/pdf"
TWO_POINT = ROOT / "output/thermal_one_loop_20260927"
THREE_POINT = ROOT / "output/thermal_draft_curve20_20260928"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_pinned(folder: Path, name: str) -> Path:
    path = folder / name
    manifest = json.loads((folder / "delivery_manifest.json").read_text())
    if digest(path) != manifest["files"][name]:
        raise ValueError(f"Saved source differs from its delivery manifest: {path}")
    return path


def read_data(outgoing: int) -> tuple[np.ndarray, list[Path]]:
    if outgoing == 1:
        source = read_pinned(TWO_POINT, "one_loop_1to1_R.csv")
        receipt = read_pinned(TWO_POINT, "one_loop_amplitudes_R.json")
        saved = json.loads(receipt.read_text())["two_point"]
        with source.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        values = []
        for row in rows:
            radius = float(row["R"])
            reference = next(r for r in saved if r["R"] == radius)
            amplitude = float(row["amplitude_imag"])
            assert amplitude == reference["amplitude_imag"]
            assert int(row["power_of_gs"]) == 2
            assert abs(radius * float(row["omega_imag"]) - 1) < 1e-14
            values.append((radius, amplitude, float(row["matrix_model_imag"])))
        expected = 10
    elif outgoing == 2:
        source = read_pinned(THREE_POINT, "thermal_draft_results.csv")
        receipt = read_pinned(THREE_POINT, "thermal_draft_results.json")
        saved = json.loads(receipt.read_text())
        assert saved["assembly_complete"] and saved["maximum_nodes_per_axis"] == 12
        assert not saved["matrix_model_used_for_numerical_values"]
        assert not saved["full_amplitude_certified"]
        with source.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        values = []
        for row in rows:
            radius = float(row["R"])
            reference = next(r for r in saved["rows"] if r["R"] == radius)
            amplitude = float(row["worldsheet_thermal_gs3"])
            assert amplitude == reference["worldsheet_thermal_gs3"]
            assert amplitude == reference["baseline_gs3"] + reference["thermal_correction"]
            assert abs(radius * float(row["t"]) - 2) < 1e-14
            values.append((radius, amplitude, float(row["matrix_model_thermal"])))
        expected = 15
    else:
        raise ValueError("Only genus-one 1->1 and 1->2 are supported")
    values = np.array(sorted(values), dtype=float)
    assert values.shape == (expected, 3) and np.all(np.isfinite(values))
    assert len(np.unique(values[:, 0])) == expected
    return values, [source, receipt, source.parent / "delivery_manifest.json"]


def matrix_component(radius: np.ndarray, outgoing: int) -> np.ndarray:
    inverse_radius = 1.0 / np.asarray(radius)
    if outgoing == 1:
        return math.pi**2 / 6 * inverse_radius**2 * (inverse_radius**2 - 1)
    if outgoing == 2:
        t = 2 * inverse_radius
        return math.pi**3 / 12 * t**3 * (t - 1) * (t - 2) * (1 + t - t**2 / 4)
    raise ValueError("Only genus-one 1->1 and 1->2 are supported")


def unspecialized_matrix_component(radius: np.ndarray, outgoing: int) -> np.ndarray:
    """Check the reduced polynomials against the continued thermal expressions."""
    omega = 1j / radius
    if outgoing == 1:
        amplitude = -(2 * math.pi)**2 / 24 * omega**2 * (omega - 1j) * (
            1 - 1j * omega + omega**2 + 1 / radius**2)
        np.testing.assert_allclose(amplitude.real, 0, atol=1e-15)
        return amplitude.imag
    total = 2 * omega
    amplitude = -1j * math.pi**3 / 3 * omega**2 * total * (total - 1j) * (
        total - 2j) * (2 * omega**2 - 1j * total + 1 + 1 / radius**2)
    np.testing.assert_allclose(amplitude.imag, 0, atol=1e-15)
    return amplitude.real


def draw_panel(ax, outgoing: int, radius: np.ndarray, amplitude: np.ndarray) -> None:
    limits = (1.9, 6.7) if outgoing == 1 else (2.0, 8.2)
    curve_radius = np.linspace(*limits, 1601)
    ax.plot(curve_radius, matrix_component(curve_radius, outgoing), color="black",
            linewidth=1.15, label="matrix model", zorder=2)
    ax.plot(radius, amplitude, linestyle="none", marker="o", markersize=3.5,
            markerfacecolor=tree.CRIMSON, markeredgecolor=tree.CRIMSON,
            markeredgewidth=0, label="numerical data", zorder=3)
    ax.set_xlim(*limits)
    ax.set_xticks([2, 3, 4, 5, 6] if outgoing == 1 else [2, 3, 4, 5, 6, 7, 8])
    ax.set_ylim(*((-0.35, 0.02) if outgoing == 1 else (-0.04, 0.64)))
    ax.set_yticks([-0.3, -0.2, -0.1, 0] if outgoing == 1 else [0, 0.2, 0.4, 0.6])
    ax.xaxis.set_major_formatter(StrMethodFormatter("${x:g}$"))
    ax.yaxis.set_major_formatter(StrMethodFormatter("${x:g}$"))
    ax.set_xlabel(r"$R$", labelpad=6)
    component = "Im" if outgoing == 1 else "Re"
    ax.set_ylabel(
        rf"$\operatorname{{{component}}}\!\left[\widetilde{{\mathcal{{A}}}}^{{T\,(1)}}_{{1\to {outgoing}}}\right]$",
        labelpad=4 if outgoing == 1 else 10,
    )
    ax.grid(True, which="major", color="0.86", linewidth=0.55, zorder=0)
    ax.tick_params(direction="in", top=True, right=True, length=3.8, width=0.5,
                   color="0.3", pad=5)
    ax.legend(loc="lower right" if outgoing == 1 else "upper right", frameon=False,
              borderaxespad=0.7, handlelength=2.1, handletextpad=0.7, labelspacing=0.4)
    assert len(ax.lines) == 2 and len(ax.collections) == 0 and len(ax.containers) == 0
    np.testing.assert_array_equal(ax.lines[1].get_xdata(), radius)
    np.testing.assert_array_equal(ax.lines[1].get_ydata(), amplitude)


def main() -> None:
    tree.configure_style()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    PDF_OUTPUT.mkdir(parents=True, exist_ok=True)
    audit_path = ROOT / "output/thermal_radius_audit_20260927/matrix_model_audit.json"
    assert json.loads(audit_path.read_text())["status"] == "passed"
    manifest = {
        "outgoing_matsubara_mode": 1,
        "external_winding": 0,
        "amplitude_notation": r"\mathcal{A}^{T\,(1)}_{1\to n}",
        "normalization": "Already in the draft g_s convention; coefficients of g_s^2 and g_s^3, with compact zero mode stripped.",
        "data_fitted_or_rescaled": False,
        "new_integration_performed": True,
        "error_bars": False,
        "uncertainty_bands": False,
        "point_color": tree.CRIMSON,
        "curve_color": "black",
        "matrix_model_audit": {"path": str(audit_path.relative_to(ROOT)), "sha256": digest(audit_path)},
        "figures": [],
    }
    panels = []
    for outgoing in (1, 2):
        values, sources = read_data(outgoing)
        radius, amplitude, saved_prediction = values.T
        prediction = matrix_component(radius, outgoing)
        np.testing.assert_allclose(prediction, saved_prediction, rtol=2e-13, atol=1e-14)
        np.testing.assert_allclose(prediction, unspecialized_matrix_component(radius, outgoing),
                                   rtol=2e-13, atol=1e-14)
        stem = f"torus_1to{outgoing}_thermal_R"
        with (OUTPUT / f"{stem}.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["R", "outgoing_matsubara_mode", "power_of_gs", "component",
                             "worldsheet_gs_coefficient", "matrix_model_gs_coefficient", "status"])
            for r, value, model in zip(radius, amplitude, prediction):
                writer.writerow([f"{r:.17g}", 1, outgoing + 1,
                                 "imaginary" if outgoing == 1 else "real",
                                 f"{value:.17g}", f"{model:.17g}",
                                 "finite numerical quadrature" if outgoing == 1 else "preliminary retained estimator"])
        fig, ax = plt.subplots(figsize=(4.8, 3.1))
        fig.subplots_adjust(left=0.155, right=0.985, bottom=0.19, top=0.975)
        draw_panel(ax, outgoing, radius, amplitude)
        pdf = PDF_OUTPUT / f"{stem}.pdf"
        fig.savefig(pdf, metadata={
            "Title": f"Thermal one-loop 1 to {outgoing}: radius dependence, n_a=1",
            "Subject": "Crimson worldsheet points and black matrix-model curve; no error bars."
                       + (" The 1 to 2 points are preliminary retained estimates." if outgoing == 2 else ""),
        })
        fig.savefig(OUTPUT / f"{stem}.png", dpi=320)
        plt.close(fig)
        panels.append((outgoing, radius, amplitude))
        manifest["figures"].append({
            "outgoing": outgoing, "point_count": len(radius),
            "component": "imaginary" if outgoing == 1 else "real",
            "data_R_min": float(radius.min()), "data_R_max": float(radius.max()),
            "pdf": str(pdf.relative_to(ROOT)),
            "sources": [{"path": str(path.relative_to(ROOT)), "sha256": digest(path)} for path in sources],
            "saved_matrix_model_max_abs_difference": float(np.max(np.abs(prediction - saved_prediction))),
            "preliminary": outgoing == 2,
        })

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.25))
    fig.subplots_adjust(left=0.078, right=0.99, bottom=0.185, top=0.975, wspace=0.28)
    for ax, panel in zip(axes, panels):
        draw_panel(ax, *panel)
    fig.savefig(OUTPUT / "thermal_one_loop_radius.png", dpi=260)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(9.8, 6.9))
    fig.subplots_adjust(left=0.078, right=0.99, bottom=0.09, top=0.975, wspace=0.28, hspace=0.38)
    for ax, outgoing in zip(axes[0], (4, 5)):
        values, _ = tree.read_draft_data(outgoing)
        values = values[np.argsort(1 / values[:, 0])]
        tree.draw_panel(ax, outgoing, 1 / values[:, 0], (2 * math.pi)**(outgoing - 1) * values[:, 1])
    for ax, panel in zip(axes[1], panels):
        draw_panel(ax, *panel)
    fig.savefig(OUTPUT / "thermal_amplitudes_R.png", dpi=260)
    plt.close(fig)

    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (OUTPUT / "README.md").write_text("""# Thermal one-loop figures for draft Section 4

The two PDFs use the tree-level figures' dimensions, fonts, thin gray grid,
crimson numerical points, and black matrix-model curves. There are no error
bars, uncertainty bands, fits, or normalization adjustments. The labels use
\\mathcal{A}^{T\\,(1)} with no frequency argument.

The 1->1 points are all ten complete-collision thermal values in
../thermal_one_loop_20260927/one_loop_1to1_R.csv (R=2 to 6.5).
The 1->2 points use the 15 radii from R=40/19 through R=8. Each is the current
refined zero-temperature value from draft Figure 4(c), converted once to
the g_s^3 convention, plus an independently integrated thermal correction.
The later refinement is selected at t=0.25 and t=0.95. The full source table,
including both earlier values, is preserved in the numerical provenance.
The numerical inputs are ../thermal_draft_curve20_20260928/thermal_draft_results.csv.
No thermal corrections are computed above R=8 in this campaign.

Both exported inputs already give the coefficients in the draft's g_s convention:
g_s^2 for 1->1 and g_s^3 for 1->2. No extra (2*pi) factor is applied.
The compact zero mode is stripped. All outgoing Matsubara numbers are 1;
the incoming numbers are respectively 1 and 2. External winding is zero.

With x=1/R and t=2/R, the black curves are

- Im A^(T (1))_(1->1) = pi^2/6 * x^2 * (x^2-1).
- Re A^(T (1))_(1->2) = pi^3/12 * t^3 * (t-1)*(t-2)*(1+t-t^2/4).

These include the genus-one thermal correction. They are not obtained by
substituting Matsubara frequencies into the zero-temperature curve alone.
The formulas were cross-checked against the input predictions, the continued
unspecialized expressions, and the prior independent matrix-model audit in
../thermal_radius_audit_20260927/matrix_model_audit.json. The finite thermal
Matsubara prescription is described in
[BRY Section 2.3](https://arxiv.org/html/2204.01747v1#S2.SS3); use the bosonic
energy/radius convention here, with mu^(-1)=2*pi*g_s.

The 1->2 points remain preliminary retained estimates. Fresh momentum grids
are used in primary and collision regions. Further coupled momentum/block
convergence and omitted collision and finite-tube remainders remain unresolved.
The draft caption retains this qualification. Sampling errors alone do not
bound these effects and have not been plotted. All original receipts and
uncertainties remain in their source folders. New integrations compute the
thermal-minus-noncompact correction at identical moduli and momentum nodes.

Source SHA256 hashes, point counts, and convention checks are in manifest.json.
The CSV files here preserve the exact plotted values. The PDF files are in
../pdf/torus_1to1_thermal_R.pdf and ../pdf/torus_1to2_thermal_R.pdf.

Reproduce with Python, NumPy, and Matplotlib from the repository root:

    python code/figures/plot_thermal_one_loop_radius.py
""")
    print(json.dumps({"figures": manifest["figures"],
                      "preview": str(OUTPUT / "thermal_amplitudes_R.png")}, indent=2))


if __name__ == "__main__":
    main()
