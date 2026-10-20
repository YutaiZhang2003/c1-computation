#!/usr/bin/env python3
"""Re-express the draft's tree amplitudes at n_a=1 as functions of R=1/t.

Uses the current figure inputs: the five-point draft table and the direct
six-point series with its four-point separating collar. No integration,
fitting or interpolation of data is performed. The output contains a matrix-model curve and
crimson numerical markers, in the draft's coupling-stripped g_s convention.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import StrMethodFormatter


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "Machine note/tree_level/figures"
OUTPUT = ROOT / "output/thermal_tree_radius"
PDF_OUTPUT = ROOT / "output/pdf"
SIX_SOURCE = ROOT / "data set/tree_level/sphere_six_point_1to5/direct_series_20260929/curve_results.json"
CRIMSON = "#DC143C"


def read_direct_six_point():
    result = json.loads(SIX_SOURCE.read_text())
    fingerprint = result.pop("fingerprint")
    actual = hashlib.sha256(json.dumps(result, sort_keys=True, separators=(",", ":"),
                                      allow_nan=False).encode()).hexdigest()
    if actual != fingerprint:
        raise ValueError("Direct six-point result fingerprint mismatch")
    if (result["general_six_point_nome_reexpansion"] is not False
            or result["four_point_separating_collar_retained"] is not True):
        raise ValueError("The radius figure requires direct six-point series plus the four-point collar")
    return result


def read_draft_data(outgoing: int) -> tuple[np.ndarray, list[Path]]:
    """Return columns t and the draft's frozen mu-expansion coefficient."""
    if outgoing == 4:
        paths = [SOURCE / f"sphere_1to4_t_{part}.dat"
                 for part in ("no_residue", "residue")]
        tables = [np.genfromtxt(path, names=True) for path in paths]
        values = np.concatenate([
            np.column_stack((table["t"], table["amplitude_real"]))
            for table in tables
        ])
    elif outgoing == 5:
        paths = [SIX_SOURCE]
        values = np.asarray([(row["t"], row["mu4_A"]["mean"][1])
                             for row in read_direct_six_point()["results"]], dtype=float)
    else:
        raise ValueError("Only 1->4 and 1->5 are supported")
    expected = {4: 29, 5: 15}[outgoing]
    if values.shape != (expected, 2):
        raise ValueError(f"Expected {expected} draft points, found {values.shape}")
    if not np.all(np.isfinite(values)) or np.any(values[:, 0] <= 0):
        raise ValueError("Nonfinite data or nonpositive imaginary frequency")
    if len(np.unique(values[:, 0])) != len(values):
        raise ValueError("Duplicate draft frequencies")
    return values, paths


def matrix_component(radius: np.ndarray, outgoing: int) -> np.ndarray:
    """Re A_1->4 or Im A_1->5, with g_s**(outgoing-1) stripped."""
    t = 1.0 / np.asarray(radius)
    polynomial = np.ones_like(t)
    for j in range(1, outgoing - 1):
        polynomial *= j - outgoing * t
    return -(2 * math.pi) ** (outgoing - 1) * outgoing * t ** (outgoing + 1) * polynomial


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["cmr10"],
        "font.size": 11,
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False,
        "axes.labelsize": 12,
        "axes.linewidth": 0.65,
        "axes.edgecolor": "0.3",
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })


def draw_panel(ax, outgoing: int, radii: np.ndarray, amplitudes: np.ndarray) -> None:
    limits = {4: (2.0, 6.0), 5: (3.0, 5.75)}[outgoing]
    radius_curve = np.linspace(*limits, 1601)
    ax.plot(radius_curve, matrix_component(radius_curve, outgoing),
            color="black", linewidth=1.15, label="matrix model", zorder=2)
    ax.plot(radii, amplitudes, linestyle="none", marker="o", markersize=3.5,
            markerfacecolor=CRIMSON, markeredgecolor=CRIMSON,
            markeredgewidth=0, label="numerical data", zorder=3)
    ax.set_xlim(*limits)
    ax.set_ylim(-0.38, 3.36 if outgoing == 4 else 3.55)
    ax.set_xticks([2, 3, 4, 5, 6] if outgoing == 4 else [3, 3.5, 4, 4.5, 5, 5.5])
    ax.set_yticks([0, 1, 2, 3])
    ax.xaxis.set_major_formatter(StrMethodFormatter("${x:g}$"))
    ax.yaxis.set_major_formatter(StrMethodFormatter("${x:g}$"))
    ax.set_xlabel(r"$R$", labelpad=6)
    component = "Re" if outgoing == 4 else "Im"
    ax.set_ylabel(
        rf"$\operatorname{{{component}}}\!\left[\widetilde{{\mathcal{{A}}}}^{{T\,(0)}}_{{1\to {outgoing}}}\right]$",
        labelpad=10,
    )
    ax.grid(True, which="major", color="0.86", linewidth=0.55, zorder=0)
    ax.tick_params(direction="in", top=True, right=True, length=3.8, width=0.5,
                   color="0.3", pad=5)
    ax.legend(loc="upper right", frameon=False, borderaxespad=0.7,
              handlelength=2.1, handletextpad=0.7, labelspacing=0.4)


def main(outgoing_channels=(4, 5)) -> None:
    configure_style()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    PDF_OUTPUT.mkdir(parents=True, exist_ok=True)
    module_path = ROOT / "code/general/conventions/c1_matrix_model.py"
    spec = importlib.util.spec_from_file_location("c1_matrix_model_reference", module_path)
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)

    manifest = {
        "outgoing_matsubara_mode": 1,
        "radius_mapping": "R = 1/t",
        "frequency_convention": "omega_out = i/R; omega_in = N*i/R",
        "string_coupling_convention": "mu^-1 = 2*pi*g_s",
        "plotted_quantity": "coefficient of g_s^(N-1), with thermal zero mode stripped",
        "point_color": CRIMSON,
        "curve_color": "black",
        "error_bars": False,
        "uncertainty_bands": False,
        "fitted_curves": False,
        "figures": [],
    }
    old_manifest = OUTPUT / "manifest.json"
    if old_manifest.exists():
        manifest["figures"] = [entry for entry in json.loads(old_manifest.read_text())["figures"]
                               if entry["outgoing"] not in outgoing_channels]
    for outgoing in outgoing_channels:
        values, sources = read_draft_data(outgoing)
        order = np.argsort(1 / values[:, 0])
        values = values[order]
        t, native = values[:, 0], values[:, 1]
        radius = 1.0 / t
        amplitude = (2 * math.pi) ** (outgoing - 1) * native
        predicted = matrix_component(radius, outgoing)
        reference = np.array([
            model_module.amplitude_coefficient(0, [1j / r] * outgoing)
            for r in radius
        ])
        reference_component = reference.real if outgoing == 4 else reference.imag
        np.testing.assert_allclose(predicted, reference_component, rtol=2e-13, atol=1e-14)
        np.testing.assert_allclose(radius * t, 1.0, rtol=2e-15, atol=0)
        np.testing.assert_allclose(amplitude / (2 * math.pi) ** (outgoing - 1),
                                   native, rtol=2e-15, atol=0)

        stem = f"sphere_1to{outgoing}_thermal_R"
        sampling_errors = ({row["t"]: row["mu4_A"]["sampling_se"][1]
                            for row in read_direct_six_point()["results"]} if outgoing == 5 else {})
        with (OUTPUT / f"{stem}.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["R", "t", "outgoing_matsubara_mode", "amplitude_native_mu_coefficient",
                             "amplitude_gs_coefficient", "matrix_model_gs_coefficient"]
                            + (["sampling_SE_native_mu_coefficient", "sampling_SE_gs_coefficient"]
                               if outgoing == 5 else []))
            for r, freq, value, plotted, model in zip(radius, t, native, amplitude, predicted):
                writer.writerow([f"{r:.17g}", f"{freq:.17g}", 1,
                                 f"{value:.17g}", f"{plotted:.17g}", f"{model:.17g}"]
                                + ([f"{sampling_errors[freq]:.17g}",
                                    f"{sampling_errors[freq] * (2 * math.pi)**4:.17g}"]
                                   if outgoing == 5 else []))

        fig, ax = plt.subplots(figsize=(4.8, 3.1))
        fig.subplots_adjust(left=0.155, right=0.985, bottom=0.19, top=0.975)
        draw_panel(ax, outgoing, radius, amplitude)
        assert len(ax.lines) == 2 and len(ax.collections) == 0 and len(ax.containers) == 0
        pdf_path = PDF_OUTPUT / f"{stem}.pdf"
        fig.savefig(pdf_path, metadata={
            "Title": f"Thermal tree-level 1 to {outgoing}: radius dependence, n_a=1",
            "Subject": "Current worldsheet data mapped with R=1/t; crimson points and black matrix-model curve",
        })
        fig.savefig(OUTPUT / f"{stem}.png", dpi=320)
        plt.close(fig)
        manifest["figures"].append({
            "outgoing": outgoing,
            "component": "real" if outgoing == 4 else "imaginary",
            "point_count": len(radius),
            "data_R_min": float(radius.min()),
            "data_R_max": float(radius.max()),
            "pdf": str(pdf_path.relative_to(ROOT)),
            "csv": str((OUTPUT / f"{stem}.csv").relative_to(ROOT)),
            "sources": [{"path": str(path.relative_to(ROOT)),
                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                        for path in sources],
            "matrix_model_crosscheck_max_abs": float(np.max(np.abs(predicted - reference_component))),
            **({"general_six_point_nome_reexpansion": False,
                "four_point_separating_collar_retained": True,
                "sampling_errors": "Recomputed over eight common scrambles; retained in CSV, not drawn"}
               if outgoing == 5 else {}),
        })

    manifest["figures"].sort(key=lambda entry: entry["outgoing"])
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.25))
    fig.subplots_adjust(left=0.078, right=0.99, bottom=0.185, top=0.975, wspace=0.28)
    for ax, outgoing in zip(axes, (4, 5)):
        values, _ = read_draft_data(outgoing)
        values = values[np.argsort(1 / values[:, 0])]
        radius = 1 / values[:, 0]
        amplitude = (2 * math.pi)**(outgoing - 1) * values[:, 1]
        draw_panel(ax, outgoing, radius, amplitude)
    fig.savefig(OUTPUT / "thermal_tree_radius.png", dpi=260)
    plt.close(fig)
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (OUTPUT / "README.md").write_text("""# Thermal tree-level radius figures

Each outgoing tachyon has Matsubara mode n_a=1, so omega_a=i/R and R=1/t.
The incoming mode is 4 or 5. The ordinate keeps the current draft convention:
the coefficient of g_s^3 for 1->4 or g_s^4 for 1->5, with mu^-1=2*pi*g_s
and the compact zero-mode factor (-iR) times momentum conservation stripped.

All 29 five-point and 15 six-point values are retained and reparameterized in R.
The six-point points now use the direct plumbing series through N=12, with
the existing N13-N18 native corrections and the separately integrated
four-point elliptic collar. All general six-point nome reexpansions are
excluded. The canonical source is
data set/tree_level/sphere_six_point_1to5/direct_series_20260929/curve_results.json.
No new CFT quadrature is performed. Source paths and hashes are recorded in
manifest.json. Six-point sampling errors are recomputed across the eight
common scrambles and saved in the CSV; no error bars or bands are plotted.

The black curves are the matrix-model predictions, with no fitting:

- Re A^T(0)_(1->4) = -(2*pi)^3 * 4/R^5 * (1-4/R)*(2-4/R).
- Im A^T(0)_(1->5) = -(2*pi)^4 * 5/R^6 * (1-5/R)*(2-5/R)*(3-5/R).

The y-values of saved data have only the same (2*pi)^(N-1) conversion already
used in the draft. CSV exports retain both native and plotted values. The
curves were independently checked against c1_matrix_model.py at every point.

Vector PDFs are in ../pdf/sphere_1to4_thermal_R.pdf and
../pdf/sphere_1to5_thermal_R.pdf. PNGs here provide standalone and paired views.
The draft text is unchanged. Previous six-point figure versions are preserved
under output/analysis/sphere_six_direct_series_20260929/figure_update/before.

Reproduce with Python, NumPy and Matplotlib:

    python code/figures/plot_thermal_tree_radius.py

""")
    print(json.dumps({"figures": manifest["figures"],
                      "preview": str(OUTPUT / "thermal_tree_radius.png")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outgoing", nargs="+", type=int, choices=(4, 5), default=[4, 5])
    main(tuple(dict.fromkeys(parser.parse_args().outgoing)))
