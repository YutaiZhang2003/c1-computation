"""Render the eight draft panels using only newly recovered numerical tables."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import numpy as np

from c1_matrix_model import amplitude_coefficient
import plot_thermal_tree_radius as tree
import plot_thermal_one_loop_radius as loop


def render_all(output):
    output = Path(output)
    destination = output / "figures"
    destination.mkdir()
    tree.configure_style()
    panels = []

    def save(figure, stem, metadata):
        for extension in ("pdf", "png"):
            path = destination / f"{stem}.{extension}"
            figure.savefig(path, dpi=160, bbox_inches="tight", pad_inches=.06)
        plt.close(figure)
        panels.append(dict(stem=stem, **metadata))

    def axes():
        figure, ax = plt.subplots(figsize=(4.8, 3.1))
        figure.subplots_adjust(left=.155, right=.985, bottom=.19, top=.975)
        return figure, ax

    for outgoing in (4, 5):
        path = output / "tree_level" / f"{outgoing+1}_point"
        data = np.genfromtxt(path / f"sphere_1to{outgoing}_gs{outgoing-1}.csv",
                             delimiter=",", names=True)
        component = "Re" if outgoing == 4 else "Im"
        column = f"{component}_A_coefficient_gs{outgoing-1}"
        figure, ax = axes()
        limits = (.17, .5) if outgoing == 4 else (.175, .333)
        t = np.linspace(*limits, 501)
        theory = np.array([amplitude_coefficient(0, [1j*x]*outgoing) for x in t])
        ax.plot(t, theory.real if outgoing == 4 else theory.imag,
                color="black", linewidth=1.15, label="matrix model")
        ax.plot(data["t"], data[column], "o", color=tree.CRIMSON, markersize=3.5,
                markeredgewidth=0, label="numerical data")
        ax.set_xlim(*limits)
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(rf"$\operatorname{{{component}}}\mathcal{{A}}^{{(0)}}_{{1\to {outgoing}}}$")
        ax.grid(color=".86", linewidth=.55)
        ax.legend(loc="upper left", frameon=False)
        save(figure, f"sphere_1to{outgoing}_{component.lower()}_amplitude",
             dict(points=len(data), horizontal_variable="outgoing imaginary energy t",
                  coupling_power=outgoing-1))

    data = np.genfromtxt(output / "one_loop/three_point/torus_1to2_gs3.csv",
                         delimiter=",", names=True)
    figure, ax = axes()
    t = np.linspace(0, .5, 501)
    theory = np.array([amplitude_coefficient(1, [1j*x]*2).real for x in t])
    ax.plot(t, theory, color="black", linewidth=1.15, label="matrix model")
    # Receipts use total incoming t; the manuscript axis uses each outgoing t.
    ax.plot(data["t"]/2, data["Re_A_coefficient_gs3"], "o", color=tree.CRIMSON,
            markersize=3.5, markeredgewidth=0, label="numerical data")
    ax.set_xlim(0, .5)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\operatorname{Re}\mathcal{A}^{(1)}_{1\to 2}(it,it)$")
    ax.grid(color=".86", linewidth=.55)
    ax.legend(loc="upper left", frameon=False)
    save(figure, "torus_1to2_re_amplitude",
         dict(points=len(data), horizontal_variable="saved incoming t / 2", coupling_power=3))

    for outgoing in (4, 5):
        data = np.genfromtxt(output / "tree_level" / f"{outgoing+1}_point" /
                             f"sphere_1to{outgoing}_thermal_R.csv", delimiter=",", names=True)
        figure, ax = axes()
        tree.draw_panel(ax, outgoing, data["R"], data["amplitude_gs_coefficient"])
        save(figure, f"sphere_1to{outgoing}_thermal_R",
             dict(points=len(data), horizontal_variable="R = 1/t", coupling_power=outgoing-1))

    for outgoing, source, column in (
        (1, "one_loop/two_point/one_loop_1to1_R.csv", "amplitude_imag"),
        (2, "one_loop/three_point_thermal/thermal_draft_results.csv", "worldsheet_thermal_gs3"),
    ):
        data = np.genfromtxt(output/source, delimiter=",", names=True)
        order = np.argsort(data["R"])
        data = data[order]
        figure, ax = axes()
        loop.draw_panel(ax, outgoing, data["R"], data[column])
        predicted = loop.matrix_component(data["R"], outgoing)
        saved_column = "matrix_model_imag" if outgoing == 1 else "matrix_model_thermal"
        np.testing.assert_allclose(predicted, data[saved_column], rtol=2e-13, atol=1e-14)
        save(figure, f"torus_1to{outgoing}_thermal_R",
             dict(points=len(data), horizontal_variable="radius R", coupling_power=outgoing+1))

    data = np.genfromtxt(output/"two_loop/genus2_free_energy_F2.csv",
                         delimiter=",", names=True)
    figure, ax = axes()
    radius = np.geomspace(.5, 16, 1001)
    model = -np.pi*(7+10/radius**2+7/radius**4)/2880
    ax.plot(radius, model, color="black", linewidth=1.15, label="matrix model")
    ax.plot(data["R"], data["F2_coefficient_gs2"], "o", color=tree.CRIMSON,
            markersize=3, markeredgewidth=0, label="numerical data")
    ax.set_xscale("log", base=2)
    ax.set_xlim(.5, 16)
    ax.set_xticks([.5, 1, 2, 4, 8, 16])
    ax.xaxis.set_major_formatter(StrMethodFormatter("${x:g}$"))
    ax.set_ylim(-.19, 0)
    ax.set_yticks([-.18, -.12, -.06, 0])
    ax.set_xlabel(r"$R$")
    ax.set_ylabel(r"$F_2(R)$")
    ax.grid(color=".86", linewidth=.55)
    ax.legend(loc="lower right", frameon=False)
    save(figure, "genus2_free_energy_F2",
         dict(points=len(data), horizontal_variable="radius R, logarithmic base 2",
              coupling_power=2))
    if len(panels) != 8:
        raise ValueError("Expected eight numerical panels")
    import json
    (destination/"panels.json").write_text(json.dumps({
        "panels": panels, "numerical_values": "recomputed from bundled inputs",
        "error_bars": False, "layout": "Matplotlib rendering of the draft panels",
        "matrix_model_used_to_generate_worldsheet_points": False,
    }, indent=2) + "\n")
    return sorted(str(path.relative_to(output)) for path in destination.iterdir())
