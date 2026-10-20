#!/usr/bin/env python3
"""Publish the saved direct six-point series and update its two figure inputs.

No conformal-block coefficients or momentum integrals are recomputed. The
separately integrated four-point collar is retained. General six-point nome
reexpansions, including transformed finite boundary corrections, are excluded.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT / "output/analysis/sphere_six_direct_series_20260929"
COMPARISON = ANALYSIS / "comparison.json"
DATA = ROOT / "data set/tree_level/sphere_six_point_1to5/direct_series_20260929"
FIGURES = ROOT / "Machine note/tree_level/figures"
STEM = "sphere_1to5_im_amplitude"
CONVERSION = 1j / (8 * np.pi**3)
COUPLING = (2 * np.pi)**4
EXCLUDED = ("native-nome", "native-multinome", "primary-face-nome")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pair(z):
    return [float(complex(z).real), float(complex(z).imag)]


def statistics(replicates):
    return {"mean": pair(replicates.mean()),
            "sampling_se": [float(replicates.real.std(ddof=1) / np.sqrt(len(replicates))),
                            float(replicates.imag.std(ddof=1) / np.sqrt(len(replicates)))]}


def preserve_previous():
    """Keep the first pre-update files, including their content hashes."""
    backup = ANALYSIS / "figure_update/before"
    paths = [FIGURES / f"{STEM}{suffix}" for suffix in
             (".tex", ".pdf", ".png", "_1200dpi.png")]
    paths += [ROOT / "output/pdf/sphere_1to5_thermal_R.pdf"]
    paths += [ROOT / "output/thermal_tree_radius" / name for name in
              ("sphere_1to5_thermal_R.png", "sphere_1to5_thermal_R.csv",
               "thermal_tree_radius.png", "manifest.json", "README.md")]
    manifest = {}
    for path in paths:
        relative = path.relative_to(ROOT)
        destination = backup / relative
        if path.is_file() and not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
        if destination.is_file():
            manifest[str(relative)] = digest(destination)
    (backup / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def prepare():
    preserve_previous()
    report = json.loads(COMPARISON.read_text())
    parent = Path(report["source_report"])
    assert digest(parent) == report["source_report_sha256"]
    parent_report = json.loads(parent.read_text())
    DATA.mkdir(parents=True, exist_ok=True)
    rows, all_replicates, source_hashes = [], [], {}
    assert len(report["results"]) == len(parent_report["results"]) == 15
    for item, old in zip(report["results"], parent_report["results"]):
        source = Path(item["source"])
        assert digest(source) == item["source_sha256"]
        assert item["t"] == old["t"]
        source_hashes[str(source)] = digest(source)
        with np.load(source, allow_pickle=False) as a:
            baseline = a["variants"].tolist().index("baseline")
            methods = a["common_methods"][0]
            restored = np.isin(methods, EXCLUDED)
            current = a["hybrid_I6"][0, :, :, baseline]
            raw = a["original_raw_native_I6"][0, :, :, baseline, :]
            treated = a["original_treated_prefix_I6"][0, :, :, baseline, -1]
            correction = a["refinement_delta_I6"][0]
            collar = a["collar_addback_I6"][0, -1, -1]
            assert current.shape == (8, 256) and not a["inside_patch"].any()
            assert a["original_orders"].tolist() == [8, 10, 12]
            assert np.all(correction[restored] == 0)
            np.testing.assert_allclose(current, treated + correction + collar,
                                       rtol=2e-12, atol=1e-14)
            # Direct C12 plus the saved higher native levels and one addback.
            direct = raw[..., -1] + correction + collar
            np.testing.assert_allclose(direct[~restored], current[~restored],
                                       rtol=2e-12, atol=1e-14)
            assert np.isfinite(direct).all()
            replicas_I6 = direct.mean(axis=1)
            replicas_A = replicas_I6 * CONVERSION
            expected = item["direct_plus_four_point_collar_mu4_A"]
            np.testing.assert_allclose(pair(replicas_A.mean()), expected["mean"],
                                       rtol=2e-12, atol=1e-18)
            np.testing.assert_allclose(statistics(replicas_A)["sampling_se"],
                                       expected["sampling_se"], rtol=2e-12, atol=1e-20)
            orders = a["refined_point_order"][0].copy()
            orders[restored] = 12
            prefix_failed = restored & (np.max(
                a["original_native_relative_absolute_increments"][0, :, :, baseline, :],
                axis=-1) >= .0025)
            pointwise = DATA / f"{item['label']}.npz"
            np.savez_compressed(
                pointwise, t=item["t"], weighted_I6=direct,
                direct_C12_I6_without_addback=raw[..., -1],
                preserved_native_refinement_I6=correction,
                analytic_four_point_addback_I6=collar, total_level=orders,
                restored_to_direct_C12=restored,
                restored_native_prefix_screen_failed=prefix_failed,
                previous_weighted_I6=current,
                restored_C8_C10_C12_I6=np.where(restored[..., None], raw, 0),
            )
        all_replicates.append(replicas_A)
        rows.append({
            "t": item["t"], "R": 1 / item["t"], "label": item["label"],
            "I6": statistics(replicas_I6), "mu4_A": statistics(replicas_A),
            "amplitude_gs_coefficient": statistics(replicas_A * COUPLING),
            "mu4_A_replicates": [pair(z) for z in replicas_A],
            "pointwise_file": pointwise.name, "pointwise_sha256": digest(pointwise),
            "restored_sample_count": int(restored.sum()),
            "restored_native_prefix_screen_failed": int(prefix_failed.sum()),
            "preserved_C18_cap_reached_count": old["cap_reached_unconverged_contributions"]["count"],
            "level_counts": {str(int(n)): int(np.count_nonzero(orders == n))
                             for n in np.unique(orders)},
            "relative_change_from_previous_percent": item["absolute_relative_amplitude_shift_percent"],
            "restored_N10_to_N12_amplitude_change_percent": item["affected_native_10_to_12_net_shift_percent"],
        })
    replicas = np.asarray(all_replicates)
    assert replicas.shape == (15, 8)
    assert sum(row["restored_sample_count"] for row in rows) == 4950
    assert sum(row["preserved_C18_cap_reached_count"] for row in rows) == 166
    payload = {
        "version": "direct-six-point-series-20260929",
        "representation_policy": "direct-six-point-series-with-four-point-separating-collar",
        "general_six_point_nome_reexpansion": False,
        "four_point_separating_collar_retained": True,
        "excluded_pointwise_representations": list(EXCLUDED),
        "samples_per_energy": 2048, "scrambles": 8, "points_per_scramble": 256,
        "source_comparison_sha256": digest(COMPARISON),
        "source_report_sha256": digest(parent), "source_pointwise_sha256": source_hashes,
        "amplitude_convention": "mu^4 A = i I6/(8 pi^3); coefficient of g_s^4 = (2 pi)^4 mu^4 A = 2 pi i I6",
        "radius_convention": "R=1/t, each outgoing KK number is one, no external winding",
        "mean_covariance_mu4_A_real_then_imag": (
            np.cov(np.concatenate((replicas.real, replicas.imag), axis=0), ddof=1) / 8).tolist(),
        "uncertainty_note": "Sampling SE over the eight common scrambles. Finite block-level changes are separate diagnostics, not bounds on omitted levels or momentum tails. Restored C12 evaluations retain failed raw prefix flags; the previous 166 C18 cap-reached contributions are also retained.",
        "results": rows,
    }
    payload["fingerprint"] = hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    (DATA / "curve_results.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    with (DATA / "amplitudes.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "R", "Im_mu4_A", "sampling_SE_mu4_A", "Im_A_gs_coefficient",
                         "sampling_SE_gs_coefficient", "change_from_previous_percent"])
        for row in rows:
            writer.writerow([row["t"], row["R"], row["mu4_A"]["mean"][1],
                             row["mu4_A"]["sampling_se"][1], row["amplitude_gs_coefficient"]["mean"][1],
                             row["amplitude_gs_coefficient"]["sampling_se"][1],
                             row["relative_change_from_previous_percent"]])
    template = (FIGURES / f"{STEM}.tex").read_text()
    table = "t ImA SE\\\\\n" + "".join(
        f"{r['t']:.4f} {r['mu4_A']['mean'][1]:.17g} {r['mu4_A']['sampling_se'][1]:.17g}\\\\\n"
        for r in rows)
    pattern = r"(\\pgfplotstableread\[row sep=\\\\\]\s*\{\s*).*?(\}\\oneToFiveData)"
    updated, count = re.subn(pattern, lambda m: m[1] + table + m[2], template, flags=re.DOTALL)
    assert count == 1
    header = (
        "% Direct six-point plumbing series with four-point separating collar, 2026-09-29.\n"
        "% General six-point nome reexpansion is disabled for every plotted value.\n"
        "% Restored 4950 pointwise values to N12; existing N13-N18 native corrections retained.\n"
        "% All 15 energies, eight common scrambles of 256 samples, unchanged momentum rules.\n"
        "% Separately integrated four-point elliptic collar addback retained once.\n"
        "% Failed native prefix and C18-cap flags retained; SE below is sampling only, not drawn.\n"
        "% Source: data set/tree_level/sphere_six_point_1to5/direct_series_20260929/curve_results.json\n"
        f"% Source result fingerprint: {payload['fingerprint']}\n"
        "% Coefficient of g_s^4: multiply the mu^-4 values below by (2*pi)^4.\n"
    )
    updated = header + updated[updated.index(r"\documentclass"):]
    (FIGURES / f"{STEM}.tex").write_text(updated)
    (DATA / "README.md").write_text(f"""# Direct six-point series: current figure data

Updated on 29 September 2026. The amplitude and thermal radius figures use the
direct six-point plumbing series. General nome reexpansions are excluded from
all 30,720 retained sample-energy evaluations. The separately integrated
four-point elliptic collar is retained once per integral.

The 4,950 previously transformed evaluations, including 101 pointwise boundary
completions with transformed finite corrections, are restored to total level
N=12. Existing direct-series refinements through N=18 at other points are
preserved. All momentum rules, moduli samples, weights and normalization stay
the same. The maximum amplitude shift is
{max(r['relative_change_from_previous_percent'] for r in rows):.8f}%.

Sampling standard errors are recomputed across the eight common scrambles.
The original failed raw-prefix flags at restored points and the 166 finite
C18 cap-reached contributions remain recorded. No claim of pointwise
convergence or bound on omitted levels follows from the small mean shift.
Changing only the restored terms from N=10 to N=12 changes the full amplitude
by 0.070466%-0.184572%. The figures retain their original marker-only style;
sampling errors and finite-order comparisons are saved separately.

## Files

- `curve_results.json`: means, sampling errors, covariance, flags and provenance.
- `amplitudes.csv`: all 15 amplitudes in both coupling conventions and R=1/t.
- `t*.npz`: direct weighted samples, preserved native corrections, collar term,
  previous samples and the restored finite-order comparison values.

The current figures are `Machine note/tree_level/figures/sphere_1to5_im_amplitude.pdf`
and `output/pdf/sphere_1to5_thermal_R.pdf`. Their shared numerical source is this
directory. The draft text is not edited. Previous figure versions are under
`output/analysis/sphere_six_direct_series_20260929/figure_update/before`.

Rebuild data and the self-contained amplitude TeX source from the frozen
comparison and its original saved arrays:

```sh
python3 code/figures/replot_sphere_six_direct_series.py --source-only
```

After validating the source, export the amplitude PDF/PNG with `--render-only`.
Regenerate the radius plot with
`python3 code/figures/plot_thermal_tree_radius.py --outgoing 5`.
The original arrays are needed only to rebuild the reduction; plotting uses
the portable local result. No new CFT quadrature is performed.
""")
    print(json.dumps({"data": str(DATA), "fingerprint": payload["fingerprint"],
                      "restored_samples": 4950, "figure_source": str(FIGURES / f"{STEM}.tex")}))


def render():
    """Export the validated, self-contained PGFPlots source to PDF and PNG."""
    build = ANALYSIS / "figure_update/build"
    build.mkdir(parents=True, exist_ok=True)
    compiler = shutil.which("pdflatex")
    renderer = shutil.which("pdftoppm")
    if compiler is None or renderer is None:
        raise RuntimeError("The existing pdflatex and pdftoppm tools are required for export")
    result = subprocess.run([compiler, "-halt-on-error", "-interaction=nonstopmode",
                             "-output-directory", str(build), str(FIGURES / f"{STEM}.tex")],
                            cwd=FIGURES, capture_output=True, text=True, timeout=60)
    if result.returncode:
        raise RuntimeError(result.stdout[-5000:] + result.stderr[-1000:])
    target = FIGURES / f"{STEM}.pdf"
    shutil.copy2(build / f"{STEM}.pdf", target)
    for dpi, suffix in ((240, ""), (1200, "_1200dpi")):
        subprocess.run([renderer, "-r", str(dpi), "-png", "-singlefile", str(target),
                        str(FIGURES / f"{STEM}{suffix}")], check=True, timeout=60)
    print(json.dumps({"pdf": str(target), "png": str(FIGURES / f"{STEM}.png")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--source-only", action="store_true")
    mode.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    draft = ROOT / "draft/c1_higher_genus.tex"
    before = digest(draft)
    prepare() if args.source_only else render()
    assert digest(draft) == before, "Draft text changed during figure generation"
