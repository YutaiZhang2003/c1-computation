"""Audit the frozen BRY-v7 result and export it in g=mu^-1=2*pi*g_s units.

This does not run a new integration, change its normalization prescription,
or edit the draft or its existing figure. The PDF plots F_2(R), with g^2
stripped off, so no numerical value of the coupling is assumed.
"""
from pathlib import Path
import csv
import hashlib
import json
import math

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "data set/two_loop/tail_completion"
OUTPUT = ROOT / "data set/two_loop/g_convention_audit_20260911"
FIGURE = ROOT / "output/pdf/genus2_free_energy_g_convention_20260911.tex"
SCALE = 4 * math.pi**2


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reconstruct(groups):
    mean = np.zeros(groups[0].shape[1])
    covariance = np.zeros((len(mean), len(mean)))
    for group in groups:
        n = len(group)
        assert n >= 2 and np.all(np.isfinite(group))
        center = group.mean(axis=0)
        delta = group - center
        mean += center
        covariance += delta.T @ delta / (n * (n - 1))
    return mean, covariance


def fit(radii, values, covariance):
    # Match the declared estimator, independently using whitened least squares.
    # Full covariance, including all cross-radius terms, enters its uncertainty.
    design = np.column_stack((radii + radii**-3, radii**-1))
    error = np.sqrt(np.diag(covariance))
    whitened = design / error[:, None]
    operator = np.linalg.lstsq(whitened, np.diag(1 / error), rcond=None)[0]
    coefficient = operator @ values
    coefficient_covariance = operator @ covariance @ operator.T
    return coefficient, coefficient_covariance


def main():
    source_files = [SOURCE / name for name in (
        "tail_audit_arrays.npz", "tail_audit_radius_values.csv", "tail_audit_summary.json"
    )]
    source_files += [ROOT / name for name in (
        "draft/c1_higher_genus.tex",
        "Machine note/two_loop/figures/genus2_free_energy.tex",
        "code/general/conventions/genus2_integrand_normalization.py",
        "code/two_loop/integrand/genus2_c1_string_integrand.py",
    )]
    original_hashes = {str(p.relative_to(ROOT)): sha(p) for p in source_files}
    summary = json.loads((SOURCE / "tail_audit_summary.json").read_text())
    assert summary["integration_kernel_convention"] == (
        "c1-bry-real-area-genus2-aut-half-dimensionless-radius-gs2-stripped-v7"
    )
    with np.load(SOURCE / "tail_audit_arrays.npz", allow_pickle=False) as saved:
        arrays = {k: saved[k].copy() for k in saved.files}
    radii = arrays["radii"]
    group_names = sorted(k for k in arrays if k.startswith("group_"))
    assert len(group_names) == 9 and len(radii) == 71
    mean, covariance = reconstruct([arrays[k] for k in group_names])
    np.testing.assert_allclose(mean, arrays["resolved_mean"], rtol=3e-14)
    np.testing.assert_allclose(covariance, arrays["resolved_covariance"], rtol=3e-12, atol=1e-22)
    table = np.genfromtxt(SOURCE / "tail_audit_radius_values.csv", delimiter=",", names=True)
    np.testing.assert_array_equal(radii, table["R"])
    np.testing.assert_allclose(mean, table["resolved_F2_over_gBRY_squared"], rtol=3e-14)
    np.testing.assert_allclose(np.sqrt(np.diag(covariance)), table["empirical_MC_SE"], rtol=3e-13)
    old_beta, old_beta_cov = fit(radii, mean, covariance)
    np.testing.assert_allclose(old_beta, [summary["fit"]["a"], summary["fit"]["b"]], rtol=3e-12)
    np.testing.assert_allclose(old_beta_cov, summary["fit"]["coefficient_covariance"], rtol=3e-11)

    converted_groups = {k: arrays[k] / SCALE for k in group_names}
    converted, converted_cov = reconstruct(list(converted_groups.values()))
    np.testing.assert_allclose(converted, mean / SCALE, rtol=3e-14)
    np.testing.assert_allclose(converted_cov, covariance / SCALE**2, rtol=3e-12, atol=1e-25)
    matrix_model = (7 * radii + 10 / radii + 7 / radii**3) / 5760
    np.testing.assert_allclose(arrays["matrix_model"] / SCALE, matrix_model, rtol=3e-14)
    np.testing.assert_allclose(table["matrix_model"] / SCALE, matrix_model, rtol=3e-14)
    np.testing.assert_allclose(converted / matrix_model, mean / arrays["matrix_model"], rtol=3e-14)
    beta, beta_cov = fit(radii, converted, converted_cov)
    np.testing.assert_allclose(beta, old_beta / SCALE, rtol=3e-12)
    np.testing.assert_allclose(beta_cov, old_beta_cov / SCALE**2, rtol=3e-11)
    one = int(np.flatnonzero(radii == 1)[0])
    bracket = 5760 * beta
    bracket_error = 5760 * np.sqrt(np.diag(beta_cov))
    result = {
        "status": "normalization_conversion_and_frozen_estimator_reconstruction_passed",
        "coupling_convention": "g = mu^-1 = 2*pi*g_s",
        "old_quantity": "resolved coefficient of g_s^2 in connected log Z",
        "new_quantity": "resolved F_2(R), coefficient of g^2 in connected log Z",
        "full_genus_two_term": "g^2 * F_2(R); g remains symbolic",
        "figure_ordinate": "F_2(R), with g^2 stripped off",
        "not_computed": "thermal free energy -log Z/(2*pi*R)",
        "mean_and_SE_divisor": SCALE,
        "covariance_divisor": SCALE**2,
        "matrix_model_F2": "(7*R+10/R+7/R^3)/5760",
        "radius_count": len(radii), "independent_scramble_groups": len(group_names),
        "fit": {
            "ansatz": "a*(R+R^-3)+b/R",
            "method": "inverse-marginal-variance least squares; full covariance propagated",
            "coefficient": beta.tolist(), "coefficient_covariance": beta_cov.tolist(),
            "bracket_coefficients_with_prefactor_1_over_5760": bracket.tolist(),
            "bracket_coefficient_empirical_SE": bracket_error.tolist(),
            "R_and_inverse_R_cubed_coefficients_constrained_equal": True,
        },
        "R1": {"resolved_F2": float(converted[one]),
               "empirical_MC_SE": float(np.sqrt(converted_cov[one, one])),
               "matrix_model_F2": float(matrix_model[one]),
               "worldsheet_over_matrix_model": float(converted[one] / matrix_model[one])},
        "largest_relative_matrix_model_difference": float(np.max(np.abs(converted / matrix_model - 1))),
        "matrix_model_used_to_calibrate": False,
        "new_integration_performed": False,
        "source_full_free_energy_available": summary["full_free_energy_available"],
        "source_certified": summary["certified"],
        "remaining_residual_region": summary["remaining_residual_region"],
        "uncertainty_scope": summary["uncertainty_scope"],
        "error_budget": summary["error_budget"],
        "source_sha256": original_hashes,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "radius_values.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["R", "resolved_F2", "empirical_MC_SE", "matrix_model_F2"])
        writer.writerows(zip(radii, converted, np.sqrt(np.diag(converted_cov)), matrix_model))
    np.savez_compressed(OUTPUT / "arrays.npz", radii=radii, resolved_F2=converted,
                        covariance_F2=converted_cov, matrix_model_F2=matrix_model,
                        fit_coefficient=beta, fit_covariance=beta_cov, **converted_groups)
    (OUTPUT / "normalization_audit.json").write_text(json.dumps(result, indent=2) + "\n")

    template = (ROOT / "Machine note/two_loop/figures/genus2_free_energy.tex").read_text()
    replacements = {
        "../../../data set/two_loop/tail_completion/tail_audit_radius_values.csv":
            "../../data set/two_loop/g_convention_audit_20260911/radius_values.csv",
        "ymin=0.10,": f"ymin={0.10 / SCALE:.16g},",
        "ymax=0.75,": f"ymax={0.75 / SCALE:.16g},",
        "ylabel={$g^2F_2(R)$}": "ylabel={$F_2(R)$}",
        "scaled y ticks=false,": r"""scaled y ticks=false,
  ytick={0.005,0.010,0.015},
  yticklabels={$0.005$,$0.010$,$0.015$},""",
        "pi^2/1440*(7*x+10/x+7/x^3)": "(7*x+10/x+7/x^3)/5760",
        "y=resolved_F2_over_gBRY_squared": "y=resolved_F2",
    }
    for old, new in replacements.items():
        assert template.count(old) == 1, f"Figure template changed: {old}"
        template = template.replace(old, new)
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    FIGURE.write_text("% The ordinate is F_2(R), with g^2 stripped off.\n"
                      "% g = mu^-1 = 2*pi*g_s; original data and errors divide by 4*pi^2.\n" + template)
    for p in source_files:
        assert sha(p) == original_hashes[str(p.relative_to(ROOT))], f"Input changed: {p}"
    print(json.dumps({k: result[k] for k in ["status", "R1", "fit"]}, indent=2))
    print("Derived data:", OUTPUT)
    print("Separate figure source:", FIGURE)


if __name__ == "__main__":
    main()
