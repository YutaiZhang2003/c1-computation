#!/usr/bin/env python3
"""Absolute BRY-v7 plot and correlated-error fit from the frozen v6 assembly.

The numerical result is a resolved contribution, NOT a certified complete
free energy. No failed-node imputation or matrix-model normalization is used.
The frozen adapter verifies its provenance and independently reconstructs
the mean/covariance from the saved Xi-v5 scramble groups before conversion.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from genus2_integrand_normalization import (
    CURRENT_INTEGRATION_KERNEL_CONVENTION as CURRENT,
    LEGACY_V5_XI_INTEGRATION_KERNEL_CONVENTION as V5,
    LEGACY_V6_BRY_INTEGRATION_KERNEL_CONVENTION as V6,
    bry_integration_kernel_target,
    c1_bry_genus2_kernel_multiplier,
    genus2_automorphism_metadata,
    integration_kernel_scale_to_current,
)

from c1_paths import PACKAGE_ROOT as ROOT, data_root, module_path
DEFAULT_SOURCE = data_root('two_loop') / 'assembly_v6'
DEFAULT_OUTPUT = data_root('two_loop') / 'reproduced_absolute_bry_v7'
GROUPS = ('old_bulk_plus_outer_residual', 'integrated_control',
          'negative_domain_correction', 'residual_band_0', 'residual_band_1',
          'residual_band_3', 'residual_band_4')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def design_matrix(radii):
    r = np.asarray(radii, dtype=float)
    if r.ndim != 1 or not np.all(np.isfinite(r)) or np.any(r <= 0):
        raise ValueError('radii must be a finite positive vector')
    return np.column_stack((r + r**-3, r**-1))


def fit_radius_ansatz(radii, values, covariance):
    """Fixed diagonal-weight LS, propagating the FULL cross-radius covariance.

    The empirical covariance is low rank (many radii reuse few scrambles).
    We do not invert it or report an independent-point chi-square. For the
    fixed linear estimator beta=L y, Cov(beta)=L Cov(y) L^T, including all
    off-diagonal correlations. Uncertainty of the estimated weights is not
    separately modelled. No matrix-model coefficient enters this fit.
    """
    x = design_matrix(radii)
    y, c = np.asarray(values, float), np.asarray(covariance, float)
    n = len(x)
    if n < 3 or y.shape != (n,) or c.shape != (n, n):
        raise ValueError('fit requires >=3 radii and matching values/covariance')
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(c)):
        raise ValueError('nonfinite fit input')
    if not np.allclose(c, c.T, rtol=1e-12, atol=1e-20) or np.any(np.diag(c) <= 0):
        raise ValueError('covariance must be symmetric with positive diagonal')
    eigenvalues = np.linalg.eigvalsh(c)
    if eigenvalues[0] < -1e-10 * eigenvalues[-1]:
        raise ValueError('covariance is not positive semidefinite')
    if np.linalg.matrix_rank(x) != 2:
        raise ValueError('radius grid does not identify both coefficients')
    weights = 1 / np.diag(c)
    linear_map = np.linalg.solve(x.T @ (weights[:, None]*x), x.T*weights)
    beta = linear_map @ y
    beta_covariance = linear_map @ c @ linear_map.T
    prediction = x @ beta
    return dict(coefficients=beta, covariance=beta_covariance,
                standard_error=np.sqrt(np.maximum(np.diag(beta_covariance), 0)),
                prediction=prediction, linear_map=linear_map,
                empirical_covariance_rank=int(np.linalg.matrix_rank(c)),
                relative_residual=prediction/y-1)


def load_frozen_assembly(source):
    source = Path(source).resolve()
    summary_path, array_path = source/'summary.json', source/'assembled_arrays.npz'
    summary = json.loads(summary_path.read_text())
    norm = summary['normalization']
    # This is an explicit adapter for an older, untagged BRY-v6 NPZ schema.
    if not (norm['worldsheet_coefficient_bare_J2'] == '16*pi^4'
            and norm['old_to_bry'] == .25 and norm['alpha_prime'] == 1.
            and norm['normalized_at_R1'] is False and norm['fitted_amplitude'] is False):
        raise ValueError('source is not the declared absolute pre-Aut BRY-v6 assembly')
    if summary['full_free_energy_available'] or summary['resolved_count'] != 39:
        raise ValueError('changed residual assembly requires an explicit adapter review')
    hashes = {str(summary_path): sha(summary_path), str(array_path): sha(array_path)}
    for name, expected in summary['input_sha256'].items():
        path = (source/name).resolve()
        if not path.is_relative_to(source) or sha(path) != expected:
            raise ValueError(f'provenance mismatch: {name}')
        hashes[str(path)] = expected
    with np.load(array_path, allow_pickle=False) as saved:
        a = {k: np.array(saved[k], copy=True) for k in saved.files}
    radii = a['radii']
    design_matrix(radii)
    if np.any(np.diff(radii) <= 0):
        raise ValueError('source radius grid must be strictly increasing')
    xi_mean = sum(a[k].mean(axis=0) for k in GROUPS)
    xi_cov = sum(np.cov(a[k], rowvar=False, ddof=1)/len(a[k]) for k in GROUPS)
    np.testing.assert_allclose(a['resolved_worldsheet_bry'], xi_mean/4, rtol=2e-14)
    np.testing.assert_allclose(a['covariance_bry'], xi_cov/16, rtol=2e-13, atol=1e-20)
    mm = np.array([bry_integration_kernel_target(float(r)) for r in radii])
    np.testing.assert_allclose(a['matrix_model_bry'], mm, rtol=2e-14)
    scale = integration_kernel_scale_to_current(V6)
    current_groups = {k: a[k]*integration_kernel_scale_to_current(V5) for k in GROUPS}
    values, covariance = a['resolved_worldsheet_bry']*scale, a['covariance_bry']*scale**2
    np.testing.assert_allclose(values, sum(v.mean(axis=0) for v in current_groups.values()), rtol=2e-14)
    return dict(radii=radii, values=values, covariance=covariance, matrix_model=mm,
                previous_values=a['previous_pilot_bry']*scale,
                previous_covariance=a['previous_covariance_bry']*scale**2,
                groups=current_groups, source_summary=summary, input_sha256=hashes)


def fit_record(fit, radii):
    return dict(ansatz='a*R+b/R+a/R^3', radius_min=float(min(radii)),
                radius_max=float(max(radii)), radius_count=len(radii),
                a=float(fit['coefficients'][0]), b=float(fit['coefficients'][1]),
                empirical_standard_errors=fit['standard_error'].tolist(),
                coefficient_covariance=fit['covariance'].tolist(),
                empirical_covariance_rank=fit['empirical_covariance_rank'],
                maximum_absolute_relative_residual=float(max(abs(fit['relative_residual']))),
                method='fixed inverse-diagonal-variance weighted LS; full covariance propagated',
                chi_squared=None, independent_radius_errors_assumed=False)


def produce(source, output):
    data = load_frozen_assembly(source)
    r, y, c, mm = (data[k] for k in ('radii', 'values', 'covariance', 'matrix_model'))
    fit = fit_radius_ansatz(r, y, c)
    se = np.sqrt(np.diag(c))
    output = Path(output).resolve()
    # Preserve previous releases and frozen sources: never overwrite a directory.
    output.mkdir(parents=True, exist_ok=False)
    source_summary = data['source_summary']
    result = dict(
        status='absolute_BRY_v7_resolved_contribution_not_complete_free_energy',
        full_free_energy_available=False, certified=False,
        created_utc=datetime.now(timezone.utc).isoformat(),
        integration_kernel_convention=CURRENT, **genus2_automorphism_metadata(),
        kernel_multiplier=c1_bry_genus2_kernel_multiplier(),
        kernel_formula='(1/2)_Aut*16*pi^4*J2=8*pi^4*J2',
        coupling='mu^-1=2*pi*g_s^BRY', alpha_prime=1., radius_definition='R_phys/sqrt(alpha_prime)',
        source_assembly=str(Path(source).resolve()), source_kernel_convention=V6,
        source_to_current_scale=integration_kernel_scale_to_current(V6),
        source_group_kernel_convention=V5,
        input_sha256=data['input_sha256'],
        code_sha256={str(Path(__file__).resolve()):sha(__file__),
                     str(module_path('genus2_integrand_normalization')):sha(module_path('genus2_integrand_normalization'))},
        parent_bundle_sha256=source_summary['parent_bundle_sha256'],
        retry_bundle_sha256=source_summary['retry_bundle_sha256'],
        estimator=source_summary['estimator'], band_status=source_summary['band_status'],
        resolved_count=source_summary['resolved_count'],
        equation='F2/g_s^2 = plotted resolved contribution + Delta_[0.005,0.01) + Delta_[0,0.001)',
        unknown_terms='Both Delta terms integrate K-G in BRY-v7 units; neither set to zero nor bounded.',
        uncertainty_scope=source_summary['uncertainty_scope'],
        matrix_model_formula='pi^2/1440*(7*R+10/R+7/R^3)',
        matrix_model_coefficients=(np.pi**2/1440*np.array([7,10])).tolist(),
        matrix_model_rescaled=False, normalized_at_R1=False,
        fit=fit_record(fit, r), selected_radii=[])
    for label, low, high in [('half_to_two', .5, 2.), ('one_to_fifteen', 1., 15.)]:
        mask = (r >= low) & (r <= high)
        subfit = fit_radius_ansatz(r[mask], y[mask], c[np.ix_(mask, mask)])
        result.setdefault('fit_range_sensitivity', {})[label] = fit_record(subfit, r[mask])
    for value in (.5, 1., 2., 5., 10., 15.):
        i = int(np.argmin(abs(r-value)))
        if not np.isclose(r[i], value):
            raise ValueError('requested reporting radius absent')
        result['selected_radii'].append(dict(radius=float(r[i]),
            resolved_worldsheet_bry=float(y[i]), empirical_SE=float(se[i]),
            matrix_model_bry=float(mm[i]), ratio=float(y[i]/mm[i]),
            fitted_value=float(fit['prediction'][i])))
    result['maximum_absolute_matrix_model_relative_difference'] = float(max(abs(y/mm-1)))
    np.savez_compressed(output/'assembled_arrays.npz', radii=r,
        resolved_worldsheet_bry=y, covariance_bry=c, matrix_model_bry=mm,
        fitted_worldsheet=fit['prediction'], fit_coefficients=fit['coefficients'],
        fit_coefficient_covariance=fit['covariance'],
        previous_pilot_bry=data['previous_values'], previous_covariance_bry=data['previous_covariance'],
        integration_kernel_convention=CURRENT, **data['groups'])
    with (output/'radius_values.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['radius', 'resolved_F2_over_gs2', 'empirical_SE',
                         'matrix_model_F2_over_gs2', 'fitted_F2_over_gs2',
                         'relative_to_matrix_model', 'integration_kernel_convention'])
        writer.writerows(zip(r,y,se,mm,fit['prediction'],y/mm,[CURRENT]*len(r)))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':11,
                         'axes.spines.top':False, 'axes.spines.right':False})
    fig, (ax, ratio) = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True,
                                  gridspec_kw={'height_ratios':[3,1.4], 'hspace':.11})
    dense = np.geomspace(r.min(), r.max(), 700)
    dense_mm = np.array([bry_integration_kernel_target(float(x)) for x in dense])
    dense_fit = design_matrix(dense) @ fit['coefficients']
    ax.plot(dense, dense_mm, color='#292929', lw=1.7, label='Matrix model (absolute prediction)', zorder=2)
    ax.plot(dense, dense_fit, color='#d87913', lw=1.6, ls='--', label=r'Fit: $aR+b/R+a/R^3$', zorder=3)
    ax.errorbar(r, y, yerr=se, fmt='o', ms=3.2, color='#1768ac', elinewidth=.8,
                capsize=0, label='Worldsheet: resolved contribution', zorder=4)
    ax.set_ylabel(r'$F_2(R)/(g_s^{\rm BRY})^2$')
    ax.set_title(r'$c=1$ string: absolute genus-two free energy', loc='left', fontsize=16, pad=22)
    ax.text(0, 1.015, r'BRY convention · generic automorphism factor $1/2$ included · $\alpha^\prime=1$',
            transform=ax.transAxes, fontsize=10, color='#555555')
    ax.legend(loc='upper center', frameon=False, fontsize=10)
    ax.grid(alpha=.15)
    ax.set_ylim(bottom=0)
    ratio.axhline(0, color='#292929', lw=1)
    ratio.fill_between(r, 100*(y/mm-1-se/mm), 100*(y/mm-1+se/mm), color='#1768ac', alpha=.13)
    ratio.errorbar(r, 100*(y/mm-1), yerr=100*se/mm, fmt='o', ms=3, color='#1768ac', elinewidth=.65)
    ratio.plot(dense, 100*(dense_fit/dense_mm-1), color='#d87913', ls='--', lw=1.6)
    ratio.set_ylabel('Difference from\nmatrix model (%)')
    ratio.set_xlabel(r'Radius $R=R_{\rm phys}/\sqrt{\alpha^\prime}$')
    ratio.set_ylim(-1., 1.)
    ratio.set_xscale('log')
    ratio.set_xticks([.5,1,2,3,5,10,15])
    ratio.xaxis.set_major_formatter(ScalarFormatter())
    ratio.minorticks_off()
    ratio.grid(alpha=.15)
    fig.subplots_adjust(left=.105, right=.975, top=.87, bottom=.18)
    fig.text(.105, .075, 'No normalization fitted to the matrix model; no division by the R=1 value.', fontsize=9.5)
    fig.text(.105, .038, 'Provisional: two residual-tail regions remain unbounded. Error bars: empirical RQMC uncertainty only.',
             fontsize=9.2, color='#8b3f24')
    for suffix in ('png','svg'):
        fig.savefig(output/f'absolute_free_energy_bry.{suffix}', dpi=180)
    plt.close(fig)

    for path, expected in data['input_sha256'].items():
        if sha(path) != expected:
            raise RuntimeError(f'frozen source changed while plotting: {path}')
    (output/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    a,b = fit['coefficients']; sa,sb=fit['standard_error']
    ma,mb = result['matrix_model_coefficients']
    one = next(v for v in result['selected_radii'] if v['radius']==1.)
    report = f'''# Absolute genus-two free energy in BRY conventions

The generic genus-two automorphism factor **1/2 is included once** in the
amplitude kernel: `K=(1/2)*16*pi^4*J2=8*pi^4*J2` at alpha'=1.
Proposal weights, CFT data, radius, and the physical matrix-model prediction
are unchanged. This is an absolute plot, not normalized at R=1.

## Numerical comparison and fit

For the resolved contribution at R=1, `F2/gs^2={one['resolved_worldsheet_bry']:.10f}
+/- {one['empirical_SE']:.10f}` (empirical RQMC SE). The matrix model gives
`{one['matrix_model_bry']:.10f}`; the central difference is
`{100*(one['ratio']-1):+.6f}%`.

Fitting all {len(r)} radii from {r.min():g} to {r.max():g} to
`a*(R+R^-3)+b/R` gives:

| Coefficient | Worldsheet fit | Empirical SE | Matrix model |
| --- | ---: | ---: | ---: |
| a | {a:.10f} | {sa:.10f} | {ma:.10f} |
| b | {b:.10f} | {sb:.10f} | {mb:.10f} |

The fit has two free coefficients; the R and R^-3 coefficients are constrained
equal. Neither coefficient is fixed to the matrix model. Central values use
inverse marginal-variance weights. The **full cross-radius covariance** is
propagated as `Cov(beta)=L Cov(F) L^T`. Its numerical rank is
{fit['empirical_covariance_rank']} for {len(r)} radii; no full-covariance inversion,
independent-radius chi-square, or chi-square confidence claim is made.
The uncertainty in estimated weights is not separately modelled.
The maximum fit/data relative residual is
{100*result['fit']['maximum_absolute_relative_residual']:.5f}%; the largest
central worldsheet/matrix-model difference is
{100*result['maximum_absolute_matrix_model_relative_difference']:.5f}%.
Range sensitivity and coefficient covariance are in `summary.json`.

## Remaining numerical limitation

This is **not yet a complete or certified free energy**. The inherited assembly
has 39/40 successful nodes, including the successful index-3 retry. The entire
incomplete residual band `0.005 <= |Omega12| < 0.01` remains unknown; its seven
successful samples are not averaged alone. The deep residual
`|Omega12| < 0.001` is also unbounded. The integrated primary control covers
these regions, but the missing terms are integrals of `K-G`, not zero:

    F2/gs^2 = plotted resolved contribution + Delta_[0.005,0.01) + Delta_[0,0.001).

Error bars and fit errors describe empirical scramble variation only. They
exclude CFT/control truncation, unresolved residuals and normalization
systematics. Agreement does not turn these unknowns into error bounds.

## Reuse and reproducibility

Source: `{Path(source).resolve()}`. All source result hashes were checked
before and after plotting. The frozen Xi-v5 scramble groups independently
reproduce the saved BRY-v6 mean and covariance. V6 values/SE divide by 2;
covariance divides by 4. All groups are converted together, including G and
the domain correction. The matrix model is checked against the analytic
BRY target and is not scaled. No CFT was recomputed and no frozen file changed.

Regenerate to a **new** output directory:

    python run.py --work-dir /tmp/c1-reproduction plot_genus2_absolute_bry --output /path/to/new/directory

See [the convention note](../../../plumbing/c1_bry_conventions.md).
'''
    (output/'REPORT.md').write_text(report)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args=parser.parse_args()
    result=produce(args.source,args.output)
    print(json.dumps(dict(output=str(args.output.resolve()), fit=result['fit'],
                          selected_radii=result['selected_radii']), indent=2))


if __name__ == '__main__':
    main()
