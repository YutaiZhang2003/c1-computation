"""Production-style MQM curve and numerical points with measured RMS bars."""

from genus1_source_paths import ROOT as _SOURCE_ROOT, source_path as _source_path
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = _SOURCE_ROOT / 'output/diagnostics/genus1_rms_errorbar_plot_20260925'
STUDY = ROOT.parent / 'genus1_moduli_rms_curve20_384_20260925'
EXPERIMENT = 'ffb308d7cde5d580af6cd610eb67d491a707e28740b894006050375117be8074'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def matrix(t):
    return t**3 * (t-1) * (t-2) * (1+t-t*t/2) / 96


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    delivery = json.loads((STUDY / 'delivery_manifest.json').read_text())
    data_path = STUDY / 'plot_data.json'
    assert sha(data_path) == delivery['local_files']['plot_data.json']
    data = json.loads(data_path.read_text())
    assert data['experiment_sha256'] == EXPERIMENT
    rows = list(data['rows'])
    assert [r['t'] for r in rows] == [.025] + [i/20 for i in range(1, 20)]
    # All numerical evaluations use one visual style. The two additional
    # values retain their own provenance and central values. Their changed
    # cells lie outside the RMS study, so absolute targeted shifts are the
    # same at that energy; no relative error is transferred between energies.
    translation_checks = []
    additional_sources = [
        ROOT.parent / 'genus1_upper_pair_t025_refinement_384_20260924/amplitude.json',
        ROOT.parent / 'genus1_primary_length_angle_t025_t095_384_20260924/energies/t0950/amplitude.json',
    ]
    for path, point in zip(additional_sources, data['later_refinements']):
        assert sha(path) == data['source_file_sha256'][str(path)]
        amplitude = json.loads(path.read_text())['payload']
        energy = point['t']
        assert amplitude['t'] == energy and amplitude['amplitude'] == point['amplitude']
        lab = f't{round(1000*energy):04d}'
        result_path = STUDY / 'energies' / lab / 'result.json'
        assert sha(result_path) == data['source_file_sha256'][str(result_path)]
        result = json.loads(result_path.read_text())['payload']
        if energy == .25:
            assert result['original_study_reference'] == amplitude['amplitude']
            assert result['unchanged_targeted_cell_integrals_verified']
            reason = 'Original RMS experiment used this central value; exact absolute shifts were preserved.'
        else:
            assert energy == .95 and amplitude['unchanged_other_cells']
            assert amplitude['previous_amplitude'] == result['reference_amplitude']
            assert not amplitude['momentum_design_changed'] and not amplitude['physical_cutoffs_changed']
            tested = {cell['cell']['id'] for cell in result['cells']}
            changed = set(amplitude['replaced_cell_ids'])
            assert tested.isdisjoint(changed)
            reason = 'All 12 replaced cells are disjoint from the 26 targeted RMS cells; other cells are exact imports.'
        rms_value = result['statistics']['rms_shift']
        rows.append(dict(t=energy, amplitude=point['amplitude'], rms=rms_value,
                         matrix_model=matrix(energy)))
        translation_checks.append(dict(t=energy, amplitude=point['amplitude'], rms=rms_value,
                                       source_file=str(path), reason=reason))
    rows.sort(key=lambda r: r['t'])
    t = np.array([r['t'] for r in rows])
    amplitudes = np.array([r['amplitude'] for r in rows])
    rms = np.array([r['rms'] for r in rows])
    theory = matrix(t)
    assert np.all(rms > 0)
    assert np.allclose(theory, [r['matrix_model'] for r in rows], rtol=2e-14, atol=0)
    scale = (2*np.pi)**3
    red = '#BF0000'
    plt.rcParams.update({
        'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix',
        'font.size': 12, 'axes.linewidth': .7,
        'axes.edgecolor': '#444444', 'savefig.facecolor': 'white',
    })
    fig, (ax, residual) = plt.subplots(2, 1, figsize=(8.5, 7.2), sharex=True,
        gridspec_kw={'height_ratios': [2.1, 1], 'hspace': .11})
    fig.subplots_adjust(left=.12, right=.975, top=.965, bottom=.14)
    fine = np.linspace(0, 1, 1001)
    ax.plot(fine, scale*matrix(fine), color='black', lw=1.5,
            label='matrix model', zorder=2)
    ax.errorbar(t, scale*amplitudes, yerr=scale*rms, fmt='o', linestyle='none',
        color=red, mfc=red, mec=red, ms=2.8, mew=.7, elinewidth=1.0,
        capsize=4, capthick=1.0, label=r'numerical data ($\pm$ RMS)', zorder=4)
    ax.set_ylabel(r'$\operatorname{Re}\!\left[\mathcal{A}^{(1)}_{1\to2}(it)\right]$', fontsize=17)
    ax.set_ylim(-.02, .55)
    ax.set_yticks([0, .1, .2, .3, .4, .5])
    ax.legend(frameon=False, loc='upper left', fontsize=13, handlelength=2.2)

    # The same absolute uncertainties in a relative view; never magnify the
    # uncertainties in the amplitude panel or connect numerical points.
    residual.axhline(0, color='black', lw=1.2, zorder=2)
    residual.errorbar(t, 100*(amplitudes/theory-1), yerr=100*rms/theory,
        fmt='o', linestyle='none', color=red, mfc=red, mec=red, ms=3.3,
        mew=.7, elinewidth=1.0, capsize=4, capthick=1.0, zorder=4)
    residual.set_ylabel('Difference from\nmatrix model (%)', fontsize=12)
    residual.set_xlabel(r'$t$', fontsize=17)
    residual.set_ylim(-1.5, 1.5)
    residual.set_yticks([-1, -.5, 0, .5, 1])
    for panel in [ax, residual]:
        panel.set_xlim(0, 1)
        panel.set_xticks(np.linspace(0, 1, 6))
        panel.grid(color='#DEDEDE', lw=.6)
        panel.set_axisbelow(True)
        panel.tick_params(direction='in', top=True, right=True, width=.65)
    fig.text(.12, .042,
        'Bars: each point\'s measured targeted-moduli RMS (0.46%–0.63%).', fontsize=10)
    fig.text(.12, .017,
        'The bars describe grid sensitivity, not total confidence intervals.', fontsize=10)
    for extension in ['png', 'svg']:
        fig.savefig(ROOT / ('amplitude_with_rms.' + extension), dpi=240)
    plt.close(fig)
    provenance = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        experiment_sha256=EXPERIMENT, source_plot_data_sha256=sha(data_path),
        normalization_factor=float(scale), normalization='draft coefficient of g_s^3',
        continuous_curve='fixed-normalization matrix model only',
        points='all 22 saved numerical evaluations at 20 energies; identical red circles and error bars',
        error_bars='each energy-specific measured RMS, unscaled except physical normalization',
        residual='100*(A_numerical/A_MQM - 1); error 100*RMS/A_MQM',
        additional_point_rms_checks=translation_checks,
        amplitude_error_data=[dict(t=float(x), amplitude=float(y), rms=float(e))
                              for x, y, e in zip(t, amplitudes, rms)],
        output_sha256={n:sha(ROOT/n) for n in ['amplitude_with_rms.png', 'amplitude_with_rms.svg']},
    )
    figure_data = ROOT.parents[2] / 'Machine note/one_loop/figures/torus_1to2_numeric_rms.dat'
    figure_data.write_text('t amplitude grid_rms\n' + ''.join(
        f'{x:.17g} {y:.17g} {e:.17g}\n' for x, y, e in zip(t, amplitudes, rms)))
    provenance['native_data_sha256'] = sha(figure_data)
    (ROOT / 'plot_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps(dict(points=len(t), bars=len(rms), rms_values_unchanged=True)))


if __name__ == '__main__':
    main()
