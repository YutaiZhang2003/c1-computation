"""Re-export saved amplitudes and thermal F_2 without running a moduli integral."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from c1_paths import data_root
from c1_observables import amplitude_from_reduced, write_genus2_free_energy, write_six_point_amplitudes
from c1_matrix_model import amplitude_coefficient


def read_json(path):
    return json.loads(Path(path).read_text())


def source_identity(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def write_amplitudes(path, points, source):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    result = dict(points=points, source=source_identity(source), new_integration_performed=False,
                  uncertainty_scope='Source sampling errors; no new bound on truncation or other systematics.')
    path.with_suffix('.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    with path.with_suffix('.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['t', 'amplitude_real', 'amplitude_imag', 'SE_real', 'SE_imag',
                         'matrix_model_real', 'matrix_model_imag'])
        for row in points:
            writer.writerow([row['t'], row['value']['real'], row['value']['imag'],
                row['standard_error']['real'], row['standard_error']['imag'],
                row['matrix_model']['real'], row['matrix_model']['imag']])


def add_comparison(point, t, genus, outgoing):
    value = amplitude_coefficient(genus, outgoing)
    return dict(point, t=float(t), matrix_model=dict(real=value.real, imag=value.imag))


def export_tree(output, six_point_source=None):
    source = data_root('tree_level')/'sphere_five_point_1to4/blind30_20260824/worldsheet_extension_12point.json'
    points = []
    for row in read_json(source)['points']:
        estimate = amplitude_from_reduced(complex(**row['raw_integral']), complex(**row['raw_standard_error']),
            factor=2j*math.pi, genus=0, punctures=5, reduced_quantity='I5',
            replicates=[complex(**z) for z in row['replicate_estimates']])
        points.append(add_comparison(estimate, row['t'], 0, [1j*row['t']]*4))
    write_amplitudes(output/'sphere_1to4', points, source)
    source = Path(six_point_source) if six_point_source is not None else (
        data_root('tree_level')/'sphere_six_point_1to5/yin_radial_curve_15_optimized_20260911_v2/curve_results.json')
    report = read_json(source)
    if report.get('complete_execution') is not True:
        raise ValueError('complete audited six-point curve required')
    converted = write_six_point_amplitudes(output/'sphere_1to5', report)
    points = [add_comparison(p, p['t'], 0, [1j*p['t']]*5) for p in converted['points']]
    write_amplitudes(output/'sphere_1to5/comparison', points, source)


def export_one_loop(output):
    source = data_root('one_loop')/'genus1_two_point_worldsheet/imaginary_hybrid_hc_t_scan10_n256_v2/worldsheet_scan_manifest.json'
    points = []
    for row in read_json(source)['points']:
        estimate = amplitude_from_reduced(row['I1'], complex(row['rqmc_standard_error']),
            factor=2j*math.pi**2, genus=1, punctures=2, reduced_quantity='I1')
        points.append(add_comparison(estimate, row['t'], 1, [1j*row['t']]))
    write_amplitudes(output/'torus_1to1', points, source)
    source = data_root('one_loop')/'genus1_three_point_worldsheet/hdominant_extension20_p8_h8l3_q030_007_n256_r4_20260902_v2/assembled/worldsheet_vs_matrix_bry.csv'
    points = []
    with source.open(newline='') as stream:
        for row in csv.DictReader(stream):
            t = float(row['t'])
            estimate = amplitude_from_reduced(-1j*float(row['minus_imag_I_1_3']),
                1j*float(row['rqmc_standard_error']), factor=2j*math.pi**2,
                genus=1, punctures=3, reduced_quantity='imaginary component of I_1_3')
            # Only the displayed real amplitude was saved in this CSV.
            estimate['value']['imag'] = None
            estimate['standard_error']['imag'] = None
            points.append(add_comparison(estimate, t, 1, [1j*t/2]*2))
    write_amplitudes(output/'torus_1to2', points, source)


def export_two_loop(output, source=None):
    source = Path(source) if source is not None else data_root('two_loop')/'tail_completion'
    summary = read_json(source/'tail_audit_summary.json')
    with np.load(source/'tail_audit_arrays.npz', allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    groups = [value for key, value in arrays.items() if key.startswith('group_')]
    if not groups or any(g.ndim != 2 or len(g) < 2 or not np.isfinite(g).all() for g in groups):
        raise ValueError('complete finite independent scramble groups required')
    np.testing.assert_allclose(sum(g.mean(axis=0) for g in groups), arrays['resolved_mean'], rtol=3e-13)
    np.testing.assert_allclose(sum(np.cov(g,rowvar=False,ddof=1)/len(g) for g in groups),
                               arrays['resolved_covariance'], rtol=3e-11, atol=1e-22)
    summary = dict(summary, convention_export_sources=[source_identity(source/name) for name in
                   ('tail_audit_arrays.npz', 'tail_audit_summary.json')])
    return write_genus2_free_energy(output/'genus2', arrays, summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sector', choices=('all','tree_level','one_loop','two_loop'), default='all')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--six-point-source', type=Path, help='Audited curve_results.json to export')
    parser.add_argument('--genus2-source', type=Path, help='Directory containing BRY-v7 tail_audit files')
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error('use a new or empty output directory to preserve existing exports')
    output.mkdir(parents=True, exist_ok=True)
    if args.sector in ('all','tree_level'): export_tree(output, args.six_point_source)
    if args.sector in ('all','one_loop'): export_one_loop(output)
    if args.sector in ('all','two_loop'): export_two_loop(output, args.genus2_source)
    (output/'conventions.json').write_text(json.dumps(dict(
        coupling='mu^-1=2*pi*g_s', amplitude_expansion='sum_h g_s^(n-1+2h) A^(h)_(1->n)',
        free_energy='F=-log Z/(2*pi*R)', matrix_model_free_energy='F_MM=-log Xi/(2*pi*R)',
        genus_two_term='g_s^2 * F2(R)',
        matrix_model_F1='(1+R^-2)*log(mu)/(48*pi)',
        matrix_model_F2='-pi*(7+10/R^2+7/R^4)/2880',
        sector=args.sector, new_integration_performed=False),indent=2)+'\n')
    print(json.dumps(dict(output=str(output), sector=args.sector,
        convention='mu^-1=2*pi*g_s; F=-log Z/(2*pi*R)', new_integration_performed=False), indent=2))


if __name__ == '__main__':
    main()
