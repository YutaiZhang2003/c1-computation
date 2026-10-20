#!/usr/bin/env python3
"""Audit and aggregate the accepted fifteen-energy curve, retaining shared-scramble covariance."""
from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[4]/'code/general/conventions'))

import numpy as np
from c1_observables import write_six_point_amplitudes


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def aggregate(root):
    root=Path(root).resolve();campaign=json.loads((root/'campaign.json').read_text())
    if campaign['version']!='sphere-six-point-yin-radial-production-curve-v1':
        raise ValueError('fifteen-point radial production campaign required')
    rows,replicates=[],[]
    for energy in campaign['energies']:
        directory=root/energy['label'];release=json.loads((directory/'input_release.json').read_text())
        if (energy['operation']=='reuse'
                and campaign.get('continuation', {}).get('completed_results')=='reference-only'):
            # User requested no further verification jobs for completed energies.
            # Retain their already audited values and scramble means for the plot.
            reference=(directory/release['reference_directory']).resolve()
            previous=json.loads((reference/'final_audit.json').read_text())
            replicates.append([value[0] for value in previous['replicate_means']])
            t=Decimal(str(energy['t']));target=float((1-5*t)*(2-5*t)*(3-5*t))
            rows.append(dict(previous,operation='reuse',matrix_model_Q5=target,
                matrix_sampling_only_pull=(previous['mean_real']-target)/previous['scramble_se_real'],
                final_audit_sha256=release['reference_files']['final_audit.json'],
                completed_reference=str(reference),retained_from_previous_audit=True))
            continue
        audit_path=directory/'final_audit.json';audit=json.loads(audit_path.read_text())
        summary_path=directory/'amplitude/main/summary.json';summary=json.loads(summary_path.read_text())
        pointwise=directory/'amplitude/main/pointwise.npz'
        if (audit['t']!=energy['t'] or audit['sample_count']!=2048
                or not audit['finite'] or not audit['exact_complete_source']
                or audit['unresolved_scope_count'] or audit['numerical_failure_count']
                or digest(pointwise)!=audit['pointwise_sha256']
                or digest(summary_path)!=audit['summary_sha256']
                or not summary['exact_original_index_coverage']
                or not summary['all_original_source_geometry_and_density_hashes_match']):
            raise ValueError('incomplete or inconsistent energy output: '+energy['label'])
        for relative,expected in release['files'].items():
            if digest(directory/relative)!=expected:raise ValueError('frozen input changed: '+relative)
        if energy['operation']=='reuse':
            reuse=json.loads((directory/'reuse_receipt.json').read_text())
            if not reuse['passed'] or reuse['original_files']!=release['reference_files']:
                raise ValueError('completed-data reuse was not verified')
            for relative in ('final_audit.json','amplitude/main/summary.json','amplitude/main/pointwise.npz'):
                if digest(directory/relative)!=release['reference_files'][relative]:
                    raise ValueError('reused output changed')
            expected_release=release['reference_files']['input_release.json']
        else:expected_release=digest(directory/'input_release.json')
        if audit['input_release_sha256']!=expected_release:
            raise ValueError('energy audit belongs to a different release')
        with np.load(pointwise,allow_pickle=False) as data:
            baseline=data['variants'].tolist().index('baseline');values=data['hybrid_I6']
            if values.shape!=(1,8,256,8) or not np.isfinite(values).all() or data['energies'].tolist()!=[energy['t']]:
                raise ValueError('wrong pointwise shape, energy or finiteness')
            scale=-1/(40*np.pi**3*energy['t']**6)
            means=values[0,...,baseline].mean(axis=1)*scale
            np.testing.assert_allclose([means.real.mean(),means.real.std(ddof=1)/np.sqrt(8)],
                [audit['mean_real'],audit['scramble_se_real']],rtol=1e-13)
            replicates.append(means.real.tolist())
            prefix=data['treated_prefix_I6'][0,...,baseline,:]*scale
            increments=[float(np.abs(prefix[...,i+1]-prefix[...,i]).mean()/abs(audit['mean_real'])) for i in range(2)] if np.isfinite(prefix).all() else None
        # Prediction is evaluated only after reading and validating finished outputs.
        t=Decimal(str(energy['t']));target=float((1-5*t)*(2-5*t)*(3-5*t))
        row=dict(audit,operation=energy['operation'],matrix_model_Q5=target,
            matrix_sampling_only_pull=(audit['mean_real']-target)/audit['scramble_se_real'],
            weighted_block_prefix_relative_absolute_increments=increments,
            final_audit_sha256=digest(audit_path))
        rows.append(row)
    if len(rows)!=15 or len({row['t'] for row in rows})!=15:
        raise ValueError('incomplete fifteen-energy curve')
    means=np.asarray(replicates)
    covariance=np.cov(means,ddof=1)/8
    np.testing.assert_allclose(np.diag(covariance),np.asarray([row['scramble_se_real']**2 for row in rows]),rtol=1e-12)
    report=dict(version='sphere-six-point-radial-fifteen-energy-results-v1',complete_execution=True,
        results=rows,energies=[row['t'] for row in rows],replicate_means=replicates,
        mean_covariance=covariance.tolist(),empirical_covariance_rank=int(np.linalg.matrix_rank(covariance)),
        covariance_note='Same scramble labels across energies. Eight replicates imply covariance rank at most seven; do not invert it as a nonsingular fifteen-point covariance.',
        all_energies_finite=True,exact_all_source_points=True,total_accuracy_certified=False,
        numerical_failures=0,scope_failures=0)
    target=root/'curve_results.json'
    if target.exists() and json.loads(target.read_text())!=report:
        raise ValueError('existing aggregate differs from current verified outputs')
    if not target.exists():target.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    write_six_point_amplitudes(root, report)
    lines=['# Fifteen-energy radial-mixture amplitude', '',
        'Each result uses eight scrambles of 256 points. Errors below are sampling standard errors; they do not include uncomputed CFT-tail systematics.', '',
        '| t | Q5 ± sampling SE | Relative SE | Matrix prediction | Sampling-only pull |',
        '|---|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['t']:.4f} | {row['mean_real']:.9g} ± {row['scramble_se_real']:.4g} | {100*row['empirical_relative_scramble_se']:.2f}% | {row['matrix_model_Q5']:.9g} | {row['matrix_sampling_only_pull']:+.2f} |")
    lines+=['','All numerical and bank-scope execution checks passed. Momentum-control and block-prefix diagnostics, reused-result provenance, and cross-energy covariance are retained in curve_results.json. Total one-percent accuracy is not certified.','']
    (root/'curve_results.md').write_text('\n'.join(lines))
    return dict(complete_execution=True,energies=len(rows),output=str(target),covariance_rank=report['empirical_covariance_rank'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    args=parser.parse_args();print(json.dumps(aggregate(args.root),indent=2))
