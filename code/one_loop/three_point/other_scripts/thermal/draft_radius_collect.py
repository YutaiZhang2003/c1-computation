"""Assemble the draft baseline plus independently integrated thermal corrections."""
import genus1_source_paths as bootstrap
bootstrap.install()
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import numpy as np

from draft_radius_campaign import OUTPUT, BASELINE, tag_of, read, TAU
from genus1_native_bank import digest
from genus1_three_point_momentum import fingerprint
from kernels import ThermalKinematics, matrix_prediction
from kernels import lattice
from study_pair_sampling import summarize


def compact(row, source):
    kin=ThermalKinematics(row['t'],row['R'])
    if kin.R>8:raise ValueError('only R<=8 corrections are requested')
    saved=next(r for r in source['rows'] if abs(r['t']-kin.t)<1e-14)
    groups=defaultdict(list);pins={}
    for item in saved['points']:
        wrapper=item['record'];r=wrapper['payload']
        if fingerprint(r)!=wrapper['sha256']:raise ValueError('changed compact source record')
        g=r['location']['geometry'];old=r['value']['value'][0]*TAU**3
        factor=lattice(kin,complex(*g['tau']),(complex(*g['w1']),complex(*g['w2']),0j))
        groups[r['location']['replicate']].append([old,old*factor,old*(factor-1)])
        pins[item['path']]=item['sha256']
    if len(groups)!=16 or any(len(v)!=4 for v in groups.values()):raise ValueError('incomplete compact source')
    result=summarize([np.mean(groups[k],axis=0) for k in sorted(groups)])
    result.update(source_folder=saved['source'],source_sha256=pins)
    return result


def component(tag,name,power):
    reps=[];halves=[];pins={};seconds=0;arithmetic=[];flags=0
    folder=OUTPUT/'sampling'/tag/name
    for replicate in range(16):
        path=folder/f'p{power}_r{replicate:02d}.npz'
        with np.load(path) as a:
            v=a['values'];assert v.shape==(2**power,3)
            if not np.all(np.isfinite(v)):raise ValueError('nonfinite numerical samples')
            bounds=a['arithmetic_bounds'];assert bounds.shape==v.shape
            if np.any(bounds<0) or not np.all(np.isfinite(bounds)):
                raise ValueError('invalid weighted arithmetic bounds')
            arithmetic.append(bounds.mean(axis=0));flags+=int(a['arithmetic_flags'])
            reps.append(v.mean(axis=0));halves.append(v[:len(v)//2].mean(axis=0));seconds+=float(a['seconds'])
        pins[str(path.relative_to(OUTPUT))]=digest(path)
    return dict(power=power,scrambles=16,statistics=summarize(reps),
        half_statistics=summarize(halves),paired_doubling_shift=summarize(np.asarray(reps)-halves),
        sample_sha256=pins,total_worker_seconds=seconds,
        sampled_absolute_arithmetic_bound=np.mean(arithmetic,axis=0).tolist(),
        bounded_small_density_requests=flags)


def main():
    import argparse
    argparse.ArgumentParser(description=__doc__).parse_args()
    baseline=read(OUTPUT/'baseline.json')
    if digest(BASELINE)!=baseline['sha256']:raise ValueError('draft baseline changed during integration')
    source=json.loads((OUTPUT/'compact_source.json').read_text());rows=[]
    for b in baseline['selected']:
        if not b['plotted']:continue
        tag=tag_of(b['t']);kin=ThermalKinematics(b['t'],b['R'])
        components={name:component(tag,name,power) for name,power in
                    [('primary',8),('pair-bulk',10),('pair-tail',10)]}
        small=compact(b,source);collision=read(OUTPUT/'collision'/f'{tag}.json')
        delta=float(collision['high'][2]+small['mean'][2]+
                    sum(v['statistics']['mean'][2] for v in components.values()))
        cov=np.array(small['covariance_of_mean'])
        for c in components.values():cov+=c['statistics']['covariance_of_mean']
        arithmetic=collision['integrated_arithmetic_bound']+sum(
            v['sampled_absolute_arithmetic_bound'][2] for v in components.values())
        if arithmetic>1e-7*max(abs(b['baseline_gs3']+delta),1e-6):
            raise FloatingPointError(f'unresolved integrated collision arithmetic at {tag}: {arithmetic}')
        rows.append(dict(**b,tag=tag,thermal_correction=delta,
            worldsheet_thermal_gs3=b['baseline_gs3']+delta,
            sampling_SE_correction=float(math.sqrt(cov[2,2])),
            components=components,compact=small,collision=collision,
            matrix_model=matrix_prediction(kin),full_amplitude_certified=False))
    if len(rows)!=15:raise ValueError('the final panel requires 15 distinct radii')
    result=dict(rows=rows,assembly_complete=True,point_count=15,R_min=40/19,R_max=8,
        baseline_sha256=digest(OUTPUT/'baseline.json'),compact_source_sha256=digest(OUTPUT/'compact_source.json'),
        formula='draft_zero_temperature_gs3 + independently_integrated_thermal_minus_noncompact',
        matrix_model_used_for_numerical_values=False,normalization_fitted=False,
        all_original_baseline_rows_preserved=True,duplicates='later refinement selected at t=0.25 and t=0.95',
        maximum_nodes_per_axis=12,full_amplitude_certified=False,
        errors_are_sampling_only=True,error_bars=False,
        limitations=['Integrated momentum-placement and block-truncation checks are separate from sampling statistics.',
            'Complete collision-forest and finite-tube remainders and global cutoff stability remain unresolved.'])
    # Cross-radius sampling covariance preserves the shared scramble streams.
    covariance=np.zeros((15,15))
    for name in components:
        r=np.array([x['components'][name]['statistics']['replicates'] for x in rows])[:,:,2]
        covariance+=np.cov(r,ddof=1)/16
    r=np.array([x['compact']['replicates'] for x in rows])[:,:,2]
    covariance+=np.cov(r,ddof=1)/16
    result['sampling_covariance_correction']=covariance.tolist()
    (OUTPUT/'thermal_draft_results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    with (OUTPUT/'thermal_draft_results.csv').open('w',newline='') as stream:
        w=csv.writer(stream);w.writerow(['R','t','draft_baseline_gs3','thermal_correction',
            'worldsheet_thermal_gs3','sampling_SE_correction','matrix_model_thermal','baseline_source_row'])
        for r in rows:w.writerow([r['R'],r['t'],r['baseline_gs3'],r['thermal_correction'],
            r['worldsheet_thermal_gs3'],r['sampling_SE_correction'],r['matrix_model']['thermal'],r['source_row']])
    pins={name:digest(OUTPUT/name) for name in ('thermal_draft_results.json','thermal_draft_results.csv',
        'baseline.json','reference_plan.json','compact_source.json')}
    (OUTPUT/'delivery_manifest.json').write_text(json.dumps(dict(files=pins,assembly_complete=True,
        full_amplitude_certified=False),indent=2)+'\n')
    print(json.dumps(dict(points=15,R_min=40/19,R_max=8,assembly_complete=True)))


if __name__=='__main__':main()
