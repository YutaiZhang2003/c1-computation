#!/usr/bin/env python3
"""Prepare a production curve with the accepted radial mixture and selected CFT banks."""
from __future__ import annotations

import argparse
import csv
import datetime
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import runpy
import shutil

ROOT=Path(__file__).resolve().parents[4]
BASE=ROOT/'data set/tree_level/sphere_six_point_1to5'
PRODUCTION=Path(__file__).resolve().parent
# Approximately uniform coverage including both endpoints and all three accepted energies.
PLOT_INDICES=(0,2,4,6,8,10,12,14,17,19,21,23,25,27,29)
REUSED=(.1805,.2505,.3155)
SEED=2026091000


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def copy_observable_export_sources(output):
    """Package the final normalization helpers with the standalone aggregator."""
    for name in ('c1_observables.py','c1_string_conventions.py'):
        shutil.copy2(ROOT/'code/general/conventions'/name,Path(output)/name)


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError('preserve prepared input: '+str(path))
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def physics_settings(config):
    variable={'energies','families','geometry_plans','sample_design','base_seed','replicates',
              'sobol_power','radial_power','radial_proposal','radial_power_prescription','input_manifest'}
    return {key:value for key,value in config.items() if key not in variable}


def copy_source(reference,directory):
    from tree_level_source_compatibility import source_matches_reference
    source=reference/'source'
    for path in source.rglob('*'):
        if not path.is_file():continue
        relative=path.relative_to(source)
        if relative.as_posix()!='code/tree_level/production/sphere_six_point_yin_curve_worker_v2.py':
            if not source_matches_reference(ROOT,relative,path):
                raise ValueError('source differs from accepted computation: '+str(relative))
    shutil.copytree(source,directory/'source')
    shutil.copy2(PRODUCTION/'six_point_run_frozen_energy_worker.py',
                 directory/'source/code/tree_level/production/sphere_six_point_yin_curve_worker_v2.py')
    shutil.copy2(PRODUCTION/'six_point_run_frozen_radial_worker.py',directory/'jobs_worker.py')


def family_record(profile,variants):
    return dict(name=profile['name'],t=profile['t'],topology=profile['topology'],
        incoming_slot=profile['incoming_slot'],anchor_decays=profile['anchor_decays'],
        variants={v:f"banks/{profile['name']}/{v}/bank.npz" for v in variants})


def selection(directory,config):
    import numpy as np
    from six_point_sample_integration_geometry import SphereGeometryPlan
    from six_point_compare_momentum_quadratures import NamedFamily,_select_families
    from six_point_estimate_collar_corrected_integrand import mark_plans
    plans=[SphereGeometryPlan.load(directory/row['path']) for row in config['geometry_plans']]
    marks=mark_plans(plans,config['collar_routing'])
    families=[NamedFamily(row['name'],row['t'],row['topology'],row['incoming_slot'],
                         tuple(row['anchor_decays']),{},{}) for row in config['families']]
    chosen,ratios,qualified,*_=_select_families(plans,tuple(config['energies']),families,
        config['maximum_per_axis_decay_ratio'],excluded=marks['inside'])
    outside=~marks['inside']
    names=np.full(outside.shape,'',dtype=object)
    names[outside]=np.asarray([families[i].name for i in chosen[0][outside]])
    if not qualified[0][outside].all():raise ValueError('uncovered original point remains')
    return names,plans,marks


def prepare_energy(directory,t,prototype,profile_templates,bank_reference):
    import numpy as np
    from six_point_prepare_radial_geometry import prepare as prepare_geometry
    from six_point_integrate_mixed_order_amplitudes import audit_geometry
    from six_point_build_momentum_banks import prepare_bank_build,_provider
    from six_point_mixed_order_node_catalog import prepare_catalog
    from six_point_momentum_control_proposals import proposal
    prepared=prepare_geometry(directory,t=t,base_seed=SEED)
    config=json.loads(json.dumps(prototype))
    config.update(energies=[t],base_seed=SEED,replicates=8,sobol_power=8,
        sample_design=prepared['sample_design'],radial_power=2*t*t,
        radial_proposal=prepared['radial_proposal'],
        radial_power_prescription='accepted face mixture: p_E=2*t^2, p_B=0.75',
        geometry_plans=[{k:row[k] for k in ('replicate','seed','path','point_sha256')}
                        for row in prepared['geometry_plans']])
    config.pop('input_manifest',None)
    profiles=[dict(row,t=t) for row in profile_templates]
    config['families']=[family_record(row,config['variants']) for row in profiles]
    write(directory/'candidate_config.json',config)
    initial=audit_geometry(directory/'candidate_config.json',directory/'initial_geometry_scope_audit')
    for index,request in enumerate(initial['requests']):
        nearest=next(row for row in profiles if row['name']==request['nearest_family'])
        extra=dict(nearest,name=f'radial_fallback_{index:03d}',
            anchor_decays=request['local_primary_decays'],
            proposal_decays=np.maximum(request['local_primary_decays'],1.).tolist(),
            source_geometry_request=request)
        profiles.append(extra)
        config['families'].append(family_record(extra,config['variants']))
    write(directory/'covered_config.json',config)
    names,plans,marks=selection(directory,config)
    used=set(names[names!=''].tolist())
    profiles=[row for row in profiles if row['name'] in used]
    config['families']=[row for row in config['families'] if row['name'] in used]
    after,_,_=selection(directory,config)
    if not np.array_equal(names,after):raise ValueError('pruning unused families changed routing')
    inventory=[]
    for profile,family in zip(profiles,config['families']):
        if profile['name']!=family['name']:raise ValueError('profile/family axes mismatch')
        manifests,paths=[],{}
        for variant in config['variants']:
            # Preserve the accepted three pilot families with direct high-order controls.
            pilot=profile['name'] in {'comb_slot0_independent_factor16_anchor0',
                'comb_slot2_independent_factor16_anchor0','star_slot0_independent_factor16_anchor0'}
            order=12 if variant=='baseline' or pilot else 10
            relative=Path('banks')/profile['name']/variant
            manifest=prepare_bank_build(directory/relative,t=t,topology=profile['topology'],
                incoming_slot=profile['incoming_slot'],proposal=proposal(profile,variant),order=order,
                coefficient_provider=_provider(bank_reference['coefficient_provider']),
                special_dps=bank_reference['special_dps'],source_dependencies=bank_reference['source_dependencies'])
            manifests.append(manifest); paths[variant]=str(relative/'bank.npz')
            spec=json.loads(manifest.read_text())
            family['variants'][variant]=dict(path=paths[variant],proposal_fingerprint=spec['proposal_fingerprint'])
        catalog=prepare_catalog(directory/'catalogs'/profile['name'],manifests)
        info=json.loads(catalog.read_text())
        inventory.append(dict(name=profile['name'],catalog=str(catalog.relative_to(directory)),
            variants=paths,node_count=info['node_count'],build_order_histogram=info['build_order_histogram']))
    write(directory/'profiles.json',profiles)
    write(directory/'full_config.json',config)
    final=audit_geometry(directory/'full_config.json',directory/'geometry_scope_audit')
    if final['fallback_request_count']:raise ValueError('final geometry coverage failed')
    write(directory/'collar_routing.json',config['collar_routing'])
    write(directory/'catalog_inventory.json',inventory)
    build_tasks=[dict(catalog=row['catalog'],start=i,stop=min(i+128,row['node_count']),
        output=str(Path(row['catalog']).parent/'shards'/f"shard_{i:07d}_{min(i+128,row['node_count']):07d}.npz"))
        for row in inventory for i in range(0,row['node_count'],128)]
    write(directory/'build_tasks.json',build_tasks)
    tasks=[dict(replicate=r,start=i,stop=i+8,config='full_config.json',output=f'full_shards/r{r}_i{i:06d}')
           for r in range(8) for i in range(0,256,8)]
    write(directory/'full_tasks.json',tasks)
    qmax=np.stack([np.abs(plan.arrays['q']).max(axis=1) for plan in plans])
    preflight=[];start=0
    for stratum in prepared['geometry_plans'][0]['radial_strata']:
        stop=start+stratum['count']
        r,offset=np.unravel_index(np.argmax(qmax[:,start:stop]),(8,stop-start))
        task=next(row for row in tasks if row['replicate']==r and row['start']<=start+offset<row['stop'])
        preflight.append(dict(task,component=stratum['component'],maximum_q=float(qmax[r,start+offset])))
        start=stop
    for request in initial['requests']:
        task=next(row for row in tasks if row['replicate']==request['replicate'] and row['start']<=request['point']<row['stop'])
        if task['output'] not in {row['output'] for row in preflight}:
            preflight.append(dict(task,component='new-bank-scope-probe'))
    write(directory/'preflight_tasks.json',preflight)
    counts={str(order):sum(row['build_order_histogram'].get(str(order),0) for row in inventory) for order in (10,12)}
    return dict(families=len(profiles),extra_family_names=[row['name'] for row in profiles if row['name'].startswith('radial_fallback_')],
        original_scope_requests=len(initial['requests']),geometry_scope_requests=0,
        build_tasks=len(build_tasks),preflight_shards=len(preflight),unique_rows_by_order=counts,
        selected_family_counts={name:int((names==name).sum()) for name in sorted(used)},
        analytic_patch_count=int(marks['inside'].sum()))


def prepare(output,accepted):
    runpy.run_path(str(ROOT/'run.py'))['bootstrap']()
    output,accepted=Path(output).resolve(),Path(accepted).resolve()
    if output.exists():raise FileExistsError('use a new production directory')
    if output.parent!=accepted.parent:raise ValueError('accepted and new releases must be sibling directories')
    plot=BASE/'order8_30point_current/order8_matrix_comparison.csv'
    with plot.open(newline='') as handle:values=[Decimal(row['t']) for row in csv.DictReader(handle)]
    if values!=[Decimal('.1805')+i*Decimal('.005') for i in range(30)]:raise ValueError('historical plot grid changed')
    template=accepted/'t0p1805'
    prototype=json.loads((template/'full_config.json').read_text())
    source_reference=BASE/'yin_three_energy_20260909_v3/t0p3155'
    profiles=[row for row in json.loads((source_reference/'profiles.json').read_text()) if not row['name'].startswith('fallback_')]
    if len(profiles)!=24:raise ValueError('expected the accepted 24 base channel families')
    bank_reference=json.loads((source_reference/'banks/comb_slot0_independent_factor16_anchor0/baseline/manifest.json').read_text())
    accepted_settings=physics_settings(prototype)
    if (prototype['maximum_per_axis_decay_ratio']!=64 or prototype['orders']!=[8,10,12]
            or prototype['base_seed']!=SEED):raise ValueError('accepted release differs from the selected production design')
    output.mkdir(parents=True)
    rows=[]
    for plot_index in PLOT_INDICES:
        value=values[plot_index];t=float(value);label='t'+str(value).replace('.','p')
        directory=output/label; reused=t in REUSED
        if reused:
            directory.mkdir()
            reference=accepted/label
            info=dict(families=0,build_tasks=0,preflight_shards=0,unique_rows_by_order={},extra_family_names=[])
            reference_files={name:digest(reference/name) for name in
                ('input_release.json','full_config.json','final_audit.json','amplitude/main/summary.json','amplitude/main/pointwise.npz')}
        else:
            info=prepare_energy(directory,t,prototype,profiles,bank_reference)
        copy_source(template,directory)
        release=dict(version='sphere-six-point-yin-radial-production-energy-v2',operation='reuse' if reused else 'compute',
            t=t,radial_power=2*t*t,base_seed=SEED,workers=192,partition='yin',account='yin_lab',
            nodes=['holy7c16611','holy7c16612','holy7c18101','holy7c18102'],sample_count=2048,
            replicates=8,points_per_replicate=256,accepted_physics_settings=accepted_settings,
            accepted_settings_source_sha256=digest(template/'full_config.json'),
            family_count=info['families'],full_task_count=0 if reused else 256,
            build_task_count=info['build_tasks'],preflight_count=info['preflight_shards'],
            unique_rows_by_order=info['unique_rows_by_order'],extra_family_names=info['extra_family_names'])
        if reused:release.update(reference_directory=f'../../{accepted.name}/{label}',reference_files=reference_files)
        release['files']={str(p.relative_to(directory)):digest(p) for p in sorted(directory.rglob('*')) if p.is_file()}
        write(directory/'input_release.json',release)
        (directory/'logs').mkdir()
        row=dict(label=label,t=t,plot_index=plot_index,operation=release['operation'],**info)
        rows.append(row)
        print(json.dumps({k:v for k,v in row.items() if k!='selected_family_counts'}),flush=True)
    shutil.copy2(PRODUCTION/'six_point_submit_radial_campaign.py',output/'submit_radial.py')
    shutil.copy2(PRODUCTION.parent/'moduli_integration/six_point_assemble_radial_curve_covariance.py',output/'aggregate_curve.py')
    copy_observable_export_sources(output)
    write(output/'campaign.json',dict(version='sphere-six-point-yin-radial-production-curve-v1',
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),energies=rows,
        selected_plot_indices=list(PLOT_INDICES),source_plot_energies=list(map(float,values)),
        source_plot=str(plot.relative_to(ROOT)),source_plot_sha256=digest(plot),
        selection_rule='Approximately uniform historical indices including endpoints and all three accepted energies; no amplitude values used.',
        base_seed=SEED,workers=192,partition='yin',account='yin_lab',initial_dependency=None,
        sequential_energies=True,accepted_reference=accepted.name,aggregate='aggregate_curve.py',
        reused_energy_count=3,new_energy_count=12,sample_count_per_energy=2048,
        cross_energy_common_scrambles=True,total_accuracy_certified=False))
    counts={str(order):sum(row['unique_rows_by_order'].get(str(order),0) for row in rows) for order in (10,12)}
    write(output/'workload.json',dict(new_energies=12,reused_energies=3,unique_coefficient_rows_by_order=counts,
        catalog_families=sum(row['families'] for row in rows),
        build_tasks=sum(row['build_tasks'] for row in rows),new_moduli_points=12*2048,
        total_curve_points=15*2048,cores=192,estimated_elapsed_hours_excluding_queue=[48,72],
        estimate_basis='Previous full-bank energies took roughly 3.6–4.5 hours each on 192 workers; allow reserve for new energies and evaluation.'))
    (output/'README.md').write_text('# Fifteen-point radial-mixture production campaign\n\n'
        'Energies: '+', '.join(str(row['t']) for row in rows)+'.\n\n'
        'The three accepted energies are verified and reused without reweighting. Twelve additional energies use the same 8 × 256 stratified samples, base seed 2026091000, p_E=2t², p_B=0.75, and per-scramble counts EEE32, BBB128, each mixed face16. '
        'All physical atlas densities are evaluated with the full mixture. The accepted CFT, collar, and momentum settings are unchanged.\n\n'
        'Only families selected by the actual geometry are built. Each family has baseline C12, common controls C10, and the same three pilot families with direct C12 controls. '
        'Each new energy runs validate → build → assemble → analytic addback → component preflight → full amplitude → reduce → audit. '
        'Build and full arrays use 192 single-CPU workers with numerical-library threads capped at one. Energy waves are sequential. Completed preflight shards are reused.\n\n'
        'The aggregate includes all 15 means, eight-scramble errors, diagnostics, and cross-energy covariance. Common scrambles correlate energies; eight scrambles imply covariance rank at most seven, so it must not be naively inverted for a 15-point fit. '
        'Matrix-model comparison is downstream of the frozen numerical runs. User acceptance does not turn the empirical errors into a total-accuracy certificate.\n\n'
        'Estimated total runtime: 48–72 hours excluding queue delays, based on the completed 192-core bank builds. Exact workload is in workload.json.\n')
    write(output/'deployment_manifest.json',{str(p.relative_to(output)):digest(p)
        for p in sorted(output.rglob('*')) if p.is_file()})
    return rows


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--accepted',type=Path,default=BASE/'yin_radial_three_energy_20260910_v2')
    args=parser.parse_args();prepare(args.output,args.accepted)
