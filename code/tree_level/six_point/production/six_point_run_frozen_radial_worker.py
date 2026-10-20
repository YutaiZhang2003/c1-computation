#!/usr/bin/env python3
"""Production worker for the accepted radial proposal, including completed-data reuse."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import runpy
import shutil
import time


def physics_settings(config):
    variable = {'energies', 'families', 'geometry_plans', 'sample_design', 'base_seed',
                'replicates', 'sobol_power', 'radial_power', 'radial_proposal',
                'radial_power_prescription', 'input_manifest'}
    return {key:value for key,value in config.items() if key not in variable}


def validate_continuation_coverage(directory, release, inventory):
    """Saved and new work must partition the original complete catalog exactly."""
    import numpy as np
    from sphere_six_point_multilevel_catalog import load_catalog, _orders, _read_shard
    tasks = json.loads((directory/'build_tasks.json').read_text())
    saved = json.loads((directory/'imported_tasks.json').read_text())
    original = json.loads((directory/'original_build_tasks.json').read_text())
    key = lambda row:(row['catalog'], row['start'], row['stop'], row['output'])
    combined = list(map(key, tasks+saved))
    if (len(tasks) != release['build_task_count']
            or len(saved) != release['imported_shard_count']
            or len(set(combined)) != len(combined)
            or set(combined) != set(map(key, original)) or len(original) != len(combined)):
        raise ValueError('continuation tasks duplicate, omit or change original work')
    imported_rows = 0
    for item in inventory:
        path = directory/item['catalog']
        manifest, nodes, _ = load_catalog(path)
        orders = _orders(path, manifest)
        coverage = np.zeros(len(nodes), dtype=np.int8)
        approved = {(r['row_start'], r['row_stop']) for r in
                    manifest.get('producer_import', {}).get('shards', [])}
        observed = set()
        for row in tasks+saved:
            if row['catalog'] != item['catalog']:
                continue
            a, b = row['start'], row['stop']
            if not 0 <= a < b <= len(nodes):
                raise ValueError('invalid continuation interval')
            coverage[a:b] += 1
        for row in saved:
            if row['catalog'] != item['catalog']:
                continue
            meta, _, _, _ = _read_shard(directory/row['output'], manifest, nodes, orders)
            pair = (meta['row_start'], meta['row_stop'])
            if pair != (row['start'], row['stop']) or 'import_origin' not in meta:
                raise ValueError('saved shard does not match its original task')
            observed.add(pair)
            imported_rows += row['stop']-row['start']
        if not np.all(coverage == 1) or observed != approved:
            raise ValueError('catalog continuation coverage or imported provenance differs')
    if imported_rows != release['imported_coefficient_rows']:
        raise ValueError('imported coefficient row count differs')
    return dict(imported_shards=len(saved), imported_rows=imported_rows,
                new_build_tasks=len(tasks), exact_original_task_partition=True)


def execute(directory, action, index=0):
    directory = Path(directory).resolve()
    runpy.run_path(str(directory / 'source/run.py'))['bootstrap']()
    import numpy as np
    import sphere_six_point_yin_curve_worker_v2 as common
    from sphere_six_point_radial_proposal import face_stratified_radial_proposal
    from sphere_six_point_named_variants import _load_plans
    directory, release = common.initialize(directory)
    if release['version'] != 'sphere-six-point-yin-radial-production-energy-v2':
        raise ValueError('production radial release required')
    if release['operation'] == 'reuse':
        if action != 'reuse':
            raise ValueError('completed energy permits verification/reuse only')
        reference = (directory / release['reference_directory']).resolve()
        for relative, expected in release['reference_files'].items():
            if common.digest(reference / relative) != expected:
                raise ValueError('completed reference changed: '+relative)
        original_release = json.loads((reference/'input_release.json').read_text())
        for relative, expected in original_release['files'].items():
            if common.digest(reference/relative) != expected:
                raise ValueError('completed reference input changed: '+relative)
        config = json.loads((reference/'full_config.json').read_text())
        output_audit = json.loads((reference/'final_audit.json').read_text())
        if (config['radial_proposal'] != face_stratified_radial_proposal(release['t'])
                or config['base_seed'] != release['base_seed']
                or physics_settings(config) != release['accepted_physics_settings']):
            raise ValueError('completed result used another design')
        _load_plans(config, reference)
        with np.load(reference/'amplitude/main/pointwise.npz') as data:
            values=data['hybrid_I6']; variant=data['variants'].tolist().index('baseline')
            if values.shape != (1,8,256,8) or not np.isfinite(values).all():
                raise ValueError('completed amplitude coverage/finiteness mismatch')
            means=values[0,...,variant].mean(axis=1)*(-1/(40*np.pi**3*release['t']**6))
            np.testing.assert_allclose([means.real.mean(),means.real.std(ddof=1)/np.sqrt(8)],
                [output_audit['mean_real'],output_audit['scramble_se_real']],rtol=1e-13)
        if (not output_audit['exact_complete_source'] or output_audit['unresolved_scope_count']
                or output_audit['numerical_failure_count']):
            raise ValueError('completed reference failed its execution audit')
        for relative in ('final_audit.json','amplitude/main/summary.json','amplitude/main/pointwise.npz'):
            target=directory/relative; target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():
                if common.digest(target)!=common.digest(reference/relative):
                    raise ValueError('existing copied amplitude differs')
            else: shutil.copyfile(reference/relative,target)
        receipt=dict(passed=True,t=release['t'],original_directory=str(reference),
            original_files=release['reference_files'],copied_without_reweighting=True,
            exact_accepted_design=True,independent_means_and_SE_verified=True)
        common.write(directory/'reuse_receipt.json',receipt)
        return receipt
    if release['operation'] != 'compute':
        raise ValueError('unknown energy operation')
    config = json.loads((directory/'full_config.json').read_text())
    if (config['radial_proposal'] != face_stratified_radial_proposal(release['t'])
            or config['base_seed'] != release['base_seed']
            or config['sample_design'] != 'stratified-sobol-radial-mixture'
            or config['replicates'] != 8 or config['sobol_power'] != 8
            or physics_settings(config) != release['accepted_physics_settings']):
        raise ValueError('production settings differ from the accepted design')
    if action == 'validate':
        from sphere_six_point_multilevel_catalog import load_catalog
        _load_plans(config,directory)
        scope=json.loads((directory/'geometry_scope_audit/summary.json').read_text())
        if (scope['config_sha256'] != common.digest(directory/'full_config.json')
                or scope['fallback_request_count'] != 0 or scope['source_point_count'] != 2048):
            raise ValueError('incomplete production geometry scope')
        inventory=json.loads((directory/'catalog_inventory.json').read_text())
        if {row['name'] for row in inventory} != {row['name'] for row in config['families']}:
            raise ValueError('catalog families disagree with evaluation')
        rows_by_order={}
        for item in inventory:
            manifest,_,_=load_catalog(directory/item['catalog'])
            if manifest['physical']['t'] != release['t'] or manifest['node_count'] != item['node_count']:
                raise ValueError('catalog energy or row count mismatch')
            for order,count in manifest['build_order_histogram'].items():
                rows_by_order[order]=rows_by_order.get(order,0)+count
        if rows_by_order != release['unique_rows_by_order']:
            raise ValueError('catalog coefficient workload mismatch')
        continuation = (validate_continuation_coverage(directory, release, inventory)
                        if release.get('optimized_continuation') else None)
        tasks=json.loads((directory/'full_tasks.json').read_text())
        pairs=[(task['replicate'],p) for task in tasks for p in range(task['start'],task['stop'])]
        if len(pairs)!=2048 or set(pairs)!={(r,p) for r in range(8) for p in range(256)}:
            raise ValueError('full tasks omit or duplicate source points')
        result=dict(passed=True,t=release['t'],families=len(inventory),rows_by_order=rows_by_order,
                    source_points=2048,settings_match_accepted_run=True,
                    continuation=continuation)
        common.write(directory/'validation.json',result)
        return result
    if action == 'preflight':
        from sphere_six_point_multilevel_scan import run_multilevel
        tasks=json.loads((directory/'preflight_tasks.json').read_text())
        if not 0<=index<len(tasks): raise ValueError('preflight index outside release')
        task=tasks[index]; output=directory/task['output']
        geometry_slice={key:task[key] for key in ('replicate','start','stop')}
        result=common.existing_scan_output(output,scope='geometry-shard',geometry_slice=geometry_slice)
        start=time.time()
        if result is None:
            result=run_multilevel(directory/'full_config.json',output,geometry_slice=geometry_slice,
                                 shared_addback=directory/'shared_addback.npz')
        if not result['finite_all_evaluated_values'] or result['unresolved_scope_count']:
            raise ValueError('production CFT preflight failed')
        record=dict(passed=True,task=task,seconds=time.time()-start,
                    summary_sha256=common.digest(output/'summary.json'))
        common.write(directory/f'progress/preflight/worker_{index:03d}.json',record)
        return record
    result=common.execute(directory,action,index)
    if action=='audit':
        result.update(version='sphere-six-point-yin-radial-production-audit-v2',
            base_seed=release['base_seed'],radial_proposal=config['radial_proposal'],
            accepted_physics_settings_verified=True,extra_family_names=release['extra_family_names'])
        with np.load(directory/'amplitude/main/pointwise.npz') as data:
            baseline=config['variants'].index('baseline')
            scale=-1/(40*np.pi**3*release['t']**6)
            prefix=data['treated_prefix_I6'][0,...,baseline,:]*scale
            norm=abs(result['mean_real'])
            if np.isfinite(prefix).all():
                result['weighted_block_prefix_relative_absolute_increments']=[
                    float(np.abs(prefix[...,i+1]-prefix[...,i]).mean()/norm) for i in range(2)]
        common.write(directory/'final_audit.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('action',choices=('reuse','validate','build','assemble','addback','preflight','full','reduce','audit'))
    parser.add_argument('--index',type=int,default=0)
    args=parser.parse_args()
    print(json.dumps(execute(args.directory,args.action,args.index),indent=2,allow_nan=False))
