"""Analyze a saved, common-method named-rule pilot without new CFT rows.

Analytic-patch rows have different order semantics and are labelled separately.
They are never replaced with the old native proxy or a repeated elliptic value.
Every regional contribution retains the complete original sample denominator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

if __name__ == '__main__' and not __package__:
    import runpy
    runpy.run_path(str(Path(__file__).resolve().parents[4]/'run.py'))['bootstrap']()

import numpy as np
from six_point_combine_truncation_orders import combine_two_level, summarize_two_level, source_hashes

VERSION='sphere-six-point-two-level-named-analysis-v1'


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _identity(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def analyze(input_directory, output_directory, *, low_orders=(10,8), high_order=12, controls=None):
    """Read named_variants results, returning and saving empirical diagnostics.

    The existing named runner supplies a COMMON method for all rules at each
    point. We use boundary_values_I6 when present, otherwise values_I6. Routed
    hybrid/collar arrays are intentionally not interchangeable with these
    native-degree labels. This pilot analyzer requires high-order tables in
    the saved named result; heterogeneous banks use evaluate_native_banks.
    """
    directory,output=Path(input_directory),Path(output_directory)
    if output.exists():
        raise FileExistsError('analysis output already exists; preserve prior source-bound results')
    paths=[directory/name for name in ('pointwise.npz','summary.json','input_manifest.json')]
    receipts={str(path.resolve()):_digest(path) for path in paths}
    summary=json.loads(paths[1].read_text());manifest=json.loads(paths[2].read_text())
    if summary.get('version')!='sphere-six-point-named-variants-v1':
        raise ValueError('explicit named-variants output required')
    if summary.get('pointwise_sha256')!=receipts[str(paths[0].resolve())]:
        raise ValueError('saved pointwise checksum mismatch')
    if (summary.get('identity')!=manifest.get('identity') or
            _identity({k:v for k,v in manifest.items() if k!='identity'})!=manifest.get('identity')):
        raise ValueError('saved source/bank manifest identity mismatch')
    with np.load(paths[0],allow_pickle=False) as saved:
        data={k:saved[k].copy() for k in saved.files}
    orders=data['orders'].tolist();variants=data['variants'].tolist();energies=data['energies'].tolist()
    if any(summary.get(name)!=value for name,value in [('orders',orders),('variants',variants),('energies',energies)]):
        raise ValueError('saved array axes disagree with summary')
    if (isinstance(high_order,bool) or int(high_order)!=high_order or high_order not in orders
            or not low_orders or any(isinstance(n,bool) or int(n)!=n or n not in orders or n>=high_order for n in low_orders)
            or len(set(low_orders))!=len(low_orders)):
        raise ValueError('distinct retained low orders below the retained high order required')
    low_orders=tuple(map(int,low_orders));high_order=int(high_order)
    controls=list(controls) if controls is not None else [v for v in variants if v!=summary['baseline']]
    if not controls or len(set(controls))!=len(controls) or any(v not in variants or v==summary['baseline'] for v in controls):
        raise ValueError('distinct configured non-baseline controls required')
    key='boundary_values_I6' if 'boundary_values_I6' in data else 'values_I6'
    values=data[key]
    if values.ndim!=5 or values.shape[0]!=len(energies) or values.shape[-2:]!=(len(variants),len(orders)):
        raise ValueError('invalid named-rule tensor axes')
    ne,nr,npnt,nv,no=values.shape
    shape=(ne,nr,npnt)
    if data['family_indices'].shape!=shape or data['qualified'].shape!=shape:
        raise ValueError('family/qualification shape mismatch')
    if data['points'].shape!=(nr,npnt,7) or data['q'].shape!=(nr,npnt,3):
        raise ValueError('source point inventory differs from value tensor')
    inside=np.asarray(data.get('collar_inside',np.zeros((nr,npnt),bool)))
    if inside.dtype!=np.dtype(bool) or inside.shape!=(nr,npnt):
        raise ValueError('analytic-patch mask shape/type mismatch')
    if key=='boundary_values_I6':
        if 'boundary_methods' not in data or data['boundary_methods'].shape!=shape:
            raise ValueError('boundary values require explicit shared method identities')
        methods=data['boundary_methods'].astype(str)
    else:
        methods=np.full(shape,'native-total-degree')
    if np.any((methods=='analytic-tested-collar') & ~inside[None]):
        raise ValueError('analytic order semantics disagree with patch mask')
    families=summary['bank_family_names']
    for family in families:
        for variant in [summary['baseline'],*controls]:
            meta=manifest['banks'][family][variant]
            if meta['order']<high_order:
                raise ValueError('named pilot lacks real high-order control; use heterogeneous native-bank API')
            diagnostic=summary['bank_diagnostics'][family][variant]
            if meta['fingerprint']!=diagnostic['bank_fingerprint']:
                raise ValueError('saved bank metadata disagrees with contraction receipt')
    eligible=(~inside[None]) & data['qualified']
    compared=[variants.index(v) for v in [summary['baseline'],*controls]]
    requested=[orders.index(o) for o in [*low_orders,high_order]]
    finite=np.isfinite(values[...,compared,:][...,requested]).all(axis=(-1,-2))
    eligible &= finite
    if np.any(eligible & ((data['family_indices']<0)|(data['family_indices']>=len(families)))):
        raise ValueError('evaluated points lack a physical bank family')
    sources=source_hashes();sources[str(Path(__file__).resolve())]=_digest(__file__)
    ids=np.broadcast_to(np.arange(nr)[:,None],(nr,npnt))
    outshape=(len(low_orders),ne,nr,npnt,len(controls))
    arrays={name:np.full(outshape,complex(np.nan,np.nan)) for name in ('estimate_Q5','direct_high_Q5','signed_mixed_defect_Q5')}
    arrays['absolute_mixed_defect_Q5']=np.full(outshape,np.nan)
    arrays['comparison_assessed']=np.zeros(outshape,bool)
    records=[];b=variants.index(summary['baseline']);h=orders.index(high_order)
    for e,t in enumerate(energies):
        if not np.isfinite(t) or t<=0:
            raise ValueError('positive finite physical energy required for Q5 normalization')
        scale=-1/(40*np.pi**3*t**6)
        regions={f'family:{name}':data['family_indices'][e]==f for f,name in enumerate(families)
                 if np.any(data['family_indices'][e]==f)}
        for graph,label in enumerate(('comb','star')):
            for slot in range(6):
                regions[f'{label}_slot{slot}']=(data['topology']==graph)&(data['incoming_slot']==slot)
        if 'boundary_marked' in data:regions['boundary_marked']=data['boundary_marked'].astype(bool)
        if 'boundary_block_prefix_pass' in data:
            regions['block_screen_unresolved']=~data['boundary_block_prefix_pass'][e]
        for method in np.unique(methods[e]):regions[f'method:{method}']=methods[e]==method
        regions['analytic_patch_separate']=inside
        for l,low in enumerate(low_orders):
            p=orders.index(low)
            for c,name in enumerate(controls):
                v=variants.index(name)
                result=combine_two_level(values[e,...,b,p]*scale,values[e,...,b,h]*scale,
                    values[e,...,v,p]*scale,control_high=values[e,...,v,h]*scale,
                    low_order=low,high_order=high_order,eligible=eligible[e],
                    baseline_method=methods[e],control_method=methods[e],
                    provenance=dict(input_identity=summary['identity'],value_array=key,energy=t,
                        baseline=summary['baseline'],control=name,physical_scale=scale,
                        common_method_scope='same saved per-point method across all rules and degree prefixes'))
                for dest,src in [('estimate_Q5','estimate'),('direct_high_Q5','direct_high'),
                                 ('signed_mixed_defect_Q5','signed_mixed_defect'),
                                 ('absolute_mixed_defect_Q5','absolute_mixed_defect'),
                                 ('comparison_assessed','comparison_assessed')]:arrays[dest][l,e,...,c]=result[src]
                report=summarize_two_level(result,replicate_ids=ids,regions=regions)
                report.update(t=t,control=name,baseline=summary['baseline'])
                records.append(report)
    for path,sha in receipts.items():
        if _digest(path)!=sha:raise RuntimeError('saved input changed during analysis')
    for path,sha in sources.items():
        if _digest(path)!=sha:raise RuntimeError('analysis source changed during evaluation')
    report=dict(version=VERSION,input_identity=summary['identity'],value_array=key,
        energies=energies,low_orders=list(low_orders),high_order=high_order,controls=controls,
        baseline=summary['baseline'],sample_design=summary['sample_design'],
        source_samples_per_energy=nr*npnt,replicates=nr,samples_per_replicate=npnt,
        analytic_patch_points_separately_unassessed=int(inside.sum()),
        missing_or_nonfinite_outside_count=int(((~inside[None])&~eligible).sum()),
        original_full_sample_denominator_preserved=True,
        common_boundary_method_from_saved_run=key=='boundary_values_I6',
        analytic_addback_not_included=True,raw_native_proxy_never_fills_patch=True,
        all_comparisons_available_on_complete_source=bool(eligible.all()),
        global_status='observed-not-budgeted' if eligible.all() else 'unassessed-points-remain',
        input_bank_metadata=manifest['banks'],input_bank_diagnostics=summary['bank_diagnostics'],
        source_run_sources=manifest['sources'],input_receipts=receipts,analysis_sources=sources,
        inherited_block_screen_status=summary.get('boundary_treatment',{}).get('block_prefix_pass_points'),
        scope='Paired finite-rule two-level pilot. Analytic patch and unavailable points are separate; observed regional sums use full source denominator. No tail bound or certified amplitude.',
        full_mixed_production_ready=False,total_accuracy_certified=False,records=records)
    output.mkdir(parents=True,exist_ok=False)
    with (output/'pointwise.npz').open('xb') as handle:
        np.savez_compressed(handle,**arrays,eligible=eligible,analytic_patch=inside,
            family_indices=data['family_indices'],points=data['points'],q=data['q'],
            common_methods=methods,energies=np.asarray(energies),low_orders=np.asarray(low_orders),
            high_order=np.asarray(high_order),controls=np.asarray(controls))
    report['pointwise_sha256']=_digest(output/'pointwise.npz')
    with (output/'summary.json').open('x') as handle:json.dump(report,handle,indent=2,allow_nan=False)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_directory');parser.add_argument('output_directory')
    parser.add_argument('--low-orders',type=int,nargs='+',default=[10,8])
    parser.add_argument('--high-order',type=int,default=12)
    args=parser.parse_args()
    report=analyze(args.input_directory,args.output_directory,low_orders=args.low_orders,high_order=args.high_order)
    print(json.dumps(dict(global_status=report['global_status'],comparisons=len(report['records']),
        analytic_patch_points=report['analytic_patch_points_separately_unassessed']),indent=2))


if __name__=='__main__':main()
