"""Full-source mixed-degree sphere scan; no invented high-order controls.

The primary result uses the direct baseline H integral. Control estimates use
Q_v f_L + Q_b(f_H-f_L), with actual bank orders and common boundary methods.
Analytic collar terms have separate axes and are inserted after combination.
Geometry shards retain original indices and cannot be reported as full means.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path
import time

if __name__=='__main__' and not __package__:
    import runpy
    runpy.run_path(str(Path(__file__).resolve().parents[4]/'run.py'))['bootstrap']()

import numpy as np
import six_point_compare_momentum_quadratures as named
from six_point_batch_block_contraction import SphereBatchEvaluator
from six_point_sample_integration_geometry import point_digest
from six_point_momentum_banks_and_reweighting import SphereSixPointMomentumBank
from six_point_combine_truncation_orders import combine_two_level, _hash_array
from six_point_measure_block_convergence import evaluate_native_prefixes

VERSION='sphere-six-point-mixed-degree-scan-v2'
BANK_REFERENCE_VERSION='sphere-six-point-bank-evidence-reference-v1'


def _stream_json_hash(value):
    """Canonical identity without constructing a second large JSON string."""
    digest=hashlib.sha256()
    for chunk in json.JSONEncoder(sort_keys=True,separators=(',',':'),allow_nan=False).iterencode(value):
        digest.update(chunk.encode())
    return digest.hexdigest()


def _compact_bank_metadata(metadata,path,file_sha256):
    """The original hash-bound NPZ keeps ALL verification/shard evidence.

    Drop only the replicated provenance from the in-memory routing descriptor.
    Physical/provider/proposal fields are retained verbatim. This is a receipt,
    never a replacement bank and never a fresh verification attestation.
    """
    if 'provenance' not in metadata or 'evidence_reference' in metadata:
        raise ValueError('expected original full bank metadata, not another compact receipt')
    reference=dict(version=BANK_REFERENCE_VERSION,path=str(Path(path).resolve()),
        file_sha256=file_sha256,metadata_sha256=_stream_json_hash(metadata),
        provenance_sha256=_stream_json_hash(metadata['provenance']))
    return {**{k:v for k,v in metadata.items() if k!='provenance'},'evidence_reference':reference}


def _load_families(records,base,energies,variants,orders):
    """Validate and compact each bank immediately; never retain full families.

    These are the named-runner's physical checks with the same bank loader at
    contraction time. In particular no coefficient, node, weight or CFT value
    is changed by the storage-only receipt schema.
    """
    families,files,provider=[],{},None
    for record in records:
        name=record['name']
        if not isinstance(name,str) or not name or name in {f.name for f in families}:
            raise ValueError('distinct nonempty family names required')
        t,topology=float(record['t']),record['topology']
        slot=named._integer(record['incoming_slot'],'incoming_slot')
        anchor=tuple(float(x) for x in record['anchor_decays'])
        if (t not in energies or topology not in {'comb','star'} or slot>5 or len(anchor)!=3
                or any(not math.isfinite(x) or x<=0 for x in anchor)):
            raise ValueError('family has invalid physical identity or anchor decays')
        if set(record['variants'])!=set(variants):raise ValueError('every family must explicitly supply every named variant')
        banks,metadata={},{}
        for variant in variants:
            entry=record['variants'][variant]
            descriptor={'path':entry} if isinstance(entry,str) else entry
            path=named._path(base,descriptor['path'])
            if not path.is_file():raise FileNotFoundError(f'missing explicit bank for {name}/{variant}: {path}')
            digest=named._digest(path);files[str(path)]=digest
            with np.load(path,allow_pickle=False) as payload:
                full=json.loads(payload['metadata'].item())
            if (full.get('version')!=named.MOMENTUM_BANK_VERSION
                    or (full['t'],full['topology'],full['incoming_slot'])!=(t,topology,slot)
                    or full['order']<max(orders)):
                raise ValueError('bank physical identity or complete block prefix mismatch')
            policy=named._provider_metadata(full['provider'])
            if provider is None:provider=policy
            if policy!=provider:raise ValueError('named banks mix coefficient provider/verification policies')
            proposal=full['proposal']
            if tuple(proposal.get('anchor_decays',()))!=anchor or proposal.get('complete_rule',True) is not True:
                raise ValueError('bank has a different node anchor or incomplete momentum rule')
            if 'variant' in proposal and proposal['variant']!=variant:
                raise ValueError('bank proposal variant label differs from its configured role')
            if descriptor.get('fingerprint',full['fingerprint'])!=full['fingerprint']:
                raise ValueError('configured bank fingerprint mismatch')
            banks[variant]=dict(path=path,fingerprint=full['fingerprint'],
                proposal_fingerprint=descriptor.get('proposal_fingerprint'))
            metadata[variant]=_compact_bank_metadata(full,path,digest)
            del full
        families.append(named.NamedFamily(name,t,topology,slot,anchor,banks,metadata))
    if not families:raise ValueError('at least one complete explicit bank family required')
    return families,files


def _validate_bank_references(manifest):
    """Verify every original bank once, including unused configured families."""
    records=manifest['banks'];cfg=manifest['config'];checked={}
    config_path=Path(manifest['config_path'])
    if (named._digest(config_path)!=manifest['config_sha256']
            or json.loads(config_path.read_text())!=cfg):
        raise ValueError('referenced scan configuration changed')
    if set(records)!={f['name'] for f in cfg['families']}:
        raise ValueError('bank evidence families disagree with configuration')
    for family in cfg['families']:
        if set(records[family['name']])!=set(cfg['variants']):
            raise ValueError('bank evidence variants disagree with configuration')
        for variant,compact in records[family['name']].items():
            ref=compact.get('evidence_reference',{})
            if ref.get('version')!=BANK_REFERENCE_VERSION or 'provenance' in compact:
                raise ValueError('compact original-bank evidence reference required')
            path=Path(ref['path'])
            descriptor=family['variants'][variant]
            configured_path=descriptor if isinstance(descriptor,str) else descriptor['path']
            if path!=named._path(config_path.parent,configured_path):
                raise ValueError('bank evidence path disagrees with configured bank')
            if manifest['files'].get(str(path))!=ref['file_sha256']:
                raise ValueError('bank evidence file hash disagrees with input manifest')
            if str(path) not in checked:
                if not path.is_file():raise FileNotFoundError(f'missing referenced original bank: {path}')
                digest=named._digest(path)
                if digest!=ref['file_sha256']:raise ValueError(f'referenced bank file checksum mismatch: {path}')
                with np.load(path,allow_pickle=False) as payload:full=json.loads(payload['metadata'].item())
                checked[str(path)]=_compact_bank_metadata(full,path,digest)
                del full
            if checked[str(path)]!=compact:
                raise ValueError('original bank metadata/provenance identity or retained physical fields changed')
            if ((compact['t'],compact['topology'],compact['incoming_slot'])!=
                    (float(family['t']),family['topology'],family['incoming_slot'])
                    or tuple(compact['proposal'].get('anchor_decays',()))!=tuple(family['anchor_decays'])):
                raise ValueError('referenced bank physical identity disagrees with configured family')
            if isinstance(descriptor,dict) and descriptor.get('fingerprint',compact['fingerprint'])!=compact['fingerprint']:
                raise ValueError('referenced bank fingerprint disagrees with configured descriptor')
    return {path:meta['evidence_reference']['file_sha256'] for path,meta in checked.items()}


def _validate_bank_diagnostics(summary,manifest,data):
    """Contraction receipts must point to the same original evidence as inputs."""
    diagnostics=summary['bank_diagnostics']
    if set(diagnostics)!=set(manifest['banks']):raise ValueError('bank diagnostic family set changed')
    cfg=manifest['config']
    for index,family_record in enumerate(cfg['families']):
        family=family_record['name'];variants=diagnostics[family]
        used=np.any((data['family_indices']==index)&~data['inside_patch'][None])
        if set(variants)!=(set(cfg['variants']) if used else set()):
            raise ValueError('bank diagnostic variants do not cover exactly the contracted source points')
        for variant,receipt in variants.items():
            if variant not in manifest['banks'][family]:raise ValueError('unknown bank diagnostic variant')
            meta=manifest['banks'][family][variant]
            if ('provenance' in receipt or receipt.get('bank_reference')!=meta['evidence_reference']
                    or receipt.get('bank_fingerprint')!=meta['fingerprint']
                    or receipt.get('actual_bank_order')!=meta['order']
                    or receipt.get('actual_evaluated_orders')!=[n for n in cfg['orders'] if n<=meta['order']]):
                raise ValueError('contraction receipt disagrees with hash-bound bank evidence')


def _write_json(path,value):
    with Path(path).open('x') as handle:
        json.dump(value,handle,indent=2,allow_nan=False)
        handle.write('\n')


class OutOfScopeBankError(ValueError):
    def __init__(self,requests):
        self.requests=requests
        super().__init__(f'{len(requests)} source points require qualified fallback momentum banks; no CFT contraction started')


def _load_config(path):
    path=Path(path).resolve();config=json.loads(path.read_text())
    energies=tuple(float(t) for t in config['energies'])
    orders=tuple(named._integer(n,'order') for n in config['orders'])
    low=named._integer(config.get('low_order',10),'low order')
    high=named._integer(config.get('high_order',12),'high order')
    variants=list(config['variants']);baseline=config['baseline']
    ratio=float(config['maximum_per_axis_decay_ratio'])
    if (not energies or len(set(energies))!=len(energies) or any(not 0<t<1/3 for t in energies)
            or len(orders)<3 or list(orders)!=sorted(set(orders)) or not 0<=low<high
            or low not in orders or high!=max(orders) or not any(n<low for n in orders)
            or not variants or len(set(variants))!=len(variants) or baseline not in variants
            or any(not isinstance(v,str) or not v for v in variants)
            or not math.isfinite(ratio) or ratio<1):
        raise ValueError('invalid physical energies, stored-prefix levels, variants or scope ratio')
    policy=config.get('out_of_scope_policy','require-fallback')
    if policy not in ('require-fallback','diagnostic'):
        raise ValueError('out_of_scope_policy must be require-fallback or explicit diagnostic')
    if 'boundary_treatment' in config and 'collar_routing' not in config:
        raise ValueError('boundary treatment requires explicit collar geometry marking')
    return path,config,energies,orders,low,high,variants,baseline,ratio


def _sources(routing=False):
    for name in ('six_point_batch_block_contraction','six_point_sample_integration_geometry','six_point_stable_puncture_coordinates',
                 'six_point_channel_coordinates','six_point_select_integration_channels','sphere_mobius_geometry',
                 'six_point_momentum_banks_and_reweighting','six_point_radial_sampling_distribution','six_point_check_sampling_variance'):
        importlib.import_module(name)
    sources=named._sources()
    for name in ('six_point_combine_truncation_orders','six_point_measure_block_convergence'):
        module=importlib.import_module(name);sources[str(Path(module.__file__).resolve())]=named._digest(module.__file__)
    sources[str(Path(__file__).resolve())]=named._digest(__file__)
    if routing:
        module=importlib.import_module('six_point_evaluate_routed_geometry_shards')
        sources.update(module.source_hashes())
    return sources


def _source_geometry(plans,config):
    return [dict(replicate=r,point_count=len(p.arrays['points']),point_sha256=p.metadata['point_sha256'],
        path=str(named._path(Path(config['_config_path']).parent,config['geometry_plans'][r]['path'])),
        output_array_hashes={key:_hash_array(p.arrays[key]) for key in
            ('points','q','topology','ordering','log_density','pair_logs')}) for r,p in enumerate(plans)]


def _requests(plans,energies,families,selected,ratios,qualified,inside,original_replicates,original_indices):
    requests=[]
    for e,r,n in np.argwhere(~qualified & ~inside[None]):
        point=plans[r].arrays;f=int(selected[e,r,n]);family=families[f]
        q=point['q'][n];decays=-2*np.log(abs(q))
        requests.append(dict(t=float(energies[e]),replicate=int(original_replicates[r]),
            point=int(original_indices[r,n]),topology=family.topology,incoming_slot=family.incoming_slot,
            nearest_family=family.name,nearest_anchor=list(family.anchor_decays),
            local_primary_decays=decays.tolist(),maximum_decay_ratio=float(ratios[e,r,n]),
            q=[[float(x.real),float(x.imag)] for x in q],
            required_action='prepare and qualify a complete momentum family covering this original point; retain point and source density'))
    return requests


def audit_geometry(config_path,output_directory):
    """Require no bank files: report every out-of-scope original geometry row."""
    path,cfg,energies,orders,low,high,variants,baseline,ratio=_load_config(config_path)
    output=Path(output_directory)
    if output.exists():raise FileExistsError('preserve previous geometry audit')
    plans,files=named._load_plans(cfg,path.parent)
    families=[]
    for record in cfg['families']:
        anchor=tuple(float(a) for a in record['anchor_decays'])
        if len(anchor)!=3 or any(not math.isfinite(a) or a<=0 for a in anchor):raise ValueError('invalid family anchor')
        families.append(named.NamedFamily(record['name'],float(record['t']),record['topology'],int(record['incoming_slot']),anchor,{},{}))
    if 'collar_routing' in cfg:
        from six_point_estimate_collar_corrected_integrand import mark_plans
        marks=mark_plans(plans,cfg['collar_routing']);inside=marks['inside']
    else:inside=np.zeros((len(plans),len(plans[0].arrays['points'])),bool)
    selected,ratios,qualified,_,_,_=named._select_families(plans,energies,families,ratio,excluded=inside)
    indices=np.broadcast_to(np.arange(inside.shape[1]),inside.shape)
    requests=_requests(plans,energies,families,selected,ratios,qualified,inside,np.arange(len(plans)),indices)
    for plan in plans:plan.require_sources()
    report=dict(version=VERSION,config_sha256=named._digest(path),sample_design=cfg['sample_design'],
        source_point_count=inside.size,analytic_patch_count=int(inside.sum()),fallback_request_count=len(requests),
        all_outside_points_in_declared_scope=not requests,requests=requests,files=files,
        coefficient_or_DOZZ_tables_evaluated=0,geometry_resampled=False,
        source_density_unchanged=True,scope_ratio_is_error_bound=False)
    output.mkdir(parents=True,exist_ok=False)
    with (output/'pointwise.npz').open('xb') as handle:np.savez_compressed(handle,
        selected=selected,decay_ratios=ratios,qualified=qualified,inside_patch=inside,
        points=np.stack([p.arrays['points'] for p in plans]),energies=energies)
    report['pointwise_sha256']=named._digest(output/'pointwise.npz')
    (output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


def _combine(values,methods,inside,orders,low,high,variants,baseline):
    shape=values.shape[:-1];b=variants.index(baseline);l=orders.index(low);h=orders.index(high)
    final=np.full(shape,complex(np.nan,np.nan));direct=np.full(shape,complex(np.nan,np.nan))
    defects=np.full(shape,complex(np.nan,np.nan));assessed=np.zeros(shape,bool)
    correction=values[...,b,h]-values[...,b,l]
    outside=~inside[None]
    final[...,b]=np.where(outside,values[...,b,h],complex(np.nan,np.nan))
    direct[...,b]=final[...,b]
    assessed[...,b]=outside & np.isfinite(final[...,b])
    defects[...,b]=np.where(assessed[...,b],0j,complex(np.nan,np.nan))
    for v,name in enumerate(variants):
        if v==b:continue
        eligible=outside & np.isfinite(values[...,b,l]) & np.isfinite(values[...,b,h]) & np.isfinite(values[...,v,l])
        available=outside & np.isfinite(values[...,v,h])
        high_values=np.where(available,values[...,v,h],complex(np.nan,np.nan))
        result=combine_two_level(values[...,b,l],values[...,b,h],values[...,v,l],
            control_high=high_values,high_available=available,eligible=eligible,
            low_order=low,high_order=high,baseline_method=methods,control_method=methods)
        final[...,v]=result['estimate'];direct[...,v]=result['direct_high']
        defects[...,v]=result['signed_mixed_defect'];assessed[...,v]=result['comparison_assessed']
    return dict(multilevel_native_I6=final,direct_high_native_I6=direct,mixed_defect_I6=defects,
        mixed_defect_assessed=assessed,baseline_correction_I6=np.where(outside,correction,complex(np.nan,np.nan)))


def _native_flags(raw,increments,selected,qualified,available,variants,baseline,tolerance,inside):
    shape=selected.shape;b=variants.index(baseline)
    baseline_pass=np.zeros(shape,bool);all_available=np.ones(shape,bool);strict=np.ones(shape,bool)
    for f in range(len(available)):
        mask=selected==f
        for v in range(len(variants)):
            columns=np.flatnonzero(available[f,v]);pairs=columns[1:]-1
            finite=np.isfinite(raw[...,v,columns]).all(axis=-1)
            tested=np.isfinite(increments[...,v,pairs]).all(axis=-1)
            if len(pairs):tested &= (increments[...,v,pairs[-2:]]<tolerance).all(axis=-1)
            else:tested[:]=False
            ok=finite & tested
            all_available[mask] &= ok[mask]
            strict[mask] &= bool(len(columns)>=3)
            if v==b:baseline_pass[mask]=(ok & bool(len(columns)>=3))[mask]
    baseline_pass &= qualified & ~inside[None]
    all_available &= qualified & ~inside[None]
    return baseline_pass,all_available,all_available & strict


def _statistics(values,energies,variants,baseline,*,full_net,scope,source_count):
    result=[]
    for e,t in enumerate(energies):
        scale=-1/(40*np.pi**3*t**6) if full_net else 1.
        row=dict(t=t,units='Q5' if full_net else 'I6 importance-weight samples',variants={})
        for v,name in enumerate(variants):
            value=values[e,...,v]*scale;difference=value-values[e,...,variants.index(baseline)]*scale
            if scope=='full-source':
                row['variants'][name]=dict(value=named._statistics(value,full_net=full_net),
                    paired_difference=named._statistics(difference,full_net=full_net),
                    absolute_difference=named._statistics(abs(difference),full_net=full_net))
            else:
                finite=bool(np.isfinite(value).all())
                row['variants'][name]=dict(complete_shard=finite,full_integral_mean=None,standard_error=None,
                    shard_sum=named._pair(value.sum()) if finite else None,
                    contribution_to_original_full_mean=named._pair(value.sum()/source_count) if finite else None,
                    original_full_denominator=source_count,
                    interpretation='partial indexed contribution; every other source index is still required')
        result.append(row)
    return result


def _original_records(value,replicates,indices):
    if isinstance(value,list):return [_original_records(v,replicates,indices) for v in value]
    if isinstance(value,dict):
        result={k:_original_records(v,replicates,indices) for k,v in value.items()}
        if 'replicate' in value and 'point' in value:
            r,n=int(value['replicate']),int(value['point'])
            result['replicate'],result['point']=int(replicates[r]),int(indices[r,n])
        return result
    return value


def run_multilevel(config_path,output_directory,*,geometry_slice=None,shared_addback=None,input_manifest=None):
    path,cfg,energies,orders,low,high,variants,baseline,ratio=_load_config(config_path)
    output=Path(output_directory).resolve()
    if output.exists():raise FileExistsError('preserve previous scan output')
    started=time.perf_counter();config_hash=named._digest(path)
    full_plans,files=named._load_plans(cfg,path.parent)
    full_count=len(full_plans[0].arrays['points']);full_replicates=len(full_plans)
    full_net=named.is_full_integral_design(cfg['sample_design'])
    sampling=[named.sampling_diagnostic(t,full_plans[0],require=full_net) for t in energies]
    families,bank_files=_load_families(cfg['families'],path.parent,energies,variants,(low,))
    files.update(bank_files)
    b=variants.index(baseline)
    available=np.asarray([[[n<=f.metadata[v]['order'] for n in orders] for v in variants] for f in families],bool)
    if not available[:,b,-1].all():raise ValueError('every baseline must store a complete actual high prefix')
    # Full and sliced execution must have the same frozen source identity.
    routing=importlib.import_module('six_point_evaluate_routed_geometry_shards')
    source_geometry=_source_geometry(full_plans,{**cfg,'_config_path':str(path)})
    if geometry_slice is None:
        plans=full_plans;original_replicates=np.arange(full_replicates,dtype=np.int64)
        original_indices=np.broadcast_to(np.arange(full_count,dtype=np.int64),(full_replicates,full_count)).copy()
        scope='full-source'
    else:
        if set(geometry_slice)!= {'replicate','start','stop'}:raise ValueError('geometry_slice requires replicate,start,stop')
        r,start,stop=(named._integer(geometry_slice[k],k) for k in ('replicate','start','stop'))
        if not (0<=r<full_replicates and 0<=start<stop<=full_count):raise ValueError('geometry shard outside original source indices')
        original_replicates=np.asarray([r],np.int64);original_indices=np.arange(start,stop,dtype=np.int64)[None]
        plans=[routing.slice_plan(full_plans[r],original_indices[0])];scope='geometry-shard'
    inside=np.zeros((len(plans),len(plans[0].arrays['points'])),bool);marks=None;collar=None
    if 'collar_routing' in cfg:
        from six_point_estimate_collar_corrected_integrand import mark_plans
        marks=mark_plans(plans,cfg['collar_routing']);inside=marks['inside']
    selected,ratios,qualified,topology,slot,q=named._select_families(plans,energies,families,ratio,excluded=inside)
    requests=_requests(plans,energies,families,selected,ratios,qualified,inside,original_replicates,original_indices)
    if requests and cfg.get('out_of_scope_policy','require-fallback')=='require-fallback':raise OutOfScopeBankError(requests)
    sources=_sources(routing is not None)
    manifest=dict(version=VERSION,bank_evidence_schema=BANK_REFERENCE_VERSION,
        config_path=str(path),config_sha256=config_hash,config=cfg,files=files,sources=sources,
        banks={f.name:f.metadata for f in families},source_geometry=source_geometry,
        full_source_replicates=full_replicates,full_source_points_per_replicate=full_count)
    if 'collar_routing' in cfg:
        if shared_addback is None:
            if scope!='full-source':raise ValueError('geometry shards require a separately computed shared addback')
            addback=routing.evaluate_marked_addback(energies,cfg['collar_routing'])
        elif isinstance(shared_addback,(str,Path)):
            addback=routing.load_marked_addback(shared_addback,energies,cfg['collar_routing'])
        else:addback=shared_addback
        points=routing.evaluate_marked_points(plans,energies,marks,cfg['collar_routing'])
        collar=routing.combine_marked(points,addback,energies,cfg['collar_routing'])
        manifest['shared_addback_identity']=dict(context_identity=addback['context_identity'],
            addback_sha256=_hash_array(addback['addback']),lower_x_sha256=_hash_array(addback['addback_lower_x']))
        manifest['shared_addback_context']=addback['context']
    manifest['identity']=named._json_hash(manifest)
    if input_manifest is not None:
        expected=json.loads(Path(input_manifest).read_text())
        if expected!=manifest:raise ValueError('frozen mixed-scan input manifest differs')
    shape=(*selected.shape,len(variants),len(orders));nan=complex(np.nan,np.nan)
    raw=np.full(shape,nan);inc=np.full((*shape[:-1],len(orders)-1),np.nan)
    absolute=np.full_like(inc,np.nan);denominators=np.full(shape[:-1],np.nan)
    zero_counts=np.full(shape[:-1],-1,np.int64)
    base=np.stack([np.stack([p.base_logarithms(t) for p in plans]) for t in energies])
    batch=named._integer(cfg.get('batch_size',16),'batch_size',1);failures=[];receipts={}
    for f,family in enumerate(families):
        e=energies.index(family.t);indices=np.argwhere((selected[e]==f)&~inside)
        receipts[family.name]={}
        if not len(indices):continue
        for v,name in enumerate(variants):
            desc=family.banks[name]
            bank=SphereSixPointMomentumBank.load(desc['path'],expected_fingerprint=desc['fingerprint'],expected_provider_metadata=family.metadata[name]['provider'])
            if desc.get('proposal_fingerprint') is not None and bank.proposal.fingerprint!=desc['proposal_fingerprint']:
                raise ValueError('actual proposal fingerprint differs from configured control')
            actual=tuple(n for n,keep in zip(orders,available[f,v]) if keep)
            evaluator=SphereBatchEvaluator(bank.as_momentum_arrays())
            reference=family.metadata[name]['evidence_reference']
            if _stream_json_hash(bank.provenance)!=reference['provenance_sha256']:
                raise ValueError('loaded bank provenance differs from input evidence reference')
            receipts[family.name][name]=dict(bank_fingerprint=bank.fingerprint,proposal_fingerprint=bank.proposal.fingerprint,
                actual_bank_order=bank.order,actual_evaluated_orders=list(actual),momentum_rows=len(bank.p_values),bank_reference=reference)
            def evaluate(rows):
                r,n=rows.T
                result=evaluate_native_prefixes(evaluator,q[r,n],base_logarithms=base[e,r,n],orders=actual,batch_size=len(rows))
                for o in actual:raw[e,r,n,v,orders.index(o)]=result['values'][o]
                for pair,values in result['relative_absolute_weighted_increments'].items():
                    j=orders.index(pair[1])-1;inc[e,r,n,v,j]=values;absolute[e,r,n,v,j]=result['absolute_weighted_increments'][pair]
                denominators[e,r,n,v]=result['normalization_denominator'];zero_counts[e,r,n,v]=result['zero_absolute_weight_node_counts']
            for start in range(0,len(indices),batch):
                rows=indices[start:start+batch]
                try:evaluate(rows)
                except (FloatingPointError,OverflowError):
                    for row in rows:
                        try:evaluate(row[None])
                        except (FloatingPointError,OverflowError) as error:
                            r,n=row;failures.append(dict(t=family.t,replicate=int(original_replicates[r]),point=int(original_indices[r,n]),family=family.name,variant=name,reason=str(error)))
            del evaluator,bank
    tolerance=float(cfg.get('boundary_treatment',{}).get('relative_tolerance',.0025))
    basepass,availpass,strictpass=_native_flags(raw,inc,selected,qualified,available,variants,baseline,tolerance,inside)
    methods=np.full(selected.shape,'native',dtype='<U40');methods[~availpass]='native-unresolved';methods[:,inside]='analytic-tested-collar'
    treatment=dict(values=raw.copy(),methods=methods,baseline_prefix_pass=basepass,available_prefix_pass=availpass,
        block_prefix_pass=strictpass,records=[],settings=None)
    if 'boundary_treatment' in cfg:
        treatment=routing.apply_mixed_treatment(plans,energies,families,selected,qualified,raw,inc,marks,cfg['boundary_treatment'],
            orders=orders,variants=variants,baseline=baseline,available_orders=available)
    if treatment['values'].shape!=raw.shape or treatment['methods'].shape!=selected.shape:
        raise ValueError('heterogeneous boundary dispatcher changed source tensor shape')
    for f in range(len(families)):
        locations=selected==f
        for v in range(len(variants)):
            for o in np.flatnonzero(~available[f,v]):
                if not np.isnan(treatment['values'][...,v,o][locations]).all():raise ValueError('boundary dispatcher invented an unavailable high prefix')
    combined=_combine(treatment['values'],treatment['methods'],inside,orders,low,high,variants,baseline)
    final=combined['multilevel_native_I6'];routed_full=final.copy();hybrid=final.copy()
    if collar is not None:
        from six_point_estimate_collar_corrected_integrand import hybridize
        routed_full,hybrid=hybridize(final[...,None],marks,collar);routed_full=routed_full[...,0];hybrid=hybrid[...,0]
    arrays=dict(raw_native_I6=raw,treated_prefix_I6=treatment['values'],**combined,
        routed_full_I6=routed_full,hybrid_I6=hybrid,native_relative_absolute_increments=inc,
        native_absolute_increments_I6=absolute,native_normalization_denominator_I6=denominators,
        zero_absolute_weight_node_counts=zero_counts,common_methods=treatment['methods'],
        baseline_prefix_pass=treatment['baseline_prefix_pass'],available_prefix_pass=treatment['available_prefix_pass'],
        block_prefix_pass=treatment['block_prefix_pass'],qualified=qualified,family_indices=selected,decay_ratios=ratios,
        inside_patch=inside,points=np.stack([p.arrays['points'] for p in plans]),q=q,topology=topology,
        ordering=np.stack([p.arrays['ordering'] for p in plans]),incoming_slot=slot,
        source_log_density=np.stack([p.arrays['log_density'] for p in plans]),
        source_pair_logs=np.stack([p.arrays['pair_logs'] for p in plans]),base_logarithms=base,
        original_replicate_ids=original_replicates,original_point_indices=original_indices,
        energies=np.asarray(energies),orders=np.asarray(orders),variants=np.asarray(variants),available_orders=available)
    if collar is not None:
        arrays.update(collar_addback_I6=collar['addback'],collar_addback_lower_x_I6=collar['addback_lower_x'],
            collar_prefix_pass=collar['prefix_pass'],collar_addback_prefix_pass=collar['addback_prefix_pass'],
            **{f'collar_{k}_I6':v for k,v in collar['arrays'].items()})
    for p in full_plans:p.require_sources()
    if named._digest(path)!=config_hash or any(named._digest(p)!=sha for p,sha in {**files,**sources}.items()):
        raise RuntimeError('mixed scan source/input changed during evaluation')
    outside=~inside[None];complete=bool(np.isfinite(hybrid).all())
    summary=dict(version=VERSION,identity=manifest['identity'],execution_scope=scope,geometry_slice=geometry_slice,
        sample_design=cfg['sample_design'],full_source_replicates=full_replicates,full_source_points_per_replicate=full_count,
        evaluated_point_count=inside.size,energies=list(energies),orders=list(orders),low_order=low,high_order=high,
        variants=variants,baseline=baseline,bank_family_names=[f.name for f in families],
        finite_all_evaluated_values=complete,complete_finite_amplitude=complete and scope=='full-source' and full_net,
        unresolved_scope_count=len(requests),fallback_requests=requests,
        baseline_prefix_unresolved_count=int((outside & ~treatment['baseline_prefix_pass']).sum()),
        available_prefix_unresolved_count=int((outside & ~treatment['available_prefix_pass']).sum()),
        missing_direct_high_control_count=int((outside[...,None]&~combined['mixed_defect_assessed']).sum()),
        numerical_failures=failures,bank_diagnostics=receipts,sampling_diagnostics=sampling,
        boundary_records=_original_records(treatment['records'],original_replicates,original_indices),
        collar_records=[] if collar is None else _original_records(collar['records'],original_replicates,original_indices),
        collar_addback_records=[] if collar is None else collar['addback_records'],
        collar_settings=None if collar is None else collar['settings'],
        analytic_patch_count=int(inside.sum()),analytic_addback_once_per_integral=collar is not None,
        all_assigned_original_indices_retained=True,all_original_indices_retained=scope=='full-source',
        primary_estimator='direct baseline high plus separately routed analytic collar',
        control_estimator='control low + baseline(high-low), with the identical separate collar term',
        unavailable_high_slots_remain_nan=True,out_of_scope_policy=cfg.get('out_of_scope_policy','require-fallback'),
        common_method_across_rules_and_levels=True,full_CFT_certified=False,momentum_tail_certified=False,
        global_finite_variance_certified=False,total_accuracy_certified=False,
        statistics=_statistics(hybrid,energies,variants,baseline,full_net=full_net,scope=scope,source_count=full_count*full_replicates),
        seconds=time.perf_counter()-started)
    output.mkdir(parents=True,exist_ok=False)
    with (output/'pointwise.npz').open('xb') as handle:np.savez_compressed(handle,**arrays)
    summary['pointwise_sha256']=named._digest(output/'pointwise.npz')
    for name,payload in [('input_manifest.json',manifest),('summary.json',summary)]:
        _write_json(output/name,payload)
    return summary


_ENERGY_POINT_KEYS={
    'raw_native_I6','treated_prefix_I6','multilevel_native_I6','direct_high_native_I6',
    'mixed_defect_I6','mixed_defect_assessed','baseline_correction_I6','routed_full_I6','hybrid_I6',
    'native_relative_absolute_increments','native_absolute_increments_I6','native_normalization_denominator_I6',
    'zero_absolute_weight_node_counts','common_methods','baseline_prefix_pass','available_prefix_pass',
    'block_prefix_pass','qualified','family_indices','decay_ratios','base_logarithms',
    'collar_prefix_pass','collar_full_I6','collar_leading_I6','collar_remainder_I6','collar_outer1_I6'}
_POINT_KEYS={'inside_patch','points','q','topology','ordering','incoming_slot','source_log_density','source_pair_logs'}
_GLOBAL_KEYS={'energies','orders','variants','available_orders','collar_addback_I6',
              'collar_addback_lower_x_I6','collar_addback_prefix_pass'}


def _same_array(a,b):
    return a.shape==b.shape and a.dtype==b.dtype and np.array_equal(a,b,equal_nan=True) if a.dtype.kind not in 'US' else a.shape==b.shape and a.dtype==b.dtype and np.array_equal(a,b)


def assemble_multilevel(shard_directories,output_directory):
    """Require exact non-overlapping coverage of every original sample index.

    This scatters pointwise contributions, never averages per-shard means.
    Shared addback arrays must agree exactly; the constant is already present
    once in each sample's estimator and is NOT added again during assembly.
    """
    paths=[Path(p).resolve() for p in shard_directories];output=Path(output_directory).resolve()
    if output.exists():raise FileExistsError('preserve previous assembly output')
    if not paths or len(set(paths))!=len(paths):raise ValueError('distinct nonempty shard directories required')
    first=None;assembled={};coverage=None;receipts=[];summaries=[];bank_files={};started=time.perf_counter()
    for path in paths:
        summary_path=path/'summary.json';manifest_path=path/'input_manifest.json';array_path=path/'pointwise.npz'
        summary=json.loads(summary_path.read_text());manifest=json.loads(manifest_path.read_text())
        if (summary.get('version')!=VERSION or summary.get('execution_scope')!='geometry-shard'
                or summary.get('identity')!=manifest.get('identity')
                or named._json_hash({k:v for k,v in manifest.items() if k!='identity'})!=manifest.get('identity')
                or named._digest(array_path)!=summary.get('pointwise_sha256')):
            raise ValueError('shard version, source identity or pointwise checksum mismatch')
        if first is None:
            first=manifest;nr=int(first['full_source_replicates']);n=int(first['full_source_points_per_replicate'])
            bank_files=_validate_bank_references(first)
            coverage=np.zeros((nr,n),bool)
        elif manifest!=first:raise ValueError('shards mix configuration, physical banks, sources or shared addback')
        with np.load(array_path,allow_pickle=False) as saved:data={k:saved[k].copy() for k in saved.files}
        _validate_bank_diagnostics(summary,first,data)
        extras=set(data)-_ENERGY_POINT_KEYS-_POINT_KEYS-_GLOBAL_KEYS-{'original_replicate_ids','original_point_indices'}
        if extras:raise ValueError(f'unknown shard array axes: {sorted(extras)}')
        required={'hybrid_I6','points','q','ordering','topology','source_log_density','source_pair_logs',
                  'original_replicate_ids','original_point_indices','energies','orders','variants'}
        if not required<=data.keys():raise ValueError('shard omitted required physical/source arrays')
        repl=data['original_replicate_ids'];indices=data['original_point_indices']
        if repl.dtype.kind not in 'iu' or repl.ndim!=1 or indices.dtype.kind not in 'iu' or indices.ndim!=2 or indices.shape[0]!=len(repl):
            raise ValueError('invalid original index arrays')
        if (np.any(repl<0) or np.any(repl>=nr) or len(np.unique(repl))!=len(repl)
                or np.any(indices<0) or np.any(indices>=n)):
            raise ValueError('shard indices outside original source')
        for r,ix in zip(repl,indices):
            if len(np.unique(ix))!=len(ix) or coverage[r,ix].any():raise ValueError('duplicate/overlapping source geometry coverage')
        if set(assembled) and set(data)-{'original_replicate_ids','original_point_indices'}!=set(assembled):
            raise ValueError('shards have different saved diagnostic schemas')
        for key,array in data.items():
            if key in ('original_replicate_ids','original_point_indices'):continue
            if key in _GLOBAL_KEYS:
                if key in assembled and not _same_array(assembled[key],array):raise ValueError(f'global shard array changed: {key}')
                assembled.setdefault(key,array.copy());continue
            energy_axis=key in _ENERGY_POINT_KEYS;offset=1 if energy_axis else 0
            if array.shape[offset:offset+2]!=indices.shape:raise ValueError(f'shard sample axes differ: {key}')
            expected_shape=((*array.shape[:offset],nr,n,*array.shape[offset+2:]))
            if key not in assembled:assembled[key]=np.empty(expected_shape,dtype=array.dtype)
            if assembled[key].shape!=expected_shape or assembled[key].dtype!=array.dtype:
                raise ValueError(f'shard tensor shape/dtype changed: {key}')
            for local,r in enumerate(repl):
                if energy_axis:assembled[key][:,r,indices[local]]=array[:,local]
                else:assembled[key][r,indices[local]]=array[local]
        for r,ix in zip(repl,indices):coverage[r,ix]=True
        receipts.append(dict(directory=str(path),summary_sha256=named._digest(summary_path),
            manifest_sha256=named._digest(manifest_path),pointwise_sha256=summary['pointwise_sha256'],
            original_replicates=repl.tolist(),original_point_indices=indices.tolist()))
        summaries.append(summary)
    if not coverage.all():raise ValueError(f'incomplete original geometry coverage: {int((~coverage).sum())} points missing')
    names={'points':'points','q':'q','topology':'topology','ordering':'ordering',
           'log_density':'source_log_density','pair_logs':'source_pair_logs'}
    for record in first['source_geometry']:
        r=record['replicate']
        if point_digest(assembled['points'][r])!=record['point_sha256']:
            raise ValueError('assembled points are not the original complete Sobol/source inventory')
        for source,destination in names.items():
            if _hash_array(assembled[destination][r])!=record['output_array_hashes'][source]:
                raise ValueError(f'assembled original source geometry/measure mismatch: {source}')
    for file,digest in {**first['files'],**first['sources']}.items():
        if named._digest(file)!=digest:raise RuntimeError('frozen scan input/source changed before assembly')
    cfg=first['config'];energies=assembled['energies'].tolist();variants=assembled['variants'].tolist();inside=assembled['inside_patch']
    for key in ('energies','orders','variants'):
        if assembled[key].tolist()!=cfg[key]:raise ValueError(f'assembled {key} axis differs from frozen configuration')
    family_names=[f['name'] for f in cfg['families']]
    expected_available=np.asarray([[[o<=first['banks'][name][v]['order'] for o in cfg['orders']]
        for v in variants] for name in family_names],bool)
    if not np.array_equal(assembled['available_orders'],expected_available):
        raise ValueError('assembled available-order mask differs from physical bank metadata')
    expected=(len(energies),nr,n,len(variants),len(cfg['orders']))
    if assembled['raw_native_I6'].shape!=expected or assembled['treated_prefix_I6'].shape!=expected:
        raise ValueError('assembled prefix tensor is inconsistent with physical axes')
    for f in range(len(family_names)):
        locations=assembled['family_indices']==f
        for v in range(len(variants)):
            for o in np.flatnonzero(~expected_available[f,v]):
                for key in ('raw_native_I6','treated_prefix_I6'):
                    if not np.isnan(assembled[key][...,v,o][locations]).all():
                        raise ValueError('assembled unavailable high prefix has been filled')
    replay=_combine(assembled['treated_prefix_I6'],assembled['common_methods'],inside,cfg['orders'],
        cfg.get('low_order',10),cfg.get('high_order',12),variants,cfg['baseline'])
    for key,value in replay.items():
        if not _same_array(assembled[key],value):raise ValueError(f'assembled nonchiral two-level algebra mismatch: {key}')
    expected_routed=assembled['multilevel_native_I6'];expected_hybrid=expected_routed
    if 'collar_addback_I6' in assembled:
        mask=inside[None,:,:,None]
        expected_routed=np.where(mask,assembled['collar_full_I6'][:,:,:,-1,-1,None],expected_routed)
        expected_hybrid=np.where(mask,assembled['collar_remainder_I6'][:,:,:,-1,-1,None],expected_hybrid)
        expected_hybrid=expected_hybrid+assembled['collar_addback_I6'][:,-1,-1,None,None,None]
        if any(s.get('collar_addback_records')!=summaries[0].get('collar_addback_records') for s in summaries):
            raise ValueError('shared addback provenance differs across shards')
    if not _same_array(assembled['routed_full_I6'],expected_routed) or not _same_array(assembled['hybrid_I6'],expected_hybrid):
        raise ValueError('assembled collar insertion/addback accounting mismatch')
    full_net=named.is_full_integral_design(cfg['sample_design']);finite=bool(np.isfinite(assembled['hybrid_I6']).all())
    outside=~inside[None]
    report=dict(version=VERSION,identity=first['identity'],execution_scope='full-source',assembled_from_shards=True,
        sample_design=cfg['sample_design'],full_source_replicates=nr,full_source_points_per_replicate=n,
        evaluated_point_count=nr*n,energies=energies,orders=assembled['orders'].tolist(),variants=variants,
        baseline=cfg['baseline'],low_order=cfg.get('low_order',10),high_order=cfg.get('high_order',12),
        finite_all_evaluated_values=finite,complete_finite_amplitude=finite and full_net,
        unresolved_scope_count=sum(s['unresolved_scope_count'] for s in summaries),
        fallback_requests=[r for s in summaries for r in s['fallback_requests']],
        baseline_prefix_unresolved_count=int((outside&~assembled['baseline_prefix_pass']).sum()),
        available_prefix_unresolved_count=int((outside&~assembled['available_prefix_pass']).sum()),
        missing_direct_high_control_count=int((outside[...,None]&~assembled['mixed_defect_assessed']).sum()),
        numerical_failures=[r for s in summaries for r in s['numerical_failures']],
        boundary_records=[r for s in summaries for r in s['boundary_records']],
        collar_records=[r for s in summaries for r in s['collar_records']],
        collar_addback_records=summaries[0].get('collar_addback_records',[]),
        collar_settings=summaries[0].get('collar_settings'),
        shard_bank_diagnostics=[s['bank_diagnostics'] for s in summaries],
        sampling_diagnostics=summaries[0]['sampling_diagnostics'],
        analytic_patch_count=int(inside.sum()),analytic_addback_once_per_integral='collar_addback_I6' in assembled,
        addback_readded_during_assembly=False,exact_original_index_coverage=True,
        all_original_source_geometry_and_density_hashes_match=True,shard_receipts=receipts,
        unavailable_high_slots_remain_nan=True,primary_estimator='direct baseline high plus separately routed analytic collar',
        control_estimator='control low + baseline(high-low), with the identical separate collar term',
        full_CFT_certified=False,momentum_tail_certified=False,global_finite_variance_certified=False,total_accuracy_certified=False,
        statistics=_statistics(assembled['hybrid_I6'],energies,variants,cfg['baseline'],full_net=full_net,
            scope='full-source',source_count=nr*n),seconds=time.perf_counter()-started)
    assembled['original_replicate_ids']=np.arange(nr,dtype=np.int64)
    assembled['original_point_indices']=np.broadcast_to(np.arange(n,dtype=np.int64),(nr,n)).copy()
    output.mkdir(parents=True,exist_ok=False)
    with (output/'pointwise.npz').open('xb') as handle:np.savez_compressed(handle,**assembled)
    report['pointwise_sha256']=named._digest(output/'pointwise.npz')
    if any(named._digest(path)!=digest for path,digest in bank_files.items()):
        raise RuntimeError('referenced bank evidence changed during assembly')
    for name,payload in [('input_manifest.json',first),('summary.json',report)]:
        _write_json(output/name,payload)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='command',required=True)
    scan=commands.add_parser('scan');scan.add_argument('config');scan.add_argument('output')
    scan.add_argument('--replicate',type=int);scan.add_argument('--start',type=int);scan.add_argument('--stop',type=int)
    scan.add_argument('--shared-addback')
    audit=commands.add_parser('audit-geometry');audit.add_argument('config');audit.add_argument('output')
    assembly=commands.add_parser('assemble');assembly.add_argument('output');assembly.add_argument('shards',nargs='+')
    args=parser.parse_args()
    if args.command=='audit-geometry':result=audit_geometry(args.config,args.output)
    elif args.command=='assemble':result=assemble_multilevel(args.shards,args.output)
    else:
        values=(args.replicate,args.start,args.stop)
        if any(v is not None for v in values) and not all(v is not None for v in values):parser.error('all three geometry slice arguments required')
        sl=None if values[0] is None else dict(zip(('replicate','start','stop'),values))
        result=run_multilevel(args.config,args.output,geometry_slice=sl,shared_addback=args.shared_addback)
    print(json.dumps({k:result[k] for k in ('version','execution_scope','complete_finite_amplitude','fallback_request_count') if k in result},indent=2))


if __name__=='__main__':main()
