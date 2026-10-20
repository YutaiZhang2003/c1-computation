"""Recover the draft's numerical curves from the bundled lightweight inputs.

Run with --output NEW_DIRECTORY. All inputs come from --data-dir; historical
absolute paths in frozen provenance are never used as runtime dependencies.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import zipfile
import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[3]


def read_json(path):
    value = json.loads(Path(path).read_text())
    if isinstance(value, dict) and 'payload' in value and 'sha256' in value:
        actual = hashlib.sha256(json.dumps(value['payload'], sort_keys=True,
            separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        if actual != value['sha256']:
            raise ValueError('Changed checked record: '+str(path))
        return value['payload']
    return value


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024), b''):h.update(chunk)
    return h.hexdigest()


def verify_data(root):
    manifest=read_json(root/'DATASET_MANIFEST.json')
    if (manifest.get('data_stage')!='after Liouville momentum integration'
            or manifest.get('coefficient_banks_included') is not False):
        raise ValueError('The dataset must contain values after Liouville momentum integration')
    for row in manifest['files']:
        p=root/row['path']
        if not p.is_file() or p.stat().st_size!=row['bytes'] or sha256(p)!=row['sha256']:
            raise ValueError('Dataset checksum failed: '+row['path'])
    actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    if actual != {r['path'] for r in manifest['files']} | {'DATASET_MANIFEST.json'}:
        raise ValueError('Dataset inventory differs from its manifest')
    for name in actual:
        if {'banks','native_coefficients','liouville'}.intersection(Path(name).parts):
            raise ValueError('Pre-integration data are outside the dataset scope: '+name)
    return {'files':len(manifest['files']), 'bytes':sum(r['bytes'] for r in manifest['files']),
            'data_stage':manifest['data_stage'],'coefficient_banks_included':False}


def close(actual,expected,rtol=2e-11,atol=1e-20):
    np.testing.assert_allclose(actual,expected,rtol=rtol,atol=atol)

def table(path,header,rows,delimiter=','):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        writer=csv.writer(f,delimiter=delimiter,lineterminator='\n')
        writer.writerow(header)
        for row in rows:writer.writerow([format(float(x),'.17g') for x in row])

def covariance_of_mean(replicates):
    replicates=np.asarray(replicates)
    assert replicates.ndim==2 and len(replicates)>=2 and np.isfinite(replicates).all()
    return np.cov(replicates,rowvar=False,ddof=1)/len(replicates)

def component_values(record):return {k:v['value']['value'][0] for k,v in record['components'].items()}

def assemble_record(record,replicas=None):
    comps=component_values(record)
    if replicas is None:replicas=record['compact_replicate_means']
    assert len(replicas)==16
    compact=math.fsum(replicas)/16;close(compact,record['compact']['mean'])
    close(np.std(replicas,ddof=1)/4,record['compact']['sampling_se'])
    if 'cells' in record:
        assert len(record['cells'])==74 and len({c['cell']['id'] for c in record['cells']})==74
        for term,value in comps.items():
            close(math.fsum(c['high']['value'][0] for c in record['cells'] if c['cell']['term']==term),value)
    boundary=math.fsum(comps.values());close(boundary,record['boundary']['value'][0])
    amplitude=boundary+compact;close(amplitude,record['amplitude'])
    return amplitude,comps,np.array(replicas)

def refine_primary(base,refinement,baseline):
    assert baseline['amplitude']['amplitude']==base['amplitude']
    before=component_values(base);updated=dict(before)
    old=baseline['primary_cells'];new=refinement['primary_cells']
    assert len(new)==len(old)==48
    assert {c['cell']['id'] for c in old}=={c['cell']['id'] for c in new}
    terms={c['cell']['term'] for c in new}
    for term in terms:
        close(math.fsum(c['high']['value'][0] for c in old if c['cell']['term']==term),before[term])
        updated[term]=math.fsum(c['high']['value'][0] for c in new if c['cell']['term']==term)
    for term,v in updated.items():close(v,component_values(refinement)[term])
    compact=math.fsum(base['compact_replicate_means'])/16
    result=math.fsum(updated.values())+compact;close(result,refinement['amplitude'])
    return result,updated

def load_moduli_samples(folder):
    """Read compact geometry without reordering the original summation terms."""
    design = json.loads((folder/'design.json').read_text())
    if design['format'] != 'ordered-moduli-weights-v1':
        raise ValueError('Unknown moduli weight format')
    rules = {}
    coverage = set()
    with np.load(folder/'weights.npz', allow_pickle=False) as arrays:
        for entry in design['rules']:
            key = (entry['cell_id'], entry['rule'])
            if key in rules or key[0] not in design['terms']:
                raise ValueError('Invalid or duplicate quadrature rule')
            indices = arrays['indices_'+entry['array']]
            weights = arrays['weights_'+entry['array']]
            if (indices.shape != weights.shape or indices.ndim != 1
                    or not np.issubdtype(indices.dtype, np.integer)
                    or np.any(indices < 0) or np.any(indices >= design['sample_count'])
                    or not np.isfinite(weights).all() or np.any(weights <= 0)):
                raise ValueError('Invalid moduli weights')
            rules[key] = (indices, weights)
            coverage.update(indices.tolist())
    if coverage != set(range(design['sample_count'])):
        raise ValueError('Incomplete moduli sample coverage')
    return {'n': design['sample_count'], 'terms': design['terms'], 'rules': rules}


def reconstruct_one_loop_samples(root):
    root=Path(root)
    def ref(rel):
        d=json.loads((root/rel).read_text());return d.get('payload',d)
    samples=root/'inputs/moduli_samples'
    def array(name):
        with np.load(samples/name,allow_pickle=False) as a:return {k:a[k] for k in a.files}
    def geometry(group):return load_moduli_samples(samples/group)
    def reduce(g,a):
        values=a['momentum_integrated_contribution'];assert values.shape==(g['n'],) and np.isfinite(values).all()
        return {k:math.fsum((values[i].real*w).tolist()) for k,(i,w) in g['rules'].items()}
    main=geometry('main');ts=[.025]+[i/20 for i in range(1,20)]
    amplitudes={};cells={};compact={};compact_replicates={}
    for t in ts:
        tag=f't{round(t*1000):04d}';a=array('main/values/'+tag+'.npz');rules=reduce(main,a)
        cells[t]={cid:v for (cid,rule),v in rules.items() if rule=='high'};assert len(cells[t])==74
        cv=a['compact_momentum_integrated_density']*a['compact_moduli_weight']
        assert cv.shape==(16,4) and np.isfinite(cv).all()
        compact_replicates[t]=cv.real.mean(axis=1);compact[t]=math.fsum(compact_replicates[t])/16
        value=math.fsum(cells[t].values())+compact[t]
        r=ref('inputs/reference_20/'+('anchor.json' if t==.25 else tag+'.json'))
        if t==.25:r=r['amplitude']
        close(value,r['amplitude']);close(compact_replicates[t],r['compact_replicate_means'])
        for term,component in r['components'].items():
            close(math.fsum(v for cid,v in cells[t].items() if main['terms'][cid]==term),component['value']['value'][0])
        if 'cells' in r:
            for c in r['cells']:close(cells[t][c['cell']['id']],c['high']['value'][0])
        amplitudes[t]=value
    refined={};primary_counts={}
    for t,tag in [(.25,'t0250'),(.95,'t0950')]:
        group='primary_'+tag;g=geometry(group);a=array(group+'/values/'+tag+'.npz');rules=reduce(g,a)
        candidate={cid:v for (cid,rule),v in rules.items() if rule=='candidate'}
        assert len(candidate)==12
        updated={**cells[t],**candidate};r=ref('inputs/refinements/'+tag+'_primary_amplitude.json')
        for c in r['primary_cells']:close(updated[c['cell']['id']],c['high']['value'][0])
        value=math.fsum(updated.values())+compact[t];close(value,r['amplitude'])
        primary_counts[tag]=g['n']
        if t==.25:
            ug=geometry('upper_t0250');ua=array('upper_t0250/values/t0250.npz');ur=reduce(ug,ua)
            high={cid:v for (cid,rule),v in ur.items() if rule=='high'};assert len(high)==1
            updated.update(high);value=math.fsum(updated.values())+compact[t]
            upper=ref('inputs/refinements/t0250_upper_amplitude.json');close(value,upper['amplitude'])
            close(next(iter(high.values())),upper['selected_cell']['high']['value'][0])
        refined[t]=value
    rms_geometry=geometry('rms');rms={}
    for t in ts:
        tag=f't{round(t*1000):04d}';a=array('rms/values/'+tag+'.npz');rules=reduce(rms_geometry,a)
        reference=ref('inputs/rms/'+tag+'.json')
        for c in reference['cells']:
            for rule,v in c['rules'].items():
                if (c['cell']['id'],rule) in rules:close(rules[c['cell']['id'],rule],v)
        shifts=[]
        for primary,tail in [('base','base'),('head_h2','axis3_h2'),('head_h4','axis3_h4'),('head_check','axis3_check')]:
            shifts.append(math.fsum(rules[cid,primary if term.startswith('primary:') else tail]-rules[cid,'base'] for cid,term in rms_geometry['terms'].items()))
        value=math.sqrt(math.fsum(x*x for x in shifts)/4);close(value,reference['statistics']['rms_shift']);rms[t]=value
    return {'amplitudes':amplitudes,'refined':refined,'rms':rms,'compact_replicates':compact_replicates,
        'report':{'data_stage':'after Liouville momentum integration, before numerical moduli integration',
            'base_moduli_samples_per_energy':main['n'],'base_energies':20,
            'primary_refinement_samples':primary_counts,'upper_refinement_samples':ug['n'],
            'rms_samples_per_energy':rms_geometry['n'],'compact_moduli_samples_per_energy':64,
            'weights_reapplied_to_sample_values':True,'all_current_amplitudes_and_rms_match':True}}

def recover_one_loop(root, output):
    read=lambda relative:read_json(root/relative)
    integrated=reconstruct_one_loop_samples(root)
    base={};means={};amps={}
    for path in sorted((root/'inputs/reference_20').glob('*.json')):
        record=read(str(path.relative_to(root)))
        if path.stem=='anchor':record=dict(record['amplitude'],t=record['t'])
        value,components,replicas=assemble_record(record)
        t=record['t'];base[t]=record
        close(value,integrated['amplitudes'][t]);amps[t]=integrated['amplitudes'][t]
        close(replicas,integrated['compact_replicates'][t]);means[t]=integrated['compact_replicates'][t]
    assert len(base)==20
    covariance_reference=read('reference/compact_covariance.json')
    ts=covariance_reference['t_values'];rep=np.array([means[t] for t in ts]).T
    covariance=covariance_of_mean(rep)
    close(rep,covariance_reference['replicate_means'])
    close(covariance,covariance_reference['covariance_of_compact_means'],atol=1e-28)
    refined={}
    for t,tag in [(.25,'t0250'),(.95,'t0950')]:
        r=read('inputs/refinements/'+tag+'_primary_amplitude.json')
        b=read('inputs/refinements/'+tag+'_primary_baseline.json')
        value,comps=refine_primary(base[t],r,b)
        if t==.25:
            upper=read('inputs/refinements/t0250_upper_amplitude.json');selected=upper['selected_cell']
            high=math.fsum(selected['by_panel']['high'].values());close(high,selected['high']['value'][0])
            term=selected['cell']['term'];comps[term]+=high-selected['old_value']['value'][0]
            for k,v in comps.items():close(v,component_values(upper)[k])
            value=math.fsum(comps.values())+math.fsum(means[t])/16;close(value,upper['amplitude'])
        close(value,integrated['refined'][t]);refined[t]=integrated['refined'][t]
    rows=[];rms_values={}
    # Reconstruct current grid RMS from the 26 saved cell rule integrals per energy.
    for t in ts:
        tag='t'+format(round(t*1000),'04d');r=read('inputs/rms/'+tag+'.json')
        cells=r['cells'];assert len(cells)==26
        shifts=[]
        for primary,tail in [('base','base'),('head_h2','axis3_h2'),('head_h4','axis3_h4'),('head_check','axis3_check')]:
            shifts.append(math.fsum(c['rules'][primary if c['cell']['term'].startswith('primary:') else tail]-c['rules']['base'] for c in cells))
        stats=r['statistics'];close(shifts,[x['shift'] for x in stats['variants']])
        rms=math.sqrt(math.fsum(x*x for x in shifts)/4)
        close(rms,stats['rms_shift']);close(np.std(shifts,ddof=1),stats['grid_standard_deviation'])
        close(rms,integrated['rms'][t]);rms_values[t]=integrated['rms'][t];rows.append([t,amps[t],rms_values[t]])
    rows.extend([t,a,rms_values[t]] for t,a in refined.items());rows.sort(key=lambda row:(row[0],row[1]))
    expected=np.loadtxt(root/'reference/torus_1to2_numeric_rms.dat',skiprows=1)
    close(rows,expected,rtol=2e-12);assert len(rows)==22
    table(output/'torus_1to2_numeric_rms.dat',['t','amplitude','grid_rms'],rows,delimiter=' ')
    scale=(2*np.pi)**3
    table(output/'torus_1to2_gs3.csv',['t','Re_A_coefficient_gs3','grid_RMS_coefficient_gs3'],
          [[t,a*scale,r*scale] for t,a,r in rows])
    np.savez_compressed(output/'compact_covariance.npz',t=ts,native=covariance,gs3=covariance*scale**2,replicate_means=rep)
    return {'momentum_integrated_sample_replay':integrated['report'],'plotted_evaluations':22,'distinct_energies':20,'complete_cell_reductions_checked':19,
        'anchor_component_reduction_checked':True,'later_refinements_reconstructed':2,
        'compact_scrambles_per_energy':16,'compact_covariance_checked':True,'grid_RMS_reconstructed_from_cell_rules':True,
        'maximum_native_amplitude_difference_from_figure':float(np.max(np.abs(np.array(rows)[:,1]-expected[:,1]))),
        'all_current_figure_values_match':True,
        'scope':'Moduli integration repeated from the saved momentum-integrated sample values and original moduli weights. Grid RMS is a sensitivity measure, not a total physical error bound.'}

def residual(nodes,kernels):
    full=kernels['full_scaled_Xi'];control=kernels['control_scaled_Xi']
    assert full.shape==control.shape==(len(nodes),71)
    assert np.isfinite(full).all() and np.isfinite(control).all()
    difference=np.where(np.array([n['in_domain'] for n in nodes])[:,None],full-control,0.)
    log_weight=np.array([n['log_weight_over_rho_squared'] for n in nodes])
    with np.errstate(divide='ignore',invalid='raise',over='raise'):
        return np.sign(difference)*np.exp(np.log(np.abs(difference))+log_weight[:,None])/8

def group_by_band(nodes,values,policy):
    result={};nominal=2**policy['power']
    for band in range(len(policy['band_edges'])-1):
        rows=[]
        for rep in range(policy['replicates']):
            ids=[i for i,n in enumerate(nodes) if n['band']==band and n['replicate']==rep]
            assert len(ids)==nominal and {nodes[i]['index'] for i in ids}==set(range(nominal))
            rows.append(values[ids].sum(axis=0)/nominal)
        result[band]=np.array(rows)
    assert sum(len(v)*nominal for v in result.values())==len(nodes)
    return result

def recover_genus2(root, output):
    read=lambda relative:read_json(root/relative)
    with np.load(root/'inputs/bulk_control_groups.npz',allow_pickle=False) as a:
        radii=a['radii'];expected_bulk=a['old_base'];groups={
            'integrated_control':a['integrated_primary']/8,'negative_domain_correction':-a['domain_correction']/8}
    with np.load(root/'inputs/momentum_integrated/bulk_values.npz',allow_pickle=False) as data:
        close(data['radii'],radii)
        v=data['worldsheet_density']-data['primary_control_density'];w=data['physical_moduli_weight'];rep=data['replicate']
        bulk=np.array([np.sum(v[rep==i]*w[rep==i,None],axis=0) for i in range(8)])
        close(bulk,expected_bulk,rtol=2e-12,atol=2e-13)
        bulk_count=len(v)
    groups={'old_bulk_plus_outer_residual':bulk/8,**groups}
    nodes=read('inputs/finite_nodes.json');policy=read('inputs/finite_policy.json')
    with np.load(root/'inputs/finite_kernels.npz',allow_pickle=False) as kernels:
        close(kernels['radii'],radii);values=residual(nodes,kernels)
    for band,reps in group_by_band(nodes,values,policy).items():groups['residual_band_'+str(band)]=reps
    deep=read('inputs/deep_nodes.json');deep_policy=read('inputs/deep_policy.json')
    with np.load(root/'inputs/deep_kernels.npz',allow_pickle=False) as kernels:
        close(kernels['radii'],radii);deep_values=residual(deep,kernels)
    deep_groups=group_by_band(deep,deep_values,deep_policy);assert set(deep_groups)=={0}
    groups['residual_band_1e5_1e3']=deep_groups[0];assert len(groups)==9
    mean=sum(v.mean(axis=0) for v in groups.values())
    covariance=sum(covariance_of_mean(v) for v in groups.values())
    with np.load(root/'reference/expected_arrays.npz',allow_pickle=False) as expected:
        for name,v in groups.items():close(v,expected['group_'+name],rtol=2e-10,atol=1e-15)
        close(mean,expected['resolved_mean'],rtol=2e-11,atol=1e-15)
        close(covariance,expected['resolved_covariance'],rtol=2e-10,atol=1e-18)
    # Historical figure input stores the coefficient of mu^-2 in log Z.
    mu_coefficient=mean/(2*np.pi)**2;mu_cov=covariance/(2*np.pi)**4
    current=np.genfromtxt(root/'reference/radius_values.csv',delimiter=',',names=True)
    close(radii,current['R']);close(mu_coefficient,current['resolved_F2'])
    close(np.sqrt(np.diag(mu_cov)),current['empirical_MC_SE'])
    # The CURRENT draft figure converts log Z into thermal vacuum free energy.
    conversion=-1/(2*np.pi*radii)
    F2=mean*conversion;F2_cov=covariance*conversion[:,None]*conversion[None,:]
    F2_se=np.sqrt(np.diag(F2_cov));model=-np.pi*(7+10/radii**2+7/radii**4)/2880
    close(F2,-2*np.pi/radii*current['resolved_F2'])
    close(model,-2*np.pi/radii*current['matrix_model_F2'])
    table(output/'radius_values.csv',['R','resolved_F2','empirical_MC_SE','matrix_model_F2'],
          zip(radii,mu_coefficient,np.sqrt(np.diag(mu_cov)),current['matrix_model_F2']))
    table(output/'genus2_free_energy_F2.csv',['R','F2_coefficient_gs2','empirical_MC_SE','matrix_model_F2'],zip(radii,F2,F2_se,model))
    # Same fixed diagonal-weight linear estimator, with full covariance propagation.
    x=np.column_stack((radii+radii**-3,radii**-1));weights=1/np.diag(covariance)
    linear=np.linalg.solve(x.T@(weights[:,None]*x),x.T*weights)
    multiplier=1440/np.pi**2
    bracket=(linear@mean)*multiplier
    bracket_cov=(linear@covariance@linear.T)*multiplier**2
    bracket_se=np.sqrt(np.maximum(np.diag(bracket_cov),0))
    fit=read('reference/expected_fit.json')
    close(bracket,fit['bracket_coefficients_with_prefactor_1_over_5760'])
    close(bracket_se,fit['bracket_coefficient_empirical_SE'])
    np.savez_compressed(output/'genus2_covariance_and_groups.npz',radii=radii,F2=F2,covariance_F2=F2_cov,
        logZ_gs2_mean=mean,logZ_gs2_covariance=covariance,fit_bracket=bracket,fit_bracket_covariance=bracket_cov,
        **{'group_'+k:v for k,v in groups.items()})
    one=int(np.flatnonzero(radii==1)[0]);scope=read('inputs/production_scope.json')
    return {'radii':len(radii),'momentum_integrated_bulk_samples':bulk_count,'bulk_moduli_weights_reapplied':True,'accepted_finite_nodes':len(nodes),'accepted_deep_nodes':len(deep),
        'independent_groups':len(groups),'full_cross_radius_covariance_checked':True,
        'F2_R1':float(F2[one]),'F2_R1_empirical_MC_SE':float(F2_se[one]),
        'fit_bracket_coefficients':bracket.tolist(),'fit_bracket_empirical_SE':bracket_se.tolist(),
        'draft_table_reproduced':'(6.99 +/- 0.02)(1+R^-4) + (10.00 +/- 0.13) R^-2, prefactor -pi/2880',
        'all_current_figure_values_match':True,'remaining_residual_region':scope['remaining_residual_region'],
        'full_free_energy_available':scope['full_free_energy_available'],
        'scope':'Current resolved vacuum free-energy contribution. The unresolved deep residual and total systematics remain unbounded. Thermal S-matrix data are excluded.'}


def recover_five_point(root, output):
    data=read_json(root/'inputs/points.json')
    audits={p['t']:p for p in data['current_uncertainty_records']}
    rows=[];replica_count=0
    for p in data['points']:
        t=p['t'];q=p['Q']
        if 'replicate_estimates' in p:
            rep=np.array([v['imag'] for v in p['replicate_estimates']])
            close(rep.mean()/(16*np.pi**2*t**5),q)
            close(rep.std(ddof=1)/np.sqrt(len(rep))/(16*np.pi**2*t**5),p['Q_standard_error'])
            replica_count+=1
        errors=[p['Q_standard_error']]
        if 'Q_block_order_4_6_8' in p:errors.append(max(abs(v-q) for v in p['Q_block_order_4_6_8']))
        if t in audits:errors.append(audits[t]['conservative_standard_error'])
        rows.append([t,-4*t**5*q,4*t**5*max(errors)])
    assert len(rows)==29
    for name,selected in [('sphere_1to4_t_no_residue.dat',[r for r in rows if r[0]<=.4]),
                          ('sphere_1to4_t_residue.dat',[r for r in rows if r[0]>.4])]:
        close(selected,np.loadtxt(root/'reference'/name,skiprows=1),rtol=1e-14)
        table(output/name,['t','amplitude_real','amplitude_real_error'],selected,delimiter=' ')
    table(output/'sphere_1to4_gs3.csv',['t','Re_A_coefficient_gs3','saved_numerical_error'],
          [[t,a*(2*np.pi)**3,se*(2*np.pi)**3] for t,a,se in rows])
    table(output/'sphere_1to4_thermal_R.csv',['R','t','amplitude_gs_coefficient'],
          sorted([[1/t,t,a*(2*np.pi)**3] for t,a,se in rows]))
    return dict(points=29,integral_replicate_reductions=replica_count,
        all_current_figure_values_match=True,
        data_form='29 saved integral means and 15 sets of replicate integrals')


def recover_six_point(root, output):
    expected=read_json(root/'reference/curve_results.json')
    check=dict(expected);saved=check.pop('fingerprint')
    assert hashlib.sha256(json.dumps(check,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()==saved
    assert expected['general_six_point_nome_reexpansion'] is False
    assert expected['four_point_separating_collar_retained'] is True
    rows=[];replicas=[];restored=0
    for e in expected['results']:
        with np.load(root/'samples'/(e['label']+'.npz'),allow_pickle=False) as a:
            value=a['direct_C12_I6_without_addback']+a['preserved_native_refinement_I6']+a['analytic_four_point_addback_I6']
            close(value,a['weighted_I6'],rtol=2e-12,atol=1e-16)
            assert value.shape==(8,256) and np.isfinite(value).all()
            assert not a['inside_analytic_patch'].any()
            assert a['q'].shape==(8,256,3) and a['family_index'].shape==(8,256)
            rep=value.mean(axis=1)*1j/(8*np.pi**3)
            close(np.column_stack((rep.real,rep.imag)),e['mu4_A_replicates'],atol=1e-35)
            mean=rep.mean();se=np.array([rep.real.std(ddof=1),rep.imag.std(ddof=1)])/np.sqrt(8)
            close([mean.real,mean.imag],e['mu4_A']['mean'],atol=1e-35)
            close(se,e['mu4_A']['sampling_se'],atol=1e-35)
            count=int(a['restored_to_direct_C12'].sum());assert count==e['restored_sample_count'];restored+=count
        rows.append([e['t'],mean.imag,se[1]]);replicas.append(rep)
    assert len(rows)==15 and restored==4950
    rep=np.array(replicas)
    covariance=np.cov(np.concatenate((rep.real,rep.imag),axis=0),ddof=1)/8
    close(covariance,expected['mean_covariance_mu4_A_real_then_imag'],atol=1e-40)
    embedded=np.loadtxt(root/'reference/sphere_1to5_native.dat',skiprows=1)
    close(rows,embedded,rtol=1e-13)
    table(output/'sphere_1to5_native.dat',['t','ImA','SE'],rows,delimiter=' ')
    table(output/'sphere_1to5_gs4.csv',['t','Im_A_coefficient_gs4','sampling_SE'],
          [[t,a*(2*np.pi)**4,se*(2*np.pi)**4] for t,a,se in rows])
    table(output/'sphere_1to5_thermal_R.csv',['R','t','amplitude_gs_coefficient'],
          sorted([[1/t,t,a*(2*np.pi)**4] for t,a,se in rows]))
    np.savez_compressed(output/'sphere_1to5_covariance.npz',t=np.array(rows)[:,0],
        replicate_mu4_A=rep,native_real_then_imag=covariance,gs4_real_then_imag=covariance*(2*np.pi)**8)
    return dict(points=15,saved_sample_energy_values=30720,common_scrambles=8,
        native_restored_values=restored,direct_series_only=True,four_point_separating_addback=True,
        all_current_figure_values_match=True,full_sampling_covariance_checked=True)


def recover_thermal_two_point(root, output):
    from genus1_two_point_thermal_report import assemble
    from recover_native_q_results import recover
    rows=assemble(root/'scan_complete_collision',output)
    expected=list(csv.DictReader((root/'reference/one_loop_1to1_R.csv').open()))
    replicas=[]
    for row,old in zip(rows,expected):
        for k,v in row.items():
            if isinstance(v,(int,float)):close(v,float(old[k]))
        saved=read_json(root/'scan_complete_collision'/f"R{row['R']:g}.json")
        a=np.asarray(saved['replicate_integrals']);assert a.shape==(8,2,2)
        values=a[:,:,0]*2*np.pi**2
        close(values.mean(axis=0),[saved['zero_temperature_imag'],saved['thermal_imag']])
        close(values.std(axis=0,ddof=1)/np.sqrt(8),[saved['sampling_SE_zero'],saved['sampling_SE_thermal']])
        close(np.std(values[:,1]-values[:,0],ddof=1)/np.sqrt(8),saved['sampling_SE_correction'])
        replicas.append(values[:,1])
    assert len(rows)==len(expected)==10
    np.savez_compressed(output/'thermal_two_point_covariance.npz',R=[r['R'] for r in rows],
        replicate_amplitudes=np.array(replicas),covariance=np.cov(replicas,ddof=1)/8)
    native_q = recover(root, output/'native_q')
    return dict(radii=10,replicates_per_radius=8,all_current_figure_values_match=True,
        native_q=native_q,
        data_form='Saved replicate integrals after momentum and moduli integration')


def recover_thermal_three_point(root, output):
    import draft_radius_collect as collect
    old_output=collect.OUTPUT;collect.OUTPUT=root
    try:
        expected=read_json(root/'reference/thermal_draft_results.json')
        manifest=read_json(root.parents[1]/'DATASET_MANIFEST.json')
        original_hashes={row['path']:row.get('original_sha256',row['sha256']) for row in manifest['files']}
        baseline=read_json(root/'baseline.json');source=read_json(root/'compact_source.json')
        assert sha256(root/'reference/torus_1to2_numeric_rms.dat')==baseline['sha256']
        records=[];rows=[]
        for b in baseline['selected']:
            if not b['plotted']:continue
            tag=collect.tag_of(b['t'])
            comps={name:collect.component(tag,name,power) for name,power in
                   [('primary',8),('pair-bulk',10),('pair-tail',10)]}
            compact=collect.compact(b,source);collision=read_json(root/'collision'/f'{tag}.json')
            delta=float(collision['high'][2]+compact['mean'][2]+sum(v['statistics']['mean'][2] for v in comps.values()))
            cov=np.array(compact['covariance_of_mean'])
            for c in comps.values():cov+=c['statistics']['covariance_of_mean']
            actual=b['baseline_gs3']+delta
            old=next(r for r in expected['rows'] if r['tag']==tag)
            close(actual,old['worldsheet_thermal_gs3']);close(delta,old['thermal_correction'])
            se=float(math.sqrt(cov[2,2]));close(se,old['sampling_SE_correction'])
            for name,c in comps.items():
                saved=old['components'][name]
                original_pins={name:original_hashes['one_loop/three_point_thermal/'+name]
                               for name in c['sample_sha256']}
                assert original_pins==saved['sample_sha256']
                close(c['statistics']['replicates'],saved['statistics']['replicates'])
                close(c['paired_doubling_shift']['mean'],saved['paired_doubling_shift']['mean'])
            records.append(dict(components=comps,compact=compact))
            rows.append([b['R'],b['t'],b['baseline_gs3'],delta,actual,se,old['matrix_model']['thermal']])
        assert len(rows)==15
        covariance=np.zeros((15,15))
        for name in ('primary','pair-bulk','pair-tail'):
            r=np.array([x['components'][name]['statistics']['replicates'] for x in records])[:,:,2]
            covariance+=np.cov(r,ddof=1)/16
        r=np.array([x['compact']['replicates'] for x in records])[:,:,2]
        covariance+=np.cov(r,ddof=1)/16
        close(covariance,expected['sampling_covariance_correction'],atol=1e-27)
        table(output/'thermal_draft_results.csv',['R','t','draft_baseline_gs3','thermal_correction',
              'worldsheet_thermal_gs3','sampling_SE_correction','matrix_model_thermal'],rows)
        np.savez_compressed(output/'thermal_three_point_covariance.npz',R=np.array(rows)[:,0],covariance=covariance)
        return dict(radii=15,scrambles=16,sampled_contributions=15*16*(256+1024+1024),
            compact_samples=15*64,collision_addbacks=15,all_current_figure_values_match=True,
            full_cross_radius_sampling_covariance_checked=True,limitations=expected['limitations'])
    finally:
        collect.OUTPUT=old_output


def render_figures(output):
    import matplotlib.pyplot as plt
    import plot_thermal_tree_radius as tree
    import plot_thermal_one_loop_radius as loop
    tree.configure_style();dest=output/'figures';dest.mkdir()
    for outgoing in (4,5):
        a=np.genfromtxt(output/'tree_level'/f'{outgoing+1}_point'/f'sphere_1to{outgoing}_thermal_R.csv',delimiter=',',names=True)
        fig,ax=plt.subplots(figsize=(4.8,3.1));fig.subplots_adjust(left=.155,right=.985,bottom=.19,top=.975)
        tree.draw_panel(ax,outgoing,a['R'],a['amplitude_gs_coefficient'])
        fig.savefig(dest/f'sphere_1to{outgoing}_thermal_R.pdf');plt.close(fig)
    for outgoing,path,column in [(1,output/'one_loop/two_point/one_loop_1to1_R.csv','amplitude_imag'),
                                 (2,output/'one_loop/three_point_thermal/thermal_draft_results.csv','worldsheet_thermal_gs3')]:
        a=np.genfromtxt(path,delimiter=',',names=True)
        fig,ax=plt.subplots(figsize=(4.8,3.1));fig.subplots_adjust(left=.155,right=.985,bottom=.19,top=.975)
        loop.draw_panel(ax,outgoing,a['R'],a[column]);fig.savefig(dest/f'torus_1to{outgoing}_thermal_R.pdf');plt.close(fig)
    return sorted(str(p.relative_to(output)) for p in dest.glob('*.pdf'))


def main():
    if sys.flags.optimize:
        raise RuntimeError('Run without -O: numerical verification requires assertions')
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,default=PACKAGE_ROOT/'data/inputs')
    p.add_argument('--output',type=Path)
    p.add_argument('--verify-only',action='store_true')
    p.add_argument('--figures',action='store_true',help='Also render all eight numerical draft panels')
    args=p.parse_args();root=args.data_dir.resolve()
    verified=verify_data(root)
    if args.verify_only:
        print(json.dumps(verified,indent=2));return
    if args.output is None:p.error('--output is required for result recovery')
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    if str(PACKAGE_ROOT) not in sys.path:sys.path.insert(0,str(PACKAGE_ROOT))
    import run
    run.bootstrap()
    jobs=[('tree_level/5_point','tree_level/five_point',recover_five_point),
          ('tree_level/6_point','tree_level/six_point',recover_six_point),
          ('one_loop/two_point','one_loop/two_point',recover_thermal_two_point),
          ('one_loop/three_point','one_loop/three_point',recover_one_loop),
          ('one_loop/three_point_thermal','one_loop/three_point_thermal',recover_thermal_three_point),
          ('two_loop','two_loop',recover_genus2)]
    results={}
    for label,source,fn in jobs:
        target=output/label;target.mkdir(parents=True)
        results[label]=fn(root/source,target)
        print('Recovered '+label,flush=True)
    # The thermal baseline must also match the independently recovered noncompact curve.
    recovered=np.loadtxt(output/'one_loop/three_point/torus_1to2_numeric_rms.dat',skiprows=1)
    saved=np.loadtxt(root/'one_loop/three_point_thermal/reference/torus_1to2_numeric_rms.dat',skiprows=1)
    close(recovered,saved,rtol=2e-12)
    if args.figures:
        from plot_recovered_results import render_all
        figures=render_all(output)
    else:
        figures=[]
    report=dict(passed=True,dataset=verified,results=results,figures=figures,
        scope='Recovery from saved production data. The full production integrations have not been rerun.')
    (output/'verification.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2,allow_nan=False))


if __name__=='__main__':main()
