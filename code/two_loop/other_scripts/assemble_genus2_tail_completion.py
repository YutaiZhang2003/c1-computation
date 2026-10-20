"""Reassemble measured tail additions without turning diagnostics into bounds.

Frozen production inputs are read-only. All reported kernels are BRY-v7;
Xi-v5 values and weighted residuals are divided by eight exactly once.
"""
from pathlib import Path
import argparse, json, csv
import numpy as np

from c1_paths import data_root, module_path
from c1_observables import genus2_free_energy_arrays, write_genus2_free_energy
from c1_string_conventions import CURRENT_STRING_COUPLING_CONVENTION, matrix_model_genus2_free_energy
HERE=data_root('two_loop')/'tail_completion'
import genus2_separating_tail_cluster as c
from genus2_separating_tail import weighted_residual, band_replicates
from genus2_separating_tail import RealUpsilonOne, primary_scaled_kernel_batch
from genus2_c1_string_integrand import compact_boson_winding_evaluation_genus2
from plot_genus2_absolute_bry import load_frozen_assembly,fit_radius_ansatz,fit_record


def covariance_of_mean(values):
    values=np.asarray(values,float)
    if values.ndim!=2 or len(values)<2 or not np.all(np.isfinite(values)):
        raise ValueError('need complete finite independent replicate curves')
    return np.cov(values,rowvar=False,ddof=1)/len(values)


def assemble(source=None, output=None):
    parent=data_root('two_loop')/'production/parent'
    source=Path(source) if source is not None else data_root('two_loop')/'assembly_v6'
    if output is not None:
        output=Path(output)
        output.mkdir(parents=True,exist_ok=False)
    frozen=load_frozen_assembly(source)
    nodes=c.read_json(parent/'design/nodes.json')
    parent_policy=c.read_json(parent/'policy.json')
    radii=frozen['radii'];one=int(np.flatnonzero(radii==1)[0])
    probes={label:c.read_json(HERE/f'node16_{label}.json')
            for label in ['roundtrip','confirmation','momentum14']}
    node=nodes[16]
    for label,probe in probes.items():
        if probe['status']!='diagnostic_evaluated' or probe['original_node']!=node:
            raise ValueError(f'incomplete/stale retry {label}')
        for key in node:
            if key!='support_reasons' and probe['node'][key]!=node[key]:
                raise ValueError('retry changed original proposal')
        p=probe['period']
        if max(p['residual'],p['saved_coordinate_residual'],p['map_step'],p['saved_coordinate_map_step'])>p['period_tolerance']:
            raise ValueError('retry period certificate failed')
    low,confirm,selected=(probes[k] for k in ['roundtrip','confirmation','momentum14'])
    for probe in [low,confirm]:
        checks=probe['cft']['checks'];policy=probe['policy']
        for key,threshold in [('block_relative_step','block_tolerance'),
                              ('momentum_relative_step','momentum_tolerance'),
                              ('precision_relative_step','precision_tolerance'),
                              ('vacuum_relative_step','vacuum_tolerance')]:
            if checks[key]>policy[threshold]:raise ValueError('retry CFT check failed')
    if selected['cft']['momentum_orders']!=[14,14,14] or selected['cft']['block_order']!=10:
        raise ValueError('changed selected retry order')
    # R dependence is entirely in R*Theta_R at fixed Omega. Check the
    # factorized reconstruction against all four directly saved radii.
    omega=c.node_omega(node)
    winding=np.array([compact_boson_winding_evaluation_genus2(omega,float(r),tolerance=1e-13).value for r in radii])
    radius_factor=radii*winding/winding[one]
    full=selected['cft']['scaled_kernel'][1]*radius_factor
    for i,r in enumerate([.5,1.,2.,15.]):
        j=int(np.flatnonzero(radii==r)[0])
        np.testing.assert_allclose(full[j],selected['cft']['scaled_kernel'][i],rtol=2e-13)
    control=primary_scaled_kernel_batch([complex(node['x1'],node['y1'])],
        [complex(node['x2'],node['y2'])],[node['length']],radii,upsilon=RealUpsilonOne(),
        momentum_order=24,bridge_order=32,block_order=2,
        kernel_convention=c.policy_kernel_convention(parent_policy['numerical_policy']))[0]
    new_weighted=weighted_residual(full[None,:],control[None,:],[node])[0]
    retry_order_shift=(selected['cft']['scaled_kernel'][1]-confirm['cft']['scaled_kernel'][1])*radius_factor
    retry_order_sensitivity=np.abs(retry_order_shift)*np.exp(node['log_weight_over_rho_squared'])/8/8
    ids=[i for i,n in enumerate(nodes) if n['band']==2]
    if len(ids)!=8:raise ValueError('changed nominal finite-band design')
    band_values=[]
    input_hashes=dict(frozen['input_sha256'])
    for i in ids:
        n=nodes[i]
        path=source/'pilot_results/nodes'/f"{n['node_id']}.json"
        old=c.read_json(path);input_hashes[str(path)]=c.sha(path)
        if old['node_sha256']!=n['node_sha256']:raise ValueError('stale parent node identity')
        if i==16:
            if old['status']=='ok':raise ValueError('original failure unexpectedly replaced')
            value=new_weighted
        else:
            if old['status']!='ok':raise ValueError('band remains incomplete')
            value=np.array(old['weighted_residual'])
            checked=weighted_residual(np.array(old['cft']['scaled_kernel'])[None,:],
                np.array(old['control_scaled'])[None,:],[n])[0]
            np.testing.assert_allclose(value,checked,rtol=1e-11,atol=1e-16)
        band_values.append(value/8)
    band_values=np.array(band_values)
    band_values_before_refinement=band_values.copy()
    refined_paths=[HERE/'band2_order_checks'/f"{nodes[i]['node_id']}.json" for i in ids if i!=16]
    band_refined=all(p.exists() and c.read_json(p).get('status')=='ok' for p in refined_paths)
    if not band_refined:raise ValueError('all seven band refinements are required')
    if band_refined:
        for k,i in enumerate(ids):
            if i==16:continue
            p=HERE/'band2_order_checks'/f"{nodes[i]['node_id']}.json"
            rec=c.read_json(p)
            original=source/'pilot_results/nodes'/f"{nodes[i]['node_id']}.json"
            if rec['node']!=nodes[i] or rec['source_sha256']!=c.sha(original):
                raise ValueError('stale band order refinement')
            if rec['cft']['block_order']!=10 or rec['cft']['momentum_orders']!=[14]*3:
                raise ValueError('changed recovered-band order policy')
            np.testing.assert_array_equal(rec['radii'],radii)
            old=c.read_json(original)
            expected_shift=(np.array(rec['cft']['scaled_kernel'])-np.array(old['cft']['scaled_kernel']))
            expected_shift*=np.exp(nodes[i]['log_weight_over_rho_squared'])/8
            np.testing.assert_allclose(expected_shift,rec['weighted_shift_BRY'],rtol=1e-12,atol=1e-20)
            band_values[k]+=expected_shift
            input_hashes[str(p)]=c.sha(p)
    reps=[]
    for rep in range(4):
        chosen=[k for k,i in enumerate(ids) if nodes[i]['replicate']==rep]
        if len(chosen)!=2:raise ValueError('missing nominal sample')
        reps.append(band_values[chosen].sum(axis=0)/2)
    reps=np.array(reps)
    groups=dict(frozen['groups'],residual_band_2=reps)
    mean=sum(v.mean(axis=0) for v in groups.values())
    covariance=sum(covariance_of_mean(v) for v in groups.values())
    result=dict(status='finite_band_completed_deep_tail_and_systematics_not_closed',
        native_quantity='coefficient of g_s^2 in connected log Z',
        thermal_output='free_energy_F2.csv; F=-log Z/(2*pi*R), genus-two term=g_s^2*F2',
        full_free_energy_available=False,certified=False,
        integration_kernel_convention=c.CURRENT_INTEGRATION_KERNEL_CONVENTION,
        original_nominal_node_count=40,resolved_node_count=40,
        incomplete_original_finite_bands=[],
        matrix_model_used_to_calibrate=False,
        finite_band_005_01=dict(mean_R1=float(reps.mean(axis=0)[one]),
            empirical_SE_R1=float(np.sqrt(covariance_of_mean(reps)[one,one])),
            replicate_means_R1=reps[:,one].tolist(),nominal_points_per_replicate=2,replicates=4,
            all_eight_nodes_B10_N14=band_refined,
            paired_refinement_shift_R1=float((band_values-band_values_before_refinement).mean(axis=0)[one]),
            sampling_covariance_recomputed_from_refined_replicates=True,
            node16_momentum_order_sensitivity_R1=float(retry_order_sensitivity[one]),
            node16_order_difference_is_bound=False),
        remaining_residual_region='0 < |Omega12| < 0.001',
        uncertainty_scope='Empirical independent-scramble covariance only; no missing term or systematic replaced by zero.')
    arrays=dict(radii=radii,finite_band_2_replicates=reps,finite_band_2_node_values=band_values,
        finite_band_2_node_values_before_refinement=band_values_before_refinement,
        node16_full_scaled_Xi=full,node16_control_scaled_Xi=control,
        node16_order_sensitivity_BRY=retry_order_sensitivity,
        finite_bands_resolved_mean=mean.copy(),finite_bands_resolved_covariance=covariance.copy(),
        matrix_model=frozen['matrix_model'])
    result['finite_bands_resolved_R1']=dict(value=float(mean[one]),empirical_SE=float(np.sqrt(covariance[one,one])))
    design_path=HERE/'deep_band_design.json'
    if design_path.exists():
        deep_nodes=c.read_json(design_path);deep_policy=c.read_json(HERE/'deep_band_policy.json')
        deep=[];problems=[]
        for n in deep_nodes:
            p=HERE/'deep_band_nodes'/f"{n['node_id']}.json"
            if not p.exists():problems.append(n['node_id']);continue
            rec=c.read_json(p)
            if rec.get('status')!='ok':problems.append(n['node_id']);continue
            if rec['node']!=n or rec['policy_sha256']!=c.object_sha(deep_policy):
                raise ValueError('stale deep sample')
            np.testing.assert_array_equal(rec['radii'],radii)
            v=np.array(rec['weighted_residual'])
            if not rec.get('domain_rejected',False):
                checked=weighted_residual(np.array(rec['cft']['scaled_kernel'])[None,:],
                    np.array(rec['control_scaled'])[None,:],[n])[0]
                np.testing.assert_allclose(v,checked,rtol=1e-11,atol=1e-16)
            elif np.any(v!=0):raise ValueError('domain rejection must be a nominal zero')
            deep.append(v/8);input_hashes[str(p)]=c.sha(p)
        result['deep_finite_band_status']=dict(nominal=len(deep_nodes),successful=len(deep),problems=problems)
        if not problems:
            deep_reps=band_replicates(deep,deep_nodes,deep_policy)[0]
            groups['residual_band_1e5_1e3']=deep_reps
            mean+=deep_reps.mean(axis=0);covariance+=covariance_of_mean(deep_reps)
            arrays.update(deep_finite_band_replicates=deep_reps,deep_finite_band_node_values=np.array(deep))
            result['deep_finite_band_status'].update(mean_R1=float(deep_reps.mean(axis=0)[one]),
                empirical_SE_R1=float(np.sqrt(covariance_of_mean(deep_reps)[one,one])),
                replicate_means_R1=deep_reps[:,one].tolist())
            result['remaining_residual_region']='0 < |Omega12| < 0.00001'
    if result.get('deep_finite_band_status',{}).get('successful')!=16:
        raise ValueError('all sixteen new deep-band samples are required')
    arrays.update(resolved_mean=mean,resolved_covariance=covariance,**{f'group_{k}':v for k,v in groups.items()})
    result['resolved_R1']=dict(value=float(mean[one]),empirical_SE=float(np.sqrt(covariance[one,one])),
        matrix_model=float(frozen['matrix_model'][one]))
    result['fit']=fit_record(fit_radius_ansatz(radii,mean,covariance),radii)
    result['error_budget']=[
        dict(component='Monte Carlo',status='estimated',scope='complete independent scramble curves; covariance retained'),
        dict(component='Remaining deep residual',status='unbounded',region=result['remaining_residual_region']),
        dict(component='Primary control',status='order sensitivity computed separately; not a uniform bound'),
        dict(component='Current bulk CFT/period/scalar',status='no complete validated global error bound'),
        dict(component='Absolute measure prescription',status='BRY-v7 adopted unchanged; independent derivation not supplied by these checks')]
    for path in [parent/'policy.json',parent/'design/nodes.json',*[HERE/f'node16_{k}.json' for k in probes]]:
        input_hashes[str(path)]=c.sha(path)
    result['input_sha256']=input_hashes
    result['code_sha256']={str(p):c.sha(p) for p in [Path(__file__),module_path('genus2_separating_residual_pilot'),
        module_path('plot_genus2_absolute_bry'),module_path('genus2_separating_tail')]}
    if output is not None:
        np.savez_compressed(output/'tail_audit_arrays.npz',**arrays)
        c.atomic_json(output/'tail_audit_summary.json',result)
        with (output/'tail_audit_radius_values.csv').open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['R','resolved_F2_over_gBRY_squared','empirical_MC_SE','matrix_model','finite_band_005_01'])
            w.writerows(zip(radii,mean,np.sqrt(np.diag(covariance)),frozen['matrix_model'],reps.mean(axis=0)))
    for path,digest in input_hashes.items():
        if c.sha(path)!=digest:raise ValueError('input changed during assembly')
    if output is not None:
        write_genus2_free_energy(output, arrays, result)
        plot_result(arrays,result,output)
    return result,arrays


def plot_result(arrays,result,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from plot_genus2_absolute_bry import design_matrix
    r=arrays['radii']
    fit=fit_radius_ansatz(r,arrays['resolved_mean'],arrays['resolved_covariance'])
    thermal=genus2_free_energy_arrays(r,arrays['resolved_mean'],arrays['resolved_covariance'],
        source_convention=CURRENT_STRING_COUPLING_CONVENTION)
    y,c,mm=(thermal[k] for k in ('F2','covariance_F2','F2_matrix_model'))
    se=np.sqrt(np.diag(c))
    dense=np.geomspace(r.min(),r.max(),500)
    mm_dense=np.array([matrix_model_genus2_free_energy(r) for r in dense])
    fig,(ax,ratio)=plt.subplots(2,1,figsize=(10,7.5),sharex=True,
        gridspec_kw={'height_ratios':[3,1.4],'hspace':.12})
    ax.plot(dense,mm_dense,color='#333333',label='Matrix model: absolute prediction')
    ax.plot(dense,-(design_matrix(dense)@fit['coefficients'])/(2*np.pi*dense),'--',color='#c47820',label='Radius fit')
    ax.errorbar(r,y,yerr=se,fmt='o',ms=3,color='#1768ac',elinewidth=.8,label='Worldsheet: current resolved sum')
    ax.set(title='c=1 genus-two thermal free energy',ylabel=r'$F_2(R)$')
    ax.legend(frameon=False);ax.grid(alpha=.15)
    ratio.axhline(0,color='#333333',lw=1)
    ratio.errorbar(r,100*(y/mm-1),yerr=100*se/np.abs(mm),fmt='o',ms=3,color='#1768ac',elinewidth=.7)
    ratio.set(xscale='log',xlabel='Radius R',ylabel='Difference from\nmatrix model (%)')
    ratio.set_xticks([.5,1,2,3,5,10,15],labels=['0.5','1','2','3','5','10','15'])
    ratio.grid(alpha=.15)
    fig.subplots_adjust(left=.12,right=.97,top=.92,bottom=.20)
    fig.text(.12,.09,'F = -log Z/(2πR); genus-two term = g_s² F₂(R); μ⁻¹ = 2πg_s.',fontsize=9)
    fig.text(.12,.05,'Error bars: empirical MC only. Residual below |Omega12|=1e-5 and total systematics remain unbounded.',fontsize=8.7)
    for suffix in ('png','svg'):fig.savefig(output/f'absolute_free_energy_bry.{suffix}',dpi=180)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,help='Rebuilt intermediate BRY-v6 assembly')
    parser.add_argument('--output',type=Path,default=data_root('two_loop')/'reproduced_tail_completion')
    args=parser.parse_args()
    result,arrays=assemble(source=args.source,output=args.output)
    thermal=genus2_free_energy_arrays(arrays['radii'],arrays['resolved_mean'],arrays['resolved_covariance'],
        source_convention=CURRENT_STRING_COUPLING_CONVENTION)
    one=int(np.flatnonzero(arrays['radii']==1)[0])
    print(json.dumps(dict(quantity='thermal F2(R), with g_s^2 stripped',
        F2_R1=float(thermal['F2'][one]),empirical_MC_SE=float(thermal['F2_empirical_MC_SE'][one]),
        matrix_model_F2_R1=float(thermal['F2_matrix_model'][one]),
        remaining_residual_region=result['remaining_residual_region']),indent=2))


if __name__=='__main__':main()
