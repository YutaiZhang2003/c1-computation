#!/usr/bin/env python3
"""Non-destructive separating-tail correction of the saved mixed-channel run.

Estimator identity, for a control G supported in rho<rho0 on product tori:
  integral_D K = E_old[(K-G)/p_old] + integral_product G
                 - integral_(product outside D) G.

This script NEVER changes a saved kernel, radius cache, or published result.
The output is a diagnostic corrected pilot, not a precision certificate.
Cutoff dependence, momentum/block convergence, and independently sampled
finite-bridge residuals must be checked before a production release.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from c1_paths import data_root

import numpy as np
from scipy.stats import qmc

from genus2_integrand_normalization import genus2_fundamental_domain_node_weight
from genus2_separating_production import saved_rows_scale_to_current
from genus2_separating_production import saved_array_scale_to_current
from genus2_separating_tail_cluster import CURRENT_INTEGRATION_KERNEL_CONVENTION
from genus2_separating_tail import (
    RealUpsilonOne, integrate_primary_tail, leading_integrated_tail,
    primary_scaled_kernel_batch, separating_proposals_from_unit_cube,
)
from genus2_siegel_fundamental_domain import in_gottschling_domain


DEFAULT_STUDY=data_root('two_loop')/'production/parent/inputs'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assemble_control_variate(original, sampled_control, integrated_control,
                            rejected_control):
    """Independent-scramble means/covariances; no accepted-count normalization."""
    arrays=[np.asarray(a,float) for a in (original,sampled_control,integrated_control,rejected_control)]
    if (any(a.ndim != 2 or a.shape[0] < 2 or np.any(~np.isfinite(a)) for a in arrays)
            or arrays[0].shape != arrays[1].shape
            or len({a.shape[1] for a in arrays}) != 1):
        raise ValueError('incompatible/nonfinite independent replicate arrays')
    residual=arrays[0]-arrays[1]
    groups=(residual,arrays[2],-arrays[3])
    mean=sum(a.mean(axis=0) for a in groups)
    covariance=sum(np.atleast_2d(np.cov(a,rowvar=False))/len(a) for a in groups)
    return mean,covariance,residual


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,default=DEFAULT_STUDY)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cutoffs',default='.005,.01,.02,.04,.08')
    p.add_argument('--power',type=int,default=12)
    p.add_argument('--replicates',type=int,default=8)
    p.add_argument('--momentum-order',type=int,default=24)
    p.add_argument('--bridge-order',type=int,default=32)
    p.add_argument('--block-order',type=int,default=2)
    p.add_argument('--domain-power',type=int,default=16)
    args=p.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite {args.output}')
    cutoffs=np.array([float(v) for v in args.cutoffs.split(',')])
    if np.any(cutoffs <= 0) or np.any(cutoffs >= 1/(2*math.pi)):
        raise ValueError('invalid separating cutoffs')
    summary=json.loads((args.study/'summary.json').read_text())
    source=args.study/'old_combined_samples.csv'
    assert digest(source)==summary['merged_sha256'],'source hash mismatch'
    with source.open() as f:
        rows=list(csv.DictReader(f))
    cache_path=args.study/'old_radius_cache.npz'
    cache_hash=digest(cache_path)
    cache=np.load(cache_path)
    np.testing.assert_array_equal(cache['rqmc_node_ids'],[r['rqmc_node_id'] for r in rows])
    radii=cache['radii'];factors=cache['factors']
    original=cache['merged_replicates']
    rep=np.array([int(r['rqmc_replicate']) for r in rows])
    weights=np.array([genus2_fundamental_domain_node_weight(r) for r in rows])
    kernels=np.array([float(r['transformed_integrand_high']) for r in rows])
    kernels=kernels*saved_rows_scale_to_current(rows)
    original=original*saved_array_scale_to_current(cache)
    np.testing.assert_allclose(np.array([np.sum(weights[rep==k]*kernels[rep==k]
        *factors[:,rep==k],axis=1) for k in range(len(original))]),original,rtol=1e-12)
    tau1=np.array([complex(float(r['x11']),float(r['y11'])) for r in rows])
    tau2=np.array([complex(float(r['x22']),float(r['y22'])) for r in rows])
    rho=np.array([abs(complex(float(r['x12']),float(r['y12']))) for r in rows])
    selected=np.flatnonzero(rho<max(cutoffs))
    upsilon=RealUpsilonOne()
    common=dict(upsilon=upsilon,momentum_order=args.momentum_order,
                bridge_order=args.bridge_order,block_order=args.block_order)
    model=np.zeros((len(rows),len(radii)))
    model[selected]=primary_scaled_kernel_batch(tau1[selected],tau2[selected],
        -np.log(2*math.pi*rho[selected]),radii,**common)/rho[selected,None]**2
    one=int(np.flatnonzero(radii==1.)[0])
    node_comparison=[dict(node_id=rows[i]['rqmc_node_id'],rho=float(rho[i]),
        full_R1=float(kernels[i]),primary_R1=float(model[i,one]),
        full_over_primary_R1=float(kernels[i]/model[i,one])) for i in selected]
    args.output.mkdir(parents=True)
    table=[];raw={};cut_summaries=[]
    for cutoff in cutoffs:
        print(f'Integrating rho<{cutoff:g}',flush=True)
        tail=integrate_primary_tail(radii,float(cutoff),power=args.power,
            replicates=args.replicates,seed=20260901,**common)
        mask=rho<cutoff
        sampled=np.array([np.sum((weights*mask)[rep==k,None]*model[rep==k],axis=0)
                          for k in range(len(original))])
        rejected=[];rejection_counts=[]
        # Independent seeds from both the old bulk run and the analytic-tail RQMC.
        for k in range(args.replicates):
            unit=qmc.Sobol(6,scramble=True,seed=20261901+k).random_base2(args.domain_power)
            proposal=separating_proposals_from_unit_cube(unit,float(cutoff))
            outside=~in_gottschling_domain(proposal.domain_matrices(),tolerance=0.)
            rejection_counts.append(int(outside.sum()))
            if np.any(outside):
                # Rebuild the selected proposal, retaining the full nominal N.
                selected_proposal=separating_proposals_from_unit_cube(unit[outside],float(cutoff))
                rejected.append(selected_proposal.weighted_primary(radii,**common).sum(axis=0)/len(unit))
            else:
                rejected.append(np.zeros(len(radii)))
        rejected=np.array(rejected)
        mean,cov,residual=assemble_control_variate(original,sampled,tail.replicates,rejected)
        if np.any(mean<=0):
            raise ArithmeticError('nonpositive corrected estimate')
        shape=mean/mean[one]
        # Propagate the independent mean covariance through normalization.
        shape_var=(np.diag(cov)+shape**2*cov[one,one]-2*shape*cov[:,one])/mean[one]**2
        for j,r in enumerate(radii):
            mm=(7*r+10/r+7/r**3)/24
            table.append(dict(rho_cutoff=float(cutoff),radius=float(r),old_total=float(original[:,j].mean()),
                sampled_primary=float(sampled[:,j].mean()),integrated_primary=float(tail.mean[j]),
                domain_correction=float(rejected[:,j].mean()),corrected_pilot_total=float(mean[j]),
                empirical_standard_error=float(math.sqrt(max(0.,cov[j,j]))),
                normalized_pilot=float(shape[j]),normalized_model=float(mm),
                normalized_relative_difference=float(shape[j]/mm-1),
                empirical_shape_standard_error=float(math.sqrt(max(0.,shape_var[j])))))
        key=f'rho_{cutoff:g}'
        raw[key+'_integrated_primary']=tail.replicates
        raw[key+'_sampled_primary']=sampled
        raw[key+'_bulk_residual']=residual
        raw[key+'_domain_correction']=rejected
        raw[key+'_mean_covariance']=cov
        info=dict(rho_cutoff=float(cutoff),old_nodes_in_tube=int(mask.sum()),
                  primary_tail_R1=float(tail.mean[one]),corrected_R1=float(mean[one]),
                  empirical_error_R1=float(math.sqrt(cov[one,one])),
                  normalized_R15=float(shape[-1]),domain_rejections=rejection_counts,
                  domain_nominal_proposals_per_replicate=2**args.domain_power,
                  leading_tail_R1=float(leading_integrated_tail([1.],float(cutoff))[0]))
        cut_summaries.append(info);print(json.dumps(info),flush=True)
    with (args.output/'radius_comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    with (args.output/'node_overlap.csv').open('w',newline='') as f:
        fields=['node_id','rho','full_R1','primary_R1','full_over_primary_R1']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(node_comparison)
    np.savez_compressed(args.output/'replicate_arrays.npz',radii=radii,original=original,
        integration_kernel_convention=CURRENT_INTEGRATION_KERNEL_CONVENTION,**raw)
    metadata=dict(status='diagnostic_corrected_pilot_not_precision_certified',
        integration_kernel_convention=CURRENT_INTEGRATION_KERNEL_CONVENTION,
        source_csv=str(source.resolve()),source_sha256=summary['merged_sha256'],
        source_radius_cache=str(cache_path.resolve()),source_radius_cache_sha256=cache_hash,
        settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        upsilon_validation_max_abs_error=upsilon.validation_max_abs_error,cutoffs=cut_summaries,
        estimator='old(K-G) + analytic-radial product-domain G - sampled outside-domain G',
        error_scope='Empirical independent-scramble covariance only; excludes CFT truncation, control quadrature bias, rare domain-boundary uncertainty, and unresolved bulk tails.',
        normalization='BRY v7 (1/2)_Aut*16*pi^4=8*pi^4; no matrix-model calibration')
    (args.output/'summary.json').write_text(json.dumps(metadata,indent=2)+'\n')
    assert digest(source)==summary['merged_sha256'] and digest(cache_path)==cache_hash
    print(f'Wrote diagnostic pilot to {args.output}',flush=True)


if __name__=='__main__':
    main()
