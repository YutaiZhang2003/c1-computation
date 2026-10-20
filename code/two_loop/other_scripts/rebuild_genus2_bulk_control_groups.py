"""Rebuild the retained bulk/control groups from nodes and the radius cache.

The original inputs use Xi-v5 units. Keep those units for the frozen assembly;
BRY-v7 conversion occurs once in the absolute plot stage. No matrix target
enters this reconstruction. Integrated G and domain corrections are saved
production inputs; the optional control driver can recompute those integrals.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from c1_paths import data_root
from c1_frozen_bundle import verify_bundle
from genus2_integrand_normalization import genus2_fundamental_domain_node_weight, LEGACY_V5_XI_INTEGRATION_KERNEL_CONVENTION as V5
from genus2_separating_tail import RealUpsilonOne, primary_scaled_kernel_batch

def rebuild(parent):
    verify_bundle(parent)
    inputs=parent/'inputs'
    rows=list(csv.DictReader((inputs/'old_combined_samples.csv').open()))
    assert {r['integration_kernel_convention'] for r in rows}=={V5}
    policy=json.loads((parent/'policy.json').read_text())
    with np.load(inputs/'old_radius_cache.npz') as f: cache={k:f[k] for k in f.files}
    with np.load(inputs/'old_primary_cache.npz') as f: saved={k:f[k] for k in f.files}
    with np.load(inputs/'reuse_arrays.npz') as f: expected={k:f[k] for k in f.files}
    np.testing.assert_array_equal(cache['rqmc_node_ids'],[r['rqmc_node_id'] for r in rows])
    radii=cache['radii'];np.testing.assert_array_equal(radii,saved['radii'])
    rep=np.array([int(r['rqmc_replicate']) for r in rows])
    weights=np.array([genus2_fundamental_domain_node_weight(r) for r in rows])
    full=np.array([float(r['transformed_integrand_high']) for r in rows])[:,None]*cache['factors'].T
    rho=np.array([abs(complex(float(r['x12']),float(r['y12']))) for r in rows])
    inside=rho<policy['control_cutoff'];primary=np.zeros_like(full)
    t1=np.array([complex(float(r['x11']),float(r['y11'])) for r in rows])
    t2=np.array([complex(float(r['x22']),float(r['y22'])) for r in rows])
    primary[inside]=primary_scaled_kernel_batch(t1[inside],t2[inside],-np.log(2*np.pi*rho[inside]),radii,
        upsilon=RealUpsilonOne(),momentum_order=24,bridge_order=32,block_order=2,kernel_convention=V5)/rho[inside,None]**2
    def old_reps(value,mask):return np.array([np.sum(value[(rep==r)&mask]*weights[(rep==r)&mask,None],axis=0) for r in range(8)])
    np.testing.assert_allclose(old_reps(full,np.ones(len(rows),bool)),cache['merged_replicates'],rtol=1e-12,atol=1e-13)
    np.testing.assert_allclose(old_reps(primary,inside),saved['rho_0.08_sampled_primary'],rtol=1e-12,atol=1e-13)
    outer=inside&(rho>=policy['band_edges'][0])
    actual=dict(radii=radii,old_base=old_reps(full,~inside)+old_reps(full-primary,outer),
        old_band_residual=np.array([old_reps(full-primary,(rho>=lo)&(rho<hi)) for hi,lo in zip(policy['band_edges'],policy['band_edges'][1:])]),
        old_deep_residual=old_reps(full-primary,rho<policy['band_edges'][-1]),
        integrated_primary=saved['rho_0.08_integrated_primary'],domain_correction=saved['rho_0.08_domain_correction'])
    for key,value in actual.items():np.testing.assert_allclose(value,expected[key],rtol=2e-12,atol=2e-13)
    return actual

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',type=Path,default=data_root('two_loop')/'production/parent')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    arrays=rebuild(a.parent);a.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.output,integration_kernel_convention=V5,**arrays)
    print(json.dumps(dict(rebuilt_saved_bulk_control_groups=True,radii=len(arrays['radii']),output=str(a.output))))

if __name__=='__main__':main()
