#!/usr/bin/env python3
"""Bounded labelled-patch CFT pilot for analytic two-outer-collar subtraction.

The complete coupled DOZZ product and all three momentum integrations remain.
This is not a full-amplitude run and does not change the existing atlas scan.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import runpy
import time

ROOT = Path(__file__).resolve().parents[4]
runpy.run_path(str(ROOT/'run.py'))['bootstrap']()
import numpy as np
from six_point_analytic_collar_subtraction import (
    CollarPatch, channel_exponents, integrate_outer_collars,
)
from six_point_collar_block_evaluator import SphereSixPointCollarBlock


def encode(value):
    if isinstance(value, dict):
        return {str(k):encode(v) for k,v in value.items()}
    if isinstance(value, (tuple,list)):
        return [encode(v) for v in value]
    if isinstance(value, np.ndarray):
        return encode(value.tolist())
    if isinstance(value, complex):
        return dict(real=value.real,imag=value.imag)
    if isinstance(value, np.generic):
        return encode(value.item())
    return value


def save(path, value):
    Path(path).write_text(json.dumps(encode(value),indent=2,allow_nan=False)+'\n')


def external_weights(t, ordering):
    return tuple(1-(25 if label==0 else 1)*t*t/4 for label in ordering)


def representation_for(ds, hs):
    """Explicit level-one weight criterion, fixed across order/cutoff ladders.

    This identifies which normalization has the smaller first nome
    coefficient. It is a proposal for numerical representation, never a
    convergence bound. The integrated order ladder still includes every node.
    """
    left,middle,right = hs
    H1 = 8*(left-ds[2])*(right-ds[3])/middle
    E1 = 8*(middle+ds[2]-left)*(middle+ds[3]-right)/middle
    return 'standard' if abs(H1) <= abs(E1) else 'complete'


def stable_sum(log_weights, values):
    weights = np.asarray(log_weights,dtype=np.clongdouble)
    values = np.asarray(values,dtype=np.longdouble)
    selected = values != 0
    if not selected.any():
        return dict(value=0j,log_abs_value=None,cancellation_condition=None)
    logs = weights[selected]+np.log(np.abs(values[selected]))
    top = logs.real.max()
    scaled = np.sign(values[selected])*np.exp(logs-top)
    total = scaled.sum(dtype=np.clongdouble)
    value = np.exp(top)*total
    if not np.isfinite(value):
        raise FloatingPointError("nonfinite spectral sum")
    return dict(value=complex(value),
                log_abs_value=float(top+np.log(abs(total))) if total else None,
                cancellation_condition=float(np.abs(scaled).sum()/abs(total)) if total else None)


def source_hashes():
    names = (
        'code/tree_level/six_point/moduli_integration/six_point_analytic_collar_subtraction.py',
        'code/tree_level/six_point/conformal_block/six_point_collar_block_evaluator.py',
        'code/general/virasoro/sphere_four_point_pillow_evaluate.py',
        'code/general/virasoro/sphere_four_point_c25.py',
        'code/tree_level/six_point/conformal_block/six_point_collar_block_evaluator.py',
        'code/tree_level/six_point/liouville_integration/six_point_collar_momentum_quadrature.py',
        'code/tree_level/six_point/liouville_integration/six_point_endpoint_and_tail_quadrature.py',
        'code/tree_level/six_point/experiments/liouville_dozz_vectorized.py',
        str(Path(__file__).relative_to(ROOT)),
    )
    return {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
            for name in names if (ROOT/name).exists()}


def fixed_momentum_checks(patch, energies, *, dps=60):
    rows=[]
    x = complex(-.4232993392902615,-.8904122668995105)
    momenta = ((.2144373749324254,.591717321247825,.2552831856954262),
               (.02,.15,.04),(.6,1.4,.7),(6.,6.,6.))
    cuts = ((.0025,.00001),(.005,.0001),(.01,.001))
    for t in energies:
        for ps in momenta:
            hs=tuple(1+p*p for p in ps)
            core=SphereSixPointCollarBlock(external_weights(t,patch.ordering),
                     (hs[0],hs[2]),order=24,dps=dps)
            slices=core.slices(hs[1],x,outer_order=2)
            direct=integrate_outer_collars(slices,t,ps,x,patch.left_radius,patch.right_radius,
                       ordering=patch.ordering,radial_order=10,angular_order=10)
            coarse=integrate_outer_collars(slices,t,ps,x,patch.left_radius,patch.right_radius,
                       ordering=patch.ordering,radial_order=6,angular_order=6)
            outer1=integrate_outer_collars({k:v for k,v in slices.items() if sum(k)<=1},
                       t,ps,x,patch.left_radius,patch.right_radius,ordering=patch.ordering,
                       radial_order=10,angular_order=10)
            stitches=[]
            for A,B in cuts:
                pieces=[]
                # Three disjoint rectangles in the pair of radial intervals:
                # inner x inner; outer-a annulus x full-b; inner-a x outer-b.
                for lower,upper in (((0.,0.),(A,B)),
                                    ((A,0.),(patch.left_radius,patch.right_radius)),
                                    ((0.,B),(A,patch.right_radius))):
                    pieces.append(integrate_outer_collars(slices,t,ps,x,*upper,
                         ordering=patch.ordering,radial_order=10,angular_order=10,
                         inner_radii=lower))
                sewn=pieces[0]['leading']+pieces[0]['remainder']+sum(p['direct'] for p in pieces[1:])
                stitches.append(dict(cutoffs=(A,B),stitched=sewn,
                      relative_to_direct=abs(sewn/direct['direct']-1)))
            rows.append(dict(t=t,momenta=ps,direct=direct,coarse=coarse,
                      radial_angular_relative=abs(coarse['total']/direct['total']-1),
                      outer1_vs2_relative=abs(outer1['total']/direct['total']-1),
                      cutoff_stitching=stitches,provider_dps=core.dps))
    return rows


def integrate_patch(t, profile, patch, *, x_order=2, block_orders=(8,12,16),
                    radial_order=6, angular_order=6, dps=60, progress=None):
    from six_point_collar_momentum_quadrature import build_collar_spectral_grid
    if patch.ordering.index(0) != 1:
        raise ValueError('this pilot spectral grid requires the incoming puncture in slot 1')
    start=time.perf_counter()
    refinements = {'refined':((6,6,6),(6,7,8)),
                   'high':((8,8,8),(8,9,10)),
                   'reference':((10,10,10),(10,11,12))}
    extra = {}
    if profile in refinements:
        extra = dict(core_orders=refinements[profile][0],tail_orders=refinements[profile][1])
    grid=build_collar_spectral_grid(t,(patch.left_radius,patch.right_radius),
                       profile='fine' if extra else profile,**extra)
    grid['metadata']['pilot_profile']=profile
    ps=np.asarray(grid['p_values'],float)
    logw=np.asarray(grid['log_weighted_dozz'],complex)
    xs,xweights=patch.middle_rule(x_order)
    orders=tuple(sorted(set(map(int,block_orders))))
    if not orders or min(orders)<0 or max(orders)>24:
        raise ValueError("block orders must lie between0 and24")
    node_values={n:{k:np.zeros(len(ps)) for k in ('leading','remainder','total','direct','outer1')}
                 for n in orders}
    groups={}
    for row,(left,middle,right) in enumerate(ps):
        groups.setdefault((float(left),float(right)),[]).append(row)
    ds=external_weights(t,patch.ordering)
    coefficient_seconds=0.;integration_seconds=0.;max_provider_dps=0
    representations=[None]*len(ps)
    for group_no,((left,right),indices) in enumerate(groups.items(),1):
        tick=time.perf_counter()
        core=SphereSixPointCollarBlock(ds,(1+left*left,1+right*right),order=max(orders),dps=dps)
        for row in indices:
            hmiddle=1+ps[row,1]**2
            mode=representation_for(ds,(1+left*left,hmiddle,1+right*right))
            if mode=='complete':
                core.complete_coefficients(hmiddle)
            else:
                core.coefficient_bank.coefficients(hmiddle)
        coefficient_seconds+=time.perf_counter()-tick
        max_provider_dps=max(max_provider_dps,core.dps)
        for row in indices:
            beta=channel_exponents(t,ps[row],patch.ordering)
            representation=representation_for(ds,tuple(1+ps[row]**2))
            representations[row]=representation
            for x,xweight in zip(xs,xweights):
                middle_measure=xweight*abs(x)**(beta[1]-2)
                for n in orders:
                    tick=time.perf_counter()
                    slices=core.slices(1+ps[row,1]**2,x,elliptic_order=n,outer_order=2,
                                      representation=representation)
                    coefficient_seconds+=time.perf_counter()-tick
                    tick=time.perf_counter()
                    result=integrate_outer_collars(slices,t,ps[row],x,patch.left_radius,
                         patch.right_radius,ordering=patch.ordering,
                         radial_order=radial_order,angular_order=angular_order)
                    if n==max(orders):
                        first=integrate_outer_collars({k:v for k,v in slices.items() if sum(k)<=1},
                              t,ps[row],x,patch.left_radius,patch.right_radius,
                              ordering=patch.ordering,radial_order=radial_order,angular_order=angular_order)
                    else:
                        first=result
                    for key in ('leading','remainder','total','direct'):
                        node_values[n][key][row]+=middle_measure*result[key]
                    node_values[n]['outer1'][row]+=middle_measure*first['total']
                    integration_seconds+=time.perf_counter()-tick
        if progress is not None and (group_no==1 or group_no==len(groups) or group_no%16==0):
            progress(dict(t=t,profile=profile,groups_done=group_no,groups=len(groups),
                          elapsed=time.perf_counter()-start))
    sums={str(n):{key:stable_sum(logw,values) for key,values in row.items()}
          for n,row in node_values.items()}
    last=sums[str(max(orders))]
    reference_values=node_values[max(orders)]['total']
    absolute_changes={str(n):stable_sum(logw.real,np.abs(node_values[n]['total']-reference_values))['value'].real
                              /abs(last['total']['value']) for n in orders}
    flagged={str(n):np.abs(node_values[n]['total']-reference_values)>
                    .01*np.maximum(np.abs(reference_values),np.finfo(float).tiny) for n in orders}
    dominant=np.argsort(logw.real+np.log(np.maximum(node_values[max(orders)]['total'],np.finfo(float).tiny)))[-5:][::-1]
    return dict(t=t,profile=profile,spectral_metadata=grid['metadata'],momentum_nodes=len(ps),
                outer_coefficient_banks=len(groups),x_order=x_order,x_nodes=len(xs),
                block_orders=orders,radial_order=radial_order,angular_order=angular_order,
                values=sums,relative_remainder=abs(last['remainder']['value']/last['total']['value']),
                outer1_vs2_relative=abs(last['outer1']['value']/last['total']['value']-1),
                middle_order_relative={str(n):abs(sums[str(n)]['total']['value']/last['total']['value']-1) for n in orders},
                absolute_weighted_middle_order_change=absolute_changes,
                node_order_change_over_one_percent={str(n):dict(count=int(flagged[str(n)].sum()),
                      absolute_reference_contribution_fraction=stable_sum(logw.real,
                         np.where(flagged[str(n)],np.abs(reference_values),0.))['value'].real/abs(last['total']['value']))
                      for n in orders},
                dominant_nodes=[dict(index=int(i),momenta=ps[i].tolist()) for i in dominant],
                seconds=time.perf_counter()-start,coefficient_seconds=coefficient_seconds,
                outer_integration_seconds=integration_seconds,maximum_provider_dps=max_provider_dps,
                measure='dP_left dP_middle dP_right / pi^3; d²a d²x d²b; all four DOZZ factors',
                omitted='overall string coupling/normalization and other labelled moduli regions',
                representation_policy='smaller absolute first nome coefficient; fixed across order and cutoff ladders; no tail certificate',
                representation_counts={mode:representations.count(mode) for mode in ('standard','complete')},
                representation_integrals={mode:stable_sum(logw,np.where(np.asarray(representations)==mode,
                                    node_values[max(orders)]['total'],0.)) for mode in ('standard','complete')},
                all_momentum_nodes_retained=True,full_amplitude=False,rigorous_error_bound=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--t',type=float,nargs='+',default=[.05,.2005,.3205])
    parser.add_argument('--profiles',nargs='+',default=['coarse','fine'])
    parser.add_argument('--x-order',type=int,default=2)
    parser.add_argument('--block-orders',type=int,nargs='+',default=[8,12,16])
    parser.add_argument('--radial-order',type=int,default=6)
    parser.add_argument('--angular-order',type=int,default=6)
    parser.add_argument('--fixed-only',action='store_true')
    parser.add_argument('--skip-fixed',action='store_true')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    patch=CollarPatch(.02,.01,(-.435,-.405,-.898,-.878))
    hashes=source_hashes()
    result=dict(scope='Explicit single labelled two-collar patch; analytic primary addback plus finite outer-degree-two remainder; no full amplitude rerun.',
                patch=vars(patch),sources=hashes,scans=[])
    if not args.skip_fixed:
        result['fixed_momentum_checks']=fixed_momentum_checks(patch,args.t)
        save(args.output/'fixed.json',result['fixed_momentum_checks'])
        print('Fixed-momentum collar, quadrature and cutoff checks complete.',flush=True)
    if not args.fixed_only:
        for t in args.t:
            for profile in args.profiles:
                row=integrate_patch(t,profile,patch,x_order=args.x_order,
                    block_orders=args.block_orders,radial_order=args.radial_order,
                    angular_order=args.angular_order,
                    progress=lambda v:print(json.dumps(v),flush=True))
                result['scans'].append(row)
                save(args.output/(f't{t:g}_{profile}.json'),row)
                save(args.output/'summary.json',result)
    if hashes!=source_hashes():
        raise RuntimeError('a numerical source changed during the pilot; rerun for a bound receipt')
    save(args.output/'summary.json',result)
    print('Saved '+str(args.output/'summary.json'),flush=True)


if __name__=='__main__':
    main()
