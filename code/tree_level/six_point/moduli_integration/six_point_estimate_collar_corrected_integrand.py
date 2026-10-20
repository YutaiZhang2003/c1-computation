"""Named-bank scan adapter for explicit boundary marking and collar addback.

Native c-recursion arrays and collar elliptic arrays remain separately named.
An unresolved degeneration still has a diagnostic value, but prevents routing
qualification. It is never silently omitted or certified by a small radius.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
from pathlib import Path
import sys

import numpy as np

from six_point_route_collar_contributions import (
    CollarCFTEvaluator, TESTED_PATCH, TESTED_PATCH_NAME, NORMALIZATION, point_from_plan,
)
from six_point_boundary_strata import mark_degenerations, boundary_divisors, compatible_strata

POLICY = 'explicit-two-pair-collar-v1'


def settings(config):
    if not isinstance(config,dict) or config.get('policy') != POLICY:
        raise ValueError('collar routing requires the explicit-two-pair-collar-v1 policy')
    known = {'policy','momentum_profiles','elliptic_orders','x_order','dps',
             'onset','deep','relative_tolerance'}
    if set(config)-known:
        raise ValueError(f'unknown collar settings: {sorted(set(config)-known)}')
    result = dict(momentum_profiles=['refined','high'],elliptic_orders=[8,12],x_order=2,
                  dps=60,onset=.1,deep=.02,relative_tolerance=.003)
    result.update(config)
    profiles = result['momentum_profiles']
    rank = {'coarse':0,'fine':1,'refined':2,'high':3,'reference':4}
    if (len(profiles)<2 or any(p not in rank for p in profiles)
            or [rank[p] for p in profiles] != sorted({rank[p] for p in profiles})):
        raise ValueError('at least two explicitly increasing momentum profiles required')
    orders = result['elliptic_orders']
    if len(orders)<2 or any(isinstance(n,bool) or int(n)!=n or not 0<=n<=24 for n in orders) or orders!=sorted(set(orders)):
        raise ValueError('at least two increasing elliptic prefixes required')
    for key,minimum in (('x_order',2),('dps',40)):
        value=result[key]
        if isinstance(value,bool) or int(value)!=value or value<minimum:
            raise ValueError(f'{key} must be an integer >= {minimum}')
    if not 0<float(result['relative_tolerance'])<1:
        raise ValueError('relative_tolerance must lie in (0,1)')
    return result


def mark_plans(plans,config):
    cfg=settings(config)
    shape=(len(plans),len(plans[0].arrays['points']))
    inside,unhandled=np.zeros(shape,bool),np.zeros(shape,bool)
    records=[]
    for r,plan in enumerate(plans):
        rows=[]
        for i in range(shape[1]):
            result=mark_degenerations(point_from_plan(plan,i),onset=cfg['onset'],deep=cfg['deep'],
                                     tested_patches={TESTED_PATCH_NAME:TESTED_PATCH})
            inside[r,i]=bool(result['tested_patch_names'])
            unhandled[r,i]=bool(result['unhandled_marked_divisor_keys'])
            rows.append(result)
        records.append(rows)
    return dict(inside=inside,unhandled=unhandled,records=records,settings=cfg,
        registry=dict(divisors=[row.as_dict() for row in boundary_divisors()],
            strata={str(n):[row.as_dict() for row in compatible_strata(n)] for n in (1,2,3)}),
        patch=dict(name=TESTED_PATCH_NAME,**asdict(TESTED_PATCH)))


def source_hashes():
    import importlib
    names=('six_point_route_collar_contributions','six_point_analytic_collar_subtraction',
        'six_point_collar_counterterm_geometry','six_point_collar_block_evaluator',
        'sphere_four_point_c25','sphere_four_point_pillow_evaluate',
        'six_point_collar_block_evaluator','six_point_collar_momentum_quadrature',
        'six_point_endpoint_and_tail_quadrature',
        'six_point_integrate_collar_patch','six_point_boundary_strata',
        'liouville_torus','c1_paths')
    paths=[Path(__file__).resolve()]+[Path(importlib.import_module(n).__file__).resolve() for n in names]
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def evaluate_marked(plans,energies,marks,config):
    cfg=settings(config)
    profiles,orders=cfg['momentum_profiles'],cfg['elliptic_orders']
    shape=(len(energies),*marks['inside'].shape,len(profiles),len(orders))
    arrays={key:np.zeros(shape,complex) for key in ('full','leading','remainder','outer1')}
    addback=np.zeros((len(energies),len(profiles),len(orders)),complex)
    addback_lower_x=np.zeros_like(addback)
    records=[]
    for e,t in enumerate(energies):
        for p,profile in enumerate(profiles):
            evaluator=CollarCFTEvaluator(t,profile=profile,block_orders=orders,dps=cfg['dps'])
            integral=evaluator.prepare_addback(cfg['x_order'])
            lower=evaluator.prepare_addback(cfg['x_order']-1)
            for o,n in enumerate(orders):
                addback[e,p,o]=integral['by_order'][str(n)]
                addback_lower_x[e,p,o]=lower['by_order'][str(n)]
            rows=[]
            for r,i in np.argwhere(marks['inside']):
                result=evaluator.evaluate(point_from_plan(plans[r],i))
                for o,n in enumerate(orders):
                    for key in arrays:
                        arrays[key][e,r,i,p,o]=result['by_order'][str(n)][key]
                rows.append(dict(replicate=int(r),point=int(i),diagnostics=result['diagnostics']))
            records.append(dict(t=float(t),profile=profile,points=rows,
                momentum_nodes=len(evaluator.ps),normalization_id=NORMALIZATION,
                spectral_metadata=evaluator.grid['metadata']))
            print(f'collar t={t} profile={profile} points={len(rows)} nodes={len(evaluator.ps)}',flush=True)
            evaluator._slices.cache_clear()
            del evaluator
    full=arrays['full'];denom=np.maximum(abs(full[:,:,:,-1,-1]),np.finfo(float).tiny)
    momentum_change=abs(full[:,:,:,-2,-1]-full[:,:,:,-1,-1])/denom
    block_change=abs(full[:,:,:,-1,-2]-full[:,:,:,-1,-1])/denom
    outer_change=abs(arrays['outer1'][:,:,:,-1,-1]-full[:,:,:,-1,-1])/denom
    prefix_pass=(momentum_change<cfg['relative_tolerance']) & (block_change<cfg['relative_tolerance']) & (outer_change<cfg['relative_tolerance'])
    # Weighted absolute prefix flags are kept separately; signed cancellation
    # cannot pass the empirical gate if the absolute weighted increment fails.
    for row in records:
        if row['profile']!=profiles[-1]:continue
        e=list(energies).index(row['t'])
        for point in row['points']:
            change=point['diagnostics']['absolute_weighted_order_change'][str(orders[-2])]
            r,i=point['replicate'],point['point']
            prefix_pass[e,r,i] &= change is not None and change<cfg['relative_tolerance']
    add_denom=np.maximum(abs(addback[:,-1,-1]),np.finfo(float).tiny)
    add_momentum=abs(addback[:,-2,-1]-addback[:,-1,-1])/add_denom
    add_middle=abs(addback_lower_x[:,-1,-1]-addback[:,-1,-1])/add_denom
    add_block=abs(addback[:,-1,-2]-addback[:,-1,-1])/add_denom
    add_pass=(add_momentum<cfg['relative_tolerance']) & (add_middle<cfg['relative_tolerance']) & (add_block<cfg['relative_tolerance'])
    return dict(arrays=arrays,addback=addback,addback_lower_x=addback_lower_x,
        momentum_change=momentum_change,block_change=block_change,outer_change=outer_change,
        prefix_pass=prefix_pass,addback_prefix_pass=add_pass,
        addback_momentum_change=add_momentum,addback_middle_change=add_middle,
        addback_block_change=add_block,records=records,settings=cfg)


def hybridize(native_values,marks,collar):
    """Keep every sample; insert collar result only on its physical support."""
    mask=marks['inside'][None,:,:,None,None]
    full=collar['arrays']['full'][:,:,:,-1,-1,None,None]
    remainder=collar['arrays']['remainder'][:,:,:,-1,-1,None,None]
    addback=collar['addback'][:,-1,-1,None,None,None,None]
    return (np.where(mask,full,native_values),
            np.where(mask,remainder,native_values)+addback)
