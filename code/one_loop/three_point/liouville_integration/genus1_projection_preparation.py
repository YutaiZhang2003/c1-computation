"""Explicit prepare/build/evaluate path for kernel-specific Liouville grids.

Pair faces, comb pair faces, scale/corner heights and H4 primary tubes get
their own physical geometry. Sampling never builds a missing coefficient.
"""
from __future__ import annotations

from genus1_source_paths import ROOT as _SOURCE_ROOT, source_path as _source_path

from pathlib import Path
import re
import math
import numpy as np

from genus1_three_point_channel_atlas import make_atlas
from genus1_native_bank import source_manifest, write_json, read_json, digest, build, BankResolver
from genus1_three_point_momentum import AxisRule, NodeCatalog, tensor_rule, fingerprint
from genus1_native_bank import SharedNativeBank, FORMAT as BANK_FORMAT
from genus1_projection_momentum import kinematics, pair_rules, comb_pair_rules, height_design, projection_rules, primary_rules, full_rules, primary_sampling_controls, total_sampling_controls, check_total_nodes, short_refinement_controls, resolved_primary_sampling

FORMAT='genus1-projection-momentum-preparation-v1'


def geometry(case):
    g=dict(case['geometry'])
    for key in ('tau','delta_total','shape','w1','w2'):
        if key in g:
            if not isinstance(g[key],(list,tuple)) or len(g[key])!=2:
                raise ValueError('complex geometry must be explicit [real, imaginary] pairs')
            g[key]=complex(*g[key])
    if g.get('pair') is not None:g['pair']=tuple(g['pair'])
    return g


def spec(atlas, case, orders, tails, width_scales=(1.,1.,1.), *, primary_sampling=None,momentum_sampling=None,short_refinement=None):
    kind=case['kind']; g=geometry(case); co=case['coefficient_orders']
    if primary_sampling is not None and momentum_sampling is not None:
        raise ValueError('choose one explicit momentum sampling policy')
    budget=None if momentum_sampling is None else total_sampling_controls(momentum_sampling)
    if short_refinement is not None and (kind!='primary' or budget is None):
        raise ValueError('short refinement requires a primary total-node budget')
    if primary_sampling is not None and kind!='primary':
        raise ValueError('primary sampling controls require a primary kernel')
    if len(co)!=3 or any(type(n) is not int or not 0<=n<=10 for n in co):
        raise ValueError('three explicit native coefficient orders in 0..10 required')
    if kind=='full':
        kin,design,rules=full_rules(atlas,case['geometry'],orders=orders,tail_orders=tails,sampling=budget)
    elif kind=='pair':
        if co[0]!=0:raise ValueError('integrated pair has no bridge descendants')
        design,rules=pair_rules(atlas,**g,orders=orders,tail_orders=tails,sampling=budget); kin=kinematics(atlas,'pair_ope',g['pair'])
    elif kind=='comb_pair_face':
        if co[0]!=0:raise ValueError('pair-face bridge is projected to level zero')
        design,rules=comb_pair_rules(atlas,**g,orders=orders,tail_orders=tails,sampling=budget);kin=kinematics(atlas,'comb_ope',g['pair'])
    elif kind in ('triple_scale_face','triple_corner'):
        if co[1]!=0 or (kind=='triple_corner' and co[0]!=0):
            raise ValueError('build only the retained scale/corner coefficient axes')
        if kind=='triple_scale_face':
            shape=g.get('shape',0j)
            if not 0<abs(shape)<1 or shape.real>.5:
                raise ValueError('scale-face preparation requires an admissible explicit shape')
        elif 'theta' not in g or not math.isfinite(g['theta']):
            raise ValueError('corner preparation requires an explicit finite angle')
        args={k:v for k,v in g.items() if k not in ('shape','theta')}
        design=height_design(atlas,**args)
        rules=projection_rules(design,orders,tail_orders=(None,tails[1],tails[2]),sampling=budget)
        kin=kinematics(atlas,'comb_ope',g['pair'])
    elif kind=='primary':
        from genus1_moduli_sampling import ShortPoint
        from genus1_analytic_boundary_layer import BoundaryPolicy
        if co!=[0,0,0]:raise ValueError('H preparation needs spectral data only; native orders must be zero')
        short=ShortPoint.from_record(g['short']); policy=BoundaryPolicy(**g['policy'])
        design,rules=primary_rules(atlas,short,orders=orders,
            lower_length=max(policy.length,short.s1,short.s2),tail_orders=tails,sampling=primary_sampling,
            momentum_sampling=budget,short_refinement=short_refinement)
        kin=kinematics(atlas,'necklace')
    else:
        raise ValueError('unknown projection kernel')
    if (len(width_scales)!=3 or any(isinstance(v,bool) or not math.isfinite(v) or v<=0 for v in width_scales)):
        raise ValueError('three positive finite momentum width scales required')
    if any(v!=1 for v in width_scales):
        # An exact change of integration variable P=s*x on [0,infinity).
        # Dilation probes the entire coupled tensor rule, with its dP Jacobian.
        # Keep the original proposal metadata and record the transformation.
        rules=[dict(r,nodes=(np.asarray(r['nodes'])*s).tolist(),
                    weights=(np.asarray(r['weights'])*s).tolist(),
                    log_weights=np.log(np.asarray(r['weights'])*s).tolist(),coordinate_dilation=float(s))
               for r,s in zip(rules,width_scales)]
        design=dict(design,width_scales=list(width_scales))
    if budget is not None:
        resolved=design.get('momentum_sampling',budget)
        check_total_nodes(rules,resolved)
        design=dict(design,momentum_sampling=resolved)
    return kin,design,rules


def quadrature(rules):
    return tuple(AxisRule(np.asarray(r['nodes']),np.asarray(r['weights']),
        {k:v for k,v in r.items() if k not in ('nodes','weights','log_weights')}) for r in rules)


def prepare(root, *, t, cases, momentum_orders=None, tail_orders=None,
            elliptic_order=4, special_dps=40, chunk_size=16, width_scales=(1.,1.,1.),
            primary_sampling=None,momentum_sampling=None,short_refinement=None,frozen_cases=None):
    root=Path(root)
    if (root/'plan.json').exists():raise FileExistsError('preserve the frozen projection plan')
    # Fresh default preparations use eight TOTAL nodes. Explicit legacy orders
    # remain available for independent diagnostic references.
    if momentum_orders is None and tail_orders is None and primary_sampling is None and momentum_sampling is None:
        momentum_sampling=total_sampling_controls({})
    momentum_orders=(8,8,8) if momentum_orders is None else momentum_orders
    tail_orders=(3,3,3) if tail_orders is None else tail_orders
    cases=tuple(cases)
    if (not cases or len(momentum_orders)!=3 or any(type(n) is not int or n<1 for n in momentum_orders)
            or type(chunk_size) is not int or chunk_size<1 or type(special_dps) is not int or special_dps<30
            or type(elliptic_order) is not int or not 2<=elliptic_order<=24):
        raise ValueError('explicit positive preparation controls required')
    has_primary=any(case.get('kind')=='primary' for case in cases)
    if primary_sampling is not None and momentum_sampling is not None:
        raise ValueError('choose one explicit momentum sampling policy')
    budget=None if momentum_sampling is None else total_sampling_controls(momentum_sampling)
    if short_refinement is not None and (not has_primary or budget is None):
        raise ValueError('short refinement requires a primary total-node budget')
    refinement=None if short_refinement is None else short_refinement_controls(short_refinement)
    if primary_sampling is not None and not has_primary:
        raise ValueError('primary sampling controls require a primary case')
    sampling=primary_sampling_controls(primary_sampling) if has_primary and budget is None else None
    atlas=make_atlas(t,special_dps=special_dps); catalog=NodeCatalog(); records={}
    sources=source_manifest()
    if frozen_cases is not None and set(frozen_cases)!={c['id'] for c in cases}:
        raise ValueError('frozen grids must cover precisely the requested cases')
    for case in cases:
        if set(case)!={'id','kind','geometry','coefficient_orders'}:
            raise ValueError('each case requires id, kind, geometry and coefficient_orders')
        label=case['id']
        if not isinstance(label,str) or not re.fullmatch(r'[A-Za-z0-9_]+',label) or label in records:
            raise ValueError('distinct alphanumeric case ids required')
        controls=dict(momentum_orders=list(momentum_orders),tail_orders=list(tail_orders),
                      width_scales=list(width_scales))
        if budget is not None:controls['momentum_sampling']=budget
        elif case['kind']=='primary':controls['primary_sampling']=sampling
        if case['kind']=='primary' and refinement is not None:controls['short_refinement']=refinement
        if frozen_cases is None:
            kin,design,rules=spec(atlas,case,tuple(momentum_orders),tuple(tail_orders),width_scales,
                primary_sampling=sampling if case['kind']=='primary' else None,momentum_sampling=budget,
                short_refinement=refinement if case['kind']=='primary' else None)
        else:
            # Consume the frozen physical nodes verbatim. LAPACK/BLAS builds
            # can differ by a few ulps when reconstructing a Gaussian rule;
            # recomputing it here breaks provenance and cross-host resume.
            from genus1_native_bank import decode_identity
            from genus1_native_bank import identity
            row=frozen_cases[label];grid=row['grid'];g=geometry(case)
            channel='necklace' if case['kind']=='primary' else 'pair_ope' if case['kind']=='pair' else 'comb_ope'
            expected=kinematics(atlas,channel,g.get('pair',(0,1)))
            if (case['kind']=='full' or row['case']!=case or grid['identity']!=identity(expected) or
                    grid['t']!=float(t) or grid['coefficient_orders']!=case['coefficient_orders'] or
                    grid['source_sha256']!=fingerprint(sources) or row['source_files']!=sources or
                    grid['special_dps']!=special_dps or row['controls']!=controls or
                    (case['kind']=='primary' and grid['elliptic_order']!=elliptic_order)):
                raise ValueError('frozen spectral grid does not implement the requested preparation')
            kin=decode_identity(grid['identity']);design=row['design'];rules=grid['rules']
            if len(rules)!=3:raise ValueError('three frozen momentum axes required')
            # AxisRule validates finite, positive physical nodes and dP weights.
            quadrature(rules)
        if budget is not None:
            resolved=budget
            if case['kind']=='primary':
                from genus1_moduli_sampling import ShortPoint
                resolved,axes=resolved_primary_sampling(atlas,ShortPoint.from_record(case['geometry']['short']),budget,refinement)
                if design.get('refined_momentum_axes')!=axes:raise ValueError('frozen regional allocation changed')
            if design.get('momentum_sampling')!=resolved:raise ValueError('frozen total-node allocation changed')
            check_total_nodes(rules,resolved)
        elif case['kind']=='primary' and 'node_budget' in sampling:
            if any(len(rule['nodes'])>sampling['node_budget'] for rule in rules):
                raise ValueError('primary momentum node budget exceeded by frozen nodes')
        request=catalog.request(kin,quadrature(rules),case['coefficient_orders'])
        records[label]=dict(case=case,design=design,rules=rules,request_id=request)
    files=dict(sources)
    command=_source_path('code/one_loop/three_point/other_scripts/genus1_projection_grid.py')
    files['code/one_loop/three_point/other_scripts/genus1_projection_grid.py']=digest(command)
    plan=dict(format=FORMAT,t=float(t),special_dps=special_dps,coefficient_chunk_size=chunk_size,
        source_files=files,catalog=catalog.record(),cases=records,
        momentum_orders=list(momentum_orders),tail_orders=list(tail_orders),elliptic_order=elliptic_order,
        width_scales=list(width_scales),
        full_amplitude_certified=False,numerical_convergence_certified=False)
    if budget is not None:plan['momentum_sampling']=budget
    elif has_primary:plan['primary_sampling']=sampling
    if refinement is not None:plan['short_refinement']=refinement
    write_json(root/'plan.json',plan)
    return plan


def load_bank(root, case_id, *, resolver=None):
    root=Path(root); resolver=BankResolver(root) if resolver is None else resolver
    if resolver.root.resolve()!=root.resolve() or resolver.plan.get('format')!=FORMAT:
        raise ValueError('dedicated kernel-specific preparation required')
    row=resolver.plan['cases'][case_id]; raw=resolver.get(row['request_id'])
    p,lw=tensor_rule(quadrature(row['rules']))
    if not np.array_equal(p,raw.arrays['momenta']) or not np.array_equal(lw,raw.arrays['log_weights']):
        raise ValueError('frozen geometry grid differs from the prepared spectral data')
    meta=dict(format=BANK_FORMAT,target_blind=True,identity=raw.metadata['identity'],
        orders=row['case']['coefficient_orders'],source_manifest_sha256=fingerprint(resolver.plan['source_files']),
        projection_only=row['case']['kind']!='full',preparation_kernel=row['case']['kind'],
        preparation_plan_sha256=resolver.plan_sha,request_id=row['request_id'],
        quadrature=dict(measure='product_dP',rules=row['rules'],threshold_design=row['design']),
        numerical_convergence_certified=False)
    return SharedNativeBank(meta,raw.arrays,expected_source_sha256=meta['source_manifest_sha256'])


def export(root, case_id):
    root=Path(root); bank=load_bank(root,case_id); path=root/'exports'/(case_id+'.npz')
    if path.exists() and path.with_suffix('.json').exists():
        from genus1_boundary_banks import load_native
        old=load_native(path)
        if old.metadata['preparation_plan_sha256']!=bank.metadata['preparation_plan_sha256']:
            raise ValueError('export belongs to a different frozen plan')
        for k in bank.arrays:
            if not np.array_equal(old.arrays[k],bank.arrays[k]):raise ValueError('changed exported spectral data')
    else:bank.freeze(path)
    return path


def build_elliptic(root, case_id, *, max_new_nodes):
    from genus1_boundary_banks import load_native, prepare_elliptic_grid, EllipticGridBank
    root=Path(root); resolver=BankResolver(root); row=resolver.plan['cases'][case_id]
    if row['case']['kind']!='primary':raise ValueError('elliptic preparation is only for a primary tube')
    native=load_native(export(root,case_id)); path=root/'elliptic'/(case_id+'.npz')
    receipt=root/'elliptic'/(case_id+'.checked.json')
    if receipt.exists() and receipt.with_suffix(receipt.suffix+'.sha256').exists():
        record=read_json(receipt)
        if record['plan_sha256']!=resolver.plan_sha:raise ValueError('wrong elliptic preparation plan')
        return EllipticGridBank.load(path,receipt_sha256=record['receipt_sha256'])
    if path.exists() and path.with_suffix('.json').exists():
        # Recovery after data/receipt publication but before the checked index.
        bank=EllipticGridBank.load(path,receipt_sha256=digest(path.with_suffix('.json')))
        if (bank.metadata['source_native_data_sha256']!=native.metadata['data_sha256']
                or bank.metadata['order']!=resolver.plan['elliptic_order']
                or bank.metadata['cut_axis']!=row['design']['cut_axis']):
            raise ValueError('unindexed elliptic bank has different inputs')
    else:
        bank=prepare_elliptic_grid(path,native,cut_axis=row['design']['cut_axis'],
            order=resolver.plan['elliptic_order'],max_new_nodes=max_new_nodes,
            checkpoint_directory=root/'elliptic_nodes'/case_id)
    for k in ('momenta','log_weights','structure_logs'):
        if not np.array_equal(bank.arrays[k],native.arrays[k]):raise ValueError('elliptic preparation changed the grid')
    write_json(receipt,dict(plan_sha256=resolver.plan_sha,receipt_sha256=digest(path.with_suffix('.json'))))
    return bank


def evaluate(root, case_id):
    root=Path(root); resolver=BankResolver(root); plan=resolver.plan
    row=plan['cases'][case_id]; case=row['case']; g=geometry(case); kind=case['kind']
    bank=load_bank(root,case_id,resolver=resolver); atlas=make_atlas(plan['t'],special_dps=plan['special_dps'])
    if kind=='full':
        from genus1_channel_evaluation import full_values
        values,report=full_values(bank,atlas,g,bank.orders);checks=report['prefixes']
    elif kind=='pair':
        from genus1_channel_evaluation import pair_projection_values
        values,report=pair_projection_values(bank,atlas,dict(g,integrated=True),bank.orders)
        checks=report['prefixes']
    elif kind=='comb_pair_face':
        from genus1_comb_collar import evaluate_projected_comb as evaluate_comb_collar
        values,report=evaluate_comb_collar(bank,atlas,term='pair_face',**g,diagnostics=True)
        checks=report['prefixes']
    elif kind in ('triple_scale_face','triple_corner'):
        from genus1_integrated_scale_collar import PreparedBank, SCOPE, evaluate as contraction
        meta=dict(evaluation_scope=SCOPE,orders=list(bank.orders),rules=row['rules'],
            pair=list(g.pop('pair')),region=g['region'],external_weights=bank.metadata['identity']['external_weights'])
        arrays={k:v for k,v in bank.arrays.items() if k!='coefficients'}
        # Catalog weights are computed from the saved dP weights themselves.
        meta['rules']=[dict(r,log_weights=np.log(r['weights']).tolist()) for r in row['rules']]
        arrays['projection_coefficients']=bank.arrays['coefficients'][:,:,0,:]
        prepared=PreparedBank(arrays,meta,dict(plan_sha256=resolver.plan_sha))
        values,checks=contraction(prepared,atlas,term=kind,**g)
    else:
        from genus1_boundary_banks import EllipticGridBank, EllipticPrimary, ContinuedCoordinates
        from genus1_moduli_sampling import ShortPoint
        from genus1_analytic_boundary_layer import BoundaryPolicy
        record=read_json(root/'elliptic'/(case_id+'.checked.json'))
        if record['plan_sha256']!=resolver.plan_sha:raise ValueError('wrong elliptic plan')
        ellipse=EllipticGridBank.load(root/'elliptic'/(case_id+'.npz'),receipt_sha256=record['receipt_sha256'])
        for k in ('momenta','log_weights','structure_logs'):
            if not np.array_equal(ellipse.arrays[k],bank.arrays[k]):raise ValueError('elliptic grid differs from the prepared request')
        primary=EllipticPrimary(atlas,{row['design']['cut_axis']:ellipse},ContinuedCoordinates(),orders=(2,plan['elliptic_order']))
        result=primary(ShortPoint.from_record(g['short']),BoundaryPolicy(**g['policy']),16)
        values=[result.value];checks=[]
    if not all(c['passed'] for c in checks):raise FloatingPointError('unresolved projection arithmetic')
    return dict(case=case_id,kind=kind,values=[[complex(v).real,complex(v).imag] for v in values],
        momentum_nodes=len(bank.arrays['momenta']),axis_node_counts=[len(r['nodes']) for r in row['rules']],
        coefficient_orders=list(bank.orders),elliptic_order=plan['elliptic_order'] if kind=='primary' else None,
        plan_sha256=resolver.plan_sha,numerical_convergence_certified=False,full_amplitude_certified=False,
        scope='complete momentum rule at explicit remaining geometry; no full moduli integral')


# Momentum preparation command
# Prepare kernel-specific Liouville grids and build their checked CFT banks.

from pathlib import Path
import argparse
import json
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('prepare','build','build-elliptic','evaluate','export'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cases',type=Path)
    p.add_argument('--case')
    p.add_argument('--t',type=float,default=.25)
    p.add_argument('--momentum-orders',type=int,nargs=3,default=(8,8,8),
                   help='legacy axis orders, used only with --reference-only')
    p.add_argument('--bridge-tail-order',type=int,default=3)
    p.add_argument('--scale-tail-order',type=int,default=3)
    p.add_argument('--loop-tail-order',type=int,default=3)
    p.add_argument('--reference-only',action='store_true',help='explicit high-order diagnostic; disables the production total-node policy')
    p.add_argument('--head-orders',type=int,nargs=3,default=(5,5,5))
    p.add_argument('--split-scales',type=float,nargs=3,default=(1.5,1.5,1.5))
    p.add_argument('--tail-width-scales',type=float,nargs=3,default=(.25,.25,.25))
    p.add_argument('--refine-short-below',type=float,
                   help='use 6 head + 3 tail nodes only on primary short axes below this length; cut remains 8')
    p.add_argument('--elliptic-order',type=int,default=4)
    p.add_argument('--primary-head-order',type=int,
                   help='short-head order only; integrated cut order stays in --momentum-orders')
    p.add_argument('--primary-split-scale',type=float)
    p.add_argument('--primary-tail-width-scale',type=float)
    p.add_argument('--primary-node-budget',type=int,
                   help='legacy maximum TOTAL nodes on every axis, including the cut')
    p.add_argument('--max-new-nodes',type=int,default=16)
    p.add_argument('--start',type=int,default=0);p.add_argument('--stop',type=int)
    a=p.parse_args()
    if a.stage=='prepare':
        if a.cases is None:p.error('prepare requires --cases with explicit kernel geometries and coefficient orders')
        sampling={k:v for k,v in dict(head_order=a.primary_head_order,split_scale=a.primary_split_scale,
            tail_width_scale=a.primary_tail_width_scale,node_budget=a.primary_node_budget).items() if v is not None}
        if sampling and not a.reference_only:p.error('legacy primary controls require --reference-only')
        if not a.reference_only and (tuple(a.momentum_orders)!=(8,8,8) or
                (a.bridge_tail_order,a.scale_tail_order,a.loop_tail_order)!=(3,3,3)):
            p.error('legacy axis orders require --reference-only; production uses eight total nodes')
        if a.reference_only and a.refine_short_below is not None:p.error('regional refinement requires the production total-node policy')
        budget=None if a.reference_only else dict(total_nodes=8,head_orders=a.head_orders,
            split_scales=a.split_scales,tail_width_scales=a.tail_width_scales)
        refinement=None if a.refine_short_below is None else dict(maximum_short_length=a.refine_short_below)
        plan=prepare(a.output,t=a.t,cases=json.loads(a.cases.read_text()),momentum_orders=tuple(a.momentum_orders),
            tail_orders=(a.bridge_tail_order,a.scale_tail_order,a.loop_tail_order),elliptic_order=a.elliptic_order,
            primary_sampling=sampling or None,momentum_sampling=budget,short_refinement=refinement)
        result=dict(cases=list(plan['cases']),unique_nodes=len(plan['catalog']['nodes']))
    elif a.stage=='build':
        build(a.output,start=a.start,stop=a.stop)
        result=dict(requested_chunk_range_complete=True)
    else:
        if a.case is None:p.error('this stage requires --case')
        if a.stage=='build-elliptic':
            b=build_elliptic(a.output,a.case,max_new_nodes=a.max_new_nodes)
            result=dict(elliptic_order=b.metadata['order'],momentum_nodes=len(b.arrays['momenta']))
        elif a.stage=='evaluate':result=evaluate(a.output,a.case)
        else:result=dict(path=str(export(a.output,a.case).resolve()))
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
