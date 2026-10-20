"""Sample-driven momentum placement at a fixed twelve-node budget."""
from __future__ import annotations

# paired_quadrature
# Kernel-weighted Gaussian rules in P for paired thermal corrections.
# 
# P^2 K_R is regular at zero, whereas P^2 K_infinity starts linearly in P.
# A polynomial rule in P resolves both endpoint expansions without treating
# the noncompact term as a square-root singularity in the quadrature variable.

import genus1_source_paths as bootstrap
bootstrap.install()
from pathlib import Path
import math
import numpy as np
from scipy.linalg import eigh_tridiagonal
from banks import make_spec
from kernels import primary_kernel_logs, height_log_integrals
from genus1_moduli_sampling import ShortPoint
from genus1_native_bank import digest


def weighted_P(log_density,decay,order,*,construction_order=384,tail_exponent=100.):
    if type(order) is not int or not 1<=order<=32 or construction_order<8*order or decay<=0:
        raise ValueError('resolved positive quadrature controls required')
    maximum=math.sqrt(tail_exponent/decay)
    x,w=np.polynomial.legendre.leggauss(construction_order)
    p=maximum*(x+1)/2;logmass=np.log(maximum*w/2)+log_density(p)
    peak=float(max(logmass));mass=np.exp(logmass-peak);norm=math.fsum(mass)
    coordinate=math.sqrt(decay)*p;v=np.sqrt(mass/norm)
    vectors=[];diagonal=[];offdiagonal=[]
    for j in range(order):
        vectors.append(v);product=coordinate*v;a=float(v@product);diagonal.append(a)
        residual=product-a*v
        if j:residual-=offdiagonal[-1]*vectors[-2]
        for _ in range(2):
            for q in vectors:residual-=(q@residual)*q
        if j<order-1:
            b=float(np.linalg.norm(residual))
            if not b>0:raise ArithmeticError('paired quadrature lost rank')
            offdiagonal.append(b);v=residual/b
    x,u=eigh_tridiagonal(diagonal,offdiagonal)
    nodes=x/math.sqrt(decay)
    if np.any(nodes<=0):raise ArithmeticError('nonpositive paired momentum')
    logs=peak+math.log(norm)+2*np.log(abs(u[0]))-log_density(nodes)
    return dict(nodes=nodes.tolist(),weights=np.exp(logs).tolist(),log_weights=logs.tolist(),
                measure='dP',order=order,polynomial_variable='P',
                construction_order=construction_order,construction_tail_exponent=tail_exponent,
                construction_maximum_P=maximum,physical_momentum_cutoff=False,
                requires_new_CFT_bank=True,scheme='thermal-paired-endpoint-P-v1')


def make_paired_spec(kin,case,order,*,construction_order=384,**kwargs):
    spec=make_spec(kin,case,order,construction_order=construction_order,**kwargs)
    kind=case['kind'];g=case['geometry']
    if kind=='primary':
        short=ShortPoint.from_record(g['short']);threshold=-math.log(g['policy']['q_max'])
        lower=max(threshold,short.s1,short.s2);axis=spec['design']['cut_axis']
        slope=spec['design']['axes'][axis]['dozz_threshold_slope'];curvature=min(slope,.5*lower)
        density=lambda p:2*np.log(p)+primary_kernel_logs(p,short,kin,threshold)[1]+curvature*p*p
        decay=2*lower-curvature
    elif kind in ('triple_corner','triple_scale_face'):
        axis=2;region=g['region'];tau1=g['tau1'];cutoff=g.get('cutoff',8.)
        lower=math.sqrt(1-tau1*tau1) if region=='bulk' else cutoff
        upper=cutoff if region=='bulk' else math.inf
        slope=spec['design']['axes'][axis]['dozz_threshold_slope'];curvature=min(slope,math.pi*lower)
        density=lambda p:2*np.log(p)+height_log_integrals(4*math.pi*p*p,lower,upper,kin.R,tau1)+curvature*p*p
        decay=4*math.pi*lower-curvature
    else:
        return spec
    rule=weighted_P(density,decay,order,construction_order=construction_order)
    rule.update(curvature=curvature,kinematics=kin.record())
    spec['grid']['rules'][axis]=rule
    spec['grid']['paired_quadrature_sources']={str(Path(__file__).relative_to(bootstrap.ROOT)):digest(__file__)}
    spec['design']['polynomial_variable']='P'
    return spec


def plan(output):
    from reference_campaign import cases
    from banks import register
    from kernels import ThermalKinematics
    from genus1_boundary_spectrum import publish
    entries=[]
    for tag,t,N in [('t035',.35,1),('t075',.75,1),('t095',.95,1),('fixed_N2',.75,2),('fixed_N3',.75,3)]:
        kin=ThermalKinematics(t,2*N/t,N)
        for label,case in cases(kin,tag.startswith('fixed')):
            if case['kind']=='pair':continue
            for order in (8,12,16):
                spec=make_paired_spec(kin,case,order);key=register(output,spec)
                entries.append(dict(tag=tag,label=label,kinematics=kin.record(),order=order,key=key,kind=case['kind']))
    result=dict(format='thermal-reference-campaign-v1',entries=entries,
        comparison_columns=['noncompact','thermal','thermal_correction'],no_matrix_model_in_grid_design=True,
        full_amplitude_certified=False,scope='Paired endpoint quadrature in P; pointwise momentum references.')
    publish(output/'reference_plan.json',result)
    return result


if __name__=='__main__':
    import json
    output=bootstrap.ROOT/'output/thermal_refinement_20260927/paired'
    print(json.dumps(dict(tasks=len(plan(output)['entries']))))


# placement
# Sample-driven momentum placement at a fixed twelve-node budget.
# 
# Only independent pilot data determine anchors. The matrix model is never a
# training target. All changed physical nodes require fresh CFT coefficients.

import genus1_source_paths as bootstrap
bootstrap.install()
from pathlib import Path
import json
import math
import numpy as np
from scipy.special import logsumexp
from thermal_saved_inputs import OUTPUT, SECTORS
from kernels import ThermalKinematics
from kernels import primary_kernel_logs
from banks import primary_case,register
from genus1_moduli_sampling import ShortPoint
from genus1_boundary_spectrum import publish
from genus1_native_bank import digest

MAXIMUM_NODES=12


def choose_anchors(coordinates,values,count=2):
    """Weighted k-medoids in log(1+short lengths), using the training pilot."""
    x=np.log1p(coordinates[:,:2]);weight=abs(values[:,2])
    if not weight.sum()>0:raise ValueError('pilot has no thermal-correction signal')
    weight=weight/weight.sum()
    distance=np.sum((x[:,None]-x[None,:])**2,axis=2)
    centers=[int(np.argmin(distance@weight))]
    while len(centers)<count:
        closest=np.min(distance[:,centers],axis=1)
        score=[np.sum(weight*np.minimum(closest,distance[:,j])) if j not in centers else math.inf
               for j in range(len(x))]
        centers.append(int(np.argmin(score)))
    for _ in range(30):
        assignment=np.argmin(distance[:,centers],axis=1);new=[]
        for j in range(count):
            candidates=np.flatnonzero(assignment==j)
            if not len(candidates):raise ValueError('empty weighted anchor cluster')
            costs=distance[np.ix_(candidates,candidates)]@weight[candidates]
            new.append(int(candidates[np.argmin(costs)]))
        if new==centers:break
        centers=new
    return [dict(pilot_index=i,coordinates=coordinates[i].tolist(),
                 correction_weight=float(weight[assignment==j].sum())) for j,i in enumerate(centers)]


def log_difference(a,b):
    """log |exp(a)-exp(b)|, preserving a small paired signal."""
    peak=np.maximum(a,b);gap=abs(a-b)
    with np.errstate(divide='ignore',invalid='raise'):
        return peak+np.log(-np.expm1(-gap))


def correction_rule(short,kin,slope,order=12,construction_order=384):
    if not 1<=order<=MAXIMUM_NODES:raise ValueError('at most 12 physical momentum nodes per axis')
    lower=max(-math.log(.05),short.s1,short.s2);curvature=min(slope,.5*lower)
    decay=2*lower-curvature;maximum=math.sqrt(100/decay)
    x,w=np.polynomial.legendre.leggauss(construction_order)
    p=maximum*(x+1)/2;logmass=np.log(maximum*w/2)
    def components(p):
        old,new=primary_kernel_logs(p,short,kin)
        base=2*np.log(p)+curvature*p*p
        return base+new,base+log_difference(old,new)
    thermal,delta=components(p)
    norms=(float(logsumexp(logmass+thermal)),float(logsumexp(logmass+delta)))
    if not all(math.isfinite(v) for v in norms):raise ValueError('unresolved correction proposal mass')
    def density(p):
        thermal,delta=components(p)
        return np.logaddexp(thermal-norms[0],delta-norms[1])-math.log(2)
    rule=weighted_P(density,decay,order,construction_order=construction_order)
    rule.update(scheme='thermal-and-correction-mixture-P-v1',curvature=curvature,
                component_log_normalizations=list(norms),mixture_weights=[.5,.5],
                kinematics=kin.record(),maximum_nodes_per_axis=MAXIMUM_NODES)
    return rule


def make_placement_spec(kin,short,variant='sample_anchor',order=12):
    if type(order) is not int or not 4<=order<=MAXIMUM_NODES:
        raise ValueError('the placement campaign permits only 4..12 physical nodes per axis')
    spec=make_paired_spec(kin,primary_case(short),order)
    if variant=='correction_mixture':
        axis=spec['design']['cut_axis']
        slope=spec['design']['axes'][axis]['dozz_threshold_slope']
        spec['grid']['rules'][axis]=correction_rule(short,kin,slope,order)
    elif variant!='sample_anchor':raise ValueError('unknown fixed-budget placement')
    spec['grid']['placement_sources']={str(Path(__file__).relative_to(bootstrap.ROOT)):digest(__file__)}
    spec['grid']['placement_variant']=variant
    spec['grid']['maximum_nodes_per_axis']=MAXIMUM_NODES
    if any(len(r['nodes'])!=order for r in spec['grid']['rules']):raise ValueError('physical node budget changed')
    spec['design']['placement_variant']=variant
    return spec


def plan_placement():
    folder=OUTPUT/'placement12';entries=[];anchors={};source={}
    for tag in ('t035','t075','t095'):
        pilot=OUTPUT/'sampling'/f'{tag}_N1/primary/pilot_p8.npz'
        with np.load(pilot) as a:selected=choose_anchors(a['coordinates'],a['values'])
        anchors[tag]=selected;source[tag]=digest(pilot)
        t=int(tag[1:])/100;kin=ThermalKinematics(t,2/t)
        for index,anchor in enumerate(selected):
            s1,s2,x1,x2=anchor['coordinates']
            for sector in SECTORS:
                short=ShortPoint(sector,s1,s2,x1,x2)
                # The alternative mixture isolates placement on the cut axis
                # at identical node count, moduli and CFT truncation.
                variants=('sample_anchor','correction_mixture') if sector==(0,2,1) else ('sample_anchor',)
                for variant in variants:
                    spec=make_placement_spec(kin,short,variant);key=register(folder,spec)
                    label=f'{variant}_anchor{index}_'+''.join(map(str,sector))
                    entries.append(dict(tag=tag,label=label,kinematics=kin.record(),order=12,key=key,
                                        kind='primary',placement_variant=variant,anchor_index=index))
    result=dict(format='thermal-reference-campaign-v1',entries=entries,
        comparison_columns=['noncompact','thermal','thermal_correction'],
        no_matrix_model_in_grid_design=True,full_amplitude_certified=False,
        scope='Fixed 12-node placement tests; anchors selected from independent pilot thermal-correction weights.',
        maximum_nodes_per_axis=MAXIMUM_NODES,anchors=anchors,pilot_sha256=source,
        frozen_before_independent_validation=True)
    publish(folder/'reference_plan.json',result)
    (OUTPUT/'node_budget_policy.json').write_text(json.dumps(dict(maximum_nodes_per_axis=12,
        method='fixed-count placement and independent validation',
        historical_higher_order_results_preserved=True,new_higher_order_evaluations_allowed=False),indent=2)+'\n')
    print(json.dumps(dict(tasks=len(entries),maximum_nodes_per_axis=12,anchors=anchors)))
    return result


if __name__=='__main__':plan_placement()
