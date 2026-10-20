"""Paired thermal corrections for the exact energies in draft Figure 4(c).

The zero-temperature baseline is immutable.  The matrix model is consulted
only by the final exporter, never by the momentum or moduli grid design.
All fresh physical momentum rules have at most twelve nodes per axis.
"""
import genus1_source_paths as bootstrap
bootstrap.install()
import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import qmc
from scipy.special import erfcx, logsumexp

from banks import make_spec, register, build_reference, height_case
from evaluation import Evaluator, columns
from genus1_boundary_spectrum import read, publish
from genus1_native_bank import digest
from genus1_moduli_sampling import ShortPoint, regular_mask
from genus1_analytic_boundary_layer import MIN_HEIGHT
from genus1_three_point_channel_atlas import nearest_torus_displacement
from kernels import ThermalKinematics
from kernels import lattice
from paired_quadrature import make_paired_spec, weighted_P
from paired_quadrature import make_placement_spec as primary_spec, log_difference
from kernels import height_log_integrals
from radius_scan import Proposal, primary_coordinates, pair_coordinates, SECTORS
from remainders import comb_term
from study_pair_sampling import features, summarize

ROOT = bootstrap.ROOT
OUTPUT = ROOT / 'output/thermal_draft_curve20_20260928'
BANKS = OUTPUT
PRIOR = ROOT / 'output/thermal_refinement_20260927'
BASELINE = ROOT / 'Machine note/one_loop/figures/torus_1to2_numeric_rms.dat'
TAU = 2 * math.pi
SEEDS = {'primary': 6280101, 'pair-bulk': 6280201, 'pair-tail': 6280301}


def tag_of(t):
    return f't{round(1000*t):04d}'


def baseline():
    data = np.loadtxt(BASELINE, skiprows=1)
    # Later rows at the same energy are the later refinements in Figure 4(c).
    selected = {}
    for row, (t, value, rms) in enumerate(data):
        selected[float(t)] = dict(t=float(t), R=2/float(t), baseline_native=float(value),
            baseline_gs3=float(value*TAU**3), source_row=row+2,
            plotted=bool(.25-1e-12 <= t <= .95+1e-12))
    rows = sorted(selected.values(), key=lambda r:r['R'])
    assert len(data)==22 and len(rows)==20 and sum(r['plotted'] for r in rows)==15
    return dict(source=str(BASELINE.relative_to(ROOT)), sha256=digest(BASELINE),
        source_rows=data.tolist(), selected=rows, duplicate_policy='last refinement at each energy',
        plotting_range=[40/19, 8.], error_bars=False, matrix_used_for_planning=False)


def medoids(x, weights, count):
    weights=np.asarray(weights, float);weights=weights/weights.sum()
    distance=np.sum((x[:,None]-x[None,:])**2,axis=2)
    centers=[int(np.argmin(distance@weights))]
    while len(centers)<count:
        closest=np.min(distance[:,centers],axis=1)
        costs=[np.sum(weights*np.minimum(closest,distance[:,i])) if i not in centers
               else math.inf for i in range(len(x))]
        centers.append(int(np.argmin(costs)))
    for _ in range(30):
        assignment=np.argmin(distance[:,centers],axis=1);new=[]
        for k in range(count):
            candidates=np.flatnonzero(assignment==k)
            if not len(candidates):raise ArithmeticError('empty placement cluster')
            costs=distance[np.ix_(candidates,candidates)]@weights[candidates]
            new.append(int(candidates[np.argmin(costs)]))
        if new==centers:break
        centers=new
    return centers


def make_collision_mesh():
    """Frequency-independent remaining-moduli cubature, with source hashes."""
    old=ROOT/'output/diagnostics/genus1_curve10_primary_384_20260923/evidence_snapshot'
    mesh=read(old/'mesh.json');design=read(old/'energies/t075/design.json')
    terms={c['cell']['id']:c['cell']['term'] for c in mesh['cells']}
    batches=[];pins={}
    for desc in mesh['batches']:
        term=terms[desc['cell_id']]
        if ':triple_' not in term:continue
        source=old/'mesh'/desc['path'];row=read(source)
        row=dict(row);row['region']=term.split(':')[0]
        for sample in row['samples'].values():
            for request,key in zip(sample['requests'],sample['grids']):
                path=ROOT/'output/thermal_one_loop_20260927/inputs/t075/grids'/design['grid_map'][key]/'spec.json'
                spec=read(path)
                request['geometry']['pair']=spec['case']['geometry']['pair']
                request['geometry'].pop('graph',None)
        batches.append(row);pins[str(source.relative_to(ROOT))]=digest(source)
    publish(OUTPUT/'collision_mesh.json',dict(batches=batches,source_sha256=pins,
        original_mesh_sha256=digest(old/'mesh.json')))


def height_correction_rule(spec,kin):
    """Positive correction-weighted P rule for R>=10.

    The m=0 winding sum integrated from zero to infinity gives
    Delta K(P)=1/[P expm1(4 pi R P)]. Its lower-height extension is
    exponentially suppressed by exp(-pi R^2/8). This is a PROPOSAL only;
    the evaluator still contracts both exact finite-lower-limit kernels.
    """
    maximum=80/(4*math.pi*kin.R)
    def density(p):
        x=4*math.pi*kin.R*p
        return np.log(p)-x-np.log(-np.expm1(-x))
    rule=weighted_P(density,100/maximum**2,12)
    rule.update(curvature=0.,scheme='height-correction-winding-P-v1',
        proposal='P / expm1(4*pi*R*P)',kinematics=kin.record(),
        exact_physical_height_kernel_retained=True,
        proposal_lower_extension_exponent=-math.pi*kin.R**2/8)
    spec['grid']['rules'][2]=rule
    spec['grid']['draft_height_rule_sha256']=digest(__file__)
    return spec


def plan():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    b=baseline();publish(OUTPUT/'baseline.json',b)
    make_collision_mesh()
    anchors={}
    for folder in ('placement12','production12'):
        anchors.update(read(PRIOR/folder/'reference_plan.json')['anchors'])
    old_t=np.array([int(tag[1:])/100 for tag in anchors])
    old_tags=list(anchors);entries=[];sources={}
    for row in b['selected']:
        if not row['plotted']:continue
        kin=ThermalKinematics(row['t'],row['R']);tag=tag_of(kin.t)
        nearest=old_tags[int(np.argmin(abs(old_t-kin.t)))]
        prior_R=2/(int(nearest[1:])/100)
        primary_anchors=[]
        for a in anchors[nearest]:
            coord=np.array(a['coordinates'])
            if kin.R>prior_R:
                # Extend the long coordinate to the actual thermal length.
                j=int(np.argmax(coord[:2]));coord[j]*=(kin.R/prior_R)**2
            primary_anchors.append(coord.tolist())
        if kin.R>8:
            primary_anchors.append([1.2,2.4,.7,2.1])
        for j,coordinate in enumerate(primary_anchors):
            for sector in SECTORS:
                short=ShortPoint(sector,*coordinate)
                variant='correction_mixture' if kin.R>8 else 'sample_anchor'
                spec=primary_spec(kin,short,variant=variant,order=12)
                key=register(OUTPUT,spec)
                entries.append(dict(tag=tag,label=f'primary_{j}_'+''.join(map(str,sector)),
                    key=key,kind='primary',order=12,kinematics=kin.record(),
                    anchor=coordinate,prior_pilot=nearest,placement_variant=variant))
        for region,count in (('bulk',3),('tail',6)):
            pilot=PRIOR/'sampling'/f'{nearest}_N1'/f'pair-{region}'/'pilot_p8.npz'
            with np.load(pilot) as a:
                unit=a['unit'];weight=abs(a['values'][:,2])
            # The prior independent pilot supplies a starting distribution;
            # new-frequency CFT values train the actual sampling proposal.
            coords,_=pair_coordinates(unit,kin.R,region)
            f=np.array([features(complex(x[0],x[1]),TAU*(x[2]+complex(x[0],x[1])*x[3])) for x in coords])
            idx=medoids(f,weight+weight.max()*1e-5,count)
            sources[str(pilot.relative_to(ROOT))]=digest(pilot)
            for j,i in enumerate(idx):
                tau1,T,x,y=coords[i];tau=complex(tau1,T);total=TAU*(x+tau*y)
                for pair in ((0,1),(0,2)):
                    case=dict(id='bank',kind='pair',coefficient_orders=[0,2,3],
                        geometry=dict(pair=list(pair),radius=.125,tau=[tau.real,tau.imag],
                                      delta_total=[total.real,total.imag]))
                    spec=make_spec(kin,case,12);key=register(OUTPUT,spec)
                    entries.append(dict(tag=tag,label=f'pair_{region}_{j}_{pair[1]}',key=key,
                        kind='pair',region=region,order=12,kinematics=kin.record(),
                        pair=list(pair),feature=f[i].tolist(),prior_pilot=nearest))
        for region in ('bulk','tail'):
            for pair in ((0,1),(0,2)):
                for kind in ('triple_corner','triple_scale_face'):
                    case=height_case(kind,pair=pair,region=region,tau1=.5)
                    if kind=='triple_scale_face':case['coefficient_orders']=[2,0,3]
                    spec=make_paired_spec(kin,case,12)
                    if region=='tail' and kin.R>8:spec=height_correction_rule(spec,kin)
                    key=register(OUTPUT,spec)
                    entries.append(dict(tag=tag,label=f'{kind}_{region}_{pair[1]}',key=key,
                        kind=kind,region=region,pair=list(pair),order=12,kinematics=kin.record()))
        for j,T in enumerate((2.,8.,max(24.,kin.R**2),max(128.,8*kin.R**2))):
            for pair in ((0,1),(0,2)):
                case=dict(id='bank',kind='comb_pair_face',coefficient_orders=[0,4,3],
                    geometry=dict(pair=list(pair),tau=[.2,T],scale=.3,phase=.9,shape_upper=.125/.3))
                spec=make_spec(kin,case,12);key=register(OUTPUT,spec)
                entries.append(dict(tag=tag,label=f'comb_{j}_{pair[1]}',key=key,
                    kind='comb_pair_face',pair=list(pair),height=T,order=12,kinematics=kin.record()))
        print(json.dumps(dict(planned=tag,R=kin.R,entries=len(entries))),flush=True)
    p=dict(format='thermal-reference-campaign-v1',entries=entries,maximum_nodes_per_axis=12,
        baseline_sha256=digest(OUTPUT/'baseline.json'),pilot_sources=sources,
        matrix_model_used_for_planning=False,full_amplitude_certified=False,
        scope='Paired retained thermal correction plus the immutable refined draft baseline.',
        primary_order=4,collision_orders='pair H8/long3; scale/corner loop3; comb radial4/loop3',
        moduli=dict(primary='geometric and R^2 length mixture on infinite support',
            pair='R^2 tail height, endpoint-aware puncture map, independent adapted Sobol scrambles',
            collision='analytic full-height corner/scale integrals; R^2-rescaled pair-face tail mesh'),
        plotted_radius_count=15,computed_radius_count=15,maximum_computed_R=8.)
    publish(OUTPUT/'reference_plan.json',p)
    return p


def build_one(index):
    row=read(OUTPUT/'reference_plan.json')['entries'][index]
    if row['kinematics']['R']>8:raise ValueError('R>8 is outside the authorized campaign')
    target=OUTPUT/'build_results'/f'{index:04d}.json'
    if target.exists():
        receipt=read(target)
        if receipt['entry']!=row:raise ValueError('changed bank specification')
        return
    start=time.monotonic()
    for _ in range(40):
        r=build_reference(OUTPUT,row['key'],cpu_budget=45.)
        if r['complete_rule']:break
        if (r['failure'] or {}).get('type')!='TimeLimit':raise RuntimeError(str(r['failure']))
    else:raise RuntimeError('resumable preparation CPU allocation exhausted')
    publish(target,dict(entry=row,seconds=time.monotonic()-start,complete=True))
    print(json.dumps(dict(built=index,tag=row['tag'],kind=row['kind'],seconds=time.monotonic()-start)),flush=True)


class Engine:
    def __init__(self,tag,component,namespace):
        self.entries=[r for r in read(OUTPUT/'reference_plan.json')['entries'] if r['tag']==tag]
        k=self.entries[0]['kinematics'];self.kin=ThermalKinematics(k['t'],k['R']);self.component=component
        if self.kin.R>8:raise ValueError('R>8 is outside the authorized campaign')
        self.evaluator=Evaluator(BANKS,self.kin)
        # One writer per component/replicate. Bank data remain shared/read-only.
        self.evaluator.output=OUTPUT/'workers'/namespace
        self.groups={};self.distances=[]
        if component=='primary':
            for sector in SECTORS:
                rows=[r for r in self.entries if r['kind']=='primary' and
                      r['label'].endswith(''.join(map(str,sector)))]
                x=np.log1p([r['anchor'][:2] for r in rows])
                self.groups[sector]=(cKDTree(x),rows)
        elif component.startswith('pair-'):
            region=component.split('-')[1]
            for pair in ((0,1),(0,2)):
                rows=[r for r in self.entries if r['kind']=='pair' and r['region']==region and tuple(r['pair'])==pair]
                self.groups[pair]=(cKDTree([r['feature'] for r in rows]),rows)

    def evaluate(self,coordinates,jacobian):
        out=np.zeros((len(coordinates),3))
        self.arithmetic_bounds=np.zeros_like(out);self.arithmetic_flags=0
        if self.component=='primary':
            for start in range(0,len(coordinates),32):
                groups=defaultdict(list)
                for j,coordinate in enumerate(coordinates[start:start+32],start):
                    for sector in SECTORS:
                        short=ShortPoint(sector,*coordinate)
                        if not regular_mask(short,.125,MIN_HEIGHT):continue
                        tree,rows=self.groups[sector];distance,i=tree.query(np.log1p(coordinate[:2]))
                        groups[rows[i]['key']].append((j,short));self.distances.append(float(distance))
                for key,items in groups.items():
                    v=self.evaluator.primary(key,[s for _,s in items])
                    for (j,_),value in zip(items,v):out[j]+=2*jacobian[j]*value
        else:
            for j,((tau1,T,x,y),jac) in enumerate(zip(coordinates,jacobian)):
                tau=complex(tau1,T);total=TAU*(x+tau*y)
                if abs(nearest_torus_displacement(total,tau))<.5:continue
                f=features(tau,total)
                for pair,multiplicity in (((0,1),1),((0,2),2)):
                    tree,rows=self.groups[pair];distance,i=tree.query(f)
                    self.distances.append(float(distance))
                    g=dict(pair=pair,tau=tau,delta_total=total,radius=.125)
                    values,bounds,flag=self.evaluator.pair(rows[i]['key'],g,
                        absolute_arithmetic=True,leading_only=True)
                    out[j]+=multiplicity*jac*values[0]
                    self.arithmetic_bounds[j]+=abs(multiplicity*jac)*bounds[0]
                    self.arithmetic_flags+=flag
        if not np.all(np.isfinite(out)):raise FloatingPointError('nonfinite paired sample')
        if not np.all(np.isfinite(self.arithmetic_bounds)):
            raise FloatingPointError('nonfinite weighted arithmetic enclosure')
        return out

    def close(self):self.evaluator.close()


def sampling(tag,component,replicate=None,power=None):
    folder=OUTPUT/'sampling'/tag/component;folder.mkdir(parents=True,exist_ok=True)
    pilot=replicate is None
    power=8 if pilot else (power if power is not None else (8 if component=='primary' else 10))
    target=folder/('pilot.npz' if pilot else f'p{power}_r{replicate:02d}.npz')
    if target.exists():return
    engine=Engine(tag,component,f'{tag}_{component}_'+('pilot' if pilot else str(replicate)))
    mapper=(lambda u:primary_coordinates(u,engine.kin.R)) if component=='primary' else (
        lambda u:pair_coordinates(u,engine.kin.R,component.split('-')[1]))
    seed=SEEDS[component]+(900000 if pilot else 1009*replicate)
    unit=qmc.Sobol(4,scramble=True,seed=seed).random_base2(power)
    if pilot:coords,jac=mapper(unit)
    else:
        proposal=Proposal(read(folder/'proposal.json')['proposal'])
        mapped,weight=proposal.transform(unit);coords,jac=mapper(mapped);jac*=weight
    start=time.monotonic()
    try:values=engine.evaluate(coords,jac)
    finally:engine.close()
    np.savez_compressed(target,unit=unit,coordinates=coords,values=values,
        seconds=time.monotonic()-start,seed=seed,
        arithmetic_bounds=engine.arithmetic_bounds,arithmetic_flags=engine.arithmetic_flags)
    if pilot:
        proposal=Proposal.train(unit,values,component)
        publish(folder/'proposal.json',dict(proposal=proposal.record,training_sha256=digest(target),
            training_seed=seed,validation_seed=SEEDS[component],matrix_used=False,
            largest_anchor_distance=max(engine.distances,default=0)))
    print(json.dumps(dict(sampled=tag,component=component,replicate=replicate,power=power,
        correction=float(values[:,2].mean()),seconds=time.monotonic()-start)),flush=True)


def collision(tag):
    target=OUTPUT/'collision'/f'{tag}.json'
    if target.exists():return
    rows=[r for r in read(OUTPUT/'reference_plan.json')['entries'] if r['tag']==tag]
    k=rows[0]['kinematics'];kin=ThermalKinematics(k['t'],k['R']);e=Evaluator(BANKS,kin)
    e.output=OUTPUT/'workers'/f'{tag}_collision'
    if kin.R>8:raise ValueError('R>8 is outside the authorized campaign')
    heights={(r['kind'],r['region'],tuple(r['pair'])):r['key'] for r in rows if r['kind'].startswith('triple_')}
    combs={pair:[r for r in rows if r['kind']=='comb_pair_face' and tuple(r['pair'])==pair]
           for pair in ((0,1),(0,2))}
    totals=defaultdict(lambda:defaultdict(lambda:np.zeros(3)));roundoff=0.;flags=0;start=time.monotonic()
    try:
        for batch in read(OUTPUT/'collision_mesh.json')['batches']:
            for point,sample in batch['samples'].items():
                v=np.zeros(3);point_bound=0.
                for req in sample['requests']:
                    g=dict(req['geometry']);pair=tuple(g['pair']);jac=1.
                    if req['kind']=='height_integrated_collar':
                        v+=req['multiplicity']*e.height(heights[(g['term'],g['region'],pair)],g)[0]
                    else:
                        tau=complex(*g['tau'])
                        if batch['region']=='tail':
                            jac=max(1.,kin.R**2/8);tau=complex(tau.real,8+jac*(tau.imag-8))
                        g['tau']=tau
                        bankrow=min(combs[pair],key=lambda r:abs(math.log(r['height']/tau.imag)))
                        _,_,bank,_=e.load(bankrow['key'])
                        old,oc=comb_term(bank,e.atlas,kin,**g,noncompact=True,diagnostics=True)
                        new,nc=comb_term(bank,e.atlas,kin,**g,diagnostics=True)
                        if not oc['passed'] or not nc['passed']:
                            # Preserve the flagged arithmetic, never promote a
                            # cancellation-dominated tiny term to certification.
                            flags+=1
                        for check in (oc['prefixes'][0],nc['prefixes'][0]):
                            log_bound=check['absolute_arithmetic_log_bound']
                            if log_bound is None:raise FloatingPointError('unbounded collision arithmetic')
                            point_bound+=2*math.pi**2*abs(jac*req['multiplicity'])*math.exp(log_bound)
                        v+=jac*req['multiplicity']*columns(old[0],new[0])
                for rule,weight,_ in batch['weights'][point]:
                    totals[batch['cell_id']][rule]+=weight*v
                    if rule=='high':roundoff+=abs(weight)*point_bound
        high=np.sum([x['high'] for x in totals.values()],axis=0)
        low=np.sum([x['low'] for x in totals.values()],axis=0)
        publish(target,dict(kinematics=kin.record(),high=high.tolist(),low=low.tolist(),
            moduli_indicator=np.sum([abs(x['high']-x['low']) for x in totals.values()],axis=0).tolist(),
            integrated_arithmetic_bound=roundoff,arithmetic_flagged_requests=flags,seconds=time.monotonic()-start,
            full_amplitude_certified=False))
    finally:e.close()
    print(json.dumps(dict(collision=tag,correction=high[2],seconds=time.monotonic()-start)),flush=True)


def tasks(stage):
    p=read(OUTPUT/'reference_plan.json');tags=list(dict.fromkeys(r['tag'] for r in p['entries']))
    if stage=='build':return [(i,) for i in range(len(p['entries']))]
    if stage=='pilot':return [(tag,c) for tag in tags for c in SEEDS]
    if stage=='sample':return [(tag,c,r) for tag in tags for c in SEEDS for r in range(16)]
    if stage=='collision':return [(tag,) for tag in tags]
    raise ValueError(stage)


def dispatch(stage,index):
    args=tasks(stage)[index]
    {'build':build_one,'pilot':sampling,'sample':sampling,'collision':collision}[stage](*args)


def main():
    global BANKS
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('plan','worker','slurm'))
    parser.add_argument('--stage',choices=('build','pilot','sample','collision'))
    parser.add_argument('--index',type=int);parser.add_argument('--banks',type=Path);a=parser.parse_args()
    if a.banks is not None:BANKS=a.banks
    if a.mode=='plan':print(json.dumps(dict(entries=len(plan()['entries']))));return
    if a.mode=='worker':dispatch(a.stage,a.index);return
    rank=int(os.environ['SLURM_PROCID']);size=int(os.environ['SLURM_NTASKS'])
    failed=[]
    for i in range(rank,len(tasks(a.stage)),size):
        try:dispatch(a.stage,i)
        except Exception as exc:
            row=dict(stage=a.stage,index=i,type=type(exc).__name__,message=str(exc));failed.append(row)
            print(json.dumps(row),flush=True)
    if failed:
        publish(OUTPUT/'failures'/f'{a.stage}_{rank:03d}.json',failed);raise SystemExit(1)


if __name__=='__main__':main()
