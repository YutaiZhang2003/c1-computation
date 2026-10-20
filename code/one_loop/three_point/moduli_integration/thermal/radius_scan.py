"""Radius-specific moduli integration of the retained genus-one estimator.

New independent pilot/validation grids replace all six primary sectors and
both isolated-pair regions. The already integrated small collision terms and
the compact-bulk receipt are retained with their explicit diagnostics. No
matrix-model value enters adaptation. Momentum banks are fixed, verified
inputs, so moduli convergence is not a momentum convergence certificate.
"""

from genus1_source_paths import ROOT as _SOURCE_ROOT, source_path as _source_path
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time

sys.dont_write_bytecode=True
HERE = _SOURCE_ROOT / 'output/thermal_radius_scan_20260927'
PILOT=HERE.parent/"thermal_sampling_20260927"
PREVIOUS=HERE.parent/"thermal_one_loop_20260927"

from study_pair_sampling import Density as PairDensity, physical_map as pair_map
from study_pair_sampling import save, summarize
from thermal_saved_inputs import Banks, SOURCE, read
from genus1_boundary_batch import GeometryStore, BatchedPrimary, SharedKernelSlice
from genus1_analytic_boundary_layer import PrimarySlice
from genus1_three_point_channel_atlas import make_atlas
from genus1_production_support import make_plan
from thermal_cut_rule import kernel_log
import kernels as thermal
from genus1_moduli_sampling import ShortPoint, regular_mask
from genus1_analytic_boundary_layer import MIN_HEIGHT
from genus1_nonseparating import _sum_logs
from scipy.spatial import cKDTree
from scipy.ndimage import convolve1d
from scipy.stats import qmc
import numpy as np

TAGS=("t035","t045","t055","t065","t075","t085","t095")
TAU=2*math.pi
SECTORS=((0,1,2),(0,2,1),(2,0,1))
SEEDS={"primary":940001,"pair-bulk":950001,"pair-tail":960001}
MAP_VERSION="radius-conditioned-maps-v1"


def pair_coordinates(u,R,region):
    """Pilot map with an algebraic inverse CDF for extreme heights."""
    u=np.asarray(u);tau1=u[:,1]-.5
    if region=="bulk":
        lower=np.sqrt(1-tau1**2);height=lower+(8-lower)*u[:,0];hj=8-lower
    else:
        scale=max(8.,R**2)
        height=8+scale*((1-u[:,0])**-2-1);hj=2*scale*(1-u[:,0])**-3
    a=R**2/(TAU*height);c=.5/(a+.5)
    target=2*np.minimum(u[:,3],1-u[:,3]);b=1.5*target
    B=3*a+1-b;D=np.sqrt(B*B+4*a*b)
    distance=np.empty(len(u));positive=B>=0
    distance[positive]=2*a[positive]*b[positive]/(D[positive]+B[positive])
    distance[~positive]=(D[~positive]-B[~positive])/2
    y=np.where(u[:,3]<.5,distance,1-distance)
    pdf=1/3+a/(3*c*(a+distance)**2)
    return np.column_stack((tau1,height,u[:,2],y)),hj/pdf*TAU**2*height


def primary_coordinates(u,R):
    """Positive short lengths with both geometric and thermal scales.

    The marginal density is .7*L/(L+s)^2 + .3/(2D)*(1+s/D)^(-3/2),
    L=-log(.05), D=max(L,R^2/2). Its monotone CDF is inverted on the log
    length; both angular coordinates remain in the original flat measure.
    """
    u=np.asarray(u);L=-math.log(.05);D=max(L,R**2/2);fraction=.7
    target=u[:,:2]
    lo=np.zeros_like(target)
    hi=np.log1p(4*max(L,D)/(1-target)**2/L)
    for _ in range(56):
        mid=(lo+hi)/2;s=L*np.expm1(mid)
        cdf=fraction*s/(L+s)+(1-fraction)*(1-np.sqrt(D/(D+s)))
        lo=np.where(cdf<target,mid,lo);hi=np.where(cdf>=target,mid,hi)
    s=L*np.expm1((lo+hi)/2)
    pdf=fraction*L/(L+s)**2+(1-fraction)/(2*D)*(1+s/D)**-1.5
    return np.column_stack((s,TAU*u[:,2:])),TAU**2/np.prod(pdf,axis=1)


class Proposal:
    """A conditional two-axis grid plus two one-axis angular grids."""
    def __init__(self,record):
        self.record=record;self.axes=tuple(record["joint_axes"])
        self.mass=np.asarray(record["joint_probability"]);self.bins=len(self.mass)
        self.marginals={int(k):np.asarray(v) for k,v in record["marginals"].items()}
        for a in (self.mass,*self.marginals.values()):
            if np.any(a<=0) or abs(a.sum()-1)>1e-12:raise ValueError("invalid proposal mass")

    @classmethod
    def train(cls,u,values,component,bins=8):
        # Full-amplitude accuracy and thermal sensitivity are both objectives.
        full=abs(values[:,1]);delta=abs(values[:,2])
        mass=.5*full/max(full.mean(),1e-300)+.5*delta/max(delta.mean(),1e-300)
        axes=(0,1) if component=="primary" else (0,3)
        h=np.histogram2d(u[:,axes[0]],u[:,axes[1]],bins=bins,
                        range=((0,1),(0,1)),weights=mass)[0]
        h=convolve1d(convolve1d(h,[.2,.6,.2],axis=0,mode="nearest"),
                     [.2,.6,.2],axis=1,mode="nearest")
        h=.75*h/h.sum()+.25/bins**2
        marginals={}
        for axis in set(range(4))-set(axes):
            v=np.histogram(u[:,axis],bins=bins,range=(0,1),weights=mass)[0]
            v=convolve1d(v,[.2,.6,.2],mode="wrap")
            marginals[str(axis)]=(.5*v/v.sum()+.5/bins).tolist()
        return cls(dict(joint_axes=list(axes),joint_probability=h.tolist(),marginals=marginals,
            joint_uniform_fraction=.25,angular_uniform_fraction=.5,
            objective="equal normalized absolute thermal integrand and paired thermal correction"))

    @staticmethod
    def inverse(x,mass):
        cdf=np.r_[0.,np.cumsum(mass)];cdf[-1]=1.
        i=np.searchsorted(cdf,x,side="right")-1
        return (i+(x-cdf[i])/mass[i])/len(mass),i

    def transform(self,unit):
        u=np.asarray(unit).copy();a,b=self.axes
        rows=self.mass.sum(axis=1)
        u[:,a],i=self.inverse(u[:,a],rows)
        conditional=self.mass[i]/rows[i,None]
        cdf=np.column_stack((np.zeros(len(u)),np.cumsum(conditional,axis=1)));cdf[:,-1]=1
        j=np.sum(unit[:,b,None]>=cdf,axis=1)-1;k=np.arange(len(u))
        u[:,b]=(j+(unit[:,b]-cdf[k,j])/conditional[k,j])/self.bins
        weight=1/(self.mass[i,j]*self.bins**2)
        for axis,mass in self.marginals.items():
            u[:,axis],indices=self.inverse(u[:,axis],mass)
            weight/=mass[indices]*len(mass)
        return u,weight


class PrimaryDensity:
    def __init__(self,tag):
        self.tag=tag;self.p=read(SOURCE/"energies"/tag/"plan.json")
        self.physics=make_plan(self.p["config"]);self.R=2/self.physics.t
        self.atlas=make_atlas(self.physics.t);self.banks=Banks(tag)
        self.geometry=GeometryStore(self.atlas,HERE/"geometry"/tag)
        self.engines={};groups=defaultdict(list)
        for path in sorted((PREVIOUS/"inputs"/tag/"grids").glob("*/spec.json")):
            s=read(path)
            if s["case"]["kind"]!="primary":continue
            short=s["design"]["short"];axis=s["design"]["cut_axis"]
            groups[axis].append((path.parent.name,np.log1p([short["s1"],short["s2"]])))
        self.groups={axis:(cKDTree([v[1] for v in rows]),[v[0] for v in rows])
                     for axis,rows in groups.items()}
        self.largest_distance=0.;self.evaluations=0

    def evaluate(self,coordinates,jacobian,*,alternate=False,order=4):
        output=np.zeros((len(coordinates),3))
        # Exchange of equal outgoing labels gives two identical copies of
        # each of these three sectors. All six source sectors are retained.
        for begin in range(0,len(coordinates),32):
            specs=[];shorts=[];owners=[];routes=[]
            for i,(s1,s2,x1,x2) in enumerate(coordinates[begin:begin+32],begin):
                for sector in SECTORS:
                    short=ShortPoint(sector,s1,s2,x1,x2)
                    if not regular_mask(short,self.physics.policy.pair_radius,MIN_HEIGHT):continue
                    axis=(2-sector.index(2))%3;tree,keys=self.groups[axis]
                    distance,index=tree.query(np.log1p([s1,s2]),k=2 if alternate else 1)
                    if alternate:distance,index=distance[1],index[1]
                    self.largest_distance=max(self.largest_distance,float(distance))
                    shorts.append(short);owners.append(i);routes.append(keys[index])
            rows=self.geometry.prepare(shorts)
            for key in dict.fromkeys(routes):
                _,_,_,bank=self.banks.get(key);axis=bank.metadata["cut_axis"]
                if key not in self.engines:
                    self.engines[key]=BatchedPrimary(self.atlas,{axis:bank},self.geometry,orders=[2,4])
                engine=self.engines[key]
                indices=[i for i,k in enumerate(routes) if k==key]
                blocks=engine.block_logs([rows[i] for i in indices],axis,[order])[0]
                p=bank.arrays["momenta"]
                base=bank.arrays["log_weights"]+bank.arrays["structure_logs"]-3*math.log(math.pi)
                unique,inverse=engine.cuts[axis]
                for j,i in enumerate(indices):
                    short=shorts[i];row=rows[i]
                    logs=[complex(*v) for v in row["short_logs"]]
                    exponent=sum(2*p[:,(axis+k)%3]**2*z.real for k,z in enumerate(logs,1))
                    nodes=PrimarySlice(p[:,axis],base+exponent+2*blocks[j],
                                       short.total,row["sine"],row["gaussian"])
                    shared=SharedKernelSlice(nodes,unique,inverse,{})
                    pos=short.positions();qy=abs((pos[0]+pos[1]-2*pos[2]).imag)
                    _,_,old_kernel=thermal.integrated_primary_components(tuple(map(float,unique)),
                        short.s1,short.s2,qy,self.R,self.physics.policy.length,32)
                    new_kernel=kernel_log(unique,short,self.R,self.physics.policy.length)
                    old=_sum_logs(shared.coefficients+row["sine"]+old_kernel)[0]
                    new=_sum_logs(shared.coefficients+row["sine"]+new_kernel)[0]
                    a=-4*math.pi**2*jacobian[owners[i]]*np.array([old.imag,new.imag])
                    output[owners[i]]+=np.r_[a,a[1]-a[0]]
            self.evaluations+=len(rows)
        if not np.all(np.isfinite(output)):raise FloatingPointError("nonfinite primary observation")
        return output

    def close(self):self.geometry.close()


def run(tag,component,power=10,scrambles=8,train_power=9,alternate=False):
    if tag not in TAGS or component not in SEEDS:raise ValueError("unknown radius or region")
    start=time.monotonic();folder=HERE/"runs"/tag/component;folder.mkdir(parents=True,exist_ok=True)
    engine=PrimaryDensity(tag) if component=="primary" else PairDensity(tag)
    mapper=(lambda u:primary_coordinates(u,engine.R)) if component=="primary" else (
        lambda u:pair_coordinates(u,engine.R,component.split('-')[1]))
    pilot_path=folder/f"pilot_p{train_power}.npz"
    if pilot_path.exists():
        with np.load(pilot_path) as a:pilot_unit,pilot_values=a["unit"],a["values"]
    else:
        pilot_unit=qmc.Sobol(4,scramble=True,seed=SEEDS[component]).random_base2(train_power)
        coords,jac=mapper(pilot_unit);pilot_values=engine.evaluate(coords,jac)
        np.savez_compressed(pilot_path,unit=pilot_unit,coordinates=coords,values=pilot_values)
    proposal=Proposal.train(pilot_unit,pilot_values,component)
    proposal_record=dict(proposal.record,R=engine.R,tag=tag,component=component,
        map_version=MAP_VERSION,training_seed=SEEDS[component],training_points=len(pilot_unit),
        training_sha256=hashlib.sha256(pilot_path.read_bytes()).hexdigest(),
        no_matrix_model_input=True)
    proposal_path=folder/"proposal.json"
    if proposal_path.exists() and json.loads(proposal_path.read_text())!=proposal_record:
        raise ValueError("frozen proposal changed; create a new run directory")
    save(proposal_path,proposal_record)
    label="alternate" if alternate else "adapted"
    reps=[];halves=[]
    for replicate in range(scrambles):
        target=folder/f"{label}_p{power}_r{replicate:02d}.npz"
        if target.exists():
            with np.load(target) as a:values=a["values"]
        else:
            unit=qmc.Sobol(4,scramble=True,seed=SEEDS[component]+20000+1009*replicate).random_base2(power)
            mapped,weights=proposal.transform(unit);coords,jac=mapper(mapped)
            # Preserve the nested validation stream and reuse its exact prefix.
            previous=folder/f"{label}_p{power-1}_r{replicate:02d}.npz"
            prefix=None
            if previous.exists():
                with np.load(previous) as a:
                    if not np.array_equal(a["unit"],unit[:len(a["unit"])]):raise ValueError("changed Sobol prefix")
                    prefix=a["values"].copy()
            first=0 if prefix is None else len(prefix)
            new=engine.evaluate(coords[first:],(jac*weights)[first:],alternate=alternate)
            values=new if prefix is None else np.concatenate((prefix,new))
            np.savez_compressed(target,unit=unit,coordinates=coords,values=values,importance=weights)
        reps.append(values.mean(axis=0));halves.append(values[:len(values)//2].mean(axis=0))
        print(json.dumps(dict(tag=tag,component=component,power=power,replicate=replicate,
            thermal=float(reps[-1][1]),delta=float(reps[-1][2]),seconds=time.monotonic()-start)),flush=True)
    result=dict(tag=tag,R=engine.R,component=component,method=label,map_version=MAP_VERSION,
        points_per_scramble=2**power,scrambles=scrambles,training_points=len(pilot_unit),
        columns=["noncompact_same_frequency","thermal","thermal_correction"],
        statistics=summarize(reps),half_statistics=summarize(halves),
        paired_doubling_shift=summarize(np.array(reps)-halves),
        normalization="-2*pi^2*Im(I); coefficient of g_s^3",
        original_CFT_banks_reused=True,coefficient_interpolation=False,
        momentum_convergence_certified=False,full_amplitude_certified=False,
        proposal_sha256=hashlib.sha256(proposal_path.read_bytes()).hexdigest(),
        seconds=time.monotonic()-start)
    if component=="primary":
        result.update(largest_bank_anchor_distance=engine.largest_distance,
                      CFT_evaluations=engine.evaluations)
        engine.close()
    save(folder/f"{label}_p{power}_s{scrambles}.json",result)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--tag",default="t075")
    p.add_argument("--component",choices=list(SEEDS),default="primary")
    p.add_argument("--power",type=int,default=10);p.add_argument("--scrambles",type=int,default=8)
    p.add_argument("--train-power",type=int,default=9);p.add_argument("--alternate",action="store_true")
    a=p.parse_args();run(a.tag,a.component,a.power,a.scrambles,a.train_power,a.alternate)
