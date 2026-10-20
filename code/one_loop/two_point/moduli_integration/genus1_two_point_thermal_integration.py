"""Thermal 1->1 worldsheet integral with an explicit compact-boson lattice.

Integrates the entire torus cusp by an infinite-support sampling map.
The Liouville momentum rule and descendant orders remain finite and must
be audited separately. No matrix-model value enters the estimator.
"""
import argparse
import cmath
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from scipy.stats import qmc
from genus1_two_point_coefficient_bank import load_bank
from genus1_three_point_worldsheet import (
    torus_prime_form_log_norm,dedekind_eta_oscillator_abs_squared)
from genus1_two_point_worldsheet import (
    LiouvilleTorusTwoPoint,MomentumRule,reduced_worldsheet_integrand)
from genus1_compact_boson import lattice_direct,lattice_poisson,validate


class Kernel:
    def __init__(self,path):
        self.metadata,self.arrays=load_bank(path)
        self.x=self.metadata["x"];self.R=1/self.x
        self.neck=tuple(self.arrays["neck_"+k] for k in ("h1","h2","weights","coefficients"))
        self.ope=tuple(self.arrays["ope_"+k] for k in ("h1","h2","weights","coefficients"))

    def bulk(self,z,tau,epsilon=.15):
        h_external=1-self.x*self.x/4
        prime_log=torus_prime_form_log_norm(z,tau)
        local=min((z,z-2*math.pi,z-2*math.pi*tau,z-2*math.pi*(1+tau)),key=abs)
        eta=dedekind_eta_oscillator_abs_squared(tau)/math.sqrt(tau.imag)
        if abs(local)<2*math.pi*epsilon:
            h1,h2,w,c=self.ope
            q=cmath.exp(2j*math.pi*tau)
            v=cmath.exp(-1j*local)-1
            block=np.einsum("tij,i,j->t",c,q**np.arange(c.shape[1]),
                            local**np.arange(c.shape[2]),optimize=False)
            exponent=(-4*math.pi*tau.imag*(h1-1)+2*h2*math.log(abs(v))
                      -4*h_external*math.log(abs(2*cmath.sin(local/2)))
                      -self.x*self.x*prime_log)
        else:
            h1,h2,w,c=self.neck
            log1=1j*z;log2=1j*(2*math.pi*tau-z)
            q1=cmath.exp(log1);q2=cmath.exp(log2)
            block=np.einsum("tij,i,j->t",c,q1**np.arange(c.shape[1]),
                            q2**np.arange(c.shape[2]),optimize=False)
            exponent=(2*(h1-1)*log1.real+2*(h2-1)*log2.real
                      -self.x*self.x*prime_log)
        return eta*np.dot(w,np.exp(exponent)*abs(block)**2)

    def disc(self,tau,radius):
        h1,h2,w,c=self.ope
        q=cmath.exp(2j*math.pi*tau)
        block=c[:,:,0]@(q**np.arange(c.shape[1]))
        p2=h2-1
        terms=np.exp(-4*math.pi*tau.imag*(h1-1)+2*p2*math.log(radius))
        return (dedekind_eta_oscillator_abs_squared(tau)/math.sqrt(tau.imag)*
                np.dot(w,terms*abs(block)**2*math.pi/p2))

    def disc_density(self,tau,radius):
        """Pointwise density whose integral is the analytic disc addback."""
        h1,h2,w,c=self.ope
        q=cmath.exp(2j*math.pi*tau)
        block=c[:,:,0]@(q**np.arange(c.shape[1]))
        exponent=-4*math.pi*tau.imag*(h1-1)+(2*(h2-1)-2)*math.log(radius)
        return (dedekind_eta_oscillator_abs_squared(tau)/math.sqrt(tau.imag)*
                np.dot(w,np.exp(exponent)*abs(block)**2))

    def validate_against_original(self):
        m=self.metadata
        old=LiouvilleTorusTwoPoint(1j*self.x,
            momentum_rule=MomentumRule.power_legendre(m["p_max"],m["momentum_order"],m["power"]),
            necklace_orders=tuple(m["necklace_orders"]),ope_orders=tuple(m["ope_orders"]))
        old._necklace_arrays=self.neck;old._ope_arrays=self.ope
        differences=[]
        for tau in (.2+1.2j,-.3+5j,.1+20j):
            for z in (.13+.09j,1.+.5j,2.+2j):
                a=self.bulk(z,tau);b=reduced_worldsheet_integrand(old,z,tau)
                differences.append(abs(a-b)/max(abs(a),1e-250))
                assert differences[-1]<2e-12
            a=self.disc(tau,.1);b=old.leading_collision_disc(tau,.1)
            assert abs(a-b)<2e-12*abs(a)
        return max(differences)


def integrate(kernel,power,replicates,radius,seed,height_scale=1.,short_scale=None,
              collision_remainder=False):
    short_scale=4/kernel.x**2 if short_scale is None else short_scale
    all_values=[];time_start=time.monotonic()
    bins=np.array([1.,2.,4.,8.,16.,32.,64.,128.,256.,512.,2048.,math.inf])
    bybin=[]
    for rep in range(replicates):
        points=qmc.Sobol(4,scramble=True,seed=seed+rep).random_base2(power)
        values=np.zeros((len(points),2),complex)
        pieces=np.zeros((len(bins),2),complex)
        for j,(a,b,c,d) in enumerate(points):
            tau1=a-.5;floor=math.sqrt(1-tau1*tau1)
            # Tail proposal proportional to tau2^(-3/2); support reaches infinity.
            tau2=floor+height_scale*((1-b)**-2-1)
            jac_tau=2*height_scale*(1-b)**-3
            tau=complex(tau1,tau2)
            max_s=math.pi*tau2
            mass=max_s/(short_scale+max_s)
            s=short_scale*d*mass/(1-d*mass)
            jac_s=short_scale*mass/(1-d*mass)**2
            z=complex(2*math.pi*c+s*tau1/tau2,s)
            evaluator=lattice_direct if tau2<kernel.R**2 else lattice_poisson
            boson=evaluator(kernel.R,tau,float(c),s/(2*math.pi*tau2))
            fused=evaluator(kernel.R,tau)
            local=min(abs(z),abs(z-2*math.pi))
            bulk=0j if local<radius else 4*math.pi*jac_s*kernel.bulk(z,tau)
            disc=kernel.disc(tau,radius)
            values[j]=jac_tau*np.array([bulk+disc,bulk*boson+disc*fused])
            if collision_remainder:
                rr=radius*math.sqrt(c)
                local_z=rr*cmath.exp(2j*math.pi*d)
                local_a=(local_z.real-local_z.imag*tau1/tau2)/(2*math.pi)
                local_b=local_z.imag/(2*math.pi*tau2)
                local_boson=evaluator(kernel.R,tau,local_a,local_b)
                parent=kernel.bulk(local_z,tau)
                leading=kernel.disc_density(tau,rr)
                values[j]+=jac_tau*math.pi*radius**2*np.array([
                    parent-leading,parent*local_boson-leading*fused])
            if np.any(~np.isfinite(values[j])):
                raise FloatingPointError(f"nonfinite sample {j}, tau={tau},z={z}")
            pieces[np.searchsorted(bins,tau2)]+=values[j]/len(points)
        all_values.append(np.mean(values,axis=0))
        bybin.append(pieces)
        print(json.dumps(dict(replicate=rep,zero_temperature=all_values[-1][0].real,
                              thermal=all_values[-1][1].real,
                              elapsed=time.monotonic()-time_start)),flush=True)
    a=np.array(all_values);means=a.mean(axis=0);se=a.real.std(axis=0,ddof=1)/math.sqrt(replicates)
    prefactor=2*math.pi**2
    return dict(
        status="worldsheet_pilot_requires_momentum_block_and_collision_convergence",
        R=kernel.R,T=1/(2*math.pi*kernel.R),omega_imag=kernel.x,
        external_KK=[1,-1],external_winding=[0,0],
        normalization="A_(1->1)^(1),T = 2*pi^2*i*I_T; full genus-one term is g_s^2*A",
        power=power,replicates=replicates,seed=seed,collision_radius=radius,
        collision_remainder_included=collision_remainder,
        tau_support="entire fundamental domain; no fitted or truncated cusp tail",
        height_scale=height_scale,short_scale=short_scale,
        zero_temperature_imag=float(prefactor*means[0].real),
        thermal_imag=float(prefactor*means[1].real),
        thermal_correction_imag=float(prefactor*(means[1]-means[0]).real),
        sampling_SE_zero=float(prefactor*se[0]),sampling_SE_thermal=float(prefactor*se[1]),
        sampling_SE_correction=float(prefactor*np.std((a[:,1]-a[:,0]).real,ddof=1)/math.sqrt(replicates)),
        replicate_integrals=[[[float(z.real),float(z.imag)] for z in r] for r in a],
        tau_bin_upper_edges=[None if math.isinf(x) else float(x) for x in bins],
        tau_bin_integrals_mean=np.mean(bybin,axis=0).real.tolist(),
        kernel=kernel.metadata,elapsed=time.monotonic()-time_start,
        limitations=["Finite internal momentum rule and Virasoro orders.",
            ("The finite-block collision parent minus leading projection is integrated inside the full disc."
             if collision_remainder else
             "Leading collision OPE addback multiplies the compact factor at fused insertions; radius dependence must be audited."),
            "Sampling errors exclude momentum, block and collision truncation errors."],
    )


def main():
    p=argparse.ArgumentParser()
    p.add_argument("bank",type=Path);p.add_argument("--power",type=int,default=11)
    p.add_argument("--replicates",type=int,default=4)
    p.add_argument("--radius",type=float,default=.1)
    p.add_argument("--seed",type=int,default=20260927)
    p.add_argument("--short-scale",type=float)
    p.add_argument("--collision-remainder",action="store_true")
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();kernel=Kernel(a.bank)
    proof=validate();proof["stable_original_max_relative_difference"]=kernel.validate_against_original()
    result=integrate(kernel,a.power,a.replicates,a.radius,a.seed,short_scale=a.short_scale,
                     collision_remainder=a.collision_remainder)
    result["validation"]=proof
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+"\n")
    print("Saved",a.output,flush=True)


if __name__ == "__main__":
    main()
