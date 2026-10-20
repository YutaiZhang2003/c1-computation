"""Compact Euclidean boson factor relative to the noncompact torus integrand.

alpha'=1; z=2*pi*(x+tau*y); integer external momenta n_i/R and no
external winding. The compact target zero mode is stripped separately.
"""
from __future__ import annotations

import math
import numpy as np


def phases(tau, positions, charges):
    tau=complex(tau)
    if tau.imag<=0:
        raise ValueError("Im(tau) must be positive")
    n=np.asarray(charges,float)
    z=np.asarray(positions,complex)
    if z.shape!=n.shape or np.any(n!=np.rint(n)) or abs(n.sum())>1e-12:
        raise ValueError("Neutral integer KK charges and equally many positions required")
    y=z.imag/(2*math.pi*tau.imag)
    x=z.real/(2*math.pi)-tau.real*y
    return float(n@x),float(n@y)


def lattice_direct(R,tau,a=0.,b=0.,*,tolerance=1e-15):
    R=float(R);tau=complex(tau)
    if R<=0 or tau.imag<=0:
        raise ValueError("Positive radius and torus height required")
    L=-math.log(tolerance)
    mcap=math.ceil(math.sqrt(L/(math.pi*R*R*tau.imag)))+1
    width=math.sqrt(L*tau.imag/(math.pi*R*R))
    terms=[]
    for m in range(-mcap,mcap+1):
        center=m*tau.real
        lo=math.floor(center-width)-1;hi=math.ceil(center+width)+1
        for w in range(lo,hi+1):
            action=math.pi*R*R/tau.imag*((w-center)**2+(m*tau.imag)**2)
            if action<L+5:
                terms.append(math.exp(-action)*math.cos(2*math.pi*(m*a+w*b)))
    return math.fsum(terms)


def lattice_poisson(R,tau,a=0.,b=0.,*,tolerance=1e-15):
    R=float(R);tau=complex(tau)
    if R<=0 or tau.imag<=0:
        raise ValueError("Positive radius and torus height required")
    L=-math.log(tolerance);a=a%1.;b=b%1.
    mcap=math.ceil(math.sqrt(L/(math.pi*R*R*tau.imag)))+1
    width=math.sqrt(L*R*R/(math.pi*tau.imag))
    terms=[]
    for ell in range(math.floor(b-width)-1,math.ceil(b+width)+2):
        p=ell-b
        common=math.pi*tau.imag*p*p/(R*R)
        for m in range(-mcap,mcap+1):
            action=common+math.pi*tau.imag*R*R*m*m
            if action<L+5:
                terms.append(math.exp(-action)*
                             math.cos(2*math.pi*m*(a-tau.real*p)))
    return math.sqrt(tau.imag)/R*math.fsum(terms)


def compact_factor(R,tau,positions,charges):
    a,b=phases(tau,positions,charges)
    evaluator=lattice_direct if tau.imag<R*R else lattice_poisson
    return evaluator(R,tau,a,b)


def validate():
    records=[]
    for R in (.6,1.,1.5,3.,8.):
        for tau in (.3+1.1j,-.45+5j,.2+80j):
            for a,b in ((0.,0.),(.37,.23),(-.9,1.4)):
                value=lattice_direct(R,tau,a,b)
                poisson=lattice_poisson(R,tau,a,b)
                scale=max(1.,abs(value))
                assert abs(value-poisson)<2e-12*scale
                assert abs(value-lattice_direct(R,tau,a+2,b-3))<2e-12*scale
                assert abs(value-lattice_direct(R,-1/tau,b,-a))<2e-12*scale
                assert abs(value-lattice_direct(R,tau+1,a-b,b))<2e-12*scale
                records.append(abs(value-poisson))
            assert abs(R*lattice_direct(R,tau)-lattice_direct(1/R,tau)/R)<2e-12
    assert abs(lattice_direct(20.,.2+1.1j,.3,.4)-1)<1e-14
    tau=.31+2.4j;z=np.array([.2+.3j,2+1j,0j]);n=[1,1,-2]
    value=compact_factor(2.7,tau,z,n)
    assert abs(value-compact_factor(2.7,tau,z+1.3+.2j,n))<2e-13
    return dict(direct_poisson_cases=len(records),
                maximum_absolute_direct_poisson_difference=max(records),
                modular_S=True,modular_T=True,integer_periodicity=True,
                neutral_translation=True,vacuum_T_duality=True,
                fixed_modulus_decompactification=True)


if __name__=="__main__":
    import json
    print(json.dumps(validate(),indent=2))
