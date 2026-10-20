"""Thermal-kernel-aware cut-momentum rule for preparing FUTURE CFT banks.

This does not replace momenta inside any existing bank. New nodes require new
DOZZ and conformal-block evaluations before use in an amplitude calculation.
"""

from genus1_source_paths import ROOT as _SOURCE_ROOT, source_path as _source_path
import math
from pathlib import Path
import sys
sys.dont_write_bytecode = True
HERE = _SOURCE_ROOT / 'output/thermal_sampling_20260927'

from thermal_saved_inputs import read, Banks, SOURCE
from genus1_moduli_sampling import ShortPoint
from kernels import integrated_primary_components, primary_kernel_log
from scipy.special import logsumexp
from scipy.linalg import eigh_tridiagonal
from scipy.integrate import quad
import numpy as np


def kernel_log(p, short, R, threshold, angle_order=32):
    q=short.positions()[0]+short.positions()[1]-2*short.positions()[2]
    offset,components,_=integrated_primary_components(tuple(map(float,p)),
        short.s1,short.s2,abs(q.imag),R,threshold,angle_order)
    factor=components@np.cos(np.arange(components.shape[1])*q.real)
    with np.errstate(divide="ignore"):
        result=offset+np.log(factor)
    # QUADPACK probes extremely large P on its infinite-interval transform.
    # Reconstruct those tiny values as logarithms, without a physical cutoff.
    bad=np.flatnonzero(factor<=0)
    if len(bad):
        p=np.asarray(p)[bad]
        floor=max(threshold,short.s1,short.s2)
        if floor+short.total>=2*math.pi:
            result[bad]=primary_kernel_log(p,short,floor,R)
        else:
            edges=[0.,math.pi];h=(floor+short.total)/(2*math.pi)
            if math.sqrt(3)/2<h<1:
                edges.insert(1,2*math.pi*math.sqrt(1-h*h))
            roots,weights=np.polynomial.legendre.leggauss(angle_order)
            logs=[]
            for a,b in zip(edges,edges[1:]):
                for x,w in zip(roots,weights):
                    theta=a+(x+1)*(b-a)/2
                    lower=max(floor,2*math.pi*math.sqrt(1-(theta/(2*math.pi))**2)-short.total)
                    logs.append(math.log(w*(b-a)/(2*math.pi))+
                                primary_kernel_log(p,short,lower,R,theta=theta))
            result[bad]=logsumexp(logs,axis=0)
    return result


def thermal_cut_rule(short, R, slope, order=8, *, threshold=-math.log(.05),
                     construction_order=256, tail_exponent=80.):
    """Gaussian quadrature in P^2 for P^2 K_R(P) exp(curvature P^2) dP.

    At finite R, the zero lattice mode makes P^2 K_R(P) approach a nonzero
    constant at P=0. The quadrature construction includes that endpoint.
    Reported weights are for dP; the construction weight is divided out.
    """
    if not R>0 or not 1<=order<=32 or construction_order<4*order:
        raise ValueError("positive R and adequately resolved construction required")
    lower=max(threshold,short.s1,short.s2)
    curvature=min(float(slope),.5*lower)
    decay=2*lower-curvature
    limit=math.sqrt(tail_exponent/decay)
    roots,weights=np.polynomial.legendre.leggauss(construction_order)
    p=limit*(roots+1)/2
    measure=limit*weights/2
    logmass=np.log(measure)+2*np.log(p)+kernel_log(p,short,R,threshold)+curvature*p*p
    peak=float(max(logmass));mass=np.exp(logmass-peak)
    norm=math.fsum(mass);x=decay*p*p
    vector=np.sqrt(mass/norm);vectors=[];diagonal=[];offdiagonal=[]
    for j in range(order):
        vectors.append(vector)
        product=x*vector
        a=float(vector@product);diagonal.append(a)
        residual=product-a*vector
        if j:residual-=offdiagonal[-1]*vectors[-2]
        for _ in range(2):
            for v in vectors:residual-=(v@residual)*v
        if j<order-1:
            b=float(np.linalg.norm(residual))
            if not b>0:raise ArithmeticError("quadrature lost rank")
            offdiagonal.append(b);vector=residual/b
    eigenvalues,eigenvectors=eigh_tridiagonal(diagonal,offdiagonal)
    if np.any(eigenvalues<=0):raise ArithmeticError("nonpositive momentum node")
    nodes=np.sqrt(eigenvalues/decay)
    logs=(peak+math.log(norm)+2*np.log(abs(eigenvectors[0]))-2*np.log(nodes)
          -kernel_log(nodes,short,R,threshold)-curvature*nodes**2)
    return dict(nodes=nodes.tolist(),weights=np.exp(logs).tolist(),
        measure="dP",order=order,R=R,curvature=curvature,
        scheme="thermal-length-kernel-weight-v1",construction_order=construction_order,
        construction_tail_exponent=tail_exponent,construction_maximum_P=limit,
        physical_momentum_cutoff=False,requires_new_CFT_bank=True,
        amplitude_convergence_certified=False)


def validate():
    import json
    # An actual primary anchor in the previous seven-radius campaign.
    folder=HERE.parent/"thermal_one_loop_20260927/inputs/t075/grids"
    spec=None
    for path in sorted(folder.glob("*/spec.json")):
        candidate=read(path)
        if candidate["case"]["kind"]=="primary":
            s=candidate["design"]["short"]
            if .5<s["s1"]<2 and .5<s["s2"]<2:
                spec=candidate;break
    if spec is None:raise ValueError("representative source anchor missing")
    short=ShortPoint.from_record(spec["design"]["short"])
    cut=spec["design"]["cut_axis"]
    old=spec["grid"]["rules"][cut]
    slope=spec["design"]["axes"][cut]["dozz_threshold_slope"]
    records=[]
    for R in (2/.95,2/.75,2/.35):
        new=thermal_cut_rule(short,R,slope)
        refined=thermal_cut_rule(short,R,slope,construction_order=512)
        node_change=float(max(abs(np.array(new["nodes"])-refined["nodes"])))
        assert node_change<2e-10
        curvature=new["curvature"]
        checks=[]
        for degree in (0,1,2,4):
            def density(p):
                return p**(2+2*degree)*math.exp(
                    kernel_log([p],short,R,-math.log(.05))[0]+curvature*p*p)
            exact,error=quad(density,0,np.inf,epsabs=1e-12,epsrel=2e-11)
            values=[]
            for rule in (old,new):
                p=np.array(rule["nodes"]);w=np.array(rule["weights"])
                values.append(float(np.dot(w,p**(2+2*degree)*np.exp(
                    kernel_log(p,short,R,-math.log(.05))+curvature*p*p))))
            assert abs(values[1]/exact-1)<2e-8
            checks.append(dict(degree_in_P_squared=degree,reference=exact,
                old_relative_error=values[0]/exact-1,new_relative_error=values[1]/exact-1))
        records.append(dict(R=R,checks=checks,new_rule=new,
                            construction_node_change=node_change))
    result=dict(anchor=short.record(),tests=records,
        interpretation="Analytic thermal kernel moments only, not a CFT convergence test.",
        existing_CFT_banks_modified=False,all_passed=True)
    (HERE/"thermal_cut_rule_checks.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"all_passed":True,"cases":len(records)*4,
        "old_constant_residual_relative_errors":[r["checks"][0]["old_relative_error"] for r in records]}))


if __name__=="__main__":validate()
