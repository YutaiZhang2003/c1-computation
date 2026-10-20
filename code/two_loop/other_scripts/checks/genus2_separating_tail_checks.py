#!/usr/bin/env python3
"""Independent identities and optional full-CFT overlap checks for the tail."""
from __future__ import annotations

import argparse
import json
import math

import mpmath as mp
import numpy as np
from scipy.special import roots_genlaguerre

from genus2_separating_tail import (
    RealUpsilonOne, bridge_length, handle_expectation, integrate_primary_tail,
    integrate_separating_residual, primary_scaled_kernel_batch,
    leading_integrated_tail, primary_scaled_kernel, reduced_diagonal_dozz,
    separating_length_proposal, separating_proposals_from_unit_cube,
    torus_elliptic_block, torus_lattice_sum,
)
from liouville_torus import UpsilonB, log_yin_structure_constant_momentum
from virasoro_blocks import TorusOnePointVirasoroBlock


def run_checks(upsilon):
    special = UpsilonB(1., dps=44)
    dozz_errors = []
    for a in (.001,.2,.8,2.):
        for p in (.0001,.1,.5,1.5):
            reference = float(mp.re(mp.exp(log_yin_structure_constant_momentum(
                special,a,p,a))))/(8*a*a*p)
            value = float(reduced_diagonal_dozz(a,p,upsilon))
            dozz_errors.append(abs(value/reference-1))
    assert max(dozz_errors) < 1e-8
    assert np.array_equal(reduced_diagonal_dozz([0.,.5,2.],0.,upsilon),np.ones(3))
    block_errors = []
    for a in (.01,.4,2.):
        for p in (0.,.3,1.4):
            for order in (0,1,2,3,4):
                q = .003+.002j
                reference = TorusOnePointVirasoroBlock(25.,1+a*a,1+p*p).elliptic_block(q,order)
                block_errors.append(abs(complex(torus_elliptic_block(q,a,p,order))-reference))
    assert max(block_errors) < 1e-13
    assert abs(complex(torus_elliptic_block(.003j,.4,0.,8))-1) < 1e-14
    # The matched proposal makes the leading radial importance weight constant.
    length,logrho,loginv = separating_length_proposal(np.array([0.,.3,.9,.99999]),.01)
    leading_weight = np.exp(loginv-1.5*np.log(length))
    assert np.max(np.abs(leading_weight/leading_weight[0]-1)) < 1e-14
    assert np.all(np.isfinite(logrho)) and logrho[-1] < -1e9
    for bad in (-.1,1.,float('nan')):
        try:
            separating_length_proposal(bad,.01)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid proposal coordinate accepted")
    # Gaussian radial integration, including cancellation of P² and angle pi.
    l0 = bridge_length(.01)
    x,w = roots_genlaguerre(32,-.5)
    radial = float(np.sum(w)/(2*math.sqrt(2*l0)))
    assert math.isclose(radial,math.sqrt(math.pi)/(2*math.sqrt(2*l0)),rel_tol=1e-14)
    x = np.array([.12,-.21]);y = np.array([1.2,1.5])
    lattice = torus_lattice_sum(x,y,np.array([.5,1.,2.,15.,1/15]))
    assert np.max(np.abs(lattice[:,0]/lattice[:,2]-1)) < 1e-13
    assert np.max(np.abs(lattice[:,3]/lattice[:,4]-1)) < 1e-13
    scaled = primary_scaled_kernel(.12+1.2j,-.21+1.5j,1e6,[1.],upsilon=upsilon)
    expected = lattice[0,1]*lattice[1,1]/(1.2*1.5)**1.5/(32*math.sqrt(2)*math.pi**1.5)
    assert abs(float(scaled[0])*1e9/expected-1) < 1e-5
    # Threshold J is known exactly; checks the two torus Jacobians and factors.
    tail = integrate_primary_tail([.5,1.,2.],1e-100,upsilon=upsilon,power=10,
                                  replicates=4,momentum_order=16,bridge_order=24)
    leading = leading_integrated_tail(tail.radii,tail.rho0)
    assert np.max(np.abs(tail.mean/leading-1)) < .015  # finite L subleading terms
    assert abs(tail.mean[0]/tail.mean[2]-4) < 1e-12
    # Check the complete six-real-dimensional Jacobian, folding, and angle.
    from scipy.stats import qmc
    from study_genus2_separating_tail_correction import assemble_control_variate
    unit=qmc.Sobol(6,scramble=True,seed=38191).random_base2(8)
    proposal=separating_proposals_from_unit_cube(unit,.01)
    assert np.all(proposal.tau1.imag <= proposal.tau2.imag)
    synthetic=np.exp(proposal.log_weight_over_rho_squared
        -1.5*np.log(proposal.tau1.imag*proposal.tau2.imag*proposal.length))
    a1=np.sqrt(1-(unit[:,0]-.5)**2);a2=np.sqrt(1-(unit[:,2]-.5)**2)
    np.testing.assert_allclose(synthetic,4*math.pi/np.sqrt(l0*a1*a2),rtol=2e-14)
    # Adding the tail to the original without subtracting its sampled control
    # is a double-counting bug. An exact synthetic control has zero residual.
    sampled=np.array([[1.,2.],[3.,4.],[7.,8.]])
    integrated=np.tile([9.,10.],(4,1));rejected=np.tile([.1,.2],(4,1))
    mean,cov,residual=assemble_control_variate(sampled,sampled,integrated,rejected)
    np.testing.assert_allclose(mean,[8.9,9.8],rtol=1e-14)
    assert np.max(np.abs(cov)) < 1e-28 and np.all(residual==0.)
    def exact_synthetic_control(proposal,radii):
        return primary_scaled_kernel_batch(proposal.tau1,proposal.tau2,proposal.length,
            radii,upsilon=upsilon,momentum_order=8,bridge_order=12)
    tail_residual=integrate_separating_residual(exact_synthetic_control,[1.,2.],.01,
        upsilon=upsilon,power=5,replicates=2,momentum_order=8,bridge_order=12)
    assert np.max(np.abs(tail_residual.replicates)) < 1e-15
    return dict(status="passed",upsilon_max_abs_error=upsilon.validation_max_abs_error,
                dozz_max_relative_error=max(dozz_errors),block_max_abs_error=max(block_errors),
                leading_large_length_ratios=(tail.mean/leading).tolist(),
                proposal_jacobian_and_control_subtraction='passed')


def full_overlap(upsilon, *, bridges=(.02,.005,.001), block_order=4, momentum_order=10):
    from free_boson_plumbing import igusa_chi10_evaluation_genus2
    from genus2_c1_string_integrand import compact_boson_winding_evaluation_genus2
    from genus2_hybrid_period_map import _collocation_at_order
    from genus2_liouville_evaluation import evaluate_liouville_rescaled
    from genus2_scalar_evaluation import evaluate_noncompact_scalar_detailed
    results = []
    for qb in bridges:
        q = (np.exp(2j*math.pi*(.12+1.2j)),np.exp(2j*math.pi*(-.21+1.5j)),qb*(.6+.8j))
        omega,_,_ = _collocation_at_order('glasses',q,32)
        omega_lower,_,_ = _collocation_at_order('glasses',q,24)
        liouville = evaluate_liouville_rescaled('glasses',q,block_order=block_order,
            quadrature_order=momentum_order,quadrature_scheme='primary-gaussian',dps=44,
            vacuum_word_length=8,vacuum_oscillator_level=40)
        scalar = evaluate_noncompact_scalar_detailed('glasses',q,omega,word_length=8,
                                                     max_mode=60,tolerance=1e-14)
        chi = igusa_chi10_evaluation_genus2(omega,tol=1e-15)
        rho = abs(omega[0,1]);length=bridge_length(rho)
        primary = primary_scaled_kernel(omega[0,0],omega[1,1],length,[1.,2.,15.],
            upsilon=upsilon,momentum_order=24,bridge_order=32,block_order=4)
        ratios = []
        full_scaled = []
        for i,radius in enumerate((1.,2.,15.)):
            theta=compact_boson_winding_evaluation_genus2(omega,radius,tolerance=1e-13).value
            logk=(math.log(8*math.pi**4)+24*math.log(2)+math.log(2*math.pi*radius*theta)
                  -13*math.log(np.linalg.det(omega.imag))-2*chi.log_abs
                  +liouville.log_partition-25*math.log(scalar.partition))
            scaled=math.exp(logk+2*math.log(rho))
            full_scaled.append(scaled);ratios.append(scaled/primary[i])
        result=dict(q_bridge=qb,rho=rho,period_step=float(np.max(np.abs(omega-omega_lower))),
                    full_scaled=full_scaled,primary_scaled=primary.tolist(),ratios=ratios,
                    block_order=block_order,momentum_order=momentum_order)
        print(json.dumps(result),flush=True)
        results.append(result)
    return results


def convergence_checks(upsilon):
    from scipy.stats import qmc
    quadrature=[]
    for rho,orders in ((.02,(16,24,32)),(.08,(20,32,48))):
        previous=None
        for order in orders:
            value=integrate_primary_tail([1.,2.,15.],rho,upsilon=upsilon,power=10,
                replicates=4,momentum_order=order,bridge_order=order,seed=33901)
            delta=None if previous is None else float(np.max(np.abs(value.mean/previous-1)))
            row=dict(rho=rho,order=order,mean=value.mean.tolist(),
                     empirical_standard_error=value.standard_error.tolist(),relative_step=delta)
            quadrature.append(row);print(json.dumps(row),flush=True);previous=value.mean
        assert delta < 1e-6
    blocks=[]
    for order in (0,1,2,4):
        value=integrate_primary_tail([1.,2.,15.],.08,upsilon=upsilon,power=7,
            replicates=2,momentum_order=12,bridge_order=16,block_order=order,seed=43901)
        row=dict(block_order=order,mean=value.mean.tolist())
        blocks.append(row);print(json.dumps(row),flush=True)
    assert np.max(np.abs(np.array(blocks[-1]['mean'])/blocks[-2]['mean']-1)) < 1e-8
    proposal_means=[];underflows=[]
    for k in range(4):
        unit=qmc.Sobol(6,scramble=True,seed=44821+k).random_base2(10)
        proposal=separating_proposals_from_unit_cube(unit,.02)
        weighted=proposal.weighted_primary([1.,2.,15.],upsilon=upsilon,
                                          momentum_order=16,bridge_order=24)
        proposal_means.append(weighted.mean(axis=0));underflows.append(int(sum(proposal.log_rho < -740)))
    proposal_means=np.array(proposal_means)
    matched=dict(mean=proposal_means.mean(axis=0).tolist(),
        empirical_standard_error=(proposal_means.std(axis=0,ddof=1)/2).tolist(),
        underflowing_rho_count=underflows,replicates=proposal_means.tolist())
    print(json.dumps(matched),flush=True)
    return dict(quadrature=quadrature,blocks=blocks,matched_log_proposal=matched)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--full-overlap',action='store_true')
    parser.add_argument('--momentum-order',type=int,default=10)
    parser.add_argument('--output',type=str)
    parser.add_argument('--convergence',action='store_true')
    args=parser.parse_args()
    upsilon=RealUpsilonOne()
    report=run_checks(upsilon)
    print(json.dumps(report,indent=2),flush=True)
    if args.full_overlap:
        report['full_overlap']=full_overlap(upsilon,momentum_order=args.momentum_order)
    if args.convergence:
        report['convergence']=convergence_checks(upsilon)
    if args.output:
        from pathlib import Path
        path=Path(args.output)
        if path.exists():
            raise FileExistsError(f'Refusing to overwrite {path}')
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
