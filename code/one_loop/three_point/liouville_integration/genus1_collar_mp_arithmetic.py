"""Multiprecision evaluation of unchanged finite collar polynomials.

Saved native-q coefficients, DOZZ values and quadrature are not recomputed.
This is an arithmetic fallback, not a block/momentum convergence certificate.
The float64 parent-cancellation proxy is replaced by a direct40/60-digit
comparison AND propagation of the saved coefficient precision diagnostics.
The relative arithmetic tolerance remains5e-4. No failed value is accepted
merely because its amplitude contribution appears small.
"""
import math
import time
import mpmath
import numpy as np


def _context(dps):
    ctx=mpmath.mp.clone();ctx.dps=dps
    return ctx


def _prime(ctx,z,tau):
    """Exact prime-form product in the centered cell, to working precision."""
    z=ctx.mpc(z);tau=ctx.mpc(tau)
    z-=2*ctx.pi*tau*ctx.floor(z.imag/(2*ctx.pi*tau.imag)+ctx.mpf('.5'))
    z-=2*ctx.pi*ctx.floor(z.real/(2*ctx.pi)+ctx.mpf('.5'))
    out=ctx.log(abs(2*ctx.sin(z/2)))-z.imag**2/(4*ctx.pi*tau.imag)
    q=ctx.exp(2j*ctx.pi*tau);qn=q;plus=ctx.exp(1j*z);minus=1/plus
    for _ in range(10000):
        if abs(qn)*max(abs(plus),abs(minus),1)<ctx.eps:break
        out+=ctx.log(abs((1-qn*plus)*(1-qn*minus)/(1-qn)**2));qn*=q
    else:raise FloatingPointError('prime product did not converge')
    return out


def _common(ctx,tau):
    q=ctx.exp(2j*ctx.pi*tau);qn=q;out=-ctx.log(tau.imag)/2
    for _ in range(10000):
        if abs(qn)<ctx.eps:break
        out+=2*ctx.log(abs(1-qn));qn*=q
    else:raise FloatingPointError('oscillator product did not converge')
    return out


def _flat(ctx,delta,d0,d1):
    return 1j*(d1-d0)*delta/2-(d0+d1)*ctx.log(2*ctx.sin(delta/2))


def _loop(ctx,total,tau):
    z=-total;y=z.imag/(2*ctx.pi*tau.imag)
    x=(z.real-2*ctx.pi*tau.real*y)/(2*ctx.pi)
    return z-2*ctx.pi*(ctx.floor(x)+ctx.floor(y)*tau)


def _polynomial(ctx,coeff,cut,powers,error):
    terms=[];uncertainty=ctx.mpf(0);rounding=ctx.mpf(2)**-52
    for i in range(cut[0]+1):
        for j in range(cut[1]+1):
            for k in range(cut[2]+1):
                mon=powers[0][i]*powers[1][j]*powers[2][k]
                raw=complex(coeff[i,j,k]);val=ctx.mpc(raw.real,raw.imag)
                terms.append(val*mon)
                # The normalized identity coefficient is exactly1, not a
                # separately rounded independent coefficient in each parent.
                err=0 if (i,j,k)==(0,0,0) and raw==1 else error*max(1,abs(val))+rounding*abs(val)
                uncertainty+=err*abs(mon)
    return ctx.fsum(terms),uncertainty


def _evaluate(atlas,selected,request,dps):
    from genus1_channel_evaluation import require_native
    from genus1_native_bank import variants
    from genus1_three_point_collision import koba_exponent,reconstruct_points
    from genus1_pair_collar import _pair_remainder_geometry
    from genus1_comb_collar import _geometry
    if selected.fallback_full is not None:
        raise FloatingPointError('MP cross-chart fallback requires a separately validated adapter')
    bank,orders=require_native(selected);g=request.geometry
    if request.kind not in ('pair_remainder','comb_collar'):
        raise FloatingPointError('unsupported MP arithmetic request')
    if request.kind=='comb_collar' and g['term'] not in ('forest_remainder','pair_face'):
        raise FloatingPointError('unsupported MP comb term')
    if request.kind=='pair_remainder':
        # Retain the existing identity, actual chart and region-membership gates.
        _pair_remainder_geometry(bank,atlas,g['pair'],g['delta_pair'],g['delta_total'],g['tau'],g['radii'])
    else:
        _geometry(bank,atlas,g['pair'],g['tau'])
        if g['term']=='pair_face':orders=(0,orders[1],orders[2])
    ctx=_context(dps);tau=ctx.mpc(g['tau']);pair=tuple(g['pair']);s=next(i for i in range(3) if i not in pair)
    d=tuple(ctx.mpf(complex(atlas.external_weights[i]).real) for i in (*pair,s))
    k=tuple(ctx.mpf(complex(koba_exponent(atlas,i,j)).real) for i,j in ((pair[0],pair[1]),(pair[0],s),(pair[1],s)))
    common=_common(ctx,tau)-3*ctx.log(ctx.pi)
    loopq=ctx.exp(2j*ctx.pi*tau)
    reverse=False;full_bank=bank
    if request.kind=='pair_remainder':
        delta=ctx.mpc(g['delta_pair']);total=ctx.mpc(g['delta_total']);radius=abs(delta)
        loop=_loop(ctx,total,tau);looplogs=(1j*loop,1j*(2*ctx.pi*tau-loop))
        v=ctx.expm1(-1j*delta);fullvars=(v,ctx.exp(looplogs[0]),ctx.exp(looplogs[1]))
        projvars=(ctx.mpc(0),fullvars[1],fullvars[2])
        flat=_flat(ctx,delta,d[0],d[1])
        koba=k[0]*_prime(ctx,delta,tau)+k[1]*_prime(ctx,delta-total,tau)+k[2]*_prime(ctx,total,tau)
        spectator=(k[1]+k[2])*_prime(ctx,total,tau)
        configurations=[(bank,fullvars,1),(bank,projvars,-1)]
    else:
        scale=ctx.mpf(g['scale']);total=scale*ctx.exp(1j*ctx.mpf(g['phase']))
        v=ctx.expm1(-1j*total);pairvars=(ctx.mpc(0),v,loopq);cornervars=(ctx.mpc(0),ctx.mpc(0),loopq)
        sinlog=ctx.log(2*ctx.sin(total/2));totalprime=_prime(ctx,total,tau)
        if g['term']=='forest_remainder':
            shape=ctx.mpc(g['shape']);delta=shape*total
            if not (0<abs(shape)<1 and 0<float(scale)<g['radii'].triple and float(abs(delta))<g['radii'].pair):
                raise ValueError('invalid forest membership')
            w1,w2=reconstruct_points(pair,complex(delta),complex(total))
            actual=pair
            if not atlas.sewing_chart(w1,w2,g['tau'],'comb_ope',actual).admissible:
                actual=pair[::-1];atlas.sewing_chart(w1,w2,g['tau'],'comb_ope',actual).require();reverse=True
                full_bank=selected.reversed_bank
                if full_bank is None:raise ValueError('missing verified reversed bank')
                for name in ('momenta','log_weights','structure_logs'):
                    if not np.array_equal(bank.arrays[name],full_bank.arrays[name]):raise ValueError('reversal changed common density')
            from genus1_native_bank import identity
            expected=identity(dict(channel='comb_ope',coefficient_basis='native-sewing-v2',
                external_weights=tuple(atlas.external_weights[i] for i in (*actual,s)),
                external_momenta=tuple(atlas.external_momenta[i] for i in (*actual,s))))
            if full_bank.metadata['identity']!=expected:raise ValueError('wrong MP full-bank graph')
            if reverse:fd,ft,dd=(-delta,total-delta,(d[1],d[0],d[2]))
            else:fd,ft,dd=(delta,total,d)
            u=ctx.expm1(-1j*fd);y=ctx.expm1(-1j*ft)
            fullvars=(u/y,y,loopq);scalevars=(shape,ctx.mpc(0),loopq)
            second=1j*(1-dd[2])*ft/2-(1+dd[2])*ctx.log(2*ctx.sin(ft/2))
            constant=2*ctx.log(abs(u))+2*ctx.log(abs(y))+2*(_flat(ctx,fd,dd[0],dd[1])+second).real
            constant+=k[0]*_prime(ctx,delta,tau)+k[1]*_prime(ctx,delta-total,tau)+k[2]*_prime(ctx,total,tau)
            configurations=[(full_bank,fullvars,1),(bank,pairvars,-1),(bank,cornervars,1),(bank,scalevars,-1)]
        else:
            upper=ctx.mpf(g['shape_upper'])
            if not 0<upper<=1:raise ValueError('invalid pair-face shape bound')
            configurations=[(bank,pairvars,1),(bank,cornervars,-1)]
    # Verify the actual (not elliptic) finite-series coordinates.
    if any(abs(q)>=1 for _,vs,_ in configurations for q in vs):raise ValueError('noncontracting MP native coordinate')
    powers=[tuple([q**n for n in range(order+1)] for q,order in zip(vs,orders)) for _,vs,_ in configurations]
    cuts=list(dict.fromkeys(cut for _,cut in variants(orders)))
    values={};errors={};conditions={};absolute_sums={}
    for cut in cuts:
        nodes=[];bounds=[];parents=[]
        for ni,ps in enumerate(bank.arrays['momenta']):
            pp=tuple(ctx.mpf(float(p))**2 for p in ps);h=tuple(1+p for p in pp)
            raw=complex(bank.arrays['structure_logs'][ni]);base=ctx.mpc(raw.real,raw.imag)+ctx.mpf(float(bank.arrays['log_weights'][ni]))+common
            if request.kind=='pair_remainder':
                base+=2*pp[1]*looplogs[0].real+2*pp[2]*looplogs[1].real
                beta=2*h[0]-2*(d[0]+d[1])+k[0]+2
                primary=[2*h[0]*ctx.log(abs(v))+2*flat.real+koba,(beta-2)*ctx.log(radius)+spectator]
            else:
                base-=4*ctx.pi*tau.imag*pp[2]
                aa=2*h[0]-2*(d[0]+d[1])+k[0]
                gamma=2*h[1]-2*sum(d)+sum(k)
                pair_primary=2*h[1]*ctx.log(abs(v))+2*(1j*(h[0]-d[2])*total/2-(h[0]+d[2])*sinlog).real+(k[1]+k[2])*totalprime
                if g['term']=='forest_remainder':
                    local=gamma*ctx.log(scale)+aa*ctx.log(abs(shape))
                    primary=[constant+2*pp[0]*ctx.log(abs(fullvars[0]))+2*pp[1]*ctx.log(abs(fullvars[1])),
                        pair_primary+aa*ctx.log(scale*abs(shape)),local,local+k[1]*ctx.log(abs(1-shape))]
                else:
                    if aa+2<=0 or gamma+4<=0:raise ValueError('nonconvergent MP radial term')
                    radial=(aa+2)*ctx.log(upper)-ctx.log(aa+2)
                    primary=[pair_primary+aa*ctx.log(scale)+radial,gamma*ctx.log(scale)+radial]
            contributions=[];node_errors=[]
            for ci,(b,_,sign) in enumerate(configurations):
                poly,unc=_polynomial(ctx,b.arrays['coefficients'][ni],cut,powers[ci],ctx.mpf(float(b.arrays['coefficient_precision_bounds'][ni])))
                weight=ctx.exp(base+primary[ci]);value=weight*abs(poly)**2
                contributions.append(sign*value)
                node_errors.append(abs(weight)*(2*abs(poly)*unc+unc**2))
            node=ctx.fsum(contributions)
            nodes.append(node);parents.append(ctx.fsum(abs(x) for x in contributions))
            # Shared DOZZ/quadrature storage uncertainty multiplies the
            # residual itself, not independent copies of its large parents.
            bounds.append(ctx.fsum(node_errors)+abs(node)*128*(ctx.mpf(2)**-52)*(1+abs(base)))
        value=ctx.fsum(nodes);absolute=ctx.fsum(parents)
        values[cut]=value;errors[cut]=ctx.fsum(bounds)+absolute*ctx.eps*10000
        conditions[cut]=absolute/abs(value) if value else ctx.inf
        absolute_sums[cut]=absolute
    return dict(values=values,errors=errors,conditions=conditions,parent_absolute_sums=absolute_sums,orders=orders)


def evaluate(atlas,selected,request):
    from genus1_native_bank import variants
    started=time.monotonic();low=_evaluate(atlas,selected,request,40);high=_evaluate(atlas,selected,request,60)
    values=[];checks=[];ctx=_context(70)
    for label,cut in variants(high['orders']):
        value=high['values'][cut];error=high['errors'][cut]+32*abs(value-low['values'][cut])
        if value:
            relative=error/abs(value);passed=relative<=ctx.mpf('0.0005')
        else:relative=ctx.inf;passed=False
        raw=complex(value)
        logbound=float(ctx.log(error)) if error else -math.inf
        if abs(value)<ctx.mpf(2)**-1074:
            # Retain the WHOLE density, not just its arithmetic error, as an
            # explicit absolute underflow enclosure through the MC Jacobian.
            logbound=float(ctx.log(abs(value)+error));passed=logbound < -745;raw=0j
        if not np.isfinite(raw):raise FloatingPointError('MP result not representable')
        values.append(raw);checks.append(dict(prefix=label,orders=list(cut),passed=bool(passed),
            estimated_relative_arithmetic_loss=float(relative) if ctx.isfinite(relative) and relative<1e300 else None,
            absolute_arithmetic_log_bound=logbound,
            parent_to_residual_condition=float(high['conditions'][cut]) if high['conditions'][cut]<1e300 else None))
    report=dict(passed=all(r['passed'] for r in checks),prefixes=checks,working_digits=[40,60],
        coefficient_recomputations=0,dozz_recomputations=0,seconds=time.monotonic()-started,
        qualification='Fixed-bank MP arithmetic comparison and empirical coefficient-input propagation; no truncation or momentum certificate.')
    return np.asarray(values),report,high['orders']
