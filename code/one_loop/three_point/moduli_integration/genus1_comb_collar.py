"""Shared-bank triple forest and analytic faces in native comb coordinates.

The full tensor is required: the scale face retains C[i,0,k] and its shape
series. Forest subtraction is performed at each momentum node, in signed
logarithms. Every returned term includes dP/pi^3 and the common factor once,
but not sampling Jacobians. This local layer is not a regional certificate.
"""
import cmath
import math
import numpy as np

from genus1_native_bank import SharedNativeBank, FORMAT, identity, point_kinematics, variants
from genus1_rebuild_policy import require_orders, classify_cft_point, encoded, weighted_cft_record
from genus1_pair_collar import _log_positive, _logsum, PAIR_REMAINDER_NUMERICS_RTOL, PAIR_REMAINDER_MAX_CONDITION

FLOAT=np.finfo(float)
TERMS=('full','pair_projection','scale_projection','corner_projection','forest_remainder',
       'pair_face','scale_face','corner_integrated')


def _difference(a,b):
    """Signed log(exp(a)-exp(b)), including exactly zero components."""
    a,b=np.broadcast_arrays(np.asarray(a,float),np.asarray(b,float))
    if np.any(np.isnan(a)|np.isnan(b)|np.isposinf(a)|np.isposinf(b)):
        raise FloatingPointError('invalid signed-log input')
    logs=np.full(a.shape,-np.inf);sign=np.zeros(a.shape)
    active=a!=b
    logs[active]=np.maximum(a[active],b[active])+np.log(-np.expm1(-abs(a[active]-b[active])))
    sign[active]=np.where(a[active]>b[active],1.,-1.)
    return logs,sign


def _signed_add(a,sa,b,sb):
    a,sa,b,sb=np.broadcast_arrays(a,sa,b,sb)
    result=np.full(a.shape,-np.inf);sign=np.zeros(a.shape)
    both=(sa!=0)&(sb!=0);same=both&(sa==sb);different=both&~same
    result[same]=np.logaddexp(a[same],b[same]);sign[same]=sa[same]
    logs,direction=_difference(a[different],b[different])
    result[different]=logs;sign[different]=sa[different]*direction
    onlya=(sa!=0)&(sb==0);onlyb=(sa==0)&(sb!=0)
    result[onlya]=a[onlya];sign[onlya]=sa[onlya]
    result[onlyb]=b[onlyb];sign[onlyb]=sb[onlyb]
    return result,sign


def _block(bank,cut,variables):
    coeff=bank.arrays['coefficients'][(slice(None),*[slice(n+1) for n in cut])]
    scales=np.max(abs(coeff),axis=(1,2,3));scales=np.where(scales>0,scales,1.)
    norm=coeff/scales[:,None,None,None]
    powers=[z**np.arange(n+1) for z,n in zip(variables,cut)]
    if (np.any((coeff!=0)&(abs(norm)<FLOAT.tiny)) or
            any(z!=0 and np.any(abs(p)<FLOAT.tiny) for z,p in zip(variables,powers))):
        raise FloatingPointError('comb tensor/powers exceed float64 range')
    value=np.einsum('tijk,i,j,k->t',norm,*powers,optimize=False)
    absolute=np.einsum('tijk,i,j,k->t',abs(norm),*[abs(p) for p in powers],optimize=False)
    if not np.all(np.isfinite(value)) or not np.all(np.isfinite(absolute)):
        raise FloatingPointError('nonfinite comb contraction')
    if np.any((absolute<FLOAT.tiny)&np.any(norm!=0,axis=(1,2,3))):
        raise FloatingPointError('underflowed absolute contraction')
    scale=2*np.log(scales)
    return scale+2*_log_positive(abs(value)),scale+2*_log_positive(absolute)


def _geometry(bank,atlas,pair,tau):
    from genus1_three_point_collision import pair_spectator,koba_exponent,_common_factor
    if (type(bank) is not SharedNativeBank or bank.metadata.get('format')!=FORMAT or
            bank.metadata.get('evaluation_scope','full_native_cft')!='full_native_cft'):
        raise ValueError('triple terms require a full native comb bank, not a projection-only slice')
    pair=tuple(pair);s=pair_spectator(pair);tau=complex(tau)
    if not all(math.isfinite(v) for v in (tau.real,tau.imag)) or tau.imag<=0 or abs(tau.real)>.5 or abs(tau)<1-8*FLOAT.eps:
        raise ValueError('modular-domain tau required by this local adapter')
    external=tuple(atlas.external_weights[i] for i in (*pair,s))
    key=identity(dict(channel='comb_ope',coefficient_basis='native-sewing-v2',external_weights=external,
        external_momenta=tuple(atlas.external_momenta[i] for i in (*pair,s))))
    if bank.metadata['identity']!=key:raise ValueError('incompatible comb graph or external frequency')
    kappas=tuple(koba_exponent(atlas,i,j) for i,j in ((pair[0],pair[1]),(pair[0],s),(pair[1],s)))
    if any(complex(x).imag!=0 for x in (*external,*kappas)):
        raise ValueError('this local signed-log adapter requires real external weights/exponents')
    d=np.asarray(external,complex).real;k=np.asarray(kappas,complex).real
    h=1+bank.arrays['momenta']**2
    a=2*h[:,0]-2*(d[0]+d[1])+k[0]
    gamma=2*h[:,1]-2*((d[0]+d[1])+d[2])+(k[0]+k[1])+k[2]
    common=cmath.log(_common_factor(atlas,tau))
    base=bank.arrays['structure_logs']+bank.arrays['log_weights']-3*math.log(math.pi)+common
    base=base-4*math.pi*tau.imag*(h[:,2]-1)
    q=cmath.exp(2j*math.pi*tau)
    return dict(pair=pair,tau=tau,spectator=s,d=d,k=k,h=h,a=a,beta=a+2,gamma=gamma,
        lam=gamma+4,base=base,q=q,common=common)


def _positive_radius(value,label):
    value=float(value)
    if not math.isfinite(value) or not 0<value<math.pi:raise ValueError('invalid '+label)
    return value


def _shape(value):
    u=complex(value)
    if not all(math.isfinite(v) for v in (u.real,u.imag)) or not 0<abs(u)<1 or u.real>.5+8*FLOAT.eps:
        raise ValueError('shape must lie inside the nearest-pair sewing sector')
    return u


def _pair_log(g,total):
    from genus1_sewing_domains import comb_chart
    from genus1_three_point_worldsheet import torus_prime_form_log_norm
    chart=comb_chart(.01*total,total,2j*math.pi*g['tau'])
    if chart.margins[1]<=1e-12:raise ValueError('comb pair-face outer contour is inadmissible')
    v=complex(np.expm1(-1j*total))
    if v==0:raise ValueError('nonzero outer sewing coordinate required')
    flat=.5j*(g['h'][:,0]-g['d'][2])*total-(g['h'][:,0]+g['d'][2])*cmath.log(2*cmath.sin(total/2))
    primary=2*g['h'][:,1]*math.log(abs(v))+2*flat.real
    primary+=(g['k'][1]+g['k'][2])*torus_prime_form_log_norm(total,g['tau'])
    return primary,v


def _log_interval(lower,upper,exponent):
    """Log positive radial integral, continuous at exponent=0."""
    e=np.asarray(exponent,float)
    if not 0<lower<=upper:raise ValueError('invalid corner radial interval')
    if lower==upper:return np.full(e.shape,-np.inf)
    length=math.log(upper/lower);x=e*length;out=np.empty(e.shape)
    small=abs(x)<1e-6
    # expm1(x)/x=1+x/2+x^2/6+...; avoid the zero exponent division entirely.
    out[small]=e[small]*math.log(lower)+math.log(length)+np.log1p(x[small]/2+x[small]**2/6+x[small]**3/24)
    high=(~small)&(x>50);other=(~small)&~high
    out[high]=e[high]*math.log(lower)+x[high]+np.log1p(-np.exp(-x[high]))-np.log(abs(e[high]))
    out[other]=e[other]*math.log(lower)+np.log(abs(np.expm1(x[other])))-np.log(abs(e[other]))
    return out


def _reduce(bank,g,components,signs,geometry_scale,cut):
    # Each component is (signed-density log, absolute coefficient bound log).
    logs=np.full(len(g['base']),-np.inf);direction=np.zeros(len(logs))
    # Pair near-equal terms before combining their residuals.
    for start in range(0,len(components),2):
        if start+1<len(components) and signs[start]==-signs[start+1]:
            part,sgn=_difference(components[start][0],components[start+1][0]);sgn*=signs[start]
        else:
            part=components[start][0];sgn=np.where(np.isfinite(part),float(signs[start]),0.)
            if start+1<len(components):raise ValueError('unsupported signed component ordering')
        logs,direction=_signed_add(logs,direction,part,sgn)
    active=direction!=0;z=g['base'][active]+logs[active]
    if np.any(z.real>math.log(FLOAT.max)) or not np.all(np.isfinite(z)):
        raise FloatingPointError('unresolved forest overflow/nonfinite density')
    nodes=np.zeros(len(logs),complex);nodes[active]=direction[active]*np.exp(z)
    value=complex(math.fsum(nodes.real),math.fsum(nodes.imag))
    if not np.isfinite(value):raise FloatingPointError('nonfinite forest sum')
    parent=np.logaddexp.reduce(np.asarray([x[0] for x in components]),axis=0)
    bound=np.logaddexp.reduce(np.asarray([x[1] for x in components]),axis=0)
    budget=128*FLOAT.eps*(math.prod(n+1 for n in cut)+geometry_scale+1+abs(g['base'])
        +np.where(np.isfinite(parent),abs(parent),0))+4*bank.arrays['coefficient_precision_bounds']
    log_value=math.log(abs(value)) if abs(value)>=FLOAT.tiny else -math.inf
    condition_log=max(0.,_logsum(g['base'].real+parent)-log_value)
    error=_logsum(g['base'].real+bound+np.log(budget))
    lost=g['base'].real+logs
    error=float(np.logaddexp(error,_logsum(lost[abs(nodes)<FLOAT.tiny])))
    relative_log=error-log_value
    def finite(log):return math.exp(log) if log<=math.log(FLOAT.max) else None
    reasons=[]
    if not math.isfinite(log_value):reasons.append('zero_or_subnormal_term')
    if condition_log>math.log(PAIR_REMAINDER_MAX_CONDITION):reasons.append('cancellation_condition_limit')
    if relative_log>math.log(PAIR_REMAINDER_NUMERICS_RTOL):reasons.append('estimated_arithmetic_loss_limit')
    return value,dict(passed=not reasons,reasons=reasons,parent_to_residual_condition=finite(condition_log),
        estimated_relative_arithmetic_loss=finite(relative_log),
        absolute_arithmetic_log_bound=float(error) if math.isfinite(error) else None,
        subtraction_before_momentum_sum=True)


def evaluate_projected_comb(bank,atlas,*,term,pair,tau,scale=None,shape=None,phase=0.,shape_upper=None,
                         theta=None,radii=None,orders=None,diagnostics=False):
    from genus1_three_point_collision import CollisionRadii,reconstruct_points,triple_shape_radius
    if term not in TERMS or type(diagnostics) is not bool:raise ValueError('explicit supported triple term required')
    if term in ('full','forest_remainder') and bank.metadata.get('projection_only',False):
        raise ValueError('a projection-only bank cannot represent the full comb parent')
    cut_orders=tuple(require_orders(bank.orders if orders is None else orders))
    if any(a>b for a,b in zip(cut_orders,bank.orders)):raise ValueError('missing comb coefficients; no automatic extension')
    g=_geometry(bank,atlas,pair,tau);geometry_scale=1.
    if term!='corner_integrated':scale=_positive_radius(scale,'triple scale')
    if term in ('full','pair_projection','scale_projection','corner_projection','forest_remainder','scale_face'):
        shape=_shape(shape);geometry_scale=max(1.,2*math.pi*abs(g['tau']))/(scale*abs(shape))
    if term in ('full','pair_projection','forest_remainder','pair_face'):
        if not math.isfinite(float(phase)):raise ValueError('finite comb phase required')
        total=scale*cmath.exp(1j*float(phase));primary,v=_pair_log(g,total)
    if term in ('full','forest_remainder'):
        if not isinstance(radii,CollisionRadii) or not scale<radii.triple or not scale*abs(shape)<radii.pair:
            raise ValueError('full forest geometry must be strictly inside the resolved triple sector')
        w1,w2=reconstruct_points(g['pair'],shape*total,total)
        point=dict(w1=encoded(w1),w2=encoded(w2),tau=encoded(g['tau']),channel='comb_ope',pair=list(g['pair']))
        kin=point_kinematics(atlas,point)
        exact_primary=(kin['constant']+(kin['common_log']-g['common']).real
            -np.einsum('ti,i->t',bank.arrays['momenta'][:,:2]**2,kin['decay'][:2]))
    if term in ('pair_face','scale_face','corner_integrated') and (np.any(g['beta']<=0) or np.any(g['lam']<=0)):
        raise ValueError('nonconvergent analytic radial integral')
    if term=='pair_face':
        shape_upper=float(shape_upper)
        if not math.isfinite(shape_upper) or not 0<shape_upper<=1:raise ValueError('invalid shape integration bound')
        geometry_scale=max(1.,2*math.pi*abs(g['tau']))/scale
    if term=='corner_integrated':
        if not isinstance(radii,CollisionRadii) or theta is None or not math.isfinite(float(theta)):
            raise ValueError('explicit corner radii and finite theta required')
        upper=triple_shape_radius(theta);split=radii.pair/radii.triple
        radial=np.logaddexp(g['lam']*math.log(radii.triple)-np.log(g['lam'])
            +g['beta']*math.log(split)-np.log(g['beta']),
            g['lam']*math.log(radii.pair)-np.log(g['lam'])+_log_interval(split,upper,g['beta']-g['lam']))
        radial+=math.log(2*math.pi)
    values=[];checks=[]
    def block(cut,variables):
        # Retain the logarithmic torus nome in the infinite modular tail.
        # The fast branch is the identical original finite polynomial.
        if 2*math.pi*g['tau'].imag*cut[2]<650:
            try:return _block(bank,cut,variables)
            except FloatingPointError:pass
        from genus1_native_log_evaluation import block_logs,projected_logs
        return block_logs(bank,cut,(*projected_logs(variables[:2]),2j*math.pi*g['tau']))
    for label,cut in variants(cut_orders):
        q=g['q'];corner=block((0,0,cut[2]),(0.,0.,q))
        def shifted(block,primary):return (block[0]+primary,block[1]+primary)
        if term in ('full','forest_remainder'):
            full=shifted(block(cut,kin['variables']),exact_primary)
        if term in ('pair_projection','forest_remainder','pair_face'):
            pair_block=block((0,cut[1],cut[2]),(0.,v,q))
        if term in ('scale_projection','forest_remainder','scale_face'):
            scale_block=block((cut[0],0,cut[2]),(shape,0.,q))
            scale_block=shifted(scale_block,g['k'][1]*math.log(abs(1-shape)))
        if term in ('scale_projection','corner_projection','forest_remainder'):
            local=g['gamma']*math.log(scale)+g['a']*math.log(abs(shape))
        if term=='full':components=[full];signs=[1]
        elif term=='pair_projection':components=[shifted(pair_block,primary+g['a']*math.log(scale*abs(shape)))];signs=[1]
        elif term=='scale_projection':components=[shifted(scale_block,local)];signs=[1]
        elif term=='corner_projection':components=[shifted(corner,local)];signs=[1]
        elif term=='forest_remainder':
            components=[full,shifted(pair_block,primary+g['a']*math.log(scale*abs(shape))),
                        shifted(corner,local),shifted(scale_block,local)];signs=[1,-1,1,-1]
        elif term=='pair_face':
            radial=g['beta']*math.log(shape_upper)-np.log(g['beta'])
            components=[shifted(pair_block,primary+g['a']*math.log(scale)+radial),
                        shifted(corner,g['gamma']*math.log(scale)+radial)];signs=[1,-1]
        elif term=='scale_face':
            radial=g['a']*math.log(abs(shape))+g['lam']*math.log(scale)-np.log(g['lam'])
            components=[shifted(scale_block,radial),shifted(corner,radial)];signs=[1,-1]
        else:components=[shifted(corner,radial)];signs=[1]
        value,check=_reduce(bank,g,components,signs,geometry_scale,cut)
        values.append(value);checks.append(dict(prefix=label,orders=list(cut),**check))
    report=dict(term=term,passed=all(c['passed'] for c in checks),prefixes=checks,
        coefficient_recomputations=0,dozz_recomputations=0,target_blind=True,
        full_region_certified=False,full_amplitude_certified=False,
        qualification='Empirical signed-log arithmetic guards, not a rigorous bound; no sampling Jacobians included.')
    if not diagnostics and not report['passed']:raise FloatingPointError('unresolved triple-term arithmetic')
    return (np.asarray(values),report) if diagnostics else np.asarray(values)





from genus1_native_log_evaluation import block_logs, projected_logs, kinematics, reduce_retained

def evaluate_comb_collar(bank,atlas,*,term,pair,tau,scale=None,shape=None,phase=0.,shape_upper=None,
                         theta=None,radii=None,orders=None,diagnostics=False,reversed_bank=None):
    from genus1_three_point_collision import CollisionRadii,reconstruct_points,triple_shape_radius
    if term not in TERMS or type(diagnostics) is not bool:raise ValueError('explicit supported triple term required')
    cut_orders=tuple(require_orders(bank.orders if orders is None else orders))
    if any(a>b for a,b in zip(cut_orders,bank.orders)):raise ValueError('missing comb coefficients; no automatic extension')
    g=_geometry(bank,atlas,pair,tau);geometry_scale=1.; reduction_bank=bank
    if term!='corner_integrated':scale=_positive_radius(scale,'triple scale')
    if term in ('full','pair_projection','scale_projection','corner_projection','forest_remainder','scale_face'):
        shape=_shape(shape);geometry_scale=max(1.,2*math.pi*abs(g['tau']))/(scale*abs(shape))
    if term in ('full','pair_projection','forest_remainder','pair_face'):
        if not math.isfinite(float(phase)):raise ValueError('finite comb phase required')
        total=scale*cmath.exp(1j*float(phase));primary,v=_pair_log(g,total)
    if term in ('full','forest_remainder'):
        if not isinstance(radii,CollisionRadii) or not scale<radii.triple or not scale*abs(shape)<radii.pair:
            raise ValueError('full forest geometry must be strictly inside the resolved triple sector')
        w1,w2=reconstruct_points(g['pair'],shape*total,total)
        point=dict(w1=encoded(w1),w2=encoded(w2),tau=encoded(g['tau']),channel='comb_ope',pair=list(g['pair']))
        actual_pair = tuple(g['pair']); full_bank = bank
        if not atlas.sewing_chart(w1,w2,g['tau'],'comb_ope',actual_pair).admissible:
            actual_pair = actual_pair[::-1]
            atlas.sewing_chart(w1,w2,g['tau'],'comb_ope',actual_pair).require()
            if reversed_bank is None: raise ValueError('missing reversed-pair full coefficients')
            full_bank = reversed_bank
            if (type(full_bank) is not SharedNativeBank or full_bank.metadata['source_manifest_sha256']
                    != bank.metadata['source_manifest_sha256']):
                raise ValueError('unverified reversed-pair coefficient source')
            for field in ('momenta','log_weights','structure_logs'):
                if not np.array_equal(full_bank.arrays[field],bank.arrays[field]):
                    raise ValueError('reorientation must preserve the exact shared momentum density')
            from types import SimpleNamespace
            reduction_bank=SimpleNamespace(arrays={'coefficient_precision_bounds':np.maximum(
                bank.arrays['coefficient_precision_bounds'],full_bank.arrays['coefficient_precision_bounds'])})
        kin=kinematics(atlas,w1,w2,g['tau'],'comb_ope',actual_pair)
        if identity(kin) != full_bank.metadata['identity']:
            raise ValueError('wrong reversed-pair coefficient graph')
        if any(a>b for a,b in zip(cut_orders,full_bank.orders)):
            raise ValueError('missing reversed-pair coefficient prefix')
        exact_primary=(kin['constant']+(kin['common_log']-g['common']).real
            -np.einsum('ti,i->t',bank.arrays['momenta'][:,:2]**2,kin['decay'][:2]))
    if term in ('pair_face','scale_face','corner_integrated') and (np.any(g['beta']<=0) or np.any(g['lam']<=0)):
        raise ValueError('nonconvergent analytic radial integral')
    if term=='pair_face':
        shape_upper=float(shape_upper)
        if not math.isfinite(shape_upper) or not 0<shape_upper<=1:raise ValueError('invalid shape integration bound')
        geometry_scale=max(1.,2*math.pi*abs(g['tau']))/scale
    if term=='corner_integrated':
        if not isinstance(radii,CollisionRadii) or theta is None or not math.isfinite(float(theta)):
            raise ValueError('explicit corner radii and finite theta required')
        upper=triple_shape_radius(theta);split=radii.pair/radii.triple
        radial=np.logaddexp(g['lam']*math.log(radii.triple)-np.log(g['lam'])
            +g['beta']*math.log(split)-np.log(g['beta']),
            g['lam']*math.log(radii.pair)-np.log(g['lam'])+_log_interval(split,upper,g['beta']-g['lam']))
        radial+=math.log(2*math.pi)
    values=[];checks=[]
    for label,cut in variants(cut_orders):
        loop_log=2j*math.pi*g['tau']
        corner=block_logs(bank,(0,0,cut[2]),(complex(-math.inf),complex(-math.inf),loop_log))
        def shifted(block,primary):return (block[0]+primary,block[1]+primary)
        if term in ('full','forest_remainder'):
            full=shifted(block_logs(full_bank,cut,kin['logs']),exact_primary)
        if term in ('pair_projection','forest_remainder','pair_face'):
            pair_block=block_logs(bank,(0,cut[1],cut[2]),(complex(-math.inf),cmath.log(v),loop_log))
        if term in ('scale_projection','forest_remainder','scale_face'):
            scale_block=block_logs(bank,(cut[0],0,cut[2]),(cmath.log(shape),complex(-math.inf),loop_log))
            scale_block=shifted(scale_block,g['k'][1]*math.log(abs(1-shape)))
        if term in ('scale_projection','corner_projection','forest_remainder'):
            local=g['gamma']*math.log(scale)+g['a']*math.log(abs(shape))
        if term=='full':components=[full];signs=[1]
        elif term=='pair_projection':components=[shifted(pair_block,primary+g['a']*math.log(scale*abs(shape)))];signs=[1]
        elif term=='scale_projection':components=[shifted(scale_block,local)];signs=[1]
        elif term=='corner_projection':components=[shifted(corner,local)];signs=[1]
        elif term=='forest_remainder':
            components=[full,shifted(pair_block,primary+g['a']*math.log(scale*abs(shape))),
                        shifted(corner,local),shifted(scale_block,local)];signs=[1,-1,1,-1]
        elif term=='pair_face':
            radial=g['beta']*math.log(shape_upper)-np.log(g['beta'])
            components=[shifted(pair_block,primary+g['a']*math.log(scale)+radial),
                        shifted(corner,g['gamma']*math.log(scale)+radial)];signs=[1,-1]
        elif term=='scale_face':
            radial=g['a']*math.log(abs(shape))+g['lam']*math.log(scale)-np.log(g['lam'])
            components=[shifted(scale_block,radial),shifted(corner,radial)];signs=[1,-1]
        else:components=[shifted(corner,radial)];signs=[1]
        value,check=reduce_retained(reduction_bank,g,components,signs,geometry_scale,cut)
        values.append(value);checks.append(dict(prefix=label,orders=list(cut),**check))
    report=dict(term=term,passed=all(c['passed'] for c in checks),prefixes=checks,
        coefficient_recomputations=0,dozz_recomputations=0,target_blind=True,
        full_region_certified=False,full_amplitude_certified=False,
        qualification='Empirical signed-log arithmetic guards, not a rigorous bound; no sampling Jacobians included.')
    if not diagnostics and not report['passed']:raise FloatingPointError('unresolved triple-term arithmetic')
    return (np.asarray(values),report) if diagnostics else np.asarray(values)
