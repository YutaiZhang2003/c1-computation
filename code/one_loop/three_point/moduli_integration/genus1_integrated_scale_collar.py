"""Versioned, bank-owned optimization of the unchanged analytic collar method."""
from __future__ import annotations

# genus1_integrated_scale_collar
# Projection-only collar integrals, including analytic integration of tau_2.
# 
# Native CCY C[i,0,k] is retained, i<=10, k<=2. The eta oscillator is expanded
# in raw q through degree 10 (uniform product error bounded below). No elliptic
# variable, fitted CFT, or matrix-model target is used. This does not evaluate
# full-CFT or remainder strata. All external and momentum data are explicit.


import cmath
import math
from types import SimpleNamespace
import numpy as np
from scipy.special import erfcx
from genus1_comb_collar import _difference, _log_interval, _log_positive, _reduce as _reference_reduce
from genus1_rebuild_policy import classify_cft_point, encoded
from genus1_three_point_collision import koba_exponent, triple_shape_radius

SCOPE='comb-scale-and-corner-tau2-integrated-v1'
ORDERS=(10,0,2)
CUTS=(('full',(10,2)),('shape8',(8,2)),('shape6',(6,2)),('loop1',(10,1)),('loop0',(10,0)))
ETA_TERMS=((0,1),(1,-1),(2,-1),(5,1),(7,1))
QMAX=math.exp(-math.pi*math.sqrt(3))
# Euler pentagonal expansion: the first missing term has degree 12.
ETA_RELATIVE_BOUND=QMAX**12/(1-QMAX)/(1-QMAX/(1-QMAX))


def height_log_integral(a,lower,upper):
    """log integral T^-1/2 exp(-a*T) dT; upper may be infinity."""
    a=np.asarray(a,float)
    if np.any(a<=0) or not np.all(np.isfinite(a)) or not 0<lower<upper:
        raise ValueError('positive decay and ordered height interval required')
    def tail(t):
        return .5*(math.log(math.pi)-np.log(a))-a*t+np.log(erfcx(np.sqrt(a*t)))
    low=tail(lower)
    if math.isinf(upper):return low
    result,sign=_difference(low,tail(upper))
    if np.any(sign!=1):raise FloatingPointError('lost height integral difference')
    return result


def height_matrices(loop_nodes,tau1,region,cutoff=8.,*,loop_order=2):
    if not -.5<=tau1<=.5 or region not in ('bulk','tail') or cutoff<=1:
        raise ValueError('invalid modular region')
    lower=math.sqrt(1-tau1*tau1) if region=='bulk' else cutoff
    upper=cutoff if region=='bulk' else math.inf
    p=np.asarray(loop_nodes,float)
    if p.ndim!=1 or np.any(p<=0):raise ValueError('positive loop momenta required')
    if type(loop_order) is not int or loop_order < 0: raise ValueError('Nonnegative handle order required')
    n=len(p); maximum=2*(loop_order+7)
    logs=height_log_integral(4*math.pi*p[:,None]**2+2*math.pi*np.arange(maximum+1),lower,upper)
    scale=logs[:,0];j=np.exp(logs-scale[:,None])
    h=np.zeros((n,loop_order+1,loop_order+1),complex); bound=np.zeros((n,loop_order+1,loop_order+1))
    for left in range(loop_order+1):
        for right in range(loop_order+1):
            for a,ca in ETA_TERMS:
                for b,cb in ETA_TERMS:
                    v=j[:,left+right+a+b]
                    h[:,left,right]+=ca*cb*cmath.exp(2j*math.pi*(left-right+a-b)*tau1)*v
                    bound[:,left,right]+=v
    if not np.all(np.isfinite(h)) or not np.all(np.isfinite(scale)):
        raise FloatingPointError('invalid height kernel')
    return h,bound,scale


def validate_arrays(arrays,metadata):
    if metadata.get('evaluation_scope')!=SCOPE or metadata.get('orders')!=list(ORDERS):
        raise ValueError('explicit analytic projection scope required')
    count=math.prod(len(r['nodes']) for r in metadata['rules'])
    shapes=dict(momenta=(count,3),log_weights=(count,),structure_logs=(count,),
        projection_coefficients=(count,11,3),coefficient_precision_bounds=(count,))
    if set(arrays)!=set(shapes):raise ValueError('projection arrays only')
    for key,shape in shapes.items():
        if arrays[key].shape!=shape or not np.all(np.isfinite(arrays[key])):raise ValueError('incomplete '+key)
    if np.any(arrays['momenta']<=0) or np.any(arrays['coefficient_precision_bounds']<0) or np.any(arrays['coefficient_precision_bounds']>2.02e-11):
        raise ValueError('bad nodes or coefficient precision')
    for axis in range(3):
        expected=np.asarray(metadata['rules'][axis]['nodes'])
        if not np.array_equal(np.unique(arrays['momenta'][:,axis]),expected):raise ValueError('changed nodes')


def evaluate_reference(arrays,metadata,atlas,*,term,region,tau1,shape=None,theta=None,
             pair_radius=2*math.pi*.01,triple_radius=2*math.pi*.02,cutoff=8.):
    """Five genuine prefixes, with height integrated over bulk or entire tail."""
    if term not in ('triple_scale_face','triple_corner'):raise ValueError('unsupported analytic term')
    if not 0<pair_radius<=triple_radius/2 or triple_radius>=math.pi:raise ValueError('collar radii')
    if metadata['region']!=region:raise ValueError('wrong term-specific momentum rule')
    pair=tuple(metadata['pair']);spectator=next(i for i in range(3) if i not in pair)
    ext=tuple(atlas.external_weights[i] for i in (*pair,spectator))
    if [encoded(x) for x in ext]!=metadata['external_weights']:raise ValueError('frequency or graph changed')
    d=np.asarray(ext,complex).real;k=np.asarray([koba_exponent(atlas,i,j).real for i,j in
        (pair,(pair[0],spectator),(pair[1],spectator))])
    p=arrays['momenta'];aa=2*(1+p[:,0]**2)-2*(d[0]+d[1])+k[0];beta=aa+2
    if abs(2-2*sum(d)+sum(k)+4)>2e-14 or np.any(beta<=0):raise ValueError('unsupported radial powers')
    lam=2*p[:,1]**2
    if term=='triple_scale_face':
        shape=complex(shape)
        if not 0<abs(shape)<1 or shape.real>.5+1e-14:raise ValueError('invalid native shape disc')
        upper=min(triple_radius,pair_radius/abs(shape))
        radial=aa*math.log(abs(shape))+lam*math.log(upper)-np.log(lam)
    else:
        if theta is None or not math.isfinite(theta):raise ValueError('corner shape angle')
        split=pair_radius/triple_radius;upper=triple_shape_radius(theta)
        radial=np.logaddexp(lam*math.log(triple_radius)-np.log(lam)+beta*math.log(split)-np.log(beta),
            lam*math.log(pair_radius)-np.log(lam)+_log_interval(split,upper,beta-lam))+math.log(2*math.pi)
        shape=0j
    h,habsolute,hscale=height_matrices(np.asarray(metadata['rules'][2]['nodes']),tau1,region,cutoff)
    repeats=math.prod(len(r['nodes']) for r in metadata['rules'][:2])
    h=np.tile(h,(repeats,1,1));habsolute=np.tile(habsolute,(repeats,1,1));hscale=np.tile(hscale,repeats)
    g=dict(base=arrays['structure_logs']+arrays['log_weights']-3*math.log(math.pi)+hscale)
    results=[];checks=[]
    for label,(shape_order,loop_order) in CUTS:
        c=arrays['projection_coefficients'][:,:shape_order+1,:loop_order+1]
        scales=np.max(abs(c),axis=(1,2));scales=np.where(scales>0,scales,1.)
        normal=c/scales[:,None,None];powers=shape**np.arange(shape_order+1)
        b=np.einsum('nik,i->nk',normal,powers,optimize=False)
        ab=np.einsum('nik,i->nk',abs(normal),abs(powers),optimize=False)
        corner=normal[:,0,:];hc=h[:,:loop_order+1,:loop_order+1];ha=habsolute[:,:loop_order+1,:loop_order+1]
        def contracted(v,bound):
            value=np.einsum('nj,njl,nl->n',v,hc,v.conjugate(),optimize=False)
            absolute=np.einsum('nj,njl,nl->n',bound,ha,bound,optimize=False)
            if np.any(value.real<=0) or np.any(abs(value.imag)>1e-11*absolute):
                raise FloatingPointError('height-integrated block positivity/arithmetic')
            return 2*np.log(scales)+np.log(value.real),2*np.log(scales)+_log_positive(absolute)
        cb=contracted(corner,abs(corner))
        def shift(block,logfactor):return block[0]+logfactor,block[1]+logfactor
        if term=='triple_scale_face':
            sb=contracted(b,ab)
            components=[shift(sb,radial+k[1]*math.log(abs(1-shape))),shift(cb,radial)];signs=[1,-1]
        else:components=[shift(cb,radial)];signs=[1]
        value,check=_reference_reduce(SimpleNamespace(arrays=arrays),g,components,signs,
            1+abs(math.log(pair_radius)),(shape_order,0,loop_order))
        results.append(value);checks.append(dict(prefix=label,**check))
    return np.asarray(results),checks


def audit(values,checks):
    v=np.asarray(values)
    if v.shape!=(4,5) or not np.all(np.isfinite(v)) or np.any(abs(v)<np.finfo(float).tiny):
        raise FloatingPointError('four rules with five nonzero retained prefixes required')
    changes=[abs(v[1]-v[0])/abs(v[1]),abs(v[2]-v[1])/abs(v[2]),abs(v[3]-v[2])/abs(v[2])]
    full=v[2,0];axes=[[float(abs(full-v[2,1])/abs(full)),float(abs(v[2,1]-v[2,2])/abs(full))],
        [0.,0.],[float(abs(full-v[2,3])/abs(full)),float(abs(v[2,3]-v[2,4])/abs(full))]]
    arith=all(c['passed'] for cs in checks for c in cs)
    c=classify_cft_point(orders=ORDERS,axis_changes=axes,chart_admissible=True,coefficients_verified=True,
        momentum_converged=bool(arith and all(np.all(d<=.005) for d in changes)))
    return c,[float(max(x)) for x in changes]


# genus1_integrated_scale_collar_fast
# Versioned, bank-owned optimization of the unchanged analytic collar method.
# 
# No tau1 preintegration, new region split, elliptic variable, or relaxed gate.
# Each bank owns immutable input snapshots and bounded prefix caches. Independent
# momentum-axis sizes support the kernel-specific production quadratures.


import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
import numpy as np
from genus1_comb_collar import _difference, _signed_add, _log_interval
from genus1_pair_collar import _log_positive, PAIR_REMAINDER_MAX_CONDITION, PAIR_REMAINDER_NUMERICS_RTOL
from genus1_rebuild_policy import encoded
from genus1_three_point_collision import koba_exponent, triple_shape_radius

SCOPE = SCOPE
ORDERS = ORDERS
CUTS = CUTS
FLOAT = np.finfo(float)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def node_inputs(metadata, start=0, stop=None):
    rules = metadata['rules']; counts = tuple(len(r['nodes']) for r in rules)
    stop = math.prod(counts) if stop is None else stop
    if not 0 <= start < stop <= math.prod(counts):
        raise ValueError('invalid spectral slice')
    ix = np.array(np.unravel_index(np.arange(start, stop), counts)).T
    momenta = np.column_stack([np.asarray(r['nodes'])[ix[:, a]] for a, r in enumerate(rules)])
    w = [np.asarray(r['log_weights'])[ix[:, a]] for a, r in enumerate(rules)]
    return momenta, (w[0] + w[1]) + w[2]


def _immutable(array):
    a = np.ascontiguousarray(array)
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


class PreparedBank:
    """Cache lifetime is exactly this verified bank's lifetime; no global IDs."""
    @classmethod
    def load(cls, path, metadata, *, plan_sha256, receipt_sha256=None):
        path = Path(path); receipt_path = path.with_suffix('.json')
        receipt_sha256 = receipt_sha256 or receipt_path.with_suffix('.json.sha256').read_text().strip()
        if digest(receipt_path) != receipt_sha256:
            raise ValueError('bank receipt checksum')
        receipt = json.loads(receipt_path.read_text())
        if (receipt.get('plan_sha256') != plan_sha256 or receipt.get('evaluation_scope') != SCOPE
                or receipt['data_sha256'] != digest(path)):
            raise ValueError('incompatible bank provenance')
        with np.load(path, allow_pickle=False) as f:
            arrays = {k: f[k] for k in f}
        return cls(arrays, metadata, dict(plan_sha256=plan_sha256,
            receipt_sha256=receipt_sha256, data_sha256=receipt['data_sha256']))

    def __init__(self, arrays, metadata, provenance):
        # Metadata is itself verified against its parent plan by the loader.
        orders=metadata.get('orders',[])
        if (metadata.get('evaluation_scope') != SCOPE or len(orders)!=3 or orders[1]!=0
                or any(type(n) is not int or n<0 for n in orders)):
            raise ValueError('explicit retained projection orders required')
        if orders==list(ORDERS):
            self.cuts = CUTS
        else:
            # Minimal corner banks still need signed handle-prefix diagnostics.
            # An absent projected axis is never presented as an order check.
            from genus1_native_bank import variants
            self.cuts = tuple((label,(cut[0],cut[2])) for label,cut in variants(orders)
                              if label=='full' or orders[int(label[4])]>0)
        if any(not isinstance(s, str) or len(s) != 64 or any(c not in '0123456789abcdef' for c in s)
               for s in provenance.values()) or not provenance:
            raise ValueError('explicit SHA256 provenance required')
        self._metadata = json.loads(json.dumps(metadata, allow_nan=False))
        self.counts = tuple(len(r['nodes']) for r in metadata['rules'])
        ps, lw = node_inputs(metadata); count = len(ps)
        shapes = dict(momenta=(count, 3), log_weights=(count,), structure_logs=(count,),
            projection_coefficients=(count, orders[0]+1, orders[2]+1), coefficient_precision_bounds=(count,))
        if set(arrays) != set(shapes):
            raise ValueError('projection arrays only')
        for key, shape in shapes.items():
            if arrays[key].shape != shape or not np.all(np.isfinite(arrays[key])):
                raise ValueError('invalid ' + key)
        if (not np.array_equal(ps, arrays['momenta']) or not np.array_equal(lw, arrays['log_weights'])
                or np.any(ps <= 0) or np.any(arrays['coefficient_precision_bounds'] < 0)
                or np.any(arrays['coefficient_precision_bounds'] > 2.02e-11)):
            raise ValueError('nodes, ordering, weights or coefficient precision changed')
        self.arrays = MappingProxyType({k: _immutable(v) for k, v in arrays.items()})
        self.identity = MappingProxyType(dict(provenance, metadata_sha256=hashlib.sha256(
            json.dumps(metadata, sort_keys=True, allow_nan=False).encode()).hexdigest()))
        self._prefixes = {}
        for label, (s, l) in self.cuts:
            c = self.arrays['projection_coefficients'][:, :s+1, :l+1]
            scales = np.max(abs(c), axis=(1, 2)); scales = np.where(scales > 0, scales, 1.)
            normal = c / scales[:, None, None]
            self._prefixes[label] = tuple(_immutable(v) for v in (scales, normal, abs(normal)))
        self.cache_bytes = sum(v.nbytes for row in self._prefixes.values() for v in row)
        if self.cache_bytes > 256 * 2**20:
            raise ValueError('per-bank normalization cache exceeds 256MiB')


def _logsum(values):
    values = np.asarray(values, float)
    if np.any(np.isnan(values)) or np.any(np.isposinf(values)):
        raise FloatingPointError('nonfinite logarithmic diagnostic scale')
    finite = values[np.isfinite(values)]
    if not len(finite):
        return -math.inf
    largest = float(np.max(finite))
    return largest + math.log(math.fsum(np.exp(finite-largest).tolist()))


def _reduce(bank, g, components, signs, geometry_scale, cut, retain_spectral=False):
    # Identical signed-log subtraction and math.fsum, with Python-float input.
    logs = np.full(len(g['base']), -np.inf); direction = np.zeros(len(logs))
    for start in range(0, len(components), 2):
        if start+1 < len(components) and signs[start] == -signs[start+1]:
            part, sgn = _difference(components[start][0], components[start+1][0]); sgn *= signs[start]
        else:
            part = components[start][0]; sgn = np.where(np.isfinite(part), float(signs[start]), 0.)
            if start+1 < len(components):
                raise ValueError('unsupported signed component ordering')
        logs, direction = _signed_add(logs, direction, part, sgn)
    active = direction != 0; z = g['base'][active] + logs[active]
    if np.any(z.real > math.log(FLOAT.max)) or not np.all(np.isfinite(z)):
        raise FloatingPointError('unresolved forest overflow/nonfinite density')
    nodes = np.zeros(len(logs), complex); nodes[active] = direction[active]*np.exp(z)
    value = complex(math.fsum(nodes.real.tolist()), math.fsum(nodes.imag.tolist()))
    if not np.isfinite(value):
        raise FloatingPointError('nonfinite forest sum')
    parent = np.logaddexp.reduce(np.asarray([x[0] for x in components]), axis=0)
    bound = np.logaddexp.reduce(np.asarray([x[1] for x in components]), axis=0)
    budget = 128*FLOAT.eps*(math.prod(n+1 for n in cut)+geometry_scale+1+abs(g['base'])
        +np.where(np.isfinite(parent), abs(parent), 0))+4*bank.arrays['coefficient_precision_bounds']
    log_value = math.log(abs(value)) if abs(value) >= FLOAT.tiny else -math.inf
    condition_log = max(0., _logsum(g['base'].real+parent)-log_value)
    error = _logsum(g['base'].real+bound+np.log(budget))
    lost = g['base'].real+logs
    error = float(np.logaddexp(error, _logsum(lost[abs(nodes) < FLOAT.tiny])))
    relative_log = error-log_value
    def finite(log):
        return math.exp(log) if log <= math.log(FLOAT.max) else None
    reasons = []
    if not math.isfinite(log_value): reasons.append('zero_or_subnormal_term')
    if condition_log > math.log(PAIR_REMAINDER_MAX_CONDITION): reasons.append('cancellation_condition_limit')
    if relative_log > math.log(PAIR_REMAINDER_NUMERICS_RTOL): reasons.append('estimated_arithmetic_loss_limit')
    check = dict(passed=not reasons, reasons=reasons, parent_to_residual_condition=finite(condition_log),
        absolute_arithmetic_log_bound=error,
        estimated_relative_arithmetic_loss=finite(relative_log), subtraction_before_momentum_sum=True)
    if retain_spectral: check['spectral_terms'] = nodes
    return value, check


def evaluate(bank, atlas, *, term, region, tau1, shape=None, theta=None,
             pair_radius=2*math.pi*.01, triple_radius=2*math.pi*.02, cutoff=8.,retain_spectral=False,elliptic=None):
    """Unchanged raw-q collar expression and retained prefixes, with owned caches."""
    arrays = bank.arrays; metadata = bank._metadata
    if term not in ('triple_scale_face', 'triple_corner'): raise ValueError('unsupported analytic term')
    if not 0 < pair_radius <= triple_radius/2 or triple_radius >= math.pi: raise ValueError('collar radii')
    if metadata['region'] != region: raise ValueError('wrong term-specific momentum rule')
    pair = tuple(metadata['pair']); spectator = next(i for i in range(3) if i not in pair)
    ext = tuple(atlas.external_weights[i] for i in (*pair, spectator))
    if [encoded(x) for x in ext] != metadata['external_weights']: raise ValueError('frequency or graph changed')
    d = np.asarray(ext, complex).real
    k = np.asarray([koba_exponent(atlas, i, j).real for i, j in (pair, (pair[0], spectator), (pair[1], spectator))])
    p = arrays['momenta']; aa = 2*(1+p[:, 0]**2)-2*(d[0]+d[1])+k[0]; beta = aa+2
    if abs(2-2*sum(d)+sum(k)+4) > 2e-14 or np.any(beta <= 0): raise ValueError('unsupported radial powers')
    lam = 2*p[:, 1]**2
    if term == 'triple_scale_face':
        shape = complex(shape)
        if not 0 < abs(shape) < 1 or shape.real > .5+1e-14: raise ValueError('invalid native shape disc')
        upper = min(triple_radius, pair_radius/abs(shape))
        radial = aa*math.log(abs(shape))+lam*math.log(upper)-np.log(lam)
    else:
        if theta is None or not math.isfinite(theta): raise ValueError('corner shape angle')
        split = pair_radius/triple_radius; upper = triple_shape_radius(theta)
        radial = np.logaddexp(lam*math.log(triple_radius)-np.log(lam)+beta*math.log(split)-np.log(beta),
            lam*math.log(pair_radius)-np.log(lam)+_log_interval(split, upper, beta-lam))+math.log(2*math.pi)
        shape = 0j
    # Deliberately keep the existing height algorithm, including tau1 sampling.
    h, habsolute, hscale = height_matrices(np.asarray(metadata['rules'][2]['nodes']), tau1, region, cutoff,
                                                   loop_order=metadata['orders'][2])
    repeats = bank.counts[0]*bank.counts[1]
    h = np.tile(h, (repeats, 1, 1)); habsolute = np.tile(habsolute, (repeats, 1, 1)); hscale = np.tile(hscale, repeats)
    g = dict(base=arrays['structure_logs']+arrays['log_weights']-3*math.log(math.pi)+hscale)
    results = []; checks = []
    if elliptic is not None and term!='triple_scale_face':raise ValueError('elliptic shape block belongs only to the scale face')
    cuts=bank.cuts if elliptic is None else elliptic.cuts(metadata['orders'][2])
    for label, (shape_order, loop_order) in cuts:
        if elliptic is None:scales, normal, normal_absolute = bank._prefixes[label]
        else:
            # C[i,0,k]=F4_i*C[0,0,k] exactly. Preserve the same height
            # matrix, DOZZ measure and corner subtraction at every node.
            scales,normal,normal_absolute=bank._prefixes['full']
            normal=normal[:,:1,:loop_order+1];normal_absolute=normal_absolute[:,:1,:loop_order+1]
        if term == 'triple_scale_face' and elliptic is None:
            powers = shape**np.arange(shape_order+1)
            b = np.einsum('nik,i->nk', normal, powers, optimize=False)
            ab = np.einsum('nik,i->nk', normal_absolute, abs(powers), optimize=False)
        corner = normal[:, 0, :]; hc = h[:, :loop_order+1, :loop_order+1]; ha = habsolute[:, :loop_order+1, :loop_order+1]
        def contracted(v, bound):
            value = np.einsum('nj,njl,nl->n', v, hc, v.conjugate(), optimize=False)
            absolute = np.einsum('nj,njl,nl->n', bound, ha, bound, optimize=False)
            if np.any(value.real <= 0) or np.any(abs(value.imag) > 1e-11*absolute):
                raise FloatingPointError('height-integrated block positivity/arithmetic')
            return 2*np.log(scales)+np.log(value.real), 2*np.log(scales)+_log_positive(absolute)
        cb = contracted(corner, abs(corner))
        def shift(block, logfactor): return block[0]+logfactor, block[1]+logfactor
        if term == 'triple_scale_face':
            if elliptic is None:sb = contracted(b, ab)
            else:
                factor=elliptic.shape_logs(shape,shape_order)
                sb=(cb[0]+factor[0],cb[1]+factor[1])
            components = [shift(sb, radial+k[1]*math.log(abs(1-shape))), shift(cb, radial)]; signs = [1, -1]
        else:
            components = [shift(cb, radial)]; signs = [1]
        value, check = _reduce(bank, g, components, signs, 1+abs(math.log(pair_radius)), (shape_order, 0, loop_order),retain_spectral)
        results.append(value); checks.append(dict(prefix=label, **check))
    return np.asarray(results), checks
