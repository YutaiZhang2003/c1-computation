"""Explicit, restartable preparation of same-channel collision H coefficients."""
from __future__ import annotations

# genus1_collision_elliptic
# Same-channel elliptic completion of collision conformal blocks at c=25.
# 
# The scale face factorizes exactly at middle level zero. For an isolated pair,
# cut the longer loop and sew its descendants using Ward identities. The short
# sphere block is evaluated in its original channel, with the complete pillow
# prefactor. No fusion transformation or momentum interpolation is involved.
# 
# All coefficient construction belongs to explicit preparation. This module only
# contracts supplied H coefficients. Independent H and long-loop prefixes are
# numerical comparisons, never claimed bounds on the missing infinite series.

from functools import lru_cache
from math import factorial
import cmath
import math

import mpmath as mp
import numpy as np

from virasoro_descendant_algebra import act_virasoro_mode, integer_partitions


def _product(a, b, order):
    return [sum(a[j]*b[n-j] for j in range(n+1)) for n in range(order+1)]


def _inverse(a, order):
    out = [1/a[0]]
    for n in range(1, order+1):
        out.append(-sum(a[j]*out[n-j] for j in range(1, n+1))/a[0])
    return out


@lru_cache(maxsize=1)
def _small_nome_series():
    """Analytic germs, avoiding subtraction of 1-z at tiny z.

    S=sum a_n z^n=2K/pi, a_n=binom(2n,n)^2/16^n, and
    log(16q/z)=4 sum a_n(H_2n-H_n)z^n / S. At |z|<=1e-4,
    retaining 32 terms leaves a geometrical remainder far below float64.
    """
    with mp.workdps(70):
        n=32;S=[mp.binomial(2*j,j)**2/16**j for j in range(n+1)]
        inv=_inverse(S,n)
        numerator=[4*S[j]*(mp.harmonic(2*j)-mp.harmonic(j)) for j in range(n+1)]
        ratio=_product(numerator,inv,n)
        derivative=[(j+1)*S[j+1] if j<n else 0 for j in range(n+1)]
        logS=_product(derivative,inv,n)
        theta=[mp.mpf(0)]+[logS[j-1]/(2*j) for j in range(1,n+1)]
        one=[mp.mpf(0)]+[-mp.mpf(1)/j for j in range(1,n+1)]
        return tuple(tuple(row) for row in (ratio,one,theta))


@lru_cache(maxsize=256)
def geometry(z, order, log_z=None):
    """Taylor jets in x=log(z/z0); E=z*d/dz, so no small-z divisions.

    K obeys (1-z) E^2 K=z(E K+K/4), and
    E log(q)=pi^2/[4(1-z)K^2]. Geometry is computed once at guard precision.
    """
    z = complex(z)
    if log_z is not None:
        log_z=complex(log_z)
        if (not np.isfinite(log_z) or log_z.real>=0 or
                abs(z-cmath.exp(log_z))>max(2e-14*abs(z),8*np.nextafter(0.,1.))):
            raise ValueError('inconsistent collision sewing logarithm')
    if not (0 <= abs(z) < 1+2e-14) or (z==0 and log_z is None) or (z.imag == 0 and z.real >= 1):
        raise ValueError('collision elliptic argument must lie in the punctured principal disk')
    if type(order) is not int or not 0 <= order <= 8:
        raise ValueError('Euler jet order must be in 0..8')
    with mp.workdps(60):
        if abs(z)<=1e-4:
            lz=mp.mpc(log_z) if log_z is not None else mp.log(mp.mpc(z))
            zz=mp.exp(lz)
            powers=[zz**j for j in range(33)]
            jets=[tuple(mp.fsum(row[n]*powers[n]*n**j for n in range(1,33))/mp.factorial(j)
                        for j in range(order+1)) for row in _small_nome_series()]
            ratio,one,theta=jets
            q=list(ratio);q[0]+=lz-mp.log(16)
            if order:q[1]+=1
            return tuple(tuple(complex(v) for v in row) for row in (q,ratio,one,theta))
        zz = mp.mpc(z); n = max(1, order)
        Z = [zz/mp.factorial(j) for j in range(n+1)]
        one = [1-Z[0]]+[-v for v in Z[1:]]
        inv = _inverse(one, n)
        ratio = _product(Z, inv, n)
        K = [mp.ellipk(zz), mp.ellipe(zz)/(2*(1-zz))-mp.ellipk(zz)/2]
        for j in range(n-1):
            K.append(sum(ratio[i]*((j-i+1)*K[j-i+1]+K[j-i]/4)
                         for i in range(j+1))/((j+1)*(j+2)))
        kinv = _inverse(K, n)
        qprime = [mp.pi**2*v/4 for v in _product(inv, _product(kinv, kinv, n), n)]
        logq = [-mp.pi*mp.ellipk(1-zz)/K[0]]+[qprime[j-1]/j for j in range(1,n+1)]
        logqz = logq[:]; logqz[0] = mp.log(16)+logq[0]-mp.log(zz); logqz[1] -= 1
        logone = [mp.log(1-zz)]+[-ratio[j-1]/j for j in range(1,n+1)]
        kp = [(j+1)*K[j+1] if j<n else 0 for j in range(n+1)]
        kt = _product(kp, kinv, n)
        logtheta = [mp.log(2*K[0]/mp.pi)/2]+[kt[j-1]/(2*j) for j in range(1,n+1)]
        return tuple(tuple(complex(v) for v in row[:order+1]) for row in (logq,logqz,logone,logtheta))


def pillow_euler_jet(external, internal, z, coefficients, *, order=0, log_z=None):
    """Return log-prefactor and scaled Euler derivatives and absolute sums.

    Factoring out the constant prefactor prevents avoidable overflow at large
    Liouville momenta. Only polynomial arithmetic is used at spectral nodes.
    """
    external=np.asarray(external,complex); internal=np.asarray(internal,complex)
    H=np.asarray(coefficients,complex)
    if external.shape!=(len(internal),4) or H.ndim!=2 or H.shape[0]!=len(internal):
        raise ValueError('explicit matching four-point weights and coefficient rows required')
    lq,lqz,lone,lt=map(np.asarray,geometry(z,order,log_z))
    logpref=((internal-1)[:,None]*lqz+(1-external[:,1]-external[:,2])[:,None]*lone
             +(12-4*external.sum(axis=1))[:,None]*lt)
    value=np.zeros((len(internal),order+1),complex); absolute=np.zeros(value.shape,float)
    for level in range(H.shape[1]):
        exponent=logpref+level*lq
        jet=[np.ones(len(internal),complex)]; jet_bound=[np.ones(len(internal))]
        for n in range(1,order+1):
            jet.append(sum(j*exponent[:,j]*jet[n-j] for j in range(1,n+1))/n)
            jet_bound.append(sum(j*abs(exponent[:,j])*jet_bound[n-j] for j in range(1,n+1))/n)
        term=(H[:,level]*np.exp(level*lq[0]))[:,None]*np.asarray(jet).T
        value+=term
        absolute+=(abs(H[:,level])*math.exp(level*lq[0].real))[:,None]*np.asarray(jet_bound).T
    fact=np.asarray([factorial(n) for n in range(order+1)])
    value*=fact;absolute*=fact
    if not (np.all(np.isfinite(logpref)) and np.all(np.isfinite(value)) and np.all(np.isfinite(absolute))):
        raise FloatingPointError('nonfinite elliptic collision contraction')
    return logpref[:,0],value,absolute


@lru_cache(maxsize=256)
def _mode_affine(mode, desc):
    a=act_virasoro_mode(mode,{desc:1},h=0,c=25,vacuum=False)
    b=act_virasoro_mode(mode,{desc:1},h=1,c=25,vacuum=False)
    return tuple((key,a.get(key,0),b.get(key,0)-a.get(key,0)) for key in sorted(a.keys()|b.keys()))


class WardOperator(dict):
    """Polynomial plus the absolute arithmetic sums used to construct it."""
    def __init__(self,coefficients,absolute):
        super().__init__(coefficients);self.absolute=absolute


def pair_ward_operators(short, long, A, B, max_level):
    """Trace inverse Gram matrices against exact descendant Ward operators.

    Operators act on f in G=z^(short-long-B) f. Coefficients are polynomials
    in z and E. The factor z^level converts Q^level to q_long^level because
    Q=z*q_long. Long descendants through level four need at most E^8 f.
    """
    if type(max_level) is not int or not 0<=max_level<=4:
        raise ValueError('explicit long-loop level in 0..4 required')
    hs,hl,A,B=np.broadcast_arrays(short,long,A,B); alpha=hs-hl-B
    count=hs.size; one=np.ones(count,complex); cache={}

    def add(dest,source,scale=1):
        for key,v in source.items():
            dest[key]=dest.get(key,0)+scale*v
            dest.absolute[key]=dest.absolute.get(key,0)+abs(scale)*source.absolute[key]

    def D(op,mode,homogeneity):
        out=WardOperator({},{})
        for (power,derivative),v in op.items():
            for key,c in (((power,derivative),homogeneity+(mode+1)*A-alpha-power),
                          ((power,derivative+1),-one),
                          ((power+mode,derivative),(mode+1)*B+alpha+power),
                          ((power+mode,derivative+1),one)):
                out[key]=out.get(key,0)+v*c
                out.absolute[key]=out.absolute.get(key,0)+op.absolute[(power,derivative)]*abs(c)
        return out

    def C(left,right):
        key=(left,right)
        if key in cache:return cache[key]
        if not left and not right:out=WardOperator({(0,0):one},{(0,0):abs(one)})
        elif not left:
            mode,*rest=right;rest=tuple(rest)
            derivative=D(C((),rest),-mode,-sum(rest)-A-B)
            out=WardOperator({k:-v for k,v in derivative.items()},derivative.absolute.copy())
        else:
            rest,mode=left[:-1],left[-1]
            out=D(C(rest,right),mode,sum(rest)-sum(right)-A-B)
            for desc,a,b in _mode_affine(mode,right):add(out,C(rest,desc),a+b*hl)
        cache[key]=out
        return out

    def inner(left,right):
        state={right:one}
        for mode in left:
            new={}
            for desc,v in state.items():
                for key,a,b in _mode_affine(mode,desc):new[key]=new.get(key,0)+v*(a+b*hl)
            state=new
        return state.get((),np.zeros(count,complex))

    result=[]
    for level in range(max_level+1):
        basis=tuple(integer_partitions(level)); m=len(basis)
        gram=np.asarray([[inner(left,right) for right in basis] for left in basis]).transpose(2,0,1)
        inv=np.linalg.inv(gram)
        residual=np.max(abs(gram@inv-np.eye(m)),axis=(1,2))
        if np.any(residual>2e-10):raise FloatingPointError('long-loop Gram inverse is unresolved')
        op=WardOperator({},{})
        for i,left in enumerate(basis):
            for j,right in enumerate(basis):add(op,C(tuple(reversed(left)),right),inv[:,j,i])
        result.append(WardOperator({(power+level,derivative):v for (power,derivative),v in op.items()},
                      {(power+level,derivative):v for (power,derivative),v in op.absolute.items()}))
        if any(power<0 for power,_ in result[-1]):raise ArithmeticError('spurious negative sewing power')
    return tuple(result)


def squared_logs(logpref,value,absolute):
    if np.any(absolute<abs(value)*(1-2e-12)):
        raise FloatingPointError('elliptic absolute contraction lost its bound')
    with np.errstate(divide='ignore'):
        return 2*logpref.real+2*np.log(abs(value)),2*logpref.real+2*np.log(absolute)


def pair_logs(external,internal,z,q_long,H,operators,long_level,*,log_z=None):
    pref,jets,absolute=pillow_euler_jet(external,internal,z,H,order=2*long_level,log_z=log_z)
    result=np.zeros(len(internal),complex); bound=np.zeros(len(internal))
    for level,op in enumerate(operators[:long_level+1]):
        for (power,derivative),weight in op.items():
            scale=weight*z**power*q_long**level
            result+=scale*jets[:,derivative]
            bound+=op.absolute[(power,derivative)]*abs(z)**power*abs(q_long)**level*absolute[:,derivative]
    return squared_logs(pref,result,bound)


# genus1_collision_elliptic_bank
# Explicit, restartable preparation of same-channel collision H coefficients.

from collections import OrderedDict
import json
from pathlib import Path

import numpy as np

from genus1_boundary_cubature import MissingBoundaryInput, fingerprint
from genus1_native_bank import freeze_arrays, digest
FORMAT='genus1-collision-elliptic-v1'


def controls(value=None):
    value=dict(h_order=8,long_order=3) if value is None else dict(value)
    if set(value)!={'h_order','long_order'} or type(value['h_order']) is not int or not 4<=value['h_order']<=24:
        raise ValueError('collision H order in 4..24 required')
    if type(value['long_order']) is not int or not 2<=value['long_order']<=4:
        raise ValueError('collision long-loop order in 2..4 required')
    return value


def inputs(native,kind):
    h=1+native.arrays['momenta']**2
    d=np.asarray([complex(*v) for v in native.metadata['identity']['external_weights']])
    if kind=='triple_scale_face':
        external=np.column_stack((np.full(len(h),d[1]),np.full(len(h),d[0]),np.full(len(h),d[2]),h[:,1]))[None,:,:]
        internal=h[:,0][None,:]
    elif kind=='pair':
        external=np.asarray([np.column_stack((h[:,long],np.full(len(h),d[2]),h[:,0],h[:,long]))
                             for long in (2,1)])
        internal=np.asarray([h[:,1],h[:,2]])
    else:raise ValueError('only scale-face and isolated-pair collision blocks factorize here')
    return external,internal


def _load(path):
    path=Path(path);meta=json.loads(path.with_suffix('.json').read_text())
    if meta['data_sha256']!=digest(path):raise ValueError('changed collision coefficient data')
    with np.load(path,allow_pickle=False) as f:arrays={k:f[k] for k in f.files}
    return meta,arrays


def prepare(path,native,kind,settings,*,max_new_groups=32,checkpoint_directory=None):
    """Group by external weights; reuse one c=25 recursion over internal h.

    Scale-face coefficients are independent of the handle momentum. Deduplicate
    them before construction and storage. Pair orientation is frozen for both
    possible long cuts. A group checkpoint is committed only after its full
    checked recursion succeeds; the final receipt pins the native input.
    """
    from sphere_four_point_c25 import FourPointC25
    settings=controls(settings);path=Path(path)
    if type(max_new_groups) is not int or max_new_groups<1:raise ValueError('positive preparation group budget required')
    metadata=dict(format=FORMAT,kind=kind,controls=settings,
        source_native_data_sha256=native.metadata['data_sha256'],
        source_manifest_sha256=native.metadata['source_manifest_sha256'])
    if path.with_suffix('.json').exists():return CollisionBank.load(path,native,kind,settings)
    ext,internal=inputs(native,kind); rows=np.column_stack((ext.reshape(-1,4),internal.ravel()))
    unique,indices=np.unique(rows,axis=0,return_inverse=True)
    families,groups=np.unique(unique[:,:4],axis=0,return_inverse=True)
    H=np.empty((len(unique),settings['h_order']+1),complex)
    checkpoint=Path(checkpoint_directory) if checkpoint_directory else path.parent/'groups'
    new=0
    for i,family in enumerate(families):
        mask=groups==i;hs=unique[mask,4]
        key=fingerprint(dict(parent=metadata,external=[[v.real,v.imag] for v in family],
                             internal=[[v.real,v.imag] for v in hs]))
        target=checkpoint/(key+'.npz')
        if target.with_suffix('.json').exists():
            receipt,arrays=_load(target)
            if receipt.get('group_sha256')!=key or set(arrays)!={'H'} or arrays['H'].shape!=(len(hs),H.shape[1]):
                raise ValueError('changed collision recursion checkpoint')
            H[mask]=arrays['H']
        else:
            if new>=max_new_groups:raise MissingBoundaryInput('collision preparation group limit')
            engine=FourPointC25(tuple(family),order=settings['h_order'],dps=40)
            values=np.asarray(engine.coefficients_many(tuple(hs)),complex)
            if values.shape!=(len(hs),H.shape[1]) or not np.all(np.isfinite(values)):
                raise FloatingPointError('nonfinite checked collision H coefficients')
            freeze_arrays(target,dict(group_sha256=key),dict(H=values));H[mask]=values;new+=1
    arrays=dict(weights=unique,indices=indices.reshape(internal.shape),H=H)
    freeze_arrays(path,metadata,arrays)
    return CollisionBank.load(path,native,kind,settings)


class CollisionBank:
    @classmethod
    def load(cls,path,native,kind,settings,*,receipt_sha256=None):
        path=Path(path)
        if receipt_sha256 is not None and digest(path.with_suffix('.json'))!=receipt_sha256:
            raise ValueError('changed collision elliptic receipt')
        metadata,arrays=_load(path);settings=controls(settings)
        if (metadata.get('format')!=FORMAT or metadata.get('kind')!=kind or metadata.get('controls')!=settings or
                metadata.get('source_native_data_sha256')!=native.metadata['data_sha256'] or
                metadata.get('source_manifest_sha256')!=native.metadata['source_manifest_sha256']):
            raise ValueError('collision elliptic bank does not implement its frozen native grid')
        ext,internal=inputs(native,kind);idx=arrays.get('indices');weights=arrays.get('weights');H=arrays.get('H')
        if (set(arrays)!={'indices','weights','H'} or idx.shape!=internal.shape or idx.dtype.kind not in 'iu'
                or weights.ndim!=2 or weights.shape[1]!=5 or H.shape!=(len(weights),settings['h_order']+1)
                or np.any(idx<0) or np.any(idx>=len(weights)) or not np.all(np.isfinite(weights))
                or not np.all(np.isfinite(H)) or not np.array_equal(weights[idx,:4],ext)
                or not np.array_equal(weights[idx,4],internal) or not np.array_equal(H[:,0],np.ones(len(H)))):
            raise ValueError('invalid collision elliptic arrays or weights')
        result=cls();result.metadata=metadata;result.arrays=arrays;result.operators=OrderedDict()
        for value in arrays.values():value.flags.writeable=False
        return result

    def cuts(self,handle_order=None):
        h=self.metadata['controls']['h_order'];long=self.metadata['controls']['long_order'] if handle_order is None else handle_order
        orders=tuple(dict.fromkeys((h-2,max(2,h-4))))
        return (('full',(h,long)),*((f'H{n}',(n,long)) for n in orders),
                ('long'+str(long-1),(h,long-1)),('long'+str(long-2),(h,long-2)))

    def shape_logs(self,z,h_order):
        a=self.arrays;w=a['weights'];pref,jet,bound=pillow_euler_jet(w[:,:4],w[:,4],z,a['H'][:,:h_order+1])
        v,b=squared_logs(pref,jet[:,0],bound[:,0]);idx=a['indices'][0]
        return v[idx],b[idx]

    def pair_logs(self,logs,h_order,long_order):
        # Native axes (1,2); pick the smaller nome for the finite long series.
        orientation=0 if logs[0].real>=logs[1].real else 1
        z=np.exp(logs[orientation]);qlong=np.exp(logs[1-orientation])
        a=self.arrays;idx=a['indices'][orientation];w=a['weights'][idx];H=a['H'][idx,:h_order+1]
        if orientation not in self.operators:
            self.operators[orientation]=pair_ward_operators(w[:,4],w[:,0],w[:,2],w[:,1],self.metadata['controls']['long_order'])
        return pair_logs(w[:,:4],w[:,4],z,qlong,H,self.operators[orientation],long_order,log_z=logs[orientation])
