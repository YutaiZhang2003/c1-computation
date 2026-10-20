"""Verified native and elliptic five-point bank adapters for boundary cubature."""
from __future__ import annotations

# genus1_boundary_orders
# Explicit tail truncation, independent of the maximum stored bank order.
# 
# H4 is the production value. H2 is an inexpensive integrated prefix diagnostic;
# it is not an error bound. H6/H8 comparisons are explicitly requested studies.


DEFAULT_ELLIPTIC_ORDERS = (2, 4)


def elliptic_orders(orders=DEFAULT_ELLIPTIC_ORDERS):
    orders = tuple(orders)
    if (len(orders) not in (1, 2) or list(orders) != sorted(set(orders)) or
            any(type(n) is not int or not 0 <= n <= 24 for n in orders) or orders[-1] < 2):
        raise ValueError('one production H order >=2, optionally preceded by one lower prefix, required')
    return orders


# genus1_boundary_banks
# Verified native and elliptic five-point bank adapters for boundary cubature.
# 
# The native adapter is a matched finite-parent diagnostic. Elliptic banks use
# fixed physical weights and the b-regulator extrapolation. Neither adapter
# silently constructs a missing grid or certifies spectral convergence.



from genus1_source_paths import ROOT as _SOURCE_ROOT, source_path as _source_path

from dataclasses import dataclass
import cmath
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import numpy as np

from genus1_boundary_cubature import Value,MissingBoundaryInput,fingerprint
import genus1_analytic_boundary_layer as boundary
from genus1_moduli_sampling import TAU
from genus1_native_bank import SharedNativeBank, freeze_arrays
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def encoded(z):return [complex(z).real,complex(z).imag]


def load_native(path):
    path=Path(path);receipt=path.with_suffix('.json');meta=json.loads(receipt.read_text())
    return SharedNativeBank.load(path,expected_source_sha256=meta['source_manifest_sha256'],
                                 expected_receipt_sha256=sha(receipt))


def check_native(atlas,bank):
    identity=bank.metadata['identity']
    if identity['channel']!='necklace':raise ValueError('necklace bank required')
    for key,expected in (('external_weights',atlas.external_weights),('external_momenta',atlas.external_momenta)):
        if not np.allclose([complex(*z) for z in identity[key]],expected,rtol=0,atol=2e-13):
            raise ValueError('bank external states disagree')


def integrated_value(short,nodes,policy,angle_order,*,prefix=None,regulator=None):
    high=boundary.integrate_primary(short,nodes,policy=policy,angle_order=2*angle_order)['value']
    low=boundary.integrate_primary(short,nodes,policy=policy,angle_order=angle_order)['value']
    block_value=None if prefix is None else boundary.integrate_primary(
        short,prefix,policy=policy,angle_order=2*angle_order)['value']
    block=None if block_value is None else abs(high-block_value)
    return Value(high,dict(block=block,momentum=None,arithmetic=None,endpoint=None,
                           regulator=regulator,chart=None,angle=abs(high-low)),
                 {} if block_value is None else {'block_prefix':block_value})


class NativePrimary:
    """One immutable native grid, shared by the parent and its own cut primary."""
    def __init__(self,atlas,bank,*,maximum_short_nome=.8,orders=None):
        check_native(atlas,bank)
        if not 0<maximum_short_nome<1:raise ValueError('explicit native convergence window required')
        self.atlas=atlas;self.bank=bank;self.maximum_short_nome=maximum_short_nome
        self.orders=tuple(bank.validate_orders(bank.orders if orders is None else orders))
        if not bank.covers(self.orders):raise MissingBoundaryInput('native primary prefix is not stored')
        self.identity=dict(method='matched-native-finite-parent',data=bank.metadata['data_sha256'],
                           orders=list(self.orders),maximum_short_nome=maximum_short_nome)

    def slices(self,short):
        g=boundary.necklace_slice_geometry(self.atlas,short);axis=g['cut_axis']
        if max(math.exp(z.real) for z in g['short_logs'])>self.maximum_short_nome:
            raise MissingBoundaryInput('short geometry exceeds the declared native-bank window')
        arrays=self.bank.arrays;orders=list(self.orders);orders[axis]=0
        def one(cuts):
            c=arrays['coefficients'][(slice(None),*[slice(n+1) for n in cuts])]
            logs=list(g['short_logs']);full_logs=[0j]*3
            for k,z in enumerate(logs,1):full_logs[(axis+k)%3]=z
            powers=[np.exp(z*np.arange(n+1)) for z,n in zip(full_logs,cuts)]
            block=np.einsum('nijk,i,j,k->n',c,*powers,optimize=False)
            return boundary.primary_from_blocks(self.atlas,short,momenta=arrays['momenta'],
                log_weights=arrays['log_weights'],structure_logs=arrays['structure_logs'],blocks=block)
        lower=[max(0,n-1) for n in orders]
        return one(orders),one(lower)

    def __call__(self,short,policy,angle_order):
        nodes,prefix=self.slices(short)
        return integrated_value(short,nodes,policy,angle_order,prefix=prefix)

    def remainder(self,point):
        from genus1_channel_evaluation import full_values
        nodes,prefix=self.slices(point.short)
        positions=point.torus_positions()
        values,report=full_values(self.bank,self.atlas,
            dict(w1=positions[0],w2=positions[1],tau=point.tau,channel='necklace',pair=None),self.orders)
        if not report['passed']:raise FloatingPointError('unsafe native parent')
        leading=nodes.point(point.length)['value']
        increment=max(abs(values[0]-v) for v in values[1:])
        increment+=abs(leading-prefix.point(point.length)['value'])
        return Value(values[0]-leading,dict(block=increment,momentum=None,arithmetic=None,endpoint=None))


class NativeHeightCollars:
    """Reuse comb coefficient slices for deterministic scale/corner integrals.

    Existing spectral weights are preserved to roundoff and explicitly remain
    unvalidated for the integrated endpoint. This adapter does not claim that
    the source pointwise momentum rule is suitable for the complete integral.
    """
    def __init__(self,atlas,banks):
        from genus1_integrated_scale_collar import PreparedBank, node_inputs, SCOPE
        self.atlas=atlas;self.banks=dict(banks);self.prepared={}
        self.identity=dict(method='native-comb-height-slice-v1',banks={str(g):b.metadata['data_sha256'] for g,b in banks.items()},
                           integrated_spectral_convergence_assessed=False)
        for graph,b in self.banks.items():
            if graph not in (0,1) or not isinstance(b,SharedNativeBank) or b.metadata['identity']['channel']!='comb_ope':
                raise ValueError('verified comb banks for graphs zero and one required')
            pair=((0,1),(0,2))[graph];ext=(*pair,next(i for i in range(3) if i not in pair))
            expected=[complex(*z) for z in b.metadata['identity']['external_weights']]
            if not np.allclose(expected,[atlas.external_weights[i] for i in ext],rtol=0,atol=2e-13):
                raise ValueError('height graph external ordering changed')
            rules=[dict(r,log_weights=np.log(r['weights']).tolist()) for r in b.metadata['quadrature']['rules']]
            meta=dict(evaluation_scope=SCOPE,rules=rules,pair=list(pair),
                      external_weights=list(map(encoded,expected)),source_sha256=b.metadata['source_manifest_sha256'])
            provenance=dict(data_sha256=b.metadata['data_sha256'],source_sha256=b.metadata['source_manifest_sha256'])
            ps,lw=node_inputs(meta)
            if not np.array_equal(ps,b.arrays['momenta']) or not np.allclose(lw,b.arrays['log_weights'],rtol=0,atol=1e-13):
                raise ValueError('height conversion would change the spectral measure')
            for region in ('bulk','tail'):
                rows=[]
                for ds,dl in ((0,0),(min(2,b.orders[0]),min(1,b.orders[2]))):
                    shape=b.orders[0]-ds;loop=b.orders[2]-dl
                    m=dict(meta,orders=[shape,0,loop],region=region)
                    a={k:b.arrays[k] for k in ('momenta','structure_logs','coefficient_precision_bounds')}
                    a.update(log_weights=lw,projection_coefficients=b.arrays['coefficients'][:,:shape+1,0,:loop+1])
                    rows.append(PreparedBank(a,m,provenance))
                self.prepared[graph,region]=rows

    def __call__(self,request):
        from genus1_integrated_scale_collar import evaluate_reference as evaluate
        if request.kind!='height_integrated_collar':raise MissingBoundaryInput('only integrated scale/corner banks supplied')
        args=dict(request.geometry);graph=args.pop('graph');key=(graph,args['region'])
        if key not in self.prepared:raise MissingBoundaryInput('height graph bank missing')
        values=[]
        for bank in self.prepared[key]:
            v,checks=evaluate(bank,self.atlas,**args)
            if not all(c['passed'] for c in checks):raise FloatingPointError('unsafe integrated collar arithmetic')
            values.append(v[0])
        return Value(values[0],dict(block=abs(values[0]-values[1]),momentum=None,arithmetic=None,endpoint=None))


def forward_nomes(p1,p2,*,tolerance=2e-15,max_terms=4096):
    """Forward sphere products and original-short-coordinate Jacobian."""
    if not 0<abs(p1)<1 or not 0<abs(p2)<1:raise MissingBoundaryInput('segment nome outside polydisc')
    q=p1*p2;z=16*q;t=4*p2*(1+p1)**2;a=1+0j;b1=2*p1/(1+p1);b2=1+0j
    for n in range(1,max_terms+1):
        even=q**(2*n);odd=q**(2*n-1)
        z*=((1+even)/(1+odd))**8;t*=((1+even)/(1+odd))**4
        derivative=(2*n)*even/(1+even)-(2*n-1)*odd/(1+odd)
        a+=8*derivative;b1+=4*derivative;b2+=4*derivative
        for j,k,sign in ((2*n+1,2*n,1),(2*n-1,2*n,1),(2*n,2*n-1,-1),(2*n-2,2*n-1,-1)):
            v=p1**j*p2**k;t*=(1+v)**(2*sign)
            b1+=2*sign*j*v/(1+v);b2+=2*sign*k*v/(1+v)
        if max(abs(q)**(2*n),abs(p1)**(2*n),abs(p2)**(2*n))<tolerance:
            derivative=np.array([[a-b1,a-b2],[b1,b2]])
            return z,t,derivative,float(abs(a*(b2-b1))**2)
    raise MissingBoundaryInput('elliptic forward products did not converge')


def _vendor():
    root=_SOURCE_ROOT
    source=root/'code/general/virasoro/vendor/genus0_elliptic_h_recursion/src'
    if str(source) not in sys.path:sys.path.insert(0,str(source))
    return source


class ReferenceContinuedCoordinates:
    """Explicit bounded continuation from positive-real x,y; no guessed patch.

    The continuation and forward checks are cached per physical short point.
    Uncovered branches raise MissingBoundaryInput for an alternate atlas patch.
    This is a reference resolver; forward-coordinate cells avoid inverse work.
    """
    def __init__(self,*,dps=45,phase_step=.2,maximum_nome=.95):
        if dps<30 or not 0<phase_step<=.5 or not 0<maximum_nome<1:
            raise ValueError('bounded elliptic coordinate controls required')
        self.dps=dps;self.phase_step=phase_step;self.maximum_nome=maximum_nome;self.cache={}
        self.identity=dict(method='continued-native-phases-v1',dps=dps,phase_step=phase_step,maximum_nome=maximum_nome)

    def __call__(self,logs):
        key=tuple(map(complex,logs))
        if key in self.cache:return self.cache[key]
        if len(key)!=2 or any(z.real>=0 or z.real<-600 for z in key):
            raise MissingBoundaryInput('short endpoint requires a separate elliptic/native patch')
        _vendor();import mpmath as mp
        from genus0_elliptic_h_recursion.geometry import (invert_aligned_coordinates,elliptic_nome,
            mobile_position_from_right_product,coordinates_from_segment_nomes,cross_ratio_from_nome)
        steps=max(1,math.ceil(max(abs(z.imag) for z in key)/self.phase_step))
        try:
            with mp.workdps(self.dps):
                def nome(z):
                    # K(1-z) loses z when the two remaining tubes are long.
                    # Invert the convergent lambda product at small z instead.
                    if abs(z)>=mp.mpf('1e-8'):return elliptic_nome(z)
                    q=z/16
                    for _ in range(8):
                        actual=cross_ratio_from_nome(q)
                        if abs(actual/z-1)<mp.mpf(10)**(-self.dps+10):return q
                        q*=z/actual
                    raise MissingBoundaryInput('small-z nome inverse failed its relative check')
                xx,yy=(mp.exp(z.real) for z in key)
                q=nome(xx*yy)
                seed=yy/(1+mp.sqrt(1-yy))**2
                try:
                    factor=mp.findroot(lambda r:mobile_position_from_right_product(seed*r,q)/yy-1,
                        (mp.mpf(1),mp.mpf('1.0001')),tol=mp.mpf(10)**(-self.dps+12),maxsteps=30)
                    right=seed*factor
                    if not 0<right<1 or not 0<q/right<1:raise ValueError('non-aligned inverse')
                except (ValueError,ZeroDivisionError):
                    inverse=invert_aligned_coordinates(xx*yy,(yy,),dps=self.dps)
                    right=inverse.segment_nomes[1]
                for j in range(steps+1):
                    x=mp.exp(mp.mpc(key[0].real,key[0].imag*j/steps))
                    y=mp.exp(mp.mpc(key[1].real,key[1].imag*j/steps));q=nome(x*y)
                    if j:
                        anchor=right
                        right=anchor*mp.findroot(lambda r:mobile_position_from_right_product(anchor*r,q)/y-1,
                            (mp.mpf(1),mp.mpf('1.0001')),tol=mp.mpf(10)**(-self.dps+12),maxsteps=30)
                    nomes=(q/right,right)
                    if max(abs(p) for p in nomes)>=self.maximum_nome:
                        raise MissingBoundaryInput('continued nomes exceed configured patch')
                    z,mobile=coordinates_from_segment_nomes(nomes)
                    if max(abs(z-x*y)/abs(x*y),abs(mobile[0]-y)/abs(y))>mp.mpf('1e-20'):
                        raise MissingBoundaryInput('elliptic continuation failed its forward check')
                result=tuple(map(complex,nomes))
        except (ValueError,ZeroDivisionError,OverflowError,AssertionError) as exc:
            raise MissingBoundaryInput('elliptic branch requires another patch: '+str(exc)) from exc
        self.cache[key]=result
        return result


def log_segment_products(logs, *, tolerance=2e-16, max_terms=4096):
    """Forward log(x), log(y) and their analytic log-nome Jacobian.

    Keeping the leading powers in logarithms also covers simultaneous long
    tubes where x*y underflows in double precision. Winding is inherited from
    the supplied log nomes; the factors 1+v stay on their principal branch
    throughout the open polydisc. The series tail decays in (p1*p2)**2.
    """
    l1,l2=map(complex,logs);p1,p2=cmath.exp(l1),cmath.exp(l2)
    if any(not math.isfinite(z.real) or not math.isfinite(z.imag) or z.real>=0 for z in (l1,l2)):
        raise MissingBoundaryInput('segment nome outside polydisc')
    q=p1*p2
    z=math.log(16)+l1+l2;t=math.log(4)+l2+2*cmath.log(1+p1)
    a=1+0j;b1=2*p1/(1+p1);b2=1+0j
    for n in range(1,max_terms+1):
        even=q**(2*n);odd=q**(2*n-1)
        term=cmath.log(1+even)-cmath.log(1+odd)
        z+=8*term;t+=4*term
        derivative=2*n*even/(1+even)-(2*n-1)*odd/(1+odd)
        a+=8*derivative;b1+=4*derivative;b2+=4*derivative
        largest=max(abs(even),abs(odd))
        for j,k,sign in ((2*n+1,2*n,1),(2*n-1,2*n,1),
                         (2*n,2*n-1,-1),(2*n-2,2*n-1,-1)):
            v=p1**j*p2**k;t+=2*sign*cmath.log(1+v)
            b1+=2*sign*j*v/(1+v);b2+=2*sign*k*v/(1+v)
            largest=max(largest,abs(v))
        if n*largest<tolerance:
            return np.asarray((z-t,t)),np.asarray(((a-b1,a-b2),(b1,b2)))
    raise MissingBoundaryInput('logarithmic elliptic products did not converge')


class ContinuedCoordinates(ReferenceContinuedCoordinates):
    """Damped log-nome continuation, followed by an independent MP check.

    Trial Newton steps stay in the polydisc and decrease the forward residual.
    Failed phase steps are bisected along the *same* phase homotopy, never
    replaced by an unrelated root. Arbitrary precision is used once at the
    endpoint instead of at every Newton trial along that homotopy.
    """
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.identity.update(method='damped-log-nome-continuation-v2',
                             endpoint_check='mp-relative-1e-20')

    @staticmethod
    def _solve(target,seed):
        current=np.asarray(seed,complex).copy()
        tolerance=2e-13
        for _ in range(40):
            actual,jacobian=log_segment_products(current)
            residual=actual-target;error=float(max(abs(residual)))
            if error<tolerance:return current
            try:delta=np.linalg.solve(jacobian,residual)
            except np.linalg.LinAlgError as exc:
                raise MissingBoundaryInput('singular elliptic coordinate Jacobian') from exc
            for k in range(24):
                trial=current-delta*2.**(-k)
                if np.any(trial.real>=-1e-13) or np.any(trial.real<-700):continue
                value,_=log_segment_products(trial)
                if max(abs(value-target))<error:
                    current=trial;break
            else:raise MissingBoundaryInput('elliptic Newton step has no admissible descent')
        raise MissingBoundaryInput('elliptic Newton iteration limit')

    def __call__(self,logs):
        key=tuple(map(complex,logs))
        if key in self.cache:return self.cache[key]
        if len(key)!=2 or any(z.real>=0 or z.real<-600 for z in key):
            raise MissingBoundaryInput('short endpoint requires a separate elliptic/native patch')
        target=np.asarray(key,complex);aligned=target.real.astype(complex)
        # Small-z seed, evaluated without forming the possibly underflowed xy.
        right=aligned[1]-2*math.log1p(math.sqrt(-math.expm1(aligned[1].real)))
        seed=(aligned.sum()-math.log(16)-right,right)
        current=self._solve(aligned,seed)
        maximum_step=min(1.,self.phase_step/max(max(abs(target.imag)),self.phase_step))
        step=maximum_step;progress=0.;attempts=0
        while progress<1:
            attempts+=1
            if attempts>1024:raise MissingBoundaryInput('elliptic continuation exceeded its bounded work budget')
            next_progress=min(1.,progress+step)
            try:new=self._solve(aligned+1j*target.imag*next_progress,current)
            except MissingBoundaryInput:
                step/=2
                if step<1e-7:raise MissingBoundaryInput('elliptic phase continuation exhausted subdivisions')
                continue
            # Large log-nome jumps are resolved by smaller homotopy steps.
            if max(abs(new-current))>.75:
                step/=2
                if step<1e-7:raise MissingBoundaryInput('elliptic continuation cannot resolve a branch jump')
                continue
            current=new;progress=next_progress;step=min(maximum_step,1.5*step)
        nomes=tuple(map(cmath.exp,current))
        if max(abs(p) for p in nomes)>=self.maximum_nome:
            raise MissingBoundaryInput('continued nomes exceed configured patch')
        _vendor();import mpmath as mp
        from genus0_elliptic_h_recursion.geometry import coordinates_from_segment_nomes
        with mp.workdps(self.dps):
            expected_z=mp.exp(mp.mpc(key[0])+mp.mpc(key[1]));expected_y=mp.exp(mp.mpc(key[1]))
            left,right=map(mp.mpc,nomes)
            _,jacobian=log_segment_products(current)
            a,b,c,d=(mp.mpc(v) for v in jacobian.flat);det=a*d-b*c
            original=(left,right)
            try:
                # At this already converged endpoint, the analytic double
                # Jacobian is ample for MP residual correction. This avoids
                # repeated numerical differentiation of expensive MP products.
                for _ in range(6):
                    z,mobile=coordinates_from_segment_nomes((left,right))
                    rz,ry=z/expected_z,mobile[0]/expected_y
                    if max(abs(rz-1),abs(ry-1))<mp.mpf('1e-25'):break
                    rx=mp.log(rz/ry);ry=mp.log(ry)
                    left*=mp.exp(-(d*rx-b*ry)/det)
                    right*=mp.exp(-(-c*rx+a*ry)/det)
                    if max(abs(left/original[0]-1),abs(right/original[1]-1))>mp.mpf('1e-9'):
                        raise MissingBoundaryInput('MP refinement changed the continued branch')
                else:raise MissingBoundaryInput('elliptic continuation failed its MP forward check')
                result=complex(left),complex(right)
            except (ValueError,ZeroDivisionError,OverflowError) as exc:
                raise MissingBoundaryInput('elliptic endpoint refinement failed: '+str(exc)) from exc
        self.cache[key]=result
        return result


class EllipticGridBank:
    """One cut axis and complete declared spectral arrays, independent of moduli."""
    FORMAT='genus1-five-point-spectral-bank-v1'

    def __init__(self,metadata,arrays):
        self.metadata=json.loads(json.dumps(metadata));self.arrays={k:np.asarray(v).copy() for k,v in arrays.items()}
        a=self.arrays;n=len(a['momenta']);levels=a['levels']
        if metadata['format']!=self.FORMAT or metadata['cut_axis'] not in (0,1,2):
            raise ValueError('explicit elliptic cut convention required')
        shapes={'momenta':(n,3),'log_weights':(n,),'structure_logs':(n,),
                'levels':(len(levels),2),'coefficients':(n,len(levels))}
        if set(a)!=set(shapes) or any(a[k].shape!=s or not np.all(np.isfinite(a[k])) for k,s in shapes.items()):
            raise ValueError('invalid elliptic spectral arrays')
        if np.any(a['momenta']<=0) or np.any(levels<0) or not np.array_equal(levels,levels.astype(int)):
            raise ValueError('positive momenta and integer coefficient levels required')
        expected={(i,j) for i in range(metadata['order']+1) for j in range(metadata['order']+1-i)}
        if set(map(tuple,levels))!=expected or len(levels)!=len(expected):raise ValueError('incomplete elliptic coefficient triangle')
        if not np.allclose(a['coefficients'][:,list(map(tuple,levels)).index((0,0))],1.,rtol=0,atol=1e-12):
            raise ValueError('elliptic primary normalization changed')
        for v in a.values():v.flags.writeable=False

    @classmethod
    def load(cls,path,*,receipt_sha256):
        path=Path(path);receipt=path.with_suffix('.json')
        if sha(receipt)!=receipt_sha256:raise ValueError('elliptic bank receipt changed')
        m=json.loads(receipt.read_text())
        if sha(path)!=m['data_sha256']:raise ValueError('elliptic bank payload changed')
        with np.load(path,allow_pickle=False) as f:a={k:f[k] for k in f}
        return cls(m,a)

    @classmethod
    def from_single_node(cls,path):
        """Import the already validated H10 diagnostic; spectral scope stays partial."""
        path=Path(path);m=json.loads(path.with_suffix('.json').read_text());node=m['source_node']
        if not m['passed'] or sha(path)!=m['data_sha256']:raise ValueError('unverified legacy H bank')
        source=Path(node['path'])
        if sha(source)!=node['data_sha256'] or sha(source.with_suffix('.json'))!=node['receipt_sha256']:
            raise ValueError('legacy spectral source changed')
        with np.load(source,allow_pickle=False) as f:
            a={k:f[k][node['row']:node['row']+1].copy() for k in ('momenta','log_weights','structure_logs')}
        with np.load(path,allow_pickle=False) as f:
            a.update(levels=f['levels'].copy(),coefficients=f['coefficients'][None,:].copy())
        return cls(dict(format=cls.FORMAT,cut_axis=0,order=m['total_degree'],
            native_external_weights=node['external_weights'],spectral_complete=False,
            regulator_checked=True,epsilon=m['epsilon'],dps=m['dps'],data_sha256=m['data_sha256'],
            source_receipt_sha256=sha(path.with_suffix('.json')),spectral_data_sha256=node['data_sha256']),a)

    def blocks(self,atlas,short,nomes,*,order=None):
        g=boundary.necklace_slice_geometry(atlas,short);axis=g['cut_axis']
        if axis!=self.metadata['cut_axis']:raise MissingBoundaryInput('elliptic bank for this cut axis is absent')
        expected=[complex(*z) for z in self.metadata['native_external_weights']]
        if not np.allclose(expected,atlas.external_weights,rtol=0,atol=2e-13) or max(abs(z.imag) for z in expected)>1e-13:
            raise ValueError('elliptic bank requires the saved real conformal weights')
        x,y=(cmath.exp(z) for z in g['short_logs']);p1,p2=nomes
        z,t,_,_=forward_nomes(p1,p2)
        if max(abs(z-x*y)/max(abs(x*y),1e-300),abs(t-y)/max(abs(y),1e-300))>3e-11:
            raise MissingBoundaryInput('nome patch does not map to the requested short geometry')
        q=p1*p2;theta3=1+0j
        for n in range(1,4096):
            term=2*q**(n*n);theta3+=term
            if abs(term)<2e-16:break
        else:raise MissingBoundaryInput('theta product did not converge')
        features=np.array([cmath.log(v) for v in (theta3,z,1-z,t*(1-t)*(t-z),4*p1,4*p2,t)])
        h=1+self.arrays['momenta']**2;d=[expected[(axis+k)%3] for k in range(3)]
        d0=h[:,axis];dz,mu,d1=d;di=d0;a1=h[:,(axis+1)%3];a2=h[:,(axis+2)%3]
        weights=np.column_stack((12-4*(d0+dz+d1+di)-2*mu,1-a1,
            np.full(len(h),1-dz-d1),np.full(len(h),-mu/2),a1-1,a2-1,-(a2-mu-a1)))
        cutoff=self.metadata['order'] if order is None else order
        if not 0<=cutoff<=self.metadata['order']:raise ValueError('unavailable elliptic prefix')
        levels=self.arrays['levels'];mask=levels.sum(axis=1)<=cutoff
        powers=p1**levels[mask,0]*p2**levels[mask,1]
        block=np.exp(np.einsum('ni,i->n',weights,features,optimize=False))*np.einsum(
            'ni,i->n',self.arrays['coefficients'][:,mask],powers,optimize=False)
        if not np.all(np.isfinite(block)):raise FloatingPointError('nonfinite elliptic block')
        return block


class EllipticPrimary:
    def __init__(self,atlas,banks,coordinates,*,allow_partial_spectrum=False,orders=DEFAULT_ELLIPTIC_ORDERS):
        if not callable(coordinates):raise ValueError('explicit checked coordinate resolver required')
        self.atlas=atlas;self.banks=dict(banks);self.coordinates=coordinates
        self.allow_partial_spectrum=bool(allow_partial_spectrum)
        self.orders=elliptic_orders(orders)
        if any(self.orders[-1]>bank.metadata['order'] for bank in self.banks.values()):
            raise MissingBoundaryInput('requested elliptic order is absent from bank')
        self.identity=dict(method='elliptic-h-c25',banks={str(k):v.metadata for k,v in self.banks.items()},
                           orders=list(self.orders),
                           coordinates=getattr(coordinates,'identity',None),
                           partial_spectrum_diagnostic=self.allow_partial_spectrum)
        if self.identity['coordinates'] is None:raise ValueError('coordinate patch identity required')

    def __call__(self,short,policy,angle_order):
        g=boundary.necklace_slice_geometry(self.atlas,short);axis=g['cut_axis']
        if axis not in self.banks:raise MissingBoundaryInput(f'missing elliptic cut-axis {axis} bank')
        bank=self.banks[axis];a=bank.arrays
        if not bank.metadata['spectral_complete'] and not self.allow_partial_spectrum:
            raise MissingBoundaryInput('partial spectral bank is diagnostic only')
        nomes=self.coordinates(g['short_logs'])
        def one(order):
            return boundary.primary_from_blocks(self.atlas,short,momenta=a['momenta'],log_weights=a['log_weights'],
                structure_logs=a['structure_logs'],blocks=bank.blocks(self.atlas,short,nomes,order=order))
        high=one(self.orders[-1]);low=one(self.orders[0]) if len(self.orders)>1 else None
        return integrated_value(short,high,policy,angle_order,prefix=low)


def regular_h_coefficients(*,central_charge,external_weights,internal_weights,order,dps,
                          workspace=None,on_recursion_call=None,direction=(1,-1)):
    """Resolve removable fixed-difference resonances in the positive spectrum.

    Positive real internal weights at c>25 are away from physical Kac poles.
    Coincident recursion poles can nevertheless occur inside the fixed-
    difference algorithm. Take a symmetric internal-weight limit at three
    spacings and compare two fourth-order Richardson estimates. Never apply
    this continuation at a physical pole or outside this real-weight domain.
    """
    import mpmath as mp
    from genus0_elliptic_h_recursion import compute_h_recursion
    def table(weights):
        if on_recursion_call is not None:on_recursion_call()
        return compute_h_recursion(central_charge=central_charge,external_weights=external_weights,
            internal_weights=weights,order=order,dps=dps,workspace=workspace).coefficients
    c=complex(central_charge)
    positive=(not c.imag and c.real>25 and len(internal_weights)==2 and
              all(not complex(h).imag and complex(h).real>0 for h in internal_weights))
    with mp.workdps(dps+20):
        delta=(mp.mpmathify(internal_weights[1])-mp.mpmathify(internal_weights[0])) if positive else None
        resonant=positive and delta.imag==0 and abs(delta)<=order and delta.real==mp.nint(delta.real)
    if not resonant:
        try:return table(internal_weights),0.
        except ZeroDivisionError:
            if not positive:raise
    with mp.workdps(dps+20):
        averages=[]
        for step in ('0.0001','0.00005','0.000025'):
            step=mp.mpf(step)
            tables=[table(tuple(mp.mpmathify(h)+sign*step*v for h,v in zip(internal_weights,direction)))
                    for sign in (-1,1)]
            averages.append({k:(tables[0][k]+tables[1][k])/2 for k in tables[0]})
        coarse={k:(4*averages[1][k]-averages[0][k])/3 for k in averages[0]}
        fine={k:(4*averages[2][k]-averages[1][k])/3 for k in averages[0]}
        change=float(max(abs(fine[k]-coarse[k])/max(1,abs(fine[k]),abs(coarse[k])) for k in fine))
        if not math.isfinite(change) or change>2e-11:
            raise MissingBoundaryInput('internal-weight resonance limit did not converge')
        return fine,change


def prepare_elliptic_grid(path,native,*,cut_axis,order,epsilon='0.0005',dps=70,
                          max_new_nodes,checkpoint_directory,node_indices=None,on_recursion_call=None):
    """Explicit, resumable bank construction; no changes to source/native banks.

    Each node is checked at epsilon/2 and at dps+20 before publication. Work
    is capped by max_new_nodes; completed nodes are retained for the next call.
    No moduli sampling occurs here. Existing final banks are never overwritten.
    """
    path=Path(path)
    if path.with_suffix('.json').exists():raise FileExistsError('preserve existing elliptic bank')
    if cut_axis not in (0,1,2) or type(order) is not int or not 2<=order<=24 or type(max_new_nodes) is not int or max_new_nodes<1 or dps<40:
        raise ValueError('explicit bounded elliptic preparation controls required')
    vendor=_vendor();import mpmath as mp
    from genus0_elliptic_h_recursion import compute_h_recursion
    from genus0_elliptic_h_recursion.recursion import RecursionWorkspace
    workspace=RecursionWorkspace()
    sources={str(p.relative_to(vendor)):sha(p) for p in sorted(vendor.rglob('*.py'))}
    source_id=fingerprint(sources);arrays=native.arrays
    if not isinstance(native,SharedNativeBank):raise ValueError('verified native spectral source required')
    selected=np.arange(len(arrays['momenta'])) if node_indices is None else np.asarray(node_indices)
    if selected.ndim!=1 or not len(selected) or not np.issubdtype(selected.dtype,np.integer) or len(set(selected.tolist()))!=len(selected) or np.any((selected<0)|(selected>=len(arrays['momenta']))):
        raise ValueError('explicit distinct spectral node indices required')
    d=tuple(complex(*v) for v in native.metadata['identity']['external_weights'])
    if native.metadata['identity']['channel']!='necklace' or max(abs(z.imag) for z in d)>1e-13:
        raise ValueError('real-weight necklace source required')
    e=float(epsilon)
    if not 0<e<.02:raise ValueError('small positive fixed-weight b regulator required')
    folder=Path(checkpoint_directory);folder.mkdir(parents=True,exist_ok=True)
    coefficients=[];audits=[];new=0;levels=np.array([(i,j) for i in range(order+1) for j in range(order+1-i)],int)
    for p in arrays['momenta'][selected]:
        h=tuple(1+p*p);external=(h[cut_axis],*(d[(cut_axis+k)%3] for k in range(3)),h[cut_axis])
        internal=tuple(h[(cut_axis+k)%3] for k in (1,2))
        identity=dict(external=list(map(encoded,external)),internal=list(map(encoded,internal)),order=order,
            epsilon=str(epsilon),dps=dps,recursion_sources=source_id,method='fixed-weight-b-duality-resonance-limit-v2')
        target=folder/(fingerprint(identity)+'.json')
        if target.exists():
            row=json.loads(target.read_text())
            if row['identity']!=identity or row['sha256']!=fingerprint(row['payload']):raise ValueError('changed node checkpoint')
            payload=row['payload']
        else:
            if new>=max_new_nodes:raise MissingBoundaryInput('elliptic preparation node limit; resume retained checkpoints')
            limits=[]
            def extrapolate(eps,precision):
                with mp.workdps(precision+20):
                    eps=mp.mpf(eps);tables=[]
                    for offset in (eps,2*eps):
                        b=1+offset
                        table,limit=regular_h_coefficients(central_charge=1+6*(b+1/b)**2,
                            external_weights=external,internal_weights=internal,order=order,dps=precision,
                            workspace=workspace,on_recursion_call=on_recursion_call)
                        tables.append(table);limits.append(limit)
                    return {k:(4*(1+eps)*tables[0][k]-(1+2*eps)*tables[1][k])/(3+2*eps)
                            for k in tables[0]}
            with mp.workdps(dps+40):
                coarse=extrapolate(epsilon,dps);fine=extrapolate(mp.mpf(epsilon)/2,dps)
                checked=extrapolate(mp.mpf(epsilon)/2,dps+20)
                difference=lambda a,b:float(max(abs(a[k]-b[k])/max(1,abs(a[k]),abs(b[k])) for k in a))
                audit=dict(regulator_scaled_change=difference(fine,coarse),precision_scaled_change=difference(checked,fine),
                           internal_weight_limit_scaled_change=max(limits))
                if audit['regulator_scaled_change']>2e-8 or audit['precision_scaled_change']>2e-11:
                    raise MissingBoundaryInput('elliptic regulator/precision refinement not adequate at this node')
                payload=dict(coefficients=[encoded(checked[tuple(k)]) for k in levels],audit=audit)
            row=dict(identity=identity,payload=payload,sha256=fingerprint(payload))
            from genus1_production_support import write_json
            write_json(target,row);new+=1
        coefficients.append([complex(*v) for v in payload['coefficients']]);audits.append(payload['audit'])
    result={k:arrays[k][selected] for k in ('momenta','log_weights','structure_logs')}
    result.update(levels=levels,coefficients=np.asarray(coefficients))
    metadata=dict(format=EllipticGridBank.FORMAT,cut_axis=cut_axis,order=order,
        native_external_weights=list(map(encoded,d)),spectral_complete=node_indices is None,
        source_node_indices=selected.tolist(),
        spectral_convergence_assessed=False,source_native_data_sha256=native.metadata['data_sha256'],
        regulator_checked=True,epsilon=str(epsilon),dps=dps,recursion_sources=sources,node_audits=audits)
    EllipticGridBank(metadata,result)
    metadata=freeze_arrays(path,metadata,result)
    return EllipticGridBank(metadata,result)
