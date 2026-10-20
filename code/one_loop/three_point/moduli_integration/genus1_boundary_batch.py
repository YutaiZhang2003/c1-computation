"""Shared checked geometry and explicitly truncated elliptic contractions.

Only geometry and polynomial contractions are prepared here. No recursion or
structure constants are computed, and no missing spectrum is approximated.
Logarithmic coordinates retain the simultaneous long-tube domain.
"""

from genus1_source_paths import ROOT as _SOURCE_ROOT, source_path as _source_path
import cmath
import math
from pathlib import Path
import time
import numpy as np

from genus1_production_support import Records
from genus1_boundary_cubature import MissingBoundaryInput, fingerprint
from genus1_boundary_banks import log_segment_products, integrated_value, sha, _vendor
from genus1_analytic_boundary_layer import necklace_slice_geometry, PrimarySlice
from genus1_moduli_sampling import TAU
from genus1_nonseparating import log_length_kernel, _sum_logs
from genus1_boundary_banks import DEFAULT_ELLIPTIC_ORDERS, elliptic_orders


class SharedKernelSlice:
    """Contract short momenta before evaluating the distinct cut kernels.

    A tensor spectral grid repeats each cut momentum N**2 times. The kernel
    depends only on that cut momentum and geometry, not the short momenta or
    block order. Complex coefficients and their absolute sums are contracted
    separately with scaled compensated sums; cancellation is not discarded.
    """
    def __init__(self,nodes,unique,inverse,cache):
        self.nodes=nodes;self.unique=unique;self.inverse=inverse;self.cache=cache
        self.short_length=nodes.short_length
        coefficients=[];absolute=[]
        for k in range(len(unique)):
            z=nodes.log_coefficients[inverse==k];z=z[np.isfinite(z.real)]
            if not len(z):coefficients.append(complex(-math.inf,0.));absolute.append(-math.inf);continue
            peak=float(max(z.real));scaled=np.exp(z-peak)
            total=complex(math.fsum(scaled.real),math.fsum(scaled.imag))
            coefficients.append(complex(-math.inf,0.) if total==0 else peak+cmath.log(total))
            absolute.append(peak+math.log(math.fsum(abs(scaled))))
        self.coefficients=np.asarray(coefficients);self.absolute=np.asarray(absolute)

    def integrated(self,lower):
        if lower not in self.cache:
            n=self.nodes
            self.cache[lower]=log_length_kernel(self.unique,lower_length=lower,
                short_length=n.short_length,coupling=math.sqrt(2*n.gaussian),log_prefactor=n.sine)
        kernel=self.cache[lower]-math.log(TAU)
        value,_=_sum_logs(self.coefficients+kernel)
        absolute,_=_sum_logs(self.absolute+kernel)
        return dict(value=value,absolute_sum=absolute.real)


class LogCoordinates:
    """Same positive-real anchor and phase homotopy, with MP endpoint check.

    Retaining log nomes avoids a floating-point cutoff on infinite short
    lengths. No asymptotic block substitution is made at underflow.
    """
    def __init__(self, dps=45, phase_step=.2, maximum_nome=.95):
        if dps < 30 or not 0 < phase_step <= .5 or not 0 < maximum_nome < 1:
            raise ValueError('invalid coordinate controls')
        self.dps, self.phase_step, self.maximum_nome = dps, phase_step, maximum_nome
        self.identity = dict(method='continued-log-nomes-mp-v1',dps=dps,
                             phase_step=phase_step,maximum_nome=maximum_nome)

    @staticmethod
    def solve(target, seed):
        current = np.asarray(seed,complex).copy()
        tolerance = max(2e-13, 8*np.finfo(float).eps*max(abs(target)))
        for _ in range(50):
            value, jac = log_segment_products(current)
            residual = value-target; error = max(abs(residual))
            if error < tolerance: return current
            try: delta = np.linalg.solve(jac,residual)
            except np.linalg.LinAlgError as exc:
                raise MissingBoundaryInput('singular log-coordinate Jacobian') from exc
            for k in range(24):
                trial = current-delta*2.**-k
                if np.any(trial.real >= -1e-14): continue
                value, _ = log_segment_products(trial)
                if max(abs(value-target)) < error:
                    current = trial; break
            else: raise MissingBoundaryInput('no admissible log-coordinate descent')
        raise MissingBoundaryInput('log-coordinate iteration ceiling')

    def __call__(self, logs):
        target = np.asarray(logs,complex)
        if target.shape != (2,) or np.any(~np.isfinite(target)) or np.any(target.real >= 0):
            raise MissingBoundaryInput('finite contracting short coordinates required')
        aligned = target.real.astype(complex)
        right = aligned[1]-2*math.log1p(math.sqrt(-math.expm1(aligned[1].real)))
        current = self.solve(aligned,(aligned.sum()-math.log(16)-right,right))
        maximum = min(1.,self.phase_step/max(max(abs(target.imag)),self.phase_step))
        step = maximum; position = 0.; attempts = 0
        while position < 1:
            attempts += 1
            if attempts > 1024: raise MissingBoundaryInput('coordinate continuation work ceiling')
            next_position = min(1.,position+step)
            try:
                trial = self.solve(aligned+1j*target.imag*next_position,current)
                if max(abs(trial-current)) > .75: raise MissingBoundaryInput('large continuation step')
            except MissingBoundaryInput:
                step /= 2
                if step < 1e-7: raise MissingBoundaryInput('coordinate phase subdivisions exhausted')
                continue
            current = trial; position = next_position; step = min(maximum,step*1.5)
        if max(current.real) >= math.log(self.maximum_nome):
            raise MissingBoundaryInput('continued nomes exceed configured patch')
        _vendor()
        import mpmath as mp
        from genus0_elliptic_h_recursion.geometry import coordinates_from_segment_nomes
        with mp.workdps(self.dps):
            expected_z = mp.exp(mp.mpc(target[0])+mp.mpc(target[1]))
            expected_y = mp.exp(mp.mpc(target[1]))
            lp, rp = map(mp.mpc,current); original = (lp,rp)
            _, jac = log_segment_products(current)
            a,b,c,d = (mp.mpc(x) for x in jac.flat); det = a*d-b*c
            for _ in range(8):
                z,t = coordinates_from_segment_nomes((mp.exp(lp),mp.exp(rp)))
                rz,ry = z/expected_z,t[0]/expected_y
                if max(abs(rz-1),abs(ry-1)) < mp.mpf('1e-25'): break
                ex,ey = mp.log(rz/ry),mp.log(ry)
                lp -= (d*ex-b*ey)/det; rp -= (-c*ex+a*ey)/det
                allowance = max(mp.mpf('1e-9'),mp.mpf('1e-14')*max(abs(lp),abs(rp)))
                if max(abs(lp-original[0]),abs(rp-original[1])) > allowance:
                    raise MissingBoundaryInput('endpoint correction changed the continued branch')
            else: raise MissingBoundaryInput('MP log-coordinate forward check failed')
            return complex(float(lp.real),float(mp.arg(mp.exp(1j*lp.imag)))), complex(float(rp.real),float(mp.arg(mp.exp(1j*rp.imag))))


def log_abs_one_minus_exp(z):
    z = complex(z.real, math.remainder(z.imag,TAU))
    return math.log(abs(-np.expm1(z)))


class GeometryStore:
    """Exact-key geometry cache shared across spectral grids and block orders."""
    def __init__(self, atlas, directory=None, attempts=None):
        self.atlas = atlas
        self.resolvers = [LogCoordinates(**row) for row in (attempts or [dict(dps=45,phase_step=.2,maximum_nome=.95)])]
        self.identity = dict(format='genus1-batch-geometry-v2',
            external_weights=[[z.real,z.imag] for z in map(complex,atlas.external_weights)],
            coordinates=[r.identity for r in self.resolvers],
            sources={Path(__file__).name:sha(__file__),
                     'genus1_boundary_banks.py':sha(_source_path(Path('code/one_loop/production') / 'genus1_boundary_banks.py')),
                     'genus1_analytic_boundary_layer.py':sha(_source_path(Path('code/one_loop/production') / 'genus1_analytic_boundary_layer.py'))})
        path = None if directory is None else Path(directory)/fingerprint(self.identity)/'geometry.sqlite3'
        self.records = Records(self.identity,path); self.memory={}; self.hits=0; self.evaluations=0
        self.cpu_seconds=0.;self.coordinate_evaluations=0;self.common_hits=0

    def remember(self,key,row):
        if len(self.memory)>=4096 and key not in self.memory:self.memory.pop(next(iter(self.memory)))
        self.memory[key]=row

    def prepare(self, shorts):
        begin=time.process_time()
        try:
            with self.records.transaction():return self._prepare(shorts)
        except BaseException:
            self.memory.clear()
            raise
        finally:self.cpu_seconds+=time.process_time()-begin

    @staticmethod
    def decorate(atlas, short, common):
        """Restore sector physics around a sector-independent coordinate row."""
        g=necklace_slice_geometry(atlas,short)
        logs=[[z.real,z.imag] for z in g['short_logs']]
        if common['short_logs']!=logs:raise ValueError('prepared common geometry belongs to another point')
        return dict(short_key=fingerprint(short.record()),cut_axis=g['cut_axis'],short_logs=logs,
                    log_nomes=common['log_nomes'],features=common['features'],
                    sine=g['sine'],gaussian=g['gaussian'])

    def _prepare(self, shorts):
        rows=[]
        for short in shorts:
            key=fingerprint(short.record())
            row=self.memory.get(key)
            if row is None: row=self.records.get(key)
            if row is not None:
                self.hits+=1;self.remember(key,row);rows.append(row);continue
            g=necklace_slice_geometry(self.atlas,short);logs=g['short_logs'];errors=[]
            common_key='common:'+fingerprint([[z.real,z.imag] for z in logs])
            common=self.records.get(common_key)
            if common is not None:
                row=self.decorate(self.atlas,short,common);self.common_hits+=1
                self.remember(key,row);rows.append(row);continue
            coordinate_key='coordinates:'+fingerprint([[z.real,z.imag] for z in logs])
            saved=self.records.get(coordinate_key)
            if saved is None:
                for resolver in self.resolvers:
                    try: lp=resolver(logs);break
                    except MissingBoundaryInput as exc: errors.append(str(exc))
                else: raise MissingBoundaryInput('; '.join(errors))
                self.records.put(coordinate_key,[[z.real,z.imag] for z in lp])
                self.coordinate_evaluations+=1
            else: lp=tuple(complex(*z) for z in saved)
            q=cmath.exp(sum(lp));theta=1+0j
            for n in range(1,4096):
                term=2*q**(n*n);theta+=term
                if abs(term)<2e-16:break
            else:raise MissingBoundaryInput('theta product did not converge')
            x,y=logs
            features=[math.log(abs(theta)),(x+y).real,log_abs_one_minus_exp(x+y),
                      2*y.real+log_abs_one_minus_exp(y)+log_abs_one_minus_exp(x),
                      math.log(4)+lp[0].real,math.log(4)+lp[1].real,y.real]
            row=dict(short_key=key,cut_axis=g['cut_axis'],short_logs=[[z.real,z.imag] for z in logs],
                log_nomes=[[z.real,z.imag] for z in lp],features=features,sine=g['sine'],gaussian=g['gaussian'])
            self.records.put(common_key,{k:row[k] for k in ('short_logs','log_nomes','features')})
            self.records.put(key,row);self.remember(key,row);rows.append(row);self.evaluations+=1
        return rows

    def close(self):self.records.close()


class BatchedPrimary:
    """Vectorized bank contractions; original analytic kernel and error fields."""
    def __init__(self, atlas, banks, geometry, *, orders=DEFAULT_ELLIPTIC_ORDERS):
        self.atlas=atlas;self.banks=dict(banks);self.geometry=geometry;self.weights={};self.cuts={}
        self.orders=elliptic_orders(orders)
        self.cpu_seconds=0.;self.points=0
        for axis,bank in self.banks.items():
            if axis!=bank.metadata['cut_axis'] or not bank.metadata['spectral_complete']:
                raise MissingBoundaryInput('complete compatible cut-axis bank required')
            if self.orders[-1]>bank.metadata['order']:
                raise MissingBoundaryInput('requested elliptic order is absent from bank')
            d=[complex(*z) for z in bank.metadata['native_external_weights']]
            if not np.allclose(d,atlas.external_weights,rtol=0,atol=2e-13) or any(z.imag for z in d):
                raise ValueError('real physical external weights required')
            h=1+bank.arrays['momenta']**2
            self.cuts[axis]=np.unique(bank.arrays['momenta'][:,axis],return_inverse=True)
            dz,mu,d1=[d[(axis+k)%3].real for k in range(3)]
            d0=h[:,axis];a1=h[:,(axis+1)%3];a2=h[:,(axis+2)%3]
            self.weights[axis]=np.column_stack((12-4*(2*d0+dz+d1)-2*mu,1-a1,
                np.full(len(h),1-dz-d1),np.full(len(h),-mu/2),a1-1,a2-1,-(a2-mu-a1)))

    def block_logs(self, rows, axis, orders):
        bank=self.banks[axis];a=bank.arrays;levels=a['levels']
        if not orders or any(type(n) is not int or not 0<=n<=bank.metadata['order'] for n in orders):
            raise ValueError('unavailable elliptic order')
        # An H8 reference bank must not make an H4 production call evaluate
        # powers of its unused higher coefficients.
        retained=levels.sum(axis=1)<=max(orders)
        levels=levels[retained];coefficients=a['coefficients'][:,retained]
        logp=np.array([[complex(*v) for v in row['log_nomes']] for row in rows])
        # Some BLAS builds propagate stale floating-point flags from their
        # internal scaling. Check the actual results of every contraction.
        with np.errstate(all='ignore'):
            powers=np.exp(levels@logp.T)
            prefactor=self.weights[axis]@np.array([row['features'] for row in rows]).T
        if not np.all(np.isfinite(powers)) or not np.all(np.isfinite(prefactor)):
            raise FloatingPointError('nonfinite elliptic powers or prefactor')
        result=[]
        for order in orders:
            mask=levels.sum(axis=1)<=order
            with np.errstate(all='ignore'):
                values=coefficients[:,mask]@powers[mask]
            if not np.all(np.isfinite(values)):
                raise FloatingPointError('nonfinite elliptic bank contraction')
            with np.errstate(divide='ignore'):
                result.append((prefactor+np.log(abs(values))).T)
        return result

    def evaluate(self, shorts, policy, angle_order, *, prepared=None):
        rows=self.geometry.prepare(shorts) if prepared is None else prepared
        if len(rows)!=len(shorts):raise ValueError('geometry batch length mismatch')
        if any(row['short_key']!=fingerprint(short.record()) for row,short in zip(rows,shorts)):
            raise ValueError('prepared geometry belongs to another short point')
        begin=time.process_time()
        try:return self._evaluate(shorts,rows,policy,angle_order)
        finally:self.cpu_seconds+=time.process_time()-begin

    def _evaluate(self,shorts,rows,policy,angle_order):
        answer=[None]*len(rows)
        for axis in sorted({row['cut_axis'] for row in rows}):
            if axis not in self.banks:raise MissingBoundaryInput('absent cut-axis bank')
            indices=[i for i,row in enumerate(rows) if row['cut_axis']==axis]
            group=[rows[i] for i in indices];bank=self.banks[axis];a=bank.arrays
            blocks=self.block_logs(group,axis,self.orders)
            high=blocks[-1];low=blocks[0] if len(blocks)>1 else None
            p=a['momenta'];base=a['log_weights']+a['structure_logs']-3*math.log(math.pi)
            for j,i in enumerate(indices):
                row=rows[i];short=shorts[i];logs=[complex(*v) for v in row['short_logs']]
                exponent=sum(2*p[:,(axis+k)%3]**2*z.real for k,z in enumerate(logs,1))
                kernels={};unique,inverse=self.cuts[axis]
                def nodes(block):return SharedKernelSlice(PrimarySlice(p[:,axis],base+exponent+2*block,
                    short.total,row['sine'],row['gaussian']),unique,inverse,kernels)
                answer[i]=integrated_value(short,nodes(high[j]),policy,angle_order,
                    prefix=None if low is None else nodes(low[j]))
        self.points+=len(rows)
        return answer
