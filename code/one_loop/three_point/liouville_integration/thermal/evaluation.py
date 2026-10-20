"""Read-only evaluation of verified thermal reference banks."""
import genus1_source_paths as bootstrap
bootstrap.install()
from contextlib import contextmanager
from functools import partial
import math
import threading
import numpy as np
from genus1_boundary_batch import GeometryStore, BatchedPrimary, SharedKernelSlice
from genus1_analytic_boundary_layer import PrimarySlice
from genus1_nonseparating import _sum_logs
from genus1_three_point_channel_atlas import make_atlas
from genus1_moduli_sampling import ShortPoint
from genus1_channel_evaluation import pair_projection_values
import genus1_integrated_scale_collar as scale
from genus1_native_bank import digest
from banks import Banks, implementation_sources
from kernels import primary_kernel_logs, pair_lattice, height_matrices
from remainders import comb_term

_HEIGHT_LOCK = threading.RLock()


class _LeadingPairPrefix:
    """Evaluate the retained prefix without repeating unused lower prefixes."""
    def __init__(self, source):
        self.source = source

    def cuts(self):
        return self.source.cuts()[:1]

    def pair_logs(self, *args):
        return self.source.pair_logs(*args)


@contextmanager
def thermal_height(R):
    # The legacy evaluator has a module-owned height callback. Serialize the
    # temporary substitution; restoration also occurs on an arithmetic error.
    with _HEIGHT_LOCK:
        original = scale.reference.height_matrices
        try:
            scale.reference.height_matrices = partial(height_matrices, R=R)
            yield
        finally:
            scale.reference.height_matrices = original


def columns(old, new):
    a = -2*math.pi**2*np.asarray([old.imag, new.imag])
    return np.r_[a, a[1]-a[0]]


class Evaluator:
    def __init__(self, output, kin):
        self.output, self.kin = output, kin
        self.atlas = make_atlas(kin.t)
        self.banks = Banks(output, limit=32)
        self.engines, self.heights, self.geometries = {}, {}, {}
        self.height_weight_rounding = {}

    def load(self, key):
        spec, pin, native, elliptic = self.banks.get(key)
        if spec['grid']['thermal_sources'] != implementation_sources():
            raise ValueError('thermal rule source changed after bank preparation')
        for field in ('paired_quadrature_sources','placement_sources'):
            for name,expected in spec['grid'].get(field,{}).items():
                if digest(bootstrap.ROOT/name)!=expected:
                    raise ValueError('quadrature source changed after preparation: '+field)
        if spec['grid']['t'] != self.kin.t:
            raise ValueError('bank has different external Liouville momenta')
        return spec, pin, native, elliptic

    def primary(self, key, shorts, *, order=4, threshold=-math.log(.05)):
        spec, pin, native, bank = self.load(key)
        axis = bank.metadata['cut_axis']
        if key not in self.engines:
            # A same-frequency bank may be replayed at several KK radii.
            # Each process family needs its own SQLite writer namespace.
            self.geometries[key] = GeometryStore(self.atlas,self.output/'geometry'/f'N{self.kin.N}'/key)
            self.engines[key] = BatchedPrimary(self.atlas,{axis:bank},self.geometries[key],orders=[2,order])
        engine = self.engines[key]
        rows = self.geometries[key].prepare(shorts)
        blocks = engine.block_logs(rows,axis,[order])[0]
        p = bank.arrays['momenta']
        base = bank.arrays['log_weights']+bank.arrays['structure_logs']-3*math.log(math.pi)
        unique, inverse = engine.cuts[axis]
        result = []
        for row, short, block in zip(rows,shorts,blocks):
            if (2-short.sector.index(2)) % 3 != axis:
                raise ValueError('wrong cut axis for primary bank')
            logs = [complex(*v) for v in row['short_logs']]
            exponent = sum(2*p[:,(axis+k)%3]**2*z.real for k,z in enumerate(logs,1))
            nodes = PrimarySlice(p[:,axis],base+exponent+2*block,short.total,row['sine'],row['gaussian'])
            shared = SharedKernelSlice(nodes,unique,inverse,{})
            old_kernel, new_kernel = primary_kernel_logs(unique,short,self.kin,threshold)
            old = _sum_logs(shared.coefficients+row['sine']+old_kernel)[0]
            new = _sum_logs(shared.coefficients+row['sine']+new_kernel)[0]
            result.append(columns(old,new))
        return np.asarray(result)

    def height(self, key, geometry=None):
        spec, pin, bank, ellipse = self.load(key)
        g = dict(spec['case']['geometry'] if geometry is None else geometry)
        pair = g.pop('pair', spec['case']['geometry']['pair'])
        g['term'] = spec['case']['kind']
        g.pop('shape_anchor',None)
        if isinstance(g.get('shape'),list):
            g['shape'] = complex(*g['shape'])
        if key not in self.heights:
            arrays = {k:v for k,v in bank.arrays.items() if k != 'coefficients'}
            arrays['projection_coefficients'] = bank.arrays['coefficients'][:,:,0,:]
            # Native tensor_rule stores log(float dP weight); a designed rule
            # can also retain its pre-exp log. Use the native convention here.
            # Cross-runtime elementary-function rounding is bounded below;
            # the immutable bank, physical nodes and dP weights are untouched.
            rules=[dict(r,log_weights=np.log(r['weights']).tolist()) for r in spec['grid']['rules']]
            meta = dict(evaluation_scope=scale.SCOPE, orders=list(bank.orders),
                rules=rules, pair=pair, region=g['region'],
                external_weights=bank.metadata['identity']['external_weights'])
            _,canonical_weights=scale.node_inputs(meta)
            difference=abs(canonical_weights-arrays['log_weights'])
            counts=tuple(len(r['nodes']) for r in rules)
            indices=np.array(np.unravel_index(np.arange(math.prod(counts)),counts)).T
            magnitude=sum(abs(np.asarray(r['log_weights'])[indices[:,a]]) for a,r in enumerate(rules))
            if np.any(difference>8*np.finfo(float).eps*np.maximum(1.,magnitude)):
                raise ValueError('native tensor weights differ beyond elementary-function rounding')
            self.height_weight_rounding[key]=float(max(difference))
            arrays['log_weights']=canonical_weights
            self.heights[key] = scale.PreparedBank(arrays,meta,
                dict(receipt_sha256=pin['files']['input/exports/bank.json']))
        prepared = self.heights[key]
        # Hold the lock for BOTH evaluations; another thread cannot make an
        # intended noncompact contraction use a temporary thermal callback.
        with _HEIGHT_LOCK:
            old, old_checks = scale.evaluate(prepared,self.atlas,**g,elliptic=ellipse)
            with thermal_height(self.kin.R):
                new, new_checks = scale.evaluate(prepared,self.atlas,**g,elliptic=ellipse)
        if not all(c['passed'] for c in (*old_checks,*new_checks)):
            raise FloatingPointError('unresolved height-contraction arithmetic')
        return np.asarray([columns(a,b) for a,b in zip(old,new)])

    def pair(self, key, geometry=None, *, absolute_arithmetic=False, leading_only=False):
        spec, pin, bank, ellipse = self.load(key)
        g = dict(spec['case']['geometry'] if geometry is None else geometry)
        for name in ('tau','delta_total'):
            if isinstance(g[name],list): g[name] = complex(*g[name])
        g['integrated'] = True
        if leading_only:
            ellipse = _LeadingPairPrefix(ellipse)
        old, checks = pair_projection_values(bank,self.atlas,g,bank.orders,elliptic=ellipse)
        flagged = not checks['passed']
        if not checks['passed']:
            # A relative check can fail near underflow, including when the
            # total is normal but constituent nodes are subnormal. Permit
            # only a tiny, explicitly bounded case when the caller propagates
            # the absolute enclosure through its thermal factor and Jacobian.
            tiny = np.all(abs(old) < 1e-250)
            bounded = all(c.get('absolute_arithmetic_log_bound') is not None
                          and c['absolute_arithmetic_log_bound'] < -600
                          for c in checks['prefixes'])
            if not (absolute_arithmetic and tiny and bounded):
                raise FloatingPointError('unresolved pair projection arithmetic')
        factor = pair_lattice(self.kin,g['tau'],g['pair'],g['delta_total'])
        values = np.asarray([columns(v,v*factor) for v in old])
        if not absolute_arithmetic:
            return values
        bounds = np.zeros_like(values)
        for i, check in enumerate(checks['prefixes']):
            if flagged or check.get('range_handling') == 'absolute_retained_density_enclosure':
                # Rounding the exponent UP avoids losing the bound to underflow.
                enclosure = math.exp(max(check['absolute_arithmetic_log_bound'], -700.))
                bounds[i] = 2*math.pi**2*enclosure*np.array([1.,abs(factor),1.+abs(factor)])
        return values, bounds, int(flagged)

    def close(self):
        for store in self.geometries.values():store.close()
