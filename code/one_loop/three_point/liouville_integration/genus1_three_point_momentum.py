"""Geometry-dependent, refinement-gated Liouville momentum integration."""
from __future__ import annotations

# genus1_three_point_momentum
# Geometry-adapted momentum proposals for the unintegrated torus 3-point CFT.
# 
# No collar denominators are included here. Gaussian/threshold/asymptotic
# weights are undone exactly. The CFT, block orders, frame, and dP/pi measure
# are unchanged. The block-only corrected exponential is an experimental
# proposal, not an approximation replacing DOZZ or the conformal block.



import cmath
import math
import numpy as np

from genus1_three_point_channel_atlas import (
    nearest_torus_displacement, canonical_loop_displacement,
    pair_disc_to_flat_log_factor,
)
from genus1_three_point_collision import _common_factor, _exact_koba
from liouville_torus import log_yin_structure_constant_momentum
from liouville_momentum_quadrature import (
    generalized_gaussian_momentum_rule, cft_asymptotic_momentum_rule,
)


def geometry(atlas, w1, w2, tau, pair, channel):
    atlas.sewing_chart(w1,w2,tau,channel,pair).require()
    points = (complex(w1), complex(w2), 0j)
    first, second = pair
    spectator = next(i for i in range(3) if i not in pair)
    delta = nearest_torus_displacement(points[first] - points[second], tau)
    v = cmath.exp(-1j*delta) - 1
    d = tuple(atlas.external_weights[i] for i in (*pair,spectator))
    flat = pair_disc_to_flat_log_factor(delta, d[0], d[1])
    if channel == 'pair_ope':
        loop = canonical_loop_displacement(points[second]-points[spectator],tau)
        logs = (cmath.log(v), 1j*loop, 1j*(2*math.pi*tau-loop))
        variables = (v, cmath.exp(logs[1]), cmath.exp(logs[2]))
        constant = 2*math.log(abs(v)) + 2*flat.real
        high = 1 + int(abs(variables[2]) > abs(variables[1]))
    elif channel == 'comb_ope':
        total = nearest_torus_displacement(points[spectator]-points[second],tau)
        vt = cmath.exp(-1j*total)-1
        if vt == 0j:
            raise ValueError('comb sewing requires a nonzero outer OPE coordinate')
        # The h_pair-dependent second disc-to-flat factor supplies 1/|vt|^(2 h_pair).
        logs = (cmath.log(v/vt), cmath.log(vt), 2j*math.pi*tau)
        variables = (v/vt,vt,cmath.exp(logs[2]))
        second_flat = .5j*(1-d[2])*total - (1+d[2])*cmath.log(2*cmath.sin(total/2))
        constant = 2*math.log(abs(v)) + 2*math.log(abs(vt)) + 2*(flat+second_flat).real
        high = 0
    else:
        raise ValueError('this proposal layer supports full pair/comb OPE, not collar projections')
    decay = np.asarray([-2*z.real for z in logs])
    if not np.all(np.isfinite(decay)) or np.any(decay<=0):
        raise ValueError('primary Gaussian not decaying on every edge; select a different channel')
    return dict(coefficient_basis='native-sewing-v2',channel=channel,pair=tuple(pair),external_indices=(*pair,spectator),
                external_weights=d,external_momenta=tuple(atlas.external_momenta[i] for i in (*pair,spectator)),
                logs=logs,variables=variables,decay=decay,constant=float(constant),high=high,
                common_log=cmath.log(_common_factor(atlas,tau)*_exact_koba(atlas,w1,w2,tau)))


def rules(kin, orders, scheme, decay_scale=1.):
    """Return dP (not dP/pi) weights for compatibility with the genus-one atlas."""
    if len(orders)!=3 or any(int(n)!=n or n<1 for n in orders):
        raise ValueError('three positive integer momentum orders required')
    if not math.isfinite(decay_scale) or decay_scale<=0:
        raise ValueError('positive finite decay scale required')
    result=[]
    for logq,n,primary in zip(kin['logs'],orders,kin['decay']):
        if scheme=='cft-asymptotic':
            q=cmath.exp(logq)
            if q==0:
                # q_eff/q -> 1. Preserve the logarithmic long-tube coefficient.
                a=float(primary);tempered=False
            else:
                correction=cft_asymptotic_momentum_rule(q,int(n),log_q_abs=logq.real)
                a=float(correction.effective_decay_coefficient);tempered=correction.asymptotic_tempered
            alpha=.5
        elif scheme=='dozz-threshold':
            # Quadratic curvature of log|DOZZ product / prod(P_i^2)| at threshold.
            # This is a proposal only; retain the complete DOZZ in log_density.
            raw=float(primary)-float(kin['dozz_threshold_slopes'][len(result)])
            a=max(.75*float(primary),raw);alpha=.5;tempered=raw<a
        elif scheme in ('primary-gaussian','threshold-gaussian'):
            a=float(primary);alpha=-.5 if scheme=='primary-gaussian' else .5;tempered=False
        else:
            raise ValueError(f'unknown momentum scheme {scheme}')
        shared=generalized_gaussian_momentum_rule(cmath.exp(logq),int(n),
            laguerre_alpha=alpha,decay_coefficient=decay_scale*a,log_q_abs=logq.real)
        weights=math.pi*np.asarray(shared.weights)
        if not np.all(np.isfinite(weights)) or np.any(weights<=0):
            raise FloatingPointError('invalid quadrature weights')
        result.append(dict(nodes=list(shared.nodes),weights=weights.tolist(),
            order=int(n),decay=decay_scale*a,primary_decay=float(primary),alpha=alpha,
            tempered=bool(tempered),scheme=scheme,measure='dP',domain='[0,infinity)'))
    return result


def structure_log(special, kin, momenta):
    p0,p1,p2=momenta
    a,b,c=kin['external_momenta']
    triples=((a,b,p0),(p1,p0,p2),(p2,c,p1)) if kin['channel']=='pair_ope' else (
        (a,b,p0),(p1,c,p0),(p2,p1,p2))
    return sum((complex(log_yin_structure_constant_momentum(special,*args)) for args in triples),0j)


def threshold_slopes(special, kin, epsilon=.005):
    """Finite-difference threshold curvature; callers audit epsilon dependence.

    The returned coefficients are derivatives with respect to P_i^2, after
    removing the three threshold zeros. Coupled/nonquadratic DOZZ structure
    remains in the exactly reweighted residual and is not asserted Gaussian.
    """
    if not math.isfinite(epsilon) or epsilon<=0:raise ValueError('invalid threshold probe')
    base=np.full(3,epsilon)
    f=structure_log(special,kin,base).real-2*np.log(base).sum()
    slopes=[]
    for i in range(3):
        point=base.copy();point[i]*=2
        g=structure_log(special,kin,point).real-2*np.log(point).sum()
        slopes.append(float((g-f)/(3*epsilon**2)))
    if not np.all(np.isfinite(slopes)):raise FloatingPointError('nonfinite threshold curvature')
    return slopes


def log_density(special, kin, momenta, coefficients):
    """Full reduced CFT density including 1/pi^3, with one final exponential.

    Coefficients are the unchanged native-sewing c-recursion output.
    Keep DOZZ, primary damping, and block magnitude in log form to
    avoid losing a tiny DOZZ factor before multiplying by a large block.
    """
    if kin.get('coefficient_basis') != 'native-sewing-v2':
        raise ValueError('native-sewing-v2 kinematics and raw coefficients required')
    powers=[z**np.arange(n) for z,n in zip(kin['variables'],coefficients.shape)]
    block=np.einsum('ijk,i,j,k->',coefficients,*powers,optimize=True)
    if not np.isfinite(block):raise FloatingPointError('nonfinite conformal block')
    if block==0:return complex(-math.inf,0)
    logc=structure_log(special,kin,momenta)
    z=logc+kin['constant']-np.dot(kin['decay'],np.square(momenta))
    z+=2*math.log(abs(block))+kin['common_log']-3*math.log(math.pi)
    if not (math.isfinite(z.real) and math.isfinite(z.imag)):
        raise FloatingPointError('nonfinite logarithmic CFT density')
    return complex(z)


def exp_weighted(logvalue, logweight=0.):
    z=complex(logvalue)+float(logweight)
    if z.real == -math.inf:return 0j
    if not math.isfinite(z.real) or z.real>math.log(np.finfo(float).max):
        raise FloatingPointError('weighted CFT density overflow')
    value=cmath.exp(z)
    if not np.isfinite(value):raise FloatingPointError('nonfinite weighted CFT density')
    return value


# genus1_spectral_controls
# Term-specific dP rules, exact node unions, and assessed mixed defects.
# 
# These are proposals and finite-rule diagnostics. They never replace the
# DOZZ product, introduce a physical momentum cutoff, or certify block tails.



from dataclasses import dataclass
import hashlib
import json
import math
import numpy as np

from genus1_moduli_sampling import integer

from liouville_momentum_quadrature import generalized_gaussian_momentum_rule


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


@dataclass(frozen=True)
class AxisRule:
    nodes: np.ndarray
    weights: np.ndarray
    metadata: dict

    def __post_init__(self):
        for name in ('nodes','weights'):
            a = np.asarray(getattr(self,name),float).copy()
            if a.ndim != 1 or not len(a) or not np.all(np.isfinite(a)) or np.any(a <= 0):
                raise ValueError('finite positive dP nodes and weights required')
            a.flags.writeable = False
            object.__setattr__(self,name,a)
        if self.nodes.shape != self.weights.shape:
            raise ValueError('axis nodes and weights have different shapes')


def endpoint_alpha(term, axis, cut_axis):
    if term in ('full','remainder'): return .5
    if term == 'addback': return 0. if axis == cut_axis else .5
    if term == 'neutral_triple_scale': return -.5 if axis == cut_axis else .5
    raise ValueError('unknown endpoint term; derive its threshold before assigning a rule')


def threshold_curvature(special, kin, epsilon=.005):
    """Proposal-only curvature, valid also for necklace graph ordering."""
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError('positive threshold probe required')
    p = np.full(3,epsilon)
    from genus1_native_bank import graph_structure_log
    base = graph_structure_log(special,kin,p).real-2*np.log(p).sum()
    slopes=[]
    for axis in range(3):
        other=p.copy();other[axis]*=2
        value=graph_structure_log(special,kin,other).real-2*np.log(other).sum()
        slopes.append(float((value-base)/(3*epsilon**2)))
    if not np.all(np.isfinite(slopes)):
        raise FloatingPointError('nonfinite DOZZ threshold curvature')
    return tuple(slopes)


def anchor_decays(decays, slopes):
    p,s = np.asarray(decays,float),np.asarray(slopes,float)
    if p.shape != (3,) or s.shape != (3,) or not np.all(np.isfinite(p+s)) or np.any(p <= 0):
        raise ValueError('three positive geometric decays and finite curvatures required')
    a = np.maximum(.75*p,p-s)
    return tuple(float(2.**round(math.log2(x))) for x in a)


def gaussian_rules(anchors, orders, *, term, cut_axis, width_scale=1., grid_design='offset-width'):
    if len(anchors) != 3 or len(orders) != 3 or cut_axis not in (0,1,2):
        raise ValueError('three momentum axes required')
    if not math.isfinite(width_scale) or width_scale <= 0:
        raise ValueError('positive physical width multiplier required')
    if grid_design not in ('offset-width','shared-order'):
        raise ValueError('unknown Gaussian tensor design')
    for attempt in range(4):
        step=1e-3*2**attempt
        scales=(1.,1.+math.sqrt(2)*step,1.+math.sqrt(3)*step) if grid_design=='offset-width' else (1.,)*3
        result=[]
        for axis,(a,n) in enumerate(zip(anchors,orders)):
            n=integer(n,'momentum order',1)
            if not math.isfinite(a) or a <= 0: raise ValueError('positive Gaussian anchor required')
            # Recompute each complete proposal, including dP weights. The
            # distinct decay multipliers match the native bank builder.
            decay = a*scales[axis]/width_scale**2
            alpha = endpoint_alpha(term,axis,cut_axis)
            r = generalized_gaussian_momentum_rule(0j,n,laguerre_alpha=alpha,
                decay_coefficient=decay,log_q_abs=-decay/2)
            result.append(AxisRule(np.asarray(r.nodes),math.pi*np.asarray(r.weights),
                dict(scheme='threshold-gaussian',alpha=alpha,decay=decay,measure='dP',
                     domain='[0,infinity)',width_scale=width_scale,
                     grid_design=grid_design,proposal_axis_scale=scales[axis])))
        hs=[1+np.square(r.nodes) for r in result]
        if grid_design=='shared-order' or not any(np.any(hs[i][:,None]==hs[j][None,:])
                                                for i in range(3) for j in range(i+1,3)):
            return tuple(result)
    raise ArithmeticError('Gaussian spectral rules still contain coincident internal weights')


def panel_rule(anchor, *, core_order=8, tail_order=16, delta=None, width_scale=1.):
    """Linear finite core plus log-mapped INFINITE tail, with exact dP weights."""
    if not math.isfinite(anchor) or anchor <= 0 or not math.isfinite(width_scale) or width_scale <= 0:
        raise ValueError('positive anchor and physical width required')
    if delta is not None and (not math.isfinite(delta) or delta <= 0):
        raise ValueError('derive a positive kernel/pole distance before using delta')
    width=width_scale/math.sqrt(anchor)
    split=width if delta is None else min(width,4*delta)
    def unit(n):
        x,w=np.polynomial.legendre.leggauss(integer(n,'panel order',1))
        return (x+1)/2,w/2
    u,v=unit(core_order);x,y=unit(tail_order)
    nodes=np.concatenate((split*u,split-2*width*np.log1p(-x)))
    weights=np.concatenate((split*v,2*width*y/(1-x)))
    return AxisRule(nodes,weights,dict(scheme='core-infinite-log-tail',measure='dP',
        domain='[0,infinity)',core_order=core_order,tail_order=tail_order,split=split,
        delta=delta,width_scale=width_scale,physical_momentum_cutoff=None))


def rule_variants(anchors, *, term, cut_axis, baseline=8, refinements=(12,16),
                  include_axes=False, include_panels=False, distances=(None,None,None)):
    base = gaussian_rules(anchors,(baseline,)*3,term=term,cut_axis=cut_axis)
    result={'baseline':base}
    if include_axes:
        for axis in range(3):
            for n in refinements:
                orders=[baseline]*3;orders[axis]=n
                result[f'axis{axis}_n{n}']=gaussian_rules(anchors,orders,term=term,cut_axis=cut_axis)
        finest=max((baseline,*refinements))
        result['fine']=gaussian_rules(anchors,(finest,)*3,term=term,cut_axis=cut_axis)
        for scale in (.8,1.25):
            result[f'fine_width_{scale:g}']=gaussian_rules(anchors,(finest,)*3,
                term=term,cut_axis=cut_axis,width_scale=scale)
    if include_panels:
        if len(distances)!=3: raise ValueError('three derived distances, or None, required')
        for axis in range(3):
            for core,tail in ((8,16),(12,16),(8,24)):
                rules=list(base)
                rules[axis]=panel_rule(anchors[axis],core_order=core,tail_order=tail,delta=distances[axis])
                result[f'panel_axis{axis}_{core}_{tail}']=tuple(rules)
    return result


def tensor_rule(rules):
    if len(rules)!=3: raise ValueError('three independent dP rules required')
    mesh=np.meshgrid(*(r.nodes for r in rules),indexing='ij')
    p=np.stack([m.ravel() for m in mesh],axis=1)
    ws=np.meshgrid(*(np.log(r.weights) for r in rules),indexing='ij')
    w=sum(ws).ravel()
    return p,w


class NodeCatalog:
    """Exact graph+IEEE-node union with maximum requested tensor per node.

    Quadrature weights belong to requests, not to the shared node. N8/N12
    rules are never treated as nested. No interpolation is performed.
    """
    def __init__(self):
        self.nodes={};self.requests={}

    def request(self, kin, rules, orders):
        if len(orders)!=3 or any(type(n) is not int or n<0 for n in orders):
            raise ValueError('three nonnegative block orders required')
        from genus1_native_bank import identity
        key=identity(kin); p,w=tensor_rule(rules)
        node_ids=[]
        for point in p:
            node_id=fingerprint(dict(identity=key,momenta_hex=[float(x).hex() for x in point]))
            previous=self.nodes.get(node_id)
            maximum=list(orders) if previous is None else [max(a,b) for a,b in zip(orders,previous['orders'])]
            self.nodes[node_id]=dict(identity=key,momenta=point.tolist(),orders=maximum)
            node_ids.append(node_id)
        request=dict(identity=key,node_ids=node_ids,log_weights=w.tolist(),orders=list(orders),
                     quadrature=[r.metadata for r in rules],measure='product_dP')
        request_id=fingerprint(request)
        self.requests[request_id]=request
        return request_id

    def record(self):
        return dict(format='genus1-cusp-node-catalog-v1',nodes=self.nodes,requests=self.requests)


def mixed_defect(coarse_low, coarse_high, fine_low, *, fine_high=None, tolerance=None):
    """Full nonchiral spectral sums only. Missing direct high never passes."""
    values=[complex(v) for v in (coarse_low,coarse_high,fine_low)]
    if not np.all(np.isfinite(values)): raise ValueError('finite complete spectral sums required')
    estimate=values[2]+values[1]-values[0]
    defect=None if fine_high is None else estimate-complex(fine_high)
    if defect is not None and not np.isfinite(defect): raise ValueError('nonfinite direct high sum')
    if tolerance is not None and (not math.isfinite(tolerance) or tolerance < 0):
        raise ValueError('finite nonnegative absolute tolerance required')
    return dict(estimate=estimate,direct_high=fine_high,defect=defect,assessed=defect is not None,
        passed=None if tolerance is None or defect is None else abs(defect)<=tolerance,
        rigorous_tail_bound=None)


# genus1_spectral_integration
# Geometry-dependent, refinement-gated Liouville momentum integration.
# 
# The returned quadrature weights undo the Gaussian proposal exactly. Blocks,
# DOZZ, primary factors, integration domain and dP/pi measure are not replaced.
# Refinement diagnostics are empirical estimates, not rigorous error bounds.


import copy
import math
from dataclasses import dataclass, field
from itertools import permutations
from numbers import Integral
import numpy as np

from genus1_two_point_worldsheet import MomentumRule
from liouville_momentum_quadrature import generalized_gaussian_momentum_rule
from liouville_torus import UpsilonB


class SpectralConvergenceError(ArithmeticError):
    def __init__(self, diagnostic):
        self.diagnostic=diagnostic
        super().__init__(f"spectral quadrature unresolved: {diagnostic}")


@dataclass
class GeometryMomentumPolicy:
    initial_order: int = 10
    step: int = 4
    maximum_order: int = 26
    rtol: float = 1e-4
    atol: float = 0.
    width_check_scale: float = .8
    # Recompute complete quadrature weights at distinct proposal widths,
    # avoiding unnecessary confluent limits while retaining N^3 nodes.
    grid_design: str = 'offset-width'
    completed: int = field(default=0,init=False)
    failed: int = field(default=0,init=False)
    largest_order: int = field(default=0,init=False)
    largest_relative_change: float = field(default=0.,init=False)
    last_diagnostic: dict = field(default_factory=dict,init=False)

    def __post_init__(self):
        if self.grid_design not in ('shared-order','offset-width','staggered','staggered-symmetric'):
            raise ValueError('unknown momentum tensor design')
        if any(int(n)!=n or n<1 for n in (self.initial_order,self.step,self.maximum_order)):
            raise ValueError('positive integer spectral orders and step required')
        if self.maximum_order < self.initial_order+2*self.step:
            raise ValueError('spectral budget must allow two successive refinement checks')
        if not (math.isfinite(self.rtol) and 0<self.rtol<1 and
                math.isfinite(self.atol) and self.atol>=0 and
                math.isfinite(self.width_check_scale) and 0<self.width_check_scale<1):
            raise ValueError('invalid spectral accuracy/proposal settings')

    def rules(self, logs, order, scale=1.):
        """Raw per-edge rule builder; rule_components applies the tensor design."""
        if len(logs)!=3 or not all(math.isfinite(complex(z).real) for z in logs):
            raise ValueError('three finite primary logarithms required')
        orders=(int(order),)*3 if isinstance(order,Integral) else tuple(order)
        if len(orders)!=3 or any(int(n)!=n or n<1 for n in orders):
            raise ValueError('three positive integer momentum orders required')
        if not math.isfinite(scale) or scale<=0:
            raise ValueError('positive finite proposal scale required')
        result=[]
        for logq,n in zip(logs,orders):
            # Integrated collars can reach a zero-decay endpoint. A positive
            # proposal floor is still exactly reweighted, and must pass the
            # order AND width checks; it is not asserted to be the true tail.
            decay=scale*max(-2*complex(logq).real,.25)
            raw=generalized_gaussian_momentum_rule(
                0j,int(n),laguerre_alpha=.5,decay_coefficient=decay,
                log_q_abs=-decay/2)
            weights=math.pi*np.asarray(raw.weights) # MomentumRule uses dP
            if not np.all(np.isfinite(weights)) or np.any(weights<=0):
                raise FloatingPointError('invalid adapted spectral weights')
            result.append(MomentumRule(np.asarray(raw.nodes),weights,math.inf,int(n),
                kind='threshold-gaussian',gaussian_width=raw.gaussian_width))
        return tuple(result)

    @staticmethod
    def coincident_weight_pairs(rules):
        """Count exact shared weights in the arithmetic used by the block API."""
        hs=[1+np.square(np.asarray(rule.nodes,float)) for rule in rules]
        return sum(int(np.count_nonzero(hs[i][:,None]==hs[j][None,:]))
                   for i in range(3) for j in range(i+1,3))

    def rule_components(self,logs,order,scale=1.):
        """Return weighted, complete tensor rules; no nodes are deleted or shifted.

        All six assignments reproduce the exchange-symmetric genus-zero
        construction. A single assignment is a cheaper, valid quadrature,
        whose finite-order exchange asymmetry can be checked separately.
        Orders alone do not prove noncoincidence for different edge widths,
        so inspect the actual nodes and rebuild the family if necessary.
        """
        if int(order)!=order or order<1:raise ValueError('positive integer order required')
        order=int(order)
        if self.grid_design=='shared-order':
            return [(1.,self.rules(logs,order,scale))]
        if self.grid_design=='offset-width':
            # Same offsets as the saved native bank builder. Rebuild the
            # complete dP rule at each width, keeping N^3 nodes. These are
            # proposal changes, never regulators of the conformal weights.
            for attempt in range(4):
                step=1e-3*2**attempt
                widths=(1.,1.+math.sqrt(2)*step,1.+math.sqrt(3)*step)
                rs=tuple(self.rules(logs,order,scale*width)[i] for i,width in enumerate(widths))
                if self.coincident_weight_pairs(rs)==0:return [(1.,rs)]
            raise ArithmeticError('offset momentum proposals still contain coincident internal weights')
        for increment in range(4):
            ns=tuple(order+increment+i for i in range(3))
            assignments=list(permutations(ns)) if self.grid_design=='staggered-symmetric' else [ns]
            components=[(1./len(assignments),self.rules(logs,assignment,scale))
                        for assignment in assignments]
            if all(self.coincident_weight_pairs(rs)==0 for _,rs in components):
                return components
        raise ArithmeticError('staggered momentum rules still contain coincident internal weights')

    def evaluate_grid(self,logs,order,evaluate,scale=1.):
        """Evaluate a fixed order, separately from the adaptive accuracy gate."""
        components=self.rule_components(logs,order,scale);values=[];records=[]
        for mixture,rs in components:
            value=complex(evaluate(rs))
            if not (math.isfinite(value.real) and math.isfinite(value.imag)):
                raise FloatingPointError('nonfinite geometry-adapted spectral integral')
            values.append(mixture*value)
            records.append(dict(mixture_weight=mixture,orders=[len(r.nodes) for r in rs],
                proposal_gaussian_widths=[r.gaussian_width for r in rs],
                coincident_weight_pairs=self.coincident_weight_pairs(rs),
                value=[value.real,value.imag]))
        total=complex(math.fsum(v.real for v in values),math.fsum(v.imag for v in values))
        return total,dict(grid_design=self.grid_design,components=records,
                         block_nodes=sum(math.prod(len(r.nodes) for r in rs) for _,rs in components))

    @staticmethod
    def _necklace_clone(necklace,rules):
        shadow=copy.copy(necklace)
        shadow.momentum_rules=rules;shadow.momentum_rule=None
        shadow._banks={};shadow._cached_structure_matrices=None
        shadow.spectral_controller=None
        # Limit special-function/momentum tensor caches to this quadrature.
        shadow.special=UpsilonB(1.,dps=necklace.special_dps)
        shadow.order_histogram=dict(necklace.order_histogram)
        shadow.high_edge_histogram=dict(necklace.high_edge_histogram)
        return shadow

    def integrate_necklace(self,necklace,logs,evaluate,*,label='necklace'):
        return self._integrate(logs,lambda rs:evaluate(self._necklace_clone(necklace,rs)),label)

    def integrate_atlas(self,atlas,logs,evaluate,*,label):
        def at_rules(rs):
            shadow=copy.copy(atlas)
            shadow.necklace=self._necklace_clone(atlas.necklace,rs)
            shadow.special=shadow.necklace.special
            shadow._banks={};shadow._comb_banks={};shadow.bank_cache_path=None
            shadow.spectral_controller=None
            shadow.channel_counts=dict(atlas.channel_counts);shadow.pair_counts=dict(atlas.pair_counts)
            # Bound methods installed by cache-only historical audit objects
            # must not accidentally load their old fixed-node banks.
            if any(name in atlas.__dict__ for name in ('_build_bank','_build_comb_bank')):
                raise ValueError('geometry-adapted integration cannot use a cache-only fixed-grid atlas')
            return evaluate(shadow)
        return self._integrate(logs,at_rules,label)

    def _integrate(self,logs,evaluate,label):
        history=[];last=None;stable=0;accepted=False;worst=math.inf
        for order in range(self.initial_order,self.maximum_order+1,self.step):
            value,grid=self.evaluate_grid(logs,order,evaluate)
            if not (math.isfinite(value.real) and math.isfinite(value.imag)):
                raise FloatingPointError('nonfinite geometry-adapted spectral integral')
            tolerance=self.atol+self.rtol*abs(value)
            delta=None if last is None else abs(value-last)
            history.append(dict(order=order,value=[value.real,value.imag],change=delta,grid=grid))
            stable=stable+1 if delta is not None and delta<=tolerance and abs(value)>0 else 0
            if stable>=2:
                alternative,width_grid=self.evaluate_grid(logs,order,evaluate,self.width_check_scale)
                width_change=abs(alternative-value)
                if not math.isfinite(width_change):
                    raise FloatingPointError('nonfinite spectral proposal-width check')
                history[-1]['width_check_change']=width_change
                history[-1]['width_check_grid']=width_grid
                worst=max(delta,width_change,history[-2]['change'])
                if worst<=tolerance:
                    accepted=True;break
                stable=0
            last=value
        diagnostic=dict(status='passed_empirical_refinement' if accepted else 'unresolved',
            label=label,method='geometry-threshold-gaussian-v1',grid_design=self.grid_design,domain='[0,infinity)^3',
            rtol=self.rtol,atol=self.atol,log_magnitudes=[complex(z).real for z in logs],
            history=history,maximum_order=order,
            absolute_refinement_estimate=worst if math.isfinite(worst) else None,
            qualification='Momentum refinement only; no block/moduli or rigorous tail certification.')
        self.last_diagnostic=diagnostic
        self.largest_order=max(self.largest_order,order)
        if not accepted:
            self.failed+=1
            raise SpectralConvergenceError(diagnostic)
        self.completed+=1
        self.largest_relative_change=max(self.largest_relative_change,worst/abs(value))
        return value

    def diagnostics(self):
        return dict(method='geometry-threshold-gaussian-v1',grid_design=self.grid_design,completed=self.completed,failed=self.failed,
                    largest_order=self.largest_order,largest_relative_change=self.largest_relative_change,
                    rtol=self.rtol,atol=self.atol,initial_order=self.initial_order,
                    step=self.step,maximum_order=self.maximum_order,width_check_scale=self.width_check_scale,
                    qualification='Empirical quadrature refinement; not a rigorous or full-amplitude error bound.')
