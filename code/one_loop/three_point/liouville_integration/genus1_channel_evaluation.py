"""Verified bank adapters for the seven-strata sampling contract.

No coefficient construction, momentum interpolation or inherited pointwise
accuracy gates. A resolver supplies a checksum-loaded compatible bank and
explicit requested orders. Regional bank selection remains a separate layer.
"""
from __future__ import annotations

import cmath
from dataclasses import dataclass
import math

import numpy as np

from genus1_native_bank import SharedNativeBank, identity, point_kinematics, variants
from genus1_native_log_evaluation import reduce_retained as _reduce
from genus1_comb_collar import evaluate_comb_collar
from genus1_pair_collar import evaluate_pair_remainder
from genus1_native_log_evaluation import block_logs, kinematics
from genus1_rebuild_policy import require_orders, encoded
from genus1_collision_sampling import MissingCFTBank


@dataclass(frozen=True)
class Selection:
    bank: object
    source_sha256: str
    receipt_sha256: str
    orders: tuple
    reversed_bank: object = None
    fallback_full: object = None
    fallback_geometry: object = None


def _linear_fallback(atlas, selected, request):
    """A chart-safe full CFT minus the UNCHANGED original collar projections.

    Different graphs require independent spectral sums. Expose that fact and
    enforce arithmetic conditioning of the final difference. This is never a
    crossing/convergence certificate and keeps each graph's prefix changes.
    """
    from genus1_pair_collar import PAIR_REMAINDER_MAX_CONDITION, PAIR_REMAINDER_NUMERICS_RTOL
    bank, orders = require_native(selected)
    other, other_orders = require_native(selected.fallback_full)
    fv, fc = full_values(other, atlas, selected.fallback_geometry, other_orders)
    g = request.geometry
    parents = [(fv, fc, 1., 'alternate_full')]
    if request.kind == 'pair_remainder':
        args = dict(pair=g['pair'],delta_total=g['delta_total'],tau=g['tau'],
                    radius=abs(g['delta_pair']),integrated=False)
        pv, pc = pair_projection_values(bank,atlas,args,orders)
        parents.append((pv,pc,-1.,'pair_projection'))
    elif request.kind == 'comb_collar' and g['term']=='forest_remainder':
        for term, sign in (('pair_projection',-1.),('corner_projection',1.),('scale_projection',-1.)):
            args=dict(g,term=term)
            pv,pc=evaluate_comb_collar(bank,atlas,**args,orders=orders,diagnostics=True)
            parents.append((pv,pc,sign,term))
    else: raise ValueError('unsupported chart fallback')
    if not all(p[1]['passed'] for p in parents):
        raise FloatingPointError('unsafe parent in cross-chart residual')
    value=complex(math.fsum(sign*v[0].real for v,_,sign,_ in parents),
                  math.fsum(sign*v[0].imag for v,_,sign,_ in parents))
    absolute=math.fsum(abs(v[0]) for v,_,_,_ in parents)
    condition=absolute/abs(value) if value else math.inf
    errors=[]
    absolute_logs=[]
    for values, report, _, _ in parents:
        loss=report['prefixes'][0]['estimated_relative_arithmetic_loss']
        log_bound=report['prefixes'][0].get('absolute_arithmetic_log_bound')
        if loss is None:
            if log_bound is None:raise FloatingPointError('unbounded parent arithmetic')
            absolute_logs.append(float(log_bound));errors.append(math.exp(max(log_bound,-700.)))
        else:errors.append(abs(values[0])*max(float(loss),128*np.finfo(float).eps))
    absolute_enclosure=None
    if len(absolute_logs)==len(parents):
        peak=max(absolute_logs)
        absolute_enclosure=peak+math.log(math.fsum(math.exp(x-peak) for x in absolute_logs))
        if absolute_enclosure>=-745:raise FloatingPointError('unresolved cross-chart underflow')
        condition=0.;loss=0.
    else: loss=math.fsum(errors)/abs(value) if value else math.inf
    if condition>PAIR_REMAINDER_MAX_CONDITION or loss>PAIR_REMAINDER_NUMERICS_RTOL:
        raise FloatingPointError('ill-conditioned cross-chart residual')
    paired={}
    for values, _, sign, label in parents:
        actual_orders=other_orders if label=='alternate_full' else orders
        for i,(prefix,_) in enumerate(variants(actual_orders, validate=(other if label=='alternate_full' else bank).validate_orders)):
            if i: paired[label+':'+prefix]=(value+sign*(values[i]-values[0]),value)
    return dict(value=value,orders=[max(a,b) for a,b in zip(orders,other_orders)],
        source_sha256=selected.source_sha256,input_receipt_sha256=selected.receipt_sha256,
        safety=dict(chart_valid=True,coefficients_verified=True,arithmetic_safe=True,provenance_verified=True),
        paired_checks={'block':paired},arithmetic=dict(passed=True,parent_to_residual_condition=condition,
            estimated_relative_arithmetic_loss=loss),momentum_error_assessed=False,collar_error_assessed=False,
        cross_chart_subtraction=True,subtraction_before_momentum_sum=False,
        absolute_arithmetic_log_bound=absolute_enclosure,
        parent_orders=[list(other_orders),list(orders)],full_amplitude_certified=False,
        coefficient_recomputations=0,dozz_recomputations=0)


def _hash(sha):
    return isinstance(sha, str) and len(sha) == 64 and all(c in '0123456789abcdef' for c in sha)


def require_native(selection):
    bank = selection.bank
    if (type(bank) is not SharedNativeBank or bank.metadata.get('evaluation_scope', 'full_native_cft') != 'full_native_cft'
            or not _hash(selection.source_sha256) or not _hash(selection.receipt_sha256)
            or bank.metadata['source_manifest_sha256'] != selection.source_sha256):
        raise ValueError('verified full native CFT bank and receipts required')
    orders = tuple(bank.validate_orders(selection.orders))
    if any(a > b for a, b in zip(orders, bank.orders)):
        raise MissingCFTBank('requested tensor prefix is not stored')
    return bank, orders


def full_values(bank, atlas, geometry, orders, *, common_log_override=None, log_scale=0.):
    """Same full CFT as the existing evaluator, with scaled arithmetic guards."""
    if bank.metadata.get('projection_only',False):
        raise ValueError('a projection-only bank cannot represent the full CFT parent')
    kin = kinematics(atlas, **geometry, include_common=common_log_override is None)
    if common_log_override is not None:
        kin['common_log'] = common_log_override
    if identity(kin) != bank.metadata['identity']:
        raise ValueError('different graph/frequency in full CFT bank')
    base = (bank.arrays['structure_logs']+bank.arrays['log_weights']-3*math.log(math.pi)
            +kin['constant']+kin['common_log']+log_scale
            -np.einsum('ti,i->t', bank.arrays['momenta']**2, kin['decay']))
    if not np.all(np.isfinite(base)): raise FloatingPointError('nonfinite full-CFT primary density')
    values = []; checks = []
    for label, cut in variants(orders, validate=bank.validate_orders):
        block = block_logs(bank, cut, kin['logs'])
        value, check = _reduce(bank, dict(base=base), [block], [1],
                               1+sum(abs(z) for z in kin['logs']), cut)
        values.append(value); checks.append(dict(prefix=label, **check))
    return np.asarray(values), dict(passed=all(c['passed'] for c in checks), prefixes=checks)


def pair_projection_values(bank, atlas, geometry, orders, *, elliptic=None):
    """Unchanged pair projection, with explicit range/cancellation checks."""
    from genus1_three_point_channel_atlas import canonical_loop_displacement, nearest_torus_displacement
    from genus1_three_point_collision import pair_spectator, koba_exponent, _common_factor
    from genus1_three_point_worldsheet import torus_prime_form_log_norm
    g = geometry; pair = tuple(g['pair']); first, second = pair; spectator = pair_spectator(pair)
    tau, total, radius = complex(g['tau']), complex(g['delta_total']), float(g['radius'])
    integrated = g['integrated']
    if type(integrated) is not bool or not all(math.isfinite(x) for x in (tau.real, tau.imag, total.real, total.imag, radius)):
        raise ValueError('finite explicit pair projection geometry required')
    if tau.imag <= 0 or not 0 < radius < min(math.pi, abs(nearest_torus_displacement(total, tau))):
        raise ValueError('invalid isolated-pair collar')
    key = identity(dict(channel='pair_ope', coefficient_basis='native-sewing-v2',
        external_weights=tuple(atlas.external_weights[i] for i in (*pair, spectator)),
        external_momenta=tuple(atlas.external_momenta[i] for i in (*pair, spectator))))
    if key != bank.metadata['identity']: raise ValueError('projection graph/frequency mismatch')
    loop = canonical_loop_displacement(-total, tau)
    logs = (1j*loop, 1j*(2*math.pi*tau-loop))
    if any(not np.isfinite(z) or z.real >= 0 for z in logs): raise ValueError('noncontracting native loop')
    ps2 = bank.arrays['momenta']**2
    beta = 2*(1+ps2[:, 0])-2*(atlas.external_weights[first]+atlas.external_weights[second])+koba_exponent(atlas, first, second)+2
    if integrated and np.any(beta.real <= 0): raise ValueError('nonconvergent radial integral')
    base = (bank.arrays['structure_logs']+bank.arrays['log_weights']-3*math.log(math.pi)
        +cmath.log(_common_factor(atlas, tau))
        +(koba_exponent(atlas, first, spectator)+koba_exponent(atlas, second, spectator))*torus_prime_form_log_norm(total, tau)
        +2*ps2[:, 1]*logs[0].real+2*ps2[:, 2]*logs[1].real)
    base = base+(beta*math.log(radius)+math.log(2*math.pi)-np.log(beta.astype(complex))
                 if integrated else (beta-2)*math.log(radius))
    if not np.all(np.isfinite(base)): raise FloatingPointError('nonfinite pair-projection density')
    values = []; checks = []
    cuts=variants(orders, validate=bank.validate_orders) if elliptic is None else elliptic.cuts()
    for label, cut in cuts:
        if elliptic is None:
            cut = (0, cut[1], cut[2]);block=block_logs(bank,cut,(complex(-math.inf),*logs))
        else:
            block=elliptic.pair_logs(logs,*cut);cut=(0,*cut)
        value, check = _reduce(bank, dict(base=base), [block], [1],
                              1+abs(math.log(radius))+sum(abs(z) for z in logs), cut)
        values.append(value); checks.append(dict(prefix=label, **check))
    return np.asarray(values), dict(passed=all(c['passed'] for c in checks), prefixes=checks)


class BankBackedCFT:
    """Real CFT leaves for the sampling wrapper; no automatic bank creation.

The resolver must return Selection from immutable, checksum-verified inputs.
Catalogue coverage and source-manifest validation are performed by the caller.
The returned bookkeeping explicitly leaves spectral/radius audits unmeasured.
"""
    def __init__(self, atlas, resolver):
        if not callable(resolver): raise ValueError('explicit verified bank resolver required')
        self.atlas, self.resolver = atlas, resolver

    def evaluate(self, request):
        selected = self.resolver(request)
        if not isinstance(selected, Selection): raise MissingCFTBank(request)
        if selected.fallback_full is not None:
            return _linear_fallback(self.atlas,selected,request)
        kw = dict(request.geometry)
        if request.kind == 'height_integrated_collar':
            from genus1_integrated_scale_collar import PreparedBank, evaluate
            if (type(selected.bank) is not PreparedBank or not _hash(selected.source_sha256)
                    or not _hash(selected.receipt_sha256)
                    or selected.bank.identity['receipt_sha256'] != selected.receipt_sha256
                    or selected.bank._metadata['source_sha256'] != selected.source_sha256
                    or tuple(selected.orders) != (10, 0, 2)):
                raise ValueError('verified projected-height bank required')
            kw.pop('graph')
            values, checks = evaluate(selected.bank, self.atlas, **kw)
            report = dict(passed=all(c['passed'] for c in checks), prefixes=checks)
            orders = (10, 0, 2) if kw['term'] == 'triple_scale_face' else (0, 0, 2)
            labels = ('full', 'axis0_8', 'axis0_6', 'axis2_1', 'axis2_0')
        else:
            bank, orders = require_native(selected)
            if request.kind == 'full_reduced': values, report = full_values(bank, self.atlas, kw, orders)
            elif request.kind == 'pair_remainder': values, report = evaluate_pair_remainder(bank, self.atlas, **kw, orders=orders, diagnostics=True)
            elif request.kind == 'pair_projection':
                orders = (0, orders[1], orders[2])
                values, report = pair_projection_values(bank, self.atlas, kw, orders)
            elif request.kind == 'comb_collar':
                if kw['term'] == 'pair_face': orders = (0, orders[1], orders[2])
                values, report = evaluate_comb_collar(bank, self.atlas, **kw, orders=orders,
                    reversed_bank=selected.reversed_bank, diagnostics=True)
            else: raise ValueError('unknown CFT request')
            labels = tuple(label for label, _ in variants(orders, validate=bank.validate_orders))
        if (report['passed'] is not True or values.shape != (len(labels),)
                or not np.all(np.isfinite(values))):
            raise FloatingPointError('unsafe CFT arithmetic: '+str(report))
        # Prefixes along absent projected axes are not fictitious order tests.
        paired = {label: (values[i], values[0]) for i, label in enumerate(labels) if i
                  and orders[int(label[4])] > 0}
        return dict(value=complex(values[0]), orders=list(orders),
            source_sha256=selected.source_sha256, input_receipt_sha256=selected.receipt_sha256,
            safety=dict(chart_valid=True, coefficients_verified=True, arithmetic_safe=True, provenance_verified=True),
            paired_checks={'block': paired}, arithmetic=report,
            absolute_arithmetic_log_bound=report['prefixes'][0].get('absolute_arithmetic_log_bound'),
            momentum_error_assessed=False, collar_error_assessed=False,
            full_amplitude_certified=False, coefficient_recomputations=0, dozz_recomputations=0)


import json
from genus1_boundary_cubature import Value

class ArithmeticSafeCFT:
    def __init__(self,mpfix,atlas,resolver):
        self.mpfix,self.atlas,self.resolver=mpfix,atlas,resolver
        self.fast=BankBackedCFT(atlas,resolver);self.fallbacks=[]
    def evaluate(self,request):
        try:return self.fast.evaluate(request)
        except FloatingPointError as exc:
            if not str(exc).startswith('unsafe CFT arithmetic:'):raise
            selected=self.resolver(request)
            values,report,orders=self.mpfix.evaluate(self.atlas,selected,request)
            if not report['passed']:raise FloatingPointError('MP arithmetic unresolved: '+str(report))
            from genus1_native_bank import variants
            labels=tuple(label for label,_ in variants(orders))
            if len(values)!=len(labels) or not np.all(np.isfinite(values)):
                raise FloatingPointError('invalid MP retained prefixes')
            paired={label:(values[i],values[0]) for i,label in enumerate(labels) if i and orders[int(label[4])]>0}
            self.fallbacks.append(dict(kind=request.kind,input_receipt_sha256=selected.receipt_sha256,
                orders=list(orders),arithmetic=report,original_rejection=str(exc)))
            print(json.dumps(dict(stage='mp_point_integrated',kind=request.kind,orders=orders,seconds=report['seconds'])),flush=True)
            return dict(value=complex(values[0]),orders=list(orders),source_sha256=selected.source_sha256,
                input_receipt_sha256=selected.receipt_sha256,paired_checks={'block':paired},
                safety=dict(chart_valid=True,coefficients_verified=True,arithmetic_safe=True,provenance_verified=True),
                absolute_arithmetic_log_bound=report['prefixes'][0]['absolute_arithmetic_log_bound'],
                arithmetic=report,coefficient_recomputations=0,dozz_recomputations=0)

class AbsoluteCollision:
    def __init__(self,runtime,point_budget):
        if not math.isfinite(point_budget) or point_budget<=0:raise ValueError('positive amplitude-unit arithmetic budget required')
        self.runtime=runtime;self.point_budget=point_budget
        self.stats=dict(relative_guard_passed=0,absolute_guard_used=0,mp_fallbacks=0,
                        maximum_budget_fraction=0.,empirical_not_rigorous=True)

    def __call__(self,request,jacobian):
        if request.kind!='comb_collar' or request.geometry.get('term')!='pair_face':
            return self.runtime.collision(request)
        if not math.isfinite(jacobian) or jacobian<=0:raise ValueError('positive physical quadrature weight required')
        selected=self.runtime.providers.select(request);bank,orders=require_native(selected)
        orders=(0,orders[1],orders[2])
        values,report=evaluate_comb_collar(bank,self.runtime.atlas,**request.geometry,
            orders=orders,reversed_bank=selected.reversed_bank,diagnostics=True)
        prefixes=variants(orders)
        if len(values)!=len(prefixes):raise ValueError('collision prefix layout differs from its retained orders')
        bound_logs=[r.get('absolute_arithmetic_log_bound') for r in report['prefixes']]
        reasons={reason for r in report['prefixes'] for reason in r['reasons']}
        eligible={'estimated_arithmetic_loss_limit','cancellation_condition_limit','zero_or_subnormal_term'}
        valid=(np.all(np.isfinite(values)) and len(values)==len(bound_logs) and len(values)>0
               and all(v is not None and math.isfinite(v) for v in bound_logs) and reasons<=eligible)
        factor=jacobian*abs(request.multiplicity)/(4*math.pi)
        # A prefix difference contains two arithmetic errors. Reserve both.
        if valid and math.log(2*factor)+max(bound_logs)<=math.log(self.point_budget):
            error=math.exp(max(max(bound_logs),math.log(np.nextafter(0.,1.))))
            value=Value(values[0],dict(block=float(max(abs(values[0]-values[1:]),default=0.)),
                                     momentum=None,arithmetic=error,endpoint=None),
                        {'native:'+label:values[i] for i,(label,_) in enumerate(prefixes)
                         if i and orders[int(label[4])]>0})
            key='relative_guard_passed' if report['passed'] else 'absolute_guard_used'
            self.stats[key]+=1
        else:
            self.stats['mp_fallbacks']+=1
            value=self.runtime.collision(request)
            error=value.errors.get('arithmetic')
            if error is None or not math.isfinite(error) or 2*factor*error>self.point_budget:
                raise FloatingPointError('collision arithmetic exceeds its weighted absolute allocation')
        self.stats['maximum_budget_fraction']=max(self.stats['maximum_budget_fraction'],2*factor*error/self.point_budget)
        return value
