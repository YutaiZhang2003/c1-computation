"""Analytic leading pair-collar projection using saved native-q banks.

Retains the established collision subtraction, dP/pi measure and normalization.
The radial power is integrated before the momentum quadrature, with its exact
momentum-dependent denominator. The isolated-pair remainder below is a LOCAL
prototype with fresh residual and cancellation diagnostics, not a regional or
production certificate. Triple forest and scale/corner addbacks are absent.
"""
import cmath
import math

import numpy as np

from genus1_native_bank import identity, variants
from genus1_rebuild_policy import require_orders


def evaluate_pair_collar(bank, atlas, *, pair, delta_total, tau, radius,
                         integrated=False, orders=None):
    """Seven full/prefix values of the projection or its radial disc integral."""
    from genus1_three_point_channel_atlas import canonical_loop_displacement, nearest_torus_displacement
    from genus1_three_point_collision import pair_spectator, koba_exponent, _common_factor
    from genus1_three_point_worldsheet import torus_prime_form_log_norm
    pair=tuple(pair); first,second=pair; spectator=pair_spectator(pair)
    requested=bank.orders if orders is None else tuple(require_orders(orders))
    if any(a>b for a,b in zip(requested,bank.orders)):
        raise ValueError('missing collar coefficients; refine only required axes within cap 10')
    if type(integrated) is not bool:
        raise ValueError('explicit boolean radial-integration choice required')
    tau,delta_total=complex(tau),complex(delta_total); radius=float(radius)
    if not all(math.isfinite(z) for z in (tau.real,tau.imag,delta_total.real,delta_total.imag,radius)):
        raise ValueError('finite collar geometry required')
    if tau.imag<=0 or not 0<radius<min(math.pi,abs(nearest_torus_displacement(delta_total,tau))):
        raise ValueError('positive isolated collar radius required')
    key=identity(dict(channel='pair_ope',coefficient_basis='native-sewing-v2',
        external_weights=tuple(atlas.external_weights[i] for i in (*pair,spectator)),
        external_momenta=tuple(atlas.external_momenta[i] for i in (*pair,spectator))))
    if key!=bank.metadata['identity']:
        raise ValueError('different CFT graph or external frequency in collar bank')
    loop=canonical_loop_displacement(-delta_total,tau)
    logs=(1j*loop,1j*(2*math.pi*tau-loop))
    if any(z.real>=0 for z in logs):
        raise ValueError('pair projection requires contracting native loop coordinates')
    q=tuple(cmath.exp(z) for z in logs)
    ps2=bank.arrays['momenta']**2
    beta=(2*(1+ps2[:,0])-2*(atlas.external_weights[first]+atlas.external_weights[second])
          +koba_exponent(atlas,first,second)+2)
    if integrated and np.any(np.real(beta)<=0):
        raise ValueError('radial pair integral is not convergent on these momentum nodes')
    common=cmath.log(_common_factor(atlas,tau))
    spectator_log=(koba_exponent(atlas,first,spectator)+koba_exponent(atlas,second,spectator))
    spectator_log*=torus_prime_form_log_norm(delta_total,tau)
    base=(bank.arrays['structure_logs']+bank.arrays['log_weights']-3*math.log(math.pi)
          +common+spectator_log+2*ps2[:,1]*logs[0].real+2*ps2[:,2]*logs[1].real)
    if integrated:
        base=base+beta*math.log(radius)+math.log(2*math.pi)-np.log(beta.astype(complex))
    else:
        base=base+(beta-2)*math.log(radius)
    result=[]
    for _,cut in variants(requested):
        block=np.einsum('tjk,j,k->t',bank.arrays['coefficients'][:,0,:cut[1]+1,:cut[2]+1],
            q[0]**np.arange(cut[1]+1),q[1]**np.arange(cut[2]+1),optimize=False)
        if not np.all(np.isfinite(block)):
            raise FloatingPointError('nonfinite pair-collar native-q block')
        nonzero=block!=0
        logvalues=base[nonzero]+2*np.log(abs(block[nonzero]))
        if not np.all(np.isfinite(logvalues)) or np.any(logvalues.real>math.log(np.finfo(float).max)):
            raise FloatingPointError('nonfinite or overflowing collar density')
        values=np.exp(logvalues)
        result.append(complex(math.fsum(values.real),math.fsum(values.imag)))
    return np.asarray(result,complex)


class PairCollarEnsemble:
    """Fresh collar-specific refinement evidence, never inherited from full CFT."""
    def __init__(self, full_ensemble):
        self.ensemble=full_ensemble

    def audit(self, atlas, *, pair, delta_total, tau, radius, integrated=False):
        from genus1_rebuild_policy import REFINEMENT_RTOL, classify_cft_point, encoded
        values=np.asarray([evaluate_pair_collar(bank,atlas,pair=pair,delta_total=delta_total,
            tau=tau,radius=radius,integrated=integrated) for bank in self.ensemble.banks])
        if not np.all(np.isfinite(values)) or np.any(abs(values[:,0])<np.finfo(float).tiny):
            raise FloatingPointError('nonfinite/underflowed collar cannot certify refinement')
        changes=[abs(values[1]-values[0])/np.maximum(abs(values[1]),1e-300),
                 abs(values[2]-values[1])/np.maximum(abs(values[2]),1e-300),
                 abs(values[3]-values[2])/np.maximum(abs(values[2]),1e-300)]
        full=values[2,0];axis=[]
        for i in range(3):
            first,second=values[2,1+2*i:3+2*i]
            axis.append([float(abs(full-first)/abs(full)),float(abs(first-second)/abs(full))])
        c=classify_cft_point(orders=self.ensemble.orders,axis_changes=axis,
            chart_admissible=True,coefficients_verified=True,
            momentum_converged=bool(all(np.all(x<=REFINEMENT_RTOL) for x in changes)))
        return dict(term='pair_projection_disc' if integrated else 'pair_projection',
            pair=list(pair),delta_total=encoded(delta_total),tau=encoded(tau),radius=radius,
            classification=c,value=encoded(full),
            prefixes={label:encoded(z) for (label,_),z in zip(variants(self.ensemble.orders),values[2])},
            maximum_momentum_changes=[float(max(x)) for x in changes],
            momentum_full_and_prefix_changes=[x.tolist() for x in changes],
            amplitude_contribution=None,target_blind=True,full_amplitude_certified=False,
            qualification='Accuracy checks for this leading projection only. The full amplitude also requires exact-minus-projection remainders and all forest/addback terms.')

    def require(self, atlas, **geometry):
        from genus1_native_bank import UnresolvedSharedBankPoint
        result=self.audit(atlas,**geometry)
        if not result['classification']['include_in_amplitude']:
            raise UnresolvedSharedBankPoint(result)
        return result

    def weighted_record(self, atlas, *, weight, point_id, **geometry):
        from genus1_rebuild_policy import weighted_cft_record
        result=self.require(atlas,**geometry);c=result['classification']
        return weighted_cft_record(c,value=complex(*result['value']),weight=weight,
            cap_prefix_values={str(i):complex(*result['prefixes'][f'axis{i}_8']) for i in c['capped_axes']},
            point_id=point_id)


# Fixed diagnostic limits, not fitted tolerances or rigorous error bounds.
# Reserve a tenth of the refinement tolerance for estimated arithmetic loss.
PAIR_REMAINDER_NUMERICS_RTOL = 5e-4
PAIR_REMAINDER_MAX_CONDITION = 1e8
_ROUNDING_SAFETY = 128.
_FLOAT = np.finfo(float)


class PairRemainderCancellationError(FloatingPointError):
    """A computed residual cannot resolve the requested relative accuracy."""
    def __init__(self, diagnostic):
        super().__init__('isolated pair residual exceeds cancellation diagnostic limits')
        self.diagnostic = diagnostic


def _log_positive(values):
    values = np.asarray(values, float)
    result = np.full(values.shape, -np.inf)
    positive = values > 0
    result[positive] = np.log(values[positive])
    return result


def _logsum(values):
    """Positive sum in log space, including an empty or identically zero sum."""
    values = np.asarray(values, float)
    if np.any(np.isnan(values)) or np.any(np.isposinf(values)):
        raise FloatingPointError('nonfinite logarithmic diagnostic scale')
    finite = values[np.isfinite(values)]
    if not len(finite):
        return -math.inf
    largest = float(max(finite))
    return largest + math.log(math.fsum(np.exp(finite-largest)))


def _scaled_node_difference(common, exact, projection):
    """Subtract exp(common+exact)-exp(common+projection) BEFORE any sum.

    Local logarithms exclude the shared DOZZ, measure, loop and tensor scale.
    A zero term is represented ONLY by (-inf+0j). No parent exponential is
    formed; even individually overflowing parents can have a finite residual.
    Returns node values and their pre-exponentiation log magnitudes, so lost
    subnormal nodes can be charged to the arithmetic diagnostic.
    """
    common, exact, projection = np.broadcast_arrays(
        np.asarray(common, complex), np.asarray(exact, complex),
        np.asarray(projection, complex))
    if common.ndim != 1 or not np.all(np.isfinite(common)):
        raise FloatingPointError('finite one-dimensional common node logs required')
    for term in (exact, projection):
        valid = np.isfinite(term) | (np.isneginf(term.real) & (term.imag == 0))
        if not np.all(valid):
            raise FloatingPointError('invalid exact/projection node logs')
    e, p = np.isfinite(exact.real), np.isfinite(projection.real)
    logs = np.full(common.shape, complex(-math.inf))
    sign = np.ones(common.shape)
    logs[e & ~p] = common[e & ~p] + exact[e & ~p]
    logs[p & ~e] = common[p & ~e] + projection[p & ~e]
    sign[p & ~e] = -1.
    for mask, first, second, orientation in (
            (e & p & (exact.real >= projection.real), exact, projection, -1.),
            (e & p & (exact.real < projection.real), projection, exact, 1.)):
        indices = np.flatnonzero(mask)
        difference = orientation * np.expm1(second[mask]-first[mask])
        nonzero = difference != 0
        indices = indices[nonzero]
        logs[indices] = common[indices] + first[indices] + np.log(difference[nonzero])
    active = np.isfinite(logs.real)
    if (np.any(np.isnan(logs)) or np.any(np.isposinf(logs.real)) or
            np.any(logs.real[active] > math.log(_FLOAT.max))):
        raise FloatingPointError('pair residual node overflow/nonfinite value')
    values = np.zeros(common.shape, complex)
    with np.errstate(under='ignore'):
        values[active] = sign[active] * np.exp(logs[active])
    if not np.all(np.isfinite(values)):
        raise FloatingPointError('nonfinite pair residual nodes')
    return values, logs.real


def _pair_remainder_geometry(bank, atlas, pair, delta_pair, delta_total, tau, radii):
    from genus1_native_bank import FORMAT
    if (bank.metadata.get('format') != FORMAT or
            bank.metadata.get('evaluation_scope','full_native_cft') != 'full_native_cft'):
        raise ValueError('isolated pair remainder requires a full native CFT bank, not a projection-only slice')
    from genus1_three_point_channel_atlas import (
        canonical_loop_displacement, nearest_torus_displacement,
        pair_disc_to_flat_log_factor)
    from genus1_three_point_collision import (
        CollisionRadii, pair_spectator, reconstruct_points, koba_exponent,
        _common_factor)
    from genus1_three_point_worldsheet import torus_prime_form_log_norm

    pair = tuple(pair)
    spectator = pair_spectator(pair)
    first, second = pair
    tau, delta, total = complex(tau), complex(delta_pair), complex(delta_total)
    if not all(math.isfinite(x) for z in (tau, delta, total) for x in (z.real, z.imag)):
        raise ValueError('finite isolated-pair geometry required')
    if not isinstance(radii, CollisionRadii):
        raise ValueError('explicit CollisionRadii required for isolated-pair membership')
    CollisionRadii(radii.pair, radii.triple)
    # The existing pi injectivity bound is valid in this modular domain.
    if tau.imag <= 0 or abs(tau.real) > .5 or abs(tau) < 1.-8*_FLOAT.eps:
        raise ValueError('local prototype requires tau in the modular fundamental domain')
    radius = abs(delta)
    if not 0 < radius < radii.pair:
        raise ValueError('pair displacement must lie strictly inside the nonzero collar')
    if abs(nearest_torus_displacement(total, tau)) < radii.triple:
        raise ValueError('triple region excluded from the isolated-pair remainder')
    if abs(nearest_torus_displacement(delta, tau)-delta) > 32*_FLOAT.eps*radius:
        raise ValueError('pair displacement is not a local torus representative')
    external = tuple(atlas.external_weights[i] for i in (*pair, spectator))
    if any(complex(d).imag != 0 for d in external):
        raise ValueError('this pair projection prototype requires real external weights')
    kin = dict(channel='pair_ope', coefficient_basis='native-sewing-v2',
        external_weights=external,
        external_momenta=tuple(atlas.external_momenta[i] for i in (*pair, spectator)))
    if identity(kin) != bank.metadata['identity']:
        raise ValueError('different CFT graph or external frequency in remainder bank')
    w1, w2 = reconstruct_points(pair, delta, total)
    atlas.sewing_chart(w1, w2, tau, 'pair_ope', pair).require()
    loop = canonical_loop_displacement(-total, tau)
    logs = (1j*loop, 1j*(2*math.pi*tau-loop))
    if any(not np.isfinite(z) or z.real >= 0 for z in logs):
        raise ValueError('pair projection requires contracting native loop coordinates')
    # expm1 preserves the small local coordinate without changing its basis.
    v = complex(np.expm1(-1j*delta))
    flat = pair_disc_to_flat_log_factor(delta, external[0], external[1])
    kpair = koba_exponent(atlas, first, second)
    kfirst = koba_exponent(atlas, first, spectator)
    ksecond = koba_exponent(atlas, second, spectator)
    total_norm = torus_prime_form_log_norm(total, tau)
    exact_koba_log = (kpair*torus_prime_form_log_norm(delta, tau)
        + kfirst*torus_prime_form_log_norm(delta-total, tau) + ksecond*total_norm)
    spectator_log = (kfirst+ksecond)*total_norm
    ps2 = bank.arrays['momenta']**2
    common = (bank.arrays['structure_logs'] + bank.arrays['log_weights']
        - 3*math.log(math.pi) + cmath.log(_common_factor(atlas, tau))
        + 2*ps2[:, 1]*logs[0].real + 2*ps2[:, 2]*logs[1].real)
    beta = 2*(1+ps2[:, 0])-2*(external[0]+external[1])+kpair+2
    exact = 2*(1+ps2[:, 0])*math.log(abs(v))+2*flat.real+exact_koba_log
    projection = (beta-2)*math.log(radius)+spectator_log
    if not all(np.all(np.isfinite(x)) for x in (common, exact, projection)):
        raise FloatingPointError('nonfinite pair remainder primary logs')
    # Conservative loss indicator for reconstruction/wrapping and small v.
    # It intentionally disqualifies unresolved extreme collars, even if the
    # stable direct-delta calculation happens to return matching small values.
    geometry_scale = max(1., abs(w1), abs(w2), abs(total), 2*math.pi*abs(tau))/radius
    return common, exact, projection, (cmath.log(v), *logs), geometry_scale


def _pair_remainder_block_logs(bank, variables, cut):
    """Scale the tensor per node before contracting; never square huge blocks."""
    coefficients = bank.arrays['coefficients'][(slice(None), *[slice(n+1) for n in cut])]
    scale = np.max(abs(coefficients), axis=(1, 2, 3))
    if not np.all(np.isfinite(scale)):
        raise FloatingPointError('nonfinite pair tensor magnitude')
    scale = np.where(scale > 0, scale, 1.)
    normalized = coefficients/scale[:, None, None, None]
    powers = [z**np.arange(n+1) for z, n in zip(variables, cut)]
    # Unlike final density underflow (accounted for below), a lost tensor
    # entry/power could be revived by the external exponential. Fail closed
    # when these intermediate products exceed this float64 prototype's range.
    if (np.any((coefficients != 0) & (abs(normalized) < _FLOAT.tiny)) or
            any(z != 0 and np.any(abs(p) < _FLOAT.tiny) for z, p in zip(variables, powers))):
        raise FloatingPointError('subnormal tensor normalization/sewing power exceeds prototype range')
    zero = np.einsum('tjk,j,k->t', normalized[:, 0], *powers[1:], optimize=False)
    extra = np.einsum('tijk,i,j,k->t', normalized[:, 1:], powers[0][1:],
                      *powers[1:], optimize=False)
    exact = zero+extra
    absolute = np.einsum('tijk,i,j,k->t', abs(normalized),
                         *[abs(p) for p in powers], optimize=False)
    absolute_zero = np.einsum('tjk,j,k->t', abs(normalized[:, 0]),
                              *[abs(p) for p in powers[1:]], optimize=False)
    if not all(np.all(np.isfinite(x)) for x in (scale, exact, zero, absolute, absolute_zero)):
        raise FloatingPointError('nonfinite scaled pair block contraction')
    if (np.any((absolute < _FLOAT.tiny) & np.any(normalized != 0, axis=(1, 2, 3))) or
            np.any((absolute_zero < _FLOAT.tiny) & np.any(normalized[:, 0] != 0, axis=(1, 2)))):
        raise FloatingPointError('subnormal absolute contraction exceeds prototype range')
    return (2*np.log(scale), 2*_log_positive(abs(exact)),
        2*_log_positive(abs(zero)), 2*_log_positive(absolute),
        2*_log_positive(absolute_zero))


def evaluate_pair_remainder(bank, atlas, *, pair, delta_pair, delta_total, tau,
                            radii, orders=None, diagnostics=False):
    """Seven SAME-NODE isolated-pair residuals, with no position/MC weights.

    Default evaluation raises on unresolved cancellation. diagnostics=True
    returns (values, diagnostic) even on arithmetic-limit failure; those raw
    values are NOT accepted contributions. This escape is for audit only.
    Stored coefficient precision and termwise contraction scales feed an
    empirical loss indicator, not a rigorous CFT error estimate.
    """
    if type(diagnostics) is not bool:
        raise ValueError('explicit boolean diagnostics choice required')
    requested = tuple(require_orders(bank.orders if orders is None else orders))
    if any(a > b for a, b in zip(requested, bank.orders)):
        raise ValueError('missing remainder coefficients; no runtime bank extension')
    common, exact, projection, sewing_logs, geometry_scale = _pair_remainder_geometry(
        bank, atlas, pair, delta_pair, delta_total, tau, radii)
    answers, rows = [], []
    for label, cut in variants(requested):
        from genus1_native_log_evaluation import block_logs
        if all(n*z.real > -650 for n,z in zip(cut,sewing_logs)):
            scale, block, zero, absolute, absolute_zero = _pair_remainder_block_logs(
                bank, tuple(cmath.exp(z) for z in sewing_logs), cut)
        else:
            block, absolute = block_logs(bank,cut,sewing_logs)
            zero, absolute_zero = block_logs(bank,(0,cut[1],cut[2]),
                (complex(-math.inf),*sewing_logs[1:]))
            scale = np.zeros(len(block))
        base = common+scale
        nodes, node_logs = _scaled_node_difference(base, exact+block, projection+zero)
        value = complex(math.fsum(nodes.real), math.fsum(nodes.imag))
        if not np.isfinite(value):
            raise FloatingPointError('nonfinite summed pair residual')
        magnitude = abs(value)
        log_value = math.log(magnitude) if magnitude else -math.inf
        parent_log = _logsum(np.logaddexp(base.real+exact.real+block,
                                         base.real+projection.real+zero))
        absolute_enclosure=_logsum(np.logaddexp(base.real+exact.real+absolute,
            base.real+projection.real+absolute_zero))+math.log(2.)
        if math.isfinite(absolute_enclosure) and absolute_enclosure < -745:
            rows.append(dict(prefix=label,orders=list(cut),passed=True,reasons=[],
                parent_to_residual_condition=None,estimated_relative_arithmetic_loss=None,
                absolute_arithmetic_log_bound=float(absolute_enclosure),
                range_handling='absolute_retained_density_enclosure'))
            answers.append(0j);continue
        # Absolute coefficient contractions expose cancellation INSIDE blocks.
        # The saved precision check is empirical, so this is explicitly only
        # a diagnostic estimate. It cannot establish a rigorous error bound.
        budget = (_ROUNDING_SAFETY*_FLOAT.eps*(math.prod(n+1 for n in cut)
            + geometry_scale + 1 + abs(base) + abs(exact) + abs(projection))
            + 4*bank.arrays['coefficient_precision_bounds'])
        error_logs = (np.logaddexp(base.real+exact.real+absolute,
            base.real+projection.real+absolute_zero) + np.log(budget))
        lost = node_logs[np.abs(nodes) < _FLOAT.tiny]
        error_log = float(np.logaddexp(_logsum(error_logs), _logsum(lost)))
        resolvable = magnitude >= _FLOAT.tiny
        condition_log = max(0., parent_log-log_value) if resolvable else math.inf
        relative_log = error_log-log_value if resolvable else math.inf
        def finite_ratio(log_ratio):
            return math.exp(log_ratio) if log_ratio <= math.log(_FLOAT.max) else None
        reasons = []
        if not resolvable:
            reasons.append('zero_or_subnormal_residual')
        if condition_log > math.log(PAIR_REMAINDER_MAX_CONDITION):
            reasons.append('parent_to_residual_condition_limit')
        if relative_log > math.log(PAIR_REMAINDER_NUMERICS_RTOL):
            reasons.append('estimated_arithmetic_loss_limit')
        rows.append(dict(prefix=label, orders=list(cut), passed=not reasons, reasons=reasons,
            parent_to_residual_condition=finite_ratio(condition_log),
            estimated_relative_arithmetic_loss=finite_ratio(relative_log),
            subnormal_or_underflowed_nodes=int(np.count_nonzero(
                np.isfinite(node_logs) & (np.abs(nodes) < _FLOAT.tiny)))))
        answers.append(value)
    result = np.asarray(answers, complex)
    report = dict(passed=all(row['passed'] for row in rows), prefixes=rows,
        maximum_parent_to_residual_condition=PAIR_REMAINDER_MAX_CONDITION,
        maximum_estimated_relative_arithmetic_loss=PAIR_REMAINDER_NUMERICS_RTOL,
        geometry_condition_indicator=float(geometry_scale),
        qualification='Empirical cancellation/arithmetic guard only; no rigorous error bound.',
        subtraction_before_momentum_sum=True, coefficient_recomputations=0,
        dozz_recomputations=0, target_blind=True, production_ready=False,
        full_amplitude_certified=False)
    if diagnostics:
        return result, report
    if not report['passed']:
        raise PairRemainderCancellationError(report)
    return result


class PairRemainderEnsemble:
    """Fresh residual refinement; no acceptance inherited from either parent."""
    def __init__(self, full_ensemble):
        from genus1_native_bank import SharedBankEnsemble
        self.ensemble = SharedBankEnsemble(full_ensemble.banks)

    def audit(self, atlas, *, pair, delta_pair, delta_total, tau, radii):
        from genus1_rebuild_policy import REFINEMENT_RTOL, classify_cft_point, encoded
        evaluated = [evaluate_pair_remainder(bank, atlas, pair=pair, delta_pair=delta_pair,
            delta_total=delta_total, tau=tau, radii=radii, diagnostics=True)
            for bank in self.ensemble.banks]
        values = np.asarray([row[0] for row in evaluated])
        if values.shape != (4, 7) or not np.all(np.isfinite(values)):
            raise FloatingPointError('invalid residual full/prefix ensemble values')
        arithmetic = all(row[1]['passed'] for row in evaluated)
        def relative(a, b, denominator):
            with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
                result = abs(a-b)/np.maximum(abs(denominator), _FLOAT.tiny)
            # Fail closed with a finite diagnostic understood by the policy.
            return np.minimum(np.nan_to_num(result, nan=_FLOAT.max, posinf=_FLOAT.max), _FLOAT.max)
        changes = [relative(values[1], values[0], values[1]),
                   relative(values[2], values[1], values[2]),
                   relative(values[3], values[2], values[2])]
        agreement = bool(all(np.all(c <= REFINEMENT_RTOL) for c in changes))
        full = values[2, 0]
        axis = [[float(relative(full, values[2, 1+2*i], full)),
                 float(relative(values[2, 1+2*i], values[2, 2+2*i], full))] for i in range(3)]
        # Agreement cannot establish spectral accuracy when the evaluated
        # residual is below its arithmetic resolution. Keep both facts visible.
        classification = classify_cft_point(orders=self.ensemble.orders, axis_changes=axis,
            chart_admissible=True, coefficients_verified=True,
            momentum_converged=bool(agreement and arithmetic))
        return dict(term='isolated_pair_remainder', pair=list(pair),
            delta_pair=encoded(delta_pair), delta_total=encoded(delta_total), tau=encoded(tau),
            radii=dict(pair=radii.pair, triple=radii.triple), value=encoded(full),
            classification=classification, momentum_node_orders=self.ensemble.node_orders,
            prefixes={label:encoded(z) for (label, _), z in zip(variants(self.ensemble.orders), values[2])},
            momentum_full_and_prefix_relative_changes=[c.tolist() for c in changes],
            maximum_momentum_relative_changes=[float(max(c)) for c in changes],
            residual_momentum_agreement=agreement, residual_arithmetic_passed=arithmetic,
            cancellation_diagnostics=[row[1] for row in evaluated],
            coefficient_recomputations=0, dozz_recomputations=0, amplitude_contribution=None,
            target_blind=True, production_ready=False, full_amplitude_certified=False,
            qualification='Local isolated-pair prototype. Fresh residual checks only; no regional, triple-forest or full-amplitude certificate.')

    def require(self, atlas, **geometry):
        from genus1_native_bank import UnresolvedSharedBankPoint
        result = self.audit(atlas, **geometry)
        if not result['classification']['include_in_amplitude']:
            raise UnresolvedSharedBankPoint(result)
        return result

    def weighted_record(self, atlas, *, weight, point_id, **geometry):
        from genus1_rebuild_policy import weighted_cft_record
        result = self.require(atlas, **geometry)
        c = result['classification']
        return weighted_cft_record(c, value=complex(*result['value']), weight=weight,
            cap_prefix_values={str(i):complex(*result['prefixes'][f'axis{i}_8']) for i in c['capped_axes']},
            point_id=point_id)
