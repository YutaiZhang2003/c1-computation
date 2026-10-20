"""Momentum designs for the evaluated projection, including its geometry."""
from __future__ import annotations

# genus1_collision_momentum
# Momentum proposals for analytically integrated collision projections.
# 
# These rules undo their reference weight: callers still evaluate the complete
# DOZZ product, radial kernel and conformal block, with the usual dP measure.
# Selecting a rule is not a convergence certificate or a change of estimator.
# Existing frozen campaigns deliberately do not select these rules implicitly.



import math
from numbers import Integral

import mpmath as mp
import numpy as np
from scipy.linalg import eigh_tridiagonal


def pair_mass_squared(atlas, pair):
    """Return mu^2 in beta(P) = 2(P^2 + mu^2), including the matter power."""
    from genus1_three_point_collision import koba_exponent, pair_spectator
    pair_spectator(pair)  # Validate the external labels before indexing.
    first, second = pair
    value = complex(2-atlas.external_weights[first]-atlas.external_weights[second]
                    + koba_exponent(atlas, first, second)/2)
    if value.imag != 0 or not math.isfinite(value.real) or value.real < 0:
        raise ValueError('a convergent real collision bridge is required')
    return value.real


def radial_bridge_rule(order, *, decay, mass_squared):
    r"""Gaussian rule matched to P^2 exp(-a P^2)/(P^2+mu^2) on [0,infinity).

    With x=a P^2 and b=a mu^2, the reference measure is
    x^(1/2) exp(-x)/(x+b) dx/(2 sqrt(a)). Its zeroth moment is
    I0 = sqrt(pi) - pi sqrt(b) exp(b) erfc(sqrt(b)).

    Dividing the Laguerre measure by x+b gives monic polynomials
    Q_n=L_n+r_n L_(n-1). Their ratios obey
    r_1=3/2+b-Gamma(3/2)/I0 and
    r_(n+1)=2n+3/2+b-n(n+1/2)/r_n.
    The resulting Jacobi matrix supplies positive nodes and weights. Compute
    the minimal-solution recurrence at extra precision before diagonalizing;
    it must not be evaluated by a forward float64 recurrence.

    At mu=0 this is exactly the alpha=-1/2 Gaussian momentum rule. Returned
    weights are for dP, NOT dP/pi. The denominator is a proposal only and is
    undone along with the Gaussian; do not divide the physical integrand by
    it again. No finite momentum cutoff is introduced.
    """
    if isinstance(order, bool) or not isinstance(order, Integral) or order < 1:
        raise ValueError('a positive integer momentum order is required')
    decay, mass_squared = float(decay), float(mass_squared)
    if not math.isfinite(decay) or decay <= 0:
        raise ValueError('a positive finite decay is required')
    if not math.isfinite(mass_squared) or mass_squared < 0:
        raise ValueError('a nonnegative finite squared mass is required')
    order = int(order)
    b_float = decay*mass_squared
    if not math.isfinite(b_float):
        raise ValueError('nonfinite dimensionless collision mass')
    # The Cauchy-transform recurrence loses roughly O(sqrt(n*b)) digits.
    dps = 80 + math.ceil(4*math.sqrt(order*b_float)/math.log(10))
    with mp.workdps(dps):
        b = mp.mpf(decay)*mp.mpf(mass_squared)
        mu0 = mp.sqrt(mp.pi)/2
        i0 = mp.sqrt(mp.pi)-mp.pi*mp.sqrt(b)*mp.exp(b)*mp.erfc(mp.sqrt(b))
        if not i0 > 0:
            raise ArithmeticError('unresolved collision reference moment')
        ratios = [mp.mpf(0), mp.mpf('1.5')+b-mu0/i0]
        for n in range(1, order):
            ratios.append(2*n+mp.mpf('1.5')+b-n*(n+mp.mpf('.5'))/ratios[n])
        diagonal = [mu0/i0-b]
        diagonal.extend(2*n+mp.mpf('1.5')+ratios[n]-ratios[n+1]
                        for n in range(1, order))
        off_squared = [ratios[1]*mu0/i0] if order > 1 else []
        off_squared.extend((n-1)*(n-mp.mpf('.5'))*ratios[n]/ratios[n-1]
                           for n in range(2, order))
        if any(v <= 0 or not mp.isfinite(v) for v in (*ratios[1:], *off_squared)):
            raise ArithmeticError('unresolved collision quadrature recurrence')
        diagonal = np.asarray([float(v) for v in diagonal])
        off_diagonal = np.asarray([float(mp.sqrt(v)) for v in off_squared])
        reference_mass = float(i0)
    # stev retains the small first-row eigenvector entries used by the
    # exponentially reweighted high-momentum nodes.
    x, vectors = eigh_tridiagonal(diagonal, off_diagonal, lapack_driver='stev')
    spectral_weights = reference_mass*vectors[0]**2
    if np.any(x <= 0) or np.any(spectral_weights <= 0):
        raise ArithmeticError('nonpositive collision quadrature nodes or weights')
    log_weights = (np.log(spectral_weights)+x+np.log(x+b_float)-np.log(x)
                   -math.log(2*math.sqrt(decay)))
    weights = np.exp(log_weights)
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ArithmeticError('nonfinite collision quadrature weights')
    return dict(nodes=np.sqrt(x/decay).tolist(), weights=weights.tolist(),
                log_weights=log_weights.tolist(), order=order, decay=decay,
                mass_squared=mass_squared, scheme='integrated-radial-bridge-v1',
                measure='dP', domain='[0,infinity)',
                reference_integral=reference_mass/(2*math.sqrt(decay)))


def collision_rules(reference, orders, *, mass_squared, decay_scales=(1., 1., 1.)):
    """Choose each axis independently; adapt only the collapsed pair bridge.

    The other axes retain their reference endpoint powers, notably alpha=-1/2
    for the integrated triple-scale momentum and alpha=0 for a height-
    integrated infinite-tail loop. This does not guess an endpoint power from
    the number of edges in a graph.
    """
    from liouville_momentum_quadrature import generalized_gaussian_momentum_rule
    if len(reference) != 3 or len(orders) != 3 or len(decay_scales) != 3:
        raise ValueError('three explicit momentum axes are required')
    if any(isinstance(n, bool) or not isinstance(n, Integral) or n < 1 for n in orders):
        raise ValueError('positive integer axis orders are required')
    if any(not math.isfinite(s) or s <= 0 for s in decay_scales):
        raise ValueError('positive finite decay scales are required')
    result = [radial_bridge_rule(orders[0], decay=reference[0]['decay']*decay_scales[0],
                                mass_squared=mass_squared)]
    for ref, n, scale in zip(reference[1:], orders[1:], decay_scales[1:]):
        if ref.get('measure') != 'dP':
            raise ValueError('reference rules must use dP weights')
        decay = ref['decay']*scale
        rule = generalized_gaussian_momentum_rule(.1, int(n),
            laguerre_alpha=ref['alpha'], decay_coefficient=decay)
        weights = [math.pi*w for w in rule.weights]
        result.append(dict(nodes=list(rule.nodes), weights=weights,
            log_weights=[math.log(w) for w in weights], order=int(n),
            alpha=ref['alpha'], decay=decay, measure='dP', domain='[0,infinity)',
            scheme='generalized-gaussian'))
    if reference[0].get('measure') != 'dP':
        raise ValueError('reference rules must use dP weights')
    return result


# genus1_collision_thresholds
# Explicit, geometry-dependent threshold designs for collision spectra.
# 
# The endpoint power belongs to an evaluated kernel, not to a saved graph.
# In particular, a pair-cusp height integral changes its long-loop power from
# P^2 to P. These opt-in designs do not modify frozen production recipes.
# All constructed rules return dP weights and retain the full positive axis.



import math
from numbers import Integral

import numpy as np

from genus1_three_point_collision import pair_spectator
from genus1_three_point_channel_atlas import canonical_loop_displacement
from liouville_momentum_quadrature import generalized_gaussian_momentum_rule

FORMAT = 'genus1-collision-threshold-design-v1'


def _positive(value, label):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError('positive finite '+label+' required')
    return value


def _order(value):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError('positive integer quadrature order required')
    return int(value)


def endpoint_panels(*, decay, threshold_scale, core_order, tail_order,
                    split_scale=1., width_scale=1.):
    """A linear P core followed by a logarithmic map to infinity.

    The core ends at the smaller of the kernel's endpoint transition and
    its Gaussian width, times an explicit split control. For a height
    kernel these are 1/sqrt(4*pi*L) and 1/sqrt(4*pi*(L-shift)), respectively.
    A quadrature node is never a physical upper momentum cutoff.
    """
    decay = _positive(decay, 'decay')
    threshold_scale = _positive(threshold_scale, 'endpoint scale')
    split_scale = _positive(split_scale, 'core split scale')
    width_scale = _positive(width_scale, 'tail width scale')
    core_order, tail_order = _order(core_order), _order(tail_order)
    gaussian_width = 1/math.sqrt(decay)
    split = split_scale*min(threshold_scale, gaussian_width)
    tail_width = width_scale*gaussian_width
    def unit(n):
        x, w = np.polynomial.legendre.leggauss(n)
        return (x+1)/2, w/2
    u, w = unit(core_order); v, z = unit(tail_order)
    nodes = np.concatenate((split*u, split-2*tail_width*np.log1p(-v)))
    weights = np.concatenate((split*w, 2*tail_width*z/(1-v)))
    if not np.all(np.isfinite(nodes+weights)) or np.any(nodes <= 0) or np.any(weights <= 0):
        raise ArithmeticError('invalid endpoint quadrature')
    return dict(nodes=nodes.tolist(), weights=weights.tolist(),
        log_weights=np.log(weights).tolist(), measure='dP', domain='[0,infinity)',
        scheme='collision-endpoint-panels-v1', core_order=core_order, tail_order=tail_order,
        decay=decay, threshold_scale=threshold_scale, split=split,
        split_scale=split_scale, width_scale=width_scale,
        physical_momentum_cutoff=None)


def _axes(atlas, pair, radius, primary, *, integrated_long=None, endpoint_scale=None,
          threshold_slopes=None):
    if not 0 < radius < 1:
        raise ValueError('a contracting pair radius in (0,1) is required')
    pair_spectator(pair)
    # Slopes are optional proposal curvature, not an endpoint exponent or a
    # physical cutoff. Keeping them explicit makes their provenance auditable.
    slopes = (0., 0., 0.) if threshold_slopes is None else tuple(threshold_slopes)
    if len(slopes) != 3 or not all(math.isfinite(s) for s in slopes):
        raise ValueError('three finite optional DOZZ proposal curvatures required')
    result = []
    for axis, (a, slope) in enumerate(zip(primary, slopes)):
        a = _positive(a, 'physical propagator decay')
        decay = max(.75*a, a-slope)
        row = dict(axis=axis, primary_decay=a, dozz_threshold_slope=slope, decay=decay,
                   curvature_tempered=decay != a-slope, measure='dP')
        if axis == 0:
            row.update(role='pair_bridge', scheme='integrated-radial-bridge-v1',
                endpoint_power=2 if pair_mass_squared(atlas, pair)>0 else 0,
                mass_squared=pair_mass_squared(atlas, pair))
        elif axis == integrated_long:
            row.update(role='height_integrated_long_loop', scheme='collision-endpoint-panels-v1',
                endpoint_power=1, threshold_scale=endpoint_scale)
        else:
            row.update(role='unintegrated_loop', scheme='generalized-gaussian',
                       endpoint_power=2, alpha=.5)
        result.append(row)
    return result


def pair_design(atlas, *, pair, radius, delta_total, tau, threshold_slopes=None):
    """Thresholds for the radially integrated pair at a fixed torus point."""
    tau, delta_total = complex(tau), complex(delta_total)
    if (not all(math.isfinite(v) for v in (tau.real, tau.imag, delta_total.real, delta_total.imag))
            or tau.imag <= 0):
        raise ValueError('finite torus geometry required')
    loop = canonical_loop_displacement(-delta_total, tau)
    primary = [-2*math.log(_positive(radius, 'pair radius')),
               2*loop.imag, 2*(2*math.pi*tau-loop).imag]
    return dict(format=FORMAT, kernel='radially_integrated_pair', pair=list(pair),
        radius=radius, tau=[tau.real, tau.imag], delta_total=[delta_total.real, delta_total.imag],
        axes=_axes(atlas, pair, radius, primary, threshold_slopes=threshold_slopes),
        geometry_dependent=True, numerical_convergence_certified=False)


def pair_cusp_design(atlas, *, pair, radius, x, y, cutoff, threshold_slopes=None):
    """Thresholds after the analytic pair-cusp tau1/height projection.

    A fixed-height pair rule must not silently supply the endpoint powers
    for this different kernel. The sign of centered y selects the long axis.
    """
    if (not all(math.isfinite(v) for v in (x, y, cutoff)) or abs(x)>math.pi
            or y == 0 or cutoff <= 1):
        raise ValueError('centered nonzero-y puncture and cutoff>1 required')
    distance = abs(y); lower = max(cutoff, distance/math.pi)
    shift = distance/(2*math.pi)
    long_axis, short_axis = (1, 2) if y > 0 else (2, 1)
    primary = [-2*math.log(_positive(radius, 'pair radius')), 0., 0.]
    primary[short_axis] = 2*distance
    primary[long_axis] = 4*math.pi*(lower-shift)
    endpoint_scale = 1/math.sqrt(4*math.pi*lower)
    return dict(format=FORMAT, kernel='height_integrated_pair_cusp', pair=list(pair),
        radius=radius, x=x, y=y, cutoff=cutoff, lower_height=lower, height_shift=shift,
        long_axis=long_axis, short_axis=short_axis,
        axes=_axes(atlas, pair, radius, primary, integrated_long=long_axis,
                   endpoint_scale=endpoint_scale, threshold_slopes=threshold_slopes),
        geometry_dependent=True, numerical_convergence_certified=False)


def build_rules(design, orders, *, long_tail_order=None, split_scale=1., width_scale=1.):
    """Construct the declared independent axes; no N8/N12 inheritance.

    On a height-integrated long axis, orders[axis] is the finite-core order
    and long_tail_order must separately specify the infinite-tail order.
    """
    if design.get('format') != FORMAT or len(design.get('axes', [])) != 3 or len(orders) != 3:
        raise ValueError('an explicit collision threshold design and three orders are required')
    orders = tuple(_order(n) for n in orders)
    has_endpoint = any(a['scheme']=='collision-endpoint-panels-v1' for a in design['axes'])
    if has_endpoint != (long_tail_order is not None):
        raise ValueError('an infinite-tail order is required only for an integrated long axis')
    result = []
    for axis, n in zip(design['axes'], orders):
        if axis['scheme']=='integrated-radial-bridge-v1':
            rule = radial_bridge_rule(n, decay=axis['decay'], mass_squared=axis['mass_squared'])
        elif axis['scheme']=='collision-endpoint-panels-v1':
            rule = endpoint_panels(decay=axis['decay'], threshold_scale=axis['threshold_scale'],
                core_order=n, tail_order=long_tail_order, split_scale=split_scale, width_scale=width_scale)
        elif axis['scheme']=='generalized-gaussian':
            base = generalized_gaussian_momentum_rule(.1, n, laguerre_alpha=axis['alpha'],
                                                      decay_coefficient=axis['decay'])
            weights = np.asarray(base.weights)*math.pi
            rule = dict(nodes=list(base.nodes), weights=weights.tolist(), log_weights=np.log(weights).tolist(),
                order=n, alpha=axis['alpha'], decay=axis['decay'], scheme='generalized-gaussian',
                measure='dP', domain='[0,infinity)')
        else:
            raise ValueError('unknown collision axis rule')
        rule.update(role=axis['role'], endpoint_power=axis['endpoint_power'])
        result.append(rule)
    return result


# genus1_projection_momentum
# Momentum designs for the evaluated projection, including its geometry.
# 
# These are exactly reweighted dP proposals. Full DOZZ functions, subtractions,
# and block coefficients remain in the evaluators. A design is not an accuracy
# certificate, and never changes the nodes of a frozen bank.



import math
import numpy as np

from genus1_three_point_momentum import threshold_slopes
from genus1_three_point_collision import pair_spectator
from genus1_three_point_momentum import threshold_curvature

PLACEMENT_POLICY = 'moduli-head-tail-v1'


def panel_placement(axis, *, total, head, split, width, policy=None):
    """Choose a tested head at fixed TOTAL budget; keep the physical dP tail.

    At b=1 the unit DOZZ scale caps the weakly damped proposals. A Gaussian
    width <= 1/sqrt(8) identifies the strongly damped boundary axes. Height
    integration has a separate finite/infinite-height endpoint and crossover.
    This is a placement policy, never a convergence certificate.
    """
    branch='explicit'
    if policy is not None:
        if policy!=PLACEMENT_POLICY or total not in (8,9):
            raise ValueError('moduli placement requires eight or nine TOTAL nodes')
        role=axis['role'];sigma=1/math.sqrt(axis['decay'])
        base=min(axis['threshold_scale'],sigma)
        if role=='height_integrated_handle':
            power=axis['endpoint_power']
            if power not in (1,2):raise ValueError('height endpoint must be linear or quadratic')
            extent=((1.5 if total==8 else 2.)*sigma if power==2 else 2.25*base)
            head=total-3;split=extent/base;width=.275
            branch='finite_height' if power==2 else 'infinite_height'
        elif role in ('unintegrated_short','unintegrated_loop','unintegrated_handle') and axis['decay']>=8.:
            head=total-3;split=2.25;width=.275;branch='strong_boundary_decay'
        else:
            branch='retained_wide_head'
    return dict(head=head,tail=total-head,split=split,width=width,branch=branch)


def kinematics(atlas, channel, pair=(0, 1)):
    indices = (0, 1, 2) if channel=='necklace' else (*pair, pair_spectator(pair))
    return dict(channel=channel, coefficient_basis='native-sewing-v2',
        external_weights=tuple(atlas.external_weights[i] for i in indices),
        external_momenta=tuple(atlas.external_momenta[i] for i in indices))


def axis_design(axis, primary, slope, *, role, endpoint_power, mass_squared=None,
                threshold_scale=None, proposal_floor=None):
    if not math.isfinite(primary) or not math.isfinite(slope):
        raise ValueError('finite physical decay and DOZZ curvature required')
    if proposal_floor is not None and (not math.isfinite(proposal_floor) or proposal_floor<=0):
        raise ValueError('positive proposal floor required')
    anchor = primary if proposal_floor is None else max(primary, proposal_floor)
    if anchor <= 0:
        raise ValueError('no contracting Gaussian scale; select a different chart or explicit proposal')
    decay = max(.75*anchor, anchor-slope)
    row = dict(axis=axis, role=role, primary_decay=primary, decay=decay,
        dozz_threshold_slope=slope, endpoint_power=endpoint_power, measure='dP',
        proposal_floor=proposal_floor, proposal_floor_used=anchor!=primary)
    if mass_squared is not None:
        row.update(scheme='integrated-radial-bridge-v1', mass_squared=mass_squared)
    elif threshold_scale is not None:
        row.update(scheme='collision-endpoint-panels-v1', threshold_scale=threshold_scale)
    else:
        row.update(scheme='generalized-gaussian', alpha=(endpoint_power-1)/2)
    return row


def height_design(atlas, *, pair, region, pair_radius, triple_radius, cutoff=8.,
                  tau1=None, shape_anchor=.5):
    """Scale/corner height integrals: bridge P²/beta, scale P⁰, handle P² or P.

    One bank can cover the modular real strip by using its smallest lower
    height. The shape anchor is explicitly a proposal: the corner's coupled
    radial kernel cannot be replaced by three separable exponentials.
    """
    pair=tuple(pair); pair_spectator(pair)
    if (region not in ('bulk', 'tail') or not math.isfinite(cutoff) or cutoff<=1
            or not 0<pair_radius<=triple_radius/2 or not triple_radius<1
            or not 0<shape_anchor<1
            or (tau1 is not None and (not math.isfinite(tau1) or abs(tau1)>.5))):
        raise ValueError('contracting radii, shape anchor and modular height interval required')
    slopes=threshold_slopes(atlas.special, kinematics(atlas, 'comb_ope', pair))
    lower=(math.sqrt(.75) if tau1 is None else math.sqrt(1-tau1*tau1)) if region=='bulk' else cutoff
    primary=(-2*math.log(shape_anchor), -2*math.log(triple_radius), 4*math.pi*lower)
    mu2=pair_mass_squared(atlas,pair)
    axes=[axis_design(0, primary[0], slopes[0], role='integrated_shape_bridge',
                     endpoint_power=2 if mu2 else 0, mass_squared=mu2),
          axis_design(1, primary[1], slopes[1], role='neutral_integrated_scale', endpoint_power=0,
                      threshold_scale=1/math.sqrt(max(.75*primary[1],primary[1]-slopes[1]))),
          axis_design(2, primary[2], slopes[2], role='height_integrated_handle',
                      endpoint_power=2 if region=='bulk' else 1,
                      threshold_scale=1/math.sqrt(4*math.pi*cutoff))]
    return dict(format=FORMAT, kernel='height_integrated_scale_and_corner', axes=axes,
        pair=list(pair), region=region, pair_radius=pair_radius, triple_radius=triple_radius,
        cutoff=cutoff, tau1=tau1, lower_height=lower,
        upper_height=cutoff if region=='bulk' else None, shape_anchor=shape_anchor,
        shape_decay_is_proposal=True, numerical_convergence_certified=False)


def total_sampling_controls(config):
    """Explicit TOTAL nodes per physical momentum integral, including its tail.

    Scalars apply to every axis; triples support a ninth node on selected
    axes only. High-order references use the separate legacy controls.
    """
    if not isinstance(config,dict) or set(config)-{'total_nodes','head_orders','split_scales','tail_width_scales','axis_order','placement_policy'}:
        raise ValueError('explicit total-node momentum sampling controls required')
    frame=config.get('axis_order','native')
    if frame not in ('native','cut_short_short'):
        raise ValueError('momentum axis order must be native or cut_short_short')
    defaults=dict(total_nodes=8,head_orders=5,split_scales=1.5,tail_width_scales=.25)
    result={}
    for name,default in defaults.items():
        value=config.get(name,default)
        values=list(value) if isinstance(value,(list,tuple)) else [value]*3
        if len(values)!=3:raise ValueError('three momentum axis controls required')
        if name in ('total_nodes','head_orders'):
            if any(type(v) is not int or v<1 for v in values):raise ValueError('integer momentum node counts required')
        elif any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0 for v in values):
            raise ValueError('positive finite momentum sampling scales required')
        result[name]=values
    if any(not 2<=n<=9 or not 1<=h<n for n,h in zip(result['total_nodes'],result['head_orders'])):
        raise ValueError('head plus tail must fit the total momentum budget of at most 9')
    if frame!='native':result['axis_order']=frame
    if 'placement_policy' in config:
        if config['placement_policy']!=PLACEMENT_POLICY or any(n not in (8,9) for n in result['total_nodes']):
            raise ValueError('unknown placement policy or incompatible TOTAL node budget')
        result['placement_policy']=PLACEMENT_POLICY
    return result


def check_total_nodes(rules, sampling):
    controls=total_sampling_controls(sampling)
    if 'axis_order' in controls:
        raise ValueError('resolve cut-relative momentum controls before checking actual nodes')
    if len(rules)!=3 or any(len(r['nodes'])!=n for r,n in zip(rules,controls['total_nodes'])):
        raise ValueError('actual momentum nodes differ from the TOTAL per-axis budget')


def short_refinement_controls(config):
    if not isinstance(config,dict) or set(config)-{'maximum_short_length','total_nodes','head_order','split_scale','tail_width_scale'}:
        raise ValueError('explicit short-region refinement controls required')
    row=dict(maximum_short_length=.125,total_nodes=9,head_order=6,split_scale=1.5,tail_width_scale=.275)
    row.update(config)
    length=row['maximum_short_length']
    if isinstance(length,bool) or not isinstance(length,(int,float)) or not math.isfinite(length) or length<=0:
        raise ValueError('positive finite short-region boundary required')
    total_sampling_controls(dict(total_nodes=row['total_nodes'],head_orders=row['head_order'],
        split_scales=row['split_scale'],tail_width_scales=row['tail_width_scale']))
    if row['total_nodes']!=9:raise ValueError('regional refinement is limited to nine total nodes')
    return row


def resolved_primary_sampling(atlas,short,sampling,refinement=None):
    from genus1_analytic_boundary_layer import necklace_slice_geometry
    controls=total_sampling_controls(sampling)
    placement=controls.pop('placement_policy',None)
    if controls.pop('axis_order','native')=='cut_short_short':
        cut=necklace_slice_geometry(atlas,short)['cut_axis']
        controls={name:[values[(i-cut)%3] for i in range(3)] for name,values in controls.items()}
    axes=[]
    if refinement is not None:
        refinement=short_refinement_controls(refinement)
        if any(n>8 for n in controls['total_nodes']):raise ValueError('generic primary budget must be at most eight')
        cut=necklace_slice_geometry(atlas,short)['cut_axis']
        for offset,length in ((1,short.s1),(2,short.s2)):
            if length<refinement['maximum_short_length']:
                axis=(cut+offset)%3;axes.append(axis)
                for field,key in (('total_nodes','total_nodes'),('head_orders','head_order'),
                                  ('split_scales','split_scale'),('tail_width_scales','tail_width_scale')):
                    controls[field][axis]=refinement[key]
    if placement is not None:controls['placement_policy']=placement
    return controls,axes


def projection_rules(design, orders, *, tail_orders, split_scales=(1.,1.,1.), width_scales=(1.,1.,1.),
                     sampling=None):
    """Independent core, outer and width controls for every endpoint panel."""
    from liouville_momentum_quadrature import generalized_gaussian_momentum_rule
    if (design.get('format')!=FORMAT or len(design['axes'])!=3
            or any(len(v)!=3 for v in (orders,tail_orders,split_scales,width_scales))
            or any(type(n) is not int or n<1 for n in orders)):
        raise ValueError('explicit design and three independent axis controls required')
    budget=None if sampling is None else total_sampling_controls(sampling)
    if budget is not None and 'axis_order' in budget:
        raise ValueError('cut-relative momentum controls require a primary kernel')
    result=[]
    for i,(a,n,tail) in enumerate(zip(design['axes'],orders,tail_orders)):
        split,width=split_scales[i],width_scales[i]
        if budget is not None:
            n=budget['total_nodes'][i];tail=None;split=width=1.
            if a['scheme']=='collision-endpoint-panels-v1':
                head=budget['head_orders'][i];tail=n-head;n=head
                split=budget['split_scales'][i];width=budget['tail_width_scales'][i]
        if a['scheme']=='collision-endpoint-panels-v1':
            placement=None
            if budget is not None and budget.get('placement_policy') is not None:
                placement=panel_placement(a,total=budget['total_nodes'][i],head=n,split=split,width=width,
                                          policy=budget['placement_policy'])
                n,tail,split,width=(placement[k] for k in ('head','tail','split','width'))
            r=endpoint_panels(decay=a['decay'],threshold_scale=a['threshold_scale'],
                core_order=n,tail_order=tail,split_scale=split,width_scale=width)
            if placement is not None:r['placement']=dict(policy=budget['placement_policy'],**placement)
        else:
            if tail is not None or split!=1 or width!=1:
                raise ValueError('panel controls supplied to a non-panel axis')
            if a['scheme']=='integrated-radial-bridge-v1':
                r=radial_bridge_rule(n,decay=a['decay'],mass_squared=a['mass_squared'])
            elif a['scheme']=='generalized-gaussian':
                base=generalized_gaussian_momentum_rule(0j,n,laguerre_alpha=a['alpha'],
                    decay_coefficient=a['decay'],log_q_abs=-a['decay']/2)
                r=dict(nodes=list(base.nodes),weights=(np.asarray(base.weights)*math.pi).tolist(),
                    order=n,alpha=a['alpha'],decay=a['decay'],scheme=a['scheme'],measure='dP',domain='[0,infinity)')
            else:
                raise ValueError('unknown projection quadrature scheme')
        r.update(log_weights=np.log(r['weights']).tolist(),role=a['role'],endpoint_power=a['endpoint_power'])
        result.append(r)
    if budget is not None:check_total_nodes(result,budget)
    return result


def pair_rules(atlas, *, pair, radius, delta_total, tau, orders, tail_orders=None, sampling=None):
    slopes=threshold_slopes(atlas.special, kinematics(atlas,'pair_ope',tuple(pair)))
    design=pair_design(atlas,pair=pair,radius=radius,delta_total=delta_total,tau=tau,threshold_slopes=slopes)
    # When one sewing length is tiny, Gaussian nodes can all miss the
    # coupled DOZZ peak near P=0, even with the correct threshold curvature.
    # Resolve [0,min(1,sigma)] directly; keep an explicit infinite outer panel.
    # The unit scale is Q/2 at b=1, not a physical momentum cutoff.
    for axis in design['axes'][1:]:
        axis.update(scheme='collision-endpoint-panels-v1',threshold_scale=min(1.,1/math.sqrt(axis['decay'])))
        axis.pop('alpha',None)
    tails=orders if tail_orders is None else tail_orders
    return design,projection_rules(design,orders,tail_orders=(None,tails[1],tails[2]),sampling=sampling)


def full_rules(atlas, encoded_geometry, *, orders, tail_orders, sampling=None):
    """Keep the full-CFT quadratic zeros and resolve its coupled DOZZ residual."""
    from genus1_native_bank import point_kinematics
    kin=point_kinematics(atlas,encoded_geometry)
    slopes=threshold_curvature(atlas.special,kin)
    axes=[]
    for i,(a,s) in enumerate(zip(kin['decay'],slopes)):
        decay=max(.75*a,a-s)
        axes.append(axis_design(i,float(a),s,role='unintegrated_full_cft',endpoint_power=2,
                                threshold_scale=1/math.sqrt(decay)))
    design=dict(format=FORMAT,kernel='full_cft',axes=axes,channel=kin['channel'],
                numerical_convergence_certified=False)
    return kin,design,projection_rules(design,orders,tail_orders=tail_orders,sampling=sampling)


def comb_pair_rules(atlas, *, pair, tau, scale, phase, shape_upper, orders, tail_orders=None, sampling=None):
    """One shared rule for the signed pair-face minus corner subtraction.

    The two parents have different exact propagator decays. Use the broader
    proposal on each axis, preserving subtraction at identical momentum nodes.
    Near a unit shape boundary a separable decay can cease to be positive;
    the explicit .125 proposal floor then requires a width/order check.
    """
    pair=tuple(pair); pair_spectator(pair); tau=complex(tau)
    if (not all(math.isfinite(x) for x in (tau.real,tau.imag,scale,phase,shape_upper))
            or tau.imag<=0 or abs(tau.real)>.5 or abs(tau)<1-1e-14
            or not 0<scale<1 or not 0<shape_upper<=1):
        raise ValueError('finite contracting comb pair-face geometry required')
    total=scale*np.exp(1j*phase); v=abs(np.expm1(-1j*total))
    parents=[[-2*math.log(scale*shape_upper/v), -2*math.log(v), 4*math.pi*tau.imag],
             [-2*math.log(shape_upper), -2*math.log(scale), 4*math.pi*tau.imag]]
    primary=np.min(parents,axis=0)
    slopes=threshold_slopes(atlas.special, kinematics(atlas,'comb_ope',pair))
    mu2=pair_mass_squared(atlas,pair)
    axes=[axis_design(i,float(a),s,role=('integrated_pair_bridge','unintegrated_scale','unintegrated_handle')[i],
                      endpoint_power=(0 if mu2==0 else 2) if i==0 else 2,
                      mass_squared=mu2 if i==0 else None, proposal_floor=.125)
          for i,(a,s) in enumerate(zip(primary,slopes))]
    design=dict(format=FORMAT,kernel='comb_pair_face',axes=axes,pair=list(pair),
        parent_decays=parents,subtraction_at_shared_nodes=True,numerical_convergence_certified=False)
    for axis in axes[1:]:
        axis.update(scheme='collision-endpoint-panels-v1',threshold_scale=min(1.,1/math.sqrt(axis['decay'])))
        axis.pop('alpha',None)
    tails=orders if tail_orders is None else tail_orders
    return design,projection_rules(design,orders,tail_orders=(None,tails[1],tails[2]),sampling=sampling)


def primary_sampling_controls(config=None):
    """Controls for the two finite-head/infinite-tail short-momentum rules.

    head_order=None inherits each axis' requested order. An explicit order
    changes only the short heads, never the kernel-weighted cut momentum.
    An optional node_budget constrains every axis, including the cut rule.
    The wider default split was checked on the physical H4 density; it is
    deliberately scoped to primary tubes, not collision endpoint panels.
    """
    if config is not None and not isinstance(config,dict):
        raise ValueError('primary sampling controls must be an object')
    config={} if config is None else dict(config)
    if set(config)-{'head_order','split_scale','tail_width_scale','node_budget'}:
        raise ValueError('unknown primary sampling control')
    result=dict(head_order=None,split_scale=2.,tail_width_scale=1.)
    result.update(config)
    n=result['head_order']
    if n is not None and (type(n) is not int or n<1):
        raise ValueError('positive integer primary head order required')
    if 'node_budget' in result and (type(result['node_budget']) is not int or result['node_budget']<2):
        raise ValueError('integer primary short-axis node budget >=2 required')
    for key in ('split_scale','tail_width_scale'):
        value=result[key]
        if (isinstance(value,bool) or not isinstance(value,(int,float)) or
                not math.isfinite(value) or value<=0):
            raise ValueError('positive finite primary '+key+' required')
        result[key]=float(value)
    return result


def primary_rules(atlas, short, *, orders, lower_length, construction_order=192,
                  tail_exponent=80., tail_orders=None, sampling=None, momentum_sampling=None,short_refinement=None):
    """Actual short geometry and full length kernel, suitable for an H4 bank.

    The kernel-weighted cut rule already resolves the linear endpoint. Its
    representative geometry must not supply the grid for arbitrarily longer
    short tubes. Direct core/outer panels on both short axes retain their
    quadratic threshold and decay while resolving the coupled DOZZ residual.
    The finite head extends to 2*min(1,1/sqrt(decay)) by default so that the
    coupled peak is resolved before the sparsely sampled outer tail begins.
    """
    from genus1_analytic_boundary_layer import necklace_slice_geometry
    from genus1_nonseparating import kernel_weighted_cut_rule
    if sampling is not None and momentum_sampling is not None:
        raise ValueError('choose one explicit momentum sampling policy')
    if short_refinement is not None and momentum_sampling is None:
        raise ValueError('regional refinement requires a total-node budget')
    budget,refined_axes=(None,[]) if momentum_sampling is None else resolved_primary_sampling(
        atlas,short,momentum_sampling,short_refinement)
    sampling=primary_sampling_controls(sampling)
    if (len(orders)!=3 or any(type(n) is not int or n<1 for n in orders)
            or not math.isfinite(lower_length) or lower_length<max(short.s1,short.s2)):
        raise ValueError('three positive orders and a valid longest-gap lower limit required')
    g=necklace_slice_geometry(atlas,short); cut_axis=g['cut_axis']
    tails=orders if tail_orders is None else tail_orders
    if len(tails)!=3 or any(type(n) is not int or n<1 for n in tails):
        raise ValueError('three positive outer-panel orders required')
    if 'node_budget' in sampling:
        for i,n in enumerate(orders):
            head=n if sampling['head_order'] is None else sampling['head_order']
            count=n if i==cut_axis else head+tails[i]
            if count>sampling['node_budget']:
                raise ValueError('primary momentum node budget exceeded')
    slopes=threshold_curvature(atlas.special, kinematics(atlas,'necklace'))
    primary=[0.]*3; primary[cut_axis]=2*lower_length
    primary[(cut_axis+1)%3]=2*short.s1; primary[(cut_axis+2)%3]=2*short.s2
    axes=[]; rules=[]
    cut=dict(length=lower_length,minimum_length=lower_length,short_length=short.total,gaussian=g['gaussian'])
    for i,(a,s,n) in enumerate(zip(primary,slopes,orders)):
        if budget is not None:n=budget['total_nodes'][i]
        row=axis_design(i,a,s,role='integrated_cut' if i==cut_axis else 'unintegrated_short',
                        endpoint_power=1 if i==cut_axis else 2)
        if i==cut_axis:
            r=kernel_weighted_cut_rule(cut,s,n,construction_order=construction_order,tail_exponent=tail_exponent)
            row.update(scheme=r['scheme'],threshold_scale=1/math.sqrt(2*(lower_length+short.total)))
        else:
            scale=min(1.,1/math.sqrt(row['decay']))
            row.update(scheme='collision-endpoint-panels-v1',threshold_scale=scale)
            head=(n if sampling['head_order'] is None else sampling['head_order']) if budget is None else budget['head_orders'][i]
            tail=tails[i] if budget is None else n-head
            split=sampling['split_scale'] if budget is None else budget['split_scales'][i]
            width=sampling['tail_width_scale'] if budget is None else budget['tail_width_scales'][i]
            placement=None
            if budget is not None and budget.get('placement_policy') is not None:
                placement=panel_placement(row,total=n,head=head,split=split,width=width,policy=budget['placement_policy'])
                head,tail,split,width=(placement[k] for k in ('head','tail','split','width'))
            r=endpoint_panels(decay=row['decay'],threshold_scale=scale,
                core_order=head,tail_order=tail,split_scale=split,width_scale=width)
            if placement is not None:r['placement']=dict(policy=budget['placement_policy'],**placement)
            row.update(scheme=r['scheme'],threshold_scale=scale)
            row.pop('alpha',None)
        r=dict(r,nodes=np.asarray(r['nodes']).tolist(),weights=np.asarray(r['weights']).tolist(),
               log_weights=np.log(r['weights']).tolist(),endpoint_power=row['endpoint_power'],role=row['role'])
        axes.append(row);rules.append(r)
    if budget is not None:check_total_nodes(rules,budget)
    design=dict(format=FORMAT,kernel='length_integrated_primary',axes=axes,cut_axis=cut_axis,
        short=short.record(),lower_length=lower_length,kernel_construction_order=construction_order,
        primary_sampling=sampling,numerical_convergence_certified=False)
    if budget is not None:
        design.pop('primary_sampling')
        design['momentum_sampling']=budget
        design['refined_momentum_axes']=refined_axes
        if short_refinement is not None:design['short_refinement']=short_refinement_controls(short_refinement)
    return design,rules
