"""Opt-in, anisotropic moduli panels for the retained analytic estimator.

These rules return physical measures. They do not set momentum rules, CFT
orders, matching radii, or tolerances, and do not alter frozen productions.
"""
import math
import numpy as np

from genus1_boundary_cubature import gauss_rule
from genus1_moduli_sampling import ShortPoint, TAU, regular_mask
from genus1_analytic_boundary_layer import MIN_HEIGHT
from genus1_collision_sampling import Request, SamplePlan
from genus1_three_point_collision import PAIR_ROWS

FORMAT = 'genus1-moduli-panels-v2'


def reference_edges(parts):
    if type(parts) is int:
        if parts < 1: raise ValueError('positive subdivision count required')
        return np.linspace(0., 1., parts+1)
    edges = np.asarray(parts, dtype=float)
    if edges.ndim != 1 or len(edges) < 2 or edges[0] != 0 or edges[-1] != 1 or not np.all(np.diff(edges) > 0):
        raise ValueError('ordered reference edges covering [0,1] required')
    return edges


def composite_interval(a, b, order, parts=1):
    """Subdivide a fixed reference interval, preserving the old one-panel rule."""
    edges = reference_edges(parts)
    if len(edges) == 2:
        return interval(a, b, order)
    edges = a+(b-a)*edges
    rules = [interval(float(x), float(y), order) for x, y in zip(edges[:-1], edges[1:])]
    return tuple(np.concatenate([r[i] for r in rules]) for i in (0, 1))


def rounded_interval(a, b, order, parts=1):
    """Quadratic endpoint approach with an exactly integrated G2 Jacobian.

    Unlike a cosine map, the cubic smoothstep has a quadratic Jacobian; a
    two-point rule already integrates a constant physical density exactly.
    """
    if order < 2 or not math.isfinite(a+b) or not a < b:
        raise ValueError('finite ordered interval and at least two nodes required')
    # Subdivide u before applying this SAME smoothstep on the original panel.
    # Restarting the smoothstep on physical children is a different sequence.
    u, w = composite_interval(0., 1., order, parts)
    return a+(b-a)*u*u*(3-2*u), w*(b-a)*6*u*(1-u)


def length_orders(order):
    """Separate maximum-length and ratio orders; an integer preserves both."""
    orders = (order, order) if type(order) is int else tuple(order)
    if len(orders) != 2 or any(type(n) is not int or n < 2 for n in orders):
        raise ValueError('two length orders of at least two required')
    return orders


def length_rule(policy, panel, order, *, ratio_bounds=(0., 1.), subdivisions=(1, 1)):
    """Physical ds1 ds2, with the same analytic transition surfaces."""
    maximum_order, ratio_order = length_orders(order)
    maximum_parts, ratio_parts = subdivisions
    a, b = panel
    if b is None:
        u, w = composite_interval(0., 1., maximum_order, maximum_parts)
        ms, mw = a+policy.length*u/(1-u), w*policy.length/(1-u)**2
    else:
        ms, mw = rounded_interval(a,b,maximum_order,maximum_parts)
    for m, wm in zip(ms,mw):
        floor = max(m,policy.length)
        rs = breaks(*ratio_bounds,(policy.pair_radius/m,policy.pair_radius/m-1,
            *((TAU*h-floor)/m-1 for h in (MIN_HEIGHT,1.))))
        for a,b in zip(rs[:-1],rs[1:]):
            ratios,rw = rounded_interval(a,b,ratio_order,ratio_parts)
            for ratio,wr in zip(ratios,rw): yield float(m),float(m*ratio),float(m*wm*wr)


def torus_lift(x, y, tau):
    """Unit-Jacobian map from a centered rectangle to the usual torus lift."""
    v = y/(TAU*tau.imag)
    return TAU*((x/TAU-v*tau.real) % 1 + (v % 1)*tau)


def pair_puncture_rule(tau, radius, edge, radial_order, angular_order, *, separation_splits=(),
                       subdivisions=(1, 1)):
    """Disjoint annulus and exact complement; no rejection or normalization.

    The disk of radius ``edge`` fits in the centered torus rectangle. The
    exterior uses horizontal slices, with y=edge*sin(phi) at the circle, so
    its Jacobian removes the square-root endpoint. Far strips retain physical
    y resolution even when the torus is tall. Tags allow separate diagnostics.
    """
    tau = complex(tau)
    radial_parts, angular_parts = subdivisions
    height = TAU*tau.imag
    if not 0 < radius < edge < min(math.pi, height/2):
        raise ValueError('annulus must fit inside the torus rectangle')
    areas, aw = composite_interval(radius*radius, edge*edge, radial_order, radial_parts)
    for k in range(4):
        angles, tw = composite_interval(k*math.pi/2, (k+1)*math.pi/2, angular_order, angular_parts)
        for a, wa in zip(areas, aw):
            for theta, wt in zip(angles, tw):
                r = math.sqrt(a)
                yield torus_lift(r*math.cos(theta), r*math.sin(theta), tau), float(wa*wt/2), 'annulus'
    for low, high in ((-math.pi/2, 0.), (0., math.pi/2)):
        angles, weights = composite_interval(low, high, radial_order, radial_parts)
        for phi, wp in zip(angles, weights):
            y, lower = edge*math.sin(phi), edge*math.cos(phi)
            for a, b in ((-math.pi, -lower), (lower, math.pi)):
                xs, xw = composite_interval(a, b, angular_order, angular_parts)
                for x, wx in zip(xs, xw):
                    yield torus_lift(x, y, tau), float(wp*lower*wx), 'near_complement'
    for low, high in ((-height/2, -edge), (edge, height/2)):
        edges=breaks(low,high,(*separation_splits,*(-v for v in separation_splits)))
        for c,d in zip(edges[:-1],edges[1:]):
            ys, yw = composite_interval(c, d, radial_order, radial_parts)
            for y, wy in zip(ys, yw):
                for a, b in ((-math.pi, 0.), (0., math.pi)):
                    xs, xw = composite_interval(a, b, angular_order, angular_parts)
                    for x, wx in zip(xs, xw):
                        yield torus_lift(x, y, tau), float(wy*wx), 'far_complement'


def pair_tau_rule(plan, region, real_order, height_order, *,
                  bounds=((0., 1.), (0., 1.)), height_splits=(2., 4.), subdivisions=(1, 1)):
    """Same modular domain as the original cell, with finite-height panels."""
    if region not in ('bulk', 'tail'):
        raise ValueError('bulk or tail required')
    xi, yi = (0, 1) if region == 'bulk' else (1, 0)
    real_parts, height_parts = subdivisions
    xs, xw = composite_interval(bounds[xi][0]-.5, bounds[xi][1]-.5, real_order, real_parts)
    for x, wx in zip(xs, xw):
        if region == 'bulk':
            floor = math.sqrt(1-x*x)
            lower, upper = (floor+u*(plan.collision_height_split-floor) for u in bounds[yi])
            edges = breaks(lower, upper, height_splits)
            for a, b in zip(edges[:-1], edges[1:]):
                ys, yw = composite_interval(a, b, height_order, height_parts)
                for y, wy in zip(ys, yw):
                    yield complex(x, y), float(wx*wy), (float(a), float(b))
        else:
            us, uw = composite_interval(*bounds[yi], height_order, height_parts)
            for u, wu in zip(us, uw):
                y = plan.collision_height_split/(1-u)
                yield complex(x, y), float(wx*wu*plan.collision_height_split/(1-u)**2), (plan.collision_height_split, None)


def pair_rule(plan, region, orders, *, bounds=((0., 1.), (0., 1.)),
              edge=None, height_splits=(2., 4.), separation_splits=(), subdivisions=(1, 1, 1, 1)):
    """Four independent moduli orders: tau1, height, radial, angular."""
    a, b, c, d = orders
    radius = plan.policy.triple_radius
    edge = 2*radius if edge is None else edge
    for tau, tw, panel in pair_tau_rule(plan, region, a, b, bounds=bounds, height_splits=height_splits,
                                      subdivisions=subdivisions[:2]):
        for z, zw, tag in pair_puncture_rule(tau, radius, edge, c, d,separation_splits=separation_splits,
                                            subdivisions=subdivisions[2:]):
            requests = tuple(Request('pair_projection', dict(delta_total=z, tau=tau,
                radius=plan.policy.pair_radius, integrated=True, pair=pair)) for pair in PAIR_ROWS)
            yield SamplePlan(tw*zw, requests), dict(puncture=tag, height_panel=panel)


def primary_length_panels(policy, band, max_bounds, ratio_bounds):
    """All topology changes for a restricted legacy cell, including its edges."""
    if band not in (0, 1):
        raise ValueError('short-length band must be zero or one')
    if not (0 <= max_bounds[0] < max_bounds[1] <= 1 and
            0 <= ratio_bounds[0] < ratio_bounds[1] <= 1):
        raise ValueError('ordered unit bounds required')
    L, R = policy.length, policy.pair_radius
    def maximum(u):
        return L*u if band == 0 else (L/(1-u) if u < 1 else math.inf)
    low, high = map(maximum, max_bounds)
    candidates = [L, R]
    for r in ratio_bounds:
        candidates.append(R/(1+r))
        if r > 0: candidates.append(R/r)
        for c in (TAU*MIN_HEIGHT, TAU):
            below, above = (c-L)/(1+r), c/(2+r)
            if 0 < below < L: candidates.append(below)
            if above > L: candidates.append(above)
    # breaks needs finite endpoints; infinity is represented only on the last
    # panel. Each finite panel keeps its actual measure, even if very narrow.
    if math.isfinite(high):
        edges = breaks(low, high, candidates)
    else:
        edges = sorted({low, *(v for v in candidates if v > low)}) + [math.inf]
    return [(a, b if math.isfinite(b) else None) for a, b in zip(edges[:-1], edges[1:])]


def angular_head_rule(s1,s2,radius,order,*,x1_bounds=(0.,TAU),x2_bounds=(0.,TAU),
                      tail_order=None, subdivisions=(1, 1),
                      near_head_subdivisions=None, near_head_distance=None):
    """Resolve allowed near-pair angular variation before the broad tail.

    The numerical horizontal head width is 2*max(rho, vertical separation),
    capped at pi. All collision exclusions still use rho. The outer rule
    splits physical ownership changes. Numerical inner panels may appear or
    disappear with zero measure; splitting all of their intersections again
    would create an unnecessary Cartesian proliferation of thin panels.

    The optional lower tail order applies only to angular intervals outside
    every local head, and only when both short lengths exceed the collision
    radius. Collision-clipped slices retain the original angular order: the
    moving exclusions couple the outer and inner angular integrations.
    It changes no momentum rule or physical exclusion. Omitting it preserves
    the original uniform-order rule exactly.

    A bounded stability test can subdivide only heads whose vertical
    separation is at most ``near_head_distance``. All original panel edges,
    Gauss orders and other intervals remain intact. This option cannot be
    combined with global angular subdivision.
    """
    if min(s1,s2,radius)<=0 or radius>=math.pi:
        raise ValueError('positive lengths and radius below pi required')
    if type(order) is not int or order<2:
        raise ValueError('at least two angular nodes per head required')
    if tail_order is not None and (type(tail_order) is not int or not 2<=tail_order<=order):
        raise ValueError('angular tail order must lie between two and the head order')
    if near_head_subdivisions is not None:
        if tuple(subdivisions)!=(1,1) or len(near_head_subdivisions)!=2:
            raise ValueError('near-head subdivisions require two axes and no global subdivision')
        if near_head_distance is None or not math.isfinite(near_head_distance) or near_head_distance<=0:
            raise ValueError('positive finite near-head distance required')
        for parts in near_head_subdivisions:reference_edges(parts)
    elif near_head_distance is not None:
        raise ValueError('near-head distance requires explicit subdivisions')
    if tail_order is not None and min(s1,s2)<=radius:
        tail_order=order
    def panel_order(midpoint,heads):
        if tail_order is None:return order
        for center,width in heads:
            delta=(midpoint-center)%TAU
            if min(delta,TAU-delta)<width:return order
        return tail_order
    def panel_parts(axis,midpoint,heads):
        if near_head_subdivisions is None:return subdivisions[axis]
        for center,width,separation in heads:
            delta=(midpoint-center)%TAU
            if separation<=near_head_distance and min(delta,TAU-delta)<width:
                return near_head_subdivisions[axis]
        return 1
    a1,a2,a12=(math.sqrt(max(0.,radius**2-s*s)) for s in (s1,s2,s1+s2))
    b1,b2,b12=(min(math.pi,2*max(radius,s)) for s in (s1,s2,s1+s2))
    left,right=max(a1,x1_bounds[0]),min(TAU-a1,x1_bounds[1])
    low,high=max(a12,x2_bounds[0]),min(TAU-a12,x2_bounds[1])
    if left>=right or low>=high:return
    fixed=breaks(low,high,(b12,math.pi,TAU-b12))
    outer=breaks(left,right,(b1,math.pi,TAU-b1,*[
        v-offset-shift for v in (low,high,math.pi) for offset in (-a2,a2)
        for shift in (-TAU,0.,TAU)]))
    for a,b in zip(outer[:-1],outer[1:]):
        xs,xw=composite_interval(a,b,panel_order((a+b)/2,((0.,b1),)),
            panel_parts(0,(a+b)/2,((0.,b1,s1),)))
        for x,wx in zip(xs,xw):
            for c,d in angular_intervals(s1,s2,radius,x):
                c,d=max(c,low),min(d,high)
                if c>=d:continue
                inner=breaks(c,d,(*fixed,*[x+sign*b2+shift
                    for sign in (-1,1) for shift in (-TAU,0.,TAU)]))
                for e,f in zip(inner[:-1],inner[1:]):
                    ys,yw=composite_interval(e,f,panel_order((e+f)/2,((0.,b12),(x,b2))),
                        panel_parts(1,(e+f)/2,((0.,b12,s1+s2),(x,b2,s2))))
                    for y,wy in zip(ys,yw):yield float(x),float(y),float(wx*wy)


def primary_cell_rule(plan, cell, length_order, angle_order, *, fitted_lengths=True,
                      fitted_angles=True, angular_heads=False,angular_tail_order=None,
                      subdivisions=(1, 1, 1, 1)):
    """A cell's unchanged support, with independently controlled length/angles."""
    sector = tuple(map(int, cell['term'].split(':')[1]))
    ordering, band = (int(v[-1]) for v in cell['key'].split(':')[-2:])
    mb, rb, xb, yb = cell['bounds']
    if ordering not in (0, 1): raise ValueError('explicit short-length ordering required')
    if len(subdivisions) != 4:
        raise ValueError('four subdivision specifications required')
    for parts in subdivisions: reference_edges(parts)
    if tuple(subdivisions[2:]) != (1, 1) and not angular_heads:
        raise ValueError('angular subdivision requires the head-panel rule')
    if angular_tail_order is not None and not (fitted_angles and angular_heads):
        raise ValueError('an angular tail order requires the fitted head panels')
    def lengths():
        maximum_order, ratio_order = length_orders(length_order)
        if fitted_lengths:
            for panel in primary_length_panels(plan.policy, band, mb, rb):
                yield from length_rule(plan.policy, panel, length_order, ratio_bounds=rb,
                                       subdivisions=subdivisions[:2])
        else:
            us, uw = composite_interval(*mb, maximum_order, subdivisions[0])
            rs, rw = composite_interval(*rb, ratio_order, subdivisions[1])
            for u, wu in zip(us, uw):
                m = plan.policy.length*u if band == 0 else plan.policy.length/(1-u)
                dm = plan.policy.length if band == 0 else plan.policy.length/(1-u)**2
                for r, wr in zip(rs, rw): yield m, m*r, float(m*dm*wu*wr)
    for m, n, lw in lengths():
        s1, s2 = (m, n) if ordering == 0 else (n, m)
        if fitted_angles:
            angles = (angular_head_rule if angular_heads else angular_rule)(s1, s2, plan.policy.pair_radius, angle_order,
                x1_bounds=tuple(TAU*x for x in xb), x2_bounds=tuple(TAU*x for x in yb),
                **({'tail_order':angular_tail_order,'subdivisions':subdivisions[2:]} if angular_heads else {}))
        else:
            xs, xw = interval(*(TAU*x for x in xb), angle_order)
            ys, yw = interval(*(TAU*x for x in yb), angle_order)
            angles = ((x, y, wx*wy) for x, wx in zip(xs, xw) for y, wy in zip(ys, yw))
        for x, y, aw in angles:
            short = ShortPoint(sector, s1, s2, float(x), float(y))
            if not regular_mask(short, plan.policy.pair_radius, MIN_HEIGHT):
                if fitted_angles: raise ArithmeticError('fitted angle entered a collision collar')
                continue
            yield short, float(lw*aw)


def interval(a, b, order, *, rounded=False):
    """Gauss rule, optionally with quadratic approach to both endpoints."""
    if not math.isfinite(a+b) or not a < b:
        raise ValueError('finite ordered interval required')
    nodes, weights = gauss_rule(order)
    u = (nodes+1)/2
    if rounded:
        return a+(b-a)*(1-np.cos(math.pi*u))/2, weights*(b-a)*math.pi*np.sin(math.pi*u)/4
    return a+(b-a)*u, weights*(b-a)/2


def breaks(a, b, candidates):
    # Periodic translations of the same intersection can differ by a few
    # ulps. Merge these internal splits, preserving both physical endpoints
    # and the full interval measure (no discarded slivers or normalization).
    tolerance = 8*np.finfo(float).eps*max(1., abs(a), abs(b))
    result = [a]
    for x in sorted(float(x) for x in candidates if a < x < b):
        if x-result[-1] > tolerance and b-x > tolerance: result.append(x)
    return [*result, b]


def angular_intervals(s1, s2, radius, x1):
    """Allowed x2 intervals after imposing all three periodic pair disks."""
    a2 = math.sqrt(max(0., radius*radius-s2*s2))
    a12 = math.sqrt(max(0., radius*radius-(s1+s2)**2))
    allowed = [(a12, TAU-a12)]
    if a2:
        for shift in (-TAU, 0., TAU):
            left, right = x1-a2+shift, x1+a2+shift
            new = []
            for a, b in allowed:
                if right <= a or left >= b:
                    new.append((a, b))
                else:
                    if a < left: new.append((a, left))
                    if right < b: new.append((right, b))
            allowed = new
    return allowed


def angular_rule(s1, s2, radius, order, *, x1_bounds=(0., TAU), x2_bounds=(0., TAU)):
    """Disjoint angular panels, including periodic wrap and all intersections.

    Optional bounds select a physical rectangular slice for independent
    comparison with old cells. Their weights are never renormalized.
    """
    if min(s1, s2, radius) <= 0 or radius >= math.pi:
        raise ValueError('positive lengths and radius < pi required')
    a1 = math.sqrt(max(0., radius*radius-s1*s1))
    a2 = math.sqrt(max(0., radius*radius-s2*s2))
    a12 = math.sqrt(max(0., radius*radius-(s1+s2)**2))
    left, right = max(a1, x1_bounds[0]), min(TAU-a1, x1_bounds[1])
    if left >= right: return
    low, high = max(a12, x2_bounds[0]), min(TAU-a12, x2_bounds[1])
    if low >= high: return
    # Every change of clipped interval topology is a panel boundary.
    edges = breaks(left, right, [math.pi, *[v-sign*a2-shift
        for v in (low, high, math.pi) for sign in (-1, 1) for shift in (-TAU, 0., TAU)]])
    for a, b in zip(edges[:-1], edges[1:]):
        xs, ws = interval(a, b, order)
        for x1, w1 in zip(xs, ws):
            for c, d in angular_intervals(s1, s2, radius, float(x1)):
                c, d = max(c, low), min(d, high)
                if c >= d: continue
                cuts = breaks(c, d, (math.pi,))
                for e, f in zip(cuts[:-1], cuts[1:]):
                    ys, vs = interval(e, f, order)
                    for x2, w2 in zip(ys, vs):
                        yield float(x1), float(x2), float(w1*w2)
