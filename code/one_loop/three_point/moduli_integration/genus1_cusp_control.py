"""Matched native CFT control on a globally owned compact cusp ray.

Only the full twist and infinite length of the CONTROL are integrated.
The finite torus remainder keeps the original matter factor. Every graph
guard below holds uniformly along the ray, by monotonicity in |q_loop|.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from types import SimpleNamespace
import cmath
import math
import numpy as np

from genus1_moduli_sampling import TAU, ShortPoint, CuspPoint, positive, cylinder_displacement, control_window, regular_mask
from genus1_native_log_evaluation import block_logs, common_log, koba_log, kinematics
from genus1_native_bank import identity
from genus1_comb_collar import _difference
from genus1_nonseparating import log_length_kernel, _sum_logs
from genus1_three_point_channel_atlas import pair_disc_to_flat_log_factor
from genus1_nonseparating import _log_abs_two_sine
from genus1_sewing_domains import pair_chart, comb_chart, necklace_chart, SewingDomainError


@dataclass(frozen=True)
class ControlConfig:
    tail_start: float = 8.
    onset: float = 8.
    inner: float = math.pi
    outer: float = TAU
    pair_radius: float = .01*TAU
    cut_level: int = 0

    def __post_init__(self):
        for key in ('tail_start','onset','inner','outer','pair_radius'):
            positive(getattr(self,key),key)
        if self.onset < self.tail_start or self.inner >= self.outer:
            raise ValueError('control onset/window are inconsistent with the cusp')
        if self.tail_start < 1:
            raise ValueError('the complete modular cusp strip requires tail_start >= 1')
        if self.outer > math.pi*self.onset:
            raise ValueError('compact ray must have S < L/2 at its onset')
        if self.pair_radius >= min(math.pi,TAU*self.tail_start/3):
            raise ValueError('collision mask does not have a uniform cylinder proof')
        if type(self.cut_level) is not int or self.cut_level < 0:
            raise ValueError('cut level must be a nonnegative integer')


@dataclass(frozen=True)
class SewingRay:
    short: ShortPoint
    channel: str
    pair: object
    axis: int
    native_short: float
    logs_at_onset: tuple
    onset: float
    constant: float
    external_weights: tuple
    external_momenta: tuple
    margins: tuple
    sine: float
    gaussian: float

    @property
    def lower_native(self):
        return TAU*self.onset-self.native_short

    def kin(self, L, theta):
        if not math.isfinite(L) or L < TAU*self.onset-1e-11:
            raise ValueError('point is below the certified sewing ray')
        logs = list(self.logs_at_onset)
        logs[self.axis] = complex(-(L-self.native_short), logs[self.axis].imag+theta)
        return dict(channel=self.channel, pair=self.pair, coefficient_basis='native-sewing-v2',
            external_weights=self.external_weights, external_momenta=self.external_momenta,
            logs=tuple(logs), variables=tuple(cmath.exp(z) for z in logs),
            decay=np.asarray([-2*z.real for z in logs]), constant=self.constant,
            chart_margins=self.margins)


def _matter(atlas, short):
    w1,w2,_ = short.positions()
    io = atlas.necklace.omega_in*atlas.necklace.omega_out
    oo = -atlas.necklace.omega_out**2
    sine = io*(_log_abs_two_sine(w1)+_log_abs_two_sine(w2))
    sine += oo*_log_abs_two_sine(w1-w2)
    gaussian = -.5*(io*(w1.imag**2+w2.imag**2)+oo*(w1.imag-w2.imag)**2)
    if abs(sine.imag)+abs(gaussian.imag) > 1e-11 or gaussian.real < -1e-11:
        raise ValueError('the implemented analytic kernel requires imaginary equal-split energies')
    return float(sine.real), max(0.,float(gaussian.real))


def make_ray(atlas, short, onset, channel, pair=None):
    """Construct a fixed native graph; no representative-twist acceptance.

    S<L/2 fixes every nearest displacement to its cylinder representative.
    Pair/comb contour inequalities depend only on fixed short variables and
    the decreasing |q_loop|, so their worst value is exactly at the onset.
    Necklace gaps are strictly positive throughout the same ray.
    """
    L = TAU*positive(onset,'onset')
    if short.total >= L/2:
        raise SewingDomainError('outside the uniform compact-ray construction')
    p = short.positions()
    if channel == 'necklace':
        if atlas.external_momenta[0] != atlas.external_momenta[1]:
            raise ValueError('the current necklace bank ordering requires equal outgoing operators')
        j = short.sector.index(2)
        labels = tuple(short.sector[(j+k) % 3] for k in range(3))
        logs = []
        axis = None
        for k in range(3):
            a,b = labels[k],labels[(k+1) % 3]
            z = p[b]-p[a]
            if a == short.sector[2]:
                z += 1j*L; axis=k
            logs.append(1j*z)
        chart = necklace_chart(logs).require()
        native_short = short.total
        d, momenta = atlas.external_weights, atlas.external_momenta
        constant = 0.
    else:
        if pair not in tuple(permutations(range(3),2)):
            raise ValueError('ordered insertion pair required')
        a,b = pair; spectator = next(i for i in range(3) if i not in pair)
        delta = cylinder_displacement(p[a]-p[b])
        total = cylinder_displacement(p[spectator]-p[b])
        d = tuple(atlas.external_weights[i] for i in (*pair,spectator))
        momenta = tuple(atlas.external_momenta[i] for i in (*pair,spectator))
        v = complex(np.expm1(-1j*delta))
        flat = pair_disc_to_flat_log_factor(delta,d[0],d[1])
        if channel == 'pair_ope':
            loop = -total
            if loop.imag < 0:
                loop += 1j*L; axis=1
            else:
                axis=2
            logs = (cmath.log(v),1j*loop,1j*(1j*L-loop))
            chart = pair_chart(delta,logs[1],logs[2]).require()
            native_short = -logs[3-axis].real
            constant = 2*math.log(abs(v))+2*flat.real
        elif channel == 'comb_ope':
            y = complex(np.expm1(-1j*total))
            chart = comb_chart(delta,total,complex(-L)).require()
            logs = (cmath.log(v/y),cmath.log(y),complex(-L))
            second = .5j*(1-d[2])*total-(1+d[2])*cmath.log(2*cmath.sin(total/2))
            constant = 2*math.log(abs(v))+2*math.log(abs(y))+2*(flat+second).real
            axis,native_short = 2,0.
        else:
            raise ValueError('unknown graph')
    sine,gaussian = _matter(atlas,short)
    return SewingRay(short,channel,pair,axis,float(native_short),tuple(logs),float(onset),
        float(constant),tuple(d),tuple(momenta),tuple(chart.margins),sine,gaussian)


def select_full_ray(atlas, short, tail_start):
    """Full-F routing depends on the physical cusp, NEVER the control window.

    This keeps the finite retained F identical during onset/window checks.
    Short-variable size chooses a graph; it is not a convergence estimate.
    """
    if short.total >= math.pi*tail_start:
        return None
    candidates = []
    for channel in ('necklace','pair_ope','comb_ope'):
        for pair in ([None] if channel == 'necklace' else permutations(range(3),2)):
            try:
                ray = make_ray(atlas,short,tail_start,channel,pair)
            except (SewingDomainError,ZeroDivisionError):
                continue
            sizes = sorted((z.real for i,z in enumerate(ray.logs_at_onset) if i != ray.axis),reverse=True)
            candidates.append((tuple(sizes),channel,pair or (),ray))
    return min(candidates,key=lambda r:r[:3])[-1] if candidates else None


def select_ray(atlas, short, config):
    """Restrict the fixed full-F graph to the matched control support/ray."""
    if not regular_mask(short,config.pair_radius,config.tail_start):
        return None
    if control_window(short,config.inner,config.outer) == 0:
        return None
    full=select_full_ray(atlas,short,config.tail_start)
    return None if full is None else make_ray(atlas,short,config.onset,full.channel,full.pair)


def prepare_point(atlas, point, config, *, forced_ray=None):
    """Return full kinematics and exactly matched active-control metadata."""
    if point.total < TAU*config.tail_start-1e-11:
        raise ValueError('sample lies outside the physical cusp')
    if not regular_mask(point.short,config.pair_radius,config.tail_start):
        return None
    full_ray = forced_ray if forced_ray is not None else select_full_ray(atlas,point.short,config.tail_start)
    weight = control_window(point.short,config.inner,config.outer)
    active = full_ray is not None and weight > 0 and point.total >= TAU*config.onset
    ray = None
    if full_ray is not None:
        if full_ray.short != point.short:
            raise ValueError('control belongs to a different short geometry/onset')
        kin = full_ray.kin(point.total,point.theta)
        axis = full_ray.axis
        if active:
            ray=make_ray(atlas,point.short,config.onset,full_ray.channel,full_ray.pair)
    else:
        p = point.torus_positions()
        choice = atlas.choose_channel(p[0],p[1],point.tau)
        kin = kinematics(atlas,p[0],p[1],point.tau,choice.channel,choice.pair)
        eligible = {'necklace':(0,1,2),'pair_ope':(1,2),'comb_ope':(2,)}[kin['channel']]
        axis = min(eligible,key=lambda i:kin['logs'][i].real)
    p = point.short.positions()
    kin['common_log'] = common_log(point.tau)+koba_log(atlas,p[0],p[1],point.tau)
    return dict(kin=kin,ray=ray,axis=axis,control_weight=weight if active else 0.)


def block_orders(short_order, loop_order, axis):
    if any(type(n) is not int or n < 0 for n in (short_order,loop_order)):
        raise ValueError('nonnegative integer block orders required')
    result = [short_order]*3; result[axis] = loop_order
    return tuple(result)


def prefixes(orders, axis):
    """Independent axes, odd/intermediate orders, and rectangular prefixes."""
    result = [('full',tuple(orders))]
    short_axes = [i for i in range(3) if i != axis]
    for label,i in [('short0',short_axes[0]),('short1',short_axes[1]),('loop',axis)]:
        for n in sorted({orders[i]-d for d in (1,2,4) if orders[i]-d >= 0},reverse=True):
            o = list(orders); o[i] = n; result.append((f'{label}_{n}',tuple(o)))
    if min(orders[i] for i in short_axes) >= 2:
        o = list(orders)
        for i in short_axes: o[i] -= 2
        result.append(('short_rectangle_minus2',tuple(o)))
    if min(orders[i] for i in short_axes) >= 4:
        o = list(orders)
        for i in short_axes: o[i] -= 4
        result.append(('short_low',tuple(o)))
    return result


def _require(bank, kin, orders):
    if identity(kin) != bank.metadata['identity']:
        raise ValueError('bank and evaluation graph/external weights differ')
    if len(orders) != 3 or any(type(n) is not int or n < 0 or n > m for n,m in zip(orders,bank.orders)):
        raise ValueError('requested tensor prefix is missing from the bank')


def _nodes(logs):
    logs = np.asarray(logs,complex)
    if np.any(np.isnan(logs)) or np.any(np.isposinf(logs.real)) or np.any(logs.real > 700):
        raise FloatingPointError('spectral node is outside the supported range')
    return np.exp(logs)


def evaluate_remainder(bank, prepared, config, orders):
    """Full/C subtraction at identical momentum nodes, before summation.

    Returns node-absolute block changes as well as the signed values. No
    conformal-block or momentum acceptance flag is inferred from smallness.
    """
    kin,ray,axis = prepared['kin'],prepared['ray'],prepared['axis']
    _require(bank,kin,orders)
    ps2 = bank.arrays['momenta']**2
    base = bank.arrays['structure_logs']+bank.arrays['log_weights']-3*math.log(math.pi)+kin['constant']
    base = base + sum(2*ps2[:,i]*z.real for i,z in enumerate(kin['logs']))
    full_base = base+kin['common_log']
    rows, first = [], None
    for label,o in prefixes(orders,axis):
        logblock,bound = block_logs(bank,o,kin['logs'])
        f = full_base+logblock
        if ray is None or prepared['control_weight'] == 0:
            c = np.full(len(f),complex(-math.inf)); r=f
            cbound = c
        else:
            level = min(config.cut_level,o[axis])
            co = list(o); co[axis]=level
            lc,bc = block_logs(bank,tuple(co),kin['logs'])
            L = -kin['logs'][axis].real+ray.native_short
            leading = .5*math.log(TAU/L)+ray.sine+ray.gaussian/L
            control_base = base+leading+math.log(prepared['control_weight'])
            c = control_base+lc; cbound=control_base+bc
            if np.max(abs(f.imag-c.imag)) > 1e-11:
                raise ArithmeticError('matched control has an inconsistent complex phase')
            lr,sign = _difference(f.real,c.real)
            r = lr+1j*(f.imag+np.where(sign < 0,math.pi,0.))
        fv,cv,rv = _nodes(f),_nodes(c),_nodes(r)
        if first is None: first=(fv,cv,rv)
        rounding = 128*np.finfo(float).eps*(1+sum(abs(z) for z in kin['logs']))
        rounding *= _sum_logs(full_base.real+bound)[1]+_sum_logs(cbound.real)[1]
        rows.append(dict(label=label,orders=o,full=_sum_logs(f)[0],control=_sum_logs(c)[0],
            value=_sum_logs(r)[0],node_absolute=float(math.fsum(abs(rv))),
            full_node_absolute=float(math.fsum(abs(fv))),control_node_absolute=float(math.fsum(abs(cv))),
            node_absolute_block_change=float(math.fsum(abs(rv-first[2]))),
            full_node_absolute_block_change=float(math.fsum(abs(fv-first[0]))),
            arithmetic_indicator=float(rounding)))
    return rows


def evaluate_addback(bank, ray, config, orders):
    """Integral d ell_native and AVERAGE d theta/(2pi); no moduli Jacobian.

    Native length is L-S_g. Keeping its own lower limit and S_g in K
    automatically includes the empty-gap/native-length conversion. After
    full twist integration only equal holomorphic/antiholomorphic cut levels
    survive; short descendant sums retain their full interference.
    """
    kin = ray.kin(TAU*config.onset,0.)
    _require(bank,kin,orders)
    weight = control_window(ray.short,config.inner,config.outer)
    if not regular_mask(ray.short,config.pair_radius,config.tail_start) or weight == 0:
        raise ValueError('addback requested outside its matched control support')
    ps2 = bank.arrays['momenta']**2
    base = (bank.arrays['structure_logs']+bank.arrays['log_weights']-3*math.log(math.pi)
            +ray.constant+math.log(weight))
    base += sum(2*ps2[:,i]*z.real for i,z in enumerate(kin['logs']) if i != ray.axis)
    rows,first = [],None
    for label,o in prefixes(orders,ray.axis):
        values = np.zeros(len(ps2),complex); absolute=0.
        for n in range(min(config.cut_level,o[ray.axis])+1):
            sel = [slice(None)]*4; sel[ray.axis+1] = slice(n,n+1)
            projected = SimpleNamespace(arrays={'coefficients':bank.arrays['coefficients'][tuple(sel)]})
            co = list(o); co[ray.axis]=0
            lb,bound = block_logs(projected,tuple(co),kin['logs'])
            kernel = log_length_kernel(np.sqrt(ps2[:,ray.axis]+n),lower_length=ray.lower_native,
                short_length=ray.native_short,coupling=math.sqrt(2*ray.gaussian),log_prefactor=ray.sine)
            values += _nodes(base+lb+kernel)
            absolute += _sum_logs(base.real+bound+kernel)[1]
        if first is None: first=values
        total = complex(math.fsum(values.real),math.fsum(values.imag))
        rows.append(dict(label=label,orders=o,value=total,node_absolute=float(math.fsum(abs(values))),
            node_absolute_block_change=float(math.fsum(abs(values-first))),
            arithmetic_indicator=float(128*np.finfo(float).eps*absolute)))
    return rows
