"""Range-safe evaluation of the SAME finite native-q polynomial.

The logarithms below are logs of the raw sewing variables, not elliptic
coordinates. No coefficient, integration measure, or retained term changes.
This versioned adapter leaves all previously frozen implementations untouched.
"""
import cmath
import math
import numpy as np

from genus1_native_bank import identity
from genus1_three_point_channel_atlas import (
    nearest_torus_displacement, canonical_loop_displacement,
    pair_disc_to_flat_log_factor,
)
from genus1_three_point_collision import koba_exponent
from genus1_three_point_worldsheet import (
    ordered_necklace_data, dedekind_eta_oscillator_abs_squared,
    torus_prime_form_log_norm,
)


def common_log(tau):
    return math.log(dedekind_eta_oscillator_abs_squared(tau))-.5*math.log(tau.imag)


def koba_log(atlas, w1, w2, tau):
    return sum(koba_exponent(atlas, a, b)*torus_prime_form_log_norm(z, tau)
               for a, b, z in ((0, 1, w1-w2), (0, 2, w1), (1, 2, w2)))


def kinematics(atlas, w1, w2, tau, channel, pair=None, *, include_common=True):
    """Same native chart and prefactors, without exp(log(Koba)) round-trip."""
    w1, w2, tau = complex(w1), complex(w2), complex(tau)
    chart = atlas.sewing_chart(w1, w2, tau, channel, pair).require()
    if channel == 'necklace':
        if atlas.necklace.external_momenta[0] != atlas.necklace.external_momenta[1]:
            raise ValueError('equal outgoing momenta required by necklace ordering')
        logs = ordered_necklace_data(w1, w2, tau)[1]
        d = atlas.necklace.external_weights; momenta = atlas.necklace.external_momenta
        constant = 0.
    else:
        first, second = pair; spectator = next(i for i in range(3) if i not in pair)
        points = (w1, w2, 0j)
        delta = nearest_torus_displacement(points[first]-points[second], tau)
        v = complex(np.expm1(-1j*delta))
        d = tuple(atlas.external_weights[i] for i in (*pair, spectator))
        momenta = tuple(atlas.external_momenta[i] for i in (*pair, spectator))
        flat = pair_disc_to_flat_log_factor(delta, d[0], d[1])
        if channel == 'pair_ope':
            loop = canonical_loop_displacement(points[second]-points[spectator], tau)
            logs = (cmath.log(v), 1j*loop, 1j*(2*math.pi*tau-loop))
            constant = 2*math.log(abs(v))+2*flat.real
        elif channel == 'comb_ope':
            total = nearest_torus_displacement(points[spectator]-points[second], tau)
            y = complex(np.expm1(-1j*total))
            logs = (cmath.log(v/y), cmath.log(y), 2j*math.pi*tau)
            second_flat = .5j*(1-d[2])*total-(1+d[2])*cmath.log(2*cmath.sin(total/2))
            constant = 2*math.log(abs(v))+2*math.log(abs(y))+2*(flat+second_flat).real
        else:
            raise ValueError('unknown native chart')
    if any(not np.isfinite(z) or z.real >= 0 for z in logs):
        raise ValueError('noncontracting native sewing logarithms')
    return dict(channel=channel, pair=pair, coefficient_basis='native-sewing-v2',
        external_weights=d, external_momenta=momenta, logs=logs,
        variables=tuple(cmath.exp(z) for z in logs), decay=np.asarray([-2*z.real for z in logs]),
        constant=float(constant), common_log=(common_log(tau)+koba_log(atlas, w1, w2, tau)
                                             if include_common else None),
        chart_margins=chart.margins)


def block_logs(bank, cut, logs):
    """Return log|sum C_ijk q^n|^2 and its absolute-term bound per node.

    Scaling is AFTER combining log|C| with the monomial logarithm. Thus a
    tiny power multiplied by a large coefficient is not lost prematurely.
    Underflow only occurs relative to the largest term in that same sum;
    the rounding allowance in the caller dominates these <exp(-700) losses.
    Exactly projected zero variables are represented by -inf logarithms.
    """
    if hasattr(bank, 'covers') and not bank.covers(cut):
        raise ValueError('native block request exceeds the saved coefficient region')
    if all((z.real == -math.inf and z.imag == 0) or
           (np.isfinite(z) and n*z.real > -650) for n, z in zip(cut, logs)):
        from genus1_comb_collar import _block
        try:
            return _block(bank, cut, tuple(cmath.exp(z) for z in logs))
        except FloatingPointError:
            pass  # Retry the identical polynomial with termwise log scaling.
    c = bank.arrays['coefficients'][(slice(None), *[slice(n+1) for n in cut])]
    if not np.all(np.isfinite(c)):
        raise FloatingPointError('nonfinite coefficients')
    monomials = []
    for n, z in zip(cut, logs):
        z = complex(z)
        if z.real == -math.inf and z.imag == 0:
            row = np.full(n+1, complex(-math.inf)); row[0] = 0j
        elif np.isfinite(z):
            row = z*np.arange(n+1)
        else:
            raise ValueError('invalid raw sewing logarithm')
        monomials.append(row)
    powers = (monomials[0][:, None, None]+monomials[1][None, :, None]
              +monomials[2][None, None, :]).reshape(-1)
    c = c.reshape(len(c), -1)
    with np.errstate(divide='ignore'):
        magnitudes = np.log(abs(c))+powers.real
    peak = np.max(magnitudes, axis=1)
    active = np.isfinite(peak)
    result = np.full(len(c), -math.inf); bound = result.copy()
    if np.any(active):
        scaled = np.exp(magnitudes[active]-peak[active, None])
        phase = np.angle(c[active])+powers.imag
        value = np.sum(scaled*np.exp(1j*phase), axis=1)
        absolute = np.sum(scaled, axis=1)
        with np.errstate(divide='ignore'):
            result[active] = 2*(peak[active]+np.log(abs(value)))
            bound[active] = 2*(peak[active]+np.log(absolute))
    if np.any(np.isnan(result)) or np.any(np.isnan(bound)):
        raise FloatingPointError('invalid native logarithmic contraction')
    return result, bound


def projected_logs(variables):
    return tuple(cmath.log(z) if z != 0 else complex(-math.inf) for z in variables)


def reduce_retained(bank, g, components, signs, geometry_scale, cut):
    """Preserve a logarithmic absolute enclosure for genuinely tiny densities.

    Relative error is undefined for a floating-point zero. This branch is
    allowed only when the sum of ALL absolute retained parent terms is tiny;
    it cannot hide a cancellation of large parents or a missing CFT value.
    The caller must propagate this absolute enclosure with the MC Jacobian.
    """
    from genus1_comb_collar import _reduce
    from genus1_pair_collar import _logsum
    absolute = np.logaddexp.reduce(np.asarray([p[1] for p in components]),axis=0)
    enclosure = _logsum(g['base'].real+absolute)+math.log(2.)
    if math.isfinite(enclosure) and enclosure < -745:
        return 0j,dict(passed=True,reasons=[],parent_to_residual_condition=None,
            estimated_relative_arithmetic_loss=None,absolute_arithmetic_log_bound=float(enclosure),
            range_handling='absolute_retained_density_enclosure',subtraction_before_momentum_sum=True)
    return _reduce(bank,g,components,signs,geometry_scale,cut)
