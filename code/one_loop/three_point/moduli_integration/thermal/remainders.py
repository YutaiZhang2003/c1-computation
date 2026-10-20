"""Matched thermal collision subtractions at common CFT momentum nodes.

Every projection carries the compact-boson lattice at its own limiting
insertion positions. These evaluators never infer missing CFT coefficients.
"""
import genus1_source_paths as bootstrap
bootstrap.install()
import cmath
import math
import numpy as np
from kernels import lattice, pair_lattice
import genus1_comb_collar as comb
import genus1_pair_collar as pair_collar
from genus1_native_bank import variants, point_kinematics
from genus1_three_point_collision import CollisionRadii, reconstruct_points
from genus1_native_log_evaluation import block_logs, projected_logs


def _finish(values, checks, term, diagnostics):
    report = dict(term=term, passed=all(c['passed'] for c in checks), prefixes=checks,
                  termwise_thermal_lattices=True, subtraction_before_momentum_sum=True,
                  coefficient_recomputations=0, numerical_convergence_certified=False)
    if not diagnostics and not report['passed']:
        raise FloatingPointError('thermal subtraction failed arithmetic checks')
    values = np.asarray(values, complex)
    return (values, report) if diagnostics else values


def pair_remainder(bank, atlas, kin, *, pair, delta_pair, delta_total, tau,
                   radii, orders=None, diagnostics=False, noncompact=False):
    if bank.metadata.get('projection_only', False):
        raise ValueError('the pair parent requires a full CFT bank')
    cut_orders = bank.validate_orders(bank.orders if orders is None else orders)
    if not bank.covers(cut_orders):
        raise ValueError('missing pair descendant coefficients')
    common, exact, projection, sewing, geometry_scale = pair_collar._pair_remainder_geometry(
        bank, atlas, pair, delta_pair, delta_total, tau, radii)
    if max(np.max(abs(exact.imag)), np.max(abs(projection.imag))) > 2e-12:
        raise ValueError('signed-real residual requires real continued weights')
    points = reconstruct_points(tuple(pair), delta_pair, delta_total)
    lf = 0. if noncompact else math.log(lattice(kin, tau, (*points, 0j)))
    lp = 0. if noncompact else math.log(pair_lattice(kin, tau, pair, delta_total))
    values, checks = [], []
    for label, cut in variants(cut_orders):
        full, absolute_full = block_logs(bank, cut, sewing)
        zero, absolute_zero = block_logs(bank, (0, cut[1], cut[2]),
                                         (complex(-math.inf), *sewing[1:]))
        components = [(exact.real+full+lf, exact.real+absolute_full+lf),
                      (projection.real+zero+lp, projection.real+absolute_zero+lp)]
        value, check = comb._reduce(bank, dict(base=common), components, [1,-1], geometry_scale, cut)
        values.append(value); checks.append(dict(prefix=label, orders=list(cut), **check))
    return _finish(values, checks, 'thermal_pair_remainder', diagnostics)


def comb_term(bank, atlas, kin, *, term, pair, tau, scale, phase=0., shape=None,
              shape_upper=None, radii=None, orders=None, diagnostics=False, noncompact=False):
    """Full triple forest or integrated pair face, including lattice differences."""
    if term not in ('forest_remainder', 'pair_face'):
        raise ValueError('this evaluator handles the forest and pair-face remainder')
    if term == 'forest_remainder' and bank.metadata.get('projection_only', False):
        raise ValueError('the forest requires a full CFT bank')
    cut_orders = bank.validate_orders(bank.orders if orders is None else orders)
    if not bank.covers(cut_orders):
        raise ValueError('missing comb descendant coefficients')
    g = comb._geometry(bank, atlas, pair, tau)
    scale = comb._positive_radius(scale, 'triple scale')
    total = scale*cmath.exp(1j*phase)
    primary, v = comb._pair_log(g, total)
    lp = 0. if noncompact else math.log(pair_lattice(kin, tau, pair, total))
    lc = 0. if noncompact else math.log(lattice(kin, tau, (0j,0j,0j)))
    geometry_scale = max(1., 2*math.pi*abs(tau))/scale
    if term == 'forest_remainder':
        shape = comb._shape(shape)
        if (not isinstance(radii, CollisionRadii) or not scale < radii.triple
                or not scale*abs(shape) < radii.pair):
            raise ValueError('forest point must lie inside its collision sector')
        positions = reconstruct_points(tuple(pair), shape*total, total)
        cft = point_kinematics(atlas, dict(w1=comb.encoded(positions[0]),
            w2=comb.encoded(positions[1]), tau=comb.encoded(tau),
            channel='comb_ope', pair=list(pair)))
        exact = (cft['constant'] + (cft['common_log']-g['common']).real
                 - np.einsum('ti,i->t', bank.arrays['momenta'][:,:2]**2, cft['decay'][:2]))
        lf = 0. if noncompact else math.log(lattice(kin, tau, (*positions,0j)))
        local = g['gamma']*math.log(scale)+g['a']*math.log(abs(shape))
        geometry_scale /= abs(shape)
    else:
        if not 0 < shape_upper <= 1 or np.any(g['beta'] <= 0):
            raise ValueError('convergent shape integral required')
        radial = g['beta']*math.log(shape_upper)-np.log(g['beta'])
    values, checks = [], []
    def block(cut, variables):
        return block_logs(bank, cut, (*projected_logs(variables[:2]), 2j*math.pi*tau))
    def shift(b, factor):
        return b[0]+factor, b[1]+factor
    for label, cut in variants(cut_orders):
        q = g['q']
        pair_block = block((0,cut[1],cut[2]), (0j,v,q))
        corner = block((0,0,cut[2]), (0j,0j,q))
        if term == 'forest_remainder':
            full = shift(block(cut,cft['variables']), exact+lf)
            pair_part = shift(pair_block, primary+g['a']*math.log(scale*abs(shape))+lp)
            scale_part = shift(block((cut[0],0,cut[2]),(shape,0j,q)),
                               local+g['k'][1]*math.log(abs(1-shape))+lc)
            components = [full, pair_part, shift(corner,local+lc), scale_part]
            signs = [1,-1,1,-1]
        else:
            components = [shift(pair_block, primary+g['a']*math.log(scale)+radial+lp),
                          shift(corner, g['gamma']*math.log(scale)+radial+lc)]
            signs = [1,-1]
        value, check = comb._reduce(bank,g,components,signs,geometry_scale,cut)
        values.append(value); checks.append(dict(prefix=label,orders=list(cut),**check))
    return _finish(values, checks, 'thermal_'+term, diagnostics)


def tube_remainder(bank, atlas, kin, point, *, orders=None, maximum_short_nome=.8):
    """Matched native finite-tube descendants with their actual insertion lattice.

    The analytic addback for this residual MUST use this same native primary.
    It cannot be combined silently with an independently truncated H bank.
    """
    from genus1_boundary_banks import NativePrimary
    if bank.metadata.get('projection_only', False):
        raise ValueError('tube descendants require a full necklace bank')
    engine = NativePrimary(atlas,bank,orders=orders,maximum_short_nome=maximum_short_nome)
    value = engine.remainder(point)
    factor = lattice(kin,point.tau,point.torus_positions())
    return dict(noncompact=value.value, thermal=factor*value.value,
                correction=(factor-1)*value.value,
                block_indicator=factor*value.errors['block'],
                required_addback=engine.identity, numerical_convergence_certified=False)
