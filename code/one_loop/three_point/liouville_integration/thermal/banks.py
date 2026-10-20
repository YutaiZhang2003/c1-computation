"""Fresh thermal reference banks, using the existing resumable CFT builder.

The explicit reference route permits more than nine nodes. No production
budget is relaxed and no coefficient is moved to a different momentum node.
"""

from genus1_source_paths import ROOT as _SOURCE_ROOT, source_path as _source_path
import genus1_source_paths as bootstrap
bootstrap.install()
from pathlib import Path
import copy
import math
import numpy as np
from genus1_native_bank import source_manifest, digest
from genus1_three_point_channel_atlas import make_atlas
from genus1_projection_preparation import spec as projection_spec
from genus1_projection_momentum import primary_sampling_controls
from genus1_native_bank import identity
from genus1_three_point_momentum import fingerprint
from genus1_boundary_spectrum import publish, grid_directory, build, Banks
from genus1_moduli_sampling import ShortPoint
from quadrature import primary_rule, height_rule

FORMAT = 'genus1-thermal-reference-grid-v1'


def implementation_sources():
    folder = Path('code/one_loop/thermal')
    # These files determine the physical grid; evaluators have separate run pins.
    return {str(_source_path(p).relative_to(bootstrap.ROOT)): digest(_source_path(p)) for p in
            (folder/'bootstrap.py', folder/'kinematics.py', folder/'kernels.py',
             folder/'quadrature.py', folder/'banks.py')}


def make_spec(kin, case, order, *, elliptic_order=4, collision_h_order=8,
              construction_order=384, thermal=True, cut_only=False,
              base_rules=None, special_dps=40, chunk_size=32):
    """Freeze exact physical nodes, their dP weights and the CFT input identity."""
    if type(order) is not int or not 4 <= order <= 32:
        raise ValueError('reference momentum order must be in 4..32')
    case = copy.deepcopy(case)
    if case['kind'] == 'full':
        raise ValueError('full CFT remainders use prepare_full, without projected inputs')
    atlas = make_atlas(kin.t, special_dps=special_dps)
    tail = max(3, order//3); head = order-tail
    controls = dict(momentum_orders=[head]*3, tail_orders=[tail]*3,
                    width_scales=[1., 1., 1.])
    if case['kind'] == 'primary':
        controls['primary_sampling'] = primary_sampling_controls(dict(
            head_order=head, split_scale=2., tail_width_scale=.3))
    cft, design, rules = projection_spec(atlas, case,
        tuple(controls['momentum_orders']), tuple(controls['tail_orders']),
        primary_sampling=controls.get('primary_sampling'))
    if cut_only:
        if base_rules is None:
            raise ValueError('cut-only refinement requires explicit frozen non-cut rules')
        rules = copy.deepcopy(base_rules)
    kind = case['kind']
    if kind == 'primary':
        axis = design['cut_axis']; short = ShortPoint.from_record(case['geometry']['short'])
        threshold = -math.log(case['geometry']['policy']['q_max'])
        if thermal:
            rules[axis] = primary_rule(short, kin, design['axes'][axis]['dozz_threshold_slope'],
                order, threshold=threshold, construction_order=construction_order)
            design['axes'][axis]['endpoint_power'] = 0
        else:
            from genus1_nonseparating import kernel_weighted_cut_rule
            from genus1_analytic_boundary_layer import necklace_slice_geometry
            lower = max(threshold, short.s1, short.s2)
            g = necklace_slice_geometry(atlas, short)
            rules[axis] = kernel_weighted_cut_rule(dict(length=lower, minimum_length=lower,
                short_length=short.total, gaussian=g['gaussian']),
                design['axes'][axis]['dozz_threshold_slope'], order)
    elif kind in ('triple_scale_face', 'triple_corner'):
        axis = 2
        if thermal:
            g = case['geometry']
            rules[axis] = height_rule(kin, design['axes'][axis]['dozz_threshold_slope'], order,
                region=g['region'], tau1=g.get('tau1', .5), cutoff=g.get('cutoff', 8.),
                construction_order=construction_order)
            design['axes'][axis]['endpoint_power'] = rules[axis]['endpoint_power']
        # The bridge rule has no extra tail. Give it the full requested order.
        if not cut_only:
            from genus1_projection_momentum import radial_bridge_rule
            a = design['axes'][0]
            rules[0] = radial_bridge_rule(order, decay=a['decay'], mass_squared=a['mass_squared'])
    elif kind == 'pair' and not cut_only:
        from genus1_projection_momentum import radial_bridge_rule
        a = design['axes'][0]
        rules[0] = radial_bridge_rule(order, decay=a['decay'], mass_squared=a['mass_squared'])
    for r in rules:
        r['nodes'] = np.asarray(r['nodes']).tolist()
        r['weights'] = np.asarray(r['weights']).tolist()
        r['log_weights'] = np.log(r['weights']).tolist()
    sources = source_manifest()
    grid = dict(format=FORMAT, identity=identity(cft), rules=rules,
        coefficient_orders=case['coefficient_orders'], special_dps=special_dps,
        source_sha256=fingerprint(sources), elliptic_order=elliptic_order if kind=='primary' else None,
        cut_axis=design.get('cut_axis'), t=kin.t,
        collision_elliptic=(dict(h_order=collision_h_order, long_order=3)
                           if kind in ('pair', 'triple_scale_face') else None),
        thermal_kinematics=kin.record(), thermal_sources=implementation_sources(),
        reference_only=True, requested_order=order, cut_only=cut_only,
        new_CFT_evaluation_required=True)
    design.update(thermal=thermal, reference_only=True, requested_order=order,
                  actual_axis_counts=[len(r['nodes']) for r in rules])
    return dict(grid=grid, case=case, design=design, controls=controls,
                source_files=sources, chunk_size=chunk_size)


def register(output, spec):
    key = fingerprint(spec['grid'])
    publish(grid_directory(output, key)/'spec.json', spec)
    return key


def build_reference(output, key, *, cpu_budget=45., chunks=128, elliptic_nodes=4096):
    return build(output, dict(format=FORMAT), key, cpu_budget=cpu_budget,
                 max_new_chunks=chunks, max_new_elliptic_nodes=elliptic_nodes)


def primary_case(short, *, q_max=.05, pair_radius=.125, triple_radius=.5):
    return dict(id='bank', kind='primary', coefficient_orders=[0,0,0],
                geometry=dict(short=short.record(), policy=dict(q_max=q_max,
                    pair_radius=pair_radius, triple_radius=triple_radius)))


def height_case(kind='triple_corner', *, pair=(0,1), region='tail', tau1=.2,
                pair_radius=.125, triple_radius=.5, shape=.17+.21j, theta=.9):
    if kind not in ('triple_corner', 'triple_scale_face'):
        raise ValueError('height-integrated term required')
    g = dict(pair=list(pair), region=region, tau1=tau1, cutoff=8.,
             pair_radius=pair_radius, triple_radius=triple_radius)
    if kind == 'triple_corner':
        g['theta'] = theta; orders = [0,0,3]
    else:
        g['shape'] = [shape.real, shape.imag]; orders = [0,0,3]
    return dict(id='bank', kind=kind, coefficient_orders=orders, geometry=g)
