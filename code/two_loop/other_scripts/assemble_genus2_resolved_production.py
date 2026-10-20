#!/usr/bin/env python3
"""Absolute BRY-coupling comparison; no fitted amplitude, no hidden missing band.

The main curve is the resolved part of the full free energy. The residual
integral on the incomplete band and below rho=.001 is an explicit unknown.
The earlier old-sample control-variate curve is retained as a diagnostic,
not used to impute the missing new band or to alter the plotted normalization.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

import numpy as np

from c1_paths import data_root
from c1_frozen_bundle import verify_bundle
import genus2_separating_residual_pilot as pilot
from genus2_integrand_normalization import string_note_integration_kernel_target
ROOT = data_root('two_loop')
PARENT = ROOT / 'production/parent'
RETRY = ROOT / 'production/retry'
SOURCE = ROOT / 'assembly_v6'
HERE = ROOT / 'reproduced_assembly_v6'
REBUILT = None


def matrix_model_bry(radius):
    r = np.asarray(radius, float)
    if np.any(~np.isfinite(r)) or np.any(r <= 0):
        raise ValueError('radii must be finite and positive')
    return math.pi**2 / 1440 * (7*r + 10/r + 7/r**3)


def independent_group_mean_covariance(groups):
    groups = [np.asarray(group, float) for group in groups]
    if not groups or any(g.ndim != 2 or len(g) < 2 or np.any(~np.isfinite(g)) for g in groups):
        raise ValueError('invalid independent replicate groups')
    if len({g.shape[1] for g in groups}) != 1:
        raise ValueError('different radius grids')
    mean = sum(group.mean(axis=0) for group in groups)
    covariance = sum(np.atleast_2d(np.cov(group, rowvar=False, ddof=1)) / len(group) for group in groups)
    return mean, covariance


def assemble():
    p, manifest, policy, nodes = pilot.load(PARENT)
    legacy = pilot
    with np.load(REBUILT if REBUILT is not None else p / 'inputs/reuse_arrays.npz') as stored:
        arrays = {key: stored[key].copy() for key in stored.files}
    radii = arrays['radii']
    values = np.full((len(nodes), len(radii)), np.nan)
    records = []
    input_hashes = {}
    failed = []
    def checked_residual(record, node):
        result = np.asarray(record['weighted_residual'], float)
        if result.shape != radii.shape or np.any(~np.isfinite(result)):
            raise ValueError('invalid residual dimensions/value')
        if not node['in_domain']:
            if not record.get('domain_rejected') or np.any(result != 0):
                raise ValueError('invalid rejected-domain weight')
        else:
            full = np.asarray(record['cft']['scaled_kernel'])
            control = np.asarray(record['control_scaled'])
            recomputed = pilot.weighted_residual(full[None, :], control[None, :], [node])[0]
            np.testing.assert_allclose(result, recomputed, rtol=1e-11, atol=1e-16)
        return result
    for i, node in enumerate(nodes):
        path = SOURCE / 'pilot_results/nodes' / (node['node_id'] + '.json')
        record = legacy.load_record(path, dict(bundle_sha256=manifest['bundle_sha256'],
            node_id=node['node_id'], node_sha256=node['node_sha256']), require_ok=False)
        input_hashes[str(path.relative_to(SOURCE))] = legacy.sha(path)
        records.append(record)
        if record['status'] == 'ok':
            values[i] = checked_residual(record, node)
        else:
            failed.append(i)
    if failed != [3, 16]:
        raise ValueError('unexpected original failures; review assembly policy')

    retry_manifest = verify_bundle(RETRY)
    path = SOURCE / 'retry_results/node.json'
    retry = legacy.load_record(path, dict(bundle_sha256=retry_manifest['bundle_sha256'],
        parent_bundle_sha256=manifest['bundle_sha256'], node_id=nodes[3]['node_id'], node_sha256=nodes[3]['node_sha256']))
    input_hashes[str(path.relative_to(SOURCE))] = legacy.sha(path)
    original_path = SOURCE / 'pilot_results/nodes' / (nodes[3]['node_id'] + '.json')
    if retry['original_result_sha256'] != legacy.sha(original_path):
        raise ValueError('retry does not correspond to this original failure')
    np.testing.assert_array_equal(retry['radii'], radii)
    for key, tolerance in [('block_relative_step', 'block_tolerance'), ('momentum_relative_step', 'momentum_tolerance')]:
        if retry['higher_order_checks'][key] > policy['numerical_policy'][tolerance]:
            raise ValueError('higher-order retry gate failed')
    if retry['final_momentum_relative_step'] > policy['numerical_policy']['momentum_tolerance']:
        raise ValueError('retry confirmation failed')
    values[3] = checked_residual(retry, nodes[3])

    group_map = dict(old_bulk_plus_outer_residual=arrays['old_base'],
                     integrated_control=arrays['integrated_primary'],
                     negative_domain_correction=-arrays['domain_correction'])
    band_info = []
    for band, (high, low) in enumerate(zip(policy['band_edges'], policy['band_edges'][1:])):
        ids = [i for i, node in enumerate(nodes) if node['band'] == band]
        finite = [i for i in ids if np.all(np.isfinite(values[i]))]
        complete = len(finite) == len(ids)
        info = dict(band=band, rho_low=low, rho_high=high, resolved=len(finite), nominal=len(ids), complete=complete)
        if complete:
            reps = []
            for rep in range(policy['replicates']):
                chosen = [i for i in ids if nodes[i]['replicate'] == rep]
                if len(chosen) != 2**policy['power']:
                    raise ValueError('missing nominal point')
                reps.append(values[chosen].sum(axis=0) / 2**policy['power'])
            group_map[f'residual_band_{band}'] = np.array(reps)
        else:
            info['status'] = 'entire band residual remains unknown; no successful-only averaging or old-sample imputation'
        band_info.append(info)
    if [b['band'] for b in band_info if b['complete']] != [0, 1, 3, 4]:
        raise ValueError('unexpected complete bands')

    known_xi, covariance_xi = independent_group_mean_covariance(list(group_map.values()))
    # This old group preserves all cross-covariances within the original scrambles.
    old_group = arrays['old_base'] + arrays['old_band_residual'].sum(axis=0) + arrays['old_deep_residual']
    previous_xi, previous_covariance_xi = independent_group_mean_covariance(
        [old_group, arrays['integrated_primary'], -arrays['domain_correction']])
    old_cache = np.load(p / 'inputs/old_primary_cache.npz')
    direct_previous = (old_cache['original'] - old_cache['rho_0.08_sampled_primary']).mean(axis=0)
    direct_previous += old_cache['rho_0.08_integrated_primary'].mean(axis=0) - old_cache['rho_0.08_domain_correction'].mean(axis=0)
    np.testing.assert_allclose(previous_xi, direct_previous, rtol=2e-13, atol=1e-13)
    np.testing.assert_allclose(previous_covariance_xi, old_cache['rho_0.08_mean_covariance'], rtol=1e-10, atol=1e-13)
    old_cache.close()

    # Exact coupling conversion, not an amplitude fit: g_B = 2 g_X.
    known_bry = known_xi / 4
    covariance_bry = covariance_xi / 16
    se_bry = np.sqrt(np.maximum(0., np.diag(covariance_bry)))
    target = matrix_model_bry(radii)
    np.testing.assert_allclose(target, np.array([string_note_integration_kernel_target(r) for r in radii])/4, rtol=3e-15)
    np.testing.assert_allclose(matrix_model_bry(1), math.pi**2/60, rtol=3e-15)
    one = int(np.flatnonzero(radii == 1)[0])
    table = [dict(radius=float(r), resolved_worldsheet_bry=float(known_bry[j]), empirical_SE=float(se_bry[j]),
                  matrix_model_bry=float(target[j]), ratio=float(known_bry[j]/target[j]),
                  ratio_empirical_SE=float(se_bry[j]/target[j]),
                  prior_control_variate_pilot_bry=float(previous_xi[j]/4),
                  resolved_minus_prior_bry=float((known_xi[j]-previous_xi[j])/4)) for j, r in enumerate(radii)]
    selected = [next(row for row in table if row['radius'] == r) for r in (.5, 1., 2., 5., 10., 15.)]
    stats = dict(status='provisional_resolved_contribution_not_complete_free_energy', certified=False,
        full_free_energy_available=False, created_utc=datetime.now(timezone.utc).isoformat(),
        parent_bundle_sha256=manifest['bundle_sha256'], retry_bundle_sha256=retry_manifest['bundle_sha256'],
        input_sha256=input_hashes, normalization=dict(coupling='g_B = 2 g_X; mu^-1 = 2 pi g_B',
            worldsheet_coefficient_bare_J2='16*pi^4', old_to_bry=0.25,
            matrix_model='pi^2/1440*(7*R + 10/R + 7/R^3)', ensemble='one-sided singlet; connected vacuum functional',
        alpha_prime=1., radius_definition='R_phys/sqrt(alpha_prime)', fitted_amplitude=False, normalized_at_R1=False,
            absolute_sewing_status='Working coefficient; independent absolute-measure certification is still open'),
        original_successes=38, successful_retry_index=3, resolved_count=39, band_status=band_info,
        estimator='old bulk+outer residual + independent integral_D G + NEW complete residual bands 0,1,3,4',
        equation='F2/g_B^2 = plotted resolved contribution + Delta_[0.005,0.01) + Delta_[0,0.001)',
        unknown_terms='Both Delta terms are integrals of (K-G)/4 in the indicated rho ranges, not zero and not bounded here.',
        refined_node_policy='Index 3 uses the successful B10 N=16 retry; all other new values remain the parent B8 baseline.',
        uncertainty_scope='Empirical covariance of independent scramble means only; excludes CFT truncation, control bias, rare unsampled domain effects, unresolved residuals and normalization systematics.',
        selected_radii=selected, ratio_min=float(np.min(known_bry/target)), ratio_max=float(np.max(known_bry/target)),
        ratio_distance_from_two_max_percent=float(np.max(np.abs(known_bry/target/2-1))*100),
        maximum_known_vs_previous_relative_change=float(np.max(np.abs(known_xi/previous_xi-1))),
        known_R1=float(known_bry[one]), previous_R1=float(previous_xi[one]/4),
        ratio_R1=float(known_bry[one]/target[one]), ratio_R1_empirical_SE=float(se_bry[one]/target[one]),
        references=['https://arxiv.org/html/1705.07151v4#S3.E8', 'https://arxiv.org/pdf/hep-th/9108019#page=56',
                    'Machine note/two_loop/c1_bry_conventions.md'])
    legacy.atomic_json(HERE/'summary.json', stats)
    legacy.atomic_json(HERE/'radius_values.json', table)
    np.savez_compressed(HERE/'assembled_arrays.npz', radii=radii, resolved_worldsheet_bry=known_bry,
        covariance_bry=covariance_bry, matrix_model_bry=target, previous_pilot_bry=previous_xi/4,
        previous_covariance_bry=previous_covariance_xi/16, **group_map)
    return stats, table






def main():
    import argparse, shutil
    global HERE, REBUILT
    p=argparse.ArgumentParser(description='Reassemble complete residual bands and bulk/control groups from frozen production inputs.')
    p.add_argument('--output',type=Path,default=HERE);p.add_argument('--rebuilt',type=Path);a=p.parse_args();HERE=a.output.resolve();REBUILT=a.rebuilt
    HERE.mkdir(parents=True,exist_ok=False)
    # Provenance inputs travel with the intermediate adapter output.
    for name in ('pilot_results','retry_results'):shutil.copytree(SOURCE/name,HERE/name)
    stats,table=assemble()
    with np.load(SOURCE/'assembled_arrays.npz') as expected, np.load(HERE/'assembled_arrays.npz') as actual:
        for key in expected.files:np.testing.assert_allclose(actual[key],expected[key],rtol=2e-13,atol=1e-14)
    print(json.dumps(dict(output=str(HERE),resolved_count=stats['resolved_count'],all_saved_arrays_reproduced=True),indent=2))

if __name__=='__main__':main()
