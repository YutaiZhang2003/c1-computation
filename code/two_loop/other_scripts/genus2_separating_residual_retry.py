#!/usr/bin/env python3
"""One-point higher-quadrature retry, using the immutable light-pilot CFT layer.

Never changes the original result or merges unlike-accuracy estimates silently.
The retry keeps the original G and nominal MC weight; only K is refined.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import numpy as np

PARENT_SHA = 'a59150b54f780f9d459013aa0106b6a690b8a5538870706ba28042648f6a055a'
NODE_ID = 'res-b00-r01-p00000001'
NODE_SHA = '6503d003637aa4981f4189ce4e289fb86cddf248268b95fa291170289c80cae5'
REMOTE_BASE = '/n/holylabs/yin_lab/Everyone/yutaizhang/StringMC/'


def parent_data(parent):
    import genus2_separating_residual_pilot as pilot
    parent, manifest, policy, nodes = pilot.load(parent)
    if manifest['bundle_sha256'] != PARENT_SHA:
        raise ValueError('not the expected frozen light pilot')
    node = nodes[3]
    if node['node_id'] != NODE_ID or node['node_sha256'] != NODE_SHA:
        raise ValueError('retry node identity mismatch')
    if node['support_reasons'] or not node['in_domain']:
        raise ValueError('retry does not bypass support/domain guards')
    return pilot, parent, policy, node


def order_schedule(original):
    orders = tuple(original)
    if len(orders) != 3 or any(not isinstance(n, int) or n < 1 for n in orders):
        raise ValueError('three positive integer momentum orders required')
    return tuple(tuple(n + delta for n in orders) for delta in (4, 6, 8))


def confirmation_step(previous, current, tolerance):
    if not all(math.isfinite(v) for v in (previous, current, tolerance)) or tolerance <= 0:
        raise ValueError('invalid confirmation input')
    step = abs(math.expm1(current - previous))
    if step > tolerance:
        raise RuntimeError(f'additional momentum confirmation failed: {step:.12g} > {tolerance:.12g}')
    return step




def load(parent, run):
    pilot, parent, policy, node = parent_data(parent)
    run = Path(run).resolve()
    from c1_frozen_bundle import verify_bundle
    manifest=verify_bundle(run)
    config = pilot.read_json(run / 'config.json')
    if (config['parent_bundle_sha256'], config['node_id'], config['node_sha256']) != (PARENT_SHA, NODE_ID, NODE_SHA):
        raise ValueError('retry config identity mismatch')
    if config['numerical_policy'] != policy['numerical_policy'] or config['momentum_offsets'] != [4, 6, 8]:
        raise ValueError('retry must retain parent tolerances and configured order schedule')
    return pilot, parent, run, manifest, policy, node


def work(parent, run):
    pilot, parent, run, manifest, policy, node = load(parent, run)
    legacy = pilot
    output = run / 'results/node.json'
    identity = dict(bundle_sha256=manifest['bundle_sha256'], parent_bundle_sha256=PARENT_SHA,
                    node_id=NODE_ID, node_sha256=NODE_SHA)
    if output.exists():
        return legacy.load_record(output, identity, require_ok=False)
    started = time.monotonic()
    result = dict(identity, status='failed', environment=legacy.runtime_environment(),
                  full_free_energy_available=False, certified=False,
                  estimator_matches_parent_baseline=False, merged_into_parent=False)
    def checkpoint(name, value):
        legacy.atomic_json(run / 'results/checkpoints' / (name + '.json'), dict(identity, value=value))
    try:
        old_path = parent / 'results/nodes' / (NODE_ID + '.json')
        old = legacy.load_record(old_path, dict(bundle_sha256=PARENT_SHA, node_id=NODE_ID, node_sha256=NODE_SHA), require_ok=False)
        if old['status'] != 'failed' or 'independent CFT-order gates failed' not in old.get('error', ''):
            raise ValueError('source is not the targeted failed CFT-order result')
        checkpoint('original_failure', old)
        result['original_result_sha256'] = legacy.sha(old_path)
        with np.load(parent / 'inputs/reuse_arrays.npz') as arrays:
            radii = arrays['radii'].copy()
        numerical = policy['numerical_policy']
        period = legacy.period_evaluation(node, numerical)
        checkpoint('period', period)
        original_orders = tuple(period['momentum_orders'])
        raised, intermediate, final_orders = order_schedule(original_orders)
        result['momentum_schedule'] = dict(original=list(original_orders), raised=list(raised),
                                           intermediate=list(intermediate), final=list(final_orders))
        baseline = legacy.cft_evaluation(node, period, numerical, radii,
                                         check_orders=False, use_baseline=True)
        checkpoint('original_baseline', baseline)
        high_period = dict(period, momentum_orders=list(raised))
        high = legacy.cft_evaluation(node, high_period, numerical, radii,
                                     check_orders=True, use_baseline=False)
        checkpoint('higher_orders', high)
        if tuple(high['momentum_orders']) != intermediate:
            raise ValueError('unexpected intermediate momentum orders')
        final_policy = dict(numerical, cft_block_low=numerical['cft_block_high'])
        final_period = dict(period, momentum_orders=list(final_orders))
        final = legacy.cft_evaluation(node, final_period, final_policy, radii,
                                      check_orders=False, use_baseline=True)
        checkpoint('momentum_confirmation', final)
        delta = confirmation_step(high['liouville_log'], final['liouville_log'], numerical['momentum_tolerance'])
        u = pilot.RealUpsilonOne()
        args = ([complex(node['x1'], node['y1'])], [complex(node['x2'], node['y2'])], [node['length']], radii)
        control = pilot.primary_scaled_kernel_batch(*args, upsilon=u, block_order=2, momentum_order=24, bridge_order=32,
            kernel_convention=legacy.policy_kernel_convention(numerical))[0]
        control_high = pilot.primary_scaled_kernel_batch(*args, upsilon=u, block_order=2, momentum_order=32, bridge_order=48,
            kernel_convention=legacy.policy_kernel_convention(numerical))[0]
        full = np.asarray(final['scaled_kernel'])
        baseline_full = np.asarray(baseline['scaled_kernel'])
        residual = pilot.weighted_residual(full[None, :], control[None, :], [node])[0]
        baseline_residual = pilot.weighted_residual(baseline_full[None, :], control[None, :], [node])[0]
        step_sum = sum(high['checks'][k] for k in ('block_relative_step', 'momentum_relative_step', 'scalar_powered_step')) + delta
        proxy = (full * step_sum + np.abs(control_high - control)) * math.exp(node['log_weight_over_rho_squared'])
        if np.any(~np.isfinite(proxy)):
            raise ValueError('nonfinite numerical-step proxy')
        result.update(status='ok', period=period, cft=final, original_baseline_cft=baseline,
                      higher_order_checks=high['checks'], final_momentum_relative_step=delta,
                      momentum_tolerance=numerical['momentum_tolerance'], radii=radii.tolist(),
                      control_scaled=control.tolist(), weighted_residual=residual.tolist(),
                      original_baseline_weighted_residual=baseline_residual.tolist(),
                      weighted_upgrade_delta=(residual - baseline_residual).tolist(),
                      relative_change_from_original_baseline=(full / baseline_full - 1).tolist(),
                      relative_residual=(full / control - 1).tolist(),
                      weighted_numerical_step_proxy=proxy.tolist(),
                      residual_smaller_than_numerical_step=(np.abs(residual) <= proxy).tolist(),
                      numerical_proxy_warning='Order changes are sensitivity diagnostics, not rigorous error bounds; G is unchanged.')
    except Exception as exc:
        result.update(status='failed', error=f'{type(exc).__name__}: {exc}')
    result['elapsed_seconds'] = time.monotonic() - started
    legacy.atomic_json(output, result)
    return result






def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    a=p.parse_args()
    if work(a.parent,a.run)['status']!='ok':raise SystemExit(1)

if __name__=='__main__':main()
