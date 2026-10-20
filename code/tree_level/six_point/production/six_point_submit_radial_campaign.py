#!/usr/bin/env python3
"""Submit the immutable radial comparison, with a recoverable submission receipt."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

PYTHON = '/n/holylabs/yin_lab/Everyone/yutaizhang/StringMC/genus2_period_table_20260718_176ab14d/.venv/bin/python'
STAGES = [('validate', None, '00:30:00', '4000M'),
          ('addback', None, '12:00:00', '6000M'),
          ('preflight', 8, '12:00:00', '3500M'),
          ('full', 192, '48:00:00', '3500M'),
          ('reduce', None, '04:00:00', '6000M'),
          ('audit', None, '01:00:00', '4000M')]


def atomic_write(path, value):
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def command(root, label, action, count, limit, memory, dependency, python=PYTHON):
    directory = root / label if label else root
    name = f'radial-{label}-{action}' if label else 'radial-aggregate'
    result = ['sbatch', '--parsable', '--partition=yin', '--account=yin_lab', '--nodes=1',
        '--cpus-per-task=1', '--mem='+memory, '--time='+limit, '--job-name='+name,
        '--output='+str(directory / 'logs' / (f'{action}-%A_%a.out' if count else f'{action}-%A.out'))]
    if dependency is not None:
        result.append('--dependency=afterok:'+str(dependency))
    if count:
        result.append(f'--array=0-{count-1}%{count}')
    executable = ([python, str(directory / 'jobs_worker.py'), str(directory), action]
                  if label else [python, str(root / 'aggregate_curve.py'), str(root)])
    wrap = shlex.join(['exec', 'env', 'OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1',
                     'MKL_NUM_THREADS=1', 'NUMEXPR_NUM_THREADS=1'] + executable)
    if count:
        wrap += ' --index "$SLURM_ARRAY_TASK_ID"'
    return result + ['--wrap', wrap]


def submit(root, *, dry_run=False, resume=False, python=PYTHON):
    root = Path(root).resolve()
    manifest_path = root / 'deployment_manifest.json'
    expected = json.loads(manifest_path.read_text())
    for relative, digest in expected.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('frozen deployment changed: '+relative)
    campaign = json.loads((root / 'campaign.json').read_text())
    if campaign.get('continuation') and not dry_run:
        gate = json.loads((root / campaign['continuation']['cluster_gate']).read_text())
        gate_input = json.loads((root/'preflight/input.json').read_text())
        if (not gate.get('passed') or gate.get('partition') != 'yin' or not gate.get('slurm_job_id')
                or gate.get('case_count') != len(gate_input['cases'])
                or len(gate.get('cases', [])) != len(gate_input['cases'])
                or gate.get('benchmark_input_sha256') != hashlib.sha256((root/'preflight/input.json').read_bytes()).hexdigest()
                or not 0 < gate.get('peak_rss_mib', 0) < gate_input['maximum_peak_rss_mib']
                or not all(r.get('all_coefficients_and_dozz_identical') and r.get('contractions_identical') for r in gate['cases'])):
            raise ValueError('optimized continuation requires a successful matching Cannon preflight')
        for relative, digest in gate['source_hashes'].items():
            if hashlib.sha256((root/relative).read_bytes()).hexdigest() != digest:
                raise ValueError('Cannon preflight source changed: '+relative)
    production = campaign['version'] == 'sphere-six-point-yin-radial-production-curve-v1'
    if production:
        selected = campaign['selected_plot_indices']
        expected_t = [campaign['source_plot_energies'][i] for i in selected]
        if (len(selected) != 15 or sorted(set(selected)) != selected
                or [row['t'] for row in campaign['energies']] != expected_t
                or len({row['label'] for row in campaign['energies']}) != 15
                or any(not 0 < t < 1/3 for t in expected_t)):
            raise ValueError('invalid fifteen-point plot selection')
    elif (campaign['version'] != 'sphere-six-point-yin-radial-mixture-campaign-v1'
          or [row['t'] for row in campaign['energies']] != [.1805, .2505, .3155]):
        raise ValueError('unexpected radial campaign')
    receipt_path = root / 'submission.json'
    receipt = dict(version='sphere-six-point-yin-radial-submission-v1',
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        deployment_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(), jobs=[])
    if receipt_path.exists():
        if not resume:
            raise FileExistsError('submission exists; use --resume for interrupted submission only')
        receipt = json.loads(receipt_path.read_text())
        if receipt['deployment_sha256'] != hashlib.sha256(manifest_path.read_bytes()).hexdigest():
            raise ValueError('submission belongs to another release')
    graph = []
    for row in campaign['energies']:
        if production and row['operation'] == 'reuse':
            if campaign.get('continuation', {}).get('completed_results') == 'reference-only':
                continue
            graph.append((row['label'], 'reuse', None, '00:30:00', '4000M'))
            continue
        graph.append((row['label'], *STAGES[0]))
        families_to_build = row['families'] if production else row['additional_bank_families']
        if families_to_build:
            graph.extend([(row['label'], 'build', 192, '72:00:00' if production else '12:00:00', '3500M'),
                          (row['label'], 'assemble', min(192, families_to_build), '02:00:00', '6000M')])
        for action, count, limit, memory in STAGES[1:]:
            graph.append((row['label'], action, row['preflight_shards'] if action == 'preflight' else count, limit, memory))
    graph.append(('', 'aggregate', None, '01:00:00', '2000M'))
    if len(receipt['jobs']) > len(graph):
        raise ValueError('invalid submission receipt length')
    dependency = campaign['initial_dependency']
    planned = []
    for position, (label, action, count, limit, memory) in enumerate(graph):
        limit = campaign.get('scheduler_time_limits', {}).get(action, limit)
        args = command(root, label, action, count, limit, memory, dependency, python)
        if position < len(receipt['jobs']):
            row = receipt['jobs'][position]
            if row['command'] != args or row['label'] != label or row['action'] != action:
                raise ValueError('submission receipt is not an exact graph prefix')
            dependency = row['job_id']
            continue
        if dry_run:
            planned.append(args)
            dependency = f'DRY-{position}'
            continue
        (root / label / 'logs').mkdir(exist_ok=True)
        # Record intent before sbatch. A crash after dispatch but before recording
        # the ID requires scheduler reconciliation, never blind resubmission.
        pending = root / 'submission_pending.json'
        if pending.exists():
            raise RuntimeError('unresolved submission intent; reconcile this job with squeue/sacct first')
        atomic_write(pending, dict(position=position, command=args))
        job_id = subprocess.check_output(args, text=True).strip().split(';')[0]
        if not job_id.isdigit():
            raise ValueError('sbatch returned an invalid job ID')
        receipt['jobs'].append(dict(label=label, action=action, job_id=job_id, command=args))
        atomic_write(receipt_path, receipt)
        pending.unlink()
        dependency = job_id
    return dict(validated_deployment_files=len(expected), commands=planned) if dry_run else receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--python', default=PYTHON)
    args = parser.parse_args()
    print(json.dumps(submit(args.root, dry_run=args.dry_run, resume=args.resume, python=args.python), indent=2))
