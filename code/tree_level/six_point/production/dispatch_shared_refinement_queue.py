"""Distribute only unfinished refinement batches across a shared 192-worker queue.

The accepted numerical implementation stays immutable.  Its extension function
is reused with only the task iterator replaced; coefficients, row checkpoints,
contractions, stopping rules and assembly retain their original implementation.
"""
from __future__ import annotations

# Resolve the package's import directories when this script is run directly.
import runpy as _six_point_runpy
from pathlib import Path as _SixPointPath
_six_point_package = next((p for p in _SixPointPath(__file__).resolve().parents
                          if (p / "run.py").is_file() and (p / "PACKAGE_MAP.json").is_file()), None)
if _six_point_package is not None:
    _six_point_runpy.run_path(str(_six_point_package / "run.py"))["bootstrap"]()


import argparse
import ast
import datetime
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import traceback

QUEUE_VERSION = 'shared_queue_v1'
CORE = None


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def load_core(root):
    global CORE
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location(
        'immutable_refinement_controller', root / 'continue_flagged_samples_to_c18.py')
    CORE = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(CORE)
    return CORE


def deployment(root):
    c = CORE
    pinned = c.read_manifest(Path(__file__).with_name('deployment.json'))
    if (pinned['queue_version'] != QUEUE_VERSION
            or pinned['script_sha256'] != c.digest(Path(__file__))
            or pinned['core_sha256'] != c.digest(root / 'continue_flagged_samples_to_c18.py')
            or pinned['source_fingerprint'] != c.source(root)['fingerprint']):
        raise ValueError('shared-queue deployment or original source changed')
    c.verify_deployment(root)
    return pinned


def queue_dir(root, order):
    return root / f'stages/C{order}/scheduling/{QUEUE_VERSION}'


def previous_jobs_stopped(root, order, record):
    """Require affirmative accounting evidence before replacing an old chain."""
    ids = [j['job_id'] for j in record['jobs'].values()]
    if not ids or any(not str(j).isdigit() for j in ids):
        raise ValueError('unreconciled original job identities')
    result = subprocess.run(['sacct', '-n', '-X', '-P', '-j', ','.join(ids),
        '--format=JobID,State%40,ExitCode'], text=True, capture_output=True, check=True)
    rows = [line.split('|')[:3] for line in result.stdout.splitlines() if line.strip()]
    for job in ids:
        members = [r for r in rows if r[0] == job or r[0].startswith(job + '_')]
        if not members:
            raise ValueError('no accounting evidence for old job ' + job)
        if any(r[1].strip().split()[0] not in {'COMPLETED', 'CANCELLED'} for r in members):
            raise ValueError('old jobs have not cleanly stopped: ' + repr(members))
    receipt = CORE.seal(dict(queue_version=QUEUE_VERSION, order=order,
        checked_utc=utc(), original_jobs=ids, accounting=rows,
        command=result.args, scheduler_changes=0))
    # Keep each observation rather than changing earlier evidence.
    CORE.write_json_once(queue_dir(root, order) / ('quiescence-' + receipt['fingerprint'][:16] + '.json'), receipt)
    return receipt


def prepare_plan(root, order):
    c = CORE
    pin = deployment(root)
    stage = c.read_manifest(root / f'stages/C{order}/manifest.json')
    path = queue_dir(root, order) / 'plan.json'
    if path.exists():
        existing = c.read_manifest(path)
        if existing['stage_fingerprint'] != stage['fingerprint']:
            raise ValueError('existing queue belongs to another stage')
        return existing
    submission = root / f'stages/C{order}/submission.json'
    quiescence = None
    if submission.exists():
        old = json.loads(submission.read_text())
        if old.get('queue_dispatcher'):
            raise ValueError('a queue submission exists without its immutable plan')
        quiescence = previous_jobs_stopped(root, order, old)
    missing, complete, checkpoints = [], [], []
    new_rows = 0
    for task in stage['tasks']:
        c.validate_task(task, order)
        if (root / task['output']).exists():
            complete.append(task['fingerprint'])
            continue
        count = 0
        checkpoint = root / (task['output'] + '.partial')
        if checkpoint.exists():
            meta, arrays = c.read_shell(root, task, order, partial=True)
            count = len(arrays['row_ids'])
            checkpoints.append(dict(task_fingerprint=task['fingerprint'],
                path=str(checkpoint.relative_to(root)), saved_rows=count,
                sha256=c.digest(checkpoint), metadata_fingerprint=meta['fingerprint']))
        new_rows += len(task['row_ids']) - count
        missing.append(task['fingerprint'])
    if len(set(missing + complete)) != len(stage['tasks']):
        raise ValueError('task partition has duplicate or missing identities')
    if not missing:
        raise ValueError('no unfinished batches; no extension submission is needed')
    ledger = c.seal(dict(queue_version=QUEUE_VERSION, order=order,
        stage_fingerprint=stage['fingerprint'], observed_utc=utc(),
        already_complete_task_ids=complete, unfinished_task_checkpoints=checkpoints,
        completed_coefficient_files_recomputed=0,
        quiescence_fingerprint=quiescence['fingerprint'] if quiescence else None))
    c.write_json_once(queue_dir(root, order) / 'preservation_ledger.json', ledger)
    plan = c.seal(dict(queue_version=QUEUE_VERSION, order=order,
        source_fingerprint=stage['source_fingerprint'], stage_fingerprint=stage['fingerprint'],
        scheduler_deployment_fingerprint=pin['fingerprint'],
        task_ids=missing, original_task_count=len(stage['tasks']),
        already_complete_task_count=len(complete), unfinished_task_count=len(missing),
        saved_checkpoint_rows=sum(x['saved_rows'] for x in checkpoints),
        new_rows_required=new_rows, preservation_ledger_fingerprint=ledger['fingerprint'],
        max_workers=min(192, len(missing)), created_utc=utc()))
    c.write_json_once(path, plan)
    c.atomic_json(queue_dir(root, order) / 'cursor.json',
        dict(plan_fingerprint=plan['fingerprint'], next=0))
    return plan


def claimed_tasks(root, manifest, index, workers):
    """One short POSIX lock assigns each batch once; no lock spans recursion."""
    c = CORE
    directory = queue_dir(root, manifest['order'])
    plan = c.read_manifest(directory / 'plan.json')
    if plan['stage_fingerprint'] != manifest['fingerprint'] or workers != plan['max_workers']:
        raise ValueError('worker queue/stage identity mismatch')
    lookup = {task['fingerprint']: task for task in manifest['tasks']}
    if len(plan['task_ids']) != len(set(plan['task_ids'])) or not set(plan['task_ids']) <= lookup.keys():
        raise ValueError('invalid queue task coverage')
    while True:
        with (directory / 'cursor.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            cursor = json.loads((directory / 'cursor.json').read_text())
            if cursor['plan_fingerprint'] != plan['fingerprint']:
                raise ValueError('queue cursor belongs to another plan')
            position = cursor['next']
            if type(position) is not int or not 0 <= position <= len(plan['task_ids']):
                raise ValueError('invalid queue position')
            if position == len(plan['task_ids']):
                return
            identity = plan['task_ids'][position]
            claim = c.seal(dict(plan_fingerprint=plan['fingerprint'], position=position,
                task_fingerprint=identity, worker=index,
                array_job_id=os.environ.get('SLURM_ARRAY_JOB_ID', 'synthetic'),
                claimed_utc=utc()))
            # An uncertain dispatch is never silently overwritten or retried.
            c.write_json_once(directory / f'claims/{position:08d}.json', claim)
            c.atomic_json(directory / 'cursor.json',
                dict(plan_fingerprint=plan['fingerprint'], next=position + 1))
        task = lookup[identity]
        yield task
        if not (root / task['output']).exists():
            raise ValueError('claimed batch returned without a complete saved output')
        c.write_json_once(directory / f'done/{position:08d}.json', c.seal(dict(
            plan_fingerprint=plan['fingerprint'], task_fingerprint=identity,
            claim_fingerprint=claim['fingerprint'], completed_utc=utc())))


def build_extension_worker(core):
    """Reuse the exact accepted worker body, changing only how it receives tasks."""
    source = Path(core.__file__).read_text()
    tree = ast.parse(source)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'extend_worker')
    original = ast.get_source_segment(source, function)
    before = "for task in manifest['tasks'][index::workers]:"
    after = 'for task in claimed_tasks(root, manifest, index, workers):'
    if original.count(before) != 1:
        raise ValueError('accepted worker structure changed; iterator adaptation needs review')
    modified = original.replace(before, after, 1)
    namespace = dict(vars(core), claimed_tasks=claimed_tasks)
    exec(compile(modified, '<shared-queue:unchanged-numerical-worker>', 'exec'), namespace)
    return namespace['extend_worker']


def submit_stage(root, order, dependency=None):
    c = CORE
    pin = deployment(root)
    with (root / '.dispatch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        plan = prepare_plan(root, order)
        stage = c.read_manifest(root / f'stages/C{order}/manifest.json')
        path = root / f'stages/C{order}/submission.json'
        if path.exists():
            record = json.loads(path.read_text())
            if not record.get('queue_dispatcher'):
                previous_jobs_stopped(root, order, record)
                archive = queue_dir(root, order) / 'original_submission.json'
                c.write_json_once(archive, record)
                record = None
        else:
            record = None
        if record is None:
            record = dict(version=c.VERSION, stage_fingerprint=stage['fingerprint'],
                jobs={}, created_utc=utc(), queue_dispatcher=dict(version=QUEUE_VERSION,
                    script=str(Path(__file__).resolve()), deployment_fingerprint=pin['fingerprint'],
                    plan_fingerprint=plan['fingerprint'], plan=str(queue_dir(root, order) / 'plan.json')))
        if (record['stage_fingerprint'] != stage['fingerprint']
                or record['queue_dispatcher']['plan_fingerprint'] != plan['fingerprint']):
            raise ValueError('existing dispatch has another identity')
        for action, workers in [('extend', plan['max_workers']),
                                ('evaluate', min(192, len(stage['evaluations']))), ('advance', 1)]:
            if action in record['jobs']:
                if not record['jobs'][action].get('job_id'):
                    raise RuntimeError('uncertain submission intent; reconcile Slurm before retrying')
                dependency = record['jobs'][action]['job_id']
                continue
            name = f'sel2p18-{stage["source_fingerprint"][:8]}-C{order}-shared-{action}'
            command = ['sbatch', '--parsable', '--partition=yin', '--account=yin_lab',
                '--nodes=1', '--cpus-per-task=1', '--mem=3500M',
                '--time=3-00:00:00' if action == 'extend' else '--time=12:00:00',
                '--job-name=' + name,
                '--output=' + str(root / f'logs/C{order}-{action}-%A_%a.out')]
            if dependency:
                command.append('--dependency=afterok:' + str(dependency))
            if action != 'advance':
                command.append(f'--array=0-{workers-1}%{workers}')
            entrypoint = root / 'continue_flagged_samples_to_c18.py' if action == 'evaluate' else Path(__file__).resolve()
            execution = ['env', 'PYTHONDONTWRITEBYTECODE=1'] + [v + '=1' for v in c.THREAD_VARIABLES]
            execution += [c.PYTHON, str(entrypoint), str(root), action, '--order', str(order), '--workers', str(workers)]
            wrap = 'exec ' + shlex.join(execution)
            if action != 'advance':
                wrap += ' --index "$SLURM_ARRAY_TASK_ID"'
            command += ['--wrap', wrap]
            record['jobs'][action] = dict(name=name, command=command,
                job_id=None, status='intent_recorded')
            c.atomic_json(path, record)
            result = subprocess.run(command, text=True, capture_output=True, check=True)
            job_id = result.stdout.strip().split(';')[0]
            if not job_id.isdigit():
                raise RuntimeError('unparseable submission; reconcile Slurm before retrying')
            record['jobs'][action].update(job_id=job_id, status='submitted', stderr=result.stderr)
            c.atomic_json(path, record)
            dependency = job_id
        return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('action', choices=['prepare', 'submit', 'extend', 'advance'])
    parser.add_argument('--order', type=int, required=True)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--index', type=int, default=0)
    args = parser.parse_args()
    root = args.root.resolve()
    c = load_core(root)
    if args.order not in (15, 16, 17, 18) or not 1 <= args.workers <= 192 or not 0 <= args.index < args.workers:
        raise ValueError('authorized order/worker envelope exceeded')
    deployment(root)
    try:
        if args.action == 'prepare':
            plan = prepare_plan(root, args.order)
            result = {k: v for k, v in plan.items() if k != 'task_ids'}
        elif args.action == 'submit':
            result = submit_stage(root, args.order, os.environ.get('SLURM_JOB_ID'))
        elif args.action == 'extend':
            if any(os.environ.get(v) != '1' for v in c.THREAD_VARIABLES):
                raise ValueError('all numerical-library thread counts must be one')
            directory = queue_dir(root, args.order)
            directory.mkdir(parents=True, exist_ok=True)
            with (directory / f'worker-{args.index}.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                result = build_extension_worker(c)(root, args.order, args.index, args.workers)
                result['queue_version'] = QUEUE_VERSION
                job = os.environ.get('SLURM_ARRAY_JOB_ID', 'local')
                c.write_json_once(root / f'progress/C{args.order}/extend-{job}-{args.index}.json', result)
        else:
            # The accepted advance function retains all stopping and assembly logic.
            # Its next-stage submission uses the same shared queue at C17 and C18.
            c.submit_stage = submit_stage
            result = c.advance(root, args.order)
        print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    except Exception:
        c.atomic_json(root / f'failures/C{args.order}-shared-{args.action}-{os.environ.get("SLURM_JOB_ID", "local")}-{args.index}.json',
            dict(traceback=traceback.format_exc(), action=args.action, order=args.order,
                index=args.index, accepted_outputs_retained=True, queue_version=QUEUE_VERSION))
        raise


if __name__ == '__main__':
    main()
