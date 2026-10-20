"""Prepare, evaluate, and assemble selective C13-C14 and C15-C18 refinement stages."""
from __future__ import annotations
import sys
import runpy as _six_point_runpy
from pathlib import Path as _SixPointPath
import argparse
import datetime
import json
import os
from pathlib import Path
import runpy
import shlex
import subprocess
import time
import traceback
import numpy as np
import fcntl
from six_point_refinement_statistics import (apply_corrections, array_hash, contract_chiral, degree_keys, digest, json_hash, load_arrays, paired_statistics, point_slots, read_manifest, require_active, save_arrays, seal, write_json_once, initial as INITIAL_POLICY, continuation as CONTINUATION_POLICY)


PYTHON = '/n/holylabs/yin_lab/Everyone/yutaizhang/StringMC/genus2_period_table_20260718_176ab14d/.venv/bin/python'


THREAD_VARIABLES = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                    'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS')


class RefinementRun:
    policy = INITIAL_POLICY

    def source(self, root):
        manifest = read_manifest(root / 'inputs/source_manifest.json')
        gate = read_manifest(root / 'inputs/completion_gate.json')
        if not gate['passed'] or gate['fingerprint'] != manifest['gate']:
            raise ValueError('original completion gate not satisfied')
        if (manifest['base_order'], manifest['cap'], manifest['tolerance']) != (12, 16, 0.005):
            raise ValueError('authorized order or tolerance changed')
        return manifest

    def bootstrap(self, root):
        runpy.run_path(str(root / 'frozen_source/run.py'))['bootstrap']()

    def verify_deployment(self, root):
        deployment = read_manifest(root / 'deployment.json')
        for relative, expected in deployment['files'].items():
            if digest(root / relative) != expected:
                raise ValueError('immutable deployment changed: ' + relative)
        return deployment

    def prepare_groups(self, root):
        """Read only required parent-bank identities and nodes; never build a bank."""
        src = self.source(root)
        groups = {}
        for bid, bank in sorted(src['banks'].items()):
            if not bank['active_ids']:
                continue
            ref = bank['parent']['evidence_reference']
            path = Path(ref['path'])
            if digest(path) != ref['file_sha256']:
                raise ValueError('parent bank content changed: ' + str(path))
            with np.load(path, allow_pickle=False) as z:
                meta = json.loads(z['metadata'].item())
                nodes = z['p_values']
            if json_hash(meta) != ref['metadata_sha256'] or json_hash(meta['provenance']) != ref['provenance_sha256']:
                raise ValueError('parent bank metadata/provenance changed')
            if (meta['fingerprint'], meta['order'], meta['t'], meta['topology'], meta['incoming_slot']) != (
                    bank['parent']['fingerprint'], 12, bank['t'], bank['topology'], bank['incoming_slot']):
                raise ValueError('parent physical identity or degree mismatch')
            if meta['provider'] != bank['parent']['provider'] or nodes.dtype != np.float64:
                raise ValueError('parent arithmetic policy or node dtype mismatch')
            if nodes.ndim != 2 or nodes.shape[1] != 3 or not np.isfinite(nodes).all():
                raise ValueError('parent nodes invalid')
            physical = dict(t_hex=float(bank['t']).hex(), topology=bank['topology'],
                incoming_slot=bank['incoming_slot'], provider=meta['provider'],
                special_dps=meta['provenance'].get('special_dps'),
                source_input_identity=bank['original_input_identity'],
                parent_recursion_source_sha256=bank['parent_recursion_source_sha256'])
            gid = json_hash(physical)
            group = groups.setdefault(gid, dict(physical=physical, nodes=[], banks=[]))
            group['banks'].append(bid)
            group['nodes'].append(nodes)
            print('read parent', bid, len(nodes), flush=True)
        descriptors = {}
        for gid, group in sorted(groups.items()):
            combined = np.concatenate(group['nodes'])
            bits = np.ascontiguousarray(combined).view(np.dtype((np.void, 24))).reshape(-1)
            _, first, inverse = np.unique(bits, return_index=True, return_inverse=True)
            appearance = np.argsort(first)
            remap = np.empty(len(appearance), dtype=np.int64)
            remap[appearance] = np.arange(len(appearance))
            inverse = remap[inverse]
            nodes = combined[first[appearance]].copy()
            arrays, cursor = {'p_values': nodes}, 0
            bank_rows = {}
            for bid, original_nodes in zip(group['banks'], group['nodes']):
                scatter = inverse[cursor:cursor + len(original_nodes)].astype(np.int64)
                cursor += len(scatter)
                if not np.array_equal(nodes[scatter].view(np.uint64), original_nodes.view(np.uint64)):
                    raise ValueError('exact binary64 scatter mismatch')
                arrays['scatter_' + bid] = scatter
                bank_rows[bid] = len(scatter)
            relative = 'groups/' + gid + '.npz'
            meta = save_arrays(root / relative, arrays, dict(version=self.policy.VERSION, kind='extension-node-union',
                source_fingerprint=src['fingerprint'], physical=group['physical'],
                parents={bid: src['banks'][bid]['parent']['fingerprint'] for bid in group['banks']}))
            descriptors[gid] = dict(id=gid, physical=group['physical'], banks=group['banks'],
                data=relative, data_fingerprint=meta['fingerprint'], row_count=len(nodes), bank_rows=bank_rows)
        result = seal(dict(version=self.policy.VERSION, source_fingerprint=src['fingerprint'], groups=descriptors,
            deduplicated_rows=sum(g['row_count'] for g in descriptors.values()),
            original_view_rows=sum(sum(g['bank_rows'].values()) for g in descriptors.values()),
            required_parent_banks=sum(len(g['banks']) for g in descriptors.values()),
            regenerated_parent_bank_outputs=0, rerun_original_amplitudes=0))
        write_json_once(root / 'groups/manifest.json', result)
        return {k: result[k] for k in ('deduplicated_rows', 'original_view_rows', 'required_parent_banks')}

    def states_before(self, root, order, src):
        if order == 13:
            return src['states'], src['fingerprint']
        previous = read_manifest(root / f'stages/C{order-1}/state.json')
        return previous['states'], previous['fingerprint']

    def group_data(self, root, descriptor):
        meta, arrays = load_arrays(root / descriptor['data'])
        if meta['fingerprint'] != descriptor['data_fingerprint']:
            raise ValueError('group nodes changed')
        return arrays

    def prepare_stage(self, root, order, chunk_size=16):
        self.policy.shell_keys(order)
        if not 1 <= chunk_size <= 64:
            raise ValueError('bounded positive shell chunk size required')
        src, deployment = self.source(root), self.verify_deployment(root)
        groups = read_manifest(root / 'groups/manifest.json')
        states, previous_fingerprint = self.states_before(root, order, src)
        tasks, evaluations = [], []
        active = {uid for uid, st in states.items() if st['status'] == f'needs_{order}'}
        for gid, desc in groups['groups'].items():
            needed_banks = {bid: [uid for uid in src['banks'][bid]['target_ids'] if uid in active]
                            for bid in desc['banks']}
            needed_banks = {bid: ids for bid, ids in needed_banks.items() if ids}
            if not needed_banks:
                continue
            arrays = self.group_data(root, desc)
            row_ids = np.unique(np.concatenate([arrays['scatter_' + bid] for bid in needed_banks]))
            for start in range(0, len(row_ids), chunk_size):
                spec = dict(group=gid, group_fingerprint=desc['data_fingerprint'], order=order,
                    row_ids=row_ids[start:start + chunk_size].tolist(), required_banks=sorted(needed_banks),
                    prior_state=previous_fingerprint, source_fingerprint=src['fingerprint'])
                task = seal(spec)
                task['output'] = f'stages/C{order}/shells/{gid[:16]}/{task["fingerprint"]}.npz'
                tasks.append(task)
            for bid, ids in needed_banks.items():
                evaluations.append(dict(bank_id=bid, group=gid, target_ids=ids))
        if set(uid for task in evaluations for uid in task['target_ids']) != active:
            raise ValueError('evaluation manifest misses active points')
        manifest = seal(dict(version=self.policy.VERSION, order=order, window=[order-4, order-2, order],
            source_fingerprint=src['fingerprint'], prior_state=previous_fingerprint,
            deployment_fingerprint=deployment['fingerprint'], groups_fingerprint=groups['fingerprint'],
            tasks=tasks, evaluations=evaluations, active_point_ids=sorted(active),
            unique_row_shell_count=sum(len(t['row_ids']) for t in tasks),
            coefficient_count=sum(len(t['row_ids']) for t in tasks) * len(self.policy.shell_keys(order)),
            coefficient_degrees=[order], regenerated_parent_bank_outputs=0, rerun_original_amplitudes=0,
            validation_only_jobs=0, timing_only_jobs=0, max_workers=192, threads_per_worker=1))
        write_json_once(root / f'stages/C{order}/manifest.json', manifest)
        return {k: manifest[k] for k in ('order', 'unique_row_shell_count', 'coefficient_count', 'fingerprint')}

    def construct_shell(self, provider, topology, external_weights, nodes, order):
        """Only this function requests new physical coefficients; mockable in tests."""
        keys = self.policy.shell_keys(order)
        values = []
        for point in nodes:
            table = provider.coefficients(topology, external_weights, tuple(1 + p*p for p in point),
                                         order, central_charge=25, minimum_total_order=order)
            if set(table) != set(map(tuple, keys)):
                raise ValueError('provider returned anything other than the requested new shell')
            values.append([table[tuple(k)] for k in keys])
        coefficients = np.asarray(values, dtype=np.complex128)
        self.policy.validate_shell(keys, coefficients, order, len(nodes))
        return coefficients

    def validate_task(self, task, order):
        spec = {k: v for k, v in task.items() if k not in ('fingerprint', 'output')}
        if task['fingerprint'] != json_hash(spec) or task['order'] != order:
            raise ValueError('shell task identity changed')
        self.policy.shell_keys(order)
        if not task['row_ids'] or len(task['row_ids']) != len(set(task['row_ids'])):
            raise ValueError('empty or duplicate extension rows')
        expected_output = f'stages/C{order}/shells/{task["group"][:16]}/{task["fingerprint"]}.npz'
        if task['output'] != expected_output or any(type(i) is not int or i < 0 for i in task['row_ids']):
            raise ValueError('invalid row identity or extension output path')

    def read_shell(self, root, task, order, partial=False):
        path = root / (task['output'] + '.partial' if partial else task['output'])
        meta, arrays = load_arrays(path)
        count = len(arrays['row_ids'])
        if (meta['task_fingerprint'] != task['fingerprint'] or meta['order'] != order
                or not np.array_equal(arrays['row_ids'], task['row_ids'][:count])
                or len(meta['precision_rows']) != count or not 0 < count <= len(task['row_ids'])
                or (not partial and count != len(task['row_ids']))):
            raise ValueError('saved extension does not match requested rows/precision coverage')
        self.policy.validate_shell(arrays['keys'], arrays['coefficients'], order, count)
        if any(p['verified_tables'] != 1 for p in meta['precision_rows']):
            raise ValueError('saved row lacks accepted arithmetic evidence')
        return meta, arrays

    def extend_worker(self, root, order, index, workers):
        deployment = self.verify_deployment(root)
        manifest = read_manifest(root / f'stages/C{order}/manifest.json')
        src, groups = self.source(root), read_manifest(root / 'groups/manifest.json')
        states, previous = self.states_before(root, order, src)
        if manifest['deployment_fingerprint'] != deployment['fingerprint'] or manifest['prior_state'] != previous:
            raise ValueError('worker source/state identity differs from stage')
        self.bootstrap(root)
        from sphere_six_point_coefficient_provider import VerifiedSphereCProvider
        from sphere_six_point_momentum_banks import external_data
        started = time.monotonic()
        cached_gid, cached_nodes, data = None, None, None
        reused = generated = 0
        for task in manifest['tasks'][index::workers]:
            self.validate_task(task, order)
            if (root / task['output']).exists():
                self.read_shell(root, task, order)
                reused += len(task['row_ids'])
                continue
            desc = groups['groups'][task['group']]
            for bid in task['required_banks']:
                if not any(states[uid]['status'] == f'needs_{order}' for uid in src['banks'][bid]['target_ids']):
                    raise ValueError('task requests rows for an already stopped bank')
            if cached_gid != task['group']:
                data = self.group_data(root, desc)
                cached_gid, cached_nodes = task['group'], data['p_values']
            required = set(np.concatenate([data['scatter_' + bid] for bid in task['required_banks']]).tolist())
            if not set(task['row_ids']) <= required:
                raise ValueError('coefficient row not required by active point')
            policy = dict(desc['physical']['provider'])
            provider_options = {k: policy[k] for k in ('base_dps', 'check_dps', 'maximum_dps', 'rtol', 'atol')}
            exemplar = src['banks'][task['required_banks'][0]]
            _, external_weights = external_data(exemplar['t'], exemplar['incoming_slot'])
            completed, precisions = [], []
            checkpoint = root / (task['output'] + '.partial')
            if checkpoint.exists():
                meta, arrays = self.read_shell(root, task, order, partial=True)
                completed = list(arrays['coefficients'])
                precisions = meta['precision_rows']
                metadata = {k: v for k, v in meta.items() if k not in ('fingerprint', 'array_hashes')}
                reused += len(completed)
            for local in range(len(completed), len(task['row_ids'])):
                if time.monotonic() - started > 70 * 3600:
                    raise TimeoutError('worker reached resumable 70-hour budget; accepted rows retained')
                row_id = task['row_ids'][local]
                provider = VerifiedSphereCProvider(**provider_options, minimum_total_order=order)
                expected_policy = dict(policy, minimum_total_order=order)
                if provider.metadata() != expected_policy:
                    raise ValueError('new shell provider policy differs from accepted parent arithmetic')
                before = time.monotonic()
                new = self.construct_shell(provider, exemplar['topology'], external_weights,
                                      cached_nodes[row_id:row_id+1], order)[0]
                completed.append(new)
                precision = provider.diagnostics()
                precision['seconds'] = time.monotonic() - before
                precisions.append(precision)
                arrays = dict(row_ids=np.asarray(task['row_ids'][:len(completed)], dtype=np.int64),
                    keys=self.policy.shell_keys(order), coefficients=np.asarray(completed, dtype=np.complex128))
                metadata = dict(version=self.policy.VERSION, kind='new-degree-extension', order=order,
                    task_fingerprint=task['fingerprint'], group_fingerprint=task['group_fingerprint'],
                    source_fingerprint=src['fingerprint'], deployment_fingerprint=deployment['fingerprint'],
                    precision_rows=precisions, parent_coefficients_regenerated=0)
                save_arrays(checkpoint, arrays, metadata, checkpoint=True)
                generated += 1
            save_arrays(root / task['output'], arrays, metadata)
            print(json.dumps(dict(completed=task['fingerprint'], rows=len(completed), order=order,
                                  seconds=sum(p['seconds'] for p in precisions))), flush=True)
        return dict(worker=index, order=order, generated_rows=generated, reused_rows=reused,
                    seconds=time.monotonic()-started)

    def load_group_shell(self, root, manifest, gid, row_count):
        order = manifest['order']
        matrix = np.full((row_count, len(self.policy.shell_keys(order))), complex(np.nan, np.nan))
        coverage = np.zeros(row_count, dtype=np.int8)
        fingerprints = []
        for task in manifest['tasks']:
            if task['group'] != gid:
                continue
            meta, arrays = self.read_shell(root, task, order)
            ids = arrays['row_ids']
            matrix[ids] = arrays['coefficients']
            coverage[ids] += 1
            fingerprints.append(meta['fingerprint'])
        if np.any(coverage > 1):
            raise ValueError('duplicate row shell coverage')
        return matrix, coverage, json_hash(fingerprints)

    def point_chiral(self, root, uid, order, context, compute):
        path = root / 'contractions' / uid / f'C{order}.npz'
        if path.exists():
            meta, arrays = load_arrays(path)
            if meta['context'] != context or meta['order'] != order:
                raise ValueError('saved point contraction has different source identity')
            return arrays['chiral']
        chiral = compute()
        if not np.isfinite(chiral).all():
            raise FloatingPointError('nonfinite point contraction')
        save_arrays(path, {'chiral': np.asarray(chiral, dtype=complex)},
                    dict(version=self.policy.VERSION, context=context, order=order))
        return chiral

    def evaluate_worker(self, root, order, index, workers):
        self.verify_deployment(root)
        src, groups = self.source(root), read_manifest(root / 'groups/manifest.json')
        stage = read_manifest(root / f'stages/C{order}/manifest.json')
        states, previous = self.states_before(root, order, src)
        if stage['prior_state'] != previous:
            raise ValueError('evaluation stage prior decisions changed')
        self.bootstrap(root)
        from sphere_six_point_momentum_banks import SphereSixPointMomentumBank
        finished = 0
        for task in stage['evaluations'][index::workers]:
            bid, gid = task['bank_id'], task['group']
            descriptor, bank_spec = groups['groups'][gid], src['banks'][bid]
            data = self.group_data(root, descriptor)
            matrix, coverage, shell_identity = self.load_group_shell(root, stage, gid, len(data['p_values']))
            scatter = data['scatter_' + bid]
            if not np.all(coverage[scatter] == 1):
                raise ValueError('incomplete required parent-bank shell coverage')
            shell = matrix[scatter]
            del matrix
            aux_path = root / 'bank_aux' / (bid + '.npz')
            parent = None
            if order in (13, 14) or not aux_path.exists():
                ref = bank_spec['parent']['evidence_reference']
                if digest(ref['path']) != ref['file_sha256']:
                    raise ValueError('parent changed before contraction')
                parent = SphereSixPointMomentumBank.load(ref['path'], expected_fingerprint=bank_spec['parent']['fingerprint'])
                aux = dict(p_values=parent.p_values, primary=parent.primary_exponents,
                           logs=parent.log_weighted_structure_constants)
                save_arrays(aux_path, aux, dict(version=self.policy.VERSION, parent_fingerprint=parent.fingerprint))
            else:
                aux_meta, aux = load_arrays(aux_path)
                if aux_meta['parent_fingerprint'] != bank_spec['parent']['fingerprint']:
                    raise ValueError('cached spectral rule changed')
            if not np.array_equal(aux['p_values'].view(np.uint64), data['p_values'][scatter].view(np.uint64)):
                raise ValueError('spectral node order differs from extension rows')
            energy = next(e for e in src['energies'] if e['label'] == bank_spec['energy'])
            target_meta, targets = load_arrays(root / energy['targets'])
            if target_meta['fingerprint'] != energy['targets_fingerprint']:
                raise ValueError('frozen point targets changed')
            for uid in task['target_ids']:
                require_active(states[uid]['status'], order)
                j = states[uid]['target_index']
                if targets['ids'][j] != uid or targets['family_ids'][j] != bid:
                    raise ValueError('original point/family mapping differs')
                decision_path = root / f'stages/C{order}/decisions/{uid}.json'
                if decision_path.exists():
                    prior = read_manifest(decision_path)
                    if prior['stage_fingerprint'] != stage['fingerprint']:
                        raise ValueError('decision belongs to another stage')
                    finished += 1
                    continue
                q, base = targets['q'][j], targets['base_logarithms'][j]
                context = dict(uid=uid, parent_fingerprint=bank_spec['parent']['fingerprint'],
                    source_fingerprint=src['fingerprint'], target_fingerprint=target_meta['fingerprint'],
                    q_hash=array_hash(q))
                def old_prefix(n):
                    def compute():
                        if parent is None:
                            raise ValueError('required cached low prefix missing after earlier stage')
                        keys = degree_keys(12)
                        if n == 9:
                            return contract_chiral(parent.coefficient_matrix[:, keys.sum(axis=1) <= 9], degree_keys(9), q)
                        if n in (10, 11):
                            lower = old_prefix(9)
                            keep = (keys.sum(axis=1) > 9) & (keys.sum(axis=1) <= n)
                        elif n == 12:
                            lower = old_prefix(11)
                            keep = keys.sum(axis=1) == 12
                        else:
                            raise ValueError('unexpected parent prefix')
                        return lower + contract_chiral(parent.coefficient_matrix[:, keep], keys[keep], q)
                    return self.point_chiral(root, uid, n, context, compute)
                def previous_chiral(n):
                    if n <= 12:
                        return old_prefix(n)
                    meta, payload = load_arrays(root / 'contractions' / uid / f'C{n}.npz')
                    if (meta['context']['source_fingerprint'] != src['fingerprint'] or meta['context']['uid'] != uid
                            or meta['order'] != n or meta['context']['parent_fingerprint'] != bank_spec['parent']['fingerprint']):
                        raise ValueError('prior new-prefix contraction belongs to different point/source')
                    return payload['chiral']
                current_context = dict(context, shell_identity=shell_identity, stage_fingerprint=stage['fingerprint'])
                current = self.point_chiral(root, uid, order, current_context,
                    lambda: previous_chiral(order-1) + contract_chiral(shell, self.policy.shell_keys(order), q))
                chirals = {order: current, order-4: previous_chiral(order-4), order-2: previous_chiral(order-2)}
                logarithms = aux['logs'] + base + np.einsum('pk,k->p', aux['primary'], np.log(abs(q)), optimize=False)
                result = self.policy.nonchiral_diagnostic(chirals, logarithms, order)
                arrays = {k: np.asarray(result[k]) for k in ('totals', 'numerators', 'denominator', 'relative')}
                diagnostic_path = f'stages/C{order}/diagnostics/{uid}.npz'
                dmeta = save_arrays(root / diagnostic_path, arrays, dict(version=self.policy.VERSION, uid=uid,
                    order=order, orders=list(result['orders']), stage_fingerprint=stage['fingerprint']))
                value = complex(result['totals'][-1])
                decision = seal(dict(version=self.policy.VERSION, uid=uid, bank_id=bid, order=order,
                    status=self.policy.classify(order, result['passed']), passed=result['passed'],
                    new_baseline_I6=[value.real, value.imag], stage_fingerprint=stage['fingerprint'],
                    diagnostic=diagnostic_path, diagnostic_fingerprint=dmeta['fingerprint'],
                    relative_gaps=[float(x) if np.isfinite(x) else None for x in result['relative']],
                    representation=str(targets['representation'][j]), momentum_controls_extended=False))
                write_json_once(decision_path, decision)
                finished += 1
            print('evaluated', order, bid, len(task['target_ids']), flush=True)
        return dict(order=order, worker=index, evaluated_points=finished)

    def advance(self, root, order):
        src, stage = self.source(root), read_manifest(root / f'stages/C{order}/manifest.json')
        before, previous = self.states_before(root, order, src)
        if previous != stage['prior_state']:
            raise ValueError('decision barrier prior state changed')
        states = {uid: dict(st) for uid, st in before.items()}
        observed = set()
        for uid in stage['active_point_ids']:
            require_active(before[uid]['status'], order)
            d = read_manifest(root / f'stages/C{order}/decisions/{uid}.json')
            meta, arrays = load_arrays(root / d['diagnostic'])
            if (d['stage_fingerprint'] != stage['fingerprint'] or d['uid'] != uid or d['order'] != order
                    or meta['fingerprint'] != d['diagnostic_fingerprint'] or not np.isfinite(arrays['totals']).all()
                    or meta['orders'] != [order-4, order-2, order]
                    or arrays['denominator'] != abs(arrays['totals'][-1])
                    or complex(*d['new_baseline_I6']) != arrays['totals'][-1]):
                raise ValueError('invalid or nonfinite point decision')
            passed, _ = self.policy.gaps_pass(arrays['numerators'], arrays['denominator'], arrays['totals'][-1])
            if d['passed'] != bool(passed) or d['status'] != self.policy.classify(order, bool(passed)):
                raise ValueError('point stopping decision violates fixed criterion')
            states[uid].update(status=d['status'], last_order=order, new_baseline_I6=d['new_baseline_I6'],
                final_order=order if d['status'] in self.policy.STATUSES else None,
                decision=f'stages/C{order}/decisions/{uid}.json')
            observed.add(uid)
        if observed != {uid for uid, st in before.items() if st['status'] == f'needs_{order}'}:
            raise ValueError('missing or duplicate per-point decisions')
        counts = {status: sum(st['status'] == status for st in states.values())
                  for status in (*self.policy.STATUSES, 'needs_' + str(order + 1))}
        state = seal(dict(version=self.policy.VERSION, order=order, stage_fingerprint=stage['fingerprint'],
                          states=states, counts=counts, source_fingerprint=src['fingerprint']))
        write_json_once(root / f'stages/C{order}/state.json', state)
        if counts.get('needs_' + str(order + 1), 0):
            self.prepare_stage(root, order + 1)
            self.submit_stage(root, order + 1, dependency=os.environ.get('SLURM_JOB_ID'))
        else:
            from apply_pointwise_amplitude_corrections import assemble
            assemble(root, state)
        return counts

    def atomic_json(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
        temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
        os.replace(temp, path)

    def submit_stage(self, root, order, dependency=None):
        self.verify_deployment(root)
        stage = read_manifest(root / f'stages/C{order}/manifest.json')
        if stage['regenerated_parent_bank_outputs'] or stage['rerun_original_amplitudes']:
            raise ValueError('dispatch would replay known output')
        for task in stage['tasks']:
            self.validate_task(task, order)
        if not stage['tasks']:
            raise ValueError('no missing shell work; do not submit an empty computation')
        path = root / f'stages/C{order}/submission.json'
        record = json.loads(path.read_text()) if path.exists() else dict(version=self.policy.VERSION,
            stage_fingerprint=stage['fingerprint'], jobs={}, created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        if record['stage_fingerprint'] != stage['fingerprint']:
            raise ValueError('submission receipt belongs to another stage')
        for action, task_count in (('extend', len(stage['tasks'])), ('evaluate', len(stage['evaluations'])), ('advance', 1)):
            workers = min(192, task_count) if action != 'advance' else 1
            if action in record['jobs']:
                job = record['jobs'][action]
                if not job.get('job_id'):
                    raise RuntimeError('unreconciled submission intent; inspect scheduler before retrying')
                dependency = job['job_id']
                continue
            name = f'selc16-{stage["source_fingerprint"][:8]}-C{order}-{action}'
            command = ['sbatch', '--parsable', '--partition=yin', '--account=yin_lab', '--nodes=1',
                '--cpus-per-task=1', '--mem=3500M', '--time=3-00:00:00' if action == 'extend' else '--time=12:00:00',
                '--job-name=' + name, '--output=' + str(root / f'logs/C{order}-{action}-%A_%a.out')]
            if dependency:
                command.append('--dependency=afterok:' + str(dependency))
            if action != 'advance':
                command.append(f'--array=0-{workers-1}%{workers}')
            execution = ['env', 'PYTHONDONTWRITEBYTECODE=1'] + [k + '=1' for k in THREAD_VARIABLES]
            execution += [PYTHON, str(root / 'selective_campaign.py'), str(root), action, '--order', str(order), '--workers', str(workers)]
            wrap = 'exec ' + shlex.join(execution)
            if action != 'advance':
                wrap += ' --index "$SLURM_ARRAY_TASK_ID"'
            command += ['--wrap', wrap]
            record['jobs'][action] = dict(name=name, command=command, job_id=None, status='intent_recorded')
            self.atomic_json(path, record)
            outcome = subprocess.run(command, text=True, capture_output=True, check=True)
            job_id = outcome.stdout.strip().split(';')[0]
            if not job_id.isdigit():
                raise RuntimeError('invalid sbatch job ID; reconcile intent before retry')
            record['jobs'][action].update(job_id=job_id, status='submitted', stderr=outcome.stderr)
            self.atomic_json(path, record)
            dependency = job_id
        return record

    def main(self):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('root', type=Path)
        parser.add_argument('action', choices=('groups', 'prepare', 'submit', 'extend', 'evaluate', 'advance'))
        parser.add_argument('--order', type=int, default=13)
        parser.add_argument('--index', type=int, default=0)
        parser.add_argument('--workers', type=int, default=1)
        args = parser.parse_args()
        root = args.root.resolve()
        if not 1 <= args.workers <= 192 or not 0 <= args.index < args.workers:
            raise ValueError('worker envelope exceeded')
        (root / 'logs').mkdir(exist_ok=True)
        if args.action in ('extend', 'evaluate'):
            if any(os.environ.get(k) != '1' for k in THREAD_VARIABLES):
                raise ValueError('all numerical-library thread limits must be one')
        try:
            actions = dict(groups=lambda: self.prepare_groups(root), prepare=lambda: self.prepare_stage(root, args.order),
                submit=lambda: self.submit_stage(root, args.order),
                extend=lambda: self.extend_worker(root, args.order, args.index, args.workers),
                evaluate=lambda: self.evaluate_worker(root, args.order, args.index, args.workers),
                advance=lambda: self.advance(root, args.order))
            result = actions[args.action]()
            print(json.dumps(result, indent=2, allow_nan=False), flush=True)
            if args.action in ('extend', 'evaluate'):
                job = os.environ.get('SLURM_ARRAY_JOB_ID', 'local')
                write_json_once(root / f'progress/C{args.order}/{args.action}-{job}-{args.index}.json', result)
        except Exception:
            path = root / f'failures/C{args.order}-{args.action}-{os.environ.get("SLURM_JOB_ID", "local")}-{args.index}.json'
            self.atomic_json(path, dict(traceback=traceback.format_exc(), action=args.action, order=args.order,
                                   index=args.index, accepted_outputs_retained=True))
            raise


class ContinuationRun(RefinementRun):
    policy = CONTINUATION_POLICY

    def source(self, root):
        manifest = read_manifest(root / 'inputs/source_manifest.json')
        if (manifest['base_order'], manifest['cap'], manifest['tolerance']) != (14, 18, 0.02):
            raise ValueError('authorized 2%/C18 policy changed')
        if len(manifest['selected_ids']) != 450 or manifest['parent_root'] == str(root):
            raise ValueError('separate continuation and exact selected targets required')
        return manifest

    def bootstrap(self, root):
        parent = Path(self.source(root)['parent_root'])
        runpy.run_path(str(parent / 'frozen_source/run.py'))['bootstrap']()

    def verify_deployment(self, root):
        deployment = read_manifest(root / 'deployment.json')
        parent = Path(self.source(root)['parent_root'])
        for base, mapping in ((root, deployment['files']), (parent, deployment['parent_files'])):
            for relative, expected in mapping.items():
                if digest(base / relative) != expected:
                    raise ValueError('immutable deployment changed: ' + str(base / relative))
        return deployment

    def states_before(self, root, order, src):
        if order == 15:
            return src['states'], src['fingerprint']
        previous = read_manifest(root / f'stages/C{order-1}/state.json')
        return previous['states'], previous['fingerprint']

    def group_data(self, root, descriptor):
        meta, arrays = load_arrays(Path(self.source(root)['parent_root']) / descriptor['data'])
        if meta['fingerprint'] != descriptor['data_fingerprint']:
            raise ValueError('group nodes changed')
        return arrays

    def prepare_stage(self, root, order, chunk_size=16):
        self.policy.shell_keys(order)
        if not 1 <= chunk_size <= 64:
            raise ValueError('bounded positive shell chunk size required')
        src, deployment = self.source(root), self.verify_deployment(root)
        preflight = read_manifest(root / "inputs/cache_manifest.json")
        if preflight["source_fingerprint"] != src["fingerprint"]:
            raise ValueError("cache preflight belongs to another continuation")
        groups = read_manifest(root / 'groups/manifest.json')
        states, previous_fingerprint = self.states_before(root, order, src)
        tasks, evaluations = [], []
        active = {uid for uid, st in states.items() if st['status'] == f'needs_{order}'}
        for gid, desc in groups['groups'].items():
            needed_banks = {bid: [uid for uid in src['banks'][bid]['target_ids'] if uid in active]
                            for bid in desc['banks']}
            needed_banks = {bid: ids for bid, ids in needed_banks.items() if ids}
            if not needed_banks:
                continue
            arrays = self.group_data(root, desc)
            row_ids = np.unique(np.concatenate([arrays['scatter_' + bid] for bid in needed_banks]))
            for start in range(0, len(row_ids), chunk_size):
                spec = dict(group=gid, group_fingerprint=desc['data_fingerprint'], order=order,
                    row_ids=row_ids[start:start + chunk_size].tolist(), required_banks=sorted(needed_banks),
                    prior_state=previous_fingerprint, source_fingerprint=src['fingerprint'])
                task = seal(spec)
                task['output'] = f'stages/C{order}/shells/{gid[:16]}/{task["fingerprint"]}.npz'
                tasks.append(task)
            for bid, ids in needed_banks.items():
                evaluations.append(dict(bank_id=bid, group=gid, target_ids=ids))
        if set(uid for task in evaluations for uid in task['target_ids']) != active:
            raise ValueError('evaluation manifest misses active points')
        manifest = seal(dict(version=self.policy.VERSION, order=order, window=[order-4, order-2, order],
            source_fingerprint=src['fingerprint'], prior_state=previous_fingerprint,
            deployment_fingerprint=deployment['fingerprint'], groups_fingerprint=groups['fingerprint'],
            cache_fingerprint=preflight['fingerprint'],
            tasks=tasks, evaluations=evaluations, active_point_ids=sorted(active),
            unique_row_shell_count=sum(len(t['row_ids']) for t in tasks),
            coefficient_count=sum(len(t['row_ids']) for t in tasks) * len(self.policy.shell_keys(order)),
            coefficient_degrees=[order], regenerated_parent_bank_outputs=0, rerun_original_amplitudes=0,
            validation_only_jobs=0, timing_only_jobs=0, max_workers=192, threads_per_worker=1))
        write_json_once(root / f'stages/C{order}/manifest.json', manifest)
        return {k: manifest[k] for k in ('order', 'unique_row_shell_count', 'coefficient_count', 'fingerprint')}

    def evaluate_worker(self, root, order, index, workers):
        self.verify_deployment(root)
        src, groups = self.source(root), read_manifest(root / 'groups/manifest.json')
        stage = read_manifest(root / f'stages/C{order}/manifest.json')
        states, previous = self.states_before(root, order, src)
        if stage['prior_state'] != previous:
            raise ValueError('evaluation stage prior decisions changed')
        parent_root = Path(src['parent_root'])
        cache = read_manifest(root / 'inputs/cache_manifest.json')
        if cache['fingerprint'] != stage['cache_fingerprint']:
            raise ValueError('saved cache inventory changed')
        finished = 0
        for task in stage['evaluations'][index::workers]:
            bid, gid = task['bank_id'], task['group']
            descriptor, bank_spec = groups['groups'][gid], src['banks'][bid]
            data = self.group_data(root, descriptor)
            matrix, coverage, shell_identity = self.load_group_shell(root, stage, gid, len(data['p_values']))
            scatter = data['scatter_' + bid]
            if not np.all(coverage[scatter] == 1):
                raise ValueError('incomplete required parent-bank shell coverage')
            shell = matrix[scatter]
            del matrix
            aux_relative = 'bank_aux/' + bid + '.npz'
            aux_meta, aux = load_arrays(parent_root / aux_relative)
            if (aux_meta['fingerprint'] != cache['arrays'][aux_relative]
                    or aux_meta['parent_fingerprint'] != bank_spec['parent']['fingerprint']):
                raise ValueError('saved spectral rule changed; do not rebuild it')
            if not np.array_equal(aux['p_values'].view(np.uint64), data['p_values'][scatter].view(np.uint64)):
                raise ValueError('spectral node order differs from extension rows')
            energy = next(e for e in src['energies'] if e['label'] == bank_spec['energy'])
            target_meta, targets = load_arrays(parent_root / energy['targets'])
            if target_meta['fingerprint'] != energy['targets_fingerprint']:
                raise ValueError('frozen point targets changed')
            for uid in task['target_ids']:
                require_active(states[uid]['status'], order)
                j = states[uid]['target_index']
                if targets['ids'][j] != uid or targets['family_ids'][j] != bid:
                    raise ValueError('original point/family mapping differs')
                decision_path = root / f'stages/C{order}/decisions/{uid}.json'
                if decision_path.exists():
                    prior = read_manifest(decision_path)
                    if prior['stage_fingerprint'] != stage['fingerprint']:
                        raise ValueError('decision belongs to another stage')
                    finished += 1
                    continue
                q, base = targets['q'][j], targets['base_logarithms'][j]
                context = dict(uid=uid, parent_fingerprint=bank_spec['parent']['fingerprint'],
                    source_fingerprint=src['fingerprint'], target_fingerprint=target_meta['fingerprint'],
                    q_hash=array_hash(q))
                def previous_chiral(n):
                    if n < 11 or n >= order:
                        raise ValueError('invalid saved-prefix request')
                    base_root = parent_root if n <= 14 else root
                    relative = 'contractions/' + uid + f'/C{n}.npz'
                    meta, payload = load_arrays(base_root / relative)
                    expected_source = src['parent_source_fingerprint'] if n <= 14 else src['fingerprint']
                    expected_context = dict(context, source_fingerprint=expected_source)
                    if (meta['order'] != n or any(meta['context'].get(k) != v for k, v in expected_context.items())
                            or (n <= 14 and meta['fingerprint'] != cache['arrays'][relative])
                            or payload['chiral'].shape != (len(aux['p_values']), 2)
                            or not np.isfinite(payload['chiral']).all()):
                        raise ValueError('saved contraction differs; missing prefixes must never be recomputed')
                    return payload['chiral']
                current_context = dict(context, shell_identity=shell_identity, stage_fingerprint=stage['fingerprint'])
                current = self.point_chiral(root, uid, order, current_context,
                    lambda: previous_chiral(order-1) + contract_chiral(shell, self.policy.shell_keys(order), q))
                chirals = {order: current, order-4: previous_chiral(order-4), order-2: previous_chiral(order-2)}
                logarithms = aux['logs'] + base + np.einsum('pk,k->p', aux['primary'], np.log(abs(q)), optimize=False)
                result = self.policy.nonchiral_diagnostic(chirals, logarithms, order)
                arrays = {k: np.asarray(result[k]) for k in ('totals', 'numerators', 'denominator', 'relative')}
                diagnostic_path = f'stages/C{order}/diagnostics/{uid}.npz'
                dmeta = save_arrays(root / diagnostic_path, arrays, dict(version=self.policy.VERSION, uid=uid,
                    order=order, orders=list(result['orders']), stage_fingerprint=stage['fingerprint']))
                value = complex(result['totals'][-1])
                decision = seal(dict(version=self.policy.VERSION, uid=uid, bank_id=bid, order=order,
                    status=self.policy.classify(order, result['passed']), passed=result['passed'],
                    new_baseline_I6=[value.real, value.imag], stage_fingerprint=stage['fingerprint'],
                    diagnostic=diagnostic_path, diagnostic_fingerprint=dmeta['fingerprint'],
                    relative_gaps=[float(x) if np.isfinite(x) else None for x in result['relative']],
                    representation=str(targets['representation'][j]), momentum_controls_extended=False,
                    refinement_required=not result['passed'] and order < 18,
                    stopping_reason='passed_2pct_gate' if result['passed'] else ('order_18_cap' if order == 18 else 'needs_higher_order')))
                write_json_once(decision_path, decision)
                finished += 1
            print('evaluated', order, bid, len(task['target_ids']), flush=True)
        return dict(order=order, worker=index, evaluated_points=finished)

    def advance(self, root, order):
        self.verify_deployment(root)
        src, stage = self.source(root), read_manifest(root / f'stages/C{order}/manifest.json')
        before, previous = self.states_before(root, order, src)
        if previous != stage['prior_state']:
            raise ValueError('decision barrier prior state changed')
        states = {uid: dict(st) for uid, st in before.items()}
        observed = set()
        for uid in stage['active_point_ids']:
            require_active(before[uid]['status'], order)
            d = read_manifest(root / f'stages/C{order}/decisions/{uid}.json')
            meta, arrays = load_arrays(root / d['diagnostic'])
            if (d['stage_fingerprint'] != stage['fingerprint'] or d['uid'] != uid or d['order'] != order
                    or meta['fingerprint'] != d['diagnostic_fingerprint'] or not np.isfinite(arrays['totals']).all()
                    or meta['orders'] != [order-4, order-2, order]
                    or arrays['denominator'] != abs(arrays['totals'][-1])
                    or complex(*d['new_baseline_I6']) != arrays['totals'][-1]):
                raise ValueError('invalid or nonfinite point decision')
            passed, _ = self.policy.gaps_pass(arrays['numerators'], arrays['denominator'], arrays['totals'][-1])
            if (d['passed'] != bool(passed) or d['status'] != self.policy.classify(order, bool(passed))
                    or d['refinement_required'] != (not bool(passed) and order < 18)):
                raise ValueError('point stopping decision violates fixed criterion')
            states[uid].update(status=d['status'], last_order=order, new_baseline_I6=d['new_baseline_I6'],
                final_order=order if d['status'] in self.policy.STATUSES else None,
                decision=f'stages/C{order}/decisions/{uid}.json',
                refinement_required=d['refinement_required'], convergence_pass=d['passed'],
                stopping_reason=d['stopping_reason'])
            observed.add(uid)
        if observed != {uid for uid, st in before.items() if st['status'] == f'needs_{order}'}:
            raise ValueError('missing or duplicate per-point decisions')
        counts = {status: sum(st['status'] == status for st in states.values())
                  for status in (*self.policy.STATUSES, 'needs_' + str(order + 1))}
        state = seal(dict(version=self.policy.VERSION, order=order, stage_fingerprint=stage['fingerprint'],
                          states=states, counts=counts, source_fingerprint=src['fingerprint']))
        write_json_once(root / f'stages/C{order}/state.json', state)
        if counts.get('needs_' + str(order + 1), 0):
            self.prepare_stage(root, order + 1)
            self.submit_stage(root, order + 1, dependency=os.environ.get('SLURM_JOB_ID'))
        else:
            from six_point_refinement_assembly import assemble_c18 as assemble
            assemble(root, state)
        return counts

    def submit_stage(self, root, order, dependency=None):
        with (root / '.dispatch.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            return self.submit_stage_locked(root, order, dependency)

    def submit_stage_locked(self, root, order, dependency=None):
        self.verify_deployment(root)
        stage = read_manifest(root / f'stages/C{order}/manifest.json')
        if stage['regenerated_parent_bank_outputs'] or stage['rerun_original_amplitudes']:
            raise ValueError('dispatch would replay known output')
        for task in stage['tasks']:
            self.validate_task(task, order)
        if not stage['tasks']:
            raise ValueError('no missing shell work; do not submit an empty computation')
        path = root / f'stages/C{order}/submission.json'
        record = json.loads(path.read_text()) if path.exists() else dict(version=self.policy.VERSION,
            stage_fingerprint=stage['fingerprint'], jobs={}, created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        if record['stage_fingerprint'] != stage['fingerprint']:
            raise ValueError('submission receipt belongs to another stage')
        for action, task_count in (('extend', len(stage['tasks'])), ('evaluate', len(stage['evaluations'])), ('advance', 1)):
            workers = min(192, task_count) if action != 'advance' else 1
            if action in record['jobs']:
                job = record['jobs'][action]
                if not job.get('job_id'):
                    raise RuntimeError('unreconciled submission intent; inspect scheduler before retrying')
                dependency = job['job_id']
                continue
            name = f'sel2p18-{stage["source_fingerprint"][:8]}-C{order}-{action}'
            command = ['sbatch', '--parsable', '--partition=yin', '--account=yin_lab', '--nodes=1',
                '--cpus-per-task=1', '--mem=3500M', '--time=3-00:00:00' if action == 'extend' else '--time=12:00:00',
                '--job-name=' + name, '--output=' + str(root / f'logs/C{order}-{action}-%A_%a.out')]
            if dependency:
                command.append('--dependency=afterok:' + str(dependency))
            if action != 'advance':
                command.append(f'--array=0-{workers-1}%{workers}')
            execution = ['env', 'PYTHONDONTWRITEBYTECODE=1'] + [k + '=1' for k in THREAD_VARIABLES]
            execution += [PYTHON, str(root / 'continue_flagged_samples_to_c18.py'), str(root), action, '--order', str(order), '--workers', str(workers)]
            wrap = 'exec ' + shlex.join(execution)
            if action != 'advance':
                wrap += ' --index "$SLURM_ARRAY_TASK_ID"'
            command += ['--wrap', wrap]
            record['jobs'][action] = dict(name=name, command=command, job_id=None, status='intent_recorded')
            self.atomic_json(path, record)
            outcome = subprocess.run(command, text=True, capture_output=True, check=True)
            job_id = outcome.stdout.strip().split(';')[0]
            if not job_id.isdigit():
                raise RuntimeError('invalid sbatch job ID; reconcile intent before retry')
            record['jobs'][action].update(job_id=job_id, status='submitted', stderr=outcome.stderr)
            self.atomic_json(path, record)
            dependency = job_id
        return record

    def main(self):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('root', type=Path)
        parser.add_argument('action', choices=('preflight', 'prepare', 'submit', 'extend', 'evaluate', 'advance'))
        parser.add_argument('--order', type=int, default=15)
        parser.add_argument('--index', type=int, default=0)
        parser.add_argument('--workers', type=int, default=1)
        args = parser.parse_args()
        root = args.root.resolve()
        if not 1 <= args.workers <= 192 or not 0 <= args.index < args.workers:
            raise ValueError('worker envelope exceeded')
        (root / 'logs').mkdir(exist_ok=True)
        if args.action in ('extend', 'evaluate'):
            if any(os.environ.get(k) != '1' for k in THREAD_VARIABLES):
                raise ValueError('all numerical-library thread limits must be one')
        try:
            from verify_saved_c14_continuation_inputs import preflight
            actions = dict(preflight=lambda: preflight(root), prepare=lambda: self.prepare_stage(root, args.order),
                submit=lambda: self.submit_stage(root, args.order),
                extend=lambda: self.extend_worker(root, args.order, args.index, args.workers),
                evaluate=lambda: self.evaluate_worker(root, args.order, args.index, args.workers),
                advance=lambda: self.advance(root, args.order))
            result = actions[args.action]()
            print(json.dumps(result, indent=2, allow_nan=False), flush=True)
            if args.action in ('extend', 'evaluate'):
                job = os.environ.get('SLURM_ARRAY_JOB_ID', 'local')
                write_json_once(root / f'progress/C{args.order}/{args.action}-{job}-{args.index}.json', result)
        except Exception:
            path = root / f'failures/C{args.order}-{args.action}-{os.environ.get("SLURM_JOB_ID", "local")}-{args.index}.json'
            self.atomic_json(path, dict(traceback=traceback.format_exc(), action=args.action, order=args.order,
                                   index=args.index, accepted_outputs_retained=True))
            raise


initial = RefinementRun()
continuation = ContinuationRun()
initial_source = initial.source
initial_bootstrap = initial.bootstrap
initial_verify_deployment = initial.verify_deployment
initial_prepare_groups = initial.prepare_groups
initial_states_before = initial.states_before
initial_group_data = initial.group_data
initial_prepare_stage = initial.prepare_stage
initial_construct_shell = initial.construct_shell
initial_validate_task = initial.validate_task
initial_read_shell = initial.read_shell
initial_extend_worker = initial.extend_worker
initial_load_group_shell = initial.load_group_shell
initial_point_chiral = initial.point_chiral
initial_evaluate_worker = initial.evaluate_worker
initial_advance = initial.advance
initial_atomic_json = initial.atomic_json
initial_submit_stage = initial.submit_stage
initial_main = initial.main
continuation_source = continuation.source
continuation_bootstrap = continuation.bootstrap
continuation_verify_deployment = continuation.verify_deployment
continuation_states_before = continuation.states_before
continuation_group_data = continuation.group_data
continuation_prepare_stage = continuation.prepare_stage
continuation_construct_shell = continuation.construct_shell
continuation_validate_task = continuation.validate_task
continuation_read_shell = continuation.read_shell
continuation_extend_worker = continuation.extend_worker
continuation_load_group_shell = continuation.load_group_shell
continuation_point_chiral = continuation.point_chiral
continuation_evaluate_worker = continuation.evaluate_worker
continuation_advance = continuation.advance
continuation_atomic_json = continuation.atomic_json
continuation_submit_stage = continuation.submit_stage
continuation_submit_stage_locked = continuation.submit_stage_locked
continuation_main = continuation.main


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('initial', 'continue'))
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return (initial if args.phase == 'initial' else continuation).main()

if __name__ == '__main__':
    main()
