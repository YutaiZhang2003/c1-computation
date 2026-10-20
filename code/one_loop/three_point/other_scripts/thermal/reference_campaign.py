"""Build fresh thermal banks within a twelve-node budget; read older results."""

from genus1_source_paths import ROOT as _SOURCE_ROOT, source_path as _source_path
import genus1_source_paths as bootstrap
bootstrap.install()
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json
import math
import os
import subprocess
import sys
import time
import numpy as np
from banks import make_spec, register, build_reference, primary_case, height_case
from genus1_boundary_spectrum import publish, read
from genus1_moduli_sampling import ShortPoint
from genus1_native_bank import digest
from kernels import ThermalKinematics
from evaluation import Evaluator

DEFAULT = bootstrap.ROOT/'output/thermal_refinement_20260927'


def cases(kin, fixed_frequency=False):
    rows = []
    geometries = [('middle_021',(0,2,1),1.2,2.4), ('long_021',(0,2,1),8.,32.)]
    if not fixed_frequency:
        geometries += [('middle_012',(0,1,2),1.2,2.4), ('middle_201',(2,0,1),1.2,2.4),
                       ('short_021',(0,2,1),.08,3.)]
    for label, sector, s1, s2 in geometries:
        short = ShortPoint(sector,s1,s2,0.,0.)
        rows.append((label,primary_case(short)))
    for pair in ((0,1),(0,2)):
        for kind in ('triple_corner','triple_scale_face'):
            rows.append((kind+'_tail_'+''.join(map(str,pair)), height_case(kind,pair=pair)))
    if not fixed_frequency:
        for pair in ((0,1),(0,2)):
            tau = .2+128j
            total = 2*math.pi*(.31+tau*.07)
            rows.append(('pair_T128_'+''.join(map(str,pair)), dict(id='bank',kind='pair',
                coefficient_orders=[0,2,2],geometry=dict(pair=list(pair),radius=.125,
                    tau=[tau.real,tau.imag],delta_total=[total.real,total.imag]))))
    return rows


def plan(output):
    entries = []
    families = [('t035',ThermalKinematics(.35,2/.35)),
                ('t075',ThermalKinematics(.75,2/.75)),
                ('t095',ThermalKinematics(.95,2/.95)),
                ('fixed_N2',ThermalKinematics(.75,16/3,2)),
                ('fixed_N3',ThermalKinematics(.75,8.,3))]
    for tag,kin in families:
        for label,case in cases(kin,tag.startswith('fixed')):
            for order in (12,):
                spec = make_spec(kin,case,order)
                key = register(output,spec)
                entries.append(dict(tag=tag,label=label,kinematics=kin.record(),
                                    order=order,key=key,kind=case['kind']))
    result = dict(format='thermal-reference-campaign-v1',entries=entries,
        maximum_nodes_per_axis=12,
        comparison_columns=['noncompact','thermal','thermal_correction'],
        no_matrix_model_in_grid_design=True, full_amplitude_certified=False,
        scope='Pointwise spectral convergence at identical remaining moduli; not a moduli integral.')
    publish(output/'reference_plan.json',result)
    return result


def worker(output,index):
    p = read(output/'reference_plan.json'); row = p['entries'][index]
    if row['order']>p.get('maximum_nodes_per_axis',row['order']):
        raise ValueError('reference task exceeds its frozen physical node budget')
    target = output/'reference_results'/f'{index:04d}.json'
    if target.exists():
        old = read(target)
        if old['entry'] != row:
            raise ValueError('changed reference task')
        return old
    if row['order']>12:
        raise ValueError('new CFT evaluations are capped at twelve nodes per momentum axis')
    started = time.monotonic(); attempts = []
    # Each call is bounded and commits complete coefficient chunks/node checks.
    for attempt in range(32):
        result = build_reference(output,row['key'],cpu_budget=45.)
        attempts.append(result)
        if result['complete_rule']:
            break
        failure = result['failure'] or {}
        if failure.get('type') != 'TimeLimit':
            raise RuntimeError(json.dumps(failure))
    else:
        raise RuntimeError('reference preparation exceeded the explicit CPU allocation')
    k = row['kinematics']; kin = ThermalKinematics(k['t'],k['R'],k['N'])
    evaluator = Evaluator(output,kin)
    try:
        spec,_,_,_ = evaluator.load(row['key'])
        if row['kind'] == 'primary':
            g = spec['case']['geometry']['short']
            shorts = [ShortPoint(tuple(g['sector']),g['s1'],g['s2'],x,y)
                      for x,y in ((.7,2.1),(2.3,5.2),(5.4,.4))]
            values = evaluator.primary(row['key'],shorts)
            prefixes = evaluator.primary(row['key'],shorts,order=2)
            geometry = [s.record() for s in shorts]
        else:
            values = (evaluator.height if row['kind'].startswith('triple_') else evaluator.pair)(row['key'])
            prefixes = None; geometry = spec['case']['geometry']
    finally:
        evaluator.close()
    result = dict(entry=row,values=values.tolist(),geometry=geometry,
        H2_values=None if prefixes is None else prefixes.tolist(),
        height_weight_rounding=evaluator.height_weight_rounding,
        preparation_attempts=attempts,wall_seconds=time.monotonic()-started,
        evaluation_sources={str(p.relative_to(bootstrap.ROOT)):digest(p) for p in
            (Path(__file__),_source_path(Path('code/one_loop/thermal') / 'evaluation.py'))},
        full_amplitude_certified=False)
    publish(target,result)
    print(json.dumps(dict(index=index,tag=row['tag'],label=row['label'],order=row['order'],
                          seconds=result['wall_seconds'])),flush=True)
    return result


def run_local(output,workers):
    p = read(output/'reference_plan.json')
    folder = output/'logs';folder.mkdir(exist_ok=True)
    def run(index):
        with (folder/f'reference_{index:04d}.log').open('a') as stream:
            result = subprocess.run([sys.executable,__file__,'worker','--output',str(output),
                                     '--index',str(index)],stdout=stream,stderr=subprocess.STDOUT)
        return index,result.returncode
    failed = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(run,i) for i in range(len(p['entries']))]):
            index,code = future.result()
            print(json.dumps(dict(index=index,exit_code=code)),flush=True)
            if code: failed.append(index)
    if failed: raise RuntimeError('failed tasks: '+str(failed))


def collect(output):
    p = read(output/'reference_plan.json'); groups={}; missing=[]
    for i,row in enumerate(p['entries']):
        path = output/'reference_results'/f'{i:04d}.json'
        if not path.exists(): missing.append(i);continue
        result=read(path)
        if result['entry']!=row: raise ValueError('reference result identity mismatch')
        groups.setdefault((row['tag'],row['label']),{})[row['order']]=result
    rows=[]
    for (tag,label),group in groups.items():
        record=dict(tag=tag,label=label,orders=sorted(group),values={str(n):r['values'] for n,r in group.items()})
        if set(group)=={8,12,16}:
            a,b,c=(np.asarray(group[n]['values']) for n in (8,12,16))
            scale=np.maximum(abs(c),1e-300)
            record.update(change_8_to_12=((b-a)/scale).tolist(),change_12_to_16=((c-b)/scale).tolist(),
                max_relative_change_12_to_16=np.max(abs(c-b)/scale,axis=0).tolist())
        rows.append(record)
    result=dict(complete=not missing,missing_indices=missing,rows=rows,
        scope=p['scope'],momentum_accuracy_not_inferred_from_sampling_error=True,
        full_amplitude_certified=False)
    (output/'reference_comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    a=argparse.ArgumentParser();a.add_argument('mode',choices=('plan','worker','run','collect','slurm'))
    a.add_argument('--output',type=Path,default=DEFAULT);a.add_argument('--index',type=int)
    a.add_argument('--workers',type=int,default=4);args=a.parse_args();args.output=args.output.resolve()
    if args.mode=='plan': print(json.dumps(dict(tasks=len(plan(args.output)['entries']))))
    elif args.mode=='worker': worker(args.output,args.index)
    elif args.mode=='run':run_local(args.output,args.workers)
    elif args.mode=='slurm':
        rank=int(os.environ['SLURM_PROCID']);size=int(os.environ['SLURM_NTASKS'])
        entries=read(args.output/'reference_plan.json')['entries']
        indices=sorted(range(len(entries)),key=lambda i:(-entries[i]['order'],i))
        failures=[]
        for i in indices[rank::size]:
            try:worker(args.output,i)
            except Exception as exc:
                failures.append(dict(index=i,type=type(exc).__name__,message=str(exc)))
                print(json.dumps(failures[-1]),flush=True)
        if failures:
            publish(args.output/'failures'/f'rank_{rank:03d}.json',failures)
            raise SystemExit(1)
    else: print(json.dumps(dict(complete=collect(args.output)['complete'])))


if __name__=='__main__': main()
