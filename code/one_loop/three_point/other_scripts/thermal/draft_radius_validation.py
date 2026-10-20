"""Independent 8-versus-12-node checks at fixed moduli, without a model target."""
import genus1_source_paths as bootstrap
bootstrap.install()
import argparse
import json
import os
from pathlib import Path
import numpy as np
from draft_radius_campaign import OUTPUT,read,publish,ThermalKinematics,ShortPoint
from paired_quadrature import make_paired_spec
from paired_quadrature import make_placement_spec as primary_spec
from banks import register
from evaluation import Evaluator
from reference_campaign import worker

FOLDER=OUTPUT/'momentum_validation'


def plan():
    original=read(OUTPUT/'reference_plan.json');entries=[]
    for tag in ('t0950','t0750','t0250'):
        chosen=[]
        for e in original['entries']:
            if e['tag']!=tag:continue
            label=e['label']
            if (label.startswith('primary_0_') or label=='primary_1_021'
                or label in ('pair_bulk_0_1','pair_bulk_0_2','pair_tail_0_1','pair_tail_0_2',
                             'triple_corner_tail_1','triple_scale_face_tail_2')):
                chosen.append(e)
        assert len(chosen)==10
        for e in chosen:
            k=e['kinematics'];kin=ThermalKinematics(k['t'],k['R'])
            old=read(OUTPUT/'prepared/spectrum/grids'/e['key']/'spec.json');case=old['case']
            if e['kind']=='primary':
                spec=primary_spec(kin,ShortPoint.from_record(case['geometry']['short']),
                                  e['placement_variant'],order=8)
            else:spec=make_paired_spec(kin,case,8)
            key=register(FOLDER,spec)
            entries.append(dict(e,key=key,order=8,production_key=e['key']))
    p=dict(format='thermal-reference-campaign-v1',entries=entries,maximum_nodes_per_axis=12,
        matrix_model_used=False,scope='Pointwise coupled momentum checks at fixed moduli; not full-amplitude error bounds.')
    publish(FOLDER/'reference_plan.json',p)
    print(json.dumps(dict(validation_grids=len(entries))))


def compare(production_banks=None):
    results=[]
    for index,row in enumerate(read(FOLDER/'reference_plan.json')['entries']):
        reference=read(FOLDER/'reference_results'/f'{index:04d}.json')
        k=row['kinematics'];e=Evaluator(OUTPUT if production_banks is None else production_banks,
                                     ThermalKinematics(k['t'],k['R']))
        e.output=FOLDER/'production_geometry'
        try:
            if row['kind']=='primary':
                shorts=[ShortPoint.from_record(g) for g in reference['geometry']]
                high=e.primary(row['production_key'],shorts)
            else:
                high=(e.height if row['kind'].startswith('triple_') else e.pair)(row['production_key'])
            low=np.asarray(reference['values'])
            results.append(dict(tag=row['tag'],kind=row['kind'],label=row['label'],
                values_8=low.tolist(),values_12=high.tolist(),change=(high-low).tolist(),
                relative_correction_change=(abs(high[:,2]-low[:,2])/np.maximum(abs(high[:,2]),1e-300)).tolist()))
        finally:e.close()
    publish(FOLDER/'comparison.json',dict(rows=results,complete=len(results)==30,
        matrix_model_used=False,full_amplitude_certified=False))


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('plan','slurm','compare'))
    p.add_argument('--production-banks',type=Path);a=p.parse_args()
    if a.mode=='plan':plan()
    elif a.mode=='compare':compare(a.production_banks)
    else:
        rank=int(os.environ['SLURM_PROCID']);size=int(os.environ['SLURM_NTASKS'])
        for i in range(rank,len(read(FOLDER/'reference_plan.json')['entries']),size):worker(FOLDER,i)


if __name__=='__main__':main()
