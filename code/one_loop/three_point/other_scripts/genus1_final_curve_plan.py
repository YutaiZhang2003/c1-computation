"""Freeze the accepted pair-cell design for twenty energies with exact reuse.

No CFT is evaluated while planning. The four pair rules come directly from
verified experiments; all other geometry is copied from the completed curve.
"""

from genus1_source_paths import ROOT as _SOURCE_ROOT, source_path as _source_path
import argparse
from collections import Counter
from copy import deepcopy
import json
import math
from pathlib import Path
import shutil

import genus1_amplitude_curve as curve
import genus1_pair_cell_refinement as refinement


def artifact(root, relative):
    for base in (root, root/'evidence', root/'evidence_snapshot'):
        path=base/relative
        if path.is_file():return path
    raise FileNotFoundError(root/relative)


def compact_replicates(root, p, locations, f):
    observations={}
    for i,location in enumerate(locations):
        row=f.spectrum.read(artifact(root,f'compact/points/{i:02d}/value.json'))
        if row['plan_sha256']!=f.fingerprint(p) or row['location']!=location:
            raise ValueError('foreign compact observation in reuse')
        key=(location['replicate'],location['index'])
        if key in observations:raise ValueError('duplicate compact observation')
        observations[key]=row['value']['value'][0]
    if set(observations)!={(i,j) for i in range(16) for j in range(4)}:
        raise ValueError('incomplete paired compact samples')
    return [math.fsum(observations[i,j] for j in range(4))/4 for i in range(16)]


def verify_replicates(means, compact):
    mean=math.fsum(means)/16
    variance=math.fsum((v-mean)**2 for v in means)/(15*16)
    refinement.close(mean,compact['mean'])
    if not math.isclose(variance,compact['sampling_se']**2,rel_tol=2e-11,abs_tol=1e-40):
        raise ValueError('retained compact uncertainty is inconsistent')


def add_saved_pair(builder, parent, names, receipt, f):
    """Pin both physical weights and geometry to already verified CFT values."""
    d=f.spectrum.read(parent/'design.json')
    if f.fingerprint(d)!=receipt['design_sha256']:
        raise ValueError('changed pair experiment design')
    rule_path=artifact(parent,d['rules']['path'])
    if f.shared.sha(rule_path)!=d['rules']['sha256']:
        raise ValueError('changed pair rule weights')
    rules=f.spectrum.read(rule_path)
    needed={k for name in names for k,_,_ in rules[name]}
    samples,values={},{}
    for rel,digest in receipt['verified_batches'].items():
        path=parent/rel
        if f.shared.sha(path)!=digest:raise ValueError('changed saved pair values')
        index=int(path.stem.split('_')[-1]);b=d['batches'][index]
        geometry=artifact(parent,b['path'])
        if f.shared.sha(geometry)!=b['sha256']:raise ValueError('changed pair geometry')
        row=f.spectrum.read(path);ss=f.spectrum.read(geometry)['samples']
        if row['design_sha256']!=f.fingerprint(d) or set(row['values'])!=set(ss):
            raise ValueError('incomplete saved pair batch')
        for k in needed.intersection(ss):
            if k in samples:raise ValueError('overlapping pair evidence')
            samples[k]=ss[k];values[k]=f.Value.from_record(row['values'][k])
    if set(samples)!=needed:raise ValueError('missing pair rule points')
    locations=json.loads((parent/'locations.json').read_text())
    spec_roots=[parent,Path(locations['source'])]
    spec_roots.extend(Path(v['local']) for v in locations.get('parents',[]))
    for root in list(spec_roots):
        path=root/'locations.json'
        if path.exists():
            spec_roots.extend(Path(v['local']) for v in json.loads(path.read_text()).get('parents',[]))
    totals={}
    for name,label in names.items():
        for k,w,tag in rules[name]:
            s=samples[k]
            if f.fingerprint(dict(kind='collision',requests=s['requests']))!=k or not math.isfinite(w) or w<=0:
                raise ValueError('changed saved pair sample or measure')
            # Reference bank IDs describe stored observations. Reconstructing
            # their Gauss nodes under another NumPy version can change ulps and
            # therefore hashes. Preserve the exact stored IDs and specifications;
            # every genuinely new energy receives its own explicitly planned bank.
            if k not in builder.samples:
                builder.samples[k]=dict(s,max_weight=w);builder.weights[k]=[]
            builder.samples[k]['max_weight']=max(builder.samples[k]['max_weight'],w)
            builder.weights[k].append([label,w,tag]);builder.rules[label]+=1
            for grid in s['grids']:
                if grid in builder.grids:continue
                found=None
                for root in spec_roots:
                    for prefix in (root,root/'evidence'):
                        path=f.spectrum.grid_directory(prefix,grid)/'spec.json'
                        if path.exists():found=path;break
                    if found is not None:break
                if found is None:raise FileNotFoundError('missing accepted pair specification '+grid)
                spec=f.spectrum.read(found);f.check_momentum_grid(builder.p,spec)
                if f.fingerprint(spec['grid'])!=grid or spec['case']['kind']!='pair':
                    raise ValueError('foreign saved pair specification')
                builder.grids[grid]=dict(kind='pair',axes=[len(r['nodes']) for r in spec['grid']['rules']])
        totals[label]=f.combine((values[k] for k,_,_ in rules[name]),
                                (w for _,w,_ in rules[name])).record()
    return totals


def plan_energy(t):
    (root,old,od,om,mesh,pc,all_batches,compact_rows,grids,old_by_t,costs)=_ENERGY_CONTEXT
    f=curve.f
    old_identity=f.fingerprint(od)
    prep_units=[];prep_costs=[];contract_units=[];contract_costs=[];groups=[]
    label=f't{round(1000*t):04d}';eroot=root/'energies'/label;p=curve.energy_plan(t)
    if (eroot/'design.json').exists():
        ed=f.spectrum.read(eroot/'design.json')
        if ed['plan_sha256']!=f.fingerprint(p) or f.spectrum.read(eroot/'plan.json')!=p:
            raise ValueError('foreign completed energy plan')
        specs={}
        for key in ed['banks']:
            path=f.spectrum.grid_directory(eroot,key)/'spec.json'
            if f.shared.sha(path)!=ed['spec_sha256'][key]:raise ValueError('changed planned bank')
            specs[key]=f.spectrum.read(path)
        mapping=ed['grid_map'];indices=ed['batch_indices'];retained_digest=ed['retained_sha256']
    else:
        f.spectrum.publish(eroot/'plan.json',p)
        f.spectrum.publish(eroot/'compact/locations.json',dict(plan_sha256=f.fingerprint(p),locations=compact_rows))
        retained_digest=None
        if t in old_by_t:
            entry=old_by_t[t];prior=old/'energies'/entry['label']
            a=f.spectrum.read(prior/'amplitude.json')
            receipt=f.spectrum.read(old/'sequential96/completed'/f"{entry['label']}.json")
            if (a['curve_sha256']!=old_identity or a['energy_design_sha256']!=entry['design_sha256'] or
                    a['plan_sha256']!=f.fingerprint(p) or not a['complete_domain'] or
                    receipt['amplitude_sha256']!=f.shared.sha(prior/'amplitude.json')):
                raise ValueError('foreign completed amplitude in reuse')
            if {c['cell']['id'] for c in a['cells']}!={c['cell']['id'] for c in om['cells']}:
                raise ValueError('retained amplitude has a different partition')
            pairs=[dict(c,moduli_points=sum(next(mc['rule_points'] for mc in om['cells']
                     if mc['cell']['id']==c['cell']['id']).values())) for c in a['cells'] if c['cell']['term']==refinement.TERM]
            means=compact_replicates(artifact(old,f"energies/{entry['label']}/compact/points/00/value.json").parents[3],p,compact_rows,f)
            verify_replicates(means,a['compact'])
            retained=dict(plan_sha256=f.fingerprint(p),amplitude=a,pair_cells=pairs,
                compact_replicate_means=means,source_curve_sha256=old_identity,
                source_amplitude_sha256=f.shared.sha(prior/'amplitude.json'))
            f.spectrum.publish(eroot/'retained.json',retained)
            retained_digest=f.shared.sha(eroot/'retained.json');indices=pc['batch_indices']
        else:indices=list(range(len(mesh['batches'])))
        planner=f.spectrum.Planner(p,f.make_atlas(t));mapping={};specs={};cache={}
        for i in indices:
            for s in all_batches[i]['samples'].values():
                primary=s['kind']=='primary';records=[s['short']] if primary else s['requests']
                for oldkey,record in zip(s['grids'],records):
                    cache_key=(('primary',tuple(record['sector']),record['s1'],record['s2']) if primary
                               else ('collision',f.fingerprint(record)))
                    if cache_key not in cache:
                        value=f.ShortPoint.from_record(record) if primary else f.request_from_record(record)
                        key,spec=planner.grid(value,primary,p['profile']);f.check_momentum_grid(p,spec)
                        cache[cache_key]=key
                        if key not in specs:
                            specs[key]=spec;f.spectrum.publish(f.spectrum.grid_directory(eroot,key)/'spec.json',spec)
                    key=cache[cache_key]
                    if oldkey in mapping and mapping[oldkey]!=key:raise ValueError('reference bank alias split at new energy')
                    mapping[oldkey]=key
        needed={g for i in indices for s in all_batches[i]['samples'].values() for g in s['grids']}
        if set(mapping)!=needed or set(mapping.values())&set(grids):
            raise ValueError('incomplete mapping or cross-energy bank reuse')
    banks=sorted(specs)
    ed=dict(t=t,label=label,plan_sha256=f.fingerprint(p),grid_map=mapping,banks=banks,
        batch_indices=indices,retained_sha256=retained_digest,
        spec_sha256={k:f.shared.sha(f.spectrum.grid_directory(eroot,k)/'spec.json') for k in banks},
        compact_locations_sha256=f.shared.sha(eroot/'compact/locations.json'),
        compact_points_to_compute=0 if retained_digest else 64,
        banks_by_family=dict(Counter(s['case']['kind'] for s in specs.values())),
        banks_by_axes=dict(Counter(str(tuple(len(r['nodes']) for r in s['grid']['rules'])) for s in specs.values())))
    f.spectrum.publish(eroot/'design.json',ed)
    entry=dict(t=t,label=label,design_sha256=f.fingerprint(ed))
    for key in banks:
        spec=specs[key];prep_units.append(dict(energy=label,kind='bank',key=key))
        prep_costs.append((100. if spec['case']['kind']=='primary' else 20.3)*math.prod(len(r['nodes']) for r in spec['grid']['rules'])/512)
    if not retained_digest:
        for i in range(64):prep_units.append(dict(energy=label,kind='compact',index=i));prep_costs.append(180.)
    for i in indices:
        contract_units.append(dict(energy=label,kind='contract',index=i));contract_costs.append(costs[i])
        groups.append((label,mesh['batches'][i]['coordinate_group']))
    print(json.dumps(dict(t=t,banks=len(banks),moduli_points=sum(mesh['batches'][i]['points'] for i in indices),
                          retained_components=bool(retained_digest))),flush=True)
    return dict(entry=entry,prepare_units=prep_units,prepare_costs=prep_costs,
                contract_units=contract_units,contract_costs=contract_costs,groups=groups,
                retained=bool(retained_digest))


def prepare(args):
    root=args.root.resolve();old=args.old_curve.resolve();fix=args.pair_fix.resolve()
    if args.resume_planning:
        if (root/'curve.json').exists():raise ValueError('cannot change a frozen curve')
        base=curve.configure(root);f=curve.f
        od=f.spectrum.read(old/'curve.json');old_identity=f.fingerprint(od)
        om=f.spectrum.read(old/'mesh.json');fd=f.spectrum.read(fix/'design.json')
        mesh=f.spectrum.read(root/'mesh.json');pc=next(c for c in mesh['cells'] if c.get('directional_pair'))
        validation=f.spectrum.read(root/'validation.json')
        if validation['previous_curve_sha256']!=old_identity or validation['pair_fix_sha256']!=f.fingerprint(fd):
            raise ValueError('changed scientific inputs while resuming planning')
        compact_rows=f.spectrum.read(curve.source/'compact/locations.json')['locations']
        grids=mesh['reference_grids']
        costs=[b['points']*(.111 if next(c['cell']['term'] for c in mesh['cells']
                if c['cell']['id']==b['cell_id']).startswith('primary:') else .080) for b in mesh['batches']]
        for name in ('genus1_amplitude_curve.py','genus1_final_curve_plan.py',
                     'genus1_moduli_amplitude.py','genus1_moduli_panel_campaign.py',
                     'genus1_pair_cell_refinement.py','genus1_moduli_amplitude_report.py'):
            shutil.copy2(_source_path(Path('code/tasks') / name),root/'pilot'/name)
    else:
        root.mkdir(parents=True,exist_ok=False);(root/'pilot').mkdir()
        for name in ('genus1_amplitude_curve.py','genus1_final_curve_plan.py',
                     'genus1_moduli_amplitude.py','genus1_moduli_panel_campaign.py',
                     'genus1_pair_cell_refinement.py','genus1_moduli_amplitude_report.py'):
            shutil.copy2(_source_path(Path('code/tasks') / name),root/'pilot'/name)
        locations=json.loads((old/'locations.json').read_text())
        locations.update(previous_curve=str(old),previous_curve_remote=args.old_curve_remote,
                         pair_fix=str(fix),pair_fix_remote=args.pair_fix_remote)
        (root/'locations.json').write_text(json.dumps(locations,indent=2)+'\n')
        base=curve.configure(root);f=curve.f
        od=f.spectrum.read(old/'curve.json');old_identity=f.fingerprint(od)
        if (od['source_plan_sha256']!=f.fingerprint(base) or
                not f.spectrum.read(old/'curve_result.json')['complete']):
            raise ValueError('completed curve with the same scientific runtime required')
        if f.shared.sha(old/'mesh.json')!=od['files']['mesh.json']:
            raise ValueError('changed completed moduli mesh')
        om=f.spectrum.read(old/'mesh.json')
        fd=f.spectrum.read(fix/'design.json');fa=f.spectrum.read(fix/'amplitude.json')
        assessment=f.spectrum.read(fix/'assessment.json');imports=f.spectrum.read(fix/'verified_imports.json')
        accepted=f.spectrum.read(fix/'candidate_rule.json')
        if (not assessment['accepted'] or fa['design_sha256']!=f.fingerprint(fd) or
                fd['source_plan_sha256']!=f.fingerprint(base) or not fa['complete_domain']):
            raise ValueError('accepted pair-cell correction is required')
        cell=accepted['cell'];target=cell['id']
        if sum(c['cell']==cell for c in om['cells'])!=1:
            raise ValueError('accepted pair support differs from the completed curve')
        mesh=dict(cells=[],batches=[],component_tolerances=om['component_tolerances'])
        costs=[];grids={}
        for c in om['cells']:
            if c['cell']['id']==target:continue
            nc=deepcopy(c);nc['batch_indices']=[]
            for index in c['batch_indices']:
                b=om['batches'][index];path=old/'mesh'/b['path']
                if f.shared.sha(path)!=b['sha256']:raise ValueError('changed retained geometry')
                dest=root/'mesh'/b['path'];dest.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(path,dest)
                row=f.spectrum.read(path)
                for s in row['samples'].values():
                    for k in s['grids']:grids[k]=om['reference_grids'][k]
                nc['batch_indices'].append(len(mesh['batches']));mesh['batches'].append(b)
                costs.append(b['points']*(.111 if c['cell']['term'].startswith('primary:') else .080))
            mesh['cells'].append(nc)
        builder=curve.production.Builder(root/'mesh/pair',base,f.make_atlas(.25),64)
        fix_locations=json.loads((fix/'locations.json').read_text())
        parent=Path(fix_locations['parents'][0]['local'])
        totals=add_saved_pair(builder,parent,dict(pair_h3_p3='high',pair_h4_p3='height',pair_h4_p4='puncture'),imports['parent'],f)
        totals.update(add_saved_pair(builder,fix,dict(pair_r3_h3_p3='real'),assessment['probe'],f))
        builder.finish_cell(cell)
        if len(builder.cells)!=1:raise ValueError('one pair replacement required')
        pc=deepcopy(builder.cells[0]);pc['directional_pair']=True;pc['batch_indices']=[]
        for b,cost in zip(builder.batches,builder.costs):
            pc['batch_indices'].append(len(mesh['batches']))
            mesh['batches'].append(dict(b,path='pair/'+b['path']));costs.append(cost)
        mesh['cells'].append(pc);grids.update(builder.grids)
        mesh.update(points=sum(b['points'] for b in mesh['batches']),reference_grids=grids,
                    replaced_cell=cell,candidate_orders=[2,3,3,3])
        if len(mesh['cells'])!=74 or mesh['points']>75000 or len(grids)>6500:
            raise ValueError('final fixed-design work census exceeds its bound')
        f.spectrum.publish(root/'mesh.json',mesh)
        base_value=totals['high']['value'][0]
        delta=dict(torus_real=totals['real']['value'][0]-base_value,
                   height=totals['height']['value'][0]-base_value,
                   puncture=totals['puncture']['value'][0]-totals['height']['value'][0])
        refinement.close(base_value,fa['replacement']['candidate']['value']['value'][0])
        refinement.close(math.fsum(abs(v) for v in delta.values()),assessment['indicator'])
        compact_rows=f.spectrum.read(curve.source/'compact/locations.json')['locations']
        anchor=deepcopy(fa)
        anchor['compact_replicate_means']=compact_replicates(curve.source,base,compact_rows,f)
        verify_replicates(anchor['compact_replicate_means'],anchor['compact'])
        anchor['pair_diagnostics']=dict(cell_id=target,directional_deltas=delta,
            relative_indicator=assessment['relative_indicator'],local_stability_flag=False)
        f.spectrum.publish(root/'anchor.json',dict(t=.25,amplitude=anchor,
            anchor_amplitude_sha256=f.shared.sha(fix/'amplitude.json'),anchor_design_sha256=f.fingerprint(fd)))
        f.spectrum.publish(root/'validation.json',dict(unchanged_cells=73,replaced_cells=1,
            identical_retained_geometry=True,pair_candidate_replay=base_value,
            pair_directional_replay=math.fsum(abs(v) for v in delta.values()),cft_recomputed=False,
            previous_curve_sha256=old_identity,pair_fix_sha256=f.fingerprint(fd)))
    all_batches=[curve.payload(root,mesh,i) for i in range(len(mesh['batches']))]
    energies=[];prep_units=[];prep_costs=[];contract_units=[];contract_costs=[];groups=[]
    old_by_t={e['t']:e for e in od['energies']}
    reused=0
    global _ENERGY_CONTEXT
    _ENERGY_CONTEXT=(root,old,od,om,mesh,pc,all_batches,compact_rows,grids,old_by_t,costs)
    # The analytic bridge rule is pure but its caller adds role metadata. Cache
    # its expensive moment recurrence and copy each result before mutation.
    from functools import lru_cache
    import genus1_projection_momentum as momentum
    original_bridge=momentum.radial_bridge_rule
    cached_bridge=lru_cache(maxsize=256)(original_bridge)
    def bridge(order, *, decay, mass_squared):
        return deepcopy(cached_bridge(order,decay=decay,mass_squared=mass_squared))
    momentum.radial_bridge_rule=bridge
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    with ProcessPoolExecutor(max_workers=4,mp_context=get_context('fork')) as pool:
        for row in pool.map(plan_energy,[t for t in curve.FINAL_T_VALUES if t!=.25]):
            energies.append(row['entry']);reused+=int(row['retained'])
            prep_units.extend(row['prepare_units']);prep_costs.extend(row['prepare_costs'])
            contract_units.extend(row['contract_units']);contract_costs.extend(row['contract_costs'])
            groups.extend(row['groups'])
    momentum.radial_bridge_rule=original_bridge
    files={str(p.relative_to(root)):f.shared.sha(p) for p in (root/'pilot').glob('*.py')}
    for rel in ('mesh.json','anchor.json','validation.json'):files[rel]=f.shared.sha(root/rel)
    d=dict(format=curve.FINAL_FORMAT,created_utc=f.shared.utc(),t_values=curve.FINAL_T_VALUES,
        source_plan_sha256=f.fingerprint(base),anchor_design_sha256=f.fingerprint(fd),
        files=files,workers=384,partition='shared',energies=energies,
        prepare_units=prep_units,contract_units=contract_units,
        summary=dict(amplitudes=20,fresh_full_amplitudes=10,partially_reused_amplitudes=reused,
            imported_amplitudes=1,new_boundary_banks=sum(u['kind']=='bank' for u in prep_units),
            fresh_compact_points=sum(u['kind']=='compact' for u in prep_units),
            contraction_batches=len(contract_units),mesh_points=mesh['points'],pair_rule_points=pc['rule_points']),
        pole_free_chamber=od['pole_free_chamber'],automatic_refinement=False,
        momentum_design_changed=False,physical_cutoffs_changed=False,
        unknown_errors=od['unknown_errors'],full_amplitude_certified=False)
    f.spectrum.publish(root/'curve.json',d)
    allocation={};summary={}
    for entry in energies:
        label=entry['label'];allocation[label]={};summary[label]={}
        for phase,units,cost_values in (('prepare',prep_units,prep_costs),('contract',contract_units,contract_costs)):
            ix=[i for i,u in enumerate(units) if u['energy']==label]
            if phase=='prepare':rows,loads=curve.campaign.assign([cost_values[i] for i in ix],384)
            else:rows,loads=curve.production.assign_geometry_groups([cost_values[i] for i in ix],[groups[i] for i in ix],384)
            allocation[label][phase]=[[ix[i] for i in row] for row in rows]
            summary[label][phase]=dict(units=len(ix),estimated_max_seconds=max(loads),active_workers=sum(bool(row) for row in rows))
    sequence=dict(format='genus1-sequential-curve-v1',curve_sha256=f.fingerprint(d),workers=384,
        order=[e['label'] for e in energies],allocation=allocation,summary=summary,
        files={k:v for k,v in files.items() if k.startswith('pilot/')},automatic_refinement=False)
    curve.validate_sequence(d,sequence)
    f.spectrum.publish(root/'sequential/execution.json',sequence)
    print(json.dumps(dict(curve_sha256=f.fingerprint(d),execution_sha256=f.fingerprint(sequence),
                          **d['summary']),indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume-planning',action='store_true')
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--old-curve',type=Path,required=True)
    parser.add_argument('--old-curve-remote',required=True)
    parser.add_argument('--pair-fix',type=Path,required=True)
    parser.add_argument('--pair-fix-remote',required=True)
    prepare(parser.parse_args())
