"""Kernel-specific spectral inputs for the staged amplitude controller.

Planning derives rules; explicit bank jobs construct coefficients. Evaluation
opens pinned completed exports only. A grid is shared by its actual spectral
definition, never merely by a family name or a nominal node count.
"""

from genus1_source_paths import ROOT as _SOURCE_ROOT, source_path as _source_path
from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import json
import math
import os
import shutil
import tempfile
import time
import uuid

import numpy as np

from genus1_boundary_cubature import Value, combine, fingerprint, MissingBoundaryInput, TimeLimit
from genus1_native_bank import encode_cusp_value as encoded, source_manifest, read_json, chunk_nodes
from genus1_boundary_banks import sha, load_native, EllipticGridBank
from genus1_native_bank import identity as graph_identity
from genus1_projection_preparation import spec as projection_spec
from genus1_projection_momentum import primary_sampling_controls, total_sampling_controls, short_refinement_controls, resolved_primary_sampling
import genus1_projection_preparation as preparation

FORMAT = 'genus1-boundary-spectrum-v1'
KINDS = ('pair', 'comb_pair_face', 'triple_scale_face', 'triple_corner', 'primary')
DEFAULT_BLOCKS = dict(pair=[0,2,2], comb_pair_face=[0,8,4],
                      triple_scale_face=[2,0,2], triple_corner=[0,0,2], primary=[0,0,0])


def settings(config=None):
    """Normalize independent momentum axes; block orders are shared by profiles."""
    config = {} if config is None else json.loads(json.dumps(config))
    if set(config)-{'format','profiles','coefficient_orders','special_dps','chunk_size','collision_elliptic','reference_only','generic_total_node_limit'}:
        raise ValueError('unknown spectrum control')
    if config.get('format',FORMAT)!=FORMAT:
        raise ValueError('unsupported spectrum format')
    reference=config.get('reference_only',False)
    if type(reference) is not bool:raise ValueError('explicit reference-only flag required')
    generic_limit=config.get('generic_total_node_limit',8)
    if type(generic_limit) is not int or generic_limit not in (8,9):
        raise ValueError('production generic total-node limit must be eight or nine')
    defaults={str(label):dict(momentum_orders=[8]*3,tail_orders=[3]*3,
        momentum_sampling=dict(total_nodes=8,head_orders=5,split_scales=1.5,tail_width_scales=.25),
        kernels={'primary':dict(short_refinement=short_refinement_controls({}))} if label==9 else {})
        for label in (8,9)}
    profiles=config.get('profiles',defaults)
    if (len(profiles)<2 or any(not str(k).isdigit() or str(int(k))!=str(k) or int(k)<1 for k in profiles)):
        raise ValueError('at least two positive integer spectral profile labels required')
    result={}
    for label,row in sorted(profiles.items(),key=lambda item:int(item[0])):
        if set(row)-{'momentum_orders','tail_orders','kernels','width_scales','momentum_sampling'}:
            raise ValueError('unknown momentum profile control')
        kernels={}
        if set(row.get('kernels',{}))-set(KINDS):raise ValueError('unknown momentum kernel')
        for kind in KINDS:
            override=row.get('kernels',{}).get(kind,{})
            allowed={'momentum_orders','tail_orders','width_scales','momentum_sampling'}
            if kind=='primary':allowed.update(('primary_sampling','short_refinement'))
            if set(override)-allowed:raise ValueError('unknown kernel control')
            axes=override.get('momentum_orders',row.get('momentum_orders'))
            tails=override.get('tail_orders',row.get('tail_orders'))
            if any(not isinstance(v,(tuple,list)) or len(v)!=3 or
                   any(type(n) is not int or n<1 for n in v) for v in (axes,tails)):
                raise ValueError('three positive core and tail orders required per kernel')
            widths=override.get('width_scales',row.get('width_scales',[1.,1.,1.]))
            if (not isinstance(widths,(tuple,list)) or len(widths)!=3 or
                    any(isinstance(v,bool) or not math.isfinite(v) or v<=0 for v in widths)):
                raise ValueError('three positive momentum width scales required')
            kernels[kind]=dict(momentum_orders=list(axes),tail_orders=list(tails),width_scales=list(widths))
            budget=override.get('momentum_sampling',row.get('momentum_sampling'))
            if budget is not None:
                if override.get('primary_sampling') is not None:raise ValueError('choose one momentum sampling policy')
                budget=total_sampling_controls(budget)
                if not reference and any(n>generic_limit for n in budget['total_nodes']):
                    raise ValueError('generic production exceeds its explicit total-node limit')
                if 'axis_order' in budget and kind!='primary':
                    raise ValueError('cut-relative momentum controls require a primary kernel')
                kernels[kind]['momentum_sampling']=budget
                kernels[kind]['momentum_orders']=list(budget['total_nodes'])
                kernels[kind]['tail_orders']=[n-h for n,h in zip(budget['total_nodes'],budget['head_orders'])]
            elif kind=='primary':
                kernels[kind]['primary_sampling']=primary_sampling_controls(override.get('primary_sampling'))
            if override.get('short_refinement') is not None:
                if budget is None:raise ValueError('regional refinement needs a total-node sampling policy')
                kernels[kind]['short_refinement']=short_refinement_controls(override['short_refinement'])
        result[str(label)]=dict(kernels=kernels)
    effective=[]
    for profile in result.values():
        active={kind:dict(momentum_orders=row['momentum_orders'],
            width_scales=row['width_scales'],
            tail_orders=row['tail_orders'] if kind=='primary' else row['tail_orders'][1:])
            for kind,row in profile['kernels'].items()}
        for kind,row in profile['kernels'].items():
            if 'momentum_sampling' in row:
                active[kind]=dict(momentum_sampling=row['momentum_sampling'],width_scales=row['width_scales'],
                    short_refinement=row.get('short_refinement'))
            elif kind=='primary':
                sampling=row['primary_sampling']
                active[kind].update(short_head_orders=(row['momentum_orders'] if sampling['head_order'] is None
                    else [sampling['head_order']]*3),split_scale=sampling['split_scale'],
                    tail_width_scale=sampling['tail_width_scale'])
        effective.append(fingerprint(active))
    if len(set(effective))!=len(effective):raise ValueError('comparison profiles must request different actual momentum rules')
    blocks=config.get('coefficient_orders',DEFAULT_BLOCKS)
    if set(blocks)!=set(KINDS):raise ValueError('explicit block orders for all kernels required')
    for kind,orders in blocks.items():
        if len(orders)!=3 or any(type(n) is not int or not 0<=n<=10 for n in orders):
            raise ValueError('three native coefficient orders in 0..10 required')
        if kind in ('pair','comb_pair_face') and (orders[0]!=0 or min(orders[1:])<2):
            raise ValueError('pair bridge is projected; both loop-prefix checks require order >=2')
        if kind in ('triple_scale_face','triple_corner') and (orders[1]!=0 or orders[2]<2):
            raise ValueError('height scale is projected; handle-prefix checks require order >=2')
        if kind=='triple_scale_face' and orders[0]<2:raise ValueError('scale-face shape prefixes required')
        if kind=='triple_corner' and orders[0]!=0:raise ValueError('corner shape is projected')
        if kind=='primary' and orders!=[0,0,0]:raise ValueError('primary native source is spectral data only')
    dps=config.get('special_dps',40);chunk=config.get('chunk_size',16)
    if type(dps) is not int or dps<30 or type(chunk) is not int or chunk<1:
        raise ValueError('explicit precision >=30 and positive chunk size required')
    from genus1_collision_elliptic import controls
    normalized=dict(format=FORMAT,profiles=result,coefficient_orders=blocks,special_dps=dps,chunk_size=chunk,reference_only=reference,
                collision_elliptic=controls(config.get('collision_elliptic')))
    if generic_limit!=8:normalized['generic_total_node_limit']=generic_limit
    return normalized


def read(path):
    path=Path(path)
    row=json.loads(path.read_text())
    if set(row)!={'payload','sha256'} or fingerprint(row['payload'])!=row['sha256']:
        raise ValueError('changed immutable spectral record: '+str(path))
    return row['payload']


def publish(path,payload):
    """Atomic create-or-compare: concurrent identical publication is harmless."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    blob=json.dumps(dict(payload=payload,sha256=fingerprint(payload)),sort_keys=True,indent=2,allow_nan=False)+'\n'
    fd,name=tempfile.mkstemp(prefix='.publish-',dir=path.parent)
    try:
        with os.fdopen(fd,'w') as f:f.write(blob);f.flush();os.fsync(f.fileno())
        try:os.link(name,path)
        except FileExistsError:
            if read(path)!=payload:raise ValueError('conflicting immutable publication: '+str(path))
    finally:Path(name).unlink(missing_ok=True)
    return payload


def grid_directory(output,key):
    if not isinstance(key,str) or len(key)!=64 or any(c not in '0123456789abcdef' for c in key):
        raise ValueError('spectral grid SHA256 required')
    return Path(output)/'prepared/spectrum/grids'/key


def batch_identity(rows,primary):
    return fingerprint(encoded([dict(short=s.record(),weight=w) for s,w in rows] if primary
                               else [asdict(sample) for sample in rows]))


class Planner:
    def __init__(self,p,atlas):
        self.p=p;self.atlas=atlas;self.settings=p['spectrum'];self.cache={}
        self.sources=source_manifest();self.source_sha=fingerprint(self.sources)

    def case(self,value,primary):
        if primary:
            # The length-kernel proposal has no angular dependence. The real
            # angles and sector matter factors remain in the evaluation rows.
            from genus1_moduli_sampling import ShortPoint
            short=ShortPoint(value.sector,value.s1,value.s2,0.,0.)
            from genus1_production_support import make_plan
            policy=make_plan(self.p['config']).policy
            kind='primary';g=dict(short=short.record(),policy=dict(q_max=policy.q_max,
                pair_radius=policy.pair_radius,triple_radius=policy.triple_radius))
        else:
            g=dict(value.geometry)
            if value.kind=='pair_projection':
                if g.pop('integrated',None) is not True:raise ValueError('integrated pair request required')
                kind='pair'
            elif value.kind=='comb_collar' and g.pop('term',None)=='pair_face':kind='comb_pair_face'
            elif value.kind=='height_integrated_collar':
                kind=g.pop('term');graph=g.pop('graph')
                if kind not in ('triple_scale_face','triple_corner') or graph not in (0,1):
                    raise ValueError('unknown integrated height request')
                g['pair']=((0,1),(0,2))[graph]
                # Height proposals do not depend on the remaining shape/angle.
                g.pop('shape',None);g.pop('theta',None)
                if kind=='triple_scale_face':g['shape']=.17+.21j
                else:g['theta']=.9
            else:raise ValueError('unsupported analytic boundary request')
        return dict(id='bank',kind=kind,geometry=encoded(g),coefficient_orders=self.settings['coefficient_orders'][kind])

    def grid(self,value,primary,label,*,sampling_context=None):
        case=self.case(value,primary);kind=case['kind']
        controls=self.settings['profiles'][str(label)]['kernels'][kind]
        selection=None
        if sampling_context is not None:
            if not primary or 'momentum_sampling' not in controls:
                raise ValueError('regional proposal context requires a primary total-node policy')
            controls=json.loads(json.dumps(controls));refinement=controls.pop('short_refinement',None)
            budget,axes=resolved_primary_sampling(self.atlas,sampling_context,controls['momentum_sampling'],refinement)
            controls.update(momentum_sampling=budget,momentum_orders=budget['total_nodes'],
                tail_orders=[n-h for n,h in zip(budget['total_nodes'],budget['head_orders'])])
            selection=dict(context=sampling_context.record(),refinement=refinement,refined_axes=axes)
        cache_key=fingerprint(dict(case=case,controls=controls))
        if cache_key in self.cache:return self.cache[cache_key]
        kin,design,rules=projection_spec(self.atlas,case,tuple(controls['momentum_orders']),tuple(controls['tail_orders']),
            controls['width_scales'],primary_sampling=controls.get('primary_sampling'),
            momentum_sampling=controls.get('momentum_sampling'),short_refinement=controls.get('short_refinement'))
        if not self.settings.get('reference_only',False) and any(len(rule['nodes'])>9 for rule in rules):
            raise ValueError('production momentum integration exceeds nine TOTAL nodes on an axis')
        # Grid identity intentionally excludes evaluation geometry. Sharing is
        # allowed only when all actual nodes, weights and coefficient axes agree.
        grid=dict(format=FORMAT,
            identity=graph_identity(kin),rules=rules,coefficient_orders=case['coefficient_orders'],
            special_dps=self.settings['special_dps'],source_sha256=self.source_sha,
            elliptic_order=max(self.p['elliptic_orders']) if primary else None,
            cut_axis=design.get('cut_axis') if primary else None)
        grid['collision_elliptic']=(self.settings.get('collision_elliptic')
                                   if kind in ('pair','triple_scale_face') else None)
        from genus1_production_support import make_plan
        grid['t']=make_plan(self.p['config']).t
        key=fingerprint(grid)
        record=dict(grid=grid,case=case,design=design,controls=controls,source_files=self.sources,
                    chunk_size=self.settings['chunk_size'])
        if selection is not None:record['sampling_selection']=selection
        self.cache[cache_key]=(key,record)
        return key,record

    def register(self,output,value,primary,label):
        key,spec=self.grid(value,primary,label);path=grid_directory(output,key)/'spec.json'
        if path.exists():
            old=read(path)
            if old['grid']!=spec['grid']:raise ValueError('spectral grid hash collision')
        else:
            # Another geometry may have exactly the same grid. Its descriptive
            # example need not be byte-identical to the first example published.
            try:publish(path,spec)
            except ValueError:
                if read(path)['grid']!=spec['grid']:raise
        return key


def ready(output,key):
    folder=grid_directory(output,key);spec=read(folder/'spec.json')
    if fingerprint(spec['grid'])!=key:raise ValueError('grid specification identity changed')
    path=folder/'ready.json'
    if not path.exists():raise MissingBoundaryInput('prepare spectral grid '+key)
    r=read(path)
    if r['grid_sha256']!=key or r['spec_sha256']!=sha(folder/'spec.json'):
        raise ValueError('completed bank belongs to another grid')
    for name,digest in r['files'].items():
        rel=Path(name)
        if rel.is_absolute() or '..' in rel.parts or sha(folder/rel)!=digest:
            raise ValueError('changed completed spectral input: '+name)
    return spec,r


def native_key(grid):
    return fingerprint({k:v for k,v in grid.items() if k not in
                        ('coefficient_orders','elliptic_order','cut_axis','collision_elliptic')})


def index_ready(output,key):
    spec,pin=ready(output,key)
    publish(Path(output)/'prepared/spectrum/reuse'/native_key(spec['grid'])/(key+'.json'),
            dict(grid_sha256=key,ready_sha256=sha(grid_directory(output,key)/'ready.json')))


def donor(output,key,grid):
    """Reuse exact saved DOZZ/nodes and coefficient prefixes, never interpolate."""
    for path in sorted((Path(output)/'prepared/spectrum/reuse'/native_key(grid)).glob('*.json')):
        ref=read(path);other=ref['grid_sha256']
        if other==key:continue
        old,_=ready(output,other);folder=grid_directory(output,other)
        if sha(folder/'ready.json')!=ref['ready_sha256'] or native_key(old['grid'])!=native_key(grid):
            raise ValueError('changed coefficient reuse index')
        plan=read_json(folder/'input/plan.json')
        if all((folder/'input/banks'/f'{i:07d}'/'receipt.json').exists() for i in range(len(chunk_nodes(plan)))):
            return folder/'input'
    return None


def build(output,p,key,*,cpu_budget=600.,max_new_chunks=64,max_new_elliptic_nodes=256):
    """Bounded explicit preparation; partial work cannot produce ready.json."""
    from genus1_production_support import ownership, cpu_deadline
    from genus1_production_support import save, utc
    if not math.isfinite(cpu_budget) or cpu_budget<=0 or min(max_new_chunks,max_new_elliptic_nodes)<1:
        raise ValueError('positive preparation budgets required')
    folder=grid_directory(output,key);spec=read(folder/'spec.json');grid=spec['grid']
    if fingerprint(grid)!=key or fingerprint(spec['source_files'])!=grid['source_sha256']:
        raise ValueError('changed planned spectral grid')
    start=time.process_time();wall=time.monotonic();complete=False;failure=None;new=0
    before_nodes=len(list((folder/'input/elliptic_nodes/bank').glob('*.json')))
    before_collision_groups=len(list((folder/'input/collision/groups').glob('*.json')))
    with ownership(Path(output)/'work/spectrum_claims'/key):
        try:
            if (folder/'ready.json').exists():ready(output,key);complete=True
            else:
                with cpu_deadline(cpu_budget):
                    work=folder/'input'
                    if not work.exists():
                        temporary=Path(tempfile.mkdtemp(prefix='.plan-',dir=folder))
                        try:
                            preparation.prepare(temporary,t=grid['t'],cases=[spec['case']],
                                **spec['controls'],elliptic_order=grid['elliptic_order'] or 4,
                                special_dps=grid['special_dps'],chunk_size=spec['chunk_size'],
                                frozen_cases={'bank':spec})
                            os.rename(temporary,work)
                        finally:
                            if temporary.exists():shutil.rmtree(temporary)
                    plan=read_json(work/'plan.json');row=plan['cases']['bank']
                    if (row['rules']!=grid['rules'] or row['case']!=spec['case'] or
                            fingerprint({k:v for k,v in plan['source_files'].items()
                                         if k!='code/one_loop/three_point/other_scripts/genus1_projection_grid.py'})!=grid['source_sha256']):
                        raise ValueError('preparation differs from requested grid or scientific source')
                    chunks=chunk_nodes(plan)
                    reuse=donor(output,key,grid)
                    missing=[i for i in range(len(chunks)) if not (work/'banks'/f'{i:07d}').exists()]
                    selected=missing[:max_new_chunks]
                    def progress(event):
                        nonlocal new
                        if event['stage']=='committed_chunk':new+=1
                    if selected:
                        # Keep verified source, DOZZ and Upsilon caches alive
                        # across this bounded batch instead of rebuilding them
                        # for every small coefficient chunk.
                        preparation.build(work,start=selected[0],stop=selected[-1]+1,reuse_root=reuse,progress=progress)
                    if len(missing)>max_new_chunks:raise TimeLimit('spectral chunk allocation reached')
                    native=preparation.export(work,'bank')
                    files={str(x.relative_to(folder)):sha(x) for x in
                           (work/'plan.json',work/'plan.json.sha256',native,native.with_suffix('.json'))}
                    if grid['elliptic_order'] is not None:
                        try:preparation.build_elliptic(work,'bank',max_new_nodes=max_new_elliptic_nodes)
                        except MissingBoundaryInput as exc:
                            if 'preparation node limit' not in str(exc):raise
                            raise TimeLimit(str(exc)) from exc
                        for x in (work/'elliptic/bank.npz',work/'elliptic/bank.json',
                                  work/'elliptic/bank.checked.json',work/'elliptic/bank.checked.json.sha256'):
                            files[str(x.relative_to(folder))]=sha(x)
                    if grid.get('collision_elliptic') is not None:
                        from genus1_collision_elliptic import prepare as prepare_collision
                        try:
                            prepare_collision(work/'collision/bank.npz',load_native(native),spec['case']['kind'],
                                grid['collision_elliptic'],max_new_groups=max_new_elliptic_nodes,
                                checkpoint_directory=work/'collision/groups')
                        except MissingBoundaryInput as exc:
                            if 'preparation group limit' not in str(exc):raise
                            raise TimeLimit(str(exc)) from exc
                        for x in (work/'collision/bank.npz',work/'collision/bank.json'):
                            files[str(x.relative_to(folder))]=sha(x)
                    publish(folder/'ready.json',dict(grid_sha256=key,spec_sha256=sha(folder/'spec.json'),files=files))
                    ready(output,key);complete=True
            if complete:index_ready(output,key)
        except Exception as exc:
            complete=False;failure=dict(type=type(exc).__name__,message=str(exc))
        result=dict(plan_sha256=fingerprint(p),grid_sha256=key,complete_rule=complete,
            failure=failure,new_chunks=new,cpu_seconds=time.process_time()-start,
            new_elliptic_nodes=len(list((folder/'input/elliptic_nodes/bank').glob('*.json')))-before_nodes,
            new_collision_groups=len(list((folder/'input/collision/groups').glob('*.json')))-before_collision_groups,
            wall_seconds=time.monotonic()-wall,utc=utc())
        save(Path(output)/'operations/spectral_attempts'/(str(uuid.uuid4())+'.json'),result)
        return result


class Banks:
    """Bounded read-only cache; validate the final pin before loading arrays."""
    def __init__(self,output,limit=8):
        if type(limit) is not int or limit<1:raise ValueError('positive bank cache limit required')
        self.output=Path(output);self.limit=limit;self.cache=OrderedDict()

    def get(self,key):
        if key in self.cache:self.cache.move_to_end(key);return self.cache[key]
        spec,pin=ready(self.output,key);folder=grid_directory(self.output,key)
        bank=load_native(folder/'input/exports/bank.npz');grid=spec['grid']
        if (bank.metadata['identity']!=grid['identity'] or list(bank.orders)!=grid['coefficient_orders'] or
                bank.metadata['quadrature']['rules']!=grid['rules'] or
                bank.metadata['preparation_plan_sha256']!=sha(folder/'input/plan.json')):
            raise ValueError('published native bank does not implement the routed grid')
        ellipse=None
        if grid['elliptic_order'] is not None:
            ellipse=EllipticGridBank.load(folder/'input/elliptic/bank.npz',receipt_sha256=pin['files']['input/elliptic/bank.json'])
            if (ellipse.metadata['cut_axis']!=grid['cut_axis'] or ellipse.metadata['order']!=grid['elliptic_order'] or
                    ellipse.metadata['source_native_data_sha256']!=bank.metadata['data_sha256']):
                raise ValueError('elliptic export does not implement the routed grid')
            for name in ('momenta','log_weights','structure_logs'):
                if not np.array_equal(ellipse.arrays[name],bank.arrays[name]):raise ValueError('elliptic/native grid mismatch')
        elif grid.get('collision_elliptic') is not None:
            from genus1_collision_elliptic import CollisionBank
            ellipse=CollisionBank.load(folder/'input/collision/bank.npz',bank,spec['case']['kind'],grid['collision_elliptic'],
                receipt_sha256=pin['files']['input/collision/bank.json'])
        value=(spec,pin,bank,ellipse)
        self.cache[key]=value
        while len(self.cache)>self.limit:self.cache.popitem(last=False)
        return value


class Runtime:
    """The existing physical evaluators, resolved through exact prepared routes."""
    def __init__(self,output,p,atlas,geometry,point_budget):
        from genus1_channel_evaluation import AbsoluteCollision
        from genus1_channel_evaluation import ArithmeticSafeCFT as Backend
        import genus1_channel_evaluation as backend
        import genus1_collar_mp_arithmetic as mpfix
        self.p=p;self.atlas=atlas;self.geometry=geometry;self.banks=Banks(output)
        self.providers=self;self.current=None;self.engines=OrderedDict();self.height_banks=OrderedDict()
        self.backend=Backend(mpfix,atlas,self.select)
        self.absolute=AbsoluteCollision(self,point_budget);self.point_budget=point_budget

    def select(self,request):
        from genus1_channel_evaluation import Selection
        key,expected=self.current
        if fingerprint(encoded(asdict(request)))!=expected:raise ValueError('request differs from its routed geometry')
        spec,pin,bank,_=self.banks.get(key)
        return Selection(bank,bank.metadata['source_manifest_sha256'],pin['files']['input/exports/bank.json'],bank.orders)

    def collision(self,request):
        from genus1_boundary_integration import legacy_value
        return legacy_value(self.backend.evaluate(request))

    def collide(self,request,key,jacobian):
        self.current=(key,fingerprint(encoded(asdict(request))))
        spec,pin,bank,ellipse=self.banks.get(key)
        if request.kind=='pair_projection' and spec['grid'].get('collision_elliptic') is not None:
            from genus1_channel_evaluation import pair_projection_values
            values,report=pair_projection_values(bank,self.atlas,request.geometry,bank.orders,elliptic=ellipse)
            return self.checked_collision(values,report['prefixes'],ellipse.cuts(),request,jacobian,'collision:')
        if request.kind!='height_integrated_collar':return self.absolute(request,jacobian)
        from genus1_integrated_scale_collar import PreparedBank, SCOPE, evaluate
        g=dict(request.geometry);g.pop('graph');pair=spec['case']['geometry']['pair']
        if key not in self.height_banks:
            arrays={k:v for k,v in bank.arrays.items() if k!='coefficients'}
            arrays['projection_coefficients']=bank.arrays['coefficients'][:,:,0,:]
            meta=dict(evaluation_scope=SCOPE,orders=list(bank.orders),rules=spec['grid']['rules'],
                pair=pair,region=g['region'],external_weights=bank.metadata['identity']['external_weights'])
            self.height_banks[key]=PreparedBank(arrays,meta,dict(receipt_sha256=pin['files']['input/exports/bank.json']))
            while len(self.height_banks)>8:self.height_banks.popitem(last=False)
        height=self.height_banks[key];values,checks=evaluate(height,self.atlas,**g,elliptic=ellipse)
        self.height_banks.move_to_end(key)
        cuts=height.cuts if ellipse is None else ellipse.cuts(bank.orders[2])
        return self.checked_collision(values,checks,cuts,request,jacobian,'native:' if ellipse is None else 'collision:')

    def checked_collision(self,values,checks,cuts,request,jacobian,prefix):
        if len(values)!=len(cuts) or not np.all(np.isfinite(values)):
            raise FloatingPointError('invalid collision prefix layout or values')
        bounds=[c.get('absolute_arithmetic_log_bound') for c in checks]
        factor=jacobian*abs(request.multiplicity)/(4*math.pi)
        eligible={'estimated_arithmetic_loss_limit','cancellation_condition_limit','zero_or_subnormal_term'}
        bounded=all(v is not None and (math.isfinite(v) or v==-math.inf) for v in bounds)
        if not all(c['passed'] for c in checks):
            if (not bounded or any(set(c['reasons'])-eligible for c in checks) or
                    math.log(2*factor)+max(bounds)>math.log(self.point_budget)):
                raise FloatingPointError('collision arithmetic exceeds its weighted allocation')
        comparisons={prefix+label:values[i] for i,(label,_) in enumerate(cuts) if i}
        return Value(values[0],dict(block=max((abs(values[0]-v) for v in comparisons.values()),default=0.),
            arithmetic=math.exp(max(bounds)) if bounded else None,momentum=None,endpoint=None),comparisons)

    def primary(self,shorts,keys,prepared,policy,angle_order):
        from genus1_boundary_batch import BatchedPrimary
        result=[None]*len(shorts)
        for key in dict.fromkeys(keys):
            _,_,_,bank=self.banks.get(key)
            if bank is None:raise ValueError('primary route needs an actual elliptic bank')
            if key not in self.engines:
                self.engines[key]=BatchedPrimary(self.atlas,{bank.metadata['cut_axis']:bank},self.geometry,
                                               orders=self.p['elliptic_orders'])
                while len(self.engines)>8:self.engines.popitem(last=False)
            self.engines.move_to_end(key)
            indices=[i for i,k in enumerate(keys) if k==key]
            values=self.engines[key].evaluate([shorts[i] for i in indices],policy,angle_order,
                                             prepared=[prepared[i] for i in indices])
            for i,value in zip(indices,values):result[i]=value
        return result
