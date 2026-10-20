"""Genus-two separating-region production, including frozen-node validation and order controls."""
from __future__ import annotations
import cmath
import csv
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import sys
import time
import numpy as np
from genus2_integrand_normalization import CURRENT_INTEGRATION_KERNEL_CONVENTION, LEGACY_V5_XI_INTEGRATION_KERNEL_CONVENTION, integration_kernel_scale_to_current
from genus2_separating_tail import tail_scale_from_legacy_xi

from genus2_separating_tail import weighted_residual
import argparse
from scipy.stats import qmc
from genus2_separating_tail import SeparatingProposal, RealUpsilonOne, separating_proposals_from_unit_cube, primary_scaled_kernel_batch
from genus2_siegel_fundamental_domain import in_gottschling_domain
from c1_paths import data_root, module_path
from genus2_separating_tail import RealUpsilonOne, primary_scaled_kernel_batch, integrate_primary_tail

# genus2_separating_tail_cluster
SCHEME='separating-log-length-product-torus-v1'


NUMERICAL_POLICY={
    'integration_kernel_convention':CURRENT_INTEGRATION_KERNEL_CONVENTION,
    'rho0':.04,'cutoffs':[.01,.02,.04],
    'tail_replicates':8,'tail_power':8,'tail_seed':20260930,
    'primary_power':12,'primary_seed':20260901,
    'primary_momentum_order':32,'primary_bridge_order':48,'primary_block_order':2,
    'control_momentum_order':24,'control_bridge_order':32,'control_block_order':2,
    'cft_block_low':8,'cft_block_high':10,'cft_dps':44,'cft_check_dps':52,
    'vacuum_word':8,'vacuum_check_word':10,'vacuum_modes':40,
    'scalar_initial_word':8,'scalar_maximum_word':12,
    'scalar_powered_tolerance':2e-4,'block_tolerance':3e-4,'momentum_tolerance':4e-4,
    'precision_tolerance':1e-8,'vacuum_tolerance':1e-6,
    'point_numerical_target':1e-3,'lattice_tolerance':1e-13,
    'period_absolute_tolerance':1e-9,'period_relative_bridge_tolerance':1e-5,
    'period_maximum_basis':160,'period_max_nfev':80,
    # A documented support guard, NOT a truncation of the sampling domain.
    # Changing it requires a new frozen bundle and new calibration.
    'legacy_bridge_minimum':1e-5,'legacy_handle_log_maximum':1e6,
    'calibration_count':16,'calibration_baseline_seconds':600.,'calibration_memory_mb':3072.,
    'target_relative_standard_error':.005,'primary_relative_order_tolerance':1e-5,
}


def policy_kernel_convention(policy):
    """Frozen pre-v6 policies are Xi; newly prepared policies declare BRY."""
    return policy.get('integration_kernel_convention', LEGACY_V5_XI_INTEGRATION_KERNEL_CONVENTION)


def saved_array_scale_to_current(cache):
    """Untagged historical radius/control caches are explicitly legacy v5."""
    convention = (str(cache['integration_kernel_convention'])
                  if 'integration_kernel_convention' in cache
                  else LEGACY_V5_XI_INTEGRATION_KERNEL_CONVENTION)
    return integration_kernel_scale_to_current(convention)


def saved_rows_scale_to_current(rows):
    conventions = {r.get('integration_kernel_convention', '') for r in rows}
    if len(conventions) != 1:
        raise ValueError('source rows must have one declared kernel convention')
    return integration_kernel_scale_to_current(next(iter(conventions)))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def read_json(path):return json.loads(Path(path).read_text())


def runtime_environment():
    import scipy,mpmath
    return dict(python=sys.version,executable=sys.executable,platform=platform.platform(),
                numpy=np.__version__,scipy=scipy.__version__,mpmath=mpmath.__version__)


def read_csv(path):
    with Path(path).open(newline='') as f:return list(csv.DictReader(f))


def write_csv(path,rows,fields=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fields=fields or list(rows[0])
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def node_omega(node):
    eps=cmath.exp(complex(node['log_rho'],node['angle']))
    return np.array([[complex(node['x1'],node['y1']),eps],
                     [eps,complex(node['x2'],node['y2'])]])


def support_reasons(node,policy):
    reasons=[]
    if node['length'] > -math.log(policy['legacy_bridge_minimum']):
        reasons.append('bridge_below_validated_binary64_Schottky_range')
    if 2*math.pi*max(node['y1'],node['y2']) > policy['legacy_handle_log_maximum']:
        reasons.append('handle_depth_requires_additional_logarithmic_precision_validation')
    return reasons


def load_record(path,identity,require_ok=True):
    r=read_json(path)
    if any(r.get(k)!=v for k,v in identity.items()):raise ValueError(f'stale/mismatched cache: {path}')
    if require_ok and r.get('status')!='ok':raise ValueError(f'unresolved cached result: {path}')
    return r


def period_evaluation(node,policy):
    if node['support_reasons']:
        raise RuntimeError('UNRESOLVED_BACKEND: '+','.join(node['support_reasons']))
    from genus2_plumbing_atlas import LeadingMarking,certify_marking
    omega=node_omega(node)
    # Put the more pinched handle in the dilation generator. This is an exact
    # symplectic permutation, not a modification of the sampling coordinates.
    perm=np.array([[0,1],[1,0]],int);matrix=np.block([[perm,np.zeros((2,2),int)],[np.zeros((2,2),int),perm]])
    marked=perm@omega@perm.T
    logs=(2j*math.pi*marked[0,0],2j*math.pi*marked[1,1],
          complex(-node['length'],node['angle']-math.pi/2))
    q=tuple(cmath.exp(complex(max(z.real,-690.),z.imag)) for z in logs)
    magnitudes=tuple(abs(z) for z in q)
    spread=max(z.real for z in logs)-min(z.real for z in logs)
    marking=LeadingMarking('glasses','tail-handle-permutation',tuple(map(tuple,matrix)),marked,
        q,magnitudes,max(magnitudes),math.exp(spread) if spread<700 else math.inf,logs)
    # Absolute 1e-7 is inadequate when Omega12 is small. Tighten in proportion
    # to its magnitude; fail if the existing backend cannot reach that bar.
    tolerance=min(policy['period_absolute_tolerance'],
                  math.exp(node['log_rho'])*policy['period_relative_bridge_tolerance'])
    chart=certify_marking(omega,marking,word_length=7,max_nfev=policy['period_max_nfev'],
        period_tolerance=tolerance,stability_tolerance=tolerance,
        maximum_collocation_basis=policy['period_maximum_basis'])
    errors=np.array([chart.period_max_residual,chart.period_map_stability])
    if not chart.inverse_success or not np.all(np.isfinite(errors)) or np.max(errors)>tolerance:
        raise RuntimeError('period inverse did not pass tail-scaled residual/step gates: '+chart.inverse_message)
    from genus2_glasses_production_gate import calibrated_momentum_orders,CALIBRATED_Q_MAXIMUM
    final_q=tuple(map(complex,chart.q))
    # The downstream CFT consumes serialized q/log_q, not the optimizer's
    # full-precision internal values. Recheck exactly those saved coordinates.
    from genus2_hybrid_period_map import HybridPeriodMapConfig,hybrid_period_matrix,period_max_residual
    forward=hybrid_period_matrix('glasses',final_q,log_q_values=tuple(map(complex,chart.log_q)),
        config=HybridPeriodMapConfig(tolerance=tolerance,agreement_tolerance=tolerance,
                                    maximum_collocation_basis=policy['period_maximum_basis']))
    roundtrip=period_max_residual(forward.omega,marked)
    if not math.isfinite(roundtrip) or roundtrip>tolerance or forward.error_estimate>tolerance:
        raise RuntimeError('saved-coordinate period round-trip failed the tail-scaled tolerance')
    orders=calibrated_momentum_orders(final_q[2])
    if max(map(abs,final_q))>CALIBRATED_Q_MAXIMUM:raise RuntimeError('q exceeds prior calibrated envelope')
    return dict(q=list(chart.q),log_q=list(chart.log_q),matrix=matrix.tolist(),
        momentum_orders=list(orders),period_tolerance=tolerance,
        residual=chart.period_max_residual,map_step=chart.period_map_stability,
        saved_coordinate_residual=roundtrip,saved_coordinate_map_step=forward.error_estimate,
        algorithm=chart.period_algorithm,marking=chart.word)


def cft_evaluation(node,period,policy,radii,calibration=False,*,check_orders=True,use_baseline=False):
    if calibration and not check_orders:raise ValueError('calibration requires independent order checks')
    from genus2_liouville_evaluation import evaluate_liouville_rescaled
    from genus2_scalar_evaluation import evaluate_glasses_noncompact_scalar_fixed_cap_convergence
    from free_boson_plumbing import igusa_chi10_evaluation_genus2
    from genus2_c1_string_integrand import compact_boson_winding_evaluation_genus2
    q=tuple(map(complex,period['q']));logs=tuple(map(complex,period['log_q']))
    omega=node_omega(node);marked=omega[::-1,::-1].copy()
    start=time.monotonic()
    scalar=evaluate_glasses_noncompact_scalar_fixed_cap_convergence(q,marked,log_q_values=logs,
        initial_word_length=policy['scalar_initial_word'],maximum_word_length=policy['scalar_maximum_word'],
        word_length_increment=2,max_mode=policy['vacuum_modes'],tolerance=1e-14,
        powered_relative_tolerance=policy['scalar_powered_tolerance'],confirmation_steps=2)
    if not scalar.converged or scalar.terminal_powered_relative_change is None:
        raise RuntimeError('powered scalar convergence gate failed')
    scalar_value=scalar.evaluation.partition
    orders=tuple(period['momentum_orders']);high_orders=tuple(n+2 for n in orders)
    calls={};timings={}
    def evaluate(label,block,quadrature,dps=None,word=None):
        tick=time.monotonic()
        value=evaluate_liouville_rescaled('glasses',q,log_q_values=logs,block_order=block,
            quadrature_order=quadrature,quadrature_scheme='primary-gaussian',
            dps=dps or policy['cft_dps'],vacuum_word_length=word or policy['vacuum_word'],
            vacuum_oscillator_level=policy['vacuum_modes'],resum_global_block=True,
            global_block_tolerance=1e-13)
        if not math.isfinite(value.log_partition) or not math.isfinite(value.scaled_partition) or value.scaled_partition<=0:
            raise RuntimeError('nonfinite/nonpositive Liouville output')
        calls[label]=value;timings[label]=time.monotonic()-tick
        return value
    base=evaluate('baseline',policy['cft_block_low'],orders)
    high_n=base
    checks=dict(scalar_powered_step=scalar.terminal_powered_relative_change)
    if check_orders:
        high_b=evaluate('block_check',policy['cft_block_high'],orders)
        high_n=evaluate('momentum_check',policy['cft_block_high'],high_orders)
        block_delta=abs(math.expm1(high_b.log_partition-base.log_partition))
        momentum_delta=abs(math.expm1(high_n.log_partition-high_b.log_partition))
        checks.update(block_relative_step=block_delta,momentum_relative_step=momentum_delta)
        if block_delta>policy['block_tolerance'] or momentum_delta>policy['momentum_tolerance']:
            raise RuntimeError('independent CFT-order gates failed: '+json.dumps(checks))
    if calibration:
        precise=evaluate('precision_check',policy['cft_block_high'],high_orders,dps=policy['cft_check_dps'])
        vacuum=evaluate('vacuum_check',policy['cft_block_high'],high_orders,word=policy['vacuum_check_word'])
        checks['precision_relative_step']=abs(math.expm1(precise.log_partition-high_n.log_partition))
        checks['vacuum_relative_step']=abs(math.expm1(vacuum.log_partition-high_n.log_partition))
        if checks['precision_relative_step']>policy['precision_tolerance'] or checks['vacuum_relative_step']>policy['vacuum_tolerance']:
            raise RuntimeError('precision/vacuum gate failed: '+json.dumps(checks))
    chi=igusa_chi10_evaluation_genus2(omega,tol=1e-15)
    chi_check=igusa_chi10_evaluation_genus2(omega,nmax=chi.nmax+2,tol=1e-15)
    if abs(chi.log_abs-chi_check.log_abs)>1e-7:raise RuntimeError('theta-product cutoff check failed')
    chosen=base if use_baseline or not check_orders else high_n
    kernel_convention=policy_kernel_convention(policy)
    common=[math.log(tail_scale_from_legacy_xi(kernel_convention)*64*math.pi**4)+24*math.log(2),
            -13*math.log(np.linalg.det(omega.imag)),-2*chi_check.log_abs,
            chosen.log_partition,-25*math.log(scalar_value),2*node['log_rho']]
    rounding_proxy=8*np.finfo(float).eps*sum(map(abs,common))
    if rounding_proxy>1e-5:raise RuntimeError('large-log cancellation requires a rescaled geometry backend')
    scaled=[];lattice_records=[]
    for radius in radii:
        theta=compact_boson_winding_evaluation_genus2(omega,float(radius),tolerance=policy['lattice_tolerance'])
        scaled.append(math.exp(math.fsum(common+[math.log(2*math.pi*radius*theta.value)])))
        lattice_records.append(float(theta.value))
    if np.any(~np.isfinite(scaled)) or np.any(np.asarray(scaled)<=0):raise RuntimeError('invalid scaled full kernel')
    checks['rounding_proxy']=rounding_proxy
    checks['theta_log_absolute_step']=abs(chi.log_abs-chi_check.log_abs)
    return dict(scaled_kernel=scaled,checks=checks,stage_seconds=timings,
        integration_kernel_convention=kernel_convention,
        total_cft_seconds=time.monotonic()-start,scalar=scalar_value,
        scalar_effective_word=scalar.evaluation.effective_word_length,
        liouville_log=chosen.log_partition,liouville_scaled=chosen.scaled_partition,
        momentum_orders=list(orders if chosen is base else high_orders),
        block_order=policy['cft_block_low'] if chosen is base else policy['cft_block_high'],
        independent_orders_checked=check_orders,estimator_uses_baseline=chosen is base,
        compact_winding=lattice_records,full_cft_layer='evaluate_liouville_rescaled + powered glass scalar',
        calibration=calibration)


# genus2_separating_residual_pilot
PILOT_POLICY=dict(format_version=2,control_cutoff=.08,
    integration_kernel_convention=CURRENT_INTEGRATION_KERNEL_CONVENTION,
    band_edges=[.04,.02,.01,.005,.002,.001],replicates=4,power=3,seed=20261030,
    control_momentum_order=24,control_bridge_order=32,control_block_order=2,
    total_accuracy_target=None,
    estimator='finite-band 1_D*(K-G); independently cached integral_D G',
    numerical_policy=dict(NUMERICAL_POLICY,target_relative_standard_error=None))


def band_proposal(unit,rho_low,rho_high):
    """Normalized finite-log-band proposal, with uncapped cusp-adapted tori."""
    if not 0<rho_low<rho_high<1/(2*math.pi):raise ValueError('invalid finite radial band')
    unit=np.asarray(unit,float)
    if unit.ndim!=2 or unit.shape[1]!=6 or not len(unit) or np.any(~np.isfinite(unit)) or np.any(unit<0) or np.any(unit>=1):
        raise ValueError('unit coordinates must have shape (N,6) and lie in [0,1)')
    # Reuse the tested torus ordering, angle and Jacobian. Setting the radial
    # coordinate to zero gives inverse p_L=2*L0; replace that by Delta L.
    origin=unit.copy();origin[:,5]=0.
    base=separating_proposals_from_unit_cube(origin,rho_high)
    l0=-math.log(2*math.pi*rho_high);width=math.log(rho_high/rho_low)
    length=l0+width*unit[:,5]
    return SeparatingProposal(base.tau1,base.tau2,length,-length-math.log(2*math.pi),
        base.angle,base.log_weight_over_rho_squared+math.log(width/(2*l0)))


def make_nodes(policy):
    edges=policy['band_edges']
    if len(edges)<2 or any(not 0<b<a for a,b in zip(edges,edges[1:])) or edges[0]>policy['control_cutoff']:
        raise ValueError('band edges must decrease within the control tube')
    if policy['replicates']<2 or policy['power']<1:raise ValueError('need >=2 replicates and positive power')
    nodes=[]
    for band,(high,low) in enumerate(zip(edges,edges[1:])):
        for rep in range(policy['replicates']):
            seed=policy['seed']+10000*band+rep
            unit=qmc.Sobol(6,scramble=True,seed=seed).random_base2(policy['power'])
            proposal=band_proposal(unit,low,high)
            inside=in_gottschling_domain(proposal.domain_matrices(),tolerance=0.)
            for index in range(len(unit)):
                n=dict(node_id=f'res-b{band:02d}-r{rep:02d}-p{index:08d}',band=band,replicate=rep,index=index,
                    nominal_count=len(unit),seed=seed,unit=unit[index].tolist(),rho_low=low,rho_high=high,
                    x1=float(proposal.tau1[index].real),y1=float(proposal.tau1[index].imag),
                    x2=float(proposal.tau2[index].real),y2=float(proposal.tau2[index].imag),
                    length=float(proposal.length[index]),log_rho=float(proposal.log_rho[index]),
                    angle=float(proposal.angle[index]),in_domain=bool(inside[index]),
                    log_weight_over_rho_squared=float(proposal.log_weight_over_rho_squared[index]))
                n['support_reasons']=support_reasons(n,policy['numerical_policy'])
                n['node_sha256']=object_sha(n);nodes.append(n)
    return nodes


def stronger_check_indices(nodes):
    """One large-handle-nome point per band, plus a combined-cusp point.

    Selection uses geometry only, before any residual has been evaluated.
    Unsupported points stay in the MC design but cannot serve as calibrators.
    """
    candidates=[i for i,n in enumerate(nodes) if n['in_domain'] and not n['support_reasons']]
    chosen=[]
    for band in sorted({n['band'] for n in nodes}):
        group=[i for i in candidates if nodes[i]['band']==band]
        if group:chosen.append(min(group,key=lambda i:nodes[i]['y1']+nodes[i]['y2']))
    if candidates:
        cusp=max(candidates,key=lambda i:nodes[i]['y1'])
        if cusp not in chosen:chosen.append(cusp)
    return sorted(chosen)


def load(run):
    from c1_frozen_bundle import verify_bundle
    run=Path(run).resolve();manifest=verify_bundle(run)
    policy=read_json(run/'policy.json');nodes=read_json(run/'design/nodes.json')
    if object_sha(policy)!=read_json(run/'campaign.json')['policy_sha256']:
        raise ValueError('policy identity mismatch')
    # Frozen coordinates are authoritative across NumPy/SciPy versions.
    # Verify their identities, not bitwise recomputation of floating geometry
    # or re-scrambling with the cluster's possibly different RNG implementation.
    count=2**policy['power'];reps=policy['replicates'];bands=len(policy['band_edges'])-1
    if len(nodes)!=bands*reps*count:raise ValueError('incomplete frozen design')
    for i,node in enumerate(nodes):
        b,remainder=divmod(i,reps*count);r,j=divmod(remainder,count)
        if (node['band'],node['replicate'],node['index'],node['nominal_count'],node['node_id'])!=(
                b,r,j,count,f'res-b{b:02d}-r{r:02d}-p{j:08d}'):
            raise ValueError('invalid frozen node identity/count')
        if object_sha({k:v for k,v in node.items() if k!='node_sha256'})!=node['node_sha256']:
            raise ValueError('frozen node fingerprint mismatch')
    return run,manifest,policy,nodes


def evaluate(run,manifest,policy,nodes,index):
    if not 0<=index<len(nodes):raise ValueError('node index outside manifest')
    node=nodes[index];path=run/'results/nodes'/f"{node['node_id']}.json"
    identity=dict(bundle_sha256=manifest['bundle_sha256'],node_id=node['node_id'],node_sha256=node['node_sha256'])
    if path.exists():
        old=load_record(path,identity,require_ok=False)
        if old['status']=='ok':return old
    start=time.monotonic();record=dict(identity,status='failed',environment=runtime_environment())
    radii=np.load(run/'inputs/reuse_arrays.npz')['radii']
    try:
        if not node['in_domain']:
            record.update(status='ok',domain_rejected=True,weighted_residual=[0.]*len(radii),
                          weighted_numerical_step_proxy=[0.]*len(radii))
        else:
            # Reuse the old full-CFT layer and its checks, without the old
            # infinite-tail cohort gate or an imposed total-accuracy target.
            period=period_evaluation(node,policy['numerical_policy'])
            checked=not policy.get('light',False) or node['node_id'] in policy['strong_check_node_ids']
            cft=cft_evaluation(node,period,policy['numerical_policy'],radii,False,
                check_orders=checked,use_baseline=policy.get('light',False))
            u=RealUpsilonOne();kwargs=dict(upsilon=u,block_order=2,
                kernel_convention=policy_kernel_convention(policy['numerical_policy']))
            control=primary_scaled_kernel_batch([complex(node['x1'],node['y1'])],[complex(node['x2'],node['y2'])],
                [node['length']],radii,momentum_order=24,bridge_order=32,**kwargs)
            full=np.array([cft['scaled_kernel']]);weighted=weighted_residual(full,control,[node])[0]
            # This is a sensitivity diagnostic from order steps, NOT a bound.
            proxy=None
            if checked:
                control_high=primary_scaled_kernel_batch([complex(node['x1'],node['y1'])],[complex(node['x2'],node['y2'])],
                    [node['length']],radii,momentum_order=32,bridge_order=48,**kwargs)
                step=sum(cft['checks'][k] for k in ('block_relative_step','momentum_relative_step','scalar_powered_step'))
                proxy=(full[0]*step+np.abs(control_high[0]-control[0]))*math.exp(node['log_weight_over_rho_squared'])
                if np.any(~np.isfinite(proxy)):raise ValueError('nonfinite numerical-step proxy')
            record.update(status='ok',period=period,cft=cft,control_scaled=control[0].tolist(),
                relative_residual=(full[0]/control[0]-1).tolist(),weighted_residual=weighted.tolist(),
                weighted_numerical_step_proxy=None if proxy is None else proxy.tolist(),
                residual_smaller_than_numerical_step=None if proxy is None else (np.abs(weighted)<=proxy).tolist(),
                stronger_checks_performed=checked,
                proxy_warning='unchecked is unknown, not zero; order steps on sentinels are not a bound on other nodes')
    except Exception as e:
        record.update(status='failed',error=f'{type(e).__name__}: {e}')
        record.pop('weighted_residual',None)
    record['elapsed_seconds']=time.monotonic()-start;atomic_json(path,record)
    print(__import__('json').dumps({k:record[k] for k in ('node_id','status','elapsed_seconds','error') if k in record}),flush=True)
    return record


def pilot_main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--index',type=int)
    args=parser.parse_args()
    run,manifest,policy,nodes=load(args.run)
    if args.index is None: print(manifest['bundle_sha256']);return
    if evaluate(run,manifest,policy,nodes,args.index)['status']!='ok':raise SystemExit(1)


# genus2_tail_completion_production
def evaluate_node(args):
    root=data_root('two_loop');tail=root/'tail_completion';parent=root/'production/parent'
    if args.family=='deep':
        nodes=read_json(tail/'deep_band_design.json')
        policy=read_json(tail/'deep_band_policy.json')
        if not 0<=args.index<len(nodes):raise ValueError('deep index must be 0..15')
        original=nodes[args.index]
        numeric=dict(policy['numerical_policy'])
    else:
        nodes=read_json(parent/'design/nodes.json')
        if not 16<=args.index<=23:raise ValueError('recovered-band index must be 16..23')
        original=nodes[args.index]
        policy=read_json(parent/'policy.json')
        # The new campaign explicitly validates the deep handle rather than
        # modifying the original campaign or silently removing its guard.
        numeric=dict(policy['numerical_policy'],legacy_handle_log_maximum=1e9,
                     cft_block_low=10,cft_block_high=12,cft_dps=60,cft_check_dps=72)
    node=dict(original)
    node['support_reasons']=support_reasons(node,numeric)
    radii=np.load(parent/'inputs/old_radius_cache.npz')['radii']
    output=args.output.resolve()
    if output.exists():raise FileExistsError('refusing to overwrite '+str(output))
    record=dict(status='started',node=node,original_node=original,radii=radii.tolist(),
        numerical_policy=numeric,source_policy_sha256=object_sha(policy),
        family=args.family,code_sha256={name:sha(module_path(name)) for name in
            ['genus2_tail_completion_production','genus2_separating_tail_cluster',
             'genus2_plumbing_atlas','genus2_hybrid_period_map','genus2_separating_tail']})
    start=time.monotonic()
    try:
        if not node['in_domain']:
            record.update(status='ok',domain_rejected=True,weighted_residual=[0.]*len(radii))
        else:
            period=period_evaluation(node,numeric)
            period['momentum_orders']=([max(12,n) for n in period['momentum_orders']]
                if args.family=='deep' else [14]*3)
            record['period']=period;atomic_json(output,record)
            full=cft_evaluation(node,period,numeric,radii,args.calibration,
                check_orders=args.family=='deep' or args.calibration)
            control=primary_scaled_kernel_batch([complex(node['x1'],node['y1'])],
                [complex(node['x2'],node['y2'])],[node['length']],radii,
                upsilon=RealUpsilonOne(),momentum_order=24,bridge_order=32,block_order=2,
                kernel_convention=policy_kernel_convention(numeric))[0]
            record.update(status='ok',cft=full,control_scaled=control.tolist(),
                weighted_residual=weighted_residual(np.array(full['scaled_kernel'])[None,:],control[None,:],[node])[0].tolist())
    except Exception as exc:
        record.update(status='failed',error=f'{type(exc).__name__}: {exc}')
    record['seconds']=time.monotonic()-start
    atomic_json(output,record)
    print(json.dumps({k:record[k] for k in ['status','seconds']},indent=2))
    if record['status']!='ok':raise RuntimeError(record['error'])


def evaluate_controls(args):
    root=data_root('two_loop');settings=read_json(root/'tail_completion/control_order_settings.json')
    radii=np.load(root/'production/parent/inputs/old_radius_cache.npz')['radii']
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    upsilon=RealUpsilonOne();arrays={'radii':radii}
    atomic_json(output/'settings.json',settings)
    for rho in [.001,.08]:
        for block,momentum,bridge in settings['settings']:
            label=f'rho{rho:g}_b{block}_n{momentum}_m{bridge}'
            result=integrate_primary_tail(radii,rho,upsilon=upsilon,
                power=settings['power'],replicates=settings['replicates'],seed=settings['seed'],
                block_order=block,momentum_order=momentum,bridge_order=bridge)
            arrays[label]=result.replicates
            np.savez_compressed(output/'control_order_arrays.npz',**arrays)
            print(label,flush=True)


def tail_main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    n=sub.add_parser('node',help='Evaluate one nominal frozen proposal using the current backend')
    n.add_argument('--family',choices=['deep','recovered'],required=True)
    n.add_argument('--index',type=int,required=True)
    n.add_argument('--output',type=Path,required=True)
    n.add_argument('--calibration',action='store_true',help='Additional block/momentum/precision/vacuum checks; may raise the selected order')
    n.set_defaults(run=evaluate_node)
    controls=sub.add_parser('controls',help='Repeat the eight paired control-order integrations')
    controls.add_argument('--output',type=Path,required=True);controls.set_defaults(run=evaluate_controls)
    args=p.parse_args();args.run(args)


def main():
    """Run current tail completion, or the explicitly selected original pilot."""
    if len(sys.argv) > 1 and sys.argv[1] == 'pilot':
        sys.argv.pop(1)
        return pilot_main()
    return tail_main()

if __name__ == '__main__':
    main()
