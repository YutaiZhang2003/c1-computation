"""Collision-first deterministic boundary driver, opt-in and target blind.

The complete partition is twelve historical collision terms at the requested
radii plus a primary and a matched remainder in each of six labeled tube
sectors. There is no endpoint addback from the old fifteen-term estimator.
Banks are supplied explicitly; this driver never prepares them implicitly.
"""
from __future__ import annotations

from genus1_source_paths import ROOT as _SOURCE_ROOT, source_path as _source_path

from dataclasses import dataclass
import math
from pathlib import Path
import numpy as np
from scipy.stats import t as student

import genus1_analytic_boundary_layer as boundary
from genus1_boundary_cubature import (Cell,Controls,MissingBoundaryInput,PointCache,
                                      Value,combine,fingerprint,integrate)
from genus1_moduli_sampling import CuspPoint, ShortPoint, SECTORS, TAU, regular_mask
from genus1_collision_sampling import EVALUATORS, MissingCFTBank

FORMAT='genus1-complete-boundary-integration-v1'
ANALYTIC_FORMAT='genus1-analytic-boundary-integration-v1'
ERRORS=('block','momentum','arithmetic','endpoint')


def source_identity():
    """Hash the retained source inputs, including lazily loaded CFT helpers."""
    import hashlib
    from genus1_source_paths import ROOT, production_sources
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in production_sources()}


def amplitude_value(reduced, jacobian=1.):
    """Convert the physical reduced density to amplitude units exactly once."""
    return Value(-reduced.value.imag*jacobian/(4*math.pi),
        {k:None if e is None else e*abs(jacobian)/(4*math.pi) for k,e in reduced.errors.items()},
        {k:None if v is None else -v.imag*jacobian/(4*math.pi) for k,v in reduced.comparisons.items()})


def legacy_value(result):
    """Adapt a verified BankBackedCFT result without manufacturing audits."""
    if not isinstance(result,dict): raise MissingBoundaryInput('missing CFT result')
    safe=result.get('safety',{})
    if not all(safe.get(k) is True for k in ('chart_valid','coefficients_verified','arithmetic_safe','provenance_verified')):
        raise ValueError('unchecked CFT result cannot enter boundary integration')
    pairs=result.get('paired_checks',{})
    errors={k:(max(abs(complex(a)-complex(b)) for a,b in pairs[k].values())
               if pairs.get(k) else None) for k in ('block','momentum')}
    log_error=result.get('absolute_arithmetic_log_bound')
    errors['arithmetic']=None if log_error is None else math.exp(log_error)
    errors['endpoint']=None
    comparisons={}
    for name,(low,high) in pairs.get('block',{}).items():
        if complex(high)!=complex(result['value']):raise ValueError('block pair has another reference value')
        comparisons['native:'+name]=complex(low)
    return Value(result['value'],errors,comparisons)


def short_map(unit, sector, order, band, policy):
    """All positive short lengths, ordered into two disjoint triangles.

    m=max(s1,s2), r=min(s1,s2)/m. Split m at the tube floor. The second
    band reaches infinity, including simultaneous long tubes. The determinant
    ds1 ds2 = m dm dr is retained along with both angular factors.
    """
    u=np.asarray(unit,float)
    if u.shape!=(4,) or np.any(~np.isfinite(u)) or np.any((u<=0)|(u>=1)):
        raise ValueError('four open unit coordinates required')
    if order not in (0,1) or band not in (0,1) or sector not in SECTORS:
        raise ValueError('explicit sector, length ordering and band required')
    scale=policy.length
    if band==0: m=scale*u[0];dm=scale
    else: m=scale/(1-u[0]);dm=scale/(1-u[0])**2
    lengths=(m,m*u[1]) if order==0 else (m*u[1],m)
    short=ShortPoint(sector,*lengths,TAU*u[2],TAU*u[3])
    return short,TAU**2*m*dm


def remainder_map(unit, sector, order, band, policy):
    u=np.asarray(unit,float)
    if u.shape!=(6,) or np.any(~np.isfinite(u)) or np.any((u<=0)|(u>=1)):
        raise ValueError('six open unit coordinates required')
    short,jac=short_map(u[:4],sector,order,band,policy)
    theta=TAU*(u[5]-.5)
    lower=max(short.s1,short.s2,policy.length,
              TAU*math.sqrt(1-(theta/TAU)**2)-short.total)
    length=lower+policy.length*u[4]/(1-u[4])
    # dtheta=2*pi du; original six-sector measure contributes 1/(2*pi)^2.
    jac*=policy.length/(1-u[4])**2/TAU
    return CuspPoint(short,length,theta),jac


def collision_axes(stratum,region):
    if stratum not in boundary.COLLISION_STRATA or region not in ('bulk','tail'):
        raise ValueError('unknown collision term')
    if stratum=='triple_scale_face': return ((0 if region=='bulk' else 1),2,3)
    if stratum=='triple_corner': return ((0 if region=='bulk' else 1),2)
    return tuple(range(EVALUATORS[stratum][0]))


def collision_unit(unit,stratum,region):
    axes=collision_axes(stratum,region);u=np.asarray(unit,float)
    if u.shape!=(len(axes),) or np.any(~np.isfinite(u)) or np.any((u<0)|(u>=1)):
        raise ValueError('wrong reduced collision coordinates')
    original=np.full(EVALUATORS[stratum][0],.5)
    original[list(axes)]=u
    return original


def collision_boxes(stratum,region):
    axes=collision_axes(stratum,region);dim=len(axes)
    shape_axis={'triple_corner':2,'triple_scale_face':2,
                'triple_forest_remainder':2,'triple_pair_face':4}.get(stratum)
    intervals=((0.,1/6),(1/6,5/6),(5/6,1.)) if shape_axis is not None else ((0.,1.),)
    result=[]
    for interval in intervals:
        b=[(0.,1.)]*dim
        if shape_axis is not None:b[axes.index(shape_axis)]=interval
        result.append(tuple(b))
    return tuple(result)


@dataclass(frozen=True)
class RunPlan:
    t: float = .25
    policy: boundary.BoundaryPolicy = boundary.BoundaryPolicy()
    collision_height_split: float = 8.
    angle_order: int = 16
    compact_points: int = 64
    mode: str = 'reference'

    def __post_init__(self):
        if self.mode not in ('reference','analytic'):
            raise ValueError('explicit reference or analytic estimator mode required')
        if not 0<self.t<1 or not math.isfinite(self.collision_height_split) or self.collision_height_split<=1:
            raise ValueError('supported imaginary energy and modular split required')
        if type(self.angle_order) is not int or self.angle_order<2 or self.compact_points!=64:
            raise ValueError('explicit angle order; retained compact candidate is 64 points')

    def record(self):
        return dict(format=ANALYTIC_FORMAT if self.mode=='analytic' else FORMAT,
            mode=self.mode,t=self.t,policy=self.policy.record(),
            collision_height_split=self.collision_height_split,angle_order=self.angle_order,
            compact_points=self.compact_points,normalization='-Im(reduced)/(4*pi)',
            cutoff_at_infinity=False,original_endpoint_addback_enabled=False)


def term_names(mode='reference'):
    if mode not in ('reference','analytic'):raise ValueError('unknown estimator mode')
    # Corner and scale face can run first, while the full collision forest stays mandatory.
    collision=('triple_corner','triple_scale_face','pair_face_analytic','triple_pair_face',
               'pair_face_remainder','triple_forest_remainder')
    names=tuple(f'{region}:{term}' for term in collision for region in ('bulk','tail'))+tuple(
        f'{kind}:'+''.join(map(str,s)) for kind in ('primary','remainder') for s in SECTORS)
    if mode=='analytic':
        names=tuple(n for n in names if not n.startswith('remainder:') and
                    n.split(':')[1] not in ('pair_face_remainder','triple_forest_remainder'))
    return names


def term_cells(name,plan,atlas,*,primary=None,collision=None,remainder=None):
    kind,label=name.split(':')
    if name not in term_names(plan.mode):raise ValueError('component excluded by estimator mode')
    if kind in ('bulk','tail'):
        def evaluate(u):
            sample=boundary.collision_plan(atlas,label,collision_unit(u,label,kind),region=kind,
                policy=plan.policy,cutoff=plan.collision_height_split)
            if not sample.requests:
                if sample.exact_zero_reason is None: raise MissingBoundaryInput('unspecified collision zero')
                return Value(0.)
            if collision is None:raise MissingBoundaryInput('collision bank resolver is absent')
            rows=[]
            for request in sample.requests:
                try: value=collision(request)
                except MissingCFTBank as exc:raise MissingBoundaryInput(str(exc)) from exc
                rows.append((value if isinstance(value,Value) else legacy_value(value)).scaled(request.multiplicity))
            return amplitude_value(combine(rows),sample.jacobian)
        return [Cell(name,box,evaluate) for box in collision_boxes(label,kind)]
    sector=tuple(map(int,label));result=[]
    for order in (0,1):
        for band in (0,1):
            def evaluate(u,order=order,band=band):
                if kind=='primary':
                    short,jac=short_map(u,sector,order,band,plan.policy)
                else:
                    point,jac=remainder_map(u,sector,order,band,plan.policy);short=point.short
                # Collision ownership is decided before requesting any coefficients.
                if not regular_mask(short,plan.policy.pair_radius,boundary.MIN_HEIGHT):return Value(0.)
                if kind=='primary':
                    if primary is None:raise MissingBoundaryInput('elliptic primary resolver is absent')
                    value=primary(short,plan.policy,plan.angle_order)
                else:
                    if remainder is None:raise MissingBoundaryInput('matched tube remainder resolver is absent')
                    value=remainder(point)
                if not isinstance(value,Value):raise TypeError('physical callback must return reduced Value')
                return amplitude_value(value,jac)
            result.append(Cell(f'{name}:order{order}:band{band}',((0.,1.),)*(4 if kind=='primary' else 6),evaluate))
    return result


def domain_manifest(plan=RunPlan()):
    rows=[]
    for name in term_names(plan.mode):
        kind,label=name.split(':')
        if kind in ('bulk','tail'):
            boxes=collision_boxes(label,kind)
            rows.append(dict(name=name,dimension=len(collision_axes(label,kind)),root_cells=len(boxes),
                             reduced_axes=list(collision_axes(label,kind))))
        else:
            rows.append(dict(name=name,dimension=4 if kind=='primary' else 6,root_cells=4,
                             sector=list(map(int,label)),short_lengths_cover_infinity=True))
    return dict(plan=plan.record(),components=rows,required_components=list(term_names(plan.mode)),
        primary_root_cells=24,remainder_root_cells=0 if plan.mode=='analytic' else 24,
        all_six_sectors=True,collision_priority=True,compact_points=64,
        omitted_components=[n for n in term_names() if n not in term_names(plan.mode)],
        remainders_diagnostic_only=plan.mode=='analytic',production_enabled=False)


def run_boundary(plan,atlas,*,primary=None,collision=None,remainder=None,bank_identity,
                 controls,cache_directory=None,selected_terms=None):
    """Run independent components, retaining explicit incomplete/error status.

    controls is a Controls instance or an explicit per-component dictionary.
    selected_terms permits bounded development runs, never a partial amplitude.
    """
    if not isinstance(bank_identity,dict) or not bank_identity:
        raise ValueError('immutable bank, spectral, chart and remainder identities required')
    if not math.isclose(complex(atlas.necklace.omega_in).imag,plan.t,abs_tol=1e-14) or complex(atlas.necklace.omega_in).real!=0:
        raise ValueError('atlas and run energy disagree')
    if plan.mode=='analytic' and remainder is not None:
        raise ValueError('analytic production cannot receive a remainder callback')
    required_names=term_names(plan.mode)
    selected=list(required_names) if selected_terms is None else list(selected_terms)
    if not selected or len(selected)!=len(set(selected)) or set(selected)-set(required_names):
        raise ValueError('explicit distinct supported terms required')
    results={};sources=source_identity()
    for name in selected:
        control=controls[name] if isinstance(controls,dict) else controls
        identity=dict(plan=plan.record(),banks=bank_identity,sources=sources,term=name)
        path=None if cache_directory is None else Path(cache_directory)/fingerprint(identity)/(name.replace(':','_')+'.sqlite3')
        with PointCache(identity,path) as cache:
            results[name]=integrate(term_cells(name,plan,atlas,primary=primary,collision=collision,
                remainder=remainder),control,cache)
    return summarize_boundary(plan,results,bank_identity=bank_identity,sources=sources)


def summarize_boundary(plan,results,*,bank_identity,sources):
    """Assemble persisted components only within the same immutable run."""
    required_names=term_names(plan.mode)
    if set(results)-set(required_names):raise ValueError('component outside selected estimator')
    missing=[name for name in required_names if name not in results or results[name]['value'] is None]
    values=[Value(complex(*r['value']),r['errors']) for r in results.values() if r['value'] is not None]
    total=combine(values) if not missing else None
    numerical=not missing and all(r['status']=='quadrature_indicator_met' for r in results.values())
    unassessed=[]
    for name,r in results.items():
        if r['errors'] is not None:
            required=set((*ERRORS,'regulator','chart') if name.startswith('primary:') else ERRORS)
            unassessed.extend(f'{name}:{e}' for e in sorted(required|set(r['errors'])) if r['errors'].get(e) is None)
    return dict(format=plan.record()['format'],plan=plan.record(),bank_identity=bank_identity,source_identity=sources,components=results,
        required_components=list(required_names),
        omitted_components=[n for n in term_names() if n not in required_names],
        analytic_approximation=plan.mode=='analytic',
        omitted_approximation_error=None if plan.mode=='analytic' else 0.,
        boundary_amplitude=None if total is None else total.value.real,
        errors=None if total is None else total.errors,
        quadrature_indicator=None if missing else math.fsum(r['quadrature_indicator'] for r in results.values()),
        complete_domain=not missing,quadrature_targets_met=numerical,
        missing_components=missing,unassessed_errors=unassessed,
        full_amplitude_certified=False,compact_included=False,
        new_evaluations=sum(r['new_evaluations'] for r in results.values()),
        cache_hits=sum(r['cache_hits'] for r in results.values()),new_mc_points=0)


def compact_estimate(observations,plan,*,density_identity):
    a=np.asarray(observations,float)
    if a.ndim!=2 or a.size!=plan.compact_points or a.shape[0]<4 or not np.all(np.isfinite(a)):
        raise ValueError('64 finite amplitude observations in >=4 independent scrambles required')
    means=a.mean(axis=1);se=float(np.std(means,ddof=1)/math.sqrt(len(means)))
    if not density_identity:raise ValueError('compact finite-density identity required')
    return dict(plan=plan.record(),mean=float(means.mean()),sampling_se=se,
        sampling_halfwidth95=float(student.ppf(.975,len(means)-1)*se),
        points=a.size,scrambles=len(means),density_identity=density_identity,
        errors={k:None for k in ERRORS},independent_validation=False)


def assemble(compact,boundary_result,*,absolute_error_target,diagnostics=None):
    if compact['plan']!=boundary_result['plan']:raise ValueError('compact/boundary domains differ')
    if not math.isfinite(absolute_error_target) or absolute_error_target<=0:
        raise ValueError('explicit positive total absolute target required')
    complete=boundary_result['complete_domain']
    value=compact['mean']+boundary_result['boundary_amplitude'] if complete else None
    errors=combine([Value(0.,compact['errors']),Value(0.,boundary_result['errors'] or {})]).errors
    unknown=list(boundary_result['unassessed_errors'])
    for label,row in (('compact',compact['errors']),('boundary',boundary_result['errors'] or {})):
        unknown.extend(f'{label}:{k}' for k in sorted(set(ERRORS)|set(row)) if row.get(k) is None)
    analytic=compact['plan'].get('mode')=='analytic'
    assessed=complete and not unknown
    indicator=(compact['sampling_halfwidth95']+boundary_result['quadrature_indicator']+
               math.fsum(errors.values())) if assessed else None
    if analytic:unknown.append('analytic_approximation:omitted_terms')
    return dict(amplitude=value,approximate_amplitude=value if analytic else None,
        analytic_approximation=analytic,complete_domain=complete,compact_points=compact['points'],
        sampling_se=compact['sampling_se'],sampling_halfwidth95=compact['sampling_halfwidth95'],
        boundary_quadrature_indicator=boundary_result['quadrature_indicator'],errors=errors,
        unassessed_errors=unknown,absolute_error_target=absolute_error_target,
        numerical_empirical_indicator=indicator,
        combined_empirical_indicator=None if analytic else indicator,
        omitted_approximation_error=None if analytic else 0.,
        diagnostic_report=diagnostics,diagnostics_added_to_amplitude=False,
        meets_numerical_target=bool(assessed and boundary_result['quadrature_targets_met'] and indicator<=absolute_error_target),
        meets_empirical_target=bool(not analytic and assessed and boundary_result['quadrature_targets_met'] and indicator<=absolute_error_target),
        rigorous_error_bound=False,full_amplitude_certified=False,new_mc_points=0)
