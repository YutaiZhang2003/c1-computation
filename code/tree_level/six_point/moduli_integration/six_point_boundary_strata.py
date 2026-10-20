"""Complete labelled boundary combinatorics and source-frame collar diagnostics.

For a stable split S|Sc define sigma as the maximum of both cross-ratio
moduli |Dij Dkl/(Dik Djl)| and |Dij Dkl/(Dil Djk)|, over i,j in S and k,l
in Sc. It vanishes at that separating boundary and is invariant under Mobius
maps, relabelling and exchange of the two sides. Unlike a minimum cross ratio,
it does not mark an unrelated split merely because reference punctures collide.

Sigma is a specified algebraic collar coordinate screen, not a hyperbolic
length, plumbing parameter in one fixed chart, or CFT error bound. Numerical
method coverage is recorded only for explicitly supplied tested patches.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
from functools import lru_cache
import itertools
import math
from collections.abc import Mapping

import numpy as np

VERSION = "sphere-six-labelled-boundaries-max-cross-ratio-v1"
LABELS = tuple(range(6))


def canonical_split(side):
    values = tuple(side)
    if (not 2 <= len(values) <= 4 or len(set(values)) != len(values)
            or any(isinstance(i,(bool,np.bool_)) or not isinstance(i,(int,np.integer))
                   or not 0 <= i < 6 for i in values)):
        raise ValueError("a stable split requires two, three or four distinct labels in 0..5")
    left = tuple(sorted(map(int,values)))
    right = tuple(i for i in LABELS if i not in left)
    return min((left,right),key=lambda group:(len(group),group))


def split_key(side):
    left = canonical_split(side)
    right = tuple(i for i in LABELS if i not in left)
    return ''.join(map(str,left))+'|'+''.join(map(str,right))


def compatible_splits(first, second):
    a,b = set(canonical_split(first)),set(canonical_split(second))
    if a == b:
        return False  # A stratum contains distinct nodes.
    ac,bc = set(LABELS)-a,set(LABELS)-b
    return any(not part for part in (a&b,a&bc,ac&b,ac&bc))


@dataclass(frozen=True)
class BoundaryDivisor:
    key: str
    split: tuple[int, ...]
    complement: tuple[int, ...]
    kind: str
    component_marked_points: tuple[int, ...]
    residual_block_type: str

    def as_dict(self):
        return asdict(self)


@lru_cache(None)
def boundary_divisors():
    sides = sorted({canonical_split(side) for n in (2,3)
                    for side in itertools.combinations(LABELS,n)},key=lambda s:(len(s),s))
    return tuple(BoundaryDivisor(split_key(s),s,tuple(i for i in LABELS if i not in s),
                                'pair' if len(s)==2 else 'triple',
                                (3,5) if len(s)==2 else (4,4),
                                'five-point' if len(s)==2 else 'two-four-point') for s in sides)


@dataclass(frozen=True)
class BoundaryComponent:
    external_labels: tuple[int, ...]
    incident_divisor_keys: tuple[str, ...]
    marked_points: int
    block_type: str


def _components(splits):
    # Orient every separating edge away from the root leaf5. Compatibility
    # makes these clusters laminar. Removing child clusters gives each stable
    # component's original marked leaves; incident edges supply its node marks.
    all_labels = frozenset(LABELS)
    clusters = {frozenset(s) if 5 not in s else all_labels-frozenset(s):split_key(s)
                for s in splits}
    nodes = sorted((*clusters,all_labels),key=lambda s:(len(s),tuple(sorted(s))))
    parents = {c:min((s for s in nodes if c<s),key=len) for c in clusters}
    answer = []
    for node in nodes:
        children = [c for c,p in parents.items() if p==node]
        leaves = set(node)-set().union(*children) if children else set(node)
        edges = [clusters[c] for c in children]
        if node != all_labels:
            edges.append(clusters[node])
        degree = len(leaves)+len(edges)
        if degree < 3:
            raise ArithmeticError("unstable component in compatible boundary tree")
        answer.append(BoundaryComponent(external_labels=tuple(sorted(leaves)),
                           incident_divisor_keys=tuple(sorted(edges)),marked_points=degree,
                           block_type='DOZZ-three-point' if degree==3 else f'{degree}-point'))
    return tuple(answer)


@dataclass(frozen=True)
class BoundaryStratum:
    key: str
    divisor_keys: tuple[str, ...]
    splits: tuple[tuple[int, ...], ...]
    codimension: int
    kind: str
    components: tuple[BoundaryComponent, ...]
    residual_block_type: str

    def as_dict(self):
        return asdict(self)


def _stratum(splits):
    sides = tuple(sorted((canonical_split(s) for s in splits),key=lambda s:(len(s),s)))
    if not 1 <= len(sides) <= 3 or len(set(sides)) != len(sides):
        raise ValueError("one to three distinct boundary nodes required")
    if not all(compatible_splits(a,b) for a,b in itertools.combinations(sides,2)):
        raise ValueError("incompatible labelled boundary splits")
    triples = sum(len(s)==3 for s in sides)
    if len(sides)==1:
        kind = 'pair' if not triples else 'triple'
        residual = 'five-point' if not triples else 'two-four-point'
    elif len(sides)==2:
        kind = 'disjoint-pair' if not triples else 'pair-triple'
        residual = 'four-point'
    else:
        kind = 'star' if not triples else 'comb'
        residual = 'DOZZ-only'
    keys = tuple(split_key(s) for s in sides)
    return BoundaryStratum('&'.join(keys),keys,sides,len(sides),kind,_components(sides),residual)


@lru_cache(None)
def compatible_strata(codimension=1):
    if (isinstance(codimension,(bool,np.bool_))
            or not isinstance(codimension,(int,np.integer)) or codimension not in (1,2,3)):
        raise ValueError("M0,6 stable boundary codimension must be 1,2 or3")
    sides = tuple(d.split for d in boundary_divisors())
    return tuple(_stratum(family) for family in itertools.combinations(sides,codimension)
                 if all(compatible_splits(a,b) for a,b in itertools.combinations(family,2)))


def maximal_trees():
    return compatible_strata(3)


@lru_cache(None)
def _ratio_catalog():
    terms,offsets,witnesses = [],[],[]
    for divisor in boundary_divisors():
        offsets.append(len(terms));local=[]
        for i,j in itertools.combinations(divisor.split,2):
            for k,l in itertools.combinations(divisor.complement,2):
                for d1,d2 in (((i,k),(j,l)),((i,l),(j,k))):
                    terms.append(((i,j),(k,l),d1,d2))
                    local.append(dict(numerator_pairs=((i,j),(k,l)),denominator_pairs=(d1,d2)))
        witnesses.append(tuple(local))
    arrays = np.asarray(terms,dtype=int)
    arrays.setflags(write=False)
    return arrays,tuple(offsets),tuple(witnesses)


def split_log_sizes(differences):
    """All25 permutation-invariant log sizes from labelled native determinants.

    This low-level read-only helper takes no CFT chart or sampling density.
    Off-diagonal determinant zeros/nonfinite values are refused.
    """
    d = np.asarray(differences,dtype=complex)
    if d.shape != (6,6):
        raise ValueError("a six-by-six determinant table is required")
    pairs = d[np.triu_indices(6,1)]
    if np.any(abs(pairs)==0) or not np.isfinite(pairs).all():
        raise ArithmeticError("distinct representable source punctures required")
    with np.errstate(divide='ignore'):
        logs = np.log(abs(d))
    terms,offsets,_ = _ratio_catalog()
    gathered = logs[terms[:,:,0],terms[:,:,1]]
    ratios = gathered[:,0]+gathered[:,1]-gathered[:,2]-gathered[:,3]
    scores = np.maximum.reduceat(ratios,np.asarray(offsets))
    if not np.isfinite(scores).all():
        raise ArithmeticError("nonfinite separating cross ratio")
    return tuple(float(x) for x in scores)


def _threshold(value,name):
    if isinstance(value,(bool,np.bool_)):
        raise ValueError(f'{name} must be a real threshold')
    value = float(value)
    if not math.isfinite(value) or not 0 < value < 1:
        raise ValueError(f'{name} must lie strictly in (0,1)')
    return value


def _exact_below(norms,terms,threshold):
    square = Fraction(threshold)**2
    for n1,n2,d1,d2 in terms:
        norm = lambda pair:norms[tuple(sorted(map(int,pair)))]
        if norm(n1)*norm(n2) >= square*norm(d1)*norm(d2):
            return False
    return True


def mark_degenerations(point, *, onset=.1, deep=.02, tested_patches=None):
    """JSON-compatible exhaustive boundary marking from the ORIGINAL sample.

    ``onset`` and ``deep`` classify geometry only. ``tested_patches`` is an
    optional name->CollarPatch mapping; a hit records that explicit tested
    subregion, not a global qualification of its entire boundary divisor.
    All compatible subsets of marked nodes are retained, including overlaps.
    """
    from six_point_stable_puncture_coordinates import (
        native_differences,_exact_difference_norms,NATIVE_GEOMETRY_VERSION,
    )
    from six_point_analytic_collar_subtraction import CollarPatch
    from six_point_collar_counterterm_geometry import patch_geometry

    onset,deep = _threshold(onset,'onset'),_threshold(deep,'deep')
    if deep > onset:
        raise ValueError("deep threshold must not exceed onset")
    if tested_patches is None:
        tested_patches = {}
    if not isinstance(tested_patches,Mapping) or any(
            not isinstance(name,str) or not name or not isinstance(patch,CollarPatch)
            for name,patch in tested_patches.items()):
        raise TypeError("tested_patches must map nonempty names to explicit CollarPatch objects")
    source = (point.sampled_topology,tuple(map(complex,point.sampled_q)),tuple(point.sampled_ordering))
    d = native_differences(*source)
    scores = split_log_sizes(d)
    terms,offsets,witnesses = _ratio_catalog()
    offlogs = np.log(abs(np.asarray(d)[np.triu_indices(6,1)]))
    uncertainty = 64*np.finfo(float).eps*max(1.,float(abs(offlogs).max()))
    norms = None
    marked = []
    records = []
    for index,(divisor,score) in enumerate(zip(boundary_divisors(),scores)):
        comparisons = {}
        begin = offsets[index];end = offsets[index+1] if index+1<len(offsets) else len(terms)
        for name,threshold in (('onset',onset),('deep',deep)):
            exact = bool(abs(score-math.log(threshold)) <= uncertainty)
            if exact:
                if norms is None:
                    norms = _exact_difference_norms(source)
                hit = _exact_below(norms,terms[begin:end],threshold)
            else:
                hit = score < math.log(threshold)
            comparisons[name] = dict(marked=bool(hit),exact_boundary_comparison=exact)
        local_terms = terms[begin:end]
        logs = np.log(abs(np.asarray(d)[local_terms[:,:,0],local_terms[:,:,1]]))
        values = logs[:,0]+logs[:,1]-logs[:,2]-logs[:,3]
        witness = witnesses[index][int(np.argmax(values))]
        size = math.exp(score) if -744 < score < 709 else None
        record = dict(**divisor.as_dict(),log_size=score,size=size,
                      size_representable=size is not None,witness_cross_ratio=witness,
                      marked_onset=comparisons['onset']['marked'],
                      marked_deep=comparisons['deep']['marked'],comparisons=comparisons,
                      required_factorization=divisor.residual_block_type,
                      numerical_support_status='unqualified-generic-boundary',
                      tested_patch_names=[])
        if record['marked_onset']:
            marked.append(divisor.split)
        records.append(record)
    # Incompatible splits contain a quartet giving reciprocal ratios, so
    # their maxima cannot both be <1. Thresholds here are strictly below1.
    if len(marked)>3 or not all(compatible_splits(a,b) for a,b in itertools.combinations(marked,2)):
        raise ArithmeticError("incompatible marked collars violate the cross-ratio identity")
    hits = []
    covered = set()
    for name,patch in tested_patches.items():
        geometry = patch_geometry(point,patch)
        if not geometry.inside:
            continue
        keys = (split_key(patch.ordering[:2]),split_key(patch.ordering[-2:]))
        covered.update(keys)
        hits.append(dict(name=name,outer_divisor_keys=keys,
                         q=[dict(real=z.real,imag=z.imag) for z in geometry.q],
                         coordinate_log_error=geometry.coordinate_log_error,
                         scope='explicit-tested-subregion-only; no global tail certificate'))
        for record in records:
            if record['key'] in keys:
                record['tested_patch_names'].append(name)
                record['numerical_support_status']='inside-explicit-tested-subregion'
    strata = []
    for n in range(1,len(marked)+1):
        for subset in itertools.combinations(marked,n):
            row = _stratum(subset).as_dict()
            row['tested_patch_names'] = [hit['name'] for hit in hits
                                        if set(row['divisor_keys'])<=set(hit['outer_divisor_keys'])]
            row['numerical_support_status'] = ('inside-explicit-tested-subregion'
                if row['tested_patch_names'] else 'unqualified-generic-stratum')
            strata.append(row)
    marked_keys = [split_key(s) for s in marked]
    return dict(version=VERSION,thresholds=dict(onset=onset,deep=deep),
                metric='maximum of both separating four-puncture cross-ratio moduli',
                threshold_is_CFT_error_bound=False,
                source=dict(topology=source[0],ordering=source[2],
                            q=[dict(real=z.real,imag=z.imag) for z in source[1]],
                            native_geometry_version=NATIVE_GEOMETRY_VERSION),
                divisors=records,marked_divisor_keys=marked_keys,
                deep_divisor_keys=[r['key'] for r in records if r['marked_deep']],
                marked_strata=strata,tested_patch_names=[h['name'] for h in hits],
                tested_patch_hits=hits,
                unhandled_marked_divisor_keys=[key for key in marked_keys if key not in covered],
                all_marked_divisors_covered_by_tested_patches=all(key in covered for key in marked_keys),
                requires_factorization=bool(marked_keys),
                global_CFT_qualification=False)


__all__ = ['BoundaryDivisor','BoundaryComponent','BoundaryStratum','canonical_split','split_key',
           'compatible_splits','boundary_divisors','compatible_strata',
           'maximal_trees','split_log_sizes','mark_degenerations']
