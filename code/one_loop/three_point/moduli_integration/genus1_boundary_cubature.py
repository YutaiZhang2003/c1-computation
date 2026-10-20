"""Bounded adaptive deterministic cubature with immutable evaluation caches.

Embedded tensor Gauss rules supply empirical quadrature indicators, not
rigorous bounds. Unknown physics errors stay unknown. A work limit or missing
input never turns an unevaluated part of a domain into zero.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import itertools
import json
import math
from pathlib import Path
import sqlite3
from functools import lru_cache
import numpy as np


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False,
                                     separators=(',',':')).encode()).hexdigest()


class MissingBoundaryInput(RuntimeError):
    pass


class WorkLimit(RuntimeError):
    pass


class TimeLimit(WorkLimit):
    pass


@dataclass(frozen=True)
class Value:
    value: complex
    errors: dict = field(default_factory=dict)
    comparisons: dict = field(default_factory=dict)

    def __post_init__(self):
        z=complex(self.value)
        if not math.isfinite(z.real) or not math.isfinite(z.imag):
            raise FloatingPointError('nonfinite boundary value')
        for name,e in self.errors.items():
            if not isinstance(name,str) or not name or (e is not None and
                    (not math.isfinite(e) or e<0)):
                raise ValueError('nonnegative error indicator or explicit None required')
        object.__setattr__(self,'value',z)
        object.__setattr__(self,'errors',dict(self.errors))
        comparisons={k:None if v is None else complex(v) for k,v in self.comparisons.items()}
        if any(not isinstance(k,str) or not k or (v is not None and
                (not math.isfinite(v.real) or not math.isfinite(v.imag))) for k,v in comparisons.items()):
            raise ValueError('finite signed comparison or explicit None required')
        object.__setattr__(self,'comparisons',comparisons)

    def scaled(self, weight):
        if not math.isfinite(weight): raise ValueError('finite real weight required')
        return Value(self.value*weight,{k:None if e is None else abs(weight)*e
                                      for k,e in self.errors.items()},
                     {k:None if v is None else weight*v for k,v in self.comparisons.items()})

    def record(self):
        row=dict(value=[self.value.real,self.value.imag],errors=self.errors)
        if self.comparisons:
            row['comparisons']={k:None if v is None else [v.real,v.imag] for k,v in self.comparisons.items()}
        return row

    @classmethod
    def from_record(cls,row):
        return cls(complex(*row['value']),row['errors'],
                   {k:None if v is None else complex(*v) for k,v in row.get('comparisons',{}).items()})


def combine(values, weights=None):
    values=list(values)
    weights=[1.]*len(values) if weights is None else list(weights)
    if len(values)!=len(weights): raise ValueError('weights do not match values')
    rows=[v.scaled(w) for v,w in zip(values,weights)]
    z=complex(math.fsum(v.value.real for v in rows),math.fsum(v.value.imag for v in rows))
    names=set().union(*(v.errors for v in rows)) if rows else set()
    errors={k:None if any(v.errors.get(k,0.) is None for v in rows)
            else math.fsum(v.errors.get(k,0.) for v in rows) for k in names}
    alternatives=set().union(*(v.comparisons for v in rows)) if rows else set()
    # Missing prefixes are unknown, never silently supplied by the high order.
    comparisons={k:None if any(v.comparisons.get(k) is None for v in rows) else
                 complex(math.fsum(v.comparisons[k].real for v in rows),
                         math.fsum(v.comparisons[k].imag for v in rows)) for k in alternatives}
    return Value(z,errors,comparisons)


class PointCache:
    """A separate SQLite cache; identities include physics, mapping and sources.

    Callers must change identity when any integrand definition changes. The
    cache rejects a different identity and verifies each persisted value.
    Missing-input exceptions are deliberately not cached.
    """
    def __init__(self, identity, path=None, *, should_stop=None):
        if not isinstance(identity,dict) or not identity:
            raise ValueError('explicit nonempty integrand identity required')
        self.identity=dict(format='genus1-boundary-cache-v1',integrand=identity)
        self.digest=fingerprint(self.identity)
        if path is not None: Path(path).parent.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(':memory:' if path is None else str(path))
        self.db.execute('CREATE TABLE IF NOT EXISTS metadata (identity TEXT NOT NULL)')
        old=self.db.execute('SELECT identity FROM metadata').fetchall()
        if old and old!=[(self.digest,)]:
            self.db.close(); raise ValueError('cache integrand identity changed')
        if not old: self.db.execute('INSERT INTO metadata VALUES (?)',(self.digest,))
        self.db.execute('CREATE TABLE IF NOT EXISTS points (key TEXT PRIMARY KEY, payload TEXT NOT NULL, sha TEXT NOT NULL)')
        self.db.commit()
        self.memory={}; self.evaluations=0; self.hits=0; self.limit=None
        self.should_stop=should_stop

    def get(self, namespace, point, evaluate):
        if self.should_stop is not None and self.should_stop():raise TimeLimit()
        key=fingerprint([namespace,[float(x).hex() for x in point]])
        if key in self.memory:
            self.hits+=1; return self.memory[key]
        row=self.db.execute('SELECT payload,sha FROM points WHERE key=?',(key,)).fetchone()
        if row:
            if hashlib.sha256(row[0].encode()).hexdigest()!=row[1]:
                raise ValueError('corrupt boundary cache value')
            value=Value.from_record(json.loads(row[0])); self.hits+=1
        else:
            if self.limit is not None and self.evaluations>=self.limit: raise WorkLimit()
            value=evaluate(np.asarray(point,float))
            if not isinstance(value,Value): raise TypeError('cubature integrand must return Value')
            payload=json.dumps(value.record(),sort_keys=True,allow_nan=False)
            self.db.execute('INSERT INTO points VALUES (?,?,?)',
                (key,payload,hashlib.sha256(payload.encode()).hexdigest()))
            self.evaluations+=1
        self.memory[key]=value
        return value

    def close(self):
        self.db.commit(); self.db.close()

    def __enter__(self): return self
    def __exit__(self,*_): self.close()


@dataclass(frozen=True)
class Cell:
    key: str
    bounds: tuple
    evaluate: object = field(compare=False,repr=False)
    depth: int = 0

    def __post_init__(self):
        if not self.key or not 1<=len(self.bounds)<=6 or not callable(self.evaluate):
            raise ValueError('named one- to six-dimensional cell required')
        if any(not math.isfinite(a+b) or not a<b for a,b in self.bounds):
            raise ValueError('finite ordered cell bounds required')

    def split(self, axis):
        a,b=self.bounds[axis];mid=(a+b)/2
        if mid==a or mid==b: raise WorkLimit('cell cannot be subdivided further')
        result=[]
        for interval in ((a,mid),(mid,b)):
            bounds=list(self.bounds);bounds[axis]=interval
            result.append(Cell(self.key,tuple(bounds),self.evaluate,self.depth+1))
        return tuple(result)


@dataclass(frozen=True)
class Controls:
    absolute_tolerance: float
    max_evaluations: int = 20000
    max_cells: int = 256
    max_depth: int = 24
    low_order: int = 2
    high_order: int = 3
    minimum_splits: int = 1

    def __post_init__(self):
        if not math.isfinite(self.absolute_tolerance) or self.absolute_tolerance<=0:
            raise ValueError('positive absolute tolerance required')
        for name in ('max_evaluations','max_cells','max_depth','low_order','high_order','minimum_splits'):
            v=getattr(self,name)
            if type(v) is not int or v<(0 if name=='minimum_splits' else 1):
                raise ValueError('invalid '+name)
        if not 1<=self.low_order<self.high_order<=9 or self.high_order%2!=1:
            raise ValueError('ordered Gauss rules with odd high order <=9 required')


@lru_cache(maxsize=9)
def gauss_rule(order):
    x,w=np.polynomial.legendre.leggauss(order)
    x.flags.writeable=False;w.flags.writeable=False
    return x,w


def _estimate(cell, cache, controls):
    totals=[];line={}; middle=controls.high_order//2; dim=len(cell.bounds)
    for order in (controls.low_order,controls.high_order):
        x,w=gauss_rule(order)
        axes=[((a+b)/2+x*(b-a)/2,w*(b-a)/2) for a,b in cell.bounds]
        values=[];weights=[]
        for index in itertools.product(range(order),repeat=dim):
            point=tuple(axes[j][0][index[j]] for j in range(dim))
            value=cache.get(cell.key,point,cell.evaluate)
            values.append(value);weights.append(math.prod(axes[j][1][index[j]] for j in range(dim)))
            if order==controls.high_order and sum(i!=middle for i in index)<=1:
                line[index]=value.value
        totals.append(combine(values,weights))
    center=(middle,)*dim; scores=[]
    # Refine where the embedded quadrature actually disagrees. Curvature is
    # unsuitable here: both Gauss rules already integrate quadratics exactly,
    # so a large quadratic term used to draw splits away from the true error.
    low_nodes,low_weights=gauss_rule(controls.low_order)
    _,high_weights=gauss_rule(controls.high_order)
    midpoint=np.asarray([(a+b)/2 for a,b in cell.bounds])
    for axis in range(dim):
        high=[];low=[]
        for index,weight in enumerate(high_weights):
            key=list(center);key[axis]=index
            high.append(weight/2*line[tuple(key)])
        a,b=cell.bounds[axis]
        for node,weight in zip(low_nodes,low_weights):
            point=midpoint.copy();point[axis]+=(b-a)/2*node
            try:low.append(weight/2*cache.get(cell.key,point,cell.evaluate).value)
            except WorkLimit as exc:
                # Both integral rules are complete; auxiliary split selection
                # must not discard their valid whole-cell estimate on a stop.
                return dict(cell=cell,value=totals[1],error=abs(totals[1].value-totals[0].value),
                            axis=axis,stop=exc)
        delta=complex(math.fsum(v.real for v in high)-math.fsum(v.real for v in low),
                      math.fsum(v.imag for v in high)-math.fsum(v.imag for v in low))
        score=abs(delta)
        noise=64*np.finfo(float).eps*max((abs(v) for v in (*high,*low)),default=0.)
        scores.append(score if score>noise else 0.)
    # On a flat rule choose the relatively longest side, avoiding endless ties.
    axis=int(np.argmax(scores)) if max(scores)>0 else max(range(dim),key=lambda j:cell.bounds[j][1]-cell.bounds[j][0])
    return dict(cell=cell,value=totals[1],error=abs(totals[1].value-totals[0].value),axis=axis)


def integrate(cells, controls, cache, *, checkpoint=None):
    """Refine the cell with the largest absolute quadrature indicator.

    All root cells are necessary. If a child is unfinished, retain its parent
    estimate; a missing root yields a null result. New work is bounded even
    when called with a warm persistent cache.
    """
    roots=list(cells)
    if not roots or len({(c.key,c.bounds) for c in roots})!=len(roots):
        raise ValueError('nonempty distinct root cells required')
    if len(roots)>controls.max_cells: raise ValueError('root cells exceed cell ceiling')
    begin=cache.evaluations;hits=cache.hits;cache.limit=begin+controls.max_evaluations
    leaves=[];splits=0;reason=None;missing=None;arithmetic=None
    def record(reason):
        full=len(leaves)>=len(roots) and reason not in ('missing_boundary_input','arithmetic_unresolved')
        total=combine(c['value'] for c in leaves) if full else None
        return dict(status='quadrature_indicator_met' if full and reason is None else reason,
            value=None if total is None else total.record()['value'],
            errors=None if total is None else total.errors,
            quadrature_indicator=math.fsum(c['error'] for c in leaves) if full else None,
            requested_absolute_tolerance=controls.absolute_tolerance,root_coverage_complete=full,
            root_cells=len(roots),leaf_cells=len(leaves),splits=splits,
            new_evaluations=cache.evaluations-begin,cache_hits=cache.hits-hits,
            missing_input=missing,arithmetic_error=arithmetic,rigorous_error_bound=False,
            cells=[dict(key=c['cell'].key,bounds=c['cell'].bounds,depth=c['cell'].depth,
                        value=c['value'].record()['value'],quadrature_indicator=c['error']) for c in leaves])
    def persist():
        if checkpoint is not None:
            cache.db.commit();checkpoint(record('in_progress'))
    try:
        for cell in roots:
            leaves.append(_estimate(cell,cache,controls));persist()
            if 'stop' in leaves[-1]:raise leaves[-1]['stop']
        while True:
            error=math.fsum(c['error'] for c in leaves)
            if error<=controls.absolute_tolerance and splits>=controls.minimum_splits: break
            candidates=[(i,c) for i,c in enumerate(leaves) if c['cell'].depth<controls.max_depth]
            if not candidates or len(leaves)>=controls.max_cells:
                reason='cell_limit';break
            i,parent=max(candidates,key=lambda x:x[1]['error'])
            children=[]
            for cell in parent['cell'].split(parent['axis']):
                child=_estimate(cell,cache,controls)
                if 'stop' in child:raise child['stop']
                children.append(child)
            leaves[i:i+1]=children;splits+=1
            persist()
    except TimeLimit:
        reason='wall_time_limit'
    except WorkLimit:
        reason='evaluation_limit'
    except MissingBoundaryInput as exc:
        reason='missing_boundary_input';missing=str(exc)
    except FloatingPointError as exc:
        reason='arithmetic_unresolved';arithmetic=str(exc)
    finally:
        cache.db.commit();cache.limit=None
    result=record(reason)
    if checkpoint is not None:checkpoint(result)
    return result
