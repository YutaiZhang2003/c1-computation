"""Conservative radial-contour domains for native torus sewing series.

These are sufficient geometric guards, not finite-order error estimates.
No series variable or recursion coefficient is re-expanded here.
"""
from __future__ import annotations
import cmath
import math
from dataclasses import dataclass


class SewingDomainError(ValueError):
    """No justified Taylor/sewing representation was supplied."""


@dataclass(frozen=True)
class SewingChart:
    channel: str
    variables: tuple[complex, complex, complex]
    logs: tuple[complex, complex, complex]
    margins: tuple[float, ...]

    @property
    def guarded(self):
        return (all(math.isfinite(z.real) and math.isfinite(z.imag) for z in self.logs)
                and all(z.real < 0 for z in self.logs)
                and all(math.isfinite(m) and m > 1e-12 for m in self.margins))

    def collar_certificate(self):
        """Construct and independently check three collars in the annulus cover.

        Radii are stored as logarithms, including in the long-cylinder limit.
        This witnesses geometric sewing only, not convergence of a truncation.
        """
        return disjoint_collars(self)

    @property
    def admissible(self):
        return self.guarded and self.collar_certificate()['passed']

    def require(self):
        if not self.admissible:
            raise SewingDomainError(
                f'{self.channel} is outside the native sewing contour domain: '
                f'|variables|={tuple(abs(z) for z in self.variables)}, margins={self.margins}')
        return self


def _logadd(a, b):
    if a == -math.inf: return b
    if b == -math.inf: return a
    high, low = max(a,b), min(a,b)
    return high + math.log1p(math.exp(low-high))


def _annulus(center, low, high, name):
    # The middle third leaves positive room on both sides of every collar.
    width = high-low
    return dict(name=name,center=[float(center),0.],
                log_inner_radius=low+width/3,log_outer_radius=high-width/3)


def verify_collars(chart, annuli):
    """Check collar widths, pairwise separation, marks and every deck image.

    A radial envelope disjoint from its Q-image proves separation from all
    nonzero powers of Q, including their phases. Nested discs are allowed;
    their annuli must be disjoint. No sampled-boundary intersection test is used.
    """
    if not chart.guarded or len(annuli)!=3:
        return dict(passed=False,reason='invalid_native_domain_or_collar_count')
    if chart.channel=='necklace':
        loop=sum(z.real for z in chart.logs)
        marks=[(1+0j,0.),(chart.variables[0],chart.logs[0].real),
               (chart.variables[0]*chart.variables[1],sum(z.real for z in chart.logs[:2]))]
    elif chart.channel=='pair_ope':
        v,a,b=chart.variables; loop=sum(z.real for z in chart.logs[1:])
        marks=[(1+0j,0.),(1+v,math.log(abs(1+v))),(a,chart.logs[1].real)]
    elif chart.channel=='comb_ope':
        x,y,q=chart.variables; loop=chart.logs[2].real
        marks=[(1+0j,0.),(1+x*y,math.log(abs(1+x*y))),
               (1+y,math.log(abs(1+y)))]
    else:
        return dict(passed=False,reason='unknown_native_chart')
    widths=[]; separation=[]; mark_clearance=[]; envelopes=[]
    for a in annuli:
        center=complex(*a['center']); low=a['log_inner_radius']; high=a['log_outer_radius']
        if center not in (0j,1+0j) or not all(math.isfinite(v) for v in (low,high)) or low>=high:
            return dict(passed=False,reason='invalid_annulus')
        widths.append(high-low)
        if center==0:
            envelopes.append((low,high))
        else:
            if high>=0:return dict(passed=False,reason='fusion_collar_contains_handle')
            radius=math.exp(high)
            envelopes.append((math.log1p(-radius),math.log1p(radius)))
        for point, log_radius in marks:
            distance=abs(point-center)
            log_distance=log_radius if center==0 else (math.log(distance) if distance else -math.inf)
            mark_clearance.append(max(low-log_distance,log_distance-high))
    for i,a in enumerate(annuli):
        for b in annuli[i+1:]:
            distance=abs(complex(*a['center'])-complex(*b['center']))
            lowa,higha=a['log_inner_radius'],a['log_outer_radius']
            lowb,highb=b['log_inner_radius'],b['log_outer_radius']
            if distance==0:
                separation.append(max(lowa-highb,lowb-higha))
            else:
                ld=math.log(distance)
                separation.append(max(ld-_logadd(higha,highb),
                    lowa-_logadd(ld,highb),lowb-_logadd(ld,higha)))
    # Include punctures in the envelope: collars must avoid their images too.
    lower=min([a for a,b in envelopes]+[r for p,r in marks])
    upper=max([b for a,b in envelopes]+[r for p,r in marks])
    deck=lower-loop-upper
    clearance=min(widths+separation+mark_clearance+[deck])
    return dict(passed=clearance>1e-12,pairwise_disjoint=min(separation)>1e-12,
        puncture_free=min(mark_clearance)>1e-12,deck_images_disjoint=deck>1e-12,
        minimum_log_clearance=clearance,deck_log_clearance=deck,
        loop_log_magnitude=loop,reason=None if clearance>1e-12 else 'collars_touch_or_overlap')


def disjoint_collars(chart):
    """A constructive witness for the existing sufficient contour guards."""
    base=dict(version='genus1-disjoint-collars-v1',channel=chart.channel,
              annuli=[],scope='native_radial_pants_with_all_multiplicative_deck_images')
    if not chart.guarded:
        return dict(base,passed=False,reason='outside_native_contour_domain')
    if chart.channel=='necklace':
        upper=0.; annuli=[]
        gluing=[]
        for i,z in enumerate(chart.logs):
            lower=upper+z.real
            annuli.append(_annulus(0,lower,upper,f'necklace_edge_{i}'))
            gluing.append((f'u/s{i}',f's{i+1}/u',z.real,upper,False))
            upper=lower
        plane='u=exp(i*w), marks at 1,q1,q1*q2'
    elif chart.channel=='pair_ope':
        v,a,b=map(abs,chart.variables)
        # R contains both fused marks and fits between the two loop contours.
        limit=min(1-a,math.expm1(-chart.logs[2].real) if -chart.logs[2].real<1 else 1.)
        radius=(v+limit)/2
        annuli=[_annulus(1,chart.logs[0].real,math.log(radius),'pair_fusion'),
                _annulus(0,chart.logs[1].real,math.log1p(-radius),'pair_inner_loop'),
                _annulus(0,math.log1p(radius),-chart.logs[2].real,'pair_outer_loop')]
        plane='u=exp(-i*(w-w_b)), marks at 1,1+v,q_first'
        gluing=[('u-1','v/(u-1)',chart.logs[0].real,0.,False),
                ('u','a/u',chart.logs[1].real,0.,False),
                ('1/u','b*u',chart.logs[2].real,0.,True)]
    elif chart.channel=='comb_ope':
        x,y,q=map(abs,chart.variables)
        limit=(1-q)/(1+q); radius=(y+limit)/2
        annuli=[_annulus(1,chart.logs[0].real+chart.logs[1].real,chart.logs[1].real,'comb_pair'),
                _annulus(1,chart.logs[1].real,math.log(radius),'comb_cluster'),
                _annulus(0,math.log1p(radius),math.log1p(-radius)-chart.logs[2].real,'comb_handle')]
        plane='u=exp(-i*(w-w_b)), marks at 1,1+x*y,1+y'
        gluing=[('(u-1)/y','x*y/(u-1)',chart.logs[0].real,chart.logs[1].real,False),
                ('u-1','y/(u-1)',chart.logs[1].real,0.,False),
                ('1/u','q*u',chart.logs[2].real,0.,True)]
    else:
        return dict(base,passed=False,reason='unknown_native_chart')
    plumbing=[]
    for a,(xi,eta,logq,scale,inverted) in zip(annuli,gluing):
        lo,hi=a['log_inner_radius'],a['log_outer_radius']
        radii=(scale-lo,logq+hi-scale) if inverted else (hi-scale,logq+scale-lo)
        plumbing.append(dict(edge=a['name'],xi=xi,eta=eta,log_sewing_magnitude=logq,
            log_coordinate_radii=list(radii),log_product_margin=sum(radii)-logq))
    report=verify_collars(chart,annuli)
    products=all(p['log_product_margin']>1e-12 and max(p['log_coordinate_radii'])<0 for p in plumbing)
    report['passed']=report['passed'] and products
    return dict(base,annuli=annuli,coordinate_plane=plane,plumbing=plumbing,
                plumbing_product_bounds_passed=products,**report)


def _log(z):
    return cmath.log(z) if z else complex(-math.inf)


def _exp(z):
    try:
        return cmath.exp(z)
    except OverflowError:
        return complex(math.inf)


def pair_chart(delta, log_first, log_second):
    v = _exp(-1j*delta)-1
    a,b = cmath.exp(log_first),cmath.exp(log_second)
    # A fusion disc centered at u=1 must fit between the spectator contours
    # |u|=|q_first| and |u|=1/|q_second|. Strict inequalities leave room
    # for the actual sewing contours outside each insertion.
    margins=(1-abs(v)-abs(a), 1-abs(b)*(1+abs(v)))
    return SewingChart('pair_ope',(v,a,b),(_log(v),log_first,log_second),margins)


def comb_chart(delta, total, log_loop):
    v,y = _exp(-1j*delta)-1,_exp(-1j*total)-1
    x = v/y if y else complex(math.inf)
    q=cmath.exp(log_loop)
    # Nested local discs require |x|<1. The outer cluster fits in
    # 1-|y| < |u| < 1+|y|; its q-image must be strictly separated.
    margins=(1-abs(x), 1-abs(y)-abs(q)*(1+abs(y)))
    return SewingChart('comb_ope',(x,y,q),(_log(x),_log(y),log_loop),margins)


def necklace_chart(logs):
    return SewingChart('necklace',tuple(map(cmath.exp,logs)),tuple(logs),
                       tuple(-z.real for z in logs))
