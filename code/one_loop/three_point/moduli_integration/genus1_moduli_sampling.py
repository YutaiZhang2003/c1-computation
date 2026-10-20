"""Torus and nonseparating coordinates, sampling, and Monte Carlo summaries."""
from __future__ import annotations

# genus1_cusp_coordinates
# Six labeled longest-gap sectors in the ORIGINAL flat torus measure.
# 
# There is no outgoing-label fold here: sum all six sectors once. These maps
# define integration geometry, not a sewing-series convergence certificate.



from dataclasses import dataclass
from itertools import permutations
import math

TAU = 2 * math.pi
SECTORS = tuple(permutations(range(3)))
SECTOR_JACOBIAN = 1 / TAU**2
TWIST_INTEGRATED_JACOBIAN = 1 / TAU


def positive(value, name):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(name + ' must be finite and positive')
    return value


@dataclass(frozen=True)
class ShortPoint:
    sector: tuple[int, int, int]
    s1: float
    s2: float
    x1: float
    x2: float

    def __post_init__(self):
        if self.sector not in SECTORS:
            raise ValueError('one of the six labeled sectors required')
        positive(self.s1, 's1'); positive(self.s2, 's2')
        if any(not math.isfinite(x) or not 0 <= x < TAU for x in (self.x1, self.x2)):
            raise ValueError('horizontal coordinates must be in [0,2pi)')

    @property
    def total(self):
        return self.s1 + self.s2

    def positions(self):
        """Unrolled representatives, with the incoming operator at zero."""
        p = [0j] * 3
        p[self.sector[1]] = complex(self.x1, self.s1)
        p[self.sector[2]] = complex(self.x2, self.total)
        origin = p[2]
        return tuple(z-origin for z in p)

    def lower_length(self, tail_start=8.):
        return max(self.s1, self.s2, TAU*positive(tail_start, 'tail start')-self.total)

    def record(self):
        return dict(sector=list(self.sector), s1=self.s1, s2=self.s2,
                    x1=self.x1, x2=self.x2)

    @classmethod
    def from_record(cls, row):
        return cls(tuple(row['sector']), *(float(row[k]) for k in ('s1','s2','x1','x2')))


@dataclass(frozen=True)
class CuspPoint:
    short: ShortPoint
    length: float
    theta: float

    def __post_init__(self):
        positive(self.length, 'empty gap')
        if self.length < max(self.short.s1, self.short.s2)-1e-12:
            raise ValueError('the cut must be a largest empty gap')
        if not math.isfinite(self.theta) or not -math.pi <= self.theta < math.pi:
            raise ValueError('twist must be in [-pi,pi)')

    @property
    def total(self):
        return self.length + self.short.total

    @property
    def tau(self):
        return complex(self.theta, self.total) / TAU

    def torus_positions(self):
        return tuple(wrap(z, self.tau) for z in self.short.positions())

    def record(self):
        return dict(short=self.short.record(), length=self.length, theta=self.theta)

    @classmethod
    def from_record(cls, row):
        return cls(ShortPoint.from_record(row['short']), row['length'], row['theta'])


def wrap(z, tau):
    """Choose the flat fundamental parallelogram; only geometry uses this."""
    tau = complex(tau)
    positive(tau.imag, 'torus height')
    y = z.imag / (TAU*tau.imag)
    x = z.real/TAU-y*tau.real
    return TAU*((x % 1) + (y % 1)*tau)


def from_torus(w1, w2, tau):
    """Inverse sector assignment, including shear when crossing the seam."""
    tau = complex(tau)
    if not -.5 <= tau.real < .5:
        raise ValueError('reduce tau1 to [-1/2,1/2) first')
    points = (wrap(complex(w1), tau), wrap(complex(w2), tau), 0j)
    L = TAU*tau.imag
    labels = sorted(range(3), key=lambda i: (points[i].imag, i))
    gaps = [((points[labels[(j+1) % 3]].imag-points[labels[j]].imag) % L)
            for j in range(3)]
    cut = max(range(3), key=lambda j: (gaps[j], -labels[j]))
    sector = tuple(labels[(cut+1+j) % 3] for j in range(3))
    p = []
    for i in sector[1:]:
        z = points[i]-points[sector[0]]
        if z.imag < 0:
            z += TAU*tau
        p.append(complex(z.real % TAU, z.imag))
    short = ShortPoint(sector, p[0].imag, p[1].imag-p[0].imag, p[0].real, p[1].real)
    return CuspPoint(short, L-short.total, TAU*tau.real)


def cylinder_displacement(z):
    return complex((z.real+math.pi) % TAU-math.pi, z.imag)


def regular_mask(short, pair_radius, tail_start=8.):
    """Exact collision MEMBERSHIP on every longest-gap length/twist ray.

    A nonhorizontal image is vertically at least the empty gap away, and
    ell >= 2pi*T/3. Its distance cannot enter these collision discs.
    """
    radius = positive(pair_radius, 'pair radius')
    if radius >= min(math.pi, TAU*positive(tail_start, 'tail start')/3):
        raise ValueError('collision radius exceeds the uniform cylinder-mask domain')
    p = short.positions()
    return min(abs(cylinder_displacement(p[a]-p[b]))
               for a,b in ((0,1),(0,2),(1,2))) >= radius


def control_window(short, inner=math.pi, outer=TAU):
    positive(inner, 'inner window'); positive(outer, 'outer window')
    if inner >= outer:
        raise ValueError('control window requires inner < outer')
    if short.total <= inner:
        return 1.
    if short.total >= outer:
        return 0.
    v = (short.total-inner)/(outer-inner)
    return 1-3*v*v+2*v*v*v


# genus1_cusp_sampling
# Normalized cusp mixtures and independently scrambled component nets.
# 
# All densities are in physical d^2tau d^2w1 d^2w2, except short_density,
# which is in sector counting measure times ds1 ds2 dx1 dx2. Collar points
# are zeros of the regular integrand, never conditionally rejected samples.



from dataclasses import dataclass
import math
import numpy as np
from scipy.stats import qmc

def integer(n, name, minimum=0):
    if isinstance(n, (bool, np.bool_)) or int(n) != n or n < minimum:
        raise ValueError(name + ' must be an integer >= ' + str(minimum))
    return int(n)


def _unit(row):
    row = np.asarray(row, float)
    if np.any(~np.isfinite(row)) or np.any(row < 0) or np.any(row >= 1):
        raise ValueError('unit coordinates must be in [0,1)')
    # Zero is a measure-zero endpoint, but can occur in a digital net.
    return np.maximum(row, np.finfo(float).eps/2)


def tail_density(L, tail_start=8.):
    L0 = TAU*positive(tail_start, 'tail start')
    return math.sqrt(L0)/(2*L**1.5) if math.isfinite(L) and L >= L0-1e-12 else 0.


def cluster_short(row, width):
    """Four coordinates; pack the discrete sector into the first angle."""
    p = _unit(row)
    if p.shape != (4,):
        raise ValueError('four short coordinates required')
    S = positive(width, 'cluster width')*math.sqrt(p[0])
    code = 6*p[2]; sector = min(5, int(code))
    return ShortPoint(SECTORS[sector], S*p[1], S*(1-p[1]),
                      TAU*(code-sector), TAU*p[3])


def map_cusp(row, *, tail_start=8., width=None):
    p = _unit(row)
    if p.shape != (6,):
        raise ValueError('six cusp coordinates required')
    L = TAU*positive(tail_start, 'tail start')/(1-p[0])**2
    theta = TAU*(p[1]-.5)
    if width is None:
        tau = complex(theta, L)/TAU
        return from_torus(TAU*(p[2]+tau*p[3]), TAU*(p[4]+tau*p[5]), tau)
    if positive(width, 'cluster width') > math.pi*tail_start:
        raise ValueError('cluster proposal would violate longest-gap ownership')
    short = cluster_short(p[2:], width)
    return CuspPoint(short, L-short.total, theta)


def cusp_density(point, *, tail_start=8., width=None):
    f = tail_density(point.total, tail_start)
    if width is None:
        return f/(TAU*point.total**2)
    positive(width, 'cluster width')
    if width > math.pi*tail_start:
        raise ValueError('invalid cluster density support')
    return f/(6*math.pi*width**2) if point.short.total < width else 0.


def short_density(short, width):
    positive(width, 'cluster width')
    return 1/(12*math.pi**2*width**2) if short.total < width else 0.


@dataclass(frozen=True)
class Mixture:
    widths: tuple = (None, TAU, math.pi/2)
    weights: tuple = (.25, .5, .25)
    short: bool = False

    def __post_init__(self):
        if not self.widths or len(self.widths) != len(self.weights):
            raise ValueError('matching nonempty component widths and weights required')
        if any(not math.isfinite(w) or w < 0 for w in self.weights):
            raise ValueError('finite nonnegative mixture weights required')
        if abs(math.fsum(self.weights)-1) > 1e-14:
            raise ValueError('mixture weights must sum to one')
        if self.short and any(b is None for b in self.widths):
            raise ValueError('the short addback has compact triangle components only')
        if not self.short and not any(b is None and w > 0 for b,w in zip(self.widths,self.weights)):
            raise ValueError('a full-support cusp component is required')
        for b in self.widths:
            if b is not None: positive(b, 'component width')
        # Complete base-two nets require a power-of-two fraction per component.
        if any(w and abs(math.log2(w)-round(math.log2(w))) > 1e-12 for w in self.weights):
            raise ValueError('each nonzero component weight must be a power of two')

    def density(self, point, tail_start=8.):
        return math.fsum(w*(short_density(point,b) if self.short else
                           cusp_density(point,tail_start=tail_start,width=b))
                         for b,w in zip(self.widths,self.weights) if w)

    def samples(self, power, seed, replicate, *, tail_start=8., stream=0):
        """Yield (stable sample id, geometry, FULL mixture density).

        Every component gets a separate scrambled complete net. SeedSequence
        separates addback/remainder streams and replicates without offsets
        that can collide. Increasing power preserves each component prefix.
        """
        power = integer(power, 'power', 1)
        seed = integer(seed, 'seed'); replicate = integer(replicate, 'replicate')
        stream = integer(stream, 'stream')
        for component,(b,w) in enumerate(zip(self.widths,self.weights)):
            if not w: continue
            exponent = power+round(math.log2(w))
            if exponent < 0:
                raise ValueError('too few samples for complete mixture component nets')
            state = np.random.SeedSequence([seed,stream,replicate,component])
            scramble = int(state.generate_state(1)[0])
            points = qmc.Sobol(d=4 if self.short else 6, scramble=True,
                              seed=scramble).random_base2(exponent)
            for index,p in enumerate(points):
                g = cluster_short(p,b) if self.short else map_cusp(p,tail_start=tail_start,width=b)
                density = self.density(g,tail_start)
                if not math.isfinite(density) or density <= 0:
                    raise FloatingPointError('invalid full proposal density')
                yield f'c{component}_{index:08d}', g, density


def summarize_independent(components):
    """Complex means/SEs, summing independent component variances, not covariances."""
    if not components:
        raise ValueError('at least one component required')
    means, variances = [], []
    for raw in components:
        z = np.asarray(raw,complex)
        if z.ndim != 1 or len(z) < 2 or not np.all(np.isfinite(z)):
            raise ValueError('at least two finite complete scramble estimates required')
        means.append(complex(np.mean(z)))
        variances.append((np.var(z.real,ddof=1)/len(z),np.var(z.imag,ddof=1)/len(z)))
    mean = sum(means,0j)
    se = complex(*(math.sqrt(math.fsum(v[i] for v in variances)) for i in (0,1)))
    return dict(mean=mean, standard_error=se,
                amplitude=-mean.imag/(4*math.pi), amplitude_sampling_se=se.imag/(4*math.pi))


# genus1_three_point_sampling
# Sampling maps shared by the retained torus three-point production run.
# 
# These helpers were extracted unchanged from the superseded equal-split
# refinement driver.  Keeping them here makes the c-recursion channel-atlas
# campaign self-contained without retaining that older production workflow.




import math
from typing import Sequence

import numpy as np
from scipy.special import betaincinv, gammaln
from scipy.stats import qmc


TAIL_SEED_OFFSET = 20000


def mean_and_standard_error(
    values: Sequence[complex],
) -> tuple[complex, complex]:
    """Return component-wise standard errors across independent scrambles."""
    array = np.asarray(values, dtype=np.complex128)
    mean = complex(np.mean(array))
    if len(array) < 2:
        return mean, complex(float("nan"), float("nan"))
    standard_error = complex(
        float(np.std(array.real, ddof=1) / math.sqrt(len(array))),
        float(np.std(array.imag, ddof=1) / math.sqrt(len(array))),
    )
    return mean, standard_error


def combine_independent_strata(
    strata: Sequence[Sequence[complex]],
) -> tuple[complex, complex]:
    """Sum stratum means and their independent-scramble variances.

    Scramble labels belonging to separately seeded strata have no meaningful
    pairing.  Estimating the error from their row-wise sums introduces noisy
    sample-covariance terms.  For independent strata, the variance of the sum
    is instead the sum of the separately estimated stratum variances.
    """
    if not strata:
        raise ValueError("at least one stratum is required")
    summaries = [mean_and_standard_error(values) for values in strata]
    mean = sum((summary[0] for summary in summaries), 0.0j)
    standard_error = complex(
        math.sqrt(sum(summary[1].real**2 for summary in summaries)),
        math.sqrt(sum(summary[1].imag**2 for summary in summaries)),
    )
    return complex(mean), standard_error


def sobol_points(dimension: int, power: int, seed: int) -> np.ndarray:
    """Generate one independently scrambled base-two Sobol net."""
    return qmc.Sobol(d=int(dimension), scramble=True, seed=int(seed)).random_base2(
        int(power)
    )


def ordered_gap_importance(
    first_coordinate: float,
    second_coordinate: float,
    *,
    alpha: float,
) -> tuple[float, float, float]:
    r"""Return ordered puncture heights and the labeled-square measure weight.

    The three necklace gaps are sampled from a symmetric
    ``Dirichlet(alpha,alpha,alpha)`` distribution using stick breaking.  The
    returned weight is ``2/pdf``: the factor two restores the two label
    orderings in the original unit square.  At ``alpha=1`` the Dirichlet
    density is two and the weight is exactly one.
    """
    alpha = float(alpha)
    if alpha <= 0.0:
        raise ValueError("gap alpha must be positive")
    epsilon = np.finfo(float).eps
    first_coordinate = min(1.0 - epsilon, max(epsilon, float(first_coordinate)))
    second_coordinate = min(1.0 - epsilon, max(epsilon, float(second_coordinate)))
    first_gap = float(betaincinv(alpha, 2.0 * alpha, first_coordinate))
    split = float(betaincinv(alpha, alpha, second_coordinate))
    second_gap = (1.0 - first_gap) * split
    third_gap = (1.0 - first_gap) * (1.0 - split)
    # For alpha<1, betaincinv can round an exponentially small positive gap
    # to zero. The endpoint has zero measure, so replace only that underflow.
    gaps = np.maximum(
        np.asarray([first_gap, second_gap, third_gap], dtype=float),
        np.finfo(float).tiny,
    )
    gaps /= np.sum(gaps)
    first_gap, second_gap, third_gap = (float(value) for value in gaps)
    log_density = float(gammaln(3.0 * alpha) - 3.0 * gammaln(alpha))
    log_density += (alpha - 1.0) * (
        math.log(first_gap) + math.log(second_gap) + math.log(third_gap)
    )
    weight = 2.0 * math.exp(-log_density)
    return first_gap, first_gap + second_gap, float(weight)


def tail_height_and_jacobian(
    coordinate: float,
    *,
    tail_start: float,
    proposal_exponent: float,
) -> tuple[float, float]:
    r"""Map a uniform coordinate to an exact integral on ``[tail_start,infinity)``.

    The proposal survival function is proportional to
    ``tau2**(1-proposal_exponent)``. It changes variance only; no asymptotic
    ansatz enters the estimator.
    """
    exponent = float(proposal_exponent)
    if exponent <= 1.0:
        raise ValueError("proposal exponent must exceed one")
    complement = 1.0 - float(coordinate)
    if not 0.0 < complement <= 1.0:
        raise ValueError("tail coordinate must lie in [0,1)")
    alpha = exponent - 1.0
    tau2 = float(tail_start) * complement ** (-1.0 / alpha)
    jacobian = (
        float(tail_start)
        / alpha
        * complement ** (-1.0 / alpha - 1.0)
    )
    return float(tau2), float(jacobian)
