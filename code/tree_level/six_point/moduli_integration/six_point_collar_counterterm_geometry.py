"""Fixed labelled collar counterterms in the unchanged sampled sphere measure.

The target is a physical comb patch, not the chart selected to evaluate CFT.
Only the original sampled frame and its full mixture density enter this
conversion. No sampler, routing policy, or density is changed by this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math

import numpy as np

VERSION = "sphere-fixed-collar-source-measure-v1"


@dataclass(frozen=True)
class PatchGeometry:
    inside: bool
    q: tuple[complex, complex, complex] | None
    ordering: tuple[int, ...]
    log_source_inverse_jacobian: float | None
    source_log_density: float
    coordinate_log_error: float | None
    exact_membership_used: bool = False


def _exact_target(source, ordering):
    # Reuse the exact binary-input determinant convention of native geometry.
    # This path only decides numerically ambiguous patch membership.
    from six_point_stable_puncture_coordinates import _ExactComplex, _difference_entries

    topology, q, source_order = source
    table = {}
    exact_q = tuple(_ExactComplex(z.real, z.imag) for z in q)
    for (i, j), value in _difference_entries(topology, exact_q).items():
        value = _ExactComplex.cast(value)
        a, b = source_order[i], source_order[j]
        table[a, b], table[b, a] = value, -value
    a, u, v, w, e, c = ordering
    factors = (((u,a),(v,c),(u,c),(v,a)),
               ((v,a),(w,c),(v,c),(w,a)),
               ((w,a),(e,c),(w,c),(e,a)))
    return tuple(table[n1]*table[n2]/(table[d1]*table[d2])
                 for n1,n2,d1,d2 in factors)


def patch_geometry(point, patch, *, coordinate_tolerance=1e-9):
    """Locate one fixed collar from the sampled frame, independently of routing.

    A potentially interior but unrepresentable or unfaithful coordinate is
    refused. Near cut boundaries, exact rational functions of the original
    binary input decide membership; no fuzzy enlargement of the patch occurs.
    The inverse Jacobian is |d z_source,moving / d(a,x,b)_patch|.
    """
    from six_point_stable_puncture_coordinates import native_differences
    from six_point_select_integration_channels import coordinate_fidelity_error

    ordering = tuple(patch.ordering)
    if sorted(ordering) != list(range(6)):
        raise ValueError("patch ordering must permute six labels")
    density = float(point.log_density)
    tolerance = float(coordinate_tolerance)
    if not math.isfinite(density) or not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("finite source density and positive coordinate tolerance required")
    source_q = tuple(map(complex, point.sampled_q))
    source_order = tuple(point.sampled_ordering)
    source = (point.sampled_topology, source_q, source_order)
    d = native_differences(*source)
    with np.errstate(divide="ignore"):
        logs = np.log(abs(d))+1j*np.angle(d)
    a, u, v, w, e, c = ordering
    qlogs = np.asarray((logs[u,a]+logs[v,c]-logs[u,c]-logs[v,a],
                        logs[v,a]+logs[w,c]-logs[v,c]-logs[w,a],
                        logs[w,a]+logs[e,c]-logs[w,c]-logs[e,a]))
    if not np.isfinite(qlogs).all():
        raise ArithmeticError("nonfinite fixed-patch cross ratio")
    finite_logs = logs.real[np.isfinite(logs.real)]
    uncertainty = 64*np.finfo(float).eps*max(1.,float(abs(finite_logs).max()))
    lo, hi, bottom, top = patch.middle_rectangle
    max_x = max(abs(complex(re, im)) for re in (lo,hi) for im in (bottom,top))
    bounds = (math.log(patch.left_radius), math.log(max_x), math.log(patch.right_radius))
    outside = lambda exact=False: PatchGeometry(False,None,ordering,None,density,None,exact)
    if any(value.real > bound+uncertainty for value,bound in zip(qlogs,bounds)):
        return outside()
    with np.errstate(over="raise", under="ignore", invalid="raise"):
        q = tuple(complex(z) for z in np.exp(qlogs))
    if any(z == 0 or not math.isfinite(abs(z)) for z in q):
        raise ArithmeticError("potentially interior patch coordinates are unrepresentable")
    ax, x, bx = q
    ambiguous = (abs(qlogs[0].real-bounds[0]) <= uncertainty
                 or abs(qlogs[2].real-bounds[2]) <= uncertainty
                 or any(abs(x.real-bound) <= uncertainty*max(1.,abs(x)) for bound in (lo,hi))
                 or any(abs(x.imag-bound) <= uncertainty*max(1.,abs(x)) for bound in (bottom,top)))
    if ambiguous:
        exact = _exact_target(source, ordering)
        inside = (exact[0].norm_squared < Fraction(patch.left_radius)**2
                  and exact[2].norm_squared < Fraction(patch.right_radius)**2
                  and Fraction(lo) <= exact[1].real < Fraction(hi)
                  and Fraction(bottom) <= exact[1].imag < Fraction(top))
        if not inside:
            return outside(True)
        q = tuple(complex(float(z.real),float(z.imag)) for z in exact)
        if any(z == 0 or not math.isfinite(abs(z)) for z in q):
            raise ArithmeticError("exact interior patch coordinates are unrepresentable")
    elif not patch.contains(q):
        return outside()
    logdet = logs[e,c].real+logs[e,a].real+logs[a,c].real
    local = tuple(float(logdet-2*(logs[i,a].real+logs[e,c].real if i == c
                                 else logs[i,c].real+logs[e,a].real)) for i in ordering)
    row = (max(map(abs,q)),max(map(abs,q)),ordering,q,local)
    error = coordinate_fidelity_error(d,"comb",row)
    if not math.isfinite(error) or error > tolerance:
        raise ArithmeticError("fixed patch coordinates fail the labelled-pair identity")
    log_j_target = math.log(abs(q[1]))+2*math.log(abs(q[2]))
    log_inverse = log_j_target-math.fsum(local)
    if not math.isfinite(log_inverse):
        raise ArithmeticError("nonfinite source-to-patch Jacobian")
    return PatchGeometry(True,q,ordering,log_inverse,density,float(error),ambiguous)


def counterterm_log_importance(geometry, log_physical_patch_density):
    """Convert a physical d²a d²x d²b density to the original importance value.

    The caller's density already includes the patch-coordinate measure powers,
    matter, block and any desired spectral weights. Do not add a second CFT
    covariance factor. A zero/outside density is represented by -infinity.
    """
    if not geometry.inside:
        return complex(-math.inf,0)
    value = complex(log_physical_patch_density)
    if value.real == -math.inf and math.isfinite(value.imag):
        return value
    if not math.isfinite(value.real) or not math.isfinite(value.imag):
        raise ValueError("finite log density or negative-infinity zero required")
    return value-2*geometry.log_source_inverse_jacobian-geometry.source_log_density


def counterterm_importance_value(geometry, log_physical_patch_density):
    """Exponentiate the converted log value, refusing finite overflow/underflow."""
    value = counterterm_log_importance(geometry,log_physical_patch_density)
    if value.real == -math.inf:
        return 0j
    try:
        with np.errstate(over="raise",invalid="raise",under="ignore"):
            answer = complex(np.exp(value))
    except (OverflowError,FloatingPointError) as exc:
        raise ArithmeticError("unrepresentable importance counterterm") from exc
    if answer == 0 or not math.isfinite(answer.real) or not math.isfinite(answer.imag):
        raise ArithmeticError("unrepresentable importance counterterm; use log form")
    return answer


__all__ = ["PatchGeometry", "patch_geometry", "counterterm_log_importance",
           "counterterm_importance_value"]
