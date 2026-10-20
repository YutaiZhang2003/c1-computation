"""Collision-region maps, physical weights, and requests to the Liouville evaluator."""
from __future__ import annotations

#!/usr/bin/env python3
# Collision maps
# One collision-resolved RQMC replicate for the genus-one 1->2 amplitude.




from c1_paths import PACKAGE_ROOT, data_root, package_path, module_path

import argparse
import json
import math
from pathlib import Path

import numpy as np

from genus1_three_point_channel_atlas import (
    LiouvilleTorusThreePointAtlas,
    nearest_torus_displacement,
    reduced_worldsheet_integrand_three_point_patched,
)
from genus1_three_point_collision import (
    CollisionRadii,
    PAIR_ROWS,
    comb_corner_integrated_radially,
    comb_pair_face_integrated_over_shape,
    comb_pair_projection,
    comb_scale_face_integrated_over_scale,
    comb_scale_projection,
    nearest_pair,
    pair_projection,
    pair_projection_disc,
    reconstruct_points,
    reduced_channel_integrand,
    triple_shape_radius,
    pair_forest_remainder,
    triple_forest_remainder,
)
from genus1_moduli_sampling import TAIL_SEED_OFFSET, sobol_points, tail_height_and_jacobian
from genus1_three_point_worldsheet import LiouvilleTorusThreePointNecklace
from genus1_two_point_worldsheet import MomentumRule


STRATA = (
    "regular",
    "pair_face_remainder",
    "pair_face_analytic",
    "triple_forest_remainder",
    "triple_pair_face",
    "triple_scale_face",
    "triple_corner",
)


def stratum_seed(seed: int, replicate_index: int, region: str, stratum: str) -> int:
    """Use the full-design index even when a worker handles just one stratum."""
    region_index = ("bulk", "tail").index(region)
    return (int(seed) + 1_000_000 * int(replicate_index)
            + 100_000 * region_index + 10_000 * STRATA.index(stratum))








def _tau(point: np.ndarray, region: str, cutoff: float, tail_exponent: float) -> tuple[complex, float]:
    if region == "bulk":
        tau1 = float(point[0]) - 0.5
        lower = math.sqrt(1.0 - tau1 * tau1)
        tau2 = lower + float(point[1]) * (float(cutoff) - lower)
        return complex(tau1, tau2), float(cutoff) - lower
    if region == "tail":
        tau2, jacobian = tail_height_and_jacobian(
            float(point[0]),
            tail_start=float(cutoff),
            proposal_exponent=float(tail_exponent),
        )
        return complex(float(point[1]) - 0.5, tau2), jacobian
    raise ValueError(f"unknown modular region {region}")


def _torus_point(first: float, second: float, tau: complex) -> complex:
    return complex(2.0 * math.pi * (float(first) + float(second) * complex(tau)))


def evaluate_regular(
    atlas: LiouvilleTorusThreePointAtlas,
    points: np.ndarray,
    *,
    region: str,
    radii: CollisionRadii,
    cutoff: float,
    tail_exponent: float,
) -> np.ndarray:
    values = np.zeros(len(points), dtype=np.complex128)
    for index, point in enumerate(points):
        tau, tau_jacobian = _tau(point, region, cutoff, tail_exponent)
        w1 = _torus_point(point[2], point[3], tau)
        w2 = _torus_point(point[4], point[5], tau)
        _, _, distance = nearest_pair(w1, w2, tau)
        if distance < radii.pair:
            continue
        reduced, _ = reduced_worldsheet_integrand_three_point_patched(
            atlas, w1, w2, tau, record_diagnostics=True
        )
        values[index] = (
            tau_jacobian
            * (2.0 * math.pi) ** 4
            * tau.imag**2
            * reduced
        )
    return values


def evaluate_pair_face_remainder(
    atlas: LiouvilleTorusThreePointAtlas,
    points: np.ndarray,
    *,
    region: str,
    radii: CollisionRadii,
    cutoff: float,
    tail_exponent: float,
) -> np.ndarray:
    values = np.zeros(len(points), dtype=np.complex128)
    for index, point in enumerate(points):
        tau, tau_jacobian = _tau(point, region, cutoff, tail_exponent)
        delta_total = _torus_point(point[2], point[3], tau)
        if abs(nearest_torus_displacement(delta_total, tau)) < radii.triple:
            continue
        radius = radii.pair * float(point[4])
        delta_pair = radius * np.exp(2.0j * math.pi * float(point[5]))
        total = 0.0j
        for pair in PAIR_ROWS:
            total += pair_forest_remainder(atlas,pair,delta_pair,delta_total,tau)
        position_jacobian = (
            (2.0 * math.pi) ** 2
            * tau.imag
            * 2.0
            * math.pi
            * radii.pair
            * radius
        )
        values[index] = tau_jacobian * position_jacobian * total
    return values


def evaluate_pair_face_analytic(
    atlas: LiouvilleTorusThreePointAtlas,
    points: np.ndarray,
    *,
    region: str,
    radii: CollisionRadii,
    cutoff: float,
    tail_exponent: float,
) -> np.ndarray:
    values = np.zeros(len(points), dtype=np.complex128)
    for index, point in enumerate(points):
        tau, tau_jacobian = _tau(point, region, cutoff, tail_exponent)
        delta_total = _torus_point(point[2], point[3], tau)
        if abs(nearest_torus_displacement(delta_total, tau)) < radii.triple:
            continue
        total = sum(
            (
                pair_projection_disc(
                    atlas, pair, delta_total, tau, radii.pair
                )
                for pair in PAIR_ROWS
            ),
            0.0j,
        )
        values[index] = (
            tau_jacobian
            * (2.0 * math.pi) ** 2
            * tau.imag
            * total
        )
    return values


def _shape(
    point_theta: float,
    point_radius: float,
) -> tuple[complex, float, float, float]:
    theta = 2.0 * math.pi * float(point_theta)
    upper = triple_shape_radius(theta)
    radius = upper * math.sqrt(float(point_radius))
    # s ds/du for s=s_max*sqrt(u).
    radial_measure = 0.5 * upper * upper
    return radius * np.exp(1.0j * theta), upper, theta, radial_measure


def evaluate_triple_forest_remainder(
    atlas: LiouvilleTorusThreePointAtlas,
    points: np.ndarray,
    *,
    region: str,
    radii: CollisionRadii,
    cutoff: float,
    tail_exponent: float,
) -> np.ndarray:
    values = np.zeros(len(points), dtype=np.complex128)
    for index, point in enumerate(points):
        tau, tau_jacobian = _tau(point, region, cutoff, tail_exponent)
        shape, shape_upper, _, shape_measure = _shape(point[2], point[3])
        shape_radius = abs(shape)
        if shape_radius == 0.0:
            continue
        scale_upper = min(radii.triple, radii.pair / shape_radius)
        scale = scale_upper * math.sqrt(float(point[4]))
        if scale == 0.0:
            continue
        phase = 2.0 * math.pi * float(point[5])
        delta_total = scale * np.exp(1.0j * phase)
        delta_pair = shape * delta_total
        total = 0.0j
        for pair in PAIR_ROWS:
            total += triple_forest_remainder(atlas,pair,delta_pair,delta_total,tau)
        # R^3 dR/du for R=R_max*sqrt(u).
        scale_measure = 0.5 * scale_upper * scale_upper * scale * scale
        position_jacobian = (
            (2.0 * math.pi) ** 2 * shape_measure * scale_measure
        )
        values[index] = tau_jacobian * position_jacobian * total
    return values


def evaluate_triple_pair_face(
    atlas: LiouvilleTorusThreePointAtlas,
    points: np.ndarray,
    *,
    region: str,
    radii: CollisionRadii,
    cutoff: float,
    tail_exponent: float,
) -> np.ndarray:
    values = np.zeros(len(points), dtype=np.complex128)
    for index, point in enumerate(points):
        tau, tau_jacobian = _tau(point, region, cutoff, tail_exponent)
        scale = radii.triple * math.sqrt(float(point[2]))
        if scale == 0.0:
            continue
        phase = 2.0 * math.pi * float(point[3])
        theta = 2.0 * math.pi * float(point[4])
        shape_upper = min(
            triple_shape_radius(theta),
            radii.pair / scale,
        )
        total = sum(
            (
                comb_pair_face_integrated_over_shape(
                    atlas, pair, scale, phase, shape_upper, tau
                )
                for pair in PAIR_ROWS
            ),
            0.0j,
        )
        scale_measure = 0.5 * radii.triple**2 * scale**2
        position_jacobian = (2.0 * math.pi) ** 2 * scale_measure
        values[index] = tau_jacobian * position_jacobian * total
    return values


def evaluate_triple_scale_face(
    atlas: LiouvilleTorusThreePointAtlas,
    points: np.ndarray,
    *,
    region: str,
    radii: CollisionRadii,
    cutoff: float,
    tail_exponent: float,
) -> np.ndarray:
    values = np.zeros(len(points), dtype=np.complex128)
    for index, point in enumerate(points):
        tau, tau_jacobian = _tau(point, region, cutoff, tail_exponent)
        shape, shape_upper, _, shape_measure = _shape(point[2], point[3])
        shape_radius = abs(shape)
        if shape_radius == 0.0:
            continue
        scale_upper = min(radii.triple, radii.pair / shape_radius)
        total = sum(
            (
                comb_scale_face_integrated_over_scale(
                    atlas, pair, shape, scale_upper, tau
                )
                for pair in PAIR_ROWS
            ),
            0.0j,
        )
        # Both angular variables have been accounted for: theta by the QMC
        # Jacobian and the overall comb phase analytically.
        position_jacobian = (2.0 * math.pi) ** 2 * shape_measure
        values[index] = tau_jacobian * position_jacobian * total
    return values


def evaluate_triple_corner(
    atlas: LiouvilleTorusThreePointAtlas,
    points: np.ndarray,
    *,
    region: str,
    radii: CollisionRadii,
    cutoff: float,
    tail_exponent: float,
) -> np.ndarray:
    values = np.zeros(len(points), dtype=np.complex128)
    for index, point in enumerate(points):
        tau, tau_jacobian = _tau(point, region, cutoff, tail_exponent)
        theta = 2.0 * math.pi * float(point[2])
        total = sum(
            (
                comb_corner_integrated_radially(
                    atlas, pair, theta, radii, tau
                )
                for pair in PAIR_ROWS
            ),
            0.0j,
        )
        values[index] = tau_jacobian * 2.0 * math.pi * total
    return values


EVALUATORS = {
    "regular": (6, evaluate_regular),
    "pair_face_remainder": (6, evaluate_pair_face_remainder),
    "pair_face_analytic": (4, evaluate_pair_face_analytic),
    "triple_forest_remainder": (6, evaluate_triple_forest_remainder),
    "triple_pair_face": (5, evaluate_triple_pair_face),
    "triple_scale_face": (4, evaluate_triple_scale_face),
    "triple_corner": (3, evaluate_triple_corner),
}














# Sampling requests
# Bank-backed wiring for all seven collision strata, with exact sampling maps.
# 
# The backend must supply verified reduced CFT values, including the common
# boson/ghost factors exactly once. This module never constructs CFT banks,
# changes native sewing variables, or substitutes missing values with zero.
# The two scale/corner terms use the existing height-integrated prescription.



from dataclasses import dataclass
import math

import numpy as np

from genus1_three_point_channel_atlas import nearest_torus_displacement
from genus1_three_point_collision import PAIR_ROWS, CollisionRadii, nearest_pair, triple_shape_radius
from genus1_moduli_sampling import sobol_points
from genus1_amplitude_report import point_record


@dataclass(frozen=True)
class Request:
    kind: str
    geometry: dict
    multiplicity: int = 1


@dataclass(frozen=True)
class SamplePlan:
    jacobian: float
    requests: tuple
    exact_zero_reason: str | None = None


class MissingCFTBank(ValueError):
    """A request has no verified compatible input; it is not a zero value."""


def sample_plan(atlas, stratum, point, *, region, radii, cutoff=8., tail_exponent=2.,
                defer_full_routing=False):
    if stratum not in STRATA or region not in ('bulk', 'tail') or not isinstance(radii, CollisionRadii):
        raise ValueError('explicit stratum, modular region and collision radii required')
    p = np.asarray(point, dtype=float)
    if p.shape != (EVALUATORS[stratum][0],) or not np.all(np.isfinite(p)) or np.any(p < 0) or np.any(p >= 1):
        raise ValueError('sample is not in its original Sobol unit cube')
    CollisionRadii(radii.pair, radii.triple)
    # Preserve the exact coordinates and unused height coordinate of the
    # previously sampled, height-integrated collar estimator.
    if stratum in ('triple_scale_face', 'triple_corner'):
        tau1 = float(p[0 if region == 'bulk' else 1])-.5
        if stratum == 'triple_scale_face':
            shape, _, theta, measure = _shape(p[2], p[3])
            jac = (2*math.pi)**2*measure
        else:
            shape = None; theta = 2*math.pi*float(p[2]); jac = 2*math.pi
        args = dict(term=stratum, region=region, tau1=tau1, shape=shape, theta=theta,
                    pair_radius=radii.pair, triple_radius=radii.triple, cutoff=cutoff)
        requests = tuple(Request('height_integrated_collar', dict(args, graph=i), m)
                         for i, m in ((0, 1), (1, 2)))
        return SamplePlan(jac, requests)
    tau, tjac = _tau(p, region, cutoff, tail_exponent)
    if stratum == 'regular':
        w1 = _torus_point(p[2], p[3], tau); w2 = _torus_point(p[4], p[5], tau)
        if nearest_pair(w1, w2, tau)[2] < radii.pair:
            return SamplePlan(0., (), 'outside_regular_stratum_inside_pair_collar')
        if defer_full_routing:
            # This is a physical CFT request, not a claimed native chart.
            args = dict(w1=w1,w2=w2,tau=tau,channel=None,pair=None)
        else:
            choice = atlas.choose_channel(w1, w2, tau)
            atlas.sewing_chart(w1, w2, tau, choice.channel, choice.pair).require()
            args = dict(w1=w1, w2=w2, tau=tau, channel=choice.channel, pair=choice.pair)
        return SamplePlan(tjac*(2*math.pi)**4*tau.imag**2, (Request('full_reduced', args),))
    if stratum in ('pair_face_remainder', 'pair_face_analytic'):
        total = _torus_point(p[2], p[3], tau)
        if abs(nearest_torus_displacement(total, tau)) < radii.triple:
            return SamplePlan(0., (), 'outside_isolated_pair_stratum_inside_triple_collar')
        if stratum == 'pair_face_remainder':
            radius = radii.pair*float(p[4]); delta = radius*np.exp(2j*math.pi*float(p[5]))
            args = dict(delta_pair=delta, delta_total=total, tau=tau, radii=radii)
            jac = tjac*((2*math.pi)**2*tau.imag*2*math.pi*radii.pair*radius)
            kind = 'pair_remainder'
        else:
            args = dict(delta_total=total, tau=tau, radius=radii.pair, integrated=True)
            jac = tjac*(2*math.pi)**2*tau.imag; kind = 'pair_projection'
        return SamplePlan(jac, tuple(Request(kind, dict(args, pair=pair)) for pair in PAIR_ROWS))
    if stratum == 'triple_forest_remainder':
        shape, _, _, measure = _shape(p[2], p[3])
        if abs(shape) == 0:
            return SamplePlan(0., (), 'zero_measure_shape_endpoint')
        upper = min(radii.triple, radii.pair/abs(shape)); scale = upper*math.sqrt(float(p[4]))
        if scale == 0:
            return SamplePlan(0., (), 'zero_measure_scale_endpoint')
        args = dict(term='forest_remainder', tau=tau, shape=shape, scale=scale,
                    phase=2*math.pi*float(p[5]), radii=radii)
        scale_measure = .5*upper*upper*scale*scale
        jac = tjac*((2*math.pi)**2*measure*scale_measure)
    else:
        scale = radii.triple*math.sqrt(float(p[2]))
        if scale == 0:
            return SamplePlan(0., (), 'zero_measure_scale_endpoint')
        theta = 2*math.pi*float(p[4])
        args = dict(term='pair_face', tau=tau, scale=scale, phase=2*math.pi*float(p[3]),
                    shape_upper=min(triple_shape_radius(theta), radii.pair/scale))
        jac = tjac*((2*math.pi)**2*(.5*radii.triple**2*scale**2))
    return SamplePlan(jac, tuple(Request('comb_collar', dict(args, pair=pair)) for pair in PAIR_ROWS))


def evaluate_sample(atlas, backend, stratum, point, *, point_id, region, radii,
                    cutoff=8., tail_exponent=2., sample_count=8192,importance_weight=1.):
    """Return the normalized contribution and per-graph weighted diagnostics.

backend.evaluate(Request) returns value, orders, safety, source_sha256 and
paired_checks (actual complex coarse/refined pairs, not local percentages).
Safety is mandatory; a missing refinement audit stays missing in the final
budget. The backend cannot inherit a pilot's accuracy classification.
"""
    if type(sample_count) is not int or sample_count <= 0:
        raise ValueError('explicit total samples per independent scramble required')
    if not math.isfinite(importance_weight) or importance_weight<=0:
        raise ValueError('finite positive original-measure importance weight required')
    plan = sample_plan(atlas, stratum, point, region=region, radii=radii,
                       cutoff=cutoff, tail_exponent=tail_exponent,
                       defer_full_routing=getattr(backend,'defer_full_routing',False))
    jacobian=plan.jacobian*importance_weight
    if not plan.requests:
        if plan.exact_zero_reason is None:
            raise ValueError('missing CFT requests are not zero contributions')
        return dict(point_id=point_id, amplitude_contribution=0., reduced_weighted_value=0j,
                    records=[], exact_zero_reason=plan.exact_zero_reason, sample_covered=True)
    records = []; reduced = []
    for i, request in enumerate(plan.requests):
        result = backend.evaluate(request)
        if result is None:
            raise MissingCFTBank(request)
        value = complex(result['value'])
        if not math.isfinite(value.real) or not math.isfinite(value.imag):
            raise FloatingPointError('nonfinite CFT result')
        checks = {category: {label: (complex(pair[0]).imag, complex(pair[1]).imag)
                  for label, pair in comparisons.items()}
                  for category, comparisons in result['paired_checks'].items()}
        record = point_record(point_id=f'{point_id}:graph{i}', value=value.imag,
            normalized_weight=-jacobian*request.multiplicity/(4*math.pi*sample_count),
            orders=result['orders'], safety=result['safety'], paired_checks=checks,
            source_sha256=result['source_sha256'])
        if 'routing' in result:
            record['cft_routing'] = result['routing']
        log_bound=result.get('absolute_arithmetic_log_bound')
        if log_bound is not None:
            weight=abs(jacobian*request.multiplicity/(4*math.pi*sample_count))
            weighted_log=float(log_bound)+math.log(weight)
            # An explicit upward floor records subnormal bounds, never zero.
            record['absolute_roundoff_enclosure']=math.exp(max(weighted_log,-700.))
        else: record['absolute_roundoff_enclosure']=0.
        records.append(record)
        reduced.append(jacobian*request.multiplicity*value)
    return dict(point_id=point_id, amplitude_contribution=math.fsum(r['contribution'] for r in records),
        reduced_weighted_value=complex(math.fsum(z.real for z in reduced), math.fsum(z.imag for z in reduced)),
        records=records, exact_zero_reason=None, sample_covered=True)


def original_chunk_points(stratum, *, region, replicate, start, stop, seed=17051301):
    """Slice the original8192-point net; never seed anew for a small shard."""
    if (stratum not in STRATA or region not in ('bulk', 'tail') or type(replicate) is not int
            or not 0 <= replicate < 16 or type(start) is not int or type(stop) is not int
            or not 0 <= start < stop <= 8192):
        raise ValueError('invalid original sampling slice')
    actual_seed = stratum_seed(seed, replicate, region, stratum)
    return actual_seed, sobol_points(EVALUATORS[stratum][0], 13, actual_seed)[start:stop]
