#!/usr/bin/env python3
"""Collision-resolved integrands for the torus three-point amplitude.

The two puncture integrations are partitioned into a regular region, three
isolated pair collars, and three nested triple-collision sectors.  In a pair
collar the leading pair-OPE radial power is integrated analytically.  In a
triple sector a two-boundary forest subtraction removes both the pair and
overall-scale singularities before numerical sampling.  The corresponding
face and corner projections are integrated in the singular radii exactly.

All coordinates below are physical flat-torus coordinates with periods
``2*pi`` and ``2*pi*tau``.  Consequently the local Jacobians are ordinary
Euclidean ``d^2 z`` Jacobians; no hidden fractional-coordinate factor is
used in this module.
"""

from __future__ import annotations

from c1_paths import PACKAGE_ROOT, data_root, package_path, module_path

import cmath
import math
from dataclasses import dataclass

import numpy as np

try:
    from genus1_three_point_channel_atlas import (
        LiouvilleTorusThreePointAtlas,
        canonical_loop_displacement,
        nearest_torus_displacement,
    )
    from genus1_three_point_worldsheet import (
        dedekind_eta_oscillator_abs_squared,
        ordered_necklace_data,
        torus_prime_form_log_norm,
    )
except ImportError:  # pragma: no cover - package-style execution
    from genus1_three_point_channel_atlas import (
        LiouvilleTorusThreePointAtlas,
        canonical_loop_displacement,
        nearest_torus_displacement,
    )
    from genus1_three_point_worldsheet import (
        dedekind_eta_oscillator_abs_squared,
        ordered_necklace_data,
        torus_prime_form_log_norm,
    )


PAIR_ROWS = ((0, 1), (0, 2), (1, 2))


@dataclass(frozen=True)
class CollisionRadii:
    """Physical radii defining the disjoint collision partition."""

    pair: float
    triple: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.pair) or self.pair <= 0.0:
            raise ValueError("pair radius must be positive")
        if not math.isfinite(self.triple) or self.triple < 2.0 * self.pair:
            raise ValueError("triple radius must be at least twice the pair radius")
        if self.triple >= math.pi:
            raise ValueError("triple radius must be below the flat-torus injectivity radius")


def pair_spectator(pair: tuple[int, int]) -> int:
    if tuple(pair) not in PAIR_ROWS:
        raise ValueError(f"unknown puncture pair {pair}")
    return next(index for index in range(3) if index not in pair)


def koba_exponent(
    atlas: LiouvilleTorusThreePointAtlas,
    first: int,
    second: int,
) -> complex:
    """Return the exponent of ``|E(w_i-w_j)|`` for the symmetric slice."""

    edge = frozenset((int(first), int(second)))
    if edge == frozenset((0, 1)):
        return complex(-atlas.necklace.omega_out * atlas.necklace.omega_out)
    if edge in (frozenset((0, 2)), frozenset((1, 2))):
        return complex(atlas.necklace.omega_in * atlas.necklace.omega_out)
    raise ValueError(f"unknown puncture edge {first}-{second}")


def reconstruct_points(
    pair: tuple[int, int],
    delta_pair: complex,
    delta_total: complex,
) -> tuple[complex, complex]:
    """Recover ``(w1,w2)`` from one comb tree.

    ``delta_pair = w_first-w_second`` and
    ``delta_total = w_spectator-w_second``.  Puncture 2 is fixed at zero.
    """

    first, second = pair
    spectator = pair_spectator(pair)
    points: list[complex | None] = [None, None, 0.0j]
    if second == 2:
        points[first] = complex(delta_pair)
        points[spectator] = complex(delta_total)
    elif spectator == 2:
        points[second] = -complex(delta_total)
        points[first] = complex(delta_pair) - complex(delta_total)
    else:  # Unreachable for PAIR_ROWS, retained as a closed-form check.
        points[second] = -complex(delta_pair)
        points[spectator] = complex(delta_total) - complex(delta_pair)
    if points[0] is None or points[1] is None:
        raise AssertionError("failed to reconstruct the moving punctures")
    return complex(points[0]), complex(points[1])


def nearest_pair(
    w1: complex,
    w2: complex,
    tau: complex,
) -> tuple[tuple[int, int], complex, float]:
    """Return the nearest pair, its local displacement, and its distance."""

    points = (complex(w1), complex(w2), 0.0j)
    rows = []
    for pair in PAIR_ROWS:
        delta = nearest_torus_displacement(points[pair[0]] - points[pair[1]], tau)
        rows.append((abs(delta), pair, delta))
    distance, pair, delta = min(rows, key=lambda row: (row[0], row[1]))
    return pair, complex(delta), float(distance)


def triple_shape_radius(theta: float) -> float:
    """Upper radial boundary of ``|u|<=1, Re(u)<=1/2`` at fixed angle."""

    cosine = math.cos(float(theta))
    if cosine <= 0.5:
        return 1.0
    return 0.5 / cosine


def _common_factor(atlas: LiouvilleTorusThreePointAtlas, tau: complex) -> complex:
    return complex(
        dedekind_eta_oscillator_abs_squared(tau) / math.sqrt(complex(tau).imag)
    )


def _exact_koba(
    atlas: LiouvilleTorusThreePointAtlas,
    w1: complex,
    w2: complex,
    tau: complex,
) -> complex:
    logs = (
        torus_prime_form_log_norm(w1 - w2, tau),
        torus_prime_form_log_norm(w1, tau),
        torus_prime_form_log_norm(w2, tau),
    )
    exponents = (
        koba_exponent(atlas, 0, 1),
        koba_exponent(atlas, 0, 2),
        koba_exponent(atlas, 1, 2),
    )
    return complex(cmath.exp(sum(exponent * log for exponent, log in zip(exponents, logs))))


def reduced_channel_integrand(
    atlas: LiouvilleTorusThreePointAtlas,
    w1: complex,
    w2: complex,
    tau: complex,
    *,
    channel: str,
    pair: tuple[int, int] | None = None,
) -> complex:
    """Evaluate one atlas channel with the common timelike/eta factors."""

    if channel == "necklace":
        _, logs = ordered_necklace_data(w1, w2, tau)
        liouville = atlas.necklace.correlator_from_logs(
            logs,
            record_diagnostics=False,
            cancel_eta_vacuum=True,
        )
    elif channel == "pair_ope":
        if pair is None:
            raise ValueError("pair_ope requires a pair")
        liouville = atlas.correlator_ope(
            w1, w2, tau, pair, record_diagnostics=False
        )
    elif channel == "comb_ope":
        if pair is None:
            raise ValueError("comb_ope requires a pair")
        liouville = atlas.correlator_comb_ope(
            w1, w2, tau, pair, record_diagnostics=False
        )
    else:
        raise ValueError(f"unknown channel {channel}")
    return _common_factor(atlas, tau) * _exact_koba(atlas, w1, w2, tau) * liouville


def _pair_loop_arrays(
    atlas: LiouvilleTorusThreePointAtlas,
    pair: tuple[int, int],
    delta_total: complex,
    tau: complex,
) -> tuple[np.ndarray, np.ndarray]:
    """Return pair-projection coefficients and pair radial exponents."""

    first, second = pair
    spectator = pair_spectator(pair)
    fused_to_spectator = canonical_loop_displacement(-complex(delta_total), tau)
    logs = (
        1.0j * fused_to_spectator,
        1.0j * (2.0 * math.pi * complex(tau) - fused_to_spectator),
    )
    if any(z.real>=0 for z in logs):
        from genus1_sewing_domains import SewingDomainError
        raise SewingDomainError('pair projection requires contracting loop coordinates')
    sewing_q = tuple(cmath.exp(value) for value in logs)
    high_loop_edge = 1 + int(abs(sewing_q[1]) > abs(sewing_q[0]))
    key = atlas._bank_key(pair, high_loop_edge)
    if key not in atlas._banks:
        atlas._banks[key] = atlas._build_bank(key[0], high_loop_edge)
        atlas._save_cache_if_requested()
    bank = atlas._banks[key]
    orders = list(bank.orders)
    if atlas.evaluation_order_cap is not None:
        orders[0] = min(orders[0], atlas.evaluation_order_cap)
    descendants = np.einsum(
        "tjk,j,k->t",
        bank.coefficients[:, 0, : orders[1] + 1, : orders[2] + 1],
        sewing_q[0] ** np.arange(orders[1] + 1),
        sewing_q[1] ** np.arange(orders[2] + 1),
        optimize=True,
    )
    h_ope, h_first, h_second = bank.internal_weights
    loop_norm = np.exp(
        2.0 * (h_first - 1.0) * logs[0].real
        + 2.0 * (h_second - 1.0) * logs[1].real
    ) * np.abs(descendants) ** 2
    kappa_pair = koba_exponent(atlas, first, second)
    beta = (
        2.0 * h_ope
        - 2.0 * (atlas.external_weights[first] + atlas.external_weights[second])
        + kappa_pair
        + 2.0
    )
    coefficient = bank.weighted_structure_constants * loop_norm
    return np.asarray(coefficient, dtype=np.complex128), np.asarray(beta, dtype=np.complex128)


def _pair_spectral_logs(delta_total,tau,radius):
    loop=canonical_loop_displacement(-complex(delta_total),tau)
    return (complex(math.log(radius)),1j*loop,1j*(2*math.pi*tau-loop))


def _scale_spectral_logs(scale,shape_radius,tau):
    return (complex(math.log(shape_radius)),complex(math.log(scale)),2j*math.pi*tau)


def pair_forest_remainder(atlas,pair,delta_pair,delta_total,tau):
    """Integrate exact-minus-projection with the SAME spectral nodes."""
    w1,w2=reconstruct_points(pair,delta_pair,delta_total)
    chart=atlas.sewing_chart(w1,w2,tau,'pair_ope',pair).require()
    controller=getattr(atlas,'spectral_controller',None)
    if controller is not None:
        projection=_pair_spectral_logs(delta_total,tau,abs(delta_pair))
        logs=tuple(complex(max(a.real,b.real)) for a,b in zip(chart.logs,projection))
        return controller.integrate_atlas(atlas,logs,lambda shadow:
            pair_forest_remainder(shadow,pair,delta_pair,delta_total,tau),label='paired_pair_remainder')
    return (reduced_channel_integrand(atlas,w1,w2,tau,channel='pair_ope',pair=pair)
            -pair_projection(atlas,pair,delta_pair,delta_total,tau))


def triple_forest_remainder(atlas,pair,delta_pair,delta_total,tau):
    """Refine the entire forest residual jointly, not four separate integrals."""
    w1,w2=reconstruct_points(pair,delta_pair,delta_total)
    chart=atlas.sewing_chart(w1,w2,tau,'comb_ope',pair).require()
    scale=abs(delta_total);shape=delta_pair/delta_total
    controller=getattr(atlas,'spectral_controller',None)
    if controller is not None:
        outer=abs(cmath.exp(-1j*delta_total)-1)
        rows=(chart.logs,_scale_spectral_logs(scale,abs(shape),tau),
              _scale_spectral_logs(outer,abs(delta_pair)/outer,tau))
        logs=tuple(complex(max(row[e].real for row in rows)) for e in range(3))
        return controller.integrate_atlas(atlas,logs,lambda shadow:
            triple_forest_remainder(shadow,pair,delta_pair,delta_total,tau),label='paired_triple_remainder')
    return (reduced_channel_integrand(atlas,w1,w2,tau,channel='comb_ope',pair=pair)
            -comb_pair_projection(atlas,pair,delta_pair,delta_total,tau)
            -comb_scale_projection(atlas,pair,scale,shape,tau)
            +comb_scale_projection(atlas,pair,scale,shape,tau,include_shape_edge=False))


def pair_projection(
    atlas: LiouvilleTorusThreePointAtlas,
    pair: tuple[int, int],
    delta_pair: complex,
    delta_total: complex,
    tau: complex,
) -> complex:
    """Leading pair-collision projection at fixed fused-puncture position."""

    controller=getattr(atlas,'spectral_controller',None)
    if controller is not None:
        return controller.integrate_atlas(atlas,_pair_spectral_logs(delta_total,tau,abs(delta_pair)),
            lambda shadow:pair_projection(shadow,pair,delta_pair,delta_total,tau),label='pair_projection')

    radius = abs(complex(delta_pair))
    if radius <= 0.0:
        raise ValueError("pair projection is singular at zero radius")
    first, second = pair
    spectator = pair_spectator(pair)
    coefficients, beta = _pair_loop_arrays(atlas, pair, delta_total, tau)
    spectator_exponent = (
        koba_exponent(atlas, first, spectator)
        + koba_exponent(atlas, second, spectator)
    )
    spectator_factor = cmath.exp(
        spectator_exponent * torus_prime_form_log_norm(delta_total, tau)
    )
    radial = np.exp((beta - 2.0) * math.log(radius))
    return complex(
        _common_factor(atlas, tau)
        * spectator_factor
        * np.dot(coefficients, radial)
    )


def pair_projection_disc(
    atlas: LiouvilleTorusThreePointAtlas,
    pair: tuple[int, int],
    delta_total: complex,
    tau: complex,
    radius: float,
) -> complex:
    """Integrate the leading pair projection over ``|delta_pair|<radius``."""

    controller=getattr(atlas,'spectral_controller',None)
    if controller is not None:
        return controller.integrate_atlas(atlas,_pair_spectral_logs(delta_total,tau,radius),
            lambda shadow:pair_projection_disc(shadow,pair,delta_total,tau,radius),label='pair_disc')

    radius = float(radius)
    if radius <= 0.0:
        raise ValueError("pair radius must be positive")
    first, second = pair
    spectator = pair_spectator(pair)
    coefficients, beta = _pair_loop_arrays(atlas, pair, delta_total, tau)
    spectator_exponent = (
        koba_exponent(atlas, first, spectator)
        + koba_exponent(atlas, second, spectator)
    )
    spectator_factor = cmath.exp(
        spectator_exponent * torus_prime_form_log_norm(delta_total, tau)
    )
    radial = 2.0 * math.pi * np.exp(beta * math.log(radius)) / beta
    return complex(
        _common_factor(atlas, tau)
        * spectator_factor
        * np.dot(coefficients, radial)
    )


def _comb_bank(
    atlas: LiouvilleTorusThreePointAtlas,
    pair: tuple[int, int],
):
    external = (pair[0], pair[1], pair_spectator(pair))
    key = atlas._comb_bank_key(external)
    if key not in atlas._comb_banks:
        atlas._comb_banks[key] = atlas._build_comb_bank(key)
        atlas._save_cache_if_requested()
    return atlas._comb_banks[key]


def _comb_corner_arrays(
    atlas: LiouvilleTorusThreePointAtlas,
    pair: tuple[int, int],
    tau: complex,
    *,
    shape: complex = 0j,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return scale-face coefficients (corner at shape=0) and radial powers.

    At fixed shape delta_pair/delta_total, the inner sewing coordinate tends
    to shape, not zero, as the overall scale vanishes. Its descendants must
    survive the scale projection; only the outer OPE level is set to zero.
    """

    bank = _comb_bank(atlas, pair)
    orders = list(bank.orders)
    if abs(shape)>=1:
        from genus1_sewing_domains import SewingDomainError
        raise SewingDomainError('scale-face shape Taylor series requires |shape|<1')
    if atlas.evaluation_order_cap is not None:
        orders[0] = min(orders[0], atlas.evaluation_order_cap)
        orders[1] = min(orders[1], atlas.evaluation_order_cap)
    q = cmath.exp(2.0j * math.pi * complex(tau))
    descendants = np.einsum(
        "tik,i,k->t",
        bank.coefficients[:, : orders[0] + 1, 0, : orders[2] + 1],
        complex(shape) ** np.arange(orders[0] + 1),
        q ** np.arange(orders[2] + 1),
        optimize=True,
    )
    h_pair, h_total, h_loop = bank.internal_weights
    loop_norm = np.exp(
        -4.0 * math.pi * complex(tau).imag * (h_loop - 1.0)
    ) * np.abs(descendants) ** 2
    first, second = pair
    spectator = pair_spectator(pair)
    kappa_pair = koba_exponent(atlas, first, second)
    kappa_shape = koba_exponent(atlas, first, spectator)
    kappa_other = koba_exponent(atlas, second, spectator)
    a = (
        2.0 * h_pair
        - 2.0 * (atlas.external_weights[first] + atlas.external_weights[second])
        + kappa_pair
    )
    gamma = (
        2.0 * h_total
        - 2.0 * sum(atlas.external_weights)
        + kappa_pair
        + kappa_shape
        + kappa_other
    )
    beta = a + 2.0
    lam = gamma + 4.0
    coefficient = (
        _common_factor(atlas, tau)
        * bank.weighted_structure_constants
        * loop_norm
    )
    return (
        np.asarray(coefficient, dtype=np.complex128),
        np.asarray(a, dtype=np.complex128),
        np.asarray(beta, dtype=np.complex128),
        np.asarray(gamma, dtype=np.complex128),
        np.asarray(lam, dtype=np.complex128),
    )


def comb_pair_projection(
    atlas: LiouvilleTorusThreePointAtlas,
    pair: tuple[int, int],
    delta_pair: complex,
    delta_total: complex,
    tau: complex,
) -> complex:
    """Pair-boundary projection of the comb block, retaining total OPE data."""

    from genus1_sewing_domains import comb_chart, SewingDomainError
    outer_chart=comb_chart(.01*complex(delta_total),complex(delta_total),2j*math.pi*tau)
    if outer_chart.margins[1]<=1e-12:
        raise SewingDomainError('comb pair projection outer sewing contours overlap')

    controller=getattr(atlas,'spectral_controller',None)
    if controller is not None:
        outer=abs(cmath.exp(-1j*delta_total)-1)
        return controller.integrate_atlas(atlas,_scale_spectral_logs(outer,abs(delta_pair)/outer,tau),
            lambda shadow:comb_pair_projection(shadow,pair,delta_pair,delta_total,tau),label='comb_pair_projection')

    pair_radius = abs(complex(delta_pair))
    if pair_radius <= 0.0:
        raise ValueError("comb pair projection is singular at zero pair radius")
    bank = _comb_bank(atlas, pair)
    orders = list(bank.orders)
    if atlas.evaluation_order_cap is not None:
        orders[0] = min(orders[0], atlas.evaluation_order_cap)
        orders[1] = min(orders[1], atlas.evaluation_order_cap)
    q = cmath.exp(2.0j * math.pi * complex(tau))
    descendants = np.einsum(
        "tjk,j,k->t",
        bank.coefficients[:, 0, : orders[1] + 1, : orders[2] + 1],
        (cmath.exp(-1.0j * complex(delta_total)) - 1.0) ** np.arange(orders[1] + 1),
        q ** np.arange(orders[2] + 1),
        optimize=True,
    )
    h_pair, h_total, h_loop = bank.internal_weights
    first, second = pair
    spectator = pair_spectator(pair)
    v_total = cmath.exp(-1.0j * complex(delta_total)) - 1.0
    if v_total == 0.0j:
        raise ValueError("comb pair projection requires nonzero total radius")
    sine = 2.0 * cmath.sin(complex(delta_total) / 2.0)
    second_flat = (
        0.5j * (h_pair - atlas.external_weights[spectator]) * complex(delta_total)
        - (h_pair + atlas.external_weights[spectator]) * cmath.log(sine)
    )
    other_norm = np.exp(
        2.0 * h_total * math.log(abs(v_total))
        - 4.0 * math.pi * complex(tau).imag * (h_loop - 1.0)
        + 2.0 * np.real(second_flat)
    ) * np.abs(descendants) ** 2
    a = (
        2.0 * h_pair
        - 2.0 * (atlas.external_weights[first] + atlas.external_weights[second])
        + koba_exponent(atlas, first, second)
    )
    spectator_exponent = (
        koba_exponent(atlas, first, spectator)
        + koba_exponent(atlas, second, spectator)
    )
    spectator_factor = cmath.exp(
        spectator_exponent * torus_prime_form_log_norm(delta_total, tau)
    )
    return complex(
        _common_factor(atlas, tau)
        * spectator_factor
        * np.dot(
            bank.weighted_structure_constants * other_norm,
            np.exp(a * math.log(pair_radius)),
        )
    )


def comb_scale_projection(
    atlas: LiouvilleTorusThreePointAtlas,
    pair: tuple[int, int],
    scale: float,
    shape: complex,
    tau: complex,
    *,
    include_shape_edge: bool = True,
) -> complex:
    """Overall triple-scale projection, optionally projected again at ``u=0``."""

    controller=getattr(atlas,'spectral_controller',None)
    if controller is not None:
        return controller.integrate_atlas(atlas,_scale_spectral_logs(scale,abs(shape),tau),
            lambda shadow:comb_scale_projection(shadow,pair,scale,shape,tau,
                include_shape_edge=include_shape_edge),label='comb_scale_projection')

    scale = float(scale)
    shape_radius = abs(complex(shape))
    if scale <= 0.0 or shape_radius <= 0.0:
        raise ValueError("comb scale projection requires nonzero scale and shape")
    coefficient, a, _, gamma, _ = _comb_corner_arrays(
        atlas, pair, tau, shape=shape if include_shape_edge else 0j)
    shape_factor: complex = 1.0 + 0.0j
    if include_shape_edge:
        shape_factor = cmath.exp(
            koba_exponent(atlas, pair[0], pair_spectator(pair))
            * math.log(abs(1.0 - complex(shape)))
        )
    return complex(
        shape_factor
        * np.dot(
            coefficient,
            np.exp(gamma * math.log(scale) + a * math.log(shape_radius)),
        )
    )


def comb_pair_face_integrated_over_shape(
    atlas: LiouvilleTorusThreePointAtlas,
    pair: tuple[int, int],
    scale: float,
    phase: float,
    shape_upper: float,
    tau: complex,
) -> complex:
    """Integrate ``S_pair-S_corner`` over the nested pair radius ``s``."""

    from genus1_sewing_domains import comb_chart, SewingDomainError
    outer_delta=scale*cmath.exp(1j*phase)
    if comb_chart(.01*outer_delta,outer_delta,2j*math.pi*tau).margins[1]<=1e-12:
        raise SewingDomainError('comb pair-face outer sewing contours overlap')

    controller=getattr(atlas,'spectral_controller',None)
    if controller is not None:
        outer=abs(cmath.exp(-1j*scale*cmath.exp(1j*phase))-1)
        logs=_scale_spectral_logs(max(scale,outer),shape_upper*max(1.,scale/outer),tau)
        return controller.integrate_atlas(atlas,logs,lambda shadow:
            comb_pair_face_integrated_over_shape(shadow,pair,scale,phase,shape_upper,tau),label='comb_pair_face')

    scale = float(scale)
    shape_upper = float(shape_upper)
    if scale <= 0.0 or shape_upper <= 0.0:
        raise ValueError("positive scale and shape upper bound are required")
    delta_total = scale * cmath.exp(1.0j * float(phase))
    bank = _comb_bank(atlas, pair)
    orders = list(bank.orders)
    if atlas.evaluation_order_cap is not None:
        orders[0] = min(orders[0], atlas.evaluation_order_cap)
        orders[1] = min(orders[1], atlas.evaluation_order_cap)
    q = cmath.exp(2.0j * math.pi * complex(tau))
    descendants = np.einsum(
        "tjk,j,k->t",
        bank.coefficients[:, 0, : orders[1] + 1, : orders[2] + 1],
        (cmath.exp(-1.0j * delta_total) - 1.0) ** np.arange(orders[1] + 1),
        q ** np.arange(orders[2] + 1),
        optimize=True,
    )
    h_pair, h_total, h_loop = bank.internal_weights
    first, second = pair
    spectator = pair_spectator(pair)
    v_total = cmath.exp(-1.0j * delta_total) - 1.0
    sine = 2.0 * cmath.sin(delta_total / 2.0)
    second_flat = (
        0.5j * (h_pair - atlas.external_weights[spectator]) * delta_total
        - (h_pair + atlas.external_weights[spectator]) * cmath.log(sine)
    )
    exact_other = np.exp(
        2.0 * h_total * math.log(abs(v_total))
        - 4.0 * math.pi * complex(tau).imag * (h_loop - 1.0)
        + 2.0 * np.real(second_flat)
    ) * np.abs(descendants) ** 2
    spectator_exponent = (
        koba_exponent(atlas, first, spectator)
        + koba_exponent(atlas, second, spectator)
    )
    exact_other = exact_other * cmath.exp(
        spectator_exponent * torus_prime_form_log_norm(delta_total, tau)
    )
    coefficient, a, beta, gamma, _ = _comb_corner_arrays(atlas, pair, tau)
    pair_coefficient = (
        _common_factor(atlas, tau)
        * bank.weighted_structure_constants
        * exact_other
    )
    shape_integral = np.exp(beta * math.log(shape_upper)) / beta
    pair_term = pair_coefficient * np.exp(a * math.log(scale))
    corner_term = coefficient * np.exp(gamma * math.log(scale))
    return complex(np.dot(pair_term - corner_term, shape_integral))


def comb_scale_face_integrated_over_scale(
    atlas: LiouvilleTorusThreePointAtlas,
    pair: tuple[int, int],
    shape: complex,
    scale_upper: float,
    tau: complex,
) -> complex:
    """Integrate ``S_scale-S_corner`` over the overall triple scale."""

    controller=getattr(atlas,'spectral_controller',None)
    if controller is not None:
        return controller.integrate_atlas(atlas,_scale_spectral_logs(scale_upper,abs(shape),tau),
            lambda shadow:comb_scale_face_integrated_over_scale(shadow,pair,shape,scale_upper,tau),label='comb_scale_face')

    shape_radius = abs(complex(shape))
    scale_upper = float(scale_upper)
    if shape_radius <= 0.0 or scale_upper <= 0.0:
        raise ValueError("positive shape and scale upper bound are required")
    coefficient, a, _, _, lam = _comb_corner_arrays(atlas, pair, tau)
    scale_coefficient, _, _, _, _ = _comb_corner_arrays(atlas, pair, tau, shape=shape)
    shape_edge = cmath.exp(
        koba_exponent(atlas, pair[0], pair_spectator(pair))
        * math.log(abs(1.0 - complex(shape)))
    )
    radial = np.exp(lam * math.log(scale_upper)) / lam
    return complex(
        np.dot(scale_coefficient * shape_edge - coefficient,
               np.exp(a * math.log(shape_radius)) * radial)
    )


def _power_interval(lower: float, upper: float, exponent: np.ndarray) -> np.ndarray:
    """Stable vectorized ``integral_lower^upper s^(exponent-1) ds``."""

    exponent = np.asarray(exponent, dtype=np.complex128)
    if lower <= 0.0 or upper < lower:
        raise ValueError("power interval requires 0<lower<=upper")
    if upper == lower:
        return np.zeros_like(exponent)
    log_ratio = math.log(upper / lower)
    regular = (
        np.exp(exponent * math.log(lower))
        * np.expm1(exponent * log_ratio)
        / exponent
    )
    return np.where(
        np.abs(exponent) < 1.0e-12,
        complex(log_ratio),
        regular,
    )


def comb_corner_integrated_radially(
    atlas: LiouvilleTorusThreePointAtlas,
    pair: tuple[int, int],
    theta: float,
    radii: CollisionRadii,
    tau: complex,
) -> complex:
    """Integrate the double corner projection over ``R,s`` and comb phase."""

    controller=getattr(atlas,'spectral_controller',None)
    if controller is not None:
        return controller.integrate_atlas(atlas,_scale_spectral_logs(radii.triple,triple_shape_radius(theta),tau),
            lambda shadow:comb_corner_integrated_radially(shadow,pair,theta,radii,tau),label='comb_corner')

    shape_upper = triple_shape_radius(theta)
    split = radii.pair / radii.triple
    if split > shape_upper + 1.0e-14:
        raise AssertionError("collision radii violate the resolved-sector geometry")
    coefficient, _, beta, _, lam = _comb_corner_arrays(atlas, pair, tau)
    first = (
        np.exp(lam * math.log(radii.triple))
        / lam
        * np.exp(beta * math.log(split))
        / beta
    )
    second = (
        np.exp(lam * math.log(radii.pair))
        / lam
        * _power_interval(split, shape_upper, beta - lam)
    )
    # The overall comb phase is absent from the local-plane corner and gives
    # an exact factor 2*pi.  The theta integral remains outside this function.
    return complex(2.0 * math.pi * np.dot(coefficient, first + second))


__all__ = [
    "CollisionRadii",
    "PAIR_ROWS",
    "comb_corner_integrated_radially",
    "comb_pair_face_integrated_over_shape",
    "comb_pair_projection",
    "comb_scale_face_integrated_over_scale",
    "comb_scale_projection",
    "koba_exponent",
    "nearest_pair",
    "pair_projection",
    "pair_projection_disc",
    "pair_spectator",
    "reconstruct_points",
    "reduced_channel_integrand",
    "triple_shape_radius",
]
