"""Local analytic long-tube layer on verified, immutable native CFT banks."""
from __future__ import annotations

#!/usr/bin/env python3
# genus1_three_point_nonseparating_tail
# Primary nonseparating control for the genus-one three-point cusp.
# 
# Cutting one edge of the torus necklace graph turns its level-zero term into
# the CCY sphere five-point linear block.  In ordered cylinder coordinates let
# 
#     log(q_e) = -ell_e + i theta_e,        ell_e > 0,
# 
# for the three necklace edges.  The torus height is
# ``tau_2=(ell_0+ell_1+ell_2)/(2*pi)``.  In a sector where edge ``k`` is the
# longest, the tail condition ``tau_2>T`` gives
# 
#     ell_k >= max(ell_i, ell_j, 2*pi*T-ell_i-ell_j).
# 
# At level zero on the cut edge the Liouville propagation is
# ``exp(-2*ell_k*P_k**2)``.  Introducing the real timelike loop momentum ``p``
# turns the free-boson zero-mode factor into ``exp(-ell_k*p**2/2)``.  The
# ``ell_k`` integral is consequently exact:
# 
#     integral_ell0^infinity d ell exp[-ell(2*P_k**2+p**2/2)]
#       = exp[-ell0(2*P_k**2+p**2/2)]/(2*P_k**2+p**2/2).
# 
# The remaining Gaussian ``p`` integral is evaluated below in a closed erfc
# form.  No matrix-model information or fitted normalization enters this
# control.  Corrections are the positive descendant levels on the cut edge and
# the exponentially small non-leading torus prime-form/eta terms; they must be
# measured as an ``exact-control`` residual before replacing production data.




import cmath
import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.special import wofz

try:
    from five_point_virasoro_block_recursion import sphere_five_point_direct_coefficients
    from genus1_three_point_worldsheet import (
        LiouvilleTorusThreePointNecklace,
        PrimaryCutTerms,
    )
    from torus_three_point_blocks import necklace_descendant_coefficients_three_point
except ImportError:  # pragma: no cover - package-style execution
    from five_point_virasoro_block_recursion import sphere_five_point_direct_coefficients
    from genus1_three_point_worldsheet import (
        LiouvilleTorusThreePointNecklace,
        PrimaryCutTerms,
    )
    from torus_three_point_blocks import necklace_descendant_coefficients_three_point


ORDERED_EDGE_MEASURE = 1.0 / (2.0 * math.pi**2)
PRIMARY_CUT_TAIL_MEASURE = 1.0 / math.pi


@dataclass(frozen=True)
class NecklaceEdgeGeometry:
    """Three positive necklace lengths and their twist phases."""

    logs: tuple[complex, complex, complex]
    lengths: tuple[float, float, float]
    phases: tuple[float, float, float]

    @property
    def total_length(self) -> float:
        return float(sum(self.lengths))

    @property
    def tau(self) -> complex:
        return complex(sum(self.logs) / (2.0j * math.pi))

    @property
    def longest_edge(self) -> int:
        return int(np.argmax(self.lengths))


@dataclass(frozen=True)
class PrimaryCutMatterData:
    """Free-boson data independent of the cut-edge length."""

    cylinder_positions: tuple[complex, complex, complex]
    short_total_length: float
    log_sine_koba_nielsen: float
    gaussian_numerator: float
    loop_linear_coupling: float


def necklace_edge_geometry(
    log_q_values: Sequence[complex],
) -> NecklaceEdgeGeometry:
    """Validate and expose ``log(q_e)=-ell_e+i*theta_e``."""

    if len(log_q_values) != 3:
        raise ValueError("three necklace edge logarithms are required")
    logs = tuple(complex(value) for value in log_q_values)
    lengths = tuple(float(-value.real) for value in logs)
    if any(not math.isfinite(value) or value <= 0.0 for value in lengths):
        raise ValueError("all necklace edge lengths must be positive and finite")
    phases = tuple(float(value.imag) for value in logs)
    if any(not math.isfinite(value) for value in phases):
        raise ValueError("all necklace twists must be finite")
    return NecklaceEdgeGeometry(logs, lengths, phases)


def logs_from_lengths_and_phases(
    lengths: Sequence[float],
    phases: Sequence[float],
) -> tuple[complex, complex, complex]:
    """Construct three contracting necklace logarithms."""

    if len(lengths) != 3 or len(phases) != 3:
        raise ValueError("three lengths and three phases are required")
    logs = tuple(
        complex(-float(length), float(phase))
        for length, phase in zip(lengths, phases)
    )
    necklace_edge_geometry(logs)
    return logs


def torus_coordinates_from_edge_logs(
    log_q_values: Sequence[complex],
) -> tuple[complex, complex, complex]:
    """Return ``(w1,w2,tau)`` for the ordered torus necklace."""

    geometry = necklace_edge_geometry(log_q_values)
    first = -1.0j * geometry.logs[0]
    second = -1.0j * (geometry.logs[0] + geometry.logs[1])
    return complex(first), complex(second), geometry.tau


def short_edges_for_cut(cut_edge: int) -> tuple[int, int]:
    cut_edge = int(cut_edge)
    if cut_edge not in (0, 1, 2):
        raise ValueError("cut_edge must be 0, 1, or 2")
    return (cut_edge + 1) % 3, (cut_edge + 2) % 3


def tail_sector_lower_length(
    short_lengths: Sequence[float],
    *,
    tail_start: float,
) -> float:
    """Lower length of the selected longest edge in ``tau_2>tail_start``."""

    if len(short_lengths) != 2:
        raise ValueError("two short-edge lengths are required")
    first, second = (float(value) for value in short_lengths)
    if min(first, second) <= 0.0 or not all(map(math.isfinite, (first, second))):
        raise ValueError("short-edge lengths must be positive and finite")
    if not math.isfinite(float(tail_start)) or float(tail_start) <= 0.0:
        raise ValueError("tail_start must be positive and finite")
    return float(
        max(first, second, 2.0 * math.pi * float(tail_start) - first - second)
    )


def vertex_positions_after_cut(
    log_q_values: Sequence[complex],
    *,
    cut_edge: int,
) -> tuple[complex, complex, complex]:
    """Cylinder representatives of ``(out_1,out_2,in)`` after cutting an edge.

    The incoming vertex is fixed at zero.  The two other representatives use
    only the uncut edge increments and are therefore independent of the
    length and twist of the cut edge.
    """

    geometry = necklace_edge_geometry(log_q_values)
    increments = tuple(-1.0j * value for value in geometry.logs)
    cut_edge = int(cut_edge)
    if cut_edge == 2:
        first = increments[0]
        second = increments[0] + increments[1]
    elif cut_edge == 0:
        second = -increments[2]
        first = -increments[2] - increments[1]
    elif cut_edge == 1:
        first = increments[0]
        second = -increments[2]
    else:
        raise ValueError("cut_edge must be 0, 1, or 2")
    return complex(first), complex(second), 0.0j


def _log_abs_two_sine(value: complex) -> float:
    value = complex(value)
    magnitude = abs(value.imag)
    if magnitude < 20.0:
        absolute = abs(2.0 * cmath.sin(value / 2.0))
        if absolute == 0.0:
            raise ZeroDivisionError("the primary cut hit an external collision")
        return float(math.log(absolute))
    correction = 1.0 + math.exp(-2.0 * magnitude)
    correction -= 2.0 * math.cos(value.real) * math.exp(-magnitude)
    return float(0.5 * magnitude + 0.5 * math.log(correction))


def primary_cut_matter_data(
    correlator: LiouvilleTorusThreePointNecklace,
    log_q_values: Sequence[complex],
    *,
    cut_edge: int,
    reality_tolerance: float = 1.0e-11,
) -> PrimaryCutMatterData:
    """Return the exact leading-cylinder free-boson factors.

    For the imaginary equal-split kinematics, the prime-form Gaussian is
    ``exp(B/L)`` with ``B>=0``.  The identity

    ``sqrt(2*pi/L) exp(B/L) = integral_R dp exp(-L*p^2/2+sqrt(2B)*p)``

    exposes the timelike handle momentum and makes the cut length a Laplace
    variable.
    """

    geometry = necklace_edge_geometry(log_q_values)
    positions = vertex_positions_after_cut(geometry.logs, cut_edge=cut_edge)
    first, second, _ = positions
    first_log = _log_abs_two_sine(first)
    second_log = _log_abs_two_sine(second)
    pair_log = _log_abs_two_sine(first - second)
    incoming_outgoing = correlator.omega_in * correlator.omega_out
    outgoing_pair = -correlator.omega_out * correlator.omega_out
    log_sine = (
        incoming_outgoing * (first_log + second_log)
        + outgoing_pair * pair_log
    )
    first_height = float(first.imag)
    second_height = float(second.imag)
    gaussian = -0.5 * (
        incoming_outgoing * (first_height**2 + second_height**2)
        + outgoing_pair * (first_height - second_height) ** 2
    )
    if abs(log_sine.imag) > float(reality_tolerance) or abs(gaussian.imag) > float(
        reality_tolerance
    ):
        raise ValueError("the analytic matter kernel currently requires imaginary energies")
    gaussian_real = float(gaussian.real)
    if gaussian_real < -float(reality_tolerance):
        raise ArithmeticError("the imaginary-frequency matter Gaussian has the wrong sign")
    gaussian_real = max(0.0, gaussian_real)
    short = short_edges_for_cut(cut_edge)
    return PrimaryCutMatterData(
        cylinder_positions=positions,
        short_total_length=float(sum(geometry.lengths[edge] for edge in short)),
        log_sine_koba_nielsen=float(log_sine.real),
        gaussian_numerator=gaussian_real,
        loop_linear_coupling=math.sqrt(2.0 * gaussian_real),
    )


def primary_cut_reduced_density(
    correlator: LiouvilleTorusThreePointNecklace,
    log_q_values: Sequence[complex],
    *,
    cut_edge: int,
    order_cap: int | None = None,
) -> complex:
    """Leading level-zero density before integrating the cut-edge length."""

    controller=getattr(correlator,'spectral_controller',None)
    if controller is not None:
        return controller.integrate_necklace(correlator,log_q_values,lambda shadow:
            primary_cut_reduced_density(shadow,log_q_values,cut_edge=cut_edge,order_cap=order_cap),label='primary_cut')

    geometry = necklace_edge_geometry(log_q_values)
    matter = primary_cut_matter_data(correlator, geometry.logs, cut_edge=cut_edge)
    terms = correlator.primary_cut_terms_from_logs(
        geometry.logs,
        cut_edge=cut_edge,
        order_cap=order_cap,
    )
    liouville = np.dot(
        terms.values,
        np.exp(-2.0 * terms.long_momenta**2 * geometry.lengths[cut_edge]),
    )
    total_length = geometry.total_length
    matter_value = math.sqrt(2.0 * math.pi / total_length) * math.exp(
        matter.log_sine_koba_nielsen
        + matter.gaussian_numerator / total_length
    )
    return complex(matter_value * liouville)


def primary_cut_support(
    log_q_values: Sequence[complex],
    *,
    cut_edge: int,
    largest_sewing_q: float,
    second_sewing_q: float,
) -> bool:
    """Whether the retained two-edge series is inside an explicit raw-q gate.

    These thresholds must be chosen and audited in native sewing coordinates;
    legacy elliptic thresholds are not translated or reused. Outside this
    support the analytic control is exactly zero and the residual estimator
    retains the full torus integrand, so this numerical gate cannot bias the
    control-variate identity.
    """

    geometry = necklace_edge_geometry(log_q_values)
    short = short_edges_for_cut(cut_edge)
    magnitudes = sorted(
        (
            abs(cmath.exp(geometry.logs[edge]))
            for edge in short
        ),
        reverse=True,
    )
    return bool(
        magnitudes[0] <= float(largest_sewing_q)
        and magnitudes[1] <= float(second_sewing_q)
    )


def integrated_matter_length_kernel(
    long_momenta: Sequence[float] | np.ndarray,
    *,
    lower_length: float,
    short_total_length: float,
    loop_linear_coupling: float,
    log_prefactor: float = 0.0,
) -> np.ndarray:
    r"""Integrate the cut length and timelike loop momentum analytically.

    This returns, for every Liouville momentum ``P``,

    ``int_R dp exp[-A p^2/2+b p-2 P^2 ell0]/(2 P^2+p^2/2)``,

    where ``A=ell0+ell_short``.  It is the result of first performing the
    cut-length integral and then the Gaussian timelike-momentum integral.
    The closed form contains ``erfc`` at a complex argument.  It is evaluated
    with the scaled Faddeeva function so that the large positive Gaussian in
    the ordinary formula is never formed separately.
    """

    momenta = np.asarray(long_momenta, dtype=float)
    ell0 = float(lower_length)
    short = float(short_total_length)
    coupling = float(loop_linear_coupling)
    log_prefactor = float(log_prefactor)
    if (
        np.any(~np.isfinite(momenta))
        or np.any(momenta <= 0.0)
        or not math.isfinite(ell0)
        or ell0 <= 0.0
        or not math.isfinite(short)
        or short < 0.0
        or not math.isfinite(coupling)
        or coupling < 0.0
        or not math.isfinite(log_prefactor)
    ):
        raise ValueError("invalid momentum or length in the analytic tail kernel")
    total = ell0 + short
    x_value = momenta * math.sqrt(2.0 * total)
    y_value = coupling / math.sqrt(2.0 * total)
    faddeeva_sum = wofz(y_value + 1.0j * x_value) + wofz(
        -y_value + 1.0j * x_value
    )
    if np.max(np.abs(faddeeva_sum.imag)) > 2.0e-12 * max(
        1.0, float(np.max(np.abs(faddeeva_sum.real)))
    ):
        raise ArithmeticError("the conjugate Faddeeva sum did not become real")
    real_sum = np.asarray(faddeeva_sum.real, dtype=float)
    if np.any(real_sum <= 0.0):
        raise FloatingPointError("the analytic Faddeeva sum lost positivity")
    # Keep the cylinder sine factor in this same exponential when requested.
    # At very large short-edge lengths it cancels the positive timelike
    # Gaussian exponent; forming the two factors separately overflows even
    # though their physical product is finite.
    log_result = (
        math.log(math.pi / 2.0)
        - np.log(momenta)
        - 2.0 * momenta**2 * ell0
        + coupling**2 / (2.0 * total)
        + log_prefactor
        + np.log(real_sum)
    )
    result = np.exp(log_result)
    # Underflow to zero is the correct floating-point representation for
    # exponentially suppressed large-momentum/long-cylinder terms.
    if np.any(~np.isfinite(result)) or np.any(result < 0.0):
        raise FloatingPointError("the analytic matter-length kernel is non-finite")
    return np.asarray(result, dtype=float)


def integrated_primary_cut_reduced_density(
    correlator: LiouvilleTorusThreePointNecklace,
    log_q_values: Sequence[complex],
    *,
    cut_edge: int,
    lower_length: float,
    order_cap: int | None = None,
) -> complex:
    """Integrate the primary-cut reduced density over the long edge.

    The twist angle of the cut edge is not included.  Its integral is
    ``2*pi``; combined with the ordered edge-coordinate Jacobian this changes
    ``ORDERED_EDGE_MEASURE`` into ``PRIMARY_CUT_TAIL_MEASURE``.
    """

    controller=getattr(correlator,'spectral_controller',None)
    if controller is not None:
        proposal_logs=list(log_q_values);proposal_logs[cut_edge]=complex(-lower_length)
        return controller.integrate_necklace(correlator,proposal_logs,lambda shadow:
            integrated_primary_cut_reduced_density(shadow,log_q_values,cut_edge=cut_edge,
                lower_length=lower_length,order_cap=order_cap),label='integrated_primary_cut')

    geometry = necklace_edge_geometry(log_q_values)
    matter = primary_cut_matter_data(correlator, geometry.logs, cut_edge=cut_edge)
    terms = correlator.primary_cut_terms_from_logs(
        geometry.logs,
        cut_edge=cut_edge,
        order_cap=order_cap,
    )
    kernel = integrated_matter_length_kernel(
        terms.long_momenta,
        lower_length=float(lower_length),
        short_total_length=matter.short_total_length,
        loop_linear_coupling=matter.loop_linear_coupling,
        log_prefactor=matter.log_sine_koba_nielsen,
    )
    return complex(np.dot(terms.values, kernel))


def sphere_five_point_weights_after_cut(
    internal_weights: Sequence[complex],
    external_weights: Sequence[complex],
    *,
    cut_edge: int,
) -> tuple[tuple[complex, ...], tuple[complex, complex]]:
    """Return the CCY linear-channel weights induced by a necklace cut."""

    if len(internal_weights) != 3 or len(external_weights) != 3:
        raise ValueError("three internal and three external weights are required")
    cut_edge = int(cut_edge)
    first_edge, second_edge = short_edges_for_cut(cut_edge)
    internal = tuple(complex(value) for value in internal_weights)
    external = tuple(complex(value) for value in external_weights)
    cut_weight = internal[cut_edge]
    sphere_external = (
        cut_weight,
        external[cut_edge],
        external[first_edge],
        external[second_edge],
        cut_weight,
    )
    return sphere_external, (internal[first_edge], internal[second_edge])


def maximum_cut_coefficient_identity_error(
    *,
    central_charge: complex,
    internal_weights: Sequence[complex],
    external_weights: Sequence[complex],
    cut_edge: int,
    order1: int,
    order2: int,
) -> float:
    """Compare the torus level-zero slice with the sphere five-point block."""

    cut_edge = int(cut_edge)
    first_edge, second_edge = short_edges_for_cut(cut_edge)
    orders = [0, 0, 0]
    orders[first_edge] = int(order1)
    orders[second_edge] = int(order2)
    torus = necklace_descendant_coefficients_three_point(
        central_charge,
        tuple(internal_weights),
        tuple(external_weights),
        tuple(orders),
    )
    sphere_external, sphere_internal = sphere_five_point_weights_after_cut(
        internal_weights,
        external_weights,
        cut_edge=cut_edge,
    )
    sphere = sphere_five_point_direct_coefficients(
        central_charge=central_charge,
        external_weights=sphere_external,
        internal_weights=sphere_internal,
        order1=int(order1),
        order2=int(order2),
    )
    errors = []
    for first_level in range(int(order1) + 1):
        for second_level in range(int(order2) + 1):
            levels = [0, 0, 0]
            levels[first_edge] = first_level
            levels[second_edge] = second_level
            errors.append(
                abs(torus[tuple(levels)] - sphere[(first_level, second_level)])
            )
    return float(max(errors, default=0.0))





# genus1_banked_nonseparating_tail
# Local analytic long-tube layer on verified, immutable native CFT banks.
# 
# Unlike the old necklace-only control, a pair-OPE loop can be cut without
# changing its local OPE variable.  Retained levels on the cut loop are summed
# diagonally after twist integration; the length kernel is analytic.  All short
# edge coefficients, DOZZ factors, dP/pi weights and frame factors are retained.
# 
# This module does NOT replace any production sample or certify a region.  The
# leading-cylinder matter factor is a control, not the full torus matter factor.
# A production estimator must retain the exact-minus-control residual with the
# same support, chart partition and collision exclusions as its analytic addback.
# Finite block order and momentum quadrature require independent checks.



import math

import numpy as np
from scipy.special import wofz

from genus1_native_log_evaluation import block_logs, kinematics
from genus1_native_bank import SharedNativeBank, identity, variants
from genus1_three_point_channel_atlas import nearest_torus_displacement
VERSION = 'banked-native-long-tube-local-v1'


def log_length_kernel(momenta, *, lower_length, short_length, coupling,
                      log_prefactor=0.):
    """Log of the existing Faddeeva length kernel, including a common factor.

    Keeping this logarithmic avoids underflow before multiplication by the
    independently large descendant/DOZZ factors.  momenta=sqrt(P^2+n) includes
    diagonal cut-loop descendant level n, without changing the spectral nodes.
    """
    p = np.asarray(momenta, float)
    ell, short, b = float(lower_length), float(short_length), float(coupling)
    if (p.ndim != 1 or not len(p) or not np.all(np.isfinite(p)) or np.any(p <= 0)
            or not all(map(math.isfinite, (ell, short, b, log_prefactor)))
            or ell <= 0 or short < 0 or b < 0):
        raise ValueError('invalid analytic length-kernel inputs')
    total = ell + short
    x, y = p*math.sqrt(2*total), b/math.sqrt(2*total)
    value = wofz(y+1j*x)+wofz(-y+1j*x)
    if np.any(abs(value.imag) > 2e-12*np.maximum(1., abs(value.real))):
        raise ArithmeticError('Faddeeva conjugate sum is not real')
    if not np.all(np.isfinite(value)) or np.any(value.real <= 0):
        raise FloatingPointError('invalid Faddeeva kernel')
    return (math.log(math.pi/2)-np.log(p)-2*p*p*ell
            +b*b/(2*total)+float(log_prefactor)+np.log(value.real))


def cut_geometry(atlas, geometry, *, cut_axis=None, minimum_length=12.):
    """Keep local coordinates fixed while selecting an actual long loop.

    The minimum length is explicit, not inferred from the tiny total nome.
    Chart validity is checked by the ordinary native-sewing atlas first.
    """
    kin = kinematics(atlas, **geometry)
    channel, logs = kin['channel'], kin['logs']
    eligible = {'necklace': (0, 1, 2), 'pair_ope': (1, 2), 'comb_ope': (2,)}[channel]
    axis = max(eligible, key=lambda i: -logs[i].real) if cut_axis is None else cut_axis
    if type(axis) is not int or axis not in eligible:
        raise ValueError('only a plumbing loop, not a local OPE axis, may be cut')
    if not math.isfinite(minimum_length) or minimum_length <= 0:
        raise ValueError('explicit positive long-tube threshold required')
    ell = -logs[axis].real
    if ell < minimum_length:
        raise ValueError('tube is outside the analytic-control support')
    if channel == 'necklace':
        positions = vertex_positions_after_cut(logs, cut_edge=axis)
        short = sum(-logs[i].real for i in eligible if i != axis)
    else:
        a, b = geometry['pair']; s = next(i for i in range(3) if i not in (a,b))
        raw = (complex(geometry['w1']), complex(geometry['w2']), 0j)
        delta = nearest_torus_displacement(raw[a]-raw[b], geometry['tau'])
        positions = [0j, 0j, 0j]
        if channel == 'pair_ope':
            # When loop1 is cut, move the pair through the torus seam; its
            # representatives then involve loop2 only.  Vice versa for loop2.
            positions[b] = 1j*logs[2] if axis == 1 else -1j*logs[1]
            positions[a] = positions[b]+delta
            short = -logs[3-axis].real
        else:
            positions[a] = delta
            positions[s] = nearest_torus_displacement(raw[s]-raw[b], geometry['tau'])
            short = 0.
        incoming = positions[2]
        positions = tuple(z-incoming for z in positions)
    first, second, _ = positions
    neck = atlas.necklace
    io = neck.omega_in*neck.omega_out; oo = -neck.omega_out**2
    sine = io*(_log_abs_two_sine(first)+_log_abs_two_sine(second))
    sine += oo*_log_abs_two_sine(first-second)
    gaussian = -.5*(io*(first.imag**2+second.imag**2)
                     +oo*(first.imag-second.imag)**2)
    if abs(sine.imag) > 1e-11 or abs(gaussian.imag) > 1e-11 or gaussian.real < -1e-11:
        raise ValueError('control supports the imaginary equal-split frequency slice only')
    return dict(kin=kin, axis=int(axis), length=ell, short_length=short,
                positions=positions, sine=float(sine.real),
                gaussian=max(0., float(gaussian.real)), minimum_length=minimum_length)


def geometry_at_length(cut, geometry, length, *, phase=None):
    """A replay geometry on the same fixed short-coordinate tube family.

    Used for local length-kernel tests; a global moduli-space partition is NOT
    defined here.  Callers must check the resulting native chart again.
    """
    length = float(length)
    if not math.isfinite(length) or length < cut['minimum_length']:
        raise ValueError('length outside declared support')
    tau = complex(geometry['tau'])
    tau += 1j*(length-cut['length'])/(2*math.pi)
    if phase is not None:
        tau += (float(phase)-cut['kin']['logs'][cut['axis']].imag)/(2*math.pi)
    def wrap(z):
        y = z.imag/(2*math.pi*tau.imag)
        x = z.real/(2*math.pi)-y*tau.real
        return 2*math.pi*((x % 1)+(y % 1)*tau)
    first, second = (wrap(z) for z in cut['positions'][:2])
    return dict(geometry, w1=first, w2=second, tau=tau)


def _sum_logs(logs):
    """Complex spectral sum plus the sum of absolute node contributions."""
    z = np.asarray(logs, complex)
    good = np.isfinite(z)
    if np.any(~good & ~((z.real == -math.inf) & np.isfinite(z.imag))):
        raise FloatingPointError('invalid spectral log term')
    z = z[good]
    if not len(z):
        return 0j, 0.
    peak = float(max(z.real))
    if peak > 700:
        raise FloatingPointError('spectral sum exceeds supported range')
    values = np.exp(z-peak)
    scale = math.exp(peak)
    return (complex(math.fsum(values.real), math.fsum(values.imag))*scale,
            math.fsum(abs(values))*scale)


def integrated_cut_rules(cut, threshold_slopes, order, *, lower_length=None,
                         width_scale=1., matched_cut_threshold=True):
    """Exactly reweighted dP rules for the LENGTH-INTEGRATED control.

    Its primary cut kernel behaves as 1/P_cut near threshold.  The DOZZ
    product's P_cut^2 zero therefore leaves P_cut, not P_cut^2: in x=a*P^2,
    the natural Laguerre weight has alpha=0 on that axis (alpha=1/2 elsewhere).
    This is a proposal only, not an approximation to the CFT or a convergence
    certificate.  Changed rules require new coefficients at their actual nodes.
    """
    from liouville_momentum_quadrature import generalized_gaussian_momentum_rule
    if type(order) is not int or order < 1 or len(threshold_slopes) != 3:
        raise ValueError('three threshold slopes and a positive node count required')
    if not math.isfinite(width_scale) or width_scale <= 0:
        raise ValueError('positive finite proposal width required')
    lower = cut['length'] if lower_length is None else float(lower_length)
    if not math.isfinite(lower) or lower < cut['minimum_length']:
        raise ValueError('integrated proposal outside declared long-tube support')
    decays = np.asarray(cut['kin']['decay'],float).copy()
    decays[cut['axis']] = 2*lower
    slopes = np.asarray(threshold_slopes,float)
    if not np.all(np.isfinite(slopes)):
        raise ValueError('finite threshold curvature required')
    result = []
    for i, (primary,slope) in enumerate(zip(decays,slopes)):
        decay = width_scale*max(.75*primary,primary-slope)
        alpha = 0. if matched_cut_threshold and i == cut['axis'] else .5
        rule = generalized_gaussian_momentum_rule(0j,order,laguerre_alpha=alpha,
            decay_coefficient=decay,log_q_abs=-decay/2)
        result.append(dict(nodes=np.asarray(rule.nodes),weights=math.pi*np.asarray(rule.weights),
            order=order,alpha=alpha,decay=decay,measure='dP',
            domain='[0,infinity)',scheme='integrated-long-tube-threshold-v1'))
    return tuple(result)


def kernel_weighted_cut_rule(cut, slope, order, *, lower_length=None,
                             construction_order=128, tail_exponent=80.):
    """Gaussian rule in P^2 for the positive P^2*K(P)*exp(slope*P^2) weight.

    Only the one-dimensional analytic kernel is used to construct this rule;
    no DOZZ or block evaluation is needed. A discrete orthogonal-polynomial
    construction on a fine Legendre rule supplies nodes and dP weights. The
    kernel/threshold/curvature weight is divided out exactly at those nodes.
    Construction-order and outer-tail changes must be checked independently.
    The finite construction interval is NOT a physical momentum cutoff.
    """
    from scipy.linalg import eigh_tridiagonal
    lower = cut['length'] if lower_length is None else float(lower_length)
    if (type(order) is not int or not 1 <= order <= 32
            or type(construction_order) is not int or construction_order < 4*order
            or not math.isfinite(tail_exponent) or tail_exponent < 40
            or not math.isfinite(slope) or not math.isfinite(lower)
            or lower < cut['minimum_length']):
        raise ValueError('invalid analytic-kernel rule construction')
    # Temper positive curvature if needed so the proposal has Gaussian tails.
    curvature = min(float(slope),.5*lower)
    decay = 2*lower-curvature
    limit = math.sqrt(tail_exponent/decay)
    x,w = np.polynomial.legendre.leggauss(construction_order)
    p = .5*limit*(x+1); measure = .5*limit*w
    log_kernel = log_length_kernel(p,lower_length=lower,short_length=cut['short_length'],
                                  coupling=math.sqrt(2*cut['gaussian']))
    logmass = np.log(measure)+2*np.log(p)+log_kernel+curvature*p*p
    peak = float(max(logmass)); weights = np.exp(logmass-peak)
    mass = math.fsum(weights); scaled_x = decay*p*p
    vectors = []; vector=np.sqrt(weights/mass); diagonal=[]; offdiagonal=[]
    for j in range(order):
        vectors.append(vector)
        value=scaled_x*vector
        a=float(np.dot(vector,value));diagonal.append(a)
        residual=value-a*vector
        if j:residual-=offdiagonal[-1]*vectors[-2]
        # Full reorthogonalization controls loss at the higher requested nodes.
        for _ in range(2):
            for v in vectors:residual-=np.dot(v,residual)*v
        if j<order-1:
            b=float(np.linalg.norm(residual))
            if not math.isfinite(b) or b<=0:raise ArithmeticError('kernel rule lost rank')
            offdiagonal.append(b);vector=residual/b
    abscissae,ev=eigh_tridiagonal(np.asarray(diagonal),np.asarray(offdiagonal))
    if np.any(abscissae<=0):raise ArithmeticError('nonpositive spectral node')
    nodes=np.sqrt(abscissae/decay)
    node_kernel=log_length_kernel(nodes,lower_length=lower,short_length=cut['short_length'],
                                 coupling=math.sqrt(2*cut['gaussian']))
    logweights=(peak+math.log(mass)+2*np.log(abs(ev[0]))-2*np.log(nodes)
                -node_kernel-curvature*nodes*nodes)
    output=np.exp(logweights)
    if np.any(~np.isfinite(output)) or np.any(output<=0):
        raise FloatingPointError('invalid kernel-weighted dP weights')
    return dict(nodes=nodes,weights=output,order=order,measure='dP',domain='[0,infinity)',
        scheme='analytic-length-kernel-weight-v1',curvature=curvature,
        construction_order=construction_order,construction_tail_exponent=tail_exponent,
        construction_maximum_P=limit,physical_momentum_cutoff=False,
        qualification='Numerically constructed quadrature; refine construction and spectral order.')


def evaluate_cut(bank, atlas, geometry, *, orders=None, cut_axis=None,
                 cut_level=0, lower_length=None, minimum_length=12.):
    """Evaluate full CFT, its long-loop control and analytic length addback.

    The addback includes integral d ell and the AVERAGE over the cut twist;
    it does not include any ordered-coordinate Jacobian or final -Im/(4pi).
    Spectrum is identical for every prefix/control/residual.  No coefficients
    or DOZZ values are constructed by this function.
    """
    if not isinstance(bank, SharedNativeBank):
        raise ValueError('a verified native bank is required')
    cut = cut_geometry(atlas, geometry, cut_axis=cut_axis, minimum_length=minimum_length)
    kin, axis = cut['kin'], cut['axis']
    if bank.metadata['identity'] != identity(kin):
        raise ValueError('bank graph or external momenta do not match the cut')
    requested = bank.orders if orders is None else tuple(orders)
    if (len(requested) != 3 or any(type(n) is not int or n < 0 or n > b
                                 for n,b in zip(requested,bank.orders))):
        raise ValueError('requested coefficients are not present in the saved bank')
    if type(cut_level) is not int or not 0 <= cut_level <= requested[axis]:
        raise ValueError('cut-loop levels must exist in the saved tensor')
    lower = cut['length'] if lower_length is None else float(lower_length)
    if not math.isfinite(lower) or lower < minimum_length:
        raise ValueError('analytic addback lower limit is outside the long-tube support')
    logs = kin['logs']; ps2 = bank.arrays['momenta']**2
    base = (bank.arrays['structure_logs']+bank.arrays['log_weights']-3*math.log(math.pi)
            +kin['constant'])
    short_base = base+sum(2*ps2[:,i]*logs[i].real for i in range(3) if i != axis)
    full_base = short_base-2*ps2[:,axis]*cut['length']+kin['common_log']
    total = cut['length']+cut['short_length']
    leading_matter = .5*math.log(2*math.pi/total)+cut['sine']+cut['gaussian']/total
    control_base = short_base-2*ps2[:,axis]*cut['length']+leading_matter
    rows = []
    for label, prefix in variants(requested):
        # Per-axis block checks keep the declared cut-level scheme fixed when
        # possible; a cut-axis prefix may retain fewer levels, explicitly.
        level = min(cut_level, prefix[axis])
        control_prefix = list(prefix); control_prefix[axis] = level
        full_log, full_bound = block_logs(bank, prefix, logs)
        control_log, control_bound = block_logs(bank, tuple(control_prefix), logs)
        full, full_abs = _sum_logs(full_base+full_log)
        control, control_abs = _sum_logs(control_base+control_log)
        _,full_abs = _sum_logs(full_base+full_bound)
        _,control_abs = _sum_logs(control_base+control_bound)
        # Subtract at the same spectral node before summation, not between
        # unrelated momentum quadratures.  The reported rounding diagnostic
        # prevents presenting cancellation below float resolution as accuracy.
        residual = []
        for x,y in zip(full_base+full_log, control_base+control_log):
            if x.real == -math.inf and y.real == -math.inf:
                residual.append(0j)
            elif x.real >= y.real:
                residual.append(-np.exp(x)*np.expm1(y-x))
            else:
                residual.append(np.exp(y)*np.expm1(x-y))
        residual = complex(math.fsum(z.real for z in residual), math.fsum(z.imag for z in residual))
        integrated = 0j
        for n in range(level+1):
            # Slicing leaves a singleton cut axis whose polynomial degree is
            # zero.  The original q_cut^n is handled by P_eff^2=P_cut^2+n.
            sel = [slice(None)]*4; sel[axis+1] = slice(n,n+1)
            from types import SimpleNamespace
            projected = SimpleNamespace(arrays={'coefficients':bank.arrays['coefficients'][tuple(sel)]})
            pc = list(prefix); pc[axis] = 0
            logblock, _ = block_logs(projected, tuple(pc), logs)
            kernel = log_length_kernel(np.sqrt(ps2[:,axis]+n), lower_length=lower,
                short_length=cut['short_length'], coupling=math.sqrt(2*cut['gaussian']),
                log_prefactor=cut['sine'])
            part, _ = _sum_logs(short_base+logblock+kernel)
            integrated += part
        roundoff = 128*np.finfo(float).eps*(1+sum(abs(z) for z in logs))*(full_abs+control_abs)
        if not all(np.isfinite(z) for z in (full,control,residual,integrated,roundoff)):
            raise FloatingPointError('nonfinite analytic-tail result')
        rows.append(dict(label=label, orders=prefix, cut_level=level, full=full,
                         control=control, residual=residual, integrated=integrated,
                         residual_roundoff_indicator=float(roundoff),
                         residual_below_roundoff_indicator=abs(residual)<=roundoff))
    return dict(version=VERSION, channel=kin['channel'], cut_axis=axis,
        length=cut['length'], lower_length=lower, short_length=cut['short_length'],
        rows=rows, coefficient_recomputations=0, dozz_recomputations=0,
        source_manifest_sha256=bank.metadata['source_manifest_sha256'],
        momentum_converged=False, block_converged=False, full_amplitude_certified=False,
        scope='Local cut family and same-bank diagnostics; not a global moduli estimator.')
