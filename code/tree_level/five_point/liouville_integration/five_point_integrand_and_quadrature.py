#!/usr/bin/env python3
"""Reusable equal-energy kernel for the sphere five-point worldsheet integral.

For ``omega_2=...=omega_5=omega`` the incoming energy is ``4*omega``.
Only the slot occupied by the incoming primary distinguishes the external
weights in an oriented linear channel.  This module therefore precomputes the
three DOZZ factors and the five-point block coefficient table on a fixed
``(P1,P2)`` quadrature grid for each of the five possible incoming slots.
Moduli evaluations then require only bivariate polynomial evaluation and
elementary powers, rather than a fresh block recursion and Upsilon evaluation.

The imaginary-frequency production integrates the convergent moduli density
without collar subtraction, retaining the required Liouville pole residues.
Unused physical-frequency collar routines live in
``code/historical/five_point_collar_subtraction``. The raw driver rejects
energies for which a power subtraction is required.
"""

from __future__ import annotations

from c1_paths import PACKAGE_ROOT, data_root, package_path, module_path

import cmath
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.stats import qmc

from momentum_quadrature import _gauss_legendre_grid

try:
    from five_point_virasoro_block_recursion import (
        CoefficientTable,
        evaluate_sphere_five_point_series,
        sphere_five_point_c_coefficients,
        sphere_five_point_h_c25_limit,
    )
    from liouville_torus import UpsilonB, log_yin_structure_constant_momentum
    from five_point_liouville_correlator_and_atlas import (
        INFINITY,
        LinearChannel,
        best_linear_channels,
        linear_channel_complex_jacobian_to_chart,
        linear_channel_from_ordering,
        linear_channel_positions_by_label,
        linear_channel_to_original_chart,
        liouville_primary_covariance,
        liouville_primary_covariance_log,
        oriented_tree_orderings,
        timelike_free_boson_factor,
        timelike_free_boson_log_factor,
    )
    from five_point_liouville_correlator_and_atlas import divergent_spin_zero_levels
    from five_point_liouville_correlator_and_atlas import equal_outgoing_signed_energies
    from five_point_liouville_correlator_and_atlas import five_point_boundary_divisors
    from five_point_liouville_correlator_and_atlas import signed_channel_energy
except ImportError:  # pragma: no cover - supports package-style execution
    from five_point_virasoro_block_recursion import (
        CoefficientTable,
        evaluate_sphere_five_point_series,
        sphere_five_point_c_coefficients,
        sphere_five_point_h_c25_limit,
    )
    from liouville_torus import UpsilonB, log_yin_structure_constant_momentum
    from five_point_liouville_correlator_and_atlas import (
        INFINITY,
        LinearChannel,
        best_linear_channels,
        linear_channel_complex_jacobian_to_chart,
        linear_channel_from_ordering,
        linear_channel_positions_by_label,
        linear_channel_to_original_chart,
        liouville_primary_covariance,
        liouville_primary_covariance_log,
        oriented_tree_orderings,
        timelike_free_boson_factor,
        timelike_free_boson_log_factor,
    )
    from five_point_liouville_correlator_and_atlas import divergent_spin_zero_levels
    from five_point_liouville_correlator_and_atlas import equal_outgoing_signed_energies
    from five_point_liouville_correlator_and_atlas import five_point_boundary_divisors
    from five_point_liouville_correlator_and_atlas import signed_channel_energy


from five_point_moduli_integration import (
    EqualEnergyQMCResult,
    _plane_map,
    integrate_convergent_equal_energy_qmc,
    _power_disk_sample,
    oriented_bidisc_mixture_density,
    oriented_bidisc_log_mixture_density_in_frame,
    integrate_convergent_equal_energy_atlas_qmc,
    write_qmc_result,
)

@dataclass(frozen=True)
class MomentumGridEntry:
    p1: complex | float
    p2: complex | float
    h1: complex | float
    h2: complex | float
    log_weighted_structure_constant: complex
    weighted_structure_constant: complex
    coefficients: CoefficientTable
    coefficient_errors: CoefficientTable


@dataclass(frozen=True)
class MomentumGridArrays:
    """Vectorized form of one incoming-slot momentum quadrature."""

    log_weighted_structure_constants: np.ndarray
    primary_x_exponents: np.ndarray
    primary_y_exponents: np.ndarray
    coefficient_keys: tuple[tuple[int, int], ...]
    coefficient_matrix: np.ndarray


@dataclass(frozen=True)
class ContinuedMomentumGridArrays:
    """Vectorized one-dimensional residue contribution after a pole crossing."""

    log_weighted_structure_constants: np.ndarray
    primary_x_exponents: np.ndarray
    primary_y_exponents: np.ndarray
    coefficient_keys: tuple[tuple[int, int], ...]
    coefficient_matrix: np.ndarray




class EqualEnergyFivePointKernel:
    """Precomputed double-momentum kernel for one complex outgoing energy."""

    def __init__(
        self,
        omega: complex,
        *,
        block_order: int = 3,
        momentum_order: int = 8,
        momentum_maximum: float = 6.0,
        momentum_panels: int = 1,
        momentum_power: float = 1.0,
        endpoint_refinement: int = 0,
        block_scheme: str = "h",
        liouville_contour: str = "real",
        special_dps: int = 35,
        h_regulator_etas: Sequence[float] = (0.16, 0.13, 0.10, 0.075, 0.055),
    ) -> None:
        self.omega = complex(omega)
        self.block_order = int(block_order)
        self.momentum_order = int(momentum_order)
        self.momentum_maximum = float(momentum_maximum)
        self.momentum_panels = int(momentum_panels)
        self.momentum_power = float(momentum_power)
        self.endpoint_refinement = int(endpoint_refinement)
        self.block_scheme = str(block_scheme)
        self.liouville_contour = str(liouville_contour)
        if (
            self.block_order < 0
            or self.momentum_order <= 0
            or self.momentum_panels <= 0
            or self.endpoint_refinement < 0
        ):
            raise ValueError("block_order must be non-negative and momentum_order positive")
        if not math.isfinite(self.momentum_maximum) or self.momentum_maximum <= 0.0:
            raise ValueError("momentum_maximum must be positive and finite")
        if not math.isfinite(self.momentum_power) or self.momentum_power < 1.0:
            raise ValueError("momentum_power must be finite and at least one")
        if self.block_scheme not in {"h", "c"}:
            raise ValueError("block_scheme must be 'h' or 'c'")
        if self.liouville_contour not in {"real", "continued"}:
            raise ValueError("liouville_contour must be 'real' or 'continued'")

        self.crossed_pole: complex | None = None
        self.crossed_cherry_residue = 0.0 + 0.0j
        if self.liouville_contour == "continued":
            if abs(self.omega.real) > 1.0e-13 or not 0.0 < self.omega.imag < 0.5:
                raise ValueError(
                    "the continued Liouville contour currently supports only "
                    "omega=i*t with 0<t<1/2"
                )
            if abs(self.omega.imag - 0.4) < 1.0e-13:
                raise ValueError(
                    "t=2/5 is a Liouville contour pinch; evaluate its limit "
                    "from either adjacent chamber"
                )
            # At an incoming--outgoing cherry, C(2it,it/2,P) has the two
            # nearest denominator poles
            #
            #   P_+=i(5t/2-1),       P_-=-P_+.
            #
            # They pinch the quotient-contour endpoint at t=2/5.  Continuing
            # through the pinch keeps the real-contour integral and adds the
            # crossed-pole residue with the BRY coefficient -2i.  A second
            # family pinches this representation at t=1/2, which is why this
            # first continuation is deliberately restricted to t<1/2.
            if self.omega.imag > 0.4:
                self.crossed_pole = 2.5 * self.omega - 1.0j

        self.signed_energies = equal_outgoing_signed_energies(self.omega)
        self.external_momenta = (
            2.0 * self.omega,
            0.5 * self.omega,
            0.5 * self.omega,
            0.5 * self.omega,
            0.5 * self.omega,
        )
        self.external_weights = tuple(
            1.0 + momentum * momentum for momentum in self.external_momenta
        )
        self.special = UpsilonB(1.0, dps=int(special_dps))
        self.h_regulator_etas = tuple(float(value) for value in h_regulator_etas)
        # Adjacent quadrature orders keep each tensor grid away from the exact
        # P1=P2 diagonal, where the common-h1 form of the CCY h-recursion has
        # coincident internal-weight poles.  Averaging N x (N+1) with its
        # transpose restores exact P1<->P2 symmetry of the quadrature rule.
        power_breakpoints: set[float] = set()
        for channel_energy in (3.0 * self.omega, -2.0 * self.omega):
            threshold = 0.25 * (channel_energy**2).real
            for level in range(self.block_order + 1):
                if threshold > level:
                    endpoint = math.sqrt(threshold - level)
                    power_breakpoints.add(endpoint)
                    local_width = abs((channel_energy**2).imag) / max(
                        4.0 * endpoint,
                        1.0e-12,
                    )
                    multiples = (0.5, 1.0, 2.0, 4.0)[: self.endpoint_refinement]
                    for multiple in multiples:
                        power_breakpoints.update(
                            (
                                endpoint - multiple * local_width,
                                endpoint + multiple * local_width,
                            )
                        )
        p_low_nodes, p_low_weights = _gauss_legendre_grid(
            self.momentum_order,
            self.momentum_maximum,
            self.momentum_panels,
            sorted(power_breakpoints),
            self.momentum_power,
        )
        p_high_nodes, p_high_weights = _gauss_legendre_grid(
            self.momentum_order + 1,
            self.momentum_maximum,
            self.momentum_panels,
            sorted(power_breakpoints),
            self.momentum_power,
        )
        self.momentum_grids = (
            (p_low_nodes, p_low_weights, p_high_nodes, p_high_weights, 0.5),
            (p_high_nodes, p_high_weights, p_low_nodes, p_low_weights, 0.5),
        )
        self.entries_by_incoming_slot = self._build_entries()
        self.arrays_by_incoming_slot = self._build_arrays()
        self.discrete_entries_by_incoming_slot = self._build_discrete_entries()
        self.discrete_arrays_by_incoming_slot = self._build_discrete_arrays()

    def requires_power_subtraction(self) -> bool:
        """Return whether any of the ten boundary projectors is nonempty."""

        return any(
            divergent_spin_zero_levels(
                signed_channel_energy(divisor, self.signed_energies)
            )
            for divisor in five_point_boundary_divisors()
        )

    def _build_entries(self) -> tuple[tuple[MomentumGridEntry, ...], ...]:
        entries_by_slot: list[tuple[MomentumGridEntry, ...]] = []
        incoming_momentum = self.external_momenta[0]
        outgoing_momentum = self.external_momenta[1]
        for incoming_slot in range(5):
            ordered_momenta = tuple(
                incoming_momentum if slot == incoming_slot else outgoing_momentum
                for slot in range(5)
            )
            ordered_weights = tuple(1.0 + value * value for value in ordered_momenta)
            pa, pb, pc, pd, pe = ordered_momenta
            slot_entries: list[MomentumGridEntry] = []
            for p1_nodes, p1_weights, p2_nodes, p2_weights, grid_scale in self.momentum_grids:
                first_logs = tuple(
                    complex(
                        log_yin_structure_constant_momentum(
                            self.special, pa, pb, float(p1)
                        )
                    )
                    for p1 in p1_nodes
                )
                last_logs = tuple(
                    complex(
                        log_yin_structure_constant_momentum(
                            self.special, float(p2), pd, pe
                        )
                    )
                    for p2 in p2_nodes
                )
                for p1_index, (p1, weight1) in enumerate(zip(p1_nodes, p1_weights)):
                    p1_value = float(p1)
                    h1 = 1.0 + p1_value**2
                    for p2_index, (p2, weight2) in enumerate(zip(p2_nodes, p2_weights)):
                        p2_value = float(p2)
                        middle_log = complex(
                            log_yin_structure_constant_momentum(
                                self.special, p1_value, pc, p2_value
                            )
                        )
                        log_weighted_structure_constant = complex(
                            math.log(
                                grid_scale
                                * float(weight1)
                                * float(weight2)
                                / (math.pi * math.pi)
                            )
                            + first_logs[p1_index]
                            + middle_log
                            + last_logs[p2_index]
                        )
                        h2 = 1.0 + p2_value**2
                        if self.block_scheme == "h":
                            coefficients, coefficient_errors = sphere_five_point_h_c25_limit(
                                external_weights=ordered_weights,
                                internal_weights=(h1, h2),
                                order1=self.block_order,
                                order2=self.block_order,
                                max_total_order=self.block_order,
                                regulator_etas=self.h_regulator_etas,
                                polynomial_degree=min(3, len(self.h_regulator_etas) - 1),
                            )
                        else:
                            coefficients = sphere_five_point_c_coefficients(
                                central_charge=25.0,
                                external_weights=ordered_weights,
                                internal_weights=(h1, h2),
                                order1=self.block_order,
                                order2=self.block_order,
                                max_total_order=self.block_order,
                            )
                            coefficient_errors = {key: 0.0 + 0.0j for key in coefficients}
                        slot_entries.append(
                            MomentumGridEntry(
                                p1=p1_value,
                                p2=p2_value,
                                h1=h1,
                                h2=h2,
                                log_weighted_structure_constant=log_weighted_structure_constant,
                                weighted_structure_constant=cmath.exp(
                                    log_weighted_structure_constant
                                ),
                                coefficients=coefficients,
                                coefficient_errors=coefficient_errors,
                            )
                        )
            entries_by_slot.append(tuple(slot_entries))
        return tuple(entries_by_slot)

    def _crossed_structure_constant_residue(
        self,
        first: complex,
        second: complex,
        pole: complex,
    ) -> complex:
        """Numerically extract ``Res_P C(first,second,P)`` symmetrically."""

        pole = complex(pole)
        step = min(1.0e-5, max(1.0e-7, 1.0e-3 * abs(pole)))

        def estimate(offset: float) -> complex:
            plus = cmath.exp(
                complex(
                    log_yin_structure_constant_momentum(
                        self.special, first, second, pole + offset
                    )
                )
            )
            minus = cmath.exp(
                complex(
                    log_yin_structure_constant_momentum(
                        self.special, first, second, pole - offset
                    )
                )
            )
            return 0.5 * (offset * plus + (-offset) * minus)

        coarse = estimate(step)
        fine = estimate(0.5 * step)
        return complex((4.0 * fine - coarse) / 3.0)

    def _build_discrete_entries(self) -> tuple[tuple[MomentumGridEntry, ...], ...]:
        """Build the one-dimensional terms generated at the t=2/5 pinch."""

        if self.crossed_pole is None:
            return tuple(tuple() for _ in range(5))

        pole = complex(self.crossed_pole)
        incoming_momentum = self.external_momenta[0]
        outgoing_momentum = self.external_momenta[1]
        residue = self._crossed_structure_constant_residue(
            incoming_momentum,
            outgoing_momentum,
            pole,
        )
        self.crossed_cherry_residue = residue
        contour_coefficient = -2.0j * residue
        entries_by_slot: list[tuple[MomentumGridEntry, ...]] = []
        for incoming_slot in range(5):
            if incoming_slot == 2:
                entries_by_slot.append(tuple())
                continue
            ordered_momenta = tuple(
                incoming_momentum if slot == incoming_slot else outgoing_momentum
                for slot in range(5)
            )
            ordered_weights = tuple(1.0 + value * value for value in ordered_momenta)
            pa, pb, pc, pd, pe = ordered_momenta
            slot_entries: list[MomentumGridEntry] = []
            incoming_on_left = incoming_slot in (0, 1)
            for p1_nodes, p1_weights, p2_nodes, p2_weights, grid_scale in self.momentum_grids:
                variable_nodes = p2_nodes if incoming_on_left else p1_nodes
                variable_weights = p2_weights if incoming_on_left else p1_weights
                for variable, weight in zip(variable_nodes, variable_weights):
                    variable = float(variable)
                    if incoming_on_left:
                        p1_value: complex | float = pole
                        p2_value: complex | float = variable
                        remaining_logs = (
                            log_yin_structure_constant_momentum(
                                self.special, pole, pc, variable
                            ),
                            log_yin_structure_constant_momentum(
                                self.special, variable, pd, pe
                            ),
                        )
                    else:
                        p1_value = variable
                        p2_value = pole
                        remaining_logs = (
                            log_yin_structure_constant_momentum(
                                self.special, pa, pb, variable
                            ),
                            log_yin_structure_constant_momentum(
                                self.special, variable, pc, pole
                            ),
                        )
                    h1 = 1.0 + p1_value**2
                    h2 = 1.0 + p2_value**2
                    logarithm = complex(
                        cmath.log(
                            contour_coefficient
                            * grid_scale
                            * float(weight)
                            / math.pi
                        )
                        + complex(remaining_logs[0])
                        + complex(remaining_logs[1])
                    )
                    if self.block_scheme == "h":
                        coefficients, coefficient_errors = sphere_five_point_h_c25_limit(
                            external_weights=ordered_weights,
                            internal_weights=(h1, h2),
                            order1=self.block_order,
                            order2=self.block_order,
                            max_total_order=self.block_order,
                            regulator_etas=self.h_regulator_etas,
                            polynomial_degree=min(3, len(self.h_regulator_etas) - 1),
                        )
                    else:
                        coefficients = sphere_five_point_c_coefficients(
                            central_charge=25.0,
                            external_weights=ordered_weights,
                            internal_weights=(h1, h2),
                            order1=self.block_order,
                            order2=self.block_order,
                            max_total_order=self.block_order,
                        )
                        coefficient_errors = {
                            key: 0.0 + 0.0j for key in coefficients
                        }
                    slot_entries.append(
                        MomentumGridEntry(
                            p1=p1_value,
                            p2=p2_value,
                            h1=h1,
                            h2=h2,
                            log_weighted_structure_constant=logarithm,
                            weighted_structure_constant=cmath.exp(logarithm),
                            coefficients=coefficients,
                            coefficient_errors=coefficient_errors,
                        )
                    )
            entries_by_slot.append(tuple(slot_entries))
        return tuple(entries_by_slot)

    def _build_discrete_arrays(
        self,
    ) -> tuple[ContinuedMomentumGridArrays | None, ...]:
        arrays: list[ContinuedMomentumGridArrays | None] = []
        incoming_momentum = self.external_momenta[0]
        outgoing_momentum = self.external_momenta[1]
        for incoming_slot, entries in enumerate(self.discrete_entries_by_incoming_slot):
            if not entries:
                arrays.append(None)
                continue
            ordered_momenta = tuple(
                incoming_momentum if slot == incoming_slot else outgoing_momentum
                for slot in range(5)
            )
            ordered_weights = tuple(1.0 + value * value for value in ordered_momenta)
            coefficient_keys = tuple(
                sorted(
                    {key for entry in entries for key in entry.coefficients},
                    key=lambda key: (key[0] + key[1], key[0], key[1]),
                )
            )
            arrays.append(
                ContinuedMomentumGridArrays(
                    log_weighted_structure_constants=np.asarray(
                        [entry.log_weighted_structure_constant for entry in entries],
                        dtype=complex,
                    ),
                    primary_x_exponents=np.asarray(
                        [
                            2.0
                            * (entry.h1 - ordered_weights[0] - ordered_weights[1])
                            for entry in entries
                        ],
                        dtype=complex,
                    ),
                    primary_y_exponents=np.asarray(
                        [
                            2.0
                            * (entry.h2 - ordered_weights[2] - entry.h1)
                            for entry in entries
                        ],
                        dtype=complex,
                    ),
                    coefficient_keys=coefficient_keys,
                    coefficient_matrix=np.asarray(
                        [
                            [
                                entry.coefficients.get(key, 0.0 + 0.0j)
                                for key in coefficient_keys
                            ]
                            for entry in entries
                        ],
                        dtype=complex,
                    ),
                )
            )
        return tuple(arrays)

    def _build_arrays(self) -> tuple[MomentumGridArrays, ...]:
        arrays: list[MomentumGridArrays] = []
        incoming_momentum = self.external_momenta[0]
        outgoing_momentum = self.external_momenta[1]
        for incoming_slot, entries in enumerate(self.entries_by_incoming_slot):
            ordered_momenta = tuple(
                incoming_momentum if slot == incoming_slot else outgoing_momentum
                for slot in range(5)
            )
            ordered_weights = tuple(1.0 + value * value for value in ordered_momenta)
            coefficient_keys = tuple(
                sorted(
                    {key for entry in entries for key in entry.coefficients},
                    key=lambda key: (key[0] + key[1], key[0], key[1]),
                )
            )
            arrays.append(
                MomentumGridArrays(
                    log_weighted_structure_constants=np.asarray(
                        [entry.log_weighted_structure_constant for entry in entries],
                        dtype=complex,
                    ),
                    primary_x_exponents=np.asarray(
                        [
                            2.0
                            * (entry.h1 - ordered_weights[0] - ordered_weights[1])
                            for entry in entries
                        ],
                        dtype=complex,
                    ),
                    primary_y_exponents=np.asarray(
                        [
                            2.0
                            * (entry.h2 - ordered_weights[2] - entry.h1)
                            for entry in entries
                        ],
                        dtype=complex,
                    ),
                    coefficient_keys=coefficient_keys,
                    coefficient_matrix=np.asarray(
                        [
                            [entry.coefficients.get(key, 0.0 + 0.0j) for key in coefficient_keys]
                            for entry in entries
                        ],
                        dtype=complex,
                    ),
                )
            )
        return tuple(arrays)

    @staticmethod
    def _series_monomials(
        q1: complex,
        q2: complex,
        keys: Sequence[tuple[int, int]],
    ) -> np.ndarray:
        return np.asarray(
            [complex(q1) ** first * complex(q2) ** second for first, second in keys],
            dtype=complex,
        )

    def _momentum_sum(
        self,
        incoming_slot: int,
        q1: complex,
        q2: complex,
        log_x: float,
        log_y: float,
        base_logarithm: complex,
    ) -> complex:
        def evaluate_arrays(arrays: MomentumGridArrays | ContinuedMomentumGridArrays) -> complex:
            holomorphic = np.sum(
                arrays.coefficient_matrix
                * self._series_monomials(q1, q2, arrays.coefficient_keys)[None, :],
                axis=1,
            )
            antiholomorphic = np.sum(
                arrays.coefficient_matrix
                * self._series_monomials(
                    complex(q1).conjugate(),
                    complex(q2).conjugate(),
                    arrays.coefficient_keys,
                )[None, :],
                axis=1,
            )
            exponentials = np.exp(
                complex(base_logarithm)
                + arrays.log_weighted_structure_constants
                + arrays.primary_x_exponents * float(log_x)
                + arrays.primary_y_exponents * float(log_y)
            )
            return complex(np.sum(exponentials * holomorphic * antiholomorphic))

        incoming_slot = int(incoming_slot)
        total = evaluate_arrays(self.arrays_by_incoming_slot[incoming_slot])
        discrete = self.discrete_arrays_by_incoming_slot[incoming_slot]
        if discrete is not None:
            total += evaluate_arrays(discrete)
        return complex(total)


    def liouville_correlator(self, channel: LinearChannel) -> complex:
        """Evaluate the precomputed double-momentum sum in one channel."""

        incoming_slot = channel.ordering.index(0)
        ordered_weights = tuple(self.external_weights[label] for label in channel.ordering)
        x = complex(channel.positions[1])
        y = complex(channel.positions[2])
        q1 = channel.q1
        q2 = channel.q2
        q1_bar = q1.conjugate()
        q2_bar = q2.conjugate()
        return self._momentum_sum(
            incoming_slot,
            q1,
            q2,
            math.log(abs(x)),
            math.log(abs(y)),
            0.0 + 0.0j,
        )

    def integrand(self, z_incoming: complex, z_outgoing: complex) -> complex:
        """Return the raw time-like times Liouville matter integrand."""

        positions = (
            complex(z_incoming),
            complex(z_outgoing),
            0.0 + 0.0j,
            1.0 + 0.0j,
            INFINITY,
        )
        channel = best_linear_channels(positions, limit=1)[0]
        incoming_slot = channel.ordering.index(0)
        ordered_weights = tuple(self.external_weights[label] for label in channel.ordering)
        x = complex(channel.positions[1])
        y = complex(channel.positions[2])
        log_x = math.log(abs(x))
        log_y = math.log(abs(y))
        base_logarithm = (
            liouville_primary_covariance_log(channel, self.external_weights)
            + timelike_free_boson_log_factor(positions, self.signed_energies)
        )
        return self._momentum_sum(
            incoming_slot,
            channel.q1,
            channel.q2,
            log_x,
            log_y,
            base_logarithm,
        )

    def integrand_linear_gauge_weighted(
        self,
        q1: complex,
        q2: complex,
        ordering: Sequence[int],
        *,
        logarithmic_weight: float = 0.0,
    ) -> complex:
        r"""Evaluate the density in an oriented ``(0,q1*q2,q2,1,inf)`` gauge.

        ``logarithmic_weight`` multiplies the answer by its exponential inside
        every momentum summand.  The atlas integrator uses this to divide by
        a possibly enormous mixture density without first overflowing either
        the raw OPE integrand or that density.
        """

        ordering_tuple = tuple(int(label) for label in ordering)
        if len(ordering_tuple) != 5 or set(ordering_tuple) != set(range(5)):
            raise ValueError("ordering must be a permutation of labels 0,...,4")
        q1 = complex(q1)
        q2 = complex(q2)
        if q1 == 0.0 or q2 == 0.0:
            raise ZeroDivisionError("the linear gauge lies on a plumbing boundary")
        positions = linear_channel_positions_by_label(q1, q2, ordering_tuple)
        # The proposal chart resolves its chosen boundary, but need not be the
        # fastest conformal-block chart for a point near another edge of its
        # bidisc.  Re-expand in the best of the fifteen trees while retaining
        # the sampled gauge for the moduli measure and time-like correlator.
        channel = best_linear_channels(positions, limit=1)[0]
        incoming_slot = channel.ordering.index(0)
        ordered_weights = tuple(
            self.external_weights[label] for label in channel.ordering
        )
        log_x = math.log(abs(complex(channel.positions[1])))
        log_y = math.log(abs(complex(channel.positions[2])))
        base_logarithm = (
            timelike_free_boson_log_factor(positions, self.signed_energies)
            + liouville_primary_covariance_log(channel, self.external_weights)
            + float(logarithmic_weight)
        )
        return self._momentum_sum(
            incoming_slot,
            channel.q1,
            channel.q2,
            log_x,
            log_y,
            base_logarithm,
        )

    def integrand_linear_gauge(
        self,
        q1: complex,
        q2: complex,
        ordering: Sequence[int],
    ) -> complex:
        """Return the raw density in an oriented linear-channel gauge."""

        return self.integrand_linear_gauge_weighted(q1, q2, ordering)














