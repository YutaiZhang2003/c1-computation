#!/usr/bin/env python3
"""Numerical worldsheet ingredients for the c=1 genus-one two-point amplitude.

This module computes the reduced CFT integral shared by Xi and BRY. It
does not include an amplitude coupling or topology prefactor. The active
amplitude wrapper uses BRY conventions (see c1_bry_conventions.md). It
constructs the Liouville torus two-point function by integrating Virasoro
blocks over the two internal Liouville momenta. Both production channels use
``c``-recursion. The generic necklace series stays in ``q1,q2``; direct
descendant contractions remain available only as independent checks.
No matrix-model expression enters the implementation.
"""

from __future__ import annotations

from c1_paths import PACKAGE_ROOT, data_root, package_path, module_path

import cmath
import math
from dataclasses import dataclass

import numpy as np

try:
    from liouville_momentum_quadrature import (
        primary_gaussian_momentum_rule,
        threshold_gaussian_momentum_rule,
    )
    from liouville_torus import UpsilonB, yin_structure_constant_momentum
    from torus_two_point_blocks import (
        evaluate_bivariate,
        necklace_c_recursion_coefficients,
        necklace_descendant_coefficients,
        ope_c_recursion_coefficients,
        ope_coefficients_in_z,
        ope_descendant_coefficients,
    )
    from virasoro_blocks import TorusTwoPointVirasoroBlock
except ImportError:  # pragma: no cover - package-style execution
    from liouville_momentum_quadrature import (
        primary_gaussian_momentum_rule,
        threshold_gaussian_momentum_rule,
    )
    from liouville_torus import UpsilonB, yin_structure_constant_momentum
    from torus_two_point_blocks import (
        evaluate_bivariate,
        necklace_c_recursion_coefficients,
        necklace_descendant_coefficients,
        ope_c_recursion_coefficients,
        ope_coefficients_in_z,
        ope_descendant_coefficients,
    )
    from virasoro_blocks import TorusTwoPointVirasoroBlock


C_LIOUVILLE = 25.0


def regulated_h_recursion_necklace_coefficients(
    h1: float,
    h2: float,
    d1: float,
    d2: float,
    orders: tuple[int, int],
    *,
    c_regulator: float = 0.04,
    weight_regulator: float = 0.001,
) -> np.ndarray:
    r"""Legacy verification helper, excluded from production backends.

    Return the finite ``c->25`` necklace block from ``h``-recursion.

    Individual fixed-``h`` residues are resonant at ``b=1`` although their
    sum is regular.  A three-point Richardson combination at
    ``c=25+epsilon,25+2*epsilon,25+4*epsilon`` removes the first two regulator
    powers and agrees with exact ``c=25`` descendant sewing.
    """

    epsilon = float(c_regulator)
    if epsilon <= 0.0:
        raise ValueError("the c=25 h-recursion regulator must be positive")
    weight_epsilon = float(weight_regulator)
    if weight_epsilon <= 0.0:
        raise ValueError("the confluent-weight regulator must be positive")

    def coefficients_at_c(central_charge: float) -> np.ndarray:
        try:
            return TorusTwoPointVirasoroBlock(
                central_charge,
                h1,
                h2,
                d1,
                d2,
            ).descendant_coefficients(orders)
        except ZeroDivisionError:
            # Equal momentum nodes produce confluent h-poles in separate
            # recursion terms.  The complete block is regular.  Take a
            # symmetric generic-weight limit and remove its O(delta^2)
            # error with one Richardson step.
            symmetric_values = []
            for scale in (1.0, 0.5):
                delta = weight_epsilon * scale
                plus = TorusTwoPointVirasoroBlock(
                    central_charge,
                    h1 + delta,
                    h2 - delta,
                    d1,
                    d2,
                ).descendant_coefficients(orders)
                minus = TorusTwoPointVirasoroBlock(
                    central_charge,
                    h1 - delta,
                    h2 + delta,
                    d1,
                    d2,
                ).descendant_coefficients(orders)
                symmetric_values.append(0.5 * (plus + minus))
            return (4.0 * symmetric_values[1] - symmetric_values[0]) / 3.0

    first = coefficients_at_c(C_LIOUVILLE + epsilon)
    second = coefficients_at_c(C_LIOUVILLE + 2.0 * epsilon)
    fourth = coefficients_at_c(C_LIOUVILLE + 4.0 * epsilon)
    return np.asarray(
        (8.0 / 3.0) * first - 2.0 * second + fourth / 3.0,
        dtype=np.complex128,
    )


def audited_h_recursion_necklace_coefficients(
    h1: float,
    h2: float,
    d1: float,
    d2: float,
    orders: tuple[int, int],
    *,
    c_regulator: float = 0.04,
    weight_regulator: float = 0.001,
    audit_tolerance: float = 1.0e-7,
) -> tuple[np.ndarray, float, bool]:
    r"""Legacy verification helper, excluded from production backends.

    Evaluate the necklace ``h``-recursion with a regulator audit.

    Widely separated large internal weights can lose precision when the
    resonant terms cancel in double precision.  A second regulator probes
    that instability.  Nodes that fail the audit fall back to the defining
    finite-level contraction; stable nodes retain the faster recursion.
    """

    audit_tolerance = float(audit_tolerance)
    if audit_tolerance <= 0.0:
        raise ValueError("h-recursion audit_tolerance must be positive")
    primary = regulated_h_recursion_necklace_coefficients(
        h1,
        h2,
        d1,
        d2,
        orders,
        c_regulator=c_regulator,
        weight_regulator=weight_regulator,
    )
    audit = regulated_h_recursion_necklace_coefficients(
        h1,
        h2,
        d1,
        d2,
        orders,
        c_regulator=1.25 * float(c_regulator),
        weight_regulator=weight_regulator,
    )
    relative_error = float(
        np.max(
            np.abs(primary - audit)
            / np.maximum(1.0, np.maximum(np.abs(primary), np.abs(audit)))
        )
    )
    if relative_error <= audit_tolerance:
        return primary, relative_error, False
    direct = necklace_descendant_coefficients(
        C_LIOUVILLE,
        h1,
        h2,
        d1,
        d2,
        *orders,
    )
    return np.asarray(direct, dtype=np.complex128), relative_error, True


@dataclass(frozen=True)
class MomentumRule:
    nodes: np.ndarray
    weights: np.ndarray
    p_max: float | None
    order: int
    kind: str = "finite-power-legendre"
    gaussian_width: float | None = None
    q_abs: float | None = None

    @classmethod
    def legendre(cls, p_max: float, order: int) -> "MomentumRule":
        if p_max <= 0:
            raise ValueError("p_max must be positive")
        if order <= 0:
            raise ValueError("momentum quadrature order must be positive")
        raw_nodes, raw_weights = np.polynomial.legendre.leggauss(int(order))
        nodes = 0.5 * float(p_max) * (raw_nodes + 1.0)
        weights = 0.5 * float(p_max) * raw_weights
        return cls(
            nodes=nodes,
            weights=weights,
            p_max=float(p_max),
            order=int(order),
            kind="finite-legendre",
        )

    @classmethod
    def power_legendre(
        cls,
        p_max: float,
        order: int,
        power: float = 2.0,
    ) -> "MomentumRule":
        """Gauss-Legendre after ``P=p_max*u**power``, ``0<u<1``.

        The quadratic default clusters nodes near ``P=0``.  This is essential
        in the collision and cusp regions, where the dominant momentum scale
        shrinks as an inverse square root of the degeneration length.
        """
        if p_max <= 0:
            raise ValueError("p_max must be positive")
        if order <= 0:
            raise ValueError("momentum quadrature order must be positive")
        if power <= 1.0:
            raise ValueError("power must exceed one for endpoint clustering")
        raw_nodes, raw_weights = np.polynomial.legendre.leggauss(int(order))
        u_nodes = 0.5 * (raw_nodes + 1.0)
        u_weights = 0.5 * raw_weights
        nodes = float(p_max) * u_nodes**float(power)
        weights = (
            u_weights
            * float(p_max)
            * float(power)
            * u_nodes ** (float(power) - 1.0)
        )
        return cls(
            nodes=nodes,
            weights=weights,
            p_max=float(p_max),
            order=int(order),
            kind=f"finite-power-legendre-{float(power):g}",
        )

    @classmethod
    def threshold_gaussian(
        cls,
        q_value: complex,
        order: int,
        *,
        log_q_abs: float | None = None,
    ) -> "MomentumRule":
        r"""Generalized-Laguerre rule matched to ``P^2 exp(-a P^2)``.

        Here ``a=-2 log|q|`` is the exact nonchiral primary-propagation
        coefficient.  The low-level shared rule includes ``dP/pi`` in its
        weights; this wrapper restores ``dP`` because the genus-one two-point
        correlator inserts the common ``1/pi^2`` measure itself.
        """

        shared = threshold_gaussian_momentum_rule(
            q_value,
            int(order),
            log_q_abs=log_q_abs,
        )
        return cls(
            nodes=np.asarray(shared.nodes, dtype=float),
            weights=math.pi * np.asarray(shared.weights, dtype=float),
            p_max=None,
            order=int(order),
            kind="threshold-gaussian",
            gaussian_width=float(shared.gaussian_width),
            q_abs=float(abs(complex(q_value))),
        )

    @classmethod
    def primary_gaussian(
        cls,
        q_value: complex,
        order: int,
        *,
        log_q_abs: float | None = None,
    ) -> "MomentumRule":
        r"""Generalized-Laguerre rule matched to ``exp(-a P^2)``.

        This is needed for the analytically integrated collision disc: its
        explicit ``1/P_ope^2`` radial factor cancels the two OPE-channel
        threshold zeros, while the loop momentum retains its ``P^2`` zero.
        """

        shared = primary_gaussian_momentum_rule(
            q_value,
            int(order),
            log_q_abs=log_q_abs,
        )
        return cls(
            nodes=np.asarray(shared.nodes, dtype=float),
            weights=math.pi * np.asarray(shared.weights, dtype=float),
            p_max=None,
            order=int(order),
            kind="primary-gaussian",
            gaussian_width=float(shared.gaussian_width),
            q_abs=float(abs(complex(q_value))),
        )


@dataclass(frozen=True)
class MomentumPairRule:
    """A non-tensor quadrature rule for the two internal momenta.

    ``weights`` integrate ``dP_first*dP_second``.  The correlator inserts the
    conventional ``1/pi^2`` exactly as for a tensor product of two
    :class:`MomentumRule` objects.
    """

    first_nodes: np.ndarray
    second_nodes: np.ndarray
    weights: np.ndarray
    radial_order: int
    angular_order: int
    kind: str
    first_q_abs: float
    second_q_abs: float
    first_gaussian_width: float
    second_gaussian_width: float
    radial_laguerre_alpha: float
    angular_jacobi_alpha: float
    angular_jacobi_beta: float


@dataclass(frozen=True)
class _BlockTerm:
    h_first: float
    h_second: float
    weighted_structure_constant: complex
    coefficients: np.ndarray


class LiouvilleTorusTwoPoint:
    """Two-channel momentum quadrature for ``<V_{omega/2} V_{omega/2}>``.

    The external energy may be imaginary.  For ``omega=i*x`` with
    ``0<x<1`` the real momentum contours are not crossed by DOZZ poles, so
    this class directly implements equations (4.2) and (4.3) of BRY.
    """

    def __init__(
        self,
        omega: complex,
        *,
        momentum_rule: MomentumRule | None = None,
        momentum_rules: tuple[MomentumRule, MomentumRule] | None = None,
        momentum_pair_rule: MomentumPairRule | None = None,
        necklace_orders: tuple[int, int] = (4, 4),
        ope_orders: tuple[int, int] = (2, 6),
        necklace_backend: str = "c-recursion",
        ope_backend: str = "c-recursion",
        h_recursion_regulator: float = 0.04,
        h_recursion_weight_regulator: float = 0.001,
        h_recursion_audit_tolerance: float = 1.0e-7,
        special_dps: int = 28,
    ) -> None:
        self.omega = complex(omega)
        self.external_momentum = self.omega / 2.0
        self.external_weight = complex(1.0 + self.omega * self.omega / 4.0)
        if abs(self.external_weight.imag) > 1.0e-12:
            raise ValueError(
                "the current nonchiral implementation requires real external weight; "
                "use purely real or purely imaginary omega"
            )
        self.external_weight = complex(self.external_weight.real)
        supplied_rule_forms = sum(
            value is not None
            for value in (momentum_rule, momentum_rules, momentum_pair_rule)
        )
        if supplied_rule_forms != 1:
            raise ValueError(
                "provide exactly one of momentum_rule, momentum_rules, or "
                "momentum_pair_rule"
            )
        if momentum_rules is None and momentum_rule is not None:
            momentum_rules = (momentum_rule, momentum_rule)
        if momentum_rules is not None and len(momentum_rules) != 2:
            raise ValueError("two internal momentum rules are required")
        self.momentum_rules = (
            None if momentum_rules is None else tuple(momentum_rules)
        )
        self.momentum_pair_rule = momentum_pair_rule
        # Retain the original public attribute for old callers.  New local
        # quadratures should use ``momentum_rules`` because the two sewing
        # cylinders generally have different Gaussian widths.
        self.momentum_rule = momentum_rule
        self.necklace_orders = tuple(int(value) for value in necklace_orders)
        self.ope_orders = tuple(int(value) for value in ope_orders)
        if necklace_backend not in {"direct-descendants", "c-recursion"}:
            raise ValueError("the native necklace uses c-recursion; direct descendants are a check")
        if ope_backend not in {"direct-descendants", "c-recursion"}:
            raise ValueError("unknown torus two-point OPE backend")
        if float(h_recursion_regulator) <= 0.0:
            raise ValueError("h_recursion_regulator must be positive")
        if float(h_recursion_weight_regulator) <= 0.0:
            raise ValueError("h_recursion_weight_regulator must be positive")
        if float(h_recursion_audit_tolerance) <= 0.0:
            raise ValueError("h_recursion_audit_tolerance must be positive")
        self.necklace_backend = str(necklace_backend)
        self.ope_backend = str(ope_backend)
        self.h_recursion_regulator = float(h_recursion_regulator)
        self.h_recursion_weight_regulator = float(h_recursion_weight_regulator)
        self.h_recursion_audit_tolerance = float(h_recursion_audit_tolerance)
        self.h_recursion_audit_max_relative_error = 0.0
        self.h_recursion_fallback_count = 0
        self.h_recursion_node_count = 0
        self.special = UpsilonB(1.0, dps=int(special_dps))
        self._necklace_terms: list[_BlockTerm] | None = None
        self._ope_terms: list[_BlockTerm] | None = None
        self._necklace_arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
        self._ope_arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
        self._ope_residue_arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
        self._crossed_pole_residue: complex | None = None

    @property
    def necklace_terms(self) -> list[_BlockTerm]:
        if self._necklace_terms is None:
            self._necklace_terms = self._build_necklace_terms()
        return self._necklace_terms

    @property
    def ope_terms(self) -> list[_BlockTerm]:
        if self._ope_terms is None:
            self._ope_terms = self._build_ope_terms()
        return self._ope_terms

    def _build_necklace_terms(self) -> list[_BlockTerm]:
        first_order, second_order = self.necklace_orders
        terms: list[_BlockTerm] = []
        if self.momentum_pair_rule is not None:
            samples = zip(
                self.momentum_pair_rule.first_nodes,
                self.momentum_pair_rule.second_nodes,
                self.momentum_pair_rule.weights,
            )
        else:
            assert self.momentum_rules is not None
            first_rule, second_rule = self.momentum_rules
            samples = (
                (p1, p2, first_rule.weights[first_index] * second_rule.weights[second_index])
                for first_index, p1 in enumerate(first_rule.nodes)
                for second_index, p2 in enumerate(second_rule.nodes)
            )
        for p1, p2, raw_quadrature_weight in samples:
            h1 = 1.0 + float(p1) ** 2
            h2 = 1.0 + float(p2) ** 2
            structure = yin_structure_constant_momentum(
                self.special,
                self.external_momentum,
                float(p1),
                float(p2),
            )
            quadrature_weight = float(raw_quadrature_weight) / math.pi**2
            coefficient_builder = (
                necklace_c_recursion_coefficients
                if self.necklace_backend == "c-recursion"
                else necklace_descendant_coefficients
            )
            coefficients = coefficient_builder(
                C_LIOUVILLE,
                h1,
                h2,
                self.external_weight,
                self.external_weight,
                first_order,
                second_order,
            )
            terms.append(
                _BlockTerm(
                    h_first=h1,
                    h_second=h2,
                    weighted_structure_constant=quadrature_weight * structure * structure,
                    coefficients=coefficients,
                )
            )
        return terms

    def _build_ope_terms(self) -> list[_BlockTerm]:
        q_order, z_order = self.ope_orders
        terms: list[_BlockTerm] = []
        if self.momentum_pair_rule is not None:
            samples = zip(
                self.momentum_pair_rule.first_nodes,
                self.momentum_pair_rule.second_nodes,
                self.momentum_pair_rule.weights,
            )
        else:
            assert self.momentum_rules is not None
            loop_rule, ope_rule = self.momentum_rules
            samples = (
                (
                    p_loop,
                    p_ope,
                    loop_rule.weights[loop_index] * ope_rule.weights[ope_index],
                )
                for loop_index, p_loop in enumerate(loop_rule.nodes)
                for ope_index, p_ope in enumerate(ope_rule.nodes)
            )
        for p_loop, p_ope, raw_quadrature_weight in samples:
            h_loop = 1.0 + float(p_loop) ** 2
            h_ope = 1.0 + float(p_ope) ** 2
            structure_external = yin_structure_constant_momentum(
                self.special,
                self.external_momentum,
                self.external_momentum,
                float(p_ope),
            )
            structure_loop = yin_structure_constant_momentum(
                self.special,
                float(p_loop),
                float(p_loop),
                float(p_ope),
            )
            quadrature_weight = float(raw_quadrature_weight) / math.pi**2
            coefficient_builder = (
                ope_c_recursion_coefficients
                if self.ope_backend == "c-recursion"
                else ope_descendant_coefficients
            )
            coefficients = coefficient_builder(
                C_LIOUVILLE,
                h_loop,
                h_ope,
                self.external_weight,
                self.external_weight,
                q_order,
                z_order,
            )
            coefficients = ope_coefficients_in_z(coefficients, z_order)
            terms.append(
                _BlockTerm(
                    h_first=h_loop,
                    h_second=h_ope,
                    weighted_structure_constant=(
                        quadrature_weight * structure_external * structure_loop
                    ),
                    coefficients=coefficients,
                )
            )
        return terms

    def correlator_necklace(self, z: complex, tau: complex) -> complex:
        """Evaluate the necklace-channel Liouville correlator."""
        z = complex(z)
        tau = complex(tau)
        log_q1 = 1.0j * z
        log_q2 = 1.0j * (2.0 * math.pi * tau - z)
        q1 = cmath.exp(log_q1)
        q2 = cmath.exp(log_q2)
        if self._necklace_arrays is None:
            terms = self.necklace_terms
            self._necklace_arrays = (
                np.asarray([term.h_first for term in terms], dtype=float),
                np.asarray([term.h_second for term in terms], dtype=float),
                np.asarray([term.weighted_structure_constant for term in terms], dtype=complex),
                np.stack([term.coefficients for term in terms]),
            )
        h1, h2, weights, coefficients = self._necklace_arrays
        powers1 = q1 ** np.arange(coefficients.shape[1])
        powers2 = q2 ** np.arange(coefficients.shape[2])
        descendants = np.einsum("tij,i,j->t", coefficients, powers1, powers2, optimize=True)
        primary_norm_squared = np.exp(
            2.0
            * (
                (h1 - C_LIOUVILLE / 24.0) * log_q1.real
                + (h2 - C_LIOUVILLE / 24.0) * log_q2.real
            )
        )
        return np.dot(weights, primary_norm_squared * np.abs(descendants) ** 2)

    def correlator_ope(self, z: complex, tau: complex) -> complex:
        """Evaluate the OPE-channel Liouville correlator."""
        z = complex(z)
        tau = complex(tau)
        q = cmath.exp(2.0j * math.pi * tau)
        v = cmath.exp(-1.0j * z) - 1.0
        flat_frame = cmath.exp(
            -2.0 * self.external_weight * cmath.log(2.0 * cmath.sin(z / 2.0))
        )
        if self._ope_arrays is None:
            terms = self.ope_terms
            self._ope_arrays = (
                np.asarray([term.h_first for term in terms], dtype=float),
                np.asarray([term.h_second for term in terms], dtype=float),
                np.asarray([term.weighted_structure_constant for term in terms], dtype=complex),
                np.stack([term.coefficients for term in terms]),
            )
        h_loop, h_ope, weights, coefficients = self._ope_arrays
        q_powers = q ** np.arange(coefficients.shape[1])
        z_powers = z ** np.arange(coefficients.shape[2])
        descendants = np.einsum("tij,i,j->t", coefficients, q_powers, z_powers, optimize=True)
        primary_norm_squared = np.exp(
            -4.0 * math.pi * tau.imag * (h_loop - C_LIOUVILLE / 24.0)
            + 2.0 * h_ope * math.log(abs(v))
        )
        block_norm_squared = (
            abs(flat_frame) ** 2 * primary_norm_squared * np.abs(descendants) ** 2
        )
        return np.dot(weights, block_norm_squared)

    def crossed_ope_pole_residue(self) -> complex:
        """Return ``Res C(omega/2,omega/2,P)`` at ``P=omega-i``.

        The pole crosses the positive-real OPE contour when
        ``1<Im(omega)<2``.  A small real displacement and a linear
        extrapolation determine the residue independently of the block.
        """
        if self._crossed_pole_residue is not None:
            return self._crossed_pole_residue
        if not (
            abs(self.omega.real) < 1.0e-12
            and 1.0 < self.omega.imag < 2.0
        ):
            raise ValueError("the implemented crossed-pole formula requires omega=i*x, 1<x<2")
        pole = self.omega - 1.0j
        displacements = np.asarray([1.0e-4, 5.0e-5, 2.5e-5], dtype=float)
        samples = np.asarray(
            [
                displacement
                * yin_structure_constant_momentum(
                    self.special,
                    self.external_momentum,
                    self.external_momentum,
                    pole + displacement,
                )
                for displacement in displacements
            ]
        )
        design = np.column_stack([np.ones_like(displacements), displacements])
        real_fit, *_ = np.linalg.lstsq(design, samples.real, rcond=None)
        imag_fit, *_ = np.linalg.lstsq(design, samples.imag, rcond=None)
        self._crossed_pole_residue = complex(real_fit[0], imag_fit[0])
        return self._crossed_pole_residue

    def _build_ope_residue_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pole = self.omega - 1.0j
        h_ope = float((1.0 + pole * pole).real)
        q_order, z_order = self.ope_orders
        residue = self.crossed_ope_pole_residue()
        terms: list[_BlockTerm] = []
        if self.momentum_pair_rule is not None:
            raise ValueError(
                "crossed-pole residue evaluation requires a one-dimensional "
                "loop rule, not a joint pair rule"
            )
        assert self.momentum_rules is not None
        loop_rule = self.momentum_rules[0]
        for loop_index, p_loop in enumerate(loop_rule.nodes):
            h_loop = 1.0 + float(p_loop) ** 2
            loop_structure = yin_structure_constant_momentum(
                self.special,
                float(p_loop),
                float(p_loop),
                pole,
            )
            coefficient_builder = (
                ope_c_recursion_coefficients
                if self.ope_backend == "c-recursion"
                else ope_descendant_coefficients
            )
            coefficients = coefficient_builder(
                C_LIOUVILLE,
                h_loop,
                h_ope,
                self.external_weight,
                self.external_weight,
                q_order,
                z_order,
            )
            coefficients = ope_coefficients_in_z(coefficients, z_order)
            terms.append(
                _BlockTerm(
                    h_first=h_loop,
                    h_second=h_ope,
                    weighted_structure_constant=(
                        -2.0j
                        * loop_rule.weights[loop_index]
                        / math.pi
                        * residue
                        * loop_structure
                    ),
                    coefficients=coefficients,
                )
            )
        return (
            np.asarray([term.h_first for term in terms], dtype=float),
            np.asarray([term.h_second for term in terms], dtype=float),
            np.asarray([term.weighted_structure_constant for term in terms], dtype=complex),
            np.stack([term.coefficients for term in terms]),
        )

    def correlator_ope_residue(self, z: complex, tau: complex) -> complex:
        """Evaluate the ``-2i Res`` contribution from the crossed OPE pole."""
        if self._ope_residue_arrays is None:
            self._ope_residue_arrays = self._build_ope_residue_arrays()
        h_loop, h_ope, weights, coefficients = self._ope_residue_arrays
        z = complex(z)
        tau = complex(tau)
        q = cmath.exp(2.0j * math.pi * tau)
        v = cmath.exp(-1.0j * z) - 1.0
        flat_frame = cmath.exp(
            -2.0 * self.external_weight * cmath.log(2.0 * cmath.sin(z / 2.0))
        )
        q_powers = q ** np.arange(coefficients.shape[1])
        z_powers = z ** np.arange(coefficients.shape[2])
        descendants = np.einsum("tij,i,j->t", coefficients, q_powers, z_powers, optimize=True)
        primary_norm_squared = np.exp(
            -4.0 * math.pi * tau.imag * (h_loop - C_LIOUVILLE / 24.0)
            + 2.0 * h_ope * math.log(abs(v))
        )
        return np.dot(
            weights,
            abs(flat_frame) ** 2 * primary_norm_squared * np.abs(descendants) ** 2,
        )

    def correlator_ope_analytically_continued(self, z: complex, tau: complex) -> complex:
        """OPE contour integral plus every crossed pole in ``1<Im omega<2``."""
        value = self.correlator_ope(z, tau)
        if abs(self.omega.real) < 1.0e-12 and 1.0 < self.omega.imag < 2.0:
            value += self.correlator_ope_residue(z, tau)
        return value

    def correlator_patched(
        self,
        z: complex,
        tau: complex,
        *,
        epsilon: float = 0.15,
    ) -> complex:
        """Use OPE discs around lattice points and necklace elsewhere."""
        z = complex(z)
        candidates = (
            z,
            z - 2.0 * math.pi,
            z - 2.0 * math.pi * complex(tau),
            z - 2.0 * math.pi * (1.0 + complex(tau)),
        )
        local_z = min(candidates, key=abs)
        if abs(local_z) < 2.0 * math.pi * float(epsilon):
            return self.correlator_ope(local_z, tau)
        return self.correlator_necklace(z, tau)

    def leading_collision_disc(self, tau: complex, radius: float) -> complex:
        """Analytically integrate the leading OPE term over ``|z|<radius``.

        After multiplying the Liouville OPE block by the timelike correlator,
        the radial dependence of an OPE channel of momentum ``P`` is
        ``r**(-2+2*P**2)``.  Its full-disc integral is

        ``pi * radius**(2*P**2) / P**2``.

        The two simple zeros at ``P=0`` (one in each structure-constant
        factor) make the remaining momentum integral finite.  This is the
        collision-disc treatment described in Appendix B.2 of BRY.
        """
        radius = float(radius)
        if radius <= 0.0:
            raise ValueError("collision-disc radius must be positive")
        tau = complex(tau)
        q = cmath.exp(2.0j * math.pi * tau)
        if self._ope_arrays is None:
            _ = self.correlator_ope(0.3 + 0.1j, tau)
        assert self._ope_arrays is not None
        h_loop, h_ope, weights, coefficients = self._ope_arrays
        q_powers = q ** np.arange(coefficients.shape[1])
        descendants_at_zero = np.einsum(
            "tij,i,j->t",
            coefficients,
            q_powers,
            np.r_[1.0 + 0.0j, np.zeros(coefficients.shape[2] - 1, dtype=complex)],
            optimize=True,
        )
        loop_norm_squared = np.exp(
            -4.0 * math.pi * tau.imag * (h_loop - C_LIOUVILLE / 24.0)
        ) * np.abs(descendants_at_zero) ** 2
        p_ope_squared = h_ope - 1.0
        radial_integrals = (
            math.pi
            * np.exp(2.0 * p_ope_squared * math.log(radius))
            / p_ope_squared
        )
        liouville_and_z = np.dot(weights, loop_norm_squared * radial_integrals)
        return (
            abs(dedekind_eta(tau)) ** 2
            * liouville_and_z
            / math.sqrt(tau.imag)
        )


def dedekind_eta(tau: complex, tolerance: float = 1.0e-16) -> complex:
    """Dedekind eta on the modular fundamental domain."""
    tau = complex(tau)
    q = cmath.exp(2.0j * math.pi * tau)
    value = cmath.exp(math.pi * 1.0j * tau / 12.0)
    q_power = q
    n = 1
    while abs(q_power) > tolerance and n < 10000:
        value *= 1.0 - q_power
        n += 1
        q_power *= q
    return value


def torus_prime_form_norm(z: complex, tau: complex, tolerance: float = 1.0e-16) -> float:
    """Return the single-valued norm used in the timelike-boson correlator.

    This is
    ``|2*pi*theta1(z/(2*pi)|tau)/theta1'(0|tau)|`` times the Gaussian.
    The theta convention has period one in its first argument.
    """
    z = complex(z)
    tau = complex(tau)
    q_theta = cmath.exp(math.pi * 1.0j * tau)
    w = z / (2.0 * math.pi)
    # Divide out the common factor 2*q_theta^(1/4).  This prevents a 0/0
    # underflow in the deep cusp while leaving the theta ratio unchanged.
    theta_reduced = 0.0 + 0.0j
    derivative_reduced = 0.0 + 0.0j
    for n in range(10000):
        coefficient = ((-1.0) ** n) * q_theta ** (n * (n + 1))
        mode = 2 * n + 1
        theta_reduced += coefficient * cmath.sin(math.pi * mode * w)
        derivative_reduced += coefficient * math.pi * mode
        if n > 0 and abs(coefficient) < tolerance:
            break
    holomorphic_norm = abs(2.0 * math.pi * theta_reduced / derivative_reduced)
    gaussian = math.exp(-(z.imag**2) / (4.0 * math.pi * tau.imag))
    return holomorphic_norm * gaussian


def reduced_worldsheet_integrand(
    correlator: LiouvilleTorusTwoPoint,
    z: complex,
    tau: complex,
    *,
    epsilon: float = 0.15,
) -> complex:
    """Return the integrand of the native reduced amplitude ``I_1``."""
    tau = complex(tau)
    if tau.imag <= 0:
        raise ValueError("tau must lie in the upper half-plane")
    prime_norm = torus_prime_form_norm(z, tau)
    timelike = cmath.exp(correlator.omega**2 * math.log(prime_norm))
    liouville = correlator.correlator_patched(z, tau, epsilon=epsilon)
    return abs(dedekind_eta(tau)) ** 2 * timelike * liouville / math.sqrt(tau.imag)
