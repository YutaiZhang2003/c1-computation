#!/usr/bin/env python3
"""Blind worldsheet kernel for the c=1 genus-one 1->2 amplitude.

The incoming energy is ``omega_in=i*t`` and the two outgoing energies are
``omega_out=i*t/2``. The necklace block uses fixed-weight c-recursion at
c=25 with verified multiprecision arithmetic. Pair and comb channels use
the same numerical policy in the atlas. Direct PBW sewing is an independent
validation benchmark, never a production fallback. No matrix-model expression
is imported or evaluated here.
"""

from __future__ import annotations

from c1_paths import PACKAGE_ROOT, data_root, package_path, module_path

import cmath
import concurrent.futures
import json
import math
import os
from pathlib import Path
from dataclasses import dataclass
from itertools import product

import numpy as np

try:
    from genus1_two_point_worldsheet import (
        MomentumRule,
        dedekind_eta,
        torus_prime_form_norm,
    )
    from liouville_torus import UpsilonB, yin_structure_constant_momentum
    from torus_c_recursion_numerics import (
        stable_c_recursion_coefficients,
        C_RECURSION_NUMERICS_VERSION,
    )
except ImportError:  # pragma: no cover - package-style execution
    from genus1_two_point_worldsheet import (
        MomentumRule,
        dedekind_eta,
        torus_prime_form_norm,
    )
    from liouville_torus import UpsilonB, yin_structure_constant_momentum
    from torus_c_recursion_numerics import (
        stable_c_recursion_coefficients,
        C_RECURSION_NUMERICS_VERSION,
    )


C_LIOUVILLE = 25.0


def cache_metadata_matches(
    actual: dict[str, object],
    expected: dict[str, object],
    *,
    relative_tolerance: float = 5.0e-13,
    absolute_tolerance: float = 5.0e-15,
) -> bool:
    """Compare cache designs while allowing last-bit quadrature roundoff.

    NumPy/SciPy builds on macOS and Cannon can produce Gauss--Legendre nodes
    and weights that differ by a few ulps.  All discrete and physical design
    fields remain exact; only the two quadrature arrays receive a tight
    numerical comparison.
    """

    if set(actual) != set(expected):
        return False
    numeric_fields = ("momentum_nodes", "momentum_weights")
    for key in actual:
        if key in numeric_fields:
            try:
                observed = np.asarray(actual[key], dtype=float)
                target = np.asarray(expected[key], dtype=float)
            except (TypeError, ValueError):
                # Momentum-sliced audit caches have different node counts
                # on different edges. Compare each edge independently;
                # NumPy rightly rejects their ragged list as one ndarray.
                try:
                    observed_rows = [np.asarray(row, dtype=float) for row in actual[key]]
                    target_rows = [np.asarray(row, dtype=float) for row in expected[key]]
                except (TypeError, ValueError):
                    return False
                if len(observed_rows) != len(target_rows) or not all(
                    row.shape == reference.shape and np.allclose(
                        row, reference, rtol=float(relative_tolerance),
                        atol=float(absolute_tolerance),
                    )
                    for row, reference in zip(observed_rows, target_rows)
                ):
                    return False
                continue
            if observed.shape != target.shape or not np.allclose(
                observed,
                target,
                rtol=float(relative_tolerance),
                atol=float(absolute_tolerance),
            ):
                return False
        elif actual[key] != expected[key]:
            return False
    return True


def elliptic_nome_from_log_q(log_q: complex) -> complex:
    """Reject legacy callers instead of silently changing their convention."""
    raise ValueError(
        "Elliptic re-expansion is disabled for genus-one three-point blocks. "
        "Use exp(log_q) with native-sewing-v2 coefficients and audit raw-q thresholds."
    )


@dataclass(frozen=True)
class _ThreePointBlockBank:
    """Vectorized data for one rectangular edge-order assignment."""

    orders: tuple[int, int, int]
    internal_weights: tuple[np.ndarray, np.ndarray, np.ndarray]
    weighted_structure_constants: np.ndarray
    coefficients: np.ndarray


@dataclass(frozen=True)
class PrimaryCutTerms:
    """Fixed-momentum terms after cutting one necklace edge at level zero.

    The cut edge becomes the pair of equal external primaries of a sphere
    five-point block.  ``values`` includes the three DOZZ constants, all
    momentum-quadrature weights, the two uncut primary propagators, and the
    two-edge descendant block.  Propagation along the cut edge is deliberately
    omitted so that its length can be integrated analytically by the cusp
    layer.
    """

    cut_edge: int
    short_edges: tuple[int, int]
    long_momenta: np.ndarray
    values: np.ndarray
    selected_orders: tuple[int, int, int]
    maximum_short_sewing_parameter: float


def _c_recursion_coefficients(
    internal_weights: tuple[float, float, float],
    external_weights: tuple[float, float, float],
    orders: tuple[int, int, int],
) -> np.ndarray:
    """Return verified fixed-weight c-recursion coefficients at exactly c=25."""
    coefficients, _, _ = stable_c_recursion_coefficients(
        "necklace", central_charge=C_LIOUVILLE,
        external_weights=external_weights, internal_weights=internal_weights,
        orders=orders,
    )
    return coefficients


def _c_recursion_task(
    task: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[int, int, int],
    ],
) -> np.ndarray:
    """Pickle-friendly worker entry point for one c-recursion tensor."""
    internal_weights, external_weights, orders = task
    return _c_recursion_coefficients(internal_weights, external_weights, orders)


def evaluate_pade_rows(
    coefficients: np.ndarray,
    value: complex,
    numerator_order: int,
    denominator_order: int,
) -> np.ndarray:
    """Evaluate one Padé approximant for every row of series coefficients."""
    coefficients = np.asarray(coefficients, dtype=np.complex128)
    if coefficients.ndim != 2:
        raise ValueError("Padé row coefficients must form a matrix")
    numerator_order = int(numerator_order)
    denominator_order = int(denominator_order)
    required_order = numerator_order + denominator_order
    if numerator_order < 0 or denominator_order <= 0:
        raise ValueError("Padé orders must satisfy L>=0 and M>0")
    if coefficients.shape[1] <= required_order:
        raise ValueError("not enough series coefficients for the requested Padé order")

    row_count = coefficients.shape[0]
    # A descendant series can span hundreds of orders of magnitude across
    # momentum nodes.  Padé is invariant under an overall rescaling of one
    # row, so normalize before forming its linear system.  This prevents an
    # algebraically harmless large common factor from overflowing either the
    # solve or the numerator convolution.
    row_scales = np.max(np.abs(coefficients[:, : required_order + 1]), axis=1)
    row_scales = np.where(row_scales > 0.0, row_scales, 1.0)
    series = coefficients[:, : required_order + 1] / row_scales[:, np.newaxis]
    matrices = np.empty(
        (row_count, denominator_order, denominator_order),
        dtype=np.complex128,
    )
    right_hand_sides = np.empty((row_count, denominator_order), dtype=np.complex128)
    for row in range(denominator_order):
        series_index = numerator_order + 1 + row
        right_hand_sides[:, row] = -series[:, series_index]
        for column in range(denominator_order):
            matrices[:, row, column] = series[
                :,
                series_index - (column + 1),
            ]
    try:
        denominator_tail = np.linalg.solve(
            matrices,
            right_hand_sides[..., np.newaxis],
        )[..., 0]
    except np.linalg.LinAlgError:
        denominator_tail = np.empty_like(right_hand_sides)
        for row in range(row_count):
            denominator_tail[row], *_ = np.linalg.lstsq(
                matrices[row],
                right_hand_sides[row],
                rcond=None,
            )
    denominator = np.column_stack(
        [np.ones(row_count, dtype=np.complex128), denominator_tail]
    )
    numerator = np.zeros((row_count, numerator_order + 1), dtype=np.complex128)
    for order in range(numerator_order + 1):
        for denominator_index in range(min(order, denominator_order) + 1):
            numerator[:, order] += (
                denominator[:, denominator_index]
                * series[:, order - denominator_index]
            )
    rational_scales = np.maximum(
        np.max(np.abs(numerator), axis=1),
        np.max(np.abs(denominator), axis=1),
    )
    rational_scales = np.where(rational_scales > 0.0, rational_scales, 1.0)
    numerator /= rational_scales[:, np.newaxis]
    denominator /= rational_scales[:, np.newaxis]
    powers_numerator = complex(value) ** np.arange(numerator_order + 1)
    powers_denominator = complex(value) ** np.arange(denominator_order + 1)
    numerator_value = numerator @ powers_numerator
    denominator_value = denominator @ powers_denominator
    result = row_scales * numerator_value / denominator_value
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("non-finite row-wise Padé approximant")
    return result


class LiouvilleTorusThreePointNecklace:
    """Three-momentum necklace representation of the torus 1->2 correlator."""

    def __init__(
        self,
        t: float,
        *,
        momentum_rule: MomentumRule | None = None,
        momentum_rules: tuple[MomentumRule, MomentumRule, MomentumRule] | None = None,
        high_order: int = 6,
        low_order: int = 2,
        adaptive_tolerance: float = 5.0e-5,
        c_regulator: float | None = None,
        block_backend: str = "c-recursion",
        special_dps: int = 28,
        coefficient_workers: int = 1,
    ) -> None:
        t = float(t)
        if not 0.0 < t < 1.0:
            raise ValueError("the direct equal-split real-contour slice requires 0<t<1")
        if high_order < low_order:
            raise ValueError("high_order must be at least low_order")
        if low_order < 0:
            raise ValueError("block orders must be non-negative")
        if not 0.0 < adaptive_tolerance < 1.0:
            raise ValueError("adaptive_tolerance must lie between zero and one")

        self.t = t
        self.omega_in = 1.0j * t
        self.omega_out = 0.5j * t
        self.external_momenta = (
            self.omega_out / 2.0,
            self.omega_out / 2.0,
            self.omega_in / 2.0,
        )
        self.external_weights = tuple(
            float((1.0 + omega * omega / 4.0).real)
            for omega in (self.omega_out, self.omega_out, self.omega_in)
        )
        self.signed_energies = (self.omega_in, -self.omega_out, -self.omega_out)
        if (momentum_rule is None) == (momentum_rules is None):
            raise ValueError("provide exactly one of momentum_rule or momentum_rules")
        if momentum_rules is None:
            assert momentum_rule is not None
            momentum_rules = (momentum_rule, momentum_rule, momentum_rule)
        if len(momentum_rules) != 3:
            raise ValueError("three internal momentum rules are required")
        self.momentum_rules = tuple(momentum_rules)
        self.momentum_rule = momentum_rule
        self.high_order = int(high_order)
        self.low_order = int(low_order)
        self.adaptive_tolerance = float(adaptive_tolerance)
        # Accepted only for old callers. No central-charge regulator enters
        # fixed-c recursion, its cache identity, or its convergence checks.
        self.c_regulator = None
        if block_backend != "c-recursion":
            raise ValueError(
                "genus-one three-point production requires c-recursion in every "
                "channel; use direct PBW functions only for validation"
            )
        self.block_backend = str(block_backend)
        self.special = UpsilonB(1.0, dps=int(special_dps))
        self.special_dps = int(special_dps)
        self.coefficient_workers = int(coefficient_workers)
        if self.coefficient_workers < 1:
            raise ValueError("coefficient_workers must be positive")
        self._banks: dict[int, _ThreePointBlockBank] = {}
        self._cached_structure_matrices: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self.order_histogram = {order: 0 for order in range(self.low_order, self.high_order + 1)}
        self.high_edge_histogram = {edge: 0 for edge in range(3)}
        self.maximum_q_seen = 0.0
        self.maximum_second_q_seen = 0.0

    def _structure_matrices(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._cached_structure_matrices is not None:
            return self._cached_structure_matrices
        matrices: list[np.ndarray] = []
        for edge, external_momentum in enumerate(self.external_momenta):
            first_nodes = np.asarray(self.momentum_rules[edge].nodes, dtype=float)
            second_nodes = np.asarray(
                self.momentum_rules[(edge + 1) % 3].nodes,
                dtype=float,
            )
            matrix = np.empty((len(first_nodes), len(second_nodes)), dtype=np.complex128)
            for first_index, first in enumerate(first_nodes):
                for second_index, second in enumerate(second_nodes):
                    matrix[first_index, second_index] = yin_structure_constant_momentum(
                        self.special,
                        external_momentum,
                        float(first),
                        float(second),
                    )
            matrices.append(matrix)
        self._cached_structure_matrices = tuple(matrices)  # type: ignore[assignment]
        return self._cached_structure_matrices

    def _build_bank(self, high_edge: int) -> _ThreePointBlockBank:
        if high_edge not in (0, 1, 2):
            raise ValueError("high_edge must be 0, 1, or 2")
        orders_list = [self.low_order] * 3
        orders_list[high_edge] = self.high_order
        orders = tuple(orders_list)
        nodes = tuple(
            np.asarray(rule.nodes, dtype=float) for rule in self.momentum_rules
        )
        quadrature_weights = tuple(
            np.asarray(rule.weights, dtype=float) for rule in self.momentum_rules
        )
        structures = self._structure_matrices()

        h_arrays = [[], [], []]
        weighted_structures: list[complex] = []
        coefficient_tensors: list[np.ndarray] = []
        index_rows = tuple(product(
            range(len(nodes[0])),
            range(len(nodes[1])),
            range(len(nodes[2])),
        ))
        coefficient_rows: list[np.ndarray] | None = None
        if self.coefficient_workers > 1:
            tasks = []
            for indices in index_rows:
                momenta = tuple(
                    float(nodes[edge][index]) for edge, index in enumerate(indices)
                )
                internal_weights = tuple(1.0 + momentum * momentum for momentum in momenta)
                tasks.append(
                    (internal_weights, self.external_weights, orders)
                )
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=self.coefficient_workers
            ) as executor:
                coefficient_rows = list(executor.map(_c_recursion_task, tasks))

        for flat_index, (first_index, second_index, third_index) in enumerate(index_rows):
            indices = (first_index, second_index, third_index)
            momenta = tuple(float(nodes[edge][index]) for edge, index in enumerate(indices))
            internal_weights = tuple(1.0 + momentum * momentum for momentum in momenta)
            structure_product = (
                structures[0][first_index, second_index]
                * structures[1][second_index, third_index]
                * structures[2][third_index, first_index]
            )
            quadrature_weight = (
                quadrature_weights[0][first_index]
                * quadrature_weights[1][second_index]
                * quadrature_weights[2][third_index]
                / math.pi**3
            )
            if coefficient_rows is None:
                raw_coefficients = _c_recursion_coefficients(
                    internal_weights, self.external_weights, orders)
            else:
                raw_coefficients = coefficient_rows[flat_index]
            coefficients = raw_coefficients
            for edge in range(3):
                h_arrays[edge].append(internal_weights[edge])
            weighted_structures.append(quadrature_weight * structure_product)
            coefficient_tensors.append(coefficients)
            if (flat_index + 1) % 32 == 0 or flat_index + 1 == len(index_rows):
                print(
                    f"three-point {self.block_backend} bank progress "
                    f"{flat_index + 1}/{len(index_rows)} for high edge {high_edge}",
                    flush=True,
                )

        return _ThreePointBlockBank(
            orders=orders,
            internal_weights=tuple(
                np.asarray(values, dtype=float) for values in h_arrays
            ),  # type: ignore[arg-type]
            weighted_structure_constants=np.asarray(weighted_structures, dtype=complex),
            coefficients=np.stack(coefficient_tensors),
        )

    def prepare(self, *, checkpoint_path: str | Path | None = None) -> None:
        """Build all three anisotropic banks before starting moduli sampling."""
        for edge in range(3):
            if edge not in self._banks:
                self._banks[edge] = self._build_bank(edge)
                print(
                    f"prepared three-point {self.block_backend} bank for high edge {edge} "
                    f"with orders {self._banks[edge].orders}",
                    flush=True,
                )
                if checkpoint_path is not None:
                    self.save_banks(checkpoint_path, prepare_missing=False)
                    print(
                        f"checkpointed necklace banks through high edge {edge} "
                        f"to {checkpoint_path}",
                        flush=True,
                    )

    def save_banks(
        self,
        path: str | Path,
        *,
        prepare_missing: bool = True,
    ) -> None:
        """Persist prepared numerical banks for restartable high-order runs."""
        if prepare_missing:
            self.prepare()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "format": "torus-three-point-bank-v3-verified-c-recursion",
            "coefficient_basis": "native-sewing-v2",
            "coefficient_numerics": C_RECURSION_NUMERICS_VERSION,
            "t": self.t,
            "external_weights": list(self.external_weights),
            "high_order": self.high_order,
            "low_order": self.low_order,
            "adaptive_tolerance": self.adaptive_tolerance,
            "block_backend": self.block_backend,
            "momentum_nodes": [rule.nodes.tolist() for rule in self.momentum_rules],
            "momentum_weights": [rule.weights.tolist() for rule in self.momentum_rules],
        }
        arrays: dict[str, np.ndarray] = {
            "metadata": np.asarray(json.dumps(metadata)),
            "edge_indices": np.asarray(sorted(self._banks), dtype=int),
        }
        for edge, bank in self._banks.items():
            arrays[f"edge{edge}_orders"] = np.asarray(bank.orders, dtype=int)
            arrays[f"edge{edge}_weighted"] = bank.weighted_structure_constants
            arrays[f"edge{edge}_coefficients"] = bank.coefficients
            for internal_edge, values in enumerate(bank.internal_weights):
                arrays[f"edge{edge}_internal{internal_edge}"] = values
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)

    def load_banks(self, path: str | Path) -> None:
        """Load banks saved by :meth:`save_banks`, failing closed on mismatch."""
        path = Path(path)
        with np.load(path, allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata"].item()))
            expected = {
                "format": "torus-three-point-bank-v3-verified-c-recursion",
                "coefficient_basis": "native-sewing-v2",
                "coefficient_numerics": C_RECURSION_NUMERICS_VERSION,
                "t": self.t,
                "external_weights": list(self.external_weights),
                "high_order": self.high_order,
                "low_order": self.low_order,
                "adaptive_tolerance": self.adaptive_tolerance,
                "block_backend": self.block_backend,
                "momentum_nodes": [rule.nodes.tolist() for rule in self.momentum_rules],
                "momentum_weights": [rule.weights.tolist() for rule in self.momentum_rules],
            }
            if not cache_metadata_matches(metadata, expected):
                raise ValueError(f"cached block bank metadata mismatch in {path}")
            banks: dict[int, _ThreePointBlockBank] = {}
            edges = (
                tuple(int(edge) for edge in payload["edge_indices"])
                if "edge_indices" in payload
                else (0, 1, 2)
            )
            for edge in edges:
                banks[edge] = _ThreePointBlockBank(
                    orders=tuple(int(value) for value in payload[f"edge{edge}_orders"]),
                    internal_weights=tuple(
                        np.asarray(payload[f"edge{edge}_internal{internal_edge}"], dtype=float)
                        for internal_edge in range(3)
                    ),
                    weighted_structure_constants=np.asarray(
                        payload[f"edge{edge}_weighted"],
                        dtype=np.complex128,
                    ),
                    coefficients=np.asarray(
                        payload[f"edge{edge}_coefficients"],
                        dtype=np.complex128,
                    ),
                )
        self._banks = banks

    def adaptive_order(self, q_abs: float) -> int:
        """Use all available levels; a nome magnitude is not a remainder bound."""
        return self.high_order

    def correlator_from_logs(
        self,
        log_q_values: tuple[complex, complex, complex],
        *,
        order_cap: int | None = None,
        record_diagnostics: bool = True,
        cancel_eta_vacuum: bool = False,
        pade_orders: tuple[int, int] | None = None,
    ) -> complex:
        """Evaluate the nonchiral Liouville correlator for three cylinder logs.

        When ``cancel_eta_vacuum`` is true, use
        ``h-c/24=P^2-1/24 -> P^2`` in the primary propagation exponent.  The
        caller then supplies only the eta oscillator product.  Since the three
        necklace logs sum to ``2*pi*i*tau``, this exactly cancels the common
        Liouville ``exp(+pi*tau2/6)`` against the eta
        ``exp(-pi*tau2/6)`` before either exponential is formed.
        """
        if len(log_q_values) != 3:
            raise ValueError("three necklace cylinder logarithms are required")
        from genus1_sewing_domains import necklace_chart
        necklace_chart(log_q_values).require()
        controller=getattr(self,'spectral_controller',None)
        if controller is not None:
            return controller.integrate_necklace(self,log_q_values,lambda shadow:
                shadow.correlator_from_logs(log_q_values,order_cap=order_cap,
                    record_diagnostics=False,cancel_eta_vacuum=cancel_eta_vacuum,
                    pade_orders=pade_orders))
        if pade_orders is not None:
            raise ValueError("native-sewing evaluation uses the raw polynomial, not Pade resummation")
        q_values = tuple(cmath.exp(log_q) for log_q in log_q_values)
        magnitudes = np.asarray([abs(value) for value in q_values], dtype=float)
        descending = np.sort(magnitudes)[::-1]
        high_edge = int(np.argmax(magnitudes))
        selected_order = self.adaptive_order(float(magnitudes[high_edge]))
        if order_cap is not None:
            selected_order = min(selected_order, max(self.low_order, int(order_cap)))
        if record_diagnostics:
            self.order_histogram[selected_order] += 1
            self.high_edge_histogram[high_edge] += 1
            self.maximum_q_seen = max(self.maximum_q_seen, float(descending[0]))
            self.maximum_second_q_seen = max(
                self.maximum_second_q_seen,
                float(descending[1]),
            )

        if high_edge not in self._banks:
            self._banks[high_edge] = self._build_bank(high_edge)
        bank = self._banks[high_edge]
        slices = [slice(None)] * 4
        slices[1 + high_edge] = slice(0, selected_order + 1)
        coefficients = bank.coefficients[tuple(slices)]
        powers = []
        for edge, q_value in enumerate(q_values):
            order = selected_order if edge == high_edge else self.low_order
            powers.append(q_value ** np.arange(order + 1))
        descendants = np.einsum(
            "tijk,i,j,k->t", coefficients, powers[0], powers[1], powers[2],
            optimize=True,
        )
        primary_exponent = np.zeros(len(descendants), dtype=float)
        reference_weight = 1.0 if cancel_eta_vacuum else C_LIOUVILLE / 24.0
        for internal_weights, log_q in zip(bank.internal_weights, log_q_values):
            primary_exponent += 2.0 * (
                internal_weights - reference_weight
            ) * float(complex(log_q).real)
        primary_norm_squared = np.exp(primary_exponent)
        return complex(
            np.dot(
                bank.weighted_structure_constants,
                primary_norm_squared * np.abs(descendants) ** 2,
            )
        )


    def primary_cut_terms_from_logs(
        self,
        log_q_values: tuple[complex, complex, complex],
        *,
        cut_edge: int,
        order_cap: int | None = None,
        record_diagnostics: bool = False,
    ) -> PrimaryCutTerms:
        """Return the sphere-five-point primary cut of a necklace block.

        The descendant level on ``cut_edge`` is fixed to zero.  The remaining
        two axes are exactly the CCY sphere-five-point linear block obtained
        by cutting that edge; only propagation along the cut edge is absent.
        This method intentionally returns the individual long-momentum terms
        rather than their sum, because the analytic tube-length kernel
        depends on that momentum.
        """

        if len(log_q_values) != 3:
            raise ValueError("three necklace cylinder logarithms are required")
        cut_edge = int(cut_edge)
        if cut_edge not in (0, 1, 2):
            raise ValueError("cut_edge must be 0, 1, or 2")
        logs = tuple(complex(value) for value in log_q_values)
        if any(value.real >= 0.0 for value in logs):
            raise ValueError("all necklace propagation multipliers must contract")

        short_edges = ((cut_edge + 1) % 3, (cut_edge + 2) % 3)
        q_values = tuple(cmath.exp(value) for value in logs)
        high_edge = max(short_edges, key=lambda edge: abs(q_values[edge]))
        selected_high_order = self.adaptive_order(abs(q_values[high_edge]))
        if order_cap is not None:
            selected_high_order = min(
                selected_high_order,
                max(self.low_order, int(order_cap)),
            )
        if high_edge not in self._banks:
            self._banks[high_edge] = self._build_bank(high_edge)
        bank = self._banks[high_edge]

        selected_orders = [self.low_order, self.low_order, self.low_order]
        selected_orders[high_edge] = selected_high_order
        selected_orders[cut_edge] = 0
        slices = tuple(slice(0, order + 1) for order in selected_orders)
        coefficients = bank.coefficients[(slice(None),) + slices]
        powers = tuple(
            q_values[edge] ** np.arange(selected_orders[edge] + 1)
            for edge in range(3)
        )
        descendants = np.einsum(
            "tijk,i,j,k->t",
            coefficients,
            powers[0],
            powers[1],
            powers[2],
            optimize=True,
        )
        short_primary_exponent = np.zeros(len(descendants), dtype=float)
        for edge in short_edges:
            short_primary_exponent += (
                2.0
                * (bank.internal_weights[edge] - 1.0)
                * logs[edge].real
            )
        long_momenta_squared = bank.internal_weights[cut_edge] - 1.0
        if np.any(long_momenta_squared < -1.0e-13):
            raise ArithmeticError("a Liouville propagation momentum became non-real")
        long_momenta = np.sqrt(np.maximum(long_momenta_squared, 0.0))
        values = (
            bank.weighted_structure_constants
            * np.exp(short_primary_exponent)
            * np.abs(descendants) ** 2
        )
        if not np.all(np.isfinite(values)):
            raise FloatingPointError("the primary-cut five-point terms are non-finite")
        if record_diagnostics:
            self.order_histogram[selected_high_order] += 1
            self.high_edge_histogram[high_edge] += 1
            self.maximum_q_seen = max(
                self.maximum_q_seen,
                *(abs(q_values[edge]) for edge in short_edges),
            )
        return PrimaryCutTerms(
            cut_edge=cut_edge,
            short_edges=short_edges,
            long_momenta=np.asarray(long_momenta, dtype=float),
            values=np.asarray(values, dtype=np.complex128),
            selected_orders=tuple(selected_orders),
            maximum_short_sewing_parameter=float(
                max(abs(q_values[edge]) for edge in short_edges)
            ),
        )

    def diagnostics(self) -> dict[str, object]:
        return {
            "adaptive_order_histogram": {
                str(order): int(count) for order, count in self.order_histogram.items()
            },
            "largest_q_seen": float(self.maximum_q_seen),
            "largest_second_q_seen": float(self.maximum_second_q_seen),
            "high_edge_histogram": {
                str(edge): int(count) for edge, count in self.high_edge_histogram.items()
            },
            "upsilon_cache_size": int(len(self.special._log_cache)),
        }


def ordered_necklace_data(
    w1: complex,
    w2: complex,
    tau: complex,
) -> tuple[tuple[complex, complex], tuple[complex, complex, complex]]:
    """Sort the two outgoing punctures and return the three positive-height gaps."""
    tau = complex(tau)
    if tau.imag <= 0.0:
        raise ValueError("tau must lie in the upper half-plane")
    ordered = tuple(sorted((complex(w1), complex(w2)), key=lambda value: value.imag))
    first, second = ordered
    logs = (
        1.0j * first,
        1.0j * (second - first),
        1.0j * (2.0 * math.pi * tau - second),
    )
    if any(log_q.real >= 1.0e-13 for log_q in logs):
        raise ValueError("puncture sorting did not produce contracting necklace cylinders")
    return ordered, logs


def dedekind_eta_oscillator_abs_squared(
    tau: complex,
    tolerance: float = 1.0e-16,
) -> float:
    r"""Return ``|prod_(n>=1) (1-q^n)|^2`` for ``q=exp(2*pi*i*tau)``.

    This is the Dedekind eta norm with its ``|q|^(1/12)`` vacuum factor
    removed.  It is paired with ``cancel_eta_vacuum=True`` above.
    """
    tau = complex(tau)
    q = cmath.exp(2.0j * math.pi * tau)
    value = 1.0 + 0.0j
    q_power = q
    n = 1
    while abs(q_power) > tolerance and n < 10000:
        value *= 1.0 - q_power
        n += 1
        q_power *= q
    return float(abs(value) ** 2)


def dedekind_eta_log_abs_squared(
    tau: complex,
    tolerance: float = 1.0e-16,
) -> float:
    r"""Return ``log(|eta(tau)|^2)`` without forming an exponentially small eta."""
    tau = complex(tau)
    q = cmath.exp(2.0j * math.pi * tau)
    log_norm = -math.pi * tau.imag / 6.0
    q_power = q
    n = 1
    while abs(q_power) > tolerance and n < 10000:
        log_norm += 2.0 * math.log(abs(1.0 - q_power))
        n += 1
        q_power *= q
    return float(log_norm)


def _log_abs_sine(value: complex) -> float:
    """Stable ``log(abs(sin(value)))`` for large imaginary part."""
    value = complex(value)
    magnitude = abs(value.imag)
    if magnitude < 20.0:
        return math.log(abs(cmath.sin(value)))
    correction = 1.0 + math.exp(-4.0 * magnitude)
    correction -= 2.0 * math.cos(2.0 * value.real) * math.exp(-2.0 * magnitude)
    return magnitude - math.log(2.0) + 0.5 * math.log(correction)


def torus_prime_form_log_norm(
    z: complex,
    tau: complex,
    tolerance: float = 1.0e-16,
) -> float:
    """Return the logarithm of the single-valued torus prime-form norm.

    The ordinary evaluator is used throughout the compact domain.  Above
    ``tau2=40`` we first reduce the point to the centered torus cell and use
    the leading theta term.  The omitted relative correction is bounded by
    ``O(exp(-pi*tau2))`` there, while evaluating the sine in logarithmic form
    avoids overflow.
    """
    z = complex(z)
    tau = complex(tau)
    if tau.imag <= 0.0:
        raise ValueError("tau must lie in the upper half-plane")
    if tau.imag <= 40.0:
        norm = torus_prime_form_norm(z, tau, tolerance=tolerance)
        if norm <= 0.0 or not math.isfinite(norm):
            raise FloatingPointError("prime-form norm is not positive and finite")
        return math.log(norm)

    vertical_coordinate = z.imag / (2.0 * math.pi * tau.imag)
    vertical_shift = math.floor(vertical_coordinate + 0.5)
    centered = z - 2.0 * math.pi * vertical_shift * tau
    horizontal_coordinate = centered.real / (2.0 * math.pi)
    horizontal_shift = math.floor(horizontal_coordinate + 0.5)
    centered -= 2.0 * math.pi * horizontal_shift
    w = centered / (2.0 * math.pi)
    log_holomorphic_norm = math.log(2.0) + _log_abs_sine(math.pi * w)
    log_gaussian = -(centered.imag**2) / (4.0 * math.pi * tau.imag)
    return float(log_holomorphic_norm + log_gaussian)


def reduced_worldsheet_integrand_three_point(
    correlator: LiouvilleTorusThreePointNecklace,
    w1: complex,
    w2: complex,
    tau: complex,
    *,
    order_cap: int | None = None,
    record_diagnostics: bool = True,
    pade_orders: tuple[int, int] | None = None,
) -> complex:
    """Return ``|eta|^2 K_X0 G_L/sqrt(tau2)`` in native normalization."""
    tau = complex(tau)
    ordered, logs = ordered_necklace_data(w1, w2, tau)
    first, second = ordered
    log_prime_first = torus_prime_form_log_norm(first, tau)
    log_prime_second = torus_prime_form_log_norm(second, tau)
    log_prime_pair = torus_prime_form_log_norm(first - second, tau)

    omega_in = correlator.omega_in
    omega_out = correlator.omega_out
    koba_nielsen = cmath.exp(
        omega_in * omega_out * log_prime_first
        + omega_in * omega_out * log_prime_second
        - omega_out * omega_out * log_prime_pair
    )
    liouville_without_vacuum = correlator.correlator_from_logs(
        logs,
        order_cap=order_cap,
        record_diagnostics=record_diagnostics,
        cancel_eta_vacuum=True,
        pade_orders=pade_orders,
    )
    return (
        dedekind_eta_oscillator_abs_squared(tau)
        * koba_nielsen
        * liouville_without_vacuum
        / math.sqrt(tau.imag)
    )
