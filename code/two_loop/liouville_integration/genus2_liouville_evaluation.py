"""Liouville integration in the theta and glasses channels, including logarithmic scaling."""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Mapping, Sequence
from ccy_plumbing_conventions import validate_genus2_plumbing_coordinates
from liouville_genus2_ccy import liouville_genus2_ccy_partition
from liouville_genus2_glasses import liouville_genus2_glasses_partition



@dataclass(frozen=True)
class RescaledLiouvillePartition:
    """Liouville partition with the common ``|q|^(2 h_min)`` kept in logs."""

    scaled_partition: float
    log_threshold_factor: float
    channel: str | None = None
    edge_names: tuple[str, str, str] = ()
    q_values: tuple[complex, complex, complex] = ()

    @property
    def log_partition(self) -> float:
        return math.log(self.scaled_partition) + self.log_threshold_factor

    @property
    def raw_partition(self) -> float:
        try:
            return math.exp(self.log_partition)
        except OverflowError:
            return math.inf


def _positive_real_partition(value: complex, name: str) -> float:
    value = complex(value)
    scale = max(abs(value.real), 1.0e-300)
    if abs(value.imag) > 1.0e-8 * scale:
        raise ValueError(f"{name} has a significant imaginary part: {value!r}")
    if not math.isfinite(value.real) or value.real <= 0.0:
        raise ValueError(f"{name} must be positive and finite, got {value!r}")
    return float(value.real)


def evaluate_liouville_rescaled(
    topology: str,
    q_values: Sequence[complex] | Mapping[str, complex],
    *,
    log_q_values: Sequence[complex] | Mapping[str, complex] | None = None,
    block_order: int,
    quadrature_order: int | Sequence[int],
    quadrature_scheme: str,
    dps: int,
    vacuum_word_length: int,
    vacuum_oscillator_level: int,
    vacuum_word_tail_tolerance: float | None = None,
    vacuum_minimum_word_length: int = 5,
    resum_global_block: bool = True,
    global_block_tolerance: float = 1.0e-13,
    global_block_max_outer_order: int = 64,
) -> RescaledLiouvillePartition:
    coordinates = validate_genus2_plumbing_coordinates(
        topology,
        q_values,
        log_q_values=log_q_values,
    )
    topology = coordinates.channel
    q_tuple = coordinates.q_values
    edge_names = coordinates.edge_names
    threshold = 1.0  # h_min=Q^2/4 at b=1.
    log_q_tuple = (
        coordinates.log_q_values
        if log_q_values is not None
        else None
    )
    common = {
        "b": 1.0,
        "block_order": int(block_order),
        "quadrature_order": (
            int(quadrature_order)
            if isinstance(quadrature_order, int)
            else tuple(int(order) for order in quadrature_order)
        ),
        "quadrature_scheme": quadrature_scheme,
        "dps": int(dps),
        # Numerical factorization only: the exact factor is restored through
        # log_threshold_factor before the period-coordinate density is used.
        "propagator_shift": threshold,
        "include_vacuum_seed": True,
        "vacuum_word_len": int(vacuum_word_length),
        "vacuum_oscillator_level_max": int(vacuum_oscillator_level),
        "vacuum_word_tail_tolerance": vacuum_word_tail_tolerance,
        "vacuum_minimum_word_length": int(vacuum_minimum_word_length),
        "include_cosmological_prefactor": False,
        "store_samples": False,
        "log_q_values": log_q_tuple,
        # Production CFT data always uses the all-level global descendant
        # resummation; block_order truncates only the finite-c pole recursion.
        "resum_global_block": bool(resum_global_block),
        "global_block_tolerance": float(global_block_tolerance),
    }
    if topology == "theta":
        result = liouville_genus2_ccy_partition(
            q1=q_tuple[0],
            q2=q_tuple[1],
            q3=q_tuple[2],
            global_block_max_outer_order=int(global_block_max_outer_order),
            **common,
        )
    elif topology == "glasses":
        result = liouville_genus2_glasses_partition(
            q_left=q_tuple[0],
            q_right=q_tuple[1],
            q_bridge=q_tuple[2],
            **common,
        )
    else:
        raise ValueError(f"unsupported topology {topology!r}")
    if (
        result.channel != topology
        or tuple(result.edge_names) != edge_names
        or tuple(result.q_values) != q_tuple
    ):
        raise AssertionError(
            "Liouville result changed the genus-two channel or ordered q tuple"
        )
    scaled = _positive_real_partition(result.value, "rescaled Liouville partition")
    if log_q_tuple is None:
        log_threshold = 2.0 * threshold * sum(
            math.log(abs(value)) for value in q_tuple
        )
    else:
        log_threshold = 2.0 * threshold * sum(value.real for value in log_q_tuple)
    return RescaledLiouvillePartition(
        scaled_partition=scaled,
        log_threshold_factor=log_threshold,
        channel=topology,
        edge_names=edge_names,
        q_values=q_tuple,
    )


def evaluate_liouville(
    topology: str,
    q_values: Sequence[complex] | Mapping[str, complex],
    *,
    log_q_values: Sequence[complex] | Mapping[str, complex] | None = None,
    block_order: int,
    quadrature_order: int,
    quadrature_scheme: str,
    dps: int,
    vacuum_word_length: int,
    vacuum_oscillator_level: int,
    vacuum_word_tail_tolerance: float | None = None,
    vacuum_minimum_word_length: int = 5,
) -> float:
    """Return the raw partition when it is representable as a float."""

    result = evaluate_liouville_rescaled(
        topology,
        q_values,
        log_q_values=log_q_values,
        block_order=block_order,
        quadrature_order=quadrature_order,
        quadrature_scheme=quadrature_scheme,
        dps=dps,
        vacuum_word_length=vacuum_word_length,
        vacuum_oscillator_level=vacuum_oscillator_level,
        vacuum_word_tail_tolerance=vacuum_word_tail_tolerance,
        vacuum_minimum_word_length=vacuum_minimum_word_length,
    )
    raw = result.raw_partition
    if not math.isfinite(raw) or raw <= 0.0:
        raise ValueError(
            "raw Liouville partition is outside floating-point range; "
            "use evaluate_liouville_rescaled"
        )
    return raw

