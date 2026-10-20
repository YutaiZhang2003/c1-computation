"""Scalar partition functions and their truncation checks in a fixed plumbing frame."""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
import numpy as np
from ccy_plumbing_conventions import validate_genus2_plumbing_coordinates
from conformal_frame_labels import GLASSES_PLUMBING_FRAME, THETA_PLUMBING_FRAME
from free_boson_plumbing import free_boson_chiral_log_factor, glasses_free_boson_powered_convergence, glasses_free_boson_product, noncompact_scalar_zero_mode_factor, theta_free_boson_powered_convergence, theta_free_boson_product
from plumbing_algorithms import theta_cusp_surviving_multipliers



@dataclass(frozen=True)
class NoncompactScalarEvaluation:
    """Scalar value together with its unamplified and powered tail estimates."""

    partition: float
    oscillator_mode_chiral_log_tail_estimate: float
    primitive_word_chiral_log_tail_estimate: float | None
    primitive_count: int
    effective_word_length: int

    def powered_log_error_estimate(self, power: float = 25.0) -> float | None:
        if self.primitive_word_chiral_log_tail_estimate is None:
            return None
        chiral_tail = (
            self.oscillator_mode_chiral_log_tail_estimate
            + self.primitive_word_chiral_log_tail_estimate
        )
        return float(2.0 * abs(float(power)) * chiral_tail)

    def powered_relative_error_estimate(self, power: float = 25.0) -> float | None:
        log_error = self.powered_log_error_estimate(power)
        if log_error is None:
            return None
        try:
            return float(math.expm1(log_error))
        except OverflowError:
            return math.inf

    def legacy_tuple(self) -> tuple[float, float, int]:
        """Return the historical ``(partition, mode tail, primitive count)`` API."""

        return (
            self.partition,
            self.oscillator_mode_chiral_log_tail_estimate,
            self.primitive_count,
        )


@dataclass(frozen=True)
class NoncompactScalarFixedCapStep:
    """One direct fixed-word-cap comparison for the scalar quotient.

    ``quotient_powered_relative_change`` compares the scalar contribution to
    ``Z_L / Z_X**power`` at this cap with the preceding fixed cap.  It is
    computed in logarithms, so a small scalar is never divided by another
    small scalar.  The empirical primitive-word tail is retained only as a
    diagnostic and is deliberately absent from ``direct_pass``.
    """

    word_length: int
    evaluation: NoncompactScalarEvaluation
    quotient_powered_relative_change: float | None
    oscillator_powered_relative_bound: float
    direct_pass: bool
    consecutive_passes: int

    def as_record(self) -> dict[str, object]:
        return {
            "word_length": self.word_length,
            "partition": self.evaluation.partition,
            "oscillator_chiral_log_tail": (
                self.evaluation.oscillator_mode_chiral_log_tail_estimate
            ),
            "primitive_word_chiral_log_tail_empirical": (
                self.evaluation.primitive_word_chiral_log_tail_estimate
            ),
            "primitive_count": self.evaluation.primitive_count,
            "quotient_powered_relative_change": (
                self.quotient_powered_relative_change
            ),
            "oscillator_powered_relative_bound": (
                self.oscillator_powered_relative_bound
            ),
            "direct_pass": self.direct_pass,
            "consecutive_passes": self.consecutive_passes,
        }


@dataclass(frozen=True)
class NoncompactScalarFixedCapConvergence:
    """Faithful fixed-cap convergence result for the scalar contribution."""

    evaluation: NoncompactScalarEvaluation
    steps: tuple[NoncompactScalarFixedCapStep, ...]
    converged: bool
    required_confirmations: int
    powered_relative_tolerance: float
    power: float

    @property
    def terminal_powered_relative_change(self) -> float | None:
        return self.steps[-1].quotient_powered_relative_change

    def records(self) -> list[dict[str, object]]:
        return [step.as_record() for step in self.steps]


def evaluate_noncompact_scalar_detailed(
    topology: str,
    q_values: Sequence[complex] | Mapping[str, complex],
    omega_chart: np.ndarray,
    *,
    log_q_values: Sequence[complex] | Mapping[str, complex] | None = None,
    word_length: int,
    max_mode: int,
    tolerance: float,
    word_tail_tolerance: float | None = None,
    minimum_word_length: int = 5,
    powered_error_exponent: float = 25.0,
    powered_relative_tolerance: float | None = None,
) -> NoncompactScalarEvaluation:
    if not math.isfinite(float(powered_error_exponent)):
        raise ValueError("powered_error_exponent must be finite")
    if powered_relative_tolerance is not None and (
        not math.isfinite(float(powered_relative_tolerance))
        or float(powered_relative_tolerance) <= 0.0
    ):
        raise ValueError("powered_relative_tolerance must be finite and positive")
    coordinates = validate_genus2_plumbing_coordinates(
        topology,
        q_values,
        log_q_values=log_q_values,
    )
    topology = coordinates.channel
    q_tuple = coordinates.q_values
    log_q_tuple = (
        coordinates.log_q_values
        if log_q_values is not None
        else None
    )
    surviving_multipliers = (
        theta_cusp_surviving_multipliers(
            *q_tuple,
            log_q_values=log_q_tuple,
            threshold=1.0e-12,
        )
        if topology == "theta" and log_q_tuple is not None
        else None
    )
    if surviving_multipliers is not None:
        oscillator_terms = tuple(
            free_boson_chiral_log_factor(
                multiplier,
                max_mode=max_mode,
                tolerance=tolerance,
            )
            for multiplier in surviving_multipliers
        )
        oscillator = math.exp(
            2.0 * math.fsum(term[0].real for term in oscillator_terms)
        )
        full_partition = oscillator * noncompact_scalar_zero_mode_factor(omega_chart)
        tail = float(math.fsum(term[1] for term in oscillator_terms))
        evaluation = NoncompactScalarEvaluation(
            partition=float(full_partition),
            oscillator_mode_chiral_log_tail_estimate=tail,
            primitive_word_chiral_log_tail_estimate=0.0,
            primitive_count=len(surviving_multipliers),
            effective_word_length=0,
        )
        powered_relative = evaluation.powered_relative_error_estimate(
            powered_error_exponent
        )
        if powered_relative_tolerance is not None and (
            powered_relative is None
            or powered_relative > float(powered_relative_tolerance)
        ):
            raise RuntimeError(
                f"{topology} scalar exceeds the powered relative-error target: "
                f"estimate={powered_relative!r}, "
                f"target={float(powered_relative_tolerance):.3e}"
            )
        return evaluation
    if topology == "theta":
        product = theta_free_boson_product(
            *q_tuple,
            max_word_length=word_length,
            max_mode=max_mode,
            tolerance=tolerance,
            word_tail_tolerance=word_tail_tolerance,
            minimum_word_length=minimum_word_length,
        )
        frame = THETA_PLUMBING_FRAME
    elif topology == "glasses":
        product = glasses_free_boson_product(
            *q_tuple,
            max_word_length=word_length,
            max_mode=max_mode,
            tolerance=tolerance,
            word_tail_tolerance=word_tail_tolerance,
            minimum_word_length=minimum_word_length,
        )
        frame = GLASSES_PLUMBING_FRAME
    else:
        raise ValueError(f"unsupported topology {topology!r}")
    if (
        product.channel != topology
        or tuple(product.edge_names) != coordinates.edge_names
        or tuple(product.q_values) != q_tuple
    ):
        raise AssertionError(
            "scalar result changed the genus-two channel, edge names, or ordered q tuple"
        )
    if word_tail_tolerance is not None and (
        product.primitive_word_tail_estimate is None
        or product.primitive_word_tail_estimate > float(word_tail_tolerance)
    ):
        raise RuntimeError(
            f"{topology} scalar exhausted its word-length safety cap before "
            "reaching the requested primitive-word tail"
        )
    full_partition = product.nonchiral_value * noncompact_scalar_zero_mode_factor(omega_chart)
    if not math.isfinite(full_partition) or full_partition <= 0.0:
        raise ValueError("noncompact scalar plumbing partition is not positive and finite")
    evaluation = NoncompactScalarEvaluation(
        partition=float(full_partition),
        oscillator_mode_chiral_log_tail_estimate=float(
            product.oscillator_mode_tail_estimate
        ),
        primitive_word_chiral_log_tail_estimate=(
            None
            if product.primitive_word_tail_estimate is None
            else float(product.primitive_word_tail_estimate)
        ),
        primitive_count=int(product.primitive_count),
        effective_word_length=int(product.max_word_length),
    )
    powered_relative = evaluation.powered_relative_error_estimate(
        powered_error_exponent
    )
    if powered_relative_tolerance is not None and (
        powered_relative is None
        or powered_relative > float(powered_relative_tolerance)
    ):
        raise RuntimeError(
            f"{topology} scalar exceeds the powered relative-error target: "
            f"estimate={powered_relative!r}, "
            f"target={float(powered_relative_tolerance):.3e}"
        )
    return evaluation


def evaluate_noncompact_scalar_fixed_cap_convergence(
    topology: str,
    q_values: Sequence[complex] | Mapping[str, complex],
    omega_chart: np.ndarray,
    *,
    log_q_values: Sequence[complex] | Mapping[str, complex] | None = None,
    initial_word_length: int = 8,
    maximum_word_length: int = 12,
    word_length_increment: int = 2,
    max_mode: int = 40,
    tolerance: float = 1.0e-14,
    powered_error_exponent: float = 25.0,
    powered_relative_tolerance: float = 2.0e-4,
    confirmation_steps: int = 2,
) -> NoncompactScalarFixedCapConvergence:
    """Stream a genus-two scalar through direct successive fixed-cap values.

    Primitive shells are generated only until the requested powered
    observable passes the successive-cap gate.  No words beyond that terminal
    checkpoint are enumerated.

    Production convergence is based only on direct changes in
    ``Z_X**(-powered_error_exponent)`` at successive fixed word caps, plus the
    rigorous oscillator-mode tail for the enumerated words.  The geometric
    primitive-shell tail remains in the records as an empirical diagnostic;
    it is never an acceptance condition.
    """

    initial = int(initial_word_length)
    maximum = int(maximum_word_length)
    increment = int(word_length_increment)
    confirmations = int(confirmation_steps)
    power = float(powered_error_exponent)
    relative_tolerance = float(powered_relative_tolerance)
    if initial <= 0 or maximum < initial:
        raise ValueError("scalar fixed word caps must be positive and ordered")
    if increment <= 0:
        raise ValueError("scalar word-length increment must be positive")
    if (maximum - initial) % increment:
        raise ValueError("maximum scalar word cap must lie on the fixed-cap scan")
    if confirmations <= 0:
        raise ValueError("scalar confirmation steps must be positive")
    if maximum - initial < confirmations * increment:
        raise ValueError(
            "scalar fixed-cap scan does not contain enough direct comparisons "
            "for the requested confirmations"
        )
    if not math.isfinite(power):
        raise ValueError("powered_error_exponent must be finite")
    if not math.isfinite(relative_tolerance) or relative_tolerance <= 0.0:
        raise ValueError("powered_relative_tolerance must be finite and positive")

    coordinates = validate_genus2_plumbing_coordinates(
        topology,
        q_values,
        log_q_values=log_q_values,
    )
    topology = coordinates.channel
    q_tuple = coordinates.q_values
    zero_mode = noncompact_scalar_zero_mode_factor(omega_chart)
    if topology == "theta" and log_q_values is not None:
        surviving = theta_cusp_surviving_multipliers(
            *q_tuple,
            log_q_values=coordinates.log_q_values,
            threshold=1.0e-12,
        )
        if surviving is not None:
            evaluation = evaluate_noncompact_scalar_detailed(
                topology,
                q_tuple,
                omega_chart,
                log_q_values=coordinates.log_q_values,
                word_length=initial,
                max_mode=max_mode,
                tolerance=tolerance,
                powered_error_exponent=power,
                powered_relative_tolerance=relative_tolerance,
            )
            oscillator_bound = math.expm1(
                2.0
                * abs(power)
                * evaluation.oscillator_mode_chiral_log_tail_estimate
            )
            return NoncompactScalarFixedCapConvergence(
                evaluation=evaluation,
                steps=(
                    NoncompactScalarFixedCapStep(
                        word_length=evaluation.effective_word_length,
                        evaluation=evaluation,
                        quotient_powered_relative_change=0.0,
                        oscillator_powered_relative_bound=oscillator_bound,
                        direct_pass=True,
                        consecutive_passes=confirmations,
                    ),
                ),
                converged=True,
                required_confirmations=confirmations,
                powered_relative_tolerance=relative_tolerance,
                power=power,
            )

    convergence_function = (
        theta_free_boson_powered_convergence
        if topology == "theta"
        else glasses_free_boson_powered_convergence
    )
    streamed = convergence_function(
        *q_tuple,
        initial_word_length=initial,
        maximum_word_length=maximum,
        word_length_increment=increment,
        max_mode=int(max_mode),
        tolerance=float(tolerance),
        power=power,
        powered_relative_tolerance=relative_tolerance,
        confirmation_steps=confirmations,
    )
    steps = tuple(
        NoncompactScalarFixedCapStep(
            word_length=source.word_length,
            evaluation=NoncompactScalarEvaluation(
                partition=float(source.nonchiral_value * zero_mode),
                oscillator_mode_chiral_log_tail_estimate=(
                    source.oscillator_mode_tail_estimate
                ),
                primitive_word_chiral_log_tail_estimate=(
                    source.primitive_word_tail_estimate
                ),
                primitive_count=source.primitive_count,
                effective_word_length=source.word_length,
            ),
            quotient_powered_relative_change=source.powered_relative_change,
            oscillator_powered_relative_bound=(
                source.oscillator_powered_relative_bound
            ),
            direct_pass=source.direct_pass,
            consecutive_passes=source.consecutive_passes,
        )
        for source in streamed.steps
    )
    terminal_evaluation = steps[-1].evaluation
    direct_terminal = streamed.product.nonchiral_value * zero_mode
    if terminal_evaluation.partition != direct_terminal:
        raise AssertionError("streamed scalar terminal value changed during wrapping")
    return NoncompactScalarFixedCapConvergence(
        evaluation=terminal_evaluation,
        steps=steps,
        converged=streamed.converged,
        required_confirmations=confirmations,
        powered_relative_tolerance=relative_tolerance,
        power=power,
    )


def evaluate_glasses_noncompact_scalar_fixed_cap_convergence(
    q_values: Sequence[complex] | Mapping[str, complex],
    omega_chart: np.ndarray,
    **kwargs: Any,
) -> NoncompactScalarFixedCapConvergence:
    """Backward-compatible glasses wrapper for streamed scalar convergence."""

    return evaluate_noncompact_scalar_fixed_cap_convergence(
        "glasses",
        q_values,
        omega_chart,
        **kwargs,
    )


def evaluate_noncompact_scalar(
    topology: str,
    q_values: Sequence[complex] | Mapping[str, complex],
    omega_chart: np.ndarray,
    *,
    log_q_values: Sequence[complex] | Mapping[str, complex] | None = None,
    word_length: int,
    max_mode: int,
    tolerance: float,
    word_tail_tolerance: float | None = None,
    minimum_word_length: int = 5,
) -> tuple[float, float, int]:
    """Backward-compatible scalar evaluator without detailed error metadata."""

    return evaluate_noncompact_scalar_detailed(
        topology,
        q_values,
        omega_chart,
        log_q_values=log_q_values,
        word_length=word_length,
        max_mode=max_mode,
        tolerance=tolerance,
        word_tail_tolerance=word_tail_tolerance,
        minimum_word_length=minimum_word_length,
    ).legacy_tuple()


def scalar_word_safety_cap(
    q_values: Sequence[complex],
    *,
    base_cap: int,
    large_q_cap: int,
    large_q_threshold: float,
) -> int:
    """Select the higher adaptive word cap only for large plumbing coordinates."""

    base = int(base_cap)
    large = int(large_q_cap)
    threshold = float(large_q_threshold)
    if base <= 0 or large < base:
        raise ValueError("scalar word caps must be positive with large_q_cap >= base_cap")
    if not math.isfinite(threshold) or not 0.0 < threshold < 1.0:
        raise ValueError("scalar large-q threshold must lie in (0,1)")
    if len(q_values) != 3:
        raise ValueError("genus-two scalar cap selection requires three q values")
    return large if max(abs(complex(value)) for value in q_values) >= threshold else base

