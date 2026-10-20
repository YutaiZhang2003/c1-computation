#!/usr/bin/env python3
"""Genus-two vacuum Virasoro block tools.

This implements the plumbing-frame c -> infinity vacuum block described in
Cho-Collier-Yin, arXiv:1703.09805, section 5:

    prod_{unoriented primitive C} prod_{n>=2} (1 - q_C**n)^(-1),

where q_C is the attracting multiplier of the primitive Schottky conjugacy
class C.  This is equivalent to CCY's exponent ``-1/2`` when C runs over
oriented conjugacy classes: the enumerator below already identifies a word
with its inverse.  The same evaluator is exposed for the genus-two glasses and
sunrise plumbing channels using the Schottky generators in
plumbing_algorithms.py.

The module also contains a direct finite-c descendant sum for the theta
pair-of-pants sewing frame of equation (5.5).  Those theta sewing variables are
not automatically identified with the glasses/sunrise Schottky parameters.
"""

from __future__ import annotations

from c1_paths import PACKAGE_ROOT, data_root, package_path, module_path

import argparse
import cmath
import math
import warnings
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Sequence

import numpy as np

try:
    from virasoro_descendant_algebra import (
        Descendant,
        State,
        act_virasoro_mode,
        descendant_inner_product,
        descendant_level,
        integer_partitions,
        normal_order_negative_word,
        prepend_negative_mode,
        project_vacuum_state,
        vacuum_descendant_basis,
        vacuum_gram_matrix,
    )
except ImportError:  # pragma: no cover - supports package-style execution
    from virasoro_descendant_algebra import (
        Descendant,
        State,
        act_virasoro_mode,
        descendant_inner_product,
        descendant_level,
        integer_partitions,
        normal_order_negative_word,
        prepend_negative_mode,
        project_vacuum_state,
        vacuum_descendant_basis,
        vacuum_gram_matrix,
    )

try:
    from plumbing_algorithms import (
        GeneratorData,
        Mobius,
        generator_data_from_mobius,
        generators_for_glasses,
        generators_for_sunrise,
    )
except ImportError:  # pragma: no cover - supports package-style execution
    from plumbing_algorithms import (
        GeneratorData,
        Mobius,
        generator_data_from_mobius,
        generators_for_glasses,
        generators_for_sunrise,
    )


Word = tuple[int, ...]


@dataclass(frozen=True)
class PrimitiveClassContribution:
    """One primitive Schottky conjugacy class contribution."""

    word: Word
    multiplier: complex
    log_factor: complex


@dataclass(frozen=True)
class PrimitiveWordTailDiagnostic:
    """Empirical convergence data for successive primitive-word shells.

    ``shell_abs_log_contributions[L-1]`` is the sum of absolute chiral
    logarithmic contributions from primitive words of length ``L``.  The tail
    estimate fits a guarded geometric envelope to the last few nonzero
    shells.  It is a useful convergence diagnostic, not a rigorous Schottky
    transfer-operator bound.
    """

    shell_abs_log_contributions: tuple[float, ...]
    recent_shell_ratios: tuple[float, ...]
    shell_group_size: int | None
    guarded_ratio: float | None
    estimated_omitted_abs_log: float | None
    empirically_convergent: bool


def primitive_word_tail_diagnostic(
    contributions: Sequence[PrimitiveClassContribution],
    max_word_length: int,
    *,
    ratio_window: int = 3,
    ratio_safety_factor: float = 1.25,
) -> PrimitiveWordTailDiagnostic:
    """Estimate the omitted primitive-word tail from absolute shell norms.

    If ``A_L`` is the absolute chiral log contribution of all primitive words
    of length ``L``, the estimate uses a one-shell or two-shell grouped
    geometric envelope.  The two-shell candidate handles the even/odd
    oscillation seen in some plumbing markings.  For the selected grouping it
    estimates

        G_J * r / (1-r),

    where ``G_J`` is the last one- or two-shell group and ``r`` is a
    safety-factor enlargement of the largest recent grouped-shell ratio.
    Returning ``None`` means that too few shells were supplied or that the
    observed shell envelope has not begun to contract.
    """

    maximum = int(max_word_length)
    window = int(ratio_window)
    safety = float(ratio_safety_factor)
    if maximum < 0:
        raise ValueError("max_word_length must be nonnegative")
    if window <= 0:
        raise ValueError("ratio_window must be positive")
    if not math.isfinite(safety) or safety < 1.0:
        raise ValueError("ratio_safety_factor must be finite and at least one")

    shell_norms = [0.0] * maximum
    for contribution in contributions:
        length = len(contribution.word)
        if 1 <= length <= maximum:
            shell_norms[length - 1] += abs(complex(contribution.log_factor))

    candidates: list[
        tuple[float, int, tuple[float, ...], float, float]
    ] = []
    fallback_ratios: tuple[float, ...] = ()
    fallback_group_size: int | None = None
    fallback_guarded_ratio: float | None = None
    for group_size in (1, 2):
        group_count = maximum // group_size
        if group_count < 3:
            continue
        start = maximum - group_count * group_size
        groups = tuple(
            sum(shell_norms[index : index + group_size])
            for index in range(start, maximum, group_size)
        )
        # Deep degeneration can drive already-computed terminal shells below
        # floating-point range.  Fit the geometric envelope to the last
        # positive shells and propagate it across those terminal zero groups
        # before estimating the first genuinely omitted group.
        positive_count = len(groups)
        while positive_count > 0 and groups[positive_count - 1] == 0.0:
            positive_count -= 1
        fitted_groups = groups[:positive_count]
        terminal_zero_groups = len(groups) - positive_count
        ratio_count = min(window, len(fitted_groups) - 1)
        if ratio_count <= 0:
            continue
        recent = fitted_groups[-(ratio_count + 1) :]
        if any(not math.isfinite(value) or value <= 0.0 for value in recent):
            continue
        ratios = tuple(
            right / left for left, right in zip(recent, recent[1:])
        )
        guarded_ratio = safety * max(ratios)
        if group_size == 1:
            fallback_ratios = ratios
            fallback_group_size = group_size
            fallback_guarded_ratio = float(guarded_ratio)
        if not math.isfinite(guarded_ratio) or guarded_ratio >= 1.0:
            continue
        tail = (
            recent[-1]
            * guarded_ratio ** (terminal_zero_groups + 1)
            / (1.0 - guarded_ratio)
        )
        ratio_spread = max(ratios) / min(ratios)
        candidates.append(
            (
                float(ratio_spread),
                group_size,
                ratios,
                float(guarded_ratio),
                float(tail),
            )
        )

    if not candidates:
        return PrimitiveWordTailDiagnostic(
            shell_abs_log_contributions=tuple(shell_norms),
            recent_shell_ratios=fallback_ratios,
            shell_group_size=fallback_group_size,
            guarded_ratio=fallback_guarded_ratio,
            estimated_omitted_abs_log=None,
            empirically_convergent=False,
        )

    _, group_size, ratios, guarded_ratio, tail = min(
        candidates,
        key=lambda item: (item[0], item[1]),
    )
    return PrimitiveWordTailDiagnostic(
        shell_abs_log_contributions=tuple(shell_norms),
        recent_shell_ratios=ratios,
        shell_group_size=group_size,
        guarded_ratio=float(guarded_ratio),
        estimated_omitted_abs_log=float(tail),
        empirically_convergent=True,
    )


@dataclass(frozen=True)
class VacuumBlockResult:
    """Truncated Schottky vacuum block product."""

    channel: str
    q_values: tuple[complex, ...] | None
    max_word_length: int
    max_mode: int
    tolerance: float
    log_value: complex
    value: complex
    primitive_count: int
    oscillator_mode_tail_estimate: float
    primitive_word_tail_estimate: float | None
    truncation_certified: bool
    contributions: tuple[PrimitiveClassContribution, ...]

    @property
    def primitive_word_convergence(self) -> PrimitiveWordTailDiagnostic:
        """Return the shell diagnostic used for the empirical word tail."""

        return primitive_word_tail_diagnostic(
            self.contributions,
            self.max_word_length,
        )

    @property
    def omitted_estimate(self) -> float:
        """Deprecated alias for the oscillator-mode tail only.

        This value excludes all primitive conjugacy classes longer than
        ``max_word_length`` and must not be treated as a bound on the full
        Schottky product.
        """

        return self.oscillator_mode_tail_estimate


@dataclass(frozen=True)
class DirectVacuumBlockResult:
    """Finite-c direct descendant-sum genus-two vacuum block."""

    channel: str
    q_values: tuple[complex, complex, complex]
    central_charge: complex
    max_level: int
    value: complex
    level_contributions: dict[tuple[int, int, int], complex]


def parse_complex(value: str) -> complex:
    return complex(value.replace("i", "j"))


def format_complex(value: complex) -> str:
    return f"{value.real:+.12e}{value.imag:+.12e}j"


def inverse_letter(letter: int) -> int:
    return letter ^ 1


def inverse_word(word: Word) -> Word:
    return tuple(inverse_letter(letter) for letter in reversed(word))


def is_reduced_word(word: Word) -> bool:
    return all(word[idx] != inverse_letter(word[idx + 1]) for idx in range(len(word) - 1))


def is_cyclically_reduced_word(word: Word) -> bool:
    return bool(word) and is_reduced_word(word) and word[-1] != inverse_letter(word[0])


def cyclic_rotations(word: Word) -> list[Word]:
    return [word[idx:] + word[:idx] for idx in range(len(word))]


def primitive_root_period(word: Word) -> int:
    """Return the shortest repeated period of a cyclic word."""

    length = len(word)
    for period in range(1, length):
        if length % period:
            continue
        root = word[:period]
        if root * (length // period) == word:
            return period
    return length


def is_primitive_cyclic_word(word: Word) -> bool:
    """Return true if a cyclic word is not a nontrivial power."""

    if not word:
        return False
    # Being a nontrivial power is invariant under cyclic rotation.  Testing
    # every rotation therefore repeats the same exact predicate ``len(word)``
    # times and was the main cost of the Schottky word catalogue.
    return primitive_root_period(word) == len(word)


def minimal_cyclic_rotation(word: Word) -> Word:
    """Return the lexicographically least cyclic rotation in linear time.

    This is Booth's algorithm specialized to tuples of integer letters.  It
    is exactly equivalent to ``min(cyclic_rotations(word))``; it changes only
    the cost of finding the canonical representative, not the representative
    or the set of primitive conjugacy classes.
    """

    if not word:
        return ()
    length = len(word)
    doubled = word + word
    first = 0
    second = 1
    offset = 0
    while first < length and second < length and offset < length:
        left = doubled[first + offset]
        right = doubled[second + offset]
        if left == right:
            offset += 1
            continue
        if left > right:
            first = first + offset + 1
            if first <= second:
                first = second + 1
        else:
            second = second + offset + 1
            if second <= first:
                second = first + 1
        offset = 0
    start = min(first, second)
    return tuple(doubled[start : start + length])


def canonical_primitive_key(word: Word) -> Word:
    """Canonical representative modulo cyclic rotation and inversion."""

    return min(
        minimal_cyclic_rotation(word),
        minimal_cyclic_rotation(inverse_word(word)),
    )


@lru_cache(maxsize=None)
def _primitive_conjugacy_word_shell_cached(
    num_generators: int,
    length: int,
) -> tuple[Word, ...]:
    """Cached implementation of one primitive-word shell."""

    rank = int(num_generators)
    target_length = int(length)
    if rank <= 0:
        raise ValueError("num_generators must be positive")
    if target_length <= 0:
        return ()

    letters = tuple(range(2 * rank))
    out: list[Word] = []

    def extend(prefix: Word) -> None:
        if len(prefix) == target_length:
            if not is_cyclically_reduced_word(prefix):
                return
            if not is_primitive_cyclic_word(prefix):
                return
            key = canonical_primitive_key(prefix)
            # The recursive traversal visits every reduced word once, hence
            # every unoriented primitive class visits its unique canonical
            # representative once.  Keeping only that representative is
            # exactly equivalent to inserting every key into ``seen``.
            if prefix == key:
                out.append(prefix)
            return
        for letter in letters:
            if prefix and letter == inverse_letter(prefix[-1]):
                continue
            extend(prefix + (letter,))

    extend(())
    # ``letters`` is ordered and the depth-first traversal is lexicographic;
    # every word in this shell has the same length, so ``out`` already has the
    # ordering returned by the former explicit sort.
    return tuple(out)


def primitive_conjugacy_word_shell(
    num_generators: int,
    length: int,
) -> list[Word]:
    """Enumerate one exact-length shell of primitive unoriented classes."""

    return list(
        _primitive_conjugacy_word_shell_cached(
            int(num_generators),
            int(length),
        )
    )


def primitive_conjugacy_words(num_generators: int, max_length: int) -> list[Word]:
    """Enumerate primitive conjugacy classes in a free Schottky group.

    Words are represented by letters 0,1,2,3,... where odd letters are inverses
    of the preceding even letter.  The output contains one canonical
    representative for each primitive class, identifying cyclic rotations and
    inverse words.
    """

    if num_generators <= 0:
        raise ValueError("num_generators must be positive")
    if max_length <= 0:
        return []
    return [
        word
        for length in range(1, int(max_length) + 1)
        for word in primitive_conjugacy_word_shell(num_generators, length)
    ]


def attracting_multiplier_from_trace_det(trace: complex, determinant: complex) -> complex:
    """Return the multiplier q with |q| <= 1 from trace and determinant.

    If the Mobius matrix has eigenvalues lambda_+ and lambda_-, the derivative
    at a fixed point is the ratio of the two eigenvalues.  This avoids the
    explicit fixed-point derivative formula, which can run into infinities for
    words whose fixed point is represented by the point at infinity.
    """

    trace = complex(trace)
    determinant = complex(determinant)
    if abs(determinant) == 0.0:
        raise ValueError("Mobius matrix has zero determinant")
    discriminant = trace * trace - 4.0 * determinant
    root = cmath.sqrt(discriminant)
    candidate_a = 0.5 * (trace + root)
    candidate_b = 0.5 * (trace - root)
    lambda_large = candidate_a if abs(candidate_a) >= abs(candidate_b) else candidate_b
    if abs(lambda_large) == 0.0:
        raise ValueError("loxodromic Mobius map has no nonzero eigenvalue")
    # Avoid subtractive cancellation in the small eigenvalue.  Since
    # lambda_large * lambda_small = determinant, the attracting multiplier is
    # lambda_small / lambda_large = determinant / lambda_large**2.
    multiplier = determinant / (lambda_large * lambda_large)
    if abs(multiplier) > 1.0:
        multiplier = 1.0 / multiplier
    if not (abs(multiplier) < 1.0):
        raise ValueError(f"expected an attracting multiplier inside the unit disk, got {multiplier!r}")
    return complex(multiplier)


def attracting_multiplier(mobius: Mobius) -> complex:
    """Return the multiplier q with |q| <= 1 for a loxodromic Mobius map."""

    trace = complex(mobius.a + mobius.d)
    determinant = complex(mobius.a * mobius.d - mobius.b * mobius.c)
    return attracting_multiplier_from_trace_det(trace, determinant)


def mobius_matrix(mobius: Mobius) -> tuple[complex, complex, complex, complex]:
    return complex(mobius.a), complex(mobius.b), complex(mobius.c), complex(mobius.d)


def matrix_multiply(
    left: tuple[complex, complex, complex, complex],
    right: tuple[complex, complex, complex, complex],
) -> tuple[complex, complex, complex, complex]:
    a, b, c, d = left
    e, f, g, h = right
    return (
        a * e + b * g,
        a * f + b * h,
        c * e + d * g,
        c * f + d * h,
    )


def matrix_determinant(matrix: tuple[complex, complex, complex, complex]) -> complex:
    a, b, c, d = matrix
    return a * d - b * c


def matrix_determinant_from_multiplier(
    matrix: tuple[complex, complex, complex, complex],
    multiplier: complex,
) -> complex:
    """Recover a projective determinant from a known eigenvalue ratio.

    If the eigenvalues have ratio ``q``, then

    ``det(M) = q * trace(M)^2 / (1 + q)^2``.

    This avoids catastrophic cancellation in ``a*d-b*c`` for highly
    non-normal Schottky generators.  A zero multiplier is used as the marker
    for generated test matrices that do not carry a multiplier hint.
    """

    q = complex(multiplier)
    if q == 0.0:
        return matrix_determinant(matrix)
    a, _, _, d = matrix
    denominator = 1.0 + q
    if abs(denominator) == 0.0:
        return matrix_determinant(matrix)
    return q * ((a + d) / denominator) ** 2


def normalized_matrix_and_determinant(
    matrix: tuple[complex, complex, complex, complex],
    determinant: complex,
) -> tuple[tuple[complex, complex, complex, complex], complex]:
    """Projectively rescale a matrix and its determinant after multiplication."""

    scale = max(abs(value) for value in matrix)
    if scale == 0.0:
        raise ValueError("zero Mobius word matrix")
    return tuple(value / scale for value in matrix), determinant / (scale * scale)


def letter_mobius(generators: Sequence[GeneratorData], letter: int) -> Mobius:
    generator = generators[letter // 2].gamma
    return generator.inv() if letter % 2 else generator


def word_trace_and_det(generators: Sequence[GeneratorData], word: Word) -> tuple[complex, complex]:
    """Return a projectively consistent trace and determinant for one word."""

    matrix = (1.0 + 0.0j, 0.0j, 0.0j, 1.0 + 0.0j)
    determinant = 1.0 + 0.0j
    for letter in word:
        letter_matrix = mobius_matrix(letter_mobius(generators, letter))
        matrix = matrix_multiply(matrix, letter_matrix)
        determinant *= matrix_determinant_from_multiplier(
            letter_matrix,
            generators[letter // 2].multiplier,
        )
        matrix, determinant = normalized_matrix_and_determinant(matrix, determinant)
    a, _, _, d = matrix
    return a + d, determinant


def word_multiplier(generators: Sequence[GeneratorData], word: Word) -> complex:
    trace, determinant = word_trace_and_det(generators, word)
    if determinant == 0.0 and trace != 0.0:
        # Products containing repeated passages through an extremely pinched
        # generator can have a genuine multiplier below the float range.  Its
        # oscillator contribution is then also below the float range, so zero
        # is the correct numerical limit rather than a singular-map error.
        return 0.0j
    return attracting_multiplier_from_trace_det(trace, determinant)


def word_multipliers_shared_prefix(
    generators: Sequence[GeneratorData],
    words: Sequence[Word],
) -> list[complex]:
    """Return exact word multipliers while reusing lexicographic prefixes.

    Primitive shells are emitted in lexicographic order.  Adjacent words
    therefore share long prefixes, and recomputing the corresponding Mobius
    products from the identity wastes most matrix multiplications.  This
    routine retains only the states along the preceding word, reuses their
    longest common prefix, and then performs the identical left-to-right
    normalized multiplications for the remaining letters.  It changes
    neither multiplication order nor floating-point values and uses only
    ``O(max_word_length)`` auxiliary memory.
    """

    if not generators:
        raise ValueError("at least one Schottky generator is required")
    letter_matrices: list[tuple[complex, complex, complex, complex]] = []
    letter_determinants: list[complex] = []
    for letter in range(2 * len(generators)):
        matrix = mobius_matrix(letter_mobius(generators, letter))
        letter_matrices.append(matrix)
        letter_determinants.append(
            matrix_determinant_from_multiplier(
                matrix,
                generators[letter // 2].multiplier,
            )
        )

    previous: Word = ()
    states: list[
        tuple[tuple[complex, complex, complex, complex], complex]
    ] = [((1.0 + 0.0j, 0.0j, 0.0j, 1.0 + 0.0j), 1.0 + 0.0j)]
    out: list[complex] = []
    for word in words:
        common = 0
        common_maximum = min(len(previous), len(word))
        while common < common_maximum and previous[common] == word[common]:
            common += 1
        del states[common + 1 :]
        for letter in word[common:]:
            matrix, determinant = states[-1]
            matrix = matrix_multiply(matrix, letter_matrices[letter])
            determinant *= letter_determinants[letter]
            matrix, determinant = normalized_matrix_and_determinant(
                matrix,
                determinant,
            )
            states.append((matrix, determinant))
        matrix, determinant = states[-1]
        trace = matrix[0] + matrix[3]
        if determinant == 0.0 and trace != 0.0:
            out.append(0.0j)
        else:
            out.append(attracting_multiplier_from_trace_det(trace, determinant))
        previous = word
    return out


def minus_log_one_minus(value: complex) -> complex:
    """Evaluate ``-log(1-value)`` without small-value cancellation."""

    z = complex(value)
    if abs(z) >= 1.0:
        raise ValueError("-log(1-z) series requires |z|<1")
    if abs(z) >= 1.0e-6:
        return -cmath.log(1.0 - z)

    total = 0.0j
    power = z
    for order in range(1, 16):
        term = power / order
        total += term
        if abs(term) <= 1.0e-18 * max(abs(total), 1.0e-300):
            break
        power *= z
    return total


def oscillator_log_factor(multiplier: complex, *, max_mode: int, tolerance: float) -> tuple[complex, float]:
    """Return -sum_{n>=2} log(1-q^n) for an unoriented primitive class.

    Every mode through ``max_mode`` is evaluated.  In particular, a requested
    oscillator level has one unambiguous meaning across all multipliers; the
    per-multiplier ``tolerance`` is retained for API compatibility but does not
    shorten the product.  The second return value bounds modes strictly above
    ``max_mode``.
    """

    q = complex(multiplier)
    q_abs = abs(q)
    if not q_abs < 1.0:
        raise ValueError(f"Schottky multiplier must have |q|<1, got |q|={q_abs:.6g}")
    if max_mode < 2:
        raise ValueError("max_mode must be at least 2")
    total = 0.0j
    for mode in range(2, int(max_mode) + 1):
        term = q**mode
        total += minus_log_one_minus(term)

    if q_abs == 0.0:
        tail = 0.0
    elif q_abs < 1.0:
        # Bound sum_{n>N} |log(1-q^n)| by using |log(1-z)| <= |z|/(1-|z|)
        # for |z|<1, then summing the geometric tail.
        first = q_abs ** (int(max_mode) + 1)
        tail = first / max(1.0e-300, (1.0 - q_abs) * (1.0 - first))
    else:
        tail = math.inf
    return total, float(tail)


def schottky_vacuum_block(
    generators: Sequence[GeneratorData],
    *,
    max_word_length: int = 8,
    max_mode: int = 50,
    tolerance: float = 1.0e-14,
    word_tail_tolerance: float | None = None,
    minimum_word_length: int = 5,
    channel: str = "schottky",
    q_values: tuple[complex, ...] | None = None,
) -> VacuumBlockResult:
    """Evaluate the truncated Schottky vacuum block product at arbitrary rank.

    The product is over primitive conjugacy classes of the free Schottky
    group generated by ``generators``.  Genus is therefore the number of
    supplied generators.  ``oscillator_mode_tail_estimate`` bounds only the
    omitted ``n`` modes for the enumerated words.
    ``primitive_word_tail_estimate`` is an empirical geometric-shell
    diagnostic.  It is not a rigorous transfer-operator bound, so
    ``truncation_certified`` remains false.  If ``word_tail_tolerance`` is
    supplied, exact-length shells are added until the empirical absolute
    chiral-log tail reaches that target or ``max_word_length`` is exhausted.
    """

    if not generators:
        raise ValueError("at least one Schottky generator is required")
    requested_maximum = int(max_word_length)
    minimum = int(minimum_word_length)
    if requested_maximum <= 0:
        raise ValueError("max_word_length must be positive")
    if minimum <= 0:
        raise ValueError("minimum_word_length must be positive")
    target = (
        None
        if word_tail_tolerance is None
        else float(word_tail_tolerance)
    )
    if target is not None and (
        not math.isfinite(target) or target <= 0.0
    ):
        raise ValueError("word_tail_tolerance must be finite and positive")

    contributions: list[PrimitiveClassContribution] = []
    log_value = 0.0j
    omitted = 0.0
    effective_maximum = 0
    for word_length in range(1, requested_maximum + 1):
        shell_words = primitive_conjugacy_word_shell(
            len(generators),
            word_length,
        )
        shell_multipliers = word_multipliers_shared_prefix(
            generators,
            shell_words,
        )
        for word, multiplier in zip(shell_words, shell_multipliers):
            log_factor, tail = oscillator_log_factor(
                multiplier,
                max_mode=int(max_mode),
                tolerance=float(tolerance),
            )
            log_value += log_factor
            omitted += tail
            contributions.append(
                PrimitiveClassContribution(
                    word=word,
                    multiplier=multiplier,
                    log_factor=log_factor,
                )
            )
        effective_maximum = word_length
        if target is not None and word_length >= minimum:
            current = primitive_word_tail_diagnostic(
                contributions,
                word_length,
            )
            if (
                current.estimated_omitted_abs_log is not None
                and current.estimated_omitted_abs_log <= target
            ):
                break
    word_diagnostic = primitive_word_tail_diagnostic(
        contributions,
        effective_maximum,
    )
    return VacuumBlockResult(
        channel=channel,
        q_values=q_values,
        max_word_length=effective_maximum,
        max_mode=int(max_mode),
        tolerance=float(tolerance),
        log_value=log_value,
        value=cmath.exp(log_value),
        primitive_count=len(contributions),
        oscillator_mode_tail_estimate=float(omitted),
        primitive_word_tail_estimate=(
            word_diagnostic.estimated_omitted_abs_log
        ),
        truncation_certified=False,
        contributions=tuple(contributions),
    )


def glasses_vacuum_block(
    q1: complex,
    q2: complex,
    q_bridge: complex,
    *,
    max_word_length: int = 8,
    max_mode: int = 50,
    tolerance: float = 1.0e-14,
    word_tail_tolerance: float | None = None,
    minimum_word_length: int = 5,
) -> VacuumBlockResult:
    """Genus-two vacuum block in the glasses/sunglasses plumbing channel."""

    q_values = (complex(q1), complex(q2), complex(q_bridge))
    return schottky_vacuum_block(
        generators_for_glasses(*q_values),
        max_word_length=max_word_length,
        max_mode=max_mode,
        tolerance=tolerance,
        word_tail_tolerance=word_tail_tolerance,
        minimum_word_length=minimum_word_length,
        channel="glasses",
        q_values=q_values,
    )


def sunrise_vacuum_block(
    q0: complex,
    q1: complex,
    q2: complex,
    *,
    max_word_length: int = 8,
    max_mode: int = 50,
    tolerance: float = 1.0e-14,
    word_tail_tolerance: float | None = None,
    minimum_word_length: int = 5,
) -> VacuumBlockResult:
    """Genus-two vacuum block in the sunrise plumbing channel."""

    q_values = (complex(q0), complex(q1), complex(q2))
    return schottky_vacuum_block(
        generators_for_sunrise(*q_values),
        max_word_length=max_word_length,
        max_mode=max_mode,
        tolerance=tolerance,
        word_tail_tolerance=word_tail_tolerance,
        minimum_word_length=minimum_word_length,
        channel="sunrise",
        q_values=q_values,
    )


def hmpz_schottky_generators(p1: complex, p2: complex, x: complex) -> list[GeneratorData]:
    """Schottky generators in the HMPZ `(p1,p2,x)` sewing coordinates.

    The fixed points are `(0, infinity)` for the first handle and `(1, x)` for
    the second handle, following the conventions around equation (4.1) of
    arXiv:1503.07111.
    """

    p1 = complex(p1)
    p2 = complex(p2)
    x = complex(x)
    if x == 1.0:
        raise ValueError("HMPZ Schottky cross-ratio x=1 is the separating degeneration")

    first = generator_data_from_mobius(Mobius(p1, 0.0j, 0.0j, 1.0 + 0.0j).normalized())
    gamma_1x = Mobius(x, 1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j).normalized()
    second = generator_data_from_mobius(
        gamma_1x.compose(Mobius(p2, 0.0j, 0.0j, 1.0 + 0.0j).normalized()).compose(gamma_1x.inv())
    )
    return [first, second]


def hmpz_large_c_vacuum_block(
    p1: complex,
    p2: complex,
    x: complex,
    *,
    max_word_length: int = 8,
    max_mode: int = 50,
    tolerance: float = 1.0e-14,
) -> VacuumBlockResult:
    """Large-c square-root Schottky product in HMPZ sewing coordinates."""

    q_values = (complex(p1), complex(p2), complex(x))
    return schottky_vacuum_block(
        hmpz_schottky_generators(*q_values),
        max_word_length=max_word_length,
        max_mode=max_mode,
        tolerance=tolerance,
        channel="hmpz-schottky",
        q_values=q_values,
    )


def genus_one_vacuum_character(q: complex, *, max_mode: int = 80, tolerance: float = 1.0e-14) -> complex:
    """Return the holomorphic vacuum character prod_{n>=2} (1-q^n)^(-1)."""

    log_value, _ = oscillator_log_factor(complex(q), max_mode=max_mode, tolerance=tolerance)
    return cmath.exp(log_value)


def hmpz_c22(x: complex, central_charge: complex) -> complex:
    """HMPZ Schottky-sewing coefficient C_{2,2}(x).

    This is equation (4.21) of Headrick-Maloney-Perlmutter-Zadeh,
    arXiv:1503.07111, in the genus-two vacuum channel with Schottky sewing
    parameters `(p1, p2, x)`.
    """

    x = complex(x)
    c = complex(central_charge)
    if x == 0:
        raise ValueError("HMPZ cross-ratio x must be nonzero")
    return (
        1.0
        + (x - 1.0) ** 4
        + ((x - 1.0) ** 4) / (x**4)
        + (8.0 / c) * ((x - 1.0) ** 2) * (1.0 - x + x * x) / (x * x)
    )


def hmpz_c44_lambda(x: complex, central_charge: complex) -> complex:
    """HMPZ level-four quasi-primary Lambda contribution to C_{4,4}(x).

    This is equation (4.30) of arXiv:1503.07111.  It is the first term that
    makes the genus-two vacuum free energy contain arbitrarily high powers of
    `1/c` away from the separating degeneration point.
    """

    x = complex(x)
    c = complex(central_charge)
    if x == 0:
        raise ValueError("HMPZ cross-ratio x must be nonzero")
    one_minus_plus = 1.0 - x + x * x
    return (
        (one_minus_plus**8) / (x**8)
        + ((32.0 / c) - 8.0) * ((x - 1.0) ** 2) * (one_minus_plus**5) / (x**6)
        + (
            4.0
            * (3704.0 + 590.0 * c + 125.0 * c * c)
            / (5.0 * c * (22.0 + 5.0 * c))
            * ((x - 1.0) ** 4)
            * (one_minus_plus**2)
            / (x**4)
        )
    )


def hmpz_schottky_vacuum_block_level2(
    p1: complex,
    p2: complex,
    x: complex,
    *,
    central_charge: complex,
) -> complex:
    """HMPZ vacuum-channel Schottky sewing block through level two.

    The truncation keeps the vacuum descendants at levels 0 and 2 on each of
    the two Schottky handles:

        Z_vac = 1 + p1^2 + p2^2 + p1^2 p2^2 C_{2,2}(x) + O(p^3).
    """

    p1 = complex(p1)
    p2 = complex(p2)
    return 1.0 + p1 * p1 + p2 * p2 + (p1 * p1) * (p2 * p2) * hmpz_c22(x, central_charge)


def hmpz_three_loop_free_energy_leading(p1: complex, p2: complex, x: complex) -> complex:
    """Leading HMPZ three-loop free-energy coefficient.

    Returns the `O(p1^4 p2^4)` term of `F_vac;3` from equation (5.8) of
    arXiv:1503.07111.
    """

    p1 = complex(p1)
    p2 = complex(p2)
    x = complex(x)
    if x == 0:
        raise ValueError("HMPZ cross-ratio x must be nonzero")
    return (
        (p1**4)
        * (p2**4)
        * 13312.0
        * ((x - 1.0) ** 4)
        * ((1.0 - x + x * x) ** 2)
        / (25.0 * x**4)
    )


def _rho_apply_mode_to_desc(
    mode: int,
    desc: Descendant,
    *,
    c: complex,
    vacuum: bool = True,
) -> State:
    return act_virasoro_mode(int(mode), {tuple(desc): 1.0 + 0.0j}, h=0.0, c=c, vacuum=vacuum)


def _rho_state_sum(
    state3: State,
    desc2: Descendant,
    state1: State,
    *,
    c: complex,
) -> complex:
    total = 0.0j
    for desc3, coeff3 in state3.items():
        for desc1, coeff1 in state1.items():
            total += coeff3 * coeff1 * rho_vacuum_descendants(desc3, desc2, desc1, c=c)
    return total


@lru_cache(maxsize=None)
def rho_vacuum_descendants(desc3: Descendant, desc2: Descendant, desc1: Descendant, c: complex) -> complex:
    """Plane three-point function rho(desc3, desc2, desc1 | z=1).

    The convention follows Appendix A of arXiv:1703.09805: `desc1` is inserted
    at 0, `desc2` at z=1, and `desc3` is BPZ conjugate at infinity.
    """

    desc3 = tuple(desc3)
    desc2 = tuple(desc2)
    desc1 = tuple(desc1)

    if desc3 and desc3[-1] == 1:
        desc3_state = project_vacuum_state({desc3: 1.0 + 0.0j})
        return _rho_state_sum(desc3_state, desc2, {desc1: 1.0 + 0.0j}, c=c)
    if desc2 and desc2[-1] == 1:
        desc2_state = project_vacuum_state({desc2: 1.0 + 0.0j})
        return sum(coeff * rho_vacuum_descendants(desc3, middle, desc1, c=c) for middle, coeff in desc2_state.items())
    if desc1 and desc1[-1] == 1:
        desc1_state = project_vacuum_state({desc1: 1.0 + 0.0j})
        return _rho_state_sum({desc3: 1.0 + 0.0j}, desc2, desc1_state, c=c)

    if not desc2:
        return descendant_inner_product(desc3, desc1, c=c, vacuum=True)

    n = int(desc2[0])
    rest2 = desc2[1:]
    if n == 1:
        exponent = descendant_level(desc3) - descendant_level(rest2) - descendant_level(desc1)
        return complex(exponent) * rho_vacuum_descendants(desc3, rest2, desc1, c=c)
    if n <= 0:
        raise ValueError("descendant modes must be positive integers")

    total = 0.0j
    max_m = max(0, descendant_level(desc3) - n, descendant_level(desc1) + 1)
    for m in range(max_m + 1):
        coefficient = math.comb(n - 2 + m, n - 2)
        left_state = _rho_apply_mode_to_desc(n + m, desc3, c=c, vacuum=True)
        if left_state:
            total += coefficient * _rho_state_sum(left_state, rest2, {desc1: 1.0 + 0.0j}, c=c)

        right_state = _rho_apply_mode_to_desc(m - 1, desc1, c=c, vacuum=True)
        if right_state:
            total += coefficient * ((-1) ** n) * _rho_state_sum(
                {desc3: 1.0 + 0.0j},
                rest2,
                right_state,
                c=c,
            )
    return total


def inverse_vacuum_gram_by_level(max_level: int, c: complex) -> dict[int, tuple[list[Descendant], np.ndarray]]:
    out: dict[int, tuple[list[Descendant], np.ndarray]] = {}
    for level in range(int(max_level) + 1):
        basis, matrix = vacuum_gram_matrix(level, c)
        if not basis:
            out[level] = (basis, np.zeros((0, 0), dtype=complex))
        else:
            out[level] = (basis, np.linalg.inv(np.asarray(matrix, dtype=complex)))
    return out


def direct_genus2_vacuum_block(
    q1: complex,
    q2: complex,
    q3: complex,
    *,
    central_charge: complex,
    max_level: int,
    channel: str = "theta",
) -> DirectVacuumBlockResult:
    """Finite-c genus-two vacuum block by direct descendant summation.

    This evaluates equation (5.5) of arXiv:1703.09805 specialized to the
    vacuum module, with all three sewing levels bounded by `max_level`.
    """

    q_values = (complex(q1), complex(q2), complex(q3))
    c = complex(central_charge)
    gram_data = inverse_vacuum_gram_by_level(int(max_level), c)
    total = 0.0j
    level_contributions: dict[tuple[int, int, int], complex] = {}

    for level1 in range(int(max_level) + 1):
        basis1, inv1 = gram_data[level1]
        for level2 in range(int(max_level) + 1):
            basis2, inv2 = gram_data[level2]
            for level3 in range(int(max_level) + 1):
                basis3, inv3 = gram_data[level3]
                contribution = 0.0j
                for i, desc_a in enumerate(basis1):
                    for j, desc_b in enumerate(basis1):
                        coeff1 = inv1[i, j]
                        if coeff1 == 0:
                            continue
                        for k, desc_c in enumerate(basis2):
                            for ell, desc_d in enumerate(basis2):
                                coeff2 = inv2[k, ell]
                                if coeff2 == 0:
                                    continue
                                for r, desc_e in enumerate(basis3):
                                    rho_left_cache = rho_vacuum_descendants(desc_a, desc_c, desc_e, c=c)
                                    if rho_left_cache == 0:
                                        continue
                                    for s, desc_f in enumerate(basis3):
                                        coeff3 = inv3[r, s]
                                        if coeff3 == 0:
                                            continue
                                        rho_right = rho_vacuum_descendants(desc_b, desc_d, desc_f, c=c)
                                        contribution += coeff1 * coeff2 * coeff3 * rho_left_cache * rho_right
                contribution *= (q_values[0] ** level1) * (q_values[1] ** level2) * (q_values[2] ** level3)
                if abs(contribution) > 0.0:
                    level_contributions[(level1, level2, level3)] = contribution
                    total += contribution

    return DirectVacuumBlockResult(
        channel=str(channel),
        q_values=q_values,
        central_charge=c,
        max_level=int(max_level),
        value=total,
        level_contributions=level_contributions,
    )


def theta_finite_vacuum_block(
    q1: complex,
    q2: complex,
    q3: complex,
    *,
    central_charge: complex,
    max_level: int,
) -> DirectVacuumBlockResult:
    """Finite-c vacuum block in the theta pair-of-pants sewing frame.

    This is the direct descendant sum of equation (5.5) in arXiv:1703.09805.
    The three `q` values are the sewing parameters attached to the three
    internal tubes of that theta decomposition.
    """

    return direct_genus2_vacuum_block(
        q1,
        q2,
        q3,
        central_charge=central_charge,
        max_level=max_level,
        channel="theta",
    )


def glasses_finite_vacuum_block(
    q1: complex,
    q2: complex,
    q_bridge: complex,
    *,
    central_charge: complex,
    max_level: int,
) -> DirectVacuumBlockResult:
    """Compatibility alias for the theta-frame finite-c descendant sum.

    This function does not convert glasses Schottky parameters into theta
    sewing coordinates.  It is kept so older exploratory calls remain usable,
    but new code should call `theta_finite_vacuum_block` unless an explicit
    glasses-to-theta coordinate map has been supplied.
    """

    warnings.warn(
        "glasses_finite_vacuum_block is a theta-frame descendant-sum alias; "
        "it does not map glasses Schottky parameters to finite-c sewing coordinates",
        RuntimeWarning,
        stacklevel=2,
    )

    return direct_genus2_vacuum_block(
        q1,
        q2,
        q_bridge,
        central_charge=central_charge,
        max_level=max_level,
        channel="theta-from-glasses-labels",
    )


def sunrise_finite_vacuum_block(
    q0: complex,
    q1: complex,
    q2: complex,
    *,
    central_charge: complex,
    max_level: int,
) -> DirectVacuumBlockResult:
    """Compatibility alias for the theta-frame finite-c descendant sum.

    This function does not convert sunrise Schottky parameters into theta
    sewing coordinates.  It is kept so older exploratory calls remain usable,
    but new code should call `theta_finite_vacuum_block` unless an explicit
    sunrise-to-theta coordinate map has been supplied.
    """

    warnings.warn(
        "sunrise_finite_vacuum_block is a theta-frame descendant-sum alias; "
        "it does not map sunrise Schottky parameters to finite-c sewing coordinates",
        RuntimeWarning,
        stacklevel=2,
    )

    return direct_genus2_vacuum_block(
        q0,
        q1,
        q2,
        central_charge=central_charge,
        max_level=max_level,
        channel="theta-from-sunrise-labels",
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def check_primitive_enumeration() -> None:
    words = primitive_conjugacy_words(2, 4)
    keys = {canonical_primitive_key(word) for word in words}
    _require(len(words) == len(keys), "primitive words contain duplicate conjugacy classes")
    _require((0,) in keys and (2,) in keys, "single-generator primitive classes missing")
    _require((0, 2) in keys, "mixed primitive class AB missing")
    _require(all(is_primitive_cyclic_word(word) for word in words), "nonprimitive word was enumerated")


def check_primitive_word_tail_diagnostic() -> None:
    geometric = tuple(
        PrimitiveClassContribution(
            word=(0,) * length,
            multiplier=0.0j,
            log_factor=0.2**length,
        )
        for length in range(1, 9)
    )
    geometric_result = primitive_word_tail_diagnostic(geometric, 8)
    exact_geometric_tail = 0.2**9 / (1.0 - 0.2)
    _require(
        geometric_result.empirically_convergent
        and geometric_result.shell_group_size == 1
        and geometric_result.estimated_omitted_abs_log is not None
        and geometric_result.estimated_omitted_abs_log >= exact_geometric_tail,
        "geometric primitive-word tail was not conservatively estimated",
    )

    alternating_norms = (
        1.0e-2,
        5.0e-3,
        2.0e-4,
        1.0e-4,
        4.0e-6,
        2.0e-6,
        8.0e-8,
        4.0e-8,
    )
    alternating = tuple(
        PrimitiveClassContribution(
            word=(0,) * length,
            multiplier=0.0j,
            log_factor=value,
        )
        for length, value in enumerate(alternating_norms, start=1)
    )
    alternating_result = primitive_word_tail_diagnostic(alternating, 8)
    _require(
        alternating_result.empirically_convergent
        and alternating_result.shell_group_size == 2
        and alternating_result.estimated_omitted_abs_log is not None,
        "two-shell primitive-word envelope did not handle an alternating tail",
    )


def check_oscillator_fixed_cutoff_tail() -> None:
    multiplier = 0.5 + 0.0j
    value, tail = oscillator_log_factor(
        multiplier,
        max_mode=40,
        tolerance=0.1,
    )
    expected_value = sum(
        (-cmath.log(1.0 - multiplier**mode) for mode in range(2, 41)),
        0.0j,
    )
    observed_remainder = sum(
        (-math.log(1.0 - multiplier.real**mode) for mode in range(41, 400)),
        0.0,
    )
    _require(
        abs(value - expected_value) < 1.0e-15,
        "vacuum oscillator product did not use the requested fixed cutoff",
    )
    _require(
        tail >= observed_remainder,
        "vacuum oscillator tail does not cover modes above the fixed cutoff",
    )


def check_vacuum_verma_gram_matrices() -> None:
    c = 17.0
    basis2, gram2 = vacuum_gram_matrix(2, c)
    basis3, gram3 = vacuum_gram_matrix(3, c)
    basis4, gram4 = vacuum_gram_matrix(4, c)

    _require(basis2 == [(2,)], f"unexpected level-2 vacuum basis: {basis2}")
    _require(basis3 == [(3,)], f"unexpected level-3 vacuum basis: {basis3}")
    _require(basis4 == [(4,), (2, 2)], f"unexpected level-4 vacuum basis: {basis4}")
    _require(abs(gram2[0][0] - c / 2.0) < 1.0e-12, "wrong level-2 vacuum norm")
    _require(abs(gram3[0][0] - 2.0 * c) < 1.0e-12, "wrong level-3 vacuum norm")
    expected4 = [
        [5.0 * c, 3.0 * c],
        [3.0 * c, 0.5 * c * (c + 8.0)],
    ]
    max_error = max(abs(gram4[i][j] - expected4[i][j]) for i in range(2) for j in range(2))
    _require(max_error < 1.0e-12, f"wrong level-4 vacuum Gram matrix: error={max_error:.3e}")


def check_rho_and_direct_level2_block() -> None:
    c = 17.0
    _require(abs(rho_vacuum_descendants((), (), (), c=c) - 1.0) < 1.0e-12, "rho(vac,vac,vac) != 1")
    _require(abs(rho_vacuum_descendants((), (2,), (), c=c)) < 1.0e-12, "one stress tensor one-point should vanish")
    _require(
        abs(rho_vacuum_descendants((2,), (), (2,), c=c) - c / 2.0) < 1.0e-12,
        "rho with identity insertion should reproduce the Gram pairing",
    )
    _require(
        abs(rho_vacuum_descendants((2,), (2,), (2,), c=c) - c) < 1.0e-12,
        "wrong three stress-tensor descendant rho",
    )

    q1 = 0.01 + 0.002j
    q2 = -0.015 + 0.001j
    q3 = 0.012 - 0.003j
    result = direct_genus2_vacuum_block(q1, q2, q3, central_charge=c, max_level=2)
    expected = (
        1.0
        + (q1 * q1) * (q2 * q2)
        + (q1 * q1) * (q3 * q3)
        + (q2 * q2) * (q3 * q3)
        + (8.0 / c) * (q1 * q1) * (q2 * q2) * (q3 * q3)
    )
    _require(abs(result.value - expected) < 1.0e-14, "direct level-2 block coefficient check failed")


def check_hmpz_vacuum_channel_formulas() -> None:
    c = 23.0
    x = 0.37 + 0.19j
    p1 = 0.021 + 0.004j
    p2 = -0.017 + 0.003j

    c22 = hmpz_c22(x, c)
    c22_crossed = hmpz_c22(1.0 / x, c)
    _require(abs(c22 - c22_crossed) < 1.0e-12, "HMPZ C_22(x) is not invariant under x -> 1/x")
    _require(abs(hmpz_c22(1.0, c) - 1.0) < 1.0e-12, "HMPZ C_22 does not factorize at x=1")

    factorized = hmpz_schottky_vacuum_block_level2(p1, p2, 1.0, central_charge=c)
    expected_factorized = (1.0 + p1 * p1) * (1.0 + p2 * p2)
    _require(
        abs(factorized - expected_factorized) < 1.0e-14,
        "HMPZ level-2 block does not factorize at the separating point",
    )

    _require(
        abs(hmpz_three_loop_free_energy_leading(p1, p2, 1.0)) < 1.0e-30,
        "HMPZ leading three-loop term should vanish at separating degeneration",
    )
    _require(
        abs(hmpz_three_loop_free_energy_leading(p1, p2, x)) > 0.0,
        "HMPZ leading three-loop term should be nonzero away from x=1 for this sample",
    )

    c44_large = hmpz_c44_lambda(x, 1.0e8)
    c44_classical_part = (
        ((1.0 - x + x * x) ** 8) / (x**8)
        - 8.0 * ((x - 1.0) ** 2) * ((1.0 - x + x * x) ** 5) / (x**6)
        + 20.0 * ((x - 1.0) ** 4) * ((1.0 - x + x * x) ** 2) / (x**4)
    )
    _require(
        abs(c44_large - c44_classical_part) / max(1.0, abs(c44_classical_part)) < 1.0e-6,
        "HMPZ C_44 Lambda large-c limit is inconsistent",
    )

    # The square of the 1703.09805 large-c block has the unsquared one-loop
    # normalization used by HMPZ.  At order p1^2 p2^2, the mixed terms in
    # C_{2,2}(x) come from the primitive length-two words AB and AB^{-1}.
    x_mixed = 0.6 + 0.2j
    epsilon = 1.0e-6
    generators = hmpz_schottky_generators(epsilon, epsilon, x_mixed)
    mixed_from_multipliers = sum(
        (word_multiplier(generators, word) / (epsilon * epsilon)) ** 2
        for word in ((0, 2), (0, 3))
    )
    mixed_hmpz = (x_mixed - 1.0) ** 4 + ((x_mixed - 1.0) ** 4) / (x_mixed**4)
    _require(
        abs(mixed_from_multipliers - mixed_hmpz) < 1.0e-5,
        "HMPZ Schottky generators do not reproduce the large-c C_22 mixed coefficient",
    )


def check_glasses_separating_degeneration() -> None:
    q1 = 0.034 * cmath.exp(0.21j)
    q2 = 0.026 * cmath.exp(-0.34j)
    bridge_values = [0.1, 0.02]
    errors: list[float] = []
    target = genus_one_vacuum_character(q1) * genus_one_vacuum_character(q2)
    for bridge in bridge_values:
        value = glasses_vacuum_block(
            q1,
            q2,
            bridge,
            max_word_length=7,
            max_mode=60,
        ).value
        errors.append(abs(value - target) / max(1.0e-30, abs(target)))
    _require(errors[-1] < errors[0], "glasses separating error did not improve as q_bridge decreased")
    _require(errors[-1] < 1.0e-9, f"glasses separating degeneration error too large: {errors[-1]:.3e}")


def check_schottky_marking_invariance() -> None:
    q1 = 0.045 * cmath.exp(0.3j)
    q2 = 0.033 * cmath.exp(-0.2j)
    qb = 0.055 * cmath.exp(0.1j)
    generators = generators_for_glasses(q1, q2, qb)
    base = schottky_vacuum_block(generators, max_word_length=7, max_mode=60).value

    swapped = [generators[1], generators[0]]
    swapped_value = schottky_vacuum_block(swapped, max_word_length=7, max_mode=60).value
    _require(abs(base - swapped_value) / abs(base) < 1.0e-12, "generator swap changed the block")

    inverted = [
        GeneratorData(
            gamma=generators[0].gamma.inv(),
            attracting=generators[0].repelling,
            repelling=generators[0].attracting,
            multiplier=generators[0].multiplier,
        ),
        generators[1],
    ]
    inverted_value = schottky_vacuum_block(inverted, max_word_length=7, max_mode=60).value
    _require(abs(base - inverted_value) / abs(base) < 1.0e-12, "generator inversion changed the block")


def check_projective_determinant_stability() -> None:
    """Exercise a nonsingular matrix whose direct determinant rounds to zero."""

    multiplier = 0.3 + 0.0j
    trace = 1.0e-8 + 0.0j
    determinant = multiplier * (trace / (1.0 + multiplier)) ** 2
    gamma = Mobius(
        1.0 + 0.0j,
        1.0 + 0.0j,
        -1.0 + trace - determinant,
        -1.0 + trace,
    )
    _require(
        matrix_determinant(mobius_matrix(gamma)) == 0.0,
        "stress matrix no longer exercises direct-determinant cancellation",
    )
    generator = GeneratorData(
        gamma=gamma,
        attracting=None,
        repelling=None,
        multiplier=multiplier,
    )
    observed = word_multiplier([generator], (0,))
    _require(
        abs(observed - multiplier) < 1.0e-12,
        f"projective determinant reconstruction returned {observed!r}",
    )

    tiny = 1.0e-202
    generators = [
        GeneratorData(
            gamma=Mobius(tiny, 0.0j, 0.0j, 1.0 + 0.0j),
            attracting=0.0j,
            repelling=None,
            multiplier=tiny,
        ),
        GeneratorData(
            gamma=Mobius(0.3, 0.0j, 0.0j, 1.0 + 0.0j),
            attracting=0.0j,
            repelling=None,
            multiplier=0.3,
        ),
    ]
    _require(
        word_multiplier(generators, (0, 2, 0, 3)) == 0.0,
        "subnormal primitive multiplier did not take its numerical zero limit",
    )


def check_nielsen_modular_invariance() -> None:
    """Check invariance under handlebody-preserving Schottky marking moves."""

    q1 = 0.025 * cmath.exp(0.2j)
    q2 = 0.019 * cmath.exp(-0.1j)
    qb = 0.021 * cmath.exp(0.15j)
    generators = generators_for_glasses(q1, q2, qb)
    base = schottky_vacuum_block(generators, max_word_length=8, max_mode=60).value

    # Nielsen transformations generate Aut(F_2).  These are the Schottky
    # marking changes that preserve the handlebody; they are the directly
    # visible modular consistency checks in Schottky coordinates.
    moves = [
        [
            GeneratorData(gamma=generators[0].gamma.compose(generators[1].gamma), attracting=None, repelling=None, multiplier=0j),
            generators[1],
        ],
        [
            generators[0],
            GeneratorData(gamma=generators[1].gamma.compose(generators[0].gamma), attracting=None, repelling=None, multiplier=0j),
        ],
    ]
    for idx, moved_generators in enumerate(moves, start=1):
        moved = schottky_vacuum_block(moved_generators, max_word_length=8, max_mode=60).value
        relative_error = abs(base - moved) / max(abs(base), 1.0e-30)
        _require(relative_error < 1.0e-12, f"Nielsen move {idx} changed the block by {relative_error:.3e}")


def check_sunrise_channel_finite() -> None:
    result = sunrise_vacuum_block(
        0.026 * cmath.exp(0.2j),
        0.031 * cmath.exp(-0.35j),
        0.023 * cmath.exp(0.51j),
        max_word_length=6,
        max_mode=50,
    )
    _require(math.isfinite(result.value.real), "sunrise value has non-finite real part")
    _require(math.isfinite(result.value.imag), "sunrise value has non-finite imaginary part")
    _require(result.primitive_count > 0, "sunrise evaluation has no primitive classes")


def check_sunrise_edge_symmetry_and_collapse() -> None:
    q_values = (
        0.5 * (0.026 + 0.005j),
        0.5 * (0.031 - 0.011j),
        0.5 * (0.023 + 0.013j),
    )
    base = sunrise_vacuum_block(*q_values, max_word_length=8, max_mode=60).value
    for permuted in (
        (q_values[0], q_values[2], q_values[1]),
        (q_values[1], q_values[0], q_values[2]),
        (q_values[1], q_values[2], q_values[0]),
        (q_values[2], q_values[0], q_values[1]),
        (q_values[2], q_values[1], q_values[0]),
    ):
        value = sunrise_vacuum_block(*permuted, max_word_length=8, max_mode=60).value
        relative_error = abs(value - base) / max(abs(base), 1.0e-30)
        _require(relative_error < 1.0e-8, f"sunrise edge permutation changed the block by {relative_error:.3e}")

    q0 = 0.027 * cmath.exp(0.2j)
    q2 = 0.021 * cmath.exp(-0.31j)
    target = genus_one_vacuum_character(q0 * q2, max_mode=80)
    errors = []
    for q1 in (0.01, 0.002):
        value = sunrise_vacuum_block(q0, q1, q2, max_word_length=7, max_mode=60).value
        errors.append(abs(value - target) / max(abs(target), 1.0e-30))
    _require(errors[-1] < errors[0], "sunrise edge-collapse error did not improve as q1 decreased")
    _require(errors[-1] < 1.0e-8, f"sunrise edge-collapse degeneration error too large: {errors[-1]:.3e}")


def run_checks() -> None:
    check_primitive_enumeration()
    check_primitive_word_tail_diagnostic()
    check_oscillator_fixed_cutoff_tail()
    check_vacuum_verma_gram_matrices()
    check_rho_and_direct_level2_block()
    check_hmpz_vacuum_channel_formulas()
    check_glasses_separating_degeneration()
    check_schottky_marking_invariance()
    check_projective_determinant_stability()
    check_nielsen_modular_invariance()
    check_sunrise_channel_finite()
    check_sunrise_edge_symmetry_and_collapse()
    print("all genus-two vacuum block checks passed")


def run(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate genus-two vacuum Virasoro blocks.")
    parser.add_argument("--channel", choices=["glasses", "sunrise"], default="glasses")
    parser.add_argument("--q", nargs=3, type=parse_complex, metavar=("Q1", "Q2", "Q3"))
    parser.add_argument("--finite-c", action="store_true", help="Use direct finite-c descendant summation.")
    parser.add_argument("--central-charge", type=parse_complex, default=30.0 + 0.0j)
    parser.add_argument("--max-level", type=int, default=2)
    parser.add_argument("--max-word-length", type=int, default=8)
    parser.add_argument("--max-mode", type=int, default=50)
    parser.add_argument("--tolerance", type=float, default=1.0e-14)
    parser.add_argument("--word-tail-tolerance", type=float)
    parser.add_argument("--minimum-word-length", type=int, default=5)
    parser.add_argument("--list-contributions", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    if args.check:
        run_checks()
        return

    q_values = tuple(args.q) if args.q is not None else (
        0.04 + 0.0j,
        0.03 + 0.0j,
        0.05 + 0.0j,
    )
    if args.finite_c:
        input_label = args.channel
        result = theta_finite_vacuum_block(
            *q_values,
            central_charge=args.central_charge,
            max_level=int(args.max_level),
        )
        print("genus-two finite-c direct vacuum Virasoro block")
        print("  computed frame: theta pair-of-pants plumbing")
        print(f"  input q label: {input_label}")
        print(f"  q values: {', '.join(format_complex(q) for q in q_values)}")
        print(f"  central charge: {format_complex(result.central_charge)}")
        print(f"  max level per tube: {result.max_level}")
        print(f"  block: {format_complex(result.value)}")
        print(f"  nonzero level triples: {len(result.level_contributions)}")
        if args.list_contributions:
            for levels, contribution in sorted(result.level_contributions.items()):
                print(f"  levels={levels} contribution={format_complex(contribution)}")
        return

    if args.channel == "glasses":
        result = glasses_vacuum_block(
            *q_values,
            max_word_length=args.max_word_length,
            max_mode=args.max_mode,
            tolerance=args.tolerance,
            word_tail_tolerance=args.word_tail_tolerance,
            minimum_word_length=args.minimum_word_length,
        )
    else:
        result = sunrise_vacuum_block(
            *q_values,
            max_word_length=args.max_word_length,
            max_mode=args.max_mode,
            tolerance=args.tolerance,
            word_tail_tolerance=args.word_tail_tolerance,
            minimum_word_length=args.minimum_word_length,
        )

    print("genus-two large-c vacuum Virasoro block")
    print(f"  channel: {result.channel}")
    print(f"  q values: {', '.join(format_complex(q) for q in q_values)}")
    print(f"  max word length: {result.max_word_length}")
    print(f"  max oscillator mode: {result.max_mode}")
    print(f"  primitive classes: {result.primitive_count}")
    print(f"  log block: {format_complex(result.log_value)}")
    print(f"  block: {format_complex(result.value)}")
    print(
        "  enumerated-word oscillator-mode tail estimate: "
        f"{result.oscillator_mode_tail_estimate:.6e}"
    )
    diagnostic = result.primitive_word_convergence
    print(
        "  empirical primitive-word chiral-log tail estimate: "
        + (
            f"{result.primitive_word_tail_estimate:.6e}"
            if result.primitive_word_tail_estimate is not None
            else "not yet convergent"
        )
    )
    if diagnostic.shell_group_size is not None:
        print(
            "  shell grouping / guarded ratio: "
            f"{diagnostic.shell_group_size} / {diagnostic.guarded_ratio:.6e}"
        )
    print(f"  full truncation certified: {result.truncation_certified}")

    if args.list_contributions:
        for contribution in result.contributions:
            word = "".join(str(letter) for letter in contribution.word)
            print(
                f"  word={word:<12s} multiplier={format_complex(contribution.multiplier)} "
                f"log_factor={format_complex(contribution.log_factor)}"
            )


if __name__ == "__main__":
    run()
