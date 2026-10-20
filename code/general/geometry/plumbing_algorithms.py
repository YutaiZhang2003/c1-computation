#!/usr/bin/env python3
"""Core plumbing algorithms for holomorphic one-forms.

This module contains the reusable numerical backbone:

1. Schottky generators derived from plumbing data.
2. Small-q holomorphic one-forms from Schottky/Poincare series.
3. All-q direct boundary collocation on plumbing seams.

The companion document is ``plumbing_algorithms.md``.  Reproducible checks live
in ``plumbing_checks.py``.
"""

from __future__ import annotations

from c1_paths import PACKAGE_ROOT, data_root, package_path, module_path

import argparse
import cmath
from functools import lru_cache
import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import numpy as np


TWO_PI_I = 2.0j * math.pi
MULTIPRECISION_ASYMPTOTIC_LOG_Q_FLOOR = -400.0
INF = None
PUNCTURES = ("zero", "one", "infty")


def parse_complex(value: str) -> complex:
    return complex(value.replace("i", "j"))


def tau_from_q(q: complex) -> complex:
    if q == 0:
        raise ValueError("q must be nonzero")
    return cmath.log(q) / TWO_PI_I


def q_from_tau(tau: complex) -> complex:
    return cmath.exp(TWO_PI_I * tau)


@dataclass(frozen=True)
class ModularReduction:
    tau_original: complex
    tau_reduced: complex
    q_reduced: complex
    transform: tuple[int, int, int, int]


def reduce_tau(tau: complex, max_steps: int = 100) -> ModularReduction:
    """Move a genus-one modulus to the standard SL(2,Z) fundamental domain."""
    a, b, c, d = 1, 0, 0, 1
    z = tau
    for _ in range(max_steps):
        n = math.floor(z.real + 0.5)
        if n:
            z = z - n
            a, b, c, d = a - n * c, b - n * d, c, d

        if abs(z) >= 1.0 - 1e-12 and -0.5 - 1e-12 <= z.real <= 0.5 + 1e-12:
            transformed = (a * tau + b) / (c * tau + d)
            if abs(transformed - z) > 1e-10:
                raise RuntimeError("internal SL(2,Z) metadata failed to reproduce reduced tau")
            return ModularReduction(tau, z, q_from_tau(z), (a, b, c, d))

        z = -1.0 / z
        a, b, c, d = -c, -d, a, b

    raise RuntimeError("failed to reduce tau")


def schottky_health(q_values: list[complex], threshold: float = 0.2) -> tuple[bool, str]:
    q_values = [complex(q) for q in q_values]
    if not q_values:
        return False, "Schottky series unsafe: no plumbing parameters supplied"
    if any(not (math.isfinite(q.real) and math.isfinite(q.imag)) for q in q_values):
        return False, "Schottky series unsafe: plumbing parameters must be finite"
    if any(abs(q) <= 1e-14 for q in q_values):
        return False, "Schottky series unsafe: plumbing parameters must be nonzero"
    max_abs = max(abs(q) for q in q_values)
    if max_abs >= 1.0:
        return False, f"Schottky series unsafe: max |q|={max_abs:.6g} is not inside the unit disk"
    if max_abs <= threshold:
        if len(q_values) == 3:
            try:
                generators = generators_for_glasses(q_values[0], q_values[1], q_values[2])
                fixed_points = [
                    point
                    for gen in generators
                    for point in (gen.attracting, gen.repelling)
                    if point is not INF
                ]
                min_sep = min(
                    abs(left - right)
                    for i, left in enumerate(fixed_points)
                    for right in fixed_points[i + 1 :]
                )
            except Exception as exc:
                return False, f"Schottky series unsafe: failed fixed-point health check ({exc})"
            if min_sep <= 1e-8:
                return False, f"Schottky series unsafe: fixed-point separation {min_sep:.3e} is too small"
            return (
                True,
                f"Schottky series expected to be healthy: max |q|={max_abs:.6g}, "
                f"min fixed-point separation={min_sep:.3e}",
            )
        return True, f"Schottky series expected to be healthy: max |q|={max_abs:.6g}"
    return False, f"Schottky series unsafe: max |q|={max_abs:.6g} exceeds {threshold:.6g}"


@dataclass(frozen=True)
class Mobius:
    a: complex
    b: complex
    c: complex
    d: complex

    def __call__(self, z: complex | None) -> complex | None:
        if z is INF:
            if abs(self.c) == 0:
                return INF
            return self.a / self.c
        denom = self.c * z + self.d
        if abs(denom) < 1e-15:
            return INF
        return (self.a * z + self.b) / denom

    def deriv(self, z: complex) -> complex:
        det = self.a * self.d - self.b * self.c
        return det / (self.c * z + self.d) ** 2

    def compose(self, other: "Mobius") -> "Mobius":
        """Return self after other: self(other(z))."""
        return Mobius(
            self.a * other.a + self.b * other.c,
            self.a * other.b + self.b * other.d,
            self.c * other.a + self.d * other.c,
            self.c * other.b + self.d * other.d,
        ).normalized()

    def inv(self) -> "Mobius":
        return Mobius(self.d, -self.b, -self.c, self.a).normalized()

    def normalized(self) -> "Mobius":
        scale = max(abs(self.a), abs(self.b), abs(self.c), abs(self.d))
        if scale == 0:
            raise ValueError("zero Mobius matrix")
        return Mobius(self.a / scale, self.b / scale, self.c / scale, self.d / scale)


IDENTITY = Mobius(1.0 + 0.0j, 0.0j, 0.0j, 1.0 + 0.0j)


@dataclass(frozen=True)
class GeneratorData:
    gamma: Mobius
    attracting: complex | None
    repelling: complex | None
    multiplier: complex


def dilation(q: complex) -> Mobius:
    return Mobius(q, 0.0j, 0.0j, 1.0 + 0.0j).normalized()


def bridge_map_mobius(q_bridge: complex) -> Mobius:
    # T(z)=1+q_bridge/(z-1)=(z+q_bridge-1)/(z-1).
    return Mobius(1.0 + 0.0j, q_bridge - 1.0, 1.0 + 0.0j, -1.0 + 0.0j).normalized()


def bridge_map(q_bridge: complex, z: complex) -> complex:
    return 1.0 + q_bridge / (z - 1.0)


def local_coordinate_map(puncture: str) -> Mobius:
    """Return phi_p(z), the local coordinate at a standard puncture."""
    if puncture == "zero":
        return IDENTITY
    if puncture == "one":
        return Mobius(1.0 + 0.0j, -1.0 + 0.0j, 0.0j, 1.0 + 0.0j).normalized()
    if puncture == "infty":
        return Mobius(0.0j, 1.0 + 0.0j, 1.0 + 0.0j, 0.0j).normalized()
    raise ValueError(f"unknown puncture {puncture!r}")


def inverse_local_coordinate_map(puncture: str) -> Mobius:
    return local_coordinate_map(puncture).inv()


def plumbing_transition(source_puncture: str, target_puncture: str, q: complex) -> Mobius:
    """Map source global coordinate to target global coordinate for uv=q."""
    q_over_u = Mobius(0.0j, q, 1.0 + 0.0j, 0.0j).normalized()
    return inverse_local_coordinate_map(target_puncture).compose(q_over_u).compose(
        local_coordinate_map(source_puncture)
    )


def mobius_fixed_points(gamma: Mobius) -> tuple[complex | None, complex | None]:
    """Return fixed points of a Mobius map, allowing infinity."""
    if gamma.c == 0:
        if abs(gamma.a - gamma.d) < 1e-14:
            raise ValueError("parabolic or identity fixed-point case is not supported here")
        finite = gamma.b / (gamma.d - gamma.a)
        return finite, INF

    linear = gamma.d - gamma.a
    discriminant = linear * linear + 4.0 * gamma.c * gamma.b
    sqrt_disc = cmath.sqrt(discriminant)
    numerators = (-linear + sqrt_disc, -linear - sqrt_disc)
    stable_numerator = max(numerators, key=abs)
    first = stable_numerator / (2.0 * gamma.c)
    # Vieta's relation avoids catastrophic cancellation for the second root.
    # This is essential in a deep plumbing cusp, where c can be 1e-100 while
    # both fixed points remain mathematically finite.
    if first == 0:
        second_numerator = min(numerators, key=abs)
        second = second_numerator / (2.0 * gamma.c)
    else:
        second = -gamma.b / (gamma.c * first)
    return first, second


def generator_data_from_mobius(gamma: Mobius) -> GeneratorData:
    """Orient a Mobius map so the stored multiplier has modulus <= 1 when possible."""
    fixed_a, fixed_b = mobius_fixed_points(gamma)
    if fixed_a is INF or fixed_b is INF:
        finite = fixed_b if fixed_a is INF else fixed_a
        if finite is INF:
            raise ValueError("expected one finite fixed point")
        multiplier = gamma.deriv(finite)
        if abs(multiplier) <= 1.0:
            return GeneratorData(gamma=gamma, attracting=finite, repelling=INF, multiplier=multiplier)
        inv = gamma.inv()
        return generator_data_from_mobius(inv)

    deriv_a = gamma.deriv(fixed_a)
    deriv_b = gamma.deriv(fixed_b)
    if abs(deriv_a) <= abs(deriv_b):
        return GeneratorData(gamma=gamma, attracting=fixed_a, repelling=fixed_b, multiplier=deriv_a)

    inv = gamma.inv()
    fixed_a, fixed_b = mobius_fixed_points(inv)
    if fixed_a is INF or fixed_b is INF:
        return generator_data_from_mobius(inv)
    deriv_a = inv.deriv(fixed_a)
    deriv_b = inv.deriv(fixed_b)
    if abs(deriv_a) <= abs(deriv_b):
        return GeneratorData(gamma=inv, attracting=fixed_a, repelling=fixed_b, multiplier=deriv_a)
    return GeneratorData(gamma=inv, attracting=fixed_b, repelling=fixed_a, multiplier=deriv_b)


def generators_for_glasses(q1: complex, q2: complex, q_bridge: complex) -> list[GeneratorData]:
    """Schottky generators for the genus-two glasses channel in the S1 coordinate."""
    t = bridge_map_mobius(q_bridge)
    g1 = dilation(q1)
    g2 = t.compose(dilation(q2)).compose(t)
    return [
        GeneratorData(gamma=g1, attracting=0.0j, repelling=INF, multiplier=q1),
        GeneratorData(gamma=g2, attracting=1.0 - q_bridge, repelling=1.0 + 0.0j, multiplier=q2),
    ]


def generators_for_sunrise(q0: complex, q1: complex, q2: complex) -> list[GeneratorData]:
    """Schottky generators for the sunrise channel.

    The three edges glue equal punctures on S1 and S2:
    zero-zero, one-one, and infinity-infinity.  The zero-zero edge is the
    spanning-tree edge.  The two non-tree edges give genus two.
    """
    tree = plumbing_transition("zero", "zero", q0)
    generators = []
    for puncture, q in [("one", q1), ("infty", q2)]:
        edge = plumbing_transition(puncture, puncture, q)
        gamma = tree.inv().compose(edge)
        generators.append(generator_data_from_mobius(gamma))
    return generators


def explicit_glasses_generators(q1: complex, q2: complex, q_bridge: complex) -> list[GeneratorData]:
    p = q2
    s = q_bridge
    g1 = Mobius(q1, 0.0j, 0.0j, 1.0 + 0.0j).normalized()
    g2 = Mobius(
        p + s - 1.0,
        (p - 1.0) * (s - 1.0),
        p - 1.0,
        p * (s - 1.0) + 1.0,
    ).normalized()
    return [
        GeneratorData(gamma=g1, attracting=0.0j, repelling=INF, multiplier=q1),
        GeneratorData(gamma=g2, attracting=1.0 - q_bridge, repelling=1.0 + 0.0j, multiplier=q2),
    ]


def conjugated_generators_for_glasses_s2(
    q1: complex, q2: complex, q_bridge: complex
) -> list[GeneratorData]:
    t = bridge_map_mobius(q_bridge)
    g1 = t.compose(dilation(q1)).compose(t)
    g2 = dilation(q2)
    return [
        GeneratorData(gamma=g1, attracting=1.0 - q_bridge, repelling=1.0 + 0.0j, multiplier=q1),
        GeneratorData(gamma=g2, attracting=0.0j, repelling=INF, multiplier=q2),
    ]

def _ordered_finite_fixed_points(transform: Mobius) -> tuple[complex, complex]:
    fixed_points = mobius_fixed_points(transform)
    if any(point is INF for point in fixed_points):
        raise ValueError("expected two finite fixed points")
    first, second = fixed_points
    def derivative_abs(point: complex) -> float:
        denominator = transform.c * point + transform.d
        if denominator == 0:
            return math.inf
        return abs(transform.deriv(point))

    first_derivative = derivative_abs(first)
    second_derivative = derivative_abs(second)
    if first_derivative <= second_derivative:
        return first, second
    return second, first


def _theta_one_generator(q_one: complex, q_infty: complex) -> GeneratorData:
    g_one = Mobius(
        q_infty * (q_one - 1.0),
        1.0 + 0.0j,
        -q_infty,
        1.0 + 0.0j,
    ).normalized()
    one_attracting, one_repelling = _ordered_finite_fixed_points(g_one)
    return GeneratorData(
        gamma=g_one,
        attracting=one_attracting,
        repelling=one_repelling,
        multiplier=g_one.deriv(one_attracting),
    )


def generators_for_theta(q_zero: complex, q_one: complex, q_infty: complex) -> list[GeneratorData]:
    r"""Schottky generators for the two-pants/theta genus-two plumbing chart.

    The two spheres have coordinates ``z`` and ``w`` and are plumbed by

        z w = q_zero,
        (z - 1) (w - 1) = q_one,
        (1 / z) (1 / w) = q_infty.

    We choose the infinity tube as the spanning-tree edge.  The zero and one
    tubes then give the two Schottky generators in the ``z`` coordinate.  The
    stored multipliers are exact Schottky multipliers, not merely the leading
    products ``q_zero*q_infty`` and ``q_one*q_infty``.
    """

    g_zero = dilation(q_zero * q_infty)
    return [
        GeneratorData(gamma=g_zero, attracting=0.0j, repelling=INF, multiplier=q_zero * q_infty),
        _theta_one_generator(q_one, q_infty),
    ]


def theta_cusp_surviving_multipliers(
    q_zero: complex,
    q_one: complex,
    q_infty: complex,
    *,
    log_q_values: Sequence[complex] | None = None,
    threshold: float = 1.0e-12,
) -> tuple[complex, ...] | None:
    """Return the rank-reduced theta Schottky multipliers in a deep cusp.

    The theta graph has three rank-one cycles, formed by the edge pairs
    ``(zero, one)``, ``(zero, infty)``, and ``(one, infty)``.  The infinity
    edge is only an auxiliary spanning-tree choice in
    :func:`generators_for_theta`; a small multiplier for one of those chosen
    generators does not by itself reduce the Schottky rank.  Reduction is
    therefore used only when at least two pair-cycle scales fall below the
    requested threshold.

    ``None`` means that the generic rank-two Schottky product must be used.
    A one-element tuple is the exact surviving rank-one multiplier in a
    single-edge pinch.  An empty tuple is the rank-zero double-cusp limit.
    """

    if not math.isfinite(threshold) or not 0.0 < threshold < 1.0:
        raise ValueError("theta cusp threshold must lie in (0,1)")
    q_values = (complex(q_zero), complex(q_one), complex(q_infty))
    logs = (
        tuple(cmath.log(value) for value in q_values)
        if log_q_values is None
        else tuple(complex(value) for value in log_q_values)
    )
    if len(logs) != 3 or any(not math.isfinite(value.real) for value in logs):
        raise ValueError("theta cusp reduction requires three finite logarithmic q values")

    cutoff = math.log(threshold)
    pair_logs = {
        (0, 1): logs[0] + logs[1],
        (0, 2): logs[0] + logs[2],
        (1, 2): logs[1] + logs[2],
    }
    vanishing_pairs = {
        pair for pair, log_value in pair_logs.items() if log_value.real < cutoff
    }
    if len(vanishing_pairs) < 2:
        return None

    surviving_pairs = tuple(pair for pair in pair_logs if pair not in vanishing_pairs)
    if not surviving_pairs:
        return ()
    if len(surviving_pairs) != 1:
        raise RuntimeError("theta cusp classification did not reduce to rank zero or one")

    surviving_pair = surviving_pairs[0]
    if surviving_pair == (0, 2):
        multiplier = cmath.exp(pair_logs[surviving_pair])
    elif surviving_pair == (1, 2):
        multiplier = _theta_one_generator(q_values[1], q_values[2]).multiplier
    else:
        # When the infinity edge pinches, the surviving word is g_zero
        # g_one^{-1}.  Removing the common q_infty scale from its matrix gives
        #
        #   [[q_zero, -q_zero], [1, q_one - 1]],
        #
        # whose trace and determinant are q_zero+q_one-1 and q_zero*q_one.
        trace = q_values[0] + q_values[1] - 1.0
        determinant = q_values[0] * q_values[1]
        root = cmath.sqrt(trace * trace - 4.0 * determinant)
        eigenvalue_a = 0.5 * (trace + root)
        eigenvalue_b = 0.5 * (trace - root)
        eigenvalue_large = (
            eigenvalue_a
            if abs(eigenvalue_a) >= abs(eigenvalue_b)
            else eigenvalue_b
        )
        if abs(eigenvalue_large) == 0.0:
            raise ValueError("surviving theta cusp cycle has no nonzero eigenvalue")
        multiplier = determinant / (eigenvalue_large * eigenvalue_large)
        if abs(multiplier) > 1.0:
            multiplier = 1.0 / multiplier

    if (
        not math.isfinite(multiplier.real)
        or not math.isfinite(multiplier.imag)
        or abs(multiplier) >= 1.0
    ):
        raise ValueError("surviving theta cusp multiplier is outside the unit disk")
    return (complex(multiplier),)


def inverse_letter(letter: int) -> int:
    return letter ^ 1


def letter_generator(generators: list[GeneratorData], letter: int) -> Mobius:
    idx = letter // 2
    gamma = generators[idx].gamma
    return gamma.inv() if letter % 2 else gamma


@lru_cache(maxsize=2)
def reduced_words(num_generators: int, max_len: int) -> tuple[tuple[int, ...], ...]:
    """Generate reduced words directly, without scanning all free words.

    For genus two this visits O(3**max_len) words instead of first generating
    O(4**max_len) candidates and discarding adjacent inverse pairs.  The small
    cache is enough to share a cutoff across all four period-matrix entries
    without retaining every adaptive cutoff for the lifetime of a worker.
    """

    if int(num_generators) < 1 or int(max_len) < 0:
        raise ValueError("word generation needs positive rank and nonnegative length")
    letters = tuple(range(2 * int(num_generators)))
    words: list[tuple[int, ...]] = [()]
    frontier: list[tuple[int, ...]] = [()]
    for _ in range(int(max_len)):
        next_frontier: list[tuple[int, ...]] = []
        for prefix in frontier:
            for letter in letters:
                if prefix and letter == inverse_letter(prefix[-1]):
                    continue
                next_frontier.append((*prefix, letter))
        words.extend(next_frontier)
        frontier = next_frontier
    return tuple(words)


def word_mobius(generators: list[GeneratorData], word: tuple[int, ...]) -> Mobius:
    out = IDENTITY
    for letter in word:
        out = out.compose(letter_generator(generators, letter))
    return out


def pole_term(z: complex, pole: complex | None) -> complex:
    if pole is INF:
        return 0.0j
    return 1.0 / (z - pole)


class SchottkySurface:
    """Normalized Schottky/Poincare-series holomorphic one-forms."""

    def __init__(self, generators: list[GeneratorData], max_word_len: int):
        self.generators = generators
        self.max_word_len = max_word_len
        self._words = reduced_words(len(generators), max_word_len)
        self._word_maps = {word: word_mobius(generators, word) for word in self._words}
        self._coset_words = {
            form_idx: [
                word
                for word in self._words
                if len(word) == 0 or word[-1] not in {2 * form_idx, 2 * form_idx + 1}
            ]
            for form_idx in range(len(generators))
        }

    def form_scalar(self, form_idx: int, z: complex) -> complex:
        gen = self.generators[form_idx]
        total = 0.0j
        for word in self._coset_words[form_idx]:
            gamma = self._word_maps[word]
            total += pole_term(z, gamma(gen.attracting)) - pole_term(z, gamma(gen.repelling))
        return total / TWO_PI_I

    def pullback_residual(self, gen_idx: int, form_idx: int, z: complex) -> complex:
        gamma = self.generators[gen_idx].gamma
        image = gamma(z)
        if image is INF:
            raise ValueError("probe point mapped to infinity")
        return self.form_scalar(form_idx, z) - self.form_scalar(form_idx, image) * gamma.deriv(z)

    def contour_period(self, form_idx: int, center: complex, radius: float, samples: int = 4096) -> complex:
        total = 0.0j
        dtheta = 2.0 * math.pi / samples
        for k in range(samples):
            theta = (k + 0.5) * dtheta
            z = center + radius * cmath.exp(1j * theta)
            dz = 1j * radius * cmath.exp(1j * theta) * dtheta
            total += self.form_scalar(form_idx, z) * dz
        return total

    def integrate_segment(self, form_idx: int, z0: complex, z1: complex, order: int = 600) -> complex:
        nodes, weights = np.polynomial.legendre.leggauss(order)
        midpoint = 0.5 * (z0 + z1)
        half = 0.5 * (z1 - z0)
        total = 0.0j
        for x, w in zip(nodes, weights):
            z = midpoint + half * x
            total += w * self.form_scalar(form_idx, z) * half
        return total

    def segment_pole_clearance(self, z0: complex, z1: complex, samples: int = 200) -> float:
        poles = self.finite_image_poles()
        if not poles:
            return math.inf
        best = math.inf
        for k in range(samples + 1):
            t = k / samples
            z = (1.0 - t) * z0 + t * z1
            best = min(best, min(abs(z - pole) for pole in poles))
        return float(best)

    def b_period_matrix(
        self,
        basepoints: list[complex],
        order: int = 600,
        pole_clearance: float | None = 1e-8,
    ) -> np.ndarray:
        genus = len(self.generators)
        omega = np.zeros((genus, genus), dtype=complex)
        for cycle_idx, z0 in enumerate(basepoints):
            z1 = self.generators[cycle_idx].gamma(z0)
            if z1 is INF:
                raise ValueError("B path endpoint landed at infinity")
            if pole_clearance is not None:
                clearance = self.segment_pole_clearance(z0, z1)
                if clearance < float(pole_clearance):
                    raise ValueError(
                        "B path is too close to a finite Schottky image pole: "
                        f"clearance={clearance:.3e}, threshold={float(pole_clearance):.3e}"
                    )
            for form_idx in range(genus):
                omega[cycle_idx, form_idx] = self.integrate_segment(form_idx, z0, z1, order=order)
        return omega.T

    def finite_image_poles(self) -> list[complex]:
        poles: list[complex] = []
        for form_idx, gen in enumerate(self.generators):
            for word in self._coset_words[form_idx]:
                gamma = self._word_maps[word]
                for fixed_point in [gen.attracting, gen.repelling]:
                    pole = gamma(fixed_point)
                    if pole is not INF:
                        poles.append(pole)
        return poles


def schottky_glasses_period_matrix(
    q1: complex,
    q2: complex,
    q_bridge: complex,
    max_word_len: int,
    b_order: int = 600,
) -> np.ndarray:
    surface = SchottkySurface(generators_for_glasses(q1, q2, q_bridge), max_word_len=max_word_len)
    return surface.b_period_matrix([-1.0 + 1.5j, 1.2 + 0.2j], order=b_order)


def schottky_sunrise_period_matrix(
    q0: complex,
    q1: complex,
    q2: complex,
    max_word_len: int,
    b_order: int = 600,
) -> np.ndarray:
    surface = SchottkySurface(generators_for_sunrise(q0, q1, q2), max_word_len=max_word_len)
    return surface.b_period_matrix([-0.45 + 0.9j, 1.45 + 0.9j], order=b_order)

def regularized_cross_ratio(
    a: complex | None,
    b: complex | None,
    c: complex | None,
    d: complex | None,
) -> complex:
    r"""Return ``((a-c)(b-d))/((a-d)(b-c))``, allowing points at infinity."""

    points = (a, b, c, d)
    if all(point is not INF for point in points):
        return ((a - c) * (b - d)) / ((a - d) * (b - c))

    shift = 0.123 + 0.456j
    finite_points = [point for point in points if point is not INF]
    while any(abs(point - shift) < 1.0e-12 for point in finite_points):
        shift += 0.271 + 0.319j

    def transform(point: complex | None) -> complex:
        if point is INF:
            return 0.0j
        return 1.0 / (point - shift)

    aa, bb, cc, dd = (transform(point) for point in points)
    return ((aa - cc) * (bb - dd)) / ((aa - dd) * (bb - cc))


@lru_cache(maxsize=8)
def schottky_double_coset_words(
    num_generators: int,
    left_generator: int,
    right_generator: int,
    max_word_len: int,
) -> tuple[tuple[int, ...], ...]:
    r"""Return canonical truncated representatives for ``<g_i>\G/<g_j>``."""

    left_letters = {2 * left_generator, 2 * left_generator + 1}
    right_letters = {2 * right_generator, 2 * right_generator + 1}
    representatives: list[tuple[int, ...]] = []
    for word in reduced_words(num_generators, max_word_len):
        if not word:
            if left_generator != right_generator:
                representatives.append(word)
            continue
        if word[0] in left_letters or word[-1] in right_letters:
            continue
        representatives.append(word)
    return tuple(representatives)


def schottky_period_matrix_cross_ratio(
    generators: list[GeneratorData],
    max_word_len: int,
) -> np.ndarray:
    r"""Compute the Schottky period matrix from the cross-ratio series.

    The formula is

    ``2*pi*i Omega_ij = delta_ij log(k_i) + sum log<a_i,b_i;gamma a_j,gamma b_j>``,

    where the sum is over representatives of
    ``<gamma_i>\Gamma/<gamma_j>`` and the identity representative is omitted on
    diagonal entries.  Increasing ``max_word_len`` generates the full finite-q
    relation order by order.
    """

    genus = len(generators)
    omega = np.zeros((genus, genus), dtype=np.complex128)
    for form_idx, left in enumerate(generators):
        for cycle_idx, right in enumerate(generators):
            total = cmath.log(left.multiplier) if form_idx == cycle_idx else 0.0j
            for word in schottky_double_coset_words(genus, form_idx, cycle_idx, max_word_len):
                transform = word_mobius(generators, word)
                total += cmath.log(
                    regularized_cross_ratio(
                        left.attracting,
                        left.repelling,
                        transform(right.attracting),
                        transform(right.repelling),
                    )
                )
            omega[form_idx, cycle_idx] = total / TWO_PI_I
    return omega


def schottky_period_matrix_cross_ratio_multiprecision(
    topology: str,
    q_values: Sequence[complex],
    max_word_len: int,
    *,
    dps: int | None = None,
    log_q_values: Sequence[complex] | None = None,
) -> np.ndarray:
    r"""Evaluate the genus-two Schottky cross-ratio sum without cusp collapse.

    In a deep plumbing cusp, coefficients such as ``q_infty * (q_one - 1)``
    lose the ``q_one`` term in binary64 arithmetic.  The resulting Mobius
    matrix is exactly singular even though the mathematical generator is
    loxodromic.  This evaluator keeps the same finite-word formula and branch
    convention as :func:`schottky_period_matrix_cross_ratio`, but performs
    generator construction, word composition, homogeneous cross ratios, and
    logarithms at multiprecision.  When ``log_q_values`` contains an edge
    beyond a useful direct precision, the finite corrections are evaluated at
    ``log|q|=-400`` and the exact tropical logarithm is restored analytically;
    the omitted dependence is then far below binary64 output precision.
    """

    import mpmath as mp

    q_complex = tuple(complex(value) for value in q_values)
    if len(q_complex) != 3:
        raise ValueError("genus-two multiprecision period map expects three q values")
    if topology not in {"theta", "glasses"}:
        raise ValueError(f"unknown topology {topology!r}")
    if log_q_values is None:
        q_abs = tuple(abs(value) for value in q_complex)
        if any(not (math.isfinite(value) and 0.0 < value < 1.0) for value in q_abs):
            raise ValueError("multiprecision period map expects 0 < |q_e| < 1")
        working_logs: tuple[complex, ...] | None = None
        required_dps = max(80, int(math.ceil(-math.log10(min(q_abs)))) + 40)
    else:
        true_logs = tuple(complex(value) for value in log_q_values)
        if len(true_logs) != 3:
            raise ValueError("genus-two multiprecision period map expects three log(q) values")
        if any(
            not (math.isfinite(value.real) and math.isfinite(value.imag) and value.real < 0.0)
            for value in true_logs
        ):
            raise ValueError("log(q) values must be finite with negative real part")
        # Terms proportional to exp(log q) below this floor are negligible at
        # binary64 output precision.  Evaluating them at the floor avoids a
        # thousands-of-digits calculation; the exact tropical logarithm is
        # restored in the period matrix below.
        working_logs = tuple(
            complex(max(value.real, MULTIPRECISION_ASYMPTOTIC_LOG_Q_FLOOR), value.imag)
            for value in true_logs
        )
        required_dps = max(
            80,
            int(math.ceil(-min(value.real for value in working_logs) / math.log(10.0))) + 40,
        )
    precision = required_dps if dps is None else max(int(dps), required_dps)

    def mpc(value: complex):
        value = complex(value)
        return mp.mpc(mp.mpf(repr(value.real)), mp.mpf(repr(value.imag)))

    def normalize_matrix(matrix):
        scale = max(abs(value) for row in matrix for value in row)
        if scale == 0:
            raise ValueError("zero multiprecision Mobius matrix")
        return tuple(tuple(value / scale for value in row) for row in matrix)

    def compose(left, right):
        return normalize_matrix(
            (
                (
                    left[0][0] * right[0][0] + left[0][1] * right[1][0],
                    left[0][0] * right[0][1] + left[0][1] * right[1][1],
                ),
                (
                    left[1][0] * right[0][0] + left[1][1] * right[1][0],
                    left[1][0] * right[0][1] + left[1][1] * right[1][1],
                ),
            )
        )

    def inverse(matrix):
        return normalize_matrix(
            ((matrix[1][1], -matrix[0][1]), (-matrix[1][0], matrix[0][0]))
        )

    def finite_point(value):
        return (value, mp.mpc(1))

    infinity_point = (mp.mpc(1), mp.mpc(0))

    def normalize_point(point):
        scale = max(abs(point[0]), abs(point[1]))
        if scale == 0:
            raise ValueError("zero multiprecision projective point")
        return (point[0] / scale, point[1] / scale)

    def apply(matrix, point):
        return normalize_point(
            (
                matrix[0][0] * point[0] + matrix[0][1] * point[1],
                matrix[1][0] * point[0] + matrix[1][1] * point[1],
            )
        )

    def point_det(left, right):
        return left[0] * right[1] - left[1] * right[0]

    def cross_ratio_log(a, b, c, d):
        determinants = (
            point_det(a, c),
            point_det(b, d),
            point_det(a, d),
            point_det(b, c),
        )
        if any(value == 0 for value in determinants):
            raise ZeroDivisionError("degenerate multiprecision Schottky cross ratio")
        return (
            mp.log(determinants[0])
            + mp.log(determinants[1])
            - mp.log(determinants[2])
            - mp.log(determinants[3])
        )

    def fixed_points(matrix):
        a, b = matrix[0]
        c, d = matrix[1]
        if c == 0:
            if a == d:
                raise ValueError("parabolic multiprecision Mobius generator")
            return finite_point(b / (d - a)), infinity_point
        linear = d - a
        root = mp.sqrt(linear * linear + 4 * c * b)
        numerators = (-linear + root, -linear - root)
        stable = max(numerators, key=abs)
        first = stable / (2 * c)
        second = min(numerators, key=abs) / (2 * c) if first == 0 else -b / (c * first)
        return finite_point(first), finite_point(second)

    def derivative(matrix, point):
        if point[1] == 0:
            raise ValueError("derivative at infinity is not needed for this generator")
        z = point[0] / point[1]
        determinant = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
        return determinant / (matrix[1][0] * z + matrix[1][1]) ** 2

    def orient_finite_generator(matrix):
        first, second = fixed_points(matrix)
        first_derivative = derivative(matrix, first)
        second_derivative = derivative(matrix, second)
        if abs(first_derivative) <= abs(second_derivative):
            return matrix, first, second, first_derivative
        return matrix, second, first, second_derivative

    with mp.workdps(precision):
        if working_logs is None:
            q0, q1, q2 = (mpc(value) for value in q_complex)
        else:
            q0, q1, q2 = (mp.exp(mpc(value)) for value in working_logs)
        zero = mp.mpc(0)
        one = mp.mpc(1)
        if topology == "theta":
            first_matrix = normalize_matrix(((q0 * q2, zero), (zero, one)))
            second_matrix = normalize_matrix(((q2 * (q1 - one), one), (-q2, one)))
            second_matrix, second_attracting, second_repelling, second_multiplier = (
                orient_finite_generator(second_matrix)
            )
            generators = (
                (first_matrix, finite_point(zero), infinity_point, q0 * q2),
                (second_matrix, second_attracting, second_repelling, second_multiplier),
            )
        else:
            first_matrix = normalize_matrix(((q0, zero), (zero, one)))
            bridge = normalize_matrix(((one, q2 - one), (one, -one)))
            second_matrix = compose(compose(bridge, ((q1, zero), (zero, one))), bridge)
            generators = (
                (first_matrix, finite_point(zero), infinity_point, q0),
                (
                    second_matrix,
                    finite_point(one - q2),
                    finite_point(one),
                    q1,
                ),
            )

        inverse_matrices = tuple(inverse(generator[0]) for generator in generators)

        def letter_matrix(letter: int):
            index = letter // 2
            return inverse_matrices[index] if letter % 2 else generators[index][0]

        def word_matrix(word: tuple[int, ...]):
            result = ((one, zero), (zero, one))
            for letter in word:
                result = compose(result, letter_matrix(letter))
            return result

        omega = np.zeros((2, 2), dtype=np.complex128)
        for form_index, left in enumerate(generators):
            for cycle_index, right in enumerate(generators):
                total = mp.log(left[3]) if form_index == cycle_index else mp.mpc(0)
                for word in schottky_double_coset_words(
                    2,
                    form_index,
                    cycle_index,
                    int(max_word_len),
                ):
                    transform = word_matrix(word)
                    total += cross_ratio_log(
                        left[1],
                        left[2],
                        apply(transform, right[1]),
                        apply(transform, right[2]),
                    )
                value = total / (2 * mp.pi * mp.j)
                omega[form_index, cycle_index] = complex(float(mp.re(value)), float(mp.im(value)))
        if working_logs is not None:
            delta_tau = tuple(
                (true - working) / TWO_PI_I
                for true, working in zip(true_logs, working_logs)
            )
            if topology == "theta":
                omega[0, 0] += delta_tau[0] + delta_tau[2]
                omega[0, 1] += delta_tau[2]
                omega[1, 0] += delta_tau[2]
                omega[1, 1] += delta_tau[1] + delta_tau[2]
            else:
                # In the glasses chart only the two handle sewing parameters
                # are logarithmic periods; the bridge controls Omega_12
                # algebraically and must remain representable.
                if true_logs[2].real != working_logs[2].real:
                    raise ValueError("an underflowing glasses bridge requires a separate degeneration chart")
                omega[0, 0] += delta_tau[0]
                omega[1, 1] += delta_tau[1]
        return omega


def theta_leading_period_matrix(q_zero: complex, q_one: complex, q_infty: complex) -> np.ndarray:
    r"""Return the tropical leading period matrix in the theta plumbing chart.

    The infinity tube is the reference edge and the cycles are
    ``zero - infinity`` and ``one - infinity``.  Branch choices are inherited
    from ``cmath.log``.
    """

    return np.asarray(
        [
            [cmath.log(q_zero * q_infty) / TWO_PI_I, cmath.log(q_infty) / TWO_PI_I],
            [cmath.log(q_infty) / TWO_PI_I, cmath.log(q_one * q_infty) / TWO_PI_I],
        ],
        dtype=np.complex128,
    )


def schottky_theta_period_matrix(
    q_zero: complex,
    q_one: complex,
    q_infty: complex,
    max_word_len: int,
    b_order: int = 600,
    basepoints: Sequence[complex] | None = None,
) -> np.ndarray:
    """Compute the theta-chart period matrix from the Schottky Poincare series."""

    if basepoints is None:
        basepoints = (-0.9 + 1.3j, 1.2 + 0.6j)
    surface = SchottkySurface(generators_for_theta(q_zero, q_one, q_infty), max_word_len=max_word_len)
    return surface.b_period_matrix(list(basepoints), order=b_order)


def schottky_theta_period_matrix_cross_ratio(
    q_zero: complex,
    q_one: complex,
    q_infty: complex,
    max_word_len: int,
) -> np.ndarray:
    """Compute the theta-chart period matrix from the Schottky cross-ratio series."""

    return schottky_period_matrix_cross_ratio(
        generators_for_theta(q_zero, q_one, q_infty),
        max_word_len=max_word_len,
    )


def _as_genus2_omega(omega) -> np.ndarray:
    omega = np.asarray(omega, dtype=np.complex128)
    if omega.shape != (2, 2):
        raise ValueError(f"Expected a 2x2 genus-two period matrix, got shape {omega.shape}.")
    if not np.all(np.isfinite(omega)):
        raise ValueError("Period matrix contains non-finite entries.")
    return omega


def genus2_symmetric_period_vector(omega) -> np.ndarray:
    """Return the six real components of a symmetric genus-two period matrix."""

    omega = _as_genus2_omega(omega)
    return np.asarray(
        [
            omega[0, 0].real,
            omega[0, 0].imag,
            omega[1, 1].real,
            omega[1, 1].imag,
            omega[0, 1].real,
            omega[0, 1].imag,
        ],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class GlassesInverseSeed:
    """Initial guess for the genus-two glasses inverse period problem."""

    tau1: complex
    tau2: complex
    q1: complex
    q2: complex
    q3: complex
    source: str


@dataclass(frozen=True)
class GlassesInverseResult:
    """Result of solving Omega(q1,q2,q3)=Omega_target in the glasses chart."""

    target_omega: np.ndarray
    omega: np.ndarray
    residual_matrix: np.ndarray
    residual_vector: np.ndarray
    residual_norm: float
    max_abs_residual: float
    tau1: complex
    tau2: complex
    q1: complex
    q2: complex
    q3: complex
    initial_seed: GlassesInverseSeed
    success: bool
    message: str
    cost: float
    optimality: float
    nfev: int
    max_word_len: int
    b_order: int
    q_abs_warning_threshold: float
    health_message: str


@dataclass(frozen=True)
class ThetaInverseSeed:
    """Initial guess for the genus-two theta-graph inverse period problem."""

    tau_zero: complex
    tau_one: complex
    tau_infty: complex
    q_zero: complex
    q_one: complex
    q_infty: complex
    source: str


@dataclass(frozen=True)
class ThetaInverseResult:
    """Result of solving Omega(q_zero,q_one,q_infty)=Omega_target."""

    target_omega: np.ndarray
    omega: np.ndarray
    residual_matrix: np.ndarray
    residual_vector: np.ndarray
    residual_norm: float
    max_abs_residual: float
    tau_zero: complex
    tau_one: complex
    tau_infty: complex
    q_zero: complex
    q_one: complex
    q_infty: complex
    initial_seed: ThetaInverseSeed
    success: bool
    message: str
    cost: float
    optimality: float
    nfev: int
    max_word_len: int
    b_order: int
    q_abs_warning_threshold: float
    health_message: str


def glasses_inverse_seed_from_omega(
    target_omega,
    *,
    bridge_linear_coefficient: complex = -TWO_PI_I,
    tau_imag_floor: float = 1.0e-6,
    tau_imag_ceiling: float = 10.0,
    q3_component_bound: float = 0.95,
) -> GlassesInverseSeed:
    r"""
    Build the standard asymptotic seed for the glasses inverse map.

    In this chart the diagonal periods have logarithmic leading terms,

        Omega_11 ~ log(q1)/(2*pi*i),   Omega_22 ~ log(q2)/(2*pi*i),

    so the seed uses ``q_i=exp(2*pi*i Omega_ii)``.  The bridge seed is the
    configurable linear approximation ``q3 ~= bridge_linear_coefficient *
    Omega_12``.  The default coefficient is the convention calibrated by the
    current forward Schottky map near the sample small-q region.
    """

    omega = _as_genus2_omega(target_omega)

    def clamp_tau_imag(tau: complex) -> complex:
        tau_imag = min(max(float(tau.imag), float(tau_imag_floor)), float(tau_imag_ceiling))
        return complex(float(tau.real), tau_imag)

    def clamp_component(value: float) -> float:
        bound = float(q3_component_bound)
        return min(max(float(value), -bound), bound)

    tau1 = clamp_tau_imag(complex(omega[0, 0]))
    tau2 = clamp_tau_imag(complex(omega[1, 1]))
    q3_seed = complex(bridge_linear_coefficient * omega[0, 1])
    q3_seed = complex(clamp_component(q3_seed.real), clamp_component(q3_seed.imag))
    if q3_seed == 0:
        q3_seed = complex(float(q3_component_bound) * 1.0e-6, 0.0)

    return GlassesInverseSeed(
        tau1=tau1,
        tau2=tau2,
        q1=q_from_tau(tau1),
        q2=q_from_tau(tau2),
        q3=q3_seed,
        source="Omega diagonal logarithms and linear bridge approximation",
    )


def theta_inverse_seed_from_omega(
    target_omega,
    *,
    tau_imag_floor: float = 1.0e-6,
    tau_imag_ceiling: float = 10.0,
) -> ThetaInverseSeed:
    r"""
    Build the standard asymptotic seed for the theta-graph inverse map.

    With the infinity tube as reference edge,

        Omega_11 ~ tau_zero + tau_infty,
        Omega_22 ~ tau_one + tau_infty,
        Omega_12 ~ tau_infty,

    where ``q_edge = exp(2*pi*i*tau_edge)``.
    """

    omega = _as_genus2_omega(target_omega)

    def clamp_tau(tau: complex) -> complex:
        tau_imag = min(max(float(tau.imag), float(tau_imag_floor)), float(tau_imag_ceiling))
        return complex(float(tau.real), tau_imag)

    tau_infty = clamp_tau(complex(omega[0, 1]))
    tau_zero = clamp_tau(complex(omega[0, 0] - omega[0, 1]))
    tau_one = clamp_tau(complex(omega[1, 1] - omega[0, 1]))

    return ThetaInverseSeed(
        tau_zero=tau_zero,
        tau_one=tau_one,
        tau_infty=tau_infty,
        q_zero=q_from_tau(tau_zero),
        q_one=q_from_tau(tau_one),
        q_infty=q_from_tau(tau_infty),
        source="theta-graph logarithmic period approximation",
    )


def _pack_glasses_inverse_variables(seed: GlassesInverseSeed) -> np.ndarray:
    return np.asarray(
        [
            seed.tau1.real,
            seed.tau1.imag,
            seed.tau2.real,
            seed.tau2.imag,
            seed.q3.real,
            seed.q3.imag,
        ],
        dtype=np.float64,
    )


def _unpack_glasses_inverse_variables(x: Sequence[float]) -> tuple[complex, complex, complex, complex, complex]:
    arr = np.asarray(x, dtype=np.float64)
    if arr.shape != (6,):
        raise ValueError(f"Expected six inverse variables, got shape {arr.shape}.")
    tau1 = complex(float(arr[0]), float(arr[1]))
    tau2 = complex(float(arr[2]), float(arr[3]))
    q1 = q_from_tau(tau1)
    q2 = q_from_tau(tau2)
    q3 = complex(float(arr[4]), float(arr[5]))
    return tau1, tau2, q1, q2, q3


def _pack_theta_inverse_variables(seed: ThetaInverseSeed) -> np.ndarray:
    return np.asarray(
        [
            seed.tau_zero.real,
            seed.tau_zero.imag,
            seed.tau_one.real,
            seed.tau_one.imag,
            seed.tau_infty.real,
            seed.tau_infty.imag,
        ],
        dtype=np.float64,
    )


def _unpack_theta_inverse_variables(x: Sequence[float]) -> tuple[complex, complex, complex, complex, complex, complex]:
    arr = np.asarray(x, dtype=np.float64)
    if arr.shape != (6,):
        raise ValueError(f"Expected six inverse variables, got shape {arr.shape}.")
    tau_zero = complex(float(arr[0]), float(arr[1]))
    tau_one = complex(float(arr[2]), float(arr[3]))
    tau_infty = complex(float(arr[4]), float(arr[5]))
    q_zero = q_from_tau(tau_zero)
    q_one = q_from_tau(tau_one)
    q_infty = q_from_tau(tau_infty)
    return tau_zero, tau_one, tau_infty, q_zero, q_one, q_infty


def solve_glasses_inverse_from_omega(
    target_omega,
    *,
    initial_seed: GlassesInverseSeed | None = None,
    initial_q: tuple[complex, complex, complex] | None = None,
    max_word_len: int = 4,
    b_order: int = 200,
    max_nfev: int = 80,
    tau_imag_bounds: tuple[float, float] = (1.0e-6, 10.0),
    q3_component_bound: float = 0.95,
    residual_weights: Sequence[float] | None = None,
    bridge_linear_coefficient: complex = -TWO_PI_I,
    q_abs_warning_threshold: float = 0.2,
    least_squares_kwargs: dict | None = None,
) -> GlassesInverseResult:
    r"""
    Invert the genus-two glasses period map by local nonlinear least squares.

    This solves, in one fixed plumbing/homology chart,

        Omega_glasses(q1,q2,q3) = target_omega.

    The diagonal plumbing multipliers are parameterized by
    ``q_i=exp(2*pi*i*tau_i)`` to avoid the logarithmic conditioning problem in
    direct q-coordinates.  The bridge parameter is solved directly as a complex
    number.  This is intentionally a local chart solver: the caller is
    responsible for putting ``target_omega`` in the same symplectic frame as
    the glasses forward map.
    """

    target = _as_genus2_omega(target_omega)
    target_vec = genus2_symmetric_period_vector(target)
    if residual_weights is None:
        weights = np.ones(6, dtype=np.float64)
    else:
        weights = np.asarray(tuple(float(value) for value in residual_weights), dtype=np.float64)
        if weights.shape != (6,):
            raise ValueError(f"residual_weights must have length 6, got shape {weights.shape}.")
        if np.any(weights <= 0.0):
            raise ValueError("residual_weights must be positive.")

    if initial_seed is not None and initial_q is not None:
        raise ValueError("Pass either initial_seed or initial_q, not both.")
    if initial_q is not None:
        q1, q2, q3 = (complex(value) for value in initial_q)
        if q1 == 0 or q2 == 0 or q3 == 0:
            raise ValueError("initial_q entries must be nonzero.")
        initial_seed = GlassesInverseSeed(
            tau1=tau_from_q(q1),
            tau2=tau_from_q(q2),
            q1=q1,
            q2=q2,
            q3=q3,
            source="caller-provided q values",
        )
    if initial_seed is None:
        initial_seed = glasses_inverse_seed_from_omega(
            target,
            bridge_linear_coefficient=bridge_linear_coefficient,
            tau_imag_floor=float(tau_imag_bounds[0]),
            tau_imag_ceiling=float(tau_imag_bounds[1]),
            q3_component_bound=float(q3_component_bound),
        )

    x0 = _pack_glasses_inverse_variables(initial_seed)
    lower = np.asarray(
        [-np.inf, float(tau_imag_bounds[0]), -np.inf, float(tau_imag_bounds[0]), -float(q3_component_bound), -float(q3_component_bound)],
        dtype=np.float64,
    )
    upper = np.asarray(
        [np.inf, float(tau_imag_bounds[1]), np.inf, float(tau_imag_bounds[1]), float(q3_component_bound), float(q3_component_bound)],
        dtype=np.float64,
    )
    x0 = np.minimum(np.maximum(x0, lower), upper)
    projected_tau1, projected_tau2, projected_q1, projected_q2, projected_q3 = _unpack_glasses_inverse_variables(x0)
    if (
        projected_tau1 != initial_seed.tau1
        or projected_tau2 != initial_seed.tau2
        or projected_q3 != initial_seed.q3
    ):
        initial_seed = GlassesInverseSeed(
            tau1=projected_tau1,
            tau2=projected_tau2,
            q1=projected_q1,
            q2=projected_q2,
            q3=projected_q3,
            source=f"{initial_seed.source} (projected to solver bounds)",
        )

    def residual(x: np.ndarray) -> np.ndarray:
        tau1, tau2, q1, q2, q3 = _unpack_glasses_inverse_variables(x)
        if tau1.imag <= 0.0 or tau2.imag <= 0.0 or abs(q1) >= 1.0 or abs(q2) >= 1.0 or q3 == 0:
            return 1.0e6 * np.ones(6, dtype=np.float64)
        try:
            omega = schottky_glasses_period_matrix(q1, q2, q3, max_word_len=max_word_len, b_order=b_order)
            return weights * (genus2_symmetric_period_vector(omega) - target_vec)
        except Exception:
            return 1.0e6 * np.ones(6, dtype=np.float64)

    try:
        from scipy.optimize import least_squares
    except ImportError as exc:  # pragma: no cover - scipy is available in the main project env
        raise RuntimeError("solve_glasses_inverse_from_omega requires scipy.optimize.least_squares.") from exc

    kwargs = dict(least_squares_kwargs or {})
    kwargs.setdefault("xtol", 1.0e-10)
    kwargs.setdefault("ftol", 1.0e-10)
    kwargs.setdefault("gtol", 1.0e-10)
    kwargs.setdefault("max_nfev", int(max_nfev))
    opt = least_squares(residual, x0, bounds=(lower, upper), **kwargs)

    tau1, tau2, q1, q2, q3 = _unpack_glasses_inverse_variables(opt.x)
    omega = schottky_glasses_period_matrix(q1, q2, q3, max_word_len=max_word_len, b_order=b_order)
    residual_matrix = np.asarray(omega - target, dtype=np.complex128)
    residual_vector = genus2_symmetric_period_vector(residual_matrix)
    residual_norm = float(np.linalg.norm(residual_vector))
    max_abs_residual = float(np.max(np.abs(residual_matrix)))
    healthy, health_message = schottky_health([q1, q2], threshold=float(q_abs_warning_threshold))
    bridge_abs = abs(q3)
    if bridge_abs > float(q_abs_warning_threshold):
        health_message = f"{health_message}; bridge |q3|={bridge_abs:.6g} exceeds {float(q_abs_warning_threshold):.6g}"
    elif healthy:
        health_message = f"{health_message}; bridge |q3|={bridge_abs:.6g}"

    return GlassesInverseResult(
        target_omega=target,
        omega=omega,
        residual_matrix=residual_matrix,
        residual_vector=residual_vector,
        residual_norm=residual_norm,
        max_abs_residual=max_abs_residual,
        tau1=tau1,
        tau2=tau2,
        q1=q1,
        q2=q2,
        q3=q3,
        initial_seed=initial_seed,
        success=bool(opt.success),
        message=str(opt.message),
        cost=float(opt.cost),
        optimality=float(opt.optimality),
        nfev=int(opt.nfev),
        max_word_len=int(max_word_len),
        b_order=int(b_order),
        q_abs_warning_threshold=float(q_abs_warning_threshold),
        health_message=health_message,
    )


def solve_theta_inverse_from_omega(
    target_omega,
    *,
    initial_seed: ThetaInverseSeed | None = None,
    initial_q: tuple[complex, complex, complex] | None = None,
    max_word_len: int = 5,
    b_order: int = 250,
    max_nfev: int = 80,
    tau_imag_bounds: tuple[float, float] = (1.0e-6, 10.0),
    residual_weights: Sequence[float] | None = None,
    q_abs_warning_threshold: float = 0.2,
    least_squares_kwargs: dict | None = None,
) -> ThetaInverseResult:
    r"""
    Invert the local theta-graph period map by nonlinear least squares.

    This is the numerical finite-q counterpart of

        q_infty ~= exp(2*pi*i Omega_12),
        q_zero  ~= exp(2*pi*i (Omega_11 - Omega_12)),
        q_one   ~= exp(2*pi*i (Omega_22 - Omega_12)).

    The forward map is ``schottky_theta_period_matrix``; therefore the solve
    includes all corrections captured by the selected Schottky word cutoff.
    """

    target = _as_genus2_omega(target_omega)
    target_vec = genus2_symmetric_period_vector(target)
    if residual_weights is None:
        weights = np.ones(6, dtype=np.float64)
    else:
        weights = np.asarray(tuple(float(value) for value in residual_weights), dtype=np.float64)
        if weights.shape != (6,):
            raise ValueError(f"residual_weights must have length 6, got shape {weights.shape}.")
        if np.any(weights <= 0.0):
            raise ValueError("residual_weights must be positive.")

    if initial_seed is not None and initial_q is not None:
        raise ValueError("Pass either initial_seed or initial_q, not both.")
    if initial_q is not None:
        q_zero, q_one, q_infty = (complex(value) for value in initial_q)
        if q_zero == 0 or q_one == 0 or q_infty == 0:
            raise ValueError("initial_q entries must be nonzero.")
        initial_seed = ThetaInverseSeed(
            tau_zero=tau_from_q(q_zero),
            tau_one=tau_from_q(q_one),
            tau_infty=tau_from_q(q_infty),
            q_zero=q_zero,
            q_one=q_one,
            q_infty=q_infty,
            source="caller-provided theta q values",
        )
    if initial_seed is None:
        initial_seed = theta_inverse_seed_from_omega(
            target,
            tau_imag_floor=float(tau_imag_bounds[0]),
            tau_imag_ceiling=float(tau_imag_bounds[1]),
        )

    x0 = _pack_theta_inverse_variables(initial_seed)
    lower = np.asarray(
        [
            -np.inf,
            float(tau_imag_bounds[0]),
            -np.inf,
            float(tau_imag_bounds[0]),
            -np.inf,
            float(tau_imag_bounds[0]),
        ],
        dtype=np.float64,
    )
    upper = np.asarray(
        [
            np.inf,
            float(tau_imag_bounds[1]),
            np.inf,
            float(tau_imag_bounds[1]),
            np.inf,
            float(tau_imag_bounds[1]),
        ],
        dtype=np.float64,
    )
    x0 = np.minimum(np.maximum(x0, lower), upper)
    projected = _unpack_theta_inverse_variables(x0)
    if (
        projected[0] != initial_seed.tau_zero
        or projected[1] != initial_seed.tau_one
        or projected[2] != initial_seed.tau_infty
    ):
        initial_seed = ThetaInverseSeed(
            tau_zero=projected[0],
            tau_one=projected[1],
            tau_infty=projected[2],
            q_zero=projected[3],
            q_one=projected[4],
            q_infty=projected[5],
            source=f"{initial_seed.source} (projected to solver bounds)",
        )

    def residual(x: np.ndarray) -> np.ndarray:
        tau_zero, tau_one, tau_infty, q_zero, q_one, q_infty = _unpack_theta_inverse_variables(x)
        if (
            tau_zero.imag <= 0.0
            or tau_one.imag <= 0.0
            or tau_infty.imag <= 0.0
            or abs(q_zero) >= 1.0
            or abs(q_one) >= 1.0
            or abs(q_infty) >= 1.0
        ):
            return 1.0e6 * np.ones(6, dtype=np.float64)
        try:
            omega = schottky_theta_period_matrix(
                q_zero,
                q_one,
                q_infty,
                max_word_len=max_word_len,
                b_order=b_order,
            )
            return weights * (genus2_symmetric_period_vector(omega) - target_vec)
        except Exception:
            return 1.0e6 * np.ones(6, dtype=np.float64)

    try:
        from scipy.optimize import least_squares
    except ImportError as exc:  # pragma: no cover - scipy is available in the main project env
        raise RuntimeError("solve_theta_inverse_from_omega requires scipy.optimize.least_squares.") from exc

    kwargs = dict(least_squares_kwargs or {})
    kwargs.setdefault("xtol", 1.0e-10)
    kwargs.setdefault("ftol", 1.0e-10)
    kwargs.setdefault("gtol", 1.0e-10)
    kwargs.setdefault("max_nfev", int(max_nfev))
    opt = least_squares(residual, x0, bounds=(lower, upper), **kwargs)

    tau_zero, tau_one, tau_infty, q_zero, q_one, q_infty = _unpack_theta_inverse_variables(opt.x)
    omega = schottky_theta_period_matrix(
        q_zero,
        q_one,
        q_infty,
        max_word_len=max_word_len,
        b_order=b_order,
    )
    residual_matrix = np.asarray(omega - target, dtype=np.complex128)
    residual_vector = genus2_symmetric_period_vector(residual_matrix)
    residual_norm = float(np.linalg.norm(residual_vector))
    max_abs_residual = float(np.max(np.abs(residual_matrix)))

    generators = generators_for_theta(q_zero, q_one, q_infty)
    max_edge_q = max(abs(q_zero), abs(q_one), abs(q_infty))
    max_multiplier = max(abs(generator.multiplier) for generator in generators)
    if max_edge_q > float(q_abs_warning_threshold) or max_multiplier > float(q_abs_warning_threshold):
        health_message = (
            f"theta chart may need higher word cutoff: max edge |q|={max_edge_q:.6g}, "
            f"max Schottky |k|={max_multiplier:.6g}, threshold={float(q_abs_warning_threshold):.6g}"
        )
    else:
        health_message = (
            f"theta Schottky series expected healthy: max edge |q|={max_edge_q:.6g}, "
            f"max Schottky |k|={max_multiplier:.6g}"
        )

    return ThetaInverseResult(
        target_omega=target,
        omega=omega,
        residual_matrix=residual_matrix,
        residual_vector=residual_vector,
        residual_norm=residual_norm,
        max_abs_residual=max_abs_residual,
        tau_zero=tau_zero,
        tau_one=tau_one,
        tau_infty=tau_infty,
        q_zero=q_zero,
        q_one=q_one,
        q_infty=q_infty,
        initial_seed=initial_seed,
        success=bool(opt.success),
        message=str(opt.message),
        cost=float(opt.cost),
        optimality=float(opt.optimality),
        nfev=int(opt.nfev),
        max_word_len=int(max_word_len),
        b_order=int(b_order),
        q_abs_warning_threshold=float(q_abs_warning_threshold),
        health_message=health_message,
    )


@dataclass(frozen=True)
class TorusCollocationResult:
    q: complex
    radius: float
    order: int
    samples: int
    singular_values: np.ndarray
    coeffs: np.ndarray
    powers: np.ndarray
    a_period: complex
    b_period: complex
    b_period_quadrature: complex
    max_seam_residual: float
    max_coeff_error: float
    tau_error: float


def torus_seam_matrix(q: complex, radius: float, order: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    powers = np.arange(-order, order + 1)
    rows = []
    for k in range(samples):
        theta = 2.0 * math.pi * k / samples
        z_left = radius * cmath.exp(1j * theta)
        z_right = z_left / q
        rows.append((z_left ** powers) - (z_right ** powers) / q)
    return np.asarray(rows, dtype=complex), powers


def eval_laurent(coeffs: np.ndarray, powers: np.ndarray, z: complex) -> complex:
    return complex(np.sum(coeffs * (z ** powers)))


def integrate_laurent_segment(
    coeffs: np.ndarray, powers: np.ndarray, z0: complex, z1: complex, order: int = 400
) -> complex:
    nodes, weights = np.polynomial.legendre.leggauss(order)
    midpoint = 0.5 * (z0 + z1)
    half = 0.5 * (z1 - z0)
    total = 0.0j
    for x, w in zip(nodes, weights):
        z = midpoint + half * x
        total += w * eval_laurent(coeffs, powers, z) * half
    return total


def solve_torus_collocation(q: complex, radius: float | None = None, order: int = 8, samples: int = 64) -> TorusCollocationResult:
    if not (0.0 < abs(q) < 1.0):
        raise ValueError("torus collocation expects 0 < |q| < 1")
    if radius is None:
        radius = math.sqrt(abs(q))
    if not (abs(q) < radius < 1.0):
        raise ValueError("choose radius with |q| < radius < 1")

    matrix, powers = torus_seam_matrix(q, radius, order, samples)
    _, singular_values, vh = np.linalg.svd(matrix)
    coeffs = vh[-1, :]
    residue_idx = np.where(powers == -1)[0][0]
    coeffs = coeffs / (TWO_PI_I * coeffs[residue_idx])
    a_period = TWO_PI_I * coeffs[residue_idx]

    z0 = radius * cmath.exp(0.37j)
    b_period_quadrature = integrate_laurent_segment(coeffs, powers, z0, q * z0)
    b_period = coeffs[residue_idx] * cmath.log(q)

    max_seam_residual = 0.0
    for k in range(samples):
        theta = 2.0 * math.pi * k / samples
        z_left = radius * cmath.exp(1j * theta)
        z_right = z_left / q
        residual = eval_laurent(coeffs, powers, z_left) - eval_laurent(coeffs, powers, z_right) / q
        max_seam_residual = max(max_seam_residual, abs(residual))

    exact = np.zeros_like(coeffs)
    exact[residue_idx] = 1.0 / TWO_PI_I
    return TorusCollocationResult(
        q=q,
        radius=radius,
        order=order,
        samples=samples,
        singular_values=singular_values,
        coeffs=coeffs,
        powers=powers,
        a_period=a_period,
        b_period=b_period,
        b_period_quadrature=b_period_quadrature,
        max_seam_residual=max_seam_residual,
        max_coeff_error=float(np.max(np.abs(coeffs - exact))),
        tau_error=abs(b_period - tau_from_q(q)),
    )


@dataclass(frozen=True)
class GlassesCollocationResult:
    q1: complex
    q2: complex
    q3: complex
    basis_order: int
    samples_per_seam: int
    schottky_word_len: int
    singular_values: np.ndarray
    constraint_singular_values: np.ndarray
    a_period_matrix: np.ndarray
    omega_b_annular: np.ndarray
    schottky_omega: np.ndarray
    max_seam_residual: float
    schottky_error_absolute: float
    schottky_error_relative: float
    omega_symmetry_error: float


@dataclass(frozen=True)
class PlumbingToRibbonResult:
    """One fixed-perimeter inverse fit from plumbing moduli to genus-two lengths."""

    channel: str
    q_values: tuple[complex, complex, complex]
    plumbing_algorithm: str
    topology: int
    edge_lengths: tuple[int, ...]
    total_edge_length: int
    omega_target: np.ndarray
    omega_ribbon: np.ndarray
    period_residual: float
    balance_penalty: float
    objective: float
    candidates_evaluated: int
    local_moves_accepted: int
    search_edge_lengths: tuple[int, ...] | None = None
    search_total_edge_length: int | None = None


@dataclass(frozen=True)
class Genus1PlumbingToRibbonResult:
    """Fixed-perimeter theta-graph representative for a genus-one plumbing modulus."""

    q: complex
    tau_target: complex
    tau_target_reduced: complex
    edge_lengths: tuple[int, int, int]
    total_edge_length: int
    tau_ribbon: complex
    tau_ribbon_reduced: complex
    tau_residual: float
    balance_penalty: float
    objective: float
    candidates_evaluated: int


@dataclass(frozen=True)
class Genus1LookupTable:
    """Coarse fixed-perimeter theta-graph atlas for genus one."""

    total_edge_length: int
    minimum_edge_length: int
    edge_lengths: np.ndarray
    tau: np.ndarray


@dataclass(frozen=True)
class Genus1TableRefinedResult:
    """Genus-one inverse from lookup-table seed plus projected simplex descent."""

    q: complex
    tau_target: complex
    tau_target_reduced: complex
    rough_edge_lengths: tuple[int, int, int]
    edge_lengths: tuple[int, int, int]
    total_edge_length: int
    tau_ribbon: complex
    tau_ribbon_reduced: complex
    tau_residual: float
    q_reconstructed: complex
    q_error_abs: float
    q_error_rel: float
    balance_penalty: float
    objective: float
    table_candidates: int
    refinement_evaluations: int
    descent_moves_accepted: int

@dataclass(frozen=True)
class ThetaCollocationResult:
    q_zero: complex
    q_one: complex
    q_infty: complex
    basis_order: int
    samples_per_seam: int
    radii: tuple[float, float, float]
    singular_values: np.ndarray
    constraint_singular_values: np.ndarray
    a_period_matrix: np.ndarray
    omega: np.ndarray
    max_seam_residual: float
    omega_symmetry_error: float


def glasses_basis_scalar(puncture: str, n: int, z: complex) -> complex:
    if puncture == "zero":
        return z ** (-n)
    if puncture == "one":
        return (z - 1.0) ** (-n)
    if puncture == "infty":
        return -(z ** (n - 2))
    raise ValueError(f"unknown puncture {puncture!r}")


def glasses_basis_index(basis_order: int) -> list[tuple[int, str, int]]:
    index = []
    for sphere in (0, 1):
        for puncture in PUNCTURES:
            first_mode = 2 if puncture == "infty" else 1
            for n in range(first_mode, basis_order + 1):
                index.append((sphere, puncture, n))
    return index


def glasses_eval_row(index: list[tuple[int, str, int]], sphere: int, z: complex) -> np.ndarray:
    row = np.zeros(len(index), dtype=complex)
    for col, (basis_sphere, puncture, n) in enumerate(index):
        if basis_sphere == sphere:
            row[col] = glasses_basis_scalar(puncture, n, z)
    return row


def glasses_seam_matrix(
    q1: complex,
    q2: complex,
    q3: complex,
    basis_order: int,
    samples_per_seam: int,
    r1: float,
    r2: float,
    r3: float,
) -> tuple[np.ndarray, list[tuple[int, str, int]]]:
    index = glasses_basis_index(basis_order)
    rows = []

    for sphere, q, radius in [(0, q1, r1), (1, q2, r2)]:
        for k in range(samples_per_seam):
            theta = 2.0 * math.pi * k / samples_per_seam
            z_left = radius * cmath.exp(1j * theta)
            z_right = z_left / q
            rows.append(glasses_eval_row(index, sphere, z_left) - glasses_eval_row(index, sphere, z_right) / q)

    for k in range(samples_per_seam):
        theta = 2.0 * math.pi * k / samples_per_seam
        z1 = 1.0 + r3 * cmath.exp(1j * theta)
        z2 = bridge_map(q3, z1)
        dz2_dz1 = -q3 / (z1 - 1.0) ** 2
        rows.append(glasses_eval_row(index, 0, z1) - dz2_dz1 * glasses_eval_row(index, 1, z2))

    return np.asarray(rows, dtype=complex), index


def validate_glasses_collocation_inputs(
    q1: complex,
    q2: complex,
    q3: complex,
    basis_order: int,
    samples_per_seam: int,
    radii: tuple[float, float, float],
) -> None:
    q_values = (complex(q1), complex(q2), complex(q3))
    if any(not (math.isfinite(q.real) and math.isfinite(q.imag)) for q in q_values):
        raise ValueError("plumbing parameters must be finite")
    if any(abs(q) <= 1e-14 for q in q_values):
        raise ValueError("plumbing parameters must be nonzero for collocation")
    if int(basis_order) < 2:
        raise ValueError("basis_order must be at least 2")
    if int(samples_per_seam) < 2 * int(basis_order):
        raise ValueError("samples_per_seam must be at least twice basis_order")
    if len(radii) != 3:
        raise ValueError("radii must contain exactly three seam radii")
    for radius in radii:
        if not math.isfinite(float(radius)) or float(radius) <= 0.0:
            raise ValueError("seam radii must be positive finite numbers")
    for q, radius in zip(q_values, radii):
        if not (abs(q) < float(radius) < 1.0):
            raise ValueError(
                "each seam radius must satisfy |q| < radius < 1; "
                f"got |q|={abs(q):.6g}, radius={float(radius):.6g}"
            )
    if abs(q3) / float(radii[2]) >= 1.0:
        raise ValueError("bridge seam image radius |q3|/r3 must be smaller than 1")


def glasses_period_constraint_matrix(index: list[tuple[int, str, int]]) -> np.ndarray:
    matrix = np.zeros((2, len(index)), dtype=complex)
    for col, (sphere, puncture, n) in enumerate(index):
        if puncture == "zero" and n == 1:
            matrix[sphere, col] = TWO_PI_I
    return matrix


def _finite_complex_matmul(left: np.ndarray, right: np.ndarray, name: str) -> np.ndarray:
    """Multiply complex arrays while checking the result, not stale BLAS flags."""

    # Some BLAS complex kernels leave floating-point status flags set even when
    # the returned product is finite.  NumPy then emits divide/overflow warnings
    # from matmul.  The explicit finiteness check is the reliable certificate.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        product = left @ right
    if not np.all(np.isfinite(product)):
        raise np.linalg.LinAlgError(f"{name} produced a non-finite matrix product")
    return product


def solve_constrained_collocation(matrix: np.ndarray, periods: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Minimize seam residuals with exact period constraints, after column scaling."""
    matrix = np.asarray(matrix, dtype=complex)
    periods = np.asarray(periods, dtype=complex)
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(periods)):
        raise ValueError("collocation matrix and period constraints must be finite")

    with np.errstate(over="ignore", invalid="ignore"):
        column_norms = np.maximum(
            np.linalg.norm(matrix, axis=0),
            np.linalg.norm(periods, axis=0),
        )
    column_norms = np.where(column_norms > 0.0, column_norms, 1.0)
    column_scale = 1.0 / column_norms

    scaled_matrix = matrix * column_scale[None, :]
    scaled_periods = periods * column_scale[None, :]

    _, s_periods, vh_periods = np.linalg.svd(scaled_periods, full_matrices=True)
    rank = int(np.sum(s_periods > 1e-12 * s_periods[0]))
    if rank < periods.shape[0]:
        raise np.linalg.LinAlgError("period constraints are numerically rank deficient")
    constraint_nullspace = vh_periods.conj().T[:, rank:]

    particular = np.linalg.lstsq(
        scaled_periods,
        np.eye(periods.shape[0], dtype=complex),
        rcond=None,
    )[0]
    lhs = _finite_complex_matmul(
        scaled_matrix,
        constraint_nullspace,
        "scaled collocation null space",
    )
    rhs = -_finite_complex_matmul(
        scaled_matrix,
        particular,
        "collocation particular solution",
    )
    correction = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    null_correction = _finite_complex_matmul(
        constraint_nullspace,
        correction,
        "collocation null-space correction",
    )
    scaled_forms = particular + null_correction
    forms = column_scale[:, None] * scaled_forms
    if not np.all(np.isfinite(forms)):
        raise np.linalg.LinAlgError("collocation coefficients are not finite")
    return forms, s_periods


def eval_glasses_form(
    coeffs: np.ndarray, index: list[tuple[int, str, int]], sphere: int, z: complex
) -> complex:
    total = 0.0j
    for coeff, (basis_sphere, puncture, n) in zip(coeffs, index):
        if basis_sphere == sphere:
            total += coeff * glasses_basis_scalar(puncture, n, z)
    return total


def integrate_glasses_log_spiral(
    coeffs: np.ndarray,
    index: list[tuple[int, str, int]],
    sphere: int,
    z_outer: complex,
    q: complex,
    order: int = 800,
) -> complex:
    log_q = cmath.log(q)
    z_inner = z_outer * q

    def one_pole_integral() -> complex:
        nodes, weights = np.polynomial.legendre.leggauss(order)
        total = 0.0j
        for node, weight in zip(nodes, weights):
            t = 0.5 * (node + 1.0)
            z = z_outer * cmath.exp(t * log_q)
            total += weight * z * log_q / (z - 1.0) * 0.5
        return total

    terms: list[complex] = []
    cached_one_pole: complex | None = None
    for coeff, (basis_sphere, puncture, n) in zip(coeffs, index):
        if basis_sphere != sphere or coeff == 0:
            continue
        if puncture == "zero":
            basis_integral = (
                log_q
                if n == 1
                else (z_inner ** (1 - n) - z_outer ** (1 - n)) / (1 - n)
            )
        elif puncture == "one":
            if n == 1:
                if cached_one_pole is None:
                    cached_one_pole = one_pole_integral()
                basis_integral = cached_one_pole
            else:
                basis_integral = (
                    (z_inner - 1.0) ** (1 - n) - (z_outer - 1.0) ** (1 - n)
                ) / (1 - n)
        elif puncture == "infty":
            basis_integral = -(z_inner ** (n - 1) - z_outer ** (n - 1)) / (n - 1)
        else:  # pragma: no cover - the basis index is validated upstream
            raise ValueError(f"unknown puncture {puncture!r}")
        terms.append(complex(coeff) * basis_integral)

    return complex(
        math.fsum(term.real for term in terms),
        math.fsum(term.imag for term in terms),
    )


def glasses_annular_b_periods(
    forms: np.ndarray,
    index: list[tuple[int, str, int]],
    q1: complex,
    q2: complex,
    r1: float,
    r2: float,
    order: int = 800,
) -> np.ndarray:
    omega = np.zeros((2, 2), dtype=complex)
    z1_outer = (r1 / abs(q1)) * cmath.exp(1.3j)
    z2_outer = (r2 / abs(q2)) * cmath.exp(1.7j)
    for form_idx in range(2):
        coeffs = forms[:, form_idx]
        omega[form_idx, 0] = integrate_glasses_log_spiral(coeffs, index, 0, z1_outer, q1, order=order)
        omega[form_idx, 1] = integrate_glasses_log_spiral(coeffs, index, 1, z2_outer, q2, order=order)
    return omega


def solve_glasses_collocation(
    q1: complex,
    q2: complex,
    q3: complex,
    basis_order: int = 100,
    samples_per_seam: int = 512,
    schottky_word_len: int = 8,
    radii: tuple[float, float, float] | None = None,
) -> GlassesCollocationResult:
    if radii is None:
        radii = (math.sqrt(abs(q1)), math.sqrt(abs(q2)), math.sqrt(abs(q3)))
    validate_glasses_collocation_inputs(q1, q2, q3, basis_order, samples_per_seam, radii)
    matrix, index = glasses_seam_matrix(q1, q2, q3, basis_order, samples_per_seam, *radii)
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    periods = glasses_period_constraint_matrix(index)
    forms, constraint_singular_values = solve_constrained_collocation(matrix, periods)
    a_matrix = _finite_complex_matmul(periods, forms, "glasses A-period matrix")
    omega = glasses_annular_b_periods(forms, index, q1, q2, radii[0], radii[1])
    schottky = schottky_glasses_period_matrix(q1, q2, q3, max_word_len=schottky_word_len)
    absolute_error = np.abs(omega - schottky)
    relative_error = absolute_error / np.maximum(np.abs(schottky), 1e-300)
    validation_matrix, _ = glasses_seam_matrix(
        q1,
        q2,
        q3,
        basis_order,
        max(int(samples_per_seam) * 2 + 1, int(samples_per_seam) + 17),
        *radii,
    )
    return GlassesCollocationResult(
        q1=q1,
        q2=q2,
        q3=q3,
        basis_order=basis_order,
        samples_per_seam=samples_per_seam,
        schottky_word_len=schottky_word_len,
        singular_values=singular_values,
        constraint_singular_values=constraint_singular_values,
        a_period_matrix=a_matrix,
        omega_b_annular=omega,
        schottky_omega=schottky,
        max_seam_residual=float(
            max(
                np.max(np.abs(_finite_complex_matmul(matrix, forms, "glasses seam residual"))),
                np.max(
                    np.abs(
                        _finite_complex_matmul(
                            validation_matrix,
                            forms,
                            "glasses validation residual",
                        )
                    )
                ),
            )
        ),
        schottky_error_absolute=float(np.max(absolute_error)),
        schottky_error_relative=float(np.max(relative_error)),
        omega_symmetry_error=float(np.max(np.abs(omega - omega.T))),
    )

def theta_seam_matrix(
    q_zero: complex,
    q_one: complex,
    q_infty: complex,
    basis_order: int,
    samples_per_seam: int,
    r_zero: float,
    r_one: float,
    r_infty: float,
) -> tuple[np.ndarray, list[tuple[int, str, int]]]:
    index = glasses_basis_index(basis_order)
    rows = []

    seam_data = (
        ("zero", q_zero, r_zero),
        ("one", q_one, r_one),
        ("infty", q_infty, r_infty),
    )
    for puncture, q, radius in seam_data:
        for k in range(samples_per_seam):
            theta = 2.0 * math.pi * k / samples_per_seam
            local = radius * cmath.exp(1j * theta)
            if puncture == "zero":
                z_left = local
                z_right = q / local
                dz_right_dz_left = -q / (z_left * z_left)
            elif puncture == "one":
                z_left = 1.0 + local
                z_right = 1.0 + q / local
                dz_right_dz_left = -q / ((z_left - 1.0) * (z_left - 1.0))
            elif puncture == "infty":
                z_left = 1.0 / local
                z_right = local / q
                dz_right_dz_left = -1.0 / (q * z_left * z_left)
            else:  # pragma: no cover - seam_data is fixed above
                raise ValueError(f"unknown puncture {puncture!r}")
            rows.append(
                glasses_eval_row(index, 0, z_left)
                - dz_right_dz_left * glasses_eval_row(index, 1, z_right)
            )

    return np.asarray(rows, dtype=complex), index


def theta_period_constraint_matrix(index: list[tuple[int, str, int]]) -> np.ndarray:
    """Use loops around the zero and one seams on sphere 0 as A-cycles."""
    matrix = np.zeros((2, len(index)), dtype=complex)
    for col, (sphere, puncture, n) in enumerate(index):
        if sphere == 0 and puncture == "zero" and n == 1:
            matrix[0, col] = TWO_PI_I
        if sphere == 0 and puncture == "one" and n == 1:
            matrix[1, col] = TWO_PI_I
    return matrix


def theta_boundary_pair(
    puncture: str,
    q: complex,
    radius: float,
    phase: float,
) -> tuple[complex, complex]:
    local = radius * cmath.exp(1j * phase)
    if puncture == "zero":
        return local, q / local
    if puncture == "one":
        return 1.0 + local, 1.0 + q / local
    if puncture == "infty":
        return 1.0 / local, local / q
    raise ValueError(f"unknown puncture {puncture!r}")


def integrate_theta_segment(
    coeffs: np.ndarray,
    index: list[tuple[int, str, int]],
    sphere: int,
    z0: complex,
    z1: complex,
    order: int = 500,
) -> complex:
    """Integrate one collocation form without evaluating an ill-conditioned sum.

    For strongly pinched theta seams, a balanced seam radius can put the
    infinity endpoint at ``|z| >> 1``.  Pointwise evaluation then combines
    very large Laurent monomials with very small coefficients before the
    cancellation that makes the differential regular.  Integrating every
    Laurent basis element analytically postpones that cancellation until the
    final compensated sum.  Simple-pole terms use the continuous logarithm
    along the straight segment; all higher Laurent modes use their elementary
    primitives.
    """

    def simple_pole_integral(pole: complex) -> complex:
        start = z0 - pole
        stop = z1 - pole
        if start == 0 or stop == 0:
            raise ValueError("theta B-period endpoint landed on a simple pole")
        # Along a straight segment that does not cross the pole, the argument
        # changes by the principal argument of stop/start (strictly less than
        # pi in magnitude).  This is the continuous logarithm on that path.
        closest_parameter = float(
            np.clip(-((start.conjugate() * (stop - start)).real) / abs(stop - start) ** 2, 0.0, 1.0)
        ) if stop != start else 0.0
        if abs(start + closest_parameter * (stop - start)) < 1.0e-14:
            raise ValueError("theta B-period segment crosses a simple pole")
        return cmath.log(stop / start)

    terms: list[complex] = []
    for coeff, (basis_sphere, puncture, n) in zip(coeffs, index):
        if basis_sphere != sphere or coeff == 0:
            continue
        if puncture == "zero":
            basis_integral = (
                simple_pole_integral(0.0j)
                if n == 1
                else (z1 ** (1 - n) - z0 ** (1 - n)) / (1 - n)
            )
        elif puncture == "one":
            basis_integral = (
                simple_pole_integral(1.0 + 0.0j)
                if n == 1
                else ((z1 - 1.0) ** (1 - n) - (z0 - 1.0) ** (1 - n)) / (1 - n)
            )
        elif puncture == "infty":
            # The infinity basis starts at n=2 and is -z^(n-2).
            basis_integral = -(z1 ** (n - 1) - z0 ** (n - 1)) / (n - 1)
        else:  # pragma: no cover - the basis index is validated upstream
            raise ValueError(f"unknown puncture {puncture!r}")
        terms.append(complex(coeff) * basis_integral)

    return complex(
        math.fsum(term.real for term in terms),
        math.fsum(term.imag for term in terms),
    )


def theta_b_periods(
    forms: np.ndarray,
    index: list[tuple[int, str, int]],
    q_zero: complex,
    q_one: complex,
    q_infty: complex,
    radii: tuple[float, float, float],
    *,
    phases: tuple[float, float, float] = (0.37, 2.11, 1.23),
    integration_order: int = 500,
) -> np.ndarray:
    """Return B-periods for the theta graph with the infinity seam as reference.

    The two B-cycles go from the infinity seam to the zero/one seam on sphere 0,
    cross that seam, and return to the infinity seam on sphere 1.  This is the
    direct analogue of choosing two independent cycles in the theta graph.
    """
    r_zero, r_one, r_infty = radii
    point_zero = theta_boundary_pair("zero", q_zero, r_zero, phases[0])
    point_one = theta_boundary_pair("one", q_one, r_one, phases[1])
    point_infty = theta_boundary_pair("infty", q_infty, r_infty, phases[2])

    omega = np.zeros((2, 2), dtype=complex)
    targets = (point_zero, point_one)
    for form_idx in range(2):
        coeffs = forms[:, form_idx]
        for cycle_idx, target in enumerate(targets):
            omega[form_idx, cycle_idx] = (
                integrate_theta_segment(
                    coeffs,
                    index,
                    0,
                    point_infty[0],
                    target[0],
                    order=integration_order,
                )
                + integrate_theta_segment(
                    coeffs,
                    index,
                    1,
                    target[1],
                    point_infty[1],
                    order=integration_order,
                )
            )
    return omega


def solve_theta_collocation(
    q_zero: complex,
    q_one: complex,
    q_infty: complex,
    basis_order: int = 24,
    samples_per_seam: int = 192,
    radii: tuple[float, float, float] | None = None,
    integration_order: int = 500,
) -> ThetaCollocationResult:
    q_values = (complex(q_zero), complex(q_one), complex(q_infty))
    if any(not (0.0 < abs(q) < 1.0) for q in q_values):
        raise ValueError("theta collocation expects 0 < |q_e| < 1 for all three edges")
    if radii is None:
        radii = tuple(math.sqrt(abs(q)) for q in q_values)
    if any(not (abs(q) < radius < 1.0) for q, radius in zip(q_values, radii)):
        raise ValueError("choose radii with |q_e| < r_e < 1")

    matrix, index = theta_seam_matrix(
        q_values[0],
        q_values[1],
        q_values[2],
        basis_order,
        samples_per_seam,
        radii[0],
        radii[1],
        radii[2],
    )
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    periods = theta_period_constraint_matrix(index)
    forms, constraint_singular_values = solve_constrained_collocation(matrix, periods)
    a_matrix = _finite_complex_matmul(periods, forms, "theta A-period matrix")
    omega_raw = theta_b_periods(
        forms,
        index,
        q_values[0],
        q_values[1],
        q_values[2],
        radii,
        integration_order=integration_order,
    )
    upper = complex(omega_raw[0, 1])
    lower = complex(omega_raw[1, 0])
    lower_branch = int(round((upper - lower).real))
    lower_aligned = lower + lower_branch
    symmetry_error = abs(upper - lower_aligned)
    omega = np.asarray(omega_raw, dtype=np.complex128).copy()
    omega[0, 1] = 0.5 * (upper + lower_aligned)
    omega[1, 0] = omega[0, 1]
    return ThetaCollocationResult(
        q_zero=q_values[0],
        q_one=q_values[1],
        q_infty=q_values[2],
        basis_order=int(basis_order),
        samples_per_seam=int(samples_per_seam),
        radii=tuple(float(radius) for radius in radii),
        singular_values=singular_values,
        constraint_singular_values=constraint_singular_values,
        a_period_matrix=a_matrix,
        omega=omega,
        max_seam_residual=float(
            np.max(np.abs(_finite_complex_matmul(matrix, forms, "theta seam residual")))
        ),
        omega_symmetry_error=float(symmetry_error),
    )


def glasses_collocation_period_matrix(
    q1: complex,
    q2: complex,
    q3: complex,
    basis_order: int = 18,
    samples_per_seam: int = 96,
    radii: tuple[float, float, float] | None = None,
) -> tuple[np.ndarray, float, float]:
    """Fast glasses collocation period matrix for bulk data generation.

    Returns ``(Omega, max_seam_residual, symmetry_error)`` and does not compute a
    Schottky reference.
    """
    if radii is None:
        radii = (math.sqrt(abs(q1)), math.sqrt(abs(q2)), math.sqrt(abs(q3)))
    validate_glasses_collocation_inputs(q1, q2, q3, basis_order, samples_per_seam, radii)
    matrix, index = glasses_seam_matrix(q1, q2, q3, basis_order, samples_per_seam, *radii)
    periods = glasses_period_constraint_matrix(index)
    forms, _ = solve_constrained_collocation(matrix, periods)
    omega = glasses_annular_b_periods(forms, index, q1, q2, radii[0], radii[1], order=160)
    validation_matrix, _ = glasses_seam_matrix(
        q1,
        q2,
        q3,
        basis_order,
        max(int(samples_per_seam) * 2 + 1, int(samples_per_seam) + 17),
        *radii,
    )
    return (
        omega,
        float(
            np.max(
                np.abs(
                    _finite_complex_matmul(
                        validation_matrix,
                        forms,
                        "glasses validation residual",
                    )
                )
            )
        ),
        float(np.max(np.abs(omega - omega.T))),
    )






def _stored_graph_to_ribbon_graph(graph_data: dict):
    edges_labeled = tuple(graph_data["edges"])
    boundary = tuple(graph_data["boundary"])
    edges = [(a, b) for _, a, b in edges_labeled]
    verts = sorted({v for _, a, b in edges_labeled for v in (a, b)})

    succ = {v: {} for v in verts}
    for i, (_, to_v, e_label) in enumerate(boundary):
        next_from, _, next_e = boundary[(i + 1) % len(boundary)]
        if next_from != to_v:
            raise ValueError(
                f"Boundary is not contiguous at segment {i + 1}: "
                f"{boundary[i]} followed by {boundary[(i + 1) % len(boundary)]}"
            )
        succ[to_v][e_label - 1] = next_e - 1

    rotation = {}
    for v in verts:
        incident = [idx for idx, (a, b) in enumerate(edges) if a == v or b == v]
        start = incident[0]
        order = [start]
        cur = start
        while True:
            nxt = succ[v][cur]
            if nxt == start:
                break
            order.append(nxt)
            cur = nxt
        rotation[v] = order

    return edges, verts, rotation








def _reduced_tau_distance(tau: complex, target: complex) -> tuple[float, complex, complex]:
    reduced_tau = reduce_tau(tau).tau_reduced
    reduced_target = reduce_tau(target).tau_reduced
    real_delta = reduced_tau.real - reduced_target.real
    real_delta = real_delta - round(real_delta)
    imag_delta = reduced_tau.imag - reduced_target.imag
    return float(max(abs(real_delta), abs(imag_delta))), reduced_tau, reduced_target


def _tau_chart_distance(tau: complex, target: complex) -> float:
    real_delta = complex(tau).real - complex(target).real
    real_delta = real_delta - round(real_delta)
    imag_delta = complex(tau).imag - complex(target).imag
    return float(max(abs(real_delta), abs(imag_delta)))






def load_genus1_lookup_table(path: str | Path) -> Genus1LookupTable:
    data = np.load(Path(path), allow_pickle=False)
    return Genus1LookupTable(
        total_edge_length=int(data["total_edge_length"].item()),
        minimum_edge_length=int(data["minimum_edge_length"].item()),
        edge_lengths=np.asarray(data["edge_lengths"], dtype=np.int64),
        tau=np.asarray(data["tau"], dtype=np.complex128),
    )




def _genus1_transfer_neighbors(
    lengths: Sequence[int],
    *,
    minimum_edge_length: int,
    step: int,
) -> list[tuple[int, int, int]]:
    base = tuple(int(x) for x in lengths)
    neighbors: list[tuple[int, int, int]] = []
    for source in range(3):
        if base[source] - int(step) < int(minimum_edge_length):
            continue
        for target in range(3):
            if source == target:
                continue
            trial = list(base)
            trial[source] -= int(step)
            trial[target] += int(step)
            neighbors.append(tuple(int(x) for x in trial))  # type: ignore[arg-type]
    return list(dict.fromkeys(neighbors))




def symmetrized_period_matrix(omega: np.ndarray) -> np.ndarray:
    return 0.5 * (np.asarray(omega, dtype=complex) + np.asarray(omega, dtype=complex).T)


def period_matrix_is_riemann(omega: np.ndarray, min_im_eigenvalue: float = 1e-10) -> bool:
    omega = np.asarray(omega, dtype=complex)
    entries = [
        omega[0, 0].real,
        omega[0, 0].imag,
        omega[0, 1].real,
        omega[0, 1].imag,
        omega[1, 0].real,
        omega[1, 0].imag,
        omega[1, 1].real,
        omega[1, 1].imag,
    ]
    if not all(math.isfinite(float(value)) for value in entries):
        return False
    return bool(np.min(np.linalg.eigvalsh(symmetrized_period_matrix(omega).imag)) > min_im_eigenvalue)


def plumbing_genus2_period_matrix(
    channel: str,
    q1: complex,
    q2: complex,
    q3: complex,
    algorithm: str = "auto",
    schottky_word_len: int = 8,
    schottky_b_order: int = 600,
    collocation_basis_order: int = 100,
    collocation_samples: int = 512,
    hybrid_tolerance: float = 1.0e-6,
    hybrid_agreement_tolerance: float = 1.0e-6,
) -> tuple[np.ndarray, str]:
    """Return the plumbing period matrix and the numerical route used.

    ``auto`` uses the adaptive hybrid policy: normalized holomorphic one-forms
    in the bulk, Schottky in long cusps, and an explicit two-method agreement
    check in their overlap.  Explicit ``collocation`` and ``schottky`` retain
    their legacy fixed-cutoff meanings.
    """
    channel = channel.lower()
    algorithm = algorithm.lower()
    if algorithm == "auto" and channel in {"theta", "glasses", "sunglasses"}:
        # Import lazily because the hybrid policy itself builds on the primitive
        # routines in this module.
        try:
            from genus2_hybrid_period_map import (
                HybridPeriodMapConfig,
                hybrid_period_matrix,
                is_schottky_algorithm,
            )
        except ImportError:  # pragma: no cover - package-style execution
            from genus2_hybrid_period_map import (
                HybridPeriodMapConfig,
                hybrid_period_matrix,
                is_schottky_algorithm,
            )

        topology = "theta" if channel == "theta" else "glasses"
        hybrid = hybrid_period_matrix(
            topology,
            (q1, q2, q3),
            config=HybridPeriodMapConfig(
                tolerance=float(hybrid_tolerance),
                agreement_tolerance=float(hybrid_agreement_tolerance),
            ),
        )
        legacy_name = (
            "schottky_series"
            if is_schottky_algorithm(hybrid.algorithm)
            else "holomorphic_form_collocation"
        )
        return symmetrized_period_matrix(hybrid.omega), legacy_name

    if channel in {"glasses", "sunglasses"}:
        if algorithm in {"schottky", "schottky_series"}:
            omega = schottky_glasses_period_matrix(q1, q2, q3, schottky_word_len, b_order=schottky_b_order)
            return symmetrized_period_matrix(omega), "schottky_series"
        if algorithm in {"collocation", "boundary_collocation", "holomorphic_form_collocation"}:
            omega, seam_residual, symmetry_error = glasses_collocation_period_matrix(
                q1,
                q2,
                q3,
                basis_order=collocation_basis_order,
                samples_per_seam=collocation_samples,
            )
            if seam_residual > 1e-6 or symmetry_error > 1e-5:
                raise RuntimeError(
                    "glasses collocation target failed diagnostics: "
                    f"seam_residual={seam_residual:.3e}, symmetry_error={symmetry_error:.3e}"
                )
            return symmetrized_period_matrix(omega), "holomorphic_form_collocation"
        raise ValueError(f"unknown plumbing algorithm {algorithm!r}")

    if channel == "theta":
        if algorithm in {"schottky", "schottky_series"}:
            omega = schottky_theta_period_matrix_cross_ratio(
                q1,
                q2,
                q3,
                max_word_len=schottky_word_len,
            )
            return symmetrized_period_matrix(omega), "schottky_series"
        if algorithm in {"collocation", "boundary_collocation", "holomorphic_form_collocation"}:
            result = solve_theta_collocation(
                q1,
                q2,
                q3,
                basis_order=collocation_basis_order,
                samples_per_seam=collocation_samples,
            )
            if result.max_seam_residual > 1e-6 or result.omega_symmetry_error > 1e-5:
                raise RuntimeError(
                    "theta collocation target failed diagnostics: "
                    f"seam_residual={result.max_seam_residual:.3e}, "
                    f"symmetry_error={result.omega_symmetry_error:.3e}"
                )
            return symmetrized_period_matrix(result.omega), "holomorphic_form_collocation"
        raise ValueError(f"unknown plumbing algorithm {algorithm!r}")

    if channel == "sunrise":
        if algorithm == "auto":
            algorithm = "schottky"
        if algorithm not in {"schottky", "schottky_series"}:
            raise ValueError("the current code only has a Schottky sunrise target solver")
        omega = schottky_sunrise_period_matrix(q1, q2, q3, schottky_word_len, b_order=schottky_b_order)
        return symmetrized_period_matrix(omega), "schottky_series"

    raise ValueError("channel must be 'glasses', 'sunglasses', 'theta', or 'sunrise'")


def _period_residual_vector(omega: np.ndarray, target: np.ndarray) -> np.ndarray:
    omega = symmetrized_period_matrix(omega)
    target = symmetrized_period_matrix(target)
    real_delta = omega.real - target.real
    real_delta = real_delta - np.rint(real_delta)
    imag_delta = omega.imag - target.imag
    return np.asarray(
        [
            real_delta[0, 0],
            real_delta[0, 1],
            real_delta[1, 1],
            imag_delta[0, 0],
            imag_delta[0, 1],
            imag_delta[1, 1],
        ],
        dtype=float,
    )


def period_matrix_residual(omega: np.ndarray, target: np.ndarray) -> float:
    """Sup-norm residual in Siegel coordinates, with real parts wrapped mod 1."""
    return float(np.max(np.abs(_period_residual_vector(omega, target))))


def edge_length_balance_penalty(edge_lengths: Sequence[int]) -> float:
    logs = np.log(np.asarray(edge_lengths, dtype=float))
    return float(np.sqrt(np.mean((logs - np.mean(logs)) ** 2)))


def _integer_composition_from_weights(weights: np.ndarray, total: int, minimum: int) -> tuple[int, ...]:
    n_edges = int(len(weights))
    if total < minimum * n_edges:
        raise ValueError("total_edge_length is too small for the requested minimum edge length")
    free_total = int(total - minimum * n_edges)
    weights = np.asarray(weights, dtype=float)
    weights = np.maximum(weights, 0.0)
    if not np.any(weights):
        weights = np.ones(n_edges, dtype=float)
    weights = weights / np.sum(weights)
    raw = free_total * weights
    floors = np.floor(raw).astype(int)
    remainder = free_total - int(np.sum(floors))
    order = np.argsort(-(raw - floors))
    floors[order[:remainder]] += 1
    return tuple(int(x + minimum) for x in floors)


def scale_edge_lengths_to_total(
    edge_lengths: Sequence[int],
    total_edge_length: int,
    minimum_edge_length: int = 1,
) -> tuple[int, ...]:
    """Scale an integer edge-length ratio to a new fixed total length."""
    edge_lengths = tuple(int(x) for x in edge_lengths)
    if not edge_lengths:
        raise ValueError("need at least one edge length")
    if any(length <= 0 for length in edge_lengths):
        raise ValueError("all source edge lengths must be positive")
    return _integer_composition_from_weights(
        np.asarray(edge_lengths, dtype=float),
        int(total_edge_length),
        int(minimum_edge_length),
    )


def _initial_length_candidates(
    n_edges: int,
    total: int,
    minimum: int,
    rng: np.random.Generator,
    random_candidates: int,
) -> list[tuple[int, ...]]:
    candidates = [_integer_composition_from_weights(np.ones(n_edges), total, minimum)]
    for concentration in (12.0, 4.0, 1.2, 0.45):
        weights = rng.dirichlet(np.full(n_edges, concentration))
        candidates.append(_integer_composition_from_weights(weights, total, minimum))
    for _ in range(int(random_candidates)):
        concentration = float(10 ** rng.uniform(-0.6, 1.0))
        weights = rng.dirichlet(np.full(n_edges, concentration))
        candidates.append(_integer_composition_from_weights(weights, total, minimum))
    return list(dict.fromkeys(candidates))








def format_complex(z: complex) -> str:
    return f"{z.real:+.12e}{z.imag:+.12e}j"


def print_matrix(name: str, matrix: np.ndarray) -> None:
    print(f"{name}:")
    for row in matrix:
        print("  " + "  ".join(format_complex(complex(z)) for z in row))




