#!/usr/bin/env python3
"""Torus one-point blocks with Virasoro-descendant external insertions.

This module evaluates the plane-frame torus one-point sewing sum

    F_X(q)=sum_n q^n sum_{|A|=|B|=n} G_h^{AB}
           rho(L_-A h, X_d, L_-B h),

where X_d is a descendant of the external primary of weight d.  It is meant as
a low-level Ward-identity baseline, not as a replacement for the much faster
Zamolodchikov recursion used for primary insertions.
"""

from __future__ import annotations

from c1_paths import PACKAGE_ROOT, data_root, package_path, module_path

import argparse
import math
from functools import lru_cache

import numpy as np

try:
    from virasoro_blocks import TorusOnePointVirasoroBlock
except ImportError:  # pragma: no cover - supports package-style execution
    from virasoro_blocks import TorusOnePointVirasoroBlock


State = tuple[int, ...]


def parse_complex(value: str) -> complex:
    return complex(value.replace("i", "j"))


def parse_state(value: str) -> State:
    if value.strip() in {"", "primary", "0"}:
        return ()
    parts = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if any(part <= 0 for part in parts):
        raise ValueError("descendant state entries must be positive Virasoro mode numbers")
    return parts


def format_complex(value: complex) -> str:
    return f"{value.real:+.12e}{value.imag:+.12e}j"


def state_level(state: State) -> int:
    return int(sum(state))


def integer_partitions(total: int, max_part: int | None = None) -> tuple[State, ...]:
    if total < 0:
        raise ValueError("partition total must be non-negative")
    if total == 0:
        return ((),)
    if max_part is None or max_part > total:
        max_part = total
    out: list[State] = []
    for first in range(max_part, 0, -1):
        for rest in integer_partitions(total - first, min(first, total - first) if total > first else 0):
            out.append((first,) + rest)
    return tuple(out)


def _add_scaled(target: dict[State, complex], state: State, coeff: complex) -> None:
    if abs(coeff) == 0:
        return
    target[state] = target.get(state, 0.0 + 0.0j) + coeff


def _merge_scaled(target: dict[State, complex], source: dict[State, complex], scale: complex) -> None:
    if abs(scale) == 0:
        return
    for state, coeff in source.items():
        _add_scaled(target, state, scale * coeff)


@lru_cache(maxsize=None)
def apply_virasoro_mode(mode: int, state: State, h: complex, c: complex) -> tuple[tuple[State, complex], ...]:
    """Apply L_mode to a descendant state L_-state |h>."""
    mode = int(mode)
    h = complex(h)
    c = complex(c)
    if mode < 0:
        return (((-mode,) + state, 1.0 + 0.0j),)
    if mode == 0:
        return ((state, h + state_level(state)),)
    if not state:
        return ()

    first = state[0]
    rest = state[1:]
    out: dict[State, complex] = {}

    commutator_mode = mode - first
    commutator_coeff = mode + first
    if commutator_mode == 0:
        _add_scaled(out, rest, commutator_coeff * (h + state_level(rest)))
    elif commutator_mode > 0:
        _merge_scaled(out, dict(apply_virasoro_mode(commutator_mode, rest, h, c)), commutator_coeff)
    else:
        _add_scaled(out, ((-commutator_mode),) + rest, commutator_coeff)

    if mode == first:
        _add_scaled(out, rest, c * mode * (mode * mode - 1.0) / 12.0)

    for next_state, coeff in apply_virasoro_mode(mode, rest, h, c):
        _add_scaled(out, (first,) + next_state, coeff)

    return tuple(out.items())


def inner_product(bra_state: State, ket_state: State, h: complex, c: complex) -> complex:
    """Return <h| L_A L_-B |h> for ordered descendant chains."""
    states: dict[State, complex] = {ket_state: 1.0 + 0.0j}
    for mode in bra_state:
        updated: dict[State, complex] = {}
        for state, coeff in states.items():
            _merge_scaled(updated, dict(apply_virasoro_mode(mode, state, h, c)), coeff)
        states = updated
        if not states:
            return 0.0 + 0.0j
    return states.get((), 0.0 + 0.0j)


def gram_matrix(h: complex, c: complex, level: int) -> tuple[tuple[State, ...], np.ndarray]:
    basis = integer_partitions(level)
    matrix = np.zeros((len(basis), len(basis)), dtype=np.complex128)
    for row, bra_state in enumerate(basis):
        for col, ket_state in enumerate(basis):
            matrix[row, col] = inner_product(bra_state, ket_state, h, c)
    return basis, matrix


@lru_cache(maxsize=None)
def rho_primary_external(
    state_at_infinity: State,
    state_at_zero: State,
    h_infinity: complex,
    h_external: complex,
    h_zero: complex,
    c: complex,
) -> complex:
    """Return rho(L_-A h3, primary_d, L_-B h1 | 1)."""
    h_infinity = complex(h_infinity)
    h_external = complex(h_external)
    h_zero = complex(h_zero)
    c = complex(c)
    if not state_at_infinity and not state_at_zero:
        return 1.0 + 0.0j

    if state_at_zero:
        mode = state_at_zero[0]
        rest_zero = state_at_zero[1:]
        total = 0.0 + 0.0j
        for new_infinity, coeff in apply_virasoro_mode(mode, state_at_infinity, h_infinity, c):
            total += coeff * rho_primary_external(new_infinity, rest_zero, h_infinity, h_external, h_zero, c)
        z_derivative_weight = (
            h_infinity
            + state_level(state_at_infinity)
            - (h_zero + state_level(rest_zero))
            - mode * h_external
        )
        total -= z_derivative_weight * rho_primary_external(
            state_at_infinity,
            rest_zero,
            h_infinity,
            h_external,
            h_zero,
            c,
        )
        return total

    mode = state_at_infinity[0]
    rest_infinity = state_at_infinity[1:]
    total = 0.0 + 0.0j
    for new_zero, coeff in apply_virasoro_mode(mode, state_at_zero, h_zero, c):
        total += coeff * rho_primary_external(rest_infinity, new_zero, h_infinity, h_external, h_zero, c)
    z_derivative_weight = (
        h_infinity
        + state_level(rest_infinity)
        - (h_zero + state_level(state_at_zero))
        + mode * h_external
    )
    total += z_derivative_weight * rho_primary_external(
        rest_infinity,
        state_at_zero,
        h_infinity,
        h_external,
        h_zero,
        c,
    )
    return total


@lru_cache(maxsize=None)
def rho_descendant_external(
    state_at_infinity: State,
    external_state: State,
    state_at_zero: State,
    h_internal: complex,
    h_external: complex,
    c: complex,
) -> complex:
    """Return rho(L_-A h, L_-M d, L_-B h | 1)."""
    h_internal = complex(h_internal)
    h_external = complex(h_external)
    c = complex(c)
    if not external_state:
        return rho_primary_external(
            state_at_infinity,
            state_at_zero,
            h_internal,
            h_external,
            h_internal,
            c,
        )

    mode = external_state[0]
    rest_external = external_state[1:]
    if mode == 1:
        exponent = (
            h_internal
            + state_level(state_at_infinity)
            - (h_external + state_level(rest_external))
            - (h_internal + state_level(state_at_zero))
        )
        return exponent * rho_descendant_external(
            state_at_infinity,
            rest_external,
            state_at_zero,
            h_internal,
            h_external,
            c,
        )

    total = 0.0 + 0.0j
    max_m = max(0, state_level(state_at_infinity) - mode, state_level(state_at_zero) + 1)
    sign = -1.0 if mode % 2 else 1.0
    for m_value in range(max_m + 1):
        binom = math.comb(mode - 2 + m_value, mode - 2)
        for new_infinity, coeff in apply_virasoro_mode(mode + m_value, state_at_infinity, h_internal, c):
            total += binom * coeff * rho_descendant_external(
                new_infinity,
                rest_external,
                state_at_zero,
                h_internal,
                h_external,
                c,
            )
        for new_zero, coeff in apply_virasoro_mode(m_value - 1, state_at_zero, h_internal, c):
            total += sign * binom * coeff * rho_descendant_external(
                state_at_infinity,
                rest_external,
                new_zero,
                h_internal,
                h_external,
                c,
            )
    return total


def torus_one_point_descendant_coefficients(
    c: complex,
    internal_weight: complex,
    external_weight: complex,
    external_state: State,
    order: int,
) -> list[complex]:
    """Return plane-frame q-series coefficients through q^order."""
    if order < 0:
        raise ValueError("order must be non-negative")
    out: list[complex] = []
    for level in range(order + 1):
        basis, gram = gram_matrix(internal_weight, c, level)
        inverse_gram = np.linalg.inv(gram)
        coefficient = 0.0 + 0.0j
        for row, state_at_infinity in enumerate(basis):
            for col, state_at_zero in enumerate(basis):
                coefficient += inverse_gram[row, col] * rho_descendant_external(
                    state_at_infinity,
                    external_state,
                    state_at_zero,
                    internal_weight,
                    external_weight,
                    c,
                )
        out.append(coefficient)
    return out


def torus_one_point_descendant_block(
    c: complex,
    internal_weight: complex,
    external_weight: complex,
    external_state: State,
    q: complex,
    order: int,
    *,
    include_prefactor: bool = False,
) -> complex:
    """Evaluate the descendant torus one-point block through q^order."""
    q = complex(q)
    coeffs = torus_one_point_descendant_coefficients(
        c,
        internal_weight,
        external_weight,
        external_state,
        order,
    )
    value = sum(coeff * (q**level) for level, coeff in enumerate(coeffs))
    if include_prefactor:
        value *= q ** (complex(internal_weight) - complex(c) / 24.0)
    return value


def lminus_one_power_multiplier(external_weight: complex, power: int) -> complex:
    """Exact plane-frame multiplier for external L_-1^power insertions."""
    if power < 0:
        raise ValueError("power must be non-negative")
    value = 1.0 + 0.0j
    h_external = complex(external_weight)
    for idx in range(power):
        value *= -(h_external + idx)
    return value


def run() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a torus one-point block with descendant external operator.")
    parser.add_argument("--c", type=parse_complex, required=True)
    parser.add_argument("--internal-weight", type=parse_complex, required=True)
    parser.add_argument("--external-weight", type=parse_complex, required=True)
    parser.add_argument("--external-state", type=parse_state, default=())
    parser.add_argument("--q", type=parse_complex, required=True)
    parser.add_argument("--order", type=int, default=3)
    parser.add_argument("--include-prefactor", action="store_true")
    args = parser.parse_args()

    coeffs = torus_one_point_descendant_coefficients(
        args.c,
        args.internal_weight,
        args.external_weight,
        args.external_state,
        args.order,
    )
    value = torus_one_point_descendant_block(
        args.c,
        args.internal_weight,
        args.external_weight,
        args.external_state,
        args.q,
        args.order,
        include_prefactor=args.include_prefactor,
    )
    primary = TorusOnePointVirasoroBlock(args.c, args.internal_weight, args.external_weight)
    primary_value = primary.chiral_block(args.q, args.order, include_prefactor=args.include_prefactor)

    print("torus one-point descendant block")
    print(f"  c={format_complex(args.c)}")
    print(f"  internal h={format_complex(args.internal_weight)}")
    print(f"  external h={format_complex(args.external_weight)}")
    print(f"  external state={args.external_state or 'primary'}")
    print(f"  q={format_complex(args.q)}")
    print(f"  order={args.order}")
    print("  coefficients:")
    for level, coeff in enumerate(coeffs):
        print(f"    q^{level}: {format_complex(coeff)}")
    print(f"  value={format_complex(value)}")
    if not args.external_state:
        print(f"  primary recursion value={format_complex(primary_value)}")
        print(f"  direct-recursion difference={format_complex(value - primary_value)}")


if __name__ == "__main__":
    run()
