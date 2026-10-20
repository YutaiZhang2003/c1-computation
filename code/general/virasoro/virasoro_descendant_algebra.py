#!/usr/bin/env python3
"""Low-level Virasoro descendant algebra in a PBW basis."""

from __future__ import annotations

from functools import lru_cache


Descendant = tuple[int, ...]
State = dict[Descendant, complex]


def _state_items(state: State) -> tuple[tuple[Descendant, complex], ...]:
    return tuple(sorted((desc, complex(coeff)) for desc, coeff in state.items() if coeff != 0))


def _combine_states(*states: tuple[complex, State]) -> State:
    out: State = {}
    for scale, state in states:
        for desc, coeff in state.items():
            value = out.get(desc, 0.0j) + scale * coeff
            if value != 0:
                out[desc] = value
            elif desc in out:
                del out[desc]
    return out


@lru_cache(maxsize=None)
def _normal_order_negative_word_items(word: Descendant) -> tuple[tuple[Descendant, complex], ...]:
    """Normal-order negative Virasoro modes into descending PBW order."""

    for idx in range(len(word) - 1):
        left = word[idx]
        right = word[idx + 1]
        if left < right:
            swapped = word[:idx] + (right, left) + word[idx + 2 :]
            commutator = word[:idx] + (left + right,) + word[idx + 2 :]
            return _state_items(
                _combine_states(
                    (1.0, dict(_normal_order_negative_word_items(swapped))),
                    (right - left, dict(_normal_order_negative_word_items(commutator))),
                )
            )
    return ((tuple(word), 1.0 + 0.0j),)


def normal_order_negative_word(word: Descendant) -> State:
    return dict(_normal_order_negative_word_items(tuple(word)))


def project_vacuum_state(state: State) -> State:
    """Project to the vacuum module quotient ``L_-1 |0> = 0``."""

    return {desc: coeff for desc, coeff in state.items() if not desc or desc[-1] != 1}


def prepend_negative_mode(mode: int, state: State, *, vacuum: bool) -> State:
    out: State = {}
    for desc, coeff in state.items():
        ordered = normal_order_negative_word((int(mode),) + desc)
        out = _combine_states((1.0, out), (coeff, ordered))
    return project_vacuum_state(out) if vacuum else out


@lru_cache(maxsize=None)
def _act_virasoro_mode_items(
    mode: int,
    desc: Descendant,
    h: complex,
    c: complex,
    vacuum: bool,
) -> tuple[tuple[Descendant, complex], ...]:
    """Act with ``L_mode`` on ``L_-desc |h>``."""

    mode = int(mode)
    desc = tuple(desc)
    if mode < 0:
        state = normal_order_negative_word((-mode,) + desc)
        return _state_items(project_vacuum_state(state) if vacuum else state)
    if mode == 0:
        return ((desc, complex(h) + sum(desc)),)
    if not desc:
        return ()

    first = desc[0]
    rest = desc[1:]

    # L_m L_-p rest = L_-p L_m rest + [L_m,L_-p] rest.
    moved = prepend_negative_mode(
        first,
        dict(_act_virasoro_mode_items(mode, rest, h, c, vacuum)),
        vacuum=vacuum,
    )
    pieces: list[tuple[complex, State]] = [(1.0, moved)]

    commutator_mode = mode - first
    commutator_scale = mode + first
    if commutator_scale != 0:
        if commutator_mode < 0:
            commutator_state = normal_order_negative_word((-commutator_mode,) + rest)
            if vacuum:
                commutator_state = project_vacuum_state(commutator_state)
        elif commutator_mode == 0:
            commutator_state = {rest: complex(h) + sum(rest)}
            if vacuum:
                commutator_state = project_vacuum_state(commutator_state)
        else:
            commutator_state = dict(_act_virasoro_mode_items(commutator_mode, rest, h, c, vacuum))
        pieces.append((commutator_scale, commutator_state))

    if mode == first:
        central = complex(c) * mode * (mode * mode - 1) / 12.0
        central_state = {rest: 1.0 + 0.0j}
        if vacuum:
            central_state = project_vacuum_state(central_state)
        pieces.append((central, central_state))

    return _state_items(_combine_states(*pieces))


def act_virasoro_mode(
    mode: int,
    state: State,
    *,
    h: complex = 0.0,
    c: complex = 1.0,
    vacuum: bool = True,
) -> State:
    """Act with one Virasoro mode on a linear combination of PBW states."""

    out: State = {}
    for desc, coeff in state.items():
        acted = dict(
            _act_virasoro_mode_items(
                int(mode),
                tuple(desc),
                complex(h),
                complex(c),
                bool(vacuum),
            )
        )
        out = _combine_states((1.0, out), (coeff, acted))
    return project_vacuum_state(out) if vacuum else out


def integer_partitions(
    total: int,
    *,
    max_part: int | None = None,
    min_part: int = 1,
) -> list[Descendant]:
    """Return descending integer partitions with a lower part bound."""

    if total < 0:
        return []
    if total == 0:
        return [()]
    if max_part is None:
        max_part = total
    out: list[Descendant] = []
    for part in range(min(int(max_part), int(total)), int(min_part) - 1, -1):
        for tail in integer_partitions(total - part, max_part=part, min_part=min_part):
            out.append((part,) + tail)
    return out


def vacuum_descendant_basis(level: int) -> list[Descendant]:
    """PBW basis of the vacuum Verma quotient at a fixed level."""

    return integer_partitions(int(level), min_part=2)


def descendant_inner_product(
    bra_desc: Descendant,
    ket_desc: Descendant,
    *,
    h: complex = 0.0,
    c: complex = 1.0,
    vacuum: bool = True,
) -> complex:
    """Return the BPZ Gram pairing of two PBW descendants."""

    state: State = {tuple(ket_desc): 1.0 + 0.0j}
    if vacuum:
        state = project_vacuum_state(state)
    # For bra_desc=(a1,a2,...), BPZ gives <h|...L_a2 L_a1.
    # The rightmost operator therefore acts in the stored tuple order.
    for mode in tuple(bra_desc):
        state = act_virasoro_mode(mode, state, h=h, c=c, vacuum=vacuum)
        if not state:
            return 0.0j
    return complex(state.get((), 0.0j))


def vacuum_gram_matrix(
    level: int,
    c: complex,
) -> tuple[list[Descendant], list[list[complex]]]:
    basis = vacuum_descendant_basis(int(level))
    matrix = [
        [descendant_inner_product(left, right, c=c, vacuum=True) for right in basis]
        for left in basis
    ]
    return basis, matrix


def descendant_level(desc: Descendant) -> int:
    return int(sum(desc))
