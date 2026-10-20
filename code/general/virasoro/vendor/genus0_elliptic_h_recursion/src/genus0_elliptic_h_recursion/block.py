"""Full sphere-block reconstruction from the reduced elliptic recursion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import mpmath as mp

from .geometry import AlignedNomes, invert_aligned_coordinates, theta3_from_nome
from .kac import as_mpmath
from .recursion import RecursionTable


Number = Any


@dataclass(frozen=True)
class SphereBlockEvaluation:
    """A reconstructed chiral sphere block and its aligned coordinates."""

    value: Number
    reduced_value: Number
    nomes: AlignedNomes
    order: int


def effective_plumbing_parameters(segment_nomes: Sequence[Number]) -> tuple[Number, ...]:
    r"""Return ``rho_i=4^(delta_i1+delta_ir) p_i``."""

    nomes = tuple(as_mpmath(value) for value in segment_nomes)
    if not nomes:
        raise ValueError("at least one segment nome is required")
    if len(nomes) == 1:
        return (16 * nomes[0],)
    return (4 * nomes[0], *nomes[1:-1], 4 * nomes[-1])


def _validate_coordinates_and_weights(
    z: Number,
    mobile_positions: Sequence[Number],
    external_weights: Sequence[Number],
) -> tuple[Number, tuple[Number, ...], tuple[Number, ...]]:
    z_value = as_mpmath(z)
    mobiles = tuple(as_mpmath(value) for value in mobile_positions)
    external = tuple(as_mpmath(value) for value in external_weights)
    if len(external) != len(mobiles) + 4:
        raise ValueError("external weights must have length 4+number_of_mobile_positions")
    return z_value, mobiles, external


def lambda_prefactor(
    *,
    kappa: Number,
    q: Number,
    z: Number,
    mobile_positions: Sequence[Number],
    external_weights: Sequence[Number],
    branch: str = "ordered_real",
) -> Number:
    r"""Return the exact ``Lambda_n^(kappa)`` conformal factor.

    ``branch='ordered_real'`` replaces ``z-t_j`` by the positive quantity
    ``t_j-z``.  Use ``branch='holomorphic'`` to retain the literal complex
    factor ``z-t_j`` and manage analytic continuation consistently.
    """

    z_value, mobiles, external = _validate_coordinates_and_weights(
        z, mobile_positions, external_weights
    )
    kappa, q = as_mpmath(kappa), as_mpmath(q)
    d_zero, d_z = external[0], external[1]
    mobile_weights = external[2:-2]
    d_one, d_infinity = external[-2], external[-1]
    theta_exponent = (
        kappa / 2
        - 4 * (d_zero + d_z + d_one + d_infinity)
        - 2 * mp.fsum(mobile_weights)
    )
    result: Number = (
        theta3_from_nome(q) ** theta_exponent
        * z_value ** (kappa / 24 - d_zero - d_z)
        * (1 - z_value) ** (kappa / 24 - d_z - d_one)
    )
    if branch not in {"ordered_real", "holomorphic"}:
        raise ValueError("branch must be 'ordered_real' or 'holomorphic'")
    for t, weight in zip(mobiles, mobile_weights):
        separation = t - z_value if branch == "ordered_real" else z_value - t
        result *= (t * (1 - t) * separation) ** (-weight / 2)
    return result


def reconstruct_sphere_block(
    table: RecursionTable,
    *,
    segment_nomes: Sequence[Number],
    z: Number,
    mobile_positions: Sequence[Number] = (),
    order: int | None = None,
    branch: str = "ordered_real",
) -> Number:
    r"""Restore the plane-normalized chiral sphere block.

    This evaluates

    ``Lambda_n^(c-1) prod_i rho_i^(h_i-(c-1)/24) H_n``.
    """

    if len(segment_nomes) != table.edge_count:
        raise ValueError("segment nome count does not match the recursion table")
    if len(mobile_positions) != table.point_count - 4:
        raise ValueError("mobile-position count does not match the recursion table")
    truncation = table.order if order is None else int(order)
    with mp.workdps(table.dps):
        nomes = tuple(as_mpmath(value) for value in segment_nomes)
        q = mp.fprod(nomes)
        kappa = table.central_charge - 1
        delta = kappa / 24
        prefactor = lambda_prefactor(
            kappa=kappa,
            q=q,
            z=z,
            mobile_positions=mobile_positions,
            external_weights=table.external_weights,
            branch=branch,
        )
        propagation: Number = mp.mpf(1)
        for rho, weight in zip(
            effective_plumbing_parameters(nomes), table.internal_weights
        ):
            propagation *= rho ** (weight - delta)
        return +(prefactor * propagation * table.evaluate(nomes, order=truncation))


def reconstruct_from_real_moduli(
    table: RecursionTable,
    *,
    z: Number,
    mobile_positions: Sequence[Number] = (),
    order: int | None = None,
) -> SphereBlockEvaluation:
    """Invert the ordered real coordinates and reconstruct the sphere block."""

    truncation = table.order if order is None else int(order)
    nomes = invert_aligned_coordinates(
        z,
        mobile_positions,
        dps=table.dps,
    )
    reduced = table.evaluate(nomes.segment_nomes, order=truncation)
    value = reconstruct_sphere_block(
        table,
        segment_nomes=nomes.segment_nomes,
        z=z,
        mobile_positions=mobile_positions,
        order=truncation,
        branch="ordered_real",
    )
    return SphereBlockEvaluation(
        value=value,
        reduced_value=reduced,
        nomes=nomes,
        order=truncation,
    )


def comb_cross_ratios(z: Number, mobile_positions: Sequence[Number] = ()) -> tuple[Number, ...]:
    r"""Return ``(z/t1,t1/t2,...,t_m)``; for four points return ``(z,)``."""

    z_value = as_mpmath(z)
    mobiles = tuple(as_mpmath(value) for value in mobile_positions)
    if not mobiles:
        return (z_value,)
    return (
        z_value / mobiles[0],
        *(mobiles[index] / mobiles[index + 1] for index in range(len(mobiles) - 1)),
        mobiles[-1],
    )


def plane_primary_factor(
    *,
    z: Number,
    mobile_positions: Sequence[Number],
    external_weights: Sequence[Number],
    internal_weights: Sequence[Number],
) -> Number:
    """Return the primary coordinate power in the stated comb channel."""

    z_value, mobiles, external = _validate_coordinates_and_weights(
        z, mobile_positions, external_weights
    )
    internal = tuple(as_mpmath(value) for value in internal_weights)
    if len(internal) != len(external) - 3:
        raise ValueError("internal-weight count does not match the comb channel")
    result: Number = z_value ** (internal[0] - external[0] - external[1])
    for index, (t, mobile_weight) in enumerate(zip(mobiles, external[2:-2])):
        result *= t ** (internal[index + 1] - internal[index] - mobile_weight)
    return result

