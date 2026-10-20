#!/usr/bin/env python3
"""Genus-two Gottschling domain and invariant Siegel-volume sampling.

Write a genus-two period matrix as

    Omega = [[x1 + i y1, x2 + i y2],
             [x2 + i y2, x3 + i y3]].

The implementation below uses Gottschling's finite description of the
fundamental domain: the real-part box, the Minkowski-reduced imaginary cone,
and the 19 height inequalities.  Boundaries have zero invariant volume and
are included with a numerical tolerance.

The invariant measure convention is

    d mu_2 = prod_{i<=j} dX_ij dY_ij / det(Y)^3,

for which the exact volume is pi^3/270.  The proposal distribution is chosen
so that accept/reject sampling from this measure has a bounded analytic
envelope.  Accepted matrices are therefore unweighted samples from the full
six-real-dimensional fundamental domain, rather than samples from a box or a
special slice.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


SQRT3_OVER_2 = math.sqrt(3.0) / 2.0
SIEGEL_VOLUME_G2 = math.pi**3 / 270.0

# In the variables used by ``draw_minkowski_proposals``, the ratio between the
# invariant density and the proposal density is bounded by this value.  The
# maximum is attained at t3=0 and r=1.
INVARIANT_WEIGHT_MAX = 128.0 / (243.0 * math.sqrt(3.0))


@dataclass(frozen=True)
class VolumeEstimate:
    proposal_count: int
    seed: int
    exact_volume: float
    importance_estimate: float
    importance_standard_error: float
    importance_z_score: float
    rejection_estimate: float
    rejection_standard_error: float
    rejection_z_score: float
    domain_fraction_under_proposal: float
    rejection_acceptance_fraction: float
    max_observed_weight: float
    analytic_weight_bound: float


@dataclass(frozen=True)
class InvariantSample:
    omega: np.ndarray
    proposal_count: int
    accepted_count: int
    seed: int | None


def gottschling_shift_matrices() -> tuple[np.ndarray, ...]:
    """Return the 15 symmetric shifts in the determinant inequalities."""

    shifts = [np.zeros((2, 2), dtype=float)]
    for e in (-1.0, 1.0):
        shifts.extend(
            [
                np.asarray([[e, 0.0], [0.0, 0.0]]),
                np.asarray([[0.0, 0.0], [0.0, e]]),
                np.asarray([[e, 0.0], [0.0, e]]),
                np.asarray([[e, 0.0], [0.0, -e]]),
                np.asarray([[0.0, e], [e, 0.0]]),
                np.asarray([[e, e], [e, 0.0]]),
                np.asarray([[0.0, e], [e, e]]),
            ]
        )
    return tuple(shifts)


GOTTSCHLING_SHIFTS = gottschling_shift_matrices()


def _as_omega_batch(omega: np.ndarray) -> tuple[np.ndarray, bool]:
    values = np.asarray(omega, dtype=np.complex128)
    scalar = values.shape == (2, 2)
    if scalar:
        values = values[np.newaxis, ...]
    if values.ndim != 3 or values.shape[1:] != (2, 2):
        raise ValueError(f"omega must have shape (2,2) or (n,2,2), got {values.shape}")
    return values, scalar


def gottschling_min_margin(omega: np.ndarray) -> np.ndarray | float:
    """Return the smallest margin among the exact finite domain inequalities.

    A nonnegative margin means that the matrix is in the closed Gottschling
    domain.  The symmetry and positive-definiteness tests are included.
    """

    values, scalar = _as_omega_batch(omega)
    x = values.real
    y = values.imag
    margins: list[np.ndarray] = []

    margins.extend(0.5 - np.abs(x[:, index, column]) for index, column in ((0, 0), (0, 1), (1, 1)))
    margins.append(y[:, 0, 0] - SQRT3_OVER_2)
    margins.append(y[:, 1, 1] - SQRT3_OVER_2)
    margins.append(y[:, 0, 1])
    margins.append(y[:, 0, 0] - 2.0 * y[:, 0, 1])
    margins.append(y[:, 1, 1] - y[:, 0, 0])

    symmetry_error = np.abs(values[:, 0, 1] - values[:, 1, 0])
    margins.append(np.where(symmetry_error <= 1.0e-13, np.inf, -symmetry_error))
    margins.append(y[:, 0, 0])
    margins.append(np.linalg.det(y).real)

    b11 = values[:, 0, 0]
    b12 = values[:, 0, 1]
    b22 = values[:, 1, 1]
    margins.append(np.abs(b11) - 1.0)
    margins.append(np.abs(b22) - 1.0)
    for e in (-1.0, 1.0):
        margins.append(np.abs(b11 + b22 - 2.0 * b12 + e) - 1.0)
    for shift in GOTTSCHLING_SHIFTS:
        margins.append(np.abs(np.linalg.det(values + shift[np.newaxis, ...])) - 1.0)

    minimum = np.min(np.stack(margins, axis=1), axis=1)
    return float(minimum[0]) if scalar else minimum


def in_gottschling_domain(omega: np.ndarray, *, tolerance: float = 1.0e-12) -> np.ndarray | bool:
    """Test membership in the closed genus-two fundamental domain."""

    margin = gottschling_min_margin(omega)
    if np.isscalar(margin):
        return bool(float(margin) >= -float(tolerance))
    return np.asarray(margin) >= -float(tolerance)


def draw_minkowski_proposals(
    rng: np.random.Generator,
    count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw proposals on the reduced imaginary cone.

    The coordinates are

        y1 = a exp(t1), y3 = y1 exp(t3), y2 = r y1/2,

    with t1~Exp(3), t3~Exp(2), r~Uniform(0,1), and a=sqrt(3)/2.
    The returned weights are the invariant density divided by the proposal
    density, before imposing the remaining Gottschling inequalities.
    """

    count = int(count)
    if count <= 0:
        raise ValueError("count must be positive")

    x = rng.uniform(-0.5, 0.5, size=(count, 3))
    t1 = rng.exponential(scale=1.0 / 3.0, size=count)
    t3 = rng.exponential(scale=1.0 / 2.0, size=count)
    r = rng.uniform(0.0, 1.0, size=count)

    y1 = SQRT3_OVER_2 * np.exp(t1)
    z = np.exp(t3)
    y3 = y1 * z
    y2 = 0.5 * r * y1

    omega = np.empty((count, 2, 2), dtype=np.complex128)
    omega[:, 0, 0] = x[:, 0] + 1.0j * y1
    omega[:, 0, 1] = x[:, 1] + 1.0j * y2
    omega[:, 1, 0] = omega[:, 0, 1]
    omega[:, 1, 1] = x[:, 2] + 1.0j * y3

    weight = (1.0 / (12.0 * SQRT3_OVER_2**3)) * (z / (z - 0.25 * r**2)) ** 3
    coordinates = np.column_stack((t1, t3, r))
    return omega, weight, coordinates


def minkowski_proposals_from_unit_cube(
    points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Map six-dimensional unit-cube points to the proposal measure.

    Columns zero through two generate the real-part box.  Columns three and
    four are transformed by the inverse exponential CDFs for ``t1`` and
    ``t3``; column five is ``r``.  This deterministic form of the proposal map
    is used by randomized quasi-Monte Carlo designs.  Points must lie in
    ``[0,1)`` so the unbounded cusp coordinates remain finite.
    """

    unit = np.asarray(points, dtype=np.float64)
    if unit.ndim != 2 or unit.shape[1] != 6:
        raise ValueError(f"points must have shape (n,6), got {unit.shape}")
    if unit.shape[0] == 0:
        raise ValueError("at least one unit-cube point is required")
    if not np.all(np.isfinite(unit)) or np.any(unit < 0.0) or np.any(unit >= 1.0):
        raise ValueError("unit-cube points must be finite and lie in [0,1)")

    x = unit[:, :3] - 0.5
    t1 = -np.log1p(-unit[:, 3]) / 3.0
    t3 = -np.log1p(-unit[:, 4]) / 2.0
    r = unit[:, 5]
    y1 = SQRT3_OVER_2 * np.exp(t1)
    z = np.exp(t3)
    y3 = y1 * z
    y2 = 0.5 * r * y1

    omega = np.empty((unit.shape[0], 2, 2), dtype=np.complex128)
    omega[:, 0, 0] = x[:, 0] + 1.0j * y1
    omega[:, 0, 1] = x[:, 1] + 1.0j * y2
    omega[:, 1, 0] = omega[:, 0, 1]
    omega[:, 1, 1] = x[:, 2] + 1.0j * y3

    weight = (1.0 / (12.0 * SQRT3_OVER_2**3)) * (z / (z - 0.25 * r**2)) ** 3
    coordinates = np.column_stack((t1, t3, r))
    return omega, weight, coordinates


def estimate_invariant_volume(*, proposal_count: int = 200_000, seed: int = 20260711) -> VolumeEstimate:
    """Estimate the domain volume in two independent Monte Carlo forms."""

    proposal_count = int(proposal_count)
    if proposal_count <= 1:
        raise ValueError("proposal_count must exceed one")
    rng = np.random.default_rng(int(seed))
    omega, weight, _ = draw_minkowski_proposals(rng, proposal_count)
    domain = np.asarray(in_gottschling_domain(omega), dtype=bool)
    weighted_indicator = weight * domain

    importance_estimate = float(np.mean(weighted_indicator))
    importance_se = float(np.std(weighted_indicator, ddof=1) / math.sqrt(proposal_count))

    acceptance_probability = np.where(domain, weight / INVARIANT_WEIGHT_MAX, 0.0)
    if float(np.max(acceptance_probability)) > 1.0 + 1.0e-12:
        raise RuntimeError("analytic invariant-weight envelope was violated")
    accepted = rng.random(proposal_count) < acceptance_probability
    accepted_fraction = float(np.mean(accepted))
    rejection_estimate = INVARIANT_WEIGHT_MAX * accepted_fraction
    rejection_se = INVARIANT_WEIGHT_MAX * math.sqrt(
        accepted_fraction * (1.0 - accepted_fraction) / proposal_count
    )

    return VolumeEstimate(
        proposal_count=proposal_count,
        seed=int(seed),
        exact_volume=SIEGEL_VOLUME_G2,
        importance_estimate=importance_estimate,
        importance_standard_error=importance_se,
        importance_z_score=(importance_estimate - SIEGEL_VOLUME_G2) / importance_se,
        rejection_estimate=rejection_estimate,
        rejection_standard_error=rejection_se,
        rejection_z_score=(rejection_estimate - SIEGEL_VOLUME_G2) / rejection_se,
        domain_fraction_under_proposal=float(np.mean(domain)),
        rejection_acceptance_fraction=accepted_fraction,
        max_observed_weight=float(np.max(weight)),
        analytic_weight_bound=INVARIANT_WEIGHT_MAX,
    )


def sample_invariant_domain(
    count: int,
    *,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
    batch_size: int = 4096,
) -> InvariantSample:
    """Return unweighted invariant-volume samples from the full domain."""

    count = int(count)
    if count <= 0:
        raise ValueError("count must be positive")
    if rng is not None and seed is not None:
        raise ValueError("pass either rng or seed, not both")
    if rng is None:
        rng = np.random.default_rng(seed)

    chunks: list[np.ndarray] = []
    accepted_count = 0
    proposal_count = 0
    while accepted_count < count:
        remaining = count - accepted_count
        current_batch = max(int(batch_size), int(math.ceil(remaining / 0.30)))
        omega, weight, _ = draw_minkowski_proposals(rng, current_batch)
        domain = np.asarray(in_gottschling_domain(omega), dtype=bool)
        probability = np.where(domain, weight / INVARIANT_WEIGHT_MAX, 0.0)
        mask = rng.random(current_batch) < probability
        selected = omega[mask]
        if selected.size:
            chunks.append(selected[:remaining])
            accepted_count += min(len(selected), remaining)
        proposal_count += current_batch

    samples = np.concatenate(chunks, axis=0)[:count]
    return InvariantSample(
        omega=samples,
        proposal_count=proposal_count,
        accepted_count=count,
        seed=None if seed is None else int(seed),
    )


def run(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Validate and sample the genus-two Gottschling domain.")
    parser.add_argument("--proposal-count", type=int, default=200_000)
    parser.add_argument("--sample-count", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--out-npz", type=Path)
    args = parser.parse_args(argv)

    estimate = estimate_invariant_volume(proposal_count=args.proposal_count, seed=args.seed)
    payload: dict[str, object] = {"volume_check": asdict(estimate)}
    print("Genus-two Gottschling-domain volume check")
    print(f"  exact pi^3/270       = {estimate.exact_volume:.12g}")
    print(
        "  importance estimate = "
        f"{estimate.importance_estimate:.12g} +/- {estimate.importance_standard_error:.2g} "
        f"(z={estimate.importance_z_score:.2f})"
    )
    print(
        "  rejection estimate  = "
        f"{estimate.rejection_estimate:.12g} +/- {estimate.rejection_standard_error:.2g} "
        f"(z={estimate.rejection_z_score:.2f})"
    )

    sample = None
    if args.sample_count:
        sample = sample_invariant_domain(args.sample_count, seed=args.seed + 1)
        payload["sample"] = {
            "count": sample.accepted_count,
            "proposal_count": sample.proposal_count,
            "seed": sample.seed,
        }
        print(f"  drew {sample.accepted_count} invariant samples from {sample.proposal_count} proposals")

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"  wrote {args.out_json}")
    if args.out_npz is not None:
        if sample is None:
            raise ValueError("--out-npz requires --sample-count")
        args.out_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.out_npz, omega=sample.omega)
        print(f"  wrote {args.out_npz}")


if __name__ == "__main__":
    run()
