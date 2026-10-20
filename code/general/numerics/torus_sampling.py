"""Torus proposal and lattice terms used by the genus-two primary control."""
from __future__ import annotations
import math
import numpy as np

def proposal_transform(unit: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if unit.ndim != 2 or unit.shape[1] != 2:
        raise ValueError("unit sample must have shape (N,2)")
    x = np.asarray(unit[:, 0], dtype=float) - 0.5
    lower = np.sqrt(1.0 - x * x)
    survival = 1.0 - np.asarray(unit[:, 1], dtype=float)
    if np.any(survival <= 0.0):
        raise ValueError("Sobol cusp coordinate reached the excluded endpoint one")
    y = lower / (survival * survival)
    return x, y, lower


def theta_terms(
    radii: np.ndarray,
    *,
    alpha_prime: float,
    cutoff: int,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.arange(-int(cutoff), int(cutoff) + 1, dtype=float)
    m_grid, k_grid = np.meshgrid(values, values, indexing="ij")
    m = m_grid.ravel()
    k = k_grid.ravel()
    alpha_prime = float(alpha_prime)
    coefficients = math.pi * (
        radii[:, None] ** 2 * m[None, :] ** 2 / alpha_prime
        + alpha_prime * k[None, :] ** 2 / radii[:, None] ** 2
    )
    fundamental_floor = math.sqrt(3.0) / 2.0
    keep = np.min(coefficients, axis=0) * fundamental_floor <= -math.log(tolerance)
    keep |= (m == 0.0) & (k == 0.0)
    return coefficients[:, keep], (m[keep] * k[keep])
