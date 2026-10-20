"""Finite-interval Liouville momentum quadrature shared by tree amplitudes.

The legacy helper name is retained for compatibility. With one panel and
power greater than one it uses a mapped Gauss-Jacobi rule; the other branches
use composite Gauss-Legendre quadrature.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy.special import roots_jacobi


def _gauss_legendre_grid(
    order: int,
    maximum: float,
    panels: int = 1,
    breakpoints: Sequence[float] = (),
    power: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a composite rule split at requested physical-P endpoints.

    ``power>1`` applies ``P=maximum*u**power`` before quadrature.  This
    concentrates nodes near the Liouville threshold while retaining every
    requested breakpoint exactly.
    """

    order = int(order)
    panels = int(panels)
    maximum = float(maximum)
    power = float(power)
    if order <= 0 or panels <= 0 or not math.isfinite(maximum) or maximum <= 0.0:
        raise ValueError("order, panels, and maximum must be positive")
    if not math.isfinite(power) or power < 1.0:
        raise ValueError("power must be finite and at least one")
    canonical_nodes, canonical_weights = np.polynomial.legendre.leggauss(order)
    boundaries = {0.0, maximum}
    boundaries.update(panel * maximum / panels for panel in range(1, panels))
    boundaries.update(
        float(value)
        for value in breakpoints
        if math.isfinite(float(value)) and 0.0 < float(value) < maximum
    )
    ordered_boundaries = sorted(boundaries)
    if len(ordered_boundaries) == 2 and power != 1.0:
        # Integrate the Jacobian weight u**(power-1) with its native
        # Gauss--Jacobi rule.  Besides resolving the threshold, this keeps
        # normalization moments exact and places the first node less
        # aggressively than an ordinary Gauss rule applied after the map.
        jacobi_nodes, jacobi_weights = roots_jacobi(
            order,
            0.0,
            power - 1.0,
        )
        nodes_u = 0.5 * (jacobi_nodes + 1.0)
        nodes_p = maximum * nodes_u**power
        weights_p = (
            maximum
            * power
            * 2.0 ** (-power)
            * np.asarray(jacobi_weights, dtype=float)
        )
        return np.asarray(nodes_p, dtype=float), weights_p
    all_nodes: list[np.ndarray] = []
    all_weights: list[np.ndarray] = []
    for left, right in zip(ordered_boundaries[:-1], ordered_boundaries[1:]):
        left_u = (left / maximum) ** (1.0 / power)
        right_u = (right / maximum) ** (1.0 / power)
        panel_width_u = right_u - left_u
        nodes_u = left_u + 0.5 * panel_width_u * (canonical_nodes + 1.0)
        nodes_p = maximum * nodes_u**power
        jacobian = maximum * power * nodes_u ** (power - 1.0)
        all_nodes.append(nodes_p)
        all_weights.append(0.5 * panel_width_u * canonical_weights * jacobian)
    return np.concatenate(all_nodes), np.concatenate(all_weights)
