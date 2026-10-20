"""Genus-zero Virasoro n-point elliptic h-recursion.

The public API is intentionally small.  See ``README.md`` and ``FORMULAS.md``
in the source distribution for conventions and validation status.
"""

from .block import (
    SphereBlockEvaluation,
    comb_cross_ratios,
    effective_plumbing_parameters,
    lambda_prefactor,
    plane_primary_factor,
    reconstruct_from_real_moduli,
    reconstruct_sphere_block,
)
from .geometry import (
    AlignedNomes,
    coordinates_from_segment_nomes,
    cross_ratio_from_nome,
    elliptic_nome,
    invert_aligned_coordinates,
    mobile_position_from_right_product,
    theta3_from_nome,
)
from .recursion import (
    KacPoleError,
    PoleContext,
    RecursionTable,
    compute_h_recursion,
    total_degree_indices,
)

__all__ = [
    "AlignedNomes",
    "KacPoleError",
    "PoleContext",
    "RecursionTable",
    "SphereBlockEvaluation",
    "comb_cross_ratios",
    "compute_h_recursion",
    "coordinates_from_segment_nomes",
    "cross_ratio_from_nome",
    "effective_plumbing_parameters",
    "elliptic_nome",
    "invert_aligned_coordinates",
    "lambda_prefactor",
    "mobile_position_from_right_product",
    "plane_primary_factor",
    "reconstruct_from_real_moduli",
    "reconstruct_sphere_block",
    "theta3_from_nome",
    "total_degree_indices",
]

__version__ = "0.1.0"

