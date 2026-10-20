"""BRY-first amplitude conventions; intrinsic CFT data are unchanged.

Ordinary area measures, alpha'=1 units, r=R_phys/sqrt(alpha'), and
mu^-1=2*pi*g_s. Xi is an explicit legacy input convention, never an
unlabelled default. A convention conversion is not an absolute-normalization
certificate and does not insert or remove any moduli-space quotient.
"""

from __future__ import annotations

import math

BRY_STRING_COUPLING_CONVENTION = "bry-mu-inverse-equals-2pi-gs"
XI_STRING_COUPLING_CONVENTION = "xi-mu-inverse-equals-4pi-gs"
MQM_INVERSE_MU_CONVENTION = "mqm-mu-inverse-expansion"
CURRENT_STRING_COUPLING_CONVENTION = BRY_STRING_COUPLING_CONVENTION
BRY_GENUS1_TWO_POINT_PREFACTOR = 2.0 * math.pi**2
BRY_RESIDUAL_SPHERE_METRIC = 1.0
BRY_REAL_TUBE_MEASURE_COEFFICIENT = 1.0
BRY_GENUS1_TWO_POINT_FORMULA = "A_1^ws(omega)=2*pi^2*i*g_s^2*I_1(omega)"
XI_GENUS1_TWO_POINT_FORMULA = "A_1^ws(omega)=8*pi^2*i*g_s^2*I_1(omega)"


def genus1_two_point_source_convention(record: dict) -> str:
    """Recognize frozen torus metadata without rescaling its reduced I."""
    formulas = {BRY_GENUS1_TWO_POINT_FORMULA: BRY_STRING_COUPLING_CONVENTION,
                XI_GENUS1_TWO_POINT_FORMULA: XI_STRING_COUPLING_CONVENTION}
    convention = formulas.get(record.get("native_normalization"))
    if convention is None or record.get("string_coupling_convention", convention) != convention:
        raise ValueError("unknown or contradictory torus amplitude convention")
    return convention


def amplitude_coefficient_scale_to_bry(
    genus: int, punctures: int, source_convention: str,
) -> float:
    """Scale a coupling-stripped amplitude, not a reduced CFT integral.

For A=g_Xi**k*a_Xi=g_BRY**k*a_BRY, k=2g-2+n, a_BRY=2**(-k)*a_Xi.
For A=mu**(-k)*a_MQM, a_BRY=(2*pi)**k*a_MQM.
Absolute standard errors scale by the same positive number; covariance by
its square. Never apply this factor again to a BRY-labelled coefficient.
"""
    for name, value in (("genus", genus), ("punctures", punctures)):
        if isinstance(value, bool) or int(value) != value or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if source_convention == MQM_INVERSE_MU_CONVENTION:
        return (2.0 * math.pi) ** (2 * int(genus) - 2 + int(punctures))
    if source_convention == BRY_STRING_COUPLING_CONVENTION:
        return 1.0
    if source_convention == XI_STRING_COUPLING_CONVENTION:
        return 2.0 ** (-(2 * int(genus) - 2 + int(punctures)))
    raise ValueError(f"unknown string-coupling convention {source_convention!r}")


def log_partition_to_free_energy_scale(radius: float, *, genus: int,
                                       source_convention: str) -> float:
    """Convert a connected log Z coefficient to F_h, with g_s^(2h-2) stripped.

    F=-log Z/(2*pi*R). A coefficient of mu^(2-2h) first acquires
    (2*pi)^(2h-2). Standard errors use abs(scale), covariance uses s_i*s_j.
    This does not change the kernel's automorphism or Fermi-sea factors.
    """
    radius = float(radius)
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be positive and finite")
    return -amplitude_coefficient_scale_to_bry(genus, 0, source_convention) / (2 * math.pi * radius)


def matrix_model_genus1_free_energy(radius: float, mu: float) -> float:
    """F_1=(1+R^-2)*log(mu)/(48*pi), in the one-sided Fermi-sea convention."""
    log_partition_to_free_energy_scale(radius, genus=1, source_convention=CURRENT_STRING_COUPLING_CONVENTION)
    bry_string_coupling_from_mqm_fermi_level(mu)
    return (1 + float(radius)**-2) * math.log(mu) / (48 * math.pi)


def matrix_model_genus2_free_energy(radius: float) -> float:
    """F_2, so the genus-two term in F is g_s^2*F_2, not F_2 itself."""
    log_partition_to_free_energy_scale(radius, genus=2, source_convention=CURRENT_STRING_COUPLING_CONVENTION)
    return -math.pi * (7 + 10 * float(radius)**-2 + 7 * float(radius)**-4) / 2880


def bry_string_coupling_from_mqm_fermi_level(mu: float) -> float:
    mu = float(mu)
    if not math.isfinite(mu) or mu <= 0:
        raise ValueError("mu must be positive and finite")
    return 1.0 / (2.0 * math.pi * mu)


def bry_sphere_normalization(g_s: float) -> float:
    g_s = float(g_s)
    if not math.isfinite(g_s) or g_s <= 0:
        raise ValueError("g_s must be positive and finite")
    return 2.0 * math.pi / g_s**2
