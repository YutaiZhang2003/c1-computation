"""Saved sampling weights, normalization conventions, and moduli-integral estimates."""
from __future__ import annotations
import math
from typing import Sequence
import numpy as np
from genus2_integrand_normalization import GENUS2_FUNDAMENTAL_DOMAIN_WEIGHT, LEGACY_STRING_NOTE_DIMENSIONLESS_SCALAR_KERNEL_CONVENTION, CURRENT_INTEGRATION_KERNEL_CONVENTION, c1_genus2_topology_correction, genus2_fundamental_domain_node_weight, c1_bry_genus2_kernel_multiplier, genus2_automorphism_metadata, integration_kernel_scale_to_current, xi_full_replacement_over_dimensionless
from genus2_moduli_sampling import estimate_rqmc_integral
from genus2_moduli_sampling import PHYSICAL_MIXTURE_SAMPLING_SCHEME
from genus2_moduli_sampling import estimate_physical_mixture_integral
from genus2_siegel_fundamental_domain import SIEGEL_VOLUME_G2


RQMC_SAMPLING_SCHEME = "scrambled_sobol_minkowski_importance"
RQMC_SAMPLING_SCHEMES = {RQMC_SAMPLING_SCHEME, PHYSICAL_MIXTURE_SAMPLING_SCHEME}
LEGACY_UNIT_MUMFORD_KERNEL_CONVENTION = "unit-mumford-residue-no-string-note-prefactor"


def canonicalize_string_note_kernel_row(
    source_row: dict[str, object],
) -> dict[str, object]:
    """Return a COPY in BRY v7; the legacy function name is retained.

    Blank tags are accepted only here for pre-tagged unit-Mumford rows.
    All other unknown conventions fail closed. CFT data, proposal weights,
    and dimensionless relative errors are never rescaled.
    """

    row = dict(source_row)
    convention = str(row.get("integration_kernel_convention", "")).strip()
    if convention == CURRENT_INTEGRATION_KERNEL_CONVENTION:
        row.update(genus2_automorphism_metadata())
        return row
    source = convention or LEGACY_UNIT_MUMFORD_KERNEL_CONVENTION
    alpha_prime = float(row.get("alpha_prime", 1.0) or 1.0)
    scale = integration_kernel_scale_to_current(source, alpha_prime)
    final_multiplier = c1_bry_genus2_kernel_multiplier(alpha_prime)
    scalar_scale = xi_full_replacement_over_dimensionless(alpha_prime)
    dimensionless_source = source in {
        LEGACY_UNIT_MUMFORD_KERNEL_CONVENTION,
        LEGACY_STRING_NOTE_DIMENSIONLESS_SCALAR_KERNEL_CONVENTION,
    }

    # Scale every stored order, including promoted/final values, coherently.
    for order in ("low", "high", "final"):
        key = f"transformed_integrand_{order}"
        if key in row and str(row[key]).strip() != "":
            row[key] = scale * float(row[key])
            row[f"unit_mumford_transformed_integrand_{order}"] = (
                row[key] / final_multiplier
            )
        for stem, is_log in (("factorized_density", False),
                             ("factorized_log_density", True)):
            key = f"{stem}_{order}"
            if key in row and str(row[key]).strip() != "" and dimensionless_source:
                row[key] = (float(row[key]) + math.log(scalar_scale)
                            if is_log else float(row[key]) * scalar_scale)
        for stem, base_stem, is_log in (
            ("string_note_kernel_density", "factorized_density", False),
            ("string_note_kernel_log_density", "factorized_log_density", True),
        ):
            key, base_key = f"{stem}_{order}", f"{base_stem}_{order}"
            if base_key in row and str(row[base_key]).strip() != "":
                row[key] = (float(row[base_key]) + math.log(final_multiplier)
                            if is_log else float(row[base_key]) * final_multiplier)
            elif key in row and str(row[key]).strip() != "":
                row[key] = (float(row[key]) + math.log(scale)
                            if is_log else float(row[key]) * scale)

    # Absolute node contributions/errors follow the amplitude, not the PDF.
    for key in ("node_contribution", "node_contribution_low",
                "node_contribution_high", "node_contribution_final",
                "weighted_order_delta", "node_absolute_error",
                "cft_absolute_error", "transformed_integrand_absolute_error"):
        if key in row and str(row[key]).strip() != "":
            row[key] = float(row[key]) * scale
    row["integration_kernel_convention"] = CURRENT_INTEGRATION_KERNEL_CONVENTION
    row.update(genus2_automorphism_metadata())
    row["string_note_kernel_multiplier"] = final_multiplier
    row["xi_scalar_over_dimensionless_full_integrand"] = scalar_scale
    row["c1_sphere_topology_multiplier"] = c1_genus2_topology_correction(alpha_prime)
    row["source_integration_kernel_convention"] = source
    row["integration_kernel_conversion_scale"] = scale
    row["legacy_kernel_converted"] = True
    return row


def omega_from_csv_row(row: dict[str, str]) -> np.ndarray:
    return np.asarray(
        [
            [
                complex(float(row["x11"]), float(row["y11"])),
                complex(float(row["x12"]), float(row["y12"])),
            ],
            [
                complex(float(row["x12"]), float(row["y12"])),
                complex(float(row["x22"]), float(row["y22"])),
            ],
        ],
        dtype=np.complex128,
    )


def node_fundamental_domain_integration_weight(
    source_row: dict[str, str],
) -> float:
    """Return the full fundamental-domain weight for one transformed node.

    Proposal data are authoritative.  Historical half-weight columns are not
    accepted as an integration input.
    """

    scheme = source_row.get("sampling_scheme", "iid_invariant_domain")
    if scheme in RQMC_SAMPLING_SCHEMES:
        return genus2_fundamental_domain_node_weight(source_row)
    if scheme in {"", "iid", "iid_invariant_domain"}:
        return GENUS2_FUNDAMENTAL_DOMAIN_WEIGHT * SIEGEL_VOLUME_G2
    raise ValueError(f"unsupported sampling scheme {scheme!r}")


def kernel_det_im_power(source_row: dict[str, object]) -> int:
    """Return the proposal-dependent power multiplying the local kernel."""

    scheme = str(source_row.get("sampling_scheme", "iid_invariant_domain"))
    if scheme == PHYSICAL_MIXTURE_SAMPLING_SCHEME:
        power = int(source_row.get("rqmc_kernel_det_im_power", 0))
        if power != 0:
            raise ValueError("physical-mixture rows must use kernel det(Y) power zero")
        return 0
    if scheme in {"", "iid", "iid_invariant_domain", RQMC_SAMPLING_SCHEME}:
        return 3
    raise ValueError(f"unsupported sampling scheme {scheme!r}")


def sampling_scheme_for_rows(rows: Sequence[dict[str, object]]) -> str:
    """Return one validated sampling scheme for a collection of rows."""

    schemes = {
        str(row.get("sampling_scheme", "iid_invariant_domain"))
        for row in rows
    }
    schemes.discard("")
    if not schemes:
        return "iid_invariant_domain"
    if len(schemes) != 1:
        raise ValueError(f"mixed sampling schemes are not supported: {sorted(schemes)}")
    scheme = schemes.pop()
    if scheme == "iid":
        return "iid_invariant_domain"
    if scheme not in {"iid_invariant_domain", *RQMC_SAMPLING_SCHEMES}:
        raise ValueError(f"unsupported sampling scheme {scheme!r}")
    return scheme


def summarize_rqmc_rows(
    rows: Sequence[dict[str, object]],
    *,
    value_key: str,
) -> dict[str, object] | None:
    """Summarize only a complete collection of at least two scrambles."""

    if not rows:
        return None
    groups: dict[int, list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(int(row["rqmc_replicate"]), []).append(row)
    if len(groups) < 2:
        return None
    for replicate, group in groups.items():
        expected = int(group[0]["rqmc_domain_count"])
        if len(group) != expected or any(row.get("status") != "ok" for row in group):
            return None
        if any(int(row["rqmc_domain_count"]) != expected for row in group):
            raise ValueError(f"replicate {replicate} has inconsistent domain counts")

    scheme = sampling_scheme_for_rows(rows)
    values = [float(row[value_key]) for row in rows]
    if scheme == PHYSICAL_MIXTURE_SAMPLING_SCHEME:
        estimate = estimate_physical_mixture_integral(rows, values)
        return {
            "sampling_scheme": PHYSICAL_MIXTURE_SAMPLING_SCHEME,
            "replicate_count": estimate.replicate_count,
            "cft_node_count": estimate.cft_node_count,
            "estimate": estimate.estimate,
            "standard_error": estimate.scramble_standard_error,
            "replicate_estimates": list(estimate.replicate_estimates),
            "volume_calibrated_estimate": None,
            "volume_calibrated_standard_error": None,
            "volume_calibrated_replicate_estimates": None,
            "interpretation": (
                "The physical-measure balance-heuristic estimator is primary. "
                "Independent complete mixture scrambles determine its standard "
                "error; no det(Im Omega)^3 factor multiplies the local kernel."
            ),
        }
    estimate = estimate_rqmc_integral(rows, values)
    return {
        "sampling_scheme": RQMC_SAMPLING_SCHEME,
        "replicate_count": estimate.replicate_count,
        "cft_node_count": estimate.cft_node_count,
        "estimate": estimate.raw_estimate,
        "standard_error": estimate.raw_scramble_standard_error,
        "replicate_estimates": list(estimate.raw_replicate_estimates),
        "volume_calibrated_estimate": estimate.volume_calibrated_estimate,
        "volume_calibrated_standard_error": (
            estimate.volume_calibrated_scramble_standard_error
        ),
        "volume_calibrated_replicate_estimates": list(
            estimate.volume_calibrated_replicate_estimates
        ),
        "interpretation": (
            "The raw direct-importance estimator is primary. Independent complete "
            "scrambles determine its standard error. Exact-volume calibration is a "
            "finite-sample biased variance-reduction diagnostic."
        ),
    }


def estimate_from_transformed_values(values: Sequence[float]) -> dict[str, object]:
    """Summarize values g_i = det(Y_i)^3 I_i from invariant-domain draws."""

    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("at least one transformed integrand value is required")
    if not np.all(np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError("transformed integrand values must be finite and nonnegative")
    prefactor = GENUS2_FUNDAMENTAL_DOMAIN_WEIGHT * SIEGEL_VOLUME_G2
    contributions = prefactor * array
    estimate = float(np.mean(contributions))
    standard_error = (
        0.0
        if len(contributions) == 1
        else float(np.std(contributions, ddof=1) / math.sqrt(len(contributions)))
    )
    total = float(np.sum(contributions))
    squared = float(np.sum(contributions**2))
    effective_sample_size = 0.0 if squared == 0.0 else total * total / squared
    ordered = np.sort(contributions)[::-1]
    running = [float(np.mean(contributions[: index + 1])) for index in range(len(contributions))]
    return {
        "sample_count": int(len(contributions)),
        "estimate": estimate,
        "standard_error": standard_error,
        "normal_95_low": estimate - 1.959963984540054 * standard_error,
        "normal_95_high": estimate + 1.959963984540054 * standard_error,
        "fundamental_domain_estimate": estimate,
        "effective_sample_size": effective_sample_size,
        "largest_sample_fraction": 0.0 if total == 0.0 else float(ordered[0] / total),
        "largest_four_sample_fraction": (
            0.0 if total == 0.0 else float(np.sum(ordered[: min(4, len(ordered))]) / total)
        ),
        "running_estimate": running,
    }

