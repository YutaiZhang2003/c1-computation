"""Invariant-measure and physical-mixture Sobol designs, with explicit sampling weights."""
from __future__ import annotations
from c1_paths import package_path
import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence
import numpy as np
from scipy.stats import qmc
from genus2_integrand_normalization import GENUS2_FUNDAMENTAL_DOMAIN_WEIGHT, GENUS2_FUNDAMENTAL_DOMAIN_WEIGHT_FIELD
from genus2_siegel_fundamental_domain import SIEGEL_VOLUME_G2, gottschling_min_margin, in_gottschling_domain, minkowski_proposals_from_unit_cube
from scipy.special import logsumexp
from genus2_integrand_normalization import GENUS2_FUNDAMENTAL_DOMAIN_WEIGHT, GENUS2_FUNDAMENTAL_DOMAIN_WEIGHT_FIELD, genus2_fundamental_domain_node_weight
from genus2_siegel_fundamental_domain import SIEGEL_VOLUME_G2, SQRT3_OVER_2, gottschling_min_margin, in_gottschling_domain

THIS_FILE = Path(__file__).resolve()


INVARIANT_DEFAULT_OUTPUT = Path(
    str(package_path('plumbing/results/genus2_c1_moduli_mc/rqmc_design_R8_M64'))
)


@dataclass(frozen=True)
class RQMCReplicateSummary:
    replicate: int
    scramble_seed: int
    power: int
    proposal_count: int
    domain_count: int
    domain_fraction: float
    coarse_domain_volume_estimate: float
    relative_volume_error: float
    proposal_discrepancy: float
    marginal_bin_count_min: int
    marginal_bin_count_max: int
    maximum_t1: float
    maximum_t3: float
    maximum_t1_tail_level: int
    maximum_t3_tail_level: int


@dataclass(frozen=True)
class RQMCIntegralEstimate:
    replicate_count: int
    cft_node_count: int
    raw_replicate_estimates: tuple[float, ...]
    raw_estimate: float
    raw_scramble_standard_error: float
    volume_calibrated_replicate_estimates: tuple[float, ...]
    volume_calibrated_estimate: float
    volume_calibrated_scramble_standard_error: float


def _tail_level(unit_coordinate: np.ndarray) -> np.ndarray:
    """Return dyadic tail levels with mass ``2^(-level-1)``."""

    return np.floor(-np.log2(1.0 - unit_coordinate)).astype(int)


def _marginal_balance(points: np.ndarray, bins: int) -> tuple[int, int]:
    counts = []
    for column in range(points.shape[1]):
        histogram, _ = np.histogram(points[:, column], bins=bins, range=(0.0, 1.0))
        counts.extend(int(value) for value in histogram)
    return min(counts), max(counts)


def generate_rqmc_replicate(
    *,
    replicate: int,
    power: int,
    scramble_seed: int,
    marginal_bins: int = 8,
) -> tuple[list[dict[str, object]], RQMCReplicateSummary]:
    """Generate one independently scrambled direct-importance replicate."""

    if power < 1:
        raise ValueError("power must be positive")
    proposal_count = 2**int(power)
    if marginal_bins < 1 or proposal_count % marginal_bins != 0:
        raise ValueError("marginal_bins must divide the proposal count")

    engine = qmc.Sobol(d=6, scramble=True, seed=int(scramble_seed))
    points = engine.random_base2(m=int(power))
    omega, invariant_weight, coordinates = minkowski_proposals_from_unit_cube(points)
    domain = np.asarray(in_gottschling_domain(omega), dtype=bool)
    margins = np.asarray(gottschling_min_margin(omega), dtype=float)
    domain_indices = np.flatnonzero(domain)
    domain_count = int(domain_indices.size)
    volume_estimate = float(np.mean(invariant_weight * domain))
    t1_tail = _tail_level(points[:, 3])
    t3_tail = _tail_level(points[:, 4])
    bin_min, bin_max = _marginal_balance(points, marginal_bins)

    rows: list[dict[str, object]] = []
    for proposal_index in domain_indices:
        value = omega[proposal_index]
        weight = float(invariant_weight[proposal_index])
        rows.append(
            {
                "sampling_scheme": "scrambled_sobol_minkowski_importance",
                "rqmc_node_id": f"r{int(replicate):03d}-p{int(proposal_index):08d}",
                "rqmc_replicate": int(replicate),
                "rqmc_scramble_seed": int(scramble_seed),
                "rqmc_power": int(power),
                "rqmc_proposal_count": proposal_count,
                "rqmc_proposal_index": int(proposal_index),
                "rqmc_domain_count": domain_count,
                "rqmc_invariant_weight": weight,
                "rqmc_coarse_volume_weight": weight / proposal_count,
                GENUS2_FUNDAMENTAL_DOMAIN_WEIGHT_FIELD: (
                    GENUS2_FUNDAMENTAL_DOMAIN_WEIGHT * weight / proposal_count
                ),
                "rqmc_u_x11": float(points[proposal_index, 0]),
                "rqmc_u_x12": float(points[proposal_index, 1]),
                "rqmc_u_x22": float(points[proposal_index, 2]),
                "rqmc_u_t1": float(points[proposal_index, 3]),
                "rqmc_u_t3": float(points[proposal_index, 4]),
                "rqmc_u_r": float(points[proposal_index, 5]),
                "rqmc_t1": float(coordinates[proposal_index, 0]),
                "rqmc_t3": float(coordinates[proposal_index, 1]),
                "rqmc_r": float(coordinates[proposal_index, 2]),
                "rqmc_t1_tail_level": int(t1_tail[proposal_index]),
                "rqmc_t3_tail_level": int(t3_tail[proposal_index]),
                "gottschling_margin": float(margins[proposal_index]),
                "det_im_omega": float(np.linalg.det(value.imag)),
                "x11": float(value[0, 0].real),
                "x12": float(value[0, 1].real),
                "x22": float(value[1, 1].real),
                "y11": float(value[0, 0].imag),
                "y12": float(value[0, 1].imag),
                "y22": float(value[1, 1].imag),
            }
        )

    summary = RQMCReplicateSummary(
        replicate=int(replicate),
        scramble_seed=int(scramble_seed),
        power=int(power),
        proposal_count=proposal_count,
        domain_count=domain_count,
        domain_fraction=domain_count / proposal_count,
        coarse_domain_volume_estimate=volume_estimate,
        relative_volume_error=volume_estimate / SIEGEL_VOLUME_G2 - 1.0,
        proposal_discrepancy=float(qmc.discrepancy(points)),
        marginal_bin_count_min=bin_min,
        marginal_bin_count_max=bin_max,
        maximum_t1=float(np.max(coordinates[:, 0])),
        maximum_t3=float(np.max(coordinates[:, 1])),
        maximum_t1_tail_level=int(np.max(t1_tail)),
        maximum_t3_tail_level=int(np.max(t3_tail)),
    )
    return rows, summary


def generate_rqmc_design(
    *,
    replicate_count: int,
    power: int,
    base_seed: int,
    marginal_bins: int = 8,
) -> tuple[list[dict[str, object]], list[RQMCReplicateSummary]]:
    """Generate all domain nodes for independent scrambled replicates."""

    if replicate_count < 2:
        raise ValueError("at least two replicates are required for an error estimate")
    rows: list[dict[str, object]] = []
    summaries: list[RQMCReplicateSummary] = []
    sample_index = 0
    for replicate in range(int(replicate_count)):
        replicate_rows, summary = generate_rqmc_replicate(
            replicate=replicate,
            power=power,
            scramble_seed=int(base_seed) + replicate,
            marginal_bins=marginal_bins,
        )
        for row in replicate_rows:
            row["sample_index"] = sample_index
            sample_index += 1
        rows.extend(replicate_rows)
        summaries.append(summary)
    return rows, summaries


def invariant_mean_and_scramble_se(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size < 2 or not np.all(np.isfinite(array)):
        raise ValueError("need at least two finite replicate values")
    return float(np.mean(array)), float(np.std(array, ddof=1) / math.sqrt(array.size))


def estimate_rqmc_integral(
    rows: Sequence[dict[str, object]],
    transformed_values: Sequence[float],
) -> RQMCIntegralEstimate:
    """Assemble raw and exact-volume-calibrated estimates by scramble."""

    if len(rows) != len(transformed_values) or not rows:
        raise ValueError("rows and transformed_values must have the same nonzero length")
    values = np.asarray(transformed_values, dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("transformed values must be finite and nonnegative")

    replicates = sorted({int(row["rqmc_replicate"]) for row in rows})
    raw: list[float] = []
    calibrated: list[float] = []
    for replicate in replicates:
        indices = [
            index
            for index, row in enumerate(rows)
            if int(row["rqmc_replicate"]) == replicate
        ]
        expected = int(rows[indices[0]]["rqmc_domain_count"])
        if len(indices) != expected:
            raise ValueError(
                f"replicate {replicate} is incomplete: {len(indices)} of {expected} domain nodes"
            )
        weights = np.asarray(
            [float(rows[index]["rqmc_invariant_weight"]) for index in indices]
        )
        proposal_count = int(rows[indices[0]]["rqmc_proposal_count"])
        if any(int(rows[index]["rqmc_proposal_count"]) != proposal_count for index in indices):
            raise ValueError(f"replicate {replicate} mixes proposal counts")
        replicate_values = values[indices]
        raw.append(
            float(
                GENUS2_FUNDAMENTAL_DOMAIN_WEIGHT
                * np.sum(weights * replicate_values)
                / proposal_count
            )
        )
        calibrated.append(
            float(
                GENUS2_FUNDAMENTAL_DOMAIN_WEIGHT
                * SIEGEL_VOLUME_G2
                * np.sum(weights * replicate_values)
                / np.sum(weights)
            )
        )

    raw_mean, raw_se = invariant_mean_and_scramble_se(raw)
    calibrated_mean, calibrated_se = invariant_mean_and_scramble_se(calibrated)
    return RQMCIntegralEstimate(
        replicate_count=len(replicates),
        cft_node_count=len(rows),
        raw_replicate_estimates=tuple(raw),
        raw_estimate=raw_mean,
        raw_scramble_standard_error=raw_se,
        volume_calibrated_replicate_estimates=tuple(calibrated),
        volume_calibrated_estimate=calibrated_mean,
        volume_calibrated_scramble_standard_error=calibrated_se,
    )


def invariant_write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def invariant_run(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate a genus-two scrambled-Sobol design.")
    parser.add_argument("--replicates", type=int, default=8)
    parser.add_argument("--power", type=int, default=6)
    parser.add_argument("--base-seed", type=int, default=20260712)
    parser.add_argument("--marginal-bins", type=int, default=8)
    parser.add_argument("--out-dir", type=Path, default=INVARIANT_DEFAULT_OUTPUT)
    args = parser.parse_args(list(argv) if argv is not None else None)

    rows, summaries = generate_rqmc_design(
        replicate_count=args.replicates,
        power=args.power,
        base_seed=args.base_seed,
        marginal_bins=args.marginal_bins,
    )
    volume_estimates = [summary.coarse_domain_volume_estimate for summary in summaries]
    volume_mean, volume_se = invariant_mean_and_scramble_se(volume_estimates)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    nodes_path = args.out_dir / "domain_nodes.csv"
    replicates_path = args.out_dir / "replicate_summary.csv"
    summary_path = args.out_dir / "summary.json"
    invariant_write_csv(nodes_path, rows)
    invariant_write_csv(replicates_path, [asdict(summary) for summary in summaries])
    payload = {
        "scope": (
            "Direct randomized-QMC importance design on the complete six-dimensional "
            "Minkowski proposal; only in-domain nodes require CFT evaluation."
        ),
        "estimator": (
            "J2=(1/2)*mean_over_all_proposals[1_F2*w*det(Im Omega)^3*I2]"
        ),
        "replicate_count": args.replicates,
        "power": args.power,
        "proposal_count_per_replicate": 2**args.power,
        "domain_cft_node_count": len(rows),
        "base_seed": args.base_seed,
        "marginal_bins": args.marginal_bins,
        "exact_coarse_domain_volume": SIEGEL_VOLUME_G2,
        "rqmc_volume_estimate": volume_mean,
        "rqmc_volume_scramble_standard_error": volume_se,
        "rqmc_volume_z_score": (volume_mean - SIEGEL_VOLUME_G2) / volume_se,
        "replicates": [asdict(summary) for summary in summaries],
        "notes": [
            "Every finite design has full proposal support in distribution; Sobol balance controls all resolved quantiles.",
            "Independent scrambles, rather than individual domain nodes, are the units used for the standard error.",
            "The raw direct-importance estimator is unbiased over the random scrambles.",
            "Exact-volume calibration is a lower-variance but finite-sample biased diagnostic and is not the primary estimator.",
            "No failed in-domain CFT node may be dropped from a replicate.",
            "The stable rqmc_node_id permits nested power-of-two extension without recomputing old CFT nodes.",
        ],
    }
    summary_path.write_text(json.dumps(payload, indent=2) + "\n")

    print("Genus-two randomized-QMC moduli design")
    print(
        f"  replicates={args.replicates}, proposals/replicate={2**args.power}, "
        f"CFT domain nodes={len(rows)}"
    )
    print(
        f"  volume={volume_mean:.12g} +/- {volume_se:.3g}; "
        f"exact={SIEGEL_VOLUME_G2:.12g}; "
        f"z={(volume_mean - SIEGEL_VOLUME_G2) / volume_se:.3g}"
    )
    print(
        "  marginal proposal-bin counts="
        f"{min(summary.marginal_bin_count_min for summary in summaries)}.."
        f"{max(summary.marginal_bin_count_max for summary in summaries)}"
    )
    print(f"  wrote {nodes_path}")
    print(f"  wrote {replicates_path}")
    print(f"  wrote {summary_path}")


PHYSICAL_MIXTURE_SAMPLING_SCHEME = "scrambled_sobol_physical_mixture"


DEFAULT_OUTPUT = Path(
    str(package_path('plumbing/results/genus2_c1_moduli_mc/physical_mixture_R8_C4_M64'))
)


@dataclass(frozen=True)
class PhysicalProposalComponent:
    """One normalized exponential density in ``(t1,t3)``."""

    name: str
    rate_t1: float
    rate_t3: float

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("component name must be nonempty")
        if not math.isfinite(self.rate_t1) or self.rate_t1 <= 0.0:
            raise ValueError("rate_t1 must be positive and finite")
        if not math.isfinite(self.rate_t3) or self.rate_t3 <= 0.0:
            raise ValueError("rate_t3 must be positive and finite")


DEFAULT_COMPONENTS = (
    PhysicalProposalComponent("bulk", 3.0, 2.0),
    PhysicalProposalComponent("one_handle_cusp", 3.0, 0.5),
    PhysicalProposalComponent("common_scale_cusp", 0.75, 2.0),
    PhysicalProposalComponent("double_cusp", 0.75, 0.5),
)


@dataclass(frozen=True)
class PhysicalMixtureReplicateSummary:
    replicate: int
    power: int
    component_count: int
    proposal_count: int
    domain_count: int
    domain_fraction: float
    invariant_volume_control: float
    invariant_volume_relative_error: float
    maximum_t1: float
    maximum_t3: float
    minimum_log_mixture_density: float
    maximum_log_physical_weight: float
    maximum_component_discrepancy: float
    component_domain_counts: tuple[int, ...]


@dataclass(frozen=True)
class PhysicalMixtureIntegralEstimate:
    replicate_count: int
    cft_node_count: int
    replicate_estimates: tuple[float, ...]
    estimate: float
    scramble_standard_error: float


def _validate_components(
    components: Sequence[PhysicalProposalComponent],
) -> tuple[PhysicalProposalComponent, ...]:
    values = tuple(components)
    if not values:
        raise ValueError("at least one proposal component is required")
    if len({component.name for component in values}) != len(values):
        raise ValueError("proposal component names must be unique")
    return values


def equal_mixture_log_density(
    t1: np.ndarray,
    t3: np.ndarray,
    components: Sequence[PhysicalProposalComponent] = DEFAULT_COMPONENTS,
) -> np.ndarray:
    """Return ``log p_mix(t1,t3)`` for the equal normalized mixture."""

    mixture = _validate_components(components)
    first = np.asarray(t1, dtype=np.float64)
    third = np.asarray(t3, dtype=np.float64)
    if first.shape != third.shape:
        raise ValueError("t1 and t3 must have the same shape")
    if np.any(first < 0.0) or np.any(third < 0.0):
        raise ValueError("Minkowski cusp coordinates must be nonnegative")
    terms = np.stack(
        [
            math.log(component.rate_t1 * component.rate_t3)
            - component.rate_t1 * first
            - component.rate_t3 * third
            for component in mixture
        ],
        axis=0,
    )
    return logsumexp(terms, axis=0) - math.log(len(mixture))


def physical_mixture_proposals_from_unit_cube(
    points: np.ndarray,
    *,
    component: PhysicalProposalComponent,
    components: Sequence[PhysicalProposalComponent] = DEFAULT_COMPONENTS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Map unit points through one component and return balance weights.

    Returns ``(omega, physical_weight, coordinates, log_mix, log_weight)``.
    The density is with respect to ``dX dt1 dt3 dr`` and the weight converts
    it to ``d^3 X d^3 Y``.
    """

    mixture = _validate_components(components)
    if component not in mixture:
        raise ValueError("selected component is absent from the mixture")
    unit = np.asarray(points, dtype=np.float64)
    if unit.ndim != 2 or unit.shape[1] != 6 or unit.shape[0] == 0:
        raise ValueError(f"points must have nonempty shape (n,6), got {unit.shape}")
    if not np.all(np.isfinite(unit)) or np.any(unit < 0.0) or np.any(unit >= 1.0):
        raise ValueError("unit-cube points must be finite and lie in [0,1)")

    x = unit[:, :3] - 0.5
    t1 = -np.log1p(-unit[:, 3]) / component.rate_t1
    t3 = -np.log1p(-unit[:, 4]) / component.rate_t3
    r = unit[:, 5]
    y1 = SQRT3_OVER_2 * np.exp(t1)
    y3 = y1 * np.exp(t3)
    y2 = 0.5 * r * y1

    omega = np.empty((unit.shape[0], 2, 2), dtype=np.complex128)
    omega[:, 0, 0] = x[:, 0] + 1.0j * y1
    omega[:, 0, 1] = x[:, 1] + 1.0j * y2
    omega[:, 1, 0] = omega[:, 0, 1]
    omega[:, 1, 1] = x[:, 2] + 1.0j * y3

    log_mix = equal_mixture_log_density(t1, t3, mixture)
    log_jacobian = (
        math.log(0.5 * SQRT3_OVER_2**3) + 3.0 * t1 + t3
    )
    log_weight = log_jacobian - log_mix
    if np.any(log_weight >= math.log(np.finfo(np.float64).max)):
        raise OverflowError("physical mixture weight exceeds float64 range")
    physical_weight = np.exp(log_weight)
    coordinates = np.column_stack((t1, t3, r))
    return omega, physical_weight, coordinates, log_mix, log_weight


def _mean_and_scramble_se(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 2 or not np.all(np.isfinite(array)):
        raise ValueError("need at least two finite replicate values")
    return float(np.mean(array)), float(np.std(array, ddof=1) / math.sqrt(array.size))


def generate_physical_mixture_replicate(
    *,
    replicate: int,
    power: int,
    base_seed: int,
    components: Sequence[PhysicalProposalComponent] = DEFAULT_COMPONENTS,
) -> tuple[list[dict[str, object]], PhysicalMixtureReplicateSummary]:
    """Generate one complete independently scrambled mixture replicate."""

    mixture = _validate_components(components)
    if power < 1:
        raise ValueError("power must be positive")
    per_component_count = 2**int(power)
    total_count = len(mixture) * per_component_count
    component_payloads: list[tuple[object, ...]] = []
    discrepancies: list[float] = []
    component_domain_counts: list[int] = []
    invariant_control_sum = 0.0

    for component_index, component in enumerate(mixture):
        seed = int(base_seed) + int(replicate) * len(mixture) + component_index
        engine = qmc.Sobol(d=6, scramble=True, seed=seed)
        points = engine.random_base2(m=int(power))
        omega, weight, coordinates, log_mix, log_weight = (
            physical_mixture_proposals_from_unit_cube(
                points,
                component=component,
                components=mixture,
            )
        )
        domain = np.asarray(in_gottschling_domain(omega), dtype=bool)
        margins = np.asarray(gottschling_min_margin(omega), dtype=np.float64)
        determinant = np.linalg.det(omega.imag)
        invariant_control_sum += float(np.sum(weight[domain] / determinant[domain] ** 3))
        discrepancies.append(float(qmc.discrepancy(points)))
        component_domain_counts.append(int(np.sum(domain)))
        component_payloads.append(
            (
                component,
                seed,
                points,
                omega,
                weight,
                coordinates,
                log_mix,
                log_weight,
                domain,
                margins,
            )
        )

    domain_count = sum(component_domain_counts)
    rows: list[dict[str, object]] = []
    maximum_t1 = 0.0
    maximum_t3 = 0.0
    minimum_log_mix = math.inf
    maximum_log_weight = -math.inf
    for component_index, payload in enumerate(component_payloads):
        (
            component,
            seed,
            points,
            omega,
            weight,
            coordinates,
            log_mix,
            log_weight,
            domain,
            margins,
        ) = payload
        maximum_t1 = max(maximum_t1, float(np.max(coordinates[:, 0])))
        maximum_t3 = max(maximum_t3, float(np.max(coordinates[:, 1])))
        minimum_log_mix = min(minimum_log_mix, float(np.min(log_mix)))
        maximum_log_weight = max(maximum_log_weight, float(np.max(log_weight)))
        for proposal_index in np.flatnonzero(domain):
            value = omega[proposal_index]
            physical_weight = float(weight[proposal_index])
            rows.append(
                {
                    "sampling_scheme": PHYSICAL_MIXTURE_SAMPLING_SCHEME,
                    "rqmc_node_id": (
                        f"r{int(replicate):03d}-c{component_index:02d}-"
                        f"p{int(proposal_index):08d}"
                    ),
                    "rqmc_replicate": int(replicate),
                    "rqmc_scramble_seed": seed,
                    "rqmc_power": int(power),
                    "rqmc_proposal_count": total_count,
                    "rqmc_component_proposal_count": per_component_count,
                    "rqmc_proposal_index": int(proposal_index),
                    "rqmc_domain_count": domain_count,
                    "rqmc_component_count": len(mixture),
                    "rqmc_component_index": component_index,
                    "rqmc_component_name": component.name,
                    "rqmc_component_rate_t1": component.rate_t1,
                    "rqmc_component_rate_t3": component.rate_t3,
                    "rqmc_component_mixture_fraction": 1.0 / len(mixture),
                    "rqmc_component_domain_count": component_domain_counts[component_index],
                    "rqmc_physical_measure_weight": physical_weight,
                    "rqmc_log_physical_measure_weight": float(log_weight[proposal_index]),
                    "rqmc_log_mixture_density": float(log_mix[proposal_index]),
                    GENUS2_FUNDAMENTAL_DOMAIN_WEIGHT_FIELD: (
                        GENUS2_FUNDAMENTAL_DOMAIN_WEIGHT
                        * physical_weight
                        / total_count
                    ),
                    "rqmc_kernel_det_im_power": 0,
                    "rqmc_u_x11": float(points[proposal_index, 0]),
                    "rqmc_u_x12": float(points[proposal_index, 1]),
                    "rqmc_u_x22": float(points[proposal_index, 2]),
                    "rqmc_u_t1": float(points[proposal_index, 3]),
                    "rqmc_u_t3": float(points[proposal_index, 4]),
                    "rqmc_u_r": float(points[proposal_index, 5]),
                    "rqmc_t1": float(coordinates[proposal_index, 0]),
                    "rqmc_t3": float(coordinates[proposal_index, 1]),
                    "rqmc_r": float(coordinates[proposal_index, 2]),
                    "gottschling_margin": float(margins[proposal_index]),
                    "det_im_omega": float(np.linalg.det(value.imag)),
                    "x11": float(value[0, 0].real),
                    "x12": float(value[0, 1].real),
                    "x22": float(value[1, 1].real),
                    "y11": float(value[0, 0].imag),
                    "y12": float(value[0, 1].imag),
                    "y22": float(value[1, 1].imag),
                }
            )

    invariant_control = invariant_control_sum / total_count
    summary = PhysicalMixtureReplicateSummary(
        replicate=int(replicate),
        power=int(power),
        component_count=len(mixture),
        proposal_count=total_count,
        domain_count=domain_count,
        domain_fraction=domain_count / total_count,
        invariant_volume_control=invariant_control,
        invariant_volume_relative_error=invariant_control / SIEGEL_VOLUME_G2 - 1.0,
        maximum_t1=maximum_t1,
        maximum_t3=maximum_t3,
        minimum_log_mixture_density=minimum_log_mix,
        maximum_log_physical_weight=maximum_log_weight,
        maximum_component_discrepancy=max(discrepancies),
        component_domain_counts=tuple(component_domain_counts),
    )
    return rows, summary


def generate_physical_mixture_design(
    *,
    replicate_count: int,
    power: int,
    base_seed: int,
    components: Sequence[PhysicalProposalComponent] = DEFAULT_COMPONENTS,
) -> tuple[list[dict[str, object]], list[PhysicalMixtureReplicateSummary]]:
    """Generate all in-domain nodes for independent mixture scrambles."""

    if replicate_count < 2:
        raise ValueError("at least two replicates are required for an error estimate")
    rows: list[dict[str, object]] = []
    summaries: list[PhysicalMixtureReplicateSummary] = []
    sample_index = 0
    for replicate in range(int(replicate_count)):
        replicate_rows, summary = generate_physical_mixture_replicate(
            replicate=replicate,
            power=power,
            base_seed=base_seed,
            components=components,
        )
        for row in replicate_rows:
            row["sample_index"] = sample_index
            sample_index += 1
        rows.extend(replicate_rows)
        summaries.append(summary)
    return rows, summaries


def estimate_physical_mixture_integral(
    rows: Sequence[dict[str, object]],
    kernel_values: Sequence[float],
) -> PhysicalMixtureIntegralEstimate:
    """Assemble the full fundamental-domain estimate by scramble."""

    if len(rows) != len(kernel_values) or not rows:
        raise ValueError("rows and kernel_values must have the same nonzero length")
    values = np.asarray(kernel_values, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("kernel values must be finite")
    if any(
        str(row.get("sampling_scheme", "")) != PHYSICAL_MIXTURE_SAMPLING_SCHEME
        for row in rows
    ):
        raise ValueError("rows do not use the physical-mixture sampling scheme")

    estimates: list[float] = []
    replicates = sorted({int(row["rqmc_replicate"]) for row in rows})
    for replicate in replicates:
        indices = [
            index
            for index, row in enumerate(rows)
            if int(row["rqmc_replicate"]) == replicate
        ]
        expected = int(rows[indices[0]]["rqmc_domain_count"])
        if len(indices) != expected:
            raise ValueError(
                f"replicate {replicate} is incomplete: {len(indices)} of "
                f"{expected} domain nodes"
            )
        if any(int(rows[index]["rqmc_domain_count"]) != expected for index in indices):
            raise ValueError(f"replicate {replicate} mixes domain counts")
        weights = np.asarray(
            [genus2_fundamental_domain_node_weight(rows[index]) for index in indices]
        )
        estimates.append(float(np.sum(weights * values[indices])))

    mean, standard_error = _mean_and_scramble_se(estimates)
    return PhysicalMixtureIntegralEstimate(
        replicate_count=len(replicates),
        cft_node_count=len(rows),
        replicate_estimates=tuple(estimates),
        estimate=mean,
        scramble_standard_error=standard_error,
    )


def physical_mixture_contribution_diagnostics(
    rows: Sequence[dict[str, object]],
    kernel_values: Sequence[float],
) -> dict[str, object]:
    """Resolve an evaluated integral by component and cusp-depth shells."""

    estimate = estimate_physical_mixture_integral(rows, kernel_values)
    values = np.asarray(kernel_values, dtype=np.float64)
    replicates = sorted({int(row["rqmc_replicate"]) for row in rows})
    contributions = np.asarray(
        [
            genus2_fundamental_domain_node_weight(row) * values[index]
            for index, row in enumerate(rows)
        ],
        dtype=np.float64,
    )

    def grouped_summary(labels: Sequence[object]) -> list[dict[str, object]]:
        unique = sorted(set(labels))
        output: list[dict[str, object]] = []
        for label in unique:
            replicate_values = []
            for replicate in replicates:
                replicate_values.append(
                    float(
                        np.sum(
                            [
                                contributions[index]
                                for index, row in enumerate(rows)
                                if int(row["rqmc_replicate"]) == replicate
                                and labels[index] == label
                            ]
                        )
                    )
                )
            mean, standard_error = _mean_and_scramble_se(replicate_values)
            output.append(
                {
                    "label": label,
                    "estimate": mean,
                    "standard_error": standard_error,
                    "fraction_of_total": (
                        mean / estimate.estimate if estimate.estimate != 0.0 else math.nan
                    ),
                    "replicate_estimates": replicate_values,
                }
            )
        return output

    component_labels = [str(row["rqmc_component_name"]) for row in rows]
    t1_shells = [int(math.floor(float(row["rqmc_t1"]))) for row in rows]
    t3_shells = [int(math.floor(float(row["rqmc_t3"]))) for row in rows]
    replicate_concentration: list[dict[str, float]] = []
    for replicate in replicates:
        selected = np.asarray(
            [
                abs(contributions[index])
                for index, row in enumerate(rows)
                if int(row["rqmc_replicate"]) == replicate
            ],
            dtype=np.float64,
        )
        total_abs = float(np.sum(selected))
        squared = float(np.sum(selected**2))
        replicate_concentration.append(
            {
                "replicate": float(replicate),
                "absolute_contribution_ess": (
                    0.0 if squared == 0.0 else total_abs * total_abs / squared
                ),
                "largest_absolute_node_fraction": (
                    0.0 if total_abs == 0.0 else float(np.max(selected) / total_abs)
                ),
            }
        )

    tail_rows: list[dict[str, object]] = []
    for coordinate in ("rqmc_t1", "rqmc_t3"):
        for threshold in (1.0, 2.0, 4.0, 8.0, 16.0):
            replicate_values = []
            for replicate in replicates:
                replicate_values.append(
                    float(
                        np.sum(
                            [
                                contributions[index]
                                for index, row in enumerate(rows)
                                if int(row["rqmc_replicate"]) == replicate
                                and float(row[coordinate]) >= threshold
                            ]
                        )
                    )
                )
            mean, standard_error = _mean_and_scramble_se(replicate_values)
            tail_rows.append(
                {
                    "coordinate": coordinate.removeprefix("rqmc_"),
                    "threshold": threshold,
                    "estimate": mean,
                    "standard_error": standard_error,
                    "fraction_of_total": (
                        mean / estimate.estimate if estimate.estimate != 0.0 else math.nan
                    ),
                    "replicate_estimates": replicate_values,
                }
            )

    return {
        "component_contributions": grouped_summary(component_labels),
        "t1_unit_shell_contributions": grouped_summary(t1_shells),
        "t3_unit_shell_contributions": grouped_summary(t3_shells),
        "cumulative_tail_contributions": tail_rows,
        "replicate_concentration": replicate_concentration,
        "interpretation": (
            "Stable estimates require the deepest occupied shells and the slow-tail "
            "components to be small within scramble errors, and no replicate to be "
            "dominated by a few nodes.  These are diagnostics, not a proof of finite variance."
        ),
    }


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def run(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate a physical-measure genus-two RQMC mixture design."
    )
    parser.add_argument("--replicates", type=int, default=8)
    parser.add_argument("--power", type=int, default=6)
    parser.add_argument("--base-seed", type=int, default=20260719)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(list(argv) if argv is not None else None)

    rows, summaries = generate_physical_mixture_design(
        replicate_count=args.replicates,
        power=args.power,
        base_seed=args.base_seed,
    )
    volume_controls = [summary.invariant_volume_control for summary in summaries]
    volume_mean, volume_se = _mean_and_scramble_se(volume_controls)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    nodes_path = args.out_dir / "domain_nodes.csv"
    replicates_path = args.out_dir / "replicate_summary.csv"
    summary_path = args.out_dir / "summary.json"
    _write_csv(nodes_path, rows)
    _write_csv(replicates_path, [asdict(summary) for summary in summaries])
    payload = {
        "scope": (
            "Exact physical-period-measure RQMC mixture on the complete "
            "Gottschling domain; only in-domain nodes require CFT evaluation."
        ),
        "sampling_scheme": PHYSICAL_MIXTURE_SAMPLING_SCHEME,
        "estimator": (
            "F2/g_s^2=(1/2)*mean_over_mixture_proposals["
            "1_F2*(J_Y/p_mix)*K2_c1]"
        ),
        "kernel_det_im_power": 0,
        "replicate_count": args.replicates,
        "power_per_component": args.power,
        "proposal_count_per_component": 2**args.power,
        "proposal_count_per_replicate": len(DEFAULT_COMPONENTS) * 2**args.power,
        "domain_cft_node_count": len(rows),
        "base_seed": args.base_seed,
        "components": [asdict(component) for component in DEFAULT_COMPONENTS],
        "exact_invariant_volume_control": SIEGEL_VOLUME_G2,
        "invariant_volume_control_estimate": volume_mean,
        "invariant_volume_control_standard_error": volume_se,
        "invariant_volume_control_z_score": (
            (volume_mean - SIEGEL_VOLUME_G2) / volume_se
            if volume_se > 0.0
            else math.nan
        ),
        "replicates": [asdict(summary) for summary in summaries],
        "notes": [
            "The balance-heuristic mixture is exact because every component has the same allocation.",
            "The local observable is K2_c1; no det(Im Omega)^3 factor is applied to it.",
            "The cheap det(Im Omega)^(-3) control must reproduce the exact invariant Siegel volume.",
            "Independent complete scrambles, not nodes, determine the sampling error.",
            "No failed in-domain CFT node may be dropped from a replicate.",
            "Finite variance is checked empirically from shell and component diagnostics; it is not assumed from integrability alone.",
        ],
    }
    temporary = summary_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(summary_path)

    print("Genus-two physical-measure randomized-QMC design")
    print(
        f"  replicates={args.replicates}, components={len(DEFAULT_COMPONENTS)}, "
        f"proposals/replicate={len(DEFAULT_COMPONENTS) * 2**args.power}, "
        f"CFT domain nodes={len(rows)}"
    )
    print(
        f"  invariant-volume control={volume_mean:.12g} +/- {volume_se:.3g}; "
        f"exact={SIEGEL_VOLUME_G2:.12g}"
    )
    print(f"  wrote {nodes_path}")
    print(f"  wrote {replicates_path}")
    print(f"  wrote {summary_path}")


def main():
    """Choose the saved proposal measure before forwarding its original options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--measure', choices=('physical', 'invariant'), default='physical')
    args, rest = parser.parse_known_args()
    return (run if args.measure == 'physical' else invariant_run)(rest)

if __name__ == "__main__":
    main()
