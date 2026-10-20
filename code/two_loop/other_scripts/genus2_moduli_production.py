"""Pilot Monte Carlo integration of the genus-two c=1 string density.

The input period matrices are iid samples from the invariant Siegel measure
on the coarse Gottschling domain,

    p(Omega) d^6 Omega = d^6 Omega / (V_2 det(Im Omega)^3).

For the BRY v7 ``g_s^2``-stripped c=1 kernel, with ``mu^-1=2*pi*g_s``,
``K_2=(1/2)_Aut*16*pi^4 I_2^physical`` at ``alpha'=1``, the estimator is

    F_2/g_s^2 = V_2 E_p[det(Im Omega)^3 K_2(Omega)].

The automorphism half is included once in K_2, leaving geometric domain
weights unchanged. Independent absolute ghost/sewing certification is separate.
The inherited Xi ledger includes the string-note identity
``(-i)(-1)(-8i)/(8 pi)=1/pi``, the residual compact
``c=1`` sphere-topology factor ``4*pi``, and the completed fixed-puncture BRST
bridge ``(2*pi)^4``; it does not include ``g_s^2``.  Each node is evaluated in
an atlas-selected theta or glasses plumbing frame.  The full noncompact
scalar plumbing partition and the Liouville partition are paired in that same
frame, so their Weyl factor cancels in Z_L / Z_X^25.  The dimensionless
plumbing scalar is converted together with the critical 26-boson seed and the
compact target zero mode to Xi's ``dk_I/(2 pi)`` convention.  At
``alpha'=1`` this correlated conversion multiplies the former dimensionless
kernel by ``1/(2 pi)``.

This driver records low/high CFT and scalar truncations independently from the
Monte Carlo standard error.  A small run is a pilot, not a precision claim:
degeneration strata can generate heavy-tailed contributions.

The driver also accepts the direct-importance scrambled-Sobol designs emitted
by ``genus2_moduli_rqmc.py``.  In that case every in-domain node carries the
explicit saved local weight (with historical weights reconstructed), and independent complete scrambles, rather than individual
nodes, determine the sampling error.

The physical-measure mixture emitted by
``genus2_moduli_physical_mixture_rqmc.py`` is also accepted.  Its node weight
already converts the mixture density to ``d^3 X d^3 Y``.  For that design the
local observable is ``K_2`` itself: no ``det(Im Omega)^3`` multiplier is used.

In particular, det(Im Omega)^3 is not part of the local ghost-matter density.
It is the Radon--Nikodym factor converting the invariant proposal measure back
to the period-coordinate volume d^3 X d^3 Y.  The local CFT constant is fixed
independently by lower-genus sewing; det(Im Omega)^3 has no role in it."""
from __future__ import annotations
from c1_paths import package_path
import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence
from genus2_c1_string_integrand import COMPACT_THETA_IMPLEMENTATION, DIMENSIONLESS_SEWING_SCALAR_NORMALIZATION, XI_PHYSICAL_MOMENTUM_SCALAR_NORMALIZATION
from genus2_integrand_normalization import FULL_CFT_FACTORIZATION_CERTIFIED, GENUS2_FUNDAMENTAL_DOMAIN_WEIGHT, CURRENT_INTEGRATION_KERNEL_CONVENTION, c1_genus2_topology_correction, c1_bry_genus2_kernel_multiplier, genus2_automorphism_metadata, critical_prefactor_ratio_to_dhp, xi_full_replacement_over_dimensionless
from genus2_holomorphic_period_table import SchottkyValidityEnvelope
from genus2_moduli_sampling import PHYSICAL_MIXTURE_SAMPLING_SCHEME
from genus2_moduli_sampling import physical_mixture_contribution_diagnostics
from genus2_period_table import Genus2PeriodMapTable
from genus2_siegel_fundamental_domain import SIEGEL_VOLUME_G2
from genus2_moduli_estimation import canonicalize_string_note_kernel_row, sampling_scheme_for_rows, summarize_rqmc_rows, estimate_from_transformed_values
from genus2_node_evaluation import evaluate_node

#!/usr/bin/env python3







USABLE_CHART_STATUSES = {"reference-q-envelope", "requires-recursion-order-study"}
RQMC_SAMPLING_SCHEME = "scrambled_sobol_minkowski_importance"
RQMC_SAMPLING_SCHEMES = {
    RQMC_SAMPLING_SCHEME,
    PHYSICAL_MIXTURE_SAMPLING_SCHEME,
}
LEGACY_UNIT_MUMFORD_KERNEL_CONVENTION = "unit-mumford-residue-no-string-note-prefactor"















# Neutral spelling for new callers.
canonicalize_amplitude_kernel_row = canonicalize_string_note_kernel_row




















































def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
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


def _resume_node_key(row: dict[str, object]) -> str:
    node_id = str(row.get("rqmc_node_id", "")).strip()
    if node_id:
        return f"rqmc:{node_id}"
    sample_index = str(row.get("sample_index", "")).strip()
    if sample_index:
        return f"sample:{sample_index}"
    raise ValueError("resume row lacks both rqmc_node_id and sample_index")


def _jsonable_argument(value: object) -> object:
    return str(value) if isinstance(value, Path) else value


def _resume_config(
    args: argparse.Namespace,
    selected: Sequence[dict[str, object]],
) -> dict[str, object]:
    return {
        "format_version": 1,
        "source_csv": str(args.source_csv.resolve()),
        "source_csv_sha256": hashlib.sha256(args.source_csv.read_bytes()).hexdigest(),
        "selected_node_keys": [_resume_node_key(row) for row in selected],
        "settings": {
            **{
                key: _jsonable_argument(value)
                for key, value in vars(args).items()
                if key not in {"source_csv", "out_dir", "resume"}
            },
            "compact_theta_implementation": COMPACT_THETA_IMPLEMENTATION,
        },
    }


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _load_resume_rows(
    *,
    csv_path: Path,
    config_path: Path,
    expected_config: dict[str, object],
    resume: bool,
) -> dict[str, dict[str, object]]:
    if not resume:
        _write_json_atomic(config_path, expected_config)
        return {}
    if csv_path.exists() and not config_path.exists():
        raise ValueError("cannot resume samples.csv without run_config.json")
    if config_path.exists():
        observed_config = json.loads(config_path.read_text())
        if observed_config != expected_config:
            raise ValueError("resume settings or source manifest do not match run_config.json")
    else:
        _write_json_atomic(config_path, expected_config)
    if not csv_path.exists():
        return {}

    expected_keys = set(expected_config["selected_node_keys"])
    rows_by_key: dict[str, dict[str, object]] = {}
    for row in csv.DictReader(csv_path.open()):
        key = _resume_node_key(row)
        if key not in expected_keys:
            raise ValueError(f"resume CSV contains unexpected node {key}")
        if key in rows_by_key:
            raise ValueError(f"resume CSV contains duplicate node {key}")
        rows_by_key[key] = dict(row)
    return rows_by_key


def run(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Pilot genus-two c=1 string Monte Carlo integral.")
    parser.add_argument(
        "--source-csv",
        type=Path,
        default=Path(str(package_path('plumbing/results/genus2_c1_moduli_mc/physical_mixture_R8_C4_M256/production_nodes.csv'))),
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--sample-count", type=int, default=8)
    parser.add_argument(
        "--rqmc-replicate",
        type=int,
        help="evaluate every in-domain node in one RQMC scramble",
    )
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--q-reference-max", type=float, default=0.16)
    parser.add_argument("--refine-above-q", type=float, default=0.16)
    parser.add_argument("--base-search-depth", type=int, default=3)
    parser.add_argument("--base-prefilter-count", type=int, default=2)
    parser.add_argument("--base-word-length", type=int, default=4)
    parser.add_argument("--base-period-tolerance", type=float, default=2.0e-5)
    parser.add_argument("--base-stability-tolerance", type=float, default=2.0e-5)
    parser.add_argument("--refined-search-depth", type=int, default=4)
    parser.add_argument("--refined-prefilter-count", type=int, default=6)
    parser.add_argument("--refined-word-length", type=int, default=6)
    parser.add_argument("--refined-period-tolerance", type=float, default=5.0e-6)
    parser.add_argument("--refined-stability-tolerance", type=float, default=5.0e-6)
    parser.add_argument(
        "--early-stop-competitive-score-ratio",
        type=float,
        default=1.02,
        help=(
            "certify one candidate from both theta and glasses before early "
            "stop when their best leading q scores are within this ratio"
        ),
    )
    parser.add_argument(
        "--table-first-maximum-word",
        type=int,
        default=5,
        help=(
            "maximum adaptive Schottky word used to certify a transported "
            "fundamental-table seed before falling back to the full atlas"
        ),
    )
    parser.add_argument(
        "--table-first-maximum-corrections",
        type=int,
        default=2,
        help="maximum analytic log-q corrections before full-atlas fallback",
    )
    parser.add_argument(
        "--period-validation-word-length",
        type=int,
        default=8,
        help="initial Schottky word order used by the hybrid period certificate",
    )
    parser.add_argument(
        "--period-validation-word-step",
        type=int,
        default=1,
        help="word-order step retained for period-certificate compatibility",
    )
    parser.add_argument("--period-validation-tolerance", type=float, default=1.0e-6)
    parser.add_argument(
        "--period-reinverse-validation-word-length",
        type=int,
        default=10,
        help="maximum Schottky validation order for cusp re-inversion",
    )
    parser.add_argument("--period-reinverse-max-nfev", type=int, default=80)
    parser.add_argument(
        "--schottky-validity-envelope",
        type=Path,
        help=(
            "optional CSV of Schottky cells calibrated against the holomorphic-form "
            "reference table; this strengthens the adaptive hybrid certificate"
        ),
    )
    parser.add_argument(
        "--period-table-index",
        type=Path,
        help=(
            "optional portable q-to-Omega table index used only to seed the "
            "certified Omega-to-q atlas inversion"
        ),
    )
    parser.add_argument(
        "--period-table-csv",
        type=Path,
        help="optional canonical table CSV used to verify the portable index hash",
    )
    parser.add_argument(
        "--period-recovery-cache",
        type=Path,
        help=(
            "assembled period_recovery.csv or a completed combined_samples.csv; "
            "accepted q and exact markings are reused without repeating "
            "Omega-to-q inversion"
        ),
    )
    parser.add_argument("--block-order-low", type=int, default=2)
    parser.add_argument("--block-order-high", type=int, default=4)
    parser.add_argument("--quadrature-order-low", type=int, default=3)
    parser.add_argument("--quadrature-order-high", type=int, default=4)
    parser.add_argument(
        "--quadrature-scheme",
        choices=("uniform", "edge-scaled", "primary-gaussian"),
        default="primary-gaussian",
    )
    parser.add_argument("--dps", type=int, default=24)
    parser.add_argument(
        "--vacuum-word-length",
        type=int,
        default=10,
        help="safety cap for adaptive CCY vacuum-seed word shells",
    )
    parser.add_argument(
        "--vacuum-word-tail-tolerance",
        type=float,
        default=1.0e-12,
        help="target absolute chiral-log tail for the CCY vacuum seed",
    )
    parser.add_argument(
        "--disable-vacuum-word-tail-check",
        action="store_true",
        help=(
            "evaluate through the requested vacuum word cap without applying "
            "the empirical primitive-word tail gate; intended only for direct "
            "fixed-cap stability comparisons"
        ),
    )
    parser.add_argument("--vacuum-minimum-word-length", type=int, default=5)
    parser.add_argument("--vacuum-oscillator-level", type=int, default=20)
    parser.add_argument("--global-block-tolerance", type=float, default=1.0e-13)
    parser.add_argument("--global-block-max-outer-order", type=int, default=64)
    parser.add_argument("--scalar-word-low", type=int, default=4)
    parser.add_argument(
        "--scalar-word-high",
        type=int,
        default=12,
        help="ordinary-q safety cap for the adaptive scalar primitive-word product",
    )
    parser.add_argument(
        "--scalar-word-high-large-q",
        type=int,
        default=14,
        help="higher adaptive word cap used when max|q_i| crosses the large-q threshold",
    )
    parser.add_argument(
        "--scalar-large-q-threshold",
        type=float,
        default=0.25,
        help="max|q_i| threshold for using --scalar-word-high-large-q",
    )
    parser.add_argument(
        "--scalar-word-tail-tolerance",
        type=float,
        default=2.0e-9,
        help="target absolute chiral-log tail before forming the nonchiral scalar",
    )
    parser.add_argument(
        "--scalar-powered-relative-tolerance",
        type=float,
        default=1.1e-7,
        help="required direct relative stability after the scalar power 25",
    )
    parser.add_argument("--scalar-word-increment", type=int, default=2)
    parser.add_argument("--scalar-confirmation-steps", type=int, default=2)
    parser.add_argument("--scalar-minimum-word-length", type=int, default=5)
    parser.add_argument("--scalar-max-mode", type=int, default=80)
    parser.add_argument("--scalar-tolerance", type=float, default=1.0e-14)
    parser.add_argument("--lattice-tolerance", type=float, default=1.0e-12)
    parser.add_argument("--theta-tolerance", type=float, default=1.0e-12)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(str(package_path('plumbing/results/genus2_c1_moduli_mc/pilot_R1_N8'))),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "reuse successful nodes from samples.csv after verifying the source "
            "manifest and every numerical setting against run_config.json"
        ),
    )
    parser.add_argument(
        "--raise-on-error",
        action="store_true",
        help="re-raise a node exception with a traceback for targeted diagnostics",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.scalar_word_increment <= 0:
        raise ValueError("--scalar-word-increment must be positive")
    if args.scalar_confirmation_steps <= 0:
        raise ValueError("--scalar-confirmation-steps must be positive")
    for scalar_cap in (args.scalar_word_high, args.scalar_word_high_large_q):
        if scalar_cap < args.scalar_word_low:
            raise ValueError("scalar high word caps must not be below the low cap")
        if scalar_cap == args.scalar_word_low:
            # A fixed-cap production or continuation run remains a supported
            # auditable mode and does not need successive adaptive checkpoints.
            continue
        if (scalar_cap - args.scalar_word_low) % args.scalar_word_increment:
            raise ValueError("scalar high word caps must lie on the fixed-cap scan")
        if (
            scalar_cap - args.scalar_word_low
            < args.scalar_confirmation_steps * args.scalar_word_increment
        ):
            raise ValueError(
                "scalar word caps do not provide enough successive direct comparisons"
            )
    schottky_envelope = (
        None
        if args.schottky_validity_envelope is None
        else SchottkyValidityEnvelope.from_csv(args.schottky_validity_envelope)
    )
    if args.period_table_csv is not None and args.period_table_index is None:
        raise ValueError("--period-table-csv requires --period-table-index")
    period_table = (
        None
        if args.period_table_index is None
        else Genus2PeriodMapTable.from_portable_index(
            args.period_table_index,
            verify_table_path=args.period_table_csv,
        )
    )
    if period_table is not None and not period_table.has_fundamental_index:
        raise ValueError(
            "--period-table-index must use schema v3 with fundamental-domain "
            "Omega coordinates and exact Sp(4,Z) markings"
        )
    period_recovery_by_index: dict[int, dict[str, str]] | None = None
    if args.period_recovery_cache is not None:
        recovery_rows = list(csv.DictReader(args.period_recovery_cache.open()))
        period_recovery_by_index = {}
        for recovery_row in recovery_rows:
            recovery_index = int(recovery_row["sample_index"])
            if recovery_index in period_recovery_by_index:
                raise ValueError(
                    f"period-recovery cache duplicates sample {recovery_index}"
                )
            period_recovery_by_index[recovery_index] = recovery_row

    source_rows = list(csv.DictReader(args.source_csv.open()))
    if args.rqmc_replicate is not None:
        selected = [
            row
            for row in source_rows
            if row.get("rqmc_replicate") != ""
            and int(row["rqmc_replicate"]) == args.rqmc_replicate
        ]
        if not selected:
            raise ValueError(f"RQMC replicate {args.rqmc_replicate} is absent")
        sample_range: list[int] | None = None
    else:
        if args.start_index < 0 or args.sample_count <= 0:
            raise ValueError("start-index must be nonnegative and sample-count positive")
        stop = args.start_index + args.sample_count
        if stop > len(source_rows):
            raise ValueError("requested sample range exceeds the source CSV")
        selected = source_rows[args.start_index:stop]
        sample_range = [args.start_index, stop]
    sampling_scheme = sampling_scheme_for_rows(selected)
    if args.rqmc_replicate is not None and sampling_scheme not in RQMC_SAMPLING_SCHEMES:
        raise ValueError("--rqmc-replicate requires scrambled-Sobol source rows")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "samples.csv"
    json_path = args.out_dir / "summary.json"
    config_path = args.out_dir / "run_config.json"
    config = _resume_config(args, selected)
    rows_by_key = _load_resume_rows(
        csv_path=csv_path,
        config_path=config_path,
        expected_config=config,
        resume=args.resume,
    )
    selected_keys = [_resume_node_key(row) for row in selected]
    completed_before = sum(
        str(row.get("status", "")) == "ok" for row in rows_by_key.values()
    )
    if completed_before:
        print(f"  resume: preserving {completed_before} successful nodes")
    for offset, source_row in enumerate(selected, start=1):
        key = _resume_node_key(source_row)
        previous = rows_by_key.get(key)
        if previous is not None and str(previous.get("status", "")) == "ok":
            print(f"  {offset}/{len(selected)} {key} status=resume")
            continue
        recovery_solution: Mapping[str, object] | None = None
        if period_recovery_by_index is not None:
            recovery_solution = period_recovery_by_index.get(
                int(source_row["sample_index"]),
                {
                    "sample_index": source_row["sample_index"],
                    "status": "missing",
                    "error": "sample is absent from the period-recovery cache",
                },
            )
        row = evaluate_node(
            source_row,
            args,
            schottky_envelope,
            period_table,
            recovery_solution,
        )
        rows_by_key[key] = row
        rows = [rows_by_key[item] for item in selected_keys if item in rows_by_key]
        _write_csv(csv_path, rows)
        if row["status"] == "ok":
            print(
                f"  {offset}/{len(selected)} index={row['sample_index']} "
                f"{row['topology']} q={float(row['q_max']):.4g} "
                f"g={float(row['transformed_integrand_high']):.4e} "
                f"delta={float(row['combined_relative_change']):+.2e} "
                f"time={float(row['runtime_seconds']):.1f}s"
            )
        else:
            print(
                f"  {offset}/{len(selected)} index={row['sample_index']} "
                f"FAILED: {row['error']}"
            )

    rows = [rows_by_key[key] for key in selected_keys]
    successful = [row for row in rows if row["status"] == "ok"]
    complete = len(successful) == len(rows)
    if sampling_scheme in RQMC_SAMPLING_SCHEMES:
        high_summary = summarize_rqmc_rows(rows, value_key="transformed_integrand_high")
        low_summary = summarize_rqmc_rows(rows, value_key="transformed_integrand_low")
    else:
        high_summary = (
            estimate_from_transformed_values(
                [float(row["transformed_integrand_high"]) for row in successful]
            )
            if successful
            else None
        )
        low_summary = (
            estimate_from_transformed_values(
                [float(row["transformed_integrand_low"]) for row in successful]
            )
            if successful
            else None
        )
    integration_complete = high_summary is not None and low_summary is not None
    if high_summary is not None:
        if sampling_scheme == PHYSICAL_MIXTURE_SAMPLING_SCHEME:
            high_summary["cusp_diagnostics"] = physical_mixture_contribution_diagnostics(
                rows,
                [float(row["transformed_integrand_high"]) for row in rows],
            )
        elif sampling_scheme != RQMC_SAMPLING_SCHEME:
            high_summary["interpretation"] = (
                "Unbiased for the requested iid sample only when complete=true. "
                "The standard error is Monte Carlo sampling error and excludes CFT truncation error."
            )
    discretization_change = (
        None
        if high_summary is None or low_summary is None
        else float(high_summary["estimate"]) / float(low_summary["estimate"]) - 1.0
    )
    node_discretization_changes = [
        float(row["combined_relative_change"])
        for row in successful
    ]
    node_discretization = (
        None
        if not node_discretization_changes
        else {
            "minimum_relative_change": min(node_discretization_changes),
            "maximum_relative_change": max(node_discretization_changes),
            "maximum_absolute_relative_change": max(
                abs(value) for value in node_discretization_changes
            ),
            "status": (
                "not established; the aggregate shift does not bound individual "
                "node truncation errors"
            ),
        }
    )
    payload = {
        "scope": (
            "Pilot integration of the string-note normalized, g_s^2-stripped "
            "c=1 genus-two kernel "
            "over one ordinary Gottschling fundamental domain."
        ),
        "source_csv": str(args.source_csv),
        "sample_range": sample_range,
        "requested_rqmc_replicate": args.rqmc_replicate,
        "requested_sample_count": len(rows),
        "successful_sample_count": len(successful),
        "complete": complete,
        "integration_complete": integration_complete,
        "sampling_scheme": sampling_scheme,
        "radius": args.radius,
        "siegel_volume": SIEGEL_VOLUME_G2,
        "fundamental_domain_weight": GENUS2_FUNDAMENTAL_DOMAIN_WEIGHT,
        "estimator": (
            "F2/g_s^2=mean[1_F2*(J_Y/p_mix)*K2_c1]"
            if sampling_scheme == PHYSICAL_MIXTURE_SAMPLING_SCHEME
            else (
                "F2/g_s^2=mean_over_all_proposals[1_F2*w*det(Im Omega)^3*K2_c1]"
                if sampling_scheme == RQMC_SAMPLING_SCHEME
                else "F2/g_s^2=Vol(F2)*mean[det(Im Omega)^3*K2_c1]"
            )
        ),
        "measure_identity": {
            "physical_volume_form": "|d^3 Omega|^2=d^3 Re(Omega) d^3 Im(Omega)",
            "local_period_form": "K2_c1 * d^3 Re(Omega) d^3 Im(Omega)",
            "unit_mumford_relation": "K2_c1=(2/pi)*I2_Xi_Mumford_residue",
            "alpha_prime": 1.0,
            "dimensionless_radius_definition": "r=R_phys/sqrt(alpha')",
            "scalar_input_normalization": DIMENSIONLESS_SEWING_SCALAR_NORMALIZATION,
            "scalar_integrand_normalization": XI_PHYSICAL_MOMENTUM_SCALAR_NORMALIZATION,
            "xi_scalar_relation": "Z_X^Xi=Z_X^p/(4*pi^2*alpha') at genus two",
            "compact_zero_mode": "2*pi*R_phys",
            "xi_full_replacement_over_dimensionless": (
                xi_full_replacement_over_dimensionless()
            ),
            "g_s_squared_stripped": True,
            "proposal_measure": (
                "four-component physical-measure Minkowski mixture"
                if sampling_scheme == PHYSICAL_MIXTURE_SAMPLING_SCHEME
                else (
                    "six-dimensional Minkowski direct-importance proposal"
                    if sampling_scheme == RQMC_SAMPLING_SCHEME
                    else (
                        "p(Omega)|d^3 Omega|^2=|d^3 Omega|^2/"
                        "(Vol(F2)*det(Im Omega)^3)"
                    )
                )
            ),
            "importance_weight": (
                "1_F2*(J_Y/p_mix) per proposal; local observable is K2_c1"
                if sampling_scheme == PHYSICAL_MIXTURE_SAMPLING_SCHEME
                else (
                    "1_F2*w*det(Im Omega)^3 per proposal"
                    if sampling_scheme == RQMC_SAMPLING_SCHEME
                    else "Vol(F2)*det(Im Omega)^3"
                )
            ),
            "global_modular_weight_application": (
                "none; integrate once over the Siegel fundamental domain"
            ),
            "warning": (
                "The physical-mixture design removes det(Im Omega)^3 from the local observable."
                if sampling_scheme == PHYSICAL_MIXTURE_SAMPLING_SCHEME
                else "det(Im Omega)^3 is proposal reweighting, not a ghost factor"
            ),
        },
        "integration_kernel_convention": CURRENT_INTEGRATION_KERNEL_CONVENTION,
        **genus2_automorphism_metadata(),
        "string_note_kernel_multiplier": (
            c1_bry_genus2_kernel_multiplier()
        ),
        "c1_sphere_topology_multiplier": c1_genus2_topology_correction(),
        "external_comparison_target": None,
        "modular_covariance_split": {
            "fundamental_domain": (
                "Igusa cusp form, det(Im Omega), all theta-function evaluations "
                "including the compact winding sum, and the coefficient of d^6 Omega"
            ),
            "plumbing_marking": (
                "the modular-invariant same-frame Z_L/Z_X^25 quotient"
            ),
            "guard": (
                "the saved exact Sp(4,Z) matrix must map the fundamental-domain "
                "period matrix to the plumbing period matrix"
            ),
        },
        "full_cft_factorization_certified": FULL_CFT_FACTORIZATION_CERTIFIED,
        "local_cft_normalization_crosscheck": {
            "repository_over_dhp_critical_prefactor": critical_prefactor_ratio_to_dhp(),
            "exact_prefactor_ratio": "(2*pi)^26",
            "scalar_loop_gaussian": (
                "coefficient one follows from delta-normalized momentum states "
                "and completeness measure dp"
            ),
            "interpretation": (
                "the DHP ratio is exactly the 26-scalar target-volume convention "
                "conversion already present at genus one"
            ),
        },
        "settings": {
            **{
                key: _jsonable_argument(value)
                for key, value in vars(args).items()
                if key not in {"source_csv", "out_dir"}
            },
            "compact_theta_implementation": COMPACT_THETA_IMPLEMENTATION,
        },
        "high_order_summary": high_summary,
        "low_order_summary": low_summary,
        "aggregate_discretization_relative_change": discretization_change,
        "node_discretization_diagnostic": node_discretization,
        "warnings": [
            "The tested q<=0.25 range is an empirical recursion benchmark, not a convergence theorem; larger-q nodes require a separate cutoff audit.",
            "A small design may have very large variance if degeneration strata dominate; inspect the shell and component diagnostics.",
            "The complete cusp is integrated with no IR subtraction; Liouville threshold stripping is an exactly restored numerical rescaling only.",
            "For RQMC input, at least two complete independent scrambles are required for a headline estimate.",
            "No failed node may be dropped from an unbiased estimate; complete=false invalidates the headline estimate.",
            "The normal confidence interval is diagnostic only for this small, potentially heavy-tailed pilot.",
            "The aggregate low/high shift can conceal much larger changes at individual nodes.",
            "No external matrix-model target or fitted normalization is used in this computation.",
        ],
    }

    _write_csv(csv_path, rows)
    _write_json_atomic(json_path, payload)
    print("Genus-two c=1 Monte Carlo pilot")
    print(
        f"  scheme={sampling_scheme}, node-complete={complete}, "
        f"integration-complete={integration_complete}, successful={len(successful)}/{len(rows)}"
    )
    if high_summary is not None:
        print(
            f"  F2/g_s^2 estimate={float(high_summary['estimate']):.12g} "
            f"+/- {float(high_summary['standard_error']):.3g}"
        )
        if sampling_scheme in RQMC_SAMPLING_SCHEMES:
            print(
                f"  independent complete scrambles={int(high_summary['replicate_count'])}"
            )
        if sampling_scheme == RQMC_SAMPLING_SCHEME:
            print(
                "  exact-volume-calibrated diagnostic="
                f"{float(high_summary['volume_calibrated_estimate']):.12g}"
            )
        elif sampling_scheme == "iid_invariant_domain":
            print(f"  effective sample size={float(high_summary['effective_sample_size']):.3f}")
            print(f"  largest sample fraction={float(high_summary['largest_sample_fraction']):.3f}")
        print(f"  low/high aggregate change={discretization_change:+.3e}")
    elif sampling_scheme in RQMC_SAMPLING_SCHEMES:
        print("  no headline estimate: selected rows do not contain two complete scrambles")
    print(f"  wrote {csv_path}")
    print(f"  wrote {json_path}")


if __name__ == "__main__":
    run()
