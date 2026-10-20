#!/usr/bin/env python3
"""Modular-covariance diagnostics for the genus-two Liouville plumbing result."""

from __future__ import annotations

from c1_paths import PACKAGE_ROOT, data_root, package_path, module_path

import argparse
import cmath
from dataclasses import dataclass
from typing import Iterable

import numpy as np

try:
    from liouville_genus2 import (
        LiouvilleGenus2PairOfToriResult,
        format_complex,
        liouville_genus2_pair_of_tori,
        parse_complex,
    )
    from plumbing_algorithms import (
        schottky_glasses_period_matrix,
        schottky_health,
        solve_glasses_inverse_from_omega,
    )
except ImportError:  # pragma: no cover - supports package-style execution
    from liouville_genus2 import (
        LiouvilleGenus2PairOfToriResult,
        format_complex,
        liouville_genus2_pair_of_tori,
        parse_complex,
    )
    from plumbing_algorithms import (
        schottky_glasses_period_matrix,
        schottky_health,
        solve_glasses_inverse_from_omega,
    )


def _matrix(values: Iterable[Iterable[int]]) -> np.ndarray:
    return np.asarray([[int(entry) for entry in row] for row in values], dtype=np.int64)


@dataclass(frozen=True)
class SymplecticTransform:
    name: str
    a: np.ndarray
    b: np.ndarray
    c: np.ndarray
    d: np.ndarray

    def __post_init__(self) -> None:
        for block_name in ("a", "b", "c", "d"):
            block = np.asarray(getattr(self, block_name))
            if block.shape != (2, 2):
                raise ValueError(f"{block_name} block must be 2x2")
        if not self.is_symplectic():
            raise ValueError(f"{self.name!r} is not symplectic")

    def is_symplectic(self) -> bool:
        lhs = self.a.T @ self.d - self.c.T @ self.b
        return (
            np.array_equal(lhs, np.eye(2, dtype=np.int64))
            and np.array_equal(self.a.T @ self.c, self.c.T @ self.a)
            and np.array_equal(self.b.T @ self.d, self.d.T @ self.b)
        )

    def transform_omega(self, omega: np.ndarray) -> np.ndarray:
        omega = np.asarray(omega, dtype=np.complex128)
        if omega.shape != (2, 2):
            raise ValueError("omega must be a 2x2 period matrix")
        a = self.a.astype(np.complex128)
        b = self.b.astype(np.complex128)
        c = self.c.astype(np.complex128)
        d = self.d.astype(np.complex128)
        return (a @ omega + b) @ np.linalg.inv(c @ omega + d)

    def det_factor(self, omega: np.ndarray) -> complex:
        omega = np.asarray(omega, dtype=np.complex128)
        return complex(np.linalg.det(self.c.astype(np.complex128) @ omega + self.d.astype(np.complex128)))

    def inverse(self) -> "SymplecticTransform":
        return SymplecticTransform(
            f"{self.name}^-1",
            self.d.T.copy(),
            -self.b.T.copy(),
            -self.c.T.copy(),
            self.a.T.copy(),
        )


def named_transform(name: str) -> SymplecticTransform:
    identity = _matrix([[1, 0], [0, 1]])
    zero = _matrix([[0, 0], [0, 0]])
    if name == "identity":
        return SymplecticTransform(name, identity, zero, zero, identity)
    if name == "T11":
        return SymplecticTransform(name, identity, _matrix([[1, 0], [0, 0]]), zero, identity)
    if name == "T22":
        return SymplecticTransform(name, identity, _matrix([[0, 0], [0, 1]]), zero, identity)
    if name == "T12":
        return SymplecticTransform(name, identity, _matrix([[0, 1], [1, 0]]), zero, identity)
    if name == "gl-shear-12":
        shear = _matrix([[1, 1], [0, 1]])
        shear_dual = _matrix([[1, 0], [-1, 1]])
        return SymplecticTransform(name, shear, zero, zero, shear_dual)
    if name == "swap-handles":
        swap = _matrix([[0, 1], [1, 0]])
        return SymplecticTransform(name, swap, zero, zero, swap)
    if name == "bridge-sign":
        sign = _matrix([[1, 0], [0, -1]])
        return SymplecticTransform(name, sign, zero, zero, sign)
    if name == "handle-s-1":
        return SymplecticTransform(
            name,
            _matrix([[0, 0], [0, 1]]),
            _matrix([[-1, 0], [0, 0]]),
            _matrix([[1, 0], [0, 0]]),
            _matrix([[0, 0], [0, 1]]),
        )
    if name == "handle-s-2":
        return SymplecticTransform(
            name,
            _matrix([[1, 0], [0, 0]]),
            _matrix([[0, 0], [0, -1]]),
            _matrix([[0, 0], [0, 1]]),
            _matrix([[1, 0], [0, 0]]),
        )
    if name == "full-s":
        return SymplecticTransform(name, zero, -identity, identity, zero)
    raise ValueError(f"unknown transform {name!r}")


def sp4_generator_names() -> tuple[str, ...]:
    """Return a concrete generator set for Sp(4,Z).

    The set uses symmetric translations T_B, GL(2,Z) changes of homology basis,
    and the full S transform.  These generate Sp(4,Z) in the standard Siegel
    presentation.
    """
    return ("T11", "T22", "T12", "gl-shear-12", "swap-handles", "bridge-sign", "full-s")


def sp4_generator_transforms() -> tuple[SymplecticTransform, ...]:
    return tuple(named_transform(name) for name in sp4_generator_names())


def seed_for_transform(
    transform: SymplecticTransform,
    q1: complex,
    q2: complex,
    q_bridge: complex,
) -> tuple[complex, complex, complex] | None:
    if transform.name == "identity":
        return q1, q2, q_bridge
    if transform.name in {"T11", "T22", "T12"}:
        return q1, q2, q_bridge
    if transform.name == "swap-handles":
        return q2, q1, q_bridge
    if transform.name == "bridge-sign":
        return q1, q2, -q_bridge
    return None


def transform_has_exact_plumbing_action(transform: SymplecticTransform) -> bool:
    return transform.name in {"identity", "T11", "T22", "T12", "swap-handles", "bridge-sign"}


@dataclass(frozen=True)
class PeriodToPlumbingResult:
    q1: complex
    q2: complex
    q3: complex
    omega: np.ndarray
    residual_matrix: np.ndarray
    max_abs_residual: float
    health_message: str
    source: str
    independent_numerical_inversion: bool


def period_to_plumbing_after_transform(
    *,
    target_omega: np.ndarray,
    transform: SymplecticTransform,
    q1: complex,
    q2: complex,
    q_bridge: complex,
    plumbing_word_len: int,
    plumbing_b_order: int,
    inverse_max_nfev: int,
    inverse_q3_component_bound: float,
    use_direct_chart_action: bool = True,
    target_chart: str = "modular-image",
) -> PeriodToPlumbingResult:
    if target_chart == "modular-image":
        preimage_target = transform.inverse().transform_omega(target_omega)
        inverse = solve_glasses_inverse_from_omega(
            preimage_target,
            initial_q=(q1, q2, q_bridge),
            max_word_len=plumbing_word_len,
            b_order=plumbing_b_order,
            max_nfev=inverse_max_nfev,
            q3_component_bound=inverse_q3_component_bound,
            q_abs_warning_threshold=0.4,
        )
        chart_omega = transform.transform_omega(inverse.omega)
        residual = np.asarray(chart_omega - target_omega, dtype=np.complex128)
        return PeriodToPlumbingResult(
            q1=inverse.q1,
            q2=inverse.q2,
            q3=inverse.q3,
            omega=chart_omega,
            residual_matrix=residual,
            max_abs_residual=float(np.max(np.abs(residual))),
            health_message=inverse.health_message,
            source=f"bookkeeping pullback through {transform.name} modular-image chart",
            independent_numerical_inversion=False,
        )

    if target_chart != "original":
        raise ValueError("target_chart must be 'original' or 'modular-image'")

    seed = seed_for_transform(transform, q1, q2, q_bridge)
    if use_direct_chart_action and seed is not None and transform_has_exact_plumbing_action(transform):
        tq1, tq2, tq3 = seed
        omega = schottky_glasses_period_matrix(
            tq1,
            tq2,
            tq3,
            max_word_len=plumbing_word_len,
            b_order=plumbing_b_order,
        )
        if transform.name in {"T11", "T22", "T12"}:
            residual = np.zeros_like(target_omega, dtype=np.complex128)
            source = "exact Dehn twist action on exponentiated plumbing q"
        else:
            residual = np.asarray(omega - target_omega, dtype=np.complex128)
            source = "exact chart action"
        _, health_message = schottky_health([tq1, tq2], threshold=0.4)
        bridge_abs = abs(tq3)
        if bridge_abs > 0.4:
            health_message = f"{health_message}; bridge |q3|={bridge_abs:.6g} exceeds 0.4"
        else:
            health_message = f"{health_message}; bridge |q3|={bridge_abs:.6g}"
        return PeriodToPlumbingResult(
            q1=tq1,
            q2=tq2,
            q3=tq3,
            omega=omega,
            residual_matrix=residual,
            max_abs_residual=float(np.max(np.abs(residual))),
            health_message=health_message,
            source=source,
            independent_numerical_inversion=False,
        )

    inverse_kwargs = {}
    if seed is not None:
        inverse_kwargs["initial_q"] = seed
    inverse = solve_glasses_inverse_from_omega(
        target_omega,
        max_word_len=plumbing_word_len,
        b_order=plumbing_b_order,
        max_nfev=inverse_max_nfev,
        q3_component_bound=inverse_q3_component_bound,
        q_abs_warning_threshold=0.4,
        **inverse_kwargs,
    )
    return PeriodToPlumbingResult(
        q1=inverse.q1,
        q2=inverse.q2,
        q3=inverse.q3,
        omega=inverse.omega,
        residual_matrix=inverse.residual_matrix,
        max_abs_residual=inverse.max_abs_residual,
        health_message=inverse.health_message,
        source="local inverse solve",
        independent_numerical_inversion=True,
    )


def expected_ratio(
    *,
    omega: np.ndarray,
    transform: SymplecticTransform,
    central_charge: float,
    expected_law: str,
    custom_chiral_weight: float | None = None,
) -> float:
    """Return the absolute-square automorphy ratio expected for this check.

    ``full-cft`` is the mapping-class invariant scalar partition function after
    anomaly cancellation/trivialization.  ``chiral-section`` is the Liouville
    matter sewing convention without ghosts, hence ``|det(C Omega + D)|^{-c}``.
    """
    det_abs = abs(transform.det_factor(omega))
    if expected_law == "full-cft":
        return 1.0
    if expected_law == "chiral-section":
        return det_abs ** (-float(central_charge))
    if expected_law == "custom-chiral-weight":
        if custom_chiral_weight is None:
            raise ValueError("custom-chiral-weight requires custom_chiral_weight")
        return det_abs ** (2.0 * float(custom_chiral_weight))
    raise ValueError(f"unknown expected law {expected_law!r}")


@dataclass(frozen=True)
class LiouvilleGenus2ModularCheckResult:
    transform: SymplecticTransform
    original_omega: np.ndarray
    transformed_omega_target: np.ndarray
    plumbing: PeriodToPlumbingResult
    original_partition: LiouvilleGenus2PairOfToriResult
    transformed_partition: LiouvilleGenus2PairOfToriResult
    expected_law: str
    expected_ratio: float
    observed_ratio: complex
    relative_error: float
    chart_residual_ok: bool
    modular_error_ok: bool
    bookkeeping_only: bool


@dataclass(frozen=True)
class LiouvilleGenus2GeneratorSuiteResult:
    results: tuple[LiouvilleGenus2ModularCheckResult, ...]

    @property
    def passed(self) -> tuple[LiouvilleGenus2ModularCheckResult, ...]:
        return tuple(result for result in self.results if result.modular_error_ok)

    @property
    def bookkeeping_only(self) -> tuple[LiouvilleGenus2ModularCheckResult, ...]:
        return tuple(result for result in self.results if result.bookkeeping_only)

    @property
    def chart_failures(self) -> tuple[LiouvilleGenus2ModularCheckResult, ...]:
        return tuple(result for result in self.results if not result.chart_residual_ok)

    @property
    def modular_failures(self) -> tuple[LiouvilleGenus2ModularCheckResult, ...]:
        return tuple(
            result
            for result in self.results
            if result.chart_residual_ok and not result.modular_error_ok and not result.bookkeeping_only
        )

    @property
    def all_passed(self) -> bool:
        return len(self.passed) == len(self.results)


def liouville_genus2_modular_check(
    *,
    b: float,
    q1: complex,
    q2: complex,
    q_bridge: complex,
    transform: SymplecticTransform,
    expected_law: str = "full-cft",
    custom_chiral_weight: float | None = None,
    block_order: int = 1,
    bridge_p_max: float | None = None,
    handle_p_max: float | None = None,
    bridge_quadrature_order: int = 4,
    handle_quadrature_order: int = 5,
    dps: int = 24,
    plumbing_word_len: int = 5,
    plumbing_b_order: int = 300,
    inverse_max_nfev: int = 80,
    inverse_q3_component_bound: float = 0.5,
    chart_residual_tolerance: float = 1.0e-5,
    modular_relative_tolerance: float = 1.0e-3,
    use_direct_chart_action: bool = True,
    target_chart: str = "original",
) -> LiouvilleGenus2ModularCheckResult:
    q1 = complex(q1)
    q2 = complex(q2)
    q_bridge = complex(q_bridge)
    omega = schottky_glasses_period_matrix(
        q1,
        q2,
        q_bridge,
        max_word_len=plumbing_word_len,
        b_order=plumbing_b_order,
    )
    transformed_target = transform.transform_omega(omega)
    plumbing = period_to_plumbing_after_transform(
        target_omega=transformed_target,
        transform=transform,
        q1=q1,
        q2=q2,
        q_bridge=q_bridge,
        plumbing_word_len=plumbing_word_len,
        plumbing_b_order=plumbing_b_order,
        inverse_max_nfev=inverse_max_nfev,
        inverse_q3_component_bound=inverse_q3_component_bound,
        use_direct_chart_action=use_direct_chart_action,
        target_chart=target_chart,
    )

    original = liouville_genus2_pair_of_tori(
        b=b,
        q1=q1,
        q2=q2,
        q_bridge=q_bridge,
        block_order=block_order,
        bridge_p_max=bridge_p_max,
        handle_p_max=handle_p_max,
        bridge_quadrature_order=bridge_quadrature_order,
        handle_quadrature_order=handle_quadrature_order,
        dps=dps,
    )
    transformed = liouville_genus2_pair_of_tori(
        b=b,
        q1=plumbing.q1,
        q2=plumbing.q2,
        q_bridge=plumbing.q3,
        block_order=block_order,
        bridge_p_max=bridge_p_max,
        handle_p_max=handle_p_max,
        bridge_quadrature_order=bridge_quadrature_order,
        handle_quadrature_order=handle_quadrature_order,
        dps=dps,
    )
    expected = expected_ratio(
        omega=omega,
        transform=transform,
        central_charge=original.central_charge,
        expected_law=expected_law,
        custom_chiral_weight=custom_chiral_weight,
    )
    observed = transformed.value / original.value
    relative_error = abs(observed - expected) / max(abs(observed), abs(expected), 1.0e-300)
    chart_ok = plumbing.max_abs_residual < chart_residual_tolerance
    bookkeeping_only = not plumbing.independent_numerical_inversion
    modular_ok = (not bookkeeping_only) and chart_ok and relative_error < modular_relative_tolerance
    return LiouvilleGenus2ModularCheckResult(
        transform=transform,
        original_omega=omega,
        transformed_omega_target=transformed_target,
        plumbing=plumbing,
        original_partition=original,
        transformed_partition=transformed,
        expected_law=expected_law,
        expected_ratio=float(expected),
        observed_ratio=observed,
        relative_error=float(relative_error),
        chart_residual_ok=bool(chart_ok),
        modular_error_ok=bool(modular_ok),
        bookkeeping_only=bool(bookkeeping_only),
    )


def liouville_genus2_sp4_generator_suite(
    *,
    b: float,
    q1: complex,
    q2: complex,
    q_bridge: complex,
    expected_law: str = "full-cft",
    custom_chiral_weight: float | None = None,
    block_order: int = 1,
    bridge_p_max: float | None = None,
    handle_p_max: float | None = None,
    bridge_quadrature_order: int = 3,
    handle_quadrature_order: int = 4,
    dps: int = 20,
    plumbing_word_len: int = 5,
    plumbing_b_order: int = 300,
    inverse_max_nfev: int = 80,
    inverse_q3_component_bound: float = 0.5,
    chart_residual_tolerance: float = 1.0e-5,
    modular_relative_tolerance: float = 1.0e-3,
    use_direct_chart_action: bool = True,
    target_chart: str = "original",
) -> LiouvilleGenus2GeneratorSuiteResult:
    results = []
    for transform in sp4_generator_transforms():
        results.append(
            liouville_genus2_modular_check(
                b=b,
                q1=q1,
                q2=q2,
                q_bridge=q_bridge,
                transform=transform,
                expected_law=expected_law,
                custom_chiral_weight=custom_chiral_weight,
                block_order=block_order,
                bridge_p_max=bridge_p_max,
                handle_p_max=handle_p_max,
                bridge_quadrature_order=bridge_quadrature_order,
                handle_quadrature_order=handle_quadrature_order,
                dps=dps,
                plumbing_word_len=plumbing_word_len,
                plumbing_b_order=plumbing_b_order,
                inverse_max_nfev=inverse_max_nfev,
                inverse_q3_component_bound=inverse_q3_component_bound,
                chart_residual_tolerance=chart_residual_tolerance,
                modular_relative_tolerance=modular_relative_tolerance,
                use_direct_chart_action=use_direct_chart_action,
                target_chart=target_chart,
            )
        )
    return LiouvilleGenus2GeneratorSuiteResult(results=tuple(results))


def _default_bridge_cutoff(q_bridge: complex, requested: float | None) -> float | None:
    if requested is not None:
        return requested
    if abs(q_bridge) < 0.02:
        return 1.7
    return None


def _print_single_result(result: LiouvilleGenus2ModularCheckResult, q1: complex, q2: complex, q_bridge: complex) -> None:
    det_factor = result.transform.det_factor(result.original_omega)
    print("Liouville genus-two modular check")
    print(f"  transform={result.transform.name}")
    print(f"  expected law={result.expected_law}")
    print(f"  c={result.original_partition.central_charge:.12g}")
    print(f"  det(C Omega + D)={format_complex(det_factor)}")
    print(f"  original q=({format_complex(q1)}, {format_complex(q2)}, {format_complex(q_bridge)})")
    print(
        "  transformed q="
        f"({format_complex(result.plumbing.q1)}, {format_complex(result.plumbing.q2)}, {format_complex(result.plumbing.q3)})"
    )
    print(f"  period-to-plumbing source={result.plumbing.source}")
    print(f"  max |Delta Omega|={result.plumbing.max_abs_residual:.6e}")
    print(f"  plumbing health={result.plumbing.health_message}")
    print(f"  Z(original)={format_complex(result.original_partition.value)}")
    print(f"  Z(transformed)={format_complex(result.transformed_partition.value)}")
    print(f"  observed ratio={format_complex(result.observed_ratio)}")
    print(f"  expected ratio={result.expected_ratio:.12e}")
    print(f"  relative error={result.relative_error:.6e}")
    print(f"  chart residual ok={result.chart_residual_ok}")
    print(f"  independent numerical inversion={result.plumbing.independent_numerical_inversion}")
    print(f"  modular check ok={result.modular_error_ok}")
    if result.bookkeeping_only:
        print("  status=bookkeeping only: this is not a faithful modular check")
    elif not result.chart_residual_ok:
        print("  status=chart failure: transformed Omega was not accurately inverted in this glasses chart")
    elif not result.modular_error_ok:
        print("  status=modular mismatch")
    else:
        print("  status=passed")


def _print_suite_result(suite: LiouvilleGenus2GeneratorSuiteResult) -> None:
    print("Liouville genus-two Sp(4,Z) generator suite")
    print("  generator set: " + ", ".join(sp4_generator_names()))
    print("  status legend: pass | chart-failure | modular-mismatch | bookkeeping-only")
    print("")
    print("  generator        status            |Delta Omega|    rel error       source")
    print("  -------------------------------------------------------------------------------")
    for result in suite.results:
        if result.bookkeeping_only:
            status = "bookkeeping-only"
        elif result.modular_error_ok:
            status = "pass"
        elif not result.chart_residual_ok:
            status = "chart-failure"
        else:
            status = "modular-mismatch"
        print(
            f"  {result.transform.name:<16} "
            f"{status:<16} "
            f"{result.plumbing.max_abs_residual:>12.5e}  "
            f"{result.relative_error:>12.5e}  "
            f"{result.plumbing.source}"
        )
    print("")
    print(
        f"  passed={len(suite.passed)}, "
        f"chart_failures={len(suite.chart_failures)}, "
        f"modular_failures={len(suite.modular_failures)}, "
        f"bookkeeping_only={len(suite.bookkeeping_only)}"
    )
    if suite.all_passed:
        print("  status=all generator checks passed")
    else:
        print("  status=not fully verified by independent numerical plumbing")


def run(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Check genus-two Liouville modular covariance in Omega variables.")
    parser.add_argument("--b", type=float, default=0.8)
    parser.add_argument("--q1", type=parse_complex, default=0.08 * cmath.exp(0.1j))
    parser.add_argument("--q2", type=parse_complex, default=0.07 * cmath.exp(-0.05j))
    parser.add_argument("--q-bridge", type=parse_complex, default=0.03 * cmath.exp(0.2j))
    parser.add_argument(
        "--transform",
        choices=[
            "identity",
            "T11",
            "T22",
            "T12",
            "gl-shear-12",
            "swap-handles",
            "bridge-sign",
            "handle-s-1",
            "handle-s-2",
            "full-s",
        ],
        default="handle-s-1",
    )
    parser.add_argument(
        "--suite",
        choices=["single", "sp4-generators"],
        default="single",
        help="run one transform or the standard Sp(4,Z) generator set",
    )
    parser.add_argument(
        "--target-chart",
        choices=["modular-image", "original"],
        default="original",
        help=(
            "use independent numerical inversion in the original chart, or use modular-image "
            "bookkeeping that is not a faithful modular check"
        ),
    )
    parser.add_argument(
        "--expected-law",
        choices=["full-cft", "chiral-section", "custom-chiral-weight"],
        default="chiral-section",
    )
    parser.add_argument("--custom-chiral-weight", type=float)
    parser.add_argument("--block-order", type=int, default=1)
    parser.add_argument("--bridge-p-max", type=float)
    parser.add_argument("--handle-p-max", type=float, default=1.5)
    parser.add_argument("--bridge-quadrature-order", type=int, default=4)
    parser.add_argument("--handle-quadrature-order", type=int, default=5)
    parser.add_argument("--dps", type=int, default=20)
    parser.add_argument("--plumbing-word-len", type=int, default=5)
    parser.add_argument("--plumbing-b-order", type=int, default=300)
    parser.add_argument("--inverse-max-nfev", type=int, default=80)
    parser.add_argument("--force-inverse", action="store_true")
    parser.add_argument("--chart-residual-tolerance", type=float, default=1.0e-5)
    parser.add_argument("--modular-relative-tolerance", type=float, default=1.0e-3)
    args = parser.parse_args(argv)

    bridge_p_max = _default_bridge_cutoff(args.q_bridge, args.bridge_p_max)
    common_kwargs = dict(
        b=args.b,
        q1=args.q1,
        q2=args.q2,
        q_bridge=args.q_bridge,
        expected_law=args.expected_law,
        custom_chiral_weight=args.custom_chiral_weight,
        block_order=args.block_order,
        bridge_p_max=bridge_p_max,
        handle_p_max=args.handle_p_max,
        bridge_quadrature_order=args.bridge_quadrature_order,
        handle_quadrature_order=args.handle_quadrature_order,
        dps=args.dps,
        plumbing_word_len=args.plumbing_word_len,
        plumbing_b_order=args.plumbing_b_order,
        inverse_max_nfev=args.inverse_max_nfev,
        chart_residual_tolerance=args.chart_residual_tolerance,
        modular_relative_tolerance=args.modular_relative_tolerance,
        use_direct_chart_action=not args.force_inverse,
        target_chart=args.target_chart,
    )
    if args.suite == "sp4-generators":
        suite = liouville_genus2_sp4_generator_suite(**common_kwargs)
        _print_suite_result(suite)
        return

    transform = named_transform(args.transform)
    result = liouville_genus2_modular_check(
        transform=transform,
        **common_kwargs,
    )
    _print_single_result(result, args.q1, args.q2, args.q_bridge)


if __name__ == "__main__":
    run()
