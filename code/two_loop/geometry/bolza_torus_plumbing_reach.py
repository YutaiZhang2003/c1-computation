#!/usr/bin/env python3
"""Search for a two-torus plumbing chart reaching the Bolza curve.

This is a reachability diagnostic for the separating/glasses channel.  It scans
symplectic representatives of the Bolza period matrix, then attempts a local
inverse solve with the bridge plumbing parameter written in polar form.  The
polar parametrization enforces |q_bridge| < q3_bound during the solve.
"""

from __future__ import annotations

from c1_paths import PACKAGE_ROOT, data_root, package_path, module_path

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    from bolza_ccy_recursion import bolza_period_matrix
    from liouville_genus2 import format_complex
    from liouville_genus2_modular_check import named_transform, sp4_generator_names
    from liouville_torus import q_from_tau
    from plumbing_algorithms import genus2_symmetric_period_vector, schottky_glasses_period_matrix
except ImportError:  # pragma: no cover - supports package-style execution
    from bolza_ccy_recursion import bolza_period_matrix
    from liouville_genus2 import format_complex
    from liouville_genus2_modular_check import named_transform, sp4_generator_names
    from liouville_torus import q_from_tau
    from plumbing_algorithms import genus2_symmetric_period_vector, schottky_glasses_period_matrix


TWO_PI_I = 2.0j * math.pi


def _block_matrix(transform) -> np.ndarray:
    return np.block([[transform.a, transform.b], [transform.c, transform.d]]).astype(int)


def _matrix_key(matrix: np.ndarray) -> tuple[int, ...]:
    return tuple(int(value) for value in matrix.ravel())


def _symplectic_blocks(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return matrix[:2, :2], matrix[:2, 2:], matrix[2:, :2], matrix[2:, 2:]


def transform_omega(matrix: np.ndarray, omega: np.ndarray) -> np.ndarray:
    a, b, c, d = _symplectic_blocks(matrix)
    return (a @ omega + b) @ np.linalg.inv(c @ omega + d)


def period_matrix_is_riemann(omega: np.ndarray, floor: float = 1.0e-10) -> bool:
    sym = 0.5 * (omega + omega.T)
    return bool(np.min(np.linalg.eigvalsh(sym.imag)) > floor)


def generator_matrices() -> tuple[tuple[str, np.ndarray], ...]:
    out: list[tuple[str, np.ndarray]] = []
    seen: set[tuple[int, ...]] = set()
    for name in sp4_generator_names():
        transform = named_transform(name)
        for label, item in ((name, transform), (f"{name}^-1", transform.inverse())):
            matrix = _block_matrix(item)
            key = _matrix_key(matrix)
            if key in seen:
                continue
            seen.add(key)
            out.append((label, matrix))
    return tuple(out)


@dataclass(frozen=True)
class Candidate:
    score: float
    leading_q1_abs: float
    leading_q2_abs: float
    leading_q3_abs: float
    word: str
    matrix: list[list[int]]
    omega11: str
    omega12: str
    omega22: str


@dataclass(frozen=True)
class InverseAttempt:
    candidate_rank: int
    word: str
    success: bool
    nfev: int
    residual_norm: float
    max_symmetric_residual: float
    omega_symmetry_error: float
    q1: str
    q2: str
    q3: str
    q1_abs: float
    q2_abs: float
    q3_abs: float
    omega11: str
    omega12: str
    omega22: str


def enumerate_symplectic_words(depth: int) -> list[tuple[str, np.ndarray]]:
    identity = np.eye(4, dtype=int)
    seen = {_matrix_key(identity): "I"}
    front = [("I", identity)]
    generators = generator_matrices()
    for _ in range(int(depth)):
        new_front: list[tuple[str, np.ndarray]] = []
        for word, matrix in front:
            for label, generator in generators:
                next_matrix = generator @ matrix
                key = _matrix_key(next_matrix)
                if key in seen:
                    continue
                next_word = f"{label} {word}"
                seen[key] = next_word
                new_front.append((next_word, next_matrix))
        front = new_front
    return [(word, np.asarray(key, dtype=int).reshape(4, 4)) for key, word in seen.items()]


def candidate_from_matrix(word: str, matrix: np.ndarray, omega_bolza: np.ndarray) -> Candidate | None:
    try:
        omega = transform_omega(matrix, omega_bolza)
    except np.linalg.LinAlgError:
        return None
    if not period_matrix_is_riemann(omega):
        return None
    q1_abs = abs(np.exp(TWO_PI_I * omega[0, 0]))
    q2_abs = abs(np.exp(TWO_PI_I * omega[1, 1]))
    q3_lead = -TWO_PI_I * omega[0, 1]
    q3_abs = abs(q3_lead)
    if not all(math.isfinite(value) for value in (q1_abs, q2_abs, q3_abs)):
        return None
    score = q3_abs + 0.05 * (q1_abs + q2_abs)
    return Candidate(
        score=float(score),
        leading_q1_abs=float(q1_abs),
        leading_q2_abs=float(q2_abs),
        leading_q3_abs=float(q3_abs),
        word=word,
        matrix=matrix.astype(int).tolist(),
        omega11=format_complex(complex(omega[0, 0])),
        omega12=format_complex(complex(omega[0, 1])),
        omega22=format_complex(complex(omega[1, 1])),
    )


def _logit(value: float) -> float:
    value = min(max(float(value), 1.0e-12), 1.0 - 1.0e-12)
    return math.log(value / (1.0 - value))


def _unpack_variables(x: np.ndarray, q3_bound: float) -> tuple[complex, complex, complex, complex, complex]:
    tau1 = complex(float(x[0]), float(x[1]))
    tau2 = complex(float(x[2]), float(x[3]))
    radius = float(q3_bound) / (1.0 + math.exp(-float(x[4])))
    phase = float(x[5])
    q3 = radius * complex(math.cos(phase), math.sin(phase))
    return tau1, tau2, q_from_tau(tau1), q_from_tau(tau2), q3


def solve_polar_glasses_inverse(
    target_omega: np.ndarray,
    *,
    q3_bound: float,
    max_word_len: int,
    b_order: int,
    max_nfev: int,
    leading_scale: float = 0.85,
) -> tuple[bool, int, float, float, float, complex, complex, complex, np.ndarray]:
    target_vec = genus2_symmetric_period_vector(target_omega)
    tau1_seed = complex(target_omega[0, 0])
    tau2_seed = complex(target_omega[1, 1])
    tau1_seed = complex(tau1_seed.real, min(max(tau1_seed.imag, 1.0e-6), 10.0))
    tau2_seed = complex(tau2_seed.real, min(max(tau2_seed.imag, 1.0e-6), 10.0))
    q3_seed = -TWO_PI_I * target_omega[0, 1]
    q3_abs = min(abs(q3_seed) * float(leading_scale), float(q3_bound) * 0.98)
    q3_phase = math.atan2(q3_seed.imag, q3_seed.real)
    x0 = np.asarray(
        [
            tau1_seed.real,
            tau1_seed.imag,
            tau2_seed.real,
            tau2_seed.imag,
            _logit(q3_abs / float(q3_bound)),
            q3_phase,
        ],
        dtype=float,
    )
    lower = np.asarray([-np.inf, 1.0e-6, -np.inf, 1.0e-6, -40.0, -np.inf], dtype=float)
    upper = np.asarray([np.inf, 10.0, np.inf, 10.0, 40.0, np.inf], dtype=float)

    def residual(x: np.ndarray) -> np.ndarray:
        tau1, tau2, q1, q2, q3 = _unpack_variables(x, q3_bound)
        if tau1.imag <= 0.0 or tau2.imag <= 0.0 or abs(q1) >= 1.0 or abs(q2) >= 1.0:
            return 1.0e6 * np.ones(6, dtype=float)
        try:
            omega = schottky_glasses_period_matrix(q1, q2, q3, max_word_len=max_word_len, b_order=b_order)
        except Exception:
            return 1.0e6 * np.ones(6, dtype=float)
        return genus2_symmetric_period_vector(omega) - target_vec

    from scipy.optimize import least_squares

    opt = least_squares(
        residual,
        x0,
        bounds=(lower, upper),
        max_nfev=int(max_nfev),
        xtol=1.0e-10,
        ftol=1.0e-10,
        gtol=1.0e-10,
    )
    _, _, q1, q2, q3 = _unpack_variables(opt.x, q3_bound)
    omega = schottky_glasses_period_matrix(q1, q2, q3, max_word_len=max_word_len, b_order=b_order)
    residual_matrix = omega - target_omega
    residual_vector = genus2_symmetric_period_vector(residual_matrix)
    return (
        bool(opt.success),
        int(opt.nfev),
        float(np.linalg.norm(residual_vector)),
        float(np.max(np.abs(residual_vector))),
        float(np.max(np.abs(omega - omega.T))),
        q1,
        q2,
        q3,
        omega,
    )


def run(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Search whether two-torus plumbing reaches the Bolza curve.")
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--candidate-count", type=int, default=20)
    parser.add_argument("--solve-count", type=int, default=8)
    parser.add_argument("--q3-bound", type=float, default=0.985)
    parser.add_argument("--word-len", type=int, default=4)
    parser.add_argument("--b-order", type=int, default=160)
    parser.add_argument("--max-nfev", type=int, default=80)
    parser.add_argument("--out-json", type=Path)
    args = parser.parse_args(argv)

    omega_bolza = bolza_period_matrix()
    candidates = []
    for word, matrix in enumerate_symplectic_words(args.depth):
        candidate = candidate_from_matrix(word, matrix, omega_bolza)
        if candidate is not None:
            candidates.append(candidate)
    candidates.sort(key=lambda item: item.score)
    candidates = candidates[: int(args.candidate_count)]

    print("Bolza two-torus plumbing reachability")
    print(f"  symplectic depth={args.depth}")
    print(f"  candidates kept={len(candidates)}")
    print(f"  q3 polar bound={args.q3_bound:.12g}")
    print("  best leading candidates:")
    for idx, candidate in enumerate(candidates[: min(8, len(candidates))]):
        print(
            f"    #{idx}: score={candidate.score:.6e} "
            f"|q|lead=({candidate.leading_q1_abs:.6e}, {candidate.leading_q2_abs:.6e}, "
            f"{candidate.leading_q3_abs:.6e}) word={candidate.word}"
        )

    attempts = []
    for idx, candidate in enumerate(candidates[: int(args.solve_count)]):
        matrix = np.asarray(candidate.matrix, dtype=int)
        target = transform_omega(matrix, omega_bolza)
        solved = solve_polar_glasses_inverse(
            target,
            q3_bound=args.q3_bound,
            max_word_len=args.word_len,
            b_order=args.b_order,
            max_nfev=args.max_nfev,
        )
        success, nfev, residual_norm, max_symmetric_residual, symmetry_error, q1, q2, q3, omega = solved
        attempts.append(
            InverseAttempt(
                candidate_rank=idx,
                word=candidate.word,
                success=success,
                nfev=nfev,
                residual_norm=residual_norm,
                max_symmetric_residual=max_symmetric_residual,
                omega_symmetry_error=symmetry_error,
                q1=format_complex(q1),
                q2=format_complex(q2),
                q3=format_complex(q3),
                q1_abs=abs(q1),
                q2_abs=abs(q2),
                q3_abs=abs(q3),
                omega11=format_complex(complex(omega[0, 0])),
                omega12=format_complex(complex(omega[0, 1])),
                omega22=format_complex(complex(omega[1, 1])),
            )
        )
        print(
            f"  solve #{idx}: sym-residual={max_symmetric_residual:.6e}, norm={residual_norm:.6e}, "
            f"symmetry={symmetry_error:.6e}, "
            f"success={success}, nfev={nfev}, "
            f"|q|=({abs(q1):.6e}, {abs(q2):.6e}, {abs(q3):.6e})"
        )
    attempts.sort(key=lambda item: item.max_symmetric_residual)

    if attempts:
        best = attempts[0]
        print("  best polar inverse:")
        print(
            f"    rank={best.candidate_rank}, sym-residual={best.max_symmetric_residual:.12e}, "
            f"norm={best.residual_norm:.12e}, symmetry={best.omega_symmetry_error:.12e}"
        )
        print(f"    q1={best.q1}  |q1|={best.q1_abs:.12e}")
        print(f"    q2={best.q2}  |q2|={best.q2_abs:.12e}")
        print(f"    q3={best.q3}  |q3|={best.q3_abs:.12e}")
        print(f"    word={best.word}")

    if args.out_json is not None:
        payload = {
            "depth": args.depth,
            "q3_bound": args.q3_bound,
            "word_len": args.word_len,
            "b_order": args.b_order,
            "max_nfev": args.max_nfev,
            "candidates": [asdict(item) for item in candidates],
            "attempts_by_residual": [asdict(item) for item in attempts],
        }
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"  wrote {args.out_json}")


if __name__ == "__main__":
    run()
