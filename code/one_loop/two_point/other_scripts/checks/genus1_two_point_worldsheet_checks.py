#!/usr/bin/env python3
"""Pointwise and channel-overlap checks for the genus-one worldsheet code."""

from __future__ import annotations

from c1_paths import PACKAGE_ROOT, data_root, package_path, module_path

import argparse
import json
import math
from pathlib import Path
import tempfile

import numpy as np

try:
    from genus1_two_point_worldsheet import (
        LiouvilleTorusTwoPoint,
        MomentumRule,
        torus_prime_form_norm,
    )
except ImportError:  # pragma: no cover
    from genus1_two_point_worldsheet import (
        LiouvilleTorusTwoPoint,
        MomentumRule,
        torus_prime_form_norm,
    )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def check_native_coefficient_bank() -> None:
    from genus1_two_point_coefficient_bank import prepare, load_bank
    from genus1_two_point_thermal_integration import Kernel

    with tempfile.TemporaryDirectory() as directory:
        bank = prepare(.5, 2, (2, 2), (2, 3), Path(directory))
        error = Kernel(bank).validate_against_original()
        metadata, arrays = load_bank(bank)
        legacy = dict(metadata)
        legacy.pop("format")
        bank.with_suffix(".json").write_text(json.dumps(legacy))
        try:
            Kernel(bank)
        except ValueError:
            pass
        else:
            raise AssertionError("an unversioned transformed bank was accepted")
        bank.with_suffix(".json").write_text(json.dumps(metadata))
        np.savez_compressed(bank, **arrays)
        try:
            Kernel(bank)
        except ValueError:
            pass
        else:
            raise AssertionError("relabelled metadata bypassed the array format check")
    print(f"\nnative bank/evaluator check: {error:.6e}; legacy banks rejected")


def run(args: argparse.Namespace) -> None:
    check_native_coefficient_bank()
    tau = 0.17 + 1.08j
    z = 0.80 + 0.50j
    rule = MomentumRule.power_legendre(args.p_max, args.momentum_order, args.momentum_power)
    correlator = LiouvilleTorusTwoPoint(
        1.0j * args.x,
        momentum_rule=rule,
        necklace_orders=(args.necklace_order, args.necklace_order),
        ope_orders=(args.ope_q_order, args.ope_z_order),
        necklace_backend="c-recursion",
        ope_backend="c-recursion",
        special_dps=args.dps,
    )
    necklace = correlator.correlator_necklace(z, tau)
    ope_uncontinued = correlator.correlator_ope(z, tau)
    ope = correlator.correlator_ope_analytically_continued(z, tau)
    scale = max(abs(necklace), abs(ope), 1.0e-300)
    relative = abs(necklace - ope) / scale
    print("Liouville torus two-point channel overlap")
    print(f"  omega={correlator.omega!r}")
    print(f"  tau={tau!r}")
    print(f"  z={z!r}, |z|/(2*pi)={abs(z)/(2*math.pi):.8f}")
    print(f"  necklace={necklace!r}")
    print(f"  OPE original contour={ope_uncontinued!r}")
    if args.x > 1.0:
        print(f"  crossed pole residue={correlator.crossed_ope_pole_residue()!r}")
        print(f"  OPE -2i residue term={correlator.correlator_ope_residue(z,tau)!r}")
    print(f"  OPE analytically continued={ope!r}")
    print(f"  symmetric relative difference={relative:.6e}")
    print(f"  cached Upsilon arguments={len(correlator.special._log_cache)}")
    require(abs(necklace.imag) < 2.0e-9 * max(abs(necklace), 1.0), "necklace result is not real")
    require(abs(ope.imag) < 2.0e-9 * max(abs(ope), 1.0), "OPE result is not real")

    direct_reference = LiouvilleTorusTwoPoint(
        1.0j * args.x,
        momentum_rule=rule,
        necklace_orders=(args.necklace_order, args.necklace_order),
        ope_orders=(args.ope_q_order, args.ope_z_order),
        necklace_backend="direct-descendants",
        ope_backend="direct-descendants",
        special_dps=args.dps,
    )
    direct_necklace = direct_reference.correlator_necklace(z, tau)
    direct_ope = direct_reference.correlator_ope(z, tau)
    necklace_backend_error = abs(necklace - direct_necklace) / max(
        abs(direct_necklace),
        1.0e-300,
    )
    ope_backend_error = abs(ope_uncontinued - direct_ope) / max(
        abs(direct_ope),
        1.0e-300,
    )
    print("\nc recursion vs direct descendant backend")
    print(f"  necklace relative difference={necklace_backend_error:.6e}")
    print(f"  OPE relative difference={ope_backend_error:.6e}")
    require(
        necklace_backend_error < 2.0e-10,
        "c-recursion disagrees with direct necklace sewing",
    )
    require(
        ope_backend_error < 2.0e-10,
        "c-recursion disagrees with direct OPE sewing",
    )

    near = 1.0e-6 + 2.0e-7j
    ratio = torus_prime_form_norm(near, tau) / abs(near)
    print("\nprime-form collision normalization")
    print(f"  E(z|tau)/|z|={ratio:.12e}")
    require(abs(ratio - 1.0) < 2.0e-6, "prime form is not normalized to E~z")

    if args.x < 1.0:
        original = correlator.correlator_patched(z, tau)
        s_tau = -1.0 / tau
        s_z = z / tau
        transformed = correlator.correlator_patched(s_z, s_tau)
        predicted = abs(tau) ** (4.0 + correlator.omega**2) * original
        s_relative = abs(transformed - predicted) / max(
            abs(transformed), abs(predicted), 1.0e-300
        )
        t_transformed = correlator.correlator_patched(z, tau + 1.0)
        t_relative = abs(t_transformed - original) / max(
            abs(t_transformed), abs(original), 1.0e-300
        )
        print("\nmodular covariance")
        print(f"  S-transform relative difference={s_relative:.6e}")
        print(f"  T-transform relative difference={t_relative:.6e}")


def parser() -> argparse.ArgumentParser:
    out = argparse.ArgumentParser()
    out.add_argument("--x", type=float, default=0.4)
    out.add_argument("--p-max", type=float, default=4.0)
    out.add_argument("--momentum-order", type=int, default=8)
    out.add_argument("--momentum-power", type=float, default=2.0)
    out.add_argument("--necklace-order", type=int, default=4)
    out.add_argument("--ope-q-order", type=int, default=2)
    out.add_argument("--ope-z-order", type=int, default=6)
    out.add_argument("--dps", type=int, default=26)
    return out


if __name__ == "__main__":
    run(parser().parse_args())
