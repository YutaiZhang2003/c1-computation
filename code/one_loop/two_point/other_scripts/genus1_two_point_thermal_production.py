"""Prepare coefficient banks and run the thermal genus-one 1->1 radius scan."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from c1_paths import PACKAGE_ROOT
from genus1_two_point_coefficient_bank import bank_stem, NATIVE_BANK_METHOD

CONFIG = Path(__file__).resolve().parents[1] / "production_settings.json"


def load_settings():
    settings = json.loads(CONFIG.read_text())
    if (settings["necklace_backend"] != "c-recursion" or
            settings["ope_backend"] != "c-recursion"):
        raise ValueError("Both torus two-point production channels must use c-recursion.")
    return settings


def run_point(radius, settings, output_dir):
    radius = float(radius)
    if not math.isfinite(radius) or radius <= 1:
        raise ValueError("The retained real-contour calculation requires R > 1.")
    output_dir = Path(output_dir).resolve()
    x = 1 / radius
    n = settings["momentum_order"]
    neck, ope = settings["necklace_orders"], settings["ope_orders"]
    stem = bank_stem(x, n, neck, ope)
    target = output_dir / "scan_native_q_complete_collision" / f"R{radius:g}.json"
    expected = {k: settings[k] for k in ("power", "replicates", "seed", "collision_radius")}
    expected.update(R=radius, collision_remainder_included=True, height_scale=1., short_scale=4/x**2)
    bank_settings = dict(NATIVE_BANK_METHOD, x=x, momentum_order=n, p_max=6., power=2.,
                         necklace_orders=neck, ope_orders=ope)
    if target.exists():
        saved = json.loads(target.read_text())
        if (any(saved.get(k) != v for k, v in expected.items()) or
                any(saved.get("kernel", {}).get(k) != v for k, v in bank_settings.items())):
            raise ValueError("Existing receipt has different settings: " + str(target))
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    launch = [sys.executable, "-B", str(PACKAGE_ROOT / "run.py")]
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    with target.with_suffix(".log").open("w") as log:
        subprocess.run(launch + ["genus1_two_point_coefficient_bank", "--x", str(x),
            "--nodes", str(n), "--necklace", *map(str, neck), "--ope", *map(str, ope),
            "--output-dir", str(output_dir / "liouville")],
            check=True, stdout=log, stderr=subprocess.STDOUT, env=env)
        subprocess.run(launch + ["genus1_two_point_thermal_integration",
            str(output_dir / "liouville" / (stem + ".npz")),
            "--power", str(settings["power"]), "--replicates", str(settings["replicates"]),
            "--seed", str(settings["seed"]), "--radius", str(settings["collision_radius"]),
            "--collision-remainder", "--output", str(target)],
            check=True, stdout=log, stderr=subprocess.STDOUT, env=env)
    saved = json.loads(target.read_text())
    print(json.dumps({k: saved[k] for k in ("R", "thermal_imag", "sampling_SE_thermal")}), flush=True)
    return target


def main():
    defaults = load_settings()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--radii", type=float, nargs="+", default=defaults["radii"])
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--nodes", type=int, default=defaults["momentum_order"])
    p.add_argument("--necklace", type=int, nargs=2, default=defaults["necklace_orders"])
    p.add_argument("--ope", type=int, nargs=2, default=defaults["ope_orders"])
    p.add_argument("--power", type=int, default=defaults["power"])
    p.add_argument("--replicates", type=int, default=defaults["replicates"])
    p.add_argument("--radius", type=float, default=defaults["collision_radius"])
    p.add_argument("--seed", type=int, default=defaults["seed"])
    a = p.parse_args()
    if a.replicates < 2 or a.workers < 1 or a.power < 0 or a.nodes < 1 or a.radius <= 0:
        p.error("Use at least two replicates, one worker and one momentum node, and a positive radius.")
    if len(set(a.radii)) != len(a.radii):
        p.error("Repeated radii would write to the same receipt.")
    settings = dict(defaults, momentum_order=a.nodes, necklace_orders=list(a.necklace),
        ope_orders=list(a.ope), power=a.power, replicates=a.replicates,
        collision_radius=a.radius, seed=a.seed)
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        list(pool.map(lambda r: run_point(r, settings, a.output_dir), a.radii))


if __name__ == "__main__":
    main()
