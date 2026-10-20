"""Independent moduli-sampling experiment for the thermal pair-collision term.

The original, verified CFT banks define a fixed regional momentum quadrature.
All sampling methods use EXACTLY the same deterministic bank resolver. No
coefficient is interpolated, and no matrix-model value enters this experiment.
This measures moduli variance, not convergence of the spectral quadrature or
of the omitted collision/finite-tube remainders.
"""

from genus1_source_paths import ROOT as _SOURCE_ROOT, source_path as _source_path
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
HERE = _SOURCE_ROOT / 'output/thermal_sampling_20260927'
PREVIOUS = HERE.parent / "thermal_one_loop_20260927"

from thermal_saved_inputs import Banks, SOURCE, read, pair_factor
from genus1_three_point_channel_atlas import make_atlas
from genus1_channel_evaluation import pair_projection_values
from genus1_three_point_channel_atlas import (
    canonical_loop_displacement, nearest_torus_displacement)
from scipy.spatial import cKDTree
from scipy.stats import qmc
import numpy as np

TAU = 2 * math.pi


class FullPrefix:
    """Evaluate the unchanged full block without unused diagnostic prefixes."""
    def __init__(self, bank):self.bank=bank
    def cuts(self):return (self.bank.cuts()[0],)
    def pair_logs(self,*args):return self.bank.pair_logs(*args)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def features(tau, total):
    loop = canonical_loop_displacement(-total, tau)
    return np.log1p([2 * loop.imag, 2 * (TAU * tau.imag - loop.imag)])


class Density:
    """Fixed function shared by all methods; regional banks are held fixed."""
    def __init__(self, tag):
        self.tag = tag
        self.t = read(SOURCE / "energies" / tag / "design.json")["t"]
        self.R = 2 / self.t
        self.atlas = make_atlas(self.t)
        self.banks = Banks(tag)
        self.groups = {}
        groups = {}
        for path in sorted((PREVIOUS / "inputs" / tag / "grids").glob("*/spec.json")):
            spec = read(path)
            if spec["case"]["kind"] != "pair":
                continue
            g = spec["case"]["geometry"]
            pair = tuple(g["pair"])
            feature = features(complex(*g["tau"]), complex(*g["delta_total"]))
            groups.setdefault(pair, []).append((path.parent.name, feature))
        for pair, rows in groups.items():
            x = np.array([v[1] for v in rows])
            self.groups[pair] = (cKDTree(x), [v[0] for v in rows], x)
        self.counts = Counter()
        self.distances = []

    def evaluate(self, coordinates, jacobian, *, alternate=False, with_prefixes=False):
        result = np.zeros((len(coordinates), 3))
        prefix_changes = np.zeros((len(coordinates), 3))
        for i, ((tau1, height, x, y), jac) in enumerate(zip(coordinates, jacobian)):
            tau = complex(tau1, height)
            total = TAU * (x + tau * y)
            if abs(nearest_torus_displacement(total, tau)) < .5:
                continue
            feature = features(tau, total)
            for pair, multiplicity in (((0, 1), 1), ((0, 2), 2)):
                tree, keys, anchors = self.groups[pair]
                distance, index = tree.query(feature, k=2 if alternate else 1)
                if alternate:
                    distance, index = distance[1], index[1]
                key = keys[index]
                _, _, bank, ellipse = self.banks.get(key)
                self.counts[key] += 1
                self.distances.append(float(distance))
                geometry = dict(pair=pair, tau=tau, delta_total=total,
                                radius=.125, integrated=True)
                block_values = pair_projection_values(bank, self.atlas, geometry,
                    bank.orders, elliptic=ellipse if with_prefixes else FullPrefix(ellipse))[0]
                value = block_values[0]
                zero = -2 * math.pi**2 * value.imag * multiplicity * jac
                factor = pair_factor(self.R, tau, pair, total)
                result[i] += [zero, zero * factor, zero * (factor - 1)]
                if with_prefixes:
                    names = [entry[0] for entry in ellipse.cuts()]
                    for k, name in enumerate(("H6", "H4", "long2")):
                        change = block_values[names.index(name)] - value
                        prefix_changes[i,k] += (-2 * math.pi**2 * change.imag *
                                                multiplicity * jac * (factor - 1))
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("nonfinite moduli observation")
        return (result, prefix_changes) if with_prefixes else result


def physical_map(unit, R, *, method="legacy", region="tail"):
    """Exact Jacobians, no height cutoff, and a correlated puncture map.

    Legacy: T=8/(1-u), uniform puncture fraction y.
    Thermal: T=8+s*((1-u)^(-2)-1), s=max(8,R^2).
    The puncture map samples its distance to the nearer cylinder end with a
    truncated rational law, plus 1/3 uniform coverage of every KK ridge.
    Inverse CDF of the MIXTURE is solved monotonically, avoiding selector jumps.
    """
    u = np.asarray(unit)
    tau1 = u[:, 1] - .5
    if region == "bulk":
        lower = np.sqrt(1 - tau1**2)
        height = lower + (8 - lower) * u[:, 0]
        height_jac = 8 - lower
    elif method == "legacy":
        height = 8 / (1 - u[:, 0])
        height_jac = 8 / (1 - u[:, 0])**2
    else:
        scale = max(8., R**2)
        height = 8 + scale * ((1 - u[:, 0])**-2 - 1)
        height_jac = 2 * scale * (1 - u[:, 0])**-3
    y = u[:, 3].copy()
    y_jac = np.ones(len(u))
    if method != "legacy":
        # On 0<y<1/2, the rational component has density proportional to
        # a/(a+y)^2. Reflect it about 1/2; include a uniform component.
        a = R**2 / (TAU * height)
        target = 2 * np.minimum(y, 1 - y)
        lo = np.zeros(len(u)); hi = np.full(len(u), .5)
        c = .5 / (a + .5)
        for _ in range(48):
            mid = (lo + hi) / 2
            cdf = (2 * mid) / 3 + (2 / 3) * mid / (a + mid) / c
            lo = np.where(cdf < target, mid, lo)
            hi = np.where(cdf >= target, mid, hi)
        distance = (lo + hi) / 2
        pdf = 1 / 3 + a / (3 * c * (a + distance)**2)
        y = np.where(y < .5, distance, 1 - distance)
        y_jac = 1 / pdf
    jac = height_jac * y_jac * TAU**2 * height
    return np.column_stack((tau1, height, u[:, 2], y)), jac


class Proposal:
    """Pilot-trained conditional histogram, frozen before validation.

    Joint adaptation in height and puncture position retains their correlation.
    Every bin retains a positive defensive probability. The twist and remaining
    puncture angle are uniform; the source data show weak thermal variation
    along those angles away from the excluded collision disc.
    """
    def __init__(self, probability):
        self.probability = np.asarray(probability, float)
        self.bins = len(self.probability)
        assert self.probability.shape == (self.bins, self.bins)
        assert np.all(self.probability > 0)
        assert abs(self.probability.sum() - 1) < 1e-12

    @classmethod
    def train(cls, unit, values, bins=8, defensive=.25):
        from scipy.ndimage import convolve1d
        mass = np.histogram2d(unit[:, 0], unit[:, 3], bins=bins,
                             range=((0, 1), (0, 1)), weights=abs(values))[0]
        mass = convolve1d(convolve1d(mass, [.2, .6, .2], axis=0, mode="nearest"),
                          [.2, .6, .2], axis=1, mode="nearest")
        if not mass.sum() > 0:
            raise ValueError("pilot has no thermal signal")
        probability = (1 - defensive) * mass / mass.sum() + defensive / bins**2
        return cls(probability)

    def transform(self, unit):
        u = np.asarray(unit).copy()
        rows = self.probability.sum(axis=1)
        cdf = np.r_[0., np.cumsum(rows)]; cdf[-1] = 1
        i = np.searchsorted(cdf, u[:, 0], side="right") - 1
        u[:, 0] = (i + (u[:, 0] - cdf[i]) / rows[i]) / self.bins
        column = self.probability[i] / rows[i, None]
        ccdf = np.column_stack((np.zeros(len(u)), np.cumsum(column, axis=1)))
        ccdf[:, -1] = 1
        j = np.sum(unit[:, 3, None] >= ccdf, axis=1) - 1
        k = np.arange(len(u))
        u[:, 3] = (j + (unit[:, 3] - ccdf[k, j]) / column[k, j]) / self.bins
        return u, 1 / (self.probability[i, j] * self.bins**2)


def summarize(replicates):
    a = np.asarray(replicates)
    return dict(mean=a.mean(axis=0).tolist(),
                sampling_SE=(a.std(axis=0, ddof=1) / math.sqrt(len(a))).tolist(),
                covariance_of_mean=(np.cov(a.T, ddof=1) / len(a)).tolist(),
                replicates=a.tolist())


def run(tag, power=9, scrambles=8, train_power=10, region="tail", methods=None):
    start = time.monotonic()
    d = Density(tag)
    folder = HERE / "pair" / tag / region
    folder.mkdir(parents=True, exist_ok=True)
    training_path = folder / f"training_p{train_power}.npz"
    if training_path.exists():
        with np.load(training_path) as a:
            unit, values = a["unit"], a["values"]
    else:
        unit = qmc.Sobol(4, scramble=True, seed=927001).random_base2(train_power)
        coords, jac = physical_map(unit, d.R, method="thermal", region=region)
        values = d.evaluate(coords, jac)
        np.savez_compressed(training_path, unit=unit, values=values, coordinates=coords)
    proposal = Proposal.train(unit, values[:, 2])
    proposal_record = dict(R=d.R, probability=proposal.probability.tolist(),
        training_points=len(unit), training_seed=927001, validation_seed=928001,
        training_sha256=hashlib.sha256(training_path.read_bytes()).hexdigest(),
        target="absolute paired thermal correction, no matrix-model input",
        defensive_fraction=.25)
    save(folder / "proposal.json", proposal_record)
    for method in methods or ("legacy", "thermal", "adapted"):
        replicates = []; prefix = []
        for replicate in range(scrambles):
            path = folder / f"{method}_p{power}_r{replicate:02d}.npz"
            if path.exists():
                with np.load(path) as a:
                    values = a["values"]
            else:
                unit = qmc.Sobol(4, scramble=True, seed=928001 + 1009 * replicate).random_base2(power)
                mapped, importance = (proposal.transform(unit) if method == "adapted"
                                      else (unit, np.ones(len(unit))))
                coords, jac = physical_map(mapped, d.R,
                    method="legacy" if method == "legacy" else "thermal", region=region)
                values = d.evaluate(coords, jac * importance)
                np.savez_compressed(path, unit=unit, values=values, coordinates=coords,
                                    importance=importance)
            replicates.append(values.mean(axis=0))
            prefix.append(values[:len(values)//2].mean(axis=0))
            print(json.dumps(dict(tag=tag, region=region, method=method, replicate=replicate,
                correction=float(replicates[-1][2]), seconds=time.monotonic()-start)), flush=True)
        result = dict(tag=tag, R=d.R, region=region, method=method,
            points_per_scramble=2**power, scrambles=scrambles,
            normalization="-2*pi^2*Im(I); genus-one coefficient multiplying g_s^3",
            columns=["zero_temperature", "thermal", "thermal_correction"],
            statistics=summarize(replicates), half_statistics=summarize(prefix),
            paired_doubling_shift=summarize(np.asarray(replicates)-prefix),
            elapsed_seconds=time.monotonic()-start,
            bank_resolver="same nearest log(1+loop decay) anchor for all methods",
            status="moduli sampling comparison at fixed regional momentum quadrature",
            full_amplitude=False, momentum_error_assessed=False,
            omitted_collision_and_finite_tube_remainders_assessed=False,
            source_banks_used=dict(d.counts),
            largest_resolver_distance=max(d.distances, default=0.))
        save(folder / f"{method}_p{power}_s{scrambles}.json", result)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="t075")
    p.add_argument("--power", type=int, default=9)
    p.add_argument("--scrambles", type=int, default=8)
    p.add_argument("--train-power", type=int, default=10)
    p.add_argument("--region", choices=["bulk", "tail"], default="tail")
    p.add_argument("--methods", nargs="+")
    a = p.parse_args()
    run(a.tag, a.power, a.scrambles, a.train_power, a.region, a.methods)
