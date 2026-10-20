"""Radial source proposals for the sphere six-point atlas.

The endpoint exponent is fixed by factorization, but it need not be used on
all three plumbing axes throughout the unit disks.  A proposal is a finite
mixture of product powers.  Its density is summed over every component before
the usual 1,440-chart atlas average is formed.

The production face mixture deliberately contains the historical all-endpoint
component with weight 1/8.  Consequently its physical density is at least
1/8 of the historical density at every point, including all simultaneous
degenerations.
"""
from __future__ import annotations
import cmath

import math

import numpy as np


VERSION = "sphere-six-point-radial-product-mixture-v1"
SCALAR_KIND = "single-power"
FACE_KIND = "face-stratified"
PATTERNS = ("EEE", "BBB", "EBB", "BEB", "BBE", "EEB", "EBE", "BEE")
FACE_WEIGHTS = {
    "EEE": 1 / 8,
    "BBB": 1 / 2,
    "EBB": 1 / 16,
    "BEB": 1 / 16,
    "BBE": 1 / 16,
    "EEB": 1 / 16,
    "EBE": 1 / 16,
    "BEE": 1 / 16,
}


def _power_disk_sample(
    radial_uniform: float,
    angular_uniform: float,
    radial_power: float,
) -> complex:
    radial_power = float(radial_power)
    if not 0.0 < radial_power <= 2.0:
        raise ValueError("radial_power must lie in (0,2]")
    u = min(
        max(float(radial_uniform), np.nextafter(0.0, 1.0)),
        np.nextafter(1.0, 0.0),
    )
    radius = u ** (1.0 / radial_power)
    angle = 2.0 * math.pi * float(angular_uniform)
    return complex(radius * cmath.exp(1.0j * angle))


def _power(value, label):
    value = float(value)
    if not math.isfinite(value) or not 0 < value <= 2:
        raise ValueError(label + " must lie in (0,2]")
    return value


def _component(name, weight, powers):
    powers = tuple(_power(value, "component power") for value in powers)
    weight = float(weight)
    if (not isinstance(name, str) or not name or len(powers) != 3
            or not math.isfinite(weight) or weight <= 0):
        raise ValueError("a radial component needs a name, positive weight and three powers")
    return {"name": name, "weight": weight, "powers": list(powers)}


def scalar_radial_proposal(power):
    """Return the historical product proposal in the common schema."""
    power = _power(power, "radial power")
    return {
        "version": VERSION,
        "kind": SCALAR_KIND,
        "components": [_component("PPP", 1.0, (power,) * 3)],
        "reference_endpoint_power": power,
        "reference_endpoint_weight": 1.0,
    }


def face_stratified_radial_proposal(t, *, bulk_power=0.75):
    """Endpoint-safe, dyadically stratified proposal for one energy."""
    t = float(t)
    if not math.isfinite(t) or not 0 < t < 1 / 3:
        raise ValueError("face-stratified proposal requires 0 < t < 1/3")
    endpoint = 2 * t * t
    bulk = _power(bulk_power, "bulk power")
    components = []
    for pattern in PATTERNS:
        powers = tuple(endpoint if label == "E" else bulk for label in pattern)
        components.append(_component(pattern, FACE_WEIGHTS[pattern], powers))
    return {
        "version": VERSION,
        "kind": FACE_KIND,
        "components": components,
        "reference_endpoint_power": endpoint,
        "reference_endpoint_weight": FACE_WEIGHTS["EEE"],
        "bulk_power": bulk,
        "weight_policy": "dyadic-face-count-v1",
    }


def normalize_radial_proposal(proposal=None, *, radial_power=None):
    """Validate a JSON proposal or promote one historical scalar power."""
    if proposal is None:
        if radial_power is None:
            radial_power = 0.2
        return scalar_radial_proposal(radial_power)
    if radial_power is not None:
        raise ValueError("specify radial_proposal or radial_power, not both")
    if not isinstance(proposal, dict) or proposal.get("version") != VERSION:
        raise ValueError("unknown radial proposal schema")
    kind = proposal.get("kind")
    if kind not in {SCALAR_KIND, FACE_KIND}:
        raise ValueError("unknown radial proposal kind")
    raw = proposal.get("components")
    if not isinstance(raw, list) or not raw:
        raise ValueError("radial proposal has no components")
    components = []
    names = set()
    for row in raw:
        if not isinstance(row, dict) or set(row) != {"name", "weight", "powers"}:
            raise ValueError("invalid radial component schema")
        component = _component(row["name"], row["weight"], row["powers"])
        if component["name"] in names:
            raise ValueError("radial component names must be distinct")
        names.add(component["name"])
        components.append(component)
    total = math.fsum(row["weight"] for row in components)
    if abs(total - 1.0) > 32 * np.finfo(float).eps:
        raise ValueError("radial component weights must sum to one")
    reference_power = _power(proposal.get("reference_endpoint_power"),
                             "reference endpoint power")
    reference_weight = float(proposal.get("reference_endpoint_weight"))
    if not math.isfinite(reference_weight) or not 0 < reference_weight <= 1:
        raise ValueError("reference endpoint weight must lie in (0,1]")
    matches = [row for row in components
               if tuple(row["powers"]) == (reference_power,) * 3]
    if not matches or math.fsum(row["weight"] for row in matches) + 1e-15 < reference_weight:
        raise ValueError("declared all-endpoint density lower bound is absent")
    result = {
        "version": VERSION,
        "kind": kind,
        "components": components,
        "reference_endpoint_power": reference_power,
        "reference_endpoint_weight": reference_weight,
    }
    for key in ("bulk_power", "weight_policy"):
        if key in proposal:
            result[key] = (_power(proposal[key], "bulk power")
                           if key == "bulk_power" else str(proposal[key]))
    if kind == SCALAR_KIND and (len(components) != 1 or components[0]["weight"] != 1.0
            or len(set(components[0]["powers"])) != 1):
        raise ValueError("single-power proposal must contain one isotropic component")
    return result


def component_intervals(proposal):
    """Return exact cumulative component intervals in the categorical coordinate."""
    proposal = normalize_radial_proposal(proposal)
    result = []
    lower = 0.0
    for index, row in enumerate(proposal["components"]):
        upper = lower + row["weight"]
        if index == len(proposal["components"]) - 1:
            upper = 1.0
        result.append((lower, upper))
        lower = upper
    return tuple(result)


def decode_component(value, proposal):
    """Decode a component and its conditional chart-uniform coordinate."""
    proposal = normalize_radial_proposal(proposal)
    value = min(max(float(value), 0.0), np.nextafter(1.0, 0.0))
    for index, (lower, upper) in enumerate(component_intervals(proposal)):
        if value < upper:
            local = (value - lower) / (upper - lower)
            local = min(max(local, 0.0), np.nextafter(1.0, 0.0))
            return index, local
    raise AssertionError("normalized categorical coordinate was not decoded")


def encode_component(value, component_index, proposal):
    """Embed a conditional chart coordinate into its component interval."""
    proposal = normalize_radial_proposal(proposal)
    index = int(component_index)
    if index != component_index or not 0 <= index < len(proposal["components"]):
        raise ValueError("component index is outside the proposal")
    value = min(max(float(value), 0.0), np.nextafter(1.0, 0.0))
    lower, upper = component_intervals(proposal)[index]
    return lower + (upper - lower) * value


def sample_source(point, proposal, disk_sampler):
    """Map one seven-cube point to a component, chart coordinate and three disks."""
    proposal = normalize_radial_proposal(proposal)
    component_index, chart_uniform = decode_component(point[6], proposal)
    component = proposal["components"][component_index]
    q = tuple(disk_sampler(point[2 * axis], point[2 * axis + 1], power)
              for axis, power in enumerate(component["powers"]))
    return component_index, chart_uniform, q


def mixture_log_radial_density(log_radii, proposal):
    """Log density in three complex disks before chart Jacobians.

    ``log_radii`` has final dimension three.  The scalar branch preserves the
    historical arithmetic grouping for exact replay checks.
    """
    proposal = normalize_radial_proposal(proposal)
    values = np.asarray(log_radii, dtype=float)
    if values.shape[-1] != 3 or not np.isfinite(values).all():
        raise ValueError("three finite logarithmic radii required")
    components = proposal["components"]
    if len(components) == 1 and len(set(components[0]["powers"])) == 1:
        power = components[0]["powers"][0]
        return 3 * math.log(power / (2 * math.pi)) + (power - 2) * values.sum(axis=-1)
    terms = []
    for row in components:
        powers = np.asarray(row["powers"], dtype=float)
        terms.append(math.log(row["weight"])
            + np.sum(np.log(powers / (2 * math.pi)) + (powers - 2) * values, axis=-1))
    terms = np.stack(terms, axis=-1)
    maximum = terms.max(axis=-1)
    return maximum + np.log(np.exp(terms - maximum[..., None]).sum(axis=-1))


def stratified_sobol_points(proposal, *, total_power, base_seed, replicate):
    """Build one deterministic-mixture replicate with dyadic component nets."""
    from scipy.stats import qmc

    proposal = normalize_radial_proposal(proposal)
    total_power, base_seed, replicate = int(total_power), int(base_seed), int(replicate)
    if total_power < 0 or base_seed < 0 or replicate < 0:
        raise ValueError("nonnegative total power, seed and replicate required")
    total = 1 << total_power
    chunks, records = [], []
    endpoint_index = next(i for i, row in enumerate(proposal["components"])
                          if row["name"] in {"EEE", "PPP"})
    for index, row in enumerate(proposal["components"]):
        count_float = row["weight"] * total
        count = int(round(count_float))
        if count <= 0 or abs(count_float - count) > 1e-12 or count & (count - 1):
            raise ValueError("every deterministic mixture allocation must be a positive power of two")
        # Preserve the historical scramble for the all-endpoint component so a
        # nested prefix of old CFT evaluations can be reused exactly.
        seed = (base_seed + replicate if index == endpoint_index else
                base_seed + 1_000_000 + 8 * replicate + index)
        local = qmc.Sobol(7, scramble=True, seed=seed).random_base2(count.bit_length() - 1)
        local[:, 6] = [encode_component(value, index, proposal) for value in local[:, 6]]
        chunks.append(local)
        records.append({"component": row["name"], "component_index": index,
                        "weight": row["weight"], "count": count, "seed": seed})
    points = np.concatenate(chunks, axis=0)
    if len(points) != total:
        raise AssertionError("stratified component counts do not fill the replicate")
    return points, records


def endpoint_lower_bound(proposal):
    proposal = normalize_radial_proposal(proposal)
    return float(proposal["reference_endpoint_weight"])


__all__ = [
    "VERSION", "SCALAR_KIND", "FACE_KIND", "PATTERNS", "FACE_WEIGHTS",
    "scalar_radial_proposal", "face_stratified_radial_proposal",
    "normalize_radial_proposal", "component_intervals", "decode_component",
    "encode_component", "sample_source", "mixture_log_radial_density",
    "stratified_sobol_points", "endpoint_lower_bound",
]
