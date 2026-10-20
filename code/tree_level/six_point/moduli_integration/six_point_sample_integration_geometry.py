"""Reusable, energy-independent geometry for the complete six-point mixture.

Plans preserve every sample and all 1,440 proposal terms. The coefficient
provider and momentum rule are deliberately absent from the plan identity.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
import itertools
import json
import math
from pathlib import Path

import numpy as np

import six_point_channel_coordinates as atlas
from six_point_select_integration_channels import normalize_routing
from six_point_radial_sampling_distribution import _power_disk_sample
from six_point_radial_sampling_distribution import (
    decode_component, normalize_radial_proposal,
)

FORMAT = "sphere-six-point-geometry-plan-v1"
PAIR_LABELS = tuple(itertools.combinations(range(6), 2))
TOPOLOGIES = ("comb", "star")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for data in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def geometry_sources(backend="native"):
    import sphere_mobius_geometry as mobius
    import six_point_radial_sampling_distribution as radial
    sources = {
        "six_point_sample_integration_geometry.py": digest(__file__),
        "six_point_channel_coordinates.py": digest(atlas.__file__),
        "sphere_mobius_geometry.py": digest(mobius.__file__),
        "power_disk_sampler": hashlib.sha256(inspect.getsource(_power_disk_sample).encode()).hexdigest(),
        "six_point_radial_sampling_distribution.py": digest(radial.__file__),
    }
    if backend == "native":
        from six_point_stable_puncture_coordinates import __file__ as native_file
        from six_point_select_integration_channels import __file__ as routing_file
        sources["six_point_stable_puncture_coordinates.py"] = digest(native_file)
        sources["six_point_select_integration_channels.py"] = digest(routing_file)
    elif backend != "legacy-fused":
        raise ValueError("unknown geometry backend")
    return sources


def point_digest(points):
    values = np.ascontiguousarray(points, dtype="<f8")
    return hashlib.sha256(values.tobytes()).hexdigest()


@dataclass(frozen=True)
class SphereGeometryPoint:
    point: tuple[float, ...]
    topology: str
    ordering: tuple[int, ...]
    q: tuple[complex, complex, complex]
    log_density: float
    pair_logs: tuple[float, ...]
    scale_logs: tuple[float, ...]
    score: float
    convergence_score: float
    recovery: int
    sampled_topology: str
    sampled_ordering: tuple[int, ...]
    sampled_q: tuple[complex, complex, complex]

    @property
    def incoming_slot(self):
        return self.ordering.index(0)

    def base_logarithm(self, t):
        if not 0 < float(t) < 1 / 3:
            raise ValueError("the plan evaluator requires the residue-free imaginary strip")
        energies = (5j * t,) + (-1j * t,) * 5
        weights = (1 - 25 * t * t / 4,) + (1 - t * t / 4,) * 5
        matter = -sum(energies[a] * energies[b] * log
                      for (a, b), log in zip(PAIR_LABELS, self.pair_logs))
        covariance = sum(2 * weights[label] * log
                         for label, log in zip(self.ordering, self.scale_logs))
        return complex(matter + covariance - self.log_density)


def sample_geometry(point, *, radial_power):
    point = tuple(float(x) for x in point)
    if len(point) != 7 or any(not math.isfinite(x) or not 0 <= x <= 1 for x in point):
        raise ValueError("seven finite unit-cube coordinates required")
    if not math.isfinite(radial_power) or not 0 < radial_power <= 2:
        raise ValueError("invalid radial proposal power")
    qs = tuple(_power_disk_sample(point[2 * i], point[2 * i + 1], radial_power)
               for i in range(3))
    orderings = atlas.oriented_comb_orderings()
    selected_index = min(int(point[6] * 2 * len(orderings)), 2 * len(orderings) - 1)
    source_topology = TOPOLOGIES[selected_index // len(orderings)]
    ordering = orderings[selected_index % len(orderings)]
    if source_topology == "comb":
        positions = atlas.linear_channel_positions_by_label(*qs, ordering)
        source = atlas.linear_channel_from_plumbing_coordinates(*qs, ordering)
        fixed, moving = (ordering[0], ordering[4], ordering[5]), (ordering[1], ordering[2], ordering[3])
    else:
        positions = atlas.star_channel_positions_by_label(*qs, ordering)
        source = atlas.star_channel_from_plumbing_coordinates(*qs, ordering)
        fixed, moving = (ordering[0], ordering[2], ordering[5]), (ordering[1], ordering[3], ordering[4])
    density, comb, star = atlas.mixed_atlas_density_and_channels_in_frame(
        positions, fixed_zero=fixed[0], fixed_one=fixed[1], fixed_infinity=fixed[2],
        moving_labels=moving, radial_power=radial_power,
    )
    if not comb:
        raise RuntimeError("no convergent comb channel; match existing selection failure")
    topology, channel = "comb", comb[0]
    if star and star[0].convergence_score < channel.convergence_score:
        topology, channel = "star", star[0]
    recovery = 0
    if any(abs(z) == 0 for z in (channel.q1, channel.q2, channel.q3)):
        if source.convergence_score < 1:
            topology, channel, recovery = source_topology, source, 1
        else:
            candidates = [("comb", c) for c in comb[:90]] + [("star", c) for c in star[:15]]
            candidates.sort(key=lambda item: item[1].convergence_score)
            for candidate_topology, candidate in candidates:
                if all(abs(z) > 0 for z in (candidate.q1, candidate.q2, candidate.q3)):
                    topology, channel, recovery = candidate_topology, candidate, 2
                    break
            else:
                raise RuntimeError("collapsed channel has no convergent alternative")
    pair_logs = []
    for a, b in PAIR_LABELS:
        if positions[a] is None or positions[b] is None:
            pair_logs.append(0.)
        else:
            separation = abs(complex(positions[a]) - complex(positions[b]))
            if separation == 0:
                raise ZeroDivisionError("the sampled chart collapsed punctures")
            pair_logs.append(math.log(separation))
    scale_logs = tuple(math.log(abs(z)) for z in channel.local_scales)
    return SphereGeometryPoint(
        point, topology, channel.ordering, (channel.q1, channel.q2, channel.q3),
        density, tuple(pair_logs), scale_logs, channel.score, channel.convergence_score,
        recovery, source_topology, ordering, qs,
    )


class SphereGeometryPlan:
    """Immutable exact sample geometry; no inherited CFT accuracy claim."""
    FIELDS = {
        "points": (7,), "topology": (), "ordering": (6,), "q": (3,),
        "log_density": (), "pair_logs": (15,), "scale_logs": (6,),
        "score": (), "convergence_score": (), "recovery": (),
        "sampled_topology": (), "sampled_ordering": (6,), "sampled_q": (3,),
    }

    def __init__(self, metadata, arrays):
        if metadata.get("format") != FORMAT or metadata.get("full_CFT_certified") is not False:
            raise ValueError("incorrect geometry plan format/scope")
        if set(arrays) != set(self.FIELDS):
            raise ValueError("incorrect geometry array schema")
        self.metadata = json.loads(json.dumps(metadata, allow_nan=False))
        self.metadata["routing"] = normalize_routing(metadata.get("routing"))
        proposal = normalize_radial_proposal(metadata.get("radial_proposal"),
            radial_power=None if "radial_proposal" in metadata else metadata.get("radial_power"))
        if abs(float(metadata.get("radial_power", proposal["reference_endpoint_power"]))
               - proposal["reference_endpoint_power"]) > 2e-15:
            raise ValueError("geometry radial power disagrees with its proposal")
        if "radial_proposal" in metadata:
            self.metadata["radial_proposal"] = proposal
        if (self.metadata["routing"]["name"] != "absolute-minimum-v1"
                and metadata.get("summation") != "complete-total-degree"):
            raise ValueError("diagonal routing requires complete total-degree sums")
        if (metadata.get("geometry_backend") != "native"
                and self.metadata["routing"]["name"] != "absolute-minimum-v1"):
            raise ValueError("expanded channel routing requires native geometry")
        self.arrays = {k: np.asarray(v) for k, v in arrays.items()}
        n = len(self.arrays["points"])
        if n == 0 or metadata.get("sample_count") != n:
            raise ValueError("empty or incomplete geometry plan")
        for key, shape in self.FIELDS.items():
            a = self.arrays[key]
            if a.shape != (n,) + shape or not np.isfinite(a).all():
                raise ValueError("incorrect/nonfinite geometry array: " + key)
            a.flags.writeable = False
        if point_digest(self.arrays["points"]) != metadata.get("point_sha256"):
            raise ValueError("geometry plan sample hash mismatch")
        if np.any((self.arrays["points"] < 0) | (self.arrays["points"] > 1)):
            raise ValueError("geometry samples outside unit cube")
        if np.any(self.arrays["convergence_score"] >= 1) or np.any(np.abs(self.arrays["q"]) == 0):
            raise ValueError("inadmissible/degenerate selected geometry")
        for key in ("topology", "sampled_topology"):
            if self.arrays[key].dtype.kind not in "iu" or not np.isin(self.arrays[key], (0, 1)).all():
                raise ValueError("unknown topology index")
        for key in ("ordering", "sampled_ordering"):
            if self.arrays[key].dtype.kind not in "iu" or not np.all(np.sort(self.arrays[key], axis=1) == np.arange(6)):
                raise ValueError("invalid puncture permutation")
        if not np.isin(self.arrays["recovery"], (0, 1, 2)).all():
            raise ValueError("unknown recovery decision")

    @classmethod
    def build(cls, points, *, radial_power=None, radial_proposal=None,
              radial_strata=None, backend="native", routing=None):
        points = np.asarray(points, dtype=float)
        policy = normalize_routing(routing)
        proposal = normalize_radial_proposal(radial_proposal, radial_power=radial_power)
        if backend == "native":
            from six_point_stable_puncture_coordinates import sample_native_geometry
            rows = [sample_native_geometry(point, radial_proposal=proposal, routing=policy)
                    for point in points]
        elif backend == "legacy-fused":
            if proposal["kind"] != "single-power":
                raise ValueError("radial mixtures require native geometry")
            if policy["name"] != "absolute-minimum-v1":
                raise ValueError("expanded channel routing requires native geometry")
            legacy_power = proposal["components"][0]["powers"][0]
            rows = [sample_geometry(point, radial_power=legacy_power) for point in points]
        else:
            raise ValueError("unknown geometry backend")
        arrays = {}
        for field in cls.FIELDS:
            values = [getattr(row, "point" if field == "points" else field) for row in rows]
            if field in ("topology", "sampled_topology"):
                values = [TOPOLOGIES.index(value) for value in values]
            dtype = (np.int8 if field in ("topology", "sampled_topology", "ordering", "sampled_ordering", "recovery")
                     else complex if field in ("q", "sampled_q") else float)
            arrays[field] = np.asarray(values, dtype=dtype)
        metadata = dict(format=FORMAT, sample_count=len(rows),
                        radial_power=float(proposal["reference_endpoint_power"]),
                        radial_proposal=proposal,
                        point_sha256=point_digest(points), sources=geometry_sources(backend),
                        geometry_backend=backend, routing=policy,
                        summation="complete-total-degree",
                        full_CFT_certified=False,
                        proposal_count=1440 * len(proposal["components"]))
        if radial_strata is not None:
            if proposal["kind"] != "face-stratified" or not isinstance(radial_strata, list):
                raise ValueError("radial strata require a face-stratified proposal")
            metadata["radial_strata"] = radial_strata
        return cls(metadata, arrays)

    def require_sources(self):
        if self.metadata["sources"] != geometry_sources(self.metadata["geometry_backend"]):
            raise ValueError("geometry source changed; regenerate or independently revalidate this plan")

    def point(self, index):
        a = self.arrays
        return SphereGeometryPoint(
            tuple(a["points"][index]), TOPOLOGIES[int(a["topology"][index])],
            tuple(map(int, a["ordering"][index])), tuple(a["q"][index]),
            float(a["log_density"][index]), tuple(a["pair_logs"][index]), tuple(a["scale_logs"][index]),
            float(a["score"][index]), float(a["convergence_score"][index]), int(a["recovery"][index]),
            TOPOLOGIES[int(a["sampled_topology"][index])], tuple(map(int, a["sampled_ordering"][index])),
            tuple(a["sampled_q"][index]),
        )

    def base_logarithms(self, t):
        return np.asarray([self.point(i).base_logarithm(float(t)) for i in range(len(self.arrays["points"]))])

    def summary(self):
        a = self.arrays
        proposal = normalize_radial_proposal(self.metadata.get("radial_proposal"),
            radial_power=None if "radial_proposal" in self.metadata else self.metadata["radial_power"])
        component_names = [row["name"] for row in proposal["components"]]
        component_counts = {name: 0 for name in component_names}
        for value in a["points"][:, 6]:
            index, _ = decode_component(value, proposal)
            component_counts[component_names[index]] += 1
        return dict(sample_count=len(a["points"]), routing=self.metadata["routing"],
                    radial_proposal_kind=proposal["kind"],
                    radial_component_counts=component_counts,
                    comb_fraction=float(np.mean(a["topology"] == 0)),
                    star_fraction=float(np.mean(a["topology"] == 1)),
                    source_chart_recovery_count=int(np.sum(a["recovery"] == 1)),
                    nondegenerate_channel_recovery_count=int(np.sum(a["recovery"] == 2)),
                    maximum_selected_radius=float(a["score"].max()),
                    maximum_selected_convergence_score=float(a["convergence_score"].max()))

    def save(self, path):
        path = Path(path)
        if path.suffix != ".npz":
            raise ValueError("geometry data must use .npz")
        if path.exists() or path.with_suffix(".json").exists():
            raise FileExistsError("preserve the existing geometry artifact")
        self.require_sources()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            np.savez_compressed(handle, **self.arrays)
        meta = dict(self.metadata, data_sha256=digest(path))
        with path.with_suffix(".json").open("x") as handle:
            json.dump(meta, handle, indent=2, allow_nan=False)
            handle.write("\n")
        return meta

    @classmethod
    def load(cls, path, *, expected_point_sha256=None):
        path = Path(path)
        meta = json.loads(path.with_suffix(".json").read_text())
        if meta.get("data_sha256") != digest(path):
            raise ValueError("geometry data checksum mismatch")
        if expected_point_sha256 is not None and meta.get("point_sha256") != expected_point_sha256:
            raise ValueError("geometry belongs to different samples")
        with np.load(path, allow_pickle=False) as data:
            result = cls(meta, {k: data[k] for k in data.files})
        result.require_sources()
        return result
