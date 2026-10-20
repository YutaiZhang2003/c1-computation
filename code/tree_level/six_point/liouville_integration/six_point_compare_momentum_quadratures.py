#!/usr/bin/env python3
"""Compare explicit momentum rules on identical frozen sphere geometry.

The named rules need not be a Gaussian width comparison or a monotone tensor
ladder. Only one bank is loaded at a time. This module never constructs nodes,
coefficients, or DOZZ factors. Anchor qualification is a geometric scope screen,
not a bound on momentum, block, or moduli error.

Configuration keys: energies, orders, baseline, variants, sample_design,
base_seed, geometry_plans, maximum_per_axis_decay_ratio, batch_size, families.
Optional collar_routing selects explicit-two-pair-collar-v1: all boundary
strata are marked from original source geometry, the tested patch bypasses
native bank contractions, and separately named hybrid arrays contain its
elliptic core, Ward remainder and matched analytic radial addback.
Optional boundary_treatment requires collar_routing and tests same-node native
prefixes before conditionally completing other slow residual blocks. Original
native arrays and separate block/momentum flags remain visible in the output.
Each family has name/t/topology/incoming_slot/anchor_decays and a variants map
from every configured name to a bank path (or a path/fingerprint descriptor).
Paths are relative to the configuration file. sample_design is
sobol-full-net, stratified-sobol-radial-mixture, or selected-qualification; the
last never produces an integral estimate or a scramble standard error.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import platform
import runpy
import sys
import time

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parents[4] / "run.py"))["bootstrap"]()

import numpy as np
import scipy
from scipy.stats import qmc

from six_point_batch_block_contraction import SphereBatchEvaluator
from six_point_select_integration_channels import normalize_routing
from six_point_sample_integration_geometry import SphereGeometryPlan, point_digest
from six_point_momentum_banks_and_reweighting import (
    MOMENTUM_BANK_VERSION, SphereSixPointMomentumBank, _provider_metadata,
)
from six_point_check_sampling_variance import (
    pair_second_moment_diagnostics, require_pair_second_moment_condition,
    radial_proposal_second_moment_diagnostics,
    require_radial_proposal_second_moment_condition,
)
from six_point_radial_sampling_distribution import (
    normalize_radial_proposal, stratified_sobol_points,
)

VERSION = "sphere-six-point-named-variants-v1"
FULL_SAMPLE_DESIGNS = {"sobol-full-net", "stratified-sobol-radial-mixture"}


def is_full_integral_design(name):
    return name in FULL_SAMPLE_DESIGNS


def plan_radial_proposal(plan):
    metadata = plan.metadata
    return normalize_radial_proposal(metadata.get("radial_proposal"),
        radial_power=None if "radial_proposal" in metadata else metadata["radial_power"])


def sampling_diagnostic(t, plan, *, require):
    proposal = plan_radial_proposal(plan)
    function = (require_radial_proposal_second_moment_condition if require
                else radial_proposal_second_moment_diagnostics)
    return function(t, proposal)


def _digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _path(base, value):
    return (base / value).resolve()


def _integer(value, label, minimum=0):
    if isinstance(value, bool) or int(value) != value or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return int(value)


def _pair(value):
    return [float(complex(value).real), float(complex(value).imag)]


def _statistics(point_values, *, full_net):
    """Input shape (replicate, point); never compute a partial-sample mean."""
    values = np.asarray(point_values, complex)
    if values.ndim != 2 or not values.size or not np.isfinite(values).all():
        return dict(complete=False, mean=None, standard_error=None)
    replicate_means = values.mean(axis=1)
    result = dict(complete=True, mean=_pair(replicate_means.mean()),
                  replicate_means=[_pair(v) for v in replicate_means],
                  standard_error=None, full_sample_denominator=values.shape[1])
    if full_net and len(replicate_means) > 1:
        result["standard_error"] = [
            float(replicate_means.real.std(ddof=1) / math.sqrt(len(replicate_means))),
            float(replicate_means.imag.std(ddof=1) / math.sqrt(len(replicate_means)))]
    result["interpretation"] = ("paired scramble estimate; global finite variance and total accuracy unproved"
        if full_net else "average over the supplied selected points; not an integral estimate")
    return result


@dataclass(frozen=True)
class NamedFamily:
    name: str
    t: float
    topology: str
    incoming_slot: int
    anchor_decays: tuple
    banks: dict
    metadata: dict


def _load_families(records, base, energies, variants, orders):
    """Read/hash metadata only; do not retain all large coefficient matrices."""
    families, files, provider = [], {}, None
    for record in records:
        name = record["name"]
        if not isinstance(name, str) or not name or name in {f.name for f in families}:
            raise ValueError("distinct nonempty family names required")
        t, topology = float(record["t"]), record["topology"]
        slot = _integer(record["incoming_slot"], "incoming_slot")
        anchor = tuple(float(x) for x in record["anchor_decays"])
        if (t not in energies or topology not in {"comb", "star"} or slot > 5
                or len(anchor) != 3 or any(not math.isfinite(x) or x <= 0 for x in anchor)):
            raise ValueError("family has invalid physical identity or anchor decays")
        if set(record["variants"]) != set(variants):
            raise ValueError("every family must explicitly supply every named variant")
        banks, metadata = {}, {}
        for variant in variants:
            entry = record["variants"][variant]
            descriptor = {"path": entry} if isinstance(entry, str) else entry
            path = _path(base, descriptor["path"])
            if not path.is_file():
                raise FileNotFoundError(f"missing explicit bank for {name}/{variant}: {path}")
            files[str(path)] = _digest(path)
            with np.load(path, allow_pickle=False) as payload:
                meta = json.loads(payload["metadata"].item())
            if (meta.get("version") != MOMENTUM_BANK_VERSION
                    or (meta["t"], meta["topology"], meta["incoming_slot"]) != (t, topology, slot)
                    or meta["order"] < max(orders)):
                raise ValueError("bank physical identity or complete block prefix mismatch")
            policy = _provider_metadata(meta["provider"])
            if provider is None:
                provider = policy
            if policy != provider:
                raise ValueError("named banks mix coefficient provider/verification policies")
            proposal = meta["proposal"]
            if (tuple(proposal.get("anchor_decays", ())) != anchor
                    or proposal.get("complete_rule", True) is not True):
                raise ValueError("bank has a different node anchor or incomplete momentum rule")
            if "variant" in proposal and proposal["variant"] != variant:
                raise ValueError("bank proposal variant label differs from its configured role")
            if descriptor.get("fingerprint", meta["fingerprint"]) != meta["fingerprint"]:
                raise ValueError("configured bank fingerprint mismatch")
            # Actual proposal fingerprints require the full arrays and are
            # checked when loading each bank, before its first contraction.
            banks[variant] = dict(path=path, fingerprint=meta["fingerprint"],
                proposal_fingerprint=descriptor.get("proposal_fingerprint"))
            metadata[variant] = meta
        families.append(NamedFamily(name, t, topology, slot, anchor, banks, metadata))
    if not families:
        raise ValueError("at least one complete explicit bank family required")
    return families, files


def _load_plans(config, base):
    design, seed = config["sample_design"], _integer(config["base_seed"], "base_seed")
    if design not in FULL_SAMPLE_DESIGNS | {"selected-qualification"}:
        raise ValueError("unknown sample_design")
    plans, files = [], {}
    for index, record in enumerate(config["geometry_plans"]):
        if _integer(record["replicate"], "replicate") != index:
            raise ValueError("geometry replicate IDs must be consecutive")
        current_seed = _integer(record.get("seed", seed + index), "seed")
        if is_full_integral_design(design) and current_seed != seed + index:
            raise ValueError("geometry seed does not match its declared scramble")
        path = _path(base, record["path"])
        for item in (path, path.with_suffix(".json")):
            files[str(item)] = _digest(item)
        plan = SphereGeometryPlan.load(path, expected_point_sha256=record.get("point_sha256"))
        if plan.metadata["geometry_backend"] != "native":
            raise ValueError("named-variant evaluation requires source-verified native geometry")
        count = len(plan.arrays["points"])
        if is_full_integral_design(design):
            if count < 1 or count & (count - 1):
                raise ValueError("full Sobol plans must contain a power-of-two net")
            if design == "sobol-full-net":
                expected = qmc.Sobol(7, scramble=True, seed=current_seed).random_base2(count.bit_length() - 1)
            else:
                expected, records = stratified_sobol_points(plan_radial_proposal(plan),
                    total_power=count.bit_length() - 1, base_seed=seed, replicate=index)
                if plan.metadata.get("radial_strata") != records:
                    raise ValueError("saved radial strata differ from the deterministic design")
            if point_digest(expected) != plan.metadata["point_sha256"]:
                raise ValueError("saved points are not the complete declared Sobol scramble")
        plans.append(plan)
    if not plans or len({len(p.arrays["points"]) for p in plans}) != 1:
        raise ValueError("equally sized nonempty plans required; no point padding or removal")
    if len({p.metadata["radial_power"] for p in plans}) != 1:
        raise ValueError("geometry plans mix radial powers")
    proposal = plan_radial_proposal(plans[0])
    if any(plan_radial_proposal(p) != proposal for p in plans):
        raise ValueError("geometry plans mix radial proposal densities")
    routing = plans[0].metadata["routing"]
    if any(p.metadata["routing"] != routing for p in plans):
        raise ValueError("geometry plans mix channel routing policies")
    if "routing" in config and normalize_routing(config["routing"]) != routing:
        raise ValueError("saved routing differs from the configured policy")
    for key, actual in (("replicates", len(plans)), ("radial_power", plans[0].metadata["radial_power"])):
        if key in config and config[key] != actual:
            raise ValueError(f"saved geometry differs from configured {key}")
    if "radial_proposal" in config and normalize_radial_proposal(config["radial_proposal"]) != proposal:
        raise ValueError("saved geometry differs from configured radial_proposal")
    if "sobol_power" in config and 2**_integer(config["sobol_power"], "sobol_power") != len(plans[0].arrays["points"]):
        raise ValueError("saved point count differs from configured sobol_power")
    return plans, files


def _sources():
    names = ("six_point_batch_block_contraction", "six_point_sample_integration_geometry", "six_point_stable_puncture_coordinates",
             "six_point_channel_coordinates", "six_point_select_integration_channels", "sphere_mobius_geometry",
             "six_point_momentum_banks_and_reweighting", "six_point_radial_sampling_distribution",
             "six_point_check_sampling_variance", "six_point_radial_sampling_distribution")
    result = {str(Path(sys.modules[name].__file__).resolve()): _digest(sys.modules[name].__file__)
              for name in names}
    result[str(Path(__file__).resolve())] = _digest(__file__)
    return result


def _select_families(plans, energies, families, maximum_ratio, excluded=None):
    topology = np.stack([p.arrays["topology"] for p in plans])
    slot = np.argmax(np.stack([p.arrays["ordering"] for p in plans]) == 0, axis=-1)
    q = np.stack([p.arrays["q"] for p in plans])
    decays = -2 * np.log(abs(q))
    shape = (len(energies), *topology.shape)
    selected, ratios = np.full(shape, -1, int), np.full(shape, np.inf)
    for e, t in enumerate(energies):
        # Strict improvement and sorted names make ties independent of the
        # coefficient values and of variant result/convergence comparisons.
        for index in sorted(range(len(families)), key=lambda i: families[i].name):
            family = families[index]
            if family.t != t:
                continue
            mask = (topology == (0 if family.topology == "comb" else 1)) & (slot == family.incoming_slot)
            per_axis = decays / np.asarray(family.anchor_decays)
            ratio = np.max(np.maximum(per_axis, 1 / per_axis), axis=-1)
            better = mask & (ratio < ratios[e])
            selected[e][better], ratios[e][better] = index, ratio[better]
        missing = selected[e] < 0
        if excluded is not None:
            missing &= ~excluded
        if np.any(missing):
            absent = sorted({(int(topology[r, n]), int(slot[r, n]))
                             for r, n in np.argwhere(missing)})
            raise ValueError(f"missing complete energy/channel/slot families at t={t}: {absent}")
    return selected, ratios, ratios <= maximum_ratio, topology, slot, q


def summarize(values, energies, variants, baseline, orders, topology, slot, *, full_net):
    """Full-denominator signed/absolute comparisons and regional contributions."""
    baseline_index = variants.index(baseline)
    records = []
    for e, t in enumerate(energies):
        factor = -1 / (40 * math.pi**3 * t**6) if full_net else 1.
        scaled = values[e] * factor
        record = dict(t=t, units="Q5" if full_net else "I6 importance-weight samples",
            normalization="-I6/(40*pi^3*t^6)" if full_net else None,
            variants={}, differences_from_baseline={}, block_prefix_differences={}, regions={})
        for v, name in enumerate(variants):
            record["variants"][name] = {str(order): _statistics(scaled[:, :, v, o], full_net=full_net)
                                            for o, order in enumerate(orders)}
            record["differences_from_baseline"][name] = {}
            for o, order in enumerate(orders):
                difference = scaled[:, :, v, o] - scaled[:, :, baseline_index, o]
                record["differences_from_baseline"][name][str(order)] = dict(
                    signed=_statistics(difference, full_net=full_net),
                    absolute=_statistics(abs(difference), full_net=full_net))
            record["block_prefix_differences"][name] = {}
            for o in range(1, len(orders)):
                difference = scaled[:, :, v, o] - scaled[:, :, v, o - 1]
                record["block_prefix_differences"][name][f"{orders[o]}-{orders[o-1]}"] = dict(
                    signed=_statistics(difference, full_net=full_net),
                    absolute=_statistics(abs(difference), full_net=full_net))
        for graph in range(2):
            for incoming in (None, *range(6)):
                mask = topology == graph
                if incoming is not None:
                    mask &= slot == incoming
                name = ("comb", "star")[graph] + ("" if incoming is None else f"_slot{incoming}")
                region = dict(sample_count=int(mask.sum()), full_sample_denominator=mask.size,
                              contributions={}, differences_from_baseline={})
                for v, variant in enumerate(variants):
                    region["contributions"][variant] = {}
                    region["differences_from_baseline"][variant] = {}
                    for o, order in enumerate(orders):
                        contribution = np.where(mask, scaled[:, :, v, o], 0.)
                        delta = np.where(mask, scaled[:, :, v, o] - scaled[:, :, baseline_index, o], 0.)
                        region["contributions"][variant][str(order)] = _statistics(contribution, full_net=full_net)
                        region["differences_from_baseline"][variant][str(order)] = dict(
                            signed=_statistics(delta, full_net=full_net),
                            absolute=_statistics(abs(delta), full_net=full_net))
                record["regions"][name] = region
        records.append(record)
    return records


def run_variants(config_path, output_directory, *, input_manifest=None):
    config_path, output = Path(config_path).resolve(), Path(output_directory).resolve()
    if output.exists():
        raise FileExistsError("preserve previous results; output directory already exists")
    started = time.perf_counter()
    config_hash = _digest(config_path)
    config = json.loads(config_path.read_text())
    energies = tuple(float(t) for t in config["energies"])
    orders = tuple(_integer(n, "block prefix") for n in config["orders"])
    variants, baseline = list(config["variants"]), config["baseline"]
    ratio = float(config["maximum_per_axis_decay_ratio"])
    batch_size = _integer(config.get("batch_size", 16), "batch_size", 1)
    if (not energies or len(set(energies)) != len(energies) or any(not 0 < t < 1/3 for t in energies)
            or not orders or list(orders) != sorted(set(orders))
            or not variants or any(not isinstance(v, str) or not v for v in variants)
            or len(set(variants)) != len(variants) or baseline not in variants
            or not math.isfinite(ratio) or ratio < 1):
        raise ValueError("invalid energies, prefixes, variant labels, baseline, or anchor qualification ratio")
    boundary_config = None
    if 'boundary_treatment' in config:
        if 'collar_routing' not in config:
            raise ValueError('boundary_treatment requires explicit collar_routing')
        if len(orders) < 3:
            raise ValueError('conditional boundary treatment requires at least three native prefixes')
        import six_point_evaluate_routed_geometry_shards as boundary
        from six_point_measure_block_convergence import evaluate_native_prefixes
        boundary_config = boundary.settings(config['boundary_treatment'])
    plans, files = _load_plans(config, config_path.parent)
    full_net = is_full_integral_design(config["sample_design"])
    sampling = [sampling_diagnostic(t, plans[0], require=full_net) for t in energies]
    families, bank_files = _load_families(config["families"], config_path.parent, energies, variants, orders)
    files.update(bank_files)
    marks, collar = None, None
    if 'collar_routing' in config:
        import six_point_estimate_collar_corrected_integrand as routed
        marks = routed.mark_plans(plans,config['collar_routing'])
    sources = _sources()
    if marks is not None:
        sources.update(routed.source_hashes())
    if boundary_config is not None:
        sources.update(boundary.source_hashes())
    manifest = dict(version=VERSION, config_sha256=config_hash, config=config, files=files, sources=sources,
        runtime=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
        banks={family.name: family.metadata for family in families})
    if input_manifest is not None:
        expected = json.loads(Path(input_manifest).read_text())
        for key in ("config_sha256", "files", "sources"):
            if expected.get(key) != manifest[key]:
                raise ValueError(f"{key} differs from the frozen input manifest")
    manifest["identity"] = _json_hash(manifest)
    selected, ratios, qualified, topology, slot, q = _select_families(plans, energies, families, ratio,
        excluded=None if marks is None else marks['inside'])
    if marks is not None:
        collar = routed.evaluate_marked(plans,energies,marks,config['collar_routing'])
    count = len(plans[0].arrays["points"])
    raw = np.full((*selected.shape, len(variants), len(orders)), complex(np.nan, np.nan))
    if boundary_config is not None:
        increment_shape = (*raw.shape[:-1],len(orders)-1)
        native_increments = np.full(increment_shape,np.nan)
        native_absolute_increments = np.full(increment_shape,np.nan)
        native_denominator = np.full(raw.shape[:-1],np.nan)
        native_zero_weight_counts = np.full(raw.shape[:-1],-1,dtype=np.int64)
    base_logs = np.stack([np.stack([p.base_logarithms(t) for p in plans]) for t in energies])
    def evaluate_selected(evaluator,e,r,n,v,width):
        r,n = np.atleast_1d(r),np.atleast_1d(n)
        if boundary_config is None:
            result = evaluator.evaluate(q[r,n],base_logarithms=base_logs[e,r,n],orders=orders,batch_size=width)
        else:
            diagnostics = evaluate_native_prefixes(evaluator,q[r,n],base_logarithms=base_logs[e,r,n],
                orders=orders,batch_size=width)
            result = diagnostics['values']
            for o,pair in enumerate(zip(orders[:-1],orders[1:])):
                native_increments[e,r,n,v,o] = diagnostics['relative_absolute_weighted_increments'][pair]
                native_absolute_increments[e,r,n,v,o] = diagnostics['absolute_weighted_increments'][pair]
            native_denominator[e,r,n,v] = diagnostics['normalization_denominator']
            native_zero_weight_counts[e,r,n,v] = diagnostics['zero_absolute_weight_node_counts']
        for o,order in enumerate(orders):
            raw[e,r,n,v,o] = result[order]
    failures, bank_diagnostics = [], {}
    for index, family in enumerate(families):
        e = energies.index(family.t)
        indices = np.argwhere((selected[e] == index) & qualified[e])
        if marks is not None:
            indices = np.asarray([(r,n) for r,n in indices if not marks['inside'][r,n]],int).reshape(-1,2)
        bank_diagnostics[family.name] = {}
        for v, variant in enumerate(variants):
            descriptor = family.banks[variant]
            # Validate complete coefficients/content even for an unused bank.
            bank = SphereSixPointMomentumBank.load(descriptor["path"],
                expected_fingerprint=descriptor["fingerprint"],
                expected_provider_metadata=family.metadata[variant]["provider"])
            if (descriptor["proposal_fingerprint"] is not None
                    and bank.proposal.fingerprint != descriptor["proposal_fingerprint"]):
                raise ValueError("configured proposal fingerprint mismatch")
            evaluator = SphereBatchEvaluator(bank.as_momentum_arrays())
            bank_diagnostics[family.name][variant] = dict(bank_fingerprint=bank.fingerprint,
                proposal_fingerprint=bank.proposal.fingerprint, momentum_rows=len(bank.p_values),
                actual_tensor_orders=bank.proposal.metadata.get("orders"),
                selected_qualified_points=len(indices), provenance=bank.provenance)
            for begin in range(0, len(indices), batch_size):
                batch = indices[begin:begin + batch_size]
                r, n = batch.T
                try:
                    evaluate_selected(evaluator,e,r,n,v,batch_size)
                except (FloatingPointError, OverflowError):
                    # Locate arithmetic failures, retaining the same rule and
                    # every original sample. Never substitute another family.
                    for replicate, point in batch:
                        try:
                            evaluate_selected(evaluator,e,replicate,point,v,1)
                        except (FloatingPointError, OverflowError) as error:
                            failures.append(dict(t=family.t, replicate=int(replicate), point=int(point),
                                                 family=family.name, variant=variant, reason=str(error)))
            del evaluator, bank
        print(f"family={family.name} qualified_points={len(indices)} variants={len(variants)}", flush=True)
    finite = np.isfinite(raw).all(axis=(-1, -2))
    summary = dict(version=VERSION, identity=manifest["identity"], sample_design=config["sample_design"],
        complete=bool(qualified.all() and finite.all()), all_samples_retained=True,
        sample_count_per_plan=count, plan_count=len(plans), energies=list(energies), orders=list(orders),
        variants=variants, baseline=baseline, bank_family_names=[f.name for f in families],
        maximum_per_axis_decay_ratio=ratio, anchor_qualification_is_error_bound=False,
        out_of_scope_point_count=int((~qualified).sum()), nonfinite_point_count=int((~finite).sum()),
        maximum_observed_decay_ratio=float(ratios.max()) if np.isfinite(ratios).all() else None,
        numerical_failures=failures,
        full_CFT_certified=False, global_finite_variance_certified=False, total_accuracy_certified=False,
        sampling_diagnostics=sampling, bank_diagnostics=bank_diagnostics,
        point_hashes=[p.metadata["point_sha256"] for p in plans],
        scope=("Complete fixed-point named-rule comparisons; unresolved points suppress complete means. "
               "Finite prefix increments and anchor qualification do not bound uncomputed tails."),
        statistics=summarize(raw, energies, variants, baseline, orders, topology, slot, full_net=full_net),
        seconds=time.perf_counter() - started)
    extra_arrays = {}
    treatment = None
    if boundary_config is not None:
        treatment = boundary.apply_treatment(plans,energies,families,selected,qualified,raw,
            native_increments,marks,config['boundary_treatment'],orders=orders,variants=variants,baseline=baseline)
        completed_orders = np.asarray(treatment['completed_elliptic_orders'],dtype=int)
        if (np.shape(treatment['values']) != raw.shape
                or tuple(completed_orders) != (-1,*boundary_config['elliptic_orders'])
                or np.shape(treatment['completed_values']) != (*raw.shape,len(completed_orders))):
            raise ValueError('boundary treatment returned inconsistent values or completed order labels')
        for key in ('methods','block_prefix_pass','momentum_pass','native_prefix_pass','unresolved_marked_blocks'):
            if np.shape(treatment[key]) != selected.shape:
                raise ValueError(f'boundary treatment returned inconsistent {key} shape')
        if np.shape(treatment['marked']) != selected.shape[1:]:
            raise ValueError('boundary treatment returned inconsistent marked-point shape')
        summary['boundary_treatment'] = dict(policy=boundary.POLICY,settings=treatment['settings'],
            method_counts={str(method):int(np.sum(treatment['methods']==method))
                           for method in np.unique(treatment['methods'])},
            native_prefix_pass_points=int(treatment['native_prefix_pass'].sum()),
            block_prefix_pass_points=int(treatment['block_prefix_pass'].sum()),
            momentum_pass_points=int(treatment['momentum_pass'].sum()),
            marked_points=int(treatment['marked'].sum()),
            completed_elliptic_orders=completed_orders.tolist(),
            completed_minus_one_slot='full mixed native tensor re-expanded in nome at the source native degree; no independent H order',
            completed_nonnegative_slots='H component prefixes with independently varying finite-correction native degree',
            original_native_arrays_preserved=True,absolute_nodewise_prefix_diagnostics=True,
            block_and_momentum_qualification_separate=True,qualification_is_tail_bound=False,
            records=treatment['records'],full_CFT_certified=False,total_accuracy_certified=False)
        extra_arrays.update(boundary_values_I6=treatment['values'],
            boundary_completed_values_I6=treatment['completed_values'],
            boundary_completed_elliptic_orders=completed_orders,boundary_methods=treatment['methods'],
            boundary_block_prefix_pass=treatment['block_prefix_pass'],
            boundary_momentum_pass=treatment['momentum_pass'],boundary_native_prefix_pass=treatment['native_prefix_pass'],
            boundary_unresolved_marked_blocks=treatment['unresolved_marked_blocks'],
            boundary_marked=treatment['marked'],native_relative_absolute_prefix_increments=native_increments,
            native_absolute_prefix_increments_I6=native_absolute_increments,
            native_prefix_normalization_denominator_I6=native_denominator,
            native_zero_absolute_weight_node_counts=native_zero_weight_counts,
            **{f'boundary_momentum_difference_{name}':delta for name,delta in treatment['momentum_differences'].items()})
    if marks is not None:
        routed_full, hybrid = routed.hybridize(raw if treatment is None else treatment['values'],marks,collar)
        hybrid_finite = np.isfinite(hybrid).all(axis=(-1,-2))
        if treatment is None:
            coverage = np.where(marks['inside'][None],collar['prefix_pass'],qualified)
            coverage &= ~marks['unhandled'][None]
        else:
            outside_coverage = qualified & treatment['block_prefix_pass'] & treatment['momentum_pass']
            coverage = np.where(marks['inside'][None],collar['prefix_pass'],outside_coverage)
        coverage &= collar['addback_prefix_pass'][:,None,None]
        if treatment is not None:
            summary['boundary_treatment']['empirical_qualification_complete'] = bool(coverage.all() and hybrid_finite.all())
            summary['boundary_treatment']['unresolved_point_energy_count'] = int((~coverage | ~hybrid_finite).sum())
        summary['complete'] = bool(hybrid_finite.all())
        hybrid_statistics=summarize(hybrid,energies,variants,baseline,orders,topology,slot,full_net=full_net)
        # The addback is a global constant in this representation, not a
        # contribution to each point's selected comb/star chart. Do not report
        # the artificial allocation as physical regional amplitudes.
        for record in hybrid_statistics:
            record.pop('regions')
        summary['collar_routing'] = dict(policy=routed.POLICY,patch=marks['patch'],
            native_contraction_bypassed_points=int(marks['inside'].sum()),
            unhandled_degenerate_points=(int(marks['unhandled'].sum()) if treatment is None else
                int(np.sum(treatment['marked'][None] & ~marks['inside'][None] & ~outside_coverage))),
            marked_points_outside_tested_patch=int(marks['unhandled'].sum()),
            unhandled_degeneration_array_scope='legacy geometric marking outside tested patch; repaired points remain marked',
            routing_scope_and_collar_prefix_checks_pass=bool(coverage.all() and hybrid_finite.all()),
            outside_native_prefix_comparisons_require_separate_assessment=treatment is None,
            complete_finite_hybrid_estimate=bool(hybrid_finite.all()),
            native_order_labels_apply_only_outside_collar=True,
            outside_boundary_completion_may_replace_native_values=treatment is not None,
            collar_elliptic_orders=collar['settings']['elliptic_orders'],
            collar_momentum_profiles=collar['settings']['momentum_profiles'],
            active_collar_elliptic_order=collar['settings']['elliptic_orders'][-1],
            active_collar_momentum_profile=collar['settings']['momentum_profiles'][-1],
            analytic_radial_addback_once_per_integral=True,
            source_mixture_density_unchanged=True,normalization_id=routed.NORMALIZATION,
            original_native_statistics_exclude_collar_and_are_incomplete=bool(marks['inside'].any()),
            full_CFT_certified=False,total_accuracy_certified=False,
            point_and_addback_records=collar['records'],
            addback_momentum_change=collar['addback_momentum_change'].tolist(),
            addback_middle_change=collar['addback_middle_change'].tolist(),
            addback_block_change=collar['addback_block_change'].tolist(),
            statistics=hybrid_statistics,
            direct_routed_statistics=summarize(routed_full,energies,variants,baseline,orders,topology,slot,full_net=full_net))
        extra_arrays.update(hybrid_values_I6=hybrid,routed_full_values_I6=routed_full,
            collar_inside=marks['inside'],unhandled_degeneration=marks['unhandled'],
            routing_empirical_qualified=coverage,collar_addback_I6=collar['addback'],
            collar_addback_lower_x_I6=collar['addback_lower_x'],
            collar_momentum_change=collar['momentum_change'],collar_block_change=collar['block_change'],
            collar_outer_change=collar['outer_change'],
            collar_elliptic_orders=np.asarray(collar['settings']['elliptic_orders']),
            collar_profiles=np.asarray(collar['settings']['momentum_profiles']),
            **{f'collar_{key}_I6':value for key,value in collar['arrays'].items()})
    # Both geometry code and explicit rows are rechecked after all contractions.
    for plan in plans:
        plan.require_sources()
    if (_digest(config_path) != config_hash
            or any(_digest(path) != digest for path, digest in {**files, **sources}.items())):
        raise RuntimeError("input/source changed during evaluation; no mixed-state output written")
    summary['seconds'] = time.perf_counter() - started
    output.mkdir(parents=True, exist_ok=False)
    with (output / "pointwise.npz").open("xb") as handle:
        np.savez_compressed(handle, values_I6=raw,
            differences_from_baseline_I6=raw - raw[:, :, :, variants.index(baseline):variants.index(baseline)+1, :],
            block_prefix_differences_I6=np.diff(raw, axis=-1),
            family_indices=selected, decay_ratios=ratios, qualified=qualified, finite_all_variants=finite,
            points=np.stack([p.arrays["points"] for p in plans]), q=q, topology=topology,
            ordering=np.stack([p.arrays["ordering"] for p in plans]), incoming_slot=slot,
            energies=np.asarray(energies), orders=np.asarray(orders), variants=np.asarray(variants),
            **extra_arrays)
    summary["pointwise_sha256"] = _digest(output / "pointwise.npz")
    if marks is not None:
        with (output/'degeneration_regions.json').open('x') as handle:
            json.dump(dict(patch=marks['patch'],registry=marks['registry'],records=marks['records']),
                      handle,indent=2,allow_nan=False)
            handle.write('\n')
        summary['collar_routing']['region_marking_sha256'] = _digest(output/'degeneration_regions.json')
    if treatment is not None:
        with (output/'boundary_treatment_records.json').open('x') as handle:
            json.dump(dict(settings=treatment['settings'],records=treatment['records'],
                           completed_elliptic_orders=completed_orders.tolist()),handle,indent=2,allow_nan=False)
            handle.write('\n')
        summary['boundary_treatment']['records_sha256'] = _digest(output/'boundary_treatment_records.json')
    for name, data in (("input_manifest.json", manifest), ("summary.json", summary)):
        with (output / name).open("x") as handle:
            json.dump(data, handle, indent=2, allow_nan=False)
            handle.write("\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path)
    args = parser.parse_args()
    result = run_variants(args.config, args.output, input_manifest=args.input_manifest)
    print(json.dumps({key: result[key] for key in ("complete", "out_of_scope_point_count", "nonfinite_point_count")}, indent=2))


if __name__ == "__main__":
    main()
