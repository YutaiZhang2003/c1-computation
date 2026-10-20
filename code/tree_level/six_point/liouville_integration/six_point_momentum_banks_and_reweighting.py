"""Reusable, exactly reweighted momentum banks for sphere six-point CFTs.

Geometry selects among explicitly prepared banks; evaluation never generates
new momenta or coefficients.  Prefixes and OPE faces use the identical stored
nodes, DOZZ factors and measure.  Numerical momentum comparisons are empirical
checks and do not certify block truncation, moduli integration, or an amplitude.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import cmath
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np

from liouville_torus import UpsilonB, log_yin_structure_constant_momentum
from liouville_momentum_quadrature import generalized_gaussian_momentum_rule

MOMENTUM_BANK_VERSION = "sphere-six-point-explicit-momentum-v1"


@dataclass(frozen=True)
class SixPointMomentumArrays:
    """One topology and incoming-slot precomputation."""

    log_weighted_structure_constants: np.ndarray
    primary_exponents: np.ndarray
    coefficient_keys: tuple[tuple[int, int, int], ...]
    coefficient_matrix: np.ndarray
    p_values: np.ndarray


class _FrozenMetadata(dict):
    def _immutable(self, *args, **kwargs):
        raise TypeError("bank metadata is immutable; construct a new proposal or bank")

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = __ior__ = _immutable


def _freeze_metadata(value):
    if isinstance(value, dict):
        return _FrozenMetadata({key: _freeze_metadata(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_metadata(item) for item in value)
    return value


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _array_hash(value):
    array = np.ascontiguousarray(value)
    return hashlib.sha256(str(array.dtype).encode() + str(array.shape).encode()
                          + array.tobytes()).hexdigest()


def _readonly(value, dtype):
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


def _three_positive(values, label):
    values = tuple(float(v) for v in values)
    if len(values) != 3 or not all(math.isfinite(v) and v > 0 for v in values):
        raise ValueError(f"three positive finite {label} required")
    return values


def degree_keys(order):
    if int(order) != order or order < 0:
        raise ValueError("nonnegative integer block order required")
    order = int(order)
    return tuple(sorted(((a, b, c) for a in range(order + 1)
                         for b in range(order + 1 - a)
                         for c in range(order + 1 - a - b)), key=lambda k: (sum(k), *k)))


def external_data(t, incoming_slot):
    t = float(t)
    if not 0 < t < 1 / 3:
        raise ValueError("the direct six-point chamber requires 0<t<1/3")
    if int(incoming_slot) != incoming_slot or not 0 <= incoming_slot < 6:
        raise ValueError("incoming slot must be an integer from zero to five")
    momenta = tuple(2.5j * t if i == incoming_slot else .5j * t for i in range(6))
    return momenta, tuple(1 + p * p for p in momenta)


@dataclass(frozen=True)
class MomentumProposal:
    """Explicit rows and logarithmic weights for dP1*dP2*dP3/pi^3."""
    p_values: np.ndarray
    log_weights: np.ndarray
    metadata: dict
    fingerprint: str = field(init=False)

    def __post_init__(self):
        points = _readonly(self.p_values, float)
        weights = _readonly(self.log_weights, float)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
            raise ValueError("nonempty N by 3 momentum rows required")
        if weights.shape != (len(points),) or not np.all(np.isfinite(weights)):
            raise ValueError("one finite log weight per momentum row required")
        if not np.all(np.isfinite(points)) or np.any(points < 0):
            raise ValueError("real nonnegative finite momentum nodes required")
        metadata = json.loads(json.dumps(self.metadata, allow_nan=False))
        if metadata.get("measure") != "dP1*dP2*dP3/pi^3":
            raise ValueError("proposal must explicitly declare the sewn measure")
        object.__setattr__(self, "p_values", points)
        object.__setattr__(self, "log_weights", weights)
        object.__setattr__(self, "metadata", _freeze_metadata(metadata))
        object.__setattr__(self, "fingerprint", _json_hash(dict(
            metadata=metadata, points=_array_hash(points), weights=_array_hash(weights))))


def explicit_tensor_proposal(rules, *, measure="dP", domain="finite", family=None,
                             metadata=None):
    """Adapt three explicit (nodes,weights) rules without moving any node.

    ``measure`` describes each input weight, and is either dP or dP/pi.
    The adapter includes the three inverse pi factors exactly once.
    """
    if len(rules) != 3 or measure not in {"dP", "dP/pi"}:
        raise ValueError("three rules and an explicit dP or dP/pi input measure required")
    nodes, logs, orders = [], [], []
    for point, weight in rules:
        point, weight = np.asarray(point, float), np.asarray(weight, float)
        if point.ndim != 1 or weight.shape != point.shape or not len(point):
            raise ValueError("each tensor rule needs equal nonempty node/weight vectors")
        if np.any(weight <= 0) or not np.all(np.isfinite(weight)):
            raise ValueError("positive finite tensor weights required")
        nodes.append(point)
        logs.append(np.log(weight) - (math.log(math.pi) if measure == "dP" else 0.))
        orders.append(len(point))
    indices = np.indices(tuple(orders)).reshape(3, -1).T
    points = np.column_stack([nodes[e][indices[:, e]] for e in range(3)])
    weights = sum(logs[e][indices[:, e]] for e in range(3))
    meta = dict(metadata or {})
    meta.update(measure="dP1*dP2*dP3/pi^3", domain=domain, orders=orders,
                scheme=meta.get("scheme", "explicit-tensor"))
    if family is not None:
        meta["refinement_family"] = str(family)
    return MomentumProposal(points, weights, meta)


def gaussian_proposal(anchor_decays, orders, *, scheme="threshold-gaussian",
                      threshold_slopes=None, width_scale=1.0):
    """Create one reusable infinite-domain rule anchored at fixed decays.

    These nodes stay fixed when evaluating another geometry. All Gaussian and
    threshold weights are undone; the complete original density is evaluated.
    """
    decays = np.asarray(_three_positive(anchor_decays, "anchor decays"))
    if len(orders) != 3 or any(int(n) != n or n < 1 for n in orders):
        raise ValueError("three positive integer momentum orders required")
    if not math.isfinite(width_scale) or width_scale <= 0:
        raise ValueError("positive finite width scale required")
    slopes = None
    if scheme == "dozz-threshold":
        slopes = np.asarray(threshold_slopes, float)
        if slopes.shape != (3,) or not np.all(np.isfinite(slopes)):
            raise ValueError("three finite DOZZ threshold slopes required")
        effective = np.maximum(.75 * decays, decays - slopes)
        alpha = .5
    elif scheme in {"primary-gaussian", "threshold-gaussian"}:
        effective = decays.copy()
        alpha = -.5 if scheme == "primary-gaussian" else .5
    else:
        raise ValueError("unknown momentum Gaussian scheme")
    effective *= width_scale
    family = dict(scheme=scheme, anchor_decays=decays.tolist(),
                  threshold_slopes=None if slopes is None else slopes.tolist(),
                  width_scale=float(width_scale))
    rules = []
    for a, n in zip(effective, orders):
        shared = generalized_gaussian_momentum_rule(
            .5, int(n), laguerre_alpha=alpha, decay_coefficient=float(a))
        rules.append((shared.nodes, shared.weights))
    return explicit_tensor_proposal(rules, measure="dP/pi", domain="[0,infinity)^3",
        family=_json_hash(family), metadata=dict(**family, effective_decays=effective.tolist(),
             width_family=_json_hash({k: v for k, v in family.items() if k != "width_scale"})))


def _triples(topology, external, points):
    a, b, c, d, e, f = external
    p, q, r = points
    if topology == "comb":
        return ((a, b, p), (p, c, q), (q, d, r), (r, e, f))
    if topology == "star":
        return ((a, b, p), (c, d, q), (r, e, f), (r, q, p))
    raise ValueError("topology must be comb or star")


def dozz_threshold_slopes(t, topology, incoming_slot, *, epsilon=.0025, dps=35):
    """Geometry-independent proposal curvature, with a half-step diagnostic."""
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("positive finite threshold probe required")
    external, _ = external_data(t, incoming_slot)
    special = UpsilonB(1., dps=int(dps))

    def residual(points):
        value = sum(complex(log_yin_structure_constant_momentum(special, *triple))
                    for triple in _triples(topology, external, points))
        return value.real - 2 * np.log(points).sum()

    def slopes(step):
        base = np.full(3, step)
        initial = residual(base)
        values = []
        for edge in range(3):
            shifted = base.copy()
            shifted[edge] *= 2
            values.append((residual(shifted) - initial) / (3 * step * step))
        return np.asarray(values)

    first, second = slopes(epsilon), slopes(epsilon / 2)
    if not np.all(np.isfinite([first, second])):
        raise FloatingPointError("nonfinite DOZZ proposal curvature")
    return second, dict(epsilon=float(epsilon), first=first.tolist(), refined=second.tolist(),
                        maximum_change=float(np.max(abs(second - first))),
                        interpretation="proposal diagnostic; full DOZZ remains in residual")


def _provider_metadata(metadata):
    metadata = json.loads(json.dumps(metadata, allow_nan=False))
    if metadata.get("backend") != "verified-fixed-weight-c" or not metadata.get("version"):
        raise ValueError("verified fixed-weight c-recursion provenance required")
    if metadata.get("h_or_pbw_fallback", False):
        raise ValueError("mixed h/PBW fallback coefficient provenance is not supported")
    if metadata.get("minimum_total_order", 0) != 0:
        raise ValueError("a complete coefficient table is required, not an unassembled shell")
    if (metadata.get("representation") != "native-plumbing-descendant"
            or metadata.get("truncation") != "total-degree"):
        raise ValueError("native-plumbing total-degree coefficient provenance required")
    return metadata


def _primary_exponents(topology, points, external_weights):
    h = 1 + points * points
    d = external_weights
    if topology == "comb":
        constants = (d[0] + d[1], d[0] + d[1] + d[2], sum(d[:4]))
        return 2 * (h - np.asarray(constants)[None, :])
    if topology == "star":
        constants = (d[0] + d[1], d[2] + d[3], d[5] - d[4])
        return 2 * (h - np.asarray(constants)[None, :])
    raise ValueError("topology must be comb or star")


def _complex_logsum(log_values):
    if np.any(np.isnan(log_values)) or np.any(np.isposinf(log_values.real)):
        raise FloatingPointError("nonfinite positive logarithmic CFT density")
    usable = np.isfinite(log_values.real)
    if not np.any(usable):
        return 0j
    top = float(np.max(log_values.real[usable]))
    terms = np.exp(log_values[usable] - top)
    total = complex(math.fsum(terms.real), math.fsum(terms.imag))
    if total == 0j:
        return 0j
    value = top + cmath.log(total)
    if value.real > math.log(np.finfo(float).max):
        raise FloatingPointError("final CFT momentum sum exceeds complex128 range")
    return cmath.exp(value)


@dataclass(frozen=True)
class SphereSixPointMomentumBank:
    t: float
    topology: str
    incoming_slot: int
    order: int
    proposal: MomentumProposal
    coefficient_matrix: np.ndarray
    log_structure_constants: np.ndarray
    provider_metadata: dict
    provenance: dict = field(default_factory=dict)
    coefficient_keys: tuple = field(init=False)
    primary_exponents: np.ndarray = field(init=False)
    log_weighted_structure_constants: np.ndarray = field(init=False)
    fingerprint: str = field(init=False)

    def __post_init__(self):
        _, weights = external_data(self.t, self.incoming_slot)
        keys = degree_keys(self.order)
        coefficients = _readonly(self.coefficient_matrix, complex)
        logs = _readonly(self.log_structure_constants, complex)
        expected = (len(self.proposal.p_values), len(keys))
        if coefficients.shape != expected or not np.all(np.isfinite(coefficients)):
            raise ValueError("finite complete coefficient rows required for this node set/order")
        if not np.allclose(coefficients[:, 0], 1., rtol=0., atol=1e-12):
            raise ValueError("normalized primary coefficient must equal one on every row")
        if (logs.shape != (expected[0],) or np.any(np.isnan(logs))
                or not np.all(np.isfinite(logs.imag)) or np.any(np.isposinf(logs.real))):
            raise ValueError("one finite or threshold-zero DOZZ log per node required")
        metadata = _provider_metadata(self.provider_metadata)
        object.__setattr__(self, "coefficient_matrix", coefficients)
        object.__setattr__(self, "log_structure_constants", logs)
        object.__setattr__(self, "provider_metadata", _freeze_metadata(metadata))
        object.__setattr__(self, "provenance", _freeze_metadata(
            json.loads(json.dumps(self.provenance, allow_nan=False))))
        object.__setattr__(self, "coefficient_keys", keys)
        object.__setattr__(self, "primary_exponents", _readonly(
            _primary_exponents(self.topology, self.proposal.p_values, weights), complex))
        object.__setattr__(self, "log_weighted_structure_constants", _readonly(
            logs + self.proposal.log_weights, complex))
        object.__setattr__(self, "fingerprint", _json_hash(dict(
            version=MOMENTUM_BANK_VERSION, t=float(self.t), topology=self.topology,
            incoming_slot=int(self.incoming_slot), order=int(self.order),
            proposal=self.proposal.fingerprint, provider=metadata,
            coefficients=_array_hash(coefficients), dozz=_array_hash(logs))))

    @property
    def p_values(self):
        return self.proposal.p_values

    def as_momentum_arrays(self):
        if self.proposal.metadata.get("complete_rule", True) is not True:
            raise ValueError("incomplete worker rows cannot be exposed as a full integration rule")
        return SixPointMomentumArrays(self.log_weighted_structure_constants,
            self.primary_exponents, self.coefficient_keys, self.coefficient_matrix, self.p_values)

    def _mask(self, order_cap=None, zero_axes=(), axis_caps=None):
        if order_cap is None:
            order_cap = self.order
        if int(order_cap) != order_cap or not 0 <= order_cap <= self.order:
            raise ValueError("requested prefix is outside the stored complete block order")
        keys = np.asarray(self.coefficient_keys, int)
        mask = keys.sum(axis=1) <= order_cap
        if any(int(e) != e or e not in (0, 1, 2) for e in zero_axes):
            raise ValueError("zero_axes must identify sewing directions zero, one or two")
        for edge in zero_axes:
            mask &= keys[:, edge] == 0
        if axis_caps is not None:
            if len(axis_caps) != 3 or any(int(n) != n or n < 0 for n in axis_caps):
                raise ValueError("three nonnegative axis caps required")
            mask &= np.all(keys <= np.asarray(axis_caps), axis=1)
        return keys[mask], self.coefficient_matrix[:, mask]

    def node_log_values(self, q_values=None, *, log_q_values=None, base_log=0j,
                        order_cap=None, zero_axes=(), axis_caps=None):
        """Full node density, with prefixes/faces evaluated on identical rows.

        A zero axis keeps C_0jk etc. It does not remove its primary propagator.
        Analytically integrated projections use the exposed arrays/exponents.
        """
        if self.proposal.metadata.get("complete_rule", True) is not True:
            raise ValueError("an incomplete momentum build shard is not an integration rule")
        if (q_values is None) == (log_q_values is None):
            raise ValueError("provide exactly one of q_values or log_q_values")
        if log_q_values is None:
            if len(q_values) != 3 or any(not 0 < abs(q) < 1 for q in q_values):
                raise ValueError("three native sewing values with 0<|q|<1 required")
            logs = np.asarray([cmath.log(q) for q in q_values], complex)
        else:
            logs = np.asarray(log_q_values, complex)
            if logs.shape != (3,) or not np.all(np.isfinite(logs)) or np.any(logs.real >= 0):
                raise ValueError("three finite sewing logarithms with negative real parts required")
        keys, coefficients = self._mask(order_cap, zero_axes, axis_caps)
        row_scale = np.max(np.abs(coefficients), axis=1)
        normalized = coefficients / row_scale[:, None]
        monomials = np.exp(keys @ logs)
        anti_monomials = np.exp(keys @ logs.conjugate())
        holomorphic = np.einsum("nk,k->n", normalized, monomials, optimize=False)
        antiholomorphic = np.einsum("nk,k->n", normalized, anti_monomials, optimize=False)
        with np.errstate(divide="ignore", invalid="ignore"):
            block_log = 2 * np.log(row_scale) + np.log(holomorphic) + np.log(antiholomorphic)
        # A zero polynomial contributes exactly zero even if both chiral logs
        # are -infinity; avoid NaN from complex arithmetic with infinities.
        zero = (holomorphic == 0) | (antiholomorphic == 0)
        block_log[zero] = complex(-math.inf, 0.)
        answer = (complex(base_log) + self.log_weighted_structure_constants
                  + np.sum(self.primary_exponents * logs.real[None, :], axis=1) + block_log)
        answer[zero | np.isneginf(self.log_structure_constants.real)] = complex(-math.inf, 0.)
        if np.any(np.isnan(answer)) or np.any(np.isposinf(answer.real)):
            raise FloatingPointError("nonfinite logarithmic bank evaluation")
        return answer

    def evaluate(self, q_values=None, **kwargs):
        return _complex_logsum(self.node_log_values(q_values, **kwargs))

    def evaluate_variants(self, q_values, variants, *, base_log=0j):
        return {name: self.evaluate(q_values, base_log=base_log, **settings)
                for name, settings in variants.items()}

    def save(self, path):
        if self.proposal.metadata.get("complete_rule", True) is not True:
            raise ValueError("save incomplete worker rows with the dedicated shard writer")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = dict(version=MOMENTUM_BANK_VERSION, t=self.t, topology=self.topology,
            incoming_slot=self.incoming_slot, order=self.order, proposal=self.proposal.metadata,
            provider=self.provider_metadata, provenance=self.provenance, fingerprint=self.fingerprint)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent,
                    prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
                temporary = Path(handle.name)
                np.savez_compressed(handle, metadata=json.dumps(metadata), p_values=self.p_values,
                    log_weights=self.proposal.log_weights, coefficients=self.coefficient_matrix,
                    log_dozz=self.log_structure_constants)
            os.replace(temporary, path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

    @classmethod
    def load(cls, path, *, expected_fingerprint=None, expected_provider_metadata=None):
        with np.load(path, allow_pickle=False) as payload:
            metadata = json.loads(payload["metadata"].item())
            if metadata["version"] != MOMENTUM_BANK_VERSION:
                raise ValueError("momentum bank format mismatch")
            proposal = MomentumProposal(payload["p_values"], payload["log_weights"], metadata["proposal"])
            bank = cls(metadata["t"], metadata["topology"], metadata["incoming_slot"], metadata["order"],
                proposal, payload["coefficients"], payload["log_dozz"], metadata["provider"], metadata["provenance"])
        if bank.fingerprint != metadata["fingerprint"] or (
                expected_fingerprint is not None and bank.fingerprint != expected_fingerprint):
            raise ValueError("momentum bank identity/content mismatch")
        if expected_provider_metadata is not None and bank.provider_metadata != expected_provider_metadata:
            raise ValueError("momentum bank coefficient provider mismatch")
        return bank


def build_momentum_bank(t, topology, incoming_slot, proposal, order, *,
                        coefficient_provider=None, coefficient_table=None,
                        coefficient_metadata=None, coefficient_provenance=None,
                        special_dps=35):
    """Build one bank, or attach an independently verified saved c table.

    Saved rows must match the explicit node set exactly. This function never
    interpolates coefficients or substitutes another recursion backend.
    """
    if (coefficient_provider is None) == (coefficient_table is None):
        raise ValueError("provide one coefficient provider or one complete saved table")
    external, weights = external_data(t, incoming_slot)
    keys = degree_keys(order)
    provenance = dict(coefficient_provenance or {})
    if coefficient_provider is not None:
        metadata = _provider_metadata(coefficient_provider.metadata())
        provenance["coefficient_diagnostics_before"] = coefficient_provider.diagnostics()
        rows = []
        for points in proposal.p_values:
            coefficients = coefficient_provider.coefficients(topology, weights,
                tuple(1 + p * p for p in points), order, central_charge=25)
            if set(coefficients) != set(keys):
                raise ValueError("provider did not return the complete requested c-recursion table")
            rows.append([coefficients[k] for k in keys])
        matrix = np.asarray(rows, complex)
        provenance["coefficient_diagnostics"] = coefficient_provider.diagnostics()
    else:
        metadata = _provider_metadata(coefficient_metadata or {})
        if not np.array_equal(np.asarray(coefficient_table.p_values), proposal.p_values):
            raise ValueError("saved coefficient momenta do not match proposal rows exactly")
        supplied = tuple(tuple(int(v) for v in k) for k in coefficient_table.coefficient_keys)
        if len(supplied) != len(set(supplied)) or not set(keys).issubset(supplied):
            raise ValueError("saved coefficient table lacks the requested complete prefix")
        source = np.asarray(coefficient_table.coefficient_matrix)
        matrix = source[:, [supplied.index(k) for k in keys]]
        expected = dict(t=float(t), topology=topology, incoming_slot=int(incoming_slot))
        if any(provenance.get(k) != v for k, v in expected.items()):
            raise ValueError("saved coefficient provenance must match energy, topology and external ordering")
        if not provenance.get("source_fingerprint"):
            raise ValueError("saved coefficient tables need a verified source fingerprint")
    special = UpsilonB(1., dps=int(special_dps))

    @lru_cache(None)
    def dozz(args):
        # This cache turns the comb's middle vertices into two-dimensional
        # tables, instead of recomputing them for every tensor-product row.
        return complex(log_yin_structure_constant_momentum(special, *args))

    try:
        logs = np.asarray([sum(dozz(tuple(triple)) for triple in
                           _triples(topology, external, points)) for points in proposal.p_values])
    finally:
        dozz.cache_clear()
    provenance.update(special_dps=int(special_dps), coefficient_table_reused=coefficient_table is not None)
    return SphereSixPointMomentumBank(float(t), topology, int(incoming_slot), int(order),
        proposal, matrix, logs, metadata, provenance)


class ProposalBankFamily:
    """Explicitly bounded registry: selection never builds a missing bank."""
    def __init__(self, max_banks=32):
        if int(max_banks) != max_banks or max_banks < 1:
            raise ValueError("positive integer bank capacity required")
        self.max_banks, self.banks = int(max_banks), {}

    def add(self, bank):
        if bank.proposal.metadata.get("complete_rule", True) is not True:
            raise ValueError("a proposal bank family cannot contain incomplete worker rows")
        if bank.fingerprint not in self.banks and len(self.banks) >= self.max_banks:
            raise RuntimeError("proposal bank capacity reached; explicit eviction or a new family required")
        self.banks[bank.fingerprint] = bank
        return bank.fingerprint

    def select(self, t, topology, incoming_slot, anchor_decays, *, scheme=None, minimum_order=0):
        target = np.asarray(_three_positive(anchor_decays, "geometry decays"))
        candidates = [b for b in self.banks.values()
            if (b.t, b.topology, b.incoming_slot) == (float(t), topology, int(incoming_slot))
            and b.order >= minimum_order
            and (scheme is None or b.proposal.metadata.get("scheme") == scheme)]
        if not candidates:
            raise KeyError("no prepared bank matches the requested geometry/graph/energy")
        # This nearest-anchor choice is a proposal selector, not a convergence
        # certificate. Unanchored explicit rules remain selectable when no
        # anchored candidate is prepared. Ties use the momentum resolution.
        return min(candidates, key=lambda b: (
            (float(np.linalg.norm(np.log(np.asarray(b.proposal.metadata["anchor_decays"]) / target)))
             if "anchor_decays" in b.proposal.metadata else math.inf),
            -len(b.p_values), b.fingerprint))


def assess_momentum_convergence(banks, alternative_width_bank, q_values, *,
                                rtol=1e-3, atol=0., base_log=0j, variant=None):
    """Require two small order increments and an independent width change.

    All evaluations use the identical block prefix/projection and external
    frame. A pass is only an empirical momentum check at this geometry.
    """
    if len(banks) < 3:
        raise ValueError("at least three momentum resolutions are required")
    if not (math.isfinite(rtol) and rtol > 0 and math.isfinite(atol) and atol >= 0):
        raise ValueError("positive finite rtol and nonnegative finite atol required")
    variant = dict(variant or {})
    high = banks[-1]
    identity = (high.t, high.topology, high.incoming_slot, high.provider_metadata)
    family = high.proposal.metadata.get("refinement_family")
    if family is None:
        raise ValueError("order comparisons require an explicit refinement-family identity")
    previous_orders = None
    previous_proposal = None
    for bank in [*banks, alternative_width_bank]:
        if (bank.t, bank.topology, bank.incoming_slot, bank.provider_metadata) != identity:
            raise ValueError("momentum comparison mixes external data or coefficient providers")
        if bank.order != high.order and "order_cap" not in variant:
            raise ValueError("select a common block prefix before comparing different block orders")
    for bank in banks:
        if bank.proposal.metadata.get("refinement_family") != family:
            raise ValueError("momentum order ladder changes proposal family")
        orders = np.asarray(bank.proposal.metadata.get("orders", []), int)
        if orders.shape != (3,) or (previous_orders is not None and np.any(orders <= previous_orders)):
            raise ValueError("all three momentum orders must increase strictly along the ladder")
        if len(bank.p_values) != int(np.prod(orders)) or (previous_proposal is not None
                and np.array_equal(bank.p_values, previous_proposal.p_values)):
            raise ValueError("momentum order audits require genuinely different tensor node rules")
        previous_orders = orders
        previous_proposal = bank.proposal
    alternate = alternative_width_bank.proposal
    if (alternate.metadata.get("orders") != high.proposal.metadata.get("orders")
            or len(alternate.p_values) != len(high.p_values)
            or alternate.fingerprint == high.proposal.fingerprint
            or np.array_equal(alternate.p_values, high.proposal.p_values)
            or not alternate.metadata.get("width_family")
            or alternate.metadata["width_family"] != high.proposal.metadata.get("width_family")
            or alternate.metadata.get("width_scale") == high.proposal.metadata.get("width_scale")):
        raise ValueError("width audit requires changed Gaussian widths at the finest momentum orders")
    values = [b.evaluate(q_values, base_log=base_log, **variant) for b in banks]
    width_value = alternative_width_bank.evaluate(q_values, base_log=base_log, **variant)
    scale = max(abs(values[-1]), abs(width_value))
    tolerance = atol + rtol * scale
    increments = [abs(b - a) for a, b in zip(values[:-1], values[1:])]
    last_two = increments[-2:]
    width_change = abs(width_value - values[-1])
    roundoff = 64 * np.finfo(float).eps * max(scale, 1e-280)
    failures = []
    if any(change > tolerance for change in last_two):
        failures.append("two_successive_momentum_increments")
    if last_two[-1] > last_two[-2] + roundoff:
        failures.append("momentum_increment_not_decreasing")
    if width_change > tolerance:
        failures.append("alternative_width")
    return dict(momentum_empirical_passed=not failures, unresolved=failures,
        values=[[z.real, z.imag] for z in values], alternative_width_value=[width_value.real, width_value.imag],
        increments=increments, width_change=width_change, tolerance=tolerance,
        bank_fingerprints=[b.fingerprint for b in banks], alternative_width_fingerprint=alternative_width_bank.fingerprint,
        variant=variant, full_cft_certified=False, full_accuracy_certified=False,
        block_truncation_checked=False, moduli_integration_checked=False,
        scope="empirical momentum comparison at one geometry and fixed finite block polynomial")


__all__ = ["MomentumProposal", "SphereSixPointMomentumBank", "ProposalBankFamily",
           "explicit_tensor_proposal", "gaussian_proposal", "dozz_threshold_slopes",
           "build_momentum_bank", "assess_momentum_convergence", "degree_keys",
           "external_data", "MOMENTUM_BANK_VERSION"]
