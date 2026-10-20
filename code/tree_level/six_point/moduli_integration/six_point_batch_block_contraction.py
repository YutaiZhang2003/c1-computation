"""Shared finite-sum contractions for sphere six-point coefficient banks.

This evaluator does not generate coefficients or certify the momentum rule.
It retains analytic continuation: both chiral factors use the same C, even
when C is complex. Geometry batches and order prefixes share contractions.
"""
from __future__ import annotations

import numpy as np


class SphereBatchEvaluator:
    def __init__(self, arrays):
        self.coefficients = np.asarray(arrays.coefficient_matrix)
        self.keys = np.asarray(arrays.coefficient_keys, dtype=int)
        self.primary = np.asarray(arrays.primary_exponents)
        self.logs = np.asarray(arrays.log_weighted_structure_constants)
        self.momenta = np.asarray(arrays.p_values)
        n = len(self.momenta)
        k = len(self.keys)
        if (n == 0 or self.momenta.shape != (n, 3) or self.keys.shape != (k, 3)
                or self.coefficients.shape != (n, k) or self.primary.shape != (n, 3)
                or self.logs.shape != (n,) or k == 0):
            raise ValueError("invalid finite sphere momentum bank")
        if not all(np.isfinite(a).all() for a in
                   (self.coefficients, self.keys, self.primary, self.logs, self.momenta)):
            raise ValueError("nonfinite coefficient bank")
        degrees = self.keys.sum(axis=1)
        if np.any(self.keys < 0) or np.any(np.diff(degrees) < 0):
            raise ValueError("nonnegative degree-ordered coefficient keys required")
        if len(set(map(tuple, self.keys))) != k or tuple(self.keys[0]) != (0, 0, 0):
            raise ValueError("duplicate keys or missing primary coefficient")
        self.degrees = degrees
        self.maximum_order = int(degrees.max())
        # The bank is a complete total-degree table, not an isolated shell.
        expected = {(i, j, l) for i in range(self.maximum_order + 1)
                    for j in range(self.maximum_order + 1 - i)
                    for l in range(self.maximum_order + 1 - i - j)}
        if set(map(tuple, self.keys)) != expected:
            raise ValueError("incomplete coefficient prefix")

    def evaluate(self, q_values, *, base_logarithms=0j, orders=None, zero_axes=(), batch_size=16):
        q = np.asarray(q_values, dtype=complex)
        if q.ndim == 1:
            q = q[None, :]
        if q.ndim != 2 or q.shape[1] != 3 or not np.isfinite(q).all() or np.any(np.abs(q) == 0):
            raise ValueError("finite nonzero native q triples required")
        count = len(q)
        base = np.broadcast_to(np.asarray(base_logarithms, dtype=complex), (count,))
        if not np.isfinite(base).all():
            raise ValueError("nonfinite geometry factor")
        orders = tuple(sorted(set((self.maximum_order,) if orders is None else map(int, orders))))
        if not orders or min(orders) < 0 or max(orders) > self.maximum_order:
            raise ValueError("requested order outside the coefficient bank")
        zero_axes = tuple(sorted(set(map(int, zero_axes))))
        if any(axis not in (0, 1, 2) for axis in zero_axes):
            raise ValueError("invalid face axis")
        if int(batch_size) != batch_size or batch_size <= 0:
            raise ValueError("positive integral batch size required")
        result = np.empty((count, len(orders)), dtype=complex)
        for start in range(0, count, int(batch_size)):
            stop = min(start + int(batch_size), count)
            qs = q[start:stop]
            width = len(qs)
            powers = np.prod(qs[:, None, :] ** self.keys[None, :, :], axis=2).T
            monomials = np.empty((len(self.keys), 2 * width), dtype=complex)
            monomials[:, ::2] = powers
            monomials[:, 1::2] = powers.conjugate()
            logarithms = (self.logs[:, None] + base[start:stop][None, :]
                          + np.einsum("pk,qk->pq", self.primary, np.log(np.abs(qs)), optimize=False))
            with np.errstate(over="raise", invalid="raise"):
                weights = np.exp(logarithms)
            chiral = np.zeros((len(self.momenta), 2 * width), dtype=complex)
            previous = 0
            for index, order in enumerate(orders):
                end = int(np.searchsorted(self.degrees, order, side="right"))
                if zero_axes:
                    keep = np.all(self.keys[previous:end, zero_axes] == 0, axis=1)
                    columns = np.flatnonzero(keep) + previous
                    matrix, monomial = self.coefficients[:, columns], monomials[columns]
                else:
                    matrix, monomial = self.coefficients[:, previous:end], monomials[previous:end]
                if matrix.shape[1]:
                    chiral += np.einsum("pk,ks->ps", matrix, monomial, optimize=True)
                values = np.sum(weights * chiral[:, ::2] * chiral[:, 1::2], axis=0)
                if not np.isfinite(values).all():
                    raise FloatingPointError("nonfinite finite-bank contraction")
                result[start:stop, index] = values
                previous = end
        return {order: result[:, index] for index, order in enumerate(orders)}


def evaluate_geometry_plan(plan, *, t, banks, orders, batch_size=16):
    """Return pointwise matched order values, grouped by graph and incoming slot.

    `banks` maps (topology, slot) to SixPointMomentumArrays or already-bound
    SphereBatchEvaluators. Keeping pointwise values permits paired error and
    boundary diagnostics without rebuilding geometry or lower-order banks.
    """
    plan.require_sources()
    orders = tuple(sorted(set(map(int, orders))))
    if not orders:
        raise ValueError("at least one block order required")
    n = len(plan.arrays["points"])
    result = {order: np.empty(n, dtype=complex) for order in orders}
    topology = plan.arrays["topology"]
    slots = np.argmax(plan.arrays["ordering"] == 0, axis=1)
    base = plan.base_logarithms(t)
    for ti, name in enumerate(("comb", "star")):
        for slot in range(6):
            indices = np.flatnonzero((topology == ti) & (slots == slot))
            if not len(indices):
                continue
            bank = banks[(name, slot)]
            evaluator = bank if isinstance(bank, SphereBatchEvaluator) else SphereBatchEvaluator(bank)
            values = evaluator.evaluate(plan.arrays["q"][indices], base_logarithms=base[indices],
                                        orders=orders, batch_size=batch_size)
            for order in orders:
                result[order][indices] = values[order]
    return result
