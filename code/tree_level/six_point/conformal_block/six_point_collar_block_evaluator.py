"""Reusable same-channel collar blocks with complete middle nome coefficients.

This module strips the ordinary native primary monomials. It retains the
middle four-point block and, optionally, every outer descendant of total
degree <=2. The Ward operators currently come from the independently checked
experimental ``validation/sphere_six_point_middle_pillow.py``; the standard
repository bootstrap must make that module importable. This explicit dependency
does not select an alternative coefficient provider or production channel.

Radius limits are declared application scope, not convergence certificates.
Passing these guards certifies neither the block tails nor any integrated
momentum/moduli accuracy. The two coefficient normalizations are kept separate:
the stored complete coefficients already include the entire elliptic prefactor.
The explicitly requested "standard" mode instead retains that prefactor exactly
and truncates H. No automatic representation selection or numerical fallback is
performed; each finite-order representation needs its own scope checks.
"""

from __future__ import annotations

from collections import OrderedDict
import copy
import operator
from typing import Any, Sequence

import mpmath as mp

from sphere_four_point_c25 import FourPointC25
from sphere_four_point_pillow_evaluate import (
    complete_q_coefficients,
    evaluate_complete_q_jet,
    evaluate_pillow_jet,
)



SPHERE_SIX_POINT_COLLAR_BLOCK_VERSION = "complete-middle-q-collar-v1"


def outer_ward_from_jet(jet, z, external, internal, *, central_charge=25,
                        outer_order=2):
    """Apply the checked Ward operators to (f, f', f'') without differentiation.

    The input jet includes the complete middle elliptic prefactor. Derivatives
    are with respect to the original middle sewing coordinate, not the nome.
    A fourth entry (the nome) is allowed, as returned by evaluate_pillow_jet.
    """
    if outer_order not in (0, 1, 2):
        raise ValueError("Only outer total degrees zero, one and two are implemented")
    hl, hm, hr = internal
    d1, d2, d3, d4, d5, d6 = external
    f, derivative, second_derivative = jet[:3]
    if outer_order == 0:
        return {(0, 0): f}
    u = z*derivative
    v = z*derivative + z*z*second_derivative
    beta_l, beta_r = (hl+d2-d1)/(2*hl), (hr+d5-d6)/(2*hr)
    al, bl = hl+d3-hm, hm+d4-hr
    ar, br = hr+d4-hm, hm+d3-hl
    tl = (z-1)*u + (al+bl*z)*f
    tr = (z-1)*u + (ar+br*z)*f
    if outer_order == 1:
        return {(0, 0): f, (1, 0): beta_l*tl, (0, 1): beta_r*tr}
    d_tl = z*u + (z-1)*v + bl*z*f + (al+bl*z)*u
    mixed = beta_l*beta_r*((z-1)*d_tl + (ar+(br-1)*z)*tl + 2*hl*z*f)

    def level_two(hend, hopposite, first, second, near, far):
        gram = mp.matrix([[4*hend+central_charge/2, 6*hend],
                          [6*hend, 4*hend*(2*hend+1)]])
        ward = mp.matrix([hend+2*second-first,
                          (hend+second-first)*(hend+second-first+1)])
        beta = mp.lu_solve(gram, ward)
        alpha = hm-hend-near
        scaling = hopposite-hend-near-far
        t2 = (z*z-1)*u + ((hend+2*near-hm)+(hm+2*far-hopposite)*z*z)*f
        shifted = u+alpha*f
        shifted2 = v+(2*alpha-1)*u+alpha*(alpha-1)*f
        t11 = ((z-1)**2*shifted2 + 2*(1-scaling)*z*(z-1)*shifted
               + scaling*(scaling-1)*z*z*f)
        return beta[0]*t2+beta[1]*t11

    return {(0, 0): f, (1, 0): beta_l*tl, (0, 1): beta_r*tr,
            (2, 0): level_two(hl, hr, d1, d2, d3, d4),
            (0, 2): level_two(hr, hl, d6, d5, d4, d3), (1, 1): mixed}


def _real_weights(values, count, name, *, positive=False):
    result = tuple(map(mp.mpmathify, values))
    if len(result) != count:
        raise ValueError(f"{name} requires {count} weights")
    if any(not mp.isfinite(x) or mp.im(x) for x in result):
        raise ValueError(f"{name} must contain finite real weights")
    result = tuple(mp.re(x) for x in result)
    if positive and any(x <= 0 for x in result):
        raise ValueError(f"{name} must be positive")
    return result


def _remember(cache, key, value, capacity):
    cache[key] = value
    cache.move_to_end(key)
    if len(cache) > capacity:
        cache.popitem(last=False)


class SphereSixPointCollarBlock:
    """Fixed six external and two outer internal weights, variable middle h.

    ``slices(h,x)`` returns a fresh dictionary of coefficients multiplying
    q_left**a*q_right**b, including (0,0), with no outer coordinates inserted.
    ``evaluate(h,(q_left,x,q_right))`` contracts those slices and reports the
    primary-stripped chiral and nonchiral values. Both operations use only the
    analytically grouped exact-c25 provider. Representation choice is explicit:
    "complete" (the default) or "standard", with an optional per-call override.
    """

    def __init__(self, external_weights: Sequence[Any], outer_weights: Sequence[Any],
                 *, order=12, dps=60, max_outer_radius="0.1", max_nome_radius="0.1",
                 cache_size=256, coefficient_bank=None, representation="complete"):
        order, dps, cache_size = map(operator.index, (order, dps, cache_size))
        if not 0 <= order <= 24 or dps < 30 or cache_size < 1:
            raise ValueError("require 0<=order<=24, dps>=30, and cache_size>=1")
        if representation not in ("complete", "standard"):
            raise ValueError("representation must be explicitly 'complete' or 'standard'")
        self.representation = representation
        self.order, self.requested_dps, self.cache_size = order, dps, cache_size
        with mp.workdps(dps):
            self.external_weights = _real_weights(external_weights, 6, "external_weights")
            self.outer_weights = _real_weights(outer_weights, 2, "outer_weights", positive=True)
            self.max_outer_radius = mp.mpf(max_outer_radius)
            self.max_nome_radius = mp.mpf(max_nome_radius)
            if not (0 < self.max_outer_radius < 1 and 0 < self.max_nome_radius < 1):
                raise ValueError("declared outer/nome radius limits must lie between zero and one")
            effective = (self.outer_weights[0], self.external_weights[2],
                         self.external_weights[3], self.outer_weights[1])
            if coefficient_bank is None:
                coefficient_bank = FourPointC25(effective, order=order, dps=dps)
            elif not isinstance(coefficient_bank, FourPointC25):
                raise TypeError("coefficient_bank must be an explicit FourPointC25 instance")
            elif (tuple(coefficient_bank.external_weights) != effective
                  or coefficient_bank.order < order or coefficient_bank.dps < dps
                  or coefficient_bank.requested_dps < dps):
                raise ValueError("coefficient_bank weights, order or precision do not match")
        self.coefficient_bank = coefficient_bank
        self.dps = coefficient_bank.dps
        self.effective_external_weights = effective
        self._complete_cache = OrderedDict()
        self._slice_cache = OrderedDict()
        self._middle_scope_cache = OrderedDict()

    def _middle_weight(self, h):
        return _real_weights((h,), 1, "middle weight", positive=True)[0]

    def complete_coefficients(self, h):
        """Return immutable complete coefficients; reuse only an exact h key.

        The formal product is checked at two arithmetic precisions with the
        same H inputs. This checks transformation arithmetic, not the provider
        formula or the infinite-order tail, which require separate validation.
        """
        with mp.workdps(self.dps):
            h = self._middle_weight(h)
            key = h._mpf_
            if key in self._complete_cache:
                self._complete_cache.move_to_end(key)
                return self._complete_cache[key][0]
            H = self.coefficient_bank.coefficients(h)[:self.order+1]
        with mp.workdps(self.dps+20):
            lower = complete_q_coefficients(self.effective_external_weights, h, H)
        with mp.workdps(self.dps+40):
            upper = complete_q_coefficients(self.effective_external_weights, h, H)
            error = max(abs(a-b)/max(1, abs(b)) for a, b in zip(lower, upper))
            tolerance = mp.power(10, -(self.requested_dps//2))
            if error > tolerance:
                raise ArithmeticError("complete-prefactor transformation failed its precision comparison")
            diagnostics = {
                "coefficient_normalization": "complete primary-stripped middle q series",
                "provider_working_dps": self.dps,
                "transform_working_dps": (self.dps+20, self.dps+40),
                "transform_max_scaled_disagreement": mp.nstr(error, 30),
                "transform_tolerance": mp.nstr(tolerance, 30),
                "provider_root": self.coefficient_bank.coefficient_diagnostics(h),
                "block_tail_certified": False,
            }
        _remember(self._complete_cache, key, (upper, diagnostics), self.cache_size)
        return upper

    def coefficient_diagnostics(self, h):
        self.complete_coefficients(h)
        with mp.workdps(self.dps):
            key = self._middle_weight(h)._mpf_
        return copy.deepcopy(self._complete_cache[key][1])

    def slices(self, h, x, *, elliptic_order=None, outer_order=2, representation=None):
        """Return all outer slices through the requested total degree (0,1,2)."""
        n = self.order if elliptic_order is None else operator.index(elliptic_order)
        outer_order = operator.index(outer_order)
        if not 0 <= n <= self.order or outer_order not in (0, 1, 2):
            raise ValueError("elliptic order exceeds bank, or outer total order is not 0/1/2")
        representation = self.representation if representation is None else representation
        if representation not in ("standard", "complete"):
            raise ValueError("representation must be 'standard' or 'complete'")
        working = max(self.dps, mp.mp.dps)
        with mp.workdps(working):
            # Canonicalize h to the provider's immutable numerical target.
            with mp.workdps(self.dps):
                h = self._middle_weight(h)
            x = mp.mpmathify(x)
            if not mp.isfinite(x):
                raise ValueError("middle coordinate must be finite")
            xkey = (mp.re(x)._mpf_, mp.im(x)._mpf_, working)
            if xkey not in self._middle_scope_cache:
                nome = evaluate_complete_q_jet(x, (1,)).nome
                if abs(nome) > self.max_nome_radius:
                    raise ValueError("middle nome exceeds the explicitly declared collar scope")
                _remember(self._middle_scope_cache, xkey, nome, self.cache_size)
            key = (h._mpf_, xkey, n, outer_order, representation)
            if key in self._slice_cache:
                self._slice_cache.move_to_end(key)
                return dict(self._slice_cache[key])
            if representation == "complete":
                jet = evaluate_complete_q_jet(x, self.complete_coefficients(h), order=n)
            else:
                jet = evaluate_pillow_jet(self.effective_external_weights, h, x,
                    self.coefficient_bank.coefficients(h), central_charge=25, order=n)
            internal = (self.outer_weights[0], h, self.outer_weights[1])
            values = outer_ward_from_jet(jet, x, self.external_weights, internal,
                                         central_charge=25, outer_order=outer_order)
            if any(not mp.isfinite(value) for value in values.values()):
                raise ArithmeticError("nonfinite outer Ward slice")
            immutable = tuple(sorted(values.items()))
            _remember(self._slice_cache, key, immutable, self.cache_size)
            return dict(immutable)

    def evaluate(self, h, q, *, elliptic_order=None, outer_order=2, representation=None):
        """Contract slices with the two actual outer coordinates, retaining all."""
        if len(q) != 3:
            raise ValueError("three native sewing coordinates are required")
        with mp.workdps(max(self.dps, mp.mp.dps)):
            ql, x, qr = tuple(map(mp.mpmathify, q))
            if not all(mp.isfinite(v) for v in (ql, x, qr)):
                raise ValueError("finite sewing coordinates are required")
            if max(abs(ql), abs(qr)) > self.max_outer_radius:
                raise ValueError("outer coordinate exceeds the explicitly declared collar scope")
            representation = self.representation if representation is None else representation
            values = self.slices(h, x, elliptic_order=elliptic_order, outer_order=outer_order,
                                 representation=representation)
            value = mp.fsum(ql**a * qr**b * v for (a, b), v in values.items())
            return {
                "chiral_descendant": value, "nonchiral_descendant": abs(value)**2,
                "outer_slice_values": values,
                "elliptic_order": self.order if elliptic_order is None else operator.index(elliptic_order),
                "outer_total_order": operator.index(outer_order), "central_charge": 25,
                "backend": SPHERE_SIX_POINT_COLLAR_BLOCK_VERSION,
                "recursion_dps": self.dps,
                "representation": representation,
                "coefficient_diagnostics": (self.coefficient_diagnostics(h)
                    if representation == "complete" else {
                        "coefficient_normalization": "standard H with exact elliptic prefactor",
                        "provider_root": self.coefficient_bank.coefficient_diagnostics(h),
                        "block_tail_certified": False}),
                "declared_scope": {
                    "maximum_outer_radius": mp.nstr(self.max_outer_radius, 30),
                    "maximum_nome_radius": mp.nstr(self.max_nome_radius, 30)},
                "rigorous_remainder_bound": False,
                "momentum_or_moduli_integrated": False,
            }
