"""Conservative generic radial second-moment range for the sphere 1->5 chamber.

At a generic isolated outgoing-pair divisor with nonzero factorization
coefficient, the full comb/star mixture density is asymptotic to r**(p-2).
Integrating the nonzero P=0 continuum face with its P²
DOZZ factor gives an importance weight proportional to
r**(2*t²-p) * (-log(r))**(-3/2). Its local second moment therefore contains
integral dr r**(4*t²-p-1) * (-log(r))**(-3), which diverges for p > 4*t².
At equality the logarithm makes this isolated local contribution integrable.
Neither equality nor a strict margin proves global finite variance.
The external BRY leg vanishes at t=1/5, where a separate limiting prescription
is needed. The same conservative range is enforced there without asserting
that the generic nonzero-factorization divergence theorem applies.
"""
from __future__ import annotations

import math

from six_point_radial_sampling_distribution import normalize_radial_proposal


VERSION = "sphere-six-point-pair-second-moment-v2"


def pair_second_moment_diagnostics(t: float, radial_power: float) -> dict:
    """Record the generic obstruction and exceptional scope without an S.E. certificate."""
    t, p = float(t), float(radial_power)
    if not math.isfinite(t) or not 0 < t < 1 / 3:
        raise ValueError("the pair second-moment condition requires 0 < t < 1/3")
    if not math.isfinite(p) or not 0 < p <= 2:
        raise ValueError("radial_power must lie in (0,2]")
    bound = 4 * t**2
    margin = bound - p
    external_leg_zero = t == 1 / 5
    return dict(
        version=VERSION, t=t, radial_power=p,
        necessary_pair_condition="radial_power <= 4*t^2",
        radial_power_upper_bound=bound, radial_power_margin=margin,
        necessary_pair_condition_satisfied=p <= bound,
        outside_conservative_pair_boundary_range=p > bound,
        known_pair_second_moment_divergence=p > bound and not external_leg_zero,
        factorization_assumption="nonzero generic outgoing-pair factorization coefficient",
        exceptional_external_leg_zero=external_leg_zero,
        exceptional_energy_requires_separate_limit=external_leg_zero,
        generic_factorization_assumption_applicable=not external_leg_zero,
        boundary_status="below" if p < bound else "equal" if p == bound else "above",
        importance_weight_radial_exponent=2 * t**2 - p,
        second_moment_log_shell_radial_exponent=margin,
        second_moment_log_shell_logarithmic_exponent=-3,
        global_finite_variance_certified=False,
        standard_error_certified=False,
        scope=("Necessary isolated outgoing-pair condition for the full momentum continuum "
               "assuming a nonzero generic factorization coefficient. At t=1/5 the BRY "
               "external leg vanishes; a separate limiting prescription is required and "
               "the generic divergence assertion is not made. "
               "Equality is locally integrable through (-log r)^-3; satisfying this "
               "condition does not establish global finite variance or validate an empirical S.E."),
    )


def require_pair_second_moment_condition(t: float, radial_power: float) -> dict:
    """Enforce the conservative generic range, without asserting exceptional divergence."""
    diagnostics = pair_second_moment_diagnostics(t, radial_power)
    if diagnostics["outside_conservative_pair_boundary_range"]:
        exception = (
            "; at t=1/5 the exceptional external-leg zero requires a separate limiting "
            "prescription, so the generic divergence assertion does not apply"
            if diagnostics["exceptional_external_leg_zero"] else ""
        )
        raise ValueError(
            f"radial_power={diagnostics['radial_power']!r} is outside the conservative generic "
            f"outgoing-pair second-moment bound 4*t^2={diagnostics['radial_power_upper_bound']!r} "
            f"at t={diagnostics['t']!r}; a valid scramble-error estimate is not established"
            + exception
        )
    return diagnostics


def radial_proposal_second_moment_diagnostics(t: float, proposal) -> dict:
    """Apply the endpoint screen to a density-bounded product mixture."""
    proposal = normalize_radial_proposal(proposal)
    endpoint = float(proposal["reference_endpoint_power"])
    result = pair_second_moment_diagnostics(t, endpoint)
    if proposal["kind"] == "single-power":
        return result
    weight = float(proposal["reference_endpoint_weight"])
    result.update(
        version="sphere-six-point-mixture-second-moment-v1",
        radial_proposal=proposal,
        endpoint_reference_power=endpoint,
        endpoint_reference_weight=weight,
        pointwise_density_lower_bound_relative_to_endpoint_product=weight,
        pointwise_importance_inflation_bound_relative_to_endpoint_product=1 / weight,
        bulk_components_may_exceed_pair_power_bound=True,
        bulk_component_reason=("the positive all-endpoint component controls every radial "
                               "degeneration in the total mixture density"),
        scope=(result["scope"] + " The mixture contains the complete all-endpoint product "
               f"with weight {weight}; its total physical atlas density is therefore at least "
               "that fraction of the endpoint-product density. Component powers are never "
               "screened as if they were standalone proposals."),
    )
    return result


def require_radial_proposal_second_moment_condition(t: float, proposal) -> dict:
    diagnostics = radial_proposal_second_moment_diagnostics(t, proposal)
    if diagnostics["outside_conservative_pair_boundary_range"]:
        raise ValueError(
            f"endpoint reference power={diagnostics['radial_power']!r} is outside "
            f"the conservative outgoing-pair bound 4*t^2={diagnostics['radial_power_upper_bound']!r}"
        )
    return diagnostics


__all__ = [
    "pair_second_moment_diagnostics", "require_pair_second_moment_condition",
    "radial_proposal_second_moment_diagnostics",
    "require_radial_proposal_second_moment_condition",
]
