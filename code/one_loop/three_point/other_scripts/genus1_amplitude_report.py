"""Amplitude assembly diagnostics and numerical error reports."""
from __future__ import annotations

# genus1_amplitude_error_budget
# Final-amplitude accuracy accounting; no per-point relative acceptance gate.
# 
# All values/weights must already carry the unchanged amplitude normalization.
# Paired refinement estimates are empirical indicators, not rigorous remainder
# bounds. The reference scale is derived ONLY from the complete worldsheet sum.
# This version does not mutate historical pointwise acceptance records.


import math
import statistics

POLICY_VERSION = 'final-amplitude-budget-v2'
AMPLITUDE_RTOL = .005
BLOCK_ORDER_CAP = 10
ERROR_CATEGORIES = ('momentum', 'block', 'collar')
SAFETY_GATES = ('chart_valid', 'coefficients_verified', 'arithmetic_safe', 'provenance_verified')


def finite(x):
    if isinstance(x, bool) or not math.isfinite(float(x)):
        raise ValueError('finite real amplitude quantity required')
    return float(x)


def point_record(*, point_id, value, normalized_weight, orders, safety,
                 paired_checks, source_sha256):
    """Keep safe data; defer refinement decisions to their amplitude impact.

paired_checks maps category -> label -> (coarse_value, refined_value), all
on the same geometry. normalized_weight includes the true Jacobian and
per-scramble sample normalization, but NOT division by the scramble count.
Never pass a local relative percentage as an absolute refinement estimate.
"""
    if not point_id or len(source_sha256) != 64 or any(c not in '0123456789abcdef' for c in source_sha256):
        raise ValueError('stable point identity and verified source checksum required')
    if (len(orders) != 3 or any(type(n) is not int or not 0 <= n <= BLOCK_ORDER_CAP for n in orders)):
        raise ValueError('native block ceiling10')
    if set(safety) != set(SAFETY_GATES) or any(type(v) is not bool or not v for v in safety.values()):
        raise ValueError('unsafe CFT data cannot enter the worldsheet estimate')
    value, weight = finite(value), finite(normalized_weight)
    if not set(paired_checks) <= set(ERROR_CATEGORIES): raise ValueError('unknown error category')
    changes = {}
    for category, comparisons in paired_checks.items():
        changes[category] = {}
        for label, pair in comparisons.items():
            if not label or len(pair) != 2: raise ValueError('named paired comparison required')
            difference = weight*(finite(pair[1])-finite(pair[0]))
            changes[category][label] = dict(signed=finite(difference), absolute=finite(abs(difference)))
    return dict(policy_version=POLICY_VERSION, point_id=point_id, orders=list(orders),
        source_sha256=source_sha256, safety=dict(safety), contribution=finite(weight*value),
        weighted_changes=changes, include_in_worldsheet_estimate=True,
        local_relative_refinement_gate=False, refinement_status='deferred_to_final_amplitude_budget',
        cap_limited=10 in orders, rigorous_error_bound=False)


def assess_amplitude(components, *, expected_components, scrambles=16,
                     rtol=AMPLITUDE_RTOL, mc_sigma_multiplier=1.):
    """Assess one frequency, after all normalized strata are assembled.

A component supplies its independent-scramble amplitude estimates and
paired-refinement absolute-contribution estimates on the same scale.
For each category take the largest assessed refinement/width indicator
per component, then sum across components. Include the uncertainty of the
refinement estimator. MCSE remains separately reported; no fake error bars.
The MC multiplier is explicit (default one standard error, not a95% claim).
"""
    rtol, multiplier = finite(rtol), finite(mc_sigma_multiplier)
    if not 0 < rtol < 1 or multiplier < 1 or type(scrambles) is not int or scrambles < 2:
        raise ValueError('invalid amplitude accuracy convention')
    expected = tuple(expected_components)
    if not expected or len(set(expected)) != len(expected): raise ValueError('explicit unique component coverage required')
    rows = list(components); ids = [r['component_id'] for r in rows]
    if len(set(ids)) != len(ids) or set(ids)-set(expected): raise ValueError('duplicate/unknown component')
    missing = sorted(set(expected)-set(ids))
    totals = [0.]*scrambles; incomplete = []; detail = {}; proxy = {k: 0. for k in ERROR_CATEGORIES}
    cap_limited = False
    for row in rows:
        if row.get('policy_version') != POLICY_VERSION or row.get('target_blind') is not True:
            raise ValueError('incompatible policy or target provenance')
        if any(row.get(k) is not True for k in ('sample_coverage_verified', 'safety_verified', 'normalization_verified')):
            raise ValueError('unverified sample coverage/safety/normalization')
        values = [finite(v) for v in row['replicate_amplitudes']]
        if len(values) != scrambles: raise ValueError('wrong independent-scramble coverage')
        totals = [math.fsum((x, y)) for x, y in zip(totals, values)]
        cap_limited |= bool(row.get('cap_limited', False))
        entries = {}
        for category in ERROR_CATEGORIES:
            diagnostic = row.get('error_audits', {}).get(category)
            if not diagnostic or diagnostic.get('contribution_coverage_verified') is not True:
                incomplete.append(row['component_id']+':'+category); continue
            comparisons = diagnostic.get('absolute_replicate_changes', {})
            if not comparisons:
                # Exact absence requires explicit evidence, never an empty0.
                if not diagnostic.get('exact_zero_evidence'):
                    incomplete.append(row['component_id']+':'+category); continue
                entries[category] = 0.; continue
            indicators = []
            for label, raw in comparisons.items():
                sample = [finite(v) for v in raw]
                if not label or len(sample) != scrambles or any(v < 0 for v in sample):
                    raise ValueError('normalized absolute refinement contributions required')
                mean = math.fsum(sample)/scrambles
                se = statistics.stdev(sample)/math.sqrt(scrambles)
                indicators.append(mean+multiplier*se)
            entries[category] = max(indicators)
            proxy[category] += entries[category]
        detail[row['component_id']] = entries
    mean = math.fsum(totals)/scrambles
    mc_se = statistics.stdev(totals)/math.sqrt(scrambles)
    complete = not missing and not incomplete
    budget = rtol*abs(mean) if complete else None
    indicator = multiplier*mc_se+math.fsum(proxy.values()) if complete else None
    passed = bool(complete and abs(mean) > 0 and indicator <= budget)
    return dict(policy_version=POLICY_VERSION, target_blind=True, relative_target=rtol,
        reference_scale_source='complete_corrected_worldsheet_sum' if complete else 'unavailable_until_complete',
        amplitude=mean if not missing else None, partial_worldsheet_sum=mean,
        replicate_amplitudes=totals, mc_standard_error=mc_se, mc_sigma_multiplier=multiplier,
        empirical_systematic_indicators=proxy, component_error_indicators=detail,
        combined_empirical_error_indicator=indicator, absolute_error_budget=budget,
        missing_components=missing, unassessed_error_coverage=incomplete,
        passes_final_amplitude_target=passed, local_relative_refinement_gate=False,
        cap_limited=cap_limited, rigorous_total_error_bound=False,
        qualification='Empirical final-amplitude budget, not a rigorous truncation or confidence bound. Sampling and systematic diagnostics remain separate.')


# genus1_moduli_amplitude_report
# Post-production numerical assessment; the dual target never enters planning.

import argparse
from fractions import Fraction
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode=True
import genus1_moduli_panel_campaign as campaign

PAPER='https://arxiv.org/html/2604.06301v2'


def matrix_amplitude(t):
    """CER torus three-point coefficient in A=-Im(I)/(4*pi) units.

    Equal ray: omega_in=i*t, omega_out1=omega_out2=i*t/2.
    The local reduced-integral convention is I/(4*pi)=F1_ws and
    F1_MM=2*F0*S_hat. See the repository torus-one-to-n note.
    """
    t=Fraction(str(t))
    return t**3*(t-1)*(t-2)*(1+t-t*t/2)/96


def family(term):
    if term.startswith('primary:'):return 'primary'
    if ':pair_face_analytic' in term:return 'pair'
    if ':triple_' in term:return 'projected triple'
    raise ValueError('unknown retained component')


def report(root):
    p=campaign.configure(root);f=campaign.f;d=campaign.checked(root,p)
    r=f.spectrum.read(root/'amplitude.json')
    if r['design_sha256']!=f.fingerprint(d) or not r['complete_domain']:
        raise ValueError('complete amplitude for this design required')
    old=f.spectrum.read(campaign.source/'amplitude_wave_000.json')
    readiness=f.spectrum.read(root/'readiness.json')
    target=matrix_amplitude(p['physics']['t']);value=r['amplitude'];delta=value-float(target)
    families={}
    for name in ('primary','pair','projected triple'):
        before=[c for k,c in old['boundary']['components'].items() if family(k)==name]
        after=[c for k,c in r['components'].items() if family(k)==name]
        families[name]=dict(previous=math.fsum(c['value']['value'][0] for c in before),
            current=math.fsum(c['value']['value'][0] for c in after),
            previous_indicator=math.fsum(c['quadrature_indicator'] for c in before),
            current_indicator=math.fsum(c['quadrature_indicator'] for c in after),
            block_indicator=math.fsum(c['value']['errors']['block'] for c in after))
    q=dict(format='genus1-moduli-amplitude-assessment-v1',amplitude_sha256=f.shared.sha(root/'amplitude.json'),
        analysis_source_sha256=f.shared.sha(Path(__file__)),
        design_sha256=f.fingerprint(d),t=p['physics']['t'],normalization='-Im(I_1,3)/(4*pi)',amplitude=value,
        matrix_model=dict(value=float(target),exact_fraction=str(target),formula='t^3*(t-1)*(t-2)*(1+t-t^2/2)/96',
            source=PAPER,equation='58',used_for_planning=False,normalization_fitted=False),
        difference=delta,relative_difference=delta/float(target),previous_amplitude=old['amplitude'],
        previous_relative_difference=(old['amplitude']-float(target))/float(target),families=families,
        pilot_directional_comparisons=readiness['pilot_result']['comparisons'],
        previous_moduli_indicator=old['boundary']['quadrature_indicator'],moduli_indicator=r['moduli_indicator'],
        relative_moduli_indicator=r['relative_moduli_indicator'],boundary_indicator_met=r['boundary_indicator_met'],
        compact_sampling_halfwidth95=r['compact']['sampling_halfwidth95'],
        block_indicator=r['boundary']['errors']['block']+r['compact']['errors']['block'],
        unknown_errors=r['unknown_errors'],full_amplitude_certified=False,
        indicator_is_error_bound=False,summary=r['summary'],timings=r['timings'])
    f.spectrum.publish(root/'assessment.json',q)
    lines=[f'# Retained genus-one amplitude at t={q["t"]}', '',
        f'The revised retained estimator gives **A = {value:.15g}** in the convention A = -Im(I_1,3)/(4π).', '',
        f'The dual matrix-model prediction is **{float(target):.15g}** ({target} exactly), so the difference is '
        f'**{delta:+.8g} ({100*q["relative_difference"]:+.5f}%)**. The normalization is inherited, not fitted. '
        f'The target was not used to select quadrature orders, thresholds, or acceptance. '
        f'[Primary source, equation (58)]({PAPER}).', '',
        'The prediction on the equal imaginary-energy ray is '
        '`t^3 (t−1) (t−2) (1+t−t²/2) / 96`. The repository convention uses '
        '`I/(4π)=F1_ws` and `F1_MM=2 F0 S_hat`.', '',
        '| Contribution | Previous | Revised | Revised moduli indicator |',
        '|---|---:|---:|---:|']
    for name,c in families.items():lines.append(f'| {name} | {c["previous"]:.10g} | {c["current"]:.10g} | {c["current_indicator"]:.5g} |')
    lines += [f'| compact (unchanged) | {r["compact"]["mean"]:.10g} | {r["compact"]["mean"]:.10g} | RQMC, separately below |','',
        f'The sum of absolute cell quadrature differences changed from {q["previous_moduli_indicator"]:.8g} '
        f'to **{q["moduli_indicator"]:.8g} ({100*q["relative_moduli_indicator"]:.5f}% of A)**. '
        f'The existing boundary indicator budget is {r["boundary_indicator_budget"]:.8g}; '
        f'it is **{"met" if r["boundary_indicator_met"] else "not met"}**. '
        'This is a convergence diagnostic, not a rigorous error bound or a confidence interval. '
        'The old and new indicators compare different, explicitly recorded rules. '
        'The pilot found opposite signed length and angular changes in its primary cell: their individual '
        'indicators were 0.287% and 0.751%. A combined low/high comparison can partly cancel such changes. '
        'The two local angular order-4/order-6 differences were 0.453% and 0.501%. These selected checks '
        'remain diagnostics, not global bounds.', '',
        f'The H/native-prefix block indicator is {q["block_indicator"]:.8g}. '
        f'The unchanged compact RQMC 95% sampling half-width is {q["compact_sampling_halfwidth95"]:.8g}. '
        'These quantities measure different effects and are not added into a certified total uncertainty.', '',
        'Momentum rules remain eight total nodes on generic axes and nine on the preselected short primary axes. '
        'No ten-node momentum change is included. Primary H4/H2, collision block orders and all physical cutoffs '
        'remain unchanged. All six labeled primary sectors, both short-length orderings and infinite length/height '
        'tails are retained. The unchanged six projected-triple components and compact estimate are explicitly '
        'imported from the pinned completed run.', '',
        'The physical collision boundaries are separated from numerical panel boundaries. Primary lengths follow '
        'collision-onset and modular-cap transitions; angular heads follow the local separation. Pair punctures use '
        'the annulus from 0.5 to 1 plus its exact torus complement, with physical separation panels and finite-height '
        'panels. Every map retains its Jacobian. No weights are renormalized to a target.', '',
        f'This production evaluated {r["summary"]["points"]:,} quadrature points in {r["summary"]["batches"]:,} batches, '
        f'prepared {r["summary"]["new_banks"]:,} new CFT banks and reused {r["summary"]["reused_banks"]:,} exact banks. '
        'Angular points sharing a momentum rule reuse its coefficient bank. Bank preparation and contraction '
        'have separate cost-balanced allocations; final reduction streams bounded batches.', '',
        'The retained approximation is **not a certified full amplitude**. The unresolved contributions are:', '']
    lines.extend('- '+x for x in q['unknown_errors'])
    lines += ['', 'Matrix-model agreement or disagreement cannot identify those missing errors by itself. '
        'No further automatic refinement is scheduled.', '', 'All scientific receipts are immutable SHA256 envelopes. '
        '`design.json`, `readiness.json`, `imports.json`, `amplitude.json` and `assessment.json` retain the provenance, '
        'domain choices, fixed settings, comparison and uncertainty status.', '']
    (root/'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps({k:q[k] for k in ('amplitude','matrix_model','relative_difference','relative_moduli_indicator','boundary_indicator_met')},indent=2))
    return q


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    report(parser.parse_args().root)
