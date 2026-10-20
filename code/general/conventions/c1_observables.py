"""Final observable exports; reduced integrals and production caches keep their units."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from c1_string_conventions import (
    CURRENT_STRING_COUPLING_CONVENTION, MQM_INVERSE_MU_CONVENTION,
    amplitude_coefficient_scale_to_bry, log_partition_to_free_energy_scale,
)


def amplitude_metadata(genus, punctures):
    amplitude_coefficient_scale_to_bry(genus, punctures, CURRENT_STRING_COUPLING_CONVENTION)
    power = 2 * genus - 2 + punctures
    return dict(quantity='amplitude_coefficient', genus=genus, punctures=punctures,
                string_coupling_convention=CURRENT_STRING_COUPLING_CONVENTION,
                coupling_power=power, full_genus_term=f'g_s^{power} * amplitude')


def complex_record(value):
    z = complex(value)
    if not (math.isfinite(z.real) and math.isfinite(z.imag)):
        raise ValueError('finite complex estimate required')
    return dict(real=z.real, imag=z.imag)


def amplitude_from_reduced(value, standard_error, *, factor, genus, punctures,
                           reduced_quantity, replicates=None):
    """Apply a declared reduced-integral prefactor, including component covariance.

    Without paired replicates, only real or imaginary prefactors are allowed:
    their component errors can be propagated without inventing a covariance.
    """
    factor = complex(factor)
    result = amplitude_metadata(genus, punctures)
    result.update(value=complex_record(factor * value), reduced_quantity=reduced_quantity,
                  reduced_to_amplitude_factor=complex_record(factor))
    error = complex(standard_error)
    if not np.isfinite([error.real, error.imag]).all() or min(error.real, error.imag) < 0:
        raise ValueError('finite nonnegative component standard errors required')
    if replicates is not None:
        samples = np.asarray(replicates, complex)
        if samples.ndim != 1 or len(samples) < 2 or not np.isfinite(samples).all():
            raise ValueError('complete independent replicate estimates required')
        np.testing.assert_allclose(samples.mean(), value, rtol=1e-10, atol=1e-20)
        converted = samples * factor
        covariance = np.cov(np.stack((converted.real, converted.imag)), ddof=1) / len(samples)
        se = np.sqrt(np.diag(covariance))
        result.update(replicate_estimates=[complex_record(z) for z in converted],
                      component_covariance=covariance.tolist())
    elif factor.real == 0:
        se = [abs(factor) * error.imag, abs(factor) * error.real]
    elif factor.imag == 0:
        se = [abs(factor) * error.real, abs(factor) * error.imag]
    else:
        raise ValueError('mixed complex prefactor requires paired replicate covariance')
    result['standard_error'] = dict(zip(('real', 'imag'), map(float, se)))
    return result


def genus1_amplitude_output(native):
    """Publish the 1->2 coefficient alongside explicitly labelled legacy statistics.

    Frozen workers, absolute tolerances and adaptive decisions use mu^-3 units.
    This function is called after those decisions, never on the observations.
    """
    if 'amplitude_gs' in native:
        raise ValueError('amplitude output was already converted')
    source = native.get('native_amplitude_convention', MQM_INVERSE_MU_CONVENTION)
    scale = amplitude_coefficient_scale_to_bry(1, 3, source)
    result = amplitude_metadata(1, 3)
    result['component'] = 'real'
    result['source_convention'] = source
    result['value'] = None if native.get('amplitude') is None else float(native['amplitude']) * scale
    for key in ('mc_standard_error', 'mc_half_width_95', 'sampling_se', 'sampling_halfwidth95',
                'boundary_quadrature_indicator', 'absolute_error_target',
                'arithmetic_absolute_indicator', 'combined_empirical_error',
                'numerical_empirical_indicator', 'combined_empirical_indicator',
                'omitted_approximation_error'):
        if key in native:
            result[key] = None if native[key] is None else float(native[key]) * scale
    for key in ('errors', 'residual_numerical_indicators'):
        if key in native:
            result[key] = None if native[key] is None else {
                name: None if value is None else float(value) * scale
                for name, value in native[key].items()}
    if 'replicate_estimates' in native:
        result['replicate_estimates'] = (np.asarray(native['replicate_estimates']) * scale).tolist()
    for key in ('analytic_approximation', 'full_amplitude_certified', 'rigorous_error_bound',
                'passes_total_2pct', 'status', 'unassessed_errors'):
        if key in native:
            result[key] = native[key]
    return dict(native, native_amplitude_convention=source,
                amplitude_gs=result)


def scale_covariance(covariance, factors):
    factors = np.asarray(factors, float)
    covariance = np.asarray(covariance, float)
    if (factors.ndim != 1 or covariance.shape != (len(factors), len(factors))
            or not np.isfinite(factors).all() or not np.isfinite(covariance).all()
            or np.any(np.diag(covariance) < 0)):
        raise ValueError('finite covariance and matching scale vector required')
    np.testing.assert_allclose(covariance, covariance.T, rtol=1e-11, atol=1e-25)
    return factors[:, None] * covariance * factors[None, :]


def genus2_free_energy_arrays(radii, log_z2, covariance, *, source_convention):
    from c1_string_conventions import matrix_model_genus2_free_energy
    radii = np.asarray(radii, float)
    log_z2 = np.asarray(log_z2, float)
    if radii.ndim != 1 or len(radii) == 0 or log_z2.shape != radii.shape or not np.isfinite(log_z2).all():
        raise ValueError('matching finite radius and log Z vectors required')
    factors = np.array([log_partition_to_free_energy_scale(r, genus=2,
                        source_convention=source_convention) for r in radii])
    converted_covariance = scale_covariance(covariance, factors)
    return dict(radii=radii, F2=factors * log_z2, covariance_F2=converted_covariance,
                F2_empirical_MC_SE=np.sqrt(np.diag(converted_covariance)),
                F2_matrix_model=np.array([matrix_model_genus2_free_energy(r) for r in radii]),
                log_Z_to_F2_scale=factors)


def write_genus2_free_energy(output, arrays, source_summary):
    """Export BRY-v7 log Z data as literal F_2; retain unresolved error scopes."""
    if source_summary.get('integration_kernel_convention') != (
            'c1-bry-real-area-genus2-aut-half-dimensionless-radius-gs2-stripped-v7'):
        raise ValueError('explicit BRY-v7 source required; no implicit quotient conversion')
    converted = genus2_free_energy_arrays(arrays['radii'], arrays['resolved_mean'],
        arrays['resolved_covariance'], source_convention=CURRENT_STRING_COUPLING_CONVENTION)
    np.testing.assert_allclose(converted['log_Z_to_F2_scale'] * arrays['matrix_model'],
                               converted['F2_matrix_model'], rtol=2e-13)
    for key, value in arrays.items():
        if key.startswith('group_'):
            converted[key] = np.asarray(value) * converted['log_Z_to_F2_scale']
    r = converted['radii']
    y = converted['F2']
    se = converted['F2_empirical_MC_SE']
    # Same inverse-marginal-variance estimator as the native radius fit.
    # Keep the full cross-radius covariance when propagating its uncertainty.
    design = np.column_stack((1 + r**-4, r**-2))
    if np.any(se <= 0):
        raise ValueError('positive sampling errors required for the radius fit')
    operator = np.linalg.lstsq(design / se[:, None], np.diag(1 / se), rcond=None)[0]
    coefficients = operator @ y
    coefficient_covariance = np.einsum('ai,ij,bj->ab', operator, converted['covariance_F2'], operator)
    converted.update(fit_coefficient=coefficients, fit_covariance=coefficient_covariance)
    report = dict(quantity='thermal_free_energy_coefficient', genus=2,
        string_coupling_convention=CURRENT_STRING_COUPLING_CONVENTION,
        definition='F=-log Z/(2*pi*R)', full_genus_term='g_s^2 * F2(R)',
        matrix_model_F2='-pi*(7+10/R^2+7/R^4)/2880',
        matrix_model_used_to_calibrate=False,
        fit=dict(ansatz='a*(1+R^-4)+b*R^-2', coefficient=coefficients.tolist(),
                 coefficient_covariance=coefficient_covariance.tolist()),
        source_summary=source_summary)
    one = np.flatnonzero(r == 1)
    if len(one) == 1:
        i = one[0]
        report['R1'] = dict(F2=float(y[i]), empirical_MC_SE=float(se[i]),
                            matrix_model_F2=float(converted['F2_matrix_model'][i]))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'free_energy_F2.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['R', 'F2', 'F2_empirical_MC_SE', 'F2_matrix_model'])
        writer.writerows(zip(r, y, se, converted['F2_matrix_model']))
    np.savez_compressed(output / 'free_energy_F2.npz', **converted)
    (output / 'free_energy_F2.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report


def write_six_point_amplitudes(output, report):
    """Export audited Q5 rows as complex A^(0)_(1->5), including scramble covariance."""
    records = []
    rows = report['results']
    for row in rows:
        t = float(row['t'])
        means = np.asarray(row['replicate_means'], float)
        if means.ndim != 2 or means.shape[1] != 2:
            raise ValueError('paired real/imaginary Q5 scramble means required')
        values = means[:, 0] + 1j * means[:, 1]
        record = amplitude_from_reduced(values.mean(), 0j, factor=-5j * t**6 * (2*math.pi)**4,
            genus=0, punctures=6, reduced_quantity='Q5', replicates=values)
        record.update(t=t, total_accuracy_certified=False)
        records.append(record)
    result = dict(amplitude_metadata(0, 6), points=records, source_quantity='Q5',
                  total_accuracy_certified=report.get('total_accuracy_certified', False))
    # Report the covariance of the displayed imaginary component across energies.
    if 'mean_covariance' in report:
        factors = [-5 * float(row['t'])**6 * (2*math.pi)**4 for row in rows]
        result['amplitude_imag_mean_covariance'] = scale_covariance(report['mean_covariance'], factors).tolist()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'amplitudes_gs.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    with (output / 'amplitudes_gs.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['t', 'amplitude_real', 'amplitude_imag', 'SE_real', 'SE_imag'])
        writer.writerows((r['t'], r['value']['real'], r['value']['imag'],
                          r['standard_error']['real'], r['standard_error']['imag']) for r in records)
    return result
