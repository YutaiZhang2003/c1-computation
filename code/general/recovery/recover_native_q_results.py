"""Recover current native-q scans and the order checks cited in the draft."""
from pathlib import Path
import csv
import json
import math

import numpy as np


def close(a, b):
    np.testing.assert_allclose(a, b, rtol=2e-11, atol=1e-20)


def receipt(path):
    from genus1_two_point_coefficient_bank import validate_metadata
    data = json.loads(Path(path).read_text())
    validate_metadata(data['kernel'])
    assert data['power'] == 16 and data['replicates'] == 8
    assert data['seed'] == 20260927 and data['collision_remainder_included']
    values = np.asarray(data['replicate_integrals'], dtype=float)
    assert values.shape == (8, 2, 2) and np.all(np.isfinite(values))
    a = values[:, :, 0] * (2 * math.pi**2)
    recovered = dict(
        zero_temperature_imag=float(a[:, 0].mean()),
        thermal_imag=float(a[:, 1].mean()),
        thermal_correction_imag=float(a[:, 1].mean() - a[:, 0].mean()),
        sampling_SE_zero=float(a[:, 0].std(ddof=1) / math.sqrt(8)),
        sampling_SE_thermal=float(a[:, 1].std(ddof=1) / math.sqrt(8)),
        sampling_SE_correction=float((a[:, 1] - a[:, 0]).std(ddof=1) / math.sqrt(8)))
    for key, value in recovered.items():
        close(value, data[key])
    data.update(recovered)
    return data


def difference(new, old, label):
    assert all(new[k] == old[k] for k in ('power', 'replicates', 'seed'))
    a = 2 * math.pi**2 * (np.asarray(new['replicate_integrals'])[:, :, 0]
                         - np.asarray(old['replicate_integrals'])[:, :, 0])
    se = a.std(axis=0, ddof=1) / math.sqrt(len(a))
    return dict(label=label, R=new['R'], omega_imag=new['omega_imag'],
        noncompact_difference=float(a[:, 0].mean()), noncompact_paired_SE=float(se[0]),
        thermal_difference=float(a[:, 1].mean()), thermal_paired_SE=float(se[1]),
        noncompact_relative_difference=float(a[:, 0].mean() / old['zero_temperature_imag']),
        thermal_relative_difference=float(a[:, 1].mean() / old['thermal_imag']))


def recover(root, output):
    from c1_matrix_model import amplitude_coefficient
    root, output = Path(root), Path(output)
    manifest = json.loads((root/'run_manifest.json').read_text())
    records = {}
    for R in manifest['unique_radii']:
        d = receipt(root/'scan_complete_collision'/f'R{R:g}.json')
        close(d['R'], R)
        assert d['kernel']['momentum_order'] == 32 and d['collision_radius'] == .5
        assert d['kernel']['necklace_orders'] == [6, 3]
        assert d['kernel']['ope_orders'] == [3, 8]
        records[f'{R:g}'] = d
    thermal, noncompact, audits = [], [], []
    for R in manifest['thermal_radii']:
        d = records[f'{R:g}']; x = 1/R
        theory = math.pi**2/6*x*x*(x*x-1)
        thermal.append(dict(R=R, omega_imag=x, amplitude_imag=d['thermal_imag'],
            sampling_SE=d['sampling_SE_thermal'], matrix_model_imag=theory,
            relative_difference=d['thermal_imag']/theory-1,
            noncompact_amplitude_imag=d['zero_temperature_imag'],
            thermal_correction_imag=d['thermal_correction_imag'],
            thermal_correction_sampling_SE=d['sampling_SE_correction']))
    for x in manifest['noncompact_omega_imag']:
        d = records[f'{1/x:g}']; theory = amplitude_coefficient(1, [1j*x]).imag
        noncompact.append(dict(omega_imag=x, amplitude_imag=d['zero_temperature_imag'],
            sampling_SE=d['sampling_SE_zero'], matrix_model_imag=theory,
            relative_difference=d['zero_temperature_imag']/theory-1))
    for R in manifest['audits']['momentum_check_radii']:
        d = receipt(root/'audits/nodes48/scan_native_q_complete_collision'/f'R{R:g}.json')
        assert d['kernel']['momentum_order'] == 48
        audits.append(difference(d, records[f'{R:g}'], '48 minus 32 momentum nodes'))
    d = receipt(root/'audits/radius01/scan_native_q_complete_collision/R4.json')
    assert d['collision_radius'] == .1
    audits.append(difference(d, records['4'], 'collision radius 0.1 minus 0.5'))
    block7 = receipt(root/'audits/block7_R2.json')
    block8 = receipt(root/'audits/block8_R2.json')
    assert block7['kernel']['necklace_orders'] == [7, 4]
    assert block8['kernel']['necklace_orders'] == [8, 4]
    assert block7['kernel']['ope_orders'] == block8['kernel']['ope_orders'] == [4, 10]
    audits.append(difference(block8, block7, 'necklace order (8,4) minus (7,4); OPE (4,10)'))
    audits.append(difference(block8, records['2'], 'necklace (8,4), OPE (4,10) minus baseline (6,3), (3,8)'))
    tables = {'thermal_1to1.csv': thermal, 'noncompact_1to1.csv': noncompact,
              'convergence_checks.csv': audits}
    output.mkdir(parents=True, exist_ok=False)
    for name, rows in tables.items():
        with (root/'reference'/name).open(newline='') as stream:
            saved = list(csv.DictReader(stream))
        assert len(saved) == len(rows)
        for recovered, reference in zip(rows, saved):
            assert set(recovered) == set(reference)
            for key, value in recovered.items():
                if isinstance(value, str):
                    assert value == reference[key]
                else:
                    close(value, float(reference[key]))
        with (output/name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    return dict(passed=True, unique_parameter_receipts=len(records),
        recovered_rows={name: len(rows) for name, rows in tables.items()},
        coefficient_banks_required=False, historical_baselines_required=False,
        scope='Saved integrated replicates; full integrations have not been rerun.')
