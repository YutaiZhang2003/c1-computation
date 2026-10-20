# Genus-one 1→2 — liouville integration

Retained computational source. For the self-contained result recovery, use
[`reproduce.py`](../../../../reproduce.py) and the [repository README](../../../../README.md).
Production integrations require prepared inputs and computational resources;
the lightweight recovery uses the bundled numerical inputs.

| Source | Purpose |
| --- | --- |
| [genus1_boundary_banks.py](genus1_boundary_banks.py) | Saved native and elliptic coefficients and their truncations |
| [genus1_boundary_spectrum.py](genus1_boundary_spectrum.py) | Preparation of momentum nodes, DOZZ factors, and block coefficients |
| [genus1_channel_evaluation.py](genus1_channel_evaluation.py) | Channel evaluation and arithmetic checks |
| [genus1_collar_mp_arithmetic.py](genus1_collar_mp_arithmetic.py) | Channel evaluation and arithmetic checks |
| [genus1_native_bank.py](genus1_native_bank.py) | Saved native and elliptic coefficients and their truncations |
| [genus1_projection_momentum.py](genus1_projection_momentum.py) | Momentum rules for the degeneration terms |
| [genus1_projection_preparation.py](genus1_projection_preparation.py) | Preparation of momentum nodes, DOZZ factors, and block coefficients |
| [genus1_three_point_momentum.py](genus1_three_point_momentum.py) | Momentum rules and convergence checks |
| [genus1_three_point_worldsheet.py](genus1_three_point_worldsheet.py) | Liouville correlator and momentum integrand |
| [thermal/banks.py](thermal/banks.py) | Thermal coefficient banks and saved momentum integrals |
| [thermal/evaluation.py](thermal/evaluation.py) | Evaluation of the thermal momentum integrand |
| [thermal/paired_quadrature.py](thermal/paired_quadrature.py) | Thermal momentum rules and node placement |
| [thermal/quadrature.py](thermal/quadrature.py) | Thermal momentum rules and node placement |
| [thermal/thermal_cut_rule.py](thermal/thermal_cut_rule.py) | Thermal momentum rules and node placement |
| [thermal/thermal_saved_inputs.py](thermal/thermal_saved_inputs.py) | Thermal coefficient banks and saved momentum integrals |
