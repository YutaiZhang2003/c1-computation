# Sphere 1→4

Retained computational source. For the self-contained result recovery, use
[`reproduce.py`](../../../reproduce.py) and the [repository README](../../../README.md).
Production integrations require prepared inputs and computational resources;
the lightweight recovery uses the bundled numerical inputs.

| Source | Purpose |
| --- | --- |
| [conformal_block/five_point_liouville_correlator_and_atlas.py](conformal_block/five_point_liouville_correlator_and_atlas.py) | Decomposition channels, coordinate frames, and channel selection |
| [conformal_block/five_point_virasoro_block_recursion.py](conformal_block/five_point_virasoro_block_recursion.py) | Five-point block from the $c$-recursion |
| [liouville_integration/five_point_integrand_and_quadrature.py](liouville_integration/five_point_integrand_and_quadrature.py) | Momentum quadrature, precomputed DOZZ factors and block coefficients, and crossed-pole residues |
| [moduli_integration/five_point_moduli_integration.py](moduli_integration/five_point_moduli_integration.py) | Sampling weight, Monte Carlo points, moduli integral, and sampling standard error |
| [other_scripts/ccy_sphere_five_point_checks.py](other_scripts/ccy_sphere_five_point_checks.py) | Independent checks of the conformal block |
| [other_scripts/export_sphere_five_point_t_axis_plot_data.py](other_scripts/export_sphere_five_point_t_axis_plot_data.py) | Amplitude values and uncertainty estimates used in the figure |
| [other_scripts/five_point_production.py](other_scripts/five_point_production.py) | Amplitude scan and its frozen twelve-point extension |
| [other_scripts/sphere_five_point_30point_audit_summary.py](other_scripts/sphere_five_point_30point_audit_summary.py) | Comparisons with saved higher-order evaluations and checks of the amplitude table |
| [other_scripts/sphere_five_point_30point_checks.py](other_scripts/sphere_five_point_30point_checks.py) | Comparisons with saved higher-order evaluations and checks of the amplitude table |
| [other_scripts/sphere_five_point_30point_matrix_comparison.py](other_scripts/sphere_five_point_30point_matrix_comparison.py) | Comparison with the matrix-model amplitude |
| [other_scripts/sphere_five_point_30point_worldsheet_fit.py](other_scripts/sphere_five_point_30point_worldsheet_fit.py) | Combining saved amplitudes and fitting their energy dependence |
| [other_scripts/sphere_five_point_continuation_checks.py](other_scripts/sphere_five_point_continuation_checks.py) | Checks of momentum quadrature and contour residues |
| [other_scripts/sphere_five_point_imaginary_ray_fit.py](other_scripts/sphere_five_point_imaginary_ray_fit.py) | Combining saved amplitudes and fitting their energy dependence |
| [other_scripts/sphere_five_point_liouville_checks.py](other_scripts/sphere_five_point_liouville_checks.py) | Checks of coordinate frames and Jacobians |
