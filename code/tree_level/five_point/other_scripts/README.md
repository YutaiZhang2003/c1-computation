# Sphere 1→4 — other scripts

Retained computational source. For the self-contained result recovery, use
[`reproduce.py`](../../../../reproduce.py) and the [repository README](../../../../README.md).
Production integrations require prepared inputs and computational resources;
the lightweight recovery uses the bundled numerical inputs.

| Source | Purpose |
| --- | --- |
| [ccy_sphere_five_point_checks.py](ccy_sphere_five_point_checks.py) | Independent checks of the conformal block |
| [export_sphere_five_point_t_axis_plot_data.py](export_sphere_five_point_t_axis_plot_data.py) | Amplitude values and uncertainty estimates used in the figure |
| [five_point_production.py](five_point_production.py) | Amplitude scan and its frozen twelve-point extension |
| [sphere_five_point_30point_audit_summary.py](sphere_five_point_30point_audit_summary.py) | Comparisons with saved higher-order evaluations and checks of the amplitude table |
| [sphere_five_point_30point_checks.py](sphere_five_point_30point_checks.py) | Comparisons with saved higher-order evaluations and checks of the amplitude table |
| [sphere_five_point_30point_matrix_comparison.py](sphere_five_point_30point_matrix_comparison.py) | Comparison with the matrix-model amplitude |
| [sphere_five_point_30point_worldsheet_fit.py](sphere_five_point_30point_worldsheet_fit.py) | Combining saved amplitudes and fitting their energy dependence |
| [sphere_five_point_continuation_checks.py](sphere_five_point_continuation_checks.py) | Checks of momentum quadrature and contour residues |
| [sphere_five_point_imaginary_ray_fit.py](sphere_five_point_imaginary_ray_fit.py) | Combining saved amplitudes and fitting their energy dependence |
| [sphere_five_point_liouville_checks.py](sphere_five_point_liouville_checks.py) | Checks of coordinate frames and Jacobians |
