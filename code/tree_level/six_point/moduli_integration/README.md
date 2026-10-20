# Sphere 1→5 — moduli integration

Retained computational source. For the self-contained result recovery, use
[`reproduce.py`](../../../../reproduce.py) and the [repository README](../../../../README.md).
Production integrations require prepared inputs and computational resources;
the lightweight recovery uses the bundled numerical inputs.

| Source | Purpose |
| --- | --- |
| [six_point_analytic_collar_subtraction.py](six_point_analytic_collar_subtraction.py) | Approximate integrand in $D_{\mathrm{sep}}$ and its integrated contribution |
| [six_point_assemble_curve_amplitudes.py](six_point_assemble_curve_amplitudes.py) | Combining Monte Carlo estimates and their covariance |
| [six_point_assemble_radial_curve_covariance.py](six_point_assemble_radial_curve_covariance.py) | Combining Monte Carlo estimates and their covariance |
| [six_point_batch_block_contraction.py](six_point_batch_block_contraction.py) | Weighted integrand $W_\Gamma$ |
| [six_point_boundary_strata.py](six_point_boundary_strata.py) | Separating domains and subtraction measure |
| [six_point_channel_coordinates.py](six_point_channel_coordinates.py) | Coordinate frames, transformations, and $\rho_\Gamma$ |
| [six_point_check_sampling_variance.py](six_point_check_sampling_variance.py) | Weight function and Monte Carlo sampling points |
| [six_point_collar_counterterm_geometry.py](six_point_collar_counterterm_geometry.py) | Separating domains and subtraction measure |
| [six_point_combine_truncation_orders.py](six_point_combine_truncation_orders.py) | Moduli integration with different block truncations |
| [six_point_estimate_collar_corrected_integrand.py](six_point_estimate_collar_corrected_integrand.py) | Separating subtraction and the remaining integrand |
| [six_point_evaluate_routed_geometry_shards.py](six_point_evaluate_routed_geometry_shards.py) | Separating subtraction and the remaining integrand |
| [six_point_integrate_collar_patch.py](six_point_integrate_collar_patch.py) | Approximate integrand in $D_{\mathrm{sep}}$ and its integrated contribution |
| [six_point_integrate_mixed_order_amplitudes.py](six_point_integrate_mixed_order_amplitudes.py) | Moduli integration with different block truncations |
| [six_point_prepare_radial_geometry.py](six_point_prepare_radial_geometry.py) | Weight function and Monte Carlo sampling points |
| [six_point_radial_sampling_distribution.py](six_point_radial_sampling_distribution.py) | Weight function and Monte Carlo sampling points |
| [six_point_refinement_assembly.py](six_point_refinement_assembly.py) | Selected higher-order corrections and final amplitude assembly |
| [six_point_route_collar_contributions.py](six_point_route_collar_contributions.py) | Separating subtraction and the remaining integrand |
| [six_point_sample_integration_geometry.py](six_point_sample_integration_geometry.py) | Coordinate frames, transformations, and $\rho_\Gamma$ |
| [six_point_stable_puncture_coordinates.py](six_point_stable_puncture_coordinates.py) | Coordinate frames, transformations, and $\rho_\Gamma$ |
