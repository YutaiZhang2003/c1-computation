# Sphere 1→5

Retained computational source. For the self-contained result recovery, use
[`reproduce.py`](../../../reproduce.py) and the [repository README](../../../README.md).
Production integrations require prepared inputs and computational resources;
the lightweight recovery uses the bundled numerical inputs.

| Source | Purpose |
| --- | --- |
| [conformal_block/six_point_arbitrary_precision_c_recursion.py](conformal_block/six_point_arbitrary_precision_c_recursion.py) | Six-point block from the $c$-recursion and checks of arithmetic precision |
| [conformal_block/six_point_block_coefficient_provider.py](conformal_block/six_point_block_coefficient_provider.py) | Six-point block from the $c$-recursion and checks of arithmetic precision |
| [conformal_block/six_point_collar_block_evaluator.py](conformal_block/six_point_collar_block_evaluator.py) | Four-point elliptic block and outer descendant corrections in $D_{\mathrm{sep}}$ |
| [conformal_block/six_point_measure_block_convergence.py](conformal_block/six_point_measure_block_convergence.py) | Assessment of block truncation |
| [conformal_block/six_point_refinement_statistics.py](conformal_block/six_point_refinement_statistics.py) | Refinement criteria and paired Monte Carlo statistics |
| [conformal_block/six_point_select_integration_channels.py](conformal_block/six_point_select_integration_channels.py) | Channel selection |
| [liouville_integration/six_point_analyze_quadrature_and_order_controls.py](liouville_integration/six_point_analyze_quadrature_and_order_controls.py) | Assessment of momentum integration and block truncation |
| [liouville_integration/six_point_build_momentum_banks.py](liouville_integration/six_point_build_momentum_banks.py) | Building coefficient tables and identifying repeated momentum nodes |
| [liouville_integration/six_point_collar_momentum_quadrature.py](liouville_integration/six_point_collar_momentum_quadrature.py) | Momentum integration in $D_{\mathrm{sep}}$ and analytic radial integrals |
| [liouville_integration/six_point_compare_momentum_quadratures.py](liouville_integration/six_point_compare_momentum_quadratures.py) | Assessment of momentum integration and block truncation |
| [liouville_integration/six_point_deduplicate_momentum_nodes.py](liouville_integration/six_point_deduplicate_momentum_nodes.py) | Building coefficient tables and identifying repeated momentum nodes |
| [liouville_integration/six_point_endpoint_and_tail_quadrature.py](liouville_integration/six_point_endpoint_and_tail_quadrature.py) | Momentum integration grids |
| [liouville_integration/six_point_mixed_order_node_catalog.py](liouville_integration/six_point_mixed_order_node_catalog.py) | Building coefficient tables and identifying repeated momentum nodes |
| [liouville_integration/six_point_momentum_banks_and_reweighting.py](liouville_integration/six_point_momentum_banks_and_reweighting.py) | Liouville momentum integration and stored coefficient tables |
| [liouville_integration/six_point_momentum_control_proposals.py](liouville_integration/six_point_momentum_control_proposals.py) | Momentum integration grids |
| [liouville_integration/six_point_reuse_saved_coefficients.py](liouville_integration/six_point_reuse_saved_coefficients.py) | Reusing previously computed coefficients |
| [moduli_integration/six_point_analytic_collar_subtraction.py](moduli_integration/six_point_analytic_collar_subtraction.py) | Approximate integrand in $D_{\mathrm{sep}}$ and its integrated contribution |
| [moduli_integration/six_point_assemble_curve_amplitudes.py](moduli_integration/six_point_assemble_curve_amplitudes.py) | Combining Monte Carlo estimates and their covariance |
| [moduli_integration/six_point_assemble_radial_curve_covariance.py](moduli_integration/six_point_assemble_radial_curve_covariance.py) | Combining Monte Carlo estimates and their covariance |
| [moduli_integration/six_point_batch_block_contraction.py](moduli_integration/six_point_batch_block_contraction.py) | Weighted integrand $W_\Gamma$ |
| [moduli_integration/six_point_boundary_strata.py](moduli_integration/six_point_boundary_strata.py) | Separating domains and subtraction measure |
| [moduli_integration/six_point_channel_coordinates.py](moduli_integration/six_point_channel_coordinates.py) | Coordinate frames, transformations, and $\rho_\Gamma$ |
| [moduli_integration/six_point_check_sampling_variance.py](moduli_integration/six_point_check_sampling_variance.py) | Weight function and Monte Carlo sampling points |
| [moduli_integration/six_point_collar_counterterm_geometry.py](moduli_integration/six_point_collar_counterterm_geometry.py) | Separating domains and subtraction measure |
| [moduli_integration/six_point_combine_truncation_orders.py](moduli_integration/six_point_combine_truncation_orders.py) | Moduli integration with different block truncations |
| [moduli_integration/six_point_estimate_collar_corrected_integrand.py](moduli_integration/six_point_estimate_collar_corrected_integrand.py) | Separating subtraction and the remaining integrand |
| [moduli_integration/six_point_evaluate_routed_geometry_shards.py](moduli_integration/six_point_evaluate_routed_geometry_shards.py) | Separating subtraction and the remaining integrand |
| [moduli_integration/six_point_integrate_collar_patch.py](moduli_integration/six_point_integrate_collar_patch.py) | Approximate integrand in $D_{\mathrm{sep}}$ and its integrated contribution |
| [moduli_integration/six_point_integrate_mixed_order_amplitudes.py](moduli_integration/six_point_integrate_mixed_order_amplitudes.py) | Moduli integration with different block truncations |
| [moduli_integration/six_point_prepare_radial_geometry.py](moduli_integration/six_point_prepare_radial_geometry.py) | Weight function and Monte Carlo sampling points |
| [moduli_integration/six_point_radial_sampling_distribution.py](moduli_integration/six_point_radial_sampling_distribution.py) | Weight function and Monte Carlo sampling points |
| [moduli_integration/six_point_refinement_assembly.py](moduli_integration/six_point_refinement_assembly.py) | Selected higher-order corrections and final amplitude assembly |
| [moduli_integration/six_point_route_collar_contributions.py](moduli_integration/six_point_route_collar_contributions.py) | Separating subtraction and the remaining integrand |
| [moduli_integration/six_point_sample_integration_geometry.py](moduli_integration/six_point_sample_integration_geometry.py) | Coordinate frames, transformations, and $\rho_\Gamma$ |
| [moduli_integration/six_point_stable_puncture_coordinates.py](moduli_integration/six_point_stable_puncture_coordinates.py) | Coordinate frames, transformations, and $\rho_\Gamma$ |
| [production/dispatch_shared_refinement_queue.py](production/dispatch_shared_refinement_queue.py) | Checking saved inputs and distributing refinement jobs |
| [production/six_point_prepare_amplitude_curve.py](production/six_point_prepare_amplitude_curve.py) | Preparing the amplitude calculation and running its stages |
| [production/six_point_refinement.py](production/six_point_refinement.py) | Selective refinement through $N=14$ and continuation through $N=18$ |
| [production/six_point_run_frozen_energy_worker.py](production/six_point_run_frozen_energy_worker.py) | Preparing the amplitude calculation and running its stages |
| [production/six_point_run_frozen_radial_worker.py](production/six_point_run_frozen_radial_worker.py) | Preparing the amplitude calculation and running its stages |
| [production/six_point_submit_radial_campaign.py](production/six_point_submit_radial_campaign.py) | Preparing the amplitude calculation and running its stages |
| [production/verify_saved_c14_continuation_inputs.py](production/verify_saved_c14_continuation_inputs.py) | Checking saved inputs and distributing refinement jobs |
