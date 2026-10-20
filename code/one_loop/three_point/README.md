# Genus-one 1→2

Retained computational source. For the self-contained result recovery, use
[`reproduce.py`](../../../reproduce.py) and the [repository README](../../../README.md).
Production integrations require prepared inputs and computational resources;
the lightweight recovery uses the bundled numerical inputs.

| Source | Purpose |
| --- | --- |
| [conformal_block/genus1_collision_elliptic.py](conformal_block/genus1_collision_elliptic.py) | Elliptic blocks for collision limits and precomputed coefficients |
| [conformal_block/genus1_native_log_evaluation.py](conformal_block/genus1_native_log_evaluation.py) | Evaluation of the retained native series without overflow |
| [conformal_block/torus_c_recursion_numerics.py](conformal_block/torus_c_recursion_numerics.py) | c-recursion and precision checks |
| [conformal_block/torus_three_point_blocks.py](conformal_block/torus_three_point_blocks.py) | Torus three-point blocks in the necklace, pair OPE, and comb channels |
| [liouville_integration/genus1_boundary_banks.py](liouville_integration/genus1_boundary_banks.py) | Saved native and elliptic coefficients and their truncations |
| [liouville_integration/genus1_boundary_spectrum.py](liouville_integration/genus1_boundary_spectrum.py) | Preparation of momentum nodes, DOZZ factors, and block coefficients |
| [liouville_integration/genus1_channel_evaluation.py](liouville_integration/genus1_channel_evaluation.py) | Channel evaluation and arithmetic checks |
| [liouville_integration/genus1_collar_mp_arithmetic.py](liouville_integration/genus1_collar_mp_arithmetic.py) | Channel evaluation and arithmetic checks |
| [liouville_integration/genus1_native_bank.py](liouville_integration/genus1_native_bank.py) | Saved native and elliptic coefficients and their truncations |
| [liouville_integration/genus1_projection_momentum.py](liouville_integration/genus1_projection_momentum.py) | Momentum rules for the degeneration terms |
| [liouville_integration/genus1_projection_preparation.py](liouville_integration/genus1_projection_preparation.py) | Preparation of momentum nodes, DOZZ factors, and block coefficients |
| [liouville_integration/genus1_three_point_momentum.py](liouville_integration/genus1_three_point_momentum.py) | Momentum rules and convergence checks |
| [liouville_integration/genus1_three_point_worldsheet.py](liouville_integration/genus1_three_point_worldsheet.py) | Liouville correlator and momentum integrand |
| [liouville_integration/thermal/banks.py](liouville_integration/thermal/banks.py) | Thermal coefficient banks and saved momentum integrals |
| [liouville_integration/thermal/evaluation.py](liouville_integration/thermal/evaluation.py) | Evaluation of the thermal momentum integrand |
| [liouville_integration/thermal/paired_quadrature.py](liouville_integration/thermal/paired_quadrature.py) | Thermal momentum rules and node placement |
| [liouville_integration/thermal/quadrature.py](liouville_integration/thermal/quadrature.py) | Thermal momentum rules and node placement |
| [liouville_integration/thermal/thermal_cut_rule.py](liouville_integration/thermal/thermal_cut_rule.py) | Thermal momentum rules and node placement |
| [liouville_integration/thermal/thermal_saved_inputs.py](liouville_integration/thermal/thermal_saved_inputs.py) | Thermal coefficient banks and saved momentum integrals |
| [moduli_integration/genus1_analytic_boundary_layer.py](moduli_integration/genus1_analytic_boundary_layer.py) | Partition of moduli space, sampling weights, and collision-region maps |
| [moduli_integration/genus1_boundary_batch.py](moduli_integration/genus1_boundary_batch.py) | Reuse of moduli-dependent factors |
| [moduli_integration/genus1_boundary_cubature.py](moduli_integration/genus1_boundary_cubature.py) | Deterministic quadrature and assembly of the moduli integral |
| [moduli_integration/genus1_boundary_integration.py](moduli_integration/genus1_boundary_integration.py) | Deterministic quadrature and assembly of the moduli integral |
| [moduli_integration/genus1_collision_sampling.py](moduli_integration/genus1_collision_sampling.py) | Partition of moduli space, sampling weights, and collision-region maps |
| [moduli_integration/genus1_comb_collar.py](moduli_integration/genus1_comb_collar.py) | Collision subtractions and integrated degeneration terms |
| [moduli_integration/genus1_cusp_control.py](moduli_integration/genus1_cusp_control.py) | Nonseparating approximation and tube integrals |
| [moduli_integration/genus1_integrated_scale_collar.py](moduli_integration/genus1_integrated_scale_collar.py) | Collision subtractions and integrated degeneration terms |
| [moduli_integration/genus1_modular_geometry.py](moduli_integration/genus1_modular_geometry.py) | Plumbing parameters, channel choice, and sewing domains |
| [moduli_integration/genus1_moduli_panels.py](moduli_integration/genus1_moduli_panels.py) | Deterministic quadrature and assembly of the moduli integral |
| [moduli_integration/genus1_moduli_sampling.py](moduli_integration/genus1_moduli_sampling.py) | Partition of moduli space, sampling weights, and collision-region maps |
| [moduli_integration/genus1_nonseparating.py](moduli_integration/genus1_nonseparating.py) | Nonseparating approximation and tube integrals |
| [moduli_integration/genus1_pair_collar.py](moduli_integration/genus1_pair_collar.py) | Collision subtractions and integrated degeneration terms |
| [moduli_integration/genus1_sewing_domains.py](moduli_integration/genus1_sewing_domains.py) | Plumbing parameters, channel choice, and sewing domains |
| [moduli_integration/genus1_three_point_channel_atlas.py](moduli_integration/genus1_three_point_channel_atlas.py) | Plumbing parameters, channel choice, and sewing domains |
| [moduli_integration/genus1_three_point_collision.py](moduli_integration/genus1_three_point_collision.py) | Collision subtractions and integrated degeneration terms |
| [moduli_integration/thermal/kernels.py](moduli_integration/thermal/kernels.py) | Thermal kinematics and integrated boundary kernels |
| [moduli_integration/thermal/radius_scan.py](moduli_integration/thermal/radius_scan.py) | Moduli sampling for the thermal correction |
| [moduli_integration/thermal/remainders.py](moduli_integration/thermal/remainders.py) | Thermal remainder terms |
| [moduli_integration/thermal/study_pair_sampling.py](moduli_integration/thermal/study_pair_sampling.py) | Moduli sampling for the thermal correction |
| [other_scripts/checks/genus1_three_point_c_recursion_checks.py](other_scripts/checks/genus1_three_point_c_recursion_checks.py) | Independent checks of the block recursion |
| [other_scripts/genus1_amplitude_curve.py](other_scripts/genus1_amplitude_curve.py) | Production inputs and preparation of the energy scan |
| [other_scripts/genus1_amplitude_report.py](other_scripts/genus1_amplitude_report.py) | Amplitude reports and numerical assessment |
| [other_scripts/genus1_compact_replay.py](other_scripts/genus1_compact_replay.py) | Production of the compact and degeneration contributions |
| [other_scripts/genus1_final_curve_plan.py](other_scripts/genus1_final_curve_plan.py) | Production inputs and preparation of the energy scan |
| [other_scripts/genus1_fresh_analytic.py](other_scripts/genus1_fresh_analytic.py) | Production of the compact and degeneration contributions |
| [other_scripts/genus1_moduli_amplitude.py](other_scripts/genus1_moduli_amplitude.py) | Production of the compact and degeneration contributions |
| [other_scripts/genus1_moduli_panel_campaign.py](other_scripts/genus1_moduli_panel_campaign.py) | Production of the compact and degeneration contributions |
| [other_scripts/genus1_moduli_stability.py](other_scripts/genus1_moduli_stability.py) | Moduli-quadrature comparisons |
| [other_scripts/genus1_moduli_stability_curve.py](other_scripts/genus1_moduli_stability_curve.py) | Moduli-quadrature comparisons |
| [other_scripts/genus1_pair_cell_refinement.py](other_scripts/genus1_pair_cell_refinement.py) | Refinement of selected integration regions |
| [other_scripts/genus1_primary_first_production.py](other_scripts/genus1_primary_first_production.py) | Production inputs and preparation of the energy scan |
| [other_scripts/genus1_primary_moduli_refinement.py](other_scripts/genus1_primary_moduli_refinement.py) | Refinement of selected integration regions |
| [other_scripts/genus1_production_support.py](other_scripts/genus1_production_support.py) | Production settings, checked records, and source paths |
| [other_scripts/genus1_rebuild_policy.py](other_scripts/genus1_rebuild_policy.py) | Amplitude reports and numerical assessment |
| [other_scripts/genus1_source_paths.py](other_scripts/genus1_source_paths.py) | Production settings, checked records, and source paths |
| [other_scripts/genus1_upper_pair_refinement.py](other_scripts/genus1_upper_pair_refinement.py) | Refinement of selected integration regions |
| [other_scripts/plot_genus1_production.py](other_scripts/plot_genus1_production.py) | Amplitude figure and numerical diagnostics |
| [other_scripts/thermal/draft_radius_campaign.py](other_scripts/thermal/draft_radius_campaign.py) | Thermal production, assembly, and momentum comparisons |
| [other_scripts/thermal/draft_radius_collect.py](other_scripts/thermal/draft_radius_collect.py) | Thermal production, assembly, and momentum comparisons |
| [other_scripts/thermal/draft_radius_validation.py](other_scripts/thermal/draft_radius_validation.py) | Thermal production, assembly, and momentum comparisons |
| [other_scripts/thermal/reference_campaign.py](other_scripts/thermal/reference_campaign.py) | Thermal production, assembly, and momentum comparisons |
